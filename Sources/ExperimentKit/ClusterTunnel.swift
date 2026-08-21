import Foundation
import Observation
#if canImport(AppKit)
import AppKit
#endif
#if canImport(Darwin)
import Darwin
#endif

// MARK: - Tunnel state

/// Connection-lifecycle state for one site's transport, published for the
/// toolbar dot (green = up, grey = idle/local-only, amber = needsAuth,
/// red = degraded).
public enum TunnelState: Equatable, Sendable {
    case idle
    /// No live SSH ControlMaster for the site host — the user must run the
    /// prepared `ssh` command in Terminal (Duo/interactive auth happens
    /// there; the app never touches credentials).
    case needsAuth
    case opening
    case up(localPort: Int)
    case degraded(String)
    case closed

    public var displayDescription: String {
        switch self {
        case .idle: "idle — no tunnel open"
        case .needsAuth: "authenticate in Terminal to open the tunnel"
        case .opening: "opening tunnel…"
        case .up(let localPort): "tunnel up on 127.0.0.1:\(localPort)"
        case .degraded(let reason): "degraded: \(reason)"
        case .closed: "tunnel closed"
        }
    }
}

// MARK: - Process seam (injectable for tests)

public struct TunnelProcessResult: Sendable {
    public var exitCode: Int32
    public var standardOutput: String
    public var standardError: String

    public init(exitCode: Int32, standardOutput: String, standardError: String) {
        self.exitCode = exitCode
        self.standardOutput = standardOutput
        self.standardError = standardError
    }
}

/// A long-lived spawned process. ControlMaster forwards use short
/// `ssh -O forward` / `ssh -O cancel` commands instead; this seam remains for
/// other SSH-backed workflows.
public protocol TunnelProcessHandle: AnyObject, Sendable {
    var isRunning: Bool { get async }
    func terminate() async
    /// Suspends until the process exits; returns its exit code. Safe to call
    /// after exit (returns the recorded code immediately).
    func waitUntilExit() async -> Int32
}

/// Seam for everything `ClusterTunnel` does with the operating system, so the
/// state machine is testable with a stub (no real ssh, no real sockets).
public protocol TunnelProcessRunner: Sendable {
    /// Run a short command to completion (`ssh -O check`, remote `cat`).
    func run(_ executablePath: String, arguments: [String]) async -> TunnelProcessResult
    /// Run a short command feeding `input` to its stdin. Secrets travel HERE,
    /// never in argv — argv is visible to `ps` on both ends of the ssh.
    func run(_ executablePath: String, arguments: [String], input: Data) async
        -> TunnelProcessResult
    /// Spawn a long-lived process for SSH-backed workflows that need one.
    func launch(_ executablePath: String, arguments: [String]) async throws -> any TunnelProcessHandle
    /// Whether 127.0.0.1:port is free to bind locally.
    func isLocalPortFree(_ port: Int) async -> Bool
}

/// Live backend: Foundation `Process`. The `Process` object never crosses an
/// isolation boundary — it is created and owned inside `SystemProcessHandle`
/// (an actor) or confined to one continuation for short commands.
public struct SystemTunnelProcessRunner: TunnelProcessRunner {

    public init() {}

    public func run(_ executablePath: String, arguments: [String]) async -> TunnelProcessResult {
        await run(executablePath, arguments: arguments, input: nil)
    }

    public func run(
        _ executablePath: String, arguments: [String], input: Data
    ) async -> TunnelProcessResult {
        await run(executablePath, arguments: arguments, input: Optional(input))
    }

    private func run(
        _ executablePath: String, arguments: [String], input: Data?
    ) async -> TunnelProcessResult {
        await withCheckedContinuation { continuation in
            let process = Process()
            process.executableURL = URL(filePath: executablePath)
            process.arguments = arguments
            let inPipe: Pipe?
            if input != nil {
                let pipe = Pipe()
                process.standardInput = pipe
                inPipe = pipe
            } else {
                process.standardInput = FileHandle.nullDevice
                inPipe = nil
            }
            let outPipe = Pipe()
            let errPipe = Pipe()
            process.standardOutput = outPipe
            process.standardError = errPipe
            let outHandle = outPipe.fileHandleForReading
            let errHandle = errPipe.fileHandleForReading
            // The commands routed here (`ssh -O check`, `cat serverd.host`)
            // emit at most a few lines, so reading after termination cannot
            // fill the pipe buffer and deadlock.
            process.terminationHandler = { finished in
                let out = (try? outHandle.readToEnd()) ?? Data()
                let err = (try? errHandle.readToEnd()) ?? Data()
                continuation.resume(
                    returning: TunnelProcessResult(
                        exitCode: finished.terminationStatus,
                        standardOutput: String(decoding: out, as: UTF8.self),
                        standardError: String(decoding: err, as: UTF8.self)))
            }
            do {
                try process.run()
                if let input, let inPipe {
                    // Small payloads only (a token, a host line): they fit the
                    // pipe buffer, so write-then-close cannot block.
                    try? inPipe.fileHandleForWriting.write(contentsOf: input)
                    try? inPipe.fileHandleForWriting.close()
                }
            } catch {
                process.terminationHandler = nil
                continuation.resume(
                    returning: TunnelProcessResult(
                        exitCode: 127, standardOutput: "",
                        standardError: error.localizedDescription))
            }
        }
    }

    public func launch(
        _ executablePath: String, arguments: [String]
    ) async throws -> any TunnelProcessHandle {
        let handle = SystemProcessHandle()
        try await handle.start(executablePath: executablePath, arguments: arguments)
        return handle
    }

    public func isLocalPortFree(_ port: Int) async -> Bool {
        Self.canBindLoopback(port: port)
    }

    /// True when 127.0.0.1:port accepts a bind (SO_REUSEADDR set, so a
    /// TIME_WAIT remnant doesn't count as busy — only live listeners do).
    static func canBindLoopback(port: Int) -> Bool {
        guard port > 0, port <= 65_535 else { return false }
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        var enable: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &enable, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(UInt16(port).bigEndian)
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
                bind(fd, sockaddrPointer, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return bound == 0
    }
}

/// Actor-owned `Process`: launch, terminate, and await exit without the
/// non-Sendable `Process` ever escaping the actor.
public actor SystemProcessHandle: TunnelProcessHandle {

    private var process: Process?
    private var exitCode: Int32?
    private var waiters: [CheckedContinuation<Int32, Never>] = []

    public init() {}

    fileprivate func start(executablePath: String, arguments: [String]) throws {
        let process = Process()
        process.executableURL = URL(filePath: executablePath)
        process.arguments = arguments
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        process.terminationHandler = { [weak self] finished in
            let code = finished.terminationStatus
            Task { await self?.noteExit(code) }
        }
        try process.run()
        self.process = process
    }

    private func noteExit(_ code: Int32) {
        exitCode = code
        process = nil
        let pending = waiters
        waiters.removeAll()
        for waiter in pending { waiter.resume(returning: code) }
    }

    public var isRunning: Bool {
        exitCode == nil && process?.isRunning == true
    }

    public func terminate() {
        process?.terminate()
    }

    public func waitUntilExit() async -> Int32 {
        if let exitCode { return exitCode }
        return await withCheckedContinuation { waiters.append($0) }
    }
}

// MARK: - Tunnel manager

/// App-side SSH tunnel manager (plan WS1), shared by both daemon topologies.
///
/// The app never handles credentials: authentication is delegated to a
/// `ControlMaster` session the user opens in Terminal (`authenticationCommand`
/// / `openAuthTerminal` — Duo happens there, and `ControlPersist=8h` keeps the
/// master warm for a work session). With a live master this manager installs
/// an explicitly owned local forward and monitors the master + HTTP endpoint.
/// A multiplex client is not itself the tunnel: OpenSSH may hand its forward
/// to the master and exit successfully while the forward remains alive.
///
/// Deliberately knows nothing about `ClusterClient`: the app injects a
/// `healthProbe` closure (wrapping a capabilities ping) instead.
@Observable @MainActor
public final class ClusterTunnel {

    public private(set) var state: TunnelState = .idle
    public private(set) var site: ClusterSiteProfile?

    /// Injected by the app: returns true when the server answers through the
    /// tunnel (e.g. a capabilities ping). Optional — without it, health is
    /// "the ssh process is alive".
    @ObservationIgnored public var healthProbe: (@Sendable () async -> Bool)?
    /// Retained as source compatibility for callers/tests that tuned the old
    /// spawned-client implementation. Control commands complete synchronously,
    /// so neither delay participates in the new lifecycle.
    @ObservationIgnored public var reopenDelays: [Duration] = []
    @ObservationIgnored public var startupGrace: Duration = .zero
    @ObservationIgnored public var healthProbeInterval: Duration = .seconds(15)

    @ObservationIgnored private let runner: any TunnelProcessRunner
    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private var activeForward: ForwardRecord?
    @ObservationIgnored private var pendingCancellation: Task<Void, Never>?
    @ObservationIgnored private var healthTask: Task<Void, Never>?

    public init(
        runner: any TunnelProcessRunner = SystemTunnelProcessRunner(),
        defaults: UserDefaults = .standard
    ) {
        self.runner = runner
        self.defaults = defaults
    }

    // MARK: Constants

    nonisolated static let sshExecutablePath = "/usr/bin/ssh"
    /// Shared multiplexing socket. `%C` hashes local host + remote host +
    /// port + user, keeping the path short (Unix sockets cap ~104 chars) and
    /// per-endpoint; ssh expands the `~` itself. The auth command and every
    /// follow-on invocation must agree on this path.
    nonisolated static let controlPathValue = "~/.ssh/steerlab-cm-%C"
    nonisolated static var controlOptions: [String] {
        ["-o", "ControlPath=\(controlPathValue)"]
    }
    nonisolated static let persistedForwardDefaultsKey = "SteerLabClusterTunnelForward"

    /// Internal (not private) so the headless `ClusterTunnelController` can
    /// compose the SAME forward identity and the SAME `ssh -O forward/cancel`
    /// commands — one tunnel implementation, two owners (plan §7.7).
    struct ForwardRecord: Codable, Equatable, Sendable {
        var host: String
        var proxyJump: String?
        var localPort: Int
        var targetHost: String
        var remotePort: Int

        var specification: String {
            "127.0.0.1:\(localPort):\(targetHost):\(remotePort)"
        }
    }

    // MARK: Authentication (delegated to Terminal)

    /// The exact one-liner the user runs in Terminal to open the site's
    /// ControlMaster (Duo/interactive auth happens there; the master then
    /// persists 8h). Nil for direct transport or a blank host.
    public nonisolated static func authenticationCommand(for site: ClusterSiteProfile) -> String? {
        guard case .ssh(let host, let proxyJump, _, _) = site.transport else { return nil }
        let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedHost.isEmpty else { return nil }
        var parts = [
            "ssh",
            "-o", "ControlMaster=auto",
            "-o", "ControlPersist=8h",
            "-o", "ControlPath=\(controlPathValue)",
        ]
        if let proxyJump, !proxyJump.isEmpty {
            parts += ["-J", proxyJump]
        }
        parts.append(trimmedHost)
        return parts.joined(separator: " ")
    }

    public func authenticationCommand() -> String? {
        site.flatMap { Self.authenticationCommand(for: $0) }
    }

    /// Writes a temporary `.command` file (0755) holding the authentication
    /// one-liner and opens it (Terminal by default), so Duo happens in the
    /// user's own terminal — the app never sees credentials.
    @discardableResult
    public func openAuthTerminal() -> Bool {
        guard let site, let command = Self.authenticationCommand(for: site) else { return false }
        do {
            let url = try Self.writeAuthCommandFile(command: command, siteName: site.name)
            #if canImport(AppKit)
            NSWorkspace.shared.open(url)
            return true
            #else
            _ = url
            return false
            #endif
        } catch {
            return false
        }
    }

    nonisolated static func writeAuthCommandFile(
        command: String,
        siteName: String,
        in directory: URL = FileManager.default.temporaryDirectory
    ) throws -> URL {
        let safeName = String(siteName.map { $0.isLetter || $0.isNumber ? $0 : "-" })
        let url = directory.appending(component: "steerlab-authenticate-\(safeName).command")
        let script = """
            #!/bin/zsh
            # SteerLab — authenticate to \(siteName)
            # Complete the login below (password / Duo). The SSH master connection
            # stays warm for 8 hours; you can close this window after signing in.
            exec \(command)

            """
        try script.write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    // MARK: Lifecycle

    /// Point the manager at a site (or nil). A different site tears down any
    /// live tunnel; the same site is a no-op so unrelated store updates never
    /// drop a healthy tunnel.
    public func configure(site newSite: ClusterSiteProfile?) {
        guard newSite != site else { return }
        invalidate(to: .idle)
        site = newSite
    }

    /// Open (or repair) the tunnel for the configured site. Direct transport
    /// needs no tunnel — the state stays `.idle` and `effectiveBaseURL` is
    /// already the site URL. For SSH transport: verify the ControlMaster
    /// (else `.needsAuth`), resolve the forward target (daemon-in-a-job reads
    /// `<metadataRoot>/serverd.host` on the login host), pick a local port,
    /// install a ControlMaster-owned forward, and monitor it. Repeated calls
    /// while that forward is up are no-ops.
    public func open() async {
        guard let site else { return }
        guard case .ssh(let host, let proxyJump, let remotePort, _) = site.transport else {
            state = .idle
            return
        }
        let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedHost.isEmpty else {
            state = .degraded("site has no SSH host configured")
            return
        }
        if case .up = state, activeForward != nil { return }
        if case .opening = state { return }
        state = .opening
        if let pendingCancellation {
            await pendingCancellation.value
            self.pendingCancellation = nil
        }
        guard await masterIsAlive(host: trimmedHost) else {
            activeForward = nil
            clearPersistedForward()
            state = .needsAuth
            return
        }
        await recoverPersistedForwardIfNeeded()
        await cancelActiveForward()
        var targetHost = "localhost"
        if site.topology == .daemonInJob {
            let read = await runner.run(
                Self.sshExecutablePath,
                arguments: Self.remoteReadArguments(
                    host: trimmedHost, proxyJump: proxyJump, path: site.daemonHostFilePath))
            let daemonHost = read.standardOutput.trimmingCharacters(in: .whitespacesAndNewlines)
            guard read.exitCode == 0, !daemonHost.isEmpty else {
                state = .degraded(
                    "could not read \(site.daemonHostFilePath) on \(trimmedHost) — "
                        + "is the daemon job running?")
                return
            }
            targetHost = daemonHost
        }
        let preferred = site.preferredLocalPort ?? 8700
        guard let localPort = await Self.firstFreeLocalPort(from: preferred, runner: runner) else {
            state = .degraded("no free local port near \(preferred)")
            return
        }
        let forward = ForwardRecord(
            host: trimmedHost, proxyJump: proxyJump, localPort: localPort,
            targetHost: targetHost, remotePort: remotePort)
        let result = await runner.run(
            Self.sshExecutablePath, arguments: Self.forwardArguments(forward))
        guard result.exitCode == 0 else {
            let detail = result.standardError
                .trimmingCharacters(in: .whitespacesAndNewlines)
            state = .degraded(
                detail.isEmpty
                    ? "could not install SSH forward (code \(result.exitCode))"
                    : "could not install SSH forward: \(detail)")
            return
        }
        activeForward = forward
        persistForward(forward)
        state = .up(localPort: localPort)
        startHealthLoop(localPort: localPort)
    }

    /// Remove the ControlMaster forward cleanly and stop monitoring.
    public func close() async {
        healthTask?.cancel()
        healthTask = nil
        state = .closed
        await cancelActiveForward()
    }

    /// Read a small trusted file through the authenticated ControlMaster.
    /// Setup uses this to import the generated API token into Keychain; callers
    /// must never log the returned contents.
    public func readRemoteTextFile(_ path: String) async -> String? {
        guard let site,
            case .ssh(let host, let proxyJump, _, _) = site.transport
        else { return nil }
        let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedHost.isEmpty, await masterIsAlive(host: trimmedHost) else { return nil }
        let result = await runner.run(
            Self.sshExecutablePath,
            arguments: Self.remoteReadArguments(
                host: trimmedHost, proxyJump: proxyJump, path: path))
        guard result.exitCode == 0 else { return nil }
        let value = result.standardOutput.trimmingCharacters(in: .whitespacesAndNewlines)
        return value.isEmpty ? nil : value
    }

    /// Write a small SECRET file through the authenticated ControlMaster.
    /// The content travels on stdin (never argv — argv is visible to `ps` on
    /// both hosts), is born private (`umask 077`), and lands atomically
    /// (tmp + `mv`). Returns a human-readable error, or nil on success.
    /// Callers must never log the contents.
    public func writeRemoteSecretFile(_ path: String, contents: String) async -> String? {
        guard let site, case .ssh(let host, let proxyJump, _, _) = site.transport else {
            return "no SSH site configured"
        }
        let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedHost.isEmpty else { return "site has no SSH host configured" }
        guard await masterIsAlive(host: trimmedHost) else {
            return "no SSH ControlMaster for \(trimmedHost) — choose Authenticate first"
        }
        let result = await runner.run(
            Self.sshExecutablePath,
            arguments: Self.remoteWriteArguments(
                host: trimmedHost, proxyJump: proxyJump, path: path),
            input: Data(contents.utf8))
        guard result.exitCode == 0 else {
            let detail = result.standardError.trimmingCharacters(in: .whitespacesAndNewlines)
            return detail.isEmpty
                ? "remote write exited \(result.exitCode)" : detail
        }
        return nil
    }

    /// Whether a NON-EMPTY file exists at `path` on the remote host — the
    /// presence check UIs use instead of ever reading a secret back. Nil when
    /// the ControlMaster is down and the question cannot be asked.
    public func remoteFileExists(_ path: String) async -> Bool? {
        guard let site, case .ssh(let host, let proxyJump, _, _) = site.transport else {
            return nil
        }
        let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedHost.isEmpty, await masterIsAlive(host: trimmedHost) else { return nil }
        var arguments = Self.controlOptions + ["-o", "BatchMode=yes"]
        if let proxyJump, !proxyJump.isEmpty {
            arguments += ["-J", proxyJump]
        }
        // Same quoting discipline as the write/read paths: the path is
        // profile-controlled data headed for a remote shell.
        arguments += [trimmedHost, "test", "-s", ClusterProvisioner.shellQuoted(path)]
        let result = await runner.run(Self.sshExecutablePath, arguments: arguments)
        return result.exitCode == 0
    }

    /// Materialize the researcher's Hugging Face token as `<hfCache>/token` —
    /// the hub's NATIVE token location (`$HF_HOME/token`), so downloads need
    /// no SteerLab token plumbing and the secret never enters job bundles,
    /// transcripts, or logs. Returns a human-readable error, or nil once the
    /// file is written and verified present.
    public func installHFToken(_ token: String) async -> String? {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "token is empty — paste an hf_… read token" }
        guard let site else { return "no cluster site configured" }
        let cacheRoot = (site.constraints.storageRoots["hfCache"] ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cacheRoot.isEmpty else {
            return "site has no HF cache storage root — set it in Edit Site… first "
                + "(the token lands at <hfCache>/token, where the hub looks for it)"
        }
        let path = cacheRoot + "/token"
        if let error = await writeRemoteSecretFile(path, contents: trimmed) {
            return error
        }
        guard await remoteFileExists(path) == true else {
            return "wrote \(path) but could not verify it — check quota and "
                + "permissions on \(cacheRoot)"
        }
        return nil
    }

    /// Remove a small file on the remote host (`rm -f` — absent is
    /// success). Returns a human-readable error, or nil.
    public func deleteRemoteFile(_ path: String) async -> String? {
        guard let site, case .ssh(let host, let proxyJump, _, _) = site.transport else {
            return "no SSH site configured"
        }
        let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedHost.isEmpty else { return "site has no SSH host configured" }
        guard await masterIsAlive(host: trimmedHost) else {
            return "no SSH ControlMaster for \(trimmedHost) — choose Authenticate first"
        }
        let result = await runner.run(
            Self.sshExecutablePath,
            arguments: Self.remoteDeleteArguments(
                host: trimmedHost, proxyJump: proxyJump, path: path))
        guard result.exitCode == 0 else {
            let detail = result.standardError.trimmingCharacters(in: .whitespacesAndNewlines)
            return detail.isEmpty
                ? "remote delete exited \(result.exitCode)" : detail
        }
        return nil
    }

    /// Sync the EXTERNAL judge key to the cluster (key-custody design,
    /// seamless-pipeline extension 2026-07-19): push the stored key to
    /// `~/.steerlab/judge-key` (mode 600, stdin transport) — or REMOVE the
    /// remote file when no key is stored. Called at every connect, so
    /// rotation is "save + reconnect" and DELETION PROPAGATES: a key
    /// cleared on this Mac cannot stay live on the cluster. Returns a
    /// human-readable error, or nil on success.
    public func syncJudgeKey(
        _ stored: JudgeKeyStore.StoredKey?
    ) async -> String? {
        guard let stored else {
            return await deleteRemoteFile(JudgeKeyStore.remotePath)
        }
        if let error = await writeRemoteSecretFile(
            JudgeKeyStore.remotePath,
            contents: JudgeKeyStore.remoteFileContents(stored))
        {
            return error
        }
        guard await remoteFileExists(JudgeKeyStore.remotePath) == true else {
            return "wrote \(JudgeKeyStore.remotePath) but could not verify "
                + "it — check home-directory quota and permissions"
        }
        return nil
    }

    /// Where HTTP should go right now: the direct URL for direct transport,
    /// `http://127.0.0.1:<localPort>` while an SSH tunnel is up, else nil.
    public var effectiveBaseURL: URL? {
        guard let site else { return nil }
        switch site.transport {
        case .direct(let baseURL):
            return baseURL
        case .ssh:
            guard case .up(let localPort) = state else { return nil }
            return ClusterSiteProfile.localhostBaseURL(port: localPort)
        }
    }

    // MARK: Internals

    private func invalidate(to newState: TunnelState) {
        healthTask?.cancel()
        healthTask = nil
        if let forward = activeForward {
            activeForward = nil
            clearPersistedForward()
            let runner = self.runner
            pendingCancellation = Task {
                _ = await runner.run(
                    Self.sshExecutablePath, arguments: Self.cancelArguments(forward))
            }
        }
        state = newState
    }

    private func masterIsAlive(host: String) async -> Bool {
        let result = await runner.run(
            Self.sshExecutablePath, arguments: Self.controlOptions + ["-O", "check", host])
        return result.exitCode == 0
    }

    /// Lightweight periodic probe while the tunnel is up: two consecutive
    /// failures mark the state degraded (the ssh process may be fine while
    /// the server behind it is not); one success restores `.up`.
    private func startHealthLoop(localPort: Int) {
        healthTask?.cancel()
        guard healthProbe != nil else { return }
        let interval = healthProbeInterval
        healthTask = Task { [weak self] in
            var consecutiveFailures = 0
            while !Task.isCancelled {
                try? await Task.sleep(for: interval)
                guard !Task.isCancelled, let self, let probe = self.healthProbe else { return }
                guard self.activeForward?.localPort == localPort else { return }
                guard let site = self.site,
                    case .ssh(let host, _, _, _) = site.transport
                else { return }
                let trimmedHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
                guard await self.masterIsAlive(host: trimmedHost) else {
                    self.activeForward = nil
                    self.clearPersistedForward()
                    self.state = .needsAuth
                    return
                }
                let healthy = await probe()
                if healthy {
                    consecutiveFailures = 0
                    if case .degraded = self.state, self.activeForward != nil {
                        self.state = .up(localPort: localPort)
                    }
                } else {
                    consecutiveFailures += 1
                    if consecutiveFailures >= 2, case .up = self.state {
                        self.state = .degraded(
                            "tunnel is up but the server is not answering on "
                                + "127.0.0.1:\(localPort)")
                    }
                }
            }
        }
    }

    // MARK: ssh invocations (pure, tested)

    nonisolated static func forwardArguments(_ forward: ForwardRecord) -> [String] {
        controlCommandArguments("forward", forward: forward)
    }

    nonisolated static func cancelArguments(_ forward: ForwardRecord) -> [String] {
        controlCommandArguments("cancel", forward: forward)
    }

    nonisolated static func controlCommandArguments(
        _ command: String, forward: ForwardRecord
    ) -> [String] {
        var arguments = controlOptions + ["-o", "BatchMode=yes"]
        if let proxyJump = forward.proxyJump, !proxyJump.isEmpty {
            arguments += ["-J", proxyJump]
        }
        arguments += ["-O", command, "-L", forward.specification, forward.host]
        return arguments
    }

    /// SSH joins everything after the host into ONE remote shell string, so
    /// every profile-controlled path must be shell-quoted: site JSON is an
    /// import/share format, and an unquoted `metadataRoot`/`hfCache` would
    /// hand an imported profile remote command execution (plus break on
    /// ordinary spaces). `shellQuoted` leaves safe words — including `~/…`,
    /// which must still expand — untouched.
    nonisolated static func remoteReadArguments(
        host: String, proxyJump: String?, path: String
    ) -> [String] {
        var arguments = controlOptions + ["-o", "BatchMode=yes"]
        if let proxyJump, !proxyJump.isEmpty {
            arguments += ["-J", proxyJump]
        }
        arguments += [host, "cat", ClusterProvisioner.shellQuoted(path)]
        return arguments
    }

    /// The stdin→file remote command: private from birth (`umask 077`),
    /// atomic on arrival (tmp + `mv -f`), parent dir created so a fresh cache
    /// root is not a failure.
    nonisolated static func remoteWriteArguments(
        host: String, proxyJump: String?, path: String
    ) -> [String] {
        var arguments = controlOptions + ["-o", "BatchMode=yes"]
        if let proxyJump, !proxyJump.isEmpty {
            arguments += ["-J", proxyJump]
        }
        let quoted = ClusterProvisioner.shellQuoted(path)
        let tmp = ClusterProvisioner.shellQuoted(path + ".tmp")
        let directory = ClusterProvisioner.shellQuoted(
            (path as NSString).deletingLastPathComponent)
        arguments += [
            host,
            "umask 077 && mkdir -p \(directory) && cat > \(tmp) && mv -f \(tmp) \(quoted)",
        ]
        return arguments
    }

    /// The remote-delete command: `rm -f` (absent is success), path quoted
    /// with the same discipline as the read/write paths — profile- or
    /// caller-controlled paths are data headed for a remote shell.
    nonisolated static func remoteDeleteArguments(
        host: String, proxyJump: String?, path: String
    ) -> [String] {
        var arguments = controlOptions + ["-o", "BatchMode=yes"]
        if let proxyJump, !proxyJump.isEmpty {
            arguments += ["-J", proxyJump]
        }
        arguments += [host, "rm -f \(ClusterProvisioner.shellQuoted(path))"]
        return arguments
    }

    private func cancelActiveForward() async {
        guard let forward = activeForward else { return }
        activeForward = nil
        clearPersistedForward()
        _ = await runner.run(
            Self.sshExecutablePath, arguments: Self.cancelArguments(forward))
    }

    /// A ControlMaster outlives the app, so a crash can leave one forward
    /// behind. Persisting its exact specification lets the next launch cancel
    /// it before choosing a port; no broad process kill or port-range sweep is
    /// needed.
    private func recoverPersistedForwardIfNeeded() async {
        guard activeForward == nil,
            let data = defaults.data(forKey: Self.persistedForwardDefaultsKey),
            let forward = try? JSONDecoder().decode(ForwardRecord.self, from: data)
        else { return }
        _ = await runner.run(
            Self.sshExecutablePath, arguments: Self.cancelArguments(forward))
        clearPersistedForward()
    }

    private func persistForward(_ forward: ForwardRecord) {
        guard let data = try? JSONEncoder().encode(forward) else { return }
        defaults.set(data, forKey: Self.persistedForwardDefaultsKey)
    }

    private func clearPersistedForward() {
        defaults.removeObject(forKey: Self.persistedForwardDefaultsKey)
    }

    /// First free local port at or above `preferred` (bounded walk).
    nonisolated static func firstFreeLocalPort(
        from preferred: Int, runner: any TunnelProcessRunner, probeLimit: Int = 25
    ) async -> Int? {
        var candidate = max(preferred, 1)
        for _ in 0..<probeLimit {
            guard candidate <= 65_535 else { return nil }
            if await runner.isLocalPortFree(candidate) { return candidate }
            candidate += 1
        }
        return nil
    }
}
