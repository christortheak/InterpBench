import Foundation
import Observation

// MARK: - Local Python runtime discovery

/// Where the CODE checkout's Python pieces live: the `Server/` engine, its
/// venv, and the shell scripts the app can run for the researcher.
///
/// **This is bucket B (WP2), and the distinction is load-bearing.** These are
/// not shipped read-only bytes: `scripts/start-local-server.sh` creates
/// `<root>/Server/.venv.nosync` and `pip install -e`s the engine into it, so
/// it writes into its OWN tree — and the first write into a signed app bundle
/// breaks its signature. So every path here resolves through
/// `CodeResources.executableCheckout()` (the compiled checkout in developer
/// mode; a checkout found BY CONTENT beside the app in the home layout
/// otherwise) and NEVER through the `serverPayload` family, which may be the
/// bundle. It is equally never the DATA workspace — the venv and scripts ship
/// with the code, while the workspace is what the server *serves*.
///
/// When there is no such checkout every accessor is nil and callers surface
/// `unavailableHint`, which names the layout remedy rather than a path that
/// does not exist.
public enum LocalPythonRuntime {

    /// The SteerLab code checkout the local engine runs from; nil when this
    /// build has none (see `unavailableHint`).
    public static var repoRoot: URL? {
        CodeResources.executableCheckoutRoot
    }

    /// The Python interpreter inside `Server/.venv.nosync` — created by
    /// `scripts/start-local-server.sh` on its first run, INSIDE the checkout
    /// (never the bundle's `ServerPayload/`, which must stay read-only).
    public static var venvPython: URL? {
        repoRoot?.appending(components: "Server", ".venv.nosync", "bin", "python")
    }

    public static var venvExists: Bool {
        guard let venvPython else { return false }
        return FileManager.default.isExecutableFile(atPath: venvPython.path)
    }

    /// The one-click local-server script (venv setup + loopback serve with
    /// an explicit `--root`). Executed from the checkout because it writes
    /// the venv into its own tree.
    public static var startServerScript: URL? {
        repoRoot?.appending(components: "scripts", "start-local-server.sh")
    }

    /// The plain-sentence remedy shown wherever the venv is required but
    /// absent.
    public static let setupHint =
        "the local Python environment (Server/.venv.nosync) is not set up yet — "
        + "use Start Local Python Server (connection dot menu, top right), "
        + "which creates it on first run. The first install pulls "
        + "torch/transformers and can take many minutes."

    /// Honest, TYPED refusal for builds with no checkout to run from: it
    /// names the home layout and what to put in it, never a missing file.
    public static var unavailableHint: String {
        switch CodeResources.executableCheckout() {
        case .success(let checkout):
            // Reached only if the checkout appears between two calls; still
            // truthful rather than stale.
            return "the local Python engine is available at \(checkout.root.path)"
        case .failure(let absence):
            return "the local Python engine needs the code checkout beside the "
                + "app — " + absence.message
        }
    }
}

// MARK: - Local-server pidfile (adoption across app relaunches)

/// The pidfile `scripts/start-local-server.sh` writes at the WORKSPACE root
/// (one line, `"<pid> <port>"`; the script `exec`s the Python server, so the
/// pid is the server itself). A relaunched app reads it to ADOPT a server
/// started earlier instead of showing "not running" next to a live server.
/// Parsing and the adoption rule are pure and unit-tested; stale files are
/// cleaned silently by whoever finds them (the script and the controller).
public enum LocalServerPidfile {

    public static let filename = ".steerlab-local-server.pid"

    public static func url(workspaceRoot: URL) -> URL {
        workspaceRoot.appending(component: filename)
    }

    /// `"<pid> <port>"` → (pid, port); nil for anything else (garbled files
    /// count as stale).
    public static func parse(_ contents: String) -> (pid: Int32, port: Int)? {
        let fields = contents.split(whereSeparator: \.isWhitespace)
        guard fields.count == 2,
            let pid = Int32(fields[0]), pid > 0,
            let port = Int(fields[1]), (1...65535).contains(port)
        else { return nil }
        return (pid, port)
    }

    /// What the adoption probe concluded — four-way, because "something is
    /// alive and listening" is NOT the same as "that is our server": pid
    /// reuse plus an unrelated listener used to enable a Stop button that
    /// would SIGTERM an innocent process, and with SEVERAL local SteerLab
    /// servers a stale pid could name server A while the port serves
    /// server B.
    public enum Adoption: Equatable, Sendable {
        /// The pid is a steerlab_server process whose argv is BOUND to the
        /// pidfile's port (and workspace root, when the argv names one) AND
        /// the port answers as a SteerLab server — adopt it, Stop is safe.
        case adopt(pid: Int32, port: Int)
        /// Something is alive/listening on the pidfile's port but identity
        /// could not be established — never enable Stop for it.
        case portInUse(port: Int)
        /// The pid IS a SteerLab server process, but its command line is
        /// not BOUND to this pidfile: it names a DIFFERENT `--port` or
        /// `--root`, or carries no `--port` at all (a pidfile written by a
        /// start-script version from before argv binding). Process identity
        /// alone must not enable Stop — with multiple local servers, a
        /// reused/stale pid can identify server A while the port serves
        /// server B, and Stop would kill the wrong one. Legacy pidfiles
        /// deliberately degrade to THIS safe state: the user stops that
        /// server manually once, and the current script writes a bindable
        /// argv from then on.
        case unbound(port: Int)
        /// Nothing to adopt; the pidfile (if any) is stale and may be
        /// cleaned.
        case stale
    }

    /// PROCESS-FAMILY rule, pure for tests: the pidfile's process must
    /// actually be the Python server. `scripts/start-local-server.sh`
    /// `exec`s `python -m steerlab_server.cli serve …`, so the argv of the
    /// recorded pid carries "steerlab_server"; a reused pid (some unrelated
    /// program) will not. This alone is NOT sufficient for adoption —
    /// `commandLineMatchesSteerLabServer(_:port:workspaceRoot:)` must also
    /// bind the argv to the pidfile's port.
    public static func commandLineIsSteerLabServer(_ command: String?) -> Bool {
        command?.contains("steerlab_server") ?? false
    }

    /// BOUND process-identity rule, pure for tests: the argv must name
    /// steerlab_server AND carry the pidfile's port (`--port <p>` or
    /// `--port=<p>` — the start script always passes it explicitly on the
    /// exec line, which is load-bearing for this check). When the argv also
    /// names a `--root`, it must match the workspace root the pidfile lives
    /// in (bound when present, never required retroactively — terminal
    /// invocations may omit it and fall back to STEERLAB_ROOT). An argv
    /// with NO `--port` (older script version) never binds: safe degrade,
    /// see `Adoption.unbound`. Root comparison is by normalized path
    /// spelling; a differently-spelled path to the same directory (symlink)
    /// demotes to the safe unbound state rather than adopting.
    public static func commandLineMatchesSteerLabServer(
        _ command: String?, port: Int, workspaceRoot: String? = nil
    ) -> Bool {
        guard let command, commandLineIsSteerLabServer(command) else {
            return false
        }
        guard argvNamesPort(command, port: port) else { return false }
        if let workspaceRoot, command.contains("--root") {
            return argvBindsRoot(command, workspaceRoot: workspaceRoot)
        }
        return true
    }

    /// Whole-token scan for `--port <p>` / `--port=<p>` — token equality,
    /// so port 8080 never matches an argv naming 80801.
    private static func argvNamesPort(_ command: String, port: Int) -> Bool {
        let value = String(port)
        let tokens = command.split(whereSeparator: \.isWhitespace)
        for (index, token) in tokens.enumerated() {
            if token == "--port", index + 1 < tokens.count,
                tokens[index + 1] == value
            {
                return true
            }
            if token == "--port=\(value)" { return true }
        }
        return false
    }

    /// `--root` value check by raw substring (workspace paths can contain
    /// spaces — e.g. iCloud's "Mobile Documents" — so whole-token parsing of
    /// `ps` output is impossible). The value must start at the flag, equal
    /// the normalized workspace root, and end at a path/argument boundary
    /// (end of line, whitespace, or a lone trailing slash).
    private static func argvBindsRoot(
        _ command: String, workspaceRoot: String
    ) -> Bool {
        var root = workspaceRoot
        while root.count > 1 && root.hasSuffix("/") { root.removeLast() }
        guard !root.isEmpty, let flag = command.range(of: "--root") else {
            return false
        }
        var rest = command[flag.upperBound...]
        if rest.first == "=" {
            rest = rest.dropFirst()
        } else {
            rest = rest.drop(while: \.isWhitespace)
        }
        guard rest.hasPrefix(root) else { return false }
        var after = rest.dropFirst(root.count)
        if after.first == "/" { after = after.dropFirst() }
        return after.isEmpty || (after.first?.isWhitespace ?? false)
    }

    /// ENDPOINT-identity rule, pure for tests: the raw HTTP response of
    /// `GET /api/info` must carry the server's self-identification.
    ///
    /// Accepted shapes (all authored in `Server/steerlab_server/api/`):
    /// - 200 with `"service": "steerlab-server"` — the key/value only
    ///   SteerLab returns.
    /// - 401 with detail "missing or invalid bearer token" — token mode
    ///   (`STEERLAB_AUTH_MODE=token`) gates EVERY route including
    ///   `/api/info`, so an unauthenticated probe of a token-protected
    ///   server gets exactly this body (app.py's own string, not a
    ///   framework default). Accepting it is identity WITHOUT
    ///   authentication: Stop still requires the bound-argv process check,
    ///   so nothing is weakened — and this code path never sends a stored
    ///   token.
    /// - 503 with a detail naming STEERLAB_AUTH_TOKEN — token mode with no
    ///   token configured; the string is unmistakably SteerLab's.
    ///
    /// A generic `{"ok": true}`, an error page, another program's JSON, or
    /// any other 401 body all fail.
    public static func responseIdentifiesSteerLab(_ response: String?) -> Bool {
        guard let response else { return false }
        if response.contains("\"service\"")
            && response.contains("steerlab-server")
        {
            return true
        }
        guard let status = httpStatusCode(response),
            response.contains("\"detail\"")
        else { return false }
        if status == 401,
            response.contains("missing or invalid bearer token")
        {
            return true
        }
        if status == 503, response.contains("STEERLAB_AUTH_TOKEN") {
            return true
        }
        return false
    }

    /// "HTTP/1.1 401 Unauthorized" → 401; nil for anything that is not an
    /// HTTP status line.
    private static func httpStatusCode(_ response: String) -> Int? {
        let line = response.prefix { $0 != "\r" && $0 != "\n" }
        let parts = line.split(separator: " ")
        guard parts.count >= 2, parts[0].hasPrefix("HTTP/") else { return nil }
        return Int(parts[1])
    }

    /// The adoption decision, pure for tests. Liveness gates first (a dead
    /// pid or silent port is stale, exactly as before); then process
    /// identity must be BOUND — the pid's argv names steerlab_server AND
    /// carries this pidfile's port (and matching `--root` when the argv has
    /// one) — and the endpoint must identify as SteerLab before the server
    /// is adopted (Stop enabled). A live listener whose pid is not even a
    /// SteerLab process (or whose endpoint fails identity) is `.portInUse`;
    /// a live SteerLab pid whose argv is not bound to this pidfile is
    /// `.unbound` — both surfaced plainly, never stoppable.
    public static func adoptionDecision(
        contents: String,
        workspaceRoot: String? = nil,
        processIsAlive: (Int32) -> Bool,
        commandLine: (Int32) -> String?,
        portIsListening: (Int) -> Bool,
        endpointIsSteerLab: (Int) -> Bool
    ) -> Adoption {
        guard let parsed = parse(contents),
            processIsAlive(parsed.pid),
            portIsListening(parsed.port)
        else { return .stale }
        let command = commandLine(parsed.pid)
        guard commandLineIsSteerLabServer(command) else {
            return .portInUse(port: parsed.port)
        }
        guard
            commandLineMatchesSteerLabServer(
                command, port: parsed.port, workspaceRoot: workspaceRoot)
        else { return .unbound(port: parsed.port) }
        guard endpointIsSteerLab(parsed.port) else {
            return .portInUse(port: parsed.port)
        }
        return .adopt(pid: parsed.pid, port: parsed.port)
    }

    /// `kill(pid, 0)` probe. EPERM still means "exists" (not our child —
    /// e.g. a server started from a terminal under a different context).
    public static func processIsAlive(_ pid: Int32) -> Bool {
        guard pid > 0 else { return false }
        if kill(pid, 0) == 0 { return true }
        return errno == EPERM
    }

    /// Loopback TCP connect probe — a refused connect returns immediately,
    /// so blocking here is fine (loopback only, never a remote host).
    public static func portIsListening(_ port: Int) -> Bool {
        guard (1...65535).contains(port) else { return false }
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return false }
        defer { close(fd) }
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(UInt16(port).bigEndian)
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        let result = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return result == 0
    }

    /// The recorded pid's command line via `ps -p <pid> -o command=` — nil
    /// when the process is gone or `ps` fails. Thin effect edge; the
    /// matching rule is `commandLineIsSteerLabServer`.
    public static func processCommandLine(_ pid: Int32) -> String? {
        guard pid > 0 else { return nil }
        let process = Process()
        process.executableURL = URL(filePath: "/bin/ps")
        process.arguments = ["-p", "\(pid)", "-o", "command="]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        process.standardInput = FileHandle.nullDevice
        do { try process.run() } catch { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        let line = String(decoding: data, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return line.isEmpty ? nil : line
    }

    /// Blocking loopback `GET /api/info` (short send/receive timeouts) —
    /// the raw response text, or nil when nothing answered. `/api/info` is
    /// an unprivileged GET the Python server always serves; the probe sends
    /// no Origin/Sec-Fetch headers, so the server's browser guard ignores
    /// it. Thin effect edge; the matching rule is
    /// `responseIdentifiesSteerLab`.
    public static func fetchInfoResponse(port: Int) -> String? {
        guard (1...65535).contains(port) else { return nil }
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }
        defer { close(fd) }
        var timeout = timeval(tv_sec: 1, tv_usec: 0)
        _ = setsockopt(
            fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
            socklen_t(MemoryLayout<timeval>.size))
        _ = setsockopt(
            fd, SOL_SOCKET, SO_SNDTIMEO, &timeout,
            socklen_t(MemoryLayout<timeval>.size))
        var address = sockaddr_in()
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(UInt16(port).bigEndian)
        address.sin_addr.s_addr = inet_addr("127.0.0.1")
        let connected = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard connected == 0 else { return nil }
        let request = "GET /api/info HTTP/1.1\r\n"
            + "Host: 127.0.0.1:\(port)\r\n"
            + "Accept: application/json\r\n"
            + "Connection: close\r\n\r\n"
        let sent = request.withCString { pointer in
            send(fd, pointer, strlen(pointer), 0)
        }
        guard sent > 0 else { return nil }
        // Connection: close → read to EOF (or the 1 s timeout), capped.
        var response = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while response.count < 262_144 {
            let received = recv(fd, &buffer, buffer.count, 0)
            guard received > 0 else { break }
            response.append(contentsOf: buffer[0..<received])
        }
        guard !response.isEmpty else { return nil }
        return String(decoding: response, as: UTF8.self)
    }

    /// Endpoint-identity probe against the live port.
    public static func endpointIsSteerLab(port: Int) -> Bool {
        responseIdentifiesSteerLab(fetchInfoResponse(port: port))
    }

    /// Read + adoption decision against the live system; a STALE pidfile is
    /// removed silently on the way out (as before). A `.portInUse` or
    /// `.unbound` pidfile is left alone — the recorded pid may be a SteerLab
    /// server that is not answering yet, and the start script cleans
    /// dead-pid files itself.
    public static func adoption(workspaceRoot: URL) -> Adoption {
        let file = url(workspaceRoot: workspaceRoot)
        guard let contents = try? String(contentsOf: file, encoding: .utf8) else {
            return .stale
        }
        let decision = adoptionDecision(
            contents: contents,
            workspaceRoot: workspaceRoot.path,
            processIsAlive: processIsAlive,
            commandLine: processCommandLine,
            portIsListening: portIsListening,
            endpointIsSteerLab: endpointIsSteerLab)
        if decision == .stale {
            try? FileManager.default.removeItem(at: file)
        }
        return decision
    }

    /// Remove the pidfile iff its process is gone (silent stale cleanup —
    /// called after a server this app started exits).
    public static func cleanIfStale(workspaceRoot: URL) {
        let file = url(workspaceRoot: workspaceRoot)
        guard let contents = try? String(contentsOf: file, encoding: .utf8) else {
            return
        }
        if let parsed = parse(contents), processIsAlive(parsed.pid) { return }
        try? FileManager.default.removeItem(at: file)
    }
}

// MARK: - One-click local Python server

/// Lifecycle owner for the LOCAL Python SteerLab server, started through
/// `scripts/start-local-server.sh` so the terminal path and the app button
/// are the same tested code path. Output streams into the shared live-log /
/// Activity pane (`ChatService.startLiveLog`); state is published for the
/// connection-dot menu (running/stopped + a Stop affordance). At init the
/// controller ADOPTS a server left running by an earlier launch via the
/// workspace-root pidfile (`LocalServerPidfile`) — Stop then works by pid.
///
/// The script serves the app's CURRENT data workspace
/// (`VectorCatalog.projectRoot`) via an explicit `serve --root`, binding
/// loopback only — the cwd hazards CLAUDE.md warns about (artifact root =
/// STEERLAB_ROOT-or-cwd) cannot occur on this path.
@MainActor
@Observable
public final class LocalServerController {

    public enum Phase: String, Sendable {
        case idle
        case starting
        case running
        case stopping
    }

    public private(set) var phase: Phase = .idle
    /// One plain sentence for the menu: what the server is doing and, on
    /// failure, what to do about it.
    public private(set) var statusLine: String
    /// 8080 by default; an ADOPTED server keeps whatever port its pidfile
    /// records.
    public private(set) var port = 8080

    /// True when the running server was adopted from the pidfile at launch
    /// (started by an earlier app run or a terminal), not started here. The
    /// connection dot's auto-connect treats adoption more gently than an
    /// explicit Start click.
    public private(set) var wasAdopted = false

    private var process: Process?
    /// PID of an adopted server (no `Process` handle exists for it); Stop
    /// signals this directly.
    private var adoptedPID: Int32?
    private var monitor: Task<Void, Never>?

    public init() {
        // Adoption across relaunch: the pidfile at the workspace root names
        // a live, listening server started earlier — surface it as running
        // (with Stop working) instead of "not running" beside a live server.
        // Stop is enabled ONLY when both identity checks pass (the pid's
        // argv names steerlab_server BOUND to this pidfile's port, and the
        // port answers as a SteerLab server) — a reused pid beside an
        // unrelated listener, or a stale pid naming server A while the port
        // serves server B, must never hand the user a Stop button that
        // SIGTERMs the wrong process. A stale pidfile is cleaned silently
        // inside `adoption`.
        switch LocalServerPidfile.adoption(workspaceRoot: VectorCatalog.projectRoot) {
        case .adopt(let pid, let adoptedPort):
            adoptedPID = pid
            port = adoptedPort
            wasAdopted = true
            phase = .running
            statusLine = "running (started earlier) at "
                + "http://127.0.0.1:\(adoptedPort) — Stop terminates it"
        case .portInUse(let busyPort):
            // Identity failed: phase stays .idle (no Stop; `stop()` has
            // nothing to signal), and the one status line says what is
            // actually known. (A token-protected SteerLab server DOES
            // identify — its 401 shape is recognized — so landing here
            // means the listener really did not answer as SteerLab.)
            port = busyPort
            statusLine = "another program is using port \(busyPort) — it did "
                + "not identify as a SteerLab server, so Stop is unavailable; "
                + "stop it yourself or start on another port"
        case .unbound(let busyPort):
            // The pid is a SteerLab server, but its command line is not
            // bound to this pidfile's port/root — a different local server,
            // or a pidfile from a start-script version that predates argv
            // binding. Never stoppable from here: with several servers,
            // Stop could kill the wrong one.
            port = busyPort
            statusLine = "a SteerLab server is alive but its command line "
                + "does not name port \(busyPort) for this workspace (another "
                + "server's port/root, or a start script from before port "
                + "binding) — Stop is unavailable; stop it once from the "
                + "terminal that started it, then start it from here"
        case .stale:
            statusLine = LocalPythonRuntime.venvExists
                ? "not running"
                : "not running — the first start creates Server/.venv.nosync "
                    + "(this can take many minutes)"
        }
    }

    /// The connection dot's auto-connect reports its outcome here so the
    /// menu's one status line says connected-or-why-not.
    public func noteAutoConnectOutcome(_ outcome: String) {
        guard phase == .running else { return }
        let base = wasAdopted ? "running (started earlier) at " : "running at "
        statusLine = base + "http://127.0.0.1:\(port) — " + outcome
    }

    /// Launches the server; progress streams into `host`'s Activity pane.
    public func start(host: ChatService) {
        guard phase == .idle else { return }
        guard let script = LocalPythonRuntime.startServerScript,
            let repoRoot = LocalPythonRuntime.repoRoot
        else {
            statusLine = "cannot start: " + LocalPythonRuntime.unavailableHint
            return
        }
        guard FileManager.default.fileExists(atPath: script.path) else {
            statusLine = "cannot start: \(script.path) is missing from the code "
                + "checkout — restore it (git checkout scripts/) and try again"
            return
        }
        let workspaceRoot = VectorCatalog.projectRoot
        let title = "Local Python server"

        let process = Process()
        process.executableURL = URL(filePath: "/bin/zsh")
        process.arguments = [
            script.path, "--root", workspaceRoot.path, "--port", "\(port)",
        ]
        process.currentDirectoryURL = repoRoot
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        process.standardInput = FileHandle.nullDevice

        do {
            try process.run()
        } catch {
            statusLine = "failed to launch \(script.lastPathComponent): "
                + "\(error.localizedDescription)"
            return
        }

        self.process = process
        adoptedPID = nil
        wasAdopted = false
        phase = .starting
        statusLine = LocalPythonRuntime.venvExists
            ? "starting — output streams in the Activity pane"
            : "first-time setup (venv + pip install, MANY minutes) — progress "
                + "streams in the Activity pane"

        let initialLine = "serving workspace \(workspaceRoot.path) on 127.0.0.1:\(port)…"
        let logID = host.startLiveLog(title: title, initialLine: initialLine)
        let handle = pipe.fileHandleForReading

        monitor = Task { [weak self, weak host] in
            var lines = [initialLine]
            do {
                for try await line in handle.bytes.lines {
                    guard let self else { break }
                    lines.append(line)
                    if lines.count > 400 { lines.removeFirst(lines.count - 400) }
                    host?.updateLiveLog(id: logID, title: title, lines: lines)
                    if self.phase == .starting,
                        line.contains("Uvicorn running on")
                            || line.contains("Application startup complete")
                    {
                        self.phase = .running
                        self.statusLine =
                            "running at http://127.0.0.1:\(self.port) — serving "
                            + workspaceRoot.lastPathComponent
                    }
                }
            } catch {
                lines.append("log stream ended: \(error.localizedDescription)")
                host?.updateLiveLog(id: logID, title: title, lines: lines)
            }
            // EOF: the server has exited (or is mid-exit) — wait briefly for
            // the recorded exit code, then publish a plain-sentence outcome.
            while process.isRunning {
                try? await Task.sleep(for: .milliseconds(50))
            }
            guard let self else { return }
            let summary = Self.exitSummary(
                code: process.terminationStatus,
                wasStopping: self.phase == .stopping,
                port: self.port)
            lines.append(summary)
            host?.updateLiveLog(id: logID, title: title, lines: lines)
            self.phase = .idle
            self.statusLine = summary
            self.process = nil
            // The script wrote a pidfile for adoption; the server is gone
            // now, so it is stale — cleaned silently.
            LocalServerPidfile.cleanIfStale(workspaceRoot: workspaceRoot)
        }
    }

    /// SIGTERM — the script `exec`s the Python server, so the signal reaches
    /// uvicorn directly and it shuts down cleanly. An ADOPTED server has no
    /// `Process` handle: it is signalled by pid, watched until it exits, and
    /// its pidfile cleaned.
    public func stop() {
        if let process, process.isRunning {
            phase = .stopping
            statusLine = "stopping…"
            process.terminate()
            return
        }
        guard let adoptedPID, phase == .running else { return }
        phase = .stopping
        statusLine = "stopping…"
        kill(adoptedPID, SIGTERM)
        monitor = Task { [weak self] in
            // uvicorn shuts down within a couple of seconds; give it ten.
            for _ in 0..<200 where LocalServerPidfile.processIsAlive(adoptedPID) {
                try? await Task.sleep(for: .milliseconds(50))
            }
            guard let self else { return }
            if LocalServerPidfile.processIsAlive(adoptedPID) {
                self.phase = .running
                self.statusLine = "did not stop: pid \(adoptedPID) is still "
                    + "running — stop it from the terminal that started it"
                return
            }
            LocalServerPidfile.cleanIfStale(
                workspaceRoot: VectorCatalog.projectRoot)
            self.adoptedPID = nil
            self.phase = .idle
            self.statusLine = "stopped"
        }
    }

    /// Exit code → one plain sentence (with remedy). Pure and testable —
    /// codes match scripts/start-local-server.sh.
    nonisolated public static func exitSummary(
        code: Int32, wasStopping: Bool, port: Int
    ) -> String {
        if wasStopping {
            return "stopped"
        }
        switch code {
        case 0:
            return "stopped"
        case 3:
            return "not started: port \(port) is already in use — if that is a "
                + "SteerLab server already running, connect to it instead; "
                + "otherwise stop the other process (details in the Activity pane)"
        case 4:
            return "not started: python3 was not found — install the Xcode "
                + "Command Line Tools or Python from python.org, then try again"
        case 15, 143:
            // SIGTERM without our Stop — killed externally.
            return "stopped (terminated externally)"
        default:
            return "the server exited unexpectedly (code \(code)) — the last "
                + "output lines are in the Activity pane"
        }
    }
}

// MARK: - Stimulus-independence screen (firewall check)

/// App-side runner for the Python stimulus-independence screen
/// (`Server/steerlab_server/experiment/stimulus_screen.py`) — the
/// circularity firewall's data half: a concept's stimulus/validation text
/// must not contain the STUDY's forbidden vocabulary, or the extracted
/// "concept" direction partly encodes the task domain and every downstream
/// effect is confounded. WHICH vocabulary is forbidden is workspace DATA
/// (`prompts/screens/forbidden-vocabulary.json`, falling back to the shipped
/// judicial-study default under `prompts/templates/screens/`) — the screen's
/// output names the governing vocabulary on its first line. The science
/// stays in Python (single implementation, both entry points); this type
/// only launches it through the local venv, streams its findings, and
/// phrases the verdict plainly.
public enum StimulusIndependenceScreen {

    public struct Outcome: Sendable {
        /// nil when the screen never ran (missing venv/directory).
        public let exitCode: Int32?
        public let verdict: String
    }

    /// Exit code + captured output → one plain sentence. Pure and testable.
    /// Codes match `stimulus_screen.main`: 0 clean, 1 findings, 2 usage.
    public static func verdict(exitCode: Int32, output: [String]) -> String {
        switch exitCode {
        case 0:
            return "Clean — no forbidden-vocabulary terms in this concept's "
                + "stimulus or validation text (the governing vocabulary is "
                + "named in the Activity pane)."
        case 1:
            let count = flaggedLineCount(in: output).map(String.init) ?? "Some"
            return "\(count) flagged line(s) contain forbidden vocabulary — "
                + "review each in the Activity pane before pinning this set. "
                + "The screen is deliberately over-broad: flagged means "
                + "\u{201C}look at it\u{201D}, not \u{201C}rejected\u{201D}."
        case 2:
            return "The screen could not run (bad arguments or a missing "
                + "directory) — details in the Activity pane."
        default:
            return "The screen failed (exit \(exitCode)) — details in the "
                + "Activity pane."
        }
    }

    /// Parses the module's summary line
    /// ("<N> flagged line(s); review before pinning this set").
    public static func flaggedLineCount(in output: [String]) -> Int? {
        for line in output.reversed() {
            guard line.contains("flagged line(s)") else { continue }
            guard let first = line.split(separator: " ").first else { continue }
            if let count = Int(first) { return count }
        }
        return nil
    }

    /// Runs the screen on one concept directory
    /// (`prompts/concepts/<name>/`), streaming each output line to `onLine`.
    /// Never throws: every failure mode becomes a plain-sentence verdict.
    @MainActor
    public static func run(
        conceptDirectory: URL,
        onLine: @escaping @MainActor (String) -> Void
    ) async -> Outcome {
        var isDirectory: ObjCBool = false
        guard
            FileManager.default.fileExists(
                atPath: conceptDirectory.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            let verdict = "No stimulus directory at \(conceptDirectory.path) — "
                + "save this concept's datasets first."
            onLine(verdict)
            return Outcome(exitCode: nil, verdict: verdict)
        }
        guard let venvPython = LocalPythonRuntime.venvPython,
            let repoRoot = LocalPythonRuntime.repoRoot
        else {
            let verdict =
                "Cannot run the screen: " + LocalPythonRuntime.unavailableHint
            onLine(verdict)
            return Outcome(exitCode: nil, verdict: verdict)
        }
        guard LocalPythonRuntime.venvExists else {
            let verdict = "Cannot run the screen: " + LocalPythonRuntime.setupHint
            onLine(verdict)
            return Outcome(exitCode: nil, verdict: verdict)
        }

        let process = Process()
        process.executableURL = venvPython
        process.arguments = [
            "-u", "-m", "steerlab_server.experiment.stimulus_screen",
            conceptDirectory.path,
        ]
        process.currentDirectoryURL = repoRoot
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        process.standardInput = FileHandle.nullDevice

        do {
            try process.run()
        } catch {
            let verdict = "Could not launch the venv Python "
                + "(\(venvPython.path)): "
                + "\(error.localizedDescription)"
            onLine(verdict)
            return Outcome(exitCode: nil, verdict: verdict)
        }

        var output: [String] = []
        let handle = pipe.fileHandleForReading
        do {
            for try await line in handle.bytes.lines {
                output.append(line)
                onLine(line)
            }
        } catch {
            onLine("output stream ended: \(error.localizedDescription)")
        }
        while process.isRunning {
            try? await Task.sleep(for: .milliseconds(50))
        }
        let code = process.terminationStatus
        return Outcome(exitCode: code, verdict: verdict(exitCode: code, output: output))
    }
}
