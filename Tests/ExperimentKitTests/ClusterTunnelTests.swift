import Foundation
import Testing

@testable import ExperimentKit

// MARK: - Stub process seam

/// Scripted stand-in for ssh: `-O check` answers from `masterAlive`, remote
/// `cat` answers from `daemonHostOutput`, launches mint `StubProcessHandle`s
/// the test finishes on demand. No real processes, no real sockets.
private actor StubTunnelRunner: TunnelProcessRunner {
    private var masterAlive = true
    private var daemonHostOutput = ""
    private var busyPorts: Set<Int> = []
    private var forwardExitCode: Int32 = 0
    private(set) var runCalls: [[String]] = []
    private(set) var launchCalls: [[String]] = []
    private var handles: [StubProcessHandle] = []

    func setMasterAlive(_ alive: Bool) { masterAlive = alive }
    func setDaemonHostOutput(_ output: String) { daemonHostOutput = output }
    func setBusyPorts(_ ports: Set<Int>) { busyPorts = ports }
    func setForwardExitCode(_ code: Int32) { forwardExitCode = code }
    func launchCallCount() -> Int { launchCalls.count }
    func recordedRunCalls() -> [[String]] { runCalls }
    func recordedLaunchCalls() -> [[String]] { launchCalls }
    func handle(at index: Int) -> StubProcessHandle? {
        index < handles.count ? handles[index] : nil
    }

    private(set) var stdinPayloads: [Data] = []
    func recordedStdinPayloads() -> [Data] { stdinPayloads }

    func run(
        _ executablePath: String, arguments: [String], input: Data
    ) async -> TunnelProcessResult {
        stdinPayloads.append(input)
        return await run(executablePath, arguments: arguments)
    }

    func run(_ executablePath: String, arguments: [String]) async -> TunnelProcessResult {
        runCalls.append([executablePath] + arguments)
        if arguments.contains("-O"), arguments.contains("check") {
            return TunnelProcessResult(
                exitCode: masterAlive ? 0 : 255, standardOutput: "", standardError: "")
        }
        if arguments.contains("cat") {
            let trimmed = daemonHostOutput.trimmingCharacters(in: .whitespacesAndNewlines)
            return TunnelProcessResult(
                exitCode: trimmed.isEmpty ? 1 : 0, standardOutput: daemonHostOutput,
                standardError: "")
        }
        if arguments.contains("-O"), arguments.contains("forward") {
            return TunnelProcessResult(
                exitCode: forwardExitCode, standardOutput: "",
                standardError: forwardExitCode == 0 ? "" : "forward failed")
        }
        return TunnelProcessResult(exitCode: 0, standardOutput: "", standardError: "")
    }

    func launch(
        _ executablePath: String, arguments: [String]
    ) async throws -> any TunnelProcessHandle {
        launchCalls.append([executablePath] + arguments)
        let handle = StubProcessHandle()
        handles.append(handle)
        return handle
    }

    func isLocalPortFree(_ port: Int) async -> Bool { !busyPorts.contains(port) }
}

private actor StubProcessHandle: TunnelProcessHandle {
    private var running = true
    private var exitCode: Int32?
    private(set) var terminated = false
    private var waiters: [CheckedContinuation<Int32, Never>] = []

    var isRunning: Bool { running }

    func terminate() {
        terminated = true
        finish(code: 143)
    }

    func finish(code: Int32) {
        guard running else { return }
        running = false
        exitCode = code
        let pending = waiters
        waiters.removeAll()
        for waiter in pending { waiter.resume(returning: code) }
    }

    func waitUntilExit() async -> Int32 {
        if let exitCode { return exitCode }
        return await withCheckedContinuation { waiters.append($0) }
    }
}

// MARK: - Tests

/// State-machine tests for the WS1 tunnel manager against the stubbed
/// process seam: needsAuth without a ControlMaster, up on the deterministic
/// local port, daemon-in-a-job target resolution, degraded + bounded-backoff
/// reopen on child exit, needsAuth when the master dies, and clean close.
@MainActor
struct ClusterTunnelTests {

    // MARK: HF token install (secret write over the master)

    @Test func installHFTokenWritesTheSecretOnStdinNeverArgv() async {
        let runner = StubTunnelRunner()
        let tunnel = makeTunnel(runner: runner)
        tunnel.configure(site: sshSite(hfCacheRoot: "/work/lab/hf-cache"))

        let error = await tunnel.installHFToken("hf_secret_value_123")

        #expect(error == nil)
        let calls = await runner.recordedRunCalls()
        // -O check, the guarded write, -O check, the test -s verification.
        let write = calls.first { $0.contains(where: { $0.contains("umask 077") }) } ?? []
        let command = write.last ?? ""
        #expect(command.contains("mkdir -p /work/lab/hf-cache"))
        #expect(command.contains("cat > /work/lab/hf-cache/token.tmp"))
        #expect(command.contains("mv -f /work/lab/hf-cache/token.tmp /work/lab/hf-cache/token"))
        // The secret travels ONLY on stdin: no argv of any call carries it.
        #expect(!calls.flatMap(\.self).contains { $0.contains("hf_secret_value_123") })
        let payloads = await runner.recordedStdinPayloads()
        #expect(payloads == [Data("hf_secret_value_123".utf8)])
        // Presence verification never reads the secret back.
        #expect(calls.contains { $0.suffix(3) == ["test", "-s", "/work/lab/hf-cache/token"] })
        #expect(!calls.contains { $0.suffix(2).first == "cat" })
    }

    // MARK: External judge key sync (key custody, 2026-07-19)

    @Test func syncJudgeKeyPushesTheKeyOnStdinToTheContractPath() async {
        let runner = StubTunnelRunner()
        let tunnel = makeTunnel(runner: runner)
        tunnel.configure(site: sshSite())

        let error = await tunnel.syncJudgeKey(
            .init(kind: "openrouter", key: "sk-or-capped-123"))

        #expect(error == nil)
        let calls = await runner.recordedRunCalls()
        let write = calls.first { $0.contains(where: { $0.contains("umask 077") }) } ?? []
        let command = write.last ?? ""
        // Born private, atomic on arrival, at the server's default-path
        // contract (`judge_credentials.DEFAULT_KEY_PATH`) — `~` stays bare
        // so it expands remotely.
        #expect(command.contains("mkdir -p ~/.steerlab"))
        #expect(command.contains("cat > ~/.steerlab/judge-key.tmp"))
        #expect(command.contains("mv -f ~/.steerlab/judge-key.tmp ~/.steerlab/judge-key"))
        // The key travels ONLY on stdin, as the server's JSON contract.
        #expect(!calls.flatMap(\.self).contains { $0.contains("sk-or-capped-123") })
        let payloads = await runner.recordedStdinPayloads()
        #expect(payloads == [Data("{\"key\":\"sk-or-capped-123\",\"kind\":\"openrouter\"}\n".utf8)])
    }

    @Test func syncJudgeKeyWithNoStoredKeyRemovesTheRemoteFile() async {
        // Deletion must PROPAGATE: a key cleared on the Mac cannot stay
        // live on the cluster — the no-key sync is an rm, not a no-op.
        let runner = StubTunnelRunner()
        let tunnel = makeTunnel(runner: runner)
        tunnel.configure(site: sshSite())

        let error = await tunnel.syncJudgeKey(nil)

        #expect(error == nil)
        let calls = await runner.recordedRunCalls()
        #expect(calls.contains { $0.last == "rm -f ~/.steerlab/judge-key" })
        // And nothing was written.
        #expect(await runner.recordedStdinPayloads().isEmpty)
    }

    @Test func remoteDeleteQuotesHostilePaths() {
        let arguments = ClusterTunnel.remoteDeleteArguments(
            host: "login.test", proxyJump: nil,
            path: "/tmp/x; touch /tmp/pwned")
        #expect(arguments.last == "rm -f '/tmp/x; touch /tmp/pwned'")
    }

    @Test func profileControlledPathsAreShellQuotedInEveryRemoteCommand() async {
        // Site JSON is an import/share format: a hostile metadataRoot or
        // hfCache root must arrive at the remote shell as inert data. ssh
        // joins post-host argv into one shell string, so quoting is the
        // only line of defense.
        let hostile = "/tmp/x; touch /tmp/pwned"
        let read = ClusterTunnel.remoteReadArguments(
            host: "login.test", proxyJump: nil, path: hostile)
        #expect(read.suffix(2) == ["cat", "'/tmp/x; touch /tmp/pwned'"])

        let runner = StubTunnelRunner()
        let tunnel = makeTunnel(runner: runner)
        tunnel.configure(site: sshSite(hfCacheRoot: "/work/evil; rm -rf $HOME"))
        _ = await tunnel.installHFToken("hf_x")
        let calls = await runner.recordedRunCalls()
        for call in calls.filter({ $0.contains(where: { $0.contains("evil") }) }) {
            let command = call.last ?? ""
            #expect(
                command.contains("'/work/evil; rm -rf $HOME/token'")
                    || command.contains("'/work/evil; rm -rf $HOME/token.tmp'")
                    || command.contains("'/work/evil; rm -rf $HOME'"),
                "unquoted hostile path reached the remote shell: \(command)")
        }
        // Benign paths stay bare so ~ still expands remotely.
        let benign = ClusterTunnel.remoteReadArguments(
            host: "login.test", proxyJump: nil, path: "~/.steerlab/serverd.host")
        #expect(benign.suffix(2) == ["cat", "~/.steerlab/serverd.host"])
    }

    @Test func installHFTokenRefusesWithoutACacheRoot() async {
        let runner = StubTunnelRunner()
        let tunnel = makeTunnel(runner: runner)
        tunnel.configure(site: sshSite())

        let error = await tunnel.installHFToken("hf_x")

        #expect(error?.contains("HF cache storage root") == true)
        #expect(await runner.recordedRunCalls().isEmpty)  // refused before any ssh
    }

    @Test func installHFTokenSurfacesAMissingMaster() async {
        let runner = StubTunnelRunner()
        await runner.setMasterAlive(false)
        let tunnel = makeTunnel(runner: runner)
        tunnel.configure(site: sshSite(hfCacheRoot: "/work/lab/hf-cache"))

        let error = await tunnel.installHFToken("hf_x")

        #expect(error?.contains("Authenticate") == true)
        #expect(await runner.recordedStdinPayloads().isEmpty)
    }

    private func sshSite(
        host: String = "login.cluster.test",
        proxyJump: String? = nil,
        remotePort: Int = 8080,
        topology: ClusterSiteProfile.Topology = .loginDaemon,
        metadataRoot: String? = nil,
        hfCacheRoot: String? = nil
    ) -> ClusterSiteProfile {
        var constraints = ClusterSiteProfile.SiteConstraints()
        if let metadataRoot { constraints.storageRoots["metadata"] = metadataRoot }
        if let hfCacheRoot { constraints.storageRoots["hfCache"] = hfCacheRoot }
        return ClusterSiteProfile(
            name: "Test Cluster",
            transport: .ssh(
                host: host, proxyJump: proxyJump, remotePort: remotePort, vpnExpected: false),
            topology: topology,
            scheduler: .slurm(ClusterSiteProfile.SlurmSiteData()),
            constraints: constraints)
    }

    private func makeTunnel(runner: StubTunnelRunner) -> ClusterTunnel {
        ClusterTunnel(
            runner: runner,
            defaults: UserDefaults(suiteName: "ClusterTunnelTests.\(UUID().uuidString)")!)
    }

    /// Poll (the reopen path hops through detached suspensions) with a
    /// generous timeout; returns the last evaluation.
    private func eventually(
        timeout: Duration = .seconds(5),
        _ condition: () async -> Bool
    ) async -> Bool {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        while clock.now < deadline {
            if await condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return await condition()
    }

    @Test func authenticationCommandIsTheExactOneLiner() {
        #expect(
            ClusterTunnel.authenticationCommand(for: sshSite(host: "slurm.example.edu"))
                == "ssh -o ControlMaster=auto -o ControlPersist=8h "
                + "-o ControlPath=~/.ssh/steerlab-cm-%C slurm.example.edu")
        let jumped = ClusterTunnel.authenticationCommand(
            for: sshSite(host: "gpu.internal", proxyJump: "login.edu"))
        #expect(jumped?.contains("-J login.edu") == true)
        #expect(ClusterTunnel.authenticationCommand(for: .gpuWorkstation) == nil)
        #expect(ClusterTunnel.authenticationCommand(for: sshSite(host: "   ")) == nil)
    }

    @Test func openWithoutMasterNeedsAuth() async {
        let runner = StubTunnelRunner()
        await runner.setMasterAlive(false)
        let tunnel = makeTunnel(runner: runner)
        tunnel.configure(site: sshSite())
        await tunnel.open()
        #expect(tunnel.state == .needsAuth)
        let runCalls = await runner.recordedRunCalls()
        #expect(runCalls.contains { $0.contains("-O") && $0.contains("check") })
        #expect(await runner.launchCallCount() == 0)
        #expect(tunnel.effectiveBaseURL == nil)
    }

    @Test func chatServiceConnectOpensTunnelAndSurfacesExactEndpointFailure() async {
        let suite = "ClusterTunnelTests.connect.\(UUID().uuidString)"
        let defaults = try! #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }

        let store = clusterStore(defaults: defaults)
        let site = sshSite(host: "user@hpc.example.edu")
        let entry = store.addSite(site)
        store.activeWorkspace = .server(entry.id)

        let runner = StubTunnelRunner()
        await runner.setMasterAlive(false)
        let tunnel = makeTunnel(runner: runner)
        store.attachTunnel(tunnel)
        let service = ChatService(cluster: store)

        await service.connectCluster()

        #expect(tunnel.state == .needsAuth)
        #expect(
            store.status
                == "connection failed: no SSH ControlMaster for "
                    + "user@hpc.example.edu — the site endpoint must exactly "
                    + "match the authenticated user@host; choose Authenticate in the "
                    + "connection menu")
        let runCalls = await runner.recordedRunCalls()
        #expect(runCalls.contains { $0.last == "user@hpc.example.edu" })
    }

    @Test func connectRetriesTheControllerStartupWindowThenStopsHonestly() async {
        // Tunnel up + nobody answering is the controller's NORMAL torch-import
        // startup window (observed live: manual Connect succeeded a couple of
        // minutes later). connectCluster must retry exactly that case — and
        // stay bounded, ending on the honest failure once the budget runs out.
        let suite = "ClusterTunnelTests.retry.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = clusterStore(defaults: defaults)
        let entry = store.addSite(sshSite())
        store.activeWorkspace = .server(entry.id)
        let runner = StubTunnelRunner()
        let tunnel = makeTunnel(runner: runner)
        store.attachTunnel(tunnel)
        let service = ChatService(cluster: store)
        service.connectStartupRetryLimit = 3
        service.connectStartupRetryDelay = .zero

        await service.connectCluster()

        #expect(tunnel.state != .needsAuth)  // the tunnel itself came up
        #expect(store.status?.contains("controller is not answering") == true)
    }

    @Test func openWithMasterGoesUpOnThePreferredPort() async throws {
        let runner = StubTunnelRunner()
        let tunnel = makeTunnel(runner: runner)
        let site = sshSite()
        tunnel.configure(site: site)
        await tunnel.open()
        let preferred = try #require(site.preferredLocalPort)
        #expect(tunnel.state == .up(localPort: preferred))
        #expect(tunnel.effectiveBaseURL?.absoluteString == "http://127.0.0.1:\(preferred)")
        let runCalls = await runner.recordedRunCalls()
        let forward = try #require(
            runCalls.first { $0.contains("-O") && $0.contains("forward") })
        #expect(forward.first == "/usr/bin/ssh")
        #expect(!forward.contains("-N"))
        #expect(forward.contains("-L"))
        #expect(forward.contains("127.0.0.1:\(preferred):localhost:8080"))
        #expect(forward.contains("ControlPath=~/.ssh/steerlab-cm-%C"))
        #expect(forward.contains("BatchMode=yes"))
        #expect(forward.last == "login.cluster.test")
        await tunnel.close()
        let finalCalls = await runner.recordedRunCalls()
        #expect(finalCalls.contains { $0.contains("-O") && $0.contains("cancel") })
    }

    @Test func busyPreferredPortWalksUpward() async throws {
        let runner = StubTunnelRunner()
        let site = sshSite()
        let preferred = try #require(site.preferredLocalPort)
        await runner.setBusyPorts([preferred, preferred + 1])
        let tunnel = makeTunnel(runner: runner)
        tunnel.configure(site: site)
        await tunnel.open()
        #expect(tunnel.state == .up(localPort: preferred + 2))
        await tunnel.close()
    }

    @Test func daemonInJobForwardsToTheJobNode() async throws {
        let runner = StubTunnelRunner()
        await runner.setDaemonHostOutput("ra3-17\n")
        let tunnel = makeTunnel(runner: runner)
        let site = sshSite(topology: .daemonInJob)
        tunnel.configure(site: site)
        await tunnel.open()
        guard case .up(let port) = tunnel.state else {
            Issue.record("expected .up, got \(tunnel.state)")
            return
        }
        let runCalls = await runner.recordedRunCalls()
        #expect(runCalls.contains { $0.contains("cat") && $0.contains("~/.steerlab/serverd.host") })
        #expect(
            runCalls.contains {
                $0.contains("forward") && $0.contains("127.0.0.1:\(port):ra3-17:8080")
            })
        await tunnel.close()
    }

    @Test func daemonInJobHonorsTheMetadataStorageRole() async {
        let runner = StubTunnelRunner()
        await runner.setDaemonHostOutput("node-1")
        let tunnel = makeTunnel(runner: runner)
        tunnel.configure(
            site: sshSite(topology: .daemonInJob, metadataRoot: "/home/me/.steerlab"))
        await tunnel.open()
        let runCalls = await runner.recordedRunCalls()
        #expect(runCalls.contains { $0.contains("/home/me/.steerlab/serverd.host") })
        await tunnel.close()
    }

    @Test func unreadableDaemonHostDegrades() async {
        let runner = StubTunnelRunner()  // daemonHostOutput stays empty → cat exits 1
        let tunnel = makeTunnel(runner: runner)
        tunnel.configure(site: sshSite(topology: .daemonInJob))
        await tunnel.open()
        guard case .degraded(let reason) = tunnel.state else {
            Issue.record("expected .degraded, got \(tunnel.state)")
            return
        }
        #expect(reason.contains("serverd.host"))
        #expect(await runner.launchCallCount() == 0)
    }

    @Test func successfulForwardCommandExitStaysUpAndRepeatedOpenIsANoOp() async throws {
        let runner = StubTunnelRunner()
        let tunnel = makeTunnel(runner: runner)
        let site = sshSite()
        tunnel.configure(site: site)
        await tunnel.open()
        let port = try #require(site.preferredLocalPort)
        #expect(tunnel.state == .up(localPort: port))
        await tunnel.open()
        let forwardCalls = await runner.recordedRunCalls().filter {
            $0.contains("-O") && $0.contains("forward")
        }
        #expect(forwardCalls.count == 1)
        #expect(tunnel.state == .up(localPort: port))
        await tunnel.close()
    }

    @Test func forwardFailureDegradesWithoutBlindRetry() async {
        let runner = StubTunnelRunner()
        await runner.setForwardExitCode(255)
        let tunnel = makeTunnel(runner: runner)
        tunnel.configure(site: sshSite())
        await tunnel.open()
        guard case .degraded(let reason) = tunnel.state else {
            Issue.record("expected .degraded, got \(tunnel.state)")
            return
        }
        #expect(reason.contains("forward failed"))
        let forwardCalls = await runner.recordedRunCalls().filter {
            $0.contains("-O") && $0.contains("forward")
        }
        #expect(forwardCalls.count == 1)
    }

    @Test func healthLoopStopsAtNeedsAuthWhenTheMasterDies() async {
        let runner = StubTunnelRunner()
        let tunnel = makeTunnel(runner: runner)
        tunnel.healthProbeInterval = .milliseconds(10)
        tunnel.healthProbe = { true }
        tunnel.configure(site: sshSite())
        await tunnel.open()
        guard case .up = tunnel.state else {
            Issue.record("expected .up, got \(tunnel.state)")
            return
        }
        await runner.setMasterAlive(false)
        let landed = await eventually { tunnel.state == .needsAuth }
        #expect(landed, "a dead master must surface the actionable needsAuth state")
        #expect(await runner.launchCallCount() == 0)
    }

    @Test func closeCancelsTheOwnedForwardCleanly() async {
        let runner = StubTunnelRunner()
        let tunnel = makeTunnel(runner: runner)
        tunnel.configure(site: sshSite())
        await tunnel.open()
        await tunnel.close()
        #expect(tunnel.state == .closed)
        let calls = await runner.recordedRunCalls()
        #expect(calls.contains { $0.contains("-O") && $0.contains("cancel") })
    }

    @Test func directTransportNeedsNoTunnel() async {
        let runner = StubTunnelRunner()
        let tunnel = makeTunnel(runner: runner)
        tunnel.configure(site: .gpuWorkstation)
        await tunnel.open()
        #expect(tunnel.state == .idle)
        #expect(tunnel.effectiveBaseURL?.absoluteString == "http://127.0.0.1:8080")
        #expect(await runner.launchCallCount() == 0)
    }

    @Test func blankHostDegradesInsteadOfLaunching() async {
        let runner = StubTunnelRunner()
        let tunnel = makeTunnel(runner: runner)
        tunnel.configure(site: sshSite(host: "  "))
        await tunnel.open()
        guard case .degraded = tunnel.state else {
            Issue.record("expected .degraded, got \(tunnel.state)")
            return
        }
        #expect(await runner.launchCallCount() == 0)
    }

    @Test func reconfigureIsANoOpForTheSameSiteAndResetsForAnother() async throws {
        let runner = StubTunnelRunner()
        let tunnel = makeTunnel(runner: runner)
        let site = sshSite()
        tunnel.configure(site: site)
        await tunnel.open()
        let port = try #require(site.preferredLocalPort)
        tunnel.configure(site: site)  // unrelated store updates must not drop a live tunnel
        #expect(tunnel.state == .up(localPort: port))
        tunnel.configure(site: nil)
        #expect(tunnel.state == .idle)
        let cancelled = await eventually {
            await runner.recordedRunCalls().contains {
                $0.contains("-O") && $0.contains("cancel")
            }
        }
        #expect(cancelled)
    }

    @Test func nextLaunchCancelsAPersistedForwardBeforeOpeningAnother() async {
        let suite = "ClusterTunnelTests.recovery.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        let runner = StubTunnelRunner()
        let first = ClusterTunnel(runner: runner, defaults: defaults)
        first.configure(site: sshSite())
        await first.open()  // simulate an app exit before close()

        let second = ClusterTunnel(runner: runner, defaults: defaults)
        second.configure(site: sshSite())
        await second.open()

        let controlCalls = await runner.recordedRunCalls().filter {
            $0.contains("-O") && ($0.contains("forward") || $0.contains("cancel"))
        }
        #expect(controlCalls.count == 3)
        #expect(controlCalls[0].contains("forward"))
        #expect(controlCalls[1].contains("cancel"))
        #expect(controlCalls[2].contains("forward"))
        await second.close()
    }

    @Test func readsBootstrapTokenThroughTheExistingMaster() async {
        let runner = StubTunnelRunner()
        await runner.setDaemonHostOutput("secret-token\n")
        let tunnel = makeTunnel(runner: runner)
        tunnel.configure(site: sshSite())
        let value = await tunnel.readRemoteTextFile("~/.steerlab-token")
        #expect(value == "secret-token")
        let calls = await runner.recordedRunCalls()
        #expect(calls.contains { $0.contains("cat") && $0.contains("~/.steerlab-token") })
    }

    @Test func authCommandFileIsExecutableAndContainsTheCommand() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "steerlab-tunnel-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = try ClusterTunnel.writeAuthCommandFile(
            command: "ssh -o ControlMaster=auto host", siteName: "Example HPC", in: directory)
        #expect(url.pathExtension == "command")
        let contents = try String(contentsOf: url, encoding: .utf8)
        #expect(contents.hasPrefix("#!/bin/zsh"))
        #expect(contents.contains("exec ssh -o ControlMaster=auto host"))
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        #expect((attributes[.posixPermissions] as? Int) == 0o755)
    }
}
