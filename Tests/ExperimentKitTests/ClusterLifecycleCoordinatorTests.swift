import Foundation
import Testing

@testable import ExperimentKit

// MARK: - Fakes (no ssh, no sockets, no Keychain)

/// Pattern-matched shell: rules are tried in order, first match wins, and
/// every argv is recorded so a test can prove a command NEVER ran.
private actor FakeClusterShell: ClusterShellRunner {
    private struct Rule {
        var match: @Sendable ([String]) -> Bool
        var result: ClusterShellResult
    }

    private var rules: [Rule] = []
    private var calls: [[String]] = []
    /// What an unmatched command does. Default: a plain failure, which every
    /// observation reads as "absent" — the cold-site shape.
    private var fallback = ClusterShellResult(exitCode: 1)

    /// Register a rule. The MOST RECENT registration wins, so a test can
    /// script a healthy site and then override one layer.
    func on(
        _ needle: String, exit: Int32 = 0, lines: [String] = []
    ) {
        rules.insert(
            Rule(
                match: { argv in argv.joined(separator: " ").contains(needle) },
                result: ClusterShellResult(exitCode: exit, lines: lines)),
            at: 0)
    }

    func run(_ argv: [String]) async -> ClusterShellResult {
        calls.append(argv)
        for rule in rules where rule.match(argv) { return rule.result }
        return fallback
    }

    func recordedCalls() -> [[String]] { calls }
    func joinedCalls() -> [String] { calls.map { $0.joined(separator: " ") } }
    func callCount(containing needle: String) -> Int {
        calls.filter { $0.joined(separator: " ").contains(needle) }.count
    }
}

/// In-memory secret store. Holds a KNOWN token so the redaction tests can
/// search every produced byte for it.
private final class FakeSecretStore: ClusterSecretStore, @unchecked Sendable {
    // @unchecked Sendable is justified: the dictionary is mutated only from
    // the serialized test body and the coordinator's single task chain, and
    // this type never escapes a test.
    private var storage: [String: String]

    init(_ storage: [String: String] = [:]) { self.storage = storage }

    func token(forKey key: String) -> String? { storage[key] }
    /// Explicit rather than inherited: the protocol has no default, so a
    /// presence claim can never silently become a secret read.
    func hasToken(forKey key: String) -> Bool { storage[key] != nil }
    func store(_ token: String, forKey key: String) throws { storage[key] = token }
    func removeToken(forKey key: String) { storage[key] = nil }
    var storedKeys: [String] { storage.keys.sorted() }
}

private struct FakeEndpointProbe: ClusterEndpointProbe {
    var reachableWithToken: Bool = true
    var build: String? = "steerlab-server 0.1.0+46bb4f9a"

    func probe(baseURL: URL, token: String?) async -> ClusterEndpointProbeResult {
        guard token != nil else {
            return ClusterEndpointProbeResult(reachable: false, authFailed: true)
        }
        guard reachableWithToken else {
            return ClusterEndpointProbeResult(reachable: false, detail: "no answer")
        }
        return ClusterEndpointProbeResult(
            reachable: true, serverBuild: build, serverRole: "controller",
            root: "/scratch/me/ws")
    }
}

private final class FakeTunnelController: ClusterTunnelControlling, @unchecked Sendable {
    // @unchecked Sendable: same rationale as FakeSecretStore.
    var observation: ClusterTunnelObservation
    private(set) var openCount = 0
    private(set) var lastTargetHost: String?

    init(observation: ClusterTunnelObservation = .absent) {
        self.observation = observation
    }

    func observe(
        site: ClusterSiteProfile, persistedPort: Int?, targetHost: String?
    ) async -> ClusterTunnelObservation {
        observation
    }

    func open(
        site: ClusterSiteProfile, targetHost: String, persistedPort: Int?
    ) async -> ClusterTunnelOpenOutcome {
        openCount += 1
        lastTargetHost = targetHost
        let port = persistedPort ?? 8712
        observation = .up(localPort: port)
        return ClusterTunnelOpenOutcome(
            observation: observation, message: "forward installed",
            forwardIdentity: "fake://\(targetHost):\(port)", changed: true)
    }

    func close(
        site: ClusterSiteProfile, localPort: Int, targetHost: String
    ) async -> String? { nil }
}

private final class FakeAuthenticationLauncher: ClusterAuthenticationLauncher,
    @unchecked Sendable
{
    // @unchecked Sendable: same rationale as FakeSecretStore.
    private(set) var openCount = 0

    func authenticationCommand(for site: ClusterSiteProfile) -> String? {
        "ssh -o ControlMaster=auto login.test"
    }

    func openAuthenticationTerminal(for site: ClusterSiteProfile) async -> Bool {
        openCount += 1
        return true
    }
}

// MARK: - Tests

/// The headless lifecycle engine (Phase B of
/// `docs/CLUSTER-CLI-LIFECYCLE-PLAN.md`): inspector + planner + extracted
/// operations + durable store, driven end to end with fakes — no ssh, no
/// sockets, no Keychain, no SwiftUI. This is the plan's Phase B acceptance
/// gate: a fake site walks from each partial state toward `connected` without
/// importing `SteerLabApp`.
struct ClusterLifecycleCoordinatorTests {

    /// The token every redaction assertion hunts for.
    private static let secretToken = "sk-SUPERSECRET-do-not-persist-42"

    /// Every readable file under `root`, as text — the redaction test's
    /// haystack. Synchronous (directory enumeration is unavailable from an
    /// async context).
    private static func allFileContents(under root: URL) -> [String] {
        guard let enumerator = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: nil)
        else { return [] }
        var out: [String] = []
        for case let url as URL in enumerator {
            if let text = try? String(contentsOf: url, encoding: .utf8) {
                out.append(text)
            }
        }
        return out
    }

    // MARK: Harness

    private struct Harness {
        var repository: ClusterSiteRepository
        var operationStore: ClusterOperationStore
        var shell: FakeClusterShell
        var secrets: FakeSecretStore
        var tunnel: FakeTunnelController
        var launcher: FakeAuthenticationLauncher
        var coordinator: ClusterLifecycleCoordinator
        var root: URL
        var payloadRoot: URL
        /// The bytes the packaged deployment manifest holds locally; the fake
        /// remote `cat` returns the same to make the payload read `current`.
        var manifestBytes: String
    }

    private func temporaryDirectory(_ name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(components: "steerlab-cluster-lifecycle", "\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func slurmProfile() -> ClusterSiteProfile {
        ClusterSiteProfile(
            name: "Test Cluster",
            transport: .ssh(
                host: "login.test", proxyJump: nil, remotePort: 8080, vpnExpected: false),
            topology: .daemonInJob,
            scheduler: .slurm(
                ClusterSiteProfile.SlurmSiteData(
                    partitions: [.init(name: "batch", maxWalltimeHours: 168)],
                    defaultPartition: "batch")),
            constraints: ClusterSiteProfile.SiteConstraints(
                storageRoots: ["workspace": "/scratch/me/ws", "hfCache": "/work/lab/hf"]))
    }

    private func makeHarness(
        _ name: String,
        profile: ClusterSiteProfile? = nil,
        storedToken: String? = nil,
        tunnel: ClusterTunnelObservation = .absent,
        endpoint: FakeEndpointProbe = FakeEndpointProbe()
    ) throws -> Harness {
        let root = try temporaryDirectory(name)
        let payloadRoot = root.appending(component: "payload")
        try FileManager.default.createDirectory(
            at: payloadRoot, withIntermediateDirectories: true)
        let manifestBytes = #"{"schemaVersion":1,"files":[]}"#
        try Data(manifestBytes.utf8).write(
            to: payloadRoot.appending(
                component: ClusterProvisioner.deploymentManifestFileName))

        let repository = ClusterSiteRepository(
            directory: root.appending(component: "cluster-sites"),
            legacyRegistryData: { nil })
        _ = try repository.upsert(profile: profile ?? slurmProfile())

        let operationStore = ClusterOperationStore(
            rootDirectory: root.appending(component: "cluster-operations"))
        let shell = FakeClusterShell()
        let secrets = FakeSecretStore(
            storedToken.map { ["login.test:8080": $0] } ?? [:])
        let tunnelController = FakeTunnelController(observation: tunnel)
        let launcher = FakeAuthenticationLauncher()

        var configuration = ClusterProvisioningConfiguration()
        configuration.localPayloadPath = payloadRoot.path
        configuration.bootstrapExecutionTarget = .slurmBatch
        configuration.bootstrapJobPartition = "batch"

        let operations = ClusterProvisioningOperations(shell: shell, secrets: secrets)
        let inspector = ClusterLifecycleInspector(
            operations: operations, tunnel: tunnelController, endpoint: endpoint,
            secrets: secrets, now: { Date(timeIntervalSince1970: 1_000) })
        let coordinator = ClusterLifecycleCoordinator(
            repository: repository, operationStore: operationStore,
            operations: operations, inspector: inspector, tunnel: tunnelController,
            endpointProbe: endpoint, authenticationLauncher: launcher,
            secrets: secrets, configuration: configuration,
            now: { Date(timeIntervalSince1970: 1_000) })

        return Harness(
            repository: repository, operationStore: operationStore, shell: shell,
            secrets: secrets, tunnel: tunnelController, launcher: launcher,
            coordinator: coordinator, root: root, payloadRoot: payloadRoot,
            manifestBytes: manifestBytes)
    }

    /// Teach the fake shell the answers a fully healthy site would give.
    private func scriptHealthySite(
        _ harness: Harness, controllerState: String = "RUNNING|None",
        daemonHost: String = "c4-12"
    ) async {
        await harness.shell.on("-O check", exit: 0)
        // The command probe behind `-O check`: a live master runs `true`.
        await harness.shell.on("ConnectTimeout", exit: 0)
        await harness.shell.on(
            ClusterProvisioner.deploymentManifestFileName, exit: 0,
            lines: [harness.manifestBytes])
        await harness.shell.on(
            "STEERLAB_PREFIX", exit: 0, lines: ["/home/me/envs/steerlab"])
        await harness.shell.on("squeue", exit: 0, lines: [controllerState])
        await harness.shell.on("serverd.host", exit: 0, lines: [daemonHost])
        await harness.shell.on(
            "profile validate", exit: 0, lines: ["OK   profile: cluster/login/slurm"])
        await harness.shell.on(
            ".steerlab-token", exit: 0, lines: [Self.secretToken])
    }

    /// Pre-seed a durable record so inspection has a controller job to
    /// reconcile — exactly what a previous CLI invocation would have left.
    private func seedControllerJob(_ harness: Harness, jobID: String = "4242") throws {
        var record = ClusterOperationRecord(
            operationID: "op-seed", siteID: "test-cluster", siteProfileHash: "seed",
            target: .controllerRunning, state: .pending,
            startedAt: Date(timeIntervalSince1970: 900))
        record.controllerJobID = jobID
        try harness.operationStore.save(record)
    }

    // MARK: Cold site — the human boundary

    @Test func aColdSiteStopsAtTheHumanAuthenticationBoundary() async throws {
        let harness = try makeHarness("cold")
        // No rules at all: every probe fails, so there is no ControlMaster.
        let result = try await harness.coordinator.ensure(
            siteReference: "test-cluster", target: .connected,
            permissions: [.push, .bootstrap, .controllerStart, .openAuthTerminal])

        #expect(result.state == .needsHumanAuthentication)
        #expect(result.step == .authenticate)
        #expect(result.retryAfterSeconds == 5)
        #expect(result.nextAction?.requiresHuman == true)
        #expect(result.exitCode == 10)
        // The Terminal is opened ONCE, and nothing remote was attempted.
        #expect(harness.launcher.openCount == 1)
        #expect(await harness.shell.callCount(containing: "/usr/bin/rsync") == 0)
        // "sbatch" is also a substring of the rendered controller
        // script's FILE NAME, which the read-only staleness probe
        // names — so "no submission" has to be asserted about the
        // SUBMISSION, not the substring (§1 field report, 2026-08-20).
        #expect(await harness.shell.callCount(containing: "&& sbatch ") == 0)
        // The failing LAYER is named, not just the step.
        #expect(result.message.contains("controlMaster"))
    }

    @Test func theAuthTerminalOpensOnlyWhenSeparatelyAuthorized() async throws {
        let harness = try makeHarness("cold-no-terminal")
        let result = try await harness.coordinator.ensure(
            siteReference: "test-cluster", target: .connected,
            permissions: .allMutations)
        #expect(result.state == .needsHumanAuthentication)
        // Every MUTATION permission granted and the window still did not open:
        // opening it is its own authorization.
        #expect(harness.launcher.openCount == 0)
    }

    // MARK: Permission gating, one mutation at a time

    @Test func pushIsNeverRunWithoutItsOwnPermission() async throws {
        let harness = try makeHarness("no-push")
        await harness.shell.on("-O check", exit: 0)
        await harness.shell.on("ConnectTimeout", exit: 0)
        // The remote manifest is absent ⇒ the payload is not deployed.
        await harness.shell.on(ClusterProvisioner.deploymentManifestFileName, exit: 1)

        let result = try await harness.coordinator.ensure(
            siteReference: "test-cluster", target: .codeDeployed,
            permissions: [.bootstrap, .controllerStart])
        #expect(result.state == .needsApproval)
        #expect(result.step == .pushCode)
        #expect(result.nextAction?.missingPermissionFlags == ["--allow-push"])
        #expect(result.exitCode == 11)
        #expect(await harness.shell.callCount(containing: "/usr/bin/rsync") == 0)
        #expect(!result.changed)
    }

    @Test func bootstrapApplyIsNeverRunWithoutItsOwnPermission() async throws {
        let harness = try makeHarness("no-bootstrap")
        await scriptHealthySite(harness)
        // …except the environment, which is absent.
        await harness.shell.on("STEERLAB_PREFIX", exit: 1)
        await harness.shell.on(
            "bootstrap", exit: 0,
            lines: [#"{"ok":true,"steps":{"condaDetect":"planned"},"envFile":"/e","prefix":"/p"}"#])

        let result = try await harness.coordinator.ensure(
            siteReference: "test-cluster", target: .bootstrapped,
            permissions: [.push, .controllerStart])
        #expect(result.state == .needsApproval)
        #expect(result.step == .bootstrapApply)
        #expect(result.nextAction?.missingPermissionFlags == ["--allow-bootstrap"])
        // The read-only DRY RUN did run (reviewing a plan needs no approval)…
        let calls = await harness.shell.joinedCalls()
        #expect(calls.contains { $0.contains("--dry-run") })
        // …and the real one did not.
        #expect(!calls.contains { $0.contains("bootstrap") && !$0.contains("--dry-run") })
    }

    @Test func controllerStartIsNeverRunWithoutItsOwnPermission() async throws {
        let harness = try makeHarness("no-controller")
        await scriptHealthySite(harness)
        // squeue answers that the job left the queue, and no job is recorded.
        await harness.shell.on("squeue", exit: 0, lines: [])
        await harness.shell.on("serverd.host", exit: 1)

        let result = try await harness.coordinator.ensure(
            siteReference: "test-cluster", target: .controllerRunning,
            permissions: [.push, .bootstrap])
        #expect(result.state == .needsApproval)
        #expect(result.step == .controllerStart)
        #expect(result.nextAction?.missingPermissionFlags == ["--allow-controller-start"])
        #expect(await harness.shell.callCount(containing: "&& sbatch ") == 0)
    }

    // MARK: Controller reconciliation

    @Test func aQueuedControllerStaysPendingAndIsNeverResubmitted() async throws {
        let harness = try makeHarness("pending")
        await scriptHealthySite(harness, controllerState: "PENDING|Resources")
        await harness.shell.on("serverd.host", exit: 1)
        try seedControllerJob(harness)

        let first = try await harness.coordinator.ensure(
            siteReference: "test-cluster", target: .controllerRunning,
            permissions: .allMutations)
        #expect(first.state == .pending)
        #expect(first.step == .controllerWait)
        #expect(first.schedulerJobID == "4242")
        #expect(first.schedulerState == "PENDING")
        #expect(first.retryAfterSeconds == 30)
        #expect(first.exitCode == 12)

        // Polling it again keeps it pending — a wait never decays into a
        // timeout failure, and it never mints a second job.
        let second = try await harness.coordinator.ensure(
            siteReference: "test-cluster", target: .controllerRunning,
            permissions: .allMutations)
        #expect(second.state == .pending)
        #expect(!second.changed)
        #expect(await harness.shell.callCount(containing: "&& sbatch ") == 0)
    }

    @Test func anUnreadableSchedulerStateDegradesAndNeverResubmits() async throws {
        let harness = try makeHarness("scheduler-down")
        await scriptHealthySite(harness)
        // ssh itself failed: the probe is unavailable, the job is unknown.
        await harness.shell.on("squeue", exit: 255, lines: ["ssh: connect: timed out"])
        try seedControllerJob(harness)

        let result = try await harness.coordinator.ensure(
            siteReference: "test-cluster", target: .controllerRunning,
            permissions: .allMutations)
        #expect(result.state == .degraded)
        #expect(result.step == .controllerWait)
        #expect(result.exitCode == 13)
        #expect(await harness.shell.callCount(containing: "&& sbatch ") == 0)
        #expect(result.observed.controller == .unknown(
            reason: "the scheduler probe failed, so the job's state is unknown — "
                + "an unproven death never licenses a resubmit"))
    }

    @Test func controllerReconciliationCoversEveryState() async throws {
        // absent (squeue answered, nothing queued) with no host file.
        let absent = try makeHarness("controller-absent")
        await scriptHealthySite(absent, controllerState: "")
        await absent.shell.on("squeue", exit: 0, lines: [])
        await absent.shell.on("serverd.host", exit: 1)
        try seedControllerJob(absent)
        var observed = try await absent.coordinator.status(siteReference: "test-cluster")
        #expect(observed.controller == .absent)
        #expect(observed.daemonHost == .absent)

        // pending, before any host file exists.
        let pending = try makeHarness("controller-pending")
        await scriptHealthySite(pending, controllerState: "PENDING|Priority")
        await pending.shell.on("serverd.host", exit: 1)
        try seedControllerJob(pending)
        observed = try await pending.coordinator.status(siteReference: "test-cluster")
        #expect(observed.controller == .pending(jobID: "4242", reason: "Priority"))
        #expect(observed.daemonHost == .absent)

        // running, but the host file has not been written yet.
        let starting = try makeHarness("controller-starting")
        await scriptHealthySite(starting)
        await starting.shell.on("serverd.host", exit: 1)
        try seedControllerJob(starting)
        observed = try await starting.coordinator.status(siteReference: "test-cluster")
        #expect(observed.controller == .running(jobID: "4242"))
        #expect(observed.daemonHost == .absent)

        // running WITH a current host file — the only trustworthy shape.
        let healthy = try makeHarness("controller-healthy")
        await scriptHealthySite(healthy)
        try seedControllerJob(healthy)
        observed = try await healthy.coordinator.status(siteReference: "test-cluster")
        #expect(observed.controller == .running(jobID: "4242"))
        #expect(observed.daemonHost == .current(host: "c4-12"))

        // failed.
        let failed = try makeHarness("controller-failed")
        await scriptHealthySite(failed, controllerState: "TIMEOUT|TimeLimit")
        try seedControllerJob(failed)
        observed = try await failed.coordinator.status(siteReference: "test-cluster")
        #expect(observed.controller == .failed(jobID: "4242", state: "TIMEOUT"))
        // A host file that outlived its job is STALE, never proof of life.
        if case .stale(let host, _) = observed.daemonHost {
            #expect(host == "c4-12")
        } else {
            Issue.record("a host file from a dead job must read stale, got \(observed.daemonHost)")
        }

        // A host file with NO recorded job of ours: unattributable, so the
        // state is unknown — which the planner refuses to treat as a start.
        let orphan = try makeHarness("controller-orphan")
        await scriptHealthySite(orphan)
        observed = try await orphan.coordinator.status(siteReference: "test-cluster")
        if case .unknown = observed.controller {} else {
            Issue.record("an unattributable host file must not read absent")
        }
    }

    // MARK: Idempotence

    @Test func repeatingEnsureAtReadyIsANoOp() async throws {
        let harness = try makeHarness(
            "warm", storedToken: Self.secretToken, tunnel: .up(localPort: 8712))
        await scriptHealthySite(harness)
        try seedControllerJob(harness)
        _ = try harness.repository.noteConnection(
            siteID: "test-cluster", endpoint: "http://127.0.0.1:8712",
            serverBuild: "steerlab-server 0.1.0+46bb4f9a")

        let first = try await harness.coordinator.ensure(
            siteReference: "test-cluster", target: .connected,
            permissions: .allMutations)
        #expect(first.state == .ready)
        #expect(first.endpoint == "http://127.0.0.1:8712")
        #expect(first.tokenAvailable)
        #expect(first.tokenSource == "keychain")
        #expect(first.serverBuild == "steerlab-server 0.1.0+46bb4f9a")
        // The only step it needed was the read-only validate; nothing mutated.
        #expect(!first.changed)
        #expect(harness.tunnel.openCount == 0)
        #expect(await harness.shell.callCount(containing: "/usr/bin/rsync") == 0)
        #expect(await harness.shell.callCount(containing: "&& sbatch ") == 0)

        let validateCalls = await harness.shell.callCount(containing: "profile validate")
        let second = try await harness.coordinator.ensure(
            siteReference: "test-cluster", target: .connected,
            permissions: .allMutations)
        #expect(second.state == .ready)
        #expect(!second.changed)
        #expect(second.exitCode == 0)
        // The remembered validation means the second pass does not even
        // re-run the read-only remote check.
        #expect(await harness.shell.callCount(containing: "profile validate") == validateCalls)
        #expect(harness.tunnel.openCount == 0)
    }

    @Test func anExistingForwardIsAdoptedNotDuplicated() async throws {
        let harness = try makeHarness(
            "adopt", storedToken: Self.secretToken, tunnel: .up(localPort: 8712))
        await scriptHealthySite(harness)
        try seedControllerJob(harness)
        // The registry does not know the endpoint yet (the APP opened this
        // forward) — the CLI must register it, not open a second one.
        let result = try await harness.coordinator.ensure(
            siteReference: "test-cluster", target: .connected,
            permissions: .allMutations)
        #expect(result.state == .ready)
        #expect(harness.tunnel.openCount == 0)
        let stored = try #require(try harness.repository.site(id: "test-cluster"))
        #expect(stored.lastEndpoint == "http://127.0.0.1:8712")
        #expect(stored.lastServerBuild == "steerlab-server 0.1.0+46bb4f9a")
    }

    // MARK: Interprocess exclusion

    @Test func aSecondCallerObservesTheRunningOperationInsteadOfDuplicatingIt()
        async throws
    {
        let harness = try makeHarness("locked")
        await scriptHealthySite(harness)
        // Another process holds the site lock and has a record in flight.
        let lock = try #require(try harness.operationStore.acquireLock(siteID: "test-cluster"))
        defer { lock.release() }
        var running = ClusterOperationRecord(
            operationID: "op-other", siteID: "test-cluster", siteProfileHash: "h",
            target: .connected, state: .pending)
        running.controllerJobID = "4242"
        try harness.operationStore.save(running)

        await #expect(throws: ClusterLifecycleError.self) {
            _ = try await harness.coordinator.ensure(
                siteReference: "test-cluster", target: .connected,
                permissions: .allMutations)
        }
        do {
            _ = try await harness.coordinator.ensure(
                siteReference: "test-cluster", target: .connected,
                permissions: .allMutations)
            Issue.record("a second caller must refuse rather than duplicate")
        } catch let error as ClusterLifecycleError {
            #expect(error.code == "operationInProgress")
            #expect(error.errorDescription?.contains("op-other") == true)
        }
        // Nothing was mutated by the refused caller.
        #expect(await harness.shell.callCount(containing: "&& sbatch ") == 0)
        #expect(await harness.shell.callCount(containing: "/usr/bin/rsync") == 0)

        // Read-only inspection is ALWAYS allowed, lock or no lock.
        let observed = try await harness.coordinator.status(siteReference: "test-cluster")
        #expect(observed.controlMaster == .alive)
    }

    // MARK: Secrets

    @Test func noSecretReachesAnyRecordResultOrRegistryByte() async throws {
        let harness = try makeHarness("secrets", tunnel: .up(localPort: 8712))
        await scriptHealthySite(harness)
        try seedControllerJob(harness)

        let result = try await harness.coordinator.ensure(
            siteReference: "test-cluster", target: .connected,
            permissions: .allMutations)
        #expect(result.state == .ready)
        // The token WAS imported (that is the point of the step)…
        #expect(harness.secrets.token(forKey: "login.test:8080") == Self.secretToken)
        #expect(result.tokenAvailable)

        // …and appears in NOTHING else. Every produced byte is searched.
        var haystacks: [String] = [
            result.message, result.operationID, result.endpoint ?? "",
            result.tokenSource ?? "", result.serverBuild ?? "",
            result.nextAction?.detail ?? "",
        ]
        haystacks += result.observed.layerSummaries.map { "\($0.layer)=\($0.state)" }
        haystacks += result.plan.transitions.map(\.reason)
        for record in harness.operationStore.records(forSite: "test-cluster") {
            haystacks += record.transcript
            haystacks += record.steps.map(\.message)
            haystacks += [record.tokenSource ?? "", record.repairAction ?? ""]
        }
        // Plus the raw files, which is the assertion that actually matters.
        haystacks += Self.allFileContents(under: harness.root)
        for haystack in haystacks {
            #expect(
                !haystack.contains(Self.secretToken),
                "token material leaked into: \(haystack.prefix(200))")
        }
        // And the record does say the token is available, by presence only.
        let record = try #require(harness.operationStore.latestRecord(forSite: "test-cluster"))
        #expect(record.tokenAvailable)
        #expect(record.tokenSource == "keychain")
    }

    // MARK: Unknown site

    @Test func anUnknownSiteRefusesWithAStableCode() async throws {
        let harness = try makeHarness("unknown-site")
        do {
            _ = try await harness.coordinator.ensure(siteReference: "nope")
            Issue.record("expected a refusal")
        } catch let error as ClusterLifecycleError {
            #expect(error.code == "unknownSite")
        }
    }
}
