import Foundation
import Testing

@testable import ExperimentKit

// A live controller on 2026-09-13: the build script's eight-second launch
// check connected to a saved site through a live tunnel, evidence auto-import
// began fetching every succeeded job's bundle the ledger had never seen, the
// script killed the app mid-transfer, and the controller wedged. These tests
// pin the two halves of the app-side fix: the offline launch-check switch and
// the bounds on automatic evidence transfer.
//
// Serialized because refused attempts are recorded process-wide; the mode
// itself is task-local, so no other suite can observe it.
@MainActor @Suite(.serialized)
struct LaunchCheckAndTransferBoundsTests {

    private func freshDefaults(_ name: String) throws -> UserDefaults {
        let suite = "steerlab.tests.launch-check.\(name)-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    private func freshWorkspace(_ name: String) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appending(components: "steerlab-tests-launch-check", "\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func source(_ workspace: URL) -> EvidenceImportOrigin {
        .init(serverIdentity: "ssh://researcher@example.invalid:8080", remoteRoot: "/remote", workspaceRoot: workspace)
    }

    private func succeededRunJob(id: String, kind: String = "experiment:run") -> RemoteJobRecord {
        RemoteJobRecord(
            id: id, kind: kind, status: "succeeded", createdAt: 1, startedAt: nil, finishedAt: 2,
            result: ["runResult": .object([
                "evidenceBundle": .object([
                    "bundlePath": .string("/remote/runs/\(id)/\(id).evidence-bundle.tar.gz"),
                    "bundleSha256": .string("sha-\(id)"),
                ]),
                "runDirectory": .string("/remote/runs/\(id)"),
            ])],
            error: nil, logTail: [], executor: "slurm", executorJobID: nil,
            cancellationRequested: false)
    }

    /// A store whose active workspace is a connected SSH site (auto-import
    /// defaults ON for SSH transport).
    private func connectedSSHStore(_ name: String) throws -> ClusterConnectionStore {
        let store = clusterStore(defaults: try freshDefaults(name))
        var site = ClusterSiteProfile.exampleCluster
        site.transport = .ssh(host: "researcher@cluster.invalid", proxyJump: nil, remotePort: 8080, vpnExpected: false)
        let entry = store.addSite(site)
        store.activeWorkspace = .server(entry.id)
        store.capabilities = try JSONDecoder().decode(
            ClusterCapabilities.self,
            from: Data(#"{"serverVersion":"1.0","engine":"python-hf-transformers"}"#.utf8))
        return store
    }

    // MARK: - The offline switch

    @Test func launchCheckDisablesSiteConnectionAndRecordsTheAttempt() async throws {
        let store = try connectedSSHStore("no-connect")
        #expect(store.client != nil)  // the ordinary launch has a client
        await LaunchCheckMode.withOverride(true) {
            #expect(LaunchCheckMode.isActive)
            #expect(store.client == nil)
            #expect(await store.connect() == false)
            #expect(store.status == ClusterConnectionStore.launchCheckStatus)
            #expect(store.lastConnectFailure == ClusterConnectionStore.launchCheckStatus)
            #expect(LaunchCheckMode.blockedAttempts == ["cluster connect"])
            #expect(LaunchCheckMode.verdictLine.hasPrefix(LaunchCheckMode.violationPrefix))
            #expect(LaunchCheckMode.verdictLine.contains("1 network attempt"))
        }
        #expect(!LaunchCheckMode.isActive)
        LaunchCheckMode.resetBlockedAttemptsForTesting()
        #expect(LaunchCheckMode.verdictLine == LaunchCheckMode.offlineLine)
    }

    @Test func launchCheckClientRefusesEveryRequestBeforeTheNetwork() async throws {
        await LaunchCheckMode.withOverride(true) {
            let client = ClusterClient(
                profile: .init(baseURL: URL(string: "http://127.0.0.1:1")!))
            do {
                _ = try await client.capabilities()
                Issue.record("a launch-check client must not complete a request")
            } catch {
                #expect(String(describing: error).contains("launch check") || (error as? URLError)?.code == .notConnectedToInternet)
            }
            let attempts = LaunchCheckMode.blockedAttempts
            #expect(attempts.count == 1)
            #expect(attempts.first?.hasPrefix("GET http://127.0.0.1:1/api/capabilities") == true)
        }
    }

    @Test func launchCheckTunnelRunsNoSSH() async throws {
        let runner = RecordingTunnelRunner()
        let defaults = try freshDefaults("tunnel")
        await LaunchCheckMode.withOverride(true) {
            let tunnel = ClusterTunnel(runner: runner, defaults: defaults)
            var site = ClusterSiteProfile.exampleCluster
            site.transport = .ssh(host: "researcher@cluster.invalid", proxyJump: nil, remotePort: 8080, vpnExpected: false)
            tunnel.configure(site: site)
            await tunnel.open()
            #expect(tunnel.state == .degraded(ClusterConnectionStore.launchCheckStatus))
            #expect(tunnel.effectiveBaseURL == nil)
            // The injected runner was never consulted: the launch-check
            // runner replaced it at construction.
            #expect(await runner.calls.isEmpty)
            #expect(LaunchCheckMode.blockedAttempts == ["ssh tunnel open"])
        }
    }

    @Test func launchCheckDisablesEvidenceAutoImport() async throws {
        let workspace = try freshWorkspace("no-autoimport")
        defer { try? FileManager.default.removeItem(at: workspace) }
        let store = try connectedSSHStore("no-autoimport")
        let fetches = Counter()
        let imports = Counter()
        let job = succeededRunJob(id: "j1")
        await LaunchCheckMode.withOverride(true) {
            let service = EvidenceAutoImportService(
                workspaceRoot: workspace, cluster: store, originProvider: { source(workspace) },
                fetchJobs: { await fetches.increment(); return [job] },
                performImport: { _ in await imports.increment(); return workspace })
            service.startPolling()
            #expect(!service.isPolling)
            // Even a forced pass ("check now" scheduled by some surface) is
            // refused, recorded in the feed, and counted against the verdict.
            let events = await service.runOnce(force: true)
            #expect(events.count == 1)
            if case .deferred(let reason)? = events.first?.outcome {
                #expect(reason.contains("launch check"))
            } else {
                Issue.record("expected a deferred outcome, got \(String(describing: events.first?.outcome))")
            }
            #expect(await fetches.value == 0)
            #expect(await imports.value == 0)
            #expect(service.events.count == 2)  // startPolling's note + the refused pass
            #expect(service.events.allSatisfy { if case .deferred = $0.outcome { return true } else { return false } })
            #expect(LaunchCheckMode.blockedAttempts.contains("evidence auto-import pass"))
        }
        // Registration through the store takes the same path.
        await LaunchCheckMode.withOverride(true) {
            let registered = store.registerEvidenceAutoImport(workspaceRoot: workspace)
            defer { registered.stopPolling() }
            #expect(!registered.isPolling)
        }
    }

    // MARK: - Settle delay

    @Test func automaticPassesWaitForTheSettleWindow() async throws {
        let workspace = try freshWorkspace("settle")
        defer { try? FileManager.default.removeItem(at: workspace) }
        let store = try connectedSSHStore("settle")
        let clock = MutableClock(start: Date(timeIntervalSince1970: 1_800_000_000))
        let imports = Counter()
        let job = succeededRunJob(id: "j1")
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace, cluster: store, originProvider: { source(workspace) },
            fetchJobs: { [job] },
            performImport: { _ in await imports.increment(); return workspace },
            now: { clock.now })
        #expect(service.configuration.settleDelay >= .seconds(30))

        // The first automatic tick arms the window and defers, once, visibly.
        let first = await service.runOnce()
        #expect(first.count == 1)
        if case .deferred(let reason)? = first.first?.outcome {
            #expect(reason.hasPrefix("settling"))
        } else {
            Issue.record("expected a settling deferral")
        }
        #expect(await imports.value == 0)
        let deadline = try #require(service.settleDeadline)
        #expect(deadline.timeIntervalSince(clock.now) >= 30)

        // A second early tick is quiet (no feed spam) and still transfers
        // nothing.
        clock.advance(by: 10)
        #expect(await service.runOnce().isEmpty)
        #expect(await imports.value == 0)

        // A person asking imports immediately, settle window or not.
        _ = await service.importNow(job: succeededRunJob(id: "j-manual"), origin: source(workspace))
        #expect(await imports.value == 1)

        // Past the deadline the automatic pass proceeds.
        clock.advance(by: 120)
        let later = await service.runOnce()
        #expect(later.contains { if case .imported = $0.outcome { return true } else { return false } })
        #expect(await imports.value == 2)
    }

    // MARK: - Per-pass cap

    @Test func automaticPassTransfersAtMostTheCapAndDefersTheRest() async throws {
        let workspace = try freshWorkspace("cap")
        defer { try? FileManager.default.removeItem(at: workspace) }
        let jobs = (1...8).map { succeededRunJob(id: "j\($0)") }
        let imports = Counter()
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace, originProvider: { source(workspace) },
            fetchJobs: { jobs },
            performImport: { _ in await imports.increment(); return workspace })
        service.configuration.maxImportsPerPass = 3

        let first = await service.runOnce(force: true)
        #expect(await imports.value == 3)
        let deferred = first.compactMap { event -> String? in
            if case .deferred(let reason) = event.outcome { return reason } else { return nil }
        }
        #expect(deferred.count == 1)
        #expect(deferred.first?.hasPrefix("5 more bundles wait for the next pass") == true)
        #expect(service.lastSummary?.contains("imported 3") == true)
        #expect(service.lastSummary?.contains("deferred 1") == true)

        // The next pass picks up where the ledger says it stopped.
        _ = await service.runOnce(force: true)
        #expect(await imports.value == 6)
        _ = await service.runOnce(force: true)
        #expect(await imports.value == 8)
        // Nothing left: no deferral is invented.
        let idle = await service.runOnce(force: true)
        #expect(idle.isEmpty)
    }

    // MARK: - Concurrency bound

    @Test func transferGateBoundsDownloadsAndSerializesExports() async throws {
        let gate = EvidenceTransferGate(maxConcurrentDownloads: 2, maxConcurrentExports: 1)
        #expect(gate.maxConcurrentDownloads == 2)
        #expect(gate.maxConcurrentExports == 1)
        let inFlight = Counter()
        let observedPeak = Counter()
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<6 {
                group.addTask {
                    try await gate.withDownloadSlot {
                        let now = await inFlight.increment()
                        await observedPeak.raise(to: now)
                        try await Task.sleep(for: .milliseconds(20))
                        await inFlight.decrement()
                    }
                }
            }
            try await group.waitForAll()
        }
        #expect(await observedPeak.value == 2)
        #expect(await gate.peakDownloads == 2)
        #expect(await gate.activeDownloads == 0)

        let exportsInFlight = Counter()
        let exportPeak = Counter()
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<3 {
                group.addTask {
                    try await gate.withExportSlot {
                        let now = await exportsInFlight.increment()
                        await exportPeak.raise(to: now)
                        try await Task.sleep(for: .milliseconds(10))
                        await exportsInFlight.decrement()
                    }
                }
            }
            try await group.waitForAll()
        }
        #expect(await exportPeak.value == 1)
        #expect(await gate.peakExports == 1)
        #expect(await gate.activeExports == 0)
        // A throwing body still returns its slot.
        struct Boom: Error {}
        await #expect(throws: Boom.self) {
            try await gate.withDownloadSlot { throw Boom() }
        }
        #expect(await gate.activeDownloads == 0)
        #expect(EvidenceTransferGate.shared.maxConcurrentDownloads == 2)
        #expect(EvidenceTransferGate.shared.maxConcurrentExports == 1)
    }

    // MARK: - No science export from auto-import

    @Test func autoImportNeverPackagesOrExportsScienceJobs() async throws {
        let workspace = try freshWorkspace("no-export")
        defer { try? FileManager.default.removeItem(at: workspace) }
        // A succeeded science job whose result even carries a bundle path is
        // not an evidence candidate: the run-kind rule excludes it.
        let science = succeededRunJob(id: "fit-1", kind: "science:jlens-fit")
        #expect(EvidenceAutoImportService.candidate(fromJob: science) == nil)
        let packaged = Counter()
        let imports = Counter()
        let battery = succeededRunJob(id: "battery", kind: "science:battery")
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace, originProvider: { source(workspace) },
            fetchJobs: { [science, battery] },
            performImport: { _ in await imports.increment(); return workspace },
            packageEvidence: { _ in
                await packaged.increment()
                return ClusterClient.EvidencePackageReceipt()
            })
        let events = await service.runOnce(force: true)
        #expect(events.isEmpty)
        #expect(await imports.value == 0)
        // The only server-side packaging seam the service has is the
        // EXPLICIT dead-pipeline import; an automatic pass never reaches it,
        // and nothing in the service addresses /api/science/jobs/{id}/export.
        #expect(await packaged.value == 0)
    }
}

// MARK: - Helpers

private actor Counter {
    private(set) var value = 0
    @discardableResult
    func increment() -> Int { value += 1; return value }
    func decrement() { value -= 1 }
    func raise(to candidate: Int) { value = max(value, candidate) }
}

private final class MutableClock: @unchecked Sendable {
    private let lock = NSLock()
    private var current: Date
    init(start: Date) { current = start }
    var now: Date {
        lock.lock()
        defer { lock.unlock() }
        return current
    }
    func advance(by seconds: TimeInterval) {
        lock.lock()
        defer { lock.unlock() }
        current = current.addingTimeInterval(seconds)
    }
}

/// Records every command; the launch-check runner must make it unnecessary.
private actor RecordingTunnelRunner: TunnelProcessRunner {
    private(set) var calls: [[String]] = []

    func run(_ executablePath: String, arguments: [String]) async -> TunnelProcessResult {
        calls.append([executablePath] + arguments)
        return TunnelProcessResult(exitCode: 0, standardOutput: "", standardError: "")
    }

    func run(_ executablePath: String, arguments: [String], input: Data) async -> TunnelProcessResult {
        await run(executablePath, arguments: arguments)
    }

    func launch(_ executablePath: String, arguments: [String]) async throws -> any TunnelProcessHandle {
        calls.append([executablePath] + arguments)
        throw ChatServiceError(reason: "not expected in this test")
    }

    func isLocalPortFree(_ port: Int) async -> Bool { true }
}
