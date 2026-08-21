import Foundation
import Testing

@testable import ExperimentKit

/// Pure-CPU tests for the same-machine workspace auto-switch coordinator
/// (`ClusterConnectionStore.synchronizeServerToLocalWorkspace`) — review
/// finding F4. No networking: connect/switch/root are injected through the
/// coordinator's test seams; the affordance verdict, serialization, stale
/// guards, refusal surfacing, and the single-refresh contract are asserted
/// against scripted stores.
@MainActor
struct WorkspaceAutoSwitchTests {

    // MARK: Scaffolding

    /// A named, wiped defaults suite so tests never touch the real prefs.
    private func freshDefaults(_ name: String) throws -> UserDefaults {
        let suite = "steerlab.tests.workspace-autoswitch.\(name)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    /// Notices feed persisted to a throwaway file, never the live workspace.
    private func freshNotices(_ name: String) -> PanelNotices {
        PanelNotices(
            fileURL: FileManager.default.temporaryDirectory.appending(
                components: "steerlab-autoswitch-tests",
                "\(name)-\(UUID().uuidString).json"))
    }

    /// Capabilities advertising the runtime workspace-switch route + a
    /// permissive live policy verdict (the switch-capable server).
    private func switchCapableCapabilities() throws -> ClusterCapabilities {
        try JSONDecoder().decode(
            ClusterCapabilities.self,
            from: Data(#"{"workspace":{"switch":true,"switchable":true}}"#.utf8))
    }

    /// A store whose active workspace is a SAME-MACHINE server (direct
    /// transport to loopback), connected (switch-capable capabilities) and
    /// serving `servedRoot` — a confirmed mismatch against any other target.
    private func sameMachineStore(
        _ name: String, servedRoot: String? = "/served/elsewhere",
        connected: Bool = true
    ) throws -> ClusterConnectionStore {
        let store = ClusterConnectionStore(defaults: try freshDefaults(name))
        store.notices = freshNotices(name)
        let entry = store.addServer(
            name: "loopback", urlString: "http://127.0.0.1:8080")
        store.activeWorkspace = .server(entry.id)  // didSet clears state
        if connected {
            store.capabilities = try switchCapableCapabilities()
        }
        if let servedRoot {
            store.remoteInfo = RemoteServerInfo(root: servedRoot)
        }
        return store
    }

    /// MainActor-confined probe the injected seams write into.
    @MainActor
    final class SyncProbe {
        var switchCalls: [String] = []
        var connectCalls = 0
        var refreshes = 0
        var switchesInFlight = 0
        var maxSwitchesInFlight = 0
    }

    /// One-shot async gate: `wait()` suspends until `open()` — lets a test
    /// hold the coordinator at a chosen suspension point.
    @MainActor
    final class Gate {
        private var opened = false
        private var waiters: [CheckedContinuation<Void, Never>] = []
        func open() {
            opened = true
            let resumed = waiters
            waiters = []
            for waiter in resumed { waiter.resume() }
        }
        func wait() async {
            if opened { return }
            await withCheckedContinuation { waiters.append($0) }
        }
    }

    // MARK: Serialization (acceptance test #4: rapid A→B lands on B)

    @Test func rapidRetargetIsCancelAndReplaceAndLandsOnTheNewestRoot() async throws {
        let store = try sameMachineStore("rapid-retarget")
        let probe = SyncProbe()
        let entered = Gate()
        let release = Gate()
        let root = Box<String?>("/ws/A")
        store.workspaceSyncLocalRootOverride = { root.value }
        store.workspaceSyncSwitchOverride = { target in
            probe.switchCalls.append(target)
            probe.switchesInFlight += 1
            probe.maxSwitchesInFlight = max(
                probe.maxSwitchesInFlight, probe.switchesInFlight)
            if probe.switchCalls.count == 1 {
                entered.open()
                await release.wait()  // hold attempt 1 mid-switch
            }
            probe.switchesInFlight -= 1
            return true
        }
        store.workspaceSyncArtifactRefresh = { probe.refreshes += 1 }

        let first = store.synchronizeServerToLocalWorkspace()
        await entered.wait()  // attempt 1 is inside its switch call
        root.value = "/ws/B"  // the user opened workspace B
        let second = store.synchronizeServerToLocalWorkspace()
        #expect(first.isCancelled)  // cancel-and-replace
        release.open()
        await first.value
        await second.value

        // Attempt 2 queued strictly behind attempt 1 (never interleaved) and
        // the server ends on the NEWEST root.
        #expect(probe.switchCalls == ["/ws/A", "/ws/B"])
        #expect(probe.maxSwitchesInFlight == 1)
        // Only the surviving attempt refreshes — the superseded one is
        // stale-guarded out of both the refresh and the notices feed.
        #expect(probe.refreshes == 1)
        #expect(store.notices.notices.isEmpty)
    }

    @Test func staleGuardBlocksSwitchWhenRootMovesDuringConnectProbe() async throws {
        // Disconnected server: the coordinator must connect first, and after
        // that await it must re-check the CURRENT root against the captured
        // target — a moved root means a newer owner, so no switch fires from
        // the stale attempt even without an explicit cancellation.
        let store = try sameMachineStore(
            "stale-guard", servedRoot: nil, connected: false)
        let probe = SyncProbe()
        let entered = Gate()
        let release = Gate()
        let root = Box<String?>("/ws/A")
        store.workspaceSyncLocalRootOverride = { root.value }
        store.workspaceSyncConnectOverride = { [weak store] in
            probe.connectCalls += 1
            entered.open()
            await release.wait()
            store?.capabilities = try? self.switchCapableCapabilities()
            store?.remoteInfo = RemoteServerInfo(root: "/served/elsewhere")
            return true
        }
        store.workspaceSyncSwitchOverride = { target in
            probe.switchCalls.append(target)
            return true
        }
        store.workspaceSyncArtifactRefresh = { probe.refreshes += 1 }

        let attempt = store.synchronizeServerToLocalWorkspace()
        await entered.wait()  // held inside the connect probe
        root.value = "/ws/B"  // root moved while connecting
        release.open()
        await attempt.value

        #expect(probe.connectCalls == 1)
        #expect(probe.switchCalls.isEmpty)  // stale attempt never switches
        #expect(probe.refreshes == 0)
        #expect(store.notices.notices.isEmpty)  // staleness is not a failure
    }

    // MARK: Refusal / failure surfacing

    @Test func refusedSwitchSurfacesStatusVerbatimAsWarningNotice() async throws {
        let store = try sameMachineStore("refusal")
        let probe = SyncProbe()
        let refusal = "workspace switch refused (409): 1 job(s) still running"
        store.workspaceSyncSwitchOverride = { [weak store] target in
            probe.switchCalls.append(target)
            store?.status = refusal  // what switchServerWorkspace does
            return false
        }
        store.workspaceSyncArtifactRefresh = { probe.refreshes += 1 }

        await store.synchronizeServerToLocalWorkspace().value

        #expect(probe.switchCalls.count == 1)
        #expect(probe.refreshes == 0)  // a refused switch never refreshes
        #expect(store.status == refusal)  // status path intact
        let notice = try #require(store.notices.notices.last)
        #expect(notice.severity == .warning)
        #expect(notice.message == refusal)  // verbatim, not paraphrased
        #expect(notice.source == "Workspace")
    }

    @Test func failedConnectProbeSurfacesWarningAndNeverSwitches() async throws {
        let store = try sameMachineStore(
            "connect-failure", servedRoot: nil, connected: false)
        let probe = SyncProbe()
        let failure = "connection failed: nothing is listening at 127.0.0.1:8080"
        store.workspaceSyncConnectOverride = { [weak store] in
            probe.connectCalls += 1
            store?.status = failure
            return false
        }
        store.workspaceSyncSwitchOverride = { target in
            probe.switchCalls.append(target)
            return true
        }

        await store.synchronizeServerToLocalWorkspace().value

        #expect(probe.connectCalls == 1)
        #expect(probe.switchCalls.isEmpty)
        let notice = try #require(store.notices.notices.last)
        #expect(notice.severity == .warning)
        #expect(notice.message == failure)
    }

    // MARK: No-ops

    @Test func alreadyServingTargetRefreshesOnceThenNoOps() async throws {
        // Reconnecting to an already-correct workspace must not leave the
        // server artifact catalogs stale: the FIRST landing on a
        // (server, root) pairing fires the registered refresh exactly once
        // — with no switch, no connect probe, no notice — and repeated
        // root-change events on the SAME pairing are pure no-ops.
        let store = try sameMachineStore("already-serving", servedRoot: nil)
        let probe = SyncProbe()
        // Pin the local root via the injectable seam: the fallback resolver
        // reads the process-global workspace root, which PARALLEL test
        // suites legitimately override (the ExperimentRootOverrideLock
        // serializes those suites, not this one) — an uninjected read here
        // races them and captures a foreign temp root.
        let target = "/ws/already-serving"
        store.workspaceSyncLocalRootOverride = { target }
        // Serve exactly the root the coordinator would target (the pairing
        // check's realpath'd form — never a re-derived variant).
        store.remoteInfo = RemoteServerInfo(root: target)
        store.workspaceSyncConnectOverride = {
            probe.connectCalls += 1
            return true
        }
        store.workspaceSyncSwitchOverride = { root in
            probe.switchCalls.append(root)
            return true
        }
        store.workspaceSyncArtifactRefresh = { probe.refreshes += 1 }

        await store.synchronizeServerToLocalWorkspace().value

        #expect(probe.connectCalls == 0)  // already connected — no probe
        #expect(probe.switchCalls.isEmpty)  // already serving — no switch
        #expect(probe.refreshes == 1)  // …but the catalogs load once
        #expect(store.notices.notices.isEmpty)

        // Repeated events for the same pairing: no churn.
        await store.synchronizeServerToLocalWorkspace().value
        await store.synchronizeServerToLocalWorkspace().value
        #expect(probe.refreshes == 1)
        #expect(probe.switchCalls.isEmpty)
        #expect(store.notices.notices.isEmpty)
    }

    @Test func switchPathRecordsThePairingSoAFollowUpEventDoesNotDoubleRefresh() async throws {
        // A successful auto-switch already refreshed for its pairing —
        // a subsequent root-change event landing on the (now correct)
        // same pairing must not refresh a second time.
        let store = try sameMachineStore("switch-then-event")
        let probe = SyncProbe()
        let target = "/ws/switch-then-event"
        store.workspaceSyncLocalRootOverride = { target }
        store.workspaceSyncSwitchOverride = { [weak store] root in
            probe.switchCalls.append(root)
            store?.remoteInfo = RemoteServerInfo(root: root)  // now serving it
            return true
        }
        store.workspaceSyncArtifactRefresh = { probe.refreshes += 1 }

        await store.synchronizeServerToLocalWorkspace().value
        #expect(probe.switchCalls == [target])
        #expect(probe.refreshes == 1)

        // Same pairing again, this time via the already-serving path.
        await store.synchronizeServerToLocalWorkspace().value
        #expect(probe.switchCalls == [target])  // no second switch
        #expect(probe.refreshes == 1)  // and no second refresh
    }

    @Test func localWorkspaceIsANoOp() async throws {
        let store = ClusterConnectionStore(defaults: try freshDefaults("local"))
        store.notices = freshNotices("local")
        let probe = SyncProbe()
        store.workspaceSyncSwitchOverride = { root in
            probe.switchCalls.append(root)
            return true
        }
        await store.synchronizeServerToLocalWorkspace().value
        #expect(probe.switchCalls.isEmpty)
        #expect(store.notices.notices.isEmpty)
    }

    // MARK: Affordance gating (remote servers stay explicit)

    @Test func sshServerNeverAutoSwitches() async throws {
        let store = ClusterConnectionStore(defaults: try freshDefaults("ssh"))
        store.notices = freshNotices("ssh")
        let entry = store.addSite(
            ClusterSiteProfile(
                name: "Cluster",
                transport: .ssh(
                    host: "login.cluster.test", proxyJump: nil, remotePort: 8080,
                    vpnExpected: false),
                topology: .loginDaemon,
                scheduler: .slurm(ClusterSiteProfile.SlurmSiteData()),
                constraints: ClusterSiteProfile.SiteConstraints()))
        store.activeWorkspace = .server(entry.id)
        // Even fully connected, switch-capable, and confirmed-mismatched, an
        // SSH site's tunnel-local URL is NOT this Mac's filesystem.
        store.capabilities = try switchCapableCapabilities()
        store.remoteInfo = RemoteServerInfo(root: "/scratch/elsewhere")
        let probe = SyncProbe()
        store.workspaceSyncConnectOverride = {
            probe.connectCalls += 1
            return true
        }
        store.workspaceSyncSwitchOverride = { root in
            probe.switchCalls.append(root)
            return true
        }

        await store.synchronizeServerToLocalWorkspace().value

        #expect(probe.connectCalls == 0)
        #expect(probe.switchCalls.isEmpty)
        #expect(store.notices.notices.isEmpty)  // hidden, not a warning
    }

    @Test func directRemoteHostNeverAutoSwitches() async throws {
        let store = ClusterConnectionStore(
            defaults: try freshDefaults("direct-remote"))
        store.notices = freshNotices("direct-remote")
        let entry = store.addServer(name: "gpu", urlString: "http://gpu-node:8080")
        store.activeWorkspace = .server(entry.id)
        store.capabilities = try switchCapableCapabilities()
        store.remoteInfo = RemoteServerInfo(root: "/data/elsewhere")
        let probe = SyncProbe()
        store.workspaceSyncSwitchOverride = { root in
            probe.switchCalls.append(root)
            return true
        }
        await store.synchronizeServerToLocalWorkspace().value
        #expect(probe.switchCalls.isEmpty)
        #expect(store.notices.notices.isEmpty)
    }

    @Test func switchIncapableServerStaysSilent() async throws {
        // Older server without the workspace.switch capability: affordances
        // (and their automation) hide rather than relay a guaranteed refusal.
        let store = try sameMachineStore("incapable", connected: false)
        store.capabilities = try JSONDecoder().decode(
            ClusterCapabilities.self, from: Data("{}".utf8))
        store.remoteInfo = RemoteServerInfo(root: "/served/elsewhere")
        let probe = SyncProbe()
        store.workspaceSyncSwitchOverride = { root in
            probe.switchCalls.append(root)
            return true
        }
        await store.synchronizeServerToLocalWorkspace().value
        #expect(probe.switchCalls.isEmpty)
        #expect(store.notices.notices.isEmpty)
    }

    // MARK: Single-refresh contract

    @Test func connectedStoreSwitchesWithZeroExtraConnectsAndOneRefresh() async throws {
        let store = try sameMachineStore("single-refresh")
        let probe = SyncProbe()
        // Injected root: see alreadyServingTargetIsASilentNoOp for why the
        // global-resolver fallback must not be read in parallel test runs.
        let target = "/ws/single-refresh"
        store.workspaceSyncLocalRootOverride = { target }
        store.workspaceSyncConnectOverride = {
            probe.connectCalls += 1
            return true
        }
        store.workspaceSyncSwitchOverride = { root in
            probe.switchCalls.append(root)
            return true
        }
        store.workspaceSyncArtifactRefresh = { probe.refreshes += 1 }

        await store.synchronizeServerToLocalWorkspace().value

        // The switch itself reconnects internally; the coordinator adds NO
        // connect of its own and ends with exactly one artifact refresh.
        #expect(probe.connectCalls == 0)
        #expect(probe.switchCalls == [target])
        #expect(probe.refreshes == 1)
    }

    @Test func disconnectedStoreProbesConnectOnceThenSwitches() async throws {
        let store = try sameMachineStore(
            "probe-then-switch", servedRoot: nil, connected: false)
        let probe = SyncProbe()
        // Injected root: see alreadyServingTargetIsASilentNoOp for why the
        // global-resolver fallback must not be read in parallel test runs.
        let target = "/ws/probe-then-switch"
        store.workspaceSyncLocalRootOverride = { target }
        store.workspaceSyncConnectOverride = { [weak store] in
            probe.connectCalls += 1
            store?.capabilities = try? self.switchCapableCapabilities()
            store?.remoteInfo = RemoteServerInfo(root: "/served/elsewhere")
            return true
        }
        store.workspaceSyncSwitchOverride = { root in
            probe.switchCalls.append(root)
            return true
        }
        store.workspaceSyncArtifactRefresh = { probe.refreshes += 1 }

        await store.synchronizeServerToLocalWorkspace().value

        #expect(probe.connectCalls == 1)  // the precondition probe, once
        #expect(probe.switchCalls == [target])
        #expect(probe.refreshes == 1)
    }
}

/// Tiny MainActor-confined mutable box for the injected root provider.
@MainActor
final class Box<Value> {
    var value: Value
    init(_ value: Value) { self.value = value }
}
