import Foundation
import Testing

@testable import ExperimentKit

// MARK: - Seams (no ssh, no sockets, no HTTP)

/// Scripted stand-in for the tunnel process seam: busy ports are a set the
/// test controls, `-O cancel` / `-O forward` answer from scripted exits, and
/// a successful cancel frees the port (the mux removing the listener).
private actor ControllerStubRunner: TunnelProcessRunner {
    private var busyPorts: Set<Int> = []
    private var cancelExit: Int32 = 0
    private var cancelStderr = ""
    private var cancelFreesPorts = true
    private var forwardExit: Int32 = 0
    private(set) var runCalls: [[String]] = []

    func setBusyPorts(_ ports: Set<Int>) { busyPorts = ports }
    func setForwardExit(_ code: Int32) { forwardExit = code }
    func setCancel(exit: Int32, stderr: String = "", freesPorts: Bool) {
        cancelExit = exit
        cancelStderr = stderr
        cancelFreesPorts = freesPorts
    }
    func recordedRunCalls() -> [[String]] { runCalls }
    func calls(containing needle: String) -> [[String]] {
        runCalls.filter { $0.joined(separator: " ").contains(needle) }
    }

    func run(_ executablePath: String, arguments: [String]) async -> TunnelProcessResult {
        runCalls.append([executablePath] + arguments)
        if arguments.contains("-O"), arguments.contains("cancel") {
            if cancelExit == 0, cancelFreesPorts { busyPorts = [] }
            return TunnelProcessResult(
                exitCode: cancelExit, standardOutput: "", standardError: cancelStderr)
        }
        if arguments.contains("-O"), arguments.contains("forward") {
            return TunnelProcessResult(
                exitCode: forwardExit, standardOutput: "",
                standardError: forwardExit == 0 ? "" : "forward failed")
        }
        return TunnelProcessResult(exitCode: 0, standardOutput: "", standardError: "")
    }

    func run(
        _ executablePath: String, arguments: [String], input: Data
    ) async -> TunnelProcessResult {
        await run(executablePath, arguments: arguments)
    }

    func launch(
        _ executablePath: String, arguments: [String]
    ) async throws -> any TunnelProcessHandle {
        throw CocoaError(.featureUnsupported)
    }

    func isLocalPortFree(_ port: Int) async -> Bool { !busyPorts.contains(port) }
}

private struct StubEndpointProbe: ClusterEndpointProbe {
    var result: ClusterEndpointProbeResult

    func probe(baseURL: URL, token: String?) async -> ClusterEndpointProbeResult {
        result
    }
}

private final class MemoryForwardStore: ClusterForwardRecordStoring, @unchecked Sendable {
    // @unchecked Sendable: mutated only from the serialized test body and the
    // single controller call under test; never escapes a test.
    var records: [String: ClusterTunnel.ForwardRecord] = [:]

    func record(forSite identity: String) -> ClusterTunnel.ForwardRecord? {
        records[identity]
    }
    func save(_ record: ClusterTunnel.ForwardRecord, forSite identity: String) {
        records[identity] = record
    }
    func removeRecord(forSite identity: String) {
        records.removeValue(forKey: identity)
    }
}

// MARK: - Tests

/// §7.7 forward custody, against the 2026-08-12 live failures: a controller
/// restart moved the daemon c4-11 → c4-15 and (a) the observation kept
/// reporting `up` because it only checked that the local port listens, and
/// (b) `tunnel open` adopted a listening port as "the existing forward"
/// without proving anything answered through it.
struct ClusterTunnelControllerTests {

    private static let identity = "ssh://login.test:8080"

    private func site() -> ClusterSiteProfile {
        ClusterSiteProfile(
            name: "Test Cluster",
            transport: .ssh(
                host: "login.test", proxyJump: nil, remotePort: 8080, vpnExpected: false),
            topology: .daemonInJob,
            scheduler: .slurm(
                ClusterSiteProfile.SlurmSiteData(
                    partitions: [.init(name: "batch", maxWalltimeHours: 168)],
                    defaultPartition: "batch")))
    }

    private func oldForward(localPort: Int = 8718) -> ClusterTunnel.ForwardRecord {
        ClusterTunnel.ForwardRecord(
            host: "login.test", proxyJump: nil, localPort: localPort,
            targetHost: "c4-11", remotePort: 8080)
    }

    private func makeController(
        runner: ControllerStubRunner,
        probe: ClusterEndpointProbeResult = ClusterEndpointProbeResult(
            reachable: true, serverBuild: "steerlab-server 0.1.0"),
        store: MemoryForwardStore
    ) -> SSHClusterTunnelController {
        SSHClusterTunnelController(
            runner: runner, endpoint: StubEndpointProbe(result: probe),
            forwards: store)
    }

    // MARK: Observation — target drift is stale, not up

    @Test func aForwardTargetingTheOldNodeReadsStaleAgainstTheCurrentDaemonHost()
        async
    {
        let runner = ControllerStubRunner()
        await runner.setBusyPorts([8718])
        let store = MemoryForwardStore()
        store.records[Self.identity] = oldForward()
        let controller = makeController(runner: runner, store: store)

        // The live bug: the port listens, so the old code said `up` while the
        // forward still pointed at the node the controller left.
        let observation = await controller.observe(
            site: site(), persistedPort: 8718, targetHost: "c4-15")
        #expect(
            observation
                == .stale(
                    localPort: 8718,
                    reason: "forward targets c4-11 but the controller now runs on c4-15"))

        // The same forward against the node it actually targets is up…
        #expect(
            await controller.observe(
                site: site(), persistedPort: 8718, targetHost: "c4-11")
                == .up(localPort: 8718))
        // …and with no daemon host known there is nothing to indict it with.
        #expect(
            await controller.observe(
                site: site(), persistedPort: 8718, targetHost: nil)
                == .up(localPort: 8718))
    }

    // MARK: Adoption — identity through the forward, or no adoption

    @Test func adoptionRefusesAListenerThatFailsTheIdentityProbe() async {
        let runner = ControllerStubRunner()
        await runner.setBusyPorts([8718])
        // No SteerLab record names the listener, and nothing answers through
        // it: the half-cancelled-mux shape. Claiming "ready" here was bug 2.
        let controller = makeController(
            runner: runner,
            probe: ClusterEndpointProbeResult(reachable: false, detail: "no answer"),
            store: MemoryForwardStore())

        let outcome = await controller.open(
            site: site(), targetHost: "c4-15", persistedPort: 8718)

        guard case .conflicted(let port, let reason) = outcome.observation else {
            Issue.record("expected .conflicted, got \(outcome.observation)")
            return
        }
        #expect(port == 8718)
        #expect(reason.contains("identity probe"))
        #expect(!outcome.changed)
        // Nothing was installed on top of the unowned listener, and nothing
        // was cancelled that is not ours.
        #expect(await runner.calls(containing: "-O forward").isEmpty)
        #expect(await runner.calls(containing: "-O cancel").isEmpty)
    }

    @Test func anIdentityVerifiedForwardIsAdoptedAsANoOp() async {
        let runner = ControllerStubRunner()
        await runner.setBusyPorts([8718])
        let store = MemoryForwardStore()
        store.records[Self.identity] = ClusterTunnel.ForwardRecord(
            host: "login.test", proxyJump: nil, localPort: 8718,
            targetHost: "c4-15", remotePort: 8080)
        // A 401 is the server ANSWERING: identity proven, token missing —
        // the registration step's problem, not the tunnel's.
        let controller = makeController(
            runner: runner,
            probe: ClusterEndpointProbeResult(reachable: false, authFailed: true),
            store: store)

        let outcome = await controller.open(
            site: site(), targetHost: "c4-15", persistedPort: 8718)

        #expect(outcome.observation == .up(localPort: 8718))
        #expect(!outcome.changed)
        #expect(await runner.calls(containing: "-O forward").isEmpty)
        #expect(await runner.calls(containing: "-O cancel").isEmpty)
    }

    @Test func staleTargetRepairCancelsTheExactRecordedSpecThenReinstalls() async {
        let runner = ControllerStubRunner()
        await runner.setBusyPorts([8718])
        await runner.setCancel(exit: 0, freesPorts: true)
        let store = MemoryForwardStore()
        store.records[Self.identity] = oldForward()
        let controller = makeController(runner: runner, store: store)

        let outcome = await controller.open(
            site: site(), targetHost: "c4-15", persistedPort: 8718)

        #expect(outcome.observation == .up(localPort: 8718))
        #expect(outcome.changed)
        // The cancel used the spec the forward was CREATED with — old target,
        // same bind-address form — because that string is the only one the
        // ControlMaster will match.
        let cancels = await runner.calls(containing: "cancel")
        #expect(cancels.count == 1)
        #expect(cancels.first?.contains("127.0.0.1:8718:c4-11:8080") == true)
        // The reinstall targets the node the controller runs on NOW, and the
        // store remembers it for the next drift.
        let forwards = await runner.calls(containing: "forward")
        #expect(forwards.first?.contains("127.0.0.1:8718:c4-15:8080") == true)
        #expect(store.records[Self.identity]?.targetHost == "c4-15")
    }

    // MARK: Cancel — a mismatched spec is an error, never success

    @Test func aCancelThatLeavesTheListenerIsAnErrorNotSuccess() async {
        let runner = ControllerStubRunner()
        await runner.setBusyPorts([8718])
        // The observed live shape: `-O cancel` answers "port not forwarded"
        // (the spec did not match what the mux holds) and the listener stays.
        await runner.setCancel(
            exit: 255, stderr: "port not forwarded: 8718", freesPorts: false)
        let store = MemoryForwardStore()
        store.records[Self.identity] = oldForward()
        let controller = makeController(runner: runner, store: store)

        let error = await controller.close(
            site: site(), localPort: 8718, targetHost: "c4-15")
        #expect(error != nil)
        #expect(error?.contains("port not forwarded") == true)
        // The record survives: the forward is still out there to cancel.
        #expect(store.records[Self.identity] != nil)

        // The same failure inside open's repair path surfaces as conflicted,
        // not as a fresh forward stacked on the stuck one.
        let outcome = await controller.open(
            site: site(), targetHost: "c4-15", persistedPort: 8718)
        guard case .conflicted = outcome.observation else {
            Issue.record("expected .conflicted, got \(outcome.observation)")
            return
        }
        #expect(outcome.message.contains("port not forwarded"))
        #expect(await runner.calls(containing: "-O forward").isEmpty)
    }

    @Test func aCancelThatSucceedsButLeavesAListenerIsStillAnError() async {
        let runner = ControllerStubRunner()
        await runner.setBusyPorts([8718])
        // The double-accepted same-port forward: cancelling one spec exits 0
        // while the mux keeps the listener for the other.
        await runner.setCancel(exit: 0, freesPorts: false)
        let store = MemoryForwardStore()
        store.records[Self.identity] = oldForward()
        let controller = makeController(runner: runner, store: store)

        let error = await controller.close(
            site: site(), localPort: 8718, targetHost: "c4-15")
        #expect(error?.contains("still listens") == true)
    }

    @Test func aVerifiedCloseRemovesTheRecordAndReportsSuccess() async {
        let runner = ControllerStubRunner()
        await runner.setBusyPorts([8718])
        await runner.setCancel(exit: 0, freesPorts: true)
        let store = MemoryForwardStore()
        store.records[Self.identity] = oldForward()
        let controller = makeController(runner: runner, store: store)

        let error = await controller.close(
            site: site(), localPort: 8718, targetHost: "c4-15")
        #expect(error == nil)
        #expect(store.records[Self.identity] == nil)
        // And the cancel named the recorded spec, not one rebuilt from the
        // CURRENT target (which would be c4-15 and fail to match).
        let cancels = await runner.calls(containing: "cancel")
        #expect(cancels.first?.contains("127.0.0.1:8718:c4-11:8080") == true)
    }
}
