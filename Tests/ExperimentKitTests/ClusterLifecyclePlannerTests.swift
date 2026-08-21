import Foundation
import Testing

@testable import ExperimentKit

/// The pure planner (Phase B of `docs/CLUSTER-CLI-LIFECYCLE-PLAN.md`, §7.3).
///
/// One function, no I/O — so every partial observed state can be walked to
/// `connected` in microseconds, and the doctrines that matter are asserted
/// rather than described:
///
///   * nothing mutating is planned as actionable without ITS OWN permission;
///   * a scheduler probe that FAILED is never a licence to start a second
///     controller;
///   * a queued controller is `pending`, never a failure;
///   * a stale `serverd.host` never satisfies the controller rung.
struct ClusterLifecyclePlannerTests {

    // MARK: Fixtures

    private func sshSite(
        topology: ClusterSiteProfile.Topology = .daemonInJob
    ) -> ClusterSiteProfile {
        ClusterSiteProfile(
            name: "Test Cluster",
            transport: .ssh(
                host: "login.test", proxyJump: nil, remotePort: 8080, vpnExpected: false),
            topology: topology,
            scheduler: .slurm(ClusterSiteProfile.SlurmSiteData()),
            constraints: ClusterSiteProfile.SiteConstraints(
                storageRoots: ["workspace": "/scratch/me/ws", "hfCache": "/work/lab/hf"]))
    }

    /// A fully connected observation — the "nothing to do" baseline every
    /// partial-state test degrades one layer of.
    private func connectedState(
        site: ClusterSiteProfile, localPort: Int = 8712
    ) -> ClusterObservedState {
        ClusterObservedState(
            siteID: "test", siteName: "Test Cluster",
            siteConfiguration: .valid,
            controlMaster: .alive,
            payload: .current(deploymentHash: "abc"),
            bootstrap: .valid(envFile: "~/steerlab-cluster.env", prefix: "/home/me/envs/s"),
            profileValidation: .pass,
            controller: .running(jobID: "4242"),
            daemonHost: .current(host: "c4-12"),
            tunnel: .up(localPort: localPort),
            serverHTTP: .reachable(build: "steerlab-server 0.1.0", role: "controller", root: "/scratch"),
            registration: .present(endpoint: "http://127.0.0.1:\(localPort)"),
            bearerToken: .accepted)
    }

    private func plan(
        _ observed: ClusterObservedState,
        site: ClusterSiteProfile? = nil,
        target: ClusterLifecycleTarget = .connected,
        permissions: ClusterLifecyclePermissions = .allMutations
    ) -> ClusterLifecyclePlan {
        ClusterLifecyclePlanner.plan(
            site: site ?? sshSite(), observed: observed, target: target,
            permissions: permissions)
    }

    // MARK: Fully satisfied

    @Test func aFullyConnectedSiteHasNothingLeftToDo() {
        let site = sshSite()
        let result = plan(connectedState(site: site))
        #expect(result.isSatisfied)
        #expect(result.remaining.isEmpty)
        #expect(result.next == nil)
        #expect(result.state(permissions: []) == .ready)
        // Even with NO permissions: a satisfied ladder never needs approval.
        let allSatisfied = result.transitions.allSatisfy(\.isSatisfied)
        #expect(allSatisfied)
    }

    @Test func aLowerTargetIgnoresTheRungsAboveIt() {
        let site = sshSite()
        var observed = connectedState(site: site)
        observed.controller = .absent
        observed.tunnel = .absent
        observed.registration = .absent
        // Asking only for `validated` must not plan a controller or a tunnel.
        let result = plan(observed, target: .validated)
        #expect(result.isSatisfied)
        #expect(!result.transitions.contains { $0.step == .controllerStart })
        #expect(!result.transitions.contains { $0.step == .tunnelOpen })
        // …while asking for `connected` does.
        let full = plan(observed, target: .connected)
        #expect(full.remaining.map(\.step).contains(.controllerStart))
    }

    // MARK: Every partial state walks to connected

    /// Each layer, degraded one at a time from the connected baseline: the
    /// planner must name a transition for it, and that transition must be the
    /// NEXT one (the lowest broken rung comes first).
    @Test func everyDegradedLayerYieldsItsOwnNextTransition() {
        let site = sshSite()
        let cases: [(String, (inout ClusterObservedState) -> Void, ClusterLifecycleStep)] = [
            ("controlMaster", { $0.controlMaster = .absent }, .authenticate),
            ("payload", { $0.payload = .absent }, .pushCode),
            ("bootstrap", { $0.bootstrap = .absent }, .bootstrapPlan),
            ("validation", { $0.profileValidation = .notRun }, .validate),
            (
                "controller",
                {
                    $0.controller = .absent
                    $0.daemonHost = .absent
                }, .controllerStart
            ),
            (
                "tunnel",
                {
                    $0.tunnel = .absent
                    $0.serverHTTP = .unreachable(reason: "no endpoint")
                    $0.registration = .absent
                }, .tunnelOpen
            ),
            ("registration", { $0.registration = .absent }, .registerConnection),
        ]
        for (label, degrade, expected) in cases {
            var observed = connectedState(site: site)
            degrade(&observed)
            let result = plan(observed)
            #expect(result.next?.step == expected, "\(label) should plan \(expected)")
            #expect(result.next?.reason.isEmpty == false, "\(label) transition must say WHY")
        }
    }

    @Test func aColdSiteWalksTheWholeLadderInOrder() {
        let site = sshSite()
        let cold = ClusterObservedState(
            siteID: "test", siteName: "Test Cluster",
            controlMaster: .absent,
            payload: .unknown(reason: "no SSH ControlMaster — authenticate first"),
            bootstrap: .unknown(reason: "no SSH ControlMaster — authenticate first"),
            profileValidation: .notRun,
            controller: .absent,
            daemonHost: .absent,
            tunnel: .absent,
            serverHTTP: .unreachable(reason: "no endpoint to probe yet"),
            registration: .absent,
            bearerToken: .absent)
        let result = plan(cold)
        #expect(
            result.remaining.map(\.step) == [
                .authenticate, .pushCode, .bootstrapPlan, .bootstrapApply, .validate,
                .controllerStart, .tunnelOpen, .registerConnection,
            ])
        // The first boundary is the HUMAN one, and it is reported as such.
        #expect(result.state(permissions: .allMutations) == .needsHumanAuthentication)
        #expect(result.next?.gating.requiresHumanAuthentication == true)
    }

    // MARK: Permission gating — one at a time

    @Test func noMutationIsPlannedAsPermittedWithoutItsOwnPermission() {
        let site = sshSite()
        // push
        var needsPush = connectedState(site: site)
        needsPush.payload = .stale(reason: "the deployed payload does not match")
        let pushPlan = plan(needsPush, permissions: [])
        #expect(pushPlan.next?.step == .pushCode)
        #expect(pushPlan.next?.gating.requiredPermission == .push)
        #expect(pushPlan.state(permissions: []) == .needsApproval)
        // The OTHER two permissions do not unlock it.
        #expect(pushPlan.state(permissions: [.bootstrap, .controllerStart]) == .needsApproval)
        #expect(pushPlan.state(permissions: [.push]) == .planned)

        // bootstrap
        var needsBootstrap = connectedState(site: site)
        needsBootstrap.bootstrap = .absent
        let bootstrapPlan = plan(needsBootstrap, permissions: [])
        // The dry run is read-only and needs NO permission — reviewing a plan
        // must never be gated behind permission to execute it.
        #expect(bootstrapPlan.next?.step == .bootstrapPlan)
        #expect(bootstrapPlan.next?.gating.requiredPermission == nil)
        #expect(bootstrapPlan.state(permissions: []) == .planned)
        let apply = bootstrapPlan.transitions.first { $0.step == .bootstrapApply }
        #expect(apply?.gating.requiredPermission == .bootstrap)
        #expect(apply?.gating.isAsynchronous == true)
        #expect(apply?.gating.isResourceConsuming == true)
        #expect(apply?.isPermitted(by: [.push, .controllerStart]) == false)
        #expect(apply?.isPermitted(by: [.bootstrap]) == true)

        // controller start
        var needsController = connectedState(site: site)
        needsController.controller = .absent
        needsController.daemonHost = .absent
        let controllerPlan = plan(needsController, permissions: [])
        #expect(controllerPlan.next?.step == .controllerStart)
        #expect(controllerPlan.next?.gating.requiredPermission == .controllerStart)
        #expect(controllerPlan.next?.gating.isResourceConsuming == true)
        #expect(controllerPlan.state(permissions: [.push, .bootstrap]) == .needsApproval)
        #expect(controllerPlan.state(permissions: [.controllerStart]) == .planned)
    }

    @Test func openingTheAuthTerminalIsNotAMutationPermission() {
        let site = sshSite()
        var observed = connectedState(site: site)
        observed.controlMaster = .absent
        let result = plan(observed, permissions: .allMutations)
        // Every mutation permission granted, and it STILL stops at the human.
        #expect(result.state(permissions: .allMutations) == .needsHumanAuthentication)
        #expect(result.next?.gating.requiredPermission == nil)
    }

    // MARK: Controller reconciliation

    @Test func aQueuedControllerIsPendingAndNeverAFailure() {
        let site = sshSite()
        var observed = connectedState(site: site)
        observed.controller = .pending(jobID: "47207294", reason: "Resources")
        observed.daemonHost = .absent
        let result = plan(observed)
        #expect(result.next?.step == .controllerWait)
        #expect(result.next?.gating.isAsynchronous == true)
        #expect(result.next?.gating.requiredPermission == nil)
        #expect(result.state(permissions: .allMutations) == .pending)
        // Never a start: the job already exists.
        #expect(!result.remaining.map(\.step).contains(.controllerStart))
        #expect(result.next?.reason.contains("wait, not a failure") == true)
    }

    @Test func aMaintenanceReservationIsNamedInThePendingReason() {
        let site = sshSite()
        var observed = connectedState(site: site)
        observed.controller = .pending(jobID: "9", reason: "ReqNodeNotAvail, Reserved for maintenance")
        observed.daemonHost = .absent
        let result = plan(observed)
        #expect(result.next?.reason.contains("maintenance reservation") == true)
        #expect(result.state(permissions: .allMutations) == .pending)
    }

    @Test func anUnprovenSchedulerStateNeverLicensesASecondController() {
        let site = sshSite()
        var observed = connectedState(site: site)
        observed.controller = .unknown(reason: "the scheduler probe failed")
        observed.daemonHost = .stale(host: "c4-12", reason: "unconfirmed")
        let result = plan(observed, permissions: .allMutations)
        // THE doctrine: a failed QUERY is not a dead job.
        #expect(result.next?.step == .controllerWait)
        #expect(!result.remaining.map(\.step).contains(.controllerStart))
        // And it is DEGRADED (retryable), not PENDING (which would imply the
        // scheduler told us something).
        #expect(result.state(permissions: .allMutations) == .degraded)
    }

    @Test func aFailedOrAbsentControllerIsARestartAndSaysWhy() {
        let site = sshSite()
        for (controller, fragment) in [
            (ClusterControllerObservation.absent, "no controller job"),
            (.failed(jobID: "77", state: "TIMEOUT"), "TIMEOUT"),
        ] {
            var observed = connectedState(site: site)
            observed.controller = controller
            observed.daemonHost = .absent
            let result = plan(observed)
            #expect(result.next?.step == .controllerStart)
            #expect(result.next?.reason.contains(fragment) == true)
        }
    }

    @Test func aStaleHostFileNeverSatisfiesTheControllerRung() {
        let site = sshSite()
        var observed = connectedState(site: site)
        // The job IS running, but the published node record is not current —
        // forwarding to it would be forwarding to a node from a dead job.
        observed.controller = .running(jobID: "4242")
        observed.daemonHost = .stale(host: "c4-12", reason: "written by an earlier job")
        let result = plan(observed)
        #expect(result.next?.step == .controllerWait)
        #expect(result.state(permissions: .allMutations) == .pending)
        #expect(!result.remaining.map(\.step).contains(.controllerStart))
    }

    @Test func aRunningControllerWithACurrentHostRecordIsSatisfied() {
        let site = sshSite()
        var observed = connectedState(site: site)
        observed.controller = .running(jobID: "4242")
        observed.daemonHost = .current(host: "c4-12")
        let result = plan(observed)
        #expect(result.isSatisfied)
    }

    // MARK: Endpoint identity

    @Test func anOpenPortIsNotAConnection() {
        let site = sshSite()
        var observed = connectedState(site: site)
        // Forward up, but nothing recognizable behind it.
        observed.serverHTTP = .unreachable(reason: "no answer")
        let result = plan(observed)
        #expect(result.next?.step == .tunnelOpen)
        #expect(result.next?.reason.contains("does not answer behind it") == true)
    }

    @Test func aRejectedTokenBlocksRegistrationAndSaysSo() {
        let site = sshSite()
        var observed = connectedState(site: site)
        observed.bearerToken = .rejected
        observed.serverHTTP = .authFailed
        let result = plan(observed)
        #expect(result.next?.step == .tunnelOpen || result.next?.step == .registerConnection)
        let registration = result.transitions.first { $0.step == .registerConnection }
        #expect(registration?.isSatisfied == false)
        #expect(registration?.reason.contains("rejected") == true)
    }

    @Test func aStaleForwardIsRepairedNotDuplicated() {
        let site = sshSite()
        var observed = connectedState(site: site)
        observed.tunnel = .stale(localPort: 8712, reason: "nothing is listening on the port")
        observed.serverHTTP = .unreachable(reason: "no answer")
        observed.registration = .absent
        let result = plan(observed)
        #expect(result.next?.step == .tunnelOpen)
        #expect(result.next?.reason.contains("repair it rather than adding another") == true)
    }

    // MARK: Transport / topology variation

    @Test func aDirectTransportSiteSkipsEverySSHRung() {
        let direct = ClusterSiteProfile.gpuWorkstation
        let observed = ClusterObservedState(
            siteID: "gpu", siteName: "GPU workstation",
            controlMaster: .notApplicable,
            payload: .notApplicable,
            bootstrap: .notApplicable,
            profileValidation: .notApplicable,
            controller: .notApplicable,
            daemonHost: .notApplicable,
            tunnel: .notApplicable,
            serverHTTP: .reachable(build: "steerlab-server 0.1.0", role: nil, root: nil),
            registration: .present(endpoint: "http://127.0.0.1:8080"),
            bearerToken: .absent)
        let result = plan(observed, site: direct, permissions: [])
        #expect(result.isSatisfied)
        #expect(result.state(permissions: []) == .ready)
    }

    @Test func aLoginDaemonTopologyNeedsNoControllerJob() {
        let site = sshSite(topology: .loginDaemon)
        var observed = connectedState(site: site)
        observed.controller = .notApplicable
        observed.daemonHost = .notApplicable
        let result = plan(observed, site: site)
        #expect(result.isSatisfied)
        let controller = result.transitions.first { $0.step == .controllerStart }
        #expect(controller?.isSatisfied == true)
        #expect(controller?.reason.contains("loginDaemon") == true)
    }

    // MARK: Blockers

    @Test func anUnusableSiteProfileBlocksBeforeAnyTransition() {
        var observed = connectedState(site: sshSite())
        observed.siteConfiguration = .invalid(["the site has no SSH host"])
        let result = plan(observed)
        #expect(result.transitions.isEmpty)
        #expect(result.blockers == ["the site has no SSH host"])
        #expect(result.state(permissions: .allMutations) == .blocked)
        #expect(!result.isSatisfied)
    }

    @Test func anUnprovableDeploymentIdentityIsSaidHonestly() {
        let site = sshSite()
        var observed = connectedState(site: site)
        observed.payload = .unknown(
            reason: "the local payload is a development checkout with no deployment manifest")
        let result = plan(observed, permissions: [])
        // A dirty dev payload is neither claimed current nor silently pushed.
        #expect(result.next?.step == .pushCode)
        #expect(result.next?.reason.contains("development checkout") == true)
        #expect(result.state(permissions: []) == .needsApproval)
    }
}
