import Foundation

// =============================================================================
// The pure planner (CLUSTER-CLI-LIFECYCLE-PLAN §7.3).
//
//     (site profile, observed state, requested target, permissions) -> plan
//
// No I/O, no clock, no actor. This is the ONE definition of "what remains to
// reach connected", shared by the app, the CLI, and the tests — which is the
// whole point: two clients that plan differently are two implementations.
//
// Doctrines encoded here, not in prose:
//
//   * Nothing mutating is ever planned as actionable without ITS OWN
//     permission (push / bootstrap / controller-start are separate).
//   * A controller whose scheduler state could not be READ is never a
//     start: an unproven death does not license a resubmit.
//   * A queued controller is a WAIT, not a failure — `controllerWait` is a
//     first-class step so `pending` can never decay into a timeout.
//   * A stale `serverd.host` never satisfies the controller rung.
// =============================================================================

public enum ClusterLifecyclePlanner {

    /// Build the ordered plan. Pure and total.
    public static func plan(
        site: ClusterSiteProfile,
        observed: ClusterObservedState,
        target: ClusterLifecycleTarget,
        permissions: ClusterLifecyclePermissions
    ) -> ClusterLifecyclePlan {
        if case .invalid(let problems) = observed.siteConfiguration {
            return ClusterLifecyclePlan(
                target: target, transitions: [], blockers: problems)
        }

        var transitions: [ClusterLifecycleTransition] = []
        transitions.append(authenticate(site: site, observed: observed))
        transitions.append(contentsOf: deploy(site: site, observed: observed))
        transitions.append(contentsOf: bootstrap(site: site, observed: observed))
        transitions.append(validate(site: site, observed: observed))
        transitions.append(contentsOf: controller(site: site, observed: observed))
        transitions.append(contentsOf: connect(site: site, observed: observed))

        // Only the rungs at or below the requested target belong in the plan.
        let scoped = transitions.filter { $0.step.target <= target }
        return ClusterLifecyclePlan(target: target, transitions: scoped)
    }

    // MARK: Rung 1 — authenticate

    private static func authenticate(
        site: ClusterSiteProfile, observed: ClusterObservedState
    ) -> ClusterLifecycleTransition {
        switch observed.controlMaster {
        case .notApplicable:
            return ClusterLifecycleTransition(
                step: .authenticate,
                reason: "direct transport — there is nothing to authenticate",
                gating: .satisfied)
        case .alive:
            return ClusterLifecycleTransition(
                step: .authenticate,
                reason: "an SSH ControlMaster is already alive for this site",
                gating: .satisfied)
        case .unresponsive:
            return ClusterLifecycleTransition(
                step: .authenticate,
                reason: "the SSH ControlMaster socket exists but does not answer — "
                    + "sign in again in a visible Terminal (the app never handles "
                    + "the credential)",
                gating: ClusterTransitionGating(requiresHumanAuthentication: true))
        case .absent:
            return ClusterLifecycleTransition(
                step: .authenticate,
                reason: "no SSH ControlMaster for this site — complete password and "
                    + "multi-factor authentication in a visible Terminal",
                gating: ClusterTransitionGating(requiresHumanAuthentication: true))
        }
    }

    // MARK: Rung 2 — deployed payload

    private static func deploy(
        site: ClusterSiteProfile, observed: ClusterObservedState
    ) -> [ClusterLifecycleTransition] {
        let step = ClusterLifecycleStep.pushCode
        switch observed.payload {
        case .notApplicable:
            return [
                ClusterLifecycleTransition(
                    step: step,
                    reason: "direct transport — the server bundle is placed on the box "
                        + "by hand",
                    gating: .satisfied)
            ]
        case .current:
            return [
                ClusterLifecycleTransition(
                    step: step,
                    reason: "the deployed payload matches the local one",
                    gating: .satisfied)
            ]
        case .absent:
            return [
                ClusterLifecycleTransition(
                    step: step,
                    reason: "no server payload is deployed at this site",
                    gating: pushGating)
            ]
        case .stale(let reason):
            return [
                ClusterLifecycleTransition(
                    step: step, reason: reason, gating: pushGating)
            ]
        case .unknown(let reason):
            // Cannot prove currency ⇒ offer the push, and say why. This is the
            // dev-checkout case: a dirty payload is identified honestly rather
            // than assumed current.
            return [
                ClusterLifecycleTransition(
                    step: step,
                    reason: "the deployed payload cannot be compared (\(reason)) — "
                        + "push to be sure",
                    gating: pushGating)
            ]
        }
    }

    private static let pushGating = ClusterTransitionGating(
        requiredPermission: .push)

    // MARK: Rung 3 — bootstrapped environment

    private static func bootstrap(
        site: ClusterSiteProfile, observed: ClusterObservedState
    ) -> [ClusterLifecycleTransition] {
        let satisfiedPlan = ClusterLifecycleTransition(
            step: .bootstrapPlan,
            reason: "the environment is already bootstrapped — no plan is needed",
            gating: .satisfied)
        switch observed.bootstrap {
        case .notApplicable:
            return [
                ClusterLifecycleTransition(
                    step: .bootstrapPlan,
                    reason: "direct transport — bootstrap runs on the box itself",
                    gating: .satisfied),
                ClusterLifecycleTransition(
                    step: .bootstrapApply,
                    reason: "direct transport — bootstrap runs on the box itself",
                    gating: .satisfied),
            ]
        case .valid:
            return [
                satisfiedPlan,
                ClusterLifecycleTransition(
                    step: .bootstrapApply,
                    reason: "the site's environment file sources cleanly",
                    gating: .satisfied),
            ]
        case .absent, .invalid, .unknown:
            let why: String
            switch observed.bootstrap {
            case .invalid(let reason): why = "the bootstrapped environment is unusable (\(reason))"
            case .unknown(let reason): why = "the environment could not be checked (\(reason))"
            default: why = "the site has no bootstrapped environment"
            }
            return [
                // The dry run is READ-ONLY and needs no approval: reviewing a
                // plan must never be gated behind permission to execute it.
                // It is SATISFIED once a plan for this exact site +
                // configuration has been reviewed — the review, not the
                // resulting environment, is what this step produces.
                ClusterLifecycleTransition(
                    step: .bootstrapPlan,
                    reason: observed.bootstrapPlanReviewed
                        ? "a bootstrap plan for this exact configuration has been reviewed"
                        : "\(why) — render the plan for review first",
                    gating: observed.bootstrapPlanReviewed
                        ? .satisfied : ClusterTransitionGating(isReadOnly: true)),
                ClusterLifecycleTransition(
                    step: .bootstrapApply,
                    reason: "\(why) — apply the reviewed plan",
                    gating: ClusterTransitionGating(
                        requiredPermission: .bootstrap,
                        isAsynchronous: true,
                        isResourceConsuming: true)),
            ]
        }
    }

    // MARK: Rung 4 — profile validation

    private static func validate(
        site: ClusterSiteProfile, observed: ClusterObservedState
    ) -> ClusterLifecycleTransition {
        switch observed.profileValidation {
        case .notApplicable:
            return ClusterLifecycleTransition(
                step: .validate,
                reason: "direct transport — run `steerlab-server profile validate` on "
                    + "the box",
                gating: .satisfied)
        case .pass:
            return ClusterLifecycleTransition(
                step: .validate,
                reason: "the remote profile validated cleanly for this site profile",
                gating: .satisfied)
        case .warn(let messages):
            return ClusterLifecycleTransition(
                step: .validate,
                reason: "the remote profile validated with \(messages.count) "
                    + "warning(s) — not blocking",
                gating: .satisfied)
        case .notRun:
            return ClusterLifecycleTransition(
                step: .validate,
                reason: "the remote profile has not been validated for this site "
                    + "profile yet",
                gating: ClusterTransitionGating(isReadOnly: true))
        case .fail(let messages):
            return ClusterLifecycleTransition(
                step: .validate,
                reason: "the remote profile last validated with failures "
                    + "(\(messages.prefix(2).joined(separator: " · "))) — re-check it",
                gating: ClusterTransitionGating(isReadOnly: true))
        }
    }

    // MARK: Rung 5 — controller job

    private static func controller(
        site: ClusterSiteProfile, observed: ClusterObservedState
    ) -> [ClusterLifecycleTransition] {
        switch observed.controller {
        case .notApplicable:
            return [
                ClusterLifecycleTransition(
                    step: .controllerStart,
                    reason: "topology is \(site.topology.rawValue) — no controller job "
                        + "is needed",
                    gating: .satisfied)
            ]
        case .running(let jobID):
            // Running is not enough on its own: the tunnel needs the node the
            // job published, and a stale record must not pass for it.
            if case .current = observed.daemonHost {
                return [
                    ClusterLifecycleTransition(
                        step: .controllerStart,
                        reason: "controller job \(jobID) is running and has published "
                            + "its node",
                        gating: .satisfied)
                ]
            }
            return [
                ClusterLifecycleTransition(
                    step: .controllerStart,
                    reason: "controller job \(jobID) is running",
                    gating: .satisfied),
                ClusterLifecycleTransition(
                    step: .controllerWait,
                    reason: "controller job \(jobID) is running but has not published a "
                        + "usable \(site.daemonHostFilePath) record yet — wait for it "
                        + "rather than starting a second controller",
                    gating: ClusterTransitionGating(
                        isReadOnly: true, isAsynchronous: true)),
            ]
        case .pending(let jobID, let reason):
            let detail = reason.isEmpty || reason == "None"
                ? "" : " (reason: \(reason))"
            var explanation =
                "controller job \(jobID) is queued\(detail) — this is a wait, not a "
                + "failure"
            if ClusterProvisioner.reasonSuggestsMaintenanceReservation(reason) {
                explanation +=
                    "; a maintenance reservation is likely blocking it — its walltime "
                    + "may not fit before the window"
            }
            return [
                ClusterLifecycleTransition(
                    step: .controllerWait, reason: explanation,
                    gating: ClusterTransitionGating(
                        isReadOnly: true, isAsynchronous: true))
            ]
        case .absent:
            return [
                ClusterLifecycleTransition(
                    step: .controllerStart,
                    reason: "no controller job is running for this site",
                    gating: controllerGating)
            ]
        case .failed(let jobID, let state):
            return [
                ClusterLifecycleTransition(
                    step: .controllerStart,
                    reason: "controller job \(jobID) is \(state) — check its log, then "
                        + "submit a new one",
                    gating: controllerGating)
            ]
        case .unknown(let reason):
            // THE doctrine: a failed QUERY is not a dead job. Never plan a
            // start here — plan another look.
            return [
                ClusterLifecycleTransition(
                    step: .controllerWait,
                    reason: "the controller job's state is unknown (\(reason)) — "
                        + "observe again rather than submitting a second controller",
                    gating: ClusterTransitionGating(isReadOnly: true))
            ]
        }
    }

    private static let controllerGating = ClusterTransitionGating(
        requiredPermission: .controllerStart,
        isAsynchronous: true,
        isResourceConsuming: true)

    // MARK: Rung 6 — tunnel + registration

    private static func connect(
        site: ClusterSiteProfile, observed: ClusterObservedState
    ) -> [ClusterLifecycleTransition] {
        var transitions: [ClusterLifecycleTransition] = []

        if !site.isSSHTransport {
            transitions.append(
                ClusterLifecycleTransition(
                    step: .tunnelOpen,
                    reason: "direct transport — no tunnel is needed",
                    gating: .satisfied))
        } else {
            switch observed.tunnel {
            case .up(let localPort):
                // An HTTP 401 is the endpoint ANSWERING: the forward is fine
                // and the missing thing is the token, which is the
                // registration step's job. Repairing the tunnel there would
                // churn a healthy forward and never fix the actual problem.
                let answered: Bool
                switch observed.serverHTTP {
                case .reachable, .authFailed: answered = true
                case .unreachable, .identityMismatch: answered = false
                }
                if answered {
                    transitions.append(
                        ClusterLifecycleTransition(
                            step: .tunnelOpen,
                            reason: "the forward on 127.0.0.1:\(localPort) is up and the "
                                + "server answers behind it",
                            gating: .satisfied))
                } else {
                    transitions.append(
                        ClusterLifecycleTransition(
                            step: .tunnelOpen,
                            reason: "the forward on 127.0.0.1:\(localPort) is up but the "
                                + "server does not answer behind it — repair the forward",
                            gating: ClusterTransitionGating(isReadOnly: false)))
                }
            case .absent:
                transitions.append(
                    ClusterLifecycleTransition(
                        step: .tunnelOpen,
                        reason: "no local forward for this site",
                        gating: ClusterTransitionGating()))
            case .stale(let localPort, let reason):
                transitions.append(
                    ClusterLifecycleTransition(
                        step: .tunnelOpen,
                        reason: "the remembered forward on 127.0.0.1:\(localPort) is "
                            + "stale (\(reason)) — repair it rather than adding another",
                        gating: ClusterTransitionGating()))
            case .conflicted(let localPort, let reason):
                transitions.append(
                    ClusterLifecycleTransition(
                        step: .tunnelOpen,
                        reason: "127.0.0.1:\(localPort) is taken (\(reason)) — the "
                            + "forward must move to a free port and the ONE saved "
                            + "registration follow it",
                        gating: ClusterTransitionGating()))
            case .notApplicable:
                transitions.append(
                    ClusterLifecycleTransition(
                        step: .tunnelOpen,
                        reason: "no tunnel is needed for this transport",
                        gating: .satisfied))
            }
        }

        let registered: Bool
        if case .present = observed.registration { registered = true } else { registered = false }
        let reachable: Bool
        if case .reachable = observed.serverHTTP { reachable = true } else { reachable = false }
        let tokenOK = observed.bearerToken == .accepted
            || (!site.isSSHTransport && observed.bearerToken != .rejected)

        if registered, reachable, tokenOK {
            transitions.append(
                ClusterLifecycleTransition(
                    step: .registerConnection,
                    reason: "the endpoint answers, its identity is verified, and the "
                        + "site is registered",
                    gating: .satisfied))
        } else {
            var why: [String] = []
            if !reachable { why.append("the endpoint's identity is not verified") }
            if !tokenOK {
                why.append(
                    observed.bearerToken == .rejected
                        ? "the stored bearer token was rejected"
                        : "no bearer token has been accepted for this site")
            }
            if !registered { why.append("the site has no current saved registration") }
            transitions.append(
                ClusterLifecycleTransition(
                    step: .registerConnection,
                    reason: why.joined(separator: "; "),
                    gating: ClusterTransitionGating()))
        }
        return transitions
    }
}
