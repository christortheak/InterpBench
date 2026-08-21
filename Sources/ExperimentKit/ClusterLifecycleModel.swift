import Foundation

// =============================================================================
// The shared cluster-lifecycle vocabulary (CLUSTER-CLI-LIFECYCLE-PLAN §7.1/§7.2).
//
// One lifecycle, two clients: the SwiftUI wizard and `steerlab-cli` both
// describe cluster state with THESE types. Nothing here talks to ssh, the
// filesystem, or SwiftUI — it is the noun vocabulary the inspector fills in,
// the planner reads, and the operation records + JSON envelope serialize.
//
// The hard design rule from the plan: **do not compress the layers into one
// `connected` Boolean.** Several independent connections are colloquially
// called "the server connection" (SSH master, deployed payload, bootstrap env,
// Slurm controller job, its `serverd.host` record, the local forward, the HTTP
// endpoint, the saved registration, the bearer token), and a single Boolean is
// exactly why the current failure modes are hard to diagnose. Every layer is
// observed and reported on its own.
//
// Security invariant (plan §4.2): no type in this file has a field that can
// hold a credential. Tokens appear only as PRESENCE + SOURCE.
// =============================================================================

// MARK: - Targets and steps

/// The lifecycle ladder, in order. `ensure --target X` means "reach X"; each
/// rung implies every rung below it.
public enum ClusterLifecycleTarget: String, CaseIterable, Codable, Sendable, Comparable {
    case authenticated
    case codeDeployed
    case bootstrapped
    case validated
    case controllerRunning
    case connected

    /// Position on the ladder (0 = lowest).
    public var order: Int {
        Self.allCases.firstIndex(of: self) ?? 0
    }

    public static func < (lhs: Self, rhs: Self) -> Bool { lhs.order < rhs.order }

    /// The CLI spelling (`--target connected`), kebab-cased.
    public var cliName: String {
        switch self {
        case .authenticated: "authenticated"
        case .codeDeployed: "code-deployed"
        case .bootstrapped: "bootstrapped"
        case .validated: "validated"
        case .controllerRunning: "controller-running"
        case .connected: "connected"
        }
    }

    /// Parse either spelling (`code-deployed` or `codeDeployed`).
    public static func parse(_ text: String) -> ClusterLifecycleTarget? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return allCases.first { $0.cliName == trimmed || $0.rawValue == trimmed }
    }
}

/// Stable step vocabulary shared by the plan, the operation record, and the
/// JSON envelope's `step` field.
public enum ClusterLifecycleStep: String, CaseIterable, Codable, Sendable {
    case authenticate
    case pushCode
    case bootstrapPlan
    case bootstrapApply
    case validate
    case controllerStart
    /// Not an action: the controller job exists and is queued. Waiting is the
    /// correct behavior, and it is a first-class step so "pending" can never
    /// be mistaken for "failed".
    case controllerWait
    case tunnelOpen
    case registerConnection

    /// The rung of the ladder this step serves — how the planner filters a
    /// plan down to the requested target.
    public var target: ClusterLifecycleTarget {
        switch self {
        case .authenticate: .authenticated
        case .pushCode: .codeDeployed
        case .bootstrapPlan, .bootstrapApply: .bootstrapped
        case .validate: .validated
        case .controllerStart, .controllerWait: .controllerRunning
        case .tunnelOpen, .registerConnection: .connected
        }
    }
}

/// Stable top-level states (plan §6.5). The JSON state is authoritative; exit
/// codes are a convenience for shell callers.
public enum ClusterLifecycleState: String, Codable, Sendable, CaseIterable {
    case ready
    case planned
    case running
    case pending
    case needsHumanAuthentication
    case needsApproval
    case blocked
    case degraded
    case failed

    /// This state's twin in the shared CLI vocabulary
    /// (WP0-AGENT-SURFACE-AUDIT §2.3). `SteerLabCLIState` reproduces all nine
    /// cluster cases name for name and adds three (`okWithAdvisories`,
    /// `refused`, `notFound`) that the cluster lifecycle does not use.
    ///
    /// Deliberately an EXHAUSTIVE switch rather than a `rawValue` lookup: a
    /// lookup would need a fallback for the impossible case, and a fallback is
    /// exactly how a renamed case would silently become exit 70. This way the
    /// compiler makes anyone adding a cluster state name its shared meaning.
    public var sharedState: SteerLabCLIState {
        switch self {
        case .ready: .ready
        case .planned: .planned
        case .running: .running
        case .pending: .pending
        case .needsHumanAuthentication: .needsHumanAuthentication
        case .needsApproval: .needsApproval
        case .blocked: .blocked
        case .degraded: .degraded
        case .failed: .failed
        }
    }

    /// Recommended process exit code (plan §6.5). Phase C maps these; they
    /// live here so both clients agree.
    ///
    /// Delegates to the shared vocabulary (audit §2.3, "shared enum, not a
    /// copy") so there is one exit-code table in the codebase rather than two
    /// that can drift. Behaviour is unchanged, case for case, and
    /// `CLIEnvelopeTests.clusterAndSharedVocabulariesAgree` plus the existing
    /// `ClusterCLIEnvelopeTests.exitCodesFollowTheStateVocabularyExactly` —
    /// which still asserts the nine literal codes — pin it from both sides.
    public var exitCode: Int32 { sharedState.exitCode }
}

// MARK: - Permissions

/// Explicit mutation authority (plan §4.3). Inspection is always allowed;
/// every remote side effect needs ITS OWN permission — deliberately not one
/// ambiguous `--yes`.
public struct ClusterLifecyclePermissions: OptionSet, Codable, Sendable, Hashable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    public static let push = ClusterLifecyclePermissions(rawValue: 1 << 0)
    public static let bootstrap = ClusterLifecyclePermissions(rawValue: 1 << 1)
    public static let controllerStart = ClusterLifecyclePermissions(rawValue: 1 << 2)
    /// Opening the visible authentication Terminal is separately authorized —
    /// it is a user-visible side effect, not a mutation of the cluster.
    public static let openAuthTerminal = ClusterLifecyclePermissions(rawValue: 1 << 3)

    /// Everything that changes remote state (NOT the auth terminal).
    public static let allMutations: ClusterLifecyclePermissions = [
        .push, .bootstrap, .controllerStart,
    ]

    /// Stable identifiers for records and JSON — never a bit pattern.
    public var identifiers: [String] {
        var out: [String] = []
        if contains(.push) { out.append("push") }
        if contains(.bootstrap) { out.append("bootstrap") }
        if contains(.controllerStart) { out.append("controller-start") }
        if contains(.openAuthTerminal) { out.append("open-auth-terminal") }
        return out
    }

    /// The CLI flag that grants one permission (plan §4.3).
    public var flags: [String] { identifiers.map { "--allow-\($0)" } }

    public static func parse(identifier: String) -> ClusterLifecyclePermissions? {
        switch identifier {
        case "push": .push
        case "bootstrap": .bootstrap
        case "controller-start", "controllerStart": .controllerStart
        case "open-auth-terminal", "openAuthTerminal": .openAuthTerminal
        default: nil
        }
    }
}

// MARK: - Per-layer observations (plan §7.2)

/// Whether the site profile can be acted on at all.
public enum ClusterSiteConfigurationObservation: Sendable, Equatable {
    case valid
    /// Blocking configuration defects, in plain language.
    case invalid([String])

    public var summary: String {
        switch self {
        case .valid: "valid"
        case .invalid(let problems): "invalid (\(problems.joined(separator: " · ")))"
        }
    }
}

/// The shared SSH ControlMaster the user authenticates once, in Terminal.
public enum ClusterControlMasterObservation: String, Sendable, Codable, CaseIterable {
    case absent
    case alive
    /// The socket exists but the probe did not answer cleanly — a repairable
    /// state, distinct from "never authenticated".
    case unresponsive
    /// Direct transport: there is no master to have.
    case notApplicable

    public var summary: String { rawValue }
}

/// The deployed server payload on the far side.
public enum ClusterPayloadObservation: Sendable, Equatable {
    case absent
    case current(deploymentHash: String?)
    case stale(reason: String)
    /// The comparison could not be made (no master, unreadable marker). NOT a
    /// licence to skip the push — but not proof one is needed either.
    case unknown(reason: String)
    case notApplicable

    public var summary: String {
        switch self {
        case .absent: "absent"
        case .current: "current"
        case .stale(let reason): "stale (\(reason))"
        case .unknown(let reason): "unknown (\(reason))"
        case .notApplicable: "notApplicable"
        }
    }
}

/// The bootstrapped Python environment.
public enum ClusterBootstrapObservation: Sendable, Equatable {
    case absent
    case valid(envFile: String, prefix: String?)
    case invalid(reason: String)
    case unknown(reason: String)
    case notApplicable

    public var summary: String {
        switch self {
        case .absent: "absent"
        case .valid: "valid"
        case .invalid(let reason): "invalid (\(reason))"
        case .unknown(let reason): "unknown (\(reason))"
        case .notApplicable: "notApplicable"
        }
    }
}

/// `steerlab-server profile validate` on the far side.
public enum ClusterProfileValidationObservation: Sendable, Equatable {
    case notRun
    case pass
    case warn(messages: [String])
    case fail(messages: [String])
    case notApplicable

    public var summary: String {
        switch self {
        case .notRun: "notRun"
        case .pass: "pass"
        case .warn(let messages): "warn (\(messages.count))"
        case .fail(let messages): "fail (\(messages.count))"
        case .notApplicable: "notApplicable"
        }
    }
}

/// The daemon-in-a-job controller, reconciled from the durable job id and the
/// scheduler.
///
/// `unknown` is load-bearing: a failed scheduler QUERY says nothing about the
/// job. The repository doctrine — an unproven death never licenses a resubmit
/// — lives in the planner, and it depends on this case existing separately
/// from `absent` (which means squeue ANSWERED that the job left the queue).
public enum ClusterControllerObservation: Sendable, Equatable {
    case absent
    case pending(jobID: String, reason: String)
    case running(jobID: String)
    case failed(jobID: String, state: String)
    case unknown(reason: String)
    case notApplicable

    public var summary: String {
        switch self {
        case .absent: "absent"
        case .pending(let jobID, _): "pending (\(jobID))"
        case .running(let jobID): "running (\(jobID))"
        case .failed(let jobID, let state): "failed (\(jobID) \(state))"
        case .unknown(let reason): "unknown (\(reason))"
        case .notApplicable: "notApplicable"
        }
    }

    public var jobID: String? {
        switch self {
        case .pending(let id, _), .running(let id), .failed(let id, _): id
        case .absent, .unknown, .notApplicable: nil
        }
    }

    public var schedulerState: String? {
        switch self {
        case .pending: "PENDING"
        case .running: "RUNNING"
        case .failed(_, let state): state
        case .absent, .unknown, .notApplicable: nil
        }
    }
}

/// The RENDERED controller script on the far side
/// (`<metadataRoot>/controller-job.sbatch`), compared against the deployed
/// template it should have come from.
///
/// A layer of its own because it is the one deployed artifact a code push does
/// NOT refresh: `cluster push` rsyncs the template, and the rendered child of
/// an older template keeps sitting in the metadata root, re-submittable by
/// hand. That is exactly how serverd 47564632 ran the template-side self-chain
/// fix for 24 h and never chained (open-issues §1 field report, 2026-08-20).
///
/// `unknown` is separate from `stale` on purpose: a probe that could not read
/// the template has learned nothing, and crying stale over it would train the
/// operator to ignore the finding.
public enum ClusterControllerScriptObservation: Sendable, Equatable {
    /// Nothing rendered yet — the first `controller start` writes it.
    case absent
    case current(templateHash: String, renderedAt: String)
    /// Rendered from different template bytes, or carrying no stamp at all.
    case stale(reason: String)
    case unknown(reason: String)
    case notApplicable

    public var summary: String {
        switch self {
        case .absent: "absent"
        case .current(let hash, _):
            "current (\(ClusterProvisioner.shortDigest(hash)))"
        case .stale(let reason): "stale (\(reason))"
        case .unknown(let reason): "unknown (\(reason))"
        case .notApplicable: "notApplicable"
        }
    }

    /// The plain-language finding a read-only flow prints, or nil when there
    /// is nothing to say. It NAMES the re-render command — a finding that does
    /// not say what to type is a finding the operator has to research.
    public func advisory(siteID: String) -> String? {
        let repair = "re-render with: "
            + ClusterProvisioner.renderControllerCommand(siteID: siteID)
        switch self {
        case .current, .notApplicable:
            return nil
        case .absent:
            return "no rendered controller script exists at this site's "
                + "metadata root yet — `cluster controller start` writes one; "
                + repair.replacingOccurrences(
                    of: "re-render with: ", with: "render it without submitting: ")
        case .stale(let reason):
            return "the RENDERED controller script is stale: \(reason). A "
                + "hand `sbatch` of it launches a controller that may not "
                + "self-chain at walltime — \(repair)"
        case .unknown(let reason):
            return "the rendered controller script could not be checked "
                + "against its template: \(reason)"
        }
    }
}

/// `<metadataRoot>/serverd.host` — the record the controller job writes so the
/// tunnel knows which compute node to forward to.
///
/// A stale host file must never be treated as proof that a controller is
/// alive (plan §7.10): the job id that wrote it is compared against the job
/// the scheduler currently reports.
public enum ClusterDaemonHostObservation: Sendable, Equatable {
    case absent
    case current(host: String)
    case stale(host: String, reason: String)
    case notApplicable

    public var summary: String {
        switch self {
        case .absent: "absent"
        case .current(let host): "current (\(host))"
        case .stale(_, let reason): "stale (\(reason))"
        case .notApplicable: "notApplicable"
        }
    }

    public var host: String? {
        switch self {
        case .current(let host), .stale(let host, _): host
        case .absent, .notApplicable: nil
        }
    }
}

/// The local SSH forward. Identity is site + remote identity + target host +
/// remote port + local port (plan §7.7) — an open local port is not a tunnel.
public enum ClusterTunnelObservation: Sendable, Equatable {
    case absent
    case up(localPort: Int)
    /// A persisted forward record whose forward is gone (or points somewhere
    /// else) — repairable, and it must be repaired rather than duplicated.
    case stale(localPort: Int, reason: String)
    /// Something else holds the local port.
    case conflicted(localPort: Int, reason: String)
    case notApplicable

    public var summary: String {
        switch self {
        case .absent: "absent"
        case .up(let port): "up (127.0.0.1:\(port))"
        case .stale(let port, let reason): "stale (\(port): \(reason))"
        case .conflicted(let port, let reason): "conflicted (\(port): \(reason))"
        case .notApplicable: "notApplicable"
        }
    }

    public var localPort: Int? {
        switch self {
        case .up(let port), .stale(let port, _), .conflicted(let port, _): port
        case .absent, .notApplicable: nil
        }
    }
}

/// What answered at the endpoint. `cluster connect` must prove endpoint
/// IDENTITY and authentication, never just an open TCP port (plan §7.11).
public enum ClusterServerHTTPObservation: Sendable, Equatable {
    case unreachable(reason: String)
    case reachable(build: String?, role: String?, root: String?)
    /// Something answered, but it is not the server this site expects.
    case identityMismatch(reason: String)
    case authFailed

    public var summary: String {
        switch self {
        case .unreachable: "unreachable"
        case .reachable: "reachable"
        case .identityMismatch: "identityMismatch"
        case .authFailed: "authFailed"
        }
    }

    public var serverBuild: String? {
        if case .reachable(let build, _, _) = self { return build }
        return nil
    }
}

/// The saved connection entry for this site in the shared repository.
public enum ClusterRegistrationObservation: Sendable, Equatable {
    case absent
    case present(endpoint: String)
    /// More than one record shares the site's canonical remote identity — a
    /// registry fork, reported rather than silently merged.
    case duplicate(count: Int)

    public var summary: String {
        switch self {
        case .absent: "absent"
        case .present(let endpoint): "present (\(endpoint))"
        case .duplicate(let count): "duplicate (\(count))"
        }
    }
}

/// The bearer token, described ONLY as presence and provenance. There is no
/// representation of the token's value anywhere in this vocabulary.
public enum ClusterBearerTokenObservation: String, Sendable, Codable, CaseIterable {
    case absent
    /// In the secret store, not yet exercised against the server.
    case present
    case accepted
    case rejected

    public var summary: String { rawValue }

    public var isAvailable: Bool { self != .absent }
}

// MARK: - The observed state

/// Everything one inspection pass learned about a site, layer by layer.
/// Produced by `ClusterLifecycleInspector`, consumed by
/// `ClusterLifecyclePlanner` (a pure function of exactly this plus the site,
/// the target, and the granted permissions).
public struct ClusterObservedState: Sendable, Equatable {
    public var siteID: String
    public var siteName: String
    /// SHA-256 of the site profile's canonical JSON — so an operation record
    /// can say which profile it acted on without copying the profile.
    public var siteProfileHash: String
    public var observedAt: Date

    public var siteConfiguration: ClusterSiteConfigurationObservation
    public var controlMaster: ClusterControlMasterObservation
    public var payload: ClusterPayloadObservation
    public var bootstrap: ClusterBootstrapObservation
    public var profileValidation: ClusterProfileValidationObservation
    public var controller: ClusterControllerObservation
    /// The rendered controller script vs the deployed template (§1 field
    /// report). Observed on the daemon-in-a-job topology only.
    public var controllerScript: ClusterControllerScriptObservation
    public var daemonHost: ClusterDaemonHostObservation
    public var tunnel: ClusterTunnelObservation
    public var serverHTTP: ClusterServerHTTPObservation
    public var registration: ClusterRegistrationObservation
    public var bearerToken: ClusterBearerTokenObservation
    /// Whether a durable bootstrap plan matching the CURRENT site +
    /// configuration has been reviewed (§7.9). Observed rather than derived
    /// so the planner stays pure: computing the expected plan hash needs the
    /// provisioning configuration, which the planner deliberately does not
    /// see. Any site, payload, root, resource, or command change makes this
    /// false again.
    public var bootstrapPlanReviewed: Bool

    public init(
        siteID: String,
        siteName: String,
        siteProfileHash: String = "",
        observedAt: Date = Date(),
        siteConfiguration: ClusterSiteConfigurationObservation = .valid,
        controlMaster: ClusterControlMasterObservation = .absent,
        payload: ClusterPayloadObservation = .unknown(reason: "not inspected"),
        bootstrap: ClusterBootstrapObservation = .unknown(reason: "not inspected"),
        profileValidation: ClusterProfileValidationObservation = .notRun,
        controller: ClusterControllerObservation = .absent,
        controllerScript: ClusterControllerScriptObservation = .notApplicable,
        daemonHost: ClusterDaemonHostObservation = .absent,
        tunnel: ClusterTunnelObservation = .absent,
        serverHTTP: ClusterServerHTTPObservation = .unreachable(reason: "not probed"),
        registration: ClusterRegistrationObservation = .absent,
        bearerToken: ClusterBearerTokenObservation = .absent,
        bootstrapPlanReviewed: Bool = false
    ) {
        self.siteID = siteID
        self.siteName = siteName
        self.siteProfileHash = siteProfileHash
        self.observedAt = observedAt
        self.siteConfiguration = siteConfiguration
        self.controlMaster = controlMaster
        self.payload = payload
        self.bootstrap = bootstrap
        self.profileValidation = profileValidation
        self.controller = controller
        self.controllerScript = controllerScript
        self.daemonHost = daemonHost
        self.tunnel = tunnel
        self.serverHTTP = serverHTTP
        self.registration = registration
        self.bearerToken = bearerToken
        self.bootstrapPlanReviewed = bootstrapPlanReviewed
    }

    /// Layer → stable summary string, for the machine envelope and for
    /// "every failure names the failing layer" diagnostics. Ordered.
    public var layerSummaries: [(layer: String, state: String)] {
        [
            ("siteConfiguration", siteConfiguration.summary),
            ("controlMaster", controlMaster.summary),
            ("payload", payload.summary),
            ("bootstrap", bootstrap.summary),
            ("profileValidation", profileValidation.summary),
            ("controller", controller.summary),
            ("controllerScript", controllerScript.summary),
            ("daemonHost", daemonHost.summary),
            ("tunnel", tunnel.summary),
            ("serverHTTP", serverHTTP.summary),
            ("registration", registration.summary),
            ("bearerToken", bearerToken.summary),
        ]
    }

    /// Non-blocking findings: things a reader must be told even though the
    /// ladder is satisfied. Distinct from `ClusterLifecyclePlan.blockers`,
    /// which stop work — a stale rendered controller script does not stop
    /// anything, it just means the next hand `sbatch` is a chain-less
    /// controller, and the operator has to know that BEFORE the walltime.
    public var advisories: [String] {
        [controllerScript.advisory(siteID: siteID)].compactMap { $0 }
    }
}

// MARK: - Plan

/// How a transition is allowed to happen. Total by construction so the CLI can
/// answer "may I run this now?" without parsing prose.
public struct ClusterTransitionGating: Sendable, Equatable {
    /// Observes only — safe to run without any permission.
    public var isReadOnly: Bool
    /// Blocked until a human completes institutional authentication in their
    /// own Terminal. The agent never handles the credential.
    public var requiresHumanAuthentication: Bool
    /// The permission the caller must have granted; nil when none is needed.
    public var requiredPermission: ClusterLifecyclePermissions?
    /// Submits work and returns — completion is observed later, never held
    /// open in a foreground ssh session.
    public var isAsynchronous: Bool
    /// Consumes a scheduler allocation or otherwise costs the researcher.
    public var isResourceConsuming: Bool
    /// Nothing to do: this rung is already reached.
    public var isAlreadySatisfied: Bool

    public init(
        isReadOnly: Bool = false,
        requiresHumanAuthentication: Bool = false,
        requiredPermission: ClusterLifecyclePermissions? = nil,
        isAsynchronous: Bool = false,
        isResourceConsuming: Bool = false,
        isAlreadySatisfied: Bool = false
    ) {
        self.isReadOnly = isReadOnly
        self.requiresHumanAuthentication = requiresHumanAuthentication
        self.requiredPermission = requiredPermission
        self.isAsynchronous = isAsynchronous
        self.isResourceConsuming = isResourceConsuming
        self.isAlreadySatisfied = isAlreadySatisfied
    }

    public static let satisfied = ClusterTransitionGating(
        isReadOnly: true, isAlreadySatisfied: true)

    /// Stable labels for the machine envelope.
    public var labels: [String] {
        var out: [String] = []
        if isAlreadySatisfied { out.append("alreadySatisfied") }
        if isReadOnly { out.append("readOnly") }
        if requiresHumanAuthentication { out.append("humanGated") }
        if requiredPermission != nil { out.append("approvalGated") }
        if isAsynchronous { out.append("asynchronous") }
        if isResourceConsuming { out.append("resourceConsuming") }
        return out
    }
}

/// One ordered step of the plan, with WHY it is required.
public struct ClusterLifecycleTransition: Sendable, Equatable, Identifiable {
    public var step: ClusterLifecycleStep
    /// Plain-language justification, rendered verbatim by both clients.
    public var reason: String
    public var gating: ClusterTransitionGating

    public var id: String { step.rawValue }

    public init(
        step: ClusterLifecycleStep, reason: String, gating: ClusterTransitionGating
    ) {
        self.step = step
        self.reason = reason
        self.gating = gating
    }

    public var isSatisfied: Bool { gating.isAlreadySatisfied }

    /// Whether this transition may execute given the granted permissions.
    public func isPermitted(by permissions: ClusterLifecyclePermissions) -> Bool {
        guard let required = gating.requiredPermission else { return true }
        return permissions.contains(required)
    }
}

/// The ordered answer to "what remains to reach the target".
public struct ClusterLifecyclePlan: Sendable, Equatable {
    public var target: ClusterLifecycleTarget
    /// Every transition on the path, satisfied ones included — the plan is a
    /// description of the whole ladder, not just the remainder.
    public var transitions: [ClusterLifecycleTransition]
    /// Hard refusals: nothing on this site can proceed until they are fixed.
    public var blockers: [String]

    public init(
        target: ClusterLifecycleTarget,
        transitions: [ClusterLifecycleTransition],
        blockers: [String] = []
    ) {
        self.target = target
        self.transitions = transitions
        self.blockers = blockers
    }

    public var remaining: [ClusterLifecycleTransition] {
        transitions.filter { !$0.isSatisfied }
    }

    public var isSatisfied: Bool { blockers.isEmpty && remaining.isEmpty }

    /// The next transition to consider — the first unsatisfied one. Order is
    /// the ladder order, so this is also "the lowest layer that is not up".
    public var next: ClusterLifecycleTransition? { remaining.first }

    /// The state this plan implies, before anything is executed.
    public func state(
        permissions: ClusterLifecyclePermissions
    ) -> ClusterLifecycleState {
        if !blockers.isEmpty { return .blocked }
        guard let next else { return .ready }
        if next.gating.requiresHumanAuthentication { return .needsHumanAuthentication }
        if !next.isPermitted(by: permissions) { return .needsApproval }
        if next.step == .controllerWait {
            // A queued job is durably PENDING (poll and it will move). A job
            // whose state could not be READ is DEGRADED — retryable, and
            // explicitly not a licence to submit another controller.
            return next.gating.isAsynchronous ? .pending : .degraded
        }
        return .planned
    }
}

// MARK: - Result envelope

/// What one `plan` / `ensure` invocation reports. This is the shape Phase C
/// serializes; it deliberately contains no credential-shaped field.
public struct ClusterLifecycleResult: Sendable, Equatable {
    public static let schemaVersion = 1

    public var operationID: String
    public var siteID: String
    public var siteName: String
    public var target: ClusterLifecycleTarget
    public var state: ClusterLifecycleState
    /// The step the state is ABOUT (the one that ran, or the one blocking).
    public var step: ClusterLifecycleStep?
    /// Whether this invocation changed anything. A repeated `ensure` at ready
    /// must report false.
    public var changed: Bool
    public var observedAt: Date
    public var message: String
    /// How long to wait before repeating, for durable-pending states.
    public var retryAfterSeconds: Int?
    /// The exact next command's shape — verb + whether a human must act.
    public var nextAction: NextAction?
    /// Local endpoint once connected (`http://127.0.0.1:<port>`).
    public var endpoint: String?
    /// PRESENCE only.
    public var tokenAvailable: Bool
    /// PROVENANCE only ("keychain"); never the value.
    public var tokenSource: String?
    public var serverBuild: String?
    public var schedulerJobID: String?
    public var schedulerState: String?
    public var observed: ClusterObservedState
    public var plan: ClusterLifecyclePlan

    public struct NextAction: Sendable, Equatable {
        public var verb: String
        public var requiresHuman: Bool
        /// Permissions the caller would have to grant, as CLI flags.
        public var missingPermissionFlags: [String]
        public var detail: String?

        public init(
            verb: String, requiresHuman: Bool = false,
            missingPermissionFlags: [String] = [], detail: String? = nil
        ) {
            self.verb = verb
            self.requiresHuman = requiresHuman
            self.missingPermissionFlags = missingPermissionFlags
            self.detail = detail
        }
    }

    public init(
        operationID: String,
        siteID: String,
        siteName: String,
        target: ClusterLifecycleTarget,
        state: ClusterLifecycleState,
        step: ClusterLifecycleStep? = nil,
        changed: Bool = false,
        observedAt: Date = Date(),
        message: String,
        retryAfterSeconds: Int? = nil,
        nextAction: NextAction? = nil,
        endpoint: String? = nil,
        tokenAvailable: Bool = false,
        tokenSource: String? = nil,
        serverBuild: String? = nil,
        schedulerJobID: String? = nil,
        schedulerState: String? = nil,
        observed: ClusterObservedState,
        plan: ClusterLifecyclePlan
    ) {
        self.operationID = operationID
        self.siteID = siteID
        self.siteName = siteName
        self.target = target
        self.state = state
        self.step = step
        self.changed = changed
        self.observedAt = observedAt
        self.message = message
        self.retryAfterSeconds = retryAfterSeconds
        self.nextAction = nextAction
        self.endpoint = endpoint
        self.tokenAvailable = tokenAvailable
        self.tokenSource = tokenSource
        self.serverBuild = serverBuild
        self.schedulerJobID = schedulerJobID
        self.schedulerState = schedulerState
        self.observed = observed
        self.plan = plan
    }

    public var exitCode: Int32 { state.exitCode }
}

// MARK: - Errors

/// Typed lifecycle failures, surfaced to run logs and the JSON envelope with a
/// stable `code` (never a localized string match).
public enum ClusterLifecycleError: Error, LocalizedError, Equatable {
    case unknownSite(String)
    case siteConfigurationInvalid([String])
    case operationInProgress(siteID: String, operationID: String)
    case bootstrapPlanMismatch(expected: String, supplied: String)
    case bootstrapPlanMissing
    case storeUnwritable(String)
    /// A hand-started controller job could not be PROVEN to be this site's
    /// live controller, so it is not recorded. Adopting an unverified job id
    /// would poison every later reconciliation: inspection would believe a
    /// controller exists and the planner would never start the real one.
    case controllerAdoptionUnverified(jobID: String, reason: String)
    /// A registry write would leave an ssh destination with NO `user@` half
    /// while the site is known to log in as `expectedUser` (open-issues §17).
    ///
    /// This is a write-time refusal because the failure at USE time is
    /// expensive and human-facing: `ssh <host>` then authenticates as the
    /// local account, and an SSO cluster reports an unknown user as
    /// unenrolled — the 2026-08-18 outage presented as "asks for my password,
    /// then tells me to enroll in 2FA, then asks again" and burned several Duo
    /// attempts before anyone suspected the registry.
    case sshLoginDropped(
        siteID: String, host: String, expectedUser: String, source: String)

    public var code: String {
        switch self {
        case .unknownSite: "unknownSite"
        case .siteConfigurationInvalid: "siteConfigurationInvalid"
        case .operationInProgress: "operationInProgress"
        case .bootstrapPlanMismatch: "bootstrapPlanMismatch"
        case .bootstrapPlanMissing: "bootstrapPlanMissing"
        case .storeUnwritable: "storeUnwritable"
        case .controllerAdoptionUnverified: "controllerAdoptionUnverified"
        case .sshLoginDropped: "sshLoginDropped"
        }
    }

    public var errorDescription: String? {
        switch self {
        case .unknownSite(let id):
            "no saved cluster site with id '\(id)' — list them with `cluster sites list`"
        case .siteConfigurationInvalid(let problems):
            "the site profile cannot be acted on: \(problems.joined(separator: " · "))"
        case .operationInProgress(let siteID, let operationID):
            "another SteerLab process is already running lifecycle operation "
                + "\(operationID) for site \(siteID) — observe it rather than "
                + "starting a second one"
        case .bootstrapPlanMismatch(let expected, let supplied):
            "the bootstrap plan changed since it was reviewed (reviewed "
                + "\(supplied.prefix(12)), current \(expected.prefix(12))) — "
                + "re-run the dry run and review the new plan"
        case .bootstrapPlanMissing:
            "no reviewed bootstrap plan for this site — run the dry run first"
        case .storeUnwritable(let detail):
            "the cluster operation store is not writable: \(detail)"
        case .controllerAdoptionUnverified(let jobID, let reason):
            "controller job \(jobID) could not be verified as this site's live "
                + "controller (\(reason)) — it was NOT recorded"
        case .sshLoginDropped(let siteID, let host, let expectedUser, let source):
            "the ssh destination for '\(siteID)' would be saved as '\(host)', "
                + "with no login half, but this site logs in as "
                + "'\(expectedUser)' (\(source)) — `ssh \(host)` would "
                + "authenticate as the LOCAL account, which an SSO cluster "
                + "reports as an unenrolled user; the registry write is refused"
        }
    }
}
