import Foundation
import Observation

// MARK: - Substrate routing (WS6.3 — one Run affordance)

/// Pure decision logic behind the unified Run control's substrate picker.
/// The view renders exactly what `decide` returns — every rule here is
/// unit-tested, none of it lives in SwiftUI:
///
/// - The picker defaults to the ACTIVE scope (substrate-as-workspace).
/// - A stochastic design (declared `temperature > 0` or `samplesPerItem > 1`)
///   pins the selection to the server with an explanatory note — stochastic
///   measured runs are server-only by design (the MLX generator has no
///   per-run sampling seed), and the UI enforces that kindly up front,
///   never as an error later.
/// - With no connected cluster site, the server option greys out with a
///   "connect first" hint instead of failing at submit time.
public enum SubstrateRouting {

    public enum Substrate: String, CaseIterable, Sendable, Hashable {
        case thisMac
        case server
    }

    public struct Inputs: Sendable, Equatable {
        /// The MANIFEST's declared temperature — the pinned study design,
        /// not the draft field.
        public var temperature: Double?
        public var samplesPerItem: Int?
        /// The study's declared outcome instruments (E1). Routing depends on
        /// the resolved execution PLAN, not on a sampling field that may be
        /// inert: a logprob-only study never samples, so a declared
        /// temperature decides nothing for it and must not move the study to
        /// another substrate.
        public var outcomeInstruments: [String]?
        public var activeWorkspaceIsServer: Bool
        /// At least one cluster site/server is registered.
        public var siteRegistered: Bool
        public var siteName: String?
        /// The active site is connected (capabilities fetched) — implies the
        /// server workspace is active, since connection state is scoped to it.
        public var serverConnected: Bool
        /// The user's manual picker override; nil follows the active scope.
        public var userSelection: Substrate?

        public init(
            temperature: Double? = nil,
            samplesPerItem: Int? = nil,
            outcomeInstruments: [String]? = nil,
            activeWorkspaceIsServer: Bool = false,
            siteRegistered: Bool = false,
            siteName: String? = nil,
            serverConnected: Bool = false,
            userSelection: Substrate? = nil
        ) {
            self.temperature = temperature
            self.samplesPerItem = samplesPerItem
            self.outcomeInstruments = outcomeInstruments
            self.activeWorkspaceIsServer = activeWorkspaceIsServer
            self.siteRegistered = siteRegistered
            self.siteName = siteName
            self.serverConnected = serverConnected
            self.userSelection = userSelection
        }
    }

    public struct Decision: Sendable, Equatable {
        /// Effective selection after the rules run.
        public var selection: Substrate
        /// The design pinned the selection to the server (picker disabled).
        public var pinnedToServer: Bool
        /// Label for the server arm of the picker (site name when known).
        public var serverLabel: String
        /// Whether the server arm is selectable (registered AND connected).
        public var serverSelectable: Bool
        /// One-line explanation shown when a stochastic design auto-selects
        /// the server. Informational — never an error.
        public var stochasticNote: String?
        /// Hint on the greyed server arm ("connect first").
        public var serverHint: String?
        /// Hint when local execution needs the Local scope active.
        public var localHint: String?
        /// Why Run is disabled right now; nil means runnable.
        public var runBlockedReason: String?
    }

    /// A study whose SAMPLING is both stochastic and operative.
    ///
    /// `outcomeInstruments` decides whether sampling is operative at all
    /// (E1). Before this, a logprob-only study was pinned to the server by a
    /// temperature its instrument ignores — the rule read the number without
    /// asking whether anything downstream reads it. Passing nil instruments
    /// preserves the historical answer exactly (nil resolves to sampled
    /// text, which is what the engine default has always been).
    public static func isStochastic(
        temperature: Double?, samplesPerItem: Int?,
        outcomeInstruments: [String]? = nil
    ) -> Bool {
        guard ExecutionPlan.resolve(instruments: outcomeInstruments)
            .samplingIsOperative
        else { return false }
        return (temperature ?? 0) > 0 || (samplesPerItem ?? 1) > 1
    }

    public static func decide(_ inputs: Inputs) -> Decision {
        let stochastic = isStochastic(
            temperature: inputs.temperature, samplesPerItem: inputs.samplesPerItem,
            outcomeInstruments: inputs.outcomeInstruments)
        let trimmedName = inputs.siteName?.trimmingCharacters(in: .whitespacesAndNewlines)
        let serverLabel = (trimmedName?.isEmpty == false ? trimmedName : nil) ?? "Server"
        let serverSelectable = inputs.siteRegistered && inputs.serverConnected

        var selection: Substrate
        if stochastic {
            selection = .server
        } else {
            selection =
                inputs.userSelection
                ?? (inputs.activeWorkspaceIsServer ? .server : .thisMac)
        }

        var decision = Decision(
            selection: selection,
            pinnedToServer: stochastic,
            serverLabel: serverLabel,
            serverSelectable: serverSelectable,
            stochasticNote: nil,
            serverHint: nil,
            localHint: nil,
            runBlockedReason: nil)

        if stochastic {
            decision.stochasticNote =
                "stochastic design (temperature > 0 or samplesPerItem > 1) runs on "
                + "the server by design — local measured runs are greedy-only, so "
                + "the substrate is preselected"
        }
        if !serverSelectable {
            decision.serverHint =
                inputs.siteRegistered
                ? "connect first — pick \(serverLabel) in the toolbar connection dot"
                : "add a cluster site in the toolbar connection dot first"
        }

        switch decision.selection {
        case .server:
            if !serverSelectable {
                decision.runBlockedReason =
                    inputs.siteRegistered
                    ? "connect to \(serverLabel) first (toolbar connection dot)"
                    : "no cluster site is registered — add one in the toolbar "
                        + "connection dot"
            }
        case .thisMac:
            if inputs.activeWorkspaceIsServer {
                // Substrate is a scope: a local run needs the Local scope so
                // its artifacts land where the rest of the app is looking.
                decision.localHint =
                    "switch Compute to Local (MLX) to run on this Mac — the "
                    + "substrate selector scopes where runs land"
                decision.runBlockedReason = decision.localHint
            }
        }
        return decision
    }

    /// Study verbs that execute the study's condition matrix — the only
    /// submissions where the sampling-policy contract between baseline and
    /// saved agents matters (verify/extract/validate/sweep/evaluate never
    /// run the matrix).
    static let matrixExecutingVerbs: Set<String> = ["run", "pipeline"]

    /// Old-server guard for study-owned sampling (2026-07-21): a STOCHASTIC
    /// study (temperature > 0 or samplesPerItem > 1) containing saved-agent
    /// (variant) conditions must not be submitted to a server that predates
    /// the `variantStudySampling` capability — such a server runs the
    /// agents one greedy path each while the baseline samples N draws, a
    /// silently unbalanced design. Returns the plain-language refusal, or
    /// nil when the submission is fine (deterministic design, no saved
    /// agents, non-matrix verb, or a server that has the capability).
    /// Pure and unit-tested; both bundle-submit paths call it before
    /// packaging.
    public static func stochasticVariantSubmissionRefusal(
        temperature: Double?,
        samplesPerItem: Int?,
        variantConditionCount: Int,
        verb: String,
        capabilities: ClusterCapabilities?
    ) -> String? {
        guard matrixExecutingVerbs.contains(verb),
            variantConditionCount > 0,
            isStochastic(temperature: temperature, samplesPerItem: samplesPerItem),
            capabilities?.supportsVariantStudySampling != true
        else { return nil }
        return "this server predates study-owned sampling for saved agents — "
            + "update the server, or the agents would run greedy while the "
            + "baseline samples"
    }

    /// What the ONE primary button says it will do.
    public static func runButtonLabel(
        decision: Decision, verb: String, dryRun: Bool, isBusy: Bool
    ) -> String {
        if isBusy { return "Running…" }
        switch decision.selection {
        case .thisMac:
            return "Run Study"
        case .server:
            let mode = dryRun ? "\(verb) (dry run)" : verb
            return "Run on \(decision.serverLabel): \(mode)"
        }
    }
}

// MARK: - Freeze routing (substrate-consistent lifecycle)

/// Pure decision logic behind the Freeze control's substrate routing.
/// Freeze is a lifecycle action; where it executes follows the WORKSPACE
/// relationship, not merely the Compute selection:
///
/// - PAIRED server workspace (one shared tree): the server performs the
///   freeze — its gates are the authority over the same files, and the
///   button says so unmistakably.
/// - KNOWN-UNPAIRED server Compute (Mac-authority mode, 2026-07-21 — the
///   researcher's real cluster configuration): the Mac workspace is the
///   source of truth for study parameters, so Freeze runs LOCALLY with the
///   full local gate set; the frozen study travels to the server as a
///   hash-pinned bundle. Validation evidence for the freeze gate is
///   matched against the RUN substrate (the server), not the freeze
///   location — see `ExperimentStore.freeze(runSubstrate:)`.
/// - Unknown pairing keeps the server route (the identity precheck is the
///   backstop), as does the browser workbench, which never sees this rule.
///
/// Every rule here is unit-tested; none lives in SwiftUI.
public enum FreezeRouting {

    public enum Target: String, Sendable, Equatable {
        case thisMac
        case server
    }

    public struct Inputs: Sendable, Equatable {
        public var activeWorkspaceIsServer: Bool
        /// The active site is connected (capabilities fetched).
        public var serverConnected: Bool
        /// Display label for the active server (site name / host).
        public var serverLabel: String
        /// Whether the study exists in the server's own experiments/ tree.
        /// nil = unknown (listing failed / not yet checked) — never blocks;
        /// the server's own refusal is the backstop.
        public var serverHasSelectedStudy: Bool?
        /// The server is KNOWN to serve a different tree than this
        /// workspace (`ServerPairing.unpaired` or `.remoteAuthoritative`).
        /// False for paired AND for unknown pairing — Mac-authority routing
        /// requires a confirmed answer, never a guess.
        public var workspaceKnownUnpaired: Bool

        public init(
            activeWorkspaceIsServer: Bool = false,
            serverConnected: Bool = false,
            serverLabel: String = "server",
            serverHasSelectedStudy: Bool? = nil,
            workspaceKnownUnpaired: Bool = false
        ) {
            self.activeWorkspaceIsServer = activeWorkspaceIsServer
            self.serverConnected = serverConnected
            self.serverLabel = serverLabel
            self.serverHasSelectedStudy = serverHasSelectedStudy
            self.workspaceKnownUnpaired = workspaceKnownUnpaired
        }
    }

    public struct Decision: Sendable, Equatable {
        /// Which substrate the freeze click executes on.
        public var target: Target
        /// The button label — the executing substrate is unmistakable.
        public var buttonLabel: String
        /// The one-way confirmation dialog's destructive-button label.
        public var confirmLabel: String
        /// One-line note under the button naming where the gates are
        /// evaluated (nil for a plain local freeze).
        public var executorNote: String?
        /// Why the freeze cannot proceed right now (server route while
        /// disconnected, or study not server-resident); nil = clickable.
        public var blockedReason: String?
    }

    public static func decide(_ inputs: Inputs) -> Decision {
        guard inputs.activeWorkspaceIsServer else {
            return Decision(
                target: .thisMac,
                buttonLabel: "Freeze Study…",
                confirmLabel: "Freeze",
                executorNote: nil,
                blockedReason: nil)
        }
        let trimmed = inputs.serverLabel.trimmingCharacters(in: .whitespacesAndNewlines)
        let label = trimmed.isEmpty ? "server" : trimmed
        if inputs.workspaceKnownUnpaired {
            // Mac-authority mode: this workspace is the source of truth —
            // freeze HERE, with the full local gate set; the frozen study
            // travels to the runner as a bundle. No server identity check,
            // no draft sync, no connection requirement. The label says
            // where the stamp lands so the routing change is legible.
            return Decision(
                target: .thisMac,
                buttonLabel: "Freeze (in this workspace)…",
                confirmLabel: "Freeze",
                executorNote: "unpaired server Compute — the freeze runs in THIS "
                    + "workspace (Mac authority): all local gates apply, and the "
                    + "validate-evidence gate matches evidence for the RUN "
                    + "substrate (\(label)), e.g. an imported server validate "
                    + "run. Submit the frozen study to \(label) as a bundle to "
                    + "run it",
                blockedReason: nil)
        }
        var blocked: String?
        if !inputs.serverConnected {
            blocked = "connect to \(label) first (toolbar connection dot)"
        } else if inputs.serverHasSelectedStudy == false {
            blocked = "study is not in \(label)'s workspace — freeze stamps the "
                + "server-resident copy only. Pair the server to this workspace "
                + "(serve --root <workspace>), or switch Compute to Local (MLX) "
                + "to freeze the local copy"
        }
        return Decision(
            target: .server,
            buttonLabel: "Freeze (on \(label))…",
            confirmLabel: "Freeze on \(label)",
            executorNote: "gates are evaluated by \(label) against SERVER-substrate "
                + "evidence and the manifest is stamped frozenBy: \"server\" — "
                + "validation evidence counts on the substrate that freezes",
            blockedReason: blocked)
    }

    // MARK: Button enablement (server gates decide remote readiness)

    /// Enablement rule for the ONE freeze button: local verification
    /// failures gate only a LOCAL freeze. When the freeze routes to the
    /// server, the SERVER's own gates are the authority — it verifies its
    /// own copy's pins at the click — so local violations render as
    /// informational context, never as a disabled button (a stale local
    /// copy must not block freezing a valid server copy, and vice versa the
    /// server's refusal surfaces verbatim).
    public static func freezeButtonDisabled(
        decision: Decision, hasLocalViolations: Bool
    ) -> Bool {
        if decision.blockedReason != nil { return true }
        return decision.target == .thisMac && hasLocalViolations
    }

    /// The informational line shown when a SERVER-routed freeze coexists
    /// with local verification failures (nil when there are none).
    public static func localViolationsContextNote(
        count: Int, serverLabel: String
    ) -> String? {
        guard count > 0 else { return nil }
        let trimmed = serverLabel.trimmingCharacters(in: .whitespacesAndNewlines)
        let label = trimmed.isEmpty ? "the server" : trimmed
        return "local verification reports \(count) issue\(count == 1 ? "" : "s") "
            + "— informational for a server freeze: \(label) verifies ITS copy's "
            + "pins and gates at the click"
    }

    // MARK: Remote-freeze manifest identity (the display/act divergence guard)

    /// What the pre-freeze identity check learned about the server's
    /// same-named manifest versus the manifest ON SCREEN. Produced by the
    /// panel (fetch + `ExperimentStore.compareManifestDocuments`), decided
    /// here so the block/proceed rule is pure and unit-tested.
    public enum RemoteManifestIdentity: Sendable, Equatable {
        /// Server body equals the local document modulo the volatile
        /// freeze-stamp keys.
        case verifiedEqual
        /// The documents differ — field-level mismatch lines attached.
        case mismatch([String])
        /// No local manifest exists (server-only study): the server copy's
        /// reported status and its canonicalized-body hash, so the
        /// researcher confirms against something real.
        case localMissing(serverStatus: String?, canonicalBodyHash: String?)
        /// The server body could not be fetched or parsed (older server
        /// without the manifest route, transient error) — reason attached.
        case unverifiable(String)
    }

    public struct RemoteFreezePrecheck: Sendable, Equatable {
        /// False = the freeze must not be submitted.
        public var proceed: Bool
        /// Blocked: the mismatch summary + remedy. Proceeding: the
        /// server-only-copy statement or the could-not-verify note (nil when
        /// verified equal — nothing to say).
        public var message: String?

        public init(proceed: Bool, message: String? = nil) {
            self.proceed = proceed
            self.message = message
        }
    }

    /// The block/proceed rule for a SERVER-routed freeze, given what the
    /// identity check learned:
    ///
    /// - verified equal → proceed silently.
    /// - mismatch → BLOCK: the server would freeze a document that is not
    ///   the one on screen. The message carries the field-level summary and
    ///   the remedy.
    /// - no local copy → proceed, but say explicitly that the server's copy
    ///   is the ONLY copy (name, status, canonicalized-body hash).
    /// - unverifiable → on a PAIRED workspace the server serves this same
    ///   tree, so residency + the server's own gates suffice: proceed with
    ///   an informational note. On an UNPAIRED workspace the identity
    ///   question is exactly what could not be answered: BLOCK.
    public static func remoteFreezePrecheck(
        identity: RemoteManifestIdentity,
        study: String,
        serverLabel: String,
        workspacePaired: Bool
    ) -> RemoteFreezePrecheck {
        let trimmed = serverLabel.trimmingCharacters(in: .whitespacesAndNewlines)
        let label = trimmed.isEmpty ? "the server" : trimmed
        switch identity {
        case .verifiedEqual:
            return RemoteFreezePrecheck(proceed: true)
        case .mismatch(let fields):
            let summary = fields.isEmpty
                ? "(no top-level field summary available)"
                : fields.map { "• \($0)" }.joined(separator: "\n")
            return RemoteFreezePrecheck(
                proceed: false,
                message: "freeze refused before submission: \(label)'s copy of "
                    + "'\(study)' is NOT the manifest you are looking at. "
                    + "Differing fields:\n\(summary)\n"
                    + "Pair the workspace / sync the study — the server's copy "
                    + "differs from what you're looking at.")
        case .localMissing(let serverStatus, let canonicalBodyHash):
            let status = serverStatus ?? "unknown"
            let hash = canonicalBodyHash.map { "\($0.prefix(12))…" } ?? "unavailable"
            return RemoteFreezePrecheck(
                proceed: true,
                message: "no local copy of '\(study)' exists — \(label)'s copy is "
                    + "the ONLY copy (status \(status), canonical body @ \(hash)); "
                    + "confirm that is the study you mean")
        case .unverifiable(let reason):
            if workspacePaired {
                return RemoteFreezePrecheck(
                    proceed: true,
                    message: "could not verify \(label)'s copy of '\(study)' "
                        + "matches the displayed manifest (\(reason)) — "
                        + "proceeding: the paired workspace shares this tree "
                        + "and \(label)'s own gates verify its pins at freeze")
            }
            return RemoteFreezePrecheck(
                proceed: false,
                message: "freeze refused before submission: could not verify "
                    + "\(label)'s copy of '\(study)' matches the displayed "
                    + "manifest (\(reason)). On an unpaired workspace the "
                    + "displayed and frozen studies can differ — pair the "
                    + "workspace (serve --root <workspace>) or retry when the "
                    + "server is reachable.")
        }
    }

    // MARK: One-click server-draft sync (2026-07-21 incident, part 3)

    /// Whether the blocked precheck may offer "Update the server's copy":
    /// exactly the MISMATCH case (the server holds a same-named document
    /// that is not the one on screen — the machinery to replace a draft
    /// already exists server-side), and only for a LOCAL DRAFT. A frozen
    /// local manifest never pushes (duplicate-never-edit: frozen documents
    /// are stamped by a freeze authority, not synced), and the other
    /// blocked case (unverifiable + unpaired) has nothing trustworthy to
    /// push over.
    public static func canOfferServerDraftSync(
        identity: RemoteManifestIdentity, localIsDraft: Bool
    ) -> Bool {
        guard localIsDraft else { return false }
        if case .mismatch = identity { return true }
        return false
    }

    /// What the affordance says after pushing and RE-RUNNING the identity
    /// check: verified-equal, or the remaining difference — honestly.
    /// `resolved` means the mismatch is gone and freeze can proceed.
    public struct ServerDraftSyncOutcome: Sendable, Equatable {
        public var resolved: Bool
        public var message: String

        public init(resolved: Bool, message: String) {
            self.resolved = resolved
            self.message = message
        }
    }

    public static func serverDraftSyncOutcome(
        recheck identity: RemoteManifestIdentity,
        study: String,
        serverLabel: String,
        canonicalBodyHash: String?
    ) -> ServerDraftSyncOutcome {
        let trimmed = serverLabel.trimmingCharacters(in: .whitespacesAndNewlines)
        let label = trimmed.isEmpty ? "the server" : trimmed
        let hash = canonicalBodyHash.map { " (canonical body @ \($0.prefix(12))…)" } ?? ""
        switch identity {
        case .verifiedEqual:
            return ServerDraftSyncOutcome(
                resolved: true,
                message: "\(label)'s copy of '\(study)' now matches the "
                    + "manifest on screen\(hash) — verified equal; freeze can "
                    + "proceed")
        case .mismatch(let fields):
            let summary = fields.isEmpty
                ? "(no top-level field summary available)"
                : fields.map { "• \($0)" }.joined(separator: "\n")
            return ServerDraftSyncOutcome(
                resolved: false,
                message: "pushed, but \(label)'s copy of '\(study)' STILL "
                    + "differs from the manifest on screen:\n\(summary)\n"
                    + "Something else is writing that copy — do not freeze "
                    + "until the re-check verifies equal.")
        case .localMissing:
            return ServerDraftSyncOutcome(
                resolved: false,
                message: "pushed, but no local copy of '\(study)' exists to "
                    + "compare against — the sync affordance should not have "
                    + "been offered; nothing was verified")
        case .unverifiable(let reason):
            return ServerDraftSyncOutcome(
                resolved: false,
                message: "pushed\(hash), but the re-check could not verify "
                    + "\(label)'s copy of '\(study)' (\(reason)) — retry the "
                    + "freeze to re-run the identity check")
        }
    }

    // MARK: The advisory-surfacing rule (prominent at freeze-decision time)

    /// The one-line rule that explains WHY a cross-substrate validate
    /// advisory is promoted to a warning at freeze-decision time.
    public static let crossSubstrateRule =
        "validation evidence counts on the substrate that freezes — freeze "
        + "where you validated, or re-validate here"

    /// Recognizes the cross-substrate validate-evidence advisory. Both
    /// engines' wording ends in this clause
    /// (`ExperimentStore.crossSubstrateValidationAdvisory` /
    /// `experiment_store.cross_substrate_validation_advisory`), so one
    /// predicate covers Mac-freeze-with-server-evidence, the reverse, and
    /// the server's response advisories.
    public static func isCrossSubstrateAdvisory(_ advisory: String) -> Bool {
        advisory.contains("re-validate on-substrate")
    }

    /// How freeze advisories render at decision time: cross-substrate
    /// evidence advisories become PROMINENT (warning style, rule appended);
    /// everything else stays a regular info row.
    public struct AdvisoryPresentation: Sendable, Equatable {
        public var prominent: [String]
        public var regular: [String]

        public init(prominent: [String] = [], regular: [String] = []) {
            self.prominent = prominent
            self.regular = regular
        }
    }

    public static func present(advisories: [String]) -> AdvisoryPresentation {
        var presentation = AdvisoryPresentation()
        for advisory in advisories {
            if isCrossSubstrateAdvisory(advisory) {
                presentation.prominent.append("\(advisory) — \(crossSubstrateRule)")
            } else {
                presentation.regular.append(advisory)
            }
        }
        return presentation
    }
}

// MARK: - Preflight presentation (WS4 surfacing rules)

/// Maps a server `PreflightReport` onto exactly what the Run affordance
/// shows — the verdict rules are data interpretation, so they are pure and
/// unit-tested here, never re-derived in a view:
///
/// - verdict "ok"   → proceed silently (no inline summary, no lines).
/// - verdict "warn" → show the non-ok check messages inline (amber), proceed.
/// - verdict "fail" → the submission stopped; show the failing checks and
///   offer "Override (forced)" ONLY behind a confirmation restating them.
public struct PreflightPresentation: Sendable, Equatable {

    public enum Verdict: String, Sendable, Equatable {
        case ok
        case warn
        case fail

        public init(raw: String) {
            switch raw.lowercased() {
            case "fail", "failed": self = .fail
            case "warn", "warning": self = .warn
            default: self = .ok
            }
        }
    }

    public struct Line: Sendable, Equatable, Identifiable {
        public var id: String
        public var severity: Verdict
        public var message: String

        public init(id: String, severity: Verdict, message: String) {
            self.id = id
            self.severity = severity
            self.message = message
        }
    }

    public var verdict: Verdict
    public var lines: [Line]

    public init(verdict: Verdict, lines: [Line]) {
        self.verdict = verdict
        self.lines = lines
    }

    public static func from(_ report: PreflightReport?) -> PreflightPresentation? {
        guard let report else { return nil }
        let lines = report.checks.map { check in
            Line(
                id: check.id,
                severity: Verdict(raw: check.status),
                message: check.message)
        }
        return PreflightPresentation(verdict: Verdict(raw: report.verdict), lines: lines)
    }

    /// The failing-submission presentation mined from a refusal body, or nil
    /// when the error was not a preflight refusal (it must then surface as
    /// the transport/server error it is). Two shapes are recognized:
    /// a JSON body carrying a `"preflight"` report object, and the server's
    /// actual refusal string (`submissions._gate_on_preflight` raises
    /// `"preflight failed — id: message; id: message (pass force=true …)"`,
    /// which FastAPI wraps as a plain-string `detail`).
    public static func refusal(fromErrorBody body: String) -> PreflightPresentation? {
        if let report = ClusterClient.preflightReport(fromErrorBody: body) {
            let presentation = from(report)
            return presentation?.verdict == .fail ? presentation : nil
        }
        return refusal(fromDetailText: detailText(fromErrorBody: body))
    }

    /// FastAPI error bodies are `{"detail": "…"}` — unwrap when possible,
    /// else treat the body as the detail itself.
    static func detailText(fromErrorBody body: String) -> String {
        struct Detail: Decodable { var detail: String }
        if let data = body.data(using: .utf8),
            let decoded = try? JSONDecoder().decode(Detail.self, from: data)
        {
            return decoded.detail
        }
        return body
    }

    /// Parse the server's string-form refusal into failing lines. The shape
    /// (pinned by `Server/…/submissions.py`) is:
    /// `preflight failed — <id>: <message>; <id>: <message> (pass force=true
    /// to submit anyway, or dryRun=true to inspect the report …)`.
    static func refusal(fromDetailText text: String) -> PreflightPresentation? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let lowered = trimmed.lowercased()
        guard lowered.hasPrefix("preflight failed") else { return nil }
        var remainder = String(trimmed.dropFirst("preflight failed".count))
        // Strip the trailing force/dry-run hint (the client renders its own
        // override affordance).
        if let hint = remainder.range(of: "(pass force=true") {
            remainder = String(remainder[..<hint.lowerBound])
        }
        // Leading separator: " — " (the server's em-dash) or ": "/" - ".
        remainder = remainder.trimmingCharacters(in: .whitespacesAndNewlines)
        for separator in ["—", "–", "-", ":"] where remainder.hasPrefix(separator) {
            remainder = String(remainder.dropFirst(separator.count))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            break
        }
        let segments = remainder
            .split(separator: ";")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let lines = segments.enumerated().map { index, segment -> Line in
            if let colon = segment.range(of: ": ") {
                return Line(
                    id: String(segment[..<colon.lowerBound]),
                    severity: .fail,
                    message: String(segment[colon.upperBound...]))
            }
            return Line(id: "check-\(index + 1)", severity: .fail, message: segment)
        }
        guard !lines.isEmpty else {
            // A refusal with no parseable checks still blocks — show the
            // detail verbatim rather than dropping the affordance.
            return PreflightPresentation(
                verdict: .fail,
                lines: [Line(id: "preflight", severity: .fail, message: trimmed)])
        }
        return PreflightPresentation(verdict: .fail, lines: lines)
    }

    /// Checks worth showing inline for a warn verdict (everything non-ok).
    public var attentionLines: [Line] { lines.filter { $0.severity != .ok } }

    public var failingLines: [Line] { lines.filter { $0.severity == .fail } }

    public var blocksSubmission: Bool { verdict == .fail }

    /// nil for a clean verdict — "ok → proceed silently" is a rule, not a
    /// styling choice.
    public var inlineSummary: String? {
        switch verdict {
        case .ok:
            return nil
        case .warn:
            let count = attentionLines.count
            return "preflight passed with \(count) warning\(count == 1 ? "" : "s")"
        case .fail:
            let count = failingLines.count
            return "preflight FAILED \(count) check\(count == 1 ? "" : "s") — "
                + "submission refused"
        }
    }

    /// Body of the forced-override confirmation: restates every failing
    /// check verbatim so the override is informed, loud, and never silent.
    public var overrideConfirmationMessage: String {
        let restated = failingLines
            .map { "• \($0.message)" }
            .joined(separator: "\n")
        return "The server refused this submission because preflight failed:\n"
            + restated
            + "\n\nForce-submitting bypasses these checks. The job may run out "
            + "of memory, exceed walltime, or fill the quota exactly as "
            + "predicted."
    }
}

// MARK: - Unified study runner (WS6.3 — the one Run button's engine)

/// The unified Run control's submission engine. Local selection delegates to
/// the panel's existing local run path untouched; server selection owns the
/// submit-bundle flow so the WS4 preflight report is captured and surfaced
/// (the legacy panel path discards the submission response).
///
/// Deliberately built ONLY from public ExperimentKit API — packaging
/// (`RunBundlePackager`), upload/submit (`ClusterClient`), and job following
/// (`ExperimentPanel.reconnectRemoteJob` + `refreshRecentServerJobs`) — so
/// the view stays a renderer.
@Observable @MainActor
public final class UnifiedStudyRunner {

    public private(set) var isSubmitting = false
    public private(set) var statusLine: String?
    /// Preflight from the LAST successful submission (ok/warn — proceeded).
    public private(set) var preflight: PreflightPresentation?
    /// Preflight refusal from the last attempt (verdict fail — stopped).
    /// Non-nil is what makes the view offer "Override (forced)".
    public private(set) var refusal: PreflightPresentation?
    public private(set) var lastJobID: String?

    public init() {}

    /// Clear surfaced preflight state (selection changed, study switched).
    public func clearPreflight() {
        preflight = nil
        refusal = nil
        statusLine = nil
    }

    /// The one entry point behind the primary Run button.
    public func run(
        manifest: ExperimentManifest,
        decision: SubstrateRouting.Decision,
        panel: ExperimentPanel,
        cluster: ClusterConnectionStore,
        force: Bool = false
    ) async {
        guard decision.runBlockedReason == nil else {
            statusLine = decision.runBlockedReason
            return
        }
        switch decision.selection {
        case .thisMac:
            // Existing local path, untouched (greedy gate, live viewer,
            // display log all included).
            await panel.runStudy()
        case .server:
            await submitBundle(
                manifest: manifest, panel: panel, cluster: cluster, force: force)
        }
    }

    /// Package → upload → submit-bundle (the portable path), capturing the
    /// WS4 preflight report on success and mining it out of a refusal.
    public func submitBundle(
        manifest: ExperimentManifest,
        panel: ExperimentPanel,
        cluster: ClusterConnectionStore,
        force: Bool
    ) async {
        guard !isSubmitting else { return }
        cluster.loadStoredToken()
        guard let client = cluster.client else {
            statusLine = "invalid server URL"
            return
        }
        isSubmitting = true
        defer { isSubmitting = false }
        refusal = nil
        preflight = nil
        // Frozen-on-server guard — shared with the legacy panel path
        // (engineer finding 2026-07-19: the primary Run button previously
        // bypassed it): a local DRAFT must not shadow the server's frozen
        // same-named study.
        if let conflict = await client.frozenOnServerConflict(
            study: manifest.name, localStatus: manifest.status)
        {
            statusLine = conflict
            panel.note(conflict, severity: .error)
            return
        }
        let verb = panel.remoteVerb
        // Old-server guard (2026-07-21): a stochastic saved-agent study on a
        // server without study-owned sampling would run the agents greedy
        // while the baseline samples — refuse BEFORE packaging/uploading.
        if let refusal = SubstrateRouting.stochasticVariantSubmissionRefusal(
            temperature: manifest.temperature,
            samplesPerItem: manifest.samplesPerItem,
            variantConditionCount: manifest.variantConditions.count,
            verb: verb,
            capabilities: cluster.capabilities)
        {
            statusLine = refusal
            panel.note(refusal, severity: .error)
            return
        }
        // Scope-drift guard (2026-08-06 field incident): a stale
        // outcomeInstrumentScope pin after Duplicate & Adjust + task-file
        // swap refuses HERE, before packaging/upload — the server would
        // refuse the same way, but only on the compute node.
        if let refusal = ExperimentTasks.scopeDriftSubmitRefusal(
            for: manifest, verb: verb)
        {
            statusLine = refusal
            panel.note(refusal, severity: .error)
            return
        }
        let dryRun = panel.remoteDryRun
        let resources = [
            "gres": panel.remoteGres,
            "walltime": panel.remoteWalltime,
        ].filter { !$0.value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        do {
            statusLine = "packaging \(manifest.name)…"
            // Packaging copies files, hashes them, and shells out to tar;
            // keep it off the main actor (same rule as the legacy path).
            let bundle = try await Task.detached {
                try RunBundlePackager.packageExperiment(manifest)
            }.value
            statusLine = "uploading \(bundle.lastPathComponent)…"
            let uploaded = try await client.uploadBundle(bundle)
            statusLine = force
                ? "submitting \(verb) (forced past preflight)…"
                : "submitting \(verb)…"
            // Resume-on-checkpoint policy (2026-07-22 incident): Slurm
            // submissions carry the panel's toggle — default ON — and the
            // transcript line below stamps what was sent.
            let resumePolicy: RemoteResumePolicy? =
                panel.remoteExecutor == "slurm" ? panel.remoteResumePolicy : nil
            let submission = try await client.submitBundle(
                path: uploaded.path,
                verb: verb,
                executor: panel.remoteExecutor,
                dryRun: dryRun,
                resources: resources,
                resumePolicy: resumePolicy,
                force: force,
                parallelJobs: panel.remoteParallelJobs)
            preflight = PreflightPresentation.from(submission.preflight)
            lastJobID = submission.jobId
            let substrate = cluster.substrateLabel
            var submitted = ExperimentPanel.bundleSubmittedStatus(
                study: manifest.name, verb: verb, dryRun: dryRun,
                substrate: substrate, jobID: submission.jobId)
            if let resumePolicy {
                submitted += " — \(resumePolicy.transcriptStamp)"
            }
            // The sharding stamp derives from the server's RESPONSE (the
            // shard ids it actually created), never from the request — the
            // server may have ignored the fan-out (finding 5, 2026-07-22).
            if let stamp = ShardedSubmission.transcriptStamp(
                shardJobIDs: submission.shardJobIDs)
            {
                submitted += " — \(stamp)"
            }
            if let summary = preflight?.inlineSummary {
                submitted += " — \(summary)"
            }
            statusLine = submitted
            // Follow through the panel so the job log lands in the same
            // places the legacy flow used (remote log lines, reconnectable
            // job id), and the recent-jobs list learns about it.
            await panel.reconnectRemoteJob(submission.jobId)
            await panel.refreshRecentServerJobs()
        } catch let error as ClusterClient.ClientError {
            if case .badResponse(_, let body) = error,
                let refused = PreflightPresentation.refusal(fromErrorBody: body)
            {
                refusal = refused
                statusLine = refused.inlineSummary
                    ?? "submission refused by preflight — review the failing checks"
            } else {
                statusLine = "remote submit failed: \(ClusterClient.unwrappingDetail(error))"
            }
        } catch {
            statusLine = "remote submit failed: \(error.localizedDescription)"
        }
    }
}
