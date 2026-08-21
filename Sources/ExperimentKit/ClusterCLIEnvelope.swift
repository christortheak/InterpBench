import Foundation

// =============================================================================
// The stable machine protocol (CLUSTER-CLI-LIFECYCLE-PLAN §6.5) — Phase C.
//
// One versioned envelope for every `cluster` verb. In `--json` mode stdout
// carries EXACTLY ONE of these documents and nothing else; human diagnostics go
// to stderr; there are no ANSI sequences anywhere.
//
// This is a SERIALIZATION VIEW of `ClusterLifecycleResult`, not a second source
// of truth: every lifecycle field is copied from the result, and the envelope
// adds only what a command line has that a lifecycle result does not (which
// verb ran, a rendered argv, a registry listing, a typed refusal).
//
// The security rule is structural rather than editorial: there is no property
// on this type, or on any type it nests, that can hold a credential. The bearer
// token appears as `tokenAvailable` (a Bool) and `tokenSource` (a provenance
// label). Command fields are argv ARRAYS, and the only commands rendered are
// ones the lifecycle composes itself — never a string a caller supplied.
// =============================================================================

public struct ClusterCLIEnvelope: Encodable, Sendable, Equatable {

    // MARK: Nested payloads

    /// The exact next command's shape. Mirrors
    /// `ClusterLifecycleResult.NextAction`.
    public struct NextAction: Encodable, Sendable, Equatable {
        public var verb: String
        public var requiresHuman: Bool
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

    /// One independently observed lifecycle layer (§7.2). Deliberately a list
    /// of layers rather than a `connected` Boolean.
    public struct Layer: Encodable, Sendable, Equatable {
        public var layer: String
        public var state: String
    }

    /// One planned transition, with why it is required and how it is gated.
    public struct PlanStep: Encodable, Sendable, Equatable {
        public var step: String
        public var reason: String
        /// `alreadySatisfied` / `readOnly` / `humanGated` / `approvalGated` /
        /// `asynchronous` / `resourceConsuming`.
        public var gating: [String]
        public var satisfied: Bool
        /// The flag that would authorize it, when it is approval-gated.
        public var requiredPermissionFlags: [String]
    }

    /// One saved site. Non-secret by construction.
    public struct Site: Encodable, Sendable, Equatable {
        public var id: String
        public var name: String
        public var transport: String
        public var topology: String
        public var scheduler: String
        public var lastEndpoint: String?
        public var lastServerBuild: String?
        /// PRESENCE only — never the token, never its length.
        public var tokenAvailable: Bool
    }

    /// `cluster import` (open-issues §20), as data. Names only — no remote
    /// path, no host, no byte of a credential — so a machine caller gets the
    /// same facts the human report prints.
    public struct ImportSummary: Encodable, Sendable, Equatable {
        public var dryRun: Bool
        /// Run directories imported (or, in a dry run, that would be).
        public var imported: [String]
        public var alreadyComplete: [String]
        /// Skipped by the policy: shard partials, library subtrees, and
        /// directories outside `--since`.
        public var skippedByPolicy: [String]
        /// Shapes the policy does not recognize. Imported conservatively.
        public var unknownShapes: [String]
        /// Shard-partial families a merge is EVIDENCED for, and so may be
        /// dropped from cluster scratch. A report, never an action.
        public var purgeEligible: [String]
        /// Families the evidence gate refused to call eligible: orphans, and
        /// merged-looking runs with no completeness stamp.
        public var purgeBlocked: [String]
        /// Studies whose live manifest holds fewer arms than their own
        /// imported run evidence — cluster authoring that never came home
        /// (§8 residual (a)). A report; the import never writes experiments/.
        public var authoringDivergences: [String]
        /// Immutability violations and verification failures, in full.
        public var violations: [String]
        /// Run directories the rebuilt catalog covers.
        public var catalogRuns: Int?
    }

    /// A typed refusal: stable code, plain-language reason, concrete repair.
    public struct Failure: Encodable, Sendable, Equatable {
        public var code: String
        public var reason: String
        public var repairAction: String
    }

    /// One durable operation record, summarized for `diagnose`.
    public struct Operation: Encodable, Sendable, Equatable {
        public var operationID: String
        public var state: String
        public var target: String
        public var startedAt: Date
        public var updatedAt: Date
        public var authorizedMutations: [String]
        public var controllerJobID: String?
        public var bootstrapJobID: String?
        public var failureCode: String?
        public var repairAction: String?
    }

    // MARK: Required common fields (§6.5)

    public let schemaVersion: Int
    /// Which verb produced this document, e.g. `cluster ensure`.
    public var verb: String
    public var state: String
    public var changed: Bool
    public var observedAt: Date
    public var message: String

    // MARK: Optional fields

    public var operationID: String?
    public var siteID: String?
    public var siteName: String?
    public var target: String?
    public var step: String?
    public var retryAfterSeconds: Int?
    public var nextAction: NextAction?
    public var endpoint: String?
    public var tokenAvailable: Bool?
    public var tokenSource: String?
    public var serverBuild: String?
    public var schedulerJobID: String?
    public var schedulerState: String?
    public var layers: [Layer]?
    public var plan: [PlanStep]?
    public var blockers: [String]?
    /// Non-blocking findings a reader must be told about even at `ready`
    /// (open-issues §1 field report, 2026-08-20). `blockers` stop work;
    /// these do not — a stale rendered controller script serves fine, it just
    /// cannot chain, and the operator has to learn that BEFORE the walltime
    /// rather than from a missing successor.
    public var advisories: [String]?
    public var sites: [Site]?
    /// argv ARRAY, never a shell string (§6.5).
    public var command: [String]?
    /// The reviewed bootstrap plan's hash, for `bootstrap apply --plan-hash`.
    public var planHash: String?
    /// Where a log lives on the far side — a path to read, not its contents.
    public var logPath: String?
    /// A file this verb wrote locally (`sites export`).
    public var outputPath: String?
    public var operations: [Operation]?
    /// `cluster preview` (WP5 §3.3): the complete generated environment and
    /// scheduler commands, so a caller can READ what will run before it runs.
    /// Rendered from the saved profile — no remote command, no clock, and no
    /// secret value (the token is the `$(cat …)` indirection the env file
    /// carries).
    public var preview: ClusterSitePreview?
    /// `cluster import`'s report.
    public var importSummary: ImportSummary?
    public var error: Failure?

    // MARK: Construction

    public init(
        verb: String,
        state: ClusterLifecycleState,
        message: String,
        changed: Bool = false,
        observedAt: Date = Date()
    ) {
        self.schemaVersion = ClusterLifecycleResult.schemaVersion
        self.verb = verb
        self.state = state.rawValue
        self.message = message
        self.changed = changed
        self.observedAt = observedAt
    }

    /// The exit code a shell caller sees (plan §6.5). Parsed back from the
    /// state so the envelope and the process agree by construction.
    public var exitCode: Int32 {
        ClusterLifecycleState(rawValue: state)?.exitCode ?? 70
    }

    // MARK: Lifecycle projection

    /// Copy every lifecycle field out of a `ClusterLifecycleResult`.
    public static func lifecycle(
        verb: ClusterCLIVerb, result: ClusterLifecycleResult
    ) -> ClusterCLIEnvelope {
        var envelope = ClusterCLIEnvelope(
            verb: verb.displayName, state: result.state, message: result.message,
            changed: result.changed, observedAt: result.observedAt)
        envelope.operationID = result.operationID
        envelope.siteID = result.siteID
        envelope.siteName = result.siteName
        envelope.target = result.target.cliName
        envelope.step = result.step?.rawValue
        envelope.retryAfterSeconds = result.retryAfterSeconds
        envelope.nextAction = result.nextAction.map {
            NextAction(
                verb: $0.verb, requiresHuman: $0.requiresHuman,
                missingPermissionFlags: $0.missingPermissionFlags, detail: $0.detail)
        }
        envelope.endpoint = result.endpoint
        envelope.tokenAvailable = result.tokenAvailable
        envelope.tokenSource = result.tokenSource
        envelope.serverBuild = result.serverBuild
        envelope.schedulerJobID = result.schedulerJobID
        envelope.schedulerState = result.schedulerState
        envelope.attach(observed: result.observed)
        envelope.attach(plan: result.plan)
        return envelope
    }

    public mutating func attach(observed: ClusterObservedState) {
        siteID = siteID ?? observed.siteID
        siteName = siteName ?? observed.siteName
        layers = observed.layerSummaries.map { Layer(layer: $0.layer, state: $0.state) }
        let findings = observed.advisories
        if !findings.isEmpty { advisories = (advisories ?? []) + findings }
    }

    public mutating func attach(plan lifecyclePlan: ClusterLifecyclePlan) {
        plan = lifecyclePlan.transitions.map { transition in
            PlanStep(
                step: transition.step.rawValue,
                reason: transition.reason,
                gating: transition.gating.labels,
                satisfied: transition.isSatisfied,
                requiredPermissionFlags: transition.gating.requiredPermission?.flags ?? [])
        }
        blockers = lifecyclePlan.blockers.isEmpty ? nil : lifecyclePlan.blockers
        target = target ?? lifecyclePlan.target.cliName
    }

    // MARK: Failure projection

    public static func failure(
        verb: String, code: String, reason: String, repairAction: String,
        state: ClusterLifecycleState = .failed, siteID: String? = nil
    ) -> ClusterCLIEnvelope {
        var envelope = ClusterCLIEnvelope(
            verb: verb, state: state, message: reason)
        envelope.siteID = siteID
        envelope.error = Failure(
            code: code, reason: reason, repairAction: repairAction)
        return envelope
    }

    // MARK: Encoding

    /// The one JSON document, newline-terminated. Sorted keys so a golden
    /// comparison is stable; ISO-8601 dates so a caller never parses a float.
    public func jsonText() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(self)
        return (String(data: data, encoding: .utf8) ?? "{}") + "\n"
    }
}

// MARK: - Human rendering

/// The non-JSON view of the same envelope. Plain text, no ANSI: the CLI is read
/// by people in terminals that may not be terminals at all.
public enum ClusterCLIRenderer {

    public static func humanText(_ envelope: ClusterCLIEnvelope) -> String {
        var lines: [String] = []
        let site = [envelope.siteName, envelope.siteID.map { "(\($0))" }]
            .compactMap { $0 }.joined(separator: " ")
        var header = "[\(envelope.state)] \(envelope.verb)"
        if !site.isEmpty { header += " — \(site)" }
        lines.append(header)
        lines.append("  \(envelope.message)")

        if let sites = envelope.sites {
            if sites.isEmpty {
                lines.append("  no saved cluster sites")
            }
            for entry in sites {
                lines.append(
                    "  \(entry.id)\t\(entry.name)\t\(entry.transport)"
                        + "\t\(entry.topology)/\(entry.scheduler)")
                let endpoint = entry.lastEndpoint ?? "not connected"
                lines.append(
                    "    endpoint: \(endpoint)"
                        + "  token: \(entry.tokenAvailable ? "in keychain" : "absent")"
                        + (entry.lastServerBuild.map { "  build: \($0)" } ?? ""))
            }
        }
        // The preview's panes are printed VERBATIM and unindented: an admin
        // copies the env file out of the terminal, and a leading two spaces
        // would corrupt every line of it.
        if let preview = envelope.preview { lines += preview.humanLines }
        if let layers = envelope.layers {
            lines.append("  layers:")
            for layer in layers { lines.append("    \(layer.layer): \(layer.state)") }
        }
        if let plan = envelope.plan, !plan.isEmpty {
            lines.append("  plan → \(envelope.target ?? "?"):")
            for step in plan {
                let mark = step.satisfied ? "✓" : "•"
                var line = "    \(mark) \(step.step): \(step.reason)"
                if !step.gating.isEmpty {
                    line += "  [\(step.gating.joined(separator: ","))]"
                }
                lines.append(line)
            }
        }
        for blocker in envelope.blockers ?? [] {
            lines.append("  BLOCKER: \(blocker)")
        }
        for advisory in envelope.advisories ?? [] {
            lines.append("  ADVISORY: \(advisory)")
        }
        if let operations = envelope.operations, !operations.isEmpty {
            lines.append("  recent operations:")
            for operation in operations {
                lines.append(
                    "    \(operation.operationID)  \(operation.state)"
                        + "  → \(operation.target)"
                        + (operation.controllerJobID.map { "  controller \($0)" } ?? ""))
                if let repair = operation.repairAction {
                    lines.append("      repair: \(repair)")
                }
            }
        }
        if let command = envelope.command {
            // Rendered for a HUMAN to run in their own Terminal (§4.1) — the
            // agent may print it and may not enrich it.
            lines.append("  command: \(command.joined(separator: " "))")
        }
        if let planHash = envelope.planHash {
            lines.append("  plan hash: \(planHash)")
        }
        if let logPath = envelope.logPath {
            lines.append("  log: \(logPath)")
        }
        if let outputPath = envelope.outputPath {
            lines.append("  wrote \(outputPath)")
        }
        if let endpoint = envelope.endpoint {
            var line = "  endpoint: \(endpoint)"
            if let source = envelope.tokenSource { line += "  token: \(source)" }
            if let build = envelope.serverBuild { line += "  build: \(build)" }
            lines.append(line)
        }
        if let jobID = envelope.schedulerJobID {
            lines.append(
                "  scheduler job \(jobID) \(envelope.schedulerState ?? "state unknown")")
        }
        if let next = envelope.nextAction {
            var line = "  next: \(next.verb)"
            if !next.missingPermissionFlags.isEmpty {
                line += " \(next.missingPermissionFlags.joined(separator: " "))"
            }
            if next.requiresHuman { line += "  (a human must authenticate)" }
            lines.append(line)
            if let detail = next.detail { lines.append("    \(detail)") }
        }
        if let retry = envelope.retryAfterSeconds {
            lines.append("  retry in \(retry)s — repeat the same command")
        }
        if let error = envelope.error {
            lines.append("  error [\(error.code)]: \(error.reason)")
            lines.append("  repair: \(error.repairAction)")
        }
        return lines.joined(separator: "\n") + "\n"
    }
}

// MARK: - Redaction

/// `diagnose --redact` (plan §6.1): a report that can be pasted into an issue.
///
/// Deliberately conservative — it removes the two things a shared cluster
/// report leaks by accident (the account name and the home-directory path),
/// and nothing else, so the report stays diagnostic.
public enum ClusterCLIRedaction {

    /// Replace `user@host` with `<user>@host` and absolute home paths with
    /// `<home>`, in any string.
    public static func redact(_ text: String) -> String {
        var out = redactUsers(in: text)
        out = redactHomePaths(in: out)
        return out
    }

    static func redactUsers(in text: String) -> String {
        var out = ""
        var pendingWord = ""
        for character in text {
            if character.isWhitespace {
                out += collapseUser(pendingWord)
                pendingWord = ""
                out.append(character)
            } else {
                pendingWord.append(character)
            }
        }
        out += collapseUser(pendingWord)
        return out
    }

    private static func collapseUser(_ word: String) -> String {
        guard let atIndex = word.firstIndex(of: "@"), atIndex != word.startIndex else {
            return word
        }
        return "<user>" + word[atIndex...]
    }

    static func redactHomePaths(in text: String) -> String {
        var out = text
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        if !home.isEmpty, home != "/" {
            out = out.replacingOccurrences(of: home, with: "<home>")
        }
        // Remote homes follow the same shape and are just as identifying.
        for root in ["/home/", "/Users/", "/scratch/", "/work/"] {
            out = redactFirstPathComponent(after: root, in: out)
        }
        return out
    }

    /// `/home/someone/x` → `/home/<user>/x`, leaving the rest of the path
    /// intact so the report still says WHERE something is.
    private static func redactFirstPathComponent(
        after root: String, in text: String
    ) -> String {
        var out = ""
        var remainder = Substring(text)
        while let range = remainder.range(of: root) {
            out += remainder[..<range.upperBound]
            let tail = remainder[range.upperBound...]
            let end = tail.firstIndex { $0 == "/" || $0.isWhitespace } ?? tail.endIndex
            if end > tail.startIndex { out += "<user>" }
            remainder = tail[end...]
        }
        out += remainder
        return out
    }
}
