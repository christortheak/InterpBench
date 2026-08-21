// Declared record-exclusion rules — manifest data, applied at analysis.
//
// Real studies exclude records for declared reasons (failed attention
// checks, unparseable endpoints, off-scale values). This module makes those
// reasons DATA: the manifest declares `exclusionRules` (a closed rule
// vocabulary), task-prompt items may declare a per-item `attentionCheck`
// graded with the capability battery's pinned grading vocabulary, and the
// analysis layer joins the two — records are excluded from the paired
// statistics ONLY, never removed from generations.jsonl (runs are
// immutable), and the analysis stamps which rules were active, how many
// records each rule excluded per condition, and the surviving N. Honesty
// about exclusions is the point.
//
// Cross-engine contract with the server's `exclusions.py`: identical
// manifest/item JSON keys, identical rule vocabulary, identical violation
// message strings, and an identical exclusion-stamp shape (fixture-tested
// on both engines). The rules live in the manifest, so freeze pins them
// through the ordinary content hash — declared before behavior is measured,
// and adding a rule after a run changes the manifest hash, so the epoch
// guard refuses to analyze pre-declaration runs (the firewall, not an
// accident).

import Foundation

/// One declared exclusion rule (manifest key `exclusionRules`, an array of
/// these). `rule` is the closed vocabulary id; `endpoint` names the
/// record-level parsed-value key for the endpoint-reading rules (default
/// "parsedMonths"); `min`/`max` declare the kept range for `outOfRange`.
/// CodingKeys are closed and every optional is omit-when-nil, so legacy
/// manifests keep their content hash byte-for-byte.
public struct ExclusionRule: Codable, Sendable, Equatable {
    enum CodingKeys: String, CodingKey {
        case rule
        case endpoint
        case min
        case max
    }

    public var rule: String
    public var endpoint: String?
    public var min: Double?
    public var max: Double?

    public init(
        rule: String, endpoint: String? = nil, min: Double? = nil,
        max: Double? = nil
    ) {
        self.rule = rule
        self.endpoint = endpoint
        self.min = min
        self.max = max
    }
}

/// A task-prompt item's declared attention check (item key
/// `attentionCheck`): the expected answer plus an optional grading mode from
/// the capability battery's pinned vocabulary (inferred from the answer's
/// shape when absent — the battery's exact rule).
public struct AttentionCheck: Sendable, Equatable {
    public let expected: String
    public let grading: CapabilityBattery.GradingMode?
}

/// The analysis-time exclusion stamp (cross-engine shape; the server writes
/// the identical object as `exclusions.json`): resolved rules with
/// plain-language descriptions, per-condition considered/surviving record
/// counts, per-condition per-rule exclusion counts (zeros included —
/// honesty about what did NOT fire), the declared record-type scope, and
/// the pairwise-deletion note.
public struct ExclusionStamp: Codable, Sendable, Equatable {
    public struct ResolvedRule: Codable, Sendable, Equatable {
        public let rule: String
        public var endpoint: String? = nil
        public var min: Double? = nil
        public var max: Double? = nil
        /// How many task-prompt items declared an attentionCheck
        /// (failedAttentionCheck rule only).
        public var checkedItems: Int? = nil
        public let description: String
    }

    public let rules: [ResolvedRule]
    public let consideredN: [String: Int]
    public let excludedByRule: [String: [String: Int]]
    public let excludedRecords: Int
    public let survivingN: [String: Int]
    public let pairwiseDeletion: Bool
    /// Which record types the rules considered (cross-engine ids
    /// `ExclusionEngine.scopeAllRecordTypes` / `.scopeSampledRecords`).
    /// Optional ONLY so pre-scope legacy stamps still decode in the
    /// Results pane; every newly written stamp carries it.
    public var scope: String? = nil
    public let note: String
}

/// The concept-agnostic exclusion engine: rule validation (verify/preflight)
/// and record-level application (analysis). Server twin: `exclusions.py`.
public enum ExclusionEngine {
    public static let ruleFailedAttentionCheck = "failedAttentionCheck"
    public static let ruleUnparseableEndpoint = "unparseableEndpoint"
    public static let ruleOutOfRange = "outOfRange"

    /// The closed rule vocabulary (cross-engine contract).
    public static let ruleVocabulary = [
        ruleFailedAttentionCheck, ruleUnparseableEndpoint, ruleOutOfRange,
    ]

    /// Default record-level endpoint key for the endpoint-reading rules.
    public static let defaultEndpoint = "parsedMonths"

    /// The declared exclusion SCOPE, stamped so a reader never has to
    /// infer which record types the rules considered (cross-engine ids;
    /// server twins `SCOPE_ALL_RECORD_TYPES` / `SCOPE_SAMPLED_RECORDS`).
    /// `allRecordTypes` (analyze + run-inline report): sampled generations
    /// AND deterministic instrument readouts are considered — endpoint
    /// rules read any record that itself carries the named endpoint, and a
    /// cell whose every sampled record fails its attention check drops its
    /// instrument readouts too. `sampledRecords` (paired-judge evaluate):
    /// only sampled generations are considered, because only they are
    /// judged.
    public static let scopeAllRecordTypes = "allRecordTypes"
    public static let scopeSampledRecords = "sampledRecords"

    /// Pinned cross-engine stamp note — states the declared scope and the
    /// pairwise-deletion semantics the paired statistics inherit from
    /// record-level exclusion.
    static let note =
        "Exclusions are applied at analysis time only; excluded records "
        + "remain in generations.jsonl. Scope: all record types — endpoint "
        + "rules (unparseableEndpoint, outOfRange) read any record, sampled "
        + "or deterministic instrument readout, that itself carries the "
        + "named endpoint (never by proxy), and a failed attention check "
        + "drops the whole (condition, item) cell, instrument readouts "
        + "included, once every sampled record of the cell fails its check. "
        + "Paired statistics use pairwise deletion: an excluded record's "
        + "item drops from that condition's paired comparison, and an item "
        + "whose baseline record is excluded drops from every condition's "
        + "pairs. A record failing several rules is excluded once and "
        + "counted under each rule it failed; with multiple samples per "
        + "item, the cell keeps its surviving samples and drops only when "
        + "every sample is excluded."

    /// Pinned cross-engine stamp note for the paired-judge `evaluate` path,
    /// where the wording differs from analyze on purpose: there the rules
    /// filter records BEFORE judging, so no judge call (or judging packet)
    /// is ever spent on an excluded record (server twin `EVALUATE_NOTE`).
    static let evaluateNote =
        "Exclusions are applied before judging: excluded records are "
        + "filtered from the pairs entering the judge panel, so no judge "
        + "call is spent on them; excluded records remain in the source "
        + "run's generations.jsonl. Scope: sampled records only — "
        + "instrument readouts are never judged, so the rules read only "
        + "the sampled generations entering the panel. Paired judging uses "
        + "pairwise deletion: an excluded record's pair is not judged, and "
        + "an item whose baseline record is excluded drops from every "
        + "condition's pairs. A record failing several rules is excluded "
        + "once and counted under each rule it failed."

    static let pinRequiredMessage =
        "exclusion rule failedAttentionCheck needs the task prompts pinned "
        + "(taskPromptsFile + taskPromptsHash) so analysis grades the same "
        + "items the run saw — pin the prompt set first"

    /// The RUNNABLE repair for that refusal (WP0 step 8). Byte-identical to
    /// the server twin `PIN_REQUIRED_REPAIR`, and it names this engine's verb
    /// on BOTH engines because pinning is Mac-authority (audit §3.2) — a
    /// repair naming a verb the answering engine does not have would send an
    /// agent in a circle, which is exactly what gate-5 dry run #1 measured.
    static let pinRequiredRepair =
        "steerlab-cli experiment pin-prompts <name> <the prompt file the run "
        + "used> — analysis grades the items the run saw, so the pin must "
        + "name that exact file"

    static let noChecksMessage =
        "exclusion rule failedAttentionCheck is declared but no task-prompt "
        + "item declares an attentionCheck — add checks to items or drop "
        + "the rule"

    /// Canonical bound formatting shared with the server (600 → "600",
    /// 0.5 → "0.5") so descriptions and messages are byte-identical.
    static func formatBound(_ value: Double) -> String {
        if value.rounded() == value, abs(value) < 1e15 {
            return String(Int64(value))
        }
        return "\(value)"
    }

    // MARK: - Manifest rule validation (verify + run/analyze preflight)

    /// Plain-language violations for a manifest's declared rules. An absent
    /// / nil declaration is no rules and no violations (legacy manifests
    /// keep their bytes and their content hash). Message strings are the
    /// cross-engine contract (server twin `rule_violations`).
    public static func violations(_ rules: [ExclusionRule]?) -> [String] {
        guard let rules else { return [] }
        var violations: [String] = []
        var seen = Set<String>()
        for rule in rules {
            guard ruleVocabulary.contains(rule.rule) else {
                violations.append(
                    "exclusion rule '\(rule.rule)' is not recognized — "
                        + "declared rules must be one of: failedAttentionCheck, "
                        + "unparseableEndpoint, outOfRange")
                continue
            }
            guard seen.insert(rule.rule).inserted else {
                violations.append(
                    "exclusion rule '\(rule.rule)' is declared more than "
                        + "once — declare each rule at most once")
                continue
            }
            if rule.rule == ruleOutOfRange {
                if rule.min == nil, rule.max == nil {
                    violations.append(
                        "exclusion rule outOfRange declares no bounds — "
                            + "declare 'min', 'max', or both")
                } else if let low = rule.min, let high = rule.max, low > high {
                    violations.append(
                        "exclusion rule outOfRange has min (\(formatBound(low))) "
                            + "greater than max (\(formatBound(high))) — the rule "
                            + "keeps records with min <= value <= max")
                }
            } else if rule.min != nil || rule.max != nil {
                violations.append(
                    "exclusion rule '\(rule.rule)' does not take 'min'/'max' "
                        + "— a numeric range applies only to outOfRange")
            }
            if rule.rule == ruleFailedAttentionCheck {
                if rule.endpoint != nil {
                    violations.append(
                        "exclusion rule failedAttentionCheck does not take "
                            + "'endpoint' — it grades each record's output against "
                            + "its item's declared attentionCheck")
                }
            } else if let endpoint = rule.endpoint,
                endpoint.trimmingCharacters(in: .whitespaces).isEmpty
            {
                violations.append(
                    "exclusion rule '\(rule.rule)' declares an empty "
                        + "'endpoint' — omit the key for the default (parsedMonths) "
                        + "or name the record's parsed-value key")
            }
        }
        return violations
    }

    static func needsChecks(_ rules: [ExclusionRule]) -> Bool {
        rules.contains { $0.rule == ruleFailedAttentionCheck }
    }

    /// Run/analyze-START gate: malformed rules, and a declared
    /// failedAttentionCheck rule with no checked items, refuse BEFORE
    /// compute is spent (a declared-but-inert exclusion rule is a data bug,
    /// not a no-op). Server twin: `exclusions.preflight`.
    static func preflight(
        rules: [ExclusionRule]?, checks: [String: AttentionCheck]
    ) throws {
        guard let rules, !rules.isEmpty else { return }
        let problems = violations(rules)
        guard problems.isEmpty else {
            throw ExperimentError(reason: problems.joined(separator: "; "))
        }
        if needsChecks(rules), checks.isEmpty {
            throw ExperimentError(reason: noChecksMessage)
        }
    }

    // MARK: - Ladder-window advisory (run-start warning, never a refusal)

    /// The numeric range implied by the items' declared options ladders:
    /// min/max over every option of every item whose options ALL coerce to
    /// finite numbers (a partially numeric ladder implies nothing). Nil when
    /// no item carries a fully numeric ladder. Server twin: `ladder_range`.
    static func ladderRange(optionLadders: [[String]]) -> ClosedRange<Double>? {
        var values: [Double] = []
        for options in optionLadders where !options.isEmpty {
            var numbers: [Double] = []
            var allNumeric = true
            for option in options {
                let trimmed = option.trimmingCharacters(in: .whitespaces)
                guard let number = Double(trimmed), number.isFinite else {
                    allNumeric = false
                    break
                }
                numbers.append(number)
            }
            if allNumeric { values.append(contentsOf: numbers) }
        }
        guard let low = values.min(), let high = values.max() else { return nil }
        return low...high
    }

    /// Run-start advisories (2026-08-06): an outOfRange keep-window whose
    /// every declared bound lies outside the range the items' numeric
    /// options ladders imply cannot bind that scale — min 0 / max 100 on a
    /// 1–7 ladder is syntactically valid and semantically inert (it
    /// excludes nothing), and a disjoint window would exclude everything.
    /// A warning, never a refusal: the endpoint may lawfully take
    /// non-ladder values (e.g. months parsed from sampled prose), so the
    /// check is a heuristic. Message strings are the cross-engine contract
    /// (server twin `ladder_warnings`).
    public static func ladderWarnings(
        rules: [ExclusionRule]?, optionLadders: [[String]]
    ) -> [String] {
        guard let rules, !rules.isEmpty,
            let span = ladderRange(optionLadders: optionLadders)
        else { return [] }
        var warnings: [String] = []
        for rule in rules where rule.rule == ruleOutOfRange {
            var bounds: [(name: String, value: Double)] = []
            if let low = rule.min { bounds.append(("min", low)) }
            if let high = rule.max { bounds.append(("max", high)) }
            guard !bounds.isEmpty,
                bounds.allSatisfy({ !span.contains($0.value) })
            else { continue }
            let declared = bounds
                .map { "\($0.name) \(formatBound($0.value))" }
                .joined(separator: " and ")
            let spanLow = formatBound(span.lowerBound)
            let spanHigh = formatBound(span.upperBound)
            var message = "exclusion rule outOfRange declares \(declared), "
            message += "but the task items' options ladder spans "
            message += "\(spanLow) to \(spanHigh) — every declared bound "
            message += "lies outside the ladder, so if the endpoint takes "
            message += "ladder values the rule can never bind (it would "
            message += "exclude nothing or everything); align the bounds "
            message += "with the scale the items use, or drop the rule"
            warnings.append(message)
        }
        return warnings
    }

    // MARK: - Per-item attention checks

    /// Plain-language violation for one item's raw attentionCheck fields
    /// (nil = valid). Reuses the capability battery's pinned grading
    /// vocabulary — message strings are the cross-engine contract (server
    /// twin `attention_check_violation`).
    static func attentionCheckViolation(
        expected: String?, grading: String?, itemID: String
    ) -> String? {
        guard let expected,
            !expected.trimmingCharacters(in: .whitespaces).isEmpty
        else {
            return "task prompts: item '\(itemID)' declares an attentionCheck "
                + "without a non-empty 'expected' string — declare the "
                + "expected answer"
        }
        if let grading, CapabilityBattery.GradingMode(rawValue: grading) == nil {
            return "task prompts: item '\(itemID)' attentionCheck grading "
                + "'\(grading)' is not a known grading mode — one of: "
                + "exact_number, yes_no, token_exact, exact_normalized, regex"
        }
        return nil
    }

    // MARK: - Application (run-inline report + analyze)

    /// The minimal per-record view the rules read: pairing identity, the
    /// sampled output (attention-check grading), and the record-level
    /// parsed endpoints. In `endpoints`, a key present with a nil value is
    /// a parse FAILURE (JSON null); an absent key means the endpoint does
    /// not apply to the record.
    struct RecordView {
        let condition: String
        let seed: UInt64
        let promptID: String
        let output: String
        let endpoints: [String: Double?]
    }

    /// The minimal view of one deterministic instrument readout (choice /
    /// ordinal record): pairing identity plus the record-level endpoints
    /// the record ITSELF carries (e.g. ordinalPosition). No output — the
    /// attention check grades sampled text, so failedAttentionCheck reaches
    /// instrument records only through their cell's sampled evidence.
    struct InstrumentRecordView {
        let condition: String
        let promptID: String
        let endpoints: [String: Double?]
    }

    struct Outcome {
        let stamp: ExclusionStamp
        /// `rowKey` values of excluded sampled records — the filter the
        /// effect-size paths apply to their MetricRows.
        let excludedKeys: Set<String>
        /// `instrumentKey` values of excluded instrument readouts — the
        /// filter the ordinal/choice effect paths apply to their readouts.
        var excludedInstrumentKeys: Set<String> = []
    }

    /// The MetricRow join key — (condition, seed, promptID) is unique for
    /// local sampled records (one greedy record per cell).
    static func rowKey(condition: String, seed: UInt64, promptID: String) -> String {
        "\(condition)\u{1F}\(seed)\u{1F}\(promptID)"
    }

    /// The instrument-readout join key — (condition, promptID) is unique
    /// for instrument records (one deterministic readout per cell).
    static func instrumentKey(condition: String, promptID: String) -> String {
        "\(condition)\u{1F}\(promptID)"
    }

    static func resolvedEndpoint(_ rule: ExclusionRule) -> String {
        guard let endpoint = rule.endpoint,
            !endpoint.trimmingCharacters(in: .whitespaces).isEmpty
        else { return defaultEndpoint }
        return endpoint
    }

    /// The rule's plain-language description (stamped in the report so a
    /// reader never has to decode the vocabulary). Cross-engine strings.
    static func description(of rule: ExclusionRule) -> String {
        switch rule.rule {
        case ruleFailedAttentionCheck:
            return "the record's output failed its item's declared attention check"
        case ruleUnparseableEndpoint:
            return "no parseable \(resolvedEndpoint(rule)) value (the endpoint "
                + "parser produced null)"
        default:
            let endpoint = resolvedEndpoint(rule)
            if let low = rule.min, let high = rule.max {
                return "parsed \(endpoint) outside the declared range "
                    + "[\(formatBound(low)), \(formatBound(high))]"
            }
            if let low = rule.min {
                return "parsed \(endpoint) below the declared minimum "
                    + formatBound(low)
            }
            return "parsed \(endpoint) above the declared maximum "
                + formatBound(rule.max ?? 0)
        }
    }

    static func resolvedRuleStamp(
        _ rule: ExclusionRule, checks: [String: AttentionCheck]
    ) -> ExclusionStamp.ResolvedRule {
        var stamp = ExclusionStamp.ResolvedRule(
            rule: rule.rule, description: description(of: rule))
        if rule.rule == ruleFailedAttentionCheck {
            stamp.checkedItems = checks.count
        } else {
            stamp.endpoint = resolvedEndpoint(rule)
        }
        if rule.rule == ruleOutOfRange {
            stamp.min = rule.min
            stamp.max = rule.max
        }
        return stamp
    }

    /// One endpoint-reading rule against one record's OWN endpoints: a
    /// record that does not carry the named endpoint can never fire it
    /// (never by proxy) — identical for sampled and instrument records.
    static func endpointRuleFired(
        _ rule: ExclusionRule, endpoints: [String: Double?]
    ) -> Bool {
        switch rule.rule {
        case ruleUnparseableEndpoint:
            if case .some(.none) = endpoints[resolvedEndpoint(rule)] {
                return true
            }
            return false
        case ruleOutOfRange:
            if case .some(.some(let value)) = endpoints[resolvedEndpoint(rule)] {
                if let low = rule.min, value < low { return true }
                if let high = rule.max, value > high { return true }
            }
            return false
        default:
            return false
        }
    }

    /// The rule ids that fire for one sampled record (a record can fail
    /// several rules; it is excluded once and counted under each).
    static func firedRules(
        for view: RecordView, rules: [ExclusionRule],
        checks: [String: AttentionCheck]
    ) -> [String] {
        var fired: [String] = []
        for rule in rules {
            switch rule.rule {
            case ruleFailedAttentionCheck:
                if let check = checks[view.promptID],
                    !CapabilityBattery.isCorrect(
                        response: view.output, answer: check.expected,
                        grading: check.grading)
                {
                    fired.append(rule.rule)
                }
            case ruleUnparseableEndpoint, ruleOutOfRange:
                if endpointRuleFired(rule, endpoints: view.endpoints) {
                    fired.append(rule.rule)
                }
            default:
                continue  // unreachable after preflight; never trap in analysis
            }
        }
        return fired
    }

    /// The rule ids that fire for one deterministic instrument readout.
    /// The attention check grades sampled OUTPUT text an instrument record
    /// does not have, so `failedAttentionCheck` fires on the CELL's
    /// evidence — only when every sampled record of the readout's
    /// (condition, promptID) cell failed the check. Endpoint rules read
    /// the record's own endpoints, exactly as for sampled records.
    static func instrumentFiredRules(
        for view: InstrumentRecordView, rules: [ExclusionRule],
        attentionFailedCell: Bool
    ) -> [String] {
        var fired: [String] = []
        for rule in rules {
            switch rule.rule {
            case ruleFailedAttentionCheck:
                if attentionFailedCell { fired.append(rule.rule) }
            case ruleUnparseableEndpoint, ruleOutOfRange:
                if endpointRuleFired(rule, endpoints: view.endpoints) {
                    fired.append(rule.rule)
                }
            default:
                continue  // unreachable after preflight; never trap in analysis
            }
        }
        return fired
    }

    /// Applies validated rules to a run's record views — sampled records
    /// plus (under the default `scopeAllRecordTypes`) deterministic
    /// instrument readouts: returns the exclusion stamp plus the excluded
    /// keys of each record type. Pure and deterministic; the caller filters
    /// its MetricRows / instrument readouts (paired statistics) or the
    /// records entering paired judging, never the records on disk. `note`
    /// defaults to the analysis-time wording; the evaluate path passes
    /// `evaluateNote` + `scopeSampledRecords` (its exclusions save judge
    /// calls, and only sampled records are judged).
    static func evaluate(
        rules: [ExclusionRule], checks: [String: AttentionCheck],
        views: [RecordView],
        instrumentViews: [InstrumentRecordView] = [],
        note: String = ExclusionEngine.note,
        scope: String = ExclusionEngine.scopeAllRecordTypes
    ) -> Outcome {
        var considered: [String: Int] = [:]
        var excludedCount: [String: Int] = [:]
        var byRule: [String: [String: Int]] = [:]
        var excludedKeys = Set<String>()
        var excludedInstrumentKeys = Set<String>()
        var totalExcluded = 0
        // Per-cell attention evidence the instrument readouts inherit: a
        // cell fails only when EVERY sampled record failed the check.
        var cellSampled: [String: Int] = [:]
        var cellAttentionFailed: [String: Int] = [:]
        for view in views {
            considered[view.condition, default: 0] += 1
            let fired = firedRules(for: view, rules: rules, checks: checks)
            let cell = instrumentKey(
                condition: view.condition, promptID: view.promptID)
            cellSampled[cell, default: 0] += 1
            if fired.contains(ruleFailedAttentionCheck) {
                cellAttentionFailed[cell, default: 0] += 1
            }
            guard !fired.isEmpty else { continue }
            totalExcluded += 1
            excludedCount[view.condition, default: 0] += 1
            excludedKeys.insert(
                rowKey(
                    condition: view.condition, seed: view.seed,
                    promptID: view.promptID))
            for ruleID in fired {
                byRule[view.condition, default: [:]][ruleID, default: 0] += 1
            }
        }
        if scope == scopeAllRecordTypes {
            for view in instrumentViews {
                considered[view.condition, default: 0] += 1
                let cell = instrumentKey(
                    condition: view.condition, promptID: view.promptID)
                let sampledCount = cellSampled[cell] ?? 0
                let attentionFailed =
                    sampledCount > 0
                    && (cellAttentionFailed[cell] ?? 0) == sampledCount
                let fired = instrumentFiredRules(
                    for: view, rules: rules,
                    attentionFailedCell: attentionFailed)
                guard !fired.isEmpty else { continue }
                totalExcluded += 1
                excludedCount[view.condition, default: 0] += 1
                excludedInstrumentKeys.insert(cell)
                for ruleID in fired {
                    byRule[view.condition, default: [:]][ruleID, default: 0] += 1
                }
            }
        }
        let ruleIDs = rules.map(\.rule)
        var excludedByRule: [String: [String: Int]] = [:]
        var survivingN: [String: Int] = [:]
        for (condition, count) in considered {
            // Last-wins merge, never `uniqueKeysWithValues:` — a duplicated
            // rule id is refused upstream, but assembling the stamp must
            // never trap after generation compute is spent.
            excludedByRule[condition] = Dictionary(
                ruleIDs.map { ($0, byRule[condition]?[$0] ?? 0) },
                uniquingKeysWith: { _, new in new })
            survivingN[condition] = count - (excludedCount[condition] ?? 0)
        }
        let stamp = ExclusionStamp(
            rules: rules.map { resolvedRuleStamp($0, checks: checks) },
            consideredN: considered,
            excludedByRule: excludedByRule,
            excludedRecords: totalExcluded,
            survivingN: survivingN,
            pairwiseDeletion: true,
            scope: scope,
            note: note)
        return Outcome(
            stamp: stamp, excludedKeys: excludedKeys,
            excludedInstrumentKeys: excludedInstrumentKeys)
    }
}
