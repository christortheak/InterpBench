import Foundation

// UI-facing helpers + the manifest setter for declared exclusion rules
// (USABILITY-PLAN Phase-4 item 19's UI affordance). A NEW file by the
// ownership rule: `ExclusionRules.swift` carries the cross-engine science
// contract (rule vocabulary, violation strings, stamp shape) and is not
// edited for UI needs. Everything here is presentation-side: researcher-
// facing rule descriptions, the pre-run attention-check probe, the results
// pane's stamp reading + plain-language summary, and the draft-edit setter.

/// Read-only exclusion-rule views for the study panel and Results pane.
public enum ExclusionRulesUI {

    /// The closed rule vocabulary, re-exported for the editor's add menu.
    public static var ruleVocabulary: [String] { ExclusionEngine.ruleVocabulary }
    public static var defaultEndpoint: String { ExclusionEngine.defaultEndpoint }

    /// Researcher-facing one-line description of a declared rule — what it
    /// DROPS, in plain words. (The engine's `description(of:)` is the
    /// stamped report wording; this is the editor's row copy.)
    public static func editorDescription(of rule: ExclusionRule) -> String {
        switch rule.rule {
        case ExclusionEngine.ruleFailedAttentionCheck:
            return "Drop answers that failed their item's attention check "
                + "(items declare the expected answer in the prompt file)."
        case ExclusionEngine.ruleUnparseableEndpoint:
            return "Drop answers where no \(resolvedEndpoint(rule)) value "
                + "could be read (parse failures)."
        case ExclusionEngine.ruleOutOfRange:
            let endpoint = resolvedEndpoint(rule)
            let bounds = boundsPhrase(min: rule.min, max: rule.max)
            return "Drop answers whose \(endpoint) is \(bounds)."
        default:
            return "Rule '\(rule.rule)' is not recognized — declared rules "
                + "must come from the closed vocabulary."
        }
    }

    /// A short column label for the Results table ("attention check",
    /// "unparseable", "out of range").
    public static func shortLabel(forRule id: String) -> String {
        switch id {
        case ExclusionEngine.ruleFailedAttentionCheck: "attention check"
        case ExclusionEngine.ruleUnparseableEndpoint: "unparseable"
        case ExclusionEngine.ruleOutOfRange: "out of range"
        default: id
        }
    }

    /// The endpoint a rule reads once the default is applied.
    public static func resolvedEndpoint(_ rule: ExclusionRule) -> String {
        ExclusionEngine.resolvedEndpoint(rule)
    }

    /// Canonical bound formatting (600 → "600", 0.5 → "0.5"), re-exported
    /// for the editor's field display so it matches the stamped wording.
    public static func formatBound(_ value: Double) -> String {
        ExclusionEngine.formatBound(value)
    }

    private static func boundsPhrase(min: Double?, max: Double?) -> String {
        let format = ExclusionEngine.formatBound
        if let low = min, let high = max {
            return "outside \(format(low)) to \(format(high))"
        }
        if let low = min { return "below \(format(low))" }
        if let high = max { return "above \(format(high))" }
        return "outside a range with no declared bounds (declare min, max, "
            + "or both)"
    }

    // MARK: - Pre-run attention-check probe

    /// How many items in the study's current task-prompts file declare an
    /// `attentionCheck` — the number the run-start gate needs to be > 0
    /// whenever `failedAttentionCheck` is declared. nil when the file is
    /// missing or does not parse (other readiness rows own THOSE findings;
    /// the editor's warning stays quiet rather than double-reporting).
    public static func attentionCheckItemCount(
        taskPromptsFile: String?
    ) -> Int? {
        let file = taskPromptsFile ?? "prompts/dev/dev-prompts.jsonl"
        let url =
            file.hasPrefix("/")
            ? URL(filePath: file)
            : ExperimentStore.resolveProjectPath(file)
        guard let data = try? Data(contentsOf: url),
            let prompts = try? ExperimentTasks.parseTaskPrompts(data)
        else { return nil }
        return prompts.filter { $0.attentionCheck != nil }.count
    }

    // MARK: - Results: reading and narrating the exclusion stamp

    /// The `exclusions` stamp of one run directory, wherever the engine
    /// put it: report.json (run-inline), analysis.json (analyze verb), or
    /// the standalone exclusions.json both engines write. nil when the run
    /// declared no rules (legacy runs included) or nothing is readable.
    public static func loadStamp(runDirectory: URL) -> ExclusionStamp? {
        struct Wrapper: Decodable {
            let exclusions: ExclusionStamp?
        }
        let decoder = JSONDecoder()
        for name in ["report.json", "analysis.json"] {
            if let data = try? Data(
                contentsOf: runDirectory.appending(component: name)),
                let wrapper = try? decoder.decode(Wrapper.self, from: data),
                let stamp = wrapper.exclusions
            {
                return stamp
            }
        }
        if let data = try? Data(
            contentsOf: runDirectory.appending(component: "exclusions.json")),
            let stamp = try? decoder.decode(ExclusionStamp.self, from: data)
        {
            return stamp
        }
        return nil
    }

    /// Per-rule exclusion totals across all conditions, in the stamp's
    /// declared rule order (zeros included — honesty about what did NOT
    /// fire).
    public static func ruleTotals(
        of stamp: ExclusionStamp
    ) -> [(rule: String, count: Int)] {
        stamp.rules.map { rule in
            let total = stamp.excludedByRule.values.reduce(0) {
                $0 + ($1[rule.rule] ?? 0)
            }
            return (rule: rule.rule, count: total)
        }
    }

    /// The one plain-language sentence the Results pane leads with:
    /// "12 of 240 answers were excluded: 8 failed attention checks, 4
    /// could not be parsed. Their paired baseline answers were dropped
    /// from the affected comparisons too." — or the explicit zero line
    /// (silence would be ambiguous).
    public static func summary(of stamp: ExclusionStamp) -> String {
        let considered = stamp.consideredN.values.reduce(0, +)
        let answers = considered == 1 ? "answer" : "answers"
        guard stamp.excludedRecords > 0 else {
            let rules = stamp.rules.count == 1
                ? "1 declared rule was" : "\(stamp.rules.count) declared rules were"
            return "No answers were excluded — \(rules) checked against all "
                + "\(considered) \(answers) and none fired."
        }
        let totals = ruleTotals(of: stamp).filter { $0.count > 0 }
        let reasons = totals.map { phrase(ruleID: $0.rule, count: $0.count) }
            .joined(separator: ", ")
        var text =
            "\(stamp.excludedRecords) of \(considered) \(answers) were "
            + "excluded: \(reasons)."
        if totals.reduce(0, { $0 + $1.count }) > stamp.excludedRecords {
            text += " (An answer failing several rules is counted under each.)"
        }
        if stamp.pairwiseDeletion {
            text += " Their paired baseline answers were dropped from the "
                + "affected comparisons too."
        }
        return text
    }

    static func phrase(ruleID: String, count: Int) -> String {
        switch ruleID {
        case ExclusionEngine.ruleFailedAttentionCheck:
            count == 1
                ? "1 failed an attention check"
                : "\(count) failed attention checks"
        case ExclusionEngine.ruleUnparseableEndpoint:
            "\(count) could not be parsed"
        case ExclusionEngine.ruleOutOfRange:
            count == 1
                ? "1 was outside the declared range"
                : "\(count) were outside the declared range"
        default:
            "\(count) matched rule '\(ruleID)'"
        }
    }
}

extension ExperimentStore {

    /// Declare (or clear) the study's exclusion rules, through the same
    /// draft-edit gate as every science-manifest setter. Malformed rule
    /// sets refuse NOW with the engine's own plain-language violations
    /// (feedback at the moment of action, not at a failing gate); an empty
    /// list normalizes to ABSENT so legacy manifests keep their bytes and
    /// their content hash.
    @discardableResult
    public static func setExclusionRules(
        _ rules: [ExclusionRule]?, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            guard let rules, !rules.isEmpty else {
                manifest.exclusionRules = nil
                return
            }
            let problems = ExclusionEngine.violations(rules)
            guard problems.isEmpty else {
                throw ExperimentError(reason: problems.joined(separator: "; "))
            }
            manifest.exclusionRules = rules
        }
    }
}
