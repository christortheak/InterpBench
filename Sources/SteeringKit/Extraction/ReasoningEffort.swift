import Foundation

/// The declared reasoning effort — `reasoningEffort` ∈ off | low | medium |
/// xhigh — on a study's protocol and on a concept's `extractionRendering`
/// (2026-09-03). It replaced the Qwen-specific boolean `qwenThinkingEnabled`.
///
/// `off` is exactly the old `false`: the chat template is rendered with
/// `enable_thinking=false` and the model writes no reasoning block. The other
/// three reach the template as `enable_thinking=true` PLUS
/// `reasoning_effort=<value>` — the variable the Qwen3.8 template reads (its
/// own default is `xhigh`; Qwen3's template ignores the extra variable, which
/// is what makes a declared study reproducible on either).
///
/// THE LEGACY READING. A manifest or rendering that still spells the boolean
/// is read as `off` (false) or `xhigh` (true) — `true` meant the template's
/// default effort, which is what every such study actually ran under — and it
/// is never rewritten: the ladder program is frozen under that spelling and
/// its hashes stand. Server twin: `prompt_render.REASONING_EFFORTS` and the
/// readers beside it.
public enum ReasoningEffort: String, CaseIterable, Codable, Sendable, Hashable {
    case off
    /// Thinking ON at the template's own default effort: `enable_thinking`
    /// true and no `reasoning_effort` variable at all — the only non-off
    /// effort a template with a thinking switch but no effort control can
    /// honour (2026-09-05, with the capability record).
    case on
    case low
    case medium
    case high
    case xhigh

    /// The closed vocabulary as wire strings, in the fixed cross-engine
    /// order: off, on, then the probe candidates a template may accept
    /// (`ModelCapabilities.effortCandidates`). Which LEVELS a given model may
    /// declare is the pinned template's answer, not the vocabulary's.
    public static let vocabulary: [String] = allCases.map(\.rawValue)

    /// The vocabulary entries that name a LEVEL (a value of
    /// `reasoning_effort`).
    public static let levels: [ReasoningEffort] = [.low, .medium, .high, .xhigh]

    /// Whether this effort names a level the template is asked for.
    public var isLevel: Bool { Self.levels.contains(self) }

    /// What a legacy `qwenThinkingEnabled: true` means: the chat template's
    /// default effort.
    public static let legacyThinking: ReasoningEffort = .xhigh

    /// Whether the model writes a reasoning block at all — the one boolean
    /// question every pre-effort renderer and gate asked.
    public var isOn: Bool { self != .off }

    /// The legacy boolean's meaning.
    public static func legacy(qwenThinkingEnabled: Bool) -> ReasoningEffort {
        qwenThinkingEnabled ? legacyThinking : .off
    }

    /// The declared effort when one was passed, else the legacy boolean's
    /// meaning — the resolution every renderer applies.
    public static func resolve(
        _ declared: ReasoningEffort?, qwenThinkingEnabled: Bool
    ) -> ReasoningEffort {
        declared ?? legacy(qwenThinkingEnabled: qwenThinkingEnabled)
    }

    // MARK: - The manifest keys

    /// Manifest / protocol key of the effort. Server twin:
    /// `prompt_render.REASONING_EFFORT_KEY`.
    public static let effortKey = "reasoningEffort"
    /// Manifest / protocol key of the reasoning block's own token cap.
    public static let budgetKey = "reasoningMaxTokens"
    /// The key this vocabulary replaced — read, never written.
    public static let legacyKey = "qwenThinkingEnabled"

    // MARK: - The declaration rules (shared sentences)

    /// Refusal sentences shared VERBATIM with the server
    /// (`prompt_render.reasoning_protocol_violations` and the constants beside
    /// it). Every writer on both engines refuses with these, and `verify`
    /// re-checks them for a hand-edited manifest.
    public static let budgetWithoutEffortReason =
        "\(budgetKey) is declared but \(effortKey) is off — the model generates "
        + "no reasoning block to cap; declare a non-off \(effortKey) or drop the "
        + "budget"

    public static func unknownEffortReason(_ value: String) -> String {
        "unknown \(effortKey) '\(value)' — known: "
            + vocabulary.joined(separator: ", ")
    }

    public static func effortWithoutBudgetReason(_ effort: ReasoningEffort) -> String {
        "\(effortKey) '\(effort.rawValue)' needs a \(budgetKey) — the reasoning "
            + "block's own token cap, declared and never defaulted; maxTokens "
            + "stays the answer budget, counted from the token after </think>"
    }

    public static func effortWithoutThinkingModeReason(
        _ effort: ReasoningEffort, modelID: String
    ) -> String {
        "\(effortKey) '\(effort.rawValue)' declared for \(modelID), whose chat "
            + "template has no thinking switch (enable_thinking changes nothing "
            + "it renders) — the template would ignore it and the study would "
            + "look as if it reasoned when it did not; declare \(effortKey) off"
    }

    /// A level on a template that has the switch but never reads
    /// `reasoning_effort` (Qwen3-14B/-32B): the study would run at the
    /// template's default while its manifest asserted the level.
    public static func effortIgnoredReason(
        _ effort: ReasoningEffort, modelID: String
    ) -> String {
        "\(effortKey) '\(effort.rawValue)' declared for \(modelID), whose chat "
            + "template reads enable_thinking but ignores reasoning_effort — the "
            + "study would run at the template's default effort while its "
            + "manifest asserts \(effort.rawValue); declare \(effortKey) "
            + "\(ReasoningEffort.on.rawValue) (thinking at the template's default), "
            + "or pin a model whose template reads the effort"
    }

    public static func effortRejectedReason(
        _ effort: ReasoningEffort, modelID: String, capabilities: ModelCapabilities
    ) -> String {
        let accepted = capabilities.acceptedEfforts.joined(separator: ", ")
        return "\(effortKey) '\(effort.rawValue)' is rejected by the chat template "
            + "of \(modelID) (the template raises on it) — accepted levels: "
            + (accepted.isEmpty ? "none" : accepted)
            + "; or declare \(effortKey) \(ReasoningEffort.on.rawValue)"
    }

    /// A level the record never judged — a heuristic record asked for a
    /// level the old family rule never assumed (`high`), or a probed record
    /// from an older candidate list.
    public static func effortUnprobedReason(
        _ effort: ReasoningEffort, modelID: String, capabilities: ModelCapabilities
    ) -> String {
        let accepted = capabilities.acceptedEfforts.joined(separator: ", ")
        return "\(effortKey) '\(effort.rawValue)' is not known to be accepted by "
            + "the chat template of \(modelID) (record source "
            + "\(capabilities.source.rawValue); accepted levels: "
            + (accepted.isEmpty ? "none" : accepted)
            + ") — probe the template (model capabilities --probe) or declare a "
            + "level it accepts"
    }

    public static func effortAssumedAdvisory(
        _ effort: ReasoningEffort, modelID: String
    ) -> String {
        "\(effortKey) '\(effort.rawValue)' on \(modelID) is ASSUMED accepted "
            + "from the model id — no probed capability record; a template that "
            + "ignores reasoning_effort would run at its default effort. Probe "
            + "it: model capabilities --probe"
    }

    public static func systemPromptUnsupportedReason(modelID: String) -> String {
        "a system prompt is declared for \(modelID), whose chat template has no "
            + "way to deliver system text (it raises on, or drops, a system turn) "
            + "— drop the system prompt, or write the frame into the task prompts "
            + "yourself and say so in METHODS"
    }

    /// The one rule about a LEVEL: the pinned template must accept it.
    /// Server twin: `prompt_render.effort_level_violations`.
    public static func effortLevelViolations(
        _ effort: ReasoningEffort, modelID: String, capabilities: ModelCapabilities
    ) -> [String] {
        switch capabilities.effortVerdict(effort.rawValue) {
        case .accepted, .assumed: return []
        case .ignored: return [effortIgnoredReason(effort, modelID: modelID)]
        case .rejected:
            return [effortRejectedReason(effort, modelID: modelID, capabilities: capabilities)]
        case nil:
            return [effortUnprobedReason(effort, modelID: modelID, capabilities: capabilities)]
        }
    }

    /// Non-blocking notes beside a declaration the gates accepted: a level
    /// ASSUMED from a heuristic record, and the record's own advisories.
    /// Server twin: `prompt_render.reasoning_protocol_advisories`.
    public static func protocolAdvisories(
        effort: ReasoningEffort, modelID: String, capabilities: ModelCapabilities
    ) -> [String] {
        // Only where the record DECIDED something — a non-off effort. A
        // study that reasons not at all is not nagged about a record it
        // never read.
        guard effort.isOn else { return [] }
        var notes: [String] = []
        if effort.isLevel, capabilities.effortVerdict(effort.rawValue) == .assumed {
            notes.append(effortAssumedAdvisory(effort, modelID: modelID))
        }
        for note in capabilities.advisories where !notes.contains(note) {
            notes.append(note)
        }
        return notes
    }

    public static func malformedBudgetReason(_ value: Int) -> String {
        "\(budgetKey) must be a positive integer — got \(value)"
    }

    /// Every rule a declared reasoning protocol must satisfy, as the
    /// sentences both engines refuse with. `effort` is the RAW declared
    /// string (nil = absent = off); `reasoningMaxTokens` nil = absent;
    /// `capabilities` is the model's record (nil: the registered record for
    /// the id, else the heuristic).
    ///
    /// - the effort is in the closed vocabulary;
    /// - a non-off effort names a template with a thinking switch;
    /// - a LEVEL names one the template accepts (`on` needs only the
    ///   switch; an ignored or rejected level is refused, an assumed one —
    ///   heuristic record — passes and is advised on elsewhere);
    /// - a non-off effort carries a positive reasoning budget;
    /// - an off effort carries none.
    public static func protocolViolations(
        effort: String?, reasoningMaxTokens: Int?, modelID: String,
        capabilities: ModelCapabilities? = nil
    ) -> [String] {
        let spelled = effort ?? ReasoningEffort.off.rawValue
        guard let parsed = ReasoningEffort(rawValue: spelled) else {
            return [unknownEffortReason(spelled)]
        }
        if let budget = reasoningMaxTokens, budget < 1 {
            return [malformedBudgetReason(budget)]
        }
        if !parsed.isOn {
            return reasoningMaxTokens == nil ? [] : [budgetWithoutEffortReason]
        }
        var problems: [String] = []
        let record = PromptRendering.capabilities(for: modelID, explicit: capabilities)
        if !record.hasThinkingSwitch {
            problems.append(effortWithoutThinkingModeReason(parsed, modelID: modelID))
        } else if parsed.isLevel {
            problems += effortLevelViolations(parsed, modelID: modelID, capabilities: record)
        }
        if reasoningMaxTokens == nil {
            problems.append(effortWithoutBudgetReason(parsed))
        }
        return problems
    }
}
