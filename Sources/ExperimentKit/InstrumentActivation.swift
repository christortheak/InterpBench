import Foundation

/// Explicit instrument activation (team finding P1, 2026-07-13): per-item
/// `options` being PRESERVED in the prompts file is not the same thing as a
/// categorical instrument being ENABLED — measurement method belongs in the
/// manifest's `outcomeInstruments` (provenance), never inferred from data.
/// These pure rules back the Studies panel's Evaluation controls, the Save &
/// Pin status line, and the headless `data check` readiness row, so all
/// three surfaces say the same thing.
public enum InstrumentActivation {

    /// The categorical (answer-token) instruments: the ones per-item
    /// `options` exist to feed.
    public static let categoricalInstruments: Set<String> = [
        "answerTokenLogprob", "choiceProbability",
    ]

    /// The instrument ids the Outcome Mode picker OWNS — the closed set
    /// BOTH mapping directions (`OutcomeMode.from` and `applying`) are
    /// defined over. Everything else declared in `outcomeInstruments` is an
    /// auxiliary (today: `repeReaderScore`): invisible to the mode axes,
    /// passed through untouched by `applying`, and displayed as its own
    /// Evaluation row via `auxiliaryInstruments(of:)`.
    public static let pickerOwnedInstruments: Set<String> =
        categoricalInstruments.union(["sampledText"])

    /// The instruments FED by per-item `options`: the picker's two
    /// answer-token instruments plus the auxiliary `ordinalScale` (which
    /// reads the same options as an ordered ladder). Drives the
    /// options-detected warning and the Save & Pin "enabled" line; the
    /// Outcome Mode picker still owns only `pickerOwnedInstruments`, so
    /// `ordinalScale` survives mode edits like any auxiliary.
    public static let optionConsumingInstruments: Set<String> =
        categoricalInstruments.union(["ordinalScale"])

    /// True when the declared instrument list enables a readout of the
    /// per-item options (categorical answer-token or ordinal-scale).
    public static func categoricalDeclared(_ instruments: [String]?) -> Bool {
        !optionConsumingInstruments.isDisjoint(with: Set(instruments ?? []))
    }

    /// The detected-capabilities line for the Evaluation section — what the
    /// DATA supports, stated separately from what is enabled. nil when no
    /// item carries options.
    public static func detectedCapabilitiesLine(optionsItemCount: Int) -> String? {
        guard optionsItemCount > 0 else { return nil }
        return optionsItemCount == 1
            ? "1 item has categorical options"
            : "\(optionsItemCount) items have categorical options"
    }

    /// The prominent pre-run warning: options are present but no categorical
    /// instrument is declared. nil = no warning (nothing detected, or an
    /// instrument is enabled).
    ///
    /// `unscorableOptionItemCount` is how many of those option-carrying items
    /// declare a `responseFormat` the answer-token instruments cannot read.
    /// Before 2026-07-26 this function did not exist in that dimension, and a
    /// reasons-arm study whose every row is `json` was told to "declare a
    /// categorical outcome instrument" — advice that produces a meaningless
    /// measurement (the scored position holds `{`, not a label) and that the
    /// run loop now refuses outright.
    public static func activationWarning(
        optionsItemCount: Int, instruments: [String]?,
        unscorableOptionItemCount: Int = 0
    ) -> String? {
        guard optionsItemCount > 0, !categoricalDeclared(instruments) else {
            return nil
        }
        guard unscorableOptionItemCount > 0 else {
            return "Options are present, but this study will only generate and "
                + "parse answer text. Declare a categorical outcome instrument "
                + "(answer-token probability) to measure choice probabilities."
        }
        if unscorableOptionItemCount >= optionsItemCount {
            // Nothing to recommend: sampled text IS the right instrument here.
            return "Options are present, but every one of these \(optionsItemCount) "
                + "items asks for a JSON or free-text response, so the choice "
                + "is not at the first generated position. Answer-token "
                + "probability cannot read them — sampled text parsed for a "
                + "choice is the appropriate instrument for this file."
        }
        let scorable = optionsItemCount - unscorableOptionItemCount
        return "Options are present on \(optionsItemCount) items, but "
            + "\(unscorableOptionItemCount) of them ask for a JSON or "
            + "free-text response that answer-token probability cannot read. "
            + "Declaring it would refuse at run start unless you also declare "
            + "an outcome-instrument scope limiting it to the \(scorable) "
            + "label item\(scorable == 1 ? "" : "s")."
    }

    /// Save & Pin reports both facts SEPARATELY: metadata preserved vs
    /// instruments enabled.
    public static func savePinSummary(
        optionsItemCount: Int, itemCount: Int, instruments: [String]?
    ) -> String {
        let metadata =
            optionsItemCount > 0
            ? "metadata preserved: \(optionsItemCount) of \(itemCount) items carry options"
            : "metadata preserved: no per-item instrument fields"
        let enabled = (instruments ?? []).filter {
            optionConsumingInstruments.contains($0)
        }
        let enabledLine =
            enabled.isEmpty
            ? "instruments enabled: none"
            : "instruments enabled: \(enabled.joined(separator: ", "))"
        return "\(metadata); \(enabledLine)"
    }

    /// The Evaluation section's outcome-mode control — presentation over
    /// `outcomeInstruments`, one rule for both directions so the picker and
    /// the manifest can never disagree.
    public enum OutcomeMode: String, CaseIterable, Sendable {
        /// `outcomeInstruments` absent — the engine default (sampled text
        /// only). Selecting a real mode writes an explicit declaration; this
        /// state is never written back.
        case notDeclared
        /// Sampled text, parsed for a choice (`parsedChoice`).
        case generatedChoice
        /// Deterministic answer-token logprob readout.
        case answerTokenProbability
        /// Both instruments on the same run.
        case both

        public var label: String {
            switch self {
            case .notDeclared: "not declared (sampled text)"
            case .generatedChoice: "Generated choice"
            case .answerTokenProbability: "Answer-token probability"
            case .both: "Both"
            }
        }

        /// What this mode writes into `outcomeInstruments`.
        public var instruments: [String]? {
            switch self {
            case .notDeclared: nil
            case .generatedChoice: ["sampledText"]
            case .answerTokenProbability: ["answerTokenLogprob"]
            case .both: ["sampledText", "answerTokenLogprob"]
            }
        }

        /// The mode a manifest's declared list reads back as, classified
        /// ONLY over `pickerOwnedInstruments` — the same closed set
        /// `applying` edits, so `from(applying(m, x)) == m` for every mode
        /// m and ANY auxiliary content x. Auxiliaries (e.g.
        /// `repeReaderScore`) are invisible to the axes here; their
        /// sampling implication is stated separately
        /// (`auxiliaryImpliesSampledGeneration` /
        /// `effectiveRecordKindsNote`), never folded into the mode (F3:
        /// the old repeReaderScore special case broke the round trip and
        /// made the picker snap back to "Both" on reader studies). A list
        /// containing only auxiliaries reads back `.notDeclared`: no
        /// picker-owned declaration exists.
        public static func from(_ instruments: [String]?) -> OutcomeMode {
            let owned = Set(instruments ?? []).intersection(pickerOwnedInstruments)
            guard !owned.isEmpty else { return .notDeclared }
            let categorical = !categoricalInstruments.isDisjoint(with: owned)
            let sampled = owned.contains("sampledText")
            switch (sampled, categorical) {
            case (true, true): return .both
            case (false, true): return .answerTokenProbability
            default: return .generatedChoice
            }
        }
    }

    /// Replace only the categorical/sampled-text axes controlled by the
    /// Outcome Mode picker. Auxiliary instruments (notably
    /// `repeReaderScore`) survive the edit; otherwise choosing "Both" on a
    /// reader study silently disabled the reader it was meant to measure.
    public static func applying(
        _ mode: OutcomeMode, to existing: [String]?
    ) -> [String]? {
        guard mode != .notDeclared else { return existing }
        var merged = (existing ?? []).filter { !pickerOwnedInstruments.contains($0) }
        for instrument in mode.instruments ?? [] where !merged.contains(instrument) {
            merged.append(instrument)
        }
        return merged.isEmpty ? nil : merged
    }

    // MARK: - Auxiliary instruments (F3)

    /// Declared instrument ids the Outcome Mode picker does not own, in
    /// manifest order (deduplicated). Rendered as their own Evaluation
    /// rows — never folded into the mode axes.
    public static func auxiliaryInstruments(of instruments: [String]?) -> [String] {
        var seen: Set<String> = []
        return (instruments ?? []).filter {
            !pickerOwnedInstruments.contains($0) && seen.insert($0).inserted
        }
    }

    /// Whether a declared auxiliary forces sampled generation regardless of
    /// the picker mode. Mirrors the run loops' dispatch rule on both
    /// engines (`ExperimentTasks` / server `_run_impl`: `wantsSampled =
    /// instruments.isEmpty || contains("sampledText") ||
    /// contains("repeReaderScore")`): the RepE reader scores generated
    /// responses, so declaring it keeps sampled generation running even in
    /// Answer-token mode.
    public static func auxiliaryImpliesSampledGeneration(_ id: String) -> Bool {
        id == "repeReaderScore"
    }

    /// The Evaluation row text for one enabled auxiliary — honest about the
    /// sampling implication where there is one.
    public static func auxiliaryDescription(_ id: String) -> String {
        switch id {
        case "repeReaderScore":
            return "repeReaderScore (reader) — scores generated responses; "
                + "sampled generation still runs even in Answer-token mode"
        case "ordinalScale":
            return "ordinalScale (ordinal ladder) — reads the model's "
                + "probability over each item's ordered options (for example "
                + "a 1–7 scale) and reports a ladder position; deterministic, "
                + "no sampled generation needed. Requires the aggregation "
                + "choice (expected value or argmax) to be declared"
        default:
            return auxiliaryImpliesSampledGeneration(id)
                ? "\(id) — auxiliary instrument; forces sampled generation"
                : "\(id) — auxiliary instrument (not controlled by the "
                    + "outcome-mode picker)"
        }
    }

    /// The pre-run effective-record-kinds note: non-nil exactly when the
    /// picker mode reads answer-token-only but a declared auxiliary forces
    /// sampled generation anyway — the one case where the mode label
    /// understates what the run will actually record.
    public static func effectiveRecordKindsNote(instruments: [String]?) -> String? {
        guard OutcomeMode.from(instruments) == .answerTokenProbability else {
            return nil
        }
        let forcing = auxiliaryInstruments(of: instruments)
            .filter(auxiliaryImpliesSampledGeneration)
        guard !forcing.isEmpty else { return nil }
        return "This run records answer-token logprobs AND sampled "
            + "generations: \(forcing.joined(separator: ", ")) scores "
            + "generated responses, so sampled generation still runs even "
            + "in Answer-token mode. Remove it for a logprob-only run."
    }
}
