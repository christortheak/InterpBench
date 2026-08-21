import Foundation

/// What a task-prompt item asks the model to EMIT — and therefore which
/// outcome instruments can legitimately read it.
///
/// The answer-token instruments (`answerTokenLogprob`, `choiceProbability`,
/// `ordinalScale`) score each option as an immediate continuation of the
/// prompt. That is only meaningful when the model was actually asked for a
/// bare label. When the prompt asks for a JSON object, the next token is `{`
/// — so scoring "A" against "B" at that position measures the model's
/// willingness to break its own output format, not its choice.
///
/// The rule-vs-justice JSONL has carried `responseFormat` since it was authored,
/// but until 2026-07-26 the field existed nowhere in executable code: it
/// survived only as a round-tripped unknown key. The consequence was
/// concrete and wrong — a reasons-arm study whose every row is `json` showed
/// the researcher "Options are present, but this study will only generate
/// and parse answer text. Declare a categorical outcome instrument
/// (answer-token probability)…", advice that would have produced a
/// meaningless measurement.
///
/// Cross-engine twin: `Server/steerlab_server/experiment/response_format.py`.
public enum ResponseFormat: String, Sendable, CaseIterable, Codable {
    /// The prompt asks for a bare option label ("A"). Answer-token scoring
    /// reads exactly the position the label occupies.
    case label
    /// The prompt asks for a JSON object (typically reasoning plus a choice
    /// field). The choice is not at the first generated position.
    case json
    /// The prompt asks for prose. No fixed answer position at all.
    case freeText

    /// True when answer-token / ordinal instruments can read this format.
    public var supportsAnswerTokenScoring: Bool { self == .label }

    /// Why a format cannot be scored — for refusals that must be actionable.
    public var scoringObjection: String? {
        switch self {
        case .label:
            return nil
        case .json:
            return "the prompt asks for a JSON object, so the first generated "
                + "token is the opening brace, not an option label"
        case .freeText:
            return "the prompt asks for prose, so no option occupies a fixed "
                + "answer position"
        }
    }

    /// Parse a declared value. Returns nil for an ABSENT field (legacy data,
    /// deliberately permissive) and throws for an unrecognised one — a typo
    /// silently degrading to "unspecified" would re-open exactly the hole
    /// this type closes.
    public static func parse(_ raw: String?) throws -> ResponseFormat? {
        guard let raw, !raw.isEmpty else { return nil }
        guard let value = ResponseFormat(rawValue: raw) else {
            throw ExperimentError(
                reason: "unknown responseFormat '\(raw)' — expected one of "
                    + allCases.map(\.rawValue).joined(separator: ", "))
        }
        return value
    }

    // MARK: The compatibility rule

    /// One item's relevant facts, so the rule below is pure.
    public struct Item: Sendable, Equatable {
        public var id: String
        public var hasOptions: Bool
        /// Whether the item declares a `target` — the option a target-
        /// dependent choice instrument reads. Defaulted `true` so every
        /// existing construction site keeps its exact meaning; the loaders
        /// pass the real value.
        public var hasTarget: Bool
        public var format: ResponseFormat?

        public init(
            id: String, hasOptions: Bool, hasTarget: Bool = true,
            format: ResponseFormat?
        ) {
            self.id = id
            self.hasOptions = hasOptions
            self.hasTarget = hasTarget
            self.format = format
        }
    }

    /// Option-consuming instruments whose ENDPOINT is the declared target's
    /// log-odds. `ordinalScale` is deliberately absent: its endpoint is the
    /// ladder position, and a rating ladder legitimately declares no target.
    /// Server twin: `response_format.TARGET_DEPENDENT_INSTRUMENTS`.
    public static let targetDependentInstruments: Set<String> = [
        "answerTokenLogprob", "choiceProbability",
    ]

    /// Items that carry options AND declare the `target` a target-dependent
    /// choice instrument reads.
    public static func targetedItems(_ items: [Item]) -> [Item] {
        items.filter { $0.hasOptions && $0.hasTarget }
    }

    /// Items that carry options but whose declared format cannot be scored
    /// by an answer-token instrument.
    ///
    /// An item with NO declared format is not counted: legacy files predate
    /// the field entirely and have been measured successfully, so treating
    /// absence as an objection would refuse studies that are fine. Absence is
    /// surfaced as an advisory elsewhere, never as a refusal.
    public static func unscorableItems(_ items: [Item]) -> [Item] {
        items.filter { item in
            guard item.hasOptions, let format = item.format else { return false }
            return !format.supportsAnswerTokenScoring
        }
    }

    /// The run-start refusal when a choice instrument is declared over items
    /// it cannot read — or over no items at all — or nil when the study is
    /// coherent. The zero-item cases (2026-08-06) close a silent hole: an
    /// option-consuming instrument declared when no in-scope item carries
    /// `options` dispatched nothing and produced zero records, burning the
    /// run's GPU time under a declaration that could never fire.
    ///
    /// `declaredScope` is the manifest's explicit applicability subset. When
    /// it is present and covers every unscorable row (by excluding it), the
    /// study is coherent: the instrument runs on the in-scope rows only.
    public static func refusal(
        items: [Item],
        declaredInstruments: [String]?,
        declaredScope: Scope?
    ) -> String? {
        let choiceInstruments = Set(declaredInstruments ?? [])
            .intersection(InstrumentActivation.optionConsumingInstruments)
        guard !choiceInstruments.isEmpty else { return nil }
        let instrument = choiceInstruments.sorted().joined(separator: ", ")

        let scoped = declaredScope.map { scope in
            items.filter(scope.includes)
        } ?? items
        if declaredScope != nil, scoped.isEmpty {
            return "the study declares \(instrument), but the declared "
                + "outcomeInstrumentScope selects zero task items — the "
                + "instrument would run on nothing and silently produce zero "
                + "records. Fix the scope's responseFormats to match the "
                + "items it should read, or drop the instrument."
        }
        if !scoped.contains(where: \.hasOptions) {
            let where_ = declaredScope != nil ? "in-scope task item" : "task item"
            let n = scoped.count
            return "the study declares \(instrument), but none of the "
                + "\(n) \(where_)\(n == 1 ? "" : "s") carries options — the "
                + "instrument scores each item's declared options ladder, so "
                + "it would silently produce zero records. Add 'options' to "
                + "the items it should read, or drop the instrument."
        }
        let unscorable = unscorableItems(scoped)
        guard !unscorable.isEmpty else {
            return targetlessRefusal(
                choiceInstruments: choiceInstruments, scoped: scoped,
                declaredScope: declaredScope)
        }

        let shown = 5
        let named = unscorable.prefix(shown).map(\.id).joined(separator: ", ")
        let more = unscorable.count > shown
            ? " … and \(unscorable.count - shown) more" : ""
        // Every distinct objection, so a mixed json/freeText file explains
        // both rather than only whichever row sorted first.
        let objections = Set(unscorable.compactMap { $0.format?.scoringObjection })
            .sorted()
            .joined(separator: "; ")
        return "the study declares \(instrument), but \(unscorable.count) "
            + "item\(unscorable.count == 1 ? "" : "s") cannot be read that way "
            + "(\(named)\(more)): \(objections). Either change those items' "
            + "responseFormat to 'label', drop the instrument, or declare "
            + "outcomeInstrumentScope to apply it to the label rows only."
    }

    /// A target-dependent choice instrument declared over items that name no
    /// `target` at all (open-issues #6).
    ///
    /// `answerTokenLogprob` / `choiceProbability` read the DECLARED target's
    /// log-odds. Until this fix the run loop invented one when the item named
    /// none (`prompt.target ?? options[0]`), which for a rating ladder stamped
    /// every record with the scale minimum. Synthesis is gone — so a study
    /// whose items never declare a target now produces NO rows for the
    /// endpoint it declared, which is exactly the failure the zero-item rules
    /// above exist to refuse before a run spends its GPU allocation.
    ///
    /// Two deliberate silences:
    ///
    /// - **Partial coverage is not the instrument's business**, the same rule
    ///   the options gate follows: a mixed file where SOME items declare a
    ///   target produces records for those items, and the untargeted rows
    ///   simply carry no choice endpoint (`ChoiceDeltas.targetIsDeclared`
    ///   skips them).
    /// - **`ordinalScale` declared alongside**: the two ride ONE record, so an
    ///   item with a ladder and no target is a legitimate ordinal-only item —
    ///   the mixed instrument (declared A/B target plus an ordinal readout) is
    ///   the case that must keep working. A ladder-only study declares
    ///   ordinalScale and never reaches here.
    ///
    /// Server twin: `response_format._targetless_refusal`.
    static func targetlessRefusal(
        choiceInstruments: Set<String>, scoped: [Item], declaredScope: Scope?
    ) -> String? {
        guard !choiceInstruments.isDisjoint(with: targetDependentInstruments),
            !choiceInstruments.contains("ordinalScale"),
            targetedItems(scoped).isEmpty
        else { return nil }
        let instrument = choiceInstruments.sorted().joined(separator: ", ")
        let where_ = declaredScope != nil ? "in-scope task item" : "task item"
        let n = scoped.count
        return "the study declares \(instrument), but none of the \(n) "
            + "\(where_)\(n == 1 ? "" : "s") declares a 'target' — the "
            + "endpoint is the DECLARED target's log-odds, so the instrument "
            + "would silently produce zero endpoint rows (it no longer guesses "
            + "the first option, which stamped rating ladders with their scale "
            + "minimum). Add 'target' to the items it should read and re-pin, "
            + "declare ordinalScale if they are a rating ladder, or drop the "
            + "instrument."
    }

    /// An explicitly declared, hash-pinned applicability subset.
    ///
    /// Running an instrument over part of a file is legitimate — a mixed
    /// file may hold a label arm and a JSON arm — but it must be DECLARED,
    /// not inferred, because "which rows were measured" is a result-bearing
    /// fact. The pin makes the subset checkable after the fact rather than
    /// recomputed from whatever the file says later.
    public struct Scope: Sendable, Equatable, Codable {
        /// Formats the option-consuming instruments apply to.
        public var responseFormats: [String]
        /// How many items that selected when declared.
        public var itemCount: Int
        /// SHA-256 over the sorted in-scope item ids.
        public var itemIDsHash: String

        public init(responseFormats: [String], itemCount: Int, itemIDsHash: String) {
            self.responseFormats = responseFormats
            self.itemCount = itemCount
            self.itemIDsHash = itemIDsHash
        }

        public func includes(_ item: ResponseFormat.Item) -> Bool {
            guard let format = item.format else {
                // An undeclared row cannot be proven in scope. Excluding it
                // is the conservative reading: a scope exists precisely
                // because the file is mixed.
                return false
            }
            return responseFormats.contains(format.rawValue)
        }

        /// Build the pin from the items a scope selects.
        public static func pin(
            responseFormats: [String], items: [ResponseFormat.Item]
        ) -> Scope {
            let draft = Scope(
                responseFormats: responseFormats, itemCount: 0, itemIDsHash: "")
            let selected = items.filter(draft.includes)
            return Scope(
                responseFormats: responseFormats,
                itemCount: selected.count,
                itemIDsHash: idsHash(selected))
        }

        public static func idsHash(_ items: [ResponseFormat.Item]) -> String {
            let joined = items.map(\.id).sorted().joined(separator: "\n")
            return ExperimentStore.sha256Hex(Data(joined.utf8))
        }

        /// Drift refusal: the pinned subset must still be the subset the file
        /// produces. Nil when it matches.
        public func driftRefusal(items: [ResponseFormat.Item]) -> String? {
            let selected = items.filter(includes)
            if selected.count != itemCount {
                return "outcomeInstrumentScope pins \(itemCount) in-scope "
                    + "item\(itemCount == 1 ? "" : "s"), but the task prompts "
                    + "now select \(selected.count) — the prompt file changed "
                    + "since the scope was declared"
            }
            let live = Scope.idsHash(selected)
            if live != itemIDsHash {
                return "outcomeInstrumentScope pins in-scope items "
                    + "\(itemIDsHash.prefix(12))…, but the task prompts now "
                    + "select \(live.prefix(12))… — the same COUNT of "
                    + "different items; the prompt file changed since the "
                    + "scope was declared"
            }
            return nil
        }
    }
}
