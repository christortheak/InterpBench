import Foundation

/// The outcomes `analyze` pairs, defined once in words.
///
/// Both engines pair the SAME outcomes from the same records, under the same
/// names. Each family says how one response is read and how one item's value
/// is formed from its responses; `pairingDefinition` says what every row
/// then does with those item values. The Python engine carries the identical
/// text (`analysis_endpoints.OUTCOME_FAMILIES`), and the shared fixture
/// `Tests/Fixtures/cross-engine/effect-outcomes.json` holds both engines to
/// it — and to the same estimates from one record set.
enum StudyAnalysisOutcomes {

    /// One outcome family: its id, how its rows are named, and its
    /// definition in words.
    struct Family: Codable, Equatable {
        let id: String
        let name: String
        let definition: String
    }

    /// What one effect row is, whatever the outcome.
    static let pairingDefinition =
        "Each row compares a condition with the baseline, item by item: the "
        + "item's value in the condition minus the same item's value at "
        + "baseline, averaged over the items that have a value on both sides."

    static let families: [Family] = [
        Family(
            id: "wordCount", name: "wordCount",
            definition: "The number of words in a response. One item's value "
                + "is the mean over its responses in the condition."),
        Family(
            id: "distinct2", name: "distinct2",
            definition: "The share of a response's adjacent word pairs that "
                + "differ from one another; lower means more repetition. One "
                + "item's value is the mean over its responses in the "
                + "condition."),
        Family(
            id: "markerDensity", name: "<concept>MarkerDensity",
            definition: "How many of the concept's marker words and "
                + "characters a response contains, per word, as recorded when "
                + "the response was generated. One item's value is the mean "
                + "over its responses in the condition; a response whose "
                + "record names no value for the concept counts as zero."),
        Family(
            id: "reasoningStyle", name: "rs_<feature>",
            definition: "The reasoning-style feature's value for a response, "
                + "worked out from the response's text with the study's "
                + "pinned taxonomy. One item's value is the mean over its "
                + "responses in the condition."),
        Family(
            id: "readerScore", name: "readerScore:<concept>",
            definition: "The concept reader's score for a response's text, "
                + "as recorded when the response was generated. One item's "
                + "value is the mean over its responses that carry a score."),
        Family(
            id: "choiceRate", name: "choiceRate",
            definition: "Whether a response's parsed choice is the item's "
                + "target option. One item's value is the share of its "
                + "responses that chose the target, among those with a "
                + "readable choice; responses with no readable choice are "
                + "left out."),
        Family(
            id: "meanMonths", name: "meanMonths",
            definition: "The number the study's numeric parser read from a "
                + "response. The name is historical: the value is in months "
                + "only when the parser reads durations. One item's value is "
                + "the mean over its responses that the parser could read; "
                + "unreadable responses are left out."),
        Family(
            id: "monthsSpread", name: "monthsSpread",
            definition: "How much the parsed numbers of one item's responses "
                + "vary. One item's value is the sample standard deviation of "
                + "its readable responses, and it needs at least two of them."),
        Family(
            id: "parsedValueMean", name: "parsedValueMean",
            definition: "The same values as meanMonths, under a neutral name. "
                + "It is reported as well when the study declares a numeric "
                + "parser that does not read durations."),
        Family(
            id: "parsedValueSpread", name: "parsedValueSpread",
            definition: "The same values as monthsSpread, under a neutral "
                + "name. It is reported as well when the study declares a "
                + "numeric parser that does not read durations."),
        Family(
            id: "choiceLogOdds", name: "choiceLogOdds",
            definition: "The log-odds of the item's declared target option, "
                + "from the answer-token readout. There is one readout for "
                + "each item and condition, so nothing is averaged. An item "
                + "that declares no target has no value."),
        Family(
            id: "ordinalPosition", name: "ordinalPosition",
            definition: "The position on the study's rating scale, from the "
                + "answer-token readout. There is one readout for each item "
                + "and condition, so nothing is averaged."),
    ]

    static let markerDensitySuffix = "MarkerDensity"
    static let readerScorePrefix = "readerScore:"
    static let reasoningStylePrefix = "rs_"

    private static let definitions = Dictionary(
        uniqueKeysWithValues: families.map { ($0.id, $0.definition) })
    /// The families whose one row name IS the family id (no concept or
    /// feature in the name).
    private static let exactNames = Set(
        families.filter { $0.name == $0.id }.map(\.id))

    /// The family id of an outcome name, or "" for a name no family covers.
    ///
    /// `markerConcepts` are the concepts whose marker density the records
    /// carry; a marker-density name is recognized by them rather than by its
    /// suffix alone, so a reasoning-style feature whose id happens to end in
    /// "MarkerDensity" keeps its own family. Python twin:
    /// `analysis_endpoints.outcome_family`.
    static func family(of name: String, markerConcepts: Set<String> = []) -> String {
        if name.hasSuffix(markerDensitySuffix),
            markerConcepts.contains(String(name.dropLast(markerDensitySuffix.count)))
        {
            return "markerDensity"
        }
        if exactNames.contains(name) { return name }
        if name.hasPrefix(reasoningStylePrefix), name.count > reasoningStylePrefix.count {
            return "reasoningStyle"
        }
        if name.hasPrefix(readerScorePrefix), name.count > readerScorePrefix.count {
            return "readerScore"
        }
        return ""
    }

    /// One outcome in `outcome-coverage.json`: it reached the effect rows
    /// (`computed`), or this analysis could not produce it (`notAvailable`,
    /// with the reason in plain words).
    struct Outcome: Codable, Equatable {
        static let computed = "computed"
        static let notAvailable = "notAvailable"

        let name: String
        let family: String
        let status: String
        let definition: String
        /// Present only on `notAvailable` outcomes (nil ⇒ key omitted).
        var reason: String? = nil
    }

    /// The `outcome-coverage.json` payload (cross-engine shape; Python twin:
    /// `analysis_endpoints.outcome_coverage`).
    struct Coverage: Codable, Equatable {
        let schemaVersion: Int
        let pairing: String
        let outcomes: [Outcome]

        var notAvailable: [Outcome] {
            outcomes.filter { $0.status == Outcome.notAvailable }
        }
    }

    /// Every outcome that reached the effect rows, with its definition in
    /// words, and every outcome this analysis could NOT produce, with the
    /// reason — so an absent row is never silent.
    static func coverage(
        outcomeNames: some Sequence<String>, markerConcepts: Set<String> = [],
        notAvailable: [(name: String, family: String, reason: String)] = []
    ) -> Coverage {
        var outcomes = Set(outcomeNames).map { name -> Outcome in
            let family = family(of: name, markerConcepts: markerConcepts)
            return Outcome(
                name: name, family: family, status: Outcome.computed,
                definition: definitions[family] ?? "")
        }
        outcomes += notAvailable.map {
            Outcome(
                name: $0.name, family: $0.family, status: Outcome.notAvailable,
                definition: definitions[$0.family] ?? "", reason: $0.reason)
        }
        // By name, then computed before notAvailable — one order for a
        // given analysis, whatever order the rows were built in.
        outcomes.sort { ($0.name, $0.status) < ($1.name, $1.status) }
        return Coverage(
            schemaVersion: 1, pairing: pairingDefinition, outcomes: outcomes)
    }

    /// Why `parsedValueMean` and `parsedValueSpread` are missing when the
    /// study's declared numeric parser cannot be read: this analysis reports
    /// the parsed numbers, and cannot say whether they are months.
    static func numericParserUnreadableReason(name: String, reason: String) -> String {
        "Not available in this analysis: the study declares the numeric "
            + "parser '\(name)', but its entry could not be read. The reader "
            + "said: \(reason). Without the entry, the analysis cannot tell "
            + "whether the parsed numbers are months, so they are reported "
            + "as meanMonths and monthsSpread only. Restore "
            + "\(ParserRegistry.registryFile) as the study used it, and "
            + "analyze again."
    }
}
