import Foundation

/// Plain-language rendering of the statistics the engines already compute —
/// "results a researcher can read" (usability plan Phase 2). One sentence per
/// effect row, a verdict line for dose-monotonicity, and the pure data
/// preparation behind the results charts (forest plot, dose–response).
///
/// Everything here is a pure function over already-computed values: no
/// statistics are re-derived, no thresholds invented — `EffectSizeRow`'s own
/// significance rule and `StudyStatistics.doseMonotonicity` stay the single
/// sources of truth. Views render these strings verbatim (thin-UI rule).
public enum EffectNarrative {

    // MARK: - A sentence per effect

    /// Fewer independent pairs than this cannot carry an interval. One or
    /// two pairs still give a bootstrap "interval", but it is two or three
    /// possible values dressed as a range, so the sentence, the table, and
    /// the chart say there are too few pairs instead of printing it. Paired
    /// responses are not independent of each other, so a row that paired
    /// responses counts its items here.
    public static let minimumPairsForInterval = 3

    /// What a row that paired responses is, and is not. The Python reader's
    /// results page says the same (`effect-units.json` holds the words).
    public static let responseCaveat =
        "These are responses, not items: the analysis paired each response "
        + "with the baseline response to the same item and seed, so its "
        + "interval treats responses to the same item as independent and is "
        + "not a finding about items"

    /// What a row whose unit nothing settles says about its pairs.
    public static let unknownUnitNote = "The unit of these pairs is not established"

    /// Why a row is read as paired responses (the Python reader's
    /// `RESPONSE_UNIT_EXPLANATION`, word for word).
    public static let responseUnitExplanation =
        "These rows count more pairs than the run has paired items, so the "
        + "analysis paired each response with the baseline response to the "
        + "same item and seed. The Mac engine did this before SteerLab 0.9.7 "
        + "when an item was sampled more than once, and stamped no unit. "
        + "Responses to the same item are not independent, so these intervals "
        + "are narrower than item-level intervals and are not findings about "
        + "items. The stored numbers are copied unchanged; analyzing the run "
        + "again computes item-level rows."

    /// One plain-language line for an effect row, e.g.:
    ///
    ///   "Steering 'fear' at layer 12 (strength 0.8) shifted 'fear' marker
    ///    density (fearMarkerDensity) by +0.31 across 24 paired items
    ///    (95% CI 0.12 to 0.48) — survives multiple-comparison correction
    ///    (corrected p = 0.012)."
    ///
    /// The count is in the row's settled unit (`EffectSizeRow.unit`): paired
    /// items, transcripts, or samples; "8 paired responses from 4 items",
    /// followed by `responseCaveat`, for an analysis that paired responses;
    /// and plain "pairs", followed by `unknownUnitNote`, when nothing
    /// settles the unit.
    ///
    /// Honest edge cases are first-class: a CI crossing zero reads
    /// "consistent with no effect", a corrected p ≥ 0.05 reads "does not
    /// survive multiple-comparison correction", and a missing CI or missing
    /// test says so instead of implying certainty.
    ///
    /// `familySize` is how many comparisons the row's correction covered
    /// (`correctionFamilySize`). A correction over ONE comparison changes
    /// nothing — the adjusted p equals the raw one — so the sentence says
    /// an effect "survives correction" only when the family has more than
    /// one member; otherwise it says what was tested. It is a required
    /// argument on purpose: a caller that does not know the family cannot
    /// claim a correction.
    ///
    /// A row with one or two pairs (for paired responses, one or two items)
    /// prints no interval and no test: it says there are too few.
    ///
    /// `intervention` is the run's intervention summary for the row's
    /// condition (`RunResults.interventionSummaries`); when it names a single
    /// steering slot the sentence leads with the concept/layer/strength,
    /// otherwise with the condition name.
    public static func sentence(
        for row: RunResults.EffectSizeRow, intervention: String? = nil,
        familySize: Int
    ) -> String {
        "\(subject(condition: row.condition, intervention: intervention)) "
            + "shifted \(metricPhrase(row.metric)) by \(signed(row.meanDiff))"
            + (countPhrase(row).map { " across " + $0 } ?? "") + ciClause(row)
            + " — " + verdict(row, familySize: familySize) + "." + unitNote(row)
    }

    /// The sentence for a row of `table`, with the row's correction family
    /// counted from the table it came from.
    public static func sentence(
        for row: RunResults.EffectSizeRow, in table: [RunResults.EffectSizeRow],
        intervention: String? = nil
    ) -> String {
        sentence(
            for: row, intervention: intervention,
            familySize: correctionFamilySize(of: row, in: table))
    }

    /// How many comparisons the row's multiple-comparison correction
    /// covered. Both engines correct one outcome at a time, across the
    /// conditions that have a defined test for it, so the family is the
    /// table's rows for the same outcome that carry an adjusted p. `table`
    /// is the pooled table the row came from.
    public static func correctionFamilySize(
        of row: RunResults.EffectSizeRow, in table: [RunResults.EffectSizeRow]
    ) -> Int {
        table.filter { $0.metric == row.metric && $0.adjustedP != nil }.count
    }

    /// True when the row has one or two independent pairs — too few for an
    /// interval or a test. Paired responses are not independent of each
    /// other, so a row that paired responses counts its ITEMS: five
    /// responses from one item are one item. A row whose `n` is 0 did not
    /// report its count (the column was absent), which is a different fact
    /// and is left alone.
    public static func hasTooFewPairs(_ row: RunResults.EffectSizeRow) -> Bool {
        let independent = row.unit.isResponses ? row.unit.pairedItems : row.n
        guard let independent else { return false }
        return (1..<minimumPairsForInterval).contains(independent)
    }

    /// What `hasTooFewPairs` counted: "items" for a row that paired
    /// responses, "pairs" otherwise.
    public static func tooFewNoun(_ row: RunResults.EffectSizeRow) -> String {
        row.unit.isResponses ? "items" : "pairs"
    }

    /// The row's count in its own unit, as the Python reader's results page
    /// states it: "24 paired items", "4 paired transcripts", "3 pairs", or
    /// "8 paired responses from 4 items". nil when the file gave no count.
    public static func countPhrase(_ row: RunResults.EffectSizeRow) -> String? {
        guard row.n >= 1 else { return nil }
        let (one, many) = unitNouns[row.unit.unit] ?? ("paired unit", "paired units")
        var text = plural(row.n, one, many)
        if row.unit.isResponses, let items = row.unit.pairedItems {
            text += " from " + plural(items, "item", "items")
        }
        return text
    }

    /// The nouns a count takes in each unit (the Python reader's `_UNITS`).
    private static let unitNouns: [String: (String, String)] = [
        "item": ("paired item", "paired items"),
        "transcript": ("paired transcript", "paired transcripts"),
        "sample": ("paired sample", "paired samples"),
        "response": ("paired response", "paired responses"),
        "unknown": ("pair", "pairs"),
    ]

    private static func plural(_ count: Int, _ one: String, _ many: String) -> String {
        "\(count) \(count == 1 ? one : many)"
    }

    /// What follows the sentence for a row that is not about items:
    /// `responseCaveat`, or `unknownUnitNote`.
    private static func unitNote(_ row: RunResults.EffectSizeRow) -> String {
        if row.unit.isResponses { return " " + responseCaveat + "." }
        if row.unit.unit == "unknown" { return " " + unknownUnitNote + "." }
        return ""
    }

    /// The unit cell of the effect table: the settled unit, marked when the
    /// analysis did not record it itself — "item (default)" when the run's
    /// records agree with the engines' rule, "response (from the records)"
    /// when they show the row paired responses (the Python reader's page
    /// marks its table the same way).
    public static func unitLabel(_ row: RunResults.EffectSizeRow) -> String {
        switch row.unit.source {
        case .engineDefault: row.unit.unit + " (default)"
        case .inferredFromRecords: row.unit.unit + " (from the records)"
        case .recorded, .notEstablished: row.unit.unit
        }
    }

    /// The forest plot's note on the rows it draws no whisker for, or nil
    /// when every row has enough independent pairs for an interval.
    public static func tooFewCaption(_ rows: [RunResults.EffectSizeRow]) -> String? {
        let few = rows.filter(hasTooFewPairs)
        guard !few.isEmpty else { return nil }
        let noun = Set(few.map(\.unit.unit)) == ["item"] ? "paired items" : "pairs"
        var text = "A row with fewer than \(minimumPairsForInterval) \(noun) "
            + "has no whisker: that is too few pairs for an interval"
        if few.contains(where: \.unit.isResponses) {
            text += ". A row of paired responses counts its items, since "
                + "responses to the same item are not independent"
        }
        return text
    }

    // MARK: - The unit of analysis, in words

    /// One "Unit of analysis" line per way the rows' units are known, in the
    /// order they first appear: the Python reader's methods summary
    /// (`unit_lines`), for the app's effect table. `records` is what the
    /// units were settled against (`Model.effectUnitRecords`), which says
    /// why a unit is not established.
    public static func unitLines(
        rows: [RunResults.EffectSizeRow], records: RunResults.PairedItems?
    ) -> [String] {
        var groups: [(source: RunResults.EffectUnit.Source, units: [String])] = []
        for row in rows {
            if let index = groups.firstIndex(where: { $0.source == row.unit.source }) {
                if !groups[index].units.contains(row.unit.unit) {
                    groups[index].units.append(row.unit.unit)
                }
            } else {
                groups.append((row.unit.source, [row.unit.unit]))
            }
        }
        return groups.map { group in
            let marked = groups.count > 1
                ? " for the rows marked " + joined(group.units) : ""
            let said: String
            switch group.source {
            case .recorded:
                said = joined(group.units) + ", as the analysis recorded."
            case .engineDefault:
                said = "the item, with an item's samples averaged within each "
                    + "condition. This is the engines' documented default; the "
                    + "analysis did not stamp the unit itself. The run's records "
                    + "agree: no such row counts more pairs than the items "
                    + "paired in the run."
            case .inferredFromRecords:
                said = "the response, not the item. " + responseUnitExplanation
            case .notEstablished:
                if records == nil {
                    said = "not established. The analysis did not stamp it, and "
                        + "the run's records are not available here."
                } else if records?.complete == false {
                    said = "not established. The analysis did not stamp it, and "
                        + "only the first part of the run's records was read "
                        + "here, which does not settle it."
                } else {
                    said = "not established. The analysis did not stamp it, and "
                        + "the run's records have no items paired with the "
                        + "baseline for these conditions."
                }
            }
            return "Unit of analysis\(marked): \(said)"
        }
    }

    private static func joined(_ items: [String]) -> String {
        switch items.count {
        case 0: ""
        case 1: items[0]
        case 2: "\(items[0]) and \(items[1])"
        default: items.dropLast().joined(separator: ", ") + ", and \(items.last!)"
        }
    }

    /// Whether the row's interval should be shown at all: it exists, and
    /// the row has enough pairs to carry one.
    public static func hasReportableInterval(
        _ row: RunResults.EffectSizeRow
    ) -> Bool {
        hasCI(row) && !hasTooFewPairs(row)
    }

    private static func subject(condition: String, intervention: String?) -> String {
        if let intervention {
            if let slot = singleSlotIntervention(intervention) {
                return "Steering '\(slot.concept)' at layer \(slot.layer) "
                    + "(strength \(plain(slot.alpha)))"
            }
            if intervention == "matched-norm random control" {
                return "The random-direction control '\(condition)'"
            }
        }
        return "Condition '\(condition)'"
    }

    private static func ciClause(_ row: RunResults.EffectSizeRow) -> String {
        if hasTooFewPairs(row) {
            return " (too few \(tooFewNoun(row)) for a confidence interval; at "
                + "least \(minimumPairsForInterval) are needed)"
        }
        guard hasCI(row) else { return " (no confidence interval available)" }
        return " (95% CI \(plain(row.ciLower)) to \(plain(row.ciUpper)))"
    }

    private static func hasCI(_ row: RunResults.EffectSizeRow) -> Bool {
        row.ciLower.isFinite && row.ciUpper.isFinite
    }

    /// The verdict clause after the dash. Ordered by what the reader must
    /// not misread: too few pairs → say so and stop; no CI → say so; CI
    /// crossing zero → "consistent with no effect" (with the p noted when
    /// it disagrees); CI excluding zero → the correction verdict when a
    /// correction really covered several comparisons, what was tested when
    /// it covered one, or the honest "uncorrected" caveat.
    private static func verdict(
        _ row: RunResults.EffectSizeRow, familySize: Int
    ) -> String {
        if hasTooFewPairs(row) {
            return "with so few \(tooFewNoun(row)) this describes these items "
                + "only, and is not a test"
        }
        let corrected = familySize > 1
        guard hasCI(row) else { return noCIVerdict(row, corrected: corrected) }
        if !row.ciExcludesZero {
            var text = "the interval crosses zero, so this is consistent "
                + "with no effect"
            if let adjusted = row.adjustedP {
                if corrected {
                    text += row.significantAfterCorrection == true
                        ? " (though the corrected p = \(pValue(adjusted)) is below 0.05)"
                        : " and does not survive multiple-comparison correction"
                } else {
                    text += adjusted < 0.05
                        ? " (though \(singleTest(adjusted)) is below 0.05)"
                        : " (\(singleTest(adjusted)))"
                }
            }
            return text
        }
        if let adjusted = row.adjustedP {
            guard corrected else {
                return singleTest(adjusted)
                    + (adjusted < 0.05
                        ? "" : " — not significant; treat as suggestive only")
            }
            return row.significantAfterCorrection == true
                ? "survives multiple-comparison correction "
                    + "(corrected p = \(pValue(adjusted))\(correctionSuffix(row)))"
                : "does not survive multiple-comparison correction "
                    + "(corrected p = \(pValue(adjusted))\(correctionSuffix(row))) "
                    + "— treat as suggestive only"
        }
        if let p = row.wilcoxonP {
            return p < 0.05
                ? "uncorrected p = \(pValue(p)) — no multiple-comparison "
                    + "correction was applied"
                : "uncorrected p = \(pValue(p)) (not significant)"
        }
        return "no test statistic available"
    }

    private static func noCIVerdict(
        _ row: RunResults.EffectSizeRow, corrected: Bool
    ) -> String {
        if let adjusted = row.adjustedP {
            guard corrected else {
                return singleTest(adjusted)
                    + (adjusted < 0.05 ? "" : " — not significant")
            }
            return row.significantAfterCorrection == true
                ? "corrected p = \(pValue(adjusted))\(correctionSuffix(row)) — "
                    + "significant after multiple-comparison correction"
                : "corrected p = \(pValue(adjusted))\(correctionSuffix(row)) — "
                    + "not significant after correction"
        }
        if let p = row.wilcoxonP {
            return "uncorrected p = \(pValue(p)) — no multiple-comparison "
                + "correction was applied"
        }
        return "no test statistic available"
    }

    /// What was tested when the correction family has one member: the one
    /// Wilcoxon signed-rank test of this outcome. Nothing was corrected, so
    /// nothing "survives" anything.
    private static func singleTest(_ p: Double) -> String {
        "p = \(pValue(p)) from the one Wilcoxon signed-rank test of this "
            + "outcome (a single comparison, so no correction applies)"
    }

    private static func correctionSuffix(_ row: RunResults.EffectSizeRow) -> String {
        row.correction.map { ", \($0)" } ?? ""
    }

    /// Plain words first, the technical term second (the usability plan's
    /// language rule): known engine metric names get a readable phrase with
    /// the engine term in parentheses; unknown names pass through quoted.
    ///
    /// The phrases have one home, the shared headline-outcome mapping
    /// (`HeadlineOutcome.plainPhrase`), so the app, both command lines, and
    /// the results explorer use the same words for the same outcome. (One
    /// of them, `meanMonths`, is a deprecated alias: any declared registry
    /// parser writes the same record key, so on other parsers the honest
    /// twin is `parsedValueMean`.)
    public static func metricPhrase(_ metric: String) -> String {
        guard let plain = HeadlineOutcome.plainPhrase(metric) else {
            return "'\(metric)'"
        }
        return "\(plain) (\(metric))"
    }

    // MARK: - The headline (what a results summary leads with)

    /// What a results header shows first: which outcome leads and which
    /// rule chose it, then the plain sentence for each condition's row of
    /// that outcome.
    public struct Headline: Sendable, Equatable {
        /// "Headline outcome: the target-choice rate (choiceRate), chosen
        /// by default order."
        public var title: String
        /// One sentence per condition for the headline outcome, in the
        /// table's own order. Empty when the headline is a judged outcome
        /// (`note` then says where it is) or the run has no rows.
        public var sentences: [String]
        /// Where to look when the headline is not in the effect rows.
        public var note: String?
    }

    /// The header for a run whose headline is `selection`, or nil when the
    /// run has nothing to lead with and nothing was declared. `rows` is the
    /// pooled effect table, untouched: this picks which rows LEAD, and the
    /// full table still follows in the engine's order.
    public static func headline(
        _ selection: HeadlineOutcome.Selection,
        rows: [RunResults.EffectSizeRow],
        interventions: [String: String] = [:]
    ) -> Headline? {
        guard let outcome = selection.outcome else {
            guard selection.declaredAbsent else { return nil }
            return Headline(
                title: "Headline outcome: " + selection.chosenBy + ".",
                sentences: [], note: nil)
        }
        let title = "Headline outcome: \(selection.summaryLine)."
        if selection.source == .evaluationReport {
            return Headline(
                title: title, sentences: [],
                note: "The judged outcome is in this run's evaluation "
                    + "report (judge-report.json or coding-report.json). "
                    + "The effect sizes below are the study's other "
                    + "measures.")
        }
        return Headline(
            title: title,
            sentences: rows.filter { $0.metric == outcome }.map {
                sentence(
                    for: $0, in: rows, intervention: interventions[$0.condition])
            },
            note: nil)
    }

    // MARK: - Dose–response verdict (the promote decision's plain line)

    /// The plain line for a dose-monotonicity result, e.g.
    /// "effect strengthens consistently with dose (ρ = 0.90) — good
    /// promotion evidence". nil (or an undefined ρ with too few points)
    /// reads as not assessable rather than as evidence either way.
    public static func doseSentence(_ dose: StudyStatistics.DoseResponse?) -> String {
        guard let dose else {
            return "dose–response not assessable — needs at least two "
                + "strengths (α) with a measured effect"
        }
        let rho = dose.spearmanRho
        if dose.isMonotone {
            var text = "effect strengthens consistently with dose "
                + "(ρ = \(rhoText(rho))) — good promotion evidence"
            if rho < 0 {
                text = "effect moves consistently DOWN as dose rises "
                    + "(ρ = \(rhoText(rho))) — monotone, but check the sign "
                    + "is what you intend"
            }
            return text
        }
        if rho.isNaN {
            return "dose–response not assessable — the effect is flat or "
                + "the strengths are tied"
        }
        if abs(rho) >= 0.5 {
            return "effect only loosely tracks dose (ρ = \(rhoText(rho))) — "
                + "non-monotone; weaker promotion evidence"
        }
        return "effect does NOT track dose (ρ = \(rhoText(rho))) — weak "
            + "promotion evidence"
    }

    // MARK: - Dose data preparation (charts + verdicts)

    /// One (strength, effect) point on a dose ladder. CI bounds present only
    /// when the source carries them (analyze artifacts do; sweep grids don't).
    public struct DosePoint: Sendable, Equatable {
        public var alpha: Double
        public var effect: Double
        public var ciLower: Double?
        public var ciUpper: Double?

        public init(
            alpha: Double, effect: Double,
            ciLower: Double? = nil, ciUpper: Double? = nil
        ) {
            self.alpha = alpha
            self.effect = effect
            self.ciLower = ciLower
            self.ciUpper = ciUpper
        }
    }

    /// A dose ladder for one (concept, layer, metric) — the unit a
    /// dose–response chart draws as one line.
    public struct DoseSeries: Sendable, Equatable, Identifiable {
        public var concept: String
        public var layer: Int
        public var metric: String
        /// Sorted by alpha ascending; always ≥ 2 distinct alphas.
        public var points: [DosePoint]

        public var id: String { "\(concept)\u{1F}L\(layer)\u{1F}\(metric)" }
        public var label: String { "\(concept) L\(layer)" }

        public init(concept: String, layer: Int, metric: String, points: [DosePoint]) {
            self.concept = concept
            self.layer = layer
            self.metric = metric
            self.points = points
        }
    }

    /// Parse a SINGLE-slot intervention summary back into its parts.
    ///
    /// `RunResults.interventionSummaries` formats one slot as
    /// "<concept> L<layer> α<alpha>"; controls, mixes (" + "), and composed
    /// summaries (" · ") return nil — a mixed condition has no single dose.
    /// A round-trip test against the real formatter pins this format.
    public static func singleSlotIntervention(
        _ summary: String
    ) -> (concept: String, layer: Int, alpha: Double)? {
        guard !summary.contains(" · "), !summary.contains(" + ") else { return nil }
        let tokens = summary.split(separator: " ")
        guard tokens.count >= 3 else { return nil }
        let alphaToken = tokens[tokens.count - 1]
        let layerToken = tokens[tokens.count - 2]
        guard alphaToken.hasPrefix("α"),
            let alpha = Double(alphaToken.dropFirst()),
            layerToken.hasPrefix("L"),
            let layer = Int(layerToken.dropFirst())
        else { return nil }
        let concept = tokens.dropLast(2).joined(separator: " ")
        guard !concept.isEmpty else { return nil }
        return (concept, layer, alpha)
    }

    /// Dose ladders hiding in a study run's effect sizes: group single-slot
    /// conditions by (concept, layer, metric) and keep the groups with at
    /// least two distinct strengths. Deterministic order (concept, layer,
    /// metric ascending); points sorted by alpha.
    public static func doseSeries(
        effectSizes: [RunResults.EffectSizeRow],
        interventions: [String: String]
    ) -> [DoseSeries] {
        struct Key: Hashable {
            let concept: String
            let layer: Int
            let metric: String
        }
        var grouped: [Key: [DosePoint]] = [:]
        for row in effectSizes {
            guard let summary = interventions[row.condition],
                let slot = singleSlotIntervention(summary),
                row.meanDiff.isFinite
            else { continue }
            let key = Key(concept: slot.concept, layer: slot.layer, metric: row.metric)
            grouped[key, default: []].append(
                DosePoint(
                    alpha: slot.alpha,
                    effect: row.meanDiff,
                    ciLower: row.ciLower.isFinite ? row.ciLower : nil,
                    ciUpper: row.ciUpper.isFinite ? row.ciUpper : nil))
        }
        return grouped
            .filter { Set($0.value.map(\.alpha)).count >= 2 }
            .map { key, points in
                DoseSeries(
                    concept: key.concept, layer: key.layer, metric: key.metric,
                    points: points.sorted { $0.alpha < $1.alpha })
            }
            .sorted {
                ($0.concept, $0.layer, $0.metric) < ($1.concept, $1.layer, $1.metric)
            }
    }

    /// The alpha ladder of one sweep-grid (concept, layer) under the sweep's
    /// declared objective: markerDensity reads the density column, any other
    /// objective reads the recorded `objective` value (cells that predate the
    /// column are skipped, never invented). Baseline rows are excluded — the
    /// ladder is the swept cells. Sorted by alpha.
    public static func dosePoints(
        sweepRows: [SweepRunCatalog.Row], concept: String, layer: Int, metric: String
    ) -> [DosePoint] {
        sweepRows
            .filter { $0.concept == concept && $0.layer == layer && !$0.isBaseline }
            .compactMap { row -> DosePoint? in
                let effect = metric == "markerDensity" ? row.markerDensity : row.objective
                guard let effect, effect.isFinite else { return nil }
                return DosePoint(alpha: row.alpha, effect: effect)
            }
            .sorted { $0.alpha < $1.alpha }
    }

    /// Dose-monotonicity over prepared points — the one wrapper the UI calls
    /// so `StudyStatistics.doseMonotonicity` (already unit-tested) stays the
    /// single implementation. nil when the ladder is too short to assess.
    public static func doseResponse(
        points: [DosePoint]
    ) -> StudyStatistics.DoseResponse? {
        guard Set(points.map(\.alpha)).count >= 2 else { return nil }
        return StudyStatistics.doseMonotonicity(
            alphas: points.map(\.alpha), effects: points.map(\.effect))
    }

    // MARK: - Number formatting

    /// Signed effect magnitude, ≤ 3 significant digits ("+0.31", "-12.4").
    static func signed(_ value: Double) -> String {
        guard value.isFinite else { return "?" }
        return String(format: "%+.3g", value)
    }

    /// Unsigned-format number, ≤ 3 significant digits ("0.12", "-0.48").
    static func plain(_ value: Double) -> String {
        guard value.isFinite else { return "?" }
        return String(format: "%.3g", value)
    }

    /// p-values: tiny ones say "< 0.0001" instead of scientific notation.
    static func pValue(_ p: Double) -> String {
        guard p.isFinite else { return "?" }
        if p < 0.0001 { return "< 0.0001" }
        return String(format: "%.4g", p)
    }

    /// Spearman ρ at two decimals (the conventional display precision).
    static func rhoText(_ rho: Double) -> String {
        guard rho.isFinite else { return "undefined" }
        return String(format: "%.2f", rho)
    }
}
