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

    /// Fewer paired items than this cannot carry an interval. One or two
    /// pairs still give a bootstrap "interval", but it is two or three
    /// possible values dressed as a range, so the sentence, the table, and
    /// the chart say there are too few pairs instead of printing it.
    public static let minimumPairsForInterval = 3

    /// One plain-language line for an effect row, e.g.:
    ///
    ///   "Steering 'fear' at layer 12 (strength 0.8) shifted 'fear' marker
    ///    density (fearMarkerDensity) by +0.31 across 24 paired items
    ///    (95% CI 0.12 to 0.48) — survives multiple-comparison correction
    ///    (corrected p = 0.012)."
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
    /// A row with one or two paired items prints no interval and no test:
    /// it says there are too few pairs.
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
            + sampleClause(row.n) + ciClause(row) + " — "
            + verdict(row, familySize: familySize) + "."
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

    /// True when the row has one or two paired items — too few for an
    /// interval or a test. A row whose `n` is 0 did not report its count
    /// (the column was absent), which is a different fact and is left alone.
    public static func hasTooFewPairs(_ row: RunResults.EffectSizeRow) -> Bool {
        (1..<minimumPairsForInterval).contains(row.n)
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

    /// `n` counts ITEMS. An item sampled several times contributes one
    /// difference — its responses are averaged within each condition before
    /// the item is paired with its own baseline — so a study of four items
    /// with three responses each reads "across 4 paired items", never 12.
    private static func sampleClause(_ n: Int) -> String {
        switch n {
        case ..<1: ""
        case 1: " across 1 paired item"
        default: " across \(n) paired items"
        }
    }

    private static func ciClause(_ row: RunResults.EffectSizeRow) -> String {
        if hasTooFewPairs(row) {
            return " (too few pairs for a confidence interval; at least "
                + "\(minimumPairsForInterval) are needed)"
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
            return "with so few pairs this describes these items only, and "
                + "is not a test"
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
