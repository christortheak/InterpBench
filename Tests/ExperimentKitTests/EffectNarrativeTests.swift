import Foundation
import Testing

@testable import ExperimentKit

/// The plain-language layer over already-computed statistics: exact
/// sentences for every effect shape (significant, corrected-away, CI
/// crossing zero, missing CI, missing test), the dose-monotonicity verdict
/// line, and the pure chart-data preparation (dose ladders from effect rows
/// and from sweep grids). Views render these strings verbatim, so the
/// strings themselves are the contract.
@Suite struct EffectNarrativeTests {

    private func row(
        condition: String = "fear-steered",
        metric: String = "fearMarkerDensity",
        n: Int = 24,
        meanDiff: Double = 0.31,
        ciLower: Double = 0.12,
        ciUpper: Double = 0.48,
        wilcoxonP: Double? = nil,
        adjustedP: Double? = nil,
        correction: String? = nil
    ) -> RunResults.EffectSizeRow {
        RunResults.EffectSizeRow(
            condition: condition, metric: metric, n: n, meanDiff: meanDiff,
            ciLower: ciLower, ciUpper: ciUpper, wilcoxonW: nil,
            wilcoxonP: wilcoxonP, adjustedP: adjustedP, correction: correction,
            modality: nil)
    }

    // MARK: - A sentence per effect

    @Test func survivesCorrectionSentence() {
        let sentence = EffectNarrative.sentence(
            for: row(adjustedP: 0.012, correction: "BH"))
        #expect(
            sentence
                == "Condition 'fear-steered' shifted 'fear' marker density "
                + "(fearMarkerDensity) by +0.31 across 24 paired items "
                + "(95% CI 0.12 to 0.48) — survives multiple-comparison "
                + "correction (corrected p = 0.012, BH).")
    }

    @Test func interventionSubjectLeadsWithConceptLayerStrength() {
        let sentence = EffectNarrative.sentence(
            for: row(adjustedP: 0.012, correction: "BH"),
            intervention: "fear L12 α0.8")
        #expect(sentence.hasPrefix(
            "Steering 'fear' at layer 12 (strength 0.8) shifted"))
    }

    @Test func matchedNormControlSubject() {
        let sentence = EffectNarrative.sentence(
            for: row(condition: "fear-random-control"),
            intervention: "matched-norm random control")
        #expect(sentence.hasPrefix(
            "The random-direction control 'fear-random-control'"))
    }

    @Test func ciCrossingZeroReadsConsistentWithNoEffect() {
        let sentence = EffectNarrative.sentence(
            for: row(meanDiff: 0.05, ciLower: -0.02, ciUpper: 0.12))
        #expect(sentence.contains(
            "the interval crosses zero, so this is consistent with no effect"))
        #expect(!sentence.contains("survives"))
    }

    @Test func ciCrossingZeroWithFailedCorrectionSaysBoth() {
        let sentence = EffectNarrative.sentence(
            for: row(
                meanDiff: 0.05, ciLower: -0.02, ciUpper: 0.12, adjustedP: 0.4))
        #expect(sentence.contains("consistent with no effect"))
        #expect(sentence.contains(
            "and does not survive multiple-comparison correction"))
    }

    @Test func ciCrossingZeroButCorrectedSignificantIsAnHonestConflict() {
        let sentence = EffectNarrative.sentence(
            for: row(
                meanDiff: 0.05, ciLower: -0.01, ciUpper: 0.12, adjustedP: 0.03))
        #expect(sentence.contains("consistent with no effect"))
        #expect(sentence.contains(
            "(though the corrected p = 0.03 is below 0.05)"))
    }

    @Test func notSignificantAfterCorrectionIsSuggestiveOnly() {
        let sentence = EffectNarrative.sentence(
            for: row(adjustedP: 0.08, correction: "Holm"))
        #expect(sentence.contains(
            "does not survive multiple-comparison correction "
                + "(corrected p = 0.08, Holm) — treat as suggestive only"))
    }

    @Test func uncorrectedSignificantCarriesTheCaveat() {
        let sentence = EffectNarrative.sentence(for: row(wilcoxonP: 0.01))
        #expect(sentence.contains(
            "uncorrected p = 0.01 — no multiple-comparison correction was applied"))
    }

    @Test func uncorrectedNonSignificant() {
        // CI excludes zero but the only test says not significant — both
        // facts render, neither is hidden.
        let sentence = EffectNarrative.sentence(for: row(wilcoxonP: 0.2))
        #expect(sentence.contains("(95% CI 0.12 to 0.48)"))
        #expect(sentence.contains("uncorrected p = 0.2 (not significant)"))
    }

    @Test func missingCISaysSo() {
        let sentence = EffectNarrative.sentence(
            for: row(ciLower: .nan, ciUpper: .nan, wilcoxonP: 0.03))
        #expect(sentence.contains("(no confidence interval available)"))
        #expect(sentence.contains("uncorrected p = 0.03"))
    }

    @Test func missingCIWithCorrectionRendersCorrectedVerdict() {
        let significant = EffectNarrative.sentence(
            for: row(ciLower: .nan, ciUpper: .nan, adjustedP: 0.02))
        #expect(significant.contains(
            "corrected p = 0.02 — significant after multiple-comparison correction"))
        let notSignificant = EffectNarrative.sentence(
            for: row(ciLower: .nan, ciUpper: .nan, adjustedP: 0.3))
        #expect(notSignificant.contains(
            "corrected p = 0.3 — not significant after correction"))
    }

    @Test func noTestStatisticAtAll() {
        let sentence = EffectNarrative.sentence(
            for: row(ciLower: .nan, ciUpper: .nan))
        #expect(sentence.hasSuffix("— no test statistic available."))
    }

    @Test func tinyPValuesAvoidScientificNotation() {
        let sentence = EffectNarrative.sentence(for: row(adjustedP: 0.00001))
        #expect(sentence.contains("corrected p = < 0.0001"))
    }

    @Test func zeroNOmitsThePairedItemsClause() {
        let sentence = EffectNarrative.sentence(for: row(n: 0))
        #expect(!sentence.contains("paired items"))
    }

    // MARK: - Metric phrases (plain words first, engine term second)

    @Test func metricPhrases() {
        #expect(EffectNarrative.metricPhrase("wordCount")
            == "response length in words (wordCount)")
        #expect(EffectNarrative.metricPhrase("distinct2")
            == "lexical variety (distinct2)")
        #expect(EffectNarrative.metricPhrase("fearMarkerDensity")
            == "'fear' marker density (fearMarkerDensity)")
        #expect(EffectNarrative.metricPhrase("rs_hedging")
            == "reasoning-style feature 'hedging' (rs_hedging)")
        #expect(EffectNarrative.metricPhrase("holdingShift") == "'holdingShift'")
    }

    /// The ordinalScale instrument's effect metric ("ordinalPosition" — the
    /// pinned cross-engine endpoint name) reads as a scale position, and the
    /// server's choice endpoint name gets the same log-odds phrase as
    /// Swift's.
    @Test func ordinalAndChoiceEndpointPhrases() {
        #expect(EffectNarrative.metricPhrase("ordinalPosition")
            == "scale position (1–K) (ordinalPosition)")
        #expect(EffectNarrative.metricPhrase("choiceLogOdds")
            == "the target option's log odds (choiceLogOdds)")
        let sentence = EffectNarrative.sentence(
            for: row(
                condition: "steered", metric: "ordinalPosition", n: 2,
                meanDiff: 0.5, ciLower: 0.3, ciUpper: 0.7, wilcoxonP: 0.5))
        #expect(sentence.contains(
            "shifted scale position (1–K) (ordinalPosition) by +0.5"))
    }

    // MARK: - Intervention-summary parsing (round-trip with the formatter)

    @Test func singleSlotParsesTheRealFormatterOutput() {
        // Round-trip against RunResults.interventionSummaries so format
        // drift there breaks THIS test, not the chart silently.
        var manifest = ExperimentManifest(
            name: "narrative-fixture", description: "", modelID: "test/model")
        manifest.conditions = [
            .init(
                name: "fear-a",
                slots: [.init(concept: "fear", layer: 14, alpha: 0.8)]),
            .init(
                name: "fear-b",
                slots: [.init(concept: "fear", layer: 14, alpha: 2)]),
            .init(
                name: "mix",
                slots: [
                    .init(concept: "fear", layer: 14, alpha: 0.8),
                    .init(concept: "joy", layer: 10, alpha: 0.5),
                ]),
            .init(
                name: "control",
                slots: [.init(concept: "fear", layer: 14, alpha: 0.8)],
                controlType: "randomMatchedNorm"),
        ]
        let summaries = RunResults.interventionSummaries(manifest: manifest)

        let a = EffectNarrative.singleSlotIntervention(summaries["fear-a"] ?? "")
        #expect(a?.concept == "fear")
        #expect(a?.layer == 14)
        #expect(a?.alpha == 0.8)
        // Integer-formatted alphas ("α2") parse too.
        let b = EffectNarrative.singleSlotIntervention(summaries["fear-b"] ?? "")
        #expect(b?.alpha == 2)
        // Mixes have no single dose; controls are not concept steering;
        // baseline is "no intervention".
        #expect(EffectNarrative.singleSlotIntervention(summaries["mix"] ?? "") == nil)
        #expect(EffectNarrative.singleSlotIntervention(summaries["control"] ?? "") == nil)
        #expect(EffectNarrative.singleSlotIntervention(
            summaries[RunResults.baselineConditionName] ?? "") == nil)
    }

    @Test func singleSlotRejectsMalformedStrings() {
        #expect(EffectNarrative.singleSlotIntervention("no intervention") == nil)
        #expect(EffectNarrative.singleSlotIntervention("no slots") == nil)
        #expect(EffectNarrative.singleSlotIntervention("fear L14") == nil)
        #expect(EffectNarrative.singleSlotIntervention("fear Lx αy") == nil)
        #expect(EffectNarrative.singleSlotIntervention("") == nil)
    }

    // MARK: - Dose series from a run's effect sizes

    @Test func doseSeriesGroupsLaddersAndSortsByAlpha() {
        let interventions = [
            "fear-hi": "fear L14 α0.8",
            "fear-lo": "fear L14 α0.2",
            "fear-mid": "fear L14 α0.4",
            "joy-only": "joy L10 α0.5",  // single alpha — no ladder
            "control": "matched-norm random control",
        ]
        let rows = [
            row(condition: "fear-hi", meanDiff: 0.5),
            row(condition: "fear-lo", meanDiff: 0.1),
            row(condition: "fear-mid", meanDiff: 0.3),
            row(condition: "joy-only", metric: "fearMarkerDensity", meanDiff: 0.2),
            row(condition: "control", meanDiff: 0.05),
            // A second metric ladders independently.
            row(condition: "fear-hi", metric: "wordCount", meanDiff: 12),
            row(condition: "fear-lo", metric: "wordCount", meanDiff: 4),
        ]
        let series = EffectNarrative.doseSeries(
            effectSizes: rows, interventions: interventions)
        #expect(series.count == 2)
        let density = series.first { $0.metric == "fearMarkerDensity" }
        #expect(density?.concept == "fear")
        #expect(density?.layer == 14)
        #expect(density?.points.map(\.alpha) == [0.2, 0.4, 0.8])
        #expect(density?.points.map(\.effect) == [0.1, 0.3, 0.5])
        // CI bounds travel onto the points.
        #expect(density?.points.first?.ciLower == 0.12)
        #expect(density?.points.first?.ciUpper == 0.48)
        let words = series.first { $0.metric == "wordCount" }
        #expect(words?.points.map(\.effect) == [4, 12])
    }

    @Test func doseSeriesNeedsTwoDistinctAlphas() {
        let interventions = ["only": "fear L14 α0.8"]
        let series = EffectNarrative.doseSeries(
            effectSizes: [row(condition: "only")], interventions: interventions)
        #expect(series.isEmpty)
    }

    // MARK: - Dose points from a sweep grid

    private func sweepRow(
        concept: String = "fear", layer: Int = 14, alpha: Double,
        density: Double, objective: Double? = nil
    ) -> SweepRunCatalog.Row {
        SweepRunCatalog.Row(
            concept: concept, layer: layer, alpha: alpha,
            markerDensity: density, distinct2: 0.6, batteryAccuracy: 1.0,
            objective: objective)
    }

    @Test func sweepDosePointsReadTheDeclaredObjective() {
        let rows = [
            sweepRow(alpha: 0, density: 0.01),  // not baseline (layer 14)
            SweepRunCatalog.Row(
                concept: "fear", layer: -1, alpha: 0, markerDensity: 0.01,
                distinct2: 0.6, batteryAccuracy: 1.0),  // baseline — skipped
            sweepRow(alpha: 0.4, density: 0.3, objective: 0.62),
            sweepRow(alpha: 0.2, density: 0.2, objective: 0.55),
            sweepRow(layer: 20, alpha: 0.2, density: 0.9),  // other layer
            sweepRow(concept: "joy", alpha: 0.2, density: 0.9),  // other concept
        ]
        let density = EffectNarrative.dosePoints(
            sweepRows: rows, concept: "fear", layer: 14, metric: "markerDensity")
        #expect(density.map(\.alpha) == [0, 0.2, 0.4])
        #expect(density.map(\.effect) == [0.01, 0.2, 0.3])

        // judgeScore reads the objective column; cells without it are
        // skipped, never invented.
        let judged = EffectNarrative.dosePoints(
            sweepRows: rows, concept: "fear", layer: 14, metric: "judgeScore")
        #expect(judged.map(\.alpha) == [0.2, 0.4])
        #expect(judged.map(\.effect) == [0.55, 0.62])
    }

    // MARK: - Dose response wrapper + verdict line

    @Test func doseResponseNeedsTwoDistinctAlphas() {
        #expect(EffectNarrative.doseResponse(points: []) == nil)
        #expect(EffectNarrative.doseResponse(
            points: [.init(alpha: 0.2, effect: 0.1)]) == nil)
        #expect(EffectNarrative.doseResponse(points: [
            .init(alpha: 0.2, effect: 0.1), .init(alpha: 0.2, effect: 0.2),
        ]) == nil)
        let dose = EffectNarrative.doseResponse(points: [
            .init(alpha: 0.2, effect: 0.1), .init(alpha: 0.4, effect: 0.3),
        ])
        #expect(dose?.isMonotone == true)
        #expect(dose?.spearmanRho == 1.0)
    }

    @Test func doseSentences() {
        #expect(
            EffectNarrative.doseSentence(
                .init(spearmanRho: 0.9, isMonotone: true))
                == "effect strengthens consistently with dose (ρ = 0.90) — "
                + "good promotion evidence")
        #expect(
            EffectNarrative.doseSentence(
                .init(spearmanRho: 0.1, isMonotone: false))
                == "effect does NOT track dose (ρ = 0.10) — weak promotion evidence")
        #expect(
            EffectNarrative.doseSentence(
                .init(spearmanRho: 0.6, isMonotone: false))
                == "effect only loosely tracks dose (ρ = 0.60) — non-monotone; "
                + "weaker promotion evidence")
        #expect(
            EffectNarrative.doseSentence(
                .init(spearmanRho: -1.0, isMonotone: true))
                == "effect moves consistently DOWN as dose rises (ρ = -1.00) — "
                + "monotone, but check the sign is what you intend")
        #expect(
            EffectNarrative.doseSentence(nil)
                == "dose–response not assessable — needs at least two "
                + "strengths (α) with a measured effect")
        #expect(
            EffectNarrative.doseSentence(
                .init(spearmanRho: .nan, isMonotone: false))
                == "dose–response not assessable — the effect is flat or "
                + "the strengths are tied")
    }

    // MARK: - End-to-end sanity: sweep ladder → verdict

    @Test func monotoneSweepLadderYieldsGoodEvidenceSentence() {
        let rows = [
            sweepRow(alpha: 0.05, density: 0.10),
            sweepRow(alpha: 0.08, density: 0.18),
            sweepRow(alpha: 0.13, density: 0.31),
        ]
        let points = EffectNarrative.dosePoints(
            sweepRows: rows, concept: "fear", layer: 14, metric: "markerDensity")
        let sentence = EffectNarrative.doseSentence(
            EffectNarrative.doseResponse(points: points))
        #expect(sentence == "effect strengthens consistently with dose "
            + "(ρ = 1.00) — good promotion evidence")
    }
}
