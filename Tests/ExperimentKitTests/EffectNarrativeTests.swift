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
        correction: String? = nil,
        unit: RunResults.EffectUnit = .init(unit: "item", source: .engineDefault)
    ) -> RunResults.EffectSizeRow {
        // An item-level row, as the Results model settles a current run's
        // rows against its records (`EffectUnitTests` covers the settling).
        RunResults.EffectSizeRow(
            condition: condition, metric: metric, n: n, meanDiff: meanDiff,
            ciLower: ciLower, ciUpper: ciUpper, wilcoxonW: nil,
            wilcoxonP: wilcoxonP, adjustedP: adjustedP, correction: correction,
            modality: nil, unit: unit)
    }

    // MARK: - A sentence per effect

    /// The table's rows for one outcome, as `count` conditions that each
    /// carry an adjusted p — a correction family of that size.
    private func family(
        _ count: Int, metric: String = "fearMarkerDensity"
    ) -> [RunResults.EffectSizeRow] {
        (0..<count).map {
            row(condition: "arm-\($0)", metric: metric, adjustedP: 0.02)
        }
    }

    @Test func survivesCorrectionSentence() {
        let sentence = EffectNarrative.sentence(
            for: row(adjustedP: 0.012, correction: "BH"), familySize: 3)
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
            intervention: "fear L12 α0.8", familySize: 3)
        #expect(sentence.hasPrefix(
            "Steering 'fear' at layer 12 (strength 0.8) shifted"))
    }

    @Test func matchedNormControlSubject() {
        let sentence = EffectNarrative.sentence(
            for: row(condition: "fear-random-control"),
            intervention: "matched-norm random control", familySize: 0)
        #expect(sentence.hasPrefix(
            "The random-direction control 'fear-random-control'"))
    }

    @Test func ciCrossingZeroReadsConsistentWithNoEffect() {
        let sentence = EffectNarrative.sentence(
            for: row(meanDiff: 0.05, ciLower: -0.02, ciUpper: 0.12),
            familySize: 0)
        #expect(sentence.contains(
            "the interval crosses zero, so this is consistent with no effect"))
        #expect(!sentence.contains("survives"))
    }

    @Test func ciCrossingZeroWithFailedCorrectionSaysBoth() {
        let sentence = EffectNarrative.sentence(
            for: row(
                meanDiff: 0.05, ciLower: -0.02, ciUpper: 0.12, adjustedP: 0.4),
            familySize: 2)
        #expect(sentence.contains("consistent with no effect"))
        #expect(sentence.contains(
            "and does not survive multiple-comparison correction"))
    }

    @Test func ciCrossingZeroButCorrectedSignificantIsAnHonestConflict() {
        let sentence = EffectNarrative.sentence(
            for: row(
                meanDiff: 0.05, ciLower: -0.01, ciUpper: 0.12, adjustedP: 0.03),
            familySize: 2)
        #expect(sentence.contains("consistent with no effect"))
        #expect(sentence.contains(
            "(though the corrected p = 0.03 is below 0.05)"))
    }

    @Test func notSignificantAfterCorrectionIsSuggestiveOnly() {
        let sentence = EffectNarrative.sentence(
            for: row(adjustedP: 0.08, correction: "Holm"), familySize: 2)
        #expect(sentence.contains(
            "does not survive multiple-comparison correction "
                + "(corrected p = 0.08, Holm) — treat as suggestive only"))
    }

    @Test func uncorrectedSignificantCarriesTheCaveat() {
        let sentence = EffectNarrative.sentence(
            for: row(wilcoxonP: 0.01), familySize: 0)
        #expect(sentence.contains(
            "uncorrected p = 0.01 — no multiple-comparison correction was applied"))
    }

    @Test func uncorrectedNonSignificant() {
        // CI excludes zero but the only test says not significant — both
        // facts render, neither is hidden.
        let sentence = EffectNarrative.sentence(
            for: row(wilcoxonP: 0.2), familySize: 0)
        #expect(sentence.contains("(95% CI 0.12 to 0.48)"))
        #expect(sentence.contains("uncorrected p = 0.2 (not significant)"))
    }

    @Test func missingCISaysSo() {
        let sentence = EffectNarrative.sentence(
            for: row(ciLower: .nan, ciUpper: .nan, wilcoxonP: 0.03),
            familySize: 0)
        #expect(sentence.contains("(no confidence interval available)"))
        #expect(sentence.contains("uncorrected p = 0.03"))
    }

    @Test func missingCIWithCorrectionRendersCorrectedVerdict() {
        let significant = EffectNarrative.sentence(
            for: row(ciLower: .nan, ciUpper: .nan, adjustedP: 0.02),
            familySize: 2)
        #expect(significant.contains(
            "corrected p = 0.02 — significant after multiple-comparison correction"))
        let notSignificant = EffectNarrative.sentence(
            for: row(ciLower: .nan, ciUpper: .nan, adjustedP: 0.3),
            familySize: 2)
        #expect(notSignificant.contains(
            "corrected p = 0.3 — not significant after correction"))
    }

    @Test func noTestStatisticAtAll() {
        let sentence = EffectNarrative.sentence(
            for: row(ciLower: .nan, ciUpper: .nan), familySize: 0)
        #expect(sentence.hasSuffix("— no test statistic available."))
    }

    @Test func tinyPValuesAvoidScientificNotation() {
        let sentence = EffectNarrative.sentence(
            for: row(adjustedP: 0.00001), familySize: 2)
        #expect(sentence.contains("corrected p = < 0.0001"))
    }

    @Test func zeroNOmitsThePairedItemsClause() {
        let sentence = EffectNarrative.sentence(for: row(n: 0), familySize: 0)
        #expect(!sentence.contains("paired items"))
        // An unreported count is not "too few pairs": the interval stays.
        #expect(sentence.contains("(95% CI 0.12 to 0.48)"))
    }

    /// A study of one item sampled many times pairs ONE item, and the
    /// sentence says so in the singular.
    @Test func oneItemIsSingular() {
        let sentence = EffectNarrative.sentence(for: row(n: 1), familySize: 0)
        #expect(sentence.contains("by +0.31 across 1 paired item ("))
        #expect(!sentence.contains("paired items"))
    }

    // MARK: - One comparison is not a correction

    /// With one treatment arm the correction family has one member, so the
    /// adjusted p equals the raw one and nothing was corrected. The sentence
    /// says what was tested and never that the effect "survives".
    @Test func aFamilyOfOneSaysWhatWasTestedNotThatItSurvives() {
        let sentence = EffectNarrative.sentence(
            for: row(wilcoxonP: 0.012, adjustedP: 0.012, correction: "bh"),
            familySize: 1)
        #expect(
            sentence
                == "Condition 'fear-steered' shifted 'fear' marker density "
                + "(fearMarkerDensity) by +0.31 across 24 paired items "
                + "(95% CI 0.12 to 0.48) — p = 0.012 from the one Wilcoxon "
                + "signed-rank test of this outcome (a single comparison, so "
                + "no correction applies).")
        #expect(!sentence.contains("survive"))
        #expect(!sentence.contains("corrected p"))
    }

    @Test func aFamilyOfOneThatIsNotSignificantIsSuggestiveOnly() {
        let sentence = EffectNarrative.sentence(
            for: row(wilcoxonP: 0.2, adjustedP: 0.2, correction: "bh"),
            familySize: 1)
        #expect(sentence.hasSuffix(
            "— p = 0.2 from the one Wilcoxon signed-rank test of this "
                + "outcome (a single comparison, so no correction applies) — "
                + "not significant; treat as suggestive only."))
        #expect(!sentence.contains("survive"))
    }

    @Test func aFamilyOfOneWithAnIntervalCrossingZero() {
        let quiet = EffectNarrative.sentence(
            for: row(
                meanDiff: 0.05, ciLower: -0.02, ciUpper: 0.12, adjustedP: 0.4),
            familySize: 1)
        #expect(quiet.contains("consistent with no effect (p = 0.4 from the one"))
        #expect(!quiet.contains("survive"))
        let conflict = EffectNarrative.sentence(
            for: row(
                meanDiff: 0.05, ciLower: -0.01, ciUpper: 0.12, adjustedP: 0.03),
            familySize: 1)
        #expect(conflict.contains(
            "(though p = 0.03 from the one Wilcoxon signed-rank test of this "
                + "outcome (a single comparison, so no correction applies) is "
                + "below 0.05)"))
    }

    @Test func aFamilyOfOneWithoutAnInterval() {
        let sentence = EffectNarrative.sentence(
            for: row(ciLower: .nan, ciUpper: .nan, adjustedP: 0.3),
            familySize: 1)
        #expect(sentence.contains("(no confidence interval available)"))
        #expect(sentence.hasSuffix("no correction applies) — not significant."))
        #expect(!sentence.contains("corrected p"))
    }

    /// The family is counted from the table: the rows of the same outcome
    /// that carry an adjusted p. Another outcome's rows, and a row whose
    /// test was undefined, are not members.
    @Test func theCorrectionFamilyIsCountedFromTheTable() {
        let single = row(adjustedP: 0.012, correction: "bh")
        let table =
            [single]
            + family(3, metric: "wordCount")
            + [row(condition: "untested", adjustedP: nil)]
        #expect(EffectNarrative.correctionFamilySize(of: single, in: table) == 1)
        #expect(
            EffectNarrative.correctionFamilySize(of: table[1], in: table) == 3)
        #expect(
            !EffectNarrative.sentence(for: single, in: table).contains("survive"))
        #expect(
            EffectNarrative.sentence(for: table[1], in: table)
                .contains("survives multiple-comparison correction"))
    }

    // MARK: - Too few pairs for an interval

    /// One or two paired items print no interval and no test: the sentence
    /// says there are too few pairs, whatever the row's columns hold.
    @Test(arguments: [1, 2])
    func fewerThanThreePairsPrintNoInterval(n: Int) {
        let sentence = EffectNarrative.sentence(
            for: row(n: n, wilcoxonP: 0.5, adjustedP: 0.5, correction: "bh"),
            familySize: 3)
        #expect(sentence.contains(
            "(too few pairs for a confidence interval; at least 3 are needed)"))
        #expect(sentence.hasSuffix(
            "— with so few pairs this describes these items only, and is "
                + "not a test."))
        #expect(!sentence.contains("95% CI"))
        #expect(!sentence.contains("p = "))
        #expect(!sentence.contains("survive"))
        #expect(EffectNarrative.hasTooFewPairs(row(n: n)))
        #expect(!EffectNarrative.hasReportableInterval(row(n: n)))
    }

    /// A row that paired responses counts its ITEMS for the minimum: eight
    /// responses from two items are too few, and the sentence says items.
    @Test func pairedResponsesCountTheirItemsForTheMinimum() {
        let responses = row(
            n: 8, adjustedP: 0.01, correction: "bh",
            unit: .init(unit: "response", source: .inferredFromRecords, pairedItems: 2))
        let sentence = EffectNarrative.sentence(for: responses, familySize: 2)
        #expect(sentence.contains(
            "across 8 paired responses from 2 items (too few items for a "
                + "confidence interval; at least 3 are needed) — with so few "
                + "items this describes these items only, and is not a test."))
        #expect(EffectNarrative.hasTooFewPairs(responses))
        #expect(EffectNarrative.tooFewNoun(responses) == "items")
        // Three items carry the stored interval, with what it is not.
        let three = row(
            n: 6, unit: .init(unit: "response", source: .inferredFromRecords, pairedItems: 3))
        #expect(!EffectNarrative.hasTooFewPairs(three))
        #expect(EffectNarrative.sentence(for: three, familySize: 0).hasSuffix(
            " " + EffectNarrative.responseCaveat + "."))
    }

    /// A transcript-level row counts transcripts, and one nothing settles
    /// counts plain pairs.
    @Test func otherUnitsCountInTheirOwnNouns() {
        #expect(EffectNarrative.countPhrase(
            row(n: 4, unit: .init(unit: "transcript", source: .recorded))) == "4 paired transcripts")
        #expect(EffectNarrative.countPhrase(row(n: 1, unit: .unresolved)) == "1 pair")
        #expect(EffectNarrative.countPhrase(row(n: 0)) == nil)
        #expect(EffectNarrative.tooFewCaption([row(n: 2)]) == "A row with fewer than 3 "
            + "paired items has no whisker: that is too few pairs for an interval")
        #expect(EffectNarrative.tooFewCaption([row(n: 3)]) == nil)
    }

    @Test func threePairsCarryAnInterval() {
        let sentence = EffectNarrative.sentence(
            for: row(n: 3, adjustedP: 0.04), familySize: 2)
        #expect(sentence.contains("across 3 paired items (95% CI 0.12 to 0.48)"))
        #expect(!EffectNarrative.hasTooFewPairs(row(n: 3)))
        #expect(EffectNarrative.hasReportableInterval(row(n: 3)))
        // A row that did not report its count is left alone.
        #expect(!EffectNarrative.hasTooFewPairs(row(n: 0)))
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
                meanDiff: 0.5, ciLower: 0.3, ciUpper: 0.7, wilcoxonP: 0.5),
            familySize: 0)
        #expect(sentence.contains(
            "shifted scale position (1–K) (ordinalPosition) by +0.5"))
    }

    /// The phrases have one home, the shared headline-outcome mapping, so
    /// the outcomes the Python engine names read in plain words here too.
    @Test func phrasesComeFromTheSharedMapping() {
        #expect(EffectNarrative.metricPhrase("choiceRate")
            == "the target-choice rate (choiceRate)")
        #expect(EffectNarrative.metricPhrase("parsedValueMean")
            == "mean parsed numeric value (parsedValueMean)")
        #expect(EffectNarrative.metricPhrase("readerScore:warmth")
            == "reader score for 'warmth' (readerScore:warmth)")
        #expect(EffectNarrative.metricPhrase("judged")
            == "the judged outcome (judged)")
        // A bare pattern names nothing.
        #expect(EffectNarrative.metricPhrase("rs_") == "'rs_'")
        #expect(EffectNarrative.metricPhrase("MarkerDensity") == "'MarkerDensity'")
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
