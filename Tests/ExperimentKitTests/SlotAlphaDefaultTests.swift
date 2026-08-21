import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The Playground's α default, as a table.
///
/// The bug (researcher, 2026-08-18): "every time I select a vector it defaults
/// to alpha = 2, even if it's in norm units". α = 2 in norm units is two whole
/// residual-stream norms — off the end of the control's own −1…1 range — and
/// nothing about the artifact's measured denominator or its promoted cell was
/// consulted. The decision is engine-pure precisely so the table below is the
/// specification rather than a description of a view.
@Suite struct SlotAlphaDefaultTests {

    private func facts(
        norms: Float?,
        convention: String? = nil,
        recommended: Double? = nil,
        recommendedNormUnits: Bool? = nil,
        agent: String? = nil,
        promotedBy: String? = nil,
        layer: Int = 18
    ) -> SlotAlphaDefault.ArtifactFacts {
        SlotAlphaDefault.ArtifactFacts(
            artifactID: "runs/20260820T000000Z-concept-fear/fear",
            layer: layer,
            residualNormAtLayer: norms,
            residualNormConvention: convention,
            recommendedAlpha: recommended,
            recommendedAlphaInNormUnits: recommendedNormUnits,
            recommendedFromAgent: agent,
            recommendedPromotedBy: promotedBy)
    }

    // MARK: Row 1 — norms + a promoted cell

    @Test func normsPlusPromotedCellAdoptsTheSelectedAlpha() {
        let decision = SlotAlphaDefault.decide(
            facts(
                norms: 42, convention: ResidualNormConvention.current,
                recommended: 0.35, recommendedNormUnits: true,
                agent: "fear-a1", promotedBy: "criterion"))

        #expect(decision.alpha == 0.35)
        #expect(decision.units == .normUnits)
        #expect(decision.alphaLabel == "α in residual-norm units at L18")
        #expect(decision.rationale.contains("sweep-selected cell"))
        #expect(decision.rationale.contains("fear-a1"))
        #expect(decision.conventionNote == "wholeCorpusMean-v1")
        // Nothing to repair — the artifact already has a denominator.
        #expect(decision.backfillHint == nil)
    }

    /// A manual override is still evidence, and still says which kind it was.
    @Test func manualOverrideIsNamedAsSuch() {
        let decision = SlotAlphaDefault.decide(
            facts(
                norms: 42, recommended: 0.6, recommendedNormUnits: true,
                agent: "fear-a2", promotedBy: "manualOverride"))
        #expect(decision.alpha == 0.6)
        #expect(decision.rationale.contains("manually overridden cell"))
    }

    /// A hand-created variant carries no certificate; the number is still the
    /// best available, and the rationale does not claim it was selected.
    @Test func handCreatedVariantIsNamedAsSuch() {
        let decision = SlotAlphaDefault.decide(
            facts(
                norms: 42, recommended: 0.5, recommendedNormUnits: true,
                agent: "fear-hand", promotedBy: nil))
        #expect(decision.alpha == 0.5)
        #expect(decision.rationale.contains("hand-created variant"))
    }

    // MARK: Row 2 — norms, no recommendation

    @Test func normsWithoutARecommendationDefaultsToOneNormUnitNotTwo() {
        let decision = SlotAlphaDefault.decide(
            facts(norms: 42, convention: ResidualNormConvention.current))

        #expect(decision.alpha == 0.1)
        #expect(decision.alpha != 2)  // the reported bug, pinned
        #expect(decision.units == .normUnits)
        #expect(decision.alphaLabel == "α in residual-norm units at L18")
        #expect(decision.conventionNote == "wholeCorpusMean-v1")
        #expect(decision.backfillHint == nil)
    }

    /// A denominator measured before the convention stamp existed is usable —
    /// it is simply named honestly, never retro-labelled with today's rule.
    @Test func legacyDenominatorIsUsableAndLabelledLegacy() {
        let decision = SlotAlphaDefault.decide(facts(norms: 42, convention: nil))
        #expect(decision.units == .normUnits)
        #expect(decision.alpha == 0.1)
        #expect(decision.conventionNote == "legacy (pre-stamp)")
    }

    // MARK: Row 3 — no norms

    @Test func noNormsGivesLabelledRawModeAndTheRepair() {
        let decision = SlotAlphaDefault.decide(facts(norms: nil))

        #expect(decision.units == .raw)
        #expect(decision.alpha == SlotAlphaDefault.rawUnitsDefault)
        #expect(decision.alphaLabel.contains("RAW units at L18"))
        #expect(decision.alphaLabel.contains("no residual-norm denominator"))
        // There is no denominator, so there is no convention to describe.
        #expect(decision.conventionNote == nil)
        // The one-click repair, naming this artifact.
        let hint = try? #require(decision.backfillHint)
        #expect(hint?.contains("vectors backfill-norms") == true)
        #expect(
            hint?.contains("runs/20260820T000000Z-concept-fear/fear") == true)
    }

    /// A zero denominator is as unusable as a missing one — treating it as
    /// present would divide by it.
    @Test func nonPositiveDenominatorCountsAsMissing() {
        #expect(SlotAlphaDefault.decide(facts(norms: 0)).units == .raw)
        #expect(SlotAlphaDefault.decide(facts(norms: -1)).units == .raw)
        #expect(SlotAlphaDefault.decide(facts(norms: .nan)).units == .raw)
    }

    // MARK: Row 4 — the unit-mismatch cases (never a silent reinterpretation)

    /// A NORM-UNIT recommendation cannot be honoured without a denominator,
    /// and its numeral is NOT reused as a raw α: at a residual norm of ~50,
    /// "0.35" raw and "0.35 norm units" differ by ~50×.
    @Test func normUnitRecommendationIsNotReusedAsARawAlpha() {
        let decision = SlotAlphaDefault.decide(
            facts(
                norms: nil, recommended: 0.35, recommendedNormUnits: true,
                agent: "fear-a1", promotedBy: "criterion"))

        #expect(decision.units == .raw)
        #expect(decision.alpha == SlotAlphaDefault.rawUnitsDefault)
        #expect(decision.alpha != 0.35)
        #expect(decision.rationale.contains("NORM-UNIT"))
        #expect(decision.rationale.contains("not reused as a raw α"))
        #expect(decision.backfillHint != nil)
    }

    /// A RAW recommendation and raw mode agree, so the number is adopted.
    @Test func rawRecommendationIsAdoptedWhenThereIsNoDenominator() {
        let decision = SlotAlphaDefault.decide(
            facts(
                norms: nil, recommended: 6, recommendedNormUnits: false,
                agent: "legacy-raw", promotedBy: "criterion"))
        #expect(decision.units == .raw)
        #expect(decision.alpha == 6)
        #expect(decision.backfillHint != nil)
    }

    /// A RAW recommendation stays raw even where norm units are expressible:
    /// the stored number means a literal α·v coefficient and is not a
    /// fraction of anything (stored artifacts are never reinterpreted).
    @Test func rawRecommendationStaysRawEvenWithADenominator() {
        let decision = SlotAlphaDefault.decide(
            facts(
                norms: 42, recommended: 6, recommendedNormUnits: false,
                agent: "legacy-raw", promotedBy: "criterion"))
        #expect(decision.units == .raw)
        #expect(decision.alpha == 6)
        #expect(decision.conventionNote == nil)
    }

    // MARK: The label always names the layer

    @Test func labelNamesTheInjectionLayer() {
        #expect(
            SlotAlphaDefault.decide(facts(norms: 42, layer: 3)).alphaLabel
                == "α in residual-norm units at L3")
        #expect(
            SlotAlphaDefault.decide(facts(norms: nil, layer: 3)).alphaLabel
                .contains("at L3"))
    }

    // MARK: Recommendation lookup

    private func variant(
        name: String, vectorID: String, alpha: Double, normUnits: Bool = true,
        promotedBy: String? = nil, mode: InterventionPlan.Mode? = nil
    ) -> ModelVariantArtifact {
        ModelVariantArtifact(
            name: name, baseModelID: "m",
            injections: [
                .init(
                    concept: "fear", vectorArtifactID: vectorID, layer: 18,
                    alpha: alpha, mode: mode)
            ],
            alphaInNormUnits: normUnits,
            promptMode: "chatAssistant", qwenThinkingEnabled: false,
            temperature: 0, systemPrompt: "",
            promotion: promotedBy.map {
                ModelVariantArtifact.Promotion(
                    experiment: "e", experimentHash: "h",
                    promotedAt: "2026-08-20T00:00:00Z", promotedBy: $0,
                    substrate: "swift-mlx", appVersion: "test")
            })
    }

    @Test func lookupFindsTheAgentThatSteersWithThisArtifact() {
        let found = SlotAlphaDefault.recommendation(
            forVectorArtifactID: "runs/x/fear",
            among: [
                variant(name: "other", vectorID: "runs/x/calm", alpha: 0.9),
                variant(
                    name: "fear-a1", vectorID: "runs/x/fear", alpha: 0.35,
                    promotedBy: "criterion"),
            ])
        #expect(found?.alpha == 0.35)
        #expect(found?.agent == "fear-a1")
        #expect(found?.promotedBy == "criterion")
        #expect(found?.normUnits == true)
    }

    /// Sweep-selected beats manual override beats hand-created — so the
    /// default never quietly prefers the weaker provenance.
    @Test func lookupPrefersTheStrongestProvenance() {
        let variants = [
            variant(name: "a-hand", vectorID: "runs/x/fear", alpha: 0.9),
            variant(
                name: "b-override", vectorID: "runs/x/fear", alpha: 0.7,
                promotedBy: "manualOverride"),
            variant(
                name: "c-criterion", vectorID: "runs/x/fear", alpha: 0.35,
                promotedBy: "criterion"),
        ]
        #expect(
            SlotAlphaDefault.recommendation(
                forVectorArtifactID: "runs/x/fear", among: variants)?.agent
                == "c-criterion")
        // Removing the strongest falls back one rank, never to the hand one.
        #expect(
            SlotAlphaDefault.recommendation(
                forVectorArtifactID: "runs/x/fear",
                among: Array(variants.prefix(2)))?.agent == "b-override")
    }

    /// λ is not α: an ablation injection never consults a residual-norm
    /// denominator, so it must not supply a steering default either.
    @Test func lookupIgnoresAblationInjections() {
        #expect(
            SlotAlphaDefault.recommendation(
                forVectorArtifactID: "runs/x/fear",
                among: [
                    variant(
                        name: "ablator", vectorID: "runs/x/fear", alpha: 1,
                        promotedBy: "criterion", mode: .ablate)
                ]) == nil)
    }

    @Test func lookupIsEmptyWhenNoAgentUsesTheArtifact() {
        #expect(
            SlotAlphaDefault.recommendation(
                forVectorArtifactID: "runs/x/fear",
                among: [variant(name: "other", vectorID: "runs/x/calm", alpha: 1)])
                == nil)
    }
}
