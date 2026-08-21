import Foundation
import Testing

@testable import ExperimentKit

/// The silent-failure firewall for chat steering rows: a row that LOOKS
/// active but would inject nothing (or less than it claims) must classify to
/// a self-naming refusal — the same pure rule feeds the send-time throw in
/// `currentInjections()` and the standing per-row caption, so behavior and
/// caption cannot drift. (Bug class: "I injected arousal at α=1 on top of
/// the agent and nothing changed" with no visible reason anywhere.)
@Suite struct SlotInjectionHonestyTests {

    private func facts(
        concept: String? = "arousal",
        vectorModelID: String? = "Qwen/Qwen3-4B",
        loadedModelID: String? = "Qwen/Qwen3-4B",
        normUnits: Bool = true,
        residualNorms: [Float]? = [10, 12],
        vectorNorms: [Float] = [1, 1],
        projection: Bool = false,
        isAblation: Bool = false
    ) -> ChatService.SlotInjectionFacts {
        ChatService.SlotInjectionFacts(
            concept: concept,
            vectorModelID: vectorModelID,
            loadedModelID: loadedModelID,
            alphaInNormUnits: normUnits,
            residualNormsAtBandLayers: residualNorms,
            vectorNormsAtBandLayers: vectorNorms,
            neutralProjectionActive: projection,
            isAblation: isAblation)
    }

    // MARK: - Local classification

    @Test func healthyRowHasNoRefusal() {
        #expect(ChatService.slotInjectionRefusal(facts()) == nil)
    }

    @Test func rawModeNeedsNoDenominator() {
        // Raw α is the literal coefficient — a norms-less legacy vector is
        // fine there; only norm units require the denominator.
        #expect(
            ChatService.slotInjectionRefusal(
                facts(normUnits: false, residualNorms: nil)) == nil)
    }

    @Test func unresolvedVectorRefusesAndNamesTheRemedy() {
        let reason = ChatService.slotInjectionRefusal(facts(concept: nil))
        #expect(reason?.contains("NOT injected") == true)
        #expect(reason?.contains("Refresh artifacts") == true)
    }

    @Test func wrongModelRefusesWithBothModelIDs() {
        let reason = ChatService.slotInjectionRefusal(
            facts(vectorModelID: "google/gemma-3-4b", loadedModelID: "Qwen/Qwen3-4B"))
        #expect(reason?.contains("google/gemma-3-4b") == true)
        #expect(reason?.contains("not the loaded model") == true)
    }

    @Test func missingNormsUnderNormUnitsPointsAtBackfill() {
        // THE reported case: a 2026-07-06 mean-diff artifact without
        // residualNormPerLayer, α typed in norm units. The refusal must name
        // the vector, say it is NOT injected, and point at the existing action
        // (not a dead end).
        //
        // Asserts the shared BREADCRUMB rather than the word "backfill":
        // reported again 2026-07-29, the advice said "in Concepts" — a renamed
        // section — and used jargon the destination's button does not ("Measure
        // norms"). A refusal has to speak the UI's words, and the constant is
        // what keeps the two from drifting.
        let reason = ChatService.slotInjectionRefusal(facts(residualNorms: nil))
        #expect(reason?.contains("NOT injected") == true)
        #expect(reason?.contains("arousal") == true)
        #expect(reason?.contains(SteeringGuidance.normBackfillLocation) == true)
        #expect(reason?.contains("raw") == true)
    }

    @Test func degenerateDenominatorInBandRefuses() {
        let reason = ChatService.slotInjectionRefusal(
            facts(residualNorms: [10, 0]))
        #expect(reason?.contains("denominator") == true)
        #expect(reason?.contains(SteeringGuidance.normBackfillLocation) == true)
    }

    @Test func fullyProjectedOutVectorNamesTheProjection() {
        let reason = ChatService.slotInjectionRefusal(
            facts(vectorNorms: [0, 0], projection: true))
        #expect(reason?.contains("neutral-direction projection") == true)
        #expect(reason?.contains("NOT injected") == true)
    }

    @Test func zeroNormWithoutProjectionCallsTheArtifactDegenerate() {
        let reason = ChatService.slotInjectionRefusal(
            facts(vectorNorms: [0], projection: false))
        #expect(reason?.contains("zero norm") == true)
    }

    @Test func partialZeroBandStillInjects() {
        // One dead band-edge layer is a skip, not a refusal — the remaining
        // layers still inject what the row promises.
        #expect(
            ChatService.slotInjectionRefusal(facts(vectorNorms: [0, 1])) == nil)
    }

    @Test func unresolvedOutranksEverythingElse() {
        // With no artifact there is nothing truthful to say about norms.
        let reason = ChatService.slotInjectionRefusal(
            facts(concept: nil, residualNorms: nil, vectorNorms: []))
        #expect(reason?.contains("does not resolve") == true)
    }

    // MARK: - Server classification (catalog facts)

    @Test func serverUnresolvedRefuses() {
        let reason = ChatService.serverSlotRefusal(
            concept: nil, alphaInNormUnits: true,
            residualNormPerLayer: nil, hasResidualNorms: nil, layer: 3)
        #expect(reason?.contains("NOT sent") == true)
        #expect(reason?.contains("Refresh artifacts") == true)
    }

    @Test func serverMissingNormsUnderNormUnitsPredictsTheRefusal() {
        let reason = ChatService.serverSlotRefusal(
            concept: "arousal", alphaInNormUnits: true,
            residualNormPerLayer: nil, hasResidualNorms: false, layer: 3)
        #expect(reason?.contains("server will refuse") == true)
        #expect(reason?.contains(SteeringGuidance.normBackfillLocation) == true)
    }

    @Test func serverDegenerateDenominatorAtLayerRefuses() {
        let reason = ChatService.serverSlotRefusal(
            concept: "arousal", alphaInNormUnits: true,
            residualNormPerLayer: [10, 0, 12], hasResidualNorms: true, layer: 1)
        #expect(reason?.contains("L1") == true)
    }

    @Test func serverHealthyAndRawRowsPass() {
        #expect(
            ChatService.serverSlotRefusal(
                concept: "arousal", alphaInNormUnits: true,
                residualNormPerLayer: [10, 11], hasResidualNorms: true, layer: 1)
                == nil)
        #expect(
            ChatService.serverSlotRefusal(
                concept: "arousal", alphaInNormUnits: false,
                residualNormPerLayer: nil, hasResidualNorms: false, layer: 0)
                == nil)
    }

    @Test func serverOlderCatalogStaysQuietRatherThanInventing() {
        // No per-layer table and no flag: the catalog cannot say — no
        // invented refusal (the server itself will still refuse honestly).
        #expect(
            ChatService.serverSlotRefusal(
                concept: "arousal", alphaInNormUnits: true,
                residualNormPerLayer: nil, hasResidualNorms: nil, layer: 0)
                == nil)
    }

    // MARK: - Ablation exemption (λ never converts through the denominator)

    @Test func ablationWithoutNormsUnderNormUnitsInjects() {
        // THE 2026-08-06 field report: a J-lens/derived direction with no
        // residual norms, session in norm units, mode ablate. λ is a plain
        // fraction of what is present — neither engine converts it through
        // the denominator, so the refusal rule must not refuse a send the
        // engine would run.
        #expect(
            ChatService.slotInjectionRefusal(
                facts(residualNorms: nil, isAblation: true)) == nil)
    }

    @Test func ablationWithDegenerateDenominatorInjects() {
        #expect(
            ChatService.slotInjectionRefusal(
                facts(residualNorms: [10, 0], isAblation: true)) == nil)
    }

    @Test func ablationExemptionIsNormUnitsOnly() {
        // A zero-norm direction cannot ablate anything: the degenerate-vector
        // refusal survives the exemption.
        let reason = ChatService.slotInjectionRefusal(
            facts(vectorNorms: [0], isAblation: true))
        #expect(reason?.contains("zero norm") == true)
        // And an unresolved row still refuses regardless of mode.
        #expect(
            ChatService.slotInjectionRefusal(
                facts(concept: nil, isAblation: true)) != nil)
    }

    @Test func serverAblationWithoutNormsIsNotPredictedToRefuse() {
        // The server's variant_injections bypasses the λ conversion, so the
        // standing caption must not claim "server will refuse this send".
        #expect(
            ChatService.serverSlotRefusal(
                concept: "jlens-token", alphaInNormUnits: true,
                residualNormPerLayer: nil, hasResidualNorms: false, layer: 3,
                isAblation: true) == nil)
    }

    // MARK: - Toggle availability gate counts only steering rows

    @Test func normDenominatorGateIgnoresAblationRows() {
        // The α-denomination toggle's availability rule must not count
        // ablate rows: an ablation of a norms-less vector must neither
        // gate nor lock the toggle (the stuck-toggle half of the 2026-08-06
        // field report).
        let ablate = ChatService.SteerSlot(
            vectorID: "runs/jlens/lens-token", mode: .ablate)
        let steer = ChatService.SteerSlot(vectorID: "runs/x/fear")
        var idle = ChatService.SteerSlot(vectorID: "runs/x/anger")
        idle.enabled = false
        let requiring = ChatService.slotsRequiringNormDenominator(
            [ablate, steer, idle, ChatService.SteerSlot()])
        #expect(requiring.map(\.id) == [steer.id])
    }

    // MARK: - Band rule (facts and injection building must agree)

    @Test func bandLayersClampAndCenter() {
        #expect(ChatService.bandLayers(center: 5, layerCount: 12, bandWidth: 1) == [5])
        #expect(
            ChatService.bandLayers(center: 5, layerCount: 12, bandWidth: 5)
                == [3, 4, 5, 6, 7])
        #expect(ChatService.bandLayers(center: 0, layerCount: 12, bandWidth: 5) == [0, 1, 2])
        #expect(
            ChatService.bandLayers(center: 11, layerCount: 12, bandWidth: 5)
                == [9, 10, 11])
        #expect(ChatService.bandLayers(center: 40, layerCount: 12, bandWidth: 1) == [11])
        #expect(ChatService.bandLayers(center: -3, layerCount: 12, bandWidth: 1) == [0])
    }

    // MARK: - Stored-variant shadow (server chat composition)

    @Test func unseededStoredVariantWithRowsWarns() {
        let warning = InlineVariantComposer.unseededStoredVariantWarning(
            storedVariantSelected: true, seeded: false, configuredSlotCount: 2)
        #expect(warning?.contains("2 steering rows") == true)
        #expect(warning?.contains("NOT ride") == true)
    }

    @Test func seededOrRowlessStoredVariantDoesNotWarn() {
        #expect(
            InlineVariantComposer.unseededStoredVariantWarning(
                storedVariantSelected: true, seeded: true, configuredSlotCount: 2)
                == nil)
        #expect(
            InlineVariantComposer.unseededStoredVariantWarning(
                storedVariantSelected: true, seeded: false, configuredSlotCount: 0)
                == nil)
        #expect(
            InlineVariantComposer.unseededStoredVariantWarning(
                storedVariantSelected: false, seeded: false, configuredSlotCount: 2)
                == nil)
    }
}
