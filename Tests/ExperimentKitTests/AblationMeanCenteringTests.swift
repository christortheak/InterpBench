import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// Neutral-mean centering of ablation directions + the mean-alignment
/// preflight.
///
/// Why this exists (2026-08-06 collapse study, 4B tier, both families):
/// extracted concept vectors routinely share a large component with the
/// residual stream's neutral-corpus mean — on real 27B artifacts, ~74% of
/// each vector's norm lies along one shared direction, and cross-concept
/// cosines average ~0.52 when distinct concepts should be near-orthogonal.
/// Ablating such a direction at λ=1 (any layer, every position) collapses
/// generation into single-token repetition; projecting the neutral mean out
/// of the direction first fully restores coherence. Qwen3 vectors with
/// |cos| ≤ 0.28 do not collapse — the collapse tracks mean-alignment, not
/// ablation per se, which is what calibrates the warn threshold.
///
/// The centering fixture numbers here are the CROSS-ENGINE contract — the
/// server's `test_ablation_mean_centering.py` asserts the identical values.
struct AblationMeanCenteringTests {

    // MARK: pure math (cross-engine fixture values)

    @Test func centeringRemovesExactlyTheMeanComponent() {
        // v = [3,4,0], m̂ = [0,1,0] → v − (v·m̂)m̂ = [3,0,0]. Same numbers in
        // the server suite.
        let centered = SteeringVectorMath.meanCentered(
            [3, 4, 0], against: [0, 2, 0])
        #expect(abs(centered[0] - 3) < 1e-6)
        #expect(abs(centered[1]) < 1e-6)
        #expect(abs(centered[2]) < 1e-6)
    }

    @Test func centeredResultIsOrthogonalToTheMean() {
        let mean: [Float] = [0.5, -1, 2]
        let centered = SteeringVectorMath.meanCentered([1, 2, 3], against: mean)
        #expect(abs(SteeringVectorMath.dot(centered, mean)) < 1e-5)
    }

    @Test func centeringIsIdempotent() {
        let mean: [Float] = [0.5, -1, 2]
        let once = SteeringVectorMath.meanCentered([1, 2, 3], against: mean)
        let twice = SteeringVectorMath.meanCentered(once, against: mean)
        for (a, b) in zip(once, twice) {
            #expect(abs(a - b) < 1e-6)
        }
    }

    @Test func centeringAgainstAZeroMeanIsIdentity() {
        let centered = SteeringVectorMath.meanCentered([1, 2, 3], against: [0, 0, 0])
        #expect(centered == [1, 2, 3])
    }

    @Test func meanAlignmentIsAbsCosineAndDegenerateSafe() {
        #expect(abs(SteeringVectorMath.meanAlignment([3, 4, 0], with: [0, 2, 0]) - 0.8) < 1e-6)
        #expect(abs(SteeringVectorMath.meanAlignment([3, -4, 0], with: [0, 2, 0]) - 0.8) < 1e-6)
        #expect(SteeringVectorMath.meanAlignment([1, 0, 0], with: [0, 0, 0]) == 0)
    }

    @Test func warnThresholdIsThePinnedCrossEngineConstant() {
        // Calibrated: Qwen3-0.6B coherent at ≤0.28; Gemma-3-4b collapses
        // from ~0.45 mean alignment. The server pins the same value.
        #expect(SteeringVectorMath.ablationMeanAlignmentWarnThreshold == 0.35)
    }

    // MARK: extraction-side mean computation

    @Test func neutralMeanPerLayerIsTheRowwiseMean() {
        let neutral = StimulusActivations(
            values: [
                [[1, 0], [10, 2]],  // text 0: layer0, layer1
                [[3, 4], [30, 6]],  // text 1
            ],
            residualNormPerLayer: [1, 1])
        let mean = ConceptExtractor.neutralMeanPerLayer(of: neutral, layerCount: 2)
        #expect(mean[0] == [2, 2])
        #expect(mean[1] == [20, 4])
    }

    // MARK: preflight report

    @Test func alignedUncenteredAblationReportsTheWorstLayer() throws {
        let warning = ExperimentTasks.reportAblationMeanAlignment(
            concept: "anger",
            vectors: [[1, 0, 0], [3, 4, 0]],  // layer 1 has |cos| 0.8
            firstLayer: 0,
            neutralMean: [[1, 0, 0], [0, 2, 0]],
            where: "test", remedy: "center it")
        let text = try #require(warning)
        #expect(text.contains("layer 0") || text.contains("layer 1"))
        #expect(text.contains("0.8") || text.contains("1.00"))
        #expect(text.contains("center it"))
    }

    @Test func orthogonalDirectionRaisesNoWarning() {
        let warning = ExperimentTasks.reportAblationMeanAlignment(
            concept: "anger",
            vectors: [[1, 0, 0]],
            firstLayer: 0,
            neutralMean: [[0, 1, 0]],
            where: "test", remedy: "center it")
        #expect(warning == nil)
    }

    @Test func missingMeanReportsCheckImpossibleNotSafe() throws {
        let warning = ExperimentTasks.reportAblationMeanAlignment(
            concept: "anger", vectors: [[1, 0, 0]], firstLayer: 0,
            neutralMean: nil, where: "test", remedy: "re-extract")
        let text = try #require(warning)
        #expect(text.contains("preflight impossible"))
    }

    // MARK: variant artifact schema (additive, hash-preserving)

    private func injection(
        mode: InterventionPlan.Mode? = nil, centering: String? = nil
    ) -> ModelVariantArtifact.InjectionRef {
        .init(
            concept: "anger", vectorArtifactID: "runs/x/anger", layer: 2,
            alpha: 1, mode: mode, centering: centering)
    }

    @Test func aSteeringInjectionEncodesExactlyAsBefore() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let text = String(decoding: try encoder.encode(injection()), as: UTF8.self)
        #expect(!text.contains("centering"))
        // Explicit "none" is likewise never written — artifact bytes are
        // hashed into promotion keys, so the default must stay byte-absent.
        let explicit = String(
            decoding: try encoder.encode(injection(centering: "none")),
            as: UTF8.self)
        #expect(explicit == text)
    }

    @Test func aCenteringDeclarationRoundTrips() throws {
        let data = try JSONEncoder().encode(
            injection(mode: .ablate, centering: "neutralMean"))
        #expect(String(decoding: data, as: UTF8.self).contains(#""centering":"neutralMean""#))
        let back = try JSONDecoder().decode(
            ModelVariantArtifact.InjectionRef.self, from: data)
        #expect(back.effectiveCentering == "neutralMean")
    }

    @Test func aLegacyInjectionDecodesAsUncentered() throws {
        let json = #"{"concept":"anger","vectorArtifactID":"runs/x/anger","layer":2,"alpha":1}"#
        let back = try JSONDecoder().decode(
            ModelVariantArtifact.InjectionRef.self, from: Data(json.utf8))
        #expect(back.centering == nil)
        #expect(back.effectiveCentering == "none")
    }

    // MARK: the centering declaration TRAVELS (inline-variant seam)

    /// Server-workspace Playground: the Mac never holds server vector bytes,
    /// so a centered ablation is centered by DECLARATION — the composed
    /// inline spec must carry the key, exactly as `mode` must travel
    /// (the 2026-07-27 dropped-mode bug, same seam).
    @Test func composedInlineSpecCarriesAblationCentering() throws {
        let resolution = InlineVariantComposer.resolveSlots(
            steeringEnabled: true,
            slots: [
                .init(vectorID: "runs/x/anger", layer: 2, alpha: 1,
                      enabled: true, mode: .ablate, centering: "neutralMean"),
                .init(vectorID: "runs/x/calm", layer: 3, alpha: 0.5,
                      enabled: true, mode: .add, centering: "neutralMean"),
            ],
            concept: { _ in "c" })
        let spec = InlineVariantComposer.compose(
            InlineVariantComposer.ControlState(
                baseModelID: "org/m", slots: resolution.slots,
                promptMode: "chatAssistant"))
        let text = String(
            decoding: try JSONEncoder().encode(spec), as: UTF8.self)
        #expect(text.contains(#""centering":"neutralMean""#))
        // A STEERING slot never carries centering, even if a stale value is
        // lying around — centering is an ablation-direction transform.
        #expect(spec.injections.count == 2)
        #expect(spec.injections[0].centering == "neutralMean")
        #expect(spec.injections[1].centering == nil)
    }

    // MARK: sidecar schema (additive)

    @Test func aLegacySidecarDecodesWithoutANeutralMeanStamp() throws {
        let json = """
            {"modelID":"org/m","concept":"anger","stimulusSetHash":"h",
             "layerCount":2,"hiddenSize":3,"normsPerLayer":[1,1],
             "extractionDate":"2026-08-06T00:00:00Z"}
            """
        let sidecar = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: Data(json.utf8))
        #expect(sidecar.neutralMeanSource == nil)
    }
}
