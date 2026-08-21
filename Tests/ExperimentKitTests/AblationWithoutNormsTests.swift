import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// Ablation-without-norms round trip on this engine (field report
/// 2026-08-06; server twin: `test_variant_ablation_without_norms_composes`).
/// λ is a plain fraction of what the residual stream contains — it never
/// converts through the residual-norm denominator — so an ablating injection
/// on a norms-less vector (the routine case for J-lens/reader-derived
/// directions) must compose under `alphaInNormUnits`, while a STEERING
/// injection on the same artifact keeps the actionable both-remedies refusal.
@Suite struct AblationWithoutNormsTests {

    /// A saved artifact whose sidecar records NO residual norms (the default
    /// sidecar init — exactly what imports and derived directions produce).
    private func writeNormlessArtifact(
        layers: Int = 3
    ) throws -> (dir: URL, id: String) {
        let vectors = ConceptVectors(
            perLayer: (0..<layers).map { [Float($0 + 1), 0, 0] })
        let sidecar = SteeringVectorSidecar(
            modelID: "test/model", concept: "lens-token",
            stimulusSetHash: "abc123", vectors: vectors)
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "ablate-nonorm-\(UUID().uuidString)")
        try SteeringVectorStore.save(
            vectors: vectors, sidecar: sidecar, to: dir, name: "lens")
        return (dir, dir.appending(component: "lens").path)
    }

    private func variant(
        injections: [ModelVariantArtifact.InjectionRef]
    ) -> ModelVariantArtifact {
        ModelVariantArtifact(
            name: "v", baseModelID: "test/model",
            injections: injections,
            bandWidth: 1, alphaInNormUnits: true,
            promptMode: "chatAssistant", qwenThinkingEnabled: false,
            temperature: 0, systemPrompt: "")
    }

    @Test func ablationWithoutNormsComposesUnderNormUnits() throws {
        let (dir, id) = try writeNormlessArtifact()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cells = try ExperimentTasks.injections(
            for: variant(injections: [
                .init(
                    concept: "lens-token", vectorArtifactID: id,
                    layer: 1, alpha: 0.75, mode: .ablate)
            ]))
        // Whole-network coverage, λ passed through untouched, mode preserved.
        #expect(cells.map(\.layer).sorted() == [0, 1, 2])
        #expect(cells.allSatisfy { $0.mode == .ablate })
        #expect(cells.allSatisfy { $0.alpha == 0.75 })
    }

    @Test func steeringWithoutNormsKeepsTheBothRemediesRefusal() throws {
        let (dir, id) = try writeNormlessArtifact()
        defer { try? FileManager.default.removeItem(at: dir) }
        do {
            _ = try ExperimentTasks.injections(
                for: variant(injections: [
                    .init(
                        concept: "lens-token", vectorArtifactID: id,
                        layer: 1, alpha: 0.75)
                ]))
            Issue.record("norm-unit steering on a norms-less vector must refuse")
        } catch let error as ExperimentError {
            // Actionable, never cryptic: BOTH remedies by name.
            #expect(error.reason.contains("backfill"))
            #expect(error.reason.contains("raw alpha"))
        }
    }
}
