import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// The ablation mode is additive to the artifact format, and "additive" has to
/// mean bit-for-bit invisible — variant JSON is hashed into promotion keys and
/// artifact hashes, so a field that appeared on every steering agent would
/// silently re-identify every agent in the library.
struct AblationSchemaTests {

    private func encoded(_ ref: ModelVariantArtifact.InjectionRef) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return String(decoding: try encoder.encode(ref), as: UTF8.self)
    }

    /// An artifact written before ablation existed decodes, and reads as
    /// steering.
    @Test func aLegacyInjectionDecodesAsSteering() throws {
        let json = """
            {"concept":"fear","vectorArtifactID":"runs/x/fear","layer":31,"alpha":1.9}
            """
        let ref = try JSONDecoder().decode(
            ModelVariantArtifact.InjectionRef.self, from: Data(json.utf8))
        #expect(ref.mode == nil)
        #expect(ref.effectiveMode == .add)
        #expect(ref.alpha == 1.9)
    }

    /// And re-encoding it produces the SAME bytes — no `"mode":"add"` appears.
    @Test func steeringInjectionsEncodeExactlyAsBefore() throws {
        let ref = ModelVariantArtifact.InjectionRef(
            concept: "fear", vectorArtifactID: "runs/x/fear", layer: 31,
            alpha: 1.9)
        let text = try encoded(ref)
        #expect(!text.contains("mode"))
        #expect(
            text == #"{"alpha":1.9,"concept":"fear","layer":31,"vectorArtifactID":"runs\/x\/fear"}"#,
            "the steering encoding changed: every existing agent's artifact hash and promotion key would move — \(text)")
    }

    /// An explicit `.add` is likewise not written: the default has exactly one
    /// on-disk spelling, so two identical agents cannot hash differently
    /// because one was built through a picker and one was not.
    @Test func anExplicitAddIsNotWrittenEither() throws {
        let ref = ModelVariantArtifact.InjectionRef(
            concept: "fear", vectorArtifactID: "runs/x/fear", layer: 31,
            alpha: 1.9, mode: .add)
        #expect(!(try encoded(ref).contains("mode")))
    }

    @Test func ablationRoundTrips() throws {
        let ref = ModelVariantArtifact.InjectionRef(
            concept: "neuroticism", vectorArtifactID: "runs/y/neuroticism",
            layer: 31, alpha: 1, mode: .ablate)
        let text = try encoded(ref)
        #expect(text.contains(#""mode":"ablate""#))
        let back = try JSONDecoder().decode(
            ModelVariantArtifact.InjectionRef.self, from: Data(text.utf8))
        #expect(back.effectiveMode == .ablate)
        #expect(back == ref)
    }

    /// The wire spelling is shared with the server's `Mode` enum — the value
    /// travels in evidence bundles, so a rename on one side must break a test
    /// rather than an import.
    @Test func theModeVocabularyIsTheCrossEngineSpelling() {
        #expect(InterventionPlan.Mode.add.rawValue == "add")
        #expect(InterventionPlan.Mode.ablate.rawValue == "ablate")
        #expect(InterventionPlan.Mode.allCases.count == 2)
    }

    /// A resolved cell defaults to steering, so every code path that builds
    /// one without mentioning a mode keeps its behaviour.
    @Test func aResolvedCellDefaultsToSteering() {
        let cell = ExperimentTasks.CellInjection(
            layer: 3, vector: [1, 0, 0], alpha: 2)
        #expect(cell.mode == .add)
        #expect(cell.planEdit.mode == .add)
        #expect(cell.planEdit.strength == 2)
    }
}
