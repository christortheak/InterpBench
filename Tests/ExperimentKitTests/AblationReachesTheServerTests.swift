import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// The Playground's ablation must survive the trip to the cluster.
///
/// It did not (2026-07-27, reported from a live 27B session): the picker set
/// the mode, `InlineVariantComposer.SlotInput` had no field for it, and the
/// server received an ordinary steering injection with `alpha = λ`. Ablating
/// at λ behaved "almost exactly like steering at α" because it WAS steering
/// at α.
///
/// The failure was silent because the wire spells the default by omission —
/// chosen so steering artifacts keep their bytes and hashes, which was right,
/// and which makes a DROPPED mode indistinguishable from a steering one. Unit
/// tests at each end passed throughout; nothing tested the seam. These tests
/// are that seam.
struct AblationReachesTheServerTests {

    private func state(
        mode: InterventionPlan.Mode, alpha: Double = 1
    ) -> InlineVariantComposer.ControlState {
        let resolution = InlineVariantComposer.resolveSlots(
            steeringEnabled: true,
            slots: [
                .init(
                    vectorID: "runs/x/anger", layer: 20, alpha: alpha,
                    enabled: true, mode: mode)
            ],
            concept: { _ in "anger" })
        return .init(
            baseModelID: "google/gemma-3-27b-it",
            slots: resolution.slots,
            adapters: [], bandWidth: 1, alphaInNormUnits: true,
            neutralPCBasisPath: nil, neutralPCBasisLabel: nil,
            promptMode: "chatAssistant", qwenThinkingEnabled: false,
            temperature: 0, systemPrompt: "")
    }

    /// The composed spec — the object that is serialized and POSTed — must
    /// carry the mode.
    @Test func anAblationSlotComposesIntoAnAblationInjection() throws {
        let spec = InlineVariantComposer.compose(state(mode: .ablate))
        let injection = try #require(spec.injections.first)
        #expect(injection.effectiveMode == .ablate)
        #expect(spec.ablatesOnly)
    }

    /// And it must survive JSON — this is the actual wire.
    @Test func theModeSurvivesSerializationToTheServer() throws {
        let spec = InlineVariantComposer.compose(state(mode: .ablate))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let wire = String(decoding: try encoder.encode(spec), as: UTF8.self)
        #expect(
            wire.contains(#""mode":"ablate""#),
            "the ablation never reaches the cluster: \(wire)")

        let back = try JSONDecoder().decode(
            ModelVariantArtifact.self, from: Data(wire.utf8))
        #expect(back.injections.first?.effectiveMode == .ablate)
    }

    /// The exact confusion that was reported: at the same numeric value, the
    /// two modes must produce DIFFERENT specs. Before the fix these were
    /// byte-identical, which is why λ behaved like α.
    @Test func steeringAndAblationAtTheSameValueAreNotTheSameSpec() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let steering = try encoder.encode(
            InlineVariantComposer.compose(state(mode: .add, alpha: 1)))
        let ablation = try encoder.encode(
            InlineVariantComposer.compose(state(mode: .ablate, alpha: 1)))
        #expect(
            steering != ablation,
            "λ = 1 ablation and α = 1 steering serialize identically — the mode is being dropped somewhere between the picker and the wire")
    }

    /// A steering spec's bytes must NOT move: they are hashed into promotion
    /// keys and artifact hashes, so the absent-means-add convention has to
    /// hold on this path too.
    @Test func aSteeringSpecIsByteIdenticalToBeforeTheModeExisted() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let wire = String(
            decoding: try encoder.encode(
                InlineVariantComposer.compose(state(mode: .add))),
            as: UTF8.self)
        #expect(!wire.contains("mode"))
    }

    /// The slot resolver is where the field was missing; pin that it carries
    /// through rather than defaulting.
    @Test func theResolverPreservesTheModeItWasGiven() {
        let resolution = InlineVariantComposer.resolveSlots(
            steeringEnabled: true,
            slots: [
                .init(vectorID: "a", layer: 1, alpha: 1, enabled: true, mode: .ablate),
                .init(vectorID: "b", layer: 2, alpha: 2, enabled: true, mode: .add),
            ],
            concept: { $0 })
        #expect(resolution.slots.map(\.mode) == [.ablate, .add])
    }
}
