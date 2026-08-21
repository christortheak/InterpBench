import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// Ablation as a STUDY condition: what the manifest records, and which
/// control answers the question ablation actually raises.
struct AblationConditionTests {

    private func slot(
        _ concept: String, layer: Int = 0, alpha: Double = 1,
        mode: InterventionPlan.Mode? = nil
    ) -> ExperimentManifest.Condition.Slot {
        .init(concept: concept, layer: layer, alpha: alpha, mode: mode)
    }

    private func condition(
        _ name: String, _ slots: [ExperimentManifest.Condition.Slot],
        normUnits: Bool = true
    ) -> ExperimentManifest.Condition {
        .init(name: name, slots: slots, bandWidth: 1, alphaInNormUnits: normUnits)
    }

    // MARK: manifest bytes

    /// Manifest bytes ARE the content hash, so a mode key on every existing
    /// condition would re-identify every frozen study in the workspace.
    @Test func aSteeringSlotEncodesExactlyAsBefore() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let text = String(
            decoding: try encoder.encode(slot("fear", layer: 14, alpha: 0.5)),
            as: UTF8.self)
        #expect(text == #"{"alpha":0.5,"concept":"fear","layer":14}"#)
        // An explicit `.add` is likewise never written, so a condition built
        // through the picker hashes identically to one built before it existed.
        let explicit = String(
            decoding: try encoder.encode(
                slot("fear", layer: 14, alpha: 0.5, mode: .add)),
            as: UTF8.self)
        #expect(explicit == text)
    }

    @Test func anAblationSlotRoundTrips() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(slot("fear", alpha: 1, mode: .ablate))
        #expect(String(decoding: data, as: UTF8.self).contains(#""mode":"ablate""#))
        let back = try JSONDecoder().decode(
            ExperimentManifest.Condition.Slot.self, from: data)
        #expect(back.effectiveMode == .ablate)
    }

    @Test func aLegacySlotDecodesAsSteering() throws {
        let json = #"{"concept":"fear","layer":14,"alpha":0.5}"#
        let back = try JSONDecoder().decode(
            ExperimentManifest.Condition.Slot.self, from: Data(json.utf8))
        #expect(back.mode == nil)
        #expect(back.effectiveMode == .add)
    }

    // MARK: the right control, chosen by what the condition does

    /// Attaching a matched-norm control to an ablation would produce a cell
    /// that ablates the CONCEPT — the substitution never fires — so the
    /// "control" would silently duplicate the treatment. The caller should not
    /// have to know that, so the vocabulary is chosen for them.
    @Test func theControlTypeFollowsTheConditionsMode() {
        let steering = ExperimentStore.randomControlCondition(
            for: condition("fear-L14", [slot("fear", layer: 14, alpha: 0.5)]))
        #expect(steering.controlType == "randomMatchedNorm")
        #expect(steering.name == "fear-L14-random")

        let ablation = ExperimentStore.randomControlCondition(
            for: condition(
                "fear-ablate", [slot("fear", alpha: 1, mode: .ablate)],
                normUnits: false))
        #expect(ablation.controlType == "randomDirectionAblation")
        #expect(ablation.name == "fear-ablate-random")
        // The control ablates at the same λ — only the DIRECTION differs.
        #expect(ablation.slots.first?.alpha == 1)
        #expect(ablation.slots.first?.effectiveMode == .ablate)
    }

    /// A mixed condition still gets the steering control: it carries a
    /// steering slot whose magnitude is exactly what matched-norm tests.
    @Test func aMixedConditionKeepsTheMatchedNormControl() {
        let mixed = ExperimentStore.randomControlCondition(
            for: condition(
                "both",
                [
                    slot("fear", alpha: 1, mode: .ablate),
                    slot("anger", layer: 14, alpha: 0.5),
                ]))
        #expect(mixed.controlType == "randomDirectionAblation")
    }

    /// λ is never recorded in residual-norm units: claiming the flag would
    /// assert a conversion the run loop deliberately does not perform.
    @Test func anAblationControlDoesNotClaimNormUnits() {
        let ablation = ExperimentStore.randomControlCondition(
            for: condition(
                "fear-ablate", [slot("fear", alpha: 1, mode: .ablate)],
                normUnits: false))
        #expect(ablation.alphaInNormUnits == false)
    }
}
