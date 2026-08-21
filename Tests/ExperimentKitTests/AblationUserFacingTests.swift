import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// Every ablation refusal names a remedy, and every advisory that fires on an
/// ablation agent is one that actually applies to it.
///
/// A gate whose only available response is `--force`, or an advisory whose
/// suggested fix does not exist, teaches the researcher to ignore both.
struct AblationUserFacingTests {

    private func artifact(
        _ name: String, _ injections: [ModelVariantArtifact.InjectionRef]
    ) -> ModelVariantArtifact {
        ModelVariantArtifact(
            name: name, baseModelID: "org/m", injections: injections,
            promptMode: "chatAssistant", qwenThinkingEnabled: false,
            temperature: 0, systemPrompt: "")
    }

    // MARK: the copy the three surfaces share

    @Test func lambdaLabelsNameWhatEachValueDoes() {
        #expect(InjectionModeCopy.lambdaLabel(1) == "full removal")
        #expect(InjectionModeCopy.lambdaLabel(0.5) == "partial removal")
        #expect(InjectionModeCopy.lambdaLabel(0) == "no effect")
        #expect(InjectionModeCopy.lambdaLabel(2).contains("reflection"))
        #expect(InjectionModeCopy.lambdaLabel(1.5).contains("overshoot"))
        // A negative λ ADDS the concept — the one value whose label should
        // point at the other mode rather than describe a removal.
        #expect(InjectionModeCopy.lambdaLabel(-1).contains("Steer"))
    }

    /// Unusual values are FLAGGED, never refused: all of them are legal, and a
    /// study may want one declared deliberately.
    @Test func unusualLambdasAreFlaggedNotBlocked() {
        #expect(!InjectionModeCopy.lambdaIsUnusual(1))
        #expect(!InjectionModeCopy.lambdaIsUnusual(0.5))
        #expect(!InjectionModeCopy.lambdaIsUnusual(2))
        #expect(InjectionModeCopy.lambdaIsUnusual(-0.5))
        #expect(InjectionModeCopy.lambdaIsUnusual(3))
    }

    /// The explanation names the concept and states the two things a reader
    /// would otherwise have to be told: all layers, and the whole prompt.
    @Test func theAblationExplanationStatesTheSurprisingParts() {
        let text = InjectionModeCopy.explanation(
            mode: .ablate, concept: "fear", strength: 1)
        #expect(text.contains("fear"))
        #expect(text.contains("every layer"))
        #expect(text.contains("prompt"))
        #expect(text.contains("Nothing is added"))
    }

    @Test func theSteeringExplanationSaysItAddsRegardless() {
        let text = InjectionModeCopy.explanation(
            mode: .add, concept: "fear", strength: 2)
        #expect(text.contains("Adds"))
        #expect(text.contains("whether or not it was already there"))
    }

    @Test func theExplanationCopesWithAnUnnamedConcept() {
        let text = InjectionModeCopy.explanation(
            mode: .ablate, concept: nil, strength: 1)
        #expect(text.contains("this concept"))
        #expect(!text.contains("nil"))
    }

    /// The picker's help must draw the distinction that actually matters,
    /// since "negative alpha" is the thing a researcher will reach for first.
    @Test func thePickerHelpDistinguishesAblationFromNegativeSteering() {
        #expect(InjectionModeCopy.pickerHelp.contains("negative"))
        #expect(InjectionModeCopy.pickerHelp.contains("opposite concept"))
    }

    // MARK: refusals carry remedies

    /// A confirmation study perturbs a steering dose. Pointed at an ablation
    /// agent it would generate α±δ arms around a λ that is not a dose, and a
    /// matched-norm control with no norm to match — so it refuses, and says
    /// what to do instead.
    @Test func theConfirmationRefusalForAnAblationAgentNamesTheAlternative() throws {
        let artifact = artifact(
            "no-fear", [
                .init(
                    concept: "fear", vectorArtifactID: "runs/x/fear",
                    layer: 0, alpha: 1, mode: ModelVariantArtifact.InjectionRef.ablateMode)
            ])
        let injection = try #require(artifact.injections.first)
        #expect(injection.effectiveMode == .ablate)
        #expect(artifact.ablatesOnly)

        // The message the researcher would see, asserted for the properties
        // that make it actionable rather than for its exact wording.
        let message = ConfirmationStudyCopy.ablationRefusal(agent: "no-fear")
        #expect(message.contains("no-fear"))
        #expect(message.contains("What to do instead"))
        #expect(message.contains("variant condition"))
        #expect(message.contains("λ"))
    }

    // MARK: advisories that apply

    /// An ablation agent is correctly unpromoted — ablation has no grid, so
    /// there was no cell to select. Advising "promote agents from sweeps"
    /// would name a remedy that does not exist.
    @Test func anAblationAgentIsNotFlaggedAsHandCreated() {
        let ablation = artifact(
            "no-fear", [
                .init(
                    concept: "fear", vectorArtifactID: "runs/x/fear",
                    layer: 0, alpha: 1, mode: ModelVariantArtifact.InjectionRef.ablateMode)
            ])
        #expect(ablation.ablatesOnly)
        #expect(ablation.hasAblation)

        // A hand-tuned STEERING agent still is flagged: the advisory keeps
        // catching what it is for.
        let steering = artifact(
            "fearful", [
                .init(
                    concept: "fear", vectorArtifactID: "runs/x/fear",
                    layer: 20, alpha: 2)
            ])
        #expect(!steering.ablatesOnly)
        #expect(!steering.hasAblation)

        // A MIXED agent is not exempt: it carries a steering cell that a
        // sweep could have selected.
        let mixed = artifact(
            "both", [
                .init(
                    concept: "fear", vectorArtifactID: "runs/x/fear",
                    layer: 0, alpha: 1, mode: .ablate),
                .init(
                    concept: "anger", vectorArtifactID: "runs/x/anger",
                    layer: 20, alpha: 2),
            ])
        #expect(!mixed.ablatesOnly)
        #expect(mixed.hasAblation)
    }

    // MARK: resolution

    /// Ablation covers every layer; steering answers with its declared one.
    @Test func resolvedLayersDependOnTheMode() {
        let ablate = ModelVariantArtifact.InjectionRef(
            concept: "fear", vectorArtifactID: "v", layer: 5, alpha: 1,
            mode: .ablate)
        #expect(ablate.resolvedLayers(layerCount: 4) == [0, 1, 2, 3])

        let steer = ModelVariantArtifact.InjectionRef(
            concept: "fear", vectorArtifactID: "v", layer: 5, alpha: 2)
        #expect(steer.resolvedLayers(layerCount: 40) == [5])

        // A declared band is honoured, and out-of-range entries are dropped
        // rather than injected at a layer that does not exist.
        let narrowed = ModelVariantArtifact.InjectionRef(
            concept: "fear", vectorArtifactID: "v", layer: 0, alpha: 1,
            mode: .ablate, layers: [1, 2, 99])
        #expect(narrowed.resolvedLayers(layerCount: 4) == [1, 2])
    }
}
