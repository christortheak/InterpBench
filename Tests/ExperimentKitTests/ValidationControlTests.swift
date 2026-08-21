import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// C2 — discriminant-validity controls as DECLARED, pinned recipe references.
///
/// Swift used to build its cosine matrix from "every other concept on disk",
/// extracting each with the first pinned paired concept's options. Two
/// separate faults:
///
/// - The control SET was ambient. It changed whenever unrelated work landed
///   in the workspace, so `worstCosinePair` was not a property of the study
///   and the same manifest produced different discriminant evidence on two
///   machines.
/// - The control RECIPE was borrowed, so a control authored for grand-mean
///   extraction was read at the wrong position by the wrong method.
///
/// Python meanwhile had no controls at all, so the engines disagreed about
/// what validate even measures.
struct ValidationControlTests {

    private func manifest(
        controls: [ExperimentManifest.ValidationControl]? = nil,
        concepts: [String] = ["fear"]
    ) -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: "vc", description: "", modelID: "test/model")
        manifest.modelRevision = "abc123"
        manifest.concepts = concepts.map {
            .init(name: $0, stimulusSetHash: String(repeating: "f", count: 64),
                  options: .init())
        }
        manifest.validationControls = controls
        return manifest
    }

    private func control(
        _ name: String, method: ExtractionMethod = .meanDifference,
        revision: String? = nil
    ) -> ExperimentManifest.ValidationControl {
        .init(
            concept: name,
            stimulusSetHash: String(repeating: "c", count: 64),
            options: .init(method: method),
            modelRevision: revision)
    }

    // MARK: the declaration is the control set

    @Test func aControlCarriesItsOwnRecipeNotABorrowedOne() {
        // The whole point: the control's options travel with the control.
        let grandMean = control("golden-gate", method: .emotionGrandMean)
        #expect(grandMean.options.method == .emotionGrandMean)
        let paired = control("hungry", method: .meanDifference)
        #expect(paired.options.method == .meanDifference)
    }

    @Test func controlsRoundTripThroughTheManifest() throws {
        let original = manifest(controls: [control("golden-gate"), control("hungry")])
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.self, from: try encoder.encode(original))
        #expect(decoded.validationControls?.count == 2)
        #expect(decoded.validationControls?.first?.concept == "golden-gate")
        #expect(
            decoded.validationControls?.first?.stimulusSetHash
                == String(repeating: "c", count: 64))
    }

    /// The regression that must never happen: new optional fields leaking
    /// into a legacy manifest's encoding and changing its content hash.
    @Test func absentControlsDoNotChangeALegacyManifestsHash() throws {
        var legacy = manifest()
        legacy.validationControls = nil
        let before = ExperimentStore.manifestHash(legacy)
        // Round-tripping must not materialise the key.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(legacy)
        #expect(!String(decoding: data, as: UTF8.self).contains("validationControls"))
        let decoded = try JSONDecoder().decode(ExperimentManifest.self, from: data)
        #expect(ExperimentStore.manifestHash(decoded) == before)
    }

    @Test func declaringAControlChangesTheManifestHash() {
        // Controls are evidence-bearing, so they belong inside the firewall's
        // content hash like every other declared input.
        let without = ExperimentStore.manifestHash(manifest())
        let with = ExperimentStore.manifestHash(
            manifest(controls: [control("golden-gate")]))
        #expect(without != with)
    }

    // MARK: nothing disappears silently

    @Test func undeclaredConceptsOnDiskAreNamed() throws {
        let advisories = ExperimentStore.undeclaredControlAdvisories(
            manifest(controls: [control("golden-gate")]),
            availableConcepts: ["fear", "golden-gate", "hungry", "sympathy"])
        let advisory = try #require(advisories.first)
        // The two undeclared ones, and not the pinned concept or the
        // declared control.
        #expect(advisory.contains("hungry"))
        #expect(advisory.contains("sympathy"))
        #expect(!advisory.contains("golden-gate"))
        // Says how to keep them — removing an implicit behaviour must not be
        // invisible to whoever relied on it.
        #expect(advisory.contains("validationControls"))
    }

    @Test func nothingIsSaidWhenEveryConceptIsAccountedFor() {
        #expect(
            ExperimentStore.undeclaredControlAdvisories(
                manifest(controls: [control("golden-gate")]),
                availableConcepts: ["fear", "golden-gate"]).isEmpty)
        #expect(
            ExperimentStore.undeclaredControlAdvisories(
                manifest(), availableConcepts: ["fear"]).isEmpty)
    }

    // MARK: controls are pinned inputs

    @Test func controlStimuliJoinThePinSurface() {
        let entries = ExperimentStore.pinnedInputEntries(
            manifest(controls: [control("golden-gate")]))
        let labels = entries.map(\.label)
        #expect(labels.contains { $0.contains("validation control 'golden-gate'") })
        // Required: a bundle missing a declared control's stimuli would
        // produce different discriminant evidence on the far side.
        let entry = entries.first { $0.label.contains("validation control") }
        #expect(entry?.required == true)
        #expect(entry?.url.path.hasSuffix("prompts/concepts/golden-gate") == true)
    }
}
