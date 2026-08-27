import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Designated reference as a vector-builder family: the pane-shaping
/// properties, the attach-policy mirror (pooled reading from token 50), the
/// reference-class gates, and the declaration the server build posts.
///
/// The engine-side method landed 2026-07-31 (`ExtractionMethod
/// .designatedReference`, attach's `--reference` pin, both engines'
/// extraction); this suite pins the BUILDER's side of it — everything the
/// Concepts panel asks the model instead of re-deriving.
@Suite(.serialized) @MainActor struct DesignatedReferenceBuilderFamilyTests {

    private func withBuilder<T>(_ body: (ConceptBuilder, URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "dr-builder") { root in
            let previous = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previous }
            return try body(ConceptBuilder(), root)
        }
    }

    /// `prompts/emotions/<name>/stories.jsonl`, the layout both engines read.
    private func writeStories(
        _ name: String, root: URL, count: Int = 4
    ) throws {
        let directory = root.appending(components: "prompts", "emotions", name)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let lines = (0 ..< count).map {
            #"{"concept": "\#(name)", "text": "a \#(name) telling \#($0)"}"#
        }
        try (lines.joined(separator: "\n") + "\n").write(
            to: directory.appending(component: "stories.jsonl"),
            atomically: true, encoding: .utf8)
    }

    // MARK: The family

    @Test func theFamilyIsOfferedAndMapsToItsRecipeMethod() {
        let family = ConceptBuilder.RecipeFamily.designatedReference
        #expect(ConceptBuilder.RecipeFamily.allCases.contains(family))
        #expect(family.recipeMethod == .designatedReference)
        #expect(family.label == "Designated reference (stories − reference)")
    }

    @Test func itIsAnUnpairedStoryCorpusExtraction() {
        let family = ConceptBuilder.RecipeFamily.designatedReference
        // Unpaired classes (stories vs a reference corpus, never authored
        // pairs) that still read activations from stimuli — so the pane
        // keeps the reading-position and rendering controls.
        #expect(!family.isPaired)
        #expect(family.extractsFromStimuli)
        #expect(family.usesStoryCorpus)
        // Both engines extract it, so the family appears in Local too.
        #expect(!family.isServerOnly)
        #expect(family.arrivesWithResidualNorms)
    }

    @Test func theRecipeMethodSharesTheEnginesSpelling() {
        // One method identity on both substrates: the recipe stamp and the
        // extraction stamp normalize to the same token, so a builder-saved
        // artifact and a lifecycle-extracted one group together.
        #expect(VectorExtractionRecipe.Method.designatedReference.rawValue
            == ExtractionMethod.designatedReference.rawValue)
        #expect(VectorRecipeGrouping.normalizedMethod(
            recipeMethod: VectorExtractionRecipe.Method.designatedReference.rawValue,
            extractionMethod: nil)
            == VectorRecipeGrouping.normalizedMethod(
                recipeMethod: nil,
                extractionMethod: ExtractionMethod.designatedReference.rawValue))
    }

    // MARK: The attach-policy mirror

    @Test func selectingTheFamilyPinsThePooledPolicyAndTheMethod() throws {
        try withBuilder { builder, _ in
            builder.recipeFamily = .designatedReference
            // `attach --method designatedReference` pools from token 50 as
            // method policy; the builder mirrors it — and the method itself
            // is the engine enum's own case, so the sidecar stamps
            // "designatedReference", never a mean-difference alias.
            #expect(builder.poolFromToken == 50)
            #expect(builder.extractionOptions.method == .designatedReference)
            #expect(builder.extractionOptions.readingPosition == .meanFromToken(50))
            builder.recipeFamily = .caaMeanDifference
            #expect(builder.poolFromToken == nil)
            #expect(builder.extractionOptions.method == .meanDifference)
        }
    }

    @Test func thePooledPolicyTravelsAsTheLegacyInteger() throws {
        try withBuilder { builder, _ in
            builder.recipeFamily = .designatedReference
            // The route's legacy reading of an absent position for this
            // method is a pool from 50 (the attach policy), so the untouched
            // builder posts the legacy spelling — byte-compatible with the
            // grand-mean route's convention.
            let declaration = builder.serverExtractionDeclaration(
                legacyPooledDefault: 50)
            #expect(declaration.poolFromToken == 50)
            #expect(declaration.readingPosition == nil)
            #expect(declaration.extractionRendering == nil)
        }
    }

    /// The family/method sync, both directions — the interaction between this
    /// family and the demotion fix of 2026-08-27, which each landed without
    /// the other in view.
    ///
    /// The method picker's reverse sync used to map every non-PCA method onto
    /// CAA. Left that way, declaring `designatedReference` through
    /// `/api/concept/options` would have seated the CAA family under a
    /// designated-reference method: the pane showing paired editors while the
    /// extraction read story corpora. And the guard that stops a family from
    /// being demoted by the very method it already means has to count this
    /// family's own method as "not news", or selecting the family and then
    /// re-declaring its method would drop the reference picker.
    @Test func theMethodPickerPromotesIntoTheFamilyAndNeverDemotesOutOfIt() throws {
        try withBuilder { builder, _ in
            #expect(builder.recipeFamily == .caaMeanDifference)
            builder.extractionMethod = .designatedReference
            #expect(builder.recipeFamily == .designatedReference)
            // The family already MEANS this method, so seeing it arrive is
            // not evidence the researcher left the family.
            #expect(ConceptBuilder.RecipeFamily.designatedReference
                .impliedExtractionMethod == .designatedReference)
            // Declaring another method IS evidence, and demotes as it should.
            builder.extractionMethod = .meanDifference
            #expect(builder.recipeFamily == .caaMeanDifference)
        }
    }

    // MARK: The reference-class gates

    @Test func noReferenceIsSelectedByDefaultAndTheBuildStaysOff() throws {
        try withBuilder { builder, root in
            try writeStories("formality", root: root)
            builder.selectConcept("formality")
            builder.recipeFamily = .designatedReference
            // The reference is part of the recipe, so it is a deliberate
            // selection with no default — and without one there is nothing
            // this button could honestly build.
            #expect(builder.designatedReferenceConcept.isEmpty)
            #expect(builder.designatedReferenceRefusal == nil)
            #expect(!builder.canSaveAndExtract)
        }
    }

    @Test func aSelfReferenceStandsRefusedAndDisablesTheBuild() throws {
        try withBuilder { builder, root in
            try writeStories("formality", root: root)
            builder.selectConcept("formality")
            builder.recipeFamily = .designatedReference
            builder.designatedReferenceConcept = "formality"
            let refusal = try #require(builder.designatedReferenceRefusal)
            #expect(refusal.contains("zero vector"))
            #expect(refusal.contains("repair:"))
            #expect(!builder.canSaveAndExtract)
        }
    }

    @Test func bothCorporaPresentEnableTheBuild() throws {
        try withBuilder { builder, root in
            try writeStories("formality", root: root)
            try writeStories("plain-exposition", root: root)
            builder.selectConcept("formality")
            builder.recipeFamily = .designatedReference
            builder.designatedReferenceConcept = "plain-exposition"
            #expect(builder.designatedReferenceRefusal == nil)
            #expect(builder.canSaveAndExtract)
            #expect(builder.designatedReferenceOptions.contains("plain-exposition"))
        }
    }

    @Test func aMissingReferenceCorpusKeepsTheBuildOff() throws {
        try withBuilder { builder, root in
            try writeStories("formality", root: root)
            builder.selectConcept("formality")
            builder.recipeFamily = .designatedReference
            builder.designatedReferenceConcept = "never-authored"
            // Same gate the build itself refuses on: no stories.jsonl for
            // the reference means nothing to subtract.
            #expect(!builder.canSaveAndExtract)
        }
    }

    // MARK: The optimizer takes it

    @Test func theComposerMapsTheRecipeStampOntoTheManifestMethod() {
        // Unlike a reader conversion or a J-lens derivation, this recipe IS
        // pinnable: attach re-derives it from the two corpora, so the
        // composer maps rather than refuses.
        let mapping = OptimizationComposer.mapMethod(
            recipeMethodRaw: "designatedReference", extractionMethodRaw: nil)
        #expect(mapping == .mapped(.designatedReference))
    }
}
