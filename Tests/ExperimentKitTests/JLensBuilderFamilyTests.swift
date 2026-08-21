import Foundation
import Testing
@testable import ExperimentKit
@testable import SteeringKit

/// J-lens as a vector-builder family: the naming rule, and the pane-shaping
/// properties the view asks about instead of re-deriving.
@Suite @MainActor struct JLensBuilderFamilyTests {

    private func builder() -> ConceptBuilder { ConceptBuilder() }

    // MARK: The family

    @Test func theFamilyIsOfferedAndMapsToItsRecipeMethod() {
        let family = ConceptBuilder.RecipeFamily.jlensTokenDirection
        #expect(ConceptBuilder.RecipeFamily.allCases.contains(family))
        #expect(family.recipeMethod == .jlensTokenDirection)
        #expect(family.label == "J-lens token direction")
    }

    @Test func itIsADerivationNotAnExtraction() {
        // The pane hides stimulus, pooling, and reading-position controls on
        // this flag — showing them would imply provenance the artifact lacks.
        #expect(ConceptBuilder.RecipeFamily.jlensTokenDirection.extractsFromStimuli == false)
        for other in ConceptBuilder.RecipeFamily.allCases
        where other != .jlensTokenDirection {
            #expect(other.extractsFromStimuli, "\(other) should extract from stimuli")
        }
    }

    @Test func itIsServerOnlyAndTheOthersAreNot() {
        // Lens artifacts are PyTorch/HF-native; there is nothing to fall back to
        // locally, so the family is absent in a local workspace rather than
        // present-and-refusing.
        #expect(ConceptBuilder.RecipeFamily.jlensTokenDirection.isServerOnly)
        for other in ConceptBuilder.RecipeFamily.allCases
        where other != .jlensTokenDirection {
            #expect(!other.isServerOnly, "\(other) should not be server-only")
        }
    }

    @Test func aFreshDerivedVectorHasNoResidualNorms() {
        // Every other family produces norm-ready vectors, so the difference has
        // to be stated rather than discovered at alpha-selection time.
        #expect(ConceptBuilder.RecipeFamily.jlensTokenDirection
            .arrivesWithResidualNorms == false)
    }

    @Test func selectingTheFamilyClearsPooling() {
        let b = builder()
        b.poolFromToken = 50
        b.recipeFamily = .jlensTokenDirection
        #expect(b.poolFromToken == nil)
    }

    // MARK: The naming rule

    @Test func theTokenIDIsAlwaysAppendedToTheUsersLabel() {
        let b = builder()
        b.jlensLabel = "courage"
        b.jlensSelectedTokenID = 23648
        b.jlensSelectedPiece = " courage"
        #expect(b.jlensArtifactName == "courage-id23648")
    }

    @Test func twoDirectionsSharingALabelCannotCollide() {
        // THE case this rule exists for: token 23648 is " courage" and 236755 is
        // the letter "c". Under a bare label plus version numbers these would be
        // "courage" and "courage-2" — recording that there are two and nothing
        // about which is which, in the artifact name where nothing downstream
        // can recover it.
        let b = builder()
        b.jlensLabel = "courage"
        b.jlensSelectedTokenID = 23648
        let word = b.jlensArtifactName
        b.jlensSelectedTokenID = 236755
        let letter = b.jlensArtifactName
        #expect(word == "courage-id23648")
        #expect(letter == "courage-id236755")
        #expect(word != letter)
    }

    @Test func anEmptyLabelFallsBackToThePieceNotToNothing() {
        let b = builder()
        b.jlensSelectedTokenID = 23648
        b.jlensSelectedPiece = " courage"
        #expect(b.jlensArtifactName == "jlens-token-courage-id23648")
    }

    @Test func labelsAreMadeFilesystemSafeWithoutLosingTheID() {
        let b = builder()
        b.jlensSelectedTokenID = 7
        for raw in ["a/b", "../escape", "with spaces", "emoji🙂here", "!!!"] {
            b.jlensLabel = raw
            let name = try! #require(b.jlensArtifactName)
            #expect(name.hasSuffix("-id7"))
            #expect(!name.contains("/"))
            #expect(!name.contains(".."))
            #expect(!name.contains(" "))
        }
    }

    @Test func aLabelOfOnlyPunctuationStillProducesAUsableName() {
        let b = builder()
        b.jlensSelectedTokenID = 7
        b.jlensLabel = "///"
        #expect(b.jlensArtifactName == "jlens-token-id7")
    }

    @Test func noNameUntilAnExactTokenIsSelected() {
        // A word is not an identity. Naming before selection would let the label
        // stand in for the token, which is the mis-selection the picker prevents.
        let b = builder()
        b.jlensLabel = "courage"
        #expect(b.jlensArtifactName == nil)
    }

    // MARK: Duplicates vs collisions

    private func record(concept: String) -> SubstrateVectorRecord {
        SubstrateVectorRecord(
            id: "runs/x/\(concept)", concept: concept,
            modelID: "google/gemma-3-27b-it",
            extractionMethod: "jlensTokenDirection")
    }

    @Test func anIdenticalDerivationIsRecognizedAsAlreadyExisting() {
        // Same name means same lens AND same token — a duplicate to reuse, not a
        // clash to version around. Auto-versioning here would mint two
        // byte-identical artifacts.
        let b = builder()
        b.jlensLabel = "courage"
        b.jlensSelectedTokenID = 23648
        #expect(b.jlensDuplicateExists(in: [record(concept: "courage-id23648")]))
    }

    @Test func adifferentTokenUnderTheSameLabelIsNotADuplicate() {
        let b = builder()
        b.jlensLabel = "courage"
        b.jlensSelectedTokenID = 236755
        #expect(!b.jlensDuplicateExists(in: [record(concept: "courage-id23648")]))
    }

    @Test func noSelectionMeansNoDuplicateClaim() {
        let b = builder()
        b.jlensLabel = "courage"
        #expect(!b.jlensDuplicateExists(in: [record(concept: "courage-id23648")]))
    }

    // MARK: The optimizer must refuse it

    @Test func theOptimizerRefusesADerivedDirectionWithItsOwnReason() {
        // Like a reader conversion, this has no stimulus-set recipe a manifest
        // could pin and re-derive — but the advice differs: telling someone to
        // "re-extract it in Data" would be wrong, because the artifact IS
        // reproducible, just not as an extraction.
        let mapping = OptimizationComposer.mapMethod(
            recipeMethodRaw: "jlensTokenDirection", extractionMethodRaw: nil)
        guard case .refused(let refusal) = mapping else {
            Issue.record("a J-lens direction must not map onto an extraction recipe")
            return
        }
        #expect(refusal.message.contains("J-lens"))
        #expect(!refusal.message.contains("re-extract it in Data to make it optimizable"))
    }

    @Test func theRefusalAlsoAppliesViaTheExtractionMethodFallback() {
        let mapping = OptimizationComposer.mapMethod(
            recipeMethodRaw: nil, extractionMethodRaw: "jlensTokenDirection")
        guard case .refused = mapping else {
            Issue.record("the raw-string path must refuse it too")
            return
        }
    }

    @Test func theRecipeMethodRoundTripsThroughTheSharedEnum() {
        #expect(VectorExtractionRecipe.Method(rawValue: "jlensTokenDirection")
            == .jlensTokenDirection)
        #expect(VectorExtractionRecipe.Method.jlensTokenDirection.label
            == "J-lens token direction")
    }
}

/// The optional concept association, and the rule it protects.
@Suite @MainActor struct JLensConceptAssociationTests {

    @Test func noAssociationByDefault() {
        // Typing a name must never file a one-token direction under a
        // stimulus-defined concept: in every concept-grouped view they would
        // read as two extractions of one thing.
        let b = ConceptBuilder()
        #expect(b.jlensAssociatedConcept.isEmpty)
    }

    @Test func theNameNeverImpliesTheAssociation() {
        // The label and the grouping key are separate on purpose. Naming a
        // direction "courage" says what the token is; it does not claim the
        // direction IS the concept courage.
        let b = ConceptBuilder()
        b.jlensLabel = "courage"
        b.jlensSelectedTokenID = 23648
        #expect(b.jlensArtifactName == "courage-id23648")
        #expect(b.jlensAssociatedConcept.isEmpty)
    }

    @Test func anAssociationIsCarriedSeparatelyFromTheName() {
        let b = ConceptBuilder()
        b.jlensLabel = "courage"
        b.jlensSelectedTokenID = 23648
        b.jlensAssociatedConcept = "courage"
        // The artifact name keeps the token id regardless — the bytes stay
        // unambiguous even when the grouping is deliberately shared.
        #expect(b.jlensArtifactName == "courage-id23648")
        #expect(b.jlensAssociatedConcept == "courage")
    }
}
