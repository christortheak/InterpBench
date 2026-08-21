import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Pure guard/assembly tests for the residual-norm backfill job — the GPU
/// measurement path (`NormBackfill.backfillNorms` end-to-end) is
/// compile-verified only and exercised by the smoke list.
@Suite struct NormBackfillTests {

    private func makeSidecar(
        modelID: String = "Qwen/Qwen3-4B-MLX-4bit",
        norms: [Float]? = nil
    ) -> SteeringVectorSidecar {
        SteeringVectorSidecar(
            modelID: modelID,
            concept: "fear",
            stimulusSetHash: "stim-hash",
            vectors: ConceptVectors(perLayer: [[1, 0], [0, 1], [1, 1]]),
            residualNormPerLayer: norms,
            residualNormSource: norms == nil ? nil : "extraction-stimuli")
    }

    // MARK: Guards (pure — no container needed)

    @Test func refusesWhenNormsAlreadyPresent() {
        let sidecar = makeSidecar(norms: [10, 20, 30])
        #expect(throws: NormBackfill.BackfillError.self) {
            try NormBackfill.validate(
                sidecar: sidecar, loadedModelID: "Qwen/Qwen3-4B-MLX-4bit")
        }
        do {
            try NormBackfill.validate(
                sidecar: sidecar, loadedModelID: "Qwen/Qwen3-4B-MLX-4bit")
        } catch let error as NormBackfill.BackfillError {
            #expect(error.reason.contains("backfill never overwrites"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    @Test func hardModelGuardRejectsMismatchedLoadedModel() {
        let sidecar = makeSidecar()
        do {
            try NormBackfill.validate(
                sidecar: sidecar, loadedModelID: "mlx-community/gemma-3-4b-it-4bit")
            Issue.record("model guard did not fire")
        } catch let error as NormBackfill.BackfillError {
            // Reader-guard message style: both ids named, per-model rationale.
            #expect(error.reason.contains("Qwen/Qwen3-4B-MLX-4bit"))
            #expect(error.reason.contains("mlx-community/gemma-3-4b-it-4bit"))
            #expect(error.reason.contains("per-model measurement"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    @Test func matchingModelWithoutNormsPassesValidation() throws {
        try NormBackfill.validate(
            sidecar: makeSidecar(), loadedModelID: "Qwen/Qwen3-4B-MLX-4bit")
    }

    // MARK: Redenomination (loud, never silent — the freeze --force pattern)

    @Test func redenominateAcceptsStimulusDenominatedArtifacts() throws {
        // extraction-stimuli norms are the legacy, concept-dependent
        // denominator — the flag re-measures on the neutral corpus.
        try NormBackfill.validate(
            sidecar: makeSidecar(norms: [10, 20, 30]),
            loadedModelID: "Qwen/Qwen3-4B-MLX-4bit",
            redenominate: true)
    }

    @Test func redenominateRefusesAlreadyNeutralArtifacts() {
        var sidecar = makeSidecar(norms: [10, 20, 30])
        sidecar.residualNormSource = "neutral-corpus abc123def456"
        do {
            try NormBackfill.validate(
                sidecar: sidecar, loadedModelID: "Qwen/Qwen3-4B-MLX-4bit",
                redenominate: true)
            Issue.record("already-neutral guard did not fire")
        } catch let error as NormBackfill.BackfillError {
            #expect(error.reason.contains("nothing to redenominate"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    @Test func redenominationStampsTheReplacedSource() {
        let updated = NormBackfill.backfilledSidecar(
            from: makeSidecar(norms: [10, 20, 30]),
            residualNormPerLayer: [1, 2, 3],
            corpusHash: "c0ffee",
            sourceArtifact: "runs/x/fear",
            sourceVectorsHash: "beef")
        #expect(updated.normBackfill?.replacedNormSource == "extraction-stimuli")
        #expect(updated.residualNormSource?.hasPrefix("neutral-corpus") == true)
        // Classic fill of a norm-less artifact records no replacement.
        let classic = NormBackfill.backfilledSidecar(
            from: makeSidecar(),
            residualNormPerLayer: [1, 2, 3],
            corpusHash: "c0ffee",
            sourceArtifact: "runs/x/fear",
            sourceVectorsHash: "beef")
        #expect(classic.normBackfill?.replacedNormSource == nil)
    }

    // MARK: Layer alignment

    @Test func alignedNormsRequireAtLeastTheVectorLayerCount() {
        #expect(throws: NormBackfill.BackfillError.self) {
            _ = try NormBackfill.alignedNorms([1, 2], layerCount: 3)
        }
    }

    @Test func alignedNormsPassThroughOnExactMatch() throws {
        #expect(try NormBackfill.alignedNorms([1, 2, 3], layerCount: 3) == [1, 2, 3])
    }

    @Test func alignedNormsTakeThePrefixForDerivedArtifacts() throws {
        // Reader-/SAE-derived vectors carry layers only up to the injection
        // layer; the full-model measurement aligns by block-index prefix.
        #expect(try NormBackfill.alignedNorms([1, 2, 3, 4], layerCount: 2) == [1, 2])
    }

    @Test func alignedNormsRejectNonFiniteMeasurementsEvenPastThePrefix() {
        // An overflow anywhere in the measurement (even at layers beyond the
        // artifact's) means the forward pass blew up — the denominator is
        // untrustworthy. Same rule as the server twin.
        do {
            _ = try NormBackfill.alignedNorms([1, 2, .nan], layerCount: 2)
            Issue.record("non-finite measurement was accepted")
        } catch let error as NormBackfill.BackfillError {
            #expect(error.reason.contains("non-finite"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
        #expect(throws: NormBackfill.BackfillError.self) {
            _ = try NormBackfill.alignedNorms([1, .infinity, 3], layerCount: 3)
        }
    }

    // MARK: Sidecar assembly

    @Test func backfilledSidecarStampsNormsSourceHashAndProvenance() throws {
        let corpusHash = "0123456789abcdef0123456789abcdef"
        let result = NormBackfill.backfilledSidecar(
            from: makeSidecar(),
            residualNormPerLayer: [10, 20, 30],
            corpusHash: corpusHash,
            sourceArtifact: "/runs/2026-07-01T000000Z-concept-fear/fear-qwen",
            sourceVectorsHash: "deadbeef",
            date: Date(timeIntervalSince1970: 1_782_000_000))

        #expect(result.residualNormPerLayer == [10, 20, 30])
        // EXACT string extraction stamps for neutral-corpus norms.
        #expect(result.residualNormSource == "neutral-corpus 0123456789ab")
        #expect(result.neutralCorpusHash == corpusHash)
        let backfill = try #require(result.normBackfill)
        #expect(
            backfill.sourceArtifact
                == "/runs/2026-07-01T000000Z-concept-fear/fear-qwen")
        #expect(backfill.sourceVectorsHash == "deadbeef")
        // ISO8601 UTC.
        #expect(backfill.date.hasSuffix("Z"))
        #expect(backfill.date.contains("T"))
    }

    @Test func backfilledSidecarPreservesEveryOtherField() throws {
        // Reader-derived provenance must survive the backfill untouched.
        var original = makeSidecar()
        original.source = "repe-reader-lat"
        original.readerID = "reader-fear-layer12.json"
        original.readerHash = "feedface"
        original.controlMode = "reading-vector activation addition"
        original.readingPosition = "last token"
        original.substrate = RepEReader.substrate

        let result = NormBackfill.backfilledSidecar(
            from: original,
            residualNormPerLayer: [1, 2, 3],
            corpusHash: "cafe",
            sourceArtifact: "/runs/x/fear",
            sourceVectorsHash: "beef")

        #expect(result.modelID == original.modelID)
        #expect(result.concept == original.concept)
        #expect(result.stimulusSetHash == original.stimulusSetHash)
        #expect(result.layerCount == original.layerCount)
        #expect(result.hiddenSize == original.hiddenSize)
        #expect(result.normsPerLayer == original.normsPerLayer)
        #expect(result.extractionDate == original.extractionDate)
        #expect(result.readingPosition == "last token")
        #expect(result.source == "repe-reader-lat")
        #expect(result.readerID == "reader-fear-layer12.json")
        #expect(result.readerHash == "feedface")
        #expect(result.controlMode == "reading-vector activation addition")
        // The engine stamp is provenance: a backfilled artifact keeps the
        // ORIGINAL's substrate (the model guard already pins the model, and
        // backfill never re-extracts on a different engine).
        #expect(result.substrate == RepEReader.substrate)
    }

    @Test func backfilledSidecarKeepsAnUnstampedOriginalUnstamped() {
        // Legacy originals with no substrate stamp stay honest: backfill
        // must not invent an engine claim the original never made.
        let result = NormBackfill.backfilledSidecar(
            from: makeSidecar(),
            residualNormPerLayer: [1, 2, 3],
            corpusHash: "cafe",
            sourceArtifact: "/runs/x/fear",
            sourceVectorsHash: "beef")
        #expect(result.substrate == nil)
    }

    // MARK: Stamped reading position

    @Test func readingPositionParsesTheSidecarLabels() throws {
        var sidecar = makeSidecar()
        sidecar.readingPosition = "last token"
        #expect(try NormBackfill.readingPosition(for: sidecar) == .lastToken)

        sidecar.readingPosition = "mean from token 50"
        #expect(try NormBackfill.readingPosition(for: sidecar) == .meanFromToken(50))

        // Pre-options artifacts (nil label) read at the last token.
        sidecar.readingPosition = nil
        #expect(try NormBackfill.readingPosition(for: sidecar) == .lastToken)

        // An unparseable label errors — never guesses.
        sidecar.readingPosition = "median token"
        #expect(throws: NormBackfill.BackfillError.self) {
            _ = try NormBackfill.readingPosition(for: sidecar)
        }
    }
}
