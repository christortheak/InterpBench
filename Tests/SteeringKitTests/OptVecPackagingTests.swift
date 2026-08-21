import Foundation
import Testing

@testable import SteeringKit

/// §24: an OptVec artifact's PACKAGING and its α denominator, decoded on this
/// engine.
///
/// The trap: every other family stores a direction the consumer scales by its
/// own α against `residualNormPerLayer`. This family stores the vector
/// pre-scaled to full trained magnitude, so a consumer that scales it again
/// overdoses by orders of magnitude. The marker is what makes that checkable
/// from the artifact rather than remembered — and this engine must decode it,
/// display it, and PRESERVE it through a backfill.
@Suite struct OptVecPackagingTests {

    /// Shaped exactly as the server's `optvec_train._save_artifact` writes it
    /// (denominated run: an α factor against a donor artifact's measured
    /// norms). Absent keys are absent, never null — the family convention.
    private let serverSidecarJSON = """
        {
          "schemaVersion": 2,
          "modelID": "google/gemma-3-27b-it",
          "concept": "optvec-toy",
          "stimulusSetHash": "optvec:abc123",
          "layerCount": 2,
          "hiddenSize": 2,
          "normsPerLayer": [0.0, 4935.09],
          "extractionDate": "2026-08-20T00:00:00Z",
          "extractionMethod": "optvec",
          "recipeMethod": "optvec",
          "substrate": "python-hf-transformers",
          "optvec": {
            "layer": 1,
            "alphaAbsolute": 4935.09,
            "alphaNormFactor": 0.1,
            "vectorPackaging": "preScaledFullMagnitude",
            "residualNorm": {
              "residualNormPerLayer": [40000.0, 49350.9],
              "residualNormArtifact": "runs/20260819T000000Z-backfill/fear",
              "residualNormSource": "neutral-corpus",
              "neutralCorpusHash": "deadbeefcafe",
              "residualNormConvention": "wholeCorpusMean-v1"
            },
            "seed": 17,
            "runID": "20260820T000000Z-optvec-toy",
            "substrate": "python-hf-transformers",
            "claim": "sufficiency"
          }
        }
        """

    private func decoded() throws -> SteeringVectorSidecar {
        try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: Data(serverSidecarJSON.utf8))
    }

    @Test func swiftDecodesTheServerMintedBlock() throws {
        let facts = try #require(OptVecPackaging.facts(from: decoded()))

        #expect(facts.packaging == "preScaledFullMagnitude")
        #expect(facts.isPreScaled)
        #expect(!facts.packagingIsUnknown)
        #expect(facts.layer == 1)
        #expect(facts.alphaAbsolute == 4935.09)
        #expect(facts.alphaNormFactor == 0.1)
        #expect(facts.residualNormSource == "neutral-corpus")
        #expect(facts.neutralCorpusHash == "deadbeefcafe")
        #expect(facts.residualNormConvention == "wholeCorpusMean-v1")
        #expect(
            facts.residualNormArtifact == "runs/20260819T000000Z-backfill/fear")
    }

    /// The whole point of the opaque passthrough: `NormBackfill` decodes →
    /// mutates → re-encodes through this struct, so a block that does not
    /// survive the round trip is a block a Swift-side backfill DESTROYS —
    /// after which `ExperimentStore.attachArtifact` refuses the result for
    /// "carrying no 'optvec' provenance block".
    @Test func theBlockSurvivesDecodeReencodeVerbatim() throws {
        let sidecar = try decoded()
        let reencoded = try JSONEncoder().encode(sidecar)
        let object = try #require(
            JSONSerialization.jsonObject(with: reencoded) as? [String: Any])
        let block = try #require(object["optvec"] as? [String: Any])

        #expect(block["vectorPackaging"] as? String == "preScaledFullMagnitude")
        #expect(block["alphaNormFactor"] as? Double == 0.1)
        #expect(block["claim"] as? String == "sufficiency")
        let denominator = try #require(block["residualNorm"] as? [String: Any])
        #expect(denominator["residualNormConvention"] as? String == "wholeCorpusMean-v1")
        #expect(denominator["neutralCorpusHash"] as? String == "deadbeefcafe")

        // And decoding the re-encoding gives the identical opaque value.
        let round = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: reencoded)
        #expect(round.optvec == sidecar.optvec)
    }

    /// Absent stays absent: a non-OptVec artifact gains no key and no facts.
    @Test func nonOptVecArtifactsAreUntouched() throws {
        let sidecar = SteeringVectorSidecar(
            modelID: "m", concept: "fear", stimulusSetHash: "h",
            vectors: ConceptVectors(perLayer: [[1, 0]]))
        #expect(sidecar.optvec == nil)
        #expect(OptVecPackaging.facts(from: sidecar) == nil)
        #expect(OptVecPackaging.advisory(for: sidecar) == nil)

        let object = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(sidecar))
                as? [String: Any])
        #expect(object["optvec"] == nil)
    }

    /// A pre-§24 OptVec artifact carries no marker. "Unknown" must not
    /// collapse into "safe to scale" — the two are different states and the
    /// advisory says so.
    @Test func preStampArtifactsReadAsUnknownNotAsSafe() throws {
        let legacy = """
            {"schemaVersion": 2, "modelID": "m", "concept": "c",
             "stimulusSetHash": "optvec:x", "layerCount": 1, "hiddenSize": 2,
             "normsPerLayer": [4935.09],
             "extractionDate": "2026-08-01T00:00:00Z",
             "extractionMethod": "optvec",
             "optvec": {"layer": 0, "alphaAbsolute": 4935.09, "seed": 1}}
            """
        let sidecar = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: Data(legacy.utf8))
        let facts = try #require(OptVecPackaging.facts(from: sidecar))

        #expect(facts.packagingIsUnknown)
        #expect(!facts.isPreScaled)
        #expect(facts.alphaNormFactor == nil)
        #expect(facts.residualNormConvention == nil)

        let advisory = try #require(OptVecPackaging.advisory(for: sidecar))
        #expect(advisory.contains("NO vectorPackaging marker"))
        #expect(advisory.contains("do not scale it"))
    }

    @Test func advisoryStatesTheTrapAndTheTrainedDose() throws {
        let advisory = try #require(OptVecPackaging.advisory(for: decoded()))
        #expect(advisory.contains("PRE-SCALED"))
        #expect(advisory.contains("do not re-normalize"))
        #expect(advisory.contains("residual-norm units"))
        #expect(advisory.contains("wholeCorpusMean-v1"))
    }

    /// A denominated artifact whose donor predated the convention stamp says
    /// "legacy (pre-stamp)" — never today's rule by default.
    @Test func legacyDonorConventionIsNamedHonestly() throws {
        let json = """
            {"schemaVersion": 2, "modelID": "m", "concept": "c",
             "stimulusSetHash": "optvec:x", "layerCount": 1, "hiddenSize": 2,
             "normsPerLayer": [1.0],
             "extractionDate": "2026-08-01T00:00:00Z",
             "extractionMethod": "optvec",
             "optvec": {"layer": 0, "alphaAbsolute": 10.0,
                        "alphaNormFactor": 0.1,
                        "vectorPackaging": "preScaledFullMagnitude",
                        "residualNorm": {"residualNormSource": "neutral-corpus"}}}
            """
        let sidecar = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: Data(json.utf8))
        let advisory = try #require(OptVecPackaging.advisory(for: sidecar))
        #expect(advisory.contains("legacy (pre-stamp)"))
    }
}
