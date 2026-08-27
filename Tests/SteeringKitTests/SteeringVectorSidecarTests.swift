import Foundation
import Testing

@testable import SteeringKit

/// Pure-CPU sidecar schema tests (no model, no GPU): the `normBackfill`
/// provenance block is a pinned cross-engine JSON contract, so its key shape
/// is asserted on the raw encoded JSON, not just through Codable symmetry.
@Suite struct SteeringVectorStoreSidecarTests {

    private func makeSidecar() -> SteeringVectorSidecar {
        SteeringVectorSidecar(
            modelID: "Qwen/Qwen3-4B-MLX-4bit",
            concept: "fear",
            stimulusSetHash: "stim-hash",
            vectors: ConceptVectors(perLayer: [[1, 0], [0, 1]]))
    }

    @Test func normBackfillEncodesThePinnedKeyShape() throws {
        var sidecar = makeSidecar()
        sidecar.normBackfill = SteeringVectorSidecar.NormBackfillProvenance(
            sourceArtifact: "runs/2026-07-01T000000Z-concept-fear/fear-qwen",
            sourceVectorsHash: "ab12cd34",
            date: "2026-07-03T12:00:00Z")

        let data = try JSONEncoder().encode(sidecar)
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        let backfill = try #require(object["normBackfill"] as? [String: Any])

        // EXACTLY these three keys — the cross-engine contract.
        #expect(Set(backfill.keys) == ["sourceArtifact", "sourceVectorsHash", "date"])
        #expect(
            backfill["sourceArtifact"] as? String
                == "runs/2026-07-01T000000Z-concept-fear/fear-qwen")
        #expect(backfill["sourceVectorsHash"] as? String == "ab12cd34")
        #expect(backfill["date"] as? String == "2026-07-03T12:00:00Z")
    }

    @Test func normBackfillRoundTripsThroughCodable() throws {
        var sidecar = makeSidecar()
        sidecar.normBackfill = SteeringVectorSidecar.NormBackfillProvenance(
            sourceArtifact: "/runs/x/fear",
            sourceVectorsHash: "deadbeef",
            date: "2026-07-03T00:00:00Z")

        let decoded = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: JSONEncoder().encode(sidecar))

        #expect(decoded.normBackfill == sidecar.normBackfill)
        #expect(decoded.concept == "fear")
    }

    @Test func sidecarsWithoutNormBackfillDecodeToNil() throws {
        // Every existing artifact on disk lacks the key; decoding must not
        // fail and must not invent provenance.
        let sidecar = makeSidecar()
        let decoded = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: JSONEncoder().encode(sidecar))
        #expect(decoded.normBackfill == nil)
    }

    @Test func freshSidecarInitStampsNoBackfillProvenance() {
        #expect(makeSidecar().normBackfill == nil)
    }

    // MARK: Substrate stamp (pinned cross-engine contract)

    @Test func freshSidecarInitLeavesSubstrateNilUntilSave() {
        // The stamp happens at save (SteeringVectorStore.stamped), not at
        // construction — a constructed-but-never-saved sidecar claims nothing.
        #expect(makeSidecar().substrate == nil)
    }

    @Test func stampedFillsThisEnginesSubstrateExactly() {
        let stamped = SteeringVectorStore.stamped(makeSidecar())
        // EXACTLY the RepEReader constant — the same string that gates
        // reader scoring; never retyped.
        #expect(stamped.substrate == RepEReader.substrate)
        #expect(stamped.substrate == "swift-mlx")
    }

    @Test func stampedNeverOverwritesAnExplicitStamp() {
        var sidecar = makeSidecar()
        sidecar.substrate = "python-hf-transformers"
        #expect(
            SteeringVectorStore.stamped(sidecar).substrate
                == "python-hf-transformers")
    }

    @Test func substrateRoundTripsThroughCodable() throws {
        var sidecar = makeSidecar()
        sidecar.substrate = RepEReader.substrate
        let decoded = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: JSONEncoder().encode(sidecar))
        #expect(decoded.substrate == RepEReader.substrate)
    }

    @Test func sidecarsWithoutSubstrateDecodeToNil() throws {
        // Legacy artifacts (and older servers) omit the key: absent must
        // decode nil (engine unknown), never fail or invent a stamp.
        let decoded = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: JSONEncoder().encode(makeSidecar()))
        #expect(decoded.substrate == nil)
        // And nil stays OFF the wire (no "substrate": null key).
        let object = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(makeSidecar()))
                as? [String: Any])
        #expect(object["substrate"] == nil)
    }

    // MARK: Gemma Scope import-convention stamp (pinned cross-engine contract,
    // WS7.2 — same JSON keys on the server's vector_store.SteeringVectorSidecar)

    @Test func gemmascopeConventionFieldsRoundTripThroughCodable() throws {
        var sidecar = makeSidecar()
        sidecar.gemmascopeConvention = "analyzed-vector-norm-match"
        sidecar.rawDecoderNorm = 0.5
        sidecar.gemmascopeTargetNorm = 12.0

        let decoded = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: JSONEncoder().encode(sidecar))
        #expect(decoded.gemmascopeConvention == "analyzed-vector-norm-match")
        #expect(decoded.rawDecoderNorm == 0.5)
        #expect(decoded.gemmascopeTargetNorm == 12.0)

        // EXACTLY the pinned key names on the wire (the server reads these).
        let object = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(sidecar))
                as? [String: Any])
        #expect(object["gemmascopeConvention"] as? String == "analyzed-vector-norm-match")
        #expect((object["rawDecoderNorm"] as? NSNumber)?.doubleValue == 0.5)
        #expect((object["gemmascopeTargetNorm"] as? NSNumber)?.doubleValue == 12.0)
    }

    @Test func sidecarsWithoutConventionFieldsDecodeToNil() throws {
        // Every pre-WS7.2 artifact on disk lacks the keys: decoding must not
        // fail (lenient reader) and must not invent a convention — absence
        // is exactly what marks a pre-convention Gemma Scope import.
        let decoded = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: JSONEncoder().encode(makeSidecar()))
        #expect(decoded.gemmascopeConvention == nil)
        #expect(decoded.rawDecoderNorm == nil)
        #expect(decoded.gemmascopeTargetNorm == nil)
        // And nil stays OFF the wire (older readers keep working unchanged).
        let object = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(makeSidecar()))
                as? [String: Any])
        #expect(object["gemmascopeConvention"] == nil)
        #expect(object["rawDecoderNorm"] == nil)
        #expect(object["gemmascopeTargetNorm"] == nil)
    }

    @Test func decodingToleratesUnknownFutureKeys() throws {
        // The server writes sorted-key JSON that may grow fields first; the
        // Swift reader must stay lenient (Codable ignores unknown keys).
        let json = """
            {"modelID": "m", "concept": "c", "stimulusSetHash": "s",
             "layerCount": 1, "hiddenSize": 2, "normsPerLayer": [1.0],
             "extractionDate": "2026-01-01T00:00:00Z",
             "gemmascopeConvention": "analyzed-vector-norm-match",
             "rawDecoderNorm": 1.5, "gemmascopeTargetNorm": 3.0,
             "someFutureKey": {"nested": true}}
            """
        let decoded = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: Data(json.utf8))
        #expect(decoded.gemmascopeConvention == "analyzed-vector-norm-match")
        #expect(decoded.rawDecoderNorm == 1.5)
        #expect(decoded.gemmascopeTargetNorm == 3.0)
    }

    // MARK: gemmascopeSource passthrough (server-authored SAE identity)

    /// A sidecar exactly as the server's direct-feature-ID importer writes it.
    private static let saeSidecarJSON = """
        {"modelID": "google/gemma-3-27b-it",
         "revision": "005ad3404e59d6023443cb575daa05336842228a",
         "concept": "strict-textual-interpretation", "stimulusSetHash": "-",
         "layerCount": 2, "hiddenSize": 2, "normsPerLayer": [1.0, 1.0],
         "extractionDate": "2026-08-13T00:00:00Z",
         "gemmascopeConvention": "residual-norm-match",
         "gemmascopeSource": {"release": "gemma-scope-2-27b-it-res",
           "saeID": "layer_40_width_65k_l0_medium", "feature": 62389,
           "layer": 40, "width": 65536, "decoderRowSHA256": "abc123",
           "revision": "f00dcafe", "importPath": "direct-id",
           "neuronpediaURL": "https://neuronpedia.org/x", "clamped": false}}
        """

    @Test func gemmascopeSourceSurvivesDecodeAndReEncode() throws {
        // The block identifies WHICH decoder row a vector is. The server's
        // qualification and promotion chains match artifacts by it, so
        // dropping it on a Swift re-save destroys a citation rather than a
        // label.
        let decoded = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: Data(Self.saeSidecarJSON.utf8))
        let round = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: JSONEncoder().encode(decoded))
        #expect(round.gemmascopeSource == decoded.gemmascopeSource)

        guard case .object(let block)? = round.gemmascopeSource else {
            Issue.record("gemmascopeSource did not decode as an object")
            return
        }
        #expect(block["saeID"] == .string("layer_40_width_65k_l0_medium"))
        #expect(block["feature"] == .number(62389))
        #expect(block["clamped"] == .bool(false))
        // …and it is still there on the wire under the pinned key name.
        let object = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(round))
                as? [String: Any])
        let source = try #require(object["gemmascopeSource"] as? [String: Any])
        #expect((source["feature"] as? NSNumber)?.intValue == 62389)
        #expect(source["decoderRowSHA256"] as? String == "abc123")
    }

    @Test func sidecarsWithoutGemmascopeSourceDecodeToNilAndAddNoKey() throws {
        // Every non-SAE artifact lacks the key; absent must stay absent, or
        // every existing sidecar would gain `"gemmascopeSource": null`.
        let decoded = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: JSONEncoder().encode(makeSidecar()))
        #expect(decoded.gemmascopeSource == nil)
        let text = String(
            decoding: try JSONEncoder().encode(makeSidecar()), as: UTF8.self)
        #expect(!text.contains("gemmascopeSource"))
        #expect(makeSidecar().gemmascopeSource == nil)
    }

    // MARK: Non-finite artifact guard (pure twin of server vector_store.save)

    @Test func validateFiniteAcceptsCleanArtifacts() throws {
        try SteeringVectorStore.validateFinite(
            vectors: ConceptVectors(perLayer: [[1, 0], [0, 1]]),
            sidecar: makeSidecar())
    }

    @Test func validateFiniteRejectsNaNVectorNamingTheLayer() {
        do {
            try SteeringVectorStore.validateFinite(
                vectors: ConceptVectors(perLayer: [[1, 0], [.nan, 1]]),
                sidecar: makeSidecar())
            Issue.record("NaN vector was accepted")
        } catch let error as SteeringVectorStore.NonFiniteArtifactError {
            #expect(error.reason.contains("layer_1"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    @Test func validateFiniteRejectsNonFiniteNormSummaries() {
        var sidecar = makeSidecar()
        sidecar.residualNormPerLayer = [10, .infinity]
        #expect(throws: SteeringVectorStore.NonFiniteArtifactError.self) {
            try SteeringVectorStore.validateFinite(
                vectors: ConceptVectors(perLayer: [[1, 0], [0, 1]]),
                sidecar: sidecar)
        }
        var nanNorms = makeSidecar()
        nanNorms.normsPerLayer = [.nan, 1]
        do {
            try SteeringVectorStore.validateFinite(
                vectors: ConceptVectors(perLayer: [[1, 0], [0, 1]]),
                sidecar: nanNorms)
            Issue.record("NaN normsPerLayer was accepted")
        } catch let error as SteeringVectorStore.NonFiniteArtifactError {
            #expect(error.reason.contains("normsPerLayer"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    // MARK: - Which artifacts may state a model's depth (round 6, finding 2)

    /// One rule, in one place. `layerCount` is a ROW count, and only some
    /// artifact kinds have one row per block — a reader-derived direction
    /// writes zeros below its layer and stops, so reading a model's depth off
    /// one of them reports the reader's layer plus one.
    @Test func theDepthDiscriminatorReadsTheStampFirstThenTheMethod() {
        func states(covers: Bool? = nil, method: String? = nil,
                    recipe: String? = nil) -> Bool
        {
            var sidecar = makeSidecar()
            sidecar.coversModelDepth = covers
            sidecar.extractionMethod = method
            sidecar.recipeMethod = recipe
            return sidecar.statesModelDepth
        }
        // Reader-derived: partial, by the stamp AND by the method (so
        // pre-stamp artifacts of that family are still recognised).
        #expect(!states(covers: false))
        #expect(!states(method: "repeReaderLAT"))
        #expect(!states(recipe: "repeReaderLAT"))
        // Full-depth families, stamped or not.
        for method in [
            "lat", "meanDifference", "emotionGrandMean", "designatedReference",
            "optvec", "jlensTokenDirection", "gemmaScopeSAE", "pinnedArtifact",
        ] {
            #expect(states(method: method), "\(method)")
        }
        // Old enough to carry no method at all: the family predates the reader.
        #expect(states())
        // An explicit stamp always wins over the method.
        #expect(states(covers: true, method: "repeReaderLAT"))
        #expect(!states(covers: false, method: "lat"))
    }

    /// The stamp is a pinned cross-engine JSON key, and absent means "read the
    /// method" — so an unstamped sidecar must not gain the key.
    @Test func theDepthStampEncodesUnderItsContractKeyAndIsOmittedWhenAbsent()
        throws
    {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var stamped = makeSidecar()
        stamped.coversModelDepth = false
        let withStamp = String(decoding: try encoder.encode(stamped), as: UTF8.self)
        #expect(withStamp.contains("\"coversModelDepth\":false"))
        let without = String(
            decoding: try encoder.encode(makeSidecar()), as: UTF8.self)
        #expect(!without.contains("coversModelDepth"))
    }
}
