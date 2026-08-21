import Foundation
import Testing
@testable import ExperimentKit

/// The Mac app's J-Space surface decodes SERVER payloads, so these tests run
/// against JSON captured from a live server rather than hand-written shapes.
/// A hand-written fixture tests my idea of the contract; a captured one tests
/// the contract.
@Suite struct JLensRemoteTests {

    private func fixture(_ name: String) throws -> Data {
        // Fixtures live beside the test file; resolved from #filePath so this
        // needs no bundle resources wiring.
        let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        return try Data(contentsOf: dir.appending(path: "Fixtures/\(name)"))
    }

    @Test func catalogDecodesFromALiveServerPayload() throws {
        let catalog = try JSONDecoder().decode(
            JLensCatalog.self, from: try fixture("jlens-catalog.json"))

        #expect(catalog.lenses.count == 1)
        let lens = catalog.lenses[0]
        #expect(lens.lensID == "google--gemma-3-4b-it--jlens-wikitext")
        #expect(lens.sourceLayers?.first == 0)
        #expect(lens.targetLayer == 33)
        #expect(lens.dModel == 2560)
        #expect(lens.converted?.dtype == "float16")
        #expect(lens.substrate == "python-hf-transformers")
        #expect(lens.layerSpan == "0…32 → 33")
    }

    @Test func theFitRevisionDecodesAsUnknownAndStaysUnknown() throws {
        // The published artifacts pin no base-model revision. Rendering the
        // runtime's there would relabel an absence of evidence as evidence, so
        // the app must be able to SEE that it is absent.
        let catalog = try JSONDecoder().decode(
            JLensCatalog.self, from: try fixture("jlens-catalog.json"))
        let fit = try #require(catalog.lenses.first?.fit)
        #expect(fit.revision == nil)
        #expect(fit.revisionKnown == false)
        #expect(fit.modelID == "google/gemma-3-4b-it")
        #expect(fit.dtype == "bfloat16")
    }

    @Test func theEvidenceTierIsCarriedPerSupportedModel() throws {
        let catalog = try JSONDecoder().decode(
            JLensCatalog.self, from: try fixture("jlens-catalog.json"))
        let byID = Dictionary(uniqueKeysWithValues:
            catalog.supported.map { ($0.modelID, $0) })

        // The tier is what keeps a 4B artifact from being read as evidence, so
        // it travels as data rather than being inferred from the model id.
        #expect(byID["google/gemma-3-27b-it"]?.isEvidenceTier == true)
        #expect(byID["google/gemma-3-4b-it"]?.isEvidenceTier == false)
        #expect(byID["google/gemma-3-4b-it"]?.tier == "testing")
    }

    @Test func anUnqualifiedLensReportsNoPassingQualifications() throws {
        let catalog = try JSONDecoder().decode(
            JLensCatalog.self, from: try fixture("jlens-catalog.json"))
        #expect(catalog.lenses.first?.passingQualifications.isEmpty == true)
    }

    @Test func aFailedQualificationIsNotCountedAsPassing() throws {
        let json = """
        {"lensID":"L","qualifications":[
          {"qualificationID":"q1","modelID":"m","revision":"r","dtype":"bfloat16","passed":false},
          {"qualificationID":"q2","modelID":"m","revision":"r","dtype":"bfloat16","passed":true}]}
        """
        let lens = try JSONDecoder().decode(
            JLensRecord.self, from: Data(json.utf8))
        #expect(lens.passingQualifications.map(\.qualificationID) == ["q2"])
    }

    @Test func tokenOptionsPreserveTheMultiTokenWarning() throws {
        // The captured payload is 'courage' on gemma-3-4b-it, which really does
        // split into 'c' + 'ourage'. Taking component [0] would derive a
        // direction for the letter c and label it courage.
        let options = try JSONDecoder().decode(
            JLensTokenOptions.self, from: try fixture("jlens-token-options.json"))

        #expect(options.selection == "explicit")   // never a recommendation
        let exact = options.candidates.filter { $0.form == "exact" }
        #expect(exact.count == 2)
        #expect(exact.allSatisfy { !$0.singleToken })
        #expect(exact.allSatisfy { ($0.note ?? "").contains("not a single token") })
        #expect(exact.allSatisfy { $0.sequence == [236755, 63587] })

        let single = try #require(
            options.candidates.first { $0.form == "leadingSpace" })
        #expect(single.singleToken)
        #expect(single.tokenID == 23648)
        #expect(single.decoded == " courage")
    }

    @Test func everyCandidateCarriesItsBytes() throws {
        // A vocabulary entry need not be printable, and two entries can render
        // identically — the hex form is what keeps them distinguishable.
        let options = try JSONDecoder().decode(
            JLensTokenOptions.self, from: try fixture("jlens-token-options.json"))
        #expect(options.candidates.allSatisfy { !$0.decodedBytes.isEmpty })
        let single = try #require(options.candidates.first { $0.singleToken })
        #expect(single.decodedBytes == Data(" courage".utf8)
            .map { String(format: "%02x", $0) }.joined())
    }

    @Test func candidateIDsAreStableAcrossFormsThatShareATokenID() throws {
        // One id can surface under two query forms; a picker keyed on the id
        // alone would collapse them into one row.
        let a = JLensTokenCandidate(
            tokenID: 5, piece: "x", decoded: "x", decodedBytes: "78",
            form: "exact", singleToken: true, sequence: [5], note: nil)
        let b = JLensTokenCandidate(
            tokenID: 5, piece: "x", decoded: "x", decodedBytes: "78",
            form: "leadingSpace", singleToken: true, sequence: [5], note: nil)
        #expect(a.id != b.id)
    }

    @Test func decodingToleratesFieldsThisEngineDoesNotKnow() throws {
        // The server owns the schema and will extend these blocks. A strict
        // decoder here would turn a server-side addition into a broken panel.
        let json = """
        {"lensID":"L","targetLayer":33,"somethingNewTheServerAdded":{"a":1},
         "source":{"repo":"r","unknownKey":true}}
        """
        let lens = try JSONDecoder().decode(
            JLensRecord.self, from: Data(json.utf8))
        #expect(lens.lensID == "L")
        #expect(lens.targetLayer == 33)
        #expect(lens.source?.repo == "r")
    }

    @Test func aLensWithNoLayersRendersAPlaceholderRatherThanCrashing() throws {
        let lens = try JSONDecoder().decode(
            JLensRecord.self, from: Data(#"{"lensID":"L"}"#.utf8))
        #expect(lens.layerSpan == "—")
        #expect(lens.passingQualifications.isEmpty)
    }
}

/// The last link in the chain the researcher actually cares about: a derived
/// direction has to arrive in the ORDINARY vector library, decoded by the same
/// `RemoteVectorRecord` every other server vector uses.
///
/// This is the failure worth guarding: `vectorArtifacts()` decodes the whole
/// list in one pass, so a jlens sidecar missing a field `RemoteVectorRecord`
/// requires would not hide one row — it would throw and empty the entire remote
/// vector picker, for every concept.
@Suite struct JLensDerivedVectorCatalogTests {

    private func fixture(_ name: String) throws -> Data {
        let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        return try Data(contentsOf: dir.appending(path: "Fixtures/\(name)"))
    }

    @Test func aDerivedDirectionDecodesAsAnOrdinaryRemoteVector() throws {
        struct Response: Decodable { var vectors: [RemoteVectorRecord] }
        let response = try JSONDecoder().decode(
            Response.self, from: try fixture("jlens-vector-catalog.json"))

        let derived = try #require(
            response.vectors.first { $0.name.hasPrefix("jlens-token-") })
        #expect(derived.concept == "jlens-token-courage-id23648")
        #expect(derived.modelID == "google/gemma-3-4b-it")
        // Model-depth, exactly like a CAA or grand-mean artifact — this is what
        // lets the ordinary agent composer pick a layer and an alpha.
        #expect(derived.layerCount == 34)
        #expect(derived.hiddenSize == 2560)
        #expect(derived.substrate == "python-hf-transformers")
    }

    @Test func theCompatibilityStimulusHashIsCarriedNotFaked() throws {
        // A derived direction has no stimulus set, and the cross-engine sidecar
        // requires that field. The prefixed form follows the Gemma Scope
        // precedent so no schema break was needed — and so a reader can tell at
        // a glance that these bytes came from a lens, not from stimuli.
        struct Response: Decodable { var vectors: [RemoteVectorRecord] }
        let response = try JSONDecoder().decode(
            Response.self, from: try fixture("jlens-vector-catalog.json"))
        let derived = try #require(
            response.vectors.first { $0.name.hasPrefix("jlens-token-") })
        let hash = try #require(derived.stimulusSetHash)
        #expect(hash.hasPrefix("jlens:"))
        #expect(hash.count > "jlens:".count + 32)
    }

    @Test func norminUnitAlphaIsHonestlyReportedAsUnavailable() throws {
        // Derivation does not measure a neutral-corpus denominator, so
        // norm-unit alpha needs the ordinary backfill first. The picker must
        // show that rather than implying the units are ready.
        struct Response: Decodable { var vectors: [RemoteVectorRecord] }
        let response = try JSONDecoder().decode(
            Response.self, from: try fixture("jlens-vector-catalog.json"))
        let derived = try #require(
            response.vectors.first { $0.name.hasPrefix("jlens-token-") })
        #expect(derived.hasResidualNorms != true)
    }
}
