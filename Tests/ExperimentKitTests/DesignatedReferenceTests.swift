import Foundation
import SteeringKit
import Testing
@testable import ExperimentKit

/// designatedReference — mean(concept stories) − mean(reference stories),
/// first-class (METHODS amendment ii). Ports the contract of
/// `Server/tests/test_designated_reference.py`: attach pins concept AND
/// reference hashes with the pooled reading as method POLICY; verify treats
/// reference drift like stimulus drift; the pin surface carries both
/// stories files; the math is the mean difference. Same serialized suite as
/// the grand-mean lifecycle tests (shared `rootOverride` seam).
extension ExperimentStoreTests {

    private func writeReferenceStories(_ name: String, texts: [String]) throws {
        let directory = ExperimentStore.emotionsDirectory.appending(component: name)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let lines = texts.enumerated().map { index, text in
            #"{"concept": "\#(name)", "text": "\#(text)", "id": "\#(name)-\#(index)"}"#
        }
        try (lines.joined(separator: "\n") + "\n").write(
            to: directory.appending(component: "stories.jsonl"),
            atomically: true, encoding: .utf8)
    }

    private func makeDesignatedReferenceExperiment() throws -> ExperimentManifest {
        try writeReferenceStories(
            "courage",
            texts: ["She held the line when the room turned against her.",
                    "He signed his name although his hands were shaking."])
        try writeReferenceStories(
            "neutral",
            texts: ["The bus arrived at seven as it usually did.",
                    "The kettle boiled and the toast browned evenly."])
        var manifest = try ExperimentStore.create(
            name: "dr", description: "", modelID: "test/model", modelRevision: "abc123")
        try ExperimentStore.save(manifest)
        manifest = try ExperimentStore.attachConcept(
            "courage", method: .designatedReference, reference: "neutral",
            experimentName: "dr")
        return manifest
    }

    @Test func designatedReferenceEnumAndMath() throws {
        #expect(ExtractionMethod(rawValue: "designatedReference") == .designatedReference)
        #expect(!ExtractionMethod.designatedReference.isPaired)
        // The math IS the mean difference — classes differ, arithmetic doesn't.
        let viaDesignated = try SteeringVectorMath.direction(
            positive: [[2, 0], [4, 0]], negative: [[0, 2], [0, 4]],
            method: .designatedReference)
        let viaMeanDiff = try SteeringVectorMath.direction(
            positive: [[2, 0], [4, 0]], negative: [[0, 2], [0, 4]],
            method: .meanDifference)
        #expect(viaDesignated == viaMeanDiff)
    }

    @Test func attachPinsReferenceAndPooledReadingPolicy() throws {
        try withTempRoot {
            let manifest = try makeDesignatedReferenceExperiment()
            let ref = try #require(manifest.concepts.first { $0.name == "courage" })
            #expect(ref.options.method == .designatedReference)
            // The POLICY: pooled from 50 without anyone remembering to set it.
            #expect(ref.options.readingPosition == .meanFromToken(50))
            #expect(!ref.stimulusSetHash.isEmpty)
            let pin = try #require(ref.designatedReference)
            #expect(pin.name == "neutral")
            #expect(!pin.hash.isEmpty)
            // Round-trips through the manifest JSON (cross-engine key).
            let reloaded = try ExperimentStore.load(name: "dr")
            #expect(reloaded.concepts.first?.designatedReference == pin)
            #expect(ExperimentStore.verify(reloaded).isEmpty)
        }
    }

    @Test func attachRefusesMissingOrUnknownReference() throws {
        try withTempRoot {
            try writeReferenceStories("courage", texts: ["A brave and steady act."])
            var manifest = try ExperimentStore.create(
                name: "dr2", description: "", modelID: "test/model",
                modelRevision: "abc123")
            try ExperimentStore.save(manifest)
            #expect(throws: ExperimentError.self) {
                _ = try ExperimentStore.attachConcept(
                    "courage", method: .designatedReference, experimentName: "dr2")
            }
            #expect(throws: ExperimentError.self) {
                _ = try ExperimentStore.attachConcept(
                    "courage", method: .designatedReference,
                    reference: "nonexistent", experimentName: "dr2")
            }
        }
    }

    @Test func referenceDriftIsAVerifyViolation() throws {
        try withTempRoot {
            _ = try makeDesignatedReferenceExperiment()
            try writeReferenceStories(
                "neutral", texts: ["A completely different reference story now."])
            let violations = ExperimentStore.verify(try ExperimentStore.load(name: "dr"))
            #expect(
                violations.contains {
                    $0.contains("reference 'neutral' stories changed")
                })
        }
    }

    @Test func lifecycleIdentityCarriesTheReference() throws {
        // Identity-construction coverage ONLY (stated precisely after
        // review round 2, finding 1): recipe identity must not demand a
        // grand-mean corpus, must carry the reference, and a sidecar
        // proving a DIFFERENT reference must hash differently. Extraction,
        // validation, and the production promotion matcher are exercised
        // end-to-end in the PYTHON twin (test_persisted_sidecar_matches_
        // through_the_production_matcher) — the MLX paths need a live
        // container this CPU suite cannot host.
        try withTempRoot {
            let manifest = try makeDesignatedReferenceExperiment()
            var pinned = manifest
            pinned.modelRevision = "abc123"
            let ref = try #require(pinned.concepts.first)
            let required = try RecipeIdentity.required(manifest: pinned, ref: ref)
            let pin = try #require(ref.designatedReference)
            #expect(required.designatedReference?.concept == pin.name)
            #expect(required.designatedReference?.hash == pin.hash)
            #expect(required.grandMeanPopulation == nil)
            let canonical = RecipeIdentity.canonicalJSON(required)
            #expect(canonical.contains(
                "\"methodParameters\":{\"referenceHash\":\"\(pin.hash)\","
                    + "\"referenceName\":\"\(pin.name)\"}"))

            var sidecar = SteeringVectorSidecar(
                modelID: "test/model", revision: "abc123", concept: "courage",
                stimulusSetHash: ref.stimulusSetHash,
                vectors: ConceptVectors(perLayer: [[1, 0]]),
                options: ref.options,
                residualNormPerLayer: [1],
                residualNormSource: pinned.neutralCorpusHash != nil
                    ? "neutral-corpus" : "extraction-stimuli",
                neutralCorpusHash: pinned.neutralCorpusHash)
            sidecar.designatedReference = ["name": pin.name, "hash": pin.hash]
            let candidate = RecipeIdentity.candidate(sidecar: sidecar)
            let components = try #require(candidate.components)
            #expect(RecipeIdentity.canonicalJSON(components) == canonical)

            sidecar.designatedReference = ["name": "other", "hash": String(
                repeating: "f", count: 64)]
            let other = try #require(
                RecipeIdentity.candidate(sidecar: sidecar).components)
            #expect(RecipeIdentity.canonicalJSON(other) != canonical)

            sidecar.designatedReference = nil
            let bare = RecipeIdentity.candidate(sidecar: sidecar)
            #expect(bare.components == nil)
            #expect(bare.missingFields.contains("designatedReference"))
        }
    }

    @Test func pinSurfaceCarriesBothStoriesFiles() throws {
        try withTempRoot {
            let manifest = try makeDesignatedReferenceExperiment()
            let entries = ExperimentStore.pinnedInputEntries(manifest)
            let byLabel = Dictionary(
                uniqueKeysWithValues: entries.map { ($0.label, $0.required) })
            #expect(byLabel["concept 'courage' stories.jsonl"] == true)
            #expect(
                byLabel[
                    "concept 'courage' designated reference 'neutral' stories.jsonl"]
                    == true)
            #expect(byLabel["concept 'courage' stimulus directory"] == false)
        }
    }

    @Test func freezeSnapshotCarriesTheReferenceBytes() throws {
        // Review round 2, finding 3: the no-git floor must actually contain
        // the reference corpus — assert the snapshot OUTPUT, not the pin
        // enumeration that feeds it.
        try withTempRoot {
            let manifest = try makeDesignatedReferenceExperiment()
            try ExperimentStore.snapshotPinnedInputs(for: manifest)
            let snapshot = ExperimentStore.directory
                .appending(components: "dr", "pinned", "emotions", "neutral",
                           "stories.jsonl")
            #expect(FileManager.default.fileExists(atPath: snapshot.path))
            let bytes = try Data(contentsOf: snapshot)
            let live = try Data(contentsOf: ExperimentStore.storiesURL(for: "neutral"))
            #expect(bytes == live)
        }
    }

    @Test func canonicalIdentityBytesAreTheCrossEngineFixture() {
        // Byte-pinned canonical form, asserted verbatim in BOTH suites
        // (Python twin: test_canonical_identity_bytes_are_the_cross_engine_
        // fixture) — the identity hash is only cross-engine if these are.
        let components = RecipeIdentity.Components(
            concept: "c", modelID: "m", revision: "v",
            extractionMethod: "designatedReference", stimulusSetHash: "aa",
            readingPositionMode: "meanFromToken", readingPositionParameter: 50,
            projectionMode: "none", projectionCount: nil,
            projectionExplainedVariance: nil, projectionBasisHash: nil,
            residualNormSource: "extraction-stimuli", normCorpusHash: nil,
            grandMeanPopulation: nil,
            designatedReference: RecipeIdentity.Member(concept: "r", hash: "bb"))
        #expect(RecipeIdentity.canonicalJSON(components)
            == "{\"concept\":\"c\",\"extractionMethod\":\"designatedReference\",\"grandMeanPopulation\":null,\"methodParameters\":{\"referenceHash\":\"bb\",\"referenceName\":\"r\"},\"modelID\":\"m\",\"neutralProjection\":{\"basisHash\":null,\"count\":null,\"explainedVariance\":null,\"mode\":\"none\"},\"normCorpusHash\":null,\"readingPosition\":{\"mode\":\"meanFromToken\",\"parameter\":50},\"residualNormSource\":\"extraction-stimuli\",\"revision\":\"v\",\"schema\":1,\"stimulusSetHash\":\"aa\"}")
    }
}
