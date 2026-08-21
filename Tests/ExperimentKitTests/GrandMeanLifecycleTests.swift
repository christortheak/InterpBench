import CryptoKit
import Foundation
import SteeringKit
import Testing
@testable import ExperimentKit

/// D′ — grand-mean extraction as a manifest-pinnable, freeze-compatible
/// method: attach → verify → validation scoring. Ports the fixture semantics
/// of `Server/tests/test_grand_mean_lifecycle.py`. Declared as an extension
/// of the serialized `ExperimentStoreTests` suite because these tests share
/// its `rootOverride` test seam (a process-global).
extension ExperimentStoreTests {

    private func writeStories(_ name: String, texts: [String]) throws {
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

    /// Two concepts (fear/calm) with two stories each, one target — the
    /// Python suite's `_grand_mean_experiment` fixture.
    private func makeGrandMeanExperiment(
        targets: [String] = ["fear"], corpus: [String] = ["fear", "calm"]
    ) throws -> ExperimentManifest {
        try writeStories(
            "fear",
            texts: ["Her heart pounded in the dark.", "The floor creaked behind her."])
        try writeStories(
            "calm",
            texts: ["Waves lapped gently at the shore.", "The tea steamed in the quiet kitchen."])
        var manifest = try ExperimentStore.create(
            name: "gm", description: "", modelID: "test/model", modelRevision: "abc123")
        try ExperimentStore.attachGrandMeanConcepts(
            targets, corpusConcepts: corpus, into: &manifest)
        try ExperimentStore.save(manifest)
        return manifest
    }

    @Test func grandMeanEnumAndPairedGuard() {
        #expect(ExtractionMethod(rawValue: "emotionGrandMean") == .emotionGrandMean)
        #expect(
            ExtractionMethod.emotionGrandMean.rawValue
                == VectorExtractionRecipe.Method.emotionGrandMean.rawValue)
        #expect(!ExtractionMethod.emotionGrandMean.isPaired)
        #expect(ExtractionMethod.meanDifference.isPaired)
        #expect(ExtractionMethod.lat.isPaired)
        #expect(throws: SteeringVectorError.notPairedMethod("emotionGrandMean")) {
            _ = try SteeringVectorMath.direction(
                positive: [[1, 0]], negative: [[0, 1]], method: .emotionGrandMean)
        }
    }

    @Test func grandMeanAttachPinsCorpusStoriesAndReadingPosition() throws {
        try withTempRoot {
            let manifest = try makeGrandMeanExperiment()
            let ref = try #require(manifest.concepts.first)
            #expect(ref.name == "fear")
            #expect(ref.options.method == .emotionGrandMean)
            // Default reading position is the emotion paper's mean-from-token-50.
            #expect(ref.options.readingPosition == .meanFromToken(50))
            #expect(ref.stimulusSetHash == ExperimentStore.storiesHash(for: "fear"))
            let corpus = try #require(manifest.grandMeanCorpus)
            #expect(corpus.concepts == ["fear", "calm"])
            #expect(Set(corpus.hashes.keys) == ["fear", "calm"])

            // Encoded JSON shape matches the server exactly.
            let json = try #require(
                try JSONSerialization.jsonObject(with: JSONEncoder().encode(manifest))
                    as? [String: Any])
            let encodedCorpus = try #require(json["grandMeanCorpus"] as? [String: Any])
            #expect(encodedCorpus["concepts"] as? [String] == ["fear", "calm"])
            #expect((encodedCorpus["hashes"] as? [String: String])?.count == 2)
            let concepts = try #require(json["concepts"] as? [[String: Any]])
            let options = try #require(concepts.first?["options"] as? [String: Any])
            #expect(options["method"] as? String == "emotionGrandMean")
            let reading = try #require(options["readingPosition"] as? [String: Any])
            #expect((reading["meanFromToken"] as? [String: Any])?["_0"] as? Int == 50)
        }
    }

    @Test func grandMeanAttachRejectsMissingCorpusMember() throws {
        try withTempRoot {
            try writeStories("fear", texts: ["text one", "text two"])
            var manifest = try ExperimentStore.create(
                name: "gm2", description: "", modelID: "test/model")
            do {
                try ExperimentStore.attachGrandMeanConcepts(
                    ["fear"], corpusConcepts: ["fear", "ghost"], into: &manifest)
                Issue.record("expected attach to reject the missing corpus member")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("ghost"))
                #expect(error.reason.contains("stories.jsonl"))
            }
        }
    }

    @Test func grandMeanVerifyCleanThenCorpusDriftInvalidates() throws {
        try withTempRoot {
            let manifest = try makeGrandMeanExperiment()
            #expect(ExperimentStore.verify(manifest).isEmpty)
            // Drift in a NON-target corpus member still invalidates: the
            // population is part of every grand-mean vector.
            try writeStories("calm", texts: ["completely different calm story"])
            let violations = ExperimentStore.verify(manifest)
            #expect(violations.contains { $0.contains("calm") && $0.contains("changed") })
        }
    }

    @Test func grandMeanVerifyFlagsMissingCorpusAndNonmemberTarget() throws {
        try withTempRoot {
            var manifest = try makeGrandMeanExperiment()

            var missingCorpus = manifest
            missingCorpus.grandMeanCorpus = nil
            #expect(
                ExperimentStore.verify(missingCorpus).contains {
                    $0.contains("no grandMeanCorpus pinned")
                })

            manifest.grandMeanCorpus?.concepts = ["calm"]  // target dropped from corpus
            #expect(
                ExperimentStore.verify(manifest).contains { $0.contains("not a member") })
        }
    }

    @Test func grandMeanValidationScopeChangesWhenCorpusWidens() throws {
        try withTempRoot {
            let manifest = try makeGrandMeanExperiment()
            let narrow = ExperimentStore.validationScopeHash(manifest)
            var widened = manifest
            widened.grandMeanCorpus?.concepts.append("extra")
            widened.grandMeanCorpus?.hashes["extra"] = String(repeating: "00", count: 32)
            #expect(ExperimentStore.validationScopeHash(widened) != narrow)
        }
    }

    @Test func scenarioAccuracyGrandMeanPureMath() {
        let direction: [Float] = [1, 0]
        let concept: [[Float]] = [[2, 0], [4, 0]]  // mean projection 3
        let population: [[Float]] = [[1, 0], [1, 0]]  // mean projection 1 → midpoint 2
        let scenarios: [[Float]] = [[3, 0], [0.5, 0], [2.5, 0]]
        let labels = [true, false, false]  // last is wrong on purpose
        let accuracy = ConceptStats.scenarioAccuracyGrandMean(
            direction: direction, concept: concept, population: population,
            scenarios: scenarios, labels: labels)
        #expect(accuracy != nil)
        #expect(abs((accuracy ?? 0) - Float(2) / 3) < 1e-6)
        #expect(
            ConceptStats.scenarioAccuracyGrandMean(
                direction: direction, concept: [], population: population,
                scenarios: scenarios, labels: labels) == nil)
        // Unlabeled fallback shares the exact same midpoint.
        let fraction = ConceptStats.fractionAboveMidpoint(
            direction: direction, concept: concept, population: population,
            scenarios: scenarios)
        #expect(abs((fraction ?? 0) - Float(2) / 3) < 1e-6)
    }

    @Test func grandMeanCorpusDecodeRoundTripAndLegacyManifestsUnaffected() throws {
        var manifest = ExperimentManifest(
            name: "gm-codable", description: "", modelID: "test/model")
        manifest.concepts.append(
            .init(
                name: "fear", stimulusSetHash: "aa",
                options: .init(
                    method: .emotionGrandMean, readingPosition: .meanFromToken(50))))
        manifest.grandMeanCorpus = .init(
            concepts: ["fear", "calm"], hashes: ["fear": "aa", "calm": "bb"])

        let encoder = JSONEncoder()
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.self, from: encoder.encode(manifest))
        #expect(decoded == manifest)
        #expect(decoded.grandMeanCorpus?.concepts == ["fear", "calm"])

        // Legacy manifest (no grandMeanCorpus key) still decodes, with nil…
        let legacy = """
            {"name": "old", "experimentDescription": "", "createdAt": "2026-01-01",
             "modelID": "test/model", "status": "draft"}
            """
        let old = try JSONDecoder().decode(
            ExperimentManifest.self, from: Data(legacy.utf8))
        #expect(old.grandMeanCorpus == nil)

        // …and nil is omitted from the encoding, so every existing
        // manifest's content hash is unchanged by this feature.
        let plain = ExperimentManifest(name: "plain", description: "", modelID: "test/model")
        let json = try #require(
            try JSONSerialization.jsonObject(with: encoder.encode(plain))
                as? [String: Any])
        #expect(json["grandMeanCorpus"] == nil)
    }
}
