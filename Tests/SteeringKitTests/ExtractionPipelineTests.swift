import Foundation
import Testing
@testable import SteeringKit

@Suite struct StimulusSetTests {

    private func makeTempConceptDir(positive: String, negative: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "concept-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try positive.write(
            to: dir.appending(component: "positive.jsonl"), atomically: true, encoding: .utf8)
        try negative.write(
            to: dir.appending(component: "negative.jsonl"), atomically: true, encoding: .utf8)
        return dir
    }

    @Test func parsesJSONLAndHashes() throws {
        let dir = try makeTempConceptDir(
            positive: "{\"text\": \"Bonjour le monde\"}\n{\"text\": \"Chat noir\"}\n",
            negative: "{\"text\": \"Hello world\"}\n{\"text\": \"Black cat\"}\n")
        defer { try? FileManager.default.removeItem(at: dir) }

        let set = try StimulusSet(directory: dir)
        #expect(set.positive == ["Bonjour le monde", "Chat noir"])
        #expect(set.negative == ["Hello world", "Black cat"])
        #expect(set.hash.count == 64)

        // Hash must be content-determined: same bytes, same hash.
        let again = try StimulusSet(directory: dir)
        #expect(again.hash == set.hash)
    }

    @Test func malformedLineThrows() throws {
        let dir = try makeTempConceptDir(
            positive: "{\"text\": \"ok\"}\nnot json\n",
            negative: "{\"text\": \"ok\"}\n")
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect(throws: StimulusSetError.self) {
            try StimulusSet(directory: dir)
        }
    }

    @Test func missingFileThrows() {
        let dir = FileManager.default.temporaryDirectory
            .appending(component: "missing-\(UUID().uuidString)")
        #expect(throws: StimulusSetError.self) {
            try StimulusSet(directory: dir)
        }
    }

    /// WP0 dry-run punch list P2-10: "stimulus schema discoverable only by
    /// brute force (errors name file:line, never the expected keys; one
    /// missing file per invocation)". Authoring a concept headlessly cost one
    /// round trip per fact.
    @Test func stimulusRefusalsNameEveryMissingFileAndTheRowShape() throws {
        let empty = FileManager.default.temporaryDirectory
            .appending(component: "empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }
        do {
            _ = try StimulusSet(directory: empty)
            Issue.record("an empty concept directory must not load")
        } catch let error as StimulusSetError {
            let text = error.description
            // BOTH files, in one refusal.
            #expect(text.contains("positive.jsonl"))
            #expect(text.contains("negative.jsonl"))
            // And the shape, so the next invocation can be the right one.
            #expect(text.contains("\"text\""))
        }

        // One side present: the refusal names only what is actually absent.
        let half = try makeTempConceptDir(
            positive: "{\"text\": \"ok\"}\n", negative: "{\"text\": \"ok\"}\n")
        defer { try? FileManager.default.removeItem(at: half) }
        try FileManager.default.removeItem(
            at: half.appending(component: "negative.jsonl"))
        do {
            _ = try StimulusSet(directory: half)
            Issue.record("a half-populated concept directory must not load")
        } catch let error as StimulusSetError {
            #expect(error.description.contains("negative.jsonl"))
            #expect(!error.description.contains("positive.jsonl"))
        }

        // A malformed ROW names the key the parser wanted, with its line.
        let malformed = try makeTempConceptDir(
            positive: "{\"text\": \"ok\"}\n{\"prompt\": \"wrong key\"}\n",
            negative: "{\"text\": \"ok\"}\n")
        defer { try? FileManager.default.removeItem(at: malformed) }
        do {
            _ = try StimulusSet(directory: malformed)
            Issue.record("a wrong-key row must not load")
        } catch let error as StimulusSetError {
            #expect(error.description.contains("positive.jsonl:2"))
            #expect(error.description.contains(StimulusSetError.textRowShape))
        }
    }
}

@Suite struct SteeringVectorStoreTests {

    /// safetensors + sidecar round trip preserves vectors and provenance.
    @Test func saveLoadRoundTrip() throws {
        let vectors = ConceptVectors(perLayer: [[1, 2, 3], [4, 5, 6], [-1, 0, 1]])
        let sidecar = SteeringVectorSidecar(
            modelID: "test/model", concept: "french",
            stimulusSetHash: "abc123", vectors: vectors)

        let dir = FileManager.default.temporaryDirectory
            .appending(component: "vectors-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        try SteeringVectorStore.save(
            vectors: vectors, sidecar: sidecar, to: dir, name: "french-test")
        let (loaded, loadedSidecar) = try SteeringVectorStore.load(
            from: dir, name: "french-test")

        #expect(loaded.perLayer == vectors.perLayer)
        #expect(loadedSidecar.modelID == "test/model")
        #expect(loadedSidecar.stimulusSetHash == "abc123")
        #expect(loadedSidecar.layerCount == 3)
        #expect(loadedSidecar.hiddenSize == 3)
        #expect(loadedSidecar.normsPerLayer.count == 3)
    }
}
