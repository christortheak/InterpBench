import Foundation
import Testing

@testable import ExperimentKit

/// The sweep's qualitative record: `dev-generations.jsonl`.
///
/// The field incident this closes (server twin:
/// `test_sweep_dev_generations.py`): the sweep generated every dev-prompt
/// text per cell, computed the constraint numbers from them, emitted a
/// preview line each — and dropped them, so a dose ladder's only prose
/// evidence was log previews. The record's SCHEMA is the cross-engine
/// contract ({kind, concept, layer, alpha, promptIndex, text}); the encoder
/// is pure, so it is pinned here without a model or a sweep run.
struct SweepDevGenerationsTests {

    private func parsed(_ line: String) throws -> [String: Any] {
        let object = try JSONSerialization.jsonObject(with: Data(line.utf8))
        return try #require(object as? [String: Any])
    }

    @Test func recordCarriesTheContractFields() throws {
        let record = try parsed(
            SweepRunCatalog.devGenerationLine(
                kind: "cell", concept: "fear", layer: 12, alpha: 0.04,
                promptIndex: 3, text: "dread filled the town"))
        #expect(
            Set(record.keys) == [
                "kind", "concept", "layer", "alpha", "promptIndex", "text",
            ])
        #expect(record["kind"] as? String == "cell")
        #expect(record["concept"] as? String == "fear")
        #expect(record["layer"] as? Int == 12)
        #expect(record["alpha"] as? Double == 0.04)
        #expect(record["promptIndex"] as? Int == 3)
        #expect(record["text"] as? String == "dread filled the town")
        // One LINE per record: embedded newlines must stay escaped.
        let multiline = try SweepRunCatalog.devGenerationLine(
            kind: "cell", concept: "fear", layer: 2, alpha: 0.1,
            promptIndex: 0, text: "first\nsecond")
        #expect(!multiline.contains("\n"))
    }

    @Test func baselineRecordsCarryANullConcept() throws {
        // The baseline texts are concept-independent (generated once for the
        // whole sweep), so the record says so: concept null, the CSV's
        // layer -1 / alpha 0 anchor convention.
        let record = try parsed(
            SweepRunCatalog.devGenerationLine(
                kind: "baseline", concept: nil, layer: -1, alpha: 0,
                promptIndex: 0, text: "the town woke slowly"))
        #expect(record["concept"] is NSNull)
        #expect(record["layer"] as? Int == -1)
    }

    @Test func overlongTextIsCappedWithAFlag() throws {
        let record = try parsed(
            SweepRunCatalog.devGenerationLine(
                kind: "cell", concept: "fear", layer: 2, alpha: 0.1,
                promptIndex: 0,
                text: String(
                    repeating: "x",
                    count: SweepRunCatalog.devGenerationTextLimit + 5)))
        #expect(record["truncated"] as? Bool == true)
        let text = try #require(record["text"] as? String)
        #expect(text.count == SweepRunCatalog.devGenerationTextLimit)
        // A short text carries NO flag — absence means whole.
        let whole = try parsed(
            SweepRunCatalog.devGenerationLine(
                kind: "cell", concept: "fear", layer: 2, alpha: 0.1,
                promptIndex: 0, text: "short"))
        #expect(whole["truncated"] == nil)
    }

    @Test func appendWritesOneDurableLinePerGeneration() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "sweep-dev-gen-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        try SweepRunCatalog.appendDevGeneration(
            runDirectory: directory, kind: "baseline", concept: nil,
            layer: -1, alpha: 0, promptIndex: 0, text: "calm morning")
        try SweepRunCatalog.appendDevGeneration(
            runDirectory: directory, kind: "cell", concept: "fear",
            layer: 2, alpha: 0.1, promptIndex: 0, text: "dread at dawn")

        let url = directory.appending(
            component: SweepRunCatalog.devGenerationsFile)
        let lines = try String(contentsOf: url, encoding: .utf8)
            .split(separator: "\n").map(String.init)
        #expect(lines.count == 2)
        #expect(try parsed(lines[0])["kind"] as? String == "baseline")
        let cell = try parsed(lines[1])
        #expect(cell["concept"] as? String == "fear")
        #expect(cell["text"] as? String == "dread at dawn")
    }

    @Test func fileNameMatchesTheServersConstant() {
        // `tasks.DEV_GENERATIONS_FILE` on the server — one name, two
        // engines, or the results explorer and the evidence reader would
        // see two different records.
        #expect(SweepRunCatalog.devGenerationsFile == "dev-generations.jsonl")
    }
}
