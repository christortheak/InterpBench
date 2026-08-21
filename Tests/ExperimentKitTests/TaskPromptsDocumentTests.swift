import CryptoKit
import Foundation
import Testing
@testable import ExperimentKit

/// The task-prompts editor data-loss fix: the editor round-trips FULL JSONL
/// records — `options`, `target`, and unknown keys survive a text edit.
struct TaskPromptsDocumentTests {

    let fixture = """
        {"id": "case-1", "text": "Affirm or reverse?", "options": ["affirm", "reverse"], "target": "affirm", "customKey": {"nested": [1, 2.5, null]}}
        {"prompt": "Sentence in months?", "anchorMonths": 24, "arm": "high"}
        {"text": "A plain prompt"}

        """

    @Test func loadExposesTextAndPreservesRawLines() throws {
        let document = try TaskPromptsDocument.load(Data(fixture.utf8))
        #expect(document.count == 3)
        #expect(
            document.texts == [
                "Affirm or reverse?", "Sentence in months?", "A plain prompt",
            ])
        #expect(document.optionsItemCount == 1)
        #expect(document.instrumentSummary?.contains("1 of 3 items") == true)
        // Serialization without edits is byte-faithful (modulo the trailing
        // newline normalization of JSONL).
        #expect(document.serialized() == Data(fixture.utf8))
    }

    @Test func editingOneTextPreservesEveryOtherKeyAndLine() throws {
        let document = try TaskPromptsDocument.load(Data(fixture.utf8))
        var blocks = TaskPromptsDocument.editorBlocks(document.editorText)
        blocks[0] = "Affirm, reverse, or remand?"
        let edited = document.applyingEditedTexts(blocks)
        let lines = String(decoding: edited.serialized(), as: UTF8.self)
            .split(separator: "\n").map(String.init)
        #expect(lines.count == 3)

        // Untouched lines are BYTE-identical to the originals.
        #expect(lines[1] == #"{"prompt": "Sentence in months?", "anchorMonths": 24, "arm": "high"}"#)
        #expect(lines[2] == #"{"text": "A plain prompt"}"#)

        // The edited line keeps options/target/unknown keys with equal JSON
        // values — only the text changed.
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(lines[0].utf8))
                as? [String: Any])
        #expect(object["text"] as? String == "Affirm, reverse, or remand?")
        #expect(object["options"] as? [String] == ["affirm", "reverse"])
        #expect(object["target"] as? String == "affirm")
        #expect(object["id"] as? String == "case-1")
        let custom = try #require(object["customKey"] as? [String: Any])
        let nested = try #require(custom["nested"] as? [Any])
        #expect(nested.count == 3)
        #expect(nested[0] as? Int == 1)
        #expect(nested[1] as? Double == 2.5)
        #expect(nested[2] is NSNull)

        // The instrument badge survives the edit.
        #expect(edited.optionsItemCount == 1)
    }

    @Test func editingPromptKeyedLineKeepsPromptKey() throws {
        let document = try TaskPromptsDocument.load(Data(fixture.utf8))
        var blocks = document.texts
        blocks[1] = "How many months?"
        let lines = String(
            decoding: document.applyingEditedTexts(blocks).serialized(), as: UTF8.self
        ).split(separator: "\n").map(String.init)
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(lines[1].utf8))
                as? [String: Any])
        #expect(object["prompt"] as? String == "How many months?")
        #expect(object["text"] == nil)
        #expect(object["anchorMonths"] as? Int == 24)
        #expect(object["arm"] as? String == "high")
    }

    @Test func addingAndRemovingBlocksPairsByIndex() throws {
        let document = try TaskPromptsDocument.load(Data(fixture.utf8))
        // Extra block becomes a fresh {"text": …} record.
        let grown = document.applyingEditedTexts(document.texts + ["A new prompt"])
        #expect(grown.count == 4)
        let lines = String(decoding: grown.serialized(), as: UTF8.self)
            .split(separator: "\n").map(String.init)
        #expect(lines[3] == #"{"text":"A new prompt"}"#)
        // Dropped block drops the tail record.
        let shrunk = document.applyingEditedTexts(Array(document.texts.prefix(2)))
        #expect(shrunk.count == 2)
        #expect(shrunk.optionsItemCount == 1)
    }

    @Test func pinHashTracksSerializedBytes() throws {
        // The pin flows exactly as before: SHA-256 over the file's raw
        // bytes. An edit changes the bytes → changes the hash; a no-op save
        // (byte-identical serialization) keeps it.
        func sha256(_ data: Data) -> String {
            SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }
        let document = try TaskPromptsDocument.load(Data(fixture.utf8))
        #expect(sha256(document.serialized()) == sha256(Data(fixture.utf8)))
        var blocks = document.texts
        blocks[2] = "A different plain prompt"
        let edited = document.applyingEditedTexts(blocks)
        #expect(sha256(edited.serialized()) != sha256(Data(fixture.utf8)))
    }

    @Test func runLoopParserReadsEditedInstrumentFile() throws {
        // End-to-end sanity with the ACTUAL run-loop parser: after a text
        // edit, the logprob instrument still sees options + target.
        let document = try TaskPromptsDocument.load(Data(fixture.utf8))
        var blocks = document.texts
        blocks[0] = "Affirm or reverse the judgment below?"
        let prompts = try ExperimentTasks.parseTaskPrompts(
            document.applyingEditedTexts(blocks).serialized())
        #expect(prompts.count == 3)
        #expect(prompts[0].text == "Affirm or reverse the judgment below?")
        #expect(prompts[0].options == ["affirm", "reverse"])
        #expect(prompts[0].target == "affirm")
        #expect(prompts[1].anchorMonths == 24)
    }

    @Test func responseFormatItemsSpeakTheRunLoadersIDVocabulary() throws {
        // 2026-08-03 field incident: the editor synthesized `item-N` ids for
        // the scope rule while both engines' run loaders resolve the file's
        // REAL id (fallback `prompt-<ordinal>`), so a scope pinned in the
        // editor refused at run start with "the same COUNT of different
        // items". The document now speaks the loaders' vocabulary; the
        // golden hash below is shared verbatim with the server test
        // (test_scope_pin_ids_match_the_run_loader_vocabulary).
        let fixture = """
            {"id": "a", "text": "t1", "options": ["A", "B"], "responseFormat": "label"}
            {"id": "b", "text": "t2", "options": ["A", "B"], "responseFormat": "label"}
            {"id": "c", "text": "t3", "options": ["A", "B"], "responseFormat": "json"}
            {"text": "t4", "options": ["A", "B"], "responseFormat": "label"}
            """
        let document = try TaskPromptsDocument.load(Data(fixture.utf8))
        let items = document.responseFormatItems
        #expect(items.map(\.id) == ["a", "b", "c", "prompt-4"])
        let scope = ResponseFormat.Scope.pin(
            responseFormats: ["label"], items: items)
        #expect(scope.itemCount == 3)
        #expect(scope.itemIDsHash
            == "e3dc84bffe0488d1ab6084ad6359d8a8f9d7fd5a6655c1763b33282bd24918cf")
        // And the pin verifies against the RUN loader's own view.
        let prompts = try ExperimentTasks.parseTaskPrompts(Data(fixture.utf8))
        let runItems = ExperimentTasks.responseFormatItems(prompts)
        #expect(scope.driftRefusal(items: runItems) == nil)
    }
}
