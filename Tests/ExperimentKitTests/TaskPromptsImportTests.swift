import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// The Import JSONL… action's parse preview and the plain-editor JSONL
/// detector — pure functions, no globals. Immutable publication and retained
/// study reviews are exercised in StudyAssemblySafetyTests.
struct TaskPromptsImportTests {

    // MARK: Parse preview

    @Test func previewCountsRecordsOptionsAndTargets() {
        let text = """
            {"text": "plain prompt"}
            {"prompt": "choice item", "options": ["yes", "no"], "target": "yes"}
            {"text": "options but no target", "options": ["a", "b"]}
            """
        guard case .preview(let preview) = TaskPromptsImport.preview(text) else {
            Issue.record("expected .preview")
            return
        }
        #expect(preview.recordCount == 3)
        #expect(preview.optionsCount == 2)
        #expect(preview.targetCount == 1)
        #expect(preview.summaryLine == "3 records — 2 with options, 1 with target")
    }

    @Test func emptyOptionsArrayDoesNotCountAsInstrumentBearing() {
        guard
            case .preview(let preview) = TaskPromptsImport.preview(
                #"{"text": "t", "options": []}"#)
        else {
            Issue.record("expected .preview")
            return
        }
        #expect(preview.optionsCount == 0)
    }

    @Test func firstErrorCarriesTheEditorLineNumber() {
        // Line numbers count EVERY line, blanks included — the number must
        // match what an editor shows.
        let text = "{\"text\": \"ok\"}\n\n{not json}\n{\"text\": \"never reached\""
        guard case .failure(let line, let message) = TaskPromptsImport.preview(text)
        else {
            Issue.record("expected .failure")
            return
        }
        #expect(line == 3)
        #expect(message.contains("not a JSON object"))
    }

    @Test func objectWithoutPromptKeyFailsWithItsLine() {
        let text = "{\"text\": \"ok\"}\n{\"options\": [\"a\"], \"target\": \"a\"}"
        guard case .failure(let line, let message) = TaskPromptsImport.preview(text)
        else {
            Issue.record("expected .failure")
            return
        }
        #expect(line == 2)
        // Scripted transcripts (2026-07-13) widened the accepted keys.
        #expect(message.contains("no \"prompt\", \"text\", or \"transcript\" key"))
    }

    @Test func whitespaceOnlyIsEmptyNotAnError() {
        #expect(TaskPromptsImport.preview("") == .empty)
        #expect(TaskPromptsImport.preview("  \n\n  ") == .empty)
    }

    // MARK: The looks-like-JSONL detector (plain-editor paste guard)

    @Test func detectorRecognizesRecordLines() {
        #expect(TaskPromptsImport.looksLikeJSONL(#"{"text": "a prompt"}"#))
        #expect(TaskPromptsImport.looksLikeJSONL(
            "\n  \n" + #"{"prompt": "p", "options": ["a"]}"# + "\nrest"))
    }

    @Test func detectorLeavesPlainProseAlone() {
        #expect(!TaskPromptsImport.looksLikeJSONL("Write an opinion about..."))
        // Prose that merely STARTS with '{' must stay prompt text.
        #expect(!TaskPromptsImport.looksLikeJSONL(
            "{the defendant argues} that the clause is void"))
        // A JSON object WITHOUT a prompt/text key is not a task record.
        #expect(!TaskPromptsImport.looksLikeJSONL(#"{"foo": 1}"#))
        // A JSON array line is not a record object.
        #expect(!TaskPromptsImport.looksLikeJSONL(#"["text"]"#))
        #expect(!TaskPromptsImport.looksLikeJSONL(""))
    }
}
