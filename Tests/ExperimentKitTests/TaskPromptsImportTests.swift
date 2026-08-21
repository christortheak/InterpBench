import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// The Import JSONL… action's parse preview and the plain-editor JSONL
/// detector — pure functions, no globals. The write→set→pin flow tests live
/// below in an `ExperimentStoreTests` extension because they use the
/// process-global workspace override (same precedent as the confirmation
/// authoring tests).
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

/// Destination + pin flow (temp workspace) — extends the serialized
/// `ExperimentStoreTests` suite because it uses the process-global
/// workspace override (`WorkspaceRoot.programmaticOverride`), the same
/// precedent as the confirmation-authoring tests.
extension ExperimentStoreTests {

    private func withImportWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "prompts-import-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        // Shared cross-suite lock: the workspace root is process-global
        // (see ExperimentRootOverrideLock).
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: temp)
        }
        return try body(temp)
    }

    @Test func importWritesToScaffoldDestinationSetsFileAndPinsHash() throws {
        try withImportWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "case1", description: "", modelID: "org/m")
            let text = """
                {"text": "plain"}
                {"prompt": "choice", "options": ["yes", "no"], "target": "yes"}
                """
            let result = try TaskPromptsImport.importIntoStudy(
                text: text, manifest: &manifest)

            // The readiness scaffold's destination rule, exactly.
            #expect(result.file
                == DataTemplates.taskPromptsDestination(experiment: "case1"))
            #expect(result.recordCount == 2)
            let url = root.appending(path: result.file)
            let written = try Data(contentsOf: url)
            // Full-record, field-preserving: options/target survive.
            let lines = String(decoding: written, as: UTF8.self)
                .split(separator: "\n")
            #expect(lines.count == 2)
            #expect(lines[1].contains("\"options\""))
            #expect(lines[1].contains("\"target\""))
            // Pinned: manifest points at the file, hash is the file bytes.
            #expect(manifest.taskPromptsFile == result.file)
            let expectedHash = SHA256.hash(data: written)
                .map { String(format: "%02x", $0) }.joined()
            #expect(manifest.taskPromptsHash == expectedHash)
            #expect(result.hash == expectedHash)
        }
    }

    @Test func importRefusesGarbageWithLineNumberAndWritesNothing() throws {
        try withImportWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "case2", description: "", modelID: "org/m")
            let before = manifest
            do {
                _ = try TaskPromptsImport.importIntoStudy(
                    text: "{\"text\": \"ok\"}\ngarbage line", manifest: &manifest)
                Issue.record("expected an ImportError")
            } catch let error as TaskPromptsImport.ImportError {
                #expect(error.message.contains("line 2"))
                #expect(error.message.contains("refusing to import"))
            }
            // Nothing landed, nothing pinned.
            #expect(manifest == before)
            let destination = root.appending(
                path: DataTemplates.taskPromptsDestination(experiment: "case2"))
            #expect(!FileManager.default.fileExists(atPath: destination.path))
        }
    }

    @Test func importRefusesEmptyContent() throws {
        try withImportWorkspace { _ in
            var manifest = try ExperimentStore.create(
                name: "case3", description: "", modelID: "org/m")
            #expect(throws: TaskPromptsImport.ImportError.self) {
                try TaskPromptsImport.importIntoStudy(
                    text: "  \n ", manifest: &manifest)
            }
            #expect(manifest.taskPromptsFile == nil)
        }
    }

    // MARK: Overwrite semantics — the study-pack write rule (shared with
    // the tabular importer): identical bytes idempotent, differing bytes
    // refuse by default, replace only via the explicit affordance.

    @Test func importRefusesDifferingExistingFileWithoutReplace() throws {
        try withImportWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "case4", description: "", modelID: "org/m")
            let first = try TaskPromptsImport.importIntoStudy(
                text: #"{"text": "original"}"#, manifest: &manifest)
            // Identical bytes: the re-import is idempotent — same pin.
            let again = try TaskPromptsImport.importIntoStudy(
                text: #"{"text": "original"}"#, manifest: &manifest)
            #expect(again.hash == first.hash)
            // Differing bytes refuse with the sheet's remedy; neither the
            // file nor the pin moves.
            do {
                _ = try TaskPromptsImport.importIntoStudy(
                    text: #"{"text": "edited"}"#, manifest: &manifest)
                Issue.record("differing existing file should refuse")
            } catch let error as TaskPromptsImport.ImportError {
                #expect(error.message.contains("never overwrite"))
                #expect(error.message.contains("Replace the existing file"))
            }
            let written = try Data(contentsOf: root.appending(path: first.file))
            #expect(String(decoding: written, as: UTF8.self)
                .contains("original"))
            #expect(manifest.taskPromptsHash == first.hash)
        }
    }

    @Test func replaceAffordanceReplacesAndRePins() throws {
        try withImportWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "case5", description: "", modelID: "org/m")
            let first = try TaskPromptsImport.importIntoStudy(
                text: #"{"text": "original"}"#, manifest: &manifest,
                persist: { try ExperimentStore.save($0) })
            let replaced = try TaskPromptsImport.importIntoStudy(
                text: #"{"text": "edited"}"#, manifest: &manifest,
                replacingExisting: true,
                persist: { try ExperimentStore.save($0) })
            #expect(replaced.hash != first.hash)
            let written = try Data(contentsOf: root.appending(path: first.file))
            #expect(String(decoding: written, as: UTF8.self)
                .contains("edited"))
            #expect(manifest.taskPromptsHash == replaced.hash)
            #expect(try ExperimentStore.load(name: "case5").taskPromptsHash
                == replaced.hash)
        }
    }

    /// Transactional even under replace: a failure AFTER the write (here
    /// the caller's persist step) restores the PREVIOUS bytes — a failed
    /// import never leaves the destination clobbered.
    @Test func replaceRollsBackToPreviousBytesOnPersistFailure() throws {
        try withImportWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "case6", description: "", modelID: "org/m")
            let first = try TaskPromptsImport.importIntoStudy(
                text: #"{"text": "original"}"#, manifest: &manifest,
                persist: { try ExperimentStore.save($0) })
            struct PersistRefused: Error {}
            do {
                _ = try TaskPromptsImport.importIntoStudy(
                    text: #"{"text": "edited"}"#, manifest: &manifest,
                    replacingExisting: true,
                    persist: { _ in throw PersistRefused() })
                Issue.record("persist failure should throw")
            } catch is PersistRefused {}
            let written = try Data(contentsOf: root.appending(path: first.file))
            #expect(String(decoding: written, as: UTF8.self)
                .contains("original"))
            #expect(try ExperimentStore.load(name: "case6").taskPromptsHash
                == first.hash)
        }
    }

    /// A persist failure on a FIRST import (nothing pre-existing) removes
    /// the file the import created — no unpinned orphan survives.
    @Test func persistFailureRollsBackACreatedFile() throws {
        try withImportWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "case7", description: "", modelID: "org/m")
            struct PersistRefused: Error {}
            do {
                _ = try TaskPromptsImport.importIntoStudy(
                    text: #"{"text": "t"}"#, manifest: &manifest,
                    persist: { _ in throw PersistRefused() })
                Issue.record("persist failure should throw")
            } catch is PersistRefused {}
            let destination = root.appending(
                path: DataTemplates.taskPromptsDestination(experiment: "case7"))
            #expect(!FileManager.default.fileExists(atPath: destination.path))
        }
    }
}
