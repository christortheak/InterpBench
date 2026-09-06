import Foundation
import Testing
@testable import ExperimentKit

@Suite(.serialized) @MainActor struct TaskPromptsAuthoringTests {
    private let original = Data("{\"id\":\"one\",\"prompt\":\"Original?\",\"options\":[\"yes\",\"no\"],\"target\":\"yes\",\"custom\":17}\n".utf8)

    private func withRoot(_ operation: (URL) throws -> Void) throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "prompt-authoring") { root in
            let previous = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previous }
            try operation(root)
        }
    }

    private func setup(_ root: URL) throws -> ExperimentPanel {
        let url = root.appending(path: "prompts/tasks/input.jsonl")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try original.write(to: url)
        var manifest = try ExperimentStore.create(name: "study", description: "original", modelID: "test/model")
        try ExperimentStore.pinTaskPrompts("prompts/tasks/input.jsonl", into: &manifest)
        try ExperimentStore.save(manifest)
        let panel = ExperimentPanel()
        panel.management.selectedName = "study"
        panel.loadTaskPrompts()
        panel.draft.taskPromptsText = "Edited?"
        return panel
    }

    @Test func staleStudyRefusesBeforeChangingPromptBytes() throws {
        try withRoot { root in
            let panel = try setup(root)
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            var updated = reviewed.manifest
            updated.experimentDescription = "concurrent edit"
            let current = try DraftAuthoringTransaction.replace(updated, reviewed: reviewed)
            panel.saveTaskPrompts()
            #expect(try Data(contentsOf: root.appending(path: "prompts/tasks/input.jsonl")) == original)
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data == current.file.data)
        }
    }

    @Test func changedInputRefusesWithoutOverwritingOrPinning() throws {
        try withRoot { root in
            let panel = try setup(root)
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            let newer = Data("{\"prompt\":\"Concurrent input\"}\n".utf8)
            try newer.write(to: root.appending(path: "prompts/tasks/input.jsonl"), options: .atomic)
            panel.saveTaskPrompts()
            #expect(try Data(contentsOf: root.appending(path: "prompts/tasks/input.jsonl")) == newer)
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data == reviewed.file.data)
        }
    }

    @Test func successfulSavesPreserveOriginalAndFullRecordsAndAdvanceReview() throws {
        try withRoot { root in
            let panel = try setup(root)
            panel.saveTaskPrompts()
            let first = try ExperimentStore.load(name: "study")
            let firstPath = try #require(first.taskPromptsFile)
            #expect(firstPath.hasPrefix("prompts/tasks/versions/"))
            let bytes = try Data(contentsOf: root.appending(path: firstPath))
            #expect(first.taskPromptsHash == ManifestFileTransaction.digest(bytes))
            let records = try ExperimentTasks.parseTaskPrompts(bytes)
            #expect(records[0].text == "Edited?")
            #expect(records[0].options == ["yes", "no"])
            #expect(records[0].target == "yes")
            let object = try #require(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
            #expect(object["custom"] as? Int == 17)
            panel.draft.taskPromptsText = "Edited again?"
            panel.saveTaskPrompts()
            let second = try ExperimentStore.load(name: "study")
            #expect(second.taskPromptsFile != firstPath)
            #expect(try Data(contentsOf: root.appending(path: firstPath)) == bytes)
            #expect(try Data(contentsOf: root.appending(path: "prompts/tasks/input.jsonl")) == original)
            #expect(panel.draft.taskPromptsFile == second.taskPromptsFile)
        }
    }

    @Test func existingSourceCannotBeAuthorizedByLoadingAtSaveTime() throws {
        try withRoot { root in
            _ = try setup(root)
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            #expect(throws: ExperimentError.self) {
                try TaskPromptsAuthoring.save(reviewed: reviewed, path: "prompts/tasks/input.jsonl",
                    source: nil, editorText: "Unreviewed edit")
            }
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data == reviewed.file.data)
            #expect(!FileManager.default.fileExists(atPath: root.appending(path: "prompts/tasks/versions").path))
        }
    }

    @Test func capturedWorkspaceControlsPublicationAndRejectsAnotherSourcesReview() throws {
        try withRoot { root in
            _ = try setup(root)
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            let source = try TaskPromptsFileReview(path: "prompts/tasks/input.jsonl", workspaceRoot: root)
            let other = root.appending(path: "other")
            let otherFile = other.appending(path: source.path)
            try FileManager.default.createDirectory(at: otherFile.deletingLastPathComponent(), withIntermediateDirectories: true)
            try original.write(to: otherFile)
            WorkspaceRoot.programmaticOverride = other
            let wrong = try TaskPromptsFileReview(path: source.path, workspaceRoot: other)
            #expect(throws: ExperimentError.self) {
                try TaskPromptsAuthoring.save(reviewed: reviewed, path: source.path, source: wrong, editorText: "Edit")
            }
            let result = try TaskPromptsAuthoring.save(reviewed: reviewed, path: source.path, source: source, editorText: "Edit")
            #expect(FileManager.default.fileExists(atPath: root.appending(path: result.prompts.path).path))
            #expect(!FileManager.default.fileExists(atPath: other.appending(path: "prompts/tasks/versions").path))
            #expect(try Data(contentsOf: otherFile) == original)
        }
    }

    @Test func newInputsAreValidatedBeforePublicationAndOutputCollisionsRefuse() throws {
        try withRoot { root in
            _ = try setup(root)
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            let path = "prompts/tasks/new.jsonl"
            #expect(throws: ExperimentError.self) {
                try TaskPromptsAuthoring.save(reviewed: reviewed, path: path, source: nil, editorText: "   ")
            }
            let data = TaskPromptsDocument.fromTexts(["New input"]).serialized()
            let output = root.appending(path: "prompts/tasks/versions/\(ManifestFileTransaction.digest(data)).jsonl")
            try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
            let unrelated = Data("existing bytes".utf8)
            try unrelated.write(to: output)
            #expect(throws: ExperimentError.self) {
                try TaskPromptsAuthoring.save(reviewed: reviewed, path: path, source: nil, editorText: "New input")
            }
            #expect(try Data(contentsOf: output) == unrelated)
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data == reviewed.file.data)
        }
    }

    @Test func symlinkOutsideWorkspaceAndRunDestinationRefuse() throws {
        try withRoot { root in
            _ = try setup(root)
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            try FileManager.default.createSymbolicLink(atPath: root.appending(path: "outside").path,
                withDestinationPath: FileManager.default.temporaryDirectory.path)
            #expect(throws: VectorCatalog.PathError.self) {
                try TaskPromptsAuthoring.save(reviewed: reviewed, path: "outside/uncreated.jsonl", source: nil, editorText: "Edit")
            }
            try FileManager.default.createDirectory(at: root.appending(path: "runs"), withIntermediateDirectories: true)
            try FileManager.default.createSymbolicLink(atPath: root.appending(path: "prompts/tasks/versions").path,
                withDestinationPath: root.appending(path: "runs").path)
            #expect(throws: ExperimentError.self) {
                try TaskPromptsAuthoring.save(reviewed: reviewed, path: "prompts/tasks/new.jsonl", source: nil, editorText: "Edit")
            }
            #expect(try FileManager.default.contentsOfDirectory(atPath: root.appending(path: "runs").path).isEmpty)
        }
    }
}
