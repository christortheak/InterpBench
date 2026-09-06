import Foundation
import Testing
@testable import ExperimentKit

@Suite(.serialized) struct TaskPromptsHTTPTests {
    private func setup(_ root: URL) throws -> DraftAuthoringSnapshot {
        var manifest = ExperimentManifest(name: "study", description: "original", modelID: "test/model")
        let url = root.appending(path: "prompts/input.jsonl")
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("{\"prompt\":\"Original?\",\"options\":[\"yes\",\"no\"],\"target\":\"yes\"}\n".utf8).write(to: url)
        try ExperimentStore.pinTaskPrompts("prompts/input.jsonl", into: &manifest, workspaceRoot: root)
        try ExperimentStore.save(manifest, allowCreate: true, workspaceRoot: root, expectedFile: .absent)
        return try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
    }

    private func send(_ study: DraftAuthoringSnapshot, saving: Bool, fields: [String: Any] = [:],
                      serving root: URL? = nil) throws -> StudyAuthoringHTTP.Response {
        var object: [String: Any] = ["name": study.manifest.name, "workspaceRoot": study.workspaceRoot.path,
                                    "file": "prompts/input.jsonl"]
        object.merge(fields) { _, new in new }
        return TaskPromptsHTTP.perform(body: try JSONSerialization.data(withJSONObject: object),
            saving: saving, workspaceRoot: root ?? study.workspaceRoot)
    }

    private func edit(_ study: DraftAuthoringSnapshot) -> [String: Any] {
        ["manifestFileSHA256": study.file.sha256, "promptsFileSHA256": study.manifest.taskPromptsHash!, "text": "Edited?"]
    }

    private func object(_ response: StudyAuthoringHTTP.Response) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: response.body) as? [String: Any])
    }

    @Test func namedReadAndSaveReturnAuthoritativeStudyAndInputVersions() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "prompt-http") { root in
            let reviewed = try setup(root)
            let before = try Data(contentsOf: root.appending(path: "prompts/input.jsonl"))
            let read = try send(reviewed, saving: false)
            #expect(read.succeeded)
            let readObject = try object(read)
            let prompts = try #require(readObject["prompts"] as? [String: Any])
            #expect(prompts["promptsFileSHA256"] as? String == reviewed.manifest.taskPromptsHash)
            #expect(prompts["text"] as? String == "Original?")
            let saved = try send(reviewed, saving: true, fields: edit(reviewed))
            #expect(saved.succeeded)
            let savedObject = try object(saved)
            let studyResult = try #require(savedObject["study"] as? [String: Any])
            let current = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            #expect(studyResult["manifestFileSHA256"] as? String == current.file.sha256)
            let output = try Data(contentsOf: root.appending(path: #require(current.manifest.taskPromptsFile)))
            #expect(current.manifest.taskPromptsHash == ManifestFileTransaction.digest(output))
            #expect(try ExperimentTasks.parseTaskPrompts(output)[0].options == ["yes", "no"])
            #expect(try Data(contentsOf: root.appending(path: "prompts/input.jsonl")) == before)
            #expect(try send(reviewed, saving: true, fields: edit(reviewed)).status == "412 Precondition Failed")
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data == current.file.data)
        }
    }

    @Test func missingMalformedAndUnknownPreconditionsRefuse() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "prompt-http") { root in
            let reviewed = try setup(root)
            #expect(try send(reviewed, saving: true, fields: ["text": "Edit"]).status == "428 Precondition Required")
            #expect(try send(reviewed, saving: true, fields: ["text": "Edit", "manifestFileSHA256": reviewed.file.sha256]).status == "428 Precondition Required")
            let invalidFields: [[String: Any]] = [
                ["promptsFileSHA256": "invented"], ["manifestFileSHA256": "invented"],
                ["unexpected": true], ["text": 123]
            ]
            for extra in invalidFields {
                var fields = edit(reviewed)
                fields.merge(extra) { _, new in new }
                #expect(try send(reviewed, saving: true, fields: fields).status == "400 Bad Request")
            }
            let empty = TaskPromptsHTTP.perform(body: Data("{}".utf8), saving: false, workspaceRoot: root)
            #expect(empty.status == "400 Bad Request")
            #expect(try object(empty)["repairAction"] != nil)
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data == reviewed.file.data)
        }
    }

    @Test func changedSourceOrServingWorkspaceNeverPublishes() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "prompt-http") { root in
            let reviewed = try setup(root)
            let newer = Data("{\"prompt\":\"Concurrent edit\"}\n".utf8)
            try newer.write(to: root.appending(path: "prompts/input.jsonl"), options: .atomic)
            #expect(try send(reviewed, saving: true, fields: edit(reviewed)).status == "412 Precondition Failed")
            let other = root.appending(path: "other")
            try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
            #expect(try send(reviewed, saving: true, fields: edit(reviewed), serving: other).status == "412 Precondition Failed")
            #expect(try send(reviewed, saving: false, serving: other).status == "412 Precondition Failed")
            #expect(try Data(contentsOf: root.appending(path: "prompts/input.jsonl")) == newer)
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data == reviewed.file.data)
        }
    }

    @Test func explicitAbsentSourceCreatesNewVersionButCannotReplaceExistingSource() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "prompt-http") { root in
            let reviewed = try setup(root)
            var fields: [String: Any] = ["manifestFileSHA256": reviewed.file.sha256, "sourceAbsent": true, "text": "New input"]
            #expect(try send(reviewed, saving: true, fields: fields).status == "412 Precondition Failed")
            fields["file"] = "prompts/new.jsonl"
            #expect(try send(reviewed, saving: true, fields: fields).succeeded)
            let current = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            #expect(current.manifest.taskPromptsFile?.hasPrefix("prompts/tasks/versions/") == true)
        }
    }

    @Test @MainActor func namedOperationsDoNotSelectOrChangeNativeEditor() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "prompt-http") { root in
            let reviewed = try setup(root)
            var other = ExperimentManifest(name: "other", description: "other", modelID: "test/model")
            other.taskPromptsFile = nil
            try ExperimentStore.save(other, allowCreate: true, workspaceRoot: root, expectedFile: .absent)
            let panel = ExperimentPanel()
            panel.management.selectedName = "other"
            panel.draft.taskPromptsText = "Unsaved editor input"
            #expect(try send(reviewed, saving: false).succeeded)
            #expect(try send(reviewed, saving: true, fields: edit(reviewed)).succeeded)
            #expect(panel.management.selectedName == "other")
            #expect(panel.draft.taskPromptsText == "Unsaved editor input")
            #expect(try ExperimentStore.load(name: "other").taskPromptsFile == nil)
        }
    }
}
