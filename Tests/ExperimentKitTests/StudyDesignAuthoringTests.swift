import Foundation
import Testing

@testable import ExperimentKit

struct StudyDesignAuthoringTests {
    private func withReview(_ body: (StudyDesignSnapshot) throws -> Void) throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "design-authoring") { root in
            let study = try ExperimentStore.create(name: "study", description: "Source", modelID: "test/model")
            try StudyTemplateStore.save(StudyTemplate(name: "design", templateDescription: "Original", study: study))
            try body(StudyDesignSnapshot(workspaceRoot: root, name: "design"))
        }
    }

    @Test func descriptionSavePreservesScientificHashAndSourceStudy() throws {
        try withReview { reviewed in
            let studyBefore = try DraftAuthoringSnapshot(workspaceRoot: reviewed.workspaceRoot, name: "study")
            let saved = try StudyDesignAuthoring.updateDescription("Revised description", reviewed: reviewed)
            #expect(saved.template.templateDescription == "Revised description")
            #expect(saved.file.sha256 != reviewed.file.sha256)
            #expect(StudyTemplateStore.hash(saved.template) == StudyTemplateStore.hash(reviewed.template))
            #expect(try DraftAuthoringSnapshot(workspaceRoot: reviewed.workspaceRoot, name: "study").file.data == studyBefore.file.data)
            let unchanged = try StudyDesignAuthoring.updateDescription("Revised description", reviewed: saved)
            #expect(unchanged.file.data == saved.file.data)
            let object = try #require(JSONSerialization.jsonObject(with: saved.file.data) as? [String: Any])
            #expect(object["fileSHA256"] == nil)
            #expect(object["workspaceRoot"] == nil)
            #expect(object["revision"] == nil)
        }
    }

    @Test(arguments: ["Replacement", "Original"])
    func oldEditorRefusesEvenWhenItsTextIsUnchanged(description: String) throws {
        try withReview { reviewed in
            var changed = reviewed.template
            changed.study.maxTokens += 1
            try StudyTemplateStore.save(changed) // Simulate another design writer.
            let before = try StudyDesignSnapshot(workspaceRoot: reviewed.workspaceRoot, name: "design")
            do {
                _ = try StudyDesignAuthoring.updateDescription(description, reviewed: reviewed)
                Issue.record("An old description review cannot authorize a later design version")
            } catch let error as StudyDesignAuthoringError {
                #expect(error.code == "designChanged")
                #expect(!error.repairAction.isEmpty)
            }
            #expect(try StudyDesignSnapshot(workspaceRoot: reviewed.workspaceRoot, name: "design").file.data == before.file.data)
        }
    }

    @Test func publicAdaptersShareTheDocumentAndRefuseStaleEdits() throws {
        try withReview { reviewed in
            let root = reviewed.workspaceRoot
            func cli(_ args: [String]) throws -> ExperimentCLIResult {
                try StudyDesignCLI.run(ExperimentCLIParser.parse(namespace: "design", args + ["--json"]),
                    workspaceRoot: root, sink: .discarding)
            }
            func http(_ op: StudyDesignHTTP.Operation, _ fields: [String: Any]) throws -> StudyAuthoringHTTP.Response {
                StudyDesignHTTP.perform(op, body: try JSONSerialization.data(withJSONObject: fields), workspaceRoot: root)
            }
            let inspected = try cli(["inspect", "design"])
            #expect(inspected.payload["designFileSHA256"] == .string(reviewed.file.sha256))
            let read = try http(.inspect, ["workspaceRoot": root.path, "name": "design"])
            #expect(read.status == "200 OK")
            let readJSON = try JSONDecoder().decode([String: JSONValue].self, from: read.body)
            #expect(readJSON["design"] == .object(inspected.payload))
            let catalog = try cli(["list"])
            let list = try http(.list, ["workspaceRoot": root.path])
            let listJSON = try JSONDecoder().decode([String: JSONValue].self, from: list.body)
            #expect(listJSON["catalog"] == catalog.payload["catalog"])
            let saved = try cli(["describe", "design", "--description", "Updated", "--file-sha256", reviewed.file.sha256])
            #expect(saved.changed)
            let before = try StudyDesignSnapshot(workspaceRoot: root, name: "design")
            var request: [String: Any] = ["workspaceRoot": root.path, "name": "design", "description": "Other", "designFileSHA256": reviewed.file.sha256]
            let stale = try http(.describe, request)
            #expect(stale.status == "412 Precondition Failed")
            #expect(String(decoding: stale.body, as: UTF8.self).contains("designChanged"))
            #expect(try StudyDesignSnapshot(workspaceRoot: root, name: "design").file.data == before.file.data)
            #expect(throws: ExperimentCLIStop.self) {
                try cli(["describe", "design", "--description", "Other", "--file-sha256", reviewed.file.sha256])
            }
            request.removeValue(forKey: "designFileSHA256")
            #expect(try http(.describe, request).status == "428 Precondition Required")
            request["designFileSHA256"] = before.file.sha256
            request["workspaceRoot"] = root.appending(component: "other").path
            #expect(try http(.describe, request).status == "409 Conflict")
            request["workspaceRoot"] = root.path
            request["force"] = true
            #expect(try http(.describe, request).status == "400 Bad Request")
            request.removeValue(forKey: "force")
            #expect(try http(.describe, request).status == "200 OK")
            #expect(try StudyDesignSnapshot(workspaceRoot: root, name: "design").template.templateDescription == "Other")
        }
    }

    @Test func catalogReportsUnreadableDesignsWithoutCreatingState() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "design-empty") { root in
            let catalog = try StudyDesignAuthoring.list(workspaceRoot: root)
            #expect(catalog.entries.isEmpty)
            #expect(!FileManager.default.fileExists(atPath: root.appending(component: "templates").path))
        }
        try withReview { reviewed in
            let broken = reviewed.workspaceRoot.appending(components: "templates", "broken")
            try FileManager.default.createDirectory(at: broken, withIntermediateDirectories: true)
            try Data("invalid".utf8).write(to: broken.appending(component: "template.json"))
            let catalog = try StudyDesignAuthoring.list(workspaceRoot: reviewed.workspaceRoot)
            #expect(catalog.entries.map(\.name) == ["design"])
            #expect(catalog.issues.count == 1)
        }
    }

    @Test func updateDoesNotRecreateADeletedDesign() throws {
        try withReview { reviewed in
            try StudyTemplateStore.delete(name: "design")
            #expect(throws: (any Error).self) { try StudyDesignAuthoring.updateDescription("New", reviewed: reviewed) }
            #expect(!FileManager.default.fileExists(atPath: reviewed.workspaceRoot.appending(components: "templates", "design").path))
        }
    }

    @Test func snapshotRefusesTraversalLinksAndMismatchedNames() throws {
        try withReview { reviewed in
            let root = reviewed.workspaceRoot
            #expect(throws: StudyDesignAuthoringError.self) { try StudyDesignSnapshot(workspaceRoot: root, name: "../design") }
            let file = root.appending(components: "templates", "design", "template.json")
            var wrongName = reviewed.template
            wrongName.name = "another"
            try JSONEncoder().encode(wrongName).write(to: file)
            #expect(throws: StudyDesignAuthoringError.self) { try StudyDesignSnapshot(workspaceRoot: root, name: "design") }
            let otherFile = root.appending(component: "other-template.json")
            try reviewed.file.data.write(to: otherFile)
            try FileManager.default.removeItem(at: file)
            try FileManager.default.createSymbolicLink(at: file, withDestinationURL: otherFile)
            #expect(throws: StudyDesignAuthoringError.self) { try StudyDesignSnapshot(workspaceRoot: root, name: "design") }
            #expect(try Data(contentsOf: otherFile) == reviewed.file.data)
        }
    }
}
