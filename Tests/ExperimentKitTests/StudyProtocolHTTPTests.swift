import Foundation
import Testing
@testable import ExperimentKit

@Suite(.serialized) struct StudyProtocolHTTPTests {
    private func create(_ root: URL, name: String = "study") throws -> DraftAuthoringSnapshot {
        let manifest = ExperimentManifest(name: name, description: "original", modelID: "test/model")
        try ExperimentStore.save(manifest, allowCreate: true, workspaceRoot: root, expectedFile: .absent)
        return try DraftAuthoringSnapshot(workspaceRoot: root, name: name)
    }

    private func payload(_ snapshot: DraftAuthoringSnapshot, _ fields: [String: Any] = [:]) throws -> Data {
        var object: [String: Any] = ["name": snapshot.manifest.name, "workspaceRoot": snapshot.workspaceRoot.path,
                                    "manifestFileSHA256": snapshot.file.sha256]
        object.merge(fields) { _, new in new }
        return try JSONSerialization.data(withJSONObject: object)
    }

    private func object(_ response: StudyAuthoringHTTP.Response) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: response.body) as? [String: Any])
    }

    @Test func missingAndStalePreconditionsRefuseWithoutPublication() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "protocol-http") { root in
            let reviewed = try create(root)
            let missing = try JSONSerialization.data(withJSONObject: ["name": "study", "workspaceRoot": root.path])
            let absent = StudyProtocolHTTP.apply(body: missing, workspaceRoot: root)
            #expect(absent.status == "428 Precondition Required")
            #expect(try object(absent)["code"] as? String == "manifest_precondition_required")
            let first = StudyProtocolHTTP.apply(body: try payload(reviewed, ["description": "updated"]), workspaceRoot: root)
            #expect(first.succeeded)
            let bytes = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data
            let stale = StudyProtocolHTTP.apply(body: try payload(reviewed, ["maxTokens": 128]), workspaceRoot: root)
            #expect(stale.status == "412 Precondition Failed")
            #expect(try object(stale)["code"] as? String == "staleManifest")
            #expect(try object(stale)["repairAction"] as? String != nil)
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data == bytes)
        }
    }

    @MainActor @Test func namedWriteIgnoresSelectionAndPreservesUnsavedNativeFields() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "protocol-http") { root in
            let reviewed = try create(root, name: "target")
            let other = try create(root, name: "selected")
            let panel = ExperimentPanel()
            panel.management.selectedName = "selected"
            panel.draft.protocolDescription = "unsaved native notes"
            let response = StudyProtocolHTTP.apply(
                body: try payload(reviewed, ["description": "agent edit"]), workspaceRoot: root)
            #expect(response.succeeded)
            #expect(panel.management.selectedName == "selected")
            #expect(panel.draft.protocolDescription == "unsaved native notes")
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "selected").file.data == other.file.data)
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "target").manifest.experimentDescription == "agent edit")

            // Editing the SAME study does not silently refresh its native review.
            panel.management.selectedName = "target"
            panel.refresh()
            panel.draft.protocolDescription = "unsaved overwrite"
            panel.draft.taskPromptsFile = ""
            let current = try DraftAuthoringSnapshot(workspaceRoot: root, name: "target")
            let second = StudyProtocolHTTP.apply(body: try payload(current, ["description": "second agent edit"]), workspaceRoot: root)
            #expect(second.succeeded)
            panel.saveProtocol()
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "target").manifest.experimentDescription == "second agent edit")
        }
    }

    @Test func saveFailureAndInvalidExclusionsNeverReportSuccessOrPartiallyPublish() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "protocol-http") { root in
            let reviewed = try create(root)
            let invalid: [[String: Any]] = [
                ["temperature": -0.5], ["maxTokens": 0], ["samplesPerItem": 0],
                ["seedPolicy": "unsupported"], ["reasoningEffort": "invalid"],
                ["taskPromptsFile": "prompts/absent.jsonl", "exclusionRules": [["rule": "unparseableEndpoint"]]],
                ["exclusionRules": [["rule": "unsupported"]]],
            ]
            for bad in invalid {
                let request = ["description": "must not publish"].merging(bad) { _, new in new }
                let response = StudyProtocolHTTP.apply(body: try payload(reviewed, request), workspaceRoot: root)
                #expect(!response.succeeded)
                #expect(try object(response)["ok"] as? Bool == false)
                #expect(try object(response)["repairAction"] as? String != nil)
                #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data == reviewed.file.data)
            }
        }
    }

    @Test func readAndWriteCarryExternalIdentityAndPreserveUntouchedFields() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "protocol-http") { root in
            let original = try create(root)
            var manifest = original.manifest
            manifest.systemPrompt = "preserved system"
            manifest.dtype = "float32"
            manifest.exclusionRules = [.init(rule: "unparseableEndpoint")]
            _ = try DraftAuthoringTransaction.replace(manifest, reviewed: original)
            let read = StudyProtocolHTTP.read(name: "study", workspaceRoot: root)
            #expect(read.succeeded)
            let snapshot = try object(read)
            let digest = try #require(snapshot["manifestFileSHA256"] as? String)
            let request = try JSONSerialization.data(withJSONObject: [
                "name": "study", "workspaceRoot": root.path, "manifestFileSHA256": digest,
                "description": "partial edit", "exclusionRules": [],
            ])
            let response = StudyProtocolHTTP.apply(body: request, workspaceRoot: root)
            #expect(response.succeeded)
            let saved = try object(response)
            let document = try #require(saved["document"] as? [String: Any])
            #expect(document["systemPrompt"] as? String == "preserved system")
            #expect(document["dtype"] as? String == "float32")
            #expect(document["exclusionRules"] == nil)
            #expect(document["manifestFileSHA256"] == nil)
            #expect(document["workspaceRoot"] == nil)
            #expect(saved["manifestFileSHA256"] as? String != digest)
        }
    }

    @Test func workspaceSwitchRefusesTheOldTarget() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "protocol-http") { root in
            let reviewed = try create(root)
            let other = root.appending(component: "other")
            let untouched = try create(other)
            let response = StudyProtocolHTTP.apply(body: try payload(reviewed, ["description": "wrong workspace"]), workspaceRoot: other)
            #expect(response.status == "412 Precondition Failed")
            #expect(try DraftAuthoringSnapshot(workspaceRoot: other, name: "study").file.data == untouched.file.data)
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data == reviewed.file.data)
        }
    }
}
