import Foundation
import Testing
@testable import ExperimentKit

@Suite(.serialized) struct StudyProtocolAuthoringTests {
    private func temporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "protocol-owner-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func create(_ root: URL, name: String = "study") throws -> DraftAuthoringSnapshot {
        let manifest = ExperimentManifest(name: name, description: "original", modelID: "test/model")
        try ExperimentStore.save(manifest, allowCreate: true, workspaceRoot: root, expectedFile: .absent)
        return try DraftAuthoringSnapshot(workspaceRoot: root, name: name)
    }

    private func write(_ data: Data, path: String, root: URL) throws {
        let url = root.appending(path: path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
    }

    private func semantic(_ root: URL) throws -> StudyProtocolScenario {
        let scenario = MultiAgentScenario(
            name: "dialogue", baseModelID: "", sharedMaterials: "Discuss the supplied material.",
            agents: [.init(id: "speaker", name: "Speaker", baseModelID: "", systemPrompt: "Discuss.")],
            turns: [.init(id: "turn", title: "Response", speakerAgentID: "speaker",
                          promptTemplate: "Respond.", outputLabel: "response")])
        try write(try JSONEncoder().encode(scenario), path: "prompts/panels/dialogue.json", root: root)
        return try StudyProtocolScenario(path: "prompts/panels/dialogue.json", workspaceRoot: root)
    }

    private func saved(_ result: StudyProtocolAuthoring.Result) throws -> DraftAuthoringSnapshot {
        guard case .saved(let snapshot, _, _) = result else {
            Issue.record("expected a published protocol")
            throw ExperimentError(reason: "expected a published protocol")
        }
        return snapshot
    }

    @Test func staleRequestRefusesBeforeReadingMissingInputsOrCompiling() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let reviewed = try create(root)
        let scenario = try semantic(root)
        var newer = reviewed.manifest
        newer.experimentDescription = "concurrent edit"
        let current = try DraftAuthoringTransaction.replace(newer, reviewed: reviewed)
        var fields = StudyProtocolFields()
        fields.studyKind = .multiAgent
        fields.judgeRubricFile = "prompts/rubrics/absent.txt"
        do {
            _ = try StudyProtocolAuthoring.save(reviewed: reviewed, fields: fields, scenario: scenario)
            Issue.record("expected stale manifest refusal before input work")
        } catch let error as ExperimentError {
            #expect(error.lifecycleRefusal?.gate == .staleManifest)
        }
        #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data == current.file.data)
        #expect(!FileManager.default.fileExists(atPath: root.appending(path: "prompts/panels/compiled").path))
    }

    @Test func capturedWorkspaceControlsEveryPinAndCompiledOutput() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let reviewed = try create(root)
        let scenario = try semantic(root)
        let prompts = Data("{\"id\":\"one\",\"prompt\":\"Respond.\"}\n".utf8)
        let rubric = Data("Assess the response against the declared criterion.".utf8)
        try write(prompts, path: "prompts/task.jsonl", root: root)
        try write(rubric, path: "prompts/rubrics/quality.txt", root: root)
        // Deliberately select another workspace after the request was captured.
        try ExperimentRootOverrideLock.withTempRoot(prefix: "other-protocol-workspace") { other in
            let otherBefore = try create(other)
            var fields = StudyProtocolFields()
            fields.protocolDescription = "captured command"
            fields.taskPromptsFile = "prompts/task.jsonl"
            fields.judgeRubricFile = "prompts/rubrics/quality.txt"
            let first = try saved(StudyProtocolAuthoring.save(reviewed: reviewed, fields: fields))
            #expect(first.manifest.taskPromptsHash == ManifestFileTransaction.digest(prompts))
            #expect(first.manifest.judgeRubricHash == ManifestFileTransaction.digest(rubric))
            fields.studyKind = .multiAgent
            fields.temperature = 0.2
            fields.maxTokens = 128
            let second = try saved(StudyProtocolAuthoring.save(reviewed: first, fields: fields, scenario: scenario))
            let path = try #require(second.manifest.multiAgentScenarioPath)
            let compiled = try Data(contentsOf: root.appending(path: path))
            #expect(MultiAgentScenarioStore.hash(compiled) == second.manifest.multiAgentScenarioHash)
            #expect(second.manifest.multiAgentSemanticScenarioHash == scenario.hash)
            #expect(second.manifest.taskPromptsHash == first.manifest.taskPromptsHash)
            #expect(!FileManager.default.fileExists(atPath: other.appending(path: "prompts/panels/compiled").path))
            #expect(try DraftAuthoringSnapshot(workspaceRoot: other, name: "study").file.data == otherBefore.file.data)
        }
    }

    @Test func capturedSemanticBytesKeepTheirOwnProvenanceWhenSourceChanges() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let reviewed = try create(root)
        let scenario = try semantic(root)
        // The service compiles captured values. A later disk edit must not stamp
        // their output with the hash of different semantic input bytes.
        try write(Data("changed after selection".utf8), path: scenario.path, root: root)
        var fields = StudyProtocolFields()
        fields.studyKind = .multiAgent
        let first = try saved(StudyProtocolAuthoring.save(reviewed: reviewed, fields: fields, scenario: scenario))
        #expect(first.manifest.multiAgentSemanticScenarioHash == scenario.hash)
        fields.maxTokens = 512
        let second = try saved(StudyProtocolAuthoring.save(reviewed: first, fields: fields))
        #expect(second.manifest.multiAgentSemanticScenarioHash == scenario.hash)
        #expect(second.manifest.multiAgentScenarioPath != first.manifest.multiAgentScenarioPath)
    }

    @Test func scenarioFromAnotherWorkspaceRefusesWithoutPublication() throws {
        let root = try temporaryRoot()
        let other = try temporaryRoot()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: other)
        }
        let reviewed = try create(root)
        let scenario = try semantic(other)
        var fields = StudyProtocolFields()
        fields.studyKind = .multiAgent
        do {
            _ = try StudyProtocolAuthoring.save(reviewed: reviewed, fields: fields, scenario: scenario)
            Issue.record("expected workspace mismatch refusal")
        } catch let error as ExperimentError {
            #expect(error.lifecycleRefusal?.gate == .staleManifest)
        }
        #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data == reviewed.file.data)
    }

    @Test func invalidSamplingValuesNeverPublish() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let reviewed = try create(root)
        for invalid in ["temperature", "maxTokens", "samplesPerItem"] {
            var fields = StudyProtocolFields()
            fields.protocolDescription = "must not publish"
            if invalid == "temperature" { fields.temperature = -1 }
            if invalid == "maxTokens" { fields.maxTokens = 0 }
            if invalid == "samplesPerItem" { fields.samplesPerItem = 0 }
            #expect(throws: ExperimentError.self) {
                try StudyProtocolAuthoring.save(reviewed: reviewed, fields: fields)
            }
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data == reviewed.file.data)
        }
    }
}
