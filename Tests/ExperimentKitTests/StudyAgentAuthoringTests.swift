import Foundation
import Testing

@testable import ExperimentKit

struct StudyAgentAuthoringTests {
    private func fixture(_ body: (DraftAuthoringSnapshot, AgentArtifactSnapshot) throws -> Void) throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "study-agent") { root in
            _ = try ExperimentStore.create(name: "study", description: "Reviewed", modelID: "test/model")
            let path = "runs/model-variants/agent/model-variant.json"
            let url = root.appending(path: path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let artifact = ModelVariantArtifact(name: "agent", baseModelID: "test/model",
                promptMode: "chatAssistant", qwenThinkingEnabled: false, temperature: 0, systemPrompt: "Instruction")
            try JSONEncoder().encode(artifact).write(to: url)
            try body(DraftAuthoringSnapshot(workspaceRoot: root, name: "study"),
                AgentArtifactSnapshot(workspaceRoot: root, path: path))
        }
    }

    @Test func publicAdaptersInspectAndAttachTheSameArtifact() throws {
        try fixture { study, agent in
            let root = study.workspaceRoot
            let inspected = try StudyAgentCLI.run(ExperimentCLIParser.parse(namespace: "agent", ["inspect", agent.path, "--json"]), workspaceRoot: root, sink: .discarding)
            let body = try JSONSerialization.data(withJSONObject: ["workspaceRoot": root.path, "artifactPath": agent.path])
            let http = StudyAgentHTTP.perform(.inspect, body: body, workspaceRoot: root)
            #expect(http.status == "200 OK")
            #expect(try JSONDecoder().decode([String: JSONValue].self, from: http.body) == inspected.payload)
            #expect(inspected.payload["artifactFileSHA256"] == .string(agent.file.sha256))
            let catalog = try StudyAgentAuthoring.list(workspaceRoot: root)
            #expect(catalog.agents.count == 1)
            #expect(catalog.agents.first?.path == agent.path)
            let invocation = try ExperimentCLIParser.parse(namespace: "experiment", ["attach-agent", "study", "--artifact", agent.path,
                "--artifact-sha256", agent.file.sha256, "--manifest-sha256", study.file.sha256, "--json"])
            let result = try StudyAgentCLI.attach(invocation, workspaceRoot: root, sink: .discarding)
            let saved = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            #expect(result.changed)
            #expect(saved.manifest.variantConditions.first?.artifactPath == agent.path)
            #expect(saved.manifest.variantConditions.first?.artifactHash == agent.file.sha256)
            #expect(saved.manifest.variantConditions.first?.artifact == agent.record.artifact)
            #expect(saved.manifest.experimentDescription == study.manifest.experimentDescription)
            #expect(try Data(contentsOf: agent.record.url) == agent.file.data)
        }
    }

    @Test func bothFileReviewsAreRequiredAndStaleWritesRefuse() throws {
        try fixture { study, agent in
            let root = study.workspaceRoot
            var fields: [String: Any] = ["workspaceRoot": root.path, "name": "study", "artifactPath": agent.path]
            func request() throws -> StudyAuthoringHTTP.Response {
                StudyAgentHTTP.perform(.attach, body: try JSONSerialization.data(withJSONObject: fields), workspaceRoot: root)
            }
            #expect(try request().status == "428 Precondition Required")
            fields["manifestFileSHA256"] = study.file.sha256
            #expect(try request().status == "428 Precondition Required")
            fields["artifactFileSHA256"] = agent.file.sha256
            #expect(try request().status == "200 OK")
            let saved = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            #expect(try request().status == "412 Precondition Failed")
            fields["manifestFileSHA256"] = saved.file.sha256
            var changed = agent.record.artifact
            changed.systemPrompt = "Other instruction"
            try JSONEncoder().encode(changed).write(to: agent.record.url)
            let refused = try request()
            #expect(refused.status == "409 Conflict")
            #expect(String(decoding: refused.body, as: UTF8.self).contains("artifactPin"))
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "study").file.data == saved.file.data)
        }
    }

    @Test func reviewedSnapshotCannotSilentlyFollowArtifactChanges() throws {
        try fixture { study, agent in
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted]
            try encoder.encode(agent.record.artifact).write(to: agent.record.url)
            #expect(throws: ExperimentError.self) { try StudyAgentAuthoring.attach(agent, reviewed: study) }
            #expect(try DraftAuthoringSnapshot(workspaceRoot: study.workspaceRoot, name: "study").file.data == study.file.data)
        }
    }

    @Test func frozenStudyWrongModelAndUnsafePathsRefuse() throws {
        try fixture { study, agent in
            let root = study.workspaceRoot
            #expect(throws: ExperimentError.self) { try StudyAgentAuthoring.attach(agent, reviewed: study, baseModelChoice: "other/model") }
            for path in ["../outside.json", "/outside.json", "experiments/study/experiment.json", "runs/../outside.json"] {
                #expect(throws: ExperimentError.self) { try AgentArtifactSnapshot(workspaceRoot: root, path: path) }
            }
            let linked = root.appending(components: "runs", "linked.json")
            try FileManager.default.createSymbolicLink(at: linked, withDestinationURL: agent.record.url)
            #expect(throws: ExperimentError.self) { try AgentArtifactSnapshot(workspaceRoot: root, path: "runs/linked.json") }
            var frozen = study.manifest
            frozen.status = .frozen // Disposable fixture represents completed freeze.
            let file = ExperimentRepository(workspaceRoot: root).manifestURL("study")
            let bytes = try JSONEncoder().encode(frozen)
            try bytes.write(to: file)
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            #expect(throws: ExperimentError.self) { try StudyAgentAuthoring.attach(agent, reviewed: reviewed) }
            #expect(try Data(contentsOf: file) == bytes)
        }
    }

    @Test func capturedRootDoesNotFollowTheGlobalWorkspace() throws {
        try fixture { study, agent in
            let otherRoot = study.workspaceRoot.appending(component: "other")
            WorkspaceRoot.programmaticOverride = otherRoot
            let saved = try StudyAgentAuthoring.attach(agent, reviewed: study)
            #expect(saved.workspaceRoot == study.workspaceRoot)
            #expect(saved.manifest.variantConditions.first?.artifactPath == agent.path)
            #expect(!FileManager.default.fileExists(atPath: otherRoot.path))
        }
    }
}
