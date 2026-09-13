import CryptoKit
import Foundation
import Testing
@testable import ExperimentKit

@Suite(.serialized) struct InterventionPolicyTests {
    private var repository: URL { URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func call(_ action: String, root: URL, settings: JSONValue, hash: JSONValue? = nil) async throws -> [String: JSONValue] {
        let python = URL(filePath: try #require(ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"]))
        var payload: [String: JSONValue] = ["workspaceRoot": .string(root.path), "settingsText": .string(try InterventionPolicyLibrary.formatted(settings))]
        payload["planSHA256"] = hash
        let result = try await DiagnosticWorkspace.perform(action, payload: payload, python: python, source: repository.appending(path: "Server"))
        guard case .object(let body) = result else { throw ExperimentError(reason: "Missing policy result.") }; return body
    }
    @Test func guidedPolicyPublishesAndNewAgentRoundTripsExactPythonBytes() async throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString).resolvingSymlinksInPath()
        defer { try? FileManager.default.removeItem(at: root) }
        let run = root.appending(path: "runs/example")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        let fixtures = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: repository.appending(path: "Tests/Fixtures/cross-engine/probe-artifacts.json")))
        let probe = try #require(fixtures["linear"])
        let probePath = "runs/example/example.probe.json"
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(probe); try bytes.write(to: root.appending(path: probePath))
        let record = ProbeLibrary.Record(path: probePath, sha256: SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined(), format: "activation-probe-v1", label: "Example", modelID: "example/model", layer: 1, method: "linear-logit-v1", createdAt: nil, limitations: [], document: probe)
        let settings = try InterventionPolicyLibrary.starter(probe: record, name: "example-policy", rule: "threshold", action: "forceToken", threshold: 0, strength: 1, lower: 0, upper: 1, slope: 1, intercept: 0, vectorPath: "", tokens: [4])
        let review = try await call("policy-review", root: root, settings: settings)
        let saved = try await call("policy-publish", root: root, settings: settings, hash: review["planSHA256"])
        let policyPath = try #require(saved["path"])
        let agent = ModelVariantArtifact(name: "example", baseModelID: "example/model", baseRevision: String(repeating: "a", count: 40), promptMode: "rawCompletion", qwenThinkingEnabled: false, temperature: 0, systemPrompt: "")
        let originalBytes = try encoder.encode(agent)
        #expect(!String(decoding: originalBytes, as: UTF8.self).contains("interventionPolicies"))
        let source = run.appending(path: "agent.json"); try originalBytes.write(to: source)
        let attachment: JSONValue = .object(["agentPath": .string("runs/example/agent.json"), "name": .string("modified"), "policyPaths": .array([policyPath])])
        let proposed = try await call("policy-attach-review", root: root, settings: attachment)
        let result = try await call("policy-attach", root: root, settings: attachment, hash: proposed["planSHA256"])
        guard case .string(let path) = result["path"] else { Issue.record("Missing agent path"); return }
        let newBytes = try Data(contentsOf: root.appending(path: path))
        let modified = try JSONDecoder().decode(ModelVariantArtifact.self, from: newBytes)
        let roundTrip = try JSONDecoder().decode(ModelVariantArtifact.self, from: encoder.encode(modified))
        #expect(modified.interventionPolicies == roundTrip.interventionPolicies)
        #expect(modified.interventionPolicies?.count == 1)
        #expect(throws: (any Error).self) { try InterventionPolicyLibrary.requireNativeExecution(modified) }
        #expect(try Data(contentsOf: source) == originalBytes)
        let snapshot = try AgentArtifactSnapshot(workspaceRoot: root, path: path)
        #expect(snapshot.record.artifact.interventionPolicies == modified.interventionPolicies)
    }
    @Test func corruptedAndDuplicatedAttachmentsFailButAbsentDoesNotChangeEncoding() throws {
        try InterventionPolicyLibrary.validateAttachments(nil)
        let text = "{\"schemaVersion\":1}"
        let sha = SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
        let good: JSONValue = .object(["json": .string(text), "sha256": .string(sha)])
        try InterventionPolicyLibrary.validateAttachments([good])
        #expect(throws: (any Error).self) { try InterventionPolicyLibrary.validateAttachments([good, good]) }
        #expect(throws: (any Error).self) { try InterventionPolicyLibrary.validateAttachments([.object(["json": .string(text + " "), "sha256": .string(sha)])]) }
    }
}
