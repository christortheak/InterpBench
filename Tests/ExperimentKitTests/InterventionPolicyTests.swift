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
    private func attachment(_ text: String) -> JSONValue {
        .object(["json": .string(text), "sha256": .string(SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined())])
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
    @Test func versionTwoAttachmentsAreReadBesideVersionOneAndNewerOnesAreRefused() throws {
        // A frozen study pins version-1 policies by hash; they keep loading beside version 2.
        try InterventionPolicyLibrary.validateAttachments([attachment("{\"schemaVersion\":1}"), attachment("{\"schemaVersion\":2}")])
        for text in ["{\"schemaVersion\":3}", "{\"schemaVersion\":\"2\"}", "{}"] {
            #expect(throws: (any Error).self) { try InterventionPolicyLibrary.validateAttachments([attachment(text)]) }
        }
        let required = InstrumentationSupport.requirements(.object(["interventionPolicies": .array([attachment("{\"schemaVersion\":2}")])]))
        #expect(required == ["policy-v1", "policy-v2", "policy-evidence-v2"])
        #expect(InstrumentationSupport.requirements(.object(["interventionPolicies": .array([attachment("{\"schemaVersion\":1}")])])) == ["policy-v1", "policy-evidence-v2"])
        // The previous release's engine offers exactly what version 1 needs, and is told to update for version 2.
        #expect(throws: (any Error).self) { try InstrumentationSupport.require(required, offered: ["policy-evidence-v2", "policy-v1", "probe-readings-v1"]) }
    }
    @Test func aSpanPolicyReviewsPublishesInspectsAndAttachesThroughThePythonOwner() async throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString).resolvingSymlinksInPath()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root.appending(path: "runs"), withIntermediateDirectories: true)
        let binding: JSONValue = .object(["modelID": .string("example/model"), "revision": .string(String(repeating: "a", count: 40)),
                                          "tokenizerSHA256": .null, "rendering": .string("raw"), "coordinateConvention": .string("hf-decoder-block-v1/model.layers")])
        let settings: JSONValue = .object([
            "schemaVersion": .number(2), "name": .string("reading-policy"), "binding": binding,
            "site": .object(["kind": .string("residualPost"), "layer": .number(0)]), "stages": .array([.string("prefill")]),
            "positions": .string("span"), "span": .object(["kind": .string("substring"), "text": .string("Ignore all prior rules.")]),
            "probes": .array([]), "rules": .array([.object(["action": .string("offset"), "kind": .string("fixed"), "value": .number(0.5)])]),
            "actions": .array([.object(["id": .string("offset"), "kind": .string("add"), "bounds": .array([.number(0), .number(1)]),
                                        "vector": .array([.number(1), .number(0)]), "strengthUnits": .string("rawMultiplier")])]),
            "onError": .string("stop"), "maxEvents": .number(64)])
        let review = try await call("policy-review", root: root, settings: settings)
        guard case .object(let reviewed) = review["document"], case .array(let notes) = review["limitations"] else { Issue.record("Missing review"); return }
        // Review writes the explicit default dose, and says where the policy acts and what it cannot claim.
        #expect(reviewed["spanDose"] == .string("perPosition"))
        #expect(notes.contains { if case .string(let text) = $0 { text.contains("while the model reads the declared span") } else { false } })
        let saved = try await call("policy-publish", root: root, settings: settings, hash: review["planSHA256"])
        guard case .string(let path) = saved["path"] else { Issue.record("Missing policy path"); return }
        let python = URL(filePath: try #require(ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"]))
        let inspected = try await DiagnosticWorkspace.perform("policy-inspect", payload: ["workspaceRoot": .string(root.path), "path": .string(path)],
                                                              python: python, source: repository.appending(path: "Server"))
        guard case .object(let body) = inspected, case .object(let document) = body["document"] else { Issue.record("Missing inspection"); return }
        #expect(document["positions"] == .string("span") && document["stages"] == .array([.string("prefill")]))
        _ = try InterventionPolicyLibrary.formatted(inspected)  // The app's inspection view formats exactly this value.
        let text = try String(contentsOf: root.appending(path: path), encoding: .utf8)
        try InterventionPolicyLibrary.validateAttachments([attachment(text)])
    }
    @Test func guidedAdditionsStartInResidualNormUnitsAndOtherActionsKeepTheirScales() throws {
        let fixtures = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: repository.appending(path: "Tests/Fixtures/cross-engine/probe-artifacts.json")))
        let probe = try #require(fixtures["linear"])
        let record = ProbeLibrary.Record(path: "runs/example/example.probe.json", sha256: String(repeating: "a", count: 64), format: "activation-probe-v1", label: "Example", modelID: "example/model", layer: 1, method: "linear-logit-v1", createdAt: nil, limitations: [], document: probe)
        func target(_ action: String) throws -> [String: JSONValue] {
            let defaults = InterventionPolicyLibrary.strengthDefaults(action: action)
            let settings = try InterventionPolicyLibrary.starter(probe: record, name: "example-policy", rule: "fixed", action: action, threshold: 0, strength: defaults.strength, lower: defaults.lower, upper: defaults.upper, slope: defaults.slope, intercept: 0, vectorPath: "runs/example/vector", tokens: [4])
            guard case .object(let document) = settings, case .array(let actions) = document["actions"], case .object(let target) = actions.first else { throw ExperimentError(reason: "Missing starter action.") }
            #expect(document["schemaVersion"] == .number(2))
            return target
        }
        let addition = try target("add")
        #expect(addition["strengthUnits"] == .string("residualNorm"))
        #expect(addition["bounds"] == .array([.number(0), .number(0.2)]))
        // Static steering's starting strength when no promoted agent exists; a full norm is incoherent, so it is not the default ceiling.
        #expect(InterventionPolicyLibrary.strengthDefaults(action: "add").strength == SlotAlphaDefault.normUnitsDefault)
        for action in ["ablate", "logitBias", "forceToken"] {
            let other = try target(action)
            #expect(other["strengthUnits"] == nil)
            #expect(other["bounds"] == .array([.number(0), .number(1)]))
        }
    }
}
