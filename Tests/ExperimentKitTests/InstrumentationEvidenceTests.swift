import Foundation
import Testing
@testable import ExperimentKit

struct InstrumentationEvidenceTests {
    private var repository: URL { URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    @Test func pythonForwardEvidenceAdmitsAndTamperedAcknowledgementsDoNot() throws {
        let file = repository.appending(path: "Tests/Fixtures/cross-engine/policy-evidence.json")
        let raw = try Data(contentsOf: file)
        let value = try JSONDecoder().decode(JSONValue.self, from: raw)
        try InstrumentationEvidence.validate(value)
        #expect(throws: (any Error).self) { try InstrumentationEvidence.validate(.array([])) }
        var document = try #require(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        var block = try #require(document["interventionDecisions"] as? [String: Any])
        var rows = try #require(block["decisions"] as? [[String: Any]])
        rows[0]["actionOutcomes"] = [:]; block["decisions"] = rows; document["interventionDecisions"] = block
        let corrupt = try JSONDecoder().decode(JSONValue.self, from: JSONSerialization.data(withJSONObject: document))
        #expect(throws: (any Error).self) { try InstrumentationEvidence.validate(corrupt) }
        document.removeValue(forKey: "interventionDecisions")
        let missing = try JSONDecoder().decode(JSONValue.self, from: JSONSerialization.data(withJSONObject: document))
        #expect(throws: (any Error).self) { try InstrumentationEvidence.validate(missing) }
    }
    @Test func completeSourceAnalysisUsesThePythonOwner() async throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString).resolvingSymlinksInPath()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root.appending(path: "runs/example"), withIntermediateDirectories: true)
        let raw = try Data(contentsOf: repository.appending(path: "Tests/Fixtures/cross-engine/policy-evidence.json"))
        let value = try JSONDecoder().decode(JSONValue.self, from: raw)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let file = root.appending(path: "runs/example/generations.jsonl")
        try (encoder.encode(value) + Data([10])).write(to: file)
        try InstrumentationEvidence.validateFile(file)
        let python = URL(filePath: try #require(ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"]))
        let report = try await DiagnosticWorkspace.perform("evidence-analyze", payload: ["workspaceRoot": .string(root.path), "path": .string("runs/example")], python: python, source: repository.appending(path: "Server"))
        guard case .object(let object) = report, case .array(let groups) = object["groups"], case .object(let group) = groups.first,
              case .object(let applied) = group["applied"], case .object(let strength) = applied["change"] else { Issue.record("Missing applied summary"); return }
        #expect(object["responses"] == .number(1))
        #expect(strength["mean"] == .number(2))
    }
    @Test func nestedPanelRequirementsAndOldServerAdmission() throws {
        let value: JSONValue = .object(["agents": .array([.object(["artifact": .object(["interventionPolicies": .array([.object([:])])])])]), "probeMeasurements": .object(["probes": .array([.object([:])])])])
        let required = InstrumentationSupport.requirements(value)
        #expect(required == ["policy-v1", "policy-evidence-v2", "probe-readings-v1"])
        #expect(throws: (any Error).self) { try InstrumentationSupport.require(required, offered: nil) }
        #expect(throws: (any Error).self) { try InstrumentationSupport.require(required, offered: ["policy-v1"]) }
        try InstrumentationSupport.require(required, offered: Array(required))
        try InstrumentationSupport.require([], offered: nil)
    }
}
