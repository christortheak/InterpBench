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
    @Test func normUnitEvidenceStatesUnitsAndScalesAndTamperedScalesDoNot() throws {
        let raw = try Data(contentsOf: repository.appending(path: "Tests/Fixtures/cross-engine/policy-evidence-norm-units.json"))
        try InstrumentationEvidence.validate(try JSONDecoder().decode(JSONValue.self, from: raw))
        let document = try #require(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        let block = try #require(document["interventionDecisions"] as? [String: Any])
        let row = try #require((block["decisions"] as? [[String: Any]])?.first)
        #expect(row["strengthUnits"] as? [String: String] == ["change": "residualNorm"])
        // The requested strength is float32(0.1); the absolute scale is 0.1 × residual norm 10 / vector norm 5.
        let scale = try #require((row["appliedScales"] as? [String: Double])?["change"])
        #expect(abs(scale - 0.2) < 1e-6)
        func tampered(_ field: String, _ value: Any) throws -> JSONValue {
            var changed = row; changed[field] = value
            var copy = block; copy["decisions"] = [changed]
            var outer = document; outer["interventionDecisions"] = copy
            return try JSONDecoder().decode(JSONValue.self, from: JSONSerialization.data(withJSONObject: outer))
        }
        for (field, value) in [("appliedScales", ["other": 1.0] as Any), ("strengthUnits", ["other": "residualNorm"] as Any), ("appliedScales", "0.2" as Any)] {
            let corrupt = try tampered(field, value)
            #expect(throws: (any Error).self) { try InstrumentationEvidence.validate(corrupt) }
        }
        // Rows recorded before units and scales existed remain admissible.
        var older = row; older.removeValue(forKey: "appliedScales"); older.removeValue(forKey: "strengthUnits")
        var copy = block; copy["decisions"] = [older]
        var outer = document; outer["interventionDecisions"] = copy
        try InstrumentationEvidence.validate(try JSONDecoder().decode(JSONValue.self, from: JSONSerialization.data(withJSONObject: outer)))
    }
    @Test func aStrengthAtAFractionalBoundIsAdmittedAtFloat32Precision() throws {
        let raw = try Data(contentsOf: repository.appending(path: "Tests/Fixtures/cross-engine/policy-evidence-norm-units.json"))
        var document = try #require(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        var block = try #require(document["interventionDecisions"] as? [String: Any])
        var rows = try #require(block["decisions"] as? [[String: Any]])
        // Execution clamps to float32(0.2), a little above the declared double 0.2.
        let bound = Double(Float(0.2))
        rows[0]["strengths"] = ["change": bound]; rows[0]["appliedStrengths"] = ["change": bound]
        block["decisions"] = rows; document["interventionDecisions"] = block
        try InstrumentationEvidence.validate(try JSONDecoder().decode(JSONValue.self, from: JSONSerialization.data(withJSONObject: document)))
        rows[0]["strengths"] = ["change": 0.21]; rows[0]["appliedStrengths"] = ["change": 0.21]
        block["decisions"] = rows; document["interventionDecisions"] = block
        let outside = try JSONDecoder().decode(JSONValue.self, from: JSONSerialization.data(withJSONObject: document))
        #expect(throws: (any Error).self) { try InstrumentationEvidence.validate(outside) }
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
