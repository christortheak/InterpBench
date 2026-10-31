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
    /// The norm-unit fixture with one change made to its only decision row and declaration.
    private func normUnitEvidence(_ change: (inout [String: Any], inout [String: Any]) -> Void) throws -> JSONValue {
        let raw = try Data(contentsOf: repository.appending(path: "Tests/Fixtures/cross-engine/policy-evidence-norm-units.json"))
        var document = try #require(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        var block = try #require(document["interventionDecisions"] as? [String: Any])
        var row = try #require((block["decisions"] as? [[String: Any]])?.first)
        var declaration = try #require((block["declarations"] as? [[String: Any]])?.first)
        change(&row, &declaration)
        block["decisions"] = [row]; block["declarations"] = [declaration]; document["interventionDecisions"] = block
        return try JSONDecoder().decode(JSONValue.self, from: JSONSerialization.data(withJSONObject: document))
    }
    @Test func unitsAndScalesMustBeTheDeclarationsOwnAndAgreeWithPythonBitForBit() throws {
        try InstrumentationEvidence.validate(try normUnitEvidence { _, _ in })
        // The scale Python recorded is Swift's Float arithmetic in vector_math.norm_unit_scale's order.
        let recorded = try normUnitEvidence { row, _ in
            let strength = (row["strengths"] as? [String: Double])?["change"] ?? .nan
            #expect((row["appliedScales"] as? [String: Double])?["change"] == Double(Float(strength) * Float(10) / Float(5)))
        }
        try InstrumentationEvidence.validate(recorded)
        let mutations: [(inout [String: Any], inout [String: Any]) -> Void] = [
            { row, _ in row["strengthUnits"] = ["change": "rawMultiplier"]; row["appliedScales"] = ["change": 999.0] },
            { row, _ in row["strengthUnits"] = ["change": "rawMultiplier"]; row["appliedScales"] = row["strengths"] },
            { row, _ in
                let scale = Float((row["appliedScales"] as? [String: Double])?["change"] ?? 0)
                row["appliedScales"] = ["change": Double(scale.nextUp)] },
            { row, _ in row["appliedScales"] = [String: Double]() },
            { _, declaration in declaration["schemaVersion"] = 1 },
            { _, declaration in declaration["schemaVersion"] = 3 },
            { _, declaration in
                var actions = declaration["actions"] as? [[String: Any]] ?? []
                actions[0].removeValue(forKey: "residualNorm"); declaration["actions"] = actions },
        ]
        for mutation in mutations {
            let corrupt = try normUnitEvidence(mutation)
            #expect(throws: (any Error).self) { try InstrumentationEvidence.validate(corrupt) }
        }
        // Declarations recorded before they stated a schema version remain admissible.
        try InstrumentationEvidence.validate(try normUnitEvidence { _, declaration in declaration.removeValue(forKey: "schemaVersion") })
    }
    @Test func aStrengthAtAFractionalBoundIsAdmittedAtFloat32Precision() throws {
        let raw = try Data(contentsOf: repository.appending(path: "Tests/Fixtures/cross-engine/policy-evidence-norm-units.json"))
        var document = try #require(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        var block = try #require(document["interventionDecisions"] as? [String: Any])
        var rows = try #require(block["decisions"] as? [[String: Any]])
        // Execution clamps to float32(0.2), a little above the declared double 0.2.
        let bound = Double(Float(0.2))
        rows[0]["strengths"] = ["change": bound]; rows[0]["appliedStrengths"] = ["change": bound]
        rows[0]["appliedScales"] = ["change": Double(Float(bound) * Float(10) / Float(5))]
        block["decisions"] = rows; document["interventionDecisions"] = block
        try InstrumentationEvidence.validate(try JSONDecoder().decode(JSONValue.self, from: JSONSerialization.data(withJSONObject: document)))
        rows[0]["strengths"] = ["change": 0.21]; rows[0]["appliedStrengths"] = ["change": 0.21]
        block["decisions"] = rows; document["interventionDecisions"] = block
        let outside = try JSONDecoder().decode(JSONValue.self, from: JSONSerialization.data(withJSONObject: document))
        #expect(throws: (any Error).self) { try InstrumentationEvidence.validate(outside) }
    }
    @Test func spanPolicyEvidenceAdmitsAndTamperedSpansDoNot() throws {
        let raw = try Data(contentsOf: repository.appending(path: "Tests/Fixtures/cross-engine/policy-evidence-span.json"))
        try InstrumentationEvidence.validate(try JSONDecoder().decode(JSONValue.self, from: raw))
        let document = try #require(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        let block = try #require(document["interventionDecisions"] as? [String: Any])
        let span = try #require((block["spans"] as? [[String: Any]])?.first)
        let rows = try #require(block["decisions"] as? [[String: Any]])
        #expect(span["tokenRanges"] as? [[Int]] == [[1, 3]] && span["doseMode"] as? String == "fixedTotal")
        // A fixedTotal dose of 0.1 over two tokens: each applies 0.05 residual norms, a scale of 0.05 × 7.5 / 0.5.
        #expect(abs(((rows[0]["appliedScales"] as? [String: Double])?["change"] ?? 0) - 0.75) < 1e-6)
        let prompt = try #require(block["promptTokenIDs"] as? [Int])
        #expect(span["promptTokensSHA256"] as? String == InstrumentationEvidence.tokenIDsSHA256(prompt.map { .number(Double($0)) }))
        func tampered(_ change: (inout [String: Any], inout [[String: Any]], inout [String: Any]) -> Void) throws -> JSONValue {
            var copy = block; var entry = span; var events = rows
            change(&entry, &events, &copy)
            if copy["spans"] != nil { copy["spans"] = [entry] }
            copy["decisions"] = events
            var outer = document; outer["interventionDecisions"] = copy
            return try JSONDecoder().decode(JSONValue.self, from: JSONSerialization.data(withJSONObject: outer))
        }
        let mutations: [(inout [String: Any], inout [[String: Any]], inout [String: Any]) -> Void] = [
            { entry, _, _ in entry["promptTokensSHA256"] = String(repeating: "a", count: 64) },
            { entry, _, _ in entry["tokenRanges"] = [[1, 9]] },
            { entry, _, _ in entry["tokenCount"] = 3 },
            { entry, _, _ in entry["doseMode"] = "perPosition" },
            { entry, _, _ in
                var actions = entry["actions"] as? [String: [String: Any]] ?? [:]
                actions["change"]?["strengthUnits"] = "rawMultiplier"; entry["actions"] = actions },
            { entry, _, _ in
                var actions = entry["actions"] as? [String: [String: Any]] ?? [:]
                actions["change"]?["summedDose"] = 0.2; entry["actions"] = actions },
            // Both summary keys are required, null when they do not apply; Python refuses their absence too.
            { entry, _, _ in
                var actions = entry["actions"] as? [String: [String: Any]] ?? [:]
                actions["change"]?.removeValue(forKey: "perPositionStrength"); entry["actions"] = actions },
            { entry, _, _ in
                var actions = entry["actions"] as? [String: [String: Any]] ?? [:]
                actions["change"]?.removeValue(forKey: "summedDose"); entry["actions"] = actions },
            { _, events, _ in events[0]["appliedScales"] = ["change": 1.5] },
            { _, events, _ in
                events[0]["inputTokenPosition"] = 0; events[0]["predictsTokenPosition"] = 1
                events[0].removeValue(forKey: "inputTokenID"); events[0].removeValue(forKey: "predictedTokenID") },
            { _, _, copy in copy.removeValue(forKey: "spans") },
        ]
        for mutation in mutations {
            let corrupt = try tampered(mutation)
            #expect(throws: (any Error).self) { try InstrumentationEvidence.validate(corrupt) }
        }
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
