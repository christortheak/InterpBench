import Foundation
import Testing
@testable import ExperimentKit

@Suite(.serialized) struct ProbeMeasurementsTests {
    private var repository: URL { URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    @Test func optionalSettingsPreserveHistoricalEncodingAndUseBothClients() async throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString).resolvingSymlinksInPath()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root.appending(path: "experiments/example"), withIntermediateDirectories: true)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let original = ExperimentManifest(name: "example", description: "", modelID: "example/model")
        let originalBytes = try encoder.encode(original)
        #expect(!String(decoding: originalBytes, as: UTF8.self).contains("probeMeasurements"))
        let manifestURL = root.appending(path: "experiments/example/experiment.json")
        try originalBytes.write(to: manifestURL)
        let run = root.appending(path: "runs/fit")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        let fixtures = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: repository.appending(path: "Tests/Fixtures/cross-engine/probe-artifacts.json")))
        let probe = try #require(fixtures["linear"])
        let bytes = try encoder.encode(probe); let path = "runs/fit/linear.probe.json"
        try bytes.write(to: root.appending(path: path))
        let python = URL(filePath: try #require(ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"]))
        let inspected = try await DiagnosticWorkspace.perform("probe-inspect", payload: ["workspaceRoot": .string(root.path), "path": .string(path)], python: python, source: repository.appending(path: "Server"))
        guard case .object(let detail) = inspected, let sha = detail["sha256"] else { Issue.record("Missing probe digest"); return }
        let config: JSONValue = .object(["schemaVersion": .number(1), "probes": .array([.object(["id": .string("reader"), "probe": .object(["path": .string(path), "sha256": sha]), "conditions": .array([]), "agents": .array([]), "stages": .array([.string("prefill"), .string("decode")]), "recordingStage": .string("postAction")])]), "onError": .string("recordMissing"), "maxReadings": .number(128), "retainActivations": .bool(false), "maxActivationBytes": .number(1024)])
        #expect(ProbeMeasurements.validShape(config))
        #expect(ProbeMeasurements.violations(config, root: root).isEmpty)
        var payload: [String: JSONValue] = ["workspaceRoot": .string(root.path), "experiment": .string("example"), "settingsText": .string(String(decoding: try encoder.encode(config), as: UTF8.self))]
        let review = try await DiagnosticWorkspace.perform("measurements-review", payload: payload, python: python, source: repository.appending(path: "Server"))
        guard case .object(let reviewed) = review, let hash = reviewed["planSHA256"] else { Issue.record("Missing review"); return }
        payload["planSHA256"] = hash
        _ = try await DiagnosticWorkspace.perform("measurements-save", payload: payload, python: python, source: repository.appending(path: "Server"))
        let saved = try JSONDecoder().decode(ExperimentManifest.self, from: Data(contentsOf: manifestURL))
        #expect(saved.probeMeasurements == config)
        let reencoded = try JSONDecoder().decode(ExperimentManifest.self, from: encoder.encode(saved))
        #expect(reencoded.probeMeasurements == config)
        var removed = saved; removed.probeMeasurements = nil
        #expect(try encoder.encode(removed) == originalBytes)
        try (bytes + Data(" ".utf8)).write(to: root.appending(path: path))
        #expect(!ProbeMeasurements.violations(config, root: root).isEmpty)
    }
    @Test func collectedReadingsDisplayWithoutLegacyMarkerMetricsAndKeepSamplesDistinct() throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let run = root.appending(path: "runs/example-run")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        try encoder.encode(ExperimentManifest(name: "example", description: "", modelID: "example/model")).write(to: run.appending(path: "experiment.json"))
        let readings: JSONValue = .object(["status": .string("complete"), "readings": .array([])])
        var lines = ""
        for index in 0..<2 {
            let record: JSONValue = .object(["condition": .string("baseline"), "promptID": .string("item"), "prompt": .string("Example"), "output": .string("Response"), "wordCount": .number(1), "distinct2": .number(0), "sampleIndex": .number(Double(index)), "probeMeasurements": readings])
            lines += String(decoding: try encoder.encode(record), as: UTF8.self) + "\n"
        }
        try lines.write(to: run.appending(path: "generations.jsonl"), atomically: true, encoding: .utf8)
        let repository = StudyResultRepository(workspaceRoot: root)
        let item = try #require(repository.list(experimentName: "example").first)
        let detail = repository.detail(for: item)
        #expect(detail.generations.count == 2)
        #expect(Set(detail.generations.map(\.id)).count == 2)
        #expect(detail.generations.allSatisfy { $0.probeMeasurements == readings })
    }

    @Test func distinctExternalProbesWithTheSameFilenameBothSurviveSnapshot() throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        let prior = ExperimentStore.rootOverride
        ExperimentStore.rootOverride = root
        defer { ExperimentStore.rootOverride = prior; try? FileManager.default.removeItem(at: root) }
        var selections: [JSONValue] = []
        for label in ["a", "b"] {
            let path = "inputs/" + label + "/trained.probe.json"
            let url = root.appending(path: path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try label.write(to: url, atomically: true, encoding: .utf8)
            selections.append(.object(["probe": .object(["path": .string(path), "sha256": .string(String(repeating: label, count: 64))])]))
        }
        var study = ExperimentManifest(name: "example", description: "", modelID: "example/model")
        study.probeMeasurements = .object(["probes": .array(selections)])
        // Snapshot enumeration is independently tested here; authoring and
        // verification validate the complete schema and digests separately.
        try ExperimentStore.snapshotPinnedInputs(for: study)
        for label in ["a", "b"] {
            let filename = "measurement-probe-" + String(repeating: label, count: 64) + "-trained.probe.json"
            #expect(try String(contentsOf: root.appending(path: "experiments/example/pinned/" + filename), encoding: .utf8) == label)
        }
    }

}
