import CryptoKit
import Foundation
import SteeringKit
import Testing
@testable import ExperimentKit

@Suite(.serialized) struct ProbeLibraryTests {
    private var repository: URL {
        URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    private func call(_ action: String, root: URL, path: String? = nil) async throws -> JSONValue {
        let python = try #require(ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"],
                                  "Set TEST_RUNNER_STEERLAB_TEST_PYTHON before Xcode tests.")
        var payload: [String: JSONValue] = ["workspaceRoot": .string(root.path)]
        if let path { payload["path"] = .string(path) }
        return try await DiagnosticWorkspace.perform(action, payload: payload,
            python: URL(filePath: python), source: repository.appending(path: "Server"))
    }

    private func fixtures(root: URL) throws -> [String: Data] {
        let bytes = try Data(contentsOf: repository.appending(path: "Tests/Fixtures/cross-engine/probe-artifacts.json"))
        let docs = try JSONDecoder().decode([String: JSONValue].self, from: bytes)
        let run = root.appending(path: "runs/example")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var saved: [String: Data] = [:]
        for (name, doc) in docs {
            let filename = name == "python" ? "example-probe.json" : name + ".probe.json"
            let path = "runs/example/" + filename
            let data = try encoder.encode(doc)
            try data.write(to: root.appending(path: path))
            saved[path] = data
        }
        return saved
    }

    @Test func nativeAndPythonArtifactsShareOneLibraryWithoutRewritingBytes() async throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let originals = try fixtures(root: root)
        let inventory: ProbeLibrary.Inventory = try ProbeLibrary.decode(await call("probe-list", root: root))
        #expect(inventory.count == 5)
        #expect(inventory.issues.isEmpty)
        #expect(Set(inventory.probes.map(\.format)) == ["activation-probe-v1", "native-reading-probe", "python-reading-probe"])
        for record in inventory.probes {
            #expect(record.document == nil)
            let detail: ProbeLibrary.Record = try ProbeLibrary.decode(await call("probe-inspect", root: root, path: record.path))
            let original = try #require(originals[record.path])
            #expect(detail.sha256 == SHA256.hash(data: original).map { String(format: "%02x", $0) }.joined())
            #expect(detail.document == (try JSONDecoder().decode(JSONValue.self, from: original)))
            #expect(try Data(contentsOf: root.appending(path: record.path)) == original)
            if detail.format == "python-reading-probe" {
                #expect(detail.createdAt == nil)
                #expect(detail.limitations.contains { $0.contains("layer selection") })
            }
        }
        // Existing native consumers still decode/score their unchanged artifact.
        let nativeBytes = try #require(originals["runs/example/native.probe.json"])
        let native = try JSONDecoder().decode(ReadingProbeArtifact.self, from: nativeBytes)
        #expect(try native.score([3, 1]) == -1)
        let minted = ReadingProbeArtifact(modelID: "example/model", concept: "Example reader", layer: 1,
            recipeName: "existing-reading-recipe",
            probe: .init(direction: [1, -1], activationCenter: [0, 0], projectionCenter: 0,
                         projectionScale: 2, orientation: -1, positiveMean: -1, negativeMean: 1),
            createdAt: try #require(ISO8601DateFormatter().date(from: "2026-01-01T00:00:00Z")),
            notes: "Original note retained.")
        #expect(try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(minted))
            == JSONDecoder().decode(JSONValue.self, from: nativeBytes))
        let oldCatalog = ProbeCatalog.scan(runsDirectory: root.appending(path: "runs"))
        #expect(oldCatalog.count == 1)
        #expect(oldCatalog[0].artifact == native)
        let legacy = try #require(inventory.probes.first { $0.format == "native-reading-probe" })
        var fields = try JSONDecoder().decode([String: JSONValue].self, from: JSONEncoder().encode(legacy))
        fields["method"] = .string("linear-logit-v1") // A user-chosen legacy recipe name is not an algorithm stamp.
        let named: ProbeLibrary.Record = try ProbeLibrary.decode(.object(fields))
        #expect(named.methodLabel == "linear-logit-v1")
    }

    @Test func nativeReencodingPreservesTheNewArtifactMeaningAndLargeSeedString() async throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try fixtures(root: root)
        let record: ProbeLibrary.Record = try ProbeLibrary.decode(await call("probe-inspect", root: root, path: "runs/example/linear.probe.json"))
        let doc = try #require(record.document)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let copied = try encoder.encode(doc)
        try copied.write(to: root.appending(path: "runs/example/new-version.probe.json"))
        let returned: ProbeLibrary.Record = try ProbeLibrary.decode(await call("probe-inspect", root: root, path: "runs/example/new-version.probe.json"))
        #expect(returned.document == doc)
        #expect(returned.sha256 != record.sha256) // Pretty-printed source bytes differ.
        guard case .object(let object) = returned.document,
              case .object(let training) = object["training"],
              case .object(let settings) = training["settings"] else { Issue.record("Missing training metadata"); return }
        #expect(settings["seed"] == .string("18446744073709551615"))
    }

    @Test func malformedArtifactsRemainVisibleAndInspectionRejectsOutsidePaths() async throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try fixtures(root: root)
        try Data("{}".utf8).write(to: root.appending(path: "runs/example/bad.probe.json"))
        let inventory: ProbeLibrary.Inventory = try ProbeLibrary.decode(await call("probe-list", root: root))
        #expect(inventory.count == 5)
        #expect(inventory.issues.map(\.path) == ["runs/example/bad.probe.json"])
        await #expect(throws: (any Error).self) {
            try await call("probe-inspect", root: root, path: "../outside.probe.json")
        }
        let body = try JSONEncoder().encode(["workspaceRoot": root.appending(path: "other").path])
        let wrongRoot = await DiagnosticWorkspace.http("probe-list", body: body, root: root)
        #expect(!wrongRoot.succeeded)
    }
}

extension ProbeLibraryTests {
    @Test func sharedTrainingDraftAndRealFittedArtifactReachTheNativeLibrary() async throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let python = try #require(ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"])
        // A real, small CPU fit, not a mocked classifier or a model download.
        let script = """
        import json, pathlib, sys
        from steerlab_server.experiment import probe_training, diagnostic_archives as a
        root=pathlib.Path(sys.argv[1]).resolve(); fixtures=json.loads(pathlib.Path(sys.argv[2]).read_text())
        rows=[{'id':str(i),'group':str(i),'sourceSHA256':a.digest(i),'label':i>1,'activation':[float(i-1.5),0.]} for i in range(4)]
        dataset={'artifactType':'activation-dataset','schemaVersion':1,'input':fixtures['linear']['input'],'rows':rows,'provenance':{'role':'fit'}}
        (root/'fit.json').write_bytes(a.encoded(dataset))
        config={'fitData':{'path':'fit.json','sha256':a.file_hash(root/'fit.json')},'label':'Native round trip','steps':5}
        result=probe_training.train(probe_training.TrainConfig.from_dict(config),root=root,log=lambda _:None)
        print(json.dumps({'path':str(pathlib.Path(result['artifactPath']).relative_to(root))}))
        """
        let process = Process(); let output = Pipe()
        process.executableURL = URL(filePath: python)
        process.arguments = ["-c", script, root.path, repository.appending(path: "Tests/Fixtures/cross-engine/probe-artifacts.json").path]
        process.environment = ProcessInfo.processInfo.environment.merging(["PYTHONPATH": repository.appending(path: "Server").path]) { _, new in new }
        process.standardOutput = output
        try process.run()
        let bytes = output.fileHandleForReading.readDataToEndOfFile(); process.waitUntilExit()
        try #require(process.terminationStatus == 0)
        let result = try JSONDecoder().decode([String: String].self, from: bytes)
        let path = try #require(result["path"])
        let inspected: ProbeLibrary.Record = try ProbeLibrary.decode(await call("probe-inspect", root: root, path: path))
        #expect(inspected.methodLabel == "Linear classifier")
        #expect(inspected.label == "Native round trip")
        #expect(inspected.sha256 == SHA256.hash(data: try Data(contentsOf: root.appending(path: path))).map { String(format: "%02x", $0) }.joined())
        let answers: JSONValue = .object([
            "purpose": .string("Predict labels"), "claim": .string("Readout only"),
            "controls": .string("Constant baselines"), "selection": .string("Fixed settings"),
            "fields": .object(["fitData": .string("fit.json"), "label": .string("Reviewed probe")]), "advanced": .object([:])
        ])
        let draft = try await DiagnosticWorkspace.perform("draft", payload: [
            "workspaceRoot": .string(root.path), "operation": .string("probe-train"),
            "answersText": .string(String(decoding: try JSONEncoder().encode(answers), as: UTF8.self))
        ], python: URL(filePath: python), source: repository.appending(path: "Server"))
        guard case .object(let object) = draft else { Issue.record("Expected shared review"); return }
        #expect(object["operationReview"] != nil)
        let workflows = try ScienceCatalog.workflows().filter { $0.id.hasPrefix("probe-") }
        #expect(Set(workflows.map(\.id)) == ["probe-capture", "probe-train", "probe-evaluate"])
        for workflow in workflows {
            #expect(workflow.fields.contains { $0.kind == "fileRef" && $0.required })
        }
    }
}
