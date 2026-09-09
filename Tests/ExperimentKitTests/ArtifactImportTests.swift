import Foundation
import Testing
@testable import ExperimentKit

@Suite(.serialized) struct ArtifactImportTests {
    @Test func localDescriptionPlansAndPublishesThroughThePortableOwner() async throws {
        let repository = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python = URL(filePath: try #require(ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"]))
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString).resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let header: [String: Any] = ["map": ["dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]]]
        var headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while headerData.count % 8 != 0 { headerData.append(0x20) }
        var length = UInt64(headerData.count).littleEndian
        var tensor = withUnsafeBytes(of: &length) { Data($0) }
        tensor.append(headerData)
        for value in [Float(1), 2, 3, 4] {
            var bits = value.bitPattern.littleEndian
            tensor.append(withUnsafeBytes(of: &bits) { Data($0) })
        }
        let weights = root.appending(component: "weights.safetensors")
        try tensor.write(to: weights)
        let description = root.appending(component: "import.json")
        let spec: [String: Any] = ["schemaVersion": 1, "kind": "jlens", "modelID": "example/model", "hiddenSize": 2,
                                  "layerCount": 4, "tensorFile": "weights.safetensors", "lens": ["targetLayer": 3, "layers": ["1": "map"]]]
        try JSONSerialization.data(withJSONObject: spec).write(to: description)
        let selection = try ArtifactImportSelection.read(description, expectedKind: "jlens")
        #expect(selection.files.count == 2)
        #expect(selection.modelID == "example/model")
        #expect(throws: (any Error).self) { try ArtifactImportSelection.read(description, expectedKind: "sae-decoder") }
        let payload: [String: JSONValue] = ["workspaceRoot": .string(root.path), "descriptionFile": .string(description.path)]
        let review = try await DiagnosticWorkspace.perform("artifact-plan", payload: payload, python: python, source: repository.appending(component: "Server"))
        guard case .object(let object) = review else { Issue.record("Missing plan"); return }
        let hash = try #require(object["planSHA256"])
        let result = try await DiagnosticWorkspace.perform("artifact-import", payload: payload.merging(["planSHA256": hash]) { _, new in new }, python: python, source: repository.appending(component: "Server"))
        guard case .object(let output) = result, case .string(let directory) = output["outputDirectory"] else { Issue.record("Missing output"); return }
        #expect(FileManager.default.fileExists(atPath: directory + "/lens.json"))
        let record = try JSONDecoder().decode(JLensRecord.self, from: Data(contentsOf: URL(filePath: directory + "/lens.json")))
        #expect(record.source?.repo == nil)
        #expect(record.fit?.modelID == "example/model")
        #expect(record.fit?.revisionKnown == false)
        #expect(record.sourceLayers == [1])
        #expect(record.qualifications?.isEmpty == true)
        #expect(try JSONDecoder().decode(JLensRecord.self, from: JSONEncoder().encode(record)) == record)
        #expect(try Data(contentsOf: URL(filePath: directory + "/source/tensorFile.safetensors")) == tensor)
        try Data("changed".utf8).write(to: description)
        await #expect(throws: (any Error).self) {
            _ = try await DiagnosticWorkspace.perform("artifact-import", payload: payload.merging(["planSHA256": hash]) { _, new in new }, python: python, source: repository.appending(component: "Server"))
        }
    }
}
