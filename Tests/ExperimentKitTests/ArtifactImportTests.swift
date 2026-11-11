import Foundation
import Testing
@testable import ExperimentKit

@Suite(.serialized) struct ArtifactImportTests {
    /// The dialog's last line is true whether the lens was imported or found.
    @Test func theDialogNamesALensTheWorkspaceAlreadyHolds() {
        let imported: [String: JSONValue] = [
            "kind": .string("jlens"), "lensID": .string("custom-lens-example"), "changed": .bool(true),
        ]
        #expect(ArtifactImportSelection.completionStatus(
            result: imported, workspaceRoot: "/workspace", modelID: "example/model", isLens: true)
            == "Imported into /workspace. Refresh the J-lens library for example/model and select the new lens.")
        let found = imported.merging(["existing": .bool(true), "changed": .bool(false)]) { _, new in new }
        #expect(ArtifactImportSelection.completionStatus(
            result: found, workspaceRoot: "/workspace", modelID: "example/model", isLens: true)
            == "This workspace already holds these files as lens custom-lens-example, so nothing was "
            + "imported. Refresh the lenses for example/model and select that lens.")
        // Its converted file was missing, and the import put it back.
        let restored = found.merging(["kind": .string("jlens"), "restored": .bool(true), "changed": .bool(true)]) { _, new in new }
        #expect(ArtifactImportSelection.completionStatus(
            result: restored, workspaceRoot: "/workspace", modelID: "example/model", isLens: true)
            == "This workspace already holds these files as lens custom-lens-example; its converted file was "
            + "missing and was put back. Refresh the lenses for example/model and select that lens.")
        // An SAE decoder import is always new, and a job with no result reads as before.
        for result in [found, nil] as [[String: JSONValue]?] {
            #expect(ArtifactImportSelection.completionStatus(
                result: result, workspaceRoot: "/workspace", modelID: "example/model", isLens: false)
                == "Imported into /workspace. Refresh Data → Concepts & Vectors or Playground → "
                + "Steering Vectors for example/model.")
        }
        #expect(ArtifactImportSelection.completionStatus(
            result: nil, workspaceRoot: "/workspace", modelID: "example/model", isLens: true)
            .hasSuffix("select the new lens."))
    }

    /// A failed import's job error is the refusal's sentence, then its
    /// repair, behind the name of the error's class (the engine's test:
    /// `test_a_refused_imports_job_error_is_its_sentence_then_its_repair`).
    /// The dialog shows the sentence and the repair, and not that name.
    @Test func theDialogShowsARefusalWithItsRepairAndWithoutAClassName() {
        let sentence = "This workspace already holds these files as lens custom-lens-example, but its converted "
            + "file (runs/jlens-lenses/custom-lens-example/jacobians.safetensors) differs from the SHA-256 its "
            + "record names. Nothing was imported, and that file was left as it is."
        let repair = "Move that file aside, then import again: the import puts back the file the record names."
        #expect(ArtifactImportSelection.failureStatus(
            jobStatus: "failed", error: "ImportRefusal: \(sentence) \(repair)")
            == "Import failed: \(sentence) \(repair)")
        // Only that name, and only in front: any other error reads as before.
        #expect(ArtifactImportSelection.failureStatus(jobStatus: "failed", error: "RuntimeError: out of memory")
            == "Import failed: RuntimeError: out of memory")
        #expect(ArtifactImportSelection.failureStatus(
            jobStatus: "failed", error: "ValueError: an ImportRefusal: was raised")
            == "Import failed: ValueError: an ImportRefusal: was raised")
        #expect(ArtifactImportSelection.failureStatus(jobStatus: "cancelled", error: nil)
            == "Import cancelled: See the job log.")
    }

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
                                  "layerCount": 4, "tensorFile": "weights.safetensors", "lens": ["tier": "evidence", "targetLayer": 3, "layers": ["1": "map"]]]
        try JSONSerialization.data(withJSONObject: spec).write(to: description)
        let selection = try ArtifactImportSelection.read(description, expectedKind: "jlens")
        #expect(selection.files.count == 2)
        #expect(selection.modelID == "example/model")
        #expect(throws: (any Error).self) { try ArtifactImportSelection.read(description, expectedKind: "sae-decoder") }
        let payload: [String: JSONValue] = ["workspaceRoot": .string(root.path), "descriptionFile": .string(description.path)]
        let review = try await DiagnosticWorkspace.perform("artifact-plan", payload: payload, python: python, source: repository.appending(component: "Server"))
        guard case .object(let object) = review else { Issue.record("Missing plan"); return }
        guard case .object(let details) = object["details"] else { Issue.record("Missing details"); return }
        #expect(details["tier"] == .string("evidence"))
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
        #expect(record.tier == "evidence")
        #expect(record.tierSource == "custom-artifact")
        #expect(record.intendedUse(catalogTier: "testing") == "evidence")
        var testing = record
        testing.tier = "testing"
        #expect(testing.intendedUse(catalogTier: "evidence") == "testing")
        testing.tierSource = "declared"
        #expect(testing.intendedUse(catalogTier: "evidence") == "evidence")
        #expect(try JSONDecoder().decode(JLensRecord.self, from: JSONEncoder().encode(record)) == record)
        #expect(try Data(contentsOf: URL(filePath: directory + "/source/tensorFile.safetensors")) == tensor)
        try Data("changed".utf8).write(to: description)
        await #expect(throws: (any Error).self) {
            _ = try await DiagnosticWorkspace.perform("artifact-import", payload: payload.merging(["planSHA256": hash]) { _, new in new }, python: python, source: repository.appending(component: "Server"))
        }
    }
}
