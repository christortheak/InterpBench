import Foundation
import Testing
@testable import ExperimentKit

/// Exercise both implementations against real interchange bytes. This tests
/// authoring compatibility; it makes no numerical/GPU parity claim.
@MainActor @Suite(.serialized)
struct PythonStudyAssemblyParityTests {
    private var repository: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
    }

    private func pythonRoundTrip(_ data: Data, root: URL) throws -> Data {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let input = root.appending(component: "incoming.json")
        try data.write(to: input)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        let localPython = repository.appending(path: "Server/.venv.nosync/bin/python").path
        let configured = ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"]
        let python = configured ?? (FileManager.default.isExecutableFile(atPath: localPython) ? localPython : "python3")
        process.arguments = [python, "-c", """
        import json, sys
        from pathlib import Path
        from steerlab_server.client import study_packs
        root = Path(sys.argv[1]).resolve()
        data = Path(sys.argv[2]).read_bytes()
        review = study_packs.preview(data, root=root)
        result = study_packs.apply(data, root=root, expected=review['reviewSHA256'])
        print(json.dumps(study_packs.export(result['study']['name'], root=root)['pack']))
        """, root.path, input.path]
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONPATH"] = repository.appending(component: "Server").path
        environment["HF_HUB_OFFLINE"] = "1"
        process.environment = environment
        let output = Pipe()
        let errors = Pipe()
        process.standardOutput = output
        process.standardError = errors
        try process.run()
        let bytes = output.fileHandleForReading.readDataToEndOfFile()
        let diagnostics = errors.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ExperimentError(reason: String(decoding: diagnostics, as: UTF8.self)
                + "\nUse STEERLAB_TEST_PYTHON (TEST_RUNNER_STEERLAB_TEST_PYTHON for xcodebuild) to select a Python environment with the client dependencies installed.")
        }
        return bytes
    }

    @Test func pythonAndMacRoundTripTheSamePackAndPinnedPromptBytes() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "python-pack-parity") { root in
            // The production exporter resolves its dependency census through
            // the active workspace; set it as well as the store's test seam.
            let previousWorkspace = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previousWorkspace }
            let fixture = try Data(contentsOf: repository.appending(path:
                "Server/tests/fixtures/study-assembly/pack.json"))
            let python = try pythonRoundTrip(fixture, root: root.appending(component: "python"))
            let preview = try StudyPackAuthoring.preview(python, workspaceRoot: root)
            let imported = try StudyPackAuthoring.apply(python, workspaceRoot: root,
                expectedReviewSHA256: preview.reviewSHA256)
            let manifest = imported.study.manifest
            #expect(manifest.status == .draft)
            #expect(manifest.freezeHash == nil)
            #expect(manifest.conditions.map(\.name) == ["baseline"])
            let prompt = root.appending(path: "prompts/tasks/shared.jsonl")
            let bytes = try Data(contentsOf: prompt)
            #expect(manifest.taskPromptsHash == ManifestFileTransaction.digest(bytes))
            let record = try JSONSerialization.jsonObject(with: bytes) as? [String: Any]
            #expect((record?["customMetadata"] as? [String: Bool])?["keep"] == true)
            let exported = try StudyPackAuthoring.export(reviewed: imported.study)
            #expect(!exported.externalDependencies.contains("prompts/tasks/shared.jsonl"))
            let returned = try pythonRoundTrip(exported.data, root: root.appending(component: "returned"))
            let raw = try #require(JSONSerialization.jsonObject(with: returned) as? [String: Any])
            let study = try #require(raw["study"] as? [String: Any])
            #expect(study["taskPromptsHash"] as? String == manifest.taskPromptsHash)
            #expect((raw["files"] as? [String: String])?["prompts/tasks/shared.jsonl"] == String(decoding: bytes, as: UTF8.self))
        }
    }
}
