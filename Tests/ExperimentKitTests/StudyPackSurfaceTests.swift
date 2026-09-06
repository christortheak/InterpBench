import Foundation
import Testing
@testable import ExperimentKit

@MainActor @Suite(.serialized)
struct StudyPackSurfaceTests {
    private func pack(name: String = "study", text: String = "A neutral task", path: String = "prompts/tasks/input.jsonl") throws -> Data {
        var manifest = ExperimentManifest(name: name, description: "A declared task", modelID: "test/model")
        manifest.taskPromptsFile = path
        manifest.conditions = [.init(name: "baseline", slots: [], bandWidth: 1, alphaInNormUnits: false)]
        let study = try JSONSerialization.jsonObject(with: JSONEncoder().encode(manifest))
        let records = String(decoding: try JSONSerialization.data(withJSONObject: ["id": "item", "text": text]), as: UTF8.self) + "\n"
        return try JSONSerialization.data(withJSONObject: ["study": study, "files": [path: records]], options: [.sortedKeys])
    }

    @Test func previewHasNoFilesystemEffectsAndApplyPinsActualBytes() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "pack-preview") { root in
            let data = try pack()
            let preview = try StudyPackAuthoring.preview(data, workspaceRoot: root)
            #expect(preview.files.map(\.disposition) == ["create"])
            #expect(!FileManager.default.fileExists(atPath: root.appending(component: "prompts").path))
            #expect(!FileManager.default.fileExists(atPath: root.appending(component: "experiments").path))
            let result = try StudyPackAuthoring.apply(data, workspaceRoot: root, expectedReviewSHA256: preview.reviewSHA256)
            let bytes = try Data(contentsOf: root.appending(path: "prompts/tasks/input.jsonl"))
            #expect(result.study.manifest.status == .draft)
            #expect(result.study.manifest.taskPromptsHash == ManifestFileTransaction.digest(bytes))
            #expect(result.filesWritten == ["prompts/tasks/input.jsonl"])
            #expect(result.violations == ExperimentStore.verify(result.study.manifest))
            #expect(throws: ExperimentError.self) {
                try StudyPackAuthoring.apply(data, workspaceRoot: root, expectedReviewSHA256: preview.reviewSHA256)
            }
        }
    }

    @Test(arguments: ["pack", "file", "root"])
    func changedPreviewRefusesWithoutCreatingStudy(change: String) throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "pack-changed") { root in
            var data = try pack()
            let preview = try StudyPackAuthoring.preview(data, workspaceRoot: root)
            var destination = root
            if change == "pack" { data = try pack(text: "Another task") }
            if change == "file" {
                let url = root.appending(path: "prompts/tasks/input.jsonl")
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try Data("other writer".utf8).write(to: url)
            }
            if change == "root" { destination = root.appending(component: "other") }
            #expect(throws: ExperimentError.self) {
                try StudyPackAuthoring.apply(data, workspaceRoot: destination, expectedReviewSHA256: preview.reviewSHA256)
            }
            #expect(!FileManager.default.fileExists(atPath: root.appending(path: "experiments/study").path))
        }
    }

    @Test func preexistingIdenticalInputIsReusedAndNeverRolledBack() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "pack-reuse") { root in
            let first = try pack(name: "first")
            let read = try StudyPackAuthoring.preview(first, workspaceRoot: root)
            _ = try StudyPackAuthoring.apply(first, workspaceRoot: root, expectedReviewSHA256: read.reviewSHA256)
            let second = try pack(name: "second")
            let reuse = try StudyPackAuthoring.preview(second, workspaceRoot: root)
            #expect(reuse.files.map(\.disposition) == ["reuse"])
            let result = try StudyPackAuthoring.apply(second, workspaceRoot: root, expectedReviewSHA256: reuse.reviewSHA256)
            #expect(result.filesWritten.isEmpty)
            #expect(result.study.manifest.taskPromptsHash == (try ExperimentStore.load(name: "first")).taskPromptsHash)
        }
    }

    @Test func referencedWorkspaceInputsAlsoBelongToThePreview() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "pack-reference") { root in
            let file = root.appending(path: "prompts/tasks/existing.jsonl")
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(#"{"text":"original"}"#.utf8).write(to: file)
            var manifest = ExperimentManifest(name: "study", description: "", modelID: "test/model")
            manifest.taskPromptsFile = "prompts/tasks/existing.jsonl"
            let data = try JSONEncoder().encode(manifest)
            let preview = try StudyPackAuthoring.preview(data, workspaceRoot: root)
            #expect(preview.referencedInputs.count == 1)
            try Data(#"{"text":"changed"}"#.utf8).write(to: file)
            #expect(throws: ExperimentError.self) {
                try StudyPackAuthoring.apply(data, workspaceRoot: root, expectedReviewSHA256: preview.reviewSHA256)
            }
            #expect(!FileManager.default.fileExists(atPath: ExperimentStore.manifestURL("study").path))
        }
    }

    @Test func uiImportRetainsItsPreviewedWorkspace() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "pack-ui-origin") { root in
            let data = try pack()
            let preview = try StudyPackAuthoring.preview(data, workspaceRoot: root)
            let panel = ExperimentPanel()
            let previous = ExperimentStore.rootOverride
            ExperimentStore.rootOverride = root.appending(component: "other")
            defer { ExperimentStore.rootOverride = previous }
            #expect(!panel.importStudyJSON(String(decoding: data, as: UTF8.self), reviewed: preview))
            #expect(!FileManager.default.fileExists(atPath: root.appending(path: "experiments/study").path))
        }
    }

    private func withWorkspace(_ body: (URL) async throws -> Void) async throws {
        ExperimentRootOverrideLock.acquire()
        let root = FileManager.default.temporaryDirectory.appending(component: "pack-journey-\(UUID())")
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = root
        defer {
            WorkspaceRoot.programmaticOverride = previous
            try? FileManager.default.removeItem(at: root)
            ExperimentRootOverrideLock.release()
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try await body(root)
    }

    @Test func cliPreviewCanBeAppliedThroughHTTPAndContinuedInThePanel() async throws {
        try await withWorkspace { root in
            let input = root.appending(component: "pack.json")
            let bytes = try pack()
            try bytes.write(to: input)
            let runner = ExperimentCLIRunner(sink: .discarding)
            let guide = await runner.run(namespace: "authoring", ["study", "conceptStudy", "--json"])
            #expect(guide.exitCode == 0)
            #expect(guide.envelope.result?["prompt"] == .string(StudyCoauthoring.prompt(for: .conceptStudy)))
            let outcome = await runner.run(namespace: "pack", ["preview", input.path, "--json"])
            #expect(outcome.exitCode == 0)
            #expect(!outcome.envelope.changed)
            guard case .string(let expected) = outcome.envelope.result?["reviewSHA256"] else {
                Issue.record("preview must return its external digest")
                return
            }
            let body = try JSONSerialization.data(withJSONObject: ["workspaceRoot": root.path,
                "text": String(decoding: bytes, as: UTF8.self), "reviewSHA256": expected])
            let response = StudyPackHTTP.perform(.apply, body: body, workspaceRoot: root)
            #expect(response.status == "200 OK")
            let decoded = try #require(JSONSerialization.jsonObject(with: response.body) as? [String: Any])
            #expect(decoded["changed"] as? Bool == true)
            #expect(decoded["verificationIssues"] is [String])
            let panel = ExperimentPanel()
            panel.management.selectedName = "study"
            let review = try panel.management.reviewStudy(named: "study")
            #expect(review.manifest.taskPromptsHash != nil)
            let inputRequest = try JSONSerialization.data(withJSONObject: ["workspaceRoot": root.path,
                "name": "study", "text": #"{"id":"item","text":"Revised task"}"#, "manifestFileSHA256": review.file.sha256])
            #expect(StudyInputHTTP.importPrompts(body: inputRequest, workspaceRoot: root).status == "200 OK")
            panel.refresh()
            #expect(panel.management.selectedDraftNeedsReload)
            let saved = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            let reread = StudyProtocolHTTP.read(name: "study", workspaceRoot: root)
            let document = try #require(JSONSerialization.jsonObject(with: reread.body) as? [String: Any])
            #expect(document["manifestFileSHA256"] as? String == saved.file.sha256)
            #expect(saved.manifest.taskPromptsFile?.hasPrefix("prompts/tasks/versions/") == true)
            let recordFile = root.appending(component: "records.jsonl")
            try Data(#"{"id":"item","text":"Third task"}"#.utf8).write(to: recordFile)
            let inputCLI = await runner.run(namespace: "experiment", ["import-prompts", "study", "--file", recordFile.path,
                "--manifest-sha256", saved.file.sha256, "--json"])
            #expect(inputCLI.exitCode == 0)
            #expect(inputCLI.envelope.changed)
            #expect(inputCLI.envelope.result?["recordCount"] == .number(1))
            let current = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            let repeated = await runner.run(namespace: "experiment", ["import-prompts", "study", "--file", recordFile.path,
                "--manifest-sha256", current.file.sha256, "--json"])
            #expect(repeated.exitCode == 0)
            #expect(!repeated.envelope.changed)
            #expect(repeated.envelope.result?["changed"] == .bool(false))
            #expect(StudyInputHTTP.importPrompts(body: inputRequest, workspaceRoot: root).status == "412 Precondition Failed")
            let exported = await runner.run(namespace: "pack", ["export", "study", "--json"])
            #expect(exported.exitCode == 0)
            #expect(exported.envelope.result?["pack"] != nil)
        }
    }

    @Test func httpRejectsMissingReviewWrongWorkspaceAndUnknownFields() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "pack-http") { root in
            let text = String(decoding: try pack(), as: UTF8.self)
            func send(_ fields: [String: Any]) throws -> String {
                StudyPackHTTP.perform(.apply, body: try JSONSerialization.data(withJSONObject: fields), workspaceRoot: root).status
            }
            #expect(try send(["workspaceRoot": root.path, "text": text]) == "428 Precondition Required")
            #expect(try send(["workspaceRoot": root.path, "text": text, "force": true]) == "400 Bad Request")
            #expect(try send(["workspaceRoot": root.appending(component: "other").path, "text": text,
                "reviewSHA256": String(repeating: "a", count: 64)]) == "412 Precondition Failed")
            #expect(!FileManager.default.fileExists(atPath: ExperimentStore.manifestURL("study").path))
        }
    }
}
