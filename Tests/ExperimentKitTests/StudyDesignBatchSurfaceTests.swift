import Foundation
import Testing

@testable import ExperimentKit

@Suite(.serialized) struct StudyDesignBatchSurfaceTests {
    private func design(_ root: URL) throws -> StudyDesignSnapshot {
        _ = try ExperimentStore.create(name: "source", description: "Reusable task", modelID: "test/model")
        return try StudyDesignSaving.create(from: StudyDesignSourceReview(study:
            DraftAuthoringSnapshot(workspaceRoot: root, name: "source")), name: "design").snapshot
    }

    private var mixedRows: [[String: Any]] {
        [
            ["casting": ["agents": []], "studyName": "draft"],
            ["casting": ["agents": [["artifactPath": "runs/missing/agent.json", "artifactFileSHA256": String(repeating: "a", count: 64)]]], "studyName": "missing"],
            ["casting": ["unexpected": true], "studyName": "malformed"],
            ["casting": ["agents": []], "studyName": "draft"],
        ]
    }

    @Test func batchReportsEveryRowAndPreservesSuccessfulDraftsAndSource() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "design-batch") { root in
            let reviewed = try design(root)
            let source = try DraftAuthoringSnapshot(workspaceRoot: root, name: "source")
            let batch = try StudyDesignBatchInput(JSONSerialization.data(withJSONObject: ["rows": mixedRows])).mint(reviewed: reviewed)
            #expect(batch.results.map(\.row) == [0, 1, 2, 3])
            #expect(batch.minted == ["draft", "draft-2"])
            #expect(batch.failures.count == 2)
            #expect(batch.results[1].issue?.state == .notFound)
            #expect(batch.results[2].issue?.code == "usage")
            #expect(batch.failures.allSatisfy { $0.issue?.repairAction.isEmpty == false })
            for name in batch.minted {
                let saved = try DraftAuthoringSnapshot(workspaceRoot: root, name: name)
                #expect(saved.manifest.status == .draft)
                #expect(saved.manifest.templateProvenance?.batchGroup == batch.batchGroup)
                #expect(saved.manifest.templateProvenance?.templateHash == StudyTemplateStore.hash(reviewed.template))
            }
            #expect(try StudyDesignSnapshot(workspaceRoot: root, name: "design").file.data == reviewed.file.data)
            #expect(try DraftAuthoringSnapshot(workspaceRoot: root, name: "source").file.data == source.file.data)
            #expect(!FileManager.default.fileExists(atPath: root.appending(path: "experiments/missing").path))
            #expect(!FileManager.default.fileExists(atPath: root.appending(path: "experiments/malformed").path))
        }
    }

    @Test func malformedBatchShapeRefusesBeforeAnyPublication() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "design-batch-shape") { root in
            let reviewed = try design(root)
            let invalid: [[String: Any]] = [
                ["rows": []], ["rows": mixedRows, "submit": true],
                ["rows": [mixedRows[0], ["casting": ["agents": []], "studyName": ""]]],
                ["rows": [mixedRows[0], ["casting": ["agents": []], "force": true]]],
            ]
            for fields in invalid {
                #expect(throws: ExperimentError.self) {
                    try StudyDesignBatchInput(JSONSerialization.data(withJSONObject: fields)).mint(reviewed: reviewed)
                }
            }
            #expect(try FileManager.default.contentsOfDirectory(atPath: root.appending(component: "experiments").path) == ["source"])
        }
    }

    @Test func laterRowsRetainTheReviewedDesignAndTypedRefusal() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "design-batch-review") { root in
            let reviewed = try design(root)
            let batch = StudyDesignInstantiation.mintBatch(reviewed: reviewed, castings: [.agents([]), .agents([])],
                names: ["first", "second"], onRow: { index, _ in
                    if index == 0 { _ = try? StudyDesignAuthoring.updateDescription("Changed between rows", reviewed: reviewed) }
                })
            #expect(batch.minted == ["first"])
            #expect(batch.results[1].issue?.code == "designChanged")
            #expect(batch.results[1].issue?.state == .refused)
            #expect(throws: StudyDesignAuthoringError.self) {
                try StudyDesignBatchInput(JSONSerialization.data(withJSONObject: ["rows": mixedRows])).mint(reviewed: reviewed)
            }
            #expect(!FileManager.default.fileExists(atPath: root.appending(path: "experiments/second").path))
        }
    }

    @Test func httpRequiresWorkspaceAndDesignReviewAndReportsPartialOutcomes() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "design-batch-http") { root in
            let reviewed = try design(root)
            var fields: [String: Any] = ["workspaceRoot": root.path, "name": "design", "rows": mixedRows]
            func send() throws -> StudyAuthoringHTTP.Response {
                StudyDesignHTTP.perform(.batch, body: try JSONSerialization.data(withJSONObject: fields), workspaceRoot: root)
            }
            #expect(try send().status == "428 Precondition Required")
            fields["designFileSHA256"] = reviewed.file.sha256
            fields["workspaceRoot"] = root.appending(component: "other").path
            #expect(try send().status == "409 Conflict")
            fields["workspaceRoot"] = root.path
            fields["submit"] = true
            #expect(try send().status == "400 Bad Request")
            fields.removeValue(forKey: "submit")
            let response = try send()
            #expect(response.status == "207 Multi-Status")
            let result = try JSONDecoder().decode([String: JSONValue].self, from: response.body)
            #expect(result["ok"] == .bool(false))
            #expect(result["changed"] == .bool(true))
            #expect(result["minted"] == .array([.string("draft"), .string("draft-2")]))
            fields["rows"] = [["casting": ["agents": []], "studyName": "clean"]]
            #expect(try send().status == "200 OK")
            _ = try StudyDesignAuthoring.updateDescription("New version", reviewed: reviewed)
            #expect(try send().status == "412 Precondition Failed")
        }
    }

    @Test func cliPartialEnvelopePreservesChangedAndStructuredResults() async throws {
        try await ExperimentCLIEnvelopeTests().withTempRoot { root in
            let reviewed = try design(root)
            let input = root.appending(component: "batch.json")
            try JSONSerialization.data(withJSONObject: ["rows": mixedRows]).write(to: input)
            let outcome = await ExperimentCLIRunner(sink: .discarding).run(namespace: "design",
                ["batch", "design", "--rows", input.path, "--file-sha256", reviewed.file.sha256, "--json"])
            #expect(outcome.exitCode == 65)
            #expect(outcome.envelope.state == .refused)
            #expect(outcome.envelope.changed)
            #expect(outcome.envelope.result?["minted"] == .array([.string("draft"), .string("draft-2")]))
            #expect(outcome.envelope.error?.code == "designBatchIncomplete")
            #expect(outcome.envelope.error?.repairAction.contains("only repaired failed rows") == true)
        }
    }
}
