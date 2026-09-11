import Foundation
import Testing
@testable import ExperimentKit

@Suite(.serialized) struct DiagnosticWorkspaceTests {
    @Test func pythonArchiveImportsThroughMacAndVerifiesOfflineUntilBytesChange() async throws {
        let repository = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python = try #require(ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"], "Set TEST_RUNNER_STEERLAB_TEST_PYTHON before xcodebuild to run the real portable client.")
        let temporary = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let process = Process(); let output = Pipe()
        process.executableURL = URL(filePath: python)
        process.arguments = ["-c", """
        import json,sys
        from pathlib import Path
        from steerlab_server.experiment import diagnostic_archives as a
        root=Path(sys.argv[1]); source=root/'remote'; target=source/'diagnostics/example'
        target.mkdir(parents=True); (target/'report.json').write_text('{"score":0.5}')
        result=a.package(source,['diagnostics/example/report.json'],root/'evidence.tar.gz',kind='diagnosticEvidence',context={'jobID':'example-job','outputRelative':'diagnostics/example'})
        print(json.dumps(result))
        """, temporary.path]
        process.environment = ProcessInfo.processInfo.environment.merging(["PYTHONPATH": repository.appending(component: "Server").path]) { _, new in new }
        process.standardOutput = output
        try process.run()
        let data = output.fileHandleForReading.readDataToEndOfFile(); process.waitUntilExit()
        #expect(process.terminationStatus == 0)
        let exported = try JSONDecoder().decode([String: JSONValue].self, from: data)
        let digest = try #require(exported["bundleSha256"])
        let local = temporary.appending(component: "local")
        try FileManager.default.createDirectory(at: local, withIntermediateDirectories: false)
        let imported = try await DiagnosticWorkspace.perform("import", payload: ["workspaceRoot": .string(local.path), "archivePath": .string(temporary.appending(component: "evidence.tar.gz").path), "archiveSHA256": digest], python: URL(filePath: python), source: repository.appending(component: "Server"))
        guard case .object(let fields) = imported else { Issue.record("Missing receipt"); return }
        let receipt = try #require(fields["receiptSHA256"])
        let payload: [String: JSONValue] = ["workspaceRoot": .string(local.path), "receiptSHA256": receipt]
        let verified = try await DiagnosticWorkspace.perform("verify-custody", payload: payload, python: URL(filePath: python), source: repository.appending(component: "Server"))
        guard case .object(let check) = verified else { Issue.record("Missing verification"); return }
        #expect(check["verified"] == .bool(true))
        try Data("changed".utf8).write(to: local.appending(path: "diagnostics/example/report.json"))
        await #expect(throws: (any Error).self) {
            try await DiagnosticWorkspace.perform("verify-custody", payload: payload, python: URL(filePath: python), source: repository.appending(component: "Server"))
        }
    }
}

@Suite(.serialized) struct FittingCorpusWorkspaceTests {
    @Test func macPreparesAndPublishesThroughThePortableOwner() async throws {
        let repository = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python = try #require(ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"])
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("A complete existing source document.\nA second line.".utf8).write(to: root.appending(component: "source.txt"))
        let spec = #"{"source":{"kind":"local","files":["source.txt"]},"count":1,"selection":"first"}"#
        let preview = try await DiagnosticWorkspace.perform("corpus-preview", payload: ["workspaceRoot": .string(root.path), "specText": .string(spec)], python: URL(filePath: python), source: repository.appending(component: "Server"))
        guard case .object(let fields) = preview else { Issue.record("Missing corpus preview"); return }
        let identifier = try #require(fields["previewID"])
        let hash = try #require(fields["planSHA256"])
        let saved = try await DiagnosticWorkspace.perform("corpus-publish", payload: ["workspaceRoot": .string(root.path), "previewID": identifier, "planSHA256": hash, "destination": .string("prompts/fitting/example")], python: URL(filePath: python), source: repository.appending(component: "Server"))
        guard case .object(let result) = saved else { Issue.record("Missing corpus publication"); return }
        #expect(result["changed"] == .bool(true))
        let bytes = try Data(contentsOf: root.appending(path: "prompts/fitting/example/corpus.jsonl"))
        let row = try JSONDecoder().decode([String: String].self, from: bytes)
        #expect(row["text"] == "A complete existing source document.\nA second line.")
        await #expect(throws: (any Error).self) {
            try await DiagnosticWorkspace.perform("corpus-publish", payload: ["workspaceRoot": .string(root.path), "previewID": identifier, "planSHA256": hash, "destination": .string("prompts/fitting/example")], python: URL(filePath: python), source: repository.appending(component: "Server"))
        }
        #expect(ExperimentCLIParser.spec(namespace: "science", verb: "corpus-preview") != nil)
        #expect(ExperimentCLIParser.spec(namespace: "science", verb: "corpus-publish")?.requiredFlags == ["--plan-sha256", "--destination"])
    }
}


@Suite(.serialized) struct PilotOperationsTests {
    @Test func stagedRequestUsesThePortableOwner() async throws {
        let repository = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python = try #require(ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"])
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let digest = String(repeating: "a", count: 64)
        let result = try await DiagnosticWorkspace.perform("staged-request", payload: ["workspaceRoot": .string(root.path), "bundleSHA256": .string(digest)], python: URL(filePath: python), source: repository.appending(component: "Server"))
        guard case .object(let fields) = result, case .string(let path) = fields["localRequestPath"] else {
            Issue.record("Missing local staged request"); return
        }
        let decoded = try JSONDecoder().decode([String: String].self, from: Data(contentsOf: URL(filePath: path)))
        #expect(decoded == ["inputBundleSHA256": digest])
    }

    @Test func controllerBuildDriftIsAdvisoryAndVisible() {
        var observed = ClusterObservedState(siteID: "fixture", siteName: "Fixture")
        observed.serverHTTP = .reachable(build: "steerlab-server 0.9+aaaaaaaa", role: "controller", root: "/runs")
        observed.deployedControllerBuild = "bbbbbbbb"
        #expect(observed.controllerBuildSummary.contains("aaaaaaaa"))
        #expect(observed.controllerBuildSummary.contains("bbbbbbbb"))
        #expect(observed.advisories.contains { $0.contains("reviewed restart") })
        observed.deployedControllerBuild = "aaaaaaaa12345678"
        #expect(observed.controllerBuildAdvisory == nil)
        observed.deployedControllerBuild = nil
        #expect(observed.controllerBuildSummary.contains("unknown"))
        #expect(observed.controllerBuildAdvisory == nil)
    }
}
