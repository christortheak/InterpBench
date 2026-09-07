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
