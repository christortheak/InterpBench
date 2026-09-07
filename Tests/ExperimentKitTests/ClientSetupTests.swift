import Foundation
import Testing
@testable import ExperimentKit

@Suite struct ClientSetupTests {
    @Test func missingRuntimeReportsAuthoringUnavailableWithoutHidingWorkspace() async throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try "workspace".write(to: root.appending(component: "WORKSPACE.md"), atomically: true, encoding: .utf8)
        let result = await ClientSetup.inspect(workspace: root, python: URL(filePath: "/missing-client-python"))
        #expect(result["clientReady"] == .bool(false))
        #expect(result["authoringReady"] == .bool(false))
        guard case .object(let workspace) = result["workspace"] else { Issue.record("Missing workspace readiness"); return }
        #expect(workspace["recognized"] == .bool(true))
    }

    @Test func installerMustMatchCompiledSourcesBeforePlanning() throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try "old source".write(to: root.appending(component: "source.sha256"), atomically: true, encoding: .utf8)
        #expect(throws: (any Error).self) { try ClientSetup.releaseDirectory(explicit: root) }
        try PythonClientIdentity.sourceSHA256.write(to: root.appending(component: "source.sha256"), atomically: true, encoding: .utf8)
        #expect(try ClientSetup.releaseDirectory(explicit: root) == root)
    }

    @Test func setupFlagsRequireExplicitApprovalAndPlanHash() throws {
        for verb in ["apply", "repair"] {
            let spec = try #require(ExperimentCLIParser.spec(namespace: "setup", verb: verb))
            #expect(spec.requiredFlags == ["--expect", "--yes"])
        }
        #expect(ExperimentCLIParser.spec(namespace: "setup", verb: "plan")?.requiredFlags == [])
    }
}
