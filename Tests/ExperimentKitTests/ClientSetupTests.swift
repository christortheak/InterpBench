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

    @Test func sharedFirstRunCreatesOnlyWithExplicitApproval() async throws {
        let repository = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python = URL(filePath: try #require(ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"]))
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        do {
            _ = try await DiagnosticWorkspace.perform("setup-start", payload: ["workspaceRoot": .string(root.path), "create": .bool(false)], python: python, source: repository.appending(component: "Server"))
            Issue.record("Missing workspace was accepted")
        } catch { #expect(!FileManager.default.fileExists(atPath: root.path)) }
        let response = try await DiagnosticWorkspace.perform("setup-start", payload: ["workspaceRoot": .string(root.path), "create": .bool(true)], python: python, source: repository.appending(component: "Server"))
        guard case .object(let result) = response, case .object(let readiness) = result["readiness"] else { Issue.record("Missing readiness"); return }
        #expect(result["changed"] == .bool(true))
        #expect(readiness["authoringReady"] == .bool(true))
        #expect(WorkspaceStore.isWorkspace(url: root))
        #expect(try String(contentsOf: root.appending(component: "AGENTS.md"), encoding: .utf8) == AgentContract.contents())
    }

    @Test func setupFlagsRequireExplicitApprovalAndPlanHash() throws {
        for verb in ["apply", "repair"] {
            let spec = try #require(ExperimentCLIParser.spec(namespace: "setup", verb: verb))
            #expect(spec.requiredFlags == ["--expect", "--yes"])
        }
        #expect(ExperimentCLIParser.spec(namespace: "setup", verb: "plan")?.requiredFlags == [])
    }
}
