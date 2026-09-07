import Foundation
import Testing
@testable import ExperimentKit

@Suite(.serialized) struct ScientificPythonRuntimeTests {
    @Test func explicitInterpreterDoesNotFallBackAndInstalledClientNeedsNoCheckout() throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        try FileManager.default.createDirectory(at: root.appending(component: "bin"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let installed = root.appending(path: "bin/python")
        try FileManager.default.createSymbolicLink(at: installed, withDestinationURL: URL(filePath: "/usr/bin/true"))
        #expect(ScientificPythonRuntime.resolve(environment: [:], clientEnvironment: root, checkoutPython: nil) == installed)
        #expect(ScientificPythonRuntime.resolve(environment: ["STEERLAB_CLIENT_PYTHON": "relative/python"], clientEnvironment: root, checkoutPython: installed) == nil)
        let missing = root.appending(component: "missing")
        #expect(ScientificPythonRuntime.resolve(environment: ["STEERLAB_CLIENT_PYTHON": missing.path], clientEnvironment: root, checkoutPython: installed) == missing)
    }

    private func sourceFromReleaseBundle(_ bundle: URL) throws -> URL {
        ExperimentRootOverrideLock.acquire()
        defer { ExperimentRootOverrideLock.release() }
        let oldBundle = CodeResources.bundleOverrideForTesting
        let oldMode = CodeResources.modeOverrideForTesting
        let oldHomes = CodeResources.executableHomesOverrideForTesting
        CodeResources.bundleOverrideForTesting = bundle
        CodeResources.modeOverrideForTesting = .release
        CodeResources.executableHomesOverrideForTesting = []
        defer {
            CodeResources.bundleOverrideForTesting = oldBundle
            CodeResources.modeOverrideForTesting = oldMode
            CodeResources.executableHomesOverrideForTesting = oldHomes
        }
        #expect(LocalPythonRuntime.repoRoot == nil)
        return try CodeResources.serverPayload()
    }

    @Test func releasePayloadAuthorsWithoutCheckoutAndRefusesSourceMismatch() async throws {
        let fm = FileManager.default
        let repository = URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python = URL(filePath: try #require(ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"]))
        let root = fm.temporaryDirectory.appending(component: UUID().uuidString)
        let payload = root.appending(path: "Resources/ServerPayload")
        try fm.createDirectory(at: payload, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }
        // Copy only shipped files, not caches, a venv, or a checkout marker.
        let package = repository.appending(path: "Server/steerlab_server").resolvingSymlinksInPath()
        let copied = payload.appending(component: "steerlab_server")
        try fm.copyItem(at: package, to: copied)
        let enumerator = try #require(fm.enumerator(at: copied, includingPropertiesForKeys: nil))
        while let path = enumerator.nextObject() as? URL {
            if path.lastPathComponent == "__pycache__" {
                enumerator.skipDescendants()
                try fm.removeItem(at: path)
            }
        }
        let source = try sourceFromReleaseBundle(root.appending(component: "Resources"))
        #expect(source.path == payload.path)
        #expect(fm.fileExists(atPath: payload.appending(path: "steerlab_server/client/diagnostic_workspace.py").path))
        let workspace = root.appending(component: "workspace")
        try fm.createDirectory(at: workspace, withIntermediateDirectories: false)
        let fields: [String: JSONValue] = ["workspaceRoot": .string(workspace.path), "operation": .string("optvec-gradient")]
        let result = try await DiagnosticWorkspace.perform("interview", payload: fields, python: python, source: source)
        guard case .object(let interview) = result else { Issue.record("Missing interview"); return }
        #expect(interview["id"] == .string("optvec-gradient"))
        #expect(!fm.fileExists(atPath: payload.appending(path: "steerlab_server/__pycache__").path))
        let owner = payload.appending(path: "steerlab_server/experiment/method_authoring.py")
        let text = try String(contentsOf: owner, encoding: .utf8)
        try (text + "\n# simulated mismatched payload\n").write(to: owner, atomically: true, encoding: .utf8)
        do {
            _ = try await DiagnosticWorkspace.perform("interview", payload: fields, python: python, source: source)
            Issue.record("Mismatched client sources were accepted")
        } catch {
            #expect(String(describing: error).contains("sources differ"))
        }
        #expect(try fm.contentsOfDirectory(atPath: workspace.path).isEmpty)
    }
}
