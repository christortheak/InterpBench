import Foundation
import Testing
@testable import ExperimentKit

@MainActor @Suite(.serialized)
struct WorkspaceBootstrapParityTests {
    @Test func pythonCreatesTheSameSeedAndInstructionsAsMac() throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let temp = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temp) }
        let mac = temp.appending(component: "mac"), python = temp.appending(component: "python")
        try WorkspaceStore.create(at: mac, seedingFrom: repository.appending(component: "WorkspaceSeed"))
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"] ?? "python3", "-m", "steerlab_server.client_cli", "workspace", "init", python.path, "--no-git", "--json"]
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONPATH"] = repository.appending(component: "Server").path
        environment.removeValue(forKey: "STEERLAB_WORKSPACE")
        process.environment = environment
        let output = Pipe()
        process.standardOutput = output
        try process.run()
        _ = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0)
        for name in WorkspaceStore.seedManifest + ["AGENTS.md", ".gitignore"] {
            #expect(try Data(contentsOf: mac.appending(path: name)) == Data(contentsOf: python.appending(path: name)), "Seed differs: \(name)")
        }
        #expect(WorkspaceStore.isWorkspace(url: python))
        #expect(try WorkspaceBootstrap.inspect(python)["missingSeedFiles"] == .array([]))
        #expect(try WorkspaceBootstrap.handoff(python)["agentGuide"] == .string(python.appending(component: "AGENTS.md").path))
    }
}
