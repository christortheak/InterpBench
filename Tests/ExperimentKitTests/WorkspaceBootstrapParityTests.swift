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

    /// Runs the Python client and returns the one document it wrote.
    func pythonDocument(_ arguments: [String], repository: URL) throws -> [String: JSONValue] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"] ?? "python3", "-m", "steerlab_server.client_cli"] + arguments
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONPATH"] = repository.appending(component: "Server").path
        environment.removeValue(forKey: "STEERLAB_WORKSPACE")
        process.environment = environment
        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()
        try process.run()
        let bytes = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0, "\(arguments)")
        guard case .object(let document) = try JSONDecoder().decode(JSONValue.self, from: bytes) else {
            throw ExperimentError(reason: "The Python client returned no document.")
        }
        return document
    }

    /// What a coding assistant is handed first. The old handoff sent it to
    /// `--help` and then to the full 75 KB method catalog, and never mentioned
    /// the study interview, which is where a new researcher actually starts.
    /// The two clients hand over the same object, apart from the executable
    /// and the spelling of the workspace flag.
    @Test func handoffLeadsWithTheInterviewOnBothClients() throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let temp = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temp) }
        let workspace = temp.appending(component: "workspace")
        try WorkspaceStore.create(at: workspace, seedingFrom: repository.appending(component: "WorkspaceSeed"))

        let mac = try WorkspaceBootstrap.handoff(workspace, executable: "steerlab-cli")
        let here = ["--workspace", workspace.path, "--json"]
        #expect(mac["discovery"] == .array([
            .array((["steerlab-cli", "authoring", "study", "<intent>"] + here).map(JSONValue.string)),
            .array((["steerlab-cli", "science", "list", "--brief"] + here).map(JSONValue.string)),
            .array(["steerlab-cli", "workspace", "guide", "verbs"].map(JSONValue.string)),
        ]))
        #expect(WorkspaceBootstrap.studyIntents.map(\.id) == StudyIntent.allCases.map(\.rawValue))
        #expect(WorkspaceBootstrap.studyIntents.allSatisfy { $0.purpose.hasSuffix(".") })
        let instructions = WorkspaceBootstrap.handoffInstructions
        #expect(mac["instructions"] == .string(instructions))
        #expect(instructions.contains("Work at the researcher's level"))
        #expect(instructions.contains("Ask before anything that spends compute or money"))
        #expect(instructions.contains("study interview") && instructions.contains("studyIntents"))

        // The Python client, pointed at the same workspace.
        let document = try pythonDocument(["workspace", "handoff", "--root", workspace.path, "--json"], repository: repository)
        guard case .object(let python) = try #require(document["result"]) else {
            Issue.record("The Python handoff has no result.")
            return
        }
        #expect(Set(python.keys) == Set(mac.keys))
        for key in ["instructions", "studyIntents", "nextAction", "agentGuidePresent", "recognized", "missingSeedFiles", "seedSchemaVersion", "changed"] {
            #expect(python[key] == mac[key], "handoff key '\(key)' differs between the clients")
        }
        // Discovery is the same three commands in the same order. Python
        // names the workspace by its own resolved spelling of the path.
        guard case .array(let executable) = try #require(python["executable"]),
              case .array(let commands) = try #require(python["discovery"]),
              case .string(let pythonRoot) = try #require(python["workspaceRoot"]) else {
            Issue.record("The Python handoff names no commands.")
            return
        }
        let tails: [[JSONValue]] = commands.map { command in
            guard case .array(let words) = command else { return [] }
            return Array(words.dropFirst(executable.count))
        }
        #expect(tails == [
            ["authoring", "study", "<intent>", "--root", pythonRoot, "--json"].map(JSONValue.string),
            ["science", "list", "--brief", "--root", pythonRoot, "--json"].map(JSONValue.string),
            ["workspace", "guide", "verbs"].map(JSONValue.string),
        ])

        // After `workspace init`, both clients point at the interview first,
        // in the same words.
        let next = WorkspaceBootstrap.initNextAction(rootPath: "/study")
        #expect(next.verb == "authoring study <intent>")
        #expect(next.detail?.contains("conceptStudy, agentComparison, and multiAgent") == true)
        #expect(next.detail?.contains("--workspace /study") == true)
        let created = try pythonDocument(["workspace", "init", temp.appending(component: "python-new").path, "--no-git", "--json"], repository: repository)
        guard case .object(let pythonNext) = try #require(created["nextAction"]),
              case .object(let createdResult) = try #require(created["result"]),
              case .string(let createdRoot) = try #require(createdResult["workspaceRoot"]) else {
            Issue.record("The Python client named no next action after workspace init.")
            return
        }
        let expected = WorkspaceBootstrap.initNextAction(rootPath: createdRoot, workspaceFlag: "--root")
        #expect(pythonNext["verb"] == .string(expected.verb))
        #expect(pythonNext["detail"] == .string(try #require(expected.detail)))
    }
}
