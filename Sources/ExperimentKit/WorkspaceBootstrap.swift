import Foundation

/// Read-only first-run information over the shared workspace seed contract.
public enum WorkspaceBootstrap {
    public static func inspect(_ root: URL) throws -> [String: JSONValue] {
        var directory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &directory), directory.boolValue else {
            throw ExperimentError(reason: "Select an existing workspace directory.")
        }
        let missing = WorkspaceStore.seedManifest.filter {
            !FileManager.default.fileExists(atPath: root.appending(path: $0).path)
        }
        return ["workspaceRoot": .string(root.resolvingSymlinksInPath().path),
                "recognized": .bool(WorkspaceStore.isWorkspace(url: root)),
                "agentGuidePresent": .bool(FileManager.default.fileExists(atPath: root.appending(component: "AGENTS.md").path)),
                "missingSeedFiles": .array(missing.map(JSONValue.string)), "seedSchemaVersion": .number(1), "changed": .bool(false)]
    }

    public static var cliExecutable: String {
        if let app = CodeResources.enclosingAppBundle {
            return app.appending(path: "Contents/Helpers/steerlab-cli").path
        }
        if URL(filePath: CommandLine.arguments[0]).lastPathComponent == "steerlab-cli" {
            return URL(filePath: CommandLine.arguments[0]).resolvingSymlinksInPath().path
        }
        return "steerlab-cli"
    }

    public static func handoff(_ root: URL, executable: String = cliExecutable) throws -> [String: JSONValue] {
        var report = try inspect(root)
        guard report["recognized"] == .bool(true), report["agentGuidePresent"] == .bool(true) else {
            throw ExperimentError(reason: "Open a workspace with AGENTS.md before preparing an agent handoff.")
        }
        report["executable"] = .array([.string(executable)])
        report["agentGuide"] = .string(root.appending(component: "AGENTS.md").path)
        report["instructions"] = .string("Read AGENTS.md before working. Discuss the research question and unresolved scientific choices with the researcher. Use this installed client's help and method catalog. Workspace data remains local; running hardware receives execution copies.")
        report["discovery"] = .array([
            .array([.string(executable), .string("--help")]),
            .array([executable, "science", "list", "--workspace", root.path, "--json"].map(JSONValue.string))])
        return report
    }
}
