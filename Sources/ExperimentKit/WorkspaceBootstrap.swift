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

    // What a coding assistant is handed first. The two clients return the same
    // object: the three literals below are duplicated in
    // Server/steerlab_server/client/workspace_bootstrap.py, and
    // WorkspaceBootstrapParityTests.handoffLeadsWithTheInterviewOnBothClients
    // holds them equal.

    /// The three interviews, each with one line a researcher would recognize.
    public static let studyIntents: [(id: String, purpose: String)] = [
        ("conceptStudy", "Steer one model along a concept, such as a tone or a stance, and measure what changes."),
        ("agentComparison", "Compare agents (configured models) with each other, or with the unmodified model, on the same prompts."),
        ("multiAgent", "Have several agents interact in a scenario, and study the conversation."),
    ]
    public static let handoffInstructions =
        "Read AGENTS.md in this workspace, then begin with the study interview: run the first discovery command with the "
        + "intent from studyIntents that fits the researcher's question. Work at the researcher's level: use plain words, "
        + "explain each choice you offer, and settle the research question and the open scientific choices with them. "
        + "Ask before anything that spends compute or money, such as downloading or running a model, submitting a job, "
        + "generating a dataset, or calling a paid judge. Use only what this installed client reports in its help and "
        + "method catalog. Workspace data stays local; running hardware receives execution copies."
    public static let handoffNextAction =
        "Ask the researcher what they want to learn, then run the study interview for the intent that fits."
    /// The placeholder in the first discovery command; `studyIntents` lists its values.
    public static let intentPlaceholder = "<intent>"

    /// What `workspace init` names next: the interview first. It used to be
    /// `experiment create <name> --model <id>`, which asks a new researcher
    /// for a model id before anyone has asked what they want to study.
    /// Python twin: `bootstrap_commands.init_next_action`, which names
    /// `--root` where this names `--workspace`.
    static func initNextAction(
        rootPath: String, workspaceFlag: String = "--workspace"
    ) -> SteerLabCLIEnvelope.NextAction {
        let ids = studyIntents.map(\.id)
        let intents = ids.dropLast().joined(separator: ", ") + ", and " + (ids.last ?? "")
        return .init(
            verb: "authoring study \(intentPlaceholder)",
            detail: "Start with the study interview; the intents are \(intents). "
                + "Name the workspace with \(workspaceFlag) \(rootPath), or export "
                + "STEERLAB_WORKSPACE=\(rootPath). workspace handoff returns these "
                + "first steps for a coding assistant.")
    }

    /// The first commands, in the order to run them: the study interview, the
    /// short method index, then the verb list. Each is a complete argument list.
    static func discovery(executable: String, root: URL) -> [[String]] {
        let here = ["--workspace", root.path, "--json"]
        return [
            [executable, "authoring", "study", intentPlaceholder] + here,
            [executable, "science", "list", "--brief"] + here,
            [executable, "--help"],
        ]
    }

    public static func handoff(_ root: URL, executable: String = cliExecutable) throws -> [String: JSONValue] {
        var report = try inspect(root)
        guard report["recognized"] == .bool(true), report["agentGuidePresent"] == .bool(true) else {
            throw ExperimentError(reason: "Open a workspace with AGENTS.md before preparing an agent handoff.")
        }
        report["executable"] = .array([.string(executable)])
        report["agentGuide"] = .string(root.appending(component: "AGENTS.md").path)
        report["instructions"] = .string(handoffInstructions)
        report["studyIntents"] = .array(studyIntents.map {
            .object(["id": .string($0.id), "purpose": .string($0.purpose)])
        })
        report["discovery"] = .array(discovery(executable: executable, root: root).map {
            .array($0.map(JSONValue.string))
        })
        report["nextAction"] = .string(handoffNextAction)
        return report
    }
}
