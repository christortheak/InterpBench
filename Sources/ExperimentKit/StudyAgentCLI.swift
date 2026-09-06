import Foundation

enum StudyAgentCLI {
    static func run(_ invocation: ExperimentCLIInvocation, workspaceRoot: URL,
                    sink: ExperimentCLISink) throws -> ExperimentCLIResult {
        let args = invocation.args
        switch args.first {
        case "list":
            guard args.count == 1 else { throw usage() }
            let catalog = try StudyAgentAuthoring.list(workspaceRoot: workspaceRoot)
            return try result(catalog, message: "\(catalog.agents.count) agent(s)", sink: sink)
        case "inspect":
            guard args.count == 2 else { throw usage() }
            let snapshot = try AgentArtifactSnapshot(workspaceRoot: workspaceRoot, path: args[1])
            return try result(AgentArtifactDocument(snapshot), message: "Agent inspected.", sink: sink)
        default: throw usage()
        }
    }

    static func attach(_ invocation: ExperimentCLIInvocation, workspaceRoot: URL,
                       sink: ExperimentCLISink) throws -> ExperimentCLIResult {
        let args = invocation.args
        func flag(_ name: String) -> String? {
            guard let index = args.firstIndex(of: name), args.indices.contains(index + 1) else { return nil }
            return args[index + 1]
        }
        guard args.count >= 2, let path = flag("--artifact"), let artifactSHA = flag("--artifact-sha256"),
            let studySHA = flag("--manifest-sha256") else { throw usage() }
        let reviewed = try DraftAuthoringSnapshot.review(name: args[1], workspaceRoot: workspaceRoot, expectedFileSHA256: studySHA)
        let artifact = try StudyAgentAuthoring.reviewArtifact(path: path, workspaceRoot: workspaceRoot, expectedFileSHA256: artifactSHA)
        let saved = try StudyAgentAuthoring.attach(artifact, reviewed: reviewed)
        return try result(StudyAuthoringHTTP.Document(saved), message: "Agent attached to reviewed draft.",
            changed: saved.file.sha256 != reviewed.file.sha256, sink: sink)
    }

    private static func result(_ value: some Encodable, message: String, changed: Bool = false,
                               sink: ExperimentCLISink) throws -> ExperimentCLIResult {
        let data = try JSONEncoder().encode(value)
        let payload = try JSONDecoder().decode([String: JSONValue].self, from: data)
        sink.out(String(decoding: data, as: UTF8.self))
        return ExperimentCLIResult(message: message, changed: changed, payload: payload)
    }

    private static func usage() -> ExperimentError {
        .malformed("Use agent list, agent inspect <path>, or experiment attach-agent <study> with both reviewed file digests.",
            repair: "Read agent --help or experiment attach-agent --help.")
    }
}
