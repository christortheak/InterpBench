import Foundation

enum StudyAgentHTTP {
    enum Operation: Sendable { case list, inspect, attach }
    private struct Request: Decodable {
        let workspaceRoot: String
        let name: String?
        let artifactPath: String?
        let artifactFileSHA256: String?
        let manifestFileSHA256: String?
    }

    static func perform(_ operation: Operation, body: Data, workspaceRoot: URL) -> StudyAuthoringHTTP.Response {
        var allowed: Set<String> = ["workspaceRoot"]
        if operation != .list { allowed.insert("artifactPath") }
        if operation == .attach { allowed.formUnion(["name", "artifactFileSHA256", "manifestFileSHA256"]) }
        guard let fields = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any],
            Set(fields.keys).isSubset(of: allowed),
            let request = try? JSONDecoder().decode(Request.self, from: body), request.workspaceRoot.hasPrefix("/"),
            operation == .list || request.artifactPath != nil,
            operation != .attach || request.name != nil else {
            return .failure("invalidAgentRequest", "Name the workspace and operation's fields explicitly.",
                repair: "Use only: " + allowed.sorted().joined(separator: ", "))
        }
        if operation == .attach, request.artifactFileSHA256 == nil || request.manifestFileSHA256 == nil {
            return .failure("attachment_preconditions_required", "Attachment requires the reviewed study and artifact file digests.",
                repair: "Inspect the study and agent, then supply manifestFileSHA256 and artifactFileSHA256.", status: "428 Precondition Required")
        }
        do {
            guard try ManifestFileTransaction.canonicalPath(URL(fileURLWithPath: request.workspaceRoot))
                == ManifestFileTransaction.canonicalPath(workspaceRoot) else {
                return .failure("agentWorkspaceChanged", "The workbench serves another workspace.",
                    repair: "Reconnect to the intended workspace and inspect its study and agent.", status: "409 Conflict")
            }
            switch operation {
            case .list:
                return .json(try StudyAgentAuthoring.list(workspaceRoot: workspaceRoot))
            case .inspect:
                return .json(try AgentArtifactDocument(AgentArtifactSnapshot(workspaceRoot: workspaceRoot, path: request.artifactPath!)))
            case .attach:
                let reviewed = try StudyAgentAuthoring.reviewStudy(name: request.name!, workspaceRoot: workspaceRoot,
                    expectedFileSHA256: request.manifestFileSHA256!)
                let artifact = try StudyAgentAuthoring.reviewArtifact(path: request.artifactPath!, workspaceRoot: workspaceRoot,
                    expectedFileSHA256: request.artifactFileSHA256!)
                let saved = try StudyAgentAuthoring.attach(artifact, reviewed: reviewed)
                return .json(try StudyAuthoringHTTP.Document(saved))
            }
        } catch { return StudyAuthoringHTTP.failure(error) }
    }
}
