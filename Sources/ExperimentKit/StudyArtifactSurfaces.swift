import Foundation

struct StudyArtifactDocument: Encodable {
    let ok = true
    let changed = true
    let study: StudyAuthoringHTTP.Document
    init(_ snapshot: DraftAuthoringSnapshot) throws { study = try .init(snapshot) }
}

enum StudyArtifactCLI {
    static func run(_ invocation: ExperimentCLIInvocation, workspaceRoot: URL,
                    sink: ExperimentCLISink) throws -> ExperimentCLIResult {
        let args = invocation.args
        func flag(_ name: String) -> String? {
            guard let i = args.firstIndex(of: name), args.indices.contains(i + 1) else { return nil }
            return args[i + 1]
        }
        let data: Data
        let changed: Bool
        if args.first == "inspect-artifact", args.count == 2 {
            data = try JSONEncoder().encode(StudyArtifactAuthoring.inspect(args[1], workspaceRoot: workspaceRoot))
            changed = false
        } else if args.first == "attach-artifact", args.count >= 3,
            !args[1].hasPrefix("--"), !args[2].hasPrefix("--"),
            args.count == 11 + 2 * ["--source-concept", "--eval-run"].compactMap(flag).count,
            let path = flag("--artifact"), let tensor = flag("--artifact-sha256"),
            let sidecar = flag("--sidecar-sha256"), let manifest = flag("--manifest-sha256") {
            let reviewed = try DraftAuthoringSnapshot.review(name: args[1], workspaceRoot: workspaceRoot, expectedFileSHA256: manifest)
            let artifact = try StudyArtifactAuthoring.inspect(path, workspaceRoot: workspaceRoot)
            try StudyArtifactAuthoring.requireDigests(artifact, tensor: tensor, sidecar: sidecar)
            data = try JSONEncoder().encode(StudyArtifactDocument(StudyArtifactAuthoring.attach(args[2], artifact: artifact,
                reviewed: reviewed, sourceConcept: flag("--source-concept"), evalRun: flag("--eval-run"))))
            changed = true
        } else {
            throw ExperimentError.malformed("Inspect a vector path, or attach it to a named study and concept with all reviewed digests.",
                repair: "steerlab-cli experiment attach-artifact --help")
        }
        sink.out(String(decoding: data, as: UTF8.self))
        return ExperimentCLIResult(message: changed ? "Reviewed vector attached; verify the study before freezing." : "Vector bytes and sidecar inspected; attachment performs scientific admission.",
            changed: changed, payload: try JSONDecoder().decode([String: JSONValue].self, from: data))
    }
}

enum StudyArtifactHTTP {
    private struct Request: Decodable {
        let workspaceRoot: String
        let artifact: String
        let name: String?
        let concept: String?
        let manifestFileSHA256: String?
        let artifactSHA256: String?
        let sidecarSHA256: String?
        let sourceConcept: String?
        let evalRun: String?
    }
    static func perform(attach: Bool, body: Data, workspaceRoot: URL) -> StudyAuthoringHTTP.Response {
        let keys: Set<String> = attach
            ? ["workspaceRoot", "artifact", "name", "concept", "manifestFileSHA256", "artifactSHA256", "sidecarSHA256", "sourceConcept", "evalRun"]
            : ["workspaceRoot", "artifact"]
        guard let raw = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any],
            Set(raw.keys).isSubset(of: keys), let request = try? JSONDecoder().decode(Request.self, from: body),
            request.workspaceRoot.hasPrefix("/"), !attach || (request.name != nil && request.concept != nil) else {
            return .failure("invalidArtifactRequest", "Supply an artifact path and the explicit study workspace.",
                repair: "Use only: " + keys.sorted().joined(separator: ", "))
        }
        if attach && (request.manifestFileSHA256 == nil || request.artifactSHA256 == nil || request.sidecarSHA256 == nil) {
            return .failure("artifact_precondition_required", "Attachment requires the reviewed manifest and both artifact file digests.",
                repair: "Inspect the study and artifact, then supply manifestFileSHA256, artifactSHA256 and sidecarSHA256.", status: "428 Precondition Required")
        }
        do {
            guard try ManifestFileTransaction.canonicalPath(URL(fileURLWithPath: request.workspaceRoot))
                == ManifestFileTransaction.canonicalPath(workspaceRoot) else {
                throw ExperimentError.refusing(.staleManifest, "The workbench is serving another workspace.",
                    repair: "Reconnect to the intended workspace and inspect the study and vector again.")
            }
            let artifact = try StudyArtifactAuthoring.inspect(request.artifact, workspaceRoot: workspaceRoot)
            guard attach else { return .json(artifact) }
            let reviewed = try DraftAuthoringSnapshot.review(name: request.name!, workspaceRoot: workspaceRoot,
                expectedFileSHA256: request.manifestFileSHA256!)
            try StudyArtifactAuthoring.requireDigests(artifact, tensor: request.artifactSHA256!, sidecar: request.sidecarSHA256!)
            return .json(try StudyArtifactDocument(StudyArtifactAuthoring.attach(request.concept!, artifact: artifact,
                reviewed: reviewed, sourceConcept: request.sourceConcept, evalRun: request.evalRun)))
        } catch { return StudyAuthoringHTTP.failure(error) }
    }
}
