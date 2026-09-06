import Foundation

struct StudyInputDocument: Encodable {
    let ok = true
    let changed: Bool
    let study: StudyAuthoringHTTP.Document
    let file: String
    let promptsFileSHA256: String
    let recordCount: Int
    init(_ result: TaskPromptsAuthoring.Result) throws {
        changed = result.changed
        study = try .init(result.study)
        file = result.prompts.path
        promptsFileSHA256 = result.prompts.file.sha256
        recordCount = try TaskPromptsDocument.load(result.prompts.file.data).count
    }
}

enum StudyInputCLI {
    static func importPrompts(_ invocation: ExperimentCLIInvocation, workspaceRoot: URL,
                              sink: ExperimentCLISink) throws -> ExperimentCLIResult {
        let args = invocation.args
        func flag(_ name: String) -> String? {
            guard let i = args.firstIndex(of: name), args.indices.contains(i + 1) else { return nil }
            return args[i + 1]
        }
        guard args.count == 6, let path = flag("--file"), let expected = flag("--manifest-sha256") else {
            throw ExperimentError.malformed("Name a draft, JSONL file and reviewed manifest digest.",
                repair: "steerlab-cli experiment import-prompts <study> --file <jsonl> --manifest-sha256 <digest> --json")
        }
        let reviewed = try DraftAuthoringSnapshot.review(name: args[1], workspaceRoot: workspaceRoot, expectedFileSHA256: expected)
        let imported = try TaskPromptsAuthoring.importJSONL(String(contentsOfFile: path, encoding: .utf8), reviewed: reviewed)
        let data = try JSONEncoder().encode(StudyInputDocument(imported))
        sink.out(String(decoding: data, as: UTF8.self))
        return ExperimentCLIResult(message: "Prompt records imported and pinned as an immutable input version.", changed: imported.changed,
            payload: try JSONDecoder().decode([String: JSONValue].self, from: data))
    }
}

enum StudyInputHTTP {
    private struct Request: Decodable {
        let workspaceRoot: String
        let name: String
        let text: String
        let manifestFileSHA256: String?
    }
    static func importPrompts(body: Data, workspaceRoot: URL) -> StudyAuthoringHTTP.Response {
        let keys: Set<String> = ["workspaceRoot", "name", "text", "manifestFileSHA256"]
        guard let raw = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any], Set(raw.keys).isSubset(of: keys),
            let request = try? JSONDecoder().decode(Request.self, from: body), request.workspaceRoot.hasPrefix("/") else {
            return .failure("invalidPromptImport", "Supply the named workspace, study and JSONL text.",
                repair: "Use only: " + keys.sorted().joined(separator: ", "))
        }
        guard let expected = request.manifestFileSHA256 else {
            return .failure("manifest_precondition_required", "Prompt import requires the reviewed study file digest.",
                repair: "Inspect the named study and supply manifestFileSHA256 after review.", status: "428 Precondition Required")
        }
        do {
            guard try ManifestFileTransaction.canonicalPath(URL(fileURLWithPath: request.workspaceRoot))
                == ManifestFileTransaction.canonicalPath(workspaceRoot) else {
                throw ExperimentError.refusing(.staleManifest, "The workbench is serving another workspace.",
                    repair: "Reconnect to the intended workspace and review its study before importing.")
            }
            let reviewed = try DraftAuthoringSnapshot.review(name: request.name, workspaceRoot: workspaceRoot, expectedFileSHA256: expected)
            return .json(try StudyInputDocument(TaskPromptsAuthoring.importJSONL(request.text, reviewed: reviewed)))
        } catch { return StudyAuthoringHTTP.failure(error) }
    }
}
