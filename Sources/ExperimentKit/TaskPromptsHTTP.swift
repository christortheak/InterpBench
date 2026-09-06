import Foundation

/// Named prompt reads and versioned edits, without observable editor state.
enum TaskPromptsHTTP {
    private struct Request: Decodable {
        let name: String
        let workspaceRoot: String
        let file: String
        let manifestFileSHA256: String?
        let promptsFileSHA256: String?
        let sourceAbsent: Bool?
        let text: String?
    }

    private struct Prompts: Encodable {
        let file: String
        let promptsFileSHA256: String
        let text: String
        let count: Int
        let instrumentSummary: String?

        init(_ review: TaskPromptsFileReview) throws {
            let document = try TaskPromptsDocument.load(review.file.data)
            file = review.path
            promptsFileSHA256 = review.file.sha256
            text = document.editorText
            count = document.count
            instrumentSummary = document.instrumentSummary
        }
    }

    private struct Result: Encodable {
        let ok = true
        let study: StudyAuthoringHTTP.Document
        let prompts: Prompts
    }

    static func perform(body: Data, saving: Bool, workspaceRoot: URL) -> StudyAuthoringHTTP.Response {
        let keys: Set<String> = saving
            ? ["name", "workspaceRoot", "file", "manifestFileSHA256", "promptsFileSHA256", "sourceAbsent", "text"]
            : ["name", "workspaceRoot", "file"]
        guard let object = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any],
            let request = try? JSONDecoder().decode(Request.self, from: body)
        else {
            return .failure("invalidPromptRequest", "Name the study, workspace and prompt file explicitly.",
                repair: "Supply name, absolute workspaceRoot and workspace-relative file in the JSON request.")
        }
        let unknown = Set(object.keys).subtracting(keys).sorted()
        guard unknown.isEmpty else {
            return .failure("unknownPromptField", "Unknown prompt fields: " + unknown.joined(separator: ", "),
                repair: "Use only: " + keys.sorted().joined(separator: ", "))
        }
        guard request.workspaceRoot.hasPrefix("/") else {
            return .failure("targetRequired", "workspaceRoot must be an absolute path.",
                repair: "Use the workspaceRoot returned by the named manifest read.")
        }
        if saving {
            guard let expected = request.manifestFileSHA256 else {
                return .failure("manifest_precondition_required", "Prompt edits require the reviewed study digest.",
                    repair: "Read and review the named study; supply its manifestFileSHA256.",
                    status: "428 Precondition Required")
            }
            guard isDigest(expected), request.text != nil else {
                return .failure("invalidPromptRequest", "Supply a lowercase SHA-256 manifest digest and edited text.",
                    repair: "Use the reviewed read's digest and a text string; do not invent a digest.")
            }
            guard request.sourceAbsent == true ? request.promptsFileSHA256 == nil : request.promptsFileSHA256 != nil else {
                return .failure("prompt_precondition_required", "Specify exactly one reviewed source state.",
                    repair: "Supply promptsFileSHA256 from Load Prompts, or sourceAbsent: true for a new source path.",
                    status: "428 Precondition Required")
            }
            if let digest = request.promptsFileSHA256, !isDigest(digest) {
                return .failure("invalid_prompt_precondition", "promptsFileSHA256 must be a lowercase SHA-256 digest.",
                    repair: "Use the promptsFileSHA256 returned by the prompt read.")
            }
        }
        do {
            guard try ManifestFileTransaction.canonicalPath(URL(fileURLWithPath: request.workspaceRoot))
                    == ManifestFileTransaction.canonicalPath(workspaceRoot) else {
                throw ExperimentError.refusing(.staleManifest, "The workbench is serving another workspace.",
                    repair: "Reconnect to the intended workspace and review the study and prompt source again.")
            }
            let study = try DraftAuthoringSnapshot(workspaceRoot: workspaceRoot, name: request.name)
            if saving, request.manifestFileSHA256 != study.file.sha256 {
                throw ExperimentError.refusing(.staleManifest, "The study changed after it was reviewed.",
                    repair: "Read and review the named study before reconstructing the intended prompt edit.")
            }
            let source: TaskPromptsFileReview?
            if saving && request.sourceAbsent == true {
                source = nil
            } else {
                let loaded = try TaskPromptsFileReview(path: request.file, workspaceRoot: workspaceRoot)
                if saving, loaded.file.sha256 != request.promptsFileSHA256 {
                    throw ExperimentError.refusing(.staleManifest, "The prompt source changed after it was reviewed.",
                        repair: "Load and review the prompt source again before applying the edit.")
                }
                source = loaded
            }
            if saving {
                let saved = try TaskPromptsAuthoring.save(reviewed: study, path: request.file,
                    source: source, editorText: request.text ?? "")
                return .json(try Result(study: StudyAuthoringHTTP.Document(saved.study), prompts: Prompts(saved.prompts)))
            }
            guard let source else {
                return .failure("promptReadFailed", "The prompt source was not loaded.",
                    repair: "Read the named prompt source again.", status: "500 Internal Server Error")
            }
            return .json(try Result(study: StudyAuthoringHTTP.Document(study), prompts: Prompts(source)))
        } catch let error as VectorCatalog.PathError {
            return .failure("invalidPromptPath", error.localizedDescription,
                repair: "Choose a workspace-relative prompt file inside the named workspace.")
        } catch { return StudyAuthoringHTTP.failure(error) }
    }

    private static func isDigest(_ value: String) -> Bool {
        value.count == 64 && value.allSatisfy { "0123456789abcdef".contains($0) }
    }
}
