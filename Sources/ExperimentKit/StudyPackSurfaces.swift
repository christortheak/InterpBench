import Foundation

struct StudyPackDocument: Encodable {
    let ok = true
    let changed = true
    let study: StudyAuthoringHTTP.Document
    let verificationIssues: [String]
    let filesWritten: [String]
    let nextSteps: [String]
    init(_ result: StudyPackAuthoring.Result) throws {
        study = try .init(result.study)
        verificationIssues = result.violations
        filesWritten = result.filesWritten
        nextSteps = result.violations.isEmpty
            ? ["Review the study design and execution prerequisites before freezing and running."]
            : ["Resolve the reported verification issues, then verify the named draft again."]
    }
}

enum StudyPackCLI {
    static func run(_ invocation: ExperimentCLIInvocation, workspaceRoot: URL,
                    sink: ExperimentCLISink) throws -> ExperimentCLIResult {
        let args = invocation.args
        func flag(_ name: String) -> String? {
            guard let i = args.firstIndex(of: name), args.indices.contains(i + 1) else { return nil }
            return args[i + 1]
        }
        guard args.count >= 2 else { throw usage() }
        let data: Data
        let changed: Bool
        switch args[0] {
        case "preview":
            guard args.count == 2 else { throw usage() }
            data = try JSONEncoder().encode(StudyPackAuthoring.preview(
                Data(contentsOf: URL(fileURLWithPath: args[1])), workspaceRoot: workspaceRoot))
            changed = false
        case "apply":
            guard args.count == 4, let expected = flag("--review-sha256") else { throw usage() }
            let result = try StudyPackAuthoring.apply(Data(contentsOf: URL(fileURLWithPath: args[1])),
                workspaceRoot: workspaceRoot, expectedReviewSHA256: expected)
            data = try JSONEncoder().encode(StudyPackDocument(result))
            changed = true
        case "export":
            guard args.count == 2 else { throw usage() }
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: workspaceRoot, name: args[1])
            let exported = try StudyPackAuthoring.export(reviewed: reviewed)
            struct Export: Encodable { let pack: JSONValue; let externalDependencies: [String] }
            data = try JSONEncoder().encode(Export(
                pack: JSONDecoder().decode(JSONValue.self, from: exported.data), externalDependencies: exported.externalDependencies))
            changed = false
        default: throw usage()
        }
        sink.out(String(decoding: data, as: UTF8.self))
        return ExperimentCLIResult(message: changed ? "Draft imported; inspect verificationIssues before running." : "Study pack inspected.",
            changed: changed, payload: try JSONDecoder().decode([String: JSONValue].self, from: data))
    }
    private static func usage() -> ExperimentError {
        .malformed("Use pack preview <file>, pack apply <file> --review-sha256 <digest>, or pack export <study>.",
            repair: "steerlab-cli pack --help")
    }
}

enum StudyPackHTTP {
    enum Operation: String { case preview, apply, export }
    private struct Request: Decodable {
        let workspaceRoot: String
        let text: String?
        let name: String?
        let reviewSHA256: String?
    }
    static func perform(_ operation: Operation, body: Data, workspaceRoot: URL) -> StudyAuthoringHTTP.Response {
        let allowed: Set<String> = operation == .export ? ["workspaceRoot", "name"]
            : operation == .apply ? ["workspaceRoot", "text", "reviewSHA256"] : ["workspaceRoot", "text"]
        guard let raw = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any],
            Set(raw.keys).isSubset(of: allowed),
            let request = try? JSONDecoder().decode(Request.self, from: body), request.workspaceRoot.hasPrefix("/"),
            operation == .export ? request.name != nil : request.text != nil else {
            return .failure("invalidPackRequest", "Supply explicit pack-operation fields.",
                repair: "Use only: " + allowed.sorted().joined(separator: ", "))
        }
        if operation == .apply, request.reviewSHA256 == nil {
            return .failure("pack_precondition_required", "Applying a pack requires its reviewed preview digest.",
                repair: "Preview the exact text and supply reviewSHA256 after inspecting the file plan.", status: "428 Precondition Required")
        }
        do {
            guard try ManifestFileTransaction.canonicalPath(URL(fileURLWithPath: request.workspaceRoot))
                == ManifestFileTransaction.canonicalPath(workspaceRoot) else {
                throw ExperimentError.refusing(.staleManifest, "The workbench is serving another workspace.",
                    repair: "Reconnect to the intended workspace and preview the pack there.")
            }
            switch operation {
            case .preview: return .json(try StudyPackAuthoring.preview(Data(request.text!.utf8), workspaceRoot: workspaceRoot))
            case .apply: return .json(try StudyPackDocument(StudyPackAuthoring.apply(Data(request.text!.utf8),
                workspaceRoot: workspaceRoot, expectedReviewSHA256: request.reviewSHA256!)))
            case .export:
                let exported = try StudyPackAuthoring.export(reviewed: DraftAuthoringSnapshot(workspaceRoot: workspaceRoot, name: request.name!))
                struct Export: Encodable { let ok = true; let pack: JSONValue; let externalDependencies: [String] }
                return .json(try Export(pack: JSONDecoder().decode(JSONValue.self, from: exported.data), externalDependencies: exported.externalDependencies))
            }
        } catch { return StudyAuthoringHTTP.failure(error) }
    }
}
