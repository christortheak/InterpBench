import Foundation

extension ClusterClient {
    public func stageDiagnostic(path: String, sha256: String) async throws -> JSONValue {
        try await post("/api/science/stage", body: JSONValue.object(["bundlePath": .string(path), "bundleSHA256": .string(sha256)]), timeout: 3600)
    }
    public func exportDiagnostic(_ jobID: String) async throws -> JSONValue {
        do { return try await post(diagnosticPath(jobID, "export"), body: JSONValue.object([:]), timeout: 3600) }
        catch let error as URLError where error.code == .timedOut {
            throw ExperimentError.malformed("Evidence export timed out; preparation may still be running.", repair: "Restore the connection and request export for the same job again. Do not resubmit the fit. Direct transfer must use the complete exported archive and its SHA-256.")
        }
    }
    public func planDiagnosticCleanup(_ jobID: String, custody: JSONValue) async throws -> JSONValue {
        try await post(diagnosticPath(jobID, "cleanup-plan"), body: JSONValue.object(["custody": custody]))
    }
    public func applyDiagnosticCleanup(_ jobID: String, custody: JSONValue, planSHA256: String) async throws -> JSONValue {
        try await post(diagnosticPath(jobID, "cleanup-apply"), body: JSONValue.object([
            "custody": custody, "planSHA256": .string(planSHA256), "confirmRemoval": .bool(true)]))
    }
    private func diagnosticPath(_ jobID: String, _ verb: String) -> String {
        let allowed = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
        return "/api/science/jobs/" + (jobID.addingPercentEncoding(withAllowedCharacters: allowed) ?? "") + "/" + verb
    }
}

public enum DiagnosticRemote {
    public static func stage(path: String, sha256: String, client: ClusterClient, root: URL) async throws -> JSONValue {
        let result = try await client.stageDiagnostic(path: path, sha256: sha256)
        guard case .object(var object) = result, object["request"] == .object(["inputBundleSHA256": .string(sha256)]) else {
            throw ExperimentError(reason: "The stage response differs from the supplied archive digest.")
        }
        let saved = try await DiagnosticWorkspace.perform("staged-request", payload: ["workspaceRoot": .string(root.path), "bundleSHA256": .string(sha256)])
        if case .object(let fields) = saved { object.merge(fields) { _, new in new } }
        return .object(object)
    }
    public static func fetch(_ jobID: String, client: ClusterClient, root: URL) async throws -> JSONValue {
        try await client.requireHTTPTransfer()
        let reference = try await client.exportDiagnostic(jobID)
        guard case .object(let object) = reference,
              case .string(let path) = object["bundlePath"], case .string(let digest) = object["bundleSha256"],
              case .object(let context) = object["context"], context["jobID"] == .string(jobID) else {
            throw ExperimentError(reason: "The server returned no bound diagnostic export reference.")
        }
        let directory = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let archive = try await client.downloadDiagnosticArchive(path: path, to: directory)
        return try await DiagnosticWorkspace.perform("import", payload: ["workspaceRoot": .string(root.path),
            "archivePath": .string(archive.path), "archiveSHA256": .string(digest), "expectedContext": .object(context)])
    }
    public static func cleanup(_ jobID: String, client: ClusterClient, root: URL, receiptSHA256: String,
                               applyPlanSHA256: String? = nil) async throws -> JSONValue {
        let verified = try await DiagnosticWorkspace.perform("verify-custody", payload: ["workspaceRoot": .string(root.path), "receiptSHA256": .string(receiptSHA256)])
        guard case .object(let document) = verified, let receipt = document["receipt"],
              case .object(let fields) = receipt, case .object(let context) = fields["context"], context["jobID"] == .string(jobID) else {
            throw ExperimentError(reason: "The receipt belongs to another diagnostic job.")
        }
        if let applyPlanSHA256 { return try await client.applyDiagnosticCleanup(jobID, custody: receipt, planSHA256: applyPlanSHA256) }
        return try await client.planDiagnosticCleanup(jobID, custody: receipt)
    }
}

enum DiagnosticRemoteCLI {
    static let verbs = ["science-call", "science-stage", "science-export", "science-fetch", "cleanup-plan", "cleanup-apply"]
    static func run(_ args: [String], client: ClusterClient, root: URL, sink: ExperimentCLISink) async throws -> ExperimentCLIResult {
        let parsed = try DiagnosticArguments(args, namespace: "remote", takesValue: true)
        let id = parsed.positional!
        let result: JSONValue
        switch parsed.verb {
        case "science-call":
            result = try await client.callScientificAction(operation: id, actionID: parsed.flags["--action"]!, document: Data(contentsOf: URL(filePath: parsed.flags["--request"]!)))
        case "science-stage": result = try await DiagnosticRemote.stage(path: id, sha256: parsed.flags["--sha256"]!, client: client, root: root)
        case "science-export": result = try await client.exportDiagnostic(id)
        case "science-fetch": result = try await DiagnosticRemote.fetch(id, client: client, root: root)
        default: result = try await DiagnosticRemote.cleanup(id, client: client, root: root, receiptSHA256: parsed.flags["--receipt-sha256"]!, applyPlanSHA256: parsed.verb == "cleanup-apply" ? parsed.flags["--plan-sha256"] : nil)
        }
        sink.out(String(decoding: try JSONEncoder().encode(result), as: UTF8.self))
        var changed = parsed.verb != "cleanup-plan"
        if case .object(let object) = result, case .bool(let value) = object["changed"] { changed = value }
        return .init(message: "Diagnostic remote operation completed; inspect retained and removed paths.", changed: changed,
                     payload: ["endpoint": .string(client.profile.baseURL.absoluteString), "workspaceRoot": .string(root.path), "response": result])
    }
}
