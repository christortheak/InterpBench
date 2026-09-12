import Foundation
import CoreFoundation

public enum ScientificRequestDocument {
    /// Preserve the only UInt64 field before decoding into the UI's generic
    /// JSON number type. The server canonicalizes both spellings to decimal text.
    public static func read(_ data: Data) throws -> JSONValue {
        guard var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ExperimentError(reason: "A diagnostic request must be a JSON object.")
        }
        if var parameters = object["parameters"] as? [String: Any],
           let seed = parameters["seed"] as? NSNumber {
            guard CFGetTypeID(seed) != CFBooleanGetTypeID() else {
                throw ExperimentError(reason: "A diagnostic seed must be an integer, not a boolean.")
            }
            guard !["d", "f"].contains(String(cString: seed.objCType)) else {
                throw ExperimentError(reason: "A diagnostic seed must be an integer or decimal string.")
            }
            parameters["seed"] = seed.stringValue
            object["parameters"] = parameters
        }
        return try JSONDecoder().decode(JSONValue.self, from: JSONSerialization.data(withJSONObject: object))
    }
}

extension ClusterClient {
    public func scientificPlan(_ request: JSONValue, gpuType: String? = nil) async throws -> JSONValue {
        // A staged bundle is re-verified file by file at plan time; a multi-gigabyte
        // checkpoint can take minutes before the first response byte. A GPU type is
        // placement, so it wraps the request instead of editing it.
        let body: JSONValue = gpuType.map { .object(["request": request, "gpuType": .string($0)]) } ?? request
        return try await post("/api/science/plan", body: body, timeout: 3600)
    }
    public func scientificSubmit(_ request: JSONValue, planSHA256: String, gpuType: String? = nil) async throws -> JSONValue {
        var body: [String: JSONValue] = ["request": request, "planSHA256": .string(planSHA256)]
        if let gpuType { body["gpuType"] = .string(gpuType) }
        let response: JSONValue = try await post("/api/science/submit", body: JSONValue.object(body), timeout: 3600)
        guard case .object(let object) = response, case .string(let id) = object["jobId"], !id.isEmpty else {
            throw ExperimentError(reason: "Diagnostic submission returned no job ID. Its outcome is uncertain; inspect jobs on this endpoint before any retry.")
        }
        return response
    }
    public func reconcileJobs() async throws -> JSONValue {
        try await post("/api/jobs/reconcile", body: JSONValue.object([:]))
    }
    public func recoveryReview(_ jobID: String) async throws -> JSONValue {
        try await get("/api/jobs/\(jobID.addingPercentEncoding(withAllowedCharacters: CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")) ?? jobID)/recovery")
    }
    public func recoverJob(_ jobID: String, reviewToken: String, reason: String) async throws -> JSONValue {
        try await post("/api/jobs/\(jobID.addingPercentEncoding(withAllowedCharacters: CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")) ?? jobID)/recover",
            body: JSONValue.object(["reviewToken": .string(reviewToken), "reason": .string(reason), "confirmOwnerExited": .bool(true)]))
    }
}

enum RemoteScientificWorkflowsCLI {
    static let verbs = ["science-plan", "science-submit", "recovery", "recover", "reconcile"]

    static func run(_ args: [String], client: ClusterClient, endpoint: URL, sink: ExperimentCLISink) async throws -> ExperimentCLIResult {
        guard let verb = args.first, verbs.contains(verb),
              let spec = ExperimentCLIParser.spec(namespace: "remote", verb: verb) else {
            throw ExperimentError(reason: "Unknown remote scientific workflow.")
        }
        var flags: [String: String] = [:]
        var positionals: [String] = []
        var index = 1
        while index < args.count {
            let word = args[index]
            if spec.valueFlags.contains(word) || spec.booleanFlags.contains(word) {
                guard flags[word] == nil else { throw ExperimentError(reason: "Supply each flag exactly once: " + word) }
                if spec.valueFlags.contains(word) {
                    guard args.indices.contains(index + 1) else { throw ExperimentError(reason: "Missing value for " + word) }
                    flags[word] = args[index + 1]; index += 2
                } else { flags[word] = ""; index += 1 }
            } else {
                guard !word.hasPrefix("--") else { throw ExperimentError(reason: "Unknown flag: " + word) }
                positionals.append(word); index += 1
            }
        }
        func flag(_ name: String) -> String? { flags[name] }
        guard positionals.count == (verb == "reconcile" ? 0 : 1) else {
            throw ExperimentError.malformed("Supply exactly the request file or job ID; reconcile takes neither.", repair: "Read remote " + verb + " --help.")
        }
        let result: JSONValue
        switch args[0] {
        case "science-plan", "science-submit":
            let request = try ScientificRequestDocument.read(Data(contentsOf: URL(filePath: positionals[0])))
            let gpuType = flag("--gpu-type")
            if let gpuType, gpuType.trimmingCharacters(in: .whitespaces).isEmpty {
                throw ExperimentError.malformed("Supply a declared GPU type with --gpu-type, or omit it for the site default.", repair: "Read cluster preview for the site's declared GPU types.")
            }
            if args[0] == "science-plan" { result = try await client.scientificPlan(request, gpuType: gpuType) }
            else {
                guard let expected = flag("--plan-sha256"), expected.count == 64,
                    expected.allSatisfy({ "0123456789abcdef".contains($0) }) else {
                    throw ExperimentError.malformed("Supply the exact planSHA256 from this endpoint's science-plan.", repair: "Review the request again and pass --plan-sha256 <digest>.")
                }
                result = try await client.scientificSubmit(request, planSHA256: expected, gpuType: gpuType)
            }
        case "reconcile": result = try await client.reconcileJobs()
        case "recovery": result = try await client.recoveryReview(positionals[0])
        default:
            guard flag("--confirm-owner-exited") != nil, let reviewToken = flag("--review-token"),
                  !reviewToken.isEmpty, let reason = flag("--reason"),
                  !reason.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ExperimentError.malformed("Recovery requires an explicit owner-exited attestation.", repair: "Inspect remote recovery, confirm the owner has exited, then supply --confirm-owner-exited, --review-token and --reason.")
            }
            result = try await client.recoverJob(positionals[0], reviewToken: reviewToken, reason: reason)
        }
        let payload: [String: JSONValue] = ["endpoint": .string(endpoint.absoluteString), "response": result]
        if args[0] == "science-submit", case .object(let object) = result, object["status"] == .string("parked") {
            throw ExperimentCLIStop(exitCode: 70, state: .failed, code: "schedulerSubmissionUncertain",
                reason: "Scheduler reply was uncertain; the durable submission record is retained.",
                repairAction: "Inspect this endpoint and schedulerSubmissionName before any retry; reconcile child records when available.", payload: payload, changed: true)
        }
        sink.out(String(decoding: try JSONEncoder().encode(payload), as: UTF8.self))
        return .init(message: "Remote request completed. Retain the endpoint and job ID to reconnect.",
                     changed: ["science-submit", "recover", "reconcile"].contains(args[0]), payload: payload)
    }
}
