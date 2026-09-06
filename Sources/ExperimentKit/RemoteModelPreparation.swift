import Foundation

public struct RemoteModelPreparationPlan: Codable, Sendable, Equatable {
    public let modelID: String
    public let revision: String?
    public let target: String
    public let cacheRoot: String
    public let computeEgress: String
    public let installationAllowed: Bool
    public let planSHA256: String
    public let cacheFileSetPresent: Bool
    public let memoryFit: String
    public let credentials: String
    public let note: String
}

extension ClusterClient {
    public func modelPreparationPlan(_ modelID: String, revision: String? = nil) async throws -> RemoteModelPreparationPlan {
        var query = URLComponents()
        query.queryItems = [URLQueryItem(name: "model", value: modelID)]
        if let revision { query.queryItems?.append(URLQueryItem(name: "revision", value: revision)) }
        return try await get("/api/models/plan", queryItems: query.queryItems)
    }
}

enum RemoteModelPreparationCLI {
    static func run(_ args: [String], client: ClusterClient, endpoint: URL, sink: ExperimentCLISink) async throws -> ExperimentCLIResult {
        guard args.count >= 2 else { throw malformed("Name the model or installation job.") }
        func flag(_ name: String) -> String? {
            guard let index = args.firstIndex(of: name), args.indices.contains(index + 1) else { return nil }
            return args[index + 1]
        }
        let result: JSONValue
        let changed: Bool
        switch args[0] {
        case "model-plan":
            result = try encode(await client.modelPreparationPlan(args[1], revision: flag("--revision")))
            changed = false
        case "model-install":
            guard let expected = flag("--plan-sha256"), expected.count == 64,
                expected.allSatisfy({ "0123456789abcdef".contains($0) }) else {
                throw malformed("Supply the exact planSHA256 from remote model-plan on this endpoint.")
            }
            let job = try await client.installModel(args[1], revision: flag("--revision"), planSHA256: expected)
            result = .object(["jobId": .string(job)])
            changed = true
        default:
            let job = try await client.job(args[1])
            guard job.id == args[1], job.kind == "model:install" else {
                throw ExperimentCLIStop(exitCode: 65, state: .refused, code: "modelJobMismatch",
                    reason: "The named job is not a model installation.", repairAction: "Inspect jobs on the original endpoint and use the model-install job ID.",
                    payload: ["endpoint": .string(endpoint.absoluteString), "response": try encode(job)])
            }
            if args[0] == "model-cancel" {
                try await client.cancelJob(args[1])
                result = try encode(await client.job(args[1]))
                changed = true
            } else {
                result = try encode(job); changed = false
                if ["failed", "cancelled"].contains(job.status) {
                    throw ExperimentCLIStop(exitCode: 70, state: .failed, code: "modelInstallationIncomplete",
                        reason: "Model installation did not complete.", repairAction: "Inspect the original job's logs and correct the reported problem before starting another installation.",
                        payload: ["endpoint": .string(endpoint.absoluteString), "response": result])
                }
            }
        }
        let payload: [String: JSONValue] = ["endpoint": .string(endpoint.absoluteString), "response": result]
        sink.out(String(decoding: try JSONEncoder().encode(payload), as: UTF8.self))
        return .init(message: "Model preparation request completed.", changed: changed, payload: payload)
    }

    private static func malformed(_ reason: String) -> ExperimentError {
        .malformed(reason, repair: "Inspect remote model-plan or remote jobs on the intended site; supply its exact plan digest or model-install job ID.")
    }

    private static func encode(_ value: some Encodable) throws -> JSONValue {
        try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(value))
    }
}
