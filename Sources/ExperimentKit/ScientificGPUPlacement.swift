import Foundation

/// Placement comes from the captured controller, independently of scientific content.
public struct ScientificGPUPlacement: Codable, Sendable {
    public let available: Bool
    public let gpuTypes: [String]
    public let defaultGPUType: String?
    public let gpuVRAMGB: [String: Int]

    public var defaultLabel: String { "Site default — " + (defaultGPUType ?? "not declared") }
    public func capacityLabel(_ selection: String) -> String {
        let type = selection.isEmpty ? defaultGPUType : selection
        guard let type, let capacity = gpuVRAMGB[type] else {
            return "The controller has not declared this GPU’s memory capacity. Workload fit is not checked."
        }
        return "Site-declared memory: \(capacity) GB per GPU. This is not a peak-memory estimate or a verified workload fit."
    }

    public static func usesGPU(_ request: JSONValue) -> Bool {
        guard case .object(let object) = request, case .string(let operation) = object["operation"],
              let method = try? ScienceCatalog.operation(operation) else { return false }
        return method.compute != "cpu"
    }

    public static func roundBody(hash: String? = nil, gpuType: String = "", overrides: [Int: String] = [:]) -> [String: JSONValue] {
        var body: [String: JSONValue] = hash.map { ["planSHA256": .string($0), "confirmAction": .bool(true)] } ?? [:]
        if !gpuType.isEmpty { body["gpuType"] = .string(gpuType) }
        let chosen = overrides.filter { !$0.value.isEmpty }
        if !chosen.isEmpty {
            body["shardGPUTypes"] = .object(Dictionary(uniqueKeysWithValues: chosen.map { (String($0.key), .string($0.value)) }))
        }
        return body
    }

    public static func submitIndices(_ value: JSONValue) -> [Int] {
        guard case .object(let object) = value, case .array(let indices) = object["submitIndices"] else { return [] }
        return indices.compactMap { if case .number(let n) = $0, let i = Int(exactly: n) { return i }; return nil }
    }

    public static func reviewLines(_ value: JSONValue) -> [String] {
        guard case .object(let object) = value else { return [] }
        var lines: [String] = []
        if case .object(let resources) = object["resources"], case .string(let gres) = resources["gres"] {
            lines.append("Reviewed GPU allocation: " + gres)
        }
        if case .object(let review) = object["gpuReview"] {
            for key in ["summary", "throughputScope"] {
                if case .string(let text) = review[key] { lines.append(text) }
            }
            if case .object(let hardware) = review["pilotHardware"], case .string(let name) = hardware["deviceName"] {
                lines.append("Throughput pilot GPU: " + name)
            }
        }
        if case .array(let shards) = object["shards"] {
            for case .object(let shard) in shards {
                guard case .number(let index) = shard["index"], let number = Int(exactly: index),
                      case .string(let status) = shard["status"] else { continue }
                let type: String
                if case .string(let name) = shard["gpuType"] { type = name } else { type = "GPU placement not recorded" }
                lines.append("Shard \(number): \(status) — \(type)")
            }
        }
        return lines
    }
}

extension ClusterClient {
    public func scientificGPUPlacement() async throws -> ScientificGPUPlacement? {
        struct Response: Decodable { let sciencePlacement: ScientificGPUPlacement? }
        let response: Response = try await get("/api/capabilities")
        return response.sciencePlacement
    }
}
