import Foundation

/// All library surfaces read the same portable owner. These are presentation
/// records, not replacement serialization for a pinned probe's original bytes.
public enum ProbeLibrary {
    public struct Record: Codable, Sendable, Identifiable {
        public let path: String
        public let sha256: String
        public let format: String
        public let label: String
        public let modelID: String
        public let layer: Int
        public let method: String
        public let createdAt: String?
        public let limitations: [String]
        public let document: JSONValue?
        public var id: String { path }
        public var methodLabel: String {
            guard format == "activation-probe-v1" else { return method }
            return switch method {
            case "mean-difference-v1": "Mean-difference reader"
            case "linear-logit-v1": "Linear classifier"
            case "mlp-relu-logit-v1": "Small nonlinear classifier (ReLU)"
            default: method
            }
        }
        public var formatLabel: String {
            switch format {
            case "activation-probe-v1": "Portable probe"
            case "native-reading-probe": "Existing Mac reader"
            case "python-reading-probe": "Existing Python reader"
            default: format
            }
        }
    }

    public struct InspectionIssue: Codable, Sendable {
        public let path: String
        public let reason: String
    }

    public struct Inventory: Codable, Sendable {
        public let probes: [Record]
        public let issues: [InspectionIssue]
        public let count: Int
    }

    public static func inventory(root: URL) async throws -> Inventory {
        try decode(await DiagnosticWorkspace.perform("probe-list", payload: ["workspaceRoot": .string(root.path)]))
    }

    public static func inspect(path: String, root: URL) async throws -> Record {
        try decode(await DiagnosticWorkspace.perform("probe-inspect", payload: ["workspaceRoot": .string(root.path), "path": .string(path)]))
    }

    public static func decode<T: Decodable>(_ value: JSONValue) throws -> T {
        try JSONDecoder().decode(T.self, from: JSONEncoder().encode(value))
    }
}
