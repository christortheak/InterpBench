import Foundation

public struct RemoteNeutralCatalog: Decodable, Sendable {
    public struct Corpus: Decodable, Sendable, Identifiable {
        public let name: String
        public let count: Int
        public let hash: String?
        public var id: String { name }
    }
    public struct Basis: Decodable, Sendable, Identifiable {
        public let runDirectory: String
        public let modelID: String?
        public let revision: String?
        public let corpusName: String?
        public let totalComponents: Int?
        public let tokenRows: Int?
        public let layers: [Int]?
        public var id: String { runDirectory }
        public var label: String { "\(corpusName ?? "Reference") · \(totalComponents ?? 0) components · \(URL(filePath: runDirectory).lastPathComponent)" }
    }
    public let corpora: [Corpus]
    public let bases: [Basis]
}

extension ClusterClient {
    public func neutralCatalog() async throws -> RemoteNeutralCatalog {
        try await get("/api/neutral/corpora")
    }

    public func buildNeutralBasis(corpus: String, modelID: String) async throws -> String {
        struct Body: Encodable {
            let corpus: String
            let expectedModelID: String
        }
        struct Reply: Decodable { let jobId: String }
        let reply: Reply = try await post("/api/neutral-pcs/build",
            body: Body(corpus: corpus, expectedModelID: modelID))
        return reply.jobId
    }

    public struct SAEFeatureImport: Encodable, Sendable {
        public var model: String
        public var release: String
        public var saeID: String
        public var feature: Int
        public var label: String
        public var residualNormArtifact: String
        public var neuronpediaURL: String?
        public init(model: String, release: String, saeID: String, feature: Int,
                    label: String, residualNormArtifact: String, neuronpediaURL: String? = nil) {
            self.model = model; self.release = release; self.saeID = saeID
            self.feature = feature; self.label = label
            self.residualNormArtifact = residualNormArtifact; self.neuronpediaURL = neuronpediaURL
        }
    }

    public func importSAEFeature(_ request: SAEFeatureImport) async throws -> String {
        struct Reply: Decodable { let jobId: String }
        let reply: Reply = try await post("/api/gemmascope/import-id", body: request)
        return reply.jobId
    }
}

extension ClusterClient {
    public struct ResolvedSAEFeature: Decodable, Sendable {
        public let release: String
        public let saeID: String
        public let model: String
        public let feature: Int
        public let neuronpediaURL: String
    }
    public func resolveSAEFeature(url: String) async throws -> ResolvedSAEFeature {
        try await get("/api/gemmascope/resolve-feature", queryItems: [.init(name: "url", value: url)])
    }
}
