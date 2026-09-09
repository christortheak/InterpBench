import CryptoKit
import Foundation

public struct ArtifactImportPlan: Codable, Sendable {
    public var kind: String
    public var modelID: String
    public var modelRevision: String?
    public var workspaceRoot: String
    public var planSHA256: String
    public var warnings: [String]
    public var details: [String: JSONValue]
}

/// File selection is metadata-only. The engine validates tensor semantics.
public struct ArtifactImportSelection: Sendable {
    public struct SourceFile: Sendable {
        public let url: URL
        public let relativePath: String
    }
    public let description: URL
    public let kind: String
    public let modelID: String
    public let files: [SourceFile]

    public static func read(_ url: URL, expectedKind: String) throws -> Self {
        struct Description: Decodable {
            let kind: String
            let modelID: String
            let tensorFile: String
            let configFile: String?
        }
        let root = url.deletingLastPathComponent().resolvingSymlinksInPath()
        let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
        guard ((attributes[.size] as? NSNumber)?.intValue ?? Int.max) <= 1_048_576 else {
            throw ExperimentError(reason: "Choose the small JSON description, not the tensor file.")
        }
        let value = try JSONDecoder().decode(Description.self, from: Data(contentsOf: url))
        guard value.kind == expectedKind else {
            throw ExperimentError(reason: "This description is for \(value.kind); choose one for \(expectedKind).")
        }
        var files = [SourceFile(url: url, relativePath: "description.json")]
        for name in [value.tensorFile, value.configFile].compactMap({ $0 }) {
            let parts = name.split(separator: "/", omittingEmptySubsequences: false)
            guard !name.isEmpty, !name.hasPrefix("/"), !name.contains("\\"),
                  !parts.contains(where: { $0.isEmpty || $0 == "." || $0 == ".." }),
                  name != "description.json" else {
                throw ExperimentError(reason: "Source paths must stay inside the description folder and use distinct filenames.")
            }
            let source = root.appending(path: name)
            guard source.resolvingSymlinksInPath().path == source.standardizedFileURL.path else {
                throw ExperimentError(reason: "Choose source files without symbolic links.")
            }
            files.append(SourceFile(url: source, relativePath: name))
        }
        guard Set(files.map(\.relativePath)).count == files.count else {
            throw ExperimentError(reason: "The description must name distinct source files.")
        }
        return Self(description: url, kind: value.kind, modelID: value.modelID, files: files)
    }

    public static func fileSHA256(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var hash = SHA256()
        while let chunk = try file.read(upToCount: 1_048_576), !chunk.isEmpty { hash.update(data: chunk) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

extension ClusterClient {
    public func artifactImportPlan(descriptionFile: String) async throws -> ArtifactImportPlan {
        try await post("/api/artifact-imports/plan", body: ["descriptionFile": descriptionFile], timeout: 1800)
    }

    public func importReviewedArtifact(descriptionFile: String, planSHA256: String) async throws -> String {
        struct Reply: Decodable { let jobId: String }
        let reply: Reply = try await post("/api/artifact-imports/import",
            body: ["descriptionFile": descriptionFile, "planSHA256": planSHA256])
        return reply.jobId
    }
}
