import Foundation

/// Manifest filesystem access bound to one explicit workspace.
/// Admission and lifecycle transitions remain in ExperimentStore; this owner
/// neither resolves a process-wide root nor decides whether a mutation is valid.
public struct ExperimentRepository: Sendable {
    public let workspaceRoot: URL

    public init(workspaceRoot: URL) {
        self.workspaceRoot = workspaceRoot
    }

    public var directory: URL {
        workspaceRoot.appending(component: "experiments")
    }

    func manifestURL(_ name: String) -> URL {
        directory.appending(components: name, "experiment.json")
    }

    public func list() -> [ExperimentManifest] {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isDirectoryKey])
        else { return [] }
        return entries
            .compactMap { try? load(name: $0.lastPathComponent) }
            .sorted { $0.createdAt > $1.createdAt }
    }

    public func snapshot(name: String) throws -> ManifestFileSnapshot {
        guard !name.isEmpty, name != ".", name != "..",
            !name.contains("/"), !name.contains("\\"), !name.contains("\0")
        else { throw ExperimentError(reason: "experiment name must be one path component") }
        return try ManifestFileTransaction.snapshot(at: manifestURL(name))
    }

    public func load(name: String) throws -> ExperimentManifest {
        let data = try Data(contentsOf: manifestURL(name))
        return try JSONDecoder().decode(ExperimentManifest.self, from: data)
    }

    /// Internal persistence primitive. The lifecycle owner must admit the
    /// transition before calling this; this is not an alternative public save API.
    func persistAdmitted(_ manifest: ExperimentManifest) throws {
        let url = manifestURL(manifest.name)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: url, options: .atomic)
    }
}
