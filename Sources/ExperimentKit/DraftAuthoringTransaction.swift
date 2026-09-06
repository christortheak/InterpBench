import Foundation

/// A reviewed document and its destination. The file digest is transport and
/// authoring metadata, never a member of ExperimentManifest or its encoding.
public struct DraftAuthoringSnapshot: Sendable {
    public let workspaceRoot: URL
    public let manifest: ExperimentManifest
    public let file: ManifestFileSnapshot

    public init(workspaceRoot: URL, name: String) throws {
        let storage = ExperimentRepository(workspaceRoot: workspaceRoot)
        let file = try storage.snapshot(name: name)
        try self.init(workspaceRoot: workspaceRoot, name: name, file: file)
    }

    public init(workspaceRoot: URL, name: String, file: ManifestFileSnapshot) throws {
        let manifest = try JSONDecoder().decode(ExperimentManifest.self, from: file.data)
        guard manifest.name == name else {
            throw ExperimentError(reason: "manifest name does not match its authoring destination")
        }
        self.workspaceRoot = workspaceRoot.standardizedFileURL
        self.manifest = manifest
        self.file = file
    }
}

/// Shared admission for a whole-document draft replacement from a reviewed
/// snapshot. Callers must reload and review after a refusal, never retry with a
/// newly fetched tag behind the researcher's back.
public enum DraftAuthoringTransaction {
    @discardableResult
    public static func replace(
        _ replacement: ExperimentManifest, reviewed: DraftAuthoringSnapshot,
        mayClearArms: Bool = false
    ) throws -> DraftAuthoringSnapshot {
        let name = reviewed.manifest.name
        guard replacement.name == name else {
            throw ExperimentError(reason: "draft replacement cannot rename its destination")
        }
        let storage = ExperimentRepository(workspaceRoot: reviewed.workspaceRoot)
        return try ManifestFileTransaction.withLock(
            manifestURL: storage.manifestURL(name), workspaceRoot: reviewed.workspaceRoot
        ) {
            try ManifestFileTransaction.requireCurrent(.sha256(reviewed.file.sha256), at: storage.manifestURL(name))
            try ManifestMutationPolicy.admitDraftEdit(reviewed.manifest)
            try ManifestMutationPolicy.admitDraftEdit(replacement)
            try ExperimentStore.save(
                replacement, mayClearArms: mayClearArms, workspaceRoot: reviewed.workspaceRoot,
                expectedFile: .sha256(reviewed.file.sha256))
            return try DraftAuthoringSnapshot(workspaceRoot: reviewed.workspaceRoot, name: name)
        }
    }
}
