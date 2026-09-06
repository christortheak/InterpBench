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

    /// A transport's external version must identify the study it actually
    /// reviewed. This read also supports frozen source studies; mutation owners
    /// decide which lifecycle states their operations admit.
    public static func review(name: String, workspaceRoot: URL, expectedFileSHA256: String) throws -> Self {
        guard expectedFileSHA256.count == 64,
            expectedFileSHA256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw ExperimentError.malformed("A reviewed study precondition must be a lowercase SHA-256 digest.",
                repair: "Use manifestFileSHA256 from the named study inspection result.")
        }
        let study = try Self(workspaceRoot: workspaceRoot, name: name)
        guard study.file.sha256 == expectedFileSHA256 else {
            throw ExperimentError.refusing(.staleManifest, "The study changed after inspection.",
                repair: "Inspect the named study and review the changes before reconstructing the intended operation.")
        }
        return study
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
    /// Admit a synchronous store command against the exact document the caller
    /// reviewed. Store policy remains authoritative. The active-root guard is
    /// required for store commands that still resolve workspace-scoped inputs.
    public static func perform<T>(
        reviewed: DraftAuthoringSnapshot, draftOnly: Bool = true,
        _ command: (String) throws -> T
    ) throws -> T {
        guard ExperimentStore.workspaceRoot.standardizedFileURL == reviewed.workspaceRoot else {
            throw ExperimentError.refusing(.staleManifest, "The authoring workspace changed.",
                repair: "Return to the reviewed workspace and reload the study before editing.")
        }
        let storage = ExperimentRepository(workspaceRoot: reviewed.workspaceRoot)
        return try ManifestFileTransaction.withLock(
            manifestURL: storage.manifestURL(reviewed.manifest.name), workspaceRoot: reviewed.workspaceRoot
        ) {
            try ManifestFileTransaction.requireCurrent(.sha256(reviewed.file.sha256),
                at: storage.manifestURL(reviewed.manifest.name))
            if draftOnly { try ManifestMutationPolicy.admitDraftEdit(reviewed.manifest) }
            return try command(reviewed.manifest.name)
        }
    }

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
