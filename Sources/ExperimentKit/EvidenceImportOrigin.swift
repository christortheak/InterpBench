import Foundation

/// Durable identity for remote evidence. SSH identity uses the remote endpoint
/// and login, never a reusable tunnel-local port or a friendly display name.
/// Only references and paths live here; no credential values are captured.
public struct EvidenceImportOrigin: Codable, Sendable, Equatable, Hashable {
    public let serverIdentity: String
    public let remoteRoot: String?
    public let workspaceRoot: URL

    public init(serverIdentity: String, remoteRoot: String?, workspaceRoot: URL) {
        self.serverIdentity = serverIdentity
        self.remoteRoot = remoteRoot
        self.workspaceRoot = workspaceRoot.standardizedFileURL.resolvingSymlinksInPath()
    }

    /// An unknown serving root is observable, but cannot establish durable
    /// identity for deduplication or any later cleanup decision.
    public var isComplete: Bool {
        !serverIdentity.isEmpty && remoteRoot?.hasPrefix("/") == true
    }

    public static let changedRepair =
        "Reconnect to the originating server and serving root, select the original local workspace, then refresh the evidence list before retrying."
}

public struct EvidenceImportKey: Sendable, Hashable {
    public let origin: EvidenceImportOrigin
    public let bundlePath: String
    public let sha256: String?

    public init(candidate: EvidenceCandidate, origin: EvidenceImportOrigin) {
        self.origin = origin
        self.bundlePath = candidate.bundlePath
        self.sha256 = candidate.sha256
    }
}

extension ClusterConnectionStore {
    /// Snapshot construction never reads a Keychain secret or authenticates.
    public var evidenceImportOrigin: EvidenceImportOrigin? {
        guard let entry = activeServer else { return nil }
        return EvidenceImportOrigin(
            serverIdentity: Self.registryKey(forEntry: entry),
            remoteRoot: activeServerServingRoot,
            workspaceRoot: VectorCatalog.projectRoot)
    }
}
