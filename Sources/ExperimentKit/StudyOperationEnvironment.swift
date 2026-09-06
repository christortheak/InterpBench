import Foundation

/// Runtime inputs supplied by a surface, without retaining its panel or loading credentials.
/// Controllers capture these values before an operation and reject a changed destination.
@MainActor
public struct StudyOperationContext {
    let workspaceRoot: URL
    let selectedName: String?
    let selectedIsDraft: Bool
    let serverURL: String?
    let isServer: Bool
    let pairing: WorkspaceScoping.ServerPairing?
    let substrate: String
    let capabilities: ClusterCapabilities?
    let client: ClusterClient?
    let hasDisplay: Bool
    let serverOrigin: EvidenceImportOrigin?

    public init(
        workspaceRoot: URL, selectedName: String?, selectedIsDraft: Bool,
        serverURL: String?, isServer: Bool, pairing: WorkspaceScoping.ServerPairing?,
        substrate: String, capabilities: ClusterCapabilities?, client: ClusterClient?,
        hasDisplay: Bool, serverOrigin: EvidenceImportOrigin? = nil
    ) {
        self.workspaceRoot = workspaceRoot.standardizedFileURL
        self.selectedName = selectedName
        self.selectedIsDraft = selectedIsDraft
        self.serverURL = serverURL
        self.isServer = isServer
        self.pairing = pairing
        self.substrate = substrate
        self.capabilities = capabilities
        self.client = client
        self.hasDisplay = hasDisplay
        self.serverOrigin = serverOrigin
    }

    func matches(_ other: Self, selection: Bool) -> Bool {
        workspaceRoot == other.workspaceRoot && serverURL == other.serverURL
            && isServer == other.isServer && pairing == other.pairing
            && client?.profile == other.client?.profile && serverOrigin == other.serverOrigin
            && (!selection || selectedName == other.selectedName)
    }

    func manifestData(name: String) -> Data? {
        try? ExperimentRepository(workspaceRoot: workspaceRoot).snapshot(name: name).data
    }
}

/// Credential access is an explicit user-action dependency; observation only reads `current`.
@MainActor
public struct StudyOperationEnvironment {
    let current: () -> StudyOperationContext?
    let connect: () -> ClusterClient?

    public init(
        current: @escaping () -> StudyOperationContext?,
        connect: @escaping () -> ClusterClient?
    ) {
        self.current = current
        self.connect = connect
    }

    func isCurrent(_ captured: StudyOperationContext, selection: Bool = false) -> Bool {
        guard !Task.isCancelled, let latest = current() else { return false }
        return captured.matches(latest, selection: selection)
    }
}
