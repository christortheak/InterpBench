import Foundation

/// Connection and destination captured when a job is submitted or explicitly
/// reconnected. Contains a Keychain reference, never a bearer token.
public struct RemoteJobOrigin: Codable, Sendable, Equatable {
    public let connection: ClusterConnectionProfile
    public let workspaceRoot: URL

    public init(connection: ClusterConnectionProfile, workspaceRoot: URL) {
        self.connection = connection
        self.workspaceRoot = workspaceRoot.standardizedFileURL
    }

    /// An origin a command line recorded in the workspace. The destination
    /// is the workspace the record was READ from: that is where this app
    /// imports to, even if the folder has moved since the submission.
    /// Nil when the recorded endpoint is not a URL.
    public init?(workspaceRecord record: WorkspaceJobOrigin, workspaceRoot: URL) {
        guard let url = URL(string: record.endpoint), url.host() != nil else { return nil }
        self.init(
            connection: ClusterConnectionProfile(
                name: record.serverDisplayName, baseURL: url,
                serverIdentity: record.serverIdentity),
            workspaceRoot: workspaceRoot)
    }

    /// Whether `client` is the server this job belongs to. When both sides
    /// know the durable server identity, that decides — an SSH site's tunnel
    /// may come back on another local port, and a command line never knows
    /// the app's Keychain reference. Otherwise the endpoint and token
    /// reference must both agree, as they always have.
    public func matches(_ client: ClusterClient) -> Bool {
        if let recorded = connection.serverIdentity, !recorded.isEmpty,
            let connected = client.profile.serverIdentity, !connected.isEmpty
        {
            return recorded == connected
        }
        return connection.baseURL == client.profile.baseURL
            && connection.tokenKey == client.profile.tokenKey
    }
}
