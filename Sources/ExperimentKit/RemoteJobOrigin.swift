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

    public func matches(_ client: ClusterClient) -> Bool {
        connection.baseURL == client.profile.baseURL
            && connection.tokenKey == client.profile.tokenKey
    }
}
