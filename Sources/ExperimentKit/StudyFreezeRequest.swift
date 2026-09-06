import Foundation

struct StudyFreezeRequest {
    let workspaceRoot: URL
    let name: String
    let localData: Data?
    let localIsDraft: Bool
    let substrate: String
    let workspacePaired: Bool
}

@MainActor
struct StudyFreezeTransport {
    let origin: RemoteJobOrigin?
    var status: (String) async throws -> String?
    var manifestBody: (String) async throws -> Data
    var freeze: (String) async throws -> RemoteFreezeResult
    var replace: (String, Data, String) async throws -> ClusterClient.RemoteManifestReplaceResult

    init(client: ClusterClient) {
        origin = RemoteJobOrigin(connection: client.profile, workspaceRoot: WorkspaceRoot.current)
        status = { try await client.experimentDetail(name: $0).status }
        manifestBody = { try await client.experimentManifestBody(name: $0) }
        freeze = { try await client.freezeExperiment(name: $0) }
        replace = { try await client.replaceExperimentManifest(name: $0, manifestBody: $1, expectedFileSHA256: $2) }
    }

    init(
        status: @escaping (String) async throws -> String?,
        manifestBody: @escaping (String) async throws -> Data,
        freeze: @escaping (String) async throws -> RemoteFreezeResult,
        replace: @escaping (String, Data, String) async throws -> ClusterClient.RemoteManifestReplaceResult
    ) {
        self.origin = nil
        self.status = status
        self.manifestBody = manifestBody
        self.freeze = freeze
        self.replace = replace
    }
}

@MainActor
struct StudyFreezePresentation {
    var note: (String, PanelNotice.Severity) -> Void = { _, _ in }
    var refresh: () -> Void = {}
    var residency: (String, Bool) -> Void = { _, _ in }
}
