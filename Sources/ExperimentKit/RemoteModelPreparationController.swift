import Foundation
import Observation

/// Captures the endpoint for a preparation request before any suspension.
@MainActor @Observable public final class RemoteModelPreparationController {
    public private(set) var plan: RemoteModelPreparationPlan?
    public private(set) var endpoint: URL?
    public private(set) var requestedModelID: String?
    public private(set) var message: String?
    private var generation = UUID()

    public init() {}

    public func preview(modelID: String, revision: String?, profile: ClusterConnectionProfile,
                        client: ClusterClient, isCurrent: @escaping @MainActor () -> Bool) async {
        let request = UUID(); generation = request
        plan = nil; endpoint = profile.baseURL; requestedModelID = modelID
        message = "Inspecting the model preparation target…"
        do {
            let value = try await client.modelPreparationPlan(modelID, revision: revision)
            guard generation == request, isCurrent(), !Task.isCancelled else { return }
            plan = value
            message = "\(value.cacheFileSetPresent ? "Cached files found" : "Model files need preparation") at \(value.cacheRoot). "
                + (value.installationAllowed ? "Installation may download files; memory fit is not checked." : "This site disallows compute-node downloads; stage on its permitted transfer host.")
        } catch {
            guard generation == request, isCurrent() else { return }
            message = "Could not inspect model preparation: \(error.localizedDescription)"
        }
    }
}
