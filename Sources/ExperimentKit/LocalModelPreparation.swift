import Foundation
import SteeringKit

public struct LocalModelPreparationError: Error, LocalizedError, Sendable {
    public let code: String
    public let reason: String
    public let repairAction: String
    public var errorDescription: String? { reason }

    static func requestChanged() -> LocalModelPreparationError {
        .init(code: "modelInstallationChanged", reason: "The observed model installation is no longer current.",
            repairAction: "Inspect local model-install status and review the current request before cancelling or waiting for it.")
    }

}

/// Local cache observation and the wire view of the app's existing installer.
/// Cache file presence is not a model load, memory-fit test or scientific test.
public enum LocalModelPreparation {
    public struct Plan: Encodable, Sendable {
        public let target = "localMLX"
        public let modelID: String
        public let requestedRevision: String
        public let cacheFileSetPresent: Bool
        public let snapshotPath: String?
        public let resolvedRevision: String?
        public let memoryFit = "notChecked"
        public let credentials = "notChecked"
        public let downloadBytes: Int? = nil
        public let note = "This checks the local cache only. Installation may download model files; it does not load weights, prove memory fit or qualify scientific results. Remote preparation uses the selected server's model-install operation and policy."
    }

    public struct Status: Encodable, Sendable {
        public let target = "localMLX"
        public let request: LocalModelInstaller.Request?
        public let state: String
        public let modelID: String?
        public let percent: Int?
        public let reason: String?

        @MainActor public init(_ installer: LocalModelInstaller) {
            request = installer.request
            switch installer.phase {
            case .idle: state = "idle"; modelID = nil; percent = nil; reason = nil
            case .installing(let model, let progress): state = "installing"; modelID = model; percent = progress; reason = nil
            case .finished(let model): state = "finished"; modelID = model; percent = 100; reason = nil
            case .cancelled(let model): state = "cancelled"; modelID = model; percent = nil; reason = "Partial cache files are retained for a later install."
            case .failed(let model, let failure): state = "failed"; modelID = model; percent = nil; reason = failure
            }
        }
    }

    public static func plan(modelID: String, revision: String? = nil, cacheRoot: URL? = nil) throws -> Plan {
        try validate(modelID: modelID, revision: revision)
        let model = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        let snapshot = SteeredContainerLoader.cachedLoadableSnapshot(modelID: model, revision: revision, cacheRoot: cacheRoot)
        return Plan(modelID: model, requestedRevision: revision ?? "main", cacheFileSetPresent: snapshot != nil,
            snapshotPath: snapshot?.path, resolvedRevision: snapshot?.lastPathComponent)
    }

    static func validate(modelID: String, revision: String?) throws {
        if let reason = LocalModelInstaller.installRefusal(slug: modelID, isInstalling: false, revision: revision) {
            throw LocalModelPreparationError(code: "invalidModelInstallRequest", reason: reason,
                repairAction: "Supply the intended owner/repo and an optional commit, branch or tag. Use model plan before installing.")
        }
    }


    @MainActor public static func start(modelID: String, revision: String? = nil, installer: LocalModelInstaller) throws -> Status {
        try validate(modelID: modelID, revision: revision)
        if let reason = installer.install(modelID, revision: revision) {
            throw LocalModelPreparationError(code: "modelInstallationBusy", reason: reason,
                repairAction: "Observe the current installation or explicitly cancel its request ID before starting another.")
        }
        return Status(installer)
    }
}
