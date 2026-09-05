import Foundation

extension ExperimentPanel {
    public var freezeReadiness: ExperimentStore.FreezeReadiness? {
        freezeCoordinator.freezeReadiness
    }
    public var remoteFreezeGateFailure: String? { freezeCoordinator.remoteFreezeGateFailure }
    public var remoteFreezeAdvisories: [String] { freezeCoordinator.remoteFreezeAdvisories }
    public var remoteFreezeIdentityWarning: String? {
        freezeCoordinator.remoteFreezeIdentityWarning
    }
    public var remoteFreezeIdentityNote: String? { freezeCoordinator.remoteFreezeIdentityNote }
    public var remoteFreezeCanSyncDraft: Bool { freezeCoordinator.remoteFreezeCanSyncDraft }
    public var isSyncingServerDraft: Bool { freezeCoordinator.isSyncingServerDraft }
    public var serverFreezeCrossSubstrateAdvisory: String? {
        freezeCoordinator.serverFreezeCrossSubstrateAdvisory
    }

    public func freeze() {
        freezeCoordinator.freeze(name: selectedName, runSubstrate: freezeEvidenceRunSubstrate)
    }

    private var freezeContextIdentity: StudyFreezeContextIdentity {
        StudyFreezeContextIdentity(
            study: selectedName, root: VectorCatalog.projectRoot.standardizedFileURL,
            serverURL: cluster?.serverURL, isServer: isServerWorkspace,
            pairing: cluster?.activeServerPairing)
    }

    private func freezeRequest(name: String) -> StudyFreezeRequest {
        StudyFreezeRequest(
            name: name, localData: ExperimentStore.manifestData(name: name),
            localIsDraft: selected?.status == .draft,
            substrate: cluster?.substrateLabel ?? "server",
            workspacePaired: cluster?.activeServerPairing == .paired)
    }

    public func freezeOnActiveServer() async {
        guard let name = selectedName else {
            note("select a study first", severity: .info)
            return
        }
        guard isServerWorkspace else {
            note(
                "no server workspace active — switch the substrate selector first", severity: .info)
            return
        }
        cluster?.loadStoredToken()
        guard let client = cluster?.client else {
            note("invalid server URL", severity: .error)
            return
        }
        let context = freezeContextIdentity
        await freezeCoordinator.freezeOnServer(
            request: freezeRequest(name: name), transport: StudyFreezeTransport(client: client),
            isCurrent: { [weak self] in self?.freezeContextIdentity == context })
    }

    public func pushManifestToActiveServer() async {
        guard let name = selectedName, let manifest = selected else {
            note("select a study first", severity: .info)
            return
        }
        guard manifest.status == .draft else {
            note(
                "only a DRAFT manifest can be pushed as the server's copy — "
                    + "frozen studies are read-only; duplicate to iterate", severity: .info)
            return
        }
        cluster?.loadStoredToken()
        guard let client = cluster?.client else {
            note("invalid server URL", severity: .error)
            return
        }
        let context = freezeContextIdentity
        await freezeCoordinator.pushManifest(
            request: freezeRequest(name: name), transport: StudyFreezeTransport(client: client),
            isCurrent: { [weak self] in self?.freezeContextIdentity == context })
    }
}

private struct StudyFreezeContextIdentity: Equatable {
    let study: String?
    let root: URL
    let serverURL: String?
    let isServer: Bool
    let pairing: WorkspaceScoping.ServerPairing?
}
