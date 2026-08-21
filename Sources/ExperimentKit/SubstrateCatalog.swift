import Foundation
import Observation
import SteeringKit

/// A substrate-neutral view of one persisted steering-vector artifact — the
/// provenance fields the freshness index compares, normalized from either the
/// local sidecar enumeration (`VectorCatalog`) or the server's catalog
/// (`GET /api/vectors`).
public struct SubstrateVectorRecord: Sendable, Equatable, Identifiable {
    /// Stable artifact identity: local = `<runDir>/<name>` path; remote = the
    /// server-reported id (same shape, server-side path).
    public var id: String
    public var concept: String
    public var modelID: String
    public var revision: String?
    /// Nil when the substrate's catalog omits it (older servers do) — an
    /// unverifiable pin, which the freshness index treats as stale, not fresh.
    public var stimulusSetHash: String?
    /// Norm-unit denominator provenance (grand-mean/neutral-calibrated
    /// artifacts); carried for display, not yet a classified pin.
    public var neutralCorpusHash: String?
    /// Method identity: the recipe method when stamped (grand-mean artifacts
    /// carry no contrastive `extractionMethod`), else the extraction method.
    public var extractionMethod: String?
    public var readingPosition: String?
    /// ISO-8601 (or date-only) extraction stamp; string comparison orders it.
    public var extractionDate: String?

    public init(
        id: String, concept: String, modelID: String, revision: String? = nil,
        stimulusSetHash: String? = nil, neutralCorpusHash: String? = nil,
        extractionMethod: String? = nil,
        readingPosition: String? = nil, extractionDate: String? = nil
    ) {
        self.id = id
        self.concept = concept
        self.modelID = modelID
        self.revision = revision
        self.stimulusSetHash = stimulusSetHash
        self.neutralCorpusHash = neutralCorpusHash
        self.extractionMethod = extractionMethod
        self.readingPosition = readingPosition
        self.extractionDate = extractionDate
    }

    public init(local artifact: VectorArtifact) {
        let sidecar = artifact.sidecar
        self.init(
            id: artifact.id,
            concept: sidecar.concept,
            modelID: sidecar.modelID,
            revision: sidecar.revision,
            stimulusSetHash: sidecar.stimulusSetHash,
            neutralCorpusHash: sidecar.neutralCorpusHash,
            extractionMethod: sidecar.recipeMethod ?? sidecar.extractionMethod,
            readingPosition: sidecar.readingPosition,
            extractionDate: sidecar.extractionDate)
    }

    public init(remote record: RemoteVectorRecord) {
        self.init(
            id: record.id,
            concept: record.concept,
            modelID: record.modelID,
            revision: record.revision,
            stimulusSetHash: record.stimulusSetHash,
            neutralCorpusHash: record.neutralCorpusHash,
            extractionMethod: record.resolvedMethod,
            readingPosition: record.resolvedReadingPosition,
            extractionDate: record.resolvedExtractionDate)
    }
}

/// Availability + freshness index over the compute substrates: which models
/// are installed and which derived vector artifacts exist *on the workspace
/// you are looking at*, and whether an artifact still matches the pins a run
/// would use today. Fed by `ClusterConnectionStore` (active-server state) and
/// `VectorCatalog` (local runs tree); pure classification is a static
/// function so it is unit-testable with synthetic records.
///
/// Scope rule (substrate-as-workspace): artifacts and models are per-substrate
/// and never merge across workspaces; recipes (concepts, stimuli, manifests)
/// are git-versioned and visible everywhere, so they are not indexed here.
/// Inactive servers are never polled — their queries return empty rather than
/// stale merged data.
@Observable @MainActor
public final class SubstrateCatalog {

    /// Freshness verdict for a requested (concept, model, pins) tuple.
    public enum ArtifactMatch: Sendable, Equatable {
        case fresh(SubstrateVectorRecord)
        case stale(SubstrateVectorRecord, reason: String)
    }

    /// Canonical staleness reasons (also the UI strings).
    public enum StaleReason {
        public static let stimuliChanged = "stimuli changed since extraction (hash mismatch)"
        public static let methodChanged = "different extraction method/options"
        public static let revisionChanged = "different model revision"
    }

    private let store: ClusterConnectionStore

    /// Local artifacts, refreshed from the runs tree on demand (`refresh`).
    public private(set) var localVectors: [VectorArtifact] = []
    /// The active server's artifact catalog; cleared when no server is active.
    public private(set) var remoteVectors: [RemoteVectorRecord] = []
    /// Last remote-catalog fetch failure, for a status row; nil on success.
    public private(set) var remoteVectorsError: String?

    public init(store: ClusterConnectionStore) {
        self.store = store
        // A server-side workspace switch repoints which tree the server's
        // catalog enumerates: refetch the remote artifact list so pickers
        // never keep offering the OLD workspace's vectors.
        store.onServerScopeInvalidated { [weak self] in
            guard let self else { return }
            Task { await self.refreshRemoteVectors() }
        }
    }

    // MARK: Refresh

    /// Re-enumerate local sidecars and, when a server workspace is active,
    /// re-fetch its vector catalog.
    public func refresh() async {
        refreshLocalVectors()
        await refreshRemoteVectors()
    }

    public func refreshLocalVectors() {
        localVectors = VectorCatalog.scan()
    }

    /// Fetch the *active* server's catalog. No-op (cleared) when the Local
    /// workspace is active — inactive servers are never polled.
    public func refreshRemoteVectors() async {
        guard case .server = store.activeWorkspace, let client = store.client else {
            remoteVectors = []
            remoteVectorsError = nil
            return
        }
        do {
            remoteVectors = try await client.vectorArtifacts()
            remoteVectorsError = nil
        } catch {
            remoteVectors = []
            remoteVectorsError = "could not list server vectors: \(error.localizedDescription)"
        }
    }

    // MARK: Availability

    /// Installed, loadable model ids for a workspace. Local = the app's
    /// pinned model tiers (`ChatService.availableModels` — a hardcoded
    /// candidate list, kept as the local source for now, not an HF-cache
    /// scan). Server = the active server's reported inventory; an *inactive*
    /// server returns empty (never polled, no last-known inventory kept).
    public func installedModels(for workspace: ClusterConnectionStore.Workspace) -> [String] {
        switch workspace {
        case .local:
            return ChatService.availableModels.map(\.id)
        case .server(let id):
            guard store.activeWorkspace == .server(id) else { return [] }
            return store.remoteState?.models ?? []
        }
    }

    /// Vector artifacts visible in a workspace, normalized for freshness
    /// classification. Same inactive-server rule as `installedModels`.
    public func vectorArtifacts(
        for workspace: ClusterConnectionStore.Workspace
    ) -> [SubstrateVectorRecord] {
        switch workspace {
        case .local:
            return localVectors.map(SubstrateVectorRecord.init(local:))
        case .server(let id):
            guard store.activeWorkspace == .server(id) else { return [] }
            return remoteVectors.map(SubstrateVectorRecord.init(remote:))
        }
    }

    // MARK: Freshness

    /// Does the active workspace already hold a usable vector for these pins?
    /// `.fresh` = every supplied pin verifiably matches; `.stale` = the
    /// concept+model artifact exists but a pin diverged (reason says which);
    /// nil = no artifact for that concept+model at all.
    public func freshVector(
        concept: String,
        modelID: String,
        revision: String? = nil,
        stimulusHash: String? = nil,
        method: String? = nil
    ) -> ArtifactMatch? {
        Self.classify(
            records: vectorArtifacts(for: store.activeWorkspace),
            concept: concept, modelID: modelID,
            revision: revision, stimulusHash: stimulusHash, method: method)
    }

    /// Pure classification over already-fetched records (unit-tested with
    /// synthetic records; no networking).
    ///
    /// Semantics: candidates must match concept + modelID exactly. Each
    /// supplied (non-nil) pin must be *verifiably* equal on the record —
    /// a record missing that field counts as a mismatch, because "fresh"
    /// means "safe to reuse without re-extraction", and unverifiable is not
    /// that. A nil pin is "don't care". Among several candidates the newest
    /// fresh one wins; otherwise the newest candidate is reported stale with
    /// the first divergent pin (stimuli → method → revision).
    public static func classify(
        records: [SubstrateVectorRecord],
        concept: String,
        modelID: String,
        revision: String? = nil,
        stimulusHash: String? = nil,
        method: String? = nil
    ) -> ArtifactMatch? {
        let candidates = records
            .filter { $0.concept == concept && $0.modelID == modelID }
            .sorted { ($0.extractionDate ?? "") > ($1.extractionDate ?? "") }
        guard !candidates.isEmpty else { return nil }
        var firstStale: ArtifactMatch?
        for record in candidates {
            if let reason = stalenessReason(
                record, revision: revision, stimulusHash: stimulusHash, method: method)
            {
                if firstStale == nil { firstStale = .stale(record, reason: reason) }
            } else {
                return .fresh(record)
            }
        }
        return firstStale
    }

    private static func stalenessReason(
        _ record: SubstrateVectorRecord,
        revision: String?,
        stimulusHash: String?,
        method: String?
    ) -> String? {
        if let stimulusHash, record.stimulusSetHash != stimulusHash {
            return StaleReason.stimuliChanged
        }
        if let method, record.extractionMethod != method {
            return StaleReason.methodChanged
        }
        if let revision, record.revision != revision {
            return StaleReason.revisionChanged
        }
        return nil
    }
}
