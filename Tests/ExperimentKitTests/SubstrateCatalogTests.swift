import Foundation
import Testing

@testable import ExperimentKit

/// Pure-CPU tests for the availability + freshness index: classification over
/// synthetic records (no networking, no filesystem beyond the local model
/// list) and the per-workspace scoping rules.
@MainActor
struct SubstrateCatalogTests {

    private func freshDefaults(_ name: String) throws -> UserDefaults {
        let suite = "steerlab.tests.substrate-catalog.\(name)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    private func record(
        id: String = "runs/2026-06-10T000000Z-x/french",
        concept: String = "french",
        modelID: String = "Qwen/Qwen3-4B-MLX-4bit",
        revision: String? = "rev-a",
        stimulusSetHash: String? = "hash-1",
        method: String? = "lat",
        date: String? = "2026-06-10T00:00:00Z"
    ) -> SubstrateVectorRecord {
        SubstrateVectorRecord(
            id: id, concept: concept, modelID: modelID, revision: revision,
            stimulusSetHash: stimulusSetHash, extractionMethod: method,
            extractionDate: date)
    }

    // MARK: Freshness classification (pure, synthetic records)

    @Test func freshWhenEveryPinMatches() {
        let match = SubstrateCatalog.classify(
            records: [record()],
            concept: "french", modelID: "Qwen/Qwen3-4B-MLX-4bit",
            revision: "rev-a", stimulusHash: "hash-1", method: "lat")
        #expect(match == .fresh(record()))
    }

    @Test func staleWhenStimuliChanged() {
        let match = SubstrateCatalog.classify(
            records: [record()],
            concept: "french", modelID: "Qwen/Qwen3-4B-MLX-4bit",
            revision: "rev-a", stimulusHash: "hash-2", method: "lat")
        #expect(
            match == .stale(record(), reason: "stimuli changed since extraction (hash mismatch)"))
    }

    @Test func staleWhenMethodDiffers() {
        let match = SubstrateCatalog.classify(
            records: [record()],
            concept: "french", modelID: "Qwen/Qwen3-4B-MLX-4bit",
            revision: "rev-a", stimulusHash: "hash-1", method: "meanDifference")
        #expect(match == .stale(record(), reason: "different extraction method/options"))
    }

    @Test func staleWhenRevisionDiffers() {
        let match = SubstrateCatalog.classify(
            records: [record()],
            concept: "french", modelID: "Qwen/Qwen3-4B-MLX-4bit",
            revision: "rev-b", stimulusHash: "hash-1", method: "lat")
        #expect(match == .stale(record(), reason: "different model revision"))
    }

    @Test func absentWhenNoConceptModelCandidate() {
        #expect(
            SubstrateCatalog.classify(
                records: [record()],
                concept: "sycophancy", modelID: "Qwen/Qwen3-4B-MLX-4bit") == nil)
        #expect(
            SubstrateCatalog.classify(
                records: [record()],
                concept: "french", modelID: "mlx-community/gemma-3-4b-it-4bit") == nil)
    }

    @Test func nilPinsAreDontCares() {
        // No pins supplied: any concept+model artifact is fresh.
        let match = SubstrateCatalog.classify(
            records: [record(revision: nil, stimulusSetHash: nil, method: nil)],
            concept: "french", modelID: "Qwen/Qwen3-4B-MLX-4bit")
        #expect(match == .fresh(record(revision: nil, stimulusSetHash: nil, method: nil)))
    }

    @Test func unverifiablePinIsStaleNotFresh() {
        // The record omits the stimulus hash (the server's current catalog
        // payload does): "fresh" must mean verifiably matching, so this is
        // stale, not fresh.
        let hashless = record(stimulusSetHash: nil)
        let match = SubstrateCatalog.classify(
            records: [hashless],
            concept: "french", modelID: "Qwen/Qwen3-4B-MLX-4bit",
            stimulusHash: "hash-1")
        #expect(
            match == .stale(hashless, reason: "stimuli changed since extraction (hash mismatch)"))
    }

    @Test func prefersAFreshRecordOverANewerStaleOne() {
        let staleNewer = record(id: "runs/b/french", stimulusSetHash: "hash-old",
                                date: "2026-06-12T00:00:00Z")
        let freshOlder = record(id: "runs/a/french", date: "2026-06-10T00:00:00Z")
        let match = SubstrateCatalog.classify(
            records: [staleNewer, freshOlder],
            concept: "french", modelID: "Qwen/Qwen3-4B-MLX-4bit",
            revision: "rev-a", stimulusHash: "hash-1", method: "lat")
        #expect(match == .fresh(freshOlder))
    }

    @Test func newestFreshRecordWins() {
        let older = record(id: "runs/a/french", date: "2026-06-10T00:00:00Z")
        let newer = record(id: "runs/b/french", date: "2026-06-12T00:00:00Z")
        let match = SubstrateCatalog.classify(
            records: [older, newer],
            concept: "french", modelID: "Qwen/Qwen3-4B-MLX-4bit",
            stimulusHash: "hash-1")
        #expect(match == .fresh(newer))
    }

    @Test func reasonReportsFirstDivergentPinInOrder() {
        // Both stimuli and revision diverge: stimuli is reported (the ladder
        // is stimuli → method → revision).
        let match = SubstrateCatalog.classify(
            records: [record()],
            concept: "french", modelID: "Qwen/Qwen3-4B-MLX-4bit",
            revision: "rev-b", stimulusHash: "hash-2", method: "lat")
        #expect(
            match == .stale(record(), reason: "stimuli changed since extraction (hash mismatch)"))
    }

    // MARK: Normalization

    @Test func remoteRecordNormalizationPrefersSidecarSpellings() {
        let remote = RemoteVectorRecord(
            id: "/runs/x/french", runDirectory: "/runs/x", name: "french",
            concept: "french", modelID: "Qwen/Qwen3-4B", revision: "rev-a",
            layerCount: 36, hiddenSize: 2560, method: "lat", reading: "last token",
            residualNormSource: "neutral-corpus", hasResidualNorms: true,
            extracted: "2026-06-10", stimulusSetHash: nil, extractionMethod: nil,
            readingPosition: nil, extractionDate: nil)
        let normalized = SubstrateVectorRecord(remote: remote)
        #expect(normalized.extractionMethod == "lat")
        #expect(normalized.readingPosition == "last token")
        #expect(normalized.extractionDate == "2026-06-10")
        #expect(normalized.stimulusSetHash == nil)
    }

    /// Server-shaped record with a recipeMethod (a CAA vector built
    /// server-side): method identity normalizes as recipeMethod ??
    /// extractionMethod — the SAME rule as local sidecars — so it classifies
    /// FRESH against the recipe's method. Regression for "always stale":
    /// the server catalog used to report only extractionMethod
    /// ("meanDifference"), which never matched the requested recipe method
    /// ("caaMeanDifference").
    @Test func serverRecordWithRecipeMethodClassifiesFresh() {
        let remote = RemoteVectorRecord(
            id: "/srv/runs/x/french", runDirectory: "/srv/runs/x", name: "french",
            concept: "french", modelID: "org/m", revision: "rev-a",
            layerCount: 36, hiddenSize: 2560, method: "meanDifference",
            reading: "last token", residualNormSource: nil, hasResidualNorms: false,
            extracted: "2026-07-01", stimulusSetHash: "hash-1",
            recipeMethod: "caaMeanDifference",
            extractionMethod: nil, readingPosition: nil, extractionDate: nil)
        let normalized = SubstrateVectorRecord(remote: remote)
        #expect(normalized.extractionMethod == "caaMeanDifference")

        let match = SubstrateCatalog.classify(
            records: [normalized],
            concept: "french", modelID: "org/m",
            revision: "rev-a", stimulusHash: "hash-1", method: "caaMeanDifference")
        #expect(match == .fresh(normalized))
    }

    /// Legacy server record (recipeMethod absent): normalization falls back
    /// to the flattened extraction method, so a plain-CAA request against
    /// "meanDifference" still classifies fresh — and a mismatched method
    /// still reports stale.
    @Test func legacyServerRecordFallsBackToExtractionMethod() {
        let remote = RemoteVectorRecord(
            id: "/srv/runs/y/french", runDirectory: "/srv/runs/y", name: "french",
            concept: "french", modelID: "org/m", revision: "rev-a",
            layerCount: 36, hiddenSize: 2560, method: "meanDifference",
            reading: nil, residualNormSource: nil, hasResidualNorms: false,
            extracted: "2026-07-01", stimulusSetHash: "hash-1",
            extractionMethod: nil, readingPosition: nil, extractionDate: nil)
        let normalized = SubstrateVectorRecord(remote: remote)
        #expect(normalized.extractionMethod == "meanDifference")

        let fresh = SubstrateCatalog.classify(
            records: [normalized],
            concept: "french", modelID: "org/m",
            revision: "rev-a", stimulusHash: "hash-1", method: "meanDifference")
        #expect(fresh == .fresh(normalized))

        let stale = SubstrateCatalog.classify(
            records: [normalized],
            concept: "french", modelID: "org/m",
            revision: "rev-a", stimulusHash: "hash-1", method: "caaMeanDifference")
        #expect(
            stale == .stale(normalized, reason: "different extraction method/options"))
    }

    // MARK: Workspace scoping

    @Test func localInstalledModelsComeFromTheAppModelList() throws {
        let store = clusterStore(defaults: try freshDefaults("local-models"))
        let catalog = SubstrateCatalog(store: store)
        let models = catalog.installedModels(for: .local)
        #expect(models == ChatService.availableModels.map(\.id))
        #expect(!models.isEmpty)
    }

    /// Availability on Local is the HF-cache scan, NOT the pinned tier list:
    /// on a fresh Mac every tier is offerable and none is installed, which is
    /// exactly the distinction the Playground's Load gate rests on.
    @Test func localAvailabilityComesFromTheCacheScan() throws {
        let store = clusterStore(defaults: try freshDefaults("local-availability"))
        let tier = ChatService.availableModels[0].id
        let catalog = SubstrateCatalog(store: store, localCacheScan: { [tier] })

        // Nothing scanned yet: offerable, but nothing claimed installed.
        #expect(catalog.installedModels(for: .local) == ChatService.availableModels.map(\.id))
        #expect(!catalog.isInstalled(tier, in: .local))

        catalog.refreshLocalInstalledModels()
        #expect(catalog.isInstalled(tier, in: .local))
        #expect(!catalog.isInstalled(ChatService.availableModels[1].id, in: .local))
    }

    /// A model installed by slug joins the SAME offerable list the builders
    /// read — appended after the pinned tiers, never a second registry.
    @Test func slugInstalledModelJoinsTheOfferableList() throws {
        let store = clusterStore(defaults: try freshDefaults("local-extras"))
        let tier = ChatService.availableModels[0].id
        let catalog = SubstrateCatalog(
            store: store,
            localCacheScan: { ["vendor-a/model-small-4bit", tier] })
        catalog.refreshLocalInstalledModels()

        let options = catalog.installedModels(for: .local)
        #expect(options.prefix(ChatService.availableModels.count)
            == ArraySlice(ChatService.availableModels.map(\.id)))
        #expect(options.last == "vendor-a/model-small-4bit")
        // The pinned tier is not duplicated by also being in the cache.
        #expect(options.filter { $0 == ChatService.availableModels[0].id }.count == 1)
        #expect(catalog.isInstalled("vendor-a/model-small-4bit", in: .local))
    }

    @Test func serverInstalledModelsReadActiveRemoteStateOnly() throws {
        let store = clusterStore(defaults: try freshDefaults("server-models"))
        let a = store.addServer(name: "A", urlString: "http://gpu-a:8080")
        let b = store.addServer(name: "B", urlString: "http://gpu-b:8080")
        let catalog = SubstrateCatalog(store: store)

        store.activeWorkspace = .server(a.id)
        store.remoteState = RemoteState(
            models: ["Qwen/Qwen3-4B"], loadedModel: nil, loadedRevision: nil,
            device: nil, isBusy: false, loadedModels: [], jobs: [])
        #expect(catalog.installedModels(for: .server(a.id)) == ["Qwen/Qwen3-4B"])
        // Inactive servers are never polled: no data rather than stale data.
        #expect(catalog.installedModels(for: .server(b.id)).isEmpty)
        #expect(catalog.vectorArtifacts(for: .server(b.id)).isEmpty)
    }
}
