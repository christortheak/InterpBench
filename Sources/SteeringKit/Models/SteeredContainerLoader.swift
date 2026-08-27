import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

/// Loads a `ModelContainer` whose model is one of the vendored, hookable
/// implementations. All Hugging Face wiring (downloader + tokenizer loader,
/// per mlx-swift-lm 3.x decoupling) lives here so experiment code never
/// touches it. Weights land in the default HF cache (`~/.cache/huggingface`),
/// never in the project folder.
public enum SteeredContainerLoader {

    public static func load(
        modelID: String,
        revision: String? = nil,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        let configuration =
            if let revision {
                ModelConfiguration(id: modelID, revision: revision)
            } else {
                ModelConfiguration(id: modelID)
            }
        return try await SteeredModels.factory.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: configuration,
            progressHandler: progressHandler
        )
    }

    /// Download a model's snapshot into the HF cache WITHOUT loading it into
    /// MLX — the "install" half of `load`, so an install can be offered as its
    /// own visible, cancellable operation instead of happening invisibly
    /// inside a button called Load. Same cache, same layout, same file set as
    /// the load path (mlx-swift-lm's `modelDownloadPatterns`, which is
    /// `package`-scoped upstream and therefore restated here), so a completed
    /// install makes `isCached` true and the next `load` a pure cache read.
    ///
    /// Cancellable: the task's cancellation propagates through the hub
    /// client's URLSession work.
    @discardableResult
    public static func downloadSnapshot(
        modelID: String,
        revision: String? = nil,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> URL {
        let downloader: any Downloader = #hubDownloader()
        return try await downloader.download(
            id: modelID,
            revision: revision,
            matching: modelDownloadPatterns,
            useLatest: false,
            progressHandler: progressHandler)
    }

    /// The file set a load resolves — weights plus the tokenizer/config JSON
    /// (and Jinja templates). Mirrors mlx-swift-lm's `modelDownloadPatterns`.
    static let modelDownloadPatterns = ["*.safetensors", "*.json", "*.jinja"]

    /// Is this model already in the local HF cache — i.e. can it be loaded
    /// without touching the network? The ONE local installed-model test:
    /// membership in the same enumeration `localModelIDs` reports, so a
    /// picker's "installed" badge and a load's refusal can never disagree.
    public static func isCached(modelID: String, cacheRoot override: URL? = nil) -> Bool {
        localModelIDs(cacheRoot: override).contains(modelID)
    }

    /// The files `LLMModelFactory._load` OPENS, and therefore the ones whose
    /// presence is what "cached" has to mean at a load site.
    ///
    /// `config.json` is read with `Data(contentsOf:)` and its absence is a
    /// hard `configurationFileError`; `AutoTokenizer.from(modelFolder:)` reads
    /// `tokenizer_config.json` and `tokenizer.json` (there is no
    /// sentencepiece-only path through that loader). `generation_config.json`
    /// is deliberately ABSENT from this list — the factory reads it with
    /// `try?`, so a snapshot without it loads fine and demanding it would
    /// refuse installs that work.
    static let requiredSnapshotFiles = [
        "config.json", "tokenizer_config.json", "tokenizer.json",
    ]

    /// Can the EXACT requested revision of this model be loaded from the
    /// local cache, with no network?
    ///
    /// The repo-id-only `isCached` answers a WEAKER question than its callers
    /// were asking it (external review round 8, finding 3). It passes when
    /// `refs/main` or a `snapshots` path merely EXISTS, so:
    ///
    /// - a cache holding revision A passed for a judge pinned to revision B,
    ///   and the load that followed — `load(modelID:revision: B)` — went
    ///   straight to the network for B. The guard's whole promise, that a
    ///   judging run never downloads weights on your behalf, was false
    ///   exactly whenever a study pinned a revision;
    /// - a HALF-downloaded snapshot passed: `snapshots/` existing is not
    ///   `snapshots/<sha>/` holding weights, and an interrupted install (or
    ///   symlinks pointing at blobs that were never fetched) then either
    ///   re-downloaded or died inside MLX's weight verification with an
    ///   error no refusal had prepared anyone for.
    ///
    /// So this resolves the revision the way the hub client does — a 40-hex
    /// commit addresses `snapshots/<commit>` directly, anything else (and
    /// `nil`, which the loader turns into `"main"`) is resolved through
    /// `refs/<ref>` — and then proves the snapshot holds the file set above
    /// plus its weights. Everything here is `stat`-level except the shard
    /// index, which is a few KB and is the only way "the weights are all
    /// here" can be answered for a sharded model at all.
    ///
    /// The repo-id-only overload stays, and stays correct, for the surfaces
    /// that ask the INSTALLED-BADGE question ("does this Mac hold this model
    /// at all") rather than the load question.
    public static func isCached(
        modelID: String, revision: String?, cacheRoot override: URL? = nil
    ) -> Bool {
        cachedLoadableSnapshot(
            modelID: modelID, revision: revision, cacheRoot: override) != nil
    }

    /// The snapshot directory `isCached(modelID:revision:)` proved, or nil.
    public static func cachedLoadableSnapshot(
        modelID: String, revision: String?, cacheRoot override: URL? = nil
    ) -> URL? {
        let cacheRoot = override ?? huggingFaceCacheRoot()
        let repo = "models--" + modelID.replacingOccurrences(of: "/", with: "--")
        let repoDirectory = cacheRoot.appending(components: "hub", repo)
        let requested = (revision ?? "main")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !requested.isEmpty else { return nil }

        let commit: String
        if isCommitHash(requested) {
            commit = requested
        } else {
            // A branch or tag: the cache's own ref file is the resolver, and
            // a ref this cache has never fetched is simply not cached.
            let ref = repoDirectory.appending(components: "refs", requested)
            guard let contents = try? String(contentsOf: ref, encoding: .utf8)
            else { return nil }
            commit = contents.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !commit.isEmpty else { return nil }
        }

        let snapshot = repoDirectory
            .appending(components: "snapshots", commit)
        var isDirectory: ObjCBool = false
        guard
            FileManager.default.fileExists(
                atPath: snapshot.path, isDirectory: &isDirectory),
            isDirectory.boolValue,
            snapshotHoldsLoadableFileSet(snapshot)
        else { return nil }
        return snapshot
    }

    /// A 40-character hex commit, the only revision spelling that addresses a
    /// snapshot directory directly (the hub client's own `isCommitHash` rule).
    static func isCommitHash(_ revision: String) -> Bool {
        revision.count == 40 && revision.allSatisfy(\.isHexDigit)
    }

    /// Does this snapshot directory hold everything a load opens? Existence
    /// only — and existence through the HF cache's blob symlinks, so a
    /// snapshot whose weight links point at blobs that were never fetched
    /// answers false, which is the partial-install case.
    static func snapshotHoldsLoadableFileSet(_ snapshot: URL) -> Bool {
        let fm = FileManager.default
        for name in requiredSnapshotFiles
        where !fm.fileExists(atPath: snapshot.appending(component: name).path) {
            return false
        }
        let index = snapshot.appending(component: "model.safetensors.index.json")
        guard fm.fileExists(atPath: index.path) else {
            // Unsharded (or unindexed) weights: at least one shard on disk.
            guard let entries = try? fm.contentsOfDirectory(atPath: snapshot.path)
            else { return false }
            return entries.contains { $0.hasSuffix(".safetensors") }
        }
        guard let data = try? Data(contentsOf: index),
            let object = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any],
            let map = object["weight_map"] as? [String: String]
        else { return false }
        let shards = Set(map.values)
        guard !shards.isEmpty else { return false }
        return shards.allSatisfy {
            fm.fileExists(atPath: snapshot.appending(component: $0).path)
        }
    }

    /// The commit hash of the locally cached snapshot for a model id, read
    /// from the HF cache's `refs/main` (honors `HF_HOME`). This is what a
    /// revision-less load actually runs, so it is the value to pin for
    /// reproducibility. Nil when the model has never been downloaded.
    public static func cachedRevision(for modelID: String) -> String? {
        let cacheRoot = huggingFaceCacheRoot()
        let repo = "models--" + modelID.replacingOccurrences(of: "/", with: "--")
        let ref = cacheRoot.appending(components: "hub", repo, "refs", "main")
        guard let contents = try? String(contentsOf: ref, encoding: .utf8) else {
            return nil
        }
        let commit = contents.trimmingCharacters(in: .whitespacesAndNewlines)
        return commit.isEmpty ? nil : commit
    }

    /// The locally cached snapshot directory for a model id (the folder
    /// `AutoTokenizer.from(modelFolder:)` loads from), resolved through the
    /// HF cache's `refs/main` exactly like `cachedRevision`. Nil when the
    /// model has never been downloaded. Lets callers load the raw
    /// swift-transformers tokenizer for renders the MLXLMCommon bridge does
    /// not expose (e.g. `addGenerationPrompt: false` for assistant-prefix
    /// continuation).
    public static func cachedSnapshotDirectory(for modelID: String) -> URL? {
        guard let revision = cachedRevision(for: modelID) else { return nil }
        let repo = "models--" + modelID.replacingOccurrences(of: "/", with: "--")
        let snapshot = huggingFaceCacheRoot()
            .appending(components: "hub", repo, "snapshots", revision)
        var isDirectory: ObjCBool = false
        guard
            FileManager.default.fileExists(
                atPath: snapshot.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else { return nil }
        return snapshot
    }

    public static func localModelIDs(cacheRoot override: URL? = nil) -> [String] {
        let cacheRoot = override ?? huggingFaceCacheRoot()
        let hub = cacheRoot.appending(component: "hub")
        guard
            let entries = try? FileManager.default.contentsOfDirectory(
                at: hub, includingPropertiesForKeys: [.isDirectoryKey])
        else { return [] }

        return entries.compactMap { url -> String? in
            let name = url.lastPathComponent
            guard name.hasPrefix("models--") else { return nil }
            let markerExists =
                FileManager.default.fileExists(
                    atPath: url.appending(components: "refs", "main").path)
                || FileManager.default.fileExists(
                    atPath: url.appending(component: "snapshots").path)
            guard markerExists else { return nil }

            let encoded = String(name.dropFirst("models--".count))
            let parts = encoded.components(separatedBy: "--")
            guard parts.count >= 2 else { return encoded }
            return parts[0] + "/" + parts.dropFirst().joined(separator: "--")
        }.sorted()
    }

    private static func huggingFaceCacheRoot() -> URL {
        return
            ProcessInfo.processInfo.environment["HF_HOME"].map { URL(filePath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appending(components: ".cache", "huggingface")
    }
}
