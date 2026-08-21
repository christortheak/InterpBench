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
