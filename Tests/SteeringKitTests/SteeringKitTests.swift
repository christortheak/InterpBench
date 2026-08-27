import Testing
import MLX
import Foundation
@testable import SteeringKit

@Suite struct LayerInterventionTests {
    /// Identity intervention returns the array unchanged — exercises the
    /// protocol shape and that MLX links and evaluates in the test host.
    @Test func identityInterventionPreservesValues() {
        struct Identity: LayerIntervention {
            func apply(_ h: MLXArray, layer: Int, offset: Int) -> MLXArray { h }
        }
        let h = MLXArray([1.0, 2.0, 3.0] as [Float]).reshaped([1, 1, 3])
        let out = Identity().apply(h, layer: 0, offset: 0)
        eval(out)
        #expect(out.shape == [1, 1, 3])
        #expect(allClose(out, h).item(Bool.self))
    }

    @Test func localModelIDsReadHuggingFaceCache() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "hf-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDir = root.appending(components: "hub", "models--Qwen--Qwen3-4B-MLX-4bit")
        try FileManager.default.createDirectory(
            at: modelDir.appending(component: "refs"), withIntermediateDirectories: true)
        try "abc123".write(to: modelDir.appending(components: "refs", "main"), atomically: true, encoding: .utf8)

        #expect(SteeredContainerLoader.localModelIDs(cacheRoot: root) == ["Qwen/Qwen3-4B-MLX-4bit"])
    }

    /// "Is it downloaded?" and "what is downloaded?" must be the same
    /// question: the load gate and the picker's availability badge both read
    /// this one enumeration, so they cannot disagree about a model.
    @Test func isCachedAgreesWithTheCacheEnumeration() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "hf-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDir = root.appending(components: "hub", "models--vendor-a--model-small-4bit")
        try FileManager.default.createDirectory(
            at: modelDir.appending(component: "refs"), withIntermediateDirectories: true)
        try "abc123".write(
            to: modelDir.appending(components: "refs", "main"),
            atomically: true, encoding: .utf8)

        #expect(
            SteeredContainerLoader.isCached(
                modelID: "vendor-a/model-small-4bit", cacheRoot: root))
        #expect(
            !SteeredContainerLoader.isCached(
                modelID: "vendor-b/model-large-8bit", cacheRoot: root))
    }

    // MARK: - The exact-revision, complete-snapshot guard (round 8, finding 3)

    /// Writes a cache entry: `refs/main` naming `commit`, and a snapshot
    /// directory under `snapshots/<commit>` holding exactly `files`.
    static func writeCacheEntry(
        root: URL, modelID: String, commit: String, ref: String? = "main",
        files: [String: String]
    ) throws {
        let fm = FileManager.default
        let repo = "models--" + modelID.replacingOccurrences(of: "/", with: "--")
        let modelDir = root.appending(components: "hub", repo)
        if let ref {
            try fm.createDirectory(
                at: modelDir.appending(component: "refs"),
                withIntermediateDirectories: true)
            try commit.write(
                to: modelDir.appending(components: "refs", ref),
                atomically: true, encoding: .utf8)
        }
        let snapshot = modelDir.appending(components: "snapshots", commit)
        try fm.createDirectory(at: snapshot, withIntermediateDirectories: true)
        for (name, contents) in files {
            try contents.write(
                to: snapshot.appending(component: name), atomically: true,
                encoding: .utf8)
        }
    }

    /// A complete single-shard snapshot: what an install that finished looks
    /// like on disk.
    static let completeSnapshot: [String: String] = [
        "config.json": "{}", "tokenizer_config.json": "{}",
        "tokenizer.json": "{}", "model.safetensors": "weights",
    ]

    static func temporaryCacheRoot() -> URL {
        FileManager.default.temporaryDirectory
            .appending(component: "hf-\(UUID().uuidString)")
    }

    /// THE finding: a cache holding revision A satisfied a load pinned to
    /// revision B, and the load then fetched B over the network — the exact
    /// thing the no-download guard exists to prevent.
    @Test func theRevisionGuardAsksAboutTheRevisionTheLoadWillRequest() throws {
        let root = Self.temporaryCacheRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let cached = String(repeating: "a", count: 40)
        let pinned = String(repeating: "b", count: 40)
        try Self.writeCacheEntry(
            root: root, modelID: "vendor-a/model-small-4bit", commit: cached,
            files: Self.completeSnapshot)

        // The repo-id-only question still answers yes — this Mac does hold
        // the model, which is the right answer for an installed badge.
        #expect(
            SteeredContainerLoader.isCached(
                modelID: "vendor-a/model-small-4bit", cacheRoot: root))
        // The LOAD question is revision-specific, and answers each spelling
        // of the cached revision the same way.
        #expect(
            SteeredContainerLoader.isCached(
                modelID: "vendor-a/model-small-4bit", revision: cached,
                cacheRoot: root))
        #expect(
            SteeredContainerLoader.isCached(
                modelID: "vendor-a/model-small-4bit", revision: nil,
                cacheRoot: root))
        #expect(
            SteeredContainerLoader.isCached(
                modelID: "vendor-a/model-small-4bit", revision: "main",
                cacheRoot: root))
        // A DIFFERENT pin is not cached, however complete the other one is.
        #expect(
            !SteeredContainerLoader.isCached(
                modelID: "vendor-a/model-small-4bit", revision: pinned,
                cacheRoot: root))
        // Nor is a ref this cache has never fetched.
        #expect(
            !SteeredContainerLoader.isCached(
                modelID: "vendor-a/model-small-4bit", revision: "some-branch",
                cacheRoot: root))
    }

    /// The other half: a snapshot that EXISTS but cannot be loaded. Each
    /// omission is one of the files `LLMModelFactory._load` actually opens.
    @Test func anIncompleteSnapshotIsNotCached() throws {
        for missing in ["config.json", "tokenizer_config.json",
                        "tokenizer.json", "model.safetensors"] {
            let root = Self.temporaryCacheRoot()
            defer { try? FileManager.default.removeItem(at: root) }
            var files = Self.completeSnapshot
            files.removeValue(forKey: missing)
            try Self.writeCacheEntry(
                root: root, modelID: "vendor-a/partial", commit: "c0ffee",
                files: files)
            #expect(
                !SteeredContainerLoader.isCached(
                    modelID: "vendor-a/partial", revision: nil, cacheRoot: root),
                "a snapshot missing \(missing) is not loadable")
        }
    }

    /// A `snapshots` PATH existing was the whole of the old marker test, and
    /// an interrupted install is exactly that shape.
    @Test func aRepoWithNoSnapshotDirectoryIsNotCachedAtAnyRevision() throws {
        let root = Self.temporaryCacheRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let fm = FileManager.default
        let modelDir = root.appending(
            components: "hub", "models--vendor-a--never-fetched")
        try fm.createDirectory(
            at: modelDir.appending(component: "refs"),
            withIntermediateDirectories: true)
        try "abc123".write(
            to: modelDir.appending(components: "refs", "main"),
            atomically: true, encoding: .utf8)

        #expect(
            SteeredContainerLoader.isCached(
                modelID: "vendor-a/never-fetched", cacheRoot: root))
        #expect(
            !SteeredContainerLoader.isCached(
                modelID: "vendor-a/never-fetched", revision: nil,
                cacheRoot: root))
    }

    /// A SHARDED model: the index names every shard, and one absent shard is
    /// an incomplete download however many of the others landed.
    @Test func aShardedSnapshotNeedsEveryShardTheIndexNames() throws {
        let index = """
            {"weight_map": {"a.weight": "model-00001-of-00002.safetensors", \
            "b.weight": "model-00002-of-00002.safetensors"}}
            """
        let root = Self.temporaryCacheRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        var partial = Self.completeSnapshot
        partial.removeValue(forKey: "model.safetensors")
        partial["model.safetensors.index.json"] = index
        partial["model-00001-of-00002.safetensors"] = "shard one"
        try Self.writeCacheEntry(
            root: root, modelID: "vendor-a/sharded", commit: "d00d",
            files: partial)
        #expect(
            !SteeredContainerLoader.isCached(
                modelID: "vendor-a/sharded", revision: nil, cacheRoot: root))

        let complete = Self.temporaryCacheRoot()
        defer { try? FileManager.default.removeItem(at: complete) }
        partial["model-00002-of-00002.safetensors"] = "shard two"
        try Self.writeCacheEntry(
            root: complete, modelID: "vendor-a/sharded", commit: "d00d",
            files: partial)
        #expect(
            SteeredContainerLoader.isCached(
                modelID: "vendor-a/sharded", revision: nil, cacheRoot: complete))
    }
}

/// A model's own `config.json` must never be able to kill the process.
/// Upstream mlx-swift-lm 3.31.3 `fatalError`s on a `rope_scaling.factor` it
/// cannot read as a float; the vendored copy throws (diff item 6 in
/// `SteeredQwen3.swift`'s header) so the failure surfaces through the
/// model-loading error path as a readable message.
@Suite struct SteeredModelConfigurationTests {

    private func configuration(
        ropeScaling: String? = nil
    ) throws -> SteeredQwen3Configuration {
        let scaling = ropeScaling.map { ", \"rope_scaling\": \($0)" } ?? ""
        let json = """
            {"hidden_size": 16, "num_hidden_layers": 1, "intermediate_size": 32,
             "num_attention_heads": 2, "rms_norm_eps": 1e-5, "vocab_size": 32,
             "num_key_value_heads": 1, "head_dim": 8,
             "tie_word_embeddings": true\(scaling)}
            """
        return try JSONDecoder().decode(
            SteeredQwen3Configuration.self, from: Data(json.utf8))
    }

    /// An INT factor was never actually broken (`StringOrNumber.asFloat()`
    /// coerces `.int`, despite its doc comment), and a string one now parses
    /// instead of refusing. Both must build a model rather than trap.
    @Test func numericAndStringFactorsBuildAModel() throws {
        for factor in ["8", "8.0", "\"8.0\""] {
            let config = try configuration(
                ropeScaling: "{\"type\": \"linear\", \"factor\": \(factor)}")
            #expect(throws: Never.self) { try SteeredQwen3Model(config) }
        }
        // No rope_scaling at all is the ordinary case (scale 1).
        #expect(throws: Never.self) { try SteeredQwen3Model(try configuration()) }
    }

    /// A factor that cannot yield a finite, nonzero 1/factor scale refuses
    /// — typed, catchable, and not a process death.
    @Test func unusableFactorThrowsInsteadOfCrashing() throws {
        for factor in ["\"eight\"", "0", "[2, 4]"] {
            let config = try configuration(
                ropeScaling: "{\"type\": \"linear\", \"factor\": \(factor)}")
            #expect(throws: SteeredModelConfigurationError.self) {
                try SteeredQwen3Model(config)
            }
        }
    }

    @Test func theRefusalNamesTheFieldAndTheRemedy() {
        let error = SteeredModelConfigurationError.unusableRopeScalingFactor(
            model: "qwen3", value: "string(\"eight\")")
        #expect(error.description.contains("rope_scaling.factor"))
        #expect(error.description.contains("config.json"))
        #expect(error.localizedDescription == error.description)
    }
}
