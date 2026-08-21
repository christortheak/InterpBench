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
