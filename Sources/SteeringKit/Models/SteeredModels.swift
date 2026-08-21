import Foundation
import MLXLLM
import MLXLMCommon

/// A vendored model implementation's refusal to build itself from a
/// configuration it cannot use. Typed and thrown (never `fatalError`) so a
/// bad `config.json` field surfaces through the model-loading error path —
/// `SteeredModels.typeRegistry`'s creator → `LLMModelFactory` →
/// `SteeredContainerLoader.load` → the caller's "load failed: …" — instead
/// of killing the app or the CLI. Lives here rather than in a vendored file
/// so the vendored diff stays a one-line throw.
public enum SteeredModelConfigurationError: Error, Equatable, CustomStringConvertible {
    /// `rope_scaling.factor` is present with `type: "linear"` but cannot
    /// yield a finite, nonzero RoPE scale.
    case unusableRopeScalingFactor(model: String, value: String)

    public var description: String {
        switch self {
        case .unusableRopeScalingFactor(let model, let value):
            "\(model): config.json rope_scaling.factor is \(value), which is "
                + "not a usable number — the RoPE scale is 1/factor, so a "
                + "non-numeric, infinite or zero factor would silently "
                + "mis-position every token. Fix the model's config.json (or "
                + "remove rope_scaling to run unscaled)"
        }
    }
}

extension SteeredModelConfigurationError: LocalizedError {
    public var errorDescription: String? { description }
}

/// Loads models through the vendored, hook-capable implementations.
///
/// Only steered model types are registered — an experiment must never
/// silently fall back to an upstream implementation without hook points.
public enum SteeredModels {

    public static let factory = LLMModelFactory(
        typeRegistry: typeRegistry,
        modelRegistry: LLMRegistry.shared
    )

    private static let typeRegistry = ModelTypeRegistry<LanguageModel>(creators: [
        "qwen3": { data -> any LanguageModel in
            try SteeredQwen3Model(
                try JSONDecoder.json5().decode(SteeredQwen3Configuration.self, from: data))
        },
        "gemma3_text": { data -> any LanguageModel in
            SteeredGemma3TextModel(
                try JSONDecoder.json5().decode(SteeredGemma3TextConfiguration.self, from: data))
        },
        // Gemma 3 repos converted from the multimodal checkpoints keep
        // model_type "gemma3"; the vendored text model already handles the
        // text_config nesting and language_model.* weight re-rooting, so
        // route them to the same implementation.
        "gemma3": { data -> any LanguageModel in
            SteeredGemma3TextModel(
                try JSONDecoder.json5().decode(SteeredGemma3TextConfiguration.self, from: data))
        },
    ])
}
