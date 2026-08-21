// SteeredQwen3.swift
//
// VENDORED from mlx-swift-lm 3.31.3, Libraries/MLXLLM/Models/Qwen3.swift
// (original by John Mai, port of mlx-lm's qwen3.py).
//
// Diff against upstream, kept minimal so the file can be re-vendored:
//   1. Types renamed Qwen3* -> SteeredQwen3* (avoids ambiguity with MLXLLM's
//      own Qwen3 implementation when both modules are imported).
//   2. `SteeredQwen3ModelInner` gains `var interventions` and calls each
//      intervention on the residual stream after every transformer block,
//      on every forward pass (prefill and per-token decode) — marked
//      [SteerLab] below.
//   3. `SteeredQwen3Model` conforms to `InterventionHookable`.
//   4. Added `import MLXLLM` (upstream lives inside that module; we need it
//      for the `LLMModel` protocol).
//   5. `SteeredQwen3Model` conforms to `LogitLensReadable`
//      (`logitsForResidualVector`): applies the model's final RMSNorm and
//      then the output head to a raw residual vector — the lens must read
//      through the same final path the model computes, not the head alone.
//   6. Upstream `Qwen3Attention.init` ends the PROCESS on an unusable
//      `rope_scaling.factor` (`fatalError("ropeScaling.factor must be a
//      float")` — verbatim in 3.31.3). A library must not kill the app or
//      the CLI over a config value, so the initializers on the path
//      Attention → TransformerBlock → ModelInner → Model are `throws` here
//      and the failure is a typed `SteeredModelConfigurationError`, which
//      surfaces through the model-loading error path
//      (`SteeredModels.typeRegistry` creator → `LLMModelFactory` →
//      `SteeredContainerLoader.load`) as a readable message. A STRING
//      factor ("8.0") is also accepted, which upstream refuses; ints
//      already worked (`StringOrNumber.asFloat()` coerces `.int`, despite
//      its doc comment). Upstream's own newer helper
//      (`MLXLMCommon.initializeRope`) silently falls back to scale 1 in the
//      same spot — this refuses instead, because a mis-scaled RoPE is a
//      wrong measurement, not a warning.
// Everything else is verbatim upstream.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

class SteeredQwen3Attention: Module {
    let args: SteeredQwen3Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    public init(_ args: SteeredQwen3Configuration) throws {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeScale: Float
        if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
            let factor = ropeScaling["factor"]
        {
            // [SteerLab] diff item 6: throw instead of fatalError. asFloat()
            // already coerces an INT factor (and a single-element int/float
            // array, and a bool); what it refuses is a string factor and a
            // multi-element array. The string is parsed here rather than
            // refused — a hand-written config writing "8.0" means 8.0 — and
            // a factor that cannot yield a finite, nonzero scale refuses,
            // because `1 / 0` is an infinite RoPE scale, not a default.
            let parsed: Float? =
                if let v = factor.asFloat() {
                    v
                } else if case .string(let text) = factor {
                    Float(text)
                } else {
                    nil
                }
            guard let v = parsed, v.isFinite, v != 0 else {
                throw SteeredModelConfigurationError.unusableRopeScalingFactor(
                    model: "qwen3", value: String(describing: factor))
            }
            ropeScale = 1 / v
        } else {
            ropeScale = 1
        }

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: args.ropeTheta,
            scale: ropeScale)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // prepare the queries, keys and values for the attention computation
        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        // Apply RoPE positioning
        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        // Use the automatic attention router that handles both quantized and regular caches
        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class SteeredQwen3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class SteeredQwen3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: SteeredQwen3Attention
    let mlp: SteeredQwen3MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: SteeredQwen3Configuration) throws {
        _attention.wrappedValue = try SteeredQwen3Attention(args)
        self.mlp = SteeredQwen3MLP(
            dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }
}

public class SteeredQwen3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [SteeredQwen3TransformerBlock]
    let norm: RMSNorm

    // [SteerLab] residual-stream hooks; mutated only inside the
    // ModelContainer actor (see InterventionHookable).
    var interventions: [any LayerIntervention] = []

    public init(_ args: SteeredQwen3Configuration) throws {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = try (0 ..< args.hiddenLayers)
            .map { _ in
                try SteeredQwen3TransformerBlock(args)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        // [SteerLab] capture the position offset before the first block's
        // attention advances the cache: 0 during prefill, the running token
        // count during per-token decode.
        let offset = cache?.first?.offset ?? 0

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
            // [SteerLab] hook fires after every block, on every forward pass.
            for intervention in interventions {
                h = intervention.apply(h, layer: i, offset: offset)
            }
        }

        return norm(h)
    }
}

public class SteeredQwen3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: SteeredQwen3ModelInner
    let configuration: SteeredQwen3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: SteeredQwen3Configuration) throws {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = try SteeredQwen3ModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights

        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        return weights
    }
}

// [SteerLab] expose the hooks to SteeringKit consumers.
extension SteeredQwen3Model: InterventionHookable {
    public var interventions: [any LayerIntervention] {
        get { model.interventions }
        set { model.interventions = newValue }
    }
}

extension SteeredQwen3Model: ContextWindowProviding {
    public var contextWindow: Int {
        configuration.maxPositionEmbeddings
    }
}

// [SteerLab] residual-stream shape, so capture-heavy paths (the neutral token
// bank's memory preflight, the middle-third layer band) can size themselves
// before the first forward pass.
extension SteeredQwen3Model: ResidualShapeProviding {
    public var residualHiddenSize: Int { configuration.hiddenSize }
    public var residualBlockCount: Int { configuration.hiddenLayers }
}

extension SteeredQwen3Model: LogitLensReadable {
    // [SteerLab] logit lens (part of the vendored delta): read a residual
    // vector through the model's OWN final path — final RMSNorm, then
    // lm_head / tied embeddings. Skipping the final norm ranks tokens the
    // model never computes (docs/METHODS.md); fixed 2026-07-06.
    public func logitsForResidualVector(_ vector: [Float]) -> [Float] {
        let input = model.norm(MLXArray(vector).reshaped([1, 1, vector.count]))
        let logits =
            if let lmHead {
                lmHead(input)
            } else {
                model.embedTokens.asLinear(input)
            }
        return logits.asType(.float32).asArray(Float.self)
    }
}

public struct SteeredQwen3Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 1_000_000
    var headDim: Int
    var ropeScaling: [String: StringOrNumber]? = nil
    var tieWordEmbeddings = false
    var maxPositionEmbeddings: Int = 32768

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<SteeredQwen3Configuration.CodingKeys> =
            try decoder.container(
                keyedBy: SteeredQwen3Configuration.CodingKeys.self)

        self.hiddenSize = try container.decode(
            Int.self, forKey: SteeredQwen3Configuration.CodingKeys.hiddenSize)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: SteeredQwen3Configuration.CodingKeys.hiddenLayers)
        self.intermediateSize = try container.decode(
            Int.self, forKey: SteeredQwen3Configuration.CodingKeys.intermediateSize)
        self.attentionHeads = try container.decode(
            Int.self, forKey: SteeredQwen3Configuration.CodingKeys.attentionHeads)
        self.rmsNormEps = try container.decode(
            Float.self, forKey: SteeredQwen3Configuration.CodingKeys.rmsNormEps)
        self.vocabularySize = try container.decode(
            Int.self, forKey: SteeredQwen3Configuration.CodingKeys.vocabularySize)
        self.kvHeads = try container.decode(
            Int.self, forKey: SteeredQwen3Configuration.CodingKeys.kvHeads)
        self.ropeTheta =
            try container.decodeIfPresent(
                Float.self, forKey: SteeredQwen3Configuration.CodingKeys.ropeTheta)
            ?? 1_000_000
        self.headDim = try container.decode(
            Int.self, forKey: SteeredQwen3Configuration.CodingKeys.headDim)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: SteeredQwen3Configuration.CodingKeys.ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
    }
}

// MARK: - LoRA

extension SteeredQwen3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
