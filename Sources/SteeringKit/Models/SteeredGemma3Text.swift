// SteeredGemma3Text.swift
//
// VENDORED from mlx-swift-lm 3.31.3, Libraries/MLXLLM/Models/Gemma3Text.swift
// (original by Anthony DePasquale, port of mlx-lm's gemma3_text.py).
//
// Diff against upstream, kept minimal so the file can be re-vendored:
//   1. Types renamed Gemma3* -> SteeredGemma3* (avoids ambiguity with
//      MLXLLM's own Gemma3 implementation when both modules are imported).
//   2. `SteeredGemma3Model` gains `var interventions` and calls each
//      intervention on the residual stream after every transformer block,
//      on every forward pass (prefill and per-token decode) — marked
//      [SteerLab] below.
//   3. `SteeredGemma3TextModel` conforms to `InterventionHookable`.
//   4. Added `import MLXLLM` (upstream lives inside that module; we need it
//      for the `LLMModel` protocol).
//   5. `init(from:)` uses the VLM-side defaults (MLXVLM/Models/Gemma3.swift:
//      8 heads, 4 KV heads, vocab 262208, maxPositionEmbeddings 4096) when
//      the configuration arrives nested under `text_config` — VLM-converted
//      repos (e.g. mlx-community/gemma-3-4b-it-4bit) omit every field that
//      matches the transformers class defaults, and upstream's text-side
//      defaults are the 1B's, which mis-sizes attention for 4B/12B/27B.
//   6. `SteeredGemma3TextModel` conforms to `LogitLensReadable`
//      (`logitsForResidualVector`): applies the model's final RMSNorm and
//      then lm_head to a raw residual vector — the lens must read through
//      the same final path the model computes, not the head alone.
// Everything else is verbatim upstream. Note: layers alternate sliding-window
// and global attention (slidingWindowPattern), but the residual stream
// between blocks is homogeneous — interventions apply identically at every
// block boundary.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

public struct SteeredGemma3TextConfiguration: Codable {
    let modelType: String
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let vocabularySize: Int
    let kvHeads: Int
    let ropeTheta: Float
    let ropeLocalBaseFreq: Float
    let ropeTraditional: Bool
    let queryPreAttnScalar: Float
    let slidingWindow: Int
    let slidingWindowPattern: Int
    let maxPositionEmbeddings: Int
    let ropeScaling: [String: StringOrNumber]?

    public init(
        modelType: String, hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int,
        attentionHeads: Int, headDim: Int, rmsNormEps: Float, vocabularySize: Int, kvHeads: Int,
        ropeTheta: Float, ropeLocalBaseFreq: Float, ropeTraditional: Bool,
        queryPreAttnScalar: Float, slidingWindow: Int, slidingWindowPattern: Int,
        maxPositionEmbeddings: Int, ropeScaling: [String: StringOrNumber]? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.ropeTheta = ropeTheta
        self.ropeLocalBaseFreq = ropeLocalBaseFreq
        self.ropeTraditional = ropeTraditional
        self.queryPreAttnScalar = queryPreAttnScalar
        self.slidingWindow = slidingWindow
        self.slidingWindowPattern = slidingWindowPattern
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeScaling = ropeScaling
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeTraditional = "rope_traditional"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeScaling = "rope_scaling"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)

        // in the case of VLM models convertered using mlx_lm.convert
        // the configuration will still match the VLMs and be under text_config
        // [SteerLab] VLM-converted repos omit fields matching the transformers
        // class defaults; fall back to the VLM-side defaults in that case
        // (see diff note 5 in the header).
        let isVLMNested = nestedContainer.contains(.textConfig)
        let container =
            if isVLMNested {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 26
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6912
        attentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
            ?? (isVLMNested ? 8 : 4)
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.0e-6
        vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
            ?? (isVLMNested ? 262208 : 262144)
        kvHeads =
            try container.decodeIfPresent(Int.self, forKey: .kvHeads)
            ?? (isVLMNested ? 4 : 1)
        ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        ropeLocalBaseFreq =
            try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10_000.0
        ropeTraditional =
            try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        queryPreAttnScalar =
            try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar) ?? 256
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 6
        ropeScaling =
            try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        let explicitMaxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        let scaledVLMContext =
            Int((ropeScaling?["factor"]?.asFloat() ?? 1) * 4096)
        maxPositionEmbeddings =
            explicitMaxPositionEmbeddings
            ?? (isVLMNested ? max(4096, scaledVLMContext) : 32768)
    }
}

class SteeredGemma3Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let repeats: Int
    let headDim: Int
    let layerIdx: Int
    let scale: Float
    let isSliding: Bool
    let slidingWindow: Int
    let slidingWindowPattern: Int

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: Gemma.RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma.RMSNorm

    @ModuleInfo var rope: RoPELayer

    init(_ config: SteeredGemma3TextConfiguration, layerIdx: Int) {
        let dim = config.hiddenSize
        self.nHeads = config.attentionHeads
        self.nKVHeads = config.kvHeads
        self.repeats = nHeads / nKVHeads
        self.headDim = config.headDim
        self.layerIdx = layerIdx
        self.slidingWindow = config.slidingWindow
        self.slidingWindowPattern = config.slidingWindowPattern

        self.scale = pow(config.queryPreAttnScalar, -0.5)

        self._queryProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._valueProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        self._outputProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

        self._queryNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma.RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        self.isSliding = (layerIdx + 1) % config.slidingWindowPattern != 0

        if isSliding {
            self.rope = initializeRope(
                dims: headDim, base: config.ropeLocalBaseFreq, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: nil)
        } else {
            self.rope = initializeRope(
                dims: headDim, base: config.ropeTheta, traditional: false,
                scalingConfig: config.ropeScaling,
                maxPositionEmbeddings: config.maxPositionEmbeddings)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = queryProj(x)
        var keys = keyProj(x)
        var values = valueProj(x)

        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

        queries = queryNorm(queries)
        keys = keyNorm(keys)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

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
        return outputProj(output)
    }
}

class SteeredGemma3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

class SteeredGemma3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: SteeredGemma3Attention
    @ModuleInfo var mlp: SteeredGemma3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm

    let numAttentionHeads: Int
    let hiddenSize: Int
    let layerIdx: Int

    init(_ config: SteeredGemma3TextConfiguration, layerIdx: Int) {
        self.numAttentionHeads = config.attentionHeads
        self.hiddenSize = config.hiddenSize
        self.layerIdx = layerIdx

        self._selfAttention.wrappedValue = SteeredGemma3Attention(config, layerIdx: layerIdx)
        self.mlp = SteeredGemma3MLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)

        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let inputNorm = inputLayerNorm(x)
        let r = selfAttention(inputNorm, mask: mask, cache: cache)
        let attnNorm = postAttentionLayerNorm(r)
        let h = Gemma.clipResidual(x, attnNorm)
        let preMLPNorm = preFeedforwardLayerNorm(h)
        let r2 = mlp(preMLPNorm)
        let postMLPNorm = postFeedforwardLayerNorm(r2)
        let out = Gemma.clipResidual(h, postMLPNorm)
        return out
    }
}

public class SteeredGemma3Model: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [SteeredGemma3TransformerBlock]
    @ModuleInfo var norm: Gemma.RMSNorm

    let config: SteeredGemma3TextConfiguration

    // [SteerLab] residual-stream hooks; mutated only inside the
    // ModelContainer actor (see InterventionHookable).
    var interventions: [any LayerIntervention] = []

    init(_ config: SteeredGemma3TextConfiguration) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
            SteeredGemma3TransformerBlock(config, layerIdx: layerIdx)
        }

        self.norm = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil
    )
        -> MLXArray
    {
        var h: MLXArray
        h = embedTokens(inputs)
        let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
        h = h * scale.asType(h.dtype)
        var layerCache = cache
        if layerCache == nil {
            layerCache = Array(repeating: nil as KVCache?, count: layers.count)
        }

        let globalMask = createAttentionMask(h: h, cache: cache?[config.slidingWindowPattern - 1])
        let slidingWindowMask =
            if config.slidingWindowPattern > 1 {
                createAttentionMask(h: h, cache: cache?[0], windowSize: config.slidingWindow)
            } else {
                MLXFast.ScaledDotProductAttentionMaskMode.none
            }

        // [SteerLab] capture the position offset before the first block's
        // attention advances the cache: 0 during prefill, the running token
        // count during per-token decode.
        let offset = cache?.compactMap { $0 }.first?.offset ?? 0

        for (i, layer) in layers.enumerated() {
            let isGlobal = (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)
            let mask = isGlobal ? globalMask : slidingWindowMask
            h = layer(h, mask: mask, cache: layerCache?[i])
            // [SteerLab] hook fires after every block, on every forward pass.
            for intervention in interventions {
                h = intervention.apply(h, layer: i, offset: offset)
            }
        }
        return norm(h)
    }
}

public class SteeredGemma3TextModel: Module, LLMModel {

    @ModuleInfo public var model: SteeredGemma3Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: SteeredGemma3TextConfiguration
    public var vocabularySize: Int { config.vocabularySize }
    public var contextWindow: Int { config.maxPositionEmbeddings }

    public init(_ config: SteeredGemma3TextConfiguration) {
        self.config = config
        self.model = SteeredGemma3Model(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, mask: nil, cache: cache)
        out = lmHead(out)
        return out
    }

    public func sanitize(weights: [String: MLXArray])
        -> [String: MLXArray]
    {
        var processedWeights = weights

        // VLM models converted using mlx_vlm.convert will still have
        // the weights are under a language_model key
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        let expectedVocab = config.vocabularySize
        let keysToCheck = [
            "model.embed_tokens.weight", "model.embed_tokens.scales", "model.embed_tokens.biases",
            "lm_head.weight", "lm_head.scales", "lm_head.biases",
        ]

        for key in keysToCheck {
            if let tensor = processedWeights[key], tensor.dim(0) > expectedVocab {
                processedWeights[key] = tensor[0 ..< expectedVocab]
            }
        }

        if processedWeights["lm_head.weight"] == nil {
            ["weight", "scales", "biases"].forEach { key in
                if let embedWeight = processedWeights["model.embed_tokens.\(key)"] {
                    processedWeights["lm_head.\(key)"] = embedWeight
                }
            }
        }
        return processedWeights
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        var caches = [KVCache]()
        let slidingWindow = config.slidingWindow
        let slidingWindowPattern = config.slidingWindowPattern

        for i in 0 ..< config.hiddenLayers {
            let isGlobalLayer = (i % slidingWindowPattern == slidingWindowPattern - 1)

            if isGlobalLayer {
                // For global layers, use standard cache but with reasonable step size for long sequences
                let cache = StandardKVCache()
                cache.step = 1024  // Larger step size for efficiency with long sequences
                caches.append(cache)
            } else {
                // For sliding window layers, use rotating cache
                caches.append(
                    RotatingKVCache(maxSize: slidingWindow, keep: 0)
                )
            }
        }

        return caches
    }

    /// Handles prompt processing for sequences
    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int? = nil
    ) throws -> PrepareResult {
        let promptTokens = input.text.tokens
        let promptCount = promptTokens.dim(0)

        guard promptCount > 0 else {
            print("Warning: Preparing with empty prompt tokens.")
            let emptyToken = MLXArray(Int32(0))[0 ..< 0]
            return .tokens(.init(tokens: emptyToken))
        }

        return .tokens(input.text)
    }
}

// [SteerLab] expose the hooks and context limit to SteeringKit consumers.
extension SteeredGemma3TextModel: ContextWindowProviding, InterventionHookable {
    public var interventions: [any LayerIntervention] {
        get { model.interventions }
        set { model.interventions = newValue }
    }
}

// [SteerLab] residual-stream shape, so capture-heavy paths (the neutral token
// bank's memory preflight, the middle-third layer band) can size themselves
// before the first forward pass.
extension SteeredGemma3TextModel: ResidualShapeProviding {
    public var residualHiddenSize: Int { config.hiddenSize }
    public var residualBlockCount: Int { config.hiddenLayers }
}

extension SteeredGemma3TextModel: LogitLensReadable {
    // [SteerLab] logit lens (part of the vendored delta): read a residual
    // vector through the model's OWN final path — final RMSNorm, then
    // lm_head. Skipping the final norm ranks tokens the model never
    // computes (docs/METHODS.md); fixed 2026-07-06.
    public func logitsForResidualVector(_ vector: [Float]) -> [Float] {
        let input = model.norm(MLXArray(vector).reshaped([1, 1, vector.count]))
        return lmHead(input).asType(.float32).asArray(Float.self)
    }
}

extension SteeredGemma3TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
