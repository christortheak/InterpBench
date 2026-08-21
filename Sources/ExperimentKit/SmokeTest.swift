import Foundation
import MLX
import MLXLMCommon
import SteeringKit
import Synchronization

public struct SmokeTestConfig: Codable, Sendable {
    public struct ModelSpec: Codable, Sendable {
        /// "qwen3" or "gemma3" — selects family-specific prompt handling.
        public var family: String
        /// Hugging Face repo id.
        public var id: String
        /// Fallback context window for budget checks before a model is loaded.
        /// Runtime generation prefers the decoded model config.
        public var contextWindow: Int?

        public init(family: String, id: String, contextWindow: Int? = nil) {
            self.family = family
            self.id = id
            self.contextWindow = contextWindow
        }

        enum CodingKeys: String, CodingKey {
            case family
            case id
            case contextWindow
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            family = try container.decode(String.self, forKey: .family)
            id = try container.decode(String.self, forKey: .id)
            contextWindow = try container.decodeIfPresent(Int.self, forKey: .contextWindow)
        }
    }

    public var models: [ModelSpec]
    public var prompt: String
    public var maxTokens: Int
    /// Strength of the random control injection, in units of the typical
    /// residual-stream norm at the chosen layer.
    public var alpha: Float
    public var seed: UInt64
}

public struct SmokeTestFailure: Error, CustomStringConvertible {
    public let model: String
    public let reason: String
    public var description: String { "[\(model)] \(reason)" }
}

/// Phase 0 smoke test, run after any change to SteeringKit (CLAUDE.md ›
/// Build & run). Per model family it proves the plumbing end to end:
///   1. the model loads through the vendored, hookable implementation;
///   2. the hook fires on every layer of every forward pass, prefill and
///      per-token decode alike;
///   3. a matched-norm random vector at mid-depth changes greedy output;
///   4. the same vector at alpha 0 reproduces the baseline exactly.
/// (Random-vector vs concept-vector divergence is asserted from Phase 1,
/// once extraction exists.)
public enum SmokeTest {

    public static func run(config: SmokeTestConfig) async throws {
        // Long generation loops balloon unified memory without a cache limit
        // (CLAUDE.md › MLX gotchas).
        MLX.Memory.cacheLimit = 2 * 1024 * 1024 * 1024

        for spec in config.models {
            try await run(spec: spec, config: config)
        }
    }

    private static func run(spec: SmokeTestConfig.ModelSpec, config: SmokeTestConfig)
        async throws
    {
        print("=== \(spec.id) ===")
        let lastReported = Mutex(-1)
        let container = try await SteeredContainerLoader.load(modelID: spec.id) { progress in
            let percent = Int(progress.fractionCompleted * 100)
            let report = lastReported.withLock { last in
                guard percent >= last + 10 || (percent == 100 && last < 100) else { return false }
                last = percent
                return true
            }
            if report { print("downloading: \(percent)%") }
        }

        // Qwen3's soft switch for disabling thinking mode; Gemma 3 has no
        // thinking mode (and no system role — plain prompt is correct).
        let prompt = spec.family == "qwen3" ? config.prompt + " /no_think" : config.prompt
        let greedy = GenerateParameters(maxTokens: config.maxTokens, temperature: 0)

        // Probe pass: record every layer once to learn layer count, hidden
        // size, and the typical residual-stream norm at the target layer.
        let recorder = ActivationRecorder(layers: 0 ..< 512)
        try await setInterventions(container, [recorder])
        _ = try await generate(
            container, prompt: prompt,
            parameters: GenerateParameters(maxTokens: 1, temperature: 0))

        let captures = recorder.captures
        guard let maxLayer = captures.map(\.layer).max() else {
            throw SmokeTestFailure(model: spec.id, reason: "probe pass recorded nothing")
        }
        let layerCount = maxLayer + 1
        let targetLayer = layerCount / 2
        guard let probe = captures.first(where: { $0.layer == targetLayer }) else {
            throw SmokeTestFailure(model: spec.id, reason: "no capture at layer \(targetLayer)")
        }
        let hiddenSize = probe.values.count
        let typicalNorm = SteeringVectorMath.l2Norm(probe.values)
        print(
            "layers: \(layerCount), hidden: \(hiddenSize), "
                + "residual norm @ L\(targetLayer): \(typicalNorm)")

        // Baseline, greedy.
        try await setInterventions(container, [])
        let baseline = try await generate(container, prompt: prompt, parameters: greedy)
        print("baseline: \(baseline.text.prefix(120))…")

        // Matched-norm random vector at mid-depth (coherence-control shape).
        var rng = SplitMix64(seed: config.seed)
        let vector = try SteeringVectorMath.randomVector(
            dimension: hiddenSize, norm: typicalNorm, using: &rng)

        let counter = HookFireCounter()
        try await setInterventions(
            container,
            [
                VectorInjector(layer: targetLayer, vector: vector, alpha: config.alpha),
                counter,
            ])
        let steered = try await generate(container, prompt: prompt, parameters: greedy)
        print("steered:  \(steered.text.prefix(120))…")

        // Hook-fire accounting: every forward pass must touch every layer,
        // and decode passes (seq length 1) must be present.
        let fires = counter.fires
        guard fires.count % layerCount == 0 else {
            throw SmokeTestFailure(
                model: spec.id,
                reason: "hook fired \(fires.count) times — not a multiple of \(layerCount) layers")
        }
        let passCount = fires.count / layerCount
        let decodeFires = fires.filter { $0.seqLen == 1 }
        guard passCount >= 2, !decodeFires.isEmpty else {
            throw SmokeTestFailure(
                model: spec.id,
                reason: "hook did not fire on decode steps (passes: \(passCount))")
        }
        let decodeOffsets = Set(decodeFires.map(\.offset))
        guard decodeOffsets.count == decodeFires.count / layerCount else {
            throw SmokeTestFailure(
                model: spec.id, reason: "decode offsets did not advance per token")
        }
        print("hook fired \(fires.count)× across \(passCount) passes ✓")

        guard steered.text != baseline.text else {
            throw SmokeTestFailure(
                model: spec.id,
                reason: "alpha \(config.alpha) random injection did not change greedy output")
        }

        // Alpha 0 must reproduce the baseline exactly under greedy decoding.
        try await setInterventions(
            container, [VectorInjector(layer: targetLayer, vector: vector, alpha: 0)])
        let alphaZero = try await generate(container, prompt: prompt, parameters: greedy)
        guard alphaZero.text == baseline.text else {
            throw SmokeTestFailure(
                model: spec.id, reason: "alpha 0 output diverged from baseline")
        }

        print("PASS \(spec.id)\n")
    }

    // MARK: - Helpers

    private static func setInterventions(
        _ container: ModelContainer, _ interventions: [any LayerIntervention]
    ) async throws {
        try await container.perform { context in
            guard let model = context.model as? InterventionHookable else {
                throw SmokeTestFailure(
                    model: "\(type(of: context.model))",
                    reason: "model is not InterventionHookable — wrong factory?")
            }
            model.interventions = interventions
        }
    }

    private static func generate(
        _ container: ModelContainer, prompt: String, parameters: GenerateParameters
    ) async throws -> (text: String, tokenCount: Int) {
        let input = try await container.prepare(input: UserInput(prompt: prompt))
        let stream = try await container.generate(input: input, parameters: parameters)
        var text = ""
        var tokenCount = 0
        for await event in stream {
            switch event {
            case .chunk(let chunk):
                text += chunk
                tokenCount += 1
            default:
                break
            }
        }
        return (text, tokenCount)
    }
}
