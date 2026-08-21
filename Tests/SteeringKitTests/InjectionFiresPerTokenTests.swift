import Foundation
import MLX
import MLXLMCommon
import Synchronization
import Testing
@testable import SteeringKit

/// Counts every hook firing. This guards the classic steering bug: an
/// intervention that fires only during prefill silently produces near-null
/// results (CLAUDE.md › The hook problem).
private final class CountingIntervention: LayerIntervention {
    struct Fire: Sendable, Equatable {
        let layer: Int
        let offset: Int
        let seqLen: Int
    }

    private let storage = Mutex<[Fire]>([])
    var fires: [Fire] { storage.withLock { $0 } }

    func apply(_ h: MLXArray, layer: Int, offset: Int) -> MLXArray {
        storage.withLock { $0.append(Fire(layer: layer, offset: offset, seqLen: h.dim(1))) }
        return h
    }
}

/// Drives a model through a prefill pass plus `decodeSteps` single-token
/// passes, the same forward-pass shapes the real generate loop produces.
private func runForwardPasses<M: LanguageModel & InterventionHookable>(
    model: M,
    counter: CountingIntervention,
    layerCount: Int,
    promptLength: Int,
    decodeSteps: Int
) {
    model.interventions = [counter]

    let cache = model.newCache(parameters: GenerateParameters?.none)

    // Prefill: full prompt.
    let prompt = MLXArray(Array(0 ..< Int32(promptLength))).reshaped([1, promptLength])
    var logits = model(prompt, cache: cache)
    eval(logits)

    // Decode: one token per pass, seq length 1.
    for step in 0 ..< decodeSteps {
        let token = MLXArray([Int32(step % 4)]).reshaped([1, 1])
        logits = model(token, cache: cache)
        eval(logits)
    }
}

private func assertFiresEveryPass(
    _ fires: [CountingIntervention.Fire],
    layerCount: Int,
    promptLength: Int,
    decodeSteps: Int
) {
    let passCount = 1 + decodeSteps
    #expect(fires.count == passCount * layerCount)

    // Every block of `layerCount` fires belongs to one forward pass and
    // covers every layer in order.
    for pass in 0 ..< passCount {
        let passFires = Array(fires[(pass * layerCount) ..< ((pass + 1) * layerCount)])
        #expect(passFires.map(\.layer) == Array(0 ..< layerCount))

        if pass == 0 {
            // Prefill: offset 0, sees the whole prompt.
            #expect(passFires.allSatisfy { $0.offset == 0 && $0.seqLen == promptLength })
        } else {
            // Decode: seq length 1, offset advances by one token per pass.
            let expectedOffset = promptLength + (pass - 1)
            #expect(passFires.allSatisfy { $0.offset == expectedOffset && $0.seqLen == 1 })
        }
    }
}

@Suite struct InjectionFiresPerTokenTests {

    @Test func qwen3HookFiresOnEveryLayerOfEveryPass() throws {
        let json = """
            {"hidden_size": 16, "num_hidden_layers": 2, "intermediate_size": 32,
             "num_attention_heads": 2, "rms_norm_eps": 1e-5, "vocab_size": 32,
             "num_key_value_heads": 1, "head_dim": 8, "tie_word_embeddings": true}
            """
        let config = try JSONDecoder().decode(
            SteeredQwen3Configuration.self, from: Data(json.utf8))
        let model = try SteeredQwen3Model(config)

        let counter = CountingIntervention()
        runForwardPasses(
            model: model, counter: counter, layerCount: 2, promptLength: 4, decodeSteps: 3)
        assertFiresEveryPass(counter.fires, layerCount: 2, promptLength: 4, decodeSteps: 3)
    }

    @Test func gemma3HookFiresOnEveryLayerOfEveryPass() throws {
        // sliding_window_pattern 2 over 4 layers exercises both the
        // sliding-window and global attention paths.
        let json = """
            {"model_type": "gemma3_text", "hidden_size": 16, "num_hidden_layers": 4,
             "intermediate_size": 32, "num_attention_heads": 2, "head_dim": 8,
             "vocab_size": 32, "num_key_value_heads": 1,
             "sliding_window": 8, "sliding_window_pattern": 2}
            """
        let config = try JSONDecoder().decode(
            SteeredGemma3TextConfiguration.self, from: Data(json.utf8))
        let model = SteeredGemma3TextModel(config)

        let counter = CountingIntervention()
        runForwardPasses(
            model: model, counter: counter, layerCount: 4, promptLength: 4, decodeSteps: 3)
        assertFiresEveryPass(counter.fires, layerCount: 4, promptLength: 4, decodeSteps: 3)
    }

    /// Chunked-prefill gating arithmetic (prompt 600 tokens, step 512):
    /// the first chunk's tail (token 511) is mid-prompt and must not be
    /// injected; the final chunk and every decode step must be.
    @Test func gateSkipsIntermediateChunksAndFiresFromPromptEnd() {
        #expect(
            !VectorInjector.shouldInject(offset: 0, seqLen: 512, promptTokenCount: 600))
        #expect(
            VectorInjector.shouldInject(offset: 512, seqLen: 88, promptTokenCount: 600))
        #expect(
            VectorInjector.shouldInject(offset: 600, seqLen: 1, promptTokenCount: 600))
        #expect(
            VectorInjector.shouldInject(offset: 601, seqLen: 1, promptTokenCount: 600))
        // Unknown prompt length: legacy last-position-of-every-pass.
        #expect(VectorInjector.shouldInject(offset: 0, seqLen: 512, promptTokenCount: nil))
    }

    /// Drives the model through chunked prefill (the real generate loop's
    /// shape for prompts longer than prefillStepSize) plus decode steps,
    /// returning the logits of every pass.
    private func runChunkedPasses<M: LanguageModel & InterventionHookable>(
        model: M, promptChunks: [Int], decodeSteps: Int
    ) -> [MLXArray] {
        let cache = model.newCache(parameters: GenerateParameters?.none)
        var logitsPerPass: [MLXArray] = []
        var position = 0
        for chunk in promptChunks {
            let tokens = MLXArray((position ..< position + chunk).map { Int32($0 % 16) })
                .reshaped([1, chunk])
            let logits = model(tokens, cache: cache)
            eval(logits)
            logitsPerPass.append(logits)
            position += chunk
        }
        for step in 0 ..< decodeSteps {
            let token = MLXArray([Int32(step % 4)]).reshaped([1, 1])
            let logits = model(token, cache: cache)
            eval(logits)
            logitsPerPass.append(logits)
        }
        return logitsPerPass
    }

    /// The audited prefill-chunking bug, end to end on a tiny model: an
    /// ungated injector steers the tail of EVERY prefill chunk (mid-prompt
    /// tokens); with promptTokenCount it must leave intermediate chunks
    /// bit-identical to baseline and steer only from the prompt end.
    @Test func gatedInjectionLeavesIntermediatePrefillChunksUntouched() throws {
        let json = """
            {"hidden_size": 16, "num_hidden_layers": 2, "intermediate_size": 32,
             "num_attention_heads": 2, "rms_norm_eps": 1e-5, "vocab_size": 32,
             "num_key_value_heads": 1, "head_dim": 8, "tie_word_embeddings": true}
            """
        let config = try JSONDecoder().decode(
            SteeredQwen3Configuration.self, from: Data(json.utf8))
        let model = try SteeredQwen3Model(config)
        let chunks = [4, 4, 2]  // 10-token prompt prefilled in three passes
        let vector = [Float](repeating: 1, count: 16)

        model.interventions = []
        let baseline = runChunkedPasses(model: model, promptChunks: chunks, decodeSteps: 2)

        model.interventions = [
            VectorInjector(layer: 1, vector: vector, alpha: 8, promptTokenCount: 10)
        ]
        let gated = runChunkedPasses(model: model, promptChunks: chunks, decodeSteps: 2)
        // Intermediate chunks: untouched. Final chunk + decode: steered.
        #expect(allClose(gated[0], baseline[0]).item(Bool.self))
        #expect(allClose(gated[1], baseline[1]).item(Bool.self))
        #expect(!allClose(gated[2], baseline[2]).item(Bool.self))
        #expect(!allClose(gated[3], baseline[3]).item(Bool.self))

        // Without the prompt length, the first chunk's tail (a mid-prompt
        // token) is steered — the bug the gate exists to prevent.
        model.interventions = [VectorInjector(layer: 1, vector: vector, alpha: 8)]
        let ungated = runChunkedPasses(model: model, promptChunks: chunks, decodeSteps: 2)
        #expect(!allClose(ungated[0], baseline[0]).item(Bool.self))
    }

    /// A real injection must change the logits; alpha 0 must not.
    @Test func injectionMovesLogitsAndAlphaZeroDoesNot() throws {
        let json = """
            {"hidden_size": 16, "num_hidden_layers": 2, "intermediate_size": 32,
             "num_attention_heads": 2, "rms_norm_eps": 1e-5, "vocab_size": 32,
             "num_key_value_heads": 1, "head_dim": 8, "tie_word_embeddings": true}
            """
        let config = try JSONDecoder().decode(
            SteeredQwen3Configuration.self, from: Data(json.utf8))
        let model = try SteeredQwen3Model(config)
        let prompt = MLXArray([0, 1, 2, 3] as [Int32]).reshaped([1, 4])

        let baseline = model(prompt, cache: nil)
        eval(baseline)

        let vector = [Float](repeating: 1, count: 16)

        model.interventions = [VectorInjector(layer: 1, vector: vector, alpha: 8)]
        let steered = model(prompt, cache: nil)
        eval(steered)
        #expect(!allClose(steered, baseline).item(Bool.self))

        model.interventions = [VectorInjector(layer: 1, vector: vector, alpha: 0)]
        let alphaZero = model(prompt, cache: nil)
        eval(alphaZero)
        #expect(allClose(alphaZero, baseline).item(Bool.self))
    }
}
