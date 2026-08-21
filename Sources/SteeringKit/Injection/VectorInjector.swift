import MLX

/// Adds `alpha * vector` to the residual stream at the configured layers,
/// at the final prompt position and the last position of every decode pass.
///
/// Firing on every decode pass is load-bearing: steering only during
/// prefill silently produces near-null results — the classic bug this
/// design guards against (CLAUDE.md › The hook problem). The converse bug
/// is firing too often: chunked prefill makes "last position of the pass"
/// a mid-prompt token for every chunk but the final one, so injection is
/// gated on `promptTokenCount` when the prompt length is known.
///
/// The vector is stored as `[Float]` because `MLXArray` is not `Sendable`;
/// it is materialized on the GPU inside the hook. The residual stream is
/// float (bf16/fp16) even for quantized models — we cast to `h.dtype` and
/// never touch the quantized weights.
public struct VectorInjector: LayerIntervention {
    public struct Injection: Sendable {
        /// Steering direction, length = model hidden size.
        public let vector: [Float]
        /// Steering strength. Report in units of the typical residual-stream
        /// norm at the layer (CLAUDE.md › Steering method).
        public let alpha: Float

        public init(vector: [Float], alpha: Float) {
            self.vector = vector
            self.alpha = alpha
        }
    }

    /// Layer index → injection applied to that block's output.
    private let injections: [Int: Injection]

    /// Total prompt token count, when known. The generate loop prefills the
    /// prompt in chunks (`GenerateParameters.prefillStepSize`, default 512),
    /// and each chunk is its own forward pass — so "last position of the
    /// pass" is a MID-PROMPT token for every chunk but the final one. With
    /// the prompt length known, injection is suppressed for passes that end
    /// before the final prompt position. nil preserves the
    /// last-position-of-every-pass behavior, which is correct only when
    /// prefill is guaranteed single-chunk (short prompts, or
    /// prefillStepSize ≥ prompt length).
    private let promptTokenCount: Int?

    public init(injections: [Int: Injection], promptTokenCount: Int? = nil) {
        self.injections = injections
        self.promptTokenCount = promptTokenCount
    }

    /// Convenience for the common single-layer case.
    public init(
        layer: Int, vector: [Float], alpha: Float, promptTokenCount: Int? = nil
    ) {
        self.init(
            injections: [layer: Injection(vector: vector, alpha: alpha)],
            promptTokenCount: promptTokenCount)
    }

    /// Whether a pass covering absolute positions `offset ..< offset+seqLen`
    /// should be injected at its last position. Pure, so the chunked-prefill
    /// gating is unit-testable without a model.
    public static func shouldInject(
        offset: Int, seqLen: Int, promptTokenCount: Int?
    ) -> Bool {
        guard let promptTokenCount else { return true }
        return offset + seqLen - 1 >= promptTokenCount - 1
    }

    public func apply(_ h: MLXArray, layer: Int, offset: Int) -> MLXArray {
        guard let injection = injections[layer] else { return h }
        guard
            Self.shouldInject(
                offset: offset, seqLen: h.dim(1), promptTokenCount: promptTokenCount)
        else { return h }  // intermediate prefill chunk: its tail is mid-prompt

        let v = (MLXArray(injection.vector) * injection.alpha).asType(h.dtype)
        let lastIndex = h.dim(1) - 1

        if lastIndex == 0 {
            // Decode step (or single-token prompt): one position, broadcast.
            return h + v
        }

        // Final prefill chunk: apply at the last position only — the true
        // prompt end — leaving earlier positions untouched (CLAUDE.md:
        // apply at the last position in all cases).
        let head = h[0..., ..<lastIndex, 0...]
        let tail = h[0..., lastIndex..., 0...] + v
        return concatenated([head, tail], axis: 1)
    }
}
