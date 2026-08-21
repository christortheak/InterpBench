import MLX

/// A hook invoked on the residual stream between transformer blocks.
///
/// Conforming types must be safe to call from the model's forward pass on
/// every layer of every forward pass — both the prefill pass (seq length =
/// prompt length) and each per-token decode pass (seq length 1). Injection
/// that fires only at prefill silently produces near-null steering results;
/// see CLAUDE.md › The hook problem.
public protocol LayerIntervention: Sendable {
    /// - Parameters:
    ///   - h: hidden state, shape [batch, seq, hidden]
    ///   - layer: index of the transformer block whose output this is
    ///   - offset: KV-cache position of the first token in `h` (0 during
    ///     prefill, the running token count during decode)
    /// - Returns: the (possibly modified) hidden state, same shape as `h`.
    func apply(_ h: MLXArray, layer: Int, offset: Int) -> MLXArray
}
