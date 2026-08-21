/// A vendored model whose residual stream accepts interventions.
///
/// Set interventions from inside `ModelContainer.update { context in ... }`
/// (or `perform`) by downcasting `context.model` — the model is confined to
/// the container's actor, so mutation there is safe under strict concurrency.
public protocol InterventionHookable: AnyObject {
    var interventions: [any LayerIntervention] { get set }
}

/// A vendored model that reports its configured maximum context window.
///
/// The value comes from the decoded model config (`max_position_embeddings`)
/// and lets callers reject impossible prompt+generation budgets before MLX
/// hits a lower-level assertion.
public protocol ContextWindowProviding: AnyObject {
    var contextWindow: Int { get }
}

/// A vendored model that reports the shape of its residual stream.
///
/// Both values come from the decoded model config, so capture-heavy work can
/// size itself BEFORE the first forward pass — the neutral token bank's memory
/// preflight (`NeutralBankBudget`) needs `rows × layers × hidden` up front, and
/// the layer-band default needs the block count.
public protocol ResidualShapeProviding: AnyObject {
    /// Width of the residual stream (`hidden_size`).
    var residualHiddenSize: Int { get }
    /// Number of transformer blocks, i.e. the number of hook points.
    var residualBlockCount: Int { get }
}

/// A vendored model that can read a residual-stream direction through its
/// output head. Used for logit-lens validation of concept vectors.
public protocol LogitLensReadable: AnyObject {
    func logitsForResidualVector(_ vector: [Float]) -> [Float]
}
