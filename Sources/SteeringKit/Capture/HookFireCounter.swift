import MLX
import Synchronization

/// Diagnostic intervention that records every hook firing without touching
/// the residual stream. Used by the smoke test to prove the intervention
/// fires on every layer of every forward pass — prefill and each decode
/// step — which is the invariant steering correctness depends on.
public final class HookFireCounter: LayerIntervention {
    public struct Fire: Sendable, Equatable {
        public let layer: Int
        public let offset: Int
        public let seqLen: Int
    }

    private let storage = Mutex<[Fire]>([])

    public init() {}

    public var fires: [Fire] { storage.withLock { $0 } }

    public func reset() { storage.withLock { $0.removeAll() } }

    public func apply(_ h: MLXArray, layer: Int, offset: Int) -> MLXArray {
        storage.withLock { $0.append(Fire(layer: layer, offset: offset, seqLen: h.dim(1))) }
        return h
    }
}
