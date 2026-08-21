import MLX
import Synchronization

/// Where in the stimulus the residual stream is read during extraction
/// (METHODS.md › Method options).
public enum ReadingPosition: Codable, Sendable, Equatable {
    /// Hidden state of the final token (RepE's convention; Phase 0 default).
    case lastToken
    /// Mean over token positions from `k` onward (the emotion paper pools
    /// from token 50 of paragraph stories). Extraction callers must enforce
    /// that token `k` exists; otherwise this reading position is not valid.
    case meanFromToken(Int)

    public var label: String {
        switch self {
        case .lastToken: "last token"
        case .meanFromToken(let k): "mean from token \(k)"
        }
    }

    /// Minimum token count required for this reading position to mean what it
    /// says. `meanFromToken(50)` needs token index 50 to exist.
    public var minimumTokenCount: Int {
        switch self {
        case .lastToken: 1
        case .meanFromToken(let k): max(1, k + 1)
        }
    }

    public var requestedStartIndex: Int? {
        switch self {
        case .lastToken: nil
        case .meanFromToken(let k): k
        }
    }

    /// Inverse of `label`: parse a sidecar's stamped reading-position string
    /// back into a position. Used by jobs that must re-measure at an
    /// artifact's recorded position (e.g. residual-norm backfill). Nil for
    /// unrecognized labels — callers must fail loudly, not guess.
    public init?(label: String) {
        if label == ReadingPosition.lastToken.label {
            self = .lastToken
        } else if label.hasPrefix("mean from token "),
            let k = Int(label.dropFirst("mean from token ".count)), k >= 0
        {
            self = .meanFromToken(k)
        } else {
            return nil
        }
    }
}

/// Records the residual stream at configured layers during a forward pass.
///
/// Values are evaluated and copied to CPU inside the hook — capturing lazy
/// GPU arrays for every layer/token explodes unified memory (CLAUDE.md ›
/// MLX gotchas). The eager copy makes this recorder appropriate for
/// extraction passes (one prefill per stimulus, a handful of layers), not
/// for per-token capture over long generations.
///
/// Assumes batch size 1.
public final class ActivationRecorder: LayerIntervention {
    public struct Capture: Sendable {
        /// Transformer block whose output was recorded.
        public let layer: Int
        /// KV-cache position of the first token of the recorded pass
        /// (0 = prefill).
        public let offset: Int
        /// Number of token positions in this forward pass.
        public let tokenCount: Int
        /// First position included in the pooled readout.
        public let startIndex: Int
        /// Hidden vector at the reading position, copied to CPU as float32.
        public let values: [Float]
        /// Mean L2 norm of the residual stream over the positions read —
        /// the "typical residual-stream norm" used to express steering
        /// strength in norm units (the emotion paper's convention).
        public let residualNorm: Float

        public init(
            layer: Int, offset: Int, tokenCount: Int = 0, startIndex: Int = 0,
            values: [Float], residualNorm: Float
        ) {
            self.layer = layer
            self.offset = offset
            self.tokenCount = tokenCount
            self.startIndex = startIndex
            self.values = values
            self.residualNorm = residualNorm
        }
    }

    private let layers: Set<Int>
    private let position: ReadingPosition
    private let storage = Mutex<[Capture]>([])

    /// - Parameters:
    ///   - layers: block indices to record; record only what you need —
    ///     every capture is an eager GPU→CPU copy.
    ///   - position: where in the sequence to read (default last token).
    public init(layers: some Sequence<Int>, position: ReadingPosition = .lastToken) {
        self.layers = Set(layers)
        self.position = position
    }

    /// All captures so far, in hook-firing order.
    public var captures: [Capture] {
        storage.withLock { $0 }
    }

    public func reset() {
        storage.withLock { $0.removeAll() }
    }

    public func apply(_ h: MLXArray, layer: Int, offset: Int) -> MLXArray {
        guard layers.contains(layer) else { return h }

        let length = h.dim(1)
        let startIndex: Int
        switch position {
        case .lastToken:
            startIndex = length - 1
        case .meanFromToken(let k):
            startIndex = min(max(0, k), length - 1)
        }

        // rows: [positions, hidden] over the reading window.
        let rows = h[0, startIndex..., 0...].asType(.float32)
        let pooled = rows.mean(axis: 0)
        let norms = sqrt(rows.square().sum(axis: -1)).mean()

        // asArray/item force evaluation and copy to CPU promptly.
        let values = pooled.asArray(Float.self)
        let residualNorm = norms.item(Float.self)
        storage.withLock {
            $0.append(
                Capture(
                    layer: layer, offset: offset, tokenCount: length, startIndex: startIndex,
                    values: values, residualNorm: residualNorm))
        }
        return h
    }
}

/// Records one residual-stream row per token position, for calibration banks
/// such as Anthropic-style neutral transcript PCA. This is intentionally
/// separate from `ActivationRecorder`, whose contract is one pooled vector per
/// stimulus.
///
/// **Ingestion is bounded, not post-hoc trimmed.** The row cap is applied
/// HERE, as rows arrive, via `selectedRowIndices` — a precomputed set of
/// global bank positions from `TokenBankDownsampler.selectedIndexSet`. Only
/// the selected positions are gathered off the GPU, so the transient CPU
/// float32 copy is `kept × hidden`, never `tokens × hidden`. Collecting every
/// row first and downsampling afterwards (the shape this recorder had until
/// 2026-08-18) makes an all-layer capture over a real corpus a multi-GB ×
/// layers transient, which fails inside MLX/native allocation and kills the
/// process before a Swift `catch` can run (CLAUDE.md › MLX gotchas).
///
/// Positions are counted per layer in ARRIVAL order — text order, then token
/// order — which is exactly the row order the old two-phase downsample
/// indexed, so the streaming filter keeps the identical rows. The cursor
/// therefore persists across `reset()` (which drains rows only); use
/// `resetAll()` to start a fresh bank.
///
/// Residual norms are accumulated for EVERY position regardless of selection:
/// one float per token is cheap, and the norm denominator should describe the
/// whole corpus, not the draw.
public final class ActivationBankRecorder: LayerIntervention {
    public struct Row: Sendable {
        public let layer: Int
        public let offset: Int
        public let tokenIndex: Int
        public let values: [Float]
        public let residualNorm: Float
    }

    /// A residual norm for a position that was NOT banked. Callers fold these
    /// into the per-layer norm average alongside the banked rows.
    public struct SkippedNorm: Sendable {
        public let layer: Int
        public let residualNorm: Float
    }

    private struct Storage {
        var rows: [Row] = []
        var skippedNorms: [SkippedNorm] = []
        /// Next global bank position for each layer. Survives `reset()`.
        var nextPositionByLayer: [Int: Int] = [:]
        /// High-water mark of `rows.count` — the retention instrument the
        /// bounded-ingestion test asserts against.
        var peakRowCount = 0
    }

    private let layers: Set<Int>
    private let startIndex: Int
    private let selectedRowIndices: Set<Int>?
    private let storage = Mutex(Storage())

    /// - Parameters:
    ///   - layers: block indices to record. Restrict this — an all-layer bank
    ///     costs `rows × layers × hidden` float32 even when bounded.
    ///   - startIndex: first token position to bank.
    ///   - selectedRowIndices: global bank positions to keep, or nil to keep
    ///     every position. Pass `TokenBankDownsampler.selectedIndexSet`.
    public init(
        layers: some Sequence<Int>,
        startIndex: Int = 0,
        selectedRowIndices: Set<Int>? = nil
    ) {
        self.layers = Set(layers)
        self.startIndex = max(0, startIndex)
        self.selectedRowIndices = selectedRowIndices
    }

    public var rows: [Row] {
        storage.withLock { $0.rows }
    }

    /// Residual norms for positions the row cap excluded, so the norm average
    /// still covers the whole corpus.
    public var skippedNorms: [SkippedNorm] {
        storage.withLock { $0.skippedNorms }
    }

    /// Largest number of rows this recorder has ever held at once. The
    /// bounded-ingestion guarantee is observable: with a selection set of size
    /// `cap`, this never exceeds `cap × layers` across a whole bank, and
    /// exactly `cap` when the driver drains per pass.
    public var peakRetainedRowCount: Int {
        storage.withLock { $0.peakRowCount }
    }

    /// Drains banked rows and skipped norms. The per-layer position cursor
    /// deliberately survives, so a driver can drain after every forward pass
    /// while the global selection stays aligned.
    public func reset() {
        storage.withLock {
            $0.rows.removeAll()
            $0.skippedNorms.removeAll()
        }
    }

    /// Full reset, cursors included — a new bank over a new corpus.
    public func resetAll() {
        storage.withLock { $0 = Storage() }
    }

    public func apply(_ h: MLXArray, layer: Int, offset: Int) -> MLXArray {
        guard layers.contains(layer) else { return h }
        let length = h.dim(1)
        guard startIndex < length else { return h }
        let positionCount = length - startIndex

        // Claim this pass's slice of the global position space before doing
        // any work, so selection is stable no matter how passes interleave.
        let base = storage.withLock { state -> Int in
            let base = state.nextPositionByLayer[layer] ?? 0
            state.nextPositionByLayer[layer] = base + positionCount
            return base
        }

        let keptPositions: [Int]
        if let selectedRowIndices {
            keptPositions = (0 ..< positionCount).filter {
                selectedRowIndices.contains(base + $0)
            }
        } else {
            keptPositions = Array(0 ..< positionCount)
        }

        // Norms first: one float per position, so this stays cheap even when
        // nothing in this pass is banked.
        let window = h[0, startIndex..., 0...].asType(.float32)
        let residualNorms = sqrt(window.square().sum(axis: -1)).asArray(Float.self)

        // Gather ONLY the kept rows on the GPU before the CPU copy. This is
        // the bound: `values` is kept × hidden, not positionCount × hidden.
        var values: [Float] = []
        if !keptPositions.isEmpty {
            let gathered =
                keptPositions.count == positionCount
                ? window
                : window.take(MLXArray(keptPositions.map { Int32($0) }), axis: 0)
            values = gathered.asArray(Float.self)
        }
        let hidden = h.dim(2)
        let keptSet = keptPositions.isEmpty ? Set<Int>() : Set(keptPositions)

        storage.withLock { state in
            for (slot, position) in keptPositions.enumerated() {
                let lower = slot * hidden
                let upper = lower + hidden
                guard upper <= values.count, position < residualNorms.count else { continue }
                state.rows.append(
                    Row(
                        layer: layer,
                        offset: offset,
                        tokenIndex: startIndex + position,
                        values: Array(values[lower ..< upper]),
                        residualNorm: residualNorms[position]))
            }
            for position in 0 ..< positionCount where !keptSet.contains(position) {
                guard position < residualNorms.count else { continue }
                state.skippedNorms.append(
                    SkippedNorm(layer: layer, residualNorm: residualNorms[position]))
            }
            state.peakRowCount = max(state.peakRowCount, state.rows.count)
        }
        return h
    }
}
