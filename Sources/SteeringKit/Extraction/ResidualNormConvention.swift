import Foundation

/// The residual-norm DENOMINATOR CONVENTION — the fixed rule by which the
/// per-layer number that denominates a norm-unit α is computed.
///
/// α is reported in units of the residual-stream norm at the injection layer
/// (CLAUDE.md › "Report steering strength in units of the residual-stream
/// norm…"). That only makes α comparable across concepts if every engine
/// computes the denominator the same way. Until 2026-08-20 the two engines
/// did NOT: this engine's neutral token bank averaged over every corpus
/// position, while the server's averaged over the BANKED positions only (the
/// deterministic row-cap draw). On a downsampled corpus those are different
/// numbers, so "α = 1 norm units" meant two different doses.
///
/// **The ruling (2026-08-20): the convention is the WHOLE-CORPUS average.**
/// Every measured position counts toward the denominator, banked or not — the
/// denominator describes the corpus, not the draw. The server was fixed to
/// match; see `Server/steerlab_server/steering/residual_norm_convention.py`
/// for the twin.
///
/// The convention is STAMPED, never retro-applied. New measurements write
/// `residualNormConvention: "wholeCorpusMean-v1"` into the sidecar; artifacts
/// without the stamp are LEGACY and are read exactly as before — no
/// migration, no recompute, no warning. `vectors backfill-norms` re-measures
/// under the current convention and stamps it, which is the researcher's
/// opt-in migration path.
public enum ResidualNormConvention {

    /// The stamp written by every fresh measurement on BOTH engines. Pinned
    /// cross-engine contract: same JSON key (`residualNormConvention`), same
    /// value string. Bump the version suffix only when the averaging RULE
    /// changes — never for an unrelated sidecar edit.
    public static let current = "wholeCorpusMean-v1"

    /// Human-readable label for an artifact that predates the stamp. Surfaces
    /// that already show norm provenance may render this; nothing may treat
    /// it as evidence of a particular rule, because a pre-stamp artifact's
    /// rule is genuinely unknown (it depends on which engine wrote it and
    /// whether its corpus was downsampled).
    public static let legacyLabel = "legacy (pre-stamp)"

    /// The convention label to display for a sidecar's stamp, `nil` meaning
    /// "no norms recorded at all" (there is no denominator to describe).
    public static func displayLabel(
        residualNormPerLayer: [Float]?, stamp: String?
    ) -> String? {
        // Empty counts as absent, matching the server twin's falsy check —
        // an empty table is no denominator, not an unstamped one.
        guard let residualNormPerLayer, !residualNormPerLayer.isEmpty else { return nil }
        return stamp ?? legacyLabel
    }

    /// Per-layer running mean of residual norms under the whole-corpus rule.
    ///
    /// Both the banked rows and the rows the token-bank cap EXCLUDED are
    /// added here; `mean(at:)` is then the corpus mean, not the draw's mean.
    /// The server's `ResidualNormTally` is the line-for-line twin, and the
    /// two are pinned against one shared fixture
    /// (`ResidualNormConventionTests` / `test_residual_norm_convention.py`).
    public struct Tally: Sendable {
        private var sums: [Int: Double] = [:]
        private var counts: [Int: Int] = [:]

        public init() {}

        /// One measured position at one layer. Call for EVERY position the
        /// forward pass produced — dropping the unbanked ones is exactly the
        /// bug this convention closes.
        public mutating func add(layer: Int, norm: Float) {
            sums[layer, default: 0] += Double(norm)
            counts[layer, default: 0] += 1
        }

        public var layers: [Int] { counts.keys.sorted() }

        public func count(at layer: Int) -> Int { counts[layer] ?? 0 }

        /// Total measured positions across every layer — the bank's
        /// `tokenRowCount` (positions counted, not rows retained).
        public var totalCount: Int { counts.values.reduce(0, +) }

        /// The whole-corpus mean at `layer`; 0 when nothing was measured
        /// there (an unmeasured layer has no denominator, and 0 is refused
        /// downstream by the `residualNorm > 0` guards rather than silently
        /// steering).
        public func mean(at layer: Int) -> Float {
            let count = counts[layer] ?? 0
            guard count > 0 else { return 0 }
            return Float((sums[layer] ?? 0) / Double(count))
        }
    }
}
