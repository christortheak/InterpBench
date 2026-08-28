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
/// A denominator table that cannot answer for the layer being dosed — the
/// truncated/empty `residualNormPerLayer` seam (2026-08-28 audit, F7/F13).
/// Its own type rather than a `SteeringVectorError` case because the refusal
/// carries prose naming the artifact and the layer, and `SteeringVectorError`
/// is a closed enum of shape faults with no room for one.
public struct ResidualNormTableError: Error, Equatable, CustomStringConvertible {
    public let reason: String

    public init(reason: String) { self.reason = reason }

    public var description: String { reason }
}

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

    /// LOAD-TIME gate on the shape of a denominator table (2026-08-28 audit,
    /// F7/F13). Returns the refusal prose, or `nil` when the table is usable.
    ///
    /// A denominator table must cover every layer of the artifact it travels
    /// with, or none of them. **Absent (or empty) is legal and stays legal**:
    /// the OptVec, J-lens and Gemma-Scope-report families are born with no
    /// norms at all and acquire them through `vectors backfill-norms`, so an
    /// absent table is a state the writers deliberately produce and the
    /// per-verb refusals already name. A table that is present but SHORTER
    /// than the artifact's depth is the state no writer produces —
    /// extraction measures one norm per layer, backfill writes exactly
    /// `layerCount` of them, and the SAE by-id import slices the donor to its
    /// layer count — so a short table is a malformed or hand-edited artifact,
    /// and every verb that reads it with a layer index would otherwise dose
    /// the layers past the end with some other layer's number.
    ///
    /// Server twin: `residual_norm_convention.table_length_problem`
    /// (byte-identical prose, pinned on both engines).
    public static func tableLengthProblem(
        _ residualNormPerLayer: [Float]?, layerCount: Int, artifact: String
    ) -> String? {
        guard let norms = residualNormPerLayer, !norms.isEmpty else { return nil }
        guard norms.count != layerCount else { return nil }
        return "vector artifact '\(artifact)' carries \(norms.count) residual "
            + "norms for \(layerCount) layers — a denominator table must "
            + "cover every layer or none, and a short one silently doses the "
            + "layers it does not reach with another layer's number; "
            + "re-measure the norms (vectors backfill-norms), or re-extract "
            + "the concept"
    }

    /// USE-SITE gate — the ONE out-of-range rule every verb applies
    /// (condition, sweep, variant), on both engines. Returns the refusal
    /// prose, or `nil` when the layer has a denominator.
    ///
    /// Before this landed the same truncated table produced four different
    /// outcomes for one artifact: the server's condition path substituted
    /// `0.0` and refused as `degenerateData`, its sweep and variant paths
    /// clamped to the last entry and dosed the deepest layers with a
    /// shallower layer's number, and this engine's condition path clamped as
    /// well — while an EMPTY table indexed `[-1]` here and crashed. One
    /// artifact, four behaviours, and three of them silent. The rule is now
    /// the house one everywhere: refuse, and say which layer had no
    /// denominator.
    ///
    /// Server twin: `residual_norm_convention.residual_norm_problem`
    /// (byte-identical prose).
    public static func residualNormProblem(
        _ residualNormPerLayer: [Float]?, layer: Int, artifact: String
    ) -> String? {
        let norms = residualNormPerLayer ?? []
        guard !norms.indices.contains(layer) else { return nil }
        return "'\(artifact)' has no residual norm at layer \(layer) — its "
            + "denominator table covers \(norms.count) layer(s), so an α in "
            + "residual-norm units cannot be denominated there; re-measure "
            + "the norms (vectors backfill-norms), or switch α to raw units"
    }

    /// The denominator at `layer`, or the typed refusal above. The four
    /// injection-building sites (condition, variant, sweep grid, sweep
    /// control) all read the table through this one accessor, so the rule
    /// cannot drift between verbs again.
    public static func residualNorm(
        _ residualNormPerLayer: [Float]?, at layer: Int, artifact: String
    ) throws -> Float {
        if let problem = residualNormProblem(
            residualNormPerLayer, layer: layer, artifact: artifact)
        {
            throw ResidualNormTableError(reason: problem)
        }
        return (residualNormPerLayer ?? [])[layer]
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
