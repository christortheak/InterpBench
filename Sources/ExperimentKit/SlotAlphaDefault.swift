import Foundation
import SteeringKit

/// What α a freshly-selected steering vector should start at, in which UNITS,
/// and what the control must say about both.
///
/// The bug this replaces (researcher report, 2026-08-18): the Playground's
/// slot carried a flat `alpha = 2` default, applied to every vector regardless
/// of what the artifact knew about itself. In norm units — the session default
/// — α = 2 is two whole residual-stream norms, off the end of the control's
/// own −1…1 range, and nothing about the artifact's measured denominator or
/// its promoted cell was consulted. "Every time I select a vector it defaults
/// to alpha = 2, even if it's in norm units."
///
/// The rules, in order:
///
/// 1. **A vector with a usable denominator steers in NORM UNITS.** If the
///    artifact has a promoted/recommended strength (a sweep-selected cell,
///    carried on an agent's promotion birth certificate), that is the default —
///    it is the one α anybody actually selected on dev data. Otherwise 0.1
///    norm units — the researcher's ruling (2026-08-20): a full residual
///    norm (1.0) reliably drives generation incoherent, while 0.1 is the
///    dose the OptVec training convention itself uses (alphaNormFactor
///    0.1), a working strength rather than a degenerate one.
/// 2. **A vector with NO stored denominator steers in RAW units, explicitly
///    labelled.** Never a silent unit switch: raw α and norm-unit α differ by
///    orders of magnitude at the same numeral, so the control says which one
///    it is and offers the repair (`vectors backfill-norms`).
/// 3. **A recommended α is adopted together with ITS units.** A norm-unit
///    recommendation is unusable without a denominator and is NOT reused as a
///    raw number — that would change dose by roughly the residual norm itself.
///
/// Engine-pure and total: the view renders the `Decision`, it does not
/// recompute any part of it. Cross-engine note — this is a LOCAL UI default,
/// not an artifact contract, so it has no server twin; what both engines share
/// is the denominator (`ResidualNormConvention`) the decision reads.
public enum SlotAlphaDefault {

    /// α = a tenth of the residual-stream norm at the injection layer —
    /// the researcher's ruling (2026-08-20): 1.0 (a full norm) leads to
    /// incoherent generation; 0.1 matches the OptVec training convention's
    /// alphaNormFactor and is a usable starting dose. Replaces the flat 2,
    /// which in norm units was already off the control's range.
    public static let normUnitsDefault: Double = 0.1
    /// The historical raw default, unchanged — raw-mode behaviour is exactly
    /// what it always was, it is merely LABELLED now.
    public static let rawUnitsDefault: Double = 2

    public enum Units: String, Sendable, Equatable, Codable {
        case normUnits
        case raw
    }

    /// Where the defaulted number came from — rendered as the control's
    /// rationale so a researcher is never guessing why a number appeared.
    public enum Provenance: Sendable, Equatable {
        /// A promoted agent's cell for this exact vector artifact.
        case promotedCell(agent: String, promotedBy: String?)
        /// No recommendation existed; the convention's unit stride.
        case conventionDefault
        /// No denominator: the labelled raw fallback.
        case rawFallback
        /// A recommendation existed but was denominated in units this
        /// artifact cannot currently express.
        case recommendationUnusable(agent: String)
    }

    /// Everything the decision reads, and nothing else — so the table below is
    /// exhaustively testable without a model, a catalog, or a view.
    public struct ArtifactFacts: Sendable, Equatable {
        /// `<runDir>/<name>` — named in the backfill hint so the repair is
        /// copy-pasteable.
        public var artifactID: String
        /// The layer the slot will inject at (already clamped by the caller).
        public var layer: Int
        /// The artifact's residual-stream norm AT `layer`. `nil` or a
        /// non-positive value both mean "no usable denominator" — a zero
        /// denominator is as unusable as a missing one, and treating it as
        /// present would divide by it.
        public var residualNormAtLayer: Float?
        /// `SteeringVectorSidecar.residualNormConvention`; `nil` = legacy.
        public var residualNormConvention: String?
        /// A promoted/recommended strength for THIS artifact, if any.
        public var recommendedAlpha: Double?
        /// The units that recommendation was expressed in. Adopted with the
        /// number, never reinterpreted.
        public var recommendedAlphaInNormUnits: Bool?
        /// The agent carrying the recommendation (for the rationale line).
        public var recommendedFromAgent: String?
        /// `Promotion.promotedBy` — "criterion" or "manualOverride"; `nil`
        /// for a hand-created variant.
        public var recommendedPromotedBy: String?

        public init(
            artifactID: String,
            layer: Int,
            residualNormAtLayer: Float? = nil,
            residualNormConvention: String? = nil,
            recommendedAlpha: Double? = nil,
            recommendedAlphaInNormUnits: Bool? = nil,
            recommendedFromAgent: String? = nil,
            recommendedPromotedBy: String? = nil
        ) {
            self.artifactID = artifactID
            self.layer = layer
            self.residualNormAtLayer = residualNormAtLayer
            self.residualNormConvention = residualNormConvention
            self.recommendedAlpha = recommendedAlpha
            self.recommendedAlphaInNormUnits = recommendedAlphaInNormUnits
            self.recommendedFromAgent = recommendedFromAgent
            self.recommendedPromotedBy = recommendedPromotedBy
        }

        /// A denominator is usable only when it is present AND positive.
        public var hasUsableDenominator: Bool {
            guard let residualNormAtLayer else { return false }
            return residualNormAtLayer.isFinite && residualNormAtLayer > 0
        }
    }

    public struct Decision: Sendable, Equatable {
        public var alpha: Double
        public var units: Units
        /// The control's own label — states the units and the layer, always.
        public var alphaLabel: String
        /// Why this number, in one line.
        public var rationale: String
        /// Which averaging rule the denominator was measured under, or
        /// "legacy (pre-stamp)". `nil` in raw mode (there is no denominator to
        /// describe).
        public var conventionNote: String?
        /// The one-click repair, raw mode only: how to give this artifact a
        /// denominator so it can steer in norm units.
        public var backfillHint: String?

        public var isNormUnits: Bool { units == .normUnits }
    }

    /// The whole table, in one total function.
    public static func decide(_ facts: ArtifactFacts) -> Decision {
        let layerLabel = "L\(facts.layer)"
        let normLabel = "α in residual-norm units at \(layerLabel)"
        let rawLabel =
            "α in RAW units at \(layerLabel) — this vector stores no "
            + "residual-norm denominator"
        let conventionNote = ResidualNormConvention.displayLabel(
            residualNormPerLayer: facts.hasUsableDenominator ? [1] : nil,
            stamp: facts.residualNormConvention)

        guard facts.hasUsableDenominator else {
            // Rule 2 + 3: no denominator. A RAW recommendation is adoptable
            // (units already match); a NORM-UNIT one is not, and reusing its
            // numeral raw would change the dose by ~the residual norm.
            if let alpha = facts.recommendedAlpha,
                facts.recommendedAlphaInNormUnits == false
            {
                return Decision(
                    alpha: alpha, units: .raw, alphaLabel: rawLabel,
                    rationale: rationale(
                        .promotedCell(
                            agent: facts.recommendedFromAgent ?? "an agent",
                            promotedBy: facts.recommendedPromotedBy),
                        alpha: alpha, units: .raw),
                    conventionNote: nil,
                    backfillHint: backfillHint(for: facts.artifactID))
            }
            let unusable = facts.recommendedAlpha != nil
                && facts.recommendedAlphaInNormUnits == true
            return Decision(
                alpha: rawUnitsDefault, units: .raw, alphaLabel: rawLabel,
                rationale: rationale(
                    unusable
                        ? .recommendationUnusable(
                            agent: facts.recommendedFromAgent ?? "an agent")
                        : .rawFallback,
                    alpha: rawUnitsDefault, units: .raw),
                conventionNote: nil,
                backfillHint: backfillHint(for: facts.artifactID))
        }

        // Rule 1 + 3: a denominator exists, so norm units are expressible.
        if let alpha = facts.recommendedAlpha {
            // A RAW recommendation stays raw even here — the number means a
            // literal α·v coefficient and is not a fraction of anything.
            let units: Units =
                facts.recommendedAlphaInNormUnits == false ? .raw : .normUnits
            return Decision(
                alpha: alpha, units: units,
                alphaLabel: units == .normUnits ? normLabel : rawLabel,
                rationale: rationale(
                    .promotedCell(
                        agent: facts.recommendedFromAgent ?? "an agent",
                        promotedBy: facts.recommendedPromotedBy),
                    alpha: alpha, units: units),
                conventionNote: units == .normUnits ? conventionNote : nil,
                backfillHint: nil)
        }
        return Decision(
            alpha: normUnitsDefault, units: .normUnits, alphaLabel: normLabel,
            rationale: rationale(
                .conventionDefault, alpha: normUnitsDefault, units: .normUnits),
            conventionNote: conventionNote,
            backfillHint: nil)
    }

    static func rationale(_ provenance: Provenance, alpha: Double, units: Units) -> String {
        let number = formatted(alpha)
        let unitWord = units == .normUnits ? "residual-norm units" : "raw units"
        switch provenance {
        case .promotedCell(let agent, let promotedBy):
            let how =
                promotedBy == "manualOverride"
                ? "manually overridden cell"
                : promotedBy == nil ? "hand-created variant" : "sweep-selected cell"
            return "α \(number) \(unitWord) — the \(how) on agent '\(agent)'"
        case .conventionDefault:
            return
                "α \(number) \(unitWord) — one residual-stream norm; this "
                + "artifact has no promoted cell to inherit"
        case .rawFallback:
            return
                "α \(number) \(unitWord) — this artifact stores no residual-norm "
                + "denominator, so α cannot be denominated"
        case .recommendationUnusable(let agent):
            return
                "α \(number) \(unitWord) — agent '\(agent)' recommends a "
                + "NORM-UNIT α, which needs a denominator this artifact does "
                + "not store; its numeral is not reused as a raw α"
        }
    }

    static func backfillHint(for artifactID: String) -> String {
        "steerlab-cli vectors backfill-norms \(artifactID) "
            + "--corpus prompts/neutral/corpus.jsonl"
    }

    private static func formatted(_ value: Double) -> String {
        value == value.rounded() && abs(value) < 1e9
            ? String(Int(value.rounded()))
            : String(format: "%g", value)
    }

    // MARK: Recommendation lookup

    /// The promoted strength for one vector artifact, drawn from the agent
    /// library.
    ///
    /// An agent's `injections` name the vector artifacts it steers with and
    /// the α selected for each; a `promotion` certificate says that α was
    /// chosen by a declared rule on dev data rather than by hand. Preference
    /// order, so the answer is deterministic and never quietly prefers a
    /// weaker provenance:
    ///
    /// 1. sweep-selected (`promotedBy == "criterion"`),
    /// 2. manual override (`promotedBy == "manualOverride"` — still recorded
    ///    evidence, still better than nothing),
    /// 3. hand-created variant (no certificate),
    ///
    /// then by agent name, so two equally-ranked agents resolve stably.
    /// ABLATION injections are skipped: λ is not an α and never consults a
    /// residual-norm denominator.
    public static func recommendation(
        forVectorArtifactID artifactID: String,
        among variants: [ModelVariantArtifact]
    ) -> (alpha: Double, normUnits: Bool, agent: String, promotedBy: String?)? {
        var best: (rank: Int, name: String, alpha: Double, normUnits: Bool,
                   promotedBy: String?)?
        for variant in variants {
            guard
                let injection = variant.injections.first(where: {
                    $0.vectorArtifactID == artifactID && $0.effectiveMode != .ablate
                })
            else { continue }
            let promotedBy = variant.promotion?.promotedBy
            let rank: Int
            switch promotedBy {
            case "criterion": rank = 0
            case "manualOverride": rank = 1
            default: rank = 2
            }
            let candidate = (
                rank: rank, name: variant.name, alpha: injection.alpha,
                normUnits: variant.alphaInNormUnits, promotedBy: promotedBy)
            if let current = best,
                (current.rank, current.name) <= (candidate.rank, candidate.name)
            {
                continue
            }
            best = candidate
        }
        guard let best else { return nil }
        return (best.alpha, best.normUnits, best.name, best.promotedBy)
    }
}
