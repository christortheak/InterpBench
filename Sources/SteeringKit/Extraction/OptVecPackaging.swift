import Foundation

/// How an OptVec (trained) artifact's bytes are PACKAGED, read off the
/// server-minted `optvec` sidecar block.
///
/// **The trap this exists to close** (open-issues §24). Every other artifact
/// family stores a DIRECTION: the consumer picks α, divides by the layer's
/// residual norm from `residualNormPerLayer`, and scales. The OptVec family
/// stores the vector **pre-scaled to its full trained magnitude** — the row
/// at the trained layer already has `alphaAbsolute` baked in, and injecting
/// it means injecting it as-is, at coefficient 1.
///
/// A consumer that treats the two the same — reading a norms slot as a
/// denominator, or re-normalizing the row and applying its own α — changes
/// the dose by ORDERS OF MAGNITUDE. Anything about to scale an artifact
/// should ask `packaging` first and say what it found rather than assuming
/// the common case.
///
/// This engine cannot MINT the block (OptVec training is a server verb on the
/// science substrate); it decodes, displays, and — via
/// `SteeringVectorSidecar.optvec` — preserves it verbatim through a backfill.
public enum OptVecPackaging {

    /// The marker the server's `optvec_train._save_artifact` stamps.
    public static let preScaledFullMagnitude = "preScaledFullMagnitude"

    /// What an OptVec artifact says about itself, for display and for
    /// scaling decisions. Every field is optional because every field is
    /// absent on some legitimate artifact: pre-§24 artifacts carry no
    /// packaging marker at all, and an `alphaAbsolute`-configured run has no
    /// norm factor and no donor denominator by construction.
    public struct Facts: Sendable, Equatable {
        /// e.g. `"preScaledFullMagnitude"`. `nil` = an OptVec artifact minted
        /// before the marker existed. Unknown packaging, never assumed —
        /// which is exactly what `isPreScaled` refuses to guess at.
        public var packaging: String?
        /// The layer the trained direction actually means anything at.
        public var layer: Int?
        /// The magnitude baked into the stored row.
        public var alphaAbsolute: Double?
        /// α as a fraction of the residual norm, when α was denominated.
        public var alphaNormFactor: Double?
        /// The donor denominator's corpus, source label, and averaging
        /// convention (`ResidualNormConvention`), when a donor supplied one.
        public var residualNormSource: String?
        public var neutralCorpusHash: String?
        public var residualNormConvention: String?
        public var residualNormArtifact: String?

        /// True only when the marker is PRESENT and says pre-scaled. An
        /// unmarked artifact answers `false` here and `true` to
        /// `packagingIsUnknown` — the difference between "safe to scale" and
        /// "nobody recorded", which must not collapse.
        public var isPreScaled: Bool { packaging == preScaledFullMagnitude }

        public var packagingIsUnknown: Bool { packaging == nil }
    }

    /// Reads the block; `nil` when the sidecar carries none (i.e. not an
    /// OptVec artifact).
    public static func facts(from sidecar: SteeringVectorSidecar) -> Facts? {
        guard case .object(let block)? = sidecar.optvec else { return nil }
        let denominator: [String: SidecarJSON]
        if case .object(let nested)? = block["residualNorm"] {
            denominator = nested
        } else {
            denominator = [:]
        }
        return Facts(
            packaging: string(block["vectorPackaging"]),
            layer: number(block["layer"]).map { Int($0) },
            alphaAbsolute: number(block["alphaAbsolute"]),
            alphaNormFactor: number(block["alphaNormFactor"]),
            residualNormSource: string(denominator["residualNormSource"]),
            neutralCorpusHash: string(denominator["neutralCorpusHash"]),
            residualNormConvention: string(denominator["residualNormConvention"]),
            residualNormArtifact: string(denominator["residualNormArtifact"]))
    }

    /// One honest sentence for any surface that shows a vector's provenance.
    /// `nil` for non-OptVec artifacts, so callers can append unconditionally.
    public static func advisory(for sidecar: SteeringVectorSidecar) -> String? {
        guard let facts = facts(from: sidecar) else { return nil }
        if facts.packagingIsUnknown {
            return
                "optvec artifact with NO vectorPackaging marker — minted before "
                + "the stamp; whether its vector is pre-scaled is unrecorded, so "
                + "do not scale it without checking its training run"
        }
        guard facts.isPreScaled else {
            return "optvec artifact packaged as '\(facts.packaging ?? "")' — "
                + "unrecognized packaging; do not scale it"
        }
        var line =
            "optvec artifact: the vector is PRE-SCALED to full trained "
            + "magnitude (do not re-normalize it or apply your own α — "
            + "injecting it is already the trained dose"
        if let alpha = facts.alphaAbsolute {
            line += ", αabsolute \(format(alpha))"
        }
        line += ")"
        if let factor = facts.alphaNormFactor {
            line += "; trained at α \(format(factor)) residual-norm units"
            if let convention = facts.residualNormConvention {
                line += " (denominator convention: \(convention))"
            } else if facts.residualNormSource != nil {
                line += " (denominator convention: \(ResidualNormConvention.legacyLabel))"
            }
        }
        return line
    }

    private static func string(_ value: SidecarJSON?) -> String? {
        if case .string(let text)? = value { return text }
        return nil
    }

    private static func number(_ value: SidecarJSON?) -> Double? {
        if case .number(let value)? = value { return value }
        return nil
    }

    private static func format(_ value: Double) -> String {
        value == value.rounded() && abs(value) < 1e9
            ? String(Int(value.rounded()))
            : String(format: "%g", value)
    }
}
