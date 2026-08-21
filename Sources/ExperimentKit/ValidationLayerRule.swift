import Foundation

/// Which layer convergent validity is read at — declared, not inferred (D4).
///
/// The historical rule was "the layer a condition steers this concept at, else
/// mid-network". That made the validation layer a SIDE EFFECT of the injection
/// conditions: to revalidate at a different depth you had to add or edit a
/// steering condition, which is a different decision entirely. On 2026-07-26 a
/// researcher testing a readout-layer hypothesis had to hand-write nine
/// conditions at layer 41 to move the validation read — the manifest offered
/// no way to say the thing they meant.
///
/// The plan's first draft replaced that with a hardcoded default (two-thirds
/// depth for grand-mean, midpoint for paired). That swaps one hidden choice
/// for another. The layer is a measurement decision, so it is declared:
///
/// 1. `validationLayer` — an absolute index, when the researcher knows it.
/// 2. `validationLayerFraction` — a depth fraction, when the study should
///    read at the same relative depth across model sizes.
/// 3. Legacy: the first condition slot steering this concept (preserved
///    exactly, so existing manifests keep their numbers and their hashes).
/// 4. Mid-network.
///
/// Both the resolved index AND the fraction are reported, because "layer 41"
/// means nothing without the depth it represents, and "0.66" means nothing
/// without the model it resolved against.
///
/// Cross-engine twin: `Server/steerlab_server/experiment/validation_layer.py`.
public enum ValidationLayerRule {

    public enum Source: String, Sendable, Equatable {
        case declaredIndex
        case declaredFraction
        case steeringCondition
        case midNetwork
    }

    public struct Resolution: Sendable, Equatable {
        public var layer: Int
        public var layerCount: Int
        public var source: Source

        /// Where the resolved layer sits in the network, 0…1.
        public var depthFraction: Double {
            guard layerCount > 1 else { return 0 }
            return Double(layer) / Double(layerCount - 1)
        }

        /// The line a run log and a report should both carry: an index is
        /// meaningless without its depth, and a depth without its model.
        public var summary: String {
            let depth = depthFraction.formatted(
                .number.precision(.fractionLength(0 ... 2)))
            let because =
                switch source {
                case .declaredIndex: "declared validationLayer"
                case .declaredFraction: "declared validationLayerFraction"
                case .steeringCondition:
                    "inherited from a steering condition (legacy rule — declare "
                        + "validationLayer to say it directly)"
                case .midNetwork:
                    "mid-network default (legacy rule — declare validationLayer "
                        + "to say it directly)"
                }
            return "layer \(layer) of \(layerCount), \(depth) depth — \(because)"
        }
    }

    /// Resolve the read layer for one concept.
    public static func resolve(
        concept: String,
        declaredLayer: Int?,
        declaredFraction: Double?,
        conditionLayer: Int?,
        layerCount: Int
    ) -> Resolution {
        let maxIndex = max(0, layerCount - 1)
        func clamp(_ value: Int) -> Int { min(max(0, value), maxIndex) }

        if let declaredLayer {
            return Resolution(
                layer: clamp(declaredLayer), layerCount: layerCount,
                source: .declaredIndex)
        }
        if let declaredFraction, declaredFraction.isFinite {
            // Truncating, clamped — the same rule `SweepSpec.resolvedLayers`
            // applies, so a fraction means one thing across the app.
            let scaled = Int(declaredFraction * Double(layerCount))
            return Resolution(
                layer: clamp(scaled), layerCount: layerCount,
                source: .declaredFraction)
        }
        if let conditionLayer {
            return Resolution(
                layer: clamp(conditionLayer), layerCount: layerCount,
                source: .steeringCondition)
        }
        return Resolution(
            layer: clamp(layerCount / 2), layerCount: layerCount,
            source: .midNetwork)
    }

    /// Every layer convergent validity reads at, in declared order.
    ///
    /// The depth LIST exists for the validate-at-the-sweep-layers policy
    /// (2026-08-01): the steering sweep spans a band of layers, and the
    /// reading certificate should cover every layer the sweep may promote —
    /// measured in ONE run, since the scenario activations are captured once
    /// for all layers.
    ///
    /// Declaring a list and a scalar together is refused by `violation`;
    /// here a declared list wins, else the scalar path resolves as before
    /// (so every existing manifest keeps its single resolution). Two
    /// declared entries that resolve to the SAME layer throw: the report
    /// keys per-depth entries by layer, and silently collapsing "0.60 and
    /// 0.61 of 6 layers" into one row would misreport what was declared.
    ///
    /// Python twin: `validation_layer.resolve_all`.
    public static func resolveAll(
        concept: String,
        declaredLayers: [Int]? = nil,
        declaredFractions: [Double]? = nil,
        declaredLayer: Int? = nil,
        declaredFraction: Double? = nil,
        conditionLayer: Int? = nil,
        layerCount: Int
    ) throws -> [Resolution] {
        let maxIndex = max(0, layerCount - 1)
        func clamp(_ value: Int) -> Int { min(max(0, value), maxIndex) }

        var resolutions: [Resolution] = []
        var declaredValues: [String] = []
        let field: String
        if let declaredLayers, !declaredLayers.isEmpty {
            field = "validationLayers"
            for layer in declaredLayers {
                if let refusal = rangeRefusal(
                    declaredLayer: layer, layerCount: layerCount)
                {
                    throw ExperimentError(reason: refusal)
                }
                resolutions.append(Resolution(
                    layer: clamp(layer), layerCount: layerCount,
                    source: .declaredIndex))
                declaredValues.append("\(layer)")
            }
        } else if let declaredFractions, !declaredFractions.isEmpty {
            field = "validationLayerFractions"
            for fraction in declaredFractions {
                resolutions.append(Resolution(
                    layer: clamp(Int(fraction * Double(layerCount))),
                    layerCount: layerCount, source: .declaredFraction))
                declaredValues.append("\(fraction)")
            }
        } else {
            return [
                resolve(
                    concept: concept, declaredLayer: declaredLayer,
                    declaredFraction: declaredFraction,
                    conditionLayer: conditionLayer, layerCount: layerCount)
            ]
        }
        var seen: [Int: String] = [:]
        for (declared, resolution) in zip(declaredValues, resolutions) {
            if let earlier = seen[resolution.layer] {
                throw ExperimentError(
                    reason: "\(field) entries \(earlier) and \(declared) both "
                        + "resolve to layer \(resolution.layer) on this "
                        + "\(layerCount)-layer model — declared depths must "
                        + "stay distinct once resolved, or the per-depth "
                        + "report would silently collapse them")
            }
            seen[resolution.layer] = declared
        }
        return resolutions
    }

    /// An EXPLICIT declaration outside the model's depth, once depth is
    /// known. Nil when the declaration fits, or when nothing was declared.
    ///
    /// `resolve` clamps, which is right for the LEGACY paths — a condition
    /// layer inherited from a study built for a deeper model, or the
    /// mid-network fallback, should bend rather than break. It is wrong for
    /// an explicit `validationLayer: 100` on a 62-layer model: that is a
    /// scientific declaration the researcher wrote down, and silently reading
    /// layer 61 instead answers a different question without saying so.
    ///
    /// Separate from `violation` because depth is not known at verify time —
    /// no model is loaded there — so this is checked where the layer is
    /// actually resolved.
    public static func rangeRefusal(
        declaredLayer: Int?, layerCount: Int
    ) -> String? {
        guard let declaredLayer, layerCount > 0 else { return nil }
        guard declaredLayer >= layerCount else { return nil }
        return "validationLayer \(declaredLayer) is outside this model's depth "
            + "(\(layerCount) layers, so the last index is \(layerCount - 1)) — "
            + "an explicit declaration is not silently clamped; declare a layer "
            + "that exists, or a validationLayerFraction if the study should "
            + "follow depth across model sizes"
    }

    /// Declaring more than one depth field is ambiguous: an index and a
    /// fraction disagree on every model whose depth is not exactly what the
    /// author assumed, and a scalar next to a list leaves which one the run
    /// reads unsaid. Exactly one of the four may be declared.
    ///
    /// Python twin: `validation_layer.violation`.
    public static func violation(
        declaredLayer: Int?, declaredFraction: Double?,
        declaredLayers: [Int]? = nil, declaredFractions: [Double]? = nil
    ) -> String? {
        var declared: [String] = []
        if declaredLayer != nil { declared.append("validationLayer") }
        if declaredFraction != nil { declared.append("validationLayerFraction") }
        if declaredLayers != nil { declared.append("validationLayers") }
        if declaredFractions != nil { declared.append("validationLayerFractions") }
        if declared.count > 1 {
            if declared == ["validationLayer", "validationLayerFraction"] {
                return "validationLayer and validationLayerFraction are both "
                    + "declared — they disagree on any model whose depth differs "
                    + "from the one assumed; declare exactly one"
            }
            return declared.joined(separator: " and ")
                + " are declared together — the run cannot know which depth "
                + "declaration to read; declare exactly one"
        }
        if let declaredLayer, declaredLayer < 0 {
            return "validationLayer must be a non-negative layer index "
                + "(got \(declaredLayer))"
        }
        if let declaredFraction,
            !declaredFraction.isFinite || declaredFraction < 0 || declaredFraction > 1
        {
            return "validationLayerFraction must be a number in [0, 1] "
                + "(got \(declaredFraction))"
        }
        if let declaredLayers {
            guard !declaredLayers.isEmpty else {
                return "validationLayers must be a non-empty list — an empty "
                    + "list declares nothing; remove the field to use the "
                    + "legacy rule"
            }
            for layer in declaredLayers where layer < 0 {
                return "validationLayers entries must be a non-negative layer "
                    + "index (got \(layer))"
            }
            if Set(declaredLayers).count != declaredLayers.count {
                return "validationLayers contains duplicate entries — each "
                    + "declared depth is measured once"
            }
        }
        if let declaredFractions {
            guard !declaredFractions.isEmpty else {
                return "validationLayerFractions must be a non-empty list — "
                    + "an empty list declares nothing; remove the field to "
                    + "use the legacy rule"
            }
            for fraction in declaredFractions
            where !fraction.isFinite || fraction < 0 || fraction > 1 {
                return "validationLayerFractions entries must be a number in "
                    + "[0, 1] (got \(fraction))"
            }
            if Set(declaredFractions).count != declaredFractions.count {
                return "validationLayerFractions contains duplicate entries — "
                    + "each declared depth is measured once"
            }
        }
        return nil
    }
}
