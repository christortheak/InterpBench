import Foundation

/// Distance-from-boundary diagnostics for the answer-token instrument (D3).
///
/// A categorical logprob arm was previously described as showing a "ceiling
/// effect" or "saturation". That was the wrong word and the wrong mental
/// model.
///
/// What was observed was a large **joint-logprob margin**: the gap between the
/// winning option and the runner-up. A large margin means the item sits far
/// from the decision boundary, so the *flip rate* — how often the argmax
/// changes — has poor sensitivity: an intervention can move the log-odds a
/// long way without ever flipping a single item. The log-odds itself keeps
/// moving continuously and remains a perfectly good readout. Saying "the
/// instrument saturated" invites the conclusion that the measurement failed,
/// when the correct conclusion is that the *flip rate* was the wrong statistic.
///
/// TRUE numerical saturation is a different thing and does occur: `logOdds`
/// clamps probabilities away from 0 and 1 by `epsilon` before taking the
/// ratio, so a degenerate softmax yields a clamp ARTEFACT rather than a
/// measurement. That is counted separately here.
///
/// **Every threshold below is a declared, versioned DIAGNOSTIC band.** None
/// gates anything. A silent numeric cutoff deciding which items "count" is
/// exactly the unrecorded analytic choice the firewall exists to prevent, so
/// the band edges travel with the report and carry a version string.
///
/// Cross-engine twin: `Server/steerlab_server/experiment/choice_margin.py`.
public enum ChoiceMarginDiagnostics {

    /// Versioned so a report says which band edges produced its counts.
    /// Bumping the bands MUST bump this — a comparison across two versions of
    /// the bands is not a comparison.
    public static let bandsVersion = "d3-2026-07-26"

    /// Joint-logprob margin bands, in nats. Diagnostic only.
    public static let marginBands: [Double] = [1.0, 2.5, 5.0, 10.0]

    /// The clamp `ChoiceResult.logOdds` applies before the ratio. Must match
    /// the instrument's own epsilon, or clamp incidence is counted against
    /// the wrong boundary.
    public static let logOddsEpsilon = 1e-12

    /// Gap between the best and second-best option, in nats. Nil when fewer
    /// than two options were scored: with one option there is no boundary, so
    /// there is no distance from it.
    public static func margin(optionLogprobs: [String: Double]) -> Double? {
        let totals = optionLogprobs.values.filter(\.isFinite).sorted(by: >)
        guard totals.count >= 2 else { return nil }
        return totals[0] - totals[1]
    }

    /// True when the softmax over these logprobs is degenerate enough that
    /// `logOdds` returns a clamp artefact rather than a measured value.
    public static func isClamped(
        optionLogprobs: [String: Double], epsilon: Double = logOddsEpsilon
    ) -> Bool {
        let totals = optionLogprobs.values.filter(\.isFinite)
        guard totals.count >= 2, let peak = totals.max() else { return false }
        let weights = totals.map { exp($0 - peak) }
        let z = weights.reduce(0, +)
        guard z > 0 else { return true }
        return weights.contains { weight in
            let p = weight / z
            return p <= epsilon || (1 - p) <= epsilon
        }
    }

    /// Linear-interpolation quantile (numpy's default "linear" method), so the
    /// two engines agree on the median of an even-length sample.
    static func quantile(_ sorted: [Double], _ q: Double) -> Double {
        guard !sorted.isEmpty else { return .nan }
        guard sorted.count > 1 else { return sorted[0] }
        let position = q * Double(sorted.count - 1)
        let low = Int(position.rounded(.down))
        let high = Int(position.rounded(.up))
        if low == high { return sorted[low] }
        let weight = position - Double(low)
        return sorted[low] * (1 - weight) + sorted[high] * weight
    }

    public struct Report: Codable, Sendable, Equatable {
        public var bandsVersion: String
        public var scoredItems: Int
        public var marginBands: [Double]
        public var clampEpsilon: Double
        public var marginMin: Double?
        public var marginP25: Double?
        public var marginMedian: Double?
        public var marginP75: Double?
        public var marginMax: Double?
        public var marginMean: Double?
        /// How many items sit further than each band from the boundary.
        public var itemsBeyondBand: [String: Int]?
        /// TRUE saturation: log-odds is a clamp artefact for these items.
        public var clampedItems: Int?
        public var interpretation: String?
    }

    public static func report(optionLogprobsPerItem: [[String: Double]]) -> Report {
        var margins: [Double] = []
        var clamped = 0
        for logprobs in optionLogprobsPerItem {
            guard let value = margin(optionLogprobs: logprobs) else { continue }
            margins.append(value)
            if isClamped(optionLogprobs: logprobs) { clamped += 1 }
        }
        guard !margins.isEmpty else {
            return Report(
                bandsVersion: bandsVersion, scoredItems: 0,
                marginBands: marginBands, clampEpsilon: logOddsEpsilon)
        }
        let ordered = margins.sorted()
        var beyond: [String: Int] = [:]
        for band in marginBands {
            let label = band.formatted(.number.precision(.fractionLength(0 ... 3)))
            beyond["above\(label)Nats"] = ordered.count { $0 > band }
        }
        return Report(
            bandsVersion: bandsVersion,
            scoredItems: ordered.count,
            marginBands: marginBands,
            clampEpsilon: logOddsEpsilon,
            marginMin: ordered.first,
            marginP25: quantile(ordered, 0.25),
            marginMedian: quantile(ordered, 0.5),
            marginP75: quantile(ordered, 0.75),
            marginMax: ordered.last,
            marginMean: ordered.reduce(0, +) / Double(ordered.count),
            itemsBeyondBand: beyond,
            clampedItems: clamped,
            interpretation: interpretation(ordered: ordered, clamped: clamped))
    }

    /// Say what the numbers mean, in the words that are actually correct.
    static func interpretation(ordered: [Double], clamped: Int) -> String {
        let median = quantile(ordered, 0.5)
        let widest = marginBands.last ?? 10
        let far = ordered.count { $0 > widest }
        var parts = [
            "median joint-logprob margin "
                + median.formatted(.number.precision(.fractionLength(2)))
                + " nats over \(ordered.count) scored "
                + "item\(ordered.count == 1 ? "" : "s")"
        ]
        if far > 0 {
            parts.append(
                "\(far) item\(far == 1 ? "" : "s") sit more than "
                    + widest.formatted(.number.precision(.fractionLength(0 ... 3)))
                    + " nats from the decision boundary, where the FLIP RATE "
                    + "has poor sensitivity — an intervention can move the "
                    + "log-odds a long way without flipping any item. Read the "
                    + "continuous log-odds shift, not the flip rate; the "
                    + "instrument has not saturated")
        }
        if clamped > 0 {
            parts.append(
                "\(clamped) item\(clamped == 1 ? "" : "s") produced a "
                    + "probability at the \(logOddsEpsilon) clamp, so their "
                    + "log-odds is a clamp ARTEFACT rather than a measurement "
                    + "— this is true numerical saturation and those items "
                    + "carry no usable magnitude")
        }
        return parts.joined(separator: "; ")
    }
}
