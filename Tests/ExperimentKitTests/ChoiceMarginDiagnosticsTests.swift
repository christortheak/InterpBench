import Foundation
import Testing

@testable import ExperimentKit

/// D3 — distance from the decision boundary, named correctly.
///
/// A categorical logprob arm was previously described as showing a "ceiling
/// effect" / "saturation". That was the wrong word and the wrong model. A
/// large joint-logprob margin means the item sits far from the boundary, so
/// the FLIP RATE has poor sensitivity — but the log-odds keeps moving
/// continuously and remains a usable readout. True numerical saturation is a
/// separate, rarer thing: the probability hitting the log-odds clamp.
struct ChoiceMarginDiagnosticsTests {

    private func items(_ pairs: [(Double, Double)]) -> [[String: Double]] {
        pairs.map { ["A": $0.0, "B": $0.1] }
    }

    // MARK: the distinction the name gets wrong

    @Test func aWideMarginIsDistanceFromTheBoundaryNotSaturation() throws {
        let report = ChoiceMarginDiagnostics.report(
            optionLogprobsPerItem: items([(-1, -21), (-2, -25), (-1.5, -30)]))
        let text = try #require(report.interpretation)
        #expect(text.contains("FLIP RATE has poor sensitivity"))
        #expect(text.contains("has not saturated"))
        // The continuous readout is still there; only the flip rate is blunt.
        #expect(text.contains("continuous log-odds shift"))
    }

    @Test func trueSaturationIsTheClampAndIsCountedSeparately() throws {
        // A degenerate softmax: log-odds returns the clamp, not a measurement.
        let report = ChoiceMarginDiagnostics.report(
            optionLogprobsPerItem: items([(0, -100), (0, -0.5)]))
        #expect(report.clampedItems == 1)
        let text = try #require(report.interpretation)
        #expect(text.contains("clamp ARTEFACT"))
        #expect(text.contains("true numerical saturation"))
    }

    @Test func aNarrowMarginSaysNothingAlarming() throws {
        let report = ChoiceMarginDiagnostics.report(
            optionLogprobsPerItem: items([(-1, -1.5), (-2, -2.2), (-3, -3.1)]))
        let text = try #require(report.interpretation)
        #expect(!text.contains("FLIP RATE"))
        #expect(report.clampedItems == 0)
    }

    // MARK: the arithmetic

    @Test func theMarginIsTheGapBetweenTheTopTwoOptions() {
        #expect(
            ChoiceMarginDiagnostics.margin(
                optionLogprobs: ["A": -1.0, "B": -3.5]) == 2.5)
        // Three options: the runner-up is the second best, not the worst.
        #expect(
            ChoiceMarginDiagnostics.margin(
                optionLogprobs: ["A": -1.0, "B": -2.0, "C": -50.0]) == 1.0)
    }

    @Test func aSingleOptionHasNoBoundaryAndThereforeNoMargin() {
        // With one option there is nothing to be distant FROM.
        #expect(ChoiceMarginDiagnostics.margin(optionLogprobs: ["A": -1.0]) == nil)
        #expect(ChoiceMarginDiagnostics.margin(optionLogprobs: [:]) == nil)
        let report = ChoiceMarginDiagnostics.report(
            optionLogprobsPerItem: [["A": -1.0]])
        #expect(report.scoredItems == 0)
        #expect(report.marginMedian == nil)
    }

    @Test func quantilesInterpolateLinearly() {
        // Pinned so an even-length sample's median agrees across engines.
        #expect(ChoiceMarginDiagnostics.quantile([1, 2, 3, 4], 0.5) == 2.5)
        #expect(ChoiceMarginDiagnostics.quantile([1, 2, 3], 0.5) == 2.0)
        #expect(ChoiceMarginDiagnostics.quantile([5], 0.5) == 5.0)
    }

    // MARK: the bands are declared, versioned, and never gates

    @Test func theBandsTravelWithTheReportAndCarryAVersion() {
        let report = ChoiceMarginDiagnostics.report(
            optionLogprobsPerItem: items([(-1, -21)]))
        // A silent numeric cutoff deciding which items "count" is exactly the
        // unrecorded analytic choice the firewall exists to prevent.
        #expect(report.bandsVersion == ChoiceMarginDiagnostics.bandsVersion)
        #expect(report.marginBands == ChoiceMarginDiagnostics.marginBands)
        #expect(report.clampEpsilon == ChoiceMarginDiagnostics.logOddsEpsilon)
    }

    @Test func bandCountsAreCumulativeAboveEachEdge() throws {
        let report = ChoiceMarginDiagnostics.report(
            optionLogprobsPerItem: items([(-1, -2), (-1, -4), (-1, -20)]))
        // Margins are 1.0, 3.0, 19.0. The bands are strict (`> band`), so an
        // item sitting exactly ON an edge is not beyond it.
        let beyond = try #require(report.itemsBeyondBand)
        #expect(beyond["above1Nats"] == 2)
        #expect(beyond["above2.5Nats"] == 2)
        #expect(beyond["above5Nats"] == 1)
        #expect(beyond["above10Nats"] == 1)
    }

    /// The clamp must match the instrument's own epsilon, or incidence is
    /// counted against the wrong boundary.
    @Test func theClampMatchesTheInstrument() {
        #expect(ChoiceMarginDiagnostics.logOddsEpsilon == 1e-12)
    }

    // MARK: cross-engine agreement

    @Test func theArithmeticMatchesThePythonTwin() throws {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine",
            "choice-margins.json")
        let cases = try #require(
            try JSONSerialization.jsonObject(with: try Data(contentsOf: url))
                as? [[String: Any]])
        #expect(cases.count >= 6)
        for entry in cases {
            let label = entry["label"] as? String ?? "?"
            let input = try #require(entry["input"] as? [[String: Double]])
            let expected = try #require(entry["report"] as? [String: Any])
            let report = ChoiceMarginDiagnostics.report(
                optionLogprobsPerItem: input)
            #expect(
                report.scoredItems == expected["scoredItems"] as? Int,
                "scored-item drift on '\(label)'")
            if let median = expected["marginMedian"] as? Double {
                let got = try #require(report.marginMedian)
                #expect(abs(got - median) < 1e-9, "median drift on '\(label)'")
            } else {
                #expect(report.marginMedian == nil)
            }
            if let clamped = expected["clampedItems"] as? Int {
                #expect(
                    report.clampedItems == clamped,
                    "clamp-incidence drift on '\(label)'")
            }
        }
    }
}
