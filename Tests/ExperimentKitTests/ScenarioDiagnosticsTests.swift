import Foundation
import Testing

@testable import ExperimentKit

/// D1 — keep the working that produced the accuracy number.
///
/// Validate computed every projection and the class means defining the
/// midpoint, then returned one number. When nine virtues all score near
/// chance, that number cannot distinguish "the direction does not read the
/// concept" from "it ranks scenarios correctly but the midpoint sits in the
/// wrong place" — a threshold problem, not a vector problem.
struct ScenarioDiagnosticsTests {

    private func report(
        projections: [Double], labels: [Bool], threshold: Double
    ) throws -> ScenarioDiagnostics.Report {
        try ScenarioDiagnostics.report(
            scenarioIDs: projections.indices.map { "s\($0)" },
            scenarioTexts: projections.indices.map { "scenario \($0)" },
            projections: projections, labels: labels, threshold: threshold,
            classMeans: ["positive": 1.0, "negative": -1.0],
            layer: 3, directionNorm: 1.0)
    }

    // MARK: the case accuracy cannot express

    @Test func aPerfectRankingWithABadThresholdScoresChanceButAUCOne() throws {
        // The whole reason AUC is here. Ranking is perfect; the threshold sits
        // above every point, so every item is predicted negative.
        let result = try report(
            projections: [2.0, 1.5, -1.5, -2.0],
            labels: [true, true, false, false], threshold: 5.0)
        #expect(result.accuracy == 0.5)
        #expect(result.auc == 1.0)
        // ...and the margins say WHY: everything is below the boundary.
        #expect(result.rows.allSatisfy { $0.margin < 0 })
    }

    @Test func aGoodThresholdAgreesWithTheRanking() throws {
        let result = try report(
            projections: [2.0, 1.5, -1.5, -2.0],
            labels: [true, true, false, false], threshold: 0.0)
        #expect(result.accuracy == 1.0)
        #expect(result.auc == 1.0)
        #expect(result.balancedAccuracy == 1.0)
    }

    // MARK: AUC edge cases, pinned deliberately

    @Test func tiesContributeExactlyOneHalf() {
        // Otherwise an all-ties direction scores 0 or 1 depending on
        // comparison order — a number that looks meaningful and is not.
        #expect(
            ScenarioDiagnostics.auc(
                projections: [1.0, 1.0, 1.0, 1.0],
                labels: [true, true, false, false]) == 0.5)
    }

    @Test func aSingleClassHasNoAUC() {
        // With one class there is nothing to separate; any number would be
        // an artefact of the formula rather than a fact about the direction.
        #expect(
            ScenarioDiagnostics.auc(
                projections: [1.0, 2.0, 3.0],
                labels: [true, true, true]) == nil)
        #expect(
            ScenarioDiagnostics.auc(projections: [], labels: []) == nil)
    }

    @Test func aPerfectlyInvertedDirectionScoresZero() {
        // Distinguishable from "no signal" (0.5) — an inverted vector is a
        // sign problem, not a dead one.
        #expect(
            ScenarioDiagnostics.auc(
                projections: [-2.0, -1.5, 1.5, 2.0],
                labels: [true, true, false, false]) == 0.0)
    }

    // MARK: the 2026-08-01 27B incident shape

    /// The `fair` validate row: a designatedReference story-corpus midpoint
    /// sat below every scenario projection, so accuracy 0.50 with tp=fp and
    /// an empty negative row — while the ranking was nearly clean. The
    /// calibration re-thresholds at the held-out classes' own midpoint and
    /// reads the separation the transfer threshold hid.
    @Test func theIncidentShapeCalibrationRecoversWhatTransferAccuracyLost() throws {
        let result = try report(
            projections: [10.0, 9.0, 8.0, 3.0, 2.0, 1.0],
            labels: [true, true, true, false, false, false], threshold: -5.0)
        #expect(result.accuracy == 0.5)
        #expect(result.confusion == ["tp": 3, "fp": 3, "tn": 0, "fn": 0])
        #expect(result.oneSidedPredictions == true)
        let calibration = try #require(result.heldOutCalibration)
        #expect(calibration.threshold == 5.5)
        #expect(calibration.classMeans == ["positive": 9.0, "negative": 2.0])
        #expect(calibration.accuracy == 1.0)
        #expect(calibration.balancedAccuracy == 1.0)
        #expect(calibration.confusion == ["tp": 3, "fp": 0, "tn": 3, "fn": 0])
    }

    @Test func aWellPlacedThresholdIsNotFlaggedOneSided() throws {
        let result = try report(
            projections: [2.0, -1.0, 1.0, -2.0],
            labels: [true, true, false, false], threshold: 0.0)
        #expect(result.oneSidedPredictions == false)
    }

    /// An inverted direction must read below 0.5 here exactly as it does in
    /// AUC — orienting it away would flip a sign problem into respectability.
    @Test func calibrationPreservesAnInvertedSign() throws {
        let result = try report(
            projections: [-2.0, -1.5, 1.5, 2.0],
            labels: [true, true, false, false], threshold: 0.0)
        #expect(try #require(result.heldOutCalibration).accuracy == 0.0)
        #expect(result.auc == 0.0)
    }

    @Test func aSingleClassHasNoCalibration() throws {
        let result = try report(
            projections: [1.0, 2.0], labels: [true, true], threshold: 0.0)
        #expect(result.heldOutCalibration == nil)
        // One-sidedness is still a fact about the transfer threshold.
        #expect(result.oneSidedPredictions == true)
    }

    // MARK: descriptive statistics only (D2)

    @Test func theConfusionMatrixAndBalancedAccuracyAreReported() throws {
        let result = try report(
            projections: [2.0, -1.0, 1.0, -2.0],
            labels: [true, true, false, false], threshold: 0.0)
        #expect(result.confusion == ["tp": 1, "fn": 1, "fp": 1, "tn": 1])
        #expect(result.classCounts == ["positive": 2, "negative": 2])
        #expect(result.sensitivity == 0.5)
        #expect(result.specificity == 0.5)
        #expect(result.balancedAccuracy == 0.5)
    }

    @Test func theWilsonIntervalStaysInsideTheScale() throws {
        // A normal-approximation interval on 4/4 would run past 1.0 and
        // report an impossible bound; these sets are small and often extreme.
        let interval = try #require(
            ScenarioDiagnostics.wilsonInterval(successes: 4, total: 4))
        #expect(interval.low > 0 && interval.low < 1)
        #expect(interval.high == 1.0)
        let zero = try #require(
            ScenarioDiagnostics.wilsonInterval(successes: 0, total: 4))
        #expect(zero.low == 0.0)
        #expect(zero.high < 1.0)
        #expect(ScenarioDiagnostics.wilsonInterval(successes: 0, total: 0) == nil)
    }

    /// No p-values. The binomial null assumes independence, and these sets are
    /// generated per concept by one agent over shared topics with matched
    /// pairs — both induce correlation, so a binomial p would understate
    /// variance. The inferential design is a decision to be declared.
    @Test func noInferentialStatisticsAreShipped() throws {
        let result = try report(
            projections: [1.0, -1.0], labels: [true, false], threshold: 0.0)
        let encoded = try JSONEncoder().encode(result)
        let text = String(decoding: encoded, as: UTF8.self)
        #expect(!text.contains("pValue"))
        #expect(!text.contains("\"p\""))
        #expect(!text.contains("fdr"))
    }

    // MARK: row identity

    @Test func everyRowCarriesPositionAndIdentity() throws {
        let result = try report(
            projections: [1.0, -1.0], labels: [true, false], threshold: 0.0)
        #expect(result.rows.map(\.index) == [0, 1])
        #expect(result.rows.map(\.id) == ["s0", "s1"])
        // A line number alone stops meaning anything once the file is
        // re-ordered, so the row's own bytes identify it too.
        #expect(
            result.rows[0].rowHash
                == ScenarioDiagnostics.rowHash(text: "scenario 0", label: true))
        #expect(result.rows[0].rowHash != result.rows[1].rowHash)
    }

    // MARK: cross-engine agreement

    /// Generated by the Python twin; regenerate with
    /// `scripts/regenerate-cross-engine-fixtures.py`. Tie handling,
    /// single-class behaviour and Wilson bounds are exactly the decisions two
    /// independent implementations drift on.
    @Test func theArithmeticMatchesThePythonTwin() throws {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine",
            "scenario-diagnostics.json")
        let cases = try #require(
            try JSONSerialization.jsonObject(with: try Data(contentsOf: url))
                as? [[String: Any]])
        #expect(cases.count >= 6)
        for entry in cases {
            let label = entry["label"] as? String ?? "?"
            let input = try #require(entry["input"] as? [String: Any])
            let expected = try #require(entry["report"] as? [String: Any])
            let result = try report(
                projections: try #require(input["projections"] as? [Double]),
                labels: try #require(input["labels"] as? [Bool]),
                threshold: try #require(input["threshold"] as? Double))

            #expect(
                result.accuracy == expected["accuracy"] as? Double,
                "accuracy drift on '\(label)'")
            #expect(
                result.auc == expected["auc"] as? Double,
                "AUC drift on '\(label)' — tie or single-class handling diverged")
            #expect(
                result.balancedAccuracy == expected["balancedAccuracy"] as? Double,
                "balanced-accuracy drift on '\(label)'")
            let interval = expected["naiveItemLevelInterval95"] as? [Double]
            if let interval, let got = result.naiveItemLevelInterval95 {
                #expect(abs(got[0] - interval[0]) < 1e-9, "Wilson low drift on '\(label)'")
                #expect(abs(got[1] - interval[1]) < 1e-9, "Wilson high drift on '\(label)'")
            } else {
                #expect(result.naiveItemLevelInterval95 == nil && interval == nil)
            }
            #expect(
                result.oneSidedPredictions == expected["oneSidedPredictions"] as? Bool,
                "one-sided flag drift on '\(label)'")
            let calibration = expected["heldOutCalibration"] as? [String: Any]
            if let calibration, let got = result.heldOutCalibration {
                #expect(
                    got.threshold == calibration["threshold"] as? Double,
                    "calibration threshold drift on '\(label)'")
                #expect(
                    got.accuracy == calibration["accuracy"] as? Double,
                    "calibration accuracy drift on '\(label)'")
                #expect(
                    got.balancedAccuracy == calibration["balancedAccuracy"] as? Double,
                    "calibration balanced-accuracy drift on '\(label)'")
                #expect(
                    got.confusion == calibration["confusion"] as? [String: Int],
                    "calibration confusion drift on '\(label)'")
            } else {
                #expect(
                    result.heldOutCalibration == nil && calibration == nil,
                    "calibration presence drift on '\(label)'")
            }
        }
    }
}

/// Honesty fixes from the engineer review (2026-07-26).
struct ScenarioDiagnosticsHonestyTests {

    @Test func identicalTextWithOppositeLabelsIsTwoRows() {
        // Hashing the text alone gave them one identity, making a diagnostic
        // record ambiguous about which row it describes.
        #expect(
            ScenarioDiagnostics.rowHash(text: "same words", label: true)
                != ScenarioDiagnostics.rowHash(text: "same words", label: false))
    }

    @Test func unequalInputsRefuseRatherThanInventLabels() {
        // This used to pad a missing label with `false` while the Python twin
        // truncated via `zip` — the same malformed input produced two
        // different answers, neither flagged.
        #expect(throws: (any Error).self) {
            try ScenarioDiagnostics.report(
                scenarioIDs: ["a", "b"], scenarioTexts: ["a", "b"],
                projections: [1.0], labels: [true, false], threshold: 0,
                classMeans: [:], layer: 0, directionNorm: 1)
        }
    }

    @Test func theIntervalNamesItsAssumption() throws {
        // An interval is an inferential object whatever it is labelled, so
        // the field states the assumption these sets violate.
        let result = try ScenarioDiagnostics.report(
            scenarioIDs: ["a", "b"], scenarioTexts: ["a", "b"],
            projections: [1.0, -1.0], labels: [true, false], threshold: 0,
            classMeans: [:], layer: 0, directionNorm: 1)
        #expect(result.naiveItemLevelInterval95 != nil)
        let text = String(decoding: try JSONEncoder().encode(result), as: UTF8.self)
        #expect(text.contains("naiveItemLevelInterval95"))
        #expect(!text.contains("accuracyInterval95"))
    }
}
