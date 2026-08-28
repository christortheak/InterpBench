import Foundation
import Testing

@testable import ExperimentKit

/// WHY a sweep selected nothing — the constraints, or the objective.
///
/// Review round 9, finding 6: this engine always recorded "no cell passed the
/// capability/coherence gates". That is one of two possible reasons and often
/// the wrong one, and the server has distinguished them since 2026-07-26 —
/// after a practicalwisdom sweep in which all 36 cells sat inside both
/// constraints and every objective value was NEGATIVE, and the run still
/// blamed the gates. A researcher reading that goes and loosens a tolerance
/// that was never binding.
///
/// Twin of `Server/tests/test_sweep_promote.py`'s
/// `test_no_selection_reason_*`, and the sentences are the server's own.
@Suite struct NoSelectionReasonTests {

    static let baseline = SweepSelectionRule.Baseline(
        metric: 0, distinct2: 0.99, batteryAccuracy: 0.9)

    private func criterion(
        _ selection: ExperimentManifest.SweepSelection? = .init(
            objective: .init(metric: "logprobShift"))
    ) throws -> SweepSelectionRule.Resolved {
        try SweepSelectionRule.resolve(selection)
    }

    /// The practicalwisdom shape: everything eligible, nothing better.
    @Test func anEligibleGridThatNeverBeatsBaselineBlamesTheObjective() throws {
        let criterion = try criterion()
        let eligibleButNegative = [
            SweepSelectionRule.Cell(
                layer: 31, alpha: 0.05, metric: -0.357, distinct2: 0.98,
                batteryAccuracy: 0.9),
            SweepSelectionRule.Cell(
                layer: 41, alpha: 0.17, metric: -2.60, distinct2: 0.97,
                batteryAccuracy: 0.9),
        ]
        #expect(
            SweepSelectionRule.select(
                cells: eligibleButNegative, baseline: Self.baseline,
                criterion: criterion) == nil)
        let reason = SweepSelectionRule.noSelectionReason(
            cells: eligibleButNegative, baseline: Self.baseline,
            criterion: criterion)
        #expect(reason.contains("beat the baseline"))
        #expect(reason.contains("OPPOSITE way"))
        #expect(reason.contains("inside both constraints"))
        #expect(!reason.contains("capability/coherence gates"))
        // The binding numbers, named where the server names them.
        #expect(
            reason == "no eligible cell beat the baseline logprobShift "
                + "(-0.357 vs baseline 0) — the objective moved the OPPOSITE "
                + "way from the declared direction; all 2 cells were inside "
                + "both constraints")
    }

    /// Genuinely gate-blocked: the old sentence is correct here, and kept —
    /// now with the numbers that decided it.
    @Test func aGridTheGatesRefusedStillBlamesTheGates() throws {
        let criterion = try criterion()
        let gated = [
            SweepSelectionRule.Cell(
                layer: 31, alpha: 0.05, metric: 5, distinct2: 0.01,
                batteryAccuracy: 0.1)
        ]
        let reason = SweepSelectionRule.noSelectionReason(
            cells: gated, baseline: Self.baseline, criterion: criterion)
        // The binding numbers, not just the verdict: a criterion that
        // declared no coherence keys means the legacy absolute rule, and the
        // sentence says which rule refused.
        #expect(
            reason == "no cell passed the capability/coherence gates "
                + "(tolerance 0.15, coherence floor 0.45 (absolute "
                + "distinct-2))")
    }

    /// Mixed: some blocked, and the survivors still lose on the objective.
    @Test func aMixedGridNamesBothHalves() throws {
        let criterion = try criterion()
        let cells = [
            SweepSelectionRule.Cell(
                layer: 31, alpha: 0.05, metric: -0.357, distinct2: 0.98,
                batteryAccuracy: 0.9),
            SweepSelectionRule.Cell(
                layer: 41, alpha: 0.17, metric: -2.60, distinct2: 0.97,
                batteryAccuracy: 0.9),
            SweepSelectionRule.Cell(
                layer: 31, alpha: 0.05, metric: 5, distinct2: 0.01,
                batteryAccuracy: 0.1),
        ]
        let reason = SweepSelectionRule.noSelectionReason(
            cells: cells, baseline: Self.baseline, criterion: criterion)
        #expect(reason.contains("beat the baseline"))
        #expect(reason.contains("1 of 3 cells also failed"))
    }

    @Test func anEmptyGridSaysSo() throws {
        #expect(
            SweepSelectionRule.noSelectionReason(
                cells: [], baseline: Self.baseline, criterion: try criterion(nil))
                == "the sweep measured no cells")
    }

    /// The gate sentence names WHICH coherence rule refused, so a reader can
    /// tell a baseline-relative bar from a legacy absolute one.
    @Test func theGateSentenceNamesTheCoherenceRuleInForce() throws {
        let degenerate = [
            SweepSelectionRule.Cell(
                layer: 20, alpha: 0.4, metric: 0.9, distinct2: 0.535,
                batteryAccuracy: 0.9)
        ]
        let relative = try SweepSelectionRule.resolve(
            .init(
                constraints: .init(
                    coherenceRatioToBaseline: 0.85,
                    coherenceAbsoluteBackstop: 0.6)))
        let reason = SweepSelectionRule.noSelectionReason(
            cells: degenerate, baseline: Self.baseline, criterion: relative)
        #expect(
            reason.contains("coherence floor 0.85× the α=0 baseline's distinct-2"))
        #expect(reason.contains("backstop 0.6"))
        let legacy = try SweepSelectionRule.resolve(
            .init(constraints: .init(coherenceFloor: 0.9)))
        #expect(
            SweepSelectionRule.noSelectionReason(
                cells: degenerate, baseline: Self.baseline, criterion: legacy)
                .contains("coherence floor 0.9 (absolute distinct-2)"))
    }

    /// The numbers are rendered as Python renders them, so the twin texts are
    /// equal byte for byte: `%g`, not Swift's `Double` description.
    @Test func theNumbersAreRenderedTheWayTheServerRendersThem() throws {
        #expect(SweepSelectionRule.g(1) == "1")
        #expect(SweepSelectionRule.g(0.85) == "0.85")
        #expect(SweepSelectionRule.g(0.1 + 0.2) == "0.3")
        let wholeRatio = try SweepSelectionRule.resolve(
            .init(
                constraints: .init(
                    coherenceRatioToBaseline: 1, coherenceAbsoluteBackstop: 0.6)))
        #expect(
            wholeRatio.coherenceSummary
                == "coherence floor 1× the α=0 baseline's distinct-2, "
                    + "backstop 0.6")
    }
}
