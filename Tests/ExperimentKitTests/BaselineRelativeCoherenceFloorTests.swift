import Foundation
import Testing

@testable import ExperimentKit

/// THE BASELINE-RELATIVE COHERENCE FLOOR (maintainer ruling, round 8).
///
/// An absolute distinct-2 floor gates against a fixed number, and a fixed
/// number cannot know what the model's own prose looks like. The cell that
/// forced this: distinct-2 0.535 against a baseline of 0.989, output 65%
/// longer, and a logprobShift that was repetition rather than steering. It
/// cleared the absolute 0.45 and was recommended.
///
/// What has to be true:
///
/// - a NEW declaration gates on `ratio × baseline` AND an absolute backstop;
/// - an EXISTING pinned criterion keeps the absolute semantics it ran under,
///   decided by the PRESENCE of the new fields and nothing else;
/// - the ranges refuse loudly, including the ascending-sanity rule;
/// - the ratio is REPORTED for every cell whichever rule is in force, and a
///   length-inflated cell is flagged without being gated.
///
/// Every expectation here is duplicated in `Server/tests/test_sweep_promote.py`.
@Suite struct BaselineRelativeCoherenceFloorTests {

    /// The degenerate cell, and the baseline it should have been read against.
    static let degenerate = SweepSelectionRule.Cell(
        layer: 20, alpha: 0.4, metric: 0.9, distinct2: 0.535,
        batteryAccuracy: 0.9)
    static let healthy = SweepSelectionRule.Cell(
        layer: 12, alpha: 0.2, metric: 0.6, distinct2: 0.93,
        batteryAccuracy: 0.9)
    static let baseline = SweepSelectionRule.Baseline(
        metric: 0.1, distinct2: 0.989, batteryAccuracy: 0.9)

    @Test func theRelativeFloorRejectsTheCellTheAbsoluteFloorAdmitted() throws {
        let legacy = try SweepSelectionRule.resolve(
            .init(constraints: .init(coherenceFloor: 0.45)))
        #expect(legacy.coherenceRatioToBaseline == nil)
        #expect(
            SweepSelectionRule.select(
                cells: [Self.degenerate, Self.healthy], baseline: Self.baseline,
                criterion: legacy)?.layer == 20)

        let relative = try SweepSelectionRule.resolve(
            .init(
                constraints: .init(
                    coherenceRatioToBaseline: 0.85,
                    coherenceAbsoluteBackstop: 0.6)))
        #expect(relative.isBaselineRelativeCoherence)
        // 0.535 is neither 0.85 × 0.989 (= 0.84) nor above the 0.6 backstop.
        #expect(
            SweepSelectionRule.select(
                cells: [Self.degenerate, Self.healthy], baseline: Self.baseline,
                criterion: relative)?.layer == 12)
        // …and the ranking the topK control walks agrees, because both read
        // the same rule.
        #expect(
            SweepSelectionRule.rankedCandidates(
                cells: [Self.degenerate, Self.healthy], baseline: Self.baseline,
                criterion: relative, k: 5
            ).map(\.layer) == [12])
    }

    @Test func bothHalvesOfTheRelativeFloorBind() throws {
        let criterion = try SweepSelectionRule.resolve(
            .init(
                constraints: .init(
                    coherenceRatioToBaseline: 0.85,
                    coherenceAbsoluteBackstop: 0.6)))
        // A cell that clears the backstop but not the relative bar.
        #expect(
            !SweepSelectionRule.coherencePasses(
                distinct2: 0.7, baselineDistinct2: 0.989, criterion: criterion))
        // A DEGENERATE baseline: 0.5 × 0.85 = 0.425, which the cell clears —
        // but the backstop is what stops a bad baseline licensing a bad
        // winner.
        #expect(
            !SweepSelectionRule.coherencePasses(
                distinct2: 0.5, baselineDistinct2: 0.5, criterion: criterion))
        #expect(
            SweepSelectionRule.coherencePasses(
                distinct2: 0.9, baselineDistinct2: 0.989, criterion: criterion))
    }

    @Test func absentNewFieldsMeanTheLegacyAbsoluteRuleForever() throws {
        // The DEFAULTS, and the deliberate asymmetry: a criterion that
        // declares nothing is a criterion written before the relative form
        // existed, and it keeps gating exactly as it did.
        let legacy = try SweepSelectionRule.resolve(nil)
        #expect(legacy.coherenceFloor == 0.45)
        #expect(legacy.coherenceRatioToBaseline == nil)
        #expect(
            legacy.coherenceSummary == "coherence floor 0.45 (absolute distinct-2)")
        // A stamped criterion round-trips to the SAME rule — provenance and
        // agent birth certificates decode forever.
        #expect(try SweepSelectionRule.resolve(legacy.asCriterion) == legacy)
        #expect(legacy.asCriterion.constraints?.coherenceRatioToBaseline == nil)
        #expect(legacy.asCriterion.constraints?.coherenceAbsoluteBackstop == nil)

        // Either field ALONE selects the relative rule and defaults the other.
        for constraints in [
            ExperimentManifest.SweepSelection.Constraints(
                coherenceRatioToBaseline: 0.85),
            ExperimentManifest.SweepSelection.Constraints(
                coherenceAbsoluteBackstop: 0.6),
        ] {
            let resolved = try SweepSelectionRule.resolve(
                .init(constraints: constraints))
            #expect(resolved.coherenceRatioToBaseline == 0.85)
            #expect(resolved.coherenceFloor == 0.6)
            #expect(
                resolved.coherenceSummary
                    == "coherence floor 0.85× the α=0 baseline's distinct-2, "
                        + "backstop 0.6")
            // The relative criterion also round-trips.
            #expect(try SweepSelectionRule.resolve(resolved.asCriterion) == resolved)
        }
    }

    @Test func theRelativeRangesAndAscendingSanityRefuseAtResolveTime() throws {
        let invalid: [(ExperimentManifest.SweepSelection.Constraints, String)] = [
            (.init(coherenceRatioToBaseline: 1.5), "coherenceRatioToBaseline"),
            (.init(coherenceRatioToBaseline: 0), "coherenceRatioToBaseline"),
            (.init(coherenceRatioToBaseline: -0.1), "coherenceRatioToBaseline"),
            (.init(coherenceRatioToBaseline: .nan), "coherenceRatioToBaseline"),
            (
                .init(coherenceRatioToBaseline: 0.9, coherenceAbsoluteBackstop: 1),
                "coherenceAbsoluteBackstop"
            ),
            (
                .init(
                    coherenceRatioToBaseline: 0.9, coherenceAbsoluteBackstop: -0.1),
                "coherenceAbsoluteBackstop"
            ),
            // Ascending sanity: a backstop at or above the ratio can never be
            // the looser of the two, so the criterion reads as relative and
            // gates absolutely.
            (
                .init(
                    coherenceRatioToBaseline: 0.6, coherenceAbsoluteBackstop: 0.6),
                "backstop must sit UNDER the relative bar"
            ),
            (
                .init(
                    coherenceRatioToBaseline: 0.5, coherenceAbsoluteBackstop: 0.8),
                "backstop must sit UNDER the relative bar"
            ),
        ]
        for (constraints, fragment) in invalid {
            do {
                _ = try SweepSelectionRule.resolve(.init(constraints: constraints))
                Issue.record("\(constraints) must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains(fragment), "\(error.reason)")
            }
        }
        // The boundary that IS legal.
        #expect(
            try SweepSelectionRule.resolve(
                .init(
                    constraints: .init(
                        coherenceRatioToBaseline: 1, coherenceAbsoluteBackstop: 0))
            ).coherenceFloor == 0)
    }

    @Test func theReportedRatioAndLengthFlagAreReportsNotGates() {
        #expect(
            SweepSelectionRule.distinct2Ratio(
                distinct2: 0.535, baselineDistinct2: 0.989).map {
                    (($0 * 1000).rounded() / 1000)
                } == 0.541)
        // A baseline with no coherence at all has no ratio — reporting 0 or ∞
        // would be a fact nobody measured.
        #expect(
            SweepSelectionRule.distinct2Ratio(
                distinct2: 0.5, baselineDistinct2: 0) == nil)
        // 65% longer is flagged; 49% is not; a zero baseline never flags.
        #expect(
            SweepSelectionRule.lengthInflated(
                meanWords: 165, baselineMeanWords: 100))
        #expect(
            !SweepSelectionRule.lengthInflated(
                meanWords: 149, baselineMeanWords: 100))
        #expect(
            !SweepSelectionRule.lengthInflated(meanWords: 200, baselineMeanWords: 0))
        // A flag is NOT a gate: the same cell still passes coherence when the
        // rule says it does.
        let criterion = SweepSelectionRule.Resolved(
            metric: "markerDensity", capabilityTolerance: 0.15,
            coherenceFloor: 0.6, coherenceRatioToBaseline: 0.85,
            matchedNormRandomMargin: nil)
        #expect(
            SweepSelectionRule.coherencePasses(
                distinct2: 0.95, baselineDistinct2: 0.989, criterion: criterion))
    }

    @Test func theSweepCSVCarriesTheRatioAndTheFlagInBothEnginesOrder() throws {
        #expect(
            SweepRunCatalog.csvHeader
                == "concept,layer,alpha,markerDensity,distinct2,distinct2Ratio,"
                    + "words,lengthInflated,batteryAccuracy")
        let rows = try SweepRunCatalog.parseCSV(
            SweepRunCatalog.csvHeader + "\n"
                + "fear,-1,0,0.01,0.989,1.0,100,false,0.9\n"
                + "fear,20,0.4,0.2,0.535,0.541,165,true,0.9\n"
                // A cell whose ratio was undefined writes an EMPTY field.
                + "fear,12,0.2,0.2,0.0,,80,false,0.9")
        #expect(rows[0].distinct2Ratio == 1.0)
        #expect(rows[0].words == 100)
        #expect(!rows[0].lengthInflated)
        #expect(rows[1].distinct2Ratio == 0.541)
        #expect(rows[1].lengthInflated)
        #expect(rows[2].distinct2Ratio == nil)
        #expect(!rows[2].lengthInflated)
    }

    @Test func theGridMarksAConstraintFailureUnderTheRelativeRule() throws {
        let criterion = try SweepSelectionRule.resolve(
            .init(
                constraints: .init(
                    coherenceRatioToBaseline: 0.85,
                    coherenceAbsoluteBackstop: 0.6)))
        let baseline = SweepRunCatalog.Row(
            concept: "fear", layer: -1, alpha: 0, markerDensity: 0.01,
            distinct2: 0.989, batteryAccuracy: 0.9)
        let degenerate = SweepRunCatalog.Row(
            concept: "fear", layer: 20, alpha: 0.4, markerDensity: 0.2,
            distinct2: 0.535, batteryAccuracy: 0.9)
        #expect(
            SweepRunCatalog.cellState(
                row: degenerate, baseline: baseline, criterion: criterion,
                winner: nil) == .failedConstraint)
        // With NO baseline row only the absolute half can judge — which under
        // the relative rule is the backstop.
        #expect(
            SweepRunCatalog.cellState(
                row: degenerate, baseline: nil, criterion: criterion,
                winner: nil) == .failedConstraint)
    }
}
