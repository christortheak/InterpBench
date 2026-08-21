import Foundation
import Testing

@testable import ExperimentKit

/// Per-item choice deltas (`analyze` → `choice-deltas.csv`): the citable,
/// engine-computed version of the per-item Δ the results explorer otherwise
/// derives client-side.
///
/// The column names, the JSON keys, the skip semantics and the sort are the
/// cross-engine contract — server twin `Server/tests/test_choice_deltas.py`.
/// (The bootstrap REPLICATES are not bit-identical across engines, by the
/// standing `StudyStatistics` note; the estimator and its conventions —
/// 10 000 draws, seed 0 — are.)
@Suite struct ChoiceDeltasTests {

    private func readout(
        _ condition: String, _ promptID: String, target: String = "A",
        logOdds: [String: Double] = ["A": 0, "B": 0],
        probability: [String: Double] = [:], selected: String = "A"
    ) -> ChoiceDeltas.Readout {
        ChoiceDeltas.Readout(
            condition: condition, promptID: promptID, target: target,
            logOdds: logOdds, choiceProbability: probability,
            selected: selected)
    }

    @Test func perItemDeltasAndFlipsAgainstTheSameItemBaseline() throws {
        let table = ChoiceDeltas.table([
            // p1: target A up 3 nats, no flip.
            readout("baseline", "p1", logOdds: ["A": 1, "B": -1],
                    probability: ["A": 0.8, "B": 0.2], selected: "A"),
            readout("steer", "p1", logOdds: ["A": 4, "B": -4],
                    probability: ["A": 0.95, "B": 0.05], selected: "A"),
            // p2: target A up 5 nats AND flips B → A.
            readout("baseline", "p2", logOdds: ["A": -2, "B": 2],
                    probability: ["A": 0.25, "B": 0.75], selected: "B"),
            readout("steer", "p2", logOdds: ["A": 3, "B": -3],
                    probability: ["A": 0.9, "B": 0.1], selected: "A"),
        ])

        #expect(table.rows.count == 2)
        #expect(
            table.rows[0] == [
                "steer", "p1", "A", "1", "4", "3", "A", "A", "0",
                "0.8", "0.95", "0.15",
            ])
        #expect(
            table.rows[1] == [
                "steer", "p2", "A", "-2", "3", "5", "B", "A", "1",
                "0.25", "0.9", "0.65",
            ])
        // The table IS the delta: a baseline row would be a value, not a change.
        #expect(!table.rows.contains { $0[0] == "baseline" })

        let block = try #require(table.summary.conditions["steer"])
        #expect(block.n == 2 && block.flipped == 1)
        #expect(abs(try #require(block.deltaTargetLogOddsMean) - 4) < 1e-9)
        // The CI comes from the EXISTING paired bootstrap at its defaults —
        // the same machinery effect-sizes.csv runs on.
        #expect(block.replicates == 10_000 && block.seed == 0)
        let lower = try #require(block.ciLower)
        let upper = try #require(block.ciUpper)
        #expect(lower >= 3 - 1e-9 && upper <= 5 + 1e-9)
        #expect(table.summary.records == 2)
        #expect(table.summary.skippedNoBaseline == 0)
    }

    @Test func csvCarriesTheContractHeader() {
        let csv = ChoiceDeltas.csv([])
        #expect(
            csv == "condition,promptID,target,baselineTargetLogOdds,"
                + "conditionTargetLogOdds,deltaTargetLogOdds,baselineSelected,"
                + "conditionSelected,flipped,baselineTargetProbability,"
                + "conditionTargetProbability,deltaTargetProbability\n")
    }

    @Test func rowsSortByConditionThenPromptRegardlessOfInputOrder() {
        let table = ChoiceDeltas.table([
            readout("baseline", "p2"), readout("zeta", "p2", logOdds: ["A": 1]),
            readout("alpha", "p2", logOdds: ["A": 2]),
            readout("baseline", "p1"), readout("zeta", "p1", logOdds: ["A": 3]),
            readout("alpha", "p1", logOdds: ["A": 4]),
        ])
        #expect(
            table.rows.map { [$0[0], $0[1]] } == [
                ["alpha", "p1"], ["alpha", "p2"], ["zeta", "p1"], ["zeta", "p2"],
            ])
    }

    @Test func aReadoutWithNoBaselinePartnerIsSkippedAndCounted() throws {
        let table = ChoiceDeltas.table([
            readout("baseline", "p1"),
            readout("steer", "p1", logOdds: ["A": 2, "B": 0]),
            // p2 was measured under steer but never under baseline.
            readout("steer", "p2", logOdds: ["A": 5, "B": 0]),
        ])
        #expect(table.rows.map { $0[1] } == ["p1"])
        #expect(table.summary.skippedNoBaseline == 1)
        let block = try #require(table.summary.conditions["steer"])
        #expect(block.skippedNoBaseline == 1 && block.n == 1)
    }

    @Test func anUnreadableTargetIsSkippedAndCountedNotZeroed() throws {
        let table = ChoiceDeltas.table([
            // logOdds present, but with no entry for the item's own target…
            readout("baseline", "p1", target: "C", logOdds: ["A": 0, "B": 0]),
            readout("steer", "p1", target: "C", logOdds: ["A": 2, "B": 0]),
            // …and a readout with no logOdds at all.
            readout("baseline", "p2"),
            readout("steer", "p2", logOdds: [:]),
        ])
        #expect(table.rows.isEmpty)
        #expect(table.summary.skippedNoTargetValue == 2)
        let block = try #require(table.summary.conditions["steer"])
        #expect(block.n == 0)
        // No pairing ⇒ no mean and no interval, rather than a zero.
        #expect(block.deltaTargetLogOddsMean == nil)
        #expect(block.ciLower == nil && block.ciUpper == nil)
        let json = String(
            decoding: try! JSONEncoder().encode(table.summary), as: UTF8.self)
        #expect(!json.contains("deltaTargetLogOddsMean"))
    }

    @Test func probabilitiesAreOptionalButTheLogOddsRowStands() {
        let table = ChoiceDeltas.table([
            readout("baseline", "p1", logOdds: ["A": 0, "B": -1]),
            readout("steer", "p1", logOdds: ["A": 2, "B": -1]),
        ])
        let row = table.rows[0]
        #expect(Array(row[0..<6]) == ["steer", "p1", "A", "0", "2", "2"])
        #expect(Array(row[9...]) == ["", "", ""])
    }

    @Test func aFlipNeedsBothSidesSelected() {
        let table = ChoiceDeltas.table([
            readout("baseline", "p1", logOdds: ["A": 0, "B": -1], selected: ""),
            readout("steer", "p1", logOdds: ["A": 2, "B": -1], selected: "A"),
        ])
        let row = table.rows[0]
        #expect(row[6] == "" && row[7] == "" && row[8] == "0")
    }

    @Test func noNonBaselineReadoutsMeansNoTable() {
        // Nothing at all…
        #expect(ChoiceDeltas.table([]).summary.conditions.isEmpty)
        // …and baseline-only: the file is per-item CHANGE, not per-item value.
        #expect(
            ChoiceDeltas.table([readout("baseline", "p1")])
                .summary.conditions.isEmpty)
    }

    @Test func allSkippedStillReportsTheSkips() throws {
        // A header-only table plus a skip count is honest; silence is not.
        let table = ChoiceDeltas.table([readout("steer", "p1", logOdds: ["A": 2])])
        #expect(table.rows.isEmpty)
        #expect(!table.summary.conditions.isEmpty)
        #expect(table.summary.skippedNoBaseline == 1)
        #expect(table.summary.records == 0)
    }

    @Test func theSameReadoutsTwiceAreByteIdentical() {
        // Real pairs, in deliberately shuffled input order and with
        // irrational-ish values, so the sort AND the bootstrap both have to
        // be reproducible for this to hold.
        var readouts: [ChoiceDeltas.Readout] = []
        for (index, promptID) in ["p3", "p1", "p4", "p2"].enumerated() {
            readouts.append(
                readout(
                    "steer", promptID,
                    logOdds: ["A": Double(index) * 0.37 + 1.5, "B": -1],
                    probability: ["A": 0.7, "B": 0.3]))
            readouts.append(
                readout(
                    "baseline", promptID,
                    logOdds: ["A": Double(index) * 0.37, "B": -1],
                    probability: ["A": 0.5, "B": 0.5]))
        }
        let first = ChoiceDeltas.table(readouts)
        let second = ChoiceDeltas.table(readouts)
        #expect(first.rows.count == 4)
        #expect(first.summary.conditions["steer"]?.n == 4)
        #expect(ChoiceDeltas.csv(first.rows) == ChoiceDeltas.csv(second.rows))
        #expect(first.summary == second.summary)
    }

    // MARK: declaredness (open-issues #6)

    /// A rating-ladder record set must NOT reach this table: its "target" was
    /// the run loop's synthesis (`options[0]`, the scale minimum), so a
    /// `choice-deltas.csv` row for it would publish the paired shift in
    /// log-odds of answering "1" under a `target` column nobody declared —
    /// pole movement entangled with distribution sharpening.
    ///
    /// Server twin: `test_choice_deltas.py`'s declaredness cases.
    @Test func anOrdinalRecordSetIsNotAChoiceMeasurement() {
        // The historical shape: no `targetSource` stamp at all, recognized by
        // the ordinal readout riding the same record.
        #expect(
            !ChoiceDeltas.targetIsDeclared(
                promptID: "p1", targetSource: nil, ordinalPosition: 2.3,
                declaredTargets: nil))
        // The new writer is explicit rather than inferred.
        #expect(
            !ChoiceDeltas.targetIsDeclared(
                promptID: "p1", targetSource: nil, ordinalPosition: nil,
                declaredTargets: ["p1": false]))
    }

    @Test func aDeclaredTargetSetKeepsItsEndpoint() {
        #expect(
            ChoiceDeltas.targetIsDeclared(
                promptID: "p2", targetSource: "declared", ordinalPosition: nil,
                declaredTargets: nil))
        // The MIXED instrument: a declared A/B target and an ordinal readout
        // on ONE record is a legitimate choice item, and only the pinned task
        // file can tell it apart from a synthesized one.
        #expect(
            ChoiceDeltas.targetIsDeclared(
                promptID: "p2", targetSource: "declared", ordinalPosition: 2.6,
                declaredTargets: ["p2": true]))
        // Legacy: no stamp, no ordinal readout — every legitimate choice
        // study declares targets in its item file, so the endpoint stands.
        #expect(
            ChoiceDeltas.targetIsDeclared(
                promptID: "p3", targetSource: nil, ordinalPosition: nil,
                declaredTargets: nil))
    }

    @Test func thePinnedTaskFileOutranksTheRecordStamp() {
        // Authority ladder, rung 1: the file is the exact record of which
        // items declared a target, and it wins over both fallbacks.
        #expect(
            !ChoiceDeltas.targetIsDeclared(
                promptID: "p1", targetSource: "declared", ordinalPosition: nil,
                declaredTargets: ["p1": false]))
        #expect(
            ChoiceDeltas.targetIsDeclared(
                promptID: "p1", targetSource: nil, ordinalPosition: 2.3,
                declaredTargets: ["p1": true]))
        // An item the map never saw falls through to the record's own facts.
        #expect(
            !ChoiceDeltas.targetIsDeclared(
                promptID: "missing", targetSource: nil, ordinalPosition: 2.3,
                declaredTargets: ["other": true]))
    }
}
