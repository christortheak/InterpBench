import Foundation
import Testing
@testable import ExperimentKit

/// Sweep-run artifact parsing and discovery for the Optimizations surface —
/// fixture strings and temp directories only, no model, no live runs.
struct SweepRunCatalogTests {

    private let fixtureCSV = """
        concept,layer,alpha,markerDensity,distinct2,batteryAccuracy
        french,-1,0,0.01,0.82,0.9
        french,12,0.2,0.05,0.8,0.88
        french,12,0.4,0.31,0.7,0.85
        french,18,0.4,0.22,0.3,0.86
        french,18,0.8,0.4,0.75,0.6
        """

    // MARK: CSV parsing

    @Test func parsesFixtureCSVIncludingBaselineRow() throws {
        let rows = try SweepRunCatalog.parseCSV(fixtureCSV)
        #expect(rows.count == 5)
        #expect(rows[0].isBaseline)
        #expect(rows[0].concept == "french")
        #expect(rows[2].layer == 12)
        #expect(rows[2].alpha == 0.4)
        #expect(rows[2].markerDensity == 0.31)
        #expect(!rows[2].isBaseline)
        #expect(SweepRunCatalog.concepts(in: rows) == ["french"])
    }

    @Test func rejectsWrongHeaderAndMalformedLines() {
        #expect(throws: ExperimentError.self) {
            _ = try SweepRunCatalog.parseCSV("wrong,header\nfrench,1,2,3,4,5")
        }
        #expect(throws: ExperimentError.self) {
            _ = try SweepRunCatalog.parseCSV(
                SweepRunCatalog.csvHeader + "\nfrench,notanint,0,0,0,0")
        }
        #expect(throws: ExperimentError.self) {
            _ = try SweepRunCatalog.parseCSV("")
        }
    }

    @Test func parsesServerShapedCSVWithWordsColumnIdentically() throws {
        // The server's sweep.csv adds a `words` column between distinct2 and
        // batteryAccuracy — rows must parse identically to the Swift shape.
        let serverCSV = """
            concept,layer,alpha,markerDensity,distinct2,words,batteryAccuracy
            french,-1,0,0.01,0.82,52.1,0.9
            french,12,0.4,0.31,0.7,49.8,0.85
            """
        let swiftCSV = """
            concept,layer,alpha,markerDensity,distinct2,batteryAccuracy
            french,-1,0,0.01,0.82,0.9
            french,12,0.4,0.31,0.7,0.85
            """
        let serverRows = try SweepRunCatalog.parseCSV(serverCSV)
        let swiftRows = try SweepRunCatalog.parseCSV(swiftCSV)
        #expect(serverRows == swiftRows)
        #expect(serverRows.count == 2)
        #expect(serverRows[0].isBaseline)

        // Column order is header-driven, not positional.
        let shuffled = try SweepRunCatalog.parseCSV(
            """
            batteryAccuracy,distinct2,markerDensity,alpha,layer,concept
            0.9,0.82,0.01,0,-1,french
            """)
        #expect(shuffled.first == swiftRows.first)
    }

    @Test func missingRequiredColumnsAreNamedInTheError() {
        do {
            _ = try SweepRunCatalog.parseCSV("concept,markerDensity\nfrench,0.1")
            Issue.record("expected a missing-column error")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("layer"))
            #expect(error.reason.contains("alpha"))
            #expect(error.reason.contains("distinct2"))
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    @Test func legacyCSVWithoutBatteryColumnParsesWithPerfectAccuracy() throws {
        let rows = try SweepRunCatalog.parseCSV(
            "concept,layer,alpha,markerDensity,distinct2\nfrench,12,0.4,0.31,0.7")
        #expect(rows.count == 1)
        #expect(rows[0].batteryAccuracy == 1.0)
        #expect(rows[0].markerDensity == 0.31)
        #expect(rows[0].objective == nil)
    }

    @Test func parsesObjectiveColumnWhenPresentAndNilWhenAbsent() throws {
        // judgeScore/logprobShift sweeps append an `objective` column on both
        // engines; the grid's big number reads it, so a judge sweep whose
        // markers.json is absent (density 0) must still carry its score.
        let judgeCSV = """
            concept,layer,alpha,markerDensity,distinct2,batteryAccuracy,objective
            fear,-1,0,0,0.82,0.9,0.5
            fear,12,0.4,0,0.7,0.85,0.83
            """
        let rows = try SweepRunCatalog.parseCSV(judgeCSV)
        #expect(rows[0].objective == 0.5)  // baseline row pins the 0.5 tie
        #expect(rows[1].objective == 0.83)
        #expect(rows[1].markerDensity == 0)  // density stays a diagnostic

        // markerDensity sweeps and pre-objective runs have no column: nil.
        let plain = try SweepRunCatalog.parseCSV(fixtureCSV)
        #expect(plain.allSatisfy { $0.objective == nil })

        // The server's shape (words column in between) reads the same value,
        // including a negative logprobShift.
        let serverCSV = """
            concept,layer,alpha,markerDensity,distinct2,words,batteryAccuracy,objective
            fear,12,0.4,0,0.7,49.8,0.85,-0.42
            """
        #expect(try SweepRunCatalog.parseCSV(serverCSV).first?.objective == -0.42)
    }

    @Test func unparseableObjectiveValueIsTolerantlyNil() throws {
        // The objective column is display data, not a load gate: a malformed
        // value must not fail the whole run's grid.
        let csv = """
            concept,layer,alpha,markerDensity,distinct2,batteryAccuracy,objective
            fear,12,0.4,0.1,0.7,0.85,notanumber
            """
        let rows = try SweepRunCatalog.parseCSV(csv)
        #expect(rows.count == 1)
        #expect(rows[0].objective == nil)
        #expect(rows[0].markerDensity == 0.1)
    }

    // MARK: recommendations parsing

    @Test func parsesMixedProvenanceAndFailureRecommendations() throws {
        let json = """
            {
              "french": {
                "sweepRun": "run-dir",
                "criterion": {
                  "objective": {"metric": "markerDensity"},
                  "constraints": {"capabilityTolerance": 0.15, "coherenceFloor": 0.45}
                },
                "devPromptsHash": "deadbeef",
                "winningCell": {"layer": 12, "alpha": 0.4},
                "metrics": {"markerDensity": 0.31}
              },
              "authority": "no cell passed the capability/coherence gates"
            }
            """
        let parsed = try SweepRunCatalog.parseRecommendations(Data(json.utf8))
        guard case .selected(let provenance)? = parsed["french"] else {
            Issue.record("expected provenance for french")
            return
        }
        #expect(provenance.winningCell.layer == 12)
        #expect(provenance.criterion.objective?.metric == "markerDensity")
        #expect(provenance.devPromptsHash == "deadbeef")
        guard case .failure(let message)? = parsed["authority"] else {
            Issue.record("expected failure message for authority")
            return
        }
        #expect(message.contains("no cell passed"))
    }

    // MARK: display criterion

    @Test func displayCriterionFillsDefaultsAndNeverThrows() {
        let defaults = SweepRunCatalog.displayCriterion(nil)
        #expect(defaults.metric == "markerDensity")
        #expect(defaults.capabilityTolerance == SweepSelectionRule.defaultCapabilityTolerance)
        #expect(defaults.coherenceFloor == SweepSelectionRule.defaultCoherenceFloor)
        #expect(defaults.matchedNormRandomMargin == nil)

        // Declared-but-unimplemented metric renders instead of throwing
        // (resolve() would refuse — display must not).
        let declared = SweepRunCatalog.displayCriterion(
            .init(
                objective: .init(metric: "judgeScore"),
                constraints: .init(capabilityTolerance: 0.1, coherenceFloor: 0.6),
                controls: .init(matchedNormRandomMargin: 0.02)))
        #expect(declared.metric == "judgeScore")
        #expect(declared.capabilityTolerance == 0.1)
        #expect(declared.coherenceFloor == 0.6)
        #expect(declared.matchedNormRandomMargin == 0.02)
    }

    // MARK: cell state

    @Test func cellStatesMarkWinnerConstraintFailuresAndBaseline() throws {
        let rows = try SweepRunCatalog.parseCSV(fixtureCSV)
        let baseline = rows.first { $0.isBaseline }
        let criterion = try SweepSelectionRule.resolve(nil)  // tol 0.15, floor 0.45
        let winner = ExperimentManifest.SelectionProvenance.Cell(layer: 12, alpha: 0.4)

        func state(_ row: SweepRunCatalog.Row) -> SweepRunCatalog.CellState {
            SweepRunCatalog.cellState(
                row: row, baseline: baseline, criterion: criterion, winner: winner)
        }

        #expect(state(rows[0]) == .baseline)
        #expect(state(rows[1]) == .pass)  // L12 α0.2
        #expect(state(rows[2]) == .winner)  // L12 α0.4
        #expect(state(rows[3]) == .failedConstraint)  // distinct2 0.3 < 0.45
        #expect(state(rows[4]) == .failedConstraint)  // battery 0.6 < 0.9 − 0.15
    }

    // MARK: directory-name matching + discovery

    @Test func directoryNameMatchingIsStrict() {
        #expect(
            SweepRunCatalog.directoryNameMatches(
                "20260707T101112123Z-exp-screen1-sweep", experiment: "screen1"))
        #expect(
            SweepRunCatalog.directoryNameMatches(
                "20260707T101112123Z-exp-screen1-sweep-2", experiment: "screen1"))
        // Another task of the same experiment: no.
        #expect(
            !SweepRunCatalog.directoryNameMatches(
                "20260707T101112123Z-exp-screen1-run", experiment: "screen1"))
        // A different experiment whose name embeds "-sweep": no capture.
        #expect(
            !SweepRunCatalog.directoryNameMatches(
                "20260707T101112123Z-exp-screen1-sweep-run", experiment: "screen1"))
        #expect(
            !SweepRunCatalog.directoryNameMatches(
                "20260707T101112123Z-exp-other-sweep", experiment: "screen1"))
    }

    @Test func discoveryFindsNewestRunWithSweepCSV() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appending(
            component: "sweep-catalog-test-\(UUID().uuidString)")
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: root) }

        func makeRun(_ name: String, csv: Bool) throws -> URL {
            let url = root.appending(component: name)
            try fm.createDirectory(at: url, withIntermediateDirectories: true)
            if csv {
                try fixtureCSV.write(
                    to: url.appending(component: "sweep.csv"),
                    atomically: true, encoding: .utf8)
            }
            return url
        }

        _ = try makeRun("20260101T000000000Z-exp-s1-sweep", csv: true)
        let newest = try makeRun("20260202T000000000Z-exp-s1-sweep", csv: true)
        _ = try makeRun("20260303T000000000Z-exp-s1-sweep", csv: false)  // no csv
        _ = try makeRun("20260404T000000000Z-exp-s1-run", csv: true)  // wrong task
        _ = try makeRun("20260505T000000000Z-exp-other-sweep", csv: true)

        let found = SweepRunCatalog.sweepRunDirectories(
            experiment: "s1", runsDirectory: root)
        #expect(found.count == 2)
        #expect(
            SweepRunCatalog.newestSweepRunDirectory(
                experiment: "s1", runsDirectory: root)?.lastPathComponent
                == newest.lastPathComponent)

        let run = SweepRunCatalog.newestSweepRun(experiment: "s1", runsDirectory: root)
        #expect(run?.rows.count == 5)
        #expect(run?.recommendations.isEmpty == true)  // file absent → empty
    }

    @Test func loadReadsRecommendationsWhenPresent() throws {
        let fm = FileManager.default
        let directory = fm.temporaryDirectory.appending(
            component: "sweep-load-test-\(UUID().uuidString)")
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: directory) }

        try fixtureCSV.write(
            to: directory.appending(component: "sweep.csv"),
            atomically: true, encoding: .utf8)
        try #"{"french": "no cell passed the capability/coherence gates"}"#.write(
            to: directory.appending(component: "recommendations.json"),
            atomically: true, encoding: .utf8)

        let run = try SweepRunCatalog.load(directory: directory)
        #expect(run.rows.count == 5)
        #expect(run.recommendations.count == 1)
        guard case .failure? = run.recommendations["french"] else {
            Issue.record("expected failure recommendation")
            return
        }
    }
}
