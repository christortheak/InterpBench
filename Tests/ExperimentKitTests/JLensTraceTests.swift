import Foundation
import Testing
@testable import ExperimentKit

/// Trace decoding and the heatmap grid, against a REAL `jlens-readout.jsonl`
/// captured from a server run on gemma-3-4b-it.
///
/// The grid arithmetic is tested here rather than in a View because the
/// interesting part of a heatmap is its normalization: a bug there produces a
/// picture that looks entirely plausible and is wrong, and no screenshot would
/// catch it.
@Suite struct JLensTraceTests {

    private func traceFixture() throws -> JLensTrace {
        let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let data = try Data(contentsOf: dir.appending(path: "Fixtures/jlens-readout.jsonl"))
        let (trace, malformed) = try JLensTrace.parse(data)
        #expect(malformed == 0)
        return trace
    }

    @Test func aRealTraceDecodes() throws {
        let trace = try traceFixture()
        #expect(trace.rows.count == 2)
        #expect(trace.isComplete)
        #expect(trace.incompleteCount == 0)
        #expect(trace.totalObservations == 64)

        let row = try #require(trace.rows.first)
        #expect(row.modelID == "google/gemma-3-4b-it")
        #expect(row.lensID == "google--gemma-3-4b-it--jlens-wikitext")
        #expect(row.evidenceTier == "testing")
        #expect(row.isEvidenceTier == false)
        #expect(row.watchlistTokenIDs?.count == 2)
        #expect(row.generatedTokenCount == 16)
    }

    @Test func everyObservationNamesTheTokenItPredicted() throws {
        // The alignment guarantee, checked on real data: an unaligned row is
        // uninterpretable, not merely incomplete.
        let trace = try traceFixture()
        for row in trace.rows {
            let observations = row.observations ?? []
            #expect(!observations.isEmpty)
            #expect(observations.allSatisfy { !$0.isUnaligned })
        }
    }

    @Test func prefillPredictsTokenZeroAndDecodeStepsFollow() throws {
        let row = try #require(try traceFixture().rows.first)
        let observations = row.observations ?? []
        let prefill = observations.filter { $0.passKind == "prefill" }
        #expect(prefill.allSatisfy { $0.predictedIndex == 0 })
        // One armed layer per step, contiguous from 0.
        let steps = Set(observations.map(\.predictedIndex)).sorted()
        #expect(steps == Array(0..<steps.count))
    }

    @Test func theLogitLensCompanionIsPresentAndDiffers() throws {
        // Without it there is no way to say transport did any work.
        let row = try #require(try traceFixture().rows.first)
        let withBoth = (row.observations ?? []).filter {
            ($0.watched?.isEmpty == false) && ($0.watchedLogitLens?.isEmpty == false)
        }
        #expect(!withBoth.isEmpty)
        #expect(withBoth.contains { $0.watched != $0.watchedLogitLens })
    }

    // MARK: Heatmap

    @Test func theHeatmapCoversEveryArmedLayerAndStep() throws {
        let row = try #require(try traceFixture().rows.first)
        let heatmap = JLensHeatmap.build(row: row)
        #expect(heatmap.layers == [16, 24])
        #expect(heatmap.steps.count == 16)
        #expect(heatmap.cells.count == 32)
        #expect(heatmap.watchedTokenID == row.watchlistTokenIDs?.first)
    }

    @Test func intensityIsNormalizedWithinTheGridsOwnRange() throws {
        let row = try #require(try traceFixture().rows.first)
        let heatmap = JLensHeatmap.build(row: row)
        let intensities = heatmap.cells.compactMap(\.intensity)
        #expect(intensities.count == heatmap.cells.count)
        #expect(intensities.allSatisfy { $0 >= 0 && $0 <= 1 })
        // The extremes must actually reach the ends, or the picture compresses
        // the very contrast it exists to show.
        #expect(intensities.min() == 0)
        #expect(intensities.max() == 1)
        let low = try #require(heatmap.minimum)
        let high = try #require(heatmap.maximum)
        #expect(low < high)
    }

    @Test func theBrightestCellIsTheHighestScore() throws {
        let row = try #require(try traceFixture().rows.first)
        let heatmap = JLensHeatmap.build(row: row)
        let brightest = try #require(
            heatmap.cells.max { ($0.intensity ?? -1) < ($1.intensity ?? -1) })
        let highest = try #require(
            heatmap.cells.max { ($0.value ?? -.infinity) < ($1.value ?? -.infinity) })
        #expect(brightest.layer == highest.layer)
        #expect(brightest.step == highest.step)
    }

    @Test func switchingTheWatchedTokenChangesTheGrid() throws {
        // Colour encodes ONE token; picking the other must actually repaint,
        // otherwise the picker is decorative.
        let row = try #require(try traceFixture().rows.first)
        let first = JLensHeatmap.build(row: row, watchlistIndex: 0)
        let second = JLensHeatmap.build(row: row, watchlistIndex: 1)
        #expect(first.watchedTokenID != second.watchedTokenID)
        #expect(first.cells.map(\.value) != second.cells.map(\.value))
    }

    @Test func anOutOfRangeWatchIndexYieldsNoValuesRatherThanCrashing() throws {
        let row = try #require(try traceFixture().rows.first)
        let heatmap = JLensHeatmap.build(row: row, watchlistIndex: 99)
        #expect(heatmap.cells.allSatisfy { $0.value == nil && $0.intensity == nil })
        #expect(heatmap.minimum == nil)
    }

    @Test func aFlatGridIsUniformRatherThanDividedByZero() throws {
        // Every value equal is legitimate data, not an error — and 0/0 would
        // render as either blank or fully saturated, both of them lies.
        let observations = (0..<3).map {
            JLensObservation(layer: 5, passKind: "decode", position: 10 + $0,
                             predictedIndex: $0, predictedTokenID: 1,
                             watched: [7.0], watchedLogitLens: nil,
                             topKIDs: nil, topKLogits: nil,
                             topKIDsLogitLens: nil, topKLogitsLogitLens: nil)
        }
        var row = JLensTraceRow()
        row.observations = observations
        row.watchlistTokenIDs = [1]
        let heatmap = JLensHeatmap.build(row: row)
        #expect(heatmap.cells.allSatisfy { $0.intensity == 0.5 })
    }

    @Test func mentionPrimingIsCarriedOntoTheCells() throws {
        var row = JLensTraceRow()
        row.observations = [
            JLensObservation(layer: 5, passKind: "decode", position: 10,
                             predictedIndex: 0, predictedTokenID: 1,
                             watched: [1.0], watchedLogitLens: nil, topKIDs: nil,
                             topKLogits: nil, topKIDsLogitLens: nil,
                             topKLogitsLogitLens: nil),
            JLensObservation(layer: 5, passKind: "decode", position: 11,
                             predictedIndex: 1, predictedTokenID: 2,
                             watched: [2.0], watchedLogitLens: nil, topKIDs: nil,
                             topKLogits: nil, topKIDsLogitLens: nil,
                             topKLogitsLogitLens: nil)]
        row.watchlistTokenIDs = [1]
        row.mentionMask = ["0": ["1": false], "1": ["1": true]]
        let heatmap = JLensHeatmap.build(row: row)
        #expect(heatmap.cell(layer: 5, step: 0)?.mentionPrimed == false)
        #expect(heatmap.cell(layer: 5, step: 1)?.mentionPrimed == true)
        #expect(row.isMentionPrimed(atStep: 1))
    }

    // MARK: Parsing robustness

    @Test func malformedLinesAreCountedNotSilentlyDropped() throws {
        // A viewer that quietly skips unparseable rows would undo the
        // completeness machinery the trace format is built around.
        let text = """
        {"run":"r","traceComplete":true,"observationCount":0}
        { not json
        {"run":"r","traceComplete":true,"observationCount":0}
        """
        let (trace, malformed) = try JLensTrace.parse(Data(text.utf8))
        #expect(trace.rows.count == 2)
        #expect(malformed == 1)
    }

    @Test func anIncompleteRowMakesTheWholeTraceIncomplete() throws {
        let text = """
        {"run":"r","traceComplete":true,"observationCount":1}
        {"run":"r","traceComplete":false,"traceFailureReason":"truncated","observationCount":1}
        """
        let (trace, _) = try JLensTrace.parse(Data(text.utf8))
        #expect(!trace.isComplete)
        #expect(trace.incompleteCount == 1)
    }

    @Test func anEmptyTraceIsNotComplete() throws {
        let (trace, _) = try JLensTrace.parse(Data())
        #expect(trace.rows.isEmpty)
        #expect(!trace.isComplete)     // vacuous truth would read as a pass
    }

    @Test func theArmingSummaryNamesTheInjection() throws {
        let cell = JLensSteeringCell(layer: 16, alpha: 1.5, mode: "add",
                                    concept: "courage")
        #expect(cell.summary == "add courage @ L16 α1.50")
    }
}
