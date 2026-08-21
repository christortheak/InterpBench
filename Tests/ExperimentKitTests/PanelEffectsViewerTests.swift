import Foundation
import Testing
@testable import ExperimentKit

/// A14: the panel-effects.csv semantic parser (server dialect —
/// `panel_effects.py` is the only writer today; Swift multi-agent runs do
/// not produce the file) and its absent-file tolerance in the Results model.
@Suite struct PanelEffectsViewerTests {

    /// Byte-shaped like the server writer: DictWriter with the fixed header,
    /// NaN estimands as EMPTY cells (`_fmt`), `%.6g` numbers.
    private let serverCSV = """
        endpoint,direct,directN,spillover,spilloverN,group,groupN,transmissionRatio,amplification,droppedTurns
        months,4.5,3,1.5,6,9,1,0.333333,2,2
        wordCount,,0,,0,,0,,,5
        """

    @Test func parsesServerDialect() throws {
        let rows = try #require(RunResults.panelEffects(fromCSV: serverCSV))
        #expect(rows.count == 2)

        let months = try #require(rows.first { $0.endpoint == "months" })
        #expect(months.direct == 4.5)
        #expect(months.directN == 3)
        #expect(months.spillover == 1.5)
        #expect(months.spilloverN == 6)
        #expect(months.group == 9)
        #expect(months.groupN == 1)
        #expect(months.transmissionRatio.map { abs($0 - 1.5 / 4.5) < 1e-4 } == true)
        #expect(months.amplification == 2)
        #expect(months.droppedTurns == 2)

        // NaN estimands are written as EMPTY cells → decoded as nil, never
        // coerced to a number.
        let wordCount = try #require(rows.first { $0.endpoint == "wordCount" })
        #expect(wordCount.direct == nil)
        #expect(wordCount.spillover == nil)
        #expect(wordCount.group == nil)
        #expect(wordCount.transmissionRatio == nil)
        #expect(wordCount.amplification == nil)
        #expect(wordCount.droppedTurns == 5)
    }

    @Test func toleratesReorderedColumnsAndCase() throws {
        // Header-indexed and case-insensitive: a future dialect that reorders
        // or re-cases columns still reads.
        let shuffled = """
            droppedTurns,AMPLIFICATION,endpoint,direct,directN,group,groupN,spillover,spilloverN,transmissionRatio
            1,2.5,months,4,2,10,1,2,4,0.5
            """
        let rows = try #require(RunResults.panelEffects(fromCSV: shuffled))
        let months = try #require(rows.first)
        #expect(months.endpoint == "months")
        #expect(months.direct == 4)
        #expect(months.amplification == 2.5)
        #expect(months.droppedTurns == 1)
    }

    @Test func refusesNonPanelEffectsCSV() {
        // No endpoint column → not a panel-effects file at all.
        #expect(RunResults.panelEffects(fromCSV: "condition,metric\nbase,1") == nil)
        #expect(RunResults.panelEffects(fromCSV: "") == nil)
    }

    @Test func resultsModelLoadsFileWhenPresentAndToleratesAbsence() throws {
        let directory = FileManager.default.temporaryDirectory.appending(
            component: "panel-effects-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        // Absent file: nil, never an error (a run without the artifact).
        #expect(RunResults.load(runDirectory: directory).panelEffects == nil)

        try serverCSV.write(
            to: directory.appending(component: "panel-effects.csv"),
            atomically: true, encoding: .utf8)
        let model = RunResults.load(runDirectory: directory)
        #expect(model.panelEffects?.count == 2)
        #expect(model.panelEffects?.first?.endpoint == "months")
    }
}
