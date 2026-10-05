import Foundation
import Testing

@testable import ExperimentKit

/// A multi-agent study is analyzed per conversation, on both engines.
///
/// Turns within a transcript are not independent: turn k is conditioned on
/// turns 1..k-1. So a turn's value pairs with the baseline's value at the
/// same turn of the same play-through, the turn differences are averaged
/// within each transcript, and the test runs over one value per transcript —
/// `n` counts transcripts, never turns — and the analysis writes
/// `unit-of-analysis.json` to say so. Before this, the Mac engine's
/// `analyze` averaged each turn over the play-throughs and tested the turns
/// as if they were independent items.
///
/// The records and the Python engine's reading of them come from the shared
/// fixture `Tests/Fixtures/cross-engine/panel-transcript-analysis.json`.
/// Server twin: `Server/tests/test_panel_transcript_analysis.py`. The
/// bootstrap interval is left out of the comparison: the engines resample
/// with different generators.
struct PanelTranscriptAnalysisTests {

    private struct Fixture: Decodable {
        let cases: [Case]
    }

    private struct Case: Decodable {
        let label: String
        let effectRows: [PythonRow]
        let unitOfAnalysis: StudyAnalysisStatistics.TranscriptUnit
    }

    private struct PythonRow: Decodable {
        let condition: String
        let endpoint: String
        let n: Int
        let deltaMean: Double
        let wilcoxonW: Double?
        let wilcoxonP: Double?
        let adjustedP: Double?
        let correction: String
        let stratifyBy: String
        let stratum: String
        let unit: String
        let estimand: String
        let inference: String
    }

    private static let fixtureURL = CodeResources.compiledCheckoutPath.appending(
        components: "Tests", "Fixtures", "cross-engine", "panel-transcript-analysis.json")

    /// One case's records as the lines of a `generations.jsonl`, exactly as
    /// committed.
    private func recordLines(_ label: String) throws -> [String] {
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: Self.fixtureURL))
        let cases = try #require((object as? [String: Any])?["cases"] as? [[String: Any]])
        let chosen = try #require(cases.first { $0["label"] as? String == label })
        let records = try #require(chosen["records"] as? [[String: Any]])
        return try records.map {
            String(
                decoding: try JSONSerialization.data(withJSONObject: $0, options: [.sortedKeys]),
                as: UTF8.self)
        }
    }

    private struct Analysis {
        let entries: [ExperimentTasks.EffectSizeEntry]
        let unit: StudyAnalysisStatistics.TranscriptUnit?
        let csv: String
    }

    /// The lines as a completed run of a multi-agent study in a fresh
    /// temporary workspace, analyzed through `experiment analyze`'s workflow.
    private func analyze(_ lines: [String], studyKind: ExperimentManifest.StudyKind = .multiAgent)
        throws -> Analysis
    {
        var manifest = ExperimentManifest(
            name: "panel", description: "", modelID: "test/model",
            createdAt: Date(timeIntervalSince1970: 0))
        manifest.studyKind = studyKind
        let root = FileManager.default.temporaryDirectory.appending(
            component: "panel-transcripts-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let run = root.appending(path: "runs/20261005T000000000Z-exp-panel-run")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(to: run.appending(component: "experiment.json"))
        try Data(ExperimentStore.manifestHash(manifest).utf8).write(
            to: run.appending(component: "experiment-hash.txt"))
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(
            to: run.appending(component: "generations.jsonl"))
        try Data("{}".utf8).write(to: run.appending(component: "report.json"))
        let out = try StudyAnalysisWorkflow.analyze(
            manifest: manifest,
            repository: StudyAnalysisRepository(workspaceRoot: root, promptRoot: root),
            allowUnverifiedEpoch: false)
        let report = try JSONDecoder().decode(
            ExperimentTasks.AnalyzeReport.self,
            from: Data(contentsOf: out.appending(component: "analysis.json")))
        let unitURL = out.appending(component: "unit-of-analysis.json")
        let unit =
            FileManager.default.fileExists(atPath: unitURL.path)
            ? try JSONDecoder().decode(
                StudyAnalysisStatistics.TranscriptUnit.self, from: Data(contentsOf: unitURL))
            : nil
        return Analysis(
            entries: report.effectSizes, unit: unit,
            csv: try String(
                contentsOf: out.appending(component: "effect-sizes.csv"), encoding: .utf8))
    }

    /// The textbook normal-approximation p of a signed-rank W (W = the
    /// smaller rank sum, `n` the nonzero differences, continuity-corrected).
    private func twoSidedP(w: Double, n: Double, tieCorrection: Double = 0) -> Double {
        let mean = n * (n + 1) / 4
        let variance = n * (n + 1) * (2 * n + 1) / 24 - tieCorrection / 48
        let z = (w - mean + 0.5) / variance.squareRoot()
        return min(1, 1 + erf(z / 2.0.squareRoot()))
    }

    /// Worked by hand from the fixture's turns. Word counts, configured
    /// minus baseline, turn by turn: transcript 0 gives 2, 4, 6 (mean 4);
    /// transcript 1 gives -1, -2, 3 (mean 0); transcript 2 gives 5, 3, 7
    /// (mean 5); transcript 3 gives -1, -2, -3 (mean -2). Four transcript
    /// values, mean 7/4. Signed ranks of the nonzero three (|−2| < 4 < 5):
    /// W+ = 2 + 3, W− = 1, so W = 1. Treating the turns as items instead
    /// would have given n = 3.
    @Test func turnsAreAveragedWithinEachTranscriptBeforeTesting() throws {
        let analysis = try analyze(recordLines("four-transcripts"))
        let words = try #require(analysis.entries.first { $0.metric == "wordCount" })
        #expect(words.condition == "configured")
        #expect(words.n == 4)
        #expect(abs(words.meanDiff - 1.75) <= 1e-12)
        #expect(words.wilcoxonW == 1)
        #expect(abs((words.wilcoxonP ?? -1) - twoSidedP(w: 1, n: 3)) <= 1e-12)
        // One condition, so the family is one row and the adjusted p is the
        // raw one.
        #expect(words.adjustedP == words.wilcoxonP)
        #expect(words.correction == "bh")
        // distinct2: transcript values 0, 0.25, 0, -0.25 — mean 0, and the
        // two nonzero values tie (ranks 1.5 each), so W = 1.5 and p = 1.
        let distinct = try #require(analysis.entries.first { $0.metric == "distinct2" })
        #expect(distinct.n == 4)
        #expect(distinct.meanDiff == 0)
        #expect(distinct.wilcoxonW == 1.5)
        #expect(distinct.wilcoxonP == 1)
        // No strata: the per-turn cells are the dependent observations.
        #expect(analysis.entries.allSatisfy { $0.stratifyBy == nil && $0.unit == nil })
        #expect(analysis.entries.count == 2)
        #expect(
            analysis.unit
                == StudyAnalysisStatistics.TranscriptUnit(skippedForSingleTranscript: false))
    }

    @Test func oneTranscriptPerArmSupportsNoIntervalAndSaysSo() throws {
        let analysis = try analyze(recordLines("one-transcript"))
        #expect(analysis.entries.isEmpty)
        #expect(analysis.unit?.skippedForSingleTranscript == true)
        #expect(analysis.unit?.unitOfAnalysis == "transcript")
    }

    /// Every case of the shared fixture: this engine's rows and unit are the
    /// Python engine's, in its order.
    @Test func bothEnginesAnalyzeTheSameTranscripts() throws {
        let fixture = try JSONDecoder().decode(
            Fixture.self, from: Data(contentsOf: Self.fixtureURL))
        #expect(fixture.cases.map(\.label) == ["four-transcripts", "one-transcript"])
        for item in fixture.cases {
            let analysis = try analyze(recordLines(item.label))
            #expect(analysis.unit == item.unitOfAnalysis, "\(item.label)")
            #expect(analysis.entries.count == item.effectRows.count, "\(item.label)")
            for (mine, theirs) in zip(analysis.entries, item.effectRows) {
                let name = "\(item.label) \(theirs.condition) \(theirs.endpoint)"
                #expect(mine.condition == theirs.condition, "\(name)")
                #expect(mine.metric == theirs.endpoint, "\(name)")
                #expect(mine.n == theirs.n, "\(name)")
                // The Python file carries six significant digits.
                #expect(abs(mine.meanDiff - theirs.deltaMean) <= 1e-6, "\(name)")
                #expect(mine.wilcoxonW == theirs.wilcoxonW, "\(name)")
                #expect(abs((mine.wilcoxonP ?? .nan) - (theirs.wilcoxonP ?? .nan)) <= 1e-6, "\(name)")
                #expect(abs((mine.adjustedP ?? .nan) - (theirs.adjustedP ?? .nan)) <= 1e-6, "\(name)")
                #expect(mine.correction == theirs.correction, "\(name)")
                #expect((mine.stratifyBy ?? "pooled") == theirs.stratifyBy, "\(name)")
                #expect((mine.stratum ?? "") == theirs.stratum, "\(name)")
                #expect((mine.unit ?? "") == theirs.unit, "\(name)")
                #expect((mine.estimand ?? "") == theirs.estimand, "\(name)")
                #expect((mine.inference ?? "") == theirs.inference, "\(name)")
            }
        }
    }

    /// The same records under any other study kind keep the item analysis
    /// they always had: the turn is the item, averaged over its samples, and
    /// no unit-of-analysis file is written.
    @Test func otherStudyKindsAreUnchanged() throws {
        let analysis = try analyze(recordLines("four-transcripts"), studyKind: .modelOutput)
        #expect(analysis.unit == nil)
        let words = try #require(
            analysis.entries.first { $0.metric == "wordCount" && $0.stratifyBy == nil })
        // Three turns, each averaged over its four play-throughs.
        #expect(words.n == 3)
    }
}
