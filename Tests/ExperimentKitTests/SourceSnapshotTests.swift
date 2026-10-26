import Foundation
import Testing

@testable import ExperimentKit

/// Which records of its source run an analysis read (`SourceSnapshot`).
///
/// Review finding (October 23, 2026): a checkpoint's analysis (6 over two
/// pairs) was later presented as the analysis of the completed run (3 over
/// four pairs), because a continuation appends to the same run directory and
/// an analysis named its source by directory alone. These tests hold the Mac
/// side of the repair against the hand-authored contract the server suite
/// reads too (`Tests/Fixtures/cross-engine/source-snapshot.json`,
/// `Server/tests/test_source_snapshot.py`), and the Results page that says
/// when an analysis read an earlier or unfinished snapshot.
struct SourceSnapshotTests {
    private struct Fixture: Decodable {
        struct DescribeCase: Decodable {
            let label: String
            let generations: String
            let expected: Expected
        }
        struct Expected: Decodable {
            let records: Int
            let generationsBytes: Int
            let generationsSHA256: String
        }
        struct StatusCase: Decodable {
            let label: String
            let statusFile: [String: JSONValue]?
            let statusFileText: String?
            let report: Bool
            let expected: String
            let continued: Bool
        }
        struct VerdictCase: Decodable {
            let label: String
            let stamp: SourceSnapshot.Stamp?
            let now: Expected?
            let continued: Bool
            let expected: Outcome
        }
        struct Outcome: Decodable {
            let state: String
            let sentence: String?
        }
        let describe: [DescribeCase]
        let status: [StatusCase]
        let verdicts: [VerdictCase]
    }

    private func fixture() throws -> Fixture {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine", "source-snapshot.json")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    private func temporary(_ label: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appending(
            component: "source-snapshot-\(label)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    // MARK: - The shared contract

    @Test func describeMatchesTheCrossEngineContract() throws {
        for item in try fixture().describe {
            let described = SourceSnapshot.describe(Data(item.generations.utf8))
            #expect(described.records == item.expected.records, "\(item.label)")
            #expect(described.generationsBytes == item.expected.generationsBytes, "\(item.label)")
            #expect(
                described.generationsSHA256 == item.expected.generationsSHA256, "\(item.label)")
        }
    }

    @Test func statusMatchesTheCrossEngineContract() throws {
        for item in try fixture().status {
            let directory = try temporary("status")
            defer { try? FileManager.default.removeItem(at: directory) }
            let statusURL = directory.appending(component: RunStatusFile.filename)
            if let text = item.statusFileText {
                try Data(text.utf8).write(to: statusURL)
            } else if let object = item.statusFile {
                try JSONEncoder().encode(object).write(to: statusURL)
            }
            if item.report {
                try Data("{}".utf8).write(to: directory.appending(component: "report.json"))
            }
            #expect(SourceSnapshot.status(of: directory) == item.expected, "\(item.label)")
            #expect(SourceSnapshot.wasContinued(directory) == item.continued, "\(item.label)")
        }
    }

    @Test func verdictsAndSentencesMatchTheCrossEngineContract() throws {
        for item in try fixture().verdicts {
            let now = item.now.map {
                SourceSnapshot.Description(
                    records: $0.records, generationsBytes: $0.generationsBytes,
                    generationsSHA256: $0.generationsSHA256)
            }
            let found = SourceSnapshot.verdict(stamp: item.stamp, now: now, continued: item.continued)
            #expect(found.state.rawValue == item.expected.state, "\(item.label)")
            #expect(SourceSnapshot.sentence(found, run: "R") == item.expected.sentence, "\(item.label)")
        }
    }

    // MARK: - The Results page

    /// A source run of `records` lines (the last `continued` of them written
    /// by a continuation that settled a park), and an analysis of it.
    private func workspace(
        records: Int, continued: Bool, analysisStamp: (URL) throws -> SourceSnapshot.Stamp?
    ) throws -> (root: URL, analysis: URL, run: String) {
        let root = try temporary("results")
        let runs = root.appending(component: "runs")
        let run = "20261004T000000000-exp-tone-run"
        let source = runs.appending(component: run)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let lines = (0..<records).map { index in
            #"{"condition":"\#(index % 2 == 0 ? "baseline" : "formal")","promptID":"item-\#(index / 2 + 1)","wordCount":10}"#
        }
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(
            to: source.appending(component: "generations.jsonl"))
        try Data("{}".utf8).write(to: source.appending(component: "report.json"))
        let status: [String: Any] =
            continued
            ? ["status": "completed", "resumedFrom": ["status": "checkpointed", "itemsWritten": 4]]
            : ["status": "completed"]
        try JSONSerialization.data(withJSONObject: status).write(
            to: source.appending(component: RunStatusFile.filename))
        let analysis = runs.appending(component: "20261004T010000000-exp-tone-analyze")
        try FileManager.default.createDirectory(at: analysis, withIntermediateDirectories: true)
        try Data((run + "\n").utf8).write(to: analysis.appending(component: "source-run.txt"))
        try Data("endpoint,condition,n,deltaMean\nwordCount,formal,2,6.0\n".utf8).write(
            to: analysis.appending(component: "effect-sizes.csv"))
        if let stamp = try analysisStamp(source) {
            try JSONEncoder().encode(stamp).write(
                to: analysis.appending(component: SourceSnapshot.stampFile))
        }
        return (root, analysis, run)
    }

    private func checkpointStamp(_ source: URL) -> SourceSnapshot.Stamp {
        SourceSnapshot.Stamp(
            schemaVersion: 1, sourceRun: source.lastPathComponent, status: "checkpointed",
            records: 4, generationsBytes: 200, generationsSHA256: String(repeating: "a", count: 64))
    }

    @Test func aCheckpointsAnalysisSaysSoOnTheCompletedRunsPage() throws {
        let made = try workspace(records: 8, continued: true) { checkpointStamp($0) }
        defer { try? FileManager.default.removeItem(at: made.root) }
        let model = RunResults.load(runDirectory: made.analysis)
        #expect(
            model.sourceSnapshotNotice
                == "This analysis read run \(made.run) while it was paused at a checkpoint, "
                + "with 4 records, and the run now holds 8. Re-analyze the run for effects "
                + "over all its records.")
    }

    @Test func anAnalysisOfTheRunAsItStandsSaysNothing() throws {
        let made = try workspace(records: 8, continued: true) { source in
            SourceSnapshot.stamp(
                runDirectory: source, status: SourceSnapshot.status(of: source),
                generations: try Data(contentsOf: source.appending(component: "generations.jsonl")))
        }
        defer { try? FileManager.default.removeItem(at: made.root) }
        #expect(RunResults.load(runDirectory: made.analysis).sourceSnapshotNotice == nil)
    }

    @Test func aLegacyAnalysisIsFlaggedOnlyWhenItsRunWasContinued() throws {
        let continued = try workspace(records: 8, continued: true) { _ in nil }
        defer { try? FileManager.default.removeItem(at: continued.root) }
        #expect(
            RunResults.load(runDirectory: continued.analysis).sourceSnapshotNotice?
                .contains("may predate the continuation") == true)
        let plain = try workspace(records: 8, continued: false) { _ in nil }
        defer { try? FileManager.default.removeItem(at: plain.root) }
        #expect(RunResults.load(runDirectory: plain.analysis).sourceSnapshotNotice == nil)
    }

    @Test func aRemotePageSpeaksFromTheStampAlone() throws {
        var artifacts = RunResults.ArtifactBytes()
        artifacts.assign(
            name: "effect-sizes.csv",
            data: Data("endpoint,condition,n,deltaMean\nwordCount,formal,2,6.0\n".utf8))
        artifacts.assign(
            name: SourceSnapshot.stampFile,
            data: try JSONEncoder().encode(
                checkpointStamp(URL(filePath: "/remote/runs/20261004T000000000-exp-tone-run"))))
        #expect(RunResults.ArtifactBytes.fileNames.contains(SourceSnapshot.stampFile))
        let model = RunResults.remoteModel(
            runID: "20261004T010000000-exp-tone-analyze", artifacts: artifacts)
        #expect(
            model.sourceSnapshotNotice
                == "This analysis read run 20261004T000000000-exp-tone-run while it was paused "
                + "at a checkpoint, with 4 records. Re-analyze the run for effects over all its "
                + "records.")
    }
}
