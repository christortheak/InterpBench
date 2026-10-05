import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Committed judge artifacts from each engine, read from beside this file.
///
/// `python-engine/` is the Python engine's own output: one evaluation of a
/// four-pair synthetic run by a two-judge panel with a scripted judge, in
/// which one pair is noncompliant for one judge, one verdict per judge is
/// read from a cut-off answer, the source run is unstamped, and three pairs
/// carry human ratings. `mac-engine/` is the Mac engine's output for the
/// same design, with the source-run path replaced by a neutral one.
enum JudgeArtifactFixture {
    static func data(_ engine: String, _ name: String) throws -> Data {
        let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        return try Data(
            contentsOf: directory.appending(
                path: "Fixtures/judge-reports/\(engine)/\(name)"))
    }

    static let pythonSourceRun = "20260101T000000000-exp-tone-study-run"
    static let macSourceRun = "20260101T000000000Z-exp-tone-study-run"
}

/// Judge rows from both engines, and the rows with no verdict.
@Suite struct StudyJudgmentRowTests {

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(component: "judge-rows-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        return url
    }

    private func review(_ engine: String) throws -> (StudyRecordReview<StudyJudgmentRow>, URL) {
        let directory = try temporaryDirectory()
        let url = directory.appending(component: "judgments.jsonl")
        try JudgeArtifactFixture.data(engine, "judgments.jsonl").write(to: url)
        return (StudyRecordReview<StudyJudgmentRow>(url: url), directory)
    }

    /// The done-when case: a file with noncompliant rows is reviewable with
    /// nothing dropped and the noncompliant rows counted.
    @Test func aPythonEngineFilesNoncompliantRowIsARowAndIsCounted() throws {
        let (review, directory) = try review("python-engine")
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(review.counts.total == 8)
        #expect(review.counts.count(.verdict) == 7)
        #expect(review.counts.count(.noncompliant) == 1)
        #expect(review.counts.count(.unreadable) == 0)
        #expect(
            review.counts.summary.map(\.label)
                == ["Verdicts", "No verdict (noncompliant)", "Unreadable"])

        let page = review.page(0)
        #expect(page.rows.count == 8)
        #expect(page.caption == "Showing all 8 judge rows.")
        let hole = try #require(page.rows.first { $0.kind == .noncompliant })
        #expect(hole.record == 4)
        #expect(hole.judge == "judge-a")
        #expect(hole.condition == "warm")
        #expect(hole.promptID == "p3")
        // No verdict is recorded and none is invented.
        #expect(hole.winner == nil)
        #expect(hole.result == nil)
        #expect(hole.confidence == nil)
        #expect(hole.reason?.contains("invalid verdict twice") == true)
        #expect(hole.statusNote?.contains("not counted") == true)
        #expect(hole.baselineWas == "B")
        #expect(hole.conditionWas == "A")

        let filtered = review.page(0, kind: .noncompliant)
        #expect(filtered.rows.map(\.record) == [4])
        #expect(
            filtered.caption
                == "Showing the 1 noncompliant row. The file holds 8 judge rows in all.")
    }

    @Test func aPythonEngineVerdictRowReadsInTheAppsWords() throws {
        let (review, directory) = try review("python-engine")
        defer { try? FileManager.default.removeItem(at: directory) }
        let rows = review.page(0).rows

        let first = rows[0]
        #expect(first.kind == .verdict)
        #expect(first.judge == "judge-a")
        // `outcome: variant` is the condition's win.
        #expect(first.result == "condition")
        #expect(first.winner == "B")
        #expect(first.baselineWas == "A")
        #expect(first.conditionWas == "B")
        #expect(first.confidence == 0.8)
        #expect(first.briefReason == "The second reply is warmer.")
        #expect(first.aScores == ["warmth": .number(3)])
        #expect(first.bScores == ["warmth": .number(6)])
        // The Python engine leaves the prompt in the run.
        #expect(first.prompt == nil)
        #expect(first.statusNote == nil)

        #expect(rows[1].result == "tie")
        #expect(rows[1].winner == "tie")

        // A verdict read from a cut-off answer: kept, labelled, and its
        // missing confidence not filled in.
        let salvaged = rows[2]
        #expect(salvaged.kind == .verdict)
        #expect(salvaged.result == "baseline")
        #expect(salvaged.answerCutOff)
        #expect(salvaged.confidence == nil)
        #expect(salvaged.statusNote?.contains("cut off") == true)
    }

    @Test func aMacEngineFilesNoncompliantRowIsARowAndIsCounted() throws {
        let (review, directory) = try review("mac-engine")
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(review.counts.total == 8)
        #expect(review.counts.count(.verdict) == 7)
        #expect(review.counts.count(.noncompliant) == 1)
        #expect(review.counts.count(.unreadable) == 0)
        let rows = review.page(0).rows
        #expect(rows.map(\.record) == Array(1...8))

        let hole = rows[3]
        #expect(hole.kind == .noncompliant)
        #expect(hole.judge == "judge-a")
        #expect(hole.promptID == "p3")
        #expect(hole.winner == nil)
        #expect(hole.result == nil)
        #expect(hole.reason?.contains("invalid verdict twice") == true)
        #expect(hole.baselineWas == "B")
        #expect(hole.conditionWas == "A")

        let first = rows[0]
        #expect(first.kind == .verdict)
        #expect(first.result == "condition")
        #expect(first.winner == "B")
        #expect(first.conditionWas == "B")
        #expect(first.judgeModel == "test/model")
        #expect(first.prompt == "Describe room 0.")
        #expect(first.aScores == ["warmth": .number(3)])

        // `reasoningTruncated` inside the verdict is the Mac engine's mark
        // of an answer that was cut off.
        #expect(rows[2].answerCutOff)
        #expect(rows[2].confidence == 0.6)
        #expect(rows[2].result == "baseline")
    }

    /// Rows written by the Mac engine's own record types, so the dialect
    /// under test is the writer's and not this test's idea of it.
    @Test func macEngineRowsWrittenByTheEnginesOwnTypesAllRead() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let verdict = ExperimentTasks.PairedJudgeRecord(
            experiment: "tone-study", experimentHash: "hash",
            sourceRunDirectory: "/workspace/runs/run-a",
            judgeName: "judge-a", judgeKind: "local", judgeModel: "test/model",
            judgePrompt: "Which response is warmer in tone?",
            judgeRubricFile: nil, judgeRubricHash: nil, structuredPrompt: nil,
            condition: "warm", sampleIndex: 2, baselineSeed: 1, variantSeed: 5,
            promptID: "p0", prompt: "Describe room 0.",
            baselineWas: "B", conditionWas: "A",
            judgment: PairedJudgeResponse(
                aScores: ["warmth": 6], bScores: ["warmth": 3],
                structuredFields: ["warmer": .string("A")],
                winner: "A", confidence: 0.8,
                briefReason: "The first reply is warmer."),
            conditionResult: "condition")
        let hole = ExperimentTasks.NoncompliantJudgmentRecord(
            promptID: "p3", sampleIndex: 0, condition: "warm",
            baselineSeed: 4, variantSeed: 8, baselineWas: "A",
            noncomplianceReason: "the judge returned an invalid verdict twice",
            judgeName: "judge-a")
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appending(component: "judgments.jsonl")
        var bytes = try encoder.encode(verdict)
        bytes.append(Data("\n".utf8))
        bytes.append(try encoder.encode(hole))
        bytes.append(Data("\nnot a judge row\n".utf8))
        try bytes.write(to: url)

        let review = StudyRecordReview<StudyJudgmentRow>(url: url)
        #expect(review.counts.total == 3)
        #expect(review.counts.count(.verdict) == 1)
        #expect(review.counts.count(.noncompliant) == 1)
        #expect(review.counts.count(.unreadable) == 1)
        let rows = review.page(0).rows

        #expect(rows[0].kind == .verdict)
        #expect(rows[0].result == "condition")
        #expect(rows[0].winner == "A")
        #expect(rows[0].conditionWas == "A")
        #expect(rows[0].judgeModel == "test/model")
        #expect(rows[0].prompt == "Describe room 0.")
        #expect(rows[0].confidence == 0.8)
        #expect(rows[0].structuredFields == ["warmer": .string("A")])
        #expect(rows[0].title == "warm · p0 · sample 3")

        // The row the old strict decoder dropped: `judgment` is null.
        #expect(rows[1].kind == .noncompliant)
        #expect(rows[1].judge == "judge-a")
        #expect(rows[1].reason == "the judge returned an invalid verdict twice")
        #expect(rows[1].result == nil)
        #expect(rows[1].conditionWas == "B")

        #expect(rows[2].kind == .unreadable)
        #expect(rows[2].rawJSON == "not a judge row")
        #expect(rows[2].reason == "This line is not valid JSON.")
    }

    @Test func aJudgeRowWithNeitherAVerdictNorANoncompliantMarkIsUnreadable() {
        let row = StudyJudgmentRow.read(
            Data(#"{"condition":"warm","promptID":"p0","judge":"judge-a"}"#.utf8),
            record: 1)
        #expect(row.kind == .unreadable)
        #expect(row.reason?.contains("neither a verdict") == true)
        #expect(row.conditionWas == nil)
    }
}
