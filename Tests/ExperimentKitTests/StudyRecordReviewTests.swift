import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The study Results view's record review shows every line of a run's
/// evidence files (release review finding E6).
///
/// What it replaced read the first 80 lines of generations.jsonl and the
/// first 200 of judgments.jsonl, dropped any line its strict decoder
/// disliked — a noncompliant judge row, an answer-option reading, a failure
/// record, every judge row the Python engine writes — and called the result
/// "every". These tests pin the three properties that undo that: one row per
/// line, pages that cover the file exactly, and header counts produced by
/// the same reader as the rows.
@Suite struct StudyRecordReviewTests {

    // MARK: - harness

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(component: "record-review-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        return url
    }

    private func line(_ object: [String: Any]) throws -> String {
        String(
            decoding: try JSONSerialization.data(
                withJSONObject: object, options: [.sortedKeys]),
            as: UTF8.self)
    }

    private func response(
        _ index: Int, condition: String = "baseline",
        extra: [String: Any] = [:]
    ) throws -> String {
        var object: [String: Any] = [
            "condition": condition, "promptID": "p\(index)",
            "prompt": "Describe room \(index).",
            "output": "A plain answer about room \(index).",
            "wordCount": 6, "distinct2": 1.0, "sampleIndex": 0,
        ]
        for (key, value) in extra { object[key] = value }
        return try line(object)
    }

    private static let noun = StudyReviewNoun("record", "records")

    // MARK: - paging

    @Test func pagesCoverEveryRowExactlyOnce() {
        let paging = StudyReviewPaging(total: 1_240, pageSize: 50)
        #expect(paging.pageCount == 25)
        var covered: [Int] = []
        for page in 0..<paging.pageCount {
            covered += Array(paging.range(forPage: page))
        }
        // No row skipped, none repeated, in order.
        #expect(covered == Array(0..<1_240))
        #expect(paging.range(forPage: 24) == 1_200..<1_240)

        // A page outside the list is brought into it, never an empty view
        // that reads as "no more rows".
        #expect(paging.range(forPage: 99) == 1_200..<1_240)
        #expect(paging.range(forPage: -3) == 0..<50)
        #expect(!paging.hasPrevious(0))
        #expect(paging.hasNext(0))
        #expect(paging.hasPrevious(24))
        #expect(!paging.hasNext(24))
    }

    @Test func theCaptionAlwaysStatesTheTotal() {
        let noun = Self.noun
        #expect(
            StudyReviewPaging(total: 1_240, pageSize: 50)
                .caption(forPage: 1, noun: noun)
                == "Showing 51–100 of 1,240 records · page 2 of 25.")
        #expect(
            StudyReviewPaging(total: 12).caption(forPage: 0, noun: noun)
                == "Showing all 12 records.")
        #expect(
            StudyReviewPaging(total: 1).caption(forPage: 0, noun: noun)
                == "Showing the 1 record.")
        #expect(
            StudyReviewPaging(total: 0).caption(forPage: 0, noun: noun)
                == "No records to show.")
        // A last page of one row.
        let uneven = StudyReviewPaging(total: 101, pageSize: 50)
        #expect(uneven.pageCount == 3)
        #expect(uneven.range(forPage: 2) == 100..<101)
        #expect(
            uneven.caption(forPage: 2, noun: noun)
                == "Showing record 101 of 101 · page 3 of 3.")
    }

    // MARK: - the file, by line

    @Test func theRecordCountIsTheFilesNonBlankLines() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appending(component: "records.jsonl")
        // Windows line endings, blank lines, indentation, and no newline
        // after the last record.
        try Data("{\"a\":1}\r\n\n   \n  {\"a\":2}  \n{\"a\":3}".utf8).write(to: url)
        let file = StudyRecordFile(url: url)
        #expect(file.isReadable)
        #expect(file.count == 3)
        #expect(String(decoding: file.line(at: 0), as: UTF8.self) == "{\"a\":1}")
        #expect(String(decoding: file.line(at: 1), as: UTF8.self) == "{\"a\":2}")
        #expect(String(decoding: file.line(at: 2), as: UTF8.self) == "{\"a\":3}")

        let missing = StudyRecordFile(url: directory.appending(component: "none.jsonl"))
        #expect(!missing.isReadable)
        #expect(missing.count == 0)
        let empty = directory.appending(component: "empty.jsonl")
        try Data().write(to: empty)
        #expect(StudyRecordFile(url: empty).isReadable)
        #expect(StudyRecordFile.recordCount(at: empty) == 0)
    }

    /// The file is streamed in chunks, never held whole. Where a chunk ends
    /// must not change what a line is: every chunk size gives the same
    /// records, including sizes that split a line, a multi-byte character,
    /// and a line ending.
    @Test func theIndexIsTheSameWhereverTheChunksFall() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appending(component: "records.jsonl")
        let lines = [
            "{\"output\":\"plain\"}",
            "{\"output\":\"café — naïve\"}",
            "",
            "{\"output\":\"" + String(repeating: "long ", count: 40) + "\"}",
            "   ",
            "{\"output\":\"last, with no newline after it\"}",
        ]
        try Data(lines.joined(separator: "\r\n").utf8).write(to: url)
        let expected = lines.filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }

        for chunkSize in [1, 2, 3, 7, 16, 64, 1 << 20] {
            var visited: [String] = []
            let file = StudyRecordFile(url: url, chunkSize: chunkSize) { line in
                visited.append(String(decoding: line, as: UTF8.self))
            }
            #expect(file.isReadable)
            #expect(file.count == 4, "chunk size \(chunkSize)")
            // What the single pass was shown, and what a later read by
            // position returns, are the same records.
            #expect(visited == expected, "chunk size \(chunkSize)")
            #expect(
                file.lines(at: 0..<file.count).map { String(decoding: $0, as: UTF8.self) }
                    == expected,
                "chunk size \(chunkSize)")
        }
    }

    /// A file that changes after it was counted must cost the reader a
    /// labelled row, never the app: the review reads by position on demand
    /// and nothing is memory-mapped.
    @Test func aFileShortenedAfterItWasCountedGivesUnreadableRows() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appending(component: "generations.jsonl")
        let lines = try (0..<3).map { try response($0) }
        try Data(lines.joined(separator: "\n").utf8).write(to: url)
        let review = StudyRecordReview<StudyResponseRow>(url: url)
        #expect(review.counts.total == 3)

        // Keep the first record only.
        try Data((lines[0] + "\n").utf8).write(to: url)
        let rows = review.page(0).rows
        #expect(rows.count == 3)
        #expect(rows[0].kind == .response)
        #expect(rows[2].kind == .unreadable)
        #expect(
            rows[2].failure
                == "This record can no longer be read from the file, which "
                + "changed after it was counted.")

        // Removed altogether: still rows, still no fault.
        try FileManager.default.removeItem(at: url)
        #expect(review.page(0).rows.allSatisfy { $0.kind == .unreadable })
    }

    // MARK: - a thousand records

    /// The done-when case: a run of 1,000 records is reviewable in full —
    /// every record on exactly one page, and the header's counts equal to
    /// what paging through the file shows.
    @Test func aRunOfAThousandRecordsIsReviewableInFull() throws {
        let run = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: run) }

        var lines: [String] = []
        for index in 0..<1_000 {
            switch index {
            case 100:
                // The Python engine's failure record: no prompt, no output.
                lines.append(
                    try line([
                        "experiment": "example", "condition": "warm",
                        "error": "the saved settings could not be loaded",
                        "variantArtifactPath": "variants/warm",
                    ]))
            case 200, 201:
                lines.append(
                    try line([
                        "condition": "baseline", "promptID": "p\(index)",
                        "prompt": "Is the room warm?",
                        "instrument": "answerTokenLogprob",
                        "options": ["yes", "no"], "selected": "yes",
                        "margin": 1.25,
                    ]))
            case 300..<320:
                lines.append(try response(index, extra: ["finishReason": "length"]))
            case 400..<405:
                lines.append(
                    try response(index, extra: ["finishReason": "lengthInReasoning"]))
            case 500:
                lines.append("this line is not JSON")
            case 600:
                lines.append(try line(["condition": "baseline", "promptID": "p600"]))
            default:
                // Half stamped `stop`, half from before the stamp existed.
                lines.append(
                    try response(
                        index, extra: index % 2 == 0 ? ["finishReason": "stop"] : [:]))
            }
        }
        let generations = run.appending(component: "generations.jsonl")
        try Data((lines.joined(separator: "\n") + "\n").utf8).write(to: generations)

        let review = StudyRecordReview<StudyResponseRow>(url: generations)
        #expect(review.counts.total == 1_000)
        #expect(review.counts.count(.response) == 970)
        #expect(review.counts.count(.cutOff) == 25)
        #expect(review.counts.count(.failed) == 1)
        #expect(review.counts.count(.reading) == 2)
        #expect(review.counts.count(.unreadable) == 2)
        #expect(review.counts.needingAttention == 28)
        // Every kind is in the header, zero or not, in a fixed order.
        #expect(
            review.counts.summary.map(\.label) == [
                "Responses", "Cut off", "Failed", "Answer-option readings",
                "Unreadable",
            ])
        #expect(review.counts.summary.map(\.count) == [970, 25, 1, 2, 2])
        #expect(
            review.counts.summary.map(\.needsAttention)
                == [false, true, true, false, true])

        // Page through the whole file.
        let first = review.page(0)
        #expect(first.paging.pageCount == 20)
        #expect(first.caption == "Showing 1–50 of 1,000 records · page 1 of 20.")
        var seen: [StudyResponseRow] = []
        for page in 0..<first.paging.pageCount {
            seen += review.page(page).rows
        }
        #expect(seen.count == 1_000)
        #expect(seen.map(\.record) == Array(1...1_000))
        // The rows on the pages ARE the header's counts.
        #expect(
            Dictionary(grouping: seen, by: \.kind).mapValues(\.count)
                == review.counts.byKind)
        #expect(
            review.page(19).caption
                == "Showing 951–1,000 of 1,000 records · page 20 of 20.")

        // A filter reaches the rows that need attention without paging past
        // the ones that do not, and still says how large the file is.
        let failed = review.page(0, kind: .failed)
        #expect(failed.rows.map(\.record) == [101])
        #expect(
            failed.caption
                == "Showing the 1 failure record. The file holds 1,000 records in all.")
        let cutOff = review.page(0, kind: .cutOff, pageSize: 10)
        #expect(cutOff.rows.map(\.record) == Array(301...310))
        #expect(
            cutOff.caption
                == "Showing 1–10 of 25 cut-off responses · page 1 of 3. "
                + "The file holds 1,000 records in all.")
        #expect(review.page(2, kind: .cutOff, pageSize: 10).rows.count == 5)
    }

    @Test func loadingOffTheCallingActorGivesTheSameReview() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appending(component: "generations.jsonl")
        let lines = try (0..<120).map { try response($0) }
        try Data(lines.joined(separator: "\n").utf8).write(to: url)

        let loaded = await StudyRecordReview<StudyResponseRow>.load(url: url)
        #expect(loaded.review.counts.total == 120)
        #expect(loaded.firstPage == loaded.review.page(0))
        let last = await loaded.review.loadPage(2)
        #expect(last.rows.map(\.record) == Array(101...120))
    }

    // MARK: - response rows say what they are

    @Test func aCutOffResponseIsLabelledAndItsTextIsKept() throws {
        let row = StudyResponseRow.read(
            Data(try response(7, extra: ["finishReason": "length"]).utf8), record: 8)
        #expect(row.kind == .cutOff)
        #expect(row.kind.needsAttention)
        #expect(row.finishReason == "length")
        #expect(row.output == "A plain answer about room 7.")
        #expect(row.wordCount == 6)
        #expect(row.statusNote?.contains("length limit") == true)
        #expect(row.statusNote?.contains("unfinished") == true)

        let reasoning = StudyResponseRow.read(
            Data(try response(7, extra: ["finishReason": "lengthInReasoning"]).utf8),
            record: 1)
        #expect(reasoning.kind == .cutOff)
        #expect(reasoning.statusNote?.contains("reasoning allowance") == true)

        let cancelled = StudyResponseRow.read(
            Data(try response(7, extra: ["finishReason": "cancelled"]).utf8), record: 1)
        #expect(cancelled.kind == .cutOff)
        #expect(cancelled.statusNote?.contains("cancelled") == true)

        // `stop`, and no reason at all, are ordinary responses with no note.
        for extra in [["finishReason": "stop"], [:]] as [[String: Any]] {
            let finished = StudyResponseRow.read(
                Data(try response(7, extra: extra).utf8), record: 1)
            #expect(finished.kind == .response)
            #expect(finished.statusNote == nil)
        }
    }

    @Test func aFailureRecordIsAFailedRowNotAMissingOne() throws {
        let text = try line([
            "experiment": "example", "condition": "warm",
            "error": "the saved settings could not be loaded",
            "variantArtifactPath": "variants/warm",
        ])
        let row = StudyResponseRow.read(Data(text.utf8), record: 3)
        #expect(row.kind == .failed)
        #expect(row.condition == "warm")
        #expect(row.promptID == nil)
        #expect(row.title == "warm")
        #expect(row.failure == "the saved settings could not be loaded")
        #expect(row.output == nil)
        #expect(row.statusNote?.contains("produced no response") == true)
        #expect(row.rawJSON.contains("variantArtifactPath"))
    }

    @Test func anAnswerOptionReadingIsShownAsAReading() throws {
        let text = try line([
            "condition": "baseline", "promptID": "p1",
            "prompt": "Is the room warm?", "instrument": "answerTokenLogprob",
            "options": ["yes", "no"], "selected": "yes", "margin": 1.25,
        ])
        let row = StudyResponseRow.read(Data(text.utf8), record: 1)
        #expect(row.kind == .reading)
        #expect(!row.kind.needsAttention)
        #expect(row.selected == "yes")
        #expect(row.margin == 1.25)
        #expect(row.output == nil)
        #expect(row.statusNote?.contains("by design") == true)
    }

    @Test func aLineThatCannotBeReadIsStillARow() throws {
        let garbage = StudyResponseRow.read(Data("not JSON at all".utf8), record: 5)
        #expect(garbage.kind == .unreadable)
        #expect(garbage.rawJSON == "not JSON at all")
        #expect(garbage.failure == "This line is not valid JSON.")
        #expect(garbage.title == "no condition recorded")

        let shapeless = StudyResponseRow.read(
            Data(try line(["condition": "baseline", "promptID": "p1"]).utf8), record: 6)
        #expect(shapeless.kind == .unreadable)
        #expect(shapeless.failure?.contains("no response text") == true)
        #expect(shapeless.statusNote?.contains("exactly as written") == true)
    }

    @Test func aRecordedParseFailureIsSaidOnTheRow() throws {
        // `parsedChoice: null` is the engines' record that no answer could
        // be read; an absent key means the item has no listed answers.
        let failed = StudyResponseRow.read(
            Data(try response(1, extra: ["parsedChoice": NSNull()]).utf8), record: 1)
        #expect(failed.kind == .response)
        #expect(failed.answerNotRead)
        #expect(failed.answerNote?.contains("recorded its answer as missing") == true)

        let parsed = StudyResponseRow.read(
            Data(try response(1, extra: ["parsedChoice": "yes"]).utf8), record: 1)
        #expect(!parsed.answerNotRead)
        #expect(parsed.answerNote == nil)
        let notApplicable = StudyResponseRow.read(Data(try response(1).utf8), record: 1)
        #expect(!notApplicable.answerNotRead)
    }

    @Test func rowTitlesTellSamplesAndSpeakersApart() throws {
        let second = StudyResponseRow.read(
            Data(
                try response(
                    4, condition: "warm",
                    extra: ["sampleIndex": 1, "speakerName": "Second seat"]
                ).utf8),
            record: 9)
        #expect(second.title == "warm · p4 · sample 2 · Second seat")
        #expect(second.id == 9)
        // A numeric prompt ID still reads.
        let numeric = StudyResponseRow.read(
            Data(try response(4, extra: ["promptID": 17]).utf8), record: 1)
        #expect(numeric.promptID == "17")
    }

    // MARK: - long text

    @Test func anExcerptSaysHowMuchOfTheTextItShows() {
        let short = StudyReviewText.excerpt("a short response")
        #expect(short.shown == "a short response")
        #expect(short.note == nil)

        let long = String(repeating: "word ", count: 2_469)
        #expect(long.count == 12_345)
        let excerpt = StudyReviewText.excerpt(long)
        #expect(excerpt.shown.count == StudyReviewText.excerptLimit)
        #expect(long.hasPrefix(excerpt.shown))
        #expect(excerpt.note == "Showing the first 4,000 of 12,345 characters.")
    }
}
