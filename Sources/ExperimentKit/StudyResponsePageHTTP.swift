import Foundation

/// The browser companion's response review: every record of the selected
/// run's generations.jsonl, served a bounded page at a time.
///
/// The state document the companion polls carries a PREVIEW of at most
/// `StudyResultRepository.previewResponseLimit` responses, and the companion
/// used to show only that, with that number on its button. This route serves
/// the whole file the way the native review does (`StudyRecordReview`):
/// one row per line, rows read by `StudyResponseRow`, the total counted from
/// the file itself, and nothing recomputed.
///
/// `GET /api/experiment/result/responses?offset=<n>&limit=<n>`: `offset` is
/// the 0-based record to start at (default 0); `limit` is how many to return
/// (default `defaultLimit`, at most `maximumLimit`; a larger value is reduced
/// to it, and the reply says which limit it used).
enum StudyResponsePageHTTP {
    static let defaultLimit = StudyReviewPaging.defaultPageSize
    static let maximumLimit = 200

    struct Row: Encodable, Equatable {
        /// 1-based position of this record in the file.
        let record: Int
        let kind: String
        let title: String
        /// What a reader must know about a row that is not an ordinary,
        /// complete response; nil for an ordinary response.
        let statusNote: String?
        let answerNote: String?
        let condition: String?
        let promptID: String?
        let sampleIndex: Int?
        let prompt: String?
        let promptNote: String?
        let output: String?
        /// Says how much of a long response is shown; nil when all of it is.
        let outputNote: String?
        let wordCount: Int?
        let distinct2: Double?
        let finishReason: String?
        let failure: String?
        let instrument: String?
        let selected: String?
        let margin: Double?
        /// The line as written, for a line that cannot be read as a record.
        let rawJSON: String?

        init(_ row: StudyResponseRow) {
            record = row.record
            kind = row.kind.rawValue
            title = row.title
            statusNote = row.statusNote
            answerNote = row.answerNote
            condition = row.condition
            promptID = row.promptID
            sampleIndex = row.sampleIndex
            let prompt = row.prompt.map { StudyReviewText.excerpt($0) }
            self.prompt = prompt?.shown
            promptNote = prompt?.note
            let output = row.output.map { StudyReviewText.excerpt($0) }
            self.output = output?.shown
            outputNote = output?.note
            wordCount = row.wordCount
            distinct2 = row.distinct2
            finishReason = row.finishReason
            failure = row.failure
            instrument = row.instrument
            selected = row.selected
            margin = row.margin
            rawJSON = row.kind == .unreadable ? row.rawJSON : nil
        }
    }

    struct Count: Encodable, Equatable {
        let label: String
        let count: Int
        let needsAttention: Bool
    }

    struct Page: Encodable {
        let ok = true
        let directoryName: String
        /// Every record the file holds — the true total, never the preview's.
        let total: Int
        /// The first record returned, 0-based, after bringing the request
        /// into range.
        let offset: Int
        /// The page size used.
        let limit: Int
        /// Per-kind totals over the whole file, every kind listed.
        let counts: [Count]
        /// States how many records are shown out of how many.
        let caption: String
        let rows: [Row]
    }

    /// The page for a request path, read from the selected result's run
    /// directory. Reads the file; call it off the main actor.
    static func page(selected detail: StudyRunDetail?, path: String) -> StudyAuthoringHTTP.Response {
        let items = URLComponents(string: "http://localhost\(path)")?.queryItems ?? []
        func value(_ name: String) -> String? { items.first { $0.name == name }?.value }
        let repair = "Use whole numbers: offset 0 or more, and limit from 1 to "
            + "\(maximumLimit)."
        var offset = 0
        if let raw = value("offset") {
            guard let parsed = Int(raw), parsed >= 0 else {
                return .failure(
                    "invalidResponsePage", "The offset '\(raw)' is not a whole number of 0 or more.",
                    repair: repair)
            }
            offset = parsed
        }
        var limit = defaultLimit
        if let raw = value("limit") {
            guard let parsed = Int(raw), parsed >= 1 else {
                return .failure(
                    "invalidResponsePage", "The limit '\(raw)' is not a whole number of 1 or more.",
                    repair: repair)
            }
            limit = min(parsed, maximumLimit)
        }
        guard let detail else {
            return .failure(
                "noResultSelected", "No run is selected, so there are no responses to show.",
                repair: "Choose a run under Results, then open its responses again.",
                status: "409 Conflict")
        }
        let url = StudyResultRepository.responsesURL(runDirectory: URL(filePath: detail.item.path))
        let review = StudyRecordReview<StudyResponseRow>(url: url)
        let total = review.counts.total
        offset = min(offset, total)
        let positions = offset..<min(offset + limit, total)
        let rows = zip(positions, review.file.lines(at: positions)).map { position, line in
            Row(StudyResponseRow.read(line, record: position + 1))
        }
        let caption = review.file.isReadable
            ? caption(offset: offset, shown: rows.count, total: total)
            : "This run has no readable generations.jsonl, so there are no responses to show."
        return .json(
            Page(
                directoryName: detail.item.directoryName, total: total, offset: offset,
                limit: limit,
                counts: review.counts.summary.map {
                    Count(label: $0.label, count: $0.count, needsAttention: $0.needsAttention)
                },
                caption: caption, rows: rows))
    }

    /// "Showing 51–100 of 1,240 records." Always states the total, so a page
    /// is never mistaken for the whole file.
    static func caption(offset: Int, shown: Int, total: Int) -> String {
        let noun = StudyResponseRow.noun
        let grouped = StudyReviewText.grouped
        guard total > 0 else { return "No \(noun.plural) to show." }
        guard shown > 0 else {
            return "No \(noun.plural) from this position; the file holds "
                + "\(noun.counted(total))."
        }
        if shown == total {
            return total == 1 ? "Showing the 1 \(noun.singular)." : "Showing all \(noun.counted(total))."
        }
        if shown == 1 {
            return "Showing \(noun.singular) \(grouped(offset + 1)) of \(grouped(total))."
        }
        return "Showing \(grouped(offset + 1))–\(grouped(offset + shown)) of \(noun.counted(total))."
    }
}
