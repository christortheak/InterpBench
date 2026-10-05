import Foundation

// The study Results view's record review: every line of generations.jsonl
// and judgments.jsonl, in pages, with nothing dropped.
//
// What this replaces read the first 80 lines of one file and the first 200
// of the other, and silently skipped any line its strict decoder disliked —
// a noncompliant judge row (its `judgment` is null), an answer-option
// reading, a failure record, every row the Python engine writes for a
// judgment. The view then said "every". Three rules here undo that:
//
//   1. ONE ROW PER LINE. A line that cannot be read is a row labelled
//      unreadable, shown as written. The row count is the line count.
//   2. PAGES, NOT A PREFIX. `StudyRecordFile` indexes the file's lines
//      without decoding them, so a page decodes only its own rows and the
//      count is exact however large the file is.
//   3. THE COUNT COMES FROM THE SAME READER AS THE ROWS. The per-kind
//      totals in the header are produced by classifying each line with the
//      code that builds its row, so the header cannot disagree with what
//      paging through the file would show.
//
// Read-only, and nothing is recomputed: a row shows what its line holds.

// MARK: - Text

public enum StudyReviewText {
    /// Whole numbers with thousands separators, the same on every machine
    /// ("1,240") so the counts read identically in the app and in tests.
    public static func grouped(_ value: Int) -> String {
        value.formatted(.number.locale(Locale(identifier: "en_US")))
    }

    /// How much of a long prompt or response a card shows before the reader
    /// asks for the rest.
    public static let excerptLimit = 4_000

    /// The text a card shows, and — when that is not all of it — a note
    /// saying exactly how much is shown. Never a bare ellipsis: a reader
    /// must be able to tell a short response from a shortened one.
    public static func excerpt(
        _ text: String, limit: Int = excerptLimit
    ) -> (shown: String, note: String?) {
        let length = text.count
        guard length > limit else { return (text, nil) }
        return (
            String(text.prefix(limit)),
            "Showing the first \(grouped(limit)) of \(grouped(length)) characters."
        )
    }
}

/// How a kind of record is named in a count ("1 judge row", "3 judge rows").
public struct StudyReviewNoun: Sendable, Equatable {
    public let singular: String
    public let plural: String

    public init(_ singular: String, _ plural: String) {
        self.singular = singular
        self.plural = plural
    }

    public func counted(_ count: Int) -> String {
        "\(StudyReviewText.grouped(count)) \(count == 1 ? singular : plural)"
    }
}

// MARK: - Paging

/// Page arithmetic for a list of `total` rows. Pure, so the count and the
/// caption can be tested without a file or a view.
public struct StudyReviewPaging: Sendable, Equatable {
    public static let defaultPageSize = 50

    public let total: Int
    public let pageSize: Int

    public init(total: Int, pageSize: Int = defaultPageSize) {
        self.total = max(0, total)
        self.pageSize = max(1, pageSize)
    }

    /// At least 1: an empty list still has one (empty) page to show.
    public var pageCount: Int {
        total == 0 ? 1 : (total + pageSize - 1) / pageSize
    }

    public func clamped(_ page: Int) -> Int {
        min(max(0, page), pageCount - 1)
    }

    /// The rows on a page, as positions in the list. Pages cover the list
    /// exactly: every position is on one page and no page overlaps another.
    public func range(forPage page: Int) -> Range<Int> {
        let lower = min(clamped(page) * pageSize, total)
        return lower..<min(lower + pageSize, total)
    }

    public func hasPrevious(_ page: Int) -> Bool { clamped(page) > 0 }
    public func hasNext(_ page: Int) -> Bool { clamped(page) < pageCount - 1 }

    /// "Showing 51–100 of 1,240 records · page 2 of 25." Always states the
    /// total, so a page is never mistaken for the whole file.
    public func caption(forPage page: Int, noun: StudyReviewNoun) -> String {
        guard total > 0 else { return "No \(noun.plural) to show." }
        guard pageCount > 1 else {
            return total == 1
                ? "Showing the 1 \(noun.singular)."
                : "Showing all \(noun.counted(total))."
        }
        let range = range(forPage: page)
        let position = " · page \(StudyReviewText.grouped(clamped(page) + 1)) of "
            + "\(StudyReviewText.grouped(pageCount))."
        // A last page holding one row reads "record 101 of 101", not the
        // range "101–101".
        guard range.count > 1 else {
            return "Showing \(noun.singular) "
                + "\(StudyReviewText.grouped(range.upperBound)) of "
                + "\(StudyReviewText.grouped(total))" + position
        }
        return "Showing \(StudyReviewText.grouped(range.lowerBound + 1))–"
            + "\(StudyReviewText.grouped(range.upperBound)) of "
            + "\(noun.counted(total))" + position
    }
}

// MARK: - The file, by line

/// A JSONL evidence file indexed by line.
///
/// Building the index streams the file once in fixed-size chunks and keeps
/// only where each record starts and ends, so the record count is exact
/// however large the file is, the file is never held in memory, and a page
/// reads only its own lines. Nothing is memory-mapped: a file that another
/// process is still appending to, or shortens, yields a row that says it
/// cannot be read — never a fault.
public struct StudyRecordFile: Sendable {
    public let url: URL
    /// false when the file is missing or could not be read. A missing file
    /// has zero records; the views say "no file", not "no records".
    public let isReadable: Bool
    /// Byte ranges of the non-blank lines, in file order.
    private let ranges: [Range<UInt64>]

    static let chunkSize = 1 << 20

    public init(url: URL) {
        self.init(url: url, chunkSize: Self.chunkSize, visit: nil)
    }

    /// `visit` is handed each record's bytes once, in file order, as the
    /// index is built — how a review classifies every line in the same
    /// single pass that counts them.
    init(url: URL, chunkSize: Int = StudyRecordFile.chunkSize, visit: ((Data) -> Void)?) {
        self.url = url
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            self.isReadable = false
            self.ranges = []
            return
        }
        defer { try? handle.close() }
        var ranges: [Range<UInt64>] = []
        // The unfinished last line of what has been read so far, and where
        // in the file it starts.
        var pending = Data()
        var pendingStart: UInt64 = 0
        var offset: UInt64 = 0
        var readable = true

        func emit(_ buffer: Data, _ lower: Int, _ upper: Int, base: UInt64) {
            var lower = lower
            var upper = upper
            while lower < upper, Self.isBlank(buffer[buffer.startIndex + lower]) { lower += 1 }
            while upper > lower, Self.isBlank(buffer[buffer.startIndex + upper - 1]) { upper -= 1 }
            guard lower < upper else { return }
            ranges.append((base + UInt64(lower))..<(base + UInt64(upper)))
            visit?(
                buffer.subdata(
                    in: (buffer.startIndex + lower)..<(buffer.startIndex + upper)))
        }

        while true {
            let chunk: Data
            do {
                guard let read = try handle.read(upToCount: max(1, chunkSize)),
                    !read.isEmpty
                else { break }
                chunk = read
            } catch {
                // A read that fails part-way leaves an honest partial index
                // only if the caller is told; an unreadable file says so.
                readable = false
                break
            }
            let base = pending.isEmpty ? offset : pendingStart
            let buffer = pending.isEmpty ? chunk : pending + chunk
            offset += UInt64(chunk.count)
            // The carried-over part holds no newline, so the search resumes
            // where the new bytes begin.
            var search = pending.count
            var lineStart = 0
            let count = buffer.count
            buffer.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
                guard let address = bytes.baseAddress else { return }
                while search < count,
                    let found = memchr(address + search, 0x0A, count - search)
                {
                    let end = address.distance(to: UnsafeRawPointer(found))
                    emit(buffer, lineStart, end, base: base)
                    lineStart = end + 1
                    search = lineStart
                }
            }
            if lineStart < count {
                pending = buffer.subdata(
                    in: (buffer.startIndex + lineStart)..<buffer.endIndex)
                pendingStart = base + UInt64(lineStart)
            } else {
                pending = Data()
            }
        }
        // A last record with no newline after it is still a record.
        if !pending.isEmpty {
            emit(pending, 0, pending.count, base: pendingStart)
        }
        self.isReadable = readable
        self.ranges = readable ? ranges : []
    }

    private static func isBlank(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 0x09 || byte == 0x0D
    }

    /// Every record the file holds: its non-blank lines.
    public var count: Int { ranges.count }

    /// The bytes of the records at these 0-based positions, in the order
    /// asked for. A record that can no longer be read (the file was removed
    /// or shortened after it was indexed) comes back empty, which a row
    /// reader reports as an unreadable line.
    public func lines(at positions: some Sequence<Int>) -> [Data] {
        let handle = try? FileHandle(forReadingFrom: url)
        defer { try? handle?.close() }
        return positions.map { position in
            let range = ranges[position]
            guard let handle,
                (try? handle.seek(toOffset: range.lowerBound)) != nil,
                let data = try? handle.read(
                    upToCount: Int(range.upperBound - range.lowerBound))
            else { return Data() }
            return data
        }
    }

    /// The bytes of one record (0-based position).
    public func line(at position: Int) -> Data {
        lines(at: CollectionOfOne(position))[0]
    }

    /// The record count of a file without keeping its index.
    public static func recordCount(at url: URL) -> Int {
        StudyRecordFile(url: url).count
    }
}

// MARK: - Rows

/// What a kind of row is called, and whether a reader should look at it.
public protocol StudyReviewRowKind: Hashable, Sendable, CaseIterable {
    /// Sentence-case plural for a filter and a count ("Cut off").
    var label: String { get }
    var noun: StudyReviewNoun { get }
    /// true for kinds that are not an ordinary, complete record.
    var needsAttention: Bool { get }
}

/// One line of an evidence file, as the review shows it.
public protocol StudyReviewRow: Identifiable, Sendable, Equatable where ID == Int {
    associatedtype Kind: StudyReviewRowKind
    /// 1-based position of this record in its file.
    var record: Int { get }
    var kind: Kind { get }
    /// What every row of this file is called, whatever its kind.
    static var noun: StudyReviewNoun { get }
    /// Reads one line. Never fails: a line that cannot be read comes back
    /// as a row of the unreadable kind, carrying the line as written.
    static func read(_ line: Data, record: Int) -> Self
    /// The kind `read` gives this line, without building the row — the
    /// same classification, so a count and a page can never disagree.
    static func kind(of line: Data) -> Kind
}

extension StudyReviewRow {
    public var id: Int { record }
}

/// Per-kind totals over a whole file.
public struct StudyReviewCounts<Kind: StudyReviewRowKind>: Sendable, Equatable {
    public internal(set) var total = 0
    public internal(set) var byKind: [Kind: Int] = [:]

    public func count(_ kind: Kind) -> Int { byKind[kind] ?? 0 }

    /// Rows of the kinds a reader should look at.
    public var needingAttention: Int {
        Kind.allCases.filter(\.needsAttention).map { count($0) }.reduce(0, +)
    }

    /// One entry per kind, zero or not, in the kinds' declared order — the
    /// header shows every kind so "none" is stated rather than implied.
    public var summary: [StudyReviewSummaryItem] {
        Kind.allCases.map { kind in
            StudyReviewSummaryItem(
                id: "\(kind)", label: kind.label, count: count(kind),
                needsAttention: kind.needsAttention && count(kind) > 0)
        }
    }
}

public struct StudyReviewSummaryItem: Identifiable, Sendable, Equatable {
    public let id: String
    public let label: String
    public let count: Int
    public let needsAttention: Bool
}

// MARK: - Review of one file

/// One evidence file, counted in full and read a page at a time.
///
/// `init` reads the file once and classifies every line; call it off the
/// main actor for a large file. After that a page is a bounded decode.
public struct StudyRecordReview<Row: StudyReviewRow>: Sendable {
    public let file: StudyRecordFile
    /// One kind per record, in file order.
    public let kinds: [Row.Kind]
    public let counts: StudyReviewCounts<Row.Kind>

    public init(url: URL) {
        var kinds: [Row.Kind] = []
        var counts = StudyReviewCounts<Row.Kind>()
        // One pass: each line is classified as it is indexed, so the counts
        // and the index describe the same reading of the file.
        let file = StudyRecordFile(url: url) { line in
            let kind = Row.kind(of: line)
            kinds.append(kind)
            counts.total += 1
            counts.byKind[kind, default: 0] += 1
        }
        self.file = file
        // An unreadable file has no rows; its partial tally is not shown.
        self.kinds = file.isReadable ? kinds : []
        self.counts = file.isReadable ? counts : StudyReviewCounts<Row.Kind>()
    }

    /// Positions (0-based) of the records of one kind, or of all of them.
    public func positions(of kind: Row.Kind?) -> [Int] {
        guard let kind else { return Array(kinds.indices) }
        return kinds.indices.filter { kinds[$0] == kind }
    }

    public struct Page: Sendable, Equatable {
        public let rows: [Row]
        /// The page shown, 0-based — the requested page, brought into range.
        public let page: Int
        public let paging: StudyReviewPaging
        /// States how many rows are shown out of how many, and — under a
        /// filter — how many the file holds in all.
        public let caption: String
    }

    public func page(
        _ page: Int, kind: Row.Kind? = nil,
        pageSize: Int = StudyReviewPaging.defaultPageSize
    ) -> Page {
        let positions = positions(of: kind)
        let paging = StudyReviewPaging(total: positions.count, pageSize: pageSize)
        let shown = Array(positions[paging.range(forPage: page)])
        let rows = zip(shown, file.lines(at: shown)).map { position, line in
            Row.read(line, record: position + 1)
        }
        var caption = paging.caption(
            forPage: page, noun: kind?.noun ?? Row.noun)
        if kind != nil {
            caption += " The file holds \(Row.noun.counted(counts.total)) in all."
        }
        return Page(
            rows: rows, page: paging.clamped(page), paging: paging,
            caption: caption)
    }

    /// Reads and counts the file, and decodes its first page, off the
    /// calling actor — what a view awaits so a large file never blocks it.
    public static func load(
        url: URL, pageSize: Int = StudyReviewPaging.defaultPageSize
    ) async -> (review: StudyRecordReview<Row>, firstPage: Page) {
        await Task.detached(priority: .userInitiated) {
            let review = StudyRecordReview<Row>(url: url)
            return (review, review.page(0, pageSize: pageSize))
        }.value
    }

    /// `page(_:kind:pageSize:)`, decoded off the calling actor.
    public func loadPage(
        _ page: Int, kind: Row.Kind? = nil,
        pageSize: Int = StudyReviewPaging.defaultPageSize
    ) async -> Page {
        await Task.detached(priority: .userInitiated) {
            self.page(page, kind: kind, pageSize: pageSize)
        }.value
    }
}

// MARK: - Lenient JSON

enum StudyReviewJSON {
    static func object(_ line: Data) -> [String: Any]? {
        (try? JSONSerialization.jsonObject(with: line)) as? [String: Any]
    }

    /// A present, non-null value.
    static func present(_ raw: Any?) -> Any? {
        guard let raw, !(raw is NSNull) else { return nil }
        return raw
    }

    /// JSON booleans are NSNumbers too; a count must not read `true` as 1.
    private static func isBoolean(_ number: NSNumber) -> Bool {
        CFGetTypeID(number) == CFBooleanGetTypeID()
    }

    static func int(_ raw: Any?) -> Int? {
        guard let number = raw as? NSNumber, !isBoolean(number) else { return nil }
        return number.intValue
    }

    static func double(_ raw: Any?) -> Double? {
        guard let number = raw as? NSNumber, !isBoolean(number) else { return nil }
        let value = number.doubleValue
        return value.isNaN ? nil : value
    }

    static func bool(_ raw: Any?) -> Bool? {
        guard let number = raw as? NSNumber, isBoolean(number) else { return nil }
        return number.boolValue
    }

    /// Identifiers are strings on both engines; a numeric one still reads.
    static func identifier(_ raw: Any?) -> String? {
        if let text = raw as? String { return text }
        if let number = raw as? NSNumber, !isBoolean(number) { return "\(number)" }
        return nil
    }

    /// Why a line that is not a record cannot be read, in plain words.
    static func unreadableReason(_ line: Data, otherwise shapeless: String) -> String {
        if line.isEmpty {
            return "This record can no longer be read from the file, which "
                + "changed after it was counted."
        }
        return object(line) == nil ? "This line is not valid JSON." : shapeless
    }

    static func value(_ raw: Any) -> JSONValue {
        switch raw {
        case is NSNull:
            return .null
        case let text as String:
            return .string(text)
        case let number as NSNumber:
            if isBoolean(number) { return .bool(number.boolValue) }
            return .number(number.doubleValue)
        case let list as [Any]:
            return .array(list.map(value))
        case let object as [String: Any]:
            return .object(object.mapValues(value))
        default:
            return .null
        }
    }

    static func values(_ raw: Any?) -> [String: JSONValue]? {
        (raw as? [String: Any])?.mapValues(value)
    }

    /// The line re-indented with sorted keys, or as written when it is not
    /// JSON.
    static func pretty(_ line: Data) -> String {
        guard
            let object = try? JSONSerialization.jsonObject(
                with: line, options: [.fragmentsAllowed]),
            let data = try? JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys, .fragmentsAllowed])
        else { return String(decoding: line, as: UTF8.self) }
        return String(decoding: data, as: UTF8.self)
    }
}

// MARK: - Response rows (generations.jsonl)

/// One line of a run's generations.jsonl.
///
/// The file holds more than responses, on both engines: an answer-option
/// reading for each item of a study that scores the listed options directly,
/// and — on the Python engine — a failure record for an agent whose saved
/// settings could not be loaded. The review shows each as what it is.
public struct StudyResponseRow: StudyReviewRow {
    public enum Kind: String, StudyReviewRowKind {
        /// A generated response that ended on its own, or whose engine
        /// recorded no reason (runs written before the reason was stamped).
        case response
        /// A generated response that was stopped before the model finished.
        case cutOff
        /// The engine recorded an error where a response would have been.
        case failed
        /// A direct reading of the model's preference among listed options.
        /// It has no generated text, by design.
        case reading
        /// A line this view cannot read as a record.
        case unreadable

        public var label: String {
            switch self {
            case .response: "Responses"
            case .cutOff: "Cut off"
            case .failed: "Failed"
            case .reading: "Answer-option readings"
            case .unreadable: "Unreadable"
            }
        }

        public var noun: StudyReviewNoun {
            switch self {
            case .response: StudyReviewNoun("response", "responses")
            case .cutOff: StudyReviewNoun("cut-off response", "cut-off responses")
            case .failed: StudyReviewNoun("failure record", "failure records")
            case .reading:
                StudyReviewNoun("answer-option reading", "answer-option readings")
            case .unreadable: StudyReviewNoun("unreadable line", "unreadable lines")
            }
        }

        public var needsAttention: Bool {
            switch self {
            case .response, .reading: false
            case .cutOff, .failed, .unreadable: true
            }
        }
    }

    public static let noun = StudyReviewNoun("record", "records")

    public let record: Int
    public let kind: Kind
    public let condition: String?
    public let promptID: String?
    public let sampleIndex: Int?
    public let prompt: String?
    /// The whole generated text. A card may show an excerpt
    /// (`StudyReviewText.excerpt`); the row is never shortened.
    public let output: String?
    public let wordCount: Int?
    public let distinct2: Double?
    /// Why generation ended, as the engine recorded it: `stop`, `length`,
    /// `lengthInReasoning`, or `cancelled`. nil on records written before
    /// the reason was stamped, and on rows that are not responses.
    public let finishReason: String?
    /// The engine's recorded error (a failure record), or why this line
    /// cannot be read (an unreadable row).
    public let failure: String?
    /// true when the engine recorded that no answer could be read from
    /// this response (`parsedChoice` or `parsedMonths` present and null).
    public let answerNotRead: Bool
    /// Answer-option readings: which option the model preferred, and by
    /// how much.
    public let instrument: String?
    public let selected: String?
    public let margin: Double?
    /// Multi-agent turns: who spoke.
    public let speakerName: String?
    public let turnTitle: String?
    public let interventionDecisions: JSONValue?
    public let probeMeasurements: JSONValue?
    /// The line, re-indented — or exactly as written when it is not JSON.
    public let rawJSON: String

    /// "condition · prompt ID", with the sample and speaker when recorded.
    public var title: String {
        var parts = [condition ?? "no condition recorded"]
        if let promptID { parts.append(promptID) }
        if let sampleIndex, sampleIndex > 0 { parts.append("sample \(sampleIndex + 1)") }
        if let speakerName { parts.append(speakerName) }
        return parts.joined(separator: " · ")
    }

    /// What a reader must know about a row that is not an ordinary,
    /// complete response — in plain words. nil for an ordinary response.
    public var statusNote: String? {
        switch kind {
        case .response:
            return nil
        case .cutOff:
            switch finishReason {
            case "lengthInReasoning":
                return "Cut off: the model used its whole reasoning allowance "
                    + "and never began its answer. The text below is "
                    + "unfinished."
            case "cancelled":
                return "Cut off: this response was cancelled while it was "
                    + "being written. The text below is unfinished."
            default:
                return "Cut off: this response reached the study's length "
                    + "limit before the model finished. The text below is "
                    + "unfinished."
            }
        case .failed:
            return "Failed: the engine recorded an error here and produced "
                + "no response."
        case .reading:
            return "Answer-option reading: the model's preference among the "
                + "listed options, measured directly. No text was generated "
                + "for this record, by design."
        case .unreadable:
            return "Unreadable: this line is not a record this view can "
                + "read. It is shown below exactly as written."
        }
    }

    /// The extra note for a response whose answer the engine could not read.
    public var answerNote: String? {
        answerNotRead
            ? "No answer could be read from this response, so the run "
                + "recorded its answer as missing."
            : nil
    }

    static let cutOffReasons: Set<String> = ["length", "lengthInReasoning", "cancelled"]

    static func classify(_ object: [String: Any]?) -> Kind {
        guard let object else { return .unreadable }
        // The engines' own rule, in their order: an `error` key marks a
        // failure record, an `instrument` key a reading, and only a record
        // with `output` is a sampled response.
        if StudyReviewJSON.present(object["error"]) != nil { return .failed }
        if StudyReviewJSON.present(object["instrument"]) != nil { return .reading }
        guard object["output"] is String else { return .unreadable }
        if let reason = object["finishReason"] as? String,
            cutOffReasons.contains(reason)
        {
            return .cutOff
        }
        return .response
    }

    public static func kind(of line: Data) -> Kind {
        classify(StudyReviewJSON.object(line))
    }

    public static func read(_ line: Data, record: Int) -> StudyResponseRow {
        let object = StudyReviewJSON.object(line)
        let kind = classify(object)
        let fields = object ?? [:]
        let isResponse = kind == .response || kind == .cutOff
        var failure: String?
        switch kind {
        case .failed:
            let raw = StudyReviewJSON.present(fields["error"])
            failure = raw as? String ?? raw.map { "\($0)" }
        case .unreadable:
            failure = StudyReviewJSON.unreadableReason(
                line,
                otherwise: "This line has no response text, no reading, and "
                    + "no recorded error.")
        case .response, .cutOff, .reading:
            failure = nil
        }
        return StudyResponseRow(
            record: record,
            kind: kind,
            condition: fields["condition"] as? String,
            promptID: StudyReviewJSON.identifier(fields["promptID"]),
            sampleIndex: StudyReviewJSON.int(fields["sampleIndex"]),
            prompt: fields["prompt"] as? String,
            output: fields["output"] as? String,
            wordCount: StudyReviewJSON.int(fields["wordCount"]),
            distinct2: StudyReviewJSON.double(fields["distinct2"]),
            finishReason: fields["finishReason"] as? String,
            failure: failure,
            answerNotRead: isResponse
                && (fields["parsedChoice"] is NSNull
                    || fields["parsedMonths"] is NSNull),
            instrument: fields["instrument"] as? String,
            selected: fields["selected"] as? String,
            margin: StudyReviewJSON.double(fields["margin"]),
            speakerName: fields["speakerName"] as? String,
            turnTitle: fields["turnTitle"] as? String,
            interventionDecisions: StudyReviewJSON.present(
                fields["interventionDecisions"]).map(StudyReviewJSON.value),
            probeMeasurements: StudyReviewJSON.present(
                fields["probeMeasurements"]).map(StudyReviewJSON.value),
            rawJSON: StudyReviewJSON.pretty(line))
    }
}

// MARK: - Judge rows (judgments.jsonl)

/// One line of a judge artifact's judgments.jsonl, from either engine.
///
/// The engines write a verdict row differently — the Mac engine carries the
/// prompt, `conditionWas`, and `conditionResult`; the Python engine carries
/// `outcome` (`variant` where the Mac engine says `condition`) and leaves
/// the prompt in the run. Both write the SAME noncompliant row: `outcome`
/// and `judgment` present and null, `noncompliant: true`, and the reason.
public struct StudyJudgmentRow: StudyReviewRow {
    public enum Kind: String, StudyReviewRowKind {
        /// A judge's verdict on one pair.
        case verdict
        /// A pair the judge answered without a usable verdict. No winner
        /// was recorded and none is invented here.
        case noncompliant
        /// A line this view cannot read as a judge row.
        case unreadable

        public var label: String {
            switch self {
            case .verdict: "Verdicts"
            case .noncompliant: "No verdict (noncompliant)"
            case .unreadable: "Unreadable"
            }
        }

        public var noun: StudyReviewNoun {
            switch self {
            case .verdict: StudyReviewNoun("verdict", "verdicts")
            case .noncompliant:
                StudyReviewNoun("noncompliant row", "noncompliant rows")
            case .unreadable: StudyReviewNoun("unreadable line", "unreadable lines")
            }
        }

        public var needsAttention: Bool { self != .verdict }
    }

    public static let noun = StudyReviewNoun("judge row", "judge rows")

    public let record: Int
    public let kind: Kind
    public let condition: String?
    public let promptID: String?
    public let sampleIndex: Int?
    /// The panel name of the judge that produced this row.
    public let judge: String?
    public let judgeModel: String?
    /// The task prompt, when the row carries it (Mac-engine rows do).
    public let prompt: String?
    /// Which blinded label, A or B, the baseline response was shown under.
    public let baselineWas: String?
    public let conditionWas: String?
    /// The judge's answer in its own blinded terms: A, B, or tie.
    public let winner: String?
    /// The verdict unblinded: `condition`, `baseline`, or `tie`. The Python
    /// engine's `variant` reads as `condition`, the app's word for the arm
    /// that is not the baseline. nil on rows with no verdict.
    public let result: String?
    public let confidence: Double?
    public let briefReason: String?
    /// Why no verdict was recorded (a noncompliant row), as the engine
    /// kept it — or why this line cannot be read (an unreadable row).
    public let reason: String?
    /// true when the judge's answer ran out of room and the winner was
    /// read from what there was (`verdictSalvaged`, `reasoningTruncated`).
    public let answerCutOff: Bool
    public let aScores: [String: JSONValue]?
    public let bScores: [String: JSONValue]?
    public let structuredFields: [String: JSONValue]?
    public let rawJSON: String

    public var title: String {
        var parts = [condition ?? "no condition recorded"]
        if let promptID { parts.append(promptID) }
        if let sampleIndex, sampleIndex > 0 { parts.append("sample \(sampleIndex + 1)") }
        return parts.joined(separator: " · ")
    }

    public var statusNote: String? {
        switch kind {
        case .verdict:
            return answerCutOff
                ? "The judge's answer was cut off. The winner was still "
                    + "readable and was kept; the reason below may be "
                    + "incomplete."
                : nil
        case .noncompliant:
            return "No verdict: the judge answered this pair twice without "
                + "giving a usable verdict. The row is kept for review and "
                + "is not counted in the wins, ties, or agreement figures."
        case .unreadable:
            return "Unreadable: this line is not a judge row this view can "
                + "read. It is shown below exactly as written."
        }
    }

    static func classify(_ object: [String: Any]?) -> Kind {
        guard let object else { return .unreadable }
        if StudyReviewJSON.bool(object["noncompliant"]) == true {
            return .noncompliant
        }
        return unblinded(object) == nil ? .unreadable : .verdict
    }

    /// The verdict in the app's words, from whichever key the engine used.
    private static func unblinded(_ object: [String: Any]) -> String? {
        if let result = object["conditionResult"] as? String { return result }
        switch object["outcome"] as? String {
        case "variant": return "condition"
        case "baseline": return "baseline"
        case "tie": return "tie"
        default: return nil
        }
    }

    public static func kind(of line: Data) -> Kind {
        classify(StudyReviewJSON.object(line))
    }

    public static func read(_ line: Data, record: Int) -> StudyJudgmentRow {
        let object = StudyReviewJSON.object(line)
        let kind = classify(object)
        let fields = object ?? [:]
        let verdict = fields["judgment"] as? [String: Any] ?? [:]
        let baselineWas = fields["baselineWas"] as? String
        // The two labels are complements by construction on both engines;
        // a Python-engine row records the baseline's and leaves the other
        // implied.
        var conditionWas = fields["conditionWas"] as? String
        if conditionWas == nil, kind != .unreadable {
            switch baselineWas {
            case "A": conditionWas = "B"
            case "B": conditionWas = "A"
            default: break
            }
        }
        let reason: String?
        switch kind {
        case .verdict:
            reason = nil
        case .noncompliant:
            reason = fields["noncomplianceReason"] as? String
        case .unreadable:
            reason = StudyReviewJSON.unreadableReason(
                line,
                otherwise: "This line records neither a verdict nor a "
                    + "noncompliant answer.")
        }
        return StudyJudgmentRow(
            record: record,
            kind: kind,
            condition: fields["condition"] as? String,
            promptID: StudyReviewJSON.identifier(fields["promptID"]),
            // Rows written before pairs were keyed by sample carried one
            // `seed`; it still tells such rows apart.
            sampleIndex: StudyReviewJSON.int(fields["sampleIndex"])
                ?? StudyReviewJSON.int(fields["seed"]),
            judge: fields["judge"] as? String,
            judgeModel: fields["judgeModel"] as? String,
            prompt: fields["prompt"] as? String,
            baselineWas: baselineWas,
            conditionWas: conditionWas,
            winner: verdict["winner"] as? String,
            result: kind == .verdict ? unblinded(fields) : nil,
            confidence: StudyReviewJSON.double(verdict["confidence"])
                ?? StudyReviewJSON.double(fields["confidence"]),
            briefReason: verdict["brief_reason"] as? String,
            reason: reason,
            answerCutOff: kind == .verdict
                && (StudyReviewJSON.bool(fields["verdictSalvaged"]) == true
                    || StudyReviewJSON.bool(verdict["reasoningTruncated"]) == true),
            aScores: StudyReviewJSON.values(verdict["a_scores"]),
            bScores: StudyReviewJSON.values(verdict["b_scores"]),
            structuredFields: StudyReviewJSON.values(verdict["structured_fields"]),
            rawJSON: StudyReviewJSON.pretty(line))
    }
}
