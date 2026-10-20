import CryptoKit
import Foundation
import SteeringKit

/// Import a concept from two row-aligned chat-format JSONL files.
///
/// A concept's stimulus files are one `{"text": …}` per line
/// (`prompts/concepts/<name>/{positive,negative}.jsonl`, read by
/// `StimulusSet`). Much published contrast data comes as CHAT rows instead —
/// `{"messages": [{"role": "user", …}, {"role": "assistant", …}]}` — in two
/// files, one per pole, whose rows answer the same prompts in the same order.
/// This turns such a pair into a concept. It is the line-for-line twin of
/// `Server/steerlab_server/experiment/chat_concept_import.py`: the same input
/// and seed write the same bytes on both engines, and refuse with the same
/// code and sentence, pinned by the cross-engine fixture
/// `Tests/Fixtures/cross-engine/chat-concept-import.json`.
///
/// The rule (also written into every `provenance.json`):
///
/// - Row *i* of the positive file pairs with row *i* of the negative file.
///   Rows are the non-blank lines, counted from 1; a leading UTF-8 byte-order
///   mark is ignored.
/// - A row holds exactly one `user` message followed by one `assistant`
///   message; `system` messages are allowed and ignored, any other role is
///   refused. A row that is not UTF-8 or JSON is refused by file and line.
/// - A pair is DROPPED, not refused, when its user prompts differ, when its
///   two replies are identical, or when either reply is empty (only spaces,
///   tabs, and line breaks). Each drop is counted and its row recorded.
/// - Kept replies are written verbatim to the same line of `positive.jsonl`
///   and `negative.jsonl`. Optionally N kept pairs are held out instead as
///   `validation.jsonl` rows, drawn by the seeded draw both engines already
///   share for the neutral token bank
///   (`TokenBankDownsampler.selectedIndices`: SplitMix64, partial
///   Fisher–Yates).
///
/// **Why its own JSON reader.** Parity is decided as much by which rows are
/// refused as by what is written, and Foundation's reader accepts trailing
/// commas and a byte-order mark mid-file where Python's refuses them (Python's
/// in turn accepts lone surrogates, `NaN`, and overflowing numbers). Both
/// engines apply one narrower rule: RFC 8259 with no lone surrogates, no
/// non-finite numbers, nesting at most `maximumDepth`, the LAST of a repeated
/// key, and text compared scalar by scalar — never by Unicode equivalence,
/// which Swift's `==` would apply and Python's would not.
///
/// The importer reads files a researcher names and writes into their
/// workspace. It bundles and downloads nothing.
public enum ChatConceptImport {

    /// Stamped into `provenance.json`. A change to what this writes for the
    /// same input is a new identifier.
    public static let importerID = "chat-pairs-v1"
    /// How held-out pairs are drawn — the neutral token bank's shared draw.
    public static let drawID = "splitmix64-partial-fisher-yates"
    /// The largest seed: every integer up to it is exact in a JSON reader that
    /// holds numbers as doubles, so the recorded seed is the seed used.
    public static let maximumSeed: UInt64 = (1 << 53) - 1
    public static let maximumNameLength = 64
    /// Arrays and objects nested deeper than this make a row unreadable.
    static let maximumDepth = 64

    static let provenanceArtifact = "steerlab-concept-import"
    static let conceptArtifact = "steerlab-concept"

    public static let rule =
        "Row i of the positive file is paired with row i of the negative "
        + "file, counting non-blank lines from 1. A pair is dropped when its "
        + "two user prompts differ, when its two assistant replies are "
        + "identical, or when either reply is empty. Each kept pair's assistant "
        + "replies are written to the same line of positive.jsonl and "
        + "negative.jsonl; held-out pairs go to validation.jsonl instead, the "
        + "positive reply then the negative one."

    public static let rowShape =
        #"each line must be one JSON object with a "messages" list holding "#
        + #"one {"role": "user", "content": "…"} message and then one "#
        + #"{"role": "assistant", "content": "…"} message; "system" "#
        + "messages are allowed and ignored"

    // MARK: Refusal

    /// A typed refusal: `code` and `state` for the envelope, a plain reason,
    /// and a repair the reader can carry out. `state` is `.refused` (65) for
    /// data the importer declines, `.notFound` (66) for a missing source file,
    /// and `.blocked` (64) for a malformed request.
    public struct Refusal: Error, Sendable, Equatable, CustomStringConvertible {
        public let code: String
        public let reason: String
        public let repairAction: String
        public let state: SteerLabCLIState

        public init(
            code: String, reason: String, repairAction: String,
            state: SteerLabCLIState = .refused
        ) {
            self.code = code
            self.reason = reason
            self.repairAction = repairAction
            self.state = state
        }

        public var description: String { "\(reason) — \(repairAction)" }

        /// The refusal in the app's words. The command line's reason and
        /// repair name its flags; the app's card has a hold-out count and a
        /// seed instead, so the few refusals that name a flag are said in
        /// those terms, and every other refusal reads as it does everywhere.
        public var appDescription: String {
            if code == "validationTooLarge" {
                let usable = reason.split(separator: " ")
                    .compactMap { Int($0) }.last
                return "Holding out that many pairs leaves none to extract from"
                    + (usable.map { ": \($0) pairs are usable. Hold out fewer than \($0), or none." }
                        ?? ". Hold out fewer pairs, or none.")
            }
            guard code == "usage", reason.hasPrefix("--") else { return description }
            if reason.hasPrefix("--seed expects") {
                return "The seed must be a whole number from 0 to \(maximumSeed)."
            }
            if reason.hasPrefix("--validation-pairs expects") {
                return "The number of pairs to hold out must be a whole number."
            }
            if reason.hasPrefix("--validation-pairs needs --seed") {
                return "Holding out pairs needs a seed, so the same pairs can be drawn again."
            }
            if reason.hasPrefix("--seed draws") {
                return "A seed is only used to hold out pairs; set a hold-out count, or clear the seed."
            }
            return description
        }
    }

    // MARK: Values written

    /// One file the import writes, in write order.
    public struct File: Sendable, Equatable {
        public let name: String
        public let data: Data
    }

    /// A small JSON tree rendered exactly as Python's
    /// `json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)`.
    public indirect enum Value: Sendable, Equatable {
        case object([String: Value])
        case array([Value])
        case string(String)
        case integer(Int)
        case bool(Bool)
        case null

        public func rendered() -> String {
            var out = ""
            render(into: &out, level: 0)
            return out
        }

        private func render(into out: inout String, level: Int) {
            switch self {
            case .object(let members):
                guard !members.isEmpty else {
                    out += "{}"
                    return
                }
                out += "{\n"
                let keys = members.keys.sorted {
                    $0.unicodeScalars.map(\.value)
                        .lexicographicallyPrecedes($1.unicodeScalars.map(\.value))
                }
                for (index, key) in keys.enumerated() {
                    out += String(repeating: "  ", count: level + 1)
                    out += ChatConceptImport.jsonString(key) + ": "
                    members[key]!.render(into: &out, level: level + 1)
                    out += index < keys.count - 1 ? ",\n" : "\n"
                }
                out += String(repeating: "  ", count: level) + "}"
            case .array(let items):
                guard !items.isEmpty else {
                    out += "[]"
                    return
                }
                out += "[\n"
                for (index, item) in items.enumerated() {
                    out += String(repeating: "  ", count: level + 1)
                    item.render(into: &out, level: level + 1)
                    out += index < items.count - 1 ? ",\n" : "\n"
                }
                out += String(repeating: "  ", count: level) + "]"
            case .string(let text): out += ChatConceptImport.jsonString(text)
            case .integer(let number): out += String(number)
            case .bool(let flag): out += flag ? "true" : "false"
            case .null: out += "null"
            }
        }

        /// The same tree as an envelope payload value.
        public var jsonValue: JSONValue {
            switch self {
            case .object(let members): .object(members.mapValues(\.jsonValue))
            case .array(let items): .array(items.map(\.jsonValue))
            case .string(let text): .string(text)
            case .integer(let number): .number(Double(number))
            case .bool(let flag): .bool(flag)
            case .null: .null
            }
        }
    }

    /// Everything an import writes, computed and checked, nothing written.
    public struct Prepared: Sendable {
        public let name: String
        public let files: [File]
        public let provenance: Value
        public let extractionPairs: Int
        public let validationPairs: Int
        public let rows: Int
        public let droppedPromptMismatch: Int
        public let droppedIdenticalReplies: Int
        public let droppedEmptyReply: Int
        /// SHA-256 over positive then negative bytes — what `StimulusSet.hash`
        /// and `experiment attach` will pin.
        public let stimulusSetHash: String
        /// SHA-256 of `validation.jsonl`, when pairs were held out.
        public let validationHash: String?

        public var droppedPairs: Int {
            droppedPromptMismatch + droppedIdenticalReplies + droppedEmptyReply
        }
    }

    /// A written import.
    public struct Outcome: Sendable {
        public let prepared: Prepared
        /// The concept directory, absolute.
        public let directory: URL
        /// `prompts/concepts/<name>` — workspace-relative.
        public let relativeDirectory: String

        public var writtenFiles: [String] {
            prepared.files.map { "\(relativeDirectory)/\($0.name)" }
        }

        /// The envelope's `result` — the same keys on both clients.
        public var payload: [String: JSONValue] {
            [
                "concept": .string(prepared.name),
                "directory": .string(relativeDirectory),
                "files": .array(writtenFiles.map(JSONValue.string)),
                "provenance": prepared.provenance.jsonValue,
                "stimulusSetHash": .string(prepared.stimulusSetHash),
                "validationHash": prepared.validationHash.map(JSONValue.string) ?? .null,
            ]
        }

        /// What a person at a terminal reads. The first line is the
        /// envelope's message.
        public var summaryLines: [String] {
            let p = prepared
            var lines = [
                "imported '\(p.name)' from chat pairs: \(p.extractionPairs) for "
                    + "extraction, \(p.validationPairs) held out for validation, "
                    + "\(p.droppedPairs) dropped",
                "wrote \(relativeDirectory)/ ("
                    + p.files.map(\.name).joined(separator: ", ") + ")",
            ]
            if p.droppedPairs > 0 {
                lines.append(
                    "dropped \(p.droppedPairs) of \(p.rows) pairs: "
                        + "\(p.droppedPromptMismatch) with different user prompts, "
                        + "\(p.droppedIdenticalReplies) with identical replies, "
                        + "\(p.droppedEmptyReply) with an empty reply (their rows "
                        + "are listed in provenance.json)")
            }
            if p.validationPairs == 0 {
                lines.append(
                    "no validation.jsonl was written: author "
                        + "\(relativeDirectory)/validation.jsonl before freezing a "
                        + "study that uses this concept, or import again under a "
                        + "new name with --validation-pairs and --seed")
            }
            return lines
        }
    }

    // MARK: Formatting shared with the Python twin

    /// UTC, whole seconds: the `createdAt` shape every concept.json has.
    public static func timestamp(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = TimeZone(identifier: "UTC")
        return formatter.string(
            from: Date(timeIntervalSince1970: date.timeIntervalSince1970.rounded(.down)))
    }

    /// One JSON string literal, non-ASCII kept as UTF-8: `"` and `\`
    /// escaped, `\b \f \n \r \t` named, every other control character as a
    /// lower-case `\u00xx` — Python's `json.dumps(text, ensure_ascii=False)` —
    /// and U+0085, U+2028, and U+2029 escaped too, so every row stays one line
    /// for a reader that splits on them (Python's `str.splitlines` does).
    public static func jsonString(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case "\u{08}": out += "\\b"
            case "\u{0C}": out += "\\f"
            default:
                if scalar.value < 0x20 || [0x85, 0x2028, 0x2029].contains(scalar.value) {
                    let hex = String(scalar.value, radix: 16)
                    out += "\\u" + String(repeating: "0", count: 4 - hex.count) + hex
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }

    static func textRow(_ text: String) -> String {
        #"{"text": "# + jsonString(text) + "}\n"
    }

    static func validationRow(_ text: String, expresses: Bool) -> String {
        #"{"text": "# + jsonString(text) + #", "expresses": "#
            + (expresses ? "true" : "false") + "}\n"
    }

    /// The concept primitive, in the exact shape the app writes it.
    static func conceptDocument(name: String, createdAt: String) -> String {
        "{\n"
            + "  \"artifactType\" : \"\(conceptArtifact)\",\n"
            + "  \"createdAt\" : \"\(createdAt)\",\n"
            + "  \"name\" : \"\(name)\",\n"
            + "  \"schemaVersion\" : 1\n"
            + "}"
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: The request

    /// A usable spelling of `raw`, or "" when nothing of it survives.
    public static func suggestedName(_ raw: String) -> String {
        var kept = ""
        for scalar in raw.unicodeScalars {
            switch scalar {
            case "A"..."Z": kept.unicodeScalars.append(Unicode.Scalar(scalar.value + 32)!)
            case "a"..."z", "0"..."9": kept.unicodeScalars.append(scalar)
            case " ", "_", "-", ".": kept += "-"
            default: break
            }
        }
        var collapsed = ""
        for character in kept where !(character == "-" && collapsed.hasSuffix("-")) {
            collapsed.append(character)
        }
        func trimmed(_ text: Substring) -> String {
            String(text.drop { $0 == "-" }.reversed().drop { $0 == "-" }.reversed())
        }
        return trimmed(Substring(trimmed(Substring(collapsed)).prefix(maximumNameLength)))
    }

    /// Lowercase letters, digits, and single hyphens, at most 64 characters.
    public static func isUsableName(_ raw: String) -> Bool {
        let scalars = Array(raw.unicodeScalars)
        guard !scalars.isEmpty, scalars.count <= maximumNameLength,
            scalars.first != "-", scalars.last != "-"
        else { return false }
        var previous: Unicode.Scalar = " "
        for scalar in scalars {
            switch scalar {
            case "a"..."z", "0"..."9": break
            case "-" where previous != "-": break
            default: return false
            }
            previous = scalar
        }
        return true
    }

    public static func validatedName(_ raw: String) throws -> String {
        if isUsableName(raw) { return raw }
        let suggestion = suggestedName(raw)
        var repair = "use lowercase letters, digits, and single hyphens, at most "
            + "\(maximumNameLength) characters"
        if !suggestion.isEmpty, suggestion != raw { repair += ", e.g. \(suggestion)" }
        throw Refusal(
            code: "usage", reason: "'\(raw)' is not a usable concept name",
            repairAction: repair, state: .blocked)
    }

    static func isDigits(_ text: String) -> Bool {
        !text.isEmpty && text.unicodeScalars.allSatisfy { ("0"..."9").contains($0) }
    }

    /// `--validation-pairs`: a whole number.
    public static func parseCount(_ text: String) throws -> Int {
        guard isDigits(text), text.count <= 9, let value = Int(text) else {
            throw Refusal(
                code: "usage",
                reason: "--validation-pairs expects a whole number, got '\(text)'",
                repairAction: "--validation-pairs <n>, the number of pairs to hold out",
                state: .blocked)
        }
        return value
    }

    /// `--seed`: a whole number from 0 to `maximumSeed`.
    public static func parseSeed(_ text: String) throws -> UInt64 {
        guard isDigits(text), text.count <= 16, let value = UInt64(text),
            value <= maximumSeed
        else {
            throw Refusal(
                code: "usage",
                reason: "--seed expects a whole number from 0 to \(maximumSeed), "
                    + "got '\(text)'",
                repairAction: "--seed <n>, a whole number from 0 to \(maximumSeed)",
                state: .blocked)
        }
        return value
    }

    /// A held-out draw needs its seed, and a seed needs a draw.
    public static func checkHoldout(validationPairs: Int, seed: UInt64?) throws {
        if validationPairs > 0, seed == nil {
            throw Refusal(
                code: "usage",
                reason: "--validation-pairs needs --seed: the held-out pairs are "
                    + "drawn with a seed that is recorded in provenance.json",
                repairAction: "add --seed <n>, a whole number from 0 to \(maximumSeed)",
                state: .blocked)
        }
        if validationPairs == 0, seed != nil {
            throw Refusal(
                code: "usage",
                reason: "--seed draws the held-out pairs, and no pairs are held out",
                repairAction: "add --validation-pairs <n>, or leave out --seed",
                state: .blocked)
        }
    }

    // MARK: Reading rows

    struct Row {
        let prompt: String
        let reply: String
    }

    static func unreadable(_ file: String, _ line: Int, _ problem: String) -> Refusal {
        Refusal(
            code: "chatRowUnreadable", reason: "\(file) line \(line) \(problem)",
            repairAction: "fix \(file) line \(line) (\(rowShape)), then import again")
    }

    static func roles(_ file: String, _ line: Int, _ reason: String) -> Refusal {
        Refusal(
            code: "chatRowRoles", reason: reason,
            repairAction: "fix \(file) line \(line) (\(rowShape)), then import again")
    }

    static func same(_ a: String, _ b: String) -> Bool {
        a.unicodeScalars.elementsEqual(b.unicodeScalars)
    }

    /// The rows, and how many carried a system message; refuses the first
    /// unreadable or wrongly shaped row.
    static func readRows(_ data: Data, file: String) throws -> (rows: [Row], withSystem: Int) {
        var bytes = [UInt8](data)
        if bytes.starts(with: [0xEF, 0xBB, 0xBF]) { bytes.removeFirst(3) }
        var rows: [Row] = []
        var withSystem = 0
        let lines = bytes.split(separator: 0x0A, omittingEmptySubsequences: false)
        for (offset, slice) in lines.enumerated() {
            let line = offset + 1
            if slice.allSatisfy({ $0 == 0x20 || $0 == 0x09 || $0 == 0x0D }) { continue }
            guard String(validating: slice, as: UTF8.self) != nil else {
                throw unreadable(file, line, "is not UTF-8 text")
            }
            let parsed: Parsed
            do {
                var parser = StrictJSONParser(Array(slice))
                parsed = try parser.document()
            } catch {
                throw unreadable(file, line, "is not valid JSON")
            }
            guard case .object = parsed else {
                throw unreadable(file, line, "is not a JSON object")
            }
            guard case .array(let messages)? = parsed.member("messages") else {
                throw unreadable(file, line, #"has no "messages" list"#)
            }
            var prompt = ""
            var reply = ""
            var users = 0
            var assistants = 0
            var userAt = 0
            var assistantAt = 0
            var hasSystem = false
            for (index, message) in messages.enumerated() {
                let position = index + 1
                guard case .object = message,
                    case .string(let role)? = message.member("role"),
                    case .string(let content)? = message.member("content")
                else {
                    throw Refusal(
                        code: "chatRowUnreadable",
                        reason: "\(file) line \(line): message \(position) needs a "
                            + #""role" string and a "content" string"#,
                        repairAction: "fix \(file) line \(line) (\(rowShape)), then import again")
                }
                if same(role, "user") {
                    users += 1
                    userAt = position
                    prompt = content
                } else if same(role, "assistant") {
                    assistants += 1
                    assistantAt = position
                    reply = content
                } else if same(role, "system") {
                    hasSystem = true
                } else {
                    throw roles(
                        file, line,
                        "\(file) line \(line): message \(position) has role "
                            + "\(jsonString(role)); only \"system\", \"user\", and "
                            + "\"assistant\" are read")
                }
            }
            guard users == 1, assistants == 1 else {
                throw roles(
                    file, line,
                    "\(file) line \(line) has \(users) user and \(assistants) "
                        + "assistant messages; each row needs exactly one of each")
            }
            guard assistantAt >= userAt else {
                throw roles(
                    file, line,
                    "\(file) line \(line) puts its assistant message before its "
                        + "user message")
            }
            rows.append(Row(prompt: prompt, reply: reply))
            if hasSystem { withSystem += 1 }
        }
        return (rows, withSystem)
    }

    static func isEmptyReply(_ reply: String) -> Bool {
        reply.unicodeScalars.allSatisfy { [" ", "\t", "\n", "\r"].contains($0) }
    }

    // MARK: The import

    /// One source file as the import reads it. `file` is the name recorded
    /// in provenance and named in refusals — the base name, never a path.
    public struct Source: Sendable {
        public let file: String
        public let data: Data

        public init(file: String, data: Data) {
            self.file = file
            self.data = data
        }
    }

    /// Everything the import writes, computed and checked; nothing written.
    public static func prepare(
        name rawName: String, positive: Source, negative: Source,
        validationPairs: Int = 0, seed: UInt64? = nil, createdAt: String
    ) throws -> Prepared {
        let name = try validatedName(rawName)
        try checkHoldout(validationPairs: validationPairs, seed: seed)
        let (positiveRows, positiveSystem) = try readRows(positive.data, file: positive.file)
        let (negativeRows, negativeSystem) = try readRows(negative.data, file: negative.file)
        guard positiveRows.count == negativeRows.count else {
            throw Refusal(
                code: "rowCountMismatch",
                reason: "\(positive.file) has \(positiveRows.count) rows and "
                    + "\(negative.file) has \(negativeRows.count); the two files "
                    + "must pair up row by row",
                repairAction: "import two files that answer the same prompts in "
                    + "the same order, one pair per row: check that neither file "
                    + "was cut short and that both come from the same release")
        }

        var promptMismatch: [Int] = []
        var identicalReplies: [Int] = []
        var emptyReply: [Int] = []
        var kept: [(row: Int, positive: String, negative: String)] = []
        for (index, (pos, neg)) in zip(positiveRows, negativeRows).enumerated() {
            let row = index + 1
            if !same(pos.prompt, neg.prompt) {
                promptMismatch.append(row)
            } else if same(pos.reply, neg.reply) {
                identicalReplies.append(row)
            } else if isEmptyReply(pos.reply) || isEmptyReply(neg.reply) {
                emptyReply.append(row)
            } else {
                kept.append((row, pos.reply, neg.reply))
            }
        }
        let droppedCount = promptMismatch.count + identicalReplies.count + emptyReply.count

        guard !kept.isEmpty else {
            throw Refusal(
                code: "noUsablePairs",
                reason: "no usable pairs: \(positiveRows.count) rows read, "
                    + "\(promptMismatch.count) with different user prompts, "
                    + "\(identicalReplies.count) with identical replies, "
                    + "\(emptyReply.count) with an empty reply",
                repairAction: "check that the two files are the positive and "
                    + "negative halves of one dataset, in the same row order, "
                    + "with the concept-present replies in the positive file")
        }
        guard validationPairs < kept.count else {
            throw Refusal(
                code: "validationTooLarge",
                reason: "holding out \(validationPairs) pairs leaves none for "
                    + "extraction: \(kept.count) pairs are usable",
                repairAction: "set --validation-pairs below \(kept.count), or leave it out")
        }

        let held: Set<Int> =
            validationPairs > 0
            ? Set(
                TokenBankDownsampler.selectedIndices(
                    count: kept.count, cap: validationPairs, seed: seed ?? 0))
            : []
        let training = kept.enumerated().filter { !held.contains($0.offset) }.map(\.element)
        let heldOut = kept.enumerated().filter { held.contains($0.offset) }.map(\.element)

        let positiveBytes = Data(training.map { textRow($0.positive) }.joined().utf8)
        let negativeBytes = Data(training.map { textRow($0.negative) }.joined().utf8)
        var files = [
            File(
                name: "concept.json",
                data: Data(conceptDocument(name: name, createdAt: createdAt).utf8)),
            File(name: "positive.jsonl", data: positiveBytes),
            File(name: "negative.jsonl", data: negativeBytes),
        ]
        var validationHash: String?
        if !heldOut.isEmpty {
            let validationBytes = Data(
                heldOut.map {
                    validationRow($0.positive, expresses: true)
                        + validationRow($0.negative, expresses: false)
                }.joined().utf8)
            files.append(File(name: "validation.jsonl", data: validationBytes))
            validationHash = sha256(validationBytes)
        }

        func source(_ source: Source, rows: Int, system: Int) -> Value {
            .object([
                "bytes": .integer(source.data.count),
                "file": .string(source.file),
                "rows": .integer(rows),
                "sha256": .string(sha256(source.data)),
                "systemMessages": .integer(system),
            ])
        }
        let provenance = Value.object([
            "artifactType": .string(provenanceArtifact),
            "concept": .string(name),
            "dropped": .object([
                "emptyReply": .array(emptyReply.map(Value.integer)),
                "identicalReplies": .array(identicalReplies.map(Value.integer)),
                "promptMismatch": .array(promptMismatch.map(Value.integer)),
            ]),
            "importedAt": .string(createdAt),
            "importer": .string(importerID),
            "pairs": .object([
                "dropped": .integer(droppedCount),
                "extraction": .integer(training.count),
                "kept": .integer(kept.count),
                "rows": .integer(positiveRows.count),
                "validation": .integer(heldOut.count),
            ]),
            "rule": .string(rule),
            "schemaVersion": .integer(1),
            "sources": .object([
                "negative": source(negative, rows: negativeRows.count, system: negativeSystem),
                "positive": source(positive, rows: positiveRows.count, system: positiveSystem),
            ]),
            "validation": heldOut.isEmpty
                ? .null
                : .object([
                    "draw": .string(drawID),
                    "pairs": .integer(heldOut.count),
                    "rows": .array(heldOut.map { Value.integer($0.row) }),
                    "seed": .integer(Int(seed ?? 0)),
                ]),
        ])
        files.append(File(name: "provenance.json", data: Data((provenance.rendered() + "\n").utf8)))

        return Prepared(
            name: name, files: files, provenance: provenance,
            extractionPairs: training.count, validationPairs: heldOut.count,
            rows: positiveRows.count,
            droppedPromptMismatch: promptMismatch.count,
            droppedIdenticalReplies: identicalReplies.count,
            droppedEmptyReply: emptyReply.count,
            stimulusSetHash: sha256(positiveBytes + negativeBytes),
            validationHash: validationHash)
    }

    static func exists(_ name: String) -> Refusal {
        Refusal(
            code: "conceptExists", reason: "prompts/concepts/\(name)/ already exists",
            repairAction: "choose a concept name that is not already in "
                + "prompts/concepts/ — the importer never adds to or replaces an "
                + "existing concept")
    }

    static func read(_ url: URL, side: String) throws -> Data {
        let path = url.standardizedFileURL.path
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Refusal(
                code: "notFound", reason: "no file at \(path)",
                repairAction: "check the path to the \(side) chat file",
                state: .notFound)
        }
        do {
            return try Data(contentsOf: url)
        } catch {
            throw Refusal(
                code: "chatFileUnreadable", reason: "could not read \(path)",
                repairAction: "check that the \(side) path names a readable file")
        }
    }

    /// Read, check, and write one concept under `root`. Refuses before
    /// writing anything; the files land together by one rename of a staging
    /// directory beside the destination.
    @discardableResult
    public static func importConcept(
        name rawName: String, positive: URL, negative: URL,
        validationPairs: Int = 0, seed: UInt64? = nil,
        root: URL, now: Date = Date()
    ) throws -> Outcome {
        let name = try validatedName(rawName)
        try checkHoldout(validationPairs: validationPairs, seed: seed)
        let fileManager = FileManager.default
        let concepts = VectorCatalog.conceptsDirectory(root: root)
        let destination = concepts.appending(component: name)
        // Anything at the name counts, a dangling link included
        // (`attributesOfItem` does not follow links; Python's `lexists`).
        func present() -> Bool {
            (try? fileManager.attributesOfItem(atPath: destination.path)) != nil
        }
        guard !present() else { throw exists(name) }
        let positiveData = try read(positive, side: "positive")
        let negativeData = try read(negative, side: "negative")
        let prepared = try prepare(
            name: name,
            positive: Source(file: positive.lastPathComponent, data: positiveData),
            negative: Source(file: negative.lastPathComponent, data: negativeData),
            validationPairs: validationPairs, seed: seed, createdAt: timestamp(now))

        try fileManager.createDirectory(at: concepts, withIntermediateDirectories: true)
        let staging = concepts.appending(
            component: ".steerlab-chat-import-\(UUID().uuidString.lowercased())")
        try fileManager.createDirectory(at: staging, withIntermediateDirectories: false)
        defer {
            if fileManager.fileExists(atPath: staging.path) {
                try? fileManager.removeItem(at: staging)
            }
        }
        for file in prepared.files {
            try file.data.write(to: staging.appending(component: file.name))
        }
        guard !present() else { throw exists(name) }
        do {
            try fileManager.moveItem(at: staging, to: destination)
        } catch {
            if present() { throw exists(name) }
            throw error
        }
        return Outcome(
            prepared: prepared, directory: destination,
            relativeDirectory: "prompts/concepts/\(name)")
    }
}

// MARK: - The strict reader

extension ChatConceptImport {

    /// What a row parses to. Object members keep their order and are looked
    /// up by SCALAR equality, last occurrence winning — Python's `dict` rule.
    indirect enum Parsed {
        case object([(key: String, value: Parsed)])
        case array([Parsed])
        case string(String)
        case number
        case bool(Bool)
        case null

        func member(_ name: String) -> Parsed? {
            guard case .object(let members) = self else { return nil }
            return members.last { ChatConceptImport.same($0.key, name) }?.value
        }
    }

    /// RFC 8259 over one line's UTF-8 bytes, with the narrowing both engines
    /// share (see the type comment).
    struct StrictJSONParser {
        struct Invalid: Error {}

        let bytes: [UInt8]
        var index = 0
        var depth = 0

        init(_ bytes: [UInt8]) { self.bytes = bytes }

        mutating func document() throws -> Parsed {
            skipWhitespace()
            let value = try parseValue()
            skipWhitespace()
            guard index == bytes.count else { throw Invalid() }
            return value
        }

        private var current: UInt8? { index < bytes.count ? bytes[index] : nil }

        private mutating func skipWhitespace() {
            while let byte = current, byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D {
                index += 1
            }
        }

        private mutating func expect(_ byte: UInt8) throws {
            guard current == byte else { throw Invalid() }
            index += 1
        }

        private mutating func parseValue() throws -> Parsed {
            guard let byte = current else { throw Invalid() }
            switch byte {
            case UInt8(ascii: "{"): return try parseObject()
            case UInt8(ascii: "["): return try parseArray()
            case UInt8(ascii: "\""): return .string(try parseString())
            case UInt8(ascii: "t"):
                try literal("true")
                return .bool(true)
            case UInt8(ascii: "f"):
                try literal("false")
                return .bool(false)
            case UInt8(ascii: "n"):
                try literal("null")
                return .null
            case UInt8(ascii: "-"), UInt8(ascii: "0")...UInt8(ascii: "9"):
                try parseNumber()
                return .number
            default: throw Invalid()
            }
        }

        private mutating func literal(_ word: String) throws {
            for byte in word.utf8 { try expect(byte) }
        }

        private mutating func enter() throws {
            depth += 1
            guard depth <= ChatConceptImport.maximumDepth else { throw Invalid() }
        }

        private mutating func parseObject() throws -> Parsed {
            try enter()
            defer { depth -= 1 }
            index += 1
            var members: [(key: String, value: Parsed)] = []
            skipWhitespace()
            if current == UInt8(ascii: "}") {
                index += 1
                return .object(members)
            }
            while true {
                skipWhitespace()
                guard current == UInt8(ascii: "\"") else { throw Invalid() }
                let key = try parseString()
                skipWhitespace()
                try expect(UInt8(ascii: ":"))
                skipWhitespace()
                members.append((key, try parseValue()))
                skipWhitespace()
                if current == UInt8(ascii: ",") {
                    index += 1
                    continue
                }
                try expect(UInt8(ascii: "}"))
                return .object(members)
            }
        }

        private mutating func parseArray() throws -> Parsed {
            try enter()
            defer { depth -= 1 }
            index += 1
            var items: [Parsed] = []
            skipWhitespace()
            if current == UInt8(ascii: "]") {
                index += 1
                return .array(items)
            }
            while true {
                skipWhitespace()
                items.append(try parseValue())
                skipWhitespace()
                if current == UInt8(ascii: ",") {
                    index += 1
                    continue
                }
                try expect(UInt8(ascii: "]"))
                return .array(items)
            }
        }

        private mutating func hex4() throws -> UInt32 {
            guard index + 4 <= bytes.count else { throw Invalid() }
            var value: UInt32 = 0
            for byte in bytes[index ..< index + 4] {
                let digit: UInt8
                switch byte {
                case UInt8(ascii: "0")...UInt8(ascii: "9"): digit = byte - UInt8(ascii: "0")
                case UInt8(ascii: "a")...UInt8(ascii: "f"): digit = byte - UInt8(ascii: "a") + 10
                case UInt8(ascii: "A")...UInt8(ascii: "F"): digit = byte - UInt8(ascii: "A") + 10
                default: throw Invalid()
                }
                value = value * 16 + UInt32(digit)
            }
            index += 4
            return value
        }

        private mutating func parseString() throws -> String {
            index += 1
            var out: [UInt8] = []
            while true {
                guard let byte = current else { throw Invalid() }
                if byte == UInt8(ascii: "\"") {
                    index += 1
                    return String(decoding: out, as: UTF8.self)
                }
                guard byte >= 0x20 else { throw Invalid() }
                guard byte == UInt8(ascii: "\\") else {
                    out.append(byte)
                    index += 1
                    continue
                }
                index += 1
                guard let escape = current else { throw Invalid() }
                index += 1
                switch escape {
                case UInt8(ascii: "\""): out.append(0x22)
                case UInt8(ascii: "\\"): out.append(0x5C)
                case UInt8(ascii: "/"): out.append(0x2F)
                case UInt8(ascii: "b"): out.append(0x08)
                case UInt8(ascii: "f"): out.append(0x0C)
                case UInt8(ascii: "n"): out.append(0x0A)
                case UInt8(ascii: "r"): out.append(0x0D)
                case UInt8(ascii: "t"): out.append(0x09)
                case UInt8(ascii: "u"):
                    var code = try hex4()
                    if (0xD800...0xDBFF).contains(code) {
                        guard current == UInt8(ascii: "\\"), index + 1 < bytes.count,
                            bytes[index + 1] == UInt8(ascii: "u")
                        else { throw Invalid() }
                        index += 2
                        let low = try hex4()
                        guard (0xDC00...0xDFFF).contains(low) else { throw Invalid() }
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                    } else if (0xDC00...0xDFFF).contains(code) {
                        throw Invalid()
                    }
                    guard let scalar = Unicode.Scalar(code) else { throw Invalid() }
                    UTF8.encode(scalar) { out.append($0) }
                default: throw Invalid()
                }
            }
        }

        private func isDigit(_ byte: UInt8?) -> Bool {
            guard let byte else { return false }
            return byte >= UInt8(ascii: "0") && byte <= UInt8(ascii: "9")
        }

        /// The JSON number grammar. Integers are never converted (they are
        /// validated and unused); a fraction or exponent must be finite.
        private mutating func parseNumber() throws {
            let start = index
            if current == UInt8(ascii: "-") { index += 1 }
            if current == UInt8(ascii: "0") {
                index += 1
            } else if isDigit(current) {
                while isDigit(current) { index += 1 }
            } else {
                throw Invalid()
            }
            var real = false
            if current == UInt8(ascii: ".") {
                index += 1
                guard isDigit(current) else { throw Invalid() }
                while isDigit(current) { index += 1 }
                real = true
            }
            if current == UInt8(ascii: "e") || current == UInt8(ascii: "E") {
                index += 1
                if current == UInt8(ascii: "+") || current == UInt8(ascii: "-") { index += 1 }
                guard isDigit(current) else { throw Invalid() }
                while isDigit(current) { index += 1 }
                real = true
            }
            if real {
                let text = String(decoding: bytes[start ..< index], as: UTF8.self)
                guard let value = Double(text), value.isFinite else { throw Invalid() }
            }
        }
    }
}
