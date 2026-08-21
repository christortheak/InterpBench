import Foundation

/// Tabular import (Usability Plan Phase 3, item 13): a researcher's table —
/// a JSON array of objects OR a CSV with a header row — becomes a pinned
/// study input without the researcher ever hand-writing JSONL. The flow is
/// parse → column mapping (auto-guessed when names match, confirmed in a
/// small sheet) → convert to the CONSUMER's format → write to the standard
/// destination → pin through the existing validating pin path
/// (`ExperimentStore.pinTaskPrompts` / `pinHumanBaseline`). Nothing here
/// weakens a pin: the converted file goes through exactly the same
/// validation and hashing as a hand-authored one.
///
/// Target shapes mirror their consumers, never an invented schema:
/// - task prompts → the run loop's JSONL (`text`, optional `id` /
///   `options` / `target`), re-checked by `TaskPromptsImport.preview` (the
///   same rules the pin's parser enforces);
/// - human baseline → the analyze loader's CSV columns, reused from
///   `PinShapeValidation.humanBaselineRequiredColumns` (one column list in
///   the codebase, never duplicated).
///
/// Errors follow the plain-language rule: one sentence saying what is wrong
/// and what to do next. Existing files with different bytes are never
/// overwritten (the study-pack rule); identical bytes are idempotent.
public enum TabularImport {

    // MARK: - Errors

    public struct Problem: Error, CustomStringConvertible, Equatable {
        public let message: String
        public var description: String { message }

        public init(_ message: String) {
            self.message = message
        }
    }

    // MARK: - Values and tables

    /// One cell. CSV cells are always strings; JSON cells keep their type
    /// so `options` arrays and numeric `target`s survive conversion.
    public enum Value: Sendable, Equatable {
        case string(String)
        case number(Double)
        case bool(Bool)
        case array([Value])

        /// The cell rendered as text (whole numbers drop the ".0" so ids
        /// and month counts read naturally).
        public var text: String {
            switch self {
            case .string(let s):
                return s
            case .number(let n):
                if n.rounded() == n, abs(n) < 1e15 {
                    return String(Int64(n))
                }
                return String(n)
            case .bool(let b):
                return b ? "true" : "false"
            case .array(let items):
                return items.map(\.text).joined(separator: "|")
            }
        }

        public var isEmpty: Bool {
            switch self {
            case .string(let s):
                return s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            case .array(let items):
                return items.isEmpty
            case .number, .bool:
                return false
            }
        }
    }

    /// A parsed table: ordered header columns plus one dictionary per data
    /// row (absent/empty cells simply have no entry).
    public struct Table: Sendable, Equatable {
        public var columns: [String]
        public var rows: [[String: Value]]

        public init(columns: [String], rows: [[String: Value]]) {
            self.columns = columns
            self.rows = rows
        }
    }

    // MARK: - Target schemas

    public enum Target: String, Sendable, CaseIterable {
        case taskPrompts
        case humanBaseline

        public var title: String {
            switch self {
            case .taskPrompts: "task prompts"
            case .humanBaseline: "human baseline"
            }
        }
    }

    /// Fields the conversion REQUIRES a source column for. The
    /// human-baseline list is `PinShapeValidation`'s — the analyze loader's
    /// columns are the one authoritative list.
    public static func requiredFields(for target: Target) -> [String] {
        switch target {
        case .taskPrompts:
            return ["text"]
        case .humanBaseline:
            return PinShapeValidation.humanBaselineRequiredColumns
        }
    }

    public static func optionalFields(for target: Target) -> [String] {
        switch target {
        case .taskPrompts:
            return ["id", "options", "target"]
        case .humanBaseline:
            return []
        }
    }

    /// One-line meaning per target field, for the mapping sheet.
    public static func fieldHelp(_ field: String, target: Target) -> String {
        switch (target, field) {
        case (.taskPrompts, "text"):
            return "the prompt the model answers (required)"
        case (.taskPrompts, "id"):
            return "stable item id (optional; auto-numbered when absent)"
        case (.taskPrompts, "options"):
            return "categorical answer options — a JSON array, or one cell "
                + "delimited with | or ; (optional)"
        case (.taskPrompts, "target"):
            return "the option the concept should move probability toward "
                + "(optional)"
        case (.humanBaseline, "endpoint"):
            return "endpoint name analyze joins on (required)"
        case (.humanBaseline, "deltaHuman"):
            return "measured human effect for this endpoint (required)"
        case (.humanBaseline, "ciLower"):
            return "lower bound of the human effect's CI (required)"
        case (.humanBaseline, "ciUpper"):
            return "upper bound of the human effect's CI (required)"
        default:
            return field
        }
    }

    // MARK: - Parsing

    /// Parse a JSON array of objects or a CSV with a header row into a
    /// table. The file extension decides; unknown extensions sniff the
    /// first non-whitespace character ("[" = JSON).
    public static func parseTable(_ data: Data, fileName: String) throws -> Table {
        let ext = (fileName as NSString).pathExtension.lowercased()
        let looksJSON: Bool
        switch ext {
        case "json":
            looksJSON = true
        case "csv", "tsv":
            looksJSON = false
        default:
            let text = String(decoding: data, as: UTF8.self)
            looksJSON =
                text.first(where: { !$0.isWhitespace }) == "["
        }
        return looksJSON
            ? try parseJSONArray(data, fileName: fileName)
            : try parseCSV(data, fileName: fileName)
    }

    static func parseJSONArray(_ data: Data, fileName: String) throws -> Table {
        guard
            let parsed = try? JSONSerialization.jsonObject(with: data),
            let array = parsed as? [Any]
        else {
            throw Problem(
                "'\(fileName)' is not a JSON array of objects — export the "
                    + "table as [{…}, {…}] or as a CSV with a header row")
        }
        guard !array.isEmpty else {
            throw Problem("'\(fileName)' has no rows — nothing to import")
        }
        var columns: [String] = []
        var seen = Set<String>()
        var rows: [[String: Value]] = []
        for (index, element) in array.enumerated() {
            guard let object = element as? [String: Any] else {
                throw Problem(
                    "'\(fileName)' row \(index + 1) is not an object — every "
                        + "row must be {\"column\": value, …}")
            }
            var row: [String: Value] = [:]
            for key in object.keys.sorted() where seen.insert(key).inserted {
                columns.append(key)
            }
            for (key, raw) in object {
                guard let value = try value(raw, row: index + 1, column: key) else {
                    continue  // null → absent cell
                }
                row[key] = value
            }
            rows.append(row)
        }
        return Table(columns: columns, rows: rows)
    }

    private static func value(
        _ raw: Any, row: Int, column: String
    ) throws -> Value? {
        if raw is NSNull { return nil }
        if let number = raw as? NSNumber {
            // CFBoolean is an NSNumber; distinguish true booleans.
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                return .bool(number.boolValue)
            }
            return .number(number.doubleValue)
        }
        if let string = raw as? String { return .string(string) }
        if let array = raw as? [Any] {
            return .array(
                try array.compactMap { try value($0, row: row, column: column) })
        }
        throw Problem(
            "row \(row) column '\(column)' holds a nested object — flatten "
                + "it to plain values (or import a JSONL records file instead)")
    }

    /// RFC-4180-ish CSV: quoted fields may contain commas, newlines, and
    /// doubled quotes; the first record is the header. All cells parse as
    /// strings (the conversion checks numbers where its consumer needs
    /// them).
    static func parseCSV(_ data: Data, fileName: String) throws -> Table {
        var text = String(decoding: data, as: UTF8.self)
        if text.hasPrefix("\u{FEFF}") { text.removeFirst() }
        var records: [[String]] = []
        var record: [String] = []
        var field = ""
        var inQuotes = false
        var index = text.startIndex
        func endField() {
            record.append(field)
            field = ""
        }
        func endRecord() {
            endField()
            // Skip records that are entirely empty (blank lines).
            if !record.allSatisfy({
                $0.trimmingCharacters(in: .whitespaces).isEmpty
            }) {
                records.append(record)
            }
            record = []
        }
        while index < text.endIndex {
            let ch = text[index]
            if inQuotes {
                if ch == "\"" {
                    let next = text.index(after: index)
                    if next < text.endIndex, text[next] == "\"" {
                        field.append("\"")
                        index = next
                    } else {
                        inQuotes = false
                    }
                } else {
                    field.append(ch)
                }
            } else {
                switch ch {
                case "\"":
                    inQuotes = true
                case ",":
                    endField()
                case "\r":
                    // CRLF or lone CR ends the record; swallow a following LF.
                    let next = text.index(after: index)
                    if next < text.endIndex, text[next] == "\n" { index = next }
                    endRecord()
                case "\n":
                    endRecord()
                default:
                    field.append(ch)
                }
            }
            index = text.index(after: index)
        }
        if !field.isEmpty || !record.isEmpty { endRecord() }
        guard let header = records.first else {
            throw Problem(
                "'\(fileName)' is empty — a CSV import needs a header row "
                    + "naming the columns, then one row per item")
        }
        let columns = header.map { $0.trimmingCharacters(in: .whitespaces) }
        guard columns.contains(where: { !$0.isEmpty }) else {
            throw Problem(
                "'\(fileName)' has no header columns — the first row must "
                    + "name the columns")
        }
        let dataRecords = records.dropFirst()
        guard !dataRecords.isEmpty else {
            throw Problem(
                "'\(fileName)' has a header but no data rows — nothing to "
                    + "import")
        }
        var rows: [[String: Value]] = []
        for record in dataRecords {
            var row: [String: Value] = [:]
            for (offset, column) in columns.enumerated() where !column.isEmpty {
                // First occurrence wins for duplicate header names.
                guard row[column] == nil else { continue }
                let cell = offset < record.count ? record[offset] : ""
                if !cell.trimmingCharacters(in: .whitespaces).isEmpty {
                    row[column] = .string(cell)
                }
            }
            rows.append(row)
        }
        return Table(columns: columns.filter { !$0.isEmpty }, rows: rows)
    }

    // MARK: - Mapping

    /// Case- and punctuation-insensitive auto-guess: "Delta_Human" maps to
    /// `deltaHuman`, "Prompt" to `text`. A field with MORE than one
    /// matching column guesses nothing — the mapping sheet's picker
    /// decides; guessing between duplicates would be silent coercion.
    public static func guessMapping(
        columns: [String], target: Target
    ) -> [String: String] {
        var mapping: [String: String] = [:]
        for field in requiredFields(for: target) + optionalFields(for: target) {
            let wanted = Set(aliases(for: field, target: target).map(normalize))
            let candidates = columns.filter { wanted.contains(normalize($0)) }
            if candidates.count == 1 { mapping[field] = candidates[0] }
        }
        return mapping
    }

    static func normalize(_ name: String) -> String {
        name.lowercased().filter { $0.isLetter || $0.isNumber }
    }

    /// Modest synonym lists — common export names only, never fuzzy
    /// matching (a wrong guess is worse than an empty picker).
    static func aliases(for field: String, target: Target) -> [String] {
        switch (target, field) {
        case (.taskPrompts, "text"):
            return ["text", "prompt"]
        case (.taskPrompts, "id"):
            return ["id", "item_id"]
        case (.taskPrompts, "options"):
            return ["options", "choices"]
        case (.taskPrompts, "target"):
            return ["target", "answer"]
        default:
            return [field]
        }
    }

    /// Throws the plain refusal when a required field has no source column
    /// or a mapped column does not exist. Names exactly what is missing.
    static func checkMapping(
        _ mapping: [String: String], table: Table, target: Target
    ) throws {
        let missing = requiredFields(for: target).filter {
            (mapping[$0] ?? "").isEmpty
        }
        guard missing.isEmpty else {
            throw Problem(
                "cannot import yet — no source column chosen for "
                    + missing.joined(separator: ", ")
                    + ". The \(target.title) import needs "
                    + requiredFields(for: target).joined(separator: ", ")
                    + "; pick the column that holds each")
        }
        for (field, column) in mapping where !column.isEmpty {
            guard table.columns.contains(column) else {
                throw Problem(
                    "the mapping for '\(field)' names column '\(column)', "
                        + "which '\(table.columns.joined(separator: ", "))' "
                        + "does not include — re-pick the column")
            }
        }
    }

    // MARK: - Conversion (pure)

    /// A delimited options cell → option strings: JSON arrays pass through;
    /// strings split on "|" when present, else ";", else stand alone.
    /// Public so the mapping sheet's preview shows options exactly as they
    /// will import.
    public static func optionStrings(_ value: Value) -> [String] {
        switch value {
        case .array(let items):
            return items.map(\.text)
        case .string(let s):
            let separator: Character? =
                s.contains("|") ? "|" : (s.contains(";") ? ";" : nil)
            guard let separator else {
                let trimmed = s.trimmingCharacters(in: .whitespaces)
                return trimmed.isEmpty ? [] : [trimmed]
            }
            return s.split(separator: separator)
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
        case .number, .bool:
            return [value.text]
        }
    }

    /// Table → task-prompts JSONL (the run loop's format), `options` and
    /// `target` preserved. The result is re-checked by
    /// `TaskPromptsImport.preview` — the same rules the pin's parser
    /// enforces — so a conversion that returns is a file that pins.
    public static func taskPromptsJSONL(
        table: Table, mapping: [String: String]
    ) throws -> String {
        try checkMapping(mapping, table: table, target: .taskPrompts)
        var lines: [String] = []
        for (index, row) in table.rows.enumerated() {
            var object: [String: Any] = [:]
            guard
                let textColumn = mapping["text"],
                let textValue = row[textColumn], !textValue.isEmpty
            else {
                throw Problem(
                    "data row \(index + 1) has no prompt text (column "
                        + "'\(mapping["text"] ?? "?")' is empty) — every task "
                        + "prompt needs its text; remove the row or fill it in")
            }
            object["text"] = textValue.text
            if let idColumn = mapping["id"], let id = row[idColumn], !id.isEmpty {
                object["id"] = id.text
            }
            if let optionsColumn = mapping["options"],
                let options = row[optionsColumn], !options.isEmpty
            {
                let parsed = optionStrings(options)
                if !parsed.isEmpty { object["options"] = parsed }
            }
            if let targetColumn = mapping["target"],
                let target = row[targetColumn], !target.isEmpty
            {
                object["target"] = jsonValue(target)
            }
            let encoded = try JSONSerialization.data(
                withJSONObject: object, options: [.sortedKeys])
            lines.append(String(decoding: encoded, as: UTF8.self))
        }
        let jsonl = lines.joined(separator: "\n") + "\n"
        // Belt: the pin's own acceptance rules (duplicate ids, schema) —
        // a conversion that "succeeded" but refused to pin would be a lie.
        if case .failure(let line, let message) = TaskPromptsImport.preview(jsonl) {
            throw Problem(
                "the converted prompts failed the task-prompt check — data "
                    + "row \(line): \(message)")
        }
        return jsonl
    }

    private static func jsonValue(_ value: Value) -> Any {
        switch value {
        case .string(let s): return s
        case .number(let n):
            if n.rounded() == n, abs(n) < 1e15 { return Int64(n) }
            return n
        case .bool(let b): return b
        case .array(let items): return items.map { jsonValue($0) }
        }
    }

    /// Table → human-baseline CSV in the analyze loader's columns
    /// (`PinShapeValidation.humanBaselineRequiredColumns`). Numeric fields
    /// must parse as numbers NOW — feedback at the moment of action, not a
    /// failure later at analyze.
    public static func humanBaselineCSV(
        table: Table, mapping: [String: String]
    ) throws -> String {
        try checkMapping(mapping, table: table, target: .humanBaseline)
        let columns = PinShapeValidation.humanBaselineRequiredColumns
        var lines = [columns.joined(separator: ",")]
        for (index, row) in table.rows.enumerated() {
            var cells: [String] = []
            for field in columns {
                guard
                    let column = mapping[field],
                    let value = row[column], !value.isEmpty
                else {
                    throw Problem(
                        "data row \(index + 1) has no \(field) (column "
                            + "'\(mapping[field] ?? "?")' is empty) — the "
                            + "analyze loader needs every row's "
                            + columns.joined(separator: ", "))
                }
                let text = value.text.trimmingCharacters(in: .whitespaces)
                if field != "endpoint", Double(text) == nil {
                    throw Problem(
                        "data row \(index + 1): \(field) is '\(text)', which "
                            + "is not a number — the analyze loader reads "
                            + "\(field) as a numeric effect value")
                }
                cells.append(csvField(text))
            }
            lines.append(cells.joined(separator: ","))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    static func csvField(_ text: String) -> String {
        guard
            text.contains(where: {
                $0 == "," || $0 == "\"" || $0.isNewline
            })
        else { return text }
        return "\"" + text.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }

    // MARK: - Import (write to the standard destination + pin)

    /// Converted task prompts land at the readiness scaffold's destination
    /// and pin through `ExperimentStore.pinTaskPrompts` (the run loop's own
    /// parser re-checks). Transactional (the study-pack importer's
    /// discipline): conversion validates BEFORE the write, and a failure
    /// after the write — the pin OR the caller's `persist` step — rolls
    /// back a file this import created, never a pre-existing one, so a
    /// refused import leaves no unpinned file behind.
    ///
    /// `pinTaskPrompts` only mutates the in-memory manifest; callers that
    /// save it afterward (the panel) MUST pass that save as `persist` so
    /// the whole import is atomic — a manifest frozen on disk between the
    /// pin and the save (the realistic race) then rolls the write back
    /// instead of orphaning it. Drafts only, same rule as every pin edit.
    @discardableResult
    public static func importTaskPrompts(
        table: Table, mapping: [String: String],
        manifest: inout ExperimentManifest,
        persist: (ExperimentManifest) throws -> Void = { _ in }
    ) throws -> TaskPromptsImport.ImportResult {
        let jsonl = try taskPromptsJSONL(table: table, mapping: mapping)
        let file = DataTemplates.taskPromptsDestination(experiment: manifest.name)
        let write = try writeRefusingDifferingOverwrite(
            Data(jsonl.utf8), relativePath: file)
        do {
            let hash = try ExperimentStore.pinTaskPrompts(file, into: &manifest)
            try persist(manifest)
            return TaskPromptsImport.ImportResult(
                file: file, recordCount: table.rows.count, hash: hash)
        } catch {
            rollBack(write, relativePath: file)
            throw error
        }
    }

    /// Converted baselines land at `prompts/baselines/<study>-human-
    /// baseline.csv` and pin through `ExperimentStore.pinHumanBaseline`
    /// (shape-validated, drafts only, persisted by the store itself).
    /// Transactional like `importTaskPrompts`: a pin refusal — e.g. a
    /// frozen manifest — rolls back a file this import created, and never
    /// deletes one that already existed.
    @discardableResult
    public static func importHumanBaseline(
        table: Table, mapping: [String: String], experimentName: String
    ) throws -> ExperimentManifest.HumanBaseline {
        let csv = try humanBaselineCSV(table: table, mapping: mapping)
        let file = DataTemplates.humanBaselineDestination(
            experiment: experimentName)
        let write = try writeRefusingDifferingOverwrite(
            Data(csv.utf8), relativePath: file)
        do {
            return try ExperimentStore.pinHumanBaseline(
                path: file, experimentName: experimentName)
        } catch {
            rollBack(write, relativePath: file)
            throw error
        }
    }

    /// What a `writeRefusingDifferingOverwrite` call did to the destination,
    /// carried to `rollBack` so a failure AFTER the write can undo exactly
    /// that — remove a file it created, restore the bytes it replaced, and
    /// never touch a pre-existing identical file.
    enum ImportWrite {
        /// The file did not exist; this call created it.
        case created
        /// Identical bytes were already there; the file was not touched.
        case identical
        /// Explicit replace: carries the previous bytes so a later failure
        /// can restore them.
        case replaced(previous: Data)
    }

    /// The study-pack write rule: identical bytes are idempotent, differing
    /// bytes refuse with the remedy, and a symlinked destination refuses
    /// outright (a write would follow it). `replacingExisting: true` is the
    /// ONE sanctioned exception — an EXPLICIT researcher affordance (the
    /// Import JSONL sheet's "Replace the existing file" checkbox), never a
    /// default — and even then a symlink or an unreadable existing file
    /// still refuses (a replace whose previous bytes can't be captured
    /// can't be rolled back).
    @discardableResult
    static func writeRefusingDifferingOverwrite(
        _ data: Data, relativePath: String,
        replacingExisting: Bool = false,
        differingRemedy: String =
            "move the existing file aside (or rename the study) and import "
            + "again"
    ) throws -> ImportWrite {
        let fm = FileManager.default
        let url = ExperimentStore.resolveProjectPath(relativePath)
        if (try? fm.destinationOfSymbolicLink(atPath: url.path)) != nil {
            throw Problem(
                "'\(relativePath)' is a symlink — imports write real files "
                    + "only; remove the link and import again")
        }
        if fm.fileExists(atPath: url.path) {
            let existing = try? Data(contentsOf: url)
            if existing == data {
                return .identical  // idempotent, just (re)pin
            }
            guard replacingExisting, let previous = existing else {
                throw Problem(
                    "'\(relativePath)' already exists with different "
                        + "contents — imports never overwrite; "
                        + differingRemedy)
            }
            try data.write(to: url, options: .atomic)
            return .replaced(previous: previous)
        }
        try fm.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
        return .created
    }

    /// Rollback half of the transaction: undoes exactly what the import's
    /// write did — a pre-existing (identical) file is never deleted by a
    /// later failure, and an explicit replace restores the previous bytes.
    static func rollBack(_ write: ImportWrite, relativePath: String) {
        let url = ExperimentStore.resolveProjectPath(relativePath)
        switch write {
        case .created:
            try? FileManager.default.removeItem(at: url)
        case .identical:
            break
        case .replaced(let previous):
            try? previous.write(to: url, options: .atomic)
        }
    }
}
