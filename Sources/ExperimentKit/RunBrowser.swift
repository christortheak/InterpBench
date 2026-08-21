import Foundation

/// Thin, read-only listing of the workspace's immutable `runs/` directories
/// for the Results browser, plus BOUNDED file previews for a selected run
/// (a run directory can hold a multi-hundred-MB generations.jsonl — previews
/// read a capped head, never the whole file). Reads only the canonical
/// per-run `config.json` stamp when present (older run types may lack
/// fields — absence is shown, never invented). No mutation: runs are
/// immutable by contract.
public enum RunBrowser {

    public struct Item: Identifiable, Sendable, Equatable {
        public var url: URL
        public var name: String
        public var runType: String?
        public var experiment: String?
        public var modelID: String?
        public var createdAt: String?
        public var revision: String?
        public var substrate: String?
        public var appVersion: String?
        /// Whether the directory carries sweep artifacts (`sweep.csv`) —
        /// lets a legacy sweep run without a config.json stamp still read as
        /// an optimization in the browser.
        public var hasSweepArtifacts: Bool

        public var id: String { url.path }

        /// User-facing run-type label: the raw `runType` stamp, except that
        /// sweep runs read in study vocabulary ("optimization (screen)").
        public var displayRunType: String? {
            RunBrowser.displayRunType(
                stamped: runType, hasSweepArtifacts: hasSweepArtifacts)
        }

        public init(
            url: URL, name: String, runType: String? = nil,
            experiment: String? = nil, modelID: String? = nil,
            createdAt: String? = nil, revision: String? = nil,
            substrate: String? = nil, appVersion: String? = nil,
            hasSweepArtifacts: Bool = false
        ) {
            self.url = url
            self.name = name
            self.runType = runType
            self.experiment = experiment
            self.modelID = modelID
            self.createdAt = createdAt
            self.revision = revision
            self.substrate = substrate
            self.appVersion = appVersion
            self.hasSweepArtifacts = hasSweepArtifacts
        }
    }

    /// Study-vocabulary label for sweep runs in the Results browser (sweep
    /// runs surface in the Agents → Optimizations region; Results speaks
    /// the same language).
    public static let optimizationRunTypeLabel = "optimization (screen)"

    /// Pure display mapping from the config.json `runType` stamp: sweep runs
    /// are labeled as optimizations; a stampless directory that carries
    /// `sweep.csv` (legacy sweep run) gets the same label; everything else
    /// passes through verbatim — the raw stamp is never rewritten on disk.
    public static func displayRunType(
        stamped: String?, hasSweepArtifacts: Bool = false
    ) -> String? {
        if stamped == "sweep" { return optimizationRunTypeLabel }
        if stamped == nil, hasSweepArtifacts { return optimizationRunTypeLabel }
        return stamped
    }

    /// One-line row summary shared by the LOCAL and REMOTE Results rows so
    /// the two browsers render identically (WS6.1 — the substrate badge is
    /// the only difference between engines): run-type badge, engine badge,
    /// experiment, model. A run without a config.json stamp renders the
    /// legacy caption exactly as before — absence is shown, never invented.
    public static func rowDetailLine(
        runType displayRunType: String?,
        substrate: String?,
        experiment: String?,
        modelID: String?
    ) -> String {
        var parts: [String] = []
        if let displayRunType { parts.append(displayRunType) }
        if let substrate { parts.append(substrate) }
        if let experiment { parts.append("exp \(experiment)") }
        if let modelID { parts.append(modelID) }
        if parts.isEmpty { parts.append("no config.json stamp (legacy run type)") }
        return parts.joined(separator: " · ")
    }

    /// The `config.json` fields the browser surfaces (RunMetadata contract).
    struct Stamp: Sendable, Equatable {
        var runType: String?
        var experiment: String?
        var modelID: String?
        var createdAt: String?
        var revision: String?
        var substrate: String?
        var appVersion: String?
    }

    /// Minimal read of a run directory's `config.json` (the RunMetadata
    /// contract; every value may be null and legacy configs may differ —
    /// unknown shapes degrade to a bare listing, never an error).
    static func readStamp(in directory: URL) -> Stamp {
        let url = directory.appending(component: RunMetadata.fileName)
        guard let data = try? Data(contentsOf: url),
            let object = try? JSONSerialization.jsonObject(with: data),
            let dictionary = object as? [String: Any]
        else { return Stamp() }
        return Stamp(
            runType: dictionary["runType"] as? String,
            experiment: dictionary["experiment"] as? String,
            modelID: dictionary["modelID"] as? String,
            createdAt: dictionary["createdAt"] as? String,
            revision: dictionary["revision"] as? String,
            substrate: dictionary["substrate"] as? String,
            appVersion: dictionary["appVersion"] as? String)
    }

    /// Top-level run directories, newest first (timestamp-prefixed names sort
    /// lexicographically). Nested artifact directories (model-variants/…) are
    /// listed by their top-level container only.
    public static func list(
        runsDirectory: URL = ExperimentStore.runsDirectory,
        limit: Int = 300
    ) -> [Item] {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: runsDirectory,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
        else { return [] }
        return entries
            .filter {
                (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
            .prefix(limit)
            .map(item(at:))
    }

    /// Build the same browser item for a known run directory. This keeps
    /// study-scoped Results and the global Results browser on one semantic
    /// rendering path instead of maintaining two subtly different summaries.
    public static func item(at directory: URL) -> Item {
        let stamp = readStamp(in: directory)
        let hasSweepCSV = FileManager.default.fileExists(
            atPath: directory.appending(component: "sweep.csv").path)
        return Item(
            url: directory,
            name: directory.lastPathComponent,
            runType: stamp.runType,
            experiment: stamp.experiment,
            modelID: stamp.modelID,
            createdAt: stamp.createdAt,
            revision: stamp.revision,
            substrate: stamp.substrate,
            appVersion: stamp.appVersion,
            hasSweepArtifacts: hasSweepCSV)
    }

    /// One-slot memo for `item(at:)` keyed by directory (F10): building an
    /// Item reads config.json and probes sweep.csv synchronously, and runs
    /// are immutable — rebuilding for the same directory is pure repeated
    /// disk I/O (an iCloud-evicted config.json can even stall). Holders keep
    /// one per selection surface; a different directory replaces the slot.
    public struct MemoizedItem: Sendable, Equatable {
        private var slot: Item?

        public init() {}

        public mutating func item(at directory: URL) -> Item {
            if let slot, slot.url == directory { return slot }
            let built = RunBrowser.item(at: directory)
            slot = built
            return built
        }
    }

    // MARK: - Per-run file listing

    public struct FileEntry: Identifiable, Sendable, Equatable {
        public var url: URL
        public var name: String
        public var size: Int
        public var isDirectory: Bool

        public var id: String { url.path }

        public init(url: URL, name: String, size: Int, isDirectory: Bool) {
            self.url = url
            self.name = name
            self.size = size
            self.isDirectory = isDirectory
        }
    }

    /// Top-level entries of a run directory, files first, alphabetical.
    /// Nested directories (variant artifacts, prepared jobs) are listed as
    /// one row each — the browser previews top-level files only.
    public static func files(in runDirectory: URL) -> [FileEntry] {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: runDirectory,
                includingPropertiesForKeys: [.isDirectoryKey, .fileSizeKey],
                options: [.skipsHiddenFiles])
        else { return [] }
        return entries
            .map { url in
                let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey])
                return FileEntry(
                    url: url,
                    name: url.lastPathComponent,
                    size: values?.fileSize ?? 0,
                    isDirectory: values?.isDirectory == true)
            }
            .sorted { a, b in
                if a.isDirectory != b.isDirectory { return !a.isDirectory }
                return a.name < b.name
            }
    }

    // MARK: - Bounded previews

    /// What the detail pane renders for one file. Every variant is bounded:
    /// parsers receive at most the capped head of the file.
    public enum FilePreview: Sendable, Equatable {
        /// Pretty top-level summary of a JSON report/config/recommendations.
        case keyValues([KeyValueRow])
        /// First rows of a metrics CSV.
        case table(header: [String], rows: [[String]], truncated: Bool)
        /// First records of a JSONL file (generations, judgments, …).
        case records([RecordExcerpt], truncated: Bool)
        /// Small plain-text head (logs, notes).
        case text(String, truncated: Bool)
        /// No preview (binary, too large, unknown) — show name+size+Finder.
        case unavailable(reason: String)
    }

    public struct KeyValueRow: Identifiable, Sendable, Equatable {
        public var key: String
        public var value: String
        public var id: String { key }

        public init(key: String, value: String) {
            self.key = key
            self.value = value
        }
    }

    /// One JSONL record, reduced to what a researcher scans for: the
    /// condition, the prompt, and an output excerpt — or, for answer-token
    /// choice records (which carry no sampled output), the instrument
    /// readout line (A6: choice records used to fall through to raw JSON).
    /// Records without those keys (judgments, custom formats) fall back to
    /// a raw-line excerpt.
    public struct RecordExcerpt: Identifiable, Sendable, Equatable {
        public var index: Int
        public var condition: String?
        public var prompt: String?
        public var output: String?
        /// Human-readable readout of a choice record ("selected …"), when
        /// the record is one.
        public var choiceSummary: String?
        public var fallback: String?
        public var id: Int { index }

        public init(
            index: Int, condition: String? = nil, prompt: String? = nil,
            output: String? = nil, choiceSummary: String? = nil,
            fallback: String? = nil
        ) {
            self.index = index
            self.condition = condition
            self.prompt = prompt
            self.output = output
            self.choiceSummary = choiceSummary
            self.fallback = fallback
        }
    }

    /// Byte caps per preview kind. JSON is parsed whole (reports are small;
    /// anything over the cap gets no preview rather than a partial parse).
    public static let jsonPreviewByteLimit = 1_048_576
    static let csvHeadByteLimit = 131_072
    static let jsonlHeadByteLimit = 262_144
    static let textHeadByteLimit = 4_096

    /// Preview dispatch by extension. Never slurps: JSON is size-gated
    /// before a full read; CSV/JSONL/text read a bounded head only.
    public static func preview(for file: FileEntry) -> FilePreview {
        if file.isDirectory {
            return .unavailable(reason: "directory — open in Finder")
        }
        switch (file.name as NSString).pathExtension.lowercased() {
        case "json":
            guard file.size <= jsonPreviewByteLimit else {
                return .unavailable(reason: "JSON too large to preview")
            }
            guard let data = try? Data(contentsOf: file.url),
                let rows = jsonKeyValues(data)
            else { return .unavailable(reason: "unreadable JSON") }
            return .keyValues(rows)
        case "csv":
            guard let head = readHead(of: file.url, maxBytes: csvHeadByteLimit),
                let table = csvTable(head.text)
            else { return .unavailable(reason: "unreadable CSV") }
            return .table(
                header: table.header, rows: table.rows,
                truncated: head.truncated || table.truncated)
        case "jsonl":
            guard let head = readHead(of: file.url, maxBytes: jsonlHeadByteLimit)
            else { return .unavailable(reason: "unreadable JSONL") }
            let parsed = jsonlRecords(head.text)
            return .records(parsed.records, truncated: head.truncated || parsed.truncated)
        case "txt", "md", "log":
            guard let head = readHead(of: file.url, maxBytes: textHeadByteLimit)
            else { return .unavailable(reason: "unreadable text") }
            return .text(head.text, truncated: head.truncated)
        default:
            return .unavailable(reason: "no preview for this file type")
        }
    }

    /// Bounded head read: at most `maxBytes`, and when the file is larger,
    /// the tail partial line is dropped so parsers only see complete lines.
    static func readHead(of url: URL, maxBytes: Int) -> (text: String, truncated: Bool)? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let data = try? handle.read(upToCount: maxBytes + 1) else { return nil }
        return headText(data, maxBytes: maxBytes)
    }

    /// Pure head-of-bytes → text reduction shared by the local file reader
    /// and the remote (server-fetched) preview path: cap at `maxBytes` and,
    /// when the input exceeds the cap, drop the tail partial line so parsers
    /// only ever see complete lines. Callers hand in up to `maxBytes + 1`
    /// bytes so truncation is detectable from the data alone.
    static func headText(_ data: Data, maxBytes: Int) -> (text: String, truncated: Bool) {
        let truncated = data.count > maxBytes
        var head = truncated ? data.prefix(maxBytes) : data[...]
        if truncated, let lastNewline = head.lastIndex(of: 0x0A) {
            head = head[..<lastNewline]
        }
        return (String(decoding: head, as: UTF8.self), truncated)
    }

    // MARK: Pure parsers (unit-tested on fixture strings)

    /// Top-level key/value summary of a JSON object. Scalars render
    /// verbatim; nested containers summarize as "{N keys}" / "[N items]"
    /// (short scalar arrays inline their values). Keys sort alphabetically —
    /// a stable, scannable report card, not a JSON re-print.
    public static func jsonKeyValues(_ data: Data, maxEntries: Int = 60) -> [KeyValueRow]? {
        guard let object = try? JSONSerialization.jsonObject(with: data),
            let dictionary = object as? [String: Any]
        else { return nil }
        return dictionary.keys.sorted().prefix(maxEntries).map { key in
            KeyValueRow(key: key, value: summarize(dictionary[key]))
        }
    }

    private static func summarize(_ value: Any?) -> String {
        switch value {
        case nil, is NSNull:
            return "null"
        case let text as String:
            return text.count > 200 ? String(text.prefix(200)) + "…" : text
        case let number as NSNumber:
            // NSNumber bridges bools too; render them as JSON would.
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                return number.boolValue ? "true" : "false"
            }
            return "\(number)"
        case let array as [Any]:
            let scalars = array.compactMap { element -> String? in
                switch element {
                case let text as String: return text
                case let number as NSNumber: return "\(number)"
                default: return nil
                }
            }
            if scalars.count == array.count, array.count <= 4 {
                return "[" + scalars.joined(separator: ", ") + "]"
            }
            return "[\(array.count) item\(array.count == 1 ? "" : "s")]"
        case let nested as [String: Any]:
            return "{\(nested.count) key\(nested.count == 1 ? "" : "s")}"
        default:
            return "\(value ?? "null")"
        }
    }

    /// First rows of a CSV: line 1 is the header. Quote-aware field split
    /// (metrics CSVs quote free-text cells); no multi-line cell support —
    /// run metrics are one record per line by contract.
    public static func csvTable(
        _ text: String, maxRows: Int = 12
    ) -> (header: [String], rows: [[String]], truncated: Bool)? {
        // CRLF-safe (field incident 2026-08-04): server CSVs are CRLF and
        // "\r\n" is one grapheme — a literal "\n" split rendered them as
        // a lone header row.
        let lines = text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard let headerLine = lines.first else { return nil }
        let header = splitCSVLine(headerLine)
        let rows = lines.dropFirst().prefix(maxRows).map(splitCSVLine)
        return (header, Array(rows), lines.count - 1 > maxRows)
    }

    static func splitCSVLine(_ line: String) -> [String] {
        var fields: [String] = []
        var current = ""
        var inQuotes = false
        var iterator = line.makeIterator()
        while let character = iterator.next() {
            switch character {
            case "\"":
                inQuotes.toggle()
            case "," where !inQuotes:
                fields.append(current)
                current = ""
            default:
                current.append(character)
            }
        }
        fields.append(current)
        return fields
    }

    /// First records of a JSONL file. Generation-shaped records surface
    /// condition/prompt/output (excerpted); anything else degrades to a raw
    /// line excerpt — shown, never dropped silently.
    public static func jsonlRecords(
        _ text: String, maxRecords: Int = 4, excerptLength: Int = 280
    ) -> (records: [RecordExcerpt], truncated: Bool) {
        let lines = text.split(whereSeparator: \.isNewline)
        let records = lines.prefix(maxRecords).enumerated().map { index, line in
            excerptRecord(
                index: index, line: String(line), excerptLength: excerptLength)
        }
        return (records, lines.count > maxRecords)
    }

    private static func excerptRecord(
        index: Int, line: String, excerptLength: Int
    ) -> RecordExcerpt {
        guard let data = line.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data),
            let dictionary = object as? [String: Any]
        else {
            return RecordExcerpt(index: index, fallback: excerpt(line, excerptLength))
        }
        let condition = dictionary["condition"] as? String
        let prompt = dictionary["prompt"] as? String
        let output = dictionary["output"] as? String
        let choiceSummary = choiceSummaryLine(dictionary)
        guard condition != nil || prompt != nil || output != nil
            || choiceSummary != nil
        else {
            return RecordExcerpt(index: index, fallback: excerpt(line, excerptLength))
        }
        return RecordExcerpt(
            index: index,
            condition: condition,
            prompt: prompt.map { excerpt($0, excerptLength) },
            output: output.map { excerpt($0, excerptLength) },
            choiceSummary: choiceSummary)
    }

    /// The readable line for an answer-token choice record (cross-engine
    /// contract keys: instrument/selected/target/choiceProbability/logOdds/
    /// margin). nil for non-instrument records.
    static func choiceSummaryLine(_ dictionary: [String: Any]) -> String? {
        guard let instrument = dictionary["instrument"] as? String else { return nil }
        var parts: [String] = [instrument]
        if let selected = dictionary["selected"] as? String {
            parts.append("selected \"\(selected)\"")
        }
        let target = dictionary["target"] as? String
        if let target {
            parts.append("target \"\(target)\"")
            if let probabilities = dictionary["choiceProbability"] as? [String: Any],
                let probability = (probabilities[target] as? NSNumber)?.doubleValue
            {
                parts.append(String(format: "P(target) %.3f", probability))
            }
            if let odds = dictionary["logOdds"] as? [String: Any],
                let logOdds = (odds[target] as? NSNumber)?.doubleValue
            {
                parts.append(String(format: "log-odds %+.3f", logOdds))
            }
        }
        if let margin = (dictionary["margin"] as? NSNumber)?.doubleValue {
            parts.append(String(format: "margin %.3f", margin))
        }
        return parts.joined(separator: " · ")
    }

    private static func excerpt(_ text: String, _ length: Int) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.count > length ? String(trimmed.prefix(length)) + "…" : trimmed
    }
}
