import Foundation

/// What one paired difference of a stored effect row is, settled from the
/// row, the analysis's stamp, and the run's records, so the app never assumes
/// it.
///
/// Before SteerLab 0.9.7 the Mac engine paired every response of a
/// multi-sample run with the baseline response to the same item and seed,
/// counted responses in `n`, and stamped no unit. Since 0.9.7 it averages an
/// item's samples within each condition, and still stamps no unit on pooled
/// rows, so an old response-level row and a new item-level row look the same
/// in the file. The run's records tell them apart: an item-level row counts
/// at most the items the records pair with the baseline.
///
/// This is the twin of the Python reader's rule (`results_export.
/// _paired_items` and `resolve_units`, which the export, its methods summary,
/// and the results page use). A recorded unit is used; otherwise a row
/// counting no more pairs than the paired items is item-level, a row counting
/// more is response-level, and a condition with no paired items is unknown.
/// `Tests/Fixtures/cross-engine/effect-units.json` holds the Python reader's
/// answers, and `EffectUnitTests` holds this copy to them. Nothing stored is
/// changed.
extension RunResults {

    public struct EffectUnit: Sendable, Equatable {
        /// Where the unit comes from, in the Python reader's words
        /// (`unit_of_analysis_source` in the export).
        public enum Source: String, Sendable, Equatable {
            /// The row or its analysis stamped it.
            case recorded
            /// Not stamped; the engines' rule, the item, and the run's
            /// records are consistent with it.
            case engineDefault = "engine_default"
            /// Not stamped; the row counts more pairs than the run has paired
            /// items, so it paired responses.
            case inferredFromRecords = "inferred_from_records"
            /// Not stamped, and the records cannot settle it.
            case notEstablished = "not_established"
        }

        /// "item", "transcript", "sample", "response", or "unknown" — or
        /// whatever unit the row or analysis stamped, as stamped.
        public var unit: String
        public var source: Source
        /// How many distinct items the run's records answer under both the
        /// row's condition and the baseline. nil when the records hold
        /// nothing for that condition, or were not read because a stamp
        /// already settled the unit.
        public var pairedItems: Int?

        public init(unit: String, source: Source, pairedItems: Int? = nil) {
            self.unit = unit
            self.source = source
            self.pairedItems = pairedItems
        }

        /// A row nothing has settled: it claims no unit.
        public static let unresolved = EffectUnit(unit: "unknown", source: .notEstablished)

        /// The row paired responses, not items.
        public var isResponses: Bool { unit == "response" }
    }

    /// Per condition, how many distinct items have a response under both
    /// that condition and the baseline in the run's records: the most pairs
    /// an item-level row of that condition can count.
    public struct PairedItems: Sendable, Equatable {
        public var counts: [String: Int]
        /// Every record of the run was read. When only the head of a large
        /// file was (a remote preview), each count is a lower bound: more
        /// records can only pair more items.
        public var complete: Bool

        public init(counts: [String: Int], complete: Bool) {
            self.counts = counts
            self.complete = complete
        }
    }

    /// Counts the paired items of a run's generations, one line at a time.
    /// A line counts as the Python reader counts it (`results_export.paired_items`):
    /// a JSON object that carries no `error` and is either a generated response
    /// (an `output`) or an instrument readout (an `instrument`), since a
    /// deterministic choice study records readouts alone. Its condition and
    /// item are the record's `condition` and `promptID`, with a missing one
    /// read as empty.
    public struct PairedItemCounter: Sendable {
        private var items: [String: Set<String>] = [:]

        public init() {}

        public mutating func add<Bytes: Collection<UInt8>>(line bytes: Bytes) {
            let text = String(decoding: bytes, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return }
            let data = Data(text.utf8)
            // Python's reader also accepts NaN and Infinity, which a strict
            // JSON reader refuses; the lenient reader is only the fallback.
            guard
                let object = (try? JSONSerialization.jsonObject(with: data))
                    ?? (try? JSONSerialization.jsonObject(with: data, options: [.json5Allowed])),
                let record = object as? [String: Any],
                record["error"] == nil,
                record["instrument"] != nil || record["output"] != nil
            else { return }
            items[Self.text(record["condition"]), default: []]
                .insert(Self.text(record["promptID"]))
        }

        public var counts: [String: Int] {
            let baseline = items["baseline"] ?? []
            var counts: [String: Int] = [:]
            for (condition, ids) in items where condition != "baseline" {
                counts[condition] = ids.intersection(baseline).count
            }
            return counts
        }

        /// A record value as the Python reader keys it (`str(value)`): only
        /// distinctness matters, so a number keeps its JSON spelling.
        private static func text(_ value: Any?) -> String {
            switch value {
            case nil: return ""
            case let text as String: return text
            case is NSNull: return "None"
            case let number as NSNumber:
                if CFGetTypeID(number) == CFBooleanGetTypeID() {
                    return number.boolValue ? "True" : "False"
                }
                return number.stringValue
            default:
                return String(describing: value!)
            }
        }
    }

    /// The paired items of a generations text held in memory.
    public static func pairedItems(fromJSONL text: String) -> [String: Int] {
        var counter = PairedItemCounter()
        for line in text.utf8.split(separator: 0x0A, omittingEmptySubsequences: true) {
            counter.add(line: line)
        }
        return counter.counts
    }

    /// The paired items of a generations file, read whole in bounded chunks
    /// (a long-text run's file can be far larger than the app's preview
    /// read). nil when the file cannot be read.
    public static func pairedItems(generationsAt url: URL) -> [String: Int]? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        var counter = PairedItemCounter()
        var pending = Data()
        while true {
            let chunk: Data
            do {
                guard let read = try handle.read(upToCount: 4 << 20), !read.isEmpty else { break }
                chunk = read
            } catch {
                return nil
            }
            pending.append(chunk)
            var start = pending.startIndex
            while let newline = pending[start...].firstIndex(of: 0x0A) {
                counter.add(line: pending[start..<newline])
                start = pending.index(after: newline)
            }
            pending = Data(pending[start...])
        }
        counter.add(line: pending)
        return counter.counts
    }

    /// The unit of one row. `n` is the row's stored count, 0 when the file
    /// gave none; `pairedItems` is nil when no records could be read.
    ///
    /// With every record read this is exactly the Python reader's rule. With
    /// only the head of the records (`complete` false) a count is a lower
    /// bound, so a row within it is still item-level — the rest of the file
    /// can only pair more items — but a row beyond it cannot be settled
    /// here and is not established.
    public static func effectUnit(
        condition: String, n: Int, recordedUnit: String?, stampedUnit: String?,
        pairedItems: PairedItems?
    ) -> EffectUnit {
        let items = pairedItems?.counts[condition]
        if let unit = recordedUnit ?? stampedUnit {
            return EffectUnit(unit: unit, source: .recorded, pairedItems: items)
        }
        guard let items, items > 0, let pairedItems else {
            return EffectUnit(unit: "unknown", source: .notEstablished, pairedItems: items)
        }
        if n <= items {
            return EffectUnit(unit: "item", source: .engineDefault, pairedItems: items)
        }
        guard pairedItems.complete else {
            return EffectUnit(unit: "unknown", source: .notEstablished, pairedItems: items)
        }
        return EffectUnit(unit: "response", source: .inferredFromRecords, pairedItems: items)
    }

    /// Settle every row's unit, once, for every view: the sentences, the
    /// table, and the chart. `records` is asked for only when some row's
    /// unit is not already stamped, so a stamped analysis never reads its
    /// run's records.
    public static func resolveUnits(
        _ rows: [EffectSizeRow], stampedUnit: String?,
        records: () -> PairedItems?
    ) -> (rows: [EffectSizeRow], pairedItems: PairedItems?) {
        let needed = stampedUnit == nil && rows.contains { $0.recordedUnit == nil }
        let pairedItems = needed ? records() : nil
        let settled = rows.map { row in
            var row = row
            row.unit = effectUnit(
                condition: row.condition, n: row.n, recordedUnit: row.recordedUnit,
                stampedUnit: stampedUnit, pairedItems: pairedItems)
            return row
        }
        return (settled, pairedItems)
    }

    /// The unit an analysis stamped for its pooled rows: the
    /// `unitOfAnalysis` of `unit-of-analysis.json` (both engines write it for
    /// a multi-agent analysis), else the one a run's own report carries.
    static func stampedUnit(unitOfAnalysisData: Data?, report: Report?) -> String? {
        if let data = unitOfAnalysisData,
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            let unit = object["unitOfAnalysis"] as? String, !unit.isEmpty
        {
            return unit
        }
        return report?.unitOfAnalysis
    }

    /// The paired items for a LOCAL run directory's effect rows: its own
    /// generations, read whole, or — for an analysis directory, which holds
    /// no generations — those of the run it analyzed, named by
    /// `source-run.txt` or `analysis.json`'s `sourceRun` and found beside
    /// it. nil when there are no records to read.
    static func localPairedItems(runDirectory: URL, artifacts: ArtifactBytes) -> PairedItems? {
        let own = runDirectory.appending(component: "generations.jsonl")
        if FileManager.default.fileExists(atPath: own.path) {
            if let text = artifacts.generationsText, !artifacts.generationsTruncated {
                return PairedItems(counts: pairedItems(fromJSONL: text), complete: true)
            }
            return pairedItems(generationsAt: own).map { PairedItems(counts: $0, complete: true) }
        }
        guard let source = analyzedRunName(runDirectory) else { return nil }
        let generations = runDirectory.standardizedFileURL.deletingLastPathComponent()
            .appending(components: source, "generations.jsonl")
        return pairedItems(generationsAt: generations).map { PairedItems(counts: $0, complete: true) }
    }

    /// The directory name of the run an analysis directory analyzed: the
    /// first line of `source-run.txt`, else `analysis.json`'s `sourceRun`
    /// (the Python reader's order). nil when the directory names none.
    static func analyzedRunName(_ directory: URL) -> String? {
        func baseName(_ raw: String) -> String? {
            var trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            while trimmed.hasSuffix("/") { trimmed.removeLast() }
            let name = trimmed.split(separator: "/").last.map(String.init) ?? ""
            return name.isEmpty || name == "." || name == ".." ? nil : name
        }
        if let text = try? String(
            contentsOf: directory.appending(component: "source-run.txt"), encoding: .utf8),
            let line = text.split(whereSeparator: \.isNewline).first,
            let name = baseName(String(line))
        {
            return name
        }
        guard
            let data = try? Data(contentsOf: directory.appending(component: "analysis.json")),
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            let source = object["sourceRun"] as? String
        else { return nil }
        return baseName(source)
    }
}
