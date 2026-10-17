import Foundation

/// What one paired difference of a stored effect row is, settled from the
/// row, the analysis's stamp, the engine its config.json names, and the run's
/// records, so the app never assumes it.
///
/// Before SteerLab 0.9.7 the Mac engine paired every response of a
/// multi-sample run with the baseline response to the same item and seed,
/// counted responses in `n`, and stamped no unit. Since 0.9.7 it averages an
/// item's samples within each condition, as the Python engine always has;
/// both stamp the unit since 0.9.8. A row counting fewer pairs than the run's
/// items does not show it counts items: declared exclusions, missing values,
/// or a subgroup leave five responses of one item looking like five items.
///
/// This is the twin of the Python reader's rule (`results_export.
/// resolve_units`, which the export, its methods summary, and the results
/// page use). The first that applies settles a row: a recorded unit; item
/// when the records hold one response or readout per item under the row's
/// condition and the baseline (the units coincide); for a pooled row, how the
/// engine named by config.json paired (`enginePairing`); response when the
/// row counts more pairs than the paired items; otherwise unknown.
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
            /// Not stamped; the records hold one response or readout per
            /// item under the row's condition and the baseline, so an item
            /// and a response are the same thing.
            case oneSamplePerItem = "one_sample_per_item"
            /// Not stamped; the engine the analysis's config.json names
            /// paired these rows in a known way.
            case analysisEngine = "analysis_engine"
            /// Not stamped; the row counts more pairs than the run has paired
            /// items, so it paired responses.
            case inferredFromRecords = "inferred_from_records"
            /// Not stamped, and nothing settles whether its pairs count items
            /// or responses.
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

    /// Per condition, how many distinct items have a record (a generated
    /// response or an instrument readout) under both that condition and the
    /// baseline in the run's records: the most pairs an item-level row of that
    /// condition can count.
    public struct PairedItems: Sendable, Equatable {
        public var counts: [String: Int]
        /// Every record of the run was read. When only the head of a large
        /// file was (a remote preview), each count is a lower bound: more
        /// records can only pair more items.
        public var complete: Bool
        /// Per condition, the most responses and the most readouts any one
        /// item has (the Python reader's `samples_per_item`). Lower bounds,
        /// like `counts`, when the records were not all read.
        public var samplesPerItem: [String: SamplesPerItem]

        public init(
            counts: [String: Int], complete: Bool,
            samplesPerItem: [String: SamplesPerItem] = [:]
        ) {
            self.counts = counts
            self.complete = complete
            self.samplesPerItem = samplesPerItem
        }
    }

    /// The most generated responses, and the most instrument readouts, that
    /// any one item has under one condition.
    public struct SamplesPerItem: Sendable, Equatable {
        public var responses: Int
        public var readouts: Int

        public init(responses: Int, readouts: Int) {
            self.responses = responses
            self.readouts = readouts
        }
    }

    /// How the engine that made an effect table paired its pooled rows, as
    /// the directory's config.json names that engine (the Python reader's
    /// `engine_pairing`).
    public enum EnginePairing: String, Sendable, Equatable {
        /// An item's samples averaged within each condition: the Python
        /// engine always, the Mac engine since SteerLab 0.9.7.
        case items
        /// Every response paired with the baseline response to the same item
        /// and seed: the Mac engine before 0.9.7, on a run with several
        /// responses per item.
        case responses
    }

    /// The first Mac engine release that averages an item's samples.
    public static let macItemPairingVersion = (0, 9, 7)

    /// The one outcome the Mac engine before 0.9.7 read from instrument
    /// readouts, one per item and condition, and so paired by item even then.
    static let macReadoutOutcomes: Set<String> = ["ordinalPosition"]

    /// `enginePairing` from a directory's config.json bytes.
    public static func enginePairing(configData: Data?) -> EnginePairing? {
        guard
            let data = configData,
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return nil }
        return enginePairing(
            substrate: object["substrate"] as? String, appVersion: object["appVersion"] as? String)
    }

    /// How the engine named by config.json's `substrate` and `appVersion`
    /// paired, or nil when they do not establish it: a missing or unreadable
    /// stamp, another engine, or a pre-release of 0.9.7 itself (either side
    /// of the change). The Python engine has always paired items; the Mac
    /// engine's "swift-app 0.9.6+1a2b3c4d" says which side of 0.9.7 it is.
    public static func enginePairing(substrate: String?, appVersion: String?) -> EnginePairing? {
        if substrate == "python-hf-transformers" { return .items }
        guard substrate == "swift-mlx", let appVersion else { return nil }
        let pattern = /swift-app (\d+)\.(\d+)\.(\d+)(-[^+\s]+)?(\+\S*)?/
        guard
            let match = appVersion.trimmingCharacters(in: .whitespacesAndNewlines)
                .wholeMatch(of: pattern),
            let major = Int(match.1), let minor = Int(match.2), let patch = Int(match.3)
        else { return nil }
        let version = (major, minor, patch)
        if match.4 != nil, version == macItemPairingVersion { return nil }
        return version >= macItemPairingVersion ? .items : .responses
    }

    /// Counts the paired items of a run's generations, one line at a time.
    /// A line counts as the Python reader counts it (`results_export.paired_items`):
    /// a JSON object that carries no `error` and is either a generated response
    /// (an `output`) or an instrument readout (an `instrument`), since a
    /// deterministic choice study records readouts alone. Its condition and
    /// item are the record's `condition` and `promptID`, with a missing one
    /// read as empty. A record carrying an `instrument` is a readout, as the
    /// Python reader sorts it; otherwise it is a response.
    public struct PairedItemCounter: Sendable {
        private var items: [String: Set<String>] = [:]
        /// condition → item → (responses, readouts).
        private var perItem: [String: [String: (responses: Int, readouts: Int)]] = [:]

        public init() {}

        public mutating func add<Bytes: Collection<UInt8>>(line bytes: Bytes) {
            let text = String(decoding: bytes, as: UTF8.self)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return }
            let data = Data(text.utf8)
            // Python's reader also accepts NaN, Infinity and -Infinity, which a
            // strict JSON reader refuses: the fallback reads those three as
            // null (the TypeScript copy's `withoutNonFinite`), never JSON5.
            // Foundation's reader itself is slightly laxer than Python's (a
            // trailing comma, -NaN); the engines never write either.
            guard
                let object = (try? JSONSerialization.jsonObject(with: data))
                    ?? (try? JSONSerialization.jsonObject(with: Data(Self.withoutNonFinite(text).utf8))),
                let record = object as? [String: Any],
                record["error"] == nil,
                record["instrument"] != nil || record["output"] != nil
            else { return }
            let condition = Self.text(record["condition"]), item = Self.text(record["promptID"])
            items[condition, default: []].insert(item)
            var tally = perItem[condition, default: [:]][item, default: (0, 0)]
            if record["instrument"] != nil { tally.readouts += 1 } else { tally.responses += 1 }
            perItem[condition, default: [:]][item] = tally
        }

        /// `text` with each NaN, Infinity and -Infinity outside a string read
        /// as null, so a strict reader takes the lines Python's reader takes.
        static func withoutNonFinite(_ text: String) -> String {
            // Unicode scalars, not Characters: a combining mark that opens a
            // string would otherwise merge with its quote and invert the
            // in-string tracking.
            var out = String.UnicodeScalarView(), inString = false, escaped = false
            var rest = Substring(text).unicodeScalars[...]
            let tokens = ["-Infinity", "Infinity", "NaN"].map { Array($0.unicodeScalars) }
            while let scalar = rest.first {
                if inString {
                    out.append(scalar); rest = rest.dropFirst()
                    if escaped { escaped = false } else if scalar == "\\" { escaped = true } else if scalar == "\"" { inString = false }
                    continue
                }
                if scalar == "\"" { inString = true; out.append(scalar); rest = rest.dropFirst(); continue }
                if let token = tokens.first(where: { rest.starts(with: $0) }) {
                    out.append(contentsOf: "null".unicodeScalars); rest = rest.dropFirst(token.count); continue
                }
                out.append(scalar); rest = rest.dropFirst()
            }
            return String(out)
        }

        public var counts: [String: Int] {
            let baseline = items["baseline"] ?? []
            var counts: [String: Int] = [:]
            for (condition, ids) in items where condition != "baseline" {
                counts[condition] = ids.intersection(baseline).count
            }
            return counts
        }

        /// Per condition (the baseline too), the most responses and readouts
        /// any one item has.
        public var samplesPerItem: [String: SamplesPerItem] {
            perItem.mapValues { byItem in
                SamplesPerItem(
                    responses: byItem.values.map(\.responses).max() ?? 0,
                    readouts: byItem.values.map(\.readouts).max() ?? 0)
            }
        }

        /// Everything counted, as the unit rule reads it.
        public func records(complete: Bool) -> PairedItems {
            PairedItems(counts: counts, complete: complete, samplesPerItem: samplesPerItem)
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
        pairedRecords(fromJSONL: text, complete: true).counts
    }

    /// Everything the unit rule reads from a generations text held in
    /// memory: the paired items and the samples per item. `complete` says
    /// whether the text is the whole file.
    public static func pairedRecords(fromJSONL text: String, complete: Bool) -> PairedItems {
        var counter = PairedItemCounter()
        for line in text.utf8.split(separator: 0x0A, omittingEmptySubsequences: true) {
            counter.add(line: line)
        }
        return counter.records(complete: complete)
    }

    /// The paired items of a generations file, read whole in bounded chunks
    /// (a long-text run's file can be far larger than the app's preview
    /// read). nil when the file cannot be read.
    public static func pairedItems(generationsAt url: URL) -> [String: Int]? {
        pairedRecords(generationsAt: url)?.counts
    }

    /// `pairedRecords(fromJSONL:complete:)` over a whole file, read in
    /// bounded chunks. nil when the file cannot be read.
    public static func pairedRecords(generationsAt url: URL) -> PairedItems? {
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
        return counter.records(complete: true)
    }

    /// The unit of one row. `n` is the row's stored count, 0 when the file
    /// gave none; `pairedItems` is nil when no records could be read.
    public static func effectUnit(
        condition: String, n: Int, recordedUnit: String?, stampedUnit: String?,
        stratifyBy: String? = nil, metric: String = "", pairing: EnginePairing? = nil,
        pairedItems: PairedItems?
    ) -> EffectUnit {
        effectUnit(condition: condition, storedN: Double(n), recordedUnit: recordedUnit,
                   stampedUnit: stampedUnit, stratifyBy: stratifyBy, metric: metric,
                   pairing: pairing, pairedItems: pairedItems)
    }

    /// `effectUnit` over the pair count as the file stored it, compared with
    /// the paired items as the Python reader compares it (8.5 pairs of 8
    /// items are more pairs than items). The first that applies:
    ///
    /// 1. a unit the row or its analysis recorded;
    /// 2. item, when the records hold at most one response and one readout
    ///    per item under the row's condition and the baseline, and the row
    ///    counts no more pairs than the paired items (the units coincide);
    /// 3. for a pooled row, how the engine named by config.json paired
    ///    (`pairing`): items → item; the Mac engine before 0.9.7 → item for
    ///    its ordinal readouts, response when the records hold several
    ///    responses per item under both the row's condition and the baseline;
    /// 4. response, when the row counts more pairs than the paired items;
    /// 5. otherwise unknown: fewer pairs than items settles nothing.
    ///
    /// With every record read this is exactly the Python reader's rule. With
    /// only the head of the records (`complete` false) the counts are lower
    /// bounds, so rules 2 and 4, which need the whole file, do not apply;
    /// several responses seen in the head are several responses in the file.
    public static func effectUnit(
        condition: String, storedN n: Double, recordedUnit: String?, stampedUnit: String?,
        stratifyBy: String? = nil, metric: String = "", pairing: EnginePairing? = nil,
        pairedItems: PairedItems?
    ) -> EffectUnit {
        let items = pairedItems?.counts[condition]
        if let unit = recordedUnit ?? stampedUnit {
            return EffectUnit(unit: unit, source: .recorded, pairedItems: items)
        }
        let samples = pairedItems?.samplesPerItem ?? [:]
        let here = samples[condition], baseline = samples["baseline"]
        let complete = pairedItems?.complete == true
        if complete, let items, items > 0, n <= Double(items), !samples.isEmpty,
            [here, baseline].allSatisfy({ ($0?.responses ?? 0) <= 1 && ($0?.readouts ?? 0) <= 1 })
        {
            return EffectUnit(unit: "item", source: .oneSamplePerItem, pairedItems: items)
        }
        if stratifyBy == nil || stratifyBy == "" || stratifyBy == "pooled" {
            switch pairing {
            case .items:
                return EffectUnit(unit: "item", source: .analysisEngine, pairedItems: items)
            case .responses:
                if macReadoutOutcomes.contains(metric) {
                    return EffectUnit(unit: "item", source: .analysisEngine, pairedItems: items)
                }
                if let items, items > 0, (here?.responses ?? 0) > 1, (baseline?.responses ?? 0) > 1 {
                    // From part of the records the paired items are a lower
                    // bound, not the most the responses can come from, so
                    // the row names none.
                    return EffectUnit(
                        unit: "response", source: .analysisEngine, pairedItems: complete ? items : nil)
                }
            case nil:
                break
            }
        }
        if complete, let items, items > 0, n > Double(items) {
            return EffectUnit(unit: "response", source: .inferredFromRecords, pairedItems: items)
        }
        return EffectUnit(unit: "unknown", source: .notEstablished, pairedItems: items)
    }

    /// Settle every row's unit, once, for every view: the sentences, the
    /// table, and the chart. `records` is asked for only when some row's
    /// unit is not already stamped, so a stamped analysis never reads its
    /// run's records. `pairing` is how the engine named by the effect
    /// table's config.json paired (`enginePairing`).
    public static func resolveUnits(
        _ rows: [EffectSizeRow], stampedUnit: String?, pairing: EnginePairing? = nil,
        records: () -> PairedItems?
    ) -> (rows: [EffectSizeRow], pairedItems: PairedItems?) {
        let needed = stampedUnit == nil && rows.contains { $0.recordedUnit == nil }
        let pairedItems = needed ? records() : nil
        let settled = rows.map { row in
            var row = row
            row.unit = effectUnit(
                condition: row.condition, storedN: row.storedN ?? Double(row.n), recordedUnit: row.recordedUnit,
                stampedUnit: stampedUnit, stratifyBy: row.stratifyBy, metric: row.metric,
                pairing: pairing, pairedItems: pairedItems)
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
                return pairedRecords(fromJSONL: text, complete: true)
            }
            return cachedPairedRecords(generationsAt: own)
        }
        guard let source = analyzedRunName(runDirectory) else { return nil }
        let generations = runDirectory.standardizedFileURL.deletingLastPathComponent()
            .appending(components: source, "generations.jsonl")
        return cachedPairedRecords(generationsAt: generations)
    }

    /// Counts already made, by the file's path, size and modification time:
    /// a completed run never changes, so selecting it again reads nothing,
    /// and a run still being written is counted afresh.
    private static let countCache = CountCache()

    private final class CountCache: @unchecked Sendable {
        private let lock = NSLock()
        private var entries: [String: PairedItems] = [:]
        func value(_ key: String) -> PairedItems? { lock.withLock { entries[key] } }
        func store(_ key: String, _ records: PairedItems) {
            lock.withLock {
                if entries.count >= 64 { entries.removeAll() }
                entries[key] = records
            }
        }
    }

    static func cachedPairedItems(generationsAt url: URL) -> [String: Int]? {
        cachedPairedRecords(generationsAt: url)?.counts
    }

    static func cachedPairedRecords(generationsAt url: URL) -> PairedItems? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.resolvingSymlinksInPath().path),
              let size = attributes[.size] as? NSNumber, let modified = attributes[.modificationDate] as? Date
        else { return pairedRecords(generationsAt: url) }
        let key = "\(url.standardizedFileURL.path)\u{1F}\(size)\u{1F}\(modified.timeIntervalSince1970)"
        if let records = countCache.value(key) { return records }
        guard let records = pairedRecords(generationsAt: url) else { return nil }
        countCache.store(key, records)
        return records
    }

    /// The directory name of the run an analysis directory analyzed: the
    /// first line of `source-run.txt`, else `analysis.json`'s `sourceRun`
    /// (the Python reader's order). nil when the directory names none.
    static func analyzedRunName(_ directory: URL) -> String? {
        analyzedRunName(
            sourceRunText: try? String(
                contentsOf: directory.appending(component: "source-run.txt"), encoding: .utf8),
            analysisData: try? Data(contentsOf: directory.appending(component: "analysis.json")))
    }

    /// `analyzedRunName(_:)` from the two files' contents, so a remote
    /// reading, which holds bytes rather than a directory, settles it alike.
    static func analyzedRunName(sourceRunText: String?, analysisData: Data?) -> String? {
        func baseName(_ raw: String) -> String? {
            var trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            while trimmed.hasSuffix("/") { trimmed.removeLast() }
            let name = trimmed.split(separator: "/").last.map(String.init) ?? ""
            return name.isEmpty || name == "." || name == ".." ? nil : name
        }
        if let text = sourceRunText,
            let line = text.split(whereSeparator: \.isNewline).first,
            let name = baseName(String(line))
        {
            return name
        }
        guard
            let data = analysisData,
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            let source = object["sourceRun"] as? String
        else { return nil }
        return baseName(source)
    }
}
