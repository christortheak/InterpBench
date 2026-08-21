import Foundation

/// Semantic decoding of the ANALYZE-verb artifacts (app gap A3) so the
/// paper's statistics are legible in Results instead of generic CSV
/// previews: `effect-sizes.csv` (both engines' dialects), the server's
/// `alien-residuals.csv` and `promoted-movers.json`, and the
/// `cosine-matrix.csv` a validate run writes (A13's discriminant half).
///
/// Cross-engine tolerance is explicit: Swift writes
/// `condition,metric,…,meanDiff,…,adjustedP,correction`; the server writes
/// `condition,endpoint,…,deltaMean,…,adjustedP,correction,modality`.
/// One row model reads both; per-engine extras stay optional.
extension RunResults {

    // MARK: - effect-sizes.csv (and report.json effectSizes)

    public struct EffectSizeRow: Identifiable, Sendable, Equatable {
        public var condition: String
        /// Swift calls it `metric`, the server `endpoint` — one field.
        public var metric: String
        public var n: Int
        /// Paired mean difference vs the same-item baseline (`meanDiff` /
        /// `deltaMean`).
        public var meanDiff: Double
        public var ciLower: Double
        public var ciUpper: Double
        public var wilcoxonW: Double?
        public var wilcoxonP: Double?
        /// Multiple-comparison correction (both engines since 2026-07-19)
        /// and the server-only modality extra.
        public var adjustedP: Double?
        public var correction: String?
        public var modality: String?

        public var id: String { "\(condition)\u{1F}\(metric)" }

        /// Significance after correction where a correction exists
        /// (`adjustedP`), else the raw Wilcoxon p; nil when neither exists.
        public var significantAfterCorrection: Bool? {
            if let adjustedP { return adjustedP < 0.05 }
            if let wilcoxonP { return wilcoxonP < 0.05 }
            return nil
        }

        /// CI excludes zero — the CI-based movement read.
        public var ciExcludesZero: Bool { ciLower > 0 || ciUpper < 0 }
    }

    /// Parse either engine's effect-sizes.csv. Returns nil when the header
    /// carries neither dialect's required columns. Since 2026-08-06 both
    /// engines append per-stratum companion rows (stratifyBy ≠ "pooled");
    /// this reader keeps the POOLED rows only — the semantic effect views
    /// model one row per (condition, metric), and the stratified rows are
    /// a CSV-level analysis surface, not a chart input. Legacy files
    /// without the column parse unchanged.
    public static func effectSizes(fromCSV text: String) -> [EffectSizeRow]? {
        guard let table = csv(text) else { return nil }
        let index = headerIndex(table.header)
        guard
            let conditionIndex = index["condition"],
            let metricIndex = index["metric"] ?? index["endpoint"],
            let meanIndex = index["meandiff"] ?? index["deltamean"],
            let lowerIndex = index["cilower"],
            let upperIndex = index["ciupper"]
        else { return nil }
        return table.rows.compactMap { row in
            if let stratifyIndex = index["stratifyby"],
                let scope = field(row, stratifyIndex), scope != "pooled"
            {
                return nil
            }
            guard
                let condition = field(row, conditionIndex),
                let metric = field(row, metricIndex),
                let mean = double(row, meanIndex),
                let lower = double(row, lowerIndex),
                let upper = double(row, upperIndex)
            else { return nil }
            return EffectSizeRow(
                condition: condition,
                metric: metric,
                n: index["n"].flatMap { double(row, $0) }.map(Int.init) ?? 0,
                meanDiff: mean,
                ciLower: lower,
                ciUpper: upper,
                wilcoxonW: index["wilcoxonw"].flatMap { double(row, $0) },
                wilcoxonP: index["wilcoxonp"].flatMap { double(row, $0) },
                adjustedP: index["adjustedp"].flatMap { double(row, $0) },
                correction: index["correction"].flatMap { field(row, $0) },
                modality: index["modality"].flatMap { field(row, $0) })
        }
    }

    /// One report.json `effectSizes` entry (Swift study runs stamp these
    /// inline; keys are the Codable contract of `EffectSizeEntry`).
    static func effectSizeRow(fromJSON entry: [String: Any]) -> EffectSizeRow? {
        guard
            let condition = entry["condition"] as? String,
            let metric = entry["metric"] as? String,
            let mean = (entry["meanDiff"] as? NSNumber)?.doubleValue,
            let lower = (entry["ciLower"] as? NSNumber)?.doubleValue,
            let upper = (entry["ciUpper"] as? NSNumber)?.doubleValue
        else { return nil }
        return EffectSizeRow(
            condition: condition,
            metric: metric,
            n: (entry["n"] as? NSNumber)?.intValue ?? 0,
            meanDiff: mean,
            ciLower: lower,
            ciUpper: upper,
            wilcoxonW: (entry["wilcoxonW"] as? NSNumber)?.doubleValue,
            wilcoxonP: (entry["wilcoxonP"] as? NSNumber)?.doubleValue,
            adjustedP: (entry["adjustedP"] as? NSNumber)?.doubleValue,
            correction: entry["correction"] as? String,
            modality: nil)
    }

    // MARK: - alien-residuals.csv (server analyze)

    public struct AlienResidualRow: Identifiable, Sendable, Equatable {
        public var condition: String
        public var endpoint: String
        public var deltaModel: Double
        public var deltaHuman: Double
        /// The headline quantity R = delta_model − delta_human.
        public var r: Double
        public var ciRLower: Double?
        public var ciRUpper: Double?
        /// Region classification the server stamps (alien / humanAligned /
        /// hyperHuman / hypoHuman / inverted / inertBoth).
        public var region: String
        public var id: String { "\(condition)\u{1F}\(endpoint)" }
    }

    public static func alienResiduals(fromCSV text: String) -> [AlienResidualRow]? {
        guard let table = csv(text) else { return nil }
        let index = headerIndex(table.header)
        guard
            let conditionIndex = index["condition"],
            let endpointIndex = index["endpoint"],
            let modelIndex = index["deltamodel"],
            let humanIndex = index["deltahuman"],
            let rIndex = index["r"],
            let regionIndex = index["region"]
        else { return nil }
        return table.rows.compactMap { row in
            guard
                let condition = field(row, conditionIndex),
                let endpoint = field(row, endpointIndex),
                let deltaModel = double(row, modelIndex),
                let deltaHuman = double(row, humanIndex),
                let r = double(row, rIndex),
                let region = field(row, regionIndex)
            else { return nil }
            return AlienResidualRow(
                condition: condition, endpoint: endpoint,
                deltaModel: deltaModel, deltaHuman: deltaHuman, r: r,
                ciRLower: index["cirlower"].flatMap { double(row, $0) },
                ciRUpper: index["cirupper"].flatMap { double(row, $0) },
                region: region)
        }
    }

    // MARK: - promoted-movers.json (server analyze, screen phase)

    public struct PromotedMover: Identifiable, Sendable, Equatable {
        public var concept: String
        public var condition: String?
        public var endpoint: String?
        public var effectEstimate: Double?
        public var adjustedP: Double?
        public var promoted: Bool
        /// Failure/success reasons the server documents per decision — the
        /// funnel is only defensible if rejections are as documented as
        /// promotions.
        public var reasons: [String]
        public var id: String { concept }
    }

    public struct PromotedMovers: Sendable, Equatable {
        public var experiment: String?
        public var promoted: [PromotedMover]
        public var rejected: [PromotedMover]
    }

    public static func promotedMovers(fromJSON data: Data) -> PromotedMovers? {
        guard
            let object = try? JSONSerialization.jsonObject(with: data),
            let dictionary = object as? [String: Any]
        else { return nil }
        func movers(_ key: String) -> [PromotedMover]? {
            guard let entries = dictionary[key] as? [[String: Any]] else { return nil }
            return entries.compactMap { entry in
                guard let concept = entry["concept"] as? String else { return nil }
                return PromotedMover(
                    concept: concept,
                    condition: entry["condition"] as? String,
                    endpoint: entry["endpoint"] as? String,
                    effectEstimate: (entry["effectEstimate"] as? NSNumber)?.doubleValue,
                    adjustedP: (entry["adjustedP"] as? NSNumber)?.doubleValue,
                    promoted: (entry["promoted"] as? NSNumber)?.boolValue
                        ?? (key == "promoted"),
                    reasons: (entry["reasons"] as? [Any])?
                        .compactMap { $0 as? String } ?? [])
            }
        }
        guard let promoted = movers("promoted"), let rejected = movers("rejected")
        else { return nil }
        return PromotedMovers(
            experiment: dictionary["experiment"] as? String,
            promoted: promoted, rejected: rejected)
    }

    // MARK: - panel-effects.csv (multi-agent runs; A14 semantic viewer)

    /// One endpoint row of the panel-effect decomposition the SERVER's
    /// multi-agent study runs write (`panel_effects.py`; the Swift engine
    /// does not write this file today — its multi-agent runs produce
    /// transcripts without the paired decomposition). Column contract:
    /// `endpoint,direct,directN,spillover,spilloverN,group,groupN,
    /// transmissionRatio,amplification,droppedTurns`; NaN estimands are
    /// written as EMPTY cells, decoded here as nil ("no paired turns of
    /// that kind"), never coerced to a number.
    public struct PanelEffectRow: Identifiable, Sendable, Equatable {
        public var endpoint: String
        /// Treated seats' shift vs their own baseline turns.
        public var direct: Double?
        public var directN: Int
        /// Untreated seats' shift while sharing a panel with treated seats.
        public var spillover: Double?
        public var spilloverN: Int
        /// The designated group-outcome turn's shift (panel disposition).
        public var group: Double?
        public var groupN: Int
        /// spillover / direct — how much of the injected stance leaks
        /// through deliberation.
        public var transmissionRatio: Double?
        /// group / direct — does deliberation amplify or damp the treated
        /// seat's shift?
        public var amplification: Double?
        /// Turns dropped from the estimand (no baseline counterpart, or an
        /// endpoint that failed to parse in either condition).
        public var droppedTurns: Int

        public var id: String { endpoint }
    }

    /// Parse panel-effects.csv. Header-indexed and case-insensitive, so a
    /// column reorder or a future engine dialect with extra columns still
    /// reads; returns nil when the header lacks the endpoint column
    /// (not a panel-effects file at all).
    public static func panelEffects(fromCSV text: String) -> [PanelEffectRow]? {
        guard let table = csv(text) else { return nil }
        let index = headerIndex(table.header)
        guard let endpointIndex = index["endpoint"] else { return nil }
        return table.rows.compactMap { row in
            guard let endpoint = field(row, endpointIndex) else { return nil }
            func count(_ key: String) -> Int {
                index[key].flatMap { double(row, $0) }.map(Int.init) ?? 0
            }
            return PanelEffectRow(
                endpoint: endpoint,
                direct: index["direct"].flatMap { double(row, $0) },
                directN: count("directn"),
                spillover: index["spillover"].flatMap { double(row, $0) },
                spilloverN: count("spillovern"),
                group: index["group"].flatMap { double(row, $0) },
                groupN: count("groupn"),
                transmissionRatio: index["transmissionratio"].flatMap {
                    double(row, $0)
                },
                amplification: index["amplification"].flatMap { double(row, $0) },
                droppedTurns: count("droppedturns"))
        }
    }

    // MARK: - cosine-matrix.csv (validate runs; A13 discriminant view)

    public struct CosineMatrix: Sendable, Equatable {
        public var concepts: [String]
        /// Row-major cosine values; nil for unparseable cells ("nan").
        public var values: [[Double?]]
        /// The layer every cell was measured at. Nil for a legacy matrix
        /// written before the column existed — a cosine without its depth
        /// cannot be compared to another, since the residual stream drifts
        /// (the same concept a few layers apart can be near-orthogonal to
        /// itself), so an unlabelled matrix says so rather than implying one.
        public var layer: Int?
        /// Rows recorded DIFFERENT layers, i.e. the matrix is asymmetric and
        /// has no defined reading. Should be impossible since 2026-07-26;
        /// surfaced rather than averaged over.
        public var hasMixedLayers: Bool = false
        /// Off-diagonal flag threshold: distinct concepts collapsing into
        /// one direction is the discriminant-validity failure mode.
        public static let flagThreshold = 0.5

        public func isFlagged(row: Int, column: Int) -> Bool {
            guard row != column, let value = values[row][column] else { return false }
            return abs(value) > Self.flagThreshold
        }
    }

    /// Parse either engine's cosine-matrix.csv: Swift writes
    /// `concept,layer,<names…>`; the server writes `concept,<names…>`.
    public static func cosineMatrix(fromCSV text: String) -> CosineMatrix? {
        guard let table = csv(text), table.header.count >= 2 else { return nil }
        let hasLayerColumn = table.header.count > 1 && table.header[1] == "layer"
        let nameStart = hasLayerColumn ? 2 : 1
        let concepts = Array(table.header.dropFirst(nameStart))
        guard !concepts.isEmpty else { return nil }
        var values: [[Double?]] = []
        var rowNames: [String] = []
        var layers: Set<Int> = []
        for row in table.rows {
            guard row.count == concepts.count + nameStart, let name = row.first
            else { continue }
            rowNames.append(name)
            if hasLayerColumn, row.count > 1, let layer = Int(row[1]) {
                layers.insert(layer)
            }
            // "nan" cells (the server writes them for degenerate vectors)
            // parse to Double.nan — treat non-finite as unparseable.
            values.append(
                row.dropFirst(nameStart).map { cell in
                    Double(cell).flatMap { $0.isFinite ? $0 : nil }
                })
        }
        guard rowNames == concepts, values.count == concepts.count else { return nil }
        return CosineMatrix(
            concepts: concepts, values: values,
            layer: layers.count == 1 ? layers.first : nil,
            hasMixedLayers: layers.count > 1)
    }

    // MARK: - CSV plumbing (full parse, unlike the bounded previews)

    /// Whole-text CSV split reusing `RunBrowser`'s quote-aware line split.
    /// Analysis artifacts are small by construction (one row per condition ×
    /// endpoint); callers still bound the read at the file layer.
    static func csv(_ text: String) -> (header: [String], rows: [[String]])? {
        let lines = text.split(separator: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard let headerLine = lines.first else { return nil }
        return (
            RunBrowser.splitCSVLine(headerLine),
            lines.dropFirst().map(RunBrowser.splitCSVLine))
    }

    private static func headerIndex(_ header: [String]) -> [String: Int] {
        var index: [String: Int] = [:]
        for (offset, name) in header.enumerated() {
            index[name.lowercased()] = offset
        }
        return index
    }

    private static func field(_ row: [String], _ index: Int) -> String? {
        guard row.indices.contains(index) else { return nil }
        let value = row[index]
        return value.isEmpty ? nil : value
    }

    private static func double(_ row: [String], _ index: Int) -> Double? {
        guard let value = field(row, index).flatMap(Double.init),
            value.isFinite
        else { return nil }
        return value
    }
}
