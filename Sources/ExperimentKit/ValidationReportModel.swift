import Foundation

/// Structured decode of a validate run's `validation-report.json` (app gap
/// A13 — previously rendered as raw pretty-printed JSON). Both engines'
/// dialects are read into one model:
///
/// - Swift: `{"validation": {name: {scenarios, layer, accuracy} | "<skip
///   message>"}, "logitLens": {name: {layer, topPositive: [{token,…}],
///   topNegative: […]}}, "worstCosinePair": "a × b = c",
///   "capabilityBattery": [{condition, accuracy, correct, total}]}`
/// - Server: `{"concepts": {name: {layer, scenarioCount, labeled,
///   scenarioAccuracy | fractionAboveMidpoint, note}}}`
///
/// Convergent accuracy is read against the 0.5 chance line (both engines'
/// scenario scoring is a balanced two-class read).
extension RunResults {

    public struct ValidationConceptRow: Identifiable, Sendable, Equatable {
        public var concept: String
        public var layer: Int?
        public var scenarios: Int?
        /// Held-out TRANSFER accuracy (0…1) — does the extraction-derived
        /// threshold carry over; nil when the gate did not run (see `note`)
        /// or the file was unlabeled. Read `calibratedAccuracy`/`auc` for
        /// the vector-quality question (review 2026-08-01: for
        /// designatedReference recipes this number sits structurally near
        /// 0.5 — a threshold artifact the flags below diagnose).
        public var accuracy: Double?
        /// Server fallback for unlabeled legacy validation files.
        public var fractionAboveMidpoint: Double?
        /// Threshold-free ranking separation (diagnostics.auc), when the
        /// report carries diagnostics.
        public var auc: Double?
        /// Separation at the held-out classes' own midpoint
        /// (diagnostics.heldOutCalibration.accuracy).
        public var calibratedAccuracy: Double?
        /// The transfer threshold put every item on one side — `accuracy`
        /// measured the threshold, not the vector.
        public var oneSided: Bool?
        /// Skip message / unlabeled-file note, verbatim.
        public var note: String?
        /// Layer-qualified so a multi-depth report (one row per declared
        /// depth) keeps row identity unique; single-depth rows are qualified
        /// too, harmlessly.
        public var id: String { layer.map { "\(concept)@L\($0)" } ?? concept }

        /// Chance line for the convergent read.
        public static let chanceAccuracy = 0.5
        public var beatsChance: Bool? {
            accuracy.map { $0 > Self.chanceAccuracy }
        }
        /// The number a reader should trust first: calibrated when present,
        /// else the transfer accuracy.
        public var headlineAccuracy: Double? { calibratedAccuracy ?? accuracy }
    }

    public struct LogitLensRow: Identifiable, Sendable, Equatable {
        public var concept: String
        public var layer: Int?
        public var topPositive: [String]
        public var topNegative: [String]
        /// Present when the lens was skipped (the report stores a message).
        public var note: String?
        public var id: String { layer.map { "\(concept)@L\($0)" } ?? concept }
    }

    public struct ValidationBatteryRow: Identifiable, Sendable, Equatable {
        public var condition: String
        public var accuracy: Double?
        public var correct: Int?
        public var total: Int?
        public var id: String { condition }
    }

    public struct ValidationReport: Sendable, Equatable {
        public var experiment: String?
        public var concepts: [ValidationConceptRow]
        public var logitLens: [LogitLensRow]
        public var worstCosinePair: String?
        public var capabilityBattery: [ValidationBatteryRow]
        /// The layer the cosine matrix was measured at (both engines stamp
        /// `cosineMatrixLayer`). Nil for reports written before the stamp.
        public var cosineMatrixLayer: Int?
        /// Every matrix layer, for multi-depth reports (one matrix per
        /// declared depth). Nil for pre-list reports.
        public var cosineMatrixLayers: [Int]?
    }

    /// Decode either engine's validation report; nil when the JSON carries
    /// neither dialect's concept block (not a validation report).
    public static func validationReport(fromJSON data: Data) -> ValidationReport? {
        guard
            let object = try? JSONSerialization.jsonObject(with: data),
            let dictionary = object as? [String: Any]
        else { return nil }
        let conceptBlock =
            dictionary["validation"] as? [String: Any]
            ?? dictionary["concepts"] as? [String: Any]
        guard let conceptBlock else { return nil }

        let concepts = conceptBlock.keys.sorted().flatMap { name in
            conceptRows(name: name, raw: conceptBlock[name])
        }

        var lens: [LogitLensRow] = []
        if let lensBlock = dictionary["logitLens"] as? [String: Any] {
            lens = lensBlock.keys.sorted().flatMap { name -> [LogitLensRow] in
                // Multi-depth reports store a LIST of lens blocks per
                // concept — one per declared depth.
                if let blocks = lensBlock[name] as? [Any] {
                    return blocks.map { logitLensRow(name: name, raw: $0) }
                }
                return [logitLensRow(name: name, raw: lensBlock[name])]
            }
        }

        var battery: [ValidationBatteryRow] = []
        if let rows = dictionary["capabilityBattery"] as? [[String: Any]] {
            battery = rows.compactMap { row in
                guard let condition = row["condition"] as? String else { return nil }
                return ValidationBatteryRow(
                    condition: condition,
                    accuracy: (row["accuracy"] as? NSNumber)?.doubleValue,
                    correct: (row["correct"] as? NSNumber)?.intValue,
                    total: (row["total"] as? NSNumber)?.intValue)
            }
        }

        return ValidationReport(
            experiment: dictionary["experiment"] as? String,
            concepts: concepts,
            logitLens: lens,
            worstCosinePair: dictionary["worstCosinePair"] as? String,
            capabilityBattery: battery,
            cosineMatrixLayer: dictionary["cosineMatrixLayer"] as? Int,
            cosineMatrixLayers: (dictionary["cosineMatrixLayers"] as? [NSNumber])?
                .map(\.intValue))
    }

    /// One row per declared depth. A multi-depth entry (declared
    /// `validationLayers[Fractions]`, 2026-08-01) carries its per-depth
    /// sub-entries under `depths` and deliberately NO flat mirror; a
    /// single-depth or pre-list entry is its own only row.
    private static func conceptRows(
        name: String, raw: Any?
    ) -> [ValidationConceptRow] {
        guard let block = raw as? [String: Any],
            let depths = block["depths"] as? [[String: Any]],
            depths.count > 1
        else {
            return [conceptRow(name: name, raw: raw)]
        }
        let scenarios =
            (block["scenarios"] as? NSNumber)?.intValue
            ?? (block["scenarioCount"] as? NSNumber)?.intValue
        return depths.map { sub in
            var row = conceptRow(name: name, raw: sub)
            row.scenarios = row.scenarios ?? scenarios
            return row
        }
    }

    private static func conceptRow(name: String, raw: Any?) -> ValidationConceptRow {
        var row = ValidationConceptRow(
            concept: name, layer: nil, scenarios: nil, accuracy: nil,
            fractionAboveMidpoint: nil, auc: nil, calibratedAccuracy: nil,
            oneSided: nil, note: nil)
        if let message = raw as? String {
            // Swift skip messages ("no validation.jsonl — convergent gate
            // NOT run") are stored as bare strings.
            row.note = message
            return row
        }
        guard let block = raw as? [String: Any] else { return row }
        row.layer = (block["layer"] as? NSNumber)?.intValue
        row.scenarios =
            (block["scenarios"] as? NSNumber)?.intValue
            ?? (block["scenarioCount"] as? NSNumber)?.intValue
        row.accuracy =
            (block["accuracy"] as? NSNumber)?.doubleValue
            ?? (block["scenarioAccuracy"] as? NSNumber)?.doubleValue
        row.fractionAboveMidpoint =
            (block["fractionAboveMidpoint"] as? NSNumber)?.doubleValue
        // D1 diagnostics (both engines write the same block since
        // 2026-08-01): the threshold-free and threshold-recalibrated
        // numbers that tell "dead vector" from "misplaced midpoint".
        if let diagnostics = block["diagnostics"] as? [String: Any] {
            row.auc = (diagnostics["auc"] as? NSNumber)?.doubleValue
            row.oneSided = diagnostics["oneSidedPredictions"] as? Bool
            if let calibration =
                diagnostics["heldOutCalibration"] as? [String: Any]
            {
                row.calibratedAccuracy =
                    (calibration["accuracy"] as? NSNumber)?.doubleValue
            }
        }
        row.note = block["note"] as? String
        return row
    }

    private static func logitLensRow(name: String, raw: Any?) -> LogitLensRow {
        var row = LogitLensRow(
            concept: name, layer: nil, topPositive: [], topNegative: [], note: nil)
        if let message = raw as? String {
            row.note = message
            return row
        }
        guard let block = raw as? [String: Any] else { return row }
        row.layer = (block["layer"] as? NSNumber)?.intValue
        func tokens(_ key: String) -> [String] {
            guard let entries = block[key] as? [[String: Any]] else { return [] }
            return entries.compactMap { $0["token"] as? String }
        }
        row.topPositive = tokens("topPositive")
        row.topNegative = tokens("topNegative")
        return row
    }
}
