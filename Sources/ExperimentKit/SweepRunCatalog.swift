import Foundation

/// Discovery and parsing of sweep RUN artifacts for the Optimizations
/// (Agents region) surface.
///
/// A sweep run directory (`runs/<stamp>-exp-<experiment>-sweep[-N]/`) carries
/// `sweep.csv` (required columns: concept,layer,alpha,markerDensity,distinct2
/// plus batteryAccuracy and — for judgeScore/logprobShift sweeps — the
/// declared `objective`; extra columns like the server's `words` are ignored;
/// the baseline row has layer -1, alpha 0) and
/// `recommendations.json` (per concept: either a `SelectionProvenance` object
/// or a failure-message string). Parsing is pure — fixture strings drive the
/// unit tests; only the discovery helpers touch the filesystem.
public enum SweepRunCatalog {

    // MARK: CSV rows

    public struct Row: Sendable, Equatable {
        public var concept: String
        public var layer: Int
        public var alpha: Double
        public var markerDensity: Double
        public var distinct2: Double
        public var batteryAccuracy: Double
        /// The declared selection objective's value for this cell — both
        /// engines append an `objective` column when the sweep's objective is
        /// not markerDensity (judgeScore / logprobShift). nil for
        /// markerDensity sweeps and for runs that predate the column; marker
        /// density then remains the only displayable expression score.
        public var objective: Double?

        /// The no-injection anchor row (written with layer -1, alpha 0).
        public var isBaseline: Bool { layer < 0 }

        public init(
            concept: String, layer: Int, alpha: Double,
            markerDensity: Double, distinct2: Double, batteryAccuracy: Double,
            objective: Double? = nil
        ) {
            self.concept = concept
            self.layer = layer
            self.alpha = alpha
            self.markerDensity = markerDensity
            self.distinct2 = distinct2
            self.batteryAccuracy = batteryAccuracy
            self.objective = objective
        }
    }

    /// The Swift writer's canonical column order. The PARSER is header-name
    /// driven (below) — it accepts any column order and ignores unknown extra
    /// columns, because the server's sweep.csv adds a `words` column and both
    /// engines' artifacts must load in the Optimizations UI.
    public static let csvHeader =
        "concept,layer,alpha,markerDensity,distinct2,batteryAccuracy"

    /// Columns every sweep.csv must carry, whatever engine wrote it.
    /// `batteryAccuracy` is expected but tolerated-absent (legacy files):
    /// rows then parse with a perfect-accuracy placeholder so the capability
    /// constraint can never fire on data that was never measured.
    static let requiredCSVColumns = [
        "concept", "layer", "alpha", "markerDensity", "distinct2",
    ]

    public static func parseCSV(_ text: String) throws -> [Row] {
        // `whereSeparator: \.isNewline` — NEVER a literal "\n" split
        // (field incident 2026-08-04): the server's csv module writes CRLF,
        // and Swift treats "\r\n" as ONE grapheme equal to neither "\n"
        // nor "\r" — a literal split saw server sweep CSVs as a single
        // header line and every server-executed sweep grid rendered EMPTY,
        // silently.
        var lines = text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard let header = lines.first else {
            throw ExperimentError(reason: "sweep.csv is empty")
        }
        let columns = header.split(separator: ",", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        var index: [String: Int] = [:]
        for (position, name) in columns.enumerated() where index[name] == nil {
            index[name] = position
        }
        let missing = requiredCSVColumns.filter { index[$0] == nil }
        guard missing.isEmpty,
            let conceptIndex = index["concept"],
            let layerIndex = index["layer"],
            let alphaIndex = index["alpha"],
            let densityIndex = index["markerDensity"],
            let distinctIndex = index["distinct2"]
        else {
            throw ExperimentError(
                reason: "sweep.csv header is missing required column(s) "
                    + missing.joined(separator: ", ")
                    + " — found '\(header)'")
        }
        let accuracyIndex = index["batteryAccuracy"]
        // Tolerant by design: the column exists only for non-markerDensity
        // objectives (and not at all in pre-objective runs), and a viewer
        // must render whatever a run recorded — absent or unparseable
        // objective values become nil, never a load failure.
        let objectiveIndex = index["objective"]
        lines.removeFirst()

        var rows: [Row] = []
        for line in lines {
            let fields = line.split(separator: ",", omittingEmptySubsequences: false)
                .map(String.init)
            guard fields.count == columns.count,
                let layer = Int(fields[layerIndex]),
                let alpha = Double(fields[alphaIndex]),
                let density = Double(fields[densityIndex]),
                let distinct = Double(fields[distinctIndex])
            else {
                throw ExperimentError(reason: "malformed sweep.csv line: \(line)")
            }
            let accuracy: Double
            if let accuracyIndex {
                guard let parsed = Double(fields[accuracyIndex]) else {
                    throw ExperimentError(reason: "malformed sweep.csv line: \(line)")
                }
                accuracy = parsed
            } else {
                accuracy = 1.0  // legacy file without a battery column
            }
            rows.append(
                Row(
                    concept: fields[conceptIndex], layer: layer, alpha: alpha,
                    markerDensity: density, distinct2: distinct,
                    batteryAccuracy: accuracy,
                    objective: objectiveIndex.flatMap { Double(fields[$0]) }))
        }
        return rows
    }

    /// Distinct concepts in appearance order (each concept's grid renders
    /// separately).
    public static func concepts(in rows: [Row]) -> [String] {
        var seen = Set<String>()
        var ordered: [String] = []
        for row in rows where !seen.contains(row.concept) {
            seen.insert(row.concept)
            ordered.append(row.concept)
        }
        return ordered
    }

    // MARK: Recommendations

    public enum Recommendation: Sendable, Equatable, Decodable {
        case selected(ExperimentManifest.SelectionProvenance)
        case failure(String)

        public init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let message = try? container.decode(String.self) {
                self = .failure(message)
                return
            }
            self = .selected(
                try container.decode(ExperimentManifest.SelectionProvenance.self))
        }
    }

    public static func parseRecommendations(
        _ data: Data
    ) throws -> [String: Recommendation] {
        try JSONDecoder().decode([String: Recommendation].self, from: data)
    }

    // MARK: Criterion for display

    /// Fill the documented defaults into a declared (or absent) selection
    /// block for DISPLAY and grid marking. Unlike `SweepSelectionRule.resolve`
    /// this never throws: resolve() fail-fasts on unimplemented metrics
    /// because it guards a run; a viewer must render whatever is declared.
    public static func displayCriterion(
        _ spec: ExperimentManifest.SweepSelection?
    ) -> SweepSelectionRule.Resolved {
        SweepSelectionRule.Resolved(
            metric: spec?.objective?.metric ?? "markerDensity",
            capabilityTolerance: spec?.constraints?.capabilityTolerance
                ?? SweepSelectionRule.defaultCapabilityTolerance,
            coherenceFloor: spec?.constraints?.coherenceFloor
                ?? SweepSelectionRule.defaultCoherenceFloor,
            matchedNormRandomMargin: spec?.controls?.matchedNormRandomMargin)
    }

    // MARK: Cell state under a declared criterion

    /// Rendering state for one grid cell — computed here (not in a view) so
    /// the constraint marking is unit-testable against the same rule the
    /// sweep applied (`SweepSelectionRule`).
    public enum CellState: String, Sendable {
        case baseline
        case winner
        case pass
        case failedConstraint
    }

    public static func cellState(
        row: Row,
        baseline: Row?,
        criterion: SweepSelectionRule.Resolved,
        winner: ExperimentManifest.SelectionProvenance.Cell?
    ) -> CellState {
        if row.isBaseline { return .baseline }
        if let winner, winner.layer == row.layer,
            abs(winner.alpha - row.alpha) < 1e-9
        {
            return .winner
        }
        if let baseline {
            let eligible =
                row.batteryAccuracy
                    >= baseline.batteryAccuracy - criterion.capabilityTolerance
                && row.distinct2 >= criterion.coherenceFloor
            if !eligible { return .failedConstraint }
        } else if row.distinct2 < criterion.coherenceFloor {
            // No baseline row: only the absolute coherence floor can judge.
            return .failedConstraint
        }
        return .pass
    }

    // MARK: Run discovery

    public struct SweepRun: Sendable {
        public var directory: URL
        public var rows: [Row]
        public var recommendations: [String: Recommendation]

        public var runName: String { directory.lastPathComponent }
    }

    /// True when a run-directory NAME belongs to this experiment's sweep
    /// task: `<stamp>-exp-<experiment>-sweep` plus the collision counter
    /// (`…-sweep-2`). Pure and strict — an experiment literally named
    /// "x-sweep" never captures experiment "x"'s runs, because the counter
    /// tail must be all digits.
    public static func directoryNameMatches(
        _ name: String, experiment: String
    ) -> Bool {
        let suffix = "-exp-\(experiment)-sweep"
        if name.hasSuffix(suffix) { return true }
        guard let range = name.range(of: suffix + "-", options: .backwards) else {
            return false
        }
        let tail = name[range.upperBound...]
        return !tail.isEmpty && tail.allSatisfy(\.isNumber)
    }

    /// Sweep run directories for an experiment, oldest → newest. Run
    /// directories are timestamp-prefixed, so lexicographic order is creation
    /// order. Each match must actually contain `sweep.csv`, and when an
    /// `experiment.json` snapshot is present its name must match (guards
    /// against name-shaped coincidences).
    public static func sweepRunDirectories(
        experiment: String,
        runsDirectory: URL = ExperimentStore.runsDirectory
    ) -> [URL] {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: runsDirectory,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
        else { return [] }
        return entries
            .filter { url in
                guard directoryNameMatches(url.lastPathComponent, experiment: experiment)
                else { return false }
                guard fm.fileExists(atPath: url.appending(component: "sweep.csv").path)
                else { return false }
                let snapshot = url.appending(component: "experiment.json")
                if let data = try? Data(contentsOf: snapshot),
                    let manifest = try? JSONDecoder().decode(
                        ExperimentManifest.self, from: data)
                {
                    return manifest.name == experiment
                }
                return true
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    public static func newestSweepRunDirectory(
        experiment: String,
        runsDirectory: URL = ExperimentStore.runsDirectory
    ) -> URL? {
        sweepRunDirectories(experiment: experiment, runsDirectory: runsDirectory).last
    }

    /// Loads a sweep run directory: `sweep.csv` is required;
    /// `recommendations.json` is optional (older runs may predate it).
    public static func load(directory: URL) throws -> SweepRun {
        let csvURL = directory.appending(component: "sweep.csv")
        guard let csvText = try? String(contentsOf: csvURL, encoding: .utf8) else {
            throw ExperimentError(reason: "no sweep.csv in \(directory.path)")
        }
        let rows = try parseCSV(csvText)
        var recommendations: [String: Recommendation] = [:]
        let recURL = directory.appending(component: "recommendations.json")
        if let data = try? Data(contentsOf: recURL) {
            recommendations = try parseRecommendations(data)
        }
        return SweepRun(
            directory: directory, rows: rows, recommendations: recommendations)
    }

    public static func newestSweepRun(
        experiment: String,
        runsDirectory: URL = ExperimentStore.runsDirectory
    ) -> SweepRun? {
        guard
            let directory = newestSweepRunDirectory(
                experiment: experiment, runsDirectory: runsDirectory)
        else { return nil }
        return try? load(directory: directory)
    }
}
