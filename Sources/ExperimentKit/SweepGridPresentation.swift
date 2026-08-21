import Foundation

/// A sweep run's grid, arranged for display (E3).
///
/// `SweepRunCatalog` already parses `sweep.csv` and already knows each cell's
/// state; what was missing was anything that turned those into a table a
/// researcher could read. The scratch `scripts/run-viewer.py` filled that gap
/// with a heatmap, struck-through constraint failures, and a control-status
/// line — this brings the same content into the app, so the scratch tool can
/// retire rather than becoming a fourth UI surface.
///
/// Pure and unit-tested: the view renders exactly what this returns.
public enum SweepGridPresentation {

    public struct Cell: Sendable, Equatable, Identifiable {
        public var layer: Int
        public var alpha: Double
        /// The objective's value for this cell — the declared objective when
        /// the sweep recorded one, else marker density.
        public var score: Double
        public var distinct2: Double
        public var batteryAccuracy: Double
        public var state: SweepRunCatalog.CellState

        public var id: String { "\(layer)-\(alpha)" }

        /// Position in the colour ramp, 0…1, against the grid's own range.
        /// Nil for cells that failed a constraint — shading a cell that will
        /// never be selected invites reading it as a contender.
        public var intensity: Double?

        /// A constraint-failed cell is struck through: it was measured, and
        /// it is not eligible. Hiding it would misrepresent the sweep;
        /// showing it plain would misrepresent the result.
        public var isStruckThrough: Bool { state == .failedConstraint }
    }

    public struct Grid: Sendable, Equatable {
        public var concept: String
        public var layers: [Int]
        public var alphas: [Double]
        public var cells: [Cell]
        public var baseline: Cell?
        public var objective: String
        /// What the sweep concluded — a winning cell, or the reason there was
        /// none. A sweep that selected nothing is a RESULT, not an empty grid.
        public var outcome: Outcome
        public var control: ControlStatus

        public func cell(layer: Int, alpha: Double) -> Cell? {
            cells.first { $0.layer == layer && abs($0.alpha - alpha) < 1e-9 }
        }
    }

    public enum Outcome: Sendable, Equatable {
        case selected(layer: Int, alpha: Double, score: Double?)
        case noRecommendation(reason: String)
        case notRecorded

        public var detail: String {
            switch self {
            case .selected(let layer, let alpha, let score):
                let cell = "L\(layer) α"
                    + alpha.formatted(.number.precision(.fractionLength(0 ... 4)))
                guard let score else { return "winner: \(cell)" }
                return "winner: \(cell) — objective "
                    + score.formatted(.number.precision(.fractionLength(0 ... 4)))
            case .noRecommendation(let reason):
                return "no cell was selected: \(reason)"
            case .notRecorded:
                return "this run recorded no recommendation for the concept"
            }
        }
    }

    /// Whether the winner was tested against a norm-matched random direction.
    public enum ControlStatus: Sendable, Equatable {
        case passed(margin: Double, metricValue: Double)
        case absent

        public var detail: String {
            switch self {
            case .passed(let margin, let value):
                "matched-norm random control passed: random direction scored "
                    + value.formatted(.number.precision(.fractionLength(0 ... 4)))
                    + ", required margin "
                    + margin.formatted(.number.precision(.fractionLength(0 ... 4)))
            case .absent:
                "NO matched-norm random control ran — the winner's margin over "
                    + "a random direction of the same norm is unmeasured, so "
                    + "this cell is exploratory rather than evidence-grade"
            }
        }

        public var isDeclared: Bool {
            if case .passed = self { return true }
            return false
        }
    }

    /// Build the grid for one concept from a parsed sweep run.
    public static func grid(
        concept: String,
        rows: [SweepRunCatalog.Row],
        recommendation: SweepRunCatalog.Recommendation?,
        criterion: SweepRunCatalog.Row? = nil
    ) -> Grid {
        let conceptRows = rows.filter { $0.concept == concept }
        let baselineRow = conceptRows.first(where: \.isBaseline)
        let gridRows = conceptRows.filter { !$0.isBaseline }

        var resolvedCriterion = SweepSelectionRule.Resolved(
            metric: "markerDensity",
            capabilityTolerance: SweepSelectionRule.defaultCapabilityTolerance,
            coherenceFloor: SweepSelectionRule.defaultCoherenceFloor,
            matchedNormRandomMargin: nil)
        var winner: ExperimentManifest.SelectionProvenance.Cell?
        var outcome: Outcome = .notRecorded
        var control: ControlStatus = .absent

        switch recommendation {
        case .selected(let provenance):
            winner = provenance.winningCell
            if let resolved = try? SweepSelectionRule.resolve(provenance.criterion) {
                resolvedCriterion = resolved
            }
            let metric = provenance.metrics[resolvedCriterion.metric]
            outcome = .selected(
                layer: provenance.winningCell.layer,
                alpha: provenance.winningCell.alpha,
                score: metric)
            if let recordedControl = provenance.control {
                control = .passed(
                    margin: recordedControl.margin,
                    metricValue: recordedControl.metricValue)
            }
        case .failure(let message):
            outcome = .noRecommendation(reason: message)
        case nil:
            outcome = .notRecorded
        }

        func score(_ row: SweepRunCatalog.Row) -> Double {
            row.objective ?? row.markerDensity
        }

        // The ramp spans only ELIGIBLE cells: a constraint-failed cell's
        // score is not a contender, and letting it set the range would
        // wash out the cells that are.
        let eligible = gridRows.filter {
            SweepRunCatalog.cellState(
                row: $0, baseline: baselineRow, criterion: resolvedCriterion,
                winner: winner) != .failedConstraint
        }
        let scores = eligible.map(score)
        let low = scores.min()
        let high = scores.max()

        let cells = gridRows.map { row -> Cell in
            let state = SweepRunCatalog.cellState(
                row: row, baseline: baselineRow, criterion: resolvedCriterion,
                winner: winner)
            var intensity: Double?
            if state != .failedConstraint, let low, let high {
                intensity = high > low ? (score(row) - low) / (high - low) : 0.5
            }
            return Cell(
                layer: row.layer, alpha: row.alpha, score: score(row),
                distinct2: row.distinct2, batteryAccuracy: row.batteryAccuracy,
                state: state, intensity: intensity)
        }

        return Grid(
            concept: concept,
            layers: Array(Set(gridRows.map(\.layer))).sorted(),
            alphas: Array(Set(gridRows.map(\.alpha))).sorted(),
            cells: cells,
            baseline: baselineRow.map { row in
                Cell(
                    layer: row.layer, alpha: row.alpha, score: score(row),
                    distinct2: row.distinct2,
                    batteryAccuracy: row.batteryAccuracy,
                    state: .baseline, intensity: nil)
            },
            objective: resolvedCriterion.metric,
            outcome: outcome,
            control: control)
    }

    /// Every concept a sweep run covers, in a stable order.
    public static func concepts(rows: [SweepRunCatalog.Row]) -> [String] {
        Array(Set(rows.map(\.concept))).sorted()
    }
}
