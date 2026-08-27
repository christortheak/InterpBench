import ExperimentKit
import SwiftUI

/// A sweep run's grid, in the app (E3).
///
/// This content existed only in `scripts/run-viewer.py`, a scratch tool built
/// to read a sweep the app could not display. Bringing it here lets that tool
/// retire instead of becoming a fourth UI surface.
///
/// Everything shown comes from `SweepGridPresentation`, which is pure and
/// unit-tested; this view only draws it.
struct SweepGridView: View {
    let grid: SweepGridPresentation.Grid

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            header
            table
            footer
        }
    }

    @ViewBuilder
    private var header: some View {
        Text(grid.concept)
            .font(.callout.weight(.medium))
        Text("objective: \(grid.objective)")
            .font(.caption2)
            .foregroundStyle(.secondary)
    }

    @ViewBuilder
    private var table: some View {
        Grid(alignment: .trailing, horizontalSpacing: 6, verticalSpacing: 3) {
            GridRow {
                Text("L \\ α").font(.caption2).foregroundStyle(.secondary)
                ForEach(grid.alphas, id: \.self) { alpha in
                    Text(format(alpha)).font(.caption2).foregroundStyle(.secondary)
                }
            }
            ForEach(grid.layers, id: \.self) { layer in
                GridRow {
                    Text("L\(layer)").font(.caption2).foregroundStyle(.secondary)
                    ForEach(grid.alphas, id: \.self) { alpha in
                        cellView(grid.cell(layer: layer, alpha: alpha))
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func cellView(_ cell: SweepGridPresentation.Cell?) -> some View {
        if let cell {
            Text(format(cell.score))
                .font(.caption2.monospacedDigit())
                // Measured but ineligible: hiding it would misrepresent the
                // sweep, showing it plain would misrepresent the result.
                .strikethrough(cell.isStruckThrough)
                .foregroundStyle(cell.isStruckThrough ? Color.secondary : Color.primary)
                .padding(.horizontal, 4)
                .padding(.vertical, 2)
                .background(background(cell))
                .overlay(
                    RoundedRectangle(cornerRadius: 3)
                        .strokeBorder(
                            cell.state == .winner ? Color.accentColor : .clear,
                            lineWidth: 1.5))
                .help(tooltip(cell))
        } else {
            Text("—").font(.caption2).foregroundStyle(.tertiary)
        }
    }

    private func background(_ cell: SweepGridPresentation.Cell) -> Color {
        guard let intensity = cell.intensity else { return .clear }
        return Color.accentColor.opacity(0.08 + 0.42 * intensity)
    }

    private func tooltip(_ cell: SweepGridPresentation.Cell) -> String {
        let score = format(cell.score)
        let distinct = format(cell.distinct2)
        let battery = format(cell.batteryAccuracy)
        let state =
            switch cell.state {
            case .winner: "selected by the declared criterion"
            case .failedConstraint:
                "FAILED a constraint (capability tolerance or coherence "
                    + "floor) — measured, but never eligible for selection"
            case .baseline: "no-injection baseline"
            case .pass: "eligible"
            }
        let ratio = cell.distinct2Ratio.map { " (\(format($0))× baseline)" } ?? ""
        let length = cell.lengthInflated
            ? ", ⚠︎ output over 1.5× baseline length" : ""
        return "L\(cell.layer) α\(format(cell.alpha)): objective \(score), "
            + "distinct-2 \(distinct)\(ratio), battery \(battery)\(length) "
            + "— \(state)"
    }

    @ViewBuilder
    private var footer: some View {
        if let baseline = grid.baseline {
            Text("baseline (no injection): \(format(baseline.score))")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        Text(grid.outcome.detail)
            .font(.caption)
            .fixedSize(horizontal: false, vertical: true)
            .textSelection(.enabled)
        // The control's ABSENCE is a fact about the evidence, not a blank.
        Label(
            grid.control.detail,
            systemImage: grid.control.isDeclared
                ? "checkmark.shield" : "exclamationmark.shield")
            .font(.caption)
            .foregroundStyle(grid.control.isDeclared ? Color.secondary : Color.orange)
            .fixedSize(horizontal: false, vertical: true)
            .textSelection(.enabled)
    }

    private func format(_ value: Double) -> String {
        value.formatted(.number.precision(.fractionLength(0 ... 4)))
    }
}
