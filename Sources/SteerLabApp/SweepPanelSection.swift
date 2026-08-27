import ExperimentKit
import SwiftUI

/// The one place a sweep's facts appear together (E2).
///
/// Previously the grid lived in "New Agent", the criterion in Optimizations,
/// and the files the sweep opens were raw strings that might or might not
/// exist — so composing a sweep meant holding the layer fractions in one
/// screen, guessing what they resolved to, and taking the paths on faith.
///
/// Every value here comes from `SweepPanelModel`, which is pure and
/// unit-tested; this view only renders it. In particular the depth→layer
/// resolution uses CACHED vector metadata, never a model load.
struct SweepPanelSection: View {
    let resolved: SweepPanelModel.Resolved
    /// Opens the editor that owns these values, so the panel is a place to
    /// act from and not only to read.
    var editAction: (() -> Void)?

    var body: some View {
        Section("Sweep — what will run") {
            gridRows
            fileRows
            criterionRows
            if let editAction {
                Button("Edit sweep spec", action: editAction)
                    .buttonStyle(.link)
                    .font(.caption)
            }
        }
    }

    @ViewBuilder
    private var gridRows: some View {
        Text(resolved.gridSummary)
            .font(.caption)
            .fixedSize(horizontal: false, vertical: true)
            .textSelection(.enabled)
        if resolved.layerCount == nil {
            // Honest rather than blank: the fractions cannot be resolved
            // until something has been extracted for this model.
            Label(
                "extract a vector for this model to see which layers the "
                    + "fractions resolve to",
                systemImage: "questionmark.circle")
                .font(.caption2)
                .foregroundStyle(.secondary)
        } else if resolved.collapsedFractions > 0 {
            // A grid of "four depths" that is really three is a silently
            // smaller sweep.
            Label(collapsedLine, systemImage: "arrow.triangle.merge")
                .font(.caption2)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var collapsedLine: String {
        let n = resolved.collapsedFractions
        return "\(n) depth fraction\(n == 1 ? "" : "s") collapsed onto a layer "
            + "already in the grid — the sweep runs fewer cells than the "
            + "fractions suggest"
    }

    @ViewBuilder
    private var fileRows: some View {
        ForEach(resolved.files, id: \.label) { file in
            LabeledContent(file.label) {
                Text(file.detail)
                    .font(.caption2)
                    .foregroundStyle(
                        !file.exists || file.drifted ? Color.orange : Color.secondary)
                    .textSelection(.enabled)
            }
        }
    }

    /// Built outside the view builder: `ExperimentsPanelView` taught us that
    /// string arithmetic inside a SwiftUI body tips the type-checker over.
    private var constraintLine: String {
        let tolerance = resolved.capabilityTolerance.formatted(
            .number.precision(.fractionLength(0 ... 4)))
        let floor = resolved.coherenceFloor.formatted(
            .number.precision(.fractionLength(0 ... 4)))
        guard let ratio = resolved.coherenceRatioToBaseline else {
            return "capability tolerance \(tolerance) · coherence floor "
                + "\(floor) (absolute)"
        }
        let relative = ratio.formatted(.number.precision(.fractionLength(0 ... 4)))
        return "capability tolerance \(tolerance) · coherence floor "
            + "\(relative)× baseline · backstop \(floor)"
    }

    @ViewBuilder
    private var criterionRows: some View {
        LabeledContent("Objective") {
            Text(resolved.objective).font(.caption)
        }
        Text(constraintLine)
            .font(.caption2)
            .foregroundStyle(.secondary)
        // The control's ABSENCE is stated, never rendered as a blank field:
        // an empty box and "no control declared" look identical and mean very
        // different things.
        Label(
            resolved.control.detail,
            systemImage: resolved.control.isDeclared
                ? "checkmark.shield" : "exclamationmark.shield")
            .font(.caption)
            .foregroundStyle(resolved.control.isDeclared ? Color.secondary : Color.orange)
            .fixedSize(horizontal: false, vertical: true)
            .textSelection(.enabled)
    }
}
