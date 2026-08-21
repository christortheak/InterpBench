import ExperimentKit
import SwiftUI

/// Right-hand viewer for the Analysis section: renders the computed cosine
/// matrix (and layer RSA) from the shared `GeometryPanel`. Controls stay in
/// the Analysis pane; results render here, where all results are viewed
/// (live-testing finding — tables used to appear as extra form sections
/// underneath the Geometry controls).
struct GeometryViewerColumn: View {
    @Bindable var service: ChatService

    private var geometry: GeometryPanel { service.geometry }

    private var isServerWorkspace: Bool {
        service.cluster.computeTarget == .server
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                if isServerWorkspace {
                    serverContent
                } else {
                    localContent
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder
    private var localContent: some View {
        if let matrix = geometry.activeMatrix, let result = geometry.result {
            sectionHeader(
                "Cosines @ L\(matrix.layer)",
                caption: "\(result.artifacts.count) vectors · "
                    + "\(result.matrices.count) common layers · "
                    + "layer stepper in the Analysis pane")
            GeometryMatrixView(matrix: matrix)
            if !result.rsa.isEmpty {
                sectionHeader(
                    "Layer RSA",
                    caption: "second-order similarity of the cosine structure across layers")
                GeometryRSAView(result: result)
            }
        } else {
            emptyState
        }
    }

    @ViewBuilder
    private var serverContent: some View {
        if let matrix = geometry.serverMatrix {
            sectionHeader(
                "Cosines @ L\(matrix.layer) (server)",
                caption: "computed on \(service.cluster.substrateLabel) over its vector catalog")
            GeometryMatrixView(matrix: matrix)
            serverLayerCaption
        } else {
            emptyState
        }
    }

    /// The server clamps the requested layer per vector (layer counts can
    /// differ); when they diverge, say so instead of implying one layer.
    @ViewBuilder
    private var serverLayerCaption: some View {
        if Set(geometry.serverLayers).count > 1 {
            let layers = geometry.serverLayers.map(String.init).joined(separator: ", ")
            Text("per-vector layers (server-clamped): \(layers)")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private var emptyState: some View {
        ContentUnavailableView(
            "No geometry computed yet",
            systemImage: "circle.grid.cross",
            description: Text(
                "Select at least two vectors in the Analysis pane and press "
                    + "Compute — the cosine table renders here."))
            .frame(maxWidth: .infinity)
            .padding(.top, 40)
    }

    private func sectionHeader(_ title: String, caption: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.headline)
            Text(caption)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.top, 4)
    }
}

// MARK: Matrix rendering (moved from GeometryPanelView — the tables now
// render only in the viewer)

struct GeometryMatrixView: View {
    let matrix: GeometryLayerMatrix

    var body: some View {
        ScrollView([.horizontal, .vertical]) {
            Grid(alignment: .trailing, horizontalSpacing: 8, verticalSpacing: 4) {
                GridRow {
                    Text("")
                    ForEach(matrix.labels, id: \.self) { label in
                        Text(shortGeometryLabel(label))
                            .font(.caption2)
                            .frame(width: 58)
                            .help(label)
                    }
                }
                ForEach(matrix.labels.indices, id: \.self) { row in
                    GridRow {
                        Text(shortGeometryLabel(matrix.labels[row]))
                            .font(.caption2)
                            .frame(width: 72, alignment: .trailing)
                            .help(matrix.labels[row])
                        ForEach(matrix.labels.indices, id: \.self) { column in
                            GeometryCell(value: matrix.values[row][column])
                        }
                    }
                }
            }
            .padding(.vertical, 4)
        }
        .frame(minHeight: 160, maxHeight: 420)
    }
}

struct GeometryRSAView: View {
    let result: GeometryAnalysisResult

    var body: some View {
        ScrollView([.horizontal, .vertical]) {
            Grid(alignment: .trailing, horizontalSpacing: 8, verticalSpacing: 4) {
                GridRow {
                    Text("")
                    ForEach(result.matrices) { matrix in
                        Text("L\(matrix.layer)")
                            .font(.caption2)
                            .frame(width: 48)
                    }
                }
                ForEach(result.matrices.indices, id: \.self) { row in
                    GridRow {
                        Text("L\(result.matrices[row].layer)")
                            .font(.caption2)
                        ForEach(result.matrices.indices, id: \.self) { column in
                            GeometryCell(value: result.rsa[row][column])
                        }
                    }
                }
            }
            .padding(.vertical, 4)
        }
        .frame(minHeight: 140, maxHeight: 320)
    }
}

private struct GeometryCell: View {
    let value: Float

    var body: some View {
        Text(display)
            .font(.caption.monospacedDigit())
            .frame(width: 48)
            .padding(.vertical, 2)
            .background(cellColor, in: RoundedRectangle(cornerRadius: 4))
            .help(helpText)
    }

    /// NaN marks a pair the server could not compare — shown, not zeroed.
    private var display: String {
        value.isNaN ? "—" : value.formatted(.number.precision(.fractionLength(2)))
    }

    private var helpText: String {
        value.isNaN
            ? "not comparable (dimension mismatch)"
            : value.formatted(.number.precision(.fractionLength(4)))
    }

    private var cellColor: Color {
        guard !value.isNaN else { return .clear }
        let opacity = Double(min(1, abs(value))) * 0.35
        if value >= 0 {
            return Color.blue.opacity(opacity)
        } else {
            return Color.red.opacity(opacity)
        }
    }
}

private func shortGeometryLabel(_ label: String) -> String {
    let head = label.split(separator: "·").first?
        .trimmingCharacters(in: .whitespacesAndNewlines) ?? label
    return head.count > 10 ? String(head.prefix(9)) + "…" : head
}
