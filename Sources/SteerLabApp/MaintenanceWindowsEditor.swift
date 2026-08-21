import ExperimentKit
import SwiftUI

/// WS3 maintenance-windows editor: a deliberately dumb sheet over
/// `POST /api/housekeeping/maintenance`. Local validation is exactly one
/// rule (end > start); every other refusal comes from the server and is
/// shown verbatim. Save replaces the stored list and reflects the server's
/// canonical answer back through `onSaved`.
struct MaintenanceWindowsEditor: View {
    let cluster: ClusterConnectionStore
    var initialWindows: [MaintenanceWindow]
    var onSaved: ([MaintenanceWindow]) -> Void

    @Environment(\.dismiss) private var dismiss

    private struct Row: Identifiable {
        let id = UUID()
        var start: Date
        var end: Date
        var label: String
    }

    @State private var rows: [Row] = []
    @State private var errorLine: String?
    @State private var isSaving = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Maintenance windows — \(cluster.substrateLabel)")
                .font(.headline)
            Text("The executor refuses submissions whose walltime would cross a "
                + "window. Times are stored in UTC (ISO-8601).")
                .font(.caption)
                .foregroundStyle(.secondary)

            if rows.isEmpty {
                Text("No windows declared.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    ForEach($rows) { $row in
                        rowEditor($row)
                    }
                }
            }
            .frame(minHeight: 80, maxHeight: 320)

            HStack {
                Button {
                    let start = Date().addingTimeInterval(86_400)
                    rows.append(Row(start: start, end: start.addingTimeInterval(4 * 3600), label: ""))
                } label: {
                    Label("Add window", systemImage: "plus")
                }
                .controlSize(.small)
                Spacer()
            }

            if let problem = localValidationProblem {
                Label(problem, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            if let errorLine {
                // Server validation messages, verbatim.
                Text(errorLine)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
                    .lineLimit(4)
            }

            HStack {
                Button("Cancel") { dismiss() }
                Spacer()
                Button(isSaving ? "Saving…" : "Save") {
                    Task { await save() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isSaving || localValidationProblem != nil)
            }
        }
        .padding(16)
        .frame(minWidth: 520)
        .onAppear(perform: populate)
    }

    @ViewBuilder
    private func rowEditor(_ row: Binding<Row>) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            DatePicker(
                "start", selection: row.start,
                displayedComponents: [.date, .hourAndMinute])
                .labelsHidden()
            Text("→").foregroundStyle(.secondary)
            DatePicker(
                "end", selection: row.end,
                displayedComponents: [.date, .hourAndMinute])
                .labelsHidden()
            TextField("label (optional)", text: row.label)
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 140)
            Button(role: .destructive) {
                rows.removeAll { $0.id == row.wrappedValue.id }
            } label: {
                Image(systemName: "trash")
            }
            .buttonStyle(.borderless)
            .help("remove this window")
        }
    }

    private var localValidationProblem: String? {
        for (index, row) in rows.enumerated() where row.end <= row.start {
            let name = row.label.isEmpty ? "window \(index + 1)" : "'\(row.label)'"
            return "\(name): end must be after start"
        }
        return nil
    }

    private func populate() {
        rows = initialWindows.compactMap { window in
            guard let start = window.startDate, let end = window.endDate else { return nil }
            return Row(start: start, end: end, label: window.label ?? "")
        }
    }

    private func save() async {
        guard localValidationProblem == nil else { return }
        cluster.loadStoredToken()
        guard let client = cluster.client else {
            errorLine = "invalid server URL"
            return
        }
        let windows = rows.map { row in
            MaintenanceWindow(
                start: HousekeepingDates.format(row.start),
                end: HousekeepingDates.format(row.end),
                label: row.label.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    ? nil
                    : row.label.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        isSaving = true
        defer { isSaving = false }
        do {
            let canonical = try await client.setMaintenanceWindows(windows)
            onSaved(canonical)
            dismiss()
        } catch {
            errorLine = "\(error)"
        }
    }
}
