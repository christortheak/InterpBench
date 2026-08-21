import ExperimentKit
import SwiftUI

/// "Import table…" (Usability Plan Phase 3, item 13): a JSON array or CSV
/// becomes pinned study data through a small column-mapping sheet. All
/// parsing, guessing, conversion, and write/pin rules live in
/// `TabularImport` (ExperimentKit, unit-tested); these views render them.
///
/// Own file on purpose: `ExperimentsPanelView` is at the type-checker's
/// limits — the panel view only drops a `TabularImportButton` next to the
/// existing affordances.

/// One parsed table headed for one import target — the sheet's item.
struct TabularImportRequest: Identifiable {
    let id = UUID()
    let target: TabularImport.Target
    let fileName: String
    let table: TabularImport.Table
}

/// The affordance: choose a .json/.csv file, parse it, then present the
/// mapping sheet. Import problems before the sheet (unreadable file, not a
/// table) surface inline under the button.
struct TabularImportButton: View {
    let target: TabularImport.Target
    let panel: ExperimentPanel
    var disabled: Bool = false

    @State private var request: TabularImportRequest?
    @State private var problem: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Button("Import table…") { choose() }
                .disabled(disabled)
                .help(
                    "choose a JSON array or CSV (header row) and map its "
                        + "columns — the converted file lands at the standard "
                        + "destination and is pinned, no hand-written "
                        + (target == .taskPrompts ? "JSONL" : "CSV"))
            if let problem {
                Text(problem)
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
        }
        .sheet(item: $request) { request in
            TabularImportMappingSheet(request: request) { mapping in
                switch request.target {
                case .taskPrompts:
                    return panel.importTaskPromptsTable(
                        table: request.table, mapping: mapping)
                case .humanBaseline:
                    return panel.importHumanBaselineTable(
                        table: request.table, mapping: mapping)
                }
            }
        }
    }

    private func choose() {
        problem = nil
        guard
            let (fileName, data) = WorkspaceFileChooser.readAnyFile(
                message: "Choose a JSON array or CSV table to import "
                    + "as \(target.title)",
                allowedTypes: WorkspaceFileChooser.tableTypes)
        else { return }
        do {
            let table = try TabularImport.parseTable(data, fileName: fileName)
            request = TabularImportRequest(
                target: target, fileName: fileName, table: table)
        } catch {
            problem = "\(error)"
        }
    }
}

/// The mapping sheet: a picker per target field (auto-guessed when column
/// names match), a 3-row preview of the conversion, and Import & Pin.
/// Refusals from the conversion/write/pin path show inline — the sheet
/// dismisses only when the import landed.
struct TabularImportMappingSheet: View {
    let request: TabularImportRequest
    /// Runs the panel's import; returns the plain problem, nil = landed.
    let onImport: ([String: String]) -> String?

    @Environment(\.dismiss) private var dismiss
    @State private var mapping: [String: String]
    @State private var problem: String?

    init(
        request: TabularImportRequest,
        onImport: @escaping ([String: String]) -> String?
    ) {
        self.request = request
        self.onImport = onImport
        _mapping = State(
            initialValue: TabularImport.guessMapping(
                columns: request.table.columns, target: request.target))
    }

    private var requiredFields: [String] {
        TabularImport.requiredFields(for: request.target)
    }

    private var optionalFields: [String] {
        TabularImport.optionalFields(for: request.target)
    }

    private var requiredMapped: Bool {
        requiredFields.allSatisfy { !(mapping[$0] ?? "").isEmpty }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Map columns — \(request.fileName)")
                .font(.headline)
            Text(
                "\(request.table.rows.count) data row"
                    + "\(request.table.rows.count == 1 ? "" : "s"), columns: "
                    + request.table.columns.joined(separator: ", "))
                .font(.caption)
                .foregroundStyle(.secondary)
            Form {
                ForEach(requiredFields, id: \.self) { field in
                    fieldPicker(field, required: true)
                }
                ForEach(optionalFields, id: \.self) { field in
                    fieldPicker(field, required: false)
                }
            }
            .formStyle(.columns)
            previewGrid
            if let problem {
                Text(problem)
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .textSelection(.enabled)
            }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Import & Pin") {
                    if let refusal = onImport(cleanMapping) {
                        problem = refusal
                    } else {
                        dismiss()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!requiredMapped)
            }
        }
        .padding(16)
        .frame(minWidth: 460)
    }

    /// The mapping without unmapped ("") entries.
    private var cleanMapping: [String: String] {
        mapping.filter { !$0.value.isEmpty }
    }

    @ViewBuilder
    private func fieldPicker(_ field: String, required: Bool) -> some View {
        LabeledContent(required ? field : "\(field) (optional)") {
            Picker(
                "",
                selection: Binding(
                    get: { mapping[field] ?? "" },
                    set: { mapping[field] = $0 })
            ) {
                Text("—").tag("")
                ForEach(request.table.columns, id: \.self) { column in
                    Text(column).tag(column)
                }
            }
            .labelsHidden()
        }
        .help(TabularImport.fieldHelp(field, target: request.target))
    }

    /// First three rows AS THEY WILL IMPORT — mapped values only, so a
    /// wrong guess is visible before anything is written or pinned.
    private var previewGrid: some View {
        let fields = (requiredFields + optionalFields).filter {
            !(mapping[$0] ?? "").isEmpty
        }
        let rows = request.table.rows.prefix(3)
        return Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 2) {
            GridRow {
                ForEach(fields, id: \.self) { field in
                    Text(field).font(.caption2.bold())
                }
            }
            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                GridRow {
                    ForEach(fields, id: \.self) { field in
                        Text(previewCell(row: row, field: field))
                            .font(.caption2.monospaced())
                            .lineLimit(1)
                            .frame(maxWidth: 180, alignment: .leading)
                    }
                }
            }
        }
        .padding(6)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 4))
    }

    private func previewCell(
        row: [String: TabularImport.Value], field: String
    ) -> String {
        guard let column = mapping[field], let value = row[column] else {
            return "·"
        }
        if field == "options" {
            return TabularImport.optionStrings(value).joined(separator: " | ")
        }
        return value.text
    }
}
