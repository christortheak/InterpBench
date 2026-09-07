import ExperimentKit
import SteeringKit
import SwiftUI
import UniformTypeIdentifiers

struct ImportJSONLSheet: View {
    @Binding var text: String
    /// Workspace-relative destination the import writes to (nil when no
    /// study is selected — the import button explains instead of failing).
    let destination: String?
    /// Prepares an immutable input version and pins it to the reviewed draft.
    /// A successful result dismisses the sheet.
    let onImport: (String) -> Bool
    /// The panel's task-prompts status line (import refusals surface there).
    let statusLine: () -> String?

    @Environment(\.dismiss) private var dismiss
    @State private var showFilePicker = false
    @State private var fileReadError: String?

    private var preview: TaskPromptsImport.Outcome {
        TaskPromptsImport.preview(text)
    }

    private var importable: Bool {
        if case .preview = preview { return destination != nil }
        return false
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Import JSONL task prompts")
                .font(.headline)
            // Required record structure, VISIBLE in the sheet (2026-07-20
            // researcher round, item 2b) — not hover-only.
            Text(StudyInfo.importRecordStructure)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            TextEditor(text: $text)
                .font(.system(.caption, design: .monospaced))
                .frame(minWidth: 480, minHeight: 220)
                .help(
                    "one JSON record per line; the preview below parses as you "
                        + "type and names the first line it cannot read")

            HStack {
                Button("Choose File…") { showFilePicker = true }
                    .help("read a .jsonl file into the text area above")
                if let fileReadError {
                    Text(fileReadError)
                        .font(.caption2)
                        .foregroundStyle(.red)
                }
            }

            previewRow

            if let destination {
                // Destination semantics, visible and TRUE to the code
                // (2026-07-20 researcher round, item 2c).
                Text(StudyInfo.importDestination(destination))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Text("Imports create an immutable input version. Existing prompt files remain unchanged.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text(
                    "select a draft study first — the import pins into the "
                        + "selected draft's manifest"
                )
                .font(.caption2)
                .foregroundStyle(.orange)
            }
            if let status = statusLine() {
                Text(status)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .help("close without importing; no file is written")
                Button("Import & Pin") {
                    if onImport(text) { dismiss() }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!importable)
                .help(
                    importable
                        ? "writes these records as a new immutable input "
                            + "version, makes it the study's prompts file, and "
                            + "pins its hash — one action from paste to pinned"
                        : "the records above have to parse, and a draft study "
                            + "has to be selected, before anything can be "
                            + "pinned")
            }
        }
        .padding(16)
        .fileImporter(
            isPresented: $showFilePicker,
            allowedContentTypes: [
                UTType(filenameExtension: "jsonl") ?? .plainText, .json, .plainText,
            ]
        ) { result in
            switch result {
            case .success(let url):
                let accessing = url.startAccessingSecurityScopedResource()
                defer { if accessing { url.stopAccessingSecurityScopedResource() } }
                do {
                    text = String(decoding: try Data(contentsOf: url), as: UTF8.self)
                    fileReadError = nil
                } catch {
                    fileReadError =
                        "could not read \(url.lastPathComponent): \(error.localizedDescription)"
                }
            case .failure(let error):
                fileReadError = error.localizedDescription
            }
        }
    }

    @ViewBuilder
    private var previewRow: some View {
        switch preview {
        case .empty:
            Label(
                "nothing to import yet — paste JSONL records above",
                systemImage: "info.circle"
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        case .preview(let parsed):
            Label(parsed.summaryLine, systemImage: "checkmark.circle")
                .font(.caption)
                .foregroundStyle(.green)
        case .failure(let line, let message):
            Label(
                "line \(line): \(message) — fix it; garbage is refused, "
                    + "never imported as prompt text",
                systemImage: "xmark.octagon"
            )
            .font(.caption)
            .foregroundStyle(.red)
            .textSelection(.enabled)
        }
    }
}

/// What Rename was opened on. Captured at open time so the sheet never
/// re-reads the store while it is up.
