import AppKit
import ExperimentKit
import SwiftUI

/// The one IN-APP editor every `FileReferenceRow` pencil opens (2026-07-20
/// researcher round, item 3): researchers may have no default handler for
/// .jsonl/.md recipe files, so "open in the default editor" dead-ended.
/// This sheet edits the file in place instead — monospaced text editor,
/// Copy All, atomic Save with the drift consequence stated visibly, and a
/// small "open in default app" affordance for users who do have one.
///
/// Honesty rules: files that are not UTF-8 text, or larger than the edit
/// limit, are refused for editing (a lossy load would corrupt bytes on
/// save) — the sheet says so and still offers Copy/reveal/default-app.
struct InlineFileEditorSheet: View {
    let reference: FileReference
    var pinnedHash: String?

    @Environment(\.dismiss) private var dismiss
    @State private var text = ""
    /// Why the file cannot be edited here (nil = editable).
    @State private var editRefusal: String?
    @State private var status: String?

    /// 2 MB — recipe files (prompts, rubrics, baselines, batteries) are far
    /// smaller; anything bigger belongs in a real editor.
    private static let editByteLimit = 2_097_152

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            Divider()
            if let editRefusal {
                Label(editRefusal, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                TextEditor(text: $text)
                    .font(.system(.caption, design: .monospaced))
                    .frame(minHeight: 300)
                if pinnedHash != nil {
                    // The drift consequence, visible at the moment of action
                    // — not hover-only (StudyInfo.inlineFileEditor is the
                    // shared corpus text).
                    Text(StudyInfo.inlineFileEditor)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            if let status {
                Text(status)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            buttonRow
        }
        .padding(16)
        .frame(minWidth: 680, minHeight: 500)
        .onAppear { load() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(reference.displayName)
                .font(.title3.weight(.semibold))
            Text(reference.originalPath)
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            hashLine
        }
    }

    @ViewBuilder
    private var hashLine: some View {
        if let pinnedHash {
            let current = reference.url.flatMap(FileReference.currentSHA256(of:))
            if let current, current != pinnedHash {
                Text("pinned @ \(String(pinnedHash.prefix(12)))… · current @ "
                    + "\(String(current.prefix(12)))… (DRIFTED)")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.orange)
            } else {
                Text("pinned @ \(String(pinnedHash.prefix(12)))…")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var buttonRow: some View {
        HStack(spacing: 8) {
            Button("Copy All") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(text, forType: .string)
                status = "copied \(text.count) characters to the clipboard"
            }
            .disabled(editRefusal != nil)
            .help("copy the whole editor text to the clipboard")
            Button("Open in Default App") {
                if let url = reference.url {
                    NSWorkspace.shared.open(url)
                }
            }
            .help("open this file in whatever app macOS associates with it — "
                + "for users who prefer a full editor; unsaved changes here "
                + "are not carried over")
            Spacer()
            Button("Cancel") { dismiss() }
            Button("Save") { save() }
                .keyboardShortcut(.defaultAction)
                .disabled(editRefusal != nil)
                .help("write the editor text back to the file (atomic write)")
        }
    }

    private func load() {
        guard let url = reference.url, let data = try? Data(contentsOf: url) else {
            editRefusal = "could not read the file — it may have been moved "
                + "or deleted; use the folder button to locate it"
            return
        }
        guard data.count <= Self.editByteLimit else {
            editRefusal = "file is larger than 2 MB — too big for the in-app "
                + "editor; use Open in Default App"
            return
        }
        let decoded = String(decoding: data, as: UTF8.self)
        // A lossy decode round-trips to different bytes; editing through it
        // would corrupt the file on save. Refuse instead.
        guard Data(decoded.utf8) == data else {
            editRefusal = "file is not UTF-8 text — editing it here would "
                + "corrupt it; use Open in Default App"
            return
        }
        text = decoded
    }

    private func save() {
        guard let url = reference.url else { return }
        do {
            try Data(text.utf8).write(to: url, options: .atomic)
            dismiss()
        } catch {
            status = "could not save: \(error.localizedDescription)"
        }
    }
}
