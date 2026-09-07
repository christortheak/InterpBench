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
    /// The bytes as loaded, so "has this been edited?" is a real question and
    /// not a guess — Save used to be enabled on an untouched file and Cancel
    /// used to drop edits silently (2026-09-06 audit).
    @State private var loadedText = ""
    /// Why the file cannot be edited here (nil = editable).
    @State private var editRefusal: String?
    @State private var status: String?
    @State private var confirmingDiscard = false

    /// 2 MB — recipe files (prompts, rubrics, baselines, batteries) are far
    /// smaller; anything bigger belongs in a real editor.
    private static let editByteLimit = 2_097_152

    private var isDirty: Bool { editRefusal == nil && text != loadedText }

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
                    .accessibilityLabel("File contents")
                    .help(
                        "the file's text — Save writes it back in place; "
                            + "nothing is written until you do")
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
            CopyButton(
                "Copy All",
                help: "copy the whole editor text to the clipboard"
            ) {
                editRefusal == nil ? text : nil
            }
            .disabled(editRefusal != nil)
            Button("Open in Default App") {
                guard let url = reference.url else {
                    status = "this path does not resolve to a file on disk"
                    return
                }
                // The sheet exists because researchers may have no handler
                // for .jsonl/.md — so "no app opened it" has to be said, not
                // swallowed (2026-09-06 audit).
                if !NSWorkspace.shared.open(url) {
                    status = "macOS has no app registered for this file type — "
                        + "edit it here, or set a default app in Finder's Get Info"
                }
            }
            .help("open this file in whatever app macOS associates with it — "
                + "for users who prefer a full editor; unsaved changes here "
                + "are not carried over")
            Spacer()
            Button("Cancel", role: .cancel) {
                if isDirty { confirmingDiscard = true } else { dismiss() }
            }
            .keyboardShortcut(.cancelAction)
            .help(
                isDirty
                    ? "close without saving — the edits in this box are discarded"
                    : "close the editor; nothing has been changed")
            .confirmationDialog(
                "Discard edits to “\(reference.displayName)”?",
                isPresented: $confirmingDiscard
            ) {
                Button("Discard Edits", role: .destructive) { dismiss() }
                Button("Keep Editing", role: .cancel) {}
            } message: {
                Text("The file on disk is unchanged; the text typed here is lost.")
            }
            Button("Save") { save() }
                .keyboardShortcut(.defaultAction)
                .disabled(editRefusal != nil || !isDirty)
                .help(
                    isDirty
                        ? "write the editor text back to the file (atomic write)"
                        : "nothing to save — the text matches the file on disk")
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
        loadedText = decoded
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
