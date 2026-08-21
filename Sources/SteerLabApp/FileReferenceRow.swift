import AppKit
import ExperimentKit
import SwiftUI

/// Reusable row for any file path a panel displays (judge rubrics, task
/// prompts, capability batteries, human baselines, robustness files): shows
/// the file name, a read-only "view" sheet with the file's text and its
/// pinned SHA-256 prefix, and Reveal in Finder. Editable recipe files
/// additionally get the pencil, which opens the IN-APP editor sheet
/// (`InlineFileEditorSheet`) — researchers may have no default handler for
/// .jsonl/.md files, so the system-editor route dead-ended (2026-07-20
/// researcher round); "open in default app" lives inside the sheet for
/// those who do.
struct FileReferenceRow: View {
    let label: String
    let path: String
    var pinnedHash: String?
    /// Historical name kept for call-site stability; the pencil now opens
    /// the in-app editor sheet instead of the system default editor.
    var allowsOpenInEditor = false

    @State private var showingViewer = false
    @State private var showingEditor = false

    private var reference: FileReference { FileReference.resolve(path) }

    /// A directory is browsed in Finder, not "viewed" (2026-07-19 paper
    /// cut: the eye on a battery FOLDER said "cannot read file").
    private var isDirectory: Bool {
        guard let url = reference.url else { return false }
        var directory: ObjCBool = false
        return FileManager.default.fileExists(
            atPath: url.path, isDirectory: &directory) && directory.boolValue
    }

    var body: some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(reference.displayName.isEmpty ? "—" : reference.displayName)
                .font(.caption.monospaced())
                .textSelection(.enabled)
                .lineLimit(1)
                .help(reference.originalPath)
            if let pinnedHash {
                Text("@ \(String(pinnedHash.prefix(12)))…")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .help("pinned SHA-256 \(pinnedHash)")
            }
            Spacer()
            if reference.url != nil, reference.exists {
                if !isDirectory {
                    Button {
                        showingViewer = true
                    } label: {
                        Image(systemName: "eye")
                    }
                    .buttonStyle(.plain)
                    .help("view the file's contents read-only")
                }
                Button {
                    if let url = reference.url {
                        NSWorkspace.shared.activateFileViewerSelecting([url])
                    }
                } label: {
                    Image(systemName: "folder")
                }
                .buttonStyle(.plain)
                .help(isDirectory
                    ? "a DIRECTORY — browse its files in Finder"
                    : "reveal in Finder")
                if allowsOpenInEditor, !isDirectory {
                    Button {
                        showingEditor = true
                    } label: {
                        Image(systemName: "square.and.pencil")
                    }
                    .buttonStyle(.plain)
                    .help("edit in the app — Save writes the file in place; "
                        + "a pinned file then shows as drift until re-pinned "
                        + "(the sheet can also open your default app)")
                }
            } else {
                Text(reference.url == nil ? "unresolvable path" : "missing")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
        }
        .sheet(isPresented: $showingViewer) {
            FileReferenceViewer(reference: reference, pinnedHash: pinnedHash)
        }
        .sheet(isPresented: $showingEditor) {
            InlineFileEditorSheet(reference: reference, pinnedHash: pinnedHash)
        }
    }
}

/// Read-only monospaced viewer for a referenced file, with the pinned hash
/// prefix (and the current bytes' hash when it differs — drift made
/// visible, matching the freeze firewall's verify semantics).
private struct FileReferenceViewer: View {
    let reference: FileReference
    let pinnedHash: String?

    @Environment(\.dismiss) private var dismiss

    private static let byteLimit = 262_144  // 256 KB is plenty for recipe files

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(reference.displayName)
                        .font(.title3.weight(.semibold))
                    Text(reference.originalPath)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                    hashLine
                }
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }

            Divider()

            ScrollView {
                Text(contents.text)
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            if contents.truncated {
                Text("… truncated preview (first 256 KB) — the pencil editor "
                    + "or your default app shows the full file")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .frame(minWidth: 640, minHeight: 460)
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

    private var contents: (text: String, truncated: Bool) {
        guard let url = reference.url, let data = try? Data(contentsOf: url) else {
            return ("(could not read file)", false)
        }
        let truncated = data.count > Self.byteLimit
        let slice = truncated ? data.prefix(Self.byteLimit) : data[...]
        return (String(decoding: slice, as: UTF8.self), truncated)
    }
}
