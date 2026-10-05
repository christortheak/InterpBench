import AppKit
import ExperimentKit
import SwiftUI

/// Studies or templates whose saved file cannot be read: the name, the file,
/// the reason, and a way to find the file in Finder. Read-only — the app never
/// writes to a file it could not read; the researcher (or their coding
/// assistant) repairs it, and Refresh picks the repair up.
struct UnreadableManifestsView: View {
    let items: [UnreadableManifest]
    /// "study" or "template", for the heading.
    let noun: String
    let workspaceRoot: URL

    var body: some View {
        if !items.isEmpty {
            VStack(alignment: .leading, spacing: 6) {
                Label(heading, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption.bold())
                    .foregroundStyle(.orange)
                ForEach(items) { item in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(item.name)
                            .font(.caption.weight(.semibold))
                        Text(item.reason)
                            .font(.caption)
                            .fixedSize(horizontal: false, vertical: true)
                            .textSelection(.enabled)
                        HStack(spacing: 8) {
                            Text(item.relativePath(workspaceRoot: workspaceRoot))
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .textSelection(.enabled)
                                .help(item.url.path)
                            Button("Reveal in Finder") {
                                NSWorkspace.shared.activateFileViewerSelecting([item.url])
                            }
                            .buttonStyle(.link)
                            .font(.caption2)
                            .help("show this file in Finder so you can open it in a text editor")
                        }
                    }
                }
                Text(
                    "SteerLab leaves these files untouched. Fix the file — or ask your "
                        + "coding assistant to — then press Refresh.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.vertical, 2)
        }
    }

    private var heading: String {
        items.count == 1
            ? "1 \(noun) could not be read"
            : "\(items.count) \(noun == "study" ? "studies" : noun + "s") could not be read"
    }
}

/// The small Refresh button every library header carries.
struct LibraryRefreshButton: View {
    let help: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label("Refresh", systemImage: "arrow.clockwise")
        }
        .buttonStyle(.borderless)
        .controlSize(.small)
        .help(help)
    }
}
