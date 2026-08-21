import AppKit
import ExperimentKit
import SwiftUI
import UniformTypeIdentifiers

/// Workspace-scoped file choosing (Usability Plan Phase 3, item 12): every
/// path a researcher used to type by hand gets a folder-plus "Choose…"
/// button that opens an NSOpenPanel at the workspace root and writes back
/// the WORKSPACE-RELATIVE path. Selections outside the workspace are
/// refused with a plain note — pins are workspace-relative by contract, so
/// an outside path could never pin.
@MainActor
enum WorkspaceFileChooser {

    enum Selection {
        case cancelled
        /// A file inside the workspace, as a workspace-relative path.
        case chosen(relativePath: String)
        /// Outside the workspace — the plain note says what to do instead.
        case outsideWorkspace(String)
    }

    /// `.jsonl` has no system UTType; fall back to plain text so the panel
    /// still allows the files.
    static var jsonlTypes: [UTType] {
        [UTType(filenameExtension: "jsonl") ?? .plainText, .json, .plainText]
    }

    static var tableTypes: [UTType] {
        [.json, .commaSeparatedText, .plainText]
    }

    /// Open a panel scoped to the workspace root (starting inside
    /// `startingSubdirectory` when it exists) and resolve the choice to a
    /// workspace-relative path.
    static func chooseWorkspaceFile(
        message: String,
        allowedTypes: [UTType],
        startingSubdirectory: String? = nil
    ) -> Selection {
        let root = VectorCatalog.projectRoot.standardizedFileURL
        let panel = NSOpenPanel()
        panel.message = message
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = allowedTypes
        var start = root
        if let startingSubdirectory {
            let candidate = root.appending(path: startingSubdirectory)
            if FileManager.default.fileExists(atPath: candidate.path) {
                start = candidate
            }
        }
        panel.directoryURL = start
        guard panel.runModal() == .OK, let url = panel.url else {
            return .cancelled
        }
        guard let relative = workspaceRelativePath(url) else {
            return .outsideWorkspace(
                "'\(url.lastPathComponent)' is outside the workspace "
                    + "(\(root.path)) — pins are workspace-relative, so copy "
                    + "the file into the workspace (e.g. its prompts/ "
                    + "folder) and choose it there")
        }
        return .chosen(relativePath: relative)
    }

    /// The workspace-relative path for a URL inside the workspace root
    /// (symlink-resolved on both sides so /tmp vs /private/tmp and linked
    /// workspaces compare correctly); nil when the URL is outside.
    static func workspaceRelativePath(_ url: URL) -> String? {
        let root = VectorCatalog.projectRoot.standardizedFileURL
            .resolvingSymlinksInPath().path
        let path = url.standardizedFileURL.resolvingSymlinksInPath().path
        guard path.hasPrefix(root + "/") else { return nil }
        return String(path.dropFirst(root.count + 1))
    }

    /// Open a panel WITHOUT the workspace scope and read the file's bytes —
    /// for import SOURCES (a spreadsheet in Downloads is fine; the
    /// converted result is what lands in the workspace).
    static func readAnyFile(
        message: String, allowedTypes: [UTType]
    ) -> (fileName: String, data: Data)? {
        let panel = NSOpenPanel()
        panel.message = message
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = allowedTypes
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        let accessing = url.startAccessingSecurityScopedResource()
        defer { if accessing { url.stopAccessingSecurityScopedResource() } }
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (url.lastPathComponent, data)
    }
}

/// The small folder-plus "Choose…" button rendered beside a path
/// TextField. The field stays editable exactly as before; this is the
/// no-typing alternative. On a valid in-workspace choice it hands the
/// relative path to `onChoose` (which triggers the field's existing
/// pin/validation flow where one exists); outside-workspace choices go to
/// `onProblem` as a plain note.
struct WorkspacePathChooseButton: View {
    let message: String
    let allowedTypes: [UTType]
    var startingSubdirectory: String?
    let onChoose: (String) -> Void
    let onProblem: (String) -> Void

    var body: some View {
        Button {
            switch WorkspaceFileChooser.chooseWorkspaceFile(
                message: message,
                allowedTypes: allowedTypes,
                startingSubdirectory: startingSubdirectory)
            {
            case .cancelled:
                break
            case .chosen(let relativePath):
                onChoose(relativePath)
            case .outsideWorkspace(let problem):
                onProblem(problem)
            }
        } label: {
            Image(systemName: "folder.badge.plus")
        }
        .buttonStyle(.plain)
        .help(
            "choose a file inside the workspace — the path is stored "
                + "workspace-relative (typing it still works too)")
    }
}
