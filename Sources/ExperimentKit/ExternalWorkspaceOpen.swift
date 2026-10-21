import Foundation

/// A workspace folder handed to the app from outside (2026-10-21): `open -a
/// SteerLab <folder>`, which is how a coding assistant offers to open the app
/// on the workspace it just set up (the handoff's `openInApp`), or a folder
/// dropped on the app's Dock icon. Before this the app ignored the folder
/// and showed whichever workspace it had open last, so "open the app" could
/// put a different study in front of the researcher.
///
/// The decision is pure, so it is tested without an app. The app opens what
/// it decides through the same path as the Workspace menu's Open
/// (`WorkspaceStore.switchTo`), which refuses a folder that is not a
/// workspace and says why.
public enum ExternalWorkspaceOpen {

    public enum Decision: Sendable, Equatable {
        /// Switch to this folder.
        case open(URL)
        /// It is already the open workspace: nothing to do.
        case alreadyOpen
        /// `STEERLAB_WORKSPACE` pins this process to one workspace, the same
        /// condition that disables Open Workspace… in the menu.
        case pinned(String)
        /// Nothing handed over was a folder.
        case noFolder(String)
    }

    /// The first folder among `urls` decides; files are ignored, because a
    /// workspace is a folder and the app opens no documents.
    public static func decide(
        urls: [URL], current: URL?, pinned: Bool,
        isDirectory: (URL) -> Bool = defaultIsDirectory
    ) -> Decision {
        guard let folder = urls.map(\.standardizedFileURL).first(where: isDirectory) else {
            return .noFolder(
                "SteerLab opens a workspace folder, and what it was given is not a "
                    + "folder. Open the workspace folder itself, for example with "
                    + "open -a SteerLab <workspace folder>.")
        }
        if let current, current.standardizedFileURL.path == folder.path {
            return .alreadyOpen
        }
        if pinned {
            return .pinned(
                "SteerLab was started with STEERLAB_WORKSPACE, which keeps it on "
                    + "one workspace, so it did not open \(folder.path). Quit SteerLab "
                    + "and open it without STEERLAB_WORKSPACE to switch.")
        }
        return .open(folder)
    }

    public static func defaultIsDirectory(_ url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
            && isDirectory.boolValue
    }
}
