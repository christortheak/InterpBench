import Foundation

/// A workspace folder handed to the app from outside (2026-10-21): `open -a
/// SteerLab <folder>`, which is how a coding assistant offers to open the app
/// on the workspace it just set up (the handoff's `openInApp`). The app
/// declares no document types, so a folder dropped on its Dock icon is not
/// offered to it. Before this the app ignored the folder
/// and showed whichever workspace it had open last, so "open the app" could
/// put a different study in front of the researcher.
///
/// The decisions are pure, so they are tested without an app. The app opens
/// what it decides through the same path as the Workspace menu's Open:
/// `refusal` first, which turns a folder that is not a workspace away in the
/// app's own words, then `WorkspaceStore.switchTo`.
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

    /// The alert a researcher reads when a folder is not a workspace: the
    /// folder's name, what a workspace is, and the two ways forward, in the
    /// app's words. It says the open workspace stays open only when one is
    /// open: on a first launch there is none. The command line keeps
    /// `WorkspaceStore.open`'s own refusal, which names the full path for
    /// whoever typed it.
    public static func notAWorkspace(folderName: String, hadWorkspace: Bool) -> String {
        "“\(folderName)” is not a SteerLab workspace"
            + (hadWorkspace ? ", so the workspace you had open stays open." : ".")
            + " A workspace is a folder SteerLab made to hold a study's examples, "
            + "designs, and results. Choose one with Open Workspace… in the "
            + "Workspace menu, or make one with New Workspace…."
    }

    /// The alert for a folder the app was asked to open, or nil when it is a
    /// workspace and the switch can go ahead. Every way into a workspace in
    /// the app asks this first: Open Workspace… and a folder handed in.
    public static func refusal(
        for folder: URL, hasWorkspace: Bool,
        isWorkspace: (URL) -> Bool = WorkspaceStore.isWorkspace(url:)
    ) -> String? {
        let folder = folder.standardizedFileURL
        if isWorkspace(folder) { return nil }
        return notAWorkspace(folderName: folder.lastPathComponent, hadWorkspace: hasWorkspace)
    }

    /// The same, when what arrived was not a folder at all.
    public static let notAFolder =
        "SteerLab was asked to open something that is not a folder. A workspace "
        + "is a folder: choose one with Open Workspace… in the Workspace menu."

    /// The first folder among `urls` decides; files are ignored, because a
    /// workspace is a folder and the app opens no documents.
    public static func decide(
        urls: [URL], current: URL?, pinned: Bool,
        isDirectory: (URL) -> Bool = defaultIsDirectory
    ) -> Decision {
        guard let folder = urls.map(\.standardizedFileURL).first(where: isDirectory) else {
            return .noFolder(notAFolder)
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
