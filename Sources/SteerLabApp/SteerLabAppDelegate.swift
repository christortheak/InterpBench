import AppKit

/// Receives folders handed to the app from outside: `open -a SteerLab
/// <workspace folder>`, the coding assistant's `openInApp`. It exists for
/// that alone; everything else stays in the SwiftUI app. (The app declares
/// no document types, so the Dock does not offer it dropped folders.)
///
/// On a cold launch the folder can arrive before the window exists, so folders
/// wait in `pending` until the window installs `onOpenFolders` and drains them.
/// `WorkspaceActions.openHandedFolders` decides what to do with them.
@MainActor
final class SteerLabAppDelegate: NSObject, NSApplicationDelegate {

    /// Set by the window once it can act; nil until then.
    var onOpenFolders: (([URL]) -> Void)? {
        didSet { deliverPending() }
    }

    private var pending: [URL] = []

    func application(_ application: NSApplication, open urls: [URL]) {
        pending.append(contentsOf: urls)
        deliverPending()
    }

    private func deliverPending() {
        guard let onOpenFolders, !pending.isEmpty else { return }
        let urls = pending
        pending = []
        onOpenFolders(urls)
    }
}
