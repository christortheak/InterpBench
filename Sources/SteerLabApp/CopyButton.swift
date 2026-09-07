import AppKit
import SwiftUI

/// The app's one clipboard affordance (UI audit 2026-09-06, headline 19):
/// writes `text()` to the pasteboard and shows "Copied" for a moment, so
/// every copy action gives the same visible feedback. Use it wherever a
/// button's whole job is to copy something; for copies that happen inside a
/// larger action, call `Clipboard.copy(_:)` and report through the surface
/// that action already uses (a status line, a notice).
struct CopyButton<Label: View>: View {
    private let help: String
    private let text: () -> String?
    private let label: () -> Label
    @State private var copied = false

    /// - Parameters:
    ///   - help: the hover tooltip for the resting state (lowercase fragment,
    ///     like every other `.help` in the app).
    ///   - text: what to copy; `nil` means there is nothing to copy yet and
    ///     the click is a no-op (callers should also `.disabled` the button).
    ///   - label: the resting label.
    init(
        help: String,
        text: @escaping () -> String?,
        @ViewBuilder label: @escaping () -> Label
    ) {
        self.help = help
        self.text = text
        self.label = label
    }

    var body: some View {
        Button {
            guard let value = text(), Clipboard.copy(value) else { return }
            copied = true
            Task { @MainActor in
                try? await Task.sleep(for: .seconds(1.5))
                copied = false
            }
        } label: {
            if copied {
                SwiftUI.Label("Copied", systemImage: "checkmark")
            } else {
                label()
            }
        }
        .help(copied ? "copied to the clipboard" : help)
        .accessibilityLabel(copied ? "Copied" : help)
    }
}

extension CopyButton where Label == SwiftUI.Label<Text, Image> {
    /// Title + symbol convenience: `CopyButton("Copy Transcript", systemImage:
    /// "doc.on.doc", help: "…") { transcriptText }`.
    init(
        _ title: String, systemImage: String = "doc.on.doc", help: String,
        text: @escaping () -> String?
    ) {
        self.init(help: help, text: text) {
            SwiftUI.Label(title, systemImage: systemImage)
        }
    }
}

/// Pasteboard write with a truthful result — `setString` can refuse (another
/// process holding the pasteboard), and callers that report "copied" should
/// only say so when it actually happened.
enum Clipboard {
    @discardableResult
    static func copy(_ string: String) -> Bool {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        return pasteboard.setString(string, forType: .string)
    }
}
