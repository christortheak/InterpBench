import ExperimentKit
import SwiftUI

/// Formatting large fitted parameters and traces must not block form editing.
struct PolicyJSONView: View {
    let value: JSONValue
    @State private var text = "Preparing the document…"
    var body: some View {
        Text(text).font(.caption.monospaced()).textSelection(.enabled)
            .task(id: value) {
                let formatted = await Task.detached(priority: .utility) {
                    (try? InterventionPolicyLibrary.formatted(value)) ?? "Could not display this document."
                }.value
                guard !Task.isCancelled else { return }
                text = formatted
            }
    }
}
