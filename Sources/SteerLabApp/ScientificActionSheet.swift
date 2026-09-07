import ExperimentKit
import SwiftUI

struct ScientificActionSheet: View {
    let operation: ScienceCatalog.Operation
    let client: ClusterClient
    @Environment(\.dismiss) private var dismiss
    @State private var actionID = ""
    @State private var requestText = "{\n  \"path\": {},\n  \"query\": {},\n  \"body\": {}\n}"
    @State private var resultText = ""
    @State private var confirmed = false
    @State private var busy = false
    private var action: ScienceCatalog.Action? { operation.actions.first { $0.id == actionID } }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(operation.title).font(.title2)
            Text(client.profile.baseURL.absoluteString).font(.caption).textSelection(.enabled)
            Text("Use the method guide and your agent to prepare this action's request. The existing server owner validates it; running an action does not establish scientific qualification.")
            Picker("Action", selection: $actionID) {
                ForEach(operation.actions) { item in Text(item.method + " " + item.path).tag(item.id) }
            }.disabled(busy)
            if let action {
                Text("Service role: " + action.serviceRole + ". " + action.authorityReason).font(.caption)
                TextEditor(text: $requestText).font(.system(.body, design: .monospaced)).disabled(busy)
                Toggle("Execute this request on the named server", isOn: $confirmed).disabled(busy)
                HStack {
                    Button("Run selected action") {
                        let captured = requestText; let selected = action.id
                        busy = true; confirmed = false
                        Task {
                            defer { busy = false }
                            do {
                                let value = try await client.callScientificAction(operation: operation.id, actionID: selected, document: Data(captured.utf8))
                                if case .object(let object) = value, case .string(let text) = object["responseJSON"] { resultText = text }
                            } catch { resultText = error.localizedDescription + "\nInspect the server before repeating a mutation whose outcome is uncertain." }
                        }
                    }.disabled(busy || !confirmed)
                    Spacer(); Button("Close") { dismiss() }.disabled(busy)
                }
            }
            ScrollView { Text(resultText).font(.system(.caption, design: .monospaced)).textSelection(.enabled) }
        }.padding().frame(minWidth: 800, minHeight: 650)
        .interactiveDismissDisabled(busy)
        .onAppear { actionID = operation.actions.first?.id ?? "" }
        .onChange(of: actionID) { _, _ in confirmed = false }
        .onChange(of: requestText) { _, _ in confirmed = false }
    }
}
