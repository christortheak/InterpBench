import ExperimentKit
import SwiftUI

/// Uses the exact catalog and guide text shipped to both agent clients.
struct ScienceGuidesView: View {
    var client: ClusterClient? = nil
    var openOptimizations: () -> Void
    var openTemplates: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var methods: [ScienceCatalog.Method] = []
    @State private var selected: String?
    @State private var guide: ScienceCatalog.Guide?
    @State private var operations: [ScienceCatalog.Operation] = []
    @State private var failure: String?
    private struct ActionTarget: Identifiable {
        let id = UUID()
        let operation: ScienceCatalog.Operation
        let client: ClusterClient
    }
    @State private var selectedOperation: ActionTarget?
    private struct AuthoringTarget: Identifiable {
        let id = UUID()
        let workflow: ScienceCatalog.Workflow
        let root: URL
        let client: ClusterClient?
    }
    @State private var authoringTarget: AuthoringTarget?
    @State private var workflows: [ScienceCatalog.Workflow] = []
    @State private var saeRoot: URL?
    @State private var custodyRoot: URL?

    var body: some View {
        VStack(alignment: .leading) {
            HStack {
                Text("Research methods and guides").font(.title2)
                Spacer()
                Button("Local diagnostic evidence…") { custodyRoot = ExperimentStore.workspaceRoot }
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            Text("Choose a method to review its inputs, scientific decisions and supported execution paths.")
                .foregroundStyle(.secondary)
            HSplitView {
                List(methods, selection: $selected) { method in
                    VStack(alignment: .leading) {
                        Text(method.title)
                        Text(method.purpose).font(.caption).foregroundStyle(.secondary)
                    }.tag(method.id)
                }.frame(minWidth: 220, idealWidth: 260)
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        if let failure { Text(failure).foregroundStyle(.red) }
                        if let guide {
                            Text(guide.text).textSelection(.enabled)
                            Divider()
                            Text("Supported operation paths").font(.headline)
                            ForEach(operations) { operation in
                                VStack(alignment: .leading, spacing: 5) {
                                    Text(operation.title).font(.headline)
                                    if let workflow = workflows.first(where: { $0.id == operation.id }) {
                                        Button("Author request…") { authoringTarget = AuthoringTarget(workflow: workflow, root: ExperimentStore.workspaceRoot, client: client) }
                                    }
                                    Text(operation.engineCLI ?? "No direct engine CLI; use the listed HTTP interface.").font(.system(.body, design: .monospaced))
                                    Text(operation.mac)
                                    Text(operation.access.restriction).font(.caption)
                                    if !operation.actions.isEmpty, let client {
                                        Button("Prepare server action…") { selectedOperation = ActionTarget(operation: operation, client: client) }
                                        Text("Target: " + client.profile.baseURL.absoluteString).font(.caption)
                                    }
                                    Text(operation.http ?? "No HTTP execution route for this operation.")
                                    Text(operation.restriction).foregroundStyle(.secondary)
                                }.textSelection(.enabled)
                            }
                            if guide.method.id == "optimization" {
                                Button("Open Optimizations") { dismiss(); openOptimizations() }
                            }
                            if guide.method.id == "sae" {
                                Button("Inspect or pin SAE roster…") { saeRoot = ExperimentStore.workspaceRoot }
                            }
                            if guide.method.id == "multi-agent" {
                                Button("Open study designs") { dismiss(); openTemplates() }
                            }
                            Text("Agents: science guide \(guide.method.id) --json returns this same text. Engine-only operations require the listed engine; discovery does not execute or qualify a study.")
                                .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                        }
                    }.padding()
                }.frame(minWidth: 470)
            }
        }.padding().frame(minWidth: 780, minHeight: 580)
        .sheet(isPresented: Binding(get: { saeRoot != nil }, set: { if !$0 { saeRoot = nil } })) {
            if let root = saeRoot { SAERosterSheet(root: root) }
        }
        .sheet(item: $authoringTarget) { target in
            MethodAuthoringSheet(workflow: target.workflow, root: target.root, client: target.client)
        }
        .sheet(item: $selectedOperation) { target in
            ScientificActionSheet(operation: target.operation, client: target.client)
        }
        .sheet(isPresented: Binding(get: { custodyRoot != nil }, set: { if !$0 { custodyRoot = nil } })) {
            if let root = custodyRoot { DiagnosticLifecycleSheet(root: root, client: nil, initialJobID: nil) }
        }
        .task {
            do {
                workflows = try ScienceCatalog.workflows()
                methods = try ScienceCatalog.catalog().methods
                selected = methods.first?.id
                loadSelection()
            } catch { failure = error.localizedDescription }
        }
        .onChange(of: selected) { _, _ in loadSelection() }
    }

    private func loadSelection() {
        guard let selected else { return }
        do {
            guide = try ScienceCatalog.guide(selected)
            operations = try ScienceCatalog.catalog().operations.filter { $0.method == selected }
            failure = nil
        } catch { guide = nil; operations = []; failure = error.localizedDescription }
    }
}
