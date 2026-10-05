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
    /// The selected guide arranged for reading, parsed once per selection.
    @State private var layout: GuideMarkdown.Layout?
    /// The coding-assistant part starts folded: a researcher reading the
    /// guide sees the method first, and the commands are one click away.
    @State private var showsAssistantPart = false
    @State private var operations: [ScienceCatalog.Operation] = []
    /// The whole catalog, read once: selecting a method used to re-read and
    /// re-decode it on every change (2026-09-06 audit).
    @State private var allOperations: [ScienceCatalog.Operation] = []
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
    @State private var didLoad = false

    var body: some View {
        VStack(alignment: .leading) {
            HStack {
                Text("Research methods and guides").font(.title2)
                Spacer()
                Button("Local diagnostic evidence…") { custodyRoot = ExperimentStore.workspaceRoot }
                    .help("verify and list the evidence receipts already in this workspace — reads only")
                Button("Done", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .help("close this reference — it changes nothing in the workspace")
            }
            Text("Choose a method to review its inputs, scientific decisions, and supported execution paths.")
                .foregroundStyle(.secondary)
            HSplitView {
                List(methods, selection: $selected) { method in
                    VStack(alignment: .leading) {
                        Text(method.title)
                        Text(method.purpose).font(.caption).foregroundStyle(.secondary)
                    }.tag(method.id)
                }
                .frame(minWidth: 220, idealWidth: 260)
                .help("the methods this workspace ships guidance for — pick one to read it")
                .overlay {
                    if didLoad, methods.isEmpty, failure == nil {
                        Text("No method guides are installed in this build.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .padding()
                    }
                }
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        if let failure { Text(failure).foregroundStyle(.red) }
                        if let guide, let layout {
                            // For a researcher, in this order: what the
                            // method is for, the shapes its data takes, the
                            // guide itself, and the ways to run it. Command
                            // lines and the prompts written for an assistant
                            // follow in their own labelled part (2026-10
                            // release review, D6); `GuideMarkdown` decides
                            // where each block of the guide goes.
                            purpose(layout, guide: guide)
                            dataShapes(layout)
                            guideBody(layout)
                            if !operations.isEmpty {
                                Divider()
                                Text("Supported operation paths").font(.headline)
                            }
                            ForEach(operations) { operation in
                                VStack(alignment: .leading, spacing: 5) {
                                    Text(operation.title).font(.headline)
                                    if let workflow = workflows.first(where: { $0.id == operation.id }) {
                                        Button("Author request…") { authoringTarget = AuthoringTarget(workflow: workflow, root: ExperimentStore.workspaceRoot, client: client) }
                                    }
                                    if !operation.outputs.isEmpty {
                                        Text("Produces: " + operation.outputs.joined(separator: "; "))
                                            .font(.caption)
                                    }
                                    if !operation.actions.isEmpty, let client {
                                        Button("Prepare server action…") { selectedOperation = ActionTarget(operation: operation, client: client) }
                                        Text("Target: " + client.profile.baseURL.absoluteString).font(.caption)
                                    }
                                    Text(operation.restriction).foregroundStyle(.secondary)
                                        .fixedSize(horizontal: false, vertical: true)
                                }.textSelection(.enabled)
                            }
                            if guide.method.id == "optimization" {
                                Button("Open Optimizations") { dismiss(); openOptimizations() }
                                    .help(
                                        "close this reference and land on Agents → "
                                            + "Optimizations, where these runs are declared")
                            }
                            if guide.method.id == "sae" {
                                Button("Inspect or pin SAE roster…") { saeRoot = ExperimentStore.workspaceRoot }
                            }
                            if guide.method.id == "multi-agent" {
                                // The section is called Templates; "study
                                // designs" was a name it never had in the
                                // sidebar (2026-09-06 audit).
                                Button("Open Templates") { dismiss(); openTemplates() }
                                    .help(
                                        "close this reference and land on Templates, "
                                            + "the template library these scenarios are cast from")
                            }
                            Divider()
                            forCodingAssistants(layout, guide: guide)
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
                let catalog = try ScienceCatalog.catalog()
                methods = catalog.methods
                allOperations = catalog.operations
                selected = methods.first?.id
                loadSelection()
            } catch { failure = error.localizedDescription }
            didLoad = true
        }
        .onChange(of: selected) { _, _ in loadSelection() }
    }

    private func loadSelection() {
        guard let selected else { return }
        do {
            let loaded = try ScienceCatalog.guide(selected)
            guide = loaded
            layout = GuideMarkdown.layout(loaded.text)
            operations = allOperations.filter { $0.method == selected }
            failure = nil
        } catch { guide = nil; layout = nil; operations = []; failure = error.localizedDescription }
    }

    // MARK: The guide's parts

    /// The guide's title and the sentence that says what the method is for.
    @ViewBuilder
    private func purpose(_ layout: GuideMarkdown.Layout, guide: ScienceCatalog.Guide) -> some View {
        Text(layout.title ?? guide.method.title)
            .font(.title2.bold())
            .fixedSize(horizontal: false, vertical: true)
            .accessibilityAddTraits(.isHeader)
        GuideBlocksView(blocks: layout.purpose)
            .font(.title3)
            .textSelection(.enabled)
    }

    /// Every table and data example, each under the heading it sat beneath
    /// in the guide and with the sentence that introduced it.
    @ViewBuilder
    private func dataShapes(_ layout: GuideMarkdown.Layout) -> some View {
        if !layout.dataShapes.isEmpty {
            Text(GuideMarkdown.dataShapesTitle)
                .font(.title3.weight(.semibold))
                .accessibilityAddTraits(.isHeader)
            ForEach(Array(layout.dataShapes.enumerated()), id: \.offset) { index, shape in
                VStack(alignment: .leading, spacing: 6) {
                    // A heading once per run of shapes from the same place.
                    if let section = shape.section,
                        index == 0 || layout.dataShapes[index - 1].section != section
                    {
                        Text(GuideMarkdown.inline(section)).font(.headline)
                    }
                    if let caption = shape.caption {
                        Text(GuideMarkdown.inline(caption))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    GuideBlockView(block: shape.block)
                }
                .textSelection(.enabled)
            }
        }
    }

    /// The rest of the guide, in its own order, with a note where a command
    /// line was moved to the coding-assistant part.
    @ViewBuilder
    private func guideBody(_ layout: GuideMarkdown.Layout) -> some View {
        if !layout.body.isEmpty {
            Divider()
            VStack(alignment: .leading, spacing: 10) {
                ForEach(Array(layout.body.enumerated()), id: \.offset) { _, element in
                    switch element {
                    case .block(let block):
                        GuideBlockView(block: block)
                    case .movedToAssistants:
                        Label(GuideMarkdown.movedNote, systemImage: "terminal")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .textSelection(.enabled)
        }
    }

    /// Command lines, the prompts written for an assistant, and each
    /// operation's engine, Mac, and HTTP routes, folded under one heading.
    private func forCodingAssistants(
        _ layout: GuideMarkdown.Layout, guide: ScienceCatalog.Guide
    ) -> some View {
        DisclosureGroup(isExpanded: $showsAssistantPart) {
            VStack(alignment: .leading, spacing: 14) {
                Text(GuideMarkdown.assistantsIntroduction)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                ForEach(Array(layout.forCodingAssistants.enumerated()), id: \.offset) { _, part in
                    VStack(alignment: .leading, spacing: 8) {
                        if let section = part.section {
                            Text(GuideMarkdown.inline(section)).font(.headline)
                        }
                        GuideBlocksView(blocks: part.blocks, copyableCode: true)
                    }
                }
                if !operations.isEmpty {
                    Text("Operation routes").font(.headline)
                    ForEach(operations) { operation in
                        VStack(alignment: .leading, spacing: 5) {
                            Text(operation.title).font(.subheadline.weight(.semibold))
                            if let command = operation.engineCLI {
                                GuideCodeView(text: command, copyable: true)
                            } else {
                                Text("No direct engine CLI; use the listed HTTP interface.")
                                    .font(.caption)
                            }
                            Text(operation.mac)
                            Text(operation.http ?? "No HTTP execution route for this operation.")
                            Text(operation.access.restriction).font(.caption)
                        }
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                    }
                }
                Text(
                    "Command line: steerlab-cli science guide "
                        + "\(guide.method.id) --json returns this same "
                        + "text. Engine-only operations require the "
                        + "listed engine; discovery does not execute or "
                        + "qualify a study.")
                    .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
            .padding(.top, 6)
        } label: {
            Text(GuideMarkdown.assistantsTitle)
                .font(.title3.weight(.semibold))
                .accessibilityAddTraits(.isHeader)
                .help(
                    "command lines and prompts for a coding assistant working in "
                        + "this workspace on your behalf")
        }
    }
}
