import AppKit
import ExperimentKit
import SwiftUI

/// Window-toolbar switcher for the DATA workspace — the folder holding
/// prompts/, experiments/, runs/ (the Compute menu next to it picks the
/// engine). Shows the current workspace's folder name; New/Open create or
/// adopt a folder through `WorkspaceStore` and then reset the in-memory
/// catalogs so every panel re-scans the new root.
struct WorkspaceSelector: View {
    @Bindable var workspace: WorkspaceStore
    let service: ChatService
    let catalog: SubstrateCatalog
    @State private var errorMessage: String?

    var body: some View {
        Menu {
            Section(workspace.rootURL.path) {
                Button("New Workspace…") { newWorkspace() }
                    .disabled(workspace.isEnvironmentPinned)
                Button("Open Workspace…") { openWorkspace() }
                    .disabled(workspace.isEnvironmentPinned)
            }
            Divider()
            computeBindingSection
            Divider()
            Button("Reveal in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([workspace.rootURL])
            }
            if workspace.isEnvironmentPinned {
                Text("pinned by STEERLAB_WORKSPACE — switch by relaunching without it")
            }
        } label: {
            // Substrate-prominence (live-testing finding): the selector
            // itself names the active substrate — "<workspace> — MLX" or
            // "<workspace> — Server: <label>" — so the researcher never has
            // to open a menu to know where builds and runs execute. The
            // folder glyph says WHICH of the toolbar's menus this is at a
            // glance (fresh-Mac finding: unlabelled toolbar controls read as
            // decoration, not as the data/compute/connection triple).
            Label(
                "\(workspace.displayName) — \(substrateSuffix)",
                systemImage: "folder")
        }
        .labelStyle(.titleAndIcon)
        .help(helpText)
        .alert(
            "Workspace", isPresented: showingError,
            actions: { Button("OK") { errorMessage = nil } },
            message: { Text(errorMessage ?? "") })
    }

    private var showingError: Binding<Bool> {
        Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })
    }

    // MARK: What this workspace computes on

    /// The workspace's DECLARED compute engine — the fact the lifecycle reads
    /// to decide whose artifacts and evidence are native here.
    ///
    /// Until this control existed the answer was inferred from the live
    /// server pairing, separately, by each verb — and they disagreed, so a
    /// cluster workspace treated its own vectors as foreign and refused
    /// promotions that were entirely legitimate. Declaring it is the point:
    /// it survives the server being offline, unpaired, or moved.
    @ViewBuilder
    private var computeBindingSection: some View {
        Section("Computes on") {
            Picker("Computes on", selection: computeBinding) {
                ForEach(WorkspaceCompute.allCases, id: \.self) { option in
                    Text(option.label).tag(option)
                }
            }
            .pickerStyle(.inline)
            .labelsHidden()
            if !workspace.isComputeDeclared {
                // An inference must not masquerade as a decision.
                Text("inferred from this workspace's runs — choose to confirm")
            }
            if let mismatch = computeMismatchNote {
                Text(mismatch)
            }
        }
    }

    private var computeBinding: Binding<WorkspaceCompute> {
        Binding(
            get: { workspace.compute },
            set: { choice in
                do { try workspace.declareCompute(choice) } catch {
                    errorMessage = "\(error)"
                }
            })
    }

    /// The Compute selector and the workspace binding disagreeing is worth
    /// saying out loud: it is the state in which a cluster study is about to
    /// be run, extracted, or swept on MLX.
    private var computeMismatchNote: String? {
        let target = service.cluster.activeWorkspace
        switch (workspace.compute, target) {
        case (.cluster, .local):
            return "Compute is set to MLX, but this workspace's data is "
                + "cluster data — switch Compute to the server before running"
        case (.localMLX, .server):
            return "Compute is set to \(service.cluster.substrateLabel), but "
                + "this workspace is declared local — its artifacts are MLX"
        default:
            return nil
        }
    }

    /// "MLX" locally, "Server: <name>" on a server workspace — appended to
    /// the selector label so the active substrate is always visible.
    private var substrateSuffix: String {
        switch service.cluster.activeWorkspace {
        case .local: return "MLX"
        case .server: return "Server: \(service.cluster.substrateLabel)"
        }
    }

    private var helpText: String {
        var text =
            "the data workspace: the folder holding prompts/, experiments/, and "
            + "runs/ (Compute picks the engine; Workspace picks the data). "
            + "Artifacts, installed models, and jobs are per-substrate — they "
            + "follow the Compute selector — while concepts, recipes, and "
            + "studies are shared data visible from every substrate. "
            + "Current: \(workspace.rootURL.path)"
        if workspace.isLegacyRepoRoot {
            text += " — the SteerLab code checkout (dev fallback); create a "
                + "workspace to keep study data out of the source tree"
        }
        text += ". Switching re-scans concepts, experiments, vectors, and "
            + "corpora in place; jobs already running keep writing to the "
            + "previous workspace until restarted."
        return text
    }

    private func newWorkspace() {
        let panel = NSSavePanel()
        panel.title = "New SteerLab Workspace"
        panel.prompt = "Create"
        panel.nameFieldStringValue = "SteerLab Workspace"
        panel.canCreateDirectories = true
        panel.showsTagField = false
        // Creation is the one moment the answer is never ambiguous, so ask
        // here rather than leaving a new workspace to be inferred later.
        // Defaults to Cluster: real studies compute there and MLX is for toy
        // runs and shakedowns.
        let chooser = ComputeChoiceAccessory(selected: .cluster)
        panel.accessoryView = chooser.view
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try workspace.createAndSwitch(to: url, computing: chooser.selected)
            resetCatalogs()
        } catch {
            errorMessage = "\(error)"
        }
    }

    private func openWorkspace() {
        let panel = NSOpenPanel()
        panel.title = "Open SteerLab Workspace"
        panel.prompt = "Open"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try workspace.switchTo(url)
            resetCatalogs()
        } catch {
            errorMessage = "\(error)"
        }
    }

    /// The existing refresh entry points, called once after a switch so
    /// panels drop state scanned from the previous root. Anything a panel
    /// caches outside these paths refreshes on its next interaction.
    private func resetCatalogs() {
        // Section-specific DISPLAY state first: the catalog refreshes below
        // re-scan lists, but none of them retired the viewer's selection, so
        // after a switch the Results viewer still showed the previous
        // workspace's run (Finder button and all) and Analysis still showed
        // its cosine tables. Each viewer has an empty state; it just needed
        // its selection dropped. The workspace-wide Activity log is kept.
        service.resetSectionViewers()
        service.experiments.refresh()
        service.datasetInventory.refresh()
        service.concepts.refreshConceptList()
        service.concepts.refreshStaleness()
        service.concepts.refreshReaderTemplates()
        service.concepts.refreshReaderArtifacts()
        service.fineTuning.refresh()
        service.refreshVectors()
        service.refreshNeutralCorpora()
        service.refreshNeutralPCBases()
        catalog.refreshLocalVectors()
    }

    // Same-machine server auto-switching is NOT a view concern: it lives on
    // `ClusterConnectionStore.synchronizeServerToLocalWorkspace()`, triggered
    // once at the workspace-root-change seam (`WorkspaceStore.onRootChange`,
    // wired in `SteerLabApp`), so every root-change path — including ones
    // this view never sees — gets the same serialized, surfaced behavior.
}

/// "Install model…" affordance shown next to any picker that lists a server
/// workspace's installed models — an empty server cache gets an obvious next
/// step. Queues a durable prefetch job through the shared store
/// (`ClusterConnectionStore.installModel`), same flow as the toolbar popover.
struct InstallModelButton: View {
    @Bindable var cluster: ClusterConnectionStore
    @State private var showingInstaller = false
    @State private var modelID = ""

    var body: some View {
        Button {
            showingInstaller = true
        } label: {
            Label("Install model…", systemImage: "square.and.arrow.down.on.square")
        }
        .help(
            "prefetch a Hugging Face repo into \(cluster.substrateLabel)'s cache "
                + "as a durable job (full-precision HF ids — MLX repos are "
                + "rejected with a family-twin hint)")
        .popover(isPresented: $showingInstaller, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Install model on \(cluster.substrateLabel)")
                    .font(.headline)
                TextField("HF repo id (e.g. Qwen/Qwen3-4B)", text: $modelID)
                    .textFieldStyle(.roundedBorder)
                    .frame(minWidth: 280)
                HStack {
                    Button("Install") {
                        Task { await cluster.installModel(modelID) }
                    }
                    .disabled(modelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    Spacer()
                }
                if let status = cluster.status {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: 300, alignment: .leading)
                }
            }
            .padding(12)
        }
    }
}

/// Local (MLX) twin of `InstallModelButton`: install a model into THIS Mac's
/// Hugging Face cache by slug, through `ChatService.installWorkspaceModel` →
/// `LocalModelInstaller`, so it lands in the same installed-models registry
/// every builder's selector reads. Shows the in-flight percentage and a
/// Cancel, because the thing it starts is measured in gigabytes.
struct AddLocalModelButton: View {
    @Bindable var service: ChatService
    @State private var showingInstaller = false
    @State private var modelID = ""

    var body: some View {
        Button {
            showingInstaller = true
        } label: {
            Label("Add Model…", systemImage: "square.and.arrow.down.on.square")
        }
        .help(
            "download a Hugging Face repo into this Mac's model cache "
                + "(~/.cache/huggingface) so it can be loaded here and picked "
                + "in the builders — MLX-quantized repos, e.g. "
                + "mlx-community/gemma-3-4b-it-4bit")
        .popover(isPresented: $showingInstaller, arrowEdge: .bottom) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Add Model to This Mac")
                    .font(.headline)
                TextField(
                    "HF repo id (e.g. mlx-community/gemma-3-4b-it-4bit)",
                    text: $modelID)
                    .textFieldStyle(.roundedBorder)
                    .frame(minWidth: 300)
                Text("Downloads the full weights — typically 3–35 GB. It keeps "
                    + "running while you work, and a cancelled download resumes.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: 320, alignment: .leading)
                HStack(spacing: 8) {
                    Button("Download") {
                        Task { await service.installWorkspaceModel(modelID) }
                    }
                    .keyboardShortcut(.defaultAction)
                    .disabled(
                        service.modelInstaller.isInstalling
                            || modelID.trimmingCharacters(in: .whitespacesAndNewlines)
                                .isEmpty)
                    if service.modelInstaller.isInstalling {
                        Button("Cancel Download") { service.modelInstaller.cancel() }
                    }
                    Spacer()
                }
                if service.modelInstaller.isInstalling {
                    ProgressView(value: installFraction)
                        .frame(maxWidth: 320)
                }
                if let status = service.modelInstaller.statusLine {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: 320, alignment: .leading)
                }
            }
            .padding(12)
        }
    }

    private var installFraction: Double {
        if case .installing(_, let percent) = service.modelInstaller.phase {
            return Double(percent) / 100
        }
        return 0
    }
}

/// Model picker over the active workspace's installed models (Local = the
/// app's pinned tiers, server = that server's inventory), bound to the shared
/// workspace selection on `ChatService`.
///
/// Strict availability: a server workspace offers ONLY that server's
/// installed models. A current selection the workspace does not have is
/// still *rendered* (so SwiftUI never silently drops the binding) but is
/// labeled "(not installed)" and selection-disabled — it can never be picked
/// as a new choice, and it disappears from the menu the moment the selection
/// moves to an installed model. An empty server inventory shows an empty
/// picker plus a caption pointing at Install model…, never the local tiers.
struct WorkspaceModelPicker: View {
    @Bindable var service: ChatService
    var label = "Model"

    var body: some View {
        Picker(label, selection: $service.workspaceSelectedModelID) {
            if service.workspaceSelectedModelID == nil {
                Text("select model…").tag(String?.none)
            }
            ForEach(installed, id: \.self) { model in
                // Models whose weights cannot fit the LIVE session's GPU are
                // unselectable with a reason (2026-07-18: a 22.7 GiB model
                // staged 15 minutes onto a 22 GiB L4, then OOM'd). Mirror of
                // the server's own load preflight; no session → no gating.
                if let note = SessionModelFit.tooBigNote(
                    cluster: service.cluster, model: model)
                {
                    Text("\(model) — \(note)")
                        .tag(String?.some(model))
                        .selectionDisabled()
                } else {
                    // Availability, per row. On a fresh Mac none of the pinned
                    // tiers are downloaded, and a picker that hides that turns
                    // Load into a silent multi-gigabyte fetch. Not-installed
                    // rows stay SELECTABLE — selecting one is how you choose
                    // what to download — but they say so.
                    Text(rowTitle(for: model)).tag(String?.some(model))
                }
            }
            if let selected = service.workspaceSelectedModelID,
                !installed.contains(selected)
            {
                Text(isServerWorkspace ? "\(selected) (not installed)" : selected)
                    .tag(String?.some(selected))
                    .selectionDisabled()
            }
        }
        .help(
            service.cluster.activeWorkspace == .local
                ? "the dual-track local model tiers plus anything else in this "
                    + "Mac's Hugging Face cache; rows marked \"not downloaded\" "
                    + "need Download before they can load. Vectors are "
                    + "model-specific, so artifact lists follow the loaded model"
                : "models installed on \(service.cluster.substrateLabel) — use "
                    + "Install model… to prefetch another")
        // Cheap directory listing; run whenever the picker appears so a model
        // installed elsewhere (a builder, another window, the CLI) shows up.
        .task { service.catalog.refreshLocalInstalledModels() }
        if isServerWorkspace, installed.isEmpty {
            Text(
                "no models installed on \(service.cluster.substrateLabel) — "
                    + "use Install model… to prefetch one")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private var isServerWorkspace: Bool {
        service.cluster.computeTarget == .server
    }

    private var installed: [String] {
        service.workspaceModelOptions
    }

    /// "<id>" when the weights are present, "<id> — not downloaded" when they
    /// are not, "<id> — downloading N%" while an install runs.
    private func rowTitle(for model: String) -> String {
        guard !isServerWorkspace else { return model }
        if case .installing(let installing, let percent) = service.modelInstaller.phase,
            installing == model
        {
            return "\(model) — downloading \(percent)%"
        }
        return service.catalog.isInstalled(model, in: .local)
            ? model : "\(model) — not downloaded"
    }
}
