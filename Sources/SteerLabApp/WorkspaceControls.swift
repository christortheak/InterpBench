import AppKit
import ExperimentKit
import SwiftUI

/// Window-toolbar switcher for the DATA workspace — the folder holding
/// prompts/, experiments/, runs/ (the Compute menu next to it picks the
/// engine). Shows the current workspace's folder name; New/Open create or
/// adopt a folder through `WorkspaceActions` (shared with Home's welcome and
/// Research Setup), which then resets the in-memory catalogs so every panel
/// re-scans the new root.
struct WorkspaceSelector: View {
    @Bindable var workspace: WorkspaceStore
    let service: ChatService
    @Bindable var actions: WorkspaceActions
    /// The three compute choices: what this workspace is set to, and the
    /// actions behind picking one.
    @Bindable var compute: ComputeChoiceCoordinator
    /// The local engine's server, for the setup sheet Research Setup can open.
    let localServer: LocalServerController
    @State private var researchSetup = ResearchSetupModel()
    @AppStorage("SteerLab.researchSetupPresented") private var researchSetupPresented = false

    var body: some View {
        Menu {
            // The folder NAME as the section header: the absolute path used
            // to be the header and widened the whole menu. The full path is
            // one hover away (this menu's help) and on Home.
            Section(workspace.displayName) {
                Button("Research Setup…") { actions.showingResearchSetup = true }
                    .help("check that this Mac can design studies, set up the "
                        + "study-design helper after reviewing its plan, and "
                        + "copy the instructions for your coding assistant")
                Button("New Workspace…") { actions.newWorkspace() }
                    .disabled(workspace.isEnvironmentPinned)
                    .help(
                        "create a new workspace folder (prompts/, experiments/, "
                            + "runs/) and switch every panel to it")
                Button("Open Workspace…") { actions.openWorkspace() }
                    .disabled(workspace.isEnvironmentPinned)
                    .help(
                        "switch to an existing workspace folder — panels re-scan "
                            + "in place; jobs already running keep writing to the "
                            + "previous one")
            }
            // Everything below describes or acts on a workspace that exists.
            if workspace.hasWorkspace {
                Divider()
                computeChoiceSection
                Divider()
                Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([workspace.rootURL])
                }
                .help("show this workspace's folder in Finder")
                if workspace.isEnvironmentPinned {
                    Text("pinned by STEERLAB_WORKSPACE — switch by relaunching without it")
                }
            }
        } label: {
            // The selector itself names where studies run — "<workspace> —
            // This Mac, quick start" or "<workspace> — <machine>" — so the
            // researcher never has to open a menu to know where builds and
            // runs execute. The folder glyph says WHICH of the toolbar's
            // menus this is at a glance (fresh-Mac finding: unlabelled
            // toolbar controls read as decoration, not as the
            // data/compute/connection triple).
            Label(menuTitle, systemImage: "folder")
        }
        .sheet(isPresented: $actions.showingResearchSetup) {
            ResearchSetupSheet(
                model: researchSetup, workspace: workspace, compute: compute,
                service: service, localServer: localServer,
                createWorkspace: { actions.newWorkspace() },
                openWorkspace: { actions.openWorkspace() },
                openDemo: { try actions.openDemoWorkspace($0) },
                demoSheetClosed: { actions.demoSheetClosed() })
        }
        .task {
            // With no workspace yet, Research Setup opens at EVERY launch.
            // The "already shown" flag used to be set before the sheet first
            // appeared, so a newcomer who dismissed it once never saw it
            // again and was left on a Home that had nothing to offer. The
            // flag now starts counting only once a workspace exists.
            let hasWorkspace = workspace.hasWorkspace
            if hasWorkspace, researchSetupPresented { return }
            if hasWorkspace {
                await researchSetup.refresh(workspace: workspace.chosenRootURL)
            }
            guard
                ResearchSetupModel.opensAtLaunch(
                    hasWorkspace: hasWorkspace,
                    alreadyPresented: researchSetupPresented,
                    authoringReady: researchSetup.authoringReady)
            else { return }
            if hasWorkspace { researchSetupPresented = true }
            actions.showingResearchSetup = true
        }
        .labelStyle(.titleAndIcon)
        .help(helpText)
        .alert(
            actions.errorTitle, isPresented: showingError,
            actions: { Button("OK", role: .cancel) { actions.errorMessage = nil } },
            message: { Text(actions.errorMessage ?? "") })
    }

    private var showingError: Binding<Bool> {
        Binding(
            get: { actions.errorMessage != nil },
            set: { if !$0 { actions.errorMessage = nil } })
    }

    /// "<folder> — <where it runs>" while a workspace is open; with none,
    /// just the plain words for that state. Never a placeholder path.
    private var menuTitle: String {
        workspace.hasWorkspace
            ? "\(workspace.displayName) — \(service.cluster.activeComputeTitle)"
            : WorkspaceStore.noWorkspaceDisplayName
    }

    // MARK: Where this workspace's studies run

    /// The workspace's DECLARED compute choice — the fact the lifecycle
    /// reads to decide whose artifacts and evidence are native here.
    ///
    /// Until this control existed the answer was inferred from the live
    /// server pairing, separately, by each verb — and they disagreed, so a
    /// cluster workspace treated its own vectors as foreign and refused
    /// promotions that were entirely legitimate. Declaring it is the point:
    /// it survives the server being offline, unpaired, or moved.
    ///
    /// The three choices are `ComputeChoice`. The second and third both
    /// write today's `cluster` binding; picking one here also switches the
    /// app to it, which for the engine on this Mac opens its setup.
    /// Hoisted out of the `Section` body: as a `+`-chain inside a
    /// `ViewBuilder` this defeated the type-checker ("unable to type-check
    /// this expression in reasonable time"). A named `String` costs nothing.
    private static let computeChoiceHelp: String =
        "where this workspace's studies run. This is a setting of the "
        + "workspace itself, kept with it: vectors and results count for its "
        + "studies only when they were made on the engine chosen here. "
        + "Choosing one also switches the app to it. What Runs Where… "
        + "compares the three"

    @ViewBuilder
    private var computeChoiceSection: some View {
        Section("This workspace runs on") {
            computeChoicePicker
            computeChoiceNotes
            Button(ComputeGuide.guideButton) { compute.showingGuide = true }
                .help("what each of the three choices can run, and what "
                    + "switching between them costs")
        }
    }

    /// One checkable row per choice, rather than a picker: a workspace that
    /// has declared nothing shows NO checkmark, and choosing the row the app
    /// had been assuming is how the researcher confirms it. (A picker would
    /// show the assumption as selected, and re-selecting it would do
    /// nothing.)
    @ViewBuilder
    private var computeChoicePicker: some View {
        ForEach(ComputeChoice.allCases) { choice in
            Toggle(isOn: declared(choice)) {
                Text(choice.title)
                Text(choice.menuCaption)
            }
            .help(Self.computeChoiceHelp)
        }
    }

    @ViewBuilder
    private var computeChoiceNotes: some View {
        if !workspace.isComputeDeclared {
            // An inference must not masquerade as a decision.
            Text(ComputeChoice.undeclaredNote(treatingAs: compute.workspaceChoice))
        }
        if let mismatch = compute.mismatchNote {
            Text(mismatch)
        }
    }

    /// Checked only for a choice the workspace has actually declared. Either
    /// direction of the click chooses it: re-choosing the current one is how
    /// the app is switched back to it.
    private func declared(_ choice: ComputeChoice) -> Binding<Bool> {
        Binding(
            get: { workspace.isComputeDeclared && compute.workspaceChoice == choice },
            set: { _ in compute.choose(choice) })
    }

    private var helpText: String {
        // No workspace: say what one is and how to get one. There is no path
        // to show, and the placeholder that stands in for one is never shown.
        guard workspace.hasWorkspace else { return FirstLaunchCopy.menuHelp }
        var text =
            "the data workspace: the folder holding prompts/, experiments/, and "
            + "runs/ (Compute picks the engine; Workspace picks the data). "
            + "Artifacts, installed models, and jobs belong to one engine — they "
            + "follow the Compute menu — while concepts, recipes, and "
            + "studies are shared data visible from every engine. "
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
}

/// "Install model…" affordance shown next to any picker that lists a server
/// workspace's installed models — an empty server cache gets an obvious next
/// step. Queues a durable prefetch job through the shared store
/// (`ClusterConnectionStore.installModel`), same flow as the toolbar popover.
struct InstallModelButton: View {
    @Bindable var cluster: ClusterConnectionStore
    @State private var showingInstaller = false
    @State private var modelID = ""

    /// Both buttons send this to the server; neither has anything to send
    /// while it is blank (Plan used to POST "" and get a server error back).
    private var trimmedModelID: String {
        modelID.trimmingCharacters(in: .whitespacesAndNewlines)
    }

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
                    .help(
                        "the Hugging Face repository to prefetch, owner/name — "
                            + "full-precision ids only (an MLX repo is refused "
                            + "with its family twin named)")
                HStack {
                    Button("Plan") {
                        Task { await cluster.previewModelPreparation(trimmedModelID) }
                    }
                    .disabled(trimmedModelID.isEmpty)
                    .help(
                        "ask the server what installing this repo would cost — "
                            + "disk, files, and whether it is already cached — "
                            + "without queueing anything")
                    Button("Install") {
                        Task { await cluster.installModel(trimmedModelID) }
                    }
                    .keyboardShortcut(.defaultAction)
                    .disabled(trimmedModelID.isEmpty)
                    .help(
                        "queue the prefetch as a durable job on "
                            + "\(cluster.substrateLabel); it keeps running after "
                            + "this popover closes and shows up in Compute › Jobs")
                    Spacer()
                }
                if cluster.modelPreparation.endpoint == cluster.connectionProfile?.baseURL,
                    cluster.modelPreparation.requestedModelID == modelID, let message = cluster.modelPreparation.message {
                    Text(message).font(.caption).textSelection(.enabled)
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
                    .help(
                        "the Hugging Face repository to download here, "
                            + "owner/name — MLX-quantized repos, since this Mac "
                            + "loads them through MLX")
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
                    .help(
                        "fetch the weights into this Mac's Hugging Face cache "
                            + "now — it keeps running while you work, and a "
                            + "cancelled download resumes where it stopped")
                    if service.modelInstaller.isInstalling {
                        Button("Cancel Download") { service.modelInstaller.cancel() }
                            .help(
                                "stop fetching — the bytes already written stay "
                                    + "in the cache and Download resumes from there")
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
