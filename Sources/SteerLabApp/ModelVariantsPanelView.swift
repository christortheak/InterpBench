import ExperimentKit
import SteeringKit
import SwiftUI

/// Creation modes within Agents → New Agent
/// (docs/AGENT_CREATION_SWEEP_UI_RECOMMENDATION.md). Adapter-based and
/// baseline agents are manual compositions (the editor includes the adapter
/// picker; zero interventions is a named baseline); importing happens via
/// Playground upload or Library → "Apply in this workspace".
enum AgentCreateMode: String, CaseIterable, Identifiable {
    case manual = "Manual composition"
    case optimize = "Optimize from concept vector"

    var id: String { rawValue }

    var explainer: String {
        switch self {
        case .manual:
            "compose a definition by hand — base model, vector injections, "
                + "adapter, prompt settings. Hand-created agents stay legal but "
                + "permanently exploratory (studies surface their provenance); "
                + "adapter-only and baseline agents are manual compositions too."
        case .optimize:
            "the evidence-grade path: a declared layer×alpha sweep selects the "
                + "winning cell by a pre-declared criterion, and the agent is "
                + "created from that cell with a birth certificate."
        }
    }
}

struct ModelVariantsPanelView: View {
    @State private var showingPolicies = false
    @State private var interventionPolicies: [JSONValue]?

    private struct InjectionDraft: Identifiable, Equatable {
        let id = UUID()
        var vectorArtifactID: VectorArtifact.ID?
        /// Concept recorded by the loaded definition for this ref (nil for
        /// rows added in the editor). Lets an unresolvable ref — e.g. a
        /// server-composed definition opened in a Local workspace — round-trip
        /// untouched through save instead of being silently dropped.
        var originalConcept: String?
        var layer: Int
        /// α when steering, λ when ablating.
        var alpha: Double
        var mode: InterventionPlan.Mode = .add

        var logic: VariantInjectionEditing.Draft {
            VariantInjectionEditing.Draft(
                vectorArtifactID: vectorArtifactID,
                originalConcept: originalConcept,
                layer: layer,
                alpha: alpha,
                mode: mode)
        }
    }

    @Bindable var service: ChatService
    /// Active region within the Agents section (Library / New Agent /
    /// Optimizations) — owned by the workbench shell so cross-section links
    /// can land on a specific region.
    @Binding var region: AgentsRegion
    /// Section navigation (Chat routes to Playground, empty states route to
    /// Data/Studies); injected by the workbench shell.
    var navigate: (WorkbenchSection) -> Void = { _ in }
    /// Creation mode within the New Agent region.
    @State private var createMode: AgentCreateMode = .manual
    /// A just-declared optimization to preselect when the Optimizations
    /// region opens (the New Agent → Optimize flow lands on its new run).
    @State private var pendingOptimizationSelection: String?
    @State private var confirmDelete = false
    @State private var editorLoadedID: String?
    // Agent Library filters (pure rules in AgentLibrary.Filter).
    @State private var filterBaseModel: String?
    @State private var filterSweepOnly = false
    @State private var filterHasAdapter = false
    @State private var filterRunnableHere = false
    @State private var name = "agent-1"
    @State private var baseModelID = ChatService.availableModels.first?.id ?? ""
    @State private var baseRevision = ""
    @State private var adapterID: String?
    /// Tells the base-model onChange whether the change came from loading an
    /// artifact (preserve the artifact's adapter/basis exactly, nil included)
    /// or from the user (clear them — never auto-pick a replacement). Pure
    /// rule in ExperimentKit (`AgentEditorSelection`), unit-tested.
    @State private var modelChangeClassifier = AgentEditorSelection.ChangeClassifier()
    @State private var injections: [InjectionDraft] = []
    @State private var bandWidth = 1
    @State private var alphaInNormUnits = true
    @State private var neutralPCBasisID: String?
    @State private var promptMode: ExperimentManifest.PromptMode = .chatAssistant
    @State private var qwenThinkingEnabled = false
    @State private var temperature = 0.0
    @State private var systemPrompt = ""

    private var panel: FineTuningPanel { service.fineTuning }

    var body: some View {
        VStack(spacing: 0) {
            regionBar
            Button("Intervention policies…") { showingPolicies = true }.padding(8)
            Divider()
            regionContent
        }
        .sheet(isPresented: $showingPolicies) { InterventionPoliciesView(root: ExperimentStore.workspaceRoot) }
        .confirmationDialog(
            deleteDialogTitle,
            isPresented: $confirmDelete,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) { panel.deleteSelectedVariant() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(
                "Removes the definition from this workspace's agent library. "
                    + "Studies that already pinned it keep their own hashed "
                    + "copy, and frozen studies and run directories are "
                    + "untouched. This cannot be undone.")
        }
        // Nothing on the tab-switch path may touch the disk (2026-08-27):
        // this used to be an `onAppear` that ran the library scan, a runs/
        // walk, and a full-file hash of every saved agent SYNCHRONOUSLY,
        // which is exactly the latency the switch showed and exactly why it
        // scaled with the number of promoted agents. `.task` runs after the
        // first draw, and both scans now run off the main actor.
        .task {
            panel.refreshAgentLibraryAsync()
            loadEditorForSelection()
            // Rows carried over from a previous visit are already on screen;
            // give them their evidence without waiting for the rescan to
            // report a CHANGE it may not have.
            refreshAgentEvidenceIfShown()
        }
        // The list landed (or changed): seed the editor if it was waiting on
        // the scan, and start the deferred evidence pass.
        .onChange(of: panel.agentIndex) {
            loadEditorForSelection()
            refreshAgentEvidenceIfShown()
        }
        // Entering the Library is what makes the robustness captions visible;
        // the other regions never render them, so they never pay for them.
        .onChange(of: region) { refreshAgentEvidenceIfShown() }
        // A completed robustness check lands in a new run directory — pick
        // its report up without waiting for the next panel appearance.
        .onChange(of: panel.lastRobustnessDirectory) {
            refreshAgentEvidenceIfShown()
        }
    }

    /// The delete dialog names the agent it is about to remove — "Delete
    /// selected agent?" put the researcher one keystroke from losing a
    /// definition without ever showing which one.
    private var deleteDialogTitle: String {
        guard let agentName = panel.selectedVariant?.artifact.name else {
            return "Delete the selected agent?"
        }
        return "Delete the agent '\(agentName)'?"
    }

    /// Start the deferred robustness overlay — one runs/ scan plus a content
    /// hash per agent, off the main actor (`AgentLibraryIndex.evidence`) —
    /// but only for the region that renders it.
    private func refreshAgentEvidenceIfShown() {
        guard region == .library else { return }
        panel.refreshAgentEvidenceAsync()
    }

    // MARK: Regions (Library / New Agent / Optimizations)

    private var regionBar: some View {
        HStack(spacing: 10) {
            Picker("Region", selection: $region) {
                ForEach(AgentsRegion.allCases) { item in
                    Text(item.rawValue).tag(item)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .help(
                "Library: browse and check saved agents · New Agent: create one "
                    + "(manually or by optimizing a concept vector) · "
                    + "Optimizations: declared sweep runs and Create Agent")
            // A15: the persistent-notices bell (shared feed with Studies).
            NoticesBellButton()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    @ViewBuilder
    private var regionContent: some View {
        switch region {
        case .library:
            libraryForm
        case .create:
            createForm
        case .optimizations:
            // The Optimizations region: optimization runs (declared
            // sweeps), grids, recommendations, Create Agent.
            OptimizationRunsView(
                service: service, navigate: navigate,
                pendingSelection: $pendingOptimizationSelection)
        }
    }

    private var libraryForm: some View {
        Form {
            // ONE scoping rule (WorkspaceScoping.artifactListPresentation):
            // any server target shows the SERVER's stored agents — labeled
            // with the workspace they actually live in — with the mismatch
            // banner when the server serves a different tree than the app's
            // selected workspace. The local *definitions* stay visible below
            // (recipes never switch), with substrate actions gated.
            if service.cluster.artifactListPresentation.showsServerArtifacts {
                if service.cluster.artifactListPresentation.showsMismatchBanner {
                    Section {
                        WorkspaceMismatchBanner(cluster: service.cluster)
                    }
                }
                serverVariantsSection
            }
            agentLibrarySection
            robustnessSection
            statusSection
        }
        .formStyle(.grouped)
    }

    private var createForm: some View {
        Form {
            createModeSection
            switch createMode {
            case .manual:
                definitionEditorSection
                actionsSection
            case .optimize:
                optimizeCreateSection
            }
            statusSection
        }
        .formStyle(.grouped)
    }

    private var createModeSection: some View {
        // A saved agent opened with Edit lands here, and a section titled
        // "New Agent" above a button reading "Save Changes" told two
        // different stories about the same form.
        Section(panel.selectedVariant == nil ? "New Agent" : "Edit Agent") {
            Picker("Creation mode", selection: $createMode) {
                ForEach(AgentCreateMode.allCases) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .help(
                "how this agent gets its layer and α: by hand, or from a "
                    + "declared layer×alpha sweep whose winning cell carries "
                    + "a birth certificate")
            Text(createMode.explainer)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var definitionEditorSection: some View {
        Section("Definition editor") {
            if service.cluster.computeTarget == .server {
                Text(
                    "agent definitions are git-versioned recipes, visible in "
                        + "every workspace. One whose base model and vector/adapter "
                        + "refs all resolve on \(service.cluster.substrateLabel) can "
                        + "be applied here directly (sent as an inline spec); to "
                        + "store it on the server, upload it from the Playground pane")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Picker(
                "Agent",
                selection: Binding<String?>(
                    get: { panel.selectedVariantID },
                    set: { id in
                        panel.selectedVariantID = id
                        loadEditorForSelection()
                    })
            ) {
                Text("New…").tag(String?.none)
                ForEach(panel.variants) { variant in
                    Text(variant.artifact.name).tag(String?.some(variant.id))
                }
            }
            .help(
                "browse or edit saved agents; New… starts a fresh definition "
                    + "from the Playground's current setup")
            if let selected = panel.selectedVariant {
                Text(Self.editingCaption(selected.artifact.name))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            editorFields
        }
    }

    /// The doc's "Sweep / Optimize From Concept Vector" creation mode: an
    /// agent born from a declared layer×alpha sweep. The Optimization
    /// Composer creates the real optimization manifest itself (objective
    /// first, vector recipes pinned from the catalog, instruments as
    /// workspace files, criterion verbatim) and lands in the Optimizations
    /// region with the new run selected — that region is the one home for
    /// grids, recommendations, and Create Agent.
    @ViewBuilder
    private var optimizeCreateSection: some View {
        OptimizationComposerView(
            service: service,
            navigate: navigate,
            onDeclared: { name in
                pendingOptimizationSelection = name
                region = .optimizations
            },
            openOptimizations: { region = .optimizations })
    }

    @ViewBuilder
    private var statusSection: some View {
        if let status = panel.status {
            Section("Status") {
                Text(status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
    }

    private static func editingCaption(_ name: String) -> String {
        "editing the saved agent '\(name)' — Save Changes updates it in "
            + "place; choose New… above to start a fresh definition instead"
    }

    /// The trimmed editor name, once.
    private var trimmedName: String {
        name.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// A saved agent that already answers to this name and is NOT the one
    /// being edited. `ModelVariantStore.save` mints a unique run directory
    /// per save, so a second save under an existing name silently produced
    /// two library rows distinguishable only by their timestamp.
    private var collidingAgentName: String? {
        let candidate = trimmedName
        guard !candidate.isEmpty else { return nil }
        let selectedID = panel.selectedVariant?.id
        return panel.variants.first { record in
            record.id != selectedID
                && record.artifact.name
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .compare(candidate, options: .caseInsensitive) == .orderedSame
        }?.artifact.name
    }

    /// Why Save is disabled, in one plain sentence — nil means saveable.
    private var saveDisabledReason: String? {
        if trimmedName.isEmpty { return "name the agent before saving" }
        if let existing = collidingAgentName {
            return "an agent named '\(existing)' is already in this "
                + "workspace's library — rename this one, or select it in "
                + "the Agent picker above to save changes to it"
        }
        return nil
    }

    private var actionsSection: some View {
        Section("Actions") {
                // Four regular-size buttons overrun the controls column's
                // 560pt minimum; the second layout is the same buttons
                // stacked (`ConceptsPanelView.emotionStoryEntry` idiom).
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) { actionButtons }
                    VStack(alignment: .leading, spacing: 6) { actionButtons }
                }
                .help("save, apply, or reset this agent")

                if let reason = saveDisabledReason {
                    Label(reason, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }

                // Non-destructive applicability caption: a definition that
                // doesn't resolve here names exactly what is missing instead
                // of a generic "needs Local".
                if isServerWorkspace, panel.selectedVariant != nil,
                    case .blocked(let reasons)? = selectedDefinitionApplicability
                {
                    Text(
                        "not applicable on \(service.cluster.substrateLabel): "
                            + reasons.joined(separator: " · "))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
        }
    }

    /// Help strings built OUTSIDE the view builder (this file's own rule).
    private var saveButtonHelp: String {
        if let saveDisabledReason { return saveDisabledReason }
        return panel.selectedVariant == nil
            ? "write this definition into the workspace's agent library as a "
                + "new artifact"
            : "overwrite the selected agent's definition in place"
    }

    private var applyInWorkspaceHelp: String {
        "seed the Playground's steering controls from this definition — it is "
            + "sent to \(service.cluster.substrateLabel) as an inline spec, "
            + "not stored there"
    }

    private var useInPlaygroundHelp: String {
        panel.selectedVariant == nil
            ? "save this definition, then load it into the Playground's "
                + "steering controls"
            : "load this agent's configuration into the Playground's steering "
                + "controls"
    }

    @ViewBuilder
    private var actionButtons: some View {
        Button(panel.selectedVariant == nil ? "Save New Agent" : "Save Changes") {
            saveVariant()
        }
        .disabled(saveDisabledReason != nil)
        .help(saveButtonHelp)

        if isServerWorkspace {
            // A definition whose base model + refs all resolve on the active
            // server seeds the live controls exactly like a stored server
            // variant — but INLINE-composed: no stored path/hash identity is
            // claimed.
            Button("Apply in this workspace") {
                if let record = panel.selectedVariant {
                    service.applyLocalDefinitionToServerSteering(record)
                }
            }
            .disabled(
                panel.selectedVariant == nil
                    || selectedDefinitionApplicability?.isApplicable != true)
            .help(applyInWorkspaceHelp)
        } else {
            Button("Use in Playground") {
                if let record = panel.selectedVariant {
                    service.applyModelVariantToSteering(record)
                } else {
                    saveVariant(applyAfterSave: true)
                }
            }
            .disabled(panel.selectedVariant == nil && saveDisabledReason != nil)
            .help(useInPlaygroundHelp)
        }

        Button("Reset From Playground") {
            loadDraftFromCurrentSteering()
        }
        .help(
            "discard the edits in this form and re-read the Playground's "
                + "current model, injections, and prompt settings")

        if panel.selectedVariant != nil {
            Button("Delete…", role: .destructive) {
                confirmDelete = true
            }
            .help("remove this agent definition from the library — asks first")
        }
    }

    private var robustnessSection: some View {
        @Bindable var panel = service.fineTuning
        return Section("Robustness Check") {
                // The check targets its OWN picked agent — never inherited
                // from the browser/editor selection above (live-testing
                // finding: the implicit coupling was unreadable). Sources
                // follow the ONE scoping rule: under a server target the
                // picker offers the SERVER's stored agents (the same source
                // as the agents list above) plus local definitions (runnable
                // there as inline specs); under Local, local records only.
                robustnessTargetPicker
                Picker(
                    "Preset",
                    selection: Binding(
                        get: { panel.robustnessPresetID },
                        set: { panel.applyRobustnessPreset($0) })
                ) {
                    ForEach(VariantRobustness.presets) { preset in
                        Text(preset.label).tag(preset.id)
                    }
                    Text("Custom").tag(VariantRobustness.customPresetID)
                }
                .help("choose a benchmark-inspired robustness preset; Custom keeps your edited file paths")
                if let preset = VariantRobustness.preset(id: panel.robustnessPresetID) {
                    Text(preset.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                DisclosureGroup("Files") {
                    Text(
                        "the two JSONL instruments this check reads — editing "
                            + "either switches the preset to Custom")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    TextField(
                        "Capability battery",
                        text: Binding(
                            get: { panel.robustnessBatteryFile },
                            set: {
                                panel.robustnessBatteryFile = $0
                                panel.robustnessPresetID = VariantRobustness.customPresetID
                            }))
                        .help("JSONL deterministic capability battery, usually under prompts/batteries/")
                    if !panel.robustnessBatteryFile.trimmingCharacters(in: .whitespaces).isEmpty {
                        FileReferenceRow(
                            label: "battery",
                            path: panel.robustnessBatteryFile)
                    }
                    TextField(
                        "Coherence prompts",
                        text: Binding(
                            get: { panel.robustnessPromptsFile },
                            set: {
                                panel.robustnessPromptsFile = $0
                                panel.robustnessPresetID = VariantRobustness.customPresetID
                            }))
                        .help("JSONL prompt file used for baseline-vs-agent coherence comparisons")
                    if !panel.robustnessPromptsFile.trimmingCharacters(in: .whitespaces).isEmpty {
                        FileReferenceRow(
                            label: "prompts",
                            path: panel.robustnessPromptsFile)
                    }
                }
                .help(
                    "the battery and coherence-prompt files this check reads "
                        + "— edit either to leave the preset and go Custom")
                // Prompts is a COUNT, not a fixed knob: it gets the same
                // affordance Max tokens has (a typed field plus stepper
                // arrows), and it states the selected file's actual supply
                // beside it. Both labels are `fixedSize` — "Max tokens" used
                // to wrap across three lines while the stepper was crushed
                // against the right edge. No frame minimums here (macOS 27):
                // the fields are fixed-width, the labels intrinsic, and the
                // captions compressible.
                let promptsBinding = Binding(
                    get: { panel.robustnessMaxPrompts },
                    set: { requested in
                        let value = RobustnessPromptSupply.clamp(requested)
                        panel.robustnessMaxPrompts = value
                        if VariantRobustness.preset(id: panel.robustnessPresetID)?
                            .maxCoherencePrompts != value
                        {
                            panel.robustnessPresetID = VariantRobustness.customPresetID
                        }
                    })
                HStack(spacing: 8) {
                    Text("Prompts")
                        .fixedSize()
                    TextField(
                        "Prompts", value: promptsBinding,
                        format: .number.grouping(.never))
                        .labelsHidden()
                        .frame(width: 48)
                        .multilineTextAlignment(.trailing)
                    Stepper("Prompts", value: promptsBinding, in: RobustnessPromptSupply.range)
                        .labelsHidden()
                    if let supply = panel.robustnessPromptSupplyCaption {
                        // Not `fixedSize`: this caption names the selected
                        // FILE, so a long path used to hold the whole row
                        // wider than the controls column. It compresses now
                        // and the tooltip carries the untruncated line.
                        Text(supply)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .help(supply)
                    }
                    Spacer(minLength: 8)
                    Text("Max tokens")
                        .fixedSize()
                    TextField(
                        "Max tokens",
                        value: Binding(
                            get: { panel.robustnessMaxTokens },
                            set: {
                                panel.robustnessMaxTokens = $0
                                if VariantRobustness.preset(id: panel.robustnessPresetID)?
                                    .maxTokens != $0
                                {
                                    panel.robustnessPresetID = VariantRobustness.customPresetID
                                }
                            }),
                        format: .number.grouping(.never))
                        .labelsHidden()
                        .frame(width: 72)
                        .multilineTextAlignment(.trailing)
                }
                .help(
                    "how many coherence prompts to read from the selected "
                        + "file, and the generation cap for each")
                // The engine takes a PREFIX when the request outruns the
                // file. That behaviour is unchanged; the row now says so
                // rather than truncating in silence.
                if let truncation = panel.robustnessPromptTruncationCaption {
                    Text(truncation)
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Toggle("AI coherence judge", isOn: $panel.robustnessUseJudge)
                    .help("asks a judge model to compare baseline and agent outputs for coherence and prompt-following")
                if panel.robustnessUseJudge {
                    JudgeModelPicker(
                        title: "Judge",
                        help:
                            "local, Claude, or OpenRouter judge used for "
                            + "pairwise coherence checks",
                        offers: panel.robustnessJudgeOffers,
                        selection: $panel.robustnessJudgeModel,
                        openRouterModel: $panel.robustnessOpenRouterModel,
                        openRouterProvider: $panel.robustnessOpenRouterProvider)
                }
                Button(panel.isRobustnessRunning ? "Checking…" : "Run Robustness Check") {
                    guard robustnessDisabledReason == nil else { return }
                    panel.runRobustnessCheck()
                }
                .disabled(robustnessDisabledReason != nil)
                .help(runRobustnessHelp)
                // A grayed button that says nothing is a bug, not a state:
                // every precondition that stops the run is written out here,
                // not only the judge one (the missing-target case used to be
                // silent).
                if let reason = robustnessDisabledReason, !panel.isRobustnessRunning {
                    Label(reason, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
                // Substrate transparency: battery/coherence items and scoring
                // are local recipe data + pure code on either route; only
                // generation moves. (localOnly is unreachable for this builder
                // today but stays handled — honest scoping if it regresses.)
                switch WorkspaceScoping.route(
                    for: .robustnessBattery, workspace: service.cluster.activeWorkspace)
                {
                case .serverJob:
                    Text(
                        "generates through \(service.cluster.substrateLabel) (inline "
                            + "agent spec, greedy); scoring runs locally")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                case .localOnly(let caption):
                    Text(caption)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                case .local:
                    EmptyView()
                }

                if panel.isRobustnessRunning {
                    HStack {
                        ProgressView()
                            .controlSize(.small)
                        Text("Generating baseline and agent responses…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        // A1: local robustness checks are cancellable between
                        // generations (server-routed checks are durable jobs
                        // with their own cancel affordance in Compute).
                        if case .local = WorkspaceScoping.route(
                            for: .robustnessBattery,
                            workspace: service.cluster.activeWorkspace)
                        {
                            Button("Stop", role: .destructive) {
                                panel.cancelRobustnessCheck()
                            }
                            .controlSize(.small)
                            .disabled(panel.robustnessCancelRequested)
                            .help(
                                "stops after the current generation — no report is "
                                    + "written (a partial battery is never scored); "
                                    + "reported as cancelled, never as an error")
                        }
                    }
                }
                if let report = panel.robustnessReport {
                    robustnessSummary(report)
                }
                if let path = panel.lastRobustnessDirectory {
                    Text(path)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .help("run directory holding this report — reports are immutable")
                }
        }
    }

    /// Why Run Robustness Check is disabled, in one plain sentence — nil
    /// means runnable. The missing-target case used to gray the button with
    /// nothing written anywhere (audit 2026-09-06, headline 24).
    private var robustnessDisabledReason: String? {
        if panel.isRobustnessRunning {
            return "a robustness check is already running"
        }
        if !panel.hasResolvableRobustnessTarget {
            return panel.robustnessTarget == nil
                ? "pick the agent this check runs against in the Agent picker "
                    + "above"
                : "the picked agent no longer resolves in this workspace — "
                    + "pick another in the Agent picker above"
        }
        return panel.robustnessJudgeDisabledReason
    }

    private var runRobustnessHelp: String {
        robustnessDisabledReason
            ?? ("generates baseline and agent answers for the battery and the "
                + "coherence prompts, then scores both — the report lands "
                + "under runs/ and in this agent's Library row")
    }

    /// The row-level variant of the same button: it POINTS the check at the
    /// row's agent first, so a missing target is not a reason it is off.
    private var rowRunRobustnessHelp: String {
        if panel.isRobustnessRunning { return "a robustness check is already running" }
        return panel.robustnessJudgeDisabledReason
            ?? ("run the configured robustness check on this agent; output "
                + "streams in the viewer")
    }

    /// The active server's saved variants: read-only listing plus selection
    /// into the chat's server-variant slot (upload happens from Steering).
    /// Titled by the ONE labeling rule so the workspace these agents live in
    /// is unmistakable (the historical "This workspace (<server>)" label hid
    /// that the server may serve a different tree than the app's workspace).
    @ViewBuilder
    private var serverVariantsSection: some View {
        Section(service.cluster.serverArtifactListTitle(kind: "Agents")) {
            if let root = service.cluster.activeServerServingRoot {
                Text("stored under \(root) (the server's serving root)")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            Text("server-resident copies — recipes auto-sync into the Agent "
                + "Library on connection/refresh, so everything here is (or "
                + "becomes) selectable in studies from the library above; "
                + "cluster storage is scratch and can be purged, the "
                + "workspace is the source of truth")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if service.cluster.remoteVariants.isEmpty {
                Text("No agents saved on this server yet — upload one from the Playground pane.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(service.cluster.remoteVariants) { variant in
                    HStack(alignment: .firstTextBaseline) {
                        VStack(alignment: .leading, spacing: 3) {
                            Text(variant.name)
                                .font(.headline)
                            Text(variant.baseModelID)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            Text(remoteVariantSummary(variant))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            Text(variant.path)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .textSelection(.enabled)
                                // Middle truncation hides exactly the part
                                // that distinguishes two stored copies.
                                .help(variant.path)
                        }
                        Spacer()
                        Button(
                            service.selectedRemoteVariantPath == variant.path
                                ? "Selected" : "Use in Chat"
                        ) {
                            service.selectedRemoteModelID = variant.baseModelID
                            // The same selection+seeding routine the Playground
                            // picker runs: fetches the spec and seeds the live
                            // controls. Setting the path alone leaves the
                            // controls showing STALE prior state that does not
                            // ride (the "not seeded" shadow, observed live).
                            Task {
                                await service.applyRemoteVariantToSteering(
                                    path: variant.path)
                            }
                        }
                        .disabled(service.selectedRemoteVariantPath == variant.path)
                        .help(
                            "chat through this server-side agent (base model + "
                                + "adapters + injections applied server-side)")
                    }
                    .padding(.vertical, 2)
                }
            }
            Button("Refresh Server Agents") {
                Task {
                    await service.cluster.refreshRemoteVariants()
                    // The refresh auto-syncs server recipes into the local
                    // library — rescan so they appear immediately.
                    panel.refresh()
                }
            }
            .help(
                "re-fetch the active server's agent list — any recipe not "
                    + "yet in the local library syncs in automatically (the "
                    + "workspace is the source of truth; cluster storage "
                    + "can be purged)")
        }
    }

    private func remoteVariantSummary(_ variant: RemoteVariantRecord) -> String {
        let injections = variant.injections ?? 0
        let adapters = variant.adapters ?? 0
        return "\(adapters) adapter\(adapters == 1 ? "" : "s")"
            + " · \(injections) injection\(injections == 1 ? "" : "s")"
    }

    private var editorFields: some View {
        Group {
            TextField("Name", text: $name)
                .help(
                    "what this agent is called in the library, in study "
                        + "conditions, and in reports — names must be unique "
                        + "within the workspace")

            // Strict, workspace-scoped model choice (same rule as the chat's
            // WorkspaceModelPicker): the options are the active workspace's
            // inventory; a definition whose base model isn't in it stays
            // rendered — "(not installed)" — but can never be picked anew.
            // Definitions themselves are global recipes; only the model-choice
            // affordance scopes.
            Picker("Base model", selection: $baseModelID) {
                if baseModelID.isEmpty {
                    Text("select model…").tag("")
                }
                ForEach(modelOptions, id: \.self) { model in
                    Text(model).tag(model)
                }
                if WorkspaceScoping.selectionOutsideInventory(baseModelID, inventory: modelOptions) {
                    Text(isServerWorkspace ? "\(baseModelID) (not installed)" : baseModelID)
                        .tag(baseModelID)
                        .selectionDisabled()
                }
            }
            .help(
                isServerWorkspace
                    ? "models installed on \(service.cluster.substrateLabel) — the "
                        + "definition's base model must exist on the workspace that runs it"
                    : "the local model tiers (plus bases of saved definitions)")
            .onChange(of: baseModelID) { _, newValue in
                // NO silent re-selection (the adapter-resurrection bug): a
                // user-initiated model change CLEARS the adapter and neutral
                // basis — the editor never picks a "compatible" replacement
                // the user didn't ask for. An artifact load preserves exactly
                // what loadEditor just set from the artifact (nil included).
                let origin = modelChangeClassifier.classify(observed: newValue)
                adapterID = AgentEditorSelection.dependentSelection(
                    after: origin, current: adapterID)
                neutralPCBasisID = AgentEditorSelection.dependentSelection(
                    after: origin, current: neutralPCBasisID)
                injections.removeAll { draft in
                    // Refs loaded from the definition are DATA: they survive
                    // a base-model change (and this onChange also fires when
                    // opening a definition programmatically sets the model) —
                    // unresolvable ones render as foreign, never auto-drop.
                    guard draft.originalConcept == nil else { return false }
                    guard let vectorID = draft.vectorArtifactID else { return false }
                    return !offeredInjectionVectorIDs.contains(vectorID)
                }
            }

            TextField("Base revision", text: $baseRevision)
                .help("optional pinned model revision, if known")

            Picker("Adapter", selection: $adapterID) {
                Text("None").tag(String?.none)
                ForEach(compatibleAdapters) { adapter in
                    Text(adapter.label).tag(String?.some(adapter.id))
                }
            }
            .help(
                "a trained LoRA adapter to load alongside the base model — "
                    + "only adapters trained on this base model are offered, "
                    + "and changing the base model clears the choice")

            Stepper("Layer band: \(bandWidth)", value: $bandWidth, in: 1 ... 11, step: 2)
                .help(
                    "how many neighbouring layers each injection covers: 1 "
                        + "injects at the chosen layer only, 3 also injects "
                        + "at the layer above and below, and so on. A wider "
                        + "band spreads the same α over more of the network")
            Toggle("Alpha in residual-norm units", isOn: $alphaInNormUnits)
                .help(
                    "on: each injection's α is a fraction of the layer's own "
                        + "residual-stream norm, so the same number means the "
                        + "same dose across concepts, layers, and models "
                        + "(0.1 is the usual starting dose). Off: α is the "
                        + "literal coefficient on the vector. Changing this "
                        + "changes what every α below MEANS — it does not "
                        + "convert them")
            if bandWidth > 1 {
                Text(layerBandCaption)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Picker("Neutral basis", selection: $neutralPCBasisID) {
                Text("None").tag(String?.none)
                ForEach(compatibleNeutralBases) { basis in
                    Text(basis.label).tag(String?.some(basis.id))
                }
            }
            .help(
                "principal components of neutral text to project out of the "
                    + "injection, so the vector carries less of whatever the "
                    + "model does anyway — built in Analysis, and offered "
                    + "only for this base model")

            Picker("Prompt mode", selection: $promptMode) {
                ForEach(ExperimentManifest.PromptMode.allCases, id: \.self) { mode in
                    Text(mode.label).tag(mode)
                }
            }
            .help(
                "how prompts reach the model — through the model's chat "
                    + "template, or as raw completion text. It changes the "
                    + "tokens the model actually sees, so a study must not "
                    + "mix modes across its arms")
            Toggle("Qwen thinking mode", isOn: $qwenThinkingEnabled)
                .disabled(!isQwenBaseModel)
                .help(
                    isQwenBaseModel
                        ? "asks Qwen for its <think> block before the answer "
                            + "— longer generations, and the thinking text is "
                            + "stripped from the recorded response"
                        : qwenThinkingDisabledHelp)
            if !isQwenBaseModel {
                Text(
                    "thinking mode is a Qwen chat-template feature; this base "
                        + "model does not offer it")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            AgentTemperatureRow(value: $temperature)
            TextField("System prompt", text: $systemPrompt, axis: .vertical)
                .lineLimit(1 ... 5)
                .help(
                    "the system message every generation starts from — part "
                        + "of the agent's definition, so it travels with it "
                        + "into studies and reports")

            injectionsEditor

            if let record = panel.selectedVariant {
                Text(record.url.path)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
    }

    /// Thinking mode is a Qwen chat-template feature; every other base model
    /// gets the toggle disabled, and now says why.
    private var isQwenBaseModel: Bool {
        baseModelID.lowercased().contains("qwen")
    }

    /// Built outside the view builder (the `ExperimentsPanelView` rule:
    /// string arithmetic in a body tips the type-checker over).
    private var qwenThinkingDisabledHelp: String {
        let model = baseModelID.isEmpty ? "no base model is selected" : "'\(baseModelID)'"
        return "only Qwen base models expose a thinking mode — \(model) does not"
    }

    private var layerBandCaption: String {
        "layer band \(bandWidth): every injection below is applied at its "
            + "layer and the \(bandWidth - 1) neighbouring layers around it"
    }

    private var injectionsEditor: some View {
        DisclosureGroup("Injections") {
            if injections.isEmpty {
                Text("No vectors in this agent.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            ForEach($injections) { $draft in
                VStack(alignment: .leading, spacing: 6) {
                    Picker("Vector", selection: $draft.vectorArtifactID) {
                        Text("None").tag(VectorArtifact.ID?.none)
                        // A ref the active workspace can't offer stays
                        // RENDERED (SwiftUI must not blank the binding) but
                        // is never pickable anew; saving keeps it verbatim.
                        if let id = draft.vectorArtifactID,
                            !offeredInjectionVectorIDs.contains(id)
                        {
                            Text(currentSelectionLabel(for: draft, id: id))
                                .tag(VectorArtifact.ID?.some(id))
                                .selectionDisabled()
                        }
                        // Recipe-grouped, like the playground's picker: one
                        // row per DISTINCT recipe, concept sections A→Z.
                        // Older re-derivations of a recipe are hidden (they
                        // derive identically); distinct recipes all stay
                        // visible, marked superseded below the newest.
                        if isServerWorkspace {
                            // Tag with the WORKSPACE-RELATIVE id where the
                            // server advertises one: that is the reference the
                            // saved variant stores (the Mac workspace is the
                            // source of truth; save fetches the bytes local).
                            ForEach(serverVectorOffers.sections) { section in
                                Section(section.concept) {
                                    ForEach(section.offers) { offer in
                                        Text(offer.displayLabel)
                                            .tag(VectorArtifact.ID?.some(offer.id))
                                    }
                                }
                            }
                        } else {
                            ForEach(localVectorOffers.sections) { section in
                                Section(section.concept) {
                                    ForEach(section.offers) { offer in
                                        Text(offer.displayLabel)
                                            .tag(VectorArtifact.ID?.some(offer.id))
                                    }
                                }
                            }
                        }
                    }
                    .help(
                        isServerWorkspace
                            ? "vectors on \(service.cluster.substrateLabel) for this "
                                + "base model, one row per distinct recipe (method + "
                                + "stimulus hash) — picking one stores a "
                                + "workspace-relative reference and Save fetches the "
                                + "artifact into the local workspace (the workspace is "
                                + "the source of truth; the server only caches)"
                            : "local vector artifacts for this base model, one row per "
                                + "distinct recipe (method + stimulus hash)")
                    HStack(alignment: .top) {
                        InjectionModeControls(
                            mode: $draft.mode,
                            layer: $draft.layer,
                            strength: $draft.alpha,
                            // The agent-level units toggle decides what this
                            // row's α MEANS — the row now says so, defaults
                            // to that denomination, and warns on a raw-scale
                            // number typed into a norm-unit field.
                            alphaInNormUnits: alphaInNormUnits,
                            layerCount: layerCount(for: draft),
                            conceptLabel: conceptLabel(for: draft))
                        Spacer()
                        Button {
                            injections.removeAll { $0.id == draft.id }
                        } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.borderless)
                        .accessibilityLabel("Remove injection")
                        .help(
                            "remove this injection from the definition — "
                                + "immediate, with no undo; Reset From "
                                + "Playground discards every unsaved edit")
                    }
                }
                .padding(.vertical, 4)
            }
            Button("Add Vector") {
                // Default to the FIRST OFFERED row, not the first record in
                // an arbitrary catalog order: that is the newest derivation
                // of the newest recipe of the first concept. α starts at the
                // default OF THIS AGENT'S DENOMINATION — the flat 1 this used
                // to seed is a whole residual stream in norm units, exactly
                // what `AlphaMagnitudeWarning` calls a typo.
                let seedAlpha = InjectionModeCopy.defaultSteeringAlpha(
                    normUnits: alphaInNormUnits)
                if isServerWorkspace {
                    let record = serverVectorOffers.offers.first?.item
                    injections.append(
                        InjectionDraft(
                            vectorArtifactID: record?.canonicalStoredID,
                            layer: record.map { $0.layerCount / 2 } ?? 0,
                            alpha: seedAlpha))
                } else {
                    let vector = localVectorOffers.offers.first?.item
                    injections.append(
                        InjectionDraft(
                            vectorArtifactID: vector?.id,
                            layer: vector?.fixedSteeringLayer
                                ?? vector.map { $0.sidecar.layerCount / 2 } ?? 0,
                            alpha: seedAlpha))
                }
            }
            .disabled(
                isServerWorkspace
                    ? serverVectorOffers.isEmpty
                    : localVectorOffers.isEmpty)
            .help(addVectorHelp)
            // Self-reporting availability: when the picker looks mysteriously
            // thin, this line names the filter inputs instead of leaving the
            // user (or the maintainer) to guess which rule starved it.
            Text(injectionAvailabilityCaption)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .help(
            "the concept vectors this agent injects, and at what layer and "
                + "strength — an agent with none is a legal baseline")
    }

    private var addVectorHelp: String {
        let dose = alphaInNormUnits
            ? "norm-unit starting dose (0.1)" : "raw-unit default (2)"
        return "add an injection row seeded with the newest recipe of the "
            + "first concept, that vector's middle layer, and the " + dose
    }

    private var injectionAvailabilityCaption: String {
        let base = baseModelID.isEmpty ? "any base model" : baseModelID
        let line: String
        let hidden: Int
        if isServerWorkspace {
            let offers = serverVectorOffers
            hidden = offers.hiddenDerivationCount
            line = "\(offers.offers.count) recipe\(offers.offers.count == 1 ? "" : "s") "
                + "from \(serverInjectionVectorOptions.count) of "
                + "\(service.catalog.remoteVectors.count) server vectors offered "
                + "for \(base) (\(service.cluster.substrateLabel))"
        } else {
            let offers = localVectorOffers
            hidden = offers.hiddenDerivationCount
            line = "\(offers.offers.count) recipe\(offers.offers.count == 1 ? "" : "s") "
                + "from \(compatibleVectors.count) of \(service.vectors.count) local "
                + "vectors offered for \(base) (Local workspace)"
        }
        guard let note = VectorRecipeGrouping.hiddenDerivationsCaption(hidden) else {
            return line
        }
        return "\(line) — \(note)"
    }

    /// Label for a draft's current selection when it is not among the
    /// workspace's offered options: known-but-filtered refs show their
    /// catalog concept; truly foreign refs are labeled as such.
    private func currentSelectionLabel(for draft: InjectionDraft, id: String) -> String {
        if let concept = workspaceCatalogConcept(for: id) {
            return "\(concept) · \(id)"
        }
        return VariantInjectionEditing.unresolvedLabel(
            concept: draft.originalConcept, ref: id)
    }

    /// The concept this row's vector names, so the mode explanation reads
    /// "removes fear" rather than "removes this concept".
    private func conceptLabel(for draft: InjectionDraft) -> String? {
        guard let id = draft.vectorArtifactID else { return draft.originalConcept }
        return workspaceCatalogConcept(for: id) ?? draft.originalConcept
    }

    /// The model's depth for this row's vector, when the active workspace can
    /// see it — bounds the layer stepper instead of letting a typo name a
    /// layer that does not exist.
    private func layerCount(for draft: InjectionDraft) -> Int? {
        guard let id = draft.vectorArtifactID else { return nil }
        if isServerWorkspace {
            return service.catalog.remoteVectors
                .first { $0.matches(reference: id) }?.layerCount
        }
        return service.vectors.first { $0.id == id }?.sidecar.layerCount
    }

    /// Concept for a vector id in the ACTIVE workspace's catalog (server
    /// listing or local sidecars), nil when the workspace cannot resolve it.
    private func workspaceCatalogConcept(for id: String) -> String? {
        if isServerWorkspace {
            return service.catalog.remoteVectors
                .first { $0.matches(reference: id) }?.concept
        }
        return service.vectors.first { $0.id == id }?.sidecar.concept
    }

    private var isServerWorkspace: Bool {
        service.cluster.computeTarget == .server
    }

    /// Robustness target sources through the ONE tested rule: server-stored
    /// agents appear only under a server target (same source as the agents
    /// list); local definitions are always offered.
    private var robustnessLocalOptions: [ModelVariantRecord] {
        WorkspaceScoping.robustnessTargetSources(
            workspaceIsServer: isServerWorkspace,
            localRecords: panel.variants,
            serverAgents: service.cluster.remoteVariants
        ).local
    }

    private var robustnessServerAgentOptions: [RemoteVariantRecord] {
        WorkspaceScoping.robustnessTargetSources(
            workspaceIsServer: isServerWorkspace,
            localRecords: panel.variants,
            serverAgents: service.cluster.remoteVariants
        ).server
    }

    private var robustnessTargetBinding: Binding<FineTuningPanel.RobustnessTarget?> {
        let panel = service.fineTuning
        return Binding(
            get: { panel.robustnessTarget },
            set: { panel.robustnessTarget = $0 })
    }

    private func robustnessOptionRow(
        name: String, target: FineTuningPanel.RobustnessTarget
    ) -> some View {
        Text(name).tag(Optional(target))
    }

    /// The Robustness Check's target picker, extracted (type-checker relief)
    /// and sourced through the ONE scoping rule: server target → the SERVER's
    /// stored agents (same list as the agents section) plus local definitions
    /// runnable there as inline specs; local target → local records only.
    private var robustnessTargetPicker: some View {
        Picker("Agent", selection: robustnessTargetBinding) {
            Text("choose an agent…").tag(nil as FineTuningPanel.RobustnessTarget?)
            if isServerWorkspace {
                Section(service.cluster.serverArtifactListTitle(kind: "Agents")) {
                    ForEach(robustnessServerAgentOptions) { variant in
                        robustnessOptionRow(
                            name: variant.name, target: .server(path: variant.path))
                    }
                }
                Section("Local definitions (run as inline spec)") {
                    ForEach(robustnessLocalOptions) { variant in
                        robustnessOptionRow(
                            name: variant.artifact.name, target: .local(variant.id))
                    }
                }
            } else {
                ForEach(robustnessLocalOptions) { variant in
                    robustnessOptionRow(
                        name: variant.artifact.name, target: .local(variant.id))
                }
            }
        }
        .help(
            isServerWorkspace
                ? "the agent this check runs against — the active server's "
                    + "stored agents (same list as above), or a local "
                    + "definition sent as an inline spec; picked here, "
                    + "independent of the browser above"
                : "the saved agent this check runs against — picked here, "
                    + "independent of the browser above")
    }

    /// Applicability of the SELECTED local definition in the active server
    /// workspace (nil in Local, or with no selection). Pure rule:
    /// `WorkspaceScoping.serverDefinitionApplicability` via
    /// `ChatService.serverApplicability`.
    private var selectedDefinitionApplicability: WorkspaceScoping.DefinitionApplicability? {
        guard isServerWorkspace, let record = panel.selectedVariant else { return nil }
        return service.serverApplicability(of: record.artifact)
    }

    /// Base-model options for the ACTIVE workspace. Server: the active
    /// server's installed models — the same inventory source the chat's
    /// WorkspaceModelPicker reads (`ChatService.workspaceModelOptions` →
    /// `SubstrateCatalog.installedModels` + models loaded on the server) —
    /// never local MLX repos. Local: the local tiers plus loaded/selected and
    /// the bases of saved definitions (unchanged behavior); an out-of-
    /// inventory selection is handled by the "(not installed)" row above,
    /// not by injecting it as a pickable option.
    private var modelOptions: [String] {
        if isServerWorkspace {
            return service.workspaceModelOptions
        }
        var seen = Set<String>()
        var options: [String] = []
        func append(_ value: String?) {
            guard let value else { return }
            let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, !seen.contains(trimmed) else { return }
            seen.insert(trimmed)
            options.append(trimmed)
        }
        append(service.loadedModelID)
        append(service.selectedModelID)
        for model in ChatService.availableModels.map(\.id) { append(model) }
        for variant in panel.variants { append(variant.artifact.baseModelID) }
        return options
    }

    private var compatibleAdapters: [FineTuneArtifactRecord] {
        // Application picker: explicitly foreign-stamped adapters (hf-peft
        // weights) are excluded — they cannot load in this engine; unstamped
        // legacy records stay offered.
        panel.adapters.filter {
            $0.artifact.baseModelID == baseModelID
                && !AdapterSubstrateGate.isExplicitlyForeign(
                    substrate: $0.artifact.substrate,
                    adapterFormat: $0.artifact.adapterFormat)
        }
    }

    private var compatibleVectors: [VectorArtifact] {
        // Same substrate rule as the chat picker: a local variant injects
        // in-process, so python-stamped sidecars are never offered.
        service.vectors.filter {
            $0.sidecar.modelID == baseModelID
                && WorkspaceScoping.offerableForLocalSteering(
                    substrate: $0.sidecar.substrate)
        }
    }

    /// Server-workspace injection options: the ACTIVE server's catalog
    /// records, same strict rule as chat steering ("swift-mlx"-stamped
    /// excluded), filtered to this definition's base model.
    ///
    /// This stays the FULL filtered list. The picker RENDERS the
    /// recipe-grouped subset (`serverVectorOffers`), but save-time
    /// localization must still resolve a ref that names a hidden older
    /// derivation — narrowing this would silently store an absolute
    /// `/scratch/…` id again (field incident 2026-08-05).
    private var serverInjectionVectorOptions: [RemoteVectorRecord] {
        WorkspaceScoping.serverInjectionVectorOptions(
            service.catalog.remoteVectors,
            baseModelID: baseModelID,
            substrate: \.substrate,
            modelID: \.modelID)
    }

    /// Recipe-grouped LOCAL picker model: one row per distinct
    /// (model, method, stimulus hash), newest derivation only, concepts A→Z.
    /// The 198-artifact / ~30-concept catalog is mostly deterministic
    /// re-derivations of the same recipes (field report 2026-08-06).
    private var localVectorOffers: VectorRecipeGrouping.Presentation<VectorArtifact> {
        VectorRecipeGrouping.present(
            compatibleVectors,
            concept: { $0.sidecar.concept },
            modelID: { $0.sidecar.modelID },
            recipeMethod: { $0.sidecar.recipeMethod },
            extractionMethod: { $0.sidecar.extractionMethod },
            stimulusSetHash: { $0.sidecar.stimulusSetHash },
            extractionDate: { $0.sidecar.extractionDate },
            name: { $0.name },
            id: { $0.id })
    }

    /// Server mirror. Identity comes from the catalog's own resolved fields
    /// (`resolvedMethod`/`resolvedExtractionDate` semantics), and rows are
    /// tagged with `canonicalStoredID` — exactly the id the picker stored
    /// before this grouping existed, so no saved variant changes shape.
    private var serverVectorOffers: VectorRecipeGrouping.Presentation<RemoteVectorRecord> {
        VectorRecipeGrouping.present(
            serverInjectionVectorOptions,
            concept: { $0.concept },
            modelID: { $0.modelID },
            recipeMethod: { $0.recipeMethod },
            extractionMethod: { $0.extractionMethod ?? $0.method },
            stimulusSetHash: { $0.stimulusSetHash },
            extractionDate: { $0.resolvedExtractionDate },
            name: { $0.name },
            id: { $0.canonicalStoredID })
    }

    /// The ids the active workspace's picker offers — anything selected but
    /// not in here renders as a non-pickable current-selection row. Computed
    /// from the OFFERED rows, so a ref pointing at a hidden older derivation
    /// falls through to that row instead of silently blanking the binding.
    private var offeredInjectionVectorIDs: Set<String> {
        // Server records answer to BOTH reference forms: the relative id new
        // saves store, and the absolute catalog id in pre-2026-08-05
        // variants — neither may auto-drop on a model change.
        isServerWorkspace
            ? Set(serverVectorOffers.offers.flatMap {
                [$0.item.id, $0.item.workspaceRelativeID].compactMap(\.self)
            })
            : localVectorOffers.offeredIDs
    }

    private var compatibleNeutralBases: [NeutralPCArtifactRecord] {
        service.neutralPCBases.filter { $0.basis.modelID == baseModelID }
    }

    private func loadEditorForSelection() {
        if let record = panel.selectedVariant {
            guard editorLoadedID != record.id else { return }
            loadEditor(from: record.artifact)
            editorLoadedID = record.id
        } else {
            guard editorLoadedID != "new" else { return }
            loadDraftFromCurrentSteering()
            editorLoadedID = "new"
        }
    }

    private func loadDraftFromCurrentSteering() {
        let artifact = currentSteeringArtifact(
            name: panel.variantName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "agent-1"
                : panel.variantName)
        loadEditor(from: artifact)
        editorLoadedID = nil
    }

    private func loadEditor(from artifact: ModelVariantArtifact) {
        name = artifact.name
        interventionPolicies = artifact.interventionPolicies
        // Programmatic write: arm the classifier so the onChange this fires
        // preserves the artifact's own adapter/basis instead of clearing or
        // re-picking them.
        modelChangeClassifier.expectProgrammaticChange(
            to: artifact.baseModelID, from: baseModelID)
        baseModelID = artifact.baseModelID
        baseRevision = artifact.baseRevision ?? ""
        adapterID = adapterID(for: artifact.adapters.first)
        injections = VariantInjectionEditing.drafts(from: artifact.injections).map {
            InjectionDraft(
                vectorArtifactID: $0.vectorArtifactID,
                originalConcept: $0.originalConcept,
                layer: $0.layer,
                alpha: $0.alpha)
        }
        bandWidth = artifact.bandWidth
        alphaInNormUnits = artifact.alphaInNormUnits
        neutralPCBasisID = neutralBasisID(for: artifact)
        promptMode = ExperimentManifest.PromptMode(rawValue: artifact.promptMode) ?? .chatAssistant
        qwenThinkingEnabled = artifact.qwenThinkingEnabled
        temperature = artifact.temperature
        systemPrompt = artifact.systemPrompt ?? ""
    }

    private func saveVariant(applyAfterSave: Bool = false) {
        // The gate the button reads, enforced in the action too: every path
        // into a save (including "Use in Playground" with nothing selected)
        // goes through here, and a name collision would otherwise mint a
        // second library row with the same name.
        if let reason = saveDisabledReason {
            panel.setStatus(reason)
            return
        }
        guard isServerWorkspace else {
            performSave(applyAfterSave: applyAfterSave)
            return
        }
        // Server workspace: before writing the definition, LOCALIZE every
        // picked server vector — fetch the sidecar + safetensors pair into
        // the local workspace and store the workspace-relative reference.
        // The authoring client's workspace is the source of truth; a
        // `/scratch/…` absolute ref exists on one substrate only and dies at
        // bundle packaging
        // ("pinned input missing", field incident 2026-08-05). Refs the
        // picker didn't offer (foreign definitions) round-trip verbatim,
        // and an older server without relative ids keeps the historical
        // verbatim behavior.
        Task {
            await localizeServerInjectionsThenSave(applyAfterSave: applyAfterSave)
        }
    }

    private func localizeServerInjectionsThenSave(applyAfterSave: Bool) async {
        let options = serverInjectionVectorOptions
        // What could not be bound to the catalog row, collected across the
        // whole save. Localization NEVER blocks on a missing digest (a
        // pre-field server would become an unexplainable save failure), so
        // the only honest alternative is to say so where the researcher is
        // looking (2026-08-06 review round 2, P2 — before this, the warning
        // was a `print` into a console nobody reads, and it only fired when
        // BOTH digests were absent).
        var unverified: [String] = []
        for draft in injections {
            guard let reference = draft.vectorArtifactID,
                let record = options.first(where: { $0.matches(reference: reference) }),
                let relative = record.workspaceRelativeID
            else { continue }
            guard let client = service.cluster.client else {
                panel.setStatus(
                    "could not save agent: no server connection to fetch "
                        + "vector '\(record.concept)' into the workspace")
                return
            }
            do {
                panel.setStatus("fetching \(record.concept) into the workspace…")
                let localized = try await RemoteVectorLocalization.localize(
                    serverID: record.id, workspaceRelativeID: relative,
                    // The catalog row's hashes bind it to the bytes: the
                    // fetch verifies what it lands, and a local pair that
                    // disagrees refuses rather than being overwritten.
                    sidecarSha256: record.sidecarSha256,
                    tensorSha256: record.tensorSha256
                ) { path, directory in
                    try await client.downloadArtifact(path: path, to: directory)
                }
                if !localized.verification.isFullyVerified {
                    unverified.append(record.concept)
                }
                if let index = injections.firstIndex(where: { $0.id == draft.id }) {
                    injections[index].vectorArtifactID = localized.relativeID
                }
            } catch {
                panel.setStatus(
                    "could not save agent: fetching vector '\(record.concept)' "
                        + "failed — \(error.localizedDescription)")
                return
            }
        }
        let saved = performSave(applyAfterSave: applyAfterSave)
        if saved, !unverified.isEmpty {
            // The wording lives in ExperimentKit (unit-tested): it is the
            // only place the researcher learns that a saved agent's bytes are
            // unvouched-for, which makes it a contract, not view text.
            panel.setWarningStatus(
                RemoteVectorLocalization.saveDisclosure(concepts: unverified))
        }
    }

    /// Returns whether the definition was written — the localization path
    /// appends its disclosure only on success, so a save FAILURE is never
    /// overwritten by a line beginning "saved".
    @discardableResult
    private func performSave(applyAfterSave: Bool = false) -> Bool {
        var artifact = artifactFromEditor()
        let selected = panel.selectedVariant
        do {
            let record: ModelVariantRecord
            if let selected {
                artifact.createdAt = selected.artifact.createdAt
                record = try ModelVariantStore.update(artifact, at: selected.url)
            } else {
                record = try ModelVariantStore.save(artifact)
            }
            panel.refresh()
            panel.selectedVariantID = record.id
            editorLoadedID = record.id
            panel.setStatus(selected == nil
                ? "saved agent \(record.artifact.name)"
                : "updated agent \(record.artifact.name)")
            if applyAfterSave {
                service.applyModelVariantToSteering(record)
            }
            return true
        } catch {
            panel.setStatus("could not save agent: \(error.localizedDescription)")
            return false
        }
    }

    private func artifactFromEditor() -> ModelVariantArtifact {
        let adapterRefs = adapterID
            .flatMap { id in panel.adapters.first { $0.id == id } }
            .map { record in
                [
                    ModelVariantArtifact.AdapterRef(
                        name: record.artifact.name,
                        artifactPath: FineTuneStore.relativePath(for: record.url),
                        adapterDirectory: record.artifact.adapterDirectory,
                        adapterHash: record.artifact.adapterHash,
                        configHash: record.artifact.configHash)
                ]
            } ?? []
        // Round-trip-safe emission: refs the active workspace resolves get
        // the catalog's concept; foreign/unknown refs are re-emitted verbatim
        // with their original concept — never silently dropped.
        let injectionRefs = VariantInjectionEditing.injectionRefs(
            drafts: injections.map(\.logic),
            resolveConcept: workspaceCatalogConcept(for:))
        let basis = neutralPCBasisID.flatMap { id in
            service.neutralPCBases.first { $0.id == id }
        }
        return ModelVariantArtifact(
            name: name.trimmingCharacters(in: .whitespacesAndNewlines),
            baseModelID: baseModelID,
            baseRevision: nilIfEmpty(baseRevision),
            adapters: adapterRefs,
            injections: injectionRefs,
            bandWidth: bandWidth,
            alphaInNormUnits: alphaInNormUnits,
            neutralPCBasisPath: basis.map(NeutralPCStore.relativePath),
            neutralPCBasisLabel: basis?.label,
            promptMode: promptMode.rawValue,
            qwenThinkingEnabled: qwenThinkingEnabled,
            temperature: temperature,
            systemPrompt: systemPrompt,
            interventionPolicies: interventionPolicies)
    }

    private func currentSteeringArtifact(name: String) -> ModelVariantArtifact {
        // Server workspace: the draft seeds from the SAME composed control
        // state a server send/save uses (server base model + server refs) —
        // never from local slots, which hold ids this workspace can't run.
        if isServerWorkspace {
            let state = service.serverControlResolution().state
            return InlineVariantComposer.captureArtifact(
                state: state,
                name: name,
                loadedServerModelID: service.serverDefaultLoadedModelID)
                ?? InlineVariantComposer.compose(state, name: name)
        }
        let adapterRefs: [ModelVariantArtifact.AdapterRef] = (
            service.adaptersEnabled ? service.selectedRuntimeAdapter : nil
        ).map { record in
            [
                .init(
                    name: record.artifact.name,
                    artifactPath: FineTuneStore.relativePath(for: record.url),
                    adapterDirectory: record.artifact.adapterDirectory,
                    adapterHash: record.artifact.adapterHash,
                        configHash: record.artifact.configHash)
            ]
        } ?? []
        let injectionRefs = service.slots.compactMap { slot -> ModelVariantArtifact.InjectionRef? in
            guard slot.enabled,
                let vector = service.artifact(for: slot)
            else { return nil }
            return .init(
                concept: vector.sidecar.concept,
                vectorArtifactID: vector.id,
                layer: Int(slot.layer),
                alpha: slot.alpha)
        }
        let basis = service.removeNeutralDirectionsAtSteering ? service.selectedNeutralPCBasis : nil
        return ModelVariantArtifact(
            name: name,
            baseModelID: service.loadedModelID ?? service.selectedModelID,
            adapters: adapterRefs,
            injections: injectionRefs,
            bandWidth: service.layerBandWidth,
            alphaInNormUnits: service.alphaInNormUnits,
            neutralPCBasisPath: basis.map(NeutralPCStore.relativePath),
            neutralPCBasisLabel: basis?.label,
            promptMode: service.promptMode.rawValue,
            qwenThinkingEnabled: service.qwenThinkingEnabled,
            temperature: service.temperature,
            systemPrompt: service.systemPrompt)
    }

    private func adapterID(for ref: ModelVariantArtifact.AdapterRef?) -> String? {
        guard let ref else { return nil }
        return panel.adapters.first {
            FineTuneStore.relativePath(for: $0.url) == ref.artifactPath
                || $0.artifact.adapterDirectory == ref.adapterDirectory
        }?.id
    }

    private func neutralBasisID(for artifact: ModelVariantArtifact) -> String? {
        guard let path = artifact.neutralPCBasisPath else { return nil }
        return service.neutralPCBases.first {
            NeutralPCStore.relativePath(for: $0) == path
                || $0.label == artifact.neutralPCBasisLabel
        }?.id
    }

    // MARK: Agent Library (Pass 3)

    /// Availability the pure chip rules judge against — the same sources the
    /// existing pickers use (local tiers + scanned artifacts; the ACTIVE
    /// server's catalog and the existing applicability rule in a server
    /// workspace).
    private var agentAvailability: AgentLibrary.Availability {
        var localModels = Set(ChatService.availableModels.map(\.id))
        if let loaded = service.loadedModelID { localModels.insert(loaded) }
        return AgentLibrary.Availability(
            localModelIDs: localModels,
            localVectorIDs: Set(service.vectors.map(\.id)),
            localAdapterKeys: Set(
                panel.adapters.flatMap {
                    [FineTuneStore.relativePath(for: $0.url), $0.artifact.adapterDirectory]
                }),
            serverApplicability: nil,
            serverVectorIDs: isServerWorkspace
                ? Set(service.catalog.remoteVectors.map(\.id)) : [])
    }

    /// Recorded-vs-current adapter hashes for the drift chip: the registered
    /// adapter artifacts' CURRENT hashes, keyed by adapter directory.
    private var currentAdapterHashes: [String: String] {
        var hashes: [String: String] = [:]
        for record in panel.adapters {
            if let hash = record.artifact.adapterHash {
                hashes[record.artifact.adapterDirectory] = hash
            }
        }
        return hashes
    }

    /// Everything a batch of rows needs to judge itself, built ONCE per body
    /// evaluation instead of once per row.
    ///
    /// `agentAvailability` and `currentAdapterHashes` are computed
    /// properties: reading them inside a per-row helper rebuilt a set of
    /// every local model, vector and adapter for EVERY agent — twice (once to
    /// filter, once for the chips). That is O(agents × catalog) of pure CPU
    /// on the main thread per render, on top of the IO the tab used to do.
    private struct LibraryContext {
        var availability: AgentLibrary.Availability
        var adapterHashes: [String: String]
        var isServerWorkspace: Bool
    }

    private var libraryContext: LibraryContext {
        LibraryContext(
            availability: agentAvailability,
            adapterHashes: currentAdapterHashes,
            isServerWorkspace: isServerWorkspace)
    }

    private func chips(
        for entry: AgentLibraryIndex.Entry, context: LibraryContext
    ) -> [AgentLibrary.Chip] {
        var availability = context.availability
        if context.isServerWorkspace {
            availability.serverApplicability = serverApplicability(of: entry)
        }
        return AgentLibrary.chips(
            for: entry.components,
            availability: availability,
            currentAdapterHashes: context.adapterHashes)
    }

    private var agentFilter: AgentLibrary.Filter {
        AgentLibrary.Filter(
            baseModelID: filterBaseModel,
            sweepPromotedOnly: filterSweepOnly,
            hasAdapter: filterHasAdapter,
            runnableHere: filterRunnableHere)
    }

    /// The server rule wants the full artifact; a row holds only its summary,
    /// so resolve the record on demand. Only reached in a server workspace,
    /// and only for rows that are actually being judged.
    private func serverApplicability(of entry: AgentLibraryIndex.Entry) -> Bool {
        guard let record = panel.variants.first(where: { $0.id == entry.id })
        else { return false }
        return service.serverApplicability(of: record.artifact).isApplicable
    }

    /// "Runnable here" = the ACTIVE workspace can run it (local resolution in
    /// Local; the existing server applicability rule in a server workspace).
    private func runnableHere(
        _ entry: AgentLibraryIndex.Entry, context: LibraryContext
    ) -> Bool {
        if context.isServerWorkspace { return serverApplicability(of: entry) }
        return AgentLibrary.isRunnableLocally(
            entry.components, availability: context.availability)
    }

    private func baselineRunnableHere(
        _ row: AgentLibrary.BaselineRow, context: LibraryContext
    ) -> Bool {
        if context.isServerWorkspace {
            return service.workspaceModelOptions.contains(row.baseModelID)
        }
        return context.availability.localModelIDs.contains(row.baseModelID)
    }

    /// Filtered rows. `runnableHere` is passed as a closure, so the
    /// availability judgement (and, on a server, the applicability rule
    /// against the remote catalog) runs only when the "Runnable here" filter
    /// is actually on — it used to be evaluated for every row every time.
    private func filteredAgents(
        context: LibraryContext
    ) -> [AgentLibraryIndex.Entry] {
        panel.agentIndex.filter { entry in
            AgentLibrary.matches(
                entry.components, filter: agentFilter,
                runnableHere: { runnableHere(entry, context: context) })
        }
    }

    private func filteredBaselines(
        context: LibraryContext
    ) -> [AgentLibrary.BaselineRow] {
        AgentLibrary.baselineRows(
            forBaseModelIDs: panel.agentIndex.map(\.baseModelID)
        ).filter { row in
            AgentLibrary.matches(
                baseline: row, filter: agentFilter,
                runnableHere: { baselineRunnableHere(row, context: context) })
        }
    }

    private var libraryBaseModelOptions: [String] {
        Set(panel.agentIndex.map(\.baseModelID)).sorted()
    }

    /// The roster scrolls inside a CONSTANT-height box rather than laying out
    /// every agent inline — the same shape as the Analysis section's vector
    /// list (`GeometryPanelView.vectorListHeight`).
    ///
    /// macOS 27 beta hazard (project memory "split-view min-size crashes"):
    /// this is a fixed MAXIMUM on a compressible `ScrollView`, and no
    /// minimum here varies with the row count. Nothing about the surrounding
    /// split-view columns changes.
    private static let agentListHeight: CGFloat = 360

    @ViewBuilder
    private var agentLibrarySection: some View {
        Section("Agent Library") {
            if panel.agentIndex.isEmpty {
                if panel.isScanningAgents || !panel.hasScannedAgents {
                    // The tab is already on screen; the library is still
                    // being read. Never the other way round.
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("Reading the agent library…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else {
                    Text(
                        "No agents yet. Create one in New Agent — by hand, or by "
                            + "optimizing a concept vector (sweep → criterion → "
                            + "Create Agent).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    HStack(spacing: 8) {
                        Button("New Agent") { region = .create }
                            .help("open the manual definition editor in New Agent")
                        Button("Optimize a Vector") {
                            createMode = .optimize
                            region = .create
                        }
                        .help(
                            "open the optimization composer — declare a "
                                + "layer×alpha sweep and a criterion, then "
                                + "create the agent from the winning cell")
                        Button("Train Adapter") { navigate(.data) }
                            .help("open Data → Adapter Training to fine-tune a LoRA first")
                    }
                    .controlSize(.small)
                }
            } else {
                Text(
                    "saved agent definitions in this workspace — pick one to "
                        + "chat with it, check it, or attach it to a draft "
                        + "study. Definitions are recipes: they travel with "
                        + "the workspace and are pinned by hash when a study "
                        + "uses one.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                libraryFilters
                agentRoster
                selectedAgentActions
            }
        }
    }

    private var agentRoster: some View {
        // The context is built here, once, and captured by the lazily
        // materialized rows below.
        let context = libraryContext
        let baselines = filteredBaselines(context: context)
        let agents = filteredAgents(context: context)
        return VStack(alignment: .leading, spacing: 4) {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(baselines) { row in
                        baselineRowView(row)
                    }
                    ForEach(agents) { entry in
                        Button {
                            panel.selectedVariantID = entry.id
                            loadEditorForSelection()
                        } label: {
                            agentRow(entry, context: context)
                        }
                        .buttonStyle(.plain)
                        .accessibilityAddTraits(
                            panel.selectedVariantID == entry.id ? [.isSelected] : [])
                        .help(Self.rosterRowHelp(entry.name))
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.vertical, 2)
            }
            .frame(maxHeight: Self.agentListHeight)

            if agents.isEmpty, baselines.isEmpty {
                Text("No agents match the current filters.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 6) {
                if panel.isScanningAgents {
                    ProgressView().controlSize(.small)
                    Text("refreshing…")
                } else {
                    Text(
                        "\(agents.count) agent\(agents.count == 1 ? "" : "s")"
                            + (agents.count == panel.agentIndex.count
                                ? "" : " of \(panel.agentIndex.count)"))
                }
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
    }

    private static func rosterRowHelp(_ name: String) -> String {
        "select '\(name)' — its actions appear below, and Edit opens its "
            + "definition in New Agent"
    }

    private var runnableHereFilterHelp: String {
        let where_ = isServerWorkspace ? service.cluster.substrateLabel : "this Mac"
        return "show only agents whose base model and every vector/adapter "
            + "reference resolve on \(where_) — the filter follows the active "
            + "substrate"
    }

    private var libraryFilters: some View {
        VStack(alignment: .leading, spacing: 6) {
            Picker("Base model", selection: $filterBaseModel) {
                Text("All").tag(String?.none)
                ForEach(libraryBaseModelOptions, id: \.self) { model in
                    Text(model).tag(String?.some(model))
                }
            }
            .help("show only agents built on one base model — a display filter, nothing is deleted")
            HStack(spacing: 12) {
                Toggle("Sweep-promoted only", isOn: $filterSweepOnly)
                    .help(
                        "show only agents created from a declared "
                            + "optimization's winning cell — the ones "
                            + "carrying a birth certificate, rather than "
                            + "hand-composed definitions")
                Toggle("Has adapter", isOn: $filterHasAdapter)
                    .help("show only agents that load a trained LoRA adapter")
                Toggle("Runnable here", isOn: $filterRunnableHere)
                    .help(runnableHereFilterHelp)
            }
            .toggleStyle(.checkbox)
            .font(.caption)
        }
        .padding(.vertical, 2)
    }

    /// One roster row, rendered entirely from its summary
    /// (`AgentLibraryIndex.Entry`) — no artifact retained, no date or
    /// component string rebuilt here, and no disk touched.
    private func agentRow(
        _ entry: AgentLibraryIndex.Entry, context: LibraryContext
    ) -> some View {
        let selected = panel.selectedVariantID == entry.id
        return VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text(entry.name)
                    .font(.headline)
                AgentKindBadge(kind: entry.kind)
                Spacer()
                Text(entry.dateLabel)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
            Text(entry.baseModelID)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(entry.componentsSummary)
                .font(.caption)
                .foregroundStyle(.secondary)
            chipRow(for: entry, context: context)
            if let line = entry.promotionLine {
                Text(line)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                if let reason = entry.overrideReason {
                    Text("override reason: \(reason)")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                        .textSelection(.enabled)
                }
            }
            if let evidence = panel.robustnessByAgentID[entry.id] {
                Text(AgentEvidence.robustnessSummaryLine(evidence.report))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .help(
                        "latest robustness report for this agent "
                            + "(\(evidence.runDirectory.lastPathComponent))")
            }
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill(selected ? Color.accentColor.opacity(0.08) : .clear))
    }

    private func chipRow(
        for entry: AgentLibraryIndex.Entry, context: LibraryContext
    ) -> some View {
        HStack(spacing: 4) {
            ForEach(chips(for: entry, context: context)) { chip in
                AgentChipView(chip: chip)
            }
        }
    }

    /// Virtual baseline row — derived from a distinct base model id among the
    /// saved agents, never written to disk (no editor, no persistence; the
    /// only action is chatting with the unmodified base).
    private func baselineRowView(_ row: AgentLibrary.BaselineRow) -> some View {
        HStack(spacing: 6) {
            Text(row.name)
                .font(.callout.weight(.medium))
            AgentKindBadge(kind: .baseline)
            Spacer()
            Button("Chat") {
                service.workspaceSelectedModelID = row.baseModelID
                navigate(.playground)
            }
            .controlSize(.small)
            .help("open Playground with this base model selected, no interventions")
        }
        .padding(.vertical, 3)
        .padding(.horizontal, 6)
        .help("derived from the saved agents' base models — not an artifact on disk")
    }

    /// Direct actions on the selected saved agent: Chat (Playground),
    /// Run robustness (existing flow), Add to study (existing attach flow).
    @ViewBuilder
    private var selectedAgentActions: some View {
        if let record = panel.selectedVariant {
            // Five small buttons plus the name overrun the controls column;
            // the second layout puts the name on its own line.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    selectedAgentName(record)
                    Spacer()
                    selectedAgentButtons(record)
                }
                VStack(alignment: .leading, spacing: 6) {
                    selectedAgentName(record)
                    HStack(spacing: 8) { selectedAgentButtons(record) }
                }
            }
            .controlSize(.small)
        }
    }

    private func selectedAgentName(_ record: ModelVariantRecord) -> some View {
        Text(record.artifact.name)
            .font(.caption.weight(.semibold))
            .lineLimit(1)
            .help(record.artifact.name)
    }

    @ViewBuilder
    private func selectedAgentButtons(_ record: ModelVariantRecord) -> some View {
        Button("Chat") { chatWithSelectedAgent(record) }
            .disabled(
                isServerWorkspace
                    && selectedDefinitionApplicability?.isApplicable != true)
            .help(
                isServerWorkspace
                    ? "seed the Playground's steering controls from this "
                        + "agent (inline spec on the active server)"
                    : "load this agent's configuration into the Playground "
                        + "steering controls")
        Button("Edit") {
            createMode = .manual
            region = .create
        }
        .help("open this agent's definition in New Agent → Manual composition")
        if let promotion = record.artifact.promotion {
            Button("Open optimization run") {
                pendingOptimizationSelection = promotion.experiment
                region = .optimizations
            }
            .help(
                "open '\(promotion.experiment)' in Optimizations — the "
                    + "grid and recommendation this agent was promoted from")
        }
        Button("Run robustness") {
            guard !panel.isRobustnessRunning,
                panel.robustnessJudgeDisabledReason == nil
            else { return }
            // Explicitly point the check at THIS agent, then run —
            // the Robustness Check section's picker follows along.
            panel.robustnessTargetVariantID = record.id
            panel.runRobustnessCheck()
        }
        .disabled(
            panel.isRobustnessRunning
                || panel.robustnessJudgeDisabledReason != nil)
        .help(rowRunRobustnessHelp)
        Button("Add to study") { addSelectedAgentToStudy(record) }
            .disabled(!canAddSelectedAgentToStudy(record))
            .help(addToStudyHelp(record))
    }

    private func chatWithSelectedAgent(_ record: ModelVariantRecord) {
        if isServerWorkspace {
            service.applyLocalDefinitionToServerSteering(record)
        } else {
            service.applyModelVariantToSteering(record)
        }
        navigate(.playground)
    }

    private func canAddSelectedAgentToStudy(_ record: ModelVariantRecord) -> Bool {
        guard let study = service.experiments.management.selected else { return false }
        return study.status == .draft
            && study.modelID == record.artifact.baseModelID
    }

    private func addSelectedAgentToStudy(_ record: ModelVariantRecord) {
        // Align the panel's draft base-model field first: addVariantCondition
        // rewrites the study's model to that field when they differ (clearing
        // attached variants) — the guard above already ensured the study's
        // model matches this agent, so this makes the attach a pure attach.
        service.experiments.draft.studyBaseModelID = record.artifact.baseModelID
        do {
            let reviewed = try AgentArtifactSnapshot(workspaceRoot: ExperimentStore.workspaceRoot, reviewedRecord: record)
            service.experiments.addVariantCondition(reviewedAgent: reviewed)
            navigate(.studies)
        } catch {
            service.experiments.note(
                "Couldn't attach the agent: \(error.localizedDescription)",
                severity: .error)
        }
    }

    private func addToStudyHelp(_ record: ModelVariantRecord) -> String {
        guard let study = service.experiments.management.selected else {
            return "select a draft study in Studies first"
        }
        guard study.status == .draft else {
            return "'\(study.name)' is \(study.status.rawValue) — duplicate it to iterate"
        }
        guard study.modelID == record.artifact.baseModelID else {
            return "'\(study.name)' uses \(study.modelID), not this agent's base model"
        }
        return "pin this agent as an agent condition of '\(study.name)' (by artifact hash)"
    }

    private func robustnessSummary(_ report: VariantRobustnessReport) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            if let presetID = report.presetID,
                let preset = VariantRobustness.preset(id: presetID)
            {
                LabeledContent("Preset", value: preset.label)
            }
            if let substrate = report.substrate {
                LabeledContent("Substrate", value: substrate)
            }
            LabeledContent(
                "Capability",
                value:
                    "\(percent(report.variantBatteryAccuracy)) agent · "
                    + "\(percent(report.baselineBatteryAccuracy)) baseline")
            LabeledContent(
                "Distinct-2",
                value:
                    report.meanVariantDistinct2.formatted(.number.precision(.fractionLength(3)))
                    + " agent · "
                    + report.meanBaselineDistinct2.formatted(.number.precision(.fractionLength(3)))
                    + " baseline")
            if let judgeModel = report.judgeModel {
                let counts = Dictionary(grouping: report.coherenceItems.compactMap(\.judgeResult)) { $0 }
                    .mapValues(\.count)
                // WHO judged, not only how it voted: with three reachable
                // backends the kind (and an OpenRouter pin) is the
                // provenance a reader needs.
                LabeledContent(
                    "Judge",
                    value: VariantRobustnessReadout.judgeStamp(
                        model: judgeModel, kind: report.judgeKind,
                        provider: report.judgeProvider))
                Text(
                    "Judge: baseline \(counts["baseline"] ?? 0) · agent \(counts["variant"] ?? 0) · ties \(counts["tie"] ?? 0)"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                .help(
                    "how the judge voted across the coherence prompts. The "
                        + "report itself records the agent arm under its "
                        + "artifact name, \"variant\"")
                ForEach(report.coherenceItems.filter { $0.judge != nil }, id: \.index) { item in
                    if let judge = item.judge {
                        DisclosureGroup("Judge \(item.index): \(item.judgeResult ?? judge.winner)") {
                            Text(judge.briefReason)
                                .font(.caption)
                                .textSelection(.enabled)
                        }
                        .help(
                            "the judge's verdict and its brief reason for "
                                + "coherence prompt \(item.index) — "
                                + "\"variant\" is the report's name for the "
                                + "agent arm")
                    }
                }
            }
            if report.warnings.isEmpty {
                Label("No robustness warnings", systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.green)
            } else {
                ForEach(report.warnings, id: \.self) { warning in
                    Label(warning, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func percent(_ value: Float) -> String {
        value.formatted(.percent.precision(.fractionLength(0)))
    }

    private func nilIfEmpty(_ value: String) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }
}

/// The agent definition's temperature, with the number visible and typeable.
///
/// Was a bare `Slider` (audit 2026-09-06): 0 and 0.1 are one step apart and
/// looked identical, yet the difference decides whether the saved agent
/// generates deterministically at all — and the only way to read it back was
/// to open the artifact JSON. The study-wide control (`TemperatureRow`) has
/// the same shape but says "study-wide" in its help, which is the wrong
/// sentence here.
///
/// Its own type for the same reason `TemperatureRow` is: an inline
/// `LabeledContent { HStack { … } }` in an already-large `body` tips the
/// Swift type-checker over.
private struct AgentTemperatureRow: View {
    @Binding var value: Double

    var body: some View {
        LabeledContent("Temperature") {
            HStack(spacing: 8) {
                Slider(value: $value, in: 0 ... 1.5, step: 0.1)
                TextField("Temperature", value: $value, format: Self.format)
                    .labelsHidden()
                    .frame(width: 56)
                    .multilineTextAlignment(.trailing)
            }
        }
        .help(
            "this agent's generation temperature. 0 is greedy and "
                + "reproducible; anything above it makes the agent's answers "
                + "sampled, so a measured study run needs a seed policy to "
                + "stay comparable")
    }

    private static let format = FloatingPointFormatStyle<Double>()
        .precision(.fractionLength(0 ... 2))
}
