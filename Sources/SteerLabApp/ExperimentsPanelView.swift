import ExperimentKit
import SteeringKit
import SwiftUI
import UniformTypeIdentifiers

/// Domain-neutral study lifecycle: create a draft protocol → attach concepts
/// → add baseline / capture steering conditions → verify → freeze (one-way)
/// → run headlessly via the CLI.
struct ExperimentsPanelView: View {
    @Bindable var service: ChatService
    /// Lands on Agents → Optimizations — Studies shows optimization
    /// provenance but does not mint agents (that happens in Agents).
    var openOptimizations: () -> Void = {}
    /// Lands on the Templates tab — the design library. Studies CASTS designs;
    /// it does not hold them (2026-08-06 restructure).
    var openTemplates: () -> Void = {}
    @State private var confirmFreeze = false
    /// A12: delete-draft confirmation (move-to-trash, never destructive).
    @State private var confirmDeleteDraft = false
    /// Runs stamped with the study being deleted, read at click time (never
    /// per frame — it scans runs/) so the confirmation can say what is at stake.
    @State private var deleteDraftRunCount = 0
    @State private var reviewSheet: ResultReviewSheet?
    /// The one Rename affordance, offered for every study whatever its
    /// status: a draft renames for real, a frozen/complete study takes a
    /// display label (see `ExperimentStore.rename` for why the two differ).
    @State private var renameSheet: RenameStudySheet?
    @State private var reconnectJobID = ""
    /// Bound expansion state for the "Remote options" disclosure so
    /// cross-links (Optimizations' preconfigured sweep) can open it directly.
    @State private var runOnServerExpanded = false
    /// WS6.3 unified Run: the substrate picker's manual override (nil follows
    /// the active scope) and the submission engine that owns the remote
    /// bundle path + WS4 preflight surfacing. All rules live in
    /// `SubstrateRouting` / `PreflightPresentation` (ExperimentKit,
    /// unit-tested); this view renders them.
    @State private var runSubstrate: SubstrateRouting.Substrate?
    @State private var runner = UnifiedStudyRunner()
    @State private var confirmForcedOverride = false
    /// Robustness reports scanned once per appearance (not per frame) so
    /// attached agent conditions can show non-blocking evidence notes.
    @State private var robustnessEvidence: [AgentEvidence.RobustnessEvidence] = []
    /// Import JSONL… (Input Data): sheet visibility and its pasted/loaded
    /// text. Parsing/preview/import rules live in `TaskPromptsImport`
    /// (ExperimentKit, unit-tested); the sheet renders them.
    @State private var showImportJSONL = false
    @State private var importJSONLText = ""
    /// Item 2 (cluster-testing): a model-running server submission parked
    /// while the shared no-GPU-session dialog asks.
    @State private var pendingModelJob: PendingModelJob?
    /// The template-instantiation sheet (the cell table) and the two modest
    /// library affordances beside it. A template is a draft of the library —
    /// never frozen, nothing stamps its name into evidence — so rename and
    /// delete are unconditional, unlike the study equivalents.
    /// The new-studies sheet (the cell table). The design LIBRARY — rename,
    /// delete, the design summary — lives in the Templates tab; Studies only
    /// casts designs into studies.
    @State private var templateSheet: TemplateInstantiationRequest?
    /// "Save back to design" confirmation — the one write in this panel that
    /// changes an artifact OUTSIDE the selected study.
    @State private var confirmSaveBackToDesign = false

    private var panel: ExperimentPanel { service.experiments }

    /// Study list row label; manifests with a declared sweep carry a small
    /// "optimization" badge (Home's studyRowLabel shows the same).
    /// Labels for the duplicate FAMILIES in the current list, keyed by study
    /// name. `x`, `x-2` and `x-2-2` are three distinct names, so nothing else
    /// treats them as ambiguous — yet they are exactly the set that cannot be
    /// told apart, since duplicating-to-iterate is how the lifecycle says to
    /// change a study.
    private var duplicateFamilyLabels: [String: ArtifactDisambiguation.Label] {
        var out: [String: ArtifactDisambiguation.Label] = [:]
        for (_, labels) in ArtifactDisambiguation.familyLabels(
            service.experiments.experiments)
        {
            for label in labels { out[label.id] = label }
        }
        return out
    }

    private func studyPickerLabel(
        _ manifest: ExperimentManifest,
        families: [String: ArtifactDisambiguation.Label] = [:]
    ) -> String {
        // A display label leads, but never REPLACES the canonical name: run
        // directories, config.json stamps and CLI arguments all speak the
        // canonical one, so it has to stay correlatable here.
        let display = panel.displayName(manifest)
        var label =
            display == manifest.name
            ? "\(manifest.name)  [\(manifest.status.rawValue)]"
            : "\(display)  ·  \(manifest.name)  [\(manifest.status.rawValue)]"
        if manifest.sweep != nil {
            label += "  · optimization"
        }
        // Lineage badge instead of a filter control: a batch of six castings
        // reads as six adjacent rows sharing one template name, which is the
        // grouping a researcher actually wants and costs no new UI.
        if let template = manifest.templateProvenance?.template {
            label += "  · from \(template)"
        }
        // What distinguishes this one from its duplicates — or that nothing
        // does, which is itself the answer worth having.
        if let distinguisher = families[manifest.name]?.distinguisher {
            label += "  · \(distinguisher)"
        }
        return label
    }

    var body: some View {
        @Bindable var panel = service.experiments
        Form {
            Section {
                Picker("Draft", selection: $panel.selectedName) {
                    Text("select…").tag(String?.none)
                    let families = duplicateFamilyLabels
                    ForEach(panel.experiments, id: \.name) { manifest in
                        // Optimization badge (mirrors Home's studyRowLabel):
                        // Studies reads as inventory/provenance; optimization
                        // authoring lives in Agents → Optimizations.
                        Text(studyPickerLabel(manifest, families: families))
                            .tag(String?.some(manifest.name))
                    }
                }
                .help("studies are versioned manifests in experiments/<name>/ — "
                    + "'optimization' marks a declared sweep (authored in "
                    + "Agents → Optimizations)")

                VStack(alignment: .leading, spacing: 6) {
                    // The new-study flow starts from a DESIGN choice
                    // (2026-08-06). "From scratch" is the blank interface
                    // exactly as before; a saved design opens the
                    // new-studies table prefilled, so the only decision left
                    // is the casting.
                    newStudyDesignPicker(panel: panel)
                    // One click in, name it after. Naming a study before it
                    // exists is a decision the researcher cannot yet make,
                    // and a draft is renamable for as long as it stays a
                    // draft — so the name is no longer a gate on starting.
                    HStack(spacing: 8) {
                        Button("New Study") { startNewStudy(panel: panel) }
                            .help(
                                panel.newStudyDesign.designName == nil
                                    ? "creates a draft pinned to the currently selected "
                                        + "model under a placeholder name, and opens "
                                        + "Rename so you can name it now"
                                    : "opens the new-studies table on this design — one "
                                        + "ordinary draft per casting")
                        if let manifest = panel.selected {
                            Button {
                                openRename(manifest)
                            } label: {
                                Label("Rename…", systemImage: "pencil")
                            }
                            .help(
                                manifest.status == .draft
                                    ? "change this draft's name, its display label, or both"
                                    : "frozen and completed studies keep their name for "
                                        + "run provenance — Rename sets a display label")
                            Button("Duplicate as Draft") { panel.duplicateSelected() }
                                .help(Self.duplicateHelp)
                            Button("Delete…", role: .destructive) {
                                deleteDraftRunCount = ExperimentStore.runsStamped(
                                    experimentName: manifest.name)
                                confirmDeleteDraft = true
                            }
                            .disabled(panel.deleteSelectedStudyRefusal != nil)
                            .help(panel.deleteSelectedStudyRefusal ?? Self.deleteStudyHelp)
                            .confirmationDialog(
                                "Delete draft '\(manifest.name)'?",
                                isPresented: $confirmDeleteDraft,
                                titleVisibility: .visible
                            ) {
                                Button(
                                    "Move '\(manifest.name)' to trash",
                                    role: .destructive
                                ) {
                                    panel.deleteSelectedDraft()
                                }
                            } message: {
                                Text(deleteDraftMessage(manifest))
                            }
                        }
                    }
                    if let refusal = panel.formErrors[.rename] {
                        Label(refusal, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .fixedSize(horizontal: false, vertical: true)
                            .textSelection(.enabled)
                    }
                    // The naming/model detail the one-click button skips.
                    // Reachable, never in the way (capability preserved).
                    DisclosureGroup("Advanced") {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Create a study with a specific name and a "
                                + "pinned model revision up front. The normal "
                                + "path is New Study, then Rename — a draft "
                                + "renames freely for as long as it stays a "
                                + "draft, and the revision auto-pins from the "
                                + "local HF cache at the first "
                                + "extract/validate. Use this when you already "
                                + "know both, e.g. reproducing a study on a "
                                + "named model snapshot.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                            TextField("new study name", text: $panel.newName)
                                .help("creates a draft pinned to the currently selected model")
                            TextField(
                                "question or purpose", text: $panel.newDescription,
                                axis: .vertical
                            )
                            .lineLimit(1 ... 3)
                            .help("short domain-neutral purpose for the draft protocol")
                            TextField(
                                "model revision (optional commit hash)",
                                text: $panel.newRevision
                            )
                            .font(.caption.monospaced())
                            .help(
                                "pins the exact HF snapshot commit up front — a frozen "
                                    + "study must never silently run another model version")
                            Text("empty = auto-pin from the local HF cache at the first "
                                + "extract/validate")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            Button("Create Draft") { panel.create() }
                                .disabled(panel.newName.isEmpty)
                        }
                    }
                }
                // New Study mints a placeholder name and asks for the rename
                // immediately — the one-click flow is only an improvement if
                // naming follows it.
                .onChange(of: panel.renameInvitation) {
                    consumeRenameInvitation(panel: panel)
                }
            } header: {
                // A15: the persistent-notices bell lives in the section
                // header so overnight failures are one click away.
                HStack {
                    Text("Study")
                    Spacer()
                    NoticesBellButton()
                }
            }

            if let manifest = panel.selected {
                // THE classifier, FIRST on the page: one "Study type"
                // control (2026-07-19 second pass — replaces the old
                // Study stage / Study Focus duo, whose disagreement plus
                // bottom-of-page placement caused both the contradiction
                // and the scroll jump; sections now only ever
                // appear/disappear BELOW the control that toggles them).
                StudyTypeSection(manifest: manifest, panel: panel)

                Section("Study Setup") {
                    TextField(
                        "question or purpose",
                        text: $panel.protocolDescription,
                        axis: .vertical
                    )
                    .lineLimit(1 ... 3)
                    .disabled(manifest.status != .draft)
                    .help("what this study asks; not tied to any domain")

                    TextField(
                        "task: what the model or agents will do",
                        text: $panel.taskDescription,
                        axis: .vertical
                    )
                    .lineLimit(2 ... 5)
                    .disabled(manifest.status != .draft)
                    .help(
                        "examples: write an opinion, play a game, answer advice "
                            + "prompts, allocate resources, classify scenarios")

                    TextField(
                        "outcome measures",
                        text: $panel.outcomeMeasures,
                        axis: .vertical
                    )
                    .lineLimit(2 ... 5)
                    .disabled(manifest.status != .draft)
                    .help(
                        "what you will measure: flips, scores, cooperation rate, "
                            + "style markers, capability battery, degeneration")

                    if panel.studyKind == .modelOutput {
                        // Workspace-scoped model choice (same strict rule as
                        // the chat's WorkspaceModelPicker): a server target
                        // offers ONLY that server's installed models; a
                        // current selection outside the inventory stays
                        // rendered — "(not installed)" — but is never
                        // pickable anew. The chosen server id flows into the
                        // manifest exactly as local ids do.
                        Picker(
                            panel.studyKind == .multiAgent
                                ? "Default model for seats"
                                : "Baseline model",
                            selection: $panel.studyBaseModelID
                        ) {
                            if panel.studyBaseModelID.isEmpty {
                                Text("select model…").tag("")
                            }
                            ForEach(panel.modelOptions, id: \.self) { model in
                                Text(model).tag(model)
                            }
                            if WorkspaceScoping.selectionOutsideInventory(
                                panel.studyBaseModelID, inventory: panel.modelOptions)
                            {
                                Text(
                                    panel.isServerWorkspace
                                        ? "\(panel.studyBaseModelID) (not installed)"
                                        : panel.studyBaseModelID)
                                    .tag(panel.studyBaseModelID)
                                    .selectionDisabled()
                            }
                        }
                        .disabled(manifest.status != .draft)
                        .help(
                            panel.studyKind == .multiAgent
                                ? "used only by panel seats that name no base model of "
                                    + "their own. Seats may each carry a different "
                                    + "model; every turn records the one it ran on, and "
                                    + "this value is not a claim about the run."
                                : panel.isServerWorkspace
                                ? "the unmodified baseline model and required base for "
                                    + "added agents — models installed on "
                                    + "\(service.cluster.substrateLabel) (the active "
                                    + "compute workspace)"
                                : "the unmodified baseline model and required base for added agents")
                        if panel.isServerWorkspace, panel.modelOptions.isEmpty {
                            Text(
                                "no models installed on \(service.cluster.substrateLabel) — "
                                    + "use Install model… (Compute menu) to prefetch one")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        // Draft-editable revision pin (audit 2026-08-01:
                        // create-time only before — changing it required
                        // duplicate-or-paste-JSON).
                        ModelRevisionControls(manifest: manifest, panel: panel)
                        studyDtypePicker(
                            manifest: manifest,
                            selection: $panel.studyDtypeField)

                        HStack(spacing: 6) {
                            Picker("Baseline prompt mode", selection: $panel.promptMode) {
                                ForEach(ExperimentManifest.PromptMode.allCases, id: \.self) { mode in
                                    Text(mode.label).tag(mode)
                                }
                            }
                            .disabled(manifest.status != .draft)
                            .help(
                                "baseline chat assistant uses the model chat template with "
                                    + "user/assistant roles; raw completion sends literal text. "
                                    + "Saved agents use their own prompt mode")
                            InfoButton(text: StudyInfo.promptMode)
                        }

                        // Gemma has NO system role — the same text is
                        // prepended to the first user turn instead, so the
                        // label says what actually happens for the chosen
                        // model family.
                        TextField(
                            manifest.modelID.lowercased().contains("gemma")
                                ? "Baseline instruction (Gemma: prepended to "
                                    + "the first user turn — no system role)"
                                : "Baseline system prompt",
                            text: $panel.systemPrompt, axis: .vertical)
                            .lineLimit(1 ... 4)
                            .disabled(manifest.status != .draft)
                            .help(
                                "optional standing instruction for the baseline "
                                    + "condition; saved agents use their own. "
                                    + "Qwen sends it as a true system message; "
                                    + "Gemma's chat template has no system role, "
                                    + "so the SAME text is prepended to the first "
                                    + "user turn (this affects prompt-set "
                                    + "hashing and prompt design)")

                        Picker("Reasoning effort", selection: $panel.reasoningEffort) {
                            ForEach(ReasoningEffort.vocabulary, id: \.self) { effort in
                                Text(effort).tag(effort)
                            }
                        }
                        .disabled(manifest.status != .draft
                            || !PromptRendering.hasThinkingMode(
                                manifest.modelID,
                                capabilities: ExperimentStore.modelCapabilities(for: manifest)))
                        .help("The reasoning effort the chat template is rendered with "
                            + "(off = no thinking block; on = thinking at the template's "
                            + "default effort). A non-off effort needs a reasoning token "
                            + "budget, and a LEVEL only where the model's capability "
                            + "record — probed from its chat template, shown in the "
                            + "Compute section — says the template accepts it.")
                        if panel.qwenThinkingEnabled {
                            TextField(
                                "Reasoning max tokens",
                                value: $panel.reasoningMaxTokens,
                                format: .number)
                            .disabled(manifest.status != .draft)
                            .help("The reasoning block's own token cap (up to </think>); "
                                + "Max tokens is then the answer budget. Required.")
                        }
                    }

                    TemperatureRow(value: $panel.runTemperature)
                        .disabled(manifest.status != .draft)

                    LabeledContent("Max tokens") {
                        TextField(
                            "", value: $panel.runMaxTokens,
                            format: .number.grouping(.never)
                        )
                        .frame(width: 72)
                        .multilineTextAlignment(.leading)
                    }
                    .disabled(manifest.status != .draft)
                    .help("study-wide per-response token cap for baseline and agents")

                    if panel.studyKind == .multiAgent {
                        Picker("Scenario",
                               selection: $panel.selectedMultiAgentScenarioID) {
                            Text("select…").tag(String?.none)
                            ForEach(panel.multiAgentScenarioOptions) { scenario in
                                Text(Self.scenarioMenuLabel(scenario))
                                    .tag(String?.some(scenario.id))
                            }
                        }
                        .disabled(manifest.status != .draft)
                        .help(
                            "the saved scenario this study runs: its roles, "
                                + "turns and materials. Selecting it does NOT "
                                + "pin it — Save study setup compiles the seat "
                                + "casting below and writes the pin, which is "
                                + "what Data & Prompts checks.")
                        if panel.selectedMultiAgentScenarioID != nil,
                            manifest.multiAgentScenarioPath == nil
                        {
                            Text("selected but not pinned — Save study setup to "
                                + "pin it")
                                .font(.caption2)
                                .foregroundStyle(.orange)
                        }
                        Toggle(
                            "Include stripped baseline arm",
                            isOn: $panel.multiAgentIncludeBaseline
                        )
                        .disabled(manifest.status != .draft)
                        .help(
                            "runs the SAME panel a second time with every "
                                + "intervention removed — adapters and steering "
                                + "vectors stripped, base models unchanged. This "
                                + "is the control the measurement subtracts.")
                        if !panel.multiAgentIncludeBaseline {
                            // Not a style preference: without the control arm
                            // there is nothing to difference against, so the
                            // whole analysis layer goes quiet.
                            Text("without the baseline arm this study produces no "
                                + "effect sizes and no panel-effect decomposition "
                                + "— there is nothing to compare the panel against")
                                .font(.caption2)
                                .foregroundStyle(.orange)
                        }
                    }

                    // Both study kinds need this. For a panel, "samples per
                    // item" IS the number of independent play-throughs
                    // (transcripts) — the unit the statistics treat as one
                    // observation, and the unit sharding splits over. It was
                    // gated to model-output studies, so panel replicates were
                    // implemented on both engines and unreachable from here.
                    samplingControls(manifest: manifest, panel: panel)

                    // The funnel phase has two REAL effects (it is not a
                    // note): analyze picks its multiple-comparison
                    // correction from it, and phase=confirm activates the
                    // held-out-pool verification rule. Never sent to any
                    // model. CONCEPT studies only (2026-07-21, issue 2):
                    // screen/confirm is concept-funnel vocabulary — the
                    // picker is hidden, not removed, for other types, and
                    // an existing manifest's value survives untouched
                    // (saveProtocol always writes phaseField back).
                    if panel.studyFocus == .conceptStudy {
                        HStack(spacing: 6) {
                            Picker("Funnel phase", selection: $panel.phaseField) {
                                Text("not declared").tag("")
                                ForEach(ExperimentStore.knownPhases, id: \.self) { phase in
                                    Text(phase).tag(phase)
                                }
                            }
                            .disabled(manifest.status != .draft)
                            .help(
                                "where this study sits in the screen→confirm→"
                                    + "triangulate→panel funnel. Two real effects: "
                                    + "analyze picks the correction from it (screens "
                                    + "get BH-FDR, confirms get the stricter Holm), "
                                    + "and 'confirm' additionally requires this "
                                    + "study's prompts to be HELD OUT from the "
                                    + "screen pool it references (a verify rule). "
                                    + "Changes no prompt; never sent to any model")
                            InfoButton(text: StudyInfo.funnelPhase)
                        }
                    }

                    if manifest.status == .draft {
                        Button("Save Study Setup") { panel.saveProtocol() }
                            .help(
                                "save the study question, baseline model, "
                                    + "generation & sampling settings (and, "
                                    + "for concept studies, the funnel phase)")
                    }
                }

                // ONE Conditions section, type-dependent (2026-07-19 second
                // pass): the arms of the study — agents for a comparison,
                // the scenario for multi-agent, the perturbation policy for
                // a confirmation. Concept studies get their derivation
                // machinery in the sections that follow.
                studyArmsSection(manifest: manifest, panel: panel)

                // WHO sits in each seat of the chosen scenario. Only
                // multi-agent studies have seats, and a scenario that declares
                // none has nothing to show.
                if panel.studyKind == .multiAgent {
                    seatsSection(manifest: manifest, panel: panel)
                }

                if panel.studyKind == .modelOutput,
                    panel.studyFocus == .conceptStudy
                {
                    conceptsSection(manifest: manifest, panel: panel)
                    conditionsSection(manifest: manifest, panel: panel)
                }

                // What data this study still needs and where it goes —
                // derived from the manifest alone (StudyDataReadiness,
                // ExperimentKit); blockers surface here instead of at a
                // failing gate. The task-prompts CONTENT editor lives
                // inside this pane too (2026-07-19: its standalone section
                // duplicated the row above it) — the row's Edit button
                // expands it in place.
                DataReadinessSection(
                    manifest: manifest,
                    onEditTaskPrompts: { panel.loadTaskPromptsInteractively() },
                    relevantCategories: panel.studyFocus.relevantDataCategories,
                    panel: panel,
                    taskPromptsEditor: panel.studyKind == .modelOutput
                        ? {
                            AnyView(
                                taskPromptsEditorContent(
                                    manifest: manifest, panel: panel))
                        }
                        : nil)

                Section("Evaluation") {
                    // Instrument activation is a MODEL-OUTPUT concern (the
                    // outcome instruments score per-prompt generations);
                    // multi-agent studies judge transcripts — their
                    // Evaluation pane shows judging only (2026-07-19 type
                    // parity cleanup).
                    if panel.studyKind == .modelOutput {
                        instrumentActivationControls(
                            manifest: manifest, panel: panel)
                    }
                    judgingControls(manifest: manifest, panel: panel)
                    // Discriminant controls + instrument scope are
                    // evaluation declarations — relocated here from the
                    // actions section (2026-08-01), where nobody thought
                    // to look for them. (The validation read DEPTH lives
                    // with the Validate button instead — review feedback:
                    // the setting belongs where validation is launched.)
                    if !manifest.concepts.isEmpty {
                        DiscriminantControlsSection(
                            manifest: manifest, panel: panel)
                    }
                    // The instrument scope is about task prompts, not
                    // concepts — a concept-less Compare agents study with
                    // mixed response formats NEEDS it to declare an
                    // answer-token instrument (2026-08-03: the run refusal
                    // named a field the UI hid).
                    InstrumentScopeSection(manifest: manifest, panel: panel)
                }

                // Declare the chain (stages + gates) as manifest data and
                // submit it — the app authors the pipeline it runs. Hidden
                // for multi-agent studies (the chain has no multi-agent
                // stages yet) unless one is already declared. Concept
                // studies also declare the promotion rule here — the
                // screen→confirm gate the funnel's promote step must pass.
                if panel.studyFocus != .multiAgent || manifest.pipeline != nil {
                    PipelineComposerSection(
                        manifest: manifest, panel: panel,
                        relevantStages: panel.studyFocus.relevantPipelineStages,
                        showsPromotionRule: panel.studyFocus == .conceptStudy,
                        // 2026-07-21 incident part 1: the pipeline verb is a
                        // model-running bundle submission like Run — route it
                        // through the same one-dialog GPU gate.
                        submitAction: {
                            let panel = panel
                            ModelJobGPUGate.submit(
                                "study pipeline", service: service,
                                pending: $pendingModelJob,
                                bundleOptions:
                                    ModelJobSubmissionPreflight.BundleOptions(
                                        executor: panel.remoteExecutor,
                                        gres: panel.remoteGres,
                                        verb: "pipeline",
                                        dryRun: panel.remoteDryRun),
                                fixOptions: {
                                    panel.applyGPUAllocationFix()
                                    runOnServerExpanded = true
                                }
                            ) { await panel.runPipelineRemotely() }
                        })
                }

                // Provenance summary only — the pinned-file rows that used
                // to repeat here live in Data & Prompts, and violations
                // moved into the Issues box below (one place for what's
                // wrong).
                Section("\(manifest.name) — \(manifest.status.rawValue)") {
                    LabeledContent("Model", value: manifest.modelID)
                        .help("runs and extraction use this model")
                    LabeledContent(
                        "Revision",
                        value: manifest.modelRevision.map { String($0.prefix(12)) + "…" }
                            ?? "unpinned")
                        .help(
                            "the exact HF snapshot commit experiment runs load — pinned "
                                + "from the local cache by the first extract/validate/"
                                + "sweep, or at freeze; without it a frozen experiment "
                                + "could silently run a different model version")
                    if let hash = manifest.freezeHash {
                        LabeledContent("Freeze hash", value: String(hash.prefix(16)) + "…")
                            .help("canonical content hash stamped at freeze; every run records it")
                        LabeledContent(
                            "Git commit", value: String(manifest.gitCommit?.prefix(8) ?? "—"))
                            .help("repo state when frozen — commit stimulus work before freezing")
                    }
                    // Lineage: which recipe this study came from, and which
                    // batch of siblings it belongs to. Panel castings ARE
                    // sibling studies (one scenario per manifest on both
                    // engines), so the batch id is the only thing that puts
                    // them back together for analysis.
                    templateLineageRows(manifest: manifest, panel: panel)
                    // The round trip's return leg, next to the lineage line
                    // that names where this study came from.
                    designWriteBackRow(manifest: manifest, panel: panel)
                }

                StudyIssuesSection(manifest: manifest, panel: panel)

                Section {
                    if manifest.status == .draft {
                        freezeControls(manifest: manifest, panel: panel)
                    }
                    // Duplicate as Draft and Delete moved UP to the Study
                    // section (2026-08-06): the four things you do TO a study
                    // — new, rename, duplicate, delete — belong together, not
                    // split across the page from the run actions.

                    if panel.studyKind == .modelOutput {
                        validationControls(manifest: manifest, panel: panel)
                        if !manifest.concepts.isEmpty {
                            extractControls(manifest: manifest, panel: panel)
                            // DiscriminantControlsSection moved to the
                            // Evaluation section (2026-08-01) — declaring
                            // controls is an evaluation decision, not a run
                            // action.
                        }
                    }
                    runControls(manifest: manifest, panel: panel)
                    remoteRunControls(manifest: manifest, panel: panel)
                }

                if !panel.awaitingSweepJudgments.isEmpty,
                    let studyName = panel.selectedName
                {
                    Section("Awaiting judgment") {
                        ForEach(panel.awaitingSweepJudgments) { awaiting in
                            HStack(spacing: 8) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(awaiting.run)
                                        .font(.caption)
                                        .truncationMode(.head)
                                    Text(
                                        (awaiting.isEvaluate
                                            ? "evaluation · " : "sweep · ")
                                            + "\(awaiting.packetCount ?? 0) "
                                            + "blinded packets · judges: "
                                            + (awaiting.judges ?? [])
                                            .compactMap(\.name)
                                            .joined(separator: ", "))
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                Button("Judge on this Mac") {
                                    Task {
                                        await panel.judgeAwaitingSweep(
                                            study: studyName,
                                            awaiting: awaiting)
                                    }
                                }
                                .disabled(panel.isJudgingSweep)
                                .help(
                                    "judge the sweep's blinded packets with "
                                        + "Claude using the key in this "
                                        + "Mac's Keychain, then let the "
                                        + "server verify pins and compute "
                                        + "the selection — the key never "
                                        + "goes to the cluster")
                            }
                        }
                        Text(
                            "this sweep generated on the cluster and emitted "
                                + "blinded comparison packets — Claude "
                                + "judging runs on this Mac (key-custody "
                                + "design); the completed selection lands "
                                + "under Optimization results")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }

                if !panel.promotableRecommendations.isEmpty {
                    Section("Optimization results") {
                        ForEach(panel.promotableRecommendations, id: \.name) { condition in
                            sweepRecommendationRow(condition, panel: panel)
                        }
                        Text(
                            "read-only provenance — agents are created from "
                                + "these cells in Agents → Optimizations, not "
                                + "here (Studies consumes agents; it does not "
                                + "mint them)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }

                liveRunViewer(panel: panel)

                // ONE results area (2026-07-19 second pass): what used to
                // be four sibling sections — Results, Server Runs,
                // Pipelines, Recent Server Jobs — whose differences the
                // researcher had to guess. Same content, one roof,
                // subheaded by WHERE the runs live and at what granularity.
                Section("Runs & Results") {
                    if !panel.recentServerJobs.isEmpty {
                        recentServerJobsGroup(panel: panel)
                    }
                    Text("Run reports — this workspace")
                        .font(.caption.bold())
                        .padding(.top, 4)
                    resultsView(panel: panel)
                    // Runs are per-substrate artifacts: in a server
                    // workspace, also list the server's runs/ tree.
                    if service.cluster.computeTarget == .server {
                        serverRunsGroup(panel: panel)
                        pipelinesGroup(panel: panel)
                    }
                }
            }

            if let status = panel.status {
                Section {
                    Text(status).font(.caption).foregroundStyle(.secondary)
                    // A durable server job in flight gets a visible cancel
                    // control right here — not buried in a disclosure.
                    if let job = panel.activeServerJob {
                        HStack(spacing: 8) {
                            ProgressView().controlSize(.small)
                            Text("server \(job.verb) job \(job.id) — '\(job.study)'")
                                .font(.caption.monospaced())
                                .textSelection(.enabled)
                            Button("Cancel Server Job", role: .destructive) {
                                Task { await panel.cancelActiveServerJob() }
                            }
                            .controlSize(.small)
                            .help(
                                "requests cancellation of the durable job on the "
                                    + "server; the run stops at the next record and "
                                    + "the job is marked cancelled")
                        }
                    }
                }
            }
        }
        .sheet(item: $reviewSheet) { sheet in
            ResultReviewWindow(sheet: sheet)
        }
        .sheet(item: $renameSheet) { sheet in
            RenameStudyWindow(sheet: sheet, panel: panel)
        }
        .sheet(item: $templateSheet) { request in
            TemplateInstantiationSheet(request: request, panel: panel)
        }
        // The Templates tab's Instantiate opens the new-studies flow PRE-LOADED
        // with that design. Consumed on CHANGE (the same-tab paths) and on
        // APPEAR (the cross-section handoff — this view is not on screen when
        // Templates sets it).
        .onChange(of: panel.templateInstantiationInvitation) {
            consumeInstantiationInvitation(panel: panel)
        }
        // Item 2 (cluster-testing): the shared no-GPU-session warning for
        // this panel's model-running server submissions (run/sweep bundle,
        // validate, extract).
        .modelJobGPUWarning(pending: $pendingModelJob, service: service)
        .formStyle(.grouped)
        .onAppear {
            panel.refresh()
            consumeInstantiationInvitation(panel: panel)
            // Same one-shot pattern, same reason as the instantiation
            // invitation: Templates' "New Template" creates the draft and
            // navigates here, so this view is not on screen when the rename
            // invitation is set and the onChange above never fires.
            consumeRenameInvitation(panel: panel)
            // A cross-link (e.g. Optimizations' "Submit Bundle: sweep") preselected
            // the study and verb — surface the Run-on-Server controls so the
            // prepared submission is visible, then clear the one-shot flag.
            if panel.pendingRevealRemoteControls {
                panel.pendingRevealRemoteControls = false
                runOnServerExpanded = true
            }
            if service.cluster.computeTarget == .server {
                Task { await panel.refreshRecentServerJobs() }
            }
        }
        // Evidence notes for attached agents: refresh the library (for the
        // artifact-resolution check) and scan robustness reports once per
        // appearance — never per frame, and never ON the appearance path:
        // both are directory walks whose cost scales with the workspace, and
        // no row needs them to draw (same rule as the Agents tab's
        // `refreshAgentLibraryAsync`). `.task` runs after the first draw;
        // the previous visit's evidence stays visible while the rescan runs,
        // and the task's own cancellation is the latest-wins guard — a
        // superseded appearance never lands its stale reports.
        .task {
            service.fineTuning.refreshAgentLibraryAsync()
            let runs = ExperimentStore.runsDirectory
            let reports = await Task.detached(priority: .utility) {
                AgentEvidence.scanRobustnessReports(runsDirectory: runs)
            }.value
            if !Task.isCancelled { robustnessEvidence = reports }
        }
        // Preflight server residency whenever the selection or the active
        // workspace changes (cached per selection inside the panel — this
        // does not hammer the experiment listing).
        .task(id: residencyTaskKey) {
            await panel.refreshServerResidency()
            if let study = panel.selectedName {
                await panel.refreshAwaitingSweepJudgments(study: study)
            }
        }
        // A different study is a different design: the substrate override
        // and any surfaced preflight belong to the previous selection.
        .onChange(of: panel.selectedName) {
            runSubstrate = nil
            runner.clearPreflight()
        }
        .sheet(isPresented: $showImportJSONL) {
            ImportJSONLSheet(
                text: $importJSONLText,
                destination: panel.selected.map {
                    DataTemplates.taskPromptsDestination(experiment: $0.name)
                },
                onImport: { text, replace in
                    panel.importTaskPromptsJSONL(
                        text, replacingExisting: replace)
                },
                statusLine: { panel.taskPromptsStatus })
        }
    }

    /// Template lineage on the study detail — one subtle line, plus the
    /// batch's other studies by DISPLAY name (the batch is read by a human,
    /// and canonical casting names are unreadable by design).
    @ViewBuilder
    private func templateLineageRows(
        manifest: ExperimentManifest, panel: ExperimentPanel
    ) -> some View {
        if let lineage = panel.templateLineage(manifest) {
            Text(lineage)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            let siblings = panel.batchSiblings(manifest)
            if !siblings.isEmpty {
                DisclosureGroup("Minted with \(siblings.count) sibling study(s)") {
                    ForEach(siblings, id: \.self) { sibling in
                        Text(sibling)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                .font(.caption2)
            }
        }
    }

    /// The two ways this study's settings become a design — both visible, both
    /// worded for what they do, neither a default.
    ///
    /// "Save back to design" is the return leg of Templates' "Edit design…": it
    /// OVERWRITES the design the lineage line names. "Save as new design" is the
    /// existing mint, which adds an entry. The difference matters enough to be
    /// two buttons rather than one button with a mode: one of them grows the
    /// library and the other does not.
    @ViewBuilder
    private func designWriteBackRow(
        manifest: ExperimentManifest, panel: ExperimentPanel
    ) -> some View {
        let target = panel.saveBackToDesignTarget(for: manifest)
        let refusal = panel.saveBackToDesignRefusal(for: manifest)
        HStack(spacing: 8) {
            if let target {
                Button("Save back to design '\(target)'") {
                    confirmSaveBackToDesign = true
                }
                .disabled(refusal != nil)
                .help(refusal ?? Self.saveBackHelp)
                .confirmationDialog(
                    "Update design '\(target)'?",
                    isPresented: $confirmSaveBackToDesign,
                    titleVisibility: .visible
                ) {
                    Button("Update '\(target)' in place") {
                        panel.saveSelectedStudyBackToDesign()
                    }
                } message: {
                    Text(Self.saveBackConfirmation(design: target))
                }
            }
            Button("Save as new design") {
                panel.newDesignFromStudy(named: manifest.name)
            }
            .help(Self.saveAsNewDesignHelp)
            if target == nil {
                Button("Open Templates") { openTemplates() }
                    .buttonStyle(.link)
                    .font(.caption2)
            }
        }
        if let refusal, target != nil {
            Text(refusal)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        if let error = panel.formErrors[.template] {
            Label(error, systemImage: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
    }

    /// Stated plainly, because it is the one thing about the round trip a
    /// researcher could be surprised by afterwards.
    private static func saveBackConfirmation(design: String) -> String {
        "Strips this study to its design form — every generation and "
            + "measurement setting, no agents — and updates design '\(design)' "
            + "in place. Its content hash changes. Studies already minted from "
            + "it keep their original lineage stamps, so their divergence "
            + "display goes on reporting what they were minted from. The "
            + "design's name, description and creation date are unchanged."
    }

    private static let saveBackHelp =
        "overwrites the design this study was minted from with this study's "
        + "current settings (agents stripped). The design's content hash "
        + "changes; studies already minted from it keep their original lineage "
        + "stamps."

    private static let saveAsNewDesignHelp =
        "adds a NEW design to the library from this study's settings, leaving "
        + "any design it came from untouched. An unchanged instance of a live "
        + "design selects that design instead of minting a near-duplicate."

    // MARK: Starting a study from a design

    /// The first control in the new-study flow: what this study starts FROM.
    ///
    /// The design LIBRARY is the Templates tab; this is only the choice, so
    /// that "new study" is one flow with two beginnings rather than two flows
    /// the researcher has to know to pick between.
    @ViewBuilder
    private func newStudyDesignPicker(panel: ExperimentPanel) -> some View {
        @Bindable var panel = panel
        Picker("Start from", selection: $panel.newStudyDesign) {
            ForEach(StudyDesignChoice.choices(designs: panel.templates)) { choice in
                Text(choice.label).tag(choice)
            }
        }
        .help(
            "From scratch opens the blank draft interface. A saved design "
                + "opens the new-studies table, prefilled with that design's "
                + "task file and pins, instruments, sampling policy and judges "
                + "— so the only thing left to decide is the casting.")
        if panel.templates.isEmpty {
            HStack(spacing: 6) {
                Text("No saved designs yet.")
                Button("Open Templates") { openTemplates() }
                    .buttonStyle(.link)
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
    }

    /// "New Study" does one of the two things the design picker selected.
    private func startNewStudy(panel: ExperimentPanel) {
        if let design = panel.newStudyDesign.designName {
            openInstantiation(design, panel: panel)
        } else {
            panel.newStudy()
        }
    }

    /// Opens the new-studies table on a design, mirroring the panel's current
    /// Remote options so the totals line counts the jobs this batch will
    /// really create.
    private func openInstantiation(
        _ name: String, panel: ExperimentPanel, permuting: [SeatOccupant] = []
    ) {
        panel.clearFormError(.template)
        templateSheet = TemplateInstantiationRequest(
            templateName: name,
            shardsPerStudy: max(1, panel.remoteParallelJobs),
            jobNoun: panel.remoteExecutor == "slurm" ? "Slurm jobs" : "jobs",
            canSubmit: panel.canSubmitBundles,
            permuting: permuting)
    }

    /// One-shot handoff for "this draft was just created — name it now".
    /// Consumed on change (the in-tab New Study path) and on appear (Templates'
    /// New Template, which creates the draft in another section).
    private func consumeRenameInvitation(panel: ExperimentPanel) {
        guard let invited = panel.renameInvitation,
            let manifest = panel.experiments.first(where: { $0.name == invited })
        else { return }
        panel.renameInvitation = nil
        openRename(manifest)
    }

    /// One-shot handoff from the Templates tab (and from any in-tab path that
    /// resolves a design): open the flow on it, then clear the flag.
    private func consumeInstantiationInvitation(panel: ExperimentPanel) {
        guard let invited = panel.templateInstantiationInvitation else { return }
        panel.templateInstantiationInvitation = nil
        panel.newStudyDesign = .design(invited.design)
        openInstantiation(
            invited.design, panel: panel, permuting: invited.permuting)
    }

    private func deleteDraftMessage(_ manifest: ExperimentManifest) -> String {
        var message =
            "Moves experiments/\(manifest.name)/ (the draft manifest and its "
            + "pinned snapshots) to a .trash-<timestamp> sibling inside "
            + "experiments/. Nothing is destructively removed — recover it "
            + "from there if needed. Run artifacts under runs/ are untouched."
        if deleteDraftRunCount > 0 {
            // A draft CAN have runs (only freeze is one-way, not running), and
            // those runs stamp this study's name — deleting the manifest
            // leaves them unresolvable from the app.
            message += " \(deleteDraftRunCount) run(s) already stamp this "
                + "draft's name; they stay in runs/ but will no longer resolve "
                + "back to a study here."
        }
        return message
    }

    private static let duplicateHelp =
        "copies EVERYTHING, agents included — the only way to iterate on a "
        + "frozen study. (Saving a study as a DESIGN, in Templates, does the "
        + "opposite: it strips the agents and re-derives the derived pins.)"

    private static let deleteStudyHelp =
        "moves the draft's directory to a .trash-<timestamp> sibling under "
        + "experiments/ — never a destructive delete. Frozen and completed "
        + "studies cannot be deleted at all: their runs stamp them."

    /// Opens Rename on `manifest`, selecting it first so the panel action
    /// (which always targets the selection) cannot act on another study.
    private func openRename(_ manifest: ExperimentManifest) {
        panel.clearFormError(.rename)
        panel.selectedName = manifest.name
        renameSheet = RenameStudySheet(
            name: manifest.name,
            status: manifest.status,
            label: panel.displayLabels[manifest.name] ?? "",
            runsStamped: ExperimentStore.runsStamped(experimentName: manifest.name))
    }

    /// Change key for the residency preflight: selection + active substrate.
    private var residencyTaskKey: String {
        let name = panel.selectedName ?? ""
        let substrate = service.cluster.substrateLabel
        return "\(name)|\(substrate)|\(panel.isServerWorkspace)"
    }

    // Long strings live outside the body — interpolating them inline blows
    // the SwiftUI type-checker budget (same fix as ChatView.provenanceLine).
    private static let freezeHelp =
        "ONE-WAY: verifies every pinned input, requires a pinned model revision, "
        + "and stamps the content hash and git commit. Agent-comparison "
        + "studies pin each agent (variant artifact) by artifact hash; concept-vector "
        + "studies also require a matching validate run. Settings must be "
        + "frozen before behavior is measured — iterate afterwards by duplicating. "
        + "CLI: freeze --force skips validation gates"

    private static let remoteFreezeHelp =
        "ONE-WAY, executed by the ACTIVE SERVER: the server verifies every "
        + "pinned input, evaluates the freeze gates against ITS OWN substrate's "
        + "validation evidence (validation evidence counts on the substrate "
        + "that freezes — validate on the server first), stamps frozenBy: "
        + "\"server\" + the content hash, and exports preregistration.md. "
        + "Freeze stamps the server-resident copy; before submitting, the app "
        + "verifies that copy IS the manifest shown here (field-level "
        + "comparison, volatile freeze stamps excluded) and refuses on a "
        + "mismatch. On a paired workspace the frozen manifest appears here "
        + "immediately. No force option in the app — forcing requires the CLI"

    private static let runHelp =
        "executes the study in-app: verifies pins, loads the model at the "
        + "pinned revision, runs the baseline plus pinned agents for every "
        + "task prompt, and writes generations, metrics.csv, report.json, "
        + "and the manifest snapshot into a new immutable runs/ directory. "
        + "Legacy concept-vector studies re-derive their vectors before running. "
        + "Currently requires Temperature = 0 because mlx-swift-lm does not "
        + "expose a per-run seed for reproducible sampling"

    private static let validateHelp =
        "creates the validation evidence required by Freeze Study: verifies "
        + "the pinned inputs, loads the model at its pinned revision (or pins "
        + "the local cached revision for drafts), re-derives concept vectors, "
        + "runs never-named validation scenarios when present, writes the "
        + "cross-concept cosine matrix, and stores a manifest snapshot in a "
        + "new runs/ validate directory"

    private static let extractHelp =
        "re-derives the pinned concepts' steering vectors from the frozen "
        + "recipe (stimulus hashes + extraction options) into a new immutable "
        + "runs/ extract directory — the CLI 'experiment extract' verb. "
        + "Deterministic re-derivation, so drafts AND frozen studies may run it"

    private static let outcomeModeHelp =
        "dispatches real instruments — not a note. Generated choice: the "
        + "run samples text and parses answers from the prose. Answer-token "
        + "probability: no endpoint prose — deterministic logprob records "
        + "over the declared options. Both: the two side by side. Written "
        + "to `outcomeInstruments`; the run's records and Results views "
        + "differ accordingly"

    private static let remoteOptionsCaption =
        "the unified Run button packages this study as a hash-pinned bundle "
        + "and submits it with these options — the portable path that works "
        + "whether or not the study exists in the server's own workspace."

    private static let unifiedRemoteRunHelp =
        "packages this study as a hash-pinned bundle, uploads it, and submits "
        + "the selected verb as a durable server job. The server preflights "
        + "the submission (memory fit, walltime, quota) — warnings show "
        + "inline; a failing verdict stops the submission unless explicitly "
        + "forced."

    private static let importEvidenceCaption =
        "Import Evidence verifies an evidence bundle's hashes and lands it "
        + "under runs/ as an immutable imported run."

    private static let recentJobsCaption =
        "jobs persist on the server — reconnect any time from Compute or here"

    private static func validateCaption(variantsPresent: Bool) -> String {
        "derives vectors from the pinned recipe and writes scope-hashed "
            + "validation evidence (held-out probe accuracy, cross-concept geometry"
            + (variantsPresent ? ", capability battery per agent" : "")
            + "). Optional while drafting — freeze REQUIRES matching evidence."
    }

    private func serverRunCaption(_ verb: String) -> String {
        "\(verb) runs on \(service.cluster.substrateLabel) as a durable job — "
            + "reconnect from Compute"
    }

    private static let studyDtypeHelp =
        "the numeric precision the study model runs in on the compute "
        + "substrate. A PIN, not a hint: greedy decoding is not "
        + "precision-proof — at a near-tie between two tokens, bf16 and fp16 "
        + "round differently and the continuation diverges — so two runs at "
        + "different precisions are not the same measurement. Leave it on "
        + "\"device default\" to let the substrate choose, which is what "
        + "every study did before this pin existed. Honored by the server; "
        + "the Mac validates it here so a bad value cannot reach the cluster."

    /// The study's precision pin. Shown beside the baseline model because it
    /// qualifies that model: same repo at two precisions is two instruments.
    @ViewBuilder
    private func studyDtypePicker(
        manifest: ExperimentManifest, selection: Binding<String>
    ) -> some View {
        let vocabulary = ExperimentStore.judgeDtypeVocabulary
        let current = selection.wrappedValue
        Picker("Precision", selection: selection) {
            Text("device default").tag("")
            ForEach(vocabulary, id: \.self) { name in
                Text(name).tag(name)
            }
            // A pin outside the vocabulary (hand-edited JSON) stays visible
            // but unpickable — the same rule the model pickers use, so a
            // manifest's real state is never silently rewritten by opening
            // it in the editor.
            if !current.isEmpty, !vocabulary.contains(current) {
                Text("\(current) (not loadable)")
                    .tag(current)
                    .selectionDisabled()
            }
        }
        .disabled(manifest.status != .draft)
        .help(Self.studyDtypeHelp)
    }

    // Judge-row helpers (localJudgeModelPicker / openRouterJudgeFields)
    // moved to JudgingSectionView.swift with the unified judging section
    // (2026-07-21).

    // MARK: Direct concept attach (App gap A8)

    /// One-step concept attachment in Studies: attached concepts with their
    /// pin status (stimulus hash, method, reading position, three-state
    /// validation pin) and a detach action, plus a draft-only picker —
    /// concept, method, reading position, grand-mean corpus — that writes
    /// through `ExperimentStore.attachConcept` exactly like the CLI attach.
    @ViewBuilder
    private func conceptsSection(
        manifest: ExperimentManifest, panel: ExperimentPanel
    ) -> some View {
        @Bindable var panel = panel
        Section {
            if manifest.concepts.isEmpty {
                Text(
                    "No concepts pinned. Injection conditions steer along "
                        + "pinned concepts; attaching pins the stimulus files "
                        + "by hash (the recipe, not vector bytes).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            ForEach(manifest.concepts, id: \.name) { ref in
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(ref.name)
                            .font(.callout)
                        Text(ExperimentPanel.conceptPinStatusLine(ref))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                    Spacer()
                    if manifest.status == .draft {
                        Button {
                            panel.detachConcept(ref.name)
                        } label: {
                            Image(systemName: "trash")
                        }
                        .buttonStyle(.plain)
                        .help(
                            "detach '\(ref.name)' from this draft — refused "
                                + "while a declaration still names it")
                    }
                }
            }
            if manifest.status == .draft {
                attachPickerRows(panel: panel)
            }
        } header: {
            InfoSectionHeader(
                title: "Build & Validate Concept Vectors",
                text: StudyInfo.conceptVectors)
        }
    }

    @ViewBuilder
    private func attachPickerRows(panel: ExperimentPanel) -> some View {
        @Bindable var panel = panel
        let sources = panel.attachableConceptSources
        if sources.isEmpty {
            Text(
                "no unattached concepts on disk — author a paired stimulus set "
                    + "under prompts/concepts/<name>/ or grand-mean stories "
                    + "under prompts/emotions/<name>/stories.jsonl")
                .font(.caption2)
                .foregroundStyle(.secondary)
        } else {
            let selectedSource = sources.first { $0.name == panel.attachConceptName }
            Picker("Attach Concept…", selection: $panel.attachConceptName) {
                Text("select…").tag("")
                ForEach(sources, id: \.name) { source in
                    Text(source.pickerLabel).tag(source.name)
                }
            }
            .help(
                "concepts found on disk that this study has not pinned yet — "
                    + "labels say what each can support (paired stimulus set, "
                    + "grand-mean stories)")
            .onChange(of: panel.attachConceptName) {
                // Snap the method to something the chosen concept's data can
                // actually support.
                if let source = sources.first(where: { $0.name == panel.attachConceptName }),
                    !source.supportedMethods.contains(panel.attachMethod),
                    let first = source.supportedMethods.first
                {
                    panel.attachMethod = first
                }
            }
            HStack(spacing: 8) {
                Picker("", selection: $panel.attachMethod) {
                    ForEach(
                        selectedSource?.supportedMethods
                            ?? ExtractionMethod.allCases.filter(\.isRecipeMethod),
                        id: \.self
                    ) { method in
                        Text(method.label).tag(method)
                    }
                }
                .frame(maxWidth: 210)
                .help(
                    "extraction method pinned into the recipe — paired methods "
                        + "read positive/negative stimuli; grand mean reads the "
                        + "multi-concept story corpus")
                ReadingPositionField(
                    choice: $panel.attachReadingPositionChoice,
                    parameter: $panel.attachReadingPositionParameter,
                    defaultCaption: "method default",
                    help:
                        "WHERE the residual stream is read, pinned into the "
                            + "recipe. 'method default' declares nothing and "
                            + "keeps the method's own position (last token for "
                            + "paired; token 50 for grand mean and designated "
                            + "reference). The content-side roles only exist "
                            + "inside a rendered turn, so they need the chat "
                            + "template beside this")
                Button("Attach") { panel.attachConceptFromPicker() }
                    .disabled(panel.attachConceptName.isEmpty)
                    .help(
                        "pins the concept at its CURRENT stimulus hash plus the "
                            + "measurement-side validation pin — same write as "
                            + "'steerlab-cli experiment attach'")
            }
            HStack(spacing: 8) {
                Text("rendering")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ExtractionRenderingField(
                    choice: $panel.attachRendering,
                    help:
                        "HOW each stimulus reaches the model. 'raw' is the "
                            + "legacy rendering and declares nothing — the bare "
                            + "string through the tokenizer. 'chat template' "
                            + "renders it the way a measured generation does; "
                            + "the two produce DIFFERENT directions, which is "
                            + "why the choice is pinned rather than inferred")
                Spacer(minLength: 0)
            }
            if panel.attachMethod == .designatedReference {
                Picker("reference", selection: $panel.attachReferenceName) {
                    Text("select reference…").tag("")
                    ForEach(sources.filter(\.hasStories), id: \.name) { source in
                        Text(source.name).tag(source.name)
                    }
                }
                .font(.caption)
                .help(
                    "the DESIGNATED reference corpus: the vector is "
                        + "mean(concept stories) − mean(reference stories), both "
                        + "pooled from token 50 — the reference pins into the "
                        + "recipe beside the concept")
            }
            if panel.attachMethod == .emotionGrandMean {
                TextField(
                    "extra corpus members (comma-separated; targets are always members)",
                    text: $panel.attachCorpusText)
                    .font(.caption)
                    .help(
                        "grand-mean vectors are concept mean − corpus grand mean, "
                            + "so the pinned population is part of the recipe — "
                            + "name extra prompts/emotions/ concepts to widen it")
            }
            Text(
                "attach pins stimulus bytes by hash + the concept's "
                    + "validation.jsonl (or its absence) — the firewall's "
                    + "measurement-side pins, identical to the CLI attach")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: Native condition editor (App gap A4)

    /// Conditions/controls authoring in Studies: list + remove, single-slot
    /// vector conditions (negative α legal), one-click sign control and
    /// matched-norm random control per condition, explicit baseline, and the
    /// Step-5 control-matrix scaffold. Draft-only; every write goes through
    /// `ExperimentStore` helpers via the panel.
    @ViewBuilder
    private func conditionsSection(
        manifest: ExperimentManifest, panel: ExperimentPanel
    ) -> some View {
        @Bindable var panel = panel
        Section {
            if manifest.conditions.isEmpty {
                Text(
                    "No steering conditions yet. Baseline is implied at run "
                        + "time; a defensible matrix adds treatments plus "
                        + "direction (−α) and matched-norm random controls.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            ForEach(manifest.conditions, id: \.name) { condition in
                conditionRow(condition, manifest: manifest, panel: panel)
            }

            if manifest.status == .draft {
                if panel.conditionConceptOptions.isEmpty {
                    Text(
                        "attach a concept first — conditions reference pinned "
                            + "concepts (Attach Concept… in the Concepts section "
                            + "above, Agents → New Agent, or capture from "
                            + "Playground)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    // One labeled row per input (2026-07-20 researcher
                    // round: the old single HStack scrunched and
                    // hyphenated its labels) — see AddConditionEditor.
                    AddConditionEditor(panel: panel)
                }
            }
        } header: {
            InfoSectionHeader(
                title: "Injection Conditions & Controls",
                text: StudyInfo.injectionConditions)
        }
    }

    @ViewBuilder
    private func conditionRow(
        _ condition: ExperimentManifest.Condition,
        manifest: ExperimentManifest,
        panel: ExperimentPanel
    ) -> some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(condition.name)
                        .font(.callout)
                    if condition.controlType == "randomMatchedNorm" {
                        Text("random-direction control")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 1)
                            .background(.orange.opacity(0.15), in: Capsule())
                            .help(
                                "matched-norm random: same layers and strength, "
                                    + "deterministic random direction — separates "
                                    + "the concept's direction from mere "
                                    + "perturbation energy")
                    } else if condition.slots.contains(where: { $0.alpha < 0 }) {
                        Text("direction control")
                            .font(.caption2.weight(.semibold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 1)
                            .background(.blue.opacity(0.12), in: Capsule())
                    }
                }
                Text(conditionSummary(condition))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if manifest.status == .draft {
                if !condition.slots.isEmpty, condition.controlType == nil {
                    Button("+ sign control") { panel.addSignControl(for: condition.name) }
                        .controlSize(.small)
                        .help("adds '\(condition.name)-neg' with every α negated")
                    Button("+ random control") {
                        panel.addMatchedNormRandomControl(for: condition.name)
                    }
                    .controlSize(.small)
                    .help(
                        "adds '\(condition.name)-random' — a random-direction "
                            + "control at the same strength ('matched-norm "
                            + "random', controlType randomMatchedNorm): same "
                            + "layers/α, deterministic random direction of "
                            + "matched norm")
                }
                Button {
                    panel.removeCondition(condition.name)
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.plain)
                .help("remove this condition from the draft")
            }
        }
    }

    // MARK: Former Science Manifest fields, in their real homes

    // The Science Manifest section is DISSOLVED (2026-07-19 second pass):
    // its fields were five different kinds of thing under one mystifying
    // title. Phase → Study Setup (funnel provenance); sampling → Study
    // Setup generation settings; case family + option-length
    // acknowledgment → Evaluation (they shape measurement/analysis);
    // human-baseline pin → Data & Prompts; promotion rule → the Pipeline
    // section (concept studies only).

    /// Sampling policy lives WITH the other generation settings — samples
    /// per item and seed policy change how the run generates, exactly like
    /// temperature (they were never inert "science notes").
    @ViewBuilder
    private func samplingControls(
        manifest: ExperimentManifest, panel: ExperimentPanel
    ) -> some View {
        @Bindable var panel = panel
        let isDraft = manifest.status == .draft
        let isPanel = panel.studyKind == .multiAgent
        LabeledContent(isPanel ? "Play-throughs (transcripts)" : "Samples per item") {
            HStack(spacing: 6) {
                TextField(
                    "", value: $panel.samplesPerItemField,
                    format: .number.grouping(.never)
                )
                .frame(width: 56)
                .multilineTextAlignment(.leading)
                .disabled(!isDraft)
                InfoButton(text: StudyInfo.sampling)
            }
        }
        .help(
            isPanel
            ? "independent play-throughs of the panel per condition. This is the "
                + "unit the statistics treat as one observation, and the unit a "
                + "sharded submission splits across GPUs. Needs a temperature "
                + "above 0 — at 0 every play-through is identical."
            : "stochastic samples per (condition, prompt); 1 = deterministic "
                + "greedy")
        // Two real choices (2026-07-21, issue 1): absent and an explicit
        // 'manifestSeeds' behave identically on both engines (each
        // resolves absent to the fixed list), so the picker offers TWO
        // options. The explicit legacy tag renders only when a manifest
        // already declares it — silently normalizing it on load would
        // make an unchanged Save rewrite manifest bytes (a hash change
        // that trips the epoch guard on existing runs), exactly the kind
        // of quiet drift the firewall exists to prevent.
        HStack(spacing: 6) {
            Picker("Seed policy", selection: $panel.seedPolicyField) {
                Text("Fixed seed list (default)").tag("")
                if panel.seedPolicyField == "manifestSeeds" {
                    Text("Fixed seed list (declared — same behavior)")
                        .tag("manifestSeeds")
                }
                Text("Derived per record — recommended for sampled runs")
                    .tag("derivedSHA256")
            }
            .disabled(!isDraft)
            .help(
                "how stochastic server runs seed each generated record — "
                    + "the local Mac engine is always greedy and ignores "
                    + "this. The ⓘ explains both policies")
            InfoButton(text: StudyInfo.seedPolicy)
        }
        // The list the fixed-list policy indexes into (audit 2026-08-01:
        // the policy had a picker; the list had no editor). Hidden under
        // the derived policy, which never reads it.
        if panel.seedPolicyField != "derivedSHA256" {
            SeedsListControls(manifest: manifest, panel: panel)
        }
        // Gentle advisory (never a blocker): a stochastic design with the
        // fixed list loses per-record reproducibility; samples > 1
        // additionally requires the derived policy at verify.
        if panel.runTemperature > 0 || panel.samplesPerItemField > 1,
            panel.seedPolicyField != "derivedSHA256"
        {
            Label(
                "this design is stochastic (temperature > 0 or several "
                    + "samples per item) — 'Derived per record' gives every "
                    + "(condition, prompt, sample) its own reproducible "
                    + "seed; the fixed list suits single-sample runs",
                systemImage: "info.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        // The server-only-stochastic rule, surfaced exactly like the
        // temperature rule: local target + samplesPerItem > 1 explains
        // itself inline instead of failing later.
        if panel.samplesPerItemField > 1, !panel.isServerWorkspace {
            Label(
                "samplesPerItem > 1 is a stochastic design — it runs on the "
                    + "Python server, which seeds PyTorch per record; the "
                    + "local MLX generator has no per-run sampling seed, so "
                    + "local runs stay greedy (temperature 0, 1 sample)",
                systemImage: "die.face.5")
                .font(.caption)
                .foregroundStyle(.orange)
        }
        // Study-owned sampling (2026-07-21): with saved agents in the
        // design, say explicitly that these knobs govern every condition.
        if !manifest.variantConditions.isEmpty {
            Label(
                "the study's sampling policy governs the baseline AND every "
                    + "saved agent — an agent's Playground temperature is "
                    + "not used in measured runs",
                systemImage: "info.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    /// Analysis-shaping declarations live WITH the evaluation settings:
    /// the case family selects the endpoint parser and the option-length
    /// acknowledgment gates the answer-token instrument — neither is an
    /// inert note.
    @ViewBuilder
    private func analysisSettings(
        manifest: ExperimentManifest, panel: ExperimentPanel
    ) -> some View {
        @Bindable var panel = panel
        let isDraft = manifest.status == .draft
        // Editable with suggestions (the manifest accepts any string —
        // `setCaseFamily` does not validate against the known list) plus
        // the honest ⓘ: only 'sentencing' has a parser today.
        CaseFamilyField(manifest: manifest, panel: panel)

        // Declared numeric parser + exclusion rules — controls live in
        // their own files (NumericParserPickerView / ExclusionRulesEditor-
        // View); these are the two wiring lines.
        NumericParserControls(manifest: manifest, panel: panel)
        ExclusionRulesEditor(manifest: manifest, panel: panel)

        // Answer-token instrument concern — model-output studies only
        // (multi-agent studies have no option scoring).
        if manifest.studyKind == .modelOutput {
            HStack(spacing: 6) {
                Toggle(
                    "Acknowledge unequal option lengths",
                    isOn: $panel.acknowledgeUnequalOptionLengthsField
                )
                .disabled(!isDraft)
                .help(
                    "opt-in: scored answer options that tokenize to unequal lengths "
                        + "bias joint logprobs toward shorter options — the run loop "
                        + "refuses unequal option sets unless this is acknowledged")
                InfoButton(text: StudyInfo.optionLengths)
            }
        }
    }

    /// ONE Conditions section, content by study type: the ARMS of the
    /// study. Agents for a comparison; the scenario for multi-agent; the
    /// perturbation policy (and the conditions it expands into) for a
    /// confirmation. Concept studies see their arms here too — the
    /// concept-derivation machinery follows in its own sections.
    @ViewBuilder
    private func studyArmsSection(
        manifest: ExperimentManifest, panel: ExperimentPanel
    ) -> some View {
        @Bindable var panel = panel
        let isDraft = manifest.status == .draft
        Section {
            if panel.studyKind == .multiAgent {
                // The picker itself now lives in Study Setup, beside Save —
                // pinning happens on save, and having the two in different
                // sections meant selecting a panel here looked like it had
                // taken effect when nothing had been written yet.
                Text(panel.selectedMultiAgentScenarioID == nil
                    ? "No scenario selected. Choose one in Study Setup, cast "
                        + "its seats below, then save to pin it."
                    : "Scenario selected in Study Setup; its seats are cast in "
                        + "Seats below. Arms:")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(panel.multiAgentIncludeBaseline
                    ? "Two arms: the configured panel, and a baseline of the "
                        + "same panel with every intervention stripped."
                    : "One arm only: the configured panel. No baseline to "
                        + "compare against.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                if isDraft {
                    HStack {
                        Picker("Add agent", selection: $panel.selectedVariantToAddID) {
                            Text("select…").tag(String?.none)
                            ForEach(panel.availableVariantsForStudy) { variant in
                                Text(variant.artifact.name).tag(String?.some(variant.id))
                            }
                        }
                        Button("Add") { panel.addVariantCondition() }
                            .disabled(panel.selectedVariantToAddID == nil)
                    }
                    .help("add a saved agent that uses the selected baseline model")
                    if !manifest.concepts.isEmpty {
                        Menu("Add sweep-created agent…") {
                            ForEach(manifest.concepts, id: \.name) { concept in
                                Button(concept.name) {
                                    panel.addForwardReferencedCondition(
                                        concept: concept.name)
                                }
                            }
                        }
                        .help(
                            "an agent that does not exist YET: this study's "
                                + "own sweep creates it at run time — the "
                                + "sweep selects the concept's best "
                                + "layer×strength cell, promote mints the "
                                + "agent, and the chain runs it as this arm, "
                                + "pinning the resolution as run evidence "
                                + "(forward-resolutions.json). Freezable "
                                + "before the agent exists")
                    }
                }
                // Confirmation authoring is the CONCEPT study's confirm
                // phase (2026-07-19 fold-in): the policy editor appears
                // when the funnel phase says confirm — or a policy is
                // already attached.
                if isDraft, panel.studyFocus == .conceptStudy,
                    panel.phaseField == "confirm"
                        || manifest.perturbationPolicy != nil
                {
                    confirmationControls(panel: panel)
                }
                if let policy = manifest.perturbationPolicy {
                    perturbationPolicySummary(policy, panel: panel)
                }
                if panel.studyFocus == .agentComparison,
                    manifest.concepts.isEmpty
                {
                    // Type parity note: comparisons consume EXISTING
                    // agents; deriving new ones (sweep-created agents,
                    // forward references) is Concept-study machinery.
                    Text("Comparisons run agents that already exist (built "
                        + "in Agents, or by a Concept study's sweep). To "
                        + "DERIVE new agents from concept data, use a "
                        + "Concept study.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if manifest.variantConditions.isEmpty {
                    Text("Baseline only. Add agents to compare conditions "
                        + "(adding saves immediately).")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(manifest.variantConditions, id: \.name) { variant in
                        attachedAgentRow(variant, isDraft: isDraft, panel: panel)
                    }
                }
            }
        } header: {
            InfoSectionHeader(
                title: "Conditions", text: StudyInfo.conditionsArms)
        }
    }

    // MARK: Seats

    /// WHO sits in each seat of the study's scenario — the casting.
    ///
    /// It exists here because a scenario chosen in Study Setup is SEMANTIC: it
    /// declares roles, turns and materials and binds no model to any seat, so a
    /// study that only picked one refuses at run start (deliberately — see
    /// `PanelComposition`). Casting used to be reachable only through a
    /// design's instantiation table, which meant a directly-authored panel
    /// study had no way to become runnable at all.
    ///
    /// Every rule rendered here lives in `SeatCasting` / `ExperimentPanel`
    /// (ExperimentKit, unit-tested). This view decides nothing: it reads the
    /// state, binds the pickers, and calls the two actions.
    @ViewBuilder
    private func seatsSection(
        manifest: ExperimentManifest, panel: ExperimentPanel
    ) -> some View {
        if let casting = panel.seatCasting {
            let refusal = panel.seatCastingRefusal(casting)
            Section("Seats") {
                if casting.seats.isEmpty {
                    Text("this scenario declares no seats — add roles to it in "
                        + "the Panels editor first")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                ForEach(casting.seats) { seat in
                    if casting.isEditable, manifest.status == .draft {
                        Picker(
                            seat.name,
                            selection: seatBinding(seat: seat.id, panel: panel)
                        ) {
                            Text("baseline").tag(String?.none)
                            ForEach(panel.availableAgentsForSeats) { agent in
                                Text(agent.artifact.name)
                                    .tag(String?.some(agent.id))
                            }
                        }
                        .help(Self.seatPickerHelp)
                    } else {
                        LabeledContent(
                            seat.name,
                            value: casting.occupants[seat.id]?.label ?? "baseline")
                            .font(.caption)
                    }
                }
                if casting.isEditable, panel.availableAgentsForSeats.isEmpty {
                    Text("no saved agents use this study's base model "
                        + "(\(manifest.modelID)) — every seat can only be "
                        + "baseline until one exists (build one in Agents)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                ForEach(casting.advisories, id: \.self) { advisory in
                    Text(advisory)
                        .font(.caption2)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
                if casting.isEditable {
                    Text(Self.castingStateLine(casting))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 8) {
                        Button("Save Casting") { panel.saveSeatCasting() }
                            .disabled(refusal != nil || casting.seats.isEmpty)
                            .help(refusal ?? Self.saveCastingHelp)
                        Button("Create permuted siblings…") {
                            panel.startPermutedSiblings()
                        }
                        .disabled(casting.form != .cast)
                        .help(
                            casting.form == .cast
                                ? Self.permutedSiblingsHelp
                                : "save this study's casting first — permuted "
                                    + "siblings re-seat the cast it is running")
                    }
                    if let refusal {
                        Text(refusal)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
    }

    private func seatBinding(
        seat: String, panel: ExperimentPanel
    ) -> Binding<String?> {
        Binding(
            get: { panel.seatAgentID(for: seat) },
            set: { panel.setSeatAgent($0, seat: seat) })
    }

    /// What the study is currently pinning, in one line — the difference
    /// between "this is what will run" and "this is what will run once you
    /// save" is the whole point of the section.
    private static func castingStateLine(_ casting: SeatCasting.State) -> String {
        switch casting.form {
        case .uncast:
            return "not cast yet: this scenario binds no model to any seat, so "
                + "the study refuses at run start until Save Casting compiles "
                + "it. Save Study Setup does the same compile."
        case .cast:
            return "cast: the study pins a compiled copy of this scenario with "
                + "every seat bound. Saving again recompiles it at the study's "
                + "current model and sampling settings."
        case .legacyBound:
            return ""
        }
    }

    private static let seatPickerHelp =
        "the agent that speaks for this role. 'baseline' is the study's own "
        + "model with no intervention — a real condition (the control "
        + "composition), not an empty seat. Only agents built on this study's "
        + "base model are eligible."

    private static let saveCastingHelp =
        "compiles this scenario plus the casting into a bound scenario under "
        + "prompts/panels/compiled/ and pins it as the study's scenario — the "
        + "same write a design's instantiation table performs. The scenario "
        + "itself is not modified."

    private static let permutedSiblingsHelp =
        "one study runs ONE casting (a panel's arms are the fixed "
        + "baseline/configured pair), so re-seating the same agents means "
        + "sibling studies. This saves the study as a design and opens the "
        + "new-studies table holding every distinct re-seating of its current "
        + "cast — swapping two identical occupants is the same panel, so those "
        + "are deduped rather than run twice."

    /// Scenarios that carry their own seat bindings are marked in the picker,
    /// matching the Panels editor: only one of two same-named entries can be
    /// cast from a study.
    private static func scenarioMenuLabel(_ record: MultiAgentScenarioRecord) -> String {
        PanelAuthoring.carriesBindings(record.scenario)
            ? "\(record.label) — bound (legacy)"
            : record.label
    }

    /// The task-prompts CONTENT editor, rendered INSIDE Data & Prompts
    /// under the row that tracks the same file (2026-07-19: the standalone
    /// "Task Prompt Contents" section read as a second, mysterious copy).
    @ViewBuilder
    private func taskPromptsEditorContent(
        manifest: ExperimentManifest, panel: ExperimentPanel
    ) -> some View {
        @Bindable var panel = panel
        HStack(spacing: 6) {
            TextField("task prompts JSONL", text: $panel.taskPromptsFile)
                .disabled(manifest.status != .draft)
                .help(
                    "relative path to a {\"text\": ...}-per-line JSONL file. "
                        + "Save & Pin Prompts pins its current hash into the manifest")
            // Phase 3 item 12: the no-typing route — pick the file, the
            // path lands workspace-relative, and the pin is made on
            // selection through the same validating pin path.
            WorkspacePathChooseButton(
                message: "Choose this study's task-prompts JSONL "
                    + "(workspace files only — the path pins on selection)",
                allowedTypes: WorkspaceFileChooser.jsonlTypes,
                startingSubdirectory: "prompts/tasks",
                onChoose: { panel.pinChosenTaskPromptsFile($0) },
                onProblem: { panel.note($0, severity: .error) })
                .disabled(manifest.status != .draft)
        }
        HStack {
            Button("Load Prompts") { panel.loadTaskPrompts() }
                .help("read the JSONL file into the editor below")
            Button("Save & Pin Prompts") { panel.saveTaskPrompts() }
                .disabled(manifest.status != .draft)
                .help(StudyInfo.taskPromptsSavePin)
            Button("Import JSONL…") {
                importJSONLText = ""
                showImportJSONL = true
            }
            .disabled(manifest.status != .draft)
            .help(
                "paste or choose a raw JSONL records file (full "
                    + "records — options/target preserved, required "
                    + "by the answer-token instrument). Parsed with "
                    + "a preview; on import the file lands at the "
                    + "study's task-prompts destination, becomes "
                    + "this study's prompts file, and its hash is "
                    + "pinned — one action from paste to pinned")
            // Phase 3 item 13: spreadsheets (JSON array / CSV) enter
            // through a column-mapping sheet instead of hand-written
            // JSONL; same destination, same pin.
            TabularImportButton(
                target: .taskPrompts, panel: panel,
                disabled: manifest.status != .draft)
            // Phase 4 items 20–21: factorial/counterbalancing generation —
            // authoring-time data generation; the emitted file pins like
            // any hand-authored prompts (sheet lives in its own file).
            FactorialDesignButton(
                panel: panel, disabled: manifest.status != .draft)
        }
        // What Save & Pin actually DOES, visible — not hover-only
        // (2026-07-20 researcher round, item 2a).
        Text(
            "Save & Pin writes this editor's text back to the file named "
                + "above, then re-pins the file's new hash into the study.")
            .font(.caption2)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        TextEditor(text: $panel.taskPromptsText)
            .font(.system(.caption, design: .monospaced))
            .frame(minHeight: 220)
            .disabled(manifest.status != .draft)
            .help(
                "write one prompt per block. Long multi-paragraph prompts are fine; "
                    + "separate prompts with a line containing only ---")
        // Paste-detection guard: content in the PLAIN prompt editor that
        // looks like JSONL records would be saved as literal prompt text
        // (options/target lost). Offer the Import JSONL path — never
        // silently reinterpret.
        if manifest.status == .draft,
            TaskPromptsImport.looksLikeJSONL(panel.taskPromptsText)
        {
            HStack(spacing: 8) {
                Label(
                    "this looks like JSONL records, not prompt text — "
                        + "Save & Pin would store the JSON itself as "
                        + "prompts and discard options/target",
                    systemImage: "exclamationmark.triangle")
                    .font(.caption2)
                    .foregroundStyle(.orange)
                Button("Import as JSONL…") {
                    importJSONLText = panel.taskPromptsText
                    showImportJSONL = true
                }
                .controlSize(.small)
            }
        }
        if let instrumentSummary = panel.taskPromptsInstrumentSummary {
            Label(instrumentSummary, systemImage: "list.bullet.rectangle")
                .font(.caption2)
                .foregroundStyle(.orange)
                .help(
                    "these items carry per-item instrument fields (options, "
                        + "target, …) that the text editor does not show — "
                        + "they are preserved byte-faithfully on save")
        }
        if let promptStatus = panel.taskPromptsStatus {
            Text(promptStatus)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    /// Split out of the Evaluation section body (type parity cleanup —
    /// the outcome instruments score per-prompt generations, so they
    /// render for model-output studies only; also relieves the
    /// type-checker).
    @ViewBuilder
    private func instrumentActivationControls(
        manifest: ExperimentManifest, panel: ExperimentPanel
    ) -> some View {
        @Bindable var panel = panel
        Group {
                    // P1 — explicit instrument activation: what the DATA
                    // supports (detected), what is ENABLED (declared in the
                    // manifest, provenance), and the honest warning when the
                    // two disagree. Never auto-enabled.
                    if let detected = panel.detectedCapabilitiesLine {
                        Label(detected, systemImage: "list.bullet.rectangle")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .help(
                                "items in the pinned prompt set carrying per-item "
                                    + "`options` — a capability of the data, not an "
                                    + "enabled measurement")
                    }
                    HStack(spacing: 6) {
                        Picker(
                            "Outcome mode",
                            selection: Binding(
                                get: { panel.outcomeMode },
                                set: { panel.setOutcomeMode($0) })
                        ) {
                            if panel.outcomeMode == .notDeclared {
                                Text(InstrumentActivation.OutcomeMode.notDeclared.label)
                                    .tag(InstrumentActivation.OutcomeMode.notDeclared)
                            }
                            Text(InstrumentActivation.OutcomeMode.generatedChoice.label)
                                .tag(InstrumentActivation.OutcomeMode.generatedChoice)
                            Text(InstrumentActivation.OutcomeMode.answerTokenProbability.label)
                                .tag(InstrumentActivation.OutcomeMode.answerTokenProbability)
                            Text(InstrumentActivation.OutcomeMode.both.label)
                                .tag(InstrumentActivation.OutcomeMode.both)
                        }
                        .disabled(manifest.status != .draft)
                        .help(Self.outcomeModeHelp)
                        InfoButton(text: StudyInfo.evaluationOutcome)
                    }
                    if let warning = panel.instrumentActivationWarning {
                        Label(warning, systemImage: "exclamationmark.triangle")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .textSelection(.enabled)
                    }

                    // F3 — auxiliary instruments (not owned by the picker)
                    // get their own rows with the sampling implication
                    // stated, plus a real remove affordance (draft-only,
                    // written through the store like every instrument
                    // edit).
                    ForEach(panel.auxiliaryOutcomeInstruments, id: \.self) { id in
                        HStack(alignment: .firstTextBaseline) {
                            Label(
                                InstrumentActivation.auxiliaryDescription(id),
                                systemImage: "waveform.and.magnifyingglass")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                            Spacer()
                            Button("Remove") {
                                panel.removeAuxiliaryInstrument(id)
                            }
                            .font(.caption)
                            .disabled(manifest.status != .draft)
                            .help(
                                "removes '\(id)' from outcomeInstruments "
                                    + "(draft-only). With the reader removed, "
                                    + "Answer-token mode runs genuinely "
                                    + "logprob-only — no sampled generation")
                        }
                    }
                    // Ordinal-scale instrument (general option-ladder
                    // endpoint) — controls live in their own file; this is
                    // the one wiring line.
                    OrdinalScaleInstrumentControls(manifest: manifest, panel: panel)
                    if let recordKinds = panel.effectiveRecordKindsNote {
                        Label(recordKinds, systemImage: "info.circle")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .textSelection(.enabled)
                    }
                    // Pin/unpin fitted reader artifacts — before 2026-08-01
                    // the button below gated on a list nothing in the app
                    // could populate.
                    ReaderPinControls(manifest: manifest, panel: panel)
                    if panel.canAddReaderInstrument {
                        Button("Add reader instrument (repeReaderScore)") {
                            panel.addReaderInstrument()
                        }
                        .font(.caption)
                        .help(
                            "declares repeReaderScore alongside the current "
                                + "mode — the pinned readers score every "
                                + "sampled response (sampled generation runs "
                                + "even in Answer-token mode)")
                    }
        }
    }

    /// Judging controls — the unified judging section (its structure and
    /// copy live in `JudgingSectionControls`, JudgingSectionView.swift)
    /// plus the analysis settings and the save action. Every study type
    /// judges; multi-agent studies judge transcripts.
    @ViewBuilder
    private func judgingControls(
        manifest: ExperimentManifest, panel: ExperimentPanel
    ) -> some View {
        Group {
                    JudgingSectionControls(
                        service: service, manifest: manifest, panel: panel)

                    analysisSettings(manifest: manifest, panel: panel)

                    if manifest.status == .draft {
                        Button("Save Evaluation Settings") { panel.saveProtocol() }
                            .help(
                                "save judges, rubric, structured output "
                                    + "instructions, case family, and the "
                                    + "option-length acknowledgment")
                    }
        }
    }

    @ViewBuilder
    private func validationControls(manifest: ExperimentManifest, panel: ExperimentPanel) -> some View {
        // Validate routes through the same server-resident path as Run in a
        // paired server workspace, so it shares the residency gate (the
        // callout in the run controls below explains the disabled state).
        // KNOWN-unpaired servers (Mac-authority mode, 2026-07-21) submit
        // validate as a hash-pinned BUNDLE job instead — no residency
        // needed, and the evidence bundle imports back into this workspace.
        let bundleValidate = panel.isKnownUnpairedServerWorkspace
        let missingOnServer = panel.isServerWorkspace && !bundleValidate
            && panel.serverHasSelectedStudy == false
        // The validation group: the declared read depth sits WITH the
        // button that launches validation (2026-08-01 review feedback —
        // it was orphaned in Evaluation below the save button, where its
        // connection to the validate verb was invisible).
        Text("Validation")
            .font(.caption.bold())
            .padding(.top, 4)
        if !manifest.concepts.isEmpty {
            ValidationDepthControls(manifest: manifest, panel: panel)
        }
        // Button row extracted (ValidateStudyButtonRow.swift): the gate
        // wiring pushed this function past the type-checker budget.
        ValidateStudyButtonRow(
            service: service,
            panel: panel,
            bundleValidate: bundleValidate,
            missingOnServer: missingOnServer,
            help: Self.validateHelp,
            pendingModelJob: $pendingModelJob,
            runOnServerExpanded: $runOnServerExpanded)

        Text(Self.validateCaption(variantsPresent: !manifest.variantConditions.isEmpty))
            .font(.caption2)
            .foregroundStyle(.secondary)
        if bundleValidate {
            Text(
                "unpaired server Compute — Validate submits the study as a "
                    + "hash-pinned bundle job on "
                    + "\(service.cluster.substrateLabel) (Remote options set "
                    + "executor/GPU/walltime); its evidence bundle imports "
                    + "back into this workspace and satisfies the local "
                    + "freeze gate for server runs")
                .font(.caption2)
                .foregroundStyle(.secondary)
        } else if panel.isServerWorkspace {
            Text(serverRunCaption("Validate Study"))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        // Direct validation executes the SERVER-RESIDENT copy only — the
        // callout explains the disabled state and points at the unified
        // Run's portable bundle path (which needs no residency).
        if missingOnServer {
            studyNotOnServerCallout(panel: panel)
        }

        if let validationDirectory = panel.lastValidationDirectory {
            Text(validationDirectory)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .help("latest completed validation run directory")
        }
    }

    /// A11: the explicit Extract action — the CLI `experiment extract` verb
    /// in-panel. Deliberately offered on drafts AND frozen studies:
    /// extraction is deterministic re-derivation from the pinned recipe (the
    /// CLI verb gates only on verify()). Server workspaces submit the
    /// server's extract verb as a durable job.
    @ViewBuilder
    private func extractControls(manifest: ExperimentManifest, panel: ExperimentPanel) -> some View {
        let missingOnServer = panel.isServerWorkspace
            && panel.serverHasSelectedStudy == false
        let busy = panel.isExtracting || panel.isRunning || panel.isValidating
            || panel.isSweeping
        let reason = ExperimentPanel.extractDisabledReason(
            busy: busy,
            hasViolations: !panel.violations.isEmpty,
            missingOnServer: missingOnServer)
        HStack(spacing: 8) {
            Button(panel.isExtracting ? "Extracting…" : "Extract Vectors") {
                // Item 2: same no-GPU-session gate as Run/Validate —
                // extraction loads the pinned model on the server.
                let panel = panel
                ModelJobGPUGate.submit(
                    "vector extraction", service: service,
                    pending: $pendingModelJob
                ) { await panel.extractStudy() }
            }
            .buttonStyle(.bordered)
            .disabled(reason != nil)
            .help(Self.extractHelp)
            if panel.isExtracting, !panel.isServerWorkspace {
                ProgressView().controlSize(.small)
                Button("Stop", role: .destructive) { panel.cancelExtract() }
                    .controlSize(.small)
                    .disabled(panel.extractCancelRequested)
                    .help(
                        "stops after the current concept; completed vectors "
                            + "stay in the run directory, marked cancelled")
            }
        }
        if let reason {
            Text(reason)
                .font(.caption2)
                .foregroundStyle(.secondary)
        } else {
            Text(
                "re-derives every pinned concept's vectors into a new "
                    + "immutable runs/ directory — allowed on drafts and "
                    + "frozen studies alike (deterministic from the pins)")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        if panel.isServerWorkspace {
            Text(serverRunCaption("Extract"))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        if let extractDirectory = panel.lastExtractDirectory {
            Text(extractDirectory)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .help("latest completed extract run directory")
        }
    }

    /// WS6.3: the substrate decision for the ONE Run control — every rule
    /// (default scope, stochastic pinning, connect-first greying) is
    /// unit-tested in `SubstrateRouting`, not here.
    private func runDecision(manifest: ExperimentManifest) -> SubstrateRouting.Decision {
        SubstrateRouting.decide(
            SubstrateRouting.Inputs(
                temperature: manifest.temperature,
                samplesPerItem: manifest.samplesPerItem,
                // E1: routing follows the resolved execution PLAN. A
                // logprob-only study never samples, so its temperature must
                // not pin it to the server.
                outcomeInstruments: manifest.outcomeInstruments,
                activeWorkspaceIsServer: panel.isServerWorkspace,
                siteRegistered: !service.cluster.servers.isEmpty,
                siteName: service.cluster.activeServer?.name
                    ?? service.cluster.servers.first?.name,
                serverConnected: service.cluster.capabilities != nil,
                userSelection: runSubstrate))
    }

    @ViewBuilder
    private func runControls(manifest: ExperimentManifest, panel: ExperimentPanel) -> some View {
        let decision = runDecision(manifest: manifest)
        // Greedy-only is a LOCAL substrate limitation (no per-run sampling
        // seed in the MLX generator). A stochastic MANIFEST already pins the
        // picker to the server; this gate catches a nonzero draft field.
        // E1: greedy-only is a LOCAL sampler limitation, so it binds only a
        // study that actually samples. A logprob-only study is deterministic
        // whatever the temperature says.
        let plan = ExecutionPlan.resolve(instruments: manifest.outcomeInstruments)
        let requiresGreedy = decision.selection == .thisMac
            && plan.samplingIsOperative
            && (manifest.temperature != 0 || panel.runTemperature != 0)
        substratePickerRow(decision: decision)
        runNoteRows(decision: decision)
        // A declared sampling setting this plan will never read. Advisory,
        // not a refusal: the run is well-defined and its result unaffected,
        // but a temperature that decides nothing is a design mistake.
        if let inert = ExecutionPlan.inertSamplingAdvisory(
            instruments: manifest.outcomeInstruments,
            temperature: manifest.temperature,
            samplesPerItem: manifest.samplesPerItem)
        {
            Label(inert, systemImage: "info.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        // P1 — prominent PRE-RUN warning: the data carries options but no
        // categorical instrument is declared, so the run will only generate
        // and parse answer text.
        if let warning = panel.instrumentActivationWarning {
            Label(warning, systemImage: "exclamationmark.triangle")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.orange)
                .textSelection(.enabled)
                .help(
                    "declare the instrument in Evaluation › Outcome mode — "
                        + "measurement method is manifest provenance, never "
                        + "inferred from the data")
        }
        primaryRunRow(
            manifest: manifest, panel: panel, decision: decision,
            requiresGreedy: requiresGreedy)
        preflightRows(manifest: manifest, panel: panel, decision: decision)
        if requiresGreedy {
            Text("Run Study requires saved Temperature = 0 for reproducible measured runs.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        if let runDirectory = panel.lastRunDirectory {
            Text(runDirectory)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .help("latest completed study run directory")
        }
        if let serverRunDirectory = panel.lastServerRunDirectory {
            Text("server run: \(serverRunDirectory)")
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .help("run directory in the active server's runs/ tree")
        }
    }

    @ViewBuilder
    private func substratePickerRow(decision: SubstrateRouting.Decision) -> some View {
        Picker(
            "Run on",
            selection: Binding(
                get: { decision.selection },
                set: { newValue in
                    runSubstrate = newValue
                    runner.clearPreflight()
                })
        ) {
            Text("This Mac").tag(SubstrateRouting.Substrate.thisMac)
            if decision.serverSelectable {
                Text(decision.serverLabel).tag(SubstrateRouting.Substrate.server)
            } else {
                // Greyed, never pickable: connect (or add a site) first.
                Text("\(decision.serverLabel) — \(decision.serverHint ?? "connect first")")
                    .tag(SubstrateRouting.Substrate.server)
                    .selectionDisabled()
            }
        }
        .pickerStyle(.segmented)
        .disabled(decision.pinnedToServer)
        .help(
            "which substrate executes this study — the substrate is a scope, "
                + "not a mode: same manifest, same lifecycle, artifacts land in "
                + "the scoped workspace with their substrate stamp")
    }

    @ViewBuilder
    private func runNoteRows(decision: SubstrateRouting.Decision) -> some View {
        if let note = decision.stochasticNote {
            // The kind enforcement: preselected, explained — never an error
            // later.
            Label(note, systemImage: "die.face.5")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        if let hint = decision.serverHint, decision.selection == .server {
            Text(hint)
                .font(.caption2)
                .foregroundStyle(.orange)
        }
        if let hint = decision.localHint {
            Text(hint)
                .font(.caption2)
                .foregroundStyle(.orange)
        }
    }

    @ViewBuilder
    private func primaryRunRow(
        manifest: ExperimentManifest,
        panel: ExperimentPanel,
        decision: SubstrateRouting.Decision,
        requiresGreedy: Bool
    ) -> some View {
        let busy = panel.isRunning || panel.isExtracting || runner.isSubmitting
        // Finding 11c: what this button will ACTUALLY submit, stated before
        // it is pressed — the verb lives in a collapsed disclosure and does
        // not consult the Pipeline Composer's declared chain.
        ExecutionPlanEchoRow(
            plan: ExecutionPlanEcho.describe(
                verb: panel.remoteVerb,
                target: decision.selection,
                serverLabel: decision.serverLabel,
                dryRun: panel.remoteDryRun,
                declaredPipelineStages: ShardedSubmission.declaredPipelineStages(
                    manifest.pipeline)),
            revealOptions: { runOnServerExpanded = true })
        HStack(spacing: 8) {
            Button(
                SubstrateRouting.runButtonLabel(
                    decision: decision, verb: panel.remoteVerb,
                    dryRun: panel.remoteDryRun, isBusy: busy)
            ) {
                let runner = runner
                let cluster = service.cluster
                let run: @MainActor () async -> Void = {
                    await runner.run(
                        manifest: manifest, decision: decision, panel: panel,
                        cluster: cluster)
                }
                // Item 2 + 2026-07-21 incident part 1: warn before a
                // GPU-less model-running server submission — no GPU session,
                // or (more specific) a bundle whose OWN options request no
                // GPU on a Slurm site (non-slurm executor / empty gres would
                // execute inside the controller's small CPU allocation).
                // Dry runs (nothing executes) and non-model verbs (analyze)
                // submit straight through — the predicate rules; so do
                // local runs.
                if decision.selection == .server {
                    ModelJobGPUGate.submit(
                        "study \(panel.remoteVerb)", service: service,
                        pending: $pendingModelJob,
                        bundleOptions: ModelJobSubmissionPreflight.BundleOptions(
                            executor: panel.remoteExecutor,
                            gres: panel.remoteGres,
                            verb: panel.remoteVerb,
                            dryRun: panel.remoteDryRun),
                        fixOptions: {
                            panel.applyGPUAllocationFix()
                            runOnServerExpanded = true
                        },
                        action: run)
                } else {
                    Task { await run() }
                }
            }
            .buttonStyle(.bordered)
            .disabled(
                busy || !panel.violations.isEmpty || requiresGreedy
                    || decision.runBlockedReason != nil)
            .help(decision.selection == .thisMac ? Self.runHelp : Self.unifiedRemoteRunHelp)
            // A1: a LOCAL in-process run is cancellable between generations
            // (server-routed runs get the Cancel Server Job control instead).
            if panel.isRunning {
                ProgressView().controlSize(.small)
                Button("Stop", role: .destructive) { panel.cancelStudyRun() }
                    .controlSize(.small)
                    .disabled(panel.studyRunCancelRequested)
                    .help(
                        "stops after the current generation; completed "
                            + "generations stay in the run directory, marked "
                            + "cancelled (no report.json) — reported as cancelled "
                            + "by user, never as an error")
            }
        }

        if decision.selection == .server {
            Text(serverRunCaption("Run"))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    /// WS4 preflight surfacing: ok → silent; warn → amber lines, job already
    /// proceeding; fail → the submission stopped, failing checks shown, and
    /// the forced override sits behind a confirmation that restates them.
    @ViewBuilder
    private func preflightRows(
        manifest: ExperimentManifest,
        panel: ExperimentPanel,
        decision: SubstrateRouting.Decision
    ) -> some View {
        if let preflight = runner.preflight, preflight.verdict == .warn {
            ForEach(preflight.attentionLines) { line in
                Label(line.message, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
        }
        if let refusal = runner.refusal {
            VStack(alignment: .leading, spacing: 4) {
                Label(
                    refusal.inlineSummary ?? "preflight failed — submission refused",
                    systemImage: "xmark.octagon")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.red)
                ForEach(refusal.failingLines) { line in
                    Text("• \(line.message)")
                        .font(.caption)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                }
                Button("Override (forced)…") { confirmForcedOverride = true }
                    .controlSize(.small)
                    .help(
                        "resubmit with force=true, bypassing the failed preflight "
                            + "checks — loud and deliberate, never silent")
                    .confirmationDialog(
                        "Force past preflight?",
                        isPresented: $confirmForcedOverride,
                        titleVisibility: .visible
                    ) {
                        Button("Submit anyway (forced)", role: .destructive) {
                            Task {
                                await runner.run(
                                    manifest: manifest, decision: decision,
                                    panel: panel, cluster: service.cluster,
                                    force: true)
                            }
                        }
                    } message: {
                        Text(refusal.overrideConfirmationMessage)
                    }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.red.opacity(0.07), in: RoundedRectangle(cornerRadius: 8))
        }
        if let status = runner.statusLine {
            Text(status)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
    }

    /// Prominent warning callout for a study that exists locally but not in
    /// the active server's workspace: Run Server Copy is disabled and this
    /// box names both ways forward (portable bundle, or pair the server).
    @ViewBuilder
    private func studyNotOnServerCallout(panel: ExperimentPanel) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(
                ExperimentPanel.residencyCalloutMessage(
                    study: panel.selectedName ?? "study",
                    substrate: service.cluster.substrateLabel),
                systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
            Button("Show remote options") { runOnServerExpanded = true }
                .controlSize(.small)
                .help(
                    "expands the Remote options — the unified Run button "
                        + "submits a portable hash-pinned bundle, which works "
                        + "without server residency")
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }

    /// Read-only browse of the active server's `runs/` tree (`GET /api/runs`).
    /// TODO(server): the server exposes only the listing plus per-run file
    /// fetch (`GET /api/runs/{id}/file`) — no structured results/report API —
    /// so result *detail* still comes home through the evidence-bundle import
    /// (auto-import, the health card, or Remote options). Extend this to a
    /// full remote result viewer once a server results endpoint exists; do
    /// not invent one client-side.
    @ViewBuilder
    private func serverRunsGroup(panel: ExperimentPanel) -> some View {
        Group {
            Text("Server runs — \(service.cluster.substrateLabel)")
                .font(.caption.bold())
                .padding(.top, 4)
            Text("Every immutable run directory on the server — any verb, "
                + "any study. Pipelines (below) is the chain-level view: one "
                + "row per chain with per-stage status and gate aborts.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Button("Refresh Server Runs") {
                Task { await panel.refreshRemoteRuns() }
            }
            .help("list the immutable run directories in the active server's runs/ tree")
            if panel.remoteRuns.isEmpty {
                Text("No server runs listed — refresh, or run something on this server first.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(panel.remoteRuns.prefix(40)) { run in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(run.id)
                            .font(.caption.monospaced())
                            .textSelection(.enabled)
                        HStack(spacing: 8) {
                            if run.hasReport { Text("report") }
                            if run.hasGenerations { Text("generations") }
                            if !run.vectorNames.isEmpty {
                                Text("\(run.vectorNames.count) vector\(run.vectorNames.count == 1 ? "" : "s")")
                            }
                            if let task = run.task, !task.isEmpty {
                                Text(task).lineLimit(1)
                            }
                        }
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 2)
                }
                if panel.remoteRuns.count > 40 {
                    Text("… and \(panel.remoteRuns.count - 40) more")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    /// Chain-runner (pipeline) states for the selected experiment — the
    /// stage-5 awaiting/aborted affordance. An ABORT is a recorded
    /// scientific determination, rendered as such: the failing stage, each
    /// gate's detail with measured vs threshold, and "Duplicate & adjust"
    /// (the lifecycle answer to a stopped chain — iterate by duplicating,
    /// never by editing the preregistered object).
    @ViewBuilder
    private func pipelinesGroup(panel: ExperimentPanel) -> some View {
        Group {
            Text("Pipelines — \(service.cluster.substrateLabel)")
                .font(.caption.bold())
                .padding(.top, 4)
            Button("Refresh Pipelines") {
                Task { await panel.refreshPipelineRuns() }
            }
            .help(
                "list this experiment's chain-runner runs on the active "
                    + "server: per-stage status, gate aborts, and promoted "
                    + "agents")
            if panel.pipelineRuns.isEmpty {
                Text("No pipelines listed — refresh, or submit the "
                    + "'pipeline' verb from Remote options.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(panel.pipelineRuns.prefix(20)) { pipeline in
                    pipelineRunRow(pipeline, panel: panel)
                }
            }
            if !panel.localPipelineRuns.isEmpty {
                Text("Imported / local")
                    .font(.caption2.bold())
                    .foregroundStyle(.secondary)
                ForEach(panel.localPipelineRuns.prefix(10)) { pipeline in
                    pipelineRunRow(pipeline, panel: panel)
                }
            }
        }
    }

    @ViewBuilder
    private func pipelineRunRow(
        _ pipeline: ClusterClient.PipelineRunSummary,
        panel: ExperimentPanel
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 8) {
                Text(pipeline.run)
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
                    .lineLimit(1)
                Text(pipeline.stateLabel)
                    .font(.caption2.bold())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 1)
                    .background(
                        pipelineStateColor(pipeline).opacity(0.18),
                        in: Capsule())
                if pipeline.manifestStatus == "draft" {
                    Text("draft (exploratory)")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
            }
            Text(pipeline.stageSummaryLine)
                .font(.caption2)
                .foregroundStyle(.secondary)
            if let updated = pipeline.updatedAt {
                // For "unfinished" chains this is the evidence for judging
                // running-vs-abandoned — the listing cannot know.
                Text("last ledger write: \(updated)")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
            }
            if let agents = pipeline.promotedAgents, !agents.isEmpty {
                ForEach(agents.keys.sorted(), id: \.self) { concept in
                    if let agent = agents[concept] {
                        Text(promotedAgentLine(concept: concept, agent: agent))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                }
            }
            if let abort = pipeline.abort {
                pipelineAbortCard(abort, panel: panel)
            }
        }
        .padding(.vertical, 2)
    }

    private func pipelineStateColor(
        _ pipeline: ClusterClient.PipelineRunSummary
    ) -> Color {
        switch pipeline.disposition {
        case "completed": .green
        case "aborted": .orange
        default: .blue
        }
    }

    private func promotedAgentLine(
        concept: String,
        agent: ClusterClient.PipelineRunSummary.PromotedAgent
    ) -> String {
        var line = "\(concept) → \(agent.artifact ?? "?")"
        if let cell = agent.winningCell, let layer = cell.layer,
            let alpha = cell.alpha
        {
            line += " (L\(layer), α\(alpha.formatted()))"
        }
        return line
    }

    /// The abort record, rendered as the determination it is — never as a
    /// job failure. Detail strings come verbatim from the server's
    /// GateResult (researcher-facing prose).
    @ViewBuilder
    private func pipelineAbortCard(
        _ abort: ClusterClient.PipelineRunSummary.Abort,
        panel: ExperimentPanel
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("Stopped at '\(abort.stage ?? "?")' — a gate said stop. "
                + "Nothing after it ran.")
                .font(.caption.bold())
            ForEach(
                Array((abort.gates ?? []).enumerated()), id: \.offset
            ) { _, gate in
                VStack(alignment: .leading, spacing: 1) {
                    if let detail = gate.detail {
                        Text(detail)
                            .font(.caption2)
                            .textSelection(.enabled)
                    }
                    if let measured = gate.measured,
                        let threshold = gate.threshold
                    {
                        Text("measured \(measured.formatted()) vs threshold "
                            + "\(threshold.formatted())")
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                    }
                }
            }
            if let evidence = abort.evidenceRunID {
                Text("evidence: \(evidence)")
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            Button("Duplicate & Adjust") { panel.duplicateSelected() }
                .font(.caption)
                .help(
                    "iterate by duplicating, never by editing: creates a "
                        + "draft copy of this experiment to adjust "
                        + "stimuli/gates/grid, leaving the preregistered "
                        + "chain and its abort record intact")
        }
        .padding(6)
        .background(.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
    }

    /// The active site profile's cap on sharded fan-out (nil = uncapped;
    /// the stepper falls back to `ShardedSubmission.defaultStepperCap`).
    private var siteMaxParallelGPUJobs: Int? {
        guard case .slurm(let slurm)? = service.cluster.activeSite?.scheduler
        else { return nil }
        return slurm.maxParallelGPUJobs
    }

    /// WS6.3: the retired "Run on Server" split is now "Remote options" —
    /// the inputs the unified Run button uses when the substrate picker
    /// selects the site (verb, executor, dry run, resources), plus job
    /// reconnect/cancel/evidence utilities. No submit button lives here:
    /// the ONE Run control above is the submission path.
    @ViewBuilder
    private func remoteRunControls(manifest: ExperimentManifest, panel: ExperimentPanel) -> some View {
        @Bindable var panel = panel
        let decision = runDecision(manifest: manifest)
        if decision.selection == .server || panel.isServerWorkspace {
        DisclosureGroup("Remote options", isExpanded: $runOnServerExpanded) {
            LabeledContent("Server", value: service.cluster.serverHostLabel)
                .help(
                    "shared server connection — edit the URL and bearer token in the "
                        + "window toolbar's substrate selector")
            if let summary = panel.remoteProfileSummary {
                LabeledContent("Backend", value: summary)
                    .font(.caption)
                    .help("server profile · executor · launch topology reported by /api/capabilities")
            }
            Picker("Verb", selection: $panel.remoteVerb) {
                Text("verify").tag("verify")
                // A11: symmetric artifact production — extract exists on the
                // bundle path (VALID_STUDY_VERBS) like every other verb.
                Text("extract").tag("extract")
                Text("validate").tag("validate")
                Text("sweep").tag("sweep")
                Text("run").tag("run")
                Text("evaluate").tag("evaluate")
                // A3 rider: analyze exists server-side (headless paired
                // statistics over the newest completed run).
                Text("analyze").tag("analyze")
                // Stage 3: the chain runner — the manifest's declared
                // pipeline stages as ONE submission (one model load,
                // gate-aborted between stages). EXPERIMENTAL until stage 5
                // lands the abort/awaiting UI: the server refuses a
                // manifest with no explicit pipeline block, and an abort
                // reads as a successful job here — check pipeline.json /
                // pipeline-abort.json in the run directory.
                Text("pipeline (experimental)").tag("pipeline")
            }
            .help("the experiment verb the unified Run button submits remotely")
            Picker("Executor", selection: $panel.remoteExecutor) {
                Text("local").tag("local")
                Text("slurm").tag("slurm")
            }
            Toggle("Dry run (prepare only, nothing executes)", isOn: $panel.remoteDryRun)
                .help(
                    "stages the bundle and renders the job without executing "
                        + "the study — the job finishes as 'prepared'")
            HStack {
                TextField("GPU gres", text: $panel.remoteGres)
                    .textFieldStyle(.roundedBorder)
                TextField("walltime", text: $panel.remoteWalltime)
                    .textFieldStyle(.roundedBorder)
            }
            // Resume-on-checkpoint (2026-07-22 incident: a checkpointed run
            // had no resume path) — DEFAULT ON: a checkpointed batch run
            // continuing is what submitting it asked for.
            HStack {
                Toggle(
                    "Resume automatically if the run checkpoints",
                    isOn: $panel.remoteResumePolicy.autoResubmit)
                Stepper(value: $panel.remoteResumePolicy.limit, in: 1...50) {
                    Text("up to \(panel.remoteResumePolicy.limit) restarts")
                        .font(.caption)
                }
                .disabled(!panel.remoteResumePolicy.autoResubmit)
            }
            .help(
                "when a Slurm run exits at the walltime margin with a clean "
                    + "checkpoint (exit 85), the server re-submits the job's "
                    + "own sbatch script and the run continues from the "
                    + "checkpoint — up to this many restarts. Off, the job "
                    + "parks as 'checkpointed (resumable)' until you press "
                    + "Resume. Slurm submissions only; the transcript line "
                    + "stamps what was sent")
            // Multi-GPU fan-out (2026-07-22): shard a Slurm run across K
            // sibling GPU jobs; the server merges the partials back into one
            // run byte-identical to a single job (records are independent).
            // Disabled-with-explanation when this submission cannot shard
            // (finding 5: the server shards only run and run-FIRST
            // pipelines — a stepper the server would ignore must say so).
            let shardingReason = ShardedSubmission.shardingUnavailableReason(
                verb: panel.remoteVerb, executor: panel.remoteExecutor,
                declaredPipelineStages: ShardedSubmission.declaredPipelineStages(
                    manifest.pipeline))
            Stepper(
                value: $panel.remoteParallelJobs,
                in: 1...ShardedSubmission.stepperCap(siteMax: siteMaxParallelGPUJobs)
            ) {
                Text(
                    panel.remoteParallelJobs > 1
                        ? "Parallel GPU jobs: \(panel.remoteParallelJobs)"
                        : "Parallel GPU jobs: 1 (no sharding)")
                    .font(.caption)
            }
            .disabled(shardingReason != nil)
            .help(
                "shard the run across this many simultaneous GPU jobs "
                    + "(Slurm, run/pipeline-first-stage-run only) — every "
                    + "record is independent, so the merged result is "
                    + "byte-identical to a single job and wall-clock scales "
                    + "with GPUs. The cap comes from the site profile's "
                    + "'Max parallel GPU jobs' (check yours with `sacctmgr "
                    + "show qos format=Name,MaxTRESPerUser`)")
            if let shardingReason {
                Text(shardingReason)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            HStack {
                Button("Test Connection") { Task { await panel.testRemoteConnection() } }
                Button("Cancel Job") { Task { await panel.cancelRemoteJob() } }
                    .disabled(panel.remoteJobID == nil)
                Button("Import Evidence") { Task { await panel.downloadRemoteEvidence() } }
                    .disabled(panel.remoteJobID == nil)
            }
            Text(Self.remoteOptionsCaption)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(Self.importEvidenceCaption)
                .font(.caption2)
                .foregroundStyle(.secondary)
            // Reconnect by job id — e.g. after an app restart while a Slurm job runs.
            HStack {
                TextField("job id to reconnect", text: $reconnectJobID)
                    .textFieldStyle(.roundedBorder)
                Button("Reconnect") {
                    Task { await panel.reconnectRemoteJob(reconnectJobID) }
                }
                .disabled(reconnectJobID.trimmingCharacters(in: .whitespaces).isEmpty)
                if !panel.remoteLogLines.isEmpty {
                    Button("Stop Log") { panel.stopRemoteLogStream() }
                }
            }
            .help("resume watching a running or finished job by its id after an app or session restart")
            if let job = panel.remoteJobID {
                LabeledContent("Remote job", value: job)
                    .font(.caption)
                    .textSelection(.enabled)
            }
            if let uploaded = panel.remoteLastUploadedBundle {
                LabeledContent("Uploaded bundle", value: uploaded)
                    .font(.caption2.monospaced())
                    .textSelection(.enabled)
            }
            if let imported = panel.remoteImportedRunDirectory {
                LabeledContent("Imported run", value: imported)
                    .font(.caption2.monospaced())
                    .textSelection(.enabled)
            }
            if let status = panel.remoteStatus {
                Text(status)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            if !panel.remoteLogLines.isEmpty {
                // The log is always titled with the job id verbatim so it can
                // be copied and reconnected to later.
                Text("job log — \(panel.remoteJobID ?? "?")")
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
                ScrollView {
                    Text(panel.remoteLogLines.joined(separator: "\n"))
                        .font(.caption2.monospaced())
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                }
                .frame(minHeight: 120, maxHeight: 260)
            }
        }
        .help(
            "inputs for the unified Run button's remote submission (verb, "
                + "executor, dry run, resources) plus job reconnect and "
                + "evidence utilities")
        }
    }

    /// Session-scoped list of server jobs submitted from this panel (run
    /// verbs and bundle submissions), ids selectable so a researcher can
    /// reconnect after an app restart.
    @ViewBuilder
    private func recentServerJobsGroup(panel: ExperimentPanel) -> some View {
        Group {
            Text("Recent server jobs")
                .font(.caption.bold())
                .padding(.top, 4)
            ForEach(panel.recentServerJobs) { job in
                HStack(spacing: 8) {
                    Text(job.id)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                    Text("\(job.verb) · \(job.study) · \(job.state)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                    // Checkpointed (resumable) jobs get the Resume button
                    // right on the row — the 2026-07-22 incident was this
                    // exact dead end.
                    if RemoteJobStatusClass.offersResume(status: job.state) {
                        Button("Resume") {
                            Task { await panel.resubmitRemoteJob(job.id) }
                        }
                        .controlSize(.small)
                        .help(
                            "re-submit this checkpointed job's own sbatch "
                                + "script — the run continues from its "
                                + "checkpoint; the status line reports the "
                                + "new Slurm job id")
                    }
                    // Completed run-verb jobs can carry an evidence bundle:
                    // the same import the Run-on-Server disclosure offers,
                    // right where the finished job is listed.
                    if ExperimentPanel.jobOffersEvidenceImport(
                        verb: job.verb, state: job.state)
                    {
                        Button("Import evidence") {
                            Task { await panel.importEvidence(fromJobID: job.id) }
                        }
                        .controlSize(.small)
                        .help(
                            "download this job's evidence bundle, verify its "
                                + "hashes, and land it under this workspace's "
                                + "runs/ — the status line names the imported "
                                + "run directory")
                    }
                }
                .padding(.vertical, 1)
            }
            if let imported = panel.remoteImportedRunDirectory {
                LabeledContent("Imported run", value: imported)
                    .font(.caption2.monospaced())
                    .textSelection(.enabled)
                    .help("already visible in Results — imports land as immutable runs/ directories")
            }
            Button("Refresh Job States") {
                Task { await panel.refreshRecentServerJobs() }
            }
            .help("re-query the active server's job list for current states")
            Text(Self.recentJobsCaption)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    /// The substrate decision for the Freeze control — same rule family as
    /// the unified Run picker, unit-tested in `FreezeRouting`, never here.
    private func freezeRoutingDecision() -> FreezeRouting.Decision {
        FreezeRouting.decide(
            FreezeRouting.Inputs(
                activeWorkspaceIsServer: panel.isServerWorkspace,
                serverConnected: service.cluster.capabilities != nil,
                serverLabel: service.cluster.substrateLabel,
                serverHasSelectedStudy: panel.serverHasSelectedStudy,
                workspaceKnownUnpaired: panel.isKnownUnpairedServerWorkspace))
    }

    /// Freeze button + readiness for a draft: the button routes to the
    /// substrate the workspace scopes to ("Freeze (on <server>)…" in a
    /// server workspace — gates evaluated there against SERVER-substrate
    /// evidence), local readiness renders as before, cross-substrate
    /// evidence advisories are promoted to warnings at this decision point,
    /// and the server's own gate refusal / advisories from the last remote
    /// attempt render in the same idiom as the local readiness items.
    @ViewBuilder
    private func freezeControls(manifest: ExperimentManifest, panel: ExperimentPanel) -> some View {
        let decision = freezeRoutingDecision()
        Button(decision.buttonLabel) { confirmFreeze = true }
            .buttonStyle(.borderedProminent)
            // Server-routed: the SERVER's gates decide remote readiness —
            // local verification failures render as context below, never as
            // a disabled button (rule unit-tested in FreezeRouting).
            .disabled(
                FreezeRouting.freezeButtonDisabled(
                    decision: decision,
                    hasLocalViolations: !panel.violations.isEmpty))
            .help(decision.target == .server ? Self.remoteFreezeHelp : Self.freezeHelp)
            .confirmationDialog(
                freezeDialogTitle(manifest.name, decision: decision),
                isPresented: $confirmFreeze
            ) {
                Button(decision.confirmLabel, role: .destructive) {
                    if decision.target == .server {
                        Task { await panel.freezeOnActiveServer() }
                    } else {
                        panel.freeze()
                    }
                }
            }
        if let note = decision.executorNote {
            Text(note)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        if let blocked = decision.blockedReason {
            Label(blocked, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
        }
        // Server-routed freeze with local verification failures: the local
        // readiness is CONTEXT here (the server verifies ITS copy's pins at
        // the click) — informational, never a disabled button.
        if decision.target == .server,
            let note = FreezeRouting.localViolationsContextNote(
                count: panel.violations.count,
                serverLabel: service.cluster.substrateLabel)
        {
            Label(note, systemImage: "info.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
        // The manifest-identity guard's answers from the last remote-freeze
        // attempt: a block (the server's same-named copy is not the document
        // on screen — field-level summary + remedy) renders prominently; a
        // proceeded-with note (server-only copy / paired-unverifiable)
        // renders as info.
        if let identityWarning = panel.remoteFreezeIdentityWarning {
            Label(identityWarning, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
            // The one-click remedy (2026-07-21 incident, part 3): the
            // mismatch block used to name a remedy the app didn't offer.
            // Push the manifest ON SCREEN as the server's draft copy, then
            // re-verify — draft manifests only, frozen copies refuse
            // server-side (freeze firewall).
            if panel.remoteFreezeCanSyncDraft {
                Button(
                    panel.isSyncingServerDraft
                        ? "Updating the server's copy…"
                        : "Update the server's copy"
                ) {
                    Task { await panel.pushManifestToActiveServer() }
                }
                .controlSize(.small)
                .disabled(panel.isSyncingServerDraft)
                .help(
                    "push the manifest you are looking at to "
                        + "\(service.cluster.substrateLabel) as its DRAFT copy "
                        + "and re-run the identity check — the freeze itself "
                        + "stays a separate, deliberate click")
            }
        }
        if let identityNote = panel.remoteFreezeIdentityNote {
            Label(identityNote, systemImage: "info.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
        // Read-only freeze readiness: the same gates freeze enforces,
        // reported before the one-way click. For a server-routed freeze the
        // gates are re-evaluated SERVER-side at the click; on a paired
        // workspace this local view reads the same shared tree.
        if let readiness = panel.freezeReadiness {
            Label(
                readiness.displayLine(),
                systemImage: readiness.ready
                    ? "checkmark.seal" : "hourglass")
                .font(.caption)
                .foregroundStyle(readiness.ready ? Color.green : Color.secondary)
                .help(
                    readiness.ready
                        ? "every freeze gate is currently satisfied"
                        : readiness.unmetGates.joined(separator: "\n"))
            // Non-blocking advisories (e.g. hand-created variants without
            // sweep-selection provenance): visible next to the gates, never
            // a refusal. Cross-substrate validate-evidence advisories are
            // PROMINENT here — this is the freeze decision they exist for.
            freezeAdvisoryRows(readiness.advisories)
        }
        // Decision-time coherence check for a server-routed freeze on a
        // paired workspace: the same evidence, seen from the server's
        // perspective (validate-locally-then-freeze-on-server warns BEFORE
        // the click).
        if decision.target == .server,
            let advisory = panel.serverFreezeCrossSubstrateAdvisory
        {
            freezeAdvisoryRows([advisory])
        }
        // The server's own answer to the last remote-freeze attempt,
        // rendered exactly like the local readiness items: a gate refusal
        // reads as an unmet gate (verbatim server wording), advisories as
        // advisory rows.
        if let failure = panel.remoteFreezeGateFailure {
            Label(
                ExperimentStore.FreezeReadiness(unmetGates: [failure]).displayLine(),
                systemImage: "hourglass")
                .font(.caption)
                .foregroundStyle(Color.secondary)
                .help(failure)
                .textSelection(.enabled)
        }
        freezeAdvisoryRows(panel.remoteFreezeAdvisories)
    }

    /// One rendering rule for freeze advisories (local readiness, the
    /// paired-workspace server-perspective check, and the server's response
    /// advisories): cross-substrate evidence advisories render as warnings
    /// with the one-line rule appended; the rest stay info rows. The split
    /// itself is unit-tested in `FreezeRouting.present`.
    @ViewBuilder
    private func freezeAdvisoryRows(_ advisories: [String]) -> some View {
        let presentation = FreezeRouting.present(advisories: advisories)
        ForEach(presentation.prominent, id: \.self) { advisory in
            Label(advisory, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
        }
        ForEach(presentation.regular, id: \.self) { advisory in
            Label(advisory, systemImage: "info.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private func freezeDialogTitle(_ name: String, decision: FreezeRouting.Decision) -> String {
        switch decision.target {
        case .thisMac:
            "Freeze '\(name)'? This is one-way — afterwards the study can "
                + "only be duplicated, never edited."
        case .server:
            "Freeze '\(name)' on \(service.cluster.substrateLabel)? The server "
                + "evaluates the gates against ITS OWN substrate's validation "
                + "evidence and stamps frozenBy: \"server\". This is one-way — "
                + "afterwards the study can only be duplicated, never edited."
        }
    }

    /// Declared perturbation-policy inputs for a confirmation study —
    /// agent picker (promoted first), α deltas, control toggle. The
    /// expansion and every refusal live in `ConfirmationStudy.attach`.
    private func confirmationControls(panel: ExperimentPanel) -> some View {
        @Bindable var panel = panel
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Picker("Confirm agent", selection: $panel.confirmAgentID) {
                    Text("select…").tag(ModelVariantRecord.ID?.none)
                    ForEach(panel.confirmableAgents) { record in
                        Text(confirmAgentLabel(record))
                            .tag(ModelVariantRecord.ID?.some(record.id))
                    }
                }
                TextField("strength deltas (α)", text: $panel.confirmDeltasText)
                    .frame(width: 120)
                    .help(
                        "comma-separated positive strength offsets around the "
                            + "anchor α, e.g. 0.2, 0.5")
                Toggle(
                    "random-direction control",
                    isOn: $panel.confirmIncludeControl)
                    .help(
                        "adds a matched-norm random control at the anchor "
                            + "strength — same norm, deterministic random "
                            + "direction")
                Button("Attach Policy") { panel.attachPerturbations() }
                    .disabled(panel.confirmAgentID == nil)
            }
            // Non-blocking: confirmation of a hand-created agent stays legal,
            // but the evidence path runs through sweep-promoted agents.
            if let record = panel.confirmableAgents.first(where: {
                $0.id == panel.confirmAgentID
            }), record.artifact.promotion == nil {
                Text(
                    "hand-created agent: confirmation of an undeclared "
                        + "selection is exploratory, not evidence-grade — "
                        + "promote from a sweep for the evidence path")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
            Text(
                "declares the perturbation policy — anchor, α ± δ, control — "
                    + "and expands it into ordinary hashed conditions in this "
                    + "draft (visible below; the firewall pins them like any "
                    + "other condition)")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private func confirmAgentLabel(_ record: ModelVariantRecord) -> String {
        record.artifact.promotion != nil
            ? "\(record.artifact.name) · sweep-promoted"
            : "\(record.artifact.name) · hand-created"
    }

    /// The attached policy plus the conditions it generated — the agent
    /// vocabulary never hides the condition machinery underneath.
    private func perturbationPolicySummary(
        _ policy: ExperimentManifest.PerturbationPolicy, panel: ExperimentPanel
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Label(policySummaryLine(policy), systemImage: "scope")
                .font(.caption)
            ForEach(panel.generatedConfirmationConditions, id: \.name) { condition in
                Text(generatedConditionLine(condition))
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func policySummaryLine(
        _ policy: ExperimentManifest.PerturbationPolicy
    ) -> String {
        let deltas = policy.alphaDeltas.map { "\($0)" }.joined(separator: ", ")
        let origin = policy.sourceAgent.promoted ? "sweep-promoted" : "hand-created"
        return "confirmation policy: '\(policy.sourceAgent.name)' (\(origin)) — "
            + "anchor L\(policy.cell.layer) α\(policy.cell.alpha), δ [\(deltas)]"
            + (policy.includeMatchedNormControl ? ", matched-norm control" : "")
    }

    private func generatedConditionLine(
        _ condition: ExperimentManifest.Condition
    ) -> String {
        guard let slot = condition.slots.first else { return condition.name }
        let control = condition.controlType == "randomMatchedNorm" ? " (control)" : ""
        return "\(condition.name) · L\(slot.layer) α\(slot.alpha)\(control)"
    }

    /// One sweep-recommended cell (selection provenance present), shown as
    /// READ-ONLY provenance — the Create Agent edge lives in Agents →
    /// Optimizations (design brief: Studies consumes agents, it does not
    /// expose sweep plumbing).
    private func sweepRecommendationRow(
        _ condition: ExperimentManifest.Condition, panel: ExperimentPanel
    ) -> some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text(condition.name)
                    .font(.callout.weight(.medium))
                if let selection = condition.selection {
                    Text(recommendationCaption(selection))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
            Spacer()
            Button("Open in Optimizations") {
                openOptimizations()
            }
            .controlSize(.small)
            .help(
                "this study's optimization run in Agents → Optimizations — "
                    + "grid, recommendation, and Create Agent live there")
        }
    }

    private func recommendationCaption(
        _ selection: ExperimentManifest.SelectionProvenance
    ) -> String {
        var parts = [
            "L\(selection.winningCell.layer) α\(selection.winningCell.alpha)"
        ]
        if let metric = selection.criterion.objective?.metric,
           let value = selection.metrics[metric]
        {
            parts.append("\(metric) \(String(format: "%.3f", value))")
        }
        parts.append("dev \(selection.devPromptsHash.prefix(8))…")
        parts.append("run \(selection.sweepRun)")
        return parts.joined(separator: " · ")
    }

    private func conditionSummary(_ condition: ExperimentManifest.Condition) -> String {
        guard !condition.slots.isEmpty else {
            return "no steering · baseline"
        }
        return condition.slots.map {
            "\($0.concept) L\($0.layer) α\($0.alpha)"
        }.joined(separator: " + ")
            + (condition.alphaInNormUnits ? " (norm units)" : "")
            + " · band \(condition.bandWidth)"
            + (condition.neutralPCBasisLabel.map { " · neutral-removed: \($0)" } ?? "")
    }

    @ViewBuilder
    private func liveRunViewer(panel: ExperimentPanel) -> some View {
        let hasLiveContent =
            panel.isRunning || panel.isEvaluating
            || panel.liveRunDirectory != nil
            || panel.liveEvaluationDirectory != nil
            || panel.liveActiveGeneration != nil
            || panel.liveActiveJudgment != nil
            || !panel.liveGenerations.isEmpty
            || !panel.liveJudgments.isEmpty

        if hasLiveContent {
            Section("Run Viewer") {
                if let directory = panel.liveRunDirectory {
                    LabeledContent("Run", value: URL(filePath: directory).lastPathComponent)
                        .font(.caption)
                        .help("the immutable run artifact directory being written")
                }
                if let directory = panel.liveEvaluationDirectory {
                    LabeledContent("Judge", value: URL(filePath: directory).lastPathComponent)
                        .font(.caption)
                        .help("the immutable paired-judge artifact directory being written")
                }

                if let active = panel.liveActiveGeneration {
                    liveGenerationCard(active)
                    // A1: the progress row carries its own Stop (the same
                    // cooperative flag as the Run/Judge buttons' Stop).
                    if panel.isRunning {
                        Button("Stop Run", role: .destructive) {
                            panel.cancelStudyRun()
                        }
                        .controlSize(.small)
                        .disabled(panel.studyRunCancelRequested)
                        .help("stops after this generation; partial artifacts stay")
                    }
                }

                if !panel.liveGenerations.isEmpty {
                    DisclosureGroup("Generated Responses (\(panel.liveGenerations.count))") {
                        ForEach(panel.liveGenerations.prefix(12)) { generation in
                            compactGenerationCard(generation)
                        }
                    }
                    .help("completed generations appear here as soon as each prompt finishes")
                }

                if let active = panel.liveActiveJudgment {
                    HStack(spacing: 10) {
                        ProgressView()
                            .controlSize(.small)
                        Text("Judging \(active.condition) · \(active.promptID)")
                            .font(.callout.weight(.medium))
                        if panel.isEvaluating {
                            Button("Stop", role: .destructive) {
                                panel.cancelPairedJudge()
                            }
                            .controlSize(.small)
                            .disabled(panel.evaluationCancelRequested)
                            .help("stops after this judgment; completed judgments stay")
                        }
                    }
                    .padding(.vertical, 4)
                    .help("the paired judge is comparing this condition response against baseline")
                }

                if !panel.liveJudgments.isEmpty {
                    DisclosureGroup("Judge Results (\(panel.liveJudgments.count))") {
                        ForEach(panel.liveJudgments.prefix(20)) { judgment in
                            compactJudgmentCard(judgment)
                        }
                    }
                    .help("completed paired-judge decisions, highlighted separately from raw generations")
                }

                if !panel.isRunning && !panel.isEvaluating {
                    Button("Clear Viewer") { panel.clearLiveViewer() }
                        .help("clears only this live display; run artifacts remain on disk")
                }
            }
        }
    }

    private func liveGenerationCard(_ active: LiveStudyGeneration) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                ProgressView()
                    .controlSize(.small)
                Text("Generating \(active.condition) · \(active.promptID)")
                    .font(.callout.weight(.medium))
                Spacer()
                Text("\(active.output.split(whereSeparator: { $0.isWhitespace }).count) words")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if !active.prompt.isEmpty {
                Text(active.prompt)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .textSelection(.enabled)
            }
            Text(active.output.isEmpty ? "waiting for first tokens…" : active.output)
                .font(.body.monospaced())
                .textSelection(.enabled)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.blue.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .padding(.vertical, 4)
    }

    private func compactGenerationCard(_ generation: StudyGenerationPreview) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("\(generation.condition) · \(generation.promptID)")
                    .font(.callout.weight(.semibold))
                Spacer()
                Text(
                    "\(generation.wordCount) words · distinct-2 "
                        + generation.distinct2.formatted(.number.precision(.fractionLength(3))))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Text(generation.output + (generation.truncated ? "\n…" : ""))
                .font(.caption.monospaced())
                .lineLimit(10)
                .textSelection(.enabled)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.25))
                .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .padding(.vertical, 4)
    }

    private func compactJudgmentCard(_ judgment: StudyJudgePreview) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("\(judgment.condition) · \(judgment.promptID)")
                    .font(.callout.weight(.semibold))
                Spacer()
                Text(judgment.conditionResult)
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(judgmentHighlight(judgment).opacity(0.18))
                    .clipShape(Capsule())
            }
            Text(
                "winner \(judgment.winner) · confidence "
                    + judgment.confidence.formatted(.number.precision(.fractionLength(2)))
            )
            .font(.caption)
            .foregroundStyle(.secondary)
            Text(judgment.briefReason)
                .font(.caption)
                .textSelection(.enabled)
            if let structured = structuredFieldsSummary(judgment) {
                Text(structured)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .padding(.vertical, 5)
    }

    private func judgmentHighlight(_ judgment: StudyJudgePreview) -> Color {
        switch judgment.conditionResult {
        case "condition": .blue
        case "baseline": .orange
        default: .secondary
        }
    }

    private func structuredFieldsSummary(_ judgment: StudyJudgePreview) -> String? {
        guard let fields = judgment.structuredFields, !fields.isEmpty else { return nil }
        let summary = fields.map { "\($0.key): \($0.value.displayString)" }
            .sorted()
            .joined(separator: ", ")
        return "structured_fields [\(summary)]"
    }

    private func variantSummary(_ artifact: ModelVariantArtifact) -> String {
        "\(artifact.adapters.count) adapter\(artifact.adapters.count == 1 ? "" : "s")"
            + " · \(artifact.injections.count) injection\(artifact.injections.count == 1 ? "" : "s")"
            + " · \(artifact.promptMode)"
            + (artifact.systemPrompt == nil ? "" : " · system prompt")
    }

    /// One attached agent condition: name, composition summary, promotion
    /// provenance when present, and the non-blocking evidence notes
    /// (`AgentEvidence` — honesty chips, never a gate on attach/freeze/run).
    private func attachedAgentRow(
        _ variant: ExperimentManifest.VariantCondition,
        isDraft: Bool,
        panel: ExperimentPanel
    ) -> some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 2) {
                Text(variant.name)
                    .font(.callout)
                if let forward = variant.fromPromotion {
                    // Stage-4 declaration: no artifact exists yet — say
                    // what the arm MEANS instead of summarizing nothing.
                    Text("forward-referenced: the agent this study's sweep "
                        + "promotes for '\(forward.concept)' — resolved and "
                        + "pinned at run time on the server")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                } else {
                    Text(variantSummary(variant.artifact))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    if let promotion = variant.artifact.promotion {
                        Text(AgentEvidence.provenanceLine(for: promotion))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                    ForEach(conditionEvidenceNotes(variant)) { note in
                        evidenceNoteCaption(note)
                    }
                }
            }
            Spacer()
            if isDraft {
                Button {
                    panel.removeVariantCondition(variant.name)
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.plain)
            }
        }
    }

    private func evidenceNoteCaption(_ note: AgentEvidence.Note) -> some View {
        Text(note.displayLine)
            .font(.caption2)
            .foregroundStyle(note.severity == .caution ? Color.orange : Color.secondary)
            .textSelection(.enabled)
    }

    /// Evidence notes for one attached condition, computed against the
    /// already-scanned robustness reports plus a library-resolution check.
    private func conditionEvidenceNotes(
        _ variant: ExperimentManifest.VariantCondition
    ) -> [AgentEvidence.Note] {
        let latest = AgentEvidence.latestRobustness(
            in: robustnessEvidence,
            variantName: variant.artifact.name,
            artifactHash: variant.artifactHash)
        var notes = AgentEvidence.notes(
            for: variant.artifact,
            currentSubstrate: currentSubstrate,
            latestRobustness: latest?.report)
        if !libraryResolves(variant) {
            notes.append(AgentEvidence.artifactNotFoundNote)
        }
        return notes
    }

    /// The active workspace's engine, in the vector-sidecar substrate
    /// vocabulary the promotion birth certificate uses.
    private var currentSubstrate: String {
        service.cluster.computeTarget == .server
            ? WorkspaceScoping.serverSubstrate
            : RepEReader.substrate
    }

    /// Whether the condition's variant reference still resolves in this
    /// workspace's agent library (by recorded relative path, then by name).
    private func libraryResolves(
        _ variant: ExperimentManifest.VariantCondition
    ) -> Bool {
        service.fineTuning.variants.contains { record in
            ModelVariantStore.relativePath(for: record) == variant.artifactPath
                || record.artifact.name == variant.artifact.name
        }
    }

    /// Run Paired Judge with never-silently-gray enablement: when no run is
    /// selected it defaults to the study's latest completed run, and any
    /// remaining disabled state names the unmet condition inline.
    @ViewBuilder
    private func pairedJudgeControls(panel: ExperimentPanel) -> some View {
        let reason = panel.pairedJudgeDisabledReason
        HStack(spacing: 8) {
            Button(panel.isEvaluating ? "Judging…" : "Run Paired Judge") {
                Task { await panel.runPairedJudgeEvaluation() }
            }
            .disabled(reason != nil)
            .help(
                "evaluates a completed run by pairing each condition response with "
                    + "its same-prompt baseline, shuffling A/B labels, asking the "
                    + "current judge prompt, and writing a separate evaluate artifact")
            // A1: paired judging is cancellable between judgments.
            if panel.isEvaluating {
                ProgressView().controlSize(.small)
                Button("Stop", role: .destructive) { panel.cancelPairedJudge() }
                    .controlSize(.small)
                    .disabled(panel.evaluationCancelRequested)
                    .help(
                        "stops after the current judgment; completed judgments "
                            + "stay in judgments.jsonl, no judge report is written "
                            + "— reported as cancelled by user, never as an error")
            }
        }
        if let reason {
            Text(reason)
                .font(.caption2)
                .foregroundStyle(.secondary)
        } else if let target = panel.pairedJudgeTarget,
            panel.selectedResult?.item.id != target.id
        {
            Text("no run selected — will judge the latest completed run: \(target.directoryName)")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private func resultsView(panel: ExperimentPanel) -> some View {
        if panel.resultRuns.isEmpty {
            Text("No run artifacts for this study yet.")
                .font(.caption)
                .foregroundStyle(.secondary)
            pairedJudgeControls(panel: panel)
        } else {
            Picker(
                "Run",
                selection: Binding<String?>(
                    get: { panel.selectedResultID },
                    set: { panel.selectedResultID = $0 })
            ) {
                ForEach(panel.resultRuns) { item in
                    Text(resultPickerLabel(item))
                        .tag(String?.some(item.id))
                }
            }
            Button("Refresh Results") { panel.refreshResults() }

            if let detail = panel.selectedResult {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Run artifact: \(detail.item.path)")
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                    if let judgeArtifactDirectory = detail.judgeArtifactDirectory {
                        Text("Judge artifact: \(judgeArtifactDirectory)")
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                }

                artifactLinks(detail)

                // F10: the browser item is built ONCE per selection by the
                // panel (config.json read + sweep.csv probe) — never inline
                // here, where the body re-evaluates on every live progress
                // note.
                if let browserItem = panel.selectedResultBrowserItem {
                    RunSemanticSectionsView(service: service, item: browserItem)
                }

                pairedJudgeControls(panel: panel)

                if let judge = detail.pairedJudgeReport {
                    DisclosureGroup("Paired Judge Report") {
                        LabeledContent("Judge", value: judge.judgeModel)
                        Text(judge.sourceRunDirectory)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                        ForEach(judge.conditions, id: \.name) { condition in
                            LabeledContent(condition.name) {
                                Text(judgeConditionLine(condition))
                            }
                            if !condition.structuredSummaries.isEmpty {
                                ForEach(condition.structuredSummaries.keys.sorted(), id: \.self) { field in
                                    if let summary = condition.structuredSummaries[field] {
                                        LabeledContent(field) {
                                            Text(structuredSummaryText(summary))
                                        }
                                        .font(.caption)
                                    }
                                }
                            }
                        }
                        if !detail.judgments.isEmpty {
                            Button("Review Judge Responses") {
                                reviewSheet = ResultReviewSheet(mode: .judgments, detail: detail)
                            }
                        }
                    }
                }

                if !detail.robustnessReports.isEmpty {
                    DisclosureGroup("Agent Robustness") {
                        ForEach(detail.robustnessReports.keys.sorted(), id: \.self) { name in
                            if let report = detail.robustnessReports[name] {
                                robustnessReportView(name: name, report: report)
                            }
                        }
                    }
                }

                if let validation = detail.validationReportText {
                    DisclosureGroup("Validation Report") {
                        Text(validation)
                            .font(.caption.monospaced())
                            .textSelection(.enabled)
                    }
                }

                if !detail.generations.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Button("Review Responses (\(detail.generations.count))") {
                            reviewSheet = ResultReviewSheet(mode: .generations, detail: detail)
                        }
                        ForEach(detail.generations.prefix(5)) { generation in
                            LabeledContent("\(generation.condition) · \(generation.promptID)") {
                                Text("\(generation.wordCount) words")
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func artifactLinks(_ detail: StudyRunDetail) -> some View {
        HStack {
            if !detail.generations.isEmpty {
                Link("generations.jsonl", destination: artifactURL(detail, "generations.jsonl"))
            }
            if !detail.judgments.isEmpty {
                Link("judgments.jsonl", destination: artifactURL(detail, "judgments.jsonl"))
            }
            if detail.report != nil || detail.validationReportText != nil {
                Link("report.json", destination: artifactURL(detail, "report.json"))
            }
            if detail.pairedJudgeReport != nil {
                Link("judge-report.json", destination: artifactURL(detail, "judge-report.json"))
            }
            if !detail.robustnessReports.isEmpty {
                Link("robustness-report.json", destination: artifactURL(detail, "robustness-report.json"))
            }
        }
        .font(.caption)
    }

    private func artifactURL(_ detail: StudyRunDetail, _ filename: String) -> URL {
        let judgeFiles = Set(["judgments.jsonl", "judge-report.json"])
        if judgeFiles.contains(filename), let judgeArtifactDirectory = detail.judgeArtifactDirectory {
            return URL(filePath: judgeArtifactDirectory).appending(component: filename)
        }
        return URL(filePath: detail.item.path).appending(component: filename)
    }

    private func resultPickerLabel(_ item: StudyRunListItem) -> String {
        switch item.kind {
        case .run:
            return "run · \(item.directoryName)"
                + (item.generationCount > 0 ? " · \(item.generationCount) outputs" : "")
        case .validate:
            return "validation · \(item.directoryName)"
        case .evaluate:
            return "judge · \(item.directoryName)"
        case .other:
            return "artifact · \(item.directoryName)"
        }
    }

    private func judgeConditionLine(_ condition: PairedJudgeReportView.Condition) -> String {
        let confidence = condition.meanConfidence.formatted(.number.precision(.fractionLength(2)))
        return "condition \(condition.conditionWins) · baseline \(condition.baselineWins)"
            + " · ties \(condition.ties) · confidence \(confidence)"
    }

    private func robustnessReportView(name: String, report: VariantRobustnessReport) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(name)
                .font(.callout.weight(.semibold))
            LabeledContent(
                "Capability",
                value:
                    percent(report.variantBatteryAccuracy) + " variant · "
                    + percent(report.baselineBatteryAccuracy) + " baseline")
            LabeledContent(
                "Distinct-2",
                value:
                    report.meanVariantDistinct2.formatted(.number.precision(.fractionLength(3)))
                    + " variant · "
                    + report.meanBaselineDistinct2.formatted(.number.precision(.fractionLength(3)))
                    + " baseline")
            if report.judgeModel != nil {
                let counts = Dictionary(grouping: report.coherenceItems.compactMap(\.judgeResult)) { $0 }
                    .mapValues(\.count)
                Text(
                    "Judge: baseline \(counts["baseline"] ?? 0) · variant \(counts["variant"] ?? 0) · ties \(counts["tie"] ?? 0)"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                ForEach(report.coherenceItems.filter { $0.judge != nil }, id: \.index) { item in
                    if let judge = item.judge {
                        DisclosureGroup("Judge \(item.index): \(item.judgeResult ?? judge.winner)") {
                            Text(judge.briefReason)
                                .font(.caption)
                                .textSelection(.enabled)
                        }
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
        .padding(.vertical, 6)
    }

    private func percent(_ value: Float) -> String {
        value.formatted(.percent.precision(.fractionLength(0)))
    }

    private func structuredSummaryText(_ summary: StructuredFieldSummaryView) -> String {
        var parts = ["n \(summary.count)"]
        if let mean = summary.numericMean {
            parts.append("mean \(mean.formatted(.number.precision(.fractionLength(3))))")
        }
        if let trueCount = summary.trueCount, let falseCount = summary.falseCount {
            parts.append("true \(trueCount)")
            parts.append("false \(falseCount)")
        }
        if let stringCounts = summary.stringCounts, !stringCounts.isEmpty {
            let counts = stringCounts.map { "\($0.key): \($0.value)" }
                .sorted()
                .joined(separator: ", ")
            parts.append(counts)
        }
        return parts.joined(separator: " · ")
    }
}

// MARK: - Persistent notices bell (App gap A15)

/// The notices bell for panel section headers (Studies, Agents): opens the
/// shared per-workspace `PanelNotices` feed in a popover. Unseen errors
/// badge the bell until the feed is viewed; the feed itself survives — it is
/// the append-only ring persisted to `.steerlab/notices.json`, so an
/// overnight failure is still consultable in the morning.
struct NoticesBellButton: View {
    @State private var showFeed = false

    var body: some View {
        let notices = PanelNotices.shared
        Button {
            showFeed = true
            notices.markViewed()
        } label: {
            Image(
                systemName: notices.hasUnseenErrors
                    ? "bell.badge.fill" : "bell")
                .foregroundStyle(
                    notices.hasUnseenErrors
                        ? AnyShapeStyle(.red) : AnyShapeStyle(.secondary))
                .imageScale(.medium)
        }
        .buttonStyle(.plain)
        .help(
            "panel notices — every Studies/Agents status event, kept (last "
                + "\(PanelNotices.capacity)) and persisted per workspace; "
                + "errors badge the bell until viewed")
        .popover(isPresented: $showFeed, arrowEdge: .bottom) {
            NoticesFeedView()
        }
    }
}

/// The feed popover: newest first, severity icons, source + timestamp, and a
/// Clear action. Read-only over the store — the panels append, this renders.
struct NoticesFeedView: View {
    var body: some View {
        let notices = PanelNotices.shared
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Notices")
                    .font(.headline)
                Spacer()
                Button("Clear") { notices.clear() }
                    .controlSize(.small)
                    .disabled(notices.notices.isEmpty)
                    .help("empties the notices ring (and its persisted file)")
            }
            if notices.notices.isEmpty {
                Text("No notices yet — Studies and Agents events land here "
                    + "and persist across the session.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(notices.recentFirst) { notice in
                            noticeRow(notice)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 360)
            }
        }
        .padding(12)
        .frame(width: 460)
    }

    private func noticeRow(_ notice: PanelNotice) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: notice.severity.symbolName)
                .foregroundStyle(severityColor(notice.severity))
                .imageScale(.small)
            VStack(alignment: .leading, spacing: 1) {
                Text(notice.message)
                    .font(.caption)
                    .textSelection(.enabled)
                Text(
                    "\(notice.source) · "
                        + notice.timestamp.formatted(
                            date: .abbreviated, time: .standard))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func severityColor(_ severity: PanelNotice.Severity) -> Color {
        switch severity {
        case .info: .secondary
        case .success: .green
        case .warning: .orange
        case .error: .red
        }
    }
}

private struct ResultReviewSheet: Identifiable {
    enum Mode {
        case generations
        case judgments
    }

    let id = UUID()
    let mode: Mode
    let detail: StudyRunDetail

    var title: String {
        switch mode {
        case .generations: "Generated Responses"
        case .judgments: "Judge Responses"
        }
    }
}

private struct ResultReviewWindow: View {
    let sheet: ResultReviewSheet

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(sheet.title)
                        .font(.title2.weight(.semibold))
                    Text(sheet.detail.item.directoryName)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                Spacer()
            }

            Divider()

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    switch sheet.mode {
                    case .generations:
                        ForEach(sheet.detail.generations) { generation in
                            generationCard(generation)
                        }
                    case .judgments:
                        ForEach(sheet.detail.judgments) { judgment in
                            judgmentCard(judgment)
                        }
                    }
                }
                .padding(.trailing, 8)
            }
        }
        .padding(18)
        .frame(minWidth: 760, minHeight: 560)
    }

    private func generationCard(_ generation: StudyGenerationPreview) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("\(generation.condition) · \(generation.promptID)")
                    .font(.headline)
                Spacer()
                Text(
                    "\(generation.wordCount) words · distinct-2 "
                        + generation.distinct2.formatted(.number.precision(.fractionLength(3))))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Text(generation.prompt)
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            Text(generation.output + (generation.truncated ? "\n…" : ""))
                .font(.body.monospaced())
                .textSelection(.enabled)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.35))
                .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .padding(12)
        .background(.background)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
    }

    private func judgmentCard(_ judgment: StudyJudgePreview) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("\(judgment.condition) · \(judgment.promptID)")
                    .font(.headline)
                Spacer()
                Text(
                    "winner \(judgment.winner) · \(judgment.conditionResult) · confidence "
                        + judgment.confidence.formatted(.number.precision(.fractionLength(2))))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Text("baseline \(judgment.baselineWas) · condition \(judgment.conditionWas)")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(judgment.prompt)
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            Text(judgment.briefReason)
                .textSelection(.enabled)
            if let scores = scoreSummary(judgment) {
                Text(scores)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            if let structured = structuredFieldsSummary(judgment) {
                Text(structured)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            DisclosureGroup("Raw judge JSON") {
                Text(judgment.rawJSON)
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
            }
        }
        .padding(12)
        .background(.background)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
    }

    private func scoreSummary(_ judgment: StudyJudgePreview) -> String? {
        let a = (judgment.aScores ?? [:]).map { "\($0.key): \($0.value)" }.sorted().joined(separator: ", ")
        let b = (judgment.bScores ?? [:]).map { "\($0.key): \($0.value)" }.sorted().joined(separator: ", ")
        guard !a.isEmpty || !b.isEmpty else { return nil }
        return "A [\(a.isEmpty ? "no scores" : a)] · B [\(b.isEmpty ? "no scores" : b)]"
    }

    private func structuredFieldsSummary(_ judgment: StudyJudgePreview) -> String? {
        guard let fields = judgment.structuredFields, !fields.isEmpty else { return nil }
        let summary = fields.map { "\($0.key): \($0.value.displayString)" }
            .sorted()
            .joined(separator: ", ")
        return "structured_fields [\(summary)]"
    }

    private func structuredSummaryText(_ summary: StructuredFieldSummaryView) -> String {
        var parts = ["n \(summary.count)"]
        if let mean = summary.numericMean {
            parts.append("mean \(mean.formatted(.number.precision(.fractionLength(3))))")
        }
        if let trueCount = summary.trueCount, let falseCount = summary.falseCount {
            parts.append("true \(trueCount)")
            parts.append("false \(falseCount)")
        }
        if let stringCounts = summary.stringCounts, !stringCounts.isEmpty {
            let counts = stringCounts.map { "\($0.key): \($0.value)" }
                .sorted()
                .joined(separator: ", ")
            parts.append(counts)
        }
        return parts.joined(separator: " · ")
    }
}

/// Import JSONL… sheet for the Input Data section: paste (text area) or
/// choose a file, watch the live parse preview — record count, how many
/// records carry `options`/`target`, or the FIRST error with its line
/// number — and import only when every line parses. Garbage is refused,
/// never coerced into prompt text. All parsing rules live in
/// `TaskPromptsImport` (ExperimentKit, unit-tested); this sheet renders
/// them.
private struct ImportJSONLSheet: View {
    @Binding var text: String
    /// Workspace-relative destination the import writes to (nil when no
    /// study is selected — the import button explains instead of failing).
    let destination: String?
    /// Runs the panel's import (write → set as prompts file → pin hash);
    /// the Bool is the "Replace the existing file" checkbox (the explicit
    /// affordance — without it a differing existing file refuses);
    /// true = landed, dismiss.
    let onImport: (String, Bool) -> Bool
    /// The panel's task-prompts status line (import refusals surface there).
    let statusLine: () -> String?

    @Environment(\.dismiss) private var dismiss
    @State private var showFilePicker = false
    @State private var fileReadError: String?
    @State private var replaceExisting = false

    private var preview: TaskPromptsImport.Outcome {
        TaskPromptsImport.preview(text)
    }

    private var importable: Bool {
        if case .preview = preview { return destination != nil }
        return false
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Import JSONL task prompts")
                .font(.headline)
            // Required record structure, VISIBLE in the sheet (2026-07-20
            // researcher round, item 2b) — not hover-only.
            Text(StudyInfo.importRecordStructure)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            TextEditor(text: $text)
                .font(.system(.caption, design: .monospaced))
                .frame(minWidth: 480, minHeight: 220)

            HStack {
                Button("Choose File…") { showFilePicker = true }
                    .help("read a .jsonl file into the text area above")
                if let fileReadError {
                    Text(fileReadError)
                        .font(.caption2)
                        .foregroundStyle(.red)
                }
            }

            previewRow

            if let destination {
                // Destination semantics, visible and TRUE to the code
                // (2026-07-20 researcher round, item 2c).
                Text(StudyInfo.importDestination(destination))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                // The study-pack write rule's ONE sanctioned exception:
                // replacing a differing existing file is an explicit,
                // per-import choice — never a default.
                Toggle(
                    "Replace the existing file if its contents differ",
                    isOn: $replaceExisting)
                    .font(.caption)
                    .help(
                        "without this, importing refuses when "
                        + "\(destination) already exists with different "
                        + "contents — nothing is overwritten silently")
            } else {
                Text("select a draft study first — the import pins into the "
                    + "selected draft's manifest")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
            if let status = statusLine() {
                Text(status)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Import & Pin") {
                    if onImport(text, replaceExisting) { dismiss() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!importable)
            }
        }
        .padding(16)
        .fileImporter(
            isPresented: $showFilePicker,
            allowedContentTypes: [
                UTType(filenameExtension: "jsonl") ?? .plainText, .json, .plainText,
            ]
        ) { result in
            switch result {
            case .success(let url):
                let accessing = url.startAccessingSecurityScopedResource()
                defer { if accessing { url.stopAccessingSecurityScopedResource() } }
                do {
                    text = String(decoding: try Data(contentsOf: url), as: UTF8.self)
                    fileReadError = nil
                } catch {
                    fileReadError = "could not read \(url.lastPathComponent): \(error.localizedDescription)"
                }
            case .failure(let error):
                fileReadError = error.localizedDescription
            }
        }
    }

    @ViewBuilder
    private var previewRow: some View {
        switch preview {
        case .empty:
            Label("nothing to import yet — paste JSONL records above",
                systemImage: "info.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
        case .preview(let parsed):
            Label(parsed.summaryLine, systemImage: "checkmark.circle")
                .font(.caption)
                .foregroundStyle(.green)
        case .failure(let line, let message):
            Label("line \(line): \(message) — fix it; garbage is refused, "
                + "never imported as prompt text",
                systemImage: "xmark.octagon")
                .font(.caption)
                .foregroundStyle(.red)
                .textSelection(.enabled)
        }
    }
}

/// What Rename was opened on. Captured at open time so the sheet never
/// re-reads the store while it is up.
private struct RenameStudySheet: Identifiable {
    let id = UUID()
    let name: String
    let status: ExperimentManifest.Status
    let label: String
    /// Run directories already stamping this study's canonical name — the
    /// number a draft rename would strand (runs are immutable and are never
    /// rewritten).
    let runsStamped: Int

    var isDraft: Bool { status == .draft }
}

/// The ONE rename affordance. A draft offers both effects — canonical name
/// and display label — in a single action; a frozen or completed study
/// offers only the label, because its canonical name is hashed into the
/// manifest and stamped into every run's provenance.
private struct RenameStudyWindow: View {
    let sheet: RenameStudySheet
    let panel: ExperimentPanel

    @Environment(\.dismiss) private var dismiss
    @State private var canonicalName = ""
    @State private var label = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Rename Study")
                    .font(.title2.weight(.semibold))
                Text(sheet.name)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }

            Divider()

            Form {
                if sheet.isDraft {
                    Section("Name") {
                        TextField("study name", text: $canonicalName)
                            .font(.body.monospaced())
                        Text(Self.canonicalNameHelp)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                        if sheet.runsStamped > 0 {
                            Label(strandedRunsNote, systemImage: "exclamationmark.triangle")
                                .font(.caption)
                                .foregroundStyle(.orange)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }

                Section("Display label") {
                    TextField("display label (optional)", text: $label)
                    Text(labelHelp)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .formStyle(.grouped)

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                Button("Rename") {
                    panel.renameSelected(
                        canonicalName: sheet.isDraft ? canonicalName : nil,
                        label: label)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!hasChange)
            }
        }
        .padding(18)
        .frame(minWidth: 480, minHeight: sheet.isDraft ? 440 : 320)
        .onAppear {
            canonicalName = sheet.name
            label = sheet.label
        }
    }

    private var hasChange: Bool {
        (sheet.isDraft && canonicalName != sheet.name) || label != sheet.label
    }

    // Long strings live outside the body — interpolating them inline blows
    // the SwiftUI type-checker budget.
    private static let canonicalNameHelp =
        "lowercase letters, digits and hyphens; anything else is dropped. This "
        + "IS the experiments/<name>/ directory and the name the CLI takes."

    private var strandedRunsNote: String {
        let plural = sheet.runsStamped == 1 ? "run" : "runs"
        return "\(sheet.runsStamped) existing \(plural) stamp '\(sheet.name)'. Runs "
            + "are immutable and a rename never rewrites them — they will no longer "
            + "list under this study."
    }

    private var labelHelp: String {
        guard !sheet.isDraft else {
            return "shown first in study lists; the name above stays the identity "
                + "everywhere else."
        }
        return "shown first in study lists. This study is \(sheet.status.rawValue): "
            + "its name is hashed into the frozen manifest and stamped into every "
            + "run, so the canonical id stays '\(sheet.name)'. A label is stored "
            + "beside the manifest and moves no hash."
    }
}
