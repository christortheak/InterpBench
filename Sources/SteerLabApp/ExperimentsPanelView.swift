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
    /// The one Rename affordance, offered for every study whatever its
    /// status: a draft renames for real, a frozen/complete study takes a
    /// display label (see `ExperimentStore.rename` for why the two differ).
    @State private var renameSheet: RenameStudySheet?
    /// Bound expansion state for the "Remote options" disclosure so
    /// cross-links (Optimizations' preconfigured sweep) can open it directly.
    @State private var runOnServerExpanded = false
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
        @Bindable var draft = panel.draft
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
                                .help(StudyControlCopy.duplicateHelp)
                            Button("Delete…", role: .destructive) {
                                deleteDraftRunCount = ExperimentStore.runsStamped(
                                    experimentName: manifest.name)
                                confirmDeleteDraft = true
                            }
                            .disabled(panel.deleteSelectedStudyRefusal != nil)
                            .help(panel.deleteSelectedStudyRefusal ?? StudyControlCopy.deleteStudyHelp)
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
                            TextField("new study name", text: $draft.newName)
                                .help("creates a draft pinned to the currently selected model")
                            TextField(
                                "question or purpose", text: $draft.newDescription,
                                axis: .vertical
                            )
                            .lineLimit(1 ... 3)
                            .help("short domain-neutral purpose for the draft protocol")
                            TextField(
                                "model revision (optional commit hash)",
                                text: $draft.newRevision
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
                        text: $draft.protocolDescription,
                        axis: .vertical
                    )
                    .lineLimit(1 ... 3)
                    .disabled(manifest.status != .draft)
                    .help("what this study asks; not tied to any domain")

                    TextField(
                        "task: what the model or agents will do",
                        text: $draft.taskDescription,
                        axis: .vertical
                    )
                    .lineLimit(2 ... 5)
                    .disabled(manifest.status != .draft)
                    .help(
                        "examples: write an opinion, play a game, answer advice "
                            + "prompts, allocate resources, classify scenarios")

                    TextField(
                        "outcome measures",
                        text: $draft.outcomeMeasures,
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
                            selection: $draft.studyBaseModelID
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
                            selection: $draft.studyDtypeField)

                        HStack(spacing: 6) {
                            Picker("Baseline prompt mode", selection: $draft.promptMode) {
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
                            text: $draft.systemPrompt, axis: .vertical)
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

                        Picker("Reasoning effort", selection: $draft.reasoningEffort) {
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
                                value: $draft.reasoningMaxTokens,
                                format: .number)
                            .disabled(manifest.status != .draft)
                            .help("The reasoning block's own token cap (up to </think>); "
                                + "Max tokens is then the answer budget. Required.")
                        }
                    }

                    TemperatureRow(value: $draft.runTemperature)
                        .disabled(manifest.status != .draft)

                    LabeledContent("Max tokens") {
                        TextField(
                            "", value: $draft.runMaxTokens,
                            format: .number.grouping(.never)
                        )
                        .frame(width: 72)
                        .multilineTextAlignment(.leading)
                    }
                    .disabled(manifest.status != .draft)
                    .help("study-wide per-response token cap for baseline and agents")

                    if panel.studyKind == .multiAgent {
                        Picker("Scenario",
                               selection: $draft.selectedMultiAgentScenarioID) {
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
                            isOn: $draft.multiAgentIncludeBaseline
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
                            Picker("Funnel phase", selection: $draft.phaseField) {
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
                        StudyPreparationControlsView(service: service, manifest: manifest,
                            pendingModelJob: $pendingModelJob, runOnServerExpanded: $runOnServerExpanded)
                    }
                    StudyRunControlsView(service: service, manifest: manifest,
                        runOnServerExpanded: $runOnServerExpanded, pendingModelJob: $pendingModelJob)
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

                StudyLiveRunView(jobs: panel.localJobs)

                // ONE results area (2026-07-19 second pass): what used to
                // be four sibling sections — Results, Server Runs,
                // Pipelines, Recent Server Jobs — whose differences the
                // researcher had to guess. Same content, one roof,
                // subheaded by WHERE the runs live and at what granularity.
                Section("Runs & Results") {
                    if !panel.recentServerJobs.isEmpty {
                        StudyRecentJobsView(jobs: panel.remoteJobs,
                            resume: { await panel.resubmitRemoteJob($0) },
                            importEvidence: { await panel.importEvidence(fromJobID: $0) },
                            refresh: { await panel.refreshRecentServerJobs() })
                    }
                    Text("Run reports — this workspace")
                        .font(.caption.bold())
                        .padding(.top, 4)
                    StudyResultsView(service: service, results: panel.results,
                        refresh: { panel.refreshResults() }) {
                        pairedJudgeControls(panel: panel)
                    }
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
                .help(refusal ?? StudyControlCopy.saveBackHelp)
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
            .help(StudyControlCopy.saveAsNewDesignHelp)
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

    // MARK: Starting a study from a design

    /// The first control in the new-study flow: what this study starts FROM.
    ///
    /// The design LIBRARY is the Templates tab; this is only the choice, so
    /// that "new study" is one flow with two beginnings rather than two flows
    /// the researcher has to know to pick between.
    @ViewBuilder
    private func newStudyDesignPicker(panel: ExperimentPanel) -> some View {
        @Bindable var panel = panel
        @Bindable var draft = panel.draft
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

    private func serverRunCaption(_ verb: String) -> String {
        "\(verb) runs on \(service.cluster.substrateLabel) as a durable job — "
            + "reconnect from Compute"
    }

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
        .help(StudyControlCopy.studyDtypeHelp)
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
        @Bindable var draft = panel.draft
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
        @Bindable var draft = panel.draft
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
            Picker("Attach Concept…", selection: $draft.attachConceptName) {
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
                Picker("", selection: $draft.attachMethod) {
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
                    choice: $draft.attachReadingPositionChoice,
                    parameter: $draft.attachReadingPositionParameter,
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
                    choice: $draft.attachRendering,
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
                Picker("reference", selection: $draft.attachReferenceName) {
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
                    text: $draft.attachCorpusText)
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
        @Bindable var draft = panel.draft
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
        @Bindable var draft = panel.draft
        let isDraft = manifest.status == .draft
        let isPanel = panel.studyKind == .multiAgent
        LabeledContent(isPanel ? "Play-throughs (transcripts)" : "Samples per item") {
            HStack(spacing: 6) {
                TextField(
                    "", value: $draft.samplesPerItemField,
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
            Picker("Seed policy", selection: $draft.seedPolicyField) {
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
        @Bindable var draft = panel.draft
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
                    isOn: $draft.acknowledgeUnequalOptionLengthsField
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
        @Bindable var draft = panel.draft
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
                        Picker("Add agent", selection: $draft.selectedVariantToAddID) {
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
                        .help(StudyControlCopy.seatPickerHelp)
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
                            .help(refusal ?? StudyControlCopy.saveCastingHelp)
                        Button("Create permuted siblings…") {
                            panel.startPermutedSiblings()
                        }
                        .disabled(casting.form != .cast)
                        .help(
                            casting.form == .cast
                                ? StudyControlCopy.permutedSiblingsHelp
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
        @Bindable var draft = panel.draft
        HStack(spacing: 6) {
            TextField("task prompts JSONL", text: $draft.taskPromptsFile)
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
        TextEditor(text: $draft.taskPromptsText)
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
        @Bindable var draft = panel.draft
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
                        .help(StudyControlCopy.outcomeModeHelp)
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
            .help(decision.target == .server ? StudyControlCopy.remoteFreezeHelp : StudyControlCopy.freezeHelp)
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
        @Bindable var draft = panel.draft
        return VStack(alignment: .leading, spacing: 6) {
            HStack {
                Picker("Confirm agent", selection: $draft.confirmAgentID) {
                    Text("select…").tag(ModelVariantRecord.ID?.none)
                    ForEach(panel.confirmableAgents) { record in
                        Text(confirmAgentLabel(record))
                            .tag(ModelVariantRecord.ID?.some(record.id))
                    }
                }
                TextField("strength deltas (α)", text: $draft.confirmDeltasText)
                    .frame(width: 120)
                    .help(
                        "comma-separated positive strength offsets around the "
                            + "anchor α, e.g. 0.2, 0.5")
                Toggle(
                    "random-direction control",
                    isOn: $draft.confirmIncludeControl)
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

}
