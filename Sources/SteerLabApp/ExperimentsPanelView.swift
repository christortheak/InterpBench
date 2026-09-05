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

                StudySetupSection(
                    manifest: manifest, panel: panel,
                    substrateLabel: service.cluster.substrateLabel)

                // ONE Conditions section, type-dependent (2026-07-19 second
                // pass): the arms of the study — agents for a comparison,
                // the scenario for multi-agent, the perturbation policy for
                // a confirmation. Concept studies get their derivation
                // machinery in the sections that follow.
                StudyArmsSection(manifest: manifest, panel: panel,
                    robustnessEvidence: robustnessEvidence, currentSubstrate: currentSubstrate,
                    availableVariants: service.fineTuning.variants)

                // WHO sits in each seat of the chosen scenario. Only
                // multi-agent studies have seats, and a scenario that declares
                // none has nothing to show.
                if panel.studyKind == .multiAgent {
                    StudySeatsSection(manifest: manifest, panel: panel)
                }

                if panel.studyKind == .modelOutput,
                    panel.studyFocus == .conceptStudy
                {
                    StudyConceptsSection(manifest: manifest, panel: panel)
                    StudyInjectionConditionsSection(manifest: manifest, panel: panel)
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
                                StudyTaskPromptsEditor(
                                    manifest: manifest, panel: panel,
                                    showImportJSONL: $showImportJSONL, importJSONLText: $importJSONLText))
                        }
                        : nil)

                StudyEvaluationSection(manifest: manifest, panel: panel) {
                    JudgingSectionControls(service: service, manifest: manifest, panel: panel)
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

    /// The active workspace's engine, in the vector-sidecar substrate
    /// vocabulary the promotion birth certificate uses.
    private var currentSubstrate: String {
        service.cluster.computeTarget == .server
            ? WorkspaceScoping.serverSubstrate
            : RepEReader.substrate
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
