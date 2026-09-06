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
    private var panel: ExperimentPanel { service.experiments }

    var body: some View {
        @Bindable var panel = service.experiments
        Form {
            StudyManagementSection(panel: panel, openTemplates: openTemplates)

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
                            let request = panel.submission.snapshot(verb: "pipeline")
                            let submit = panel.pipelineSubmissionAction(manifest: manifest, request: request)
                            ModelJobGPUGate.submit(
                                "study pipeline", service: service,
                                pending: $pendingModelJob,
                                bundleOptions:
                                    ModelJobSubmissionPreflight.BundleOptions(
                                        executor: request.executor,
                                        gres: request.gres,
                                        verb: request.verb,
                                        dryRun: request.dryRun),
                                fixOptions: {
                                    panel.applyGPUAllocationFix()
                                    runOnServerExpanded = true
                                }
                            ) { await submit() }
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
                    StudyDesignActionsView(
                        manifest: manifest, management: panel.management,
                        openTemplates: openTemplates)
                }

                StudyIssuesSection(manifest: manifest, panel: panel)

                Section {
                    if manifest.status == .draft {
                        StudyFreezeControlsView(
                            manifest: manifest, coordinator: panel.freezeCoordinator,
                            inputs: FreezeRouting.Inputs(
                                activeWorkspaceIsServer: panel.isServerWorkspace,
                                serverConnected: service.cluster.capabilities != nil,
                                serverLabel: service.cluster.substrateLabel,
                                serverHasSelectedStudy: panel.serverHasSelectedStudy,
                                workspaceKnownUnpaired: panel.isKnownUnpairedServerWorkspace),
                            violations: panel.violations,
                            freezeLocally: { panel.freeze() },
                            freezeOnServer: { await panel.freezeOnActiveServer() },
                            syncDraft: { await panel.pushManifestToActiveServer() })
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
                        StudyServerRunsView(runs: panel.remoteRuns,
                            substrateLabel: service.cluster.substrateLabel,
                            refresh: { await panel.refreshRemoteRuns() })
                    }
                    if service.cluster.computeTarget == .server || !panel.pipelines.localPipelineRuns.isEmpty {
                        StudyPipelinesView(pipelines: panel.pipelines,
                            substrateLabel: service.cluster.substrateLabel,
                            refresh: { await panel.refreshPipelineRuns() },
                            duplicateStudy: { panel.duplicateSelected() })
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
        // Item 2 (cluster-testing): the shared no-GPU-session warning for
        // this panel's model-running server submissions (run/sweep bundle,
        // validate, extract).
        .modelJobGPUWarning(pending: $pendingModelJob, service: service)
        .formStyle(.grouped)
        .onAppear {
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

    /// Change key for the residency preflight: selection + active substrate.
    private var residencyTaskKey: String {
        let name = panel.selectedName ?? ""
        let substrate = service.cluster.substrateLabel
        return "\(name)|\(substrate)|\(panel.isServerWorkspace)"
    }

    // Long strings live outside the body — interpolating them inline blows
    // the SwiftUI type-checker budget (same fix as ChatView.provenanceLine).

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
