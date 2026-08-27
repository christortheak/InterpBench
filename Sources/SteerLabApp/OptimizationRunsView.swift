import ExperimentKit
import SwiftUI

/// Optimization runs (the research funnel's "screen" stage) — a LENS over
/// experiments whose manifests carry a sweep spec (or whose conditions
/// already carry sweep-selection provenance). NOT a new object type: the
/// firewall machinery stays experiment-scoped; this surface renders the
/// declared criterion, the newest sweep run's grid, the recommendation
/// provenance, and the Create Agent (promote) edge. Embedded as the
/// Agents → Optimizations region (docs/AGENT_CREATION_SWEEP_UI_RECOMMENDATION.md).
///
/// Substrate-aware SOURCE MODEL (the compute target decides where a sweep
/// EXECUTES; declaring is always a local manifest edit):
/// - Local compute: the local optimizations list, declarable and editable.
/// - PAIRED server (`/api/info` root == this workspace): the local and
///   server trees are the same files — ONE list, the local editable view,
///   with Run Sweep/Promote executing on the server into the same tree.
/// - UNPAIRED (or pairing-unknown) server: BOTH lists, labeled by source —
///   "this workspace" (declarable/editable; run via Submit Bundle in
///   Studies) and the server's own optimizations (read-only; run/promote there).
/// All server traffic goes through `ExperimentPanel`; this view never
/// touches the cluster client.
struct OptimizationRunsView: View {
    @Bindable var service: ChatService
    let navigate: (WorkbenchSection) -> Void
    /// An optimization run name to select on next refresh (set by the
    /// New Agent → Optimize flow so declaring lands on the new run);
    /// consumed once the run appears in the list.
    @Binding var pendingSelection: String?

    @State private var selectedRef: OptimizationRef?
    @State private var sweepRun: SweepRunCatalog.SweepRun?
    /// Item 2 (cluster-testing): a sweep submission parked while the shared
    /// no-GPU-session dialog asks.
    @State private var pendingModelJob: PendingModelJob?
    @State private var selectedCell: SelectedCell?
    @State private var overridePromotion: OverridePromotion?
    @State private var showDeclareSheet = false
    /// Experiments named by some local agent's promotion birth certificate —
    /// the lifecycle strip's "Promoted" evidence (local tree; on a PAIRED
    /// server that IS the server's tree too. The server's variant listing
    /// carries no promotion block, so server-source optimizations stay tri-state).
    ///
    /// Read off the agent library's row index, which `refreshOptimizations`
    /// rescans asynchronously — this used to be a second full
    /// `ModelVariantStore.scan()` on the main thread just to collect these
    /// names. Previous rows stay visible while a rescan is in flight.
    private var promotedExperiments: Set<String> {
        Set(service.fineTuning.agentIndex.compactMap(\.promotedExperiment))
    }

    private var panel: ExperimentPanel { service.experiments }

    private var isServer: Bool { panel.isServerWorkspace }

    /// Pairing verdict for the active server (nil in the Local workspace).
    private var pairing: WorkspaceScoping.ServerPairing? {
        service.cluster.activeServerPairing
    }

    private var isPaired: Bool { pairing == .paired }

    /// ONE scoping rule (WorkspaceScoping.artifactListPresentation): the
    /// separate server-source list appears whenever the server is NOT known
    /// to share this workspace's tree (unpaired, or pairing unknown), and a
    /// CONFIRMED mismatch additionally shows the standing banner.
    private var presentation: WorkspaceScoping.ArtifactListPresentation {
        service.cluster.artifactListPresentation
    }

    private var showsServerList: Bool {
        if case .serverAuthoritative = presentation { return true }
        return false
    }

    private var substrate: String { service.cluster.substrateLabel }

    /// One optimization, tagged by the tree it was read from.
    private struct OptimizationItem: Identifiable {
        let source: OptimizationSource
        let name: String
        let statusLabel: String
        let isDraft: Bool
        let modelID: String
        let conceptCount: Int
        /// Declared sweep spec: the local manifest's, or the server detail's
        /// verbatim top-level `sweep` (older servers omit it — nil).
        let sweep: ExperimentManifest.SweepSpec?
        /// Selection provenance by condition name (`<concept>-recommended`).
        let selections: [String: ExperimentManifest.SelectionProvenance]
        var ref: OptimizationRef { OptimizationRef(source: source, name: name) }
        var id: String { "\(source.rawValue):\(name)" }
    }

    /// The lens predicate, local arm: a declared sweep spec, or
    /// provenance-bearing conditions from an already-executed sweep.
    private var localOptimizationManifests: [ExperimentManifest] {
        panel.experiments.filter { manifest in
            manifest.sweep != nil
                || manifest.conditions.contains { $0.selection != nil }
        }
    }

    /// Local optimizations are visible in EVERY compute mode — hiding them in
    /// server mode was the hidden-state bug this layout replaces.
    private var localOptimizationItems: [OptimizationItem] {
        localOptimizationManifests.map { manifest in
            OptimizationItem(
                source: .local,
                name: manifest.name,
                statusLabel: manifest.status.rawValue,
                isDraft: manifest.status == .draft,
                modelID: manifest.modelID,
                conceptCount: manifest.concepts.count,
                sweep: manifest.sweep,
                selections: Dictionary(
                    manifest.conditions.compactMap { condition in
                        condition.selection.map { (condition.name, $0) }
                    },
                    uniquingKeysWith: { first, _ in first }))
        }
    }

    private var serverOptimizationItems: [OptimizationItem] {
        panel.remoteOptimizations.map { record in
            OptimizationItem(
                source: .server,
                name: record.name,
                statusLabel: record.status ?? "?",
                isDraft: record.status == "draft",
                modelID: record.modelID ?? "?",
                conceptCount: record.concepts?.count ?? 0,
                sweep: record.sweep,
                selections: Dictionary(
                    (record.conditions ?? []).compactMap { condition in
                        condition.selection.map { (condition.name, $0) }
                    },
                    uniquingKeysWith: { first, _ in first }))
        }
    }

    private var optimizations: [OptimizationItem] {
        showsServerList ? localOptimizationItems + serverOptimizationItems : localOptimizationItems
    }

    private var selectedOptimization: OptimizationItem? {
        optimizations.first { $0.ref == selectedRef }
    }

    var body: some View {
        Form {
            if presentation.showsMismatchBanner {
                Section {
                    WorkspaceMismatchBanner(cluster: service.cluster)
                }
            }
            localListSection
            if showsServerList {
                serverListSection
            }
            if let optimization = selectedOptimization {
                lifecycleSection(optimization)
                criterionSection(optimization)
                sweepSpecSection(optimization)
                gridSections(optimization)
                sweepGridSection(optimization)
                recommendationsSection(optimization)
            }
            if let status = panel.status {
                Section {
                    Text(status)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
        }
        .formStyle(.grouped)
        // Item 2 (cluster-testing): no-GPU-session warning before a
        // server-executed sweep submission.
        .modelJobGPUWarning(pending: $pendingModelJob, service: service)
        .onAppear { refreshOptimizations() }
        .onChange(of: service.cluster.activeWorkspace) { refreshOptimizations() }
        // Pairing knowledge arrives asynchronously with /api/info — the
        // source layout (one shared list vs two labeled lists) follows it.
        .onChange(of: service.cluster.remoteInfo) { refreshOptimizations() }
        .onChange(of: selectedRef) { reloadSweepRun() }
        .sheet(item: $overridePromotion) { promotion in
            OverridePromotionSheet(promotion: promotion) { reason in
                panel.promote(
                    experimentName: promotion.experiment,
                    concept: promotion.concept,
                    cell: (layer: promotion.layer, alpha: promotion.alpha),
                    overrideReason: reason,
                    route: promotion.mintsOnServer ? .activeServer : .local,
                    // Pins CAPTURED when the sheet opened. Re-deriving them
                    // here would silently promote unpinned if the loaded run
                    // changed or cleared while the sheet was up.
                    pins: promotion.pins)
            }
        }
        .sheet(isPresented: $showDeclareSheet) {
            DeclareOptimizationSheet(
                drafts: declarableDraftNames,
                declare: { name, objective in
                    declareOptimization(named: name, objective: objective)
                },
                openStudies: { navigate(.studies) })
        }
    }

    /// Draft studies with no declared sweep yet — the declare sheet's
    /// candidates. Declaring is a LOCAL manifest edit and is available in
    /// EVERY compute mode; the compute target only decides where the sweep
    /// later executes.
    private var declarableDraftNames: [String] {
        panel.experiments
            .filter { $0.status == .draft && $0.sweep == nil }
            .map(\.name)
    }

    /// Declaring writes the DEFAULT grid with the EXPLICITLY CHOSEN
    /// objective — an optimization's selection rule is declared data, never
    /// an implied fallback. The spec editor then opens on the new run.
    /// (The engine rule that an ABSENT selection block resolves to the
    /// historical markerDensity default is a backward-compat contract in
    /// `SweepSelectionRule` and is untouched here.)
    private func declareOptimization(named name: String, objective: String) {
        if Self.declareOptimization(named: name, objective: objective, panel: panel) {
            refreshOptimizations()
            selectedRef = OptimizationRef(source: .local, name: name)
        }
    }

    /// The one declare rule, callable from outside the view too. Writes the
    /// default grid with the CHOSEN objective metric; `setSweepSpec` runs
    /// the engine's criterion validation (judgeScore needs the draft's
    /// rubric + judge pins; logprobShift needs a loadable choice file) and
    /// refuses loudly into `panel.status` when the instrument is missing.
    static func declareOptimization(
        named name: String, objective: String, panel: ExperimentPanel
    ) -> Bool {
        var spec = ExperimentManifest.SweepSpec()
        spec.selection = ExperimentManifest.SweepSelection(
            objective: .init(metric: objective),
            constraints: .init(
                capabilityTolerance: SweepSelectionRule.defaultCapabilityTolerance,
                coherenceFloor: SweepSelectionRule.defaultCoherenceFloor))
        return panel.setSweepSpec(spec, for: name)
    }

    private func refreshOptimizations() {
        panel.refresh()
        // The "Promoted" lifecycle evidence (`promotedExperiments`) reads
        // the agent library's row index; rescan it off the main actor so a
        // just-minted agent shows up without a blocking library walk here.
        service.fineTuning.refreshAgentLibraryAsync()
        if showsServerList {
            Task {
                await panel.refreshRemoteOptimizations()
                reconcileSelection()
            }
        } else {
            reconcileSelection()
        }
    }

    private func reconcileSelection() {
        // A pending selection (declared from New Agent → Optimize) wins as
        // soon as the run appears; it stays pending until the list has it.
        if let pending = pendingSelection {
            let ref = OptimizationRef(source: .local, name: pending)
            if optimizations.contains(where: { $0.ref == ref }) {
                selectedRef = ref
                pendingSelection = nil
                reloadSweepRun()
                return
            }
        }
        if selectedRef == nil
            || !optimizations.contains(where: { $0.ref == selectedRef })
        {
            selectedRef = optimizations.first?.ref
        }
        reloadSweepRun()
    }

    private func reloadSweepRun() {
        selectedCell = nil
        sweepRun = nil
        guard let ref = selectedRef else { return }
        switch ref.source {
        case .local:
            // Local run directories — on a PAIRED server this is literally
            // the server's runs/ tree, so server-executed sweeps appear too.
            sweepRun = SweepRunCatalog.newestSweepRun(experiment: ref.name)
            // A server-executed sweep auto-pins the model revision into the
            // SERVER's manifest copy; the results bring the run (and its
            // snapshot) home but not that mutation, so the local draft
            // stayed revision-less and promote's epoch guard refused over
            // the pin the researcher's own sweep resolved (field incident
            // 2026-08-04). Adopt the snapshot's revision on DISCOVERY —
            // the same unit-tested reconciliation the evidence-import path
            // runs, with its loud conflict arm intact.
            if let directory = sweepRun?.directory {
                panel.noteEvidenceRevisionAdoption(forImportedRun: directory)
                // And the sweep's projected conditions (the server wrote
                // them into ITS manifest copy; a local `run` submission
                // needs the arms too) — conflict-safe, loud, idempotent.
                let outcome = SweepConditionAdoption.adoptProjectedConditions(
                    fromSweepRun: directory)
                if let notice = SweepConditionAdoption.notice(for: outcome) {
                    panel.note(
                        notice.message,
                        severity: notice.isWarning ? .warning : .success)
                    if case .adopted = outcome { panel.refresh() }
                }
            }
        case .server:
            Task {
                let run = await panel.loadRemoteSweepRun(experiment: ref.name)
                if selectedRef == ref { sweepRun = run }
            }
        }
    }

    // MARK: Optimization lists

    /// The local ("this workspace") list — present in EVERY compute mode.
    @ViewBuilder
    private var localListSection: some View {
        Section(
            showsServerList
                ? "Optimization runs — this workspace" : "Optimization runs")
        {
            if localOptimizationItems.isEmpty {
                Text(
                    "No optimization runs in this workspace yet. An "
                        + "optimization run is a draft study with a declared "
                        + "layer×alpha sweep and selection criterion — its "
                        + "recommended cell becomes an agent.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(localOptimizationItems) { optimization in
                    optimizationRow(optimization)
                }
            }
            localListButtons
            Text(localSourceCaption)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .help(
                    "declaring an optimization is a manifest edit in the LOCAL "
                        + "workspace — available in every compute mode; the "
                        + "compute target decides where the sweep executes")
        }
    }

    /// The server's own list — only for a NON-paired server workspace (on a
    /// paired server the trees are the same files; one list, no duplicates).
    @ViewBuilder
    private var serverListSection: some View {
        Section(service.cluster.serverArtifactListTitle(kind: "Optimization runs")) {
            if serverOptimizationItems.isEmpty {
                Text(
                    "No optimization runs in \(substrate)'s workspace. Its "
                        + "runs are read-only here — declare in the workspace "
                        + "paired to that server, or Submit Bundle (verb "
                        + "sweep) from Studies.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(serverOptimizationItems) { optimization in
                    optimizationRow(optimization)
                }
            }
            Text(
                "read-only listing of \(substrate)'s experiments/ tree — "
                    + "Optimize and Create Agent execute there; artifacts stay "
                    + "in that workspace")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private var localSourceCaption: String {
        if isServer, isPaired {
            return "paired server — declare and edit here; Optimize and "
                + "Create Agent execute on \(substrate) into this same workspace"
        }
        if isServer {
            return "declare and edit here; run on \(substrate) via Submit "
                + "Bundle (verb sweep) in Studies — the server is not paired "
                + "to this workspace"
        }
        return "optimization runs and sweep grids read from the local workspace"
    }

    private func optimizationRow(_ optimization: OptimizationItem) -> some View {
        Button {
            selectedRef = optimization.ref
        } label: {
            HStack(spacing: 8) {
                Image(systemName: selectedRef == optimization.ref
                    ? "inset.filled.circle" : "circle")
                    .foregroundStyle(
                        selectedRef == optimization.ref ? Color.accentColor : Color.secondary)
                VStack(alignment: .leading, spacing: 1) {
                    Text("\(optimization.name) · \(optimization.statusLabel)")
                        .font(.callout)
                    Text(optimizationRowCaption(optimization))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(
            "experiments whose manifest declares a sweep spec or carries "
                + "sweep-selection provenance — a lens over studies (the "
                + "research funnel's 'screen' stage), not a new object type")
    }

    private func optimizationRowCaption(_ optimization: OptimizationItem) -> String {
        "\(optimization.modelID) · \(optimization.conceptCount) concept"
            + (optimization.conceptCount == 1 ? "" : "s")
    }

    @ViewBuilder
    private var localListButtons: some View {
        HStack(spacing: 8) {
            Button("Declare an Optimization…") { showDeclareSheet = true }
                .controlSize(.small)
                .help(
                    "turn a draft study into an optimization run by declaring "
                        + "its layer×alpha sweep and selection criterion — a "
                        + "local manifest edit, available in every compute mode")
            Button("Open Studies") { navigate(.studies) }
                .controlSize(.small)
                .help("create or edit draft studies")
        }
    }

    /// Context-carrying jump to Studies for the bundle-sweep path: preselects
    /// the study, preconfigures Submit Bundle (verb sweep, real run), and
    /// asks the Studies view to open its Run-on-Server disclosure — one
    /// click away from submitting the RIGHT verb for the RIGHT study.
    private func openStudiesForBundleSweep(study name: String) {
        if panel.experiments.contains(where: { $0.name == name }) {
            panel.selectedName = name
        }
        panel.remoteVerb = "sweep"
        panel.remoteDryRun = false
        panel.pendingRevealRemoteControls = true
        navigate(.studies)
    }

    // MARK: Declared criterion

    private func criterionSection(_ optimization: OptimizationItem) -> some View {
        let declared = optimization.sweep?.selection
        // When no declared spec travelled (older server, or a local manifest
        // that only carries provenance), fall back to the criterion the
        // sweep STAMPED (verbatim provenance) for display and marking.
        let stamped = optimization.selections.sorted { $0.key < $1.key }
            .first?.value.criterion
        let effective = declared ?? stamped
        let resolved = SweepRunCatalog.displayCriterion(effective)
        return Section("Declared criterion") {
            if declared == nil {
                Text(missingDeclaredCaption(optimization, hasStamped: stamped != nil))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            LabeledContent("Objective") {
                Text(criterionValue(
                    resolved.metric, isDefault: effective?.objective?.metric == nil))
            }
            LabeledContent("Capability tolerance") {
                Text(criterionValue(
                    format(resolved.capabilityTolerance),
                    isDefault: effective?.constraints?.capabilityTolerance == nil))
            }
            LabeledContent("Coherence floor (distinct-2)") {
                Text(criterionValue(
                    format(resolved.coherenceFloor),
                    isDefault: effective?.constraints?.coherenceFloor == nil))
            }
            LabeledContent("Matched-norm random control") {
                Text(
                    resolved.matchedNormRandomMargin.map {
                        "margin \(format($0))"
                    } ?? "none declared")
            }
        }
    }

    private func missingDeclaredCaption(
        _ optimization: OptimizationItem, hasStamped: Bool
    ) -> String {
        if optimization.source == .server {
            return hasStamped
                ? "\(substrate) reported no declared sweep spec (older server, "
                    + "or none declared) — showing the criterion the sweep "
                    + "stamped on its recommendation"
                : "\(substrate) reported no declared sweep spec (older server, "
                    + "or none declared) and no stamped provenance exists yet "
                    + "— showing the documented defaults"
        }
        return "no selection block declared — the sweep applies the "
            + "documented defaults, shown below"
    }

    private func criterionValue(_ value: String, isDefault: Bool) -> String {
        isDefault ? "\(value) (default)" : value
    }

    /// E2: the resolved view of the sweep — fractions as layer indices, the
    /// files with their real state, and the control's ABSENCE stated. Shown
    /// above the editor so the panel reads as "what will run", then "change
    /// it", rather than leaving the researcher to resolve it themselves.
    @ViewBuilder
    private func resolvedSweepSection(_ optimization: OptimizationItem) -> some View {
        if let manifest = panel.experiments.first(where: { $0.name == optimization.name }),
            let resolved = SweepPanelModel.resolve(manifest: manifest)
        {
            SweepPanelSection(resolved: resolved)
        }
    }

    @ViewBuilder
    private func sweepSpecSection(_ optimization: OptimizationItem) -> some View {
        resolvedSweepSection(optimization)
        // The inline editor works for LOCAL draft optimizations in EVERY compute
        // mode — editing is a local manifest write; only execution routes.
        if optimization.source == .local, optimization.isDraft {
            SweepSpecEditorSection(
                experimentName: optimization.name,
                spec: optimization.sweep,
                panel: panel,
                onSaved: { refreshOptimizations() },
                runControls: { runSweepControls(optimization) })
                .id(optimization.id)
        } else {
            readOnlySweepSpecSection(optimization)
        }
    }

    private func readOnlySweepSpecSection(_ optimization: OptimizationItem) -> some View {
        Section("Sweep spec") {
            if let sweep = optimization.sweep {
                LabeledContent(
                    "Layer fractions",
                    value: sweep.layerFractions.map(format).joined(separator: ", "))
                LabeledContent(
                    "Alphas (norm units)",
                    value: sweep.alphas.map(format).joined(separator: ", "))
                LabeledContent("Dev prompts", value: sweep.devPromptsFile)
                LabeledContent("Battery", value: sweep.batteryFile)
                LabeledContent("Max tokens", value: "\(sweep.maxTokens)")
                Text(readOnlySpecFootnote(optimization))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else if optimization.source == .server {
                Text(
                    "\(substrate)'s listing carries no declared sweep spec "
                        + "(older server, or none declared) — the grid and "
                        + "recommendations below are read from its runs/")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text(
                    "no sweep spec on this manifest — it appears here because "
                        + "its conditions carry sweep-selection provenance")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            runSweepControls(optimization)
        }
    }

    private func readOnlySpecFootnote(_ optimization: OptimizationItem) -> String {
        if optimization.source == .server {
            return "declared spec read verbatim from \(substrate)'s manifest — "
                + "read-only here; declare/edit in the workspace paired to "
                + "that server"
        }
        return "frozen manifests are immutable — the declared spec is "
            + "pinned data; duplicate the study in Studies to iterate"
    }

    // MARK: Run sweep

    /// Where this optimization's sweep would execute (the compute target decides;
    /// the optimization's SOURCE decides which tree the study lives in).
    private enum SweepExecution {
        /// In-process MLX run (local compute target).
        case local
        /// Durable server job for the server-resident copy: a server-source
        /// optimization, or a local optimization on a PAIRED server (same files).
        case server
        /// Local optimization + unpaired server: no direct path — the portable
        /// route is Submit Bundle (verb sweep) from Studies.
        case bundleViaStudies
    }

    private func sweepExecution(_ optimization: OptimizationItem) -> SweepExecution {
        guard isServer else { return .local }
        if optimization.source == .server || isPaired { return .server }
        return .bundleViaStudies
    }

    @ViewBuilder
    private func runSweepControls(_ optimization: OptimizationItem) -> some View {
        if sweepExecution(optimization) == .bundleViaStudies {
            bundleSweepControls(optimization)
        } else {
            directSweepControls(optimization)
        }
    }

    @ViewBuilder
    private func directSweepControls(_ optimization: OptimizationItem) -> some View {
        HStack(spacing: 8) {
            Button(
                sweepExecution(optimization) == .server
                    ? "Optimize on \(substrate)" : "Optimize"
            ) {
                // Item 2: a server-executed sweep is a model-running durable
                // job — the shared gate warns when no GPU session is up
                // (local sweeps pass straight through).
                let panel = panel
                ModelJobGPUGate.submit(
                    "optimization sweep", service: service,
                    pending: $pendingModelJob
                ) {
                    await panel.runSweep(experimentName: optimization.name)
                    refreshOptimizations()
                }
            }
            .disabled(sweepDisabledReason(optimization) != nil)
            if panel.isSweeping {
                ProgressView()
                    .controlSize(.small)
            }
            if panel.isSweeping || panel.activeSweepJob != nil {
                Button("Cancel Optimization", role: .destructive) {
                    Task { await panel.cancelSweep() }
                }
                .disabled(panel.activeSweepJob == nil && panel.sweepCancelRequested)
                .help(
                    "requests cancellation; the engine stops after the "
                        + "current generation — partial rows stay in the "
                        + "run directory")
            }
        }
        if let reason = sweepDisabledReason(optimization) {
            Text(reason)
                .font(.caption2)
                .foregroundStyle(.orange)
        } else {
            Text(runSweepHelp(optimization))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    /// The unpaired-server path for a LOCAL optimization: the sweep can't execute
    /// directly (direct verbs run server-resident copies only), so the button
    /// jumps to Studies with the study preselected and Submit Bundle
    /// preconfigured for verb `sweep` — context-carrying, not a bare link.
    @ViewBuilder
    private func bundleSweepControls(_ optimization: OptimizationItem) -> some View {
        Button("Submit Bundle: sweep — in Studies…") {
            openStudiesForBundleSweep(study: optimization.name)
        }
        .controlSize(.small)
        Text(
            "'\(optimization.name)' lives in this workspace; \(substrate) is not "
                + "paired to it. Submit Bundle sends a hash-pinned portable "
                + "copy and runs the sweep there — this button preselects the "
                + "study and the sweep verb in Studies.")
            .font(.caption2)
            .foregroundStyle(.secondary)
    }

    /// Why Run Sweep is disabled, in one plain sentence — nil means runnable.
    /// A silently gray button is a bug, not a state.
    private func sweepDisabledReason(_ optimization: OptimizationItem) -> String? {
        if panel.isSweeping {
            return "a sweep is already running — follow it in the activity pane"
        }
        if panel.isRunning || panel.isValidating {
            return "another study task is running — wait for it to finish"
        }
        if optimization.source == .local {
            if !optimization.isDraft, optimization.sweep == nil {
                return "'\(optimization.name)' is \(optimization.statusLabel) with no "
                    + "declared sweep spec — the spec is pinned data; duplicate "
                    + "the study in Studies to declare one"
            }
            if case .declaredAhead(let metric) = SweepSpecForm.validateSelection(
                optimization.sweep?.selection)
            {
                return "objective '\(metric)' is not implemented on this "
                    + "engine — the sweep refuses at start; switch to an "
                    + "implemented objective to run now"
            }
        }
        return nil
    }

    private func runSweepHelp(_ optimization: OptimizationItem) -> String {
        if sweepExecution(optimization) == .server {
            return optimization.source == .local
                ? "submits a durable sweep job on \(substrate) — the paired "
                    + "server shares this workspace's files, so the grid and "
                    + "recommendations land right here"
                : "submits a durable sweep job for the server-resident copy "
                    + "and follows it in the activity pane; the grid and "
                    + "recommendations refresh on completion"
        }
        return "runs the layer×alpha sweep now — it loads the study's pinned "
            + "model itself (no model needs to be loaded in Playground); "
            + "progress streams to the activity pane"
    }

    // MARK: Lifecycle strip

    private func lifecycleStates(_ optimization: OptimizationItem) -> OptimizationLifecycle.States {
        switch optimization.source {
        case .local:
            return OptimizationLifecycle.derive(
                hasSweepSpec: optimization.sweep != nil,
                hasSweepRun: sweepRun != nil,
                hasRecommendation: hasRecommendation(optimization),
                hasPromotedAgent: promotedExperiments.contains(optimization.name))
        case .server:
            // Declared is DEFINITE when the record carries the spec (the
            // server now returns it verbatim); tri-state survives only for
            // genuinely unknowable fields — an absent spec could be an older
            // server, and the server's variant listing carries no promotion
            // block.
            return OptimizationLifecycle.derive(
                hasSweepSpec: optimization.sweep != nil ? true : nil,
                hasSweepRun: sweepRun != nil,
                hasRecommendation: hasRecommendation(optimization),
                hasPromotedAgent: nil)
        }
    }

    private func hasRecommendation(_ optimization: OptimizationItem) -> Bool {
        if !optimization.selections.isEmpty { return true }
        guard let run = sweepRun else { return false }
        return run.recommendations.values.contains { recommendation in
            if case .selected = recommendation { return true }
            return false
        }
    }

    private func lifecycleSection(_ optimization: OptimizationItem) -> some View {
        let states = lifecycleStates(optimization)
        return Section("Lifecycle") {
            lifecycleStrip(states)
            Text(OptimizationLifecycle.nextStep(states))
                .font(.caption)
                .foregroundStyle(.secondary)
            if states.promoted == true {
                Button("Open Studies — Confirm agent") {
                    openStudiesForConfirmation(optimization)
                }
                .controlSize(.small)
                .help(
                    "creates a NEW confirmation draft duplicated from this "
                        + "study (the screen study itself is untouched) and "
                        + "opens it in Studies — test the promoted agent "
                        + "under a declared perturbation policy (α ± δ, "
                        + "matched-norm control) on held-out prompts")
            }
        }
    }

    /// Context-carrying confirm link: a confirmation is a NEW preregistered
    /// study, so the shortcut creates a fresh confirmation draft duplicated
    /// from this optimization's (screen) study — never flipping the screen
    /// study's own phase (P1 fix 2026-07-19) — then navigates to Studies
    /// with the new draft selected and the promoted agent preselected.
    private func openStudiesForConfirmation(_ optimization: OptimizationItem) {
        panel.createConfirmationDraft(from: optimization.name)
        navigate(.studies)
    }

    private func lifecycleStrip(_ states: OptimizationLifecycle.States) -> some View {
        HStack(spacing: 10) {
            stageChip("Declared", states.declared)
            stageArrow
            stageChip("Optimized", states.swept)
            stageArrow
            stageChip("Recommended", states.recommended)
            stageArrow
            stageChip("Agent created", states.promoted)
        }
    }

    private var stageArrow: some View {
        Image(systemName: "arrow.right")
            .font(.caption2)
            .foregroundStyle(.tertiary)
    }

    /// nil state = not knowable on this substrate (rendered as "?", never as
    /// a false negative).
    private func stageChip(_ label: String, _ state: Bool?) -> some View {
        HStack(spacing: 3) {
            Image(systemName: stageSymbol(state))
                .foregroundStyle(state == true ? Color.green : Color.secondary)
            Text(label)
        }
        .font(.caption)
        .help(state == nil ? "not derivable from the server API" : "")
    }

    private func stageSymbol(_ state: Bool?) -> String {
        switch state {
        case true?: "checkmark.circle.fill"
        case false?: "circle"
        case nil: "questionmark.circle"
        }
    }

    // MARK: Grid

    @ViewBuilder
    private func gridSections(_ optimization: OptimizationItem) -> some View {
        if let run = sweepRun {
            Section("Optimization grid — \(run.runName)") {
                ForEach(SweepRunCatalog.concepts(in: run.rows), id: \.self) { concept in
                    conceptGrid(concept: concept, run: run, optimization: optimization)
                }
                promoteSelectedCellControls(optimization)
            }
        } else {
            Section("Optimization grid") {
                Text(missingSweepRunCaption(optimization))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func missingSweepRunCaption(_ optimization: OptimizationItem) -> String {
        switch sweepExecution(optimization) {
        case .server where optimization.source == .server:
            return "No sweep run found for '\(optimization.name)' in "
                + "\(substrate)'s runs/ — use Optimize above to submit one "
                + "on that server."
        case .server:
            return "No sweep run found for '\(optimization.name)' in this (paired) "
                + "workspace's runs/ — use Optimize above to submit one on "
                + "\(substrate); it lands here."
        case .bundleViaStudies:
            return "No sweep run found for '\(optimization.name)' in this "
                + "workspace's runs/ — submit the sweep to \(substrate) via "
                + "Submit Bundle in Studies (button above), or switch the "
                + "compute target to Local (MLX) to run it here."
        case .local:
            return "No sweep run found for '\(optimization.name)' in this "
                + "workspace's runs/ — use Optimize above (equivalent to "
                + "steerlab-cli experiment sweep \(optimization.name))."
        }
    }

    /// The stamped `-recommended` condition for a concept (winner + resolved
    /// criterion verbatim as the sweep applied it). In a server workspace
    /// this comes from the experiment detail's condition selection blocks.
    private func stampedSelection(
        concept: String, optimization: OptimizationItem
    ) -> ExperimentManifest.SelectionProvenance? {
        optimization.selections["\(concept)-recommended"]
    }

    /// Constraint marking uses the criterion the sweep actually STAMPED when
    /// available (the embedded object, verbatim); otherwise the manifest's
    /// declared block with defaults filled for display.
    private func gridCriterion(
        concept: String, optimization: OptimizationItem
    ) -> SweepSelectionRule.Resolved {
        SweepRunCatalog.displayCriterion(
            stampedSelection(concept: concept, optimization: optimization)?.criterion
                ?? optimization.sweep?.selection)
    }

    @ViewBuilder
    private func conceptGrid(
        concept: String, run: SweepRunCatalog.SweepRun,
        optimization: OptimizationItem
    ) -> some View {
        let rows = run.rows.filter { $0.concept == concept }
        let baseline = rows.first { $0.isBaseline }
        let selection = stampedSelection(concept: concept, optimization: optimization)
        let winner = selection?.sweepRun == run.runName ? selection?.winningCell : nil
        let criterion = gridCriterion(concept: concept, optimization: optimization)

        VStack(alignment: .leading, spacing: 6) {
            Text(concept)
                .font(.callout.weight(.semibold))
            if let baseline {
                Text(baselineLine(baseline, criterion: criterion))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            gridTable(
                concept: concept, rows: rows.filter { !$0.isBaseline },
                baseline: baseline, criterion: criterion, winner: winner)
            if let selection, selection.sweepRun != run.runName {
                Text(
                    "recommendation on the manifest is from run "
                        + "\(selection.sweepRun), not this (newest) run")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
            gridLegend
            if criterion.metric != "markerDensity" {
                Text(gridMetricCaption(criterion))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }

    /// Names the number the cells display, so a judgeScore grid is never
    /// misread as a density grid.
    private func gridMetricCaption(_ criterion: SweepSelectionRule.Resolved) -> String {
        "cells show \(criterion.metric) — marker density stays a diagnostic "
            + "in the cell hover"
    }

    private func baselineLine(
        _ row: SweepRunCatalog.Row, criterion: SweepSelectionRule.Resolved
    ) -> String {
        let tail = "density \(format(row.markerDensity)) · "
            + "distinct-2 \(format(row.distinct2)) · battery \(format(row.batteryAccuracy))"
        if criterion.metric == "markerDensity" {
            return "baseline (no injection): " + tail
        }
        // judgeScore's baseline is the pinned 0.5 tie; logprobShift's is 0 —
        // read from the run's own baseline row when recorded (post-objective
        // runs), else reconstructed from the pinned rule.
        let value = row.objective ?? SweepSelectionRule.baselineMetric(
            criterion.metric, baselineDensity: row.markerDensity)
        return "baseline (no injection): \(criterion.metric) \(format(value)) · " + tail
    }

    private func gridTable(
        concept: String,
        rows: [SweepRunCatalog.Row],
        baseline: SweepRunCatalog.Row?,
        criterion: SweepSelectionRule.Resolved,
        winner: ExperimentManifest.SelectionProvenance.Cell?
    ) -> some View {
        let layers = Set(rows.map(\.layer)).sorted()
        let alphas = Set(rows.map(\.alpha)).sorted()
        return ScrollView(.horizontal, showsIndicators: false) {
            Grid(alignment: .center, horizontalSpacing: 4, verticalSpacing: 4) {
                GridRow {
                    Text("L \\ α")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                    ForEach(alphas, id: \.self) { alpha in
                        Text(format(alpha))
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                    }
                }
                ForEach(layers, id: \.self) { layer in
                    GridRow {
                        Text("L\(layer)")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                        ForEach(alphas, id: \.self) { alpha in
                            cellView(
                                concept: concept,
                                row: rows.first {
                                    $0.layer == layer && abs($0.alpha - alpha) < 1e-9
                                },
                                baseline: baseline, criterion: criterion,
                                winner: winner)
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func cellView(
        concept: String,
        row: SweepRunCatalog.Row?,
        baseline: SweepRunCatalog.Row?,
        criterion: SweepSelectionRule.Resolved,
        winner: ExperimentManifest.SelectionProvenance.Cell?
    ) -> some View {
        if let row {
            let state = SweepRunCatalog.cellState(
                row: row, baseline: baseline, criterion: criterion, winner: winner)
            let cellID = SelectedCell(
                concept: concept, layer: row.layer, alpha: row.alpha)
            Button {
                selectedCell = selectedCell == cellID ? nil : cellID
            } label: {
                VStack(spacing: 1) {
                    Text(cellNumber(row, criterion: criterion))
                        .font(.caption.monospacedDigit())
                    if state == .winner {
                        Text("winner")
                            .font(.system(size: 8, weight: .bold))
                    }
                }
                .frame(minWidth: 52)
                .padding(.vertical, 4)
                .padding(.horizontal, 4)
                .background(
                    RoundedRectangle(cornerRadius: 5).fill(cellColor(state)))
                .overlay(
                    RoundedRectangle(cornerRadius: 5)
                        .stroke(
                            selectedCell == cellID ? Color.accentColor : .clear,
                            lineWidth: 2))
            }
            .buttonStyle(.plain)
            .help(cellHelp(row, state: state, criterion: criterion))
        } else {
            Text("—")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .frame(minWidth: 52)
        }
    }

    private func cellColor(_ state: SweepRunCatalog.CellState) -> Color {
        switch state {
        case .winner: .green.opacity(0.3)
        case .pass: .secondary.opacity(0.1)
        case .failedConstraint: .red.opacity(0.16)
        case .baseline: .blue.opacity(0.12)
        }
    }

    /// The cell's big number: the DECLARED objective's value. For
    /// markerDensity sweeps that IS the density; for judgeScore/logprobShift
    /// it is the run's `objective` column — "—" for runs that predate the
    /// column (the hover still carries the diagnostics).
    private func cellNumber(
        _ row: SweepRunCatalog.Row, criterion: SweepSelectionRule.Resolved
    ) -> String {
        if criterion.metric == "markerDensity" {
            return format(row.markerDensity)
        }
        guard let objective = row.objective else { return "—" }
        return format(objective)
    }

    private func cellHelp(
        _ row: SweepRunCatalog.Row, state: SweepRunCatalog.CellState,
        criterion: SweepSelectionRule.Resolved
    ) -> String {
        let diagnostics = "density \(format(row.markerDensity)), "
            + "distinct-2 \(format(row.distinct2)), battery "
            + "\(format(row.batteryAccuracy))"
        let objectivePart = criterion.metric == "markerDensity"
            ? ""
            : "\(criterion.metric) "
                + "\(row.objective.map(format) ?? "not recorded (pre-objective run)"), "
        return "L\(row.layer) α\(format(row.alpha)) — " + objectivePart + diagnostics
            + " · \(stateLabel(state)). Click to select for Create Agent."
    }

    private func stateLabel(_ state: SweepRunCatalog.CellState) -> String {
        switch state {
        case .winner: "winner under the declared criterion"
        case .pass: "passes constraints"
        case .failedConstraint: "fails a declared constraint"
        case .baseline: "baseline"
        }
    }

    private var gridLegend: some View {
        HStack(spacing: 10) {
            legendSwatch(.green.opacity(0.3), "winner")
            legendSwatch(.secondary.opacity(0.1), "pass")
            legendSwatch(.red.opacity(0.16), "fails constraint")
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
    }

    private func legendSwatch(_ color: Color, _ label: String) -> some View {
        HStack(spacing: 3) {
            RoundedRectangle(cornerRadius: 2).fill(color).frame(width: 10, height: 10)
            Text(label)
        }
    }

    // MARK: Promote selected cell (winner or override)

    /// Where a promotion mints for this optimization: on the server for
    /// server-source optimizations and for local optimizations on a PAIRED server (same
    /// tree — execution follows the compute target); locally otherwise. An
    /// UNPAIRED server must never be asked to mint from a local manifest's
    /// provenance — that tree doesn't hold this study.
    ///
    /// Foreign-substrate EVIDENCE is not a routing question, though it read
    /// like one for a day. A cluster workspace's sweeps are all foreign by
    /// construction — data on the Mac, compute on the cluster — and the Mac
    /// is exactly where such a study is promoted, the same way it is frozen.
    /// What blocked it was capability, not routing: the vector matcher and
    /// the epoch guard keyed on this engine instead of the workspace's
    /// declared compute substrate (`WorkspaceCompute`). With that fixed,
    /// sending the promotion to a server that does not hold the study would
    /// trade one dead end for another.
    private func promotionRoute(_ optimization: OptimizationItem) -> ExperimentPanel.PromotionRoute {
        isServer && (isPaired || optimization.source == .server)
            ? .activeServer : .local
    }

    private func mintsOnServer(_ optimization: OptimizationItem) -> Bool {
        promotionRoute(optimization) == .activeServer
    }

    @ViewBuilder
    private func promoteSelectedCellControls(_ optimization: OptimizationItem) -> some View {
        if let cell = selectedCell {
            let selection = stampedSelection(concept: cell.concept, optimization: optimization)
            let isWinner =
                selection?.winningCell.layer == cell.layer
                && abs((selection?.winningCell.alpha ?? .nan) - cell.alpha) < 1e-9
            // The dose-monotonicity read for the SELECTED cell's layer — the
            // promote decision should see whether the effect tracks dose
            // before a Create Agent click (sentence only; the chart lives on
            // the recommendation row).
            if let run = sweepRun {
                SweepDoseMonotonicityView(
                    run: run,
                    concept: cell.concept,
                    layer: cell.layer,
                    metric: gridCriterion(
                        concept: cell.concept, optimization: optimization).metric,
                    showsChart: false)
            }
            HStack(spacing: 8) {
                Text("selected: \(cell.concept) L\(cell.layer) α\(format(cell.alpha))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if isWinner {
                    Button("Create Agent") {
                        promoteFromDisplayedRun(
                            optimization: optimization, concept: cell.concept)
                    }
                    .disabled(sweepRun == nil)
                    .help(promoteWinnerHelp(optimization))
                } else {
                    Button("Create Agent (override)…") {
                        overridePromotion = OverridePromotion(
                            experiment: optimization.name, concept: cell.concept,
                            layer: cell.layer, alpha: cell.alpha,
                            mintsOnServer: mintsOnServer(optimization),
                            pins: pinsForDisplayedRun(
                                experimentName: optimization.name,
                                concept: cell.concept))
                    }
                    .help(
                        "this is NOT the winner under the declared criterion — "
                            + "creating an agent from it requires a reason and "
                            + "stamps promotedBy: manualOverride")
                }
            }
        }
    }

    private func promoteWinnerHelp(_ optimization: OptimizationItem) -> String {
        let base = "mint an agent from the criterion-selected winning cell "
            + "(promotedBy: criterion)"
        return mintsOnServer(optimization)
            ? base + " — minted on \(substrate)" : base
    }

    // MARK: The measured grid (E3)

    /// The sweep's actual cells — heatmap, struck-through constraint
    /// failures, control status. This content previously existed only in
    /// `scripts/run-viewer.py`, a scratch tool built because the app could
    /// not display a sweep it had just run.
    @ViewBuilder
    private func sweepGridSection(_ optimization: OptimizationItem) -> some View {
        if let run = sweepRun, !run.rows.isEmpty {
            Section("Measured grid — \(run.runName)") {
                ForEach(SweepGridPresentation.concepts(rows: run.rows), id: \.self) { concept in
                    SweepGridView(
                        grid: SweepGridPresentation.grid(
                            concept: concept, rows: run.rows,
                            recommendation: run.recommendations[concept]))
                }
            }
        }
    }

    // MARK: Recommendations

    @ViewBuilder
    private func recommendationsSection(_ optimization: OptimizationItem) -> some View {
        if let run = sweepRun, !run.recommendations.isEmpty {
            Section("Recommended agent settings — \(run.runName)") {
                ForEach(run.recommendations.keys.sorted(), id: \.self) { concept in
                    recommendationRow(
                        concept: concept,
                        recommendation: run.recommendations[concept],
                        optimization: optimization)
                }
            }
        }
    }

    @ViewBuilder
    private func recommendationRow(
        concept: String,
        recommendation: SweepRunCatalog.Recommendation?,
        optimization: OptimizationItem
    ) -> some View {
        switch recommendation {
        case .selected(let provenance):
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(concept)
                        .font(.callout.weight(.medium))
                    Text(provenanceLine(provenance))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                    if let control = provenance.control {
                        Text(controlLine(control))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    // Promotion defensibility (Phase 2, item 11): does the
                    // effect track dose at the winning layer? Computed from
                    // the loaded sweep grid by tested ExperimentKit code.
                    if let run = sweepRun {
                        SweepDoseMonotonicityView(
                            run: run,
                            concept: concept,
                            layer: provenance.winningCell.layer,
                            metric: SweepRunCatalog.displayCriterion(
                                provenance.criterion).metric)
                    }
                }
                Spacer()
                Button("Create Agent") {
                    // Guarded like the grid-cell button: an unpinned promote
                    // is exactly the ambient resolution the contract removes.
                    promoteFromDisplayedRun(
                        optimization: optimization, concept: concept)
                }
                .disabled(sweepRun == nil)
                .help(promoteRecommendationHelp(optimization))
            }
        case .failure(let message):
            VStack(alignment: .leading, spacing: 2) {
                Text(concept)
                    .font(.callout.weight(.medium))
                Label(message, systemImage: "xmark.octagon")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
        case nil:
            EmptyView()
        }
    }

    /// Pins built from the sweep run THIS VIEW loaded — local or downloaded
    /// from the server, whichever the selected optimization refers to.
    ///
    /// Sourcing them from the local runs tree instead (as an earlier version
    /// did) is wrong in a server workspace: it finds nothing, or finds a
    /// same-named local run and sends its name and hash to a server where
    /// neither means anything.
    private func pinsForDisplayedRun(
        experimentName: String, concept: String
    ) -> AgentPromotion.Pins? {
        guard let sweepRun else { return nil }
        return ExperimentPanel.promotionPins(
            experimentName: experimentName, concept: concept,
            sweepRun: sweepRun,
            localManifestHash: panel.localManifestHash(experimentName))
    }

    /// Evidence-bearing promotion REFUSES rather than promoting unpinned.
    /// A Create Agent that silently fell back to ambient resolution is the
    /// exact failure the pinned contract exists to remove, so the button
    /// says why instead of doing it.
    private func promoteFromDisplayedRun(
        optimization: OptimizationItem, concept: String
    ) {
        guard
            let pins = pinsForDisplayedRun(
                experimentName: optimization.name, concept: concept)
        else {
            panel.refuse(
                .sweepSpec,
                "cannot promote '\(concept)': this view has no loaded sweep "
                    + "run to pin the promotion to. Reload the optimization, "
                    + "or use the CLI's explicit --sweep-run if you mean to "
                    + "name the evidence yourself")
            return
        }
        panel.promote(
            experimentName: optimization.name, concept: concept,
            route: promotionRoute(optimization), pins: pins)
    }

    private func promoteRecommendationHelp(_ optimization: OptimizationItem) -> String {
        let base = "mint a reusable agent from this criterion-selected cell — "
            + "it carries the birth certificate (run, criterion, dev "
            + "split, metrics) into the Agents library"
        return mintsOnServer(optimization)
            ? base + " on \(substrate) (Agents lists its stored agents under "
                + "that server's workspace)"
            : base
    }

    /// The provenance line renders the EMBEDDED criterion verbatim (no
    /// criterion hash exists by design — verbatim beats canonicalization).
    private func provenanceLine(
        _ provenance: ExperimentManifest.SelectionProvenance
    ) -> String {
        var parts = [
            "winner L\(provenance.winningCell.layer) α\(format(provenance.winningCell.alpha))"
        ]
        let criterion = SweepRunCatalog.displayCriterion(provenance.criterion)
        parts.append(
            "criterion \(criterion.metric) · tol \(format(criterion.capabilityTolerance)) "
                + "· floor \(format(criterion.coherenceFloor))")
        if let metric = provenance.metrics[criterion.metric] {
            parts.append("\(criterion.metric) \(format(metric))")
        }
        parts.append("dev \(provenance.devPromptsHash.prefix(8))…")
        return parts.joined(separator: " · ")
    }

    private func controlLine(
        _ control: ExperimentManifest.SelectionProvenance.Control
    ) -> String {
        "control \(control.type): metric \(format(control.metricValue)) · "
            + "required margin \(format(control.margin)) — passed"
    }

    private func format(_ value: Double) -> String {
        value.formatted(.number.precision(.fractionLength(0 ... 3)))
    }
}

// MARK: - Sweep spec editor (draft manifests only)

/// The "Optimization (sweep) spec" editor for a SELECTED draft optimization. Its @State
/// is seeded from the saved spec in init — the parent keys this view by
/// experiment name (`.id`) so switching optimizations reseeds the fields. Saving
/// goes through `ExperimentPanel.setSweepSpec` (draft-only, criterion
/// validated at save via `SweepSpecForm`).
private struct SweepSpecEditorSection<RunControls: View>: View {
    let experimentName: String
    let panel: ExperimentPanel
    let onSaved: () -> Void
    @ViewBuilder let runControls: () -> RunControls

    @State private var layerFractionsText: String
    @State private var alphasText: String
    @State private var devPromptsFile: String
    @State private var batteryFile: String
    @State private var maxTokensText: String
    @State private var metric: String
    @State private var choicePromptsFile: String
    /// Per-concept choice instruments (choicePromptsFiles) — review
    /// 2026-08-02 round 2, P1: the editor stored only the singular field,
    /// so editing a mapped optimization's grid reconstructed the selection
    /// WITHOUT the map and the shared validation refused the save — the
    /// study was effectively uneditable here.
    @State private var choiceFilesByConcept: [String: String]
    @State private var toleranceText: String
    @State private var floorText: String
    @State private var marginText: String
    @State private var controlApplyTo: String
    @State private var controlTopKText: String
    @State private var formError: String?
    /// The field values as last WRITTEN to the manifest — the reference for
    /// `isDirty`. Reseeded on every successful save.
    @State private var savedSnapshot: Snapshot

    /// Freeze-time pins as they stand on the manifest right now (nil until
    /// freeze stamps them). Not editable here — pinning is freeze's job —
    /// so plain `let`s seeded in init are correct.
    private let pinnedDevPromptsHash: String?
    private let pinnedBatteryHash: String?

    /// Every editable field, compared verbatim. Deliberately the RAW text and
    /// not the parsed spec: "0.10" vs "0.1" is an unsaved edit as far as the
    /// researcher is concerned, and claiming otherwise is how a believed-set
    /// control margin went unsaved (finding 5, observed 2026-07-26).
    private struct Snapshot: Equatable {
        var layerFractions = ""
        var alphas = ""
        var devPromptsFile = ""
        var batteryFile = ""
        var maxTokens = ""
        var metric = ""
        var choicePromptsFile = ""
        var choiceFilesMap = ""
        var tolerance = ""
        var floor = ""
        var margin = ""
        var controlApplyTo = ""
        var controlTopK = ""
    }

    /// Canonical text form of the per-concept map for verbatim dirty
    /// comparison (same rule as every other field: raw text, not parsed).
    private static func mapText(_ map: [String: String]) -> String {
        map.sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: "\n")
    }

    private var currentSnapshot: Snapshot {
        .init(
            layerFractions: layerFractionsText, alphas: alphasText,
            devPromptsFile: devPromptsFile, batteryFile: batteryFile,
            maxTokens: maxTokensText, metric: metric,
            choicePromptsFile: choicePromptsFile,
            choiceFilesMap: Self.mapText(choiceFilesByConcept),
            tolerance: toleranceText,
            floor: floorText, margin: marginText,
            controlApplyTo: controlApplyTo, controlTopK: controlTopKText)
    }

    private var isDirty: Bool { currentSnapshot != savedSnapshot }

    init(
        experimentName: String,
        spec: ExperimentManifest.SweepSpec?,
        panel: ExperimentPanel,
        onSaved: @escaping () -> Void,
        @ViewBuilder runControls: @escaping () -> RunControls
    ) {
        self.experimentName = experimentName
        self.panel = panel
        self.onSaved = onSaved
        self.runControls = runControls
        let initial = spec ?? ExperimentManifest.SweepSpec()
        _layerFractionsText = State(
            initialValue: SweepSpecForm.numberListText(initial.layerFractions))
        _alphasText = State(
            initialValue: SweepSpecForm.numberListText(initial.alphas))
        _devPromptsFile = State(initialValue: initial.devPromptsFile)
        _batteryFile = State(initialValue: initial.batteryFile)
        _maxTokensText = State(initialValue: String(initial.maxTokens))
        _metric = State(
            initialValue: initial.selection?.objective?.metric ?? "markerDensity")
        _choicePromptsFile = State(
            initialValue: initial.selection?.objective?.choicePromptsFile ?? "")
        // A manifest may legally declare one SINGULAR instrument for several
        // attached concepts (the same file serves each). The editor renders
        // per-concept map rows in that case, so seed them from the singular
        // — unseeded they display blank while the real declaration hides
        // (review 2026-08-03 round 2, P2). Seeded into the saved snapshot
        // too: displaying the manifest's own declaration is not an edit.
        var seededMap = initial.selection?.objective?.choicePromptsFiles ?? [:]
        let manifestSingular = (initial.selection?.objective?.choicePromptsFile ?? "")
            .trimmingCharacters(in: .whitespaces)
        let manifestConcepts = (panel.experiments
            .first { $0.name == experimentName }?.concepts ?? [])
            .map(\.name)
        if !manifestSingular.isEmpty, manifestConcepts.count > 1 {
            for concept in manifestConcepts where seededMap[concept] == nil {
                seededMap[concept] = manifestSingular
            }
        }
        _choiceFilesByConcept = State(initialValue: seededMap)
        let tolerance = initial.selection?.constraints?.capabilityTolerance
            ?? SweepSelectionRule.defaultCapabilityTolerance
        _toleranceText = State(initialValue: "\(tolerance)")
        let floor = initial.selection?.constraints?.coherenceFloor
            ?? SweepSelectionRule.defaultCoherenceFloor
        _floorText = State(initialValue: "\(floor)")
        let margin = initial.selection?.controls?.matchedNormRandomMargin
        _marginText = State(initialValue: margin.map { "\($0)" } ?? "")
        let applyTo = initial.selection?.controls?.applyTo ?? "winner"
        let topK = initial.selection?.controls?.topK
        _controlApplyTo = State(initialValue: applyTo)
        _controlTopKText = State(initialValue: topK.map(String.init) ?? "")
        self.pinnedDevPromptsHash = initial.devPromptsHash
        self.pinnedBatteryHash = initial.batteryHash
        _savedSnapshot = State(
            initialValue: Snapshot(
                layerFractions: SweepSpecForm.numberListText(initial.layerFractions),
                alphas: SweepSpecForm.numberListText(initial.alphas),
                devPromptsFile: initial.devPromptsFile,
                batteryFile: initial.batteryFile,
                maxTokens: String(initial.maxTokens),
                metric: initial.selection?.objective?.metric ?? "markerDensity",
                choicePromptsFile: initial.selection?.objective?.choicePromptsFile ?? "",
                choiceFilesMap: Self.mapText(seededMap),
                tolerance: "\(tolerance)",
                floor: "\(floor)",
                margin: margin.map { "\($0)" } ?? "",
                controlApplyTo: applyTo,
                controlTopK: topK.map(String.init) ?? ""))
    }

    var body: some View {
        Section("Optimization (sweep) spec — draft, editable") {
            gridFields
            criterionFields
            saveControls
            // Finding 5: Optimize executes the SAVED spec, so running with
            // unsaved edits silently sweeps something other than what is on
            // screen. Rather than warn, make it unreachable.
            runControls()
                .disabled(isDirty)
        }
    }

    @ViewBuilder
    private var gridFields: some View {
        TextField("Layer fractions (0–1, comma-separated)", text: $layerFractionsText)
            .help("network-depth fractions the sweep maps to layers, e.g. 0.5, 0.7, 0.85")
        TextField("Alphas (norm units, comma-separated)", text: $alphasText)
            .help(
                "steering strengths in residual-norm units, e.g. 0.05, 0.08, 0.13 — "
                    + "the no-injection baseline cell is always implied")
        AlphaMagnitudeWarning(alphasText: alphasText)
        TextField("Dev prompts file", text: $devPromptsFile)
            .help("workspace-relative JSONL of dev-split prompts (hashed into provenance)")
        instrumentFileRow(label: "dev prompts", path: devPromptsFile)
        TextField("Capability battery file", text: $batteryFile)
            .help("workspace-relative battery the capability constraint is scored on")
        instrumentFileRow(label: "capability battery", path: batteryFile)
        sweepInputPinCaption
        TextField("Max tokens per generation", text: $maxTokensText)
    }

    /// What the manifest actually records for the two sweep inputs.
    ///
    /// This caption used to say the files were "not hash-pinned" and point at
    /// docs/STATUS.md §3. That stopped being true on 2026-07-20, when
    /// `sweep.devPromptsHash` / `sweep.batteryHash` landed on both engines:
    /// `ExperimentStore.pinSweepInputs` stamps them at freeze, freeze REFUSES
    /// when a named file cannot be read, and sweep start refuses on drift
    /// (`ExperimentTasks.swift:5234`). Leaving the old caption up told the
    /// researcher their evidence was weaker than it was.
    @ViewBuilder
    private var sweepInputPinCaption: some View {
        let pinned = [
            ("dev prompts", pinnedDevPromptsHash),
            ("capability battery", pinnedBatteryHash),
        ]
        if pinned.allSatisfy({ $0.1 == nil }) {
            Text("not pinned yet — freeze records the SHA-256 of both files "
                + "and refuses if either is missing; after that, any drift "
                + "refuses sweep start")
                .font(.caption2)
                .foregroundStyle(.secondary)
        } else {
            let parts = pinned.map { label, hash in
                "\(label) \(hash.map { String($0.prefix(8)) + "…" } ?? "UNPINNED")"
            }
            Text("pinned at freeze — \(parts.joined(separator: " · ")); "
                + "drift from these bytes refuses sweep start")
                .font(.caption2)
                .foregroundStyle(
                    pinned.contains { $0.1 == nil } ? .orange : .secondary)
        }
    }

    /// Every instrument file the spec names gets the same inspection
    /// affordance the manifest pins get elsewhere: name, view sheet,
    /// Reveal in Finder, and a loud "missing" when the path resolves to
    /// nothing. Display only — never a save gate.
    @ViewBuilder
    private func instrumentFileRow(label: String, path: String) -> some View {
        let trimmed = path.trimmingCharacters(in: .whitespaces)
        if !trimmed.isEmpty {
            FileReferenceRow(label: label, path: trimmed)
        }
    }

    @ViewBuilder
    private var criterionFields: some View {
        Picker("Objective metric", selection: $metric) {
            ForEach(SweepSelectionRule.knownMetrics, id: \.self) { name in
                Text(metricLabel(name)).tag(name)
            }
        }
        .help(
            "the declared selection objective — markerDensity is a "
                + "diagnostic/manipulation check, never the promotion objective "
                + "when the claim is about a substantive outcome")
        if !SweepSelectionRule.implementedMetrics.contains(metric) {
            Text("objective '\(metric)' is not implemented on this engine — "
                + "saving is allowed (declaring ahead is the point), but the "
                + "sweep refuses at start")
                .font(.caption2)
                .foregroundStyle(.orange)
        }
        if metric == "logprobShift" {
            let concepts = attachedConcepts
            if concepts.count > 1 {
                // One instrument per attached concept (the map form) — the
                // same rows the New Agent composer authors, so a mapped
                // optimization is EDITABLE here (review 2026-08-02 round 2,
                // P1).
                Text("one choice instrument per attached concept — each "
                    + "concept's cells are scored on its own rows; every "
                    + "file is pinned by hash at freeze")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                ForEach(concepts, id: \.self) { concept in
                    TextField(
                        "Choice prompts — \(concept)",
                        text: Binding(
                            get: { choiceFilesByConcept[concept] ?? "" },
                            set: { choiceFilesByConcept[concept] = $0 }))
                        .help(
                            "workspace-relative JSONL of choice rows scored "
                                + "for '\(concept)' only")
                    instrumentFileRow(
                        label: "choice prompts '\(concept)'",
                        path: choiceFilesByConcept[concept] ?? "")
                    choicePromptsAdvisory(
                        for: choiceFilesByConcept[concept] ?? "")
                }
            } else {
                // One concept: the singular representation, but READ from
                // whichever representation holds a value — a one-concept
                // map must not display as an empty field and be dropped on
                // save (review 2026-08-02 round 4, P2; save migrates it
                // into the singular field explicitly).
                let concept = concepts.first
                TextField(
                    concept.map { "Choice prompts — \($0) (JSONL)" }
                        ?? "Choice prompts file (JSONL)",
                    text: Binding(
                        get: { singleConceptChoiceFile(concept) },
                        set: { newValue in
                            choicePromptsFile = newValue
                            if let concept {
                                choiceFilesByConcept[concept] = newValue
                            }
                        }))
                    .help(
                        "workspace-relative JSONL of choice rows (prompt + ≥2 "
                            + "options, optional target) — the shift objective is "
                            + "mean Δ logP(target) vs baseline; pinned by hash "
                            + "at freeze")
                instrumentFileRow(
                    label: "choice prompts",
                    path: singleConceptChoiceFile(concept))
                choicePromptsAdvisory(for: singleConceptChoiceFile(concept))
            }
        }
        if metric == "judgeScore" {
            judgePinRows
        }
        TextField("Capability tolerance (0–1)", text: $toleranceText)
            .help("battery accuracy may drop at most this far below baseline")
        TextField("Coherence floor (distinct-2, 0–1)", text: $floorText)
            .help("cells below this distinct-bigram ratio fail the constraint")
        TextField("Matched-norm random control margin (empty = none)", text: $marginText)
            .help(
                "when set, the winner must beat a norm-matched random direction "
                    + "by at least this margin or the concept gets no recommendation")
        ControlScopeControls(
            applyTo: $controlApplyTo, topKText: $controlTopKText,
            marginText: marginText)
    }

    private func metricLabel(_ name: String) -> String {
        SweepSelectionRule.implementedMetrics.contains(name)
            ? name
            : name + " (not implemented on this engine — refuses at sweep start)"
    }

    /// Live advisory under the choice-prompts field: what the logprobShift
    /// instrument would measure (rows, options range, explicit vs defaulted
    /// targets), or the ENGINE loader's exact refusal. Advisory only —
    /// nothing here gates typing or saving.
    /// The manifest's attached concepts — drives one instrument row each
    /// under logprobShift.
    private var attachedConcepts: [String] {
        (panel.experiments.first { $0.name == experimentName }?.concepts ?? [])
            .map(\.name).sorted()
    }

    /// The one-concept value: the concept's MAP entry when the key exists,
    /// else the singular field. Map-first is deliberate (review 2026-08-03,
    /// P2) — every edit path writes the map, so a present key is always at
    /// least as fresh as the singular, while singular-first resurrected a
    /// stale value after an A → A+B (edit A's row) → A round trip.
    private func singleConceptChoiceFile(_ concept: String?) -> String {
        concept.flatMap { choiceFilesByConcept[$0] } ?? choicePromptsFile
    }

    @ViewBuilder
    private func choicePromptsAdvisory(for file: String) -> some View {
        switch SweepSpecForm.previewChoicePrompts(file: file) {
        case .noFile:
            EmptyView()
        case .ok(let preview):
            Text(SweepSpecForm.choicePromptsSummary(preview))
                .font(.caption2)
                .foregroundStyle(.secondary)
        case .problem(let reason):
            Text(choiceProblemCaption(reason))
                .font(.caption2)
                .foregroundStyle(.orange)
        }
    }

    private func choiceProblemCaption(_ reason: String) -> String {
        """
        \(reason) — advisory preview from the engine's own loader; the sweep \
        applies the same check for real at start
        """
    }

    /// The judgeScore instrument's ACTUAL state on this study's manifest —
    /// the pinned rubric (inspectable, with hash prefix) and the judge
    /// panel, or exactly what is missing. `setSweepSpec` already refuses a
    /// judgeScore spec without these pins; this makes that refusal
    /// unsurprising.
    @ViewBuilder
    private var judgePinRows: some View {
        let manifest = panel.experiments.first { $0.name == experimentName }
        let rubricFile = manifest?.judgeRubricFile
        let rubricHash = manifest?.judgeRubricHash
        let judges = manifest?.judges ?? []
        if let rubricFile, rubricHash != nil, !judges.isEmpty {
            FileReferenceRow(
                label: "judge rubric (pinned)",
                path: rubricFile,
                pinnedHash: rubricHash)
            Text(judgesCaption(judges))
                .font(.caption2)
                .foregroundStyle(.secondary)
        } else {
            Text(missingJudgePinsCaption(
                hasRubric: rubricFile != nil && rubricHash != nil,
                judgeCount: judges.count))
                .font(.caption2)
                .foregroundStyle(.orange)
        }
    }

    private func judgesCaption(_ judges: [ExperimentManifest.JudgeRef]) -> String {
        let list = judges.map { judge in
            judge.model.map { "\(judge.name) (\(judge.kind) · \($0))" }
                ?? "\(judge.name) (\(judge.kind))"
        }
        .joined(separator: ", ")
        return """
        judges (\(judges.count)): \(list) — the sweep pairs steered vs \
        baseline under these manifest pins
        """
    }

    private func missingJudgePinsCaption(
        hasRubric: Bool, judgeCount: Int
    ) -> String {
        var missing: [String] = []
        if !hasRubric {
            missing.append("a pinned rubric (judgeRubricFile + judgeRubricHash)")
        }
        if judgeCount == 0 {
            missing.append("at least one judge")
        }
        let what = missing.joined(separator: " and ")
        return """
        not pinned yet — judgeScore needs \(what) on this study; configure \
        in Studies › Evaluation (Save refuses until then)
        """
    }

    @ViewBuilder
    private var saveControls: some View {
        HStack(spacing: 8) {
            Button(isDirty ? "Save Sweep Spec (unsaved changes)" : "Save Sweep Spec") {
                save()
            }
            .keyboardShortcut("s", modifiers: .command)
            Spacer()
        }
        // Precedence: this form's own parse refusal, then the engine's
        // refusal from `setSweepSpec` — which used to speak ONLY into the
        // panel-top notice area (finding 11a), several hundred points above
        // this button.
        if let refusal = formError ?? panel.formErrors[.sweepSpec] {
            Label(refusal, systemImage: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        } else if isDirty {
            Label(
                "unsaved edits — Optimize runs the SAVED spec, so it stays "
                    + "disabled until you save",
                systemImage: "pencil.circle.fill")
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        } else {
            Text("saved. The spec is hashed manifest data — freeze pins it; "
                + "Optimize executes this saved spec")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private func save() {
        guard let fractions = SweepSpecForm.parseNumberList(layerFractionsText) else {
            formError = "layer fractions: enter comma-separated numbers, e.g. 0.5, 0.7, 0.85"
            return
        }
        guard let alphas = SweepSpecForm.parseNumberList(alphasText) else {
            formError = "alphas: enter comma-separated numbers, e.g. 0.05, 0.08, 0.13"
            return
        }
        guard let maxTokens = Int(maxTokensText.trimmingCharacters(in: .whitespaces)) else {
            formError = "max tokens: enter a whole number"
            return
        }
        guard let tolerance = Double(toleranceText.trimmingCharacters(in: .whitespaces)) else {
            formError = "capability tolerance: enter a number in [0, 1]"
            return
        }
        guard let floor = Double(floorText.trimmingCharacters(in: .whitespaces)) else {
            formError = "coherence floor: enter a number in [0, 1]"
            return
        }
        let trimmedMargin = marginText.trimmingCharacters(in: .whitespaces)
        var margin: Double?
        if !trimmedMargin.isEmpty {
            guard let parsed = Double(trimmedMargin) else {
                formError = "control margin: enter a number ≥ 0, or leave empty for none"
                return
            }
            margin = parsed
        }
        var spec = ExperimentManifest.SweepSpec(
            layerFractions: fractions,
            alphas: alphas,
            devPromptsFile: devPromptsFile.trimmingCharacters(in: .whitespaces),
            batteryFile: batteryFile.trimmingCharacters(in: .whitespaces),
            maxTokens: maxTokens)
        var objective = ExperimentManifest.SweepSelection.Objective(metric: metric)
        let concepts = attachedConcepts
        if metric == "logprobShift", concepts.count > 1 {
            // The per-concept map — one instrument per attached concept.
            var map: [String: String] = [:]
            for concept in concepts {
                let value = (choiceFilesByConcept[concept] ?? "")
                    .trimmingCharacters(in: .whitespaces)
                if !value.isEmpty { map[concept] = value }
            }
            objective.choicePromptsFiles = map.isEmpty ? nil : map
        } else if metric == "logprobShift" {
            // One concept: the same map-first resolution the field displays
            // (round 4 migrated JSON-loaded maps; 2026-08-03 P2 made the
            // map win so a stale singular can never be what gets saved).
            let resolved = singleConceptChoiceFile(concepts.first)
                .trimmingCharacters(in: .whitespaces)
            if !resolved.isEmpty {
                objective.choicePromptsFile = resolved
            }
        }
        // Rebuild the controls block whole so applyTo/topK survive a save
        // instead of silently reverting to winner-only (review 2026-08-03,
        // P1). A topK scope without a margin is composed as declared — the
        // shared resolver refuses it with the engine's own message.
        let scopedTopK = controlApplyTo == "topK"
        var controls: ExperimentManifest.SweepSelection.Controls?
        if margin != nil || scopedTopK {
            controls = .init(
                matchedNormRandomMargin: margin,
                applyTo: scopedTopK ? "topK" : nil,
                topK: scopedTopK
                    ? Int(controlTopKText.trimmingCharacters(in: .whitespaces))
                    : nil)
        }
        spec.selection = ExperimentManifest.SweepSelection(
            objective: objective,
            constraints: .init(capabilityTolerance: tolerance, coherenceFloor: floor),
            controls: controls)
        formError = nil
        // setSweepSpec refuses structural/criterion problems; since 2026-07-26
        // it records them in `panel.formErrors[.sweepSpec]` as well as the
        // notice feed, so `saveControls` can render them next to the button.
        if panel.setSweepSpec(spec, for: experimentName) {
            savedSnapshot = currentSnapshot
            onSaved()
        }
    }
}

// MARK: - Declare an Optimization sheet

/// Entry point from the Optimization runs list: pick a DRAFT study with no
/// declared sweep, choose the selection objective EXPLICITLY (no default —
/// the criterion is pre-declared data), and declare the default grid with
/// that objective — the inline editor then opens on the new run.
private struct DeclareOptimizationSheet: View {
    let drafts: [String]
    let declare: (String, String) -> Void
    let openStudies: () -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var candidate: String?
    /// Starts UNSELECTED; Declare is disabled until an objective is chosen.
    @State private var objective: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Declare an Optimization")
                .font(.headline)
            Text("An optimization run is a draft study with a declared "
                + "layer×alpha sweep and selection criterion — all hashed "
                + "manifest data that freeze pins before any behavior is "
                + "measured.")
                .font(.caption)
                .foregroundStyle(.secondary)
            sheetBody
            HStack {
                Button("Cancel") { dismiss() }
                Spacer()
            }
        }
        .padding(16)
        .frame(minWidth: 440)
        .onAppear {
            if candidate == nil { candidate = drafts.first }
        }
    }

    @ViewBuilder
    private var sheetBody: some View {
        if drafts.isEmpty {
            Text("No draft studies without a declared optimization in this workspace.")
                .font(.callout)
            Button("Create a draft in Studies…") {
                dismiss()
                openStudies()
            }
        } else {
            Picker("Draft study", selection: $candidate) {
                ForEach(drafts, id: \.self) { name in
                    Text(name).tag(String?.some(name))
                }
            }
            objectivePicker
            objectiveCaption
            HStack(spacing: 8) {
                Button("Declare Optimization") {
                    guard let candidate, let objective else { return }
                    dismiss()
                    declare(candidate, objective)
                }
                .keyboardShortcut(.defaultAction)
                .disabled(candidate == nil || objective == nil)
                Button("New draft in Studies…") {
                    dismiss()
                    openStudies()
                }
            }
            Text("declares the default grid with the chosen objective — edit "
                + "everything in the spec editor that opens on the new run")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private var objectivePicker: some View {
        Picker("Selection objective", selection: $objective) {
            Text("choose…").tag(String?.none)
            Text("judge score — outcome instrument "
                + "(recommended when the claim is about a substantive outcome)")
                .tag(String?.some("judgeScore"))
            Text("logprob shift — outcome instrument "
                + "(recommended when the claim is about a substantive outcome)")
                .tag(String?.some("logprobShift"))
            Text("marker density — smoke-test / manipulation check")
                .tag(String?.some("markerDensity"))
        }
        .help(
            "the declared selection objective — no default: markerDensity is "
                + "a diagnostic/manipulation check, never the promotion "
                + "objective when the claim is about a substantive outcome")
    }

    @ViewBuilder
    private var objectiveCaption: some View {
        if objective == "markerDensity" {
            Text("marker density is a smoke-test / manipulation check — never "
                + "the promotion objective when the claim is about a "
                + "substantive outcome")
                .font(.caption2)
                .foregroundStyle(.orange)
        }
        if objective == "judgeScore" || objective == "logprobShift" {
            Text("this objective needs its instrument on the draft: judgeScore "
                + "a pinned rubric + judges (Studies › Evaluation), logprobShift "
                + "a choice-prompts file (spec editor) — Declare surfaces the "
                + "engine's refusal if they are missing")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
}

// MARK: - Supporting types

/// Which tree an optimization row was read from: this workspace's manifests, or the
/// active (non-paired) server's experiment listing.
enum OptimizationSource: String {
    case local
    case server
}

/// Identity of a selected optimization across the two source lists — names can
/// collide between the local workspace and an unpaired server's tree.
struct OptimizationRef: Hashable {
    let source: OptimizationSource
    let name: String
}

private struct SelectedCell: Equatable {
    let concept: String
    let layer: Int
    let alpha: Double
}

struct OverridePromotion: Identifiable {
    let experiment: String
    let concept: String
    let layer: Int
    let alpha: Double
    /// Route decided by the originating optimization's source + pairing (see
    /// `OptimizationRunsView.promotionRoute`).
    let mintsOnServer: Bool
    /// Pins CAPTURED when the sheet opened, from the run the view was
    /// displaying then. Deriving them at confirm time instead would silently
    /// promote unpinned if the loaded run changed or cleared while the sheet
    /// was up — an override is still evidence and still names the run it
    /// deviates from.
    let pins: AgentPromotion.Pins?

    var id: String { "\(experiment)|\(concept)|\(layer)|\(alpha)" }
}

/// The loud path: promoting a NON-winning cell requires a written reason and
/// says exactly what stamp the agent will carry.
struct OverridePromotionSheet: View {
    let promotion: OverridePromotion
    let promote: (String) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var reason = ""

    private var trimmedReason: String {
        reason.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Manual override required")
                .font(.headline)
            Text(
                "Cell L\(promotion.layer) α\(promotion.alpha) of "
                    + "'\(promotion.concept)' is not the winner under the "
                    + "declared criterion.")
                .font(.callout)
            Label(
                "This agent will be stamped promotedBy: manualOverride.",
                systemImage: "exclamationmark.triangle.fill")
                .font(.callout)
                .foregroundStyle(.orange)
            TextField("Reason (required — stamped into the birth certificate)",
                      text: $reason, axis: .vertical)
                .lineLimit(2 ... 4)
                .textFieldStyle(.roundedBorder)
            HStack {
                Button("Cancel") { dismiss() }
                Spacer()
                Button("Create Agent with override") {
                    promote(trimmedReason)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(trimmedReason.isEmpty)
            }
        }
        .padding(16)
        .frame(minWidth: 420)
    }
}
