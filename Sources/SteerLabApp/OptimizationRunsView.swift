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
    /// The concept whose Create Agent is in flight — the re-entry guard AND
    /// the busy indicator. The server route is a durable mint; a second click
    /// used to submit a second one.
    @State private var promotingConcept: String?
    /// The outcome of the last Create Agent, rendered BESIDE the button. It
    /// used to land only in the bottom status section and the bell, several
    /// hundred points below where it was pressed.
    @State private var promotionOutcome: PromotionOutcome?
    /// Bumped by "Reload Spec": part of the sweep editor's keyed identity, so
    /// re-creating it reseeds every field from the manifest on disk.
    @State private var specReloadToken = 0

    /// A finished (or refused) promotion, as the row renders it.
    private struct PromotionOutcome {
        let concept: String
        let message: String
        let isFailure: Bool
    }
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
        panel.management.experiments.filter { manifest in
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
                statusLabel: Self.statusLabel(manifest.status.rawValue),
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
                statusLabel: Self.statusLabel(record.status),
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

    /// The manifest status in words. A server listing that carries no status
    /// used to render as a bare "?" in the row title, which reads as a
    /// rendering fault rather than as missing information.
    private static func statusLabel(_ raw: String?) -> String {
        switch raw {
        case "draft": "draft"
        case "frozen": "frozen (pinned)"
        case "complete": "complete"
        case let value? where !value.isEmpty && value != "?": value
        default: "status unknown"
        }
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
                promoteOverride(promotion, reason: reason)
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
        panel.management.experiments
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
        // A NEW declaration takes the baseline-relative coherence floor, and
        // writes it EXPLICITLY — a constraints block with neither relative
        // field keeps meaning the legacy absolute rule, forever, so the
        // default has to be stated rather than inferred.
        spec.selection = ExperimentManifest.SweepSelection(
            objective: .init(metric: objective),
            constraints: .init(
                capabilityTolerance: SweepSelectionRule.defaultCapabilityTolerance,
                coherenceRatioToBaseline: SweepSelectionRule.defaultCoherenceRatio,
                coherenceAbsoluteBackstop: SweepSelectionRule.defaultCoherenceBackstop))
        guard let reviewed = try? DraftAuthoringSnapshot(workspaceRoot: ExperimentStore.workspaceRoot, name: name) else {
            panel.note("Reload the intended draft before declaring an optimization.", severity: .warning)
            return false
        }
        return panel.setSweepSpec(spec, reviewed: reviewed)
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
                panel.noteEvidenceRevisionAdoption(forImportedRun: directory, workspaceRoot: ExperimentStore.workspaceRoot)
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
        Section {
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
        } header: {
            InfoSectionHeader(
                title: showsServerList
                    ? "Optimization runs — this workspace" : "Optimization runs",
                text: Self.optimizationsInfo)
        }
    }

    private static let optimizationsInfo = """
        An "optimization run" is not a new kind of object. It is a LENS over \
        studies: every experiment whose manifest declares a layer×alpha sweep \
        spec, or whose conditions already carry sweep-selection provenance, \
        shows up in this list.

        That is the research funnel's SCREEN stage. Declaring the grid and the \
        selection criterion is a manifest edit and can be done in any compute \
        mode; running the sweep executes wherever the compute target points. \
        The winning cell of a completed sweep is what "Create Agent" mints, \
        with the run, criterion, dev split and metrics recorded in the agent's \
        birth certificate.

        Frozen studies are immutable, so iterating on a grid means duplicating \
        the study in Studies — never editing a frozen one.
        """

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
        .accessibilityAddTraits(selectedRef == optimization.ref ? .isSelected : [])
        // The row's help says what CLICKING it does. The concept behind the
        // list lives on the section's ⓘ, where it is read once rather than
        // repeated identically on every row.
        .help("show '\(optimization.name)' below — its criterion, grid, "
            + "recommendations and the Create Agent edge")
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
        if panel.management.experiments.contains(where: { $0.name == name }) {
            panel.management.selectedName = name
        }
        panel.submission.remoteVerb = "sweep"
        panel.submission.remoteDryRun = false
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
        return Section {
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
            if let ratio = resolved.coherenceRatioToBaseline {
                LabeledContent("Coherence floor (relative to baseline)") {
                    Text(criterionValue(
                        "\(format(ratio))× the α=0 baseline's distinct-2",
                        isDefault: effective?.constraints?
                            .coherenceRatioToBaseline == nil))
                }
                LabeledContent("Coherence backstop (absolute distinct-2)") {
                    Text(criterionValue(
                        format(resolved.coherenceFloor),
                        isDefault: effective?.constraints?
                            .coherenceAbsoluteBackstop == nil))
                }
            } else {
                LabeledContent("Coherence floor (absolute distinct-2)") {
                    Text(criterionValue(
                        format(resolved.coherenceFloor),
                        isDefault: effective?.constraints?.coherenceFloor == nil))
                }
            }
            LabeledContent("Matched-norm random control") {
                Text(
                    resolved.matchedNormRandomMargin.map {
                        "margin \(format($0))"
                    } ?? "none declared")
            }
        } header: {
            InfoSectionHeader(
                title: "Declared criterion", text: Self.criterionInfo)
        }
    }

    private static let criterionInfo = """
        The criterion is the rule that picks a winning cell out of the grid, \
        declared BEFORE the sweep runs and hashed with the study. Four parts:

        • Objective — the number being maximised. judgeScore and logprobShift \
        are outcome instruments; markerDensity is a manipulation check and \
        never the promotion objective when the claim is about a substantive \
        outcome.
        • Capability tolerance — how far the capability battery may fall \
        below the no-injection baseline before a cell is ineligible.
        • Coherence floor — the distinct-bigram floor a cell's output must \
        clear. Declared either RELATIVE to the α=0 baseline (a ratio, plus an \
        absolute backstop no cell may fall below whatever the baseline was) or \
        as a plain absolute floor. Which of the two is in force is itself \
        declared data, and this panel never converts one into the other.
        • Matched-norm random control — when a margin is declared, the winner \
        must beat a norm-matched random direction by at least that much, or \
        the concept gets no recommendation at all.

        Values marked "(default)" are not on the manifest: the engine's \
        documented defaults are shown so the rule that will actually apply is \
        never left implicit.
        """

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
        if let manifest = panel.management.experiments.first(where: { $0.name == optimization.name }),
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
                // "Reload Spec" re-reads the manifest by re-creating the
                // editor: the keyed identity changes, so init reseeds every
                // field AND re-derives the review handle whose absence is what
                // the stale refusal is about.
                reload: {
                    refreshOptimizations()
                    specReloadToken += 1
                },
                runControls: { runSweepControls(optimization) })
                .id("\(optimization.id)#\(specReloadToken)")
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
                // Re-entry guard on the same condition the button reads: the
                // GPU-session dialog can park this submission, and a second
                // click while it is parked is a second durable job.
                guard sweepDisabledReason(optimization) == nil else { return }
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
            .help(sweepDisabledReason(optimization) ?? runSweepHelp(optimization))
            // A sweep is in flight on EITHER route: local MLX, or a durable
            // job on the server. The spinner used to follow the local flag
            // only, so a running server sweep looked idle.
            if panel.localJobs.isSweeping || panel.remoteJobs.activeSweepJob != nil {
                ProgressView()
                    .controlSize(.small)
            }
            if panel.localJobs.isSweeping || panel.remoteJobs.activeSweepJob != nil {
                let cancelling = panel.remoteJobs.activeSweepJob == nil
                    && panel.localJobs.sweepCancelRequested
                Button(
                    cancelling ? "Cancelling…" : "Cancel Optimization",
                    role: .destructive
                ) {
                    guard !cancelling else { return }
                    Task { await panel.cancelSweep() }
                }
                .disabled(cancelling)
                .help(
                    cancelling
                        ? "cancellation requested — the engine stops after the "
                            + "current generation; the rows already written "
                            + "stay in the run directory"
                        : "requests cancellation; the engine stops after the "
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
        .help("jump to Studies with '\(optimization.name)' selected and Submit "
            + "Bundle preconfigured for the sweep verb — nothing is submitted "
            + "from here")
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
        if panel.localJobs.isSweeping {
            return "a sweep is already running — follow it in the activity pane"
        }
        // The server route had no guard at all: the button stayed live while a
        // durable sweep job was in flight, and each click submitted another
        // one (the Cancel control beside it exists precisely because one IS
        // running).
        if let job = panel.remoteJobs.activeSweepJob {
            return "sweep job \(job.id) is running on \(substrate) — follow it "
                + "in the activity pane, or cancel it here first"
        }
        if panel.localJobs.isRunning || panel.localJobs.isValidating {
            return "another study task is running — wait for it to finish"
        }
        if optimization.source == .local {
            if !optimization.isDraft, optimization.sweep == nil {
                return "'\(optimization.name)' is not a draft "
                    + "(\(optimization.statusLabel)) and declares no sweep "
                    + "spec — the spec is pinned data; duplicate the study in "
                    + "Studies to declare one"
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

    private static let lifecycleInfo = """
        Where this study stands in the screen funnel. Each stage is DERIVED \
        from evidence on disk, never stored:

        • Declared — the manifest carries a sweep spec.
        • Optimized — a sweep run exists for it (or a stamped recommendation \
        proves one ran, even if the run directory has since been pruned).
        • Recommended — the criterion selected a winning cell for at least \
        one concept.
        • Agent created — some agent's birth certificate names this study.

        A stage can read "status unknown" rather than "no": a server's \
        listing does not expose every field, and reporting an unknown as a \
        false negative would be worse than saying so.
        """

    private func lifecycleSection(_ optimization: OptimizationItem) -> some View {
        let states = lifecycleStates(optimization)
        return Section {
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
        } header: {
            InfoSectionHeader(title: "Lifecycle", text: Self.lifecycleInfo)
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
            stageChip(
                "Declared", states.declared,
                done: "the manifest declares a layer×alpha sweep spec",
                todo: "no sweep spec on the manifest yet — declare one below")
            stageArrow
            stageChip(
                "Optimized", states.swept,
                done: "a sweep run exists for this study (or a stamped "
                    + "recommendation proves one ran)",
                todo: "the declared sweep has not run yet — use Optimize")
            stageArrow
            stageChip(
                "Recommended", states.recommended,
                done: "the declared criterion selected a winning cell",
                todo: "no winning cell yet — either the sweep has not run, or "
                    + "the constraints refused every cell")
            stageArrow
            stageChip(
                "Agent created", states.promoted,
                done: "an agent's birth certificate names this study",
                todo: "no agent has been minted from this study's winning cell")
        }
    }

    private var stageArrow: some View {
        Image(systemName: "arrow.right")
            .font(.caption2)
            .foregroundStyle(.tertiary)
    }

    /// nil state = not knowable on this substrate (rendered as "?", never as
    /// a false negative). Every state explains itself: the known ones used to
    /// carry an EMPTY tooltip, which is the common case.
    private func stageChip(
        _ label: String, _ state: Bool?, done: String, todo: String
    ) -> some View {
        HStack(spacing: 3) {
            Image(systemName: stageSymbol(state))
                .foregroundStyle(state == true ? Color.green : Color.secondary)
            Text(label)
        }
        .font(.caption)
        .help(stageHelp(state, done: done, todo: todo))
    }

    private func stageHelp(_ state: Bool?, done: String, todo: String) -> String {
        switch state {
        case true?: "done — " + done
        case false?: "not yet — " + todo
        case nil:
            "status unknown — \(substrate)'s listing does not carry this, so "
                + "it is reported as unknown rather than as a no"
        }
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
            Section("Optimization grid (α in norm units) — \(run.runName)") {
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
                    // Colour is never the only carrier: a failing cell is
                    // struck through (the same cue the measured grid uses)
                    // and a winner is badged in words.
                    Text(cellNumber(row, criterion: criterion))
                        .font(.caption.monospacedDigit())
                        .strikethrough(state == .failedConstraint)
                    if state == .winner {
                        Text("✓ winner")
                            .font(.caption2.weight(.bold))
                    }
                    if state == .failedConstraint {
                        Text("✕ fails")
                            .font(.caption2.weight(.bold))
                    }
                }
                .frame(minWidth: 56)
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
            .accessibilityAddTraits(selectedCell == cellID ? .isSelected : [])
            .help(cellHelp(row, state: state, criterion: criterion))
        } else {
            Text("—")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .frame(minWidth: 56)
                .help("no measured cell at this layer × alpha")
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
        // The RATIO is shown beside the raw distinct-2 whichever coherence
        // rule is in force, and the length flag beside them: a metric that
        // repetition can inflate should never be read without both.
        let ratioPart = row.distinct2Ratio.map {
            " (\(format($0))× baseline)"
        } ?? ""
        let lengthPart = row.lengthInflated
            ? ", ⚠︎ output over 1.5× baseline length"
            : ""
        let diagnostics = "density \(format(row.markerDensity)), "
            + "distinct-2 \(format(row.distinct2))\(ratioPart), battery "
            + "\(format(row.batteryAccuracy))\(lengthPart)"
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
            legendSwatch(.green.opacity(0.3), "✓ winner")
            legendSwatch(.secondary.opacity(0.1), "pass")
            legendSwatch(.red.opacity(0.16), "✕ fails constraint (struck through)")
            Text("click a cell to select it")
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
            // The recommendation row below already offers Create Agent for a
            // criterion-selected winner, and both buttons call the same
            // function — so the grid keeps its button only when no such row
            // exists for this concept in the loaded run.
            let hasRecommendationRow = recommendationRowExists(for: cell.concept)
            HStack(spacing: 8) {
                Text("selected: \(cell.concept) L\(cell.layer) α\(format(cell.alpha))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if isWinner, hasRecommendationRow {
                    Text("this is the criterion-selected winner — Create Agent "
                        + "for it is on its row under \"Recommended agent "
                        + "settings\" below")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                } else if isWinner {
                    Button(promotingConcept == cell.concept
                        ? "Creating Agent…" : "Create Agent") {
                        promoteFromDisplayedRun(
                            optimization: optimization, concept: cell.concept)
                    }
                    .disabled(sweepRun == nil || promotingConcept != nil)
                    .help(promoteWinnerHelp(optimization))
                    if promotingConcept == cell.concept {
                        ProgressView().controlSize(.small)
                    }
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
                    .disabled(promotingConcept != nil)
                    .help(
                        "this is NOT the winner under the declared criterion — "
                            + "creating an agent from it requires a written "
                            + "reason and is recorded as a manual override "
                            + "(promotedBy: manualOverride)")
                }
            }
            // Only when this concept has no recommendation row below — that
            // row renders the same outcome, and one message twice on one
            // screen reads as two events.
            if !hasRecommendationRow {
                promotionOutcomeLine(for: cell.concept)
            }
        }
    }

    /// Does the loaded run carry a criterion-selected recommendation row for
    /// this concept? (A manifest can hold provenance from an OLDER run than
    /// the one on screen, in which case there is no row and the grid keeps
    /// its own button.)
    private func recommendationRowExists(for concept: String) -> Bool {
        guard let run = sweepRun,
            case .selected = run.recommendations[concept]
        else { return false }
        return true
    }

    /// The Create Agent outcome, INLINE. Success and refusal used to speak
    /// only into the bottom status section and the bell.
    @ViewBuilder
    private func promotionOutcomeLine(for concept: String) -> some View {
        if let outcome = promotionOutcome, outcome.concept == concept {
            Label(
                outcome.message,
                systemImage: outcome.isFailure
                    ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(outcome.isFailure ? Color.orange : Color.green)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
            if !outcome.isFailure {
                Button("Open Agents") { navigate(.agents) }
                    .controlSize(.small)
                    .help("show the minted agent in the Agents library, with "
                        + "its birth certificate")
            }
        }
    }

    private func promoteWinnerHelp(_ optimization: OptimizationItem) -> String {
        let base = "mint an agent from the criterion-selected winning cell — "
            + "recorded as chosen by the declared criterion "
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
            Section("Measured grid (α in norm units) — \(run.runName)") {
                ForEach(SweepGridPresentation.concepts(rows: run.rows), id: \.self) { concept in
                    SweepGridView(
                        grid: SweepGridPresentation.grid(
                            concept: concept, rows: run.rows,
                            recommendation: run.recommendations[concept]))
                }
                measuredGridLegend
            }
        }
    }

    /// The measured grid's shading, border and strikethrough had no legend at
    /// all — only a per-cell tooltip. Same three states as the clickable grid
    /// above, said once for the whole section.
    private var measuredGridLegend: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 12) {
                HStack(spacing: 3) {
                    RoundedRectangle(cornerRadius: 3)
                        .strokeBorder(Color.accentColor, lineWidth: 1.5)
                        .frame(width: 12, height: 12)
                    Text("winner (outlined)")
                }
                HStack(spacing: 3) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.accentColor.opacity(0.3))
                        .frame(width: 10, height: 10)
                    Text("darker = higher objective")
                }
                HStack(spacing: 3) {
                    Text("0.00").strikethrough()
                    Text("fails a constraint")
                }
            }
            Text("Same cells as the grid above, shaded by objective value "
                + "rather than by constraint state. Hover a cell for its "
                + "objective, distinct-2, battery accuracy and verdict.")
                .fixedSize(horizontal: false, vertical: true)
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
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
                    // The outcome lands HERE, beside the button that caused
                    // it — not only in the status section far below and the
                    // bell.
                    promotionOutcomeLine(for: concept)
                }
                Spacer()
                let busy = promotingConcept == concept
                VStack(alignment: .trailing, spacing: 4) {
                    Button(busy ? "Creating Agent…" : "Create Agent") {
                        // Guarded like the grid-cell button: an unpinned
                        // promote is exactly the ambient resolution the
                        // contract removes.
                        promoteFromDisplayedRun(
                            optimization: optimization, concept: concept)
                    }
                    .disabled(sweepRun == nil || promotingConcept != nil)
                    .help(promoteRecommendationHelp(optimization))
                    if busy {
                        ProgressView().controlSize(.small)
                    }
                }
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
        // Re-entry guard on the same flag the buttons read: a server promote
        // is a durable mint, and a second click was a second one.
        guard promotingConcept == nil else { return }
        guard
            let pins = pinsForDisplayedRun(
                experimentName: optimization.name, concept: concept)
        else {
            let refusal = "cannot promote '\(concept)': this view has no "
                + "loaded sweep run to pin the promotion to. Reload the "
                + "optimization, or use the CLI's explicit --sweep-run if you "
                + "mean to name the evidence yourself"
            panel.refuse(.sweepSpec, refusal)
            promotionOutcome = PromotionOutcome(
                concept: concept, message: refusal, isFailure: true)
            return
        }
        promotionOutcome = nil
        promotingConcept = concept
        let mark = panel.notices.notices.last?.id
        let route = promotionRoute(optimization)
        Task {
            if route == .activeServer {
                // The same call `panel.promote` makes for this route, awaited
                // so the busy state and the inline outcome are real.
                await panel.promoteOnActiveServer(
                    experimentName: optimization.name, concept: concept,
                    pins: pins)
            } else {
                panel.promote(
                    experimentName: optimization.name, concept: concept,
                    route: .local, pins: pins)
            }
            promotingConcept = nil
            promotionOutcome = Self.outcome(
                concept: concept, after: mark, in: panel.notices.notices)
        }
    }

    /// The override path, given the same busy state, re-entry guard and
    /// inline outcome as the criterion path. Pins were CAPTURED when the sheet
    /// opened; re-deriving them here would silently promote unpinned if the
    /// loaded run changed or cleared while the sheet was up.
    private func promoteOverride(_ promotion: OverridePromotion, reason: String) {
        guard promotingConcept == nil else { return }
        promotionOutcome = nil
        promotingConcept = promotion.concept
        let mark = panel.notices.notices.last?.id
        let cell = (layer: promotion.layer, alpha: promotion.alpha)
        Task {
            if promotion.mintsOnServer {
                await panel.promoteOnActiveServer(
                    experimentName: promotion.experiment,
                    concept: promotion.concept,
                    cell: cell, overrideReason: reason, pins: promotion.pins)
            } else {
                panel.promote(
                    experimentName: promotion.experiment,
                    concept: promotion.concept,
                    cell: cell, overrideReason: reason, route: .local,
                    pins: promotion.pins)
            }
            promotingConcept = nil
            promotionOutcome = Self.outcome(
                concept: promotion.concept, after: mark,
                in: panel.notices.notices)
        }
    }

    /// What the promotion said, read off the notices it recorded. Verdict
    /// notices (success/warning/error) win over the "promoting…" chatter and
    /// over anything a follow-up refresh appended.
    private static func outcome(
        concept: String, after mark: UUID?, in notices: [PanelNotice]
    ) -> PromotionOutcome? {
        var fresh = notices
        if let mark, let index = notices.firstIndex(where: { $0.id == mark }) {
            fresh = Array(notices[notices.index(after: index)...])
        }
        let decisive = fresh.last {
            $0.severity == .success || $0.severity == .error
                || $0.severity == .warning
        }
        guard let notice = decisive ?? fresh.last else { return nil }
        return PromotionOutcome(
            concept: concept,
            message: notice.message,
            isFailure: notice.severity == .error || notice.severity == .warning)
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

    private func format(_ value: Double) -> String { AlphaFormat.text(value) }
}

// MARK: - Sweep spec editor (draft manifests only)

/// The "Optimization (sweep) spec" editor for a SELECTED draft optimization. Its @State
/// is seeded from the saved spec in init — the parent keys this view by
/// experiment name (`.id`) so switching optimizations reseeds the fields. Saving
/// goes through `ExperimentPanel.setSweepSpec` (draft-only, criterion
/// validated at save via `SweepSpecForm`).
private struct SweepSpecEditorSection<RunControls: View>: View {
    @State private var reviewed: DraftAuthoringSnapshot?
    let experimentName: String
    let panel: ExperimentPanel
    let onSaved: () -> Void
    /// Ask the host to re-create this editor from the manifest on disk.
    let reload: () -> Void
    @ViewBuilder let runControls: () -> RunControls
    /// Set when Reload Spec was pressed on a dirty editor.
    @State private var confirmReload = false

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
    /// Non-nil = the criterion under edit declares the baseline-relative
    /// coherence floor. Carried, never edited here.
    @State private var coherenceRatio: Double?
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
    /// The pinned model's depth from any extracted vector's sidecar, seeded
    /// once in init (the catalog scan is disk I/O — not per keystroke). nil
    /// until something has been extracted for the model; the caption states
    /// that rather than guessing.
    private let cachedLayerCount: Int?

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
        reload: @escaping () -> Void,
        @ViewBuilder runControls: @escaping () -> RunControls
    ) {
        self.experimentName = experimentName
        self.panel = panel
        self.onSaved = onSaved
        self.reload = reload
        self.runControls = runControls
        let initial = spec ?? ExperimentManifest.SweepSpec()
        let source = try? DraftAuthoringSnapshot(workspaceRoot: ExperimentStore.workspaceRoot, name: experimentName)
        _reviewed = State(initialValue: (source?.manifest.sweep ?? .init()) == initial ? source : nil)
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
        let manifest = panel.management.experiments.first { $0.name == experimentName }
        let manifestConcepts = (manifest?.concepts ?? []).map(\.name)
        // Model id AND pinned revision: the depth this editor displays
        // absolute layers against must be the depth of the revision the
        // manifest pins, not of whatever else this workspace has extracted
        // for the same checkpoint (review round 7, finding 4).
        self.cachedLayerCount = manifest.flatMap {
            SweepPanelModel.cachedLayerCount(
                modelID: $0.modelID, revision: $0.modelRevision)
        }
        if !manifestSingular.isEmpty, manifestConcepts.count > 1 {
            for concept in manifestConcepts where seededMap[concept] == nil {
                seededMap[concept] = manifestSingular
            }
        }
        _choiceFilesByConcept = State(initialValue: seededMap)
        let tolerance = initial.selection?.constraints?.capabilityTolerance
            ?? SweepSelectionRule.defaultCapabilityTolerance
        _toleranceText = State(initialValue: "\(tolerance)")
        // Which coherence RULE this criterion declares is not editable here,
        // and is never silently converted: the ratio travels through the
        // editor untouched, and the one number the field edits is whichever
        // absolute value that rule uses — the backstop under the relative
        // rule, the floor under the legacy one. Read through the shared
        // presence rule (`SweepSpecForm.editorCoherenceForm`), so a criterion
        // that declared a backstop and left the ratio to the default arrives
        // here as the RELATIVE criterion it is — before, it arrived with a
        // nil ratio and saving converted it to a legacy absolute floor
        // (review round 9, finding 4).
        let form = SweepSpecForm.editorCoherenceForm(initial.selection?.constraints)
        _coherenceRatio = State(initialValue: form.ratio)
        let floor = form.floor
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

    private static let specInfo = """
        Everything the sweep will do, as manifest data. The GRID is the cost: \
        each layer fraction × each alpha is one generated cell per dev prompt, \
        plus the capability battery, and the no-injection baseline cell is \
        always implied. Layer fractions are network-depth fractions (0–1) \
        resolved against the pinned model's depth; alphas are steering \
        strengths in residual-norm units, never raw vector multiples.

        The CRITERION is the rule that then picks a winning cell — objective, \
        capability tolerance, coherence floor, and the optional matched-norm \
        random control.

        Saving validates the whole block the way freeze will, so a criterion \
        whose instrument is missing is refused here rather than at sweep \
        start. Optimize executes the SAVED spec, which is why it stays \
        unreachable while the editor holds unsaved edits. Freeze then pins \
        this spec and the SHA-256 of the dev-prompts and battery files; after \
        that, drift in those bytes refuses sweep start.
        """

    var body: some View {
        Section {
            gridFields
            criterionFields
            saveControls
            // Finding 5: Optimize executes the SAVED spec, so running with
            // unsaved edits silently sweeps something other than what is on
            // screen. Rather than warn, make it unreachable.
            runControls()
                .disabled(isDirty)
        } header: {
            InfoSectionHeader(
                title: "Optimization (sweep) spec — draft, editable",
                text: Self.specInfo)
        }
    }

    @ViewBuilder
    private var gridFields: some View {
        TextField("Layer fractions (0–1, comma-separated)", text: $layerFractionsText)
            .help("network-depth fractions the sweep maps to layers, e.g. 0.5, 0.7, 0.85")
        resolvedLayersCaption
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
            .help("how long each cell's generation may run — the same budget "
                + "for every cell, so the grid stays comparable; it multiplies "
                + "straight into the sweep's cost")
    }

    /// §4.15(b): the grid is a cost and a preregistration, so the human
    /// deciding it is shown the fractions AND the absolute layers they
    /// resolve to at this model's depth, side by side, LIVE as they edit —
    /// not only after a save. Hidden while the text does not parse or names
    /// an out-of-range fraction (the form's own parse error and the save
    /// refusal own those); collapse onto fewer layers is stated by the
    /// shared caption text.
    @ViewBuilder
    private var resolvedLayersCaption: some View {
        if let fractions = SweepSpecForm.parseNumberList(layerFractionsText),
            fractions.allSatisfy({ $0 >= 0 && $0 <= 1 })
        {
            Text("→ " + SweepSpecForm.resolvedLayersText(
                fractions: fractions, layerCount: cachedLayerCount))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
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
        TextField(
            coherenceRatio == nil
                ? "Coherence floor (absolute distinct-2, 0–1)"
                : "Coherence backstop (absolute distinct-2, 0–1)",
            text: $floorText
        )
        .help(
            coherenceRatio.map {
                "this criterion gates coherence at \($0)× the α=0 baseline's "
                    + "distinct-2; no cell passes below this absolute backstop "
                    + "whatever the baseline was"
            } ?? "cells below this distinct-bigram ratio fail the constraint")
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
        (panel.management.experiments.first { $0.name == experimentName }?.concepts ?? [])
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
        let manifest = panel.management.experiments.first { $0.name == experimentName }
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
            .help("write the grid and criterion into this draft's manifest — "
                + "validated the way freeze will validate it; Optimize then "
                + "executes exactly what is saved (⌘S)")
            // The repair the stale-review refusal names, as a button: the
            // editor is re-created from the manifest, which is the only way
            // to re-derive the review handle it needs.
            Button("Reload Spec") {
                if isDirty { confirmReload = true } else { reload() }
            }
            .help("re-read the saved spec from the manifest, discarding the "
                + "unsaved edits in this editor (it asks first when there are "
                + "any) — this is the repair for \"the displayed sweep is "
                + "stale or unavailable\"")
            .confirmationDialog(
                "Discard the unsaved sweep-spec edits to '\(experimentName)'?",
                isPresented: $confirmReload
            ) {
                Button("Discard and reload", role: .destructive) { reload() }
                Button("Keep editing", role: .cancel) {}
            } message: {
                Text("The grid and criterion values you have changed here are "
                    + "not in the manifest and cannot be recovered. The saved "
                    + "spec is re-read from disk.")
            }
            Spacer()
        }
        // Precedence: this form's own parse refusal, then the engine's
        // refusal from `setSweepSpec` — which used to speak ONLY into the
        // panel-top notice area (finding 11a), several hundred points above
        // this button.
        if let refusal = formError ?? panel.draft.formErrors[.sweepSpec] {
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
            // The exact inverse of the read above, so the form this editor
            // loaded is the form it writes back.
            constraints: SweepSpecForm.editorCoherenceConstraints(
                capabilityTolerance: tolerance,
                form: .init(ratio: coherenceRatio, floor: floor)),
            controls: controls)
        formError = nil
        // setSweepSpec refuses structural/criterion problems; since 2026-07-26
        // it records them in `panel.formErrors[.sweepSpec]` as well as the
        // notice feed, so `saveControls` can render them next to the button.
        guard let reviewed else {
            formError = "The displayed sweep is stale or unavailable — press "
                + "Reload Spec to re-read this draft's saved spec from the "
                + "manifest. Your edits here are not saved."
            return
        }
        if panel.setSweepSpec(spec, reviewed: reviewed, onSaved: { self.reviewed = $0 }) {
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
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .help("close without declaring anything")
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
            .help("leave this sheet and create a study in Studies — an "
                + "optimization is declared on a draft study")
        } else {
            Picker("Draft study", selection: $candidate) {
                ForEach(drafts, id: \.self) { name in
                    Text(name).tag(String?.some(name))
                }
            }
            .help("which draft study gets the sweep grid and selection "
                + "criterion — only drafts without one are listed, because the "
                + "spec is pinned data once a study is frozen")
            objectivePicker
            objectiveCaption
            HStack(spacing: 8) {
                Button("Declare Optimization") {
                    guard let candidate, let objective else { return }
                    dismiss()
                    declare(candidate, objective)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(candidate == nil || objective == nil)
                .help(objective == nil
                    ? "choose a selection objective first — there is no "
                        + "default, because the criterion is pre-declared data"
                    : "write the default layer×alpha grid and this objective "
                        + "into the chosen draft's manifest, then open the spec "
                        + "editor on it")
                Button("New draft in Studies…") {
                    dismiss()
                    openStudies()
                }
                .help("leave this sheet and create a study in Studies — come "
                    + "back here to declare its sweep")
            }
            Text("declares the default grid with the chosen objective — edit "
                + "everything in the spec editor that opens on the new run")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    /// Short menu labels: the two outcome instruments carried the SAME
    /// sentence-long recommendation, in a 440 pt sheet, and `objectiveCaption`
    /// below already says which kind each one is.
    private var objectivePicker: some View {
        Picker("Selection objective", selection: $objective) {
            Text("choose…").tag(String?.none)
            Text("judge score (outcome)").tag(String?.some("judgeScore"))
            Text("logprob shift (outcome)").tag(String?.some("logprobShift"))
            Text("marker density (manipulation check)")
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
                .fixedSize(horizontal: false, vertical: true)
        }
        if objective == "judgeScore" || objective == "logprobShift" {
            Text("an outcome instrument — recommended when the claim is about "
                + "a substantive outcome. It needs its instrument on the "
                + "draft: judgeScore a pinned rubric + judges (Studies › "
                + "Evaluation), logprobShift a choice-prompts file (spec "
                + "editor) — Declare surfaces the engine's refusal if they are "
                + "missing")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

// MARK: - Supporting types

/// ONE number formatter for this surface. The override sheet used to
/// interpolate the raw `Double`, so the same cell could read "α0.13" in the
/// grid and "α0.13000000000000001" in the dialog that promotes it.
enum AlphaFormat {
    static func text(_ value: Double) -> String {
        value.formatted(.number.precision(.fractionLength(0 ... 3)))
    }
}

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
                "Cell L\(promotion.layer) α\(AlphaFormat.text(promotion.alpha)) "
                    + "of '\(promotion.concept)' is not the winner under the "
                    + "declared criterion.")
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
            Label(
                "This agent will be recorded as a manual override "
                    + "(promotedBy: manualOverride).",
                systemImage: "exclamationmark.triangle.fill")
                .font(.callout)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
            TextField("Reason (required — stamped into the birth certificate)",
                      text: $reason, axis: .vertical)
                .lineLimit(2 ... 4)
                .textFieldStyle(.roundedBorder)
                .help("why this cell and not the criterion's winner — stored "
                    + "verbatim in the agent's birth certificate and read by "
                    + "everyone who later asks how it was chosen")
            HStack {
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .help("close without minting anything")
                Spacer()
                Button("Create Agent with override") {
                    promote(trimmedReason)
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(trimmedReason.isEmpty)
                .help(trimmedReason.isEmpty
                    ? "write a reason first — an override without one is "
                        + "exactly the undocumented deviation this path exists "
                        + "to prevent"
                    : "mint an agent from this non-winning cell, recorded as a "
                        + "manual override with the reason above")
            }
        }
        .padding(16)
        .frame(minWidth: 420)
    }
}
