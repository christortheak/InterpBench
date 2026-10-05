import ExperimentKit
import Foundation
import SwiftUI

/// The launch screen: a compact dashboard of REAL workspace state (no
/// marketing) — where am I, what compute, what models, what agents, what's
/// running, what exists, and what to do next. Every empty state carries its
/// next action (design brief › Empty States).
///
/// It opens with the first-study checklist (2026-10 release review, A6): the
/// seven steps from a workspace to exported results, each judged from what
/// the workspace holds. The cards below follow the same order — workspace,
/// compute, models, studies — with the advanced ways to make an agent last.
struct HomeDashboardView: View {
    @Bindable var service: ChatService
    let workspace: WorkspaceStore
    /// New, Open, and the Demo Workspace: the same actions the toolbar and
    /// Research Setup use.
    let actions: WorkspaceActions
    let navigate: (WorkbenchSection) -> Void
    /// Lands on Agents → Optimizations (declared sweep runs).
    var openOptimizations: () -> Void = {}
    /// Lands on Data → Adapter Training. A plain `navigate(.data)` lands on
    /// whichever Data tool was last shown, so a button named after a tool
    /// needs the tool-setting route (2026-09-06 audit).
    var openAdapterTraining: () -> Void = {}
    /// Opens ONE agent: selects it, then lands on Agents → Library.
    var openAgent: (ModelVariantRecord.ID) -> Void = { _ in }
    /// The three compute choices, from the environment the main window sets.
    @Environment(ComputeChoiceCoordinator.self) private var compute:
        ComputeChoiceCoordinator?
    /// What the checklist read from the workspace folder, or nil before the
    /// first read lands.
    @State private var scannedFacts: FirstStudyChecklist.Facts?
    /// The researcher's own Show/Hide choice for this visit; nil follows the
    /// rule (shown while steps remain, hidden once all are done).
    @State private var checklistShownOverride: Bool?
    @State private var showingDemos = false

    var body: some View {
        Form {
            firstStudySection
            workspaceSection
            computeSection
            // WS3: the cluster-chores card, shown only when a cluster site is
            // the active scope. All verdicts come from ExperimentKit; the
            // evidence ledger root follows the importer's workspace
            // resolution (VectorCatalog.projectRoot).
            if isServerWorkspace {
                ClusterHealthCard(service: service)
            }
            modelsSection
            studiesSection
            jobsSection
            agentsSection
            advancedSection
        }
        .formStyle(.grouped)
        // Both scans are IO that nothing on the appearance path needs before
        // the dashboard draws. `.task` runs after the first draw, the agent
        // scan runs off the main actor (latest-wins inside the panel), and
        // the previous visit's rows stay visible while they land — the study
        // refresh used to run synchronously in `.onAppear`, ahead of the
        // first frame.
        .task {
            service.experiments.refresh()
            service.fineTuning.refreshAgentLibraryAsync()
        }
        // The checklist's own read, again whenever the workspace changes (a
        // Demo Workspace opened from here switches it) and whenever work
        // that was running finishes, since a run may just have completed.
        .task(id: workspace.rootURL) {
            // Another workspace: its own steps, shown by the rule again.
            checklistShownOverride = nil
            await rescanChecklist()
        }
        .onChange(of: runningItems.map(\.id)) { _, now in
            if now.isEmpty { Task { await rescanChecklist() } }
        }
        .sheet(isPresented: $showingDemos, onDismiss: { actions.demoSheetClosed() }) {
            DemoWorkspaceSheet(
                demos: DemoWorkspace.available(), open: { try actions.openDemoWorkspace($0) })
        }
    }

    // MARK: First study

    /// The checklist's facts: the folder's, read off the main actor, with
    /// the two the app knows live — the model inventory of the engine in use,
    /// and where the workspace is set to run (re-read on every draw, so
    /// choosing a place here shows at once).
    private var checklistFacts: FirstStudyChecklist.Facts {
        var facts = scannedFacts ?? FirstStudyChecklist.Facts(hasWorkspace: true)
        // A developer build standing on its own checkout has no workspace of
        // the researcher's: the first step stays open, as the warning in the
        // Workspace card says.
        facts.hasWorkspace = workspace.chosenRootURL != nil
        facts.modelCount = service.workspaceModelOptions.count
        facts.computeChoice =
            workspace.isComputeDeclared ? (compute?.workspaceChoice ?? facts.computeChoice) : nil
        return facts
    }

    private func rescanChecklist() async {
        let root = workspace.rootURL
        let facts = await Task.detached(priority: .utility) {
            FirstStudyChecklist.scan(root: root, carriedDemos: DemoWorkspace.carriedRoot())
        }.value
        // A switch while the read ran: the newer read owns the state.
        guard root == workspace.rootURL else { return }
        scannedFacts = facts
    }

    /// Whether the checklist's steps are showing: while steps remain, unless
    /// the researcher hid them for this visit; once all are done, only when
    /// asked for.
    private func checklistShown(_ items: [FirstStudyChecklist.Item]) -> Bool {
        checklistShownOverride ?? (FirstStudyChecklist.nextStep(items) != nil)
    }

    private var firstStudySection: some View {
        let items = FirstStudyChecklist.items(checklistFacts)
        let allDone = FirstStudyChecklist.nextStep(items) == nil
        let shown = checklistShown(items)
        // Home is rebuilt on every visit, so until this visit's read lands
        // the section says it is checking rather than flashing steps as
        // not done that are done.
        let checked = scannedFacts != nil
        return Section {
            if !checked {
                Text("Checking this workspace…")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if shown {
                Text(FirstStudyChecklist.introduction)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                FirstStudyChecklistRows(
                    items: items, hasWorkspace: workspace.hasWorkspace,
                    workspacePinned: workspace.isEnvironmentPinned,
                    perform: perform,
                    chooseCompute: compute.map { coordinator -> (ComputeChoice) -> Void in
                        { choice in coordinator.choose(choice) }
                    })
            } else {
                Text(allDone ? FirstStudyChecklist.finished : FirstStudyChecklist.progress(items) + ".")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        } header: {
            HStack(spacing: 8) {
                Text(FirstStudyChecklist.title)
                if checked {
                    Text(FirstStudyChecklist.progress(items))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button(shown ? "Hide Steps" : "Show Steps") {
                        checklistShownOverride = !shown
                    }
                    .buttonStyle(.link)
                    .controlSize(.small)
                    .help(
                        shown
                            ? "fold the checklist away for this visit; it opens again "
                                + "while steps remain"
                            : "show the seven steps again")
                }
            }
        }
    }

    /// One checklist button, performed with the same actions the rest of the
    /// app uses for it.
    private func perform(_ action: FirstStudyChecklist.Action) {
        switch action {
        case .newWorkspace: actions.newWorkspace()
        case .openWorkspace: actions.openWorkspace()
        case .chooseCompute: actions.showingResearchSetup = true
        case .openPlayground: navigate(.playground)
        case .openDemoWorkspace: showingDemos = true
        case .openStudies(let name):
            if let name { service.experiments.management.selectedName = name }
            navigate(.studies)
        case .openResults: navigate(.results)
        }
    }

    // MARK: Workspace

    private var workspaceSection: some View {
        Section("Workspace") {
            LabeledContent("Folder", value: workspace.displayName)
            Text(workspace.rootURL.path)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .lineLimit(1)
                .truncationMode(.middle)
                // Middle-truncated: the whole path has to stay reachable.
                .help(workspace.rootURL.path)
            if workspace.isLegacyRepoRoot {
                Label(
                    "running against the code checkout (dev fallback) — create a "
                        + "workspace in the toolbar to keep study data out of the "
                        + "source tree",
                    systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            // The Demo Workspace has one button on Home at a time: the
            // checklist's fourth step offers it while the steps show, and
            // this card offers it once they are hidden.
            if scannedFacts != nil, !checklistShown(FirstStudyChecklist.items(checklistFacts)) {
                Button(DemoWorkspaceCopy.button) { showingDemos = true }
                    .controlSize(.small)
                    .disabled(workspace.isEnvironmentPinned)
                    .help(
                        workspace.isEnvironmentPinned
                            ? DemoWorkspaceCopy.unavailableWhilePinned
                            : "open a copy of a finished study, with a draft to run, "
                                + "in a folder you choose")
            }
        }
    }

    // MARK: Compute

    private var isServerWorkspace: Bool {
        service.cluster.computeTarget == .server
    }

    private var computeSection: some View {
        Section("Compute") {
            // One name for each place everywhere: the three compute choices,
            // as the Compute and Workspace menus say them.
            LabeledContent("Running on", value: service.cluster.activeComputeTitle)
            Text(service.cluster.activeComputeChoice.summary)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if isServerWorkspace {
                LabeledContent(
                    "Connection", value: service.cluster.status ?? "not connected")
            } else {
                LabeledContent("Connection", value: "in this app — nothing to connect")
            }
            // The workspace is set to one engine and the app is using the
            // other: the full sentence, where there is room to read it.
            if let mismatch = compute?.mismatchNote {
                Label(mismatch, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 8) {
                Button("Open Compute") { navigate(.compute) }
                    .help(
                        "the Compute section: connections, jobs, logs, and "
                            + "model installs")
                if let compute {
                    Button(ComputeGuide.guideButton) { compute.showingGuide = true }
                        .help(
                            "what each of the three places can run, and what "
                                + "switching between them costs")
                }
            }
            .controlSize(.small)
        }
    }

    // MARK: Models

    private var loadedModelLine: String? {
        if isServerWorkspace {
            return service.serverDefaultLoadedModelID
        }
        return service.loadedModelID
    }

    private var modelsSection: some View {
        Section("Models") {
            let available = service.workspaceModelOptions
            if let loaded = loadedModelLine {
                LabeledContent("Loaded", value: loaded)
            } else {
                // Name the compute target the way the Compute card above
                // names it — the host:port spelling read as a third place
                // (2026-09-06 audit, headline 18).
                Text(
                    isServerWorkspace
                        ? "No model is loaded on \(service.cluster.substrateLabel)."
                        : "No model loaded.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            LabeledContent(
                "Available",
                value: "\(available.count) model\(available.count == 1 ? "" : "s")")
            HStack(spacing: 8) {
                Button("Open Playground") { navigate(.playground) }
                    .controlSize(.small)
                    .help("select and load a model in the Playground's Model section")
                if isServerWorkspace {
                    InstallModelButton(cluster: service.cluster)
                        .controlSize(.small)
                }
            }
        }
    }

    // MARK: Agents

    private var recentAgents: [ModelVariantRecord] {
        Array(service.fineTuning.variants.prefix(4))
    }

    private var agentsSection: some View {
        Section("Recent agents") {
            if recentAgents.isEmpty {
                Text(
                    "No agents yet. Create one in Agents → New Agent — by "
                        + "hand, or by optimizing a concept vector.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button("Open Agents") { navigate(.agents) }
                    .controlSize(.small)
                    .help("the Agents section — library, New Agent, and optimization runs")
            } else {
                ForEach(recentAgents) { record in
                    agentRow(record)
                }
                Button("Open Agent Library") { navigate(.agents) }
                    .controlSize(.small)
                    .help("Agents → Library: every saved agent, with its readiness chips")
            }
        }
    }

    // MARK: Advanced

    /// The two expert ways to make an agent, which used to sit beside "Open
    /// Agents" as if a first study needed them (2026-10 release review, A6).
    /// They keep one home each, here, below the basics; the sidebar's
    /// Advanced group holds the advanced sections themselves.
    private var advancedSection: some View {
        Section("Advanced") {
            Text(
                "Beyond a first study: search layers and strengths for the best "
                    + "steering point, or train an adapter from a dataset. Probes, "
                    + "Multi-Agent, and Analysis are under Advanced in the sidebar.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            // Two routes at Home's 420 pt floor: `ViewThatFits` keeps them on
            // one line where there is room and stacks them where there is
            // not, instead of clipping.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) { advancedButtons }
                VStack(alignment: .leading, spacing: 6) { advancedButtons }
            }
            .controlSize(.small)
        }
    }

    @ViewBuilder
    private var advancedButtons: some View {
        Button("Optimize") { openOptimizations() }
            .help(
                "Agents → Optimizations: declare a run that searches layers "
                    + "and strengths for the best steering point")
        Button("Train Adapter") { openAdapterTraining() }
            .help("Data → Adapter Training: fine-tune a LoRA adapter from a dataset")
    }

    /// Context-carrying, like the study rows below: an agent row IS a link to
    /// that agent — it selects it and opens Agents → Library, where the
    /// browser can show the selection.
    private func agentRow(_ record: ModelVariantRecord) -> some View {
        Button {
            openAgent(record.id)
        } label: {
            agentRowLabel(record)
        }
        .buttonStyle(.plain)
        .help("open '\(record.artifact.name)' in Agents → Library")
    }

    private func agentRowLabel(_ record: ModelVariantRecord) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(record.artifact.name)
                    .font(.callout.weight(.medium))
                Text(record.artifact.baseModelID)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            AgentKindBadge(kind: AgentLibrary.kind(of: record.artifact))
            Spacer()
            Text(shortDate(record.artifact.createdAt))
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
        }
        .contentShape(Rectangle())
    }

    // MARK: Jobs

    private struct RunningItem: Identifiable {
        let id: String
        let label: String
        let systemImage: String
    }

    private var runningItems: [RunningItem] {
        var items: [RunningItem] = []
        if service.isGenerating {
            items.append(.init(id: "chat", label: "chat generation", systemImage: "text.cursor"))
        }
        if service.fineTuning.isTraining {
            items.append(
                .init(
                    id: "train",
                    label: service.fineTuning.trainingProgress ?? "adapter training",
                    systemImage: "slider.horizontal.2.square.on.square"))
        }
        if service.fineTuning.isRobustnessRunning {
            items.append(
                .init(id: "robust", label: "robustness check", systemImage: "checklist.checked"))
        }
        if service.experiments.localJobs.isRunning || service.experiments.localJobs.isValidating
            || service.experiments.localJobs.isEvaluating
        {
            items.append(.init(id: "study", label: "study task", systemImage: "checkmark.seal"))
        }
        if service.multiAgent.isRunning {
            items.append(
                .init(id: "scenario", label: "multi-agent run", systemImage: "person.3.sequence"))
        }
        if let job = service.experiments.remoteJobs.activeServerJob {
            items.append(
                .init(
                    id: "server-\(job.id)",
                    label: "server \(job.verb) job \(job.id) ('\(job.study)')",
                    systemImage: "server.rack"))
        }
        return items
    }

    private var jobsSection: some View {
        Section("Running now") {
            if runningItems.isEmpty {
                Text("Nothing running.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(runningItems) { item in
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.mini)
                        Label(item.label, systemImage: item.systemImage)
                            .font(.caption)
                    }
                }
            }
            if isServerWorkspace, let server = service.cluster.activeServer,
                let badge = service.cluster.runningJobsBadge(for: server.id)
            {
                LabeledContent("Server jobs (last check)", value: badge)
                    .font(.caption)
            }
            // No second "Open Compute" here: the Compute card above already
            // carries it, and Home showed the same button three times
            // (2026-09-06 audit, headline 23).
        }
    }

    // MARK: Studies

    private var recentStudies: [ExperimentManifest] {
        Array(service.experiments.management.experiments.prefix(4))
    }

    private var studiesSection: some View {
        Section("Recent studies") {
            if recentStudies.isEmpty {
                Text(
                    "No study protocols in this workspace yet — the Studies "
                        + "section is where the first draft is created.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                // "Create study" created nothing; it navigated. One honest
                // label for one action (2026-09-06 audit, headline 23).
                Button("Open Studies") { navigate(.studies) }
                    .controlSize(.small)
                    .help("the Studies section, where a draft study is created and edited")
            } else {
                ForEach(recentStudies, id: \.name) { manifest in
                    studyRow(manifest)
                }
                Button("Open Studies") { navigate(.studies) }
                    .controlSize(.small)
                    .help("the Studies section: draft, freeze, run, and import evidence")
            }
        }
    }

    /// Context-carrying: a study row IS a link to that study — it opens
    /// Studies with the study selected, never a bare navigate.
    private func studyRow(_ manifest: ExperimentManifest) -> some View {
        Button {
            service.experiments.management.selectedName = manifest.name
            navigate(.studies)
        } label: {
            studyRowLabel(manifest)
        }
        .buttonStyle(.plain)
        .help("open '\(manifest.name)' in Studies")
    }

    private func studyRowLabel(_ manifest: ExperimentManifest) -> some View {
        // A display label leads; the canonical name stays visible because
        // run directories and logs speak only that.
        let display = service.experiments.management.displayName(manifest)
        return HStack(alignment: .firstTextBaseline, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text(display)
                    .font(.callout.weight(.medium))
                Text(
                    display == manifest.name
                        ? manifest.modelID
                        : "\(manifest.name) · \(manifest.modelID)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Text(manifest.status.rawValue)
                .font(.caption2.weight(.semibold))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Capsule().fill(statusColor(manifest.status)))
            if manifest.sweep != nil {
                Text("optimization")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .help("this study declares a sweep — visible in Agents → Optimizations")
            }
            Spacer()
            Text(shortDate(manifest.createdAt))
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
        }
        .contentShape(Rectangle())
    }

    private func statusColor(_ status: ExperimentManifest.Status) -> Color {
        switch status {
        case .draft: .secondary.opacity(0.14)
        case .frozen: .blue.opacity(0.16)
        case .complete: .green.opacity(0.18)
        }
    }

    // MARK: Dates

    // A "Next actions" section used to sit here, repeating Open Playground,
    // Optimize, Create study and Open Compute — four buttons that every card
    // above already offers, three of them without help, in one non-wrapping
    // row at Home's 420 pt floor. Each action now has exactly one home, in
    // the card it belongs to (2026-09-06 audit, headline 23).

    /// Artifact timestamps are ISO-8601 UTC, in the two spellings the
    /// producing stores emit (with and without fractional seconds). Render
    /// them in the researcher's own locale and time zone instead of slicing
    /// the string by hand, which showed UTC without saying so; an
    /// unparseable stamp falls back to its own text, never to a wrong date.
    /// Formatters are built per call — `ISO8601DateFormatter` is not
    /// `Sendable`, and this runs at most eight times per dashboard draw.
    private func shortDate(_ iso: String) -> String {
        guard !iso.isEmpty else { return "—" }
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let parsed = fractional.date(from: iso) ?? ISO8601DateFormatter().date(from: iso)
        guard let parsed else { return iso }
        return parsed.formatted(date: .abbreviated, time: .shortened)
    }
}
