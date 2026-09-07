import ExperimentKit
import Foundation
import SwiftUI

/// The launch screen: a compact dashboard of REAL workspace state (no
/// marketing) — where am I, what compute, what models, what agents, what's
/// running, what exists, and what to do next. Every empty state carries its
/// next action (design brief › Empty States).
struct HomeDashboardView: View {
    @Bindable var service: ChatService
    let workspace: WorkspaceStore
    let navigate: (WorkbenchSection) -> Void
    /// Lands on Agents → Optimizations (declared sweep runs).
    var openOptimizations: () -> Void = {}
    /// Lands on Data → Adapter Training. A plain `navigate(.data)` lands on
    /// whichever Data tool was last shown, so a button named after a tool
    /// needs the tool-setting route (2026-09-06 audit).
    var openAdapterTraining: () -> Void = {}
    /// Opens ONE agent: selects it, then lands on Agents → Library.
    var openAgent: (ModelVariantRecord.ID) -> Void = { _ in }

    var body: some View {
        Form {
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
            agentsSection
            jobsSection
            studiesSection
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
        }
    }

    // MARK: Compute

    private var isServerWorkspace: Bool {
        service.cluster.computeTarget == .server
    }

    private var computeSection: some View {
        Section("Compute") {
            LabeledContent("Target", value: service.cluster.substrateLabel)
            if isServerWorkspace {
                LabeledContent(
                    "Connection", value: service.cluster.status ?? "not connected")
            } else {
                // One spelling for the substrate everywhere ("Local (MLX)",
                // as `substrateLabel` and `WorkspaceCompute.label` say it).
                LabeledContent("Connection", value: "in this app — no server")
            }
            Button("Open Compute") { navigate(.compute) }
                .controlSize(.small)
                .help(
                    "the Compute section: server connections, jobs, logs, and "
                        + "model installs")
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
                // Three routes at Home's 420 pt floor: `ViewThatFits` keeps
                // them on one line where there is room and stacks them into
                // two rows where there is not, instead of clipping.
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 8) { agentEmptyStateButtons }
                    VStack(alignment: .leading, spacing: 6) { agentEmptyStateButtons }
                }
                .controlSize(.small)
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

    @ViewBuilder
    private var agentEmptyStateButtons: some View {
        Button("Open Agents") { navigate(.agents) }
            .help("the Agents section — library, New Agent, and optimization runs")
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
