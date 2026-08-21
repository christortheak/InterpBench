import ExperimentKit
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
            nextActionsSection
        }
        .formStyle(.grouped)
        .onAppear {
            service.fineTuning.refresh()
            service.experiments.refresh()
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
                LabeledContent("Connection", value: "in-process MLX")
            }
            Button("Open Compute") { navigate(.compute) }
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
                Text(
                    isServerWorkspace
                        ? "No model is loaded on \(service.cluster.serverHostLabel)."
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
                HStack(spacing: 8) {
                    Button("Open Agents") { navigate(.agents) }
                    Button("Optimize") { openOptimizations() }
                    Button("Train Adapter") { navigate(.data) }
                }
                .controlSize(.small)
            } else {
                ForEach(recentAgents) { record in
                    agentRow(record)
                }
                Button("Open Agent Library") { navigate(.agents) }
                    .controlSize(.small)
            }
        }
    }

    private func agentRow(_ record: ModelVariantRecord) -> some View {
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
        if service.experiments.isRunning || service.experiments.isValidating
            || service.experiments.isEvaluating
        {
            items.append(.init(id: "study", label: "study task", systemImage: "checkmark.seal"))
        }
        if service.multiAgent.isRunning {
            items.append(
                .init(id: "scenario", label: "multi-agent run", systemImage: "person.3.sequence"))
        }
        if let job = service.experiments.activeServerJob {
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
            Button("Open Compute") { navigate(.compute) }
                .controlSize(.small)
        }
    }

    // MARK: Studies

    private var recentStudies: [ExperimentManifest] {
        Array(service.experiments.experiments.prefix(4))
    }

    private var studiesSection: some View {
        Section("Recent studies") {
            if recentStudies.isEmpty {
                Text("No study protocols in this workspace yet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button("Create study") { navigate(.studies) }
                    .controlSize(.small)
            } else {
                ForEach(recentStudies, id: \.name) { manifest in
                    studyRow(manifest)
                }
                Button("Open Studies") { navigate(.studies) }
                    .controlSize(.small)
            }
        }
    }

    /// Context-carrying: a study row IS a link to that study — it opens
    /// Studies with the study selected, never a bare navigate.
    private func studyRow(_ manifest: ExperimentManifest) -> some View {
        Button {
            service.experiments.selectedName = manifest.name
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
        let display = service.experiments.displayName(manifest)
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

    // MARK: Next actions

    private var nextActionsSection: some View {
        Section("Next actions") {
            HStack(spacing: 8) {
                Button("Open Playground") { navigate(.playground) }
                Button("Optimize") { openOptimizations() }
                    .help("Agents → Optimizations: declared sweep runs and Create Agent")
                Button("Create study") { navigate(.studies) }
                Button("Open Compute") { navigate(.compute) }
            }
            .controlSize(.small)
        }
    }

    private func shortDate(_ iso: String) -> String {
        let trimmed = iso.replacingOccurrences(of: "T", with: " ")
            .replacingOccurrences(of: "Z", with: "")
        return String(trimmed.prefix(min(16, trimmed.count)))
    }
}
