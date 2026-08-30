import ExperimentKit
import SwiftUI

/// WS3 Cluster Health card (Home dashboard): one glanceable section that
/// makes "you never need to ssh in" checkable — connection, storage bars,
/// purge risk, HF cache, next maintenance window, evidence pending, and the
/// housekeeping scan's freshness. Rendering only: every threshold, verdict,
/// and countdown comes from ExperimentKit (`StorageSeverity`,
/// `HousekeepingPurgeRisk.isCritical`, `MaintenanceWindow.countdown…`,
/// `EvidenceAutoImportService`), where it is unit-tested.
struct ClusterHealthCard: View {
    @Bindable var service: ChatService

    @State private var model = ClusterHealthModel()
    @State private var showingMaintenanceEditor = false

    private var cluster: ClusterConnectionStore { service.cluster }

    var body: some View {
        Section("Cluster health — \(cluster.substrateLabel)") {
            connectionRow
            if cluster.capabilities == nil {
                Text("connect to see housekeeping (toolbar connection dot)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if cluster.capabilities?.supportsHousekeeping != true {
                // Capability-gated graceful degradation (WS6.4): one quiet
                // line, not a mystery failure.
                Text("server predates housekeeping — update steerlab-server for "
                    + "quota, purge, and maintenance visibility")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                housekeepingRows
            }
        }
        .task(id: healthTaskKey) {
            // Auto-import needs only the jobs API — register it even when
            // the server predates housekeeping (idempotent).
            cluster.registerEvidenceAutoImport()
            await model.refresh(cluster: cluster)
        }
        .sheet(isPresented: $showingMaintenanceEditor) {
            MaintenanceWindowsEditor(
                cluster: cluster,
                initialWindows: model.status?.maintenance?.windowList ?? [],
                onSaved: { canonical in
                    model.noteMaintenanceWindows(canonical)
                })
        }
    }

    /// Re-fetch when the workspace/connection identity changes.
    private var healthTaskKey: String {
        let connected = cluster.capabilities != nil
        return "\(cluster.substrateLabel)|\(connected)|\(cluster.capabilities?.supportsHousekeeping == true)"
    }

    // MARK: Connection

    @ViewBuilder
    private var connectionRow: some View {
        let line = connectionLine
        LabeledContent("Connection") {
            Label(line.text, systemImage: "circle.fill")
                .font(.caption)
                .foregroundStyle(line.color)
        }
    }

    private var connectionLine: (text: String, color: Color) {
        if cluster.activeSite?.isSSHTransport == true,
            let state = cluster.attachedTunnel?.state
        {
            switch state {
            case .up:
                return (state.displayDescription, .green)
            case .needsAuth, .opening:
                return (state.displayDescription, .orange)
            case .degraded:
                return (state.displayDescription, .red)
            case .idle, .closed:
                return (state.displayDescription, .secondary)
            }
        }
        let status = cluster.status ?? "not connected"
        if status.contains("failed") || status.contains("invalid") { return (status, .red) }
        if status.hasSuffix("...") { return (status, .orange) }
        return (status, cluster.capabilities == nil ? .secondary : .green)
    }

    // MARK: Housekeeping rows

    @ViewBuilder
    private var housekeepingRows: some View {
        if let error = model.errorLine {
            Label(error, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .textSelection(.enabled)
        }
        if let status = model.status {
            storageRows(status)
            purgeRow(status)
            cacheRow(status)
            throughputRows(status)
            maintenanceRow(status)
            evidenceRows(status)
            freshnessRow(status)
        } else if model.isLoading {
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini)
                Text("fetching housekeeping status…")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: Storage

    @ViewBuilder
    private func storageRows(_ status: RemoteHousekeepingStatus) -> some View {
        let roles = status.rootMap.keys.sorted()
        ForEach(roles, id: \.self) { role in
            if let root = status.rootMap[role] {
                storageRow(role: role, root: root)
            }
        }
        // df-level numbers cover the WHOLE shared filesystem; on a cluster
        // your quota is far smaller and bites first. Say so instead of
        // letting a 2 PB bar imply headroom — and, since WP5 Step 11, show
        // the site's own quota command's output when it declared one. The
        // engine never parses that text, so neither does this view: it is
        // displayed verbatim beside the bars it corrects (audit c46).
        if status.rootMap.values.contains(where: { $0.scope == "filesystem" }) {
            Text(
                status.quota?.hasContent == true
                    ? "Bars show whole-filesystem usage. Your allocation's quota, "
                        + "from this site's own command, is below."
                    : "Bars show whole-filesystem usage, not your allocation's quota."
            )
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        if let quota = status.quota, quota.hasContent {
            quotaRow(quota)
        }
    }

    @ViewBuilder
    private func quotaRow(_ quota: HousekeepingQuota) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            if let output = quota.output, !output.isEmpty {
                Text(output)
                    .font(.caption2.monospaced())
                    .textSelection(.enabled)
            }
            if let error = quota.error, !error.isEmpty {
                Text(error)
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
        }
        .help(quota.command.map { "site quota command: \($0)" } ?? "site quota command")
    }

    @ViewBuilder
    private func storageRow(role: String, root: HousekeepingRoot) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 8) {
                Text(role)
                    .font(.caption.weight(.medium))
                Spacer()
                Text(storageSummary(root))
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
            }
            ProgressView(value: min(max(root.usedFraction ?? 0, 0), 1))
                .tint(severityColor(root.severity))
                .controlSize(.small)
            if let warning = root.warning, !warning.isEmpty {
                // The server's own words, verbatim.
                Text(warning)
                    .font(.caption2)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
        }
        .help(root.path ?? role)
    }

    private func storageSummary(_ root: HousekeepingRoot) -> String {
        let used = HousekeepingFormat.gigabytes(root.usedBytes)
        let total = HousekeepingFormat.gigabytes(root.totalBytes)
        if let fraction = root.usedFraction {
            return "\(used) / \(total) (\(Int((fraction * 100).rounded()))%)"
        }
        return "\(used) / \(total)"
    }

    private func severityColor(_ severity: StorageSeverity) -> Color {
        switch severity {
        case .ok: .green
        case .amber: .orange
        case .red: .red
        }
    }

    // MARK: Purge risk

    @ViewBuilder
    private func purgeRow(_ status: RemoteHousekeepingStatus) -> some View {
        if let risk = status.purgeRisk {
            let critical = risk.isCritical
            LabeledContent("Purge risk") {
                Text(purgeSummary(risk))
                    .font(.caption)
                    .foregroundStyle(critical ? Color.red : ((risk.fileCount ?? 0) > 0 ? .orange : .secondary))
            }
            .help(purgeHelp(risk))
        } else {
            LabeledContent("Purge risk") {
                Text("not scanned")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func purgeSummary(_ risk: HousekeepingPurgeRisk) -> String {
        let count = risk.fileCount ?? 0
        guard count > 0 else { return "none" }
        var line = "\(count) file\(count == 1 ? "" : "s"), "
            + HousekeepingFormat.gigabytes(risk.totalBytes)
        if let worst = risk.worstOffenders.compactMap(\.ageDays).max() {
            line += ", oldest \(Int(worst))d"
        }
        if let threshold = risk.thresholdDays {
            // A purge window is one institution's policy. When the server says
            // it fell back to its own default, say so rather than presenting
            // a stranger's number as this site's (WP5 Step 11).
            line += risk.usesDefaultPolicy
                ? " (assumed purge at \(threshold)d — site policy undeclared)"
                : " (purge at \(threshold)d)"
        }
        return line
    }

    private func purgeHelp(_ risk: HousekeepingPurgeRisk) -> String {
        guard !risk.worstOffenders.isEmpty else {
            return "files whose access age approaches the site's purge window"
        }
        return risk.worstOffenders.prefix(5)
            .map { "\($0.path ?? "?") — \(Int($0.ageDays ?? 0))d, \(HousekeepingFormat.gigabytes($0.bytes))" }
            .joined(separator: "\n")
    }

    // MARK: HF cache

    @ViewBuilder
    private func cacheRow(_ status: RemoteHousekeepingStatus) -> some View {
        if let cache = status.hfCache {
            LabeledContent("HF cache") {
                Text("\(cache.modelList.count) model\(cache.modelList.count == 1 ? "" : "s"), "
                    + HousekeepingFormat.gigabytes(cache.totalBytes))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .help(cache.root ?? "server model cache")
        }
    }

    // MARK: Throughput

    /// The server folds each finished job into a global rate AND a
    /// per-instrument-family rate for the same (model, GPU) (2026-08-20), so
    /// several rows per pair are expected — each is labeled with its scope
    /// (`familyLabel`; "all families" is the global row) rather than looking
    /// like duplicates. Absent section (older server, nothing observed yet)
    /// renders nothing.
    @ViewBuilder
    private func throughputRows(_ status: RemoteHousekeepingStatus) -> some View {
        let entries = status.throughput?.entryList ?? []
        if !entries.isEmpty {
            VStack(alignment: .leading, spacing: 2) {
                Text("Throughput")
                    .font(.caption.weight(.medium))
                ForEach(Array(entries.enumerated()), id: \.offset) { _, entry in
                    Text(throughputSummary(entry))
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
            .help("observed records/hour per (model, GPU) — the walltime "
                + "estimator's feed; one pair lists a row per instrument "
                + "family plus the all-families average, and a token-bounded "
                + "rate is quoted at its recorded token basis")
        }
    }

    private func throughputSummary(_ entry: HousekeepingThroughputEntry) -> String {
        var rate = entry.recordsPerHour.map { "\(Int($0.rounded()))/h" } ?? "?/h"
        // A token-bounded rate is normalized to its entry's basis server-side,
        // so "600/h" alone misreads by whatever ratio the next submission's
        // maxTokens differs; say the budget the number is quoted at.
        if let basis = entry.tokensBasis {
            rate += " @ \(Int(basis.rounded())) tok"
        }
        var line = "\(entry.modelID ?? "?") · \(entry.gpuType ?? "?") — \(rate) · \(entry.familyLabel)"
        if let samples = entry.samples {
            line += " (n=\(samples))"
        }
        return line
    }

    // MARK: Maintenance

    @ViewBuilder
    private func maintenanceRow(_ status: RemoteHousekeepingStatus) -> some View {
        HStack(alignment: .firstTextBaseline) {
            LabeledContent("Maintenance") {
                Text(maintenanceSummary(status.maintenance))
                    .font(.caption)
                    .foregroundStyle(status.maintenance?.next == nil ? .secondary : .primary)
            }
            Button("Edit…") { showingMaintenanceEditor = true }
                .controlSize(.mini)
                .help("edit the server's maintenance windows — submissions that "
                    + "would cross a window are refused by the executor")
        }
        if status.maintenance?.stale == true {
            Text("maintenance calendar is stale — confirm the windows against the "
                + "site's announcements")
                .font(.caption2)
                .foregroundStyle(.orange)
        }
    }

    private func maintenanceSummary(_ maintenance: HousekeepingMaintenance?) -> String {
        guard let maintenance else { return "no calendar" }
        guard let next = maintenance.next else {
            let count = maintenance.windowList.count
            return count == 0 ? "no windows declared" : "no upcoming window (\(count) past)"
        }
        var line = next.label?.isEmpty == false ? next.label ?? "" : "next window"
        if let countdown = next.countdownDescription() {
            line += " — \(countdown)"
        }
        if let start = next.startDate {
            line += " (\(start.formatted(date: .abbreviated, time: .shortened)))"
        }
        return line
    }

    // MARK: Evidence

    @ViewBuilder
    private func evidenceRows(_ status: RemoteHousekeepingStatus) -> some View {
        let importer = cluster.evidenceAutoImport
        let pending = importer?.pendingCandidates(
            fromHousekeepingBundles: status.evidence?.bundleList ?? [])
            ?? []
        HStack(alignment: .firstTextBaseline) {
            LabeledContent("Evidence pending") {
                Text(pending.isEmpty
                    ? "none — results are home"
                    : "\(pending.count) bundle\(pending.count == 1 ? "" : "s") on the server")
                    .font(.caption)
                    .foregroundStyle(pending.isEmpty ? Color.secondary : Color.orange)
            }
            if !pending.isEmpty {
                Button(importer?.isImporting == true ? "Importing…" : "Import now") {
                    Task {
                        let importer = cluster.registerEvidenceAutoImport()
                        await importer.importNow(candidates: pending)
                        await model.refresh(cluster: cluster, force: false)
                    }
                }
                .controlSize(.mini)
                .disabled(importer?.isImporting == true)
                .help("download each bundle, verify its hash manifest, and land "
                    + "it under this workspace's runs/")
            }
        }
        if let entry = cluster.activeServer {
            Toggle(
                "Auto-import evidence from this site",
                isOn: Binding(
                    get: { cluster.autoImportEnabled(for: entry) },
                    set: { cluster.setAutoImportEnabled($0, for: entry) }))
                .font(.caption)
                .controlSize(.mini)
                .help("when on, succeeded run jobs' evidence bundles are "
                    + "hash-verified and imported into this workspace "
                    + "automatically (default: on for SSH sites, off for "
                    + "direct/localhost servers)")
        }
        if let summary = importer?.lastSummary {
            Text(summary)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
        if let importer, !importer.failures.isEmpty {
            ForEach(importer.failures.keys.sorted(), id: \.self) { path in
                if let failure = importer.failures[path] {
                    Label(
                        "import failed (\(failure.attempts)×"
                            + (failure.exhausted ? ", retries exhausted" : "")
                            + "): \(URL(filePath: path).lastPathComponent) — \(failure.message)",
                        systemImage: "exclamationmark.triangle")
                        .font(.caption2)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                        .lineLimit(3)
                }
            }
        }
    }

    // MARK: Freshness

    @ViewBuilder
    private func freshnessRow(_ status: RemoteHousekeepingStatus) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            LabeledContent("Scanned") {
                Text(freshnessText(status))
                    .font(.caption2)
                    .foregroundStyle(status.isStale() ? .orange : .secondary)
            }
            Button(model.isForcing ? "Refreshing…" : "Refresh") {
                Task { await model.forceRefresh(cluster: cluster) }
            }
            .controlSize(.mini)
            .disabled(model.isForcing || !hasToken)
            .help(hasToken
                ? "force a server-side rescan (privileged route)"
                : "forcing a rescan is a privileged route — save a bearer token "
                    + "for this server first")
        }
    }

    private func freshnessText(_ status: RemoteHousekeepingStatus) -> String {
        guard let date = status.generatedDate else { return "unknown" }
        let text = date.formatted(.relative(presentation: .named))
        return status.isStale() ? "\(text) — stale (tick not running?)" : text
    }

    private var hasToken: Bool {
        if !cluster.token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return true
        }
        guard let entry = cluster.activeServer else { return false }
        return cluster.storedToken(for: entry)?.isEmpty == false
    }
}

/// Fetch-and-hold state for the health card. Interpretation stays in
/// ExperimentKit; this only talks to the client and stores the answer.
@Observable @MainActor
final class ClusterHealthModel {
    private(set) var status: RemoteHousekeepingStatus?
    private(set) var errorLine: String?
    private(set) var isLoading = false
    private(set) var isForcing = false

    func refresh(cluster: ClusterConnectionStore, force: Bool = false) async {
        guard cluster.capabilities?.supportsHousekeeping == true,
            let client = cluster.client
        else {
            status = nil
            return
        }
        // Make sure the auto-import service exists so pending/imported state
        // renders (registration is idempotent and workspace-scoped).
        cluster.registerEvidenceAutoImport()
        isLoading = true
        defer { isLoading = false }
        do {
            status = try await client.housekeepingStatus()
            errorLine = nil
        } catch {
            errorLine = "could not fetch housekeeping status: \(error.localizedDescription)"
        }
    }

    func forceRefresh(cluster: ClusterConnectionStore) async {
        guard let client = cluster.client else { return }
        isForcing = true
        defer { isForcing = false }
        do {
            if let refreshed = try await client.refreshHousekeeping() {
                status = refreshed
                errorLine = nil
            } else {
                await refresh(cluster: cluster)
            }
        } catch {
            // Privileged route — auth refusals surface verbatim.
            errorLine = "refresh refused: \(error.localizedDescription)"
        }
    }

    /// The editor saved — reflect the server's canonical list immediately.
    func noteMaintenanceWindows(_ windows: [MaintenanceWindow]) {
        guard var current = status else { return }
        var maintenance = current.maintenance ?? HousekeepingMaintenance()
        maintenance.windows = windows
        maintenance.next = windows
            .filter { ($0.endDate ?? .distantPast) > Date() }
            .sorted { ($0.startDate ?? .distantFuture) < ($1.startDate ?? .distantFuture) }
            .first
        current.maintenance = maintenance
        status = current
    }
}
