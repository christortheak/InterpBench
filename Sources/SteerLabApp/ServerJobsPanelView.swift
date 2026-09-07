import AppKit
import ExperimentKit
import SwiftUI

struct ServerJobsPanelView: View {
    @Bindable var service: ChatService
    private struct DiagnosticTarget: Identifiable {
        let id = UUID()
        let client: ClusterClient
        let endpoint: String
    }
    private struct RecoveryTarget: Identifiable {
        let id = UUID()
        let client: ClusterClient
        let jobID: String
    }
    private struct CustodyTarget: Identifiable {
        let id = UUID()
        let root: URL
        let client: ClusterClient?
        let jobID: String?
    }
    @State private var custodyTarget: CustodyTarget?
    /// A cancel parked behind its confirmation (UI audit 2026-09-06,
    /// headline 7): cancelling a Slurm job kills a possibly hours-long run
    /// and loses its queue slot, and both cancel affordances used to fire on
    /// the click.
    private struct CancelTarget: Identifiable {
        let id: String
        let kind: String
    }
    @State private var cancelTarget: CancelTarget?
    @State private var recoveryTarget: RecoveryTarget?
    @State private var diagnosticTarget: DiagnosticTarget?
    @State private var jobs: [RemoteJobRecord] = []
    @State private var jobsOrigin: EvidenceImportOrigin?
    @State private var pipelines: [ClusterClient.PipelineRunSummary] = []
    @State private var selectedJobID: String?
    @State private var logLines: [String] = []
    @State private var status: String?
    @State private var isRefreshing = false
    @State private var isReconciling = false
    @State private var isStreaming = false
    @State private var streamTask: Task<Void, Never>?
    @State private var streamIdentity: UUID?
    /// When the rows on screen were fetched — a queued job can sit reading
    /// "queued" for a long time, and nothing used to say how old the row was.
    @State private var lastRefreshedAt: Date?
    /// How many head lines the 2,000-line cap has dropped from this stream,
    /// so the viewer can say so instead of silently losing the start.
    @State private var droppedLogLines = 0
    /// The last workspace-import report, in full, for the button's tooltip.
    /// A `@State` string rather than a row: this column's minimum height must
    /// not move while an import streams (the 2026-08-05 crash class).
    @State private var importDetail: String?
    @State private var isImporting = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                // One connection status per column: the Compute header above
                // already prints `cluster.status`, and this sub-caption
                // printed it a second time (UI audit 2026-09-06).
                Text(panelTitle)
                    .font(.headline)
                    .help(panelTitleHelp)
                Spacer()
                Button(isReconciling ? "Reconciling…" : "Reconcile jobs") {
                    reconcile()
                }
                .disabled(!hasServerClient || isRefreshing || isReconciling)
                .help("ask the server to re-read its child job records and "
                    + "finish any half-done shard merge — it reads and repairs "
                    + "bookkeeping, and never submits or cancels work")
                Button("Inputs, evidence and cleanup…") {
                    custodyTarget = CustodyTarget(root: ExperimentStore.workspaceRoot,
                        client: clientForRows(origin: jobsOrigin), jobID: selectedJobID)
                }
                .help("review this job's inputs, bring its evidence home with a "
                    + "custody receipt, or plan a policy-bound cleanup — every "
                    + "step is a separate explicit action")
                Button("Scientific diagnostic…") {
                    if let client = service.cluster.client {
                        diagnosticTarget = DiagnosticTarget(client: client, endpoint: client.profile.baseURL.absoluteString)
                    }
                }
                .disabled(!hasServerClient)
                .help("open the plan-then-submit form for a capability battery "
                    + "or an extraction-stability check on this server")
                if isRefreshing {
                    ProgressView()
                        .controlSize(.small)
                }
                Button {
                    Task { await refreshJobs(selectFirstWhenEmpty: false) }
                } label: {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
                .disabled(!hasServerClient || isRefreshing)
                .help("re-read the durable job list from the active compute "
                    + "target — the status line below stamps when")

                // The workspace import (open-issues §20). The evidence
                // auto-import beside it brings ONE run home per finished job,
                // through the API, as a bundle; this sweeps the whole remote
                // runs/ over rsync under the shared policy and rebuilds the
                // catalog. They are complementary, not alternatives.
                Button {
                    Task { await importClusterRuns() }
                } label: {
                    Label("Import runs", systemImage: "square.and.arrow.down.on.square")
                }
                .disabled(!canImportClusterRuns || isImporting)
                // The last report is appended, never substituted: replacing
                // the description meant the button stopped saying what it
                // does after the first import (UI audit 2026-09-06).
                .help(importDetail.map { Self.importHelp + "\n\nLast import:\n" + $0 }
                    ?? Self.importHelp)
            }

            // Always-present, single-line slot: the status text changes on
            // every refresh/submit/import, and a row that appears and
            // disappears changes this split-view column's minimum height
            // while the data is landing — the 2026-08-05 crash class (see
            // jobsRegion below). A constant slot never moves the layout.
            Text(status ?? " ")
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                // This slot is the ONLY place a refusal or its repair action
                // renders; selection is what lets the researcher copy one
                // (UI audit 2026-09-06). It does not change the height.
                .textSelection(.enabled)
                .help(status ?? "")

            jobsRegion
        }
        .padding(12)
        .sheet(item: $custodyTarget) { target in
            DiagnosticLifecycleSheet(root: target.root, client: target.client, initialJobID: target.jobID)
        }
        .confirmationDialog(
            cancelTarget.map { "Cancel \($0.kind) job \($0.id)?" }
                ?? "Cancel this job?",
            isPresented: cancelPresented,
            titleVisibility: .visible,
            presenting: cancelTarget
        ) { target in
            Button("Cancel job", role: .destructive) {
                let origin = jobsOrigin
                Task { await cancel(target.id, origin: origin) }
            }
            Button("Keep running", role: .cancel) {}
        } message: { _ in
            Text("The allocation on \(service.cluster.substrateLabel) is "
                + "cancelled and its queue slot is lost. Whatever the run "
                + "already wrote stays on the server, and a checkpointed job "
                + "can be resumed from its last checkpoint.")
        }
        .sheet(item: $recoveryTarget) { target in
            JobRecoverySheet(client: target.client, jobID: target.jobID)
        }
        .sheet(item: $diagnosticTarget) { target in
            ScientificExecutionSheet(client: target.client, endpoint: target.endpoint)
        }
        .task(id: service.cluster.computeTarget.rawValue) {
            await refreshJobs(selectFirstWhenEmpty: true)
        }
        .onChange(of: service.cluster.evidenceImportOrigin) { _, _ in
            streamIdentity = nil
            streamTask?.cancel()
            isStreaming = false
            jobsOrigin = nil
            jobs = []
            pipelines = []
            selectedJobID = nil
            logLines = []
            droppedLogLines = 0
            lastRefreshedAt = nil
            status = nil
            Task { await refreshJobs(selectFirstWhenEmpty: true) }
        }
        .onDisappear {
            streamIdentity = nil
            streamTask?.cancel()
        }
    }

    /// The empty-state ↔ jobs+log region, clamped to ONE constant floor.
    ///
    /// Crash 2026-08-05 20:12 (incident 28545C32): submitting a job and then
    /// clicking Compute aborted in `-[NSWindow _postWindowNeedsUpdateConstraints]`
    /// — the async jobs fetch swapped ContentUnavailableView for the
    /// jobList+logViewer stack, which RAISED this HSplitView column's
    /// minimum height mid-display-cycle, and macOS 27 beta treats a hosting
    /// view reporting new min/max sizes during `_willUpdateConstraintsForSubtree`
    /// as fatal (same family as the four 2026-08-05 morning crashes, commit
    /// 8932454). The fix is to make the reported minimum CONSTANT: every
    /// branch here has a content minimum below 280 (the panes' floors are
    /// compressible at 100), so this frame — not the arriving data — always
    /// decides the region's minimum height.
    @ViewBuilder
    private var jobsRegion: some View {
        VStack(alignment: .leading, spacing: 12) {
            if service.cluster.computeTarget == .local {
                // One framing of one fact for the Local target, instead of
                // the old stack of three (UI audit 2026-09-06).
                ContentUnavailableView(
                    "No Jobs on Local Compute",
                    systemImage: "laptopcomputer",
                    description: Text(
                        "Local (MLX) runs everything inside this app. Jobs "
                            + "appear here when the Compute selector in the "
                            + "window toolbar points at a server."))
            } else if !hasServerClient {
                ContentUnavailableView(
                    "No Active Server",
                    systemImage: "server.rack",
                    description: Text("Choose a server compute target in the window toolbar, then connect."))
            } else if jobs.isEmpty && awaitingPipelines.isEmpty && isRefreshing {
                // The first fetch used to render an empty List and an empty
                // log box, then swap to the "No Jobs" empty state — layout-safe
                // but a visible flash (UI audit 2026-09-06).
                ContentUnavailableView(
                    "Loading Jobs…",
                    systemImage: "clock.arrow.circlepath",
                    description: Text("Reading the durable job list from \(service.cluster.substrateLabel)."))
            } else if jobs.isEmpty && awaitingPipelines.isEmpty {
                ContentUnavailableView(
                    "No Jobs",
                    systemImage: "checkmark.circle",
                    description: Text("Queued model installs, remote builds, studies, and fine-tunes will appear here."))
            } else {
                jobList
                Divider()
                logViewer
            }
        }
        .frame(minHeight: 280, maxHeight: .infinity)
    }

    private var hasServerClient: Bool {
        service.cluster.computeTarget == .server && service.cluster.client != nil
    }

    private var panelTitle: String {
        service.cluster.computeTarget == .server ? "Server Jobs" : "Jobs"
    }

    private var panelTitleHelp: String {
        service.cluster.computeTarget == .server
            ? "every durable job this compute target is running or has run — "
                + "select one to read its log"
            : "jobs exist only on a server compute target; Local (MLX) runs "
                + "everything inside this app"
    }

    private var selectedJob: RemoteJobRecord? {
        guard let selectedJobID else { return nil }
        return jobs.first { $0.id == selectedJobID }
    }

    private var cancelPresented: Binding<Bool> {
        Binding(
            get: { cancelTarget != nil },
            set: { if !$0 { cancelTarget = nil } })
    }

    /// Server-side bookkeeping repair, with the re-entry guard the button
    /// needs: `isRefreshing` alone did not cover the reconcile itself.
    private func reconcile() {
        guard !isReconciling else { return }
        let origin = jobsOrigin
        guard let client = clientForRows(origin: origin) else { return }
        isReconciling = true
        Task {
            defer { isReconciling = false }
            do {
                _ = try await client.reconcileJobs()
                guard service.cluster.evidenceImportOrigin == origin else { return }
                await refreshJobs(selectFirstWhenEmpty: false)
                status = "child records reconciled and the merge pass completed"
            } catch let error as ClusterClient.ClientError {
                status = "reconcile failed: "
                    + ClusterClient.unwrappingDetail(error).description
            } catch {
                status = "reconcile failed: \(error.localizedDescription)"
            }
        }
    }

    private var jobList: some View {
        // Sharded parents render one row with per-shard chips; their shard
        // children leave the top level (grouping rule lives in ExperimentKit,
        // unit-tested — the view only lays it out).
        //
        // The pipelines-awaiting-import rows live INSIDE this List, not
        // above it: the List is the column's compressible container, so
        // parked chains arriving from a refresh change what scrolls, never
        // the column's incompressible minimum height — which both
        // 2026-08-05 crashes proved must stay small and constant.
        List(selection: $selectedJobID) {
            if hasServerClient && !awaitingPipelines.isEmpty {
                Section {
                    ForEach(awaitingPipelines) { row in
                        pipelineAwaitingRow(row)
                    }
                } header: {
                    VStack(alignment: .leading, spacing: 4) {
                        Label(
                            "Pipelines awaiting import",
                            systemImage: "shippingbox.and.arrow.backward")
                            .font(.subheadline.weight(.semibold))
                        Text(Self.pipelinesAwaitingCaption)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            ForEach(ShardedJobGrouping.topLevel(jobs)) { job in
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(job.kind)
                            .font(.subheadline.weight(.semibold))
                        Spacer()
                        // Completed run-verb study jobs can carry an evidence
                        // bundle — "imported ✓" once the auto-import ledger
                        // knows it, else the same verified import the Studies
                        // recent-jobs rows offer, right on the Compute row.
                        if ExperimentPanel.jobOffersEvidenceImport(
                            kind: job.kind, state: job.status)
                        {
                            if jobEvidenceImported(job) {
                                Label("imported ✓", systemImage: "checkmark.seal")
                                    .font(.caption)
                                    .foregroundStyle(.green)
                                    .labelStyle(.titleOnly)
                                    .help(
                                        "this job's evidence bundle is in the "
                                            + "local ledger — the run is durable "
                                            + "in this workspace")
                            } else {
                                Button("Import evidence") {
                                    let origin = jobsOrigin
                                    Task { await importEvidence(job, origin: origin) }
                                }
                                .buttonStyle(.borderless)
                                .controlSize(.small)
                                .help(
                                    "download this job's evidence bundle, verify "
                                        + "its hashes, and land it under this "
                                        + "workspace's runs/ — the status line "
                                        + "names the imported run directory")
                            }
                        }
                        // A FAILED job whose server-side packaging saved what
                        // the run produced offers retrieval of that failure
                        // record (retention 2026-07-24). Deliberately worded
                        // and coloured as diagnostics, never as results: the
                        // 2026-07-23 shakedown ended with useful data on the
                        // cluster and no affordance but SSH.
                        if ExperimentPanel.jobOffersPartialEvidenceImport(job) {
                            if jobEvidenceImported(job) {
                                Label("partial ✓", systemImage: "exclamationmark.triangle")
                                    .font(.caption)
                                    .foregroundStyle(.orange)
                                    .labelStyle(.titleOnly)
                                    .help(
                                        "this failed job's PARTIAL evidence is "
                                            + "in the local ledger — the data it "
                                            + "produced is durable in this "
                                            + "workspace, and is a failure "
                                            + "record, not a result")
                            } else {
                                Button("Retrieve partial data") {
                                    let origin = jobsOrigin
                                    Task { await importEvidence(job, origin: origin) }
                                }
                                .buttonStyle(.borderless)
                                .controlSize(.small)
                                .foregroundStyle(.orange)
                                .help(Self.partialRetrievalHelp(for: job))
                            }
                            retryEvaluateButton(for: job)
                        }
                        // A checkpointed job is RESUMABLE — offer the resume
                        // right where the state is shown (2026-07-22
                        // incident: the state rendered with no way to act).
                        if RemoteJobStatusClass.offersResume(
                            status: job.status, resubmittedAs: job.resubmittedAs)
                        {
                            Button("Resume") {
                                let origin = jobsOrigin
                                Task { await resubmit(job.id, origin: origin) }
                            }
                            .buttonStyle(.borderless)
                            .controlSize(.small)
                            .help(
                                "re-submit this job's own sbatch script — "
                                    + "the run continues from its checkpoint; "
                                    + "the status line reports the new Slurm "
                                    + "job id")
                        } else if let continuation = job.resubmittedAs,
                            RemoteJobStatusClass.classify(status: job.status)
                                == .resumable
                        {
                            Text("resumed → \(continuation)")
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                                .help(
                                    "this checkpointed job was already "
                                        + "resubmitted — the named "
                                        + "continuation record is carrying "
                                        + "the run")
                        }
                        Text(RemoteJobStatusClass.displayText(for: job.status))
                            .font(.caption)
                            .foregroundStyle(statusColor(for: job))
                    }
                    HStack(spacing: 10) {
                        Text(job.id)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .help("this workspace's job id: \(job.id)")
                        if let executorJobID = job.executorJobID, !executorJobID.isEmpty {
                            Text("scheduler \(executorJobID)")
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .help("the scheduler's own id for this job: "
                                    + executorJobID)
                        }
                    }
                    Text(jobTimeSummary(job))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    parkedRecoveryLine(for: job)
                    shardChips(for: job)
                    if let error = job.error, !error.isEmpty {
                        // Two caption lines rarely hold a server failure
                        // reason, and a Text inside a List row cannot be
                        // selected — the tooltip and the context-menu copy
                        // are how the whole thing is reachable.
                        Text(error)
                            .font(.caption2)
                            .foregroundStyle(.red)
                            .lineLimit(2)
                            .help(error)
                    }
                }
                .padding(.vertical, 5)
                .tag(job.id)
                .contextMenu {
                    Button("Copy Job ID") { copy(job.id, describedAs: "job id") }
                        .help("put \(job.id) on the clipboard")
                    if let error = job.error, !error.isEmpty {
                        Button("Copy Error") {
                            copy(error, describedAs: "failure reason")
                        }
                        .help("put this job's whole failure reason on the "
                            + "clipboard — the row shows only its first lines")
                    }
                    if RemoteJobStatusClass.offersResume(
                        status: job.status, resubmittedAs: job.resubmittedAs)
                    {
                        Button("Resume from Checkpoint") {
                            let origin = jobsOrigin
                            Task { await resubmit(job.id, origin: origin) }
                        }
                        .help("re-submit this job's own sbatch script — the "
                            + "run continues from its checkpoint")
                    }
                    if job.finishedAt == nil {
                        Button("Cancel Job", role: .destructive) {
                            cancelTarget = CancelTarget(id: job.id, kind: job.kind)
                        }
                        .help("cancel this job on the compute target — asks "
                            + "first; the queue slot is lost")
                    }
                }
            }
        }
        // SOFT minimum (2026-08-05): a hard 220 made the Compute column's
        // total minimum height cross the window's available height once the
        // section gained an extra warning row — and macOS 27 beta's
        // NavigationSplitView answers an over-tall column with an infinite
        // update-constraints loop and an NSGenericException (4 crashes that
        // morning, bisected to pure geometry: ANY ~40pt of extra fixed
        // height reproduced it, removing any similar chunk fixed it). Keep
        // pane minimums compressible so the column can always fit. 100, not
        // 120, since the evening crash the same day: both panes' floors must
        // sum under jobsRegion's constant 280 so the empty↔populated swap
        // never changes the column's reported minimum.
        .frame(minHeight: 100, idealHeight: 220)
        .onChange(of: selectedJobID) { _, newValue in
            guard let id = newValue else { return }
            startStreaming(id)
        }
    }

    /// The recovery action under a PARKED job's row (2026-08-06 review round
    /// 2). A parked job is terminal but unfinished: it stopped with durable
    /// state and something for the researcher to do, and the server's reason
    /// IS that something. Showing it in the row is what keeps the state from
    /// being a colour with no consequence — the chain's own import affordance
    /// lives in the "Pipelines awaiting import" section above, which the same
    /// park stamps.
    ///
    /// A separate `@ViewBuilder` because the row body is already at the
    /// type-checker's budget — this file has hit that ceiling before.
    @ViewBuilder
    private func parkedRecoveryLine(for job: RemoteJobRecord) -> some View {
        if RemoteJobStatusClass.classify(status: job.status) == .parked {
            let guidance = RemoteJobStatusClass.parkedGuidance(
                reason: job.parkedReason)
            Label(guidance, systemImage: "exclamationmark.triangle")
                .font(.caption2)
                .foregroundStyle(.orange)
                .lineLimit(3)
                .help(guidance)
        }
    }

    /// Per-shard chips under a sharded parent row: each chip selects that
    /// shard job, so the shared log viewer / Cancel / Resume affordances
    /// apply to the shard like any other job. Empty for ordinary jobs.
    @ViewBuilder
    private func shardChips(for parent: RemoteJobRecord) -> some View {
        let children = ShardedJobGrouping.children(of: parent, in: jobs)
        if !children.isEmpty {
            VStack(alignment: .leading, spacing: 4) {
                if let aggregate = ShardedJobGrouping.aggregateLine(children: children) {
                    Text(aggregate)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                HStack(spacing: 6) {
                    ForEach(Array(children.enumerated()), id: \.element.id) { position, child in
                        Button {
                            selectedJobID = child.id
                        } label: {
                            Text(ShardedJobGrouping.chipLabel(
                                child: child, position: position))
                                .font(.caption2.monospaced())
                                .foregroundStyle(statusColor(for: child))
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                        .help(
                            "select this shard job — its log, Cancel, and "
                                + "Resume work like any other job's "
                                + "(id \(child.id))")
                    }
                }
            }
        }
    }

    private var logViewer: some View {
        VStack(alignment: .leading, spacing: 8) {
            logHeader
            logBox
        }
    }

    private var logHeader: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Log")
                    .font(.headline)
                Text(selectedJobID ?? "Select a job")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .help(selectedJobID.map { "log of job \($0)" }
                        ?? "select a job in the list above to read its log")
            }
            Spacer()
            if isStreaming {
                ProgressView()
                    .controlSize(.small)
                // "Stop stream", never a bare "Stop": it sits beside the
                // Cancel that kills the JOB (UI audit 2026-09-06).
                Button("Stop stream") {
                    streamTask?.cancel()
                    streamTask = nil
                    isStreaming = false
                }
                .help("stop following this log — the job keeps running, and "
                    + "Stream picks the tail up again")
            }
            // Icon-only: this header already carries four controls, and the
            // column's 560 pt floor has no room for a fifth title.
            CopyButton(
                help: "copy everything in the box below, truncation marker "
                    + "included, to the clipboard",
                text: { hasLogText ? logText : nil }
            ) {
                Label("Copy log", systemImage: "doc.on.doc")
            }
            .labelStyle(.iconOnly)
            .disabled(!hasLogText)
            .accessibilityLabel("Copy log")
            if let selectedJobID {
                Button("Recovery review…") {
                    if let client = clientForRows(origin: jobsOrigin) {
                        recoveryTarget = RecoveryTarget(client: client, jobID: selectedJobID)
                    }
                }
                .help("inspect who owns this job before asserting that the "
                    + "original controller has exited")
                Button {
                    startStreaming(selectedJobID)
                } label: {
                    Label("Stream", systemImage: "waveform")
                }
                .disabled(isStreaming)
                .help("follow this job's log live — new lines append and the "
                    + "box stays at the bottom")
                Button(role: .destructive) {
                    cancelTarget = CancelTarget(
                        id: selectedJobID, kind: selectedJob?.kind ?? "server")
                } label: {
                    Label("Cancel", systemImage: "stop.fill")
                }
                .disabled(selectedJob?.finishedAt != nil)
                .help("cancel this job on the compute target — asks first; "
                    + "the queue slot is lost")
            }
        }
    }

    /// Bottom-anchored while streaming (UI audit 2026-09-06): appended lines
    /// used to fill below the fold with nothing following them.
    private var logBox: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    Text(logText)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.primary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(10)
                    Color.clear
                        .frame(height: 1)
                        .id(Self.logBottomAnchor)
                }
            }
            .background(Color(nsColor: .textBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(.quaternary, lineWidth: 1)
            )
            // Compressible for the same layout-loop reason as the job list
            // above (2026-08-05); 100 so both floors fit under jobsRegion's
            // constant 280.
            .frame(minHeight: 100, idealHeight: 220)
            .onChange(of: logLines.count) { _, _ in
                guard isStreaming else { return }
                proxy.scrollTo(Self.logBottomAnchor, anchor: .bottom)
            }
            .onChange(of: selectedJobID) { _, _ in
                proxy.scrollTo(Self.logBottomAnchor, anchor: .bottom)
            }
        }
    }

    private static let logBottomAnchor = "log-bottom"

    private var hasLogText: Bool {
        !logLines.isEmpty || !(selectedJob?.logTail.isEmpty ?? true)
    }

    private var logText: String {
        if logLines.isEmpty {
            if let selectedJob, !selectedJob.logTail.isEmpty {
                return selectedJob.logTail.joined(separator: "\n")
            }
            return "No log output yet."
        }
        let body = logLines.joined(separator: "\n")
        guard droppedLogLines > 0 else { return body }
        // The cap used to drop the head in silence, so a log could begin
        // mid-sentence with nothing saying why.
        return "… \(droppedLogLines) earlier line"
            + (droppedLogLines == 1 ? "" : "s")
            + " dropped — this viewer keeps the last 2,000 …\n" + body
    }

    private func refreshJobs(selectFirstWhenEmpty: Bool) async {
        guard hasServerClient, let client = service.cluster.client,
            let origin = service.cluster.evidenceImportOrigin else {
            jobsOrigin = nil
            jobs = []
            selectedJobID = nil
            status = "Connect to a server workspace to inspect jobs."
            return
        }
        isRefreshing = true
        defer { isRefreshing = false }
        // WS3: make sure the evidence auto-import service exists (idempotent)
        // so the "imported ✓" chips reflect the ledger and background imports
        // run while this workspace is connected.
        service.cluster.registerEvidenceAutoImport()
        do {
            let fetched = try await client.jobs()
                .sorted { $0.createdAt > $1.createdAt }
            // Cross-experiment pipeline listing (2026-08-06): dead chains
            // with completed stages surface here for one-click import. An
            // older server without the route simply lists none.
            let fetchedPipelines = (try? await client.allPipelineRuns()) ?? []
            guard service.cluster.evidenceImportOrigin == origin else { return }
            jobsOrigin = origin
            jobs = fetched
            pipelines = fetchedPipelines
            lastRefreshedAt = Date()
            if !(selectedJobID.map { id in fetched.contains { $0.id == id } } ?? false) {
                selectedJobID = selectFirstWhenEmpty ? fetched.first?.id : nil
            }
            // "as of <time>": there is no auto-poll, so a queued row can be
            // minutes old with nothing saying so (UI audit 2026-09-06).
            status = "\(fetched.count) job\(fetched.count == 1 ? "" : "s")"
                + " · as of \(Self.clock.string(from: lastRefreshedAt ?? Date()))"
        } catch {
            guard service.cluster.evidenceImportOrigin == origin else { return }
            status = "could not list jobs: \(error.localizedDescription)"
        }
    }

    // MARK: Workspace import (open-issues §20)

    private static let importHelp: String =
        "Bring this cluster's run directories into the workspace under the "
        + "shared import policy: runs, analyses, evaluations, submit receipts "
        + "(including final adapter weights) and vector artifacts come home; "
        + "evidence tarballs, training checkpoints, and merged shard partials "
        + "stay on the cluster. Verified by file count and per-file size, "
        + "idempotent (gaps are filled, nothing is overwritten), and it "
        + "DELETES nothing — it reports what cluster scratch may now drop."

    /// An SSH-transport site is the only shape a run-directory import can
    /// travel over: run directories are GB-scale and ride rsync.
    private var canImportClusterRuns: Bool {
        guard let site = service.cluster.activeSite else { return false }
        return site.isSSHTransport
    }

    /// The manual affordance. It calls the SAME `WorkspaceRunImport` engine
    /// the CLI verb does — the policy has one implementation — and streams its
    /// progress lines into the panel's constant status slot.
    private func importClusterRuns() async {
        guard let entry = service.cluster.activeServer else {
            status = "Select a cluster workspace first."
            return
        }
        isImporting = true
        defer { isImporting = false }
        status = "importing run directories from \(entry.name)…"
        let engine: WorkspaceRunImport.Engine
        do {
            engine = try await WorkspaceRunImport.liveEngine(
                site: entry.resolvedSite, siteID: entry.id.uuidString,
                workspaceRoot: ExperimentStore.workspaceRoot,
                shell: ProvisionShellRunner())
        } catch let error as WorkspaceRunImport.SetupError {
            status = "import refused: \(error.reason)"
            importDetail = error.errorDescription
            return
        } catch {
            status = "import refused: \(error.localizedDescription)"
            return
        }
        // Off the main actor: an import walks a remote tree and rsyncs GBs.
        let report = await Task.detached { await WorkspaceRunImport.run(engine: engine) }.value
        importDetail = WorkspaceRunImport.summaryLines(report).joined(separator: "\n")
        // A drifted directory's cluster copy, brought home beside it, is an
        // import too — under its `-reimport` name.
        let imported = report.imported.count + report.reimported.count
        var line = imported == 0
            ? "nothing new to import"
            : "imported \(imported) run director\(imported == 1 ? "y" : "ies")"
        if !report.violations.isEmpty {
            line += " — \(report.violations.count) violation(s); nothing was overwritten"
        } else if report.hasLoudPurgeFindings {
            line += " — shard partials without an evidenced merge (hover for detail)"
        }
        status = line
        await refreshJobs(selectFirstWhenEmpty: false)
    }

    private func clientForRows(origin: EvidenceImportOrigin?) -> ClusterClient? {
        guard let origin, service.cluster.evidenceImportOrigin == origin else {
            // The internal code used to be the visible prefix; it belongs in
            // the tooltip, not in front of the repair sentence.
            status = EvidenceImportOrigin.changedRepair
            return nil
        }
        return service.cluster.client
    }

    private func startStreaming(_ jobID: String) {
        streamTask?.cancel()
        let origin = jobsOrigin
        guard hasServerClient, let client = clientForRows(origin: origin) else { return }
        let identity = UUID()
        streamIdentity = identity
        logLines = jobs.first(where: { $0.id == jobID })?.logTail ?? []
        droppedLogLines = 0
        isStreaming = true
        streamTask = Task {
            do {
                try await client.streamJobLog(jobID: jobID) { line in
                    await MainActor.run {
                        guard streamIdentity == identity, jobsOrigin == origin, service.cluster.evidenceImportOrigin == origin,
                            selectedJobID == jobID else { return }
                        if logLines.last != line {
                            logLines.append(line)
                        }
                        if logLines.count > 2_000 {
                            let excess = logLines.count - 2_000
                            logLines.removeFirst(excess)
                            droppedLogLines += excess
                        }
                    }
                }
                guard streamIdentity == identity else { return }
                await MainActor.run {
                    isStreaming = false
                    streamTask = nil
                }
                await refreshJobs(selectFirstWhenEmpty: false)
            } catch is CancellationError {
                guard streamIdentity == identity else { return }
                await MainActor.run {
                    isStreaming = false
                    streamTask = nil
                }
            } catch {
                guard streamIdentity == identity else { return }
                await MainActor.run {
                    isStreaming = false
                    streamTask = nil
                    status = "log stream ended: \(error.localizedDescription)"
                }
            }
        }
    }

    /// Help text for the failed-job retrieval action. Built here rather
    /// than inline so the view body stays type-checkable, and so the
    /// wording — "failure record", not "results" — lives in one place.
    static func partialRetrievalHelp(for job: RemoteJobRecord) -> String {
        var text = """
            This job FAILED, but the data it produced before failing — \
            generations, judgments, raw judge responses, scheduler logs — \
            was packaged. Download and verify it into this workspace's \
            runs/.

            It imports as a FAILURE RECORD: inspect it, retry from it, \
            never cite it as a completed run.
            """
        if let summary = job.failureSummary, !summary.isEmpty {
            text += "\n\nFailure: \(summary)"
        }
        return text
    }

    /// Submit a targeted retry: finish this failed evaluation by judging
    /// only the cells it never decided.
    ///
    /// The refusals live on the SERVER, which verifies the partial run's
    /// pins before reusing a row — so a refusal here is a real scientific
    /// stop and is surfaced verbatim rather than being softened.
    /// The action the retrieval help text used to promise and not provide
    /// (external review 2026-07-24, finding 3): judge only the cells this
    /// evaluation never decided, reusing the verdicts it already produced.
    ///
    /// A separate `@ViewBuilder` because the row body is already at the
    /// type-checker's budget — this file has hit that ceiling before.
    @ViewBuilder
    private func retryEvaluateButton(for job: RemoteJobRecord) -> some View {
        if let retry = job.retryableEvaluate {
            Button("Retry missing judgments") {
                let origin = jobsOrigin
                Task { await retryEvaluate(job, retry, origin: origin) }
            }
            .buttonStyle(.borderless)
            .controlSize(.small)
            .help(Self.retryHelp(experiment: retry.experiment))
        }
    }

    static func retryHelp(experiment: String) -> String {
        """
        Re-run '\(experiment)' evaluate, judging ONLY the pairs this run \
        never decided and reusing the verdicts it already produced.

        The server verifies every pin of the partial first — a changed \
        rubric, epoch, source run, or judge configuration refuses rather \
        than mixing two evaluations into one table.
        """
    }

    private func retryEvaluate(
        _ job: RemoteJobRecord,
        _ retry: (experiment: String, partialRunID: String), origin: EvidenceImportOrigin?
    ) async {
        status = "retrying missing judgments for \(retry.experiment)…"
        guard let client = clientForRows(origin: origin) else { return }
        do {
            // Route by how the ORIGINAL job ran. A Slurm evaluate retries
            // through study submission — the same path the original used,
            // so it lands on a GPU allocation rather than in the
            // controller's process.
            //
            // Note what this does NOT do: judge fan-out is wired for the
            // `pipeline` verb only (`_check_local_judge_deliverability`
            // returns early for standalone evaluate), so a multi-model
            // panel here judges sequentially in one job rather than one
            // worker per judge model. Each model is loaded once for its
            // whole column (2026-07-24), so that is slow, not quadratic —
            // but it is not the fan-out, and this comment used to say it
            // was.
            let jobID: String
            if job.executor == "slurm" {
                let submission = try await client.submitStudy(
                    experiment: retry.experiment, verb: "evaluate",
                    executor: "slurm", resumeFrom: retry.partialRunID)
                jobID = submission.jobId
            } else {
                let submission = try await client.submitExperimentJobDetailed(
                    experiment: retry.experiment, verb: "evaluate",
                    resumeFrom: retry.partialRunID)
                jobID = submission.jobId
            }
            guard service.cluster.evidenceImportOrigin == origin else { return }
            status = "retry submitted as job \(jobID) — it reuses the "
                + "verdicts \(retry.partialRunID) already produced"
            await refreshJobs(selectFirstWhenEmpty: false)
        } catch {
            status = "retry refused: \(error.localizedDescription)"
        }
    }

    /// Server pipelines with completed stage runs whose evidence is not in
    /// this workspace yet — parked (dead, stamped by the daemon's startup
    /// reconcile) or terminal. Triage rule lives in ExperimentKit,
    /// unit-tested; the view only lays it out.
    private var awaitingPipelines: [ClusterClient.PipelineRunSummary] {
        guard let origin = jobsOrigin, service.cluster.evidenceImportOrigin == origin else { return [] }
        return PipelineImportTriage.awaitingImport(
            pipelines,
            importedRunIDs:
                service.cluster.evidenceAutoImport?.importedRunIDs(origin: origin) ?? [])
    }

    private static let pipelinesAwaitingCaption: String =
        "Chains with completed stage runs whose evidence is not in "
        + "this workspace. A parked chain was orphaned by a "
        + "server restart — import brings its finished stages "
        + "home; resubmit the pipeline to run what remains."

    private static let pipelineImportHelp: String =
        "package this chain's evidence on the server (every "
        + "completed stage run), download it, verify its "
        + "hashes, and land it under this workspace's runs/ — "
        + "including the model-revision reconciliation analyze needs"

    /// A separate builder with precomputed strings — this file has hit the
    /// type-checker's budget before.
    private func pipelineAwaitingRow(
        _ row: ClusterClient.PipelineRunSummary
    ) -> some View {
        let title: String = (row.experiment ?? "?") + " · " + row.run
        let summary: String = row.stateLabel + " — " + row.stageSummaryLine
        return HStack(alignment: .firstTextBaseline) {
            // Single-line texts: a long experiment name or park reason
            // truncates (full text in .help) rather than growing the row.
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.caption.monospaced())
                    .lineLimit(1)
                Text(summary)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .help(summary)
                if let reason = row.parked?.reason {
                    Text(reason)
                        .font(.caption2)
                        .foregroundStyle(.orange)
                        .lineLimit(1)
                        .help(reason)
                }
            }
            Spacer()
            Button("Import evidence") {
                let origin = jobsOrigin
                Task { await importPipelineEvidence(row, origin: origin) }
            }
            .buttonStyle(.borderless)
            .controlSize(.small)
            .help(Self.pipelineImportHelp)
        }
    }

    /// One-click import of a dead/parked chain's finished stages: the
    /// server packages from the ledger, then the same verified auto-import
    /// path lands it (hash check, importer, revision adoption).
    private func importPipelineEvidence(
        _ row: ClusterClient.PipelineRunSummary, origin: EvidenceImportOrigin?
    ) async {
        guard let origin, clientForRows(origin: origin) != nil else { return }
        status = "packaging evidence for pipeline \(row.run) on the server…"
        let importer = service.cluster.registerEvidenceAutoImport()
        if let event = await importer.importPipeline(runID: row.run, origin: origin) {
            guard service.cluster.evidenceImportOrigin == origin else { return }
            switch event.outcome {
            case .imported(let runDirectory):
                status = "pipeline \(row.run) evidence imported → "
                    + "runs/\(URL(filePath: runDirectory).lastPathComponent)"
                    + " (hashes verified)"
            case .skippedUnbundleable(let note):
                status = "pipeline \(row.run) skipped — \(note)"
            case .refused(let code, let repair):
                status = "\(code): \(repair)"
            case .failed(let message):
                status = "pipeline evidence import failed: \(message)"
            }
        } else {
            status = importer.lastSummary
                ?? "pipeline evidence import did not start"
        }
        await refreshJobs(selectFirstWhenEmpty: false)
    }

    /// Whether this job's evidence bundle is already in the local
    /// origin-scoped ledger, with a matching bundle version.
    private func jobEvidenceImported(_ job: RemoteJobRecord) -> Bool {
        guard let origin = jobsOrigin, service.cluster.evidenceImportOrigin == origin,
            let importer = service.cluster.evidenceAutoImport,
            let candidate = EvidenceAutoImportService.candidate(fromJob: job)
        else { return false }
        return importer.isImported(candidate: candidate, origin: origin)
    }

    /// Import through the auto-import service when it can see the bundle
    /// (records the ledger entry, so the chip flips to "imported ✓"). A row
    /// without a bundle needs packaging on its originating server first.
    private func importEvidence(_ job: RemoteJobRecord, origin: EvidenceImportOrigin?) async {
        guard let origin, clientForRows(origin: origin) != nil else { return }
        status = "importing evidence from job \(job.id)…"
        let importer = service.cluster.registerEvidenceAutoImport()
        if let event = await importer.importNow(job: job, origin: origin) {
            guard service.cluster.evidenceImportOrigin == origin else { return }
            switch event.outcome {
            case .imported(let runDirectory):
                status = "evidence from job \(job.id) imported → "
                    + "runs/\(URL(filePath: runDirectory).lastPathComponent) (hashes verified)"
            case .skippedUnbundleable(let note):
                status = "run \(event.runId ?? "?") skipped — \(note)"
            case .refused(let code, let repair):
                status = "\(code): \(repair)"
            case .failed(let message):
                status = "evidence import failed: \(message)"
            }
            return
        }
        // No bundle in this captured row: never substitute another panel's
        // selected job inventory or current connection.
        status = importer.lastSummary ?? "This job has no packaged evidence. Refresh its originating server's job list or package the run before importing."
    }

    /// Manual resume of a checkpointed job: the server re-sbatches the
    /// job's own run.sbatch (the same implementation auto-resume uses) and
    /// the run continues from its checkpoint. Refusal details (already
    /// resubmitted / still running / cancelled) surface verbatim.
    private func resubmit(_ jobID: String, origin: EvidenceImportOrigin?) async {
        guard let client = clientForRows(origin: origin) else { return }
        do {
            let result = try await client.resubmitJob(jobID)
            guard service.cluster.evidenceImportOrigin == origin else { return }
            status = RemoteJobStatusClass.resumedStatusLine(
                jobID: jobID, slurmJobID: result.slurmJobID,
                continuationJobID: result.jobId)
            await refreshJobs(selectFirstWhenEmpty: false)
        } catch let error as ClusterClient.ClientError {
            status = "resume failed: \(ClusterClient.unwrappingDetail(error).description)"
        } catch {
            status = "resume failed: \(error.localizedDescription)"
        }
    }

    private func cancel(_ jobID: String, origin: EvidenceImportOrigin?) async {
        guard let client = clientForRows(origin: origin) else { return }
        do {
            try await client.cancelJob(jobID)
            guard service.cluster.evidenceImportOrigin == origin else { return }
            status = "cancel requested for \(jobID)"
            await refreshJobs(selectFirstWhenEmpty: false)
        } catch let error as ClusterClient.ClientError {
            // A 502 here means scancel itself failed — the allocation may
            // still be running; the server's detail says so and names the
            // job. Show its words, not a JSON blob.
            status = "cancel failed: \(ClusterClient.unwrappingDetail(error).description)"
        } catch {
            status = "cancel failed: \(error.localizedDescription)"
        }
    }

    private func jobTimeSummary(_ job: RemoteJobRecord) -> String {
        var parts = ["created \(formatTimestamp(job.createdAt))", job.executor]
        if let startedAt = job.startedAt {
            parts.append("started \(formatTimestamp(startedAt))")
        }
        if let finishedAt = job.finishedAt {
            parts.append("finished \(formatTimestamp(finishedAt))")
        }
        // WS2 child-record enrichments (walltime used, records written so
        // far) — the honest "is it moving?" numbers for resumable jobs.
        if let elapsed = job.resolvedElapsedSeconds, elapsed > 0 {
            parts.append("elapsed \(formatElapsed(elapsed))")
        }
        if let records = job.resolvedRecordCount, records > 0 {
            parts.append("\(records) record\(records == 1 ? "" : "s")")
        }
        if job.cancellationRequested {
            parts.append("cancel requested")
        }
        return parts.joined(separator: " • ")
    }

    private func formatElapsed(_ seconds: Double) -> String {
        let total = Int(seconds.rounded())
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        if hours > 0 { return "\(hours)h \(String(format: "%02d", minutes))m" }
        if minutes > 0 { return "\(minutes)m \(String(format: "%02d", total % 60))s" }
        return "\(total)s"
    }

    /// Cached: this used to allocate a `DateFormatter` per call, several per
    /// row per render.
    private static let stamp: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .medium
        return formatter
    }()

    /// Time only — the "as of" stamp on the status line is always today.
    private static let clock: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .none
        formatter.timeStyle = .medium
        return formatter
    }()

    private func formatTimestamp(_ timestamp: Double) -> String {
        Self.stamp.string(from: Date(timeIntervalSince1970: timestamp))
    }

    /// Colors come from the shared, unit-tested classifier: "checkpointed"
    /// is amber (resumable, NON-terminal — never failed), "parked" is amber
    /// too (terminal, but it needs the researcher — never green, never red),
    /// and an unknown status stays neutral. "cancelling" keeps its historical
    /// amber (transitional), distinct from plain in-flight blue.
    private func statusColor(for job: RemoteJobRecord) -> Color {
        if job.status.lowercased().contains("cancelling") { return .orange }
        switch RemoteJobStatusClass.classify(status: job.status, finishedAt: job.finishedAt) {
        case .resumable: return .orange
        case .parked: return .orange
        case .inFlight: return .blue
        case .failed: return .red
        case .succeeded: return .green
        case .neutral: return .secondary
        }
    }

    /// A context-menu copy cannot show "Copied" the way `CopyButton` does —
    /// the menu is gone by then — so it reports through the always-present
    /// status slot instead (UI audit 2026-09-06, headline 19).
    private func copy(_ value: String, describedAs what: String) {
        status = Clipboard.copy(value)
            ? "copied the \(what) to the clipboard"
            : "could not write to the clipboard — another app is holding it"
    }
}
