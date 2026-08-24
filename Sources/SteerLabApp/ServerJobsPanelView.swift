import AppKit
import ExperimentKit
import SwiftUI

struct ServerJobsPanelView: View {
    @Bindable var service: ChatService
    @State private var jobs: [RemoteJobRecord] = []
    @State private var pipelines: [ClusterClient.PipelineRunSummary] = []
    @State private var selectedJobID: String?
    @State private var logLines: [String] = []
    @State private var status: String?
    @State private var isRefreshing = false
    @State private var isStreaming = false
    @State private var streamTask: Task<Void, Never>?
    /// The last workspace-import report, in full, for the button's tooltip.
    /// A `@State` string rather than a row: this column's minimum height must
    /// not move while an import streams (the 2026-08-05 crash class).
    @State private var importDetail: String?
    @State private var isImporting = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Server Jobs")
                        .font(.headline)
                    Text(connectionSummary)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
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
                .help("Refresh the durable job list from the active server.")

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
                .help(importDetail ?? Self.importHelp)
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
                .help(status ?? "")

            jobsRegion
        }
        .padding(12)
        .task(id: service.cluster.computeTarget.rawValue) {
            await refreshJobs(selectFirstWhenEmpty: true)
        }
        .onDisappear {
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
            if !hasServerClient {
                ContentUnavailableView(
                    "No Active Server",
                    systemImage: "server.rack",
                    description: Text("Choose a server workspace in the toolbar, then connect."))
            } else if jobs.isEmpty && awaitingPipelines.isEmpty && !isRefreshing {
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

    private var connectionSummary: String {
        guard service.cluster.computeTarget == .server else {
            return "Local workspace selected"
        }
        return service.cluster.status ?? service.cluster.serverURL
    }

    private var selectedJob: RemoteJobRecord? {
        guard let selectedJobID else { return nil }
        return jobs.first { $0.id == selectedJobID }
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
                                    Task { await importEvidence(job) }
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
                                    Task { await importEvidence(job) }
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
                                Task { await resubmit(job.id) }
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
                        if let executorJobID = job.executorJobID, !executorJobID.isEmpty {
                            Text("scheduler \(executorJobID)")
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                    Text(jobTimeSummary(job))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    parkedRecoveryLine(for: job)
                    shardChips(for: job)
                    if let error = job.error, !error.isEmpty {
                        Text(error)
                            .font(.caption2)
                            .foregroundStyle(.red)
                            .lineLimit(2)
                    }
                }
                .padding(.vertical, 5)
                .tag(job.id)
                .contextMenu {
                    Button("Copy Job ID") { copyToClipboard(job.id) }
                    if RemoteJobStatusClass.offersResume(
                        status: job.status, resubmittedAs: job.resubmittedAs)
                    {
                        Button("Resume from Checkpoint") {
                            Task { await resubmit(job.id) }
                        }
                    }
                    if job.finishedAt == nil {
                        Button("Cancel Job", role: .destructive) {
                            Task { await cancel(job.id) }
                        }
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
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Log")
                        .font(.headline)
                    Text(selectedJobID ?? "Select a job")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
                if isStreaming {
                    ProgressView()
                        .controlSize(.small)
                    Button("Stop") {
                        streamTask?.cancel()
                        streamTask = nil
                        isStreaming = false
                    }
                }
                if let selectedJobID {
                    Button {
                        startStreaming(selectedJobID)
                    } label: {
                        Label("Stream", systemImage: "waveform")
                    }
                    .disabled(isStreaming)
                    .help("Stream live log output for the selected job.")
                    Button(role: .destructive) {
                        Task { await cancel(selectedJobID) }
                    } label: {
                        Label("Cancel", systemImage: "stop.fill")
                    }
                    .disabled(selectedJob?.finishedAt != nil)
                }
            }

            ScrollView {
                Text(logText)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(10)
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
        }
    }

    private var logText: String {
        if logLines.isEmpty {
            if let selectedJob, !selectedJob.logTail.isEmpty {
                return selectedJob.logTail.joined(separator: "\n")
            }
            return "No log output yet."
        }
        return logLines.joined(separator: "\n")
    }

    private func refreshJobs(selectFirstWhenEmpty: Bool) async {
        guard hasServerClient, let client = service.cluster.client else {
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
            jobs = fetched
            // Cross-experiment pipeline listing (2026-08-06): dead chains
            // with completed stages surface here for one-click import. An
            // older server without the route simply lists none.
            pipelines = (try? await client.allPipelineRuns()) ?? []
            await service.cluster.refreshRemoteState()
            if let selectedJobID, fetched.contains(where: { $0.id == selectedJobID }) {
                status = "\(fetched.count) job\(fetched.count == 1 ? "" : "s")"
            } else {
                selectedJobID = selectFirstWhenEmpty ? fetched.first?.id : nil
                status = "\(fetched.count) job\(fetched.count == 1 ? "" : "s")"
            }
        } catch {
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
        let imported = report.imported.count
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

    private func startStreaming(_ jobID: String) {
        streamTask?.cancel()
        guard hasServerClient, let client = service.cluster.client else { return }
        logLines = jobs.first(where: { $0.id == jobID })?.logTail ?? []
        isStreaming = true
        streamTask = Task {
            do {
                try await client.streamJobLog(jobID: jobID) { line in
                    await MainActor.run {
                        if logLines.last != line {
                            logLines.append(line)
                        }
                        if logLines.count > 2_000 {
                            logLines.removeFirst(logLines.count - 2_000)
                        }
                    }
                }
                await MainActor.run {
                    isStreaming = false
                    streamTask = nil
                }
                await refreshJobs(selectFirstWhenEmpty: false)
            } catch is CancellationError {
                await MainActor.run {
                    isStreaming = false
                    streamTask = nil
                }
            } catch {
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
                Task { await retryEvaluate(job, retry) }
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
        _ retry: (experiment: String, partialRunID: String)
    ) async {
        status = "retrying missing judgments for \(retry.experiment)…"
        guard let client = service.cluster.client else {
            status = "not connected to a server"
            return
        }
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
        PipelineImportTriage.awaitingImport(
            pipelines,
            importedRunIDs:
                service.cluster.evidenceAutoImport?.importedRunIDs ?? [],
            localRunExists: EvidenceAutoImportService.localRunExists)
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
                Task { await importPipelineEvidence(row) }
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
        _ row: ClusterClient.PipelineRunSummary
    ) async {
        status = "packaging evidence for pipeline \(row.run) on the server…"
        let importer = service.cluster.registerEvidenceAutoImport()
        if let event = await importer.importPipeline(runID: row.run) {
            switch event.outcome {
            case .imported(let runDirectory):
                status = "pipeline \(row.run) evidence imported → "
                    + "runs/\(URL(filePath: runDirectory).lastPathComponent)"
                    + " (hashes verified)"
            case .skippedAlreadyPresent:
                status = "pipeline \(row.run) is already in this workspace "
                    + "— recorded in the ledger"
            case .skippedUnbundleable(let note):
                status = "pipeline \(row.run) skipped — \(note)"
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
    /// auto-import ledger (imported here, or found already present).
    private func jobEvidenceImported(_ job: RemoteJobRecord) -> Bool {
        guard let importer = service.cluster.evidenceAutoImport,
            let candidate = EvidenceAutoImportService.candidate(fromJob: job)
        else { return false }
        return importer.isImported(bundlePath: candidate.bundlePath)
    }

    /// Import through the auto-import service when it can see the bundle
    /// (records the ledger entry, so the chip flips to "imported ✓");
    /// otherwise the same verified panel path the Studies rows use. Either
    /// way the landing is the existing hash-verified importer.
    private func importEvidence(_ job: RemoteJobRecord) async {
        status = "importing evidence from job \(job.id)…"
        let importer = service.cluster.registerEvidenceAutoImport()
        if let event = await importer.importNow(job: job) {
            switch event.outcome {
            case .imported(let runDirectory):
                status = "evidence from job \(job.id) imported → "
                    + "runs/\(URL(filePath: runDirectory).lastPathComponent) (hashes verified)"
            case .skippedAlreadyPresent:
                status = "run \(event.runId ?? "?") is already in this workspace — "
                    + "recorded in the ledger"
            case .skippedUnbundleable(let note):
                status = "run \(event.runId ?? "?") skipped — \(note)"
            case .failed(let message):
                status = "evidence import failed: \(message)"
            }
            return
        }
        // No bundle in the job's result payload — fall back to the panel
        // path, which reports its own reason.
        await service.experiments.importEvidence(fromJobID: job.id)
        status = service.experiments.remoteStatus ?? status
    }

    /// Manual resume of a checkpointed job: the server re-sbatches the
    /// job's own run.sbatch (the same implementation auto-resume uses) and
    /// the run continues from its checkpoint. Refusal details (already
    /// resubmitted / still running / cancelled) surface verbatim.
    private func resubmit(_ jobID: String) async {
        guard let client = service.cluster.client else { return }
        do {
            let result = try await client.resubmitJob(jobID)
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

    private func cancel(_ jobID: String) async {
        guard let client = service.cluster.client else { return }
        do {
            try await client.cancelJob(jobID)
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

    private func formatTimestamp(_ timestamp: Double) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .medium
        return formatter.string(from: Date(timeIntervalSince1970: timestamp))
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

    private func copyToClipboard(_ value: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(value, forType: .string)
    }
}
