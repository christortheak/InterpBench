import ExperimentKit
import SteeringKit
import SwiftUI
import UniformTypeIdentifiers

struct StudyRunControlsView: View {
    @Bindable var service: ChatService
    let manifest: ExperimentManifest
    @Binding var runOnServerExpanded: Bool
    @Binding var pendingModelJob: PendingModelJob?
    /// Takes the researcher to the Compute section, where the durable job
    /// lives after submission (audit 10: the panel named a job id and then
    /// offered no way to reach it). Optional so this view stays usable
    /// without a navigation host; `ExperimentsPanelView` supplies it.
    var openCompute: (() -> Void)?
    @State private var runner = UnifiedStudyRunner()
    @State private var runSubstrate: SubstrateRouting.Substrate?
    @State private var confirmForcedOverride = false
    @State private var confirmCancelJob = false
    @State private var reconnectJobID = ""
    /// The job whose log stream the user stopped by hand — a stopped stream
    /// must not keep looking live.
    @State private var stoppedLogJobID: String?
    private var panel: ExperimentPanel { service.experiments }
    private func serverRunCaption(_ verb: String) -> String {
        "\(verb) runs on \(service.cluster.substrateLabel) as a durable job — reconnect from Compute"
    }

    var body: some View {
        Group {
            runControls(manifest: manifest, panel: panel)
            remoteRunControls(manifest: manifest, panel: panel)
        }
        .onChange(of: manifest.name) {
            runSubstrate = nil
            runner.clearPreflight()
        }
        .onChange(of: panel.remoteJobs.remoteJobID) { stoppedLogJobID = nil }
    }

    /// WS6.3: the substrate decision for the ONE Run control — every rule
    /// (default scope, stochastic pinning, connect-first greying) is
    /// unit-tested in `SubstrateRouting`, not here.
    private func runDecision(manifest: ExperimentManifest) -> SubstrateRouting.Decision {
        SubstrateRouting.decide(
            SubstrateRouting.Inputs(
                temperature: manifest.temperature,
                samplesPerItem: manifest.samplesPerItem,
                // E1: routing follows the resolved execution PLAN. A
                // logprob-only study never samples, so its temperature must
                // not pin it to the server.
                outcomeInstruments: manifest.outcomeInstruments,
                activeWorkspaceIsServer: panel.isServerWorkspace,
                siteRegistered: !service.cluster.servers.isEmpty,
                siteName: service.cluster.activeServer?.name
                    ?? service.cluster.servers.first?.name,
                serverConnected: service.cluster.capabilities != nil,
                userSelection: runSubstrate))
    }

    @ViewBuilder
    private func runControls(manifest: ExperimentManifest, panel: ExperimentPanel) -> some View {
        let decision = runDecision(manifest: manifest)
        // Greedy-only is a LOCAL substrate limitation (no per-run sampling
        // seed in the MLX generator). A stochastic MANIFEST already pins the
        // picker to the server; this gate catches a nonzero draft field.
        // E1: greedy-only is a LOCAL sampler limitation, so it binds only a
        // study that actually samples. A logprob-only study is deterministic
        // whatever the temperature says.
        let plan = ExecutionPlan.resolve(instruments: manifest.outcomeInstruments)
        let requiresGreedy =
            decision.selection == .thisMac
            && plan.samplingIsOperative
            && (manifest.temperature != 0 || panel.draft.runTemperature != 0)
        substratePickerRow(decision: decision)
        runNoteRows(decision: decision)
        // A declared sampling setting this plan will never read. Advisory,
        // not a refusal: the run is well-defined and its result unaffected,
        // but a temperature that decides nothing is a design mistake.
        if let inert = ExecutionPlan.inertSamplingAdvisory(
            instruments: manifest.outcomeInstruments,
            temperature: manifest.temperature,
            samplesPerItem: manifest.samplesPerItem)
        {
            Label(inert, systemImage: "info.circle")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        // P1 — prominent PRE-RUN warning: the data carries options but no
        // categorical instrument is declared, so the run will only generate
        // and parse answer text.
        if let warning = panel.instrumentActivationWarning {
            Label(warning, systemImage: "exclamationmark.triangle")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.orange)
                .textSelection(.enabled)
                .help(
                    "declare the instrument in Evaluation › Outcome mode — "
                        + "measurement method is manifest provenance, never "
                        + "inferred from the data")
        }
        primaryRunRow(
            manifest: manifest, panel: panel, decision: decision,
            requiresGreedy: requiresGreedy)
        preflightRows(manifest: manifest, panel: panel, decision: decision)
        if requiresGreedy {
            Text("Run Study requires saved Temperature = 0 for reproducible measured runs.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        if let runDirectory = panel.localJobs.lastRunDirectory {
            Text(runDirectory)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .help("latest completed study run directory")
        }
        if let serverRunDirectory = panel.remoteJobs.lastServerRunDirectory {
            Text("server run: \(serverRunDirectory)")
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .help("run directory in the active server's runs/ tree")
        }
    }

    @ViewBuilder
    private func substratePickerRow(decision: SubstrateRouting.Decision) -> some View {
        Picker(
            "Run on",
            selection: Binding(
                get: { decision.selection },
                set: { newValue in
                    runSubstrate = newValue
                    runner.clearPreflight()
                })
        ) {
            Text("This Mac").tag(SubstrateRouting.Substrate.thisMac)
            if decision.serverSelectable {
                Text(decision.serverLabel).tag(SubstrateRouting.Substrate.server)
            } else {
                // Greyed, never pickable: connect (or add a site) first.
                // SHORT here — the full hint is a caption row below, because
                // a 60-character segment truncates at the 560 pt panel floor.
                Text("\(decision.serverLabel) (connect first)")
                    .tag(SubstrateRouting.Substrate.server)
                    .selectionDisabled()
            }
        }
        .pickerStyle(.segmented)
        .disabled(decision.pinnedToServer)
        .help(
            "which substrate executes this study — the substrate is a scope, "
                + "not a mode: same manifest, same lifecycle, artifacts land in "
                + "the scoped workspace with their substrate stamp")
    }

    @ViewBuilder
    private func runNoteRows(decision: SubstrateRouting.Decision) -> some View {
        if let note = decision.stochasticNote {
            // The kind enforcement: preselected, explained — never an error
            // later.
            Label(note, systemImage: "die.face.5")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        // The greyed segment now says only "(connect first)", so its full
        // hint has to render here whenever the server arm is unavailable —
        // not just when the server happens to be the selection.
        if let hint = decision.serverHint,
            decision.selection == .server || !decision.serverSelectable
        {
            Text(hint)
                .font(.caption2)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        }
        if let hint = decision.localHint {
            Text(hint)
                .font(.caption2)
                .foregroundStyle(.orange)
        }
    }

    @ViewBuilder
    private func primaryRunRow(
        manifest: ExperimentManifest,
        panel: ExperimentPanel,
        decision: SubstrateRouting.Decision,
        requiresGreedy: Bool
    ) -> some View {
        let busy = panel.localJobs.isRunning || panel.localJobs.isExtracting || runner.isSubmitting
        // Finding 11c: what this button will ACTUALLY submit, stated before
        // it is pressed — the verb lives in a collapsed disclosure and does
        // not consult the Pipeline Composer's declared chain.
        ExecutionPlanEchoRow(
            plan: ExecutionPlanEcho.describe(
                verb: panel.submission.remoteVerb,
                target: decision.selection,
                serverLabel: decision.serverLabel,
                dryRun: panel.submission.remoteDryRun,
                declaredPipelineStages: ShardedSubmission.declaredPipelineStages(
                    manifest.pipeline)),
            revealOptions: { runOnServerExpanded = true })
        HStack(spacing: 8) {
            Button(
                SubstrateRouting.runButtonLabel(
                    decision: decision, verb: panel.submission.remoteVerb,
                    dryRun: panel.submission.remoteDryRun, isBusy: busy)
            ) {
                let runner = runner
                let cluster = service.cluster
                let request = panel.submission.snapshot
                let run: @MainActor () async -> Void = {
                    await runner.run(
                        manifest: manifest, decision: decision, request: request,
                        jobs: panel.remoteJobs, runLocal: { await panel.runStudy() },
                        note: { panel.note($0, severity: $1) }, cluster: cluster)
                }
                // Item 2 + 2026-07-21 incident part 1: warn before a
                // GPU-less model-running server submission — no GPU session,
                // or (more specific) a bundle whose OWN options request no
                // GPU on a Slurm site (non-slurm executor / empty gres would
                // execute inside the controller's small CPU allocation).
                // Dry runs (nothing executes) and non-model verbs (analyze)
                // submit straight through — the predicate rules; so do
                // local runs.
                if decision.selection == .server {
                    ModelJobGPUGate.submit(
                        "study \(panel.submission.remoteVerb)", service: service,
                        pending: $pendingModelJob,
                        bundleOptions: ModelJobSubmissionPreflight.BundleOptions(
                            executor: panel.submission.remoteExecutor,
                            gres: panel.submission.remoteGres,
                            verb: panel.submission.remoteVerb,
                            dryRun: panel.submission.remoteDryRun),
                        fixOptions: {
                            panel.applyGPUAllocationFix()
                            runOnServerExpanded = true
                        },
                        action: run)
                } else {
                    Task { await run() }
                }
            }
            .buttonStyle(.bordered)
            .disabled(
                busy || !panel.violations.isEmpty || requiresGreedy
                    || decision.runBlockedReason != nil
            )
            .help(
                decision.selection == .thisMac
                    ? StudyControlCopy.runHelp : StudyControlCopy.unifiedRemoteRunHelp)
            // A1: a LOCAL in-process run is cancellable between generations
            // (server-routed runs get the Cancel Server Job control instead).
            if panel.localJobs.isRunning {
                ProgressView().controlSize(.small)
                Button("Stop", role: .destructive) { panel.cancelStudyRun() }
                    .controlSize(.small)
                    .disabled(panel.localJobs.studyRunCancelRequested)
                    .help(
                        "stops after the current generation; completed "
                            + "generations stay in the run directory, marked "
                            + "cancelled (no report.json) — reported as cancelled "
                            + "by user, never as an error")
            }
        }

        if decision.selection == .server {
            Text(serverRunCaption("Run"))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    /// WS4 preflight surfacing: ok → silent; warn → amber lines, job already
    /// proceeding; fail → the submission stopped, failing checks shown, and
    /// the forced override sits behind a confirmation that restates them.
    @ViewBuilder
    private func preflightRows(
        manifest: ExperimentManifest,
        panel: ExperimentPanel,
        decision: SubstrateRouting.Decision
    ) -> some View {
        if let preflight = runner.preflight, preflight.verdict == .warn {
            ForEach(preflight.attentionLines) { line in
                Label(line.message, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .textSelection(.enabled)
            }
        }
        if let refusal = runner.refusal {
            VStack(alignment: .leading, spacing: 4) {
                Label(
                    refusal.inlineSummary ?? "preflight failed — submission refused",
                    systemImage: "xmark.octagon"
                )
                .font(.caption.weight(.semibold))
                .foregroundStyle(.red)
                ForEach(refusal.failingLines) { line in
                    Text("• \(line.message)")
                        .font(.caption)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                }
                Button("Override (forced)…") { confirmForcedOverride = true }
                    .controlSize(.small)
                    .help(
                        "resubmit with force=true, bypassing the failed preflight "
                            + "checks — loud and deliberate, never silent"
                    )
                    .confirmationDialog(
                        "Force past preflight?",
                        isPresented: $confirmForcedOverride,
                        titleVisibility: .visible
                    ) {
                        Button("Submit anyway (forced)", role: .destructive) {
                            Task {
                                await runner.run(
                                    manifest: manifest, decision: decision,
                                    request: panel.submission.snapshot, jobs: panel.remoteJobs,
                                    runLocal: { await panel.runStudy() },
                                    note: { panel.note($0, severity: $1) },
                                    cluster: service.cluster,
                                    force: true)
                            }
                        }
                    } message: {
                        Text(refusal.overrideConfirmationMessage)
                    }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.red.opacity(0.07), in: RoundedRectangle(cornerRadius: 8))
        }
        if let status = runner.statusLine {
            Text(status)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
        }
    }

    /// Prominent warning callout for a study that exists locally but not in
    /// the active server's workspace: Run Server Copy is disabled and this
    /// box names both ways forward (portable bundle, or pair the server).

    /// WS6.3: the retired "Run on Server" split is now "Remote options" —
    /// the inputs the unified Run button uses when the substrate picker
    /// selects the site (verb, executor, dry run, resources), plus job
    /// reconnect/cancel/evidence utilities. No submit button lives here:
    /// the ONE Run control above is the submission path.
    @ViewBuilder
    private func remoteRunControls(manifest: ExperimentManifest, panel: ExperimentPanel)
        -> some View
    {
        @Bindable var panel = panel
        @Bindable var options = panel.submission
        let decision = runDecision(manifest: manifest)
        if decision.selection == .server || panel.isServerWorkspace {
            DisclosureGroup("Remote options", isExpanded: $runOnServerExpanded) {
                LabeledContent("Server", value: service.cluster.serverHostLabel)
                    .help(
                        "shared server connection — edit the URL and bearer token in the "
                            + "window toolbar's substrate selector")
                if let summary = panel.remoteJobs.remoteProfileSummary {
                    LabeledContent("Backend", value: summary)
                        .font(.caption)
                        .help(
                            "server profile · executor · launch topology reported by /api/capabilities"
                        )
                }
                Picker("Verb", selection: $options.remoteVerb) {
                    Text("verify").tag("verify")
                    // A11: symmetric artifact production — extract exists on the
                    // bundle path (VALID_STUDY_VERBS) like every other verb.
                    Text("extract").tag("extract")
                    Text("validate").tag("validate")
                    Text("sweep").tag("sweep")
                    Text("run").tag("run")
                    Text("evaluate").tag("evaluate")
                    // A3 rider: analyze exists server-side (headless paired
                    // statistics over the newest completed run).
                    Text("analyze").tag("analyze")
                    // Stage 3: the chain runner — the manifest's declared
                    // pipeline stages as ONE submission (one model load,
                    // gate-aborted between stages). EXPERIMENTAL until stage 5
                    // lands the abort/awaiting UI: the server refuses a
                    // manifest with no explicit pipeline block, and an abort
                    // reads as a successful job here — check pipeline.json /
                    // pipeline-abort.json in the run directory.
                    Text("pipeline (experimental)").tag("pipeline")
                }
                .help("the experiment verb the unified Run button submits remotely")
                // "local" here is the SERVER's own controller process, not
                // this Mac — two rows under a "Run on: This Mac / <site>"
                // picker that reading was a real trap (audit 10). The wire
                // values are unchanged; only what the researcher reads is.
                Picker("Executor", selection: $options.remoteExecutor) {
                    Text("controller (no scheduler)").tag("local")
                    Text("Slurm batch job").tag("slurm")
                }
                .help(
                    "where on the server the job runs: inside the server's own "
                        + "controller process — a small CPU allocation, no "
                        + "scheduler — or as a Slurm batch job with the GPU and "
                        + "walltime below. Neither runs on this Mac")
                Toggle("Dry run (prepare only, nothing executes)", isOn: $options.remoteDryRun)
                    .help(
                        "stages the bundle and renders the job without executing "
                            + "the study — the job finishes as 'prepared'")
                // Labelled, because the defaults are non-empty and therefore
                // the placeholders never show: two bare boxes reading 'A100'
                // and '04:00:00' said nothing about which was which.
                LabeledContent("GPU type (gres)") {
                    TextField("e.g. A100", text: $options.remoteGres)
                        .textFieldStyle(.roundedBorder)
                        .help(Self.gresHelp)
                }
                .help(Self.gresHelp)
                LabeledContent("Walltime (HH:MM:SS)") {
                    TextField("e.g. 04:00:00", text: $options.remoteWalltime)
                        .textFieldStyle(.roundedBorder)
                        .help(Self.walltimeHelp)
                }
                .help(Self.walltimeHelp)
                // Resume-on-checkpoint (2026-07-22 incident: a checkpointed run
                // had no resume path) — DEFAULT ON: a checkpointed batch run
                // continuing is what submitting it asked for.
                HStack {
                    Toggle(
                        "Resume automatically if the run checkpoints",
                        isOn: $options.remoteResumePolicy.autoResubmit)
                    Stepper(value: $options.remoteResumePolicy.limit, in: 1...50) {
                        Text("up to \(panel.submission.remoteResumePolicy.limit) restarts")
                            .font(.caption)
                    }
                    .disabled(!panel.submission.remoteResumePolicy.autoResubmit)
                }
                .help(
                    "when a Slurm run exits at the walltime margin with a clean "
                        + "checkpoint (exit 85), the server re-submits the job's "
                        + "own sbatch script and the run continues from the "
                        + "checkpoint — up to this many restarts. Off, the job "
                        + "parks as 'checkpointed (resumable)' until you press "
                        + "Resume. Slurm submissions only; the transcript line "
                        + "stamps what was sent")
                // Multi-GPU fan-out (2026-07-22): shard a Slurm run across K
                // sibling GPU jobs; the server merges the partials back into one
                // run byte-identical to a single job (records are independent).
                // Disabled-with-explanation when this submission cannot shard
                // (finding 5: the server shards only run and run-FIRST
                // pipelines — a stepper the server would ignore must say so).
                let shardingReason = ShardedSubmission.shardingUnavailableReason(
                    verb: panel.submission.remoteVerb, executor: panel.submission.remoteExecutor,
                    declaredPipelineStages: ShardedSubmission.declaredPipelineStages(
                        manifest.pipeline))
                Stepper(
                    value: $options.remoteParallelJobs,
                    in: 1...ShardedSubmission.stepperCap(siteMax: siteMaxParallelGPUJobs)
                ) {
                    Text(
                        panel.submission.remoteParallelJobs > 1
                            ? "Parallel GPU jobs: \(panel.submission.remoteParallelJobs)"
                            : "Parallel GPU jobs: 1 (no sharding)"
                    )
                    .font(.caption)
                }
                .disabled(shardingReason != nil)
                .help(
                    "shard the run across this many simultaneous GPU jobs "
                        + "(Slurm, run/pipeline-first-stage-run only) — every "
                        + "record is independent, so the merged result is "
                        + "byte-identical to a single job and wall-clock scales "
                        + "with GPUs. The cap comes from the site profile's "
                        + "'Max parallel GPU jobs' (check yours with `sacctmgr "
                        + "show qos format=Name,MaxTRESPerUser`)")
                if let shardingReason {
                    Text(shardingReason)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                remoteJobActionsRow(panel: panel)
                Text(StudyControlCopy.remoteOptionsCaption)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text(StudyControlCopy.importEvidenceCaption)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                // Reconnect by job id — e.g. after an app restart while a Slurm job runs.
                HStack {
                    TextField("job id to reconnect", text: $reconnectJobID)
                        .textFieldStyle(.roundedBorder)
                    Button("Reconnect") {
                        Task { await panel.reconnectRemoteJob(reconnectJobID) }
                    }
                    .disabled(reconnectJobID.trimmingCharacters(in: .whitespaces).isEmpty)
                    if !panel.remoteJobs.remoteLogLines.isEmpty {
                        Button("Stop Log") {
                            stoppedLogJobID = panel.remoteJobs.remoteJobID
                            panel.stopRemoteLogStream()
                        }
                        .help(
                            "stops following this job's log here — the job "
                                + "itself keeps running on the server, and "
                                + "Reconnect picks the stream back up")
                    }
                }
                .help(
                    "resume watching a running or finished job by its id after an app or session restart"
                )
                if let job = panel.remoteJobs.remoteJobID {
                    HStack(spacing: 8) {
                        LabeledContent("Remote job", value: job)
                            .font(.caption)
                            .textSelection(.enabled)
                        if let openCompute {
                            Button("Show in Compute", action: openCompute)
                                .buttonStyle(.link)
                                .font(.caption)
                                .help(
                                    "opens the Compute section, where every "
                                        + "server job of this workspace is "
                                        + "listed with its state and log")
                        }
                    }
                }
                if let uploaded = panel.remoteJobs.remoteLastUploadedBundle {
                    LabeledContent("Uploaded bundle", value: uploaded)
                        .font(.caption2.monospaced())
                        .textSelection(.enabled)
                }
                if let imported = panel.remoteJobs.remoteImportedRunDirectory {
                    LabeledContent("Imported run", value: imported)
                        .font(.caption2.monospaced())
                        .textSelection(.enabled)
                }
                if let status = panel.remoteJobs.remoteStatus {
                    Text(status)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                if !panel.remoteJobs.remoteLogLines.isEmpty {
                    // The log is always titled with the job id verbatim so it can
                    // be copied and reconnected to later.
                    HStack(spacing: 6) {
                        Text("job log — \(panel.remoteJobs.remoteJobID ?? "?")")
                            .font(.caption.monospaced())
                            .textSelection(.enabled)
                        // A stopped or failed stream leaves the last lines on
                        // screen unchanged, which reads exactly like a live
                        // one (audit 10). Say which it is.
                        if let badge = logStreamBadge {
                            Label(badge, systemImage: "pause.circle")
                                .font(.caption2)
                                .foregroundStyle(.orange)
                                .help(
                                    "these lines are frozen where the stream "
                                        + "ended — the job may still be running "
                                        + "on the server; Reconnect resumes "
                                        + "following it")
                        }
                    }
                    ScrollView {
                        Text(panel.remoteJobs.remoteLogLines.joined(separator: "\n"))
                            .font(.caption2.monospaced())
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(minHeight: 120, maxHeight: 260)
                }
            }
            .help(
                "inputs for the unified Run button's remote submission (verb, "
                    + "executor, dry run, resources) plus job reconnect and "
                    + "evidence utilities")
        }
    }

    /// Session-scoped list of server jobs submitted from this panel (run
    /// verbs and bundle submissions), ids selectable so a researcher can
    /// reconnect after an app restart.

    /// Connection test, job cancel and evidence import for the current
    /// durable job. Own function: the cancel is destructive and now carries a
    /// confirmation, and `remoteRunControls` is already a long builder.
    @ViewBuilder
    private func remoteJobActionsRow(panel: ExperimentPanel) -> some View {
        HStack {
            Button("Test Connection") { Task { await panel.testRemoteConnection() } }
                .help(
                    "asks the server for its capabilities and reports what "
                        + "came back — submits nothing and changes nothing")
            Button("Cancel Job") { confirmCancelJob = true }
                .disabled(panel.remoteJobs.remoteJobID == nil)
                .help(
                    "cancels the durable job on "
                        + "\(service.cluster.substrateLabel) — a queued Slurm "
                        + "job loses its queue slot and a running one loses "
                        + "the work it has not written")
                .confirmationDialog(
                    cancelJobDialogTitle,
                    isPresented: $confirmCancelJob,
                    titleVisibility: .visible
                ) {
                    Button("Cancel Job", role: .destructive) {
                        Task { await panel.cancelRemoteJob() }
                    }
                    Button("Keep Running", role: .cancel) {}
                } message: {
                    Text(Self.cancelJobConsequence)
                }
            Button("Import Evidence") { Task { await panel.downloadRemoteEvidence() } }
                .disabled(panel.remoteJobs.remoteJobID == nil)
                .help(
                    "downloads this job's evidence bundle, verifies its hashes "
                        + "and lands it as a new immutable runs/ directory in "
                        + "this workspace")
        }
    }

    private static let cancelJobConsequence =
        "The job stops on the server. A queued job loses its place in the "
        + "queue; a running job keeps only what it already wrote to its run "
        + "directory, and no report is produced. This cannot be undone — "
        + "resubmitting starts a new job."

    private static let gresHelp =
        "the Slurm generic resource this job asks for (sent as "
        + "--gres=gpu:<type>) — the vocabulary is the active site profile's "
        + "GPU types. Blank asks for no GPU, which on a model-running verb "
        + "means the controller's small CPU allocation"

    private static let walltimeHelp =
        "the Slurm time limit for this job, in Slurm's own format — HH:MM:SS, "
        + "or D-HH:MM:SS for more than a day. The scheduler kills the job at "
        + "the limit, which is what the resume policy below exists for"

    /// Names the object a destructive click will act on: the job id, the
    /// verb it was submitted with (from the recent-jobs record, not the
    /// picker's current value), and the substrate it runs on.
    private var cancelJobDialogTitle: String {
        let id = panel.remoteJobs.remoteJobID ?? "?"
        let verb =
            panel.remoteJobs.recentServerJobs.first { $0.id == id }?.verb
            ?? panel.submission.remoteVerb
        return "Cancel job \(id) (\(verb)) on \(service.cluster.substrateLabel)?"
    }

    /// "stream stopped" / "stream failed" for the log header — nil while the
    /// stream is (as far as this view can tell) live.
    private var logStreamBadge: String? {
        if let status = panel.remoteJobs.remoteStatus,
            status.hasPrefix("remote log stream failed")
                || status.hasPrefix("log follow refused")
        {
            return "stream failed"
        }
        if let stoppedLogJobID, stoppedLogJobID == panel.remoteJobs.remoteJobID {
            return "stream stopped"
        }
        return nil
    }

    /// The active site profile's cap on sharded fan-out (nil = uncapped;
    /// the stepper falls back to `ShardedSubmission.defaultStepperCap`).
    private var siteMaxParallelGPUJobs: Int? {
        guard case .slurm(let slurm)? = service.cluster.activeSite?.scheduler
        else { return nil }
        return slurm.maxParallelGPUJobs
    }
}
