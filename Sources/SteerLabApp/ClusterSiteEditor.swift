import ExperimentKit
import SwiftUI

/// Sheet payload for editing an existing site. Present the editor with
/// `.sheet(item:)` and this wrapper — never `.sheet(isPresented:)` plus an
/// `if let` in the content closure, which intermittently renders the nil
/// branch (a blank, dismissable modal) when presentation races the state
/// write.
struct SiteEditTarget: Identifiable {
    let id: ClusterConnectionStore.ServerEntry.ID
}

/// WS1 site-profile editor (turnkey-cluster plan): a form over the ENTIRE
/// `ClusterSiteProfile`, driving `ClusterConnectionStore.updateSite`/`addSite`.
/// Reachable from the connection dot's "Edit Site…" and from the setup
/// wizard's "New Site…". The view is a renderer: every field→profile rule,
/// validation message, and the live STEERLAB_* env preview comes from
/// `SiteEditorModel` (ExperimentKit), where it is unit-tested.
///
/// Layout note (macOS 27 beta): this is a fixed-minimum sheet, never a
/// split-view column. The schema-2 long tail lives in `DisclosureGroup`s inside
/// the existing sections, so the form's own minimum height never changes with
/// what is expanded.
struct ClusterSiteEditor: View {
    @Bindable var cluster: ClusterConnectionStore
    /// nil = create a new site entry on Save.
    let entryID: ClusterConnectionStore.ServerEntry.ID?
    /// Seed profile for a brand-new site (ignored when `entryID` is set).
    var seedProfile: ClusterSiteProfile? = nil
    /// Called with the saved entry (new or updated) before dismissing.
    var onSaved: ((ClusterConnectionStore.ServerEntry) -> Void)? = nil

    @Environment(\.dismiss) private var dismiss
    @State private var model: SiteEditorModel

    init(
        cluster: ClusterConnectionStore,
        entryID: ClusterConnectionStore.ServerEntry.ID?,
        seedProfile: ClusterSiteProfile? = nil,
        onSaved: ((ClusterConnectionStore.ServerEntry) -> Void)? = nil
    ) {
        self.cluster = cluster
        self.entryID = entryID
        self.seedProfile = seedProfile
        self.onSaved = onSaved
        let initial = entryID.flatMap { cluster.server(id: $0)?.resolvedSite } ?? seedProfile
        _model = State(initialValue: SiteEditorModel(profile: initial))
    }

    var body: some View {
        @Bindable var model = model
        VStack(spacing: 0) {
            Form {
                Group {
                    identitySection($model)
                    transportSection($model)
                    topologySection($model)
                    schedulerSection($model)
                }
                Group {
                    environmentSection($model)
                    constraintsSection($model)
                    policySection($model)
                    bootstrapSection($model)
                    issuesSection
                    environmentPreviewSection
                }
            }
            .formStyle(.grouped)
            Divider()
            footer
        }
        .frame(minWidth: 620, idealWidth: 680, minHeight: 620, idealHeight: 760)
    }

    // MARK: Sections

    private func identitySection(_ model: Bindable<SiteEditorModel>) -> some View {
        Section("Identity") {
            TextField("Name", text: model.name)
                .help("label shown in the substrate menu and on exports")
            VStack(alignment: .leading, spacing: 4) {
                Text("Notes")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextEditor(text: model.notes)
                    .font(.callout)
                    .frame(minHeight: 56, maxHeight: 110)
                    .overlay(
                        RoundedRectangle(cornerRadius: 4)
                            .strokeBorder(.quaternary))
            }
            .help("free-form operator notes (storage-root reminders, VPN caveats) — never parsed")
            if self.model.schemaVersion < ClusterSiteProfile.currentSchemaVersion {
                Text(
                    "loaded as profile schema v\(self.model.schemaVersion) — carried through "
                        + "unchanged on save (legacy defaults still apply to it)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func transportSection(_ model: Bindable<SiteEditorModel>) -> some View {
        Section("Transport") {
            Picker("Kind", selection: model.transportKind) {
                Text("Direct URL").tag(SiteEditorModel.TransportKind.direct)
                Text("SSH tunnel").tag(SiteEditorModel.TransportKind.ssh)
            }
            .pickerStyle(.segmented)
            switch self.model.transportKind {
            case .direct:
                TextField("Base URL", text: model.directURLString)
                    .help("directly reachable server URL (LAN workstation, this machine)")
            case .ssh:
                TextField("SSH user@host", text: model.sshHost)
                    .help(
                        "exact login identity used for authentication (for example "
                            + "user@hpc.example.edu), or an ~/.ssh/config alias "
                            + "that supplies the user — the ControlMaster identity must match")
                Text("Include the cluster username unless your SSH config supplies it.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("ProxyJump (optional)", text: model.proxyJump)
                    .help("intermediate jump host for -J, when the daemon's node is not directly reachable")
                TextField("Remote port", text: model.remotePortText)
                    .help("port the SteerLab server listens on at the far side (default 8080)")
                Toggle("VPN expected", isOn: model.vpnExpected)
                    .help("used only for friendlier error messages — never enforced")
            }
        }
    }

    private func topologySection(_ model: Bindable<SiteEditorModel>) -> some View {
        Section("Topology") {
            Picker("Daemon runs as", selection: model.topology) {
                Text("External server").tag(ClusterSiteProfile.Topology.externalServer)
                Text("Login-node daemon").tag(ClusterSiteProfile.Topology.loginDaemon)
                Text("Daemon in a job").tag(ClusterSiteProfile.Topology.daemonInJob)
            }
            Text(SiteEditorModel.topologyExplanation(self.model.topology))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: Scheduler

    @ViewBuilder
    private func schedulerSection(_ model: Bindable<SiteEditorModel>) -> some View {
        Section("Scheduler") {
            Picker("Scheduler", selection: model.schedulerKind) {
                Text("None").tag(SiteEditorModel.SchedulerKind.none)
                Text("Slurm").tag(SiteEditorModel.SchedulerKind.slurm)
            }
            .pickerStyle(.segmented)
            if self.model.schedulerKind == .slurm {
                partitionsTable(model)
                gpuTable(model)
                TextField("Default gres", text: model.defaultGres)
                    .help("e.g. gpu:A100:1 — becomes STEERLAB_SLURM_GRES")
                TextField("Default partition", text: model.defaultPartition)
                    .help(
                        "becomes STEERLAB_SLURM_PARTITION; empty falls back to the first "
                            + "gpu-named partition, then the first partition")
                Toggle("Account required", isOn: model.accountRequired)
                    .help("whether sbatch refuses submissions without --account at this site")
                TextField("Account", text: model.account)
                    .help("Slurm --account / STEERLAB_SLURM_ACCOUNT (lab allocation name)")
                Toggle("Billed allocations", isOn: model.billedAllocations)
                    .help(
                        "drives daemon-idle policy: a daemon-in-a-job starts on session-"
                            + "open and stops on session-close instead of staying resident")
                // Grouped so the schema-2 long tail stays inside one
                // ViewBuilder slot (and collapsed by default).
                Group {
                    directivesGroup(model)
                    commandsGroup(model)
                    jobDefaultsGroup(model)
                    limitsGroup(model)
                    jobClassesGroup(model)
                }
            }
        }
    }

    private func partitionsTable(_ model: Bindable<SiteEditorModel>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Partitions")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                Text("name").frame(maxWidth: .infinity, alignment: .leading)
                Text("max h").frame(width: 60, alignment: .leading)
                Text("allowed GPU types").frame(width: 150, alignment: .leading)
                Text("qos").frame(width: 80, alignment: .leading)
                Spacer().frame(width: 22)
            }
            .font(.caption2)
            .foregroundStyle(.tertiary)
            ForEach(model.partitions) { $row in
                HStack {
                    TextField("name", text: $row.name)
                    TextField("hours", text: $row.maxWalltimeHoursText)
                        .frame(width: 60)
                    TextField("all", text: $row.allowedGPUTypesText)
                        .frame(width: 150)
                    TextField("site", text: $row.qos)
                        .frame(width: 80)
                    Button {
                        self.model.removePartition(id: row.id)
                    } label: {
                        Image(systemName: "minus.circle")
                    }
                    .buttonStyle(.borderless)
                    .help("remove this partition row")
                }
            }
            Button {
                self.model.addPartition()
            } label: {
                Label("Add partition", systemImage: "plus")
            }
            .controlSize(.small)
            .help(
                "allowed GPU types: comma-separated subset of the inventory below; empty "
                    + "means the whole site vocabulary. qos overrides the site-wide QOS.")
        }
    }

    /// The site's hardware inventory — the hand/UI authoring route of WP5 §4.3.
    private func gpuTable(_ model: Bindable<SiteEditorModel>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("GPU inventory")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                Text("type").frame(maxWidth: .infinity, alignment: .leading)
                Text("VRAM (GB)").frame(width: 90, alignment: .leading)
                Text("compute capability").frame(width: 140, alignment: .leading)
                Spacer().frame(width: 22)
            }
            .font(.caption2)
            .foregroundStyle(.tertiary)
            ForEach(model.gpuRows) { $row in
                HStack {
                    TextField("A100", text: $row.gpuType)
                    TextField("80", text: $row.vramGBText)
                        .frame(width: 90)
                    TextField("sm_80", text: $row.computeCapability)
                        .frame(width: 140)
                    Button {
                        self.model.removeGPURow(id: row.id)
                    } label: {
                        Image(systemName: "minus.circle")
                    }
                    .buttonStyle(.borderless)
                    .help("remove this GPU type")
                }
            }
            Button {
                self.model.addGPURow()
            } label: {
                Label("Add GPU type", systemImage: "plus")
            }
            .controlSize(.small)
            .help(
                "the site's --gres vocabulary: one row per concrete GPU type, with VRAM "
                    + "for the memory-fit preflight and CUDA compute capability (sm_80) so "
                    + "a torch build that ships no kernels for it can be refused")
        }
    }

    @ViewBuilder
    private func directivesGroup(_ model: Bindable<SiteEditorModel>) -> some View {
        DisclosureGroup("Directives & required headers") {
            TextField("QOS", text: model.qos)
                .help("site-wide --qos; a partition row may override it")
            TextField("Constraints", text: model.schedulerConstraintsText)
                .help("--constraint node-feature tokens, AND-ed (comma separated)")
            TextField("Reservation", text: model.reservation)
                .help("--reservation name, when the site has granted one")
            TextField("Required headers", text: model.requiredHeadersText)
                .help(
                    "headers this site's sbatch rejects a job without: "
                        + SiteEditorModel.requiredHeaderVocabulary.joined(separator: ", "))
            linesEditor(
                "Extra #SBATCH arguments (one per line)", text: model.extraSbatchText,
                help: "emitted verbatim for every job, in order (e.g. --exclusive)")
        }
    }

    @ViewBuilder
    private func commandsGroup(_ model: Bindable<SiteEditorModel>) -> some View {
        DisclosureGroup("Scheduler commands") {
            TextField("Submit command", text: model.submitCommand)
                .help("binary NAME only, never a shell string (default sbatch)")
            TextField("Query command", text: model.queryCommand)
                .help("default squeue — some sites ship a wrapper")
            TextField("Accounting command", text: model.accountingCommand)
                .help("default sacct")
            TextField("Cancel command", text: model.cancelCommand)
                .help("default scancel")
        }
    }

    @ViewBuilder
    private func jobDefaultsGroup(_ model: Bindable<SiteEditorModel>) -> some View {
        DisclosureGroup("Job defaults & interruption") {
            TextField("Default memory", text: model.defaultMemory)
                .help("--mem for a generic study job, e.g. 80G")
            TextField("Default walltime", text: model.defaultWalltime)
                .help("HH:MM:SS for a generic study job")
            TextField("Default CPUs per task", text: model.defaultCPUsPerTaskText)
            Toggle("Requeue on preemption", isOn: model.requeue)
                .help("#SBATCH --requeue: the site may restart an interrupted job")
            Toggle("Auto-resubmit", isOn: model.autoResubmit)
                .help("the engine resubmits from its checkpoint when a job hits the wall")
            TextField("Auto-resubmit limit", text: model.autoResubmitLimitText)
            TextField("Checkpoint signal lead (s)", text: model.signalSecondsText)
                .help("seconds before the walltime wall that the checkpoint signal fires")
            Picker("Signal target", selection: model.signalTarget) {
                ForEach(SiteEditorModel.signalTargetVocabulary, id: \.self) { target in
                    Text(target).tag(target)
                }
            }
            Picker("Export mode", selection: model.exportMode) {
                ForEach(SiteEditorModel.exportModeVocabulary, id: \.self) { mode in
                    Text(mode).tag(mode)
                }
            }
            .help("#SBATCH --export: jobs normally run with a clean environment (none)")
        }
    }

    @ViewBuilder
    private func limitsGroup(_ model: Bindable<SiteEditorModel>) -> some View {
        DisclosureGroup("Limits & submission") {
            TextField("Max parallel GPU jobs", text: model.maxParallelGPUJobsText)
                .help(ShardedSubmission.siteFieldHelp)
            TextField("Max submitted jobs", text: model.maxSubmittedJobsText)
                .help("per-user queued cap (sacctmgr show qos format=Name,MaxTRESPerUser)")
            TextField("Max running jobs", text: model.maxRunningJobsText)
            TextField("Accounting visibility grace (s)", text: model.accountingVisibilityGraceSecondsText)
                .help(
                    "how long sacct/squeue may lag a fresh submission before “unknown” "
                        + "stops meaning “accounting lag”")
            Toggle("Submit from the bundle directory", isOn: model.submitFromBundleDirectory)
            TextField("Job-name prefix", text: model.jobNamePrefix)
        }
    }

    @ViewBuilder
    private func jobClassesGroup(_ model: Bindable<SiteEditorModel>) -> some View {
        DisclosureGroup("Job classes") {
            jobClassFields(
                "Controller job", fields: model.controllerJob, showPort: true,
                showIdleMinutes: false)
            Divider()
            jobClassFields(
                "Setup (bootstrap) job", fields: model.setupJob, showPort: false,
                showIdleMinutes: false)
            Divider()
            jobClassFields(
                "GPU session", fields: model.gpuSession, showPort: true, showIdleMinutes: true)
        }
    }

    @ViewBuilder
    private func jobClassFields(
        _ title: String,
        fields: Binding<SiteEditorModel.JobClassFields>,
        showPort: Bool,
        showIdleMinutes: Bool
    ) -> some View {
        Text(title)
            .font(.caption)
            .foregroundStyle(.secondary)
        TextField("Partition", text: fields.partition)
        TextField("CPUs per task", text: fields.cpusPerTaskText)
        TextField("Memory", text: fields.memory)
        TextField("Walltime", text: fields.walltime)
        TextField("gres", text: fields.gres)
        if showPort {
            TextField("Port", text: fields.portText)
                .help(
                    "serving port for this job class — the controller's must match the "
                        + "transport's remote port, which is what the tunnel dials")
        }
        if showIdleMinutes {
            TextField("Idle minutes", text: fields.idleMinutesText)
                .help("idle-shutdown timeout for a GPU session")
        }
        linesEditor(
            "Extra #SBATCH arguments (one per line)", text: fields.extraSbatchText,
            help: "emitted for this job class only")
    }

    // MARK: Environment

    @ViewBuilder
    private func environmentSection(_ model: Bindable<SiteEditorModel>) -> some View {
        Section("Environment") {
            Picker("Module system", selection: model.moduleSystem) {
                Text("None").tag(ClusterSiteProfile.SiteEnvironment.ModuleSystem.none)
                Text("Lmod").tag(ClusterSiteProfile.SiteEnvironment.ModuleSystem.lmod)
                Text("Environment Modules")
                    .tag(ClusterSiteProfile.SiteEnvironment.ModuleSystem.environmentModules)
            }
            TextField("Modules", text: model.modulesText)
                .help("loaded in order, comma separated (e.g. CUDA/12.4.0, Miniforge3)")
            Picker("Python provider", selection: model.pythonProvider) {
                ForEach(
                    ClusterSiteProfile.SiteEnvironment.PythonProvider.allCases, id: \.self
                ) { provider in
                    Text(provider.rawValue).tag(provider)
                }
            }
            .help("do NOT assume conda: a site may ship modules, a venv, or system python")
            DisclosureGroup("Python details") {
                TextField("Module init script", text: model.moduleInitScript)
                    .help("sourced before any module call (e.g. /etc/profile.d/modules.sh)")
                TextField("Python version", text: model.pythonVersion)
                TextField("Env prefix", text: model.envPrefix)
                    .help("conda/mamba env prefix, or the venv root")
                TextField("Conda profile script", text: model.condaProfileScript)
                TextField("Conda env name", text: model.condaEnvName)
                TextField("Venv path", text: model.venvPath)
                TextField("Python executable", text: model.pythonExecutable)
                    .help("absolute interpreter for child jobs, when the controller's is not valid on compute nodes")
            }
            DisclosureGroup("Packages") {
                TextField("Torch index URL", text: model.torchIndexURL)
                    .help("empty = default PyPI")
                TextField("Torch variant", text: model.torchVariant)
                    .help("cu128 | rocm6.2 | cpu — must ship kernels for the GPUs above")
                TextField("Server extras", text: model.serverExtrasText)
                    .help("extras installed with the server, comma separated (e.g. all)")
            }
            DisclosureGroup("Paths & hosts") {
                TextField("Env-file path", text: model.envFilePath)
                    .help("remote path of the sourced env file; $HOME/~ expand on the far side")
                TextField("Token-file path", text: model.tokenFilePath)
                    .help("path INDIRECTION only — the token value never enters a profile")
                TextField("Remote repo path", text: model.remoteRepoPath)
                TextField("Interactive allocation command", text: model.interactiveAllocationCommand)
                    .help("printed in refusals (e.g. interact -c 4 --mem 16g) — never executed")
                TextField("Transfer host", text: model.transferHost)
                    .help("egress-capable host used when compute nodes have no network")
                TextField("SSH ControlPersist", text: model.sshControlPersist)
                    .help("multiplexed control-master lifetime (e.g. 8h)")
            }
        }
    }

    // MARK: Constraints / storage

    @ViewBuilder
    private func constraintsSection(_ model: Bindable<SiteEditorModel>) -> some View {
        Section("Constraints") {
            Picker("Compute-node egress", selection: model.computeEgress) {
                Text("Yes").tag(ClusterSiteProfile.SiteConstraints.Egress.yes)
                Text("No").tag(ClusterSiteProfile.SiteConstraints.Egress.no)
                Text("Unknown").tag(ClusterSiteProfile.SiteConstraints.Egress.unknown)
            }
            .help("whether compute nodes can reach the model hub — routes model installs")
            TextField("Workspace root", text: model.workspaceRoot)
                .help("exported as STEERLAB_ROOT (e.g. /scratch/<MyID>/steerlab-workspace)")
            TextField("HF cache root", text: model.hfCacheRoot)
                .help("exported as HF_HOME (e.g. /work/<lab>/hf-cache)")
            TextField("Archive root", text: model.archiveRoot)
                .help("cold storage for purge protection (e.g. /project/<lab>)")
            TextField("Metadata root", text: model.metadataRoot)
                .help(
                    "job DB + serverd.host location; default ~/.steerlab (keep it OFF "
                        + "Lustre — SQLite locking)")
            TextField("Purge days", text: model.purgeDaysText)
                .help("days until untouched workspace files are purged (scratch is commonly 30)")
            TextField("Purge warn days", text: model.purgeWarnDaysText)
                .help("age at which purge risk escalates to a warning (commonly 20)")
            TextField("Maintenance source", text: model.maintenanceSource)
                .help("where maintenance windows are announced (URL or free text); empty = manual entry")
            storageGroup(model)
        }
    }

    @ViewBuilder
    private func storageGroup(_ model: Bindable<SiteEditorModel>) -> some View {
        DisclosureGroup("Storage details") {
            TextField("Node staging template", text: model.nodeStageDirTemplate)
                .help(
                    "expanded ON THE NODE, $VAR placeholders kept verbatim "
                        + "(e.g. /lscratch/$SLURM_JOB_ID)")
            TextField("Node scratch gres", text: model.nodeScratchGres)
                .help(
                    "Slurm gres token requesting the node-local scratch a staged "
                        + "model needs, e.g. lscratch:100 (GB). Rides the "
                        + "GPU-bearing job classes only; scheduling accounting "
                        + "only — Slurm does not enforce it. Empty = not requested.")
            Picker("Model-hub offline mode", selection: model.hubOfflineMode) {
                Text("Auto (from egress)")
                    .tag(ClusterSiteProfile.SiteStorage.OfflineMode.auto)
                Text("Offline").tag(ClusterSiteProfile.SiteStorage.OfflineMode.offline)
                Text("Online").tag(ClusterSiteProfile.SiteStorage.OfflineMode.online)
            }
            Toggle(
                "Metadata root needs a local filesystem",
                isOn: model.metadataRequiresLocalFilesystem
            )
            .help("the SQLite job DB needs POSIX locks — not the parallel filesystem")
            TextField("Housekeeping scan file cap", text: model.scanFileCapText)
            TextField("Free-space warn (GB)", text: model.freeSpaceWarnGBText)
            TextField("Free-space fail (GB)", text: model.freeSpaceFailGBText)
            TextField("Pre-stage minimum free (GB)", text: model.prestageMinFreeGBText)
            TextField("Maintenance calendar stale days", text: model.calendarStaleDaysText)
            TextField("Quota command", text: model.quotaCommand)
                .help("its output is DISPLAYED, never parsed")
            TextField("Scanned storage roles", text: model.scannedRolesText)
                .help("roles the housekeeping scan walks; empty = workspace, metadata, hfCache")
        }
    }

    // MARK: Policy

    @ViewBuilder
    private func policySection(_ model: Bindable<SiteEditorModel>) -> some View {
        Section("Policy") {
            Toggle("Login nodes may run compute", isOn: model.loginNodesAllowCompute)
                .help("off = the bootstrap refuses to compute on a matching host")
            Toggle("Require an allocation", isOn: model.loginNodesRequireAllocation)
                .help("additionally require SLURM_JOB_ID to be set")
            linesEditor(
                "Login-node hostname patterns (one regex per line)",
                text: model.loginNodeHostnamePatternsText,
                help: "matched against `hostname`; empty = no hostname rule, which never refuses")
            TextField("Maintenance calendar path", text: model.maintenanceCalendarPath)
                .help("remote path of the hand-authored window file the engine reads")
            TextField("Maintenance source URL", text: model.maintenanceSourceURL)
                .help("fetched by the app, never by a job")
            TextField("Maintenance note", text: model.maintenanceSourceNote)
            TextField("Transfer method", text: model.transferMethod)
                .help("bulk artifact movement, e.g. rsync | globus")
            Picker("External-service egress", selection: model.externalServiceEgress) {
                Text("Yes").tag(ClusterSiteProfile.SiteConstraints.Egress.yes)
                Text("No").tag(ClusterSiteProfile.SiteConstraints.Egress.no)
                Text("Unknown").tag(ClusterSiteProfile.SiteConstraints.Egress.unknown)
            }
            .help("HTTP services OTHER than the model hub (judging catalogues, provider APIs)")
            DisclosureGroup("Server posture overrides") {
                TextField("Bind override", text: model.bindOverride)
                    .help(
                        "empty = the built-in rule (daemon-in-a-job ⇒ 0.0.0.0 + token, "
                            + "everything else ⇒ loopback). A site may only be STRICTER.")
                TextField("Auth mode override", text: model.authModeOverride)
                    .help("token = every route gated; a non-loopback bind without it stays refused")
            }
        }
    }

    private func bootstrapSection(_ model: Bindable<SiteEditorModel>) -> some View {
        Section("Bootstrap") {
            TextField("bootstrap.sh path (optional)", text: model.bootstrapPath)
                .help(
                    "site-local path of the WS5 bootstrap script once provisioned; empty "
                        + "uses <remote repo>/Server/scripts/bootstrap.sh")
        }
    }

    @ViewBuilder
    private var issuesSection: some View {
        let issues = model.issues
        if !issues.isEmpty {
            Section("Checks") {
                ForEach(issues) { issue in
                    Label(
                        issue.message,
                        systemImage: issue.severity == .error
                            ? "xmark.octagon" : "exclamationmark.triangle")
                        .font(.caption)
                        .foregroundStyle(issue.severity == .error ? .red : .orange)
                }
            }
        }
    }

    /// WP5 §3.3's preview surface: the complete generated environment and
    /// scheduler commands, readable before anything runs. Every byte comes from
    /// `SiteEditorModel.preview` (ExperimentKit), which re-renders as the fields
    /// change — this view chooses no wording of its own.
    private var environmentPreviewSection: some View {
        Section("What this site will run") {
            ClusterSitePreviewPanes(preview: model.preview)
        }
    }

    /// A labelled multi-line box for the one-entry-per-line lists (verbatim
    /// `#SBATCH` arguments, hostname regexes), which cannot be tokenized because
    /// an entry may contain spaces and commas.
    private func linesEditor(
        _ title: String, text: Binding<String>, help: String
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            TextEditor(text: text)
                .font(.caption.monospaced())
                .frame(minHeight: 44, maxHeight: 88)
                .overlay(
                    RoundedRectangle(cornerRadius: 4)
                        .strokeBorder(.quaternary))
        }
        .help(help)
    }

    // MARK: Footer

    private var footer: some View {
        HStack {
            if model.isDirty {
                Text("unsaved changes")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Cancel") { dismiss() }
                .keyboardShortcut(.cancelAction)
            Button("Save") { save() }
                .keyboardShortcut(.defaultAction)
                .disabled(!model.canSave)
                .help(
                    model.canSave
                        ? "save this site profile to the registry"
                        : "fix the errors above first (warnings never block)")
        }
        .padding(12)
    }

    /// All mutation goes through the store (views stay thin).
    private func save() {
        let profile = model.builtProfile()
        let entry: ClusterConnectionStore.ServerEntry?
        if let entryID {
            cluster.updateSite(id: entryID, profile: profile)
            entry = cluster.server(id: entryID)
        } else {
            entry = cluster.addSite(profile)
        }
        if let entry { onSaved?(entry) }
        dismiss()
    }
}
