import ExperimentKit
import SwiftUI

/// WS5.3 "Set Up Cluster…" wizard: a veneer over `ClusterProvisioner`
/// (ExperimentKit), which owns the step machine, every composed command, the
/// dry-run-before-real gate, and all transcripts. The wizard renders step
/// state, forwards button presses, and supplies the one piece of app glue the
/// provisioner cannot know: the connect+register handler (store + tunnel +
/// capabilities through the existing client).
struct ClusterSetupWizard: View {
    @Bindable var cluster: ClusterConnectionStore
    var tunnel: ClusterTunnel
    let service: ChatService

    @Environment(\.dismiss) private var dismiss
    @State private var provisioner = ClusterProvisioner(runner: SystemProvisionRunner())
    @State private var currentStep: ProvisionStep = .site
    @State private var selectedEntryID: ClusterConnectionStore.ServerEntry.ID?
    @State private var showingNewSiteEditor = false
    @State private var editingEntry: SiteEditTarget?
    @State private var stepError: String?

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                stepsRail
                    .frame(width: 190)
                Divider()
                detailPane
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            }
            Divider()
            footer
        }
        .frame(minWidth: 880, idealWidth: 940, minHeight: 620, idealHeight: 700)
        .onAppear { installConnectHandler() }
        .task(id: authPollKey) { await pollAuthentication() }
        .sheet(isPresented: $showingNewSiteEditor) {
            ClusterSiteEditor(
                cluster: cluster, entryID: nil, seedProfile: .genericSlurm,
                onSaved: { entry in
                    selectedEntryID = entry.id
                    provisioner.selectSite(entry.resolvedSite)
                })
        }
        .sheet(item: $editingEntry) { target in
            ClusterSiteEditor(
                cluster: cluster, entryID: target.id,
                onSaved: { entry in
                    provisioner.selectSite(entry.resolvedSite)
                })
        }
    }

    // MARK: Steps rail

    private var stepsRail: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Set Up Cluster")
                .font(.headline)
                .padding(.bottom, 6)
            ForEach(ProvisionStep.allCases) { step in
                Button {
                    currentStep = step
                } label: {
                    HStack(spacing: 6) {
                        statusIcon(for: provisioner.record(for: step).status)
                        Text(step.title)
                            .fontWeight(step == currentStep ? .semibold : .regular)
                        Spacer(minLength: 0)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .padding(.vertical, 3)
                .padding(.horizontal, 6)
                .background(
                    step == currentStep
                        ? AnyShapeStyle(.selection) : AnyShapeStyle(.clear),
                    in: RoundedRectangle(cornerRadius: 5))
            }
            Spacer()
            Text("The CLI path is the source of truth; this wizard just drives it. "
                + "Duo/interactive auth always happens in your own Terminal.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(12)
    }

    @ViewBuilder
    private func statusIcon(for status: ProvisionStepStatus) -> some View {
        switch status {
        case .pending:
            Image(systemName: "circle").foregroundStyle(.secondary)
        case .running:
            Image(systemName: "circle.dotted").foregroundStyle(.orange)
        case .awaitingConfirmation:
            Image(systemName: "questionmark.circle").foregroundStyle(.orange)
        case .succeeded:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .failed:
            Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
        case .skipped:
            Image(systemName: "arrow.right.to.line.circle").foregroundStyle(.orange)
        }
    }

    // MARK: Detail pane

    @ViewBuilder
    private var detailPane: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 10) {
                Text(currentStep.title)
                    .font(.title3.bold())
                statusLine(for: currentStep)
                switch currentStep {
                case .site: siteStep
                case .authenticate: authenticateStep
                case .pushCode: pushStep
                case .bootstrap: bootstrapStep
                case .validate: validateStep
                case .controllerJob: controllerStep
                case .connect: connectStep
                }
                if let stepError {
                    Label(stepError, systemImage: "exclamationmark.triangle")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
                transcriptView(for: currentStep)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder
    private func statusLine(for step: ProvisionStep) -> some View {
        let status = provisioner.record(for: step).status
        switch status {
        case .pending:
            EmptyView()
        case .running:
            Label("running…", systemImage: "circle.dotted")
                .font(.caption).foregroundStyle(.orange)
        case .awaitingConfirmation(let note):
            Label(note, systemImage: "questionmark.circle")
                .font(.caption).foregroundStyle(.orange)
        case .succeeded(let note):
            Label(note, systemImage: "checkmark.circle.fill")
                .font(.caption).foregroundStyle(.green)
        case .failed(let note):
            Label(note, systemImage: "xmark.circle.fill")
                .font(.caption).foregroundStyle(.red)
                .textSelection(.enabled)
        case .skipped(let reason):
            Label("SKIPPED — \(reason)", systemImage: "arrow.right.to.line.circle")
                .font(.caption).foregroundStyle(.orange)
        }
    }

    @ViewBuilder
    private func transcriptView(for step: ProvisionStep) -> some View {
        let transcript = provisioner.record(for: step).transcript
        if !transcript.isEmpty {
            VStack(alignment: .leading, spacing: 4) {
                Text("Transcript")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ScrollView {
                    Text(transcript.joined(separator: "\n"))
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(6)
                }
                .frame(minHeight: 80, maxHeight: 220)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
            }
        }
    }

    // MARK: Step 1 — Site

    private var siteStep: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Pick the cluster site to provision — site facts (partitions, GPU "
                + "vocabulary, storage roots, purge policy) are data on the profile.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Picker("Site", selection: $selectedEntryID) {
                Text("none selected").tag(ClusterConnectionStore.ServerEntry.ID?.none)
                ForEach(cluster.servers) { server in
                    Text("\(server.displayName) (\(server.resolvedSite.isSSHTransport ? "ssh" : "direct"))")
                        .tag(Optional(server.id))
                }
            }
            .onChange(of: selectedEntryID) { _, newValue in
                if let newValue, let entry = cluster.server(id: newValue) {
                    provisioner.selectSite(entry.resolvedSite)
                }
            }
            HStack {
                ForEach(cluster.missingPresets, id: \.name) { preset in
                    Button("Add \(preset.name)") {
                        let entry = cluster.addPreset(preset)
                        selectedEntryID = entry.id
                        provisioner.selectSite(entry.resolvedSite)
                    }
                    .controlSize(.small)
                }
            }
            HStack {
                Button("New Site…") { showingNewSiteEditor = true }
                    .controlSize(.small)
                Button("Edit Selected…") {
                    editingEntry = selectedEntryID.map { SiteEditTarget(id: $0) }
                }
                .controlSize(.small)
                .disabled(selectedEntryID == nil)
            }
            if let site = provisioner.site {
                // WP5 §3.3: the same preview the site editor and
                // `steerlab-cli cluster preview` show — the complete generated
                // environment and scheduler commands, readable before the
                // wizard provisions anything. Panes start collapsed so step 1
                // stays a step; the topology line is the wizard's own framing.
                GroupBox {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("topology: \(site.topology.rawValue) — "
                            + SiteEditorModel.topologyExplanation(site.topology))
                            .font(.caption)
                            .fixedSize(horizontal: false, vertical: true)
                        ClusterSitePreviewPanes(
                            preview: ClusterSitePreview(site), paneHeight: 160,
                            expandsEnvironment: false)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }

    // MARK: Step 2 — Authenticate

    private var authenticateStep: some View {
        VStack(alignment: .leading, spacing: 8) {
            if provisioner.site?.isSSHTransport != true {
                Text("Direct transport — nothing to authenticate. Continue.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("One interactive login (password / Duo) opens an SSH ControlMaster "
                    + "that persists 8 hours. The app never sees credentials — the login "
                    + "happens in your own Terminal, and this step polls until the master "
                    + "is alive.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if let site = provisioner.site,
                    let command = ClusterTunnel.authenticationCommand(for: site)
                {
                    Text(command)
                        .font(.caption.monospaced())
                        .textSelection(.enabled)
                        .padding(6)
                        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
                }
                Button("Open Terminal") {
                    tunnel.configure(site: provisioner.site)
                    if !tunnel.openAuthTerminal() {
                        stepError = "could not open Terminal — run the command above yourself"
                    }
                }
                .help("opens Terminal with the prepared ssh command; complete Duo there")
            }
        }
    }

    private var authPollKey: String {
        "\(currentStep.rawValue)|\(provisioner.site?.name ?? "")"
    }

    /// Poll `ssh -O check` every 2 s while the Authenticate step is showing;
    /// stops on success, step change, or dismissal (task cancellation).
    private func pollAuthentication() async {
        guard currentStep == .authenticate, provisioner.site?.isSSHTransport == true else {
            return
        }
        while !Task.isCancelled {
            let alive = await provisioner.runAuthenticateCheck()
            if alive { return }
            try? await Task.sleep(for: .seconds(2))
        }
    }

    // MARK: Step 3 — Push code

    private var pushStep: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Rsync a compact server bundle over the shared ControlMaster connection. "
                + "This sends Server/ and parity fixtures only; the macOS app, papers, docs, "
                + "models, workspaces, local runs, and build products stay on this Mac.")
                .font(.caption)
                .foregroundStyle(.secondary)
            @Bindable var provisioner = provisioner
            TextField("Local server payload", text: $provisioner.localRepoPath)
                .font(.caption.monospaced())
                .help(
                    "SteerLab checkout (developer mode) or packaged deployment payload "
                        + "from which the server bundle is selected. Payloads carrying a "
                        + "deployment-manifest.json are verified before every push.")
            TextField("Remote server bundle root", text: $provisioner.remoteRepoPath)
                .font(.caption.monospaced())
                .help("Dedicated code path containing Server/ (bootstrap --repo); default ~/steerlab")
            if let preview = provisioner.pushCommandPreview {
                commandPreview(preview)
            }
            Button("Run Push") {
                Task { await provisioner.runPushCode() }
            }
            .disabled(provisioner.record(for: .pushCode).status == .running)
        }
    }

    // MARK: Step 4 — Bootstrap

    private var bootstrapStep: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Creates the Python environment, installs Torch and SteerLab Server, writes "
                + "the canonical environment file, and validates it. On Slurm clusters, run "
                + "this work in a small CPU job rather than on the login host.")
                .font(.caption)
                .foregroundStyle(.secondary)
            @Bindable var provisioner = provisioner

            Picker("Run bootstrap on", selection: $provisioner.bootstrapExecutionTarget) {
                ForEach(BootstrapExecutionTarget.allCases) { target in
                    Text(target.title).tag(target)
                }
            }
            .pickerStyle(.segmented)
            .help("controls where conda and pip actually execute; the dry run remains scheduler-free")

            if provisioner.bootstrapExecutionTarget == .slurmBatch {
                Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 6) {
                    GridRow {
                        Text("CPU partition")
                        TextField("partition", text: $provisioner.bootstrapJobPartition)
                    }
                    GridRow {
                        Text("CPUs")
                        TextField(
                            "CPUs", value: $provisioner.bootstrapJobCPUs,
                            format: .number.grouping(.never))
                    }
                    GridRow {
                        Text("Memory")
                        TextField("memory", text: $provisioner.bootstrapJobMemory)
                    }
                    GridRow {
                        Text("Setup walltime")
                        TextField("HH:MM:SS", text: $provisioner.bootstrapJobWalltime)
                    }
                    GridRow {
                        Text("Queue query")
                        TextField("squeue", text: $provisioner.bootstrapSqueueCommand)
                            .help(
                                "squeue-compatible executable used to follow the setup job; "
                                    + "use the site's compatible wrapper when required")
                    }
                }
                .font(.caption)
                .textFieldStyle(.roundedBorder)
                Text("These resources are only for environment setup. Model jobs continue to "
                    + "use the GPU partition and gres recorded in the site profile.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                Label(
                    "Advanced: bootstrap.sh will refuse a protected login node. Use this only "
                        + "for a policy-safe workstation, transfer host, or existing allocation.",
                    systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }

            GroupBox("Effective configuration") {
                VStack(alignment: .leading, spacing: 5) {
                    bootstrapConfigurationRow("Server bundle", provisioner.remoteRepoPath)
                    bootstrapConfigurationRow(
                        "Environment",
                        provisioner.envPrefix.isEmpty
                            ? "$HOME/envs/steerlab (default)" : provisioner.envPrefix)
                    bootstrapConfigurationRow(
                        "Python",
                        provisioner.pythonVersion.isEmpty ? "3.12 (default)" : provisioner.pythonVersion)
                    bootstrapConfigurationRow(
                        "Workspace", bootstrapSiteValue("workspace", fallback: "not configured"))
                    bootstrapConfigurationRow(
                        "HF cache", bootstrapSiteValue("hfCache", fallback: "not configured"))
                    if let site = provisioner.site, case .slurm(let slurm) = site.scheduler {
                        bootstrapConfigurationRow(
                            "Future model jobs",
                            [
                                slurm.resolvedDefaultPartition,
                                slurm.defaultGres,
                                ClusterProvisioner.bootstrapWalltime(for: slurm),
                            ].compactMap { $0 }.joined(separator: " · "))
                        bootstrapConfigurationRow("Account", slurm.account ?? "site default")
                    }
                }
                .font(.caption)
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack {
                TextField("Environment prefix (optional)", text: $provisioner.envPrefix)
                TextField("Python version", text: $provisioner.pythonVersion)
                    .frame(width: 130)
                Button("Edit Site Settings…") {
                    editingEntry = selectedEntryID.map { SiteEditTarget(id: $0) }
                }
                .disabled(selectedEntryID == nil)
            }
            .textFieldStyle(.roundedBorder)

            ForEach(provisioner.bootstrapConfigurationErrors, id: \.self) { issue in
                Label(issue, systemImage: "xmark.octagon.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
            ForEach(provisioner.bootstrapConfigurationWarnings, id: \.self) { issue in
                Label(issue, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }

            HStack {
                Toggle("Rewrite existing environment file", isOn: $provisioner.bootstrapForce)
                Toggle("Submit GPU hello job afterward", isOn: $provisioner.bootstrapHello)
            }
            .font(.caption)
            if let preview = provisioner.bootstrapCommandPreview {
                commandPreview(preview)
            }
            HStack {
                Button("Run Dry-Run") {
                    Task { await provisioner.runBootstrap(dryRun: true) }
                }
                .disabled(
                    !provisioner.bootstrapConfigurationErrors.isEmpty
                        || provisioner.record(for: .bootstrap).status == .running)
                Button(
                    provisioner.bootstrapExecutionTarget == .slurmBatch
                        ? "Submit Bootstrap Job" : "Run on SSH Host"
                ) {
                    Task { await provisioner.runBootstrap(dryRun: false) }
                }
                .disabled(
                    !provisioner.realBootstrapUnlocked
                        || provisioner.record(for: .bootstrap).status == .running)
                .help(
                    provisioner.realBootstrapUnlocked
                        ? (provisioner.bootstrapExecutionTarget == .slurmBatch
                            ? "submits the reviewed plan as a CPU Slurm job and streams its log"
                            : "runs the reviewed plan on the current SSH host")
                        : "locked until a dry-run with the current settings completes")
            }
            if let report = provisioner.realReport ?? provisioner.dryRunReport {
                reportRows(
                    report,
                    isPlan: provisioner.realReport == nil,
                    transcript: provisioner.record(for: .bootstrap).transcript)
            }
        }
    }

    private func bootstrapSiteValue(_ role: String, fallback: String) -> String {
        let value = provisioner.site?.constraints.storageRoots[role]?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value, !value.isEmpty else { return fallback }
        return value
    }

    private func bootstrapConfigurationRow(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(label)
                .foregroundStyle(.secondary)
                .frame(width: 112, alignment: .trailing)
            Text(value.isEmpty ? "not configured" : value)
                .font(.caption.monospaced())
                .textSelection(.enabled)
            Spacer(minLength: 0)
        }
    }

    @ViewBuilder
    private func reportRows(_ report: BootstrapReport, isPlan: Bool, transcript: [String])
        -> some View
    {
        GroupBox(isPlan ? "Dry-run plan" : "Bootstrap report") {
            VStack(alignment: .leading, spacing: 3) {
                ForEach(report.orderedSteps, id: \.name) { entry in
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 6) {
                            reportStatusIcon(entry.status)
                            Text(entry.name)
                                .font(.caption.monospaced())
                            Text(entry.status)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        if entry.status == "failed" {
                            let detail = BootstrapReport.failureDetail(
                                forStep: entry.name, inTranscript: transcript)
                            if !detail.isEmpty {
                                // The step's stderr tail, verbatim.
                                Text(detail.joined(separator: "\n"))
                                    .font(.caption2.monospaced())
                                    .foregroundStyle(.red)
                                    .textSelection(.enabled)
                                    .padding(.leading, 20)
                            }
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder
    private func reportStatusIcon(_ status: String) -> some View {
        switch status {
        case "ok":
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case "failed":
            Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
        case "skipped":
            Image(systemName: "arrow.right.to.line.circle").foregroundStyle(.orange)
        case "planned":
            Image(systemName: "circle.dashed").foregroundStyle(.blue)
        default:
            Image(systemName: "circle").foregroundStyle(.secondary)
        }
    }

    // MARK: Step 5 — Validate

    private var validateStep: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Sources \(provisioner.bootstrapEnvFile) and runs "
                + "`steerlab-server profile validate` remotely — the same check "
                + "bootstrap ran, standalone.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button("Run Validate") {
                Task { await provisioner.runValidate() }
            }
            .disabled(provisioner.record(for: .validate).status == .running)
            if !provisioner.validateLines.isEmpty {
                GroupBox("Checks") {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(provisioner.validateLines) { line in
                            Text(line.text)
                                .font(.caption.monospaced())
                                .foregroundStyle(validateColor(line.kind))
                                .textSelection(.enabled)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }

    private func validateColor(_ kind: ProfileValidateLine.Kind) -> Color {
        switch kind {
        case .ok: .green
        case .warn: .orange
        case .fail: .red
        case .other: .secondary
        }
    }

    // MARK: Step 6 — Controller job

    private var controllerStep: some View {
        VStack(alignment: .leading, spacing: 8) {
            if provisioner.site?.topology == .daemonInJob {
                Text("Renders the controller template with this site's partition, "
                    + "walltime, account, and python, submits it with sbatch, then waits "
                    + "for the job to publish its node in serverd.host — the tunnel "
                    + "forwards there next.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button("Submit Controller Job") {
                    Task { await provisioner.runControllerJob() }
                }
                .disabled(provisioner.record(for: .controllerJob).status == .running)
                if let jobID = provisioner.controllerJobID {
                    LabeledContent("Job") { Text(jobID).font(.caption.monospaced()) }
                        .font(.caption)
                }
                if let host = provisioner.daemonHost {
                    LabeledContent("Node") { Text(host).font(.caption.monospaced()) }
                        .font(.caption)
                }
            } else {
                Text("This site's topology is "
                    + "\(provisioner.site?.topology.rawValue ?? "unset") — no controller "
                    + "job is needed. The step stamps itself skipped.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .task {
                        if case .pending = provisioner.record(for: .controllerJob).status {
                            await provisioner.runControllerJob()  // stamps the loud skip
                        }
                    }
            }
        }
    }

    // MARK: Step 7 — Connect + register

    private var connectStep: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Makes the site the active substrate, opens the tunnel, and fetches "
                + "/api/capabilities through the existing client.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button("Connect") {
                Task { await provisioner.runConnect() }
            }
            .disabled(provisioner.record(for: .connect).status == .running)
            GroupBox("Summary") {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(provisioner.summaryLines, id: \.self) { line in
                        Text(line)
                            .font(.caption.monospaced())
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            if case .succeeded = provisioner.record(for: .connect).status {
                Label(
                    "Done — the Cluster health card on the Home dashboard now shows "
                        + "quota, purge risk, cache freshness, and maintenance for this site.",
                    systemImage: "checkmark.seal")
                    .font(.caption)
                    .foregroundStyle(.green)
                Button("Close") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
        }
    }

    // MARK: Shared bits

    private func commandPreview(_ command: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("Command")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(command)
                .font(.caption.monospaced())
                .textSelection(.enabled)
                .padding(6)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
        }
    }

    private var footer: some View {
        HStack {
            Button("Cancel") { dismiss() }
                .keyboardShortcut(.cancelAction)
            Spacer()
            if currentStep != .site, currentStep != .connect {
                Button("Skip Step") {
                    provisioner.skip(currentStep)
                    advance()
                }
                .help("stamps the step SKIPPED in the summary — loud, never silent")
            }
            Button("Back") { retreat() }
                .disabled(currentStep == ProvisionStep.allCases.first)
            Button("Continue") { advance() }
                .disabled(!canAdvance)
        }
        .padding(12)
    }

    private var canAdvance: Bool {
        guard currentStep != ProvisionStep.allCases.last else { return false }
        if currentStep == .site { return provisioner.site != nil }
        return true
    }

    private func advance() {
        stepError = nil
        let all = ProvisionStep.allCases
        guard let index = all.firstIndex(of: currentStep), index + 1 < all.count else { return }
        currentStep = all[index + 1]
    }

    private func retreat() {
        stepError = nil
        let all = ProvisionStep.allCases
        guard let index = all.firstIndex(of: currentStep), index > 0 else { return }
        currentStep = all[index - 1]
    }

    /// Step 7 glue: the one thing the provisioner cannot do itself.
    private func installConnectHandler() {
        guard provisioner.connectHandler == nil else { return }
        provisioner.connectHandler = { @MainActor in
            guard let selectedEntryID, let entry = cluster.server(id: selectedEntryID) else {
                throw WizardGlueError.noSiteSelected
            }
            cluster.activeWorkspace = .server(entry.id)
            let site = entry.resolvedSite
            tunnel.configure(site: site)
            if site.isSSHTransport {
                // bootstrap.sh creates this token with mode 0600. Import it
                // over the already-authenticated ControlMaster directly into
                // Keychain; never put the secret in the setup transcript.
                guard let token = await tunnel.readRemoteTextFile("~/.steerlab-token") else {
                    throw WizardGlueError.connect(
                        "could not import ~/.steerlab-token — complete Authenticate, "
                            + "then retry Connect + register")
                }
                cluster.setStoredToken(token, for: entry)
            }
            // ChatService is the single connection coordinator and opens the
            // tunnel exactly once. Opening it here as well used to create two
            // multiplexed forwards for one button press.
            await service.connectCluster()
            guard let capabilities = cluster.capabilities else {
                throw WizardGlueError.connect(cluster.status ?? "no capabilities answer")
            }
            return ClusterProvisioner.capabilitySummary(capabilities)
        }
    }
}

/// App-glue failures for the wizard's connect step (rendered verbatim).
private enum WizardGlueError: Error, LocalizedError {
    case noSiteSelected
    case tunnel(String)
    case connect(String)

    var errorDescription: String? {
        switch self {
        case .noSiteSelected: "no site selected in step 1"
        case .tunnel(let state): "tunnel did not come up: \(state)"
        case .connect(let status): "server did not answer capabilities: \(status)"
        }
    }
}
