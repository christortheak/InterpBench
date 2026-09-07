import AppKit
import ExperimentKit
import SwiftUI
import UniformTypeIdentifiers

/// WS1 connection dot (turnkey-cluster plan): one glanceable circle for the
/// active site's transport — green = up (tunnel live / direct server
/// connected), grey = idle or Local-only, amber = authenticate/in-flight or
/// the local Python server starting, red = degraded. Direct-transport state
/// comes from `ClusterConnectionStore.connectionPhase` (in-flight flag,
/// capabilities, last connect failure), never from the shared `status`
/// sentence. The menu carries the connection lifecycle: authenticate
/// (Duo happens in Terminal — the app never touches credentials), connect/
/// disconnect, a site picker, one-click preset adds, and site JSON
/// import/export. The view is glue only: every decision lives in
/// `ClusterConnectionStore` (registry, tokens) or `ClusterTunnel` (SSH
/// lifecycle).
struct ClusterConnectionDot: View {
    @Bindable var cluster: ClusterConnectionStore
    var tunnel: ClusterTunnel
    let service: ChatService
    /// One-click LOCAL Python server (owned by the App so it survives view
    /// churn): start/stop + a status sentence, with output streaming to the
    /// Activity pane. Lives in this menu because it is a connection concern
    /// — the started server is exactly what "Add Server…" then points at.
    var localServer: LocalServerController
    /// WP3 — the local engine's own provisioning state machine (engine source,
    /// uv, venv, serve, acceptance). Owned by the App beside `localServer`,
    /// because a materialization or a 2 GB wheel install must survive toolbar
    /// view churn. It PROVISIONS; `localServer` still owns the running
    /// server's lifecycle once there is one.
    var localEngine: LocalEngineProvisioner

    @State private var showingEngineSetup = false
    @State private var showingImporter = false
    @State private var exportDocument: SiteProfileJSONDocument?
    @State private var exportFilename = "cluster-site"
    @State private var importError: String?
    /// A pending import the researcher has to answer for: a login-less ssh
    /// destination, or a site the registry already holds.
    @State private var importConfirmation: SiteImportConfirmation?
    @State private var siteEditTarget: SiteEditTarget?
    @State private var hfTokenTarget: SiteEditTarget?
    @State private var showingSetupWizard = false
    /// One auto-connect attempt per running episode of the local server —
    /// reset when it stops, so a restart connects again.
    @State private var localServerAutoConnectAttempted = false
    /// Terminal could not be opened for the interactive login: the command is
    /// shown so the researcher can run it themselves (UI audit 2026-09-06).
    @State private var authFailure: AuthTerminalFailure?
    /// Export has its own alert — a failed export under a "Site Import" title
    /// was the audit's finding.
    @State private var exportError: String?
    /// Non-nil presents the GPU-session stop confirmation; the message comes
    /// from `GPUSessionStopCheck`, exactly as the toolbar control's does.
    @State private var gpuStopMessage: String?
    @State private var isCheckingGPUJobs = false

    var body: some View {
        Menu {
            connectionSection
            gpuSessionSection
            localServerSection
            Divider()
            sitePickerSection
            Divider()
            siteManagementSection
        } label: {
            // Fresh-Mac finding: a bare coloured circle is unreadable to
            // anyone who has not been told what it means — it looks like
            // decoration, and its four states are indistinguishable to a
            // colour-blind reader. Same one-glance state, now spelled out:
            // a network glyph tinted by state PLUS the state's own word.
            Label {
                Text(connectionTitle)
            } icon: {
                // Only the glyph carries the state tint — a fully coloured
                // toolbar label reads as an alert, and macOS toolbars are
                // monochrome apart from deliberate status indicators.
                Image(systemName: connectionSymbol)
                    .foregroundStyle(dotColor)
            }
        }
        .labelStyle(.titleAndIcon)
        .help("connection — \(titleLine): \(stateLine)")
        .onChange(of: cluster.activeSite, initial: true) { _, newSite in
            tunnel.configure(site: newSite)
        }
        // B3 auto-connect: the moment the local server is reachable (started
        // here, or adopted from an earlier launch), connect to it — the
        // researcher never hand-types 127.0.0.1:8080. Once per running
        // episode; the reset arm re-arms it for a restart.
        .onChange(of: localServer.phase, initial: true) { _, phase in
            if phase == .running {
                guard !localServerAutoConnectAttempted else { return }
                localServerAutoConnectAttempted = true
                autoConnectLocalServer()
            } else {
                localServerAutoConnectAttempted = false
            }
        }
        .task { installHealthProbeIfNeeded() }
        .fileImporter(isPresented: $showingImporter, allowedContentTypes: [.json]) { result in
            handleImport(result)
        }
        .fileExporter(
            isPresented: exporterPresented,
            document: exportDocument,
            contentType: .json,
            defaultFilename: exportFilename
        ) { result in
            exportDocument = nil
            // A write that failed at save time used to be swallowed here.
            // A cancelled save panel is not a failure and says nothing.
            if case .failure(let error) = result,
                (error as? CocoaError)?.code != .userCancelled
            {
                exportError = "could not save the site profile: "
                    + error.localizedDescription
            }
        }
        .alert(
            "Site Import", isPresented: importErrorPresented,
            actions: { Button("OK") { importError = nil } },
            message: { Text(importError ?? "") })
        .alert(
            "Site Export", isPresented: exportErrorPresented,
            actions: { Button("OK") { exportError = nil } },
            message: { Text(exportError ?? "") })
        .alert(
            "Authenticate", isPresented: authFailurePresented,
            presenting: authFailure
        ) { failure in
            if let command = failure.command {
                Button("Copy ssh Command") { Clipboard.copy(command) }
            }
            Button("OK", role: .cancel) { authFailure = nil }
        } message: { failure in
            Text(failure.message)
        }
        .confirmationDialog(
            "Stop the GPU session on \(cluster.substrateLabel)?",
            isPresented: gpuStopPresented, titleVisibility: .visible
        ) {
            Button("Stop session", role: .destructive) {
                Task { await cluster.gpuSession.stop() }
            }
            Button("Keep running", role: .cancel) {}
        } message: {
            Text(gpuStopMessage ?? "")
        }
        .alert(
            "Site Import", isPresented: importConfirmationPresented,
            presenting: importConfirmation
        ) { pending in
            Button(pending.proceedTitle) {
                importConfirmation = nil
                performImport(
                    pending.data, force: pending.force,
                    acknowledgingWarnings: pending.acknowledgingWarnings)
            }
            Button("Cancel", role: .cancel) { importConfirmation = nil }
        } message: { pending in
            Text(pending.message)
        }
        // Registry edits made elsewhere — by `steerlab-cli`, by a git pull on
        // another machine's work, or in an editor — are picked up when the
        // researcher comes back to the app and when this menu opens. A file
        // watcher is deliberately not here: a pull that rewrites the directory
        // must not race a panel someone is typing into.
        .onReceive(
            NotificationCenter.default.publisher(
                for: NSApplication.didBecomeActiveNotification)
        ) { _ in
            cluster.reloadSitesFromDisk()
        }
        .sheet(item: $siteEditTarget) { target in
            // Full WS1 site editor (transport/topology/scheduler/constraints
            // + live env preview) — logic in SiteEditorModel (ExperimentKit).
            ClusterSiteEditor(cluster: cluster, entryID: target.id)
        }
        .sheet(item: $hfTokenTarget) { target in
            HFTokenInstallSheet(cluster: cluster, tunnel: tunnel, entryID: target.id)
        }
        .sheet(isPresented: $showingEngineSetup) {
            LocalEngineSetupSheet(
                engine: localEngine, service: service, server: localServer)
        }
        .sheet(isPresented: $showingSetupWizard) {
            // WS5 wizard — a veneer over ClusterProvisioner (ExperimentKit).
            ClusterSetupWizard(cluster: cluster, tunnel: tunnel, service: service)
        }
    }

    // MARK: Menu sections
    //
    // Extracted from `body` so each stays inside the type-checker's comfort
    // zone; the menu is one long list of small controls.

    @ViewBuilder
    private var connectionSection: some View {
        Section(titleLine) {
            Text(stateLine)
            if cluster.activeSite?.isSSHTransport == true {
                Button("Authenticate…") { authenticate() }
                    .help(
                        "opens Terminal with this site's ssh command so you "
                            + "can complete the interactive login (password / "
                            + "Duo) — the app never sees the credentials, and "
                            + "the resulting connection is reused for 8 hours")
            }
            if cluster.activeWorkspace != .local {
                if showsDisconnect {
                    Button("Disconnect") { disconnect() }
                        .help(
                            "closes the SSH tunnel to this site; queued and "
                                + "running cluster jobs keep going, and "
                                + "Connect reopens it")
                } else {
                    Button("Connect") { connect() }
                        .help(
                            "opens the transport if this site needs one and "
                                + "asks the server for its capabilities — the "
                                + "handshake every server-side panel waits on")
                }
            }
        }
    }

    // GPU session at a glance (plan §2.7): one status line, Stop when
    // active. Capability-gated; the dot's own color stays the CONTROLLER
    // connection — a session ending never reads as a disconnect here.
    @ViewBuilder
    private var gpuSessionSection: some View {
        if cluster.activeWorkspace != .local,
            cluster.capabilities?.supportsGPUSession == true
        {
            Section("GPU Session") {
                Text(gpuSessionLine)
                if cluster.gpuSession.isActive {
                    Button(isCheckingGPUJobs ? "Checking Jobs…" : "Stop GPU Session") {
                        requestGPUSessionStop()
                    }
                    .disabled(isCheckingGPUJobs)
                    .help(
                        "ends the GPU session — its worker job is cancelled "
                            + "and the queue slot is lost; asks first, and "
                            + "says whether server jobs are still running. "
                            + "The controller connection stays up")
                }
            }
        }
    }

    // One-click local Python server: no terminal, no venv incantation, no
    // cwd hazard (the script serves the current workspace via an explicit
    // --root). Output streams to the Activity pane; a busy port fails with a
    // sentence, not a traceback.
    @ViewBuilder
    private var localServerSection: some View {
        Section("Local Python Server") {
            Text(localServer.statusLine)
            // WP3: the setup affordance sits ABOVE the start/stop controls
            // and answers the question those controls used to fail at —
            // "there is no Python environment here yet". A step in flight is
            // a status LINE plus the progress button; a disabled button whose
            // title promised an action was the audit's finding.
            Text("Local engine: " + localEngineLine)
            if case .running = localEngine.phase {
                Button("Show Setup Progress…") { showingEngineSetup = true }
                    .help(
                        "opens the setup sheet on the step now running — it "
                            + "carries the step-by-step report and Cancel")
            } else {
                Button(localEngineButtonTitle) { showingEngineSetup = true }
                    .help(
                        "provisions the local Python engine end to end: "
                            + "engine source (a code checkout, or the "
                            + "bundled engine copied to ~/SteerLab/Engine), "
                            + "a pinned sha256-verified uv and a managed "
                            + "CPython \(PinnedCPython.minor), a venv "
                            + "installed from the committed platform lock, "
                            + "the loopback server, and site qualify. Every "
                            + "step checks before it acts, so re-running "
                            + "continues rather than restarting")
            }
            switch localServer.phase {
            case .idle:
                Button("Start Local Python Server") {
                    localServer.start(host: service)
                }
                .help(
                    "runs scripts/start-local-server.sh: creates "
                        + "Server/.venv.nosync on first use and installs the "
                        + "full workbench incl. LoRA/PDF/Gemma Scope extras "
                        + "(many minutes — progress streams in the Activity "
                        + "pane), then serves the current workspace on "
                        + "127.0.0.1:\(localServer.port) (loopback only). "
                        + "Once running, the app connects to it "
                        + "automatically")
            case .starting, .running:
                Button("Stop Local Python Server") { localServer.stop() }
                    .help(
                        "terminates the local server process on "
                            + "127.0.0.1:\(localServer.port); files it has "
                            + "already written to the workspace stay, and "
                            + "anything it was computing is lost")
            case .stopping:
                Text("stopping…")
            }
        }
    }

    @ViewBuilder
    private var sitePickerSection: some View {
        Picker("Site", selection: siteSelection) {
            Text("Local (MLX)").tag(ClusterConnectionStore.Workspace.local)
            ForEach(cluster.servers) { server in
                Text(server.displayName).tag(ClusterConnectionStore.Workspace.server(server.id))
            }
        }
        .pickerStyle(.inline)
        .help(
            "which compute this workspace talks to — Local (MLX) runs in this "
                + "app, a site runs on its server; picking a site connects to "
                + "it. Same selection as the toolbar's compute menu")
    }

    @ViewBuilder
    private var siteManagementSection: some View {
        ForEach(cluster.missingPresets, id: \.name) { preset in
            Button("Add \(preset.name) preset…") { addPreset(preset) }
                .help(
                    "adds a ready-made profile for \(preset.name) to your "
                        + "Sites registry and makes it the active site — you "
                        + "still fill in your own login and storage roots")
        }
        Button("Import Site JSON…") { showingImporter = true }
            .help(
                "copies a profile file into your Sites registry "
                    + "(\(HomeLayout.clusterSitesDirectory.path)) through the "
                    + "same checks the command line uses; credentials stay in "
                    + "this Mac's Keychain")
        if let active = cluster.activeServer {
            Button("Export “\(active.displayName)”…") { export(active) }
                .help(
                    "writes this site's profile as JSON to share it — no "
                        + "token or password is ever written into the file")
        }
        Divider()
        if let active = cluster.activeServer {
            Button("Edit Site…") {
                siteEditTarget = SiteEditTarget(id: active.id)
            }
            .help(
                "opens the full profile for this site — transport, scheduler, "
                    + "storage roots, policy — with a live preview of the "
                    + "environment and job headers it will generate")
            if cluster.activeSite?.isSSHTransport == true {
                Button("Install HF Token…") {
                    hfTokenTarget = SiteEditTarget(id: active.id)
                }
                .help(
                    "writes a Hugging Face read token into this site's model "
                        + "cache so gated model installs authenticate; the "
                        + "value travels over the authenticated connection "
                        + "and is kept in this Mac's Keychain")
            }
        }
        Button("Set Up Cluster…") { showingSetupWizard = true }
            .help(
                "opens the step-by-step wizard — pick a site, authenticate, "
                    + "push the server bundle, bootstrap its Python "
                    + "environment, validate, and connect")
    }

    // MARK: Labels

    private var titleLine: String {
        cluster.activeSite?.name ?? "Local (MLX)"
    }

    /// The engine's one line in the menu. Deliberately short — the sheet is
    /// where the step-by-step report lives.
    private var localEngineLine: String {
        switch localEngine.phase {
        case .unknown: return "not checked yet"
        case .planning: return "checking…"
        case .running(let step): return "setting up — \(step.title)"
        case .needsSetup(let count):
            return "not set up (\(count) step\(count == 1 ? "" : "s") to run)"
        case .ready: return "ready"
        case .cancelled: return "setup cancelled — re-running continues"
        case .failed(let reason): return "needs attention — \(reason)"
        }
    }

    private var localEngineButtonTitle: String {
        switch localEngine.phase {
        case .ready: return "Local Engine Details…"
        case .failed, .cancelled: return "Resume Local Engine Setup…"
        default: return "Set Up Local Engine…"
        }
    }

    /// "GPU session: Idle 18m · 1h 42m walltime left" — state + walltime,
    /// straight from the controller's tested display mapping.
    private var gpuSessionLine: String {
        var line = "GPU session: \(cluster.gpuSession.displayState.label)"
        if let walltime = cluster.gpuSession.remainingWalltimeDescription {
            line += " · \(walltime)"
        }
        return line
    }

    private var stateLine: String {
        guard case .server = cluster.activeWorkspace, let site = cluster.activeSite else {
            if localServer.phase != .idle {
                return "local Python server — \(localServer.statusLine)"
            }
            return "no cluster site active"
        }
        if site.isSSHTransport { return tunnel.state.displayDescription }
        return cluster.status ?? "not connected"
    }

    /// One word for the transport's state, next to the glyph. Deliberately
    /// short — the sentence is in the menu and the tooltip.
    private var connectionTitle: String {
        switch connectionState {
        case .connected: return "Connected"
        case .authenticate: return "Authenticate"
        case .working: return "Connecting"
        case .degraded: return "Degraded"
        case .offline: return "Not Connected"
        case .local: return "Local"
        case .localServerStarting: return "Starting Server"
        case .localServerStopping: return "Stopping Server"
        case .localServerReady: return "Server Ready"
        }
    }

    private var connectionSymbol: String {
        switch connectionState {
        case .connected: return "network"
        case .authenticate: return "lock"
        case .working: return "network"
        case .degraded: return "exclamationmark.triangle.fill"
        case .offline: return "network.slash"
        case .local: return "network.slash"
        case .localServerStarting, .localServerStopping: return "hourglass"
        case .localServerReady: return "network"
        }
    }

    /// The transport's state, normalized across the two transports so the
    /// label, the glyph, and the colour cannot drift apart.
    private enum ConnectionState {
        case connected, authenticate, working, degraded, offline
        /// Local (MLX) workspace with no local server in flight.
        case local
        /// The one-click local Python server's own phases, which used to be
        /// invisible on the dot (the audit's item (a)): a venv build can run
        /// for many minutes, and "Local" grey said nothing about it.
        case localServerStarting, localServerStopping, localServerReady
    }

    private var connectionState: ConnectionState {
        guard case .server = cluster.activeWorkspace, let site = cluster.activeSite else {
            guard cluster.activeWorkspace == .local else { return .offline }
            switch localServer.phase {
            case .starting: return .localServerStarting
            case .stopping: return .localServerStopping
            case .running: return .localServerReady
            case .idle: return .local
            }
        }
        if site.isSSHTransport {
            switch tunnel.state {
            case .up: return .connected
            case .needsAuth: return .authenticate
            case .opening: return .working
            case .degraded: return .degraded
            case .idle, .closed: return .offline
            }
        }
        // Direct transport: the phase is three facts (in-flight, capabilities,
        // last connect failure) held by the store — never the shared free-text
        // `status` line, which model installs, workspace switches and agent
        // sync also write (UI audit 2026-09-06, headline 6).
        switch cluster.connectionPhase {
        case .connected: return .connected
        case .connecting: return .working
        case .failed: return .degraded
        case .idle: return .offline
        }
    }

    /// Same four colours as before, now derived from the single normalized
    /// state so the tint can never contradict the word beside it.
    private var dotColor: Color {
        switch connectionState {
        case .connected: return .green
        // Actionable, not broken: an idle local server the researcher can
        // switch to reads the same as "Authenticate".
        case .authenticate, .working, .localServerStarting, .localServerStopping,
            .localServerReady:
            return .orange
        case .degraded: return .red
        case .offline, .local: return .secondary  // grey: Local-only or not connected
        }
    }

    private var showsDisconnect: Bool {
        guard cluster.activeSite?.isSSHTransport == true else { return false }
        switch tunnel.state {
        case .up, .opening, .degraded: return true
        case .idle, .needsAuth, .closed: return false
        }
    }

    // MARK: Actions (thin — store/tunnel own the logic)

    private var siteSelection: Binding<ClusterConnectionStore.Workspace> {
        Binding(
            get: { cluster.activeWorkspace },
            set: { workspace in
                cluster.activeWorkspace = workspace
                if case .server = workspace { connect() }
            })
    }

    private func connect() {
        // ChatService is the single connection coordinator. Keeping tunnel
        // setup there means this button, the Compute picker, the site editor,
        // and the setup wizard all perform the same observable operation.
        Task { await service.connectCluster() }
    }

    /// Open Terminal for the interactive login. `openAuthTerminal` answers
    /// whether it could; when it could not, the click used to do nothing
    /// visible — now the researcher gets the reason AND the command, the way
    /// the setup wizard's Authenticate step already did.
    private func authenticate() {
        guard !tunnel.openAuthTerminal() else { return }
        let command = cluster.activeSite.flatMap(ClusterTunnel.authenticationCommand(for:))
        authFailure = AuthTerminalFailure(
            command: command,
            message: command == nil
                ? "This site has no SSH destination to log in to — set “SSH "
                    + "user@host” in Edit Site… first."
                : "Could not open Terminal. Run this command in your own "
                    + "Terminal, finish the login there, then choose "
                    + "Connect:\n\n\(command ?? "")")
    }

    /// The toolbar control's guarded stop, reached from the menu: the same
    /// `GPUSessionStopCheck` rules decide what the confirmation SAYS, and the
    /// menu always asks — cancelling the worker job loses the queue slot.
    private func requestGPUSessionStop() {
        guard !isCheckingGPUJobs else { return }
        isCheckingGPUJobs = true
        Task {
            var count: Int?
            if let client = cluster.client, let jobs = try? await client.jobs() {
                count = GPUSessionStopCheck.unfinishedJobCount(jobs)
            }
            isCheckingGPUJobs = false
            gpuStopMessage = GPUSessionStopCheck.confirmationMessage(
                unfinishedJobCount: count)
                ?? "The worker job is cancelled and its queue slot is lost; "
                    + "no server job is unfinished. The controller connection "
                    + "stays up."
        }
    }

    /// Connects to the one-click local server without hand-typing its URL:
    /// registers (or reuses) the registry entry for 127.0.0.1:<port>,
    /// activates it, and runs the ONE shared connect flow. Never hijacks a
    /// DIFFERENT active server, and an ADOPTED server never steals the
    /// Local (MLX) workspace at launch — those cases register the entry and
    /// say why in the status line instead. The outcome (connected-or-why-not)
    /// lands in the menu's local-server status line.
    private func autoConnectLocalServer() {
        let hostLabel = "127.0.0.1:\(localServer.port)"
        let urlString = "http://\(hostLabel)"
        let existing = cluster.servers.first { $0.hostLabel == hostLabel }
        if case .server(let activeID) = cluster.activeWorkspace,
            existing?.id != activeID
        {
            _ = existing
                ?? cluster.addServer(name: "Local Python Server", urlString: urlString)
            localServer.noteAutoConnectOutcome(
                "not auto-connected: \(cluster.substrateLabel) is the active "
                    + "site — pick the local server in the Site menu to switch")
            return
        }
        if localServer.wasAdopted, cluster.activeWorkspace == .local {
            _ = existing
                ?? cluster.addServer(name: "Local Python Server", urlString: urlString)
            localServer.noteAutoConnectOutcome(
                "select it in the Site menu to connect")
            return
        }
        let entry = existing
            ?? cluster.addServer(name: "Local Python Server", urlString: urlString)
        cluster.activeWorkspace = .server(entry.id)
        Task {
            await service.connectCluster()
            localServer.noteAutoConnectOutcome(
                cluster.status == "connected"
                    ? "connected"
                    : "auto-connect: \(cluster.status ?? "no connection status")")
        }
    }

    private func disconnect() {
        Task { await tunnel.close() }
    }

    private func addPreset(_ preset: ClusterSiteProfile) {
        let entry = cluster.addPreset(preset)
        cluster.activeWorkspace = .server(entry.id)
        connect()  // SSH presets land on .needsAuth — the actionable state
    }

    private func handleImport(_ result: Result<URL, Error>) {
        switch result {
        case .failure(let error):
            importError = error.localizedDescription
        case .success(let url):
            let scoped = url.startAccessingSecurityScopedResource()
            defer { if scoped { url.stopAccessingSecurityScopedResource() } }
            guard let data = try? Data(contentsOf: url) else {
                importError = "could not read \(url.lastPathComponent)"
                return
            }
            performImport(data)
        }
    }

    /// Copy a profile into the canonical Sites registry.
    ///
    /// Import now MEANS "put this file in `Sites/cluster-sites/`", through the
    /// same repository entry point `steerlab-cli cluster sites import` uses —
    /// so the app runs the same validations, and the two clients cannot
    /// disagree about what a legal site is. Two of those validations are
    /// questions rather than refusals, and each gets its own confirmation:
    /// a login-less ssh destination (legal via `~/.ssh/config`, and also
    /// exactly what an accidental drop looks like) and a site the registry
    /// already holds (never replaced in silence — the registry is a git
    /// repository the researcher syncs).
    private func performImport(
        _ data: Data, force: Bool = false, acknowledgingWarnings: Bool = false
    ) {
        do {
            let entry = try cluster.importSite(
                from: data, force: force,
                acknowledgingWarnings: acknowledgingWarnings)
            cluster.activeWorkspace = .server(entry.id)
            connect()
        } catch let error as ClusterLifecycleError {
            switch error {
            case .sshLoginMissing, .siteFileExists:
                importConfirmation = SiteImportConfirmation(
                    data: data,
                    message: error.errorDescription ?? error.code,
                    proceedTitle: error.code == "siteFileExists"
                        ? "Replace" : "Import Anyway",
                    force: force || error.code == "siteFileExists",
                    acknowledgingWarnings: acknowledgingWarnings
                        || error.code == "sshLoginMissing")
            default:
                importError = "could not import site: \(error.localizedDescription)"
            }
        } catch {
            importError = "could not import site: \(error.localizedDescription)"
        }
    }

    private func export(_ entry: ClusterConnectionStore.ServerEntry) {
        do {
            guard let data = try cluster.exportSite(id: entry.id) else { return }
            let name = entry.resolvedSite.name
            exportFilename = name.isEmpty ? "cluster-site" : Self.sanitizedFilename(name)
            exportDocument = SiteProfileJSONDocument(data: data)
        } catch {
            exportError = "could not export site: \(error.localizedDescription)"
        }
    }

    /// One health probe for the lifetime of the tunnel manager: a
    /// capabilities ping through the store's current client (fetched on the
    /// main actor; the request itself runs off it).
    private func installHealthProbeIfNeeded() {
        guard tunnel.healthProbe == nil else { return }
        let cluster = self.cluster
        tunnel.healthProbe = { @Sendable in
            let client = await MainActor.run { cluster.client }
            guard let client else { return false }
            return (try? await client.capabilities()) != nil
        }
    }

    private static func sanitizedFilename(_ name: String) -> String {
        String(name.lowercased().map { $0.isLetter || $0.isNumber ? $0 : "-" })
    }

    private var exporterPresented: Binding<Bool> {
        Binding(
            get: { exportDocument != nil },
            set: { if !$0 { exportDocument = nil } })
    }

    private var importErrorPresented: Binding<Bool> {
        Binding(
            get: { importError != nil },
            set: { if !$0 { importError = nil } })
    }

    private var importConfirmationPresented: Binding<Bool> {
        Binding(
            get: { importConfirmation != nil },
            set: { if !$0 { importConfirmation = nil } })
    }

    private var exportErrorPresented: Binding<Bool> {
        Binding(
            get: { exportError != nil },
            set: { if !$0 { exportError = nil } })
    }

    private var authFailurePresented: Binding<Bool> {
        Binding(
            get: { authFailure != nil },
            set: { if !$0 { authFailure = nil } })
    }

    private var gpuStopPresented: Binding<Bool> {
        Binding(
            get: { gpuStopMessage != nil },
            set: { if !$0 { gpuStopMessage = nil } })
    }
}

/// "Authenticate…" could not hand the login to Terminal: the sentence to
/// show, and the command to run by hand when there is one.
struct AuthTerminalFailure: Identifiable {
    let id = UUID()
    let command: String?
    let message: String
}

/// One import waiting on the researcher's answer.
struct SiteImportConfirmation: Identifiable {
    let id = UUID()
    let data: Data
    let message: String
    let proceedTitle: String
    let force: Bool
    let acknowledgingWarnings: Bool
}

/// Minimal JSON wrapper for `fileExporter` — the bytes come straight from
/// `ClusterConnectionStore.exportSite` (sorted keys, pretty printed).
struct SiteProfileJSONDocument: FileDocument {
    static let readableContentTypes: [UTType] = [.json]

    var data: Data

    init(data: Data) {
        self.data = data
    }

    init(configuration: ReadConfiguration) throws {
        data = configuration.file.regularFileContents ?? Data()
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: data)
    }
}

/// Paste-once Hugging Face token install: the value is kept in the Mac's
/// Keychain (rotation without re-pasting) and MATERIALIZED as
/// `<hfCache>/token` on the cluster over the authenticated ControlMaster —
/// the hub's native token location, so downloads and gated-repo access need
/// no further plumbing. The secret travels on stdin and is never echoed
/// back: the sheet reports presence, not contents.
struct HFTokenInstallSheet: View {
    @Bindable var cluster: ClusterConnectionStore
    var tunnel: ClusterTunnel
    let entryID: ClusterConnectionStore.ServerEntry.ID

    @Environment(\.dismiss) private var dismiss
    @State private var token = ""
    @State private var isInstalling = false
    @State private var statusMessage: String?
    @State private var statusIsError = false
    /// Presence of the Keychain copy, checked once on appear. The value is
    /// deliberately never read into the field.
    @State private var hasStoredToken = false

    private var entry: ClusterConnectionStore.ServerEntry? {
        cluster.servers.first { $0.id == entryID }
    }

    private var tokenPath: String? {
        let root = (entry?.site?.constraints.storageRoots["hfCache"] ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return root.isEmpty ? nil : root + "/token"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Install Hugging Face Token")
                .font(.headline)
            Text("Create a READ token at huggingface.co → Settings → Access Tokens, "
                + "and accept each gated model's license (e.g. Gemma) with the same "
                + "account. The token is stored in your Mac's Keychain and written to "
                + "the cluster's HF cache, where model installs look for it.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if let tokenPath {
                Text("Destination: \(tokenPath)")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            } else {
                Label("This site has no HF cache storage root — set it in Edit Site… first.",
                    systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            // Presence, never contents (the type's own contract): the stored
            // secret is not read back into the field, so nothing here can
            // re-push a stale token by accident.
            if hasStoredToken {
                Text("A token for this site is already in this Mac's Keychain. "
                    + "Paste a new one to replace it on the cluster.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            SecureField("hf_…", text: $token)
                .textFieldStyle(.roundedBorder)
                .help(
                    "the read token itself — kept in this Mac's Keychain and "
                        + "written to the cluster's model cache; never stored "
                        + "in a site profile or shown again")
            if let statusMessage {
                Label(statusMessage,
                    systemImage: statusIsError ? "xmark.octagon.fill" : "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(statusIsError ? .red : .green)
                    .textSelection(.enabled)
            }
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .help("closes without saving or sending anything")
                Button(isInstalling ? "Installing…" : "Install on Cluster") { install() }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                    .disabled(
                        isInstalling || tokenPath == nil
                            || token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .help(
                        tokenPath == nil
                            ? "this site has no HF cache root, so there is "
                                + "nowhere to write the token — set one in "
                                + "Edit Site… first"
                            : "keeps the token in this Mac's Keychain and "
                                + "writes it to the cluster's model cache, "
                                + "replacing any token already there")
            }
        }
        .padding(16)
        .frame(width: 460)
        .onAppear {
            hasStoredToken = entry.flatMap { cluster.storedHFToken(for: $0) }?.isEmpty == false
        }
    }

    private func install() {
        guard let entry, !isInstalling else { return }
        isInstalling = true
        statusMessage = nil
        let value = token
        Task {
            // A Keychain refusal is reported, not swallowed: the cluster copy
            // may still land, and the researcher needs to know the Mac-side
            // copy (rotation without re-pasting) did not.
            let saved = cluster.setStoredHFToken(value, for: entry)
            let error = await tunnel.installHFToken(value)
            isInstalling = false
            statusIsError = error != nil
            if let error {
                statusMessage = error
            } else {
                hasStoredToken = saved
                statusMessage = "token installed — model installs can now "
                    + "authenticate (gated models also need their license "
                    + "accepted)"
                    + (saved
                        ? ""
                        : ". This Mac's Keychain refused to keep a copy, so "
                            + "you will have to paste the token again next time")
            }
        }
    }
}
