import AppKit
import ExperimentKit
import SwiftUI
import UniformTypeIdentifiers

/// WS1 connection dot (turnkey-cluster plan): one glanceable circle for the
/// active site's transport — green = up (tunnel live / direct server
/// connected), grey = idle or Local-only, amber = authenticate/in-flight,
/// red = degraded. The menu carries the connection lifecycle: authenticate
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
    @State private var siteEditTarget: SiteEditTarget?
    @State private var hfTokenTarget: SiteEditTarget?
    @State private var showingSetupWizard = false
    /// One auto-connect attempt per running episode of the local server —
    /// reset when it stops, so a restart connects again.
    @State private var localServerAutoConnectAttempted = false

    var body: some View {
        Menu {
            Section(titleLine) {
                Text(stateLine)
                if cluster.activeSite?.isSSHTransport == true {
                    Button("Authenticate…") { tunnel.openAuthTerminal() }
                }
                if cluster.activeWorkspace != .local {
                    if showsDisconnect {
                        Button("Disconnect") { disconnect() }
                    } else {
                        Button("Connect") { connect() }
                    }
                }
            }
            // GPU session at a glance (plan §2.7): one status line, Stop when
            // active. Capability-gated; the dot's own color stays the
            // CONTROLLER connection — a session ending never reads as a
            // disconnect here.
            if cluster.activeWorkspace != .local,
                cluster.capabilities?.supportsGPUSession == true
            {
                Section("GPU Session") {
                    Text(gpuSessionLine)
                    if cluster.gpuSession.isActive {
                        Button("Stop GPU Session") {
                            Task { await cluster.gpuSession.stop() }
                        }
                    }
                }
            }
            // One-click local Python server: no terminal, no venv incantation,
            // no cwd hazard (the script serves the current workspace via an
            // explicit --root). Output streams to the Activity pane; a busy
            // port fails with a sentence, not a traceback.
            Section("Local Python Server") {
                Text(localServer.statusLine)
                // WP3: the setup affordance sits ABOVE the start/stop controls
                // and answers the question those controls used to fail at —
                // "there is no Python environment here yet". Three states:
                //   * a step in flight: the named step, disabled, with Cancel
                //     available in the sheet and progress in the Activity pane;
                //   * ready: one line saying so; the existing Start/Stop
                //     controls below are the ones the researcher then uses;
                //   * anything else: the setup entry point.
                Text("Local engine: " + localEngineLine)
                if case .running = localEngine.phase {
                    Button("Setting Up Local Engine…") {}
                        .disabled(true)
                    Button("Show Setup Progress…") { showingEngineSetup = true }
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
                case .stopping:
                    Button("Stop Local Python Server") {}
                        .disabled(true)
                }
            }
            Divider()
            Picker("Site", selection: siteSelection) {
                Text("Local (MLX)").tag(ClusterConnectionStore.Workspace.local)
                ForEach(cluster.servers) { server in
                    Text(server.displayName).tag(ClusterConnectionStore.Workspace.server(server.id))
                }
            }
            .pickerStyle(.inline)
            Divider()
            ForEach(cluster.missingPresets, id: \.name) { preset in
                Button("Add \(preset.name) preset…") { addPreset(preset) }
            }
            Button("Import Site JSON…") { showingImporter = true }
            if let active = cluster.activeServer {
                Button("Export “\(active.displayName)”…") { export(active) }
            }
            Divider()
            if let active = cluster.activeServer {
                Button("Edit Site…") {
                    siteEditTarget = SiteEditTarget(id: active.id)
                }
                if cluster.activeSite?.isSSHTransport == true {
                    Button("Install HF Token…") {
                        hfTokenTarget = SiteEditTarget(id: active.id)
                    }
                }
            }
            Button("Set Up Cluster…") { showingSetupWizard = true }
        } label: {
            Image(systemName: "circle.fill")
                .foregroundStyle(dotColor)
                .imageScale(.small)
        }
        .help("cluster connection — \(titleLine): \(stateLine)")
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
        ) { _ in
            exportDocument = nil
        }
        .alert(
            "Site Import", isPresented: importErrorPresented,
            actions: { Button("OK") { importError = nil } },
            message: { Text(importError ?? "") })
        .sheet(item: $siteEditTarget) { target in
            // Full WS1 site editor (transport/topology/scheduler/constraints
            // + live env preview) — logic in SiteEditorModel (ExperimentKit).
            ClusterSiteEditor(cluster: cluster, entryID: target.id)
        }
        .sheet(item: $hfTokenTarget) { target in
            HFTokenInstallSheet(cluster: cluster, tunnel: tunnel, entryID: target.id)
        }
        .sheet(isPresented: $showingEngineSetup) {
            LocalEngineSetupSheet(engine: localEngine, service: service)
        }
        .sheet(isPresented: $showingSetupWizard) {
            // WS5 wizard — a veneer over ClusterProvisioner (ExperimentKit).
            ClusterSetupWizard(cluster: cluster, tunnel: tunnel, service: service)
        }
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
            return "no cluster site active"
        }
        if site.isSSHTransport { return tunnel.state.displayDescription }
        return cluster.status ?? "not connected"
    }

    private var dotColor: Color {
        guard case .server = cluster.activeWorkspace, let site = cluster.activeSite else {
            return .secondary  // grey: Local-only
        }
        if site.isSSHTransport {
            switch tunnel.state {
            case .up: return .green
            case .needsAuth, .opening: return .orange
            case .degraded: return .red
            case .idle, .closed: return .secondary
            }
        }
        guard let status = cluster.status else { return .secondary }
        if status.contains("failed") || status.contains("invalid") || status.contains("rejected") {
            return .red
        }
        if status.hasSuffix("...") { return .orange }
        return .green
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
            do {
                let data = try Data(contentsOf: url)
                let entry = try cluster.importSite(from: data)
                cluster.activeWorkspace = .server(entry.id)
                connect()
            } catch {
                importError = "could not import site: \(error.localizedDescription)"
            }
        }
    }

    private func export(_ entry: ClusterConnectionStore.ServerEntry) {
        do {
            guard let data = try cluster.exportSite(id: entry.id) else { return }
            let name = entry.resolvedSite.name
            exportFilename = name.isEmpty ? "cluster-site" : Self.sanitizedFilename(name)
            exportDocument = SiteProfileJSONDocument(data: data)
        } catch {
            importError = "could not export site: \(error.localizedDescription)"
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
            SecureField("hf_…", text: $token)
                .textFieldStyle(.roundedBorder)
            if let statusMessage {
                Label(statusMessage,
                    systemImage: statusIsError ? "xmark.octagon.fill" : "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(statusIsError ? .red : .green)
                    .textSelection(.enabled)
            }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button(isInstalling ? "Installing…" : "Install on Cluster") { install() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(
                        isInstalling || tokenPath == nil
                            || token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(16)
        .frame(width: 460)
        .onAppear {
            if let entry, let stored = cluster.storedHFToken(for: entry) {
                token = stored
            }
        }
    }

    private func install() {
        guard let entry else { return }
        isInstalling = true
        statusMessage = nil
        let value = token
        Task {
            cluster.setStoredHFToken(value, for: entry)
            let error = await tunnel.installHFToken(value)
            isInstalling = false
            statusIsError = error != nil
            statusMessage = error ?? "token installed — model installs can now "
                + "authenticate (gated models also need their license accepted)"
        }
    }
}
