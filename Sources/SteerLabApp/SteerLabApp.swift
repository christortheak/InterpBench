import AppKit
import ExperimentKit
import Foundation
import SwiftUI

@main
struct SteerLabApp: App {
    /// Workspace-global substrate state (server registry + active workspace +
    /// connection), created once and injected into ChatService (whose panels
    /// reach it through their host). The toolbar workspace switcher edits it;
    /// every panel just reads it.
    @State private var cluster: ClusterConnectionStore
    /// SSH tunnel lifecycle for the active site (WS1) — one instance so a
    /// live tunnel survives view churn; the toolbar dot drives it.
    @State private var tunnel: ClusterTunnel
    /// Workspace-global availability + freshness index (installed models,
    /// vector artifacts per substrate). One instance, injected alongside the
    /// store — panels read models/artifacts through it, never `/api/state`.
    @State private var catalog: SubstrateCatalog
    @State private var service: ChatService
    /// The DATA workspace (prompts/experiments/runs folder) — distinct from
    /// the Compute substrate switcher. Created before the catalogs so their
    /// first scans already resolve against the persisted workspace choice.
    @State private var workspace: WorkspaceStore
    /// One-click local Python server lifecycle (venv setup + loopback serve
    /// of the current workspace) — owned here so a running server survives
    /// toolbar view churn; the connection dot renders and drives it.
    @State private var localServer = LocalServerController()
    /// WP3 — the local ENGINE provisioner (engine source, pinned uv, managed
    /// CPython, lock-installed venv, serve, acceptance). Owned here for the
    /// same reason as `localServer`: a materialization or a multi-gigabyte
    /// wheel install must survive toolbar view churn. It provisions;
    /// `localServer` keeps owning the running server's lifecycle.
    @State private var localEngine = LocalEngineProvisioner()
    /// Update SIGNPOST — the once-a-day "a newer release exists" check and
    /// its menu item. Never downloads or installs; see UpdateSignpost.swift.
    @State private var updates = UpdateSignpostModel()

    init() {
        // WP2 — claim the seam `CodeResources.releaseModeAsserted` was
        // reserved for. Running from an .app bundle IS a distributed build,
        // whatever happens to exist at the compiled-in source path of the
        // machine that built it. Without this, a bundle assembled on the
        // researcher's own Mac silently stayed in DEVELOPER mode: a family
        // packaging forgot would fall back to the build machine's checkout
        // instead of failing closed, and — worse — the local Python engine
        // would run out of THAT tree rather than the checkout sitting beside
        // the app in the home layout.
        //
        // Guarded on the bundle shape, so the dev workflow is untouched: the
        // SwiftPM executable `run-app.sh` launches is not an .app, stays in
        // developer mode, and resolves exactly as before.
        if Bundle.main.bundleURL.pathExtension == "app" {
            CodeResources.releaseModeAsserted = true
        }
        // WP2 — the packaged tier's own proof. A bundled build resolves its
        // shipped resource families out of Contents/Resources, and there is
        // no test that can fake `Bundle.main`, so the check runs for real at
        // launch and reports to stderr. `scripts/build-app.sh` launches the
        // assembled bundle with no DYLD_* environment and prints whatever it
        // writes (launch-check.log), which is exactly where a family that
        // packaging forgot to stage becomes visible — before the researcher
        // finds it as a broken button. Resolution only: nothing is hashed
        // here (that is `steerlab-cli install verify`), so it costs
        // microseconds.
        Self.reportResourceSelfCheck()
        let workspace = WorkspaceStore()
        let cluster = ClusterConnectionStore()
        let catalog = SubstrateCatalog(store: cluster)
        _workspace = State(initialValue: workspace)
        _cluster = State(initialValue: cluster)
        _tunnel = State(initialValue: ClusterTunnel())
        _catalog = State(initialValue: catalog)
        _service = State(initialValue: ChatService(cluster: cluster, catalog: catalog))
        cluster.attachTunnel(_tunnel.wrappedValue)  // WS3: the client + health card read tunnel state through the store
        // F4: same-machine server auto-switch is policy on the cluster store,
        // triggered from the ONE workspace-root-change seam — picker New/Open,
        // programmatic switches, and the launch-restored root below all
        // funnel through it. Remote servers stay explicit (the mismatch
        // banner's manual affordances).
        workspace.onRootChange = { _ in
            cluster.synchronizeServerToLocalWorkspace()
        }
        cluster.synchronizeServerToLocalWorkspace()  // launch-restored root
        // The workspace-contract upkeep line, once per OPEN. `switchTo`
        // raises it for every in-session switch; the launch-restored root
        // never goes through `switchTo` (the store resolves it in `init`), so
        // it is raised here too — the same two-call shape the line above uses,
        // and for the same reason.
        workspace.noteAgentContractUpkeep()
        // Running as a bare SPM executable (no .app bundle): claim regular
        // app status so the window gets focus and a menu bar.
        DispatchQueue.main.async {
            NSApp.setActivationPolicy(.regular)
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    /// Write the resource self-check to stderr: one summary line always, plus
    /// a line per genuinely broken family. A healthy build is one quiet line;
    /// a build packaged without a family names it and says to reinstall.
    private static func reportResourceSelfCheck() {
        let check = CodeResources.selfCheck()
        var text = check.summaryLine + "\n"
        for row in check.problems {
            text += "  \(row.family.rawValue): \(row.problem ?? "unresolved")\n"
        }
        // Bucket B is the other half of the tier question, and it is the half
        // the researcher actually feels ("why is Start Local Python Server
        // greyed out?"), so it is always on the line rather than only on
        // failure.
        if let checkout = check.executableCheckout {
            text += "  local engine: \(checkout.root.path) "
                + "(\(checkout.origin.displayName))\n"
        } else {
            text += "  local engine: unavailable — "
                + (check.executableCheckoutProblem ?? "no checkout") + "\n"
        }
        FileHandle.standardError.write(Data(text.utf8))
    }

    var body: some Scene {
        WindowGroup("SteerLab") {
            VStack(spacing: 0) {
                UpdateBanner(model: updates)
                ChatView(service: service, workspace: workspace)
            }
            // Floor: sidebar (~160) + section controls (≥560 for the dense
            // panels) + viewer (≥420). Below this the sidebar collapses
            // and section labels truncate (live-testing finding).
            .frame(minWidth: 1200, minHeight: 700)
            // After the window is up, never before: the check is entirely
            // async and off the main actor, so launch time is unaffected.
            .task { await updates.runAutomaticCheckIfDue() }
            .alert(
                "Software Update",
                isPresented: Binding(
                    get: { updates.manualReport != nil },
                    set: { if !$0 { updates.manualReport = nil } })
            ) {
                Button("OK", role: .cancel) { updates.manualReport = nil }
                if updates.downloadPage != nil {
                    Button("Open Releases…") {
                        updates.openDownloadPage()
                        updates.manualReport = nil
                    }
                }
            } message: {
                Text(updates.manualReport ?? "")
            }
            .toolbar {
                ToolbarItemGroup {
                    WorkspaceSelector(
                        workspace: workspace, service: service, catalog: catalog)
                    SubstrateSelector(cluster: cluster, service: service)
                    // Item 1 (cluster-testing): GPU session start/stop at
                    // a glance, beside the connection dot — visible only
                    // on server workspaces whose profile supports GPU
                    // sessions. Same controller state as the Playground
                    // section.
                    GPUSessionToolbarControl(service: service)
                    ClusterConnectionDot(
                        cluster: cluster, tunnel: tunnel, service: service,
                        localServer: localServer, localEngine: localEngine)
                }
            }
        }
        // Without an explicit default, the first launch opens near the
        // content minimum and the sidebar truncates section names ("tabs cut
        // off"). 1440×900 fits every section's controls + viewer comfortably.
        .defaultSize(width: 1440, height: 900)
        // "Check for Updates…" lives where macOS apps put it: the app menu,
        // just under About. The toggle beside it is the visible off switch
        // for the automatic once-a-day check.
        .commands {
            CommandGroup(after: .appInfo) {
                UpdateCommands(model: updates)
                Divider()
            }
        }
    }
}

/// Compact window-toolbar workspace switch: Local (MLX) plus every saved
/// server, a connection-status dot, and a popover for adding/editing servers
/// (name, URL, bearer token) and — on server workspaces — installing models.
private struct SubstrateSelector: View {
    @Bindable var cluster: ClusterConnectionStore
    let service: ChatService
    @State private var showingServerEditor = false
    /// nil while the editor is adding a new server; otherwise the entry being
    /// edited.
    @State private var editingServerID: ClusterConnectionStore.ServerEntry.ID?

    var body: some View {
        // The "Compute:" prefix on the collapsed menu keeps it self-describing
        // in the toolbar (discoverability finding): it reads
        // "Compute: Local (MLX)" / "Compute: <server name>".
        Menu {
            Picker("Workspace", selection: workspaceSelection) {
                Text("Local (MLX)").tag(ClusterConnectionStore.Workspace.local)
                ForEach(cluster.servers) { server in
                    serverMenuItem(server)
                        .tag(ClusterConnectionStore.Workspace.server(server.id))
                }
            }
            .pickerStyle(.inline)
            Divider()
            Button("Add Server…") {
                editingServerID = nil
                showingServerEditor = true
            }
            if let active = cluster.activeServer {
                Button("Edit “\(active.name)”…") {
                    editingServerID = active.id
                    showingServerEditor = true
                }
            }
            // Which workspace the active server actually serves — surfaced in
            // the Compute UI itself so "whose artifacts am I looking at?" never
            // needs a second control (workspace-scoping finding).
            if case .server = cluster.activeWorkspace {
                Divider()
                if let root = cluster.activeServerServingRoot {
                    Text("serving \(root)")
                } else {
                    Text("serving root unknown (not connected)")
                }
                // Runtime workspace switching (capability-gated: hidden on
                // servers without POST /api/workspace/switch). "Serve Current
                // Workspace" is offered only when the server shares this
                // Mac's filesystem (direct loopback transport — never send a
                // Mac path to a remote host); the submenu offers SERVER-side
                // roots (site profile + per-server recents) on any transport.
                if cluster.activeServerSupportsWorkspaceSwitch {
                    if let localRoot = cluster.localWorkspaceRootForServerSwitch,
                        localRoot != cluster.activeServerServingRoot
                    {
                        Button("Serve Current Workspace") {
                            Task {
                                if await cluster.switchServerWorkspace(to: localRoot) {
                                    await service.connectCluster()
                                }
                            }
                        }
                        .help(
                            "repoint this server's serving root at the app's "
                                + "selected workspace (\(localRoot)) without a "
                                + "restart — refused while server jobs are running")
                    }
                    let candidates = cluster.serverWorkspaceSwitchCandidates
                    if !candidates.isEmpty {
                        Menu("Recent Server Workspaces") {
                            ForEach(candidates, id: \.self) { root in
                                Button(root) {
                                    Task {
                                        if await cluster.switchServerWorkspace(to: root) {
                                            await service.connectCluster()
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } label: {
            Label("Compute: \(cluster.substrateLabel)", systemImage: "cpu")
        }
        .labelStyle(.titleAndIcon)
        .help(
            "which workspace the app is scoped to: the MLX engine in this app, "
                + "or a saved Python SteerLab server (its installed models, "
                + "artifacts, runs, and jobs — everything it lists lives under "
                + "its serving root, shown in this menu). Recipes — concepts, "
                + "stimuli, manifests — are git-versioned and visible in every "
                + "workspace. Connection state lives on the dot to the right")
        // The server add/edit popover stays reachable from this menu's
        // Add Server…/Edit… items; the selector's former duplicate
        // connection dot is gone — ClusterConnectionDot (right) is the ONE
        // connection affordance (tunnel state, auth, sites, setup wizard).
        .popover(isPresented: $showingServerEditor, arrowEdge: .bottom) {
            ServerEditorView(cluster: cluster, service: service, serverID: editingServerID)
        }

        pairingIndicator
    }

    /// Compact standing indicator next to the Compute selector: the active
    /// server's artifact root is NOT this app's data workspace, so its
    /// builds/authoring/runs write elsewhere (the root incident). The badge
    /// stays visible everywhere the toolbar is; the full sentence is in the
    /// help text and the Compute section header.
    @ViewBuilder
    private var pairingIndicator: some View {
        if let badge = cluster.activeServerPairingBadge {
            Label(badge, systemImage: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.orange)
                .help(pairingIndicatorHelp)
        }
    }

    private var pairingIndicatorHelp: String {
        cluster.activeServerPairingWarning
            ?? "server root differs from the active workspace"
    }

    /// Menu row for one saved server: a dot marks the active one, and the
    /// last-known running-job count badges entries where it is known
    /// ("gpu-a · 2 jobs" — a memory of the last check, not live polling).
    private func serverMenuItem(_ server: ClusterConnectionStore.ServerEntry) -> some View {
        var title = server.name
        if let badge = cluster.runningJobsBadge(for: server.id) {
            title += " · \(badge)"
        }
        return Group {
            if cluster.activeWorkspace == .server(server.id) {
                Label(title, systemImage: "circle.fill")
            } else {
                Text(title)
            }
        }
    }

    /// Selecting a server workspace connects to it; switching away is
    /// non-destructive (server jobs persist in the server's durable store).
    private var workspaceSelection: Binding<ClusterConnectionStore.Workspace> {
        Binding(
            get: { cluster.activeWorkspace },
            set: { workspace in
                cluster.activeWorkspace = workspace
                if case .server = workspace {
                    Task { await service.connectCluster() }
                }
            })
    }

}

/// Add/edit popover for a saved server: name, URL, bearer token, Connect,
/// Remove, and (when the edited server is the active workspace) a minimal
/// install-model affordance that queues a durable prefetch job on the server.
private struct ServerEditorView: View {
    @Bindable var cluster: ClusterConnectionStore
    let service: ChatService
    /// nil = add a new server.
    let serverID: ClusterConnectionStore.ServerEntry.ID?
    @Environment(\.dismiss) private var dismiss

    @State private var name = ""
    @State private var urlString = ClusterConnectionStore.defaultServerURL
    @State private var token = ""
    @State private var installModelID = ""

    private var existing: ClusterConnectionStore.ServerEntry? {
        serverID.flatMap { cluster.server(id: $0) }
    }

    private var isActiveServer: Bool {
        guard let serverID else { return false }
        return cluster.activeWorkspace == .server(serverID)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(existing == nil ? "Add Server" : "Edit Server")
                .font(.headline)
            TextField("name (defaults to host:port)", text: $name)
                .textFieldStyle(.roundedBorder)
                .help("label shown in the Compute workspace menu")
            TextField("server URL", text: $urlString)
                .textFieldStyle(.roundedBorder)
                .help("base URL for the Python SteerLab server, usually an SSH tunnel or OOD URL")
            SecureField("bearer token", text: $token)
                .textFieldStyle(.roundedBorder)
                .help("optional STEERLAB_AUTH_TOKEN; saved to the Keychain per host:port on connect")
            HStack {
                Button("Connect") { saveAndConnect() }
                    .help(
                        "save this server, switch the workspace to it, and fetch its "
                            + "capabilities, model inventory, jobs, and agents")
                Spacer()
                if let existing {
                    Button("Remove", role: .destructive) {
                        cluster.removeServer(id: existing.id)
                        dismiss()
                    }
                    .help("forget this server (its Keychain token included); jobs keep running server-side")
                }
            }
            if isActiveServer {
                Divider()
                HStack {
                    TextField("Install model… (HF repo id)", text: $installModelID)
                        .textFieldStyle(.roundedBorder)
                        .help(
                            "prefetch a Hugging Face repo into the server's cache as a "
                                + "durable job (full-precision HF ids — MLX repos are rejected "
                                + "with a family-twin hint)")
                    Button("Install") {
                        Task { await cluster.installModel(installModelID) }
                    }
                    .disabled(
                        installModelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
            if isActiveServer, let status = cluster.status {
                Text(status)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: 300, alignment: .leading)
            }
        }
        .padding(12)
        .frame(minWidth: 320)
        .onAppear(perform: populate)
    }

    private func populate() {
        guard let existing else { return }
        name = existing.name
        urlString = existing.urlString
        token = cluster.storedToken(for: existing) ?? ""
    }

    /// All mutation goes through the store (views stay thin): upsert the
    /// entry, stash the token, activate the workspace, connect.
    private func saveAndConnect() {
        let entry: ClusterConnectionStore.ServerEntry
        if let existing {
            cluster.renameServer(id: existing.id, to: name)
            cluster.updateServerURL(id: existing.id, urlString: urlString)
            entry = cluster.server(id: existing.id) ?? existing
        } else {
            entry = cluster.addServer(name: name, urlString: urlString)
        }
        cluster.setStoredToken(token, for: entry)
        cluster.activeWorkspace = .server(entry.id)
        cluster.token = token.trimmingCharacters(in: .whitespacesAndNewlines)
        Task { await service.connectCluster() }
    }
}
