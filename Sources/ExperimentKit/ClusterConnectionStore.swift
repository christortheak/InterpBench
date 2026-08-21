import Foundation
import Observation

/// Workspace-global substrate state: the registry of saved Python servers,
/// which workspace is active (Local MLX vs one of the servers), the active
/// server's connection status/capabilities/model inventory, and per-server
/// last-known running-job counts.
///
/// The design rule (substrate-as-workspace): the toolbar control is a
/// *workspace switch* — Local | server A | server B…. Connecting scopes the UI
/// to that substrate's installed models, derived artifacts, runs, and jobs.
/// Recipes (concepts, stimuli, manifests, variant definitions) never switch —
/// they are the git-versioned truth visible in every workspace. Switching away
/// from a server is non-destructive: its jobs persist in the server's durable
/// job store, and the last-known running-job count stays here so an inactive
/// workspace can badge "2 jobs (as of last check)". Inactive servers are never
/// polled automatically.
///
/// This is deliberately NOT owned by any one panel — the chat surface, the
/// Experiments panel's run-on-server flow, and variant upload all depend on
/// which workspace is active and share one connection. The app creates a
/// single instance and injects it into `ChatService`; panels reach it through
/// their host. Chat-specific selections (which remote model/variant to chat
/// with) stay on `ChatService`.
@Observable @MainActor
public final class ClusterConnectionStore {

    // MARK: Workspaces

    /// A saved Python server. The bearer token is NOT stored here — it lives
    /// in the Keychain (`ClusterTokenStore`), keyed per host:port for direct
    /// servers and per REMOTE identity (host:remotePort) for SSH sites (see
    /// `tokenKey(forEntry:)`).
    public struct ServerEntry: Codable, Sendable, Identifiable, Hashable {
        public var id: UUID
        public var name: String
        /// The URL HTTP actually targets: the server itself for direct
        /// transport, or the tunnel's 127.0.0.1:<localPort> label for SSH
        /// sites (kept in sync by `noteTunnelLocalPort`).
        public var urlString: String
        /// WS1 site profile this entry IS. Optional in storage so registries
        /// persisted before site profiles existed keep decoding (synthesized
        /// Codable uses decodeIfPresent); read through `resolvedSite`, which
        /// views a legacy URL-only entry as a direct-transport site.
        public var site: ClusterSiteProfile?

        public init(
            id: UUID = UUID(), name: String, urlString: String,
            site: ClusterSiteProfile? = nil
        ) {
            self.id = id
            self.name = name
            self.urlString = urlString
            self.site = site
        }

        /// "host:port" derived from the URL, falling back to the raw string
        /// while the URL is mid-edit.
        public var hostLabel: String {
            ClusterConnectionStore.hostLabel(forURLString: urlString)
        }

        /// What pickers and menus show. The SITE's name wins over the
        /// entry's stored label: registry churn (a rebuilt entry, a
        /// tunnel-local URL rename) can leave `name` as a stale
        /// "127.0.0.1:8726"-style label while the profile inside still says
        /// "Lab cluster" — the researcher must never have to recognize
        /// their cluster by its ephemeral port.
        public var displayName: String {
            let siteName = site?.name.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return siteName.isEmpty ? name : siteName
        }

        /// The site profile, synthesizing the legacy direct-transport view
        /// (topology externalServer, no scheduler, empty constraints) for
        /// entries saved before site profiles existed.
        public var resolvedSite: ClusterSiteProfile {
            ClusterConnectionStore.resolvedSite(for: self)
        }
    }

    /// Which substrate the UI is scoped to: the in-process MLX engine or one
    /// of the saved Python servers.
    public enum Workspace: Hashable, Sendable {
        case local
        case server(ServerEntry.ID)
    }

    /// Legacy local-vs-server view of `activeWorkspace`, kept so call sites
    /// that only care *which engine generates* (chat send path, panel
    /// enable/disable) keep working unchanged.
    public enum ComputeTarget: String, CaseIterable, Sendable {
        case local = "Local"
        case server = "Server"
    }

    // MARK: Persistence keys

    /// `nonisolated`: the shared `ClusterSiteRepository` migration (a
    /// headless, non-main-actor path) reads the legacy registry under this
    /// exact key — a key string is immutable data, not main-actor state.
    public nonisolated static let serversDefaultsKey = "SteerLabClusterServers"
    /// The app's own defaults domain, for OTHER processes (the CLI) reading
    /// the legacy registry during site-repository migration. Dev builds
    /// persist under the bare app name; a future bundled/signed app would
    /// change this in one place.
    public nonisolated static let legacyAppDefaultsDomain = "SteerLabApp"

    /// Whether a legacy registry payload decodes to at least one server —
    /// public because the site repository's DEFAULT-argument migration
    /// closure must rank candidate defaults domains by content, and default
    /// arguments cannot reference internal helpers.
    public nonisolated static func legacyRegistryPayloadHasSites(
        _ data: Data
    ) -> Bool {
        decodeServersLeniently(from: data)?.isEmpty == false
    }
    public static let activeWorkspaceDefaultsKey = "SteerLabActiveWorkspace"
    /// Legacy single-server schema; read once to migrate, never written.
    public static let serverURLDefaultsKey = "SteerLabClusterServerURL"
    public static let computeTargetDefaultsKey = "SteerLabComputeTarget"
    // Matches the Python server's default (`steerlab_server.cli serve` → 8080).
    public static let defaultServerURL = "http://127.0.0.1:8080"

    // MARK: State

    /// Saved servers, in user order. Manage through `addServer` /
    /// `renameServer` / `updateServerURL` / `removeServer`; persisted to
    /// UserDefaults as JSON.
    public private(set) var servers: [ServerEntry] {
        didSet { persistServers() }
    }

    /// The active workspace. Switching resets the connection state (it
    /// belongs to the active server) but keeps per-server job counts, and is
    /// otherwise non-destructive — server jobs live in the server's durable
    /// job store.
    public var activeWorkspace: Workspace {
        didSet {
            guard oldValue != activeWorkspace else { return }
            persistWorkspace()
            if case .server(let id) = activeWorkspace { lastActiveServerID = id }
            // Connection state applies to the *active* server only.
            status = nil
            capabilities = nil
            remoteState = nil
            remoteInfo = nil
            remoteVariants = []
            token = ""
            // An operation caption from the previous server is stale here.
            activity = nil
            // GPU-session state belongs to the previous server too. A
            // deliberate reset, not ended-detection — no notice fires.
            gpuSession.resetForConnectionChange()
        }
    }

    /// Most recently active server, so the legacy `computeTarget = .server`
    /// setter can reactivate it. In-memory only.
    private var lastActiveServerID: ServerEntry.ID?

    /// Bearer token (STEERLAB_AUTH_TOKEN) for the active server. Held in
    /// memory while editing; persisted to the Keychain per host:port once it
    /// authenticates.
    public var token = ""

    /// Human-readable CONNECTION line ("connecting...", "connected",
    /// tunnel/controller errors). Owned by the connect path only — operation
    /// progress lives in `activity` (engineer review 2026-07-17: activity
    /// writes overwriting this line made a healthy server read "not
    /// connected" after a cancelled stream, and load failures tripped the
    /// connection dot's failure-word matching). internal(set) keeps mutation
    /// inside ExperimentKit.
    public internal(set) var status: String?

    /// Human-readable OPERATION line ("loading gemma…", "uploading
    /// variant…", stream captions, "response complete", operation errors).
    /// Set by ChatService's server-side actions; nil when nothing is in
    /// flight and no result is worth showing. UI shows it alongside — never
    /// instead of — the connection line.
    public internal(set) var activity: String?
    public internal(set) var capabilities: ClusterCapabilities?
    public internal(set) var remoteState: RemoteState? {
        didSet {
            // Track the running-job count for the active server whenever
            // fresh state arrives; keep the last-known value when state is
            // cleared (workspace switch) so inactive entries can badge
            // "2 jobs (as of last check)".
            guard let remoteState, case .server(let id) = activeWorkspace else { return }
            runningJobsByServer[id] = remoteState.jobs.filter { $0.finishedAt == nil }.count
        }
    }
    public internal(set) var remoteVariants: [RemoteVariantRecord] = []

    /// `GET /api/info` from the active server: its artifact root (realpath)
    /// and whether that root looks like the SteerLab source checkout. Drives
    /// the standing unpaired-server indicator; nil until connected (or when
    /// an older server doesn't answer) — no root, no warning.
    public internal(set) var remoteInfo: RemoteServerInfo?

    /// Last-known count of unfinished jobs per server, refreshed whenever the
    /// active server reports state. Inactive servers are never polled — their
    /// entry is a memory of the last check, not live truth.
    public private(set) var runningJobsByServer: [ServerEntry.ID: Int] = [:]

    /// GPU-session lifecycle for the ACTIVE server (plan Wave 2): the latest
    /// session record, the §2.7 display state, and the start/stop/keepalive
    /// actions. Owned here — like the connection itself — so the Playground
    /// control and the connection dot render one truth; wired to `client` /
    /// `capabilities` at init and reset on every workspace switch. All its
    /// affordances (and its polling) gate on `chat.gpuSession`.
    public let gpuSession = GPUSessionController()

    private let defaults: UserDefaults

    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
        if let data = defaults.data(forKey: Self.serversDefaultsKey),
            let decoded = Self.decodeServersLeniently(from: data)
        {
            let deduplicated = Self.deduplicatedServersWithAliases(decoded)
            // Heal names eaten by earlier registry churn: an entry whose
            // stored name is just its (ephemeral, tunnel-local) host label
            // while its site profile carries a real name adopts the site's.
            self.servers = deduplicated.servers.map { entry in
                var entry = entry
                let siteName =
                    entry.site?.name.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                if !siteName.isEmpty,
                    entry.name.isEmpty || entry.name == entry.hostLabel
                {
                    entry.name = siteName
                }
                return entry
            }
            let persistedWorkspace = defaults.string(forKey: Self.activeWorkspaceDefaultsKey)
            let parsedWorkspace = Self.parseWorkspace(
                persistedWorkspace, servers: deduplicated.servers)
            if parsedWorkspace == .local,
                let aliasID = Self.serverID(fromWorkspaceString: persistedWorkspace),
                let retainedID = deduplicated.aliases[aliasID]
            {
                self.activeWorkspace = .server(retainedID)
            } else {
                self.activeWorkspace = parsedWorkspace
            }
        } else if let legacyURL = defaults.string(forKey: Self.serverURLDefaultsKey),
            !legacyURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            // First launch with the registry schema: seed one entry from the
            // old single-server default, named by its host, and preserve the
            // old compute-target selection as the active workspace. The
            // Keychain token needs no migration — it is already keyed by
            // host:port.
            let entry = ServerEntry(
                name: Self.hostLabel(forURLString: legacyURL), urlString: legacyURL)
            self.servers = [entry]
            let legacyTarget =
                defaults.string(forKey: Self.computeTargetDefaultsKey)
                    .flatMap(ComputeTarget.init(rawValue:)) ?? .local
            self.activeWorkspace = legacyTarget == .server ? .server(entry.id) : .local
        } else {
            self.servers = []
            self.activeWorkspace = .local
        }
        if case .server(let id) = activeWorkspace { lastActiveServerID = id }
        // didSet does not fire during init: stamp the new schema explicitly so
        // the legacy keys are consulted exactly once.
        persistServers()
        persistWorkspace()
        // GPU-session wiring: the controller reads the ACTIVE server's client
        // and the last-fetched capability verdict through these closures, so
        // URL/token edits and reconnects apply without re-wiring.
        gpuSession.transportProvider = { [weak self] in
            guard let self, let client = self.client else { return nil }
            return GPUSessionTransport(client: client)
        }
        gpuSession.capabilityProvider = { [weak self] in
            self?.capabilities?.supportsGPUSession == true
        }
        // When the session becomes reachable, /api/state starts answering for
        // the WORKER — any state fetched before that moment (no loaded model)
        // is stale and leaves the chat composer locked until some unrelated
        // action refreshes it (live shakedown: typing stayed disabled after a
        // load raced the session's startup). One refresh at the transition
        // keeps the gate honest.
        gpuSession.onBecameReady = { [weak self] in
            guard let self else { return }
            Task { @MainActor [weak self] in
                guard let self, let client = self.client else { return }
                if let state = try? await client.state() {
                    self.remoteState = state
                }
            }
        }
    }

    // MARK: Registry management

    /// Add a server (empty name defaults to its host:port label).
    @discardableResult
    public func addServer(name: String, urlString: String) -> ServerEntry {
        let trimmedURL = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if let existingIndex = servers.firstIndex(where: {
            Self.registryKey(forEntry: $0) == Self.normalizedEndpointKey(trimmedURL)
        }) {
            if !trimmedName.isEmpty {
                servers[existingIndex].name = trimmedName
            }
            return servers[existingIndex]
        }
        let entry = ServerEntry(
            name: trimmedName.isEmpty ? Self.hostLabel(forURLString: trimmedURL) : trimmedName,
            urlString: trimmedURL)
        servers.append(entry)
        return entry
    }

    public func renameServer(id: ServerEntry.ID, to name: String) {
        guard let index = servers.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        servers[index].name = trimmed.isEmpty ? servers[index].hostLabel : trimmed
    }

    public func updateServerURL(id: ServerEntry.ID, urlString: String) {
        guard let index = servers.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        // SSH-transport sites: the URL is only the tunnel's local label — the
        // token stays keyed by the REMOTE identity, and endpoint-dedupe
        // against other entries' local labels would be meaningless.
        if servers[index].site?.isSSHTransport == true {
            guard servers[index].urlString != trimmed else { return }
            servers[index].urlString = trimmed
            if activeWorkspace == .server(id) {
                status = nil
                capabilities = nil
                remoteState = nil
                remoteInfo = nil
                remoteVariants = []
                // token deliberately kept: its Keychain key (host:remotePort)
                // does not move with the local port.
            }
            return
        }
        let newKey = Self.normalizedEndpointKey(trimmed)
        if let duplicateIndex = servers.firstIndex(where: {
            $0.id != id && Self.registryKey(forEntry: $0) == newKey
        }) {
            let duplicate = servers[duplicateIndex]
            if !servers[index].name.isEmpty,
                servers[index].name != servers[index].hostLabel,
                duplicate.name == duplicate.hostLabel
            {
                servers[duplicateIndex].name = servers[index].name
            }
            if activeWorkspace == .server(id) {
                activeWorkspace = .server(duplicate.id)
            }
            removeServer(id: id)
            return
        }
        guard servers[index].urlString != trimmed else { return }
        servers[index].urlString = trimmed
        // A direct-transport entry that carries a site keeps the profile in
        // sync so exports/shares reflect the edited URL.
        if servers[index].site != nil, let url = Self.endpointURL(from: trimmed) {
            servers[index].site?.transport = .direct(baseURL: url)
        }
        // A different URL is a different endpoint: the token key changes and
        // any connection state for this entry is stale.
        if activeWorkspace == .server(id) {
            status = nil
            capabilities = nil
            remoteState = nil
            remoteInfo = nil
            remoteVariants = []
            token = ""
        }
    }

    /// Remove a saved server. If it was active, fall back to Local. Its
    /// Keychain token is deleted unless another entry shares the same token
    /// key (URL host:port for direct entries, remote identity for SSH sites).
    public func removeServer(id: ServerEntry.ID) {
        guard let index = servers.firstIndex(where: { $0.id == id }) else { return }
        let entry = servers.remove(at: index)
        runningJobsByServer[id] = nil
        if lastActiveServerID == id { lastActiveServerID = nil }
        if activeWorkspace == .server(id) { activeWorkspace = .local }
        let key = Self.tokenKey(forEntry: entry)
        if !servers.contains(where: { Self.tokenKey(forEntry: $0) == key }) {
            ClusterTokenStore.delete(key: key)
        }
    }

    /// The entry for the active server workspace, if any.
    public var activeServer: ServerEntry? {
        guard case .server(let id) = activeWorkspace else { return nil }
        return servers.first { $0.id == id }
    }

    public func server(id: ServerEntry.ID) -> ServerEntry? {
        servers.first { $0.id == id }
    }

    // MARK: Site profiles (WS1 — cluster site profiles + connection lifecycle)

    /// Every saved entry viewed as a site profile (legacy URL-only entries
    /// surface as direct-transport sites).
    public var sites: [ClusterSiteProfile] { servers.map(\.resolvedSite) }

    /// The active server's site profile; nil in the Local workspace.
    public var activeSite: ClusterSiteProfile? { activeServer?.resolvedSite }

    /// Seed URL an entry starts with for a given profile: the base URL for
    /// direct transport, or the deterministic tunnel-local label for SSH
    /// (refined by `noteTunnelLocalPort` once a tunnel actually opens).
    nonisolated static func seedURLString(for profile: ClusterSiteProfile) -> String {
        if let direct = profile.directURLString { return direct }
        return "http://127.0.0.1:\(profile.preferredLocalPort ?? 8700)"
    }

    /// Add a saved entry that IS a site profile, or refresh the profile of
    /// the entry already registered for the same endpoint (direct sites
    /// dedupe by URL like `addServer`; SSH sites dedupe by remote identity,
    /// so tunnel-local port labels never mint duplicates). A custom name the
    /// user typed is kept; a defaulted host label adopts the profile's name.
    @discardableResult
    public func addSite(_ profile: ClusterSiteProfile) -> ServerEntry {
        let key = Self.canonicalKey(Self.registryKey(forProfile: profile))
        if let index = servers.firstIndex(where: {
            Self.canonicalKey(Self.registryKey(forEntry: $0)) == key
        }) {
            let keptName = servers[index].name
            let wasDefaultName = keptName.isEmpty || keptName == servers[index].hostLabel
            servers[index].site = profile
            if wasDefaultName, !profile.name.isEmpty {
                servers[index].name = profile.name
            }
            if let direct = profile.directURLString {
                servers[index].urlString = direct
            }
            return servers[index]
        }
        let seedURL = Self.seedURLString(for: profile)
        let name =
            profile.name.isEmpty ? Self.hostLabel(forURLString: seedURL) : profile.name
        let entry = ServerEntry(name: name, urlString: seedURL, site: profile)
        servers.append(entry)
        return entry
    }

    /// Replace the site profile of a saved entry (site editor, import-over).
    /// Name and URL follow the profile; connection state resets when it is
    /// the active workspace — the endpoint may have changed. The in-memory
    /// token is cleared (the Keychain copy, keyed by the new profile's
    /// identity, reloads on the next connect).
    public func updateSite(id: ServerEntry.ID, profile: ClusterSiteProfile) {
        guard let index = servers.firstIndex(where: { $0.id == id }) else { return }
        servers[index].site = profile
        if !profile.name.isEmpty { servers[index].name = profile.name }
        servers[index].urlString = Self.seedURLString(for: profile)
        if activeWorkspace == .server(id) {
            status = nil
            capabilities = nil
            remoteState = nil
            remoteInfo = nil
            remoteVariants = []
            token = ""
        }
    }

    /// Decode a shared site-profile JSON (schema-checked) and register it.
    ///
    /// Import-over is the app twin of `cluster sites import`, and it refuses
    /// on the same finding (open-issues §17): the canonical identity ignores
    /// the ssh `user@` half, so a login-less profile lands on the
    /// login-carrying entry and replaces it — after which `ssh <host>`
    /// authenticates as the local account. `addSite` itself stays silent
    /// because its other callers (presets, a freshly built profile from the
    /// editor) carry no stored login to lose; `addPreset` already refuses to
    /// clobber an edited entry.
    @discardableResult
    public func importSite(from data: Data) throws -> ServerEntry {
        let profile = try ClusterSiteProfile.decode(from: data)
        let key = Self.canonicalKey(Self.registryKey(forProfile: profile))
        let existing = servers.first {
            Self.canonicalKey(Self.registryKey(forEntry: $0)) == key
        }?.site
        if case .refuse(let user, let host, let source) =
            ClusterSiteRepository.sshLoginFinding(
                incoming: profile, existing: existing) {
            throw ClusterLifecycleError.sshLoginDropped(
                siteID: profile.name.isEmpty ? host : profile.name,
                host: host, expectedUser: user, source: source)
        }
        return addSite(profile)
    }

    /// A saved entry's site as shareable JSON (sorted keys, pretty printed);
    /// nil when the id names no entry.
    public func exportSite(id: ServerEntry.ID) throws -> Data? {
        guard let entry = server(id: id) else { return nil }
        return try entry.resolvedSite.encoded()
    }

    /// Register one of the shipped presets. NEVER clobbers an edited site:
    /// when an entry for the same site already exists (canonical identity
    /// ignoring ssh `user@`, or the preset's name), it is returned untouched
    /// unless it is itself a pristine preset copy (which refreshes to the
    /// current shipped values). Adding a preset over the researcher's edited
    /// cluster profile must not reset the host or empty the storage roots.
    @discardableResult
    public func addPreset(_ preset: ClusterSiteProfile) -> ServerEntry {
        if let index = servers.firstIndex(where: { Self.matches($0, preset: preset) }) {
            if servers[index].site == nil || Self.isPresetShaped(servers[index]) {
                return addSite(preset)
            }
            return servers[index]
        }
        return addSite(preset)
    }

    /// Whether a saved entry IS a given preset's site, for presence checks:
    /// canonical registry identity (ssh `user@` ignored) or the preset's
    /// name. The researcher's edited `user@host…` entry must count as
    /// "that preset is present" — re-offering it forks the registry.
    nonisolated static func matches(_ entry: ServerEntry, preset: ClusterSiteProfile) -> Bool {
        if canonicalKey(registryKey(forEntry: entry))
            == canonicalKey(registryKey(forProfile: preset))
        {
            return true
        }
        return !preset.name.isEmpty && entry.name == preset.name
    }

    /// Shipped presets not yet represented in the registry — the connection
    /// dot offers these as one-click adds. An edited entry for the same site
    /// (matched by canonical remote identity ignoring ssh `user@`, or by the
    /// preset's name) counts as present.
    public var missingPresets: [ClusterSiteProfile] {
        ClusterSiteProfile.presets.filter { preset in
            !servers.contains { Self.matches($0, preset: preset) }
        }
    }

    /// WS1 glue: when the tunnel reports `.up`, align the active SSH entry's
    /// effective local URL with the actual local port. The Keychain token is
    /// keyed by the REMOTE identity (host:remotePort), so this never orphans
    /// a token; only the HTTP client's target moves.
    public func noteTunnelLocalPort(_ port: Int) {
        guard case .server(let id) = activeWorkspace,
            let entry = server(id: id),
            entry.resolvedSite.isSSHTransport
        else { return }
        updateServerURL(id: id, urlString: "http://127.0.0.1:\(port)")
    }

    // MARK: Compatibility accessors (pre-registry call sites)

    /// Local-vs-server selection derived from `activeWorkspace`. Setting
    /// `.server` reactivates the most recently active saved server (or the
    /// first one); with an empty registry it is a no-op — there is nothing to
    /// connect to.
    public var computeTarget: ComputeTarget {
        get { activeWorkspace == .local ? .local : .server }
        set {
            switch newValue {
            case .local:
                activeWorkspace = .local
            case .server:
                if case .server = activeWorkspace { return }
                if let id = lastActiveServerID, servers.contains(where: { $0.id == id }) {
                    activeWorkspace = .server(id)
                } else if let first = servers.first {
                    activeWorkspace = .server(first.id)
                }
            }
        }
    }

    /// Base URL of the active server (or, in the Local workspace, the first
    /// saved server — legacy call sites use this only for labels/clients that
    /// are inert while local).
    public var serverURL: String {
        activeServer?.urlString ?? servers.first?.urlString ?? Self.defaultServerURL
    }

    // MARK: Connection

    /// Keychain account key for the active server's bearer token, so
    /// distinct servers keep distinct tokens (shared by chat + Experiments).
    /// Direct servers keep the historical URL host:port key; SSH sites key
    /// by REMOTE identity (host:remotePort) so tunnel-local port changes
    /// never orphan a token.
    public var tokenKey: String {
        if let activeServer { return Self.tokenKey(forEntry: activeServer) }
        return ClusterTokenStore.key(forURLString: serverURL)
    }

    /// Token key for one entry: the SSH remote identity when the entry is an
    /// SSH site, else the historical URL-derived host:port key — existing
    /// Keychain items keep working with no migration.
    nonisolated static func tokenKey(forEntry entry: ServerEntry) -> String {
        if let identity = entry.site?.remoteTokenIdentity { return identity }
        return ClusterTokenStore.key(forURLString: entry.urlString)
    }

    /// Client for the active server's URL + token; nil when the URL doesn't
    /// parse. Rebuilt on access so edits to the URL/token apply immediately.
    ///
    /// SSH sites prefer the attached tunnel's LIVE local URL when the tunnel
    /// for this exact site is up — a reopened tunnel that landed on a
    /// different local port is picked up immediately instead of waiting for
    /// the next `noteTunnelLocalPort` sync. Everything else (direct sites,
    /// tunnel down, no tunnel attached) keeps today's stored-URL path.
    /// Test seam: the `URLSession` handed to every `ClusterClient` this store
    /// builds. Production leaves it `nil` (`.shared`); tests install a
    /// `URLProtocol` stub so paths that run catalog responses through real
    /// composition — the adapter-catalog → composed-agent mapping — can be
    /// exercised end to end rather than asserted a layer at a time.
    public var clientSessionOverride: URLSession?

    public var client: ClusterClient? {
        var url = URL(string: serverURL.trimmingCharacters(in: .whitespacesAndNewlines))
        if let site = activeSite, site.isSSHTransport,
            let tunnel = attachedTunnel, tunnel.site == site,
            let tunnelURL = tunnel.effectiveBaseURL
        {
            url = tunnelURL
        }
        guard let url else { return nil }
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        return ClusterClient(
            profile: ClusterConnectionProfile(baseURL: url, tokenKey: tokenKey),
            token: trimmed.isEmpty ? nil : trimmed,
            session: clientSessionOverride ?? .shared)
    }

    // MARK: Tunnel attachment (WS1/WS3 glue)

    /// The app's one tunnel manager, attached at launch. Lets `client`
    /// prefer a live tunnel's local URL and lets the health card render
    /// tunnel state without threading the instance through every view.
    /// Both objects are app-lifetime main-actor singletons; holding it
    /// strongly is deliberate.
    public private(set) var attachedTunnel: ClusterTunnel?

    public func attachTunnel(_ tunnel: ClusterTunnel) {
        attachedTunnel = tunnel
    }

    // MARK: Evidence auto-import (WS3 — registration + per-site flag)

    /// The evidence auto-import service, registered once (lazily) by the
    /// first surface that needs it. Owned here so the poll loop survives
    /// view churn; the service itself gates every tick on the active
    /// workspace, connection state, and the per-site flag below.
    public private(set) var evidenceAutoImport: EvidenceAutoImportService?

    /// Register (or return) the auto-import service for a workspace root and
    /// start its poll loop. Idempotent; the first caller wins the root —
    /// callers pass nil to use the resolved data workspace.
    @discardableResult
    public func registerEvidenceAutoImport(
        workspaceRoot: URL? = nil
    ) -> EvidenceAutoImportService {
        if let evidenceAutoImport { return evidenceAutoImport }
        let root = workspaceRoot ?? VectorCatalog.projectRoot
        let service = EvidenceAutoImportService(workspaceRoot: root, cluster: self)
        evidenceAutoImport = service
        service.startPolling()
        return service
    }

    /// UserDefaults key prefix for the per-site auto-import flag (keyed by
    /// the same registry identity as the entry itself).
    public static let autoImportDefaultsKeyPrefix = "SteerLabAutoImportEnabled:"

    /// Whether evidence auto-import is on for a saved site. Default ON for
    /// SSH sites (results should come home from a remote cluster by
    /// default), OFF for direct/localhost entries where the "remote" tree
    /// may be this very machine.
    public func autoImportEnabled(for entry: ServerEntry) -> Bool {
        let key = Self.autoImportDefaultsKeyPrefix + Self.registryKey(forEntry: entry)
        if defaults.object(forKey: key) != nil { return defaults.bool(forKey: key) }
        return entry.resolvedSite.isSSHTransport
    }

    public func setAutoImportEnabled(_ enabled: Bool, for entry: ServerEntry) {
        let key = Self.autoImportDefaultsKeyPrefix + Self.registryKey(forEntry: entry)
        defaults.set(enabled, forKey: key)
    }

    /// The active site's flag; false in the Local workspace.
    public var activeAutoImportEnabled: Bool {
        guard let activeServer else { return false }
        return autoImportEnabled(for: activeServer)
    }

    /// Compact label for the active workspace (window toolbar).
    public var substrateLabel: String {
        switch activeWorkspace {
        case .local: "Local (MLX)"
        case .server: activeServer?.name ?? serverHostLabel
        }
    }

    /// "host:port" for the active server, falling back to the raw string
    /// while the URL is mid-edit.
    public var serverHostLabel: String {
        Self.hostLabel(forURLString: serverURL)
    }

    nonisolated static func hostLabel(forURLString urlString: String) -> String {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = endpointURL(from: trimmed), let host = url.host() else { return trimmed }
        guard let port = url.port else { return host }
        return "\(host):\(port)"
    }

    nonisolated static func normalizedEndpointKey(_ urlString: String) -> String {
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = endpointURL(from: trimmed), let host = url.host()?.lowercased() else {
            return trimmed.lowercased()
        }
        let scheme = (url.scheme ?? "http").lowercased()
        let port = url.port ?? (scheme == "https" ? 443 : 80)
        return "\(scheme)://\(host):\(port)"
    }

    /// Per-entry lenient registry decode: one corrupt or foreign-schema entry
    /// must cost THAT entry, never the whole registry. The previous
    /// all-or-nothing `decode([ServerEntry].self)` meant any single entry
    /// failing to decode silently dropped every saved site to the legacy
    /// fallback branches — and init's re-persist then overwrote the stored
    /// registry, turning a schema wobble into permanent total loss (the
    /// researcher re-entered their cluster site on every launch). Returns nil
    /// only when the payload is not a JSON array at all (the legacy
    /// single-URL fallbacks apply then, exactly as before).
    nonisolated static func decodeServersLeniently(from data: Data) -> [ServerEntry]? {
        if let decoded = try? JSONDecoder().decode([ServerEntry].self, from: data) {
            return decoded
        }
        struct LenientEntry: Decodable {
            let entry: ServerEntry?
            init(from decoder: any Decoder) {
                entry = try? ServerEntry(from: decoder)
            }
        }
        guard let lenient = try? JSONDecoder().decode([LenientEntry].self, from: data)
        else { return nil }
        return lenient.compactMap(\.entry)
    }

    nonisolated static func deduplicatedServers(_ servers: [ServerEntry]) -> [ServerEntry] {
        deduplicatedServersWithAliases(servers).servers
    }

    /// Load-time dedup, keyed by CANONICAL registry identity (ssh user@
    /// stripped — `user@hpc.example.edu` and
    /// `hpc.example.edu` are the same site). On a collision the
    /// EDITED entry wins over a preset-shaped sibling: user edits are never
    /// the copy that loses. Between two entries of the same shape the one that
    /// carries an ssh LOGIN wins (open-issues §17: array order used to decide,
    /// so a bare-host duplicate could silently swallow the `user@host` entry
    /// this registry authenticates with — and the one-time migration into
    /// `cluster-sites.json` runs through here); otherwise the first wins, as
    /// before.
    nonisolated static func deduplicatedServersWithAliases(_ servers: [ServerEntry])
        -> (servers: [ServerEntry], aliases: [ServerEntry.ID: ServerEntry.ID])
    {
        var indexByKey: [String: Int] = [:]
        var result: [ServerEntry] = []
        var aliases: [ServerEntry.ID: ServerEntry.ID] = [:]
        for server in servers {
            let key = canonicalKey(registryKey(forEntry: server))
            if let existingIndex = indexByKey[key] {
                let incumbent = result[existingIndex]
                let incumbentIsPreset = isPresetShaped(incumbent)
                let candidateIsPreset = isPresetShaped(server)
                let candidateRescuesTheLogin =
                    !candidateIsPreset && !incumbentIsPreset
                    && sshLogin(of: incumbent) == nil
                    && sshLogin(of: server) != nil
                if (incumbentIsPreset && !candidateIsPreset)
                    || candidateRescuesTheLogin {
                    result[existingIndex] = server
                    aliases[incumbent.id] = server.id
                } else {
                    aliases[server.id] = result[existingIndex].id
                }
                continue
            }
            indexByKey[key] = result.count
            result.append(server)
        }
        return (result, aliases)
    }

    /// Canonical form of a registry key for MATCHING (dedup, preset
    /// presence, add-over): ssh keys drop the `user@` login prefix — the
    /// preset ships the bare host, the researcher's edited site
    /// authenticates as `user@host`, and treating those as different sites
    /// is how the preset re-offer + relaunch dedup used to eat the edited
    /// entry. Direct-URL keys pass through unchanged. Token/prefs keys keep
    /// the exact form (a login identity IS part of the SSH identity there).
    nonisolated static func canonicalKey(_ key: String) -> String {
        guard key.hasPrefix("ssh://"), let at = key.lastIndex(of: "@") else { return key }
        return "ssh://" + key[key.index(after: at)...]
    }

    /// The ssh login an entry authenticates with (`alice@host` →
    /// `alice`), or nil for a bare host or a non-ssh entry. The single
    /// definition lives on the repository so the dedup and the write-time
    /// guard read a destination the same way.
    nonisolated static func sshLogin(of entry: ServerEntry) -> String? {
        guard let site = entry.site, case .ssh(let host, _, _, _) = site.transport
        else { return nil }
        return ClusterSiteRepository.sshLogin(in: host)
    }

    /// Whether an entry's profile is byte-for-byte one of the shipped
    /// presets (an untouched one-click add). Anything the user edited —
    /// host, storage roots, scheduler data — differs and counts as edited.
    nonisolated static func isPresetShaped(_ entry: ServerEntry) -> Bool {
        guard let site = entry.site else { return false }
        return ClusterSiteProfile.presets.contains(site)
    }

    /// Registry-dedupe key: SSH sites are identified by their remote
    /// endpoint (two sites sharing a tunnel-local 127.0.0.1:port label must
    /// NOT merge), everything else by normalized URL as before.
    nonisolated static func registryKey(forEntry entry: ServerEntry) -> String {
        if let site = entry.site { return registryKey(forProfile: site) }
        return normalizedEndpointKey(entry.urlString)
    }

    nonisolated static func registryKey(forProfile profile: ClusterSiteProfile) -> String {
        if let identity = profile.registryIdentity { return identity }
        return normalizedEndpointKey(profile.directURLString ?? "")
    }

    /// The site profile an entry presents: its own, or the legacy
    /// direct-transport view for entries saved before site profiles existed
    /// (transport = their URL, topology externalServer, no scheduler, empty
    /// constraints).
    nonisolated static func resolvedSite(for entry: ServerEntry) -> ClusterSiteProfile {
        if let site = entry.site { return site }
        let trimmed = entry.urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        let url = endpointURL(from: trimmed) ?? ClusterSiteProfile.fallbackDirectBaseURL
        return ClusterSiteProfile(
            name: entry.name.isEmpty ? hostLabel(forURLString: entry.urlString) : entry.name,
            transport: .direct(baseURL: url),
            topology: .externalServer,
            scheduler: .none,
            constraints: ClusterSiteProfile.SiteConstraints())
    }

    nonisolated static func endpointURL(from trimmedURLString: String) -> URL? {
        guard !trimmedURLString.isEmpty else { return nil }
        if let url = URL(string: trimmedURLString), url.host() != nil {
            return url
        }
        return URL(string: "http://\(trimmedURLString)")
    }

    nonisolated static func serverID(fromWorkspaceString workspaceString: String?) -> ServerEntry.ID? {
        guard let workspaceString, workspaceString.hasPrefix("server:") else { return nil }
        return UUID(uuidString: String(workspaceString.dropFirst("server:".count)))
    }

    /// Load a previously saved bearer token for the active server into the
    /// in-memory field (called before connecting so the user needn't retype it).
    public func loadStoredToken() {
        if token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            let stored = ClusterTokenStore.load(key: tokenKey)
        {
            token = stored
        }
    }

    /// Persist (or clear) the current bearer token in the Keychain.
    public func persistToken() {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            ClusterTokenStore.delete(key: tokenKey)
        } else {
            try? ClusterTokenStore.save(trimmed, key: tokenKey)
        }
    }

    /// Stored Keychain token for an arbitrary saved server (the add/edit
    /// popover edits entries that may not be active).
    public func storedToken(for entry: ServerEntry) -> String? {
        ClusterTokenStore.load(key: Self.tokenKey(forEntry: entry))
    }

    /// Stored Hugging Face token for a saved site (Keychain). The DURABLE
    /// home is the cluster's `$HF_HOME/token` — this Mac-side copy only makes
    /// re-installs and rotation possible without re-pasting.
    public func storedHFToken(for entry: ServerEntry) -> String? {
        ClusterTokenStore.load(key: Self.tokenKey(forEntry: entry) + ".hf")
    }

    /// Save (or clear, when empty) the Hugging Face token for a saved site.
    public func setStoredHFToken(_ token: String, for entry: ServerEntry) {
        let key = Self.tokenKey(forEntry: entry) + ".hf"
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            ClusterTokenStore.delete(key: key)
        } else {
            try? ClusterTokenStore.save(trimmed, key: key)
        }
    }

    /// Save (or clear, when empty) a token for an arbitrary saved server.
    public func setStoredToken(_ token: String, for entry: ServerEntry) {
        let key = Self.tokenKey(forEntry: entry)
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            ClusterTokenStore.delete(key: key)
        } else {
            try? ClusterTokenStore.save(trimmed, key: key)
        }
        if activeWorkspace == .server(entry.id) {
            self.token = trimmed
        }
    }

    /// Fetch capabilities, state, and variants from the active server; on
    /// success the just-authenticated token is remembered in the Keychain and
    /// the server's running-job count is refreshed.
    @discardableResult
    public func connect() async -> Bool {
        loadStoredToken()
        guard let client else {
            status = "invalid server URL"
            return false
        }
        do {
            status = "connecting..."
            async let caps = client.capabilities()
            async let state = client.state()
            async let variants = client.variants()
            async let info = client.serverInfo()
            capabilities = try await caps
            // State and variants are BEST-EFFORT during the handshake:
            // capabilities is the gate. Older controllers answered /api/state
            // with 503 while a GPU session was queued/starting, and a strict
            // fetch here turned a healthy reconnect into "connection failed"
            // (live 2026-07-17; newer servers compose controller state
            // instead, but the handshake must not depend on that).
            if let fetched = try? await state { remoteState = fetched }
            remoteVariants = (try? await variants) ?? remoteVariants
            // Pairing info is best-effort: an older server without a usable
            // /api/info must not fail the whole connection. Capabilities
            // report the same serving root top-level, so a failed info fetch
            // still yields a pairing verdict (never a silent "unknown" while
            // server-scoped panels show that root's artifacts).
            remoteInfo = try? await info
            if remoteInfo?.root == nil, let capabilitiesRoot = capabilities?.root {
                remoteInfo = RemoteServerInfo(
                    service: remoteInfo?.service, root: capabilitiesRoot,
                    rootLooksLikeSourceCheckout: remoteInfo?.rootLooksLikeSourceCheckout)
            }
            persistToken()  // remember a token that just authenticated
            status = "connected"
            // A session may already be running (controller restarts are a
            // blip — the record survives them): look once and resume the
            // countdown/polling. No-op without the capability.
            gpuSession.refreshAfterConnect()
            return true
        } catch {
            status = Self.friendlyConnectionFailure(
                error, urlString: serverURL,
                throughSSHTunnel: activeSite?.isSSHTransport == true)
            return false
        }
    }

    /// One human-readable line instead of an NSError dump — this string is
    /// rendered in the toolbar popover and the Steering pane status row.
    static func friendlyConnectionFailure(
        _ error: Error, urlString: String, throughSSHTunnel: Bool = false
    ) -> String {
        let host = URL(string: urlString).map {
            "\($0.host ?? "?"):\($0.port.map(String.init) ?? "80")"
        } ?? urlString
        if let urlError = error as? URLError {
            if throughSSHTunnel,
                [
                    URLError.Code.cannotConnectToHost, .cannotFindHost, .timedOut,
                    .notConnectedToInternet, .networkConnectionLost,
                ].contains(urlError.code)
            {
                return "connection failed: the SSH tunnel opened, but the SteerLab "
                    + "controller is not answering behind it — start or reconnect the "
                    + "controller batch job and check its serverd.host record"
            }
            switch urlError.code {
            case .cannotConnectToHost, .cannotFindHost:
                return "connection failed: nothing is listening at \(host) — "
                    + "is the server running? (default port 8080; start it from "
                    + "the project root)"
            case .timedOut:
                return "connection failed: \(host) timed out"
            case .userAuthenticationRequired:
                return "connection failed: \(host) requires a bearer token"
            case .notConnectedToInternet, .networkConnectionLost:
                return "connection failed: network unavailable"
            default:
                return "connection failed: \(urlError.localizedDescription)"
            }
        }
        return "connection failed: \(error.localizedDescription)"
    }

    /// Refresh the server's model/job state (e.g. after a remote generation).
    /// Also updates the active server's running-job count (via `remoteState`)
    /// and re-fetches the pairing info (a restarted server may report a
    /// different artifact root); a failed info fetch keeps the last value.
    public func refreshRemoteState() async {
        guard let client else { return }
        // Keep the last-known state on a failed fetch (like `remoteInfo`
        // below): /api/state proxies to the GPU-session worker, so a
        // transient failure while a session starts (or its worker is
        // unreachable) used to NIL the whole inventory and render a
        // successful model install as "no models installed".
        if let state = try? await client.state() {
            remoteState = state
        }
        if let info = try? await client.serverInfo() {
            remoteInfo = info
        }
    }

    // MARK: Workspace pairing

    /// Pairing verdict for the active server against the app's DATA
    /// workspace: `.paired` when the server's artifact root IS the active
    /// local workspace (realpath comparison), `.unpaired` when server-side
    /// authoring/builds/runs land somewhere else (its root is carried for
    /// display), `.unknown` before the root is known. nil in the Local
    /// workspace — pairing is a server-workspace concern.
    public var activeServerPairing: WorkspaceScoping.ServerPairing? {
        guard case .server = activeWorkspace else { return nil }
        return WorkspaceScoping.serverPairing(
            localWorkspacePath: VectorCatalog.projectRoot
                .resolvingSymlinksInPath().path,
            serverRoot: remoteInfo?.root,
            rootLooksLikeSourceCheckout: remoteInfo?.rootLooksLikeSourceCheckout,
            // Path comparison is meaningful only on this machine: an SSH
            // cluster's /scratch/… reads as remote-authoritative, never as a
            // mismatch to "repair".
            serverSharesLocalFilesystem: activeServerSharesLocalFilesystem)
    }

    /// Full standing-warning line for an unpaired active server (nil when
    /// paired/unknown/local) — shared by the Compute section header and the
    /// toolbar indicator's help text.
    public var activeServerPairingWarning: String? {
        activeServerPairing.flatMap(WorkspaceScoping.serverPairingWarning)
    }

    /// Compact toolbar badge for an unpaired active server (nil otherwise).
    public var activeServerPairingBadge: String? {
        activeServerPairing.flatMap(WorkspaceScoping.serverPairingBadge)
    }

    /// Informative (non-warning) line naming a remote server's authoritative
    /// workspace — shown where the unpaired warning would have been.
    public var activeServerPairingDescription: String? {
        activeServerPairing.flatMap(WorkspaceScoping.serverPairingDescription)
    }

    /// The active server's serving root (the workspace it was started with
    /// via `serve --root`), from `/api/info` (or the capabilities fallback).
    /// Nil in the Local workspace or before the root is known.
    public var activeServerServingRoot: String? {
        guard case .server = activeWorkspace else { return nil }
        return remoteInfo?.root
    }

    /// THE artifact-list scoping verdict for the active target — every panel
    /// that lists per-substrate artifacts (agents, runs, robustness targets,
    /// optimizations, vector catalogs) renders through this one rule.
    public var artifactListPresentation: WorkspaceScoping.ArtifactListPresentation {
        WorkspaceScoping.artifactListPresentation(
            workspaceIsServer: activeWorkspace != .local,
            pairing: activeServerPairing)
    }

    /// Prominent mismatch-banner text when the app's selected workspace is
    /// NOT the tree the active server serves (nil when paired/unknown/local).
    public var workspaceMismatchBanner: String? {
        WorkspaceScoping.workspaceMismatchBanner(pairing: activeServerPairing)
    }

    /// Title for a server-scoped artifact list, naming the serving workspace.
    public func serverArtifactListTitle(kind: String) -> String {
        WorkspaceScoping.serverArtifactListTitle(
            kind: kind, serverName: substrateLabel, pairing: activeServerPairing)
    }

    // MARK: Server workspace switching (runtime `serve --root` repoint)

    /// UserDefaults key prefix for per-server recent serving roots (keyed by
    /// the same registry identity as the entry itself, so SSH sites keep
    /// their recents across tunnel-local port changes).
    public static let recentServerRootsDefaultsKeyPrefix =
        "SteerLabServerRecentWorkspaceRoots:"
    static let recentServerRootsCap = 8

    /// Registered server-scope cache invalidations (e.g. SubstrateCatalog's
    /// remote vector list), fired after a successful workspace switch — the
    /// server's catalogs now enumerate a different tree, so every cached
    /// server-scoped listing is stale at once.
    private var serverScopeInvalidationHandlers: [@MainActor () -> Void] = []

    public func onServerScopeInvalidated(_ handler: @escaping @MainActor () -> Void) {
        serverScopeInvalidationHandlers.append(handler)
    }

    /// The active server supports AND permits runtime workspace switching
    /// (capability `workspace.switch` + the deployment's live policy
    /// verdict). False before connect and on older servers — affordances
    /// stay hidden rather than relaying a guaranteed refusal.
    public var activeServerSupportsWorkspaceSwitch: Bool {
        guard case .server = activeWorkspace else { return false }
        return capabilities?.supportsWorkspaceSwitch == true
            && capabilities?.workspaceSwitchAllowed == true
    }

    /// Transport gate for offering the APP's workspace path as a server
    /// root: only when the server demonstrably runs on THIS machine —
    /// direct transport to a loopback host. An SSH site's effective URL is
    /// also loopback (the tunnel's local label), so the transport must be
    /// checked, not just the host; a direct URL to a remote host is a
    /// different filesystem. Never send a Mac path to a remote host.
    nonisolated static func serverSharesLocalFilesystem(
        site: ClusterSiteProfile, urlString: String
    ) -> Bool {
        guard !site.isSSHTransport else { return false }
        let trimmed = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = endpointURL(from: trimmed),
            let host = url.host()?.lowercased()
        else { return false }
        return ["127.0.0.1", "localhost", "::1"].contains(host)
    }

    public var activeServerSharesLocalFilesystem: Bool {
        guard let activeServer else { return false }
        return Self.serverSharesLocalFilesystem(
            site: activeServer.resolvedSite, urlString: activeServer.urlString)
    }

    /// The app's data-workspace path, offerable as a serving root only for a
    /// same-machine server (same realpath'd form the pairing check uses, so
    /// a switch to it lands `.paired`). Nil when the server is remote.
    public var localWorkspaceRootForServerSwitch: String? {
        guard activeServerSharesLocalFilesystem else { return nil }
        return VectorCatalog.projectRoot.resolvingSymlinksInPath().path
    }

    /// Most-recent-first serving roots this app has switched the server to
    /// (persisted per server registry identity).
    public func recentServerWorkspaceRoots(for entry: ServerEntry) -> [String] {
        defaults.stringArray(
            forKey: Self.recentServerRootsDefaultsKeyPrefix
                + Self.registryKey(forEntry: entry)) ?? []
    }

    func noteServerWorkspaceRoot(_ root: String, for entry: ServerEntry) {
        var roots = recentServerWorkspaceRoots(for: entry).filter { $0 != root }
        roots.insert(root, at: 0)
        defaults.set(
            Array(roots.prefix(Self.recentServerRootsCap)),
            forKey: Self.recentServerRootsDefaultsKeyPrefix
                + Self.registryKey(forEntry: entry))
    }

    /// Server-side switch candidates for the active server: the site
    /// profile's declared workspace root first (for SSH sites this is the
    /// operator-declared `storageRoots.workspace`), then the recents —
    /// deduplicated, minus the current serving root (a no-op switch). These
    /// are SERVER-side paths, safe to offer for any transport.
    public var serverWorkspaceSwitchCandidates: [String] {
        guard let activeServer else { return [] }
        var out: [String] = []
        if let siteRoot = activeServer.resolvedSite.constraints.storageRoots["workspace"],
            !siteRoot.isEmpty
        {
            out.append(siteRoot)
        }
        out += recentServerWorkspaceRoots(for: activeServer)
        var seen = Set<String>()
        return out.filter { seen.insert($0).inserted && $0 != activeServerServingRoot }
    }

    /// The one affordance verdict for the mismatch banner / Compute menu
    /// (pure rule in `WorkspaceScoping`, gated on capability + transport).
    public var workspaceSwitchAffordance: WorkspaceScoping.WorkspaceSwitchAffordance {
        WorkspaceScoping.workspaceSwitchAffordance(
            pairing: activeServerPairing,
            supportsSwitch: activeServerSupportsWorkspaceSwitch,
            sharesLocalFilesystem: activeServerSharesLocalFilesystem,
            localWorkspaceRoot: localWorkspaceRootForServerSwitch,
            serverSideCandidates: serverWorkspaceSwitchCandidates)
    }

    /// Ask the active server to repoint its serving root, then refresh the
    /// whole scoped view through the ONE existing path (`connect()`), so
    /// pairing, capabilities, state, and variants flip together. Server
    /// refusals (busy jobs, containment) surface verbatim in `status`.
    @discardableResult
    public func switchServerWorkspace(to root: String) async -> Bool {
        guard case .server = activeWorkspace, let client else {
            status = "no active server workspace to switch"
            return false
        }
        do {
            status = "switching server workspace to \(root)..."
            let info = try await client.switchWorkspace(toRoot: root)
            applyServerWorkspaceSwitch(info: info)
            _ = await connect()
            status = "server now serving \(info.root ?? root)"
            return true
        } catch ClusterClient.ClientError.badResponse(let code, let body) {
            status = "workspace switch refused (\(code)): \(Self.errorDetail(from: body))"
            return false
        } catch {
            status = "workspace switch failed: \(error.localizedDescription)"
            return false
        }
    }

    /// Post-switch bookkeeping, factored from the network call so the state
    /// contract is testable without a live transport: the reported root
    /// becomes the pairing truth immediately (every scoped panel recomputes
    /// off `remoteInfo`), the root lands in the per-server recents, and all
    /// registered server-scope cache invalidations fire.
    func applyServerWorkspaceSwitch(info: RemoteServerInfo) {
        remoteInfo = info
        if let root = info.root, let activeServer {
            noteServerWorkspaceRoot(root, for: activeServer)
        }
        for handler in serverScopeInvalidationHandlers { handler() }
    }

    // MARK: Same-machine workspace synchronization (the auto-switch coordinator)

    /// The one in-flight auto-synchronization attempt. Stored so re-entry
    /// cancels and replaces it — two attempts can never interleave.
    private var workspaceSynchronizationTask: Task<Void, Never>?

    /// Persistent notice feed for refused/failed auto-switches: the refusal
    /// must outlive the single-slot `status` line (never silently dropped).
    /// Injectable for tests; the app uses the shared per-workspace feed.
    public var notices: PanelNotices = .shared

    /// Post-switch artifact refresh, wired by `ChatService` (vector catalog +
    /// adapters + stored agents). The switch itself already reconnected
    /// (`switchServerWorkspace` runs `connect()` internally), so this closure
    /// is the ONLY additional refresh an auto-switch performs — never a
    /// second connect.
    var workspaceSyncArtifactRefresh: (@MainActor () async -> Void)?

    /// The last (server, root) pairing the artifact refresh ran for.
    /// Refresh-once rule: the refresh fires when the coordinator lands on a
    /// pairing it has NOT refreshed yet — including the already-serving
    /// no-switch path (reconnecting to an already-correct workspace used to
    /// return early and leave the vector/adapter/agent catalogs stale) —
    /// and never again for the same pairing, so repeated root-change events
    /// stay churn-free.
    private struct WorkspaceSyncPairing: Equatable {
        var server: ServerEntry.ID
        var root: String
    }
    private var workspaceSyncLastRefreshedPairing: WorkspaceSyncPairing?

    // Test seams (nil = live behavior). The root provider backs BOTH the
    // captured target and every stale-guard re-check, so tests can move the
    // "current workspace root" between suspension points without touching
    // process-global workspace state.
    var workspaceSyncLocalRootOverride: (@MainActor () -> String?)?
    var workspaceSyncConnectOverride: (@MainActor () async -> Bool)?
    var workspaceSyncSwitchOverride: (@MainActor (String) async -> Bool)?

    /// The auto-switch target: the app's data-workspace root in the exact
    /// form the pairing check uses (`localWorkspaceRootForServerSwitch`,
    /// realpath'd). Never re-derived at call sites — form drift between the
    /// switch argument and the pairing comparison is how a successful switch
    /// used to land "mismatched".
    private var workspaceSyncLocalRoot: String? {
        if let workspaceSyncLocalRootOverride {
            return workspaceSyncLocalRootOverride()
        }
        return localWorkspaceRootForServerSwitch
    }

    /// Keep a SAME-MACHINE server's serving root following the app's data
    /// workspace. Triggered from the one workspace-root-change observation
    /// (`WorkspaceStore.onRootChange`, wired once by the app) — picker
    /// New/Open, programmatic `switchTo`, and the launch-restored root all
    /// funnel here; no view re-implements the policy.
    ///
    /// Rules:
    /// - Only when the affordance verdict offers the same-machine switch
    ///   (`workspaceSwitchAffordance` → `.pointServerAtLocalWorkspace`).
    ///   Remote servers — SSH sites or direct URLs to another host — stay
    ///   explicit: a Mac path never auto-travels.
    /// - Serialized: one stored cancel-and-replace task. Re-entry cancels the
    ///   in-flight attempt and AWAITS its unwind before starting, so two
    ///   switch requests can never interleave; after every await the current
    ///   root is re-checked against the captured target (stale guard), so a
    ///   superseded attempt neither switches nor refreshes — last-write-wins
    ///   is closed even across cancellation gaps.
    /// - On success, exactly one refresh path: the switch already
    ///   reconnected, so only the artifact refresh runs afterwards.
    /// - A refusal or failure (busy jobs, containment, network) surfaces via
    ///   `status` AND the notices feed — never silently dropped.
    @discardableResult
    public func synchronizeServerToLocalWorkspace() -> Task<Void, Never> {
        let previous = workspaceSynchronizationTask
        previous?.cancel()
        let task = Task { [weak self] in
            // Serialize behind the superseded attempt: it fully unwinds
            // before we read state, so its switch/connect can never land
            // after ours.
            await previous?.value
            guard !Task.isCancelled else { return }
            await self?.runWorkspaceSynchronization()
        }
        workspaceSynchronizationTask = task
        return task
    }

    private func runWorkspaceSynchronization() async {
        // Same-machine gate first: remote servers keep the explicit
        // banner/menu affordances only.
        guard case .server(let serverID) = activeWorkspace,
            activeServerSharesLocalFilesystem,
            let target = workspaceSyncLocalRoot
        else { return }
        // A disconnected server has no capability/pairing record yet —
        // connect once to make the same guarded decision (a precondition
        // probe, not the post-switch refresh).
        if capabilities == nil {
            let connected: Bool
            if let workspaceSyncConnectOverride {
                connected = await workspaceSyncConnectOverride()
            } else {
                connected = await connect()
            }
            // Stale guard: superseded, or the root moved on while we
            // connected — the newer attempt owns the newer target.
            guard !Task.isCancelled, workspaceSyncLocalRoot == target else { return }
            guard connected else {
                surfaceWorkspaceSyncProblem()
                return
            }
        }
        // Already serving the target — no switch, but the FIRST landing on
        // this (server, root) pairing must still run the registered artifact
        // refresh: a reconnect to an already-correct workspace otherwise
        // leaves the vector/adapter/agent catalogs stale. Same pairing
        // repeatedly = no-op (no per-event churn).
        guard activeServerServingRoot != target else {
            await runWorkspaceSyncRefreshOnce(server: serverID, root: target)
            return
        }
        // The one affordance verdict: only a confirmed mismatch on a
        // switch-capable same-machine server offers the local root.
        // Unsupported/unknown stays silent — affordances (and their
        // automation) hide rather than relay a guaranteed refusal.
        guard case .pointServerAtLocalWorkspace = workspaceSwitchAffordance else {
            return
        }
        let switched: Bool
        if let workspaceSyncSwitchOverride {
            switched = await workspaceSyncSwitchOverride(target)
        } else {
            switched = await switchServerWorkspace(to: target)
        }
        // Superseded mid-switch: the newer attempt owns the outcome, and a
        // cancellation-induced failure is not a refusal worth a notice.
        guard !Task.isCancelled else { return }
        guard switched else {
            surfaceWorkspaceSyncProblem()
            return
        }
        // Stale guard on the success side: the root moved again while we
        // switched — the attempt already queued behind us refreshes against
        // ITS target.
        guard workspaceSyncLocalRoot == target else { return }
        workspaceSyncLastRefreshedPairing = WorkspaceSyncPairing(
            server: serverID, root: target)
        await workspaceSyncArtifactRefresh?()
    }

    /// Fire the registered artifact refresh for a pairing at most once.
    /// The pairing is recorded BEFORE the await so a re-entrant event for
    /// the same pairing no-ops instead of doubling an in-flight refresh.
    private func runWorkspaceSyncRefreshOnce(
        server: ServerEntry.ID, root: String
    ) async {
        let pairing = WorkspaceSyncPairing(server: server, root: root)
        guard workspaceSyncLastRefreshedPairing != pairing else { return }
        workspaceSyncLastRefreshedPairing = pairing
        await workspaceSyncArtifactRefresh?()
    }

    /// A refused/failed auto-switch reaches the operator twice: `status`
    /// (already set verbatim by the failing step) and the persistent feed.
    private func surfaceWorkspaceSyncProblem() {
        notices.record(
            source: "Workspace", severity: .warning,
            message: status ?? "server workspace synchronization failed")
    }

    /// Re-fetch the active server's variant list (e.g. after an upload made
    /// elsewhere, or when the Variants panel asks for fresh state). No-op in
    /// the Local workspace — inactive servers are never polled.
    public func refreshRemoteVariants() async {
        guard case .server = activeWorkspace, let client else { return }
        do {
            remoteVariants = try await client.variants()
            await syncRemoteAgentRecipesIntoLibrary()
        } catch {
            status = "could not list server variants: \(error.localizedDescription)"
        }
    }

    /// How many agent recipes the last variant refresh copied into the
    /// local library (display: the sync should be visible, not silent).
    public private(set) var lastAgentSyncImportedCount = 0

    /// THE WORKSPACE IS THE SOURCE OF TRUTH (design directive 2026-08-03):
    /// cluster storage is compute-side scratch that can be PURGED, so an
    /// agent recipe existing only server-side is one purge away from gone —
    /// and it was also unselectable in local study authoring (the
    /// two-libraries field report). Connection itself reconciles: every
    /// server-resident agent recipe not already in the local library is
    /// copied in automatically, server-absolute vector references
    /// normalized to workspace-relative on the way. There is deliberately
    /// NO import button — from the researcher's perspective there is one
    /// local data structure, and the cluster is where compute happens.
    /// (Vector artifact BYTES are heavier and ride the results/evidence
    /// import path; the recipe carries the workspace-relative reference
    /// either engine resolves under its own root.)
    private func syncRemoteAgentRecipesIntoLibrary() async {
        guard let client else { return }
        let localArtifacts = ModelVariantStore.scan().map(\.artifact)
        var imported = 0
        // Failures are REPORTED, never swallowed (review 2026-08-03, P1):
        // a sync that silently skips a recipe leaves the researcher
        // believing the library is complete when it is not.
        var failures: [String] = []
        for record in remoteVariants {
            do {
                let detail = try await client.variantDetail(path: record.path)
                let artifact = normalizedToWorkspaceRelative(detail.variant)
                if localArtifacts.contains(artifact) { continue }
                _ = try ModelVariantStore.save(artifact)
                imported += 1
            } catch {
                failures.append("\(record.name): \(error.localizedDescription)")
            }
        }
        lastAgentSyncImportedCount = imported
        if !failures.isEmpty {
            status = "agent sync: \(failures.count) recipe\(failures.count == 1 ? "" : "s") "
                + "failed to import — \(failures[0])"
        } else if imported > 0 {
            status = "synced \(imported) server agent recipe\(imported == 1 ? "" : "s") "
                + "into the local library (the workspace is the source of truth)"
        }
    }

    /// Strip the active server's serving root from absolute artifact
    /// references: the recipe's references must be workspace-relative so
    /// each engine resolves them under its OWN root.
    private func normalizedToWorkspaceRelative(
        _ artifact: ModelVariantArtifact
    ) -> ModelVariantArtifact {
        Self.workspaceRelativeArtifact(
            artifact, servingRoot: activeServerServingRoot)
    }

    /// Pure half of the agent-recipe sync (unit-tested): server-absolute
    /// vector references under `servingRoot` become workspace-relative;
    /// everything else passes through untouched.
    static func workspaceRelativeArtifact(
        _ artifact: ModelVariantArtifact, servingRoot: String?
    ) -> ModelVariantArtifact {
        guard let root = servingRoot, !root.isEmpty else { return artifact }
        let prefix = root.hasSuffix("/") ? root : root + "/"
        func relative(_ path: String) -> String {
            path.hasPrefix(prefix) ? String(path.dropFirst(prefix.count)) : path
        }
        var normalized = artifact
        normalized.injections = normalized.injections.map { injection in
            var updated = injection
            updated.vectorArtifactID = relative(updated.vectorArtifactID)
            return updated
        }
        // EVERY server-root-absolute reference normalizes, not just vectors
        // (review 2026-08-03, P1): adapters and the neutral-PC basis carry
        // paths too, and a recipe half-normalized is half-portable.
        normalized.adapters = normalized.adapters.map { adapter in
            var updated = adapter
            updated.artifactPath = relative(updated.artifactPath)
            updated.adapterDirectory = relative(updated.adapterDirectory)
            return updated
        }
        if let basis = normalized.neutralPCBasisPath {
            normalized.neutralPCBasisPath = relative(basis)
        }
        return normalized
    }

    // MARK: Jobs badge

    /// "1 job" / "2 jobs" for entries with a known non-zero count — the
    /// last-observed value, since inactive servers are never polled.
    public func runningJobsBadge(for id: ServerEntry.ID) -> String? {
        guard let count = runningJobsByServer[id], count > 0 else { return nil }
        return count == 1 ? "1 job" : "\(count) jobs"
    }

    // MARK: Model install

    /// Ask the active server to prefetch a HF repo into its cache as a
    /// durable job. The server's 400 detail (e.g. the MLX family-twin hint)
    /// is surfaced verbatim in `status`.
    public func installModel(_ modelID: String, revision: String? = nil) async {
        guard let client else {
            status = "invalid server URL"
            return
        }
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            status = "enter a model id to install"
            return
        }
        // The one behavior the site's egress declaration routes: a site whose
        // compute nodes are DECLARED offline can only fail a direct install —
        // point at the transfer-host staging path instead of queueing a job
        // that will die on the first HTTP request. `.unknown` proceeds: the
        // install's own error mapping names egress if it turns out missing.
        if activeSite?.constraints.computeEgress == .no {
            status = "this site declares no compute-node internet — stage the "
                + "model from a transfer/xfer host into the shared HF cache "
                + "(runbook Phase 3)"
            return
        }
        do {
            status = "requesting install of \(trimmed)..."
            let jobID = try await client.installModel(trimmed, revision: revision)
            status = "install queued as job \(jobID) — open Compute to stream progress"
            await refreshRemoteState()
        } catch ClusterClient.ClientError.badResponse(let code, let body) {
            status = "install rejected (\(code)): \(Self.errorDetail(from: body))"
        } catch {
            status = "install failed: \(error.localizedDescription)"
        }
    }

    /// FastAPI error bodies are `{"detail": "..."}` — unwrap so the operator
    /// sees the server's message (e.g. the family-twin hint), not JSON.
    static func errorDetail(from body: String) -> String {
        struct Detail: Decodable { var detail: String }
        if let data = body.data(using: .utf8),
            let decoded = try? JSONDecoder().decode(Detail.self, from: data)
        {
            return decoded.detail
        }
        return body
    }

    // MARK: Persistence

    private func persistServers() {
        if let data = try? JSONEncoder().encode(servers) {
            defaults.set(data, forKey: Self.serversDefaultsKey)
        }
    }

    private func persistWorkspace() {
        defaults.set(Self.encodeWorkspace(activeWorkspace), forKey: Self.activeWorkspaceDefaultsKey)
    }

    static func encodeWorkspace(_ workspace: Workspace) -> String {
        switch workspace {
        case .local: "local"
        case .server(let id): "server:\(id.uuidString)"
        }
    }

    /// Parse a persisted workspace, falling back to `.local` when the string
    /// is missing/garbled or names a server no longer in the registry.
    static func parseWorkspace(_ string: String?, servers: [ServerEntry]) -> Workspace {
        guard let string, string.hasPrefix("server:"),
            let id = UUID(uuidString: String(string.dropFirst("server:".count))),
            servers.contains(where: { $0.id == id })
        else { return .local }
        return .server(id)
    }
}
