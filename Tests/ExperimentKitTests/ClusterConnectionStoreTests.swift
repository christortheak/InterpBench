import Foundation
import Testing

@testable import ExperimentKit

/// Pure-CPU tests for the workspace-global connection store: UserDefaults
/// persistence of the server registry + active workspace, migration from the
/// old single-URL schema, connection-state reset on switch, per-server job
/// counts, and the compatibility accessors. No networking.
@MainActor
struct ClusterConnectionStoreTests {

    /// A named, wiped defaults suite so tests never touch the real prefs.
    private func freshDefaults(_ name: String) throws -> UserDefaults {
        let suite = "steerlab.tests.cluster-store.\(name)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    private func makeJob(id: String, finished: Bool) -> RemoteJobRecord {
        RemoteJobRecord(
            id: id, kind: "extract", status: finished ? "succeeded" : "running",
            createdAt: 1, startedAt: 1, finishedAt: finished ? 2 : nil,
            result: nil, error: nil, logTail: [], executor: "local",
            executorJobID: nil, cancellationRequested: false)
    }

    private func makeState(jobs: [RemoteJobRecord] = []) -> RemoteState {
        RemoteState(
            models: ["Qwen/Qwen3-4B"], loadedModel: nil, loadedRevision: nil,
            device: nil, isBusy: false, loadedModels: [], jobs: jobs)
    }

    @Test func defaultsWhenNothingPersisted() throws {
        let store = clusterStore(defaults: try freshDefaults("empty"))
        #expect(store.servers.isEmpty)
        #expect(store.activeWorkspace == .local)
        #expect(store.computeTarget == .local)
        #expect(store.serverURL == ClusterConnectionStore.defaultServerURL)
        #expect(store.substrateLabel == "Local (MLX)")
    }

    @Test func registryRoundTripsAcrossInstances() throws {
        let defaults = try freshDefaults("registry-roundtrip")
        let store = clusterStore(defaults: defaults)
        let a = store.addServer(name: "Cluster A", urlString: "http://gpu-a:8080")
        let b = store.addServer(name: "", urlString: "http://gpu-b:8443")
        #expect(b.name == "gpu-b:8443")  // empty name defaults to host label
        store.activeWorkspace = .server(b.id)

        let reloaded = clusterStore(defaults: defaults)
        // Identity, order, and labels survive the round trip through the
        // canonical registry. Entries do NOT come back byte-identical: a
        // registry file holds a PROFILE, so a URL-only entry reads back as the
        // direct-transport site it always was (`resolvedSite`) rather than as
        // `site == nil`.
        #expect(reloaded.servers.map(\.id) == [a.id, b.id])
        #expect(reloaded.servers.map(\.name) == [a.name, b.name])
        #expect(reloaded.servers.map(\.urlString) == [a.urlString, b.urlString])
        #expect(reloaded.servers.map(\.resolvedSite) == [a.resolvedSite, b.resolvedSite])
        #expect(reloaded.activeWorkspace == .server(b.id))
        #expect(reloaded.computeTarget == .server)
        #expect(reloaded.serverURL == "http://gpu-b:8443")
        #expect(reloaded.substrateLabel == "gpu-b:8443")
        #expect(reloaded.serverHostLabel == "gpu-b:8443")
    }

    @Test func addServerReusesSameEndpoint() throws {
        let store = clusterStore(defaults: try freshDefaults("dedupe-add"))
        let first = store.addServer(name: "localhost", urlString: "http://127.0.0.1:8080")
        let second = store.addServer(name: "", urlString: " http://127.0.0.1:8080 ")
        let third = store.addServer(name: "", urlString: "127.0.0.1:8080")
        #expect(first.id == second.id)
        #expect(first.id == third.id)
        #expect(store.servers.count == 1)
        #expect(store.servers.first?.name == "localhost")
    }

    @Test func persistedDuplicateServersAreCollapsedOnLoadAndKeepActiveAlias() throws {
        let defaults = try freshDefaults("dedupe-load")
        let a = ClusterConnectionStore.ServerEntry(
            name: "localhost", urlString: "http://127.0.0.1:8080")
        let b = ClusterConnectionStore.ServerEntry(
            name: "duplicate", urlString: "http://127.0.0.1:8080")
        defaults.set(
            try JSONEncoder().encode([a, b]),
            forKey: ClusterConnectionStore.serversDefaultsKey)
        defaults.set(
            ClusterConnectionStore.encodeWorkspace(.server(b.id)),
            forKey: ClusterConnectionStore.activeWorkspaceDefaultsKey)

        let store = clusterStore(defaults: defaults)
        #expect(store.servers.map(\.id) == [a.id])
        #expect(store.servers.map(\.name) == [a.name])
        // The loser's UUID still resolves: the migration recorded it as an
        // alias, so the researcher's active server survives the collapse.
        #expect(store.activeWorkspace == .server(a.id))
    }

    @Test func migratesLegacySingleServerSchema() throws {
        let defaults = try freshDefaults("migration-server")
        defaults.set("http://gpu-node:8443", forKey: ClusterConnectionStore.serverURLDefaultsKey)
        defaults.set("Server", forKey: ClusterConnectionStore.computeTargetDefaultsKey)

        let store = clusterStore(defaults: defaults)
        #expect(store.servers.count == 1)
        let entry = try #require(store.servers.first)
        #expect(entry.urlString == "http://gpu-node:8443")
        #expect(entry.name == "gpu-node:8443")  // named by its host
        #expect(store.activeWorkspace == .server(entry.id))
        #expect(store.computeTarget == .server)

        // Migration is one-shot: the new schema is stamped, so a reload no
        // longer consults the legacy keys.
        defaults.removeObject(forKey: ClusterConnectionStore.serverURLDefaultsKey)
        let reloaded = clusterStore(defaults: defaults)
        #expect(reloaded.servers.map(\.id) == [entry.id])
        #expect(reloaded.servers.map(\.urlString) == [entry.urlString])
        #expect(reloaded.activeWorkspace == .server(entry.id))
    }

    @Test func migrationPreservesLocalComputeTarget() throws {
        let defaults = try freshDefaults("migration-local")
        defaults.set("http://gpu-node:8443", forKey: ClusterConnectionStore.serverURLDefaultsKey)
        defaults.set("Local", forKey: ClusterConnectionStore.computeTargetDefaultsKey)

        let store = clusterStore(defaults: defaults)
        #expect(store.servers.count == 1)
        #expect(store.activeWorkspace == .local)
        #expect(store.computeTarget == .local)
    }

    @Test func persistedWorkspaceNamingAMissingServerFallsBackToLocal() throws {
        let defaults = try freshDefaults("stale-workspace")
        let store = clusterStore(defaults: defaults)
        let entry = store.addServer(name: "A", urlString: "http://gpu-a:8080")
        store.activeWorkspace = .server(entry.id)
        // The site leaves the registry — deleted on another machine and
        // pulled, say — while the app's persisted workspace string survives.
        let siteID = try #require(entry.siteID)
        try store.siteRegistry.remove(id: siteID)

        let reloaded = clusterStore(defaults: defaults)
        #expect(reloaded.servers.isEmpty)
        #expect(reloaded.activeWorkspace == .local)
    }

    @Test func switchingWorkspaceResetsConnectionStateButKeepsJobCounts() throws {
        let store = clusterStore(defaults: try freshDefaults("switch-reset"))
        let entry = store.addServer(name: "A", urlString: "http://gpu-a:8080")
        store.activeWorkspace = .server(entry.id)
        store.status = "connected"
        store.remoteState = makeState(jobs: [
            makeJob(id: "j1", finished: false),
            makeJob(id: "j2", finished: false),
            makeJob(id: "j3", finished: true),
        ])
        store.remoteVariants = []
        store.token = "secret"
        #expect(store.runningJobsByServer[entry.id] == 2)

        store.activeWorkspace = .local
        #expect(store.status == nil)
        #expect(store.capabilities == nil)
        #expect(store.remoteState == nil)
        #expect(store.remoteVariants.isEmpty)
        #expect(store.token.isEmpty)
        // The badge is a memory of the last check, not live state.
        #expect(store.runningJobsByServer[entry.id] == 2)
        #expect(store.runningJobsBadge(for: entry.id) == "2 jobs")
    }

    @Test func jobsBadgeFormatting() throws {
        let store = clusterStore(defaults: try freshDefaults("badge"))
        let entry = store.addServer(name: "A", urlString: "http://gpu-a:8080")
        #expect(store.runningJobsBadge(for: entry.id) == nil)  // never checked
        store.activeWorkspace = .server(entry.id)
        store.remoteState = makeState(jobs: [makeJob(id: "j1", finished: false)])
        #expect(store.runningJobsBadge(for: entry.id) == "1 job")
        store.remoteState = makeState(jobs: [makeJob(id: "j1", finished: true)])
        #expect(store.runningJobsBadge(for: entry.id) == nil)  // zero hides the badge
        #expect(store.runningJobsByServer[entry.id] == 0)
    }

    @Test func computeTargetCompatibilityAccessor() throws {
        let store = clusterStore(defaults: try freshDefaults("compat"))
        // No servers: selecting .server has nothing to activate.
        store.computeTarget = .server
        #expect(store.activeWorkspace == .local)

        let a = store.addServer(name: "A", urlString: "http://gpu-a:8080")
        let b = store.addServer(name: "B", urlString: "http://gpu-b:8080")
        store.computeTarget = .server
        #expect(store.activeWorkspace == .server(a.id))  // first, none was active yet

        // Round-trip through .local reactivates the last active server.
        store.activeWorkspace = .server(b.id)
        store.computeTarget = .local
        #expect(store.activeWorkspace == .local)
        store.computeTarget = .server
        #expect(store.activeWorkspace == .server(b.id))
    }

    @Test func renameAndUpdateURL() throws {
        let store = clusterStore(defaults: try freshDefaults("rename"))
        let entry = store.addServer(name: "A", urlString: "http://gpu-a:8080")
        store.renameServer(id: entry.id, to: "Big Cluster")
        #expect(store.server(id: entry.id)?.name == "Big Cluster")
        store.renameServer(id: entry.id, to: "   ")
        #expect(store.server(id: entry.id)?.name == "gpu-a:8080")  // blank → host label

        store.activeWorkspace = .server(entry.id)
        store.status = "connected"
        store.updateServerURL(id: entry.id, urlString: "http://gpu-a:9090")
        #expect(store.server(id: entry.id)?.urlString == "http://gpu-a:9090")
        #expect(store.status == nil)  // new endpoint → stale connection state
        #expect(store.tokenKey == "gpu-a:9090")
    }

    @Test func removingActiveServerFallsBackToLocal() throws {
        let store = clusterStore(defaults: try freshDefaults("remove"))
        let entry = store.addServer(
            name: "A", urlString: "http://steerlab-tests-nonexistent-host:8080")
        store.activeWorkspace = .server(entry.id)
        store.removeServer(id: entry.id)
        #expect(store.servers.isEmpty)
        #expect(store.activeWorkspace == .local)
        #expect(store.runningJobsByServer[entry.id] == nil)
    }

    @Test func tokenKeyFollowsActiveServer() throws {
        let store = clusterStore(defaults: try freshDefaults("token-key"))
        let a = store.addServer(name: "A", urlString: "http://localhost:8000")
        let b = store.addServer(name: "B", urlString: "https://ood.test")
        store.activeWorkspace = .server(a.id)
        #expect(store.tokenKey == "localhost:8000")
        store.activeWorkspace = .server(b.id)
        #expect(store.tokenKey == "ood.test")
    }

    @Test func clientRequiresParseableURL() throws {
        let store = clusterStore(defaults: try freshDefaults("client"))
        let entry = store.addServer(name: "A", urlString: "")
        store.activeWorkspace = .server(entry.id)
        #expect(store.client == nil)
        store.updateServerURL(id: entry.id, urlString: "http://127.0.0.1:8000")
        #expect(store.client != nil)
    }

    @Test func errorDetailUnwrapsFastAPIBodies() {
        #expect(
            ClusterConnectionStore.errorDetail(
                from: #"{"detail":"'Qwen/Qwen3-4B-MLX-4bit' is an MLX-quantized repo"}"#)
                == "'Qwen/Qwen3-4B-MLX-4bit' is an MLX-quantized repo")
        #expect(ClusterConnectionStore.errorDetail(from: "plain text") == "plain text")
    }

    @Test func tunneledNetworkFailureNamesTheMissingController() {
        let message = ClusterConnectionStore.friendlyConnectionFailure(
            URLError(.notConnectedToInternet), urlString: "http://127.0.0.1:8700",
            throughSSHTunnel: true)
        #expect(
            message
                == "connection failed: the SSH tunnel opened, but the SteerLab controller "
                    + "is not answering behind it — start or reconnect the controller "
                    + "batch job and check its serverd.host record")

        let direct = ClusterConnectionStore.friendlyConnectionFailure(
            URLError(.notConnectedToInternet), urlString: "http://127.0.0.1:8080")
        #expect(direct == "connection failed: network unavailable")
    }

    @Test func workspaceStringCodec() {
        let id = UUID()
        let servers = [
            ClusterConnectionStore.ServerEntry(id: id, name: "A", urlString: "http://a:1")
        ]
        let encoded = ClusterConnectionStore.encodeWorkspace(.server(id))
        #expect(ClusterConnectionStore.parseWorkspace(encoded, servers: servers) == .server(id))
        #expect(ClusterConnectionStore.parseWorkspace("local", servers: servers) == .local)
        #expect(ClusterConnectionStore.parseWorkspace(nil, servers: servers) == .local)
        #expect(ClusterConnectionStore.parseWorkspace("server:not-a-uuid", servers: servers) == .local)
    }

    // MARK: Serving-root surfacing + the one artifact-list scoping rule

    @Test func servingRootAndScopingConveniencesFollowTheReportedRoot() throws {
        let store = clusterStore(defaults: try freshDefaults("serving-root"))
        // Local workspace: no server scope, no banner, local-only lists.
        #expect(store.activeServerServingRoot == nil)
        #expect(store.artifactListPresentation == .localOnly)
        #expect(store.workspaceMismatchBanner == nil)

        // Same-machine server (loopback): path comparison is meaningful, so
        // mismatch/paired verdicts apply.
        let entry = store.addServer(name: "local-a", urlString: "http://127.0.0.1:9911")
        store.activeWorkspace = .server(entry.id)

        // Connected but no root reported yet (older server / info failed):
        // server lists show, labeled "serving root unknown", no false alarm.
        #expect(store.activeServerServingRoot == nil)
        #expect(store.artifactListPresentation == .serverAuthoritative(mismatch: false))
        #expect(store.workspaceMismatchBanner == nil)
        #expect(
            store.serverArtifactListTitle(kind: "Agents")
                == "Agents — local-a (serving root unknown)")

        // A reported root that differs from the app's workspace: confirmed
        // mismatch — banner text names the root and both remedies.
        store.remoteInfo = RemoteServerInfo(
            service: "steerlab", root: "/srv/other-ws",
            rootLooksLikeSourceCheckout: false)
        #expect(store.activeServerServingRoot == "/srv/other-ws")
        #expect(store.artifactListPresentation == .serverAuthoritative(mismatch: true))
        #expect(
            store.workspaceMismatchBanner
                == "server is serving workspace /srv/other-ws — switch the app "
                + "to it, or restart the server with --root <your workspace>")
        #expect(
            store.serverArtifactListTitle(kind: "Agents")
                == "Agents — local-a, serving /srv/other-ws")

        // Paired (the server serves THIS app workspace): one tree, no banner.
        store.remoteInfo = RemoteServerInfo(
            service: "steerlab",
            root: VectorCatalog.projectRoot.resolvingSymlinksInPath().path,
            rootLooksLikeSourceCheckout: false)
        #expect(store.artifactListPresentation == .serverShared)
        #expect(store.workspaceMismatchBanner == nil)
        #expect(
            store.serverArtifactListTitle(kind: "Agents")
                == "Agents — this workspace (on local-a)")

        // Remote server (non-loopback direct URL, or SSH transport): a Mac
        // path can never equal /scratch — the server's root is simply the
        // authoritative remote workspace, never a mismatch to "repair".
        let remote = store.addServer(name: "gpu-a", urlString: "http://gpu-a:8080")
        store.activeWorkspace = .server(remote.id)
        store.remoteInfo = RemoteServerInfo(
            service: "steerlab", root: "/scratch/me/ws",
            rootLooksLikeSourceCheckout: false)
        #expect(store.artifactListPresentation == .serverAuthoritative(mismatch: false))
        #expect(store.workspaceMismatchBanner == nil)
        #expect(store.activeServerPairingWarning == nil)
        #expect(
            store.activeServerPairingDescription?.contains("/scratch/me/ws") == true)
        #expect(
            store.serverArtifactListTitle(kind: "Agents")
                == "Agents — gpu-a, workspace /scratch/me/ws")
    }

    @Test func capabilitiesRootDecodesForTheInfoFallback() throws {
        // GET /api/capabilities reports the serving root top-level (same
        // value as /api/info) — the connect flow falls back to it when the
        // info fetch fails, so pairing still resolves.
        let json = #"{"serverVersion":"1.0","engine":"python-hf-transformers","root":"/srv/ws"}"#
        let capabilities = try JSONDecoder().decode(
            ClusterCapabilities.self, from: Data(json.utf8))
        #expect(capabilities.root == "/srv/ws")
    }

    // MARK: Runtime server-workspace switching

    private func switchCapabilities(
        switchable: Bool = true, parent: String? = nil
    ) throws -> ClusterCapabilities {
        var workspace = "\"switch\": true, \"switchable\": \(switchable)"
        if let parent { workspace += ", \"parent\": \"\(parent)\"" }
        let json = "{\"serverVersion\": \"1.0\", \"workspace\": {\(workspace)}}"
        return try JSONDecoder().decode(ClusterCapabilities.self, from: Data(json.utf8))
    }

    @Test func serverSharesLocalFilesystemGatesOnTransportAndHost() {
        let direct = ClusterSiteProfile(
            name: "local", transport: .direct(baseURL: URL(string: "http://127.0.0.1:8080")!),
            topology: .externalServer, scheduler: .none,
            constraints: ClusterSiteProfile.SiteConstraints())
        // Direct loopback: the server is this machine — local paths valid.
        #expect(
            ClusterConnectionStore.serverSharesLocalFilesystem(
                site: direct, urlString: "http://127.0.0.1:8080"))
        #expect(
            ClusterConnectionStore.serverSharesLocalFilesystem(
                site: direct, urlString: "http://localhost:8080"))
        // Direct to a REMOTE host: a different filesystem.
        #expect(
            !ClusterConnectionStore.serverSharesLocalFilesystem(
                site: direct, urlString: "http://gpu-a.lab:8080"))
        // SSH site: its effective URL is the tunnel's loopback label, but the
        // server is on the far side — a Mac path must never be offered.
        let ssh = ClusterSiteProfile(
            name: "cluster",
            transport: .ssh(
                host: "login.cluster.edu", proxyJump: nil, remotePort: 8080,
                vpnExpected: false),
            topology: .loginDaemon, scheduler: .none,
            constraints: ClusterSiteProfile.SiteConstraints())
        #expect(
            !ClusterConnectionStore.serverSharesLocalFilesystem(
                site: ssh, urlString: "http://127.0.0.1:8700"))
    }

    @Test func switchAffordanceIsCapabilityGated() throws {
        let store = clusterStore(defaults: try freshDefaults("switch-gate"))
        let entry = store.addServer(name: "local", urlString: "http://127.0.0.1:8080")
        store.activeWorkspace = .server(entry.id)
        store.remoteInfo = RemoteServerInfo(
            service: "steerlab", root: "/srv/other-ws",
            rootLooksLikeSourceCheckout: false)

        // Confirmed mismatch, but no capabilities yet / an older server:
        // no affordance — the banner stays text-only.
        #expect(store.activeServerSupportsWorkspaceSwitch == false)
        #expect(store.workspaceSwitchAffordance == .unavailable)

        // Capability present on a same-machine server: one-click local path.
        store.capabilities = try switchCapabilities()
        #expect(store.activeServerSupportsWorkspaceSwitch)
        let expectedLocal = VectorCatalog.projectRoot.resolvingSymlinksInPath().path
        #expect(store.localWorkspaceRootForServerSwitch == expectedLocal)
        #expect(
            store.workspaceSwitchAffordance
                == .pointServerAtLocalWorkspace(localRoot: expectedLocal))

        // The deployment's live policy verdict says no (non-loopback bind
        // without a parent allowlist): affordances hide rather than relaying
        // a guaranteed 403.
        store.capabilities = try switchCapabilities(switchable: false)
        #expect(store.activeServerSupportsWorkspaceSwitch == false)
        #expect(store.workspaceSwitchAffordance == .unavailable)
    }

    @Test func switchAffordanceOffersOnlyServerSideRootsForSSHSites() throws {
        let store = clusterStore(defaults: try freshDefaults("switch-ssh"))
        var site = ClusterSiteProfile(
            name: "cluster",
            transport: .ssh(
                host: "login.cluster.edu", proxyJump: nil, remotePort: 8080,
                vpnExpected: false),
            topology: .loginDaemon, scheduler: .none,
            constraints: ClusterSiteProfile.SiteConstraints())
        site.constraints.storageRoots["workspace"] = "/scratch/user/steerlab-ws"
        let entry = store.addSite(site)
        store.activeWorkspace = .server(entry.id)
        store.capabilities = try switchCapabilities(parent: "/scratch/user")
        store.remoteInfo = RemoteServerInfo(
            service: "steerlab", root: "/scratch/user/old-ws",
            rootLooksLikeSourceCheckout: false)

        // Never the Mac path over a tunnel; the site's declared workspace
        // root is the candidate.
        #expect(store.localWorkspaceRootForServerSwitch == nil)
        #expect(
            store.workspaceSwitchAffordance
                == .offerServerSideRoots(["/scratch/user/steerlab-ws"]))

        // Recents join the candidates (site root first, then most recent
        // first), and the current serving root is excluded.
        store.noteServerWorkspaceRoot("/scratch/user/ws-b", for: entry)
        store.noteServerWorkspaceRoot("/scratch/user/old-ws", for: entry)
        #expect(
            store.serverWorkspaceSwitchCandidates
                == ["/scratch/user/steerlab-ws", "/scratch/user/ws-b"])
    }

    @Test func recentServerRootsPersistPerServerAndCapAndDedupe() throws {
        let defaults = try freshDefaults("switch-recents")
        let store = clusterStore(defaults: defaults)
        let a = store.addServer(name: "a", urlString: "http://gpu-a:8080")
        let b = store.addServer(name: "b", urlString: "http://gpu-b:8080")

        store.noteServerWorkspaceRoot("/ws/one", for: a)
        store.noteServerWorkspaceRoot("/ws/two", for: a)
        store.noteServerWorkspaceRoot("/ws/one", for: a)  // re-note moves to front
        #expect(store.recentServerWorkspaceRoots(for: a) == ["/ws/one", "/ws/two"])
        // Per-server isolation.
        #expect(store.recentServerWorkspaceRoots(for: b).isEmpty)

        // Cap: oldest entries fall off.
        for i in 0 ..< 12 { store.noteServerWorkspaceRoot("/ws/n\(i)", for: b) }
        let recents = store.recentServerWorkspaceRoots(for: b)
        #expect(recents.count == ClusterConnectionStore.recentServerRootsCap)
        #expect(recents.first == "/ws/n11")

        // Persistence across store instances (same defaults suite).
        let reloaded = clusterStore(defaults: defaults)
        #expect(reloaded.recentServerWorkspaceRoots(for: a) == ["/ws/one", "/ws/two"])
    }

    @Test func applyServerWorkspaceSwitchUpdatesPairingRecentsAndFiresInvalidation() throws {
        // The post-switch contract without a live transport: the reported
        // root becomes the pairing truth, lands in recents, and registered
        // server-scope invalidations fire (SubstrateCatalog registers its
        // remote-vector refetch through this same hook).
        let store = clusterStore(defaults: try freshDefaults("switch-apply"))
        let entry = store.addServer(name: "local", urlString: "http://127.0.0.1:8080")
        store.activeWorkspace = .server(entry.id)
        store.remoteInfo = RemoteServerInfo(
            service: "steerlab", root: "/srv/old-ws", rootLooksLikeSourceCheckout: false)
        #expect(store.artifactListPresentation == .serverAuthoritative(mismatch: true))

        var invalidations = 0
        store.onServerScopeInvalidated { invalidations += 1 }

        let localRoot = VectorCatalog.projectRoot.resolvingSymlinksInPath().path
        store.applyServerWorkspaceSwitch(
            info: RemoteServerInfo(
                service: "steerlab-server", root: localRoot,
                rootLooksLikeSourceCheckout: false))

        #expect(store.activeServerServingRoot == localRoot)
        // Switched to the app's workspace: paired, banner gone, lists shared.
        #expect(store.artifactListPresentation == .serverShared)
        #expect(store.workspaceMismatchBanner == nil)
        #expect(store.recentServerWorkspaceRoots(for: entry) == [localRoot])
        #expect(invalidations == 1)
    }
}
