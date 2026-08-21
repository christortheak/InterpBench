import Foundation
import Testing

@testable import ExperimentKit

/// WS1 store integration: legacy persisted registries (URL-only JSON from
/// before site profiles existed) still load and present as direct-transport
/// sites with their historical Keychain keying intact; SSH sites persist,
/// dedupe by remote identity, and key tokens by host:remotePort so tunnel
/// local-port changes never orphan a token. No networking; no Keychain
/// reads/writes (only key-string assertions).
@MainActor
struct ClusterSiteStoreTests {

    private func freshDefaults(_ name: String) throws -> UserDefaults {
        let suite = "steerlab.tests.cluster-sites.\(name)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    private func sshProfile(named name: String, host: String) -> ClusterSiteProfile {
        ClusterSiteProfile(
            name: name,
            transport: .ssh(host: host, proxyJump: nil, remotePort: 8080, vpnExpected: false),
            topology: .loginDaemon,
            scheduler: .slurm(ClusterSiteProfile.SlurmSiteData()),
            constraints: ClusterSiteProfile.SiteConstraints())
    }

    // MARK: Legacy migration

    @Test func legacyPersistedServersStillLoadAsDirectSites() throws {
        let defaults = try freshDefaults("legacy")
        // Raw pre-profile JSON exactly as older builds wrote it: no "site".
        let legacyJSON = """
            [{"id":"11111111-2222-3333-4444-555555555555",
              "name":"gpu-a","urlString":"http://gpu-a:8080"}]
            """
        defaults.set(Data(legacyJSON.utf8), forKey: ClusterConnectionStore.serversDefaultsKey)

        let store = ClusterConnectionStore(defaults: defaults)
        #expect(store.servers.count == 1)
        let entry = try #require(store.servers.first)
        #expect(entry.name == "gpu-a")
        #expect(entry.site == nil)  // storage untouched until the user edits the site
        let site = entry.resolvedSite
        let url = try #require(URL(string: "http://gpu-a:8080"))
        #expect(site.transport == .direct(baseURL: url))
        #expect(site.topology == .externalServer)
        #expect(site.scheduler == .none)
        #expect(site.constraints == ClusterSiteProfile.SiteConstraints())
        // Keychain keying is unchanged for legacy entries: URL host:port —
        // existing stored tokens keep working with no migration.
        #expect(ClusterConnectionStore.tokenKey(forEntry: entry) == "gpu-a:8080")
        store.activeWorkspace = .server(entry.id)
        #expect(store.tokenKey == "gpu-a:8080")
        #expect(store.activeSite == site)
    }

    // MARK: SSH sites

    @Test func sshSitePersistsAndKeysItsTokenByRemoteIdentity() throws {
        let defaults = try freshDefaults("ssh-persist")
        let store = ClusterConnectionStore(defaults: defaults)
        let entry = store.addSite(.exampleCluster)
        let preferred = try #require(ClusterSiteProfile.exampleCluster.preferredLocalPort)
        #expect(entry.urlString == "http://127.0.0.1:\(preferred)")
        #expect(
            ClusterConnectionStore.tokenKey(forEntry: entry) == "slurm.example.edu:8080")

        // The profile round-trips through UserDefaults persistence.
        let reloaded = ClusterConnectionStore(defaults: defaults)
        #expect(reloaded.servers.first?.site == ClusterSiteProfile.exampleCluster)
        #expect(reloaded.sites.first == ClusterSiteProfile.exampleCluster)

        // A tunnel landing on a different local port relabels the URL but the
        // token key (remote identity) does not move.
        store.activeWorkspace = .server(entry.id)
        store.token = "secret-token"
        store.noteTunnelLocalPort(9123)
        #expect(store.server(id: entry.id)?.urlString == "http://127.0.0.1:9123")
        #expect(store.tokenKey == "slurm.example.edu:8080")
        // In-memory token survives the relabel too (ssh branch keeps it).
        #expect(store.token == "secret-token")
    }

    @Test func noteTunnelLocalPortIgnoresDirectAndLocalWorkspaces() throws {
        let store = ClusterConnectionStore(defaults: try freshDefaults("note-port"))
        let direct = store.addSite(.gpuWorkstation)
        store.activeWorkspace = .server(direct.id)
        store.noteTunnelLocalPort(9999)
        #expect(store.server(id: direct.id)?.urlString == "http://127.0.0.1:8080")
        store.activeWorkspace = .local
        store.noteTunnelLocalPort(9998)  // no active server — no-op
        #expect(store.server(id: direct.id)?.urlString == "http://127.0.0.1:8080")
    }

    @Test func addSiteDedupesByRemoteIdentityAndKeepsCustomNames() throws {
        let store = ClusterConnectionStore(defaults: try freshDefaults("dedupe"))
        // An IMPORTED site (WP5 §4.2: real sites arrive as JSON now, not as
        // shipped presets) added twice is one entry — remote identity, not the
        // tunnel-local URL label, is the key.
        store.addPreset(.exampleCluster)
        store.addPreset(.exampleCluster)
        #expect(store.servers.count == 1)
        // None of the shipped presets IS that site, so all three stay on offer.
        #expect(store.missingPresets.map(\.name) == ClusterSiteProfile.presets.map(\.name))

        let entry = try #require(store.servers.first)
        store.renameServer(id: entry.id, to: "My Cluster")
        store.addPreset(.exampleCluster)  // refresh must not clobber the custom name
        #expect(store.server(id: entry.id)?.name == "My Cluster")
        #expect(store.servers.count == 1)
    }

    /// WP5 Step 12 added a SECOND hostless Slurm template (§6.6). Both ship
    /// with an empty SSH host, so an endpoint-only registry key would make
    /// them the same site and adding one would silently overwrite the other —
    /// `registryIdentity` keys hostless templates by name for exactly this.
    @Test func theTwoHostlessSlurmTemplatesAreDistinctRegistryEntries() throws {
        let store = ClusterConnectionStore(defaults: try freshDefaults("templates"))
        store.addPreset(.genericSlurm)
        store.addPreset(.genericSlurmModules)
        #expect(store.servers.count == 2)
        #expect(
            store.servers.map(\.name)
                == [ClusterSiteProfile.genericSlurm.name,
                    ClusterSiteProfile.genericSlurmModules.name])
        #expect(store.missingPresets.map(\.name) == [ClusterSiteProfile.gpuWorkstation.name])
    }

    @Test func twoSSHSitesSharingALocalLabelNeverMerge() throws {
        let defaults = try freshDefaults("no-merge")
        let store = ClusterConnectionStore(defaults: defaults)
        let a = store.addSite(sshProfile(named: "A", host: "a.test"))
        let b = store.addSite(sshProfile(named: "B", host: "b.test"))
        // Force both entries onto the same tunnel-local URL label.
        store.updateServerURL(id: a.id, urlString: "http://127.0.0.1:9000")
        store.updateServerURL(id: b.id, urlString: "http://127.0.0.1:9000")
        #expect(store.servers.count == 2)
        // The load-time dedupe keys by remote identity, not the local label.
        let reloaded = ClusterConnectionStore(defaults: defaults)
        #expect(reloaded.servers.count == 2)
        // And a plain direct add at that URL is a third, distinct endpoint.
        let direct = store.addServer(name: "", urlString: "http://127.0.0.1:9000")
        #expect(direct.id != a.id && direct.id != b.id)
        #expect(store.servers.count == 3)
    }

    @Test func directSiteURLEditKeepsTheProfileInSync() throws {
        let store = ClusterConnectionStore(defaults: try freshDefaults("direct-sync"))
        let entry = store.addSite(.gpuWorkstation)
        store.updateServerURL(id: entry.id, urlString: "http://10.0.0.5:8080")
        let site = try #require(store.server(id: entry.id)?.site)
        #expect(site.directURLString == "http://10.0.0.5:8080")
        #expect(
            ClusterConnectionStore.tokenKey(
                forEntry: try #require(store.server(id: entry.id))) == "10.0.0.5:8080")
    }

    @Test func updateSiteReplacesTheProfileAndResetsActiveConnectionState() throws {
        let store = ClusterConnectionStore(defaults: try freshDefaults("update-site"))
        let entry = store.addSite(.gpuWorkstation)
        store.activeWorkspace = .server(entry.id)
        store.status = "connected"
        var edited = ClusterSiteProfile.gpuWorkstation
        edited.name = "Bench box"
        edited.transport = .ssh(
            host: "bench.test", proxyJump: nil, remotePort: 8080, vpnExpected: false)
        store.updateSite(id: entry.id, profile: edited)
        #expect(store.server(id: entry.id)?.name == "Bench box")
        #expect(store.status == nil)
        #expect(store.tokenKey == "bench.test:8080")
        let preferred = try #require(edited.preferredLocalPort)
        #expect(store.server(id: entry.id)?.urlString == "http://127.0.0.1:\(preferred)")
    }

    // MARK: Import / export

    @Test func importAndExportRoundTripThroughTheStore() throws {
        let store = ClusterConnectionStore(defaults: try freshDefaults("import-export"))
        let data = try ClusterSiteProfile.exampleCluster.encoded()
        let entry = try store.importSite(from: data)
        #expect(entry.resolvedSite == ClusterSiteProfile.exampleCluster)
        let exported = try #require(try store.exportSite(id: entry.id))
        #expect(try ClusterSiteProfile.decode(from: exported) == ClusterSiteProfile.exampleCluster)
        // Exporting a legacy entry synthesizes the direct-site view.
        let legacy = store.addServer(name: "old", urlString: "http://old.test:8080")
        let legacyData = try #require(try store.exportSite(id: legacy.id))
        let legacySite = try ClusterSiteProfile.decode(from: legacyData)
        #expect(legacySite.directURLString == "http://old.test:8080")
        #expect(legacySite.topology == .externalServer)
    }

    @Test func importRefusesGarbageAndNewerSchemas() throws {
        let store = ClusterConnectionStore(defaults: try freshDefaults("import-refuse"))
        #expect(throws: (any Error).self) {
            try store.importSite(from: Data("not json".utf8))
        }
        let future = Data(
            """
            {"schemaVersion": 99, "name": "X",
             "transport": {"kind": "ssh", "host": "h"}}
            """.utf8)
        #expect(throws: (any Error).self) {
            try store.importSite(from: future)
        }
        #expect(store.servers.isEmpty)
    }
}
