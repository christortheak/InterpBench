import Foundation
import Testing

@testable import ExperimentKit

/// Registry persistence across relaunch (live cluster defect, 2026-07-17):
/// the researcher's EDITED site (login `user@` on the ssh host, storage
/// roots filled) must survive quit+relaunch. Four contracts:
///
/// (a) today's types round-trip through UserDefaults byte-losslessly;
/// (b) registry JSON from OLDER type shapes (fields added since) decodes
///     leniently — an old entry must never vanish;
/// (c) one corrupt entry costs THAT entry only, never the whole registry
///     (the old all-or-nothing decode + init's re-persist destroyed every
///     saved site whenever any single entry stopped decoding);
/// (d) an edited entry counts as the shipped preset being
///     PRESENT (no re-offer forking the registry), `addPreset` never
///     clobbers the edits, and load-time dedup never prefers a
///     preset-shaped copy over the edited one.
@MainActor
struct ClusterRegistryPersistenceTests {

    private func freshDefaults(_ name: String) throws -> UserDefaults {
        let suite = "steerlab.tests.cluster-registry.\(name)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    /// The researcher's actual shape: the example preset, then Edit Site to put
    /// the login identity on the host and fill the storage roots.
    private func editedExampleCluster() -> ClusterSiteProfile {
        var site = ClusterSiteProfile.exampleCluster
        site.transport = .ssh(
            host: "user@slurm.example.edu", proxyJump: nil,
            remotePort: 8080, vpnExpected: true)
        site.constraints.storageRoots = [
            "workspace": "/scratch/user/steerlab-workspace",
            "hfCache": "/work/lab1/hf-cache",
            "archive": "/project/lab1",
        ]
        return site
    }

    /// The shipped conda template after the researcher configured it: the site
    /// keeps the preset's NAME (nobody renames a site to configure it) while
    /// its host, roots, and inventory become real. This is the modern shape of
    /// "an edited preset" now that presets ship hostless.
    private func configuredGenericSlurm() -> ClusterSiteProfile {
        var site = ClusterSiteProfile.genericSlurm
        site.transport = .ssh(
            host: "user@slurm.example.edu", proxyJump: nil,
            remotePort: 8080, vpnExpected: true)
        site.constraints.storageRoots = [
            "workspace": "/scratch/user/steerlab-workspace",
            "hfCache": "/work/lab1/hf-cache",
        ]
        return site
    }

    /// The same template with only its storage roots filled — still hostless,
    /// so it shares the incumbent's registry key and can collide with a
    /// pristine copy at load time.
    private func partlyEditedGenericSlurm() -> ClusterSiteProfile {
        var site = ClusterSiteProfile.genericSlurm
        site.constraints.storageRoots = ["workspace": "/scratch/user/steerlab-workspace"]
        return site
    }

    // MARK: (e) stale host-label names heal to the site's own name

    @Test func hostLabelNamesHealToTheSiteNameAtLoad() throws {
        // Live 2026-07-19-era registries carry scars from the earlier eating
        // bugs: an entry rebuilt around a tunnel-local URL keeps
        // "127.0.0.1:8726" as its NAME while the profile inside still says
        // "Example HPC" — the picker, export item, and wizard then show the
        // ephemeral port. Load must heal the label; display must prefer the
        // site's name regardless.
        let defaults = try freshDefaults("heal-name")
        let scarred = ClusterConnectionStore.ServerEntry(
            name: "127.0.0.1:8726", urlString: "http://127.0.0.1:8726",
            site: editedExampleCluster())
        defaults.set(
            try JSONEncoder().encode([scarred]),
            forKey: ClusterConnectionStore.serversDefaultsKey)

        let store = clusterStore(defaults: defaults)

        #expect(store.servers.count == 1)
        #expect(store.servers.first?.name == "Example HPC")
        #expect(store.servers.first?.displayName == "Example HPC")
        // A deliberate custom name is NOT clobbered by the heal.
        let custom = ClusterConnectionStore.ServerEntry(
            name: "My Cluster", urlString: "http://127.0.0.1:9000",
            site: editedExampleCluster())
        #expect(custom.displayName == "Example HPC")  // display prefers site
        #expect(custom.name == "My Cluster")          // stored name untouched
    }

    // MARK: (a) today's types round-trip

    @Test func editedSiteSurvivesRelaunchByteForByte() throws {
        let defaults = try freshDefaults("roundtrip")
        let store = clusterStore(defaults: defaults)
        let entry = store.addSite(.exampleCluster)
        store.updateSite(id: entry.id, profile: editedExampleCluster())
        store.activeWorkspace = .server(entry.id)

        let reloaded = clusterStore(defaults: defaults)
        #expect(reloaded.servers.count == 1)
        let survivor = try #require(reloaded.servers.first)
        #expect(survivor.id == entry.id)
        let site = try #require(survivor.site)
        #expect(site == editedExampleCluster())
        #expect(
            site.constraints.storageRoots["hfCache"] == "/work/lab1/hf-cache")
        if case .ssh(let host, _, _, _) = site.transport {
            #expect(host == "user@slurm.example.edu")
        } else {
            Issue.record("edited transport did not survive decode")
        }
        #expect(reloaded.activeWorkspace == .server(entry.id))
    }

    // MARK: (b) older type shapes decode leniently

    @Test func olderRegistryShapesSurviveLenientDecode() throws {
        let defaults = try freshDefaults("older-shape")
        // Hand-authored plausible OLDER persisted JSON: a site entry missing
        // recently added optional profile fields (no scheduler, no
        // constraints, no topology, no bootstrapPath; ssh transport without
        // remotePort/vpnExpected), plus a pre-profile URL-only entry.
        let olderJSON = """
            [{"id":"AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE",
              "name":"Example HPC",
              "urlString":"http://127.0.0.1:8700",
              "site":{"schemaVersion":1,
                      "name":"Example HPC",
                      "transport":{"kind":"ssh","host":"user@slurm.example.edu"}}},
             {"id":"11111111-2222-3333-4444-555555555555",
              "name":"gpu-a","urlString":"http://gpu-a:8080"}]
            """
        defaults.set(Data(olderJSON.utf8), forKey: ClusterConnectionStore.serversDefaultsKey)

        let store = clusterStore(defaults: defaults)
        #expect(store.servers.count == 2)
        let example = try #require(store.servers.first)
        #expect(example.name == "Example HPC")
        let site = try #require(example.site)
        // Absent fields resolve to the documented lenient defaults.
        #expect(site.topology == .externalServer)
        #expect(site.scheduler == .none)
        #expect(site.constraints == ClusterSiteProfile.SiteConstraints())
        if case .ssh(let host, _, let remotePort, _) = site.transport {
            #expect(host == "user@slurm.example.edu")
            #expect(remotePort == 8080)  // lenient default
        } else {
            Issue.record("ssh transport did not survive older-shape decode")
        }
    }

    // MARK: (c) per-entry lenient decode

    @Test func corruptEntryCostsOnlyItself() throws {
        let defaults = try freshDefaults("corrupt-entry")
        let good = ClusterConnectionStore.ServerEntry(
            name: "gpu-a", urlString: "http://gpu-a:8080")
        var editedEntry = ClusterConnectionStore.ServerEntry(
            name: "Example HPC", urlString: "http://127.0.0.1:8700",
            site: editedExampleCluster())
        editedEntry.name = "Example HPC"
        let goodData = try JSONEncoder().encode([good, editedEntry])
        var array = try #require(
            try JSONSerialization.jsonObject(with: goodData) as? [Any])
        // A corrupt middle entry: present `site` whose transport kind is
        // garbage — a strict array decode dies HERE and used to take the
        // whole registry (and, via init's re-persist, the stored copy) down.
        array.insert(
            [
                "id": "99999999-8888-7777-6666-555555555555",
                "name": "broken",
                "urlString": "http://broken:1",
                "site": ["schemaVersion": 1, "name": "broken",
                         "transport": ["kind": "carrier-pigeon"]],
            ] as [String: Any], at: 1)
        defaults.set(
            try JSONSerialization.data(withJSONObject: array),
            forKey: ClusterConnectionStore.serversDefaultsKey)

        let store = clusterStore(defaults: defaults)
        #expect(store.servers.count == 2)
        #expect(store.servers.map(\.name) == ["gpu-a", "Example HPC"])
        #expect(store.servers.last?.site == editedExampleCluster())

        // Direct contract on the decoder: nil only for non-array payloads.
        #expect(
            ClusterConnectionStore.decodeServersLeniently(from: Data("not json".utf8)) == nil)
        #expect(
            ClusterConnectionStore.decodeServersLeniently(from: Data("{}".utf8)) == nil)
    }

    // MARK: (d) preset presence + dedup preference

    /// Since WP5 Step 12 the shipped presets are hostless TEMPLATES (§4.2), so
    /// the researcher's first edit is typing the host — which moves the entry's
    /// endpoint identity. The contract survives on the OTHER half of `matches`:
    /// the preset's NAME. Re-offering the template must still find the
    /// configured entry rather than fork the registry.
    @Test func editedTemplateEntryCountsAsThePresetAndIsNeverClobbered() throws {
        let defaults = try freshDefaults("preset-presence")
        let store = clusterStore(defaults: defaults)
        let entry = store.addSite(.genericSlurm)
        store.updateSite(id: entry.id, profile: configuredGenericSlurm())

        // The configured entry still counts as that preset being PRESENT:
        // no re-offer, so the one-click add can never fork the registry.
        #expect(!store.missingPresets.contains { $0.name == ClusterSiteProfile.genericSlurm.name })

        // And an explicit addPreset returns the edited entry UNTOUCHED.
        let returned = store.addPreset(.genericSlurm)
        #expect(returned.id == entry.id)
        #expect(store.servers.count == 1)
        #expect(store.server(id: entry.id)?.site == configuredGenericSlurm())
    }

    @Test func dedupPrefersTheEditedEntryOverAPresetShapedCopy() throws {
        let defaults = try freshDefaults("dedupe-preference")
        // Simulate the forked registry earlier builds could mint: a pristine
        // preset-shaped copy FIRST, the researcher's edited entry second
        // (same site once the login identity is ignored).
        let presetShaped = ClusterConnectionStore.ServerEntry(
            name: ClusterSiteProfile.genericSlurm.name, urlString: "http://127.0.0.1:8701",
            site: .genericSlurm)
        let edited = ClusterConnectionStore.ServerEntry(
            name: ClusterSiteProfile.genericSlurm.name, urlString: "http://127.0.0.1:8702",
            site: partlyEditedGenericSlurm())
        defaults.set(
            try JSONEncoder().encode([presetShaped, edited]),
            forKey: ClusterConnectionStore.serversDefaultsKey)
        // The active workspace pointed at the preset-shaped copy: the alias
        // must carry it over to the retained (edited) entry.
        defaults.set(
            ClusterConnectionStore.encodeWorkspace(.server(presetShaped.id)),
            forKey: ClusterConnectionStore.activeWorkspaceDefaultsKey)

        let store = clusterStore(defaults: defaults)
        #expect(store.servers.count == 1)
        let survivor = try #require(store.servers.first)
        #expect(survivor.id == edited.id)
        #expect(survivor.site == partlyEditedGenericSlurm())
        #expect(store.activeWorkspace == .server(edited.id))

        // Same-shape collisions keep today's first-wins behavior.
        var editedTwin = edited
        editedTwin.id = UUID()
        let (deduped, aliases) = ClusterConnectionStore.deduplicatedServersWithAliases(
            [edited, editedTwin])
        #expect(deduped.map(\.id) == [edited.id])
        #expect(aliases[editedTwin.id] == edited.id)
    }
}
