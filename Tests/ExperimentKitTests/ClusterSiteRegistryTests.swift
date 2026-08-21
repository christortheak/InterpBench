import Foundation
import Testing

@testable import ExperimentKit

/// THE canonical cluster-site registry (maintainer ruling, 2026-08-21):
/// `<SteerLab home>/Sites/cluster-sites/`, one JSON file per site, read and
/// written by BOTH clients.
///
/// What is pinned here is the ruling's whole substance:
///
///  (a) one store — a site either client writes is a site the other sees;
///  (b) the three-way split — profile in `Sites/`, secrets in the Keychain,
///      runtime state in a per-machine cache — enforced at the write, not
///      merely documented, and proven by a connect cycle that leaves the
///      registry BYTE-IDENTICAL;
///  (c) migration from both legacy stores, once, loudly, and never over a file
///      the canonical directory already holds;
///  (d) a plain directory: no git metadata, no git invocation, no requirement;
///  (e) import refuses to clobber in silence, and every import path runs the
///      same ssh-login validation.
///
/// Fixture vocabulary is deliberately neutral throughout (`site-a`,
/// `cluster.example.edu`) — no institution belongs in this tree.
struct ClusterSiteRegistryTests {

    // MARK: Fixtures

    /// A SteerLab home in a temp directory, laid out exactly as the real one.
    private func temporaryHome(_ name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(components: "steerlab-site-registry", "\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func registry(
        in home: URL, legacyRegistryData: @escaping @Sendable () -> Data? = { nil }
    ) -> ClusterSiteRepository {
        // Laid out exactly as the real thing: the shared registry inside
        // `Sites/`, the per-machine files OUTSIDE it (Application Support in
        // production, a sibling folder here).
        ClusterSiteRepository(
            directory: home.appending(components: "Sites", "cluster-sites"),
            runtimeStateURL: home.appending(
                components: "Support", "site-runtime.json"),
            legacyDocumentURL: home.appending(component: "cluster-sites.json"),
            legacyRegistryData: legacyRegistryData)
    }

    private func sshProfile(
        named name: String, host: String
    ) -> ClusterSiteProfile {
        ClusterSiteProfile(
            name: name,
            transport: .ssh(
                host: host, proxyJump: nil, remotePort: 8080, vpnExpected: false),
            topology: .daemonInJob,
            scheduler: .slurm(ClusterSiteProfile.SlurmSiteData()),
            constraints: ClusterSiteProfile.SiteConstraints(
                storageRoots: ["workspace": "/scratch/me/ws"]))
    }

    /// Every byte of the registry directory, keyed by filename — the shape a
    /// "did anything change?" assertion needs.
    private func snapshot(_ repository: ClusterSiteRepository) -> [String: Data] {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: repository.directoryURL, includingPropertiesForKeys: nil)
        else { return [:] }
        var out: [String: Data] = [:]
        for url in entries {
            out[url.lastPathComponent] = (try? Data(contentsOf: url)) ?? Data()
        }
        return out
    }

    // MARK: (a) One store, two clients

    @MainActor
    @Test func aSiteTheAppSavesIsTheSameSiteTheCLIReads() throws {
        let home = try temporaryHome("app-writes")
        let repository = registry(in: home)
        let defaults = try #require(
            UserDefaults(suiteName: "steerlab.tests.registry.app-writes"))
        defaults.removePersistentDomain(
            forName: "steerlab.tests.registry.app-writes")
        let store = ClusterConnectionStore(defaults: defaults, siteRegistry: repository)

        let entry = store.addSite(
            sshProfile(named: "Site A", host: "me@cluster.example.edu"))
        #expect(entry.siteID == "site-a")

        // The CLI's own repository — a separate value over the same directory,
        // exactly as a separately launched process would build it.
        let cli = registry(in: home)
        let seen = try #require(try cli.site(id: "site-a"))
        #expect(seen.profile == store.server(id: entry.id)?.site)
        #expect(
            FileManager.default.fileExists(
                atPath: home.appending(
                    components: "Sites", "cluster-sites", "site-a.json").path))
    }

    @MainActor
    @Test func aSiteTheCLISavesIsTheSameSiteTheAppReads() throws {
        let home = try temporaryHome("cli-writes")
        let cli = registry(in: home)
        _ = try cli.upsert(
            profile: sshProfile(named: "Site B", host: "me@cluster.example.edu"))

        let defaults = try #require(
            UserDefaults(suiteName: "steerlab.tests.registry.cli-writes"))
        defaults.removePersistentDomain(
            forName: "steerlab.tests.registry.cli-writes")
        let store = ClusterConnectionStore(
            defaults: defaults, siteRegistry: registry(in: home))
        #expect(store.servers.map(\.siteID) == ["site-b"])
        #expect(store.servers.first?.name == "Site B")

        // And a site added while the app is RUNNING is picked up by the
        // re-scan the app runs on activation / when the cluster UI opens.
        _ = try cli.upsert(
            profile: sshProfile(named: "Site C", host: "me@other.example.edu"))
        store.reloadSitesFromDisk()
        #expect(store.servers.compactMap(\.siteID).sorted() == ["site-b", "site-c"])
    }

    // MARK: (b) The three-way split

    @Test func aSiteFileIsExactlyAnExportedProfile() throws {
        let home = try temporaryHome("split")
        let repository = registry(in: home)
        let profile = sshProfile(named: "Site A", host: "me@cluster.example.edu")
        let record = try repository.upsert(profile: profile)
        let url = repository.fileURL(forSite: record.id)
        let bytes = try Data(contentsOf: url)

        // ONE format: a registry file IS an exported profile, byte for byte.
        // Dropping an export into `Sites/cluster-sites/` adds a site, and
        // copying a site file out is an export — there is no envelope to make
        // the two "almost" the same.
        #expect(try ClusterSiteSanitizer.encodedExport(for: record) == bytes)
        #expect(try ClusterSiteProfile.decode(from: bytes) == profile)

        // The id is the FILENAME and appears nowhere inside, so there is no
        // second place for it to be wrong.
        let text = try #require(String(data: bytes, encoding: .utf8))
        #expect(url.lastPathComponent == "site-a.json")
        let object = try #require(
            try JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        #expect(object["id"] == nil)
        // Diff-friendly by construction: pretty printed, stable key order,
        // one trailing newline.
        #expect(text.hasSuffix("}\n"))
        #expect(text.contains("\n  \"name\" :"))
        #expect(
            text.range(of: "\"constraints\"")!.lowerBound
                < text.range(of: "\"transport\"")!.lowerBound)
    }

    @Test func anExportedProfileDroppedIntoTheDirectoryIsASite() throws {
        // The workflow the one-format decision buys: export on one machine,
        // copy the file in on another, and it is simply there.
        let home = try temporaryHome("drop-in")
        let repository = registry(in: home)
        let profile = sshProfile(named: "Site D", host: "me@cluster.example.edu")
        try FileManager.default.createDirectory(
            at: repository.directoryURL, withIntermediateDirectories: true)
        try ClusterSiteSanitizer.encodedExport(
            for: ClusterSiteRecord(id: "hand-placed", displayName: "", profile: profile))
            .write(to: repository.fileURL(forSite: "hand-placed"))

        let sites = try repository.sites()
        #expect(sites.map(\.id) == ["hand-placed"])
        #expect(sites.first?.profile == profile)
        #expect(sites.first?.displayName == "Site D")
        #expect(repository.unreadableFiles().isEmpty)
    }

    @Test func noCredentialBearingKeyCanReachASiteFile() throws {
        // The sanitizer is the single enforcement point, and it REFUSES
        // rather than writing: a token that reached a git repository would be
        // in every clone of it, forever.
        let secrets = [
            #"{"token": "s3cret"}"#, #"{"bearerToken": "s3cret"}"#,
            #"{"password": "s3cret"}"#, #"{"apiKey": "s3cret"}"#,
            #"{"nested": {"privateKey": "s3cret"}}"#,
            #"{"sites": [{"passphrase": "s3cret"}]}"#,
        ]
        for json in secrets {
            #expect(!ClusterSiteSanitizer.offendingKeys(in: Data(json.utf8)).isEmpty)
        }
        // …while the site's DECLARED token-file PATH is an indirection, not a
        // secret, and must stay legal or the guard would refuse every honest
        // profile and get deleted.
        #expect(
            ClusterSiteSanitizer.offendingKeys(
                in: Data(#"{"tokenFilePath": "/home/me/.steerlab/token"}"#.utf8))
                .isEmpty)
        // Runtime facts are rejected by the same gate, for the same reason:
        // they are per machine and would make a connect dirty a shared file.
        #expect(
            ClusterSiteSanitizer.offendingKeys(
                in: Data(#"{"lastEndpoint": "http://127.0.0.1:8712"}"#.utf8))
                == ["lastEndpoint"])

        let home = try temporaryHome("leak")
        let repository = registry(in: home)
        let record = try repository.upsert(
            profile: sshProfile(named: "Site A", host: "me@cluster.example.edu"))
        let bytes = try Data(contentsOf: repository.fileURL(forSite: record.id))
        #expect(ClusterSiteSanitizer.offendingKeys(in: bytes).isEmpty)
    }

    @Test func exportCarriesNoSecretAndNoRuntimeKey() throws {
        let home = try temporaryHome("export")
        let repository = registry(in: home)
        var record = try repository.upsert(
            profile: sshProfile(named: "Site A", host: "me@cluster.example.edu"))
        record.lastEndpoint = "http://127.0.0.1:8712"
        record.lastServerBuild = "steerlab-server 0.1.0+deadbeef"
        record.legacyEntryID = UUID()
        let data = try ClusterSiteSanitizer.encodedExport(for: record)
        #expect(ClusterSiteSanitizer.offendingKeys(in: data).isEmpty)
        let text = try #require(String(data: data, encoding: .utf8))
        #expect(!text.contains("127.0.0.1:8712"))
        #expect(!text.contains("deadbeef"))
        #expect(text.hasSuffix("\n"))
        // Still a complete, importable profile.
        #expect(try ClusterSiteProfile.decode(from: data) == record.profile)
    }

    @Test func aConnectCycleLeavesTheRegistryByteIdentical() throws {
        let home = try temporaryHome("connect-cycle")
        let repository = registry(in: home)
        let record = try repository.upsert(
            profile: sshProfile(named: "Site A", host: "me@cluster.example.edu"))
        let before = snapshot(repository)
        #expect(before.count == 1)

        // What `connect` then `disconnect` record.
        _ = try repository.noteConnection(
            siteID: record.id, endpoint: "http://127.0.0.1:8712",
            serverBuild: "steerlab-server 0.1.0+deadbeef")
        _ = try repository.noteConnection(
            siteID: record.id, endpoint: nil, serverBuild: nil)

        #expect(snapshot(repository) == before)
        // The facts are not lost — they are per machine, in the runtime cache
        // beside the operation records.
        #expect(
            repository.runtime.state(forSite: record.id).lastServerBuild
                == "steerlab-server 0.1.0+deadbeef")
        #expect(
            FileManager.default.fileExists(atPath: repository.runtime.fileURL.path))
        // …and outside `Sites/` entirely, which is what keeps a shared
        // registry free of one machine's forward ports.
        #expect(!repository.runtime.fileURL.path.contains("/Sites/"))
    }

    @MainActor
    @Test func theAppReSavingItsRegistryIsANoOpOnDisk() throws {
        // The app re-persists the whole registry on any registry mutation —
        // including a tunnel relabelling an ssh entry's local URL, which is
        // runtime, not profile. A researcher whose `git status` went dirty
        // every time they connected would stop trusting the directory.
        let home = try temporaryHome("no-op-write")
        let defaults = try #require(
            UserDefaults(suiteName: "steerlab.tests.registry.no-op"))
        defaults.removePersistentDomain(forName: "steerlab.tests.registry.no-op")
        let repository = registry(in: home)
        let store = ClusterConnectionStore(defaults: defaults, siteRegistry: repository)
        let entry = store.addSite(
            sshProfile(named: "Site A", host: "me@cluster.example.edu"))
        let before = snapshot(repository)

        store.activeWorkspace = .server(entry.id)
        store.noteTunnelLocalPort(9123)
        #expect(store.server(id: entry.id)?.urlString == "http://127.0.0.1:9123")
        #expect(snapshot(repository) == before)

        // An actual PROFILE edit does change the file.
        store.updateSite(
            id: entry.id, profile: sshProfile(named: "Site A", host: "me@moved.example.edu"))
        #expect(snapshot(repository) != before)
    }

    // MARK: (c) Migration

    private func legacyDefaultsPayload() throws -> Data {
        try JSONEncoder().encode([
            ClusterConnectionStore.ServerEntry(
                name: "Site A", urlString: "http://127.0.0.1:8701",
                site: sshProfile(named: "Site A", host: "me@cluster.example.edu"))
        ])
    }

    /// The pre-2026-08-21 single-file registry, as older builds wrote it.
    private func writeLegacyDocument(
        _ home: URL, sites: [(id: String, host: String)]
    ) throws {
        let document = ClusterSiteRegistryDocument(
            sites: sites.map {
                ClusterSiteRecord(
                    id: $0.id, displayName: $0.id,
                    profile: sshProfile(named: $0.id, host: $0.host))
            })
        try ClusterSupportPaths.encoder().encode(document)
            .write(to: home.appending(component: "cluster-sites.json"))
    }

    @Test func migratesTheAppsSavedServersPreference() throws {
        let home = try temporaryHome("migrate-defaults")
        let payload = try legacyDefaultsPayload()
        let repository = registry(in: home, legacyRegistryData: { payload })
        let report = try repository.migrateLegacyStoresIfNeeded()
        #expect(report.migrated == ["site-a"])
        #expect(report.skipped.isEmpty)
        #expect(report.sources == ["the app's saved-servers preference"])
        #expect(try repository.sites().map(\.id) == ["site-a"])
        // Loud, and it says whose job the git half is.
        let summary = try #require(report.summary)
        #expect(summary.contains("site-a"))
        #expect(summary.contains("SteerLab never runs git"))
        // Once. A second pass reports nothing and resurrects nothing.
        try repository.remove(id: "site-a")
        #expect(try repository.migrateLegacyStoresIfNeeded().didAnything == false)
        #expect(try repository.sites().isEmpty)
    }

    @Test func migratesTheOldSingleFileRegistry() throws {
        let home = try temporaryHome("migrate-file")
        try writeLegacyDocument(home, sites: [("site-b", "me@cluster.example.edu")])
        let repository = registry(in: home)
        let report = try repository.migrateLegacyStoresIfNeeded()
        #expect(report.migrated == ["site-b"])
        #expect(report.sources == ["cluster-sites.json"])
        #expect(try repository.sites().map(\.id) == ["site-b"])
    }

    @Test func migratesBothStoresAndTheCanonicalFileAlwaysWins() throws {
        let home = try temporaryHome("migrate-both")
        // A site already in the canonical directory, under an id BOTH legacy
        // stores also claim.
        let repository = registry(in: home)
        let kept = try repository.upsert(
            profile: sshProfile(named: "Site A", host: "kept@cluster.example.edu"))
        #expect(kept.id == "site-a")

        try writeLegacyDocument(
            home,
            sites: [
                ("site-a", "stale@cluster.example.edu"),
                ("site-b", "me@other.example.edu"),
            ])
        let payload = try legacyDefaultsPayload()
        let both = registry(in: home, legacyRegistryData: { payload })

        let report = try both.migrateLegacyStoresIfNeeded()
        #expect(report.migrated == ["site-b"])
        // The collision is REPORTED, never resolved in silence: the
        // researcher is the one who knows which copy is right.
        #expect(report.skipped.contains("site-a"))
        #expect(report.sources.count == 2)
        #expect(try both.sites().map(\.id) == ["site-a", "site-b"])

        // The existing file is untouched, host and all.
        let survivor = try #require(try both.site(id: "site-a"))
        guard case .ssh(let host, _, _, _) = survivor.profile.transport else {
            Issue.record("the transport stopped being ssh")
            return
        }
        #expect(host == "kept@cluster.example.edu")
    }

    @MainActor
    @Test func theAppSurfacesWhatTheMigrationMoved() throws {
        let home = try temporaryHome("migrate-app")
        let payload = try legacyDefaultsPayload()
        let defaults = try #require(
            UserDefaults(suiteName: "steerlab.tests.registry.migrate-app"))
        defaults.removePersistentDomain(
            forName: "steerlab.tests.registry.migrate-app")
        let store = ClusterConnectionStore(
            defaults: defaults,
            siteRegistry: registry(in: home, legacyRegistryData: { payload }))
        #expect(store.servers.map(\.siteID) == ["site-a"])
        let summary = try #require(store.siteMigrationSummary)
        #expect(summary.contains("site-a"))
    }

    // MARK: (d) A plain directory

    @Test func theRegistryIsAPlainDirectoryWithNoGitAnywhere() throws {
        let home = try temporaryHome("plain")
        let repository = registry(in: home)
        _ = try repository.upsert(
            profile: sshProfile(named: "Site A", host: "me@cluster.example.edu"))
        _ = try repository.upsert(
            profile: sshProfile(named: "Site B", host: "me@other.example.edu"))
        try repository.remove(id: "site-a")

        // git is the SYNC mechanism, never a dependency: SteerLab creates no
        // repository, no index, no hooks, and runs no git command.
        let fm = FileManager.default
        for suspect in [".git", ".gitignore", ".gitattributes"] {
            #expect(
                !fm.fileExists(
                    atPath: repository.directoryURL.appending(component: suspect).path))
            #expect(
                !fm.fileExists(
                    atPath: home.appending(components: "Sites", suspect).path))
        }
        let contents = try fm.contentsOfDirectory(
            atPath: repository.directoryURL.path)
        #expect(contents == ["site-b.json"])
        #expect(try repository.sites().map(\.id) == ["site-b"])
    }

    @Test func theRegistryResolvesThroughTheHomeLayout() throws {
        let home = try temporaryHome("home-layout")
        let previous = HomeLayout.homeOverride
        defer { HomeLayout.homeOverride = previous }
        HomeLayout.homeOverride = home
        #expect(
            ClusterSupportPaths.sitesDirectory
                == home.appending(components: "Sites", "cluster-sites")
                    .standardizedFileURL)
        // `Sites/` STAYS EMPTY at materialization: a clone must be able to
        // land in it. The registry subdirectory is born at the first write.
        try HomeLayout.apply(try HomeLayout.plan(home: home))
        #expect(
            try FileManager.default.contentsOfDirectory(
                atPath: HomeLayout.sitesDirectory.path).isEmpty)
        #expect(
            !FileManager.default.fileExists(
                atPath: HomeLayout.clusterSitesDirectory.path))
        // The per-machine caches never live in the home: Application Support
        // is where this Mac's runtime state and operation records belong.
        #expect(!ClusterSupportPaths.runtimeStateFile.path.hasPrefix(home.path))
        #expect(!ClusterSupportPaths.legacySitesFile.path.hasPrefix(home.path))
    }

    // MARK: (e) Import

    @Test func importRefusesToClobberASiteTheRegistryAlreadyHolds() throws {
        let home = try temporaryHome("clobber")
        let repository = registry(in: home)
        _ = try repository.upsert(
            profile: sshProfile(named: "Site A", host: "me@cluster.example.edu"))
        let before = snapshot(repository)

        var edited = sshProfile(named: "Site A", host: "me@cluster.example.edu")
        edited.constraints.storageRoots = ["workspace": "/scratch/me/elsewhere"]
        let data = try edited.encoded()

        do {
            _ = try repository.importProfile(from: data)
            Issue.record("expected the import to refuse")
        } catch let error as ClusterLifecycleError {
            #expect(error.code == "siteFileExists")
            #expect(error.errorDescription?.contains("--force") == true)
        }
        #expect(snapshot(repository) == before)

        // --force is the deliberate replacement, and it refreshes the SAME
        // site rather than forking the registry.
        let replaced = try repository.importProfile(from: data, force: true)
        #expect(replaced.id == "site-a")
        #expect(try repository.sites().count == 1)
        #expect(
            try repository.site(id: "site-a")?.profile.constraints.storageRoots
                == ["workspace": "/scratch/me/elsewhere"])
    }

    @Test func everyImportPathRefusesALoginLessDestinationOutLoud() throws {
        // The live 2026-08-21 finding: the app accepted a profile whose ssh
        // destination had no `user@` and the researcher first learned of it at
        // Duo. Both clients now run the SAME validation at import time.
        let home = try temporaryHome("loginless")
        let repository = registry(in: home)
        let loginless = try sshProfile(
            named: "Site A", host: "cluster.example.edu").encoded()

        var warnings: [String] = []
        _ = try repository.importProfile(from: loginless, warn: { warnings.append($0) })
        let note = try #require(warnings.first)
        #expect(note.contains("carries no `user@` login"))
        // The message names the FIX, not just the problem.
        #expect(note.contains("user@cluster.example.edu"))

        // A bare transfer host inherits the login and stays legal…
        var withTransfer = sshProfile(named: "Site B", host: "cluster.example.edu")
        withTransfer.environment.transferHost = "xfer.example.edu"
        #expect(
            ClusterSiteRepository.sshLoginFinding(incoming: withTransfer, existing: nil)
                != .ok)  // still the login-less WARNING, not a refusal
        if case .refuse = ClusterSiteRepository.sshLoginFinding(
            incoming: withTransfer, existing: nil) {
            Issue.record("a bare transfer host must not be a refusal")
        }

        // …while a transfer host naming a DIFFERENT user contradicts the
        // profile and is refused, as it always was.
        var contradictory = sshProfile(named: "Site C", host: "cluster.example.edu")
        contradictory.environment.transferHost = "someone@cluster.example.edu"
        guard case .refuse(let user, _, _) = ClusterSiteRepository.sshLoginFinding(
            incoming: contradictory, existing: nil) else {
            Issue.record("expected the contradictory transfer host to refuse")
            return
        }
        #expect(user == "someone")
    }

    @MainActor
    @Test func theAppsImportSurfacesTheLoginLessFindingBeforeSavingAnything() throws {
        let home = try temporaryHome("app-loginless")
        let defaults = try #require(
            UserDefaults(suiteName: "steerlab.tests.registry.app-loginless"))
        defaults.removePersistentDomain(
            forName: "steerlab.tests.registry.app-loginless")
        let repository = registry(in: home)
        let store = ClusterConnectionStore(defaults: defaults, siteRegistry: repository)
        let loginless = try sshProfile(
            named: "Site A", host: "cluster.example.edu").encoded()

        do {
            _ = try store.importSite(from: loginless)
            Issue.record("expected the app's import to surface the finding")
        } catch let error as ClusterLifecycleError {
            #expect(error.code == "sshLoginMissing")
            #expect(error.errorDescription?.contains("user@cluster.example.edu") == true)
        }
        // Nothing was saved: a half-imported site in a shared registry is
        // worse than none.
        #expect(store.servers.isEmpty)
        #expect(try repository.sites().isEmpty)

        // Acknowledged, it imports — the check asks, it does not forbid.
        _ = try store.importSite(from: loginless, acknowledgingWarnings: true)
        #expect(store.servers.count == 1)
        #expect(try repository.sites().map(\.id) == ["site-a"])
    }

    // MARK: Deletion

    @MainActor
    @Test func removingASiteInTheAppDeletesItsFileAndItsRuntimeSlot() throws {
        let home = try temporaryHome("remove")
        let defaults = try #require(
            UserDefaults(suiteName: "steerlab.tests.registry.remove"))
        defaults.removePersistentDomain(forName: "steerlab.tests.registry.remove")
        let repository = registry(in: home)
        let store = ClusterConnectionStore(defaults: defaults, siteRegistry: repository)
        let entry = store.addSite(
            sshProfile(named: "Site A", host: "me@cluster.example.edu"))
        _ = try repository.noteConnection(
            siteID: "site-a", endpoint: "http://127.0.0.1:8712", serverBuild: nil)

        store.removeServer(id: entry.id)
        #expect(
            !FileManager.default.fileExists(
                atPath: repository.fileURL(forSite: "site-a").path))
        #expect(repository.runtime.load().sites["site-a"] == nil)
    }
}
