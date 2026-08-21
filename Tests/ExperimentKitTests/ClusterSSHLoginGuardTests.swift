import Foundation
import Testing

@testable import ExperimentKit

/// The ssh-login guard on registry writes (open-issues §17, 2026-08-19).
///
/// The 2026-08-18 outage: the saved transport for the cluster had lost the
/// `user@` half of its ssh destination, so every `cluster auth open` ran
/// `ssh <host>` as the LOCAL account. An SSO front end can treat an unknown user as
/// unenrolled, which presents as "asks for my password, then tells me to
/// enroll in 2FA, then asks again" — a server-side outage, apparently, and
/// several human Duo attempts before anyone suspected the registry.
///
/// Two facts make the drop possible and invisible:
///
/// * `ClusterConnectionStore.canonicalKey` strips `user@` ON PURPOSE, so
///   `host` and `user@host` are ONE site (that is what lets a re-offered
///   preset dedupe against the researcher's edited entry); and
/// * every registry writer then REPLACES the matched record's whole profile.
///
/// So a login-less profile lands on the login-carrying record and wins. These
/// tests pin the guard: a known login that would be dropped refuses at write
/// time; a site with no login anywhere warns and is still stored (some sites
/// legitimately authenticate through a `User` entry in `~/.ssh/config`); and
/// the destination round-trips byte-for-byte through everything else.
struct ClusterSSHLoginGuardTests {

    private func temporaryDirectory(_ name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(components: "steerlab-ssh-login-guard",
                       "\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: url, withIntermediateDirectories: true)
        return url
    }

    private func profile(
        named name: String = "Example HPC", host: String,
        proxyJump: String? = nil, transferHost: String? = nil
    ) -> ClusterSiteProfile {
        var built = ClusterSiteProfile(
            name: name,
            transport: .ssh(host: host, proxyJump: proxyJump, remotePort: 8080,
                            vpnExpected: false),
            topology: .daemonInJob,
            scheduler: .slurm(ClusterSiteProfile.SlurmSiteData()),
            constraints: ClusterSiteProfile.SiteConstraints(
                storageRoots: ["workspace": "/scratch/me/ws"]))
        built.environment.transferHost = transferHost
        return built
    }

    private func repository(_ name: String) throws -> ClusterSiteRepository {
        ClusterSiteRepository(
            directory: try temporaryDirectory(name).appending(component: "cluster-sites"),
            legacyRegistryData: { nil })
    }

    private func storedHost(_ record: ClusterSiteRecord?) -> String? {
        guard case .ssh(let host, _, _, _) = record?.profile.transport else {
            return nil
        }
        return host
    }

    // MARK: The finding itself

    @Test func aDestinationWithALoginIsFine() {
        #expect(
            ClusterSiteRepository.sshLoginFinding(
                incoming: profile(host: "alice@login.test"), existing: nil)
            == .ok)
    }

    @Test func aLoginlessSiteWithNoKnownLoginAnywhereOnlyWarns() throws {
        let finding = ClusterSiteRepository.sshLoginFinding(
            incoming: profile(host: "login.test"), existing: nil)
        guard case .warn(let note) = finding else {
            Issue.record("expected a warning, got \(finding)")
            return
        }
        // The message must say WHY this is legal, or the next reader "repairs"
        // a site that was correct.
        #expect(note.contains("~/.ssh/config"))
        #expect(note.contains("user@login.test"))
    }

    @Test func replacingALoginCarryingSiteWithALoginlessOneRefuses() {
        let finding = ClusterSiteRepository.sshLoginFinding(
            incoming: profile(host: "login.test"),
            existing: profile(host: "alice@login.test"))
        #expect(finding == .refuse(
            user: "alice", host: "login.test",
            source: "the site it replaces is saved as 'alice@login.test'"))
    }

    @Test func aProfileThatContradictsItselfRefuses() {
        // The login is right there in the same document, on another
        // destination for the same machine.
        let viaTransfer = ClusterSiteRepository.sshLoginFinding(
            incoming: profile(host: "login.test",
                              transferHost: "alice@login.test"),
            existing: nil)
        #expect(viaTransfer == .refuse(
            user: "alice", host: "login.test",
            source: "the same profile's transfer host is 'alice@login.test'"))

        // A proxy jump into a DIFFERENT machine is not evidence about this
        // one — bastions legitimately have their own logins.
        #expect(
            ClusterSiteRepository.sshLoginFinding(
                incoming: profile(host: "login.test",
                                  proxyJump: "alice@bastion.test"),
                existing: nil)
            != .refuse(user: "alice", host: "login.test",
                       source: "the same profile's proxy jump is "
                           + "'alice@bastion.test'"))
    }

    @Test func aDifferentHostIsADifferentSiteNotADroppedLogin() {
        // Same record, genuinely re-pointed at another cluster: nothing was
        // dropped, so nothing may refuse on that ground.
        let finding = ClusterSiteRepository.sshLoginFinding(
            incoming: profile(host: "other.test"),
            existing: profile(host: "alice@login.test"))
        guard case .warn = finding else {
            Issue.record("expected a warning, got \(finding)")
            return
        }
    }

    @Test func directTransportHasNoLoginToLose() {
        var direct = profile(host: "login.test")
        direct.transport = .direct(baseURL: URL(string: "http://gpu-a:8080")!)
        #expect(
            ClusterSiteRepository.sshLoginFinding(incoming: direct, existing: nil)
            == .ok)
        // A hostless TEMPLATE (a shipped preset before it is configured) is
        // not a site with a missing login.
        #expect(
            ClusterSiteRepository.sshLoginFinding(
                incoming: profile(host: "  "), existing: nil)
            == .ok)
    }

    @Test func theLoginSplitIsOnTheLastAtSign() {
        #expect(ClusterSiteRepository.sshLogin(in: "alice@login.test")
                == "alice")
        #expect(ClusterSiteRepository.sshLogin(in: "login.test") == nil)
        #expect(ClusterSiteRepository.sshLogin(in: "@login.test") == nil)
        // Same rule `canonicalKey` uses, so the guard and the dedup cannot
        // disagree about where the login ends.
        #expect(ClusterSiteRepository.sshLogin(in: "a@b@login.test") == "a@b")
    }

    // MARK: The repository write

    @Test func importingALoginlessProfileOverALoginCarryingSiteRefuses() throws {
        let repository = try repository("refuse")
        let created = try repository.upsert(
            profile: profile(host: "alice@login.test"))

        #expect(throws: ClusterLifecycleError.sshLoginDropped(
            siteID: created.id, host: "login.test", expectedUser: "alice",
            source: "the site it replaces is saved as 'alice@login.test'")
        ) {
            try repository.upsert(profile: profile(host: "login.test"))
        }

        // The refusal is a NO-OP on the file: the stored destination is
        // exactly what it was, byte for byte.
        #expect(storedHost(try repository.site(id: created.id))
                == "alice@login.test")
        #expect(try repository.sites().count == 1)
    }

    @Test func theRefusalNamesTheRepairAsACommand() {
        let error = ClusterLifecycleError.sshLoginDropped(
            siteID: "test-site", host: "login.test", expectedUser: "alice",
            source: "the site it replaces is saved as 'alice@login.test'")
        #expect(error.code == "sshLoginDropped")
        let repair = ClusterCLIRunner.repairAction(for: error)
        #expect(repair.contains("alice@login.test"))
        #expect(repair.contains("cluster sites import"))
        // And says how a genuinely login-less site is stored, so the guard is
        // not a wall.
        #expect(repair.contains("~/.ssh/config"))
    }

    @Test func addingALoginIsAlwaysAllowed() throws {
        let repository = try repository("add-login")
        let created = try repository.upsert(profile: profile(host: "login.test"))
        let updated = try repository.upsert(
            profile: profile(host: "alice@login.test"))
        #expect(updated.id == created.id)
        #expect(storedHost(try repository.site(id: created.id))
                == "alice@login.test")
    }

    @Test func aLoginlessSiteIsStoredWithALoudWarning() throws {
        let repository = try repository("warn")
        var warnings: [String] = []
        let record = try repository.upsert(
            profile: profile(host: "login.test"),
            warn: { warnings.append($0) })
        #expect(warnings.count == 1)
        #expect(warnings.first?.contains("no `user@` login") == true)
        #expect(storedHost(try repository.site(id: record.id)) == "login.test")
    }

    @Test func theDestinationRoundTripsThroughExportAndImport() throws {
        let repository = try repository("round-trip")
        let original = profile(host: "alice@login.test",
                               proxyJump: "alice@bastion.test",
                               transferHost: "alice@xfer.test")
        let created = try repository.upsert(profile: original)

        // Export → decode → import, the exact path the manual repair used.
        let data = try created.profile.encoded()
        let decoded = try ClusterSiteProfile.decode(from: data)
        #expect(decoded == original)
        let reimported = try repository.upsert(profile: decoded)
        #expect(reimported.id == created.id)

        // Read back through a SECOND repository over the same file: the JSON
        // on disk carries the login, not just the in-memory value.
        let reader = ClusterSiteRepository(
            directory: repository.directoryURL, legacyRegistryData: { nil })
        let stored = try #require(try reader.site(id: created.id))
        #expect(storedHost(stored) == "alice@login.test")
        #expect(stored.profile.environment.transferHost == "alice@xfer.test")
        let text = try String(
            contentsOf: repository.fileURL(forSite: created.id), encoding: .utf8)
        #expect(text.contains("alice@login.test"))
    }

    @Test func noteConnectionNeverTouchesTheDestination() throws {
        // What `cluster ensure`/`connect` re-persist: endpoint and build only.
        let repository = try repository("note-connection")
        let created = try repository.upsert(
            profile: profile(host: "alice@login.test"))
        _ = try repository.noteConnection(
            siteID: created.id, endpoint: "http://127.0.0.1:8712",
            serverBuild: "steerlab-server 0.1.0+abc")
        #expect(storedHost(try repository.site(id: created.id))
                == "alice@login.test")
    }

    // MARK: The lossy path the audit found

    @Test func theMigrationDedupKeepsTheEntryThatCarriesTheLogin() throws {
        // BOTH entries are edited (neither is a shipped preset), so the old
        // rule was "the first in array order wins" — which silently discarded
        // the login when the bare-host duplicate happened to come first. The
        // one-time migration into cluster-sites.json runs through exactly this
        // code, which makes it a candidate for how the outage's registry lost
        // its `user@` half.
        let directory = try temporaryDirectory("migrate-login")
        let bare = profile(host: "login.test")
        let withLogin = profile(host: "alice@login.test")
        let legacy = [
            ClusterConnectionStore.ServerEntry(
                name: "Example HPC", urlString: "http://127.0.0.1:8701",
                site: bare),
            ClusterConnectionStore.ServerEntry(
                name: "Example HPC", urlString: "http://127.0.0.1:8702",
                site: withLogin),
        ]
        let payload = try JSONEncoder().encode(legacy)
        let repository = ClusterSiteRepository(
            directory: directory.appending(component: "cluster-sites"),
            legacyRegistryData: { payload })

        let sites = try repository.sites()
        #expect(sites.count == 1)
        #expect(storedHost(sites.first) == "alice@login.test")
    }

    @Test func theMigrationStillPrefersAnEditedEntryOverAPreset() throws {
        // The older rule this one is layered on top of must survive: a
        // pristine preset never wins over an edited entry, login or not.
        let directory = try temporaryDirectory("migrate-preset")
        var configured = ClusterSiteProfile.genericSlurm
        configured.constraints.storageRoots = ["workspace": "/scratch/me/ws"]
        let legacy = [
            ClusterConnectionStore.ServerEntry(
                name: ClusterSiteProfile.genericSlurm.name,
                urlString: "http://127.0.0.1:8703", site: .genericSlurm),
            ClusterConnectionStore.ServerEntry(
                name: ClusterSiteProfile.genericSlurm.name,
                urlString: "http://127.0.0.1:8704", site: configured),
        ]
        let payload = try JSONEncoder().encode(legacy)
        let repository = ClusterSiteRepository(
            directory: directory.appending(component: "cluster-sites"),
            legacyRegistryData: { payload })
        #expect(try repository.sites().first?.profile == configured)
    }

    // MARK: The app registry's import-over

    private func freshDefaults(_ name: String) throws -> UserDefaults {
        let suite = "steerlab.tests.ssh-login-guard.\(name)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    @MainActor
    @Test func theAppsImportOverRefusesTheSameDrop() throws {
        let store = clusterStore(defaults: try freshDefaults("import-over"))
        _ = store.addSite(profile(host: "alice@login.test"))
        let loginless = try profile(host: "login.test").encoded()
        #expect(throws: ClusterLifecycleError.self) {
            try store.importSite(from: loginless)
        }
        guard case .ssh(let host, _, _, _) = store.servers.first?.site?.transport
        else {
            Issue.record("the site vanished")
            return
        }
        #expect(host == "alice@login.test")

        // Into a registry that never knew a login the same profile is LEGAL
        // (`~/.ssh/config` can supply `User`) — but the app now says so at
        // IMPORT time rather than letting the researcher discover it at Duo
        // (live finding, 2026-08-21). It asks; it does not forbid.
        let fresh = clusterStore(defaults: try freshDefaults("import-fresh"))
        do {
            _ = try fresh.importSite(from: loginless)
            Issue.record("expected the login-less import to be surfaced")
        } catch let error as ClusterLifecycleError {
            #expect(error.code == "sshLoginMissing")
            #expect(error.errorDescription?.contains("user@login.test") == true)
        }
        #expect(fresh.servers.isEmpty)
        // Acknowledged, it imports.
        _ = try fresh.importSite(from: loginless, acknowledgingWarnings: true)
        #expect(fresh.servers.count == 1)
    }
}
