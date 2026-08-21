import Foundation
import Testing

@testable import ExperimentKit

/// Phase A of `docs/CLUSTER-CLI-LIFECYCLE-PLAN.md`: the shared, file-backed
/// state the app and a separately launched CLI must agree on.
///
/// Covered here: the site registry's one-time UserDefaults migration, its
/// round-trip and canonical-identity dedup, stable ids that survive renames,
/// operation-record atomicity and recovery, and the per-site interprocess
/// lock (the second caller observes; it never starts a duplicate).
///
/// Every store is constructed with an EXPLICIT temporary URL — no
/// process-global root override — so the suite stays hermetic without a
/// blocking lock.
struct ClusterLifecycleStorageTests {

    // MARK: Fixtures

    private func temporaryDirectory(_ name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(components: "steerlab-cluster-tests", "\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func sshProfile(
        named name: String, host: String, remotePort: Int = 8080
    ) -> ClusterSiteProfile {
        ClusterSiteProfile(
            name: name,
            transport: .ssh(
                host: host, proxyJump: nil, remotePort: remotePort, vpnExpected: false),
            topology: .daemonInJob,
            scheduler: .slurm(ClusterSiteProfile.SlurmSiteData()),
            constraints: ClusterSiteProfile.SiteConstraints(
                storageRoots: ["workspace": "/scratch/me/ws", "hfCache": "/work/lab/hf"]))
    }

    // MARK: Site repository — migration

    @Test func migratesTheLegacyUserDefaultsRegistryExactlyOnce() throws {
        let directory = try temporaryDirectory("migrate")
        let file = directory.appending(component: "cluster-sites.json")
        // Exactly what older builds persisted: a URL-only legacy entry plus a
        // site-carrying entry.
        let legacy = [
            ClusterConnectionStore.ServerEntry(
                name: "gpu-a", urlString: "http://gpu-a:8080"),
            ClusterConnectionStore.ServerEntry(
                name: "Example HPC", urlString: "http://127.0.0.1:8726",
                site: .exampleCluster),
        ]
        let payload = try JSONEncoder().encode(legacy)
        let repository = ClusterSiteRepository(
            fileURL: file, legacyRegistryData: { payload })

        let document = try repository.load()
        #expect(document.schemaVersion == ClusterSiteRegistryDocument.currentSchemaVersion)
        #expect(document.migratedFromUserDefaultsAt != nil)
        #expect(document.sites.count == 2)
        #expect(document.sites.map(\.id) == ["gpu-a", "example-hpc"])
        #expect(document.sites.map(\.legacyEntryID) == legacy.map { Optional($0.id) })
        // The SSH site's ephemeral tunnel URL is NOT adopted as an endpoint —
        // it is a local label, not a durable fact about the site.
        #expect(document.sites.last?.lastEndpoint == nil)
        #expect(document.sites.first?.lastEndpoint == "http://gpu-a:8080")
        #expect(FileManager.default.fileExists(atPath: file.path))

        // Second load reads the FILE; the legacy payload is never consulted
        // again (a re-migration would resurrect deleted sites).
        try repository.remove(id: "gpu-a")
        let reloaded = try repository.load()
        #expect(reloaded.sites.map(\.id) == ["example-hpc"])
    }

    @Test func migrationDedupesByCanonicalRemoteIdentity() throws {
        let directory = try temporaryDirectory("migrate-dedupe")
        var edited = ClusterSiteProfile.exampleCluster
        edited.transport = .ssh(
            host: "user@slurm.example.edu", proxyJump: nil, remotePort: 8080,
            vpnExpected: true)
        let legacy = [
            ClusterConnectionStore.ServerEntry(
                name: "Example HPC", urlString: "http://127.0.0.1:8701", site: .exampleCluster),
            ClusterConnectionStore.ServerEntry(
                name: "Example HPC", urlString: "http://127.0.0.1:8702", site: edited),
        ]
        let payload = try JSONEncoder().encode(legacy)
        let repository = ClusterSiteRepository(
            fileURL: directory.appending(component: "sites.json"),
            legacyRegistryData: { payload })

        let sites = try repository.sites()
        // One site: `user@host` and the bare host are the same cluster.
        #expect(sites.count == 1)
        #expect(sites.first?.profile.name == "Example HPC")
    }

    /// The other half of the migration's dedup rule: on a collision the EDITED
    /// entry survives, never the pristine shipped-preset copy. Since WP5 Step
    /// 12 the presets are hostless templates (§4.2), so the collision that
    /// exercises this is a template added twice and configured once — both
    /// entries still key by the template's name, and the one with storage
    /// roots is the one the researcher touched.
    @Test func migrationKeepsTheEditedEntryOverAPristinePresetCopy() throws {
        let directory = try temporaryDirectory("migrate-preset-preference")
        var configured = ClusterSiteProfile.genericSlurm
        configured.constraints.storageRoots = ["workspace": "/scratch/user/steerlab-workspace"]
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
            fileURL: directory.appending(component: "sites.json"),
            legacyRegistryData: { payload })

        let sites = try repository.sites()
        #expect(sites.count == 1)
        #expect(sites.first?.profile == configured)
    }

    @Test func migrationOfAnAbsentLegacyRegistryYieldsAnEmptyDocument() throws {
        let directory = try temporaryDirectory("migrate-empty")
        let repository = ClusterSiteRepository(
            fileURL: directory.appending(component: "sites.json"),
            legacyRegistryData: { nil })
        let document = try repository.load()
        #expect(document.sites.isEmpty)
        #expect(document.migratedFromUserDefaultsAt == nil)
    }

    // MARK: Site repository — round-trip, ids, dedup

    @Test func upsertRoundTripsAndKeepsTheIDStableAcrossProfileEdits() throws {
        let directory = try temporaryDirectory("roundtrip")
        let repository = ClusterSiteRepository(
            fileURL: directory.appending(component: "sites.json"),
            legacyRegistryData: { nil })

        let created = try repository.upsert(profile: sshProfile(named: "Example HPC", host: "cluster.test"))
        #expect(created.id == "example-hpc")

        // Edit the profile (login identity + a storage root). Same canonical
        // identity ⇒ same record, same id.
        var edited = created.profile
        edited.transport = .ssh(
            host: "me@cluster.test", proxyJump: nil, remotePort: 8080, vpnExpected: true)
        edited.constraints.storageRoots["archive"] = "/project/lab"
        let updated = try repository.upsert(profile: edited)
        #expect(updated.id == created.id)
        #expect(try repository.sites().count == 1)

        // The file is the truth: a second repository over the same URL sees it.
        let reader = ClusterSiteRepository(
            fileURL: repository.fileURL, legacyRegistryData: { nil })
        let stored = try #require(try reader.site(id: "example-hpc"))
        #expect(stored.profile == edited)
        #expect(stored.profile.constraints.storageRoots["archive"] == "/project/lab")
    }

    @Test func distinctSitesGetDistinctStableIDsAndAreNeverMerged() throws {
        let directory = try temporaryDirectory("distinct")
        let repository = ClusterSiteRepository(
            fileURL: directory.appending(component: "sites.json"),
            legacyRegistryData: { nil })
        let a = try repository.upsert(profile: sshProfile(named: "Cluster", host: "a.test"))
        let b = try repository.upsert(profile: sshProfile(named: "Cluster", host: "b.test"))
        #expect(a.id == "cluster")
        #expect(b.id == "cluster-2")  // same display name, different site
        #expect(try repository.sites().count == 2)
        #expect(try repository.duplicateCount(forIdentity: a.canonicalIdentity) == 1)
    }

    @Test func resolveAcceptsIDsAndUnambiguousNamesOnly() throws {
        let directory = try temporaryDirectory("resolve")
        let repository = ClusterSiteRepository(
            fileURL: directory.appending(component: "sites.json"),
            legacyRegistryData: { nil })
        _ = try repository.upsert(profile: sshProfile(named: "Bench", host: "a.test"))
        // A unique display name resolves (case-insensitively) as a convenience.
        #expect(try repository.resolve(reference: "bench")?.id == "bench")
        #expect(try repository.resolve(reference: "Bench")?.id == "bench")

        // A second site with the SAME display name makes the name ambiguous:
        // names are labels, not identity, so the lookup refuses rather than
        // guessing which cluster the caller meant. The id still resolves.
        _ = try repository.upsert(profile: sshProfile(named: "Bench", host: "b.test"))
        #expect(try repository.resolve(reference: "Bench") == nil)
        #expect(try repository.resolve(reference: "bench")?.id == "bench")
        #expect(try repository.resolve(reference: "bench-2")?.id == "bench-2")
        #expect(try repository.resolve(reference: "not-a-site") == nil)
    }

    @Test func noteConnectionRecordsEndpointAndBuildOnly() throws {
        let directory = try temporaryDirectory("note-connection")
        let repository = ClusterSiteRepository(
            fileURL: directory.appending(component: "sites.json"),
            legacyRegistryData: { nil })
        let site = try repository.upsert(profile: sshProfile(named: "Cluster", host: "a.test"))
        _ = try repository.noteConnection(
            siteID: site.id, endpoint: "http://127.0.0.1:8712",
            serverBuild: "steerlab-server 0.1.0+46bb4f9a")
        let stored = try #require(try repository.site(id: site.id))
        #expect(stored.lastEndpoint == "http://127.0.0.1:8712")
        #expect(stored.lastServerBuild == "steerlab-server 0.1.0+46bb4f9a")
        // The token key is derived, never stored: the file must not contain
        // any token-shaped field except the site's DECLARED token-file PATH
        // (schema 2, `environment.tokenFilePath`), which is an indirection.
        #expect(stored.tokenKey == "a.test:8080")
        let bytes = try String(contentsOf: repository.fileURL, encoding: .utf8)
        #expect(
            !ClusterProfileTokenScrub.withoutDeclaredTokenPath(bytes).lowercased()
                .contains("token"))
    }

    @Test func aCorruptRecordCostsOnlyItself() throws {
        let directory = try temporaryDirectory("corrupt")
        let file = directory.appending(component: "sites.json")
        let json = """
            {"schemaVersion": 1,
             "sites": [
               {"id": "good", "displayName": "Good", "createdAt": "2026-08-11T00:00:00Z",
                "updatedAt": "2026-08-11T00:00:00Z",
                "profile": {"schemaVersion": 1, "name": "Good",
                            "transport": {"kind": "ssh", "host": "good.test"}}},
               {"id": "broken", "displayName": "Broken", "createdAt": "2026-08-11T00:00:00Z",
                "updatedAt": "2026-08-11T00:00:00Z",
                "profile": {"schemaVersion": 1, "name": "Broken",
                            "transport": {"kind": "carrier-pigeon"}}}
             ]}
            """
        try Data(json.utf8).write(to: file)
        let repository = ClusterSiteRepository(fileURL: file, legacyRegistryData: { nil })
        #expect(try repository.sites().map(\.id) == ["good"])
    }

    @Test func aNewerSchemaRefusesRatherThanSilentlyDowngrading() throws {
        let directory = try temporaryDirectory("newer-schema")
        let file = directory.appending(component: "sites.json")
        try Data(#"{"schemaVersion": 99, "sites": []}"#.utf8).write(to: file)
        let repository = ClusterSiteRepository(fileURL: file, legacyRegistryData: { nil })
        #expect(throws: (any Error).self) { try repository.load() }
    }

    // MARK: Operation store — records

    @Test func operationRecordsRoundTripAndRecoverAcrossProcesses() throws {
        let directory = try temporaryDirectory("operations")
        let store = ClusterOperationStore(rootDirectory: directory)
        var record = ClusterOperationRecord(
            operationID: "cluster-20260811-000000-aa",
            siteID: "example-hpc",
            siteProfileHash: "abc123",
            target: .connected,
            authorizedMutations: ClusterLifecyclePermissions.allMutations.identifiers,
            startedAt: Date(timeIntervalSince1970: 1_000))
        record.note(
            step: .bootstrapApply, status: .awaitingScheduler,
            message: "bootstrap job 47207294 queued",
            now: Date(timeIntervalSince1970: 1_010))
        record.bootstrapJobID = "47207294"
        record.bootstrapPlanHash = String(repeating: "a", count: 64)
        record.state = .pending
        try store.save(record)

        // A NEW store object (the "later CLI invocation") recovers everything.
        let recovered = ClusterOperationStore(rootDirectory: directory)
        let read = try #require(
            recovered.record(siteID: "example-hpc", operationID: record.operationID))
        #expect(read == record)
        #expect(recovered.lastBootstrapPlanHash(forSite: "example-hpc") == record.bootstrapPlanHash)
        #expect(recovered.activeRecord(forSite: "example-hpc")?.operationID == record.operationID)
        #expect(read.steps.first?.status == .awaitingScheduler)
    }

    @Test func recordsAreOrderedNewestFirstAndTerminalOnesAreNotActive() throws {
        let directory = try temporaryDirectory("operations-order")
        let store = ClusterOperationStore(rootDirectory: directory)
        for (index, state) in [ClusterLifecycleState.ready, .pending].enumerated() {
            var record = ClusterOperationRecord(
                operationID: "op-\(index)", siteID: "s", siteProfileHash: "h",
                target: .connected,
                startedAt: Date(timeIntervalSince1970: TimeInterval(index)))
            record.state = state
            record.controllerJobID = state == .pending ? "999" : nil
            try store.save(record)
        }
        #expect(store.records(forSite: "s").map(\.operationID) == ["op-1", "op-0"])
        #expect(store.activeRecord(forSite: "s")?.operationID == "op-1")
        #expect(store.lastControllerJobID(forSite: "s") == "999")
    }

    // MARK: Pending bootstrap jobs (review finding 5)

    @Test func aPendingBootstrapJobRoundTripsAndIsResumedUntilItIsClosedOut() throws {
        // The whole point of the record: a bootstrap job submitted by one
        // process is findable by the NEXT one, so a retry after a sleep or a
        // dropped SSH session resumes it instead of queueing a second job.
        let directory = try temporaryDirectory("pending-bootstrap")
        let store = ClusterOperationStore(rootDirectory: directory)
        #expect(store.pendingBootstrapJob(forSite: "s") == nil)

        let submitted = Date(timeIntervalSince1970: 1_000)
        try store.recordPendingBootstrapJob(
            ClusterPendingBootstrapJob(
                siteID: "s", jobID: "4242",
                statusFile: "/scratch/me/ws/steerlab-bootstrap.x.status",
                planHash: "abc123", submittedAt: submitted))

        // A SEPARATE store object, as a second process would be.
        let reader = ClusterOperationStore(rootDirectory: directory)
        let pending = try #require(reader.pendingBootstrapJob(forSite: "s"))
        #expect(pending.jobID == "4242")
        #expect(pending.statusFile == "/scratch/me/ws/steerlab-bootstrap.x.status")
        #expect(pending.planHash == "abc123")
        #expect(pending.submittedAt == submitted)
        #expect(reader.lastBootstrapJobID(forSite: "s") == "4242")
        // It reads as work still in flight, so nothing treats it as done.
        #expect(reader.activeRecord(forSite: "s")?.bootstrapJobID == "4242")
        #expect(reader.records(forSite: "s").first?.steps.first?.status
            == .awaitingScheduler)

        // Another site's job is never this site's.
        #expect(reader.pendingBootstrapJob(forSite: "other") == nil)

        // A LATER terminal record does not hide the still-pending job…
        var later = ClusterOperationRecord(
            operationID: "op-later", siteID: "s", siteProfileHash: "h",
            target: .bootstrapped, startedAt: Date(timeIntervalSince1970: 2_000))
        later.state = .failed
        try store.save(later)
        #expect(reader.pendingBootstrapJob(forSite: "s")?.jobID == "4242")

        // …and closing the job out is what finally stops the resume.
        store.finishPendingBootstrapJob(
            forSite: "s", jobID: "4242", succeeded: true, message: "bootstrap ok")
        #expect(reader.pendingBootstrapJob(forSite: "s") == nil)
        // Closing an unknown job is a no-op, never a crash or a new record.
        store.finishPendingBootstrapJob(
            forSite: "s", jobID: "9999", succeeded: false, message: "gone")
        #expect(reader.records(forSite: "s").count == 2)
    }

    @Test func recordWritesAreAtomicAndNeverLeaveAPartialFile() throws {
        let directory = try temporaryDirectory("atomic")
        let store = ClusterOperationStore(rootDirectory: directory)
        var record = ClusterOperationRecord(
            operationID: "op", siteID: "s", siteProfileHash: "h", target: .connected)
        for iteration in 0..<25 {
            record.note(
                step: .validate, status: .running, message: "pass \(iteration)")
            try store.save(record)
            // Every intermediate state on disk is a COMPLETE, decodable
            // record — the atomic replace never exposes a truncated file.
            let onDisk = try #require(store.record(siteID: "s", operationID: "op"))
            #expect(onDisk.steps.first?.message == "pass \(iteration)")
        }
        let names = try FileManager.default.contentsOfDirectory(
            atPath: store.directory(forSite: "s").path)
        #expect(names.filter { $0.hasSuffix(".json") } == ["op.json"])
    }

    @Test func siteIdentifiersCannotEscapeTheStoreDirectory() {
        #expect(ClusterOperationStore.safeComponent("../../etc") == "------etc")
        #expect(ClusterOperationStore.safeComponent("..") == "--")
        #expect(ClusterOperationStore.safeComponent("") == "site")
        #expect(ClusterOperationStore.safeComponent("example-hpc") == "example-hpc")
        let store = ClusterOperationStore(rootDirectory: URL(filePath: "/tmp/root"))
        #expect(!store.directory(forSite: "../escape").path.contains(".."))
    }

    // MARK: Operation store — per-site exclusion

    @Test func theSecondCallerObservesInsteadOfStartingADuplicate() throws {
        let directory = try temporaryDirectory("lock")
        let store = ClusterOperationStore(rootDirectory: directory)

        // Caller A starts an operation and takes the site lock.
        let held = try #require(try store.acquireLock(siteID: "example-hpc"))
        var running = ClusterOperationRecord(
            operationID: "op-a", siteID: "example-hpc", siteProfileHash: "h",
            target: .connected)
        running.state = .pending
        running.controllerJobID = "47207294"
        try store.save(running)

        // Caller B — a separate store object, as a second process would be —
        // cannot take the lock, and finds the operation to report instead.
        let second = ClusterOperationStore(rootDirectory: directory)
        #expect(try second.acquireLock(siteID: "example-hpc") == nil)
        let observed = try #require(second.activeRecord(forSite: "example-hpc"))
        #expect(observed.operationID == "op-a")
        #expect(observed.controllerJobID == "47207294")

        // A DIFFERENT site is never blocked by this one.
        let otherSite = try second.acquireLock(siteID: "other")
        #expect(otherSite != nil)
        otherSite?.release()

        // Once A releases, B may proceed.
        held.release()
        let acquired = try second.acquireLock(siteID: "example-hpc")
        #expect(acquired != nil)
        acquired?.release()
    }

    @Test func releasingTheLockTwiceIsHarmless() throws {
        let directory = try temporaryDirectory("lock-release")
        let store = ClusterOperationStore(rootDirectory: directory)
        let lock = try #require(try store.acquireLock(siteID: "s"))
        lock.release()
        lock.release()
        let again = try store.acquireLock(siteID: "s")
        #expect(again != nil)
    }

    // MARK: Vocabulary invariants

    @Test func targetsAreOrderedAndParseBothSpellings() {
        #expect(ClusterLifecycleTarget.authenticated < .connected)
        #expect(ClusterLifecycleTarget.allCases.map(\.order) == Array(0..<6))
        #expect(ClusterLifecycleTarget.parse("code-deployed") == .codeDeployed)
        #expect(ClusterLifecycleTarget.parse("codeDeployed") == .codeDeployed)
        #expect(ClusterLifecycleTarget.parse("nope") == nil)
        // Each step serves exactly one rung, and no step outranks connected.
        for step in ClusterLifecycleStep.allCases {
            #expect(step.target <= .connected)
        }
    }

    @Test func permissionsAreIndividualAndNameThemselvesStably() {
        #expect(ClusterLifecyclePermissions.allMutations.identifiers
            == ["push", "bootstrap", "controller-start"])
        #expect(ClusterLifecyclePermissions.push.flags == ["--allow-push"])
        #expect(ClusterLifecyclePermissions.parse(identifier: "controller-start")
            == .controllerStart)
        // The auth terminal is NOT part of the mutation bundle.
        #expect(!ClusterLifecyclePermissions.allMutations.contains(.openAuthTerminal))
    }

    @Test func statesMapToTheDocumentedExitCodes() {
        #expect(ClusterLifecycleState.ready.exitCode == 0)
        #expect(ClusterLifecycleState.needsHumanAuthentication.exitCode == 10)
        #expect(ClusterLifecycleState.needsApproval.exitCode == 11)
        #expect(ClusterLifecycleState.pending.exitCode == 12)
        #expect(ClusterLifecycleState.degraded.exitCode == 13)
        #expect(ClusterLifecycleState.blocked.exitCode == 64)
        #expect(ClusterLifecycleState.failed.exitCode == 70)
    }
}
