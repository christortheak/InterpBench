import Foundation
import Testing

@testable import ExperimentKit

/// WS3/WS4 wire contract: the housekeeping status payload (camelCase,
/// ISO-8601 dates), the capability flags, the submission preflight report,
/// and the pure presentation rules the health card renders (thresholds are
/// data interpretation — tested here, never re-derived in a view).
struct ClusterHousekeepingTests {

    // MARK: Server-side revision lookup (review round 5, finding 5)

    /// The judge picker lists SERVER-installed models, so resolving a
    /// revision only against the Mac's cache left the cluster-only judge —
    /// the common case — needing a hand-copied hash. The housekeeping scan
    /// already carries each cached repo's refs/main.
    @Test func cachedRevisionReadsTheServersOwnModelCache() throws {
        let status = try JSONDecoder().decode(
            RemoteHousekeepingStatus.self, from: Data(Self.fullStatusJSON.utf8))
        #expect(status.cachedRevision(forModel: "Qwen/Qwen3-14B") != nil)
        // A model the server does not hold answers nil — a real answer the
        // caller reports, never a hash borrowed from somewhere else.
        #expect(status.cachedRevision(forModel: "org/not-installed") == nil)
        #expect(status.cachedRevision(forModel: "  ") == nil)
    }

    @Test func cachedRevisionIsNilWhenTheServerReportsNoCache() throws {
        let empty = try JSONDecoder().decode(
            RemoteHousekeepingStatus.self, from: Data("{}".utf8))
        #expect(empty.cachedRevision(forModel: "Qwen/Qwen3-14B") == nil)
        // A cached repo whose refs/main was unreadable reports a null
        // revision; that is "not resolved", not an empty-string pin.
        let blank = RemoteHousekeepingStatus(
            hfCache: .init(models: [.init(modelID: "org/m", revision: "   ")]))
        #expect(blank.cachedRevision(forModel: "org/m") == nil)
    }

    // MARK: Status decode (the concurrent server agent's exact contract)

    private static let fullStatusJSON = """
        {
          "generatedAt": "2026-07-12T08:00:00Z",
          "roots": {
            "workspace": {
              "path": "/scratch/me/workspace",
              "totalBytes": 1000000000000,
              "freeBytes": 30000000000,
              "usedBytes": 970000000000,
              "warning": "lfs quota: 97% of block quota used"
            },
            "hfCache": {
              "path": "/work/lab/hf-cache",
              "totalBytes": 500000000000,
              "freeBytes": 400000000000,
              "usedBytes": 100000000000
            }
          },
          "quota": {
            "command": "lfs quota -u me /scratch",
            "output": "Disk quotas for usr me: 412G of 500G used",
            "exitCode": 0,
            "error": null,
            "ranAt": "2026-07-12T07:00:00Z"
          },
          "purgeRisk": {
            "scannedAt": "2026-07-12T07:00:00Z",
            "thresholdDays": 30,
            "warnDays": 20,
            "policySource": "site",
            "fileCount": 12,
            "totalBytes": 4200000000,
            "worst": [
              {"path": "/scratch/me/workspace/runs/old-run/generations.jsonl",
               "ageDays": 24, "bytes": 4000000000}
            ]
          },
          "hfCache": {
            "root": "/work/lab/hf-cache",
            "models": [
              {"modelId": "Qwen/Qwen3-14B", "revision": "abc123",
               "sizeBytes": 29000000000, "lastUsedAt": "2026-07-10T00:00:00Z"},
              {"modelId": "google/gemma-3-12b-it", "sizeBytes": 25000000000}
            ]
          },
          "maintenance": {
            "calendarPath": "/home/me/.steerlab/maintenance.json",
            "windows": [
              {"start": "2026-07-20T06:00:00Z", "end": "2026-07-20T18:00:00Z",
               "label": "site quarterly"}
            ],
            "next": {"start": "2026-07-20T06:00:00Z", "end": "2026-07-20T18:00:00Z",
                     "label": "site quarterly"},
            "stale": false
          },
          "evidence": {
            "bundles": [
              {"jobId": "job-9", "runId": "20260711-study",
               "path": "/scratch/me/workspace/runs/20260711-study/20260711-study.evidence-bundle.tar.gz",
               "sizeBytes": 123456, "createdAt": "2026-07-11T22:00:00Z"}
            ]
          },
          "throughput": {
            "entries": [
              {"modelId": "Qwen/Qwen3-14B", "gpuType": "A100",
               "recordsPerHour": 340.5, "samples": 4,
               "updatedAt": "2026-07-11T22:00:00Z"},
              {"modelId": "Qwen/Qwen3-14B", "gpuType": "A100",
               "recordsPerHour": 1100.0, "samples": 2,
               "updatedAt": "2026-07-11T22:00:00Z",
               "instrumentFamily": "deterministicLogprob"}
            ]
          }
        }
        """

    @Test func decodesTheFullContractPayload() throws {
        let status = try JSONDecoder().decode(
            RemoteHousekeepingStatus.self, from: Data(Self.fullStatusJSON.utf8))

        #expect(status.rootMap.count == 2)
        let workspace = try #require(status.rootMap["workspace"])
        #expect(workspace.warning?.contains("97%") == true)
        #expect(workspace.usedFraction.map { $0 > 0.95 } == true)

        let risk = try #require(status.purgeRisk)
        #expect(risk.fileCount == 12)
        #expect(risk.warnDays == 20)
        #expect(risk.worstOffenders.count == 1)
        // WP5 Step 11: the window is this SITE's declared policy, not the
        // server's 30/20 fallback, and the payload says which.
        #expect(risk.usesDefaultPolicy == false)

        let cache = try #require(status.hfCache)
        #expect(cache.modelList.count == 2)
        #expect(cache.modelList.first?.modelID == "Qwen/Qwen3-14B")
        #expect(cache.totalBytes == 54_000_000_000)

        let maintenance = try #require(status.maintenance)
        #expect(maintenance.next?.label == "site quarterly")
        #expect(maintenance.stale == false)

        #expect(status.evidence?.bundleList.first?.runId == "20260711-study")
        #expect(status.throughput?.entryList.first?.modelID == "Qwen/Qwen3-14B")
        #expect(status.generatedDate != nil)
    }

    /// Open issues §7 (2026-08-20): the server folds each finished job into a
    /// global throughput entry AND a per-instrument-family entry, so one
    /// (model, GPU) pair carries several rows with different rates. The
    /// family key is optional — a family-less entry is the global row (also
    /// all an older server sends) and labels as "all families".
    @Test func decodesPerFamilyThroughputRows() throws {
        let status = try JSONDecoder().decode(
            RemoteHousekeepingStatus.self, from: Data(Self.fullStatusJSON.utf8))
        let entries = try #require(status.throughput?.entryList)
        #expect(entries.count == 2)

        let global = entries[0]
        #expect(global.instrumentFamily == nil)
        #expect(global.familyLabel == "all families")

        let family = entries[1]
        #expect(family.modelID == "Qwen/Qwen3-14B")
        #expect(family.gpuType == "A100")
        #expect(family.instrumentFamily == "deterministicLogprob")
        #expect(family.familyLabel == "deterministicLogprob")
        #expect(family.recordsPerHour == 1100.0)
        #expect(family.samples == 2)
    }

    /// WP5 Step 11 (audit c46). The df numbers beside it are whole-filesystem
    /// and overstate the headroom on a quota'd tier; this is the number that
    /// bites. The engine never parses it — no two sites' quota tools share a
    /// format — so the client's whole job is to show the text and say where it
    /// came from.
    @Test func decodesTheSitesOwnQuotaReport() throws {
        let status = try JSONDecoder().decode(
            RemoteHousekeepingStatus.self, from: Data(Self.fullStatusJSON.utf8))
        let quota = try #require(status.quota)
        #expect(quota.command == "lfs quota -u me /scratch")
        #expect(quota.output?.contains("412G of 500G") == true)
        #expect(quota.exitCode == 0)
        #expect(quota.error == nil)
        #expect(quota.hasContent)

        // A command that failed still has content worth showing: the reason.
        let failed = HousekeepingQuota(
            command: "quota_report", output: nil, exitCode: 3,
            error: "quota command exited 3")
        #expect(failed.hasContent)
        // A site that declared no command reports nothing, and an older server
        // omits the key entirely — both decode as absence, never a failure.
        #expect(HousekeepingQuota().hasContent == false)
    }

    /// A pre-Step-11 server sends neither key. The client must read that as
    /// "this server cannot say", not as a site with no quota and no policy.
    @Test func anOlderServerOmitsTheStepElevenFieldsWithoutFailing() throws {
        let legacy = """
            {"generatedAt": "2026-07-12T08:00:00Z",
             "purgeRisk": {"thresholdDays": 30, "warnDays": 20, "fileCount": 0}}
            """
        let status = try JSONDecoder().decode(
            RemoteHousekeepingStatus.self, from: Data(legacy.utf8))
        #expect(status.quota == nil)
        #expect(status.purgeRisk?.policySource == nil)
        #expect(status.purgeRisk?.usesDefaultPolicy == false)
    }

    @Test func decodesAnEmptyOrPartialPayload() throws {
        // A server mid-first-scan (or a newer server with sections pruned)
        // must decode — absence is shown, never a decode failure.
        let empty = try JSONDecoder().decode(
            RemoteHousekeepingStatus.self, from: Data("{}".utf8))
        #expect(empty.rootMap.isEmpty)
        #expect(empty.purgeRisk == nil)
        #expect(empty.isStale())  // no generatedAt → stale by definition

        let nullSections = """
            {"generatedAt": "2026-07-12T08:00:00Z", "purgeRisk": null,
             "hfCache": null, "roots": {}}
            """
        let partial = try JSONDecoder().decode(
            RemoteHousekeepingStatus.self, from: Data(nullSections.utf8))
        #expect(partial.purgeRisk == nil)
        #expect(partial.hfCache == nil)
    }

    // MARK: Dates

    @Test func parsesISODatesWithAndWithoutFractionsAndZones() {
        #expect(HousekeepingDates.parse("2026-07-12T08:00:00Z") != nil)
        #expect(HousekeepingDates.parse("2026-07-12T08:00:00.123456Z") != nil)
        #expect(HousekeepingDates.parse("2026-07-12T08:00:00") != nil)  // naive Python isoformat
        #expect(HousekeepingDates.parse(nil) == nil)
        #expect(HousekeepingDates.parse("not a date") == nil)
        // format → parse round-trips.
        let now = Date(timeIntervalSince1970: 1_780_000_000)
        #expect(HousekeepingDates.parse(HousekeepingDates.format(now)) == now)
    }

    // MARK: Presentation rules (amber past 80 %, red past 95 %)

    @Test func storageSeverityThresholds() {
        #expect(StorageSeverity.forUsedFraction(0.5) == .ok)
        #expect(StorageSeverity.forUsedFraction(0.80) == .ok)  // "past 80%" is strict
        #expect(StorageSeverity.forUsedFraction(0.81) == .amber)
        #expect(StorageSeverity.forUsedFraction(0.95) == .amber)
        #expect(StorageSeverity.forUsedFraction(0.96) == .red)
        #expect(StorageSeverity.forUsedFraction(nil) == .ok)
    }

    @Test func serverWarningPromotesACalmRootToAmber() {
        let calm = HousekeepingRoot(totalBytes: 100, freeBytes: 90, usedBytes: 10)
        #expect(calm.severity == .ok)
        let warned = HousekeepingRoot(
            totalBytes: 100, freeBytes: 90, usedBytes: 10, warning: "inode quota at 99%")
        #expect(warned.severity == .amber)
    }

    @Test func purgeRiskIsCriticalWhenAnyFilePassesWarnAge() {
        let calm = HousekeepingPurgeRisk(
            warnDays: 20, fileCount: 3,
            worst: [.init(path: "/a", ageDays: 12, bytes: 10)])
        #expect(!calm.isCritical)
        let critical = HousekeepingPurgeRisk(
            warnDays: 20, fileCount: 3,
            worst: [.init(path: "/a", ageDays: 21, bytes: 10)])
        #expect(critical.isCritical)
        // No warn age declared → never "critical" (nothing to compare against).
        let unconfigured = HousekeepingPurgeRisk(
            fileCount: 3, worst: [.init(path: "/a", ageDays: 400, bytes: 10)])
        #expect(!unconfigured.isCritical)
    }

    @Test func stalenessUsesTheScanAge() {
        let reference = Date(timeIntervalSince1970: 1_780_000_000)
        let fresh = RemoteHousekeepingStatus(
            generatedAt: HousekeepingDates.format(reference.addingTimeInterval(-600)))
        #expect(!fresh.isStale(asOf: reference))
        let stale = RemoteHousekeepingStatus(
            generatedAt: HousekeepingDates.format(reference.addingTimeInterval(-3 * 3600)))
        #expect(stale.isStale(asOf: reference))
    }

    @Test func maintenanceCountdownDescribesFutureAndInProgressWindows() {
        let reference = Date(timeIntervalSince1970: 1_780_000_000)
        let future = MaintenanceWindow(
            start: HousekeepingDates.format(reference.addingTimeInterval(2 * 86_400 + 3 * 3600)),
            end: HousekeepingDates.format(reference.addingTimeInterval(3 * 86_400)))
        #expect(future.countdownDescription(now: reference) == "in 2d 3h")
        let active = MaintenanceWindow(
            start: HousekeepingDates.format(reference.addingTimeInterval(-600)),
            end: HousekeepingDates.format(reference.addingTimeInterval(600)))
        #expect(active.countdownDescription(now: reference) == "in progress")
        let past = MaintenanceWindow(
            start: HousekeepingDates.format(reference.addingTimeInterval(-7200)),
            end: HousekeepingDates.format(reference.addingTimeInterval(-3600)))
        #expect(past.countdownDescription(now: reference) == nil)
    }

    // MARK: Capability flags

    @Test func capabilityFlagsDecodeFromTheClusterBlock() throws {
        // The server's actual shape (profile.py): flags live inside the
        // "cluster" capability block, siblings of the Slurm facts.
        let json = """
            {"serverVersion": "1.9",
             "cluster": {"allocationScopedModelResidency": false,
                         "batchSubmitAndExit": true,
                         "housekeeping": true, "preflight": true},
             "remoteStudy": {"submitBundle": true}}
            """
        let caps = try JSONDecoder().decode(ClusterCapabilities.self, from: Data(json.utf8))
        #expect(caps.supportsHousekeeping)
        #expect(caps.supportsPreflight)
    }

    @Test func capabilityFlagsDefaultOffAndHonorAlternateSpots() throws {
        let old = try JSONDecoder().decode(
            ClusterCapabilities.self, from: Data(#"{"serverVersion":"1.0"}"#.utf8))
        #expect(!old.supportsHousekeeping)
        #expect(!old.supportsPreflight)

        // Robustness against a key move: top-level booleans and a `features`
        // object are honored too.
        let topLevel = try JSONDecoder().decode(
            ClusterCapabilities.self,
            from: Data(#"{"housekeeping": true, "preflight": true}"#.utf8))
        #expect(topLevel.supportsHousekeeping)
        #expect(topLevel.supportsPreflight)

        let nested = try JSONDecoder().decode(
            ClusterCapabilities.self,
            from: Data(#"{"features": {"housekeeping": true}}"#.utf8))
        #expect(nested.supportsHousekeeping)
        #expect(!nested.supportsPreflight)
    }

    // MARK: Preflight decode (submission responses + refusal bodies)

    @Test func submissionDecodesWithAndWithoutPreflight() throws {
        let bare = """
            {"jobId": "j1", "experiment": "s", "verb": "run", "executor": "slurm",
             "dryRun": false, "runBundle": {}, "command": [],
             "recordsDirectory": "/r", "submissionDirectory": "/s"}
            """
        let legacy = try JSONDecoder().decode(
            RemoteStudySubmission.self, from: Data(bare.utf8))
        #expect(legacy.preflight == nil)

        let withReport = """
            {"jobId": "j2", "experiment": "s", "verb": "run", "executor": "slurm",
             "dryRun": false, "runBundle": {}, "command": [],
             "recordsDirectory": "/r", "submissionDirectory": "/s",
             "preflight": {"verdict": "warn", "checks": [
                {"id": "walltime", "status": "warn",
                 "message": "estimate 9h exceeds 8h requested",
                 "data": {"estimateHours": 9}}
             ]}}
            """
        let modern = try JSONDecoder().decode(
            RemoteStudySubmission.self, from: Data(withReport.utf8))
        let preflight = try #require(modern.preflight)
        #expect(preflight.verdict == "warn")
        #expect(preflight.checks.first?.id == "walltime")
    }

    @Test func experimentJobSubmissionDecodesOptionalPreflight() throws {
        let bare = try JSONDecoder().decode(
            RemoteExperimentJobSubmission.self, from: Data(#"{"jobId":"j1"}"#.utf8))
        #expect(bare.jobId == "j1")
        #expect(bare.preflight == nil)

        let rich = try JSONDecoder().decode(
            RemoteExperimentJobSubmission.self,
            from: Data(
                #"{"jobId":"j2","preflight":{"verdict":"ok","checks":[]}}"#.utf8))
        #expect(rich.preflight?.verdict == "ok")
    }

    @Test func minesPreflightReportsOutOfRefusalBodies() throws {
        // FastAPI shape 1: detail is an object carrying the report.
        let nested = """
            {"detail": {"message": "preflight failed", "preflight": {
              "verdict": "fail",
              "checks": [{"id": "memoryFit", "status": "fail",
                          "message": "needs ≥ 39 GB: use gpu:A100:1, not gpu:L4:1"}]}}}
            """
        let mined = try #require(ClusterClient.preflightReport(fromErrorBody: nested))
        #expect(mined.verdict == "fail")
        #expect(mined.checks.first?.message.contains("gpu:A100:1") == true)

        // Shape 2: report at the top level next to detail.
        let flat = """
            {"detail": "submission refused",
             "preflight": {"verdict": "fail", "checks": []}}
            """
        #expect(ClusterClient.preflightReport(fromErrorBody: flat)?.verdict == "fail")

        // Non-preflight errors mine nothing — they must surface as themselves.
        #expect(ClusterClient.preflightReport(fromErrorBody: #"{"detail":"boom"}"#) == nil)
        #expect(ClusterClient.preflightReport(fromErrorBody: "plain text 500") == nil)
    }

    // MARK: Job-record enrichments + status classes

    @Test func jobRecordDecodesCheckpointedWithEnrichments() throws {
        // The landed server folds elapsedSeconds/recordCount INTO `result`
        // (jobs.reconcile) — the resolved accessors read that spot.
        let json = """
            {"id": "job-1", "kind": "experiment:run", "status": "checkpointed",
             "createdAt": 1.0, "logTail": [], "executor": "slurm",
             "cancellationRequested": false,
             "result": {"elapsedSeconds": 4210.5, "recordCount": 128}}
            """
        let job = try JSONDecoder().decode(RemoteJobRecord.self, from: Data(json.utf8))
        #expect(job.status == "checkpointed")
        #expect(job.resolvedElapsedSeconds == 4210.5)
        #expect(job.resolvedRecordCount == 128)

        // A future top-level promotion wins over the result-folded copy.
        let topLevel = """
            {"id": "job-3", "kind": "experiment:run", "status": "running",
             "createdAt": 1.0, "logTail": [], "executor": "slurm",
             "cancellationRequested": false, "elapsedSeconds": 60.0,
             "result": {"elapsedSeconds": 1.0}}
            """
        let promoted = try JSONDecoder().decode(
            RemoteJobRecord.self, from: Data(topLevel.utf8))
        #expect(promoted.resolvedElapsedSeconds == 60.0)

        // Older servers omit the enrichments entirely.
        let legacy = """
            {"id": "job-2", "kind": "experiment:run", "status": "running",
             "createdAt": 1.0, "logTail": [], "executor": "local",
             "cancellationRequested": false}
            """
        let old = try JSONDecoder().decode(RemoteJobRecord.self, from: Data(legacy.utf8))
        #expect(old.resolvedElapsedSeconds == nil)
        #expect(old.resolvedRecordCount == nil)
    }

    // MainActor: `jobOffersEvidenceImport` is a static on the MainActor
    // panel type.
    @Test @MainActor func checkpointedClassifiesResumableAndUnknownStaysNeutral() {
        #expect(RemoteJobStatusClass.classify(status: "checkpointed") == .resumable)
        #expect(RemoteJobStatusClass.displayText(for: "checkpointed")
            == "checkpointed (resumable)")
        #expect(RemoteJobStatusClass.classify(status: "failed") == .failed)
        #expect(RemoteJobStatusClass.classify(status: "cancelled") == .failed)
        #expect(RemoteJobStatusClass.classify(status: "cancelling") == .inFlight)
        #expect(RemoteJobStatusClass.classify(status: "running") == .inFlight)
        #expect(RemoteJobStatusClass.classify(status: "succeeded") == .succeeded)
        #expect(RemoteJobStatusClass.classify(status: "prepared") == .succeeded)
        // WS2: requeue states read as queued (in flight), never as failed.
        #expect(RemoteJobStatusClass.classify(status: "requeued") == .inFlight)
        // The rule this type exists for: a status this build has never seen
        // is neutral — NEVER rendered as a failure.
        #expect(RemoteJobStatusClass.classify(status: "hibernating") == .neutral)
        // A checkpointed job must never offer the evidence import either.
        #expect(!ExperimentPanel.jobOffersEvidenceImport(
            kind: "experiment:run", state: "checkpointed"))
    }

    // MARK: Maintenance windows round-trip

    @Test func maintenanceWindowsEncodeTheContractShape() throws {
        let window = MaintenanceWindow(
            start: "2026-07-20T06:00:00Z", end: "2026-07-20T18:00:00Z", label: "quarterly")
        let data = try JSONEncoder().encode(window)
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["start"] as? String == "2026-07-20T06:00:00Z")
        #expect(object["end"] as? String == "2026-07-20T18:00:00Z")
        #expect(object["label"] as? String == "quarterly")

        // Envelope and bare-array responses both decode client-side.
        struct Envelope: Decodable { var windows: [MaintenanceWindow] }
        let envelope = try JSONDecoder().decode(
            Envelope.self,
            from: Data(#"{"windows":[{"start":"a","end":"b"}]}"#.utf8))
        #expect(envelope.windows.count == 1)
    }
}

// MARK: - Store ↔ tunnel client routing (WS1 glue, item 1)

/// The store's `client` prefers a live tunnel's LOCAL URL for the active SSH
/// site — a reopened tunnel on a new port is picked up immediately — and
/// falls back to the stored URL when the tunnel is down or belongs to a
/// different site.
@MainActor
struct ClientTunnelRoutingTests {

    private func freshDefaults() throws -> UserDefaults {
        let suite = "steerlab.tests.tunnel-routing.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    private var sshSite: ClusterSiteProfile {
        ClusterSiteProfile(
            name: "cluster",
            transport: .ssh(
                host: "hpc.example.edu", proxyJump: nil, remotePort: 8080,
                vpnExpected: false),
            topology: .loginDaemon,
            scheduler: .slurm(ClusterSiteProfile.SlurmSiteData()),
            constraints: ClusterSiteProfile.SiteConstraints())
    }

    @Test func clientPrefersALiveTunnelForTheActiveSSHSite() async throws {
        let store = ClusterConnectionStore(defaults: try freshDefaults())
        let entry = store.addSite(sshSite)
        store.activeWorkspace = .server(entry.id)

        let tunnel = ClusterTunnel(runner: RoutingStubRunner())
        tunnel.startupGrace = .zero
        tunnel.configure(site: sshSite)
        await tunnel.open()
        guard case .up(let port) = tunnel.state else {
            Issue.record("tunnel did not come up: \(tunnel.state)")
            return
        }

        store.attachTunnel(tunnel)
        let routed = try #require(store.client)
        #expect(routed.profile.baseURL.port == port)
        #expect(routed.profile.baseURL.host() == "127.0.0.1")

        // Tunnel closed → back to the stored URL (honest failure surface).
        await tunnel.close()
        let fallback = try #require(store.client)
        #expect(fallback.profile.baseURL.absoluteString == entry.urlString
            || fallback.profile.baseURL.absoluteString == store.serverURL)
    }

    @Test func aTunnelForAnotherSiteNeverHijacksTheClient() async throws {
        let store = ClusterConnectionStore(defaults: try freshDefaults())
        let entry = store.addServer(name: "box", urlString: "http://gpu-a:8080")
        store.activeWorkspace = .server(entry.id)

        let tunnel = ClusterTunnel(runner: RoutingStubRunner())
        tunnel.startupGrace = .zero
        tunnel.configure(site: sshSite)  // a DIFFERENT site's tunnel
        await tunnel.open()
        store.attachTunnel(tunnel)

        let client = try #require(store.client)
        #expect(client.profile.baseURL.host() == "gpu-a")
        await tunnel.close()
    }
}

/// Minimal scripted ssh: master always alive, every port free, launches a
/// handle that stays up until terminated.
private actor RoutingStubRunner: TunnelProcessRunner {
    func run(_ executablePath: String, arguments: [String]) async -> TunnelProcessResult {
        TunnelProcessResult(exitCode: 0, standardOutput: "", standardError: "")
    }

    func run(
        _ executablePath: String, arguments: [String], input: Data
    ) async -> TunnelProcessResult {
        await run(executablePath, arguments: arguments)
    }

    func launch(
        _ executablePath: String, arguments: [String]
    ) async throws -> any TunnelProcessHandle {
        RoutingStubHandle()
    }

    func isLocalPortFree(_ port: Int) async -> Bool { true }
}

private actor RoutingStubHandle: TunnelProcessHandle {
    private var running = true
    private var waiters: [CheckedContinuation<Int32, Never>] = []

    var isRunning: Bool { running }

    func terminate() {
        guard running else { return }
        running = false
        let pending = waiters
        waiters.removeAll()
        for waiter in pending { waiter.resume(returning: 143) }
    }

    func waitUntilExit() async -> Int32 {
        if !running { return 143 }
        return await withCheckedContinuation { waiters.append($0) }
    }
}
