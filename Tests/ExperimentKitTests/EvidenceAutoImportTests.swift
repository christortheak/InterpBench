import Foundation
import Testing

@testable import ExperimentKit

/// WS3 evidence auto-import: ledger round-trip, candidate extraction from
/// the job payloads, skip-already-imported, and visible failure retention
/// with capped backoff. Everything runs against injected seams — no
/// networking, no real server, no real bundles.
@MainActor
struct EvidenceAutoImportTests {

    private func freshWorkspace(_ name: String) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appending(components: "steerlab-tests-autoimport", "\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func succeededRunJob(
        id: String, bundlePath: String, sha: String? = "abc123",
        runDirectory: String? = nil
    ) -> RemoteJobRecord {
        var evidence: [String: JSONValue] = ["bundlePath": .string(bundlePath)]
        if let sha { evidence["bundleSha256"] = .string(sha) }
        var runResult: [String: JSONValue] = ["evidenceBundle": .object(evidence)]
        if let runDirectory { runResult["runDirectory"] = .string(runDirectory) }
        return RemoteJobRecord(
            id: id, kind: "experiment:run", status: "succeeded", createdAt: 1,
            startedAt: nil, finishedAt: 2,
            result: ["runResult": .object(runResult)],
            error: nil, logTail: [], executor: "slurm", executorJobID: nil,
            cancellationRequested: false)
    }

    // MARK: Ledger round-trip

    @Test func ledgerRoundTripsAndToleratesAbsence() throws {
        let workspace = try freshWorkspace("ledger")
        let url = EvidenceAutoImportService.ledgerURL(workspaceRoot: workspace)
        // Absent file → empty ledger, no throw (create-on-first-write).
        #expect(EvidenceAutoImportService.loadLedger(at: url).isEmpty)

        let entries = [
            EvidenceAutoImportService.LedgerEntry(
                bundlePath: "/remote/runs/a/a.evidence-bundle.tar.gz",
                runId: "a", importedAt: "2026-07-12T08:00:00Z", sha256: "deadbeef"),
            EvidenceAutoImportService.LedgerEntry(
                bundlePath: "/remote/runs/b/b.evidence-bundle.tar.gz",
                runId: "b", importedAt: "2026-07-12T09:00:00Z", sha256: nil),
        ]
        try EvidenceAutoImportService.saveLedger(entries, to: url)
        #expect(EvidenceAutoImportService.loadLedger(at: url) == entries)

        // The on-disk shape is the documented contract: a bare array of
        // {bundlePath, runId, importedAt, sha256} objects.
        let raw = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [[String: Any]])
        #expect(raw.count == 2)
        #expect(raw.first?["bundlePath"] as? String != nil)
    }

    // MARK: Candidate extraction

    @Test func extractsCandidatesFromSucceededRunJobsOnly() {
        let good = succeededRunJob(
            id: "j1",
            bundlePath: "/remote/runs/20260711-s/20260711-s.evidence-bundle.tar.gz",
            runDirectory: "/remote/runs/20260711-s")
        let candidate = EvidenceAutoImportService.candidate(fromJob: good)
        #expect(candidate?.bundlePath.hasSuffix("evidence-bundle.tar.gz") == true)
        #expect(candidate?.runId == "20260711-s")
        #expect(candidate?.sha256 == "abc123")
        #expect(candidate?.jobId == "j1")

        // Non-run kinds, non-terminal states, dry runs, and bundle-less
        // results never qualify.
        var sweep = good
        sweep = RemoteJobRecord(
            id: "j2", kind: "experiment:sweep", status: "succeeded", createdAt: 1,
            startedAt: nil, finishedAt: 2, result: sweep.result, error: nil,
            logTail: [], executor: "slurm", executorJobID: nil,
            cancellationRequested: false)
        #expect(EvidenceAutoImportService.candidate(fromJob: sweep) == nil)

        var checkpointed = good
        checkpointed.status = "checkpointed"
        #expect(EvidenceAutoImportService.candidate(fromJob: checkpointed) == nil)

        var noBundle = good
        noBundle.result = ["runResult": .object([:])]
        #expect(EvidenceAutoImportService.candidate(fromJob: noBundle) == nil)
    }

    // MARK: Partial (failure-record) retrieval — retention 2026-07-24

    private func failedJobWithPartialEvidence(
        id: String = "jf", partialMarker: Bool = true,
        error: String? = "RuntimeError: CUDA out of memory"
    ) -> RemoteJobRecord {
        var result: [String: JSONValue] = [
            "evidenceBundle": .object([
                "bundlePath": .string(
                    "/remote/runs/20260724-s/20260724-s.partial.evidence-bundle.tar.gz"),
                "bundleSha256": .string("cafe01"),
                "evidenceComplete": .bool(false),
            ]),
            "runDirectory": .string("/remote/runs/20260724-s"),
        ]
        if partialMarker { result["partialEvidence"] = .bool(true) }
        return RemoteJobRecord(
            id: id, kind: "experiment:run", status: "failed", createdAt: 1,
            startedAt: nil, finishedAt: 2, result: result, error: error,
            logTail: [], executor: "slurm", executorJobID: nil,
            cancellationRequested: false)
    }

    @Test func failedJobWithPackagedEvidenceOffersRetrieval() {
        let job = failedJobWithPartialEvidence()
        #expect(ExperimentPanel.jobOffersPartialEvidenceImport(job))
        // ... but it is NOT a result: the completed-run predicate that every
        // evidence-grade surface is wired to still refuses it.
        #expect(!ExperimentPanel.jobOffersEvidenceImport(
            kind: job.kind, state: job.status))

        let candidate = EvidenceAutoImportService.candidate(fromJob: job)
        #expect(candidate?.isPartial == true)
        #expect(candidate?.runId == "20260724-s")
        #expect(candidate?.sha256 == "cafe01")
        #expect(candidate?.failureSummary?.contains("out of memory") == true)
    }

    @Test func succeededJobsAreNeverMarkedPartial() {
        let good = succeededRunJob(
            id: "j1", bundlePath: "/remote/runs/a/a.evidence-bundle.tar.gz",
            runDirectory: "/remote/runs/a")
        #expect(!ExperimentPanel.jobOffersPartialEvidenceImport(good))
        #expect(EvidenceAutoImportService.candidate(fromJob: good)?.isPartial
            == false)
    }

    @Test func failedJobWithoutThePartialMarkerOffersNothing() {
        // A bundle on a failed record without the explicit marker could be
        // a stale artifact from an earlier attempt. Presenting that as THIS
        // failure's evidence would be its own small dishonesty, so the
        // marker is required rather than inferred.
        let job = failedJobWithPartialEvidence(partialMarker: false)
        #expect(!ExperimentPanel.jobOffersPartialEvidenceImport(job))
        #expect(EvidenceAutoImportService.candidate(fromJob: job) == nil)
    }

    // MARK: Targeted retry affordance (review finding 3, 2026-07-24)

    @Test func failedEvaluateOffersARetry() {
        var result: [String: JSONValue] = [
            "evidenceBundle": .object([
                "bundlePath": .string("/r/x.partial.evidence-bundle.tar.gz"),
            ]),
            "partialEvidence": .bool(true),
            "verb": .string("evaluate"),
            "experiment": .string("alien-stance"),
            "partialRunID": .string("20260724-exp-alien-stance-evaluate"),
        ]
        let job = RemoteJobRecord(
            id: "j1", kind: "experiment:evaluate", status: "failed",
            createdAt: 1, startedAt: nil, finishedAt: 2, result: result,
            error: "judge died", logTail: [], executor: "local",
            executorJobID: nil, cancellationRequested: false)
        let retry = job.retryableEvaluate
        #expect(retry?.experiment == "alien-stance")
        #expect(retry?.partialRunID == "20260724-exp-alien-stance-evaluate")

        // Other verbs have no notion of "cells never decided", so offering
        // retry for them would be a button that cannot work.
        result["verb"] = .string("sweep")
        var sweep = job
        sweep.result = result
        #expect(sweep.retryableEvaluate == nil)
    }

    @Test func aSucceededJobIsNeverRetryable() {
        let good = succeededRunJob(
            id: "j1", bundlePath: "/r/a.evidence-bundle.tar.gz",
            runDirectory: "/r/a")
        #expect(good.retryableEvaluate == nil)
    }

    @Test func partialImportsAreMarkedInTheLedger() throws {
        let workspace = try freshWorkspace("partial-ledger")
        let url = EvidenceAutoImportService.ledgerURL(workspaceRoot: workspace)
        let entries = [
            EvidenceAutoImportService.LedgerEntry(
                bundlePath: "/remote/runs/a/a.evidence-bundle.tar.gz",
                runId: "a", importedAt: "2026-07-24T08:00:00Z",
                sha256: "deadbeef"),
            EvidenceAutoImportService.LedgerEntry(
                bundlePath: "/remote/runs/b/b.partial.evidence-bundle.tar.gz",
                runId: "b", importedAt: "2026-07-24T09:00:00Z",
                sha256: nil, isPartial: true),
        ]
        try EvidenceAutoImportService.saveLedger(entries, to: url)
        let reloaded = EvidenceAutoImportService.loadLedger(at: url)
        #expect(reloaded == entries)
        #expect(reloaded.first?.isPartial == false)
        #expect(reloaded.last?.isPartial == true)
    }

    @Test func legacyLedgersDecodeAsComplete() throws {
        // Every entry written before partial import existed was, correctly,
        // a completed run — the absent key must not read as "partial".
        let workspace = try freshWorkspace("legacy-ledger")
        let url = EvidenceAutoImportService.ledgerURL(workspaceRoot: workspace)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(#"""
            [{"bundlePath": "/r/a.evidence-bundle.tar.gz", "runId": "a",
              "importedAt": "2026-07-12T08:00:00Z"}]
            """#.utf8).write(to: url)
        let entries = EvidenceAutoImportService.loadLedger(at: url)
        #expect(entries.count == 1)
        #expect(entries.first?.isPartial == false)
    }

    @Test func derivesRunIDFromBundleFilenames() {
        #expect(
            EvidenceAutoImportService.runID(
                fromBundlePath: "/x/20260711-study.evidence-bundle.tar.gz")
                == "20260711-study")
        #expect(EvidenceAutoImportService.runID(fromBundlePath: "/x/random.tar.gz") == nil)
    }

    @Test func housekeepingBundlesBecomeCandidatesAndFilterAgainstLedger() async throws {
        let workspace = try freshWorkspace("pending")
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace,
            fetchJobs: { [] },
            performImport: { _ in workspace })
        let bundles = [
            HousekeepingEvidenceBundle(
                jobId: "j1", runId: "run-a", path: "/remote/a.evidence-bundle.tar.gz"),
            HousekeepingEvidenceBundle(
                jobId: "j2", path: "/remote/run-b.evidence-bundle.tar.gz"),
        ]
        #expect(service.pendingCandidates(fromHousekeepingBundles: bundles).count == 2)

        // Importing one drops it from pending (ledger-backed, persisted).
        _ = await service.importNow(candidates: [
            EvidenceCandidate(bundlePath: "/remote/a.evidence-bundle.tar.gz", runId: "run-a")
        ])
        let pending = service.pendingCandidates(fromHousekeepingBundles: bundles)
        #expect(pending.map(\.bundlePath) == ["/remote/run-b.evidence-bundle.tar.gz"])
    }

    // MARK: Skip-already-imported

    @Test func skipsBundlesTheLedgerAlreadyKnows() async throws {
        let workspace = try freshWorkspace("skip")
        let bundlePath = "/remote/runs/x/x.evidence-bundle.tar.gz"
        try EvidenceAutoImportService.saveLedger(
            [.init(bundlePath: bundlePath, runId: "x",
                   importedAt: "2026-07-12T08:00:00Z", sha256: nil)],
            to: EvidenceAutoImportService.ledgerURL(workspaceRoot: workspace))

        let importCounter = Counter()
        let job = succeededRunJob(id: "j1", bundlePath: bundlePath)
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace,
            fetchJobs: { [job] },
            performImport: { _ in
                await importCounter.increment()
                return workspace
            })

        let events = await service.runOnce(force: true)
        #expect(events.isEmpty)  // nothing to do, nothing invented
        #expect(await importCounter.value == 0)
        #expect(service.isImported(bundlePath: bundlePath))
    }

    @Test func importerCollisionRefusalIsRecordedAsAlreadyPresent() async throws {
        // The verified importer refuses to overwrite an existing run (e.g. a
        // manual import beat us to it) — that outcome ledgers the bundle and
        // never retries, instead of looping on a permanent failure.
        let workspace = try freshWorkspace("collision")
        let job = succeededRunJob(
            id: "j1", bundlePath: "/remote/runs/y/y.evidence-bundle.tar.gz")
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace,
            fetchJobs: { [job] },
            performImport: { _ in
                throw ChatServiceError(reason: "refusing to overwrite existing run y")
            })

        let events = await service.runOnce(force: true)
        #expect(events.count == 1)
        #expect(events.first?.outcome == .skippedAlreadyPresent)
        #expect(service.isImported(bundlePath: "/remote/runs/y/y.evidence-bundle.tar.gz"))
        #expect(service.failures.isEmpty)
    }

    // MARK: One-click pipeline import (dead/parked chains, 2026-08-06)

    @Test func importPipelinePackagesThenRunsTheVerifiedImportPath() async throws {
        // The dead-chain affordance: package on the server (the
        // packager walks the ledger to every completed stage run), then the
        // same verified path auto-import uses — including the revision
        // adoption inside performImport. The candidate carries the
        // server-stamped hash and the completeness tier.
        let workspace = try freshWorkspace("pipeline-import")
        let runID = "20260806T015828917-exp-replication-1-pipeline"
        let bundlePath = "/data/runs/\(runID)/\(runID).evidence-bundle.tar.gz"
        let imported = ImportedCandidateBox()
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace,
            performImport: { candidate in
                await imported.record(candidate)
                return workspace
            },
            packageEvidence: { runDirectory in
                #expect(runDirectory == runID)
                return .init(bundlePath: bundlePath,
                             bundleSha256: "deadbeef", runID: runID,
                             evidenceComplete: true, missingEvidence: nil)
            })

        let event = await service.importPipeline(runID: runID)
        #expect(event?.outcome == .imported(runDirectory: workspace.path))
        let candidate = try #require(await imported.value)
        #expect(candidate.bundlePath == bundlePath)
        #expect(candidate.runId == runID)
        #expect(candidate.sha256 == "deadbeef")
        #expect(!candidate.isPartial)
        #expect(service.isImported(bundlePath: bundlePath))
    }

    @Test func importPipelineMarksIncompleteBundlesPartial() async throws {
        // A chain the server could not finish still comes home — but as a
        // PARTIAL, never as a completed result.
        let workspace = try freshWorkspace("pipeline-partial")
        let imported = ImportedCandidateBox()
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace,
            performImport: { candidate in
                await imported.record(candidate)
                return workspace
            },
            packageEvidence: { _ in
                .init(bundlePath: "/data/runs/c/c.evidence-bundle.tar.gz",
                      bundleSha256: nil, runID: "c",
                      evidenceComplete: false,
                      missingEvidence: ["stage 'analyze': not found"])
            })
        _ = await service.importPipeline(runID: "c")
        #expect(await imported.value?.isPartial == true)
        #expect(service.ledgerEntries.first?.isPartial == true)
    }

    @Test func importPipelineStructuredSkipIsNotedNeverFailed() async throws {
        // A 2026-08-11 memo-study import: the server answers a
        // structured skip for a ledger-only failure record (a refused
        // continuation's pipeline dir). The import notes it in the summary
        // — no failure, no retry backoff, and no ledger entry (a later
        // resume can still produce evidence; a ledger entry would suppress
        // its import).
        let workspace = try freshWorkspace("pipeline-skip")
        let imported = ImportedCandidateBox()
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace,
            performImport: { candidate in
                await imported.record(candidate)
                return workspace
            },
            packageEvidence: { _ in
                .init(bundlePath: nil, bundleSha256: nil,
                      runID: "20260811T122558132-exp-c20-doctrine-memo-pipeline",
                      evidenceComplete: nil, missingEvidence: nil,
                      skipped: true,
                      reason: "pipeline failure record with no stage outputs "
                          + "— nothing to bundle beyond the ledger snapshot")
            })
        let event = await service.importPipeline(runID: "r")
        #expect(event?.outcome == .skippedUnbundleable(
            note: "pipeline failure record with no stage outputs "
                + "— nothing to bundle beyond the ledger snapshot"))
        #expect(event?.runId
            == "20260811T122558132-exp-c20-doctrine-memo-pipeline")
        #expect(await imported.value == nil)  // nothing was downloaded
        #expect(service.ledgerEntries.isEmpty)
        #expect(service.failures.isEmpty)
        #expect(service.lastSummary?.contains("skipped") == true)
        #expect(service.lastSummary?.contains("nothing to bundle") == true)
    }

    @Test func importPipelinePackagingFailureIsLoudAndProducesNoEvent() async throws {
        let workspace = try freshWorkspace("pipeline-pack-fail")
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace,
            performImport: { _ in workspace },
            packageEvidence: { _ in
                throw ChatServiceError(reason: "runs root refused the path")
            })
        let event = await service.importPipeline(runID: "x")
        #expect(event == nil)
        #expect(service.lastSummary?.contains("could not package") == true)
        #expect(service.ledgerEntries.isEmpty)
    }

    // MARK: Failure retention + capped backoff

    @Test func failuresAreRetainedBackedOffAndCapped() async throws {
        let workspace = try freshWorkspace("failures")
        let bundlePath = "/remote/runs/z/z.evidence-bundle.tar.gz"
        let job = succeededRunJob(id: "j1", bundlePath: bundlePath, sha: nil)

        let clock = MutableClock(start: Date(timeIntervalSince1970: 1_780_000_000))
        let importCounter = Counter()
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace,
            fetchJobs: { [job] },
            performImport: { _ in
                await importCounter.increment()
                throw ChatServiceError(reason: "download failed: connection reset")
            },
            now: { clock.now })
        service.configuration.retryBase = 120
        service.configuration.maxAttempts = 2

        // First pass: one attempt, one visible failure.
        var events = await service.runOnce(force: true)
        #expect(events.count == 1)
        if case .failed(let message)? = events.first?.outcome {
            #expect(message.contains("connection reset"))
        } else {
            Issue.record("expected a failed outcome")
        }
        let failure = try #require(service.failure(forBundlePath: bundlePath))
        #expect(failure.attempts == 1)
        #expect(!failure.exhausted)
        #expect(await importCounter.value == 1)

        // Within the backoff window: no retry, failure stays visible.
        events = await service.runOnce(force: true)
        #expect(events.isEmpty)
        #expect(await importCounter.value == 1)

        // Past the backoff window: retried once more, then capped out
        // (maxAttempts 2) — still visible, marked exhausted, no more retries.
        clock.advance(by: 3600)
        events = await service.runOnce(force: true)
        #expect(events.count == 1)
        #expect(await importCounter.value == 2)
        let exhausted = try #require(service.failure(forBundlePath: bundlePath))
        #expect(exhausted.attempts == 2)
        #expect(exhausted.exhausted)

        clock.advance(by: 100_000)
        events = await service.runOnce(force: true)
        #expect(events.isEmpty)
        #expect(await importCounter.value == 2)
        #expect(!service.isImported(bundlePath: bundlePath))
    }

    @Test func successAfterFailureClearsTheFailureAndLedgers() async throws {
        let workspace = try freshWorkspace("recovery")
        let bundlePath = "/remote/runs/w/w.evidence-bundle.tar.gz"
        let job = succeededRunJob(id: "j1", bundlePath: bundlePath)
        let clock = MutableClock(start: Date(timeIntervalSince1970: 1_780_000_000))
        let gate = FailureGate(failuresBeforeSuccess: 1)
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace,
            fetchJobs: { [job] },
            performImport: { _ in
                if await gate.shouldFail() {
                    throw ChatServiceError(reason: "transient")
                }
                return workspace.appending(component: "runs-w")
            },
            now: { clock.now })

        _ = await service.runOnce(force: true)
        #expect(service.failure(forBundlePath: bundlePath) != nil)

        clock.advance(by: 7200)
        let events = await service.runOnce(force: true)
        #expect(events.count == 1)
        if case .imported(let directory)? = events.first?.outcome {
            #expect(directory.hasSuffix("runs-w"))
        } else {
            Issue.record("expected an imported outcome")
        }
        #expect(service.failure(forBundlePath: bundlePath) == nil)
        #expect(service.isImported(bundlePath: bundlePath))
        // Persisted, not just in memory.
        let reloaded = EvidenceAutoImportService.loadLedger(
            at: EvidenceAutoImportService.ledgerURL(workspaceRoot: workspace))
        #expect(reloaded.contains { $0.bundlePath == bundlePath && $0.sha256 == "abc123" })
    }

    // MARK: Per-site flag defaults (store-side registration config)

    @Test func autoImportDefaultsOnForSSHSitesOffForDirect() throws {
        let suite = "steerlab.tests.autoimport-flags.\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        let store = clusterStore(defaults: defaults)

        let ssh = store.addSite(
            ClusterSiteProfile(
                name: "cluster",
                transport: .ssh(host: "hpc.example.edu", proxyJump: nil,
                                remotePort: 8080, vpnExpected: false),
                topology: .loginDaemon,
                scheduler: .slurm(ClusterSiteProfile.SlurmSiteData()),
                constraints: ClusterSiteProfile.SiteConstraints()))
        let direct = store.addServer(name: "box", urlString: "http://127.0.0.1:8080")

        #expect(store.autoImportEnabled(for: ssh))       // remote: results come home
        #expect(!store.autoImportEnabled(for: direct))   // localhost may BE the workspace

        store.setAutoImportEnabled(false, for: ssh)
        #expect(!store.autoImportEnabled(for: ssh))
        store.setAutoImportEnabled(true, for: direct)
        #expect(store.autoImportEnabled(for: direct))
        defaults.removePersistentDomain(forName: suite)
    }
}

// MARK: - Test doubles

private actor Counter {
    private(set) var value = 0
    func increment() { value += 1 }
}

private actor ImportedCandidateBox {
    private(set) var value: EvidenceCandidate?
    func record(_ candidate: EvidenceCandidate) { value = candidate }
}

private actor FailureGate {
    private var remaining: Int
    init(failuresBeforeSuccess: Int) { remaining = failuresBeforeSuccess }
    func shouldFail() -> Bool {
        guard remaining > 0 else { return false }
        remaining -= 1
        return true
    }
}

/// Sendable mutable clock for backoff tests (lock-guarded, no actor hop so
/// the `now` closure stays synchronous).
private final class MutableClock: @unchecked Sendable {
    private let lock = NSLock()
    private var current: Date
    init(start: Date) { current = start }
    var now: Date {
        lock.lock()
        defer { lock.unlock() }
        return current
    }
    func advance(by seconds: TimeInterval) {
        lock.lock()
        defer { lock.unlock() }
        current = current.addingTimeInterval(seconds)
    }
}
