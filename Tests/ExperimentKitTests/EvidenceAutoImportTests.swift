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

    private func source(_ workspace: URL) -> EvidenceImportOrigin {
        .init(serverIdentity: "ssh://researcher@example.invalid:8080", remoteRoot: "/remote", workspaceRoot: workspace)
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

    @Test func housekeepingPartialFilenamesStayFailureRecords() {
        let candidate = EvidenceAutoImportService.candidates(fromHousekeepingBundles: [
            .init(runId: "run.partial", path: "/remote/run.partial.evidence-bundle.tar.gz")
        ]).first
        #expect(candidate?.isPartial == true)
        #expect(candidate?.runId == "run")
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
            workspaceRoot: workspace, originProvider: { source(workspace) },
            fetchJobs: { [] },
            performImport: { _ in workspace })
        let bundles = [
            HousekeepingEvidenceBundle(
                jobId: "j1", runId: "run-a", path: "/remote/a.evidence-bundle.tar.gz"),
            HousekeepingEvidenceBundle(
                jobId: "j2", path: "/remote/run-b.evidence-bundle.tar.gz"),
        ]
        #expect(service.pendingCandidates(fromHousekeepingBundles: bundles, origin: source(workspace)).count == 2)

        // A path-only listing cannot prove the current remote bytes match an
        // earlier import, so it remains pending until a stamped listing exists.
        _ = await service.importNow(candidates: [
            EvidenceCandidate(bundlePath: "/remote/a.evidence-bundle.tar.gz", runId: "run-a")
        ], origin: source(workspace))
        let pending = service.pendingCandidates(fromHousekeepingBundles: bundles, origin: source(workspace))
        #expect(pending.count == 2)
    }

    // MARK: Skip-already-imported

    @Test func skipsBundlesTheLedgerAlreadyKnows() async throws {
        let workspace = try freshWorkspace("skip")
        let bundlePath = "/remote/runs/x/x.evidence-bundle.tar.gz"
        try EvidenceAutoImportService.saveLedger(
            [.init(bundlePath: bundlePath, runId: "x",
                   importedAt: "2026-07-12T08:00:00Z", sha256: "abc123", contentsVerified: true, origin: source(workspace))],
            to: EvidenceAutoImportService.ledgerURL(workspaceRoot: workspace))

        let importCounter = Counter()
        let job = succeededRunJob(id: "j1", bundlePath: bundlePath)
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace, originProvider: { source(workspace) },
            fetchJobs: { [job] },
            performImport: { _ in
                await importCounter.increment()
                return workspace
            })

        let events = await service.runOnce(force: true)
        #expect(events.isEmpty)  // nothing to do, nothing invented
        #expect(await importCounter.value == 0)
        #expect(service.isImported(candidate: .init(bundlePath: bundlePath, sha256: "abc123"), origin: source(workspace)))
    }

    @Test func legacyPresenceOnlyLedgerCannotSuppressVerification() async throws {
        let workspace = try freshWorkspace("legacy-unverified")
        defer { try? FileManager.default.removeItem(at: workspace) }
        let path = "/remote/run.evidence-bundle.tar.gz"
        try EvidenceAutoImportService.saveLedger([
            .init(bundlePath: path, runId: "run", importedAt: "2026-07-12T08:00:00Z", sha256: nil)
        ], to: EvidenceAutoImportService.ledgerURL(workspaceRoot: workspace))
        let counter = Counter()
        let service = EvidenceAutoImportService(workspaceRoot: workspace, originProvider: { source(workspace) }, performImport: { _ in
            await counter.increment()
            throw ChatServiceError(reason: "content verification failed")
        })
        #expect(!service.isImported(candidate: .init(bundlePath: path, sha256: "abc123"), origin: source(workspace)))
        #expect(service.importedRunIDs(origin: source(workspace)).isEmpty)
        _ = await service.importNow(candidates: [.init(bundlePath: path, runId: "run")], origin: source(workspace))
        #expect(await counter.value == 1)
        #expect(!service.isImported(candidate: .init(bundlePath: path, sha256: "abc123"), origin: source(workspace)))
        #expect(service.ledgerEntries.count == 1)
    }

    @Test func importerCollisionRefusalIsNotEvidenceOfLocalCustody() async throws {
        // A refusal proves no content match. It must stay a visible failure
        // and must never create a ledger entry that later authorizes cleanup.
        let workspace = try freshWorkspace("collision")
        let job = succeededRunJob(
            id: "j1", bundlePath: "/remote/runs/y/y.evidence-bundle.tar.gz")
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace, originProvider: { source(workspace) },
            fetchJobs: { [job] },
            performImport: { _ in
                throw ChatServiceError(reason: "refusing to overwrite existing run y")
            })

        let events = await service.runOnce(force: true)
        #expect(events.count == 1)
        guard case .failed? = events.first?.outcome else {
            Issue.record("An overwrite refusal must remain a failed import")
            return
        }
        #expect(!service.isImported(candidate: .init(bundlePath: "/remote/runs/y/y.evidence-bundle.tar.gz", sha256: "abc123"), origin: source(workspace)))
        #expect(service.failures.count == 1)
    }

    @Test func anExistingEmptyRunStillReachesTheVerifier() async throws {
        let workspace = try freshWorkspace("empty-run")
        defer { try? FileManager.default.removeItem(at: workspace) }
        try FileManager.default.createDirectory(
            at: workspace.appending(components: "runs", "run"), withIntermediateDirectories: true)
        let counter = Counter()
        let service = EvidenceAutoImportService(workspaceRoot: workspace, originProvider: { source(workspace) }, performImport: { _ in
            await counter.increment()
            throw ChatServiceError(reason: "existing run lacks declared evidence")
        })
        let events = await service.importNow(candidates: [
            EvidenceCandidate(bundlePath: "/remote/run.evidence-bundle.tar.gz", runId: "run")
        ], origin: source(workspace))
        #expect(await counter.value == 1)
        #expect(service.ledgerEntries.isEmpty)
        #expect(events.count == 1)
        guard case .failed? = events.first?.outcome else {
            Issue.record("Empty local directories must not bypass verification")
            return
        }
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
            workspaceRoot: workspace, originProvider: { source(workspace) },
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

        let event = await service.importPipeline(runID: runID, origin: source(workspace))
        #expect(event?.outcome == .imported(runDirectory: workspace.path))
        let candidate = try #require(await imported.value)
        #expect(candidate.bundlePath == bundlePath)
        #expect(candidate.runId == runID)
        #expect(candidate.sha256 == "deadbeef")
        #expect(!candidate.isPartial)
        #expect(service.isImported(candidate: .init(bundlePath: bundlePath, sha256: "deadbeef"), origin: source(workspace)))
    }

    @Test func importPipelineMarksIncompleteBundlesPartial() async throws {
        // A chain the server could not finish still comes home — but as a
        // PARTIAL, never as a completed result.
        let workspace = try freshWorkspace("pipeline-partial")
        let imported = ImportedCandidateBox()
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace, originProvider: { source(workspace) },
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
        _ = await service.importPipeline(runID: "c", origin: source(workspace))
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
            workspaceRoot: workspace, originProvider: { source(workspace) },
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
        let event = await service.importPipeline(runID: "r", origin: source(workspace))
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
            workspaceRoot: workspace, originProvider: { source(workspace) },
            performImport: { _ in workspace },
            packageEvidence: { _ in
                throw ChatServiceError(reason: "runs root refused the path")
            })
        let event = await service.importPipeline(runID: "x", origin: source(workspace))
        #expect(event == nil)
        #expect(service.lastSummary?.contains("could not package") == true)
        #expect(service.ledgerEntries.isEmpty)
    }

    @Test func sameRemotePathIsScopedByServerAndSurvivesRestart() async throws {
        let workspace = try freshWorkspace("server-identity")
        defer { try? FileManager.default.removeItem(at: workspace) }
        let first = source(workspace)
        let second = EvidenceImportOrigin(serverIdentity: "ssh://researcher@second.invalid:8080", remoteRoot: "/remote", workspaceRoot: workspace)
        let selected = SelectedImportOrigin(first)
        let counter = Counter()
        let job = succeededRunJob(id: "same-job", bundlePath: "/remote/same.evidence-bundle.tar.gz")
        let candidate = try #require(EvidenceAutoImportService.candidate(fromJob: job))
        let service = EvidenceAutoImportService(workspaceRoot: workspace,
            originProvider: { selected.value }, fetchJobs: { [job] }, performImport: { _ in
                await counter.increment()
                return workspace
            })
        _ = await service.runOnce(force: true)
        selected.value = second
        #expect(!service.isImported(candidate: candidate, origin: second))
        _ = await service.runOnce(force: true)
        #expect(await counter.value == 2)
        #expect(service.ledgerEntries.map(\.origin) == [first, second])
        let restarted = EvidenceAutoImportService(workspaceRoot: workspace,
            originProvider: { selected.value }, fetchJobs: { [job] }, performImport: { _ in
                await counter.increment()
                return workspace
            })
        #expect(await restarted.runOnce(force: true).isEmpty)
        #expect(await counter.value == 2)
        #expect(restarted.isImported(candidate: candidate, origin: first))
        #expect(restarted.isImported(candidate: candidate, origin: second))
    }

    @Test func verifiedButUnscopedLegacyReceiptDoesNotAuthorizeAnotherOrigin() async throws {
        let workspace = try freshWorkspace("unscoped-ledger")
        defer { try? FileManager.default.removeItem(at: workspace) }
        let origin = source(workspace)
        let candidate = EvidenceCandidate(bundlePath: "/remote/evidence.tar.gz", sha256: "stamp")
        try EvidenceAutoImportService.saveLedger([
            .init(bundlePath: candidate.bundlePath, runId: "run", importedAt: "2026-09-05T00:00:00Z",
                  sha256: candidate.sha256, contentsVerified: true)
        ], to: EvidenceAutoImportService.ledgerURL(workspaceRoot: workspace))
        let service = EvidenceAutoImportService(workspaceRoot: workspace,
            originProvider: { origin }, performImport: { _ in workspace })
        #expect(!service.isImported(candidate: candidate, origin: origin))
        _ = await service.importNow(candidates: [candidate], origin: origin)
        #expect(service.ledgerEntries.count == 2)
        #expect(service.ledgerEntries.first?.origin == nil)
        #expect(service.ledgerEntries.last?.origin == origin)
    }

    @Test func changedRemoteBytesAndUnknownOriginsCannotReuseALedgerEntry() async throws {
        let workspace = try freshWorkspace("bundle-versions")
        defer { try? FileManager.default.removeItem(at: workspace) }
        let origin = source(workspace)
        let candidate = EvidenceCandidate(bundlePath: "/remote/evidence.tar.gz", sha256: "first")
        let service = EvidenceAutoImportService(workspaceRoot: workspace,
            originProvider: { origin }, performImport: { _ in workspace })
        _ = await service.importNow(candidates: [candidate], origin: origin)
        #expect(service.isImported(candidate: candidate, origin: origin))
        var changed = candidate
        changed.sha256 = "second"
        #expect(!service.isImported(candidate: changed, origin: origin))
        let unknown = EvidenceImportOrigin(serverIdentity: origin.serverIdentity, remoteRoot: nil, workspaceRoot: workspace)
        #expect(!service.isImported(candidate: candidate, origin: unknown))
        let moved = EvidenceImportOrigin(serverIdentity: origin.serverIdentity, remoteRoot: "/other", workspaceRoot: workspace)
        #expect(!service.isImported(candidate: candidate, origin: moved))
        _ = await service.importNow(candidates: [changed], origin: origin)
        #expect(service.ledgerEntries.count == 2)
    }

    @Test func aSelectionChangeDuringListingRefusesBeforeImport() async throws {
        let workspace = try freshWorkspace("late-listing")
        defer { try? FileManager.default.removeItem(at: workspace) }
        let origin = source(workspace)
        let selected = SelectedImportOrigin(origin)
        let job = succeededRunJob(id: "job", bundlePath: "/remote/evidence.tar.gz")
        let counter = Counter()
        let service = EvidenceAutoImportService(workspaceRoot: workspace,
            originProvider: { selected.value }, fetchJobs: {
                await selected.clear()
                return [job]
            }, performImport: { _ in
                await counter.increment()
                return workspace
            })
        let events = await service.runOnce(force: true)
        #expect(events.first?.outcome == .refused(code: "evidenceContextChanged", repairAction: EvidenceImportOrigin.changedRepair))
        #expect(await counter.value == 0)
        #expect(service.ledgerEntries.isEmpty)
    }

    @Test(arguments: ["disconnected", "server", "remote-root", "local-workspace"])
    func staleManualActionsNeverPackageOrDownload(change: String) async throws {
        let workspace = try freshWorkspace("stale-actions")
        defer { try? FileManager.default.removeItem(at: workspace) }
        let origin = source(workspace)
        let current: EvidenceImportOrigin? = change == "disconnected" ? nil : .init(
            serverIdentity: change == "server" ? "ssh://other.invalid:8080" : origin.serverIdentity,
            remoteRoot: change == "remote-root" ? "/different" : origin.remoteRoot,
            workspaceRoot: change == "local-workspace" ? workspace.appending(component: "other") : workspace)
        let counter = Counter()
        let service = EvidenceAutoImportService(workspaceRoot: workspace,
            originProvider: { current }, performImport: { _ in
                await counter.increment()
                return workspace
            }, packageEvidence: { _ in
                await counter.increment()
                return .init(bundlePath: nil, bundleSha256: nil, runID: nil, evidenceComplete: nil, missingEvidence: nil)
            })
        let events = await service.importNow(candidates: [.init(bundlePath: "/remote/evidence.tar.gz")], origin: origin)
        let pipeline = await service.importPipeline(runID: "run", origin: origin)
        for event in events + [try #require(pipeline)] {
            #expect(event.outcome == .refused(code: "evidenceContextChanged", repairAction: EvidenceImportOrigin.changedRepair))
        }
        #expect(await counter.value == 0)
    }

    @Test func completedImportKeepsItsCapturedOriginAfterSelectionChanges() async throws {
        let workspace = try freshWorkspace("captured-receipt")
        defer { try? FileManager.default.removeItem(at: workspace) }
        let origin = source(workspace)
        let selected = SelectedImportOrigin(origin)
        let service = EvidenceAutoImportService(workspaceRoot: workspace,
            originProvider: { selected.value }, performImport: { _ in
                await selected.clear()
                return workspace
            })
        let events = await service.importNow(candidates: [
            .init(bundlePath: "/remote/evidence.tar.gz", sha256: "first")
        ], origin: origin)
        #expect(events.first?.origin == origin)
        #expect(service.ledgerEntries.first?.origin == origin)
        #expect(service.ledgerEntries.first?.contentsVerified == true)
    }

    @Test func independentImportersMergeReceiptsAndPreserveCorruptLedgers() async throws {
        let workspace = try freshWorkspace("ledger-merge")
        defer { try? FileManager.default.removeItem(at: workspace) }
        let origin = source(workspace)
        let first = EvidenceAutoImportService(workspaceRoot: workspace,
            originProvider: { origin }, performImport: { _ in workspace })
        let second = EvidenceAutoImportService(workspaceRoot: workspace,
            originProvider: { origin }, performImport: { _ in workspace })
        _ = await first.importNow(candidates: [.init(bundlePath: "/remote/first.tar.gz", sha256: "first")], origin: origin)
        _ = await second.importNow(candidates: [.init(bundlePath: "/remote/second.tar.gz", sha256: "second")], origin: origin)
        let url = EvidenceAutoImportService.ledgerURL(workspaceRoot: workspace)
        #expect(EvidenceAutoImportService.loadLedger(at: url).count == 2)
        let damaged = Data("incomplete ledger bytes".utf8)
        try damaged.write(to: url)
        let events = await second.importNow(candidates: [.init(bundlePath: "/remote/third.tar.gz", sha256: "third")], origin: origin)
        guard case .failed? = events.first?.outcome else {
            Issue.record("A corrupt ledger must not be replaced with partial history")
            return
        }
        #expect(try Data(contentsOf: url) == damaged)
    }

    @Test func failureBackoffDoesNotCrossServerBoundaries() async throws {
        let workspace = try freshWorkspace("scoped-backoff")
        defer { try? FileManager.default.removeItem(at: workspace) }
        let origin = source(workspace)
        let selected = SelectedImportOrigin(origin)
        let job = succeededRunJob(id: "job", bundlePath: "/remote/evidence.tar.gz")
        let counter = Counter()
        let service = EvidenceAutoImportService(workspaceRoot: workspace,
            originProvider: { selected.value }, fetchJobs: { [job] }, performImport: { _ in
                await counter.increment()
                throw ChatServiceError(reason: "transport failed")
            })
        _ = await service.runOnce(force: true)
        #expect(await service.runOnce(force: true).isEmpty)
        selected.value = .init(serverIdentity: "ssh://other.invalid:8080", remoteRoot: "/remote", workspaceRoot: workspace)
        _ = await service.runOnce(force: true)
        #expect(await counter.value == 2)
        #expect(service.failures.count == 2)
    }

    // MARK: Failure retention + capped backoff

    @Test func failuresAreRetainedBackedOffAndCapped() async throws {
        let workspace = try freshWorkspace("failures")
        let bundlePath = "/remote/runs/z/z.evidence-bundle.tar.gz"
        let job = succeededRunJob(id: "j1", bundlePath: bundlePath, sha: nil)

        let clock = MutableClock(start: Date(timeIntervalSince1970: 1_780_000_000))
        let importCounter = Counter()
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace, originProvider: { source(workspace) },
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
        let failure = try #require(service.failure(for: .init(bundlePath: bundlePath, sha256: nil), origin: source(workspace)))
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
        let exhausted = try #require(service.failure(for: .init(bundlePath: bundlePath, sha256: nil), origin: source(workspace)))
        #expect(exhausted.attempts == 2)
        #expect(exhausted.exhausted)

        clock.advance(by: 100_000)
        events = await service.runOnce(force: true)
        #expect(events.isEmpty)
        #expect(await importCounter.value == 2)
        #expect(!service.isImported(candidate: .init(bundlePath: bundlePath, sha256: nil), origin: source(workspace)))
    }

    @Test func successAfterFailureClearsTheFailureAndLedgers() async throws {
        let workspace = try freshWorkspace("recovery")
        let bundlePath = "/remote/runs/w/w.evidence-bundle.tar.gz"
        let job = succeededRunJob(id: "j1", bundlePath: bundlePath)
        let clock = MutableClock(start: Date(timeIntervalSince1970: 1_780_000_000))
        let gate = FailureGate(failuresBeforeSuccess: 1)
        let service = EvidenceAutoImportService(
            workspaceRoot: workspace, originProvider: { source(workspace) },
            fetchJobs: { [job] },
            performImport: { _ in
                if await gate.shouldFail() {
                    throw ChatServiceError(reason: "transient")
                }
                return workspace.appending(component: "runs-w")
            },
            now: { clock.now })

        _ = await service.runOnce(force: true)
        #expect(service.failure(for: .init(bundlePath: bundlePath, sha256: "abc123"), origin: source(workspace)) != nil)

        clock.advance(by: 7200)
        let events = await service.runOnce(force: true)
        #expect(events.count == 1)
        if case .imported(let directory)? = events.first?.outcome {
            #expect(directory.hasSuffix("runs-w"))
        } else {
            Issue.record("expected an imported outcome")
        }
        #expect(service.failure(for: .init(bundlePath: bundlePath, sha256: "abc123"), origin: source(workspace)) == nil)
        #expect(service.isImported(candidate: .init(bundlePath: bundlePath, sha256: "abc123"), origin: source(workspace)))
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

@MainActor
private final class SelectedImportOrigin {
    var value: EvidenceImportOrigin?
    init(_ value: EvidenceImportOrigin?) { self.value = value }
    func clear() { value = nil }
}
