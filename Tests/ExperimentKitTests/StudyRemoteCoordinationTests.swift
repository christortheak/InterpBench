import Foundation
import Testing

@testable import ExperimentKit

@MainActor
struct StudyRemoteCoordinationTests {
    private func withWorkspace(_ body: (ExperimentManifest) async throws -> Void) async throws {
        ExperimentRootOverrideLock.acquire()
        let root = FileManager.default.temporaryDirectory.appending(
            component: "remote-owner-\(UUID())")
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = root
        defer {
            WorkspaceRoot.programmaticOverride = previous
            try? FileManager.default.removeItem(at: root)
            ExperimentRootOverrideLock.release()
        }
        let manifest = try ExperimentStore.create(
            name: "study", description: "", modelID: "test/model")
        try await body(manifest)
    }

    private func response(_ id: String = "job") -> RemoteStudySubmission {
        .init(
            jobId: id, experiment: "study", verb: "run", executor: "slurm", dryRun: false,
            runBundle: [:], slurmBundle: nil, slurmJobID: nil, command: [],
            recordsDirectory: "records", submissionDirectory: "submission")
    }

    private func bundleTransport() -> StudyBundleTransport {
        StudyBundleTransport(
            frozenConflict: { _ in nil },
            package: { _ in URL(filePath: "/private/tmp/fake-study.tar.gz") },
            upload: { _ in "uploaded/study.tar.gz" }, submit: { _, _ in response() })
    }

    private func terminal(_ id: String) -> RemoteJobRecord {
        .init(
            id: id, kind: "experiment:run", status: "succeeded", createdAt: 1,
            startedAt: nil, finishedAt: 2, result: nil, error: nil, logTail: [],
            executor: "slurm", executorJobID: nil, cancellationRequested: false)
    }

    @Test func bundleOptionsStayCapturedWhilePreflightWaits() async throws {
        try await withWorkspace { manifest in
            let options = StudySubmissionOptions()
            options.remoteExecutor = "slurm"
            options.remoteParallelJobs = 3
            let request = options.snapshot
            let jobs = StudyRemoteJobController()
            let owner = StudyBundleSubmissionController(jobs: jobs)
            var io = bundleTransport()
            io.frozenConflict = { _ in
                // Model an edit arriving at the first suspension point.
                await Task.yield()
                options.remoteExecutor = "local"
                options.remoteDryRun = true
                options.remoteGres = "changed"
                options.remoteWalltime = "00:01:00"
                options.remoteParallelJobs = 1
                options.remoteVerb = "verify"
                return nil
            }
            var captured: StudySubmissionRequest?
            io.submit = { path, request in
                #expect(path == "uploaded/study.tar.gz")
                captured = request
                var submitted = response()
                submitted.shardJobIDs = ["child-a", "child-b"]
                return submitted
            }
            let result = await owner.submit(
                manifest, request: request, capabilities: nil,
                substrate: "original server", transport: io)
            #expect(captured == request)
            #expect(captured?.effectiveResumePolicy != nil)
            #expect(captured?.resources["walltime"] == "04:00:00")
            #expect(jobs.remoteStatus?.contains("original server") == true)
            #expect(jobs.remoteStatus?.contains("dry run") == false)
            #expect(jobs.recentServerJobs.first?.verb == "run (bundle)")
            guard case .success("job") = result else {
                Issue.record("expected a recorded submission")
                return
            }
        }
    }

    @Test func frozenServerRefusalStopsBeforePackaging() async throws {
        try await withWorkspace { manifest in
            let jobs = StudyRemoteJobController()
            var io = bundleTransport()
            io.frozenConflict = { _ in "server copy is frozen — duplicate it" }
            io.package = { _ in
                Issue.record("must refuse before packaging")
                throw CancellationError()
            }
            let result = await StudyBundleSubmissionController(jobs: jobs).submit(
                manifest,
                request: StudySubmissionOptions().snapshot, capabilities: nil,
                substrate: nil, transport: io)
            #expect(jobs.remoteStatus == "server copy is frozen — duplicate it")
            #expect(jobs.recentServerJobs.isEmpty)
            guard case .failure = result else {
                Issue.record("must refuse")
                return
            }
        }
    }

    @Test func changedContextAfterUploadCannotSubmit() async throws {
        try await withWorkspace { manifest in
            let jobs = StudyRemoteJobController()
            var current = true
            var io = bundleTransport()
            io.upload = { _ in
                current = false
                return "uploaded"
            }
            io.submit = { _, _ in
                Issue.record("must not submit into changed context")
                throw CancellationError()
            }
            let result = await StudyBundleSubmissionController(jobs: jobs).submit(
                manifest,
                request: StudySubmissionOptions().snapshot, capabilities: nil,
                substrate: nil, transport: io, isCurrent: { current })
            guard case .failure = result else {
                Issue.record("must stop")
                return
            }
            #expect(jobs.recentServerJobs.isEmpty)
        }
    }

    @Test func batchDoesNotSeizeFollowerAndFailuresDoNotStampJobs() async throws {
        try await withWorkspace { manifest in
            let jobs = StudyRemoteJobController()
            let owner = StudyBundleSubmissionController(jobs: jobs)
            var followed = 0
            let request = StudySubmissionOptions().snapshot
            _ = await owner.submit(
                manifest, request: request, capabilities: nil,
                substrate: nil, transport: bundleTransport(),
                follow: { _, _, _, _ in followed += 1 })
            _ = await owner.submit(
                manifest, request: request, capabilities: nil,
                substrate: nil, transport: bundleTransport())
            #expect(followed == 1)
            #expect(jobs.recentServerJobs.count == 1)
            var failing = bundleTransport()
            failing.upload = { _ in throw CocoaError(.fileReadUnknown) }
            let result = await owner.submit(
                manifest, request: request, capabilities: nil,
                substrate: nil, transport: failing)
            guard case .failure = result else {
                Issue.record("must fail upload")
                return
            }
            #expect(jobs.recentServerJobs.count == 1)
        }
    }

    @Test func acceptedJobRemainsReconnectableAfterContextChanges() async throws {
        try await withWorkspace { manifest in
            let jobs = StudyRemoteJobController()
            var current = true
            var io = bundleTransport()
            io.submit = { _, _ in
                current = false
                return response("accepted")
            }
            _ = await StudyBundleSubmissionController(jobs: jobs).submit(
                manifest,
                request: StudySubmissionOptions().snapshot, capabilities: nil,
                substrate: nil, transport: io, isCurrent: { current })
            #expect(jobs.remoteJobID == "accepted")
            #expect(jobs.recentServerJobs.first?.id == "accepted")
        }
    }

    @Test func resumeAndResourcesRemainExecutionOnly() {
        let options = StudySubmissionOptions()
        options.remoteGres = " "
        options.remoteWalltime = "\n"
        let request = options.snapshot(verb: "pipeline")
        #expect(request.verb == "pipeline")
        #expect(options.remoteVerb == "run")
        #expect(request.resources.isEmpty)
        #expect(request.effectiveResumePolicy == nil)
    }

    @Test func directMissingStudyRefusesButUnknownResidencyStillSubmits() async {
        let jobs = StudyRemoteJobController()
        let owner = StudyServerJobCoordinator(jobs: jobs)
        var submitted = 0
        var io = StudyServerJobTransport(
            experimentNames: { [] },
            submit: { _, _ in
                submitted += 1
                return "direct"
            }, follow: { _, _, _, _ in nil })
        await owner.run(experimentName: "study", verb: "sweep", substrate: "server", transport: io)
        #expect(submitted == 0)
        io.experimentNames = { throw CocoaError(.fileReadUnknown) }
        await owner.run(experimentName: "study", verb: "sweep", substrate: "server", transport: io)
        #expect(submitted == 1)
        // A timed-out follower must retain both cancellation slots.
        #expect(jobs.activeServerJob?.id == "direct")
        #expect(jobs.activeSweepJob?.id == "direct")
    }

    @Test func oldTerminalJobCannotClearNewerCancellationSlots() async {
        let jobs = StudyRemoteJobController()
        let owner = StudyServerJobCoordinator(jobs: jobs)
        let io = StudyServerJobTransport(
            experimentNames: { ["study"] },
            submit: { _, _ in "old" },
            follow: { _, _, _, _ in
                jobs.activeServerJob = .init(id: "new-run", verb: "run", study: "another")
                jobs.activeSweepJob = .init(id: "new-sweep", verb: "sweep", study: "another")
                return terminal("old")
            })
        await owner.run(experimentName: "study", verb: "sweep", substrate: "server", transport: io)
        #expect(jobs.activeServerJob?.id == "new-run")
        #expect(jobs.activeSweepJob?.id == "new-sweep")
        #expect(jobs.recentServerJobs.first?.state == "succeeded")
    }

    @Test func directContextChangeStopsBeforeSubmission() async {
        let jobs = StudyRemoteJobController()
        var current = true
        let io = StudyServerJobTransport(
            experimentNames: {
                current = false
                return ["study"]
            },
            submit: { _, _ in
                Issue.record("must stop before submitting")
                return "bad"
            },
            follow: { _, _, _, _ in
                Issue.record("must not follow")
                return nil
            })
        await StudyServerJobCoordinator(jobs: jobs).run(
            experimentName: "study", verb: "run",
            substrate: "server", transport: io, isCurrent: { current })
        #expect(jobs.recentServerJobs.isEmpty)
    }

    @Test func supersededPipelineResponseCannotOverwriteNewerListing() async {
        let owner = StudyPipelineController()
        await owner.refresh(
            name: "old", local: { _ in [] },
            remote: { _ in
                await owner.refresh(
                    name: "new", local: { _ in [.init(run: "new-local", stages: [])] },
                    remote: { _ in [.init(run: "new-server", stages: [])] })
                return [.init(run: "old-server", stages: [])]
            })
        #expect(owner.pipelineRuns.map(\.run) == ["new-server"])
        #expect(owner.localPipelineRuns.map(\.run) == ["new-local"])
        owner.resetSelection()
        #expect(owner.pipelineRuns.isEmpty && owner.localPipelineRuns.isEmpty)
    }

    @Test func pipelineContextChangeAndOldServerFailureDoNotMixEvidence() async {
        let owner = StudyPipelineController()
        var current = true
        await owner.refresh(
            name: "study", local: { _ in [.init(run: "local", stages: [])] },
            remote: { _ in
                current = false
                return [.init(run: "stale", stages: [])]
            }, isCurrent: { current })
        #expect(owner.pipelineRuns.isEmpty)
        #expect(owner.localPipelineRuns.map(\.run) == ["local"])
        await owner.refresh(
            name: "study", local: { _ in [.init(run: "local", stages: [])] },
            remote: { _ in throw CocoaError(.fileReadUnknown) })
        #expect(owner.pipelineRuns.isEmpty)
        #expect(owner.localPipelineRuns.count == 1)
    }

    @Test func deferredPipelineActionRefusesChangedManifest() async throws {
        try await withWorkspace { manifest in
            let panel = ExperimentPanel()
            let submit = panel.pipelineSubmissionAction(
                manifest: manifest,
                request: panel.submission.snapshot)
            var changed = manifest
            changed.maxTokens += 1
            try ExperimentStore.save(changed)
            await submit()
            #expect(panel.status?.contains("study changed") == true)
            #expect(panel.remoteJobs.recentServerJobs.isEmpty)
        }
    }

    @Test func pipelineDeclarationRemainsDraftOnly() async throws {
        try await withWorkspace { manifest in
            let owner = StudyPipelineController()
            var refreshes = 0
            owner.presentation.refresh = { refreshes += 1 }
            let reviewed = try DraftAuthoringSnapshot(workspaceRoot: ExperimentStore.workspaceRoot, name: manifest.name)
            owner.saveDeclaration(nil, reviewed: reviewed)
            #expect(refreshes == 1)
            var frozen = manifest
            frozen.status = .frozen
            let data = try JSONEncoder().encode(frozen)
            try data.write(to: ExperimentRepository(workspaceRoot: reviewed.workspaceRoot).manifestURL(manifest.name))
            let frozenReview = try DraftAuthoringSnapshot(workspaceRoot: reviewed.workspaceRoot, name: manifest.name,
                file: ManifestFileSnapshot(data: data))
            owner.saveDeclaration(nil, reviewed: frozenReview)
            #expect(refreshes == 1)
        }
    }
}
