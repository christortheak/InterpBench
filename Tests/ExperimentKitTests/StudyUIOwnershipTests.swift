import Foundation
import Observation
import Synchronization
import Testing

@testable import ExperimentKit

@MainActor
struct StudyUIOwnershipTests {
    private func manifest(_ name: String) -> ExperimentManifest {
        ExperimentManifest(name: name, description: "saved description", modelID: "test/model")
    }

    @Test func refreshPreservesUnsavedDraftUntilExplicitSelectionSync() {
        let draft = StudyDraftState()
        let original = manifest("one")
        #expect(draft.synchronize(original, defaults: .init()))
        draft.taskDescription = "unsaved edit"
        #expect(!draft.synchronize(original, defaults: .init(baseModelID: "other")))
        #expect(draft.taskDescription == "unsaved edit")
        #expect(draft.synchronize(original, defaults: .init(), force: true))
        #expect(draft.taskDescription == "")
        #expect(draft.studyBaseModelID == "test/model")
        #expect(
            draft.synchronize(
                nil, defaults: .init(baseModelID: "server/model", judgeModel: "judge/model")))
        #expect(draft.studyBaseModelID == "server/model")
        #expect(draft.judgeModel == "judge/model")
        #expect(draft.runTemperature == 0)
    }

    @Test func selectionResetDoesNotCancelOrEraseAnActiveJob() {
        let draft = StudyDraftState()
        let results = StudyResultsState()
        let jobs = StudyLocalJobController()
        jobs.isRunning = true
        jobs.handleStudyProgress(
            .generationStarted(condition: "steered", promptID: "p", prompt: "question"))
        jobs.handleStudyProgress(
            .generationChunk(condition: "steered", promptID: "p", output: "partial"))
        _ = draft.synchronize(manifest("another"), defaults: .init())
        results.clearSelectionAndRuns()
        #expect(jobs.isRunning)
        #expect(jobs.liveActiveGeneration?.output == "partial")
        #expect(!jobs.studyRunCancelRequested)
        jobs.cancelStudyRun()
        #expect(jobs.studyRunCancelRequested)
        #expect(!jobs.validationCancelRequested)
        #expect(!jobs.sweepCancelRequested)
    }

    @Test func compatibilityBindingsObserveOwnerChangesAndViceVersa() {
        ExperimentRootOverrideLock.withTempRoot(prefix: "ui-observation") { _ in
            let panel = ExperimentPanel()
            let changed = Mutex(false)
            withObservationTracking {
                _ = panel.draft.taskDescription
            } onChange: {
                changed.withLock { $0 = true }
            }
            panel.draft.taskDescription = "through owner"
            #expect(changed.withLock { $0 })
            #expect(panel.draft.taskDescription == "through owner")
            panel.draft.taskDescription = "through compatibility binding"
            #expect(panel.draft.taskDescription == "through compatibility binding")
            panel.draft.formErrors[.addCondition] = "old refusal"
            panel.draft.conditionMode = .ablate
            #expect(panel.draft.conditionAlphaText == "1")
            #expect(panel.draft.formErrors[.addCondition] == nil)
        }
    }

    @Test func submissionOptionsAreCapturedIndependentlyOfLaterEdits() {
        let options = StudySubmissionOptions()
        options.remoteExecutor = "slurm"
        options.remoteVerb = "validate"
        options.remoteResumePolicy = RemoteResumePolicy(autoResubmit: true, limit: 3)
        options.remoteParallelJobs = 2
        let request = options.snapshot
        options.remoteExecutor = "local"
        options.remoteVerb = "sweep"
        options.remoteDryRun = true
        options.remoteResumePolicy.limit = 9
        options.remoteParallelJobs = 8
        #expect(request.executor == "slurm")
        #expect(request.verb == "validate")
        #expect(!request.dryRun)
        #expect(request.resumePolicy.limit == 3)
        #expect(request.parallelJobs == 2)
    }

    @Test func lateRemoteDetailFailureCannotOverwriteANewerSuccessfulSelection() async {
        let results = StudyResultsState()
        let ready = AsyncStream<Void>.makeStream()
        let release = AsyncStream<Void>.makeStream()
        let run = RemoteStampedRunRecord(
            id: "old", path: "/server/old", files: ["report.json"],
            fileEntries: [RemoteRunFileEntry(name: "report.json", size: 2)])
        let old = Task {
            await results.loadRemoteRunDetail(run: run, client: nil) { _, _ in
                ready.continuation.yield(())
                for await _ in release.stream { break }
                throw CocoaError(.fileReadUnknown)
            }
        }
        for await _ in ready.stream { break }
        var newer = run
        newer.id = "new"
        _ = await results.loadRemoteRunDetail(run: newer, client: nil) { _, _ in
            RemoteRunFileHead(data: Data("{}".utf8))
        }
        release.continuation.yield(())
        _ = await old.value
        ready.continuation.finish()
        release.continuation.finish()
        #expect(results.remoteResultsStatus == nil)
    }

    @Test func stoppedLogStreamCannotAppendLateLinesOrReplaceNewJobStatus() async {
        let jobs = StudyRemoteJobController()
        let ready = AsyncStream<Void>.makeStream()
        let release = AsyncStream<Void>.makeStream()
        let old = Task {
            await jobs.streamRemoteJobLog(
                jobID: "old",
                stream: { _, receive in
                    await receive("first")
                    ready.continuation.yield(())
                    for await _ in release.stream { break }
                    await receive("late old line")
                    throw CocoaError(.fileReadUnknown)
                }, job: { _ in throw CancellationError() })
        }
        for await _ in ready.stream { break }
        jobs.stopRemoteLogStream()
        jobs.remoteLogLines = []
        jobs.remoteStatus = "new job selected"
        release.continuation.yield(())
        await old.value
        ready.continuation.finish()
        release.continuation.finish()
        #expect(jobs.remoteLogLines.isEmpty)
        #expect(jobs.remoteStatus == "new job selected")
    }

    @Test func leavingServerResultsClearsAnOlderLoadingIndicator() async {
        let results = StudyResultsState()
        results.isLoadingRemoteResults = true
        await results.refreshRemoteResultsRuns(cluster: nil)
        #expect(!results.isLoadingRemoteResults)
        #expect(results.remoteResultsRuns.isEmpty)
        #expect(results.remoteResultsStatus == nil)
    }

    @Test func resultsWithIdenticalRunIDsRemainBoundToTheirWorkspace() throws {
        let root = FileManager.default.temporaryDirectory.appending(
            component: "ui-results-\(UUID())")
        defer { try? FileManager.default.removeItem(at: root) }
        let study = manifest("one")
        let results = StudyResultsState()
        for workspace in ["a", "b"] {
            let workspaceRoot = root.appending(component: workspace)
            let run = workspaceRoot.appending(path: "runs/20260905-exp-one-run")
            try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
            try JSONEncoder().encode(study).write(to: run.appending(component: "experiment.json"))
            try Data(workspace.utf8).write(to: run.appending(component: "workspace-id.txt"))
            results.refresh(
                experimentName: "one",
                repository: StudyResultRepository(workspaceRoot: workspaceRoot))
            let selected = try #require(results.selectedResult)
            let marker = URL(filePath: selected.item.path).appending(component: "workspace-id.txt")
            #expect(try String(contentsOf: marker, encoding: .utf8) == workspace)
        }
        results.clearSelectionAndRuns()
        #expect(results.selectedResultID == nil)
        #expect(results.selectedResultBrowserItem == nil)
    }
}
