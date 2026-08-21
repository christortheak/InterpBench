import Foundation
import Testing

@testable import ExperimentKit

// Cluster-testing session fixes (2026-07-21):
// - item 2: the "no GPU session" warning predicate before model-running
//   cluster submissions (warn with an override, never a refusal),
// - item 1's stop half: the stop-with-jobs-running confirmation rule,
// - item 3: FineTuningPanel's honest cancel-path selection (server durable
//   job vs local in-process task) and its cancel-failure wording.

// MARK: - Item 2: warning predicate

@Suite struct GPUSessionPreflightTests {

    @Test func warnsExactlyWhenCapableAndNoSessionAndNoStartInFlight() {
        #expect(
            GPUSessionPreflight.shouldWarn(
                capabilityAvailable: true, isSessionActive: false,
                isStartInFlight: false))
    }

    @Test func activeSessionNeverWarns() {
        #expect(
            !GPUSessionPreflight.shouldWarn(
                capabilityAvailable: true, isSessionActive: true,
                isStartInFlight: false))
    }

    @Test func startAlreadyInFlightNeverWarns() {
        #expect(
            !GPUSessionPreflight.shouldWarn(
                capabilityAvailable: true, isSessionActive: false,
                isStartInFlight: true))
    }

    @Test func capabilityLessServersAndLocalNeverWarn() {
        // No GPU-session capability (older server, workstation, or Local
        // workspace wiring) — there is nothing to start, so no warning.
        #expect(
            !GPUSessionPreflight.shouldWarn(
                capabilityAvailable: false, isSessionActive: false,
                isStartInFlight: false))
    }

    @Test func modelRunningVerbs() {
        for verb in ["run", "sweep", "extract", "validate", "evaluate"] {
            #expect(GPUSessionPreflight.isModelRunningVerb(verb), "\(verb)")
        }
        // analyze is statistics over an existing run — never warned.
        #expect(!GPUSessionPreflight.isModelRunningVerb("analyze"))
        #expect(!GPUSessionPreflight.isModelRunningVerb(" Analyze "))
        // Unknown/future verbs take the safe default (warn).
        #expect(GPUSessionPreflight.isModelRunningVerb("someNewVerb"))
    }

    @Test func warningMessageNamesTheSubstrateAndBothWaysForward() {
        let message = GPUSessionPreflight.warningMessage(substrate: "Example HPC")
        #expect(message.contains("Example HPC"))
        #expect(message.contains("No GPU session"))
        #expect(message.contains("wait or fail"))
        #expect(message.contains("submit anyway"))
    }

    /// Controller-level predicate: same rule against live state.
    @MainActor
    @Test func controllerPredicateFollowsSessionState() {
        let controller = GPUSessionController()
        controller.capabilityProvider = { true }
        #expect(controller.shouldWarnBeforeModelRunningJob)

        controller.ingest(GPUSessionRecord(sessionGeneration: "g", state: "ready"))
        #expect(!controller.shouldWarnBeforeModelRunningJob)

        // Session over (terminal) — a new submission would wait again.
        controller.ingest(GPUSessionRecord(sessionGeneration: "g", state: "ended"))
        #expect(controller.shouldWarnBeforeModelRunningJob)

        controller.capabilityProvider = { false }
        #expect(!controller.shouldWarnBeforeModelRunningJob)
        controller.resetForConnectionChange()
    }
}

// MARK: - Item 1 (stop half): stop-with-jobs-running rule

@Suite struct GPUSessionStopCheckTests {

    private func job(id: String, finished: Bool) -> RemoteJobRecord {
        RemoteJobRecord(
            id: id, kind: "finetune:train", status: finished ? "succeeded" : "running",
            createdAt: 1, startedAt: 1, finishedAt: finished ? 2 : nil,
            result: nil, error: nil, logTail: [], executor: "local",
            executorJobID: nil, cancellationRequested: false)
    }

    @Test func countsOnlyUnfinishedJobs() {
        let jobs = [
            job(id: "a", finished: true),
            job(id: "b", finished: false),
            job(id: "c", finished: false),
        ]
        #expect(GPUSessionStopCheck.unfinishedJobCount(jobs) == 2)
        #expect(GPUSessionStopCheck.unfinishedJobCount([]) == 0)
    }

    @Test func noUnfinishedJobsMeansNoConfirmation() {
        #expect(GPUSessionStopCheck.confirmationMessage(unfinishedJobCount: 0) == nil)
    }

    @Test func unfinishedJobsConfirmWithTheCount() {
        let one = GPUSessionStopCheck.confirmationMessage(unfinishedJobCount: 1)
        #expect(one?.contains("1 server job has") == true)
        let three = GPUSessionStopCheck.confirmationMessage(unfinishedJobCount: 3)
        #expect(three?.contains("3 server jobs have") == true)
    }

    @Test func uncheckableJobListConfirmsHonestly() {
        // The job list could not be fetched: never pretend we checked.
        let message = GPUSessionStopCheck.confirmationMessage(unfinishedJobCount: nil)
        #expect(message?.contains("Could not check") == true)
    }
}

// MARK: - Item 3: fine-tune cancel-path selection + honest wording

@Suite struct FineTuneCancelPathTests {

    @Test func notTrainingMeansNothingToCancel() {
        #expect(
            FineTuningPanel.trainingCancelPath(isTraining: false, serverJobID: "job-9")
                == .none)
        #expect(
            FineTuningPanel.trainingCancelPath(isTraining: false, serverJobID: nil)
                == .none)
    }

    @Test func serverJobIDSelectsTheServerCancelPath() {
        // The confirmed bug: server training must cancel the CLUSTER job,
        // not just the local log stream.
        #expect(
            FineTuningPanel.trainingCancelPath(isTraining: true, serverJobID: "job-42")
                == .serverJob(id: "job-42"))
    }

    @Test func localTrainingKeepsImmediateLocalCancel() {
        #expect(
            FineTuningPanel.trainingCancelPath(isTraining: true, serverJobID: nil)
                == .localTask)
        // An empty id is no id.
        #expect(
            FineTuningPanel.trainingCancelPath(isTraining: true, serverJobID: "")
                == .localTask)
    }

    @Test func serverCancelWordingNeverClaimsCancelled() {
        let requested = FineTuningPanel.serverCancelRequestedMessage(jobID: "job-42")
        #expect(requested.contains("cancel requested for job job-42"))
        #expect(requested.contains("the cluster decides"))
        // "cancel requested" is the strongest claim; never a past-tense
        // "cancelled" for a job that may still be running.
        #expect(!requested.lowercased().contains("cancelled"))

        let failed = FineTuningPanel.serverCancelFailureMessage(
            jobID: "job-42", detail: "scancel failed on the login node")
        #expect(failed.contains("cancel request failed"))
        #expect(failed.contains("scancel failed on the login node"))
        #expect(failed.contains("job job-42 is still running"))
        #expect(!failed.lowercased().contains("cancelled"))
    }
}
