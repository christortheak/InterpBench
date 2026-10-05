import Foundation
import Testing

@testable import ExperimentKit

/// A person can resume a run they cancelled (2026-10-04). The server decides
/// whether a cancelled job can continue and says so on the job record
/// (`cancelResume`); these tests pin what the app's Resume control and the
/// command line do with that, and the sentences a researcher reads.
@Suite struct ResumeAfterCancelTests {
    private static func record(_ json: String) throws -> RemoteJobRecord {
        try JSONDecoder().decode(RemoteJobRecord.self, from: Data(json.utf8))
    }

    // MARK: - the Resume control's enablement

    @Test func aCancelledJobOffersResumeOnlyWhenTheServerSaysItCanContinue() {
        let offered = RemoteJobCancelResume(
            offered: true, explanation: "Resume continues this run.")
        #expect(
            RemoteJobStatusClass.resumeOffer(
                status: "cancelled", cancelResume: offered) == .afterCancel)
        // The status alone never offers it: an older server sends no hint,
        // and the row then shows no Resume and no explanation, as before.
        #expect(!RemoteJobStatusClass.offersResume(status: "cancelled"))
        #expect(
            RemoteJobStatusClass.resumeOffer(status: "cancelled")
                == .notOffered(note: nil))
    }

    @Test func aCancelledJobThatCannotContinueExplainsWhyInPlainWords() {
        let local = RemoteJobCancelResume(
            offered: false,
            explanation: "This local run cannot be resumed yet; submit it again.")
        #expect(
            RemoteJobStatusClass.resumeOffer(
                status: "cancelled", cancelResume: local)
                == .notOffered(
                    note: "This local run cannot be resumed yet; submit it again."))
    }

    @Test func aCancelledJobThatWasAlreadyResumedRetiresTheControl() {
        let resumed = RemoteJobCancelResume(
            offered: false,
            explanation: "This cancelled run was resumed as job child9. "
                + "Follow that job; this record stays cancelled.",
            continuation: "child9")
        #expect(
            RemoteJobStatusClass.resumeOffer(
                status: "cancelled", resubmittedAs: "child9",
                cancelResume: resumed)
                == .notOffered(note: resumed.explanation))
        // Belt and braces: a stale "offered" beside a continuation still
        // offers nothing — one resume per cancelled job.
        #expect(
            RemoteJobStatusClass.resumeOffer(
                status: "cancelled", resubmittedAs: "child9",
                cancelResume: RemoteJobCancelResume(offered: true))
                == .notOffered(note: nil))
    }

    @Test func theCheckpointResumeRuleIsUnchanged() {
        #expect(
            RemoteJobStatusClass.resumeOffer(status: "checkpointed")
                == .checkpoint)
        #expect(
            RemoteJobStatusClass.resumeOffer(status: "cancelledResumable")
                == .checkpoint)
        #expect(
            RemoteJobStatusClass.resumeOffer(
                status: "checkpointed", resubmittedAs: "child77")
                == .notOffered(note: nil))
        for status in ["running", "submitted", "succeeded", "failed", "prepared"] {
            #expect(
                RemoteJobStatusClass.resumeOffer(status: status)
                    == .notOffered(note: nil))
        }
    }

    // MARK: - the wire

    @Test func theJobRecordCarriesTheServersCancelResumeHint() throws {
        let cancelled = try Self.record(
            """
            {"id": "abc", "kind": "study-submit", "status": "cancelled",
             "createdAt": 1, "logTail": [], "executor": "slurm",
             "cancellationRequested": true,
             "cancelResume": {"offered": true,
                              "explanation": "Resume continues this run."}}
            """)
        #expect(cancelled.cancelResume?.offered == true)
        #expect(
            RemoteJobStatusClass.resumeOffer(
                status: cancelled.status, resubmittedAs: cancelled.resubmittedAs,
                cancelResume: cancelled.cancelResume) == .afterCancel)
        // An older server's record decodes, and offers nothing.
        let older = try Self.record(
            """
            {"id": "abc", "kind": "study-submit", "status": "cancelled",
             "createdAt": 1, "logTail": [], "executor": "slurm",
             "cancellationRequested": true}
            """)
        #expect(older.cancelResume == nil)
    }

    @Test func theCancelAnswerDecodesWithOrWithoutTheNewFields() throws {
        let decoder = JSONDecoder()
        let bare = try decoder.decode(
            RemoteJobCancellation.self, from: Data(#"{"ok": true}"#.utf8))
        #expect(bare.ok == true)
        #expect(bare.cancelResume == nil)
        let full = try decoder.decode(
            RemoteJobCancellation.self,
            from: Data(
                """
                {"ok": true, "message": "Cancel requested.",
                 "cancelResume": {"offered": true, "explanation": "x"}}
                """.utf8))
        #expect(full.cancelResume?.offered == true)
        #expect(full.message == "Cancel requested.")
        // The client reads a 2xx body leniently: an accepted cancel is never
        // reported as a failure because its body had an unexpected shape.
        let fromObject = ClusterClient.cancellation(
            from: .object([
                "ok": .bool(true),
                "cancelResume": .object(["offered": .bool(true)]),
            ]))
        #expect(fromObject.cancelResume?.offered == true)
        #expect(ClusterClient.cancellation(from: .bool(true)).cancelResume == nil)
        #expect(
            ClusterClient.cancellation(
                from: .object(["cancelResume": .string("unexpected")])
            ).cancelResume == nil)
    }

    @Test func theResumeAnswerCarriesTheAfterCancelStamp() throws {
        let answer = try JSONDecoder().decode(
            RemoteJobResubmission.self,
            from: Data(
                """
                {"ok": true, "jobId": "child9", "resubmitOf": "abc",
                 "slurmJobID": "9002", "manualResubmit": true,
                 "resumedAfterCancel": true, "completedRecords": 3,
                 "message": "Job abc was cancelled; it now continues as job child9."}
                """.utf8))
        #expect(answer.resumedAfterCancel == true)
        #expect(answer.completedRecords == 3)
        // An ordinary checkpoint resume omits them all.
        let checkpoint = try JSONDecoder().decode(
            RemoteJobResubmission.self,
            from: Data(#"{"ok": true, "jobId": "child9"}"#.utf8))
        #expect(checkpoint.resumedAfterCancel == nil)
        #expect(checkpoint.message == nil)
    }

    // MARK: - the sentences

    @Test func theResumeHelpSaysWhatHappensAndWhatIsKept() {
        let help = RemoteJobStatusClass.resumeAfterCancelHelp
        #expect(help.contains("responses it already completed"))
        #expect(help.contains("fully stopped"))
        #expect(help.contains("Nothing is generated twice"))
        #expect(help.contains("stays cancelled"))
    }

    @Test func aResumeAfterACancelReportsTheServersOwnSentence() throws {
        var result = try JSONDecoder().decode(
            RemoteJobResubmission.self,
            from: Data(
                """
                {"ok": true, "jobId": "child9", "slurmJobID": "9002",
                 "resumedAfterCancel": true,
                 "message": "Job abc was cancelled; it now continues as job child9."}
                """.utf8))
        #expect(
            RemoteJobStatusClass.resumedStatusLine(jobID: "abc", result: result)
                == "Job abc was cancelled; it now continues as job child9.")
        // No sentence from the server: still a plain account, never the
        // checkpoint wording.
        result.message = nil
        let fallback = RemoteJobStatusClass.resumedStatusLine(
            jobID: "abc", result: result)
        #expect(fallback.contains("was cancelled"))
        #expect(fallback.contains("as job child9"))
        #expect(!fallback.contains("checkpoint"))
        // A checkpoint resume keeps the line it has always had.
        result.resumedAfterCancel = nil
        #expect(
            RemoteJobStatusClass.resumedStatusLine(jobID: "abc", result: result)
                == "job abc resumed as Slurm job 9002 — continuing from the checkpoint")
    }

    @Test func theCancelResultSaysResponsesAreKeptAndHowToResume() {
        let resumable = RemoteJobCancellation(
            ok: true, message: "Cancel requested.",
            cancelResume: RemoteJobCancelResume(offered: true))
        let inApp = RemoteJobStatusClass.cancelRequestedLine(
            jobID: "abc", cancellation: resumable, surface: .app)
        #expect(inApp.hasPrefix("cancel requested for abc"))
        #expect(inApp.contains("already completed are kept"))
        #expect(inApp.contains("press Resume on its row in Server Jobs"))
        #expect(inApp.contains("nothing resumes it automatically"))
        let onTheCommandLine = RemoteJobStatusClass.cancelRequestedLine(
            jobID: "abc", cancellation: resumable, surface: .commandLine)
        #expect(
            onTheCommandLine.contains("run `steerlab-cli remote resubmit abc`"))
        #expect(!onTheCommandLine.contains("Server Jobs"))
    }

    @Test func aCancelThatCannotBeResumedSaysSoAndAnOlderServerReadsAsBefore() {
        let local = RemoteJobCancellation(
            ok: true,
            cancelResume: RemoteJobCancelResume(
                offered: false,
                explanation: "This local run cannot be resumed yet; submit it again."))
        #expect(
            RemoteJobStatusClass.cancelRequestedLine(
                jobID: "abc", cancellation: local, surface: .app)
                == "cancel requested for abc — This local run cannot be "
                + "resumed yet; submit it again.")
        #expect(
            RemoteJobStatusClass.cancelRequestedLine(
                jobID: "abc", cancellation: RemoteJobCancellation(ok: true),
                surface: .app) == "cancel requested for abc")
        #expect(
            RemoteJobStatusClass.cancelRequestedLine(
                jobID: "abc", cancellation: nil, surface: .commandLine)
                == "cancel requested for abc")
    }

    @Test func theCancelConfirmationNoLongerSaysTheWorkIsLost() {
        let text = RemoteJobStatusClass.cancelConsequence(substrate: "the cluster")
        #expect(text.contains("on the cluster"))
        #expect(text.contains("already completed are kept"))
        #expect(text.contains("continue the run later with Resume"))
        #expect(text.contains("Nothing resumes a cancelled job automatically"))
    }

    // MARK: - the command line's refusal repair

    @Test func aRefusedResubmitGetsARepairThatMatchesItsReason() {
        let wait = ExperimentCLIRunner.resubmitRefusalRepair(
            detail: "job abc was cancelled, but the scheduler has not yet "
                + "confirmed that it stopped. Two jobs must never write to "
                + "the same run folder, so it cannot be resumed yet — wait a "
                + "minute, then try again")
        #expect(wait.hasPrefix("wait a minute for the cancelled job to stop"))
        #expect(wait.contains("steerlab-cli remote resubmit"))
        let nothingKept = ExperimentCLIRunner.resubmitRefusalRepair(
            detail: "job abc was cancelled before it had started generating "
                + "responses, so there is nothing to resume; submit the "
                + "study again")
        #expect(nothingKept.contains("steerlab-cli remote jobs --json"))
        #expect(nothingKept.contains("a cancelled study run that kept"))
        #expect(nothingKept.contains("submitting the study again"))
    }
}
