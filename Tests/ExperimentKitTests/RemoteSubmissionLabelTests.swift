import Testing

@testable import ExperimentKit

/// Bundle-submission UX helpers: the submit button must say what it will do,
/// the post-submission status must say what was submitted and where to watch
/// it, and the not-on-server callout must name both ways forward. Pinned here
/// because the live-testing failure was exactly these defaults/labels
/// whispering (verify + dry-run defaults looked like "nothing happened").
@MainActor
struct RemoteSubmissionLabelTests {

    @Test func submitButtonLabelSaysWhatItWillDo() {
        #expect(
            ExperimentPanel.bundleSubmitLabel(verb: "run", dryRun: false)
                == "Submit Bundle: run")
        #expect(
            ExperimentPanel.bundleSubmitLabel(verb: "verify", dryRun: true)
                == "Submit Bundle: verify (dry run)")
        #expect(
            ExperimentPanel.bundleSubmitLabel(verb: "sweep", dryRun: false)
                == "Submit Bundle: sweep")
    }

    @Test func submittedStatusNamesStudySubstrateJobAndPane() {
        let status = ExperimentPanel.bundleSubmittedStatus(
            study: "alien-stance", verb: "run", dryRun: false,
            substrate: "cluster-a", jobID: "abc123")
        #expect(status.contains("bundled study 'alien-stance' submitted: run on cluster-a"))
        #expect(status.contains("job abc123"))
        #expect(status.contains("activity pane"))
    }

    @Test func dryRunStatusSaysNothingExecutes() {
        let status = ExperimentPanel.bundleSubmittedStatus(
            study: "s", verb: "verify", dryRun: true, substrate: "srv", jobID: "j1")
        #expect(status.contains("verify (dry run — prepared only, nothing executes)"))
    }

    @Test func residencyCalloutNamesBothPathsForward() {
        let message = ExperimentPanel.residencyCalloutMessage(
            study: "s1", substrate: "gpu-box")
        #expect(message.contains("'s1' exists locally, not on gpu-box"))
        #expect(message.contains("Submit Bundle"))
        #expect(message.contains("serve --root"))
    }

    @Test func bundleDefaultsPreselectARealRun() {
        // verify + dryRun defaults betrayed the "run my study" intent: the
        // submission prepared a dry run and appeared to do nothing.
        let panel = ExperimentPanel()
        #expect(panel.remoteVerb == "run")
        #expect(panel.remoteDryRun == false)
        #expect(panel.submitBundleButtonLabel == "Submit Bundle: run")
    }

    @Test func resumePolicyDefaultsOnInThePanel() {
        // 2026-07-22 incident: a checkpointed run continuing is what the
        // researcher asked for by submitting it — the panel's toggle ships
        // ON with the server's default restart cap prefilled (editable).
        let panel = ExperimentPanel()
        #expect(panel.remoteResumePolicy.autoResubmit == true)
        #expect(panel.remoteResumePolicy.limit == RemoteResumePolicy.serverDefaultLimit)
    }
}
