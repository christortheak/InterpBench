import Foundation
import Testing

@testable import ExperimentKit

// 2026-07-21 cluster incident, part 1: a study bundle submitted with
// executor "local" on a Slurm site executed INSIDE the controller job's
// allocation (1 CPU, 16 GB, no GPU) and died mid-load — the GPU-SESSION
// warning said nothing. The submission-resource preflight warns about the
// submission's OWN missing GPU allocation, composes with the session
// warning into ONE dialog, and offers a fix that prefills the Remote
// options. Warning with an override, never a refusal.

@Suite struct ModelJobSubmissionPreflightTests {

    private func options(
        executor: String = "local", gres: String = "",
        verb: String = "run", dryRun: Bool = false
    ) -> ModelJobSubmissionPreflight.BundleOptions {
        ModelJobSubmissionPreflight.BundleOptions(
            executor: executor, gres: gres, verb: verb, dryRun: dryRun)
    }

    // MARK: The allocation predicate

    @Test func nonSlurmExecutorOrEmptyGresIsMissingAllocation() {
        #expect(ModelJobSubmissionPreflight.missingGPUAllocation(
            executor: "local", gres: "A100"))
        #expect(ModelJobSubmissionPreflight.missingGPUAllocation(
            executor: "slurm", gres: ""))
        #expect(ModelJobSubmissionPreflight.missingGPUAllocation(
            executor: "slurm", gres: "   "))
        #expect(ModelJobSubmissionPreflight.missingGPUAllocation(
            executor: "local", gres: ""))
        #expect(!ModelJobSubmissionPreflight.missingGPUAllocation(
            executor: "slurm", gres: "A100"))
        // Trimming + case: " Slurm " is the slurm executor.
        #expect(!ModelJobSubmissionPreflight.missingGPUAllocation(
            executor: " Slurm ", gres: " L4 "))
    }

    // MARK: The one-dialog rule (site × executor × gres × verb matrix)

    @Test func slurmSiteWithGPUlessOptionsWarnsAboutTheAllocation() {
        // The incident exactly: Slurm site, executor local, model-running
        // verb — the allocation concern fires regardless of session state.
        let concern = ModelJobSubmissionPreflight.concern(
            siteUsesSlurmScheduler: true,
            options: options(executor: "local", gres: "A100"),
            sessionWarning: false)
        #expect(concern == .noGPUAllocation(alsoNoSession: false))
    }

    @Test func emptyGresAloneAlsoWarns() {
        let concern = ModelJobSubmissionPreflight.concern(
            siteUsesSlurmScheduler: true,
            options: options(executor: "slurm", gres: ""),
            sessionWarning: false)
        #expect(concern == .noGPUAllocation(alsoNoSession: false))
    }

    @Test func bothConcernsComposeIntoOneDialog() {
        // Missing allocation is MORE specific than missing session: the
        // allocation concern wins the dialog and carries the session note.
        let concern = ModelJobSubmissionPreflight.concern(
            siteUsesSlurmScheduler: true,
            options: options(executor: "local", gres: ""),
            sessionWarning: true)
        #expect(concern == .noGPUAllocation(alsoNoSession: true))
    }

    @Test func properSlurmOptionsSubmitSilentlyRegardlessOfSession() {
        // Executor slurm + non-empty gres on a Slurm site: the batch job
        // OWNS its GPU allocation — the interactive session's absence is
        // irrelevant, so a correct bundle never warns (external review
        // 2026-07-22; previously sessionWarning still produced a spurious
        // "no GPU session" dialog).
        #expect(
            ModelJobSubmissionPreflight.concern(
                siteUsesSlurmScheduler: true,
                options: options(executor: "slurm", gres: "A100"),
                sessionWarning: true)
                == nil)
        #expect(
            ModelJobSubmissionPreflight.concern(
                siteUsesSlurmScheduler: true,
                options: options(executor: "slurm", gres: "A100"),
                sessionWarning: false)
                == nil)
    }

    @Test func nonSlurmSitesNeverWarnAboutAllocations() {
        // A workstation/local dev server: executor "local" is simply
        // correct — only the session rule (if any) applies.
        #expect(
            ModelJobSubmissionPreflight.concern(
                siteUsesSlurmScheduler: false,
                options: options(executor: "local", gres: ""),
                sessionWarning: false)
                == nil)
        #expect(
            ModelJobSubmissionPreflight.concern(
                siteUsesSlurmScheduler: false,
                options: options(executor: "local", gres: ""),
                sessionWarning: true)
                == .noGPUSession)
    }

    @Test func dryRunsAndNonModelVerbsSubmitSilently() {
        // Nothing executes on a dry run; analyze is statistics over an
        // existing run. Neither warns — even with both concerns present.
        #expect(
            ModelJobSubmissionPreflight.concern(
                siteUsesSlurmScheduler: true,
                options: options(executor: "local", gres: "", dryRun: true),
                sessionWarning: true)
                == nil)
        #expect(
            ModelJobSubmissionPreflight.concern(
                siteUsesSlurmScheduler: true,
                options: options(executor: "local", gres: "", verb: "analyze"),
                sessionWarning: true)
                == nil)
        // Every model-running bundle verb warns, unknown verbs included
        // (same safe default as GPUSessionPreflight.isModelRunningVerb).
        for verb in ["run", "sweep", "extract", "validate", "pipeline", "newVerb"] {
            #expect(
                ModelJobSubmissionPreflight.concern(
                    siteUsesSlurmScheduler: true,
                    options: options(executor: "local", gres: "", verb: verb),
                    sessionWarning: false)
                    == .noGPUAllocation(alsoNoSession: false), "\(verb)")
        }
    }

    @Test func nilOptionsKeepTheHistoricalSessionOnlyRule() {
        // Direct server verbs (validate/extract/LoRA) carry no Remote
        // options — exactly the old behavior, unchanged.
        #expect(
            ModelJobSubmissionPreflight.concern(
                siteUsesSlurmScheduler: true, options: nil, sessionWarning: true)
                == .noGPUSession)
        #expect(
            ModelJobSubmissionPreflight.concern(
                siteUsesSlurmScheduler: true, options: nil, sessionWarning: false)
                == nil)
    }

    // MARK: Copy

    @Test func allocationMessageNamesProblemConsequenceAndWaysForward() {
        let message = ModelJobSubmissionPreflight.message(
            for: .noGPUAllocation(alsoNoSession: false),
            substrate: "Example HPC",
            options: options(executor: "local", gres: ""))
        #expect(message.contains("without a GPU allocation"))
        #expect(message.contains("Example HPC"))
        #expect(message.contains("'local'"))
        #expect(message.contains("gres field is empty"))
        #expect(message.contains("controller's small CPU allocation"))
        #expect(message.contains("multi-billion-parameter model will fail"))
        // The legitimate override is named, not implied.
        #expect(message.contains("CPU smoke tests"))
    }

    @Test func composedMessageSaysBothWhenBothApply() {
        let both = ModelJobSubmissionPreflight.message(
            for: .noGPUAllocation(alsoNoSession: true),
            substrate: "Example HPC",
            options: options(executor: "slurm", gres: ""))
        #expect(both.contains("without a GPU allocation"))
        #expect(both.contains("no GPU session"))
        // gres-only reason: the executor sentence must not appear.
        #expect(!both.contains("execute inside the controller's own"))

        let single = ModelJobSubmissionPreflight.message(
            for: .noGPUAllocation(alsoNoSession: false),
            substrate: "Example HPC",
            options: options(executor: "slurm", gres: ""))
        #expect(!single.contains("no GPU session"))
    }

    @Test func sessionConcernReusesTheHistoricalCopy() {
        #expect(
            ModelJobSubmissionPreflight.message(
                for: .noGPUSession, substrate: "Example HPC", options: nil)
                == GPUSessionPreflight.warningMessage(substrate: "Example HPC"))
        #expect(
            ModelJobSubmissionPreflight.title(
                for: .noGPUSession, jobLabel: "study run")
                == "No GPU session — submit study run?")
        #expect(
            ModelJobSubmissionPreflight.title(
                for: .noGPUAllocation(alsoNoSession: false), jobLabel: "study run")
                == "No GPU allocation — submit study run?")
    }

    // MARK: The "Fix options" prefill

    @Test func fixedOptionsSnapExecutorAndPrefillGresFromSiteVocabulary() {
        let fixed = ModelJobSubmissionPreflight.fixedOptions(
            executor: "local", gres: "", siteGPUTypes: ["A100", "H100", "L4"])
        #expect(fixed.executor == "slurm")
        #expect(fixed.gres == "A100")
    }

    @Test func fixedOptionsKeepAResearcherChosenGres() {
        let fixed = ModelJobSubmissionPreflight.fixedOptions(
            executor: "local", gres: "L4", siteGPUTypes: ["A100"])
        #expect(fixed.executor == "slurm")
        #expect(fixed.gres == "L4")
    }

    @Test func fixedOptionsWithoutVocabularyLeaveGresForTheResearcher() {
        let fixed = ModelJobSubmissionPreflight.fixedOptions(
            executor: "local", gres: "", siteGPUTypes: [])
        #expect(fixed.executor == "slurm")
        #expect(fixed.gres.isEmpty)
    }
}
