import Foundation
import Testing

@testable import ExperimentKit

/// Records every argv and replays canned answers by index — enough to prove a
/// refusal happened BEFORE any command ran.
private actor RecordingShell: ClusterShellRunner {
    private var queued: [ClusterShellResult]
    private var calls: [[String]] = []

    init(_ queued: [ClusterShellResult] = []) { self.queued = queued }

    func run(_ argv: [String]) async -> ClusterShellResult {
        calls.append(argv)
        guard !queued.isEmpty else { return ClusterShellResult(exitCode: 1) }
        return queued.removeFirst()
    }

    func recordedCalls() -> [[String]] { calls }
    func callCount() -> Int { calls.count }
}

private final class RecordingSecretStore: ClusterSecretStore, @unchecked Sendable {
    // @unchecked Sendable: mutated only from the serialized test body and the
    // single operation under test; never escapes the test.
    private var storage: [String: String] = [:]
    func token(forKey key: String) -> String? { storage[key] }
    func hasToken(forKey key: String) -> Bool { storage[key] != nil }
    func store(_ token: String, forKey key: String) throws { storage[key] = token }
    func removeToken(forKey key: String) { storage[key] = nil }
    var snapshot: [String: String] { storage }
}

/// An in-memory stand-in for the durable bootstrap journal, so the
/// submit → persist → poll ORDER is assertable without touching disk.
///
/// @unchecked Sendable: mutated only from the serialized test body and the
/// single operation under test; never escapes the test.
private final class FakeBootstrapJournal: ClusterBootstrapJobJournal, @unchecked Sendable {
    var pending: ClusterPendingBootstrapJob?
    var recorded: [ClusterPendingBootstrapJob] = []
    var finished: [(jobID: String, succeeded: Bool, message: String)] = []
    var saveError: (any Error)?

    func pendingBootstrapJob(forSite siteID: String) -> ClusterPendingBootstrapJob? {
        pending?.siteID == siteID ? pending : nil
    }

    func recordPendingBootstrapJob(_ job: ClusterPendingBootstrapJob) throws {
        if let saveError { throw saveError }
        recorded.append(job)
        pending = job
    }

    func finishPendingBootstrapJob(
        forSite siteID: String, jobID: String, succeeded: Bool, message: String
    ) {
        finished.append((jobID: jobID, succeeded: succeeded, message: message))
        if pending?.jobID == jobID { pending = nil }
    }
}

/// The extracted, headless provisioning operations (Phase B). These are the
/// same steps the wizard runs, now callable with no UI — so the contracts that
/// used to live inside an `@Observable @MainActor` step machine are testable
/// on their own.
struct ClusterProvisioningOperationsTests {

    private func site() -> ClusterSiteProfile {
        ClusterSiteProfile(
            name: "Test Cluster",
            transport: .ssh(
                host: "login.test", proxyJump: nil, remotePort: 8080, vpnExpected: false),
            topology: .daemonInJob,
            scheduler: .slurm(
                ClusterSiteProfile.SlurmSiteData(
                    partitions: [.init(name: "batch", maxWalltimeHours: 168)],
                    defaultPartition: "batch")),
            constraints: ClusterSiteProfile.SiteConstraints(
                storageRoots: ["workspace": "/scratch/me/ws", "hfCache": "/work/lab/hf"]))
    }

    /// The default configuration for this site — materializing, per WP5 Step 7.
    private func configuration() -> ClusterProvisioningConfiguration {
        var configuration = ClusterProvisioningConfiguration()
        configuration.localPayloadPath = "/tmp/payload"
        configuration.bootstrapExecutionTarget = .slurmBatch
        configuration.bootstrapJobPartition = "batch"
        return configuration
    }

    /// The same configuration with materialization OFF — WP5 Step 7's manual,
    /// no-profile path, where `bootstrap.sh` writes its own fallback env file.
    ///
    /// The submit → persist → poll cases below use it deliberately: they assert
    /// the SCHEDULER machinery against an exactly-queued shell, and the
    /// environment push is one more call in that queue. Materialization has its
    /// own cases (including one that proves the DEFAULT pushes), so pinning
    /// these to the legacy argv keeps each test about one thing.
    private func legacyConfiguration() -> ClusterProvisioningConfiguration {
        var configuration = configuration()
        configuration.materializeEnvironmentFile = false
        return configuration
    }

    // MARK: The dry-run-before-real gate (§7.9), now durable

    @Test func bootstrapApplyRefusesAMismatchedPlanHashBeforeRunningAnything() async throws {
        let shell = RecordingShell()
        let operations = ClusterProvisioningOperations(
            shell: shell, secrets: RecordingSecretStore())
        await #expect(throws: ClusterLifecycleError.self) {
            _ = try await operations.bootstrapApply(
                site: site(), configuration: configuration(),
                reviewedPlanHash: String(repeating: "f", count: 64))
        }
        // The refusal is BEFORE the command — nothing reached the cluster.
        #expect(await shell.callCount() == 0)

        do {
            _ = try await operations.bootstrapApply(
                site: site(), configuration: configuration(),
                reviewedPlanHash: String(repeating: "f", count: 64))
            Issue.record("a stale plan must refuse")
        } catch let error as ClusterLifecycleError {
            #expect(error.code == "bootstrapPlanMismatch")
            #expect(error.errorDescription?.contains("re-run the dry run") == true)
        }
    }

    @Test func bootstrapApplyRefusesWhenNoPlanWasEverReviewed() async throws {
        let shell = RecordingShell()
        let operations = ClusterProvisioningOperations(
            shell: shell, secrets: RecordingSecretStore())
        do {
            _ = try await operations.bootstrapApply(
                site: site(), configuration: configuration(), reviewedPlanHash: nil)
            Issue.record("an unreviewed plan must refuse")
        } catch let error as ClusterLifecycleError {
            #expect(error.code == "bootstrapPlanMissing")
        }
        #expect(await shell.callCount() == 0)
    }

    @Test func theReviewedPlanHashRunsAndAnySettingsChangeInvalidatesIt() async throws {
        let shell = RecordingShell([
            submitResult(jobID: "7788"),
            completedResult(jobID: "7788", exitCode: 0),
        ])
        let operations = pollingOperations(shell: shell)
        let hash = try #require(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: site(), configuration: legacyConfiguration()))
        let applied = try await operations.bootstrapApply(
            site: site(), configuration: legacyConfiguration(), reviewedPlanHash: hash)
        #expect(applied.outcome.succeeded)
        #expect(applied.jobID == "7788")
        // Submit and status are separate short commands — never one long one.
        #expect(await shell.callCount() == 2)

        // Every input the plan depends on moves the hash.
        var changedResources = legacyConfiguration()
        changedResources.bootstrapJobMemory = "24G"
        #expect(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: site(), configuration: changedResources) != hash)
        var changedRoot = legacyConfiguration()
        changedRoot.remoteRepoPath = "~/elsewhere"
        #expect(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: site(), configuration: changedRoot) != hash)
        var editedSite = site()
        editedSite.constraints.storageRoots["workspace"] = "/scratch/other"
        #expect(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: editedSite, configuration: legacyConfiguration()) != hash)
        // …and a profile edit the command does not mention still invalidates
        // it, because the site's own hash is part of the plan hash.
        var renotedSite = site()
        renotedSite.notes = "a note the bootstrap command never sees"
        #expect(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: renotedSite, configuration: legacyConfiguration()) != hash)
    }

    // MARK: Bootstrap submit → persist → poll (review finding 5)

    /// The helper's submit-mode output.
    private func submitResult(
        jobID: String, statusFile: String? = "/scratch/me/ws/b.status",
        adopted: Bool = false
    ) -> ClusterShellResult {
        var lines = ["bootstrap-job: submitted Slurm job \(jobID)"]
        if adopted { lines.append("STEERLAB_BOOTSTRAP_ADOPT=\(jobID)") }
        lines.append("STEERLAB_BOOTSTRAP_JOB_ID=\(jobID)")
        if let statusFile { lines.append("STEERLAB_BOOTSTRAP_STATUS_FILE=\(statusFile)") }
        return ClusterShellResult(exitCode: 0, lines: lines)
    }

    private func statusResult(_ verdict: String, jobID: String) -> ClusterShellResult {
        ClusterShellResult(
            exitCode: 0, lines: ["STEERLAB_BOOTSTRAP_STATUS=\(verdict) job=\(jobID)"])
    }

    /// A terminal status, with the job log the helper forwards behind it.
    private func completedResult(jobID: String, exitCode: Int) -> ClusterShellResult {
        let verdict = exitCode == 0 ? "completed-ok" : "completed-code-\(exitCode)"
        let step = exitCode == 0 ? "ok" : "failed"
        return ClusterShellResult(
            exitCode: 0,
            lines: [
                "STEERLAB_BOOTSTRAP_STATUS=\(verdict) job=\(jobID) exit=\(exitCode)",
                "installing…",
                #"{"ok":"# + (exitCode == 0 ? "true" : "false")
                    + #","steps":{"condaDetect":"\#(step)"},"envFile":"/e","prefix":"/p"}"#,
            ])
    }

    /// Operations that poll with no delay and a small budget, so a test can
    /// exercise the loop without waiting on a real cadence.
    private func pollingOperations(
        shell: RecordingShell, pollLimit: Int = 6
    ) -> ClusterProvisioningOperations {
        ClusterProvisioningOperations(
            shell: shell, secrets: RecordingSecretStore(),
            bootstrapPollDelay: .zero, bootstrapPollLimit: pollLimit)
    }

    @Test func aSubmittedBootstrapJobIsPersistedBeforeItIsEverPolled() async throws {
        // The defect this closes: the old apply held one SSH session open from
        // sbatch to completion, so a sleeping Mac between the two lost every
        // record of a job that had really been queued.
        let shell = RecordingShell([
            submitResult(jobID: "9001"),
            statusResult("pending", jobID: "9001"),
            statusResult("running", jobID: "9001"),
            completedResult(jobID: "9001", exitCode: 0),
        ])
        let journal = FakeBootstrapJournal()
        let operations = pollingOperations(shell: shell)
        let hash = try #require(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: site(), configuration: legacyConfiguration()))
        let applied = try await operations.bootstrapApply(
            site: site(), configuration: legacyConfiguration(), reviewedPlanHash: hash,
            journal: journal, siteID: "site-1")

        #expect(applied.outcome.succeeded)
        #expect(applied.jobID == "9001")
        #expect(applied.statusFile == "/scratch/me/ws/b.status")
        #expect(applied.verdict == .completed(exitCode: 0))
        #expect(applied.stillPending == false)
        // Persisted with the job id AND its status file, before any poll.
        let recorded = try #require(journal.recorded.first)
        #expect(recorded.jobID == "9001")
        #expect(recorded.statusFile == "/scratch/me/ws/b.status")
        #expect(recorded.planHash == hash)
        #expect(journal.finished.first?.jobID == "9001")
        #expect(journal.finished.first?.succeeded == true)

        // One submit, then N independent status probes — no held connection.
        let calls = await shell.recordedCalls()
        #expect(calls.count == 4)
        #expect(calls[0].joined(separator: " ").contains("submit-bootstrap-job.sh"))
        for probe in calls.dropFirst() {
            let text = probe.joined(separator: " ")
            #expect(text.contains("--status"))
            #expect(text.contains("--job-id 9001"))
            #expect(text.contains("--status-file /scratch/me/ws/b.status"))
        }
    }

    @Test func aPendingRecordIsResumedInsteadOfSubmittingASecondJob() async throws {
        // The retry-after-sleep path: the journal already names a job, so the
        // FIRST command this makes is a status probe, not an sbatch.
        let shell = RecordingShell([completedResult(jobID: "4242", exitCode: 0)])
        let journal = FakeBootstrapJournal()
        journal.pending = ClusterPendingBootstrapJob(
            siteID: "site-1", jobID: "4242", statusFile: "/scratch/me/ws/old.status")
        let operations = pollingOperations(shell: shell)
        let hash = try #require(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: site(), configuration: legacyConfiguration()))
        let applied = try await operations.bootstrapApply(
            site: site(), configuration: legacyConfiguration(), reviewedPlanHash: hash,
            journal: journal, siteID: "site-1")

        #expect(applied.outcome.succeeded)
        #expect(applied.jobID == "4242")
        #expect(applied.adopted)
        #expect(journal.recorded.isEmpty)   // nothing new was submitted
        let calls = await shell.recordedCalls()
        #expect(calls.count == 1)
        #expect(calls[0].joined(separator: " ").contains("--status"))
        #expect(!calls[0].joined(separator: " ").contains("--bootstrap-script"))
        #expect(applied.outcome.transcript.contains {
            $0.contains("resuming job 4242")
        })
    }

    @Test func theHelpersOwnAdoptMarkerIsCarriedIntoTheOutcome() async throws {
        // Even with no local record (the wizard's path), the cluster-side
        // breadcrumb refuses to queue a second bootstrap and says so.
        let shell = RecordingShell([
            submitResult(jobID: "5150", adopted: true),
            completedResult(jobID: "5150", exitCode: 0),
        ])
        let operations = pollingOperations(shell: shell)
        let hash = try #require(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: site(), configuration: legacyConfiguration()))
        let applied = try await operations.bootstrapApply(
            site: site(), configuration: legacyConfiguration(), reviewedPlanHash: hash)
        #expect(applied.adopted)
        #expect(applied.jobID == "5150")
        #expect(applied.outcome.transcript.contains { $0.contains("adopted, not resubmitted") })
    }

    @Test func anUnfinishedJobStaysPendingAndAdoptableRatherThanFailing() async throws {
        let shell = RecordingShell([
            submitResult(jobID: "777"),
            statusResult("pending", jobID: "777"),
            statusResult("pending", jobID: "777"),
        ])
        let journal = FakeBootstrapJournal()
        let operations = pollingOperations(shell: shell, pollLimit: 2)
        let hash = try #require(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: site(), configuration: legacyConfiguration()))
        let applied = try await operations.bootstrapApply(
            site: site(), configuration: legacyConfiguration(), reviewedPlanHash: hash,
            journal: journal, siteID: "site-1")

        #expect(applied.stillPending)
        #expect(applied.jobID == "777")
        // NOT reported as bootstrapped…
        #expect(!applied.outcome.succeeded)
        // …and NOT closed out, so the next invocation resumes this job.
        #expect(journal.finished.isEmpty)
        #expect(journal.recorded.count == 1)
        #expect(applied.outcome.message.contains("resumes following this job"))
    }

    @Test func aVanishedJobFailsAndAFailedProbeNeverDoes() async throws {
        // A job that left the queue with no status record is a real failure…
        let vanished = RecordingShell([
            submitResult(jobID: "31"),
            ClusterShellResult(
                exitCode: 0, lines: ["STEERLAB_BOOTSTRAP_STATUS=vanished job=31"]),
        ])
        let journal = FakeBootstrapJournal()
        let hash = try #require(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: site(), configuration: legacyConfiguration()))
        let gone = try await pollingOperations(shell: vanished).bootstrapApply(
            site: site(), configuration: legacyConfiguration(), reviewedPlanHash: hash,
            journal: journal, siteID: "site-1")
        #expect(!gone.outcome.succeeded)
        #expect(!gone.stillPending)
        #expect(gone.verdict == .vanished(state: nil))
        #expect(journal.finished.first?.succeeded == false)

        // …but an unreadable probe is not: an unproven death never licenses a
        // resubmit, so the job stays pending and adoptable.
        let unreadable = RecordingShell([
            submitResult(jobID: "32"),
            ClusterShellResult(exitCode: 255, lines: ["ssh: connection closed"]),
            ClusterShellResult(exitCode: 255, lines: ["ssh: connection closed"]),
        ])
        let secondJournal = FakeBootstrapJournal()
        let stuck = try await pollingOperations(shell: unreadable, pollLimit: 2)
            .bootstrapApply(
                site: site(), configuration: legacyConfiguration(), reviewedPlanHash: hash,
                journal: secondJournal, siteID: "site-1")
        #expect(stuck.stillPending)
        #expect(secondJournal.finished.isEmpty)
    }

    @Test func aFailedBootstrapJobNamesTheStepFromTheForwardedLog() async throws {
        let shell = RecordingShell([
            submitResult(jobID: "88"),
            completedResult(jobID: "88", exitCode: 3),
        ])
        let journal = FakeBootstrapJournal()
        let hash = try #require(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: site(), configuration: legacyConfiguration()))
        let applied = try await pollingOperations(shell: shell).bootstrapApply(
            site: site(), configuration: legacyConfiguration(), reviewedPlanHash: hash,
            journal: journal, siteID: "site-1")
        #expect(!applied.outcome.succeeded)
        #expect(applied.verdict == .completed(exitCode: 3))
        // The status probe forwards the job log, so the report still parses.
        #expect(applied.report?.ok == false)
        #expect(applied.outcome.message.contains("condaDetect"))
        #expect(journal.finished.first?.succeeded == false)
    }

    @Test func aSubmissionThatReportsNoJobIdIsAFailure() async throws {
        let shell = RecordingShell([
            ClusterShellResult(exitCode: 0, lines: ["bootstrap-job: nothing to say"])
        ])
        let journal = FakeBootstrapJournal()
        let hash = try #require(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: site(), configuration: legacyConfiguration()))
        let applied = try await pollingOperations(shell: shell).bootstrapApply(
            site: site(), configuration: legacyConfiguration(), reviewedPlanHash: hash,
            journal: journal, siteID: "site-1")
        #expect(!applied.outcome.succeeded)
        #expect(applied.jobID == nil)
        #expect(journal.recorded.isEmpty)
        #expect(await shell.callCount() == 1)   // never polled a job we do not have
    }

    // MARK: Parsing the helper's two machine-readable contracts

    @Test func theSubmitMarkersParseIntoASubmission() {
        let parsed = ClusterBootstrapSubmission.parse(lines: [
            "bootstrap-job: submitted Slurm job 4242",
            "STEERLAB_BOOTSTRAP_JOB_ID=4242",
            "STEERLAB_BOOTSTRAP_STATUS_FILE=/scratch/ws/b.status",
            "STEERLAB_BOOTSTRAP_LOG=/scratch/ws/b.log",
        ])
        #expect(parsed == ClusterBootstrapSubmission(
            jobID: "4242", statusFile: "/scratch/ws/b.status", adopted: false))

        let adopted = ClusterBootstrapSubmission.parse(lines: [
            "STEERLAB_BOOTSTRAP_ADOPT=99",
            "STEERLAB_BOOTSTRAP_JOB_ID=99",
        ])
        #expect(adopted?.adopted == true)
        #expect(adopted?.statusFile == nil)

        // No marker, no job: never invent one from prose.
        #expect(ClusterBootstrapSubmission.parse(lines: ["submitted job 4242"]) == nil)
        #expect(ClusterBootstrapSubmission.parse(
            lines: ["STEERLAB_BOOTSTRAP_JOB_ID="]) == nil)
    }

    @Test func everyStatusVerdictParsesAndOnlyTerminalOnesEndThePoll() {
        func parse(_ line: String) -> ClusterBootstrapJobVerdict? {
            ClusterBootstrapJobVerdict.parse(lines: ["noise", line, ])
        }
        #expect(parse("STEERLAB_BOOTSTRAP_STATUS=pending job=7 state=PENDING")
            == .pending(state: "PENDING"))
        #expect(parse("STEERLAB_BOOTSTRAP_STATUS=running job=7 state=RUNNING")
            == .running(state: "RUNNING"))
        #expect(parse("STEERLAB_BOOTSTRAP_STATUS=completed-ok job=7 exit=0")
            == .completed(exitCode: 0))
        #expect(parse("STEERLAB_BOOTSTRAP_STATUS=completed-code-12 job=7 exit=12")
            == .completed(exitCode: 12))
        #expect(parse("STEERLAB_BOOTSTRAP_STATUS=vanished job=7") == .vanished(state: nil))
        guard case .unknown = parse("STEERLAB_BOOTSTRAP_STATUS=unknown job=7") else {
            Issue.record("an unreadable state must parse as unknown")
            return
        }
        #expect(parse("bootstrap-job: still waiting") == nil)
        #expect(parse("STEERLAB_BOOTSTRAP_STATUS=completed-code-x job=7") == nil)

        // The verdict LINE outranks the forwarded log that follows it.
        #expect(ClusterBootstrapJobVerdict.parse(lines: [
            "STEERLAB_BOOTSTRAP_STATUS=completed-ok job=7 exit=0",
            #"{"ok":true,"steps":{}}"#,
        ]) == .completed(exitCode: 0))

        #expect(ClusterBootstrapJobVerdict.completed(exitCode: 0).isTerminal)
        #expect(ClusterBootstrapJobVerdict.vanished(state: nil).isTerminal)
        #expect(!ClusterBootstrapJobVerdict.pending(state: nil).isTerminal)
        #expect(!ClusterBootstrapJobVerdict.running(state: nil).isTerminal)
        // An unreadable status must NOT end the poll — it is not a verdict.
        #expect(!ClusterBootstrapJobVerdict.unknown(reason: "ssh died").isTerminal)
    }

    @Test func theStatusProbeIsShortLivedAndRefusesUnsafeInputs() throws {
        let argv = try #require(
            ClusterProvisioner.bootstrapJobStatusArgv(
                site: site(), remoteRepoPath: "~/steerlab", squeueCommand: "squeue",
                jobID: "4242", statusFile: "/scratch/me/ws/b.status"))
        let text = argv.joined(separator: " ")
        #expect(text.contains("submit-bootstrap-job.sh --status"))
        #expect(text.contains("--job-workspace /scratch/me/ws"))
        #expect(text.contains("--job-id 4242"))
        #expect(!text.contains("--wait"))

        // A job id is DATA off a disk record — it never becomes shell words.
        #expect(
            ClusterProvisioner.bootstrapJobStatusArgv(
                site: site(), remoteRepoPath: "~/steerlab", squeueCommand: "squeue",
                jobID: "4242; rm -rf /", statusFile: nil) == nil)
        #expect(
            ClusterProvisioner.bootstrapJobStatusArgv(
                site: site(), remoteRepoPath: "~/steerlab",
                squeueCommand: "squeue; touch /tmp/x", jobID: "1", statusFile: nil) == nil)
        // No workspace, no job to read a status from.
        var noWorkspace = site()
        noWorkspace.constraints.storageRoots["workspace"] = ""
        #expect(
            ClusterProvisioner.bootstrapJobStatusArgv(
                site: noWorkspace, remoteRepoPath: "~/steerlab", squeueCommand: "squeue",
                jobID: "1", statusFile: nil) == nil)
    }

    @Test func theSubmitCommandNeverWaitsAndTheDryRunNeverForcesANewJob() throws {
        let submit = try #require(
            ClusterProvisioningOperations.bootstrapArgv(
                site: site(), configuration: configuration(), dryRun: false))
            .joined(separator: " ")
        #expect(!submit.contains("--wait"))
        #expect(!submit.contains("--force-new"))

        let forced = try #require(
            ClusterProvisioningOperations.bootstrapArgv(
                site: site(), configuration: configuration(), dryRun: false,
                forceNewJob: true))
            .joined(separator: " ")
        #expect(forced.contains("--force-new"))

        // The reviewed plan must not depend on a recovery decision.
        let plan = try #require(
            ClusterProvisioningOperations.bootstrapArgv(
                site: site(), configuration: configuration(), dryRun: true,
                forceNewJob: true))
            .joined(separator: " ")
        #expect(!plan.contains("--force-new"))
        #expect(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: site(), configuration: configuration())
                == ClusterProvisioningOperations.bootstrapPlanHash(
                    site: site(), configuration: configuration()))
    }

    // MARK: WP5 Steps 6–7 — environment materialization (the default path)

    /// Materialization stated explicitly. Identical to `configuration()` since
    /// Step 7 made it the default; kept so the cases below name what they test.
    private func materializing() -> ClusterProvisioningConfiguration {
        var configuration = configuration()
        configuration.materializeEnvironmentFile = true
        return configuration
    }

    /// **WP5 Step 7 — the profile is the authority by default.** A caller that
    /// composes nothing special materializes: the argv carries the render's
    /// path and digest, so what the cluster sources is the site profile's
    /// environment rather than `bootstrap.sh`'s built-in constants.
    ///
    /// Opting out reproduces the pre-WP5 argv exactly — same words, same plan
    /// hash literal — which is what keeps `--no-materialize-env` an escape
    /// hatch rather than a third behaviour.
    @Test func materializationIsOnByDefaultAndTheLegacyPathIsExplicit() throws {
        #expect(ClusterProvisioningConfiguration().materializeEnvironmentFile == true)
        #expect(configuration().materializeEnvironmentFile == true)
        #expect(configuration() == materializing())
        #expect(
            ClusterProvisioningOperations.renderedEnvironment(
                site: site(), configuration: configuration()) != nil)
        for dryRun in [true, false] {
            let argv = try #require(
                ClusterProvisioningOperations.bootstrapArgv(
                    site: site(), configuration: configuration(), dryRun: dryRun))
                .joined(separator: " ")
            #expect(argv.contains("--env-file-from ~/.steerlab/rendered-cluster.env"))
            #expect(
                argv.contains(
                    "--env-file-sha256 \(ClusterEnvironmentRenderer.envFileDigest(site()))"))
        }

        // The opt-out: no rendered environment, and not a word about one.
        #expect(legacyConfiguration().materializeEnvironmentFile == false)
        #expect(
            ClusterProvisioningOperations.renderedEnvironment(
                site: site(), configuration: legacyConfiguration()) == nil)
        for dryRun in [true, false] {
            let argv = try #require(
                ClusterProvisioningOperations.bootstrapArgv(
                    site: site(), configuration: legacyConfiguration(), dryRun: dryRun))
                .joined(separator: " ")
            #expect(!argv.contains("--env-file-from"))
            #expect(!argv.contains("--env-file-sha256"))
        }
        // Both hashes are literals, so a regression cannot be masked by
        // recomputing "what it is now" on both sides. The second is the value
        // this project has approved since long before WP5.
        //
        // The FIRST moved at WP5 step 10 (was
        // c1fe221bb1ecdf3d8138addb8a7dff81124244081140cb9f8192d9dd16715eee):
        // the rendered environment now states the site's login-node policy, and
        // the environment's digest rides in this argv. That is the firewall
        // working as designed — a plan approved before the environment changed
        // must be re-planned and re-approved, not silently reused. The legacy
        // (non-materializing) hash below is unchanged, which is the proof that
        // only the rendered environment moved.
        #expect(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: site(), configuration: configuration())
                == "b25de7f311582371d13c8be980e2f87d50f243bceea7c83a5e8243da01e3d689")
        #expect(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: site(), configuration: legacyConfiguration())
                == "4e1921dd2191197a055d796f25f0f9ad6b5cca54fe0ecb6a91ed9fb9a17eb155")
    }

    /// Opting in appends exactly two words, and the digest they carry is the
    /// digest of the rendered environment — which is what puts the environment
    /// inside the reviewed plan hash (audit §3.3, §6.4).
    @Test func materializationPutsTheRenderedEnvironmentInsideThePlanHash() throws {
        let plan = try #require(
            ClusterProvisioningOperations.renderedEnvironment(
                site: site(), configuration: materializing()))
        #expect(plan.remotePath == "~/.steerlab/rendered-cluster.env")
        #expect(plan.text == ClusterEnvironmentRenderer.renderEnvFile(site()))
        #expect(plan.sha256 == ClusterEnvironmentRenderer.envFileDigest(site()))

        let argv = try #require(
            ClusterProvisioningOperations.bootstrapArgv(
                site: site(), configuration: materializing(), dryRun: true))
        let words = argv.joined(separator: " ")
        #expect(words.contains("--env-file-from ~/.steerlab/rendered-cluster.env"))
        #expect(words.contains("--env-file-sha256 \(plan.sha256)"))

        // Different plan, different approval — in both directions.
        let materialized = try #require(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: site(), configuration: materializing()))
        #expect(
            materialized
                != ClusterProvisioningOperations.bootstrapPlanHash(
                    site: site(), configuration: legacyConfiguration()))
        // An environment-only profile edit — one the legacy argv never
        // mentions — now invalidates the plan, because the digest moved.
        var edited = site()
        edited.environment.modules = ["Miniforge3"]
        #expect(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: edited, configuration: materializing()) != materialized)
    }

    /// The push is one short command, it runs AFTER the plan-hash gate and
    /// BEFORE anything is submitted, and it carries the whole file — the
    /// cluster never reconstructs it.
    @Test func theRenderedEnvironmentIsPushedBeforeAnythingIsSubmitted() async throws {
        let shell = RecordingShell([
            ClusterShellResult(exitCode: 0),
            submitResult(jobID: "3141"),
            completedResult(jobID: "3141", exitCode: 0),
        ])
        let operations = pollingOperations(shell: shell)
        let hash = try #require(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: site(), configuration: materializing()))
        let applied = try await operations.bootstrapApply(
            site: site(), configuration: materializing(), reviewedPlanHash: hash)
        #expect(applied.outcome.succeeded)

        let calls = await shell.recordedCalls()
        #expect(calls.count == 3)
        let push = calls[0].joined(separator: " ")
        #expect(push.contains("~/.steerlab/rendered-cluster.env"))
        #expect(push.contains("export STEERLAB_SERVER_PROFILE=cluster"))
        #expect(push.contains("mv "))
        // …and only then the submit.
        #expect(calls[1].joined(separator: " ").contains("submit-bootstrap-job.sh"))
        // The transcript names the digest rather than dumping the heredoc.
        let digest = ClusterEnvironmentRenderer.envFileDigest(site())
        #expect(applied.outcome.transcript.contains { $0.contains(digest) })
        #expect(applied.outcome.transcript.first?.contains("export ") != true)
    }

    /// A push that fails submits nothing. Half-materializing a site — new argv,
    /// old env file — is the failure mode the digest check exists to prevent,
    /// and it must never be reachable in the first place.
    @Test func aFailedEnvironmentPushSubmitsNothing() async throws {
        let shell = RecordingShell([
            ClusterShellResult(exitCode: 1, lines: ["permission denied"])
        ])
        let operations = pollingOperations(shell: shell)
        let hash = try #require(
            ClusterProvisioningOperations.bootstrapPlanHash(
                site: site(), configuration: materializing()))
        let applied = try await operations.bootstrapApply(
            site: site(), configuration: materializing(), reviewedPlanHash: hash)
        #expect(!applied.outcome.succeeded)
        #expect(applied.jobID == nil)
        #expect(applied.outcome.message.contains("nothing was submitted"))
        #expect(await shell.callCount() == 1)
    }

    /// **WP5 Step 7 — the opt-out is never silent.** A plan that lets
    /// `bootstrap.sh` write its own fallback env file is a plan whose site
    /// profile does not reach the cluster, so the transcript a human reviews
    /// has to say that before the hash is approved. The materializing plan
    /// names the digest instead, and neither transcript disturbs the report
    /// line the plan parser reads.
    @Test func aPlanThatWillNotMaterializeSaysSoBeforeItIsApproved() async throws {
        let reportLine = #"{"ok":true,"steps":{"envFile":"planned"},"envFile":"/e","prefix":"/p"}"#
        let legacy = await ClusterProvisioningOperations(
            shell: RecordingShell([ClusterShellResult(exitCode: 0, lines: [reportLine])]),
            secrets: RecordingSecretStore()
        ).bootstrapPlan(site: site(), configuration: legacyConfiguration())
        #expect(legacy.outcome.succeeded)
        #expect(legacy.report?.ok == true)
        let warning = try #require(legacy.outcome.transcript.first)
        #expect(warning.contains("WARNING"))
        #expect(warning.contains("materialization is OFF"))
        #expect(warning.contains("--no-materialize-env"))

        let materialized = await ClusterProvisioningOperations(
            shell: RecordingShell([ClusterShellResult(exitCode: 0, lines: [reportLine])]),
            secrets: RecordingSecretStore()
        ).bootstrapPlan(site: site(), configuration: materializing())
        let preamble = try #require(materialized.outcome.transcript.first)
        #expect(!preamble.contains("WARNING"))
        #expect(preamble.contains(ClusterEnvironmentRenderer.envFileDigest(site())))
    }

    /// An unreviewed plan still refuses before ANY command — the push must not
    /// have moved the gate.
    @Test func materializationDoesNotPushBeforeThePlanHashGate() async throws {
        let shell = RecordingShell()
        let operations = ClusterProvisioningOperations(
            shell: shell, secrets: RecordingSecretStore())
        await #expect(throws: ClusterLifecycleError.self) {
            _ = try await operations.bootstrapApply(
                site: self.site(), configuration: self.materializing(),
                reviewedPlanHash: nil)
        }
        #expect(await shell.callCount() == 0)
    }

    // MARK: Deployment identity (§7.8)

    @Test func aDevelopmentCheckoutIsIdentifiedHonestlyRatherThanAssumedCurrent()
        async throws
    {
        let shell = RecordingShell([ClusterShellResult(exitCode: 0, lines: ["{}"])])
        let operations = ClusterProvisioningOperations(
            shell: shell, secrets: RecordingSecretStore())
        var configuration = configuration()
        configuration.localPayloadPath = FileManager.default.temporaryDirectory
            .appending(component: "steerlab-dev-checkout-\(UUID().uuidString)").path
        let observation = await operations.observePayload(
            site: site(), configuration: configuration)
        guard case .unknown(let reason) = observation else {
            Issue.record("a manifest-less payload has no provable identity, got \(observation)")
            return
        }
        #expect(reason.contains("development checkout"))
    }

    @Test func aMatchingDeploymentManifestReadsCurrentAndAMismatchReadsStale()
        async throws
    {
        let payload = FileManager.default.temporaryDirectory
            .appending(component: "steerlab-payload-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: payload, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: payload) }
        let bytes = #"{"schemaVersion":1,"files":[]}"#
        try Data(bytes.utf8).write(
            to: payload.appending(
                component: ClusterProvisioner.deploymentManifestFileName))
        var configuration = configuration()
        configuration.localPayloadPath = payload.path

        let matching = ClusterProvisioningOperations(
            shell: RecordingShell([ClusterShellResult(exitCode: 0, lines: [bytes])]),
            secrets: RecordingSecretStore())
        #expect(
            await matching.observePayload(site: site(), configuration: configuration)
                == .current(
                    deploymentHash: ClusterSupportPaths.sha256Hex(Data(bytes.utf8))))

        let drifted = ClusterProvisioningOperations(
            shell: RecordingShell([
                ClusterShellResult(exitCode: 0, lines: [#"{"schemaVersion":1,"files":["x"]}"#])
            ]),
            secrets: RecordingSecretStore())
        guard case .stale = await drifted.observePayload(
            site: site(), configuration: configuration)
        else {
            Issue.record("a different deployed manifest must read stale")
            return
        }

        let absent = ClusterProvisioningOperations(
            shell: RecordingShell([ClusterShellResult(exitCode: 1)]),
            secrets: RecordingSecretStore())
        #expect(
            await absent.observePayload(site: site(), configuration: configuration)
                == .absent)
    }

    // MARK: Controller reconciliation (the doctrine, at the operation level)

    @Test func aFailedSchedulerQueryIsUnknownAndNeverAbsent() async throws {
        // ssh itself failed: the probe is unavailable.
        let unavailable = ClusterProvisioningOperations(
            shell: RecordingShell([
                ClusterShellResult(exitCode: 255, lines: ["ssh: connect: timed out"])
            ]),
            secrets: RecordingSecretStore())
        let unknown = await unavailable.observeController(
            site: site(), configuration: configuration(), recordedJobID: "4242",
            daemonHostPresent: true)
        guard case .unknown = unknown else {
            Issue.record("a failed probe must not claim the job is gone, got \(unknown)")
            return
        }

        // squeue ANSWERED that the job is not queued: that is real evidence.
        let answered = ClusterProvisioningOperations(
            shell: RecordingShell([ClusterShellResult(exitCode: 0, lines: [])]),
            secrets: RecordingSecretStore())
        #expect(
            await answered.observeController(
                site: site(), configuration: configuration(), recordedJobID: "4242",
                daemonHostPresent: false) == .absent)
    }

    @Test func hostRecordClassificationNeverTrustsAStaleFile() {
        typealias Operations = ClusterProvisioningOperations
        #expect(
            Operations.classifyDaemonHost(host: nil, controller: .running(jobID: "1"))
                == .absent)
        #expect(
            Operations.classifyDaemonHost(host: "c4-12", controller: .running(jobID: "1"))
                == .current(host: "c4-12"))
        for controller: ClusterControllerObservation in [
            .absent, .pending(jobID: "1", reason: "Resources"),
            .failed(jobID: "1", state: "TIMEOUT"), .unknown(reason: "probe failed"),
        ] {
            guard case .stale = Operations.classifyDaemonHost(
                host: "c4-12", controller: controller)
            else {
                Issue.record("host record must be stale for controller \(controller)")
                continue
            }
        }
    }

    // MARK: Token import

    @Test func theImportedTokenGoesToTheSecretStoreAndNowhereElse() async throws {
        let secret = "sk-do-not-leak-1234567890"
        let secrets = RecordingSecretStore()
        let shell = RecordingShell([ClusterShellResult(exitCode: 0, lines: [secret])])
        let operations = ClusterProvisioningOperations(shell: shell, secrets: secrets)
        let outcome = await operations.importRemoteToken(
            site: site(), tokenKey: "login.test:8080")
        #expect(outcome == .imported)
        #expect(secrets.snapshot["login.test:8080"] == secret)
        // The value is in NO argv (argv is visible to `ps` on both hosts) and
        // in no returned description.
        for argv in await shell.recordedCalls() {
            #expect(!argv.joined(separator: " ").contains(secret))
        }
        #expect(!"\(outcome)".contains(secret))

        // An already-present token is not re-read at all.
        let second = await operations.importRemoteToken(
            site: site(), tokenKey: "login.test:8080")
        #expect(second == .alreadyPresent)
        #expect(await shell.callCount() == 1)
    }

    @Test func anUnreadableTokenReportsWhyWithoutInventingOne() async throws {
        let secrets = RecordingSecretStore()
        let operations = ClusterProvisioningOperations(
            shell: RecordingShell([ClusterShellResult(exitCode: 1)]), secrets: secrets)
        let outcome = await operations.importRemoteToken(
            site: site(), tokenKey: "login.test:8080")
        guard case .unavailable(let reason) = outcome else {
            Issue.record("expected .unavailable, got \(outcome)")
            return
        }
        #expect(reason.contains("~/.steerlab-token"))
        #expect(secrets.snapshot.isEmpty)
    }

    // MARK: Configuration problems

    @Test func anUnusableSiteIsRefusedBeforeAnyProbe() {
        var blankHost = site()
        blankHost.transport = .ssh(
            host: "  ", proxyJump: nil, remotePort: 8080, vpnExpected: false)
        let problems = ClusterLifecycleInspector.configurationProblems(
            site: blankHost, configuration: configuration())
        #expect(problems.contains { $0.contains("no SSH host") })

        var noScheduler = site()
        noScheduler.scheduler = .none
        #expect(
            ClusterLifecycleInspector.configurationProblems(
                site: noScheduler, configuration: configuration())
                .contains { $0.contains("no Slurm scheduler") })

        // The healthy site has nothing to complain about.
        #expect(
            ClusterLifecycleInspector.configurationProblems(
                site: site(), configuration: configuration()).isEmpty)
    }

    // MARK: ControlMaster — `-O check` alone is not proof of life

    @Test func aZombieMasterThatPassesCheckButRunsNothingReadsUnresponsive() async {
        // Observed live 2026-08-12: a master that survives a network change
        // answers `-O check` from its local socket while every real command
        // fails. First answer: the check. Second: the `true` probe.
        let shell = RecordingShell([
            ClusterShellResult(exitCode: 0),
            ClusterShellResult(exitCode: 255, lines: ["Connection timed out"]),
        ])
        let operations = ClusterProvisioningOperations(
            shell: shell, secrets: RecordingSecretStore())
        #expect(await operations.checkControlMaster(site: site()) == .unresponsive)
        // The second command was a REAL one through the master, cheap and
        // bounded, not another `-O` control query.
        let calls = await shell.recordedCalls()
        #expect(calls.count == 2)
        #expect(Array(calls[1].suffix(2)) == ["login.test", "true"])
        #expect(calls[1].contains("ConnectTimeout=5"))
    }

    @Test func aMasterThatChecksAndRunsCommandsReadsAlive() async {
        let shell = RecordingShell([
            ClusterShellResult(exitCode: 0),
            ClusterShellResult(exitCode: 0),
        ])
        let operations = ClusterProvisioningOperations(
            shell: shell, secrets: RecordingSecretStore())
        #expect(await operations.checkControlMaster(site: site()) == .alive)

        // A failed `-O check` never reaches the command probe: one call only,
        // and the wedged-socket classification is preserved.
        let wedged = RecordingShell([
            ClusterShellResult(
                exitCode: 255, lines: ["Control socket connect: refused"])
        ])
        let wedgedOperations = ClusterProvisioningOperations(
            shell: wedged, secrets: RecordingSecretStore())
        #expect(
            await wedgedOperations.checkControlMaster(site: site()) == .unresponsive)
        #expect(await wedged.callCount() == 1)
    }

    // MARK: Forward identity (§7.7)

    @Test func forwardIdentityIsSiteRemoteTargetAndPortTogether() {
        let identity = SSHClusterTunnelController.forwardIdentity(
            site: site(), targetHost: "c4-12", localPort: 8712)
        #expect(identity == "ssh://login.test/c4-12:8080->127.0.0.1:8712")
        // A different node, remote port, or local port is a DIFFERENT forward.
        #expect(
            SSHClusterTunnelController.forwardIdentity(
                site: site(), targetHost: "c4-13", localPort: 8712) != identity)
        #expect(
            SSHClusterTunnelController.forwardIdentity(
                site: site(), targetHost: "c4-12", localPort: 8713) != identity)
        #expect(
            SSHClusterTunnelController.forwardIdentity(
                site: .gpuWorkstation, targetHost: "c4-12", localPort: 8712) == nil)
    }

    // MARK: The deploy repairs the rendered controller script (§1, 2026-08-20)

    /// The gap the field report named: `cluster push` rsyncs the controller-job
    /// TEMPLATE and leaves last week's RENDERED child of it in the metadata
    /// root, where the operator's standing `sbatch` ritual finds it. A deploy
    /// is a flow that may write, so it repairs rather than reporting.
    @Test func pushReRendersAStaleControllerScriptAgainstTheTemplateItJustShipped() async {
        let digest = String(repeating: "a", count: 64)
        let shell = RecordingShell([
            ClusterShellResult(exitCode: 0),                       // rsync
            ClusterShellResult(                                    // staleness probe
                exitCode: 0,
                lines: [
                    "rendered-exists: yes",
                    "rendered-stamp: sha256=\(String(repeating: "b", count: 64)) "
                        + "renderedAt=2026-08-17T04:00:00Z",
                    "rendered-chain: template-\(String(repeating: "b", count: 64))",
                    "template-sha256: \(digest)",
                ]),
            ClusterShellResult(exitCode: 0, lines: ["/home/me/envs/steerlab"]),  // prefix
            ClusterShellResult(                                    // the re-render
                exitCode: 0,
                lines: ["template=controller-job.sbatch.template sha256=\(digest)"]),
        ])
        let operations = ClusterProvisioningOperations(
            shell: shell, secrets: RecordingSecretStore())
        let outcome = await operations.push(
            site: site(), configuration: configuration())
        #expect(outcome.succeeded)
        #expect(outcome.message.contains("re-rendered"))
        #expect(outcome.message.contains("bbbbbbbbbbbb"))
        let calls = await shell.recordedCalls()
        #expect(calls.count == 4)
        // The LAST call is the render, and it submits nothing: a deploy must
        // never queue a controller as a side effect.
        let render = calls.last?.last ?? ""
        #expect(render.contains("> ~/.steerlab/controller-job.sbatch"))
        #expect(!render.contains("&& sbatch "))
    }

    @Test func pushLeavesACurrentControllerScriptAlone() async {
        let digest = String(repeating: "c", count: 64)
        let shell = RecordingShell([
            ClusterShellResult(exitCode: 0),
            ClusterShellResult(
                exitCode: 0,
                lines: [
                    "rendered-exists: yes",
                    "rendered-stamp: sha256=\(digest) renderedAt=2026-08-20T09:00:00Z",
                    "rendered-chain: template-\(digest)",
                    "template-sha256: \(digest)",
                ]),
        ])
        let operations = ClusterProvisioningOperations(
            shell: shell, secrets: RecordingSecretStore())
        let outcome = await operations.push(
            site: site(), configuration: configuration())
        #expect(outcome.succeeded)
        #expect(!outcome.message.contains("re-rendered"))
        #expect(await shell.callCount() == 2)
    }

    /// An unreadable probe never triggers a blind re-render, and never fails
    /// the deploy — but it does say the exact command, because silence is the
    /// failure mode this whole change exists to end.
    @Test func pushWarnsRatherThanGuessingWhenTheScriptCannotBeChecked() async {
        let shell = RecordingShell([
            ClusterShellResult(exitCode: 0),
            ClusterShellResult(exitCode: 255),
        ])
        let operations = ClusterProvisioningOperations(
            shell: shell, secrets: RecordingSecretStore())
        let outcome = await operations.push(
            site: site(), configuration: configuration())
        #expect(outcome.succeeded)
        #expect(outcome.message.contains("could not be checked"))
        #expect(outcome.message.contains("--render-only"))
        #expect(await shell.callCount() == 2)
    }

    /// A login-daemon site has no controller script at all — the deploy must
    /// not grow a remote round-trip for a topology that cannot use it.
    @Test func pushProbesNothingExtraOnALoginDaemonSite() async {
        var loginSite = site()
        loginSite.topology = .loginDaemon
        let shell = RecordingShell([ClusterShellResult(exitCode: 0)])
        let operations = ClusterProvisioningOperations(
            shell: shell, secrets: RecordingSecretStore())
        let outcome = await operations.push(
            site: loginSite, configuration: configuration())
        #expect(outcome.succeeded)
        #expect(await shell.callCount() == 1)
    }

}
