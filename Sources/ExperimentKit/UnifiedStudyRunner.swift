import Foundation
import Observation

/// Unified run submission over captured options and explicit job/presentation capabilities.
@Observable @MainActor
public final class UnifiedStudyRunner {

    public private(set) var isSubmitting = false
    public private(set) var statusLine: String?
    /// Preflight from the LAST successful submission (ok/warn — proceeded).
    public private(set) var preflight: PreflightPresentation?
    /// Preflight refusal from the last attempt (verdict fail — stopped).
    /// Non-nil is what makes the view offer "Override (forced)".
    public private(set) var refusal: PreflightPresentation?
    public private(set) var lastJobID: String?

    public init() {}

    /// Clear surfaced preflight state (selection changed, study switched).
    public func clearPreflight() {
        preflight = nil
        refusal = nil
        statusLine = nil
    }

    /// The one entry point behind the primary Run button.
    public func run(
        manifest: ExperimentManifest,
        decision: SubstrateRouting.Decision,
        request: StudySubmissionRequest,
        jobs: StudyRemoteJobController,
        runLocal: @MainActor () async -> Void,
        note: @MainActor (String, PanelNotice.Severity) -> Void,
        cluster: ClusterConnectionStore,
        force: Bool = false
    ) async {
        guard decision.runBlockedReason == nil else {
            statusLine = decision.runBlockedReason
            return
        }
        switch decision.selection {
        case .thisMac:
            // Existing local path, untouched (greedy gate, live viewer,
            // display log all included).
            await runLocal()
        case .server:
            await submitBundle(
                manifest: manifest, request: request, jobs: jobs, note: note,
                cluster: cluster, force: force)
        }
    }

    /// Package → upload → submit-bundle (the portable path), capturing the
    /// WS4 preflight report on success and mining it out of a refusal.
    public func submitBundle(
        manifest: ExperimentManifest,
        request: StudySubmissionRequest,
        jobs: StudyRemoteJobController,
        note: @MainActor (String, PanelNotice.Severity) -> Void,
        cluster: ClusterConnectionStore,
        force: Bool
    ) async {
        guard !isSubmitting else { return }
        cluster.loadStoredToken()
        guard let client = cluster.client else {
            statusLine = "invalid server URL"
            return
        }
        let origin = RemoteJobOrigin(connection: client.profile, workspaceRoot: ExperimentStore.workspaceRoot)
        let capabilities = cluster.capabilities
        let substrate = cluster.substrateLabel
        isSubmitting = true
        // Install Update waits for this (it quits the app).
        UpdateBusyWork.shared.beginSubmission()
        defer {
            isSubmitting = false
            UpdateBusyWork.shared.endSubmission()
        }
        refusal = nil
        preflight = nil
        // Frozen-on-server guard — shared with the legacy panel path
        // (engineer finding 2026-07-19: the primary Run button previously
        // bypassed it): a local DRAFT must not shadow the server's frozen
        // same-named study.
        if let conflict = await client.frozenOnServerConflict(
            study: manifest.name, localStatus: manifest.status)
        {
            statusLine = conflict
            note(conflict, .error)
            return
        }
        let verb = request.verb
        // Old-server guard (2026-07-21): a stochastic saved-agent study on a
        // server without study-owned sampling would run the agents greedy
        // while the baseline samples — refuse BEFORE packaging/uploading.
        if let refusal = SubstrateRouting.stochasticVariantSubmissionRefusal(
            temperature: manifest.temperature,
            samplesPerItem: manifest.samplesPerItem,
            variantConditionCount: manifest.variantConditions.count,
            verb: verb,
            capabilities: capabilities)
        {
            statusLine = refusal
            note(refusal, .error)
            return
        }
        // Temperature 0 with several seeds (public issue #9): refused BEFORE
        // packaging/uploading, on every server, with the repair the command
        // line gives (a frozen study is duplicated first).
        if let refusal = SubstrateRouting.greedySeedListSubmissionRefusal(
            manifest: manifest, verb: verb)
        {
            let line = SubstrateRouting.greedySeedListSubmissionLine(
                refusal: refusal, manifest: manifest)
            statusLine = line
            note(line, .error)
            return
        }
        // An attached agent arm that breaks a settings-only rule (FR6-B
        // round 2): refused before packaging, on every server.
        if let refusal = SubstrateRouting.agentArmSubmissionRefusal(
            manifest: manifest, verb: verb)
        {
            let line = "\(refusal.reason). To run it: "
                + (refusal.lifecycleRefusal?.repairAction ?? "")
            statusLine = line
            note(line, .error)
            return
        }
        // Old-server guard (round 3 of issue #9): a readout-only study at
        // temperature 0 with several seeds or samples per item would be
        // refused after the model load by a server without the fix.
        if let refusal = SubstrateRouting.greedyRulesSubmissionRefusal(
            manifest: manifest, verb: verb, capabilities: capabilities,
            server: client.profile.baseURL.absoluteString)
        {
            let line = refusal.reason + " " + refusal.repair
            statusLine = line
            note(line, .error)
            return
        }
        // Old-server guard (issue #3): a declared derivedSHA256 at one sample
        // per item would run with the fixed seed list on a server without
        // the declared seed policy — refuse BEFORE packaging/uploading.
        if let refusal = SubstrateRouting.declaredSeedPolicySubmissionRefusal(
            manifest: manifest, verb: verb, capabilities: capabilities,
            server: client.profile.baseURL.absoluteString)
        {
            statusLine = refusal
            note(refusal, .error)
            return
        }
        // Old-server guard (issue #6): a declared prompt reading would be
        // ignored by a server without it — refuse BEFORE packaging/uploading.
        if let refusal = SubstrateRouting.promptReadingSubmissionRefusal(
            manifest: manifest, verb: verb, capabilities: capabilities,
            server: client.profile.baseURL.absoluteString)
        {
            statusLine = refusal
            note(refusal, .error)
            return
        }
        // Scope-drift guard (2026-08-06 field incident): a stale
        // outcomeInstrumentScope pin after Duplicate & Adjust + task-file
        // swap refuses HERE, before packaging/upload — the server would
        // refuse the same way, but only on the compute node.
        if let refusal = ExperimentTasks.scopeDriftSubmitRefusal(
            for: manifest, verb: verb)
        {
            statusLine = refusal
            note(refusal, .error)
            return
        }
        let dryRun = request.dryRun
        let resources = request.resources
        // Custom code: a step that can execute the study's agents is not sent
        // while an expert provider in it is unacknowledged (CustomCodeNotice).
        if let refusal = CustomCodeNotice.submissionRefusalText(
            study: manifest.name, verb: verb, dryRun: dryRun,
            workspaceRoot: origin.workspaceRoot)
        {
            statusLine = refusal
            note(refusal, .error)
            return
        }
        do {
            statusLine = "packaging \(manifest.name)…"
            // Packaging copies files, hashes them, and shells out to tar;
            // keep it off the main actor (same rule as the legacy path).
            let source = RunBundlePackager.captureSource(manifest)
            let bundle = try await Task.detached {
                try RunBundlePackager.packageExperiment(manifest, source: source)
            }.value
            statusLine = "uploading \(bundle.lastPathComponent)…"
            let uploaded = try await client.uploadBundle(bundle)
            statusLine =
                force
                ? "submitting \(verb) (forced past preflight)…"
                : "submitting \(verb)…"
            // Resume-on-checkpoint policy (2026-07-22 incident): Slurm
            // submissions carry the panel's toggle — default ON — and the
            // transcript line below stamps what was sent.
            let resumePolicy = request.effectiveResumePolicy
            let submission = try await client.submitBundle(
                path: uploaded.path,
                verb: verb,
                executor: request.executor,
                dryRun: dryRun,
                resources: resources,
                resumePolicy: resumePolicy,
                force: force,
                parallelJobs: request.parallelJobs)
            preflight = PreflightPresentation.from(submission.preflight)
            lastJobID = submission.jobId
            jobs.recordOrigin(origin, jobID: submission.jobId)
            var submitted = StudySubmissionPresentation.bundleSubmittedStatus(
                study: manifest.name, verb: verb, dryRun: dryRun,
                substrate: substrate, jobID: submission.jobId)
            if let resumePolicy {
                submitted += " — \(resumePolicy.transcriptStamp)"
            }
            // The sharding stamp derives from the server's RESPONSE (the
            // shard ids it actually created), never from the request — the
            // server may have ignored the fan-out (finding 5, 2026-07-22).
            if let stamp = ShardedSubmission.transcriptStamp(
                shardJobIDs: submission.shardJobIDs)
            {
                submitted += " — \(stamp)"
            }
            if let summary = preflight?.inlineSummary {
                submitted += " — \(summary)"
            }
            // Issue #11: the server answered with a job it had already made.
            if let already = submission.alreadySubmittedStatus {
                submitted = already
            }
            statusLine = submitted
            // Follow through the job controller so the log lands in the same
            // places the legacy flow used (remote log lines, reconnectable
            // job id), and the recent-jobs list learns about it.
            await jobs.reconnectRemoteJob(submission.jobId, client: client)
            await jobs.refreshRecentServerJobs(client: client)
        } catch let unknown as SubmissionOutcomeUnknown {
            // Issue #11: not a failure; the outcome is unknown.
            statusLine = unknown.appSentence
        } catch let error as ClusterClient.ClientError {
            if case .badResponse(_, let body) = error,
                let refused = PreflightPresentation.refusal(fromErrorBody: body)
            {
                refusal = refused
                statusLine =
                    refused.inlineSummary
                    ?? "submission refused by preflight — review the failing checks"
            } else {
                statusLine = "remote submit failed: \(ClusterClient.unwrappingDetail(error))"
            }
        } catch {
            statusLine = "remote submit failed: \(error.localizedDescription)"
        }
    }
}
