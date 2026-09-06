import Foundation
import SteeringKit

/// Portable bundle submission for legacy, batch and pipeline entry points.
/// Owns sequencing and refusal reporting; job identity/logs stay in the shared job owner.
@MainActor
public final class StudyBundleSubmissionController {
    private let jobs: StudyRemoteJobController
    init(jobs: StudyRemoteJobController) { self.jobs = jobs }
    private func note(_ text: String, severity: PanelNotice.Severity) {
        jobs.presentation.note(text, severity)
    }
    private func contextChanged() -> Result<String, StudyBatchSubmission.Failure> {
        let reason =
            "bundle submission stopped because the workspace or server changed; submit again from the intended workspace"
        jobs.remoteStatus = reason
        note(reason, severity: .warning)
        return .failure(.init(reason: reason))
    }

    @discardableResult
    func submit(
        _ manifest: ExperimentManifest,
        request: StudySubmissionRequest,
        capabilities: ClusterCapabilities?,
        substrate: String?,
        transport: StudyBundleTransport,
        isCurrent: @escaping @MainActor () -> Bool = { true },
        follow: (@MainActor (String, String, String, Bool) -> Void)? = nil
    ) async -> Result<String, StudyBatchSubmission.Failure> {
        let submissionVerb = request.verb
        guard isCurrent(), !Task.isCancelled else { return contextChanged() }
        // Frozen-on-server guard — SHARED with every bundle path
        // (`ClusterClient.frozenOnServerConflict`): a local draft must not
        // shadow the server's frozen same-named study.
        if let conflict = await transport.frozenConflict(manifest) {
            jobs.remoteStatus = conflict
            note(conflict, severity: .error)
            return .failure(StudyBatchSubmission.Failure(reason: conflict))
        }
        guard isCurrent(), !Task.isCancelled else { return contextChanged() }
        // Old-server guard (2026-07-21): a stochastic saved-agent study on a
        // server without study-owned sampling would run the agents greedy
        // while the baseline samples — refuse BEFORE packaging/uploading
        // (same rule as UnifiedStudyRunner.submitBundle).
        if let refusal = SubstrateRouting.stochasticVariantSubmissionRefusal(
            temperature: manifest.temperature,
            samplesPerItem: manifest.samplesPerItem,
            variantConditionCount: manifest.variantConditions.count,
            verb: submissionVerb,
            capabilities: capabilities)
        {
            jobs.remoteStatus = refusal
            note(refusal, severity: .error)
            return .failure(StudyBatchSubmission.Failure(reason: refusal))
        }
        // Scope-drift guard (2026-08-06 field incident) — same rule as
        // UnifiedStudyRunner.submitBundle: a stale outcomeInstrumentScope
        // pin refuses at SUBMIT, before packaging/upload, instead of on the
        // compute node after the model staged.
        if let refusal = ExperimentTasks.scopeDriftSubmitRefusal(
            for: manifest, verb: submissionVerb)
        {
            jobs.remoteStatus = refusal
            note(refusal, severity: .error)
            return .failure(StudyBatchSubmission.Failure(reason: refusal))
        }
        do {
            jobs.remoteStatus = "packaging \(manifest.name)..."
            jobs.remoteLogLines = []
            // Packaging copies files, hashes them, and shells out to tar; keep it
            // off the main actor so the UI doesn't freeze on a real study bundle.
            let bundle = try await transport.package(manifest)
            guard isCurrent(), !Task.isCancelled else { return contextChanged() }
            jobs.remoteStatus = "uploading \(bundle.lastPathComponent)..."
            let uploaded = try await transport.upload(bundle)
            jobs.remoteLastUploadedBundle = uploaded
            guard isCurrent(), !Task.isCancelled else { return contextChanged() }
            jobs.remoteStatus = "submitting \(submissionVerb)..."
            let resumePolicy = request.effectiveResumePolicy
            let submission = try await transport.submit(uploaded, request)
            jobs.recordOrigin(transport.origin, jobID: submission.jobId)
            jobs.remoteJobID = submission.jobId
            let substrate = substrate ?? submission.executor
            var submitted = StudySubmissionPresentation.bundleSubmittedStatus(
                study: manifest.name, verb: submissionVerb, dryRun: request.dryRun,
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
            jobs.remoteStatus = submitted
            note(submitted, severity: .info)
            jobs.noteRecentServerJob(
                id: submission.jobId, verb: "\(submissionVerb) (bundle)",
                study: manifest.name, state: "submitted")
            follow?(submission.jobId, submissionVerb, manifest.name, request.dryRun)
            return .success(submission.jobId)
        } catch {
            jobs.remoteStatus = "remote submit failed: \(error)"
            let refusal =
                "Couldn't submit the bundle to the server — nothing is "
                + "running; check the connection in Compute and submit "
                + "again. Details: \(error)"
            note(refusal, severity: .error)
            return .failure(StudyBatchSubmission.Failure(reason: refusal))
        }
    }
}
