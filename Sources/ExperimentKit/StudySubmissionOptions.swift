import Foundation
import Observation
import SteeringKit

/// Editable compute options, separate from scientific draft fields and submitted jobs.
@Observable @MainActor
public final class StudySubmissionOptions {
    public var remoteExecutor = "local"
    // Defaults match the common intent — "run my study" — not the most
    // cautious combination: verify+dryRun defaults produced submissions that
    // appeared to do nothing. Dry run stays available as an explicit toggle.
    public var remoteVerb = "run"
    public var remoteDryRun = false
    public var remoteGres = "A100"
    /// 4 hours (2026-08-03): the old 30-minute default walltime-killed the
    /// first real 27B sweep twenty minutes in — 30m fit only smoke tests,
    /// and a killed job costs a full queue wait to retry. 4h covers a
    /// trimmed-grid multi-concept sweep or validate with headroom; the
    /// field stays editable for anything bigger.
    public var remoteWalltime = "04:00:00"
    /// Resume-on-checkpoint policy for Slurm submissions — DEFAULT ON with
    /// the server's shipped limit (2026-07-22 incident: a checkpointed run
    /// continuing is what the researcher asked for by submitting it; OFF is
    /// the surprising choice). Sent only for the slurm executor; what was
    /// sent is stamped into the submission transcript line.
    public var remoteResumePolicy = RemoteResumePolicy()
    /// "Parallel GPU jobs" (2026-07-22): shard a Slurm run across K sibling
    /// GPU jobs (default 1 = single job, the historical path). Encoded on
    /// the submission only when it applies (`ShardedSubmission` rule);
    /// execution logistics only — never in the manifest or content hash.
    public var remoteParallelJobs = 1
    public init() {}
    public var snapshot: StudySubmissionRequest {
        StudySubmissionRequest(
            executor: remoteExecutor, verb: remoteVerb, dryRun: remoteDryRun,
            gres: remoteGres, walltime: remoteWalltime, resumePolicy: remoteResumePolicy,
            parallelJobs: remoteParallelJobs)
    }

}

/// Values captured before the first await of a submission.
public struct StudySubmissionRequest: Sendable, Equatable {
    public let executor: String
    public let verb: String
    public let dryRun: Bool
    public let gres: String
    public let walltime: String
    public let resumePolicy: RemoteResumePolicy
    public let parallelJobs: Int
}

extension StudySubmissionOptions {
    public func snapshot(verb: String?) -> StudySubmissionRequest {
        guard let verb else { return snapshot }
        return snapshot.replacingVerb(verb)
    }
}

extension StudySubmissionRequest {
    public func replacingVerb(_ verb: String) -> Self {
        Self(
            executor: executor, verb: verb, dryRun: dryRun, gres: gres,
            walltime: walltime, resumePolicy: resumePolicy, parallelJobs: parallelJobs)
    }
    var resources: [String: String] {
        ["gres": gres, "walltime": walltime]
            .filter { !$0.value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }
    var effectiveResumePolicy: RemoteResumePolicy? {
        executor == "slurm" ? resumePolicy : nil
    }
}
