import Foundation

/// Multi-GPU sharded submissions (2026-07-22): pure rules for the app side of
/// the server's `parallelJobs` fan-out, kept out of view code so every rule is
/// unit-tested. The server shards a Slurm `run` (or run-first `pipeline`)
/// submission into K sibling sbatch jobs under one parent job record and
/// merges the partials back into one ordinary run — byte-identical to a
/// single-job run because every generation record is independent
/// (per-record derived seeding, no cross-record state). `parallelJobs` is
/// execution logistics only: it never enters the manifest or content hash.
public enum ShardedSubmission {

    /// Verbs whose submission the server can shard. `pipeline` shards only
    /// when its declared chain starts with `run` — the server decides that
    /// from the bundle's manifest and logs a note otherwise, so the client
    /// encodes for both and lets the authority rule.
    public static let shardableVerbs: Set<String> = ["run", "pipeline"]

    /// Default stepper cap when the site profile declares no
    /// `maxParallelGPUJobs` (uncapped at the site, bounded in the UI).
    public static let defaultStepperCap = 16

    /// The `parallelJobs` value to ENCODE on a submission body, or nil to
    /// omit the field entirely (older servers must never see it, and 1 is
    /// the implicit default). Encoded only when > 1 AND the executor is
    /// slurm AND the verb shards.
    public static func encodedParallelJobs(
        requested: Int, executor: String, verb: String
    ) -> Int? {
        guard requested > 1, executor == "slurm", shardableVerbs.contains(verb)
        else { return nil }
        return requested
    }

    /// The submission-transcript stamp, derived from the server's RESPONSE
    /// (the shard job ids it actually created) — never from the request
    /// (external review 2026-07-22, finding 5: the server shards only `run`
    /// and run-FIRST pipelines; a stamp from the requested count claimed
    /// sharding for chains the server ran as one job). Nil when the server
    /// did not shard.
    public static func transcriptStamp(shardJobIDs: [String]?) -> String? {
        guard let shardJobIDs, shardJobIDs.count > 1 else { return nil }
        return "sharded across \(shardJobIDs.count) GPU jobs"
    }

    /// Why the "Parallel GPU jobs" stepper cannot apply to this submission
    /// (nil = sharding is available). Mirrors the server's
    /// `_resolve_parallel_jobs` rule: the Slurm `run` verb always shards; a
    /// Slurm `pipeline` shards only when its DECLARED chain starts with
    /// `run`; nothing else shards. The Remote-options stepper disables with
    /// this explanation instead of letting a request the server will ignore
    /// look honored.
    public static func shardingUnavailableReason(
        verb: String, executor: String, declaredPipelineStages: [String]?
    ) -> String? {
        guard executor == "slurm" else {
            return "sharding unavailable — only Slurm submissions shard "
                + "across GPU jobs"
        }
        if verb == "run" { return nil }
        if verb == "pipeline" {
            guard let stages = declaredPipelineStages, let first = stages.first
            else {
                return "sharding unavailable — this manifest declares no "
                    + "pipeline stages to shard; only run-first chains shard"
            }
            if first == "run" { return nil }
            return "sharding ignored — this pipeline starts with "
                + "'\(first)'; only run-first chains shard"
        }
        return "sharding unavailable — the '\(verb)' verb does not shard; "
            + "only 'run' (and a run-first pipeline) has an independent "
            + "per-record record set"
    }

    /// The manifest's DECLARED pipeline stage list, mirroring the server
    /// resolver (`pipeline_spec.resolve_pipeline`): nil when no pipeline
    /// block exists (the pipeline verb refuses anyway); the default chain
    /// `extract → validate → sweep → promote → run` when the block declares
    /// no/empty stages; the declared names otherwise (validity is the
    /// verify layer's job — this only answers "what is the first stage?").
    public static func declaredPipelineStages(_ block: JSONValue?) -> [String]? {
        guard let block, case .object(let pipeline) = block else { return nil }
        let defaultStages = ["extract", "validate", "sweep", "promote", "run"]
        guard case .array(let items)? = pipeline["stages"] else {
            return defaultStages
        }
        let names = items.compactMap { item -> String? in
            if case .string(let name) = item { return name }
            return nil
        }
        return names.isEmpty ? defaultStages : names
    }

    /// The "Parallel GPU jobs" stepper's upper bound: the site profile's
    /// `maxParallelGPUJobs` when declared (clamped to at least 1), else the
    /// default cap.
    public static func stepperCap(siteMax: Int?) -> Int {
        guard let siteMax else { return defaultStepperCap }
        return max(1, siteMax)
    }

    /// Help text for the site-profile field: where the researcher finds the
    /// real per-user GPU-job limit at a Slurm site.
    public static let siteFieldHelp =
        "the most GPU jobs one sharded submission may fan out into at this "
        + "site — find your real per-user limit with `sacctmgr show qos "
        + "format=Name,MaxTRESPerUser` (empty = uncapped stepper, max "
        + "\(defaultStepperCap))"
}

// MARK: - Shard fields on job records

extension RemoteJobRecord {

    /// The K shard child JOB RECORD ids of a sharded parent, in shard order
    /// (`result.shardJobs`). Nil/empty for ordinary jobs.
    public var shardJobIDs: [String]? {
        guard case .array(let values)? = result?["shardJobs"] else { return nil }
        let ids = values.compactMap { value -> String? in
            if case .string(let s) = value { return s }
            return nil
        }
        return ids.isEmpty ? nil : ids
    }

    /// A shard child's own identity (`result.shard = {index, count}`),
    /// stamped by the shard's bundle-execute child record and preserved
    /// across reconciler folds.
    public var shardIndex: Int? {
        guard case .object(let shard)? = result?["shard"],
            case .number(let value)? = shard["index"]
        else { return nil }
        return Int(value)
    }

    public var shardCount: Int? {
        guard case .object(let shard)? = result?["shard"],
            case .number(let value)? = shard["count"]
        else { return nil }
        return Int(value)
    }

    /// The sharded parent this job belongs to (shard children and the
    /// pipeline continuation both carry it).
    public var parentJobID: String? {
        if case .string(let value)? = result?["parentJob"] { return value }
        return nil
    }

    /// The merged run directory a completed sharded parent reports.
    public var mergedRunDirectory: String? {
        if case .string(let value)? = result?["runDirectory"] { return value }
        return nil
    }
}

// MARK: - Display grouping

/// Groups a job list so sharded parents render one row with per-shard chips
/// while every other job renders as before. Pure and unit-tested; the view
/// only lays out what this derives.
public enum ShardedJobGrouping {

    /// Jobs to render at the TOP level: everything except shard children /
    /// continuations whose parent record is present in the list (those
    /// render as chips under the parent row). A child whose parent is
    /// missing stays top-level — never hide a job.
    public static func topLevel(_ jobs: [RemoteJobRecord]) -> [RemoteJobRecord] {
        let ids = Set(jobs.map(\.id))
        return jobs.filter { job in
            guard let parent = job.parentJobID else { return true }
            return !ids.contains(parent)
        }
    }

    /// The shard children (plus any pipeline continuation) of a parent, in
    /// shard order — resolved through the parent's own `shardJobs` list so
    /// the order is the shard order, not list order.
    public static func children(
        of parent: RemoteJobRecord, in jobs: [RemoteJobRecord]
    ) -> [RemoteJobRecord] {
        guard let ids = parent.shardJobIDs else { return [] }
        let byID = Dictionary(uniqueKeysWithValues: jobs.map { ($0.id, $0) })
        var rows = ids.compactMap { byID[$0] }
        // A pipeline continuation is parented but not a shard: append last.
        rows += jobs.filter {
            $0.parentJobID == parent.id && !ids.contains($0.id)
        }
        return rows
    }

    /// True for a JUDGE WORKER child (post-generation judge fan-out,
    /// 2026-07-23): one sibling job per distinct local judge model, kind
    /// `study-judge-worker` on the wire.
    public static func isJudgeWorker(_ child: RemoteJobRecord) -> Bool {
        child.kind == "study-judge-worker"
    }

    /// True for a parented child row that is NOT a shard and not a judge
    /// worker — the pipeline continuation (the same rule `chipLabel` uses
    /// to label it).
    public static func isContinuation(_ child: RemoteJobRecord) -> Bool {
        child.shardIndex == nil && child.shardCount == nil
            && child.parentJobID != nil && !isJudgeWorker(child)
    }

    /// One-line aggregate for the parent row, e.g.
    /// "3 shards: 2 succeeded · 1 running". Counts SHARDS only — the
    /// pipeline continuation and judge workers render as their own chips
    /// and must not inflate the shard count (display nit, 2026-07-22;
    /// judge fan-out 2026-07-23).
    public static func aggregateLine(children: [RemoteJobRecord]) -> String? {
        let shards = children.filter { !isContinuation($0) && !isJudgeWorker($0) }
        guard !shards.isEmpty else { return nil }
        var counts: [String: Int] = [:]
        var order: [String] = []
        for child in shards {
            let status = child.status
            if counts[status] == nil { order.append(status) }
            counts[status, default: 0] += 1
        }
        let parts = order.map { "\(counts[$0] ?? 0) \($0)" }
        let n = shards.count
        return "\(n) shard\(n == 1 ? "" : "s"): " + parts.joined(separator: " · ")
    }

    /// A shard chip's label: "shard 1/3 · running" (1-based for humans; the
    /// wire index is 0-based).
    public static func chipLabel(child: RemoteJobRecord, position: Int) -> String {
        let count = child.shardCount
        let index = (child.shardIndex).map { $0 + 1 } ?? (position + 1)
        let of = count.map { "/\($0)" } ?? ""
        if isJudgeWorker(child) {
            return "judge worker · \(RemoteJobStatusClass.displayText(for: child.status))"
        }
        if child.shardIndex == nil && child.shardCount == nil
            && child.parentJobID != nil
        {
            return "continuation · \(RemoteJobStatusClass.displayText(for: child.status))"
        }
        return "shard \(index)\(of) · \(RemoteJobStatusClass.displayText(for: child.status))"
    }
}
