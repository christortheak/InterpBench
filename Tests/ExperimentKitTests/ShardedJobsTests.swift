import Foundation
import Testing

@testable import ExperimentKit

/// Multi-GPU sharded submissions (2026-07-22): the app-side pure rules —
/// when `parallelJobs` is encoded, how the transcript stamps it, the
/// stepper cap, the site-profile field's round trip, shard fields on job
/// records, the display grouping, and the "merging" status class.
struct ShardedJobsTests {

    // MARK: Encoding rule

    @Test func parallelJobsEncodesOnlyForShardingSlurmSubmissions() {
        // The happy path: slurm + run + K > 1.
        #expect(
            ShardedSubmission.encodedParallelJobs(
                requested: 4, executor: "slurm", verb: "run") == 4)
        // Pipeline is shardable client-side (the server rules on its stages).
        #expect(
            ShardedSubmission.encodedParallelJobs(
                requested: 2, executor: "slurm", verb: "pipeline") == 2)
        // 1 (or nonsense) is never encoded — older servers must not see it.
        #expect(
            ShardedSubmission.encodedParallelJobs(
                requested: 1, executor: "slurm", verb: "run") == nil)
        #expect(
            ShardedSubmission.encodedParallelJobs(
                requested: 0, executor: "slurm", verb: "run") == nil)
        // Local executor and non-sharding verbs never encode.
        #expect(
            ShardedSubmission.encodedParallelJobs(
                requested: 4, executor: "local", verb: "run") == nil)
        #expect(
            ShardedSubmission.encodedParallelJobs(
                requested: 4, executor: "slurm", verb: "validate") == nil)
    }

    @Test func transcriptStampDerivesFromTheServersShardResponse() {
        // Finding 5 (2026-07-22): the stamp names what the server actually
        // DID (returned shard ids), never what the request asked for.
        #expect(
            ShardedSubmission.transcriptStamp(shardJobIDs: ["a", "b", "c"])
                == "sharded across 3 GPU jobs")
        // The server ignored the fan-out (non-run-first pipeline, older
        // server, unsharded submission): no stamp, whatever was requested.
        #expect(ShardedSubmission.transcriptStamp(shardJobIDs: nil) == nil)
        #expect(ShardedSubmission.transcriptStamp(shardJobIDs: []) == nil)
        #expect(ShardedSubmission.transcriptStamp(shardJobIDs: ["only"]) == nil)
    }

    // MARK: Sharding availability (stepper disable rule)

    @Test func shardingUnavailableReasonMirrorsTheServerRule() {
        // Slurm run always shards.
        #expect(
            ShardedSubmission.shardingUnavailableReason(
                verb: "run", executor: "slurm", declaredPipelineStages: nil)
                == nil)
        // A run-first pipeline shards.
        #expect(
            ShardedSubmission.shardingUnavailableReason(
                verb: "pipeline", executor: "slurm",
                declaredPipelineStages: ["run", "evaluate", "analyze"]) == nil)
        // A chain starting with extract does NOT — and the reason names the
        // actual first stage.
        let extractFirst = ShardedSubmission.shardingUnavailableReason(
            verb: "pipeline", executor: "slurm",
            declaredPipelineStages: ["extract", "validate", "run"])
        #expect(extractFirst?.contains("starts with 'extract'") == true)
        #expect(extractFirst?.contains("only run-first chains shard") == true)
        // No declared stages at all: unavailable, honestly worded.
        #expect(
            ShardedSubmission.shardingUnavailableReason(
                verb: "pipeline", executor: "slurm",
                declaredPipelineStages: nil)?
                .contains("declares no pipeline stages") == true)
        // Non-sharding verbs and non-slurm executors.
        #expect(
            ShardedSubmission.shardingUnavailableReason(
                verb: "validate", executor: "slurm",
                declaredPipelineStages: nil)?
                .contains("does not shard") == true)
        #expect(
            ShardedSubmission.shardingUnavailableReason(
                verb: "run", executor: "local", declaredPipelineStages: nil)?
                .contains("only Slurm") == true)
    }

    @Test func declaredPipelineStagesMirrorsTheServerDefaultChain() {
        // No pipeline block: nil (the pipeline verb refuses anyway).
        #expect(ShardedSubmission.declaredPipelineStages(nil) == nil)
        // A block with declared stages: those names.
        let declared = ShardedSubmission.declaredPipelineStages(
            .object(["stages": .array([.string("run"), .string("analyze")])]))
        #expect(declared == ["run", "analyze"])
        // A block with no/empty stages resolves to the server's DEFAULT
        // chain, which starts with extract — so it does not shard.
        let defaulted = ShardedSubmission.declaredPipelineStages(.object([:]))
        #expect(defaulted?.first == "extract")
        #expect(
            ShardedSubmission.declaredPipelineStages(
                .object(["stages": .array([])]))?.first == "extract")
    }

    @Test func stepperCapUsesSiteMaxElseDefault() {
        #expect(ShardedSubmission.stepperCap(siteMax: nil)
            == ShardedSubmission.defaultStepperCap)
        #expect(ShardedSubmission.stepperCap(siteMax: 8) == 8)
        // A degenerate site value never collapses the stepper below 1.
        #expect(ShardedSubmission.stepperCap(siteMax: 0) == 1)
    }

    // MARK: Site profile field

    @Test func maxParallelGPUJobsRoundTripsAndDecodesLeniently() throws {
        var profile = ClusterSiteProfile.genericSlurm
        guard case .slurm(var slurm) = profile.scheduler else {
            Issue.record("generic preset should be slurm")
            return
        }
        slurm.maxParallelGPUJobs = 6
        profile.scheduler = .slurm(slurm)
        let decoded = try ClusterSiteProfile.decode(from: profile.encoded())
        guard case .slurm(let round) = decoded.scheduler else {
            Issue.record("decode lost the scheduler")
            return
        }
        #expect(round.maxParallelGPUJobs == 6)

        // Lenient decode: an older profile without the field decodes nil.
        let legacy = try ClusterSiteProfile.decode(
            from: ClusterSiteProfile.exampleCluster.encoded())
        guard case .slurm(let slurmLegacy) = legacy.scheduler else {
            Issue.record("preset should be slurm")
            return
        }
        #expect(slurmLegacy.maxParallelGPUJobs == nil)
    }

    @Test @MainActor func siteEditorModelRoundTripsTheCap() {
        var seed = ClusterSiteProfile.genericSlurm
        guard case .slurm(var slurm) = seed.scheduler else { return }
        slurm.maxParallelGPUJobs = 12
        seed.scheduler = .slurm(slurm)
        let model = SiteEditorModel(profile: seed)
        #expect(model.maxParallelGPUJobsText == "12")
        model.maxParallelGPUJobsText = " 5 "
        guard case .slurm(let built) = model.builtProfile().scheduler else {
            Issue.record("built profile lost the scheduler")
            return
        }
        #expect(built.maxParallelGPUJobs == 5)
        model.maxParallelGPUJobsText = ""
        guard case .slurm(let cleared) = model.builtProfile().scheduler else {
            return
        }
        #expect(cleared.maxParallelGPUJobs == nil)
    }

    // MARK: Job-record shard fields

    private func record(
        id: String, status: String = "running",
        result: [String: JSONValue]? = nil
    ) -> RemoteJobRecord {
        RemoteJobRecord(
            id: id, kind: "study-submit-bundle", status: status,
            createdAt: 0, startedAt: nil, finishedAt: nil, result: result,
            error: nil, logTail: [], executor: "slurm", executorJobID: nil,
            cancellationRequested: false)
    }

    @Test func shardFieldsReadFromResult() {
        let parent = record(
            id: "p",
            result: [
                "shardJobs": .array([.string("a"), .string("b")]),
                "runDirectory": .string("/runs/merged"),
            ])
        #expect(parent.shardJobIDs == ["a", "b"])
        #expect(parent.mergedRunDirectory == "/runs/merged")

        let child = record(
            id: "a",
            result: [
                "parentJob": .string("p"),
                "shard": .object(["index": .number(1), "count": .number(3)]),
            ])
        #expect(child.parentJobID == "p")
        #expect(child.shardIndex == 1)
        #expect(child.shardCount == 3)

        let plain = record(id: "x")
        #expect(plain.shardJobIDs == nil)
        #expect(plain.parentJobID == nil)
        #expect(plain.shardIndex == nil)
    }

    // MARK: Display grouping

    @Test func groupingNestsShardChildrenUnderTheirPresentParent() {
        let parent = record(
            id: "p",
            result: ["shardJobs": .array([.string("s0"), .string("s1")])])
        let shard0 = record(
            id: "s0", status: "succeeded",
            result: [
                "parentJob": .string("p"),
                "shard": .object(["index": .number(0), "count": .number(2)]),
            ])
        let shard1 = record(
            id: "s1", status: "running",
            result: [
                "parentJob": .string("p"),
                "shard": .object(["index": .number(1), "count": .number(2)]),
            ])
        let unrelated = record(id: "z")
        // An orphan child whose parent record is not in the list stays
        // visible at the top level — never hide a job.
        let orphan = record(
            id: "o", result: ["parentJob": .string("gone")])
        let jobs = [shard1, parent, unrelated, shard0, orphan]

        let top = ShardedJobGrouping.topLevel(jobs)
        #expect(top.map(\.id) == ["p", "z", "o"])

        let children = ShardedJobGrouping.children(of: parent, in: jobs)
        #expect(children.map(\.id) == ["s0", "s1"])  // shard order, not list order

        #expect(ShardedJobGrouping.aggregateLine(children: children)
            == "2 shards: 1 succeeded · 1 running")
        #expect(ShardedJobGrouping.chipLabel(child: shard1, position: 1)
            == "shard 2/2 · running")
        #expect(ShardedJobGrouping.aggregateLine(children: []) == nil)
    }

    @Test func groupingAppendsPipelineContinuationAfterShards() {
        let parent = record(
            id: "p", result: ["shardJobs": .array([.string("s0")])])
        let shard0 = record(
            id: "s0", status: "succeeded",
            result: [
                "parentJob": .string("p"),
                "shard": .object(["index": .number(0), "count": .number(1)]),
            ])
        let continuation = record(
            id: "c", status: "running",
            result: ["parentJob": .string("p")])
        let jobs = [continuation, shard0, parent]
        let children = ShardedJobGrouping.children(of: parent, in: jobs)
        #expect(children.map(\.id) == ["s0", "c"])
        #expect(ShardedJobGrouping.chipLabel(child: continuation, position: 1)
            == "continuation · running")
        #expect(ShardedJobGrouping.topLevel(jobs).map(\.id) == ["p"])
    }

    @Test func aggregateLineCountsShardsOnlyNeverTheContinuation() {
        // Display nit (2026-07-22): a 2-shard pipeline with its continuation
        // chip used to read "3 shards".
        let shard0 = record(
            id: "s0", status: "succeeded",
            result: [
                "parentJob": .string("p"),
                "shard": .object(["index": .number(0), "count": .number(2)]),
            ])
        let shard1 = record(
            id: "s1", status: "succeeded",
            result: [
                "parentJob": .string("p"),
                "shard": .object(["index": .number(1), "count": .number(2)]),
            ])
        let continuation = record(
            id: "c", status: "running",
            result: ["parentJob": .string("p")])
        #expect(ShardedJobGrouping.isContinuation(continuation))
        #expect(!ShardedJobGrouping.isContinuation(shard0))
        #expect(
            ShardedJobGrouping.aggregateLine(
                children: [shard0, shard1, continuation])
                == "2 shards: 2 succeeded")
        // A children list that is somehow ONLY the continuation draws no
        // shard line at all.
        #expect(ShardedJobGrouping.aggregateLine(children: [continuation]) == nil)
    }

    @Test func judgeWorkersLabelAsJudgeWorkersNeverContinuations() {
        // Judge fan-out (2026-07-23): worker siblings (kind
        // study-judge-worker) render as labeled judge-worker chips under
        // the parent, distinct from the pipeline continuation, and never
        // inflate the shard count.
        let parent = record(
            id: "p", result: ["shardJobs": .array([.string("s0")])])
        let shard0 = record(
            id: "s0", status: "succeeded",
            result: [
                "parentJob": .string("p"),
                "shard": .object(["index": .number(0), "count": .number(1)]),
            ])
        var worker = record(
            id: "w0", status: "running",
            result: ["parentJob": .string("p")])
        worker.kind = "study-judge-worker"
        #expect(ShardedJobGrouping.isJudgeWorker(worker))
        #expect(!ShardedJobGrouping.isContinuation(worker))
        #expect(ShardedJobGrouping.chipLabel(child: worker, position: 1)
            == "judge worker · running")
        let children = ShardedJobGrouping.children(
            of: parent, in: [worker, shard0, parent])
        #expect(children.map(\.id) == ["s0", "w0"])
        #expect(ShardedJobGrouping.aggregateLine(children: children)
            == "1 shard: 1 succeeded")
    }

    // MARK: Status class

    @Test func mergingClassifiesAsInFlightNeverFailed() {
        #expect(RemoteJobStatusClass.classify(status: "merging") == .inFlight)
        // And it never offers Resume (only checkpointed does).
        #expect(!RemoteJobStatusClass.offersResume(status: "merging"))
        #expect(RemoteJobStatusClass.offersResume(status: "checkpointed"))
    }
}
