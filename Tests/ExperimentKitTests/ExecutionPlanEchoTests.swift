import Foundation
import Testing

@testable import ExperimentKit

/// Finding 11c: the submit control must state what it will actually do.
/// The failure being fixed is concrete — a declared `[extract, validate,
/// sweep]` chain submitted as a plain `run`, with no refusal possible
/// because both states are individually legal (2026-07-26).
struct ExecutionPlanEchoTests {

    private func plan(
        verb: String,
        target: SubstrateRouting.Substrate = .server,
        dryRun: Bool = false,
        stages: [String]? = nil
    ) -> ExecutionPlanEcho.Plan {
        ExecutionPlanEcho.describe(
            verb: verb, target: target, serverLabel: "example-hpc",
            dryRun: dryRun, declaredPipelineStages: stages)
    }

    @Test("a declared chain bypassed by a single-verb submission is named")
    func mismatchNamesTheIgnoredStages() {
        let result = plan(verb: "run", stages: ["extract", "validate", "sweep"])
        let mismatch = try! #require(result.mismatch)
        #expect(mismatch.contains("extract → validate → sweep"))
        #expect(mismatch.contains("'run'"))
        // The remedy must be named, not merely the problem.
        #expect(mismatch.contains("pipeline"))
        #expect(result.summary.contains("this verb only"))
    }

    @Test("no pipeline declared: a single verb is exactly what was asked for")
    func noChainMeansNoMismatch() {
        #expect(plan(verb: "run", stages: nil).mismatch == nil)
        #expect(plan(verb: "sweep", stages: []).mismatch == nil)
    }

    @Test("a chain of exactly the submitted verb is not a mismatch")
    func singleStageChainMatchingTheVerbIsSilent() {
        #expect(plan(verb: "sweep", stages: ["sweep"]).mismatch == nil)
        // ...but a one-stage chain naming a DIFFERENT verb still is.
        #expect(plan(verb: "run", stages: ["sweep"]).mismatch != nil)
    }

    @Test("the pipeline verb echoes the stage list in execution order")
    func pipelineEchoesStages() {
        let result = plan(verb: "pipeline", stages: ["extract", "validate", "sweep"])
        #expect(result.mismatch == nil)
        #expect(result.summary.contains("extract → validate → sweep"))
        #expect(result.summary.contains("3 stages"))
        #expect(result.summary.contains("one model load"))
    }

    @Test("the pipeline verb with no declared chain warns before submission")
    func pipelineWithoutAChainWarns() {
        // The server refuses this; saying so here saves a round trip to the
        // cluster to learn it.
        let mismatch = try! #require(plan(verb: "pipeline", stages: []).mismatch)
        #expect(mismatch.contains("no pipeline block"))
    }

    @Test("a dry run says nothing executes")
    func dryRunIsStated() {
        let result = plan(verb: "run", dryRun: true)
        #expect(result.summary.contains("NOTHING executes"))
        #expect(result.summary.contains("example-hpc"))
    }

    @Test("a local run does not describe a remote verb it will not use")
    func localRunIgnoresTheVerb() {
        // The verb picker is a remote control; echoing it for a local run
        // would reproduce the same confusion in mirror image.
        let result = plan(verb: "pipeline", target: .thisMac, stages: ["extract"])
        #expect(result.mismatch == nil)
        #expect(result.summary.contains("this Mac"))
        #expect(!result.summary.contains("extract"))
    }
}
