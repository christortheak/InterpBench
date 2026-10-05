import Foundation
import Testing
@testable import ExperimentKit

@Suite struct FittingReviewSummaryTests {
    private func draft(_ json: String) throws -> JSONValue {
        try JSONDecoder().decode(JSONValue.self, from: Data(json.utf8))
    }

    @Test func distinguishesGlobalBudgetFromPerShardAndUnknownStorage() throws {
        let value = try draft(#"{"operationReview":{"globalRowBudget":8,"shards":[{},{}],"minimumLensAndCheckpointBytes":null}}"#)
        let lines = FittingReviewSummary.lines(operation: "jlens-fit-round", draft: value)
        #expect(lines.contains { $0.contains("8 total corpus rows") && $0.contains("2 shards") })
        #expect(lines.contains { $0.contains("Storage estimate unavailable") })
        #expect(lines.contains { $0.contains("starts no GPU fitting") })
    }

    @Test func partialCoverageAndPilotLimitationsStayVisible() throws {
        let value = try draft(#"{"fittingReview":{"pilotMeasurement":{"rowsPerHour":2,"extrapolatedHoursAtRowCap":4}},"operationReview":{"promptsFitted":3,"rowsConsidered":4,"partial":true,"missingRows":[4,5]}}"#)
        let lines = FittingReviewSummary.lines(operation: "jlens-fit-merge", draft: value)
        #expect(lines.contains { $0.contains("Measured pilot:") })
        #expect(lines.contains { $0.contains("Row lengths, hardware, and contention") })
        #expect(lines.contains { $0.contains("Partial merge: 2 planned rows") })
    }

    @Test func benchmarkBudgetAndAssessmentBoundaryUseOwnerValues() throws {
        let benchmark = try draft(#"{"operationReview":{"totalRowBudget":12,"cases":[{},{},{}]}}"#)
        #expect(FittingReviewSummary.lines(operation: "jlens-fit-benchmark", draft: benchmark).first?.contains("12 row evaluations across 3 benchmark cases") == true)
        let assessment = try draft(#"{"operationReview":{"rows":4,"sourceLayers":[0,1]}}"#)
        let lines = FittingReviewSummary.lines(operation: "jlens-fit-assess", draft: assessment)
        #expect(lines.first?.contains("4 corpus rows across 2 source layers") == true)
        #expect(lines.last?.contains("does not prove") == true)
    }

    @Test func assessmentListFormNamesComparisonsAndMaximumBudget() throws {
        let value = try draft(#"{"operationReview":{"rows":4,"sourceLayers":[0,1],"comparisons":6,"candidateLensIDs":["a","b"],"corpora":[{"path":"x","sha256":"0","rows":4},{"path":"y","sha256":"1","rows":2},{"path":"z","sha256":"2","rows":3}]}}"#)
        let lines = FittingReviewSummary.lines(operation: "jlens-fit-assess", draft: value)
        #expect(lines.contains { $0.contains("2 candidate lenses on 3 held-out corpora: 6 comparisons") && $0.contains("not a sum") })
        let single = try draft(#"{"operationReview":{"rows":4,"sourceLayers":[0,1]}}"#)
        #expect(!FittingReviewSummary.lines(operation: "jlens-fit-assess", draft: single).contains { $0.contains("comparisons in one job") })
        let workflow = try #require(ScienceCatalog.workflows().first { $0.id == "jlens-fit-assess" })
        for id in ["candidateLensIDs", "corpora", "candidateLensID", "corpus"] {
            #expect(workflow.fields.first { $0.id == id }?.required == false)
        }
    }

    @Test func assessmentStorageIsSeparateFromPeakMemory() throws {
        let value = try draft(#"{"operationReview":{"rows":16,"sourceLayers":[0,1],"resources":{"temporaryActivationBytesUpperBound":1073741824,"float32LensPairBytes":536870912}}}"#)
        let lines = FittingReviewSummary.lines(operation: "jlens-fit-assess", draft: value)
        #expect(lines.contains { $0.contains("1.00 GiB of temporary tensor storage") && $0.contains("0.50 GiB at float32") })
        #expect(lines.contains { $0.contains("not peak memory") })
    }

    @Test func assessmentReadoutExplainsAdditionalStorageAndBaseline() throws {
        let value = try draft(#"{"operationReview":{"rows":4,"sourceLayers":[0],"readoutReview":{"requested":"float32","summary":"Paired native and float32; additional output-head storage."}}}"#)
        let lines = FittingReviewSummary.lines(operation: "jlens-fit-assess", draft: value)
        #expect(lines.contains { $0.contains("plain-residual") })
        #expect(lines.contains { $0.contains("additional output-head storage") })
        let field = try #require(ScienceCatalog.workflows().first { $0.id == "jlens-fit-assess" }?.fields.first { $0.id == "readoutDtype" })
        #expect(!field.required)
        #expect(field.help.contains("float32"))
    }

    @Test func costMeasurementShowsItsWorkloadAndAdvisoriesBeforeAnythingRuns() throws {
        // Two planned configurations, two views, two prompts, and one warm-up plus three measured rounds.
        let value = try draft(#"{"operationReview":{"generations":32,"prompts":2,"warmupRounds":1,"measuredRounds":3,"generatedTokenBudget":192,"configurations":[{"id":"baseline","status":"planned"},{"id":"probeReadings","status":"planned"},{"id":"fixedPolicy","status":"notRequested","reason":"No fixedPolicy file was given."}],"advisories":["The policy given as conditional declares no rule that depends on a probe score."]}}"#)
        let lines = FittingReviewSummary.lines(operation: "instrumentation-cost", draft: value)
        #expect(lines.first?.contains("32 responses: 2 of 6 configurations on 2 prompts") == true)
        #expect(lines.first?.contains("1 warm-up and 3 measured rounds") == true)
        #expect(lines.contains { $0.contains("192 generated tokens") && $0.contains("nothing is downloaded") })
        #expect(lines.contains { $0.contains("declares no rule that depends on a probe score") })
        #expect(lines.last?.contains("sets no target") == true)
        // A review without the owner's numbers says so instead of showing zero.
        let empty = FittingReviewSummary.lines(operation: "instrumentation-cost", draft: try draft(#"{"operationReview":{}}"#))
        #expect(empty.first?.contains("unavailable responses: unavailable of 6 configurations") == true)
        let workflow = try #require(ScienceCatalog.workflows().first { $0.id == "instrumentation-cost" })
        #expect(workflow.fields.filter(\.required).map(\.id) == ["modelID", "revision", "prompts"])
        #expect(Set(workflow.fields.map(\.kind)).isSubset(of: ["text", "integer", "number", "boolean", "fileRef", "fileRefs", "texts"]))
    }

    @Test func queueCapacityExplainsOtherJobsUsingOwnerSummary() throws {
        let value = try draft(#"{"capacity":{"summary":"Two jobs occupy the controller capacity.","activeJobs":[{"jobID":"first","status":"running","belongsToThisRound":true},{"jobID":"second","status":"submitted","belongsToThisRound":false}]}}"#)
        #expect(FittingReviewSummary.capacityLines(value) == [
            "Two jobs occupy the controller capacity.",
            "first: running (this round).", "second: submitted (another scientific task).",
        ])
        #expect(FittingReviewSummary.capacityLines(.object([:])).isEmpty)
    }
}
