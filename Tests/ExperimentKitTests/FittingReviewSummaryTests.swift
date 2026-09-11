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
}
