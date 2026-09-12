import Testing
@testable import ExperimentKit

@Suite struct FittingRoundReviewTests {
    private let queue: JSONValue = .object([
        "planSHA256": .string("queue-review"),
        "capacity": .object(["summary": .string("Two slots available")]),
        "shards": .array([.object(["index": .number(0), "status": .string("pending"), "gpuType": .string("H100")])]),
    ])

    @Test func selectionChangesAndActionsRetireApprovalButPreserveTheReviewedContext() {
        var review = FittingRoundReview()
        review.record(queue, jobID: "round-one", kind: .queue)
        #expect(review.hash(for: .queue) == "queue-review")
        #expect(review.hash(for: .merge) == nil)
        #expect(!review.confirmed)
        review.confirmed = true
        review.invalidate()
        #expect(review.planSHA256 == nil)
        #expect(!review.confirmed)
        #expect(review.snapshot?.jobID == "round-one")
        #expect(review.snapshot?.document == queue)
        #expect(review.snapshot?.lines == ["Two slots available", "Shard 0: pending — H100"])
        #expect(review.snapshot?.label.contains("not live status") == true)
        // Repeated errors or invalidation must not erase the last review either.
        review.invalidate()
        #expect(review.snapshot?.document == queue)
    }

    @Test func switchingJobsClearsTheSnapshotAndApproval() {
        var review = FittingRoundReview()
        review.record(queue, jobID: "round-one", kind: .queue)
        review.confirmed = true
        review.changeJob()
        #expect(review.snapshot == nil)
        #expect(review.planSHA256 == nil)
        #expect(!review.confirmed)
    }

    @Test func mergeReviewReplacesQueueApprovalAndRequiresNewConfirmation() {
        var review = FittingRoundReview()
        review.record(queue, jobID: "round-one", kind: .queue)
        review.confirmed = true
        let merge: JSONValue = .object(["planSHA256": .string("merge-review")])
        review.record(merge, jobID: "round-one", kind: .merge)
        #expect(review.hash(for: .queue) == nil)
        #expect(review.hash(for: .merge) == "merge-review")
        #expect(!review.confirmed)
        #expect(review.snapshot?.document == merge)
        #expect(review.snapshot?.label.contains("merge plan for round round-one") == true)
    }

    @Test func reviewWithoutHashCannotReuseAnEarlierApproval() {
        var review = FittingRoundReview()
        review.record(queue, jobID: "round-one", kind: .queue)
        review.confirmed = true
        review.record(.object([:]), jobID: "round-one", kind: .queue)
        #expect(review.planSHA256 == nil)
        #expect(!review.confirmed)
        review.record(.object(["planSHA256": .string("")]), jobID: "round-one", kind: .queue)
        #expect(review.planSHA256 == nil)
    }
}
