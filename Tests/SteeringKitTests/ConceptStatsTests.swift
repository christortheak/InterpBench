import Testing
@testable import SteeringKit

/// Fixtures: two clusters separated along the first axis, with small
/// deterministic jitter on the second axis.
private func makeCluster(center: Float, count: Int) -> [[Float]] {
    (0 ..< count).map { index in
        [center + Float(index % 3) * 0.01, Float(index % 5) * 0.02 - 0.04, 0, 0]
    }
}

/// Row-aligned stimulus texts: the held-out and split-half splits are drawn
/// from the CONTENT (audit F3), so every call site needs them.
private func texts(_ prefix: String, _ count: Int) -> [String] {
    (0 ..< count).map { "\(prefix) stimulus \($0)" }
}

@Suite struct ConceptStatsTests {

    @Test func heldOutAccuracyIsPerfectOnSeparatedClusters() {
        let positive = makeCluster(center: 1, count: 10)
        let negative = makeCluster(center: -1, count: 10)
        let heldOut = ConceptStats.heldOutAccuracy(
            positive: positive, negative: negative,
            positiveTexts: texts("positive", 10), negativeTexts: texts("negative", 10))
        #expect(heldOut != nil)
        #expect(heldOut?.accuracy == 1.0)
        // Every 5th of the content-hash ORDER of 10, both classes — the count
        // the positional rule produced, now drawn from the texts (audit F3).
        #expect(heldOut?.testCount == 4)
    }

    @Test func heldOutNilWhenTooFewStimuli() {
        let positive = makeCluster(center: 1, count: 3)
        let negative = makeCluster(center: -1, count: 3)
        #expect(
            ConceptStats.heldOutAccuracy(
                positive: positive, negative: negative,
                positiveTexts: texts("positive", 3), negativeTexts: texts("negative", 3))
                == nil)
        // Five per class is below the shared floor too: a ~20% holdout of five
        // rows is one row, and its "accuracy" is 0% or 100% by coin flip.
        #expect(
            ConceptStats.heldOutAccuracy(
                positive: makeCluster(center: 1, count: 5),
                negative: makeCluster(center: -1, count: 5),
                positiveTexts: texts("positive", 5), negativeTexts: texts("negative", 5))
                == nil)
    }

    @Test func splitHalfNearOneOnConsistentClusters() throws {
        let positive = makeCluster(center: 1, count: 10)
        let negative = makeCluster(center: -1, count: 10)
        let cosine = try #require(
            ConceptStats.splitHalfCosine(
                positive: positive, negative: negative,
                positiveTexts: texts("positive", 10),
                negativeTexts: texts("negative", 10)))
        #expect(cosine > 0.99)
    }

    @Test func outlierFindsPlantedBadStimulus() throws {
        var positive = makeCluster(center: 1, count: 8)
        positive[5] = [-1.5, 0, 0, 0]  // sits on the negative side
        let negative = makeCluster(center: -1, count: 8)

        let direction = try SteeringVectorMath.meanDifference(
            positive: positive, negative: negative)
        let outliers = ConceptStats.outliers(
            direction: direction, positive: positive, negative: negative)

        let worst = try #require(outliers.first)
        #expect(worst.isPositive)
        #expect(worst.index == 5)
        #expect(worst.margin < 0)
    }

    @Test func dotProduct() {
        #expect(ConceptStats.dot([1, 2, 3], [4, 5, 6]) == 32)
    }
}
