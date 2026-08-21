import Testing
@testable import SteeringKit

/// Fixtures: two clusters separated along the first axis, with small
/// deterministic jitter on the second axis.
private func makeCluster(center: Float, count: Int) -> [[Float]] {
    (0 ..< count).map { index in
        [center + Float(index % 3) * 0.01, Float(index % 5) * 0.02 - 0.04, 0, 0]
    }
}

@Suite struct ConceptStatsTests {

    @Test func heldOutAccuracyIsPerfectOnSeparatedClusters() {
        let positive = makeCluster(center: 1, count: 10)
        let negative = makeCluster(center: -1, count: 10)
        let heldOut = ConceptStats.heldOutAccuracy(positive: positive, negative: negative)
        #expect(heldOut != nil)
        #expect(heldOut?.accuracy == 1.0)
        #expect(heldOut?.testCount == 4)  // every 5th of 10, both classes
    }

    @Test func heldOutNilWhenTooFewStimuli() {
        let positive = makeCluster(center: 1, count: 3)
        let negative = makeCluster(center: -1, count: 3)
        #expect(ConceptStats.heldOutAccuracy(positive: positive, negative: negative) == nil)
    }

    @Test func splitHalfNearOneOnConsistentClusters() throws {
        let positive = makeCluster(center: 1, count: 10)
        let negative = makeCluster(center: -1, count: 10)
        let cosine = try #require(
            ConceptStats.splitHalfCosine(positive: positive, negative: negative))
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
