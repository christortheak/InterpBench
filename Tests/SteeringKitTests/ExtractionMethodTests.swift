import Testing
@testable import SteeringKit

@Suite struct ExtractionMethodTests {

    /// Rows spread along the first axis with small jitter on the second:
    /// PC1 must align with the first axis (up to sign).
    @Test func firstPrincipalComponentFindsDominantAxis() throws {
        let rows: [[Float]] = (0 ..< 12).map { index in
            [Float(index) - 5.5, Float(index % 3) * 0.05, 0, 0]
        }
        let pc = try SteeringVectorMath.firstPrincipalComponent(of: rows)
        #expect(abs(SteeringVectorMath.l2Norm(pc) - 1) < 1e-4)
        #expect(abs(pc[0]) > 0.99)
    }

    @Test func principalComponentRejectsTooFewRows() {
        #expect(throws: SteeringVectorError.self) {
            try SteeringVectorMath.firstPrincipalComponent(of: [[1, 2, 3]])
        }
    }

    @Test func principalComponentsReportExplainedVariance() throws {
        let rows: [[Float]] = [
            [-3, 0], [-2, 0], [-1, 0], [1, 0], [2, 0], [3, 0],
        ]
        let pcs = try SteeringVectorMath.principalComponents(
            of: rows, minimumExplainedVariance: 0.5)
        #expect(pcs.components.count == 1)
        #expect(pcs.totalExplainedVariance > 0.99)
        #expect(abs(pcs.components[0][0]) > 0.99)
    }

    /// LAT direction must be sign-aligned with the mean difference and
    /// norm-matched to it, so alpha semantics stay comparable.
    @Test func latIsSignAlignedAndNormMatched() throws {
        // Pair differences all point near axis 0 with varying magnitude and
        // small off-axis jitter — the realistic shape (concept direction +
        // noise).
        let positive: [[Float]] = (0 ..< 8).map { i in
            [10 + Float(i % 4) * 2, Float(i % 3) * 0.2, 0]
        }
        let negative: [[Float]] = (0 ..< 8).map { i in
            [Float(i % 5) * 0.3, Float(i % 3) * 0.2 - 0.05, 0]
        }

        let meanDiff = try SteeringVectorMath.direction(
            positive: positive, negative: negative, method: .meanDifference)
        let lat = try SteeringVectorMath.direction(
            positive: positive, negative: negative, method: .lat)

        #expect(SteeringVectorMath.dot(lat, meanDiff) > 0)
        #expect(
            abs(SteeringVectorMath.l2Norm(lat) - SteeringVectorMath.l2Norm(meanDiff)) < 1e-3)
        let cosine = try SteeringVectorMath.cosineSimilarity(lat, meanDiff)
        #expect(cosine > 0.9)  // same dominant axis on this clean fixture
    }

    /// RepE C.1 normalizes each pair difference before PCA, so one
    /// huge-norm pair cannot dominate PC1. Five unit pairs along axis B vs
    /// one 100× pair along axis A: raw PCA would return A; normalized LAT
    /// must return B.
    @Test func latIsNotDominatedByHighNormPairs() throws {
        var positive: [[Float]] = (0 ..< 5).map { i in
            [0, 1, Float(i % 2) * 0.05]  // axis B (unit norm)
        }
        var negative: [[Float]] = (0 ..< 5).map { _ in [0, 0, 0] }
        positive.append([100, 0, 0])  // axis A, 100× the norm
        negative.append([0, 0, 0])

        let lat = try SteeringVectorMath.direction(
            positive: positive, negative: negative, method: .lat)
        let axisA: [Float] = [1, 0, 0]
        let axisB: [Float] = [0, 1, 0]
        #expect(
            abs(try SteeringVectorMath.cosineSimilarity(lat, axisB))
                > abs(try SteeringVectorMath.cosineSimilarity(lat, axisA)))
    }

    /// Identical pair differences all normalize to the same unit vector;
    /// with the ± orientation symmetry, PC1 recovers exactly that shared
    /// direction — LAT degrades gracefully to the mean-difference direction
    /// rather than throwing.
    @Test func latOnIdenticalDifferencesRecoversTheSharedDirection() throws {
        let positive: [[Float]] = (0 ..< 6).map { _ in [10, 0, 0] }
        let negative: [[Float]] = (0 ..< 6).map { _ in [0, 0, 0] }
        let lat = try SteeringVectorMath.direction(
            positive: positive, negative: negative, method: .lat)
        let meanDiff = try SteeringVectorMath.direction(
            positive: positive, negative: negative, method: .meanDifference)
        #expect(try SteeringVectorMath.cosineSimilarity(lat, meanDiff) > 0.999)
    }

    /// Zero differences (positive == negative) cannot be normalized; with
    /// fewer than two usable pairs LAT must throw rather than return noise.
    @Test func latThrowsWhenAllDifferencesAreZero() {
        let positive: [[Float]] = (0 ..< 4).map { _ in [1, 2, 3] }
        let negative: [[Float]] = (0 ..< 4).map { _ in [1, 2, 3] }
        #expect(throws: SteeringVectorError.degenerateData) {
            try SteeringVectorMath.direction(
                positive: positive, negative: negative, method: .lat)
        }
    }

    // MARK: - Cross-engine method vocabulary

    /// Every raw value is a CONTRACT with the server's `ExtractionMethod`
    /// (and with sidecars already on disk): a member Swift does not know is
    /// an attach refusal on the Mac for an artifact the server accepts —
    /// which is exactly what `gemmaScopeSAE` was until it was added here.
    @Test func rawValuesMatchTheServersVocabulary() {
        #expect(
            Set(ExtractionMethod.allCases.map(\.rawValue))
                == [
                    "meanDifference", "lat", "emotionGrandMean",
                    "designatedReference", "pinnedArtifact", "optvec",
                    "gemmaScopeSAE",
                ])
    }

    /// A Gemma Scope decoder row is a coordinate in a published dictionary:
    /// no stimuli, no source concept, no held-out validation.jsonl, and
    /// never an attachable RECIPE — it enters only as a pinned artifact's
    /// SOURCE method. Server twin: `has_source_concept` / the
    /// `_GEMMA_SCOPE_METHOD` attach refusal.
    @Test func gemmaScopeSAEAnswersTheLifecycleQuestionsLikeOptvec() {
        let sae = ExtractionMethod.gemmaScopeSAE
        #expect(ExtractionMethod(rawValue: "gemmaScopeSAE") == sae)
        #expect(sae.isGemmaScopeSAE)
        #expect(!sae.hasSourceConcept)
        #expect(!sae.isPaired)
        #expect(!sae.isRecipeMethod)
        #expect(!sae.isPinnedArtifact)
        #expect(!sae.isOptvec)
        #expect(!sae.isGrandMean)
        #expect(!sae.usesStoryCorpus)
        #expect(!sae.usesContrastiveValidation)
        // The only two methods with no source concept.
        #expect(
            ExtractionMethod.allCases.filter { !$0.hasSourceConcept }
                == [.optvec, .gemmaScopeSAE])
        // …and it never reaches the direction math.
        #expect(throws: SteeringVectorError.self) {
            try SteeringVectorMath.direction(
                positive: [[1, 0]], negative: [[0, 0]], method: sae)
        }
    }

    @Test func latRequiresPairedStimuli() {
        let positive: [[Float]] = [[1, 0], [2, 0], [3, 0]]
        let negative: [[Float]] = [[0, 0], [0, 1]]
        #expect(throws: SteeringVectorError.self) {
            try SteeringVectorMath.direction(
                positive: positive, negative: negative, method: .lat)
        }
    }
}
