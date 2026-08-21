import Testing
@testable import SteeringKit

/// Numerical tests against tiny inline fixtures, not live model output
/// (CLAUDE.md › Conventions).
@Suite struct VectorMathTests {

    @Test func meanOfRows() throws {
        let result = try SteeringVectorMath.mean([[1, 2], [3, 4]])
        #expect(result == [2, 3])
    }

    @Test func meanDifferenceIsCAADirection() throws {
        let positive: [[Float]] = [[2, 4], [4, 6]]   // mean [3, 5]
        let negative: [[Float]] = [[1, 1], [1, 3]]   // mean [1, 2]
        let v = try SteeringVectorMath.meanDifference(positive: positive, negative: negative)
        #expect(v == [2, 3])
    }

    @Test func grandMeanDifferenceUsesPopulationCenter() throws {
        let joy: [[Float]] = [[4, 2], [6, 2]]
        let anger: [[Float]] = [[0, 2], [2, 2]]
        let all = joy + anger
        let v = try SteeringVectorMath.grandMeanDifference(concept: joy, population: all)
        // joy mean [5, 2], grand mean [3, 2]
        #expect(v == [2, 0])
    }

    @Test func l2NormOfThreeFourFive() {
        #expect(SteeringVectorMath.l2Norm([3, 4]) == 5)
    }

    @Test func cosineSimilarityOrthogonalAndParallel() throws {
        #expect(try SteeringVectorMath.cosineSimilarity([1, 0], [0, 1]) == 0)
        #expect(try SteeringVectorMath.cosineSimilarity([1, 1], [2, 2]).isApproximatelyOne)
    }

    @Test func rescaledMatchesTargetNorm() throws {
        let v = try SteeringVectorMath.rescaled([3, 4], toNorm: 10)
        #expect(abs(SteeringVectorMath.l2Norm(v) - 10) < 1e-5)
    }

    /// GOLDEN: the seeded random-control vector is pinned component-by-
    /// component so the algorithm (`gaussian-isotropic-v1`: Box–Muller over
    /// SplitMix64, single rescale to the target norm) can never silently
    /// change again — matched-norm random controls must be regenerable from
    /// the recorded seed, forever. If this fails you changed the recipe:
    /// that requires a NEW `randomVectorAlgorithm` identifier, new goldens,
    /// and a doc note — old runs' controls are otherwise irreproducible.
    @Test func randomVectorGoldenValuesForSeed42() throws {
        let golden: [Float] = [
            0.2664175, 0.4192848, -0.5729510,
            0.8523627, 1.1110967, -1.2099137,
        ]
        var rng = SplitMix64(seed: 42)
        let v = try SteeringVectorMath.randomVector(
            dimension: 6, norm: 2, using: &rng)
        #expect(v.count == golden.count)
        for (got, want) in zip(v, golden) {
            #expect(abs(got - want) < 1e-6, "\(v) != \(golden)")
        }
        #expect(abs(SteeringVectorMath.l2Norm(v) - 2) < 1e-5)
        // Determinism contract: same seed → same vector.
        var rng2 = SplitMix64(seed: 42)
        let again = try SteeringVectorMath.randomVector(
            dimension: 6, norm: 2, using: &rng2)
        #expect(v == again)
        #expect(SteeringVectorMath.randomVectorAlgorithm == "gaussian-isotropic-v1")
    }

    /// Distribution sanity (seeded, deterministic): the control direction is
    /// an isotropic GAUSSIAN, not the legacy cube-uniform. Discriminators at
    /// dimension 4096: coordinate kurtosis ≈ 3 for a Gaussian vs ≈ 1.8 for
    /// uniform[-1,1] (rescaling is per-coordinate linear, so kurtosis is
    /// invariant to it), and max|coord|·√n ≈ 3.4 here while cube-uniform-
    /// then-rescale is capped near √3 ≈ 1.73 (no coordinate exceeds 1/‖raw‖
    /// with ‖raw‖ ≈ √(n/3) — the cube-corner bias the Gaussian removes).
    @Test func randomVectorIsGaussianNotCubeUniform() throws {
        var rng = SplitMix64(seed: 7)
        let n = 4096
        let v = try SteeringVectorMath.randomVector(
            dimension: n, norm: 1, using: &rng)
        let mean = v.reduce(0, +) / Float(n)
        #expect(abs(mean) < 1e-3, "direction should be mean-centered")
        let variance = v.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(n)
        let m4 = v.map { value in
            let c = value - mean
            return c * c * c * c
        }.reduce(0, +) / Float(n)
        let kurtosis = m4 / (variance * variance)
        #expect(kurtosis > 2.5, "cube-uniform coordinates have kurtosis ≈ 1.8, got \(kurtosis)")
        let maxAbsScaled = (v.map { abs($0) }.max() ?? 0) * Float(n).squareRoot()
        #expect(maxAbsScaled > 2.0, "cube-uniform caps max|coord|·√n near √3, got \(maxAbsScaled)")
    }

    /// The norm-unit invariant: the injected perturbation `scale·v` must
    /// have L2 norm alpha × residualNorm REGARDLESS of ‖v‖ — vector norms
    /// vary by layer, method, and stimulus count, and must not leak into
    /// the dose. (The audit-flagged bug: scaling by alpha·residualNorm
    /// alone made effective strength proportional to ‖v‖.)
    @Test func normUnitScaleIsInvariantToVectorNorm() throws {
        let alpha: Float = 0.05
        let residualNorm: Float = 12.5
        for vector in [[3, 4], [300, 400], [0.003, 0.004]] as [[Float]] {
            let scale = try SteeringVectorMath.normUnitScale(
                alpha: alpha, residualNorm: residualNorm,
                vectorNorm: SteeringVectorMath.l2Norm(vector))
            let injectedNorm = SteeringVectorMath.l2Norm(vector.map { $0 * scale })
            #expect(abs(injectedNorm - alpha * residualNorm) < 1e-4)
        }
    }

    @Test func normUnitScaleThrowsOnDegenerateNorms() {
        #expect(throws: SteeringVectorError.degenerateData) {
            try SteeringVectorMath.normUnitScale(
                alpha: 0.1, residualNorm: 10, vectorNorm: 0)
        }
        #expect(throws: SteeringVectorError.degenerateData) {
            try SteeringVectorMath.normUnitScale(
                alpha: 0.1, residualNorm: 0, vectorNorm: 1)
        }
    }

    @Test func dimensionMismatchThrows() {
        #expect(throws: SteeringVectorError.dimensionMismatch(expected: 2, found: 3)) {
            try SteeringVectorMath.mean([[1, 2], [1, 2, 3]])
        }
    }

    @Test func emptyInputThrows() {
        #expect(throws: SteeringVectorError.emptyInput) {
            try SteeringVectorMath.mean([])
        }
    }

    @Test func scalarProbeOrientsPositiveScoresAboveZero() throws {
        let direction: [Float] = [-1, 0]  // deliberately backwards
        let positive: [[Float]] = [[3, 0], [4, 0]]
        let negative: [[Float]] = [[0, 0], [1, 0]]
        let probe = try SteeringVectorMath.scalarProbe(
            direction: direction, positive: positive, negative: negative)

        #expect(try probe.score([4, 0]) > 0)
        #expect(try probe.score([0, 0]) < 0)
        #expect(try probe.classifiesPositive([3, 0]))
    }
}

extension Float {
    fileprivate var isApproximatelyOne: Bool { abs(self - 1) < 1e-6 }
}
