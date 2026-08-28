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

    /// 2026-08-28 audit, F2 (server twin:
    /// `test_principal_components_never_exceed_the_data_rank`). n centred
    /// rows span at most n−1 dimensions, so after n−1 deflations the residual
    /// is float32 round-off — whose own Gram trace is tiny but POSITIVE,
    /// which is exactly what the power iteration's RELATIVE degenerate-start
    /// floor accepts. The count branch therefore used to normalise rounding
    /// noise into unit "components", and the neutral-PC projection then
    /// removed the concept vector's component along those arbitrary
    /// directions.
    ///
    /// Clamped rather than refused: `neutralPCCount` is a study-level knob
    /// applied to whatever neutral corpus each concept has, so over-asking is
    /// an honest declaration, not an error.
    @Test func principalComponentsNeverExceedTheDataRank() throws {
        let rows: [[Float]] = [
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        ]
        let over = try SteeringVectorMath.principalComponentsWithVariance(
            of: rows, count: 6)
        #expect(over.components.count == 3)
        #expect(over.explainedVariance.count == 3)
        // Every component returned carries real variance — no float-noise
        // tail masquerading as a direction.
        #expect(over.explainedVariance.allSatisfy { $0 > 1e-6 })
        // A request within the rank is untouched.
        #expect(try SteeringVectorMath.principalComponents(of: rows, count: 3).count == 3)
        #expect(try SteeringVectorMath.principalComponents(of: rows, count: 2).count == 2)
    }

    // MARK: - F2 second pass: deflation stops at the EFFECTIVE rank

    /// Mathematically rank-one and deliberately NOT axis-aligned: every row is
    /// a multiple of one oblique direction, so no component past the first
    /// exists and no coordinate accidentally hides the residue.
    static let rankOneRows: [[Float]] = {
        let raw: [Float] = [0.3, -0.5, 0.2, 0.8]
        let norm = SteeringVectorMath.l2Norm(raw)
        let unit = raw.map { $0 / norm }
        let scales: [Float] = [1, 2, 3.5, -1]
        return scales.map { scale in unit.map { $0 * scale } }
    }()

    /// 2026-08-28 audit, F2 second pass (server twin:
    /// `test_deflation_stops_at_the_effective_rank_not_the_theoretical_one`).
    /// The rank cap bounds by SHAPE (n−1, and now also the column count);
    /// rank-deficient data runs out sooner. Four rank-one rows with count = 3
    /// used to return THREE unit components with shares
    /// [1.0, 1.8e-15, 3.3e-17] — normalised float32 residue handed back as if
    /// it were a direction, which is exactly what the rank cap was written to
    /// prevent, arriving through the other door.
    @Test func deflationStopsAtTheEffectiveRankNotTheTheoreticalOne() throws {
        let result = try SteeringVectorMath.principalComponentsWithVariance(
            of: Self.rankOneRows, count: 3)
        #expect(result.components.count == 1)
        #expect(result.explainedVariance.count == 1)
        #expect(result.diagnostics.count == 1)
        #expect(abs(result.explainedVariance[0] - 1) < 1e-5)
    }

    @Test func deflationStopsAtTheEffectiveRankOfARankTwoCloud() throws {
        let first: [Float] = [1, 0, 0.5, 0.2]
        let second: [Float] = [0.1, 1, -0.3, 0.4]
        let coefficients: [(Float, Float)] = [
            (1, 0.2), (2, -0.5), (0.5, 1.3), (-1, 0.7), (3, -2),
        ]
        let rows: [[Float]] = coefficients.map { pair in
            zip(first, second).map { pair.0 * $0 + pair.1 * $1 }
        }
        let over = try SteeringVectorMath.principalComponentsWithVariance(
            of: rows, count: 4)
        #expect(over.components.count == 2)
        #expect(abs(over.explainedVariance.reduce(0, +) - 1) < 1e-5)
        // A request AT the effective rank is untouched: the components
        // computed before the stop are the same ones, bit for bit.
        let exact = try SteeringVectorMath.principalComponentsWithVariance(
            of: rows, count: 2)
        #expect(exact.components == over.components)
        #expect(exact.explainedVariance == over.explainedVariance)
    }

    /// Six rows of two columns span at most TWO dimensions however many rows
    /// arrive; the cap is min(count, rows − 1, columns).
    @Test func theColumnCountIsARankBoundToo() throws {
        let rows: [[Float]] = [[1, 2], [2, 1], [3, -1], [0, 0.5], [-2, 1], [4, 4]]
        let result = try SteeringVectorMath.principalComponentsWithVariance(
            of: rows, count: 4)
        #expect(result.components.count == 2)
    }

    /// The float32 first share of a rank-one cloud is 0.9999999, fractionally
    /// short of a 1.0 target, and the loop used to spend the shortfall on a
    /// second component made of round-off.
    @Test func aVarianceTargetOfOneStopsAtTheRankNotAtTheTarget() throws {
        let result = try SteeringVectorMath.principalComponents(
            of: Self.rankOneRows, minimumExplainedVariance: 1)
        #expect(result.components.count == 1)
        #expect(abs(result.totalExplainedVariance - 1) < 1e-5)
    }

    /// The floor is `factor · (rows + columns) · eps²` of the ORIGINAL total
    /// variance (server twin: `DEFLATION_RESIDUAL_TRACE_FLOOR_EPS_FACTOR` /
    /// `_residual_trace_floor`). Pinned here so a change to either engine's
    /// value is a deliberate two-engine edit.
    @Test func theResidualFloorIsANamedTwinConstant() {
        #expect(SteeringVectorMath.deflationResidualTraceFloorEpsFactor == 8)
        let value = SteeringVectorMath.residualTraceFloor(rows: 4, columns: 4)
        #expect(abs(value - 8 * 8 * Float.ulpOfOne * Float.ulpOfOne) < 1e-20)
        // Between the two measured populations: far above round-off residue,
        // far below the smallest genuine component ever measured.
        #expect(value > 1e-13 && value < 1e-11)
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
            positive: positive, negative: negative, method: .pairedDifferencePCA)

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
            positive: positive, negative: negative, method: .pairedDifferencePCA)
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
            positive: positive, negative: negative, method: .pairedDifferencePCA)
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
                positive: positive, negative: negative, method: .pairedDifferencePCA)
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
                    "gemmaScopeSAE", "repeReaderLAT",
                ])
    }

    /// A reader-derived direction is the artifact the FAITHFUL RepE pipeline
    /// produces, and until 2026-08-27 it was the one artifact a study could
    /// not attach: `attachArtifact` resolves the sidecar's extractionMethod to
    /// ask where the concept's held-out data lives, and an unknown method is
    /// refused. Its data questions have honest answers — they are just not a
    /// plain concept's.
    @Test func repeReaderLATAnswersTheLifecycleQuestions() {
        let reader = ExtractionMethod.repeReaderLAT
        #expect(ExtractionMethod(rawValue: "repeReaderLAT") == reader)
        #expect(reader.isRepeReaderLAT)
        #expect(!reader.hasSourceConcept)
        #expect(!reader.isPaired)
        #expect(!reader.isRecipeMethod)
        #expect(!reader.isPinnedArtifact)
        #expect(!reader.isGrandMean)
        #expect(!reader.usesStoryCorpus)
        #expect(!reader.usesContrastiveValidation)
        let absence = ExtractionMethod.repeReaderLAT.sourceConceptAbsence
        #expect(absence?.evidence.contains("held-out accuracy") == true)
        #expect(absence?.hashReferent.contains("prompts/readers/") == true)
        // …and it never reaches the direction math.
        #expect(throws: SteeringVectorError.self) {
            try SteeringVectorMath.direction(
                positive: [[1, 0]], negative: [[0, 0]], method: reader)
        }
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
        // The three methods with no source concept, and each says WHY in
        // its own words instead of a call site carrying a two-way condition.
        #expect(
            ExtractionMethod.allCases.filter { !$0.hasSourceConcept }
                == [.optvec, .gemmaScopeSAE, .repeReaderLAT])
        #expect(
            ExtractionMethod.allCases.allSatisfy {
                ($0.sourceConceptAbsence == nil) == $0.hasSourceConcept
            })
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
                positive: positive, negative: negative, method: .pairedDifferencePCA)
        }
    }
}
