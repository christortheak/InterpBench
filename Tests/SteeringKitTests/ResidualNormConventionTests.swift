import Foundation
import Testing

@testable import SteeringKit

/// The residual-norm DENOMINATOR CONVENTION — whole-corpus average, stamped.
///
/// Cross-engine twin: `Server/tests/test_residual_norm_convention.py`. Both
/// suites drive the SAME fixture and assert the same three numbers, so a
/// divergence in the averaging rule fails on whichever engine drifted rather
/// than surfacing months later as an uncomparable α.
///
/// The fixture is chosen so the banked-positions mean and the whole-corpus
/// mean are DIFFERENT numbers: a test that passed under either rule would not
/// have caught the bug this convention closes (the server averaged the draw,
/// this engine averaged the corpus).
@Suite struct ResidualNormConventionTests {

    // One layer, eight measured token positions. The row cap banked the four
    // SMALL ones and excluded the four LARGE ones — the realistic shape,
    // because the draw is blind to magnitude and any imbalance biases a
    // banked-only mean.
    private let fixtureLayer = 3
    private let fixtureBanked: [Float] = [10, 12, 14, 16]
    private let fixtureSkipped: [Float] = [40, 44, 48, 52]
    /// mean(banked) = 52/4 — the SUPERSEDED rule the server used pre-ruling.
    private let bankedOnlyMean: Float = 13
    /// mean(banked + skipped) = 236/8 — THE CONVENTION.
    private let wholeCorpusMean: Float = 29.5

    private func filledTally() -> ResidualNormConvention.Tally {
        var tally = ResidualNormConvention.Tally()
        for norm in fixtureBanked { tally.add(layer: fixtureLayer, norm: norm) }
        for norm in fixtureSkipped { tally.add(layer: fixtureLayer, norm: norm) }
        return tally
    }

    /// Guard on the guard: if these ever coincide, everything below proves
    /// nothing.
    @Test func fixtureActuallyDiscriminatesTheTwoRules() {
        #expect(bankedOnlyMean != wholeCorpusMean)
    }

    @Test func tallyAveragesOverTheWholeCorpusNotTheDraw() {
        let tally = filledTally()
        #expect(tally.mean(at: fixtureLayer) == wholeCorpusMean)
        #expect(tally.mean(at: fixtureLayer) != bankedOnlyMean)
        #expect(tally.count(at: fixtureLayer) == fixtureBanked.count + fixtureSkipped.count)
        #expect(tally.totalCount == fixtureBanked.count + fixtureSkipped.count)
        #expect(tally.layers == [fixtureLayer])
    }

    /// Feeding ONLY the banked rows reproduces the old divergent value —
    /// which is what makes the assertion above a real discrimination and not
    /// an accident of the fixture.
    @Test func bankedOnlyReproducesTheSupersededNumber() {
        var tally = ResidualNormConvention.Tally()
        for norm in fixtureBanked { tally.add(layer: fixtureLayer, norm: norm) }
        #expect(tally.mean(at: fixtureLayer) == bankedOnlyMean)
    }

    /// 0, never a borrowed neighbour: the downstream `> 0` guards refuse it
    /// rather than steering against a made-up denominator.
    @Test func unmeasuredLayerHasNoDenominator() {
        #expect(ResidualNormConvention.Tally().mean(at: 99) == 0)
    }

    @Test func stampIsThePinnedCrossEngineString() {
        #expect(ResidualNormConvention.current == "wholeCorpusMean-v1")
    }

    @Test func extractionResultCarriesTheConvention() {
        let result = ExtractionResult(
            vectors: ConceptVectors(perLayer: [[1, 0]]),
            residualNormPerLayer: [1],
            residualNormSource: "neutral-corpus",
            options: ExtractionOptions())
        #expect(result.residualNormConvention == ResidualNormConvention.current)
    }

    // MARK: The stamping

    private func sidecar(
        residualNormPerLayer: [Float]?, source: String?, convention: String?
    ) -> SteeringVectorSidecar {
        SteeringVectorSidecar(
            modelID: "m", concept: "c", stimulusSetHash: "h",
            vectors: ConceptVectors(perLayer: [[1, 0], [0, 1]]),
            residualNormPerLayer: residualNormPerLayer,
            residualNormSource: source,
            residualNormConvention: convention)
    }

    @Test func freshMeasurementStampsTheConvention() throws {
        let encoded = try JSONEncoder().encode(
            sidecar(
                residualNormPerLayer: [2, 3], source: "neutral-corpus abc123",
                convention: ResidualNormConvention.current))
        let object = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(object["residualNormConvention"] as? String == "wholeCorpusMean-v1")
    }

    /// Family convention: an unknown field is omitted, never written as
    /// `null`. A legacy artifact must be indistinguishable on the wire from
    /// one written before the field existed.
    @Test func unstampedIsAbsentNotNull() throws {
        let encoded = try JSONEncoder().encode(
            sidecar(
                residualNormPerLayer: [2, 3], source: "extraction-stimuli",
                convention: nil))
        let object = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(object["residualNormConvention"] == nil)
        #expect(object.keys.contains("residualNormPerLayer"))
    }

    /// The never-retro-applied rule: a sidecar written before the stamp
    /// existed decodes with no convention, keeps its norms untouched, and
    /// throws nothing. No migration, no recompute, no warning storm.
    @Test func legacySidecarReadsExactlyAsBefore() throws {
        let legacy = """
            {"schemaVersion": 2, "modelID": "m", "concept": "c",
             "stimulusSetHash": "h", "layerCount": 2, "hiddenSize": 2,
             "normsPerLayer": [1.0, 1.0],
             "extractionDate": "2026-01-01T00:00:00Z",
             "residualNormPerLayer": [2.0, 3.0],
             "residualNormSource": "neutral-corpus deadbeef"}
            """
        let decoded = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: Data(legacy.utf8))

        #expect(decoded.residualNormConvention == nil)
        #expect(decoded.residualNormPerLayer == [2, 3])
        #expect(decoded.residualNormSource == "neutral-corpus deadbeef")
        // And the display rule names it honestly rather than guessing a rule.
        #expect(
            ResidualNormConvention.displayLabel(
                residualNormPerLayer: decoded.residualNormPerLayer,
                stamp: decoded.residualNormConvention) == "legacy (pre-stamp)")
    }

    @Test func displayLabelRules() {
        #expect(
            ResidualNormConvention.displayLabel(
                residualNormPerLayer: [1], stamp: ResidualNormConvention.current)
                == "wholeCorpusMean-v1")
        #expect(
            ResidualNormConvention.displayLabel(residualNormPerLayer: [1], stamp: nil)
                == "legacy (pre-stamp)")
        // No norms at all: there is no denominator to describe.
        #expect(
            ResidualNormConvention.displayLabel(residualNormPerLayer: nil, stamp: nil)
                == nil)
        #expect(
            ResidualNormConvention.displayLabel(
                residualNormPerLayer: [], stamp: ResidualNormConvention.current) == nil)
    }

    // MARK: The denominator-table gate (2026-08-28 audit, F7/F13)
    //
    // Twin prose, asserted as literals on both engines
    // (`Server/tests/test_denominator_table_gate.py`). One truncated table
    // used to produce four different behaviours across the two engines'
    // verbs — a 0.0 substitution, two silent clamps, and (on an EMPTY table
    // here) an index [-1] crash. The sentences below ARE the single rule.

    @Test func loadTimeGateNamesTheArtifactAndBothNumbers() {
        #expect(
            ResidualNormConvention.tableLengthProblem(
                [1, 2], layerCount: 5, artifact: "runs/r/c")
                == """
                vector artifact 'runs/r/c' carries 2 residual norms for 5 \
                layers — a denominator table must cover every layer or none, \
                and a short one silently doses the layers it does not reach \
                with another layer's number; re-measure the norms (vectors \
                backfill-norms), or re-extract the concept
                """)
    }

    /// The born-without families (OptVec, J-lens, Gemma Scope report imports)
    /// write no norms at all and acquire them through the backfill. Absent is
    /// a state the writers deliberately produce; SHORT is a state none of
    /// them produces, which is why only short refuses.
    @Test func absentAndEmptyTablesAreLegalAtLoadTime() {
        #expect(
            ResidualNormConvention.tableLengthProblem(
                nil, layerCount: 5, artifact: "a") == nil)
        #expect(
            ResidualNormConvention.tableLengthProblem(
                [], layerCount: 5, artifact: "a") == nil)
        #expect(
            ResidualNormConvention.tableLengthProblem(
                [1, 1, 1, 1, 1], layerCount: 5, artifact: "a") == nil)
    }

    @Test func useSiteGateNamesTheLayerAndTheCoverage() {
        #expect(
            ResidualNormConvention.residualNormProblem(
                [1, 2], layer: 4, artifact: "fear")
                == """
                'fear' has no residual norm at layer 4 — its denominator \
                table covers 2 layer(s), so an α in residual-norm units \
                cannot be denominated there; re-measure the norms (vectors \
                backfill-norms), or switch α to raw units
                """)
    }

    /// The empty table is the case that indexed `[-1]` here: a clamp of
    /// `min(layer, count - 1)` against a zero-length table is `-1`, which is
    /// a fatal index on this engine and a silent 0.0 substitution on the
    /// server's condition path.
    @Test func useSiteGateRefusesEmptyTablesAndNegativeLayers() {
        #expect(
            ResidualNormConvention.residualNormProblem([], layer: 0, artifact: "f")
                != nil)
        #expect(
            ResidualNormConvention.residualNormProblem(nil, layer: 0, artifact: "f")
                != nil)
        #expect(
            ResidualNormConvention.residualNormProblem([1, 2], layer: -1, artifact: "f")
                != nil)
        #expect(
            ResidualNormConvention.residualNormProblem([1, 2], layer: 1, artifact: "f")
                == nil)
    }

    @Test func theAccessorReturnsTheDenominatorOrThrowsTheRefusal() throws {
        #expect(
            try ResidualNormConvention.residualNorm([2, 4, 6], at: 2, artifact: "f")
                == 6)
        #expect(throws: ResidualNormTableError.self) {
            try ResidualNormConvention.residualNorm([2, 4], at: 2, artifact: "f")
        }
        #expect(throws: ResidualNormTableError.self) {
            try ResidualNormConvention.residualNorm([], at: 0, artifact: "f")
        }
    }
}
