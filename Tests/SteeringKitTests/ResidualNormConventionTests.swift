import Foundation
import Testing

@testable import SteeringKit

/// The residual-norm DENOMINATOR CONVENTION — two averaging rules, two stamps.
///
/// Cross-engine twin: `Server/tests/test_residual_norm_convention.py`. Both
/// suites drive the SAME fixtures and assert the same numbers, so a
/// divergence in an averaging rule fails on whichever engine drifted rather
/// than surfacing months later as an uncomparable α.
///
/// Two fixtures, each chosen so that a test passing under either rule would
/// prove nothing:
///
/// * the TALLY fixture separates the whole-corpus mean from the banked-only
///   mean the server used before the 2026-08-20 ruling;
/// * the POOLED fixture (`fixtureWindows`) separates the two rules that shared
///   one stamp until the 2026-08-28 audit (F1) — variable-length texts read at
///   a pooled position, where the per-position mean and the mean of per-text
///   window-means are different numbers.
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

    /// Twin literals. Each string names ONE averaging rule; the version suffix
    /// moves only when that rule does.
    @Test func stampsAreThePinnedCrossEngineStrings() {
        #expect(ResidualNormConvention.wholeCorpusMean == "wholeCorpusMean-v1")
        #expect(ResidualNormConvention.perTextMean == "perTextMean-v1")
        #expect(ResidualNormConvention.wholeCorpusMean != ResidualNormConvention.perTextMean)
    }

    /// Every sidecar on disk stamped `wholeCorpusMean-v1` was written by a
    /// PER-TEXT writer — extraction or backfill — because the tally has never
    /// reached a sidecar on either engine. Frozen bytes are never rewritten,
    /// so the reader-side rule is that such a stamp means the per-text number
    /// it has always meant.
    @Test func theLegacyStampIsGrandfatheredOntoThePerTextRule() {
        #expect(
            ResidualNormConvention.grandfatheredPerTextStamp
                == ResidualNormConvention.wholeCorpusMean)
    }

    /// `extract` measures every denominator through `activations`, so the rule
    /// it stamps is the per-text one — not the tally's.
    @Test func extractionResultCarriesThePerTextRule() {
        let result = ExtractionResult(
            vectors: ConceptVectors(perLayer: [[1, 0]]),
            residualNormPerLayer: [1],
            residualNormSource: "neutral-corpus",
            options: ExtractionOptions())
        #expect(result.residualNormConvention == ResidualNormConvention.perTextMean)
    }

    /// The rewrap seam: the grand-mean path builds an `ExtractionResult` out
    /// of a `MultiConceptExtractionResult` whose denominator may have come
    /// from the token bank's per-position tally. The carrier must not
    /// re-default the stamp on the way through, or a bank-denominated vector
    /// is filed under the per-text rule.
    @Test func aCarriedConventionSurvivesTheRewrap() {
        let result = ExtractionResult(
            vectors: ConceptVectors(perLayer: [[1, 0]]),
            residualNormPerLayer: [1],
            residualNormSource: "neutral-token-bank",
            options: ExtractionOptions(),
            residualNormConvention: ResidualNormConvention.wholeCorpusMean)
        #expect(result.residualNormConvention == ResidualNormConvention.wholeCorpusMean)
    }

    // MARK: The two rules, discriminated
    //
    // THE fixture the audit asked for (F1): variable-length texts at a POOLED
    // reading position. Text A is read over 2 positions of norm 10; text B
    // over 6 positions of norm 2.
    //
    //   per-TEXT mean     = (10 + 2) / 2      = 6.0
    //   per-POSITION mean = (10·2 + 2·6) / 8  = 4.0
    //
    // The server twin drives both numbers out of real code (`activations` and
    // the bank driver); here the MLX forward pass needs a model, so the
    // per-position half runs through the shared `Tally` and the per-text half
    // through the same arithmetic `ConceptExtractor.activations` performs on
    // its captures — one window-mean per text, averaged over texts.
    private let fixtureWindows: [(positions: Int, norm: Float)] =
        [(2, 10), (6, 2)]
    private let pooledPerTextMean: Float = 6
    private let pooledPerPositionMean: Float = 4

    /// Guard on the guard, again: equal-length texts (the shape the pre-audit
    /// fixtures used) cannot tell the two rules apart, which is exactly why no
    /// test caught the shared stamp.
    @Test func pooledFixtureActuallyDiscriminatesTheTwoRules() {
        #expect(pooledPerTextMean != pooledPerPositionMean)
        #expect(Set(fixtureWindows.map(\.positions)).count > 1)
    }

    @Test func theTwoRulesDisagreeOnTheSamePooledCorpus() {
        // Per-position: every measured position counts once.
        var tally = ResidualNormConvention.Tally()
        for window in fixtureWindows {
            for _ in 0 ..< window.positions { tally.add(layer: 0, norm: window.norm) }
        }
        #expect(tally.mean(at: 0) == pooledPerPositionMean)

        // Per-text: each capture already holds ITS OWN window mean, and those
        // are averaged with equal weight per text.
        let windowMeans = fixtureWindows.map(\.norm)
        let perText = windowMeans.reduce(0, +) / Float(windowMeans.count)
        #expect(perText == pooledPerTextMean)
        #expect(perText != tally.mean(at: 0))
    }

    /// Why the grandfathering is safe: at a single-position reading every text
    /// contributes exactly one position, so per-text and per-position
    /// weighting are the same arithmetic and a legacy stamp names a number
    /// both rules produce.
    @Test func theTwoRulesAgreeAtASinglePositionReading() {
        var tally = ResidualNormConvention.Tally()
        for window in fixtureWindows { tally.add(layer: 0, norm: window.norm) }
        let windowMeans = fixtureWindows.map(\.norm)
        let perText = windowMeans.reduce(0, +) / Float(windowMeans.count)
        #expect(perText == tally.mean(at: 0))
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
                convention: ResidualNormConvention.perTextMean))
        let object = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        #expect(object["residualNormConvention"] as? String == "perTextMean-v1")
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
        // Both stamps display as themselves — the ONLY consumer of the string
        // on either engine is display/provenance (this label,
        // `SlotAlphaDefault`'s convention note, the OptVec packaging
        // advisory). Nothing gates on it, so a new-stamp artifact and an
        // old-stamp artifact cannot refuse against each other, and a reader
        // sees which rule they hold rather than a normalized fiction.
        #expect(
            ResidualNormConvention.displayLabel(
                residualNormPerLayer: [1], stamp: ResidualNormConvention.perTextMean)
                == "perTextMean-v1")
        #expect(
            ResidualNormConvention.displayLabel(
                residualNormPerLayer: [1], stamp: ResidualNormConvention.wholeCorpusMean)
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
                residualNormPerLayer: [], stamp: ResidualNormConvention.perTextMean) == nil)
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
