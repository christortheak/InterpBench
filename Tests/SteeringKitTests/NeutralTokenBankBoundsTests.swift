import Foundation
import MLX
import Testing

@testable import SteeringKit

/// The neutral token bank's crash-grade hazard: it used to record EVERY
/// eligible token at EVERY configured layer into CPU float32 arrays and only
/// then apply the 4096-row cap. At 4B+ scale over a real corpus that transient
/// is multi-GB × layers, and it fails inside MLX/native allocation — the
/// process dies before a Swift `catch` runs (CLAUDE.md › MLX gotchas; hit live
/// building a neutral-PC basis in the app, 2026-08-18).
///
/// The fix caps ingestion instead of trimming afterwards, WITHOUT changing
/// which rows are selected. These tests pin both halves: identical rows, and a
/// bounded peak.
@Suite struct NeutralTokenBankBoundsTests {

    /// Deterministic synthetic pass: values encode (text, position, hidden)
    /// so every row is distinguishable.
    private func pass(text: Int, length: Int, hidden: Int) -> MLXArray {
        var values: [Float] = []
        values.reserveCapacity(length * hidden)
        for position in 0 ..< length {
            for d in 0 ..< hidden {
                values.append(Float(text * 1000 + position * 10 + d))
            }
        }
        return MLXArray(values).reshaped([1, length, hidden])
    }

    /// Drives a recorder over a fixed synthetic corpus, draining after each
    /// pass exactly as `ConceptExtractor.neutralActivationBank` does.
    private func drive(
        recorder: ActivationBankRecorder, lengths: [Int], layers: [Int], hidden: Int
    ) -> [Int: [[Float]]] {
        var rowsByLayer: [Int: [[Float]]] = [:]
        for (text, length) in lengths.enumerated() {
            recorder.reset()
            let h = pass(text: text, length: length, hidden: hidden)
            for layer in layers {
                _ = recorder.apply(h, layer: layer, offset: 0)
            }
            for row in recorder.rows {
                rowsByLayer[row.layer, default: []].append(row.values)
            }
        }
        return rowsByLayer
    }

    private let lengths = [7, 11, 5, 9]  // 32 bankable positions
    private let hidden = 3

    /// Artifact continuity: the streaming filter keeps EXACTLY the rows the
    /// old collect-then-downsample path kept. If this ever fails, bases built
    /// before and after the fix are no longer comparable.
    @Test func streamingSelectionKeepsTheSameRowsAsPostHocDownsampling() {
        let cap = 8
        let seed = TokenBankDownsampler.seed(fromCorpusHash: "corpus-a")
        let total = lengths.reduce(0, +)

        // Pre-fix shape: bank everything, then downsample.
        let unbounded = ActivationBankRecorder(layers: [0], startIndex: 0)
        let allRows = drive(recorder: unbounded, lengths: lengths, layers: [0], hidden: hidden)[0]!
        #expect(allRows.count == total)
        let keptIndices = TokenBankDownsampler.selectedIndices(
            count: allRows.count, cap: cap, seed: seed)
        let expected = keptIndices.map { allRows[$0] }

        // Post-fix shape: the same draw, precomputed, applied during capture.
        let bounded = ActivationBankRecorder(
            layers: [0], startIndex: 0,
            selectedRowIndices: TokenBankDownsampler.selectedIndexSet(
                count: total, cap: cap, seed: seed))
        let streamed = drive(recorder: bounded, lengths: lengths, layers: [0], hidden: hidden)[0]!

        #expect(streamed.count == cap)
        #expect(streamed == expected)
    }

    /// Same corpus + same cap, twice → the same rows. The draw is a pure
    /// function of (corpus hash, cap, row count), not of timing or ordering.
    @Test func selectionIsDeterministicAcrossRuns() {
        let cap = 6
        let seed = TokenBankDownsampler.seed(fromCorpusHash: "corpus-b")
        let total = lengths.reduce(0, +)
        let selection = TokenBankDownsampler.selectedIndexSet(
            count: total, cap: cap, seed: seed)

        func run() -> [[Float]] {
            let recorder = ActivationBankRecorder(
                layers: [0, 1], startIndex: 0, selectedRowIndices: selection)
            return drive(recorder: recorder, lengths: lengths, layers: [0, 1], hidden: hidden)[1]!
        }
        let first = run()
        let second = run()
        #expect(first.count == cap)
        #expect(first == second)

        // A different corpus reshuffles loudly rather than silently.
        let otherSeed = TokenBankDownsampler.seed(fromCorpusHash: "corpus-c")
        let otherRecorder = ActivationBankRecorder(
            layers: [0, 1], startIndex: 0,
            selectedRowIndices: TokenBankDownsampler.selectedIndexSet(
                count: total, cap: cap, seed: otherSeed))
        let other = drive(
            recorder: otherRecorder, lengths: lengths, layers: [0, 1], hidden: hidden)[1]!
        #expect(other != first)
    }

    /// The cap binds DURING ingestion. `peakRetainedRowCount` is the recorder's
    /// high-water mark, so this measures retention, not just the final bank.
    @Test func peakRetentionIsBoundedByTheCapDuringIngestion() {
        let cap = 8
        let total = lengths.reduce(0, +)
        let seed = TokenBankDownsampler.seed(fromCorpusHash: "corpus-a")

        // Pre-fix retention: every eligible row, at every layer, before any cap.
        let unbounded = ActivationBankRecorder(layers: [0, 1, 2], startIndex: 0)
        _ = drive(recorder: unbounded, lengths: lengths, layers: [0, 1, 2], hidden: hidden)
        // The longest pass alone retains length × layers rows.
        #expect(unbounded.peakRetainedRowCount == (lengths.max() ?? 0) * 3)

        let bounded = ActivationBankRecorder(
            layers: [0, 1, 2], startIndex: 0,
            selectedRowIndices: TokenBankDownsampler.selectedIndexSet(
                count: total, cap: cap, seed: seed))
        let banked = drive(recorder: bounded, lengths: lengths, layers: [0, 1, 2], hidden: hidden)

        // Never more than the cap per layer, ever — including mid-capture.
        #expect(bounded.peakRetainedRowCount <= cap * 3)
        #expect(bounded.peakRetainedRowCount < unbounded.peakRetainedRowCount)
        // Measured on this fixture: 33 rows retained at peak before the fix
        // (the 11-token pass × 3 layers), 9 after (the pass contributing the
        // most of the 8-row draw, × 3 layers). Pinned exactly so a regression
        // that reintroduces collect-then-trim fails loudly rather than merely
        // staying under a loose bound.
        #expect(unbounded.peakRetainedRowCount == 33)
        #expect(bounded.peakRetainedRowCount == 9)
        for layer in [0, 1, 2] {
            #expect(banked[layer]?.count == cap)
        }
    }

    /// Every layer banks the SAME token positions, so per-layer banks stay
    /// row-aligned (the PCA and the neutral-mean estimate both assume it).
    @Test func everyLayerBanksTheSamePositions() {
        let cap = 5
        let total = lengths.reduce(0, +)
        let recorder = ActivationBankRecorder(
            layers: [0, 1], startIndex: 0,
            selectedRowIndices: TokenBankDownsampler.selectedIndexSet(
                count: total, cap: cap, seed: 42))
        var tokenIndicesByLayer: [Int: [Int]] = [:]
        for (text, length) in lengths.enumerated() {
            recorder.reset()
            let h = pass(text: text, length: length, hidden: hidden)
            for layer in [0, 1] { _ = recorder.apply(h, layer: layer, offset: 0) }
            for row in recorder.rows {
                tokenIndicesByLayer[row.layer, default: []].append(row.tokenIndex)
            }
        }
        #expect(tokenIndicesByLayer[0] == tokenIndicesByLayer[1])
        #expect(tokenIndicesByLayer[0]?.count == cap)
    }

    /// Under the cap, nothing is dropped and nothing is skipped.
    @Test func noSelectionKeepsEveryRow() {
        let recorder = ActivationBankRecorder(layers: [0], startIndex: 0, selectedRowIndices: nil)
        let banked = drive(recorder: recorder, lengths: lengths, layers: [0], hidden: hidden)
        #expect(banked[0]?.count == lengths.reduce(0, +))
    }

    /// Residual norms must describe the CORPUS, not the draw: excluded
    /// positions still report a norm so the denominator is unbiased.
    @Test func excludedPositionsStillReportResidualNorms() {
        let cap = 4
        let total = lengths.reduce(0, +)
        let recorder = ActivationBankRecorder(
            layers: [0], startIndex: 0,
            selectedRowIndices: TokenBankDownsampler.selectedIndexSet(
                count: total, cap: cap, seed: 11))
        var banked = 0
        var skipped = 0
        for (text, length) in lengths.enumerated() {
            recorder.reset()
            _ = recorder.apply(pass(text: text, length: length, hidden: hidden), layer: 0, offset: 0)
            banked += recorder.rows.count
            skipped += recorder.skippedNorms.count
        }
        #expect(banked == cap)
        #expect(banked + skipped == total)
    }

    /// `startIndex` still governs which positions are bankable at all, and the
    /// global cursor counts from the first BANKABLE position.
    @Test func startIndexShiftsTheBankablePositions() {
        let recorder = ActivationBankRecorder(layers: [0], startIndex: 3, selectedRowIndices: nil)
        _ = recorder.apply(pass(text: 0, length: 7, hidden: hidden), layer: 0, offset: 0)
        #expect(recorder.rows.map(\.tokenIndex) == [3, 4, 5, 6])
        // A pass shorter than startIndex banks nothing and does not advance.
        recorder.reset()
        _ = recorder.apply(pass(text: 1, length: 2, hidden: hidden), layer: 0, offset: 0)
        #expect(recorder.rows.isEmpty)
    }

    /// `reset()` drains rows but keeps the cursor (per-pass draining must not
    /// restart the global selection); `resetAll()` starts a fresh bank.
    @Test func resetDrainsRowsButKeepsTheCursor() {
        // Keep only global positions 4 and 5 — reachable only if the cursor
        // survives the first pass's reset.
        let recorder = ActivationBankRecorder(
            layers: [0], startIndex: 0, selectedRowIndices: [4, 5])
        _ = recorder.apply(pass(text: 0, length: 4, hidden: hidden), layer: 0, offset: 0)
        #expect(recorder.rows.isEmpty)
        recorder.reset()
        _ = recorder.apply(pass(text: 1, length: 4, hidden: hidden), layer: 0, offset: 4)
        #expect(recorder.rows.map(\.tokenIndex) == [0, 1])

        recorder.resetAll()
        _ = recorder.apply(pass(text: 2, length: 4, hidden: hidden), layer: 0, offset: 0)
        #expect(recorder.rows.isEmpty)  // cursor restarted at 0
        #expect(recorder.peakRetainedRowCount == 0)
    }
}

/// The bank's memory preflight — refuse a projected-over-budget capture before
/// the first forward pass, because the failure mode it guards is a process
/// kill, not a throw.
@Suite struct NeutralBankBudgetTests {

    @Test func estimateIsRowsTimesLayersTimesHidden() {
        let preflight = NeutralBankBudget.preflight(
            sourceRowsPerLayer: 50_000,
            rowCapPerLayer: 4096,
            layerCount: 62,
            hiddenSize: 5376,
            budgetBytes: 1 << 40)
        // The cap binds: 4096 rows, not 50k.
        #expect(preflight.retainedRowsPerLayer == 4096)
        #expect(preflight.sourceRowsPerLayer == 50_000)
        let expected =
            4096 * 62 * 5376 * NeutralBankBudget.bytesPerValue
            * NeutralBankBudget.workingSetMultiplier
        #expect(preflight.estimatedBytes == expected)
        #expect(preflight.fits)
    }

    /// An honest all-layer 27B bank (62 × 4096 × 5376) must still RUN on the
    /// 64 GB machine — the preflight exists to catch runaway shapes, not to
    /// forbid the real work.
    @Test func realisticAllLayerBankFitsOn64GB() {
        let preflight = NeutralBankBudget.preflight(
            sourceRowsPerLayer: 200_000,
            rowCapPerLayer: 4096,
            layerCount: 62,
            hiddenSize: 5376,
            budgetBytes: NeutralBankBudget.defaultBudgetBytes(
                physicalMemory: 64 * 1024 * 1024 * 1024))
        #expect(preflight.fits)
    }

    @Test func overBudgetRefusalNamesBothRemedies() {
        let preflight = NeutralBankBudget.preflight(
            sourceRowsPerLayer: 4_000_000,
            rowCapPerLayer: 1_000_000,
            layerCount: 62,
            hiddenSize: 5376,
            budgetBytes: NeutralBankBudget.defaultBudgetBytes(
                physicalMemory: 64 * 1024 * 1024 * 1024))
        #expect(!preflight.fits)

        let message = ConceptExtractorError.neutralBankTooLarge(preflight).description
        // The remedies the researcher can actually apply.
        #expect(message.contains("layer band"))
        #expect(message.contains("middle third"))
        #expect(message.contains("row cap"))
        #expect(message.contains("1000000"))  // the cap currently in force
        #expect(message.contains("62 layers"))
    }

    @Test func budgetHasAFloorSoSmallMachinesStillRun() {
        #expect(
            NeutralBankBudget.defaultBudgetBytes(physicalMemory: 2 * 1024 * 1024 * 1024)
                == NeutralBankBudget.minimumBudgetBytes)
        #expect(
            NeutralBankBudget.defaultBudgetBytes(physicalMemory: 64 * 1024 * 1024 * 1024)
                == 16 * 1024 * 1024 * 1024)
    }
}

/// The layer-band default the app's neutral-PC build now uses.
@Suite struct LayerBandTests {

    @Test func middleThirdIsTheMiddleThird() {
        #expect(LayerBand.middleThird(ofBlockCount: 48) == Set(16 ..< 32))
        #expect(LayerBand.middleThird(ofBlockCount: 62) == Set(20 ..< 41))
        #expect(LayerBand.middleThird(ofBlockCount: 36) == Set(12 ..< 24))
    }

    @Test func degenerateBlockCountsKeepEveryLayer() {
        #expect(LayerBand.middleThird(ofBlockCount: 0).isEmpty)
        #expect(LayerBand.middleThird(ofBlockCount: 1) == [0])
        #expect(LayerBand.middleThird(ofBlockCount: 2) == [0, 1])
        // Never empty for a real model — an empty band would bank nothing.
        for count in 1 ... 80 {
            #expect(!LayerBand.middleThird(ofBlockCount: count).isEmpty)
        }
    }

    @Test func descriptionNamesTheRule() {
        #expect(
            LayerBand.description(of: Set(0 ..< 48), blockCount: 48) == "all 48 layers")
        #expect(
            LayerBand.description(of: LayerBand.middleThird(ofBlockCount: 48), blockCount: 48)
                == "middle-third band 16–31 of 48")
        #expect(LayerBand.description(of: [], blockCount: 48) == "no layers")
    }
}

/// The hard component ceiling alongside the variance target.
@Suite struct NeutralPCComponentCapTests {

    /// Rows with a deliberately FLAT spectrum: an explained-variance target
    /// alone would keep deflating to `rows − 1` components per layer.
    private func flatRows(count: Int, dimension: Int) -> [[Float]] {
        (0 ..< count).map { i in
            (0 ..< dimension).map { d in d == i % dimension ? Float(1 + i / dimension) : 0 }
        }
    }

    private func bank(rows: [[Float]]) -> NeutralActivationBank {
        NeutralActivationBank(
            layers: [0],
            rowsByLayer: [rows],
            residualNormPerLayer: [1],
            screening: ExtractionScreening(
                readingPosition: .lastToken, sourceCount: rows.count,
                includedCount: rows.count, excludedShortCount: 0),
            tokenRowCount: rows.count,
            sourceRowsPerLayer: rows.count,
            usedRowsPerLayer: rows.count,
            downsampleSeed: nil)
    }

    @Test func varianceTargetIsCappedByTheComponentCeiling() throws {
        let rows = flatRows(count: 24, dimension: 24)
        let capped = try bank(rows: rows).componentsByLayer(
            selection: .explainedVariance(0.99, maximumCount: 3))
        #expect(capped[0].count == 3)

        // The target still governs below the cap.
        let targeted = try bank(rows: rows).componentsByLayer(
            selection: .explainedVariance(0.05, maximumCount: 16))
        #expect(targeted[0].count < 16)
        #expect(!targeted[0].isEmpty)
    }

    @Test func defaultCeilingApplies() throws {
        #expect(VectorExtractionRecipe.NeutralPCSelection.defaultMaximumComponentCount == 64)
        let selection = VectorExtractionRecipe.NeutralPCSelection.explainedVariance(0.5)
        #expect(selection.maximumCount == nil)
        #expect(selection.resolvedMaximumCount == 64)
        // A zero/negative declared ceiling is not a way to disable the bound.
        #expect(
            VectorExtractionRecipe.NeutralPCSelection(kind: .explainedVariance, maximumCount: 0)
                .resolvedMaximumCount == 64)
    }

    @Test func fixedCountIsAlsoClamped() throws {
        let rows = flatRows(count: 24, dimension: 24)
        let components = try bank(rows: rows).componentsByLayer(
            selection: VectorExtractionRecipe.NeutralPCSelection(
                kind: .fixedCount, count: 20, maximumCount: 4))
        #expect(components[0].count == 4)
    }

    /// An omitted ceiling must not perturb pinned recipe hashes.
    @Test func omittedCeilingDoesNotChangeRecipeEncoding() throws {
        let recipe = VectorExtractionRecipe(
            name: "r", method: .emotionGrandMean, targetConcept: "c", datasets: [],
            neutralPCSelection: .explainedVariance(0.5))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = String(decoding: try encoder.encode(recipe), as: UTF8.self)
        #expect(!json.contains("maximumCount"))
    }
}
