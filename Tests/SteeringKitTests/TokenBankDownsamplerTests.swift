import Foundation
import Testing
@testable import SteeringKit

/// Pure downsampling seam for the neutral token bank: deterministic for the
/// same corpus hash, bounded by the cap, order-stable (kept rows preserve
/// original relative order), and pass-through under the cap.
///
/// Cross-engine twin: `Server/tests/test_token_bank_downsampling.py`. The
/// LITERAL block below appears byte-identically in both suites — same seed
/// derivation, same SplitMix64 stream, same selected indices. The server
/// adopted this engine's algorithm on 2026-08-28 (audit convention note 9),
/// where the draw had been `random.sample` seeded from a differently-derived
/// number: CPython promises reproducibility across versions only for
/// `Random.random()`, so an interpreter upgrade could have silently rebuilt a
/// neutral-PC basis from different token positions with every stamp claiming
/// the recipe was unchanged. Literals, not properties, are what make that
/// impossible on either side. The default row CAP deliberately did NOT
/// converge (4096 here behind `NeutralBankBudget.preflight`, 2048 on the
/// server, which has no preflight), so a bit-identical bank needs the same cap
/// passed on both engines.
@Suite struct TokenBankDownsamplerTests {

    // MARK: Twin literals, asserted verbatim on both engines

    private let literalSeedABC123: UInt64 = 7_827_605_053_139_634_307
    private let literalSplitMix64From7: [UInt64] = [
        7_191_089_600_892_374_487,
        309_689_372_594_955_804,
        16_616_101_746_815_609_346,
        10_753_165_928_301_472_203,
    ]
    private let literalSelection1000_8_42 = [26, 182, 338, 396, 413, 552, 855, 942]
    private let literalSelection10000_12_ABC123 = [
        1174, 1287, 2726, 2780, 3665, 4125, 5279, 5775, 6565, 6917, 7573, 8481,
    ]
    private let literalCorpusHashABC =
        "601d5476e2ccfe2c87a2bba7a322659734a05749d5b5aa781f513e4912db0d5f"

    @Test func splitMix64StreamIsThePinnedSequence() {
        var rng = SplitMix64(seed: 7)
        #expect((0 ..< 4).map { _ in rng.next() } == literalSplitMix64From7)
    }

    @Test func seedDerivationIsThePinnedCrossEngineRule() {
        #expect(TokenBankDownsampler.seed(fromCorpusHash: "abc123") == literalSeedABC123)
    }

    @Test func selectionIsThePinnedDraw() {
        #expect(
            TokenBankDownsampler.selectedIndices(count: 1000, cap: 8, seed: 42)
                == literalSelection1000_8_42)
        #expect(
            TokenBankDownsampler.selectedIndices(
                count: 10_000, cap: 12, seed: literalSeedABC123)
                == literalSelection10000_12_ABC123)
    }

    @Test func deterministicForSameCorpusHash() {
        let seed = TokenBankDownsampler.seed(fromCorpusHash: "abc123")
        #expect(seed == TokenBankDownsampler.seed(fromCorpusHash: "abc123"))
        #expect(seed != TokenBankDownsampler.seed(fromCorpusHash: "abc124"))

        let a = TokenBankDownsampler.selectedIndices(count: 10_000, cap: 512, seed: seed)
        let b = TokenBankDownsampler.selectedIndices(count: 10_000, cap: 512, seed: seed)
        #expect(a == b)
        let other = TokenBankDownsampler.selectedIndices(
            count: 10_000, cap: 512,
            seed: TokenBankDownsampler.seed(fromCorpusHash: "abc124"))
        #expect(a != other)
    }

    @Test func boundedAndOrderStable() {
        let picked = TokenBankDownsampler.selectedIndices(count: 5_000, cap: 128, seed: 7)
        #expect(picked.count == 128)
        // Valid, distinct indices…
        #expect(Set(picked).count == picked.count)
        #expect(picked.allSatisfy { (0 ..< 5_000).contains($0) })
        // …in ascending order, so kept rows preserve their original relative
        // order (order-stable).
        #expect(picked == picked.sorted())
    }

    @Test func passThroughWhenUnderCap() {
        #expect(
            TokenBankDownsampler.selectedIndices(count: 100, cap: 4096, seed: 1)
                == Array(0 ..< 100))
        #expect(
            TokenBankDownsampler.selectedIndices(count: 4096, cap: 4096, seed: 1)
                == Array(0 ..< 4096))
        #expect(TokenBankDownsampler.selectedIndices(count: 0, cap: 16, seed: 1) == [])
        #expect(TokenBankDownsampler.selectedIndices(count: 16, cap: 0, seed: 1) == [])
    }

    /// The streaming form must be the SAME draw as the two-phase one — that
    /// equality is what lets bounded ingestion keep artifact continuity.
    @Test func indexSetMatchesTheTwoPhaseDraw() {
        let seed = TokenBankDownsampler.seed(fromCorpusHash: "abc123")
        let set = TokenBankDownsampler.selectedIndexSet(count: 10_000, cap: 512, seed: seed)
        #expect(set == Set(TokenBankDownsampler.selectedIndices(
            count: 10_000, cap: 512, seed: seed)))
        #expect(set?.count == 512)

        // nil means "keep everything" — no membership check needed.
        #expect(TokenBankDownsampler.selectedIndexSet(count: 100, cap: 4096, seed: 1) == nil)
        #expect(TokenBankDownsampler.selectedIndexSet(count: 4096, cap: 4096, seed: 1) == nil)
        // A non-positive cap keeps nothing (not everything).
        #expect(TokenBankDownsampler.selectedIndexSet(count: 16, cap: 0, seed: 1) == [])
    }

    @Test func textCorpusHashIsLengthPrefixed() {
        // Length-prefixing means boundary shifts change the hash.
        let a = TokenBankDownsampler.corpusHash(texts: ["ab", "c"])
        let b = TokenBankDownsampler.corpusHash(texts: ["a", "bc"])
        #expect(a != b)
        #expect(a == TokenBankDownsampler.corpusHash(texts: ["ab", "c"]))
        #expect(a == literalCorpusHashABC)
    }
}
