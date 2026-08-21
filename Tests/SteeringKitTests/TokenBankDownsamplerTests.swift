import Foundation
import Testing
@testable import SteeringKit

/// Pure downsampling seam for the neutral token bank: deterministic for the
/// same corpus hash, bounded by the cap, order-stable (kept rows preserve
/// original relative order), and pass-through under the cap.
@Suite struct TokenBankDownsamplerTests {

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
    }
}
