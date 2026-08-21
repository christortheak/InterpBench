import CryptoKit
import Foundation

/// Deterministic row downsampling for neutral token banks.
///
/// The all-layer per-token float32 bank feeding the Gram-matrix PCA is the
/// extraction path's memory/compute hazard at scale (CLAUDE.md › MLX
/// gotchas): the Gram solve is O(rows²) per layer, so rows must be bounded
/// before large corpora are usable. The bound is `defaultRowCapPerLayer`
/// (4096 rows/layer — a 4096×4096 Gram matrix stays well inside a few
/// hundred MB and PCA over thousands of rows remains statistically
/// meaningful for the ≤ dozens of PCs any basis keeps).
///
/// Selection is pure and reproducible: the seed derives from the corpus
/// hash, so re-running the same pinned corpus reproduces the same bank
/// (and a corpus change reshuffles loudly, never silently).
public enum TokenBankDownsampler {

    /// Maximum token rows kept per layer for basis PCA.
    public static let defaultRowCapPerLayer = 4096

    /// Seed derived from a corpus hash (hex or any string): the first 8
    /// bytes of SHA-256 over the hash's UTF-8, big-endian. Stable across
    /// runs and engines that adopt the same derivation.
    public static func seed(fromCorpusHash hash: String) -> UInt64 {
        let digest = SHA256.hash(data: Data(hash.utf8))
        return digest.prefix(8).reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
    }

    /// Deterministic hash of raw corpus texts, for callers that hold texts
    /// but no file hash (e.g. the grand-mean neutral branch). Length-prefixed
    /// so ["ab","c"] and ["a","bc"] differ.
    public static func corpusHash(texts: [String]) -> String {
        var data = Data()
        for text in texts {
            var count = UInt64(text.utf8.count).bigEndian
            withUnsafeBytes(of: &count) { data.append(contentsOf: $0) }
            data.append(contentsOf: Array(text.utf8))
        }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// The row indices to KEEP out of `count`, at most `cap`, chosen by a
    /// seeded partial Fisher–Yates draw and returned sorted ascending —
    /// deterministic for a (count, cap, seed) triple, bounded by `cap`, and
    /// order-stable (kept rows preserve their original relative order).
    public static func selectedIndices(count: Int, cap: Int, seed: UInt64) -> [Int] {
        guard count > 0, cap > 0 else { return [] }
        guard count > cap else { return Array(0 ..< count) }
        var rng = SplitMix64(seed: seed)
        var pool = Array(0 ..< count)
        for position in 0 ..< cap {
            let remaining = count - position
            let offset = Int(rng.next() % UInt64(remaining))
            pool.swapAt(position, position + offset)
        }
        return pool.prefix(cap).sorted()
    }

    /// The same selection as `selectedIndices`, expressed as a membership set
    /// so a bank can DROP over-cap rows AS THEY ARRIVE instead of
    /// materializing every row and downsampling afterwards. `nil` means "keep
    /// every row" (count ≤ cap), which lets callers skip the lookup entirely.
    ///
    /// This is deliberately built FROM `selectedIndices` rather than
    /// re-derived from a row hash. The draw is a seeded partial Fisher–Yates
    /// over `0 ..< count`, so it depends on the TOTAL row count and cannot be
    /// reproduced by a row-local rule — but the total is knowable before any
    /// forward pass (tokenize first, then `max(0, tokens - startIndex)` per
    /// text). Precomputing the set therefore keeps the pre-fix two-phase path
    /// and the bounded streaming path selecting BYTE-IDENTICAL rows for the
    /// same (count, cap, seed): bases built before and after the
    /// bounded-ingestion fix remain directly comparable.
    public static func selectedIndexSet(count: Int, cap: Int, seed: UInt64) -> Set<Int>? {
        guard cap > 0 else { return [] }
        guard count > cap else { return nil }
        return Set(selectedIndices(count: count, cap: cap, seed: seed))
    }
}
