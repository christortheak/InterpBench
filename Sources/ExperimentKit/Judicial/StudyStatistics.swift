import Foundation

/// Statistics for reported studies (Swift twin of the server's
/// `steerlab_server/experiment/study_stats.py` — shared fixture values, not
/// bit-identical randomness). Everything the reporting policy in CLAUDE.md
/// requires, implemented pure-Swift and deterministic so a re-run of a report
/// reproduces identical numbers on this substrate:
///
/// - paired bootstrap CIs on per-item (treatment − same-case-baseline)
///   differences,
/// - Wilcoxon signed-rank (robustness companion to the bootstrap),
/// - Benjamini–Hochberg FDR adjustment for Phase-1 screening across concepts,
/// - Holm step-down adjustment for the pre-registered confirm family,
/// - dose-monotonicity over an alpha grid (promotion-rule criterion).
public enum StudyStatistics {

    // MARK: - Paired bootstrap

    public struct BootstrapCI: Codable, Sendable, Equatable {
        public let n: Int
        public let mean: Double
        public let ciLower: Double
        public let ciUpper: Double
        public let replicates: Int
        public let seed: UInt64

        public init(
            n: Int, mean: Double, ciLower: Double, ciUpper: Double,
            replicates: Int, seed: UInt64
        ) {
            self.n = n
            self.mean = mean
            self.ciLower = ciLower
            self.ciUpper = ciUpper
            self.replicates = replicates
            self.seed = seed
        }
    }

    /// SplitMix64 — a tiny, seedable, deterministic generator. Determinism
    /// matters within-substrate (a report re-run reproduces its numbers); the
    /// Python twin uses Mersenne Twister, so bootstrap replicates are NOT
    /// bit-identical across engines, only the estimator semantics.
    private struct SplitMix64 {
        private var state: UInt64

        init(seed: UInt64) {
            state = seed
        }

        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    /// Percentile bootstrap CI on the mean of paired per-item differences.
    ///
    /// `diffs` must already be treatment − baseline for the SAME item — the
    /// pairing is the caller's job and the reason every study keeps a
    /// same-case baseline arm. Deterministic for a given seed. An empty input
    /// yields an all-NaN CI with n == 0 (the Python twin raises instead; a
    /// non-throwing NaN result keeps report assembly total).
    public static func pairedBootstrapCI(
        _ diffs: [Double], replicates: Int = 10_000, alpha: Double = 0.05,
        seed: UInt64 = 0
    ) -> BootstrapCI {
        guard !diffs.isEmpty else {
            return BootstrapCI(
                n: 0, mean: .nan, ciLower: .nan, ciUpper: .nan,
                replicates: replicates, seed: seed)
        }
        let n = diffs.count
        let mean = diffs.reduce(0, +) / Double(n)
        var rng = SplitMix64(seed: seed)
        var means: [Double] = []
        means.reserveCapacity(replicates)
        for _ in 0 ..< replicates {
            var total = 0.0
            for _ in 0 ..< n {
                total += diffs[Int(rng.next() % UInt64(n))]
            }
            means.append(total / Double(n))
        }
        means.sort()
        return BootstrapCI(
            n: n, mean: mean,
            ciLower: percentile(means, alpha / 2),
            ciUpper: percentile(means, 1 - alpha / 2),
            replicates: replicates, seed: seed)
    }

    /// Linear-interpolation percentile over a pre-sorted, non-empty list.
    private static func percentile(_ ordered: [Double], _ q: Double) -> Double {
        let position = q * Double(ordered.count - 1)
        let low = Int(position.rounded(.down))
        let high = Swift.min(low + 1, ordered.count - 1)
        let weight = position - Double(low)
        return ordered[low] * (1 - weight) + ordered[high] * weight
    }

    // MARK: - Wilcoxon signed-rank

    /// Two-sided Wilcoxon signed-rank test on paired differences.
    ///
    /// Zeros are dropped (standard treatment); ties get average ranks; the
    /// p-value uses the normal approximation with tie correction and
    /// continuity correction. Adequate at screening/confirm sample sizes
    /// (n ≥ ~10); at smaller n the test is reported but the bootstrap CI is
    /// the primary quantity. Returns (W, p) where W = min(W+, W−);
    /// (NaN, NaN) when every difference is zero.
    public static func wilcoxonSignedRank(_ diffs: [Double]) -> (w: Double, p: Double) {
        let nonzero = diffs.filter { $0 != 0 }
        let n = nonzero.count
        guard n > 0 else { return (.nan, .nan) }
        let byMagnitude = (0 ..< n).sorted { abs(nonzero[$0]) < abs(nonzero[$1]) }
        var ranks = [Double](repeating: 0, count: n)
        var tieCorrection = 0.0
        var i = 0
        while i < n {
            var j = i
            while j + 1 < n,
                abs(nonzero[byMagnitude[j + 1]]) == abs(nonzero[byMagnitude[i]])
            {
                j += 1
            }
            let averageRank = Double(i + j) / 2 + 1
            for k in i ... j {
                ranks[byMagnitude[k]] = averageRank
            }
            let t = Double(j - i + 1)
            tieCorrection += t * t * t - t
            i = j + 1
        }
        var wPlus = 0.0
        var wMinus = 0.0
        for (rank, diff) in zip(ranks, nonzero) {
            if diff > 0 { wPlus += rank } else { wMinus += rank }
        }
        let w = Swift.min(wPlus, wMinus)
        let count = Double(n)
        let meanW = count * (count + 1) / 4
        let variance = count * (count + 1) * (2 * count + 1) / 24 - tieCorrection / 48
        guard variance > 0 else { return (w, .nan) }
        let z = (w - meanW + 0.5) / variance.squareRoot()  // continuity-corrected
        let p = 2 * normalCDF(z)
        return (w, Swift.min(1, Swift.max(0, p)))
    }

    private static func normalCDF(_ z: Double) -> Double {
        0.5 * (1 + erf(z / 2.0.squareRoot()))
    }

    // MARK: - Inter-judge agreement

    /// Fraction of positions where the two label sequences agree. NaN for
    /// empty or length-mismatched inputs (a mismatch is a pairing bug the
    /// caller must surface, not average away).
    public static func percentAgreement<Label: Equatable>(
        _ a: [Label], _ b: [Label]
    ) -> Double {
        guard !a.isEmpty, a.count == b.count else { return .nan }
        let agreements = zip(a, b).count { $0 == $1 }
        return Double(agreements) / Double(a.count)
    }

    /// Cohen's kappa between two judges' label sequences (same items, same
    /// order — pairing is the caller's job, as with the bootstrap).
    ///
    /// κ = (p_o − p_e) / (1 − p_e) with p_e from each judge's own marginal
    /// label distribution. Degenerate case: when p_e == 1 (both judges
    /// constant on the SAME label), κ is 1 for perfect agreement and NaN
    /// otherwise (chance-corrected agreement is undefined). NaN for empty
    /// or length-mismatched inputs.
    public static func cohensKappa<Label: Hashable>(
        _ a: [Label], _ b: [Label]
    ) -> Double {
        guard !a.isEmpty, a.count == b.count else { return .nan }
        let n = Double(a.count)
        let observed = percentAgreement(a, b)
        var countsA: [Label: Int] = [:]
        var countsB: [Label: Int] = [:]
        for (left, right) in zip(a, b) {
            countsA[left, default: 0] += 1
            countsB[right, default: 0] += 1
        }
        var expected = 0.0
        for (label, countA) in countsA {
            expected += (Double(countA) / n) * (Double(countsB[label] ?? 0) / n)
        }
        if expected >= 1 {
            return observed == 1 ? 1 : .nan
        }
        return (observed - expected) / (1 - expected)
    }

    // MARK: - Multiplicity corrections

    /// Benjamini–Hochberg adjusted p-values (q-values), input order preserved,
    /// clipped to 1. The Phase-1 screening correction, applied across concepts.
    public static func bhFDR(_ pValues: [Double]) -> [Double] {
        let m = pValues.count
        guard m > 0 else { return [] }
        let order = (0 ..< m).sorted { pValues[$0] < pValues[$1] }
        var adjusted = [Double](repeating: 0, count: m)
        var runningMin = 1.0
        for position in stride(from: m - 1, through: 0, by: -1) {
            let index = order[position]
            let candidate = pValues[index] * Double(m) / Double(position + 1)
            runningMin = Swift.min(runningMin, candidate)
            adjusted[index] = Swift.min(1, runningMin)
        }
        return adjusted
    }

    /// Holm step-down adjusted p-values, input order preserved, clipped to 1.
    /// The confirm-phase familywise correction for the pre-registered
    /// hypothesis family.
    public static func holm(_ pValues: [Double]) -> [Double] {
        let m = pValues.count
        guard m > 0 else { return [] }
        let order = (0 ..< m).sorted { pValues[$0] < pValues[$1] }
        var adjusted = [Double](repeating: 0, count: m)
        var runningMax = 0.0
        for (position, index) in order.enumerated() {
            let candidate = pValues[index] * Double(m - position)
            runningMax = Swift.max(runningMax, candidate)
            adjusted[index] = Swift.min(1, runningMax)
        }
        return adjusted
    }

    // MARK: - Dose response

    public struct DoseResponse: Codable, Sendable, Equatable {
        public let spearmanRho: Double
        public let isMonotone: Bool

        public init(spearmanRho: Double, isMonotone: Bool) {
            self.spearmanRho = spearmanRho
            self.isMonotone = isMonotone
        }
    }

    /// Is the effect monotone in dose? (Promotion-rule criterion.)
    ///
    /// Sorts by alpha, reports Spearman's rho, and checks that consecutive
    /// effects never move against the overall direction by more than
    /// `tolerance × effect range`. A flat or inverted dose-response fails
    /// promotion no matter how large a single-alpha effect looks. Fewer than
    /// two points is never monotone.
    ///
    /// Nondecreasing GEOMETRY is not the ACCEPTANCE criterion, and conflating
    /// the two is what let a flat ladder through (external review,
    /// 2026-09-05, SCI-04): identical effects satisfy every consecutive step
    /// trivially (0 ≥ -slack) while carrying no dose information at all, and
    /// their Spearman rho is undefined. So a ZERO effect range (max == min)
    /// is not monotone. The refusal is exactly zero variation, not a minimum
    /// effect size — an increase of 1e-9 still passes, and this helper must
    /// not quietly become an effect-size gate.
    ///
    /// DIRECTION IS OBSERVED, NOT PRESPECIFIED. The sign comes from the
    /// sorted endpoints (last vs first), so a consistently DOWNWARD ladder is
    /// monotone here, and the random-floor criterion in the server's
    /// `promotion.decide` compares magnitudes (`abs`) by the same design. A
    /// study that wants to reject effects running opposite its hypothesis
    /// needs a declared expected direction, which this helper does not take.
    ///
    /// Input policy, matched on the Python twin except where noted:
    ///
    /// - fewer than two points → not monotone (rho needs two ranks anyway);
    /// - unequal input lengths → (NaN, false), the refusal shape
    ///   `percentAgreement` and `cohensKappa` already use in this file for a
    ///   mismatched pairing. A mismatch is a pairing bug in the caller, and
    ///   `zip` would silently truncate and score a ladder nobody measured.
    ///   (`study_stats.dose_monotonicity` raises `ValueError` from this
    ///   position instead; report assembly there is allowed to fail loudly,
    ///   which is why the shared fixture carries no mismatch case.)
    /// - any nonfinite alpha or effect → not monotone, rho NaN, decided
    ///   BEFORE the sort. NaN is not orderable — sorting on it is not a
    ///   strict weak ordering — and every comparison against it is false, so
    ///   an unguarded NaN can smuggle a true verdict out;
    /// - repeated doses (ties in `alphas`) are kept, not merged —
    ///   deliberately unchanged behaviour, pinned by the cross-engine
    ///   fixture. Pairs sort by (alpha, effect), so rows sharing a dose are
    ///   ordered by their own effect and a repeat never manufactures a step
    ///   violation; Spearman gives the tied doses their average rank, which
    ///   holds |rho| below 1 even for a perfect ladder.
    public static func doseMonotonicity(
        alphas: [Double], effects: [Double], tolerance: Double = 0
    ) -> DoseResponse {
        guard alphas.count == effects.count else {
            return DoseResponse(spearmanRho: .nan, isMonotone: false)
        }
        guard alphas.allSatisfy(\.isFinite), effects.allSatisfy(\.isFinite)
        else {
            return DoseResponse(spearmanRho: .nan, isMonotone: false)
        }
        let pairs = zip(alphas, effects)
            .sorted { $0.0 == $1.0 ? $0.1 < $1.1 : $0.0 < $1.0 }
        let ordered = pairs.map(\.1)
        let rho = spearman(pairs.map(\.0), ordered)
        guard let first = ordered.first, let last = ordered.last,
            ordered.count >= 2, let low = ordered.min(), let high = ordered.max()
        else {
            return DoseResponse(spearmanRho: rho, isMonotone: false)
        }
        let effectRange = high - low
        guard effectRange != 0 else {
            // SCI-04: flat is not monotone. rho is already NaN here (zero
            // rank variance), so the verdict and the correlation agree that
            // there is nothing to see.
            return DoseResponse(spearmanRho: rho, isMonotone: false)
        }
        let direction: Double = last >= first ? 1 : -1
        let slack = tolerance * effectRange
        let monotone = (0 ..< ordered.count - 1).allSatisfy { i in
            direction * (ordered[i + 1] - ordered[i]) >= -slack
        }
        return DoseResponse(spearmanRho: rho, isMonotone: monotone)
    }

    private static func spearman(_ xs: [Double], _ ys: [Double]) -> Double {
        guard xs.count >= 2 else { return .nan }
        let rankX = ranks(xs)
        let rankY = ranks(ys)
        let meanX = rankX.reduce(0, +) / Double(rankX.count)
        let meanY = rankY.reduce(0, +) / Double(rankY.count)
        let varX = rankX.map { ($0 - meanX) * ($0 - meanX) }.reduce(0, +)
        let varY = rankY.map { ($0 - meanY) * ($0 - meanY) }.reduce(0, +)
        guard varX != 0, varY != 0 else { return .nan }
        let cov = zip(rankX, rankY)
            .map { ($0 - meanX) * ($1 - meanY) }.reduce(0, +)
        return cov / (varX * varY).squareRoot()
    }

    private static func ranks(_ values: [Double]) -> [Double] {
        let order = (0 ..< values.count).sorted { values[$0] < values[$1] }
        var result = [Double](repeating: 0, count: values.count)
        var i = 0
        while i < order.count {
            var j = i
            while j + 1 < order.count, values[order[j + 1]] == values[order[i]] {
                j += 1
            }
            let average = Double(i + j) / 2 + 1
            for k in i ... j {
                result[order[k]] = average
            }
            i = j + 1
        }
        return result
    }
}
