import Foundation
import Testing
@testable import ExperimentKit

/// Mirrors `Server/tests/test_study_stats.py`: hand-computed fixtures for the
/// corrections, property tests for the bootstrap/Wilcoxon, dose-monotonicity
/// gates. Bootstrap replicates are deterministic within-substrate only (the
/// server uses a different RNG); the hand-computed correction values must
/// match the Python twin exactly.
@Suite struct StudyStatisticsTests {

    private func isClose(_ a: Double, _ b: Double, tolerance: Double = 1e-9) -> Bool {
        abs(a - b) <= tolerance * max(1, max(abs(a), abs(b)))
    }

    @Test func bhFDRHandComputed() {
        // sorted: .005*4/1=.02, .01*4/2=.02, .03*4/3=.04, .04*4/4=.04
        let adjusted = StudyStatistics.bhFDR([0.01, 0.04, 0.03, 0.005])
        let expected = [0.02, 0.04, 0.04, 0.02]
        #expect(adjusted.count == expected.count)
        for (a, e) in zip(adjusted, expected) {
            #expect(isClose(a, e, tolerance: 1e-6))
        }
        #expect(StudyStatistics.bhFDR([]) == [])
    }

    @Test func holmHandComputed() {
        // sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04→max carries .06
        let adjusted = StudyStatistics.holm([0.01, 0.04, 0.03, 0.005])
        let expected = [0.03, 0.06, 0.06, 0.02]
        #expect(adjusted.count == expected.count)
        for (a, e) in zip(adjusted, expected) {
            #expect(isClose(a, e, tolerance: 1e-6))
        }
        #expect(StudyStatistics.holm([]) == [])
    }

    /// Cross-engine parity pin: the expected values below were COMPUTED with
    /// the server's `bh_fdr`/`holm` (see
    /// `Server/tests/test_correction_parity.py`, which pins the identical
    /// literals). Both engines perform the same IEEE-754 operations in the
    /// same order, so agreement to 1e-12 is the contract. If this test and
    /// the server twin ever disagree, one engine's correction drifted — fix
    /// the engine, never the fixture.
    @Test func correctionParityFixturesMatchServer() {
        // Ties on both sides of the sort (0.02 twice, 0.04 twice): tie order
        // must not affect the adjusted values on either engine.
        let pTies = [0.02, 0.04, 0.002, 0.04, 0.31, 0.001, 0.02]
        let bhTies = [
            0.035, 0.04666666666666667, 0.007, 0.04666666666666667,
            0.31, 0.007, 0.035,
        ]
        let holmTies = [0.1, 0.12, 0.012, 0.12, 0.31, 0.007, 0.1]
        // Clipping at 1.0 and a raw p of exactly 1.0.
        let pClip = [0.049, 0.001, 0.049, 0.75, 1.0]
        let bhClip = [
            0.08166666666666667, 0.005, 0.08166666666666667, 0.9375, 1.0,
        ]
        let holmClip = [0.196, 0.005, 0.196, 1.0, 1.0]
        for (input, expected, method) in [
            (pTies, bhTies, StudyStatistics.bhFDR),
            (pTies, holmTies, StudyStatistics.holm),
            (pClip, bhClip, StudyStatistics.bhFDR),
            (pClip, holmClip, StudyStatistics.holm),
        ] {
            let adjusted = method(input)
            #expect(adjusted.count == expected.count)
            for (a, e) in zip(adjusted, expected) {
                #expect(isClose(a, e, tolerance: 1e-12))
            }
        }
        // m == 1: both corrections return the raw p (the common
        // one-condition study must read identically on both engines).
        #expect(isClose(StudyStatistics.bhFDR([0.0421])[0], 0.0421, tolerance: 1e-12))
        #expect(isClose(StudyStatistics.holm([0.0421])[0], 0.0421, tolerance: 1e-12))
    }

    @Test func adjustedPNeverExceedsOneAndPreservesOrder() {
        let p = [0.9, 0.8, 0.99]
        for method in [StudyStatistics.bhFDR, StudyStatistics.holm] {
            let adjusted = method(p)
            #expect(adjusted.allSatisfy { $0 >= 0 && $0 <= 1 })
        }
        // BH keeps the ranking of raw p-values weakly.
        let adjusted = StudyStatistics.bhFDR(p)
        #expect(adjusted[1] <= adjusted[0] && adjusted[0] <= adjusted[2])
    }

    @Test func cohensKappaHandComputed() {
        // Classic 2×2 fixture (Wikipedia's Cohen's-kappa example): n=50,
        // both-yes 20, both-no 15 (po = 0.7); A: 25 yes / 25 no,
        // B: 30 yes / 20 no ⇒ pe = 0.5*0.6 + 0.5*0.4 = 0.5 ⇒
        // κ = (0.7 − 0.5)/0.5 = 0.4.
        var a: [String] = []
        var b: [String] = []
        a += Array(repeating: "yes", count: 20) + Array(repeating: "yes", count: 5)
        b += Array(repeating: "yes", count: 20) + Array(repeating: "no", count: 5)
        a += Array(repeating: "no", count: 10) + Array(repeating: "no", count: 15)
        b += Array(repeating: "yes", count: 10) + Array(repeating: "no", count: 15)
        #expect(isClose(StudyStatistics.cohensKappa(a, b), 0.4, tolerance: 1e-9))
        #expect(isClose(StudyStatistics.percentAgreement(a, b), 0.7, tolerance: 1e-9))
    }

    @Test func cohensKappaProperties() {
        // Perfect agreement over multiple labels ⇒ κ = 1.
        let labels = ["condition", "baseline", "tie", "condition", "baseline"]
        #expect(isClose(StudyStatistics.cohensKappa(labels, labels), 1))
        // Chance-level structure ⇒ κ = 0: po equals pe by construction.
        let a = ["x", "x", "y", "y"]
        let b = ["x", "y", "x", "y"]
        #expect(isClose(StudyStatistics.cohensKappa(a, b), 0, tolerance: 1e-9))
        // Degenerate marginals: both judges constant on the same label —
        // p_e = 1, trivially perfect ⇒ 1; constant on DIFFERENT labels —
        // p_o = p_e = 0 ⇒ κ = 0 by the plain formula.
        #expect(StudyStatistics.cohensKappa(["t", "t"], ["t", "t"]) == 1)
        #expect(isClose(StudyStatistics.cohensKappa(["t", "t"], ["u", "u"]), 0))
        // Empty / length-mismatch ⇒ NaN, never a fabricated number.
        #expect(StudyStatistics.cohensKappa([String](), []).isNaN)
        #expect(StudyStatistics.cohensKappa(["a"], ["a", "b"]).isNaN)
        #expect(StudyStatistics.percentAgreement(["a"], ["a", "b"]).isNaN)
    }

    @Test func pairedBootstrapDeterministicAndDegenerate() {
        let ci = StudyStatistics.pairedBootstrapCI(
            [2.0, 2.0, 2.0, 2.0], replicates: 500, seed: 7)
        #expect(ci.mean == 2.0 && ci.ciLower == 2.0 && ci.ciUpper == 2.0)
        let varied = [1.0, 3.0, 2.0, 4.0, 0.0, 2.0]
        let a = StudyStatistics.pairedBootstrapCI(varied, replicates: 2000, seed: 42)
        let b = StudyStatistics.pairedBootstrapCI(varied, replicates: 2000, seed: 42)
        #expect(a.ciLower == b.ciLower && a.ciUpper == b.ciUpper)
        #expect(a.ciLower <= a.mean && a.mean <= a.ciUpper)
    }

    @Test func wilcoxonSignedRankProperties() {
        // Strongly one-sided differences at n=12 → small p.
        let diffs = [1.0, 2.0, 1.5, 3.0, 2.5, 1.2, 0.8, 2.2, 1.9, 2.8, 1.1, 0.9]
        let (w, p) = StudyStatistics.wilcoxonSignedRank(diffs)
        #expect(w == 0.0)
        #expect(p < 0.01)
        // Sign symmetry: mirrored diffs give the identical statistic and p.
        let (w2, p2) = StudyStatistics.wilcoxonSignedRank(diffs.map { -$0 })
        #expect(w2 == w && p2 == p)
        // All zeros → no evidence either way.
        let (w3, p3) = StudyStatistics.wilcoxonSignedRank([0.0, 0.0])
        #expect(w3.isNaN && p3.isNaN)
        // Balanced diffs → p near 1.
        let (_, p4) = StudyStatistics.wilcoxonSignedRank(
            [1, -1, 2, -2, 3, -3, 4, -4, 5, -5])
        #expect(p4 > 0.8)
    }

    @Test func doseMonotonicity() {
        let up = StudyStatistics.doseMonotonicity(
            alphas: [0.5, 1.0, 2.0], effects: [0.1, 0.3, 0.7])
        #expect(up.isMonotone && isClose(up.spearmanRho, 1.0))
        let down = StudyStatistics.doseMonotonicity(
            alphas: [0.5, 1.0, 2.0], effects: [-0.1, -0.3, -0.7])
        #expect(down.isMonotone && isClose(down.spearmanRho, -1.0))
        let inverted = StudyStatistics.doseMonotonicity(
            alphas: [0.5, 1.0, 2.0], effects: [0.5, 0.7, 0.1])
        #expect(!inverted.isMonotone)
        let single = StudyStatistics.doseMonotonicity(alphas: [1.0], effects: [0.4])
        #expect(!single.isMonotone)
    }
}
