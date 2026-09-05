"""Statistics module: hand-computed fixtures for the corrections, property
tests for the bootstrap/Wilcoxon, dose-monotonicity gates."""

import json
import math
import os

import pytest

from steerlab_server.experiment import manifest, promotion, study_stats


def test_bh_fdr_hand_computed():
    # sorted: .005*4/1=.02, .01*4/2=.02, .03*4/3=.04, .04*4/4=.04
    adjusted = study_stats.bh_fdr([0.01, 0.04, 0.03, 0.005])
    assert [round(a, 6) for a in adjusted] == [0.02, 0.04, 0.04, 0.02]
    assert study_stats.bh_fdr([]) == []


def test_holm_hand_computed():
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04→max carries .06
    adjusted = study_stats.holm([0.01, 0.04, 0.03, 0.005])
    assert [round(a, 6) for a in adjusted] == [0.03, 0.06, 0.06, 0.02]


def test_adjusted_p_never_exceeds_one_and_preserves_order():
    p = [0.9, 0.8, 0.99]
    for method in (study_stats.bh_fdr, study_stats.holm):
        adjusted = method(p)
        assert all(0 <= a <= 1 for a in adjusted)
    # BH keeps the ranking of raw p-values weakly.
    adjusted = study_stats.bh_fdr(p)
    assert adjusted[1] <= adjusted[0] <= adjusted[2]


def test_paired_bootstrap_deterministic_and_degenerate():
    diffs = [2.0, 2.0, 2.0, 2.0]
    ci = study_stats.paired_bootstrap_ci(diffs, replicates=500, seed=7)
    assert ci.mean == 2.0 and ci.ci_lower == 2.0 and ci.ci_upper == 2.0
    varied = [1.0, 3.0, 2.0, 4.0, 0.0, 2.0]
    a = study_stats.paired_bootstrap_ci(varied, replicates=2000, seed=42)
    b = study_stats.paired_bootstrap_ci(varied, replicates=2000, seed=42)
    assert (a.ci_lower, a.ci_upper) == (b.ci_lower, b.ci_upper)
    assert a.ci_lower <= a.mean <= a.ci_upper


def test_wilcoxon_signed_rank_properties():
    # Strongly one-sided differences at n=12 → small p.
    diffs = [1.0, 2.0, 1.5, 3.0, 2.5, 1.2, 0.8, 2.2, 1.9, 2.8, 1.1, 0.9]
    w, p = study_stats.wilcoxon_signed_rank(diffs)
    assert w == 0.0
    assert p < 0.01
    # Sign symmetry: mirrored diffs give the identical statistic and p.
    w2, p2 = study_stats.wilcoxon_signed_rank([-d for d in diffs])
    assert (w2, p2) == (w, p)
    # All zeros → no evidence either way.
    w3, p3 = study_stats.wilcoxon_signed_rank([0.0, 0.0])
    assert math.isnan(w3) and math.isnan(p3)
    # Balanced diffs → p near 1.
    _, p4 = study_stats.wilcoxon_signed_rank([1, -1, 2, -2, 3, -3, 4, -4, 5, -5])
    assert p4 > 0.8


def test_dose_monotonicity():
    up = study_stats.dose_monotonicity([0.5, 1.0, 2.0], [0.1, 0.3, 0.7])
    assert up.is_monotone and math.isclose(up.spearman_rho, 1.0)
    down = study_stats.dose_monotonicity([0.5, 1.0, 2.0], [-0.1, -0.3, -0.7])
    assert down.is_monotone and math.isclose(down.spearman_rho, -1.0)
    inverted = study_stats.dose_monotonicity([0.5, 1.0, 2.0], [0.5, 0.7, 0.1])
    assert not inverted.is_monotone
    single = study_stats.dose_monotonicity([1.0], [0.4])
    assert not single.is_monotone
    # SCI-04 (external review, 2026-09-05): a flat ladder satisfies every
    # consecutive step trivially (0 >= -slack) but carries no dose evidence,
    # and its rho is undefined. Nondecreasing GEOMETRY is not the ACCEPTANCE
    # criterion — zero effect range is not monotone.
    flat = study_stats.dose_monotonicity([0.5, 1.0, 2.0], [0.3, 0.3, 0.3])
    assert not flat.is_monotone and math.isnan(flat.spearman_rho)


def test_dose_monotonicity_rejects_unequal_lengths():
    """A length mismatch is a pairing bug in the caller. ``zip`` would
    silently truncate and score a ladder nobody measured, so the helper
    raises (external review, 2026-09-05). The Swift twin cannot throw from
    this position and answers (NaN, false) instead — its own suite pins that.
    """
    with pytest.raises(ValueError):
        study_stats.dose_monotonicity([0.5, 1.0, 2.0], [0.1, 0.3])
    with pytest.raises(ValueError):
        study_stats.dose_monotonicity([0.5], [0.1, 0.3])


# --- the cross-engine contract ----------------------------------------------

#: Hand-authored (not regenerated) expectation table, replayed by the Swift
#: twin ``StudyStatisticsTests.theDoseVerdictMatchesTheCrossEngineFixture``.
#: See the file's own "note" for why it is not produced by
#: scripts/regenerate-cross-engine-fixtures.py.
DOSE_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Tests", "Fixtures", "cross-engine", "dose-monotonicity.json")

#: JSON carries no NaN/Infinity literal (and Foundation refuses the
#: non-standard tokens), so the fixture spells nonfinite inputs as strings.
_NONFINITE = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}


def _fixture_number(value) -> float:
    return _NONFINITE[value] if isinstance(value, str) else float(value)


def test_dose_monotonicity_matches_the_cross_engine_fixture():
    with open(DOSE_FIXTURE, encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = payload["cases"]
    assert len(cases) >= 14
    for case in cases:
        dose = study_stats.dose_monotonicity(
            [_fixture_number(a) for a in case["alphas"]],
            [_fixture_number(e) for e in case["effects"]],
            tolerance=case["tolerance"])
        assert dose.is_monotone == case["expectMonotone"], case["name"]
        assert (not math.isnan(dose.spearman_rho)) == case["expectRhoDefined"], \
            case["name"]


def test_flat_dose_response_fails_the_promotion_gate():
    """The point of SCI-04 at the gate it protects: judged proportions tie
    often, so an exactly flat ladder is realistic, and before the fix it
    walked through ``dose_monotone`` while its rho was undefined."""
    rule = manifest.PromotionRule(fdr_threshold=0.05, dose_monotone=True,
                                  exceeds_random_floor=False,
                                  capability_gate=None)

    def candidate(effects):
        row = study_stats.effect_row(
            "fear-a2", "meanMonths",
            [3.0, 4.0, 2.0, 5.0, 3.5, 2.5, 4.5, 3.0, 2.0, 4.0],
            replicates=200, seed=0)
        row.adjusted_p = 0.01
        row.correction = "bh"
        return promotion.PromotionCandidate(
            concept="fear", condition="fear-a2", endpoint="meanMonths",
            effect=row,
            dose=study_stats.dose_monotonicity([0.5, 1.0, 2.0], effects))

    flat = promotion.decide(candidate([0.3, 0.3, 0.3]), rule)
    assert not flat.promoted
    assert any("dose-response is not monotone" in r for r in flat.reasons)
    # An undefined rho must serialize as null, never as a valid zero.
    assert flat.as_json()["doseSpearmanRho"] is None
    assert flat.as_json()["doseMonotone"] is False

    rising = promotion.decide(candidate([0.1, 0.3, 0.7]), rule)
    assert rising.promoted and not rising.reasons
    assert rising.as_json()["doseSpearmanRho"] == 1.0


def test_effect_row_and_correction():
    rows = [
        study_stats.effect_row("fear-a2", "meanMonths", [3.0, 4.0, 2.0, 5.0, 3.5,
                                                         2.5, 4.5, 3.0, 2.0, 4.0],
                               replicates=500, seed=1),
        study_stats.effect_row("boredom-a2", "meanMonths", [0.1, -0.2, 0.05, -0.1,
                                                            0.15, -0.05, 0.0, 0.1,
                                                            -0.15, 0.2],
                               replicates=500, seed=1),
    ]
    study_stats.apply_correction(rows, method="bh")
    assert rows[0].correction == "bh"
    assert rows[0].adjusted_p <= rows[1].adjusted_p
    csv_row = rows[0].as_csv_row()
    assert csv_row[0] == "fear-a2" and csv_row[1] == "meanMonths"
    assert len(csv_row) == len(study_stats.EFFECT_SIZES_HEADER)


def test_percent_agreement_and_cohens_kappa_hand_computed():
    a = ["y", "y", "n", "n"]
    b = ["y", "n", "n", "n"]
    assert study_stats.percent_agreement(a, b) == 0.75
    # po = .75; pe = .5*.25 + .5*.75 = .5; kappa = (.75-.5)/(1-.5) = .5
    assert math.isclose(study_stats.cohens_kappa(a, b), 0.5)


def test_cohens_kappa_edge_cases():
    # Perfect agreement (multi-label and constant-same-label) is 1.0.
    assert study_stats.cohens_kappa(["v", "b", "t"], ["v", "b", "t"]) == 1.0
    assert study_stats.cohens_kappa(["v", "v"], ["v", "v"]) == 1.0
    # Constant raters on DIFFERENT labels: zero agreement, zero chance overlap.
    assert study_stats.cohens_kappa(["v", "v"], ["b", "b"]) == 0.0
    # Chance-level agreement corrects to 0.
    assert math.isclose(
        study_stats.cohens_kappa(["v", "v", "b", "b"], ["v", "b", "v", "b"]), 0.0)
    with pytest.raises(ValueError):
        study_stats.cohens_kappa(["v"], ["v", "b"])
    with pytest.raises(ValueError):
        study_stats.cohens_kappa([], [])
    with pytest.raises(ValueError):
        study_stats.percent_agreement([], [])
