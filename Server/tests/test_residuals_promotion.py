"""Alien-residual computation + screen→confirm promotion artifacts."""

import hashlib
import json
import os

import pytest

from steerlab_server.experiment import residuals, study_stats
from steerlab_server.experiment.manifest import PromotionRule
from steerlab_server.experiment.promotion import (
    PromotionCandidate, decide, write_promoted_movers)


def _row(condition, endpoint, delta_model, ci_model, delta_human, ci_human):
    return residuals.ResidualRow(
        condition=condition, endpoint=endpoint, delta_model=delta_model,
        ci_model=ci_model, delta_human=delta_human, ci_human=ci_human)


def test_region_classification():
    assert _row("c", "e", 2.0, (1.0, 3.0), 2.1, (1.5, 2.7)).region == "humanAligned"
    assert _row("c", "e", 5.0, (4.0, 6.0), 1.0, (0.5, 1.5)).region == "hyperHuman"
    assert _row("c", "e", 0.5, (0.2, 0.8), 2.0, (1.5, 2.5)).region == "hypoHuman"
    assert _row("c", "e", 2.0, (1.0, 3.0), 0.0, (-0.5, 0.5)).region == "alien"
    assert _row("c", "e", 0.0, (-0.5, 0.5), 0.0, (-0.5, 0.5)).region == "inertBoth"
    assert _row("c", "e", -2.0, (-3.0, -1.0), 2.0, (1.5, 2.5)).region == "inverted"


def test_residual_interval_is_conservative():
    row = _row("c", "e", 2.0, (1.0, 3.0), 1.0, (0.5, 1.5))
    assert row.residual == 1.0
    assert row.ci_residual == (1.0 - 1.5, 3.0 - 0.5)


def test_load_human_baseline_hash_gate(tmp_path):
    csv_text = ("endpoint,deltaHuman,ciLower,ciUpper\n"
                "meanMonths,4.2,2.0,6.4\n")
    path = tmp_path / "baseline.csv"
    path.write_text(csv_text)
    digest = hashlib.sha256(csv_text.encode()).hexdigest()
    table = residuals.load_human_baseline(str(path), digest)
    assert table["meanMonths"].delta == 4.2
    with pytest.raises(ValueError, match="drifted"):
        residuals.load_human_baseline(str(path), "0" * 64)
    bad = tmp_path / "bad.csv"
    bad.write_text("endpoint,delta\nx,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        residuals.load_human_baseline(str(bad))


def test_load_human_baseline_schema_error_names_required_and_found(tmp_path):
    # The shipped template historically carried different column names
    # (source, measure, ..., delta, ci_low, ...). The loader's columns ARE
    # the contract; a mismatch must fail loudly, naming both sides.
    bad = tmp_path / "template-shaped.csv"
    bad.write_text("source,measure,population,delta,ci_low,ci_high,n,notes\n"
                   "spamann,meanMonths,judges,4.2,2.0,6.4,123,\n")
    with pytest.raises(ValueError) as excinfo:
        residuals.load_human_baseline(str(bad))
    message = str(excinfo.value)
    for required in residuals.HUMAN_BASELINE_FIELDS:
        assert required in message  # names every required column
    assert "source" in message and "ci_low" in message  # names found columns
    assert "Fix:" in message  # one-line remedy


def test_load_human_baseline_minimal_valid_csv(tmp_path):
    path = tmp_path / "minimal.csv"
    path.write_text("endpoint,deltaHuman,ciLower,ciUpper\n"
                    "meanMonths,4.2,2.0,6.4\n"
                    "anchorSlope,0.5,0.2,0.8\n")
    table = residuals.load_human_baseline(str(path))
    assert set(table) == {"meanMonths", "anchorSlope"}
    assert table["meanMonths"].delta == 4.2
    assert table["anchorSlope"].ci_lower == 0.2
    assert table["anchorSlope"].ci_upper == 0.8


def test_residual_rows_join_and_csv(tmp_path):
    effect = study_stats.effect_row(
        "fear-a2", "meanMonths", [3.0, 5.0, 4.0, 6.0], replicates=200, seed=0)
    other = study_stats.effect_row(
        "fear-a2", "unmatchedEndpoint", [1.0, 2.0], replicates=200, seed=0)
    human = {"meanMonths": residuals.HumanEffect("meanMonths", 4.2, 2.0, 6.4)}
    rows = residuals.residual_rows([effect, other], human)
    assert len(rows) == 1  # endpoints without a human row have no residual
    out = tmp_path / "alien-residuals.csv"
    residuals.write_alien_residuals_csv(str(out), rows)
    lines = out.read_text().strip().splitlines()
    assert lines[0].split(",") == residuals.ALIEN_RESIDUALS_HEADER
    assert len(lines) == 2


def _candidate(**overrides):
    effect = study_stats.effect_row(
        "fear-a2", "meanMonths",
        [3.0, 4.0, 2.0, 5.0, 3.5, 2.5, 4.5, 3.0, 2.0, 4.0],
        replicates=300, seed=0)
    effect.adjusted_p = 0.01
    effect.correction = "bh"
    fields = dict(
        concept="fear", condition="fear-a2", endpoint="meanMonths",
        effect=effect,
        dose=study_stats.DoseResponse(spearman_rho=1.0, is_monotone=True),
        random_floor_effect=0.2, capability_passed=True)
    fields.update(overrides)
    return PromotionCandidate(**fields)


def test_promotion_pass_and_named_failures():
    rule = PromotionRule(fdr_threshold=0.05, dose_monotone=True,
                         exceeds_random_floor=True, capability_gate="battery")
    assert decide(_candidate(), rule).promoted

    failed_p = _candidate()
    failed_p.effect.adjusted_p = 0.2
    decision = decide(failed_p, rule)
    assert not decision.promoted and any("FDR" in r for r in decision.reasons)

    flat = decide(_candidate(dose=study_stats.DoseResponse(0.0, False)), rule)
    assert any("monotone" in r for r in flat.reasons)

    floor = decide(_candidate(random_floor_effect=10.0), rule)
    assert any("random floor" in r for r in floor.reasons)

    capability = decide(_candidate(capability_passed=False), rule)
    assert any("capability" in r for r in capability.reasons)


def test_write_promoted_movers(tmp_path):
    rule = PromotionRule()
    decisions = [decide(_candidate(), rule),
                 decide(_candidate(concept="boredom",
                                   random_floor_effect=99.0), rule)]
    path = tmp_path / "promoted-movers.json"
    write_promoted_movers(str(path), decisions, experiment="screen-1",
                          experiment_hash="abc123", rule=rule)
    payload = json.loads(path.read_text())
    assert payload["experiment"] == "screen-1"
    assert len(payload["promoted"]) == 1
    assert len(payload["rejected"]) == 1
    assert payload["rejected"][0]["reasons"]
    assert payload["promotionRule"]["fdrThreshold"] == 0.05
    assert os.path.exists(path)
