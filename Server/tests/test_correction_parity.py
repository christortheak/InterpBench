"""Cross-engine multiple-comparison-correction parity (2026-07-19 review
finding: Swift emitted only raw Wilcoxon p-values).

These fixtures pin the EXACT numeric outputs of ``bh_fdr`` / ``holm`` and the
family semantics of ``apply_correction``. The identical literals are pinned in
the Swift suite (``StudyStatisticsTests.correctionParityFixturesMatchServer``
and the ``applyCorrection`` tests in ``AnalyzeEffectSizesTests.swift``): both
engines perform the same IEEE-754 operations in the same order
(p * m / rank, running min/max, clip at 1), so agreement to 1e-12 is the
contract. If this file and the Swift tests ever disagree, one engine's
correction drifted — fix the engine, never the fixture.

Deliberately a NEW test file: the parity pin must not entangle with the
existing ``test_study_stats.py`` hand-computed fixtures.
"""

import math

import pytest

from steerlab_server.experiment import study_stats

# Vector 1: ties on both sides of the sort (0.02 twice, 0.04 twice) — tie
# order must not affect the adjusted values on either engine.
P_TIES = [0.02, 0.04, 0.002, 0.04, 0.31, 0.001, 0.02]
BH_TIES = [0.035, 0.04666666666666667, 0.007, 0.04666666666666667,
           0.31, 0.007, 0.035]
HOLM_TIES = [0.1, 0.12, 0.012, 0.12, 0.31, 0.007, 0.1]

# Vector 2: clipping at 1.0 and a raw p of exactly 1.0.
P_CLIP = [0.049, 0.001, 0.049, 0.75, 1.0]
BH_CLIP = [0.08166666666666667, 0.005, 0.08166666666666667, 0.9375, 1.0]
HOLM_CLIP = [0.196, 0.005, 0.196, 1.0, 1.0]


def test_bh_fdr_parity_vectors():
    assert study_stats.bh_fdr(P_TIES) == pytest.approx(BH_TIES, rel=1e-12)
    assert study_stats.bh_fdr(P_CLIP) == pytest.approx(BH_CLIP, rel=1e-12)


def test_holm_parity_vectors():
    assert study_stats.holm(P_TIES) == pytest.approx(HOLM_TIES, rel=1e-12)
    assert study_stats.holm(P_CLIP) == pytest.approx(HOLM_CLIP, rel=1e-12)


def _row(condition: str, endpoint: str, p: float) -> study_stats.EffectRow:
    ci = study_stats.BootstrapCI(n=3, mean=1.0, ci_lower=0.5, ci_upper=1.5,
                                 replicates=100, seed=0)
    return study_stats.EffectRow(condition=condition, endpoint=endpoint,
                                 ci=ci, wilcoxon_w=0.0, wilcoxon_p=p)


def test_apply_correction_family_semantics_parity():
    """One correction family (the analyze verb groups per endpoint before
    calling this): NaN p rows are SKIPPED by the adjustment but still
    stamped with the method; the dense values match the Swift
    ``applyCorrection`` fixtures exactly."""
    rows = [_row("c1", "wordCount", 0.02),
            _row("c2", "wordCount", float("nan")),
            _row("c3", "wordCount", 0.03)]
    study_stats.apply_correction(rows, method="bh")
    # bh([0.02, 0.03]) == [0.03, 0.03] — pinned in Swift too.
    assert rows[0].adjusted_p == pytest.approx(0.03, rel=1e-12)
    assert math.isnan(rows[1].adjusted_p)
    assert rows[2].adjusted_p == pytest.approx(0.03, rel=1e-12)
    assert [r.correction for r in rows] == ["bh", "bh", "bh"]

    rows = [_row("c1", "wordCount", 0.02),
            _row("c2", "wordCount", float("nan")),
            _row("c3", "wordCount", 0.03)]
    study_stats.apply_correction(rows, method="holm")
    # holm([0.02, 0.03]) == [0.04, 0.04] — pinned in Swift too.
    assert rows[0].adjusted_p == pytest.approx(0.04, rel=1e-12)
    assert math.isnan(rows[1].adjusted_p)
    assert rows[2].adjusted_p == pytest.approx(0.04, rel=1e-12)
    assert [r.correction for r in rows] == ["holm", "holm", "holm"]


def test_single_test_family_is_identity():
    """m == 1: both corrections return the raw p — the common one-condition
    study must read identically on both engines."""
    assert study_stats.bh_fdr([0.0421]) == pytest.approx([0.0421], rel=1e-12)
    assert study_stats.holm([0.0421]) == pytest.approx([0.0421], rel=1e-12)
