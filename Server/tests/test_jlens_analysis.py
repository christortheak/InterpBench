"""Target/control aggregation, the permutation null, epoch guard, bundle gate."""

import json
import os

import pytest

from steerlab_server.jlens import analysis, trace
from steerlab_server.jlens.schemas import JLensError

WATCH = [11, 22, 33]          # targets 11,22 ; control 33


def _obs(index, layer, scores, ll=None):
    row = {"layer": layer, "predictedIndex": index, "position": 10 + index,
           "passKind": "decode", "watched": scores}
    if ll is not None:
        row["watchedLogitLens"] = ll
    return row


def _row(observations, mention=None, **kw):
    row = {"run": "r", "condition": "baseline", "promptID": "p0",
           "sampleIndex": 0, "observations": observations,
           "observationCount": len(observations), "traceComplete": True,
           "watchlistTokenIDs": WATCH}
    if mention is not None:
        row["mentionMask"] = mention
    row.update(kw)
    return row


# --- token sets --------------------------------------------------------------

def test_a_token_set_needs_targets_and_disjoint_controls():
    with pytest.raises(JLensError, match="at least one target"):
        analysis.TokenSet().validate(WATCH)
    with pytest.raises(JLensError, match="both target and control"):
        analysis.TokenSet(targets=[11], controls=[11]).validate(WATCH)


def test_a_token_set_must_name_only_watched_tokens():
    """The trace contains no scores for a token the readout never watched, so
    naming one would silently drop it from the contrast."""
    with pytest.raises(JLensError, match="never watched"):
        analysis.TokenSet(targets=[11], controls=[99]).validate(WATCH)


def test_the_set_hash_is_order_independent_and_content_bound():
    a = analysis.TokenSet(targets=[22, 11], controls=[33], name="x")
    b = analysis.TokenSet(targets=[11, 22], controls=[33], name="x")
    assert a.hash() == b.hash()
    assert a.hash() != analysis.TokenSet(
        targets=[11], controls=[33], name="x").hash()


# --- the aggregate -----------------------------------------------------------

def test_the_score_is_target_minus_control_not_a_level():
    """A raw level carries the layer's overall scale, and scale differences
    across layers or conditions look exactly like loading differences."""
    ts = analysis.TokenSet(targets=[11, 22], controls=[33])
    got = analysis.concept_score(_obs(0, 5, [10.0, 20.0, 5.0]), ts, WATCH)
    assert got["score"] == pytest.approx(15.0 - 5.0)
    assert got["raw"] == pytest.approx(15.0)
    assert got["convention"] == analysis.SCORE_CONVENTION
    assert (got["targetCount"], got["controlCount"]) == (2, 1)


def test_declared_absent_controls_report_a_level_and_say_so():
    ts = analysis.TokenSet(targets=[11, 22])
    got = analysis.concept_score(_obs(0, 5, [10.0, 20.0, 5.0]), ts, WATCH)
    assert got["score"] == pytest.approx(15.0)
    assert got["convention"] == analysis.RAW_CONVENTION


def test_mentioned_tokens_are_excluded_from_the_aggregate_not_the_trace():
    """A primed token sits near ceiling for reasons unrelated to the model's
    state, so it must not be averaged in — but the row stays in the trace."""
    ts = analysis.TokenSet(targets=[11, 22], controls=[33])
    mask = {"11": True, "22": False, "33": False}
    got = analysis.concept_score(_obs(0, 5, [99.0, 20.0, 5.0]), ts, WATCH,
                                 mention_mask=mask)
    assert got["targetCount"] == 1                 # 11 primed out
    assert got["score"] == pytest.approx(20.0 - 5.0)


def test_nothing_usable_returns_none_not_zero():
    """Zero is a value; absence is not. Averaging a fabricated zero into a
    condition mean would bias it toward no effect."""
    ts = analysis.TokenSet(targets=[11], controls=[33])
    mask = {"11": True}
    assert analysis.concept_score(_obs(0, 5, [1.0, 2.0, 3.0]), ts, WATCH,
                                  mention_mask=mask) is None
    assert analysis.concept_score({"layer": 5, "predictedIndex": 0}, ts,
                                  WATCH) is None


def test_the_band_restricts_which_layers_count():
    ts = analysis.TokenSet(targets=[11], controls=[33])
    row = _row([_obs(0, 5, [10.0, 0.0, 0.0]), _obs(0, 9, [100.0, 0.0, 0.0])])
    banded = analysis.aggregate_over_band(row, ts, WATCH, band=[5])
    assert banded["score"] == pytest.approx(10.0)
    assert banded["band"] == [5]
    everything = analysis.aggregate_over_band(row, ts, WATCH)
    assert everything["score"] == pytest.approx(55.0)


def test_counts_travel_with_the_number():
    """An aggregate over two observations and one over two hundred are
    different claims."""
    ts = analysis.TokenSet(targets=[11], controls=[33])
    row = _row([_obs(i, 5, [float(i), 0.0, 0.0]) for i in range(4)])
    got = analysis.aggregate_over_band(row, ts, WATCH)
    assert got["scoredObservations"] == 4
    assert got["excludedObservations"] == 0
    assert got["tokenSetHash"] == ts.hash()


def test_the_logit_lens_companion_can_be_aggregated_the_same_way():
    ts = analysis.TokenSet(targets=[11], controls=[33])
    row = _row([_obs(0, 5, [10.0, 0.0, 1.0], ll=[2.0, 0.0, 1.0])])
    j = analysis.aggregate_over_band(row, ts, WATCH)
    ll = analysis.aggregate_over_band(row, ts, WATCH, use_logit_lens=True)
    assert j["score"] == pytest.approx(9.0)
    assert ll["score"] == pytest.approx(1.0)
    assert ll["usedLogitLens"] is True


# --- the null ----------------------------------------------------------------

def _series_row(values, layer=5):
    return _row([_obs(i, layer, [v, 0.0, 0.0]) for i, v in enumerate(values)])


def test_a_permutation_invariant_statistic_is_refused_by_name():
    """The bug this prevents, found on a real trace: a shuffled-position null
    for a band-averaged MEAN returns the observed value every draw and reports
    p = 1.0 exactly — indistinguishable in a table from a strong negative
    result, and actually a null incapable of finding anything.

    `peak`/`max` are in the refused list because they are the TEMPTING version
    of the same mistake: the largest value in a set does not change when you
    shuffle which position holds it. This module made that error twice — once
    for the mean, then again by defaulting to peak."""
    ts = analysis.TokenSet(targets=[11], controls=[33])
    row = _series_row([1.0, 5.0, 2.0, 9.0])
    for stat in ("mean", "meanOverPositions", "sum", "peak", "max", "median"):
        with pytest.raises(JLensError, match="invariant under position"):
            analysis.permutation_null(row, ts, WATCH, statistic=stat)


@pytest.mark.parametrize("statistic", sorted(analysis.STATISTICS))
def test_every_supported_statistic_has_a_null_that_moves(statistic):
    """The property that makes a permutation null meaningful at all: the draws
    must not all equal the observed value. Asserted per statistic so a future
    addition cannot quietly be another invariant one."""
    import random as _r

    rng = _r.Random(0)
    series = {i: rng.random() for i in range(8)}
    fn = analysis.STATISTICS[statistic]
    order = sorted(series)
    values = [series[i] for i in order]
    draws = []
    for _ in range(200):
        rng.shuffle(values)
        draws.append(fn(dict(zip(order, values))))
    assert max(draws) - min(draws) > 1e-9


def test_an_unknown_statistic_is_refused():
    ts = analysis.TokenSet(targets=[11], controls=[33])
    with pytest.raises(JLensError, match="unknown statistic"):
        analysis.permutation_null(_series_row([1.0, 2.0]), ts, WATCH,
                                  statistic="vibes")


def test_early_minus_late_has_a_moving_null():
    ts = analysis.TokenSet(targets=[11], controls=[33])
    front = _series_row([10.0, 9.0, 8.0, 1.0, 0.0, 0.0])
    got = analysis.permutation_null(front, ts, WATCH,
                                   statistic="earlyMinusLate",
                                   permutations=200)
    assert got["observed"] > 0
    # A front-loaded series should be unusual against shuffled positions.
    assert got["null"]["p"] < 0.2
    assert got["null"]["mean"] != pytest.approx(got["observed"])


def test_the_null_is_deterministic_from_the_trace_content():
    """Same trace, same null, on any machine — no seeds table to carry."""
    ts = analysis.TokenSet(targets=[11], controls=[33])
    row = _series_row([float(i) for i in range(6)])
    a = analysis.permutation_null(row, ts, WATCH, permutations=50)
    b = analysis.permutation_null(row, ts, WATCH, permutations=50)
    assert a["null"]["seed"] == b["null"]["seed"]
    assert a["null"]["p"] == b["null"]["p"]


def test_the_null_never_replays_the_model():
    """A null built by re-generating would read residuals the model never had
    at those steps — the same objection that makes post-hoc replay a non-goal."""
    ts = analysis.TokenSet(targets=[11], controls=[33])
    got = analysis.permutation_null(_series_row([float(i) for i in range(6)]),
                                   ts, WATCH, permutations=20)
    assert got["null"]["replayed"] is False
    assert got["null"]["convention"] == "positionPermutationWithinBand"


def test_the_p_value_can_never_be_zero():
    """(k+1)/(n+1): an empirical null of n draws cannot license a claim
    stronger than 1/(n+1)."""
    ts = analysis.TokenSet(targets=[11], controls=[33])
    got = analysis.permutation_null(_series_row([float(i) for i in range(8)]),
                                   ts, WATCH, permutations=10)
    assert got["null"]["p"] >= 1 / 11


def test_too_few_steps_reports_no_null_rather_than_a_fake_one():
    ts = analysis.TokenSet(targets=[11], controls=[33])
    got = analysis.permutation_null(_series_row([1.0]), ts, WATCH)
    assert got["null"] is None and "at least 2" in got["reason"]


def test_zero_permutations_refuses():
    ts = analysis.TokenSet(targets=[11], controls=[33])
    with pytest.raises(JLensError):
        analysis.permutation_null(_series_row([1.0, 2.0]), ts, WATCH,
                                  permutations=0)


def test_the_step_series_is_position_resolved():
    ts = analysis.TokenSet(targets=[11], controls=[33])
    row = _row([_obs(0, 5, [1.0, 0, 0]), _obs(0, 9, [3.0, 0, 0]),
                _obs(1, 5, [7.0, 0, 0]), _obs(1, 9, [7.0, 0, 0])])
    series = analysis.step_series(row, ts, WATCH)
    assert series == {0: pytest.approx(2.0), 1: pytest.approx(7.0)}


# --- reportability: completeness + epoch ------------------------------------

def _write_trace(tmp_path, rows):
    path = tmp_path / trace.TRACE_FILENAME
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(tmp_path)


def test_read_summary_is_derived_from_the_file_not_a_stamp(tmp_path):
    """A summary that reports on itself cannot detect truncation after the
    fact, which is the failure this guards."""
    run = _write_trace(tmp_path, [_row([_obs(0, 5, [1.0, 0, 0])])])
    summary = trace.read_summary(run)
    assert summary["traceRows"] == 1 and summary["complete"] is True
    assert len(summary["traceSHA256"]) == 64


def test_an_incomplete_row_makes_the_run_unreportable(tmp_path):
    run = _write_trace(tmp_path, [_row([], traceComplete=False)])
    with pytest.raises(JLensError, match="incomplete"):
        trace.require_reportable(run, name="s")


def test_a_missing_trace_is_named_rather_than_treated_as_empty(tmp_path):
    with pytest.raises(JLensError, match="no J-lens trace"):
        trace.require_reportable(str(tmp_path), name="s")


def test_an_unstamped_run_is_refused_by_the_epoch_guard(tmp_path):
    """A trace produced before the manifest changed describes settings that are
    no longer the study's — and a changed watchlist or band makes the stored
    numbers answers to a different question."""
    run = _write_trace(tmp_path, [_row([_obs(0, 5, [1.0, 0, 0])])])
    with pytest.raises(JLensError, match="no experiment-hash stamp"):
        trace.require_reportable(run, name="s", live_hash="abc123")


def test_a_matching_epoch_stamp_passes(tmp_path):
    run = _write_trace(tmp_path, [_row([_obs(0, 5, [1.0, 0, 0])])])
    (tmp_path / "experiment-hash.txt").write_text("abc123")
    summary = trace.require_reportable(run, name="s", live_hash="abc123")
    assert summary["complete"] is True
    assert summary["epochUnverified"] is False


def test_a_stale_epoch_stamp_refuses(tmp_path):
    run = _write_trace(tmp_path, [_row([_obs(0, 5, [1.0, 0, 0])])])
    (tmp_path / "experiment-hash.txt").write_text("OLD")
    with pytest.raises(JLensError, match="different manifest epoch"):
        trace.require_reportable(run, name="s", live_hash="abc123")


def test_legacy_unstamped_runs_can_be_admitted_explicitly(tmp_path):
    run = _write_trace(tmp_path, [_row([_obs(0, 5, [1.0, 0, 0])])])
    summary = trace.require_reportable(run, name="s", live_hash="abc123",
                                       allow_unverified_epoch=True)
    assert summary["epochUnverified"] is True


# --- the bundle --------------------------------------------------------------

def test_an_incomplete_trace_blocks_the_evidence_complete_marker(tmp_path):
    """A bundle stamping evidenceComplete over a truncated readout would assert
    something false about its own contents."""
    from steerlab_server.experiment import bundles

    run = tmp_path / "run"
    run.mkdir()
    _write_trace(run, [_row([], traceComplete=False)])
    assert "incomplete" in (bundles._jlens_trace_problem(str(run)) or "")


def test_a_complete_trace_is_not_a_bundle_problem(tmp_path):
    from steerlab_server.experiment import bundles

    run = tmp_path / "run"
    run.mkdir()
    _write_trace(run, [_row([_obs(0, 5, [1.0, 0, 0])])])
    assert bundles._jlens_trace_problem(str(run)) is None


def test_no_trace_at_all_is_not_a_bundle_problem(tmp_path):
    """Most runs declare no readout; absent must stay free."""
    from steerlab_server.experiment import bundles

    assert bundles._jlens_trace_problem(str(tmp_path)) is None
