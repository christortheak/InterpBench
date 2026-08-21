"""D3 — distance from the decision boundary, named correctly.

One categorical arm's logprob readout was previously described as showing a
"ceiling effect" /
"saturation". That was the wrong word and the wrong model. A large
joint-logprob margin means the item sits far from the boundary, so the FLIP
RATE has poor sensitivity — but the log-odds keeps moving continuously and
remains a usable readout. True numerical saturation is a separate, rarer
thing: the probability hitting the log-odds clamp.

Mirror of Swift ``ChoiceMarginDiagnosticsTests``.
"""

import json
import os

from steerlab_server.experiment import choice_margin as cm


def _items(pairs):
    return [{"optionLogprobs": {"A": a, "B": b}} for a, b in pairs]


# --- the distinction the name gets wrong -------------------------------------

def test_a_wide_margin_is_distance_from_the_boundary_not_saturation():
    report = cm.diagnostics(_items([(-1, -21), (-2, -25), (-1.5, -30)]))
    text = report["interpretation"]
    assert "FLIP RATE has poor sensitivity" in text
    assert "has not saturated" in text
    # The continuous readout is still there; only the flip rate is blunt.
    assert "continuous log-odds shift" in text


def test_true_saturation_is_the_clamp_and_is_counted_separately():
    report = cm.diagnostics(_items([(0, -100), (0, -0.5)]))
    assert report["clampedItems"] == 1
    text = report["interpretation"]
    assert "clamp ARTEFACT" in text
    assert "true numerical saturation" in text


def test_a_narrow_margin_says_nothing_alarming():
    report = cm.diagnostics(_items([(-1, -1.5), (-2, -2.2), (-3, -3.1)]))
    assert "FLIP RATE" not in report["interpretation"]
    assert report["clampedItems"] == 0


# --- the arithmetic ----------------------------------------------------------

def test_the_margin_is_the_gap_between_the_top_two_options():
    assert cm.margin({"A": -1.0, "B": -3.5}) == 2.5
    # Three options: the runner-up is the second best, not the worst.
    assert cm.margin({"A": -1.0, "B": -2.0, "C": -50.0}) == 1.0


def test_a_single_option_has_no_boundary_and_therefore_no_margin():
    assert cm.margin({"A": -1.0}) is None
    assert cm.margin({}) is None
    report = cm.diagnostics([{"optionLogprobs": {"A": -1.0}}])
    assert report["scoredItems"] == 0
    assert "marginMedian" not in report


def test_quantiles_interpolate_linearly():
    # Pinned so an even-length sample's median agrees across engines.
    assert cm._quantile([1, 2, 3, 4], 0.5) == 2.5
    assert cm._quantile([1, 2, 3], 0.5) == 2.0
    assert cm._quantile([5], 0.5) == 5


# --- the bands are declared, versioned, and never gates ----------------------

def test_the_bands_travel_with_the_report_and_carry_a_version():
    report = cm.diagnostics(_items([(-1, -21)]))
    # A silent numeric cutoff deciding which items "count" is exactly the
    # unrecorded analytic choice the firewall exists to prevent.
    assert report["bandsVersion"] == cm.BANDS_VERSION
    assert report["marginBands"] == list(cm.MARGIN_BANDS)
    assert report["clampEpsilon"] == cm.LOG_ODDS_EPSILON


def test_band_counts_are_cumulative_above_each_edge():
    # Margins are 1.0, 3.0, 19.0. The bands are strict (`> band`), so an item
    # sitting exactly ON an edge is not beyond it.
    report = cm.diagnostics(_items([(-1, -2), (-1, -4), (-1, -20)]))
    beyond = report["itemsBeyondBand"]
    assert beyond["above1Nats"] == 2
    assert beyond["above2.5Nats"] == 2
    assert beyond["above5Nats"] == 1
    assert beyond["above10Nats"] == 1


def test_the_clamp_matches_the_instrument():
    # Must equal ChoiceResult.log_odds's epsilon, or incidence is counted
    # against the wrong boundary.
    assert cm.LOG_ODDS_EPSILON == 1e-12


# --- the fixture the Swift suite consumes ------------------------------------

def test_the_committed_fixture_matches_this_implementation():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Tests", "Fixtures", "cross-engine", "choice-margins.json")
    with open(path, encoding="utf-8") as handle:
        cases = json.load(handle)
    assert len(cases) >= 6
    for case in cases:
        report = cm.diagnostics(
            [{"optionLogprobs": lp} for lp in case["input"]])
        assert report["scoredItems"] == case["report"]["scoredItems"], case["label"]
        assert report.get("marginMedian") == case["report"].get("marginMedian"), \
            case["label"]
        assert report.get("clampedItems") == case["report"].get("clampedItems"), \
            case["label"]
