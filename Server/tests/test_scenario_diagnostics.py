"""D1 — keep the working that produced the accuracy number.

Validate computed every projection and the class means defining the midpoint,
then returned one number. When nine virtues all score near chance, that number
cannot distinguish "the direction does not read the concept" from "it ranks
scenarios correctly but the midpoint sits in the wrong place" — a threshold
problem, not a vector problem.

Mirror of Swift ``ScenarioDiagnosticsTests``.
"""

import json

import pytest

from steerlab_server.experiment import scenario_diagnostics as sd


def _report(projections, labels, threshold):
    scenarios = [{"id": f"s{i}", "text": f"scenario {i}"}
                 for i in range(len(projections))]
    return sd.diagnostics(
        direction=[1.0, 0.0], scenarios=scenarios, projections=projections,
        labels=labels, threshold=threshold,
        class_means={"positive": 1.0, "negative": -1.0},
        layer=3, direction_norm=1.0)


# --- the case accuracy cannot express ---------------------------------------

def test_a_perfect_ranking_with_a_bad_threshold_scores_chance_but_auc_one():
    report = _report([2.0, 1.5, -1.5, -2.0], [True, True, False, False], 5.0)
    assert report["accuracy"] == 0.5
    assert report["auc"] == 1.0
    # ...and the margins say WHY: everything sits below the boundary.
    assert all(row["margin"] < 0 for row in report["rows"])


def test_a_good_threshold_agrees_with_the_ranking():
    report = _report([2.0, 1.5, -1.5, -2.0], [True, True, False, False], 0.0)
    assert report["accuracy"] == 1.0
    assert report["auc"] == 1.0
    assert report["balancedAccuracy"] == 1.0


# --- AUC edge cases, pinned deliberately -------------------------------------

def test_ties_contribute_exactly_one_half():
    # Otherwise an all-ties direction scores 0 or 1 depending on comparison
    # order — a number that looks meaningful and is not.
    assert sd.auc([1.0, 1.0, 1.0, 1.0], [True, True, False, False]) == 0.5


def test_a_single_class_has_no_auc():
    assert sd.auc([1.0, 2.0, 3.0], [True, True, True]) is None
    assert sd.auc([], []) is None


def test_a_perfectly_inverted_direction_scores_zero():
    # Distinguishable from "no signal" (0.5): an inverted vector is a sign
    # problem, not a dead one.
    assert sd.auc([-2.0, -1.5, 1.5, 2.0], [True, True, False, False]) == 0.0


# --- the 2026-08-01 27B incident shape ---------------------------------------

def test_the_incident_shape_calibration_recovers_what_transfer_accuracy_lost():
    """The `fair` validate row: a designatedReference story-corpus midpoint
    sat below every scenario projection, so accuracy 0.50 with tp=fp and an
    empty negative row — while the ranking was nearly clean. The calibration
    re-thresholds at the held-out classes' own midpoint and reads the
    separation the transfer threshold hid."""
    report = _report([10.0, 9.0, 8.0, 3.0, 2.0, 1.0],
                     [True, True, True, False, False, False], -5.0)
    assert report["accuracy"] == 0.5
    assert report["confusion"] == {"tp": 3, "fp": 3, "tn": 0, "fn": 0}
    assert report["oneSidedPredictions"] is True
    calibration = report["heldOutCalibration"]
    assert calibration["threshold"] == 5.5
    assert calibration["classMeans"] == {"positive": 9.0, "negative": 2.0}
    assert calibration["accuracy"] == 1.0
    assert calibration["balancedAccuracy"] == 1.0
    assert calibration["confusion"] == {"tp": 3, "fp": 0, "tn": 3, "fn": 0}


def test_a_well_placed_threshold_is_not_flagged_one_sided():
    report = _report([2.0, -1.0, 1.0, -2.0], [True, True, False, False], 0.0)
    assert report["oneSidedPredictions"] is False


def test_calibration_preserves_an_inverted_sign():
    """An inverted direction must read below 0.5 here exactly as it does in
    AUC — orienting it away would flip a sign problem into respectability."""
    report = _report([-2.0, -1.5, 1.5, 2.0], [True, True, False, False], 0.0)
    calibration = report["heldOutCalibration"]
    assert calibration["accuracy"] == 0.0
    assert report["auc"] == 0.0


def test_a_single_class_has_no_calibration():
    report = sd.diagnostics(
        direction=[1.0], scenarios=[{"text": "a"}, {"text": "b"}],
        projections=[1.0, 2.0], labels=[True, True], threshold=0.0,
        class_means={}, layer=0, direction_norm=1.0)
    assert report["heldOutCalibration"] is None
    # One-sidedness is still a fact about the transfer threshold.
    assert report["oneSidedPredictions"] is True


# --- descriptive statistics only (D2) ----------------------------------------

def test_the_confusion_matrix_and_balanced_accuracy_are_reported():
    report = _report([2.0, -1.0, 1.0, -2.0], [True, True, False, False], 0.0)
    assert report["confusion"] == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}
    assert report["classCounts"] == {"positive": 2, "negative": 2}
    assert report["sensitivity"] == 0.5
    assert report["specificity"] == 0.5
    assert report["balancedAccuracy"] == 0.5


def test_the_wilson_interval_stays_inside_the_scale():
    # A normal-approximation interval on 4/4 would run past 1.0 and report an
    # impossible bound; these sets are small and often extreme.
    low, high = sd.wilson_interval(4, 4)
    assert 0 < low < 1 and high == 1.0
    low, high = sd.wilson_interval(0, 4)
    assert low == 0.0 and high < 1.0
    assert sd.wilson_interval(0, 0) is None


def test_no_inferential_statistics_are_shipped():
    """The binomial null assumes independence, and these sets are generated
    per concept by one agent over shared topics with matched pairs — both
    induce correlation, so a binomial p would understate variance. The
    inferential design is a decision to be declared, not a default."""
    text = json.dumps(_report([1.0, -1.0], [True, False], 0.0))
    assert "pValue" not in text
    assert "fdr" not in text.lower()


# --- row identity ------------------------------------------------------------

def test_every_row_carries_position_and_identity():
    report = _report([1.0, -1.0], [True, False], 0.0)
    assert [r["index"] for r in report["rows"]] == [0, 1]
    assert [r["id"] for r in report["rows"]] == ["s0", "s1"]
    # A line number alone stops meaning anything once the file is re-ordered.
    assert report["rows"][0]["rowHash"] == sd.row_hash("scenario 0", True)
    assert report["rows"][0]["rowHash"] != report["rows"][1]["rowHash"]


def test_an_id_is_synthesised_when_the_row_has_none():
    report = sd.diagnostics(
        direction=[1.0], scenarios=[{"text": "a"}], projections=[1.0],
        labels=[True], threshold=0.0, class_means={}, layer=0,
        direction_norm=1.0)
    assert report["rows"][0]["id"] == "scenario-1"


# --- the fixture the Swift suite consumes ------------------------------------

def test_the_committed_fixture_matches_this_implementation():
    """The fixture is generated FROM this module and consumed by Swift; if it
    drifts from the code that produced it, the parity test on the other side
    is checking a stale contract."""
    import os
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "Tests", "Fixtures", "cross-engine", "scenario-diagnostics.json")
    with open(path, encoding="utf-8") as handle:
        cases = json.load(handle)
    assert len(cases) >= 6
    for case in cases:
        inputs = case["input"]
        report = _report(inputs["projections"], inputs["labels"],
                         inputs["threshold"])
        assert report["accuracy"] == case["report"]["accuracy"], case["label"]
        assert report["auc"] == case["report"]["auc"], case["label"]
        assert report["balancedAccuracy"] == case["report"]["balancedAccuracy"], \
            case["label"]
        assert report["heldOutCalibration"] == \
            case["report"]["heldOutCalibration"], case["label"]
        assert report["oneSidedPredictions"] == \
            case["report"]["oneSidedPredictions"], case["label"]


# --- honesty fixes (engineer review, 2026-07-26) -----------------------------

def test_identical_text_with_opposite_labels_is_two_rows():
    """Hashing the text alone gave them one identity, making a diagnostic
    record ambiguous about which row it describes."""
    assert sd.row_hash("same words", True) != sd.row_hash("same words", False)


def test_unequal_inputs_refuse_rather_than_truncate():
    """`zip` silently dropped the tail of a longer array while Swift invented
    `False` for a missing label — the same malformed input produced two
    different answers, neither flagged."""
    with pytest.raises(ValueError, match="unequal inputs"):
        sd.diagnostics(
            direction=[1.0], scenarios=[{"text": "a"}, {"text": "b"}],
            projections=[1.0], labels=[True, False], threshold=0.0,
            class_means={}, layer=0, direction_norm=1.0)


def test_the_interval_names_its_assumption():
    """An interval is an inferential object whatever it is labelled, so the
    field states the assumption these sets violate."""
    report = _report([1.0, -1.0], [True, False], 0.0)
    assert "naiveItemLevelInterval95" in report
    assert "accuracyInterval95" not in report
