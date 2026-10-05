"""Both engines pair the same outcomes, from the same records.

``analyze`` reports one paired effect row per outcome a study measured. The
Mac engine and this one must report the SAME outcomes, under the same names
and definitions, from the same records — otherwise a study's results depend
on where it ran.

The shared fixture ``Tests/Fixtures/cross-engine/effect-outcomes.json`` holds
one record set that reaches every outcome, with the rows and the outcome list
this engine's ``analyze`` writes for it. The Mac engine reads the same file
(``EffectOutcomeCoverageTests``) and must produce the same rows. The bootstrap
interval is left out of the comparison on purpose: the two engines resample
with different generators, so its bounds agree only loosely.

Every expected value below is worked out by hand from the measurements, not
read back from either engine.
"""

import csv
import hashlib
import json
import math
import os

import pytest

from steerlab_server.experiment import analysis_endpoints, tasks
from steerlab_server.experiment.manifest import Manifest

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Tests", "Fixtures", "cross-engine", "effect-outcomes.json")

REGENERATE = ("stale fixture — re-run "
              "`Server/.venv.nosync/bin/python "
              "scripts/regenerate-cross-engine-fixtures.py` and commit")

#: The columns the fixture pins.
COLUMNS = ("condition", "endpoint", "n", "deltaMean", "wilcoxonW", "wilcoxonP",
           "adjustedP", "correction", "stratifyBy", "stratum", "unit",
           "estimand", "inference")

#: Every outcome the fixture's records reach — the one list both engines
#: must report.
OUTCOMES = ["choiceLogOdds", "choiceRate", "distinct2", "meanMonths",
            "monthsSpread", "ordinalPosition", "parsedValueMean",
            "parsedValueSpread", "readerScore:warm", "rs_hedge",
            "warmMarkerDensity", "wordCount"]

ITEM = ("item", "itemLevel", "corrected")
SAMPLE = ("sample", "withinItemSamples", "diagnostic")


def _fixture():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def _cases():
    return [pytest.param(case, id=case["label"]) for case in _fixture()["cases"]]


def _manifest(fixture, phase, *, concepts=(), exclusion_rules=()):
    manifest = {
        "name": "outcomes", "modelID": "test/model",
        "concepts": [{"name": name, "stimulusSetHash": "0" * 64}
                     for name in concepts],
        "taskPromptsFile": None,
        "conditions": [{"name": "steered",
                        "slots": [{"concept": "c", "layer": 1, "alpha": 2.0}]},
                       {"name": "steeredHigh",
                        "slots": [{"concept": "c", "layer": 1, "alpha": 4.0}]}],
        "numericParser": fixture["numericParser"],
        "reasoningStyleTaxonomyPath": fixture["taxonomy"]["path"],
        "reasoningStyleTaxonomyHash": hashlib.sha256(
            fixture["taxonomy"]["text"].encode("utf-8")).hexdigest(),
    }
    if phase is not None:
        manifest["phase"] = phase
    if exclusion_rules:
        manifest["exclusionRules"] = list(exclusion_rules)
    return manifest


def _analyze(tmp_path, records, *, phase=None, concepts=(),
             exclusion_rules=(), log=None):
    """The real entry point over ``records``, in a temporary workspace: the
    analysis directory."""
    fixture = _fixture()
    manifest = _manifest(fixture, phase, concepts=concepts,
                         exclusion_rules=exclusion_rules)
    for pinned in (fixture["parserRegistry"], fixture["taxonomy"]):
        path = tmp_path / pinned["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pinned["text"], encoding="utf-8")
    experiment_dir = tmp_path / "experiments" / "outcomes"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "experiment.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    run_dir = tmp_path / "runs" / "20261004T000000000-exp-outcomes-run"
    run_dir.mkdir(parents=True)
    (run_dir / "experiment-hash.txt").write_text(
        Manifest.from_dict(manifest).content_hash() + "\n", encoding="utf-8")
    (run_dir / "generations.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8")
    return tasks.analyze("outcomes", root=str(tmp_path),
                         log=log or (lambda _: None))


def _rows(out):
    with open(os.path.join(out, "effect-sizes.csv"), encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _coverage(out):
    with open(os.path.join(out, "outcome-coverage.json"),
              encoding="utf-8") as handle:
        return json.load(handle)


def _pinned(row):
    """One CSV row in the fixture's shape."""
    def cell(column):
        value = row[column]
        if column == "n":
            return int(value)
        if column in ("deltaMean", "wilcoxonW", "wilcoxonP", "adjustedP"):
            return float(value) if value else None
        return value
    return {column: cell(column) for column in COLUMNS}


def _one(rows, outcome, stratify_by="pooled", stratum="", condition="steered"):
    [row] = [r for r in rows
             if (r["endpoint"], r["stratifyBy"], r["stratum"], r["condition"])
             == (outcome, stratify_by, stratum, condition)]
    return row


def _none(rows, outcome, stratify_by="pooled", stratum="", condition="steered"):
    return not [r for r in rows
                if (r["endpoint"], r["stratifyBy"], r["stratum"],
                    r["condition"]) == (outcome, stratify_by, stratum, condition)]


def _two_sided_p(w, n, tie_correction=0.0):
    """The normal-approximation p of a signed-rank W, written out from the
    textbook formula rather than through the engine: mean n(n+1)/4, variance
    n(n+1)(2n+1)/24 less the tie correction, continuity-corrected. ``n``
    counts the NONZERO differences."""
    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24 - tie_correction / 48
    z = (w - mean + 0.5) / math.sqrt(variance)
    return min(1.0, 1 + math.erf(z / math.sqrt(2)))


def _check(row, *, n, mean, w, p, stamps, adjusted="raw", correction="bh"):
    """One row against its hand-computed values. ``w``/``p`` are None when
    every difference is zero (the test is undefined, the cells empty).
    ``adjusted`` is "raw" for a correction family of one, None for a row
    held out of every family, else the adjusted p."""
    assert row["n"] == str(n)
    # The file carries six significant digits.
    assert float(row["deltaMean"]) == pytest.approx(mean, rel=5e-6, abs=1e-12)
    if w is None:
        assert (row["wilcoxonW"], row["wilcoxonP"]) == ("", "")
    else:
        assert float(row["wilcoxonW"]) == w
        assert float(row["wilcoxonP"]) == pytest.approx(p, abs=5e-7)
    assert (row["unit"], row["estimand"], row["inference"]) == stamps
    if stamps == SAMPLE:
        # A within-item row is a diagnostic: no adjusted p, no family.
        assert (row["adjustedP"], row["correction"]) == ("", "")
        return
    assert row["correction"] == correction
    if w is None:
        assert row["adjustedP"] == ""
    elif adjusted == "raw":
        assert row["adjustedP"] == row["wilcoxonP"]
    else:
        assert float(row["adjustedP"]) == pytest.approx(adjusted, abs=1e-6)


POOLED = ("", "", "")


# --- the fixture is what this engine writes -----------------------------------

@pytest.mark.parametrize("case", _cases())
def test_fixture_rows_are_what_analyze_writes(tmp_path, case):
    fixture = _fixture()
    out = _analyze(tmp_path, fixture["records"], phase=case["phase"],
                   exclusion_rules=case["exclusionRules"])
    assert [_pinned(row) for row in _rows(out)] == case["effectRows"], REGENERATE
    assert _coverage(out) == case["outcomeCoverage"], REGENERATE


def test_fixture_carries_the_one_definition_of_each_outcome():
    """The definitions in words are the fixture's: this engine's text and the
    Mac engine's are both held to it."""
    fixture = _fixture()
    assert fixture["outcomeFamilies"] == analysis_endpoints.OUTCOME_FAMILIES, \
        REGENERATE
    ids = [family["id"] for family in fixture["outcomeFamilies"]]
    assert len(ids) == len(set(ids)) == 12
    assert all(family["definition"].strip() and family["name"]
               for family in fixture["outcomeFamilies"])


@pytest.mark.parametrize("case", _cases())
def test_every_outcome_reaches_the_effect_rows(case):
    """One record set, every outcome: nothing an engine pairs is missing
    from the list, and the coverage file says the same thing the rows do."""
    assert sorted({row["endpoint"] for row in case["effectRows"]}) == OUTCOMES
    coverage = case["outcomeCoverage"]
    assert coverage["schemaVersion"] == 1
    assert coverage["pairing"] == analysis_endpoints.PAIRING_DEFINITION
    assert [entry["name"] for entry in coverage["outcomes"]] == OUTCOMES
    definitions = {family["id"]: family["definition"]
                   for family in analysis_endpoints.OUTCOME_FAMILIES}
    families = {"readerScore:warm": "readerScore", "rs_hedge": "reasoningStyle",
                "warmMarkerDensity": "markerDensity"}
    for entry in coverage["outcomes"]:
        assert entry["status"] == "computed"
        assert entry["family"] == families.get(entry["name"], entry["name"])
        assert entry["definition"] == definitions[entry["family"]]
        assert "reason" not in entry


def _case(label):
    [case] = [c for c in _fixture()["cases"] if c["label"] == label]
    return case


def test_the_fixture_holds_three_readings_of_one_record_set():
    assert [(case["label"], case["phase"], case["exclusionRules"])
            for case in _fixture()["cases"]] == [
        ("unphased", None, []), ("confirm", "confirm", []),
        ("excluded", None, [{"rule": "unparseableEndpoint"}])]


def test_the_phase_changes_only_the_correction():
    """Same records, same estimates; an unphased study is corrected by
    Benjamini–Hochberg and a confirm-phase one by Holm."""
    unphased, confirm = _case("unphased"), _case("confirm")
    assert len(unphased["effectRows"]) == len(confirm["effectRows"]) == 97

    def without_correction(rows):
        return [{k: v for k, v in row.items()
                 if k not in ("adjustedP", "correction")} for row in rows]
    assert without_correction(unphased["effectRows"]) \
        == without_correction(confirm["effectRows"])
    for mine, theirs in zip(unphased["effectRows"], confirm["effectRows"]):
        if mine["inference"] == "diagnostic":
            assert (mine["correction"], theirs["correction"]) == ("", "")
        else:
            assert (mine["correction"], theirs["correction"]) == ("bh", "holm")


# --- marker density: the outcome this engine gained ----------------------------

def test_marker_density_matches_the_hand_computation(tmp_path):
    """``warmMarkerDensity``: the recorded marker density, averaged over an
    item's responses, paired to the same item's baseline.

    Three responses per cell (baseline | steered):

        item-1  0     0     0     → 0      |  0.25  0.5   0.75  → 0.5    +0.5
        item-2  0.125 0.125 0.125 → 0.125  |  0.125 0.25  0.375 → 0.25   +0.125
        item-3  0.25  0     0.5   → 0.25   |  0.25  0.25  0.25  → 0.25    0
        item-4  0.5   0.5   0.5   → 0.5    |  0.25  0.25  0.25  → 0.25   −0.25

    Four paired items, mean (0.5 + 0.125 + 0 − 0.25) / 4 = 0.09375. The zero
    drops out of the signed-rank test; |0.125| < |−0.25| < |0.5| rank 1, 2,
    3, so W+ = 4, W− = 2, W = 2 over three differences.
    """
    rows = _rows(_analyze(tmp_path, _fixture()["records"]))

    _check(_one(rows, "warmMarkerDensity"), n=4, mean=0.09375, w=2.0,
           p=_two_sided_p(2, 3), stamps=POOLED)            # p = 0.789268
    # Only "steered" has sampled responses: a family of one.
    assert _none(rows, "warmMarkerDensity", condition="steeredHigh")

    # Arm x = {item-1, item-3}: +0.5 and 0 → mean 0.25; one nonzero
    # difference, W = 0, z = 0. Arm y = {item-2, item-4}: +0.125 and −0.25 →
    # mean −0.0625; ranks 1 (+) and 2 (−), W = 1, z = 0. Both p = 1, so the
    # Benjamini–Hochberg pair stays at 1.
    _check(_one(rows, "warmMarkerDensity", "arm", "x"), n=2, mean=0.25,
           w=0.0, p=1.0, stamps=ITEM, adjusted=1.0)
    _check(_one(rows, "warmMarkerDensity", "arm", "y"), n=2, mean=-0.0625,
           w=1.0, p=1.0, stamps=ITEM, adjusted=1.0)

    # A one-item stratum drops to the sample axis: sample k minus baseline
    # sample k.
    within = {
        "item-1": (0.5, 0.0, _two_sided_p(0, 3)),         # +0.25 +0.5 +0.75
        "item-2": (0.125, 0.0, _two_sided_p(0, 2)),       # 0 +0.125 +0.25
        "item-3": (0.0, 1.5, 1.0),                        # 0 +0.25 −0.25: tie
        "item-4": (-0.25, 0.0, _two_sided_p(0, 3, 24)),   # −0.25 three times
    }
    for item, (mean, w, p) in within.items():
        _check(_one(rows, "warmMarkerDensity", "promptID", item), n=3,
               mean=mean, w=w, p=p, stamps=SAMPLE)


def _response(condition, item, words, density=..., sample=0):
    record = {"condition": condition, "seed": 1, "sampleIndex": sample,
              "promptIndex": 1, "promptID": item, "wordCount": words,
              "distinct2": 0.5}
    if density is not ...:
        record["markerDensity"] = density
    return record


def test_a_response_whose_record_names_no_value_counts_as_zero():
    """The Mac engine's rule, mirrored: once a run recorded marker density
    for ANY concept, every sampled response counts toward every concept —
    zero where its record names none.

        p1  baseline {warm 0.25}            steered {warm 0.75, calm 0.5}
        p2  baseline (no markerDensity)     steered {calm 0.25}

    warm: p1 0.75 − 0.25 = +0.5; p2 0 − 0 = 0.
    calm: p1 0.5 − 0 = +0.5; p2 0.25 − 0 = +0.25.
    """
    records = [
        _response("baseline", "p1", 10, {"warm": 0.25}),
        _response("steered", "p1", 12, {"warm": 0.75, "calm": 0.5}),
        _response("baseline", "p2", 10),
        _response("steered", "p2", 12, {"calm": 0.25}),
        # An answer-token readout is not a sampled response: it never
        # counts toward the mean, whatever it carries.
        {"condition": "steered", "promptID": "p1",
         "instrument": "answerTokenLogprob", "markerDensity": {"warm": 9.0}},
    ]
    assert analysis_endpoints.marker_density_concepts(records) \
        == ["calm", "warm"]
    endpoints = analysis_endpoints.endpoint_values(records)
    assert endpoints["warmMarkerDensity"] == {
        "baseline": {"p1": 0.25, "p2": 0.0},
        "steered": {"p1": 0.75, "p2": 0.0}}
    assert endpoints["calmMarkerDensity"] == {
        "baseline": {"p1": 0.0, "p2": 0.0},
        "steered": {"p1": 0.5, "p2": 0.25}}


def test_marker_density_is_read_at_the_mac_engines_precision():
    """The Mac engine measures and stores marker density as a 32-bit float,
    and reads it back as one. 0.1 is not a 32-bit float: the nearest one is
    13421773 / 2**27 = 0.100000001490116…, and that is the value both
    engines must pair."""
    records = [_response("baseline", "p1", 10, {"warm": 0.1})]
    [value] = analysis_endpoints.endpoint_values(
        records)["warmMarkerDensity"]["baseline"].values()
    assert value == 13421773 / 2 ** 27
    assert value != 0.1


def test_a_run_of_this_engine_says_marker_density_is_not_available(tmp_path):
    """This engine's own run records no marker density. The study below
    declares the concept "warm" and its records carry no ``markerDensity``:
    the analysis must SAY the outcome is not available — in the coverage
    file and in its log — and never leave a silently missing row."""
    fixture = _fixture()
    records = [{key: value for key, value in record.items()
                if key != "markerDensity"} for record in fixture["records"]]
    lines = []
    out = _analyze(tmp_path, records, concepts=("warm",), log=lines.append)

    rows = _rows(out)
    assert sorted({row["endpoint"] for row in rows}) \
        == [name for name in OUTCOMES if name != "warmMarkerDensity"]

    coverage = _coverage(out)
    assert [entry["name"] for entry in coverage["outcomes"]] == OUTCOMES
    [missing] = [entry for entry in coverage["outcomes"]
                 if entry["status"] != "computed"]
    assert missing == {
        "name": "warmMarkerDensity", "family": "markerDensity",
        "status": "notAvailable",
        "definition": next(family["definition"]
                           for family in analysis_endpoints.OUTCOME_FAMILIES
                           if family["id"] == "markerDensity"),
        "reason": analysis_endpoints.MARKER_DENSITY_NOT_RECORDED,
    }
    assert missing["reason"].startswith("Not available on this engine: ")
    assert ("warmMarkerDensity: "
            + analysis_endpoints.MARKER_DENSITY_NOT_RECORDED) in lines

    # Every other row is exactly what the same records analyze to WITH the
    # marker density: gaining an outcome moved no other number.
    with_density = [_pinned(row) for row in _rows(_analyze(
        tmp_path / "with-density", fixture["records"]))]
    assert [_pinned(row) for row in rows] == [
        row for row in with_density if row["endpoint"] != "warmMarkerDensity"]


def test_marker_density_is_not_called_missing_where_none_was_due():
    """No declared concept, or no sampled response, or a run that DID record
    (even an empty object): there is no marker density to be missing."""
    sampled = [_response("baseline", "p1", 10), _response("steered", "p1", 12)]
    not_recorded = analysis_endpoints.marker_density_not_recorded
    assert not_recorded(sampled, ["warm", "calm"]) == ["warm", "calm"]
    assert not_recorded(sampled, []) == []
    assert not_recorded(
        [{"condition": "baseline", "promptID": "p1",
          "instrument": "answerTokenLogprob"}], ["warm"]) == []
    recorded = [_response("baseline", "p1", 10, {}),
                _response("steered", "p1", 12, {})]
    assert not_recorded(recorded, ["warm"]) == []


def test_outcome_names_resolve_to_their_family():
    family = analysis_endpoints.outcome_family
    for name in ("wordCount", "distinct2", "choiceRate", "meanMonths",
                 "monthsSpread", "parsedValueMean", "parsedValueSpread",
                 "choiceLogOdds", "ordinalPosition"):
        assert family(name) == name
    assert family("rs_hedge") == "reasoningStyle"
    assert family("readerScore:warm") == "readerScore"
    assert family("warmMarkerDensity", ["warm"]) == "markerDensity"
    # A marker-density name is known by its concept, not its suffix: a
    # reasoning-style feature that happens to end the same way keeps its
    # own family.
    assert family("rs_toneMarkerDensity", ["warm"]) == "reasoningStyle"
    assert family("rs_toneMarkerDensity", ["rs_tone"]) == "markerDensity"
    # A family id is not an outcome name.
    for name in ("markerDensity", "reasoningStyle", "readerScore", "rs_",
                 "readerScore:", "unheardOf"):
        assert family(name) == ""


# --- the outcomes the Mac engine gained, by hand, from the same records ------

def test_choice_log_odds_matches_the_hand_computation(tmp_path):
    """``choiceLogOdds``: the log-odds of the declared target, one readout
    per item and condition.

                baseline   steered   Δ       steeredHigh   Δ
        item-1   −1.0       0.5     +1.5      1.0         +2.0
        item-2    0.25      0.75    +0.5      1.25        +1.0
        item-3    2.0       1.0     −1.0      1.5         −0.5
        item-4   −0.5       2.0     +2.5      2.5         +3.0
        item-5   declares no target: no value, on either engine

    steered: mean (1.5 + 0.5 − 1 + 2.5) / 4 = 0.875; the one negative
    difference has rank 2, so W = 2. steeredHigh: mean (2 + 1 − 0.5 + 3) / 4
    = 1.375; the negative difference has rank 1, so W = 1.
    """
    fixture = _fixture()
    p_steered, p_high = _two_sided_p(2, 4), _two_sided_p(1, 4)
    assert (round(p_steered, 6), round(p_high, 6)) == (0.361310, 0.201243)

    unphased = _rows(_analyze(tmp_path / "bh", fixture["records"]))
    # Benjamini–Hochberg over the two conditions: the smaller p doubles to
    # 0.402485, which exceeds the larger, so both rows take the larger.
    _check(_one(unphased, "choiceLogOdds"), n=4, mean=0.875, w=2.0,
           p=p_steered, stamps=POOLED, adjusted=p_steered)
    _check(_one(unphased, "choiceLogOdds", condition="steeredHigh"), n=4,
           mean=1.375, w=1.0, p=p_high, stamps=POOLED, adjusted=p_steered)

    confirm = _rows(_analyze(tmp_path / "holm", fixture["records"],
                             phase="confirm"))
    # Holm: the smaller p doubles, and the larger may not fall below it.
    for condition, (mean, w, p) in {"steered": (0.875, 2.0, p_steered),
                                    "steeredHigh": (1.375, 1.0, p_high)}.items():
        _check(_one(confirm, "choiceLogOdds", condition=condition), n=4,
               mean=mean, w=w, p=p, stamps=POOLED, adjusted=2 * p_high,
               correction="holm")

    # Arm x = {item-1, item-3}: one difference of each sign, the negative
    # one smaller, so W = 1 and z = 0. Arm y = {item-2, item-4}: both
    # positive, W = 0. The arm family holds all four rows (two conditions ×
    # two arms): p = 1, 0.371093, 1, 0.371093.
    p_same_sign = _two_sided_p(0, 2)
    arm = {("x", "steered"): (0.25, 1.0, 1.0),          # +1.5, −1.0
           ("y", "steered"): (1.5, 0.0, p_same_sign),   # +0.5, +2.5
           ("x", "steeredHigh"): (0.75, 1.0, 1.0),      # +2.0, −0.5
           ("y", "steeredHigh"): (2.0, 0.0, p_same_sign)}   # +1.0, +3.0
    for (stratum, condition), (mean, w, p) in arm.items():
        # Benjamini–Hochberg, m = 4: the two 0.371093s take rank 2 →
        # 0.371093 × 4 / 2. Holm: 0.371093 × 4 > 1, so every row is 1.
        _check(_one(unphased, "choiceLogOdds", "arm", stratum, condition),
               n=2, mean=mean, w=w, p=p, stamps=ITEM,
               adjusted=1.0 if stratum == "x" else 2 * p_same_sign)
        _check(_one(confirm, "choiceLogOdds", "arm", stratum, condition),
               n=2, mean=mean, w=w, p=p, stamps=ITEM, adjusted=1.0,
               correction="holm")

    # One readout per item: an item's own stratum is ONE difference, never
    # a within-item row (there is nothing sampled to pair).
    own = {"item-1": (1.5, 2.0), "item-2": (0.5, 1.0),
           "item-3": (-1.0, -0.5), "item-4": (2.5, 3.0)}
    for item, means in own.items():
        for condition, mean in zip(("steered", "steeredHigh"), means):
            _check(_one(unphased, "choiceLogOdds", "promptID", item, condition),
                   n=1, mean=mean, w=0.0, p=1.0, stamps=ITEM, adjusted=1.0)
    # The rating item reaches the scale position and not the log-odds.
    for condition in ("steered", "steeredHigh"):
        assert _none(unphased, "choiceLogOdds", "promptID", "item-5", condition)
        assert _one(unphased, "ordinalPosition", "promptID", "item-5",
                    condition)["n"] == "1"
    assert _one(unphased, "ordinalPosition")["n"] == "5"


def test_choice_rate_matches_the_hand_computation(tmp_path):
    """``choiceRate``: the share of an item's responses that chose the
    target "A", among those with a readable choice ("–" is unreadable).

        item-1   A B –  → 1/2    |  A A A → 1      +0.5
        item-2   B B B  → 0      |  A B – → 1/2    +0.5
        item-3   A A A  → 1      |  B – A → 1/2    −0.5
        item-4   – – –  → none   |  A A A → 1      the item does not pair

    Three paired items (not four), mean (0.5 + 0.5 − 0.5) / 3 = 1/6. All
    three differences tie at rank 2: W+ = 4, W− = 2, W = 2.
    """
    rows = _rows(_analyze(tmp_path, _fixture()["records"]))
    _check(_one(rows, "choiceRate"), n=3, mean=1 / 6, w=2.0,
           p=_two_sided_p(2, 3, 24), stamps=POOLED)        # p = 0.772830

    # Arm x = {item-1, item-3}: +0.5 and −0.5 tie, W = 1.5, p clamps to 1.
    _check(_one(rows, "choiceRate", "arm", "x"), n=2, mean=0.0, w=1.5,
           p=1.0, stamps=ITEM)
    # Arm y = {item-2, item-4}: only item-2 pairs. ONE paired item drops
    # to that item's samples — sample 0 (A against B: 1 − 0) and sample 1
    # (B against B: 0 − 0); sample 2 is unreadable — and the row is a
    # diagnostic, so arm x above is a correction family of one.
    _check(_one(rows, "choiceRate", "arm", "y"), n=2, mean=0.5, w=0.0,
           p=1.0, stamps=SAMPLE)

    within = {"item-1": 0.5,      # sample 0: 1 − 1; sample 1: 1 − 0
              "item-2": 0.5,      # sample 0: 1 − 0; sample 1: 0 − 0
              "item-3": -0.5}     # sample 0: 0 − 1; sample 2: 1 − 1
    for item, mean in within.items():
        _check(_one(rows, "choiceRate", "promptID", item), n=2, mean=mean,
               w=0.0, p=1.0, stamps=SAMPLE)
    assert _none(rows, "choiceRate", "promptID", "item-4")


def test_parsed_numbers_match_the_hand_computation(tmp_path):
    """``meanMonths`` and ``monthsSpread`` — and, because the study's parser
    reads a percentage rather than a duration, the same values again as
    ``parsedValueMean`` and ``parsedValueSpread``.

    The parsed numbers ("–" is a response the parser could not read), each
    cell's mean and sample standard deviation:

        item-1   10 20 30 → 20, 10   |  30 40 50   → 40, 10      +20     0
        item-2   50 60 70 → 60, 10   |  55 –  75   → 65, √200    +5      √200 − 10
        item-3   20 40 60 → 40, 20   |  30 30 30   → 30, 0       −10     −20
        item-4   80 –  –  → 80, none |  90 100 110 → 100, 10     +20     does not pair

    Mean: four items, (20 + 5 − 10 + 20) / 4 = 8.75; ranks 1 (5), 2 (−10),
    and 3.5 twice (the two 20s), so W− = 2. Spread: three items, since a
    cell with one readable response has no spread; the zero drops, and the
    two that remain have ranks 1 (+) and 2 (−), so W = 1 and z = 0.
    """
    rows = _rows(_analyze(tmp_path, _fixture()["records"]))
    spread_2 = math.sqrt(200) - 10                         # 4.142136
    for mean_name, spread_name in (("meanMonths", "monthsSpread"),
                                   ("parsedValueMean", "parsedValueSpread")):
        _check(_one(rows, mean_name), n=4, mean=8.75, w=2.0,
               p=_two_sided_p(2, 4, 6), stamps=POOLED)     # p = 0.357273
        _check(_one(rows, spread_name), n=3, mean=(math.sqrt(200) - 30) / 3,
               w=1.0, p=1.0, stamps=POOLED)                # mean −5.285955

        # Arm x = {item-1, item-3}: +20, −10 → W = 1, z = 0. Arm y =
        # {item-2, item-4}: +5, +20 → W = 0. Benjamini–Hochberg over the
        # pair doubles the smaller p.
        _check(_one(rows, mean_name, "arm", "x"), n=2, mean=5.0, w=1.0,
               p=1.0, stamps=ITEM, adjusted=1.0)
        _check(_one(rows, mean_name, "arm", "y"), n=2, mean=12.5, w=0.0,
               p=_two_sided_p(0, 2), stamps=ITEM,
               adjusted=2 * _two_sided_p(0, 2))
        # Spread, arm x: 0 and −20 → one nonzero difference. Arm y: only
        # item-2 pairs, and a spread has no sample axis to drop to — it
        # stays ONE item-level difference.
        _check(_one(rows, spread_name, "arm", "x"), n=2, mean=-10.0, w=0.0,
               p=1.0, stamps=ITEM, adjusted=1.0)
        _check(_one(rows, spread_name, "arm", "y"), n=1, mean=spread_2,
               w=0.0, p=1.0, stamps=ITEM, adjusted=1.0)

        # An item's own spread: one difference. item-1's is exactly zero,
        # so its test is undefined; item-4 has none.
        _check(_one(rows, spread_name, "promptID", "item-1"), n=1, mean=0.0,
               w=None, p=None, stamps=ITEM)
        _check(_one(rows, spread_name, "promptID", "item-2"), n=1,
               mean=spread_2, w=0.0, p=1.0, stamps=ITEM, adjusted=1.0)
        _check(_one(rows, spread_name, "promptID", "item-3"), n=1,
               mean=-20.0, w=0.0, p=1.0, stamps=ITEM, adjusted=1.0)
        assert _none(rows, spread_name, "promptID", "item-4")

    # An item's own mean, under the months name, drops to its samples when
    # two of them were read on both sides.
    _check(_one(rows, "meanMonths", "promptID", "item-1"), n=3, mean=20.0,
           w=0.0, p=_two_sided_p(0, 3, 24), stamps=SAMPLE)   # +20 +20 +20
    _check(_one(rows, "meanMonths", "promptID", "item-2"), n=2, mean=5.0,
           w=0.0, p=_two_sided_p(0, 2, 6), stamps=SAMPLE)    # +5 · +5
    _check(_one(rows, "meanMonths", "promptID", "item-3"), n=3, mean=-10.0,
           w=1.5, p=_two_sided_p(1.5, 3, 6), stamps=SAMPLE)  # +10 −10 −30
    # item-4: one sample read on both sides is not enough — it stays the
    # one item-level difference, 100 − 80, and the only corrected row of
    # its family.
    _check(_one(rows, "meanMonths", "promptID", "item-4"), n=1, mean=20.0,
           w=0.0, p=1.0, stamps=ITEM)
    # Under the neutral name an item's own row is ALWAYS the one item-level
    # difference (this engine has never read that name sample by sample).
    for item, mean in {"item-1": 20.0, "item-2": 5.0, "item-3": -10.0,
                       "item-4": 20.0}.items():
        _check(_one(rows, "parsedValueMean", "promptID", item), n=1,
               mean=mean, w=0.0, p=1.0, stamps=ITEM, adjusted=1.0)


def test_reader_scores_match_the_hand_computation(tmp_path):
    """``readerScore:warm``: the recorded reader score, averaged over the
    responses that carry one.

        item-1    1    1.5  2   → 1.5   |   2    2.5   3    → 2.5     +1
        item-2    0.25 0.75 ·   → 0.5   |   1    1     1    → 1       +0.5
        item-3   −1    0    1   → 0     |  −1   −0.75 −0.5  → −0.75   −0.75
        item-4    2    2    2   → 2     |   4    4.5   5    → 4.5     +2.5

    (item-2's third baseline record carries no score, so its cell averages
    two.) Mean (1 + 0.5 − 0.75 + 2.5) / 4 = 0.8125; the negative difference
    has rank 2, so W = 2.
    """
    rows = _rows(_analyze(tmp_path, _fixture()["records"]))
    _check(_one(rows, "readerScore:warm"), n=4, mean=0.8125, w=2.0,
           p=_two_sided_p(2, 4), stamps=POOLED)            # p = 0.361310
    # Arm x = {item-1, item-3}: +1, −0.75 → W = 1. Arm y = {item-2,
    # item-4}: +0.5, +2.5 → W = 0.
    _check(_one(rows, "readerScore:warm", "arm", "x"), n=2, mean=0.125,
           w=1.0, p=1.0, stamps=ITEM, adjusted=1.0)
    _check(_one(rows, "readerScore:warm", "arm", "y"), n=2, mean=1.5, w=0.0,
           p=_two_sided_p(0, 2), stamps=ITEM,
           adjusted=2 * _two_sided_p(0, 2))
    within = {
        "item-1": (3, 1.0, _two_sided_p(0, 3, 24)),    # +1 +1 +1
        "item-2": (2, 0.5, _two_sided_p(0, 2)),        # +0.75 +0.25 ·
        "item-3": (3, -0.75, _two_sided_p(0, 2)),      # 0 −0.75 −1.5
        "item-4": (3, 2.5, _two_sided_p(0, 3)),        # +2 +2.5 +3
    }
    for item, (n, mean, p) in within.items():
        _check(_one(rows, "readerScore:warm", "promptID", item), n=n,
               mean=mean, w=0.0, p=p, stamps=SAMPLE)


# --- a declared exclusion leaves the same responses out of every outcome ------

def test_a_declared_exclusion_reaches_every_outcome(tmp_path):
    """The study declares ``unparseableEndpoint``: a response the numeric
    parser could not read is left out — of EVERY outcome, not only the
    parsed number. Three responses go: item-2's second steered response, and
    item-4's second and third baseline responses.

    wordCount (the excluded responses in brackets):

        item-2   20 20 23 → 21      |  22 [25] 25 → 23.5    +2.5  (was +3)
        item-4   40 [41] [45] → 40  |  50 52 54   → 52      +12   (was +10)

    so the mean is (6 + 2.5 − 1 + 12) / 4 = 4.875.

    choiceRate: item-2's steered cell loses its "B" and keeps "A" and an
    unreadable response → 1 of 1; baseline 0 → +1 (was +0.5). item-4's
    baseline had no readable choice and still has none. Mean (0.5 + 1 − 0.5)
    / 3 = 1/3; the two 0.5s tie at rank 1.5, so W− = 1.5.

    The parsed number itself does not move: the responses that went had no
    number to contribute. The answer-token readouts carry no parsed number,
    so the rule does not touch them.
    """
    out = _analyze(tmp_path, _fixture()["records"],
                   exclusion_rules=[{"rule": "unparseableEndpoint"}])
    with open(os.path.join(out, "exclusions.json"), encoding="utf-8") as handle:
        assert json.load(handle)["excludedRecords"] == 3
    rows = _rows(out)

    _check(_one(rows, "wordCount"), n=4, mean=4.875, w=1.0,
           p=_two_sided_p(1, 4), stamps=POOLED)
    _check(_one(rows, "choiceRate"), n=3, mean=1 / 3, w=1.5,
           p=_two_sided_p(1.5, 3, 6), stamps=POOLED)       # p = 0.586214
    _check(_one(rows, "meanMonths"), n=4, mean=8.75, w=2.0,
           p=_two_sided_p(2, 4, 6), stamps=POOLED)
    _check(_one(rows, "readerScore:warm"), n=4, mean=0.8125, w=2.0,
           p=_two_sided_p(2, 4), stamps=POOLED)
    # item-2's steered marker densities 0.125 [0.25] 0.375 still average
    # 0.25, so this mean happens not to move either.
    _check(_one(rows, "warmMarkerDensity"), n=4, mean=0.09375, w=2.0,
           p=_two_sided_p(2, 3), stamps=POOLED)
    _check(_one(rows, "choiceLogOdds"), n=4, mean=0.875, w=2.0,
           p=_two_sided_p(2, 4), stamps=POOLED, adjusted=_two_sided_p(2, 4))

    # item-2's own word-count row pairs the two samples both arms still
    # have: 22 − 20 and 25 − 23. item-4 has one such sample, which is not
    # enough for a within-item row: it stays the one item-level difference.
    _check(_one(rows, "wordCount", "promptID", "item-2"), n=2, mean=2.0,
           w=0.0, p=_two_sided_p(0, 2, 6), stamps=SAMPLE)
    _check(_one(rows, "wordCount", "promptID", "item-4"), n=1, mean=12.0,
           w=0.0, p=1.0, stamps=ITEM)
    # item-2's choice rate has ONE sample readable on both sides now (A
    # against B), so it too is the item-level difference, 1 − 0.
    _check(_one(rows, "choiceRate", "promptID", "item-2"), n=1, mean=1.0,
           w=0.0, p=1.0, stamps=ITEM)
    _check(_one(rows, "choiceRate", "arm", "y"), n=1, mean=1.0, w=0.0,
           p=1.0, stamps=ITEM, adjusted=1.0)
