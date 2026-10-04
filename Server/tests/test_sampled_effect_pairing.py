"""Paired effects when a study samples several responses per item.

The unit of analysis is the ITEM. ``analyze`` averages each (condition, item)
cell over its samples and pairs the treatment mean to the same item's
baseline mean — by promptID, never by seed. A derived seed includes the
condition name, so the two records of a pair carry different seeds by design;
an analysis that joined on the seed would find no pairs at all, and one that
treated every (seed, item) pair as its own observation would report three
times the items it measured.

The shared fixture ``Tests/Fixtures/cross-engine/sampled-effect-pairing.json``
holds one set of measurements under both seed policies, with the rows this
engine's ``analyze`` writes for them. The Mac engine reads the same file
(``SampledEffectPairingTests``) and must produce the same rows.
"""

import csv
import json
import math
import os

import pytest

from steerlab_server.experiment import sampling, tasks
from steerlab_server.experiment.manifest import Manifest

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Tests", "Fixtures", "cross-engine", "sampled-effect-pairing.json")

REGENERATE = ("stale fixture — re-run "
              "`Server/.venv.nosync/bin/python "
              "scripts/regenerate-cross-engine-fixtures.py` and commit")

#: The columns the fixture pins (the bootstrap interval is left out: the two
#: engines resample with different generators).
COLUMNS = ("condition", "endpoint", "n", "deltaMean", "wilcoxonW", "wilcoxonP",
           "adjustedP", "correction", "stratifyBy", "stratum", "unit",
           "estimand", "inference")

ITEMS = ("item-1", "item-2", "item-3", "item-4")


def _fixture():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def _cases():
    return [pytest.param(case, id=case["label"]) for case in _fixture()["cases"]]


def _analyze(tmp_path, records):
    """The real entry point over ``records``: effect-sizes.csv as dict rows."""
    manifest = {
        "name": "pairing", "modelID": "test/model", "concepts": [],
        "taskPromptsFile": None,
        "conditions": [{"name": "steered",
                        "slots": [{"concept": "c", "layer": 1, "alpha": 2.0}]}],
    }
    experiment_dir = tmp_path / "experiments" / "pairing"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "experiment.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    run_dir = tmp_path / "runs" / "20261004T000000000-exp-pairing-run"
    run_dir.mkdir(parents=True)
    (run_dir / "experiment-hash.txt").write_text(
        Manifest.from_dict(manifest).content_hash() + "\n", encoding="utf-8")
    (run_dir / "generations.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8")
    out = tasks.analyze("pairing", root=str(tmp_path), log=lambda _: None)
    with open(os.path.join(out, "effect-sizes.csv"), encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _one(rows, **match):
    [row] = [r for r in rows if all(r[k] == v for k, v in match.items())]
    return row


def _two_sided_p(w, n, tie_correction=0.0):
    """The normal-approximation p of a signed-rank W, written out from the
    textbook formula rather than through the engine: mean n(n+1)/4, variance
    n(n+1)(2n+1)/24 less the tie correction, continuity-corrected."""
    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24 - tie_correction / 48
    z = (w - mean + 0.5) / math.sqrt(variance)
    return min(1.0, 1 + math.erf(z / math.sqrt(2)))


@pytest.mark.parametrize("case", _cases())
def test_fixture_rows_are_what_analyze_writes(tmp_path, case):
    rows = [_pinned(row) for row in _analyze(tmp_path, case["records"])]
    assert rows == case["effectRows"], REGENERATE


def test_fixture_seeds_follow_their_policy():
    """The derived case carries the engine's own derived seeds, and they never
    agree across the two arms of a pair; the manifest case shares them."""
    fixture = _fixture()
    by_label = {case["label"]: case for case in fixture["cases"]}

    derived = by_label["derivedSHA256"]["records"]
    for record in derived:
        assert record["seed"] == sampling.derive_seed(
            fixture["experimentHash"], record["condition"],
            record["promptID"], record["sampleIndex"]), REGENERATE
    seeds = {(r["condition"], r["promptID"], r["sampleIndex"]): r["seed"]
             for r in derived}
    assert len(seeds) == 24
    for prompt_id in ITEMS:
        for sample in range(3):
            assert seeds[("baseline", prompt_id, sample)] \
                != seeds[("steered", prompt_id, sample)]
    # Stronger: no steered record shares a (seed, item) key with ANY baseline
    # record, so a join on that key would be empty.
    baseline_keys = {(r["seed"], r["promptID"]) for r in derived
                     if r["condition"] == "baseline"}
    assert not any((r["seed"], r["promptID"]) in baseline_keys
                   for r in derived if r["condition"] == "steered")

    shared = by_label["manifestSeeds"]
    for record in shared["records"]:
        assert record["seed"] == shared["manifestSeeds"][record["sampleIndex"]]


def test_both_seed_policies_analyze_to_the_same_rows():
    derived, shared = _fixture()["cases"]
    assert derived["effectRows"] == shared["effectRows"]

    def measurements(case):
        return [{key: value for key, value in record.items()
                 if key not in ("seed", "seedPolicy")}
                for record in case["records"]]
    assert measurements(derived) == measurements(shared)


@pytest.mark.parametrize("case", _cases())
def test_paired_differences_match_the_hand_computation(tmp_path, case):
    """Expected values worked by hand from the measurements, independently of
    the engine.

    wordCount, three samples per cell (baseline | steered):

        item-1  10 12 14 → 12  |  15 18 21 → 18   difference  +6
        item-2  20 20 23 → 21  |  22 25 25 → 24   difference  +3
        item-3  30 33 36 → 33  |  31 32 33 → 32   difference  −1
        item-4  40 41 45 → 42  |  50 52 54 → 52   difference +10

    Four paired items, mean (6 + 3 − 1 + 10) / 4 = 4.5. Signed ranks of
    |−1| < |3| < |6| < |10| are 1, 2, 3, 4, so W− = 1, W+ = 9, W = 1.
    """
    # The cell means and per-item differences, straight from the records.
    cells = {}
    for record in case["records"]:
        cells.setdefault((record["condition"], record["promptID"]), []) \
             .append(record["wordCount"])
    assert all(len(values) == 3 for values in cells.values())
    means = {key: sum(values) / 3 for key, values in cells.items()}
    assert [means[("baseline", item)] for item in ITEMS] == [12, 21, 33, 42]
    assert [means[("steered", item)] for item in ITEMS] == [18, 24, 32, 52]
    differences = [means[("steered", item)] - means[("baseline", item)]
                   for item in ITEMS]
    assert differences == [6, 3, -1, 10]

    rows = _analyze(tmp_path, case["records"])

    # Pooled: one difference per ITEM (n = 4, not 12 sample pairs and not 0).
    pooled = _one(rows, stratifyBy="pooled", endpoint="wordCount")
    assert pooled["condition"] == "steered"
    assert pooled["n"] == "4"
    assert float(pooled["deltaMean"]) == 4.5
    assert float(pooled["wilcoxonW"]) == 1.0
    assert float(pooled["wilcoxonP"]) == pytest.approx(
        _two_sided_p(1, 4), abs=5e-7)          # 0.201243
    # A family of one: the adjusted p is the raw p.
    assert pooled["adjustedP"] == pooled["wilcoxonP"]
    assert pooled["correction"] == "bh"
    # Pooled rows leave unit, estimand, and inference empty: their unit is
    # the run's default, the item.
    assert (pooled["unit"], pooled["estimand"], pooled["inference"]) \
        == ("", "", "")
    # The interval is a bootstrap of those four differences.
    assert -1 <= float(pooled["ciLower"]) <= 4.5 <= float(pooled["ciUpper"]) <= 10

    # distinct2, per-item differences +0.25, 0, +0.25, −0.5: mean 0; the zero
    # drops, the two 0.25s tie at rank 1.5, so W+ = W− = 3 and p clamps to 1.
    distinct = _one(rows, stratifyBy="pooled", endpoint="distinct2")
    assert distinct["n"] == "4"
    assert float(distinct["deltaMean"]) == 0.0
    assert float(distinct["wilcoxonW"]) == 3.0
    assert float(distinct["wilcoxonP"]) == 1.0

    # Strata with two items stay item-level: arm x = {item-1, item-3} →
    # (6 − 1) / 2; arm y = {item-2, item-4} → (3 + 10) / 2.
    arm_x = _one(rows, stratifyBy="arm", stratum="x", endpoint="wordCount")
    assert (arm_x["n"], float(arm_x["deltaMean"])) == ("2", 2.5)
    assert float(arm_x["wilcoxonW"]) == 1.0          # ranks −1 → 1, +6 → 2
    assert float(arm_x["wilcoxonP"]) == 1.0          # z = 0
    arm_y = _one(rows, stratifyBy="arm", stratum="y", endpoint="wordCount")
    assert (arm_y["n"], float(arm_y["deltaMean"])) == ("2", 6.5)
    assert float(arm_y["wilcoxonW"]) == 0.0
    assert float(arm_y["wilcoxonP"]) == pytest.approx(
        _two_sided_p(0, 2), abs=5e-7)                # 0.371093
    # Benjamini–Hochberg over the two arm rows: the smaller p doubles.
    assert float(arm_y["adjustedP"]) == pytest.approx(
        2 * _two_sided_p(0, 2), abs=1e-6)            # 0.742187
    assert float(arm_x["adjustedP"]) == 1.0
    for row in (arm_x, arm_y):
        assert (row["unit"], row["estimand"], row["inference"]) \
            == ("item", "itemLevel", "corrected")

    # A one-item stratum drops to the sample axis, pairing sample k to
    # baseline sample k (by sampleIndex, not by seed), and is a diagnostic.
    by_sample = {
        "item-1": ([5, 6, 7], 6.0, 0.0, 0.0),         # 15−10, 18−12, 21−14
        "item-2": ([2, 5, 2], 3.0, 0.0, 6.0),         # the two 2s tie
        "item-3": ([1, -1, -3], -1.0, 1.5, 6.0),      # |1| and |−1| tie
        "item-4": ([10, 11, 9], 10.0, 0.0, 0.0),
    }
    for item, (sample_diffs, mean, w, ties) in by_sample.items():
        assert sum(sample_diffs) / 3 == mean
        row = _one(rows, stratifyBy="promptID", stratum=item,
                   endpoint="wordCount")
        assert row["n"] == "3"
        assert float(row["deltaMean"]) == mean
        assert float(row["wilcoxonW"]) == w
        assert float(row["wilcoxonP"]) == pytest.approx(
            _two_sided_p(w, 3, ties), abs=5e-7)
        assert (row["unit"], row["estimand"], row["inference"]) \
            == ("sample", "withinItemSamples", "diagnostic")
        assert (row["adjustedP"], row["correction"]) == ("", "")
