"""Stratified effect rows beside the pooled rows (analyze verb).

The motivating failure mode (verified on real K&Z data 2026-08-06): pooling
choice/numeric endpoints across all task items HID a real single-cell effect
(one live item at 23/50 vs 8/50 among eleven 0/1-saturated items → pooled
choiceRate +0.025, Wilcoxon p 1.0) and, in the mirror direction, MANUFACTURED
pooled numeric effects out of one item's parse garbage. The stratified rows
keep the pooled rows byte-compatible (same stats, same correction family,
labeled ``pooled``) and add per-stratum rows — by promptID always, and by the
record-carried factor metadata (arm/caseID/factors) — each family corrected
independently. Swift twin: AnalyzeEffectSizesTests' stratified tests (same
CSV column vocabulary: stratifyBy, stratum, unit)."""

import csv
import json

from steerlab_server.experiment import study_stats, tasks
from steerlab_server.experiment.manifest import Manifest

CASES = ("loan", "lease", "tort", "contract")
ARMS = ("legal", "notLegal", "control")
LIVE_ITEM = "loan-notLegal"
SAMPLES = 50
LIVE_BASELINE_HITS = 8
LIVE_STEERED_HITS = 23


def _manifest_dict():
    return {
        "name": "study", "modelID": "test/model",
        "concepts": [], "taskPromptsFile": None,
        "conditions": [{"name": "steered",
                        "slots": [{"concept": "extraversion", "layer": 1,
                                   "alpha": 2.0}]}],
    }


def _write_run(tmp_path, manifest_dict, records):
    exp_dir = tmp_path / "experiments" / "study"
    exp_dir.mkdir(parents=True)
    (exp_dir / "experiment.json").write_text(
        json.dumps(manifest_dict), encoding="utf-8")
    run_dir = tmp_path / "runs" / "20260806T000000000-exp-study-run"
    run_dir.mkdir(parents=True)
    (run_dir / "experiment-hash.txt").write_text(
        Manifest.from_dict(manifest_dict).content_hash() + "\n",
        encoding="utf-8")
    (run_dir / "generations.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return run_dir


def _choice_records():
    """12 items (4 caseIDs × 3 arms), 50 samples per condition. Every item is
    saturated at choiceRate 1.0 in both arms EXCEPT the live cell, which
    moves 8/50 → 23/50 (hits at the low sample indices)."""
    records = []
    for case in CASES:
        for arm in ARMS:
            prompt_id = f"{case}-{arm}"
            for condition in ("baseline", "steered"):
                if prompt_id == LIVE_ITEM:
                    hits = (LIVE_BASELINE_HITS if condition == "baseline"
                            else LIVE_STEERED_HITS)
                else:
                    hits = SAMPLES
                for sample in range(SAMPLES):
                    records.append({
                        "condition": condition, "promptID": prompt_id,
                        "sampleIndex": sample, "target": "A",
                        "parsedChoice": "A" if sample < hits else "B",
                        "arm": arm, "caseID": case,
                    })
    return records


def _rows_by(rows, **match):
    return [r for r in rows
            if all(r[key] == value for key, value in match.items())]


def test_saturated_cells_no_longer_mask_a_live_cell(tmp_path):
    """The exact failure mode: the pooled choiceRate row stays weak (its
    semantics are untouched — Wilcoxon over 12 item diffs, 11 of them zero,
    p = 1), while the stratified rows surface the live cell at sample
    resolution with a small corrected p."""
    _write_run(tmp_path, _manifest_dict(), _choice_records())
    out = tasks.analyze("study", root=str(tmp_path))
    with open(f"{out}/effect-sizes.csv", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == study_stats.EFFECT_SIZES_HEADER
        rows = list(reader)

    # Pooled row: unchanged semantics, labeled pooled, unit empty, and both
    # estimand columns empty (its estimand is the run's declared unit). The
    # live cell's +0.3 is diluted to +0.025 and the sign test over item diffs
    # is blind to it — exactly the bug this table now makes visible.
    [pooled] = _rows_by(rows, stratifyBy="pooled", endpoint="choiceRate")
    assert pooled["condition"] == "steered"
    assert pooled["stratum"] == "" and pooled["unit"] == ""
    assert pooled["estimand"] == "" and pooled["inference"] == ""
    assert pooled["n"] == "12"
    assert abs(float(pooled["deltaMean"]) - 0.025) < 1e-9
    assert float(pooled["wilcoxonP"]) > 0.99
    assert float(pooled["adjustedP"]) > 0.99

    # promptID stratum for the live item: per-sample pairs, n = 50, the true
    # +0.3, and a raw Wilcoxon p that locates the mover.
    [live] = _rows_by(rows, stratifyBy="promptID", stratum=LIVE_ITEM,
                      endpoint="choiceRate")
    assert live["unit"] == "sample"
    assert live["n"] == str(SAMPLES)
    assert abs(float(live["deltaMean"]) - 0.3) < 1e-9
    assert float(live["wilcoxonP"]) < 0.01
    assert live["modality"] == "injection"
    saturated = _rows_by(rows, stratifyBy="promptID", endpoint="choiceRate")
    assert len(saturated) == 12

    # The factor CROSS family names the cell the way the researcher reads it:
    # arm×caseID → notLegal×loan (keys sorted, levels in key order). One item
    # per cell here, so it carries the same sample-resolution stats.
    [cell] = _rows_by(rows, stratifyBy="arm×caseID", stratum="notLegal×loan",
                      endpoint="choiceRate")
    assert cell["unit"] == "sample"
    assert cell["n"] == str(SAMPLES)
    assert abs(float(cell["deltaMean"]) - 0.3) < 1e-9

    # Marginal factor families: the notLegal arm pools its 4 items at ITEM
    # resolution (the pooled machinery restricted to the stratum), so it IS a
    # corrected cross-item estimate within its stratum.
    [arm_row] = _rows_by(rows, stratifyBy="arm", stratum="notLegal",
                         endpoint="choiceRate")
    assert arm_row["unit"] == "item"
    assert arm_row["estimand"] == "itemLevel"
    assert arm_row["inference"] == "corrected"
    assert arm_row["n"] == "4"
    assert abs(float(arm_row["deltaMean"]) - 0.075) < 1e-9
    assert {r["stratum"] for r in _rows_by(rows, stratifyBy="caseID")} == set(CASES)

    # fingerprints.csv stays a pure reshape of the POOLED rows only.
    with open(f"{out}/fingerprints.csv", encoding="utf-8") as handle:
        fingerprint_rows = list(csv.DictReader(handle))
    assert len(fingerprint_rows) == 1


def test_within_item_rows_declare_their_estimand_and_carry_no_adjusted_p(tmp_path):
    """The estimand defect (review 2026-08-06). A single-item stratum drops to
    per-SAMPLE differences inside one prompt — a prompt-specific stochastic
    quantity that cannot generalize across prompts — yet those rows used to
    flow through the same correction machinery as the item-level rows and
    emerge with a corrected p that read exactly like a cross-item finding.
    Each stratified row now declares its estimand, and within-item rows are
    marked diagnostic and held OUT of the correction: estimate, interval and
    raw Wilcoxon p survive as a locator, adjustedP does not exist."""
    _write_run(tmp_path, _manifest_dict(), _choice_records())
    out = tasks.analyze("study", root=str(tmp_path))
    with open(f"{out}/effect-sizes.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    # Every stratified row declares an estimand; the two columns agree by
    # construction with `unit`, and pooled rows stay blank.
    stratified = [r for r in rows if r["stratifyBy"] != "pooled"]
    assert stratified
    for row in stratified:
        expected = ("withinItemSamples" if row["unit"] == "sample"
                    else "itemLevel")
        assert row["estimand"] == expected
        assert row["inference"] == ("diagnostic" if row["unit"] == "sample"
                                    else "corrected")
    assert all(r["estimand"] == "" and r["inference"] == ""
               for r in rows if r["stratifyBy"] == "pooled")

    # The live cell: diagnostic, uncorrected, and still the loudest row in
    # the table by raw p — demoted, not hidden.
    [live] = _rows_by(rows, stratifyBy="promptID", stratum=LIVE_ITEM,
                      endpoint="choiceRate")
    assert live["estimand"] == "withinItemSamples"
    assert live["inference"] == "diagnostic"
    assert live["adjustedP"] == "" and live["correction"] == ""
    assert float(live["wilcoxonP"]) < 0.01
    assert abs(float(live["deltaMean"]) - 0.3) < 1e-9
    assert live["ciLower"] and live["ciUpper"]

    # No within-item row anywhere carries an adjusted p or a correction
    # family — the correction machinery no longer touches this estimand.
    diagnostics = [r for r in rows if r["inference"] == "diagnostic"]
    assert diagnostics
    assert all(r["adjustedP"] == "" and r["correction"] == ""
               for r in diagnostics)

    # …and holding them out did not change what the item-level rows get: the
    # arm family's four marginal rows are still corrected among themselves.
    arm_rows = _rows_by(rows, stratifyBy="arm", endpoint="choiceRate")
    assert len(arm_rows) == 3
    assert all(r["inference"] == "corrected" for r in arm_rows)
    assert all(r["correction"] == "bh" for r in arm_rows
               if r["wilcoxonP"] != "")


def test_stratified_rows_localize_a_manufactured_numeric_effect(tmp_path):
    """The mirror failure: one item's parse garbage (9999 months) manufactures
    a large pooled meanMonths effect. The promptID strata pin the entire
    effect on the garbage item; every other stratum reads zero."""
    records = []
    for prompt_id, steered_months in (("p1", 10.0), ("p2", 10.0),
                                      ("p3", 10.0), ("p4", 9999.0)):
        for condition, months in (("baseline", 10.0),
                                  ("steered", steered_months)):
            records.append({"condition": condition, "promptID": prompt_id,
                            "sampleIndex": 0, "parsedMonths": months})
    _write_run(tmp_path, _manifest_dict(), records)
    out = tasks.analyze("study", root=str(tmp_path))
    with open(f"{out}/effect-sizes.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    [pooled] = _rows_by(rows, stratifyBy="pooled", endpoint="meanMonths")
    assert abs(float(pooled["deltaMean"]) - 9989.0 / 4) < 1e-6

    strata = _rows_by(rows, stratifyBy="promptID", endpoint="meanMonths")
    assert len(strata) == 4
    by_item = {r["stratum"]: r for r in strata}
    # Single item, single sample: nothing to resample — a descriptive row
    # (n = 1, degenerate CI) that still names the mover. There is no sample
    # axis to fall back to, so it stays an item-level row.
    assert by_item["p4"]["unit"] == "item" and by_item["p4"]["n"] == "1"
    assert by_item["p4"]["estimand"] == "itemLevel"
    assert abs(float(by_item["p4"]["deltaMean"]) - 9989.0) < 1e-6
    for item in ("p1", "p2", "p3"):
        assert float(by_item[item]["deltaMean"]) == 0.0
    # No factor metadata on these records → no factor families appear.
    assert not _rows_by(rows, stratifyBy="arm")
    assert {r["stratifyBy"] for r in rows} == {"pooled", "promptID"}
