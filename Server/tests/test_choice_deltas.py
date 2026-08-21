"""Per-item choice deltas (``analyze`` → ``choice-deltas.csv``): the citable,
engine-computed version of the per-item Δ the results explorer otherwise
derives client-side.

Covered: correct deltas/flips from a synthetic run, the ``skippedNoBaseline``
count (silent truncation reads as coverage), absence over empty artifacts when
a run carries no choice readouts, and byte-identical output on a second
analyze of the same run. Column names, the JSON keys and the skip semantics
are the cross-engine contract — Swift twin ``ChoiceDeltasTests``.
"""

import csv
import json
import math
import os

from steerlab_server.experiment import choice_deltas, tasks
from steerlab_server.experiment.manifest import Manifest


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, str) else json.dumps(payload)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)


def _instrument(condition, prompt_id, *, target="A", log_odds=None,
                selected=None, probability=None, **extra):
    """One answer-token instrument record, shaped like the run loop's
    (``logprob.ChoiceResult.as_record_fields`` + the prompt metadata)."""
    odds = {"A": 0.0, "B": 0.0} if log_odds is None else dict(log_odds)
    record = {
        "condition": condition,
        "promptID": prompt_id,
        "promptIndex": 0,
        "instrument": "answerTokenLogprob",
        "options": sorted(odds),
        "optionLogprobs": {k: -abs(v) for k, v in odds.items()},
        "logOdds": odds,
        "target": target,
        "selected": selected if selected is not None else max(odds, key=odds.get),
        "margin": 1.0,
    }
    if probability is not None:
        record["choiceProbability"] = dict(probability)
    record.update(extra)
    return record


def _sampled(condition, prompt_id, *, output="text"):
    return {"condition": condition, "promptID": prompt_id, "promptIndex": 0,
            "seed": 1, "sampleIndex": 0, "output": output,
            "wordCount": len(output.split()), "distinct2": 1.0}


def _fixture(tmp_path, records, manifest_extra=None):
    """A manifest + one epoch-stamped source run (the test_exclusions pattern)."""
    root = str(tmp_path)
    manifest_dict = {"name": "study", "modelID": "org/m", "concepts": [],
                     "conditions": [{"name": "steer", "slots": [
                         {"concept": "fear", "layer": 1, "alpha": 2.0}]}]}
    manifest_dict.update(manifest_extra or {})
    _write(os.path.join(root, "experiments", "study", "experiment.json"),
           manifest_dict)
    run_dir = os.path.join(root, "runs", "20260805T000000000-exp-study-run")
    os.makedirs(run_dir, exist_ok=True)
    _write(os.path.join(run_dir, "experiment-hash.txt"),
           Manifest.from_dict(manifest_dict).content_hash() + "\n")
    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return root, run_dir


def _rows(out):
    with open(os.path.join(out, "choice-deltas.csv"), encoding="utf-8") as handle:
        return list(csv.reader(handle))


# --- the table ---------------------------------------------------------------

def test_analyze_writes_per_item_deltas_and_flips(tmp_path):
    records = [
        # p1: target A moves up 3 nats, no flip (A stays selected).
        _instrument("baseline", "p1", target="A",
                    log_odds={"A": 1.0, "B": -1.0}, selected="A",
                    probability={"A": 0.8, "B": 0.2}),
        _instrument("steer", "p1", target="A",
                    log_odds={"A": 4.0, "B": -4.0}, selected="A",
                    probability={"A": 0.95, "B": 0.05}),
        # p2: target A moves up 5 nats AND flips B → A.
        _instrument("baseline", "p2", target="A",
                    log_odds={"A": -2.0, "B": 2.0}, selected="B",
                    probability={"A": 0.25, "B": 0.75}),
        _instrument("steer", "p2", target="A",
                    log_odds={"A": 3.0, "B": -3.0}, selected="A",
                    probability={"A": 0.9, "B": 0.1}),
        _sampled("baseline", "p1"), _sampled("steer", "p1"),
    ]
    root, run_dir = _fixture(tmp_path, records)

    out = tasks.analyze("study", root=root, source_run=run_dir,
                        log=lambda *_: None)

    rows = _rows(out)
    assert rows[0] == choice_deltas.CHOICE_DELTAS_HEADER
    assert len(rows) == 3  # header + one row per non-baseline item
    assert rows[1] == ["steer", "p1", "A", "1", "4", "3", "A", "A", "0",
                       "0.8", "0.95", "0.15"]
    assert rows[2] == ["steer", "p2", "A", "-2", "3", "5", "B", "A", "1",
                       "0.25", "0.9", "0.65"]
    # Baseline rows are never their own condition: the table IS the delta.
    assert not any(row[0] == "baseline" for row in rows[1:])

    summary = json.load(open(os.path.join(out, "choice-deltas.json")))
    assert summary["records"] == 2
    assert summary["skippedNoBaseline"] == 0
    block = summary["conditions"]["steer"]
    assert block["n"] == 2 and block["flipped"] == 1
    assert math.isclose(block["deltaTargetLogOddsMean"], 4.0)
    # The CI comes from the EXISTING paired bootstrap at its defaults — the
    # same machinery effect-sizes.csv runs on, not a second implementation.
    assert block["replicates"] == 10_000 and block["seed"] == 0
    assert block["ciLower"] <= block["deltaTargetLogOddsMean"] <= block["ciUpper"]


def test_rows_sort_by_condition_then_prompt_regardless_of_run_order(tmp_path):
    records = [
        _instrument("baseline", "p2", log_odds={"A": 0.0, "B": -1.0}),
        _instrument("zeta", "p2", log_odds={"A": 1.0, "B": -1.0}),
        _instrument("alpha", "p2", log_odds={"A": 2.0, "B": -1.0}),
        _instrument("baseline", "p1", log_odds={"A": 0.0, "B": -1.0}),
        _instrument("zeta", "p1", log_odds={"A": 3.0, "B": -1.0}),
        _instrument("alpha", "p1", log_odds={"A": 4.0, "B": -1.0}),
    ]
    root, run_dir = _fixture(tmp_path, records)

    out = tasks.analyze("study", root=root, source_run=run_dir,
                        log=lambda *_: None)

    assert [(r[0], r[1]) for r in _rows(out)[1:]] == [
        ("alpha", "p1"), ("alpha", "p2"), ("zeta", "p1"), ("zeta", "p2")]


def test_a_readout_with_no_baseline_partner_is_skipped_and_counted(tmp_path):
    records = [
        _instrument("baseline", "p1", log_odds={"A": 0.0, "B": -1.0}),
        _instrument("steer", "p1", log_odds={"A": 2.0, "B": -1.0}),
        # p2 was measured under steer but never under baseline.
        _instrument("steer", "p2", log_odds={"A": 5.0, "B": -1.0}),
    ]
    root, run_dir = _fixture(tmp_path, records)

    out = tasks.analyze("study", root=root, source_run=run_dir,
                        log=lambda *_: None)

    assert [r[1] for r in _rows(out)[1:]] == ["p1"]
    summary = json.load(open(os.path.join(out, "choice-deltas.json")))
    assert summary["skippedNoBaseline"] == 1
    assert summary["conditions"]["steer"]["skippedNoBaseline"] == 1
    assert summary["conditions"]["steer"]["n"] == 1


def test_a_readout_missing_its_own_target_is_skipped_and_counted(tmp_path):
    # p1: logOdds present but with no entry for the item's own target.
    # p2: no logOdds at all (malformed/legacy readout). Both are unreadable
    # measurements, not zeros, and both are COUNTED.
    unreadable = _instrument("steer", "p2", log_odds={"A": 1.0, "B": 0.0})
    del unreadable["logOdds"]
    records = [
        _instrument("baseline", "p1", target="C", log_odds={"A": 0.0, "B": 0.0}),
        _instrument("steer", "p1", target="C", log_odds={"A": 2.0, "B": 0.0}),
        _instrument("baseline", "p2", log_odds={"A": 0.0, "B": 0.0}),
        unreadable,
    ]
    root, run_dir = _fixture(tmp_path, records)

    out = tasks.analyze("study", root=root, source_run=run_dir,
                        log=lambda *_: None)

    assert _rows(out) == [choice_deltas.CHOICE_DELTAS_HEADER]
    summary = json.load(open(os.path.join(out, "choice-deltas.json")))
    assert summary["skippedNoTargetValue"] == 2
    assert summary["conditions"]["steer"]["n"] == 0
    # No pairing ⇒ no mean and no interval, rather than a zero.
    assert "deltaTargetLogOddsMean" not in summary["conditions"]["steer"]


def test_probabilities_are_optional_but_the_log_odds_row_stands(tmp_path):
    records = [
        _instrument("baseline", "p1", log_odds={"A": 0.0, "B": -1.0}),
        _instrument("steer", "p1", log_odds={"A": 2.0, "B": -1.0}),
    ]
    root, run_dir = _fixture(tmp_path, records)

    out = tasks.analyze("study", root=root, source_run=run_dir,
                        log=lambda *_: None)

    row = _rows(out)[1]
    assert row[:6] == ["steer", "p1", "A", "0", "2", "2"]
    assert row[9:] == ["", "", ""]


def test_a_flip_needs_both_sides_selected(tmp_path):
    records = [
        _instrument("baseline", "p1", log_odds={"A": 0.0, "B": -1.0},
                    selected=""),
        _instrument("steer", "p1", log_odds={"A": 2.0, "B": -1.0},
                    selected="A"),
    ]
    root, run_dir = _fixture(tmp_path, records)

    out = tasks.analyze("study", root=root, source_run=run_dir,
                        log=lambda *_: None)

    row = _rows(out)[1]
    assert (row[6], row[7], row[8]) == ("", "", "0")


# --- absence over empty artifacts --------------------------------------------

def test_a_run_without_choice_readouts_grows_no_table(tmp_path):
    root, run_dir = _fixture(
        tmp_path, [_sampled("baseline", "p1"), _sampled("steer", "p1")])

    out = tasks.analyze("study", root=root, source_run=run_dir,
                        log=lambda *_: None)

    assert not os.path.exists(os.path.join(out, "choice-deltas.csv"))
    assert not os.path.exists(os.path.join(out, "choice-deltas.json"))


def test_baseline_only_readouts_grow_no_table(tmp_path):
    """Nothing to delta: the file is per-item CHANGE, not per-item value."""
    root, run_dir = _fixture(
        tmp_path,
        [_instrument("baseline", "p1", log_odds={"A": 0.0, "B": -1.0}),
         _sampled("baseline", "p1"), _sampled("steer", "p1")])

    out = tasks.analyze("study", root=root, source_run=run_dir,
                        log=lambda *_: None)

    assert not os.path.exists(os.path.join(out, "choice-deltas.csv"))


def test_all_skipped_still_reports_the_skips(tmp_path):
    """A header-only CSV plus a skip count is honest; silence is not."""
    root, run_dir = _fixture(
        tmp_path, [_instrument("steer", "p1", log_odds={"A": 2.0, "B": -1.0})])

    out = tasks.analyze("study", root=root, source_run=run_dir,
                        log=lambda *_: None)

    assert _rows(out) == [choice_deltas.CHOICE_DELTAS_HEADER]
    summary = json.load(open(os.path.join(out, "choice-deltas.json")))
    assert summary["skippedNoBaseline"] == 1 and summary["records"] == 0


# --- determinism -------------------------------------------------------------

def test_the_same_run_analyzed_twice_is_byte_identical(tmp_path):
    records = []
    for index, prompt_id in enumerate(["p3", "p1", "p2"]):
        records.append(_instrument("baseline", prompt_id,
                                   log_odds={"A": float(index), "B": -1.0},
                                   probability={"A": 0.5, "B": 0.5}))
        records.append(_instrument("steer", prompt_id,
                                   log_odds={"A": float(index) + 1.5, "B": -1.0},
                                   probability={"A": 0.7, "B": 0.3}))
    root, run_dir = _fixture(tmp_path, records)

    first = tasks.analyze("study", root=root, source_run=run_dir,
                          log=lambda *_: None)
    second = tasks.analyze("study", root=root, source_run=run_dir,
                           log=lambda *_: None)

    assert first != second  # two immutable analyze directories
    for artifact in ("choice-deltas.csv", "choice-deltas.json"):
        with open(os.path.join(first, artifact), "rb") as handle:
            a = handle.read()
        with open(os.path.join(second, artifact), "rb") as handle:
            b = handle.read()
        assert a == b, artifact


# --- exclusions ---------------------------------------------------------------

def test_an_excluded_readout_does_not_reappear_as_a_citable_delta(tmp_path):
    # Targets are DECLARED in the task file (open-issues #6): a
    # choice study's items carry their own target, and analyze reads
    # declaredness from the pinned file, never from record synthesis.
    prompts = ('{"id": "p1", "prompt": "a", "target": "A", "attentionCheck": '
               '{"expected": "a", "grading": "token_exact"}}\n'
               '{"id": "p2", "prompt": "b", "target": "A"}\n')
    prompts_path = os.path.join(str(tmp_path), "prompts", "t.jsonl")
    _write(prompts_path, prompts)
    import hashlib
    with open(prompts_path, "rb") as handle:
        prompts_hash = hashlib.sha256(handle.read()).hexdigest()
    records = [
        _sampled("baseline", "p1", output="a"),
        _sampled("baseline", "p2", output="b"),
        _sampled("steer", "p1", output="c"),  # fails its attention check
        _sampled("steer", "p2", output="d"),
        _instrument("baseline", "p1", log_odds={"A": 0.0, "B": -1.0}),
        _instrument("baseline", "p2", log_odds={"A": 0.0, "B": -1.0}),
        _instrument("steer", "p1", log_odds={"A": 9.0, "B": -1.0}),
        _instrument("steer", "p2", log_odds={"A": 1.0, "B": -1.0}),
    ]
    root, run_dir = _fixture(
        tmp_path, records,
        manifest_extra={"exclusionRules": [{"rule": "failedAttentionCheck"}],
                        "taskPromptsFile": "prompts/t.jsonl",
                        "taskPromptsHash": prompts_hash})

    out = tasks.analyze("study", root=root, source_run=run_dir,
                        log=lambda *_: None)

    # p1's steer cell failed the check: its readout drops with the cell, and
    # the surviving delta is p2's alone.
    assert [(r[1], r[5]) for r in _rows(out)[1:]] == [("p2", "1")]
