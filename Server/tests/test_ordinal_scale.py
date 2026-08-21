"""The general ordinal-scale outcome instrument (``ordinalScale``): pure
aggregation math, the declared-aggregation contract (``ordinalAggregation`` —
declared, never silently defaulted), record/report fields, the closed
vocabularies, and the statistics layer (analyze's paired ``ordinalPosition``
endpoint). The math fixtures are cross-engine twins of Swift's
``OrdinalScaleInstrumentTests`` — SAME numbers asserted on both engines."""

import csv
import json
import math
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import logprob as logprob_mod
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import (
    KNOWN_ORDINAL_AGGREGATIONS, Manifest)
from steerlab_server.steering.vector_store import ConceptVectors

LADDER = ["1", "2", "3"]


# --- pure aggregation math (Swift OrdinalScaleInstrumentTests twins) ----------

def test_expected_value_and_argmax_on_canonical_fixture():
    """The canonical cross-engine fixture: ladder of 3, probs
    [0.2, 0.3, 0.5] → expected value 2.3, argmax position 3."""
    distribution = [0.2, 0.3, 0.5]
    assert math.isclose(
        logprob_mod.ordinal_position(distribution, "expectedValue"), 2.3)
    assert logprob_mod.ordinal_position(distribution, "argmax") == 3.0


def test_ordinal_distribution_renormalizes():
    assert logprob_mod.ordinal_distribution([0.2, 0.2]) == [0.5, 0.5]
    unchanged = logprob_mod.ordinal_distribution([0.2, 0.3, 0.5])
    assert all(math.isclose(a, b)
               for a, b in zip(unchanged, [0.2, 0.3, 0.5]))
    # Negative inputs clamp to 0 before renormalizing.
    clamped = logprob_mod.ordinal_distribution([-1.0, 1.0, 1.0])
    assert clamped == [0.0, 0.5, 0.5]
    # Degenerate all-zero total → uniform; empty in → empty out.
    assert logprob_mod.ordinal_distribution([0, 0, 0, 0]) == [0.25] * 4
    assert logprob_mod.ordinal_distribution([]) == []


def test_argmax_tie_breaks_to_first_maximum():
    """Tie rule pinned cross-engine: FIRST max wins (Swift strictly-greater
    scan twin)."""
    assert logprob_mod.ordinal_position([0.5, 0.5], "argmax") == 1.0
    assert logprob_mod.ordinal_position([0.2, 0.4, 0.4], "argmax") == 2.0


def test_empty_distribution_degenerates_to_zero():
    assert logprob_mod.ordinal_position([], "expectedValue") == 0.0
    assert logprob_mod.ordinal_position([], "argmax") == 0.0


def test_ordinal_position_refuses_unknown_aggregation():
    with pytest.raises(ValueError, match="unknown ordinalAggregation"):
        logprob_mod.ordinal_position([0.5, 0.5], "median")


def _canonical_choice(options=None, probs=(0.2, 0.3, 0.5)):
    return logprob_mod.ChoiceResult(options=[
        logprob_mod.OptionScore(option=o, token_ids=[i + 1],
                                token_logprobs=[math.log(p)])
        for i, (o, p) in enumerate(zip(options or LADDER, probs))
    ], prompt_token_count=5)


def test_ordered_probabilities_follow_ladder_order():
    """Position i always means the i-th declared option, matching the
    dictionary softmax value-for-value."""
    result = _canonical_choice()
    ordered = result.ordered_probabilities
    assert math.isclose(sum(ordered), 1.0)
    assert all(math.isclose(a, b, rel_tol=1e-9)
               for a, b in zip(ordered, [0.2, 0.3, 0.5]))
    for i, score in enumerate(result.options):
        assert math.isclose(ordered[i], result.probability[score.option])


# --- vocabulary (cross-engine list parity) ------------------------------------

def test_vocabularies_match_the_swift_literals():
    # Twin of Swift's vocabulariesMatchTheServerLiterals — same literals,
    # verbatim.
    assert es.KNOWN_OUTCOME_INSTRUMENTS == (
        "sampledText", "answerTokenLogprob", "choiceProbability",
        "repeReaderScore", "ordinalScale")
    assert es.KNOWN_ORDINAL_AGGREGATIONS == ("expectedValue", "argmax")
    assert KNOWN_ORDINAL_AGGREGATIONS == ("expectedValue", "argmax")
    assert logprob_mod.ORDINAL_AGGREGATIONS == ("expectedValue", "argmax")
    assert tasks.CHOICE_INSTRUMENTS == {
        "answerTokenLogprob", "choiceProbability", "ordinalScale"}


# --- verify gates (declared, never defaulted) ---------------------------------

def _manifest(**extra):
    d = {"name": "ord", "modelID": "test/model",
         "concepts": [], "conditions": [], "taskPromptsFile": None}
    d.update(extra)
    return Manifest.from_dict(d)


def test_verify_refuses_ordinal_scale_without_aggregation(tmp_path):
    manifest = _manifest(outcomeInstruments=["ordinalScale"])
    violations = manifest.verify(str(tmp_path))
    assert any("ordinalScale but ordinalAggregation" in v for v in violations)


def test_verify_refuses_unknown_aggregation(tmp_path):
    manifest = _manifest(outcomeInstruments=["ordinalScale"],
                         ordinalAggregation="median")
    violations = manifest.verify(str(tmp_path))
    assert any("unknown ordinalAggregation 'median'" in v for v in violations)


def test_verify_accepts_declared_aggregations(tmp_path):
    for aggregation in KNOWN_ORDINAL_AGGREGATIONS:
        manifest = _manifest(outcomeInstruments=["ordinalScale"],
                             ordinalAggregation=aggregation)
        assert not any("ordinalAggregation" in v
                       for v in manifest.verify(str(tmp_path)))
    # No ordinalScale, no aggregation — nothing to declare.
    plain = _manifest(outcomeInstruments=["answerTokenLogprob"])
    assert not any("ordinalAggregation" in v
                   for v in plain.verify(str(tmp_path)))


def test_thinking_mode_gate_covers_ordinal_scale(tmp_path):
    manifest = _manifest(outcomeInstruments=["ordinalScale"],
                         ordinalAggregation="expectedValue",
                         qwenThinkingEnabled=True)
    violations = manifest.verify(str(tmp_path))
    assert any("thinking-mode answers are marginals" in v for v in violations)
    off = _manifest(outcomeInstruments=["ordinalScale"],
                    ordinalAggregation="expectedValue")
    assert not any("marginals" in v for v in off.verify(str(tmp_path)))


# --- run loop: records + report (fake-model harness) --------------------------

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _study_fixture(root, name, *, ordinal_aggregation="expectedValue",
                   instruments=("ordinalScale",), conditions=None,
                   declare_target=False):
    """A two-item ordinal-ladder study: per-item options = the declared
    ladder, no target required. A concept is attached only when
    ``conditions`` gives it an arm — attaching one and then declaring no
    injection condition is the baseline-only-that-measures-nothing shape the
    run loop refuses (2026-08-17, ``no_measured_conditions_problem``), and it
    was never what these record/report-shape fixtures meant: the concept is
    unused here and ``_extract_all`` is stubbed out anyway."""
    es.create(name, model_id="org/m", revision="abc", root=root)
    if conditions is not None:
        concept_dir = os.path.join(root, "prompts", "concepts", "fear")
        _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "dread"}\n')
        _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "calm"}\n')
        es.attach(name, ["fear"], root=root)
    raw = es.load_raw(name, root)
    raw["seeds"] = [0]
    raw["temperature"] = 0.0
    raw["maxTokens"] = 16
    raw["outcomeInstruments"] = list(instruments)
    if ordinal_aggregation is not None:
        raw["ordinalAggregation"] = ordinal_aggregation
    if conditions is not None:
        raw["conditions"] = conditions
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    # `declare_target` makes these genuine CHOICE items rather than a rating
    # ladder. A target-dependent instrument (answerTokenLogprob /
    # choiceProbability) declared without ordinalScale over items that name no
    # target is refused at run start (open-issues #6): its endpoint is the
    # declared target's log-odds, and the engine no longer guesses one.
    lines = [json.dumps({"id": f"p{i}", "prompt": f"Rate item {i}.",
                         "options": LADDER,
                         **({"target": LADDER[0]} if declare_target else {})})
             for i in range(2)]
    _write(prompts_path, "\n".join(lines) + "\n")
    return prompts_path


@contextmanager
def _fake_model(model_id, revision=None):
    yield SimpleNamespace(model_id=model_id, revision=revision or "abc")


def _fake_score_options():
    """Per-prompt ladder distributions: p0 → [0.2, 0.3, 0.5] (EV 2.3),
    p1 → [0.5, 0.3, 0.2] (EV 1.7) — report mean 2.0, population SD 0.3."""
    def score_options(model, prompt, options, *, model_id=None, injections=None,
                      prompt_mode=None, system_prompt=None,
                      qwen_thinking_enabled=False):
        probs = (0.2, 0.3, 0.5) if "0" in prompt else (0.5, 0.3, 0.2)
        return _canonical_choice(options=list(options), probs=probs)
    return score_options


def _patch(monkeypatch):
    monkeypatch.setattr(
        tasks, "_extract_all",
        lambda model, manifest, root: {})
    monkeypatch.setattr(logprob_mod, "score_options", _fake_score_options())


def test_run_stamps_ordinal_record_fields_and_report_summary(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "ord-run")
    _patch(monkeypatch)
    run_dir = tasks.run("ord-run", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    with open(os.path.join(run_dir, "generations.jsonl"),
              encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    instruments = [r for r in records if "instrument" in r]
    assert len(instruments) == 2
    by_id = {r["promptID"]: r for r in instruments}
    assert math.isclose(by_id["p0"]["ordinalPosition"], 2.3, rel_tol=1e-6)
    assert math.isclose(by_id["p1"]["ordinalPosition"], 1.7, rel_tol=1e-6)
    for record in instruments:
        distribution = record["ordinalDistribution"]
        assert len(distribution) == 3
        assert math.isclose(sum(distribution), 1.0)
        # ordinalScale-only study: no sampled generation ran.
        assert record["instrument"] == "answerTokenLogprob"
    assert not [r for r in records
                if "instrument" not in r and "error" not in r]

    report = json.loads(
        open(os.path.join(run_dir, "report.json"), encoding="utf-8").read())
    baseline = report["conditions"]["baseline"]
    assert math.isclose(baseline["ordinalMean"], 2.0, rel_tol=1e-6)
    assert math.isclose(baseline["ordinalSD"], 0.3, rel_tol=1e-6)
    assert baseline["choiceReadouts"] == 2


def test_run_argmax_aggregation(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "ord-argmax",
                             ordinal_aggregation="argmax")
    _patch(monkeypatch)
    run_dir = tasks.run("ord-argmax", prompts, root,
                        model_provider=_fake_model, log=lambda *_: None)
    with open(os.path.join(run_dir, "generations.jsonl"),
              encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    by_id = {r["promptID"]: r for r in records if "instrument" in r}
    assert by_id["p0"]["ordinalPosition"] == 3.0
    assert by_id["p1"]["ordinalPosition"] == 1.0
    report = json.loads(
        open(os.path.join(run_dir, "report.json"), encoding="utf-8").read())
    assert math.isclose(report["conditions"]["baseline"]["ordinalMean"], 2.0)
    assert math.isclose(report["conditions"]["baseline"]["ordinalSD"], 1.0)


def test_run_refuses_undeclared_aggregation(tmp_path, monkeypatch):
    """Draft verify only WARNS, so the run itself must refuse — the
    instrument-design choice is declared, never silently defaulted."""
    root = str(tmp_path)
    prompts = _study_fixture(root, "ord-missing", ordinal_aggregation=None)
    _patch(monkeypatch)
    with pytest.raises(RuntimeError, match="ordinalAggregation"):
        tasks.run("ord-missing", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None)


# --- statistics layer: analyze's paired ordinalPosition endpoint --------------

def _fake_bundle():
    return tasks.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source="test", stimulus_hash="h")


def _fake_score_options_with_shift():
    """Steering shifts the ladder distribution up: baseline p0/p1 EVs are
    2.3/1.7 (the canonical fixtures); steered EVs are 2.6/2.3 — paired diffs
    +0.3 and +0.6, mean +0.45."""
    def score_options(model, prompt, options, *, model_id=None, injections=None,
                      prompt_mode=None, system_prompt=None,
                      qwen_thinking_enabled=False):
        if injections:
            probs = (0.1, 0.2, 0.7) if "0" in prompt else (0.2, 0.3, 0.5)
        else:
            probs = (0.2, 0.3, 0.5) if "0" in prompt else (0.5, 0.3, 0.2)
        return _canonical_choice(options=list(options), probs=probs)
    return score_options


def test_endpoint_values_extract_ordinal_position():
    """The endpoint NAME is the pinned cross-engine contract
    ("ordinalPosition" — Swift effectSizes twin), extracted from instrument
    records only when they carry the stamp."""
    records = [
        {"condition": "baseline", "promptID": "p1",
         "instrument": "answerTokenLogprob", "target": "1",
         "logOdds": {"1": -1.0}, "ordinalPosition": 2.3},
        {"condition": "steered", "promptID": "p1",
         "instrument": "answerTokenLogprob", "target": "1",
         "logOdds": {"1": -2.0}, "ordinalPosition": 2.6},
        # Non-ordinal instrument record: no ordinalPosition endpoint cell.
        {"condition": "baseline", "promptID": "p2",
         "instrument": "answerTokenLogprob", "target": "1",
         "logOdds": {"1": 0.5}},
    ]
    endpoints = tasks._endpoint_values(records)
    assert endpoints["ordinalPosition"]["baseline"] == {"p1": 2.3}
    assert endpoints["ordinalPosition"]["steered"] == {"p1": 2.6}
    # Open-issues #6: the ordinal record's target was synthesized, so it
    # contributes NO choice endpoint; the plain record (no ordinal stamp,
    # historical) keeps it.
    assert endpoints["choiceLogOdds"]["baseline"] == {"p2": 0.5}


def test_run_then_analyze_produces_ordinal_effect_row(tmp_path, monkeypatch):
    """End to end on the fake-model harness: run (baseline + steered) →
    report.json ordinal summaries → analyze → a paired "ordinalPosition"
    effect row with CI fields in effect-sizes.csv."""
    root = str(tmp_path)
    prompts = _study_fixture(
        root, "ord-analyze",
        conditions=[{"name": "steered",
                     "slots": [{"concept": "fear", "layer": 1, "alpha": 2.0}]}])
    monkeypatch.setattr(
        tasks, "_extract_all",
        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(
        logprob_mod, "score_options", _fake_score_options_with_shift())
    run_dir = tasks.run("ord-analyze", prompts, root,
                        model_provider=_fake_model, log=lambda *_: None)

    report = json.loads(
        open(os.path.join(run_dir, "report.json"), encoding="utf-8").read())
    assert math.isclose(report["conditions"]["baseline"]["ordinalMean"], 2.0)
    assert math.isclose(report["conditions"]["steered"]["ordinalMean"], 2.45,
                        rel_tol=1e-6)

    out = tasks.analyze("ord-analyze", root=root, source_run=run_dir,
                        log=lambda *_: None)
    with open(os.path.join(out, "effect-sizes.csv"), encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # The POOLED row (stratified companion rows ride in the same file).
    ordinal = [r for r in rows if r["endpoint"] == "ordinalPosition"
               and r["stratifyBy"] == "pooled"]
    assert len(ordinal) == 1
    row = ordinal[0]
    assert row["condition"] == "steered"
    assert row["n"] == "2"
    # Paired diffs +0.3/+0.6 → mean +0.45; the bootstrap CI stays inside the
    # diff range, and the correction column is stamped like every endpoint.
    assert math.isclose(float(row["deltaMean"]), 0.45, rel_tol=1e-6)
    assert 0.3 - 1e-9 <= float(row["ciLower"]) <= float(row["ciUpper"]) <= 0.6 + 1e-9
    assert row["correction"] in ("bh", "holm")
    assert row["modality"] == "injection"


def test_answer_token_records_omit_ordinal_fields(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "ord-plain", ordinal_aggregation=None,
                             instruments=("answerTokenLogprob",),
                             declare_target=True)
    _patch(monkeypatch)
    run_dir = tasks.run("ord-plain", prompts, root, model_provider=_fake_model,
                        log=lambda *_: None)
    with open(os.path.join(run_dir, "generations.jsonl"),
              encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    instruments = [r for r in records if "instrument" in r]
    assert instruments
    for record in instruments:
        assert "ordinalPosition" not in record
        assert "ordinalDistribution" not in record
    report = json.loads(
        open(os.path.join(run_dir, "report.json"), encoding="utf-8").read())
    assert "ordinalMean" not in report["conditions"]["baseline"]
    assert "ordinalSD" not in report["conditions"]["baseline"]


def test_a_ladder_run_writes_no_synthesized_target(tmp_path, monkeypatch):
    """Open-issues #6 at the WRITER, end to end.

    The run loop used to stamp `target = prompt.get("target") or
    prompt["options"][0]`, so every record of a rating-ladder study claimed
    the scale minimum as its target and analyze reported a `choiceLogOdds`
    endpoint for it. The record now says what is true: no target, and a
    `targetSource` that is explicitly null rather than absent."""
    root = str(tmp_path)
    prompts = _study_fixture(root, "ord-null-target")
    _patch(monkeypatch)
    run_dir = tasks.run("ord-null-target", prompts, root,
                        model_provider=_fake_model, log=lambda *_: None)
    with open(os.path.join(run_dir, "generations.jsonl"),
              encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    instruments = [r for r in records if "instrument" in r]
    assert instruments
    for record in instruments:
        assert record["target"] is None
        assert record["targetSource"] is None
        assert record["ordinalPosition"] is not None
        # The ladder's minimum is present as an OPTION, never as a target.
        assert LADDER[0] in record["options"]


def test_ordinal_records_emit_no_choice_log_odds():
    """Open-issues #6: an ordinalScale record's target was SYNTHESIZED
    (options[0] = the scale minimum), so its log-odds must never surface as
    the declared choiceLogOdds endpoint. Historical records (no targetSource
    stamp) are recognized by carrying ordinalPosition; new records are
    explicit via targetSource."""
    from steerlab_server.experiment import tasks

    records = [
        # Historical likert record: synthesized target, ordinal stamp.
        {"condition": "baseline", "promptID": "p1",
         "instrument": "answerTokenLogprob", "target": "1",
         "logOdds": {"1": 9.4}, "ordinalPosition": 2.3},
        {"condition": "steered", "promptID": "p1",
         "instrument": "answerTokenLogprob", "target": "1",
         "logOdds": {"1": 12.7}, "ordinalPosition": 2.6},
        # New-style declared-target choice record: endpoint stands.
        {"condition": "baseline", "promptID": "p2",
         "instrument": "answerTokenLogprob", "target": "B",
         "targetSource": "declared", "logOdds": {"B": -0.5}},
        # New-style ordinal record: explicit None targetSource, no endpoint
        # even without an ordinalPosition key present.
        {"condition": "baseline", "promptID": "p3",
         "instrument": "answerTokenLogprob", "target": "1",
         "targetSource": None, "logOdds": {"1": 3.0}},
    ]
    endpoints = tasks._endpoint_values(records)
    assert endpoints["ordinalPosition"]["baseline"] == {"p1": 2.3}
    assert endpoints["choiceLogOdds"] == {"baseline": {"p2": -0.5}}


def test_choice_deltas_skip_ordinal_records():
    """The choice-deltas artifact applies the same declared-target rule: a
    likert run must produce an EMPTY artifact, not pole-token deltas."""
    from steerlab_server.experiment import choice_deltas

    records = [
        {"condition": "baseline", "promptID": "p1",
         "instrument": "answerTokenLogprob", "target": "1",
         "logOdds": {"1": 1.0}, "ordinalPosition": 2.0},
        {"condition": "steered", "promptID": "p1",
         "instrument": "answerTokenLogprob", "target": "1",
         "logOdds": {"1": 5.0}, "ordinalPosition": 2.4},
    ]
    rows, summary = choice_deltas.rows(records)
    assert rows == []
    assert not summary.get("conditions")


def test_declared_target_map_overrides_record_heuristic():
    """The pinned task file is the authority: a mixed-instrument record
    (declared target AND ordinalPosition, the s4-framings shape) keeps its
    choice endpoint when the map says the item declared one — and a likert
    item loses it even if a record lacks the ordinal stamp."""
    from steerlab_server.experiment import tasks, choice_deltas

    records = [
        # s4-framings shape: declared target, ordinal stamp on same record.
        {"condition": "baseline", "promptID": "mix1",
         "instrument": "answerTokenLogprob", "target": "B",
         "logOdds": {"A": -2.0, "B": 2.0}, "ordinalPosition": 1.9},
        {"condition": "steered", "promptID": "mix1",
         "instrument": "answerTokenLogprob", "target": "B",
         "logOdds": {"A": -1.0, "B": 1.0}, "ordinalPosition": 1.5},
        # likert shape whose record happens to lack the ordinal stamp.
        {"condition": "baseline", "promptID": "lik1",
         "instrument": "answerTokenLogprob", "target": "1",
         "logOdds": {"1": 8.0}},
    ]
    declared = {"mix1": True, "lik1": False}
    endpoints = tasks._endpoint_values(records, declared_targets=declared)
    assert endpoints["choiceLogOdds"]["baseline"] == {"mix1": 2.0}
    assert endpoints["choiceLogOdds"]["steered"] == {"mix1": 1.0}
    rows, summary = choice_deltas.rows(records, declared_targets=declared)
    assert [r[1] for r in rows] == ["mix1"]
