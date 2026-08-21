"""Effective-evaluation declaration (2026-07-22 incident).

A researcher's FROZEN study carried pinned judges + a pinned rubric file,
``pipeline: {stages: [run, evaluate, analyze]}`` — and ``evaluation: null``,
because the app's rubric-FILE path never wrote the explicit block. The
pipeline generated successfully, then died at the evaluate stage. Contract
under test (cross-engine — Swift twin: EvaluationDeclarationTests):

- pinned judges + a pinned rubric file ARE a paired-judge declaration:
  evaluate synthesizes the spec from those pins and stamps the judge report
  ``evaluationSource: "pinnedRubric"``; an explicit ``evaluation`` block
  always wins and stamps ``"manifest"``;
- the researcher's exact manifest shape now evaluates successfully WITHOUT
  re-freezing (the regression fixture mirrors it);
- verify() carries a coherence gate: a pipeline declaring evaluate with no
  effective evaluation is a violation with the shared remedy wording;
- the pipeline preflight refuses the same incoherence with the same words;
- the gateless-pipeline log line is calm when the chain has no gateable
  stage and stays a WARNING when one runs ungated.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import (
    EVALUATE_WITHOUT_JUDGING_MESSAGE, Manifest)

RUBRIC_TEXT = "Which response expresses more dread?"


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _fixture(tmp_path, *, status="frozen", evaluation=None, judges=True,
             pin_rubric=True, pipeline=None):
    """The researcher's manifest shape by default: frozen, NO evaluation
    block, two pinned judges, a hash-pinned rubric file, and a
    run → evaluate → analyze pipeline. Plus a source run to judge."""
    root = str(tmp_path)
    d = {"name": "ev", "modelID": "org/m", "status": status}
    if evaluation is not None:
        d["evaluation"] = evaluation
    if judges:
        d["judges"] = [
            {"name": "j-variant", "kind": "claude", "model": "claude-a"},
            {"name": "j-baseline", "kind": "claude", "model": "claude-b"}]
    if pin_rubric:
        rubric_hash = _write(
            os.path.join(root, "prompts", "rubrics", "default-paired-v1.md"),
            RUBRIC_TEXT)
        d["judgeRubricFile"] = "prompts/rubrics/default-paired-v1.md"
        d["judgeRubricHash"] = rubric_hash
    d["pipeline"] = pipeline if pipeline is not None else {
        "stages": ["run", "evaluate", "analyze"]}
    _write(os.path.join(root, "experiments", "ev", "experiment.json"),
           json.dumps(d, indent=2))
    run_dir = os.path.join(root, "runs", "20260722T000000000-exp-ev-run")
    os.makedirs(run_dir)
    _write(os.path.join(run_dir, "experiment-hash.txt"),
           Manifest.from_dict(d).content_hash() + "\n")
    generations = [
        {"promptID": "p0", "seed": 0, "condition": "baseline", "output": "calm"},
        {"promptID": "p0", "seed": 0, "condition": "fear", "output": "scared"},
        {"promptID": "p1", "seed": 0, "condition": "baseline", "output": "fine"},
        {"promptID": "p1", "seed": 0, "condition": "fear", "output": "afraid"},
    ]
    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for g in generations:
            handle.write(json.dumps(g) + "\n")
    return root


def _fake_judge_pair(model, rubric, a, b, structured=None, task_prompt=None):
    assert rubric == RUBRIC_TEXT  # judged from the PINNED FILE
    return {"winner": "A", "confidence": 0.9}


# --- Fix 1: pinned judges + rubric ARE the declaration ------------------------

def test_frozen_manifest_with_pins_and_null_evaluation_evaluates(
        tmp_path, monkeypatch):
    """THE regression: the researcher's exact manifest shape (frozen,
    evaluation absent, judges + rubric pinned, run→evaluate→analyze
    pipeline) evaluates successfully without re-freezing."""
    from steerlab_server.experiment import paired_judge
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    root = _fixture(tmp_path)
    monkeypatch.setattr(paired_judge, "judge_pair", _fake_judge_pair)
    out = tasks.evaluate("ev", root=root, log=lambda *_: None)

    report = json.load(open(os.path.join(out, "judge-report.json")))
    assert report["evaluationSource"] == "pinnedRubric"
    assert report["rubricFile"] == "prompts/rubrics/default-paired-v1.md"
    assert [b["name"] for b in report["judges"]] == ["j-variant", "j-baseline"]
    assert report["pairs"] == 2


def test_explicit_evaluation_block_wins_and_stamps_manifest(
        tmp_path, monkeypatch):
    from steerlab_server.experiment import paired_judge
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    root = _fixture(tmp_path, evaluation={
        "kind": "pairedJudge", "judgeModel": "claude-opus-4-8"})
    monkeypatch.setattr(paired_judge, "judge_pair", _fake_judge_pair)
    out = tasks.evaluate("ev", root=root, log=lambda *_: None)
    report = json.load(open(os.path.join(out, "judge-report.json")))
    assert report["evaluationSource"] == "manifest"


def test_evaluate_still_refuses_with_no_declaration_at_all(tmp_path):
    # Judges alone (no rubric file) never produced a judged report on any
    # code version — the refusal stays, now with the remedy in it.
    root = _fixture(tmp_path, status="draft", judges=True, pin_rubric=False,
                    pipeline={"stages": ["run"]})
    with pytest.raises(RuntimeError,
                       match="no pairedJudge evaluation configured"):
        tasks.evaluate("ev", root=root, log=lambda *_: None)


def test_synthesized_spec_still_refuses_rubric_drift(tmp_path, monkeypatch):
    from steerlab_server.experiment import paired_judge
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    root = _fixture(tmp_path)
    with open(os.path.join(root, "prompts", "rubrics",
                           "default-paired-v1.md"), "a",
              encoding="utf-8") as handle:
        handle.write("EDITED\n")
    monkeypatch.setattr(paired_judge, "judge_pair", _fake_judge_pair)
    with pytest.raises(RuntimeError, match="drifted from the pinned hash"):
        tasks.evaluate("ev", root=root, log=lambda *_: None)


def test_deferred_emission_carries_evaluation_source(tmp_path, monkeypatch):
    # External judges with no credential DEFER; the packets carry the spec
    # provenance so the completed report matches the inline path's stamp.
    root = _fixture(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE",
                       os.path.join(root, "no-such-key"))
    out = tasks.evaluate("ev", root=root, log=lambda *_: None)
    assert os.path.exists(os.path.join(out, "awaiting-judgment.json"))
    jm = json.load(open(os.path.join(out, "judging-manifest.json")))
    assert jm["evaluationSource"] == "pinnedRubric"


# --- Fix 2: verify() coherence gate ------------------------------------------

def test_verify_flags_evaluate_stage_with_no_judging(tmp_path):
    root = _fixture(tmp_path, status="draft", judges=False, pin_rubric=False)
    violations = Manifest.load("ev", root).verify(root)
    assert EVALUATE_WITHOUT_JUDGING_MESSAGE in violations


def test_verify_accepts_judges_plus_rubric_as_declaration(tmp_path):
    # The researcher's shape is COHERENT under the resolution rule — no
    # violation, no re-freeze needed.
    root = _fixture(tmp_path)
    violations = Manifest.load("ev", root).verify(root)
    assert EVALUATE_WITHOUT_JUDGING_MESSAGE not in violations


def test_verify_ignores_pipelines_without_evaluate(tmp_path):
    root = _fixture(tmp_path, status="draft", judges=False, pin_rubric=False,
                    pipeline={"stages": ["run", "analyze"]})
    violations = Manifest.load("ev", root).verify(root)
    assert EVALUATE_WITHOUT_JUDGING_MESSAGE not in violations


def test_verify_flags_explicit_none_evaluation_with_evaluate_stage(tmp_path):
    # An explicit block always wins — kind "none" declares NO judging even
    # next to pinned judges, so evaluate could never run.
    root = _fixture(tmp_path, status="draft", evaluation={"kind": "none"})
    violations = Manifest.load("ev", root).verify(root)
    assert EVALUATE_WITHOUT_JUDGING_MESSAGE in violations


# --- pipeline preflight + gates copy -----------------------------------------

def _concept_workspace(root, *, pipeline):
    d = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(d, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(d, "negative.jsonl"), '{"text": "calm morning"}\n')
    es.create("chain", model_id="org/m", revision="abc", root=root)
    es.attach("chain", ["fear"], root=root)
    raw = es.load_raw("chain", root)
    raw["pipeline"] = pipeline
    es.save_raw(raw, root)
    return "chain"


@contextmanager
def _fake_provider():
    @contextmanager
    def provider(model_id, revision=None):
        yield SimpleNamespace(revision=revision or "abc")
    yield provider


def _fake_run_and_analyze(monkeypatch, root):
    def fake_run(name, prompts_file=None, r=None, dtype="auto", device=None,
                 **kwargs):
        run_dir = kwargs.get("run_directory") or os.path.join(
            root, "runs", "20260722T000000001-exp-chain-run")
        os.makedirs(run_dir, exist_ok=True)
        if kwargs.get("on_run_directory"):
            kwargs["on_run_directory"](run_dir)
        _write(os.path.join(run_dir, "report.json"), "{}")
        return run_dir

    def fake_analyze(name, r=None, source_run=None, **kwargs):
        run_dir = os.path.join(root, "runs",
                               "20260722T000000002-exp-chain-analyze")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    monkeypatch.setattr(tasks, "run", fake_run)
    monkeypatch.setattr(tasks, "analyze", fake_analyze)


def test_pipeline_preflight_refuses_evaluate_without_judging(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    name = _concept_workspace(
        root, pipeline={"stages": ["run", "evaluate", "analyze"]})
    with _fake_provider() as provider:
        with pytest.raises(RuntimeError) as excinfo:
            tasks.pipeline(name, root, model_provider=provider,
                           log=lambda *_: None)
    assert EVALUATE_WITHOUT_JUDGING_MESSAGE in str(excinfo.value)


def test_gateless_run_analyze_chain_logs_calm_information(
        tmp_path, monkeypatch):
    # No gateable stage in the chain → the informational line, no WARNING.
    root = str(tmp_path)
    name = _concept_workspace(root, pipeline={"stages": ["run", "analyze"]})
    _fake_run_and_analyze(monkeypatch, root)
    lines: list = []
    with _fake_provider() as provider:
        tasks.pipeline(name, root, model_provider=provider,
                       log=lines.append)
    gates_lines = [l for l in lines if "declares no gates" in l]
    assert gates_lines == [
        "pipeline declares no gates — a run → analyze chain has none to "
        "declare"]
    assert not any(l.startswith("WARNING") for l in gates_lines)


def test_gateless_chain_with_gateable_stage_still_warns(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    name = _concept_workspace(root, pipeline={"stages": ["validate", "run"]})

    def fake_validate(name, r=None, dtype="auto", device=None, **kwargs):
        run_dir = os.path.join(root, "runs",
                               "20260722T000000003-exp-chain-validate")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    monkeypatch.setattr(tasks, "validate", fake_validate)
    _fake_run_and_analyze(monkeypatch, root)
    lines: list = []
    with _fake_provider() as provider:
        tasks.pipeline(name, root, model_provider=provider,
                       log=lines.append)
    assert any(l.startswith("WARNING: pipeline declares no gates")
               for l in lines)
