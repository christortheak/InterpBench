"""Evaluate's local-judge model resolution (unified with the sweep rule,
2026-07-22 incident).

A researcher's evaluate run died on an offline compute node because the
local branch of ``_judge_callable`` resolved ``ref.model or ref.name`` — the
judge's NAME ('judge-1') was sent to HuggingFace as a model id. The name is a
LABEL: a local judge with an empty/absent model now resolves to the STUDY
model (manifest modelID + pinned revision) through the normal provider slot,
exactly like the sweep. Contract under test: the researcher's exact shape
(two pinned local judges named judge-1/judge-2, empty model) judges through
the study model and NEVER requests any other model id; a different-model
local judge on a single-slot server refuses at start with the
second-resident wording; a genuinely-declared judge model that fails to load
raises the plain-language wrapped error, never the raw HF network dump; and
the legacy single-judge roster synthesis (name = model = a model id) still
resolves as a model.
"""

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering import model_loader

RUBRIC = "Which response expresses more dread?"
VERDICT = '{"winner": "A", "confidence": 0.9, "reasoning": "r"}'


def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = (json.dumps(content, indent=2) if isinstance(content, dict)
            else content)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _fixture(tmp_path, *, judges, evaluation=None):
    """A draft study with a pinned rubric + judge panel and one completed
    paired run (the researcher's shape: baseline + one condition)."""
    root = str(tmp_path)
    rubric_hash = _write(os.path.join(root, "prompts", "rubrics", "r.md"),
                         RUBRIC)
    d = {"name": "lj", "modelID": "org/study-model",
         "modelRevision": "abc123", "status": "draft",
         "judgeRubricFile": "prompts/rubrics/r.md",
         "judgeRubricHash": rubric_hash}
    if judges is not None:
        d["judges"] = judges
    if evaluation is not None:
        d["evaluation"] = evaluation
    _write(os.path.join(root, "experiments", "lj", "experiment.json"), d)
    run_dir = os.path.join(root, "runs", "20260101T000000000-exp-lj-run")
    os.makedirs(run_dir)
    _write(os.path.join(run_dir, "experiment-hash.txt"),
           Manifest.from_dict(d).content_hash() + "\n")
    generations = [
        {"promptID": "p0", "seed": 0, "condition": "baseline",
         "prompt": "Describe the cellar.", "output": "calm"},
        {"promptID": "p0", "seed": 0, "condition": "fear",
         "prompt": "Describe the cellar.", "output": "scared"},
    ]
    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for g in generations:
            handle.write(json.dumps(g) + "\n")
    return root


def _recording_provider(requested, *, fail_for=None):
    """A fake registry provider: records every (model_id, revision) it is
    asked for; optionally raises the raw HF-flavored load error for one id."""
    @contextmanager
    def provider(model_id, revision=None):
        requested.append((model_id, revision))
        if fail_for is not None and model_id == fail_for:
            raise model_loader.ModelLoadError(
                f"could not load '{model_id}': We couldn't connect to "
                "'https://huggingface.co' to load this file")
        yield SimpleNamespace(model_id=model_id)
    return provider


def _fake_generate(monkeypatch):
    def generate(model, prompt, *, model_id=None, max_tokens=0,
                 temperature=0.0, injections=None, prompt_mode=None,
                 system_prompt=None, qwen_thinking_enabled=False):
        return VERDICT
    monkeypatch.setattr(tasks, "generate", generate)


# --- the researcher's exact shape ------------------------------------------------

def test_empty_model_local_judges_judge_through_the_study_model(
        tmp_path, monkeypatch):
    # Two pinned local judges named judge-1/judge-2 with EMPTY model — the
    # UI's "study model (default)". The dead fallback sent 'judge-1' to
    # HuggingFace as a model id; the unified rule resolves both judges to
    # the study model at its pinned revision and never requests anything
    # else.
    root = _fixture(tmp_path, judges=[
        {"name": "judge-1", "kind": "local"},
        {"name": "judge-2", "kind": "local", "model": "  "}])
    _fake_generate(monkeypatch)
    requested: list = []
    logs: list = []

    eval_dir = tasks.evaluate(
        "lj", root=root, model_provider=_recording_provider(requested),
        max_loaded=1, log=lambda *parts: logs.append(" ".join(
            str(p) for p in parts)))

    # Every acquisition is the study model at its pinned revision — the
    # judge names were never treated as model ids.
    assert requested and set(requested) == {("org/study-model", "abc123")}
    report = json.load(open(os.path.join(eval_dir, "judge-report.json")))
    assert [b["name"] for b in report["judges"]] == ["judge-1", "judge-2"]
    assert {b["requestedModel"] for b in report["judges"]} == {
        "org/study-model"}
    assert {b["actualModel"] for b in report["judges"]} == {
        "org/study-model"}
    assert report["pairs"] == 1
    # The resolution is logged at evaluate start (cross-engine wording).
    assert any("local judge 'judge-1' resolves to the study model "
               "org/study-model" in line for line in logs)
    assert any("local judge 'judge-2' resolves to the study model "
               "org/study-model" in line for line in logs)


# --- different-model local judges -------------------------------------------------

def test_different_model_local_judge_refuses_on_a_single_slot_server(
        tmp_path, monkeypatch):
    root = _fixture(tmp_path, judges=[
        {"name": "judge-1", "kind": "local", "model": "org/other-judge"},
        {"name": "judge-2", "kind": "local"}])
    _fake_generate(monkeypatch)
    requested: list = []

    with pytest.raises(RuntimeError) as excinfo:
        tasks.evaluate(
            "lj", root=root, model_provider=_recording_provider(requested),
            max_loaded=1, log=lambda *_: None)

    message = str(excinfo.value)
    assert "local judge 'judge-1'" in message
    assert "org/other-judge" in message
    assert "needs a second resident model" in message
    assert "leave the judge's model empty" in message
    # Refused at evaluate start — no model was ever requested.
    assert requested == []


def test_different_model_local_judge_runs_with_capacity_for_two(
        tmp_path, monkeypatch):
    # max_loaded >= 2 keeps the second-resident-model semantics: the
    # declared model is requested (unpinned — the study revision pins only
    # the study model), never the judge's name.
    root = _fixture(tmp_path, judges=[
        {"name": "judge-1", "kind": "local", "model": "org/other-judge"}])
    _fake_generate(monkeypatch)
    requested: list = []

    eval_dir = tasks.evaluate(
        "lj", root=root, model_provider=_recording_provider(requested),
        max_loaded=2, log=lambda *_: None)

    assert set(requested) == {("org/other-judge", None)}
    report = json.load(open(os.path.join(eval_dir, "judge-report.json")))
    assert report["judges"][0]["requestedModel"] == "org/other-judge"


def test_declared_judge_model_load_failure_is_wrapped_honestly(
        tmp_path, monkeypatch):
    # A genuinely-declared judge model that cannot load must name the
    # judge, the model, and the remedies — never surface the raw HF
    # "couldn't connect" dump (misleading on an air-gapped node).
    root = _fixture(tmp_path, judges=[
        {"name": "judge-1", "kind": "local", "model": "org/other-judge"}])
    _fake_generate(monkeypatch)

    with pytest.raises(model_loader.ModelLoadError) as excinfo:
        tasks.evaluate(
            "lj", root=root,
            model_provider=_recording_provider([], fail_for="org/other-judge"),
            log=lambda *_: None)

    message = str(excinfo.value)
    assert "local judge 'judge-1'" in message
    assert "org/other-judge" in message
    assert "install the model on the server" in message
    assert "leave the judge's model empty" in message
    assert "huggingface.co" not in message
    # The original loader error stays chained for debugging.
    assert "huggingface.co" in str(excinfo.value.__cause__)


def test_judge_load_capacity_refusal_carries_the_loaders_own_advice(
        tmp_path, monkeypatch):
    # The loader's TYPED refusals (advice_complete) carry complete remedies —
    # "unload the other model, use the study model as judge" — and the judge
    # wrapper must not overwrite them with "install the model": the model IS
    # installed, and the co-residency calibration failure (2026-08-28) sent
    # an agent to reinstall it on that advice.
    root = _fixture(tmp_path, judges=[
        {"name": "judge-1", "kind": "local", "model": "org/other-judge"}])
    _fake_generate(monkeypatch)

    @contextmanager
    def capacity_refusing(model_id, revision=None):
        if model_id == "org/other-judge":
            raise model_loader.ModelLoadError(
                "'org/other-judge' needs ~22.7 GiB of weights but only "
                "23.3 GiB of cuda (A100)'s 79.3 GiB is free — another model "
                "is already resident. Two models fit only if the device has "
                "room for both: unload the other model, use the study model "
                "as judge, or pin an external judge.",
                advice_complete=True)
        yield SimpleNamespace(model_id=model_id)

    with pytest.raises(model_loader.ModelLoadError) as excinfo:
        tasks.evaluate("lj", root=root, model_provider=capacity_refusing,
                       log=lambda *_: None)

    message = str(excinfo.value)
    assert "local judge 'judge-1'" in message
    assert "another model is already resident" in message
    assert "unload the other model" in message
    assert "install the model" not in message
    assert getattr(excinfo.value, "advice_complete", False)
    # The chained cause is the capacity refusal itself here.
    assert "another model is already resident" in str(excinfo.value.__cause__)


def test_study_model_load_failure_is_not_wrapped(tmp_path, monkeypatch):
    # When the STUDY model itself fails to load there is no judge-declared
    # model to remedy — the loader error propagates unwrapped.
    root = _fixture(tmp_path, judges=[{"name": "judge-1", "kind": "local"}])
    _fake_generate(monkeypatch)

    with pytest.raises(model_loader.ModelLoadError) as excinfo:
        tasks.evaluate(
            "lj", root=root,
            model_provider=_recording_provider(
                [], fail_for="org/study-model"),
            log=lambda *_: None)

    assert "could not load 'org/study-model'" in str(excinfo.value)


# --- the legacy single-judge synthesis --------------------------------------------

def test_legacy_single_judge_roster_still_resolves_as_a_model(
        tmp_path, monkeypatch):
    # No pinned judges: the roster synthesizes one judge from
    # evaluation.judgeModel with name AND model set to the model id — which
    # remains coherent under the unified rule (the declared model wins; the
    # study model is never substituted).
    root = _fixture(
        tmp_path, judges=None,
        evaluation={"kind": "pairedJudge", "judgeModel": "org/legacy-judge"})
    _fake_generate(monkeypatch)
    requested: list = []

    eval_dir = tasks.evaluate(
        "lj", root=root, model_provider=_recording_provider(requested),
        max_loaded=2, log=lambda *_: None)

    assert set(requested) == {("org/legacy-judge", None)}
    report = json.load(open(os.path.join(eval_dir, "judge-report.json")))
    assert report["judges"][0]["name"] == "org/legacy-judge"
    assert report["judges"][0]["requestedModel"] == "org/legacy-judge"
