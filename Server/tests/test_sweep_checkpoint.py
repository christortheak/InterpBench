"""Sweep checkpoint/resume (2026-08-03).

The field incident: a bundle-submitted sweep hit its 30-minute walltime,
received the SIGUSR1 warning, and died with ~20 minutes of finished cells —
the sbatch harness was ready for exit-85 resumable checkpoints (study runs
use them) but the sweep verb had no handler. Now the sweep parks between
cells: completed rows and per-concept recommendations are durably appended
to ``sweep-progress.jsonl`` as they happen, a checkpoint writes
``resume-state.json`` (verb "sweep") and raises ``CheckpointRequested``
(the caller's exit-85 path), and a resume skips exactly the completed work.
``recommendations.json`` is the sweep's completion marker for the resume
pointer.
"""

import json
import os
from types import SimpleNamespace
from contextlib import contextmanager

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import logprob, resume, tasks
from steerlab_server.steering.vector_store import ConceptVectors


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _workspace(root, name, concepts=("fear",)):
    for concept in concepts:
        d = os.path.join(root, "prompts", "concepts", concept)
        _write(os.path.join(d, "positive.jsonl"), '{"text": "I feel dread"}\n')
        _write(os.path.join(d, "negative.jsonl"), '{"text": "calm morning"}\n')
        _write(os.path.join(d, "markers.json"),
               json.dumps({"words": ["dread"]}))
    _write(os.path.join(root, "prompts", "dev", "dev.jsonl"),
           '{"text": "Write about the town."}\n')
    _write(os.path.join(root, "prompts", "batteries", "b.jsonl"),
           '{"prompt": "What is 1+1?", "answer": "2"}\n')
    _write(os.path.join(root, "prompts", "dev", "choices.jsonl"),
           '{"id": "c1", "prompt": "p", "options": ["A", "B"]}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, list(concepts), root=root)
    d = es.load_raw(name, root)
    d["sweep"] = {
        "layerFractions": [0.5], "alphas": [0.1, 0.2],
        "devPromptsFile": "prompts/dev/dev.jsonl",
        "batteryFile": "prompts/batteries/b.jsonl", "maxTokens": 16,
        "selection": {"objective": {"metric": "logprobShift",
                                    "choicePromptsFile": "prompts/dev/choices.jsonl"}}}
    es.save_raw(d, root)


@contextmanager
def _fake_model(model_id, revision):
    yield SimpleNamespace(revision=revision)


def _fake_bundle():
    return tasks.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source="test", stimulus_hash="h")


def _fake_score(model, manifest, prompt, options, injections):
    gain = 0.5 if injections else 0.0
    return logprob.ChoiceResult(options=[
        logprob.OptionScore(option=option, token_ids=[7],
                            token_logprobs=[(-1.0 + gain) if i == 0 else -3.0])
        for i, option in enumerate(options)])


def _counting_generate(counter):
    def generate(model, prompt, *, injections=None, **kwargs):
        counter.append(prompt)
        return "the town woke slowly 2"
    return generate


def test_sweep_checkpoints_between_cells_and_resumes_without_rework(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    _workspace(root, "cp")
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "_score_choice", _fake_score)

    # Phase 1: request the checkpoint after the FIRST cell finishes — the
    # trigger fires from inside generate, the poll observes it at the next
    # cell boundary. Baseline (1 dev + 1 battery) + cell 1 (1 dev + 1
    # battery) = 4 generations before the flag trips.
    flag = resume.CheckpointFlag()
    calls: list = []

    def tripping_generate(model, prompt, *, injections=None, **kwargs):
        calls.append(prompt)
        if len(calls) == 4:
            flag.request()
        return "the town woke slowly 2"
    monkeypatch.setattr(tasks, "generate", tripping_generate)

    pointed: list = []
    with pytest.raises(resume.CheckpointRequested) as caught:
        tasks.sweep("cp", root, model_provider=_fake_model,
                    checkpoint=flag, on_run_directory=pointed.append,
                    log=lambda *_: None)
    run_dir = caught.value.run_directory
    assert pointed == [run_dir]
    assert caught.value.verb == "sweep"
    # Durable state: resume-state.json + the baseline and first cell rows.
    assert resume.is_resumable(run_dir, "sweep")
    rows, recs = tasks._load_sweep_progress(run_dir)
    assert [(r["layer"], r["alpha"]) for r in rows] == [(-1, 0), (2, 0.1)]
    assert recs == {}
    assert not os.path.exists(os.path.join(run_dir, "recommendations.json"))

    # Phase 2: resume. Only the SECOND cell generates (1 dev + 1 battery);
    # the baseline and first cell come from the durable rows.
    resumed_calls: list = []
    monkeypatch.setattr(tasks, "generate", _counting_generate(resumed_calls))
    out_dir = tasks.sweep("cp", root, model_provider=_fake_model,
                          run_directory=run_dir, log=lambda *_: None)
    assert out_dir == run_dir
    assert len(resumed_calls) == 2
    # Completion marker + full grid + a real recommendation.
    assert resume.is_complete(run_dir, "sweep")
    with open(os.path.join(run_dir, "sweep.csv"), encoding="utf-8") as h:
        lines = h.read().strip().splitlines()
    assert len(lines) == 1 + 3  # header + baseline + two cells
    with open(os.path.join(run_dir, "recommendations.json"),
              encoding="utf-8") as h:
        recs = json.load(h)
    assert recs["fear"]["winningCell"]["layer"] == 2
    # Lifecycle closes (review 2026-08-03, P2): completion clears the
    # resume pointer, and a direct caller aiming run_directory at the
    # complete sweep is refused by the verb-aware gate.
    assert resume.read_state(run_dir) is None
    assert not resume.is_resumable(run_dir, "sweep")
    with pytest.raises(resume.ResumeError, match="recommendations.json"):
        tasks.sweep("cp", root, model_provider=_fake_model,
                    run_directory=run_dir, log=lambda *_: None)


def test_multi_concept_draft_sweep_checkpoints_between_concepts(
        tmp_path, monkeypatch):
    """The field shape that broke (review 2026-08-03, P1): a DRAFT sweep
    over several concepts used to project each `<concept>-recommended`
    condition into the manifest as it finished, changing content_hash() —
    so a checkpoint taken after concept 1 was refused by the resume epoch
    guard. Projection is now deferred to durable completion: the manifest
    is untouched at the checkpoint, the resume succeeds, and BOTH concepts'
    conditions are projected at the end (resumed ones included)."""
    root = str(tmp_path)
    _workspace(root, "mc", concepts=("alpha", "beta"))
    monkeypatch.setattr(
        tasks, "_extract_all",
        lambda model, manifest, root: {"alpha": _fake_bundle(),
                                       "beta": _fake_bundle()})
    monkeypatch.setattr(tasks, "_score_choice", _fake_score)

    # Trip after concept alpha's grid: shared baseline (1 dev + 1 battery)
    # + two cells (2 × (1 dev + 1 battery)) = 6 generations. The poll
    # observes the flag at concept beta's top-of-loop boundary — AFTER
    # alpha's recommendation is journaled.
    flag = resume.CheckpointFlag()
    calls: list = []

    def tripping_generate(model, prompt, *, injections=None, **kwargs):
        calls.append(prompt)
        if len(calls) == 6:
            flag.request()
        return "the town woke slowly 2"
    monkeypatch.setattr(tasks, "generate", tripping_generate)

    with pytest.raises(resume.CheckpointRequested) as caught:
        tasks.sweep("mc", root, model_provider=_fake_model,
                    checkpoint=flag, log=lambda *_: None)
    run_dir = caught.value.run_directory
    rows, recs = tasks._load_sweep_progress(run_dir)
    assert set(recs) == {"alpha"}
    # The heart of the fix: alpha's recommendation is durable in the
    # JOURNAL, and the manifest carries NO projected condition yet.
    assert es.load_raw("mc", root).get("conditions", []) == []
    assert resume.is_resumable(run_dir, "sweep")

    # Resume completes concept beta only, then projects BOTH conditions.
    resumed_calls: list = []
    monkeypatch.setattr(tasks, "generate", _counting_generate(resumed_calls))
    out_dir = tasks.sweep("mc", root, model_provider=_fake_model,
                          run_directory=run_dir, log=lambda *_: None)
    assert out_dir == run_dir
    assert len(resumed_calls) == 6  # beta's baseline + two cells; alpha skipped
    with open(os.path.join(run_dir, "recommendations.json"),
              encoding="utf-8") as h:
        recs = json.load(h)
    assert set(recs) == {"alpha", "beta"}
    conditions = es.load_raw("mc", root)["conditions"]
    assert sorted(c["name"] for c in conditions) == [
        "alpha-recommended", "beta-recommended"]
    for condition in conditions:
        assert condition["selection"]["winningCell"] == {
            "layer": 2, "alpha": condition["slots"][0]["alpha"]}


def test_resume_refuses_a_changed_manifest(tmp_path, monkeypatch):
    root = str(tmp_path)
    _workspace(root, "cpx")
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "_score_choice", _fake_score)
    flag = resume.CheckpointFlag()
    flag.request()  # park immediately, before any cell
    monkeypatch.setattr(tasks, "generate",
                        _counting_generate([]))
    with pytest.raises(resume.CheckpointRequested) as caught:
        tasks.sweep("cpx", root, model_provider=_fake_model,
                    checkpoint=flag, log=lambda *_: None)
    run_dir = caught.value.run_directory
    # Edit the manifest: the checkpoint belongs to a different epoch now.
    d = es.load_raw("cpx", root)
    d["sweep"]["alphas"] = [0.1]
    es.save_raw(d, root)
    with pytest.raises(RuntimeError, match="manifest changed since the checkpoint"):
        tasks.sweep("cpx", root, model_provider=_fake_model,
                    run_directory=run_dir, log=lambda *_: None)


def test_cancelled_sweep_parks_and_resumes_to_completion(
        tmp_path, monkeypatch):
    """A cancel is an explicit resumable PARTIAL (review 2026-08-03 round 2,
    P1): no completion marker, a cancel-reason state file, and the ordinary
    resume path finishes the grid without regenerating completed cells."""
    root = str(tmp_path)
    _workspace(root, "cn")
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "_score_choice", _fake_score)
    calls: list = []

    def counting(model, prompt, *, injections=None, **kwargs):
        calls.append(prompt)
        return "the town woke slowly 2"
    monkeypatch.setattr(tasks, "generate", counting)

    # Cancel observable after cell 1's generations (baseline 2 + cell 2 = 4
    # calls): it is observed inside cell 1's choice scoring, so the
    # interrupted cell is DROPPED (existing cancel semantics) and only the
    # baseline row is durable.
    run_dir = tasks.sweep("cn", root, model_provider=_fake_model,
                          should_cancel=lambda: len(calls) >= 4,
                          log=lambda *_: None)
    assert not os.path.exists(os.path.join(run_dir, "recommendations.json"))
    state = resume.read_state(run_dir)
    assert state is not None and state["reason"] == "cancel"
    assert resume.is_resumable(run_dir, "sweep")
    rows, recs = tasks._load_sweep_progress(run_dir)
    assert [(r["layer"], r["alpha"]) for r in rows] == [(-1, 0)]
    assert recs == {}

    # Resume: the baseline comes from the journal; both grid cells
    # regenerate (2 × (1 dev + 1 battery)) and the run completes durably.
    resumed: list = []
    monkeypatch.setattr(tasks, "generate", _counting_generate(resumed))
    out_dir = tasks.sweep("cn", root, model_provider=_fake_model,
                          run_directory=run_dir, log=lambda *_: None)
    assert out_dir == run_dir
    assert len(resumed) == 4
    assert resume.is_complete(run_dir, "sweep")
    assert resume.read_state(run_dir) is None


def test_no_completion_marker_when_projection_fails(tmp_path, monkeypatch):
    """The marker is written atomically LAST: if draft-condition projection
    dies, recommendations.json must not exist — the directory reads as an
    honest partial (not complete, not resumable) and a fresh rerun
    re-projects idempotently."""
    root = str(tmp_path)
    _workspace(root, "pf")
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "_score_choice", _fake_score)
    monkeypatch.setattr(tasks, "generate", _counting_generate([]))

    def exploding_add_conditions(name, conditions, root=None):
        raise OSError("disk full")
    # tasks.py imports experiment_store function-locally — patch the module.
    # Projection is the ATOMIC batch verb (round 3, P2): one load, one save.
    monkeypatch.setattr(es, "add_conditions", exploding_add_conditions)
    captured: list = []
    with pytest.raises(OSError, match="disk full"):
        tasks.sweep("pf", root, model_provider=_fake_model,
                    on_run_directory=captured.append, log=lambda *_: None)
    run_dir = captured[0]
    assert not os.path.exists(os.path.join(run_dir, "recommendations.json"))
    assert not resume.is_complete(run_dir, "sweep")
    assert not resume.is_resumable(run_dir, "sweep")
    # All-or-nothing: the failed projection left NO partial condition.
    assert es.load_raw("pf", root).get("conditions", []) == []


def test_recommendations_json_is_the_sweep_completion_marker(tmp_path):
    d = str(tmp_path)
    assert not resume.is_complete(d, "sweep")
    _write(os.path.join(d, "recommendations.json"), "{}")
    assert resume.is_complete(d, "sweep")
    # The run verb's marker is unchanged.
    assert not resume.is_complete(d, "run")
    _write(os.path.join(d, "report.json"), "{}")
    assert resume.is_complete(d, "run")
