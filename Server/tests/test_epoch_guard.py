"""Epoch guard for evaluate/analyze (2026-07-13): a source run is eligible
ONLY if its stamped experiment hash equals the live manifest's content hash —
a pre-edit draft run can no longer be judged/analyzed under a frozen manifest.
Legacy unstamped runs refuse unless ``allowUnverifiedEpoch`` accepts them, and
the output is then stamped ``epochUnverified: true``. CLI flag:
``--allow-unverified-epoch``; HTTP body key: ``allowUnverifiedEpoch``."""

import hashlib
import json
import os
import time

import pytest

from steerlab_server.experiment import experiment_store as es, tasks
from steerlab_server.experiment.manifest import Manifest

RUBRIC_TEXT = "Prefer the more fearful response.\n"


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = payload if isinstance(payload, str) else json.dumps(payload)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(data)
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _analyze_fixture(tmp_path, *, stamp="hash-file"):
    """A minimal manifest + one source run with sentencing records. ``stamp``
    chooses how the run's epoch is recorded: 'hash-file' (experiment-hash.txt),
    'config' (config.json experimentHash), or None (legacy unstamped)."""
    root = str(tmp_path)
    manifest_dict = {
        "name": "study", "modelID": "org/m", "concepts": [],
        "conditions": [{"name": "fear-a2",
                        "slots": [{"concept": "fear", "layer": 1, "alpha": 2.0}]}]}
    _write(os.path.join(root, "experiments", "study", "experiment.json"),
           manifest_dict)
    run_dir = os.path.join(root, "runs", "20260101T000000000-exp-study-run")
    os.makedirs(run_dir)
    live = Manifest.from_dict(manifest_dict).content_hash()
    if stamp == "hash-file":
        _write(os.path.join(run_dir, "experiment-hash.txt"), live + "\n")
    elif stamp == "config":
        _write(os.path.join(run_dir, "config.json"),
               {"schemaVersion": 2, "runType": "run", "experimentHash": live})
    records = []
    for condition, shift in (("baseline", 0.0), ("fear-a2", 6.0)):
        for prompt_id, months in (("p1", 12.0), ("p2", 18.0)):
            records.append({"condition": condition, "promptID": prompt_id,
                            "output": "text", "wordCount": 5, "distinct2": 0.9,
                            "parsedMonths": months + shift})
    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return root, run_dir


def _evaluate_fixture(tmp_path, *, stamp=True):
    root = str(tmp_path)
    rubric_hash = _write(os.path.join(root, "prompts", "rubrics", "r.md"),
                         RUBRIC_TEXT)
    d = {"name": "ev", "modelID": "org/m", "status": "draft",
         "evaluation": {"kind": "pairedJudge", "judgeModel": "claude-a",
                        "judgePrompt": "inline"},
         "judgeRubricFile": "prompts/rubrics/r.md",
         "judgeRubricHash": rubric_hash,
         "judges": [{"name": "j1", "kind": "claude", "model": "claude-a"},
                    {"name": "j2", "kind": "claude", "model": "claude-b"}]}
    _write(os.path.join(root, "experiments", "ev", "experiment.json"), d)
    run_dir = os.path.join(root, "runs", "20260101T000000000-exp-ev-run")
    os.makedirs(run_dir)
    if stamp:
        _write(os.path.join(run_dir, "experiment-hash.txt"),
               Manifest.from_dict(d).content_hash() + "\n")
    generations = [
        {"promptID": "p0", "seed": 0, "condition": "baseline", "output": "calm"},
        {"promptID": "p0", "seed": 0, "condition": "fear", "output": "scared"},
    ]
    with open(os.path.join(run_dir, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for g in generations:
            handle.write(json.dumps(g) + "\n")
    return root, run_dir


def _fake_judge(model, rubric, a, b, structured=None, task_prompt=None):
    return {"winner": "tie", "confidence": 0.5}


# --- analyze -----------------------------------------------------------------

def test_analyze_accepts_matching_hash_file_stamp(tmp_path):
    root, _run = _analyze_fixture(tmp_path, stamp="hash-file")
    out = tasks.analyze("study", root=root)
    assert os.path.exists(os.path.join(out, "effect-sizes.csv"))
    assert not os.path.exists(os.path.join(out, "epoch-unverified.json"))
    config = json.load(open(os.path.join(out, "config.json")))
    assert config["notes"] == {}


def test_analyze_accepts_config_json_stamp_fallback(tmp_path):
    root, _run = _analyze_fixture(tmp_path, stamp="config")
    out = tasks.analyze("study", root=root)
    assert os.path.exists(os.path.join(out, "effect-sizes.csv"))


def test_analyze_refuses_epoch_mismatch_naming_run_and_hashes(tmp_path):
    root, run_dir = _analyze_fixture(tmp_path, stamp="hash-file")
    stamped = Manifest.load("study", root=root).content_hash()
    # The manifest drifts AFTER the run (a draft edit).
    d = es.load_raw("study", root)
    d["maxTokens"] = 999
    es.save_raw(d, root)
    live = Manifest.load("study", root=root).content_hash()
    with pytest.raises(RuntimeError) as excinfo:
        tasks.analyze("study", root=root)
    message = str(excinfo.value)
    assert os.path.basename(run_dir) in message
    assert stamped in message and live in message
    assert "re-run under the current manifest" in message.lower()
    assert "allowUnverifiedEpoch for legacy unstamped runs" in message
    # The flag does NOT bypass a stamped mismatch — only unstamped legacy runs.
    with pytest.raises(RuntimeError, match="different manifest epoch"):
        tasks.analyze("study", root=root, allow_unverified_epoch=True)


def test_analyze_refuses_unstamped_run_then_accepts_with_flag(tmp_path):
    root, run_dir = _analyze_fixture(tmp_path, stamp=None)
    with pytest.raises(RuntimeError, match="allowUnverifiedEpoch"):
        tasks.analyze("study", root=root)
    warnings = []
    out = tasks.analyze("study", root=root, allow_unverified_epoch=True,
                        log=warnings.append)
    assert any("epochUnverified" in w for w in warnings)
    stamp = json.load(open(os.path.join(out, "epoch-unverified.json")))
    assert stamp["epochUnverified"] is True
    assert stamp["sourceRun"] == os.path.basename(run_dir)
    config = json.load(open(os.path.join(out, "config.json")))
    assert config["notes"]["epochUnverified"] is True


def test_analyze_explicit_source_run_is_also_guarded(tmp_path):
    root, run_dir = _analyze_fixture(tmp_path, stamp=None)
    with pytest.raises(RuntimeError, match="allowUnverifiedEpoch"):
        tasks.analyze("study", root=root, source_run=run_dir)


# --- evaluate ----------------------------------------------------------------

def test_evaluate_accepts_matching_stamp(tmp_path, monkeypatch):
    from steerlab_server.experiment import paired_judge
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # inline custody
    monkeypatch.setattr(paired_judge, "judge_pair", _fake_judge)
    root, _run = _evaluate_fixture(tmp_path, stamp=True)
    out = tasks.evaluate("ev", root=root)
    report = json.load(open(os.path.join(out, "judge-report.json")))
    assert "epochUnverified" not in report


def test_evaluate_refuses_epoch_mismatch(tmp_path, monkeypatch):
    from steerlab_server.experiment import paired_judge
    monkeypatch.setattr(paired_judge, "judge_pair", _fake_judge)
    root, run_dir = _evaluate_fixture(tmp_path, stamp=True)
    d = es.load_raw("ev", root)
    d["temperature"] = 0.7
    es.save_raw(d, root)
    with pytest.raises(RuntimeError) as excinfo:
        tasks.evaluate("ev", root=root)
    assert os.path.basename(run_dir) in str(excinfo.value)
    assert "different manifest epoch" in str(excinfo.value)


def test_evaluate_unstamped_needs_flag_and_stamps_report(tmp_path, monkeypatch):
    from steerlab_server.experiment import paired_judge
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")  # inline custody
    monkeypatch.setattr(paired_judge, "judge_pair", _fake_judge)
    root, _run = _evaluate_fixture(tmp_path, stamp=False)
    with pytest.raises(RuntimeError, match="allowUnverifiedEpoch"):
        tasks.evaluate("ev", root=root)
    out = tasks.evaluate("ev", root=root, allow_unverified_epoch=True)
    report = json.load(open(os.path.join(out, "judge-report.json")))
    assert report["epochUnverified"] is True
    config = json.load(open(os.path.join(out, "config.json")))
    assert config["notes"]["epochUnverified"] is True


# --- CLI verb parity -----------------------------------------------------------

def test_cli_analyze_verb_runs_same_task(tmp_path):
    from steerlab_server import cli
    root, _run_dir = _analyze_fixture(tmp_path, stamp="hash-file")
    assert cli.main(["experiment", "analyze", "study", "--root", root]) == 0
    runs = os.listdir(os.path.join(root, "runs"))
    (analyze_run,) = [r for r in runs if "-exp-study-analyze" in r]
    out = os.path.join(root, "runs", analyze_run)
    assert os.path.exists(os.path.join(out, "effect-sizes.csv"))
    config = json.load(open(os.path.join(out, "config.json")))
    assert config["runType"] == "analyze"


def test_cli_analyze_allow_unverified_epoch_flag(tmp_path):
    from steerlab_server import cli
    root, _run = _analyze_fixture(tmp_path, stamp=None)
    # Without the flag the task raises (legacy unstamped run).
    with pytest.raises(RuntimeError, match="allowUnverifiedEpoch"):
        cli.main(["experiment", "analyze", "study", "--root", root])
    assert cli.main(["experiment", "analyze", "study", "--root", root,
                     "--allow-unverified-epoch"]) == 0
    runs = os.listdir(os.path.join(root, "runs"))
    (analyze_run,) = [r for r in runs if "-exp-study-analyze" in r]
    stamp = json.load(open(os.path.join(root, "runs", analyze_run,
                                        "epoch-unverified.json")))
    assert stamp["epochUnverified"] is True


def test_cli_evaluate_allow_unverified_epoch_flag(tmp_path, monkeypatch):
    from steerlab_server import cli
    from steerlab_server.experiment import paired_judge
    monkeypatch.setattr(paired_judge, "judge_pair", _fake_judge)
    root, _run = _evaluate_fixture(tmp_path, stamp=False)
    assert cli.main(["experiment", "evaluate", "ev", "--root", root,
                     "--allow-unverified-epoch"]) == 0


# --- HTTP body key parity --------------------------------------------------------

def _wait_for_job(client, job_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs = client.get("/api/jobs").json()["jobs"]
        job = next((j for j in jobs if j["id"] == job_id), None)
        if job and job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


def test_http_analyze_threads_allow_unverified_epoch(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from steerlab_server.api.app import app
    from steerlab_server.experiment import tasks as tasks_mod
    client = TestClient(app)

    seen = {}

    def fake_analyze(name, root=None, source_run=None, *, model_provider=None,
                     should_cancel=None, log=None, allow_unverified_epoch=False):
        seen[name] = allow_unverified_epoch
        return "/tmp/fake-analyze"

    monkeypatch.setattr(tasks_mod, "analyze", fake_analyze)
    job_id = client.post("/api/experiment/epoch-a/analyze",
                         json={"allowUnverifiedEpoch": True}).json()["jobId"]
    assert _wait_for_job(client, job_id)["status"] == "succeeded"
    assert seen["epoch-a"] is True

    # Omitted body defaults to the strict guard.
    job_id = client.post("/api/experiment/epoch-b/analyze").json()["jobId"]
    assert _wait_for_job(client, job_id)["status"] == "succeeded"
    assert seen["epoch-b"] is False


def test_http_evaluate_threads_allow_unverified_epoch(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from steerlab_server.api.app import app
    from steerlab_server.experiment import tasks as tasks_mod
    client = TestClient(app)

    seen = {}

    def fake_evaluate(name, root=None, source_run=None, *, model_provider=None,
                      should_cancel=None, log=None,
                      allow_unverified_epoch=False, max_loaded=None,
                      resume_from=None, **_kwargs):
        seen[name] = allow_unverified_epoch
        return "/tmp/fake-evaluate"

    monkeypatch.setattr(tasks_mod, "evaluate", fake_evaluate)
    job_id = client.post("/api/experiment/epoch-c/evaluate",
                         json={"allowUnverifiedEpoch": True}).json()["jobId"]
    assert _wait_for_job(client, job_id)["status"] == "succeeded"
    assert seen["epoch-c"] is True


def test_self_projected_conditions_are_the_runs_own_epoch(tmp_path):
    """Field incident 2026-08-04: the sweep projects <concept>-recommended
    AFTER its start-time snapshot (and the Mac adopts the same projection
    into its local draft) — the guard then refused the exact state the run
    itself produced. Conditions whose selection.sweepRun IS the checked run
    are the run's own completion, never a different epoch; conditions from
    any OTHER run still refuse."""
    from steerlab_server.experiment import run_epoch

    root = str(tmp_path)
    es.create("sp", model_id="org/m", revision="abc", root=root)
    before = Manifest.load("sp", root)
    run_name = "20260804T000000000-exp-sp-sweep"
    run_dir = os.path.join(root, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    _write(os.path.join(run_dir, "experiment-hash.txt"),
           before.content_hash())
    _write(os.path.join(run_dir, "experiment.json"), before.raw)

    es.add_condition("sp", {
        "name": "fear-recommended",
        "slots": [{"concept": "fear", "layer": 31, "alpha": 0.2}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": {"sweepRun": run_name,
                      "winningCell": {"layer": 31, "alpha": 0.2}}}, root)
    # The Studies panel stamps studyType when the study is OPENED —
    # absent vs declared-as-derived is the same study.
    d = es.load_raw("sp", root)
    d["studyType"] = "conceptStudy"
    es.save_raw(d, root)
    live = Manifest.load("sp", root)
    refusal, unverified, _ = run_epoch.epoch_refusal(
        "promote", "sp", live.content_hash(), run_dir,
        allow_unverified=False, live_manifest=live)
    assert refusal is None and unverified is False

    # A genuinely DIFFERENT declared type is a real epoch difference.
    d = es.load_raw("sp", root)
    d["studyType"] = "agentComparison"
    es.save_raw(d, root)
    diff_live = Manifest.load("sp", root)
    diff_refusal, _, _ = run_epoch.epoch_refusal(
        "promote", "sp", diff_live.content_hash(), run_dir,
        allow_unverified=False, live_manifest=diff_live)
    assert diff_refusal is not None and "studyType" in diff_refusal
    d = es.load_raw("sp", root)
    d["studyType"] = "conceptStudy"
    es.save_raw(d, root)

    # A projection from a DIFFERENT run is a real epoch difference.
    es.add_condition("sp", {
        "name": "anger-recommended",
        "slots": [{"concept": "anger", "layer": 37, "alpha": 0.1}],
        "bandWidth": 1, "alphaInNormUnits": True,
        "selection": {"sweepRun": "20260101T000000000-exp-sp-sweep",
                      "winningCell": {"layer": 37, "alpha": 0.1}}}, root)
    live2 = Manifest.load("sp", root)
    refusal2, _, _ = run_epoch.epoch_refusal(
        "promote", "sp", live2.content_hash(), run_dir,
        allow_unverified=False, live_manifest=live2)
    assert refusal2 is not None
    assert "anger-recommended" in refusal2
