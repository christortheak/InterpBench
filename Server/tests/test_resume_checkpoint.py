"""Record-level resume + checkpoint-on-signal (TURNKEY-CLUSTER-PLAN WS2).

Three layers, mirroring the reliability contract:

1. Pure units — record keys, resume-state round trip, the resume gate, and
   torn-tail recovery in ``load_completed``.
2. The writer loop — pre-populate K of N records, resume with a deterministic
   fake generator, and require exactly N−K appends plus byte equality with an
   uninterrupted run. Plus the same property through the REAL study loop
   (``tasks.run``) with faked extraction/generation, including the cooperative
   cancel path (reason "cancel") and the choice-instrument records.
3. The signal path — a real subprocess running the real helpers is sent
   SIGUSR1/SIGTERM, must exit 85 with a valid resume-state.json and a flushed
   JSONL, and a re-run must complete to a byte-identical union.

Bundle-execute wiring (the Slurm child entry point) is covered at the end:
pointer-based resume across a simulated requeue, idempotent already-complete
re-execution, and the child-record contract fields.
"""

import glob
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import bundles, experiment_store as es
from steerlab_server.experiment import resume
from steerlab_server.experiment import tasks
from steerlab_server.steering.vector_store import ConceptVectors

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(SERVER_DIR, "tests", "checkpoint_harness.py")


# --- record identity ----------------------------------------------------------

def test_record_key_separates_kinds_and_prompt_positions():
    sampled = {"condition": "c", "promptIndex": 0, "promptID": "p", "sampleIndex": 1,
               "output": "text"}
    instrument = {"condition": "c", "promptIndex": 0, "promptID": "p",
                  "instrument": "answerTokenLogprob"}
    error = {"condition": "c", "error": "boom"}
    keys = {resume.record_key(sampled), resume.record_key(instrument),
            resume.record_key(error)}
    assert len(keys) == 3
    # Duplicate prompt IDs at different positions stay distinct records.
    other_position = dict(sampled, promptIndex=7)
    assert resume.record_key(other_position) != resume.record_key(sampled)
    assert resume.record_key(sampled) == resume.make_key(
        "c", 0, "p", 1, resume.KIND_SAMPLED)


# --- resume-state.json ----------------------------------------------------------

def test_state_round_trip_and_contract_fields(tmp_path):
    run_dir = str(tmp_path)
    resume.write_state(run_dir, run_id="20260712-exp-x-run", verb="run",
                       completed_records=7, reason="signal")
    state = resume.read_state(run_dir)
    assert state["runId"] == "20260712-exp-x-run"
    assert state["verb"] == "run"
    assert state["completedRecords"] == 7
    assert state["reason"] == "signal"
    # updatedAt is ISO-8601 parseable.
    from datetime import datetime
    datetime.fromisoformat(state["updatedAt"])
    assert resume.is_resumable(run_dir)
    resume.clear_state(run_dir)
    assert resume.read_state(run_dir) is None
    resume.clear_state(run_dir)  # idempotent


def test_state_reason_is_constrained(tmp_path):
    with pytest.raises(ValueError, match="signal.*cancel"):
        resume.write_state(str(tmp_path), run_id="r", verb="run",
                           completed_records=0, reason="whim")


def test_resume_gate_refuses_complete_and_uncheckpointed(tmp_path):
    complete = tmp_path / "done"
    complete.mkdir()
    (complete / "report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(resume.ResumeError, match="immutable"):
        resume.require_resumable(str(complete), verb="run")

    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(resume.ResumeError, match="resume-state.json"):
        resume.require_resumable(str(bare), verb="run")

    wrong_verb = tmp_path / "wrong"
    wrong_verb.mkdir()
    resume.write_state(str(wrong_verb), run_id="r", verb="run",
                       completed_records=1, reason="signal")
    with pytest.raises(resume.ResumeError, match="verb"):
        resume.require_resumable(str(wrong_verb), verb="extract")
    # And the matching verb passes, returning the state.
    assert resume.require_resumable(str(wrong_verb), verb="run")[
        "completedRecords"] == 1


def test_load_completed_truncates_torn_tail(tmp_path):
    path = tmp_path / "generations.jsonl"
    good = [{"condition": "c", "promptIndex": i, "promptID": f"p{i}",
             "sampleIndex": 0, "output": "t"} for i in range(3)]
    blob = "".join(json.dumps(r) + "\n" for r in good)
    path.write_text(blob + '{"condition": "c", "promptIndex": 3, "promp',
                    encoding="utf-8")
    records, keys = resume.load_completed(str(path))
    assert [r["promptIndex"] for r in records] == [0, 1, 2]
    assert len(keys) == 3
    # The torn tail was truncated so the append-mode writer starts clean.
    assert path.read_text(encoding="utf-8") == blob


# --- the writer loop: K of N pre-populated, N−K appended, byte-equal union -------

def _drive_loop(run_dir, total, *, checkpoint=None, stop_after=None):
    """A miniature deterministic runner over the real GenerationWriter."""
    resuming = resume.is_resumable(run_dir)
    writer = resume.GenerationWriter(run_dir, verb="run", checkpoint=checkpoint,
                                     resume=resuming)
    appended = 0
    try:
        for index in range(total):
            if writer.skip("cond", index, f"prompt-{index}", 0,
                           resume.KIND_SAMPLED):
                continue
            writer.emit({"condition": "cond", "promptIndex": index,
                         "promptID": f"prompt-{index}", "sampleIndex": 0,
                         "output": f"text {index}", "wordCount": 2,
                         "distinct2": 1.0})
            appended += 1
            if stop_after is not None and appended >= stop_after:
                writer.interrupt(reason="cancel")
                return appended
    finally:
        writer.close()
    resume.clear_state(run_dir)
    return appended


def test_prepopulated_run_appends_exactly_the_missing_records(tmp_path):
    total, prepopulated = 9, 4
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert _drive_loop(str(fresh), total) == total
    fresh_bytes = (fresh / "generations.jsonl").read_bytes()

    partial = tmp_path / "partial"
    partial.mkdir()
    assert _drive_loop(str(partial), total, stop_after=prepopulated) == prepopulated
    assert resume.is_resumable(str(partial))
    assert resume.read_state(str(partial))["completedRecords"] == prepopulated

    appended = _drive_loop(str(partial), total)
    assert appended == total - prepopulated
    assert (partial / "generations.jsonl").read_bytes() == fresh_bytes
    assert not resume.is_resumable(str(partial))


def test_writer_emit_is_idempotent_on_key(tmp_path):
    writer = resume.GenerationWriter(str(tmp_path), verb="run")
    record = {"condition": "c", "promptIndex": 0, "promptID": "p",
              "sampleIndex": 0, "output": "t"}
    writer.emit(record)
    writer.emit(dict(record))  # same key: silently dropped
    writer.close()
    lines = (tmp_path / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_writer_checkpoint_flag_parks_and_raises(tmp_path):
    flag = resume.CheckpointFlag()
    writer = resume.GenerationWriter(str(tmp_path), verb="run", checkpoint=flag)
    writer.emit({"condition": "c", "promptIndex": 0, "promptID": "p0",
                 "sampleIndex": 0, "output": "a"})
    flag.request()
    with pytest.raises(resume.CheckpointRequested) as excinfo:
        writer.emit({"condition": "c", "promptIndex": 1, "promptID": "p1",
                     "sampleIndex": 0, "output": "b"})
    writer.close()
    # The record that was mid-flight when the signal arrived is preserved.
    assert excinfo.value.completed_records == 2
    assert excinfo.value.reason == "signal"
    state = resume.read_state(str(tmp_path))
    assert state["completedRecords"] == 2
    assert state["reason"] == "signal"
    lines = (tmp_path / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


# --- the signal path, for real (subprocess + SIGUSR1/SIGTERM) --------------------

def _spawn_harness(run_dir, total, delay):
    env = dict(os.environ)
    env["PYTHONPATH"] = SERVER_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [sys.executable, HARNESS, str(run_dir), str(total), str(delay)],
        cwd=SERVER_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _wait_for_records(path, minimum, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            with open(path, "rb") as handle:
                complete = sum(1 for line in handle if line.endswith(b"\n"))
            if complete >= minimum:
                return complete
        time.sleep(0.05)
    raise AssertionError(f"harness never wrote {minimum} records to {path}")


@pytest.mark.parametrize("signal_number", [signal.SIGUSR1, signal.SIGTERM])
def test_signal_checkpoints_with_exit_85_then_resume_completes(tmp_path, signal_number):
    total = 500
    interrupted = tmp_path / "interrupted"
    control = tmp_path / "control"

    proc = _spawn_harness(interrupted, total, delay=0.05)
    try:
        _wait_for_records(str(interrupted / "generations.jsonl"), minimum=3)
        proc.send_signal(signal_number)
        stdout, stderr = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == resume.CHECKPOINT_EXIT_CODE, (
        f"expected checkpoint exit 85, got {proc.returncode}\n{stdout}\n{stderr}")

    # Valid contract state: flushed JSONL of complete lines + resume-state.json.
    state = resume.read_state(str(interrupted))
    assert state is not None and state["reason"] == "signal"
    assert state["verb"] == "run"
    with open(interrupted / "generations.jsonl", "rb") as handle:
        lines = handle.read().splitlines(keepends=True)
    assert 0 < len(lines) < total
    assert all(line.endswith(b"\n") for line in lines)
    assert state["completedRecords"] == len(lines)
    assert not (interrupted / "report.json").exists()

    # Resume (the same command re-executed, as a requeue would) completes...
    finish = _spawn_harness(interrupted, total, delay=0.0)
    stdout, stderr = finish.communicate(timeout=120)
    assert finish.returncode == 0, f"resume failed\n{stdout}\n{stderr}"
    assert not resume.is_resumable(str(interrupted))
    assert (interrupted / "report.json").exists()

    # ...to the byte-identical union of an uninterrupted run.
    uninterrupted = _spawn_harness(control, total, delay=0.0)
    stdout, stderr = uninterrupted.communicate(timeout=120)
    assert uninterrupted.returncode == 0, f"control failed\n{stdout}\n{stderr}"
    assert ((interrupted / "generations.jsonl").read_bytes()
            == (control / "generations.jsonl").read_bytes())


# --- the real study loop (tasks.run) with faked extraction/generation ------------

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _study_fixture(root, name):
    """A draft study: one concept, one steered condition, three prompts (one
    carrying answer options so the choice instrument emits too)."""
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "calm morning"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    es.add_condition(name, {"name": "fear-a1", "bandWidth": 1,
                            "alphaInNormUnits": False,
                            "slots": [{"concept": "fear", "layer": 2, "alpha": 1.0}]},
                     root)
    raw = es.load_raw(name, root)
    raw["seeds"] = [0]
    raw["temperature"] = 0.0
    raw["maxTokens"] = 16
    raw["outcomeInstruments"] = ["answerTokenLogprob", "sampledText"]
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    _write(prompts_path,
           '{"id": "p0", "prompt": "Decide the case."}\n'
           '{"id": "p1", "prompt": "Choose.", "options": ["A", "B"], "target": "B"}\n'
           '{"id": "p2", "prompt": "Explain the ruling."}\n')
    return prompts_path


def _fake_bundle():
    return tasks.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source="test", stimulus_hash="h")


@contextmanager
def _fake_model(model_id, revision=None):
    yield SimpleNamespace(model_id=model_id, revision=revision or "abc")


class _FakeOptionScore:
    def __init__(self, option, logprob):
        self.option = option
        self.token_ids = [7]
        self.logprob = logprob


class _FakeChoice:
    def __init__(self, options):
        self.options = [_FakeOptionScore(o, -0.5 - i)
                        for i, o in enumerate(options)]

    def as_record_fields(self):
        return {"instrument": "answerTokenLogprob",
                "options": [s.option for s in self.options],
                "selected": self.options[0].option,
                "choiceProbability": {s.option: 0.5 for s in self.options},
                "logOdds": {s.option: 0.0 for s in self.options}}


def _patch_study_fakes(monkeypatch, generate_fn):
    from steerlab_server.experiment import logprob as logprob_mod
    monkeypatch.setattr(tasks, "_extract_all",
                        lambda model, manifest, root: {"fear": _fake_bundle()})
    monkeypatch.setattr(tasks, "generate", generate_fn)
    monkeypatch.setattr(logprob_mod, "score_options",
                        lambda model, prompt, options, **kw: _FakeChoice(options))


def _deterministic_generate(counter=None, arm_flag_at=None, flag=None):
    def generate(model, prompt, *, model_id=None, max_tokens=0, temperature=0.0,
                 injections=None, prompt_mode=None, system_prompt=None,
                 qwen_thinking_enabled=False):
        if counter is not None:
            counter[0] += 1
            if arm_flag_at is not None and counter[0] == arm_flag_at:
                flag.request()
        steered = "steered" if injections else "plain"
        return f"{steered} answer to {prompt}"
    return generate


def test_tasks_run_checkpoint_then_resume_is_byte_identical(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "ws2")

    # Control: an uninterrupted run.
    _patch_study_fakes(monkeypatch, _deterministic_generate())
    control_dir = tasks.run("ws2", prompts, root, model_provider=_fake_model,
                            log=lambda *_: None)
    control_bytes = open(os.path.join(control_dir, "generations.jsonl"), "rb").read()
    assert os.path.exists(os.path.join(control_dir, "report.json"))
    assert not os.path.exists(os.path.join(control_dir, "resume-state.json"))

    # Interrupted: the checkpoint flag arms itself during the 3rd generation.
    flag = resume.CheckpointFlag()
    counter = [0]
    _patch_study_fakes(monkeypatch,
                       _deterministic_generate(counter, arm_flag_at=3, flag=flag))
    seen = {}
    with pytest.raises(resume.CheckpointRequested) as excinfo:
        tasks.run("ws2", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, checkpoint=flag,
                  on_run_directory=lambda d: seen.setdefault("dir", d))
    run_dir = seen["dir"]
    assert excinfo.value.run_directory == run_dir
    assert resume.is_resumable(run_dir)
    assert not os.path.exists(os.path.join(run_dir, "report.json"))
    interrupted_lines = open(os.path.join(run_dir, "generations.jsonl"), "rb").read()
    assert 0 < len(interrupted_lines.splitlines()) < len(control_bytes.splitlines())

    # Resume into the SAME directory: only the missing records generate.
    resumed_counter = [0]
    _patch_study_fakes(monkeypatch, _deterministic_generate(resumed_counter))
    resumed_dir = tasks.run("ws2", prompts, root, model_provider=_fake_model,
                            log=lambda *_: None, run_directory=run_dir)
    assert resumed_dir == run_dir
    assert resumed_counter[0] + counter[0] == 6  # 2 conditions × 3 prompts, no re-runs
    union = open(os.path.join(run_dir, "generations.jsonl"), "rb").read()
    assert union == control_bytes
    assert os.path.exists(os.path.join(run_dir, "report.json"))
    assert not os.path.exists(os.path.join(run_dir, "resume-state.json"))
    # The completed union also rebuilt the derived artifacts.
    assert os.path.exists(os.path.join(run_dir, "metrics.csv"))


def test_tasks_run_cancel_parks_resumable_and_resume_completes(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "ws2c")

    _patch_study_fakes(monkeypatch, _deterministic_generate())
    control_dir = tasks.run("ws2c", prompts, root, model_provider=_fake_model,
                            log=lambda *_: None)
    control_bytes = open(os.path.join(control_dir, "generations.jsonl"), "rb").read()

    counter = [0]
    _patch_study_fakes(monkeypatch, _deterministic_generate(counter))
    seen = {}
    cancelled_dir = tasks.run(
        "ws2c", prompts, root, model_provider=_fake_model, log=lambda *_: None,
        should_cancel=lambda: counter[0] >= 2,
        on_run_directory=lambda d: seen.setdefault("dir", d))
    assert cancelled_dir == seen["dir"]
    state = resume.read_state(cancelled_dir)
    assert state is not None and state["reason"] == "cancel"
    assert not os.path.exists(os.path.join(cancelled_dir, "report.json"))

    _patch_study_fakes(monkeypatch, _deterministic_generate())
    tasks.run("ws2c", prompts, root, model_provider=_fake_model,
              log=lambda *_: None, run_directory=cancelled_dir)
    union = open(os.path.join(cancelled_dir, "generations.jsonl"), "rb").read()
    assert union == control_bytes
    assert os.path.exists(os.path.join(cancelled_dir, "report.json"))
    assert not resume.is_resumable(cancelled_dir)


def test_tasks_run_refuses_to_resume_a_complete_directory(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "ws2r")
    _patch_study_fakes(monkeypatch, _deterministic_generate())
    complete_dir = tasks.run("ws2r", prompts, root, model_provider=_fake_model,
                             log=lambda *_: None)
    with pytest.raises(resume.ResumeError, match="immutable"):
        tasks.run("ws2r", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, run_directory=complete_dir)


def test_tasks_run_refuses_resume_across_experiment_hash_drift(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study_fixture(root, "ws2h")
    flag = resume.CheckpointFlag()
    counter = [0]
    _patch_study_fakes(monkeypatch,
                       _deterministic_generate(counter, arm_flag_at=2, flag=flag))
    seen = {}
    with pytest.raises(resume.CheckpointRequested):
        tasks.run("ws2h", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, checkpoint=flag,
                  on_run_directory=lambda d: seen.setdefault("dir", d))
    # The manifest drifts (draft edit) after the checkpoint.
    raw = es.load_raw("ws2h", root)
    raw["maxTokens"] = 32
    es.save_raw(raw, root)
    _patch_study_fakes(monkeypatch, _deterministic_generate())
    with pytest.raises(resume.ResumeError, match="refusing to mix"):
        tasks.run("ws2h", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, run_directory=seen["dir"])


# --- bundle execute: pointer plumbing across a simulated requeue ------------------

def _bundle_fixture(tmp_path):
    source = str(tmp_path / "source")
    _study_fixture(source, "bexec")
    meta = bundles.package_experiment("bexec", root=source)
    target = str(tmp_path / "target")
    record_path = str(tmp_path / "records" / "job1.json")
    return meta["bundlePath"], target, record_path


def test_bundle_execute_resumes_via_pointer_then_idempotent_when_complete(
        tmp_path, monkeypatch):
    bundle_path, target, record_path = _bundle_fixture(tmp_path)
    pointer = resume.pointer_path_for_record(record_path)
    assert not pointer.endswith(".json")  # must not look like a child record

    calls = []

    def checkpointing_run(name, prompts_path=None, root=None, dtype="auto",
                          device=None, *, checkpoint=None, run_directory=None,
                          on_run_directory=None, **kwargs):
        calls.append({"run_directory": run_directory})
        run_dir = os.path.join(root, "runs", "20260712T000000000-exp-bexec-run")
        os.makedirs(run_dir, exist_ok=True)
        if on_run_directory is not None:
            on_run_directory(run_dir)
        with open(os.path.join(run_dir, "generations.jsonl"), "w",
                  encoding="utf-8") as fh:
            fh.write('{"condition": "baseline", "promptIndex": 0, "promptID": "p0", '
                     '"sampleIndex": 0, "output": "a"}\n'
                     '{"condition": "baseline", "promptIndex": 1, "promptID": "p1", '
                     '"sampleIndex": 0, "output": "b"}\n')
        resume.write_state(run_dir, run_id=os.path.basename(run_dir), verb="run",
                           completed_records=2, reason="signal")
        raise resume.CheckpointRequested(run_dir, "run", 2)

    monkeypatch.setattr(tasks, "run", checkpointing_run)
    with pytest.raises(resume.CheckpointRequested):
        bundles.execute_run_bundle(bundle_path, verb="run", target_root=target,
                                   package_evidence_on_complete=False,
                                   record_path=record_path)
    record = json.loads(open(record_path, encoding="utf-8").read())
    assert record["status"] == "checkpointed"
    assert record["recordCount"] == 2
    assert isinstance(record["elapsedSeconds"], float)
    assert record["result"]["resumeState"] == {"completedRecords": 2,
                                               "reason": "signal"}
    pointed = resume.read_pointer(pointer)
    run_dir = pointed["runDirectory"]
    assert os.path.isdir(run_dir)

    # Re-execution (the requeued sbatch script, verbatim) CONTINUES that run.
    def resuming_run(name, prompts_path=None, root=None, dtype="auto",
                     device=None, *, checkpoint=None, run_directory=None,
                     on_run_directory=None, **kwargs):
        calls.append({"run_directory": run_directory})
        assert run_directory == run_dir
        with open(os.path.join(run_directory, "generations.jsonl"), "a",
                  encoding="utf-8") as fh:
            fh.write('{"condition": "baseline", "promptIndex": 2, "promptID": "p2", '
                     '"sampleIndex": 0, "output": "c"}\n')
        with open(os.path.join(run_directory, "report.json"), "w",
                  encoding="utf-8") as fh:
            fh.write("{}")
        resume.clear_state(run_directory)
        return run_directory

    monkeypatch.setattr(tasks, "run", resuming_run)
    result = bundles.execute_run_bundle(bundle_path, verb="run", target_root=target,
                                        package_evidence_on_complete=False,
                                        record_path=record_path)
    assert result["resumedFrom"] == run_dir
    assert result["runDirectory"] == run_dir
    record = json.loads(open(record_path, encoding="utf-8").read())
    assert record["status"] == "succeeded"
    assert record["recordCount"] == 3

    # A third execution of a COMPLETE run is idempotent: no new run directory,
    # no task invocation.
    def must_not_run(*args, **kwargs):
        raise AssertionError("tasks.run must not execute for a complete run")

    monkeypatch.setattr(tasks, "run", must_not_run)
    result = bundles.execute_run_bundle(bundle_path, verb="run", target_root=target,
                                        package_evidence_on_complete=False,
                                        record_path=record_path)
    assert result["alreadyComplete"] is True
    assert result["runDirectory"] == run_dir
    assert len(calls) == 2
    assert calls[0]["run_directory"] is None
    assert calls[1]["run_directory"] == run_dir


def test_bundle_execute_without_record_path_never_writes_pointer(tmp_path, monkeypatch):
    bundle_path, target, _record = _bundle_fixture(tmp_path)

    def plain_run(name, prompts_path=None, root=None, dtype="auto", device=None,
                  *, checkpoint=None, run_directory=None, on_run_directory=None,
                  **kwargs):
        assert run_directory is None
        run_dir = os.path.join(root, "runs", "r1")
        os.makedirs(run_dir, exist_ok=True)
        # The callback IS plumbed without a record (retention 2026-07-24 —
        # the failure path needs the directory to package partial
        # evidence), so the invariant under test is what it DOES: calling
        # it must not create a resume pointer when there is no record to
        # point at.
        assert on_run_directory is not None
        on_run_directory(run_dir)
        return run_dir

    monkeypatch.setattr(tasks, "run", plain_run)
    result = bundles.execute_run_bundle(bundle_path, verb="run", target_root=target,
                                        package_evidence_on_complete=False)
    assert result["runDirectory"].endswith("r1")
    assert "pointerError" not in result
    pointers = glob.glob(os.path.join(str(tmp_path), "**", "*.resume"),
                         recursive=True)
    assert pointers == []
