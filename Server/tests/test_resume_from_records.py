"""A run stopped in the middle of a response resumes from its records.

When a managed run is cancelled, its child has a short grace period to finish
the response in progress and save its place (``resume-state.json``). A long
response outlasts the grace period: the child is killed, and no place is
saved. The run directory still holds every completed response, one flushed
line each. This file pins that such a run is continued from those lines, and
the one rule that makes it safe:

    A response is complete only when its record line is.

- the directory can be continued from its records alone, and nothing the
  state file carried is missing;
- a run killed in mid-response keeps every completed record, generates only
  the cut-off response again, and ends byte-identical to an uninterrupted run
  under the same seeds (ordinary runs, the capability battery, and panels);
- a last line cut off in mid-write is dropped;
- whatever the cut-off response left in a side stream is removed, never
  counted;
- a policy-agent run's decisions and readings reconcile, with no duplicates;
- a panel cut off in mid-turn resumes at that turn;
- a directory with no records still refuses, with the reason;
- the automatic path (a scheduler requeue) still starts a fresh directory.

What the controller does with this when a person resumes a cancelled job is
in test_resume_from_records_after_cancel.py.

"Killed" is simulated two ways. A real ``SIGKILL`` to a subprocess running
the real writer, and, inside one process, an exception that is NOT an
``Exception`` raised from the fake model in mid-response: like a kill, it
runs no ``except Exception`` handler, so nothing gets a chance to save a
place or write a failure record.
"""

import json
import os
import signal
from contextlib import ExitStack, closing
from types import SimpleNamespace

import pytest
import torch

import steerlab_server.experiment.execution_reporting as _owner_execution_reporting
import steerlab_server.experiment.panel_workflow as panel_workflow
import steerlab_server.experiment.run_readouts as _owner_run_readouts
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import multi_agent, resume, run_status, tasks
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.jlens import trace

from test_resume_checkpoint import _spawn_harness, _wait_for_records
from test_sharding import (_fake_model, _patch_study_fakes, _read,
                           _seed_sensitive_generate, _study_fixture)

STUDY_HASH = "c" * 64


class Killed(BaseException):
    """The process was killed here. Not an ``Exception`` on purpose."""


# --- builders ----------------------------------------------------------------------

def _stamp(run_dir, study_hash=STUDY_HASH, task="run"):
    """What every run directory carries from its first moment: which version
    of the study wrote it, and which verb."""
    with open(os.path.join(run_dir, "experiment-hash.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(study_hash + "\n")
    with open(os.path.join(run_dir, "task.txt"), "w", encoding="utf-8") as handle:
        handle.write(task + "\n")


def _record(index, **extra):
    return {"condition": "baseline", "promptIndex": index,
            "promptID": f"p{index}", "sampleIndex": 0,
            "experimentHash": STUDY_HASH, "output": f"answer {index}", **extra}


def _stopped_mid_response(tmp_path, *, completed=3, torn=True, name="run"):
    """A run directory as a kill in mid-response leaves it: its stamps, the
    completed responses, perhaps the first bytes of the next line, and no
    ``resume-state.json``."""
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    _stamp(str(run_dir))
    text = "".join(json.dumps(_record(i)) + "\n" for i in range(completed))
    if torn:
        text += json.dumps(_record(completed))[:25]
    (run_dir / "generations.jsonl").write_text(text, encoding="utf-8")
    return run_dir


def _lines(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _killing_generate(counter, kill_at):
    """The seed-sensitive fake model, killed during its ``kill_at``-th
    response: the work (and the random draw) happens, the response never
    comes back."""
    inner = _seed_sensitive_generate(counter)

    def generate(model, prompt, **kwargs):
        text = inner(model, prompt, **kwargs)
        if counter[0] == kill_at:
            raise Killed()
        return text
    return generate


def _killed_run(root, name, prompts, monkeypatch, kill_at, **run_kwargs):
    """Run the study until the fake model is killed in mid-response. Returns
    ``(run directory, responses started)``."""
    counter = [0]
    _patch_study_fakes(monkeypatch, _killing_generate(counter, kill_at))
    seen = {}
    with pytest.raises(Killed):
        tasks.run(name, prompts, root, model_provider=_fake_model,
                  log=lambda *_: None,
                  on_run_directory=lambda created: seen.setdefault("dir", created),
                  **run_kwargs)
    return seen["dir"], counter[0]


def _reference_run(root, name, prompts, monkeypatch):
    total = [0]
    _patch_study_fakes(monkeypatch, _seed_sensitive_generate(total))
    directory = tasks.run(name, prompts, root, model_provider=_fake_model,
                          log=lambda *_: None)
    return directory, total[0]


# --- 1. the directory alone is enough ----------------------------------------------

def test_nothing_the_state_file_carried_is_missing_from_the_directory(
        tmp_path, monkeypatch):
    """Park a run cleanly, read what it saved, delete it, and rebuild the
    same answer from the directory: the run id is the folder's name, the verb
    is in task.txt, and the count is the number of record lines. The
    timestamp and the reason are read back by nothing."""
    root = str(tmp_path)
    prompts = _study_fixture(root, "rebuild")
    flag = resume.CheckpointFlag()
    counter = [0]
    _patch_study_fakes(monkeypatch, _seed_sensitive_generate(
        counter, arm_flag_at=4, flag=flag))
    with pytest.raises(resume.CheckpointRequested) as parked:
        tasks.run("rebuild", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, checkpoint=flag)
    run_dir = parked.value.run_directory
    saved = resume.read_state(run_dir)
    assert set(saved) == {"runId", "verb", "completedRecords", "updatedAt",
                          "reason"}

    resume.clear_state(run_dir)
    rebuilt = resume.require_resumable(run_dir, verb="run")

    assert rebuilt["runId"] == saved["runId"] == os.path.basename(run_dir)
    assert rebuilt["verb"] == saved["verb"] == "run"
    assert rebuilt["completedRecords"] == saved["completedRecords"]
    assert rebuilt["reason"] == "records"
    # Reading it changed nothing: the place is still not saved.
    assert resume.read_state(run_dir) is None


def test_a_stopped_run_qualifies_from_its_complete_lines_only(tmp_path):
    run_dir = _stopped_mid_response(tmp_path, completed=3, torn=True)
    before = _read(str(run_dir / "generations.jsonl"))

    basis = resume.records_basis(str(run_dir))

    assert basis["qualifies"] is True
    assert basis["completedRecords"] == 3       # the cut-off line is not one
    assert basis["tornTail"] is True
    assert basis["unit"] == "response"
    assert basis["experimentHash"] == STUDY_HASH
    # Reads only.
    assert _read(str(run_dir / "generations.jsonl")) == before
    assert not resume.is_resumable(str(run_dir))


@pytest.mark.parametrize("damage,code,words", [
    ("no-records", "noRecords", "holds no completed response"),
    ("only-a-torn-line", "noRecords", "holds no completed response"),
    ("no-stamp", "noStudyStamp", "which version of the study wrote it"),
    ("other-study", "studyHashMismatch", "different version of the study"),
    ("failed-status", "failedRun", "stopped on an error"),
    ("failure-note", "failedRun", "stopped on an error"),
    ("sweep-folder", "notAStudyRun", "written by a sweep"),
    ("complete", "complete", "already finished"),
    ("missing", "missing", "no longer on disk"),
])
def test_a_directory_that_cannot_be_continued_says_why(tmp_path, damage, code,
                                                      words):
    run_dir = _stopped_mid_response(tmp_path, completed=2, torn=False)
    generations = run_dir / "generations.jsonl"
    if damage == "no-records":
        generations.unlink()
    elif damage == "only-a-torn-line":
        generations.write_text(json.dumps(_record(0))[:30], encoding="utf-8")
    elif damage == "no-stamp":
        (run_dir / "experiment-hash.txt").unlink()
    elif damage == "other-study":
        with open(generations, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_record(2, experimentHash="d" * 64)) + "\n")
    elif damage == "failed-status":
        status = run_status.RunStatus(str(run_dir), stage="run")
        status.fail(RuntimeError("out of memory"))
        (run_dir / run_status.FAILURE_NOTE_FILENAME).unlink()
    elif damage == "failure-note":
        (run_dir / run_status.FAILURE_NOTE_FILENAME).write_text("# run FAILED\n")
    elif damage == "sweep-folder":
        (run_dir / "task.txt").write_text("sweep\n", encoding="utf-8")
    elif damage == "complete":
        (run_dir / resume.COMPLETION_FILENAME).write_text("{}", encoding="utf-8")
    elif damage == "missing":
        import shutil
        shutil.rmtree(run_dir)

    basis = resume.records_basis(str(run_dir))

    assert basis["qualifies"] is False
    assert basis["code"] == code
    assert words in basis["reason"]
    assert basis["completedRecords"] == 0
    if damage not in ("complete", "missing"):
        with pytest.raises(resume.ResumeError, match="resume-state.json"):
            resume.require_resumable(str(run_dir), verb="run")
        with pytest.raises(resume.ResumeError, match="response records"):
            resume.adopt_records(str(run_dir))
        assert resume.read_state(str(run_dir)) is None


def test_a_sweep_still_needs_the_place_it_saved_itself(tmp_path):
    run_dir = _stopped_mid_response(tmp_path, completed=2, torn=False)
    assert resume.records_basis(str(run_dir), verb="sweep")["code"] == "notAStudyRun"
    with pytest.raises(resume.ResumeError, match="only checkpointed runs"):
        resume.require_resumable(str(run_dir), verb="sweep")


# --- 2. saving the place afterwards ------------------------------------------------

def test_adopting_the_records_leaves_what_a_clean_park_would_have(tmp_path):
    run_dir = _stopped_mid_response(tmp_path, completed=3, torn=True)
    complete = "".join(json.dumps(_record(i)) + "\n" for i in range(3))
    (run_dir / "battery.jsonl").write_text('{"condition": "baseline", "pro',
                                           encoding="utf-8")

    adopted = resume.adopt_records(str(run_dir))

    assert adopted["completedRecords"] == 3
    state = resume.read_state(str(run_dir))
    assert state["reason"] == "records"
    assert state["verb"] == "run"
    assert state["completedRecords"] == 3
    assert state["runId"] == run_dir.name
    # The cut-off lines are gone; the complete ones are untouched.
    assert (run_dir / "generations.jsonl").read_text(encoding="utf-8") == complete
    assert (run_dir / "battery.jsonl").read_text(encoding="utf-8") == ""
    assert resume.is_resumable(str(run_dir))
    assert resume.require_resumable(str(run_dir), verb="run")["reason"] == "records"


def test_a_place_the_run_saved_itself_is_never_rewritten(tmp_path):
    run_dir = _stopped_mid_response(tmp_path, completed=3, torn=False)
    resume.write_state(str(run_dir), run_id=run_dir.name, verb="run",
                       completed_records=3, reason="signal")
    before = _read(str(run_dir / resume.RESUME_STATE_FILENAME))

    adopted = resume.adopt_records(str(run_dir))

    assert adopted["parkedByRun"] is True
    assert _read(str(run_dir / resume.RESUME_STATE_FILENAME)) == before


def test_a_place_saved_from_records_is_refreshed_when_the_run_went_on(tmp_path):
    """Resumed from its records, the run added responses and was stopped in
    mid-response again: the earlier saved place under-counts, so it is
    counted again from the lines on disk."""
    run_dir = _stopped_mid_response(tmp_path, completed=2, torn=False)
    assert resume.adopt_records(str(run_dir))["completedRecords"] == 2
    with open(run_dir / "generations.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(2)) + "\n")
        handle.write(json.dumps(_record(3))[:20])

    assert resume.adopt_records(str(run_dir))["completedRecords"] == 3
    assert resume.read_state(str(run_dir))["completedRecords"] == 3
    assert len(_lines(run_dir / "generations.jsonl")) == 3


def test_the_automatic_path_still_starts_fresh_until_a_place_is_saved(tmp_path):
    """A scheduler requeue re-executes the script with nobody having
    established that the earlier process is gone, so the pointer does not
    hand back a directory with no saved place, however good its records."""
    run_dir = _stopped_mid_response(tmp_path, completed=3, torn=True)
    pointer = str(tmp_path / "records" / "job.resume")
    resume.write_pointer(pointer, str(run_dir), verb="run", experiment="demo")
    assert resume.records_basis(str(run_dir))["qualifies"] is True

    assert resume.resolve_pointer(pointer, verb="run") == ("fresh", None)

    resume.adopt_records(str(run_dir))
    assert resume.resolve_pointer(pointer, verb="run") == ("resume", str(run_dir))


def test_the_re_executed_script_follows_the_pointer_the_same_way(
        tmp_path, monkeypatch):
    """The same decision, through the entry point the batch script runs
    (``bundle execute`` with the job's record path). Killed with no saved
    place, a re-execution starts a new run. Once the place is saved from the
    records, a re-execution of the very same command continues that run."""
    from steerlab_server.experiment import bundles
    from test_resume_checkpoint import _bundle_fixture
    bundle_path, target, record_path = _bundle_fixture(tmp_path)
    calls = []

    def killed_in_mid_response(name, prompts_path=None, root=None, dtype="auto",
                               device=None, *, run_directory=None,
                               on_run_directory=None, **kwargs):
        calls.append(run_directory)
        created = os.path.join(root, "runs", f"exp-bexec-run-{len(calls)}")
        os.makedirs(created)
        _stamp(created)
        on_run_directory(created)
        with open(os.path.join(created, "generations.jsonl"), "w",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(_record(0)) + "\n"
                         + json.dumps(_record(1)) + "\n"
                         + json.dumps(_record(2))[:20])
        raise Killed()

    monkeypatch.setattr(tasks, "run", killed_in_mid_response)
    for _attempt in range(2):       # the original execution, then a requeue
        with pytest.raises(Killed):
            bundles.execute_run_bundle(
                bundle_path, verb="run", target_root=target,
                package_evidence_on_complete=False, record_path=record_path)
    assert calls == [None, None]    # the requeue was NOT handed the directory
    pointer = resume.pointer_path_for_record(record_path)
    stopped = resume.read_pointer(pointer)["runDirectory"]
    assert stopped.endswith("exp-bexec-run-2")

    assert resume.adopt_records(stopped)["completedRecords"] == 2

    def continuing(name, prompts_path=None, root=None, dtype="auto",
                   device=None, *, run_directory=None, **kwargs):
        calls.append(run_directory)
        with open(os.path.join(run_directory, "report.json"), "w",
                  encoding="utf-8") as handle:
            handle.write("{}")
        resume.clear_state(run_directory)
        return run_directory

    monkeypatch.setattr(tasks, "run", continuing)
    result = bundles.execute_run_bundle(
        bundle_path, verb="run", target_root=target,
        package_evidence_on_complete=False, record_path=record_path)

    assert calls[-1] == stopped
    assert result["resumedFrom"] == stopped


def test_the_state_reason_set_grew_by_exactly_one(tmp_path):
    assert resume.STATE_REASONS == ("signal", "cancel", "records")
    resume.write_state(str(tmp_path), run_id="r", verb="run",
                       completed_records=1, reason="records")
    assert resume.read_state(str(tmp_path))["reason"] == "records"
    with pytest.raises(ValueError, match="signal.*cancel.*records"):
        resume.write_state(str(tmp_path), run_id="r", verb="run",
                           completed_records=1, reason="guess")


# --- 3. a real kill ----------------------------------------------------------------

def test_a_process_killed_in_mid_response_resumes_from_its_records(tmp_path):
    """SIGKILL, for real: the subprocess is inside a response (the harness
    sleeps there) and gets no chance to flush, save a place, or exit. What it
    had flushed is complete; a re-run told to continue from the records
    finishes byte-identical to a run that was never interrupted."""
    total = 400
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    _stamp(str(interrupted))
    control = tmp_path / "control"

    proc = _spawn_harness(interrupted, total, delay=0.05)
    try:
        _wait_for_records(str(interrupted / "generations.jsonl"), minimum=3)
        proc.send_signal(signal.SIGKILL)
        proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == -signal.SIGKILL

    assert resume.read_state(str(interrupted)) is None        # no place saved
    assert not (interrupted / "report.json").exists()
    kept = _read(str(interrupted / "generations.jsonl"))
    basis = resume.records_basis(str(interrupted))
    assert basis["qualifies"] is True
    assert 3 <= basis["completedRecords"] < total
    # What a kill in mid-WRITE adds on top: the first bytes of the next line.
    with open(interrupted / "generations.jsonl", "ab") as handle:
        handle.write(b'{"condition": "cond", "promptIndex": 9999, "prom')

    # The harness alone would start over (it resumes only a saved place)...
    assert not resume.is_resumable(str(interrupted))
    # ...so it is told to continue from the records.
    finish = _spawn_harness_from_records(interrupted, total)
    stdout, stderr = finish.communicate(timeout=120)
    assert finish.returncode == 0, f"resume failed\n{stdout}\n{stderr}"

    uninterrupted = _spawn_harness(control, total, delay=0.0)
    stdout, stderr = uninterrupted.communicate(timeout=120)
    assert uninterrupted.returncode == 0, f"control failed\n{stdout}\n{stderr}"
    final = _read(str(interrupted / "generations.jsonl"))
    assert final.startswith(kept)              # completed records untouched
    assert final == _read(str(control / "generations.jsonl"))
    assert (interrupted / "report.json").exists()


def _spawn_harness_from_records(run_dir, total):
    import subprocess
    import sys
    from test_resume_checkpoint import HARNESS, SERVER_DIR
    env = dict(os.environ)
    env["PYTHONPATH"] = SERVER_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [sys.executable, HARNESS, str(run_dir), str(total), "0",
         "--resume-from-records"],
        cwd=SERVER_DIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


# --- 4. the real study loop, killed in mid-response --------------------------------

@pytest.mark.parametrize("kill_at,where", [
    (2, "the first condition"),
    (9, "the second condition"),
    (14, "the capability battery"),
])
def test_a_run_killed_in_mid_response_resumes_to_the_uninterrupted_result(
        tmp_path, monkeypatch, kill_at, where):
    """Sampling at temperature 0.7 with per-record derived seeds, a choice
    instrument, and a pinned capability battery. The fake model's output
    embeds a random draw, so the resumed run matches only if every record is
    regenerated under exactly the seed an uninterrupted run would give it."""
    root = str(tmp_path / "workspace")
    prompts = _study_fixture(root, "midresponse")
    reference_dir, total = _reference_run(root, "midresponse", prompts,
                                          monkeypatch)
    reference = _read(os.path.join(reference_dir, "generations.jsonl"))
    reference_battery = _read(os.path.join(reference_dir, "battery.jsonl"))
    assert kill_at < total

    run_dir, started = _killed_run(root, "midresponse", prompts, monkeypatch,
                                   kill_at)

    # No place was saved, and nothing claims the run finished.
    assert started == kill_at
    assert resume.read_state(run_dir) is None
    assert not resume.is_resumable(run_dir)
    assert not resume.is_complete(run_dir)
    kept = _read(os.path.join(run_dir, "generations.jsonl"))
    assert kept and reference.startswith(kept)
    # Killed in the battery, every response is already on disk and it is a
    # battery item that was cut off; killed earlier, some responses remain.
    in_battery = where == "the capability battery"
    assert (kept == reference) is in_battery
    kept_battery = (_read(os.path.join(run_dir, "battery.jsonl"))
                    if in_battery else b"")
    assert reference_battery.startswith(kept_battery)
    assert kept_battery != reference_battery
    basis = resume.records_basis(run_dir)
    assert basis["qualifies"] is True
    assert basis["completedRecords"] == len(kept.splitlines())

    after = [0]
    _patch_study_fakes(monkeypatch, _seed_sensitive_generate(after))
    lines = []
    finished = tasks.run("midresponse", prompts, root,
                         model_provider=_fake_model, log=lines.append,
                         run_directory=run_dir)

    assert finished == run_dir
    final = _read(os.path.join(run_dir, "generations.jsonl"))
    assert final.startswith(kept)               # every completed record kept
    assert final == reference                   # and the whole equals uninterrupted
    assert _read(os.path.join(run_dir, "battery.jsonl")) == reference_battery
    # Only the cut-off response was generated a second time.
    assert started + after[0] == total + 1
    assert resume.is_complete(run_dir)
    assert resume.read_state(run_dir) is None
    assert any("stopped before it could save its place" in line
               and "generated again" in line for line in lines), where


def test_a_line_cut_off_in_mid_write_is_dropped_and_its_response_regenerated(
        tmp_path, monkeypatch):
    root = str(tmp_path / "workspace")
    prompts = _study_fixture(root, "tornline")
    reference_dir, total = _reference_run(root, "tornline", prompts, monkeypatch)
    reference = _read(os.path.join(reference_dir, "generations.jsonl"))
    run_dir, started = _killed_run(root, "tornline", prompts, monkeypatch, 6)
    path = os.path.join(run_dir, "generations.jsonl")
    kept = _read(path)
    # The kill landed while the next record's line was being written: its
    # first bytes are on disk, its terminator is not.
    next_line = reference[len(kept):].splitlines(keepends=True)[0]
    with open(path, "ab") as handle:
        handle.write(next_line[:len(next_line) // 2])
    assert resume.records_basis(run_dir)["completedRecords"] == len(
        kept.splitlines())

    after = [0]
    _patch_study_fakes(monkeypatch, _seed_sensitive_generate(after))
    lines = []
    tasks.run("tornline", prompts, root, model_provider=_fake_model,
              log=lines.append, run_directory=run_dir)

    assert _read(path) == reference
    assert started + after[0] == total + 1      # the half-written one, once more
    assert any("cut off in mid-write" in line for line in lines)


def test_a_run_killed_before_its_first_record_still_refuses(tmp_path, monkeypatch):
    root = str(tmp_path / "workspace")
    prompts = _study_fixture(root, "norecords")
    run_dir, _started = _killed_run(root, "norecords", prompts, monkeypatch, 1)
    assert os.path.isdir(run_dir)
    assert resume.records_basis(run_dir)["code"] == "noRecords"

    _patch_study_fakes(monkeypatch, _seed_sensitive_generate())
    with pytest.raises(resume.ResumeError) as refused:
        tasks.run("norecords", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, run_directory=run_dir)

    message = str(refused.value)
    assert "resume-state.json" in message
    assert "holds no completed response" in message


def test_a_study_changed_since_the_kill_refuses_to_mix(tmp_path, monkeypatch):
    root = str(tmp_path / "workspace")
    prompts = _study_fixture(root, "drifted")
    run_dir, _started = _killed_run(root, "drifted", prompts, monkeypatch, 5)
    raw = es.load_raw("drifted", root)
    raw["maxTokens"] = 32
    es.save_raw(raw, root)

    _patch_study_fakes(monkeypatch, _seed_sensitive_generate())
    with pytest.raises(resume.ResumeError, match="refusing to mix"):
        tasks.run("drifted", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, run_directory=run_dir)


def test_a_run_that_stopped_on_an_error_is_not_continued_from_its_records(
        tmp_path, monkeypatch):
    """A failure record is evidence of a failure. It was packaged and reported
    as one, so it is not quietly turned into a result."""
    root = str(tmp_path / "workspace")
    prompts = _study_fixture(root, "failed")
    run_dir, _started = _killed_run(root, "failed", prompts, monkeypatch, 5)
    run_status.RunStatus(run_dir, stage="run").fail(RuntimeError("out of memory"))

    _patch_study_fakes(monkeypatch, _seed_sensitive_generate())
    with pytest.raises(resume.ResumeError, match="stopped on an error"):
        tasks.run("failed", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, run_directory=run_dir)


# --- 5. side streams: nothing of the cut-off response survives ---------------------

def test_the_only_per_response_side_stream_is_the_jlens_trace():
    assert resume.RESPONSE_SIDE_STREAMS == (trace.TRACE_FILENAME,)


def _trace_row(index, attempt="first"):
    return {"run": "r", "condition": "baseline", "promptIndex": index,
            "promptID": f"p{index}", "sampleIndex": 0,
            "recordType": "jlensReadout", "observations": [],
            "observationCount": 0, "traceComplete": True, "attempt": attempt}


def test_a_trace_row_without_a_record_line_is_removed_on_resume(tmp_path):
    """The J-lens trace row of a response is written BEFORE the response's
    record line. A kill between the two leaves a row for a response that
    never completed; the trace writer's own idempotence would then keep it
    and drop the row the regenerated response writes."""
    run_dir = _stopped_mid_response(tmp_path, completed=2, torn=False)
    first = trace.TraceWriter(str(run_dir))
    for index in range(3):                       # the third has no record line
        first.write(_trace_row(index))
    first.close()
    kept_bytes = b"".join(
        _read(first.path).splitlines(keepends=True)[:2])

    lines = []
    with closing(resume.GenerationWriter(str(run_dir), resume=True,
                                         log=lines.append)) as writer:
        assert writer.removed_side_rows == {trace.TRACE_FILENAME: 1}
        assert _read(first.path) == kept_bytes   # kept rows: same bytes, same order
        # The response is generated again and writes both, afresh.
        again = trace.TraceWriter(str(run_dir), resume=True)
        assert not again.skip("baseline", 2, "p2", 0)
        assert again.skip("baseline", 1, "p1", 0)
        again.write(_trace_row(2, attempt="second"))
        again.close()
        writer.emit(_record(2))

    rows = _lines(first.path)
    assert [(row["promptID"], row["attempt"]) for row in rows] == [
        ("p0", "first"), ("p1", "first"), ("p2", "second")]
    assert any("removed 1 row(s) from jlens-readout.jsonl" in line
               for line in lines)
    assert not os.path.exists(first.path + ".tmp")


def test_reconciling_a_consistent_trace_rewrites_nothing(tmp_path):
    run_dir = _stopped_mid_response(tmp_path, completed=2, torn=False)
    writer = trace.TraceWriter(str(run_dir))
    for index in range(2):
        writer.write(_trace_row(index))
    writer.close()
    identity = os.stat(writer.path).st_ino
    _records, keys = resume.load_completed(str(run_dir / "generations.jsonl"))

    assert resume.reconcile_side_streams(str(run_dir), keys) == {}
    assert os.stat(writer.path).st_ino == identity     # not even replaced

    # A trace row cut off in mid-write is dropped with nothing else removed.
    whole = _read(writer.path)
    with open(writer.path, "ab") as handle:
        handle.write(b'{"run": "r", "condition": "baseline", "promptInd')
    assert resume.reconcile_side_streams(str(run_dir), keys) == {}
    assert _read(writer.path) == whole


def test_a_stray_trace_row_between_kept_rows_goes_without_disturbing_them(
        tmp_path):
    """Not a shape a single kill leaves (its row is always the last one),
    but the rule is about records, not positions: a row whose response has
    no record line is removed wherever it sits."""
    run_dir = _stopped_mid_response(tmp_path, completed=2, torn=False)
    writer = trace.TraceWriter(str(run_dir))
    for index in (0, 7, 1):
        writer.write(_trace_row(index))
    writer.close()
    rows = _read(writer.path).splitlines(keepends=True)
    _records, keys = resume.load_completed(str(run_dir / "generations.jsonl"))

    assert resume.reconcile_side_streams(str(run_dir), keys) == {
        trace.TRACE_FILENAME: 1}

    assert _read(writer.path) == rows[0] + rows[2]
    assert not os.path.exists(writer.path + ".tmp")


class _FakeTraceSession:
    """The run loop's view of a J-lens trace, over the REAL trace writer."""

    def __init__(self, run_directory, resuming, attempt, kill_after_rows=None):
        self.writer = trace.TraceWriter(run_directory, resume=resuming)
        self.attempt = attempt
        self.kill_after_rows = kill_after_rows
        self.rows = 0

    def recorder_for(self, prompt):
        return SimpleNamespace(complete=True, observations=[],
                               failureReason=None)

    def record_generation(self, recorder, eff, prompt, prompt_index,
                          sample_index, *, model, manifest, generated_ids):
        self.writer.write({
            "run": "r", "condition": eff.name, "promptIndex": prompt_index,
            "promptID": prompt["id"], "sampleIndex": sample_index,
            "recordType": "jlensReadout", "observations": [],
            "observationCount": 0, "traceComplete": True,
            "attempt": self.attempt}, recorder)
        self.rows += 1
        if self.kill_after_rows is not None and self.rows == self.kill_after_rows:
            raise Killed()      # the trace row is on disk; the record never is
        return {"trace": trace.TRACE_FILENAME, "configHash": "h",
                "observations": 0, "complete": True}

    def close(self, *, expected_records=None):
        self.writer.close()
        return self.writer.summary(expected_records=expected_records)


def test_a_run_killed_between_the_trace_row_and_the_record_line(
        tmp_path, monkeypatch):
    root = str(tmp_path / "workspace")
    prompts = _study_fixture(root, "traced")

    def arm(attempt, kill_after_rows=None):
        monkeypatch.setattr(
            _owner_run_readouts, "open_jlens_trace",
            lambda manifest, model, root, *, run_directory, resuming, **kw:
                _FakeTraceSession(run_directory, resuming, attempt,
                                  kill_after_rows))

    arm("first", kill_after_rows=5)
    _patch_study_fakes(monkeypatch, _seed_sensitive_generate())
    seen = {}
    with pytest.raises(Killed):
        tasks.run("traced", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None,
                  on_run_directory=lambda created: seen.setdefault("dir", created))
    run_dir = seen["dir"]
    trace_path = os.path.join(run_dir, trace.TRACE_FILENAME)
    traced = {resume.record_key(row) for row in _lines(trace_path)}
    recorded = {resume.record_key(row)
                for row in _lines(os.path.join(run_dir, "generations.jsonl"))
                if "jlensReadout" in row}
    orphan = traced - recorded
    assert len(orphan) == 1 and len(traced) == 5      # one row, no record line

    arm("second")
    lines = []
    tasks.run("traced", prompts, root, model_provider=_fake_model,
              log=lines.append, run_directory=run_dir)

    rows = _lines(trace_path)
    keys = [resume.record_key(row) for row in rows]
    records = [row for row in _lines(os.path.join(run_dir, "generations.jsonl"))
               if "jlensReadout" in row]
    # One trace row per sampled response, each response's row exactly once.
    assert len(keys) == len(set(keys)) == len(records) == 12
    assert set(keys) == {resume.record_key(row) for row in records}
    # The four responses that completed keep their first-attempt rows; the
    # cut-off response's row is the one its SECOND attempt wrote.
    by_key = {resume.record_key(row): row["attempt"] for row in rows}
    assert by_key[next(iter(orphan))] == "second"
    assert sorted(by_key.values()).count("first") == 4
    assert any("removed 1 row(s) from jlens-readout.jsonl" in line
               for line in lines)


# --- 6. a policy-agent run: decisions and readings reconcile -----------------------

def _policy_study(tmp_path):
    """A sampled study whose one condition is an agent with an intervention
    policy, with a probe measurement declared on the study: every response
    records decisions and readings."""
    from steerlab_server.experiment import condition_execution as execution
    from steerlab_server.experiment.model_variant import ModelVariant
    from test_intervention_policies import attached, setup as policy_setup
    from test_probe_measurements import prepared
    doc, _probe, model = policy_setup(tmp_path)
    config, _doc, _model = prepared(tmp_path)
    variant = ModelVariant(
        name="modified", base_model_id="example/model", base_revision="a" * 40,
        prompt_mode="rawCompletion",
        intervention_policies=attached(doc, tmp_path))
    study = Manifest(name="example", model_id="example/model",
                     raw={"probeMeasurements": config}, seeds=[0, 1],
                     temperature=0, max_tokens=4)
    eff = execution.EffectiveCondition("modified", [], {}, "rawCompletion",
                                       None, False, 0, variant=variant)
    prompts = [{"id": f"item-{i}", "prompt": "example"} for i in range(3)]
    kwargs = dict(name="example", manifest=study, experiment_hash=STUDY_HASH,
                  wants_choice=False, wants_sampled=True, reader_scorers=[],
                  should_cancel=None, log=lambda _: None, root=tmp_path)
    return execution, model, eff, prompts, kwargs


def _observed_generation(stop_at=None, stop_with=None, calls=None):
    """A response the policy and the probe observe for real (they hook a real
    forward pass). At ``stop_at`` the response is cut off AFTER they have
    made their decisions and readings, which exist then only in memory."""
    from steerlab_server.experiment import prompt_render
    calls = calls if calls is not None else [0]

    def generate(model, prompt, *, observers=(), token_ids_out=None, **kwargs):
        calls[0] += 1
        with model.hooked.session([]), ExitStack() as sessions:
            for observer in observers:
                sessions.enter_context(observer.observe_session(
                    model, prompt_render.RenderedPrompt(prompt, [1, 2, 3], 3)))
            model.model(input_ids=torch.tensor([[1, 2, 3]]),
                        attention_mask=torch.ones((1, 3)))
            if stop_at is not None and calls[0] == stop_at:
                raise stop_with
        if token_ids_out is not None:
            token_ids_out.append(4)
        return "example response"
    return generate


def _without_clocks(value):
    """Host timings differ between any two runs; nothing else may."""
    if isinstance(value, dict):
        return {key: _without_clocks(item) for key, item in value.items()
                if key not in ("timing", "hostSeconds")}
    if isinstance(value, list):
        return [_without_clocks(item) for item in value]
    return value


def _evidence_counts(records):
    decisions = sum(len(row["interventionDecisions"]["decisions"])
                    for row in records)
    readings = sum(len(row["probeMeasurements"]["readings"]) for row in records)
    return decisions, readings


@pytest.mark.parametrize("stopped_by", ["a kill", "an error"])
def test_a_policy_agent_runs_decisions_and_readings_reconcile_after_resume(
        tmp_path, monkeypatch, stopped_by):
    execution, model, eff, prompts, kwargs = _policy_study(tmp_path)

    reference_dir = tmp_path / "runs" / "reference"
    reference_dir.mkdir(parents=True)
    monkeypatch.setattr(execution.generate, "generate", _observed_generation())
    with closing(resume.GenerationWriter(str(reference_dir))) as writer:
        execution.execute_condition(model, eff, prompts, writer, **kwargs)
    reference = _lines(reference_dir / "generations.jsonl")
    assert len(reference) == 6                       # 3 items x 2 seeds
    expected = _evidence_counts(reference)
    assert expected[0] > 0 and expected[1] > 0

    run_dir = tmp_path / "runs" / "stopped"
    run_dir.mkdir(parents=True)
    _stamp(str(run_dir))
    stop_with = Killed() if stopped_by == "a kill" else RuntimeError("device lost")
    monkeypatch.setattr(execution.generate, "generate",
                        _observed_generation(stop_at=4, stop_with=stop_with))
    with pytest.raises(type(stop_with)):
        with closing(resume.GenerationWriter(str(run_dir))) as writer:
            execution.execute_condition(model, eff, prompts, writer, **kwargs)
    kept = _read(str(run_dir / "generations.jsonl"))
    assert len(kept.splitlines()) == 3               # the fourth was cut off
    side_files = sorted(p.name for p in run_dir.glob("*-failure-*.json"))
    if stopped_by == "a kill":
        # A kill writes nothing: the cut-off response's decisions and
        # readings were only ever in memory.
        assert side_files == []
    else:
        # An error writes what the observers had, as a failure record.
        assert [name.split("-failure-")[0] for name in side_files] == [
            "policy", "probe"]

    assert resume.require_resumable(str(run_dir), verb="run")[
        "completedRecords"] == 3
    after = [0]
    monkeypatch.setattr(execution.generate, "generate",
                        _observed_generation(calls=after))
    with closing(resume.GenerationWriter(str(run_dir), resume=True)) as writer:
        execution.execute_condition(model, eff, prompts, writer, **kwargs)

    final = _read(str(run_dir / "generations.jsonl"))
    records = _lines(run_dir / "generations.jsonl")
    assert final.startswith(kept)                    # completed records untouched
    assert after[0] == 3                             # the cut-off one and the rest
    keys = [resume.record_key(row) for row in records]
    assert len(keys) == len(set(keys)) == 6          # one record per response
    assert _evidence_counts(records) == expected     # no decision counted twice
    assert _without_clocks(records) == _without_clocks(reference)
    # Whatever the cut-off attempt left says it is partial, belongs to the
    # one response that was cut off, and holds no completed response.
    for path in run_dir.glob("*-failure-*.json"):
        left = json.loads(path.read_text(encoding="utf-8"))
        assert left["status"] == "partial" and left["failures"]
        assert (left["promptID"], left["sampleIndex"]) == ("item-1", 1)
        assert left["outputTokenIDs"] == []
    assert sorted(p.name for p in run_dir.glob("*-failure-*.json")) == side_files


# --- 7. panels: a conversation cut off in mid-turn resumes at that turn ------------

def _panel_workspace(tmp_path, monkeypatch):
    """Two conditions (configured and baseline) x two replicates x three
    turns, sampled at temperature 0.7 so every turn draws from its own
    seeded stream."""
    root = tmp_path / "ws"
    for sub in ("prompts/panels", "experiments", "runs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    panel = {
        "schemaVersion": 1, "name": "panel-a", "baseModelID": "model-a",
        "description": "", "sharedMaterials": "materials",
        "temperature": 0.7, "maxTokens": 16,
        "agents": [{"id": "agent-a", "name": "Agent A",
                    "baseModelID": "model-a", "systemPrompt": "",
                    "variantArtifactPath": None, "variantArtifactHash": None}],
        "turns": [{"id": f"turn-{letter}", "title": f"Turn {letter.upper()}",
                   "speakerAgentID": "agent-a",
                   "promptTemplate": "go {{agent.context}}",
                   "outputLabel": f"out-{letter}", "routing": "all",
                   "routedAgentIDs": [], "includeScenarioMaterials": True,
                   "includeSpeakerContext": True, "maxTokens": None}
                  for letter in ("a", "b", "c")],
    }
    (root / "prompts/panels/panel-a.json").write_text(json.dumps(panel))
    spec = {"name": "panel-a", "modelID": "model-a", "studyKind": "multiAgent",
            "multiAgentScenarioPath": "prompts/panels/panel-a.json",
            "multiAgentIncludeBaseline": True, "samplesPerItem": 2,
            "temperature": 0.7, "seeds": [0]}
    (root / "experiments/panel-a.json").write_text(json.dumps(spec))
    monkeypatch.setattr(_owner_execution_reporting, "advise_cross_substrate",
                        lambda *a, **k: None)
    monkeypatch.setattr(_owner_execution_reporting,
                        "advise_dependency_lock_drift", lambda *a, **k: None)
    return root, Manifest.from_dict(spec)


def _panel_model():
    return SimpleNamespace(model_id="model-a", revision="rev-a", device="cpu")


def _panel_generate(calls, kill_at=None):
    def generate(model, prompt, *, temperature=0.0, **kwargs):
        calls[0] += 1
        draw = int(torch.randint(0, 10 ** 9, (1,)).item())
        if kill_at is not None and calls[0] == kill_at:
            raise Killed()
        return f"said [{draw}] after {len(prompt)} characters"
    return generate


def _panel_files(run_directory):
    """Every transcript's turn records, by transcript."""
    return {os.path.relpath(path, run_directory): _read(path)
            for path in resume.panel_turn_files(run_directory)}


@pytest.mark.parametrize("torn", [False, True])
def test_a_panel_cut_off_in_mid_turn_resumes_at_that_turn(tmp_path, monkeypatch,
                                                         torn):
    root, manifest = _panel_workspace(tmp_path, monkeypatch)
    total = [0]
    monkeypatch.setattr(multi_agent, "generate", _panel_generate(total))
    reference_dir = panel_workflow.run_multi_agent_study(
        "panel-a", manifest, _panel_model(), str(root), log=lambda _: None)
    assert total[0] == 12

    # Killed during the fifth turn: the second turn of the second transcript.
    started = [0]
    monkeypatch.setattr(multi_agent, "generate",
                        _panel_generate(started, kill_at=5))
    seen = {}
    with pytest.raises(Killed):
        panel_workflow.run_multi_agent_study(
            "panel-a", manifest, _panel_model(), str(root), log=lambda _: None,
            on_run_directory=lambda created: seen.setdefault("dir", created))
    run_dir = seen["dir"]

    # A panel's records are its transcripts' turn lines, flushed per turn.
    # The root view is written only when a run parks or ends, so a killed
    # run has neither it nor a saved place.
    assert resume.read_state(run_dir) is None
    assert not os.path.exists(os.path.join(run_dir, "generations.jsonl"))
    cut_off = os.path.join(run_dir, "configured", "replicate-1", "turns.jsonl")
    assert len(_lines(cut_off)) == 1
    if torn:
        with open(cut_off, "ab") as handle:
            handle.write(b'{"turnID": "turn-b", "turnIndex": 2, "outp')
    basis = resume.records_basis(run_dir)
    assert basis["qualifies"] is True
    assert (basis["unit"], basis["completedRecords"]) == ("turn", 4)
    kept = _panel_files(run_dir)

    after = [0]
    monkeypatch.setattr(multi_agent, "generate", _panel_generate(after))
    finished = panel_workflow.run_multi_agent_study(
        "panel-a", manifest, _panel_model(), str(root), log=lambda _: None,
        run_directory=run_dir)

    assert finished == run_dir
    # The cut-off turn and everything after it: 12 turns, 4 already complete.
    assert after[0] == 8
    final = _panel_files(run_dir)
    assert final == _panel_files(reference_dir)       # same seeds, same turns
    for transcript, before in kept.items():
        complete = before if not torn else before[:before.rfind(b"\n") + 1]
        assert final[transcript].startswith(complete)
    assert (_read(os.path.join(run_dir, "generations.jsonl"))
            == _read(os.path.join(reference_dir, "generations.jsonl")))
    assert resume.is_complete(run_dir)
    assert resume.read_state(run_dir) is None


def test_a_panel_killed_during_its_first_turn_still_refuses(tmp_path,
                                                           monkeypatch):
    root, manifest = _panel_workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(multi_agent, "generate", _panel_generate([0], kill_at=1))
    seen = {}
    with pytest.raises(Killed):
        panel_workflow.run_multi_agent_study(
            "panel-a", manifest, _panel_model(), str(root), log=lambda _: None,
            on_run_directory=lambda created: seen.setdefault("dir", created))

    assert resume.records_basis(seen["dir"])["code"] == "noRecords"
    monkeypatch.setattr(multi_agent, "generate", _panel_generate([0]))
    with pytest.raises(resume.ResumeError, match="holds no completed response"):
        panel_workflow.run_multi_agent_study(
            "panel-a", manifest, _panel_model(), str(root), log=lambda _: None,
            run_directory=seen["dir"])


def test_the_turn_count_is_the_panel_runners_own_reading(tmp_path):
    """Two readers of one file must not disagree about which turns are done:
    complete lines, a complete record that only lost its newline, and a line
    cut off in mid-write."""
    path = tmp_path / "turns.jsonl"
    path.write_text(
        json.dumps({"turnID": "t1", "output": "a"}) + "\n"
        + json.dumps({"turnID": "t2", "output": "b"}) + "\n"
        + json.dumps({"turnID": "t3", "output": "c"}), encoding="utf-8")
    assert resume.completed_turn_count(str(path)) == len(
        multi_agent._completed_turns(str(path))) == 3
    path.write_text(
        json.dumps({"turnID": "t1", "output": "a"}) + "\n"
        + '{"turnID": "t2", "outp', encoding="utf-8")
    assert resume.completed_turn_count(str(path)) == len(
        multi_agent._completed_turns(str(path))) == 1
