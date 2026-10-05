"""A person resumes a job that was cancelled in the middle of a response.

The engine half is in test_resume_from_records.py: a run directory with no
saved place can be continued from its completed response records. This file
pins what the controller does with that when a person resumes a cancelled
job (``JobManager.resubmit``):

- the check that the cancelled job has ENDED comes first and is unchanged;
  until it passes the answer is "wait", and the run directory is not touched;
- then the run's place is saved from its records, before anything is
  submitted, and the same script continues the run through its pointer;
- the answer says how many completed responses were kept, and that the
  response in progress when the run stopped is generated again;
- a run folder that cannot be continued from its records is refused, with
  the reason: no completed response, no study stamp, another version of the
  study, a failure record, or another part of a sharded run;
- pipelines and sharded runs: the run stage, and each part, by the same rule.

Scheduler job numbers here are made up for the fake scheduler.
"""

import json
import os

import pytest

from steerlab_server.api.jobs import ResubmitRefused
from steerlab_server.experiment import resume, run_status, tasks

from test_manual_resubmit import (  # noqa: F401 - fixtures by import
    _dummy_script, _manager, fake_slurm)
from test_resume_after_cancel import (_cancelled, _end_marker, _pipeline_job,
                                      _sharded_run)
from test_resume_from_records import (Killed, _lines, _record,
                                      _reference_run, _stamp)
from test_sharding import (_fake_model, _patch_study_fakes, _read,
                           _seed_sensitive_generate, _study_fixture)


# --- builders ----------------------------------------------------------------------

def _unpark(run_dir, *, torn=True):
    """Turn a parked run directory into the one a kill in mid-response
    leaves: stamped, its completed responses on disk, no saved place."""
    os.remove(os.path.join(run_dir, resume.RESUME_STATE_FILENAME))
    _stamp(str(run_dir))
    if torn:
        with open(os.path.join(run_dir, "generations.jsonl"), "ab") as handle:
            handle.write(b'{"condition": "baseline", "promptIndex": 99, "pro')


def test_a_job_stopped_in_mid_response_resumes_from_its_records(
        tmp_path, fake_slurm, monkeypatch):
    mgr = _manager(tmp_path)
    job, script, _records, run_dir = _cancelled(mgr, tmp_path, "8101",
                                                completed=4)
    _unpark(run_dir)
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "8102")

    outcome = mgr.resubmit(job.id)

    assert outcome["ok"] is True
    assert outcome["resumedAfterCancel"] is True
    assert outcome["resumedFromRecords"] is True
    # It says how many completed responses were kept, and that the response
    # in progress when the run stopped will be generated again.
    assert outcome["completedRecords"] == 4
    message = outcome["message"]
    assert "The 4 response records it had completed are kept" in message
    assert "will not be generated again" in message
    assert "stopped while a response was still being generated" in message
    assert "it will be generated again" in message
    # The SAME script, once; it finds a saved place and continues the run.
    calls = fake_slurm.calls("sbatch")
    assert len(calls) == 1 and calls[0].endswith(script)
    state = resume.read_state(str(run_dir))
    assert (state["reason"], state["completedRecords"]) == ("records", 4)
    pointer = resume.pointer_path_for_record(
        os.path.join(str(tmp_path / "records"), f"{job.id}.json"))
    assert resume.resolve_pointer(pointer, verb="run") == ("resume", str(run_dir))
    # The cut-off line is gone, the four complete ones are untouched.
    assert len(_lines(run_dir / "generations.jsonl")) == 4
    assert _read(str(run_dir / "generations.jsonl")).endswith(b"\n")
    # The record: still cancelled, and it says how the run was continued.
    original = mgr.get(job.id)
    assert original.status == "cancelled"
    stamp = original.result["resumedAfterCancel"]
    assert stamp["state"] == "records"
    assert stamp["completedRecords"] == 4
    assert stamp["endProof"] == "its exit marker is on disk"
    logs = original.all_logs()
    assert any("saved it from the 4 completed response record(s)" in line
               for line in logs)
    assert any("will be generated again" in line and "stays cancelled" in line
               for line in logs)
    child = mgr.get(outcome["jobId"])
    assert child.status == "submitted"
    assert child.requested_resources["resumedAfterCancel"] is True
    assert any("from its completed response records" in line
               for line in child.all_logs())


def test_the_check_that_the_cancelled_job_ended_still_comes_first(
        tmp_path, fake_slurm):
    """Not confirmed ended: the answer is "wait", exactly as for a parked
    run, and the run directory is not touched. Two processes must never
    write to one run directory, and a job still winding down may be about to
    save its own place."""
    mgr = _manager(tmp_path)
    job, _script, _records, run_dir = _cancelled(mgr, tmp_path, "8201",
                                                 ended=False)
    _unpark(run_dir)
    before = _read(str(run_dir / "generations.jsonl"))
    fake_slurm.set_state("8201", "CANCELLED by 1000", exit_code="0:15",
                         queue="COMPLETING")

    with pytest.raises(ResubmitRefused) as refused:
        mgr.resubmit(job.id)

    assert refused.value.wait is True
    assert "has not yet confirmed that it stopped" in str(refused.value)
    assert "wait a minute, then try again" in str(refused.value)
    assert fake_slurm.calls("sbatch") == []
    assert resume.read_state(str(run_dir)) is None
    assert _read(str(run_dir / "generations.jsonl")) == before


@pytest.mark.parametrize("damage,words", [
    ("no-records", "its run folder holds no completed response"),
    ("no-stamp", "does not say which version of the study wrote it"),
    ("other-study", "written under a different version of the study"),
    ("failed", "stopped on an error"),
])
def test_a_stopped_job_that_cannot_use_its_records_is_refused_with_the_reason(
        tmp_path, fake_slurm, damage, words):
    mgr = _manager(tmp_path)
    job, _script, _records, run_dir = _cancelled(mgr, tmp_path, "8301")
    _unpark(run_dir, torn=False)
    if damage == "no-records":
        (run_dir / "generations.jsonl").write_text("", encoding="utf-8")
    elif damage == "no-stamp":
        (run_dir / "experiment-hash.txt").unlink()
    elif damage == "other-study":
        with open(run_dir / "generations.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_record(7, experimentHash="d" * 64)) + "\n")
    elif damage == "failed":
        run_status.RunStatus(str(run_dir), stage="run").fail(
            RuntimeError("out of memory"))

    with pytest.raises(ResubmitRefused) as refused:
        mgr.resubmit(job.id)

    message = str(refused.value)
    assert refused.value.wait is False
    assert "stopped before it could save its place" in message
    assert words in message
    assert "submit the study again" in message
    assert fake_slurm.calls("sbatch") == []
    assert resume.read_state(str(run_dir)) is None
    assert mgr.get(job.id).status == "cancelled"


def test_a_run_folder_of_the_wrong_shard_is_refused_from_records_too(
        tmp_path, fake_slurm):
    mgr = _manager(tmp_path)
    job, _script, _records, run_dir = _cancelled(mgr, tmp_path, "8401")
    _unpark(run_dir)
    (run_dir / "shard.json").write_text(json.dumps(
        {"shardIndex": 1, "shardCount": 3}), encoding="utf-8")

    with pytest.raises(ResubmitRefused) as refused:
        mgr.resubmit(job.id)

    assert "its run folder holds part 1/3 but the job is a whole run" in str(
        refused.value)
    assert fake_slurm.calls("sbatch") == []
    assert resume.read_state(str(run_dir)) is None


def test_a_submission_that_fails_leaves_a_place_the_next_resume_finds(
        tmp_path, fake_slurm, monkeypatch):
    """The place is saved before sbatch, so a failed (or crashed) submission
    cannot leave a continuation that would start the study over. The next
    resume reads the records again and submits exactly once."""
    mgr = _manager(tmp_path)
    job, _script, _records, run_dir = _cancelled(mgr, tmp_path, "8501")
    _unpark(run_dir)
    monkeypatch.setenv("FAKE_SBATCH_FAIL", "sbatch: error: scheduler is down")

    with pytest.raises(Exception, match="scheduler is down"):
        mgr.resubmit(job.id)

    assert resume.read_state(str(run_dir))["reason"] == "records"
    waiting = mgr.get(job.id)
    assert waiting.status == "cancelled"
    assert not (waiting.result or {}).get("resubmittedAs")
    assert not (waiting.result or {}).get("resubmitClaim")

    monkeypatch.delenv("FAKE_SBATCH_FAIL")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "8502")
    outcome = mgr.resubmit(job.id)

    assert outcome["resumedFromRecords"] is True
    assert outcome["completedRecords"] == 3
    assert "it will be generated again" in outcome["message"]
    assert mgr.get(job.id).result["resumedAfterCancel"]["state"] == "records"


def test_a_pipeline_whose_run_stage_was_stopped_in_mid_response_resumes(
        tmp_path, fake_slurm, monkeypatch):
    mgr = _manager(tmp_path)
    job = _pipeline_job(mgr, tmp_path, "8601",
                        {"sweep": {"status": "completed"},
                         "run": {"status": "started"}}, run_state=None)
    run_dir = tmp_path / "runs" / f"exp-demo-run-{job.id}"
    pipeline_dir = tmp_path / "runs" / f"exp-demo-pipeline-{job.id}"
    _stamp(str(run_dir))
    (run_dir / "generations.jsonl").write_text(
        "".join(json.dumps(_record(i)) + "\n" for i in range(5)),
        encoding="utf-8")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "8602")

    outcome = mgr.resubmit(job.id)

    assert outcome["resumedFromRecords"] is True
    assert outcome["completedRecords"] == 5
    # The place is saved in the run stage's own folder, which is where the
    # pipeline's run stage looks; the pipeline folder gets none.
    assert resume.is_resumable(str(run_dir), "run")
    assert resume.read_state(str(pipeline_dir)) is None


def test_a_sharded_run_with_a_part_stopped_in_mid_response_resumes_whole(
        tmp_path, fake_slurm, monkeypatch):
    mgr = _manager(tmp_path)
    parent, children, script = _sharded_run(
        mgr, tmp_path, 8700, states=["parked", "unparked"])
    stopped = tmp_path / "runs" / f"exp-demo-run-{children[1].id}"
    _stamp(str(stopped))
    (stopped / "generations.jsonl").write_text(
        "".join(json.dumps(_record(i)) + "\n" for i in range(2)),
        encoding="utf-8")
    _end_marker(script, "8700")
    _end_marker(script, "8701", status="137")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "8710")

    outcome = mgr.resubmit(parent.id)

    assert outcome["resumedAfterCancel"] is True
    assert outcome["resumedFromRecords"] is True
    assert len(fake_slurm.calls("sbatch")) == 2
    parked, from_records = outcome["resumedShards"]
    assert "resumedFromRecords" not in parked
    assert from_records["resumedFromRecords"] is True
    assert from_records["completedRecords"] == 2
    assert "was still being generated when its part stopped" in outcome["message"]
    assert resume.read_state(str(stopped))["reason"] == "records"
    assert mgr.get(parent.id).status == "running"


def test_a_sharded_run_with_a_part_that_completed_no_response_resumes_whole(
        tmp_path, fake_slurm, monkeypatch):
    """One part had begun and was stopped during its first response. Its
    folder holds nothing to continue from, and nothing to lose: like a part
    that never began, it starts from the beginning, so the responses the
    other parts completed are not thrown away with it. (A single job in the
    same state is still refused: starting it again IS submitting the study
    again.)"""
    mgr = _manager(tmp_path)
    parent, children, script = _sharded_run(
        mgr, tmp_path, 8750, states=["parked", "unparked"])
    empty = tmp_path / "runs" / f"exp-demo-run-{children[1].id}"
    _stamp(str(empty))
    (empty / "generations.jsonl").write_text(json.dumps(_record(0))[:30],
                                             encoding="utf-8")
    _end_marker(script, "8750")
    _end_marker(script, "8751", status="137")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "8760")

    outcome = mgr.resubmit(parent.id)

    assert outcome["resumedAfterCancel"] is True
    assert "resumedFromRecords" not in outcome
    assert len(fake_slurm.calls("sbatch")) == 2
    _parked, restarted = outcome["resumedShards"]
    assert "before it had completed a response" in restarted["message"]
    assert "starts from the beginning" in restarted["message"]
    assert restarted["completedRecords"] is None
    stamp = mgr.get(children[1].id).result["resumedAfterCancel"]
    assert stamp["state"] == "empty"
    # No place is saved for it: the re-executed script starts a fresh folder.
    assert resume.read_state(str(empty)) is None
    pointer = resume.pointer_path_for_record(
        os.path.join(str(tmp_path / "records"), f"{children[1].id}.json"))
    assert resume.resolve_pointer(pointer, verb="run") == ("fresh", None)


def test_a_resumed_job_stopped_in_mid_response_again_counts_its_records_afresh(
        tmp_path, fake_slurm, monkeypatch):
    """The continuation ran on from the saved place, added responses, and was
    cancelled in mid-response too. The place saved for it is out of date, so
    the answer counts the lines on disk."""
    mgr = _manager(tmp_path)
    job, script, _records, run_dir = _cancelled(mgr, tmp_path, "8801",
                                                completed=3)
    _unpark(run_dir, torn=False)
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "8802")
    first = mgr.resubmit(job.id)
    assert first["completedRecords"] == 3
    with open(run_dir / "generations.jsonl", "a", encoding="utf-8") as handle:
        for index in range(3, 7):
            handle.write(json.dumps(_record(index)) + "\n")
        handle.write(json.dumps(_record(7))[:18])
    assert mgr.cancel(first["jobId"]) is True
    _end_marker(script, "8802", status="137")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "8803")

    second = mgr.resubmit(first["jobId"])

    assert second["resumedFromRecords"] is True
    assert second["completedRecords"] == 7
    assert resume.read_state(str(run_dir))["completedRecords"] == 7
    assert len(fake_slurm.calls("sbatch")) == 2


def test_the_whole_path_from_a_cancel_in_mid_response_to_the_finished_run(
        tmp_path, fake_slurm, monkeypatch):
    """The cancelled job's child is killed in the middle of a response. A
    person resumes the job; the controller confirms the job ended and saves
    the run's place from its records; the re-executed script resolves the
    same pointer and finishes the run. The result is byte-identical to a run
    that was never interrupted, and only the cut-off response was generated
    twice."""
    root = str(tmp_path / "workspace")
    prompts = _study_fixture(root, "cancelmid")
    reference_dir, total = _reference_run(root, "cancelmid", prompts, monkeypatch)
    reference = _read(os.path.join(reference_dir, "generations.jsonl"))

    mgr = _manager(tmp_path)
    script = _dummy_script(tmp_path)
    records = tmp_path / "records"
    records.mkdir()
    job = mgr.record_external(
        "study-submit", status="running", executor="slurm",
        executor_job_id="8901",
        requested_resources={"scriptPath": script, "walltime": "00:10:00",
                             "recordsDirectory": str(records)})
    pointer = resume.pointer_path_for_record(str(records / f"{job.id}.json"))

    # The cancel's signal arrives during the seventh response and asks the
    # child to save its place once that response is done. The response
    # outlasts the grace period and the child is killed instead: no place
    # saved, no child record.
    flag = resume.CheckpointFlag()
    counter = [0]
    inner = _seed_sensitive_generate(counter)

    def cancelled_in_mid_response(model, prompt, **kwargs):
        text = inner(model, prompt, **kwargs)
        if counter[0] == 7:
            flag.request()
            raise Killed()
        return text

    _patch_study_fakes(monkeypatch, cancelled_in_mid_response)
    with pytest.raises(Killed):
        tasks.run("cancelmid", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, checkpoint=flag,
                  on_run_directory=lambda created: resume.write_pointer(
                      pointer, created, verb="run", experiment="cancelmid"))
    started = counter[0]
    run_dir = resume.read_pointer(pointer)["runDirectory"]
    assert mgr.cancel(job.id) is True
    _end_marker(script, "8901", status="137")
    kept = _read(os.path.join(run_dir, "generations.jsonl"))
    assert resume.read_state(run_dir) is None
    assert resume.resolve_pointer(pointer, verb="run") == ("fresh", None)

    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "8902")
    outcome = mgr.resubmit(job.id)
    assert outcome["resumedFromRecords"] is True
    assert outcome["completedRecords"] == len(kept.splitlines())

    disposition, directory = resume.resolve_pointer(pointer, verb="run")
    assert (disposition, directory) == ("resume", run_dir)
    after = [0]
    _patch_study_fakes(monkeypatch, _seed_sensitive_generate(after))
    finished = tasks.run("cancelmid", prompts, root, model_provider=_fake_model,
                         log=lambda *_: None, run_directory=directory)

    assert finished == run_dir
    final = _read(os.path.join(run_dir, "generations.jsonl"))
    assert final.startswith(kept)
    assert final == reference
    assert started + after[0] == total + 1
    assert resume.is_complete(run_dir)
    assert resume.read_state(run_dir) is None
