"""A person can resume a run they cancelled (2026-10-04).

Cancelling a scheduler job parks its run the way a walltime checkpoint does
(the cancel's SIGTERM sets the same flag), so the same sbatch script can
continue it and every completed response is kept. This file pins the gate
that makes that safe, and what the record says afterwards:

- cancelled, parked, and confirmed ended: the MANUAL verb resumes it, exactly
  once under a double request;
- cancelled but possibly still running: refused, with "wait, then try again"
  (two processes must never write to one run directory);
- cancelled with no kept state: refused, with "submit the study again";
- the AUTOMATIC path still refuses every cancelled job;
- after the resume, completed responses are not generated again;
- the cancelled record stays cancelled and links to its continuation, which
  is an ordinary job;
- sharded parents, judging workers, later pipeline stages, and local jobs
  each get the outcome the report describes.
"""

import json
import os
import threading
import time

import pytest

from steerlab_server.api.executors import job_end_marker_path
from steerlab_server.api.jobs import (ResubmitRefused, cancel_resume_hint)
from steerlab_server.experiment import resume, tasks

from test_manual_resubmit import (  # noqa: F401 - fixtures by import
    _dummy_script, _manager, fake_slurm)
from test_resubmit_race import SlowFakeExecutor
from test_sharding import (_fake_model, _patch_study_fakes, _read,
                           _seed_sensitive_generate, _study_fixture)


# --- builders ----------------------------------------------------------------------

def _park(tmp_path, records, job_id, slurm_id, *, completed=3, verb="run",
          shard=None):
    """What a cancelled child leaves when it parks: the flushed responses,
    resume-state.json, the resume pointer beside the child record, and the
    child record itself (status ``checkpointed``)."""
    run_dir = tmp_path / "runs" / f"exp-demo-run-{job_id}"
    run_dir.mkdir(parents=True)
    (run_dir / "generations.jsonl").write_text("".join(
        json.dumps({"condition": "baseline", "promptIndex": i,
                    "promptID": f"p{i}", "sampleIndex": 0,
                    "response": f"answer {i}"}) + "\n"
        for i in range(completed)), encoding="utf-8")
    resume.write_state(str(run_dir), run_id=run_dir.name, verb=verb,
                       completed_records=completed, reason="signal")
    if shard is not None:
        (run_dir / "shard.json").write_text(json.dumps(
            {"shardIndex": shard[0], "shardCount": shard[1]}), encoding="utf-8")
    record_path = records / f"{job_id}.json"
    resume.write_pointer(resume.pointer_path_for_record(str(record_path)),
                         str(run_dir), verb=verb, experiment="demo")
    record_path.write_text(json.dumps({
        "id": job_id, "status": "checkpointed", "executor": "slurm",
        "executorJobID": str(slurm_id), "finishedAt": time.time(),
        "result": {"runDirectory": str(run_dir),
                   "resumeState": {"completedRecords": completed,
                                   "reason": "signal"}},
        "logs": []}), encoding="utf-8")
    return run_dir


def _end_marker(script, slurm_id, status="85"):
    """The marker the script's EXIT trap writes last: the job has ended."""
    with open(job_end_marker_path(os.path.dirname(script), str(slurm_id)),
              "w", encoding="utf-8") as handle:
        handle.write(f"{status}\n")


def _cancelled(mgr, tmp_path, slurm_id, *, parked=True, completed=3,
               kind="study-submit", extra_rr=None, ended=True):
    """A scheduler job that was running, kept ``completed`` responses, and
    was then cancelled through the manager (fake scancel accepts)."""
    script = _dummy_script(tmp_path)
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    rr = {"scriptPath": script, "walltime": "00:10:00",
          "recordsDirectory": str(records)}
    rr.update(extra_rr or {})
    job = mgr.record_external(kind, status="running", executor="slurm",
                              executor_job_id=str(slurm_id),
                              requested_resources=rr)
    run_dir = None
    if parked:
        shard = ((rr["shardIndex"], rr["shardCount"])
                 if "shardCount" in rr else None)
        run_dir = _park(tmp_path, records, job.id, slurm_id,
                        completed=completed, shard=shard)
    assert mgr.cancel(job.id) is True
    assert mgr.get(job.id).status == "cancelled"
    if ended:
        _end_marker(script, slurm_id)
    return job, script, records, run_dir


# --- 1. cancelled, parked, and confirmed ended: resumes ----------------------------

def test_cancelled_parked_and_ended_job_resumes_as_a_new_ordinary_job(
        tmp_path, fake_slurm, monkeypatch):
    mgr = _manager(tmp_path)
    job, script, _records, run_dir = _cancelled(mgr, tmp_path, "9101")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9102")

    outcome = mgr.resubmit(job.id)

    assert outcome["ok"] is True
    assert outcome["resubmitOf"] == job.id
    assert outcome["slurmJobID"] == "9102"
    assert outcome["manualResubmit"] is True
    assert outcome["resumedAfterCancel"] is True
    assert outcome["completedRecords"] == 3
    assert "kept and will not be generated again" in outcome["message"]
    # The SAME script, once.
    calls = fake_slurm.calls("sbatch")
    assert len(calls) == 1 and calls[0].endswith(script)

    # The continuation: stamped manual and resumed-after-cancel, linked to
    # the original, and NOT itself cancelled.
    child = mgr.get(outcome["jobId"])
    assert child.status == "submitted"
    assert child.cancelled is False
    rr = child.requested_resources
    assert rr["manualResubmit"] is True
    assert rr["resumedAfterCancel"] is True
    assert rr["resubmitOf"] == job.id
    assert rr["resubmitChain"] == [job.id]

    # The original: still cancelled, with the cancellation still on record,
    # and linked to its continuation.
    original = mgr.get(job.id)
    assert original.status == "cancelled"
    assert original.cancelled is True
    assert original.result["resubmittedAs"] == child.id
    stamp = original.result["resumedAfterCancel"]
    assert stamp["continuation"] == child.id
    assert stamp["cancelledSchedulerJobID"] == "9101"
    assert stamp["completedRecords"] == 3
    assert stamp["state"] == "parked"
    assert stamp["runDirectory"] == str(run_dir)
    assert stamp["endProof"] == "its exit marker is on disk"
    assert any("resumed by a person as" in line and "stays cancelled" in line
               for line in original.all_logs())
    # The durable store says the same after a restart.
    reloaded = _manager(tmp_path)
    assert reloaded.get(job.id).status == "cancelled"
    assert reloaded.get(job.id).cancelled is True
    assert reloaded.get(child.id).cancelled is False


def test_the_schedulers_own_record_proves_the_end_without_a_marker(
        tmp_path, fake_slurm, monkeypatch):
    """A job the scheduler killed outright writes no marker. The scheduler's
    record then decides: gone from the queue and CANCELLED in accounting."""
    mgr = _manager(tmp_path)
    job, _script, _records, _run = _cancelled(mgr, tmp_path, "9111", ended=False)
    fake_slurm.set_state("9111", "CANCELLED by 1000", exit_code="0:15")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9112")

    outcome = mgr.resubmit(job.id)

    assert outcome["resumedAfterCancel"] is True
    assert len(fake_slurm.calls("sbatch")) == 1
    proof = mgr.get(job.id).result["resumedAfterCancel"]["endProof"]
    assert "accounting records it as CANCELLED" in proof


def test_a_double_request_resumes_exactly_once(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    monkeypatch.delenv("STEERLAB_MAINTENANCE_CALENDAR", raising=False)
    from steerlab_server.api.jobs import DurableJobStore, JobManager
    fake = SlowFakeExecutor(delay=0.5)
    mgr = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")),
                     capability_provider=lambda: {}, slurm_executor=fake)
    job, _script, _records, _run = _cancelled(mgr, tmp_path, "9121")

    barrier = threading.Barrier(2)
    results, errors = [], []

    def resume_it():
        barrier.wait()
        try:
            results.append(mgr.resubmit(job.id))
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=resume_it) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert len(fake.submits) == 1               # exactly ONE submission
    assert len({r["jobId"] for r in results}) == 1
    assert sum(1 for r in results if r.get("alreadyResumed")) == 1
    stamped = mgr.get(job.id).result
    assert stamped["resubmittedAs"] == results[0]["jobId"]
    assert "resubmitClaim" not in stamped
    # A third, later request is told where the run went; still one submission.
    with pytest.raises(ResubmitRefused, match="already resubmitted as"):
        mgr.resubmit(job.id)
    assert len(fake.submits) == 1


def _leave_a_dead_claim(mgr, job_id, token="slre-dead-abc123"):
    """The crash window: an earlier resume claimed the job and its process
    died, perhaps after its submission went through."""
    from steerlab_server.api.jobs import DurableJobStore
    job = mgr.get(job_id)
    job.result = {**(job.result or {}), "resubmitClaim": {
        "claimant": "manual:dead:000000", "token": token,
        "at": time.time() - DurableJobStore.RESUBMIT_CLAIM_STALE_SECONDS - 60}}
    mgr.store.update(job)


def test_a_crashed_resume_whose_submission_went_through_is_adopted(
        tmp_path, fake_slurm, monkeypatch):
    mgr = _manager(tmp_path)
    job, _script, _records, _run = _cancelled(mgr, tmp_path, "9131")
    _leave_a_dead_claim(mgr, job.id)
    monkeypatch.setenv("FAKE_SQUEUE_NAMED", "9132")   # the scheduler knows it

    outcome = mgr.resubmit(job.id)

    assert outcome["alreadyResumed"] is True
    assert outcome["resumedAfterCancel"] is True
    assert fake_slurm.calls("sbatch") == []            # nothing submitted twice
    child = mgr.get(outcome["jobId"])
    assert child.executor_job_id == "9132"
    assert child.requested_resources["resumedAfterCancel"] is True
    assert child.requested_resources["manualResubmit"] is True
    assert child.requested_resources["adoptedFromToken"] is True
    original = mgr.get(job.id)
    assert original.status == "cancelled"
    assert original.result["resumedAfterCancel"]["continuation"] == child.id
    assert "resubmitClaim" not in original.result


def test_a_crashed_resume_that_never_submitted_is_resumed_once(
        tmp_path, fake_slurm, monkeypatch):
    mgr = _manager(tmp_path)
    job, _script, _records, _run = _cancelled(mgr, tmp_path, "9141")
    _leave_a_dead_claim(mgr, job.id)
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9142")

    outcome = mgr.resubmit(job.id)

    assert outcome["resumedAfterCancel"] is True
    assert "alreadyResumed" not in outcome
    assert len(fake_slurm.calls("sbatch")) == 1
    child = mgr.get(outcome["jobId"])
    assert child.requested_resources["resumedAfterCancel"] is True
    assert mgr.get(job.id).result["resubmittedAs"] == child.id


# --- 2. cancelled but possibly still running: wait, then try again -----------------

@pytest.mark.parametrize("state,queue", [
    ("CANCELLED by 1000", "COMPLETING"),   # accounting says cancelled; still winding down
    ("RUNNING", "RUNNING"),
    (None, None),                          # the scheduler says nothing at all
])
def test_a_cancelled_job_not_confirmed_ended_is_refused_with_wait(
        tmp_path, fake_slurm, state, queue):
    mgr = _manager(tmp_path)
    job, _script, _records, _run = _cancelled(mgr, tmp_path, "9201", ended=False)
    if state is not None:
        fake_slurm.set_state("9201", state, exit_code="0:15", queue=queue)

    with pytest.raises(ResubmitRefused) as refused:
        mgr.resubmit(job.id)

    assert refused.value.wait is True
    message = str(refused.value)
    assert "has not yet confirmed that it stopped" in message
    assert "wait a minute, then try again" in message
    assert fake_slurm.calls("sbatch") == []
    original = mgr.get(job.id)
    assert original.status == "cancelled"
    assert not (original.result or {}).get("resubmittedAs")
    assert not (original.result or {}).get("resubmitClaim")


def test_a_queue_that_cannot_be_read_is_not_an_ended_job(
        tmp_path, fake_slurm, monkeypatch):
    """Accounting alone is not proof: it reads CANCELLED while the job's
    processes are still being stopped, so a failed queue query refuses."""
    mgr = _manager(tmp_path)
    job, _script, _records, _run = _cancelled(mgr, tmp_path, "9211", ended=False)
    fake_slurm.set_state("9211", "CANCELLED by 1000", exit_code="0:15")
    monkeypatch.setenv("FAKE_SQUEUE_FAIL", "Unable to contact slurm controller")
    with pytest.raises(ResubmitRefused, match="queue could not be read"):
        mgr.resubmit(job.id)
    assert fake_slurm.calls("sbatch") == []
    # The queue answering "I no longer know that id" is the forgotten,
    # finished job: accounting then decides.
    monkeypatch.setenv("FAKE_SQUEUE_FAIL",
                       "slurm_load_jobs error: Invalid job id specified")
    assert mgr.resubmit(job.id)["resumedAfterCancel"] is True
    assert len(fake_slurm.calls("sbatch")) == 1


# --- 3. cancelled with no kept state: submit the study again ------------------------

def test_a_job_cancelled_before_it_started_is_refused(tmp_path, fake_slurm):
    mgr = _manager(tmp_path)
    job, _script, _records, _run = _cancelled(mgr, tmp_path, "9301",
                                              parked=False)
    with pytest.raises(ResubmitRefused) as refused:
        mgr.resubmit(job.id)
    assert refused.value.wait is False
    assert "submit the study again" in str(refused.value)
    assert fake_slurm.calls("sbatch") == []


def test_a_job_stopped_before_it_could_save_its_place_is_refused(
        tmp_path, fake_slurm):
    """A started run with no resume-state.json: re-executing the script would
    begin a fresh run directory and generate every response again."""
    mgr = _manager(tmp_path)
    job, _script, _records, run_dir = _cancelled(mgr, tmp_path, "9311")
    os.remove(os.path.join(run_dir, resume.RESUME_STATE_FILENAME))
    with pytest.raises(ResubmitRefused) as refused:
        mgr.resubmit(job.id)
    message = str(refused.value)
    assert "before it could save its place" in message
    assert "submit the study again" in message
    assert fake_slurm.calls("sbatch") == []
    assert mgr.get(job.id).status == "cancelled"


def test_a_run_that_had_finished_when_the_cancel_landed_is_only_reported(
        tmp_path, fake_slurm, monkeypatch):
    """The cancel raced the end of the run: the directory is complete. The
    script recognises a complete run and only reports and packages it, so
    the resume is admitted (a refusal here would strand finished results,
    and for one part of a sharded run would block the merge)."""
    mgr = _manager(tmp_path)
    job, _script, _records, run_dir = _cancelled(mgr, tmp_path, "9331")
    os.remove(os.path.join(run_dir, resume.RESUME_STATE_FILENAME))
    (run_dir / resume.COMPLETION_FILENAME).write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9332")

    outcome = mgr.resubmit(job.id)

    assert outcome["resumedAfterCancel"] is True
    assert "had already finished its run" in outcome["message"]
    assert "Nothing is generated again" in outcome["message"]
    assert len(fake_slurm.calls("sbatch")) == 1
    assert mgr.get(job.id).result["resumedAfterCancel"]["state"] == "complete"


def test_a_run_folder_of_the_wrong_shard_is_refused(tmp_path, fake_slurm):
    mgr = _manager(tmp_path)
    job, _script, _records, run_dir = _cancelled(mgr, tmp_path, "9321")
    (run_dir / "shard.json").write_text(json.dumps(
        {"shardIndex": 1, "shardCount": 3}), encoding="utf-8")
    with pytest.raises(ResubmitRefused) as refused:
        mgr.resubmit(job.id)
    message = str(refused.value)
    assert "its run folder holds part 1/3 but the job is a whole run" in message
    assert "submit the study again" in message
    assert fake_slurm.calls("sbatch") == []


# --- 4. the automatic path still refuses every cancelled job ------------------------

def test_the_automatic_path_never_revives_a_cancelled_job(tmp_path, fake_slurm):
    mgr = _manager(tmp_path)
    job, _script, records, _run = _cancelled(
        mgr, tmp_path, "9401",
        extra_rr={"auto_resubmit": True, "auto_resubmit_limit": 5})
    # Resumable by a person in every respect — and the reconciler leaves it.
    mgr.poll_slurm()
    mgr.poll_slurm()
    assert fake_slurm.calls("sbatch") == []
    assert mgr.get(job.id).status == "cancelled"

    # The shape a late child-record fold leaves: "checkpointed" beside the
    # cancel flag, with the scheduler itself reporting the checkpoint exit.
    fake_slurm.set_state("9401", "FAILED", exit_code="85:0")
    assert mgr.reconcile(str(records), force=True) == 1
    revived = mgr.get(job.id)
    assert revived.status == "checkpointed" and revived.cancelled is True
    mgr._checkpoint_witnessed[job.id] = time.time()
    mgr.poll_slurm()
    assert mgr._maybe_auto_resubmit(revived) is False
    assert fake_slurm.calls("sbatch") == []
    assert any("cancelled beats checkpointed" in line
               for line in revived.all_logs())
    # The shared submit core refuses a cancelled job without a person's
    # admission, whoever calls it.
    with pytest.raises(RuntimeError, match="cancellation on record"):
        mgr._perform_resubmit(revived, limit=5)
    with pytest.raises(RuntimeError, match="manual verb only"):
        mgr._perform_resubmit(revived, limit=5, after_cancel={"state": "parked"})
    assert fake_slurm.calls("sbatch") == []


def test_a_person_can_resume_the_fold_restored_shape_and_it_reads_cancelled(
        tmp_path, fake_slurm, monkeypatch):
    mgr = _manager(tmp_path)
    job, _script, records, _run = _cancelled(mgr, tmp_path, "9411")
    assert mgr.reconcile(str(records), force=True) == 1
    assert mgr.get(job.id).status == "checkpointed"     # fold restored it
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9412")

    outcome = mgr.resubmit(job.id)

    assert outcome["resumedAfterCancel"] is True
    original = mgr.get(job.id)
    assert original.status == "cancelled"
    assert original.finished_at is not None
    assert original.result["resubmittedAs"] == outcome["jobId"]


def test_the_continuation_is_an_ordinary_job_to_the_automatic_path(
        tmp_path, fake_slurm, monkeypatch):
    """Resumed by a person, then it hits its own walltime: the reconciler
    resubmits the CONTINUATION like any other checkpointed job, and the
    cancelled original is left exactly as it was."""
    mgr = _manager(tmp_path)
    job, _script, _records, _run = _cancelled(
        mgr, tmp_path, "9421",
        extra_rr={"auto_resubmit": True, "auto_resubmit_limit": 5})
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9422")
    child_id = mgr.resubmit(job.id)["jobId"]

    fake_slurm.set_state("9422", "FAILED", exit_code="85:0")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9423")
    mgr.poll_slurm()

    assert len(fake_slurm.calls("sbatch")) == 2
    child = mgr.get(child_id)
    grandchild = mgr.get(child.result["resubmittedAs"])
    assert grandchild.executor_job_id == "9423"
    assert grandchild.requested_resources["resubmitOf"] == child_id
    # The automatic continuation was not born from a cancel.
    assert "resumedAfterCancel" not in grandchild.requested_resources
    assert mgr.get(job.id).status == "cancelled"


# --- 5. completed responses are not generated again --------------------------------

def test_completed_responses_are_not_generated_again_after_the_resume(
        tmp_path, fake_slurm, monkeypatch):
    """The cancelled child parks through the same flag a walltime checkpoint
    uses. After a person resumes the job, the re-executed script resolves the
    pointer, skips every completed record, and generates only the rest: the
    finished run is byte-identical to one that was never interrupted, and no
    response was generated twice."""
    root = str(tmp_path / "workspace")
    prompts = _study_fixture(root, "cancelresume")
    total = [0]
    _patch_study_fakes(monkeypatch, _seed_sensitive_generate(total))
    uninterrupted = tasks.run("cancelresume", prompts, root,
                              model_provider=_fake_model, log=lambda *_: None)
    reference = _read(os.path.join(uninterrupted, "generations.jsonl"))

    mgr = _manager(tmp_path)
    script = _dummy_script(tmp_path)
    records = tmp_path / "records"
    records.mkdir()
    job = mgr.record_external(
        "study-submit", status="running", executor="slurm",
        executor_job_id="9501",
        requested_resources={"scriptPath": script, "walltime": "00:10:00",
                             "recordsDirectory": str(records)})
    pointer = resume.pointer_path_for_record(str(records / f"{job.id}.json"))

    # The cancelled job's child: the cancel arrives during the fourth
    # response, which finishes; the run then parks instead of going on.
    flag = resume.CheckpointFlag()
    before = [0]
    _patch_study_fakes(monkeypatch, _seed_sensitive_generate(
        before, arm_flag_at=4, flag=flag))
    with pytest.raises(resume.CheckpointRequested):
        tasks.run("cancelresume", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, checkpoint=flag,
                  on_run_directory=lambda created: resume.write_pointer(
                      pointer, created, verb="run", experiment="cancelresume"))
    assert mgr.cancel(job.id) is True
    _end_marker(script, "9501")
    parked_dir = resume.read_pointer(pointer)["runDirectory"]
    kept = _read(os.path.join(parked_dir, "generations.jsonl"))
    kept_records = len(kept.splitlines())
    assert 0 < kept_records < len(reference.splitlines())

    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9502")
    outcome = mgr.resubmit(job.id)
    assert outcome["completedRecords"] == kept_records

    # The continuation's child: the same script consults the same pointer.
    disposition, directory = resume.resolve_pointer(pointer, verb="run")
    assert (disposition, directory) == ("resume", parked_dir)
    after = [0]
    _patch_study_fakes(monkeypatch, _seed_sensitive_generate(after))
    finished = tasks.run("cancelresume", prompts, root,
                         model_provider=_fake_model, log=lambda *_: None,
                         run_directory=directory)

    assert finished == parked_dir
    final = _read(os.path.join(finished, "generations.jsonl"))
    assert final.startswith(kept)          # what was kept is untouched
    assert final == reference              # and the whole equals uninterrupted
    assert before[0] + after[0] == total[0]    # nothing generated twice
    assert resume.is_complete(finished)


# --- 6. the original stays cancelled and links to its continuation ------------------

def test_the_continuations_record_folds_onto_the_continuation_not_the_original(
        tmp_path, fake_slurm, monkeypatch):
    mgr = _manager(tmp_path)
    job, _script, records, run_dir = _cancelled(mgr, tmp_path, "9601")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9602")
    child_id = mgr.resubmit(job.id)["jobId"]

    # The record the CANCELLED job left is still on disk. Folding it must not
    # touch the continuation (it has only just been submitted) or revive the
    # original.
    mgr.reconcile(str(records), force=True)
    assert mgr.get(child_id).status == "submitted"
    assert mgr.get(child_id).executor_job_id == "9602"
    assert mgr.get(job.id).status == "cancelled"

    # The continuation finishes. The same script writes the same record file,
    # under the ORIGINAL's id.
    (records / f"{job.id}.json").write_text(json.dumps({
        "id": job.id, "status": "succeeded", "executor": "slurm",
        "executorJobID": "9602", "finishedAt": time.time(),
        "result": {"runDirectory": str(run_dir),
                   "evidenceBundle": {"bundlePath": "bundles/demo.tar.gz"}},
        "recordCount": 14, "logs": ["child: finished"]}), encoding="utf-8")
    fake_slurm.set_state("9602", "COMPLETED")
    mgr.poll_slurm()

    child = mgr.get(child_id)
    assert child.status == "succeeded"
    assert child.executor_job_id == "9602"
    assert child.result["runDirectory"] == str(run_dir)
    assert child.result["evidenceBundle"]["bundlePath"] == "bundles/demo.tar.gz"
    assert child.result["recordCount"] == 14
    original = mgr.get(job.id)
    assert original.status == "cancelled"
    assert original.cancelled is True
    assert original.executor_job_id == "9601"
    assert original.result["resubmittedAs"] == child_id
    assert "evidenceBundle" not in original.result

    # A restarted controller folds everything afresh and reaches the same
    # two records.
    restarted = _manager(tmp_path)
    restarted.reconcile(str(records), force=True)
    assert restarted.get(job.id).status == "cancelled"
    assert restarted.get(child_id).status == "succeeded"


def test_a_cancel_later_in_a_resubmit_chain_folds_onto_the_new_continuation(
        tmp_path, fake_slurm, monkeypatch):
    """The cancelled link need not be the first one. A run that checkpointed,
    was resubmitted, and was THEN cancelled writes its record under the first
    job's id all along; after the resume, that record belongs to the newest
    continuation, and neither earlier record is rewritten."""
    mgr = _manager(tmp_path)
    script = _dummy_script(tmp_path)
    records = tmp_path / "records"
    records.mkdir()
    root = mgr.record_external(
        "study-submit", status="checkpointed", executor="slurm",
        executor_job_id="9621",
        requested_resources={"scriptPath": script, "walltime": "00:10:00",
                             "recordsDirectory": str(records)})
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9622")
    first_id = mgr.resubmit(root.id)["jobId"]      # an ordinary checkpoint resume
    # The first continuation runs, keeps six responses, and is cancelled.
    run_dir = _park(tmp_path, records, root.id, "9622", completed=6)
    assert mgr.cancel(first_id) is True
    _end_marker(script, "9622")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9623")

    outcome = mgr.resubmit(first_id)

    assert outcome["resumedAfterCancel"] is True
    assert outcome["completedRecords"] == 6
    second_id = outcome["jobId"]
    (records / f"{root.id}.json").write_text(json.dumps({
        "id": root.id, "status": "succeeded", "executor": "slurm",
        "executorJobID": "9623", "finishedAt": time.time(),
        "result": {"runDirectory": str(run_dir)}, "logs": []}),
        encoding="utf-8")
    fake_slurm.set_state("9623", "COMPLETED")
    mgr.poll_slurm()

    second = mgr.get(second_id)
    assert second.status == "succeeded"
    assert second.result["runDirectory"] == str(run_dir)
    assert mgr.get(first_id).status == "cancelled"
    assert mgr.get(first_id).result["resubmittedAs"] == second_id
    untouched = mgr.get(root.id)
    assert untouched.status == "checkpointed"
    assert untouched.executor_job_id == "9621"
    assert "runDirectory" not in (untouched.result or {})


def test_cancelling_the_original_again_stops_its_continuation(
        tmp_path, fake_slurm, monkeypatch):
    mgr = _manager(tmp_path)
    job, _script, _records, _run = _cancelled(mgr, tmp_path, "9611")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9612")
    child_id = mgr.resubmit(job.id)["jobId"]
    assert mgr.cancel(job.id) is True
    assert mgr.get(child_id).status == "cancelled"
    assert "9612" in fake_slurm.calls("scancel")[-1]


# --- pipelines: the ledger is the kept state ----------------------------------------

def _pipeline_job(mgr, tmp_path, slurm_id, ledger, *, run_state=None):
    script = _dummy_script(tmp_path)
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    job = mgr.record_external(
        "study-submit-bundle", status="running", executor="slurm",
        executor_job_id=str(slurm_id),
        requested_resources={"scriptPath": script, "walltime": "00:10:00",
                             "recordsDirectory": str(records)})
    pipeline_dir = tmp_path / "runs" / f"exp-demo-pipeline-{job.id}"
    pipeline_dir.mkdir(parents=True)
    run_dir = tmp_path / "runs" / f"exp-demo-run-{job.id}"
    run_dir.mkdir()
    if run_state == "parked":
        resume.write_state(str(run_dir), run_id=run_dir.name, verb="run",
                           completed_records=5, reason="signal")
    stage_results = dict(ledger)
    if "run" in stage_results:
        stage_results["run"] = {**stage_results["run"],
                                "runDirectory": str(run_dir)}
    (pipeline_dir / "pipeline.json").write_text(json.dumps({
        "experiment": "demo", "stages": ["sweep", "run", "analyze"],
        "disposition": None, "stageResults": stage_results}), encoding="utf-8")
    resume.write_pointer(
        resume.pointer_path_for_record(str(records / f"{job.id}.json")),
        str(pipeline_dir), verb="pipeline", experiment="demo")
    assert mgr.cancel(job.id) is True
    _end_marker(script, slurm_id)
    return job


def test_a_pipeline_cancelled_in_its_run_stage_resumes_from_the_parked_run(
        tmp_path, fake_slurm, monkeypatch):
    mgr = _manager(tmp_path)
    job = _pipeline_job(mgr, tmp_path, "9701",
                        {"sweep": {"status": "completed"},
                         "run": {"status": "started"}}, run_state="parked")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9702")
    outcome = mgr.resubmit(job.id)
    assert outcome["resumedAfterCancel"] is True
    assert outcome["completedRecords"] == 5


def test_a_pipeline_whose_run_stage_saved_no_place_is_refused(
        tmp_path, fake_slurm):
    mgr = _manager(tmp_path)
    job = _pipeline_job(mgr, tmp_path, "9711",
                        {"sweep": {"status": "completed"},
                         "run": {"status": "started"}}, run_state=None)
    with pytest.raises(ResubmitRefused) as refused:
        mgr.resubmit(job.id)
    message = str(refused.value)
    assert "before it could save its place" in message
    assert "submit the study again" in message
    assert fake_slurm.calls("sbatch") == []


def test_a_pipeline_with_no_finished_stage_has_nothing_to_resume(
        tmp_path, fake_slurm):
    mgr = _manager(tmp_path)
    job = _pipeline_job(mgr, tmp_path, "9721", {})
    with pytest.raises(ResubmitRefused) as refused:
        mgr.resubmit(job.id)
    message = str(refused.value)
    assert "before it had started generating responses" in message
    assert "submit the study again" in message
    assert fake_slurm.calls("sbatch") == []


# --- sharded parents ----------------------------------------------------------------

def _sharded_run(mgr, tmp_path, base, *, states):
    """A parent with one shard per entry of ``states``: ``succeeded``,
    ``parked`` (running, kept responses), ``queued`` (never started), or
    ``unparked`` (started, saved no place). The whole run is then cancelled."""
    script = _dummy_script(tmp_path)
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    count = len(states)
    parent = mgr.record_external(
        "study-submit-bundle", status="pending", executor="slurm",
        requested_resources={"recordsDirectory": str(records),
                             "parallelJobs": count, "shardChildren": [],
                             "shardMerge": {"experiment": "demo", "verb": "run",
                                            "targetRoot": str(tmp_path)}})
    children = []
    for index, state in enumerate(states):
        slurm_id = str(base + index)
        child = mgr.record_external(
            "study-submit-bundle-shard",
            status="succeeded" if state == "succeeded" else
                   "submitted" if state == "queued" else "running",
            executor="slurm", executor_job_id=slurm_id,
            requested_resources={"scriptPath": script, "walltime": "00:10:00",
                                 "recordsDirectory": str(records),
                                 "shardIndex": index, "shardCount": count,
                                 "parentJob": parent.id},
            result={"recordsDirectory": str(records), "parentJob": parent.id,
                    "shard": {"index": index, "count": count},
                    **({"runDirectory": "runs/done"}
                       if state == "succeeded" else {})})
        if state in ("parked", "unparked"):
            run_dir = _park(tmp_path, records, child.id, slurm_id,
                            completed=2, shard=(index, count))
            if state == "unparked":
                os.remove(os.path.join(run_dir, resume.RESUME_STATE_FILENAME))
        children.append(child)
    parent.requested_resources = {**parent.requested_resources,
                                  "shardChildren": [c.id for c in children]}
    parent.status = "running"
    mgr.store.update(parent)
    assert mgr.cancel(parent.id) is True
    assert mgr.get(parent.id).status == "cancelled"
    return parent, children, script


def test_a_cancelled_sharded_run_resumes_every_cancelled_part(
        tmp_path, fake_slurm, monkeypatch):
    mgr = _manager(tmp_path)
    parent, children, script = _sharded_run(
        mgr, tmp_path, 9800, states=["succeeded", "parked", "queued"])
    _end_marker(script, "9801")                       # the parked shard
    fake_slurm.set_state("9802", "CANCELLED by 1000")  # the queued one never ran
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9810")

    outcome = mgr.resubmit(parent.id)

    assert outcome["ok"] is True and outcome["resumedAfterCancel"] is True
    assert outcome["resubmitOf"] == parent.id
    assert [entry["resubmitOf"] for entry in outcome["resumedShards"]] == [
        children[1].id, children[2].id]
    assert len(fake_slurm.calls("sbatch")) == 2
    # The parked shard continues from what it kept; the queued one starts.
    parked, queued = outcome["resumedShards"]
    assert parked["completedRecords"] == 2
    assert "starts from the beginning" in queued["message"]
    # Each cancelled shard stays cancelled and links to its continuation.
    for child, entry in zip(children[1:], outcome["resumedShards"]):
        original = mgr.get(child.id)
        assert original.status == "cancelled"
        assert original.result["resubmittedAs"] == entry["jobId"]
        continuation = mgr.get(entry["jobId"])
        assert continuation.cancelled is False
        assert continuation.requested_resources["resumedAfterCancel"] is True
    # The parent follows its parts again, with the history on the record.
    revived = mgr.get(parent.id)
    assert revived.status == "running"
    assert revived.cancelled is False
    assert revived.finished_at is None
    assert len(revived.result["resumedAfterCancel"]["resumed"]) == 2
    mgr._reconcile_shard_parents()
    assert mgr.get(parent.id).status in ("submitted", "running")
    # A second request finds nothing left to resume, and submits nothing.
    with pytest.raises(ResubmitRefused):
        mgr.resubmit(parent.id)
    assert len(fake_slurm.calls("sbatch")) == 2


def test_cancelling_a_resumed_sharded_run_stops_its_continuations(
        tmp_path, fake_slurm, monkeypatch):
    """Each resumed shard's own record stays cancelled, so stopping the fleet
    again must look past it to the continuation that is carrying the run."""
    mgr = _manager(tmp_path)
    parent, _children, script = _sharded_run(
        mgr, tmp_path, 9860, states=["parked", "parked"])
    _end_marker(script, "9860")
    _end_marker(script, "9861")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9870")
    outcome = mgr.resubmit(parent.id)
    continuations = [entry["jobId"] for entry in outcome["resumedShards"]]
    assert len(continuations) == 2
    cancels_before = len(fake_slurm.calls("scancel"))

    assert mgr.cancel(parent.id) is True

    assert mgr.get(parent.id).status == "cancelled"
    for continuation in continuations:
        assert mgr.get(continuation).status == "cancelled"
    assert len(fake_slurm.calls("scancel")) == cancels_before + 2
    # And the stopped run can be resumed once more.
    hint = cancel_resume_hint(mgr.get(parent.id))
    assert hint["offered"] is True


def test_a_sharded_run_waits_until_every_part_is_confirmed_ended(
        tmp_path, fake_slurm):
    mgr = _manager(tmp_path)
    parent, _children, script = _sharded_run(
        mgr, tmp_path, 9820, states=["parked", "parked"])
    _end_marker(script, "9820")
    fake_slurm.set_state("9821", "CANCELLED by 1000", queue="COMPLETING")
    with pytest.raises(ResubmitRefused) as refused:
        mgr.resubmit(parent.id)
    assert refused.value.wait is True
    assert "wait a minute, then try again" in str(refused.value)
    assert fake_slurm.calls("sbatch") == []        # nothing partial
    assert mgr.get(parent.id).status == "cancelled"


def test_a_sharded_run_with_a_part_that_saved_no_place_is_refused_whole(
        tmp_path, fake_slurm):
    mgr = _manager(tmp_path)
    parent, _children, script = _sharded_run(
        mgr, tmp_path, 9830, states=["parked", "unparked"])
    _end_marker(script, "9830")
    _end_marker(script, "9831", status="137")
    with pytest.raises(ResubmitRefused) as refused:
        mgr.resubmit(parent.id)
    assert refused.value.wait is False
    assert "submit the study again" in str(refused.value)
    assert fake_slurm.calls("sbatch") == []


def test_one_part_of_a_sharded_run_is_resumed_from_its_parent(
        tmp_path, fake_slurm):
    mgr = _manager(tmp_path)
    parent, children, script = _sharded_run(
        mgr, tmp_path, 9840, states=["parked", "parked"])
    _end_marker(script, "9840")
    with pytest.raises(ResubmitRefused,
                       match=f"resume the whole run from job {parent.id}"):
        mgr.resubmit(children[0].id)
    assert fake_slurm.calls("sbatch") == []


def test_a_run_cancelled_during_a_later_stage_is_refused_plainly(
        tmp_path, fake_slurm):
    mgr = _manager(tmp_path)
    parent, _children, _script = _sharded_run(
        mgr, tmp_path, 9850, states=["succeeded", "parked"])
    parent = mgr.get(parent.id)
    parent.requested_resources = {**parent.requested_resources,
                                  "continuationJob": "c0ffee000000"}
    parent.result = {**(parent.result or {}),
                     "mergedRunDirectory": "runs/exp-demo-run-merged"}
    mgr.store.update(parent)
    with pytest.raises(ResubmitRefused) as refused:
        mgr.resubmit(parent.id)
    message = str(refused.value)
    assert "judging or analysis" in message
    assert "runs/exp-demo-run-merged" in message
    assert "submit it again" in message
    assert fake_slurm.calls("sbatch") == []


# --- judging workers, later stages, and local jobs ----------------------------------

def test_a_cancelled_judging_worker_is_refused(tmp_path, fake_slurm):
    mgr = _manager(tmp_path)
    job, _script, _records, _run = _cancelled(
        mgr, tmp_path, "9901", kind="study-judge-worker", parked=False,
        extra_rr={"judgeWorker": {"model": "org/judge"}, "parentJob": "p"})
    with pytest.raises(ResubmitRefused, match="keeps no partial progress"):
        mgr.resubmit(job.id)
    assert cancel_resume_hint(mgr.get(job.id))["offered"] is False
    assert fake_slurm.calls("sbatch") == []


def test_a_cancelled_pipeline_continuation_is_refused(tmp_path, fake_slurm):
    mgr = _manager(tmp_path)
    job, _script, _records, _run = _cancelled(
        mgr, tmp_path, "9911", kind="study-submit-bundle-continuation",
        parked=False, extra_rr={"parentJob": "p"})
    with pytest.raises(ResubmitRefused, match="judging or analysis"):
        mgr.resubmit(job.id)
    assert fake_slurm.calls("sbatch") == []


def test_a_cancelled_local_job_says_it_cannot_be_resumed_yet(tmp_path,
                                                            fake_slurm):
    mgr = _manager(tmp_path)
    local = mgr.record_external("experiment:pipeline", status="cancelled",
                                executor="local")
    with pytest.raises(
            ResubmitRefused,
            match="this local run cannot be resumed yet; submit it again"):
        mgr.resubmit(local.id)
    hint = cancel_resume_hint(local)
    assert hint == {"offered": False,
                    "explanation": "This local run cannot be resumed yet; "
                                   "submit it again."}
    assert fake_slurm.calls("sbatch") == []


# --- what the record and the cancel answer tell a client ----------------------------

def test_the_job_record_says_whether_resume_is_offered(tmp_path, fake_slurm,
                                                      monkeypatch):
    mgr = _manager(tmp_path)
    running = mgr.record_external("study-submit", status="running",
                                  executor="slurm", executor_job_id="9951")
    assert "cancelResume" not in running.to_dict()
    # Not a study job: no hint at all, cancelled or not.
    other = mgr.record_external("model-install", status="cancelled",
                                executor="local")
    assert "cancelResume" not in other.to_dict()

    job, _script, _records, _run = _cancelled(mgr, tmp_path, "9952")
    hint = mgr.get(job.id).to_dict()["cancelResume"]
    assert hint["offered"] is True
    assert "responses it had already completed" in hint["explanation"]

    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9953")
    child_id = mgr.resubmit(job.id)["jobId"]
    hint = mgr.get(job.id).to_dict()["cancelResume"]
    assert hint["offered"] is False
    assert hint["continuation"] == child_id
    # The continuation is an ordinary job: nothing to say about a cancel.
    assert "cancelResume" not in mgr.get(child_id).to_dict()


def test_the_cancel_answer_says_responses_are_kept_and_how_to_go_on(
        tmp_path, fake_slurm):
    mgr = _manager(tmp_path)
    script = _dummy_script(tmp_path)
    job = mgr.record_external(
        "study-submit", status="running", executor="slurm",
        executor_job_id="9961", requested_resources={"scriptPath": script})
    assert mgr.cancel(job.id) is True
    answer = mgr.cancel_answer(job.id)
    assert answer["ok"] is True
    assert answer["cancelResume"]["offered"] is True
    assert "already completed are kept" in answer["message"]
    assert f"resume job {job.id}" in answer["message"]
    assert "nothing resumes it automatically" in answer["message"]
    assert any("a person can resume it" in line
               for line in mgr.get(job.id).all_logs())
    # A job that is not a study run answers the bare acknowledgement.
    other = mgr.record_external("slurm-submit", status="running",
                                executor="slurm", executor_job_id="9962")
    assert mgr.cancel(other.id) is True
    assert mgr.cancel_answer(other.id) == {"ok": True}


def test_the_routes_carry_the_cancel_answer_and_the_resume(
        tmp_path, fake_slurm, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "root"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "route-jobs.sqlite"))
    state = ServiceState()
    app = FastAPI()
    app.include_router(build_router(state))
    client = TestClient(app)

    script = _dummy_script(tmp_path)
    records = tmp_path / "records"
    records.mkdir()
    job = state.jobs.record_external(
        "study-submit", status="running", executor="slurm",
        executor_job_id="9971",
        requested_resources={"scriptPath": script, "walltime": "00:10:00",
                             "recordsDirectory": str(records)})
    _park(tmp_path, records, job.id, "9971", completed=4)

    cancelled = client.post(f"/api/jobs/{job.id}/cancel")
    assert cancelled.status_code == 200
    body = cancelled.json()
    assert body["ok"] is True
    assert body["cancelResume"]["offered"] is True
    assert "already completed are kept" in body["message"]
    listed = client.get(f"/api/jobs/{job.id}").json()
    assert listed["status"] == "cancelled"
    assert listed["cancelResume"]["offered"] is True

    # Not yet confirmed ended: a 409 whose detail is the plain reason.
    early = client.post(f"/api/jobs/{job.id}/resubmit")
    assert early.status_code == 409
    assert "wait a minute, then try again" in early.json()["detail"]

    _end_marker(script, "9971")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9972")
    resumed = client.post(f"/api/jobs/{job.id}/resubmit")
    assert resumed.status_code == 200
    answer = resumed.json()
    assert answer["resumedAfterCancel"] is True
    assert answer["completedRecords"] == 4
    assert answer["slurmJobID"] == "9972"
    assert client.get(f"/api/jobs/{job.id}").json()["status"] == "cancelled"
