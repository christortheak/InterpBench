"""Issue #12: a split run that meets the cluster's job limit HOLDS its
remaining parts instead of cancelling the parts already accepted.

Driven through the real fan-out (``submit_run_bundle --parallel``) and the
real reconciler tick (``JobManager.poll_slurm``), with the committed fake
scheduler binaries on PATH. The fake sbatch refuses from a chosen call on
with a chosen message (``FAKE_SBATCH_FAIL_AFTER`` + ``FAKE_SBATCH_FAIL_MESSAGE``)
and numbers its job ids by call (``FAKE_SLURM_JOB_ID_FROM_CALL``). Nothing
contacts a real scheduler.
"""

import json
import os

import pytest

from steerlab_server.api import executors
from steerlab_server.api.jobs import HELD_PART_NOTE, ResubmitRefused

from test_sharding import _manager, _staged_bundle, fake_slurm  # noqa: F401

#: What a real Slurm sbatch prints when the per-user submit limit is met:
#: the reason code on one line, the general sentence on the next.
LIMIT = ("sbatch: error: QOSMaxSubmitJobPerUserLimit\\n"
         "sbatch: error: Batch job submission failed: Job violates "
         "accounting/QOS policy (job submit limit, user's size and/or time "
         "limits)")
#: The general sentence alone, with no reason code.
POLICY = ("sbatch: error: Batch job submission failed: Job violates "
          "accounting/QOS policy (job submit limit, user's size and/or time "
          "limits)")


# --- the vocabulary ---------------------------------------------------------

@pytest.mark.parametrize("message", [
    "sbatch: error: QOSMaxSubmitJobPerUserLimit",
    "sbatch: error: AssocMaxSubmitJobLimit",
    "sbatch: error: QOSMaxSubmitJobPerAccountLimit",
    "sbatch: error: QOSGrpSubmitJobsLimit",
    "sbatch: error: AssocGrpSubmitJobsLimit",
    "SBATCH: ERROR: qosmaxsubmitjobperuserlimit",
])
def test_the_submit_limit_reason_codes_are_job_limits(message):
    assert executors.submit_refusal_kind(message) == "jobLimit"
    assert executors.is_job_limit_refusal(message, same_request_accepted=False)


def test_the_general_policy_sentence_is_a_job_limit_only_after_an_acceptance():
    """On its own the sentence may name a per-job size or time limit, which
    never frees; once an identical part was accepted, only a count can
    refuse the next one."""
    assert executors.submit_refusal_kind(POLICY) == "policy"
    assert not executors.is_job_limit_refusal(POLICY,
                                              same_request_accepted=False)
    assert executors.is_job_limit_refusal(POLICY, same_request_accepted=True)


@pytest.mark.parametrize("message", [
    "sbatch: error: fake quota exceeded",
    "sbatch: error: Batch job submission failed: Invalid partition name specified",
    "sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified",
    "",
    None,
])
def test_other_refusals_are_not_job_limits(message):
    assert executors.submit_refusal_kind(message) in (None,)
    assert not executors.is_job_limit_refusal(message,
                                              same_request_accepted=True)


@pytest.mark.parametrize("message", [
    "sbatch: error: Batch job submission failed: Unable to contact slurm controller (connect failure)",
    "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation",
    "requested walltime crosses a configured maintenance window",
])
def test_not_now_refusals_say_retry_later(message):
    assert executors.submit_refusal_kind(message) == "retryLater"
    assert not executors.is_job_limit_refusal(message,
                                              same_request_accepted=True)


def test_a_refused_sbatch_raises_the_typed_error(tmp_path, monkeypatch,
                                                 fake_slurm):
    monkeypatch.setenv("FAKE_SBATCH_FAIL", "sbatch: error: QOSMaxSubmitJobPerUserLimit")
    script = tmp_path / "run.sbatch"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    bundle = executors.JobBundle(
        bundle_dir=str(tmp_path), command=[], env={},
        resources=executors.SlurmResources(), stdout_path="", stderr_path="",
        script_path=str(script), manifest_path="")
    with pytest.raises(executors.SchedulerSubmitError) as raised:
        executors.SlurmExecutor().submit(bundle)
    assert isinstance(raised.value, RuntimeError)
    assert raised.value.answered is True
    assert "QOSMaxSubmitJobPerUserLimit" in str(raised.value)


# --- the fan-out --------------------------------------------------------------

def _limited(monkeypatch, *, accepted: int, message: str = LIMIT) -> None:
    monkeypatch.setenv("FAKE_SBATCH_FAIL_AFTER", str(accepted))
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE", message)


def _slot_frees(monkeypatch) -> None:
    monkeypatch.delenv("FAKE_SBATCH_FAIL_AFTER", raising=False)
    monkeypatch.delenv("FAKE_SBATCH_FAIL_MESSAGE", raising=False)


def _submit(tmp_path, monkeypatch, *, name, parallel=4):
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "target"))
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "7000")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID_FROM_CALL", "1")
    os.makedirs(tmp_path / "target", exist_ok=True)
    from steerlab_server.api.submissions import submit_run_bundle
    bundle_path = _staged_bundle(tmp_path, name=name)
    jobs = _manager(tmp_path)
    submission = submit_run_bundle(
        bundle_path, verb="run", jobs=jobs, executor="slurm",
        target_root=str(tmp_path / "target"),
        resources={"gres": "A100", "walltime": "00:30:00"},
        parallel_jobs=parallel)
    return jobs, submission


def _parts(jobs, submission):
    return [jobs.get(cid) for cid in submission.shard_job_ids]


def _tick(jobs):
    """One reconciler tick, with the retry pause behind it."""
    jobs._held_retry_after = 0.0
    return jobs.poll_slurm()


def test_a_job_limit_holds_the_remaining_parts_and_keeps_the_accepted_ones(
        tmp_path, monkeypatch, fake_slurm):
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldfan")

    # Nothing was cancelled: the two accepted parts keep their slots.
    assert fake_slurm.calls("scancel") == []
    parent = jobs.get(submission.job_id)
    assert parent.status == "submitted"
    assert parent.error is None
    parts = _parts(jobs, submission)
    assert [p.status for p in parts] == ["submitted", "submitted",
                                         "pending", "pending"]
    assert [p.executor_job_id for p in parts] == ["7001", "7002", None, None]
    for part in parts[2:]:
        held = part.result["held"]
        assert held["reason"] == "clusterJobLimit"
        assert "QOSMaxSubmitJobPerUserLimit" in held["schedulerMessage"]
        assert held["since"] > 0
        assert jobs.is_held(part)
    # The part after the refused one was never sent: 2 accepted + 1 refused.
    assert len(fake_slurm.calls("sbatch")) == 3

    shards_held = submission.to_dict()["shardsHeld"]
    assert shards_held["count"] == 2
    assert shards_held["partCount"] == 4
    assert shards_held["submittedCount"] == 2
    assert shards_held["jobIDs"] == [parts[2].id, parts[3].id]
    assert shards_held["message"].startswith(
        "2 of 4 parts were submitted now; 2 are held by SteerLab because "
        "the cluster's job limit is reached.")
    assert any("held by SteerLab" in line for line in parent.all_logs())


def test_a_limit_at_the_first_part_holds_every_part(tmp_path, monkeypatch,
                                                    fake_slurm):
    _limited(monkeypatch, accepted=0)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldall",
                               parallel=3)
    assert jobs.get(submission.job_id).status == "submitted"
    assert [p.status for p in _parts(jobs, submission)] == ["pending"] * 3
    assert submission.to_dict()["shardsHeld"]["submittedCount"] == 0
    assert len(fake_slurm.calls("sbatch")) == 1


def test_the_general_policy_sentence_holds_only_after_an_accepted_part(
        tmp_path, monkeypatch, fake_slurm):
    from steerlab_server.api.submissions import ShardFanoutFailed
    _limited(monkeypatch, accepted=1, message=POLICY)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldpolicy",
                               parallel=3)
    assert [p.status for p in _parts(jobs, submission)] == [
        "submitted", "pending", "pending"]

    # At the first part, the same sentence may be a per-job limit that never
    # frees: the submission is refused, as before, and nothing was accepted.
    _slot_frees(monkeypatch)
    _limited(monkeypatch, accepted=len(fake_slurm.calls("sbatch")),
             message=POLICY)
    with pytest.raises(ShardFanoutFailed):
        _submit(tmp_path / "second", monkeypatch, name="heldpolicy2",
                parallel=3)


def test_held_parts_are_submitted_in_order_as_slots_free(
        tmp_path, monkeypatch, fake_slurm):
    _limited(monkeypatch, accepted=1)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldorder")
    parts = _parts(jobs, submission)
    assert len(fake_slurm.calls("sbatch")) == 2

    # Still full: the first held part is asked about, the later ones wait
    # their turn, and nothing is cancelled.
    _tick(jobs)
    assert len(fake_slurm.calls("sbatch")) == 3
    assert all(jobs.is_held(p) for p in _parts(jobs, submission)[1:])
    assert fake_slurm.calls("scancel") == []
    # Within the retry pause the scheduler is not asked again.
    jobs.poll_slurm()
    assert len(fake_slurm.calls("sbatch")) == 3

    # One slot frees: exactly the next part in order goes, and the rest wait.
    _limited(monkeypatch, accepted=4)
    _tick(jobs)
    states = [p.status for p in _parts(jobs, submission)]
    assert states == ["submitted", "submitted", "pending", "pending"]
    second = jobs.get(parts[1].id)
    assert second.executor_job_id == "7004"
    assert second.result["heldBeforeSubmit"]["reason"] == "clusterJobLimit"
    assert "held" not in second.result
    calls = fake_slurm.calls("sbatch")
    assert f"--job-name=slhd-{parts[1].id}-" in calls[3]
    assert second.requested_resources["submissionToken"].startswith(
        f"slhd-{parts[1].id}-")

    # Every slot frees: the rest go, in order.
    _slot_frees(monkeypatch)
    _tick(jobs)
    final = _parts(jobs, submission)
    assert [p.status for p in final] == ["submitted"] * 4
    assert [p.executor_job_id for p in final] == ["7001", "7004", "7006",
                                                  "7007"]
    assert fake_slurm.calls("scancel") == []
    assert jobs.get(submission.job_id).status == "submitted"


def test_a_non_limit_refusal_at_submission_cancels_fails_and_raises(
        tmp_path, monkeypatch, fake_slurm):
    from steerlab_server.api.submissions import ShardFanoutFailed
    _limited(monkeypatch, accepted=2,
             message="sbatch: error: Invalid partition name specified")
    with pytest.raises(ShardFanoutFailed) as raised:
        _submit(tmp_path, monkeypatch, name="heldother")
    submission = raised.value.submission
    assert len(fake_slurm.calls("scancel")) == 2
    from steerlab_server.api.jobs import DurableJobStore, JobManager
    jobs = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")),
                      capability_provider=lambda: {})
    parent = jobs.get(submission.job_id)
    assert parent.status == "failed"
    assert "Invalid partition name specified" in str(raised.value)
    assert all(jobs.get(cid).status == "cancelled"
               for cid in parent.requested_resources["shardChildren"])


def test_a_held_part_refused_later_for_another_reason_cancels_and_fails(
        tmp_path, monkeypatch, fake_slurm):
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldlater")
    monkeypatch.setenv("FAKE_SBATCH_FAIL",
                       "sbatch: error: Invalid account or account/partition "
                       "combination specified")
    _tick(jobs)
    parent = jobs.get(submission.job_id)
    assert parent.status == "failed"
    assert "part 3 of 4" in parent.error
    assert "Invalid account" in parent.error
    assert "submit the study again" in parent.error
    parts = _parts(jobs, submission)
    assert [p.status for p in parts] == ["cancelled"] * 4
    # The accepted parts were stopped through the scheduler; the held ones
    # never reached it.
    assert sorted(fake_slurm.calls("scancel")) == ["7001", "7002"]
    for part in parts[2:]:
        assert part.executor_job_id is None
        assert "heldWhenCancelled" in part.result


def test_a_scheduler_that_cannot_be_reached_keeps_parts_held(
        tmp_path, monkeypatch, fake_slurm):
    """Cancelling parts that have run for hours because the scheduler was
    briefly unreachable would throw away work that waiting keeps."""
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldflaky")
    monkeypatch.setenv("FAKE_SBATCH_FAIL",
                       "sbatch: error: Batch job submission failed: Unable "
                       "to contact slurm controller (connect failure)")
    _tick(jobs)
    assert jobs.get(submission.job_id).status == "submitted"
    assert fake_slurm.calls("scancel") == []
    held = [p for p in _parts(jobs, submission) if jobs.is_held(p)]
    assert len(held) == 2
    # The attempt stays on record: the next one asks whether it landed.
    assert held[0].result["held"]["attempt"]["submissionName"].startswith(
        f"slhd-{held[0].id}-")


# --- lifecycle ----------------------------------------------------------------

def test_cancelling_the_parent_withdraws_its_held_parts(tmp_path, monkeypatch,
                                                        fake_slurm):
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldcancel")
    assert jobs.cancel(submission.job_id) is True
    parent = jobs.get(submission.job_id)
    assert parent.status == "cancelled"
    parts = _parts(jobs, submission)
    assert [p.status for p in parts] == ["cancelled"] * 4
    assert sorted(fake_slurm.calls("scancel")) == ["7001", "7002"]
    for part in parts[2:]:
        assert not jobs.is_held(part)
        assert part.result["heldWhenCancelled"]["reason"] == "clusterJobLimit"
        assert part.cancelled
        assert any("never reached the scheduler" in line
                   for line in part.all_logs())
        # Its row points at the run's Resume, not at "submit it again".
        from steerlab_server.api.jobs import cancel_resume_hint
        hint = cancel_resume_hint(part)
        assert hint["offered"] is False
        assert f"Resume the whole run from job {submission.job_id}" in \
            hint["explanation"]

    # A slot frees: nothing is submitted for the cancelled run.
    _slot_frees(monkeypatch)
    calls = len(fake_slurm.calls("sbatch"))
    _tick(jobs)
    assert len(fake_slurm.calls("sbatch")) == calls


def test_resuming_a_cancelled_run_holds_its_never_submitted_parts_again(
        tmp_path, monkeypatch, fake_slurm):
    """Resume on a cancelled split run continues every part: a part that was
    held when the run was cancelled never reached the scheduler, so it is
    held again (a new record) and the controller submits it."""
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldresume")
    assert jobs.cancel(submission.job_id) is True
    for slurm_id in ("7001", "7002"):
        fake_slurm.set_state(slurm_id, "CANCELLED", exit_code="0:15")
    _slot_frees(monkeypatch)

    answer = jobs.resubmit(submission.job_id)
    assert answer["ok"] is True
    entries = {entry["resubmitOf"]: entry for entry in answer["resumedShards"]}
    originals = _parts(jobs, submission)
    assert set(entries) == {part.id for part in originals}
    for part in originals[2:]:
        entry = entries[part.id]
        assert entry["held"] is True
        continuation = jobs.get(entry["jobId"])
        assert jobs.is_held(continuation)
        assert continuation.requested_resources["resubmitOf"] == part.id
        assert continuation.requested_resources["shardIndex"] == \
            part.requested_resources["shardIndex"]
        record = jobs.get(part.id)
        assert record.status == "cancelled"
        assert record.result["resubmittedAs"] == continuation.id
        assert record.result["resumedAfterCancel"]["continuation"] == \
            continuation.id
    parent = jobs.get(submission.job_id)
    assert parent.status == "running"

    # The next tick submits the held continuations, in part order.
    calls = len(fake_slurm.calls("sbatch"))
    _tick(jobs)
    new_calls = fake_slurm.calls("sbatch")[calls:]
    assert [call.split()[-1].split("/")[-2] for call in new_calls] == [
        "slurm-shard-2", "slurm-shard-3"]
    for part in originals[2:]:
        assert jobs.get(entries[part.id]["jobId"]).status == "submitted"


def test_a_run_that_fails_withdraws_its_held_parts(tmp_path, monkeypatch,
                                                   fake_slurm):
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldfail")
    fake_slurm.set_state("7001", "FAILED", exit_code="1:0")
    _slot_frees(monkeypatch)
    calls = len(fake_slurm.calls("sbatch"))
    _tick(jobs)
    assert jobs.get(submission.job_id).status == "failed"
    parts = _parts(jobs, submission)
    assert [p.status for p in parts[2:]] == ["cancelled", "cancelled"]
    assert len(fake_slurm.calls("sbatch")) == calls


def test_held_parts_survive_a_controller_restart(tmp_path, monkeypatch,
                                                 fake_slurm):
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldrestart")
    held_ids = [p.id for p in _parts(jobs, submission)[2:]]

    # The controller stops and another starts on the same store, and the
    # restart sweep runs (the old owner has exited).
    monkeypatch.setattr("steerlab_server.api.job_ownership.process_state",
                        lambda host, pid: "exited")
    restarted = _manager(tmp_path)
    for part_id in held_ids:
        part = restarted.get(part_id)
        assert restarted.is_held(part)
        assert part.result["held"]["reason"] == "clusterJobLimit"
    assert restarted.get(submission.job_id).status == "submitted"

    _slot_frees(monkeypatch)
    restarted.poll_slurm()   # a fresh controller asks at once
    assert [restarted.get(i).status for i in held_ids] == ["submitted"] * 2
    assert fake_slurm.calls("scancel") == []


def test_an_attempt_that_landed_unrecorded_is_adopted_never_resubmitted(
        tmp_path, monkeypatch, fake_slurm):
    """At most once: a controller that stopped between sbatch accepting a
    held part and recording it finds the job by its submission name."""
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldadopt")
    part = _parts(jobs, submission)[2]
    token = f"slhd-{part.id}-deadbeef"
    part.result = {**part.result, "held": {**part.result["held"],
                                           "attempt": {"submissionName": token,
                                                       "at": 1.0}}}
    jobs.store.update(part)
    monkeypatch.setenv("FAKE_SQUEUE_NAMED", "8800")
    calls = len(fake_slurm.calls("sbatch"))
    _tick(jobs)
    adopted = jobs.get(part.id)
    assert adopted.executor_job_id == "8800"
    assert adopted.status == "submitted"
    assert any(token in line for line in fake_slurm.calls("squeue"))
    # Part 3 was not submitted a second time: the one new sbatch call is
    # part 4's turn, which the still-full queue refused.
    new_calls = fake_slurm.calls("sbatch")[calls:]
    assert len(new_calls) == 1
    assert new_calls[0].endswith("slurm-shard-3/run.sbatch")
    assert jobs.is_held(_parts(jobs, submission)[3])


def test_an_unconfirmed_attempt_waits_rather_than_risking_a_second_job(
        tmp_path, monkeypatch, fake_slurm):
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldunsure")
    part = _parts(jobs, submission)[2]
    part.result = {**part.result, "held": {**part.result["held"],
                                           "attempt": {"submissionName":
                                                       f"slhd-{part.id}-0",
                                                       "at": 1.0}}}
    jobs.store.update(part)
    monkeypatch.setenv("FAKE_SACCT_FAIL", "slurmdbd unreachable")
    _slot_frees(monkeypatch)
    calls = len(fake_slurm.calls("sbatch"))
    _tick(jobs)
    assert len(fake_slurm.calls("sbatch")) == calls
    assert jobs.is_held(jobs.get(part.id))

    # The scheduler answers that the attempt never landed: it is submitted.
    monkeypatch.delenv("FAKE_SACCT_FAIL")
    _tick(jobs)
    assert jobs.get(part.id).status == "submitted"


def test_a_timed_out_sbatch_keeps_the_attempt_for_the_name_search(
        tmp_path, monkeypatch, fake_slurm):
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldtimeout")
    real_submit = executors.SlurmExecutor.submit

    def timing_out(self, bundle, **kwargs):
        raise executors.SchedulerSubmitError(
            "sbatch did not answer within 120s and was abandoned",
            returncode=executors.SCHEDULER_TIMEOUT_EXIT)

    monkeypatch.setattr(executors.SlurmExecutor, "submit", timing_out)
    _tick(jobs)
    part = jobs.get(_parts(jobs, submission)[2].id)
    assert jobs.is_held(part)
    token = part.result["held"]["attempt"]["submissionName"]

    # Next tick: the scheduler knows the job by that name, so it is adopted.
    monkeypatch.setattr(executors.SlurmExecutor, "submit", real_submit)
    monkeypatch.setenv("FAKE_SQUEUE_NAMED", "8811")
    _tick(jobs)
    assert jobs.get(part.id).executor_job_id == "8811"
    assert any(token in line for line in fake_slurm.calls("squeue"))


def test_resubmit_never_submits_a_held_part(tmp_path, monkeypatch, fake_slurm):
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldresub")
    part = _parts(jobs, submission)[2]
    calls = len(fake_slurm.calls("sbatch"))
    with pytest.raises(ResubmitRefused, match="held by SteerLab"):
        jobs.resubmit(part.id)
    # Resuming the parent touches only checkpointed parts.
    with pytest.raises(ResubmitRefused):
        jobs.resubmit(submission.job_id)
    assert len(fake_slurm.calls("sbatch")) == calls
    assert jobs.is_held(jobs.get(part.id))


def test_the_merge_waits_for_held_parts(tmp_path, monkeypatch, fake_slurm):
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldmerge")
    merges = []
    monkeypatch.setattr(jobs, "_merge_shard_parent",
                        lambda parent, children, run_dirs: merges.append(
                            parent.id) or True)
    for part in _parts(jobs, submission)[:2]:
        part.status = "succeeded"
        part.result = {**part.result, "runDirectory": str(tmp_path / part.id)}
        jobs.store.update(part)
    jobs.run_merge_pass()
    assert merges == []
    assert jobs.get(submission.job_id).status == "submitted"

    # The held parts run and succeed: only now is the run merged.
    for part in _parts(jobs, submission)[2:]:
        part.status = "succeeded"
        part.executor_job_id = "1"
        part.result = {k: v for k, v in part.result.items() if k != "held"}
        part.result["runDirectory"] = str(tmp_path / part.id)
        jobs.store.update(part)
    jobs.run_merge_pass()
    assert merges == [submission.job_id]


# --- what a status read says ---------------------------------------------------

def test_the_status_document_names_held_parts_in_plain_words(
        tmp_path, monkeypatch, fake_slurm):
    from steerlab_server.api.job_status import job_status
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldstatus")
    document = job_status(jobs, jobs.get(submission.job_id))
    shards = document["shards"]
    assert [s["held"] is not None for s in shards] == [False, False, True,
                                                       True]
    held = shards[2]
    assert held["status"] == "pending"
    assert held["heldNote"] == HELD_PART_NOTE == (
        "held by SteerLab: the cluster's job limit is reached; it is "
        "submitted when a slot frees")
    assert held["held"]["reason"] == "clusterJobLimit"
    # The generic "no scheduler id yet" note would mislead here.
    assert not any("no scheduler job id" in n for n in held["notes"])
    assert shards[0]["heldNote"] is None
    assert any(note.startswith("2 of 4 parts are held by SteerLab")
               and "while the controller is not running they wait" in note
               for note in document["notes"])
    json.dumps(document)   # the document stays JSON


def test_the_submit_route_answers_a_failed_fanout_as_an_error(
        tmp_path, monkeypatch, fake_slurm):
    """Before, the route answered 200 with the failed parent's submission,
    and both clients exited 0."""
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "target"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "route.sqlite"))
    os.makedirs(tmp_path / "target", exist_ok=True)
    bundle_path = _staged_bundle(tmp_path, name="heldroute")
    state = ServiceState()
    app = FastAPI()
    app.include_router(build_router(state))
    client = TestClient(app)
    body = {"bundlePath": bundle_path, "verb": "run", "executor": "slurm",
            "parallelJobs": 3, "targetRoot": str(tmp_path / "target"),
            "resources": {"gres": "A100", "walltime": "00:30:00"}}

    _limited(monkeypatch, accepted=1,
             message="sbatch: error: Invalid partition name specified")
    response = client.post("/api/studies/submit-bundle", json=body)
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "the split run was not started" in detail
    assert "Invalid partition name specified" in detail

    # A job limit is a success with held parts.
    _slot_frees(monkeypatch)
    _limited(monkeypatch, accepted=len(fake_slurm.calls("sbatch")) + 1)
    response = client.post("/api/studies/submit-bundle", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["shardsHeld"]["count"] == 2
    client.close()
