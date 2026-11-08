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
])
def test_not_now_refusals_say_retry_later(message):
    assert executors.submit_refusal_kind(message) == "retryLater"
    assert not executors.is_job_limit_refusal(message,
                                              same_request_accepted=True)


def test_a_maintenance_window_is_its_own_kind():
    message = "requested walltime crosses a configured maintenance window"
    assert executors.submit_refusal_kind(message) == "maintenanceWindow"
    assert not executors.is_job_limit_refusal(message,
                                              same_request_accepted=True)


#: What sbatch prints for a walltime above the QOS maximum: a reason code
#: that is not a job count, then the general sentence.
MAX_WALL = ("sbatch: error: QOSMaxWallDurationPerJobLimit\\n"
            "sbatch: error: Batch job submission failed: Job violates "
            "accounting/QOS policy (job submit limit, user's size and/or time "
            "limits)")


@pytest.mark.parametrize("code", [
    "QOSMaxWallDurationPerJobLimit", "AssocMaxWallDurationPerJobLimit",
    "PartitionTimeLimit", "QOSMaxGRESPerJob", "AssocGrpGRES",
    "QOSMaxCpuPerJobLimit",
])
def test_any_other_limit_code_never_waits(code):
    """Review round 2: the general sentence beside a limit code that is not
    a job count names a limit on the job itself, which never frees."""
    message = (f"sbatch: error: {code}\nsbatch: error: Batch job submission "
               "failed: Job violates accounting/QOS policy (job submit "
               "limit, user's size and/or time limits)")
    assert executors.submit_refusal_kind(message) == "otherLimit"
    assert executors.other_limit_codes(message) == [code]
    assert not executors.is_job_limit_refusal(message,
                                              same_request_accepted=True)
    # A submit-limit code beside it does not make it wait either.
    assert executors.submit_refusal_kind(
        message + "\nsbatch: error: QOSMaxSubmitJobPerUserLimit") == "otherLimit"


def test_a_permanent_refusal_names_its_repair():
    assert executors.refusal_repair(MAX_WALL.replace("\\n", "\n")) == (
        "request a walltime within the cluster's limit for one job "
        "(--walltime), then submit the study again")
    assert executors.refusal_repair("sbatch: error: QOSMaxGRESPerJob") == (
        "request resources within the cluster's limit for one job, then "
        "submit the study again")
    assert executors.refusal_repair("sbatch: error: Invalid account") == (
        "correct what the scheduler said, then submit the study again")


#: Review round 4: the classification table. A count of jobs (submitted, or
#: running at once) frees as jobs leave, so it waits; a size, time, or
#: resource limit never frees, so it is permanent.
_GENERAL = ("\nsbatch: error: Batch job submission failed: Job violates "
            "accounting/QOS policy (job submit limit, user's size and/or time "
            "limits)")


@pytest.mark.parametrize("code", [
    "QOSMaxSubmitJobPerUserLimit", "QOSMaxSubmitJobPerAccountLimit",
    "QOSGrpSubmitJobsLimit", "AssocMaxSubmitJobLimit",
    "AssocGrpSubmitJobsLimit",
    "QOSMaxJobsPerUserLimit", "QOSMaxJobsPerAccountLimit",
    "QOSMaxJobsPerAccount", "QOSGrpJobsLimit", "QOSJobLimit",
    "AssocMaxJobsLimit", "AssocGrpJobsLimit", "AssocJobLimit",
])
def test_every_job_count_code_waits(code):
    message = f"sbatch: error: {code}" + _GENERAL
    assert executors.submit_refusal_kind(message) == "jobLimit", code
    assert executors.other_limit_codes(message) == [], code
    assert executors.is_job_limit_refusal(message,
                                          same_request_accepted=False), code


@pytest.mark.parametrize("code", [
    "QOSMaxWallDurationPerJobLimit", "AssocMaxWallDurationPerJobLimit",
    "PartitionTimeLimit", "QOSMaxGRESPerJob", "AssocGrpGRES",
    "QOSMaxCpuPerJobLimit", "AssocGrpCPURunMinutesLimit", "QOSGrpCpuLimit",
    "QOSMaxMemoryPerJob", "AssocGrpMemLimit", "QOSMaxNodePerJobLimit",
    "PartitionNodeLimit", "QOSMaxTRESPerJob",
])
def test_every_size_time_or_resource_code_is_permanent(code):
    message = f"sbatch: error: {code}" + _GENERAL
    assert executors.submit_refusal_kind(message) == "otherLimit", code
    assert not executors.is_job_limit_refusal(message,
                                              same_request_accepted=True), code


@pytest.mark.parametrize("code, repair", [
    ("AssocGrpCPURunMinutesLimit", "resources"),
    ("QOSMaxCpuPerJobLimit", "resources"),
    ("QOSGrpCPULimit", "resources"),
    ("QOSMaxMemoryPerJob", "resources"),
    ("PartitionNodeLimit", "resources"),
    ("QOSMaxTRESPerJob", "resources"),
    ("QOSMaxWallDurationPerJobLimit", "walltime"),
    ("PartitionTimeLimit", "walltime"),
])
def test_the_repair_reads_the_code_whatever_its_case(code, repair):
    """``CPU`` and ``Cpu`` both name a CPU limit (review round 4)."""
    expected = {
        "resources": ("request resources within the cluster's limit for one "
                      "job, then submit the study again"),
        "walltime": ("request a walltime within the cluster's limit for one "
                     "job (--walltime), then submit the study again"),
    }[repair]
    assert executors.refusal_repair(f"sbatch: error: {code}" + _GENERAL) == \
        expected


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


def _submit_into(jobs, tmp_path, *, name, parallel=2):
    """A second split run on the same manager (``_submit`` sets the env)."""
    from steerlab_server.api.submissions import submit_run_bundle
    bundle_path = _staged_bundle(tmp_path / name, name=name)
    return submit_run_bundle(
        bundle_path, verb="run", jobs=jobs, executor="slurm",
        target_root=str(tmp_path / "target"),
        resources={"gres": "A100", "walltime": "00:30:00"},
        parallel_jobs=parallel)


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


def test_a_running_job_count_refusal_keeps_parts_held(tmp_path, monkeypatch,
                                                      fake_slurm):
    """Review round 4: a count of jobs running at once frees as they end;
    the run waits, never fails."""
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldrunning")
    monkeypatch.setenv("FAKE_SBATCH_FAIL",
                       "sbatch: error: QOSMaxJobsPerUserLimit" + _GENERAL)
    _tick(jobs)
    assert jobs.get(submission.job_id).status == "submitted"
    assert fake_slurm.calls("scancel") == []
    assert [jobs.is_held(p) for p in _parts(jobs, submission)] == [
        False, False, True, True]


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


# --- review round 2: refusals that never free ---------------------------------

def _scheduler_entry(job_id: str, **entry) -> None:
    path = os.environ["FAKE_SLURM_STATE_FILE"]
    table = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            table = json.load(handle)
    table[job_id] = entry
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(table, handle)


def _override_held_walltime(jobs, submission, walltime):
    for part in _parts(jobs, submission):
        if jobs.is_held(part):
            part.requested_resources = {**part.requested_resources,
                                        "walltimeOverride": walltime}
            jobs.store.update(part)


def test_a_fanout_refused_for_another_limit_is_not_held(tmp_path, monkeypatch,
                                                        fake_slurm):
    from steerlab_server.api.submissions import ShardFanoutFailed
    _limited(monkeypatch, accepted=1, message=MAX_WALL)
    with pytest.raises(ShardFanoutFailed) as raised:
        _submit(tmp_path, monkeypatch, name="heldmaxwallfan", parallel=3)
    assert "QOSMaxWallDurationPerJobLimit" in str(raised.value)


def test_a_walltime_above_the_maximum_fails_its_run_and_never_stalls_the_next(
        tmp_path, monkeypatch, fake_slurm):
    """Review round 2: the general sentence beside a per-job limit code was
    read as "not now", so the part waited for ever and every later run's
    held parts waited behind it."""
    _limited(monkeypatch, accepted=1)
    jobs, first = _submit(tmp_path, monkeypatch, name="heldwalla", parallel=2)
    second = _submit_into(jobs, tmp_path, name="heldwallb", parallel=2)
    assert [p.status for p in _parts(jobs, first)] == ["submitted", "pending"]
    assert [p.status for p in _parts(jobs, second)] == ["pending", "pending"]
    _override_held_walltime(jobs, first, "48:00:00")

    _slot_frees(monkeypatch)
    monkeypatch.setenv("FAKE_SBATCH_FAIL_IF_ARG", "--time=48:00:00")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE", MAX_WALL)
    _tick(jobs)

    parent = jobs.get(first.job_id)
    assert parent.status == "failed"
    assert "QOSMaxWallDurationPerJobLimit" in parent.error
    assert parent.error.endswith(
        "Request a walltime within the cluster's limit for one job "
        "(--walltime), then submit the study again")
    assert [p.status for p in _parts(jobs, first)] == ["cancelled",
                                                       "cancelled"]
    # The next run was asked about on the same tick, and went.
    assert [p.status for p in _parts(jobs, second)] == ["submitted",
                                                        "submitted"]
    assert jobs.get(second.job_id).status == "submitted"


def _cancelled_run_whose_ran_parts_finished(tmp_path, monkeypatch, name):
    """A split run with two finished parts and two held ones, cancelled:
    Resume continues only the held parts."""
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name=name)
    for part in _parts(jobs, submission)[:2]:
        part.status = "succeeded"
        part.finished_at = 1.0
        part.result = {**part.result, "runDirectory": str(tmp_path / part.id)}
        jobs.store.update(part)
    assert jobs.cancel(submission.job_id) is True
    assert jobs.get(submission.job_id).status == "cancelled"
    _slot_frees(monkeypatch)
    return jobs, submission


def _held_tails(jobs, submission):
    return [jobs._shard_chain_tail(part) for part in _parts(jobs, submission)[2:]]


def test_resuming_with_a_walltime_above_the_maximum_is_refused_at_once(
        tmp_path, monkeypatch, fake_slurm):
    """Review rounds 2 and 3: a Resume with a --walltime above the QOS
    maximum, where only never-submitted parts continue. The first of them is
    handed to the scheduler in the request itself, and its refusal refuses
    the Resume there, with the repair, instead of failing the run on the
    next tick. The run stays cancelled and can be resumed again."""
    jobs, submission = _cancelled_run_whose_ran_parts_finished(
        tmp_path, monkeypatch, "heldresumewall")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_IF_ARG", "--time=48:00:00")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE", MAX_WALL)
    with pytest.raises(ResubmitRefused) as refused:
        jobs.resubmit(submission.job_id, walltime="48:00:00")
    message = str(refused.value)
    assert "QOSMaxWallDurationPerJobLimit" in message
    assert ("Request a walltime within the cluster's limit for one job "
            "(--walltime), then resume the run again") in message
    assert "the run stays cancelled" in message
    assert any("--time=48:00:00" in call
               for call in fake_slurm.calls("sbatch"))
    parent = jobs.get(submission.job_id)
    assert parent.status == "cancelled" and parent.cancelled
    assert not any(jobs.is_held(tail) for tail in _held_tails(jobs, submission))

    # Resumed again within the limit, the parts continue.
    answer = jobs.resubmit(submission.job_id)
    assert answer["ok"] is True
    assert jobs.get(submission.job_id).status == "running"
    _tick(jobs)
    assert [tail.status for tail in _held_tails(jobs, submission)] == [
        "submitted", "submitted"]


def test_a_resume_refused_without_a_reason_keeps_its_parts_held(
        tmp_path, monkeypatch, fake_slurm):
    """Review round 5: the general sentence alone, for this Resume's new
    walltime, may be the job count (which frees) or a limit on the request
    (which does not). The parts are not withdrawn: they stay held, and the
    answer says plainly that the scheduler gave no reason."""
    from steerlab_server.api.job_status import job_status
    from steerlab_server.api.jobs import HELD_PART_UNEXPLAINED_NOTE
    jobs, submission = _cancelled_run_whose_ran_parts_finished(
        tmp_path, monkeypatch, "heldresumevague")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_IF_ARG", "--time=02:00:00")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE", POLICY)
    answer = jobs.resubmit(submission.job_id, walltime="02:00:00")
    assert answer["ok"] is True
    assert "without saying why" in answer["message"]
    assert "check the walltime against the cluster's limit for one job" in \
        answer["message"]
    # Review round 6: the answer says the parts are held, not that they
    # continue, with a typed field and the scheduler's sentence.
    assert answer["heldUnexplained"] is True
    assert "Job violates accounting/QOS policy" in answer["schedulerMessage"]
    assert "are held, not running" in answer["message"]
    assert "now continue" not in answer["message"]
    held = [jobs.get(entry["jobId"]) for entry in answer["resumedShards"]]
    assert len(held) == 2 and all(jobs.is_held(part) for part in held)
    assert jobs.get(submission.job_id).status == "running"
    shards = job_status(jobs, jobs.get(submission.job_id))["shards"]
    assert shards[2]["heldNote"] == HELD_PART_UNEXPLAINED_NOTE

    # The slot frees, or the cause was the count after all: at the next
    # attempt (its wait is on record, review round 6), they go.
    monkeypatch.delenv("FAKE_SBATCH_FAIL_IF_ARG")
    assert held[0].result["held"]["nextAttemptAt"] > \
        held[0].result["held"]["lastAttemptAt"]
    _due(jobs, held[0])
    _tick(jobs)
    assert [part.status for part in
            (jobs.get(p.id) for p in held)] == ["submitted", "submitted"]


def test_a_resume_whose_part_has_no_script_names_what_is_missing(
        tmp_path, monkeypatch, fake_slurm):
    """Review round 5: a missing script is not a scheduler refusal."""
    jobs, submission = _cancelled_run_whose_ran_parts_finished(
        tmp_path, monkeypatch, "heldresumenoscript")
    tail = _held_tails(jobs, submission)[0]
    os.remove(tail.requested_resources["scriptPath"])
    calls = len(fake_slurm.calls("sbatch"))
    with pytest.raises(ResubmitRefused) as refused:
        jobs.resubmit(submission.job_id, walltime="02:00:00")
    message = str(refused.value)
    assert "cannot be submitted, because its sbatch script is not on disk" \
        in message
    assert message.endswith("submit the study again")
    assert "scheduler" not in message.split("cannot be submitted")[1]
    assert len(fake_slurm.calls("sbatch")) == calls
    assert jobs.get(submission.job_id).status == "cancelled"
    assert not any(jobs.is_held(t) for t in _held_tails(jobs, submission))


def test_a_held_part_with_no_script_fails_its_run_with_a_fresh_submission(
        tmp_path, monkeypatch, fake_slurm):
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldnoscript")
    os.remove(_parts(jobs, submission)[2].requested_resources["scriptPath"])
    _slot_frees(monkeypatch)
    _tick(jobs)
    parent = jobs.get(submission.job_id)
    assert parent.status == "failed"
    assert "its sbatch script is not on disk" in parent.error
    assert parent.error.endswith("Submit the study again")
    assert "Correct what the scheduler said" not in parent.error


def test_two_resumes_at_once_hold_each_part_again_only_once(
        tmp_path, monkeypatch, fake_slurm):
    """Review round 5: both Resumes found the same cancelled held parts
    before either took the lock. Without the check under the lock, each held
    them again, and one continuation per part stayed pending for ever."""
    import threading
    jobs, submission = _cancelled_run_whose_ran_parts_finished(
        tmp_path, monkeypatch, "heldresumetwice")
    gate = threading.Barrier(2)

    class GatedLock:
        """The held lock, but the first two entries wait for each other,
        so both Resumes have found their parts before either holds them."""

        def __init__(self, inner):
            self.inner, self.entered, self.count = inner, threading.Lock(), 0

        def __enter__(self):
            with self.entered:
                self.count += 1
                waits = self.count <= 2
            if waits:
                gate.wait(timeout=10)
            self.inner.acquire()
            return self

        def __exit__(self, *exc):
            self.inner.release()

        def acquire(self, *args, **kwargs):
            return self.inner.acquire(*args, **kwargs)

        def release(self):
            self.inner.release()

    jobs._held_lock = GatedLock(jobs._held_lock)
    answers, refusals = [], []

    def resume():
        try:
            answers.append(jobs.resubmit(submission.job_id))
        except ResubmitRefused as exc:
            refusals.append(str(exc))

    threads = [threading.Thread(target=resume) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert len(answers) == 1 and len(refusals) == 1, (answers, refusals)
    assert "no cancelled part left to resume" in refusals[0]
    for part in _parts(jobs, submission)[2:]:
        continuations = [job for job in jobs.list()
                         if (job.requested_resources or {}).get("resubmitOf")
                         == part.id]
        assert len(continuations) == 1, part.id
    _tick(jobs)
    assert all(tail.status == "submitted"
               for tail in _held_tails(jobs, submission))


def test_a_resume_within_the_limit_submits_its_first_held_part_at_once(
        tmp_path, monkeypatch, fake_slurm):
    jobs, submission = _cancelled_run_whose_ran_parts_finished(
        tmp_path, monkeypatch, "heldresumeok")
    answer = jobs.resubmit(submission.job_id, walltime="02:00:00")
    assert answer["ok"] is True
    first, second = answer["resumedShards"]
    assert first["held"] is False and first["slurmJobID"]
    assert second["held"] is True
    assert jobs.get(first["jobId"]).status == "submitted"
    assert jobs.is_held(jobs.get(second["jobId"]))
    assert "--time=02:00:00" in fake_slurm.calls("sbatch")[-1]


def test_a_resume_refused_for_a_part_that_ran_holds_nothing_again(
        tmp_path, monkeypatch, fake_slurm):
    """The scheduler refuses this Resume's walltime for a part that had run:
    the held parts, which ask for the same, are not held again only to meet
    the same refusal, and the Resume is refused with the repair."""
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldresumeran")
    assert jobs.cancel(submission.job_id) is True
    for slurm_id in ("7001", "7002"):
        fake_slurm.set_state(slurm_id, "CANCELLED", exit_code="0:15")
    _slot_frees(monkeypatch)
    monkeypatch.setenv("FAKE_SBATCH_FAIL_IF_ARG", "--time=48:00:00")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE", MAX_WALL)
    calls = len(fake_slurm.calls("sbatch"))
    with pytest.raises(ResubmitRefused,
                       match="then resume the run again") as refused:
        jobs.resubmit(submission.job_id, walltime="48:00:00")
    assert "QOSMaxWallDurationPerJobLimit" in str(refused.value)
    # Only the two parts that had run were asked about.
    assert len(fake_slurm.calls("sbatch")) == calls + 2
    held_ids = {part.id for part in _parts(jobs, submission)[2:]}
    assert not [job for job in jobs.list()
                if (job.requested_resources or {}).get("resubmitOf") in held_ids]
    assert jobs.get(submission.job_id).status == "cancelled"


def test_the_general_sentence_waits_when_the_same_request_was_accepted(
        tmp_path, monkeypatch, fake_slurm):
    _limited(monkeypatch, accepted=1)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldsentence",
                               parallel=2)
    _limited(monkeypatch, accepted=0, message=POLICY)
    _tick(jobs)
    assert jobs.get(submission.job_id).status == "submitted"
    assert jobs.is_held(_parts(jobs, submission)[1])


def test_the_general_sentence_alone_holds_and_never_stalls_another_run(
        tmp_path, monkeypatch, fake_slurm):
    """Review round 5 (it failed the run in round 2): every part held at the
    first, by its reason code; later the sentence alone, for a request no
    part has had accepted. It may be the job count, which frees, or a limit
    on this request, which does not: not enough to fail on. The parts stay
    held, the note says the scheduler gave no reason, and a later run's held
    parts are still submitted."""
    from steerlab_server.api.job_status import job_status
    from steerlab_server.api.jobs import HELD_PART_UNEXPLAINED_NOTE
    _limited(monkeypatch, accepted=0)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldsentence0",
                               parallel=2)
    later = _submit_into(jobs, tmp_path, name="heldsentence1", parallel=2)
    _slot_frees(monkeypatch)
    first_script = _parts(jobs, submission)[0].requested_resources["scriptPath"]
    monkeypatch.setenv("FAKE_SBATCH_FAIL_IF_ARG", first_script)
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE", POLICY)
    _tick(jobs)
    parent = jobs.get(submission.job_id)
    assert parent.status == "submitted" and parent.error is None
    assert all(jobs.is_held(p) for p in _parts(jobs, submission))
    assert fake_slurm.calls("scancel") == []
    first = _parts(jobs, submission)[0]
    assert first.result["held"]["lastRefusalUnexplained"] is True
    assert any("without saying why" in line and "check its walltime" in line
               for line in first.all_logs())
    entry = job_status(jobs, parent)["shards"][0]
    assert entry["heldNote"] == HELD_PART_UNEXPLAINED_NOTE
    # The later run was not kept waiting behind it.
    assert [p.status for p in _parts(jobs, later)] == ["submitted",
                                                       "submitted"]


def _due(jobs, part):
    """Make ``part``'s next attempt due now (as if its wait had passed)."""
    part = jobs.get(part.id)
    held = dict(part.result["held"])
    held["nextAttemptAt"] = 0.0
    part.result = {**part.result, "held": held}
    jobs.store.update(part)
    jobs._held_deferred_until.clear()


def _unexplained_run(tmp_path, monkeypatch, name):
    """A run every part of which was held at the first by a reason code,
    and whose scheduler now refuses with the general sentence alone."""
    _limited(monkeypatch, accepted=0)
    jobs, submission = _submit(tmp_path, monkeypatch, name=name, parallel=2)
    _limited(monkeypatch, accepted=0, message=POLICY)
    return jobs, submission


def test_unexplained_refusals_back_off_to_half_an_hour(tmp_path, monkeypatch,
                                                       fake_slurm):
    """Review round 6: one sbatch a minute for ever was the old answer. Each
    refusal of the same request that gives no reason waits longer: 1, 2, 4,
    8, 16, then 30 minutes, stamped on the record."""
    jobs, submission = _unexplained_run(tmp_path, monkeypatch, "heldbackoff")
    first = _parts(jobs, submission)[0]
    delays = []
    for attempt in range(1, 8):
        _due(jobs, first)
        _tick(jobs)
        held = jobs.get(first.id).result["held"]
        assert held["unexplainedRefusals"] == attempt
        delays.append(round(held["nextAttemptAt"] - held["lastAttemptAt"]))
    assert delays == [60, 120, 240, 480, 960, 1800, 1800]
    # Within the wait, the scheduler is not asked again.
    calls = len(fake_slurm.calls("sbatch"))
    jobs._held_deferred_until.clear()
    _tick(jobs)
    assert len(fake_slurm.calls("sbatch")) == calls
    assert jobs.get(submission.job_id).status == "submitted"


def test_a_day_of_unexplained_refusals_fails_the_run_resumably(
        tmp_path, monkeypatch, fake_slurm):
    from steerlab_server.api.jobs import cancel_resume_hint
    jobs, submission = _unexplained_run(tmp_path, monkeypatch, "heldgiveup")
    first = _parts(jobs, submission)[0]
    _due(jobs, first)
    _tick(jobs)
    held = dict(jobs.get(first.id).result["held"])
    held["unexplainedSince"] = held["lastAttemptAt"] - 24 * 3600 - 1
    held["nextAttemptAt"] = 0.0
    part = jobs.get(first.id)
    part.result = {**part.result, "held": held}
    jobs.store.update(part)
    jobs._held_deferred_until.clear()
    _tick(jobs)

    parent = jobs.get(submission.job_id)
    assert parent.status == "failed"
    assert "without saying why" in parent.error
    assert "Job violates accounting/QOS policy" in parent.error
    # Review round 8: the repair names both ways on, on both clients.
    assert ("Choose one: resume it with a walltime within the cluster's "
            "limit for one job") in parent.error
    assert (f"steerlab-cli remote resubmit {parent.id} "
            "--retry-refused-walltime") in parent.error
    assert (f"steerlab runner resubmit {parent.id} --retry-refused-walltime "
            "--runner <url>") in parent.error
    assert parent.result["heldGaveUp"]["refusals"] == 2
    assert parent.result["heldGaveUp"]["walltime"] == "00:30:00"
    assert not any(jobs.is_held(p) for p in _parts(jobs, submission))
    # Review round 7: the app's Resume sends no walltime, and resuming at the
    # refused one would start another day of refusals. It is not offered
    # there; the line names the command that helps.
    hint = cancel_resume_hint(parent)
    assert hint["offered"] is False and hint["needsWalltime"] is True
    assert hint["refusedWalltime"] == "00:30:00"
    # Review round 8: both choices, on both clients, and never "refused the
    # walltime": the cause may have been the job count.
    job_id = submission.job_id
    for command in (f"steerlab-cli remote resubmit {job_id} --walltime HH:MM:SS",
                    f"steerlab runner resubmit {job_id} --walltime HH:MM:SS",
                    f"steerlab-cli remote resubmit {job_id} "
                    "--retry-refused-walltime",
                    f"steerlab runner resubmit {job_id} "
                    "--retry-refused-walltime"):
        assert command in hint["explanation"], command
    assert "refused its parts at walltime 00:30:00, without a reason, for " \
        "a day" in hint["explanation"]

    # At that walltime (none given, or the same spelled another way) and
    # without the acknowledgement, the Resume is refused at once, with both
    # choices, and nothing moves.
    _slot_frees(monkeypatch)
    calls = len(fake_slurm.calls("sbatch"))
    for asked in (None, "30:00"):
        with pytest.raises(ResubmitRefused) as refused:
            jobs.resubmit(submission.job_id, walltime=asked)
        message = str(refused.value)
        assert ("the scheduler refused its parts at walltime 00:30:00, "
                "without a reason, for a day") in message
        assert "refused its walltime" not in message
        assert (f"steerlab-cli remote resubmit {job_id} --walltime "
                "HH:MM:SS") in message
        assert (f"steerlab-cli remote resubmit {job_id} "
                "--retry-refused-walltime") in message
    assert len(fake_slurm.calls("sbatch")) == calls
    assert jobs.get(submission.job_id).status == "failed"

    # Within the limit, the parts are held again and go.
    answer = jobs.resubmit(submission.job_id, walltime="00:20:00")
    assert answer["ok"] is True
    # Worded by what happened (review round 8): it stopped, not cancelled.
    assert answer["message"].startswith(
        f"Run {job_id} stopped after a day of refusals that gave no reason;")
    parent = jobs.get(submission.job_id)
    assert parent.status == "running"
    assert "heldGaveUp" not in parent.result
    assert parent.result["heldGaveUpBefore"]["part"] == first.id
    _tick(jobs)
    assert [tail.status for tail in
            (jobs._shard_chain_tail(p) for p in _parts(jobs, submission))] == [
        "submitted", "submitted"]


def _gave_up_run(tmp_path, monkeypatch, name):
    """A run whose parts were held at the first by a reason code, then
    refused without a reason for a day: it gave up."""
    jobs, submission = _unexplained_run(tmp_path, monkeypatch, name)
    first = _parts(jobs, submission)[0]
    _due(jobs, first)
    _tick(jobs)
    held = dict(jobs.get(first.id).result["held"])
    held["unexplainedSince"] = held["lastAttemptAt"] - 24 * 3600 - 1
    held["nextAttemptAt"] = 0.0
    part = jobs.get(first.id)
    part.result = {**part.result, "held": held}
    jobs.store.update(part)
    jobs._held_deferred_until.clear()
    _tick(jobs)
    assert jobs.get(submission.job_id).status == "failed"
    return jobs, submission


def test_a_gave_up_run_retries_the_same_walltime_when_asked_to(
        tmp_path, monkeypatch, fake_slurm):
    """Review round 8: "unexplained" may have been the job count, as in issue
    #12's queues that stay full for days. Once slots free, a Resume at the
    same, correct walltime runs when the request acknowledges it."""
    jobs, submission = _gave_up_run(tmp_path, monkeypatch, "heldretrysame")
    _slot_frees(monkeypatch)
    with pytest.raises(ResubmitRefused):
        jobs.resubmit(submission.job_id)
    calls = len(fake_slurm.calls("sbatch"))
    answer = jobs.resubmit(submission.job_id, retry_refused_walltime=True)
    assert answer["ok"] is True
    assert jobs.get(submission.job_id).status == "running"
    # Review round 9: the walltime that was refused is asked for, and the
    # first part is handed to the scheduler in the request, as for an
    # explicit --walltime, so the answer says what the scheduler did.
    assert answer["retriedRefusedWalltime"] == "00:30:00"
    sent = fake_slurm.calls("sbatch")[calls:]
    assert len(sent) == 1 and "--time=00:30:00" in sent[0]
    assert [entry["held"] for entry in answer["resumedShards"]] == [False, True]
    assert ("stopped after a day of refusals that gave no reason; retried "
            "at walltime 00:30:00, 2 of its parts now continue (1 held")  \
        in answer["message"]
    _tick(jobs)
    tails = [jobs._shard_chain_tail(p) for p in _parts(jobs, submission)]
    assert [tail.status for tail in tails] == ["submitted", "submitted"]
    sent = fake_slurm.calls("sbatch")[calls:]
    assert len(sent) == 2 and all("--time=00:30:00" in call for call in sent)


def _give_up(jobs, part_id):
    """A day of ``part_id``'s refusals without a reason, then its next
    attempt: its run gives up (no other part of it is live)."""
    part = jobs.get(part_id)
    held = dict(part.result["held"])
    held["unexplainedSince"] = held["lastAttemptAt"] - 24 * 3600 - 1
    held["nextAttemptAt"] = 0.0
    part.result = {**part.result, "held": held}
    jobs.store.update(part)
    jobs._held_deferred_until.clear()
    _tick(jobs)


def test_a_retry_asks_for_the_walltime_refused_not_the_scripts(
        tmp_path, monkeypatch, fake_slurm):
    """Review round 9: the run gave up at W1, was resumed at W2, and gave up
    at W2. "Retry at W2" asks for W2. Before, a retry without --walltime
    dropped the earlier Resume's override, and the parts went back at W1."""
    from steerlab_server.api.jobs import cancel_resume_hint
    jobs, submission = _gave_up_run(tmp_path, monkeypatch, "heldretryw2")
    # The scheduler refuses W2 without a reason too.
    answer = jobs.resubmit(submission.job_id, walltime="02:00:00")
    assert answer["heldUnexplained"] is True
    assert "retriedRefusedWalltime" not in answer
    first = answer["resumedShards"][0]["jobId"]
    assert jobs.get(first).requested_resources["walltimeOverride"] == "02:00:00"
    _give_up(jobs, first)
    parent = jobs.get(submission.job_id)
    assert parent.status == "failed"
    assert parent.result["heldGaveUp"]["walltime"] == "02:00:00"
    assert cancel_resume_hint(parent)["refusedWalltime"] == "02:00:00"
    assert "retry walltime 02:00:00 (" in parent.error

    _slot_frees(monkeypatch)
    calls = len(fake_slurm.calls("sbatch"))
    answer = jobs.resubmit(submission.job_id, retry_refused_walltime=True)
    assert answer["retriedRefusedWalltime"] == "02:00:00"
    sent = fake_slurm.calls("sbatch")[calls:]
    assert len(sent) == 1 and "--time=02:00:00" in sent[0]
    waiting = [jobs.get(entry["jobId"]) for entry in answer["resumedShards"][1:]]
    assert waiting and all(
        part.requested_resources["walltimeOverride"] == "02:00:00"
        for part in waiting)
    _tick(jobs)
    sent = fake_slurm.calls("sbatch")[calls:]
    assert len(sent) == 2 and all("--time=02:00:00" in call for call in sent)
    assert not any("--time=00:30:00" in call for call in sent)


@pytest.mark.parametrize("message,field,words", [
    (LIMIT, "heldReason",
     "was resumed at walltime 00:30:00; its 2 parts are held until the "
     "cluster's job limit frees a slot"),
    (POLICY, "heldUnexplained",
     "was resumed at walltime 00:30:00, but its 2 parts are held, not "
     "running"),
])
def test_a_retry_refused_again_says_what_the_scheduler_did(
        tmp_path, monkeypatch, fake_slurm, message, field, words):
    jobs, submission = _gave_up_run(tmp_path, monkeypatch,
                                    f"heldretryagain{field.lower()}")
    _limited(monkeypatch, accepted=0, message=message)
    answer = jobs.resubmit(submission.job_id, retry_refused_walltime=True)
    assert answer["ok"] is True and field in answer
    assert answer["retriedRefusedWalltime"] == "00:30:00"
    assert words in answer["message"]
    assert "now continue" not in answer["message"]
    if field == "heldUnexplained":
        # Review round 9: never "stops the run" without the exception.
        assert ("stops the run after 24 hours of such refusals, unless "
                "another part of it is with the scheduler") in answer["message"]


def _walltime_in_days(jobs, submission, walltime):
    """A run whose parts' own walltime is a site default in days: its
    gave-up stamp records it in that spelling, as ``_give_up_unexplained``
    copies a part's walltime."""
    parent = jobs.get(submission.job_id)
    parent.result = {**parent.result, "heldGaveUp": {
        **parent.result["heldGaveUp"], "walltime": walltime}}
    jobs.store.update(parent)
    for part in _parts(jobs, submission):
        for record in {part.id: part,
                       jobs._shard_chain_tail(part).id:
                           jobs._shard_chain_tail(part)}.values():
            rr = {**record.requested_resources, "walltime": walltime}
            rr.pop("walltimeOverride", None)
            record.requested_resources = rr
            jobs.store.update(record)


@pytest.mark.parametrize("days,hours", [("1-00:00:00", "24:00:00"),
                                        ("2-12:30:00", "60:30:00")])
def test_a_retry_of_a_walltime_in_days_runs(tmp_path, monkeypatch,
                                            fake_slurm, days, hours):
    """Review round 10 (the reviewer's probe): the stamp is the parts' own
    spelling, which a site default may give in days. The retry asks for it
    as sbatch's override takes it, never answers that the scheduler refuses
    SteerLab's own validation, and the run continues."""
    jobs, submission = _gave_up_run(tmp_path, monkeypatch,
                                    f"heldretrydays{hours[:2]}")
    _walltime_in_days(jobs, submission, days)
    _slot_frees(monkeypatch)
    calls = len(fake_slurm.calls("sbatch"))
    answer = jobs.resubmit(submission.job_id, retry_refused_walltime=True)
    assert answer["ok"] is True
    assert answer["retriedRefusedWalltime"] == days
    assert f"retried at walltime {days}, 2 of its parts now continue" in \
        answer["message"]
    sent = fake_slurm.calls("sbatch")[calls:]
    assert len(sent) == 1 and f"--time={hours}" in sent[0]
    _tick(jobs)
    tails = [jobs._shard_chain_tail(p) for p in _parts(jobs, submission)]
    assert [tail.status for tail in tails] == ["submitted", "submitted"]
    sent = fake_slurm.calls("sbatch")[calls:]
    assert len(sent) == 2 and all(f"--time={hours}" in call for call in sent)
    # Without the acknowledgement, the same walltime is still refused.
    jobs2, submission2 = _gave_up_run(tmp_path / "again", monkeypatch,
                                      f"heldrefusedays{hours[:2]}")
    _walltime_in_days(jobs2, submission2, days)
    with pytest.raises(ResubmitRefused, match=f"at walltime {days}"):
        jobs2.resubmit(submission2.job_id)


def test_a_retry_in_days_resubmits_a_checkpointed_part_at_it(
        tmp_path, monkeypatch, fake_slurm):
    """Review round 10: a part that checkpointed is resubmitted by the retry
    through the resubmission's own validation, which reads no days. The
    walltime reaches it as sbatch's override takes it, so the part resumes
    instead of being refused as an invalid walltime."""
    _limited(monkeypatch, accepted=1)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldckptdays",
                               parallel=2)
    running, waiting = _parts(jobs, submission)
    _override_held_walltime(jobs, submission, "48:00:00")
    _slot_frees(monkeypatch)
    monkeypatch.setenv("FAKE_SBATCH_FAIL_IF_ARG", "--time=48:00:00")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE", POLICY)
    checkpointed = jobs.get(running.id)
    checkpointed.status = "checkpointed"
    jobs.store.update(checkpointed)
    fake_slurm.set_state(running.executor_job_id, "FAILED", "85:0")
    _due(jobs, waiting)
    _tick(jobs)
    _give_up(jobs, waiting.id)
    parent = jobs.get(submission.job_id)
    assert parent.status == "failed"
    # The same 48 hours, in a site's spelling of it.
    parent.result = {**parent.result, "heldGaveUp": {
        **parent.result["heldGaveUp"], "walltime": "2-00:00:00"}}
    jobs.store.update(parent)
    monkeypatch.delenv("FAKE_SBATCH_FAIL_IF_ARG")
    _slot_frees(monkeypatch)
    calls = len(fake_slurm.calls("sbatch"))
    answer = jobs.resubmit(submission.job_id, retry_refused_walltime=True)
    assert "notResumed" not in answer, answer
    assert answer["retriedRefusedWalltime"] == "2-00:00:00"
    assert len(answer["resumedShards"]) == 2
    _tick(jobs)
    sent = fake_slurm.calls("sbatch")[calls:]
    assert len(sent) == 2 and all("--time=48:00:00" in call for call in sent)


def test_a_walltime_steerlab_cannot_read_is_never_the_schedulers(
        tmp_path, monkeypatch, fake_slurm):
    """Review round 10: a walltime SteerLab cannot read is refused in its
    own words, with the repair, and nothing is asked of the scheduler."""
    jobs, submission = _gave_up_run(tmp_path, monkeypatch, "heldretryunread")
    _walltime_in_days(jobs, submission, "UNLIMITED")
    _slot_frees(monkeypatch)
    calls = len(fake_slurm.calls("sbatch"))
    with pytest.raises(ResubmitRefused) as refused:
        jobs.resubmit(submission.job_id, retry_refused_walltime=True)
    text = str(refused.value)
    assert "SteerLab could not read the walltime its parts were refused at" \
        in text
    assert "scheduler refuses" not in text
    assert refused.value.code == "unreadableWalltime"
    assert refused.value.repair.startswith(
        "resume it with an explicit walltime within the cluster's limit")
    assert f"steerlab-cli remote resubmit {submission.job_id} --walltime " \
        "HH:MM:SS" in refused.value.repair
    assert fake_slurm.calls("sbatch")[calls:] == []
    assert jobs.get(submission.job_id).status == "failed"


@pytest.mark.parametrize("override,submitted", [("1-00:00:00", "24:00:00"),
                                                ("UNLIMITED", None)])
def test_the_held_pass_reads_a_stored_override_itself(
        tmp_path, monkeypatch, fake_slurm, override, submitted):
    """Review round 10, at the held pass too: a stored override in days is
    asked for as sbatch takes it; one SteerLab cannot read fails the run in
    SteerLab's words, with nothing sent, never "the scheduler refused it"."""
    _limited(monkeypatch, accepted=1)
    jobs, submission = _submit(tmp_path, monkeypatch,
                               name=f"heldoverride{bool(submitted)}".lower(),
                               parallel=2)
    _override_held_walltime(jobs, submission, override)
    _slot_frees(monkeypatch)
    calls = len(fake_slurm.calls("sbatch"))
    _tick(jobs)
    waiting = _parts(jobs, submission)[1]
    sent = fake_slurm.calls("sbatch")[calls:]
    if submitted:
        assert jobs.get(waiting.id).status == "submitted"
        assert len(sent) == 1 and f"--time={submitted}" in sent[0]
    else:
        assert sent == []
        parent = jobs.get(submission.job_id)
        assert parent.status == "failed"
        assert "SteerLab could not read its walltime ('UNLIMITED')" in \
            parent.error
        assert "the scheduler refused it" not in parent.error
        assert "--walltime HH:MM:SS" in parent.error
        # The run failed outright, so it cannot be resumed: the repair says
        # to submit again, never to resume it.
        assert "submit the study again" in parent.error.lower()
        assert "remote resubmit" not in parent.error


def test_a_retry_refused_for_a_limit_that_never_frees_is_refused(
        tmp_path, monkeypatch, fake_slurm):
    jobs, submission = _gave_up_run(tmp_path, monkeypatch, "heldretrymax")
    _limited(monkeypatch, accepted=0, message=MAX_WALL)
    with pytest.raises(ResubmitRefused) as refused:
        jobs.resubmit(submission.job_id, retry_refused_walltime=True)
    assert "QOSMaxWallDurationPerJobLimit" in str(refused.value)
    # A run that gave up stays stopped, not "cancelled", and the refusal
    # carries its repair, typed, for the clients' repairAction.
    assert "the run stays stopped" in str(refused.value)
    assert "stays cancelled" not in str(refused.value)
    assert refused.value.code == "schedulerRefused"
    assert refused.value.repair == (
        "request a walltime within the cluster's limit for one job "
        "(--walltime), then resume the run again")
    assert jobs.get(submission.job_id).status == "failed"


def test_the_route_reads_the_acknowledgement(tmp_path, monkeypatch,
                                             fake_slurm):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router
    jobs, submission = _gave_up_run(tmp_path, monkeypatch, "heldretryroute")
    _slot_frees(monkeypatch)
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "route-jobs.sqlite"))
    state = ServiceState()
    state._jobs = jobs  # the routes see the run that gave up
    app = FastAPI()
    app.include_router(build_router(state))
    client = TestClient(app)
    refused = client.post(f"/api/jobs/{submission.job_id}/resubmit", json={})
    assert refused.status_code == 409
    body = refused.json()
    # ``detail`` stays the plain string every client reads; the way on rides
    # beside it, typed (review round 9).
    assert isinstance(body["detail"], str)
    assert "--retry-refused-walltime" in body["detail"]
    assert body["code"] == "refusedWalltime"
    assert body["repairAction"].startswith("Choose one: resume it with a "
                                           "walltime within the cluster's")
    assert (f"steerlab-cli remote resubmit {submission.job_id} "
            "--retry-refused-walltime") in body["repairAction"]
    assert "retry walltime 00:30:00 (" in body["repairAction"]
    accepted = client.post(f"/api/jobs/{submission.job_id}/resubmit",
                           json={"retryRefusedWalltime": True})
    assert accepted.status_code == 200, accepted.text
    client.close()


def test_walltimes_compare_in_every_slurm_spelling():
    from steerlab_server.api.jobs import same_walltime
    assert same_walltime("1-00:00:00", "24:00:00")
    assert same_walltime("30:00", "00:30:00")
    assert same_walltime("1-12", "36:00:00")
    assert not same_walltime("2-00:00:00", "24:00:00")
    assert not same_walltime("", "")
    assert not same_walltime("UNLIMITED", "UNLIMITED")


def test_a_site_default_in_days_is_compared_as_the_same_walltime(
        tmp_path, monkeypatch, fake_slurm):
    jobs, submission = _gave_up_run(tmp_path, monkeypatch, "helddays")
    parent = jobs.get(submission.job_id)
    parent.result = {**parent.result, "heldGaveUp": {
        **parent.result["heldGaveUp"], "walltime": "1-00:00:00"}}
    jobs.store.update(parent)
    with pytest.raises(ResubmitRefused, match="at walltime 1-00:00:00"):
        jobs.resubmit(submission.job_id, walltime="24:00:00")


def test_a_checkpointed_part_is_not_live(tmp_path, monkeypatch, fake_slurm):
    """Review round 8 (the reviewer's probe): a checkpointed part's job has
    ended. It does not keep the run from giving up, and the log never says
    it is with the scheduler."""
    _limited(monkeypatch, accepted=1)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldckptlive",
                               parallel=2)
    running, waiting = _parts(jobs, submission)
    _override_held_walltime(jobs, submission, "48:00:00")
    _slot_frees(monkeypatch)
    monkeypatch.setenv("FAKE_SBATCH_FAIL_IF_ARG", "--time=48:00:00")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE", POLICY)
    checkpointed = jobs.get(running.id)
    checkpointed.status = "checkpointed"
    jobs.store.update(checkpointed)
    assert jobs._live_parts_of(jobs.get(submission.job_id),
                               excluding=waiting.id) == []
    _due(jobs, waiting)
    _tick(jobs)
    held = dict(jobs.get(waiting.id).result["held"])
    held["unexplainedSince"] = held["lastAttemptAt"] - 24 * 3600 - 1
    part = jobs.get(waiting.id)
    part.result = {**part.result, "held": held}
    jobs.store.update(part)
    _due(jobs, waiting)
    _tick(jobs)
    parent = jobs.get(submission.job_id)
    assert parent.status == "failed"
    assert "heldGaveUp" in parent.result
    assert not any("with the scheduler" in line
                   for line in jobs.get(waiting.id).all_logs())


def test_a_run_with_a_live_part_never_gives_up(tmp_path, monkeypatch,
                                               fake_slurm):
    """Review round 7: giving up cancels every live part. A run any part of
    which is with the scheduler keeps waiting instead."""
    _limited(monkeypatch, accepted=1)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldliveguard",
                               parallel=2)
    running, waiting = _parts(jobs, submission)
    assert running.status == "submitted" and jobs.is_held(waiting)
    # The held part asks for another walltime, so the running part's
    # acceptance says nothing about it: its refusals stay unexplained.
    _override_held_walltime(jobs, submission, "48:00:00")
    _slot_frees(monkeypatch)
    monkeypatch.setenv("FAKE_SBATCH_FAIL_IF_ARG", "--time=48:00:00")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE", POLICY)
    _due(jobs, waiting)
    _tick(jobs)
    held = dict(jobs.get(waiting.id).result["held"])
    assert held["unexplainedRefusals"] == 1
    held["unexplainedSince"] = held["lastAttemptAt"] - 24 * 3600 - 1
    part = jobs.get(waiting.id)
    part.result = {**part.result, "held": held}
    jobs.store.update(part)
    _due(jobs, waiting)
    _tick(jobs)

    parent = jobs.get(submission.job_id)
    assert parent.status == "submitted" and parent.error is None
    assert "heldGaveUp" not in parent.result
    assert fake_slurm.calls("scancel") == []
    assert jobs.get(running.id).status == "submitted"
    assert jobs.is_held(jobs.get(waiting.id))
    assert jobs.get(waiting.id).result["held"]["unexplainedRefusals"] == 2
    # Review round 8: past the bound with a live part, the log says it keeps
    # waiting, never that it stops the run.
    past_bound = [line for line in jobs.get(waiting.id).all_logs()
                  if "Refusal 2 of this kind" in line]
    assert len(past_bound) == 1
    assert "keeps waiting rather than stop them" in past_bound[0]
    assert "stops the run after" not in past_bound[0]


def test_a_resume_held_for_a_named_job_count_says_so(tmp_path, monkeypatch,
                                                     fake_slurm):
    """Review round 7: held for a job count is a success, worded as held
    until a slot frees, never "N parts now continue"."""
    jobs, submission = _cancelled_run_whose_ran_parts_finished(
        tmp_path, monkeypatch, "heldresumecount")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_IF_ARG", "--time=02:00:00")
    monkeypatch.setenv("FAKE_SBATCH_FAIL_MESSAGE", LIMIT)
    answer = jobs.resubmit(submission.job_id, walltime="02:00:00")
    assert answer["ok"] is True
    assert answer["heldReason"] == "jobLimit"
    assert "heldUnexplained" not in answer
    assert "QOSMaxSubmitJobPerUserLimit" in answer["schedulerMessage"]
    assert "2 parts are held until the cluster's job limit frees a slot" in \
        answer["message"]
    assert "now continue" not in answer["message"]
    held = [jobs.get(entry["jobId"]) for entry in answer["resumedShards"]]
    assert all(jobs.is_held(part) for part in held)


def test_a_resume_whose_sbatch_went_unanswered_says_so(tmp_path, monkeypatch,
                                                       fake_slurm):
    jobs, submission = _cancelled_run_whose_ran_parts_finished(
        tmp_path, monkeypatch, "heldresumesilent")

    def unanswered(self, bundle, **kwargs):
        raise executors.SchedulerSubmitError(
            "sbatch did not answer within 120s and was abandoned",
            returncode=executors.SCHEDULER_TIMEOUT_EXIT)

    monkeypatch.setattr(executors.SlurmExecutor, "submit", unanswered)
    answer = jobs.resubmit(submission.job_id, walltime="02:00:00")
    assert answer["ok"] is True
    assert answer["heldReason"] == "schedulerUnanswered"
    assert "are held until the scheduler answers" in answer["message"]
    assert "did not answer about the first just now" in answer["message"]
    assert "first checks whether that attempt landed" in answer["message"]
    assert "now continue" not in answer["message"]


def test_a_slot_that_frees_first_submits_the_part(tmp_path, monkeypatch,
                                                  fake_slurm):
    """The cause was the job count after all: when the slot frees before
    the bound, the part is accepted at its next attempt, and the record
    keeps how long it waited."""
    jobs, submission = _unexplained_run(tmp_path, monkeypatch, "heldslotfree")
    first = _parts(jobs, submission)[0]
    for _ in range(2):
        _due(jobs, first)
        _tick(jobs)
    assert jobs.get(first.id).result["held"]["unexplainedRefusals"] == 2
    _slot_frees(monkeypatch)
    _due(jobs, first)
    _tick(jobs)
    assert [p.status for p in _parts(jobs, submission)] == ["submitted",
                                                            "submitted"]
    assert jobs.get(first.id).result["heldBeforeSubmit"][
        "unexplainedRefusals"] == 2
    assert jobs.get(submission.job_id).status == "submitted"


def test_a_named_job_count_ends_a_run_of_unexplained_refusals(
        tmp_path, monkeypatch, fake_slurm):
    jobs, submission = _unexplained_run(tmp_path, monkeypatch, "heldnamed")
    first = _parts(jobs, submission)[0]
    for _ in range(2):
        _due(jobs, first)
        _tick(jobs)
    _limited(monkeypatch, accepted=0)
    _due(jobs, first)
    _tick(jobs)
    held = jobs.get(first.id).result["held"]
    for key in ("lastRefusalUnexplained", "unexplainedRefusals",
                "unexplainedSince", "nextAttemptAt"):
        assert key not in held, key


def test_a_maintenance_window_defers_one_run_and_not_the_next(
        tmp_path, monkeypatch, fake_slurm):
    _limited(monkeypatch, accepted=1)
    jobs, first = _submit(tmp_path, monkeypatch, name="heldwindowa",
                          parallel=2)
    second = _submit_into(jobs, tmp_path, name="heldwindowb", parallel=2)
    _slot_frees(monkeypatch)
    waiting = _parts(jobs, first)[1]
    script = waiting.requested_resources["scriptPath"]
    real_submit = executors.SlurmExecutor.submit
    refused = []

    def windowed(self, bundle, **kwargs):
        if bundle.script_path == script:
            refused.append(script)
            raise RuntimeError(
                "requested walltime crosses a configured maintenance window")
        return real_submit(self, bundle, **kwargs)

    monkeypatch.setattr(executors.SlurmExecutor, "submit", windowed)
    _tick(jobs)
    assert jobs.is_held(jobs.get(waiting.id))
    assert jobs.get(first.job_id).status == "submitted"
    assert [p.status for p in _parts(jobs, second)] == ["submitted",
                                                        "submitted"]
    # The deferred run is not asked again within its pause.
    _tick(jobs)
    assert len(refused) == 1
    # The window passes: it goes.
    monkeypatch.setattr(executors.SlurmExecutor, "submit", real_submit)
    jobs._held_deferred_until.clear()
    _tick(jobs)
    assert jobs.get(waiting.id).status == "submitted"


# --- review round 2: the name search reaches back to the attempt ---------------

def test_the_name_search_reaches_back_past_midnight(tmp_path, monkeypatch,
                                                    fake_slurm):
    """An attempt made yesterday whose job ran and ended before midnight.
    sacct --name searches from today's midnight unless told otherwise, so
    the old search read it as absent and the part ran twice."""
    import datetime as dt
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldmidnight")
    part = _parts(jobs, submission)[2]
    midnight = dt.datetime.now().replace(hour=0, minute=0, second=0,
                                         microsecond=0)
    attempted = (midnight - dt.timedelta(hours=2)).timestamp()
    ended = (midnight - dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")
    token = f"slhd-{part.id}-yesterday"
    part.result = {**part.result, "held": {**part.result["held"],
                                           "attempt": {"submissionName": token,
                                                       "at": attempted}}}
    jobs.store.update(part)
    _scheduler_entry("9900", state="COMPLETED", exit="0:0", name=token,
                     end=ended)
    _slot_frees(monkeypatch)
    calls = len(fake_slurm.calls("sbatch"))
    _tick(jobs)

    adopted = jobs.get(part.id)
    assert adopted.executor_job_id == "9900"
    assert not any("slurm-shard-2/" in call
                   for call in fake_slurm.calls("sbatch")[calls:])
    search = [call for call in fake_slurm.calls("sacct") if token in call]
    start = executors.sacct_time(attempted
                                 - executors.NAME_SEARCH_MARGIN_SECONDS)
    assert search and f"-S {start} -E now" in search[0]


def test_no_way_to_search_is_not_proof_that_an_attempt_never_landed(
        tmp_path, monkeypatch, fake_slurm):
    _limited(monkeypatch, accepted=2)
    jobs, submission = _submit(tmp_path, monkeypatch, name="heldnosearch")
    part = _parts(jobs, submission)[2]
    part.result = {**part.result, "held": {**part.result["held"],
                                           "attempt": {"submissionName":
                                                       f"slhd-{part.id}-0",
                                                       "at": 1.0}}}
    jobs.store.update(part)
    submitted = []

    class NoSearch:
        def submit(self, bundle, **kwargs):
            submitted.append(bundle.script_path)
            return "9999"

    jobs._slurm_executor = NoSearch()
    _slot_frees(monkeypatch)
    _tick(jobs)
    assert submitted == []
    assert jobs.is_held(jobs.get(part.id))


def test_the_name_search_uses_the_sites_sacct_and_never_guesses(
        tmp_path, monkeypatch, fake_slurm):
    import shutil
    import time
    bindir = tmp_path / "bin"
    shutil.copy(bindir / "sacct", bindir / "site-sacct")
    (bindir / "sacct").write_text("#!/bin/sh\necho 'not the site one' >&2\n"
                                  "exit 1\n", encoding="utf-8")
    monkeypatch.setenv("STEERLAB_SLURM_SACCT", "site-sacct")
    _scheduler_entry("9911", state="RUNNING", name="slhd-x-1")
    executor = executors.SlurmExecutor()
    assert executor.find_job_by_name("slhd-x-1",
                                     since=time.time() - 60) == "9911"
    assert executor.find_job_by_name("slhd-x-2",
                                     since=time.time() - 60) is None
    monkeypatch.setenv("FAKE_SACCT_FAIL", "accounting is down")
    with pytest.raises(RuntimeError, match="absence cannot be proven"):
        executor.find_job_by_name("slhd-x-2", since=time.time() - 60)


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
    # Review round 8: unanswered is not "not taken".
    assert any("did not answer about" in line and "was not taken" not in line
               for line in part.all_logs())

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


@pytest.mark.parametrize("value", ["+90", "1_000", "1:-5:00", "-5", "\u0661\u0662"])
def test_a_walltime_field_is_ascii_digits_only(value):
    """Review round 10: ``int`` takes these, sbatch does not; refuse them
    where the walltime is typed, never later at the held pass."""
    from steerlab_server.api import executors
    with pytest.raises(ValueError):
        executors.validated_walltime(value)


def test_a_maintenance_window_reads_a_walltime_in_days(tmp_path):
    """Review round 10: a site default written in days must price against
    the maintenance calendar, not raise as if the scheduler refused it."""
    import datetime as _dt
    import json as _json
    from steerlab_server.api import executors
    now = _dt.datetime.now(_dt.timezone.utc)
    calendar = tmp_path / "maintenance.json"
    calendar.write_text(_json.dumps([{
        "start": (now + _dt.timedelta(hours=20)).isoformat(),
        "end": (now + _dt.timedelta(hours=30)).isoformat(),
        "label": "window"}]))
    assert executors.first_crossing_window("1-00:00:00", str(calendar)) is not None
    assert executors.first_crossing_window("12:00:00", str(calendar)) is None
    with pytest.raises(ValueError):
        executors.first_crossing_window("UNLIMITED", str(calendar))
