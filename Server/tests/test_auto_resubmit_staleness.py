"""Auto-resubmit staleness and provenance guards (live controller incident,
2026-09-13).

A controller started that evening auto-resubmitted a SEVEN-WEEK-OLD test
record: its Slurm job had been cancelled in July, but the record read
``checkpointed`` because every terminal transition re-folded the whole records
directory and the old child record restored that status. The poll loop then
logged the same ``→ cancelled`` transition every tick. Properties under test:

- the OBSERVED scheduler state ``cancelled`` vetoes the automatic path
  (cancelled beats checkpointed, not only via the record's own cancel flag);
- a record inactive longer than the bound (48 h shipped default, per-request
  ``autoResubmitMaxAgeSeconds``, env ``STEERLAB_AUTO_RESUBMIT_MAX_AGE``) is
  skipped with a note and otherwise left alone;
- a controller never auto-resubmits a checkpointed record it merely ADOPTED at
  start; it may only repair a missing ``resubmittedAs`` stamp;
- a fresh checkpoint this instance witnessed still resubmits (unchanged);
- a fold-restored ``checkpointed`` over a terminal scheduler state is handled
  ONCE, not re-logged every tick;
- the manual Resume verb keeps working on a stale record (explicit consent).
"""

import json
import os
import time

import pytest

from steerlab_server.api import executors
from steerlab_server.api import submissions as sub
from steerlab_server.api.executors import SlurmExecutor
from steerlab_server.api.jobs import DurableJobStore, JobManager, ResubmitRefused

from test_auto_resubmit import (  # noqa: F401 - fixtures by import
    _checkpointed, _dummy_script, _manager, fake_slurm)

HOUR = 3600.0


def _sacct_time(epoch: float) -> str:
    """sacct's ``End`` column for an epoch second (scheduler-local time)."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(epoch))


def _backdate(mgr, job, *, to: float) -> None:
    """Fabricated records are born now; a real one was created and started
    when its Slurm job was submitted. Move the lifecycle stamps back so the
    record's own history reads the way the incident's did."""
    job.created_at = to
    if job.started_at is not None:
        job.started_at = to
    mgr.store.insert(job)   # INSERT OR REPLACE persists created_at; update() does not


def _child_record(records_dir, job, *, finished_at: float, logs=None) -> str:
    """A child record stamped ``checkpointed`` the way the runner's checkpoint
    trap writes one."""
    path = records_dir / f"{job.id}.json"
    path.write_text(json.dumps({
        "id": job.id, "status": "checkpointed",
        "result": {"runDirectory": "runs/z",
                   "resumeState": {"completedRecords": 3, "reason": "signal"}},
        "recordCount": 3, "finishedAt": finished_at,
        "logs": list(logs or []),
    }), encoding="utf-8")
    return str(path)


def _incident_record(tmp_path, mgr, *, slurm_id, script, age_seconds):
    """The incident's shape: a checkpointed record whose child record parked
    ``age_seconds`` ago and whose Slurm job the scheduler recorded CANCELLED
    at that same time."""
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    job = _checkpointed(mgr, slurm_id, script=script, extra_rr={
        "auto_resubmit": True, "auto_resubmit_limit": 5,
        "recordsDirectory": str(records)})
    then = time.time() - age_seconds
    _child_record(records, job, finished_at=then, logs=["child: parked"])
    return job, then


# --- 1. the observed scheduler state "cancelled" vetoes the automatic path ------

def test_scheduler_cancelled_vetoes_auto_resubmit_even_when_fold_restores_checkpointed(
        tmp_path, fake_slurm):
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    job, then = _incident_record(tmp_path, mgr, slurm_id="47001", script=script,
                                 age_seconds=49 * 24 * HOUR)
    fake_slurm.set_state("47001", "CANCELLED", exit_code="0:15",
                         end=_sacct_time(then))
    # A restarted controller adopts the record — exactly the incident.
    restarted = _manager(tmp_path)
    restarted.poll_slurm()
    assert fake_slurm.calls("sbatch") == []
    after = restarted.get(job.id)
    assert after.status == "checkpointed"        # the fold's verdict stands
    assert not (after.result or {}).get("resubmittedAs")
    logs = after.all_logs()
    assert any("the scheduler recorded Slurm job 47001 as cancelled "
               "(cancelled beats checkpointed)" in line for line in logs)
    # The observed scheduler state is durable on the record.
    stamp = after.result["lastSchedulerState"]
    assert stamp["state"] == "cancelled"
    assert stamp["endedAt"] == pytest.approx(then, abs=1.0)


def test_scheduler_cancelled_vetoes_within_the_same_instance(tmp_path, fake_slurm):
    # Lineage and age are both fine here; the scheduler's word alone decides.
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    job, then = _incident_record(tmp_path, mgr, slurm_id="47002", script=script,
                                 age_seconds=60)
    fake_slurm.set_state("47002", "CANCELLED", exit_code="0:15", end=_sacct_time(then))
    mgr.poll_slurm()
    mgr.poll_slurm()
    assert fake_slurm.calls("sbatch") == []
    assert mgr.get(job.id).status == "checkpointed"
    assert sum("cancelled beats checkpointed" in line
               for line in mgr.get(job.id).all_logs()) == 1


def test_scheduler_cancelled_without_a_child_record_simply_reads_cancelled(
        tmp_path, fake_slurm):
    # No fold to restore anything: the record becomes cancelled honestly and
    # the auto gate is never consulted (nothing is checkpointed any more).
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    job = _checkpointed(mgr, "47008", script=script, extra_rr={"auto_resubmit": True})
    fake_slurm.set_state("47008", "CANCELLED", exit_code="0:15",
                         end=_sacct_time(time.time() - 60))
    mgr.poll_slurm()
    assert fake_slurm.calls("sbatch") == []
    assert mgr.get(job.id).status == "cancelled"


# --- 2. the age bound ---------------------------------------------------------------

def test_genuinely_checkpointed_but_stale_record_is_not_resubmitted(
        tmp_path, fake_slurm, monkeypatch):
    """This instance WITNESSES the checkpoint (running → FAILED 85), but the
    scheduler recorded the exit three days ago: the controller was down and
    is only now catching up. Beyond the 48 h default it parks the record for
    a person."""
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    ended = time.time() - 72 * HOUR
    job = mgr.record_external(
        "study-submit", status="running", executor="slurm",
        executor_job_id="47003",
        requested_resources={"auto_resubmit": True, "scriptPath": script,
                             "walltime": "00:10:00"})
    _backdate(mgr, job, to=ended - HOUR)
    fake_slurm.set_state("47003", "FAILED", exit_code="85:0", end=_sacct_time(ended))
    mgr.poll_slurm()
    after = mgr.get(job.id)
    assert after.status == "checkpointed"
    assert fake_slurm.calls("sbatch") == []
    assert after.result["checkpointedAt"] == pytest.approx(ended, abs=1.0)
    stale = [line for line in after.all_logs()
             if "auto-resubmit skipped: record inactive for 72.0 hours "
                "(bound 48 hours) — resubmit manually if it should continue" in line]
    assert len(stale) == 1
    # Quiet on later ticks; the record is left exactly as it is.
    mgr.poll_slurm()
    assert sum("record inactive" in line for line in mgr.get(job.id).all_logs()) == 1
    assert fake_slurm.calls("sbatch") == []


def test_age_bound_is_configurable_per_request_and_by_env(tmp_path, fake_slurm,
                                                          monkeypatch):
    script = _dummy_script(tmp_path)
    ended = time.time() - 2 * HOUR
    # Env tightens the shipped default to one hour: a two-hour-old checkpoint
    # is stale.
    monkeypatch.setenv("STEERLAB_AUTO_RESUBMIT_MAX_AGE", str(HOUR))
    mgr = _manager(tmp_path)
    tight = mgr.record_external(
        "study-submit", status="running", executor="slurm",
        executor_job_id="47004",
        requested_resources={"auto_resubmit": True, "scriptPath": script,
                             "walltime": "00:10:00"})
    _backdate(mgr, tight, to=ended - HOUR)
    fake_slurm.set_state("47004", "FAILED", exit_code="85:0", end=_sacct_time(ended))
    mgr.poll_slurm()
    assert fake_slurm.calls("sbatch") == []
    assert any("(bound 1 hours)" in line for line in mgr.get(tight.id).all_logs())
    # The per-request key beats the env: a four-hour bound lets it through.
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "47006")
    wide = mgr.record_external(
        "study-submit", status="running", executor="slurm",
        executor_job_id="47005",
        requested_resources={"auto_resubmit": True, "scriptPath": script,
                             "walltime": "00:10:00",
                             "autoResubmitMaxAgeSeconds": 4 * HOUR})
    _backdate(mgr, wide, to=ended - HOUR)
    fake_slurm.set_state("47005", "FAILED", exit_code="85:0", end=_sacct_time(ended))
    mgr.poll_slurm()
    assert len(fake_slurm.calls("sbatch")) == 1
    assert mgr.get(wide.id).result["resubmittedAs"]


def test_malformed_age_bound_env_refuses_loudly(monkeypatch):
    monkeypatch.setenv("STEERLAB_AUTO_RESUBMIT_MAX_AGE", "two days")
    with pytest.raises(ValueError, match="STEERLAB_AUTO_RESUBMIT_MAX_AGE"):
        executors._parse_resubmit_max_age(os.environ["STEERLAB_AUTO_RESUBMIT_MAX_AGE"])
    with pytest.raises(ValueError, match="positive"):
        executors._parse_resubmit_max_age("0")
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT_MAX_AGE")
    assert executors._parse_resubmit_max_age(None) == 48 * HOUR


def test_request_key_pins_the_bound_on_the_resources(monkeypatch):
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT", raising=False)
    res = sub._resources_from_dict({"autoResubmit": True,
                                    "autoResubmitMaxAgeSeconds": 7200}, "e", "run")
    assert res.auto_resubmit_max_age_seconds == 7200.0
    assert sub._resources_from_dict({}, "e", "run").auto_resubmit_max_age_seconds is None
    with pytest.raises(ValueError, match="autoResubmitMaxAgeSeconds"):
        sub._resources_from_dict({"autoResubmitMaxAgeSeconds": -1}, "e", "run")


# --- 3. a fresh checkpoint this instance witnessed still resubmits -------------------

def test_fresh_witnessed_checkpoint_is_resubmitted(tmp_path, fake_slurm, monkeypatch):
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "47011")
    job = mgr.record_external(
        "study-submit", status="running", executor="slurm",
        executor_job_id="47010",
        requested_resources={"auto_resubmit": True, "scriptPath": script,
                             "walltime": "00:10:00"})
    fake_slurm.set_state("47010", "FAILED", exit_code="85:0",
                         end=_sacct_time(time.time() - 30))
    assert mgr.poll_slurm() >= 1
    after = mgr.get(job.id)
    assert after.status == "checkpointed"
    child = mgr.get(after.result["resubmittedAs"])
    assert child.executor_job_id == "47011"
    assert len(fake_slurm.calls("sbatch")) == 1
    assert after.result["lastSchedulerState"]["state"] == "checkpointed"


# --- 4. adoption: a new controller never auto-resubmits what it did not witness ------

def test_adopted_checkpointed_record_is_surfaced_for_manual_resume(tmp_path, fake_slurm):
    script = _dummy_script(tmp_path)
    first = _manager(tmp_path)
    job = _checkpointed(first, "47020", script=script, extra_rr={"auto_resubmit": True})
    fake_slurm.set_state("47020", "FAILED", exit_code="85:0",
                         end=_sacct_time(time.time() - 60))
    # Fresh, genuinely checkpointed, enabled — but this controller only
    # adopted it from the store.
    restarted = _manager(tmp_path)
    restarted.poll_slurm()
    restarted.poll_slurm()
    assert fake_slurm.calls("sbatch") == []
    after = restarted.get(job.id)
    assert after.status == "checkpointed"
    assert not (after.result or {}).get("resubmittedAs")
    notes = [line for line in after.all_logs()
             if "auto-resubmit skipped: this controller adopted the record "
                "already checkpointed" in line]
    assert len(notes) == 1
    assert f"POST /api/jobs/{job.id}/resubmit" in notes[0]


def test_adoption_still_repairs_a_missing_resubmitted_as_stamp(tmp_path, fake_slurm):
    script = _dummy_script(tmp_path)
    first = _manager(tmp_path)
    original = _checkpointed(first, "47030", script=script,
                             extra_rr={"auto_resubmit": True})
    child = first.record_external(
        "study-submit", status="submitted", executor="slurm",
        executor_job_id="47031",
        requested_resources={"scriptPath": script, "walltime": "00:10:00",
                             "auto_resubmit": True, "resubmitOf": original.id,
                             "resubmitChain": [original.id], "resubmitCount": 1})
    fake_slurm.set_state("47031", "RUNNING")
    restarted = _manager(tmp_path)
    restarted.poll_slurm()
    assert fake_slurm.calls("sbatch") == []
    repaired = restarted.get(original.id)
    assert repaired.result["resubmittedAs"] == child.id
    assert any("repaired the resubmittedAs stamp" in line
               for line in repaired.all_logs())


# --- 5. quiescence: the fold-restored transition is handled once -------------------

def test_fold_restored_checkpointed_over_cancelled_is_logged_once(tmp_path, fake_slurm):
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    job, then = _incident_record(tmp_path, mgr, slurm_id="47040", script=script,
                                 age_seconds=10 * 24 * HOUR)
    fake_slurm.set_state("47040", "CANCELLED", exit_code="0:15", end=_sacct_time(then))
    restarted = _manager(tmp_path)
    for _ in range(6):
        restarted.poll_slurm()
    logs = restarted.get(job.id).all_logs()
    assert sum("slurm 47040 → cancelled" in line for line in logs) == 1
    # The child's own log lines were folded once, not once per tick.
    assert sum(line == "child: parked" for line in logs) == 1
    assert fake_slurm.calls("sbatch") == []
    assert restarted.get(job.id).status == "checkpointed"


def test_a_changed_scheduler_state_is_handled_afresh(tmp_path, fake_slurm):
    """Quiescence keys on the OBSERVED state: when the scheduler's answer
    changes, the loop transitions again."""
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    job, then = _incident_record(tmp_path, mgr, slurm_id="47041", script=script,
                                 age_seconds=HOUR)
    fake_slurm.set_state("47041", "CANCELLED", exit_code="0:15", end=_sacct_time(then))
    mgr.poll_slurm()
    mgr.poll_slurm()
    fake_slurm.set_state("47041", "COMPLETED", exit_code="0:0", end=_sacct_time(then))
    mgr.poll_slurm()
    logs = mgr.get(job.id).all_logs()
    assert sum("slurm 47041 → cancelled" in line for line in logs) == 1
    assert sum("slurm 47041 → succeeded" in line for line in logs) == 1
    assert mgr.get(job.id).result["lastSchedulerState"]["state"] == "succeeded"


def test_reconcile_skips_unchanged_record_files_it_already_folded(tmp_path):
    records = tmp_path / "records"
    records.mkdir()
    mgr = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")),
                     capability_provider=lambda: {})
    job = mgr.record_external("study-submit", status="running", executor="slurm",
                              executor_job_id="47050")
    path = _child_record(records, job, finished_at=time.time(), logs=["child: parked"])
    assert mgr.reconcile(str(records)) == 1
    assert mgr.reconcile(str(records)) == 0          # same bytes: a no-op
    assert mgr.reconcile(str(records), force=True) == 1   # the operator's verb
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    data["status"] = "succeeded"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    assert mgr.reconcile(str(records)) == 1          # rewritten: folds again
    assert mgr.get(job.id).status == "succeeded"


# --- 6. the manual verb is explicit consent and keeps working ------------------------

def test_manual_resubmit_still_works_on_a_stale_adopted_record(tmp_path, fake_slurm,
                                                               monkeypatch):
    script = _dummy_script(tmp_path)
    first = _manager(tmp_path)
    job = _checkpointed(first, "47060", script=script, extra_rr={
        "auto_resubmit": True, "auto_resubmit_limit": 5})
    parked = time.time() - 60 * 24 * HOUR
    _backdate(first, job, to=parked - HOUR)
    fake_slurm.set_state("47060", "FAILED", exit_code="85:0", end=_sacct_time(parked))
    restarted = _manager(tmp_path)
    restarted.poll_slurm()
    assert fake_slurm.calls("sbatch") == []          # automatic path refused
    assert any("record inactive" in line for line in restarted.get(job.id).all_logs())
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "47061")
    outcome = restarted.resubmit(job.id)
    assert outcome["ok"] is True
    assert len(fake_slurm.calls("sbatch")) == 1
    child = restarted.get(outcome["jobId"])
    assert child.requested_resources["manualResubmit"] is True
    assert restarted.get(job.id).result["resubmittedAs"] == child.id


def test_manual_resubmit_keeps_refusing_a_scheduler_cancelled_record_only_when_terminal(
        tmp_path, fake_slurm):
    """The manual gate is unchanged: a record the fold left ``checkpointed``
    is still resumable by a person, and a record that READS cancelled is
    still refused — neither side of the manual verb widened."""
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    plain = _checkpointed(mgr, "47070", script=script, status="cancelled")
    with pytest.raises(ResubmitRefused, match="already finished"):
        mgr.resubmit(plain.id)


# --- executor: sacct's End column rides the observation ------------------------------

def test_poll_observation_reads_sacct_end_time(fake_slurm):
    ended = time.time() - 3 * HOUR
    fake_slurm.set_state("555", "FAILED", exit_code="85:0", end=_sacct_time(ended))
    found = SlurmExecutor().poll_observation("555")
    assert found.state == "checkpointed"
    assert found.query_ok is True
    assert found.raw_state == "FAILED"
    assert found.ended_at == pytest.approx(ended, abs=1.0)
    # Without an End column (older wrappers, running jobs) the time is None
    # and the state still reads through.
    fake_slurm.set_state("556", "RUNNING")
    assert SlurmExecutor().poll_observation("556").ended_at is None
    assert executors._parse_sacct_time("Unknown") is None
    assert executors._parse_sacct_time("") is None
