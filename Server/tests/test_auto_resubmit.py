"""Auto-resubmit of checkpointed Slurm jobs (TURNKEY-CLUSTER-PLAN WS2).

A job that CHECKPOINTS (exit 85 → status "checkpointed", with resume-state and
the requeue-safe pointer already landed) is resubmitted by the reconciler
itself when auto-resubmit is enabled: the SAME run.sbatch re-executes verbatim
and the resume pointer continues the run. Properties under test:

- off by default; enabled by STEERLAB_AUTO_RESUBMIT or per-request
  ``resources.autoResubmit`` (per-request wins over env, in both directions);
- the resubmission is a NEW job record carrying resubmitOf / resubmitChain /
  resubmitCount, the original gains result.resubmittedAs + a loud log line;
- bounded by a chain cap counted from the ROOT job;
- at most ONE resubmission per checkpoint across ticks and restarts (the
  submit-first-then-stamp crash is repaired, never double-submitted);
- deferred loudly (not dropped) across maintenance windows;
- cancelling the original cascades down the chain, and a cancelled job is
  never auto-resubmitted (cancelled beats checkpointed).
"""

import json
import os
import shutil
import stat
from datetime import datetime, timedelta, timezone

import pytest

from steerlab_server.api import submissions as sub
from steerlab_server.api.executors import SlurmResources
from steerlab_server.api.jobs import DurableJobStore, JobManager
from steerlab_server.api.profile import capability_snapshot
from steerlab_server.api.submissions import submit_study
from steerlab_server.experiment import experiment_store as es

FAKEBIN_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fakebin")


@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    """The committed fakebin doubles on PATH (see test_fake_scheduler), plus
    isolated metadata root and a clean auto-resubmit env."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        target = bindir / name
        shutil.copy(os.path.join(FAKEBIN_SOURCE, name), target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log_dir = tmp_path / "calls"
    log_dir.mkdir()
    state_file = tmp_path / "slurm-state.json"
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_SLURM_LOG", str(log_dir))
    monkeypatch.setenv("FAKE_SLURM_STATE_FILE", str(state_file))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    monkeypatch.delenv("FAKE_SBATCH_FAIL", raising=False)
    monkeypatch.delenv("FAKE_SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("STEERLAB_MAINTENANCE_CALENDAR", raising=False)
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT", raising=False)
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT_LIMIT", raising=False)
    monkeypatch.delenv("STEERLAB_SLURM_SACCT", raising=False)
    monkeypatch.delenv("STEERLAB_SLURM_SQUEUE", raising=False)

    class Handle:
        def set_state(self, job_id, state, exit_code="0:0", queue=None):
            table = {}
            if state_file.exists():
                table = json.loads(state_file.read_text(encoding="utf-8"))
            table[str(job_id)] = {"state": state, "exit": exit_code, "queue": queue}
            state_file.write_text(json.dumps(table), encoding="utf-8")

        def calls(self, binary):
            path = log_dir / f"{binary}.calls"
            if not path.exists():
                return []
            return path.read_text(encoding="utf-8").splitlines()

    return Handle()


@pytest.fixture
def study_site(tmp_path, monkeypatch, fake_slurm):
    """A submittable study workspace with the Slurm executor declared."""
    root = str(tmp_path / "source")
    concept = os.path.join(root, "prompts", "concepts", "fair")
    os.makedirs(concept, exist_ok=True)
    open(os.path.join(concept, "positive.jsonl"), "w", encoding="utf-8").write(
        '{"text":"fair"}\n')
    open(os.path.join(concept, "negative.jsonl"), "w", encoding="utf-8").write(
        '{"text":"unfair"}\n')
    es.create("Auto Study", model_id="org/m", revision="abc", root=root)
    es.attach("auto-study", ["fair"], root=root)
    monkeypatch.setenv("STEERLAB_ROOT", root)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    return root


def _manager(tmp_path, name="jobs.sqlite"):
    return JobManager(DurableJobStore(str(tmp_path / name)),
                      capability_provider=lambda: {})


def _submit(jobs, resources=None):
    return submit_study("auto-study", verb="run", jobs=jobs, executor="slurm",
                        resources={"gres": "A100", "walltime": "00:30:00",
                                   **(resources or {})})


def _dummy_script(tmp_path, name="run.sbatch"):
    bundle_dir = tmp_path / "fabricated-bundle"
    bundle_dir.mkdir(exist_ok=True)
    script = bundle_dir / name
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    return str(script)


def _checkpointed(mgr, slurm_id, *, script, extra_rr=None, status="checkpointed"):
    """A fabricated checkpointed Slurm job carrying the durable stamps a real
    submission would have (scriptPath / walltime in requestedResources)."""
    rr = {"scriptPath": script, "walltime": "00:10:00"}
    rr.update(extra_rr or {})
    return mgr.record_external("study-submit", status=status, executor="slurm",
                               executor_job_id=str(slurm_id),
                               requested_resources=rr)


# --- 1. enabled via env: full flow through submit_study + fakebin ---------------

def test_env_enabled_resubmits_same_script_with_chain_fields(
        tmp_path, study_site, fake_slurm, monkeypatch):
    monkeypatch.setenv("STEERLAB_AUTO_RESUBMIT", "1")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9001")
    jobs = _manager(tmp_path)
    submission = _submit(jobs)
    assert submission.slurm_job_id == "9001"
    original = jobs.get(submission.job_id)
    assert original.requested_resources["auto_resubmit"] is True
    assert (original.requested_resources["scriptPath"]
            == submission.slurm_bundle["script_path"])
    assert (original.requested_resources["recordsDirectory"]
            == submission.records_directory)

    fake_slurm.set_state("9001", "FAILED", exit_code="85:0")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9002")
    assert jobs.poll_slurm() >= 1

    original = jobs.get(submission.job_id)
    assert original.status == "checkpointed"
    child_id = (original.result or {}).get("resubmittedAs")
    assert child_id, "original must be stamped resubmittedAs"
    child = jobs.get(child_id)
    assert child is not None
    assert child.executor == "slurm"
    assert child.executor_job_id == "9002"
    assert child.status == "submitted"
    rr = child.requested_resources
    assert rr["resubmitOf"] == original.id
    assert rr["resubmitChain"] == [original.id]
    assert rr["resubmitCount"] == 1
    # The child's stamped result lets ITS terminal fold find the records dir.
    assert child.result["recordsDirectory"] == submission.records_directory

    calls = fake_slurm.calls("sbatch")
    assert len(calls) == 2
    # The SAME sbatch script, verbatim — the resubmission adds only the
    # durable submission token as the job name (resume crash-safety,
    # 2026-07-23: squeue/sacct --name can find this exact submission if the
    # controller dies before writing the continuation record).
    assert calls[0].split()[-1] == calls[1].split()[-1]
    assert "--job-name=slre-" in calls[1]
    assert submission.slurm_bundle["script_path"] in calls[1]
    assert (child.requested_resources.get("submissionToken") or "").startswith(
        "slre-")
    assert any(f"checkpointed — auto-resubmitted as {child_id} (1/5)" in line
               for line in original.all_logs())

    # Idempotent across further ticks: exactly one resubmission per checkpoint.
    fake_slurm.set_state("9002", "RUNNING")
    jobs.poll_slurm()
    assert len(fake_slurm.calls("sbatch")) == 2
    # ...and across a server restart (fresh manager over the same store).
    restarted = _manager(tmp_path)
    restarted.poll_slurm()
    assert len(fake_slurm.calls("sbatch")) == 2


# --- 2. off by default -----------------------------------------------------------

def test_disabled_by_default_stays_checkpointed_without_resubmit(
        tmp_path, study_site, fake_slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9101")
    jobs = _manager(tmp_path)
    submission = _submit(jobs)
    fake_slurm.set_state("9101", "FAILED", exit_code="85:0")
    jobs.poll_slurm()
    jobs.poll_slurm()
    job = jobs.get(submission.job_id)
    assert job.status == "checkpointed"
    assert not (job.result or {}).get("resubmittedAs")
    assert len(fake_slurm.calls("sbatch")) == 1  # only the original submission
    # Disabled is silent — the existing checkpoint guidance line is the only
    # resubmission-related message.
    assert not any("auto-resubmit" in line for line in job.all_logs())


# --- 3. per-request beats env, both directions -----------------------------------

def test_per_request_true_overrides_env_off(tmp_path, study_site, fake_slurm,
                                            monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9201")
    jobs = _manager(tmp_path)
    submission = _submit(jobs, resources={"autoResubmit": True})
    assert jobs.get(submission.job_id).requested_resources["auto_resubmit"] is True
    fake_slurm.set_state("9201", "FAILED", exit_code="85:0")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9202")
    jobs.poll_slurm()
    job = jobs.get(submission.job_id)
    assert (job.result or {}).get("resubmittedAs")
    assert len(fake_slurm.calls("sbatch")) == 2


def test_per_request_false_beats_env_on(tmp_path, study_site, fake_slurm,
                                        monkeypatch):
    monkeypatch.setenv("STEERLAB_AUTO_RESUBMIT", "1")  # on at submit AND reconcile
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9301")
    jobs = _manager(tmp_path)
    submission = _submit(jobs, resources={"autoResubmit": False})
    assert jobs.get(submission.job_id).requested_resources["auto_resubmit"] is False
    fake_slurm.set_state("9301", "FAILED", exit_code="85:0")
    jobs.poll_slurm()
    job = jobs.get(submission.job_id)
    assert job.status == "checkpointed"
    assert not (job.result or {}).get("resubmittedAs")
    assert len(fake_slurm.calls("sbatch")) == 1


# --- 4. the chain cap, counted from the root --------------------------------------

def test_chain_at_cap_refuses_loudly_and_leaves_checkpointed(tmp_path, fake_slurm):
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    capped = _checkpointed(mgr, "9401", script=script, extra_rr={
        "auto_resubmit": True, "auto_resubmit_limit": 2,
        "resubmitOf": "parent1", "resubmitChain": ["root0", "parent1"],
        "resubmitCount": 2})
    mgr.poll_slurm()
    assert fake_slurm.calls("sbatch") == []
    job = mgr.get(capped.id)
    assert job.status == "checkpointed"
    assert not (job.result or {}).get("resubmittedAs")
    limit_lines = [line for line in job.all_logs()
                   if "auto-resubmit limit reached (2) — resubmit manually" in line]
    assert len(limit_lines) == 1
    # A later tick with the identical situation stays quiet (loud ≠ spam).
    mgr.poll_slurm()
    assert sum("limit reached" in line for line in mgr.get(capped.id).all_logs()) == 1
    assert fake_slurm.calls("sbatch") == []


def test_below_cap_resubmits_and_extends_the_chain(tmp_path, fake_slurm, monkeypatch):
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9451")
    mid = _checkpointed(mgr, "9450", script=script, extra_rr={
        "auto_resubmit": True, "auto_resubmit_limit": 2,
        "resubmitOf": "root0", "resubmitChain": ["root0"], "resubmitCount": 1})
    mgr.poll_slurm()
    assert len(fake_slurm.calls("sbatch")) == 1
    job = mgr.get(mid.id)
    child = mgr.get(job.result["resubmittedAs"])
    assert child.requested_resources["resubmitChain"] == ["root0", mid.id]
    assert child.requested_resources["resubmitCount"] == 2
    assert child.requested_resources["resubmitOf"] == mid.id
    assert any(f"auto-resubmitted as {child.id} (2/2)" in line
               for line in job.all_logs())


# --- 5. restart/race repair: submitted but not stamped ----------------------------

def test_repair_stamps_existing_child_instead_of_double_submitting(
        tmp_path, fake_slurm):
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    # The crash window: sbatch succeeded (child job record exists, live) but
    # the server died before stamping resubmittedAs on the original.
    original = _checkpointed(mgr, "9501", script=script,
                             extra_rr={"auto_resubmit": True})
    child = mgr.record_external(
        "study-submit", status="submitted", executor="slurm",
        executor_job_id="9502",
        requested_resources={"scriptPath": script, "walltime": "00:10:00",
                             "auto_resubmit": True, "resubmitOf": original.id,
                             "resubmitChain": [original.id], "resubmitCount": 1})
    fake_slurm.set_state("9502", "RUNNING")

    mgr.poll_slurm()
    assert fake_slurm.calls("sbatch") == []  # repaired, never re-submitted
    repaired = mgr.get(original.id)
    assert repaired.result["resubmittedAs"] == child.id
    assert any("repaired the resubmittedAs stamp" in line
               for line in repaired.all_logs())

    # A restart replays the same reconcile: still zero submissions, stamp keeps.
    restarted = _manager(tmp_path)
    restarted.poll_slurm()
    assert fake_slurm.calls("sbatch") == []
    assert restarted.get(original.id).result["resubmittedAs"] == child.id


# --- 6. cancelling the original kills the chain -----------------------------------

def test_cancel_cascades_to_newest_live_descendant(tmp_path, fake_slurm):
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    root = _checkpointed(mgr, "9601", script=script,
                         extra_rr={"auto_resubmit": True})
    mid = _checkpointed(mgr, "9602", script=script, extra_rr={
        "auto_resubmit": True, "resubmitOf": root.id,
        "resubmitChain": [root.id], "resubmitCount": 1})
    tip = mgr.record_external(
        "study-submit", status="running", executor="slurm",
        executor_job_id="9603",
        requested_resources={"scriptPath": script, "walltime": "00:10:00",
                             "auto_resubmit": True, "resubmitOf": mid.id,
                             "resubmitChain": [root.id, mid.id],
                             "resubmitCount": 2})
    root.result = {"resubmittedAs": mid.id}
    mgr.store.update(root)
    mid.result = {"resubmittedAs": tip.id}
    mgr.store.update(mid)

    assert mgr.cancel(root.id) is True
    assert fake_slurm.calls("scancel") == ["9601", "9602", "9603"]
    assert mgr.get(root.id).status == "cancelled"
    assert mgr.get(mid.id).status == "cancelled"
    assert mgr.get(tip.id).status == "cancelled"
    assert any(f"cancelled with ancestor {root.id}" in line
               for line in mgr.get(tip.id).all_logs())
    # Nothing in the killed chain ever auto-resubmits.
    mgr.poll_slurm()
    assert fake_slurm.calls("sbatch") == []


# --- 7. maintenance window: deferred, then submitted when clear --------------------

def _calendar(tmp_path, start_in_minutes, duration_hours=4):
    start = datetime.now(timezone.utc) + timedelta(minutes=start_in_minutes)
    end = start + timedelta(hours=duration_hours)
    path = tmp_path / "maintenance.json"
    path.write_text(json.dumps([{"start": start.isoformat(),
                                 "end": end.isoformat()}]), encoding="utf-8")
    return path


def test_resubmit_deferred_across_maintenance_window_then_submitted(
        tmp_path, fake_slurm, monkeypatch):
    calendar = _calendar(tmp_path, start_in_minutes=30)
    monkeypatch.setenv("STEERLAB_MAINTENANCE_CALENDAR", str(calendar))
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    job = _checkpointed(mgr, "9701", script=script, extra_rr={
        "auto_resubmit": True, "walltime": "04:00:00"})  # crosses the window

    mgr.poll_slurm()
    assert fake_slurm.calls("sbatch") == []  # deferred, not submitted
    deferral_lines = [line for line in mgr.get(job.id).all_logs()
                      if "auto-resubmit deferred" in line
                      and "maintenance window" in line]
    assert len(deferral_lines) == 1
    # Still blocked on the next tick: no submission, no log spam.
    mgr.poll_slurm()
    assert fake_slurm.calls("sbatch") == []
    assert sum("auto-resubmit deferred" in line
               for line in mgr.get(job.id).all_logs()) == 1

    # The window clears (calendar rewritten, as the housekeeping editor would).
    calendar.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9702")
    mgr.poll_slurm()
    calls = fake_slurm.calls("sbatch")
    assert len(calls) == 1 and script in calls[0]
    resubmitted = mgr.get(job.id)
    child = mgr.get(resubmitted.result["resubmittedAs"])
    assert child.executor_job_id == "9702"
    assert any("(1/5)" in line for line in resubmitted.all_logs())


# --- 8. cancelled is never auto-resubmitted ----------------------------------------

def test_cancelled_job_is_never_auto_resubmitted(tmp_path, fake_slurm, monkeypatch):
    monkeypatch.setenv("STEERLAB_AUTO_RESUBMIT", "1")
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    job = _checkpointed(mgr, "9801", script=script, extra_rr={"auto_resubmit": True})
    assert mgr.cancel(job.id) is True
    # scancel through the scheduler and the honest terminal state: cancelled,
    # not checkpointed — the state precedence under test.
    assert "9801" in fake_slurm.calls("scancel")
    assert mgr.get(job.id).status == "cancelled"
    mgr.poll_slurm()
    mgr.poll_slurm()
    assert fake_slurm.calls("sbatch") == []
    assert not (mgr.get(job.id).result or {}).get("resubmittedAs")


def test_cancel_flag_beats_checkpointed_status_even_unstamped(tmp_path, fake_slurm):
    # The precedence race: a cancel was requested but a late child-record fold
    # (or a crash) left the status reading "checkpointed". The persisted cancel
    # flag must still veto auto-resubmission.
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    job = _checkpointed(mgr, "9851", script=script, extra_rr={"auto_resubmit": True})
    job._cancel.set()
    mgr.store.mark_cancel_requested(job.id)
    mgr.poll_slurm()
    assert fake_slurm.calls("sbatch") == []
    assert mgr.get(job.id).status == "checkpointed"
    assert any("cancellation was requested" in line
               for line in mgr.get(job.id).all_logs())
    # The veto survives a restart (the flag is persisted in the store).
    restarted = _manager(tmp_path)
    restarted.poll_slurm()
    assert fake_slurm.calls("sbatch") == []


# --- the child-record trigger (sacct masked the 85) --------------------------------

def test_child_record_checkpoint_triggers_resubmit_when_sacct_reports_failed(
        tmp_path, fake_slurm, monkeypatch):
    """The AND clause: a transition to "checkpointed" via a folded child record
    (wrapper reported plain FAILED 1:0) must auto-resubmit exactly like the
    sacct 85-exit mapping. Also covers the camelCase resource alias."""
    script = _dummy_script(tmp_path)
    records = tmp_path / "records"
    records.mkdir()
    mgr = _manager(tmp_path)
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9902")
    job = mgr.record_external(
        "study-submit", status="running", executor="slurm",
        executor_job_id="9901",
        requested_resources={"autoResubmit": True, "scriptPath": script,
                             "walltime": "00:10:00",
                             "recordsDirectory": str(records)},
        result={"recordsDirectory": str(records)})
    (records / f"{job.id}.json").write_text(json.dumps({
        "id": job.id, "status": "checkpointed",
        "result": {"runDirectory": "runs/z",
                   "resumeState": {"completedRecords": 3, "reason": "signal"}},
        "recordCount": 3,
    }), encoding="utf-8")
    fake_slurm.set_state("9901", "FAILED", exit_code="1:0")  # 85 masked

    assert mgr.poll_slurm() >= 1
    folded = mgr.get(job.id)
    assert folded.status == "checkpointed"
    assert folded.finished_at is None  # non-terminal invariant kept after fold
    child = mgr.get(folded.result["resubmittedAs"])
    assert child is not None
    assert child.executor_job_id == "9902"
    assert child.requested_resources["resubmitOf"] == job.id
    assert len(fake_slurm.calls("sbatch")) == 1


# --- units: env/per-request wiring + capability flag --------------------------------

def test_slurm_resources_env_wiring(monkeypatch):
    monkeypatch.setenv("STEERLAB_AUTO_RESUBMIT", "1")
    monkeypatch.setenv("STEERLAB_AUTO_RESUBMIT_LIMIT", "7")
    res = SlurmResources.from_env()
    assert res.auto_resubmit is True
    assert res.auto_resubmit_limit == 7
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT")
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT_LIMIT")
    res = SlurmResources.from_env()
    assert res.auto_resubmit is False   # off by default
    assert res.auto_resubmit_limit == 5
    monkeypatch.setenv("STEERLAB_AUTO_RESUBMIT_LIMIT", "many")
    with pytest.raises(ValueError, match="STEERLAB_AUTO_RESUBMIT_LIMIT"):
        SlurmResources.from_env()


def test_resources_from_dict_per_request_wins_both_directions(monkeypatch):
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT", raising=False)
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT_LIMIT", raising=False)
    res = sub._resources_from_dict({"autoResubmit": True, "autoResubmitLimit": 9},
                                   "e", "run")
    assert res.auto_resubmit is True and res.auto_resubmit_limit == 9
    monkeypatch.setenv("STEERLAB_AUTO_RESUBMIT", "1")
    res = sub._resources_from_dict({"autoResubmit": False}, "e", "run")
    assert res.auto_resubmit is False   # explicit request beats env-on
    res = sub._resources_from_dict({}, "e", "run")
    assert res.auto_resubmit is True    # absent request falls back to env


def test_capability_snapshot_advertises_auto_resubmit():
    assert capability_snapshot()["cluster"]["autoResubmit"] is True


# --- record-less terminal deaths get a visible reason (2026-08-03) -----------

def test_a_recordless_failure_is_stamped_with_the_scheduler_reason(
        tmp_path, study_site, fake_slurm, monkeypatch):
    """Field incident 2026-08-02: a sweep hit its 30-minute walltime, the
    child was killed before writing its record, and the app showed a bare
    failure with no reason anywhere. The reconciler now asks sacct HOW the
    scheduler ended a record-less terminal job and stamps that as the
    error."""
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9401")
    jobs = _manager(tmp_path)
    submission = _submit(jobs)
    # The scheduler kills the job (walltime); the child wrote NO record.
    fake_slurm.set_state("9401", "TIMEOUT", exit_code="0:1")
    jobs.poll_slurm()
    job = jobs.get(submission.job_id)
    assert job.status == "failed"
    error = (job.result or {}).get("error") or ""
    assert "Slurm reported TIMEOUT" in error
    assert "wrote no completion record" in error
    assert "raise the submission's walltime" in error


def test_a_succeeded_job_gets_no_synthetic_error(
        tmp_path, study_site, fake_slurm, monkeypatch):
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "9402")
    jobs = _manager(tmp_path)
    submission = _submit(jobs)
    fake_slurm.set_state("9402", "COMPLETED", exit_code="0:0")
    jobs.poll_slurm()
    job = jobs.get(submission.job_id)
    assert job.status == "succeeded"
    assert not (job.result or {}).get("error")
