"""Manual resume of checkpointed jobs (POST /api/jobs/{id}/resubmit).

Live incident 2026-07-22: a frozen pipeline checkpointed cleanly at the
walltime margin (exit 85), the job log honestly said "requeue or resubmit to
continue it" — and neither the app nor the API offered a way to do either.
This file pins the manual half of the remedy:

- happy path: the job's OWN run.sbatch re-executes (the SAME implementation
  auto-resubmit uses — one core, two gates), the continuation record carries
  resubmitOf/resubmitChain/resubmitCount plus the ``manualResubmit: true``
  consent stamp, the original gains result.resubmittedAs, and the new Slurm
  id comes back to the caller;
- refusals are plain-language 409s for every non-resumable state (running,
  terminal, cancelled, already-resubmitted, missing script) and 404 for an
  unknown id;
- a human click may EXCEED the auto-resubmit chain cap (explicit consent,
  logged) — the reconciler's own limit gate is unchanged;
- the route is privileged exactly like its mutating siblings (token on Slurm
  deployments / token mode), with the 503/401 shapes those siblings answer.
"""

import json
import os
import shutil
import stat

import pytest

from steerlab_server.api.jobs import (DurableJobStore, JobManager,
                                      ResubmitRefused)

FAKEBIN_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fakebin")


@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    """The committed fakebin doubles on PATH (same shape as
    test_auto_resubmit's fixture), isolated metadata root, clean env."""
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


def _manager(tmp_path, name="jobs.sqlite"):
    return JobManager(DurableJobStore(str(tmp_path / name)),
                      capability_provider=lambda: {})


def _dummy_script(tmp_path, name="run.sbatch"):
    bundle_dir = tmp_path / "fabricated-bundle"
    bundle_dir.mkdir(exist_ok=True)
    script = bundle_dir / name
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    return str(script)


def _checkpointed(mgr, slurm_id, *, script, extra_rr=None, status="checkpointed"):
    rr = {"scriptPath": script, "walltime": "00:10:00"}
    rr.update(extra_rr or {})
    return mgr.record_external("study-submit", status=status, executor="slurm",
                               executor_job_id=str(slurm_id),
                               requested_resources=rr)


# --- manager-level happy path -----------------------------------------------------

def test_manual_resubmit_reuses_the_auto_core_and_stamps_consent(
        tmp_path, fake_slurm, monkeypatch):
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "7002")
    job = _checkpointed(mgr, "7001", script=script,
                        extra_rr={"recordsDirectory": str(tmp_path / "records")})

    outcome = mgr.resubmit(job.id)

    assert outcome["ok"] is True
    assert outcome["slurmJobID"] == "7002"
    assert outcome["manualResubmit"] is True
    assert outcome["resubmitCount"] == 1
    assert outcome["resubmitOf"] == job.id
    child = mgr.get(outcome["jobId"])
    assert child is not None
    assert child.executor == "slurm"
    assert child.executor_job_id == "7002"
    assert child.status == "submitted"
    rr = child.requested_resources
    assert rr["resubmitOf"] == job.id
    assert rr["resubmitChain"] == [job.id]
    assert rr["resubmitCount"] == 1
    # THE consent stamp: manual, distinct from reconciler auto-resubmits.
    assert rr["manualResubmit"] is True
    # The child's stamped result lets its terminal fold find the records dir.
    assert child.result["recordsDirectory"] == str(tmp_path / "records")
    # The original is stamped and loudly logged.
    original = mgr.get(job.id)
    assert original.result["resubmittedAs"] == child.id
    assert any("manually resubmitted as" in line for line in original.all_logs())
    # The SAME sbatch script, verbatim — auto's implementation, not a fork.
    calls = fake_slurm.calls("sbatch")
    assert len(calls) == 1 and script in calls[0]


def test_manual_resubmit_allowed_past_the_auto_limit(tmp_path, fake_slurm,
                                                     monkeypatch):
    """count >= limit refuses the RECONCILER (unchanged) but a human click is
    explicit consent: manual proceeds, logs the exceedance."""
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "7102")
    job = _checkpointed(mgr, "7101", script=script, extra_rr={
        "auto_resubmit": True, "auto_resubmit_limit": 2,
        "resubmitChain": ["r0", "p1"], "resubmitCount": 2})
    # The reconciler refuses at the cap...
    mgr.poll_slurm()
    assert fake_slurm.calls("sbatch") == []
    assert any("limit reached" in line for line in mgr.get(job.id).all_logs())
    # ...the human does not.
    outcome = mgr.resubmit(job.id)
    assert outcome["resubmitCount"] == 3
    assert outcome["autoResubmitLimit"] == 2
    assert len(fake_slurm.calls("sbatch")) == 1
    assert any("explicit consent" in line for line in mgr.get(job.id).all_logs())
    child = mgr.get(outcome["jobId"])
    assert child.requested_resources["manualResubmit"] is True
    assert child.requested_resources["resubmitChain"] == ["r0", "p1", job.id]


# --- manager-level refusals -------------------------------------------------------

def test_manual_resubmit_refuses_every_non_resumable_state(tmp_path, fake_slurm):
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    running = _checkpointed(mgr, "7201", script=script, status="running")
    done = _checkpointed(mgr, "7202", script=script, status="succeeded")
    failed = _checkpointed(mgr, "7203", script=script, status="failed")

    with pytest.raises(ResubmitRefused, match="still with the scheduler"):
        mgr.resubmit(running.id)
    with pytest.raises(ResubmitRefused, match="already finished"):
        mgr.resubmit(done.id)
    with pytest.raises(ResubmitRefused, match="already finished"):
        mgr.resubmit(failed.id)
    with pytest.raises(ResubmitRefused, match="no job"):
        mgr.resubmit("nonexistent")
    assert fake_slurm.calls("sbatch") == []


def test_manual_resubmit_refuses_cancelled_and_already_resubmitted(
        tmp_path, fake_slurm):
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    cancelled = _checkpointed(mgr, "7301", script=script)
    cancelled._cancel.set()
    mgr.store.mark_cancel_requested(cancelled.id)
    with pytest.raises(ResubmitRefused, match="cancelled beats checkpointed"):
        mgr.resubmit(cancelled.id)

    stamped = _checkpointed(mgr, "7302", script=script)
    stamped.result = {"resubmittedAs": "child99"}
    mgr.store.update(stamped)
    with pytest.raises(ResubmitRefused, match="already resubmitted as child99"):
        mgr.resubmit(stamped.id)
    assert fake_slurm.calls("sbatch") == []


def test_manual_resubmit_repairs_an_unstamped_existing_child(tmp_path, fake_slurm):
    """The submit-then-stamp crash window: a live continuation exists but the
    marker is missing. Resume must repair the stamp and refuse — never race a
    second sbatch."""
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    original = _checkpointed(mgr, "7401", script=script)
    child = mgr.record_external(
        "study-submit", status="submitted", executor="slurm",
        executor_job_id="7402",
        requested_resources={"scriptPath": script, "walltime": "00:10:00",
                             "resubmitOf": original.id,
                             "resubmitChain": [original.id], "resubmitCount": 1})
    with pytest.raises(ResubmitRefused, match=f"already resubmitted as {child.id}"):
        mgr.resubmit(original.id)
    assert mgr.get(original.id).result["resubmittedAs"] == child.id
    assert fake_slurm.calls("sbatch") == []


def test_manual_resubmit_refuses_missing_script(tmp_path, fake_slurm):
    mgr = _manager(tmp_path)
    job = _checkpointed(mgr, "7501", script=str(tmp_path / "gone.sbatch"))
    with pytest.raises(ResubmitRefused, match="not on record or not on disk"):
        mgr.resubmit(job.id)
    assert fake_slurm.calls("sbatch") == []


def test_manual_resubmit_refuses_across_maintenance_window(tmp_path, fake_slurm,
                                                           monkeypatch):
    from datetime import datetime, timedelta, timezone
    start = datetime.now(timezone.utc) + timedelta(minutes=5)
    end = start + timedelta(hours=4)
    calendar = tmp_path / "maintenance.json"
    calendar.write_text(json.dumps([{"start": start.isoformat(),
                                     "end": end.isoformat()}]), encoding="utf-8")
    monkeypatch.setenv("STEERLAB_MAINTENANCE_CALENDAR", str(calendar))
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    job = _checkpointed(mgr, "7601", script=script,
                        extra_rr={"walltime": "04:00:00"})
    with pytest.raises(ResubmitRefused, match="maintenance window"):
        mgr.resubmit(job.id)
    assert fake_slurm.calls("sbatch") == []


def test_manual_then_auto_never_double_submits(tmp_path, fake_slurm, monkeypatch):
    """After a manual resume the reconciler must treat the checkpoint as
    handled — the resubmittedAs stamp gates the auto path."""
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "7702")
    job = _checkpointed(mgr, "7701", script=script,
                        extra_rr={"auto_resubmit": True})
    mgr.resubmit(job.id)
    assert len(fake_slurm.calls("sbatch")) == 1
    fake_slurm.set_state("7701", "FAILED", exit_code="85:0")
    mgr.poll_slurm()
    mgr.poll_slurm()
    assert len(fake_slurm.calls("sbatch")) == 1


# --- walltime override (field incident 2026-08-29) ---------------------------------
# A shard that checkpointed AT its walltime would only checkpoint again under
# the same limit. The override rides sbatch's COMMAND LINE (flag beats
# `#SBATCH` header), so the rendered script is still re-submitted
# byte-for-byte — the reliability contract's identical-script rule holds.

def test_manual_resubmit_walltime_override_rides_the_command_line(
        tmp_path, fake_slurm, monkeypatch):
    script = _dummy_script(tmp_path)
    script_bytes_before = open(script, "rb").read()
    mgr = _manager(tmp_path)
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "8102")
    job = _checkpointed(mgr, "8101", script=script)

    outcome = mgr.resubmit(job.id, walltime="08:00:00")

    assert outcome["ok"] is True
    assert outcome["slurmJobID"] == "8102"
    # The echo is the caller's proof the server understood the override.
    assert outcome["walltime"] == "08:00:00"
    calls = fake_slurm.calls("sbatch")
    assert len(calls) == 1
    # The override is a CLI flag; the submitted script is the original path
    # with its bytes untouched — never a re-render.
    assert "--time=08:00:00" in calls[0]
    assert calls[0].endswith(script)
    assert open(script, "rb").read() == script_bytes_before
    # The continuation's durable record prices the limit it actually runs
    # under, and says the limit was an override.
    child = mgr.get(outcome["jobId"])
    assert child.requested_resources["walltime"] == "08:00:00"
    assert child.requested_resources["walltimeOverride"] == "08:00:00"
    assert any("walltime raised to 08:00:00" in line
               for line in child.all_logs())


def test_manual_resubmit_without_override_keeps_the_command_line_bare(
        tmp_path, fake_slurm, monkeypatch):
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "8202")
    job = _checkpointed(mgr, "8201", script=script)
    outcome = mgr.resubmit(job.id)
    assert "walltime" not in outcome
    calls = fake_slurm.calls("sbatch")
    assert len(calls) == 1 and "--time=" not in calls[0]
    child = mgr.get(outcome["jobId"])
    assert "walltimeOverride" not in child.requested_resources


def test_manual_resubmit_refuses_a_malformed_walltime(tmp_path, fake_slurm):
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    job = _checkpointed(mgr, "8301", script=script)
    with pytest.raises(ResubmitRefused, match="invalid walltime"):
        mgr.resubmit(job.id, walltime="8 hours")
    with pytest.raises(ResubmitRefused, match="must be positive"):
        mgr.resubmit(job.id, walltime="0")
    assert fake_slurm.calls("sbatch") == []


# --- the scheduler's exit code beats a stale record (field incident 2026-08-29) ----

def test_manual_resubmit_corrects_a_record_whose_exit_code_is_85(
        tmp_path, fake_slurm, monkeypatch):
    """A checkpoint the record missed — 'failed' from a wrapper that reported
    the checkpoint exit as a plain failure, or 'running' with no reconciler
    tick yet — is re-asked of the scheduler before refusing: sacct's ExitCode
    85 means the run parked resumably, and the resume proceeds."""
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "8402")
    stale_failed = _checkpointed(mgr, "8401", script=script, status="failed")
    fake_slurm.set_state("8401", "FAILED", exit_code="85:0")

    outcome = mgr.resubmit(stale_failed.id)

    assert outcome["ok"] is True
    assert outcome["slurmJobID"] == "8402"
    original = mgr.get(stale_failed.id)
    assert original.status == "checkpointed"
    assert original.finished_at is None
    assert any("record corrected to checkpointed" in line
               for line in original.all_logs())
    assert len(fake_slurm.calls("sbatch")) == 1


def test_manual_resubmit_still_refuses_a_truly_failed_record(
        tmp_path, fake_slurm):
    """The correction is for the checkpoint exit code ONLY — a job the
    scheduler confirms failed stays refused, and succeeded/cancelled records
    are never even re-asked (cancelled beats checkpointed)."""
    script = _dummy_script(tmp_path)
    mgr = _manager(tmp_path)
    hard_failed = _checkpointed(mgr, "8501", script=script, status="failed")
    fake_slurm.set_state("8501", "FAILED", exit_code="1:0")
    with pytest.raises(ResubmitRefused, match="already finished"):
        mgr.resubmit(hard_failed.id)

    done = _checkpointed(mgr, "8502", script=script, status="succeeded")
    fake_slurm.set_state("8502", "FAILED", exit_code="85:0")
    with pytest.raises(ResubmitRefused, match="already finished"):
        mgr.resubmit(done.id)
    # Succeeded was refused WITHOUT a scheduler round trip.
    assert not any("-j 8502" in call for call in fake_slurm.calls("sacct"))
    assert fake_slurm.calls("sbatch") == []


# --- the HTTP route ---------------------------------------------------------------

def _route_client(tmp_path, monkeypatch):
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
    return TestClient(app), state


def test_resubmit_route_happy_path_returns_new_slurm_id(tmp_path, fake_slurm,
                                                        monkeypatch):
    client, state = _route_client(tmp_path, monkeypatch)
    script = _dummy_script(tmp_path)
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "7802")
    job = _checkpointed(state.jobs, "7801", script=script)

    resp = client.post(f"/api/jobs/{job.id}/resubmit")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["slurmJobID"] == "7802"
    assert body["manualResubmit"] is True
    assert body["resubmitOf"] == job.id
    assert state.jobs.get(body["jobId"]).executor_job_id == "7802"


def test_resubmit_route_maps_refusals_to_409_and_unknown_to_404(
        tmp_path, fake_slurm, monkeypatch):
    client, state = _route_client(tmp_path, monkeypatch)
    script = _dummy_script(tmp_path)
    running = _checkpointed(state.jobs, "7901", script=script, status="running")
    done = _checkpointed(state.jobs, "7902", script=script, status="succeeded")

    r = client.post(f"/api/jobs/{running.id}/resubmit")
    assert r.status_code == 409
    assert "resume applies only to a checkpointed job" in r.json()["detail"]
    r = client.post(f"/api/jobs/{done.id}/resubmit")
    assert r.status_code == 409
    assert "already finished" in r.json()["detail"]
    assert client.post("/api/jobs/ghost/resubmit").status_code == 404
    assert fake_slurm.calls("sbatch") == []


def test_resubmit_route_walltime_override_round_trip(tmp_path, fake_slurm,
                                                     monkeypatch):
    """The body's walltime reaches sbatch as a --time flag, the identical
    script is what is submitted, and the response echoes the override."""
    client, state = _route_client(tmp_path, monkeypatch)
    script = _dummy_script(tmp_path)
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "8602")
    job = _checkpointed(state.jobs, "8601", script=script)

    resp = client.post(f"/api/jobs/{job.id}/resubmit",
                       json={"walltime": "12:00:00"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["slurmJobID"] == "8602"
    assert body["walltime"] == "12:00:00"
    calls = fake_slurm.calls("sbatch")
    assert len(calls) == 1
    assert "--time=12:00:00" in calls[0] and calls[0].endswith(script)


def test_resubmit_route_refuses_malformed_walltime_with_400(
        tmp_path, fake_slurm, monkeypatch):
    client, state = _route_client(tmp_path, monkeypatch)
    script = _dummy_script(tmp_path)
    job = _checkpointed(state.jobs, "8701", script=script)
    resp = client.post(f"/api/jobs/{job.id}/resubmit",
                       json={"walltime": "tomorrow"})
    assert resp.status_code == 400
    assert "invalid walltime" in resp.json()["detail"]
    # Refused before the job was even considered — nothing submitted, the
    # parked run untouched.
    assert fake_slurm.calls("sbatch") == []
    assert state.jobs.get(job.id).status == "checkpointed"


def test_resubmit_route_refuses_walltime_on_in_process_resume(
        tmp_path, fake_slurm, monkeypatch):
    """A cancel-parked in-process run has no scheduler limit to raise — a
    walltime override there would be silently meaningless, so it refuses."""
    client, state = _route_client(tmp_path, monkeypatch)
    job = state.jobs.record_external("experiment:sweep",
                                     status="cancelledResumable",
                                     executor="local")
    resp = client.post(f"/api/jobs/{job.id}/resubmit",
                       json={"walltime": "02:00:00"})
    assert resp.status_code == 400
    assert "no scheduler limit to raise" in resp.json()["detail"]


def test_resubmit_route_surfaces_sbatch_failure_as_502(tmp_path, fake_slurm,
                                                       monkeypatch):
    client, state = _route_client(tmp_path, monkeypatch)
    script = _dummy_script(tmp_path)
    job = _checkpointed(state.jobs, "8001", script=script)
    monkeypatch.setenv("FAKE_SBATCH_FAIL", "1")
    resp = client.post(f"/api/jobs/{job.id}/resubmit")
    assert resp.status_code == 502
    assert "checkpointed run is unchanged" in resp.json()["detail"]
    # The parked job is untouched — still resumable once sbatch works again.
    assert state.jobs.get(job.id).status == "checkpointed"
    assert not (state.jobs.get(job.id).result or {}).get("resubmittedAs")


# --- auth tier (through the real app, whose middleware owns the gate) ---------------

def test_resubmit_is_privileged_like_its_mutating_siblings(tmp_path, monkeypatch):
    """Same gate as the _PRIVILEGED_PREFIXES peers (test_switch_is_privileged
    shape): on a Slurm deployment the route needs a bearer token even at the
    default auth mode — 503 with no token configured, 401 on a wrong one."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from steerlab_server.api.app import app

    client = TestClient(app)
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    monkeypatch.delenv("STEERLAB_AUTH_TOKEN", raising=False)
    resp = client.post("/api/jobs/abc123/resubmit")
    assert resp.status_code == 503
    assert "STEERLAB_AUTH_TOKEN" in resp.json()["detail"]

    monkeypatch.setenv("STEERLAB_AUTH_TOKEN", "sekrit")
    resp = client.post("/api/jobs/abc123/resubmit",
                       headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401
    # A correct token passes the middleware; the unknown id then 404s in the
    # route proper — proof the tier check sits in front of the verb.
    resp = client.post("/api/jobs/abc123/resubmit",
                       headers={"Authorization": "Bearer sekrit"})
    assert resp.status_code == 404


# --- cancelledResumable: in-process cancel-parked runs (2026-08-03 round 3) --------
# A cancel-parked sweep was resumable on disk but invisible to the app: the
# worker dropped its result (so the job record never learned the run
# directory) and every resume affordance keyed on "checkpointed". The worker
# now retains the result and stamps cancelledResumable when the directory
# really parked; the resubmit route re-dispatches the verb in-process.

import time as _time

from steerlab_server.experiment import resume as _resume_mod


def _wait_terminal(mgr, job_id, timeout=10.0):
    from steerlab_server.api.jobs import TERMINAL
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        job = mgr.get(job_id)
        if job is not None and job.status in TERMINAL:
            return job
        _time.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached a terminal state")


def _parked(tmp_path, name="parked", verb="sweep"):
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    _resume_mod.write_state(str(run_dir), run_id=name, verb=verb,
                            completed_records=1, reason="cancel")
    return str(run_dir)


def test_worker_cancel_retains_result_and_stamps_resumable(tmp_path):
    mgr = _manager(tmp_path)
    run_dir = _parked(tmp_path)

    def work(job):
        job._cancel.set()  # a live cancel observed mid-work
        return {"runDirectory": run_dir, "experiment": "x"}

    job = mgr.submit("experiment:sweep", work)
    got = _wait_terminal(mgr, job.id)
    assert got.status == "cancelledResumable"
    assert got.result["runDirectory"] == run_dir
    assert got.result["experiment"] == "x"


def test_worker_cancel_without_parked_state_stays_plain_cancelled(tmp_path):
    mgr = _manager(tmp_path)

    def work(job):
        job._cancel.set()  # a live cancel observed mid-work
        return {"runDirectory": str(tmp_path / "never-parked"),
                "experiment": "x"}

    job = mgr.submit("experiment:sweep", work)
    got = _wait_terminal(mgr, job.id)
    assert got.status == "cancelled"
    # The result is retained either way — dropping it was the original hole.
    assert got.result["runDirectory"].endswith("never-parked")


def test_resubmit_route_resumes_cancelled_in_process_run(tmp_path, monkeypatch):
    client, state = _route_client(tmp_path, monkeypatch)
    from steerlab_server.experiment import tasks as tasks_mod
    run_dir = _parked(tmp_path / "root" / "runs", name="exp-cx-sweep-p")
    job = state.jobs.record_external(
        "experiment:sweep", status="cancelledResumable", executor="local",
        result={"runDirectory": run_dir, "experiment": "cx"})
    resumed: list = []

    def fake_sweep(name, **kwargs):
        resumed.append((name, kwargs.get("run_directory")))
        return run_dir
    monkeypatch.setattr(tasks_mod, "sweep", fake_sweep)

    resp = client.post(f"/api/jobs/{job.id}/resubmit")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["manualResubmit"] is True
    assert body["resubmitOf"] == job.id
    continuation = _wait_terminal(state.jobs, body["jobId"])
    assert continuation.status == "succeeded"
    assert resumed == [("cx", run_dir)]
    assert (state.jobs.get(job.id).result or {})["resubmittedAs"] == body["jobId"]
    # A second click refuses: the continuation is carrying the run.
    r2 = client.post(f"/api/jobs/{job.id}/resubmit")
    assert r2.status_code == 409
    assert "already resumed" in r2.json()["detail"]


def test_resubmit_route_refuses_cancelled_resumable_without_parked_dir(
        tmp_path, monkeypatch):
    client, state = _route_client(tmp_path, monkeypatch)
    job = state.jobs.record_external(
        "experiment:sweep", status="cancelledResumable", executor="local",
        result={"runDirectory": str(tmp_path / "gone"), "experiment": "cx"})
    r = client.post(f"/api/jobs/{job.id}/resubmit")
    assert r.status_code == 409
    assert "missing, already complete, or carries no resume state" in r.json()["detail"]


def test_resubmit_route_single_winner_under_concurrent_clicks(
        tmp_path, monkeypatch):
    """Round 4, P1: the in-process resume uses the same durable SQLite
    compare-and-set claim the Slurm core does — two GENUINELY concurrent
    clicks produce exactly one continuation.

    Flake diagnosis 2026-08-05 (this test failed ~1 in 5 solo runs):
    - The warm-up request below is load-bearing. FastAPI's deferred
      ``include_router`` materialization builds its effective-route cache
      lazily on the FIRST request with no lock; when the two barrier'd
      clicks were the app's first requests ever, they raced that build and
      the loser intermittently drew the framework's default ``Not Found``
      404 before the resubmit route ran at all.
    - The loser's honest answer is legitimately EITHER a 409 refusal (live
      claim, or already-resumed) OR a 200 pointing at the winner's
      continuation (the pre-claim repair path, once the continuation is
      durably visible) — which one depends on scheduler timing. The
      invariant under test is exactly-one-submission, not which honest
      answer the loser drew."""
    import threading

    client, state = _route_client(tmp_path, monkeypatch)
    from steerlab_server.experiment import tasks as tasks_mod
    run_dir = _parked(tmp_path / "root" / "runs", name="exp-cc-sweep-p")
    job = state.jobs.record_external(
        "experiment:sweep", status="cancelledResumable", executor="local",
        result={"runDirectory": run_dir, "experiment": "cc"})
    monkeypatch.setattr(tasks_mod, "sweep", lambda name, **kw: run_dir)

    # Warm-up: force the framework's one-time lazy route build to happen
    # before concurrency, and prove the route is registered (the app's own
    # "no such job" 404, not the framework's default "Not Found").
    warmup = client.post("/api/jobs/no-such-job/resubmit")
    assert warmup.status_code == 404
    assert "no such job" in warmup.json()["detail"]

    barrier = threading.Barrier(2)
    responses: list = []

    def click():
        barrier.wait()
        r = client.post(f"/api/jobs/{job.id}/resubmit")
        responses.append((r.status_code, r.json()))

    threads = [threading.Thread(target=click) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Exactly one continuation exists, and the original is stamped with it.
    continuations = state.jobs.store.find_resubmissions_of(job.id)
    assert len(continuations) == 1
    assert (state.jobs.get(job.id).result or {})["resubmittedAs"] \
        == continuations[0].id
    codes = sorted(code for code, _ in responses)
    assert codes in ([200, 409], [200, 200]), responses
    for code, body in responses:
        if code == 200:
            # Every success answer points at the ONE continuation.
            assert body["jobId"] == continuations[0].id
        else:
            # And every refusal is a resume-race refusal, nothing else.
            assert "resume" in body["detail"].lower()


def test_resubmit_route_repairs_missing_stamp_from_existing_continuation(
        tmp_path, monkeypatch):
    """Crash between submission and stamping: the continuation exists
    durably (its resubmitOf marker) but the original was never stamped —
    Resume repairs the stamp and NEVER submits a second time."""
    client, state = _route_client(tmp_path, monkeypatch)
    from steerlab_server.experiment import tasks as tasks_mod
    run_dir = _parked(tmp_path / "root" / "runs", name="exp-cr-sweep-p")
    job = state.jobs.record_external(
        "experiment:sweep", status="cancelledResumable", executor="local",
        result={"runDirectory": run_dir, "experiment": "cr"})
    child = state.jobs.record_external(
        "experiment:sweep", status="succeeded", executor="local",
        requested_resources={"resubmitOf": job.id})
    submitted: list = []
    monkeypatch.setattr(tasks_mod, "sweep",
                        lambda name, **kw: submitted.append(name) or run_dir)

    resp = client.post(f"/api/jobs/{job.id}/resubmit")
    assert resp.status_code == 200
    assert resp.json()["jobId"] == child.id
    assert submitted == []  # repaired, not resubmitted
    assert (state.jobs.get(job.id).result or {})["resubmittedAs"] == child.id


def test_resubmit_route_takes_over_a_stale_claim_with_no_child(
        tmp_path, monkeypatch):
    """Round 5, P2 (claim-then-crash): the claimant died between claiming
    and the continuation's durable insertion. Once the claim is stale and
    no resubmitOf child exists, a manual Resume reclaims and submits —
    while a LIVE claim keeps refusing."""
    from steerlab_server.api.jobs import DurableJobStore

    client, state = _route_client(tmp_path, monkeypatch)
    from steerlab_server.experiment import tasks as tasks_mod
    run_dir = _parked(tmp_path / "root" / "runs", name="exp-st-sweep-p")
    stale_claim = {"claimant": "dead-click", "at": _time.time()
                   - DurableJobStore.RESUBMIT_CLAIM_STALE_SECONDS - 1}
    job = state.jobs.record_external(
        "experiment:sweep", status="cancelledResumable", executor="local",
        result={"runDirectory": run_dir, "experiment": "st",
                "resubmitClaim": stale_claim})
    monkeypatch.setattr(tasks_mod, "sweep", lambda name, **kw: run_dir)

    resp = client.post(f"/api/jobs/{job.id}/resubmit")
    assert resp.status_code == 200
    continuation = _wait_terminal(state.jobs, resp.json()["jobId"])
    assert continuation.status == "succeeded"
    assert (state.jobs.get(job.id).result or {})["resubmittedAs"] \
        == continuation.id


def test_resubmit_route_still_refuses_a_live_claim(tmp_path, monkeypatch):
    client, state = _route_client(tmp_path, monkeypatch)
    run_dir = _parked(tmp_path / "root" / "runs", name="exp-lv-sweep-p")
    job = state.jobs.record_external(
        "experiment:sweep", status="cancelledResumable", executor="local",
        result={"runDirectory": run_dir, "experiment": "lv",
                "resubmitClaim": {"claimant": "other-click",
                                  "at": _time.time()}})
    r = client.post(f"/api/jobs/{job.id}/resubmit")
    assert r.status_code == 409
    assert "live resume claim" in r.json()["detail"]
