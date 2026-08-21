"""GPU-session Wave 1 (docs/GPU-SESSION-PLAN.md): roles, session record,
idle lifecycle, and the streaming reverse proxy.

Hermetic throughout: Slurm is the fakebin doubles, the worker is either an
in-process ASGI app (round-trip/keepalive/refusals) or a real uvicorn thread
(the SSE incremental-arrival proof, which an ASGI transport cannot give —
it buffers whole responses).
"""

import json
import os
import socket
import sys
import threading
import time
import types
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from steerlab_server.api import gpu_session
from steerlab_server.api.app import app
from steerlab_server.api.profile import capability_snapshot, server_role
from steerlab_server.api.routes import ServiceState, build_router

FAKEBIN = os.path.join(os.path.dirname(__file__), "fakebin")
AUTH = {"Authorization": "Bearer sekrit"}


def _fresh_client() -> TestClient:
    """A router over a FRESH ServiceState (own JobManager over the
    test-scoped metadata root) — session jobs must not accumulate as
    forever-non-terminal records in the shared import-time store."""
    test_app = FastAPI()
    test_app.include_router(build_router(ServiceState()))
    return TestClient(test_app)


# --- fixtures -----------------------------------------------------------------


@pytest.fixture
def controller_env(tmp_path, monkeypatch):
    """A controller-role server with fakebin Slurm and an isolated metadata
    root (session records must never land in the shared test store)."""
    log_dir = tmp_path / "slurm-log"
    log_dir.mkdir()
    monkeypatch.setenv("PATH", FAKEBIN + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("FAKE_SLURM_LOG", str(log_dir))
    monkeypatch.setenv("FAKE_SLURM_STATE_FILE", str(tmp_path / "slurm-state.json"))
    monkeypatch.setenv("FAKE_SLURM_JOB_ID", "424242")
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "controller")
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN", "sekrit")
    monkeypatch.delenv("STEERLAB_AUTH_MODE", raising=False)
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT", raising=False)
    # Process-wide proxy-traffic evidence must not leak between tests: a
    # proxy test's bytes would otherwise make a later reconcile read "busy".
    monkeypatch.setattr(gpu_session, "_LAST_PROXIED_TRAFFIC_MONOTONIC", 0.0)
    return {"tmp": tmp_path, "log": log_dir,
            "state_file": tmp_path / "slurm-state.json",
            "client": _fresh_client()}


@pytest.fixture
def worker_client(tmp_path, monkeypatch):
    """A gpu-session-role server (loopback bind, so no token needed)."""
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "gpu-session")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    monkeypatch.setenv("STEERLAB_SESSION_GENERATION", "gen-w")
    monkeypatch.setenv("STEERLAB_SESSION_IDLE_MINUTES", "45")
    monkeypatch.setenv("STEERLAB_SESSION_PORT", "9191")
    gpu_session.reset_worker_for_tests()
    yield TestClient(app)
    gpu_session.reset_worker_for_tests()


def _install_session_records(generation="gen-proxy", *, node="127.0.0.1",
                             port=59999, session_state="ready",
                             worker_state="ready",
                             worker_started_at="2026-07-16T00:00:00+00:00"):
    """Write a session record + matching worker discovery record, as a
    reconciled controller (or a restarted one) would find them on disk.
    ``worker_started_at`` is the timestamp-injection seam: the worker
    record's startedAt is authoritative (it overrides scheduler-observed
    stamps), so elapsed-time behaviors are driven from here."""
    gpu_session.write_session_record({
        "sessionGeneration": generation, "slurmJobID": "424242",
        "node": node, "port": port, "state": session_state,
        "workspaceRoot": "/tmp", "gpuType": "A100", "gres": "gpu:A100:1",
        "partition": None, "walltime": "02:00:00", "idleMinutes": 30,
        "startedAt": worker_started_at, "expiresAt": None,
        "serverVersion": "test", "role": "gpu-session",
    })
    gpu_session._write_json_atomic(gpu_session.worker_record_path(), {
        "sessionGeneration": generation, "node": node, "port": port,
        "startedAt": worker_started_at, "state": worker_state,
    })


@pytest.fixture
def proxy_controller(tmp_path, monkeypatch):
    """Controller role WITHOUT the Slurm executor (the proxy needs neither
    sbatch nor a token gate), plus isolated records."""
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "controller")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    yield
    gpu_session._TRANSPORT = None


# --- roles ---------------------------------------------------------------------


def test_role_derivation(monkeypatch):
    monkeypatch.delenv("STEERLAB_SERVER_ROLE", raising=False)
    monkeypatch.delenv("STEERLAB_LAUNCH_TOPOLOGY", raising=False)
    monkeypatch.delenv("STEERLAB_SERVER_PROFILE", raising=False)
    monkeypatch.delenv("STEERLAB_EXECUTOR", raising=False)
    assert server_role() == "workstation"
    # Legacy alias 1: the daemon-in-a-job topology has always meant "submit
    # daemon" — it derives to controller without a new variable.
    monkeypatch.setenv("STEERLAB_LAUNCH_TOPOLOGY", "batch")
    assert server_role() == "controller"
    monkeypatch.delenv("STEERLAB_LAUNCH_TOPOLOGY")
    # Legacy alias 2: the cluster submit-daemon combination.
    monkeypatch.setenv("STEERLAB_SERVER_PROFILE", "cluster")
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    assert server_role() == "controller"
    # The explicit role always wins over derivation.
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "gpu-session")
    assert server_role() == "gpu-session"


def test_capabilities_advertise_role_and_gpu_session(monkeypatch):
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "controller")
    snap = capability_snapshot()
    assert snap["serverRole"] == "controller"
    assert snap["chat"]["gpuSession"] is True
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "workstation")
    snap = capability_snapshot()
    assert snap["serverRole"] == "workstation"
    assert snap["chat"]["gpuSession"] is False


# --- role refusal matrix ---------------------------------------------------------


def test_worker_refusal_matrix(worker_client):
    # No recursion: a worker never starts another worker.
    resp = worker_client.post("/api/session/start", json={})
    assert resp.status_code == 409
    assert "no recursion" in resp.json()["detail"]
    # Studies go through the controller (acceptance criterion 4).
    for path, body in (("/api/studies/submit", {"experiment": "x"}),
                       ("/api/studies/submit-bundle", {"bundlePath": "x"}),
                       ("/api/slurm/submit", {"command": ["echo", "hi"]})):
        resp = worker_client.post(path, json=body)
        assert resp.status_code == 409, path
        assert "controller" in resp.json()["detail"], path
    # Jobs routes answer the single-writer refusal, not a 500.
    resp = worker_client.get("/api/jobs")
    assert resp.status_code == 409
    assert "no job subsystem (role: gpu-session)" in resp.json()["detail"]
    assert worker_client.post("/api/jobs/xyz/cancel").status_code == 409


def test_worker_allows_resident_loads(worker_client, monkeypatch):
    # The controller's role-keyed 409 must NOT fire on the worker: the load
    # proceeds to the ordinary loader (which then 400s on a fake model id).
    monkeypatch.delenv("STEERLAB_ALLOW_RESIDENT_MODELS", raising=False)
    resp = worker_client.post("/api/load", json={"model": "org/tiny"})
    assert resp.status_code != 409


def test_workstation_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("STEERLAB_SERVER_ROLE", raising=False)
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    client = TestClient(app)
    assert client.get("/api/jobs").status_code == 200
    assert "jobs" in client.get("/api/jobs").json()
    assert client.get("/api/session").json() == {"session": None}
    resp = client.post("/api/session/start", json={})
    assert resp.status_code == 409
    assert "controller-role" in resp.json()["detail"]
    assert client.post("/api/session/keepalive").status_code == 409
    assert client.delete("/api/session").status_code == 409


def test_worker_constructs_no_job_subsystem(tmp_path, monkeypatch):
    # Acceptance criterion 2 (single-writer): role=gpu-session builds NO
    # JobManager and touches no durable store — not even an empty SQLite file.
    meta = tmp_path / "meta"
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(meta))
    monkeypatch.delenv("STEERLAB_JOBS_DB", raising=False)
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "gpu-session")
    state = ServiceState()
    assert state.job_manager_or_none is None
    assert state._jobs is None
    assert not (meta / "jobs.sqlite").exists()
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "workstation")
    assert ServiceState().job_manager_or_none is not None


# --- session record + controller lifecycle ---------------------------------------


def test_session_start_submits_worker_job(controller_env, monkeypatch):
    # Even a site env that defaults auto-resubmit ON must not make a chat
    # session respawn on a billed allocation.
    monkeypatch.setenv("STEERLAB_AUTO_RESUBMIT", "1")
    client = controller_env["client"]
    resp = client.post("/api/session/start", json={}, headers=AUTH)
    assert resp.status_code == 200
    record = resp.json()["session"]
    # Exact cross-engine contract keys — closed set, camelCase.
    assert set(record) == set(gpu_session.SESSION_RECORD_KEYS)
    assert record["state"] == "queued"
    assert record["slurmJobID"] == "424242"
    assert record["gpuType"] == "A100"
    assert record["idleMinutes"] == 30
    assert record["role"] == "gpu-session"
    assert record["node"] is None and record["startedAt"] is None
    # 2026-07-16 contract additions: submittedAt anchors the accounting-
    # visibility grace; stateDetail is null until there is something to say.
    assert record["submittedAt"]
    assert record["stateDetail"] is None
    # The record is on disk, atomically (no temp debris).
    meta = controller_env["tmp"] / "meta"
    assert json.loads((meta / "session.json").read_text())["sessionGeneration"] \
        == record["sessionGeneration"]
    assert not [n for n in os.listdir(meta) if n.endswith(".tmp")]
    # sbatch really ran, through the ordinary executor.
    assert (controller_env["log"] / "sbatch.calls").exists()
    # The rendered script carries the role/bind/auth-mode env and the
    # generation — but NEVER the token value (bundle files are durable
    # on-disk artifacts).
    scripts = list((controller_env["tmp"] / "runs").glob("*/slurm/run.sbatch"))
    assert len(scripts) == 1
    script = scripts[0].read_text()
    assert "STEERLAB_SERVER_ROLE=gpu-session" in script
    assert "STEERLAB_BIND=0.0.0.0" in script
    assert "STEERLAB_AUTH_MODE=token" in script
    assert record["sessionGeneration"] in script
    assert "sekrit" not in script
    assert "sekrit" not in scripts[0].with_name("bundle.json").read_text()
    # No checkpoint signal (live shakedown 2026-07-16): the worker has no
    # USR1 handler, so inheriting the study default (--signal USR1@600)
    # would kill the session ~10 min before the walltime the app displays.
    assert "--signal" not in script
    # Durable job record: kind session:gpu, auto-resubmit stamped OFF.
    jobs = client.get("/api/jobs", headers=AUTH).json()["jobs"]
    session_jobs = [j for j in jobs if j["kind"] == "session:gpu"
                    and j["executorJobID"] == "424242"]
    assert session_jobs
    rr = session_jobs[0]["requestedResources"]
    assert rr["auto_resubmit"] is False
    assert rr["requeue"] is False
    assert rr["sessionGeneration"] == record["sessionGeneration"]


def test_double_start_returns_existing_session(controller_env):
    client = controller_env["client"]
    first = client.post("/api/session/start", json={}, headers=AUTH)
    assert first.status_code == 200
    generation = first.json()["session"]["sessionGeneration"]
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "RUNNING"}}))
    second = client.post("/api/session/start", json={}, headers=AUTH)
    assert second.status_code == 409
    # Idempotency: the live record rides in the 409 body under "session".
    assert second.json()["session"]["sessionGeneration"] == generation
    # Never two workers: exactly one sbatch call happened.
    calls = (controller_env["log"] / "sbatch.calls").read_text().splitlines()
    assert len(calls) == 1


def test_get_session_reconciles_states(controller_env, monkeypatch):
    client = controller_env["client"]
    record = client.post("/api/session/start", json={},
                         headers=AUTH).json()["session"]
    generation = record["sessionGeneration"]
    meta = controller_env["tmp"] / "meta"

    # Job pending → queued.
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "PENDING"}}))
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "queued"

    # Job running, worker not yet discovered → starting.
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "RUNNING"}}))
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "starting"

    # A worker record from ANOTHER generation is stale and never trusted.
    gpu_session._write_json_atomic(str(meta / "session-worker.json"), {
        "sessionGeneration": "yesterdays-allocation", "node": "gpu-node-9",
        "port": 8081, "startedAt": "2026-07-15T00:00:00+00:00",
        "state": "ready"})
    assert gpu_session.read_worker_record(generation) is None
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "starting"
    assert body["node"] is None

    # Matching generation + reachable worker → the worker's own status,
    # node/startedAt derived, expiresAt = startedAt + walltime.
    gpu_session._write_json_atomic(str(meta / "session-worker.json"), {
        "sessionGeneration": generation, "node": "gpu-node-7", "port": 8081,
        "startedAt": "2026-07-16T10:00:00+00:00", "state": "ready"})
    monkeypatch.setattr(
        gpu_session, "probe_worker",
        lambda node, port, timeout=2.0: {"state": "busy", "busy": True,
                                         "idleRemainingSeconds": 1200,
                                         "sessionGeneration": generation})
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "busy"
    assert body["busy"] is True
    assert body["idleRemainingSeconds"] == 1200
    assert body["node"] == "gpu-node-7"
    assert body["startedAt"] == "2026-07-16T10:00:00+00:00"
    assert body["expiresAt"] == "2026-07-16T12:00:00+00:00"

    # Job running, probe misses RIGHT AFTER a success → hysteresis: one
    # missed short-timeout probe under heavy GPU work (cold load, long
    # decode) is latency, not death — the state holds with a detail saying
    # so (live 2026-07-17: the state flapped Ready↔Starting through every
    # model load).
    monkeypatch.setattr(gpu_session, "probe_worker",
                        lambda node, port, timeout=2.0: None)
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "busy"
    assert "missed a health probe" in body["stateDetail"]

    # Sustained silence (past the lapse window) → starting, honestly.
    monkeypatch.setenv("STEERLAB_SESSION_PROBE_LAPSE_SECONDS", "0")
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "starting"
    monkeypatch.delenv("STEERLAB_SESSION_PROBE_LAPSE_SECONDS")

    # Job gone: COMPLETED/CANCELLED → ended; FAILED → failed.
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "COMPLETED"}}))
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "ended"
    # Terminal states stick (no resurrection on the next poll).
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "RUNNING"}}))
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "ended"


def test_reconcile_failed_exit(controller_env):
    client = controller_env["client"]
    client.post("/api/session/start", json={}, headers=AUTH)
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "FAILED", "exit": "1:0"}}))
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "failed"


def test_reconcile_survives_controller_restart(controller_env, monkeypatch):
    # Plan §2.1: a resubmitted controller must rediscover a still-running
    # worker from DISK alone. Simulate the successor by writing the records
    # directly — no in-memory session state, no prior start in this process
    # path — and reconciling.
    _install_session_records(generation="gen-restart", node="gpu-node-3",
                             port=8081, session_state="queued")
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "RUNNING"}}))
    monkeypatch.setattr(
        gpu_session, "probe_worker",
        lambda node, port, timeout=2.0: {"state": "ready", "busy": False,
                                         "idleRemainingSeconds": 1500})
    body = controller_env["client"].get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "ready"
    assert body["node"] == "gpu-node-3"
    assert body["idleRemainingSeconds"] == 1500


def test_worker_stamped_idle_expiry_reconciles_to_ended(controller_env):
    # The worker wrote state:"ended" (idle expiry) before exiting; while the
    # scheduler still shows the job draining the session reads "ending", and
    # once the job is gone it reads "ended" — the app never sees a dead
    # socket surprise.
    _install_session_records(generation="gen-idle", worker_state="ended")
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "RUNNING"}}))
    client = controller_env["client"]
    assert client.get("/api/session",
                      headers=AUTH).json()["session"]["state"] == "ending"
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "COMPLETED"}}))
    # "ending" is non-terminal, so the next poll converges it.
    assert client.get("/api/session",
                      headers=AUTH).json()["session"]["state"] == "ended"


def test_delete_session_scancels_and_allows_fresh_start(controller_env):
    client = controller_env["client"]
    first = client.post("/api/session/start", json={},
                        headers=AUTH).json()["session"]
    resp = client.delete("/api/session", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    # Stop never claims what it can't prove (live shakedown 2026-07-16):
    # scancel succeeded, so the record reads "ending" — "ended" only arrives
    # when the SCHEDULER confirms the job terminal/gone.
    assert resp.json()["session"]["state"] == "ending"
    assert "424242" in (controller_env["log"] / "scancel.calls").read_text()
    # While "ending" the slot stays occupied: a start here could hide a
    # still-billing allocation behind a fresh session.
    assert client.post("/api/session/start", json={},
                       headers=AUTH).status_code == 409
    # The scheduler confirms the cancel → reconcile converges to "ended".
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "CANCELLED"}}))
    assert client.get("/api/session",
                      headers=AUTH).json()["session"]["state"] == "ended"
    # Acceptance criterion 3: a new session starts cleanly after the end —
    # no stale record blocks it.
    again = client.post("/api/session/start", json={}, headers=AUTH)
    assert again.status_code == 200
    assert again.json()["session"]["sessionGeneration"] \
        != first["sessionGeneration"]
    # End it, converge, and with no live session DELETE says so.
    client.delete("/api/session", headers=AUTH)
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "CANCELLED"}}))
    client.get("/api/session", headers=AUTH)
    assert client.delete("/api/session", headers=AUTH).status_code == 409


# --- live-shakedown fixes (2026-07-16): env inheritance, honest stop,
# --- transactional + lag-tolerant acquisition, parameter validation ------------


def _session_script(controller_env) -> str:
    scripts = list((controller_env["tmp"] / "runs").glob("*/slurm/run.sbatch"))
    assert len(scripts) == 1
    return scripts[0].read_text()


def test_session_worker_inherits_hf_cache_policy(controller_env, monkeypatch):
    # The controller sourced the bootstrap env file (HF_HOME on /work,
    # HF_HUB_OFFLINE=1); under --export=NONE the worker sees NEITHER unless
    # the bundle carries them — the first live chat broke exactly here (the
    # worker looked at ~/.cache and found no model).
    monkeypatch.setenv("HF_HOME", "/work/lab/hf-cache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("STEERLAB_NODE_STAGE_DIR", "/lscratch/$SLURM_JOB_ID")
    # The judge-key PATH (never the key) rides along too (2026-07-19) — a
    # worker's inline judging must resolve the same credential file.
    monkeypatch.setenv("STEERLAB_JUDGE_KEY_FILE", "/home/me/.steerlab/jk")
    resp = controller_env["client"].post("/api/session/start", json={},
                                         headers=AUTH)
    assert resp.status_code == 200
    script = _session_script(controller_env)
    assert "export HF_HOME=/work/lab/hf-cache" in script
    assert "export HF_HUB_OFFLINE=1" in script
    # The staging dir rides along VERBATIM — shell-quoted so $SLURM_JOB_ID
    # survives to the loader, which expands it with the WORKER's job id.
    assert "export STEERLAB_NODE_STAGE_DIR='/lscratch/$SLURM_JOB_ID'" in script
    assert "export STEERLAB_JUDGE_KEY_FILE=/home/me/.steerlab/jk" in script


def test_end_session_scancel_failure_is_not_terminal(controller_env,
                                                     monkeypatch):
    client = controller_env["client"]
    client.post("/api/session/start", json={}, headers=AUTH)
    monkeypatch.setenv("FAKE_SCANCEL_FAIL", "scancel: error: kill failed")
    resp = client.delete("/api/session", headers=AUTH)
    # A failed scancel is an actionable error naming the job id — never a
    # claimed success (the live failure: "ended" sessions still billing).
    assert resp.status_code == 502
    assert "424242" in resp.json()["detail"]
    assert "scancel" in resp.json()["detail"]
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "ending"          # live-ish, NOT terminal
    assert "424242" in body["stateDetail"]
    # CRITICAL companion: "ending" still occupies the slot — a fresh start
    # must not hide the possibly-running billed allocation.
    blocked = client.post("/api/session/start", json={}, headers=AUTH)
    assert blocked.status_code == 409
    assert blocked.json()["session"]["state"] == "ending"
    calls = (controller_env["log"] / "sbatch.calls").read_text().splitlines()
    assert len(calls) == 1
    # scancel works again → retrying DELETE succeeds and reconciliation
    # converges once the scheduler confirms.
    monkeypatch.delenv("FAKE_SCANCEL_FAIL")
    assert client.delete("/api/session", headers=AUTH).status_code == 200
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "CANCELLED"}}))
    assert client.get("/api/session",
                      headers=AUTH).json()["session"]["state"] == "ended"


def test_reconcile_never_resurrects_an_ending_session(controller_env,
                                                      monkeypatch):
    # A stop was requested but the job still runs (failed scancel, or Slurm
    # draining): a reachable worker must NOT flip the record back to "ready".
    _install_session_records(generation="gen-ending", node="gpu-node-2",
                             port=8081, session_state="ending")
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "RUNNING"}}))
    monkeypatch.setattr(
        gpu_session, "probe_worker",
        lambda node, port, timeout=2.0: {"state": "ready", "busy": False,
                                         "idleRemainingSeconds": 999})
    body = controller_env["client"].get("/api/session",
                                        headers=AUTH).json()["session"]
    assert body["state"] == "ending"
    # Scheduler confirms terminal → ended.
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "CANCELLED"}}))
    body = controller_env["client"].get("/api/session",
                                        headers=AUTH).json()["session"]
    assert body["state"] == "ended"


def test_visibility_lag_never_fails_a_fresh_session(controller_env):
    # THE live-shakedown orphan bug: two sessions were declared dead while
    # their jobs sat PENDING, invisible to sacct/squeue for a while after
    # submit. Scheduler-unknown within the grace window must stay "queued".
    client = controller_env["client"]
    client.post("/api/session/start", json={}, headers=AUTH)
    # No FAKE_SLURM_STATE_FILE entry: both sacct and squeue know nothing.
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "queued"
    assert client.post("/api/session/start", json={},
                       headers=AUTH).status_code == 409  # still occupied


def test_past_grace_unknown_scheduler_goes_unknown_and_blocks_start(
        controller_env):
    # Follow-up review 2026-07-16: a scheduler that never answered is
    # evidence about the CONVERSATION, not the allocation — past grace the
    # session goes nonterminal "unknown", NEVER "failed" (a terminal state
    # here would let a fresh start bury a possibly-running billed job).
    client = controller_env["client"]
    client.post("/api/session/start", json={}, headers=AUTH)
    # Age the submission past the grace window (timestamp injection).
    record = gpu_session.read_session_record()
    record["submittedAt"] = (datetime.now(timezone.utc)
                             - timedelta(hours=2)).isoformat()
    gpu_session.write_session_record(record)
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "unknown"
    # The detail names the job id and the exact manual check.
    assert "424242" in body["stateDetail"]
    assert "sacct -j 424242" in body["stateDetail"]
    # OCCUPIED: unknown blocks a new start.
    blocked = client.post("/api/session/start", json={}, headers=AUTH)
    assert blocked.status_code == 409
    assert blocked.json()["session"]["state"] == "unknown"


def test_unknown_closes_on_positive_scheduler_terminal(controller_env):
    client = controller_env["client"]
    client.post("/api/session/start", json={}, headers=AUTH)
    record = gpu_session.read_session_record()
    record["submittedAt"] = (datetime.now(timezone.utc)
                             - timedelta(hours=2)).isoformat()
    gpu_session.write_session_record(record)
    assert client.get("/api/session",
                      headers=AUTH).json()["session"]["state"] == "unknown"
    # The scheduler answers again with a POSITIVE terminal → closed.
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "COMPLETED"}}))
    assert client.get("/api/session",
                      headers=AUTH).json()["session"]["state"] == "ended"
    assert client.post("/api/session/start", json={},
                       headers=AUTH).status_code == 200
    # And a FAILED report maps to failed (as appropriate), also terminal.
    record = gpu_session.read_session_record()
    record["state"] = "unknown"
    gpu_session.write_session_record(record)
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "FAILED", "exit": "1:0"}}))
    assert client.get("/api/session",
                      headers=AUTH).json()["session"]["state"] == "failed"


def test_started_worker_gone_silent_goes_unknown_then_recovers(
        controller_env, monkeypatch):
    # A once-started worker whose probe AND scheduler both stop answering is
    # "unknown" (was: "ended" — the slow-motion orphan door), and RECOVERS
    # the moment the worker answers again.
    _install_session_records(generation="gen-silent", node="gpu-node-5",
                             port=8081, session_state="ready")
    # No scheduler state written: sacct/squeue both silent.
    monkeypatch.setattr(gpu_session, "probe_worker",
                        lambda node, port, timeout=2.0: None)
    client = controller_env["client"]
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "unknown"
    assert "424242" in body["stateDetail"]
    assert client.post("/api/session/start", json={},
                       headers=AUTH).status_code == 409
    # The worker becomes reachable again → ready/busy/idle, detail cleared.
    monkeypatch.setattr(
        gpu_session, "probe_worker",
        lambda node, port, timeout=2.0: {"state": "ready", "busy": False,
                                         "idleRemainingSeconds": 1000})
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "ready"
    assert body["stateDetail"] is None


def test_force_clear_releases_unknown_session(controller_env, monkeypatch):
    # The operator clear: force attempts scancel best-effort and stamps
    # "ended" REGARDLESS (even a failing scancel), recording that a human
    # verified the job by hand. Both parameter shapes are contract: the
    # ?force=1 query and the {"force": true} JSON body.
    client = controller_env["client"]
    client.post("/api/session/start", json={}, headers=AUTH)
    record = gpu_session.read_session_record()
    record["submittedAt"] = (datetime.now(timezone.utc)
                             - timedelta(hours=2)).isoformat()
    gpu_session.write_session_record(record)
    assert client.get("/api/session",
                      headers=AUTH).json()["session"]["state"] == "unknown"
    monkeypatch.setenv("FAKE_SCANCEL_FAIL", "scancel: error")
    resp = client.delete("/api/session?force=1", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    session = resp.json()["session"]
    assert session["state"] == "ended"
    assert session["stateDetail"] == ("cleared by operator after manual "
                                      "verification (job 424242)")
    # scancel WAS attempted (best-effort), it just didn't have to succeed.
    assert "424242" in (controller_env["log"] / "scancel.calls").read_text()
    # Released: a fresh session can start.
    assert client.post("/api/session/start", json={},
                       headers=AUTH).status_code == 200
    # JSON-body form of force (the Swift client's shape).
    monkeypatch.delenv("FAKE_SCANCEL_FAIL")
    resp = client.request("DELETE", "/api/session", headers=AUTH,
                          json={"force": True})
    assert resp.status_code == 200
    assert resp.json()["session"]["state"] == "ended"
    assert "cleared by operator" in resp.json()["session"]["stateDetail"]


def test_unknown_state_is_in_vocabulary_and_nonterminal():
    assert "unknown" in gpu_session.SESSION_STATES
    assert "unknown" not in gpu_session.TERMINAL_SESSION_STATES


def test_recent_proxied_traffic_beats_a_failed_probe(controller_env,
                                                     monkeypatch):
    # Bytes beat probes (live 2026-07-17): mid-load the worker's CPUs are
    # pegged and it misses the 2s probe — while its heartbeat bytes flow
    # through the controller's own proxy. Recent proxied traffic must read
    # "busy with a named reason", never the "try curl" demotion; stale
    # traffic falls back to the ordinary diagnosis.
    _install_session_records(
        generation="gen-traffic", node="gpu-node-7", port=8081,
        session_state="starting",
        worker_started_at=(datetime.now(timezone.utc)
                           - timedelta(minutes=10)).isoformat())
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "RUNNING"}}))
    monkeypatch.setattr(gpu_session, "probe_worker",
                        lambda node, port, timeout=2.0: None)
    gpu_session.note_proxied_traffic()
    body = controller_env["client"].get("/api/session",
                                        headers=AUTH).json()["session"]
    assert body["state"] == "busy"
    assert "proxied request" in body["stateDetail"]
    assert body["busy"] is True
    # Stale traffic (older than the window) proves nothing current. Zero the
    # probe lapse too: the first reconcile stamped lastReachableAt, and the
    # hysteresis it feeds is a separate (already-tested) mechanism.
    monkeypatch.setattr(gpu_session, "_LAST_PROXIED_TRAFFIC_MONOTONIC",
                        gpu_session.time.monotonic() - 3600)
    monkeypatch.setenv("STEERLAB_SESSION_PROBE_LAPSE_SECONDS", "0")
    body = controller_env["client"].get("/api/session",
                                        headers=AUTH).json()["session"]
    assert body["state"] == "starting"


def test_running_but_unreachable_worker_gets_a_diagnosis(controller_env,
                                                         monkeypatch):
    # Compute→compute reachability is the plan-§5 open assumption: once the
    # job has been RUNNING past the diagnosis threshold with the probe still
    # failing, "starting" stays nonterminal but stateDetail says exactly
    # what to try by hand.
    _install_session_records(
        generation="gen-diag", node="gpu-node-7", port=8081,
        session_state="starting",
        worker_started_at=(datetime.now(timezone.utc)
                           - timedelta(minutes=10)).isoformat())
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "RUNNING"}}))
    monkeypatch.setattr(gpu_session, "probe_worker",
                        lambda node, port, timeout=2.0: None)
    body = controller_env["client"].get("/api/session",
                                        headers=AUTH).json()["session"]
    assert body["state"] == "starting"                      # nonterminal
    detail = body["stateDetail"]
    assert "gpu-node-7:8081" in detail                      # node:port
    assert "curl http://gpu-node-7:8081/api/capabilities" in detail
    assert "compute-to-compute" in detail
    assert "running for" in detail and "s but" in detail    # elapsed time


def test_running_within_diagnosis_threshold_stays_quiet(controller_env,
                                                        monkeypatch):
    _install_session_records(generation="gen-quiet", node="gpu-node-7",
                             port=8081, session_state="starting",
                             worker_started_at=gpu_session._utc_now_iso())
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "RUNNING"}}))
    monkeypatch.setattr(gpu_session, "probe_worker",
                        lambda node, port, timeout=2.0: None)
    body = controller_env["client"].get("/api/session",
                                        headers=AUTH).json()["session"]
    assert body["state"] == "starting"
    assert body["stateDetail"] is None


def test_diagnosis_threshold_env_override(controller_env, monkeypatch):
    _install_session_records(generation="gen-env", node="gpu-node-7",
                             port=8081, session_state="starting",
                             worker_started_at=gpu_session._utc_now_iso())
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "RUNNING"}}))
    monkeypatch.setattr(gpu_session, "probe_worker",
                        lambda node, port, timeout=2.0: None)
    monkeypatch.setenv("STEERLAB_SESSION_STARTING_DIAGNOSIS_SECONDS", "0")
    body = controller_env["client"].get("/api/session",
                                        headers=AUTH).json()["session"]
    assert body["state"] == "starting"
    assert "compute-to-compute" in body["stateDetail"]


def test_jobs_cancel_route_surfaces_scancel_failure(controller_env,
                                                    monkeypatch):
    # Generic jobs route: {"ok": true} on a failed scancel hid a possibly
    # still-running allocation from the caller. False from cancel() → 502.
    client = controller_env["client"]
    client.post("/api/session/start", json={}, headers=AUTH)
    jobs = client.get("/api/jobs", headers=AUTH).json()["jobs"]
    job_id = next(j["id"] for j in jobs if j["kind"] == "session:gpu"
                  and j["executorJobID"] == "424242")
    monkeypatch.setenv("FAKE_SCANCEL_FAIL", "scancel: error")
    resp = client.post(f"/api/jobs/{job_id}/cancel", headers=AUTH)
    assert resp.status_code == 502
    assert "424242" in resp.json()["detail"]
    assert "still be running" in resp.json()["detail"]
    # Retry once scancel works again → honest ok.
    monkeypatch.delenv("FAKE_SCANCEL_FAIL")
    resp = client.post(f"/api/jobs/{job_id}/cancel", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    # Unknown ids still 404 (not 502).
    assert client.post("/api/jobs/nope/cancel",
                       headers=AUTH).status_code == 404


def test_visibility_grace_window_arithmetic(monkeypatch):
    now = gpu_session._utc_now_iso()
    assert gpu_session._within_visibility_grace(now) is True
    assert gpu_session._within_visibility_grace(None) is False       # legacy
    assert gpu_session._within_visibility_grace("garbled") is False  # legacy
    monkeypatch.setenv("STEERLAB_SESSION_VISIBILITY_GRACE_SECONDS", "0")
    assert gpu_session._within_visibility_grace(now) is False


def test_concurrent_starts_submit_exactly_once(controller_env):
    # The cross-process lifecycle lock serializes check-then-submit: exactly
    # one thread wins, the other adopts via SessionConflict — never two
    # billed workers.
    jobs = ServiceState().jobs
    barrier = threading.Barrier(2)
    outcomes = []

    def attempt():
        barrier.wait()
        try:
            record = gpu_session.start_session({}, jobs)
            outcomes.append(("ok", record))
        except gpu_session.SessionConflict as exc:
            outcomes.append(("conflict", exc.session))

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert sorted(kind for kind, _ in outcomes) == ["conflict", "ok"]
    calls = (controller_env["log"] / "sbatch.calls").read_text().splitlines()
    assert len(calls) == 1
    # The loser adopted the winner's session, not a phantom.
    winner = next(r for kind, r in outcomes if kind == "ok")
    loser = next(r for kind, r in outcomes if kind == "conflict")
    assert loser["sessionGeneration"] == winner["sessionGeneration"]


def test_persist_failure_rolls_back_the_submitted_job(controller_env):
    # sbatch succeeded but the record write blew up: the fresh job must be
    # scancel'd before the error surfaces — an allocation the app cannot see
    # is the worst outcome. (Nested MonkeyPatch context: undoing the test's
    # own monkeypatch would also strip the controller_env fixture's env.)
    def explode(record):
        raise OSError("disk full")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gpu_session, "write_session_record", explode)
        resp = controller_env["client"].post("/api/session/start", json={},
                                             headers=AUTH)
        assert resp.status_code == 400
        assert "424242" in (controller_env["log"] / "scancel.calls").read_text()
    # Nothing tracked, nothing blocking: a clean start works afterwards.
    assert gpu_session.read_session_record() is None
    assert controller_env["client"].post(
        "/api/session/start", json={}, headers=AUTH).status_code == 200


@pytest.mark.parametrize("body,field", [
    ({"idleMinutes": 0}, "idleMinutes"),      # 0 disables the idle timer —
    ({"idleMinutes": -5}, "idleMinutes"),     # unreachable via API by design
    ({"idleMinutes": 241}, "idleMinutes"),
    ({"idleMinutes": "abc"}, "idleMinutes"),
    ({"idleMinutes": True}, "idleMinutes"),
    ({"idleMinutes": 30.5}, "idleMinutes"),
    ({"walltime": "2-00:00:00"}, "walltime"),
    ({"walltime": "02:60:00"}, "walltime"),
    ({"walltime": "1234:00:00"}, "walltime"),
    ({"port": 80}, "port"),
    ({"port": 70000}, "port"),
    ({"cpus": 0}, "cpus"),
    ({"cpus": 65}, "cpus"),
    ({"memory": "0G"}, "memory"),
    ({"memory": "64Q"}, "memory"),
    ({"memory": "-1G"}, "memory"),
    ({"gres": "gpu:A100:1; rm -rf /"}, "gres"),
    ({"partition": "gpu p"}, "partition"),
])
def test_start_rejects_bad_parameters(controller_env, body, field):
    resp = controller_env["client"].post("/api/session/start", json=body,
                                         headers=AUTH)
    assert resp.status_code == 400
    assert field in resp.json()["detail"]  # actionable: names the field
    # Nothing was submitted for a refused request.
    assert not (controller_env["log"] / "sbatch.calls").exists()


def test_start_accepts_valid_parameters(controller_env):
    resp = controller_env["client"].post("/api/session/start", json={
        "idleMinutes": 45, "walltime": "01:30:00", "port": 8090, "cpus": 4,
        "memory": "32G", "gres": "gpu:H100:1", "partition": "gpu_p",
    }, headers=AUTH)
    assert resp.status_code == 200
    record = resp.json()["session"]
    assert record["idleMinutes"] == 45
    assert record["walltime"] == "01:30:00"
    assert record["port"] == 8090
    assert record["gres"] == "gpu:H100:1"
    assert record["partition"] == "gpu_p"
    assert record["gpuType"] == "H100"


def test_workbench_serves_session_controls(tmp_path, monkeypatch):
    # Web parity (plan acceptance criterion 5): the served workbench carries
    # the capability-gated session panel. No JS harness exists in this repo;
    # asserting the served page wires the session endpoint is the contract.
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    resp = TestClient(app).get("/")
    assert resp.status_code == 200
    assert "/api/session" in resp.text
    assert "gpuSession" in resp.text          # capability gate
    assert 'id="sessionPanel"' in resp.text


def test_session_start_requires_slurm_executor(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "controller")
    monkeypatch.setenv("STEERLAB_EXECUTOR", "local")
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    resp = _fresh_client().post("/api/session/start", json={})
    assert resp.status_code == 403
    assert "STEERLAB_EXECUTOR=slurm" in resp.json()["detail"]


# --- idle timer (pure unit tests) -------------------------------------------------


def _fake_clock():
    state = {"t": 0.0}

    def clock():
        return state["t"]

    return state, clock


def test_idle_timer_fires_once_when_idle():
    state, clock = _fake_clock()
    fired = []
    timer = gpu_session.IdleTimer(60, lambda: fired.append(1), clock=clock)
    state["t"] = 59.0
    assert timer.check() is False
    state["t"] = 61.0
    assert timer.check() is True
    assert fired == [1]
    state["t"] = 500.0
    assert timer.check() is False  # exactly once


def test_idle_timer_activity_resets_countdown():
    state, clock = _fake_clock()
    fired = []
    timer = gpu_session.IdleTimer(60, lambda: fired.append(1), clock=clock)
    state["t"] = 50.0
    timer.begin()
    timer.end()  # an allowlisted request completed at t=50
    state["t"] = 100.0
    assert timer.check() is False  # only 50s idle since the request
    assert timer.remaining_seconds() == 10
    state["t"] = 111.0
    assert timer.check() is True
    assert fired == [1]


def test_idle_timer_inflight_blocks_expiry():
    state, clock = _fake_clock()
    fired = []
    timer = gpu_session.IdleTimer(60, lambda: fired.append(1), clock=clock)
    timer.begin()  # a generation stream is running
    state["t"] = 10_000.0
    assert timer.busy is True
    assert timer.check() is False  # NEVER fires while work is in flight
    assert timer.remaining_seconds() == 60  # countdown pinned at full
    assert fired == []
    timer.end()
    state["t"] = 10_059.0
    assert timer.check() is False
    state["t"] = 10_061.0
    assert timer.check() is True


def test_idle_timer_disabled_when_zero():
    state, clock = _fake_clock()
    timer = gpu_session.IdleTimer(0, lambda: pytest.fail("must never fire"),
                                  clock=clock)
    assert timer.enabled is False
    assert timer.remaining_seconds() is None
    state["t"] = 1e9
    assert timer.check() is False


def test_activity_allowlist():
    # The app polls these every ~15s — they must never hold a GPU.
    for path in ("/api/capabilities", "/api/info", "/api/state",
                 "/api/session", "/api/runs", "/api/concepts",
                 "/api/vectors", "/healthz"):
        assert gpu_session.is_activity(path) is False, path
    # Real work does count — including the keepalive ping.
    for path in ("/api/load", "/api/models/unload", "/api/generate",
                 "/api/generate/stream", "/api/variant/generate/stream",
                 "/api/reader/score", "/api/session/keepalive",
                 "/api/concept/fear/stats", "/api/concept/fear/probe-train"):
        assert gpu_session.is_activity(path) is True, path


# --- worker-side integration -------------------------------------------------------


def test_worker_status_and_discovery_record(worker_client, tmp_path):
    body = worker_client.get("/api/session").json()
    assert body["sessionGeneration"] == "gen-w"
    assert body["busy"] is False
    assert body["state"] == "ready"
    assert 2600 <= body["idleRemainingSeconds"] <= 2700  # 45 min window
    record = json.loads((tmp_path / "meta" / "session-worker.json").read_text())
    assert record["sessionGeneration"] == "gen-w"
    assert record["port"] == 9191
    assert record["node"]
    assert record["startedAt"]
    assert record["state"] == "ready"


def test_worker_capability_polls_do_not_reset_idle(worker_client):
    worker = gpu_session.ensure_worker()
    t0 = worker.timer.last_activity
    time.sleep(0.02)
    for _ in range(3):
        assert worker_client.get("/api/capabilities").status_code == 200
    assert worker_client.get("/api/state").status_code == 200
    assert worker_client.get("/api/session").status_code == 200
    assert worker.timer.last_activity == t0  # polling never counts
    resp = worker_client.post("/api/session/keepalive")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert worker.timer.last_activity > t0  # the explicit ping does


def test_worker_state_reports_empty_jobs(worker_client):
    body = worker_client.get("/api/state").json()
    assert body["jobs"] == []


# --- the streaming reverse proxy ---------------------------------------------------


def _fake_worker_app(release: threading.Event):
    worker = FastAPI()

    @worker.post("/api/generate")
    async def generate(request: Request):
        return {"who": "worker", "auth": request.headers.get("authorization"),
                "echo": await request.json()}

    @worker.post("/api/generate/stream")
    def generate_stream():
        def chunks():
            yield "data: one\n\n"
            release.wait(timeout=5)
            yield "data: two\n\n"

        return StreamingResponse(chunks(), media_type="text/event-stream")

    @worker.get("/api/jobs")
    def jobs():
        return {"who": "worker"}

    @worker.post("/api/concept/{name}/stats")
    def concept_stats(name: str):
        return {"who": "worker"}

    @worker.post("/api/multiconcept/extract")
    def mc_extract():
        # The worker's job-backed verbs answer SYNCHRONOUSLY (no job
        # subsystem): the shape _run_or_submit produces on a gpu-session.
        return {"ok": True, "sync": True,
                "result": {"runDirectory": "/runs/x", "concepts": ["anger"]},
                "logs": ["built 2 grand-mean vectors"]}

    @worker.post("/api/vectors/backfill-norms")
    def backfill_norms():
        # Same sync shape: the worker snapshots ITS active model and measures.
        return {"ok": True, "sync": True,
                "result": {"runDirectory": "/runs/bf",
                           "artifact": "/runs/bf/fear",
                           "residualNormSource": "neutral-corpus",
                           "layerCount": 2},
                "logs": ["measured 2-layer residual norms"]}

    @worker.post("/api/session/keepalive")
    def keepalive(request: Request):
        # Echo the auth header so tests can prove the controller fills in
        # its own token when the client sent none (2026-07-17 fix 1).
        return {"ok": True, "idleRemainingSeconds": 99, "busy": False,
                "auth": request.headers.get("authorization")}

    return worker


def test_proxy_forwards_allowlisted_route_with_auth(proxy_controller,
                                                    monkeypatch):
    worker = _fake_worker_app(threading.Event())
    monkeypatch.setattr(gpu_session, "_TRANSPORT",
                        httpx.ASGITransport(app=worker))
    _install_session_records()
    client = TestClient(app)
    resp = client.post("/api/generate", json={"text": "hi"},
                       headers={"Authorization": "Bearer tok-abc"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["who"] == "worker"
    assert body["auth"] == "Bearer tok-abc"   # forwarded verbatim
    assert body["echo"] == {"text": "hi"}     # body forwarded verbatim


def test_proxy_forwards_only_synchronous_authoring_compute(
        proxy_controller, monkeypatch):
    # Concept stats is genuinely synchronous → proxied. The JOB-backed verbs
    # must NOT proxy (engineer review 2026-07-18: the worker has no job
    # subsystem, so proxying them answered 409 "no job subsystem") — they
    # stay controller-owned and DELEGATE via a controller job instead.
    worker = _fake_worker_app(threading.Event())
    monkeypatch.setattr(gpu_session, "_TRANSPORT",
                        httpx.ASGITransport(app=worker))
    _install_session_records()
    client = TestClient(app)
    body = client.post("/api/concept/anger/stats", json={},
                       headers={"Authorization": "Bearer tok-abc"}).json()
    assert body.get("who") == "worker"


def test_should_proxy_scopes_the_concept_verbs():
    assert gpu_session.should_proxy("/api/concept/anger/stats")
    # Job-backed verbs stay on the controller (single-writer rule): the
    # controller's job DELEGATES to the worker via call_worker_sync.
    assert not gpu_session.should_proxy("/api/concept/anger/extract")
    assert not gpu_session.should_proxy("/api/multiconcept/extract")
    assert not gpu_session.should_proxy("/api/reader/fit")
    assert not gpu_session.should_proxy("/api/vectors/backfill-norms")
    assert not gpu_session.should_proxy("/api/extract")
    assert not gpu_session.should_proxy("/api/concept/anger/probe-train")
    # …and authoring CRUD (shared-filesystem file ops) never proxies.
    assert not gpu_session.should_proxy("/api/concept/anger/save")
    assert not gpu_session.should_proxy("/api/concept/anger/delete")
    assert not gpu_session.should_proxy("/api/concepts")


def test_job_backed_authoring_delegates_through_a_controller_job(
        proxy_controller, monkeypatch):
    # The full delegation shape (engineer review 2026-07-18): the client
    # POSTs the controller's route and gets a jobId (contract unchanged);
    # the controller JOB re-POSTs the body to the worker's synchronous
    # route and relays its result and log lines into the durable job log.
    worker = _fake_worker_app(threading.Event())
    monkeypatch.setattr(gpu_session, "_TRANSPORT",
                        httpx.ASGITransport(app=worker))
    _install_session_records()
    client = TestClient(app)
    resp = client.post("/api/multiconcept/extract",
                       json={"targets": ["anger"]},
                       headers={"Authorization": "Bearer tok-abc"})
    assert resp.status_code == 200
    job_id = resp.json()["jobId"]

    deadline = time.monotonic() + 10
    job = None
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}",
                         headers={"Authorization": "Bearer tok-abc"}).json()
        if job.get("finishedAt"):
            break
        time.sleep(0.05)
    assert job is not None and job.get("finishedAt"), job
    assert job.get("error") in (None, ""), job
    log_text = "\n".join(job.get("logLines") or []) + str(job)
    assert "delegating grandmean-extract" in log_text
    assert "[worker] built 2 grand-mean vectors" in log_text


def _install_backfillable_artifact(tmp_path):
    """A norms-less vector artifact + neutral corpus under the workspace
    root, so /api/vectors/backfill-norms passes its synchronous validation
    on the controller (the shared-workspace reality: the controller SEES the
    artifact it cannot measure)."""
    from steerlab_server.steering.vector_store import (
        ConceptVectors, SteeringVectorSidecar, save)
    vectors = ConceptVectors(per_layer=[[1.0, 0.0], [2.0, 0.0]])
    run_dir = tmp_path / "runs" / "legacy"
    save(vectors,
         SteeringVectorSidecar.make(model_id="org/m", concept="fear",
                                    stimulus_set_hash="h", vectors=vectors),
         str(run_dir), "fear")
    corpus = tmp_path / "prompts" / "neutral.jsonl"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text(
        "\n".join('{"text": "line %d"}' % i for i in range(6)) + "\n")
    return {"vectorID": "runs/legacy/fear",
            "neutralCorpusPath": "prompts/neutral.jsonl"}


def test_backfill_norms_delegates_through_a_controller_job(
        proxy_controller, tmp_path, monkeypatch):
    # The 2026-08-06 live failure: backfill-norms needs a forward pass but
    # ran controller-side in NEITHER cluster dispatch class — not proxied
    # like interactive verbs, not delegated like the authoring jobs — so it
    # 409'd with an idle GPU session serving, and the "backfill norms"
    # remedy every norm-units refusal names was a dead end on exactly the
    # deployment that runs studies. It now rides the reader-fit delegation
    # seam: jobId contract unchanged, the controller job re-POSTs the body
    # to the worker, and with no explicit modelID the WORKER snapshots its
    # own active model (the controller holds none by design).
    body = _install_backfillable_artifact(tmp_path)
    worker = _fake_worker_app(threading.Event())
    monkeypatch.setattr(gpu_session, "_TRANSPORT",
                        httpx.ASGITransport(app=worker))
    _install_session_records()
    client = TestClient(app)
    resp = client.post("/api/vectors/backfill-norms", json=body,
                       headers={"Authorization": "Bearer tok-abc"})
    assert resp.status_code == 200, resp.text
    job_id = resp.json()["jobId"]

    deadline = time.monotonic() + 10
    job = None
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}",
                         headers={"Authorization": "Bearer tok-abc"}).json()
        if job.get("finishedAt"):
            break
        time.sleep(0.05)
    assert job is not None and job.get("finishedAt"), job
    assert job.get("error") in (None, ""), job
    log_text = "\n".join(job.get("logLines") or []) + str(job)
    assert "delegating vector:backfill-norms" in log_text
    assert "[worker] measured 2-layer residual norms" in log_text
    assert (job.get("result") or {}).get("residualNormSource") == "neutral-corpus"


def test_backfill_norms_without_session_names_the_flow(
        proxy_controller, tmp_path):
    # Controller with NO session: the actionable session-flow refusal, NOW —
    # never a queued job that dies on the role gate minutes later.
    body = _install_backfillable_artifact(tmp_path)
    client = TestClient(app)
    resp = client.post("/api/vectors/backfill-norms", json=body)
    assert resp.status_code == 409
    assert "no GPU session running" in resp.json()["detail"]


def test_call_worker_sync_names_the_missing_session():
    # No session records at all: the delegation seam answers the actionable
    # session-flow message, never a connection traceback.
    with pytest.raises(RuntimeError, match="no GPU session worker"):
        gpu_session.call_worker_sync("/api/multiconcept/extract", {})


def test_worker_status_reports_actual_gpu_capacity(worker_client,
                                                   monkeypatch):
    # The client's memory-fit gating needs CUDA's ACTUAL total (an "L4
    # 24 GB" reports 22.05 GiB usable), never the profile's marketing GB —
    # the GiB/GB confusion passed the exact 12B-on-L4 case (engineer review
    # 2026-07-18). Off-GPU workers simply omit the field.
    monkeypatch.setattr(gpu_session, "_GPU_TOTAL_MEMORY_CACHE", 23_672_599_552)
    body = worker_client.get("/api/session").json()
    assert body["gpuTotalMemoryBytes"] == 23_672_599_552
    monkeypatch.setattr(gpu_session, "_GPU_TOTAL_MEMORY_CACHE", None)
    assert "gpuTotalMemoryBytes" not in worker_client.get("/api/session").json()


def test_reconcile_persists_worker_gpu_capacity(controller_env, monkeypatch):
    _install_session_records(generation="gen-vram", node="gpu-node-7",
                             port=8081)
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "RUNNING"}}))
    monkeypatch.setattr(
        gpu_session, "probe_worker",
        lambda node, port, timeout=2.0: {
            "state": "ready", "busy": False, "idleRemainingSeconds": 900,
            "gpuTotalMemoryBytes": 23_672_599_552})
    body = controller_env["client"].get("/api/session",
                                        headers=AUTH).json()["session"]
    assert body["gpuTotalMemoryBytes"] == 23_672_599_552
    # Persisted in the record: it survives a later probe gap.
    assert gpu_session.read_session_record()["gpuTotalMemoryBytes"] \
        == 23_672_599_552


def test_worker_runs_authoring_compute_synchronously_not_via_jobs(
        worker_client):
    # THE REAL ROUTER under gpu-session role (engineer review 2026-07-18:
    # the earlier fake-worker test proved routing, not viability). With no
    # model loaded, the job-backed authoring verbs must refuse on the MODEL
    # — reaching the job subsystem would 409 "no job subsystem", the exact
    # regression this seam exists to prevent.
    for path in ("/api/concept/anger/extract", "/api/multiconcept/extract"):
        resp = worker_client.post(path, json={})
        assert resp.status_code == 409, path
        detail = resp.json()["detail"]
        assert "no model loaded" in detail, (path, detail)
        assert "job subsystem" not in detail, (path, detail)


def test_proxy_never_forwards_controller_owned_routes(proxy_controller,
                                                      monkeypatch):
    worker = _fake_worker_app(threading.Event())
    monkeypatch.setattr(gpu_session, "_TRANSPORT",
                        httpx.ASGITransport(app=worker))
    _install_session_records()
    body = TestClient(app).get("/api/jobs").json()
    assert body.get("who") != "worker"
    assert "jobs" in body  # the controller's own job list answered


def test_proxy_keepalive_forwards_to_worker(proxy_controller, monkeypatch):
    worker = _fake_worker_app(threading.Event())
    monkeypatch.setattr(gpu_session, "_TRANSPORT",
                        httpx.ASGITransport(app=worker))
    _install_session_records()
    resp = TestClient(app).post("/api/session/keepalive")
    assert resp.status_code == 200
    assert resp.json()["idleRemainingSeconds"] == 99


def test_keepalive_409_without_session(proxy_controller):
    resp = TestClient(app).post("/api/session/keepalive")
    assert resp.status_code == 409
    assert "no GPU session running" in resp.json()["detail"]


def test_proxy_worker_down_maps_502(proxy_controller):
    # A real closed port: the connect refusal must surface as the actionable
    # 502, not a raw httpx traceback.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    closed_port = probe.getsockname()[1]
    probe.close()
    _install_session_records(port=closed_port)
    resp = TestClient(app).post("/api/generate", json={"text": "hi"})
    assert resp.status_code == 502
    assert "may have expired" in resp.json()["detail"]


def test_no_session_means_zero_regression(proxy_controller):
    # Without a session the controller's own routes answer exactly as today.
    resp = TestClient(app).post("/api/generate", json={"text": "hi"})
    assert resp.status_code == 409
    # The controller's answer points at the session flow — "load a model
    # first" is a dead end on a controller (2026-07-18).
    assert "no GPU session running" in resp.json()["detail"]


def _install_undiscovered_session(state, *, state_detail=None,
                                  started_at=None):
    """Session record WITHOUT a worker discovery record — the live 2026-07-17
    shape (worker job submitted/running but its record never appeared)."""
    gpu_session.write_session_record({
        "sessionGeneration": "gen-undiscovered", "slurmJobID": "424242",
        "node": None, "port": 8081, "state": state,
        "workspaceRoot": "/tmp", "gpuType": "A100", "gres": "gpu:A100:1",
        "partition": None, "walltime": "02:00:00", "idleMinutes": 30,
        "submittedAt": "2026-07-17T00:00:00+00:00", "startedAt": started_at,
        "expiresAt": None, "serverVersion": "test", "role": "gpu-session",
        "stateDetail": state_detail,
    })


def test_proxy_blocks_queued_session_with_actionable_503(proxy_controller):
    # Live failure 2026-07-17: with a session started but its worker not yet
    # discovered, /api/load fell through to the controller's role refusal
    # ("no GPU session running: start one" — right after one WAS started) and
    # a variant-generate fell through and hung. A NON-terminal session must
    # block the allowlisted routes with a 503 that explains the wait.
    _install_undiscovered_session("queued")
    client = TestClient(app)
    for path, method, body in (("/api/load", "post", {"model": "org/x"}),
                               ("/api/variant/generate", "post", {}),
                               ("/api/generate", "post", {"text": "hi"})):
        resp = getattr(client, method)(path, json=body)
        assert resp.status_code == 503, path
        detail = resp.json()["detail"]
        assert detail == ("GPU session is queued (job 424242) — wait for it "
                          "to start"), path


def test_proxy_blocks_starting_session_and_carries_the_diagnosis(
        proxy_controller):
    diagnosis = ("Slurm reports job 424242 running for 300s but the worker "
                 "has not written its discovery record — check the job's "
                 "slurm-*.err log in its bundle directory (the worker may "
                 "have failed at startup)")
    _install_undiscovered_session(
        "starting", state_detail=diagnosis,
        started_at="2026-07-17T00:00:05+00:00")
    resp = TestClient(app).post("/api/generate", json={"text": "hi"})
    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert detail.startswith("GPU session worker is starting (job 424242)")
    assert diagnosis in detail  # the existing starting-diagnosis rides along


def test_proxy_blocks_unknown_session_with_its_state_detail(proxy_controller):
    unknown_detail = gpu_session._unknown_detail({"slurmJobID": "424242"})
    _install_undiscovered_session(
        "unknown", state_detail=unknown_detail,
        started_at="2026-07-17T00:00:05+00:00")
    resp = TestClient(app).post("/api/load", json={"model": "org/x"})
    assert resp.status_code == 503
    assert resp.json()["detail"] == unknown_detail


def test_proxy_falls_through_after_terminal_session(proxy_controller):
    # Only NO session or a TERMINAL one falls through to today's behavior.
    _install_undiscovered_session("ended")
    resp = TestClient(app).post("/api/generate", json={"text": "hi"})
    assert resp.status_code == 409
    # The controller's answer points at the session flow — "load a model
    # first" is a dead end on a controller (2026-07-18).
    assert "no GPU session running" in resp.json()["detail"]
    _install_undiscovered_session("failed")
    resp = TestClient(app).post("/api/load", json={"model": "org/x"})
    assert resp.status_code == 409
    assert "no GPU session running" in resp.json()["detail"]


# --- 2026-07-17 live fixes: authenticated probe classification, /api/state
# --- composition, ending-closure proof, allocation-unique ports, SSE guard ----


class _FakeProbeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


def _stub_httpx_get(monkeypatch, response, seen):
    def fake_get(url, headers=None, timeout=None):
        seen["url"] = url
        seen["headers"] = dict(headers or {})
        return response

    monkeypatch.setattr(httpx, "get", fake_get)


def test_probe_worker_sends_controller_bearer_token(monkeypatch):
    # THE live 2026-07-17 defect: the probe reached a token-enforcing worker
    # unauthenticated, was 401'd, and reconcile read "not answering".
    seen = {}
    _stub_httpx_get(monkeypatch,
                    _FakeProbeResponse(200, {"state": "ready", "busy": False}),
                    seen)
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN", "sekrit")
    result = gpu_session.probe_worker("c5-23", 8081)
    assert result == {"state": "ready", "busy": False}   # 200 path unchanged
    assert seen["url"] == "http://c5-23:8081/api/session"
    assert seen["headers"]["Authorization"] == "Bearer sekrit"


def test_probe_worker_hydrates_token_from_file(monkeypatch, tmp_path):
    # Same token FILE both processes hydrate from: a controller whose env
    # never exported the token must still authenticate its probes.
    (tmp_path / "token").write_text("tok-999\n")
    monkeypatch.delenv("STEERLAB_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN_FILE", str(tmp_path / "token"))
    seen = {}
    _stub_httpx_get(monkeypatch, _FakeProbeResponse(200, {"state": "idle"}),
                    seen)
    assert gpu_session.probe_worker("c5-23", 8081) == {"state": "idle"}
    assert seen["headers"]["Authorization"] == "Bearer tok-999"


@pytest.mark.parametrize("status", [401, 403])
def test_probe_worker_classifies_auth_rejection(monkeypatch, status):
    seen = {}
    _stub_httpx_get(
        monkeypatch,
        _FakeProbeResponse(status,
                           {"detail": "missing or invalid bearer token"}),
        seen)
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN", "sekrit")
    assert gpu_session.probe_worker("c5-23", 8081) \
        is gpu_session.PROBE_AUTH_REJECTED


def test_auth_rejected_probe_gets_token_mismatch_diagnosis(controller_env,
                                                           monkeypatch):
    # Live 2026-07-17: worker up and enforcing auth (a login-node curl
    # answered the 401 detail) but the session sat at Starting with the
    # "not answering" diagnosis. 401 must produce the DISTINCT config
    # diagnosis, stay nonterminal, and recover once the tokens align.
    _install_session_records(generation="gen-auth", node="c5-23", port=8081,
                             session_state="starting")
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "RUNNING"}}))
    monkeypatch.setattr(
        gpu_session, "probe_worker",
        lambda node, port, timeout=2.0: gpu_session.PROBE_AUTH_REJECTED)
    client = controller_env["client"]
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "starting"                       # nonterminal
    detail = body["stateDetail"]
    assert "c5-23:8081" in detail
    assert "rejected the controller's bearer token" in detail
    assert "STEERLAB_AUTH_TOKEN" in detail                   # the remedy
    assert "not answering" not in detail                     # old misdiagnosis
    # Tokens fixed → the next successful probe converges the session.
    monkeypatch.setattr(
        gpu_session, "probe_worker",
        lambda node, port, timeout=2.0: {"state": "ready", "busy": False,
                                         "idleRemainingSeconds": 1000})
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "ready"
    assert body["stateDetail"] is None


def test_keepalive_fills_in_controller_token(proxy_controller, monkeypatch):
    worker = _fake_worker_app(threading.Event())
    monkeypatch.setattr(gpu_session, "_TRANSPORT",
                        httpx.ASGITransport(app=worker))
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN", "ctl-token")
    _install_session_records()
    client = TestClient(app)
    # Client sent no Authorization: the controller's own token fills in —
    # the worker runs token mode on EVERY route.
    resp = client.post("/api/session/keepalive")
    assert resp.status_code == 200
    assert resp.json()["auth"] == "Bearer ctl-token"
    # A client-supplied token still forwards verbatim.
    resp = client.post("/api/session/keepalive",
                       headers={"Authorization": "Bearer client-tok"})
    assert resp.json()["auth"] == "Bearer client-tok"


def test_state_composes_during_blocked_session(proxy_controller):
    # Fix 2 (engineer review 2026-07-17): /api/state is the Swift connection
    # handshake — it must answer 200 from a healthy controller even while
    # the session's worker is undialable; compute routes keep their 503.
    client = TestClient(app)
    for session_state in ("queued", "starting", "unknown"):
        _install_undiscovered_session(session_state)
        resp = client.get("/api/state")
        assert resp.status_code == 200, session_state
        body = resp.json()
        assert body["loadedModel"] is None, session_state
        assert isinstance(body["jobs"], list), session_state
        blocked = client.post("/api/load", json={"model": "org/x"})
        assert blocked.status_code == 503, session_state


def test_state_overlays_controller_jobs_over_reachable_worker(
        proxy_controller, monkeypatch):
    # Single-writer rule: the worker's jobs array is ALWAYS empty, which
    # made the jobs badge read zero during a live session — the proxied
    # /api/state carries the controller's own jobs instead.
    worker = FastAPI()

    @worker.get("/api/state")
    def worker_state():
        return {"who": "worker", "loadedModel": "org/tiny",
                "jobs": [{"id": "worker-phantom"}]}

    monkeypatch.setattr(gpu_session, "_TRANSPORT",
                        httpx.ASGITransport(app=worker))
    _install_session_records()
    resp = TestClient(app).get("/api/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["who"] == "worker"                # the worker's state answered
    assert body["loadedModel"] == "org/tiny"
    assert isinstance(body["jobs"], list)
    assert not any(j.get("id") == "worker-phantom" for j in body["jobs"])


def test_state_falls_back_to_controller_when_worker_state_errors(
        proxy_controller, monkeypatch):
    # A worker that errors on /api/state must not fail the handshake: the
    # controller answers its own state (degraded beats broken).
    worker = FastAPI()

    @worker.get("/api/state")
    def worker_state():
        return JSONResponse({"detail": "boom"}, status_code=500)

    monkeypatch.setattr(gpu_session, "_TRANSPORT",
                        httpx.ASGITransport(app=worker))
    _install_session_records()
    resp = TestClient(app).get("/api/state")
    assert resp.status_code == 200
    body = resp.json()
    assert "loadedModel" in body                  # the controller's own shape
    assert body.get("who") != "worker"
    assert isinstance(body["jobs"], list)


def test_ending_with_failing_scheduler_query_stays_ending(controller_env,
                                                          monkeypatch):
    # 2026-07-17 review, finding 1: polling ERRORS and positive absence both
    # collapsed to None, so an "ending" session could close — releasing the
    # double-start slot — on scheduler silence, without proof the billed
    # allocation stopped. A failed query must keep "ending".
    client = controller_env["client"]
    client.post("/api/session/start", json={}, headers=AUTH)
    assert client.delete("/api/session", headers=AUTH).status_code == 200
    # Age past the visibility grace: the OLD rule would now have closed it.
    record = gpu_session.read_session_record()
    record["submittedAt"] = (datetime.now(timezone.utc)
                             - timedelta(hours=2)).isoformat()
    gpu_session.write_session_record(record)
    monkeypatch.setenv("FAKE_SACCT_FAIL", "sacct: error: connection refused")
    monkeypatch.setenv("FAKE_SQUEUE_FAIL", "squeue: error: connection refused")
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "ending"                       # still occupied
    assert "scheduler query is failing" in body["stateDetail"]
    assert "424242" in body["stateDetail"]
    assert client.post("/api/session/start", json={},
                       headers=AUTH).status_code == 409
    # The scheduler answers again — positive absence from a SUCCESSFUL query
    # (fake sacct exits 0 with no rows) — NOW the stop confirms.
    monkeypatch.delenv("FAKE_SACCT_FAIL")
    monkeypatch.delenv("FAKE_SQUEUE_FAIL")
    body = client.get("/api/session", headers=AUTH).json()["session"]
    assert body["state"] == "ended"


def test_ending_closes_on_positive_terminal_state(controller_env):
    client = controller_env["client"]
    client.post("/api/session/start", json={}, headers=AUTH)
    client.delete("/api/session", headers=AUTH)
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "COMPLETED"}}))
    assert client.get("/api/session",
                      headers=AUTH).json()["session"]["state"] == "ended"


def test_poll_state_detailed_disambiguates(controller_env, monkeypatch):
    from steerlab_server.api.executors import SlurmExecutor
    executor = SlurmExecutor()
    # Positive absence: sacct succeeds with no rows for the job.
    assert executor.poll_state_detailed("424242") == (None, True)
    # Query failure: both commands error — silence about the conversation.
    monkeypatch.setenv("FAKE_SACCT_FAIL", "sacct: error")
    monkeypatch.setenv("FAKE_SQUEUE_FAIL", "squeue: error")
    assert executor.poll_state_detailed("424242") == (None, False)
    # Positive state still reads through (and poll_state stays a thin view).
    monkeypatch.delenv("FAKE_SACCT_FAIL")
    monkeypatch.delenv("FAKE_SQUEUE_FAIL")
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "RUNNING"}}))
    assert executor.poll_state_detailed("424242") == ("running", True)
    assert executor.poll_state("424242") == "running"


def test_derive_session_port_from_allocation(monkeypatch):
    # Fix 4 (engineer finding 3): fixed 8081 collides when two session jobs
    # land on one multi-GPU node — the port derives from the allocation id.
    monkeypatch.setenv("SLURM_JOB_ID", "1234567")
    assert gpu_session.derive_session_port() == 10000 + (1234567 % 50000)
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    assert gpu_session.derive_session_port() == 10042
    monkeypatch.delenv("SLURM_JOB_ID")
    assert gpu_session.derive_session_port() == gpu_session.DEFAULT_PORT
    monkeypatch.setenv("SLURM_JOB_ID", "not-a-job")
    assert gpu_session.derive_session_port() == gpu_session.DEFAULT_PORT


def test_session_start_defaults_to_auto_port(controller_env):
    resp = controller_env["client"].post("/api/session/start", json={},
                                         headers=AUTH)
    assert resp.status_code == 200
    record = resp.json()["session"]
    assert record["port"] is None            # resolved by the worker
    script = _session_script(controller_env)
    assert "STEERLAB_SESSION_PORT=auto" in script
    assert "--port" not in script            # the worker derives its own


def test_session_start_explicit_port_still_wins(controller_env):
    resp = controller_env["client"].post("/api/session/start",
                                         json={"port": 8090}, headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["session"]["port"] == 8090
    script = _session_script(controller_env)
    assert "--port 8090" in script
    assert "STEERLAB_SESSION_PORT=8090" in script


def test_session_shape_comes_from_the_site_not_the_constants(controller_env,
                                                             monkeypatch):
    """WP5 Step 9 (audit c20): `scheduler.gpuSession` reaches the worker job.

    The controller sources the rendered site env file, so the session class's
    declared facts arrive as STEERLAB_SESSION_* keys. They beat the site's
    STUDY defaults (which beat the module constants), and a request parameter
    still beats everything."""
    monkeypatch.setenv("STEERLAB_SLURM_PARTITION", "gpu_p")
    monkeypatch.setenv("STEERLAB_SLURM_MEMORY", "120G")
    monkeypatch.setenv("STEERLAB_SLURM_GRES", "gpu:A100:1")
    monkeypatch.setenv("STEERLAB_SESSION_PARTITION", "gpu_short")
    monkeypatch.setenv("STEERLAB_SESSION_CPUS", "12")
    monkeypatch.setenv("STEERLAB_SESSION_MEMORY", "96G")
    monkeypatch.setenv("STEERLAB_SESSION_WALLTIME", "04:00:00")
    monkeypatch.setenv("STEERLAB_SESSION_GRES", "gpu:H100:1")
    monkeypatch.setenv("STEERLAB_SESSION_IDLE_MINUTES", "45")
    monkeypatch.setenv("STEERLAB_SESSION_EXTRA_SBATCH", "--nice=100")
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "A100,H100")
    resp = controller_env["client"].post("/api/session/start", json={},
                                         headers=AUTH)
    assert resp.status_code == 200, resp.json()
    record = resp.json()["session"]
    assert record["idleMinutes"] == 45
    script = _session_script(controller_env)
    assert "#SBATCH --partition=gpu_short" in script   # session over study
    assert "#SBATCH --cpus-per-task=12" in script
    assert "#SBATCH --mem=96G" in script
    assert "#SBATCH --time=04:00:00" in script
    assert "#SBATCH --gres=gpu:H100:1" in script
    assert "#SBATCH --nice=100" in script
    # The session shape never turns the hard rules back on.
    assert "#SBATCH --requeue" not in script
    assert "--signal=" not in script

    # A request parameter still wins over the site's declaration.
    controller_env["client"].delete("/api/session", headers=AUTH)
    controller_env["state_file"].write_text(json.dumps(
        {"424242": {"state": "CANCELLED"}}))
    controller_env["client"].get("/api/session", headers=AUTH)
    resp = controller_env["client"].post(
        "/api/session/start", json={"memory": "48G", "idleMinutes": 10},
        headers=AUTH)
    assert resp.status_code == 200, resp.json()
    assert resp.json()["session"]["idleMinutes"] == 10
    scripts = sorted((controller_env["tmp"] / "runs").glob("*/slurm/run.sbatch"))
    assert "#SBATCH --mem=48G" in scripts[-1].read_text()


def test_session_site_port_declaration_is_honored_and_bounded(controller_env,
                                                              monkeypatch):
    """A site MAY pin the worker port (`scheduler.gpuSession.port`). It behaves
    exactly like an explicit request port — and an out-of-range declaration is
    an authoring error the operator sees, not a number silently clamped."""
    monkeypatch.setenv("STEERLAB_SESSION_DEFAULT_PORT", "10500")
    resp = controller_env["client"].post("/api/session/start", json={},
                                         headers=AUTH)
    assert resp.status_code == 200, resp.json()
    assert resp.json()["session"]["port"] == 10500
    assert "--port 10500" in _session_script(controller_env)

    monkeypatch.setenv("STEERLAB_SESSION_DEFAULT_PORT", "80")
    controller_env["client"].delete("/api/session", headers=AUTH)
    controller_env["state_file"].write_text(json.dumps(
        {"424242": {"state": "CANCELLED"}}))
    controller_env["client"].get("/api/session", headers=AUTH)
    resp = controller_env["client"].post("/api/session/start", json={},
                                         headers=AUTH)
    assert resp.status_code == 400
    assert "STEERLAB_SESSION_DEFAULT_PORT" in resp.json()["detail"]


def test_worker_derives_and_advertises_actual_port(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "gpu-session")
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    monkeypatch.setenv("STEERLAB_SESSION_GENERATION", "gen-port")
    monkeypatch.setenv("STEERLAB_SESSION_IDLE_MINUTES", "0")
    monkeypatch.setenv("STEERLAB_SESSION_PORT", "auto")
    monkeypatch.setenv("SLURM_JOB_ID", "70123")
    gpu_session.reset_worker_for_tests()
    try:
        worker = gpu_session.ensure_worker()
        expected = 10000 + (70123 % 50000)
        assert worker.port == expected
        # The discovery record carries the ACTUAL port — the record the
        # controller's reconcile and proxy target prefer.
        record = json.loads(
            (tmp_path / "meta" / "session-worker.json").read_text())
        assert record["port"] == expected
    finally:
        gpu_session.reset_worker_for_tests()


def test_reconcile_folds_worker_port_into_record(controller_env, monkeypatch):
    # Auto-port sessions carry port null until the worker advertises the one
    # it actually bound; reconcile folds it in so diagnosis texts and the UI
    # name the real port.
    _install_session_records(generation="gen-fold", node="gpu-node-1",
                             port=30123)
    record = gpu_session.read_session_record()
    record["port"] = None
    gpu_session.write_session_record(record)
    controller_env["state_file"].write_text(
        json.dumps({"424242": {"state": "RUNNING"}}))
    monkeypatch.setattr(
        gpu_session, "probe_worker",
        lambda node, port, timeout=2.0: {"state": "ready", "busy": False,
                                         "idleRemainingSeconds": 5})
    body = controller_env["client"].get("/api/session",
                                        headers=AUTH).json()["session"]
    assert body["port"] == 30123


def test_cli_serve_derives_auto_port(tmp_path, monkeypatch):
    from steerlab_server import cli
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "gpu-session")
    monkeypatch.setenv("STEERLAB_BIND", "127.0.0.1")   # loopback: no token gate
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    # cli._serve mutates STEERLAB_SESSION_PORT via os.environ; setenv
    # registers the restore (same pattern as the startup-guard test).
    monkeypatch.setenv("STEERLAB_SESSION_PORT", "auto")
    monkeypatch.setenv("SLURM_JOB_ID", "61234")
    calls = {}
    fake_uvicorn = types.SimpleNamespace(
        run=lambda *args, **kwargs: calls.update(args=args, kwargs=kwargs))
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    assert cli.main(["serve"]) == 0
    expected = 10000 + (61234 % 50000)
    assert calls["kwargs"]["port"] == expected
    # The env now advertises the RESOLVED number for the discovery record.
    assert os.environ["STEERLAB_SESSION_PORT"] == str(expected)
    # An explicit --port still wins over auto.
    assert cli.main(["serve", "--port", "9555"]) == 0
    assert calls["kwargs"]["port"] == 9555
    assert os.environ["STEERLAB_SESSION_PORT"] == "9555"


def test_workbench_sse_guard_present(tmp_path, monkeypatch):
    # Fix 5 (engineer finding 4): streamSSE must check res.ok before reading
    # the body as a stream — a blocked-session 503 otherwise rendered as an
    # EMPTY answer instead of its diagnostic detail. No JS harness exists in
    # this repo; asserting the served page carries the guard is the contract.
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    text = TestClient(app).get("/").text
    idx = text.find("async function streamSSE")
    assert idx >= 0
    snippet = text[idx:idx + 900]
    assert "if(!res.ok)" in snippet
    assert "detail" in snippet                 # surfaces the JSON error body
    assert "throw new Error" in snippet        # through the existing display


def test_controller_acquire_model_refuses_before_any_load(tmp_path,
                                                          monkeypatch):
    # Belt-and-braces at the chokepoint: EVERY in-process acquisition path
    # (variant generate, reader fit/score, norm backfill, experiment verbs)
    # funnels through ServiceState.acquire_model — on role=controller it must
    # answer the same 409 family as /api/load, never touch the loader.
    from fastapi import HTTPException

    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "controller")
    monkeypatch.delenv("STEERLAB_ALLOW_RESIDENT_MODELS", raising=False)
    state = ServiceState()

    def exploding_acquire(*args, **kwargs):
        raise AssertionError("controller attempted an in-process model load")

    monkeypatch.setattr(state.registry, "acquire", exploding_acquire)
    monkeypatch.setattr(state.registry, "get_or_load", exploding_acquire)
    with pytest.raises(HTTPException) as excinfo:
        state.acquire_model("google/gemma-3-4b-it")
    assert excinfo.value.status_code == 409
    assert "model loading is disabled on the controller" in excinfo.value.detail

    # The documented escape hatch keeps working: with it set, acquisition
    # proceeds to the registry (our stub proves the call happens).
    monkeypatch.setenv("STEERLAB_ALLOW_RESIDENT_MODELS", "1")
    with pytest.raises(AssertionError, match="in-process model load"):
        state.acquire_model("google/gemma-3-4b-it")


def _start_uvicorn(asgi_app):
    import uvicorn
    config = uvicorn.Config(asgi_app, host="127.0.0.1", port=0,
                            log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            pytest.fail("uvicorn did not start")
        time.sleep(0.01)
    return server, thread, server.servers[0].sockets[0].getsockname()[1]


def test_proxy_streams_sse_incrementally(proxy_controller):
    # The load-bearing proxy property: chunks pass through AS THEY ARRIVE.
    # The worker holds its second chunk behind an event for up to 5 s; if the
    # proxy buffered, the first chunk could only arrive after that wait.
    # BOTH ends run on real sockets: the ASGI test transport and TestClient
    # each buffer whole responses (verified empirically), so only a
    # socket-to-socket path can prove incremental arrival.
    pytest.importorskip("uvicorn")
    release = threading.Event()
    worker_server, worker_thread, worker_port = _start_uvicorn(
        _fake_worker_app(release))
    controller_server, controller_thread, controller_port = _start_uvicorn(app)
    try:
        _install_session_records(port=worker_port)
        started = time.monotonic()
        with httpx.Client() as client:
            with client.stream(
                    "POST",
                    f"http://127.0.0.1:{controller_port}/api/generate/stream"
            ) as resp:
                assert resp.status_code == 200
                lines = resp.iter_lines()
                first = next(line for line in lines if line.strip())
                first_arrival = time.monotonic() - started
                assert "one" in first
                # Incremental proof: the first chunk arrived while the worker
                # was still holding the second one behind the event.
                assert first_arrival < 3.0, (
                    f"first SSE chunk took {first_arrival:.1f}s — the proxy "
                    "is buffering the stream")
                release.set()
                rest = "\n".join(lines)
                assert "two" in rest
    finally:
        release.set()
        worker_server.should_exit = True
        controller_server.should_exit = True
        worker_thread.join(5)
        controller_thread.join(5)


# --- worker startup guard -------------------------------------------------------


def test_worker_serve_refuses_nonloopback_without_token(tmp_path, monkeypatch,
                                                        capsys):
    from steerlab_server import cli
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "gpu-session")
    monkeypatch.setenv("STEERLAB_BIND", "0.0.0.0")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    # cli._serve mutates these two via os.environ directly (token hydration,
    # port advertisement); setenv-then-delenv registers the restore so the
    # mutation cannot leak into later tests.
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN", "placeholder")
    monkeypatch.delenv("STEERLAB_AUTH_TOKEN")
    monkeypatch.setenv("STEERLAB_SESSION_PORT", "0")
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN_FILE", str(tmp_path / "absent"))
    assert cli.main(["serve"]) == 64
    assert "refusing to start" in capsys.readouterr().err

    # With a token FILE present the same start hydrates the env and proceeds
    # (the secret rides a file, never the sbatch bundle).
    (tmp_path / "token").write_text("tok-123\n")
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN_FILE", str(tmp_path / "token"))
    calls = {}
    fake_uvicorn = types.SimpleNamespace(
        run=lambda *args, **kwargs: calls.update(args=args, kwargs=kwargs))
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    assert cli.main(["serve"]) == 0
    assert os.environ["STEERLAB_AUTH_TOKEN"] == "tok-123"
    assert calls["kwargs"]["host"] == "0.0.0.0"


# --- experiment-verb delegation (2026-07-21: the live validate failure) --------
#
# With a LIVE session registered, the app's Validate Study still answered the
# controller's "no GPU session running" 409 — the 2026-07-18 delegation seam
# wired only the authoring verbs. experiment extract/validate now ride the
# same contract; run/sweep/pipeline refuse to the batch path immediately and
# honestly.


def _fake_experiment_worker(seen: dict):
    """A session worker double for the experiment verbs: records the re-POST
    target, writes a run directory into the SHARED workspace (the
    single-writer claim: the worker writes runs/, never jobs.sqlite), and
    answers the synchronous _run_or_submit shape."""
    worker = FastAPI()

    @worker.post("/api/experiment/{name}/{verb}")
    async def experiment_verb(name: str, verb: str, request: Request):
        seen["name"] = name
        seen["verb"] = verb
        seen["body"] = await request.json()
        run_dir = os.path.join(os.environ["STEERLAB_ROOT"], "runs",
                               f"exp-{name}-{verb}-delegated")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "validation-report.json"), "w",
                  encoding="utf-8") as handle:
            handle.write("{}")
        return {"ok": True, "sync": True,
                "result": {"runDirectory": run_dir},
                "logs": [f"{verb} finished for {name}"]}

    return worker


def _wait_for_job_payload(client, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    job = None
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job.get("finishedAt"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished: {job}")


def test_experiment_validate_delegates_through_a_controller_job(
        proxy_controller, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(gpu_session, "_TRANSPORT",
                        httpx.ASGITransport(app=_fake_experiment_worker(seen)))
    _install_session_records()
    client = TestClient(app)
    resp = client.post("/api/experiment/demo/validate", json={})
    assert resp.status_code == 200
    job = _wait_for_job_payload(client, resp.json()["jobId"])
    assert job.get("error") in (None, ""), job
    # Re-POST target is the SAME route on the worker, body forwarded verbatim.
    assert seen == {"name": "demo", "verb": "validate", "body": {}}
    log_text = "\n".join(job.get("logTail") or [])
    assert "delegating experiment:validate" in log_text
    assert "[worker] validate finished for demo" in log_text
    # The worker's validate run landed in the shared workspace and the
    # controller job relays it as the result.
    run_dir = (job.get("result") or {}).get("runDirectory")
    assert run_dir and os.path.isdir(run_dir), job
    assert os.path.exists(os.path.join(run_dir, "validation-report.json"))


def test_experiment_extract_rides_the_same_delegation_seam(
        proxy_controller, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(gpu_session, "_TRANSPORT",
                        httpx.ASGITransport(app=_fake_experiment_worker(seen)))
    _install_session_records()
    client = TestClient(app)
    resp = client.post("/api/experiment/demo/extract", json={})
    assert resp.status_code == 200
    job = _wait_for_job_payload(client, resp.json()["jobId"])
    assert job.get("error") in (None, ""), job
    assert seen["verb"] == "extract"


def test_experiment_validate_runs_synchronously_on_worker(worker_client,
                                                          monkeypatch):
    # gpu-session role: no job subsystem — the verb runs synchronously and
    # answers the {"ok","sync","result","logs"} shape the delegating
    # controller job consumes.
    from steerlab_server.experiment import tasks as tasks_mod

    def fake_validate(name, root=None, dtype="auto", device=None, *,
                      model_provider=None, should_cancel=None, log=None):
        (log or print)(f"validated {name}")
        return f"/runs/exp-{name}-validate"

    monkeypatch.setattr(tasks_mod, "validate", fake_validate)
    resp = worker_client.post("/api/experiment/demo/validate", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True and body["sync"] is True
    assert body["result"] == {"runDirectory": "/runs/exp-demo-validate",
                          "experiment": "demo"}
    assert any("validated demo" in line for line in body["logs"])


def test_experiment_validate_refuses_now_on_controller_without_session(
        proxy_controller):
    # No queued job dying on the role gate minutes later: the refusal is
    # immediate, and its text is TRUE (no session is registered).
    resp = TestClient(app).post("/api/experiment/demo/validate", json={})
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "no GPU session running" in detail
    assert "/api/session/start" in detail


def test_experiment_validate_workstation_unchanged(tmp_path, monkeypatch):
    # Default role: the durable in-process job, exactly as before.
    from steerlab_server.experiment import tasks as tasks_mod
    monkeypatch.delenv("STEERLAB_SERVER_ROLE", raising=False)
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    seen = {}

    def fake_validate(name, root=None, dtype="auto", device=None, *,
                      model_provider=None, should_cancel=None, log=None):
        seen["name"] = name
        return "/runs/exp-demo-validate"

    monkeypatch.setattr(tasks_mod, "validate", fake_validate)
    client = _fresh_client()
    resp = client.post("/api/experiment/demo/validate", json={})
    assert resp.status_code == 200
    job = _wait_for_job_payload(client, resp.json()["jobId"])
    assert job.get("error") in (None, ""), job
    assert seen == {"name": "demo"}
    assert job["result"] == {"runDirectory": "/runs/exp-demo-validate",
                         "experiment": "demo"}


@pytest.mark.parametrize("verb", ["run", "sweep", "pipeline"])
def test_batch_only_verbs_refuse_to_batch_with_session_registered(
        proxy_controller, verb):
    # Deliberately NOT delegated (they can outlive the session's idle timer
    # and walltime): the refusal is immediate, names the batch path, and
    # NEVER claims session absence while one is registered.
    _install_session_records()
    resp = TestClient(app).post(f"/api/experiment/demo/{verb}", json={})
    assert resp.status_code == 409, verb
    detail = resp.json()["detail"]
    assert "POST /api/studies/submit" in detail, detail
    assert "no GPU session running" not in detail, detail
    assert "a GPU session is registered" in detail, detail
    assert "batch-scale" in detail, detail


@pytest.mark.parametrize("verb", ["run", "sweep", "pipeline"])
def test_batch_only_verbs_refuse_to_batch_without_session(proxy_controller,
                                                          verb):
    # Session or no session, a controller answers the same honest remedy —
    # and does not send the caller off to start a session that would not
    # accept this work anyway.
    resp = TestClient(app).post(f"/api/experiment/demo/{verb}", json={})
    assert resp.status_code == 409, verb
    detail = resp.json()["detail"]
    assert "POST /api/studies/submit" in detail, detail
    assert "start one" not in detail, detail


@pytest.mark.parametrize("verb", ["run", "sweep", "pipeline"])
def test_batch_only_verbs_still_run_in_process_on_workstation(
        tmp_path, monkeypatch, verb):
    from steerlab_server.experiment import tasks as tasks_mod
    monkeypatch.delenv("STEERLAB_SERVER_ROLE", raising=False)
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))

    def fake_verb(name, *args, **kwargs):
        return f"/runs/exp-{name}-{verb}"

    monkeypatch.setattr(tasks_mod, verb, fake_verb)
    client = _fresh_client()
    resp = client.post(f"/api/experiment/demo/{verb}", json={})
    assert resp.status_code == 200, verb
    job = _wait_for_job_payload(client, resp.json()["jobId"])
    assert job.get("error") in (None, ""), job


def test_evaluate_delegates_only_for_local_judge_panels(proxy_controller,
                                                        monkeypatch):
    # A local-judge panel needs the session's GPU → delegate. A Claude panel
    # is CPU+network work that stays a plain controller job (compute nodes
    # may have no outbound HTTP).
    import types as types_mod

    from steerlab_server.api import routes as routes_mod
    from steerlab_server.experiment import tasks as tasks_mod

    seen: dict = {}
    monkeypatch.setattr(gpu_session, "_TRANSPORT",
                        httpx.ASGITransport(app=_fake_experiment_worker(seen)))
    _install_session_records()
    judges = [types_mod.SimpleNamespace(kind="local", name="local-judge")]
    fake_manifest = types_mod.SimpleNamespace(
        load=lambda name, root=None: types_mod.SimpleNamespace(judges=judges))
    monkeypatch.setattr(routes_mod, "Manifest", fake_manifest)
    client = TestClient(app)
    resp = client.post("/api/experiment/demo/evaluate", json={})
    assert resp.status_code == 200
    job = _wait_for_job_payload(client, resp.json()["jobId"])
    assert seen.get("verb") == "evaluate"
    assert "delegating experiment:evaluate" in "\n".join(job.get("logTail") or [])

    # Claude-judged panel: no delegation — the controller job runs evaluate
    # in-process (no model acquisition happens for API judges).
    seen.clear()
    judges[:] = [types_mod.SimpleNamespace(kind="claude", name="judge-a")]
    ran = {}

    def fake_evaluate(name, root=None, source_run=None, *, model_provider=None,
                      should_cancel=None, log=None,
                      allow_unverified_epoch=False, max_loaded=None,
                      resume_from=None, **_kwargs):
        ran["name"] = name
        return "/runs/exp-demo-evaluate"

    monkeypatch.setattr(tasks_mod, "evaluate", fake_evaluate)
    resp = client.post("/api/experiment/demo/evaluate", json={})
    assert resp.status_code == 200
    job = _wait_for_job_payload(client, resp.json()["jobId"])
    assert job.get("error") in (None, ""), job
    assert ran == {"name": "demo"}
    assert seen == {}  # the worker was never dialed


def test_evaluate_local_judge_without_session_refuses_now(proxy_controller,
                                                          monkeypatch):
    # External review 2026-07-22 (P1): a local-judge evaluate on a
    # controller with NO session used to queue a plain controller job that
    # died minutes later at acquire_model on the role gate. It now refuses
    # immediately, with the session-flow remedy.
    import types as types_mod

    from steerlab_server.api import routes as routes_mod

    judges = [types_mod.SimpleNamespace(kind="local", name="local-judge")]
    fake_manifest = types_mod.SimpleNamespace(
        load=lambda name, root=None: types_mod.SimpleNamespace(judges=judges))
    monkeypatch.setattr(routes_mod, "Manifest", fake_manifest)
    resp = TestClient(app).post("/api/experiment/demo/evaluate", json={})
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert "no GPU session running" in detail
    assert "/api/session/start" in detail


@pytest.mark.parametrize("panel", [
    [{"kind": "claude", "name": "judge-a"}],  # API judge: CPU+network work
    [],                                       # judge-free evaluate
])
def test_evaluate_without_local_judge_still_queues_without_session(
        proxy_controller, monkeypatch, panel):
    # Claude/OpenRouter panels and judge-free evaluates never need the
    # session's GPU — a controller with no session keeps running them as
    # plain controller jobs (unchanged).
    import types as types_mod

    from steerlab_server.api import routes as routes_mod
    from steerlab_server.experiment import tasks as tasks_mod

    judges = [types_mod.SimpleNamespace(**j) for j in panel]
    fake_manifest = types_mod.SimpleNamespace(
        load=lambda name, root=None: types_mod.SimpleNamespace(judges=judges))
    monkeypatch.setattr(routes_mod, "Manifest", fake_manifest)
    ran = {}

    def fake_evaluate(name, root=None, source_run=None, *, model_provider=None,
                      should_cancel=None, log=None,
                      allow_unverified_epoch=False, max_loaded=None,
                      resume_from=None, **_kwargs):
        ran["name"] = name
        return "/runs/exp-demo-evaluate"

    monkeypatch.setattr(tasks_mod, "evaluate", fake_evaluate)
    client = TestClient(app)
    resp = client.post("/api/experiment/demo/evaluate", json={})
    assert resp.status_code == 200
    job = _wait_for_job_payload(client, resp.json()["jobId"])
    assert job.get("error") in (None, ""), job
    assert ran == {"name": "demo"}


def test_evaluate_manifest_load_failure_keeps_the_fall_through(
        proxy_controller):
    # DELIBERATE (commented in routes.py): when the manifest cannot be
    # loaded, the panel inspection falls through to the plain controller
    # job, whose own error names the actual problem — more informative than
    # a routing refusal. "demo" does not exist under this root.
    client = TestClient(app)
    resp = client.post("/api/experiment/demo/evaluate", json={})
    assert resp.status_code == 200
    job = _wait_for_job_payload(client, resp.json()["jobId"])
    assert job.get("error"), job  # the job's own failure surfaces


def test_resident_load_refusal_is_session_aware(proxy_controller):
    # The parameterized 409 (honesty rule): the message must never claim
    # "no GPU session running" while one is registered — each variant names
    # what is true and the working remedy.
    from fastapi import HTTPException

    state = ServiceState()
    # 1) genuinely no session: the legacy session-flow text stands.
    with pytest.raises(HTTPException) as excinfo:
        state.acquire_model("org/x")
    assert "no GPU session running" in excinfo.value.detail
    # 2) registered but not yet dialable: name the wait, not absence.
    gpu_session.write_session_record({
        "sessionGeneration": "gen-h", "slurmJobID": "424242",
        "state": "starting", "node": None, "port": None,
    })
    with pytest.raises(HTTPException) as excinfo:
        state.acquire_model("org/x")
    assert "no GPU session running" not in excinfo.value.detail
    assert "state: starting" in excinfo.value.detail
    assert "not" in excinfo.value.detail and "reachable" in excinfo.value.detail
    # 3) live and dialable, but the verb was not delegated: say THAT.
    _install_session_records()
    with pytest.raises(HTTPException) as excinfo:
        state.acquire_model("org/x")
    assert "no GPU session running" not in excinfo.value.detail
    assert "a GPU session IS running" in excinfo.value.detail
    assert "POST /api/studies/submit" in excinfo.value.detail
