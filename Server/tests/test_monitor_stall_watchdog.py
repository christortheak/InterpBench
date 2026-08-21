"""Open-issues §2: the in-flight shard-merge pass stalling in a long-running
daemon, and the reconcile endpoint that could not substitute for a restart.

The framing that matters, established by reading the call graph: **there is no
separate boot-time merge path.** ``sharding.merge_shard_runs`` has exactly one
caller chain — ``JobManager._reconcile_shard_parents`` ← ``poll_slurm`` ← the
``slurm-monitor`` thread — so "a fresh daemon boot merges everything in
seconds" is that same code on a new thread in a new process. A merge that never
happens therefore means the MONITOR THREAD stopped ticking, not that the
periodic pass evaluated some condition the boot pass skips.

Two mechanisms could stop it, and both are covered here:

1. **Root cause (fixed).** Every scheduler call in ``executors.py`` ran as an
   unbounded ``subprocess.run`` on that one thread. A wedged ``sacct`` /
   ``squeue`` / ``scancel`` blocks ``poll_slurm`` forever, silently, and
   ``_reconcile_shard_parents`` runs at the END of ``poll_slurm`` — so a hang
   while polling an unrelated sibling job is enough to strand every completed
   shard set in the daemon. ``scheduler_run`` now bounds the wait.
2. **Residual (watchdog).** Any OTHER way the single thread can wedge is
   covered by a watchdog that notices the missing ticks and runs the merge pass
   itself, plus ``_monitor_stop.clear()`` so a stopped monitor can restart at
   all.
"""

import json
import os
import subprocess
import sys
import threading
import time

import pytest

from steerlab_server.api import executors
from test_sharding import (  # noqa: F401 - fake_slurm is a fixture
    _manager, _read, _run_shards, _run_single, _study_fixture)


def _wait(predicate, timeout=10.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class _HangingSlurm:
    """A scheduler whose ``poll_state`` never returns — the wedged-subprocess
    stall, made deterministic. ``release()`` lets the thread finish so the test
    does not leak a blocked daemon thread."""

    def __init__(self):
        self.gate = threading.Event()
        self.entered = threading.Event()

    def poll_state(self, job_id):
        self.entered.set()
        self.gate.wait(30)
        return None

    def poll_state_detailed(self, job_id):
        return self.poll_state(job_id), False

    def death_detail(self, job_id):
        return None

    def cancel(self, job_id):
        return False

    def release(self):
        self.gate.set()


def _succeeded_shard_parent(tmp_path, monkeypatch, jobs, name="stallrec"):
    """A parent whose every shard has succeeded and whose merge is therefore
    due — the exact state the two field incidents were stuck in."""
    root = str(tmp_path)
    prompts = _study_fixture(root, name)
    single_dir = _run_single(root, name, prompts, monkeypatch)
    shard_dirs = _run_shards(root, name, prompts, monkeypatch, count=2)
    children = []
    for index, directory in enumerate(shard_dirs):
        children.append(jobs.record_external(
            "study-submit-bundle-shard", status="succeeded", executor="slurm",
            executor_job_id=str(100 + index),
            result={"runDirectory": directory,
                    "shard": {"index": index, "count": 2}}))
    parent = jobs.record_external(
        "study-submit-bundle", status="running", executor="slurm",
        requested_resources={
            "shardChildren": [c.id for c in children],
            "recordsDirectory": str(tmp_path / "records"),
            "shardMerge": {"experiment": name, "verb": "run",
                           "targetRoot": root, "packageEvidence": False}},
        result={"shardJobs": [c.id for c in children]})
    return parent, single_dir


# --- 1. the root cause: unbounded scheduler subprocesses ----------------------


def test_scheduler_run_reports_a_timeout_as_a_failed_query(monkeypatch):
    monkeypatch.setenv("STEERLAB_SCHEDULER_TIMEOUT", "0.3")
    started = time.time()
    proc = executors.scheduler_run(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        text=True, capture_output=True, check=False)
    assert time.time() - started < 10, "the call must not have waited it out"
    assert proc.returncode == executors.SCHEDULER_TIMEOUT_EXIT
    assert "did not answer within" in proc.stderr
    assert "never as an answer" in proc.stderr


def test_a_wedged_sacct_no_longer_blocks_forever_and_says_nothing(monkeypatch,
                                                                  tmp_path):
    """The honesty contract survives the timeout: a timed-out query is
    ``(None, False)`` — "the scheduler did not answer" — never positive
    absence, which callers are allowed to act on."""
    sleeper = tmp_path / "sleepy"
    sleeper.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    sleeper.chmod(0o755)
    monkeypatch.setenv("STEERLAB_SLURM_SACCT", str(sleeper))
    monkeypatch.setenv("STEERLAB_SLURM_SQUEUE", str(sleeper))
    monkeypatch.setenv("STEERLAB_SCHEDULER_TIMEOUT", "0.3")
    started = time.time()
    state, query_ok = executors.SlurmExecutor().poll_state_detailed("4242")
    assert time.time() - started < 10
    assert state is None
    assert query_ok is False


def test_the_timeout_is_configurable_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("STEERLAB_SCHEDULER_TIMEOUT", raising=False)
    assert executors.scheduler_timeout() == executors.DEFAULT_SCHEDULER_TIMEOUT
    monkeypatch.setenv("STEERLAB_SCHEDULER_TIMEOUT", "45")
    assert executors.scheduler_timeout() == 45.0
    # The historical unbounded behaviour stays reachable, explicitly.
    monkeypatch.setenv("STEERLAB_SCHEDULER_TIMEOUT", "0")
    assert executors.scheduler_timeout() is None


def test_every_scheduler_call_goes_through_the_bounded_runner():
    """A new raw ``subprocess.run`` on a scheduler binary would reintroduce the
    stall silently, so the module is pinned: the only raw call left is the one
    inside ``scheduler_run`` itself."""
    import re
    source = open(executors.__file__, encoding="utf-8").read()
    invocations = re.findall(r"^\s*(?:return |\w+ = )subprocess\.run\(",
                             source, re.MULTILINE)
    assert len(invocations) == 1, (
        "scheduler commands must go through scheduler_run() — see §2; found "
        f"{len(invocations)} raw subprocess.run call sites")


# --- 2. the stall, and the watchdog that survives it -------------------------


def test_a_wedged_poll_stalls_the_monitor_and_the_watchdog_merges(
        tmp_path, monkeypatch):
    """The §2 symptom, reproduced against fakes: all shards COMPLETED, the
    monitor thread wedged inside a scheduler call, and — before this fix — the
    merged run directory never materialising.

    Note WHERE the wedge is: on an unrelated sibling job. That is enough,
    because ``_reconcile_shard_parents`` runs at the end of ``poll_slurm``."""
    jobs = _manager(tmp_path)
    hanging = _HangingSlurm()
    jobs._slurm_executor = hanging
    parent, single_dir = _succeeded_shard_parent(tmp_path, monkeypatch, jobs)
    # An unrelated, non-terminal Slurm job: this is what the poll wedges on.
    jobs.record_external("unrelated", status="running", executor="slurm",
                         executor_job_id="999")
    try:
        jobs.start_monitor(interval=0.01, housekeeping_interval=0,
                           watchdog_interval=0.05, stall_after=0.3)
        assert hanging.entered.wait(10), "the poll never reached the wedge"
        # The merge is due but the monitor can no longer reach it.
        assert _wait(lambda: jobs.get(parent.id).status == "succeeded", 15.0), (
            "the watchdog did not merge a completed shard set")
        merged = (jobs.get(parent.id).result or {})["runDirectory"]
        assert (_read(os.path.join(merged, "generations.jsonl"))
                == _read(os.path.join(single_dir, "generations.jsonl")))
        health = jobs.monitor_health()
        assert health["stalls"] >= 1
        assert health["stalledSeconds"] >= 0.3
        assert "WATCHDOG" in (health["lastError"] or "")
    finally:
        jobs.stop_monitor()
        hanging.release()


def test_the_watchdog_stays_quiet_while_the_monitor_ticks(tmp_path):
    """A healthy daemon must never see a watchdog line — the stall count is
    the signal an operator reads."""
    jobs = _manager(tmp_path)
    jobs._slurm_executor = _HangingSlurm()  # never consulted: no slurm jobs
    try:
        jobs.start_monitor(interval=0.01, housekeeping_interval=0,
                           watchdog_interval=0.02, stall_after=1.0)
        assert _wait(lambda: jobs.monitor_health()["ticks"] >= 5)
        time.sleep(0.3)
        assert jobs.monitor_health()["stalls"] == 0
    finally:
        jobs.stop_monitor()


def test_a_stopped_monitor_can_be_started_again(tmp_path):
    """``stop_monitor`` set the event and ``start_monitor`` never cleared it, so
    the restarted thread exited on its first wait — a live daemon with a dead
    reconciler and no log line."""
    jobs = _manager(tmp_path)
    jobs.stop_monitor()
    try:
        jobs.start_monitor(interval=0.01, housekeeping_interval=0,
                           watchdog_interval=0)
        assert _wait(lambda: jobs.monitor_health()["ticks"] >= 2), (
            "the restarted monitor never ticked")
        assert jobs.monitor_health()["running"] is True
    finally:
        jobs.stop_monitor()


def test_a_failing_merge_pass_is_reported_not_swallowed(tmp_path, capsys):
    """The bare ``except Exception: pass`` around the merge pass made a broken
    reconciler and an idle one look identical from outside the process."""
    jobs = _manager(tmp_path)

    def boom():
        raise RuntimeError("merge exploded")

    jobs._reconcile_shard_parents = boom
    jobs.poll_slurm()
    assert "merge exploded" in (jobs.monitor_health()["lastError"] or "")
    assert "shard-merge pass failed" in capsys.readouterr().err


def test_the_merge_pass_is_serialised(tmp_path):
    """Watchdog and monitor must never merge the same parent concurrently; a
    non-blocking caller that loses the race reports 0 rather than queueing
    behind a wedged holder."""
    jobs = _manager(tmp_path)
    jobs._merge_lock.acquire()
    try:
        assert jobs.run_merge_pass(blocking=False) == 0
    finally:
        jobs._merge_lock.release()
    assert jobs.run_merge_pass() == 0  # nothing to merge, but it ran


# --- 3. reconcile-everything, at the manager level ---------------------------


def test_reconcile_all_visits_every_known_records_directory(tmp_path):
    jobs = _manager(tmp_path)
    first = tmp_path / "records-a"
    second = tmp_path / "records-b"
    for directory in (first, second):
        directory.mkdir()
    child = jobs.record_external("study", status="running", executor="slurm",
                                 executor_job_id="7001")
    (first / f"{child.id}.json").write_text(json.dumps(
        {"id": child.id, "status": "succeeded",
         "result": {"runDirectory": str(tmp_path / "run-a")}}), encoding="utf-8")
    jobs.record_external(
        "study", status="running", executor="slurm", executor_job_id="7002",
        requested_resources={"recordsDirectory": str(first)})
    other = jobs.record_external(
        "study", status="running", executor="slurm", executor_job_id="7003",
        result={"recordsDirectory": str(second)})
    (second / f"{other.id}.json").write_text(json.dumps(
        {"id": other.id, "status": "failed"}), encoding="utf-8")

    known = jobs.known_records_directories()
    assert str(first) in known and str(second) in known

    reconciled, visited = jobs.reconcile_all()
    assert reconciled == 2
    assert sorted(visited) == sorted([str(first), str(second)])
    assert jobs.get(child.id).status == "succeeded"
    assert jobs.get(other.id).status == "failed"


def test_reconcile_all_skips_directories_that_are_gone(tmp_path):
    jobs = _manager(tmp_path)
    jobs.record_external(
        "study", status="running", executor="slurm", executor_job_id="7004",
        requested_resources={"recordsDirectory": str(tmp_path / "vanished")})
    reconciled, visited = jobs.reconcile_all()
    assert (reconciled, visited) == (0, [])


# --- 4. the endpoint's ergonomics --------------------------------------------


@pytest.fixture
def api_client(monkeypatch):
    from fastapi.testclient import TestClient
    from steerlab_server.api.app import app
    monkeypatch.setenv("STEERLAB_EXECUTOR", "local")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    monkeypatch.delenv("STEERLAB_AUTH_TOKEN", raising=False)
    client = TestClient(app)
    client.get("/api/health")  # warm the lazy route build (see MEMORY note)
    return client


def test_a_bare_post_now_reconciles_everything_and_merges(api_client):
    """§2's ask verbatim: bare POST used to be 422 and ``{}`` used to be 400,
    so the one endpoint an operator reached for during a stall could not be
    called without already knowing which submission was stuck."""
    for payload in (None, {}):
        resp = (api_client.post("/api/jobs/reconcile") if payload is None
                else api_client.post("/api/jobs/reconcile", json=payload))
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["scope"] == "all"
        assert isinstance(body["reconciled"], int)
        assert isinstance(body["recordsDirectories"], list)
        # The merge pass ran, and the response says so — the missing half.
        assert isinstance(body["merged"], int)
        assert "stalledSeconds" in body["monitor"]


def test_malformed_non_empty_bodies_still_refuse(api_client):
    """An operator who typed a key wrong must not silently get the
    reconcile-everything behaviour."""
    resp = api_client.post("/api/jobs/reconcile", json={"recordsDir": "x"})
    assert resp.status_code == 400
    assert "unknown field" in resp.json()["detail"]

    resp = api_client.post("/api/jobs/reconcile", json={"recordsDirectory": ""})
    assert resp.status_code == 400
    assert "non-empty string" in resp.json()["detail"]

    resp = api_client.post("/api/jobs/reconcile", json={"recordsDirectory": 7})
    assert resp.status_code == 400


def test_an_unknown_records_directory_still_404s(api_client):
    resp = api_client.post("/api/jobs/reconcile",
                           json={"recordsDirectory": "no-such-records-dir"})
    assert resp.status_code == 404
