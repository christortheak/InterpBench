"""Observing another process's store must not recover or cancel its jobs."""

import json
import os
from pathlib import Path
import subprocess
import sys

from steerlab_server.api.jobs import DurableJobStore, Job


def test_cli_listing_preserves_live_jobs_and_pending_fanout(tmp_path):
    metadata = tmp_path / "metadata"
    store = DurableJobStore(str(metadata / "jobs.sqlite"))
    jobs = [
        Job(id="local", kind="experiment:run", status="running"),
        Job(id="child", kind="study-submit", status="submitted",
            executor="slurm", executor_job_id="123"),
        Job(id="parent", kind="study-submit", status="pending",
            executor="slurm", requested_resources={
                "parallelJobs": 2, "shardChildren": ["child"]}),
    ]
    for job in jobs:
        store.insert(job)
    before = {key: value.to_dict() for key, value in store.load_all().items()}
    cancellation = tmp_path / "unexpected-cancel"
    # A real second process opens the same SQLite store. Replace the scheduler
    # boundary there so a regression cannot reach any actual allocation.
    program = """
import sys
from pathlib import Path
from steerlab_server.api.executors import SlurmExecutor
def cancel(*args, **kwargs):
    Path(sys.argv[1]).write_text('scheduler cancellation attempted')
    return True
SlurmExecutor.cancel = cancel
from steerlab_server import cli
cli._jobs(['list'])
"""
    env = dict(os.environ, STEERLAB_ROOT=str(tmp_path / "workspace"),
               STEERLAB_METADATA_ROOT=str(metadata),
               PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run(
        [sys.executable, "-c", program, str(cancellation)], env=env,
        capture_output=True, text=True, timeout=30, check=True)
    listed = {job["id"]: job for job in json.loads(result.stdout)["jobs"]}
    after = {key: value.to_dict() for key, value in store.load_all().items()}
    assert not cancellation.exists(), "listing attempted to cancel a shard"
    assert after == before
    assert listed == before


def test_second_manager_preserves_live_owner(tmp_path):
    from steerlab_server.api.jobs import JobManager

    store = DurableJobStore(str(tmp_path / "jobs.sqlite"))
    owner = JobManager(store, sweep_orphans=False)
    local = owner.record_external("run", status="running", executor="local")
    parent = owner.record_external("study-submit", status="pending", executor="slurm",
                                   requested_resources={"parallelJobs": 2})
    observer = JobManager(DurableJobStore(store.path))
    assert observer.get(local.id).status == "running"
    assert observer.get(parent.id).status == "pending"


def exited_job(path):
    program = """\
import sys
from steerlab_server.api.jobs import DurableJobStore, JobManager
manager = JobManager(DurableJobStore(sys.argv[1]), sweep_orphans=False)
print(manager.record_external('run', status='running', executor='local').id)
"""
    result = subprocess.run([sys.executable, "-c", program, str(path)],
                            capture_output=True, text=True, check=True, timeout=30)
    return result.stdout.strip()


def test_recovery_requires_exited_process_and_is_claimed_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from steerlab_server.api.jobs import JobManager

    path = str(tmp_path / "jobs.sqlite")
    job_id = exited_job(path)
    stores = [DurableJobStore(path), DurableJobStore(path)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda store: store.claim_orphan(job_id), stores))
    assert sum(job is not None for job in claims) == 1
    # This process now owns recovery, so another constructor cannot steal it.
    assert JobManager(DurableJobStore(path)).get(job_id).status == "running"


def test_startup_recovers_exited_process(tmp_path):
    from steerlab_server.api.jobs import JobManager

    path = str(tmp_path / "jobs.sqlite")
    job_id = exited_job(path)
    recovered = JobManager(DurableJobStore(path)).get(job_id)
    assert recovered.status == "failed"
    assert "orphaned" in recovered.error


def test_unknown_and_foreign_owners_are_not_recovered(tmp_path):
    import sqlite3
    from steerlab_server.api.jobs import JobManager

    store = DurableJobStore(str(tmp_path / "jobs.sqlite"))
    manager = JobManager(store, sweep_orphans=False)
    legacy = manager.record_external("run", status="running", executor="local")
    foreign = manager.record_external("run", status="pending", executor="local")
    with sqlite3.connect(store.path) as conn:
        conn.execute("DELETE FROM job_owners WHERE job_id = ?", (legacy.id,))
        conn.execute("UPDATE job_owners SET host = ? WHERE job_id = ?",
                     ("unreachable-fictional-controller", foreign.id))
    restored = JobManager(DurableJobStore(store.path))
    assert restored.get(legacy.id).status == "running"
    assert restored.get(foreign.id).status == "pending"


def test_liveness_uncertainty_is_not_death(monkeypatch):
    from steerlab_server.api import job_ownership

    def denied(pid, signal):
        raise PermissionError("not permitted")

    monkeypatch.setattr(job_ownership.os, "kill", denied)
    assert not job_ownership.owner_has_exited(*job_ownership.current_owner())
