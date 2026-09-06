"""Controller recovery uses proof or explicit reviewed attestation, never age."""
import json
import sqlite3
from subprocess import CompletedProcess

import pytest

from steerlab_server.api import job_ownership as ownership
from steerlab_server.api.jobs import DurableJobStore, Job, JobManager


ALLOCATION = dict(dbIndex="17", jobID="42", cluster="test", submitted="2026-01-01T01:00:00",
                  started="2026-01-01T01:01:00")


@pytest.fixture
def foreign(tmp_path):
    store = DurableJobStore(str(tmp_path / "jobs.sqlite"))
    store.insert(Job(id="orphan", kind="run", status="running", executor="local"))
    with store._connect() as conn:
        conn.execute("UPDATE job_owners SET host = ?, allocation_json = ? WHERE job_id = ?",
                     ("other-controller", json.dumps(ALLOCATION), "orphan"))
    return store


def accounting(monkeypatch, state="COMPLETED", **changes):
    row = dict(ALLOCATION, state=state, **changes)
    monkeypatch.setattr(ownership, "allocation_rows", lambda *args: [row])


@pytest.mark.parametrize("state,expected", [
    ("COMPLETED", "failed"), ("FAILED", "failed"), ("CANCELLED", "failed"),
    ("RUNNING", "running"), ("SUSPENDED", "running"), ("COMPLETING", "running"),
    ("PREEMPTED", "running"), ("REQUEUED", "running"), ("UNKNOWN", "running"),
])
def test_startup_uses_exact_controller_allocation(foreign, monkeypatch, state, expected):
    accounting(monkeypatch, state)
    assert JobManager(foreign).get("orphan").status == expected


@pytest.mark.parametrize("field,value", [
    ("dbIndex", "18"), ("jobID", "43"), ("cluster", "different"),
    ("submitted", "2026-01-02T01:00:00"), ("started", "2026-01-02T01:01:00"),
])
def test_reused_id_or_requeued_allocation_is_not_proof(foreign, monkeypatch, field, value):
    accounting(monkeypatch, **{field: value})
    assert foreign.claim_orphan("orphan") is None


def test_accounting_absence_and_ambiguity_are_not_proof(foreign, monkeypatch):
    for rows in ([], [dict(ALLOCATION, state="COMPLETED")] * 2):
        monkeypatch.setattr(ownership, "allocation_rows", lambda *args: rows)
        assert foreign.claim_orphan("orphan") is None


def test_scheduler_query_is_outside_write_lock_and_changed_owner_invalidates_proof(foreign, monkeypatch):
    def query(*args):
        # A separate writer can commit while accounting is being queried.
        # Even a same-host/same-PID replacement must invalidate this review.
        with sqlite3.connect(foreign.path, timeout=0.05) as conn:
            conn.execute("UPDATE job_owners SET instance = 'replacement' WHERE job_id = 'orphan'")
        return [dict(ALLOCATION, state="COMPLETED")]
    monkeypatch.setattr(ownership, "allocation_rows", query)
    assert foreign.claim_orphan("orphan") is None
    assert foreign.load_all()["orphan"].status == "running"


def test_changed_job_invalidates_review(foreign, monkeypatch):
    accounting(monkeypatch)
    token = foreign.recovery_report("orphan")["reviewToken"]
    job = foreign.load_all()["orphan"]
    job.status = "cancelling"
    foreign.update(job)
    with pytest.raises(ValueError, match="stale"):
        foreign.claim_orphan("orphan", review_token=token, reason="Controller exit independently verified")


def test_live_controller_cannot_be_overridden(foreign, monkeypatch):
    accounting(monkeypatch, "RUNNING")
    report = foreign.recovery_report("orphan")
    with pytest.raises(ValueError, match="still live"):
        foreign.claim_orphan("orphan", review_token=report["reviewToken"], reason="requested")
    assert foreign.load_all()["orphan"].status == "running"


def test_legacy_jobs_report_then_require_explicit_review_and_record_audit(foreign, capsys):
    with foreign._connect() as conn:
        conn.execute("DELETE FROM job_owners")
    manager = JobManager(foreign)
    assert "jobRecoveryRequired: 1" in capsys.readouterr().err
    report = foreign.recovery_report("orphan")
    assert report["ownerState"] == "unknown"
    assert report["owner"] is None
    assert manager.get("orphan").status == "running"
    assert manager.recover_orphan("orphan", report["reviewToken"], "Controller shutdown verified by operator")
    assert manager.get("orphan").status == "failed"
    with foreign._connect() as conn:
        row = conn.execute("SELECT * FROM job_recoveries").fetchone()
        assert row["previous_owner_json"] == "null"
        assert row["evidence"] == "operator-attested-exit"
        assert row["reason"] == "Controller shutdown verified by operator"
        assert row["review_token"] == report["reviewToken"]
    assert not manager.recover_orphan("orphan", report["reviewToken"], "repeat")


def test_legacy_schema_is_migrated_without_changing_owners(tmp_path):
    path = str(tmp_path / "jobs.sqlite")
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE job_owners (job_id TEXT PRIMARY KEY, host TEXT NOT NULL, pid INTEGER NOT NULL)")
        conn.execute("INSERT INTO job_owners VALUES ('legacy', 'old-host', 123)")
    store = DurableJobStore(path)
    with store._connect() as conn:
        row = ownership.read(conn, "legacy")
    assert row == dict(job_id="legacy", host="old-host", pid=123, allocation_json=None, instance=None)


@pytest.mark.parametrize("failure", ["timeout", "permission", "missing", "empty", "step", "malformed"])
def test_accounting_failures_cannot_prove_exit(monkeypatch, failure):
    from steerlab_server.api import executors
    def run(command, **kwargs):
        if failure == "missing":
            raise FileNotFoundError()
        if failure == "permission":
            raise PermissionError()
        output = {"step": "17|42.batch|test|2026-01-01T01:00:00|2026-01-01T01:01:00|COMPLETED",
                  "malformed": "17|42|test|Unknown|Unknown|COMPLETED"}.get(failure, "")
        return CompletedProcess(command, 124 if failure == "timeout" else 0, output, "")
    monkeypatch.setattr(executors, "scheduler_run", run)
    assert ownership.allocation_rows("42", "test") == []


def test_capture_and_query_use_controller_identity_and_configured_accounting(monkeypatch):
    from steerlab_server.api import executors
    monkeypatch.setenv("STEERLAB_SLURM_SACCT", "/test/accounting-wrapper")
    observed = []
    def run(command, **kwargs):
        observed.append((command, kwargs))
        return CompletedProcess(command, 0,
            "17|42|test|2026-01-01T01:00:00|2026-01-01T01:01:00|RUNNING\n", "")
    monkeypatch.setattr(executors, "scheduler_run", run)
    ownership.capture_allocation.cache_clear()
    assert ownership.capture_allocation(100, "42", "test") == ALLOCATION
    command, kwargs = observed[0]
    assert command[0] == "/test/accounting-wrapper"
    assert "--duplicates" in command and "-X" in command
    assert command[command.index("-j") + 1] == "42"
    assert kwargs["env"]["TZ"] == "UTC"
    ownership.capture_allocation.cache_clear()


def test_cli_review_is_read_only_and_recovery_is_explicit(foreign, monkeypatch, capsys):
    from steerlab_server import cli
    monkeypatch.setenv("STEERLAB_JOBS_DB", foreign.path)
    with foreign._connect() as conn:
        conn.execute("DELETE FROM job_owners")
    assert cli.main(["jobs", "recovery", "orphan", "--json"]) == 0
    review = json.loads(capsys.readouterr().out)
    assert not review["changed"]
    token = review["result"]["reviewToken"]
    assert foreign.load_all()["orphan"].status == "running"
    assert cli.main(["jobs", "recover", "orphan", "--json"]) == 65
    capsys.readouterr()
    assert foreign.load_all()["orphan"].status == "running"
    assert cli.main(["jobs", "recover", "orphan", "--json", "--review-token", token,
                     "--confirm-owner-exited", "--reason", "Controller shutdown verified"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["changed"] and result["result"]["recovered"]
    assert foreign.load_all()["orphan"].status == "failed"


def test_job_insert_records_controller_allocation_not_compute_allocation(tmp_path, monkeypatch):
    from steerlab_server.api import executors
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "test")
    store = DurableJobStore(str(tmp_path / "jobs.sqlite"))
    def run(command, **kwargs):
        assert command[command.index("-j") + 1] == "42"
        # Insertion has not acquired its write transaction yet.
        with sqlite3.connect(store.path, timeout=0.05) as conn:
            conn.execute("INSERT INTO job_logs (job_id, timestamp, message) VALUES ('probe', 0, 'probe')")
        return CompletedProcess(command, 0,
            "17|42|test|2026-01-01T01:00:00|2026-01-01T01:01:00|RUNNING\n", "")
    monkeypatch.setattr(executors, "scheduler_run", run)
    ownership.capture_allocation.cache_clear()
    try:
        store.insert(Job(id="child", kind="run", status="submitted",
                         executor="slurm", executor_job_id="999"))
        with store._connect() as conn:
            owner = ownership.read(conn, "child")
        assert json.loads(owner["allocation_json"]) == ALLOCATION
        assert store.load_all()["child"].executor_job_id == "999"
    finally:
        ownership.capture_allocation.cache_clear()


def test_startup_queries_a_shared_controller_allocation_only_once(foreign, monkeypatch):
    foreign.insert(Job(id="second", kind="run", status="running", executor="local"))
    with foreign._connect() as conn:
        conn.execute("UPDATE job_owners SET host = ?, allocation_json = ? WHERE job_id = 'second'",
                     ("other-controller", json.dumps(ALLOCATION)))
    queries = []
    def query(*args):
        queries.append(args)
        return [dict(ALLOCATION, state="COMPLETED")]
    monkeypatch.setattr(ownership, "allocation_rows", query)
    manager = JobManager(foreign)
    assert manager.get("orphan").status == manager.get("second").status == "failed"
    assert len(queries) == 1
