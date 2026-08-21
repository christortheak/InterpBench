"""Resume crash-window safety (external review 2026-07-23, finding 2).

The crash window: sbatch succeeded, the continuation record was never
written, the controller died. A stale claim must then NEVER be blindly
taken over — the reconciler first searches the scheduler for the claim's
durable submission token (the ``--job-name`` sbatch ran under):

- token FOUND on the scheduler → ADOPT the running continuation (write the
  missing record, repair ``resubmittedAs``), submit nothing;
- token positively ABSENT → the claimant died before sbatch: reclaim and
  submit exactly once;
- scheduler UNREACHABLE → durably mark the claim ``reconciliationRequired``
  with a plain-language operator message and refuse automatic resubmission
  (at-most-once beats liveness); it heals on a later tick once the
  scheduler answers.

Also: the manual loser-wait re-reads the DURABLE store, so a continuation
written by a sibling process over the same SQLite file is found.
"""

import time

import pytest

from steerlab_server.api.jobs import (DurableJobStore, JobManager,
                                      ResubmitRefused)


class SearchableFakeExecutor:
    """A Slurm executor double with the token-search surface: ``submit``
    accepts the ``job_name`` token and ``find_job_by_name`` answers from a
    programmable map (or raises, simulating an unreachable scheduler)."""

    def __init__(self):
        self.submits = []          # (script_path, job_name)
        self.known = {}            # token -> slurm id
        self.search_error = None   # exception to raise on search
        self.searches = []

    def submit(self, bundle, *, job_name=None):
        self.submits.append((bundle.script_path, job_name))
        return str(8000 + len(self.submits))

    def find_job_by_name(self, name):
        self.searches.append(name)
        if self.search_error is not None:
            raise self.search_error
        return self.known.get(name)

    def cancel(self, job_id):
        return True

    def poll_state(self, job_id):
        return None


@pytest.fixture(autouse=True)
def _isolated_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    monkeypatch.setenv("STEERLAB_AUTO_RESUBMIT", "1")
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT_LIMIT", raising=False)
    monkeypatch.delenv("STEERLAB_MAINTENANCE_CALENDAR", raising=False)


def _manager(tmp_path, executor, store=None):
    return JobManager(store or DurableJobStore(str(tmp_path / "jobs.sqlite")),
                      capability_provider=lambda: {},
                      slurm_executor=executor)


def _script(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(exist_ok=True)
    script = bundle_dir / "run.sbatch"
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    return str(script)


def _checkpointed_with_stale_claim(mgr, script, token="slre-dead-abc123"):
    job = mgr.record_external(
        "study-submit", status="checkpointed", executor="slurm",
        executor_job_id="9001",
        requested_resources={"scriptPath": script, "walltime": "00:10:00"})
    stale_at = time.time() - DurableJobStore.RESUBMIT_CLAIM_STALE_SECONDS - 60
    claim = {"claimant": "dead-server", "at": stale_at}
    if token is not None:
        claim["token"] = token
    job.result = {"resubmitClaim": claim}
    mgr.store.update(job)
    return job


def test_token_found_adopts_instead_of_resubmitting(tmp_path):
    executor = SearchableFakeExecutor()
    executor.known["slre-dead-abc123"] = "7777"
    mgr = _manager(tmp_path, executor)
    job = _checkpointed_with_stale_claim(mgr, _script(tmp_path))
    assert mgr.poll_slurm() >= 1
    job = mgr.get(job.id)
    child_id = (job.result or {}).get("resubmittedAs")
    assert child_id, "the found continuation must be adopted"
    child = mgr.get(child_id)
    assert child.executor_job_id == "7777"
    assert child.requested_resources["resubmitOf"] == job.id
    assert child.requested_resources.get("adoptedFromToken") is True
    assert executor.submits == [], "NOTHING may be resubmitted on adoption"
    assert any("adopted" in line for line in job.all_logs())


def test_token_absent_reclaims_and_submits_once(tmp_path):
    executor = SearchableFakeExecutor()   # knows no tokens: positive absence
    mgr = _manager(tmp_path, executor)
    job = _checkpointed_with_stale_claim(mgr, _script(tmp_path))
    assert mgr.poll_slurm() >= 1
    job = mgr.get(job.id)
    child_id = (job.result or {}).get("resubmittedAs")
    assert child_id
    assert len(executor.submits) == 1
    script_path, submitted_token = executor.submits[0]
    assert submitted_token and submitted_token.startswith("slre-")
    assert submitted_token != "slre-dead-abc123", \
        "the reclaim submits under a FRESH token"
    # Idempotent across further ticks.
    mgr.poll_slurm()
    assert len(executor.submits) == 1


def test_scheduler_unreachable_marks_reconciliation_required(tmp_path):
    executor = SearchableFakeExecutor()
    executor.search_error = RuntimeError("sacct: connection refused")
    mgr = _manager(tmp_path, executor)
    job = _checkpointed_with_stale_claim(mgr, _script(tmp_path))
    mgr.poll_slurm()
    job = mgr.get(job.id)
    assert not (job.result or {}).get("resubmittedAs")
    assert executor.submits == [], \
        "an unreachable scheduler must refuse automatic resubmission"
    claim = mgr.store.read_result(job.id).get("resubmitClaim") or {}
    assert claim.get("reconciliationRequired") is True
    message = claim.get("operatorMessage") or ""
    assert "reconciliation required" in message
    assert "slre-dead-abc123" in message
    assert any("reconciliation required" in line for line in job.all_logs())
    # Manual Resume surfaces the same situation as a refusal, not a resubmit.
    with pytest.raises(ResubmitRefused, match="reconciliation required"):
        mgr.resubmit(job.id)
    assert executor.submits == []
    # ...and it HEALS once the scheduler answers again (positive absence).
    executor.search_error = None
    assert mgr.poll_slurm() >= 1
    job = mgr.get(job.id)
    assert (job.result or {}).get("resubmittedAs")
    assert len(executor.submits) == 1


def test_tokenless_stale_claim_auto_blocks_manual_reclaims(tmp_path):
    """Legacy (pre-token) stale claims cannot be scheduler-verified: the
    auto path refuses (reconciliationRequired); a human's Resume click is
    explicit consent to reclaim, logged loudly."""
    executor = SearchableFakeExecutor()
    mgr = _manager(tmp_path, executor)
    job = _checkpointed_with_stale_claim(mgr, _script(tmp_path), token=None)
    mgr.poll_slurm()
    assert executor.submits == []
    claim = mgr.store.read_result(job.id).get("resubmitClaim") or {}
    assert claim.get("reconciliationRequired") is True
    outcome = mgr.resubmit(job.id)
    assert outcome["ok"] is True
    assert len(executor.submits) == 1
    job = mgr.get(job.id)
    assert any("explicit consent" in line for line in job.all_logs())


def test_loser_wait_reads_durable_store_across_processes(tmp_path):
    """The manual loser-wait must see a continuation written by ANOTHER
    process over the same durable store — polling the in-memory catalog
    missed it (finding 2c)."""
    import threading

    db = str(tmp_path / "jobs.sqlite")
    executor_a = SearchableFakeExecutor()
    mgr_a = _manager(tmp_path, executor_a, store=DurableJobStore(db))
    script = _script(tmp_path)
    job = mgr_a.record_external(
        "study-submit", status="checkpointed", executor="slurm",
        executor_job_id="9001",
        requested_resources={"scriptPath": script, "walltime": "00:10:00"})
    # A sibling "process" over the same SQLite file, loaded before the race.
    executor_b = SearchableFakeExecutor()
    mgr_b = _manager(tmp_path, executor_b, store=DurableJobStore(db))
    # A foreign process holds a LIVE claim...
    store = mgr_a.store
    claimed, _ = store.claim_resubmit(job.id, "foreign-process",
                                      token="slre-foreign-1")
    assert claimed

    child_ids: list[str] = []

    def winner():
        time.sleep(0.3)
        # ...then finishes: writes the continuation record and the durable
        # resubmittedAs stamp, all through ITS OWN store handle.
        child = mgr_a.record_external(
            "study-submit", status="submitted", executor="slurm",
            executor_job_id="7042",
            requested_resources={"resubmitOf": job.id})
        child_ids.append(child.id)
        result = store.read_result(job.id)
        result.pop("resubmitClaim", None)
        result["resubmittedAs"] = child.id
        job_a = mgr_a.get(job.id)
        job_a.result = result
        store.update(job_a)

    thread = threading.Thread(target=winner)
    thread.start()
    outcome = mgr_b.resubmit(job.id)
    thread.join()
    assert outcome.get("alreadyResumed") is True
    assert outcome.get("jobId") == child_ids[0]
    assert executor_b.submits == [], \
        "the loser must adopt the durable continuation, never double-submit"

    # A sibling manager whose memory predates a FINISHED resubmission also
    # answers honestly through the durable scan (repair + refusal naming
    # the continuation), with no second sbatch.
    executor_c = SearchableFakeExecutor()
    mgr_c = _manager(tmp_path, executor_c, store=DurableJobStore(db))
    with pytest.raises(ResubmitRefused, match="already resubmitted"):
        mgr_c.resubmit(job.id)
    assert executor_c.submits == []
