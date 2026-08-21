"""Resubmission is single-flight (external review 2026-07-22, finding 1).

The manual Resume verb and the auto-resubmit reconcile tick both ran
check-for-existing → sbatch → stamp with no shared claim, so a double click
(or a manual Resume overlapping the auto tick) could launch TWO Slurm
processes appending to the same run directory. The fix is a durable
compare-and-set claim on the job record (``DurableJobStore.claim_resubmit``)
wrapped around the entire resubmission; the loser of a race gets the honest
"already resumed as <id>" outcome, never an error and never a second sbatch.
"""

import threading
import time

import pytest

from steerlab_server.api.jobs import DurableJobStore, JobManager


class SlowFakeExecutor:
    """A Slurm executor double whose submit is SLOW — wide enough a race
    window that two concurrent resubmit attempts genuinely overlap."""

    def __init__(self, delay=0.5, fail_first=0):
        self.delay = delay
        self.fail_first = fail_first
        self.submits = []
        self.attempts = 0
        self._lock = threading.Lock()

    def submit(self, bundle):
        time.sleep(self.delay)
        with self._lock:
            self.attempts += 1
            if self.attempts <= self.fail_first:
                raise RuntimeError("fake sbatch refused")
            self.submits.append(bundle.script_path)
            return str(7000 + len(self.submits))

    def cancel(self, job_id):
        return True

    def poll_state(self, job_id):
        return None


@pytest.fixture(autouse=True)
def _isolated_metadata(tmp_path, monkeypatch):
    """Keep the maintenance-calendar default lookup (and any store default)
    inside the test tree, off the developer's real metadata root."""
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT", raising=False)
    monkeypatch.delenv("STEERLAB_AUTO_RESUBMIT_LIMIT", raising=False)
    monkeypatch.delenv("STEERLAB_MAINTENANCE_CALENDAR", raising=False)


def _manager(tmp_path, executor):
    return JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")),
                      capability_provider=lambda: {},
                      slurm_executor=executor)


def _script(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(exist_ok=True)
    script = bundle_dir / "run.sbatch"
    script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    return str(script)


def _checkpointed(mgr, script, slurm_id="9001", extra_rr=None):
    rr = {"scriptPath": script, "walltime": "00:10:00"}
    rr.update(extra_rr or {})
    return mgr.record_external("study-submit", status="checkpointed",
                               executor="slurm", executor_job_id=slurm_id,
                               requested_resources=rr)


# --- the races ------------------------------------------------------------------

def test_two_concurrent_manual_resumes_submit_exactly_once(tmp_path):
    fake = SlowFakeExecutor(delay=0.5)
    mgr = _manager(tmp_path, fake)
    job = _checkpointed(mgr, _script(tmp_path))

    barrier = threading.Barrier(2)
    results, errors = [], []

    def resume():
        barrier.wait()
        try:
            results.append(mgr.resubmit(job.id))
        except Exception as exc:  # noqa: BLE001 - recorded for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=resume) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert len(fake.submits) == 1          # exactly ONE sbatch
    assert len(results) == 2
    ids = {r["jobId"] for r in results}
    assert len(ids) == 1                   # both callers name the SAME child
    child_id = ids.pop()
    assert all(r["ok"] is True for r in results)
    # One of the two is the honest race loser, not an error.
    assert sum(1 for r in results if r.get("alreadyResumed")) == 1
    # The durable stamp is coherent and the claim was consumed by it.
    stamped = mgr.get(job.id).result
    assert stamped["resubmittedAs"] == child_id
    assert "resubmitClaim" not in stamped


def test_manual_resume_overlapping_auto_tick_submits_once(tmp_path):
    fake = SlowFakeExecutor(delay=0.5)
    mgr = _manager(tmp_path, fake)
    job = _checkpointed(mgr, _script(tmp_path),
                        extra_rr={"auto_resubmit": True})

    barrier = threading.Barrier(2)
    manual_results, errors = [], []

    def manual():
        barrier.wait()
        try:
            manual_results.append(mgr.resubmit(job.id))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def auto():
        barrier.wait()
        try:
            mgr._maybe_auto_resubmit(job)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=manual), threading.Thread(target=auto)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert len(fake.submits) == 1          # never a double submission
    stamped = (mgr.get(job.id).result or {}).get("resubmittedAs")
    assert stamped
    assert len(manual_results) == 1
    assert manual_results[0]["ok"] is True
    assert manual_results[0]["jobId"] == stamped


def test_failed_sbatch_releases_the_claim_so_a_later_tick_retries(tmp_path):
    fake = SlowFakeExecutor(delay=0.0, fail_first=1)
    mgr = _manager(tmp_path, fake)
    job = _checkpointed(mgr, _script(tmp_path),
                        extra_rr={"auto_resubmit": True})

    # First tick: sbatch fails — no submission, and the claim is RELEASED so
    # the retry is not blocked by our own dead claim.
    assert mgr._maybe_auto_resubmit(job) is False
    assert fake.submits == []
    assert "resubmitClaim" not in (mgr.get(job.id).result or {})

    # Next tick: succeeds.
    assert mgr._maybe_auto_resubmit(job) is True
    assert len(fake.submits) == 1
    assert (mgr.get(job.id).result or {}).get("resubmittedAs")


# --- the claim primitive --------------------------------------------------------

def test_claim_is_atomic_and_claimant_scoped(tmp_path):
    mgr = _manager(tmp_path, SlowFakeExecutor(delay=0.0))
    job = _checkpointed(mgr, _script(tmp_path))
    store = mgr.store

    claimed, result = store.claim_resubmit(job.id, "actor-a")
    assert claimed is True
    assert result["resubmitClaim"]["claimant"] == "actor-a"

    # A live claim blocks every other claimant.
    assert store.claim_resubmit(job.id, "actor-b")[0] is False

    # Release is claimant-scoped: the wrong claimant releases nothing.
    store.release_resubmit_claim(job.id, "actor-b")
    assert store.claim_resubmit(job.id, "actor-b")[0] is False
    store.release_resubmit_claim(job.id, "actor-a")
    assert store.claim_resubmit(job.id, "actor-b")[0] is True


def test_claim_refuses_an_already_resubmitted_job_with_the_stamp(tmp_path):
    mgr = _manager(tmp_path, SlowFakeExecutor(delay=0.0))
    job = _checkpointed(mgr, _script(tmp_path))
    job.result = {"resubmittedAs": "child99"}
    mgr.store.update(job)
    claimed, result = mgr.store.claim_resubmit(job.id, "actor-a")
    assert claimed is False
    assert result["resubmittedAs"] == "child99"  # the honest answer rides along


def test_stale_claim_is_never_blindly_claimable(tmp_path):
    """Resume crash-safety (2026-07-23): a stale claim is NOT taken over by
    ``claim_resubmit`` — the dead claimant may have crashed AFTER sbatch, so
    blind takeover could double-submit. Only ``takeover_stale_claim`` (used
    after the manager's scheduler search proved the token absent, or with a
    human's explicit Resume consent) replaces it."""
    mgr = _manager(tmp_path, SlowFakeExecutor(delay=0.0))
    job = _checkpointed(mgr, _script(tmp_path))
    stale_at = time.time() - DurableJobStore.RESUBMIT_CLAIM_STALE_SECONDS - 60
    job.result = {"resubmitClaim": {"claimant": "dead-server", "at": stale_at,
                                    "token": "slre-dead-t0"}}
    mgr.store.update(job)
    claimed, result = mgr.store.claim_resubmit(job.id, "actor-b")
    assert claimed is False
    assert result["resubmitClaim"]["claimant"] == "dead-server"
    assert DurableJobStore.claim_is_stale(result)
    taken, after = mgr.store.takeover_stale_claim(job.id, "actor-b",
                                                  token="slre-b-t1")
    assert taken is True
    assert after["resubmitClaim"]["claimant"] == "actor-b"
    assert after["resubmitClaim"]["token"] == "slre-b-t1"
    # A LIVE claim can never be taken over.
    taken_again, _ = mgr.store.takeover_stale_claim(job.id, "actor-c")
    assert taken_again is False


def test_unknown_job_cannot_be_claimed(tmp_path):
    mgr = _manager(tmp_path, SlowFakeExecutor(delay=0.0))
    assert mgr.store.claim_resubmit("nope", "actor-a") == (False, {})


def test_loser_sync_never_resurrects_a_consumed_claim(tmp_path):
    """The interleaving CI caught (2026-08-20, slow runner): the loser's
    durable read lands BEFORE the winner's stamp write, while the shared
    in-memory job already carries the stamp the winner assigned. `durable`
    still holds the claim, so a plain merge would produce
    {resubmittedAs, resubmitClaim} — a poisoned record that any later
    store.update() persists. Frozen deterministically: no threads, the
    exact mid-flight state constructed by hand."""
    fake = SlowFakeExecutor(delay=0.0)
    mgr = _manager(tmp_path, fake)
    job = _checkpointed(mgr, _script(tmp_path))

    # The winner's claim is durable; its stamp write has NOT happened yet.
    claimed, result_now = mgr.store.claim_resubmit(
        job.id, "manual:test:winner", token="slre-test-token")
    assert claimed
    assert "resubmitClaim" in mgr.store.read_result(job.id)

    # The winner's in-memory strip+stamp assignment HAS happened (shared
    # Job instance, same process).
    job.result = {"resubmittedAs": "child-mid-flight"}

    # The loser's wait loop runs against exactly that state.
    outcome = mgr._await_concurrent_resubmission(job, result_now={})

    assert outcome["ok"] is True
    assert outcome["jobId"] == "child-mid-flight"
    assert outcome["alreadyResumed"] is True
    # The sync must not resurrect the claim beside the stamp: once a
    # resubmittedAs stamp exists anywhere, the claim is consumed.
    assert job.result["resubmittedAs"] == "child-mid-flight"
    assert "resubmitClaim" not in job.result
