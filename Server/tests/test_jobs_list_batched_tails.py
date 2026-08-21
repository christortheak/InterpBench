"""GET /api/jobs must not pay one store round-trip per job for log tails.

The jobs list fetches a 50-line tail for every job. Doing that through
``Job.log_tail()`` opens a fresh sqlite connection per job — over an
NFS-resident database that took ~100 s at 344 jobs, so the Mac app's
Compute panel timed out on every refresh (observed 2026-08-09). The
batched ``DurableJobStore.log_tails()`` must return exactly what the
per-job reads would have, and the list route must use it.
"""

from steerlab_server.api.jobs import DurableJobStore, JobManager


def _store(tmp_path):
    return DurableJobStore(str(tmp_path / "jobs.sqlite"))


def test_log_tails_matches_per_job_logs(tmp_path):
    store = _store(tmp_path)
    jobs = JobManager(store)
    a = jobs.record_external("study-submit", status="submitted",
                             executor="slurm", executor_job_id="1")
    b = jobs.record_external("study-submit", status="submitted",
                             executor="slurm", executor_job_id="2")
    for i in range(60):
        a.log(f"a line {i}")
    b.log("b only line")

    tails = store.log_tails(limit=50)
    assert tails[a.id] == store.logs(a.id, limit=50)
    assert len(tails[a.id]) == 50
    assert tails[a.id][-1] == "a line 59"
    assert tails[b.id] == ["b only line"]


def test_log_tails_omits_logless_jobs(tmp_path):
    store = _store(tmp_path)
    jobs = JobManager(store)
    silent = jobs.record_external("study-submit", status="submitted",
                                  executor="slurm", executor_job_id="3")
    tails = store.log_tails(limit=50)
    assert silent.id not in tails


def test_to_dict_accepts_prefetched_tail(tmp_path):
    store = _store(tmp_path)
    jobs = JobManager(store)
    job = jobs.record_external("study-submit", status="submitted",
                               executor="slurm", executor_job_id="4",
                               log="born")
    # Prefetched tail is used verbatim; default path still reads the store.
    assert job.to_dict(log_tail=["injected"])["logTail"] == ["injected"]
    assert job.to_dict()["logTail"] == ["born"]


def test_job_logs_index_exists(tmp_path):
    store = _store(tmp_path)
    import sqlite3
    conn = sqlite3.connect(store.path)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'")}
    assert "idx_job_logs_job_seq" in names
