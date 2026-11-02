"""A job's status and log tail, from the command line (issue #7).

``GET /api/jobs/{id}/status`` and ``GET /api/jobs/{id}/log-tail`` against the
committed fake scheduler binaries (``tests/fakebin``): nothing here contacts a
real scheduler. Expected values are written by hand from the fake's state
file, never computed by the code under test.
"""

import json
import os
import time

import pytest

from steerlab_server.api import job_status
from steerlab_server.api.executors import (
    SlurmExecutor, _parse_slurm_duration)
from steerlab_server.api.profile import capability_snapshot
from steerlab_server.experiment import resume as resume_mod

from test_auto_resubmit import _manager, fake_slurm  # noqa: F401 - fixture
from test_fake_scheduler import wrapper_only_slurm  # noqa: F401 - fixture


def _scheduler(job_id: str, **entry) -> None:
    """One job's entry in the fake scheduler's state file. Keys: state, exit,
    queue, start, end, node, elapsed, timelimit, reason, queue_start."""
    path = os.environ["FAKE_SLURM_STATE_FILE"]
    table = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            table = json.load(handle)
    table[job_id] = entry
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(table, handle)


def _local(text: str) -> float:
    """A scheduler-local ``2026-11-02T17:09:00`` as an epoch second."""
    return time.mktime(time.strptime(text, "%Y-%m-%dT%H:%M:%S"))


def _preflight(planned: int) -> dict:
    return {"verdict": "ok", "checks": [
        {"id": "walltime", "status": "ok", "message": "fits",
         "data": {"plannedRecords": planned}}]}


def _study_job(manager, tmp_path, slurm_id, *, status="submitted",
               name="study", extra_rr=None, planned=None, bundle=True):
    """A study job record shaped like ``submit_study``'s: records directory,
    the script it was submitted from, the command, and its Slurm bundle."""
    records = tmp_path / "records"
    records.mkdir(exist_ok=True)
    bundle_dir = tmp_path / f"{name}-slurm"
    bundle_dir.mkdir(exist_ok=True)
    rr = {"recordsDirectory": str(records),
          "scriptPath": str(bundle_dir / "run.sbatch"),
          **({"preflight": _preflight(planned)} if planned is not None else {}),
          **(extra_rr or {})}
    result = {"recordsDirectory": str(records),
              "command": ["python", "-m", "steerlab_server.cli", "bundle",
                          "execute", "--verb", "run"]}
    if bundle:
        result["slurmBundle"] = {"bundle_dir": str(bundle_dir)}
    return manager.record_external(
        "study-submit", status=status, executor="slurm",
        executor_job_id=slurm_id, requested_resources=rr, result=result,
        log=f"submitted study as Slurm job {slurm_id}")


def _run_directory(tmp_path, job, name, *, lines, torn=False, shard=None,
                   root_id=None):
    """A run directory with ``lines`` complete record lines (and a line cut
    off in mid-write when ``torn``), and the resume pointer that names it."""
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True)
    with open(run_dir / "generations.jsonl", "w", encoding="utf-8") as handle:
        for index in range(lines):
            handle.write(json.dumps({"condition": "baseline",
                                     "promptIndex": index,
                                     "text": f"answer {index}"}) + "\n")
        if torn:
            handle.write('{"condition": "baseline", "promptIn')
    if shard is not None:
        with open(run_dir / "shard.json", "w", encoding="utf-8") as handle:
            json.dump({"shardIndex": shard[0], "shardCount": shard[1],
                       "expectedRecords": shard[2]}, handle)
    records = job.requested_resources["recordsDirectory"]
    resume_mod.write_pointer(
        resume_mod.pointer_path_for_record(
            os.path.join(records, f"{root_id or job.id}.json")),
        str(run_dir), verb="run", experiment="study")
    return run_dir


# --- the scheduler's view ------------------------------------------------------

def test_a_pending_job_reports_the_schedulers_reason_and_expected_start(
        tmp_path, fake_slurm):
    manager = _manager(tmp_path)
    job = _study_job(manager, tmp_path, "5101", planned=50)
    _scheduler("5101", state="PENDING", queue="PENDING", reason="Priority",
               queue_start="2026-11-02T18:30:00", timelimit="08:00:00")

    document = job_status.job_status(manager, job)

    view = document["scheduler"]
    assert view["queryOk"] is True
    assert view["rawState"] == "PENDING"
    assert view["state"] == "submitted"
    assert view["reason"] == "Priority"
    assert view["expectedStartAt"] == _local("2026-11-02T18:30:00")
    assert view["startedAt"] is None
    assert view["nodeList"] is None
    assert view["timeLimitSeconds"] == 8 * 3600
    assert view["walltimeRemainingSeconds"] is None
    assert view["schedulerOutcome"] is None
    assert document["deathDetail"] is None
    # A waiting job has not started: its record says so (it used to carry
    # the submission time as its start).
    assert document["startedAt"] is None
    assert document["createdAt"] == job.created_at
    # Nothing written yet; the whole run's planning figure is labelled as one.
    assert document["progress"]["recordsWritten"] is None
    assert document["progress"]["recordsExpected"] == 50
    assert document["progress"]["recordsExpectedSource"] == "preflightEstimate"
    # The queue was asked for exactly the fixed codes.
    assert any("-j 5101 -h -o %T|%r|%S|%N|%M|%l" in call
               for call in fake_slurm.calls("squeue"))


def test_a_running_job_reports_records_written_counted_like_a_resume(
        tmp_path, fake_slurm):
    manager = _manager(tmp_path)
    job = _study_job(manager, tmp_path, "5102", status="running", planned=50)
    run_dir = _run_directory(tmp_path, job, "run-a", lines=3, torn=True)
    _scheduler("5102", state="RUNNING", queue="RUNNING",
               start="2026-11-02T17:43:00", node="gpu-node-07",
               elapsed="01:02:03", timelimit="08:00:00")

    document = job_status.job_status(manager, job)

    view = document["scheduler"]
    assert view["rawState"] == "RUNNING"
    assert view["state"] == "running"
    assert view["startedAt"] == _local("2026-11-02T17:43:00")
    assert view["nodeList"] == "gpu-node-07"
    assert view["elapsed"] == "01:02:03"
    assert view["elapsedSeconds"] == 3723          # 1 h + 2 min + 3 s
    assert view["timeLimitSeconds"] == 28800       # 8 h
    assert view["walltimeRemainingSeconds"] == 25077   # 28800 - 3723
    assert view["expectedStartAt"] is None
    # A running job is not asked about in the queue: accounting answers.
    assert view["source"] == "sacct"
    progress = document["progress"]
    # Three complete lines; the line cut off in mid-write does not count,
    # exactly as a resume would drop it.
    assert progress["recordsWritten"] == 3
    assert progress["tornTail"] is True
    assert progress["unit"] == "response"
    assert progress["recordsExpected"] == 50
    assert progress["lastRecordAt"] == os.path.getmtime(
        run_dir / "generations.jsonl")
    assert progress["runID"] == "run-a"
    assert progress["stage"] == "run"
    # The read is a read: the torn line is still there for the writer.
    assert (run_dir / "generations.jsonl").read_text().endswith('"promptIn')
    assert document["deathDetail"] is None


def test_a_timed_out_job_carries_the_scheduler_outcome_and_death_detail(
        tmp_path, fake_slurm):
    manager = _manager(tmp_path)
    job = _study_job(manager, tmp_path, "5103", status="running")
    _scheduler("5103", state="TIMEOUT", exit="0:1",
               start="2026-11-02T09:00:00", end="2026-11-02T17:00:00",
               elapsed="08:00:00", timelimit="08:00:00", node="gpu-node-02")

    manager.poll_slurm()

    stored = manager.get(job.id)
    # No "timed out" status: the job failed, and the field says who ended it.
    assert stored.status == "failed"
    assert stored.result["schedulerOutcome"] == "TIMEOUT"
    document = job_status.job_status(manager, stored)
    assert document["schedulerOutcome"] == "TIMEOUT"
    assert document["scheduler"]["schedulerOutcome"] == "TIMEOUT"
    assert document["scheduler"]["endedAt"] == _local("2026-11-02T17:00:00")
    assert document["deathDetail"] == (
        "Slurm reported TIMEOUT after 08:00:00 (walltime limit 08:00:00, "
        "exit 0:1)")


def test_an_out_of_memory_job_carries_the_scheduler_outcome(
        tmp_path, fake_slurm):
    manager = _manager(tmp_path)
    job = _study_job(manager, tmp_path, "5104", status="running")
    _scheduler("5104", state="OUT_OF_MEMORY", exit="0:125",
               start="2026-11-02T09:00:00", end="2026-11-02T09:20:00",
               elapsed="00:20:00", timelimit="08:00:00")

    manager.poll_slurm()

    stored = manager.get(job.id)
    assert stored.status == "failed"
    assert stored.result["schedulerOutcome"] == "OUT_OF_MEMORY"
    document = job_status.job_status(manager, stored)
    assert document["schedulerOutcome"] == "OUT_OF_MEMORY"
    assert document["deathDetail"].startswith(
        "Slurm reported OUT_OF_MEMORY after 00:20:00")


def test_a_job_that_completed_on_its_own_has_no_scheduler_outcome(
        tmp_path, fake_slurm):
    manager = _manager(tmp_path)
    job = _study_job(manager, tmp_path, "5105", status="running")
    _scheduler("5105", state="COMPLETED", exit="0:0",
               start="2026-11-02T09:00:00", end="2026-11-02T10:00:00")

    manager.poll_slurm()

    stored = manager.get(job.id)
    assert stored.status == "succeeded"
    assert "schedulerOutcome" not in stored.result
    assert job_status.job_status(manager, stored)["schedulerOutcome"] is None


def test_a_scheduler_outcome_survives_a_record_fold(tmp_path, fake_slurm):
    manager = _manager(tmp_path)
    job = _study_job(manager, tmp_path, "5106", status="running")
    _scheduler("5106", state="TIMEOUT", exit="0:1")
    manager.poll_slurm()
    records = job.requested_resources["recordsDirectory"]
    with open(os.path.join(records, f"{job.id}.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"id": job.id, "status": "failed",
                   "result": {"error": "killed"}}, handle)

    manager.reconcile(records, force=True)

    assert manager.get(job.id).result["schedulerOutcome"] == "TIMEOUT"


# --- the real start time ---------------------------------------------------------

def test_started_at_is_the_schedulers_start_not_the_submission(
        tmp_path, fake_slurm):
    manager = _manager(tmp_path)
    job = _study_job(manager, tmp_path, "5201")
    submitted = job.created_at
    assert job.started_at is None
    _scheduler("5201", state="RUNNING", queue="RUNNING",
               start="2026-11-02T17:43:00")

    manager.poll_slurm()

    stored = manager.get(job.id)
    assert stored.status == "running"
    assert stored.started_at == _local("2026-11-02T17:43:00")
    assert stored.created_at == submitted
    # Durable: a fresh controller reads the same start from the store.
    reread = manager.store.load_all()[job.id]
    assert reread.started_at == _local("2026-11-02T17:43:00")


def test_a_job_already_running_gets_its_real_start_without_a_transition(
        tmp_path, fake_slurm):
    # A record written before this change carried the submission time as its
    # start; the next tick replaces it with the scheduler's.
    manager = _manager(tmp_path)
    job = _study_job(manager, tmp_path, "5202", status="running")
    job.started_at = job.created_at
    manager.store.update(job)
    _scheduler("5202", state="RUNNING", queue="RUNNING",
               start="2026-11-02T17:43:00")

    assert manager.poll_slurm() == 0     # no status changed
    assert manager.store.load_all()[job.id].started_at == _local(
        "2026-11-02T17:43:00")


def test_a_local_job_still_starts_when_it_is_recorded(tmp_path):
    manager = _manager(tmp_path)
    job = manager.record_external("experiment:run", status="running",
                                  executor="local")
    assert job.started_at == job.created_at


# --- refusals and scope ------------------------------------------------------------

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


def test_an_unknown_id_is_refused_and_the_scheduler_is_never_asked(
        tmp_path, fake_slurm, monkeypatch):
    client, state = _route_client(tmp_path, monkeypatch)
    _study_job(state.jobs, tmp_path, "5301")
    _scheduler("5301", state="RUNNING", queue="RUNNING")
    # Neither an unknown SteerLab id nor a bare scheduler id passes through.
    for asked in ("nosuchjob", "5301", "5301;ls"):
        for route in ("status", "log-tail"):
            response = client.get(f"/api/jobs/{asked}/{route}")
            assert response.status_code == 404, (asked, route)
            assert response.json()["detail"] == "no such job"
    assert fake_slurm.calls("sacct") == []
    assert fake_slurm.calls("squeue") == []


def test_the_status_route_answers_a_known_job(tmp_path, fake_slurm,
                                              monkeypatch):
    client, state = _route_client(tmp_path, monkeypatch)
    job = _study_job(state.jobs, tmp_path, "5302", status="running")
    _scheduler("5302", state="RUNNING", queue="RUNNING", node="gpu-node-01")
    response = client.get(f"/api/jobs/{job.id}/status")
    assert response.status_code == 200
    body = response.json()
    assert body["schemaVersion"] == 1
    assert body["jobId"] == job.id
    assert body["executorJobID"] == "5302"
    assert body["scheduler"]["nodeList"] == "gpu-node-01"


def test_a_stored_id_that_is_not_a_scheduler_id_is_never_sent(
        tmp_path, fake_slurm):
    manager = _manager(tmp_path)
    job = _study_job(manager, tmp_path, "5303; rm -rf /")
    document = job_status.job_status(manager, job)
    assert document["scheduler"] is None
    assert "not one the scheduler could have issued" in document["notes"][0]
    assert fake_slurm.calls("sacct") == []
    with pytest.raises(ValueError):
        SlurmExecutor().status_observation("5303 --all")


def test_a_local_job_has_no_scheduler_view(tmp_path, fake_slurm):
    manager = _manager(tmp_path)
    job = manager.record_external("experiment:run", status="running",
                                  executor="local")
    document = job_status.job_status(manager, job)
    assert document["scheduler"] is None
    assert "runs on the server itself" in document["notes"][0]
    assert fake_slurm.calls("sacct") == []


# --- a parent and its shards ---------------------------------------------------------

def test_a_parent_reports_each_shard_and_their_progress_summed(
        tmp_path, fake_slurm):
    manager = _manager(tmp_path)
    shard_a = _study_job(manager, tmp_path, "5401", status="running",
                         name="shard-0",
                         extra_rr={"shardIndex": 0, "shardCount": 2})
    shard_b = _study_job(manager, tmp_path, "5402", status="checkpointed",
                         name="shard-1",
                         extra_rr={"shardIndex": 1, "shardCount": 2})
    # Shard 1 checkpointed and was resumed: its continuation carries it now.
    continuation = _study_job(
        manager, tmp_path, "5403", status="running", name="shard-1",
        extra_rr={"shardIndex": 1, "shardCount": 2,
                  "resubmitChain": [shard_b.id], "resubmitOf": shard_b.id})
    shard_b.result = {**shard_b.result, "resubmittedAs": continuation.id}
    manager.store.update(shard_b)
    parent = manager.record_external(
        "study-submit-bundle", status="running", executor="slurm",
        requested_resources={"shardChildren": [shard_a.id, shard_b.id],
                             "parallelJobs": 2,
                             "preflight": _preflight(50)})
    _run_directory(tmp_path, shard_a, "part-0", lines=20, shard=(0, 2, 25))
    # The continuation writes to the run its chain root started.
    _run_directory(tmp_path, continuation, "part-1", lines=7, shard=(1, 2, 25),
                   root_id=shard_b.id)
    _scheduler("5401", state="RUNNING", queue="RUNNING", node="gpu-a")
    _scheduler("5403", state="RUNNING", queue="RUNNING", node="gpu-b")

    document = job_status.job_status(manager, parent)

    assert document["scheduler"] is None
    shards = document["shards"]
    assert [s["jobId"] for s in shards] == [shard_a.id, continuation.id]
    assert [s["shardIndex"] for s in shards] == [0, 1]
    assert shards[1]["shardJob"] == shard_b.id
    assert [s["scheduler"]["nodeList"] for s in shards] == ["gpu-a", "gpu-b"]
    assert [s["progress"]["recordsWritten"] for s in shards] == [20, 7]
    assert [s["progress"]["recordsExpected"] for s in shards] == [25, 25]
    assert {s["progress"]["recordsExpectedSource"] for s in shards} == {
        "shardStamp"}
    assert document["progress"]["recordsWritten"] == 27      # 20 + 7
    assert document["progress"]["recordsExpected"] == 50     # 25 + 25
    assert document["progress"]["recordsExpectedSource"] == "shardStamp"


def test_a_shard_that_has_not_started_never_borrows_the_whole_studys_figure(
        tmp_path, fake_slurm):
    manager = _manager(tmp_path)
    shard = _study_job(manager, tmp_path, "5501", planned=50,
                       extra_rr={"shardIndex": 0, "shardCount": 2})
    _scheduler("5501", state="PENDING", queue="PENDING", reason="Resources")
    progress = job_status.job_status(manager, shard)["progress"]
    assert progress["recordsExpected"] is None
    assert progress["recordsExpectedSource"] is None


# --- a site that publishes wrapper commands -------------------------------------------

def test_status_goes_through_the_sites_wrapper_commands(
        tmp_path, wrapper_only_slurm, monkeypatch):
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    manager = _manager(tmp_path)
    job = _study_job(manager, tmp_path, "5601")
    _scheduler("5601", state="PENDING", queue="PENDING", reason="QOSMaxJobsPerUserLimit")
    document = job_status.job_status(manager, job)
    assert document["scheduler"]["reason"] == "QOSMaxJobsPerUserLimit"
    # Only the wrapper doubles are installed; their call logs prove they ran.
    assert any("-j 5601" in call for call in wrapper_only_slurm.calls("sacct"))
    assert any("-j 5601" in call for call in wrapper_only_slurm.calls("squeue"))


# --- the log tail ------------------------------------------------------------------------

def test_a_log_tail_returns_the_last_lines_and_the_jobs_own_output(
        tmp_path, fake_slurm):
    manager = _manager(tmp_path)
    job = _study_job(manager, tmp_path, "5701", status="running")
    for index in range(5):
        job.log(f"controller line {index}")
    bundle_dir = tmp_path / "study-slurm"
    (bundle_dir / "slurm-5701.out").write_text(
        "".join(f"record {index}\n" for index in range(10)), encoding="utf-8")

    document = job_status.log_tail(manager, job, 3)

    assert document["lines"] == ["controller line 2", "controller line 3",
                                 "controller line 4"]
    assert document["truncated"] is True
    out = document["schedulerOutput"]["stdout"]
    assert out["name"] == "slurm-5701.out"
    assert out["exists"] is True
    assert out["lines"] == ["record 7", "record 8", "record 9"]
    assert out["truncated"] is True
    err = document["schedulerOutput"]["stderr"]
    assert err["exists"] is False
    assert err["lines"] == []
    assert err["detail"] == ("slurm-5701.err does not exist (the job may not "
                             "have started yet, or the file was removed)")


def test_a_waiting_jobs_missing_output_is_explained(tmp_path, fake_slurm):
    manager = _manager(tmp_path)
    job = _study_job(manager, tmp_path, "5702")
    out = job_status.log_tail(manager, job, 10)["schedulerOutput"]["stdout"]
    assert out["exists"] is False
    assert out["detail"] == ("slurm-5702.out does not exist yet: the "
                             "scheduler creates it when the job starts")


def test_a_log_tail_reads_a_bounded_number_of_bytes(tmp_path, fake_slurm):
    manager = _manager(tmp_path)
    job = _study_job(manager, tmp_path, "5703", status="running")
    line = "x" * 99 + "\n"                          # 100 bytes per line
    lines = 4000                                    # 400,000 bytes in all
    path = tmp_path / "study-slurm" / "slurm-5703.err"
    path.write_text(line * (lines - 1) + "last line\n", encoding="utf-8")

    err = job_status.log_tail(manager, job, 1000)["schedulerOutput"]["stderr"]

    assert err["sizeBytes"] == 100 * (lines - 1) + len("last line\n")
    assert err["bytesRead"] == job_status.MAX_TAIL_BYTES == 256 * 1024
    assert err["lines"][-1] == "last line"
    assert len(err["lines"]) == 1000
    assert all(len(text) == 99 for text in err["lines"][:-1])
    assert err["truncated"] is True


def test_the_output_paths_come_from_the_record_only(tmp_path, fake_slurm):
    # After the child record folds, the bundle entry is gone; the directory
    # of the script the job was submitted from still names it.
    manager = _manager(tmp_path)
    job = _study_job(manager, tmp_path, "5704", status="running", bundle=False)
    (tmp_path / "study-slurm" / "slurm-5704.out").write_text("hello\n")
    out = job_status.log_tail(manager, job, 5)["schedulerOutput"]["stdout"]
    assert out["lines"] == ["hello"]


def test_a_local_job_has_no_scheduler_output(tmp_path):
    manager = _manager(tmp_path)
    job = manager.record_external("experiment:run", status="running",
                                  executor="local", log="started")
    document = job_status.log_tail(manager, job, 5)
    assert document["lines"] == ["started"]
    assert document["schedulerOutput"] is None


def test_the_log_tail_route_validates_its_line_count(tmp_path, fake_slurm,
                                                     monkeypatch):
    client, state = _route_client(tmp_path, monkeypatch)
    job = _study_job(state.jobs, tmp_path, "5705", status="running")
    for bad in ("0", "1001", "ten"):
        response = client.get(f"/api/jobs/{job.id}/log-tail",
                              params={"lines": bad})
        assert response.status_code == 400, bad
        assert "lines must be" in response.json()["detail"]
    response = client.get(f"/api/jobs/{job.id}/log-tail", params={"lines": "2"})
    assert response.status_code == 200
    assert response.json()["lines"] == ["submitted study as Slurm job 5705"]


# --- vocabulary --------------------------------------------------------------------------

def test_slurm_durations_read_as_seconds():
    assert _parse_slurm_duration("1-02:03:04") == 93784     # 1 d 2 h 3 m 4 s
    assert _parse_slurm_duration("02:03:04") == 7384
    assert _parse_slurm_duration("03:04") == 184
    assert _parse_slurm_duration("2-12") == 216000           # 2 d 12 h
    assert _parse_slurm_duration("90") == 5400               # minutes
    assert _parse_slurm_duration("UNLIMITED") is None
    assert _parse_slurm_duration("Partition_Limit") is None
    assert _parse_slurm_duration("") is None


def test_the_capability_snapshot_advertises_both_reads():
    cluster = capability_snapshot()["cluster"]
    assert cluster["jobStatus"] is True
    assert cluster["jobLogTail"] is True
