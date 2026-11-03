"""``steerlab runner status`` and ``runner logs --tail`` (issue #7).

The Python client's twins of the Mac client's ``remote status`` and
``remote logs --tail``, driven through the real parser and envelope against
the real ASGI app (``test_client_runner.wired_cli``), with the committed fake
scheduler binaries on PATH. Nothing contacts a real scheduler.
"""

import json
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from steerlab_server import client_cli

from test_auto_resubmit import fake_slurm  # noqa: F401 - fixture
from test_client_runner import (  # noqa: F401 - fixtures
    _REAL_RUNNER_CLIENT, _envelope, adapter, runner_env, service, wired_cli)

RUNNER = ["--runner", "http://testserver"]


def _scheduler(job_id: str, **entry) -> None:
    path = os.environ["FAKE_SLURM_STATE_FILE"]
    table = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            table = json.load(handle)
    table[job_id] = entry
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(table, handle)


def _slurm_job(wired, slurm_id, *, status="running"):
    return wired["state"].jobs.record_external(
        "study-submit", status=status, executor="slurm",
        executor_job_id=slurm_id,
        log=f"submitted study as Slurm job {slurm_id}")


def test_runner_status_reports_the_schedulers_view(wired_cli, fake_slurm,
                                                   capsys):
    job = _slurm_job(wired_cli, "6101")
    _scheduler("6101", state="RUNNING", queue="RUNNING", node="gpu-node-03",
               elapsed="00:10:00", timelimit="01:00:00")

    code = client_cli.main(["runner", "status", job.id, *RUNNER])
    printed = capsys.readouterr().out
    assert code == 0
    assert f"job {job.id} [running] — scheduler job 6101" in printed
    assert ("scheduler: RUNNING on gpu-node-03; elapsed 00:10:00 of "
            "01:00:00 (50m left)") in printed

    code, document, _ = _envelope(["runner", "status", job.id, *RUNNER], capsys)
    assert code == 0, document
    response = document["result"]["response"]
    assert response["scheduler"]["nodeList"] == "gpu-node-03"
    assert response["scheduler"]["walltimeRemainingSeconds"] == 3000
    assert document["message"] == f"job {job.id} is running"


def test_runner_status_of_an_unknown_job_is_not_found(wired_cli, capsys):
    code, document, _ = _envelope(["runner", "status", "nosuchjob", *RUNNER],
                                  capsys)
    assert code == 66, document
    assert document["error"]["code"] == "notFound"


def test_runner_status_refuses_a_path_shaped_id(wired_cli, capsys):
    for bad in ("../capabilities", "a/b", ".."):
        code, document, _ = _envelope(["runner", "status", bad, *RUNNER],
                                      capsys)
        assert code == 64, (bad, document)
        assert "is not a job id" in document["error"]["reason"]


def test_runner_status_on_an_older_runner_is_refused(wired_cli, capsys,
                                                     monkeypatch):
    job = _slurm_job(wired_cli, "6102")
    monkeypatch.setattr(_REAL_RUNNER_CLIENT, "capabilities",
                        lambda self: {"cluster": {"manualResubmit": True}})
    code, document, _ = _envelope(["runner", "status", job.id, *RUNNER], capsys)
    assert code == 65, document
    assert document["error"]["code"] == "jobStatusUnsupported"


def test_runner_logs_tail_returns_the_last_lines(wired_cli, fake_slurm,
                                                 capsys):
    job = _slurm_job(wired_cli, "6103")
    for index in range(4):
        job.log(f"line {index}")
    code, document, _ = _envelope(
        ["runner", "logs", job.id, "--tail", "2", *RUNNER], capsys)
    assert code == 0, document
    assert document["result"]["response"]["lines"] == ["line 2", "line 3"]
    assert document["result"]["tailLines"] == 2
    # No bundle on this record, so the scheduler output says where it is not.
    output = document["result"]["response"]["schedulerOutput"]
    assert output["available"] is False


def test_runner_logs_tail_validates_its_count_and_excludes_follow(
        wired_cli, capsys):
    job = _slurm_job(wired_cli, "6104")
    for extra in (["--tail", "0"], ["--tail", "1001"], ["--tail", "ten"],
                  ["--tail", "5", "--follow"]):
        code, document, _ = _envelope(
            ["runner", "logs", job.id, *extra, *RUNNER], capsys)
        assert code == 64, (extra, document)


def test_the_plain_logs_verb_is_unchanged(wired_cli, capsys):
    job = _slurm_job(wired_cli, "6105")
    code, document, _ = _envelope(["runner", "logs", job.id, *RUNNER], capsys)
    assert code == 0, document
    assert document["result"]["lines"] == ["submitted study as Slurm job 6105"]
    assert document["result"]["followed"] is False


def test_the_summary_says_what_the_scheduler_recorded():
    # Review 2026-11-02: "ended by the scheduler" was wrong for a person's
    # own cancel and for a preempted job queued again; the runner's sentence
    # says what the scheduler recorded, and names a SteerLab cancel.
    from steerlab_server.client import job_reads
    document = {
        "jobId": "j1", "status": "cancelled", "executorJobID": "6201",
        "schedulerOutcome": "CANCELLED",
        "schedulerOutcomeNote": ("The scheduler recorded CANCELLED: the job "
                                 "was cancelled through SteerLab."),
        "notes": [],
    }
    lines = job_reads.status_lines(document)
    assert ("  The scheduler recorded CANCELLED: the job was cancelled "
            "through SteerLab.") in lines
    assert job_reads.status_message(document) == (
        "job j1 is cancelled (the scheduler recorded CANCELLED)")
    del document["schedulerOutcomeNote"]
    assert "  The scheduler recorded CANCELLED." in job_reads.status_lines(
        document)
    assert not any("ended by the scheduler" in line
                   for line in job_reads.status_lines(document))


def test_a_record_behind_the_scheduler_says_what_the_scheduler_reports():
    # Seen live 2026-11-03: the scheduler said RUNNING one controller check
    # before SteerLab's record did, and the message said only "submitted".
    from steerlab_server.client import job_reads
    document = {
        "jobId": "j2", "status": "submitted", "executorJobID": "6301",
        "scheduler": {"queryOk": True, "rawState": "RUNNING",
                      "state": "running", "nodeList": "n7"},
        "progress": {"recordsWritten": None}, "notes": [],
    }
    assert job_reads.status_message(document) == (
        "job j2 is submitted (the scheduler already reports RUNNING: "
        "SteerLab's record catches up on the controller's next check)")
    assert ("  SteerLab's record still says submitted: it catches up on the "
            "controller's next check.") in job_reads.status_lines(document)

    document["status"] = "running"
    document["scheduler"] = {"queryOk": True, "rawState": "COMPLETED",
                             "state": "succeeded"}
    assert job_reads.scheduler_ahead(document) == "COMPLETED"

    # Agreement, an ended record, and a scheduler that did not answer add
    # nothing.
    for status, view in (
            ("running", {"queryOk": True, "rawState": "RUNNING",
                         "state": "running"}),
            ("failed", {"queryOk": True, "rawState": "COMPLETED",
                        "state": "succeeded"}),
            ("submitted", {"queryOk": False, "rawState": "RUNNING",
                           "state": "running"})):
        quiet = {**document, "status": status, "scheduler": view}
        assert job_reads.scheduler_ahead(quiet) is None, status
        assert "already reports" not in job_reads.status_message(quiet)
        assert not any("record still says" in line
                       for line in job_reads.status_lines(quiet))
