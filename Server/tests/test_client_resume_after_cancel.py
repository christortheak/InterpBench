"""The Python client's side of resuming a cancelled run (2026-10-04).

The engine decides whether a cancelled run can continue and says so in its
answer to the cancel. ``runner jobs <id> --cancel`` relays that sentence and
names the exact command that resumes on THIS client; ``runner resubmit``
reports the engine's own account of what was kept. An engine that predates
the feature answers the bare acknowledgement, and the client says nothing it
was not told.
"""

import json

import pytest

pytest.importorskip("httpx")

import httpx

from steerlab_server import client_cli
from steerlab_server.client import runner

ENDPOINT = "https://runner.example.invalid"

KEPT = ("Cancel requested. Responses this run has already completed are "
        "kept. Once the scheduler confirms the job has stopped, you can "
        "resume job example-job and it continues from there; nothing "
        "resumes it automatically.")
RESUMED = ("Job example-job was cancelled; it now continues as job next-job. "
           "The 3 response records it had completed are kept and will not be "
           "generated again. The cancelled job's record stays cancelled.")


def _wire(monkeypatch, respond):
    original = runner.RunnerClient
    monkeypatch.setattr(runner, "RunnerClient", lambda **kwargs: original(
        **kwargs, http_client=httpx.Client(
            transport=httpx.MockTransport(respond))))


def _job_record(request):
    return httpx.Response(200, json={
        "id": "example-job", "kind": "study-submit", "status": "cancelled",
        "executor": "slurm"})


def test_cancel_relays_the_engines_sentence_and_names_the_resume_command(
        monkeypatch, capsys):
    def respond(request):
        if request.url.path.endswith("/cancel"):
            return httpx.Response(200, json={
                "ok": True, "message": KEPT,
                "cancelResume": {"offered": True, "explanation": "…"}})
        return _job_record(request)

    _wire(monkeypatch, respond)
    assert client_cli.main(["runner", "jobs", "example-job", "--cancel",
                            "--runner", ENDPOINT, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["result"]
    assert result["cancelAccepted"] is True
    assert result["cancelNote"] == KEPT
    assert result["resumeCommand"] == (
        f"steerlab runner resubmit example-job --runner {ENDPOINT}")

    # The human reading the terminal sees both lines.
    assert client_cli.main(["runner", "jobs", "example-job", "--cancel",
                            "--runner", ENDPOINT]) == 0
    printed = capsys.readouterr().out
    assert KEPT in printed
    assert ("To resume it later: steerlab runner resubmit example-job "
            f"--runner {ENDPOINT}") in printed


def test_cancel_of_a_run_that_cannot_resume_names_no_command(monkeypatch,
                                                            capsys):
    note = ("Cancel requested. This local run cannot be resumed yet; submit "
            "it again.")

    def respond(request):
        if request.url.path.endswith("/cancel"):
            return httpx.Response(200, json={
                "ok": True, "message": note,
                "cancelResume": {"offered": False, "explanation": "…"}})
        return _job_record(request)

    _wire(monkeypatch, respond)
    assert client_cli.main(["runner", "jobs", "example-job", "--cancel",
                            "--runner", ENDPOINT, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["result"]
    assert result["cancelNote"] == note
    assert "resumeCommand" not in result


def test_an_older_engines_bare_acknowledgement_reads_as_before(monkeypatch,
                                                               capsys):
    def respond(request):
        if request.url.path.endswith("/cancel"):
            return httpx.Response(200, json={"ok": True})
        return _job_record(request)

    _wire(monkeypatch, respond)
    assert client_cli.main(["runner", "jobs", "example-job", "--cancel",
                            "--runner", ENDPOINT, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)["result"]
    assert result["cancelAccepted"] is True
    assert "cancelNote" not in result
    assert "resumeCommand" not in result


def test_resubmit_reports_the_engines_account_of_a_resume_after_a_cancel(
        monkeypatch, capsys):
    def respond(request):
        if request.url.path.endswith("/resubmit"):
            return httpx.Response(200, json={
                "ok": True, "jobId": "next-job", "resubmitOf": "example-job",
                "resumedAfterCancel": True, "completedRecords": 3,
                "message": RESUMED})
        return _job_record(request)

    _wire(monkeypatch, respond)
    assert client_cli.main(["runner", "resubmit", "example-job",
                            "--runner", ENDPOINT, "--json"]) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["message"] == RESUMED
    assert envelope["result"]["response"]["jobId"] == "next-job"
