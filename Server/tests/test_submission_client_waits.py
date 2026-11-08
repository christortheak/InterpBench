"""The Python client's half of public issue #11.

A submit waits long enough (its own budget, not the one-minute default); a
submit that was sent and never answered is ``submitOutcomeUnknown`` /
``pending`` (exit 12) rather than a failure, with a repair that says whether
a repeat is safe on this runner; and the key it sends is derived from what is
submitted. The server's half is ``test_submission_repeats.py``.

Only a loopback HTTP server is reached.
"""

import http.server
import json
import os
import threading
import time

import pytest

pytest.importorskip("httpx")

from steerlab_server.client import runner as runner_api




def test_the_capability_says_whether_a_repeat_is_safe():
    assert runner_api.submits_once({"cluster": {"submissionKey": True}}) is True
    assert runner_api.submits_once({"cluster": {}}) is False
    assert runner_api.submits_once(None) is False


def test_the_key_is_derived_from_what_is_submitted():
    body = {"bundlePath": "/runs/upload-1/b.tar.gz", "verb": "run",
            "dryRun": False, "packageEvidence": True, "executor": "slurm",
            "parallelJobs": 4, "resources": {"walltime": "08:00:00"}}
    key = runner_api.submission_key(body, "d" * 64)
    assert len(key) == 64 and int(key, 16) >= 0
    # A re-upload stages the same bytes at a new path: the same key.
    moved = {**body, "bundlePath": "/runs/upload-2/b.tar.gz"}
    assert runner_api.submission_key(moved, "d" * 64) == key
    # Key order never matters.
    assert runner_api.submission_key(dict(reversed(list(body.items()))),
                                     "d" * 64) == key
    # Anything that changes what is submitted changes it.
    for changed in ({"verb": "analyze"}, {"dryRun": True},
                    {"parallelJobs": 8}, {"executor": "local"},
                    {"resources": {"walltime": "09:00:00"}},
                    {"targetRoot": "/elsewhere"}):
        assert runner_api.submission_key({**body, **changed}, "d" * 64) != key
    assert runner_api.submission_key(body, "e" * 64) != key


def test_the_key_matches_its_hand_written_derivation():
    """Independent of the code under test: the canonical document, spelled
    out by hand, hashed by hashlib."""
    import hashlib
    body = {"bundlePath": "/x.tar.gz", "verb": "run", "dryRun": False}
    canonical = ('{"bundleContent":"' + "f" * 64 + '","dryRun":false,'
                 '"submissionKeyVersion":2,"verb":"run"}')
    assert runner_api.submission_key(body, "f" * 64) == \
        hashlib.sha256(canonical.encode()).hexdigest()
    # The same digest the Swift client's derivation is pinned to
    # (SubmissionRepeatCLITests.theKeyMatchesItsHandWrittenDerivation).
    assert runner_api.submission_key(body, "f" * 64) == (
        "e5a4afb57cef35974fb494d3a5ababbeb528e81a8082c0f0a682a27b5cc195e1")


class _SlowRunner(http.server.BaseHTTPRequestHandler):
    """A loopback runner whose submit answers after ``delay`` seconds."""

    delay = 0.0
    submit_key = True
    bodies: list = []

    def log_message(self, *args):  # quiet
        pass

    def _answer(self, document):
        payload = json.dumps(document).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/api/capabilities":
            cluster = {"submissionKey": True} if self.submit_key else {}
            self._answer({"cluster": cluster, "instrumentation": []})
        elif self.path == "/api/jobs/job1":
            self._answer({"id": "job1", "status": "checkpointed"})
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/api/bundles/inspect":
            self._answer({"kind": "runBundle", "experiment": "demo",
                          "bundleSha256": "d" * 64,
                          "runtimeRequirements": []})
            return
        if self.path == "/api/jobs/job1/resubmit":
            type(self).bodies.append(body)
            time.sleep(self.delay)
            try:
                self._answer({"jobId": "job2", "resubmittedFrom": "job1"})
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        if self.path == "/api/studies/submit-bundle":
            type(self).bodies.append(body)
            time.sleep(self.delay)
            try:
                self._answer({"jobId": "job-slow-1", "experiment": "demo",
                              "verb": body.get("verb"), "executor": "slurm",
                              "dryRun": False, "runBundle": {},
                              "command": [], "recordsDirectory": "/r",
                              "submissionDirectory": "/s"})
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.send_error(404)


@pytest.fixture
def slow_runner():
    handler = type("Handler", (_SlowRunner,), {"bodies": []})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield handler, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_a_submit_waits_past_the_one_request_budget(slow_runner, monkeypatch):
    """A submit the runner answers after longer than an ordinary request's
    budget is waited for (the issue: 60 s, against an 8-minute measurement;
    scaled down here to 0.5 s against 1.2 s)."""
    handler, url = slow_runner
    handler.delay = 1.2
    monkeypatch.setattr(runner_api, "DEFAULT_TIMEOUT", 0.5)
    monkeypatch.setattr(runner_api, "SUBMIT_TIMEOUT", 10.0)
    with runner_api.RunnerClient(base_url=url) as client:
        assert client.timeout == 0.5
        assert client.submit_timeout == 10.0
        submission = client.submit_uploaded_bundle(
            remote_path="/runs/b.tar.gz", verb="run", executor="slurm")
    assert submission["jobId"] == "job-slow-1"
    assert handler.bodies[0]["submissionKey"] == runner_api.submission_key(
        {k: v for k, v in handler.bodies[0].items() if k != "submissionKey"},
        "d" * 64)


def test_an_explicit_request_budget_still_takes_precedence(slow_runner,
                                                           monkeypatch):
    handler, url = slow_runner
    monkeypatch.setattr(runner_api, "SUBMIT_TIMEOUT", 10.0)
    with runner_api.RunnerClient(base_url=url, timeout=7.0) as client:
        assert client.submit_timeout == 7.0


@pytest.mark.parametrize("advertised", [True, False])
def test_a_submit_that_is_never_answered_is_pending_not_failed(
        slow_runner, monkeypatch, advertised):
    handler, url = slow_runner
    handler.delay = 2.0
    handler.submit_key = advertised
    monkeypatch.setattr(runner_api, "SUBMIT_TIMEOUT", 0.4)
    with runner_api.RunnerClient(base_url=url) as client:
        with pytest.raises(runner_api.SubmitOutcomeUnknown) as raised:
            client.submit_uploaded_bundle(remote_path="/runs/b.tar.gz",
                                          verb="run")
    error = raised.value
    assert error.code == "submitOutcomeUnknown"
    assert error.state == "pending"
    assert error.detail["repeatIsSafe"] is advertised
    assert "never answered" in error.reason
    assert f"steerlab runner jobs --runner {url}" in error.repair_action
    if advertised:
        assert "run the same command again with the same flags" in \
            error.repair_action
        assert "A command with any flag changed is a new submission" in \
            error.repair_action
    else:
        assert "a second submit is a second study" in error.repair_action
    # Exactly one attempt: nothing is retried.
    assert len(handler.bodies) == 1


def test_runner_submit_exits_12_with_the_unknown_outcome(slow_runner,
                                                         monkeypatch, capsys,
                                                         tmp_path):
    from steerlab_server import client_cli
    from steerlab_server.experiment import experiment_store as es

    handler, url = slow_runner
    handler.delay = 2.0
    monkeypatch.setattr(runner_api, "SUBMIT_TIMEOUT", 0.4)
    monkeypatch.delenv(client_cli.RUNNER_TOKEN_ENV, raising=False)
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)

    def info(self):
        return {"service": "steerlab-server", "engineVersion": "test",
                "root": "/srv", "capabilities": {}}

    monkeypatch.setattr(runner_api.RunnerClient, "info", info)
    del es
    code = client_cli.main(
        ["runner", "submit", "--runner", url, "--bundle-path",
         "/runs/b.tar.gz", "--bundle-sha", "d" * 64, "--verb", "run",
         "--executor", "slurm", "--root", workspace, "--json"])
    document = json.loads(capsys.readouterr().out)
    assert code == 12, document
    assert document["state"] == "pending"
    assert document["error"]["code"] == "submitOutcomeUnknown"
    assert document["result"]["repeatIsSafe"] is True


def test_runner_submit_says_plainly_when_it_was_already_submitted(
        slow_runner, monkeypatch, capsys, tmp_path):
    """A runner that recognised the key answers with the existing job: the
    verb says so, and reports nothing changed."""
    from steerlab_server import client_cli

    handler, url = slow_runner
    monkeypatch.delenv(client_cli.RUNNER_TOKEN_ENV, raising=False)
    note = ("Already submitted: this request was submitted before as job "
            "job-slow-1, which is running. Nothing was submitted again.")
    original = handler._answer

    def answer(self, document):
        if document.get("jobId") == "job-slow-1":
            document = {**document, "alreadySubmitted": True,
                        "alreadySubmittedNote": note,
                        "currentJobID": "job-slow-1",
                        "currentStatus": "running"}
        original(self, document)

    monkeypatch.setattr(handler, "_answer", answer)
    monkeypatch.setattr(runner_api.RunnerClient, "info", lambda self: {
        "service": "steerlab-server", "engineVersion": "test", "root": "/srv",
        "capabilities": {}})
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    code = client_cli.main(
        ["runner", "submit", "--runner", url, "--bundle-path",
         "/runs/b.tar.gz", "--bundle-sha", "d" * 64, "--verb", "run",
         "--executor", "slurm", "--root", workspace, "--json"])
    document = json.loads(capsys.readouterr().out)
    assert code == 0, document
    assert document["changed"] is False
    assert document["result"]["alreadySubmitted"] is True
    assert document["result"]["alreadySubmittedNote"] == note
    assert document["message"].startswith("already submitted:")


def test_a_resume_waits_as_long_as_a_submission(slow_runner, monkeypatch):
    """Review round 4: a resume can run sbatch inside the request. Answered
    after longer than an ordinary request's budget (0.5 s here), it is
    waited for."""
    handler, url = slow_runner
    handler.delay = 1.2
    monkeypatch.setattr(runner_api, "DEFAULT_TIMEOUT", 0.5)
    monkeypatch.setattr(runner_api, "SUBMIT_TIMEOUT", 10.0)
    with runner_api.RunnerClient(base_url=url) as client:
        assert client.resubmit_job("job1")["jobId"] == "job2"


def test_an_unanswered_resume_is_pending_with_the_repair_to_check(
        slow_runner, monkeypatch, capsys, tmp_path):
    from steerlab_server import client_cli
    handler, url = slow_runner
    handler.delay = 2.0
    monkeypatch.setattr(runner_api, "SUBMIT_TIMEOUT", 0.4)
    monkeypatch.delenv(client_cli.RUNNER_TOKEN_ENV, raising=False)
    with runner_api.RunnerClient(base_url=url) as client:
        with pytest.raises(runner_api.ResumeOutcomeUnknown) as raised:
            client.resubmit_job("job1")
    assert raised.value.state == "pending"
    assert raised.value.repair_action.startswith(
        f"steerlab runner status job1 --runner {url}")
    workspace = str(tmp_path / "ws")
    os.makedirs(workspace)
    code = client_cli.main(["runner", "resubmit", "job1", "--runner", url,
                            "--root", workspace, "--json"])
    document = json.loads(capsys.readouterr().out)
    assert code == 12, document
    assert document["error"]["code"] == "resumeOutcomeUnknown"
    assert len(handler.bodies) == 2, "one attempt each, nothing retried"
