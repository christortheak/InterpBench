"""The submission key names what is submitted, not when it was packaged
(public issue #11, review round 2).

Running ``steerlab run`` again repackages the study, and every packaging
stamps its own ``createdAt``, so the archive's digest changes even when
nothing in the study did. A key derived from that digest made the "same"
command a different submission. The key now names the bundle by its
CONTENT — the study's name and content hash and every packaged file's path
and digest — together with every request parameter, so:

- the same command with the same flags gives the same key however often the
  study is repackaged and uploaded;
- any flag that changes what runs gives a different key;
- the repair after an unanswered submit names that exact command.

Only an in-process stand-in for the runner is reached; the packaging, the
inspection of the bundle, and the key derivation are the real ones.
"""

import hashlib
import json
import os

import pytest

pytest.importorskip("httpx")

from steerlab_server import client_cli
from steerlab_server.client import runner as runner_api
from steerlab_server.experiment import bundles

from test_run_orchestration import RUNNER_URL, STUDY_NAME, _frozen_workspace


def test_the_content_digest_ignores_packaging_and_matches_its_derivation():
    """Independent of the code under test: the canonical document spelled out
    by hand (``printf … | shasum -a 256``); the Swift client is pinned to the
    same digest (``SubmissionRepeatCLITests``)."""
    metadata = {"experiment": "demo", "experimentContentHash": "cafe",
                "createdAt": 1.0, "bundleSha256": "f" * 64,
                "entries": [
                    {"path": "prompts/tasks/t.jsonl", "sha256": "2" * 64,
                     "bytes": 9},
                    {"path": "experiments/demo/experiment.json",
                     "sha256": "1" * 64, "bytes": 5}]}
    digest = runner_api.bundle_content_digest(metadata)
    assert digest == (
        "06d989f81b0723e471b1241772721085cbf85a14a628ae82fcbba5393b78cb1f")
    repackaged = {**metadata, "createdAt": 2.0, "bundleSha256": "e" * 64,
                  "bundlePath": "/elsewhere.tar.gz"}
    assert runner_api.bundle_content_digest(repackaged) == digest
    changed = {**metadata, "entries": [*metadata["entries"][:1],
                                       {"path": "experiments/demo/"
                                        "experiment.json", "sha256": "3" * 64}]}
    assert runner_api.bundle_content_digest(changed) != digest
    assert runner_api.bundle_content_digest({"experiment": "demo"}) is None


def test_two_real_packagings_of_one_study_have_one_content_digest(
        tmp_path, monkeypatch):
    root = _frozen_workspace(tmp_path, monkeypatch)
    first = bundles.package_experiment(STUDY_NAME, root=root)
    second = bundles.package_experiment(STUDY_NAME, root=root)
    assert first["createdAt"] != second["createdAt"]
    assert first["bundleSha256"] != second["bundleSha256"]
    assert runner_api.bundle_content_digest(
        bundles.inspect_bundle(first["bundlePath"])) == \
        runner_api.bundle_content_digest(
            bundles.inspect_bundle(second["bundlePath"]))


class _Runner(runner_api.RunnerClient):
    """The real adapter, with only the network replaced: the bundle it
    "staged" is the local archive, its inspection is the real one, and the
    submit answers (or, with ``unanswered``, times out) here."""

    bodies: list = []
    archives: list = []
    unanswered = False

    def info(self):
        return {"service": "steerlab-server", "engineVersion": "test",
                "root": "/runner-root", "devices": ["cpu"],
                "loadedModels": [], "capabilities": self.capabilities()}

    def capabilities(self):
        return {"schedulerMode": "slurm", "serverRole": "controller",
                "availableJobTypes": ["run", "extract", "validate", "sweep"],
                "instrumentation": ["policy-evidence-v2", "policy-v1",
                                    "probe-readings-v1"],
                "cluster": {"submissionKey": True}}

    def upload_run_bundle(self, path):
        digest = runner_api.sha256_file(path)
        type(self).archives.append(digest)
        return {"path": path, "filename": os.path.basename(path),
                "sha256": digest, "localSha256": digest,
                "bytes": os.path.getsize(path),
                "localBytes": os.path.getsize(path), "executable": True,
                "stagingDirectory": os.path.dirname(path),
                "bundle": {"kind": "runBundle", "experiment": STUDY_NAME}}

    def inspect_remote_bundle(self, remote_path):
        return bundles.inspect_bundle(remote_path)

    def _json(self, method, path, **kwargs):
        if path != "/api/studies/submit-bundle":
            return super()._json(method, path, **kwargs)
        type(self).bodies.append(dict(kwargs["json_body"]))
        if type(self).unanswered:
            import httpx
            error = runner_api.RunnerUnreachable(
                "ReadTimeout reaching /api/studies/submit-bundle",
                repair_action="look")
            raise error from httpx.ReadTimeout("timed out")
        return {"jobId": f"job-{len(type(self).bodies)}",
                "experiment": STUDY_NAME, "verb": "run", "executor": "slurm",
                "dryRun": False, "runBundle": {}, "command": [],
                "recordsDirectory": "/r", "submissionDirectory": "/s"}


@pytest.fixture
def wired(tmp_path, monkeypatch):
    root = _frozen_workspace(tmp_path, monkeypatch)
    runner = type("Runner", (_Runner,), {"bodies": [], "archives": [],
                                         "unanswered": False})
    monkeypatch.setattr(runner_api, "RunnerClient", runner)
    return root, runner


def _run(root, *flags, capsys):
    code = client_cli.main(["--root", root, "run", STUDY_NAME, "--runner",
                            RUNNER_URL, *flags, "--json"])
    return code, json.loads(capsys.readouterr().out)


def test_running_the_same_command_again_carries_the_same_key(wired, capsys):
    """THE review's probe: `steerlab run X --parallel 4`, twice. Each run
    repackages the study (a new archive digest), and both carry one key."""
    root, runner = wired
    flags = ("--parallel", "4", "--executor", "slurm", "--no-wait")
    assert _run(root, *flags, capsys=capsys)[0] == 12      # detached
    assert _run(root, *flags, capsys=capsys)[0] == 12
    first, second = runner.bodies
    assert runner.archives[0] != runner.archives[1], "each run repackages"
    assert first["parallelJobs"] == 4
    assert first["submissionKey"] == second["submissionKey"]


@pytest.mark.parametrize("changed", [
    ("--parallel", "2", "--executor", "slurm", "--no-wait"),
    ("--parallel", "4", "--executor", "local", "--no-wait"),
    ("--parallel", "4", "--executor", "slurm", "--no-wait", "--no-evidence"),
    ("--parallel", "4", "--executor", "slurm", "--no-wait", "--dtype",
     "bfloat16"),
    ("--executor", "slurm", "--no-wait"),
])
def test_changing_a_flag_changes_the_key(wired, capsys, changed):
    root, runner = wired
    _run(root, "--parallel", "4", "--executor", "slurm", "--no-wait",
         capsys=capsys)
    _run(root, *changed, capsys=capsys)
    first, second = runner.bodies
    assert first["submissionKey"] != second["submissionKey"]


def test_the_unanswered_repair_names_the_same_command_and_its_key_matches(
        wired, capsys):
    """The repair's command is this command with every flag; running it
    gives the same key the unanswered submit carried."""
    root, runner = wired
    runner.unanswered = True
    code, document = _run(root, "--parallel", "4", "--executor", "slurm",
                          "--dtype", "bfloat16", "--no-evidence", "--no-wait",
                          capsys=capsys)
    assert code == 12, document
    assert document["error"]["code"] == "submitOutcomeUnknown"
    assert document["result"]["repeatIsSafe"] is True
    again = document["result"]["sameCommand"]
    assert again in document["error"]["repairAction"]
    for flag in ("--parallel 4", "--executor slurm", "--dtype bfloat16",
                 "--no-evidence", "--no-wait", f"--runner {RUNNER_URL}",
                 "--json",
                 f"--root {root}", f"run {STUDY_NAME}"):
        assert flag in again, flag
    # Run exactly what the repair says (it answers this time, and detaches).
    import shlex
    runner.unanswered = False
    words = shlex.split(again)
    assert words[0] == "steerlab"
    assert client_cli.main(words[1:]) == 12
    capsys.readouterr()
    assert runner.archives[0] != runner.archives[1], "repackaged"
    assert runner.bodies[0]["submissionKey"] == runner.bodies[1]["submissionKey"]


def test_a_refused_connection_is_not_an_unknown_outcome(monkeypatch):
    """Nothing was sent: the adapter raises the ordinary connection failure."""
    import httpx

    class Refusing(runner_api.RunnerClient):
        def inspect_remote_bundle(self, remote_path):
            return {"kind": "runBundle", "runtimeRequirements": [],
                    "entries": [{"path": "a", "sha256": "1" * 64}]}

    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    client = Refusing(base_url="http://127.0.0.1:9",
                      http_client=httpx.Client(
                          transport=httpx.MockTransport(refuse)))
    with pytest.raises(runner_api.RunnerUnreachable) as raised:
        client.submit_uploaded_bundle(remote_path="/b.tar.gz", verb="run")
    assert not isinstance(raised.value, runner_api.SubmitOutcomeUnknown)
    assert raised.value.request_sent is False
    assert raised.value.code == "runnerUnreachable"


def test_a_read_timeout_is_an_unknown_outcome(monkeypatch):
    import httpx

    class Slow(runner_api.RunnerClient):
        def inspect_remote_bundle(self, remote_path):
            return {"kind": "runBundle", "runtimeRequirements": [],
                    "entries": [{"path": "a", "sha256": "1" * 64}]}

    def slow(request):
        if request.url.path == "/api/capabilities":
            return httpx.Response(200, json={"cluster": {}})
        raise httpx.ReadTimeout("timed out", request=request)

    client = Slow(base_url="http://127.0.0.1:9",
                  http_client=httpx.Client(transport=httpx.MockTransport(slow)))
    with pytest.raises(runner_api.SubmitOutcomeUnknown) as raised:
        client.submit_uploaded_bundle(remote_path="/b.tar.gz", verb="run")
    assert raised.value.state == "pending"
    assert raised.value.detail["repeatIsSafe"] is False


def test_hashlib_agrees_with_the_hand_derivation_of_a_key():
    body = {"bundlePath": "/x.tar.gz", "verb": "run", "dryRun": False}
    canonical = ('{"bundleContent":"' + "f" * 64 + '","dryRun":false,'
                 '"submissionKeyVersion":2,"verb":"run"}')
    assert runner_api.submission_key(body, "f" * 64) == \
        hashlib.sha256(canonical.encode()).hexdigest()


def test_the_same_command_keeps_where_its_document_goes(wired, capsys,
                                                       tmp_path):
    """`--out` names where the document is written: kept in the repeat,
    and not part of the key (it is never sent)."""
    root, runner = wired
    runner.unanswered = True
    out = str(tmp_path / "doc.json")
    code = client_cli.main(["--root", root, "run", STUDY_NAME, "--runner",
                            RUNNER_URL, "--executor", "slurm", "--no-wait",
                            "--json", "--out", out])
    capsys.readouterr()
    assert code == 12
    with open(out, encoding="utf-8") as handle:
        document = json.load(handle)
    again = document["result"]["sameCommand"]
    assert again.endswith(f"--json --out {out}")
    import shlex
    runner.unanswered = False
    other = str(tmp_path / "other.json")
    words = shlex.split(again.replace(out, other))
    assert client_cli.main(words[1:]) == 12
    capsys.readouterr()
    assert runner.bodies[0]["submissionKey"] == runner.bodies[1]["submissionKey"]


def _client(handler):
    import httpx

    class Staged(runner_api.RunnerClient):
        def inspect_remote_bundle(self, remote_path):
            return {"kind": "runBundle", "runtimeRequirements": [],
                    "entries": [{"path": "a", "sha256": "1" * 64}]}

    def route(request):
        if request.url.path == "/api/capabilities":
            return httpx.Response(200, json={"cluster": {"submissionKey": True}})
        return handler(request)

    return Staged(base_url="http://127.0.0.1:9",
                  http_client=httpx.Client(transport=httpx.MockTransport(route)))


def test_an_unreadable_success_is_an_unknown_outcome():
    """The runner took the request and answered 200 with something this
    client cannot read: the job may have landed (the Mac reads it so too)."""
    import httpx
    client = _client(lambda request: httpx.Response(200, text="<html>ok</html>"))
    with pytest.raises(runner_api.SubmitOutcomeUnknown) as raised:
        client.submit_uploaded_bundle(remote_path="/b.tar.gz", verb="run")
    assert "could not be read" in raised.value.reason
    assert raised.value.detail["repeatIsSafe"] is True


def test_a_connect_timeout_sent_nothing():
    """A connection that was never made sent nothing: an ordinary connection
    failure, as the Mac reads one from its metrics."""
    import httpx

    def never(request):
        raise httpx.ConnectTimeout("connect timed out", request=request)

    with pytest.raises(runner_api.RunnerUnreachable) as raised:
        _client(never).submit_uploaded_bundle(remote_path="/b.tar.gz",
                                              verb="run")
    assert not isinstance(raised.value, runner_api.SubmitOutcomeUnknown)
    assert raised.value.request_sent is False
