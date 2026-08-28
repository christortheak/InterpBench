"""The runner HTTP adapter and the ``runner`` verb family (Phase 2 of the
portability program).

Phase 1b's client authors a workspace and talks to nothing. Phase 2 is the
other half: hand a hash-pinned bundle to an engine, watch the job, bring the
evidence home **verified**. What is pinned here, and how each half fails:

1. **The walk.** upload → submit → status → logs → evidence, driven through
   :class:`steerlab_server.client.runner.RunnerClient` against the real ASGI
   app. If this breaks, the client cannot reach a runner at all.
2. **The two integrity checks**, which are the reason the adapter exists
   rather than a shell script: an upload the runner hashes differently is
   refused, and an evidence archive whose digest does not match what the
   runner reported is refused *before anything reaches the destination path*.
   If these break, a truncated or substituted archive becomes citable
   evidence.
3. **The token discipline.** A distinctive fake token is driven through every
   surface — success, refusal, HTTP error, unreachable runner, ``repr`` — and
   must appear in none of them. If this breaks, a bearer token leaks into a
   log, an envelope, or a traceback.
4. **The wire, for real, once.** One test launches the actual server as a
   subprocess on a loopback TCP port in token mode and runs the adapter
   against it end to end. The in-process tests share a process with the app;
   this one shares only a socket, which is the only place "does the token
   really travel as a header?" has an answer.
5. **The import graph.** Out of process, like Phase 1b's: the ``runner`` family
   may pull httpx and nothing heavier.

Fixtures are neutral throughout — a concept named ``signal`` with two
one-word stimuli. Nothing here is about a study.
"""

import errno
import json
import os
import socket
import subprocess
import sys
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx

from steerlab_server import client_cli
from steerlab_server.client import runner as runner_api
from steerlab_server.experiment import bundles, experiment_store as es
from steerlab_server.experiment import paths

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Distinctive enough that a substring search for it is meaningful, and
#: obviously not a credential anyone could mistake for real.
FAKE_TOKEN = "steerlab-test-token-DO-NOT-LEAK-8f2c1a"

#: The neutral study every test uploads. One concept, two one-word stimuli,
#: no arms: `bundle package` pins what the manifest declares, and this
#: declares almost nothing on purpose — the bundle is the fixture, not the
#: science.
STUDY_NAME = "runner-fixture"
CONCEPT = "signal"


# =============================================================================
# fixtures
# =============================================================================


def _authored_workspace(root: str) -> str:
    """A workspace holding one draft study, and the run bundle packaged from
    it. Returns the bundle path."""
    directory = os.path.join(root, "prompts", "concepts", CONCEPT)
    os.makedirs(directory, exist_ok=True)
    for filename, text in (("positive.jsonl", '{"text": "on"}\n'),
                           ("negative.jsonl", '{"text": "off"}\n')):
        with open(os.path.join(directory, filename), "w",
                  encoding="utf-8") as handle:
            handle.write(text)
    es.create(STUDY_NAME, model_id="org/tiny", revision="0" * 40, root=root)
    es.attach(STUDY_NAME, [CONCEPT], root=root)
    return bundles.package_experiment(STUDY_NAME, root=root)["bundlePath"]


@pytest.fixture
def runner_env(tmp_path, monkeypatch):
    """The runner's own roots, isolated per test.

    The client and the engine share a filesystem here (they are one process),
    which is exactly the shape of a loopback dev runner and is why the
    ``targetRoot`` below is a DIFFERENT tree from the authoring root: the
    engine imports the bundle rather than reading the client's workspace.
    """
    source = str(tmp_path / "workspace")
    os.makedirs(source, exist_ok=True)
    monkeypatch.setenv("STEERLAB_ROOT", source)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "jobs.sqlite"))
    monkeypatch.delenv(client_cli.RUNNER_TOKEN_ENV, raising=False)
    return {"source": source, "target": str(tmp_path / "target"),
            "bundle": _authored_workspace(source), "tmp": tmp_path}


@pytest.fixture
def service(runner_env):
    """The real router on a real ASGI app, plus the ``ServiceState`` behind it.

    ``TestClient`` IS an ``httpx.Client``, so the adapter's ``http_client``
    seam drives the genuine routes with no second implementation and no fake —
    and the suite's ``conftest`` gives every ``TestClient`` a 127.0.0.1 peer,
    so ``auth_middleware``'s fail-closed peer check behaves as it would for a
    loopback caller.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.routes import ServiceState, build_router

    state = ServiceState()
    app = FastAPI()
    app.include_router(build_router(state))
    client = TestClient(app)
    try:
        yield {"http": client, "state": state, **runner_env}
    finally:
        client.close()


@pytest.fixture
def adapter(service):
    """A :class:`RunnerClient` wired to the in-process app."""
    with runner_api.RunnerClient(base_url="http://testserver",
                                 token=FAKE_TOKEN,
                                 http_client=service["http"]) as client:
        yield client


@pytest.fixture
def wired_cli(monkeypatch, service):
    """Point ``client_cli``'s runner verbs at the in-process app.

    The CLI builds its own ``RunnerClient`` (it must: the token and the TLS
    policy are its business, not a caller's). The only thing swapped is the
    transport, so every verb below exercises the real parsing, the real
    envelope, and the real adapter.
    """
    real = runner_api.RunnerClient

    def factory(**kwargs):
        # `setdefault`, not assignment: a test that wants a DIFFERENT transport
        # (a mock returning 401, say) must be able to say so and be obeyed,
        # rather than have this fixture quietly hand it the live app back.
        kwargs.setdefault("http_client", service["http"])
        return real(**kwargs)

    monkeypatch.setattr(runner_api, "RunnerClient", factory)
    return service


def _cli(argv, capsys=None):
    """Drive the client and return ``(exit_code, stdout, stderr)``."""
    code = client_cli.main(list(argv))
    if capsys is None:
        return code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _envelope(argv, capsys):
    """Drive the client under ``--json`` and return the parsed document."""
    code = client_cli.main(list(argv) + ["--json"])
    captured = capsys.readouterr()
    return code, json.loads(captured.out), captured.err


def _await_terminal(adapter_client, job_id, *, deadline=120.0):
    """Poll ``GET /api/jobs/{id}`` until the job stops moving.

    Polling with a deadline rather than sleeping a guessed interval: the job
    finishes in milliseconds in isolation and can be starved for seconds under
    a full-suite run, and a fixed sleep is how that difference becomes a
    flake.
    """
    from steerlab_server.api.jobs import TERMINAL

    limit = time.monotonic() + deadline
    record = adapter_client.job(job_id)
    while record.get("status") not in TERMINAL:
        if time.monotonic() > limit:
            raise AssertionError(
                f"job {job_id} still {record.get('status')!r} after "
                f"{deadline}s; log tail: {record.get('logTail')}")
        time.sleep(0.02)
        record = adapter_client.job(job_id)
    return record


def _evidence_bearing_job(service, *, run_name="evidence-fixture"):
    """A real job record on the real job manager whose result points at a real
    evidence bundle under the runner's runs root.

    **Why fabricate the job rather than run one.** The only bundled verb that
    needs no model is ``verify``, and ``verify`` writes no run directory — so
    it packages no evidence (``bundles._execute_run_bundle_inner``). Every
    verb that does produce a run directory loads a model. The EVIDENCE half of
    the walk is therefore driven over a job created through the engine's own
    ``JobManager.submit`` — the same seam the API routes use — carrying an
    archive ``bundles.package_evidence`` really wrote. Everything the adapter
    touches (the job route, the result shape, the archive, the digest, the
    download route) is genuine; only the compute that would have produced it
    is skipped.
    """
    run_directory = paths.make_unique_run_directory(run_name)
    with open(os.path.join(run_directory, "config.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"experiment": STUDY_NAME, "verb": "run"}, handle)
    with open(os.path.join(run_directory, "records.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"record": 1}\n')
    meta = bundles.package_evidence(run_directory)

    result = {"experiment": STUDY_NAME, "verb": "run",
              "runDirectory": run_directory,
              # The shape a local `submit-bundle` really produces:
              # `submissions._read_child_record` folds the child's document in
              # under `runResult`.
              "runResult": {"evidenceBundle": meta}}
    job = service["state"].jobs.submit("study-submit-bundle",
                                       lambda _job: result)
    limit = time.monotonic() + 30.0
    while job.status not in ("succeeded", "failed"):
        assert time.monotonic() < limit, "fixture job never finished"
        time.sleep(0.01)
    assert job.status == "succeeded", job.error
    return job.id, meta


def _lying_evidence(service) -> dict:
    """A real archive under the runner's runs root, pointed at by a digest
    that is not its own — the substitution case, staged honestly."""
    _job_id, meta = _evidence_bearing_job(service, run_name="lying-fixture")
    return {"bundlePath": meta["bundlePath"], "bundleSha256": "e" * 64,
            "experiment": STUDY_NAME}


def _job_pointing_at(service, reference: dict) -> str:
    job = service["state"].jobs.submit(
        "study-submit-bundle",
        lambda _job: {"runResult": {"evidenceBundle": reference}})
    limit = time.monotonic() + 30.0
    while job.status not in ("succeeded", "failed"):
        assert time.monotonic() < limit, "fixture job never finished"
        time.sleep(0.01)
    return job.id


# =============================================================================
# 1. The walk
# =============================================================================


def test_the_adapter_walks_upload_submit_status_logs_and_evidence(
        adapter, service):
    """CONTRACT: the whole Phase-2 path, end to end, over the real routes.

    upload (digest agreed) → submit (identity pre-checked) → status → logs →
    evidence (downloaded and verified, NOT imported).
    """
    # -- upload -------------------------------------------------------------
    uploaded = adapter.upload_run_bundle(service["bundle"])
    assert uploaded["localSha256"] == uploaded["sha256"], \
        "the adapter accepted an upload it should have compared"
    assert uploaded["executable"] is True
    assert uploaded["bundle"]["kind"] == "runBundle"
    assert uploaded["bundle"]["experiment"] == STUDY_NAME
    assert uploaded["localBytes"] == uploaded["bytes"]
    # The digest is the one the CLIENT computed off the file it sent.
    assert uploaded["localSha256"] == runner_api.sha256_file(service["bundle"])

    # -- submit -------------------------------------------------------------
    submission = adapter.submit_uploaded_bundle(
        remote_path=uploaded["path"],
        expected_sha256=uploaded["bundle"]["bundleSha256"],
        verb="verify", executor="local", target_root=service["target"])
    assert submission["experiment"] == STUDY_NAME
    assert submission["verb"] == "verify"
    assert submission["jobId"]

    # -- status -------------------------------------------------------------
    record = _await_terminal(adapter, submission["jobId"])
    assert record["status"] == "succeeded", record.get("error")
    assert os.path.exists(os.path.join(
        service["target"], "experiments", STUDY_NAME, "experiment.json")), \
        "the runner executed the bundle it was given, into the named target"
    assert any(row["id"] == submission["jobId"] for row in adapter.jobs())

    # -- logs ---------------------------------------------------------------
    logs = adapter.job_logs(submission["jobId"])
    assert logs["followed"] is False
    assert logs["status"] == "succeeded"
    assert logs["lineCount"] == len(logs["lines"])
    assert any("verify" in line for line in logs["lines"]), logs["lines"]

    # -- evidence -----------------------------------------------------------
    job_id, meta = _evidence_bearing_job(service)
    reference = adapter.evidence_reference(job_id)
    assert reference["bundleSha256"] == meta["bundleSha256"]

    destination = os.path.join(str(service["tmp"]), "home", "evidence.tar.gz")
    downloaded = adapter.download_bundle(
        remote_path=reference["bundlePath"],
        expected_sha256=reference["bundleSha256"],
        destination=destination,
        temp_path=destination + ".partial")
    assert downloaded["verified"] is True
    assert downloaded["imported"] is False, \
        "this adapter downloads and verifies; importing is a separate act"
    assert downloaded["sha256"] == meta["bundleSha256"]
    assert os.path.isfile(destination)
    assert not os.path.exists(destination + ".partial"), "staging debris"
    with open(destination, "rb") as got, open(meta["bundlePath"], "rb") as want:
        assert got.read() == want.read()


def test_the_cli_walks_the_same_path_and_reports_it_in_the_envelope(
        wired_cli, capsys):
    """The same walk through the VERBS, checking the documents an agent reads.

    The envelope is where a caller learns the job id, the runner's identity,
    and the digest that ties them together — and it is where the credential
    must not be.
    """
    service = wired_cli
    runner = ["--runner", "http://testserver"]

    code, document, _ = _envelope(["runner", "capabilities", *runner], capsys)
    assert code == 0, document
    assert document["state"] == "ready"
    assert document["result"]["service"] == "steerlab-server"
    assert document["result"]["engineVersion"].startswith("steerlab-server ")
    assert document["result"]["tokenPresent"] is False

    code, document, _ = _envelope(
        ["runner", "upload", service["bundle"], *runner], capsys)
    assert code == 0, document
    digest = document["result"]["sha256"]
    staged = document["result"]["runnerPath"]
    assert digest == runner_api.sha256_file(service["bundle"])
    assert document["changed"] is True
    # The next action is runnable as printed — it carries both halves.
    assert f"--bundle-sha {digest}" in document["nextAction"]["verb"]

    code, document, _ = _envelope(
        ["runner", "submit", *runner, "--bundle-path", staged,
         "--bundle-sha", digest, "--verb", "verify", "--executor", "local",
         "--target-root", service["target"]], capsys)
    assert code == 0, document
    job_id = document["result"]["jobId"]
    assert document["result"]["bundleSha256"] == digest
    assert document["result"]["runnerIdentity"]["service"] == "steerlab-server"
    assert document["result"]["experiment"] == STUDY_NAME

    from steerlab_server.api.jobs import TERMINAL
    limit = time.monotonic() + 120.0
    while True:
        code, document, _ = _envelope(
            ["runner", "jobs", job_id, *runner], capsys)
        assert code == 0, document
        if document["result"]["job"]["status"] in TERMINAL:
            break
        assert time.monotonic() < limit, document
        time.sleep(0.02)
    assert document["result"]["job"]["status"] == "succeeded"

    code, document, _ = _envelope(["runner", "logs", job_id, *runner], capsys)
    assert code == 0, document
    assert document["result"]["lines"]


def test_runner_evidence_verifies_downloads_and_names_the_import_command(
        wired_cli, capsys):
    """CONTRACT: `runner evidence` stops at the verified file.

    Importing writes into a workspace and can overwrite; it is a separate,
    named act, and the verb says the exact command rather than performing it.
    """
    service = wired_cli
    job_id, meta = _evidence_bearing_job(service)
    destination = os.path.join(str(service["tmp"]), "home", "run.tar.gz")

    code, document, _ = _envelope(
        ["runner", "evidence", job_id, "--runner", "http://testserver",
         "--out", destination], capsys)

    assert code == 0, document
    result = document["result"]
    assert result["verified"] is True
    assert result["imported"] is False
    assert result["sha256"] == meta["bundleSha256"]
    assert os.path.isfile(destination)
    expected = f"steerlab bundle import {destination} --sha256 " \
               f"{meta['bundleSha256']}"
    assert result["importCommand"] == expected
    assert document["nextAction"]["verb"] == (
        f"bundle import {destination} --sha256 {meta['bundleSha256']}")


def test_evidence_from_a_job_that_packaged_none_is_a_typed_refusal(
        wired_cli, capsys):
    """A `verify` job really does produce no evidence — the refusal says so
    and names what to check, instead of failing on a missing key."""
    service = wired_cli
    job = service["state"].jobs.submit("study-submit-bundle",
                                       lambda _job: {"violations": []})
    limit = time.monotonic() + 30.0
    while job.status not in ("succeeded", "failed"):
        assert time.monotonic() < limit
        time.sleep(0.01)

    code, document, _ = _envelope(
        ["runner", "evidence", job.id, "--runner", "http://testserver",
         "--out", os.path.join(str(service["tmp"]), "nope.tar.gz")], capsys)

    assert code == 65, document
    assert document["error"]["code"] == client_cli.NO_EVIDENCE_CODE
    assert "packageEvidence" in document["error"]["repairAction"]
    assert not os.path.exists(os.path.join(str(service["tmp"]), "nope.tar.gz"))


def test_evidence_reference_reads_every_shape_the_engine_writes(adapter):
    """Three code paths write the evidence pointer into three different
    places. A client that knew one would report "no evidence" for runs that
    have some."""
    reference = {"bundlePath": "/runs/x.tar.gz", "bundleSha256": "a" * 64}
    for result in ({"evidenceBundle": reference},
                   {"runResult": {"evidenceBundle": reference}},
                   {"result": {"evidenceBundle": reference}}):
        assert adapter.evidence_reference(
            "ignored", record={"result": result}) == reference
    assert adapter.evidence_reference("ignored", record={"result": {}}) is None
    assert adapter.evidence_reference("ignored", record={}) is None
    # A pointer with no path is not a pointer.
    assert adapter.evidence_reference(
        "ignored",
        record={"result": {"evidenceBundle": {"bundleSha256": "b" * 64}}}
    ) is None


# =============================================================================
# 2. The two integrity checks
# =============================================================================


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler),
                        base_url="http://runner.invalid")


def test_an_upload_the_runner_hashes_differently_is_refused(
        service, tmp_path):
    """CONTRACT: a bundle that did not arrive intact never becomes a
    submission.

    Driven through a mock transport rather than by breaking the server: the
    thing under test is the CLIENT's comparison, and the honest way to make a
    runner disagree is to have one disagree.
    """
    wrong = "0" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/bundles/upload"
        return httpx.Response(200, json={
            "path": "/runs/staged/x.tar.gz", "filename": "x.tar.gz",
            "sha256": wrong, "bytes": 10, "bundle": {"kind": "runBundle"},
            "executable": True, "stagingDirectory": "/runs/staged"})

    with runner_api.RunnerClient(base_url="http://runner.invalid",
                                 http_client=_mock_client(handler)) as client:
        with pytest.raises(runner_api.RunnerRefusal) as caught:
            client.upload_run_bundle(service["bundle"])

    exc = caught.value
    assert exc.code == "uploadDigestMismatch"
    assert exc.state == "refused"
    # BOTH numbers are named: a refusal that reports only one leaves the
    # reader unable to tell which side is wrong.
    local = runner_api.sha256_file(service["bundle"])
    assert wrong in exc.reason and local in exc.reason
    assert exc.detail["reportedSha256"] == wrong
    assert exc.detail["localSha256"] == local
    assert "do NOT submit" in exc.repair_action


def test_a_download_whose_digest_disagrees_writes_nothing_to_the_destination(
        adapter, service):
    """CONTRACT (the client half of ``evidence-outer-hash-pre-extraction``):
    verified BEFORE the file reaches the caller's path.

    The archive here is real and the route is real; only the digest the caller
    was told to expect is wrong — which is precisely the substitution case a
    per-member check cannot catch, because a swapped bundle is internally
    consistent.
    """
    _job_id, meta = _evidence_bearing_job(service)
    destination = os.path.join(str(service["tmp"]), "home", "evidence.tar.gz")
    temp_path = destination + ".partial"

    with pytest.raises(runner_api.RunnerRefusal) as caught:
        adapter.download_bundle(
            remote_path=meta["bundlePath"], expected_sha256="f" * 64,
            destination=destination, temp_path=temp_path)

    exc = caught.value
    assert exc.code == "evidenceDigestMismatch"
    assert exc.detail["expectedSha256"] == "f" * 64
    assert exc.detail["downloadedSha256"] == meta["bundleSha256"]
    # The whole point: nothing at the destination, and no debris beside it.
    assert not os.path.exists(destination)
    assert not os.path.exists(temp_path)
    assert "do NOT import this file" in exc.repair_action


def test_a_download_over_the_size_limit_stops_mid_stream(adapter, service):
    """The cap is enforced as the bytes arrive, not after — an adapter that
    read a runaway body into memory first would have nothing left to refuse
    with."""
    _job_id, meta = _evidence_bearing_job(service)
    destination = os.path.join(str(service["tmp"]), "home", "capped.tar.gz")

    with pytest.raises(runner_api.RunnerRefusal) as caught:
        adapter.download_bundle(
            remote_path=meta["bundlePath"],
            expected_sha256=meta["bundleSha256"], destination=destination,
            temp_path=destination + ".partial", max_bytes=16)

    assert caught.value.code == "downloadTooLarge"
    assert not os.path.exists(destination)
    assert not os.path.exists(destination + ".partial")


def test_a_download_refuses_to_overwrite_an_existing_destination(
        adapter, service):
    """Evidence is immutable. A silent overwrite is how one run's results
    quietly become a different run's."""
    _job_id, meta = _evidence_bearing_job(service)
    destination = os.path.join(str(service["tmp"]), "home", "taken.tar.gz")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as handle:
        handle.write(b"already here")

    with pytest.raises(runner_api.RunnerRefusal) as caught:
        adapter.download_bundle(
            remote_path=meta["bundlePath"],
            expected_sha256=meta["bundleSha256"], destination=destination,
            temp_path=destination + ".partial")

    assert caught.value.code == "destinationExists"
    with open(destination, "rb") as handle:
        assert handle.read() == b"already here"


def test_a_download_with_no_expected_digest_is_refused_outright(adapter):
    """There is no "just fetch it" mode. An unverifiable evidence bundle is
    not a faster path to the same place — it is a different artifact."""
    with pytest.raises(runner_api.RunnerRefusal) as caught:
        adapter.download_bundle(remote_path="/runs/x.tar.gz",
                                expected_sha256="", destination="/tmp/x",
                                temp_path="/tmp/x.partial")
    assert caught.value.code == "unverifiableDownload"


def test_submit_refuses_a_staged_path_that_is_not_the_pinned_bundle(
        adapter, service):
    """CONTRACT: the pre-check happens while checking is still free.

    Submit is not idempotent — it creates a job and can spend an allocation —
    so the identity of what is about to run is confirmed on an idempotent
    ``inspect`` first. Here two different bundles are staged and the wrong
    digest is pinned: nothing is submitted.
    """
    uploaded = adapter.upload_run_bundle(service["bundle"])
    before = len(adapter.jobs())

    with pytest.raises(runner_api.RunnerRefusal) as caught:
        adapter.submit_uploaded_bundle(
            remote_path=uploaded["path"], expected_sha256="c" * 64,
            verb="verify", executor="local")

    exc = caught.value
    assert exc.code == "bundleDigestMismatch"
    assert exc.detail["runnerSha256"] == uploaded["bundle"]["bundleSha256"]
    assert "nothing was submitted" in exc.reason
    assert len(adapter.jobs()) == before, "a refused submit created a job"


# =============================================================================
# 3. The token discipline
# =============================================================================


def test_there_is_no_token_flag_on_any_runner_verb():
    """CONTRACT, structural: argv is public.

    ``ps`` on a shared login node shows every argument of every process, and
    shell history keeps them afterwards. The token comes from a file or the
    environment; there is no flag that could hold it, so no wrapper can
    accidentally put one on a command line.
    """
    for spec in client_cli.CLIENT_VERB_SPECS:
        for flag in spec.declared_flags:
            assert flag != "--token", f"{spec.label} declares --token"
            # Matched on WORD segments, not substrings. A CREDENTIAL flag says
            # "token" singular (`--token`, `--auth-token`, `--bearer-token`);
            # `--max-tokens` is plural and counts LLM tokens, which is a
            # generation budget and not a secret. The substring form flagged
            # it — a false positive that would have to be silenced somehow,
            # and silencing a secret guard is worse than tightening it.
            if "token" in flag.lstrip("-").split("-"):
                assert flag == "--token-file", (
                    f"{spec.label} declares {flag} — the only token spelling "
                    "on this surface is --token-file (a path, not a secret)")
    # …and the environment variable is the client's own, never the engine's:
    # a machine running both must not send its local server's secret to a
    # remote host.
    assert client_cli.RUNNER_TOKEN_ENV == "STEERLAB_RUNNER_TOKEN"
    assert client_cli.RUNNER_TOKEN_ENV != "STEERLAB_AUTH_TOKEN"


def test_the_token_never_reaches_any_envelope_log_line_or_exception(
        wired_cli, monkeypatch, capsys, tmp_path):
    """CONTRACT: the credential appears in NOTHING the client emits.

    Every surface a token could escape through is driven with the same
    distinctive fake value: a successful envelope, human-mode stdout/stderr,
    the ``repr`` of the client object, a refusal document, an HTTP-error
    message, and an unreachable-runner message. The only thing any of them may
    say about the credential is the presence boolean.
    """
    service = wired_cli
    monkeypatch.setenv(client_cli.RUNNER_TOKEN_ENV, FAKE_TOKEN)
    runner = ["--runner", "http://testserver"]
    transcript: list[str] = []

    # (1) a success, in both output modes
    code, document, err = _envelope(["runner", "capabilities", *runner],
                                    capsys)
    assert code == 0
    assert document["result"]["tokenPresent"] is True, \
        "the presence boolean is the ONE thing a document may say"
    transcript.extend([json.dumps(document), err])
    code, out, err = _cli(["runner", "capabilities", *runner], capsys)
    transcript.extend([out, err])

    # (2) a refusal document (an upload of a file that is not there)
    code, document, err = _envelope(
        ["runner", "upload", str(tmp_path / "missing.tar.gz"), *runner],
        capsys)
    assert code == 66, document
    transcript.extend([json.dumps(document), err])

    # (3) a real download refusal, with a payload. The archive and the route
    # are real; the job simply points at a digest that is not the one on disk,
    # which is the substitution case the outer hash exists to catch.
    job_id = _job_pointing_at(service, _lying_evidence(service))
    destination = os.path.join(str(tmp_path), "leak.tar.gz")
    code, document, err = _envelope(
        ["runner", "evidence", job_id, *runner, "--out", destination], capsys)
    assert code == 65, document
    assert document["error"]["code"] == "evidenceDigestMismatch"
    transcript.extend([json.dumps(document), err])

    # (4) the object's own repr, and every adapter-raised message
    with runner_api.RunnerClient(base_url="http://testserver",
                                 token=FAKE_TOKEN,
                                 http_client=service["http"]) as client:
        assert client.has_token is True
        transcript.append(repr(client))
        assert "<present>" in repr(client)

    # (5) an HTTP error the runner produced, and an unreachable runner
    def refusing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "missing or invalid bearer "
                                                   "token"})

    with runner_api.RunnerClient(base_url="http://runner.invalid",
                                 token=FAKE_TOKEN,
                                 http_client=_mock_client(refusing)) as client:
        with pytest.raises(runner_api.RunnerHTTPError) as caught:
            client.info()
        transcript.extend([caught.value.reason, caught.value.repair_action,
                           json.dumps(caught.value.detail), repr(client),
                           str(caught.value)])

    def exploding(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with runner_api.RunnerClient(base_url="http://runner.invalid",
                                 token=FAKE_TOKEN,
                                 http_client=_mock_client(exploding)) as client:
        with pytest.raises(runner_api.RunnerUnreachable) as caught:
            client.info()
        transcript.extend([caught.value.reason, str(caught.value)])
        # And the belt-and-braces scrubber, proven rather than assumed.
        assert client.scrub(f"oops {FAKE_TOKEN} oops") == \
            "oops <token redacted> oops"

    # (6) a credential smuggled in the LOCATOR rather than the header.
    # `--runner https://user:pass@host/` used to be accepted verbatim and then
    # reproduced by every surface above — including the `remote-execution.json`
    # provenance record, which is written once and read forever. It is now a
    # typed refusal, and the refusal redacts what it refused.
    url_secret = "hunter2-DO-NOT-LEAK-9d41b7"
    embedded = f"https://someone:{url_secret}@testserver/"
    code, document, err = _envelope(
        ["runner", "capabilities", "--runner", embedded], capsys)
    assert code == 65, document
    assert document["error"]["code"] == runner_api.URL_REFUSED_CODE
    transcript.extend([json.dumps(document), err])
    code, out, err = _cli(["runner", "capabilities", "--runner", embedded],
                          capsys)
    assert code == 65
    transcript.extend([out, err])
    # …and the composite verb, whose provenance record is the durable surface.
    code, document, err = _envelope(
        ["run", STUDY_NAME, "--runner", embedded,
         "--root", service["source"]], capsys)
    assert code == 65, document
    assert document["error"]["code"] == runner_api.URL_REFUSED_CODE, \
        "the composite must refuse the locator BEFORE it reads the study"
    transcript.extend([json.dumps(document), err])
    for text in transcript:
        assert url_secret not in text, (
            "a URL-embedded credential escaped into client output: "
            + text[:400])
        assert "someone" not in text, (
            "the userinfo escaped into client output: " + text[:400])

    for text in transcript:
        assert FAKE_TOKEN not in text, (
            "the bearer token escaped into client output: "
            + text[:400])
    # The transcript must actually contain something, or this proves nothing.
    assert sum(len(t) for t in transcript) > 500


def test_the_token_travels_only_as_an_authorization_header(service):
    """Where the token DOES go: one header, never a URL.

    A token in a query string lands in every proxy log and every browser
    history on the path; a header does not. Checked by watching the request
    the adapter actually builds.
    """
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"service": "steerlab-server"})

    with runner_api.RunnerClient(base_url="http://runner.invalid",
                                 token=FAKE_TOKEN,
                                 http_client=_mock_client(handler)) as client:
        client.info()

    assert seen["auth"] == f"Bearer {FAKE_TOKEN}"
    assert FAKE_TOKEN not in seen["url"]

    # …and with no token, no header at all — an unauthenticated loopback
    # runner must not be sent an empty Bearer, which reads as a bad token.
    seen.clear()
    with runner_api.RunnerClient(base_url="http://runner.invalid", token=None,
                                 http_client=_mock_client(handler)) as client:
        assert client.has_token is False
        assert "<absent>" in repr(client)
        client.info()
    assert seen["auth"] is None


def test_a_token_file_is_read_and_a_world_readable_one_is_announced(
        wired_cli, tmp_path, capsys, monkeypatch):
    """``--token-file`` is a PATH, and the client says so when the path is
    readable by everyone — the same call the engine's serve-time posture
    makes."""
    monkeypatch.delenv(client_cli.RUNNER_TOKEN_ENV, raising=False)
    path = tmp_path / "runner.token"
    path.write_text(FAKE_TOKEN + "\n", encoding="utf-8")
    os.chmod(path, 0o644)

    code, document, err = _envelope(
        ["runner", "capabilities", "--runner", "http://testserver",
         "--token-file", str(path)], capsys)

    assert code == 0, document
    assert document["result"]["tokenPresent"] is True
    assert "mode 0644" in err and "chmod 600" in err
    assert FAKE_TOKEN not in err and FAKE_TOKEN not in json.dumps(document)


def test_an_unreadable_or_empty_token_file_refuses_by_name(
        wired_cli, tmp_path, capsys):
    missing = tmp_path / "no-such.token"
    code, document, _ = _envelope(
        ["runner", "capabilities", "--runner", "http://testserver",
         "--token-file", str(missing)], capsys)
    assert code == 66, document
    assert document["error"]["code"] == "notFound"
    assert client_cli.RUNNER_TOKEN_ENV in document["error"]["repairAction"]

    empty = tmp_path / "empty.token"
    empty.write_text("   \n", encoding="utf-8")
    code, document, _ = _envelope(
        ["runner", "capabilities", "--runner", "http://testserver",
         "--token-file", str(empty)], capsys)
    assert code == 64, document
    assert document["error"]["code"] == client_cli.RUNNER_USAGE_CODE


# =============================================================================
# 4. Capabilities, and the shape of the surface
# =============================================================================


def test_capabilities_parses_and_both_routes_agree(adapter):
    """CONTRACT (e): the identity document a client branches on.

    ``/api/info`` embeds the same snapshot ``/api/capabilities`` serves, which
    is why the verb makes ONE request in the common case. If they ever stop
    agreeing, the verb is reporting a snapshot from a route it did not call.
    """
    from steerlab_server import build_identity

    info = adapter.info()
    assert info["service"] == "steerlab-server"
    assert info["engineVersion"] == build_identity.engine_version()
    assert isinstance(info["devices"], list) and "cpu" in info["devices"]
    assert info["root"] == os.path.realpath(paths.project_root())

    snapshot = adapter.capabilities()
    assert isinstance(snapshot, dict)
    assert info["capabilities"] == snapshot

    identity = adapter.identity(info)
    assert set(identity) == set(runner_api.IDENTITY_KEYS)
    assert identity["service"] == "steerlab-server"
    assert identity["rootLooksLikeSourceCheckout"] is False


def test_a_url_that_is_not_a_runner_says_so_rather_than_crashing():
    """Pointing ``--runner`` at a proxy, a static site, or the wrong port is
    the commonest first mistake. It gets a typed refusal, not a JSON decode
    traceback."""
    def html(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>hello</html>")

    with runner_api.RunnerClient(base_url="http://runner.invalid",
                                 http_client=_mock_client(html)) as client:
        with pytest.raises(runner_api.RunnerError) as caught:
            client.info()
    assert caught.value.code == "notARunner"
    assert caught.value.state == "blocked"
    assert "/api/info" in caught.value.repair_action


def test_an_unknown_job_is_66_and_a_401_is_64(wired_cli, capsys, monkeypatch):
    """The two statuses a caller most often meets, mapped to the states the
    exit-code table already defines. 404 is notFound (66) because a mistyped
    job id is the commonest miss; 401 is blocked (64) because the repair is a
    credential the caller can supply, not a gate the study failed."""
    code, document, _ = _envelope(
        ["runner", "jobs", "no-such-job", "--runner", "http://testserver"],
        capsys)
    assert code == 66, document
    assert document["error"]["code"] == "notFound"

    def refusing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "missing or invalid bearer "
                                                   "token"})

    real = runner_api.RunnerClient

    def factory(**kwargs):
        kwargs["http_client"] = _mock_client(refusing)
        return real(**kwargs)

    monkeypatch.setattr(runner_api, "RunnerClient", factory)
    code, document, _ = _envelope(
        ["runner", "capabilities", "--runner", "https://runner.invalid"],
        capsys)
    assert code == 64, document
    assert document["error"]["code"] == "runnerUnauthorized"
    assert client_cli.RUNNER_TOKEN_ENV in document["error"]["repairAction"]
    assert "--token-file" in document["error"]["repairAction"]


def test_the_runner_family_needs_no_workspace_but_honours_one(
        wired_cli, monkeypatch, capsys, tmp_path):
    """A verb that addresses a REMOTE engine and names its local paths
    explicitly has no reason to demand a local workspace — while a named
    workspace is still resolved, so the envelope's ``workspace`` field stays
    truthful."""
    monkeypatch.delenv(client_cli.WORKSPACE_ENV, raising=False)
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    code, document, _ = _envelope(
        ["runner", "capabilities", "--runner", "http://testserver"], capsys)
    assert code == 0, document
    assert "workspace" not in document, \
        "a runner verb that resolved no workspace must not report one"

    # An authoring verb in the same state still refuses — the waiver is per
    # family, not a new default.
    code, document, _ = _envelope(["experiment", "list"], capsys)
    assert code == 64
    assert document["error"]["code"] == client_cli.WORKSPACE_UNSET_CODE

    # And a named workspace is honoured by the runner family.
    workspace = tmp_path / "named-ws"
    workspace.mkdir()
    code, document, _ = _envelope(
        ["--root", str(workspace), "runner", "capabilities",
         "--runner", "http://testserver"], capsys)
    assert code == 0, document
    assert document["workspace"] == os.path.realpath(str(workspace))

    # A --root that names nothing still refuses, on every family.
    code, document, _ = _envelope(
        ["--root", str(tmp_path / "absent"), "runner", "capabilities",
         "--runner", "http://testserver"], capsys)
    assert code == 66, document


def test_a_runner_verb_without_a_locator_says_which_flag_is_missing(capsys):
    code, document, _ = _envelope(["runner", "capabilities"], capsys)
    assert code == 64, document
    assert document["error"]["code"] == client_cli.RUNNER_USAGE_CODE
    assert "--runner" in document["error"]["repairAction"]


def test_out_belongs_to_the_verb_that_declares_it(wired_cli, capsys, tmp_path):
    """``--out`` is the envelope's destination on every verb EXCEPT the two
    that declare it as their own argument.

    Before Phase 2 the global branch ran first unconditionally, so
    ``bundle package --out`` silently wrote the DOCUMENT to the path and
    packaged to the default location — a declared flag quietly doing something
    else. The declaration now wins; this pins both halves.
    """
    service = wired_cli
    job_id, meta = _evidence_bearing_job(service)
    destination = os.path.join(str(tmp_path), "declared.tar.gz")
    code = client_cli.main(["runner", "evidence", job_id, "--runner",
                            "http://testserver", "--out", destination])
    capsys.readouterr()
    assert code == 0
    # The ARCHIVE is there, not an envelope.
    assert os.path.isfile(destination)
    with open(destination, "rb") as handle:
        assert handle.read(2) == b"\x1f\x8b", "that is not a gzip archive"

    # A verb that does NOT declare --out still gets the envelope destination.
    document_path = tmp_path / "envelope.json"
    code = client_cli.main(["runner", "jobs", job_id, "--runner",
                            "http://testserver", "--out", str(document_path)])
    capsys.readouterr()
    assert code == 0
    assert json.loads(document_path.read_text(
        encoding="utf-8"))["verb"] == "runner jobs"


def test_cancel_needs_a_job_id(wired_cli, capsys):
    code, document, _ = _envelope(
        ["runner", "jobs", "--cancel", "--runner", "http://testserver"],
        capsys)
    assert code == 64, document
    assert "whole queue" in document["error"]["reason"]


# =============================================================================
# 5. One real socket
# =============================================================================


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_runner(client: runner_api.RunnerClient, process,
                     *, deadline: float) -> dict:
    """Poll ``GET /api/info`` until the server answers, or give up.

    A DEADLINE and a poll, never a sleep: a fixed sleep is either too short
    (flaky on a loaded machine) or too long (a slow suite), and it cannot
    notice that the subprocess already died. This can.
    """
    limit = time.monotonic() + deadline
    last = ""
    while time.monotonic() < limit:
        if process.poll() is not None:
            raise AssertionError(
                f"the server exited with {process.returncode} before it "
                f"served; last error: {last}")
        try:
            return client.info()
        except runner_api.RunnerError as exc:
            last = exc.reason
        time.sleep(0.1)
    raise AssertionError(f"no runner on {client.base_url} within {deadline}s; "
                         f"last error: {last}")


def test_the_adapter_works_against_a_real_server_over_tcp_in_token_mode(
        tmp_path):
    """CONTRACT: the wire, once, for real.

    Every other test in this file shares a process with the app. That is the
    right trade for the logic — but it cannot answer "does the bearer token
    actually travel as a header over a socket, and does a real uvicorn accept
    a chunked upload body?", because no socket and no HTTP parser are
    involved. This one launches the genuine ``steerlab-server serve`` on a
    loopback ephemeral port in TOKEN mode and runs the adapter against it.

    Not marked slow: this suite has no slow/integration marker, and inventing
    one to hold a single test would give every other test a category nobody
    maintains. It is kept cheap instead — a dry-run submit, no model, no
    compute — and it is written not to flake: readiness is a polled deadline
    on ``/api/info`` rather than a sleep, the subprocess is watched for early
    exit while polling, and teardown terminates then kills in a ``finally``.
    """
    pytest.importorskip("uvicorn")

    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace, exist_ok=True)
    bundle = _authored_workspace(workspace)

    token_file = tmp_path / "runner.token"
    token_file.write_text(FAKE_TOKEN, encoding="utf-8")
    os.chmod(token_file, 0o600)

    port = _free_port()
    env = {
        **os.environ,
        "PYTHONPATH": SERVER_DIR,
        "STEERLAB_ROOT": workspace,
        "STEERLAB_RUN_ROOT": str(tmp_path / "runs"),
        "STEERLAB_JOBS_DB": str(tmp_path / "jobs.sqlite"),
        "STEERLAB_METADATA_ROOT": str(tmp_path / "meta"),
        # WP-S: token mode on every route, with the value hydrated from the
        # FILE rather than passed on a command line — the same discipline this
        # client follows on its own side.
        "STEERLAB_AUTH_MODE": "token",
        "STEERLAB_AUTH_TOKEN_FILE": str(token_file),
        "STEERLAB_EXECUTOR": "local",
        "HF_HUB_OFFLINE": "1",
    }
    env.pop("STEERLAB_AUTH_TOKEN", None)

    process = subprocess.Popen(
        [sys.executable, "-m", "steerlab_server.cli", "serve",
         "--host", "127.0.0.1", "--port", str(port)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base_url = f"http://127.0.0.1:{port}"
    try:
        with runner_api.RunnerClient(base_url=base_url,
                                     token=FAKE_TOKEN) as client:
            info = _wait_for_runner(client, process, deadline=120.0)
            assert info["service"] == "steerlab-server"
            assert os.path.realpath(info["root"]) == os.path.realpath(
                workspace)
            assert isinstance(client.capabilities(), dict)

            # The token really is required: the same routes, no credential.
            with runner_api.RunnerClient(base_url=base_url,
                                         token=None) as anonymous:
                with pytest.raises(runner_api.RunnerHTTPError) as caught:
                    anonymous.info()
                assert caught.value.status == 401
                assert caught.value.code == "runnerUnauthorized"

            # Upload — a chunked request body through a real HTTP parser, with
            # the digest agreed across the socket.
            uploaded = client.upload_run_bundle(bundle)
            assert uploaded["sha256"] == uploaded["localSha256"]
            assert uploaded["sha256"] == runner_api.sha256_file(bundle)
            assert uploaded["executable"] is True

            # Submit — DRY RUN: this test is about the wire, not about compute.
            submission = client.submit_uploaded_bundle(
                remote_path=uploaded["path"],
                expected_sha256=uploaded["bundle"]["bundleSha256"],
                verb="verify", executor="local", dry_run=True)
            job_id = submission["jobId"]
            assert submission["dryRun"] is True

            record = client.job(job_id)
            assert record["status"] == "prepared"
            assert any(row["id"] == job_id for row in client.jobs())

            logs = client.job_logs(job_id)
            assert logs["status"] == "prepared"
            # `prepared` is terminal, so the SSE stream CLOSES and the follow
            # path terminates on its own rather than on a timeout.
            followed = client.job_logs(job_id, follow=True)
            assert followed["followed"] is True
            assert followed["lines"][-1] == "[prepared]"

            # Download the staged artifact back and verify it byte for byte —
            # the streaming, size-capped, hash-as-it-arrives path, over TCP.
            destination = str(tmp_path / "home" / "returned.tar.gz")
            got = client.download_bundle(
                remote_path=uploaded["path"],
                expected_sha256=uploaded["sha256"],
                destination=destination,
                temp_path=destination + ".partial")
            assert got["verified"] is True
            assert not os.path.exists(destination + ".partial")
            with open(destination, "rb") as a, open(bundle, "rb") as b:
                assert a.read() == b.read()
    finally:
        process.terminate()
        try:
            process.communicate(timeout=20)
        except subprocess.TimeoutExpired:      # pragma: no cover - stubborn
            process.kill()
            process.communicate(timeout=20)


# =============================================================================
# 6. The import graph
# =============================================================================


#: What must NOT be imported to reach a runner. Same list Phase 1b guards the
#: authoring verbs with.
HEAVY = ("torch", "transformers", "fastapi", "uvicorn", "peft", "sae_lens")

#: What the runner family MAY pull, beyond the package itself: httpx and its
#: REQUIRED closure. Every name here is already pinned in both committed
#: platform locks (see
#: `test_client_cli.py::test_the_new_client_dependency_was_already_in_the_locks`),
#: which is what made declaring httpx a client dependency free.
ALLOWED_TOP_LEVEL = {
    "steerlab_server",
    "httpx", "httpcore", "h11", "anyio", "certifi", "idna",
}

#: httpx imports its OPTIONAL command-line dependencies when they happen to be
#: installed — ``httpx/__init__.py`` is literally ``try: from ._main import
#: main / except ImportError:`` with a stub. Those are the ``httpx[cli]``
#: extra, which a bare client install does NOT resolve; they are present in
#: THIS venv only because the ENGINE stack drags them in (huggingface_hub and
#: uvicorn want click, typer wants rich, pytest wants pygments). Tolerated and
#: NAMED, so the guard still fails on a genuinely new dependency rather than
#: on somebody else's — and the claim that they are optional is not taken on
#: trust: `test_httpx_needs_none_of_its_optional_cli_dependencies` blocks them
#: and imports httpx anyway.
HTTPX_OPTIONAL_CLI = {"click", "rich", "pygments", "markdown_it", "mdurl",
                      "typing_extensions"}

_PROBE = """
import io, json, sys, contextlib
from steerlab_server import client_cli
with contextlib.redirect_stdout(io.StringIO()), \
        contextlib.redirect_stderr(io.StringIO()):
    # 127.0.0.1:1 refuses instantly: this exercises the whole verb — parsing,
    # the adapter's construction, a real request attempt, and the refusal
    # document — without needing a runner or a network.
    code = client_cli.main(["runner", "capabilities", "--runner",
                            "http://127.0.0.1:1", "--timeout", "2"])
sys.__stderr__.write(json.dumps({
    "code": code,
    "heavy": sorted(m for m in %(heavy)r if m in sys.modules),
    "topLevel": sorted({m.split(".")[0] for m in sys.modules
                        if not m.startswith("_")}
                       - set(sys.stdlib_module_names)),
}))
"""


def test_the_runner_family_stays_within_the_clients_light_import_set(tmp_path):
    """CONTRACT (the light install, extended to Phase 2).

    Out of process, for the reason Phase 1b gives: by the time this module
    runs, half the suite has imported torch, and an in-process assertion about
    ``sys.modules`` would pass or fail on test ORDER.

    If this fails, the repair is to move the offending import INSIDE the verb
    that needs it — never to restructure the engine module it reached.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE % {"heavy": list(HEAVY)}],
        cwd=str(tmp_path), text=True, capture_output=True, check=False,
        env={**os.environ, "PYTHONPATH": SERVER_DIR})
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stderr[proc.stderr.rindex("{"):])

    assert report["heavy"] == [], (
        "`steerlab runner capabilities` pulled: " + ", ".join(report["heavy"]))
    # An unreachable runner is an operational failure, not a refusal: nothing
    # declined anything, the client could not talk to it.
    assert report["code"] == 70, report
    unexpected = sorted(set(report["topLevel"])
                        - ALLOWED_TOP_LEVEL - HTTPX_OPTIONAL_CLI)
    assert unexpected == [], (
        "the runner family pulled third-party modules outside the client's "
        "declared set: " + ", ".join(unexpected))
    assert "httpx" in report["topLevel"], \
        "the probe never reached the adapter — it proves nothing"


def test_httpx_needs_none_of_its_optional_cli_dependencies(tmp_path):
    """The claim :data:`HTTPX_OPTIONAL_CLI` rests on, checked rather than
    asserted.

    ``import httpx`` in this venv also pulls click/rich/pygments, which would
    look like the client having quietly grown four dependencies. It has not:
    httpx reaches for them only if they are already there. Blocking them and
    importing httpx anyway is the difference between believing that and
    knowing it — and it is what a bare ``pip install steerlab-server`` really
    looks like.
    """
    probe = """
import json, sys

class Blocker:
    BLOCKED = %(blocked)r
    def find_module(self, name, path=None):
        return self if name.split(".")[0] in self.BLOCKED else None
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCKED:
            raise ImportError(f"blocked: {name}")
        return None

sys.meta_path.insert(0, Blocker())
import httpx
from steerlab_server.client import runner
client = runner.RunnerClient(base_url="http://127.0.0.1:1")
sys.__stderr__.write(json.dumps({
    "version": httpx.__version__,
    "leaked": sorted(m for m in %(blocked)r if m in sys.modules),
}))
"""
    blocked = sorted(HTTPX_OPTIONAL_CLI - {"typing_extensions"})
    proc = subprocess.run(
        [sys.executable, "-c", probe % {"blocked": blocked}],
        cwd=str(tmp_path), text=True, capture_output=True, check=False,
        env={**os.environ, "PYTHONPATH": SERVER_DIR})
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stderr[proc.stderr.rindex("{"):])
    assert report["leaked"] == [], report
    assert report["version"]


# =============================================================================
# 9. The download COMMITS without ever overwriting (external review 2026-08-24)
# =============================================================================


def test_a_destination_that_appears_mid_download_is_refused_not_overwritten(
        service):
    """The existence check runs before a transfer that can take minutes.

    Between that check and the move there is a window, and ``os.replace``
    would have walked straight through it — silently replacing evidence
    another process had just landed. The commit primitive cannot overwrite, so
    the race ends in the SAME refusal the up-front check gives.
    """
    _job_id, meta = _evidence_bearing_job(service)
    with open(meta["bundlePath"], "rb") as handle:
        body = handle.read()
    destination = os.path.join(str(service["tmp"]), "home", "raced.tar.gz")
    intruder = b"somebody else's evidence"

    def handler(request: httpx.Request) -> httpx.Response:
        # The window itself: another client lands its own file at the
        # destination while this download is in flight.
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as handle:
            handle.write(intruder)
        return httpx.Response(200, content=body)

    with runner_api.RunnerClient(base_url="http://runner.invalid", token=None,
                                 http_client=_mock_client(handler)) as client:
        with pytest.raises(runner_api.RunnerRefusal) as caught:
            client.download_bundle(remote_path=meta["bundlePath"],
                                   expected_sha256=meta["bundleSha256"],
                                   destination=destination)

    assert caught.value.code == "destinationExists"
    assert "appeared while this download was in flight" in caught.value.reason
    with open(destination, "rb") as handle:
        assert handle.read() == intruder, \
            "the download overwrote a file that appeared after its check"
    assert os.listdir(os.path.dirname(destination)) == ["raced.tar.gz"], \
        "the refusal left staging debris beside the destination"


def test_the_staging_file_is_unique_so_concurrent_downloads_cannot_collide(
        adapter, service):
    """Two downloads aimed at one destination used to write the SAME staging
    path — ``<destination>.partial`` — so each verified bytes the other was
    still writing. Simulated by pre-creating that predictable name: the
    adapter must not touch it, must not read it, and must not delete it.
    """
    _job_id, meta = _evidence_bearing_job(service)
    destination = os.path.join(str(service["tmp"]), "home", "shared.tar.gz")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    rival = destination + ".partial"
    rival_bytes = b"the other download's half-written archive"
    with open(rival, "wb") as handle:
        handle.write(rival_bytes)

    downloaded = adapter.download_bundle(
        remote_path=meta["bundlePath"],
        expected_sha256=meta["bundleSha256"], destination=destination)

    assert downloaded["verified"] is True
    with open(rival, "rb") as handle:
        assert handle.read() == rival_bytes, \
            "the adapter reused the predictable staging name"
    with open(destination, "rb") as got, open(meta["bundlePath"], "rb") as want:
        assert got.read() == want.read()
    # …and its OWN staging file is gone.
    debris = [name for name in os.listdir(os.path.dirname(destination))
              if name.startswith(".steerlab-download-")]
    assert debris == []


def test_an_explicit_temp_path_is_still_honoured(adapter, service):
    """``--temp`` exists for a destination on a small volume, and an operator
    who names a staging path owns it. The default merely stops being a
    guessable one."""
    _job_id, meta = _evidence_bearing_job(service)
    destination = os.path.join(str(service["tmp"]), "home", "explicit.tar.gz")
    temp_path = os.path.join(str(service["tmp"]), "elsewhere", "stage.part")

    downloaded = adapter.download_bundle(
        remote_path=meta["bundlePath"],
        expected_sha256=meta["bundleSha256"], destination=destination,
        temp_path=temp_path)

    assert downloaded["verified"] is True
    assert not os.path.exists(temp_path)
    assert os.path.isfile(destination)


def test_an_explicit_temp_path_that_exists_is_refused_not_truncated(
        adapter, service):
    """Naming a staging path says WHERE the bytes may wait, not that whatever
    is already there may be destroyed (third-round review, 2026-08-24).

    It was opened ``"wb"``, which truncates. The usual reason to pass --temp is
    "this volume has room the destination's has not" — which is exactly the
    kind of volume that already holds somebody's large file. Exclusive create,
    typed refusal, and the file is untouched.
    """
    _job_id, meta = _evidence_bearing_job(service)
    destination = os.path.join(str(service["tmp"]), "home", "notrunc.tar.gz")
    temp_path = os.path.join(str(service["tmp"]), "elsewhere", "occupied.part")
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    standing = b"somebody else's forty gigabytes, in miniature\n"
    with open(temp_path, "wb") as handle:
        handle.write(standing)

    with pytest.raises(runner_api.RunnerRefusal) as caught:
        adapter.download_bundle(remote_path=meta["bundlePath"],
                                expected_sha256=meta["bundleSha256"],
                                destination=destination, temp_path=temp_path)

    assert caught.value.code == "tempPathExists"
    assert caught.value.state == "refused"
    assert temp_path in caught.value.reason
    with open(temp_path, "rb") as handle:
        assert handle.read() == standing, "--temp truncated a standing file"
    assert not os.path.exists(destination), \
        "the destination was created despite the refusal"


def _swap_staging_for_symlink(adapter, monkeypatch, *, victim, locate):
    """Reproduce the reviewer's window: the staging name is RESERVED, the
    bytes have not been written yet, and something replaces the name with a
    symlink to ``victim``.

    ``self._client.stream(...)`` is the first thing the download does after the
    reservation and before the first write, so wrapping it lands exactly in the
    gap the old ``open(temp_path, "wb")`` reopen was exposed to. ``locate``
    returns the reserved staging path (the caller knows it for ``--temp``, and
    has to go and find the minted one otherwise).
    """
    real_stream = adapter._client.stream
    swapped = {}

    def _stream(*args, **kwargs):
        path = locate()
        if path and not swapped:
            os.unlink(path)
            os.symlink(victim, path)
            swapped["path"] = path
        return real_stream(*args, **kwargs)

    monkeypatch.setattr(adapter._client, "stream", _stream)
    return swapped


def test_a_swapped_staging_name_cannot_overwrite_what_it_points_at(
        adapter, service, monkeypatch):
    """The staging file is reserved by descriptor and written by descriptor —
    the name is never opened twice (external review, 2026-08-26).

    It used to be: ``mkstemp`` (or ``"xb"``) reserved the name, the descriptor
    was thrown away, and ``open(temp_path, "wb")`` opened it again. Between the
    two the name can become a symlink, and ``"wb"`` follows it and truncates
    the target — the reviewer overwrote an unrelated file and still had
    ``verified: True`` handed back. A descriptor cannot be redirected, so the
    victim keeps its bytes; the commit then notices the name no longer means
    the verified inode and publishes nothing.
    """
    _job_id, meta = _evidence_bearing_job(service)
    destination = os.path.join(str(service["tmp"]), "home", "swapped.tar.gz")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    victim = os.path.join(str(service["tmp"]), "home", "unrelated.txt")
    victim_bytes = b"an unrelated file that has nothing to do with any download"
    with open(victim, "wb") as handle:
        handle.write(victim_bytes)

    def _minted():
        for name in sorted(os.listdir(os.path.dirname(destination))):
            if name.startswith(".steerlab-download-"):
                return os.path.join(os.path.dirname(destination), name)
        return None

    swapped = _swap_staging_for_symlink(adapter, monkeypatch, victim=victim,
                                        locate=_minted)

    with pytest.raises(runner_api.RunnerRefusal) as caught:
        adapter.download_bundle(remote_path=meta["bundlePath"],
                                expected_sha256=meta["bundleSha256"],
                                destination=destination)

    assert swapped, "the repro never reached the staging window"
    assert caught.value.code == "stagingPathHijacked"
    assert caught.value.state == "refused"
    with open(victim, "rb") as handle:
        assert handle.read() == victim_bytes, \
            "the download wrote through the staging NAME and hit its target"
    assert not os.path.exists(destination), \
        "an unverified inode was published under the destination's name"
    # The swapped-in entry SURVIVES the cleanup (external review round 5).
    # It used to be deleted: cleanup unlinked the staging name on the premise
    # that the entry was the one this client reserved — the very premise this
    # refusal exists to say is false. What sits there is somebody else's
    # symlink, and removing it is not this client's business.
    left = [name for name in os.listdir(os.path.dirname(destination))
            if name.startswith(".steerlab-download-")]
    assert left == [os.path.basename(swapped["path"])], \
        "cleanup removed an entry this download did not create"
    assert os.path.islink(swapped["path"])
    assert os.readlink(swapped["path"]) == victim
    # …and the refusal SAYS so, rather than leaving the person to discover it.
    assert caught.value.detail["stagingPathForeignEntry"] == swapped["path"]
    assert caught.value.detail["stagingPathNote"] == \
        runner_api.FOREIGN_STAGING_NOTE


def test_a_swapped_explicit_temp_path_cannot_overwrite_its_target(
        adapter, service, monkeypatch):
    """The same window on the ``--temp`` branch, which reserved the name with
    ``"xb"`` and then closed the handle — the reservation was sound and the
    reopen threw it away."""
    _job_id, meta = _evidence_bearing_job(service)
    destination = os.path.join(str(service["tmp"]), "home", "swapped2.tar.gz")
    temp_path = os.path.join(str(service["tmp"]), "elsewhere", "swap.part")
    victim = os.path.join(str(service["tmp"]), "elsewhere", "payroll.csv")
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    victim_bytes = b"row,one\nrow,two\n"
    with open(victim, "wb") as handle:
        handle.write(victim_bytes)

    swapped = _swap_staging_for_symlink(adapter, monkeypatch, victim=victim,
                                        locate=lambda: temp_path)

    with pytest.raises(runner_api.RunnerRefusal) as caught:
        adapter.download_bundle(remote_path=meta["bundlePath"],
                                expected_sha256=meta["bundleSha256"],
                                destination=destination, temp_path=temp_path)

    assert swapped, "the repro never reached the staging window"
    assert caught.value.code == "stagingPathHijacked"
    with open(victim, "rb") as handle:
        assert handle.read() == victim_bytes, "--temp truncated its symlink's "\
            "target"
    assert not os.path.exists(destination)
    # The symlink somebody put on the staging name is still there: cleanup
    # unlinks the name only while it still refers to the inode this client
    # reserved, and here it plainly does not.
    assert os.path.islink(temp_path)
    assert os.readlink(temp_path) == victim


def test_a_foreign_file_moved_onto_the_staging_path_is_never_deleted(
        adapter, service, monkeypatch):
    """THE round-5 finding, in the reviewer's own shape: not a symlink but a
    plain REGULAR file moved onto the staging path mid-download.

    The refusal fired correctly and the cleanup then deleted the replacement.
    If that name was the file's only link, a client whose entire purpose is to
    refuse to overwrite evidence had just destroyed a file it was never asked
    to touch — quieter than an overwrite, and worse.

    Cleanup now proves the name still refers to the inode it reserved
    (``lstat`` of the name against ``fstat`` of its own descriptor) and, when
    it does not, leaves the name completely alone.
    """
    _job_id, meta = _evidence_bearing_job(service)
    destination = os.path.join(str(service["tmp"]), "home", "foreign.tar.gz")
    temp_path = os.path.join(str(service["tmp"]), "elsewhere", "foreign.part")
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    victim_bytes = b"the only copy of something that matters\n"
    victim = os.path.join(str(service["tmp"]), "elsewhere", "only-copy.txt")
    with open(victim, "wb") as handle:
        handle.write(victim_bytes)

    # Move the file ONTO the staging name — one link, and the staging path is
    # now it. Same window as the symlink repro: after the reservation, before
    # the first byte.
    real_stream = adapter._client.stream
    moved = {}

    def _stream(*args, **kwargs):
        if not moved:
            os.unlink(temp_path)
            os.rename(victim, temp_path)
            moved["done"] = True
        return real_stream(*args, **kwargs)

    monkeypatch.setattr(adapter._client, "stream", _stream)

    with pytest.raises(runner_api.RunnerRefusal) as caught:
        adapter.download_bundle(remote_path=meta["bundlePath"],
                                expected_sha256=meta["bundleSha256"],
                                destination=destination, temp_path=temp_path)

    assert moved, "the repro never reached the staging window"
    assert caught.value.code == "stagingPathHijacked"
    assert caught.value.state == "refused"
    # The directory ENTRY survives…
    assert os.path.exists(temp_path), \
        "cleanup deleted a file this download did not create"
    # …and so do the bytes under it, exactly.
    with open(temp_path, "rb") as handle:
        assert handle.read() == victim_bytes
    assert not os.path.exists(destination), \
        "an unverified inode was published under the destination's name"
    # The refusal names what it found and says it did not touch it.
    assert caught.value.detail["stagingPathForeignEntry"] == temp_path
    assert caught.value.detail["stagingPathNote"] == \
        runner_api.FOREIGN_STAGING_NOTE


def test_bytes_written_into_the_staging_inode_are_caught_before_the_commit(
        adapter, service, monkeypatch):
    """A descriptor stops the staging NAME from being redirected; it does not
    stop somebody who already holds the staging file from writing into it.

    So the digest that decides is taken from the inode, through the download's
    own descriptor, immediately before the commit — not only from the bytes as
    they arrived. Here a second writer scribbles over the staged archive after
    the last chunk lands and before the commit.
    """
    _job_id, meta = _evidence_bearing_job(service)
    destination = os.path.join(str(service["tmp"]), "home", "scribbled.tar.gz")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temp_path = os.path.join(str(service["tmp"]), "elsewhere", "held.part")
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    real_stream = adapter._client.stream

    class _ScribbleAtEnd:
        """The real streamed response, with one extra act as it closes — which
        is after the download's last write and flush, and before it verifies
        what is on the disk."""

        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, *exc_info):
            with open(temp_path, "r+b") as sneak:
                sneak.seek(0)
                sneak.write(b"not the archive that was verified")
            return self._inner.__exit__(*exc_info)

    monkeypatch.setattr(adapter._client, "stream",
                        lambda *a, **k: _ScribbleAtEnd(real_stream(*a, **k)))

    with pytest.raises(runner_api.RunnerRefusal) as caught:
        adapter.download_bundle(remote_path=meta["bundlePath"],
                                expected_sha256=meta["bundleSha256"],
                                destination=destination, temp_path=temp_path)

    assert caught.value.code == "stagedBytesChanged"
    assert caught.value.state == "refused"
    assert caught.value.detail["expectedSha256"] == meta["bundleSha256"]
    assert caught.value.detail["stagedSha256"] != meta["bundleSha256"]
    assert not os.path.exists(destination)
    assert not os.path.exists(temp_path)


def test_the_linkless_commit_leaves_no_short_file_behind(adapter, service,
                                                          monkeypatch):
    """The O_EXCL fallback (filesystems without hardlinks) RESERVES the
    destination name before it has any bytes to put there. A copy that dies
    half way must take the reservation back out rather than leave a short file
    wearing the destination's name — every exit from that block, including one
    that never got a file object at all."""
    _job_id, meta = _evidence_bearing_job(service)
    destination = os.path.join(str(service["tmp"]), "home", "linkless.tar.gz")

    def _no_links(src, dst, *args, **kwargs):
        raise OSError(errno.EPERM, "hardlinks unsupported here")

    monkeypatch.setattr(runner_api.os, "link", _no_links)

    def _dies_half_way(source, target, *args, **kwargs):
        target.write(b"half of an archive")
        raise OSError("the volume went away mid-copy")

    monkeypatch.setattr(runner_api.shutil, "copyfileobj", _dies_half_way)
    with pytest.raises(OSError, match="went away mid-copy"):
        adapter.download_bundle(remote_path=meta["bundlePath"],
                                expected_sha256=meta["bundleSha256"],
                                destination=destination)
    assert not os.path.exists(destination), \
        "a short file kept the destination's name after a failed copy"

    # The same fallback, working: it commits the whole archive and cannot
    # overwrite — the reservation is the thing that makes it safe.
    monkeypatch.undo()
    monkeypatch.setattr(runner_api.os, "link", _no_links)
    adapter.download_bundle(remote_path=meta["bundlePath"],
                            expected_sha256=meta["bundleSha256"],
                            destination=destination)
    with open(destination, "rb") as got, open(meta["bundlePath"], "rb") as want:
        assert got.read() == want.read()


def test_a_commit_that_cannot_drop_its_staging_name_takes_the_destination_back(
        adapter, service, monkeypatch):
    """Review round 10, finding 8 — the invariant held in all three mirrors of
    ``_commit_no_replace`` (``steering.pole_mirror``,
    ``experiment.bundles``, here).

    THIS mirror was already safe, and deliberately so: its last step goes
    through ``_unlink_if_reserved``, which reports a stranger on the staging
    name by RETURNING False rather than raising. The guard is written anyway
    so all three read alike, and this test pins the OUTCOME the other two now
    share — a commit that cannot finish does not leave its destination
    standing.
    """
    _job_id, meta = _evidence_bearing_job(service)
    destination = os.path.join(str(service["tmp"]), "home", "guarded.tar.gz")

    real_unlink = runner_api._unlink_if_reserved
    fired = {"guard": False, "interrupt": False}

    def failing_on_the_staging_drop(path, reserved):
        # The staging drop is the call made while the DESTINATION is already
        # landed. Anything before it is an ordinary cleanup and must work.
        if path != destination and os.path.exists(destination):
            fired["interrupt"] = True
            raise KeyboardInterrupt("interrupted between commit and cleanup")
        if path == destination and fired["interrupt"]:
            fired["guard"] = True
        return real_unlink(path, reserved)

    monkeypatch.setattr(runner_api, "_unlink_if_reserved",
                        failing_on_the_staging_drop)
    with pytest.raises(KeyboardInterrupt):
        adapter.download_bundle(remote_path=meta["bundlePath"],
                                expected_sha256=meta["bundleSha256"],
                                destination=destination)
    monkeypatch.undo()
    assert fired["interrupt"], "the staging drop was never reached"
    assert fired["guard"], \
        "the both-or-neither guard never ran on the landed destination"
    assert not os.path.exists(destination), \
        "a commit that could not finish left its destination standing"

    # …and the ordinary path still publishes.
    adapter.download_bundle(remote_path=meta["bundlePath"],
                            expected_sha256=meta["bundleSha256"],
                            destination=destination)
    with open(destination, "rb") as got, open(meta["bundlePath"], "rb") as want:
        assert got.read() == want.read()


# =============================================================================
# 10. The runner URL is NORMALIZED at the boundary
# =============================================================================


def test_a_runner_url_with_userinfo_is_refused_and_never_echoed():
    """A credential in a locator is a credential in the provenance record.

    ``remote-execution.json`` is written once and read forever; the envelope
    and every diagnostic reproduce the same string. So userinfo is REFUSED —
    not stripped, because a silently unauthenticated request fails in a way
    nobody can trace back to the password they thought they were using — and
    the refusal itself redacts what it is refusing.
    """
    secret = "hunter2-DO-NOT-LEAK"
    with pytest.raises(runner_api.RunnerRefusal) as caught:
        runner_api.normalize_base_url(
            f"https://someone:{secret}@runner.example.edu:8443/")
    exc = caught.value
    assert exc.code == runner_api.URL_REFUSED_CODE
    assert exc.state == "refused"
    for text in (exc.reason, exc.repair_action, json.dumps(exc.detail)):
        assert secret not in text
        assert "someone" not in text
    assert "<credentials redacted>" in exc.reason
    assert "--token-file" in exc.repair_action

    # …and the constructor is the boundary, so no client can hold one.
    with pytest.raises(runner_api.RunnerRefusal):
        runner_api.RunnerClient(base_url=f"http://u:{secret}@host:8080")


def test_a_runner_url_must_name_http_or_https_and_a_host():
    for locator in ("runner.example.edu:8080", "ssh://runner.example.edu",
                    "file:///tmp/runner", "http://", ""):
        with pytest.raises(runner_api.RunnerRefusal) as caught:
            runner_api.normalize_base_url(locator)
        assert caught.value.code == runner_api.URL_REFUSED_CODE


def test_a_query_string_or_fragment_on_a_runner_url_is_refused():
    """Both are silently lost the moment a route path is appended, so a caller
    who put a token in one would never learn it was ignored."""
    for locator in ("http://runner.example.edu/?token=abc",
                    "http://runner.example.edu/#frag"):
        with pytest.raises(runner_api.RunnerRefusal) as caught:
            runner_api.normalize_base_url(locator)
        assert caught.value.code == runner_api.URL_REFUSED_CODE


@pytest.mark.parametrize("shape", ["query", "fragment"])
def test_a_query_or_fragment_refusal_never_echoes_what_it_refused(shape):
    """The userinfo rule, applied to the other two places a secret lands.

    ``?token=…`` and ``#token=…`` are the commonest way a credential ends up
    in a locator by mistake — and the refusal used to interpolate the very
    component it was warning about, putting the token into the envelope, the
    diagnostic and the shell history while telling the caller not to do that
    (third-round review, 2026-08-24). The message names the CATEGORY now.
    """
    secret = "hunter2-DO-NOT-LEAK-4c19ab"
    separator = "?" if shape == "query" else "#"
    locator = f"http://runner.example.edu:8443/{separator}token={secret}"

    with pytest.raises(runner_api.RunnerRefusal) as caught:
        runner_api.normalize_base_url(locator)
    exc = caught.value
    assert exc.code == runner_api.URL_REFUSED_CODE
    assert exc.state == "refused"
    for text in (exc.reason, exc.repair_action, json.dumps(exc.detail),
                 str(exc), repr(exc)):
        assert secret not in text, f"the {shape} escaped into: {text[:200]}"
        assert "token=" not in text.replace("$STEERLAB_RUNNER_TOKEN", ""), \
            f"the {shape} escaped into: {text[:200]}"
    # It still says WHAT was wrong — a refusal that named nothing would be
    # unrepairable.
    assert shape in exc.reason

    # …and the constructor is the boundary, so no client can hold one.
    with pytest.raises(runner_api.RunnerRefusal):
        runner_api.RunnerClient(base_url=locator)


@pytest.mark.parametrize("shape", ["query", "fragment"])
def test_a_secret_in_a_query_or_fragment_reaches_no_client_surface(
        service, capsys, shape):
    """The same nonappearance proof the userinfo case gets, driven through the
    whole CLI: envelope, human mode, stderr, and the composite verb whose
    provenance record is the durable surface."""
    secret = "hunter2-DO-NOT-LEAK-71e0f5"
    separator = "?" if shape == "query" else "#"
    locator = f"http://testserver/{separator}token={secret}"
    transcript: list[str] = []

    code, document, err = _envelope(
        ["runner", "capabilities", "--runner", locator], capsys)
    assert code == 65, document
    assert document["error"]["code"] == runner_api.URL_REFUSED_CODE
    transcript.extend([json.dumps(document), err])

    code, out, err = _cli(["runner", "capabilities", "--runner", locator],
                          capsys)
    assert code == 65
    transcript.extend([out, err])

    code, document, err = _envelope(
        ["run", STUDY_NAME, "--runner", locator,
         "--root", service["source"]], capsys)
    assert code == 65, document
    assert document["error"]["code"] == runner_api.URL_REFUSED_CODE, \
        "the composite must refuse the locator BEFORE it reads the study"
    transcript.extend([json.dumps(document), err])

    for text in transcript:
        assert secret not in text, (
            f"a {shape}-embedded credential escaped into client output: "
            + text[:400])
    assert sum(len(t) for t in transcript) > 200


@pytest.mark.parametrize("locator", [
    "http://runner.example.edu:notaport",       # not a number at all
    "http://runner.example.edu:99999",          # a number, out of range
    "http://runner.example.edu:8080abc/prefix",
])
def test_a_malformed_port_is_a_typed_refusal_not_a_crash(locator):
    """``parts.port`` raises only when the attribute is READ, and every branch
    of the normalizer reads it — ``_redacted`` included. An uncaught
    ``ValueError`` escaped this boundary as a generic malformed-invocation 64
    with the raw locator in its traceback (third-round review, 2026-08-24). It
    is a locator problem like any other: typed, 65, and naming the category
    rather than the text."""
    with pytest.raises(runner_api.RunnerRefusal) as caught:
        runner_api.normalize_base_url(locator)
    exc = caught.value
    assert exc.code == runner_api.URL_REFUSED_CODE
    assert exc.state == "refused"
    assert "port" in exc.reason
    for text in (exc.reason, exc.repair_action):
        assert "runner.example.edu" not in text, \
            "the malformed authority was echoed back"
    # The constructor is the boundary here too.
    with pytest.raises(runner_api.RunnerRefusal):
        runner_api.RunnerClient(base_url=locator)


def test_the_normalized_form_is_what_the_client_reports():
    """One spelling reaches every surface: lowercased scheme and host, port
    kept, proxy prefix kept, trailing slash gone."""
    assert runner_api.normalize_base_url("HTTP://Runner.Example.EDU:8080/") \
        == "http://runner.example.edu:8080"
    assert runner_api.normalize_base_url("https://host/steerlab/") \
        == "https://host/steerlab"
    assert runner_api.normalize_base_url("http://127.0.0.1:8080") \
        == "http://127.0.0.1:8080"
    # Idempotent, so normalizing at more than one boundary is safe.
    once = runner_api.normalize_base_url("HTTP://Host:8080/prefix/")
    assert runner_api.normalize_base_url(once) == once

    client = runner_api.RunnerClient(base_url="HTTP://TestServer/",
                                     http_client=_mock_client(
                                         lambda request: httpx.Response(200)))
    try:
        assert client.base_url == "http://testserver"
        assert "http://testserver" in repr(client)
    finally:
        client.close()
