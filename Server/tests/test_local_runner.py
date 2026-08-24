"""Localhost as a MANAGED runner (Phase 3 of the portability program).

Phase 1b's client authors a workspace; Phase 2 taught it to hand a hash-pinned
bundle to a runner and bring the evidence home. Phase 3 is the runner a person
has on the machine in front of them — ``steerlab runner serve`` — and the
whole point of this file is that it is **not a shortcut**.

The ruling, because every assertion below is a consequence of it: **the bundle
protocol binds BATCH execution.** Local and remote execution use the IDENTICAL
round trip — upload → submit → evidence → import, each hop hash-pinned — and
there is no privileged localhost path into the client's workspace. (The app's
local WORKBENCH, which serves a live workspace interactively, is the other
service role and is untouched by this phase.) A managed runner therefore gets
a RUNNER-OWNED root, and the three assertions that make that a fact rather
than an intention are:

(a) the runner root holds the staging and the cache; the client workspace
    gains only what ``bundle import`` put there;
(b) **deleting the entire runner root after the import removes nothing the
    client workspace holds** — the imported run's bytes are re-verified with
    the runner root gone;
(c) the engine's own ``/api/info`` root IS the runner root — asked over the
    wire, not assumed.

What is real here and what is not, stated rather than implied: the runner is
the genuine ``steerlab-server serve``, launched exactly as the verb launches
it (subprocess, loopback, ephemeral port, token mode, a minted 0600 token
file); the study is authored by the real client verbs into a real workspace;
the upload, the submit, the job, the download and the import are the real
routes over a real socket. The one thing skipped is the GPU compute that would
have written a run directory — the only bundled verb needing no model is
``verify``, and ``verify`` writes no run directory, so it packages no evidence
(``bundles._execute_run_bundle_inner``). The evidence half is therefore driven
over a job seeded into the runner's OWN durable job store before it boots,
carrying an archive ``bundles.package_evidence`` really wrote into the
runner's runs root — the same allowance ``test_client_runner.py``'s
``_evidence_bearing_job`` makes, one process further out. A real ``verify``
submission runs beside it, so the submit path is exercised for real too.

Fixtures are neutral throughout: one concept named ``signal`` with two
one-word stimuli.
"""

import hashlib
import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("httpx")

from steerlab_server import client_cli
from steerlab_server.experiment import bundles, paths

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STUDY_NAME = "local-runner-fixture"
CONCEPT = "signal"

#: Variables that would let this machine's shell reach through the verb under
#: test and put the runner's artifacts somewhere else. Cleared for every
#: subprocess: the two-roles rule is about what the VERB decides, and a test
#: that inherited a developer's ``STEERLAB_ROOT`` would be measuring the
#: developer.
_AMBIENT = ("STEERLAB_ROOT", "STEERLAB_WORKSPACE", "STEERLAB_RUN_ROOT",
            "STEERLAB_METADATA_ROOT", "STEERLAB_JOBS_DB",
            "STEERLAB_AUTH_TOKEN", "STEERLAB_AUTH_TOKEN_FILE",
            "STEERLAB_RUNNER_TOKEN", "STEERLAB_AUTH_MODE",
            "STEERLAB_EXECUTOR", "STEERLAB_DEV_OPEN_LOOPBACK")


# =============================================================================
# fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _isolated_environment():
    """Every test starts from a SteerLab-free environment and leaves one.

    Both halves matter. Starting clean means the machine's own shell cannot
    make an assertion pass for the wrong reason; ending clean matters because
    ``client_cli.resolve_workspace`` EXPORTS ``STEERLAB_ROOT`` as a deliberate
    consequence of naming a workspace, and an in-process client call would
    otherwise hand the rest of the suite a root it never asked for.
    """
    saved = {name: os.environ.get(name) for name in _AMBIENT}
    for name in _AMBIENT:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _clean_env(**overrides) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _AMBIENT}
    env["PYTHONPATH"] = SERVER_DIR
    # The engine loads no model here; an accidental Hub reach would only add
    # latency and a network dependency to a test about file custody.
    env["HF_HUB_OFFLINE"] = "1"
    env.update(overrides)
    return env


def _run_client(argv, capsys=None):
    """Drive the client IN PROCESS and return ``(code, document)``.

    ``--json`` throughout: this file reads documents, not columns. The client
    is the same object either way — ``main`` is what the console script calls.
    """
    code = client_cli.main([*argv, "--json"])
    if capsys is None:
        return code, None
    text = capsys.readouterr().out
    assert text.count("\n}") == 1, f"more than one document on stdout: {text}"
    return code, json.loads(text)


def _author_study(workspace: str, capsys) -> dict:
    """Author and freeze a small study with the LIGHT client verbs, then
    package it. Returns the ``bundle package`` result."""
    directory = os.path.join(workspace, "prompts", "concepts", CONCEPT)
    os.makedirs(directory, exist_ok=True)
    for filename, text in (("positive.jsonl", '{"text": "on"}\n'),
                           ("negative.jsonl", '{"text": "off"}\n')):
        with open(os.path.join(directory, filename), "w",
                  encoding="utf-8") as handle:
            handle.write(text)

    root = ["--root", workspace]
    for argv in (
            [*root, "experiment", "create", STUDY_NAME, "--model", "org/tiny",
             "--revision", "0" * 40],
            [*root, "experiment", "attach", STUDY_NAME, CONCEPT],
            [*root, "experiment", "declare-condition", STUDY_NAME, "baseline",
             "--baseline", "--alpha-units", "norm"],
            [*root, "experiment", "verify", STUDY_NAME],
            # `--force` skips the EVIDENCE gates: there is no validate run in a
            # client-only workspace and producing one needs a model. The same
            # allowance `test_client_cli.py`'s freeze test makes.
            [*root, "experiment", "freeze", STUDY_NAME, "--force"]):
        code, document = _run_client(argv, capsys)
        assert code == 0, document
    code, document = _run_client([*root, "bundle", "package", STUDY_NAME],
                                 capsys)
    assert code == 0, document
    return document["result"]


def _seed_evidence_job(runner_root: str, monkeypatch) -> tuple:
    """A completed run and its evidence bundle, in the RUNNER's own root.

    Seeded BEFORE the runner boots, because ``JobManager`` reads its durable
    store once at construction — which is exactly what makes this honest: the
    job the client asks about over HTTP is a row in the runner's real
    database, the archive is one ``bundles.package_evidence`` really wrote,
    and the digest the client verifies is the one that archive really has.
    Only the compute that would have produced the run directory is skipped.

    Returns ``(job_id, evidence_meta)``.
    """
    from steerlab_server.api.jobs import DurableJobStore, JobManager

    metadata = os.path.join(runner_root, ".steerlab")
    runs = os.path.join(runner_root, "runs")
    os.makedirs(metadata, exist_ok=True)
    os.makedirs(runs, exist_ok=True)
    # The seeding helpers read the roots from the environment, exactly as the
    # runner will; scoped to this call so nothing leaks into the client's own
    # resolution afterwards.
    monkeypatch.setenv("STEERLAB_ROOT", runner_root)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", runs)
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", metadata)

    run_directory = paths.make_unique_run_directory(f"{STUDY_NAME}-run")
    with open(os.path.join(run_directory, "config.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"experiment": STUDY_NAME, "verb": "run"}, handle)
    with open(os.path.join(run_directory, "records.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"record": 1, "condition": "baseline"}\n')
    meta = bundles.package_evidence(run_directory)

    manager = JobManager(
        DurableJobStore(path=os.path.join(metadata, "jobs.sqlite")),
        # No capability snapshot: it probes the machine's accelerators, which
        # is torch's job and none of this test's business.
        capability_provider=lambda: {}, sweep_orphans=False)
    job = manager.record_external(
        "study-submit-bundle", status="succeeded", executor="local",
        result={"experiment": STUDY_NAME, "verb": "run",
                "runDirectory": run_directory,
                # The shape a local `submit-bundle` really produces:
                # `submissions._read_child_record` folds the child's document
                # in under `runResult`.
                "runResult": {"evidenceBundle": meta}},
        log="seeded: the run this evidence came from")
    for name in ("STEERLAB_ROOT", "STEERLAB_RUN_ROOT", "STEERLAB_METADATA_ROOT"):
        monkeypatch.delenv(name, raising=False)
    return job.id, meta


def _drain(stream, sink) -> None:      # pragma: no cover - thread body
    """Read a pipe to exhaustion into ``sink``.

    Not optional: the managed runner forwards every engine log line to its own
    stderr, and an undrained pipe fills its buffer and wedges the process
    under test.
    """
    try:
        for line in stream:
            sink.append(line)
    except (ValueError, OSError):
        pass


class ManagedRunner:
    """The verb under test, running."""

    def __init__(self, process, out: list, err: list) -> None:
        self.process = process
        self.out = out
        self.err = err
        self.envelope: dict = {}

    @property
    def result(self) -> dict:
        return self.envelope.get("result", {})

    @property
    def url(self) -> str:
        return self.result["url"]

    @property
    def token_file(self) -> str:
        return self.result["tokenFile"]

    @property
    def stderr_text(self) -> str:
        return "".join(self.err)

    def stop(self) -> int:
        """SIGTERM the verb and wait. The engine must not outlive it."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:   # pragma: no cover - stubborn
                self.process.kill()
                self.process.wait(timeout=60)
        return int(self.process.returncode)


def _await_startup_envelope(runner: ManagedRunner, *, deadline: float) -> dict:
    """The ONE document ``runner serve`` writes to stdout before it serves.

    A deadline and a poll over the accumulated text, never a sleep: the
    document is pretty-printed across many lines, so "read a line" is not a
    read of the document, and the process is watched for early exit while
    waiting.
    """
    limit = time.monotonic() + deadline
    while time.monotonic() < limit:
        text = "".join(runner.out)
        if text.strip():
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        if runner.process.poll() is not None:
            raise AssertionError(
                f"runner serve exited {runner.process.returncode} before it "
                f"emitted a document:\n{runner.stderr_text}")
        time.sleep(0.05)
    raise AssertionError(
        f"no startup envelope within {deadline}s:\n{runner.stderr_text}")


def _await_info(url: str, runner: ManagedRunner, *, deadline: float) -> int:
    """Poll ``GET /api/info`` until the engine answers. Returns the status.

    ANY status counts — in token mode an unauthenticated ``/api/info`` is
    SUPPOSED to be refused, and 401 proves a SteerLab engine is on the socket
    just as well as 200 would. The subprocess is watched for early exit while
    polling; there is no sleep-and-hope anywhere in this file.
    """
    host, _, port = url.rsplit("/", 1)[-1].partition(":")
    limit = time.monotonic() + deadline
    last = "no answer"
    while time.monotonic() < limit:
        if runner.process.poll() is not None:
            raise AssertionError(
                f"the runner exited {runner.process.returncode} while it was "
                f"being polled:\n{runner.stderr_text}")
        connection = http.client.HTTPConnection(host, int(port), timeout=2.0)
        try:
            connection.request("GET", "/api/info")
            response = connection.getresponse()
            response.read()
            return int(response.status)
        except OSError as exc:
            last = str(exc)
        finally:
            connection.close()
        time.sleep(0.05)
    raise AssertionError(f"no engine on {url} within {deadline}s ({last})")


@pytest.fixture
def managed_runner(tmp_path, monkeypatch):
    """``steerlab runner serve``, launched the way an operator launches it.

    A subprocess of the real console-script module, an EPHEMERAL port (the
    default — the verb picks one and prints it), a runner root of its own, and
    an environment scrubbed of everything that could point it elsewhere.
    """
    runner_root = str(tmp_path / "runner-root")
    job_id, meta = _seed_evidence_job(runner_root, monkeypatch)

    process = subprocess.Popen(
        [sys.executable, "-m", "steerlab_server.client_cli",
         "runner", "serve", "--runner-root", runner_root, "--json"],
        env=_clean_env(), cwd=str(tmp_path), text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    runner = ManagedRunner(process, [], [])
    for stream, sink in ((process.stdout, runner.out),
                         (process.stderr, runner.err)):
        threading.Thread(target=_drain, args=(stream, sink),
                         daemon=True).start()
    try:
        runner.envelope = _await_startup_envelope(runner, deadline=180.0)
        assert _await_info(runner.url, runner, deadline=60.0) in (200, 401)
        yield {"runner": runner, "root": runner_root, "tmp": tmp_path,
               "evidenceJob": job_id, "evidenceMeta": meta}
    finally:
        runner.stop()


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# =============================================================================
# 1. The round trip, and the two roles
# =============================================================================


def test_the_whole_round_trip_runs_against_a_managed_local_runner(
        managed_runner, tmp_path, capsys):
    """CONTRACT: the local runner is reached the same way a remote one is.

    author → freeze → package → upload → submit → poll → evidence → import,
    every step a client verb against a runner that is a separate process with
    a root of its own. Then the three two-roles assertions, which are what
    this phase exists to make true rather than intended.
    """
    runner = managed_runner["runner"]
    runner_root = managed_runner["root"]
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace)
    connection = ["--runner", runner.url,
                  "--token-file", runner.result["tokenFile"]]

    # -- the startup envelope, before anything else uses it -----------------
    assert runner.envelope["state"] == "ready"
    assert runner.result["tokenFilePresent"] is True
    assert runner.result["runnerRoot"] == os.path.realpath(runner_root)
    assert runner.result["authMode"] == "token"
    # The token VALUE is never printed — not in the document, not on stderr.
    with open(runner.result["tokenFile"], encoding="utf-8") as handle:
        secret = handle.read().strip()
    assert secret and secret not in json.dumps(runner.envelope)
    assert secret not in runner.stderr_text
    assert oct(os.stat(runner.result["tokenFile"]).st_mode)[-3:] == "600"

    # -- (c) the engine's root IS the runner root, asked over the wire ------
    code, document = _run_client(["runner", "capabilities", *connection],
                                 capsys)
    assert code == 0, document
    assert document["result"]["root"] == os.path.realpath(runner_root)
    assert document["result"]["root"] != os.path.realpath(workspace)
    assert document["result"]["tokenPresent"] is True

    # -- author and package, in the CLIENT workspace ------------------------
    packaged = _author_study(workspace, capsys)
    bundle_path, bundle_digest = packaged["bundlePath"], packaged["bundleSha256"]
    assert os.path.realpath(bundle_path).startswith(os.path.realpath(workspace))
    runs_before = set(os.listdir(os.path.join(workspace, "runs")))

    # -- upload: the digest the client computed IS the one the runner read --
    code, document = _run_client(
        ["runner", "upload", bundle_path, *connection], capsys)
    assert code == 0, document
    assert document["result"]["sha256"] == _sha256(bundle_path)
    staged = document["result"]["runnerPath"]
    assert os.path.realpath(staged).startswith(os.path.realpath(runner_root)), \
        "the runner staged the upload outside its own root"

    # -- submit: a real, model-free execution on the far side ---------------
    code, document = _run_client(
        ["runner", "submit", *connection, "--bundle-path", staged,
         "--bundle-sha", document["result"]["sha256"], "--verb", "verify",
         "--executor", "local"], capsys)
    assert code == 0, document
    job_id = document["result"]["jobId"]
    assert document["result"]["experiment"] == STUDY_NAME

    # -- poll, on a deadline ------------------------------------------------
    from steerlab_server.api.jobs import TERMINAL
    limit = time.monotonic() + 180.0
    while True:
        code, document = _run_client(["runner", "jobs", job_id, *connection],
                                     capsys)
        assert code == 0, document
        status = document["result"]["job"]["status"]
        if status in TERMINAL:
            break
        assert time.monotonic() < limit, document
        time.sleep(0.05)
    assert status == "succeeded", document
    # The runner really executed it, into ITS root and nowhere else.
    assert os.path.isfile(os.path.join(runner_root, "experiments", STUDY_NAME,
                                       "experiment.json"))

    # -- evidence: downloaded and verified, deliberately NOT imported -------
    destination = str(tmp_path / "home" / "evidence.tar.gz")
    code, document = _run_client(
        ["runner", "evidence", managed_runner["evidenceJob"], "--out",
         destination, *connection], capsys)
    assert code == 0, document
    evidence = document["result"]
    assert evidence["verified"] is True
    assert evidence["imported"] is False
    assert evidence["sha256"] == managed_runner["evidenceMeta"]["bundleSha256"]
    assert _sha256(destination) == evidence["sha256"]
    assert not os.path.exists(destination + ".partial"), "staging debris"

    # -- import: the separate, named act, with the out-of-band pin ----------
    code, document = _run_client(
        ["--root", workspace, "bundle", "import", destination,
         "--sha256", evidence["sha256"]], capsys)
    assert code == 0, document
    assert document["result"]["extracted"]

    # -- (a) what each tree holds -------------------------------------------
    runs_after = set(os.listdir(os.path.join(workspace, "runs")))
    gained = runs_after - runs_before
    assert len(gained) == 1, (
        "the client workspace gained something other than the imported run: "
        f"{sorted(gained)}")
    imported_run = os.path.join(workspace, "runs", gained.pop())
    imported_file = os.path.join(imported_run, "records.jsonl")
    imported_digest = _sha256(imported_file)

    # The runner's tree carries the staging and the cache…
    runner_runs = os.listdir(os.path.join(runner_root, "runs"))
    assert any(name.endswith("uploaded-bundle") or "uploaded-bundle" in name
               for name in runner_runs), runner_runs
    assert any("submit-bundle" in name for name in runner_runs), runner_runs
    assert os.path.isfile(os.path.join(runner_root, ".steerlab",
                                       "jobs.sqlite"))
    # …and NONE of it is in the workspace. No staging directory, no job
    # database, no token: the client workspace never served anything.
    for name in os.listdir(os.path.join(workspace, "runs")):
        assert "uploaded-bundle" not in name and "submit-bundle" not in name, \
            f"engine staging landed in the client workspace: {name}"
    assert not os.path.exists(os.path.join(workspace, ".steerlab"))
    assert not os.path.exists(os.path.join(workspace,
                                           client_cli.RUNNER_TOKEN_FILENAME))

    # -- (b) the runner root is disposable ----------------------------------
    assert runner.stop() == 0
    shutil.rmtree(runner_root)
    assert not os.path.exists(runner_root)
    assert os.path.isfile(imported_file), (
        "deleting the runner root took the imported run with it — the "
        "workspace was holding a reference, not a copy")
    assert _sha256(imported_file) == imported_digest
    assert os.path.isfile(os.path.join(workspace, "experiments", STUDY_NAME,
                                       "experiment.json"))
    # The downloaded archive is the client's too, and still verifies.
    assert _sha256(destination) == evidence["sha256"]


def test_the_document_is_a_startup_envelope_and_then_a_stream(managed_runner):
    """CONTRACT: envelope-then-stream, the shape a long-running verb needs.

    Every other client verb writes ONE document when it finishes. This one
    finishes only when it is stopped, so it writes one document when the
    engine is READY and everything after that goes to stderr. An agent that
    parsed stdout would otherwise have to wait for the runner to die to learn
    the URL of the runner it just started.
    """
    runner = managed_runner["runner"]
    text = "".join(runner.out)

    # Exactly one JSON value on stdout, and it parses whole.
    assert json.loads(text) == runner.envelope
    assert text.count("\n}") == 1, text

    # The stream is on stderr, and it is the ENGINE's own voice: the artifact
    # root banner `steerlab-server serve` prints on every start.
    assert "artifact root: " in runner.stderr_text
    assert os.path.realpath(managed_runner["root"]) in runner.stderr_text
    # …plus this verb's human banner, which under --json joins the stream
    # rather than the document.
    assert f"runner: {runner.url}" in runner.stderr_text
    assert "RUNNER-OWNED" in runner.stderr_text

    # The document says what a caller needs and nothing about the secret.
    assert set(runner.result) >= {"url", "runnerRoot", "tokenFilePresent",
                                 "tokenFile", "port", "authMode"}
    assert runner.envelope["nextAction"]["verb"].startswith(
        f"runner capabilities --runner {runner.url}")


def test_a_second_runner_on_the_same_port_refuses_rather_than_serving_quietly(
        managed_runner, tmp_path, capsys):
    """CONTRACT: a port collision is a typed refusal, not a surprise.

    Two failure modes this rules out: a second runner that silently serves on
    a DIFFERENT port than the one asked for (so the caller's URL points at the
    first one), and a uvicorn traceback from a child that dies half a second
    after the parent claimed success. The bind is tested before the engine is
    started at all, so nothing is left half-running either.
    """
    runner = managed_runner["runner"]
    port = runner.result["port"]
    second_root = str(tmp_path / "second-runner-root")

    code, document = _run_client(
        ["runner", "serve", "--runner-root", second_root, "--port",
         str(port)], capsys)

    assert code == 65, document
    assert document["error"]["code"] == client_cli.RUNNER_PORT_UNAVAILABLE_CODE
    assert str(port) in document["error"]["reason"]
    assert runner.url in document["error"]["repairAction"], \
        "the repair should offer to ASK what is already there"
    # Nothing was started, and nothing was left behind: every refusing check
    # runs before the first write, so the second root was never created and no
    # second credential exists to confuse anyone later.
    assert not os.path.exists(second_root)
    assert runner.process.poll() is None, "the first runner was disturbed"

    # …and the first runner still answers, which is the point of refusing.
    assert _await_info(runner.url, runner, deadline=30.0) in (200, 401)


# =============================================================================
# 2. The two-roles refusal, structurally
# =============================================================================


def test_serving_the_client_workspace_as_the_runner_root_is_refused(
        tmp_path, monkeypatch, capsys):
    """CONTRACT: a runner may not be rooted in the workspace it serves.

    This is the whole ruling in one refusal. A runner rooted in the client's
    workspace would read and write the tree directly, which no remote runner
    can do — so a study that "worked" against it would prove nothing about
    one that has to travel, and the bundle round trip the rest of this file
    exercises would be optional in practice.

    Both spellings of the mistake are refused: the workspace named through
    ``--root`` and through ``$STEERLAB_WORKSPACE``, and nesting in EITHER
    direction — a runner root inside the workspace writes the engine's cache
    into it just as surely as being it.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)

    for argv, note in (
            (["--root", str(workspace), "runner", "serve", "--runner-root",
              str(workspace)], "the workspace itself"),
            (["--root", str(workspace), "runner", "serve", "--runner-root",
              str(workspace / "runner")], "a directory inside it"),
    ):
        code, document = _run_client(argv, capsys)
        assert code == 65, (note, document)
        assert document["error"]["code"] == \
            client_cli.RUNNER_ROOT_IS_WORKSPACE_CODE, note
        # The refusal names the RULE, not just the paths: a reader who has
        # never heard of the two service roles has to be able to learn why.
        assert "upload → submit → evidence" in document["error"]["reason"]
        assert "--runner-root" in document["error"]["repairAction"]
        # Nothing was created, in particular no token in the workspace.
        assert not os.path.exists(str(workspace / "runner.token"))
        assert not os.path.exists(str(workspace / "runner"))

    # The environment spelling refuses identically — `resolve_workspace`
    # exports the same root either way, so there is one check, not two.
    monkeypatch.setenv(client_cli.WORKSPACE_ENV, str(workspace))
    code, document = _run_client(
        ["runner", "serve", "--runner-root", str(workspace)], capsys)
    assert code == 65, document
    assert document["error"]["code"] == \
        client_cli.RUNNER_ROOT_IS_WORKSPACE_CODE


def test_a_sibling_of_the_workspace_is_a_legal_runner_root(tmp_path,
                                                           monkeypatch):
    """The refusal is CONTAINMENT, not string prefix.

    ``/tmp/ws-runner`` starts with ``/tmp/ws`` and is not inside it. A prefix
    test would refuse a legitimate root and send the reader looking for a
    nesting that is not there.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("STEERLAB_ROOT", str(workspace))
    # No exception: the check declines to fire.
    client_cli._refuse_workspace_as_runner_root(str(tmp_path / "ws-runner"))
    client_cli._refuse_workspace_as_runner_root(str(tmp_path / "elsewhere"))
    with pytest.raises(client_cli.ClientRefusal):
        client_cli._refuse_workspace_as_runner_root(str(workspace / "inside"))


def test_the_runner_root_default_is_the_platforms_data_directory(monkeypatch,
                                                                 tmp_path):
    """CONTRACT: the default is a per-platform RUNNER-owned directory.

    Never the current directory, never ``$STEERLAB_WORKSPACE``, never
    ``$STEERLAB_ROOT`` — the whole value of the default is that a person who
    types ``steerlab runner serve`` in their workspace does not thereby serve
    it.
    """
    home = str(tmp_path / "somebody")
    monkeypatch.setenv("HOME", home)
    monkeypatch.setattr(os.path, "expanduser",
                        lambda path: path.replace("~", home, 1))

    monkeypatch.setattr(sys, "platform", "darwin")
    assert client_cli.default_runner_root() == os.path.join(
        home, "Library", "Application Support", "SteerLab", "local-runner")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    assert client_cli.default_runner_root() == os.path.join(
        home, ".local", "share", "steerlab", "local-runner")
    xdg = str(tmp_path / "xdg")
    monkeypatch.setenv("XDG_DATA_HOME", xdg)
    assert client_cli.default_runner_root() == os.path.join(
        xdg, "steerlab", "local-runner")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "name", "nt")
    local_app_data = str(tmp_path / "AppData" / "Local")
    monkeypatch.setenv("LOCALAPPDATA", local_app_data)
    assert client_cli.default_runner_root() == os.path.join(
        local_app_data, "SteerLab", "local-runner")

    # …and it is nobody's workspace and nobody's cwd.
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setenv("STEERLAB_ROOT", os.getcwd())
    monkeypatch.setenv(client_cli.WORKSPACE_ENV, os.getcwd())
    default = client_cli.default_runner_root()
    assert not client_cli._contains(os.getcwd(), default)


# =============================================================================
# 3. The declared surface
# =============================================================================


def test_serve_declares_no_locator_no_token_and_no_host():
    """CONTRACT, structural: what this verb may NOT grow.

    - no ``--runner``: it BECOMES one; a locator here would mean the verb had
      two meanings.
    - no token flag of any spelling: the token is MINTED, and §8.4's rule that
      argv is public did not stop applying because the secret is now ours.
    - no ``--host``: binding a managed runner to the network is the engine's
      decision to make through ``steerlab-server serve``, where the posture
      refusals that gate a non-loopback bind are written.
    """
    spec = client_cli.spec_for("runner", "serve")
    assert spec is not None, "the verb is not in the declared table"
    assert spec.family == client_cli.RUNNER_FAMILY
    # Its own surface is three flags; the rest of `declared_flags` is the
    # envelope's globals (`--help`, `--json`, `--out`), which every verb has.
    assert sorted(spec.value_flags | spec.boolean_flags) == [
        "--port", "--runner-root", "--timeout"]
    for flag in spec.declared_flags:
        assert "token" not in flag
        assert flag not in ("--runner", "--host", "--bind")
    # It requires nothing: `steerlab runner serve` with no flags at all is the
    # supported invocation, and the default root is what makes that safe.
    assert spec.required_flags == frozenset()
    assert client_cli.synopsis(spec).startswith("steerlab runner serve")


def test_a_light_install_is_refused_by_name_before_anything_starts(tmp_path,
                                                                   capsys,
                                                                   monkeypatch):
    """CONTRACT: the light install is told what it is missing.

    `pip install steerlab-server` yields the CLIENT — no fastapi, no uvicorn.
    Asking that install to serve must name the extra, not fail somewhere
    inside an import of a module the person never heard of, and must do so
    before a root is prepared or a token minted.
    """
    monkeypatch.setattr(client_cli, "RUNNER_EXTRA_MODULES",
                        ("steerlab_no_such_engine_module",))
    runner_root = str(tmp_path / "runner-root")

    code, document = _run_client(
        ["runner", "serve", "--runner-root", runner_root], capsys)

    assert code == 65, document
    assert document["error"]["code"] == client_cli.RUNNER_EXTRA_MISSING_CODE
    assert "[runner]" in document["error"]["repairAction"]
    assert not os.path.exists(runner_root), \
        "the refusal created a runner root before refusing"


def test_the_engine_environment_is_built_rather_than_inherited(monkeypatch):
    """CONTRACT: no ambient variable reaches through this verb.

    The two-roles rule is only as strong as the environment the child gets. A
    shell that exports ``STEERLAB_ROOT`` (the commonest thing a SteerLab user
    has in their shell) must not be able to move the runner's artifacts into
    the workspace it names — nor may an inherited ``STEERLAB_AUTH_TOKEN``
    outrank the token file whose path the verb prints.
    """
    layout = {"root": "/runner", "runs": "/runner/runs",
              "metadata": "/runner/.steerlab",
              "jobsDatabase": "/runner/.steerlab/jobs.sqlite",
              "tokenFile": "/runner/runner.token"}
    for name, value in (("STEERLAB_ROOT", "/somebodys/workspace"),
                        ("STEERLAB_RUN_ROOT", "/somebodys/workspace/runs"),
                        ("STEERLAB_METADATA_ROOT", "/somebodys/meta"),
                        ("STEERLAB_JOBS_DB", "/somebodys/jobs.sqlite"),
                        ("STEERLAB_AUTH_TOKEN", "inherited-secret"),
                        ("STEERLAB_AUTH_MODE", "none"),
                        ("STEERLAB_DEV_OPEN_LOOPBACK", "1"),
                        ("STEERLAB_EXECUTOR", "slurm")):
        monkeypatch.setenv(name, value)

    env = client_cli._runner_environment(layout, "127.0.0.1")

    assert env["STEERLAB_ROOT"] == "/runner"
    assert env["STEERLAB_RUN_ROOT"] == "/runner/runs"
    assert env["STEERLAB_METADATA_ROOT"] == "/runner/.steerlab"
    assert env["STEERLAB_JOBS_DB"] == "/runner/.steerlab/jobs.sqlite"
    assert env["STEERLAB_AUTH_TOKEN_FILE"] == "/runner/runner.token"
    # Token mode, on loopback, executing locally — declared, not defaulted.
    assert env["STEERLAB_AUTH_MODE"] == "token"
    assert env["STEERLAB_BIND"] == "127.0.0.1"
    assert env["STEERLAB_EXECUTOR"] == "local"
    # …and the two variables that could quietly undo it are GONE.
    assert "STEERLAB_AUTH_TOKEN" not in env
    assert "STEERLAB_DEV_OPEN_LOOPBACK" not in env
    # Nothing was written back into this process, either.
    assert os.environ["STEERLAB_ROOT"] == "/somebodys/workspace"


def test_a_reused_token_file_is_kept_and_a_loose_one_is_announced(tmp_path,
                                                                  capsys):
    """Restarting a runner must not invalidate the credential an operator
    already put in a script — and a token file the whole machine can read is
    announced, the same call ``api/posture.hydrate_token`` makes at serve
    time."""
    path = str(tmp_path / "runner.token")

    assert client_cli._mint_runner_token(path) is True
    minted = open(path, encoding="utf-8").read()
    assert minted.strip()
    assert oct(os.stat(path).st_mode)[-3:] == "600"
    capsys.readouterr()

    assert client_cli._mint_runner_token(path) is False
    assert open(path, encoding="utf-8").read() == minted
    assert capsys.readouterr().err == ""

    os.chmod(path, 0o644)
    assert client_cli._mint_runner_token(path) is False
    warning = capsys.readouterr().err
    assert "0644" in warning and "chmod 600" in warning
    assert minted.strip() not in warning, "the warning printed the token"
