"""``steerlab run`` — the composite orchestration verb (Phase 5 of the
portability program).

Phases 1b, 2 and 3 made six explicit, separately refusable acts: author,
package, upload, submit, watch, fetch, import. Phase 5 is the **one command**
that walks them — and the whole risk of a composite is that it turns seven
honest refusals into one opaque failure. So what this file pins is not "does
the happy path work" (it does, and the tiers below prove it against a real
engine); it is **that every stage still refuses in its own words, and that the
document says which stage it was.**

Two tiers, and the split is deliberate:

1. **Unit, with an injected adapter and no sockets.** Every stage's failure
   mode, driven through the real CLI (real parsing, real envelope, real
   workspace, real bundles) against a scripted :class:`FakeRunner`. This tier
   is about the STATE MACHINE. It does not re-pin the adapter's own integrity
   checks — the upload digest comparison and the download verify-then-move are
   ``test_client_runner.py``'s, over the genuine routes — it pins what the
   machine does when one of them declines: which stage stops, what the
   envelope says, and (the assertion that matters most) what never happened.
2. **End to end, against the Phase-3 managed runner.** The harness is
   imported rather than re-written: a real ``steerlab runner serve``
   subprocess, an ephemeral loopback port, token mode, a runner-owned root.
   One test drives the whole verb through argv; the other brings a real
   archive home over the same socket and stamps it.

**The one allowance, stated rather than implied.** The only bundled verb that
needs no model is ``verify``, and ``verify`` writes no run directory
(``bundles._execute_run_bundle_inner``), so it packages no evidence. A real
end-to-end ``run`` therefore either executes a GPU verb or ends at a typed
"this job packaged none" — and both are pinned below, the second as
``test_the_whole_machine_runs_against_a_managed_runner``. The evidence half
gets its own end-to-end test over the job Phase 3 already seeds into the
runner's durable store, carrying an archive ``bundles.package_evidence``
really wrote. Everything that travels — the archive, the digests, the socket,
the importer, the stamp — is genuine; only the GPU compute that would have
produced the run directory is skipped.

Fixtures are neutral throughout: one concept named ``signal`` with two
one-word stimuli.
"""

import hashlib
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from steerlab_server import client_cli
from steerlab_server.client import runner as runner_api
from steerlab_server.experiment import bundles, paths

# The Phase-3 managed-runner harness, reused rather than re-written. Importing
# `_isolated_environment` brings its autouse behaviour with it (an autouse
# fixture is autouse wherever its name is visible), which is what keeps this
# module's in-process client calls from inheriting — or leaving behind — a
# `STEERLAB_ROOT` the developer's shell happened to export.
from test_local_runner import (  # noqa: F401 - fixtures used by name
    STUDY_NAME as MANAGED_STUDY,
    _author_study,
    _await_info,
    _isolated_environment,
    _run_client,
    _sha256,
    managed_runner,
)

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Distinctive enough that a substring search for it is meaningful, and
#: obviously not a credential anyone could mistake for real. Same idiom as
#: ``test_client_runner.FAKE_TOKEN``.
FAKE_TOKEN = "steerlab-run-token-DO-NOT-LEAK-4b91de"

STUDY_NAME = "run-fixture"
CONCEPT = "signal"

RUNNER_URL = "http://runner.invalid:8080"


# =============================================================================
# the injected adapter
# =============================================================================


class Script:
    """What the fake runner will do, and what it was asked to do.

    A single mutable object the test sets up and then reads back, so an
    assertion about what NEVER happened (``script.uploads == []``) is as easy
    to write as one about what did — and those are the assertions a
    refuse-before-you-spend contract is made of.
    """

    def __init__(self) -> None:
        self.scheduler_mode = "local"
        self.engine_version = "steerlab-server 0.9.1+abcd1234"
        self.root = "/runner-root"
        self.available_job_types = ["run", "extract", "validate", "sweep",
                                    "evaluate"]
        self.root_looks_like_checkout = False

        #: What the runner will claim the upload hashed to. ``None`` = agree.
        self.reported_upload_sha: str | None = None
        #: Raised INSTEAD of returning, per call.
        self.upload_error: BaseException | None = None
        self.submit_error: BaseException | None = None
        self.download_error: BaseException | None = None

        self.job_status = "succeeded"
        self.job_error: str | None = None
        #: The ``evidenceBundle`` reference the job's result points at, or
        #: ``None`` for a job that packaged none.
        self.evidence: dict | None = None
        #: A real archive on disk, copied to the destination by the fake
        #: download so the importer downstream sees genuine bytes.
        self.archive_source: str | None = None

        # -- what was asked for -------------------------------------------
        self.info_calls = 0
        self.uploads: list = []
        self.submits: list = []
        self.job_polls: list = []
        self.downloads: list = []
        self.cancels: list = []
        self.tokens: list = []


class FakeRunner:
    """A :class:`~steerlab_server.client.runner.RunnerClient` shaped hole.

    Only the methods ``run`` actually calls, answering from a :class:`Script`.
    It deliberately does NOT re-implement the adapter's integrity checks: when
    a test wants "the runner hashed it differently", the script raises the
    adapter's own ``RunnerRefusal`` with the adapter's own code. Re-deriving
    that comparison here would make this file's tests pass while the real one
    was broken, which is the failure mode a fake exists to avoid.
    """

    def __init__(self, script: Script, **kwargs) -> None:
        self.script = script
        self.base_url = str(kwargs["base_url"]).rstrip("/")
        self._token = kwargs.get("token")
        script.tokens.append(kwargs.get("token"))
        self.closed = False

    # -- identity ---------------------------------------------------------

    @property
    def has_token(self) -> bool:
        return bool(self._token)

    def close(self) -> None:
        self.closed = True

    def info(self) -> dict:
        self.script.info_calls += 1
        return {
            "service": "steerlab-server",
            "engineVersion": self.script.engine_version,
            "root": self.script.root,
            "rootLooksLikeSourceCheckout": self.script.root_looks_like_checkout,
            "devices": ["cpu"],
            "loadedModels": [],
            "capabilities": {
                "schedulerMode": self.script.scheduler_mode,
                "serverRole": "workbench",
                "availableJobTypes": list(self.script.available_job_types),
            },
        }

    def capabilities(self) -> dict:      # pragma: no cover - info embeds it
        return self.info()["capabilities"]

    def identity(self, info: dict | None = None) -> dict:
        document = info if info is not None else self.info()
        return {key: document.get(key) for key in runner_api.IDENTITY_KEYS}

    # -- bundles ----------------------------------------------------------

    def upload_run_bundle(self, path: str) -> dict:
        if self.script.upload_error is not None:
            raise self.script.upload_error
        local = runner_api.sha256_file(path)
        reported = self.script.reported_upload_sha or local
        self.script.uploads.append({"path": path, "sha256": local})
        return {"path": f"/runner-root/runs/staged/{os.path.basename(path)}",
                "filename": os.path.basename(path), "sha256": reported,
                "localSha256": local, "bytes": os.path.getsize(path),
                "localBytes": os.path.getsize(path), "executable": True,
                "stagingDirectory": "/runner-root/runs/staged",
                "bundle": {"kind": "runBundle", "experiment": STUDY_NAME}}

    def submit_uploaded_bundle(self, **kwargs) -> dict:
        self.script.submits.append(dict(kwargs))
        if self.script.submit_error is not None:
            raise self.script.submit_error
        return {"jobId": "job-0001", "experiment": STUDY_NAME,
                "verb": kwargs.get("verb"),
                "executor": kwargs.get("executor") or "local",
                "dryRun": bool(kwargs.get("dry_run")),
                "runBundle": {"bundleSha256": kwargs.get("expected_sha256")},
                "slurmJobID": None, "shardJobIDs": [],
                "submissionDirectory": "/runner-root/runs/submit-bundle",
                "recordsDirectory": "/runner-root/runs/submit-bundle/records",
                "preflight": None}

    # -- jobs -------------------------------------------------------------

    def job(self, job_id: str) -> dict:
        self.script.job_polls.append(job_id)
        result: dict = {"experiment": STUDY_NAME, "verb": "run",
                        "runDirectory": "/runner-root/runs/the-run"}
        if self.script.evidence is not None:
            result["runResult"] = {"evidenceBundle": self.script.evidence}
        return {"id": job_id, "kind": "study-submit-bundle",
                "status": self.script.job_status, "result": result,
                "error": self.script.job_error, "logTail": ["seeded"],
                "executor": "local", "createdAt": 0.0, "finishedAt": 1.0}

    def cancel_job(self, job_id: str) -> dict:   # pragma: no cover - never
        self.script.cancels.append(job_id)       # called; asserted absent
        return {"ok": True}

    def evidence_reference(self, job_id: str, *,
                           record: dict | None = None) -> dict | None:
        return self.script.evidence

    def download_bundle(self, *, remote_path, expected_sha256, destination,
                        temp_path, max_bytes=None) -> dict:
        self.script.downloads.append(
            {"remotePath": remote_path, "expected": expected_sha256,
             "destination": destination, "maxBytes": max_bytes})
        if self.script.download_error is not None:
            raise self.script.download_error
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        with open(self.script.archive_source, "rb") as source, \
                open(destination, "wb") as sink:
            payload = source.read()
            sink.write(payload)
        return {"path": destination,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload), "remotePath": remote_path,
                "verified": True, "imported": False}


# =============================================================================
# fixtures
# =============================================================================


def _quiet_client(argv) -> None:
    """One client verb, with its human output swallowed. Fixtures cannot take
    ``capsys``, and a fixture that printed a study's whole authoring
    transcript would bury the assertion that follows it."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()) as out, \
            contextlib.redirect_stderr(io.StringIO()) as err:
        code = client_cli.main(list(argv))
    assert code == 0, (argv, code, out.getvalue(), err.getvalue())


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A client workspace holding ONE frozen study, authored by the light
    verbs the way a person would."""
    root = str(tmp_path / "workspace")
    directory = os.path.join(root, "prompts", "concepts", CONCEPT)
    os.makedirs(directory, exist_ok=True)
    for filename, text in (("positive.jsonl", '{"text": "on"}\n'),
                           ("negative.jsonl", '{"text": "off"}\n')):
        with open(os.path.join(directory, filename), "w",
                  encoding="utf-8") as handle:
            handle.write(text)
    monkeypatch.setenv(client_cli.WORKSPACE_ENV, root)
    monkeypatch.delenv(client_cli.RUNNER_TOKEN_ENV, raising=False)

    # Authored by the CLIENT verbs, not by reaching into the store: the study
    # a person hands to `run` is one the documented lifecycle produced.
    # `--force` skips the EVIDENCE gates — there is no validate run in a
    # client-only workspace and producing one needs a model, the same
    # allowance `test_client_cli.py` and `test_local_runner.py` both make.
    for argv in (
            ["experiment", "create", STUDY_NAME, "--model", "org/tiny",
             "--revision", "0" * 40],
            ["experiment", "attach", STUDY_NAME, CONCEPT],
            ["experiment", "declare-condition", STUDY_NAME, "baseline",
             "--baseline", "--alpha-units", "norm"],
            ["experiment", "freeze", STUDY_NAME, "--force"]):
        _quiet_client(["--root", root, *argv])
    return root


@pytest.fixture
def evidence_archive(tmp_path):
    """A real evidence archive, packaged from a real run directory.

    Not a fabricated tarball: the fake runner hands the machine bytes that
    ``bundles.package_evidence`` wrote, so the importer downstream is doing
    its genuine job and the digests are genuine digests.
    """
    runs_root = tmp_path / "runner-side-runs"
    runs_root.mkdir()
    run_directory = str(runs_root / "the-run")
    os.makedirs(run_directory)
    with open(os.path.join(run_directory, "config.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"experiment": STUDY_NAME, "verb": "run"}, handle)
    with open(os.path.join(run_directory, "records.jsonl"), "w",
              encoding="utf-8") as handle:
        handle.write('{"record": 1, "condition": "baseline"}\n')
    meta = bundles.package_evidence(run_directory)
    return {"path": meta["bundlePath"], "sha256": meta["bundleSha256"],
            "runID": meta["runID"], "meta": meta}


@pytest.fixture
def script(monkeypatch, evidence_archive):
    """The injected adapter, installed over ``runner_api.RunnerClient``.

    The CLI builds its own client (it must: the token and the TLS policy are
    its business). Only the CLASS is swapped, so every verb below exercises
    the real parsing, the real envelope, the real bundles and the real
    workspace — and nothing at all touches a socket.
    """
    scripted = Script()
    scripted.archive_source = evidence_archive["path"]
    monkeypatch.setattr(runner_api, "RunnerClient",
                        lambda **kwargs: FakeRunner(scripted, **kwargs))
    # The machine's poll loop is real; a test must not pay a second per poll
    # for a fake that answers instantly.
    monkeypatch.setattr(client_cli, "POLL_INTERVAL_START", 0.0)
    monkeypatch.setattr(client_cli, "POLL_INTERVAL_CAP", 0.0)
    return scripted


def _evidence_reference(archive: dict) -> dict:
    return {"bundlePath": archive["path"], "bundleSha256": archive["sha256"],
            "experiment": STUDY_NAME, "runID": archive["runID"]}


def _run(argv, capsys, *, root=None):
    """Drive ``steerlab run`` under ``--json`` and return
    ``(code, document, stderr)``."""
    prefix = ["--root", root] if root else []
    code = client_cli.main([*prefix, "run", *argv, "--json"])
    captured = capsys.readouterr()
    assert captured.out.count("\n}") == 1, captured.out
    return code, json.loads(captured.out), captured.err


def _stages(document: dict) -> dict:
    return {row["stage"]: row for row in document["result"]["stages"]}


def _manifest_bytes(root: str) -> bytes:
    path = os.path.join(root, "experiments", STUDY_NAME, "experiment.json")
    if not os.path.exists(path):
        path = os.path.join(root, "experiments", f"{STUDY_NAME}.json")
    with open(path, "rb") as handle:
        return handle.read()


def _workspace_runs(root: str) -> set:
    directory = os.path.join(root, "runs")
    return set(os.listdir(directory)) if os.path.isdir(directory) else set()


# =============================================================================
# 1. The machine, stage by stage
# =============================================================================


def test_the_whole_machine_succeeds_and_reports_every_stage(
        workspace, script, evidence_archive, capsys):
    """CONTRACT: one command, nine stages, and a document that names them all.

    The happy path exists here to establish the shape every refusal below
    departs from: which stages ran, in which order, and what each one put in
    the table a caller reads.
    """
    script.evidence = _evidence_reference(evidence_archive)
    before = _manifest_bytes(workspace)

    code, document, stderr = _run([STUDY_NAME, "--runner", RUNNER_URL], capsys,
                                  root=workspace)

    assert code == 0, document
    assert document["state"] == "ready"
    assert document["verb"] == "run", \
        "a SOLO family must name the command a person types, not 'run run'"
    result = document["result"]
    assert result["outcome"] == "succeeded"
    assert result["imported"] is True

    stages = _stages(document)
    assert [row["stage"] for row in result["stages"]] == \
        list(client_cli.RUN_STAGES)
    assert all(stages[name]["state"] == client_cli.STAGE_OK
               for name in client_cli.RUN_STAGES), stages

    # Each stage carried the fact the NEXT one needs, which is what makes the
    # table a trace rather than a checklist.
    assert stages["package"]["bundleSha256"] == script.uploads[0]["sha256"]
    assert stages["upload"]["sha256"] == stages["package"]["bundleSha256"]
    assert stages["submit"]["jobId"] == "job-0001"
    assert stages["evidence"]["sha256"] == evidence_archive["sha256"]
    assert stages["import"]["runID"] == evidence_archive["runID"]
    assert stages["provenance"]["filename"] == client_cli.PROVENANCE_FILENAME

    # The evidence really landed, and the frozen study really did not move.
    imported = os.path.join(workspace, "runs", evidence_archive["runID"],
                            "records.jsonl")
    assert os.path.isfile(imported)
    assert _manifest_bytes(workspace) == before

    # One human line per transition, on stderr in both modes.
    for stage in client_cli.RUN_STAGES:
        assert f"run[{stage}]:" in stderr, (stage, stderr)


def test_an_unfrozen_study_is_refused_before_the_runner_is_addressed(
        workspace, script, capsys):
    """CONTRACT: stage 1 refuses locally, and the fake sees NOTHING.

    A draft has no freeze hash for a remote citation to rest on. Discovering
    that after the upload would mean an archive staged on somebody else's disk
    for a study that was never submittable.
    """
    from steerlab_server.experiment import experiment_store as store
    store.create("draft-study", model_id="org/tiny", revision="0" * 40,
                 root=workspace)

    code, document, _ = _run(["draft-study", "--runner", RUNNER_URL], capsys,
                             root=workspace)

    assert code == 65, document
    assert document["error"]["code"] == client_cli.NOT_FROZEN_CODE
    assert "experiment freeze draft-study" in document["error"]["repairAction"]
    assert document["result"]["failedStage"] == "load"
    stages = _stages(document)
    assert stages["load"]["state"] == client_cli.STAGE_REFUSED
    assert stages["package"]["state"] == client_cli.STAGE_NOT_REACHED
    # Nothing was built and nobody was called — not even to ask who they are.
    assert script.info_calls == 0
    assert script.uploads == [] and script.submits == []


def test_a_drifted_pin_is_refused_with_the_stores_own_gate(
        workspace, script, capsys):
    """CONTRACT: drift refuses with ``pinDrift`` — the STORE's vocabulary, not
    a composite's paraphrase of it.

    A lifecycle refusal's ``code`` IS its gate id and ``error.gate`` is
    present; an agent that learned to switch on that from ``experiment verify``
    must meet the identical shape here.
    """
    stimulus = os.path.join(workspace, "prompts", "concepts", CONCEPT,
                            "positive.jsonl")
    with open(stimulus, "w", encoding="utf-8") as handle:
        handle.write('{"text": "changed after the freeze"}\n')

    code, document, _ = _run([STUDY_NAME, "--runner", RUNNER_URL], capsys,
                             root=workspace)

    assert code == 65, document
    from steerlab_server.experiment import lifecycle_gates
    assert document["error"]["code"] == lifecycle_gates.PIN_DRIFT
    assert document["error"]["gate"] == lifecycle_gates.PIN_DRIFT
    assert document["result"]["failedStage"] == "load"
    assert document["result"]["violations"]
    assert script.info_calls == 0 and script.uploads == []


def test_a_capability_refusal_happens_before_any_upload(workspace, script,
                                                        capsys):
    """CONTRACT: the runner is asked what it can do BEFORE it is handed
    anything — and the refusal names what it offers.

    ``--executor slurm`` against a locally-scheduling runner refuses inside
    the submit route today, after the archive is staged. Reading
    ``schedulerMode`` costs one GET that has already happened.
    """
    script.scheduler_mode = "local"

    code, document, _ = _run(
        [STUDY_NAME, "--runner", RUNNER_URL, "--executor", "slurm"], capsys,
        root=workspace)

    assert code == 65, document
    assert document["error"]["code"] == client_cli.RUNNER_CANNOT_EXECUTE_CODE
    assert "'local'" in document["error"]["reason"]
    offers = document["result"]["runnerOffers"]
    assert offers["schedulerMode"] == "local"
    assert offers["studyVerbs"] == list(client_cli.RUN_STUDY_VERBS)
    assert offers["engineVersion"] == script.engine_version

    stages = _stages(document)
    assert stages["package"]["state"] == client_cli.STAGE_OK
    assert stages["capabilities"]["state"] == client_cli.STAGE_REFUSED
    assert stages["upload"]["state"] == client_cli.STAGE_NOT_REACHED
    # THE assertion: the runner was asked, and never given anything.
    assert script.info_calls == 1
    assert script.uploads == [], "an unexecutable submission was uploaded"
    assert script.submits == []


def test_an_unknown_study_verb_is_refused_before_any_upload(workspace, script,
                                                            capsys):
    """The other half of the capability gate: a verb no runner executes.

    The roster travels in the refusal, because "unsupported" with no list
    beside it sends the reader to a different machine to find out what they
    should have typed.
    """
    code, document, _ = _run(
        [STUDY_NAME, "--runner", RUNNER_URL, "--verb", "frobnicate"], capsys,
        root=workspace)

    assert code == 65, document
    assert document["error"]["code"] == client_cli.RUNNER_CANNOT_EXECUTE_CODE
    for verb in client_cli.RUN_STUDY_VERBS:
        assert verb in document["error"]["reason"]
    assert script.uploads == []


def test_a_dry_run_may_name_slurm_on_any_runner(workspace, script, capsys):
    """The capability gate is not a blanket: a dry run schedules nothing, and
    the submit route accepts it against any runner. A precheck stricter than
    the rule it anticipates would refuse work that would have succeeded."""
    script.scheduler_mode = "local"
    script.job_status = "prepared"

    code, document, _ = _run(
        [STUDY_NAME, "--runner", RUNNER_URL, "--executor", "slurm",
         "--dry-run"], capsys, root=workspace)

    assert code == 0, document
    assert document["result"]["outcome"] == "succeeded"
    assert script.submits[0]["dry_run"] is True
    stages = _stages(document)
    # A dry run runs nothing, so there is nothing to bring home and the
    # machine says so rather than refusing.
    assert stages["evidence"]["state"] == client_cli.STAGE_SKIPPED
    assert stages["evidence"]["reason"] == "--dry-run"


def test_an_upload_the_runner_hashes_differently_stops_before_submit(
        workspace, script, capsys):
    """CONTRACT: the adapter's upload refusal stops the machine at ``upload``.

    The comparison itself is the adapter's and is pinned over the real routes
    in ``test_client_runner.py``. What this pins is the consequence: a
    truncated archive never becomes a submitted study.
    """
    script.upload_error = runner_api.RunnerRefusal(
        "the runner received different bytes than were sent",
        code="uploadDigestMismatch",
        repair_action="do NOT submit the staged path — re-upload",
        detail={"reportedSha256": "f" * 64, "localSha256": "a" * 64})

    code, document, _ = _run([STUDY_NAME, "--runner", RUNNER_URL], capsys,
                             root=workspace)

    assert code == 65, document
    assert document["error"]["code"] == "uploadDigestMismatch"
    assert document["result"]["failedStage"] == "upload"
    assert document["result"]["reportedSha256"] == "f" * 64
    assert _stages(document)["submit"]["state"] == client_cli.STAGE_NOT_REACHED
    assert script.submits == []


def test_a_submit_transport_error_is_never_retried_and_says_to_look(
        workspace, script, capsys):
    """CONTRACT: ``submit`` is the one call this machine will not repeat.

    A timeout after submit is genuinely ambiguous — the job may exist, and on
    Slurm it may already hold an allocation. Retrying would be a second study.
    The repair is to LOOK, and the exact command is in the document.
    """
    script.submit_error = runner_api.RunnerUnreachable(
        "ReadTimeout reaching /api/studies/submit-bundle",
        repair_action="check the runner")

    code, document, _ = _run([STUDY_NAME, "--runner", RUNNER_URL], capsys,
                             root=workspace)

    assert code == 70, document
    assert document["state"] == "failed"
    assert document["error"]["code"] == client_cli.SUBMIT_OUTCOME_UNKNOWN_CODE
    assert document["result"]["failedStage"] == "submit"
    assert document["result"]["retried"] is False
    repair = document["error"]["repairAction"]
    assert f"steerlab runner jobs --runner {RUNNER_URL}" in repair
    assert "runner submit" in repair, \
        "the repair should offer to resume from the already-staged bundle"
    # Exactly one attempt, and no poll, no download, no cancel.
    assert len(script.submits) == 1
    assert script.job_polls == [] and script.downloads == []
    assert script.cancels == []


def test_an_upload_transport_error_is_retried_once(workspace, script, capsys,
                                                   monkeypatch,
                                                   evidence_archive):
    """The other side of the retry table: upload IS idempotent.

    Each upload lands in its own server-minted staging directory, so a retry
    costs disk and produces a second path, never a second effect.
    """
    script.evidence = _evidence_reference(evidence_archive)
    attempts: list = []
    real_upload = FakeRunner.upload_run_bundle

    def flaky(self, path):
        attempts.append(path)
        if len(attempts) == 1:
            raise runner_api.RunnerUnreachable(
                "ConnectError reaching /api/bundles/upload",
                repair_action="check the runner")
        return real_upload(self, path)

    monkeypatch.setattr(FakeRunner, "upload_run_bundle", flaky)

    code, document, stderr = _run([STUDY_NAME, "--runner", RUNNER_URL], capsys,
                                  root=workspace)

    assert code == 0, document
    assert len(attempts) == 2, "an idempotent call was not retried"
    assert "retrying once" in stderr
    assert _stages(document)["upload"]["state"] == client_cli.STAGE_OK


def test_a_failed_job_with_partial_evidence_still_comes_home(
        workspace, script, evidence_archive, capsys):
    """CONTRACT: evidence comes home from a FAILURE too — loudly.

    "The data still exists somewhere under /scratch" is not a retention
    policy. A failed stage's partial output is exactly what a researcher
    needs; what must never happen is that it arrives quietly enough to be
    mistaken for a result.
    """
    script.job_status = "failed"
    script.job_error = "RuntimeError: the model would not load"
    script.evidence = _evidence_reference(evidence_archive)

    code, document, _ = _run([STUDY_NAME, "--runner", RUNNER_URL], capsys,
                             root=workspace)

    # The RUN failed — 70, typed, with the runner's own sentence in it.
    assert code == 70, document
    assert document["state"] == "failed"
    assert document["error"]["code"] == client_cli.REMOTE_JOB_FAILED_CODE
    assert "the model would not load" in document["error"]["reason"]
    assert "never a result" in document["error"]["repairAction"]

    # …and the evidence is nonetheless HERE, imported and stamped.
    result = document["result"]
    assert result["imported"] is True
    assert result["outcome"] == "jobFailed"
    stages = _stages(document)
    assert stages["wait"]["state"] == client_cli.STAGE_FAILED
    assert stages["evidence"]["state"] == client_cli.STAGE_OK
    assert stages["import"]["state"] == client_cli.STAGE_OK
    assert stages["provenance"]["state"] == client_cli.STAGE_OK

    run_directory = os.path.join(workspace, "runs", evidence_archive["runID"])
    assert os.path.isfile(os.path.join(run_directory, "records.jsonl"))
    with open(os.path.join(run_directory, client_cli.PROVENANCE_FILENAME),
              encoding="utf-8") as handle:
        stamp = json.load(handle)
    assert stamp["outcome"] == "jobFailed"
    assert stamp["job"]["status"] == "failed"
    assert stamp["job"]["error"] == script.job_error


def test_a_terminal_job_with_no_evidence_reports_a_typed_outcome(
        workspace, script, capsys):
    """CONTRACT: "there is nothing to bring home" is an ANSWER, not a crash.

    A job that completed and packaged no bundle is the ordinary shape of
    ``--verb verify`` (it writes no run directory) — so the refusal names that
    rather than failing on a missing key.
    """
    script.job_status = "succeeded"
    script.evidence = None

    code, document, _ = _run(
        [STUDY_NAME, "--runner", RUNNER_URL, "--verb", "verify"], capsys,
        root=workspace)

    assert code == 65, document
    assert document["error"]["code"] == client_cli.NO_EVIDENCE_CODE
    assert document["result"]["outcome"] == "noEvidence"
    assert document["result"]["failedStage"] == "evidence"
    assert "verify` writes none by design" in document["error"]["repairAction"]
    stages = _stages(document)
    assert stages["wait"]["state"] == client_cli.STAGE_OK
    assert stages["evidence"]["state"] == client_cli.STAGE_REFUSED
    assert stages["import"]["state"] == client_cli.STAGE_NOT_REACHED
    assert all("bundle-" in name for name in _workspace_runs(workspace)), \
        "nothing but the packaged run bundle should be in the workspace runs/"


def test_no_evidence_requested_is_a_skip_and_not_a_refusal(workspace, script,
                                                           capsys):
    """``--no-evidence`` asked for exactly this. Refusing what the caller
    requested would make the flag unusable."""
    code, document, _ = _run(
        [STUDY_NAME, "--runner", RUNNER_URL, "--no-evidence"], capsys,
        root=workspace)

    assert code == 0, document
    assert document["result"]["imported"] is False
    assert script.submits[0]["package_evidence"] is False
    for stage in ("evidence", "import", "provenance"):
        row = _stages(document)[stage]
        assert row["state"] == client_cli.STAGE_SKIPPED
        assert row["reason"] == "--no-evidence"


def test_a_download_whose_digest_disagrees_imports_nothing(
        workspace, script, evidence_archive, capsys):
    """CONTRACT: a substituted archive stops at ``evidence``.

    The verify-then-move is the adapter's (``test_client_runner.py`` pins it
    over the real route). What this pins is that the machine does not carry
    on: nothing is imported, and no provenance stamp claims a run arrived.
    """
    script.evidence = _evidence_reference(evidence_archive)
    script.download_error = runner_api.RunnerRefusal(
        "the downloaded archive hashes to abc…, not the def… the runner "
        "reported",
        code="evidenceDigestMismatch",
        repair_action="do NOT import this file",
        detail={"expectedSha256": "d" * 64, "downloadedSha256": "a" * 64})
    before = _workspace_runs(workspace)

    code, document, _ = _run([STUDY_NAME, "--runner", RUNNER_URL], capsys,
                             root=workspace)

    assert code == 65, document
    assert document["error"]["code"] == "evidenceDigestMismatch"
    assert document["result"]["failedStage"] == "evidence"
    stages = _stages(document)
    assert stages["import"]["state"] == client_cli.STAGE_NOT_REACHED
    assert stages["provenance"]["state"] == client_cli.STAGE_NOT_REACHED
    # The only thing the machine added to the workspace is the run bundle it
    # PACKAGED — no imported run, no partial, no debris.
    gained = _workspace_runs(workspace) - before
    assert all("bundle-" in name for name in gained), sorted(gained)
    assert not os.path.exists(os.path.join(workspace, "runs",
                                           evidence_archive["runID"]))
    assert document["result"].get("provenancePath") in (None, "")


def test_no_wait_detaches_and_prints_the_exact_follow_ups(workspace, script,
                                                          capsys):
    """CONTRACT: ``--no-wait`` is a SUCCESS document for work in flight.

    ``pending`` (exit 12) is what the shared vocabulary is for, and it is what
    the engine's own ``study submit`` already answers for the same situation.
    The three commands that finish the job by hand travel in the document,
    runnable as printed.
    """
    code, document, _ = _run([STUDY_NAME, "--runner", RUNNER_URL, "--no-wait"],
                             capsys, root=workspace)

    assert code == 12, document
    assert document["state"] == "pending"
    assert "error" not in document, "a detach is not a refusal"
    result = document["result"]
    assert result["outcome"] == "detached"
    assert result["job"]["id"] == "job-0001"
    follow_ups = result["followUps"]
    assert follow_ups[0] == f"steerlab runner jobs job-0001 --runner {RUNNER_URL}"
    assert follow_ups[1].startswith("steerlab runner evidence job-0001 --out ")
    assert follow_ups[2].startswith("steerlab bundle import ")
    assert document["nextAction"]["verb"] == follow_ups[0]
    for stage in ("wait", "evidence", "import", "provenance"):
        row = _stages(document)[stage]
        assert row["state"] == client_cli.STAGE_SKIPPED
        assert row["reason"] == "--no-wait"
    # Detaching never polls and never cancels.
    assert script.job_polls == [] and script.cancels == []


def test_the_wait_deadline_stops_watching_and_never_cancels(workspace, script,
                                                            capsys):
    """CONTRACT: ``--timeout`` bounds this client's patience, never the
    runner's work.

    A composite that cancelled on its own deadline would destroy hours of
    compute because a laptop lid closed.
    """
    script.job_status = "running"

    code, document, _ = _run(
        [STUDY_NAME, "--runner", RUNNER_URL, "--timeout", "0.01"], capsys,
        root=workspace)

    assert code == 70, document
    assert document["error"]["code"] == client_cli.WAIT_DEADLINE_CODE
    assert "NOT cancelled" in document["error"]["reason"]
    assert document["result"]["cancelled"] is False
    assert document["result"]["failedStage"] == "wait"
    assert script.cancels == [], "the deadline cancelled a remote job"
    assert script.job_polls, "the deadline expired without polling at all"


def test_an_interrupt_during_the_wait_detaches_rather_than_cancelling(
        workspace, script, capsys, monkeypatch):
    """CONTRACT: ctrl-c stops the WATCHING.

    Driven through the real handler: the fake's ``job`` raises SIGINT into
    this process on its first answer, exactly as a person pressing ctrl-c
    would while the machine is between polls.
    """
    import signal

    script.job_status = "running"
    delivered: list = []
    real_job = FakeRunner.job

    def interrupting(self, job_id):
        record = real_job(self, job_id)
        if not delivered:
            delivered.append(job_id)
            os.kill(os.getpid(), signal.SIGINT)
        return record

    monkeypatch.setattr(FakeRunner, "job", interrupting)

    code, document, stderr = _run([STUDY_NAME, "--runner", RUNNER_URL], capsys,
                                  root=workspace)

    assert code == 12, document
    assert document["state"] == "pending"
    assert "error" not in document
    assert document["result"]["outcome"] == "interrupted"
    assert document["result"]["cancelled"] is False
    assert "not cancelling" in stderr
    assert document["result"]["followUps"][0].endswith(
        f"--runner {RUNNER_URL}")
    assert script.cancels == []
    # The handler is restored: an interrupt after this verb behaves exactly as
    # it did before it.
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


# =============================================================================
# 2. Provenance — additive, and never a credential
# =============================================================================


def test_the_provenance_record_lands_in_the_run_and_names_its_facts(
        workspace, script, evidence_archive, capsys):
    """CONTRACT: the record answers what the imported bytes cannot.

    WHICH engine produced them, from WHICH bundle, under WHICH job, and with
    what outcome. It lands inside the imported run directory — the house
    pattern ``adjudication.STAMP_FILENAME`` and ``import_bundle``'s own
    ``pipeline-portable.json`` both follow — as part of the landing write.
    """
    script.evidence = _evidence_reference(evidence_archive)

    code, document, _ = _run(
        [STUDY_NAME, "--runner", RUNNER_URL, "--verb", "run", "--executor",
         "local"], capsys, root=workspace)
    assert code == 0, document

    run_directory = os.path.join(workspace, "runs", evidence_archive["runID"])
    path = os.path.join(run_directory, client_cli.PROVENANCE_FILENAME)
    assert document["result"]["provenancePath"] == path
    with open(path, encoding="utf-8") as handle:
        stamp = json.load(handle)

    assert stamp["schemaVersion"] == client_cli.PROVENANCE_SCHEMA
    assert stamp["kind"] == "remoteExecution"
    assert stamp["experiment"] == STUDY_NAME
    assert stamp["outcome"] == "succeeded"
    assert stamp["client"]["role"] == client_cli.ROLE

    # WHICH engine — read off /api/info, never assumed.
    assert stamp["runner"]["url"] == RUNNER_URL
    assert stamp["runner"]["engineVersion"] == script.engine_version
    assert stamp["runner"]["service"] == "steerlab-server"

    # WHICH bundle, WHICH job — full digests, never elided.
    assert stamp["runBundle"]["sha256"] == script.uploads[0]["sha256"]
    assert len(stamp["runBundle"]["sha256"]) == 64
    assert stamp["runBundle"]["runnerPath"]
    assert stamp["job"]["id"] == "job-0001"
    assert stamp["job"]["verb"] == "run"
    assert stamp["job"]["executor"] == "local"
    assert stamp["evidence"]["sha256"] == evidence_archive["sha256"]
    assert stamp["evidence"]["verified"] is True

    # WHEN, in this client's own clock, in the envelope's own format.
    for key in ("startedAt", "submittedAt", "terminalAt", "importedAt"):
        assert stamp["timestamps"][key].endswith("Z"), stamp["timestamps"]

    # …and the manifest's identity, so the run can be tied back to the study
    # WITHOUT the study having been modified to point at the run.
    assert stamp["manifest"]["status"] == "frozen"
    assert len(stamp["manifest"]["contentHash"]) == 64
    assert stamp["note"].startswith("Remote execution provenance")


def test_the_provenance_stamp_leaves_the_frozen_study_untouched(
        workspace, script, evidence_archive, capsys):
    """CONTRACT: ADDITIVE means additive.

    Nothing under ``experiments/`` moves, no ``pinned/`` file appears, and the
    frozen manifest is byte-identical before and after — which is the only
    version of "we recorded provenance without rewriting the study" that can
    be checked rather than believed.
    """
    script.evidence = _evidence_reference(evidence_archive)
    experiments = os.path.join(workspace, "experiments")
    before_bytes = _manifest_bytes(workspace)
    before_tree = {}
    for dirpath, _dirs, names in os.walk(experiments):
        for name in names:
            full = os.path.join(dirpath, name)
            with open(full, "rb") as handle:
                before_tree[os.path.relpath(full, experiments)] = handle.read()

    code, document, _ = _run([STUDY_NAME, "--runner", RUNNER_URL], capsys,
                             root=workspace)
    assert code == 0, document

    after_tree = {}
    for dirpath, _dirs, names in os.walk(experiments):
        for name in names:
            full = os.path.join(dirpath, name)
            with open(full, "rb") as handle:
                after_tree[os.path.relpath(full, experiments)] = handle.read()

    assert after_tree == before_tree, \
        "the composite wrote into experiments/ — provenance must be additive"
    assert _manifest_bytes(workspace) == before_bytes
    # The pin surface the freeze wrote is still exactly what the freeze wrote
    # — no new member, and none rewritten (the tree comparison above covers
    # the bytes; this names the directory the contract calls out by name).
    assert not any(name.startswith(f"{STUDY_NAME}/pinned/")
                   and name not in before_tree for name in after_tree)
    # The stamp is where it belongs: inside the run it describes.
    run_directory = os.path.join(workspace, "runs", evidence_archive["runID"])
    assert os.path.isfile(os.path.join(run_directory,
                                       client_cli.PROVENANCE_FILENAME))


def test_the_token_reaches_no_document_no_stamp_and_no_log_line(
        workspace, script, evidence_archive, capsys, monkeypatch, tmp_path):
    """CONTRACT (§8.4, extended to the composite): presence, never the value.

    One distinctive fake token is driven through a whole successful machine —
    a success envelope, every stage line on stderr, and the durable provenance
    record — and must appear in none of them. The provenance file is the new
    surface this phase added and therefore the one most likely to leak.
    """
    token_file = tmp_path / "runner.token"
    token_file.write_text(FAKE_TOKEN, encoding="utf-8")
    os.chmod(token_file, 0o600)
    script.evidence = _evidence_reference(evidence_archive)

    code, document, stderr = _run(
        [STUDY_NAME, "--runner", RUNNER_URL, "--token-file", str(token_file)],
        capsys, root=workspace)

    assert code == 0, document
    assert script.tokens == [FAKE_TOKEN], "the token never reached the adapter"
    assert document["result"]["tokenPresent"] is True
    assert document["result"]["runner"]["tokenPresent"] is True

    rendered = json.dumps(document)
    assert FAKE_TOKEN not in rendered
    assert FAKE_TOKEN not in stderr

    stamp_path = os.path.join(workspace, "runs", evidence_archive["runID"],
                              client_cli.PROVENANCE_FILENAME)
    with open(stamp_path, encoding="utf-8") as handle:
        stamp_text = handle.read()
    assert FAKE_TOKEN not in stamp_text
    assert json.loads(stamp_text)["runner"]["tokenPresent"] is True
    # Nothing anywhere in the workspace carries it, either.
    for dirpath, _dirs, names in os.walk(workspace):
        for name in names:
            with open(os.path.join(dirpath, name), "rb") as handle:
                assert FAKE_TOKEN.encode() not in handle.read(), name


def test_the_provenance_record_is_written_once(tmp_path):
    """``O_EXCL``: a run's provenance is written as part of its landing write
    and never rewritten. In practice the importer refuses an existing member
    long before this matters, which is exactly why the guard is cheap."""
    run_directory = str(tmp_path / "run")
    os.makedirs(run_directory)
    first = client_cli._write_provenance(run_directory, {"outcome": "first"})
    assert first and os.path.isfile(first)
    assert client_cli._write_provenance(run_directory,
                                        {"outcome": "second"}) is None
    with open(first, encoding="utf-8") as handle:
        assert json.load(handle)["outcome"] == "first"
    # No run directory to stamp is not an error; it is nothing to do.
    assert client_cli._write_provenance("", {"outcome": "x"}) is None
    assert client_cli._write_provenance(str(tmp_path / "nope"), {}) is None


# =============================================================================
# 3. The declared surface, and the vocabularies it transcribes
# =============================================================================


def test_run_is_a_solo_family_spelled_as_one_word():
    """CONTRACT: ``steerlab run <experiment>``, never ``steerlab run run``.

    The synopsis, the help page, the envelope's ``verb`` and every repair
    sentence name the command a person types.
    """
    spec = client_cli.spec_for("run", "run")
    assert spec is not None
    assert client_cli.SOLO_FAMILIES[client_cli.RUN_FAMILY] == spec.verb
    assert client_cli.verb_label(spec) == "run"
    assert client_cli.synopsis(spec).startswith(
        "steerlab run <experiment> --runner <url>")
    invocation = client_cli.parse("run", ["some-study", "--runner", RUNNER_URL])
    assert invocation.positionals == ["some-study"]
    assert invocation.label == "run"
    # Every other family is unaffected: the verb word is still consumed.
    other = client_cli.parse("runner", ["capabilities", "--runner", RUNNER_URL])
    assert other.positionals == [] and other.label == "runner capabilities"


def test_there_is_no_token_flag_on_the_composite():
    """§8.4's rule did not stop applying because the verb got bigger: argv is
    readable by every process on a shared login node."""
    spec = client_cli.spec_for("run", "run")
    spellings = [flag for flag in spec.declared_flags if "token" in flag]
    assert spellings == ["--token-file"], spellings
    assert "--token" not in spec.declared_flags


def test_the_study_verb_vocabulary_twins_the_route():
    """CONTRACT: the capability precheck's vocabulary IS the submit route's.

    A client literal that drifted from ``VALID_STUDY_VERBS`` would refuse work
    the runner would have accepted (or wave through work it would not), which
    is worse than not checking. The literal exists because importing the route
    module pulls FastAPI into the light client.
    """
    from steerlab_server.api.submissions import VALID_STUDY_VERBS
    assert set(client_cli.RUN_STUDY_VERBS) == set(VALID_STUDY_VERBS)
    assert list(client_cli.RUN_STUDY_VERBS) == sorted(VALID_STUDY_VERBS)
    assert client_cli.DEFAULT_STUDY_VERB in client_cli.RUN_STUDY_VERBS


def test_the_terminal_job_statuses_twin_the_job_manager():
    """Same discipline for "has this job stopped moving": the machine's poll
    loop and the runner's job manager must agree on what terminal means, or a
    finished job is watched until the deadline."""
    from steerlab_server.api.jobs import TERMINAL
    assert set(client_cli.JOB_TERMINAL_STATUSES) == set(TERMINAL)
    assert list(client_cli.JOB_TERMINAL_STATUSES) == sorted(TERMINAL)
    assert set(client_cli.JOB_SUCCESS_STATUSES) <= set(TERMINAL)
    assert "failed" not in client_cli.JOB_SUCCESS_STATUSES


def test_the_executor_vocabulary_twins_the_profile():
    from steerlab_server.api.profile import _VALID_EXECUTORS
    assert set(client_cli.RUN_EXECUTORS) == set(_VALID_EXECUTORS)


def test_the_stage_table_is_the_declared_order():
    """The stages a document reports are the stages the module declares, in
    order, always all of them — a caller must be able to see which never
    happened rather than infer it from absence."""
    table = client_cli._stage_table({"load": {"stage": "load", "state": "ok"}})
    assert [row["stage"] for row in table] == list(client_cli.RUN_STAGES)
    assert table[0]["state"] == client_cli.STAGE_OK
    assert all(row["state"] == client_cli.STAGE_NOT_REACHED
               for row in table[1:])


# =============================================================================
# 4. End to end, against the Phase-3 managed runner
# =============================================================================


def test_the_whole_machine_runs_against_a_managed_runner(managed_runner,
                                                         tmp_path, capsys):
    """CONTRACT: the composite works against a REAL served engine.

    One command — `steerlab run <exp> --runner <url> --verb verify --executor
    local` — over a real loopback socket, in token mode, against the genuine
    ``steerlab-server serve`` the Phase-3 verb starts. Everything is real: the
    study, the archive, the digests, the upload, the submission, the
    execution, the polling.

    It ends at a TYPED "this job packaged none", and that is the honest
    outcome rather than a shortfall: ``verify`` is the only bundled verb that
    needs no model, and it writes no run directory, so it has no evidence to
    package (``bundles._execute_run_bundle_inner``). The import-and-stamp half
    is exercised against the same runner by the test below, over an archive
    ``package_evidence`` really wrote.
    """
    runner = managed_runner["runner"]
    runner_root = managed_runner["root"]
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace)
    _author_study(workspace, capsys)
    before = _manifest_bytes_named(workspace, MANAGED_STUDY)
    runs_before = _workspace_runs(workspace)

    code = client_cli.main([
        "--root", workspace, "run", MANAGED_STUDY,
        "--runner", runner.url, "--token-file", runner.result["tokenFile"],
        "--verb", "verify", "--executor", "local", "--json"])
    text = capsys.readouterr().out
    assert text.count("\n}") == 1, text
    document = json.loads(text)

    assert code == 65, document
    assert document["error"]["code"] == client_cli.NO_EVIDENCE_CODE
    result = document["result"]
    stages = {row["stage"]: row for row in result["stages"]}
    for stage in ("load", "package", "capabilities", "upload", "submit",
                  "wait"):
        assert stages[stage]["state"] == client_cli.STAGE_OK, (stage, stages)
    assert stages["wait"]["status"] == "succeeded"

    # The runner really is the runner, read over the wire.
    assert result["runner"]["root"] == os.path.realpath(runner_root)
    assert result["runner"]["root"] != os.path.realpath(workspace)
    assert result["runner"]["engineVersion"].startswith("steerlab-server ")
    assert result["runner"]["tokenPresent"] is True

    # It really executed the bundle it was given, into ITS root.
    assert os.path.isfile(os.path.join(runner_root, "experiments",
                                       MANAGED_STUDY, "experiment.json"))
    staged = stages["upload"]["runnerPath"]
    assert os.path.realpath(staged).startswith(os.path.realpath(runner_root))
    assert stages["upload"]["sha256"] == _sha256(
        stages["package"]["bundlePath"])

    # …and the client workspace gained exactly the packaged bundle, no
    # staging directory, no job database, no token.
    gained = _workspace_runs(workspace) - runs_before
    assert all("bundle-" in name for name in gained), sorted(gained)
    assert not os.path.exists(os.path.join(workspace, ".steelab"))
    assert not os.path.exists(os.path.join(workspace, ".steerlab"))
    assert not os.path.exists(os.path.join(
        workspace, client_cli.RUNNER_TOKEN_FILENAME))
    assert _manifest_bytes_named(workspace, MANAGED_STUDY) == before

    # The runner is still serving: a typed refusal disturbed nothing.
    assert _await_info(runner.url, runner, deadline=30.0) in (200, 401)


def test_evidence_comes_home_verified_and_stamped_over_a_real_socket(
        managed_runner, tmp_path, capsys):
    """CONTRACT: stages 7–9, for real — download, verify, import, stamp.

    Over the same managed runner and the same socket, against the job Phase 3
    seeds into the runner's own durable store carrying an archive
    ``bundles.package_evidence`` really wrote. Entered at the stage boundary
    rather than through argv because no model-free verb produces evidence (see
    the test above); everything the stages touch — the HTTP download route,
    the outer-digest verification, the importer, the stamp writer — is the
    production code path.
    """
    runner = managed_runner["runner"]
    runner_root = managed_runner["root"]
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace)
    _author_study(workspace, capsys)
    job_id = managed_runner["evidenceJob"]
    expected = managed_runner["evidenceMeta"]["bundleSha256"]

    with open(runner.result["tokenFile"], encoding="utf-8") as handle:
        secret = handle.read().strip()
    client = runner_api.RunnerClient(base_url=runner.url, token=secret)
    try:
        record = client.job(job_id)
        reference = client.evidence_reference(job_id, record=record)
        assert reference is not None and reference["bundleSha256"] == expected

        destination = str(tmp_path / "home" / "evidence.tar.gz")
        downloaded = client_cli._download_evidence(
            client, reference=reference, destination=destination,
            temp_path=destination + ".partial", max_bytes=None)
        assert downloaded["verified"] is True
        assert downloaded["sha256"] == expected
        assert not os.path.exists(destination + ".partial"), "staging debris"

        imported = bundles.import_bundle(
            downloaded["path"], target_root=workspace,
            expected_sha256=downloaded["sha256"])
        run_id = imported["bundle"]["runID"]
        run_directory = os.path.join(workspace, "runs", run_id)
        assert os.path.isdir(run_directory)

        stamp_path = client_cli._write_provenance(
            run_directory,
            client_cli._provenance_document(
                experiment=MANAGED_STUDY,
                manifest_facts={"status": "frozen", "contentHash": "0" * 64,
                                "freezeHash": None, "freezeForced": True,
                                "modelID": "org/tiny", "modelRevision": None},
                runner={"url": runner.url, "service": "steerlab-server",
                        "engineVersion": "steerlab-server test",
                        "root": os.path.realpath(runner_root),
                        "schedulerMode": "local", "serverRole": "workbench",
                        "tokenPresent": True},
                bundle={"sha256": "b" * 64, "runnerPath": "/staged"},
                job={"id": job_id, "status": "succeeded", "verb": "run",
                     "executor": "local"},
                evidence={"sha256": downloaded["sha256"],
                          "bytes": downloaded["bytes"], "verified": True},
                outcome="succeeded",
                timestamps={"importedAt": "2026-01-01T00:00:00Z"}))
    finally:
        client.close()

    assert stamp_path == os.path.join(run_directory,
                                      client_cli.PROVENANCE_FILENAME)
    with open(stamp_path, encoding="utf-8") as handle:
        stamp = json.load(handle)
    assert stamp["runner"]["tokenPresent"] is True
    assert secret not in json.dumps(stamp), "the stamp carries the token"
    assert stamp["evidence"]["sha256"] == expected

    # The two-roles assertions, repeated for the composite's landing write:
    # the runner root is disposable, and what came home is a COPY.
    imported_file = os.path.join(run_directory, "records.jsonl")
    digest = _sha256(imported_file)
    assert runner.stop() == 0
    import shutil
    shutil.rmtree(runner_root)
    assert not os.path.exists(runner_root)
    assert _sha256(imported_file) == digest
    assert os.path.isfile(stamp_path)


def _manifest_bytes_named(root: str, name: str) -> bytes:
    path = os.path.join(root, "experiments", name, "experiment.json")
    if not os.path.exists(path):
        path = os.path.join(root, "experiments", f"{name}.json")
    with open(path, "rb") as handle:
        return handle.read()


# =============================================================================
# 5. The light-install guard, extended to the composite
# =============================================================================


#: What must NOT be imported to reach a runner. The same list Phases 1b and 2
#: guard their halves with.
HEAVY = ("torch", "transformers", "fastapi", "uvicorn", "peft", "sae_lens")

#: What the composite MAY pull: the package, the client's declared runtime
#: dependencies, and httpx's required closure. Union of Phase 1b's authoring
#: set and Phase 2's adapter set, because `run` is both.
ALLOWED_TOP_LEVEL = {
    "steerlab_server", "numpy", "safetensors",
    "httpx", "httpcore", "h11", "anyio", "certifi", "idna",
}

#: Present in a dev venv only, and tolerated by name — see
#: ``test_client_runner.py::test_httpx_needs_none_of_its_optional_cli_dependencies``,
#: which proves a bare client install resolves none of them.
HTTPX_OPTIONAL_CLI = {"click", "rich", "pygments", "markdown_it", "mdurl",
                      "typing_extensions"}


def test_the_run_verb_stays_within_the_clients_light_import_set(tmp_path):
    """CONTRACT: ``run`` composes LIGHT pieces plus the adapter.

    Out of process, for Phase 1b's reason: by the time this module runs, half
    the suite has imported torch, and an in-process assertion about
    ``sys.modules`` would pass or fail on test ORDER.

    The probe drives the verb far enough to matter — it authors and freezes a
    study, packages it, and then fails to reach a runner on a dead port — so
    the assertion covers stages 1, 2 and 3, which are exactly the stages that
    touch ``manifest``, ``experiment_store`` and ``bundles``. If it fails, the
    repair is to move the offending import INSIDE the code that needs it,
    never to weaken the guard.
    """
    workspace = tmp_path / "ws"
    (workspace / "prompts" / "concepts" / "signal").mkdir(parents=True)
    for filename, text in (("positive.jsonl", '{"text": "on"}\n'),
                           ("negative.jsonl", '{"text": "off"}\n')):
        (workspace / "prompts" / "concepts" / "signal" / filename).write_text(
            text, encoding="utf-8")

    probe = """
import io, json, sys, contextlib
from steerlab_server import client_cli
root = %(root)r
heavy = %(heavy)r

def step(argv, expect):
    with contextlib.redirect_stdout(io.StringIO()) as out, \\
            contextlib.redirect_stderr(io.StringIO()):
        code = client_cli.main(["--root", root] + argv)
    assert code == expect, (argv, code, out.getvalue())

step(["experiment", "create", "probe", "--model", "org/tiny",
      "--revision", "0" * 40], 0)
step(["experiment", "attach", "probe", "signal"], 0)
step(["experiment", "freeze", "probe", "--force"], 0)
# 127.0.0.1:1 refuses instantly: the whole machine runs — load, package, and
# a real request attempt — without a runner or a network.
step(["run", "probe", "--runner", "http://127.0.0.1:1",
      "--request-timeout", "2"], 70)

sys.__stderr__.write(json.dumps({
    "heavy": sorted(m for m in heavy if m in sys.modules),
    "topLevel": sorted({m.split(".")[0] for m in sys.modules
                        if not m.startswith("_")}
                       - set(sys.stdlib_module_names)),
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c",
         probe % {"root": str(workspace), "heavy": list(HEAVY)}],
        cwd=str(tmp_path), text=True, capture_output=True, check=False,
        env={**os.environ, "PYTHONPATH": SERVER_DIR,
             "STEERLAB_WORKSPACE": "", "HF_HUB_OFFLINE": "1"})
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stderr[proc.stderr.rindex("{"):])

    assert report["heavy"] == [], (
        "`steerlab run` pulled: " + ", ".join(report["heavy"]))
    unexpected = sorted(set(report["topLevel"])
                        - ALLOWED_TOP_LEVEL - HTTPX_OPTIONAL_CLI)
    assert unexpected == [], (
        "the composite pulled third-party modules outside the client's "
        "declared set: " + ", ".join(unexpected))
    assert "httpx" in report["topLevel"], \
        "the probe never reached the adapter — it proves nothing"
