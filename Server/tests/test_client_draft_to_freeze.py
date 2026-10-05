"""The app-free route from a draft to a frozen study, WITHOUT ``--force``.

The cross-platform client loads no model. Freeze asks for validation evidence
for any study with a concept or a condition, and that evidence can only be
produced where a model is. Until this file existed, ``steerlab run`` refused
every study that was not already frozen — so the client's only tested road to
a frozen study was ``freeze --force``, which stamps the study as not citable.

What changed, and what is pinned here:

* ``steerlab run <draft> --runner <url> --verb validate`` (and ``--verb
  extract``) hands a DRAFT to a runner. The evidence comes home through the
  same verified import as any other run, and the document says whether freeze's
  ``validateEvidence`` gate is now satisfied.
* The measured ``run`` and every verb after it still need a frozen study. The
  refusal says which verbs accept a draft, and names the route from THIS draft
  to a frozen one.

Two tiers, the same split ``test_run_orchestration.py`` makes:

1. **End to end, against a managed runner whose model is replaced.** A real
   ``steerlab runner serve`` subprocess on loopback, in token mode, with a root
   of its own. Everything a runner does is genuine — the upload route, the
   submission, the job, the import of the bundle into the runner's root, the
   validate and run workflows, the evidence file and its scope hash, the
   evidence archive, the download, the verified import. The one thing replaced
   is the model: the runner starts its batch children through
   ``$STEERLAB_PYTHON``, and here that is a launcher for
   ``model_free_engine_child.py``, which stands in for the arithmetic a GPU
   would have done and nothing else.
2. **Unit, with the scripted adapter.** The state machine's decisions about a
   draft: which verbs it may leave the workspace for, what the refusal says,
   and what the document reports when the evidence that came home is vacuous
   or is for different pins.

Fixtures are neutral throughout: one concept named ``signal`` with one-line
stimuli. Nothing here is about a study.
"""

import json
import os
import stat
import subprocess
import sys
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("httpx")

from steerlab_server import client_cli
from steerlab_server.client import runner as runner_api
from steerlab_server.experiment import bundles
from steerlab_server.experiment import experiment_store as store
from steerlab_server.experiment.manifest import Manifest

import model_free_engine_child
from test_local_runner import (  # noqa: F401 - fixtures used by name
    ManagedRunner,
    _await_info,
    _await_startup_envelope,
    _clean_env,
    _drain,
    _isolated_environment,
    _run_client,
)
from test_run_orchestration import FakeRunner, Script

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

STUDY = "journey"
CONCEPT = "signal"
MODEL = "org/tiny"
REVISION = "0" * 40
RUNNER_URL = "http://runner.invalid:8080"


# =============================================================================
# authoring, by the client's own verbs
# =============================================================================


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _author_inputs(workspace: str, *, held_out_probe: bool = True) -> None:
    """The files a person authors before the first verb: the concept's two
    poles, its held-out probe, and the task prompts the measured run reads."""
    directory = os.path.join(workspace, "prompts", "concepts", CONCEPT)
    _write(os.path.join(directory, "positive.jsonl"), '{"text": "on"}\n')
    _write(os.path.join(directory, "negative.jsonl"), '{"text": "off"}\n')
    if held_out_probe:
        _write(os.path.join(directory, "validation.jsonl"),
               '{"text": "a lamp is lit", "expresses": true}\n'
               '{"text": "a lamp is dark", "expresses": false}\n')
    _write(os.path.join(workspace, "incoming", "items.jsonl"),
           '{"id": "item-1", "prompt": "Describe the room."}\n'
           '{"id": "item-2", "prompt": "Describe the street."}\n')


def _ok(argv, capsys) -> dict:
    code, document = _run_client(argv, capsys)
    assert code == 0, document
    return document


def _author_draft(workspace: str, capsys, *, revision: str | None = REVISION,
                  held_out_probe: bool = True,
                  task_prompts: bool = True) -> None:
    """A draft with one concept, a baseline, and one steered arm — authored
    with the client's verbs, the way the workspace guide walks it."""
    _author_inputs(workspace, held_out_probe=held_out_probe)
    root = ["--root", workspace]
    create = [*root, "experiment", "create", STUDY, "--model", MODEL]
    if revision:
        create += ["--revision", revision]
    _ok(create, capsys)
    _ok([*root, "experiment", "attach", STUDY, CONCEPT], capsys)
    if task_prompts:
        inspected = _ok([*root, "experiment", "inspect", STUDY], capsys)
        _ok([*root, "experiment", "import-prompts", STUDY, "--file",
             os.path.join(workspace, "incoming", "items.jsonl"),
             "--manifest-sha256", inspected["result"]["manifestFileSHA256"]],
            capsys)
    _ok([*root, "experiment", "declare-condition", STUDY, "baseline",
         "--baseline", "--alpha-units", "raw"], capsys)
    _ok([*root, "experiment", "declare-condition", STUDY, "steered",
         "--slots", f"{CONCEPT}:2:1.0", "--alpha-units", "raw"], capsys)
    _ok([*root, "experiment", "verify", STUDY], capsys)


def _run_verb(argv, capsys):
    """Drive ``steerlab run`` under ``--json``: ``(code, document, stderr)``."""
    code = client_cli.main([*argv, "--json"])
    captured = capsys.readouterr()
    assert captured.out.count("\n}") == 1, captured.out
    return code, json.loads(captured.out), captured.err


def _stages(document: dict) -> dict:
    return {row["stage"]: row for row in document["result"]["stages"]}


def _runs(workspace: str) -> set:
    directory = os.path.join(workspace, "runs")
    return set(os.listdir(directory)) if os.path.isdir(directory) else set()


# =============================================================================
# 1. End to end, against a runner whose model is replaced
# =============================================================================


def _python_without_a_model(directory: str) -> str:
    """What the runner will be told to start its batch children with.

    A two-line launcher, because ``$STEERLAB_PYTHON`` names an executable. It
    runs the same interpreter the suite runs under, on
    ``model_free_engine_child.py``, with the runner's own arguments.
    """
    path = os.path.join(directory, "python-without-a-model")
    child = os.path.join(TESTS_DIR, "model_free_engine_child.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#!/bin/sh\n"
                     f'exec "{sys.executable}" "{child}" "$@"\n')
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def model_free_runner(tmp_path):
    """``steerlab runner serve``, launched the way ``test_local_runner.py``
    launches it, with one variable more: the engine starts its children with
    the model-free launcher. That variable is the engine's own seam
    (``api/submissions._bundle_execute_command``), so nothing is patched."""
    runner_root = str(tmp_path / "runner-root")
    process = subprocess.Popen(
        [sys.executable, "-m", "steerlab_server.client_cli",
         "runner", "serve", "--runner-root", runner_root, "--json"],
        env=_clean_env(STEERLAB_PYTHON=_python_without_a_model(str(tmp_path))),
        cwd=str(tmp_path), text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    runner = ManagedRunner(process, [], [])
    for stream, sink in ((process.stdout, runner.out),
                         (process.stderr, runner.err)):
        threading.Thread(target=_drain, args=(stream, sink),
                         daemon=True).start()
    try:
        runner.envelope = _await_startup_envelope(runner, deadline=180.0)
        assert _await_info(runner.url, runner, deadline=60.0) in (200, 401)
        yield {"runner": runner, "root": runner_root,
               "connection": ["--runner", runner.url, "--token-file",
                              runner.result["tokenFile"]]}
    finally:
        runner.stop()


def test_a_draft_reaches_a_frozen_study_without_force_and_then_runs(
        model_free_runner, tmp_path, capsys):
    """CONTRACT: draft → validate on a runner → import → freeze WITHOUT force
    → measured run → import. The whole app-free journey, on this client's
    verbs alone.

    The study is created with NO model revision, which is how a researcher who
    does not know a commit id starts. The runner resolves one when it loads
    the model, and importing its evidence pins that commit into the local
    draft — so the ``revision`` gate is answered by the same step that answers
    ``validateEvidence``.
    """
    connection = model_free_runner["connection"]
    runner_root = model_free_runner["root"]
    workspace = str(tmp_path / "workspace")
    root = ["--root", workspace]
    _ok(["workspace", "init", workspace], capsys)
    _author_draft(workspace, capsys, revision=None)

    # -- the draft cannot freeze yet, and the refusal names this client's
    #    route to the evidence it is missing --------------------------------
    code, document = _run_client([*root, "experiment", "freeze", STUDY],
                                 capsys)
    assert code == 65, document
    assert document["error"]["code"] == "freezeGateFailed"
    assert set(document["error"]["gates"]) == {"revision", "validateEvidence"}
    assert (f"steerlab run {STUDY} --runner <url> --verb validate"
            in document["error"]["repairAction"])
    assert store.load_raw(STUDY, workspace)["status"] == "draft"

    # -- validate the DRAFT on the runner -----------------------------------
    runs_before = _runs(workspace)
    code, document, stderr = _run_verb(
        [*root, "run", STUDY, *connection, "--verb", "validate"], capsys)
    assert code == 0, (document, stderr)
    result = document["result"]
    assert result["outcome"] == "succeeded"
    assert result["studyVerb"] == "validate"
    stages = _stages(document)
    assert all(stages[name]["state"] == client_cli.STAGE_OK
               for name in client_cli.RUN_STAGES), stages
    assert stages["load"]["status"] == "draft"

    # The evidence is what the ENGINE wrote, in the runner's root, from the
    # bundle it was handed — and it came home into runs/, nowhere else.
    validate_run = result["importedRunDirectory"]
    assert os.path.basename(validate_run).endswith(f"exp-{STUDY}-validate")
    assert os.path.realpath(validate_run).startswith(
        os.path.realpath(os.path.join(workspace, "runs")))
    assert os.path.isfile(os.path.join(
        runner_root, "runs", os.path.basename(validate_run),
        "validation-evidence.json"))
    with open(os.path.join(validate_run, "validation-evidence.json"),
              encoding="utf-8") as handle:
        evidence = json.load(handle)
    assert evidence["task"] == "validate"
    assert evidence["substrate"] == "python-hf-transformers"
    assert evidence["vacuousConcepts"] == []

    # The runner resolved a commit for the unpinned draft; the import pinned
    # it here, and said so.
    assert result["revisionAdoption"]["outcome"] == "adopted"
    assert result["revisionAdoption"]["revision"] == \
        model_free_engine_child.RESOLVED_REVISION
    assert [a["code"] for a in document["advisories"]] == ["revisionAdoption"]
    draft = store.load_raw(STUDY, workspace)
    assert draft["status"] == "draft"
    assert draft["modelRevision"] == model_free_engine_child.RESOLVED_REVISION

    # The document says the gate is satisfied, by the store's own reading, and
    # the scope the evidence carries IS the draft's.
    assert result["vacuous"] is False
    assert result["validation"][0]["concept"] == CONCEPT
    gate = result["validateEvidence"]
    assert gate == {"needed": True, "present": True, "satisfied": True,
                    "validationScopeHash": evidence["validationScopeHash"]}
    assert document["nextAction"]["verb"] == f"experiment freeze {STUDY}"

    # -- freeze, WITHOUT --force --------------------------------------------
    code, document = _run_client([*root, "experiment", "freeze", STUDY],
                                 capsys)
    assert code == 0, document
    assert document["state"] == "ready"
    assert document["result"]["forced"] is False
    frozen = store.load_raw(STUDY, workspace)
    assert frozen["status"] == "frozen"
    assert frozen["freezeHash"] == document["result"]["freezeHash"]
    assert "freezeForced" not in frozen
    assert "forcedGatesSkipped" not in frozen
    frozen_bytes = _manifest_bytes(workspace)

    # -- the measured run, on the frozen study ------------------------------
    code, document, stderr = _run_verb([*root, "run", STUDY, *connection],
                                       capsys)
    assert code == 0, (document, stderr)
    result = document["result"]
    assert result["outcome"] == "succeeded"
    assert result["studyVerb"] == "run"
    measured = result["importedRunDirectory"]
    assert os.path.basename(measured).endswith(f"exp-{STUDY}-run")
    with open(os.path.join(measured, "generations.jsonl"),
              encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    assert {r["condition"] for r in records} == {"baseline", "steered"}
    assert {r["promptID"] for r in records} == {"item-1", "item-2"}
    with open(os.path.join(measured, client_cli.PROVENANCE_FILENAME),
              encoding="utf-8") as handle:
        stamp = json.load(handle)
    assert stamp["manifest"]["status"] == "frozen"
    assert stamp["manifest"]["freezeHash"] == frozen["freezeHash"]
    assert stamp["manifest"]["freezeForced"] is False

    # The next step is the one that reads a measured run, on this client.
    # (It used to be `bundle inspect <run directory>`, which reads an archive
    # and fails on a directory.)
    assert document["nextAction"]["verb"] == \
        f"run {STUDY} --runner {model_free_runner['runner'].url} " \
        "--verb analyze"
    assert document["nextAction"]["requiresHuman"] is False

    # The frozen study did not move, and the workspace gained exactly what
    # was imported plus the two packaged bundles.
    assert _manifest_bytes(workspace) == frozen_bytes
    gained = _runs(workspace) - runs_before
    assert {name for name in gained if "bundle-" not in name} == {
        os.path.basename(validate_run), os.path.basename(measured)}


def test_the_same_route_one_step_at_a_time(model_free_runner, tmp_path,
                                           capsys):
    """CONTRACT: the composite adds nothing the separate verbs cannot do.

    ``bundle package`` → ``runner upload`` → ``runner submit --verb validate``
    → ``runner jobs`` → ``runner evidence`` → ``bundle import`` → ``experiment
    freeze``, on a draft, with no ``--force``. This is the route the workspace
    guide shows beside ``run --verb validate``, and it is what a caller who
    detached with ``--no-wait`` continues with.
    """
    connection = model_free_runner["connection"]
    workspace = str(tmp_path / "workspace")
    root = ["--root", workspace]
    os.makedirs(workspace)
    _author_draft(workspace, capsys)

    packaged = _ok([*root, "bundle", "package", STUDY], capsys)["result"]
    uploaded = _ok(["runner", "upload", packaged["bundlePath"], *connection],
                   capsys)["result"]
    submitted = _ok(
        ["runner", "submit", *connection, "--bundle-path",
         uploaded["runnerPath"], "--bundle-sha", uploaded["sha256"],
         "--verb", "validate"], capsys)["result"]
    job_id = submitted["jobId"]

    from steerlab_server.api.jobs import TERMINAL
    limit = time.monotonic() + 180.0
    while True:
        job = _ok(["runner", "jobs", job_id, *connection],
                  capsys)["result"]["job"]
        if job["status"] in TERMINAL:
            break
        assert time.monotonic() < limit, job
        time.sleep(0.05)
    assert job["status"] == "succeeded", job

    archive = str(tmp_path / "home" / "validation.tar.gz")
    fetched = _ok(["runner", "evidence", job_id, "--out", archive,
                   *connection], capsys)["result"]
    assert fetched["verified"] is True and fetched["imported"] is False
    _ok([*root, "bundle", "import", archive, "--sha256", fetched["sha256"]],
        capsys)

    assert store.validate_evidence_readiness(STUDY, workspace)["satisfied"]
    frozen = _ok([*root, "experiment", "freeze", STUDY], capsys)["result"]
    assert frozen["forced"] is False
    assert "freezeForced" not in store.load_raw(STUDY, workspace)


def _manifest_bytes(workspace: str) -> bytes:
    with open(os.path.join(workspace, "experiments", STUDY,
                           "experiment.json"), "rb") as handle:
        return handle.read()


# =============================================================================
# 2. The state machine's decisions about a draft (scripted adapter, no sockets)
# =============================================================================


@pytest.fixture
def draft(tmp_path, capsys):
    """A workspace holding ONE draft, authored by the client's verbs."""
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace)
    _author_draft(workspace, capsys)
    return workspace


@pytest.fixture
def script(monkeypatch):
    """The scripted adapter of ``test_run_orchestration.py``, installed over
    the runner client. Only the class is swapped: parsing, envelope, bundles,
    and workspace are all real, and nothing touches a socket."""
    scripted = Script()
    monkeypatch.setattr(runner_api, "RunnerClient",
                        lambda **kwargs: FakeRunner(scripted, **kwargs))
    monkeypatch.setattr(client_cli, "POLL_INTERVAL_START", 0.0)
    monkeypatch.setattr(client_cli, "POLL_INTERVAL_CAP", 0.0)
    return scripted


def _validation_archive(tmp_path, workspace: str, *, vacuous=(),
                        scope_hash: str | None = None) -> dict:
    """A validate run directory with the files freeze reads, packaged by the
    real packager — the evidence a runner would hand back for this draft.

    ``scope_hash`` defaults to the draft's own, which is what a runner that
    validated this draft's bundle stamps."""
    manifest = Manifest.load(STUDY, workspace)
    run_directory = str(tmp_path / "runner-side-runs"
                        / f"20260101T000000000-exp-{STUDY}-validate")
    _write(os.path.join(run_directory, "validation-evidence.json"),
           json.dumps({
               "schemaVersion": 1, "task": "validate", "experiment": STUDY,
               "substrate": "python-hf-transformers",
               "reportFile": "validation-report.json",
               "validationScopeHash": (scope_hash
                                       or manifest.validation_scope_hash()),
               "vacuousConcepts": sorted(vacuous)}))
    _write(os.path.join(run_directory, "validation-report.json"),
           json.dumps({
               "experiment": STUDY,
               "concepts": {CONCEPT: {"scenarioAccuracy": 0.9,
                                      "scenarioCount": 2}},
               "vacuousConcepts": sorted(vacuous)}))
    _write(os.path.join(run_directory, "experiment.json"),
           json.dumps(store.load_raw(STUDY, workspace)))
    meta = bundles.package_evidence(run_directory)
    return {"bundlePath": meta["bundlePath"],
            "bundleSha256": meta["bundleSha256"],
            "experiment": STUDY, "runID": meta["runID"]}


def _hand_back(script: Script, reference: dict) -> None:
    script.evidence = reference
    script.archive_source = reference["bundlePath"]


def test_a_draft_is_refused_for_the_measured_run_and_told_what_accepts_one(
        draft, script, capsys):
    """CONTRACT: the frozen-only rule stays for the measured run, and the
    refusal is a route rather than a wall.

    It says which steps a draft is accepted for, and its repair is three
    commands on this client, in order, for THIS draft: validate on the runner
    the caller already named, freeze, then the run that was asked for.
    """
    code, document, _ = _run_verb(
        ["--root", draft, "run", STUDY, "--runner", RUNNER_URL], capsys)

    assert code == 65, document
    error = document["error"]
    assert error["code"] == client_cli.NOT_FROZEN_CODE
    assert "extract and validate" in error["reason"]
    assert document["result"]["draftVerbs"] == ["extract", "validate"]
    repair = error["repairAction"]
    validate = f"steerlab run {STUDY} --runner {RUNNER_URL} --verb validate"
    freeze = f"steerlab experiment freeze {STUDY}"
    measured = f"steerlab run {STUDY} --runner {RUNNER_URL}"
    assert repair.index(validate) < repair.index(freeze) \
        < repair.rindex(measured)
    assert repair.endswith(measured), repair

    assert document["result"]["failedStage"] == "load"
    # Refused locally: the runner was never asked who it is.
    assert script.info_calls == 0
    assert script.uploads == [] and script.submits == []


@pytest.mark.parametrize("verb", sorted(
    set(client_cli.RUN_STUDY_VERBS) - set(client_cli.DRAFT_STUDY_VERBS)))
def test_every_step_after_freeze_still_needs_a_frozen_study(
        draft, script, capsys, verb):
    """The allowance is exactly two verbs wide. Everything that measures, or
    reads a measurement, stamps its output with a freeze hash."""
    code, document, _ = _run_verb(
        ["--root", draft, "run", STUDY, "--runner", RUNNER_URL, "--verb",
         verb], capsys)

    assert code == 65, document
    assert document["error"]["code"] == client_cli.NOT_FROZEN_CODE
    assert f"'{verb}' step needs a frozen study" in document["error"]["reason"]
    assert script.uploads == [] and script.submits == []


def test_the_route_skips_validation_once_the_evidence_is_home(
        draft, script, tmp_path, capsys):
    """The repair is read from the draft, not recited: with matching
    validation evidence already in the workspace, the next step is freeze."""
    reference = _validation_archive(tmp_path, draft)
    bundles.import_bundle(reference["bundlePath"], target_root=draft,
                          expected_sha256=reference["bundleSha256"])

    code, document, _ = _run_verb(
        ["--root", draft, "run", STUDY, "--runner", RUNNER_URL], capsys)

    assert code == 65, document
    repair = document["error"]["repairAction"]
    assert "--verb validate" not in repair
    assert f"steerlab experiment freeze {STUDY}" in repair


@pytest.mark.parametrize("verb", client_cli.DRAFT_STUDY_VERBS)
def test_a_draft_is_handed_to_the_runner_for_the_steps_before_freeze(
        draft, script, tmp_path, capsys, verb):
    """CONTRACT: ``validate`` and ``extract`` accept a draft, and the bundle
    the runner receives is the draft's own."""
    _hand_back(script, _validation_archive(tmp_path, draft))
    before = _manifest_bytes(draft)

    code, document, stderr = _run_verb(
        ["--root", draft, "run", STUDY, "--runner", RUNNER_URL, "--verb",
         verb], capsys)

    assert code == 0, document
    stages = _stages(document)
    assert stages["load"]["state"] == client_cli.STAGE_OK
    assert stages["load"]["status"] == "draft"
    assert "is a draft" in stderr and "comes before freeze" in stderr
    assert len(script.uploads) == 1
    assert script.submits[0]["verb"] == verb
    meta = bundles.inspect_bundle(script.uploads[0]["path"])
    assert meta["experimentContentHash"] == \
        Manifest.load(STUDY, draft).content_hash()
    # Handing a draft to a runner changes nothing about the draft.
    assert _manifest_bytes(draft) == before
    assert document["result"]["imported"] is True
    assert "still a draft" in document["message"]
    # The run directory is stamped with what was handed over: a draft.
    with open(os.path.join(document["result"]["importedRunDirectory"],
                           client_cli.PROVENANCE_FILENAME),
              encoding="utf-8") as handle:
        stamp = json.load(handle)
    assert stamp["manifest"]["status"] == "draft"
    assert stamp["manifest"]["freezeHash"] is None


def test_validation_that_came_home_points_at_freeze(draft, script, tmp_path,
                                                    capsys):
    """CONTRACT: the document answers the question the caller has next — will
    freeze accept this? — and the store that owns the gate is who answers."""
    reference = _validation_archive(tmp_path, draft)
    _hand_back(script, reference)

    code, document, _ = _run_verb(
        ["--root", draft, "run", STUDY, "--runner", RUNNER_URL, "--verb",
         "validate"], capsys)

    assert code == 0, document
    assert document["state"] == "ready"
    result = document["result"]
    assert result["vacuous"] is False and result["vacuousConcepts"] == []
    assert result["validation"] == [{
        "concept": CONCEPT, "accuracy": 0.9, "scenarios": 2,
        "oneSidedPredictions": False, "atOrBelowChance": False}]
    assert result["validateEvidence"]["satisfied"] is True
    assert result["revisionAdoption"]["outcome"] == "alreadyPinned"
    assert document["nextAction"]["verb"] == f"experiment freeze {STUDY}"
    assert document["nextAction"]["requiresHuman"] is False

    # And it is true: the very next freeze, unforced, succeeds.
    frozen = _ok(["--root", draft, "experiment", "freeze", STUDY],
                 capsys)["result"]
    assert frozen["forced"] is False


def test_vacuous_validation_is_said_at_once_and_does_not_point_at_freeze(
        draft, script, tmp_path, capsys):
    """CONTRACT: a validation that scored no held-out probe exits 0 on the
    runner and looks the same on the surface. The client says so when the
    evidence lands, in the engine's own vocabulary, instead of leaving the
    researcher to find out from the freeze refusal."""
    _hand_back(script, _validation_archive(tmp_path, draft,
                                           vacuous=[CONCEPT]))

    code, document, _ = _run_verb(
        ["--root", draft, "run", STUDY, "--runner", RUNNER_URL, "--verb",
         "validate"], capsys)

    assert code == 0, document
    assert document["state"] == "okWithAdvisories"
    result = document["result"]
    assert result["vacuous"] is True
    assert result["vacuousConcepts"] == [CONCEPT]
    assert [a["code"] for a in document["advisories"]] == ["vacuousValidation"]
    gate = result["validateEvidence"]
    assert gate["present"] is True and gate["satisfied"] is False
    assert "VACUOUS" in gate["problem"]
    assert f"prompts/concepts/{CONCEPT}/validation.jsonl" in gate["problem"]
    # The gate's sentence names this client's route, not the engine's verb.
    assert "steerlab-server" not in gate["problem"]
    assert f"steerlab run {STUDY} --runner <url> --verb validate" \
        in gate["problem"]
    action = document["nextAction"]
    assert action["requiresHuman"] is True
    assert action["verb"] == \
        f"run {STUDY} --runner {RUNNER_URL} --verb validate"

    # And freeze agrees, under the same gate.
    code, document = _run_client(["--root", draft, "experiment", "freeze",
                                  STUDY], capsys)
    assert code == 65, document
    assert document["error"]["gate"] == "validateEvidence"


def test_validation_for_different_pins_is_reported_as_not_matching(
        draft, script, tmp_path, capsys):
    """Evidence is keyed by the study's pins. An archive whose scope is not
    this draft's is imported — it is a real run — and reported as what it is:
    not the evidence freeze is waiting for."""
    _hand_back(script, _validation_archive(tmp_path, draft,
                                           scope_hash="c" * 64))

    code, document, _ = _run_verb(
        ["--root", draft, "run", STUDY, "--runner", RUNNER_URL, "--verb",
         "validate"], capsys)

    assert code == 0, document
    gate = document["result"]["validateEvidence"]
    assert gate["present"] is False and gate["satisfied"] is False
    assert gate["validationScopeHash"] == \
        Manifest.load(STUDY, draft).validation_scope_hash()
    assert document["nextAction"]["verb"] == f"experiment verify {STUDY}"


def test_a_draft_whose_pins_drifted_is_refused_before_the_runner_is_asked(
        draft, script, capsys):
    """Everything checkable happens before the upload, for a draft too:
    evidence validated against bytes that are not the pinned ones could never
    satisfy the gate, and the compute would be spent finding that out."""
    _write(os.path.join(draft, "prompts", "concepts", CONCEPT,
                        "positive.jsonl"), '{"text": "changed"}\n')

    code, document, _ = _run_verb(
        ["--root", draft, "run", STUDY, "--runner", RUNNER_URL, "--verb",
         "validate"], capsys)

    assert code == 65, document
    assert document["result"]["failedStage"] == "load"
    assert document["result"]["violations"]
    assert script.info_calls == 0 and script.uploads == []


def test_a_forced_freeze_is_still_available_and_still_stamped(draft, capsys):
    """``--force`` is a choice a researcher may make, and it stays loud: the
    study is frozen, and it says which gates it skipped. It is no longer the
    only way to a frozen study (the tests above), which is the point."""
    document = _ok(["--root", draft, "experiment", "freeze", STUDY,
                    "--force"], capsys)

    assert document["result"]["forced"] is True
    assert [a["code"] for a in document["advisories"]] == ["freezeGateSkipped"]
    frozen = store.load_raw(STUDY, draft)
    assert frozen["freezeForced"] is True
    assert frozen["forcedGatesSkipped"] == ["validateEvidence"]


def test_readiness_asks_the_gates_own_questions(draft, tmp_path):
    """``validate_evidence_readiness`` is not a second opinion about the gate:
    it is satisfied exactly when freeze's ``validateEvidence`` gate passes."""
    def gate_fails() -> bool:
        document = store.load_raw(STUDY, draft)
        manifest = Manifest.from_dict(document)
        failures = store._evaluate_freeze_gates(STUDY, document, manifest,
                                                draft)
        return any(gate == "validateEvidence" for gate, _ in failures)

    readiness = store.validate_evidence_readiness(STUDY, draft)
    assert readiness["needed"] and not readiness["present"]
    assert not readiness["satisfied"] and gate_fails()

    reference = _validation_archive(tmp_path, draft)
    bundles.import_bundle(reference["bundlePath"], target_root=draft,
                          expected_sha256=reference["bundleSha256"])
    readiness = store.validate_evidence_readiness(STUDY, draft)
    assert readiness["present"] and readiness["satisfied"]
    assert not gate_fails()

    # A study with nothing to validate owes nothing.
    store.create("bare", model_id=MODEL, revision=REVISION, root=draft)
    assert store.validate_evidence_readiness("bare", draft) == {
        "needed": False, "present": False, "satisfied": True,
        "validationScopeHash":
            Manifest.load("bare", draft).validation_scope_hash()}


def test_a_measured_run_with_no_task_prompts_is_refused_before_upload(
        tmp_path, script, capsys):
    """Freeze does not ask for task prompts (a multi-agent study has none),
    so a study can freeze without them — and the runner then refused the run
    after the bundle had been uploaded and scheduled, with a bare error and
    no repair. The client asks first, in the engine's own words, and the
    repair is the route a frozen study has: a duplicate, which is a draft."""
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace)
    _author_draft(workspace, capsys, task_prompts=False)
    _ok(["--root", workspace, "experiment", "freeze", STUDY, "--force"],
        capsys)

    code, document, _ = _run_verb(
        ["--root", workspace, "run", STUDY, "--runner", RUNNER_URL], capsys)

    assert code == 65, document
    error = document["error"]
    assert error["code"] == "missingPrerequisite"
    assert error["gate"] == "missingPrerequisite"
    assert "pins no task prompts" in error["reason"]
    assert error["repairAction"].startswith(
        f"steerlab experiment duplicate {STUDY} {STUDY}-v2 && steerlab "
        f"experiment import-prompts {STUDY}-v2 --file ")
    assert "steerlab-cli" not in error["repairAction"]
    assert document["result"]["failedStage"] == "load"
    assert script.info_calls == 0 and script.uploads == []

    # A step that reads no task prompts is not refused for their absence.
    script.evidence = None
    code, document, _ = _run_verb(
        ["--root", workspace, "run", STUDY, "--runner", RUNNER_URL,
         "--verb", "verify"], capsys)
    assert document["error"]["code"] == client_cli.NO_EVIDENCE_CODE
    assert len(script.uploads) == 1


def test_validating_a_draft_with_no_task_prompts_points_at_pinning_them(
        tmp_path, script, capsys):
    """Freeze would succeed here and leave a study that cannot be measured,
    because a frozen study cannot gain task prompts. So when the validation
    comes home, the next action is pinning them, not freezing."""
    workspace = str(tmp_path / "workspace")
    os.makedirs(workspace)
    _author_draft(workspace, capsys, task_prompts=False)
    _hand_back(script, _validation_archive(tmp_path, workspace))

    code, document, _ = _run_verb(
        ["--root", workspace, "run", STUDY, "--runner", RUNNER_URL, "--verb",
         "validate"], capsys)

    assert code == 0, document
    result = document["result"]
    assert result["validateEvidence"]["satisfied"] is True
    assert result["taskPromptsPinned"] is False
    action = document["nextAction"]
    assert action["verb"] == f"experiment inspect {STUDY}"
    assert f"steerlab experiment import-prompts {STUDY}" in action["detail"]


def test_after_a_judged_measured_run_the_next_step_asks_for_a_person(
        draft, monkeypatch):
    """A study that declares judges is coded before it is analyzed, and
    coding spends compute or provider credit — so that step asks first."""
    monkeypatch.setenv("STEERLAB_ROOT", draft)
    action = client_cli._after_frozen_step(
        STUDY, "run", run_directory="runs/x", runner_url=RUNNER_URL)
    assert action["verb"] == \
        f"run {STUDY} --runner {RUNNER_URL} --verb analyze"
    assert action["requiresHuman"] is False

    document = store.load_raw(STUDY, draft)
    document["judges"] = [{"name": "a", "kind": "claude"}]
    store.save_raw(document, draft)
    action = client_cli._after_frozen_step(
        STUDY, "run", run_directory="runs/x", runner_url=RUNNER_URL)
    assert action["verb"] == \
        f"run {STUDY} --runner {RUNNER_URL} --verb evaluate"
    assert action["requiresHuman"] is True

    # Any other step ends the round trip.
    assert client_cli._after_frozen_step(
        STUDY, "analyze", run_directory="runs/x",
        runner_url=RUNNER_URL)["verb"] == "experiment list"
