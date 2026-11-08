"""A study its settings alone make unrunnable is refused before the queue.

Public issue #9: a temperature-0 study that kept a list of several seeds
passed verify and freeze, waited for a GPU, loaded its model, and only then
refused at run start, because at temperature 0 every seed gives the same
greedy response. The rule now has one sentence on both engines
(``sampling.greedy_seed_list_problem``) and applies only to a study that
generates sampled text: an answer-token readout reads each item once
whatever the seeds, so a readout-only study runs and is told its list is not
used. The rule fires at verify on a draft (so freeze refuses), at the engine's
submission preflight before anything is written or queued (a study frozen
under an older release meets it there, and verify warns of it with
``refusedWhenRun``), at a chain's start when the chain runs the study, and at
run start before the model loads.

The other refusals of run start that depend on the settings alone are asked
at the same places through the same function
(``study_admission.settings_refusal``): several samples per item at
temperature 0, and an ordinal scale with no declared aggregation. The
submission also asks run start's pinned-prompt rules (an answer-token
instrument over items it cannot read, an attention-check exclusion with no
checked item), a ``pipeline`` submission's block, and, for ``run`` only, a
concept study that declares no arm at all (a chain mints its arm at promote).
"""

import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from steerlab_server import cli_envelope, client_cli
from steerlab_server.api.jobs import DurableJobStore, JobManager
from steerlab_server.api import submissions
from steerlab_server.experiment import (bundles, lifecycle_gates, run_workflow,
                                        sampling, study_admission, tasks)
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import verification_refusal as refusal
from steerlab_server.experiment.manifest import Manifest
from test_run_orchestration import evidence_archive, script  # noqa: F401 - fixtures

GREEDY_THREE = sampling.greedy_seed_list_problem(0.0, [11, 22, 33])


def _manifest(**fields):
    return Manifest.from_dict({"name": "demo", "modelID": "org/m",
                               "concepts": [], **fields})


# --- the rule and its words ------------------------------------------------------

def test_the_sentence_says_what_is_wrong_and_how_to_repair_it():
    assert GREEDY_THREE == (
        "at temperature 0 decoding is greedy, so every seed gives the same "
        "response, and the study's 3 seeds would write 3 identical records "
        "per item. Set one seed, or raise the temperature above 0 to sample")
    # Never read as drift, which would send the reader to restore files.
    assert refusal.gate([GREEDY_THREE]) == "studyDeclaration"


@pytest.mark.parametrize("policy", ["manifestSeeds", "derivedSHA256", None])
def test_verify_refuses_a_greedy_draft_with_several_seeds(policy):
    fields = {"temperature": 0.0, "seeds": [11, 22, 33]}
    if policy:
        fields["seedPolicy"] = policy
    assert GREEDY_THREE in _manifest(**fields).verify()
    # One seed, a temperature above 0, or a panel: no such violation.
    assert not any(v.startswith(sampling.GREEDY_SEED_LIST_PREFIX)
                   for v in _manifest(**{**fields, "seeds": [11]}).verify())
    assert not any(v.startswith(sampling.GREEDY_SEED_LIST_PREFIX)
                   for v in _manifest(**{**fields, "temperature": 0.7}).verify())
    assert sampling.greedy_seed_list_problem(
        0.0, [11, 22, 33], study_kind="multiAgent") is None


def test_a_frozen_study_still_verifies_so_its_runs_can_be_read():
    """verify also admits analysis and evaluation of runs that exist. A study
    frozen under an older release keeps verifying; it is refused where it
    would RUN, with a repair that duplicates it first."""
    frozen = _manifest(temperature=0.0, seeds=[11, 22, 33], status="frozen")
    assert GREEDY_THREE not in frozen.verify()
    refused = study_admission.settings_refusal(frozen)
    assert str(refused) == GREEDY_THREE
    assert refused.gate == lifecycle_gates.STUDY_DECLARATION
    assert refused.repair_action.startswith(
        "on your authoring client: steerlab-cli experiment duplicate demo "
        "demo-v2 && steerlab-cli experiment set-sampling demo-v2 --seeds <n>")


def test_the_advisory_never_calls_a_refused_setting_one_with_no_effect():
    # Refused: nothing is said about the policy.
    assert sampling.unused_seed_settings(0.0, 1, "derivedSHA256",
                                         [11, 22, 33]) == []
    # Still true, still said: derivedSHA256 at temperature 0 with one seed …
    [greedy] = sampling.unused_seed_settings(0.0, 1, "derivedSHA256", [11])
    assert "has no effect at temperature 0" in greedy
    # … and a list of several seeds beside derivedSHA256 above temperature 0.
    [warm] = sampling.unused_seed_settings(0.7, 1, "derivedSHA256",
                                           [11, 22, 33])
    assert "(3 seeds) is not used" in warm


def test_the_verify_repair_names_the_verb_of_the_client_reading_it():
    assert refusal.repair("demo", [GREEDY_THREE], program="steerlab").startswith(
        "Set one seed: steerlab experiment set-protocol demo --set "
        "seeds='[<n>]', or raise the temperature above 0 to sample. ")
    assert refusal.repair(
        "demo", [GREEDY_THREE], program="steerlab-cli").startswith(
        "Set one seed: steerlab-cli experiment set-sampling demo --seeds <n>, "
        "or raise the temperature above 0 to sample. ")


# --- the client: verify, freeze, and the repair verb ------------------------------

@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    monkeypatch.setenv(client_cli.WORKSPACE_ENV, str(root))
    monkeypatch.setattr(cli_envelope, "now", lambda: 1_000.0)
    return root


def _client(argv, capsys):
    code = client_cli.main([*argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


def _verifiable_draft(workspace, capsys):
    pairs = workspace / "french-pairs.jsonl"
    pairs.write_text('{"positive": "bonjour", "negative": "hello"}\n',
                     encoding="utf-8")
    for argv in (["concept", "import", "french", "--file", str(pairs)],
                 ["experiment", "create", "demo", "--model", "org/m",
                  "--revision", "0123456789abcdef0123456789abcdef01234567"],
                 ["experiment", "attach", "demo", "french"],
                 ["experiment", "declare-condition", "demo", "baseline",
                  "--baseline", "--alpha-units", "norm"]):
        assert client_cli.main(argv) == 0, argv
    capsys.readouterr()


def test_client_verify_and_freeze_refuse_and_the_repair_verb_clears_it(
        workspace, capsys):
    _verifiable_draft(workspace, capsys)
    code, _ = _client(["experiment", "set-protocol", "demo", "--set",
                       "seeds=[11,22,33]", "--set", "temperature=0"], capsys)
    assert code == 0
    for verb in ("verify", "freeze"):
        code, document = _client(["experiment", verb, "demo"], capsys)
        assert code == 65, (verb, document)
        assert document["error"]["code"] == "studyDeclaration", verb
        said = (document.get("result") or {}).get("violations") or []
        assert (GREEDY_THREE in said
                or GREEDY_THREE in document["error"]["reason"]), verb
        assert ("steerlab experiment set-protocol demo --set seeds='[<n>]'"
                in document["error"]["repairAction"]), verb
    assert es.load_raw("demo", str(workspace))["status"] == "draft"
    # The repair, as written (the shell quotes removed), verifies.
    code, _ = _client(["experiment", "set-protocol", "demo", "--set",
                       "seeds=[11]"], capsys)
    assert code == 0
    code, verified = _client(["experiment", "verify", "demo"], capsys)
    assert code == 0, verified
    assert es.load_raw("demo", str(workspace))["seeds"] == [11]


@pytest.mark.parametrize("value, words", [
    ("[]", "the seed list cannot be empty"),
    ("[7,7]", "duplicate seeds"),
    ("[1.5]", "whole numbers"),
    ("[-1]", "whole numbers"),
    ("[true]", "whole numbers"),
    ("5", "whole numbers"),
    ('["x"]', "whole numbers"),
    (f"[{2 ** 64}]", "whole numbers"),
])
def test_set_protocol_keeps_the_apps_seed_list_rules(workspace, capsys, value,
                                                      words):
    _verifiable_draft(workspace, capsys)
    code, document = _client(["experiment", "set-protocol", "demo", "--set",
                              f"seeds={value}"], capsys)
    assert code != 0, document
    assert words in document["error"]["reason"]
    # Nothing was written, and the manifest still loads.
    assert es.load_raw("demo", str(workspace)).get("seeds") == [0]
    Manifest.load("demo", str(workspace))
    # Order is kept.
    code, _ = _client(["experiment", "set-protocol", "demo", "--set",
                       "seeds=[33,11,22]", "--set", "temperature=0.7"], capsys)
    assert code == 0
    assert es.load_raw("demo", str(workspace))["seeds"] == [33, 11, 22]


# --- run start: before the model loads --------------------------------------------

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _study(root, name, **fields):
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "calm"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    raw = es.load_raw(name, root)
    raw.update({"maxTokens": 16, "outcomeInstruments": ["sampledText"],
                "conditions": [{"name": "fear-on", "alphaInNormUnits": True,
                                "slots": [{"concept": "fear", "layer": 1,
                                           "alpha": 0.1}]}]})
    raw.update(fields)
    es.save_raw(raw, root)
    prompts = os.path.join(root, "prompts", "tasks", "items.jsonl")
    _write(prompts, json.dumps({"id": "item-1", "prompt": "Decide."}) + "\n")
    return prompts


def _frozen_under_an_older_release(root, name, monkeypatch, **fields):
    """A study frozen by a release without the rule: its freeze hash is
    sound, and only today's release refuses its settings."""
    prompts = _study(root, name, **fields)
    with monkeypatch.context() as older:
        older.setattr(sampling, "greedy_seed_list_problem",
                      lambda *args, **kwargs: None)
        older.setattr(sampling, "manifest_greedy_seed_list_problem",
                      lambda manifest: None)
        es.freeze(name, force=True, root=root)
    assert es.load_raw(name, root)["status"] == "frozen"
    assert Manifest.load(name, root).verify(root) == []
    return prompts


@pytest.mark.parametrize("frozen", [False, True])
def test_run_start_refuses_before_the_model_loads(tmp_path, monkeypatch, frozen):
    root = str(tmp_path)
    fields = {"temperature": 0.0, "seeds": [11, 22, 33]}
    prompts = (_frozen_under_an_older_release(root, "greedy", monkeypatch, **fields)
               if frozen else _study(root, "greedy", **fields))
    loads = []

    @contextmanager
    def counting_model(model_id, revision=None):
        loads.append(model_id)
        yield SimpleNamespace(model_id=model_id, revision=revision or "abc")

    with pytest.raises(lifecycle_gates.LifecycleError) as refused:
        tasks.run("greedy", prompts, root, model_provider=counting_model,
                  log=lambda *_: None)
    assert str(refused.value) == GREEDY_THREE
    assert refused.value.gate == "studyDeclaration"
    assert ("duplicate greedy greedy-v2" in refused.value.repair_action) is frozen
    assert loads == []
    assert not os.path.isdir(os.path.join(root, "runs")) or not [
        d for d in os.listdir(os.path.join(root, "runs")) if "greedy-run" in d]


def test_the_run_backstop_refuses_a_direct_caller_with_the_same_sentence(
        tmp_path):
    manifest = _manifest(temperature=0.0, seeds=[11, 22, 33])
    with pytest.raises(lifecycle_gates.LifecycleError) as refused:
        run_workflow._run_impl("demo", manifest, None, str(tmp_path), None,
                               None, lambda *_: None)
    assert str(refused.value) == GREEDY_THREE


@pytest.mark.parametrize("fields, words", [
    ({"temperature": 0.0, "samplesPerItem": 3, "seedPolicy": "derivedSHA256"},
     "samplesPerItem > 1 requires temperature > 0"),
    ({"outcomeInstruments": ["ordinalScale"]},
     "outcomeInstruments includes ordinalScale but ordinalAggregation"),
])
def test_the_other_settings_refusals_come_before_the_model_too(
        tmp_path, fields, words):
    """A draft only WARNS at verify on this engine, so these used to refuse
    after the model load (`_run_impl`)."""
    root = str(tmp_path)
    prompts = _study(root, "settings", **fields)
    loads = []

    @contextmanager
    def counting_model(model_id, revision=None):
        loads.append(model_id)
        yield SimpleNamespace(model_id=model_id, revision=revision or "abc")

    with pytest.raises(lifecycle_gates.LifecycleError) as refused:
        tasks.run("settings", prompts, root, model_provider=counting_model,
                  log=lambda *_: None)
    assert words in str(refused.value)
    assert refused.value.gate == "studyDeclaration"
    assert refused.value.repair_action
    assert loads == []


def test_one_seed_at_temperature_zero_runs(tmp_path, monkeypatch):
    import steerlab_server.experiment.generate as generate
    import steerlab_server.experiment.vector_materialization as materialization
    from steerlab_server.steering.vector_store import ConceptVectors
    root = str(tmp_path)
    prompts = _study(root, "one-seed", temperature=0.0, seeds=[11])
    bundle = materialization.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4, residual_norm_source="test",
        stimulus_hash="h")
    monkeypatch.setattr(materialization, "extract_all",
                        lambda model, manifest, root: {"fear": bundle})
    monkeypatch.setattr(generate, "generate",
                        lambda model, prompt, **_: "an answer")

    @contextmanager
    def fake_model(model_id, revision=None):
        yield SimpleNamespace(model_id=model_id, revision=revision or "abc")

    run_dir = tasks.run("one-seed", prompts, root, model_provider=fake_model,
                        log=lambda *_: None)
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as h:
        records = [json.loads(line) for line in h if line.strip()]
    sampled = [r for r in records if "instrument" not in r and "error" not in r]
    # One record per (arm, item): baseline and fear-on.
    assert len(sampled) == 2


# --- the engine's submission preflight: before anything is written or queued ------

@pytest.fixture
def server(tmp_path, monkeypatch):
    root = str(tmp_path / "source")
    monkeypatch.setenv("STEERLAB_ROOT", root)
    monkeypatch.setenv("STEERLAB_RUN_ROOT", str(tmp_path / "scratch"))
    jobs = JobManager(DurableJobStore(str(tmp_path / "jobs.sqlite")))
    return root, jobs


def _submissions_written(root):
    """Submission directories, in the workspace's runs/ and in the run root
    an engine without a ``root`` argument writes them to."""
    found = []
    for runs in (os.path.join(root, "runs"), os.environ["STEERLAB_RUN_ROOT"]):
        if os.path.isdir(runs):
            found += [d for d in os.listdir(runs) if "submit" in d]
    return found


def test_a_frozen_study_is_refused_at_submission_before_anything_is_written(
        server, monkeypatch):
    root, jobs = server
    _frozen_under_an_older_release(root, "greedy", monkeypatch,
                                   temperature=0.0, seeds=[11, 22, 33])
    with pytest.raises(submissions.SubmissionRefusal) as refused:
        submissions.submit_study("greedy", verb="run", jobs=jobs,
                                 executor="slurm", dry_run=True,
                                 resources={"gres": "A100",
                                            "walltime": "00:30:00"})
    assert GREEDY_THREE in str(refused.value)
    assert "Nothing was submitted" in str(refused.value)
    assert refused.value.code == "studyDeclaration"
    assert "steerlab-cli experiment duplicate greedy greedy-v2" in str(
        refused.value)
    assert refused.value.repair_action in str(refused.value)
    assert _submissions_written(root) == []


def test_a_bundle_is_refused_at_submission_before_anything_is_written(
        server, monkeypatch, tmp_path):
    root, jobs = server
    source = str(tmp_path / "author")
    _frozen_under_an_older_release(source, "greedy", monkeypatch,
                                   temperature=0.0, seeds=[11, 22, 33])
    meta = bundles.package_experiment("greedy", root=source)
    before = _submissions_written(root)
    with pytest.raises(submissions.SubmissionRefusal) as refused:
        submissions.submit_run_bundle(meta["bundlePath"], verb="run",
                                      jobs=jobs, executor="slurm",
                                      dry_run=True,
                                      resources={"gres": "A100",
                                                 "walltime": "00:30:00"})
    assert GREEDY_THREE in str(refused.value)
    assert _submissions_written(root) == before


@pytest.mark.parametrize("fields, words", [
    ({"temperature": 0.0, "samplesPerItem": 3, "seedPolicy": "derivedSHA256"},
     "samplesPerItem > 1 requires temperature > 0"),
    ({"outcomeInstruments": ["ordinalScale"]},
     "outcomeInstruments includes ordinalScale but ordinalAggregation"),
])
def test_the_other_settings_refusals_reach_the_submission_too(server, fields,
                                                              words):
    root, jobs = server
    _study(root, "settings", **fields)
    with pytest.raises(submissions.SubmissionRefusal) as refused:
        submissions.submit_study("settings", verb="run", jobs=jobs,
                                 executor="slurm", dry_run=True,
                                 resources={"gres": "A100",
                                            "walltime": "00:30:00"})
    assert words in str(refused.value)
    assert _submissions_written(root) == []


def test_a_concept_study_with_no_arm_is_refused_at_submission(
        server, monkeypatch):
    """Run start refused it before the model loaded, but only after the queue
    wait. A chain is not refused for it: its promote stage mints the arm."""
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "L4,A100,H100")
    root, jobs = server
    _study(root, "no-arm", conditions=[],
           pipeline={"stages": ["extract", "validate", "sweep", "promote",
                                "run"]})
    with pytest.raises(ValueError) as refused:
        submissions.submit_study("no-arm", verb="run", jobs=jobs,
                                 executor="slurm", dry_run=True,
                                 resources={"gres": "A100",
                                            "walltime": "00:30:00"})
    assert "measure nothing" in str(refused.value)
    assert _submissions_written(root) == []
    # Steps that do not run the study, and a chain that mints its arm, are
    # not refused for it.
    for verb in ("verify", "pipeline"):
        submissions.submit_study("no-arm", verb=verb, jobs=jobs,
                                 executor="slurm", dry_run=True,
                                 resources={"gres": "A100",
                                            "walltime": "00:30:00"})


DECLARE_AN_ARM = (
    "on your authoring client: steerlab-cli experiment declare-condition "
    "{name} <arm> --slots <concept>:<layer>:<alpha>  (Mac command line), or "
    "steerlab experiment declare-condition {name} <arm> --slots "
    "<concept>:<layer>:<alpha> --alpha-units norm|raw  (cross-platform "
    "client)")


@pytest.mark.parametrize("fields, name, repair", [
    ({"conditions": []}, "no-arm", DECLARE_AN_ARM),
    ({"studyType": "agentComparison"}, "inert",
     "declare arms the study's studyType actually runs — " + DECLARE_AN_ARM),
])
def test_the_arm_refusals_are_typed_on_the_engine_cli_and_the_route(
        server, monkeypatch, capsys, fields, name, repair):
    """They raised a plain ValueError, so `study submit` printed
    "ValueError: …" and exited 1 with no code or repair, and the route's
    400 carried prose alone. Run start has always typed them."""
    from starlette.testclient import TestClient
    from steerlab_server import cli
    from steerlab_server.api.app import app
    root, jobs = server
    _study(root, name, **fields)
    repair = repair.format(name=name)
    with pytest.raises(submissions.SubmissionRefusal) as refused:
        submissions.submit_study(name, verb="run", jobs=jobs,
                                 executor="slurm", dry_run=True,
                                 resources={"gres": "A100",
                                            "walltime": "00:30:00"})
    assert refused.value.code == "inertConditions"
    assert refused.value.repair_action == repair
    # The engine's command line answers with the code and the repair.
    capsys.readouterr()
    code = cli.main(["study", "submit", name, "--verb", "run", "--executor",
                     "slurm", "--dry-run", "--gres", "A100", "--walltime",
                     "00:30:00", "--json"])
    document = json.loads(capsys.readouterr().out)
    assert code != 0, document
    assert document["state"] == "refused"
    assert document["error"]["code"] == "inertConditions"
    assert document["error"]["repairAction"] == repair
    # The route's 400 carries them beside the message both clients read.
    monkeypatch.delenv("STEERLAB_SERVER_ROLE", raising=False)
    response = TestClient(app).post("/api/studies/submit", json={
        "experiment": name, "verb": "run", "executor": "slurm",
        "dryRun": True, "resources": {"gres": "A100", "walltime": "00:30:00"}})
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "inertConditions"
    assert detail["repairAction"] == repair
    assert detail["message"] == str(refused.value)
    assert _submissions_written(root) == []


def test_steps_that_do_not_run_the_study_are_not_refused(server, monkeypatch):
    """A frozen study's existing evidence can still be analysed, and a
    pipeline that does not run it is not refused for its seeds."""
    root, jobs = server
    _frozen_under_an_older_release(root, "greedy", monkeypatch,
                                   temperature=0.0, seeds=[11, 22, 33])
    manifest = Manifest.load("greedy", root)
    submissions._refuse_settings(manifest, "analyze", None)
    submissions._refuse_settings(manifest, "pipeline", ["evaluate", "analyze"])
    with pytest.raises(submissions.SubmissionRefusal):
        submissions._refuse_settings(manifest, "pipeline", ["run", "analyze"])


# --- the pinned task prompts' rules, before the queue too -------------------------

def test_an_instrument_that_cannot_read_its_items_is_refused_at_submission(
        server):
    """Run start refused it before the model loaded
    (``run_preflight.instrument_preflight``), but only after the queue wait:
    an answer-token instrument over items that carry no options."""
    root, jobs = server
    prompts = _study(root, "formats",
                     outcomeInstruments=["answerTokenLogprob"])
    with pytest.raises(submissions.SubmissionRefusal) as refused:
        submissions.submit_study("formats", verb="run", jobs=jobs,
                                 executor="slurm", dry_run=True,
                                 prompts_path=prompts,
                                 resources={"gres": "A100",
                                            "walltime": "00:30:00"})
    assert refused.value.code == "responseFormat"
    assert "none of the 1 task item carries options" in str(refused.value)
    assert refused.value.repair_action in str(refused.value)
    assert _submissions_written(root) == []
    # Items the instrument can read submit.
    _write(prompts, json.dumps({"id": "item-1", "prompt": "Pick one.",
                                "options": ["A", "B"], "target": "A",
                                "responseFormat": "label"}) + "\n")
    submissions.submit_study("formats", verb="run", jobs=jobs,
                             executor="slurm", dry_run=True,
                             prompts_path=prompts,
                             resources={"gres": "A100", "walltime": "00:30:00"})


def test_an_attention_check_rule_with_no_checked_item_is_refused_at_submission(
        server):
    root, jobs = server
    prompts = _study(root, "checks",
                     exclusionRules=[{"rule": "failedAttentionCheck"}])
    with pytest.raises(submissions.SubmissionRefusal) as refused:
        submissions.submit_study("checks", verb="run", jobs=jobs,
                                 executor="slurm", dry_run=True,
                                 prompts_path=prompts,
                                 resources={"gres": "A100",
                                            "walltime": "00:30:00"})
    assert "no task-prompt item declares an attentionCheck" in str(
        refused.value)
    assert refused.value.code == "studyDeclaration"
    assert _submissions_written(root) == []
    # A step that does not run the study is not refused for it.
    submissions.submit_study("checks", verb="analyze", jobs=jobs,
                             executor="slurm", dry_run=True,
                             prompts_path=prompts,
                             resources={"gres": "A100", "walltime": "00:30:00"})


# --- only sampled text loops over the seeds (review round 1) ----------------------

READOUT_ITEM = {"id": "item-1", "prompt": "Pick one.", "options": ["A", "B"],
                "target": "A", "responseFormat": "label"}


def _pin_items(root, name, prompts, item):
    _write(prompts, json.dumps(item) + "\n")
    raw = es.load_raw(name, root)
    raw["taskPromptsFile"] = os.path.relpath(prompts, root)
    with open(prompts, "rb") as handle:
        raw["taskPromptsHash"] = hashlib.sha256(handle.read()).hexdigest()
    es.save_raw(raw, root)


def _frozen_readout_study(root, name):
    """A readout-only study at temperature 0 with three seeds, its items
    pinned, frozen. The seed rule does not apply to it, so it freezes with
    no help from an older release."""
    prompts = _study(root, name, temperature=0.0, seeds=[11, 22, 33],
                     outcomeInstruments=["answerTokenLogprob"])
    _pin_items(root, name, prompts, READOUT_ITEM)
    assert Manifest.load(name, root).verify(root) == []
    es.freeze(name, force=True, root=root)
    assert es.load_raw(name, root)["status"] == "frozen"
    return prompts


def test_a_readout_only_study_with_several_seeds_is_not_refused():
    """A readout reads each item once whatever the seeds, so the sentence
    ("would write 3 identical records per item") would be false of it."""
    for status in ("draft", "frozen"):
        readout = _manifest(temperature=0.0, seeds=[11, 22, 33], status=status,
                            outcomeInstruments=["answerTokenLogprob"])
        assert sampling.manifest_greedy_seed_list_problem(readout) is None
        assert not any(v.startswith(sampling.GREEDY_SEED_LIST_PREFIX)
                       for v in readout.verify())
        assert study_admission.settings_refusal(readout) is None
        assert sampling.frozen_greedy_seed_list_advisory(readout) is None
        [note] = sampling.manifest_unused_seed_settings(readout)
        assert note.startswith("the seed list (3 seeds) is not used: this "
                               "study records answer-token readouts only")
    # A readout BESIDE sampled text still loops the text over the seeds.
    both = _manifest(temperature=0.0, seeds=[11, 22, 33],
                     outcomeInstruments=["answerTokenLogprob", "sampledText"])
    assert sampling.manifest_greedy_seed_list_problem(both) == GREEDY_THREE
    assert GREEDY_THREE in both.verify()


def test_a_frozen_readout_only_study_with_several_seeds_runs_once_per_item(
        tmp_path, monkeypatch):
    """The old run-start check refused it unconditionally, after the model
    load; it runs now, one readout record per arm and item."""
    import steerlab_server.experiment.generate as generate
    import steerlab_server.experiment.vector_materialization as materialization
    from steerlab_server.experiment import logprob
    from steerlab_server.steering.vector_store import ConceptVectors
    root = str(tmp_path)
    _frozen_readout_study(root, "readout")
    bundle = materialization.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4, residual_norm_source="test",
        stimulus_hash="h")
    monkeypatch.setattr(materialization, "extract_all",
                        lambda model, manifest, root: {"fear": bundle})
    generated = []
    monkeypatch.setattr(generate, "generate",
                        lambda *args, **kwargs: generated.append(1) or "text")
    read = []

    def score_options(model, prompt, options, **kwargs):
        read.append(prompt)
        return logprob.ChoiceResult(
            options=[logprob.OptionScore(option=o, token_ids=[i + 1],
                                         token_logprobs=[-1.0 - i])
                     for i, o in enumerate(options)],
            prompt_token_count=3)
    monkeypatch.setattr(logprob, "score_options", score_options)

    @contextmanager
    def fake_model(model_id, revision=None):
        yield SimpleNamespace(model_id=model_id, revision=revision or "abc",
                              tokenizer=object())

    run_dir = tasks.run("readout", None, root, model_provider=fake_model,
                        log=lambda *_: None)
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as h:
        records = [json.loads(line) for line in h if line.strip()]
    assert generated == []
    # Two arms (baseline and fear-on), one item: each read once, not 3 times.
    assert len(read) == 2
    assert sorted(r["condition"] for r in records) == ["baseline", "fear-on"]


def test_a_frozen_readout_only_study_is_not_refused_at_submission(server):
    root, jobs = server
    _frozen_readout_study(root, "readout")
    submission = submissions.submit_study(
        "readout", verb="run", jobs=jobs, executor="slurm", dry_run=True,
        resources={"gres": "A100", "walltime": "00:30:00"})
    assert submission.dry_run


# --- verify warns of a frozen study the rule will refuse (review round 1) ---------

FROZEN_NOTE = (
    "this study will be refused when its run is submitted or started, because "
    + GREEDY_THREE + ". To run it: ")


def test_verify_warns_of_a_frozen_study_the_seed_rule_refuses(
        workspace, capsys, monkeypatch):
    """It still verifies, so its runs stay readable, and the reader learns
    at verify, not at submission."""
    root = str(workspace)
    _frozen_under_an_older_release(root, "greedy", monkeypatch,
                                   temperature=0.0, seeds=[11, 22, 33])
    capsys.readouterr()
    code, document = _client(["experiment", "verify", "greedy"], capsys)
    assert code == 0, document
    assert document["state"] == "okWithAdvisories"
    [note] = [a for a in document["advisories"]
              if a["code"] == "refusedWhenRun"]
    assert note["detail"] == FROZEN_NOTE + (
        "steerlab experiment duplicate greedy greedy-v2 && steerlab "
        "experiment set-protocol greedy-v2 --set seeds='[<n>]'  (one seed; or "
        "raise the temperature above 0 to sample)")
    # The engine's own verify says it too, with both clients' spellings.
    monkeypatch.setenv("STEERLAB_ROOT", root)
    from steerlab_server import cli
    code = cli.main(["experiment", "verify", "greedy", "--json"])
    engine = json.loads(capsys.readouterr().out)
    assert code == 0, engine
    [note] = [a for a in engine["advisories"] if a["code"] == "refusedWhenRun"]
    assert note["detail"].startswith(FROZEN_NOTE + "on your authoring client: ")
    # A draft is refused at verify instead, so it gets no such advisory; and
    # a frozen study the rule does not refuse gets none.
    assert sampling.frozen_greedy_seed_list_advisory(
        _manifest(temperature=0.0, seeds=[11, 22, 33])) is None
    assert sampling.frozen_greedy_seed_list_advisory(
        _manifest(temperature=0.0, seeds=[11], status="frozen")) is None


# --- the pipeline holes (review round 1) ------------------------------------------

def test_a_pipeline_with_no_block_is_refused_at_submission(server, monkeypatch):
    """The reviewer's probe: with no pipeline block the submission could not
    tell which steps the chain runs, so a frozen study the seed rule refuses
    reached the queue and was refused at the chain's start."""
    from steerlab_server.experiment import pipeline_spec
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "L4,A100,H100")
    root, jobs = server
    _frozen_under_an_older_release(root, "greedy", monkeypatch,
                                   temperature=0.0, seeds=[11, 22, 33])
    with pytest.raises(submissions.SubmissionRefusal) as refused:
        submissions.submit_study("greedy", verb="pipeline", jobs=jobs,
                                 executor="slurm", dry_run=True,
                                 resources={"gres": "A100",
                                            "walltime": "00:30:00"})
    assert pipeline_spec.missing_block_problem("greedy") in str(refused.value)
    assert refused.value.code == "studyDeclaration"
    assert refused.value.repair_action.startswith(
        "on your authoring client: steerlab-cli experiment duplicate greedy "
        "greedy-v2 && steerlab-cli experiment set-pipeline greedy-v2 --file")
    assert _submissions_written(root) == []


def test_a_pipeline_block_that_does_not_resolve_is_refused_at_submission(
        server, monkeypatch):
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "L4,A100,H100")
    root, jobs = server
    _study(root, "chain", pipeline={"stages": ["run", "extract"]})
    with pytest.raises(submissions.SubmissionRefusal) as refused:
        submissions.submit_study("chain", verb="pipeline", jobs=jobs,
                                 executor="slurm", dry_run=True,
                                 resources={"gres": "A100",
                                            "walltime": "00:30:00"})
    from steerlab_server.experiment import pipeline_spec
    with pytest.raises(ValueError) as chain_says:
        pipeline_spec.resolve_pipeline({"stages": ["run", "extract"]})
    assert str(chain_says.value) in str(refused.value)
    assert _submissions_written(root) == []


def test_a_declared_chain_that_runs_the_study_is_refused_at_submission(
        server, monkeypatch):
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "L4,A100,H100")
    root, jobs = server
    _frozen_under_an_older_release(
        root, "greedy", monkeypatch, temperature=0.0, seeds=[11, 22, 33],
        pipeline={"stages": ["run", "analyze"]})
    with pytest.raises(submissions.SubmissionRefusal) as refused:
        submissions.submit_study("greedy", verb="pipeline", jobs=jobs,
                                 executor="slurm", dry_run=True,
                                 resources={"gres": "A100",
                                            "walltime": "00:30:00"})
    assert GREEDY_THREE in str(refused.value)
    assert _submissions_written(root) == []


def test_a_chain_meets_the_settings_refusals_before_any_stage_or_model(
        tmp_path, monkeypatch):
    """Inside a chain the refusal fired only at the run stage, after the
    model was acquired and extract, validate, sweep, and promote had run."""
    root = str(tmp_path)
    _frozen_under_an_older_release(
        root, "greedy", monkeypatch, temperature=0.0, seeds=[11, 22, 33],
        pipeline={"stages": ["extract", "validate", "sweep", "promote", "run"]})
    loads = []

    @contextmanager
    def counting_model(model_id, revision=None):
        loads.append(model_id)
        yield SimpleNamespace(model_id=model_id, revision=revision or "abc")

    with pytest.raises(lifecycle_gates.LifecycleError) as refused:
        tasks.pipeline("greedy", root, model_provider=counting_model,
                       log=lambda *_: None)
    assert str(refused.value) == GREEDY_THREE
    assert refused.value.gate == "studyDeclaration"
    assert "duplicate greedy greedy-v2" in refused.value.repair_action
    assert loads == []


# --- the clients refuse before submitting (review round 1) -------------------------


def test_the_client_refuses_before_any_runner_is_addressed(
        tmp_path, monkeypatch, capsys, script):
    """So even a runner from before the rule, which would refuse only after
    the queue wait and the model load, is never reached."""
    from test_run_orchestration import (RUNNER_URL, STUDY_NAME,
                                        _frozen_workspace, _run, _stages)
    with monkeypatch.context() as older:
        older.setattr(sampling, "manifest_greedy_seed_list_problem",
                      lambda manifest: None)
        root = _frozen_workspace(tmp_path, monkeypatch,
                                 protocol=("seeds=[11,22,33]",))
    assert es.load_raw(STUDY_NAME, root)["status"] == "frozen"
    code, document, _ = _run([STUDY_NAME, "--runner", RUNNER_URL], capsys,
                             root=root)
    assert code == 65, document
    assert document["error"]["code"] == "studyDeclaration"
    assert document["error"]["reason"] == GREEDY_THREE
    assert document["error"]["repairAction"].startswith(
        f"steerlab experiment duplicate {STUDY_NAME} {STUDY_NAME}-v2 && "
        f"steerlab experiment set-protocol {STUDY_NAME}-v2 --set seeds='[<n>]'")
    assert document["result"]["failedStage"] == "load"
    assert _stages(document)["package"]["state"] == client_cli.STAGE_NOT_REACHED
    assert script.info_calls == 0
    assert script.uploads == [] and script.submits == []


def test_runner_submit_refuses_a_staged_study_the_seed_rule_refuses(tmp_path,
                                                                     monkeypatch):
    """`runner submit` names only a staged bundle: its study is read from the
    local workspace when the bundle's name and content hash match it."""
    from steerlab_server.client import runner as runner_api
    root = str(tmp_path)
    _frozen_under_an_older_release(root, "greedy", monkeypatch,
                                   temperature=0.0, seeds=[11, 22, 33])
    staged = {"experiment": "greedy", "experimentContentHash":
              Manifest.load("greedy", root).content_hash()}
    with pytest.raises(runner_api.RunnerRefusal) as refused:
        runner_api.require_greedy_seed_list(staged, "run", root)
    assert str(refused.value) == GREEDY_THREE
    assert refused.value.code == "studyDeclaration"
    assert "duplicate greedy greedy-v2" in refused.value.repair_action
    # Steps that do not run the study, a bundle that is not this study, and
    # no workspace to vouch for it: nothing to refuse.
    runner_api.require_greedy_seed_list(staged, "analyze", root)
    runner_api.require_greedy_seed_list(
        {**staged, "experimentContentHash": "0" * 64}, "run", root)
    runner_api.require_greedy_seed_list(staged, "run", None)
    # A pipeline is refused only when its declared chain runs the study.
    manifest = Manifest.load("greedy", root)
    assert sampling.submission_greedy_seed_list_problem(manifest, "pipeline") \
        is None  # no block: the engine refuses the block itself
    manifest.raw["pipeline"] = {"stages": ["run", "analyze"]}
    assert sampling.submission_greedy_seed_list_problem(
        manifest, "pipeline") == GREEDY_THREE
    manifest.raw["pipeline"] = {"stages": ["extract", "validate"]}
    assert sampling.submission_greedy_seed_list_problem(
        manifest, "pipeline") is None


# --- a resumed chain is asked of what remains (review round 3) ---------------------

def _parked_chain(root, name, stage_results):
    """A pipeline run directory whose ledger records ``stage_results``."""
    from steerlab_server.experiment import pipeline_ledger
    manifest = Manifest.load(name, root)
    directory = os.path.join(root, "runs", f"{name}-pipeline")
    os.makedirs(directory)
    pipeline_ledger.write_pipeline_ledger(directory, {
        "schema": pipeline_ledger.PIPELINE_LEDGER_SCHEMA,
        "experiment": name, "experimentHash": manifest.content_hash(),
        "stages": list(manifest.raw["pipeline"]["stages"]),
        "disposition": None, "stageResults": stage_results})
    return directory


def _unexpected(*args, **kwargs):
    raise AssertionError("this capability must not be used")


def _chain_stages(**overrides):
    from steerlab_server.experiment.pipeline_workflow import PipelineStages
    callbacks = dict.fromkeys(("extract", "validate", "sweep", "run",
                               "evaluate", "analyze", "promote"), _unexpected)
    return PipelineStages(**{**callbacks, **overrides})


def test_a_chain_whose_run_completed_resumes_into_analyze_unrefused(
        tmp_path, monkeypatch):
    """Every post-run continuation still lists `run` among its declared
    stages, so asking the declared list refused a chain parked in analyze
    after its run had completed: the orphan reconcile, the post-merge
    continuation, and a requeue of a finished chain all raised."""
    from steerlab_server.experiment.pipeline_workflow import (PipelineModels,
                                                              pipeline)
    root = str(tmp_path)
    _frozen_under_an_older_release(
        root, "greedy", monkeypatch, temperature=0.0, seeds=[11, 22, 33],
        pipeline={"stages": ["run", "analyze"]})
    source = os.path.join(root, "runs", "completed-run")
    os.makedirs(source)
    directory = _parked_chain(root, "greedy", {
        "run": {"status": "completed", "runDirectory": source}})
    analysed = []

    def analyze(name, root, **kwargs):
        analysed.append(kwargs["source_run"])
        result = os.path.join(root, "runs", "analysis")
        os.makedirs(result)
        return result

    out = pipeline("greedy", root, pipeline_run_directory=directory,
                   stages=_chain_stages(analyze=analyze),
                   models=PipelineModels(acquire=_unexpected,
                                         pin_revision=_unexpected),
                   log=lambda _: None)
    assert out == directory
    assert analysed == [source]
    # A requeue of the finished chain returns idempotently.
    assert pipeline("greedy", root, pipeline_run_directory=directory,
                    stages=_chain_stages(),
                    models=PipelineModels(acquire=_unexpected,
                                          pin_revision=_unexpected),
                    log=lambda _: None) == directory


def test_a_chain_with_its_run_ahead_is_refused_before_the_model(
        tmp_path, monkeypatch):
    from steerlab_server.experiment.pipeline_workflow import (PipelineModels,
                                                              pipeline)
    root = str(tmp_path)
    _frozen_under_an_older_release(
        root, "greedy", monkeypatch, temperature=0.0, seeds=[11, 22, 33],
        pipeline={"stages": ["extract", "run", "analyze"]})
    directory = _parked_chain(root, "greedy", {
        "extract": {"status": "completed",
                    "runDirectory": os.path.join(root, "runs", "x")}})
    with pytest.raises(lifecycle_gates.LifecycleError) as refused:
        pipeline("greedy", root, pipeline_run_directory=directory,
                 stages=_chain_stages(),
                 models=PipelineModels(acquire=_unexpected,
                                       pin_revision=_unexpected),
                 log=lambda _: None)
    assert str(refused.value) == GREEDY_THREE


def test_a_resumed_chain_is_asked_of_what_remains_at_submission(
        server, monkeypatch):
    monkeypatch.setenv("STEERLAB_SLURM_GPU_TYPES", "L4,A100,H100")
    root, jobs = server
    _frozen_under_an_older_release(
        root, "greedy", monkeypatch, temperature=0.0, seeds=[11, 22, 33],
        pipeline={"stages": ["run", "analyze"]})
    source = os.path.join(root, "runs", "completed-run")
    os.makedirs(source)
    directory = _parked_chain(root, "greedy", {
        "run": {"status": "completed", "runDirectory": source}})
    # Its run completed: the continuation is not refused for its seeds.
    submissions.submit_study("greedy", verb="pipeline", jobs=jobs,
                             executor="slurm", dry_run=True,
                             resume_directory=directory,
                             resources={"gres": "A100",
                                        "walltime": "00:30:00"})
    # A fresh chain that runs the study still is.
    with pytest.raises(submissions.SubmissionRefusal):
        submissions.submit_study("greedy", verb="pipeline", jobs=jobs,
                                 executor="slurm", dry_run=True,
                                 resources={"gres": "A100",
                                            "walltime": "00:30:00"})


# --- an older engine is refused on the client (review round 3) ----------------------

READOUT_SEEDS = ('seeds=[11,22,33]', 'outcomeInstruments=["answerTokenLogprob"]')


def test_the_engine_advertises_the_greedy_rules_and_a_client_can_read_it():
    from steerlab_server.api.profile import capability_snapshot
    snapshot = capability_snapshot()
    assert snapshot["remoteStudy"]["greedyRulesSampledTextOnly"] is True
    assert sampling.remote_has_greedy_rules(snapshot)
    assert sampling.remote_has_greedy_rules(
        {"features": {"remoteStudy.greedyRulesSampledTextOnly": True}})
    assert not sampling.remote_has_greedy_rules(
        {"remoteStudy": {"declaredSeedPolicy": True}})
    assert not sampling.remote_has_greedy_rules(None)


def test_the_client_refuses_a_readout_study_for_an_engine_from_before_the_fix(
        tmp_path, monkeypatch, capsys, script):
    """A 0.9.21 engine refused a readout-only study at temperature 0 with
    several seeds after its model load: exactly the issue #9 failure. This
    release runs it; the client refuses to send it to an engine that does
    not say so."""
    from test_run_orchestration import (RUNNER_URL, STUDY_NAME,
                                        _frozen_workspace, _run)
    root = _frozen_workspace(tmp_path, monkeypatch, protocol=READOUT_SEEDS)
    code, document, _ = _run([STUDY_NAME, "--runner", RUNNER_URL, "--dry-run"],
                             capsys, root=root)
    assert code == 65, document
    assert document["error"]["code"] == "greedyRulesUnsupported"
    assert ("predates the fix for public issue #9"
            in document["error"]["reason"])
    assert ("remoteStudy.greedyRulesSampledTextOnly"
            in document["error"]["reason"])
    repair = document["error"]["repairAction"]
    assert "cluster push --site <id>, then restart its controller" in repair
    assert (f"steerlab experiment set-protocol {STUDY_NAME}-v2 --set "
            "seeds='[<n>]' --set samplesPerItem=1") in repair
    assert document["result"]["failedStage"] == "capabilities"
    assert script.uploads == [] and script.submits == []
    # An engine from this release takes it.
    script.greedy_rules = True
    script.job_status = "prepared"
    code, document, _ = _run([STUDY_NAME, "--runner", RUNNER_URL, "--dry-run"],
                             capsys, root=root)
    assert code == 0, document
    assert len(script.submits) == 1


def test_runner_submit_refuses_a_readout_study_for_an_older_engine(tmp_path):
    from steerlab_server.client import runner as runner_api
    root = str(tmp_path)
    _frozen_readout_study(root, "readout")
    staged = {"experiment": "readout", "experimentContentHash":
              Manifest.load("readout", root).content_hash()}

    def check(capabilities, verb="run", workspace=root):
        runner_api.require_greedy_rules(staged, verb, capabilities,
                                        "http://runner.invalid", workspace)

    with pytest.raises(runner_api.RunnerRefusal) as refused:
        check({"remoteStudy": {"declaredSeedPolicy": True}})
    assert refused.value.code == "greedyRulesUnsupported"
    # Copied from `SubstrateRouting.greedyRulesSubmissionRefusal`.
    assert str(refused.value) == (
        "The engine at http://runner.invalid predates the fix for public "
        "issue #9 that lets an answer-token readout study run at temperature "
        "0 with several seeds or several samples per item: its capabilities "
        "do not list remoteStudy.greedyRulesSampledTextOnly, so it would "
        "refuse this study after the queue wait and the model load. Nothing "
        "was submitted.")
    check({"remoteStudy": {"greedyRulesSampledTextOnly": True}})
    check({}, verb="analyze")
    check({}, workspace=None)
    # A study with one seed and one sample needs nothing of the engine.
    assert not sampling.submission_needs_greedy_rules(
        _manifest(temperature=0.0, seeds=[11],
                  outcomeInstruments=["answerTokenLogprob"]), "run")
    assert sampling.submission_needs_greedy_rules(
        _manifest(temperature=0.0, seeds=[11], samplesPerItem=3,
                  outcomeInstruments=["answerTokenLogprob"]), "run")
    assert not sampling.submission_needs_greedy_rules(
        _manifest(temperature=0.7, seeds=[11, 22],
                  outcomeInstruments=["answerTokenLogprob"]), "run")


# --- samplesPerItem only where the run reads it (review round 3) --------------------

def test_a_readout_only_study_with_several_samples_runs_and_is_advised(
        workspace, capsys):
    """samplesPerItem 3 at temperature 0 on a readout-only study was refused
    at verify, submission, and run start, while the inert advisory called the
    value inert. The readout reads each item once whatever the count."""
    _verifiable_draft(workspace, capsys)
    code, _ = _client(["experiment", "set-protocol", "demo", "--set",
                       "samplesPerItem=3", "--set",
                       'outcomeInstruments=["answerTokenLogprob"]'], capsys)
    assert code == 0
    code, document = _client(["experiment", "verify", "demo"], capsys)
    assert code == 0, document
    codes = [a["code"] for a in document["advisories"]]
    assert "choiceItemsWithoutInstrument" in codes
    manifest = Manifest.load("demo", str(workspace))
    assert study_admission.settings_refusal(manifest) is None
    # Beside sampled text it is still refused.
    both = _manifest(temperature=0.0, samplesPerItem=3,
                     seedPolicy="derivedSHA256",
                     outcomeInstruments=["answerTokenLogprob", "sampledText"])
    assert ("samplesPerItem > 1 requires temperature > 0 (greedy decoding "
            "makes every sample identical)") in both.verify()
    assert study_admission.settings_refusal(both) is not None


# --- review round 4 -----------------------------------------------------------------

def test_a_refused_resume_leaves_its_ledger_and_manifest_untouched(
        tmp_path, monkeypatch):
    """The refusal sat below the epoch guard and the parked-stamp clear, so a
    parked chain with its run still ahead lost its `parked` stamp (and could
    gain a drift stamp or a restored pin) before it was refused."""
    from steerlab_server.experiment import pipeline_ledger
    from steerlab_server.experiment.pipeline_workflow import (PipelineModels,
                                                              pipeline)
    root = str(tmp_path)
    _frozen_under_an_older_release(
        root, "greedy", monkeypatch, temperature=0.0, seeds=[11, 22, 33],
        pipeline={"stages": ["extract", "run", "analyze"]})
    directory = _parked_chain(root, "greedy", {
        "extract": {"status": "completed",
                    "runDirectory": os.path.join(root, "runs", "x")}})
    # Parked by the reconcile, and stamped with a manifest hash the live one
    # does not match, so the drift stamp would be written on the way through.
    ledger = pipeline_ledger.read_pipeline_ledger(directory)
    ledger["parked"] = {"reason": "the orphan reconcile parked it"}
    ledger["experimentHash"] = "0" * 64
    pipeline_ledger.write_pipeline_ledger(directory, ledger)
    ledger_path = pipeline_ledger.pipeline_ledger_path(directory)
    manifest_path = os.path.join(root, "experiments", "greedy",
                                 "experiment.json")
    with open(ledger_path, "rb") as handle:
        ledger_before = handle.read()
    with open(manifest_path, "rb") as handle:
        manifest_before = handle.read()
    with pytest.raises(lifecycle_gates.LifecycleError) as refused:
        pipeline("greedy", root, pipeline_run_directory=directory,
                 stages=_chain_stages(),
                 models=PipelineModels(acquire=_unexpected,
                                       pin_revision=_unexpected),
                 log=lambda _: None)
    assert str(refused.value) == GREEDY_THREE
    with open(ledger_path, "rb") as handle:
        assert handle.read() == ledger_before
    with open(manifest_path, "rb") as handle:
        assert handle.read() == manifest_before


def test_the_old_engine_refusal_is_worded_by_the_study(tmp_path):
    """A study with sampled text that every engine refuses is told the
    rule's own sentence, since updating the engine would not help; only a
    readout-only study is told the engine predates the fix. Both are refused
    before the queue."""
    from steerlab_server.client import runner as runner_api
    root = str(tmp_path)
    _study(root, "sampled", temperature=0.0, seeds=[11], samplesPerItem=3,
           seedPolicy="derivedSHA256")
    staged = {"experiment": "sampled", "experimentContentHash":
              Manifest.load("sampled", root).content_hash()}
    older = {"remoteStudy": {"declaredSeedPolicy": True}}
    with pytest.raises(runner_api.RunnerRefusal) as refused:
        runner_api.require_greedy_rules(staged, "run", older,
                                        "http://runner.invalid", root)
    assert refused.value.code == "studyDeclaration"
    assert str(refused.value) == (
        "samplesPerItem > 1 requires temperature > 0 (greedy decoding makes "
        "every sample identical)")
    assert "--samples-per-item 1" in refused.value.repair_action
    assert "predates" not in str(refused.value)
    # The function on both branches, and on a several-seed sampled study.
    manifest = Manifest.load("sampled", root)
    code, reason, _ = sampling.greedy_rules_submission_refusal(manifest, "here")
    assert (code, reason) == ("studyDeclaration", str(refused.value))
    seeds = _manifest(temperature=0.0, seeds=[11, 22, 33])
    code, reason, repair = sampling.greedy_rules_submission_refusal(seeds, "here")
    assert (code, reason) == ("studyDeclaration", GREEDY_THREE)
    readout = _manifest(temperature=0.0, seeds=[11, 22, 33],
                        outcomeInstruments=["answerTokenLogprob"])
    code, reason, repair = sampling.greedy_rules_submission_refusal(readout,
                                                                    "here")
    assert code == "greedyRulesUnsupported"
    assert reason.startswith("The engine at here predates the fix")
    assert "cluster push --site <id>, then restart its controller" in repair
