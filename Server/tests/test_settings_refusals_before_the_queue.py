"""A study its settings alone make unrunnable is refused before the queue.

Public issue #9: a temperature-0 study that kept a list of several seeds
passed verify and freeze, waited for a GPU, loaded its model, and only then
refused at run start, because at temperature 0 every seed gives the same
greedy response. The rule now has one sentence on both engines
(``sampling.greedy_seed_list_problem``), and it fires at verify on a draft (so
freeze refuses), at the engine's submission preflight before anything is
written or queued (a study frozen under an older release meets it there), and
at run start before the model loads.

The other refusals of run start that depend on the settings alone are asked
at the same two places through the same function
(``study_admission.settings_refusal``): several samples per item at
temperature 0, and an ordinal scale with no declared aggregation. A concept
study that declares no arm at all is now refused at submission too.
"""

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


def test_a_concept_study_with_no_arm_is_refused_at_submission(server):
    """Run start refused it; the submission let it reach the queue first."""
    root, jobs = server
    _study(root, "no-arm", conditions=[])
    with pytest.raises(ValueError) as refused:
        submissions.submit_study("no-arm", verb="run", jobs=jobs,
                                 executor="slurm", dry_run=True,
                                 resources={"gres": "A100",
                                            "walltime": "00:30:00"})
    assert "measure nothing" in str(refused.value)
    assert _submissions_written(root) == []
    # A step that does not run the study is not refused for it.
    submissions.submit_study("no-arm", verb="verify", jobs=jobs,
                             executor="slurm", dry_run=True,
                             resources={"gres": "A100", "walltime": "00:30:00"})


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
