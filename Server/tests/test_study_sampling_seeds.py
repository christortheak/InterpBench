"""A declared seed policy decides the seeds (public issue #3).

A study could declare ``seedPolicy: derivedSHA256`` and still have every item
sampled with the same seeds from the fixed list: the engine chose the rule
from ``samplesPerItem`` alone and never read the declaration. With one sample
per item and a temperature above 0, two items with the same text then drew
the same stream, so a design that samples respondents with replacement
reproduced earlier answers exactly. The rule now lives in one place
(``sampling.derives_seeds``), and the run loop, the per-item count, sharding,
the J-lens budget, and submission sizing all read it.

The shared fixture ``Tests/Fixtures/cross-engine/study-sampling-seeds.json``
holds each case's seeds, count, and advisories; the Mac engine reads the same
file (``StudySamplingTests``) and must give the same answers.
"""

import json
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import steerlab_server.experiment.generate as _owner_generate
import steerlab_server.experiment.vector_materialization as _owner_vector_materialization
from steerlab_server import cli_envelope, client_cli
from steerlab_server.api import submissions
from steerlab_server.experiment import condition_execution, sampling, tasks
from steerlab_server.experiment import resume as resume_mod
from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering.vector_store import ConceptVectors

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Tests", "Fixtures", "cross-engine", "study-sampling-seeds.json")

REGENERATE = ("stale fixture — re-run "
              "`Server/.venv.nosync/bin/python "
              "scripts/regenerate-cross-engine-fixtures.py` and commit")


def _fixture():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def _case(label):
    [case] = [c for c in _fixture()["cases"] if c["label"] == label]
    return case


def _manifest(fields):
    return Manifest.from_dict({"name": "seeds", "modelID": "test/model",
                               "concepts": [], **fields})


# --- the shared fixture --------------------------------------------------------

@pytest.mark.parametrize("case", _fixture()["cases"],
                         ids=lambda case: case["label"])
def test_fixture_is_what_the_seed_rule_gives(case):
    fixture = _fixture()
    manifest = _manifest(case["manifest"])
    records = []
    for condition in fixture["conditions"]:
        for prompt in fixture["prompts"]:
            policy, samples = sampling.record_seeds(
                manifest, experiment_hash=fixture["experimentHash"],
                condition=condition, prompt_id=prompt["id"])
            assert policy == case["seedPolicy"], REGENERATE
            records.extend({"condition": condition, "promptID": prompt["id"],
                            "sampleIndex": index, "seed": seed}
                           for index, seed in samples)
    assert records == case["records"], REGENERATE
    assert sampling.manifest_derives_seeds(manifest) == case["derivesSeeds"]
    # The run's config.json stamps the policy its records stamp.
    assert sampling.run_seed_policy(manifest) == case["seedPolicy"]
    assert sampling.sample_count(manifest) == case["sampleCount"], REGENERATE
    assert condition_execution.effective_sample_count(manifest) \
        == case["sampleCount"]
    assert sampling.manifest_unused_seed_settings(manifest) \
        == case["advisories"], REGENERATE


def test_the_two_items_share_their_text_and_differ_by_id():
    """The design the fixture stands for: only the prompt ID tells the
    repeats apart."""
    prompts = _fixture()["prompts"]
    assert len({p["prompt"] for p in prompts}) == 1
    assert len({p["id"] for p in prompts}) == 2


def test_a_declared_derived_policy_derives_seeds_with_one_sample_per_item():
    case = _case("derivedSHA256, one sample per item, warm, seed list")
    fixture = _fixture()
    assert case["derivesSeeds"] and case["seedPolicy"] == "derivedSHA256"
    assert case["sampleCount"] == 1
    # One record per (condition, item): the three listed seeds are not used.
    assert len(case["records"]) == 2 * 2
    seeds = {(r["condition"], r["promptID"]): r["seed"] for r in case["records"]}
    assert not set(seeds.values()) & set(case["manifest"]["seeds"])
    for (condition, prompt_id), seed in seeds.items():
        assert seed == sampling.derive_seed(
            fixture["experimentHash"], condition, prompt_id, 0)
    # Same text, different IDs: different seeds.
    for condition in fixture["conditions"]:
        assert seeds[(condition, "respondent-001")] \
            != seeds[(condition, "respondent-002")]
    # The derivation includes the condition, so arms draw distinct streams
    # for the same item: paired at the prompt level, not common random
    # numbers.
    for prompt in fixture["prompts"]:
        assert seeds[("baseline", prompt["id"])] \
            != seeds[("steered", prompt["id"])]
    # The seed list does not change the derived seeds.
    assert case["records"] == _case(
        "derivedSHA256, one sample per item, warm, one seed")["records"]


def test_fixed_list_studies_are_unchanged():
    """Undeclared and manifestSeeds studies run each item once per listed
    seed, with the literal seeds, in every arm."""
    for label in ("manifestSeeds, warm", "undeclared, warm"):
        case = _case(label)
        assert not case["derivesSeeds"]
        assert case["seedPolicy"] == "manifestSeeds"
        assert case["sampleCount"] == 3
        assert [r["seed"] for r in case["records"]] == [11, 22, 33] * 4
        assert [r["sampleIndex"] for r in case["records"]] == [0, 1, 2] * 4
        assert case["advisories"] == []


def test_at_temperature_zero_the_declaration_decides_nothing():
    case = _case("derivedSHA256, greedy")
    assert not case["derivesSeeds"]
    assert case["seedPolicy"] == "manifestSeeds"
    assert case["records"] == _case("undeclared, greedy")["records"]
    [note] = case["advisories"]
    assert "greedy" in note and "no seed is used" in note


def test_more_than_one_sample_per_item_still_derives_whatever_is_declared():
    """The rule's other half is today's behaviour, kept: samplesPerItem above
    1 at a positive temperature derives seeds (verify separately requires the
    declaration)."""
    assert sampling.derives_seeds(0.7, 3, "manifestSeeds")
    assert sampling.derives_seeds(0.7, 3, None)
    assert not sampling.derives_seeds(0.0, 3, "derivedSHA256")
    assert not sampling.derives_seeds(0.7, 1, "manifestSeeds")
    assert sampling.derives_seeds(0.7, 1, "derivedSHA256")


def test_verify_keeps_requiring_the_declaration_for_several_samples():
    manifest = _manifest({"temperature": 0.7, "samplesPerItem": 3})
    assert any("requires seedPolicy 'derivedSHA256'" in v
               for v in manifest.verify())


def test_a_panel_gets_no_seed_advisory():
    """A panel's turns take seeds by their own rule (multi_agent), so neither
    advisory is true of one."""
    assert sampling.unused_seed_settings(0.0, 1, "derivedSHA256", [1],
                                         study_kind="multiAgent") == []
    assert sampling.unused_seed_settings(0.7, 1, "derivedSHA256", [1, 2],
                                         study_kind="multiAgent") == []


# --- every count reads the same rule -------------------------------------------

PROMPTS_2 = "\n".join(json.dumps({"id": f"p{i}", "prompt": "same text"})
                      for i in range(2))


def test_submission_sizing_prices_what_the_run_emits():
    def planned(**fields):
        manifest = Manifest.from_dict({
            "name": "m", "modelID": "test/model",
            "conditions": [{"name": "c1", "slots": []}], **fields})
        return submissions._planned_records(manifest, PROMPTS_2)

    # (baseline + c1) × 2 prompts × per-item count.
    assert planned(seeds=[1, 2, 3], temperature=0.7,
                   seedPolicy="derivedSHA256") == 4
    assert planned(seeds=[1, 2, 3], temperature=0.7) == 12
    assert planned(seeds=[1, 2, 3], temperature=0.7,
                   seedPolicy="manifestSeeds") == 12
    assert planned(seeds=[1, 2, 3], temperature=0.7, samplesPerItem=5,
                   seedPolicy="derivedSHA256") == 20


# --- the run loop draws the rule's seeds ---------------------------------------

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _study(root, name, **sampling_fields):
    """Baseline plus one saved agent, two items with the same text and
    different IDs, sampled text only."""
    concept_dir = os.path.join(root, "prompts", "concepts", "fear")
    _write(os.path.join(concept_dir, "positive.jsonl"), '{"text": "I feel dread"}\n')
    _write(os.path.join(concept_dir, "negative.jsonl"), '{"text": "calm morning"}\n')
    es.create(name, model_id="org/m", revision="abc", root=root)
    es.attach(name, ["fear"], root=root)
    raw = es.load_raw(name, root)
    raw.update(sampling_fields)
    raw["maxTokens"] = 16
    raw["outcomeInstruments"] = ["sampledText"]
    raw["variantConditions"] = [
        {"name": "steered", "artifactPath": "runs/variants/s.json",
         "artifactHash": "aa" * 32,
         "artifact": {"name": "steered", "baseModelID": "org/m",
                      "promptMode": "chatAssistant", "temperature": 0.0,
                      "alphaInNormUnits": False, "injections": [],
                      "adapters": []}},
    ]
    es.save_raw(raw, root)
    prompts_path = os.path.join(root, "prompts", "tasks", "items.jsonl")
    _write(prompts_path, "".join(
        json.dumps({"id": prompt_id, "prompt": "Describe your week."}) + "\n"
        for prompt_id in ("respondent-001", "respondent-002")))
    return prompts_path


@contextmanager
def _fake_model(model_id, revision=None):
    yield SimpleNamespace(model_id=model_id, revision=revision or "abc")


def _patch(monkeypatch):
    bundle = _owner_vector_materialization.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]] * 4),
        residual_norm_per_layer=[1.0] * 4,
        residual_norm_source="test", stimulus_hash="h")
    monkeypatch.setattr(_owner_vector_materialization, "extract_all",
                        lambda model, manifest, root: {"fear": bundle})

    def generate(model, prompt, *, temperature=0.0, token_ids_out=None, **_):
        return f"an answer to {prompt} (T={temperature})"
    monkeypatch.setattr(_owner_generate, "generate", generate)


def _sampled_records(run_dir):
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as h:
        records = [json.loads(line) for line in h if line.strip()]
    return [r for r in records if "instrument" not in r and "error" not in r]


def _config_seed_policy(run_dir):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as h:
        return json.load(h)["seedPolicy"]


def test_the_run_loop_derives_seeds_for_one_declared_sample(tmp_path,
                                                             monkeypatch):
    root = str(tmp_path)
    prompts = _study(root, "seeds-derived", temperature=0.7, samplesPerItem=1,
                     seedPolicy="derivedSHA256", seeds=[11, 22, 33])
    _patch(monkeypatch)
    run_dir = tasks.run("seeds-derived", prompts, root,
                        model_provider=_fake_model, log=lambda *_: None)
    with open(os.path.join(run_dir, "experiment-hash.txt"), encoding="utf-8") as h:
        experiment_hash = h.read().strip()
    records = _sampled_records(run_dir)
    assert len(records) == 2 * 2  # one per (condition, item), not three
    for record in records:
        assert record["seedPolicy"] == "derivedSHA256"
        assert record["sampleIndex"] == 0
        assert record["seed"] == sampling.derive_seed(
            experiment_hash, record["condition"], record["promptID"], 0)
    seeds = {(r["condition"], r["promptID"]): r["seed"] for r in records}
    assert len(set(seeds.values())) == 4
    assert _config_seed_policy(run_dir) == "derivedSHA256"


def test_the_run_loop_keeps_the_fixed_list_when_nothing_is_declared(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts = _study(root, "seeds-fixed", temperature=0.7, seeds=[11, 22, 33])
    _patch(monkeypatch)
    run_dir = tasks.run("seeds-fixed", prompts, root,
                        model_provider=_fake_model, log=lambda *_: None)
    records = _sampled_records(run_dir)
    assert len(records) == 2 * 2 * 3
    for record in records:
        assert record["seedPolicy"] == "manifestSeeds"
        assert record["seed"] == [11, 22, 33][record["sampleIndex"]]
    assert _config_seed_policy(run_dir) == "manifestSeeds"


# --- a client can tell whether an engine has the rule -------------------------

def test_the_engine_advertises_the_rule_and_a_client_can_read_it():
    from steerlab_server.api.profile import capability_snapshot
    snapshot = capability_snapshot()
    assert snapshot["remoteStudy"]["declaredSeedPolicy"] is True
    assert sampling.remote_honors_declared_seed_policy(snapshot)
    assert sampling.remote_honors_declared_seed_policy(
        {"features": {"remoteStudy.declaredSeedPolicy": True}})
    assert not sampling.remote_honors_declared_seed_policy(
        {"remoteStudy": {"variantStudySampling": True}})
    assert not sampling.remote_honors_declared_seed_policy(None)
    # A nested value wins over the dotted one when both are present, as in
    # ClusterCapabilities.supportsDeclaredSeedPolicy; a nested null does not
    # count as present.
    assert not sampling.remote_honors_declared_seed_policy(
        {"remoteStudy": {"declaredSeedPolicy": False},
         "features": {"remoteStudy.declaredSeedPolicy": True}})
    assert sampling.remote_honors_declared_seed_policy(
        {"remoteStudy": {"declaredSeedPolicy": True},
         "features": {"remoteStudy.declaredSeedPolicy": False}})
    assert sampling.remote_honors_declared_seed_policy(
        {"remoteStudy": {"declaredSeedPolicy": None},
         "features": {"remoteStudy.declaredSeedPolicy": True}})


def test_the_refusal_is_the_mac_clients_sentence():
    """Copied from ``SubstrateRoutingTests.aDeclaredSeedPolicyRefusesOnAServerWithoutTheCapability``."""
    assert sampling.outdated_engine_reason("http://runner.invalid") == (
        "The engine at http://runner.invalid predates the declared seed policy "
        "at one sample per item: its capabilities do not list "
        "remoteStudy.declaredSeedPolicy, so it would run this study's "
        "derivedSHA256 at samplesPerItem 1 with the fixed seed list, the same "
        "seeds for every item. Nothing was submitted.")
    assert "cluster push --site" in sampling.outdated_engine_repair()


def test_only_the_changed_shape_needs_the_rule():
    changed = sampling.declared_policy_changes_seeds
    assert changed(0.7, 1, "derivedSHA256")
    assert not changed(0.0, 1, "derivedSHA256")  # greedy either way
    assert not changed(0.7, 3, "derivedSHA256")  # derived either way
    assert not changed(0.7, 1, "manifestSeeds")
    assert not changed(0.7, 1, None)
    assert not changed(0.7, 1, "derivedSHA256", "multiAgent")


def test_runner_submit_refuses_a_staged_study_an_older_engine_would_misseed(
        tmp_path):
    """`runner submit` names only a staged bundle: its study is read from the
    local workspace when the bundle's name and content hash match it."""
    from steerlab_server.client import runner as runner_api
    root = str(tmp_path)
    es.create("staged", model_id="org/m", revision="abc", root=root)
    raw = es.load_raw("staged", root)
    raw.update({"temperature": 0.7, "seedPolicy": "derivedSHA256"})
    es.save_raw(raw, root)
    staged = {"experiment": "staged", "experimentContentHash":
              Manifest.load("staged", root).content_hash()}

    def check(capabilities, *, verb="run", bundle=staged, workspace=root):
        runner_api.require_declared_seed_policy(
            bundle, verb, capabilities, "http://runner.invalid", workspace)

    with pytest.raises(runner_api.RunnerRefusal) as refusal:
        check({"remoteStudy": {"variantStudySampling": True}})
    assert refusal.value.code == "declaredSeedPolicyUnsupported"
    assert "cluster push --site" in refusal.value.repair_action
    check({"remoteStudy": {"declaredSeedPolicy": True}})  # the engine has it
    check({}, verb="extract")  # a step that samples no study text
    check({}, bundle={**staged, "experimentContentHash": "0" * 64})  # not ours
    check({}, workspace=None)  # no workspace named: nothing to vouch for


# --- a resume never mixes two seed rules ----------------------------------------

def _old_rule(temperature, samples_per_item, seed_policy):
    """The seed rule before issue #3: samplesPerItem alone decided."""
    return bool(temperature) and temperature > 0 and (samples_per_item or 1) > 1


def _checkpointed(root, name, monkeypatch, *, after, **sampling_fields):
    """A run of ``name`` parked after ``after`` sampled generations, made
    under the old seed rule (an engine released before the fix)."""
    prompts = _study(root, name, **sampling_fields)
    _patch(monkeypatch)
    flag = resume_mod.CheckpointFlag()
    counter = [0]

    def generate(model, prompt, *, temperature=0.0, token_ids_out=None, **_):
        counter[0] += 1
        if counter[0] == after:
            flag.request()
        return f"an answer to {prompt} (T={temperature})"
    monkeypatch.setattr(_owner_generate, "generate", generate)
    seen = {}
    with monkeypatch.context() as old_engine:
        old_engine.setattr(sampling, "derives_seeds", _old_rule)
        with pytest.raises(resume_mod.CheckpointRequested):
            tasks.run(name, prompts, root, model_provider=_fake_model,
                      log=lambda *_: None, checkpoint=flag,
                      on_run_directory=lambda d: seen.setdefault("dir", d))
    return prompts, seen["dir"]


def test_a_resume_across_the_seed_rule_change_refuses(tmp_path, monkeypatch):
    """The issue's shape, parked by the old engine after its first item's
    three manifestSeeds records: resuming under the new rule would add one
    derivedSHA256 record per remaining cell to the same run."""
    root = str(tmp_path)
    prompts, run_dir = _checkpointed(
        root, "seeds-resume", monkeypatch, after=3, temperature=0.7,
        samplesPerItem=1, seedPolicy="derivedSHA256", seeds=[11, 22, 33])
    parked = _sampled_records(run_dir)
    assert {r["seedPolicy"] for r in parked} == {"manifestSeeds"}
    before = open(os.path.join(run_dir, "generations.jsonl"), "rb").read()
    _patch(monkeypatch)
    with pytest.raises(resume_mod.ResumeRefusal) as refusal:
        tasks.run("seeds-resume", prompts, root, model_provider=_fake_model,
                  log=lambda *_: None, run_directory=run_dir)
    assert "refusing to mix two seed rules in one run" in str(refusal.value)
    assert "seedPolicy manifestSeeds with up to 3 sample(s)" in str(refusal.value)
    assert refusal.value.gate == "manifestEpoch"
    assert "steerlab-server experiment run seeds-resume" in refusal.value.repair_action
    assert isinstance(refusal.value, resume_mod.ResumeError)
    # Nothing was written: the parked run is still the old engine's.
    assert open(os.path.join(run_dir, "generations.jsonl"), "rb").read() == before
    assert resume_mod.is_resumable(run_dir)


def test_the_refusal_comes_before_the_model_loads(tmp_path, monkeypatch):
    root = str(tmp_path)
    prompts, run_dir = _checkpointed(
        root, "seeds-early", monkeypatch, after=1, temperature=0.7,
        samplesPerItem=1, seedPolicy="derivedSHA256", seeds=[11])
    loads = []

    @contextmanager
    def counting_model(model_id, revision=None):
        loads.append(model_id)
        yield SimpleNamespace(model_id=model_id, revision=revision or "abc")
    # One listed seed: the per-item count agrees, the policy does not.
    with pytest.raises(resume_mod.ResumeRefusal):
        tasks.run("seeds-early", prompts, root, model_provider=counting_model,
                  log=lambda *_: None, run_directory=run_dir)
    assert loads == []


def test_a_resume_under_an_unchanged_rule_continues(tmp_path, monkeypatch):
    """A fixed-list study's rule did not change, so its parked run resumes
    as it always has and completes with every listed seed."""
    root = str(tmp_path)
    prompts, run_dir = _checkpointed(
        root, "seeds-fixed-resume", monkeypatch, after=4, temperature=0.7,
        seeds=[11, 22, 33])
    _patch(monkeypatch)
    resumed = tasks.run("seeds-fixed-resume", prompts, root,
                        model_provider=_fake_model, log=lambda *_: None,
                        run_directory=run_dir)
    assert resumed == run_dir
    records = _sampled_records(run_dir)
    assert len(records) == 2 * 2 * 3
    assert {r["seedPolicy"] for r in records} == {"manifestSeeds"}


def test_a_shard_stamp_from_the_old_rule_refuses(tmp_path):
    """A shard partial with no sampled record yet still names, in its
    expectedKeys, sample indices the new rule no longer generates."""
    from steerlab_server.experiment import run_preflight
    run_dir = tmp_path / "partial"
    run_dir.mkdir()
    (run_dir / "shard.json").write_text(json.dumps({
        "shardIndex": 0, "shardCount": 2,
        "expectedKeys": [["baseline", 0, "respondent-001", index, "sampled"]
                         for index in range(3)]}), encoding="utf-8")
    manifest = _manifest({"temperature": 0.7, "seedPolicy": "derivedSHA256",
                          "seeds": [11, 22, 33]})
    with pytest.raises(resume_mod.ResumeRefusal):
        run_preflight.seed_rule_admission(
            str(run_dir), name="seeds", manifest=manifest,
            shard_stamp=json.loads((run_dir / "shard.json").read_text()))
    # The same stamp under a fixed-list study is that study's own plan.
    run_preflight.seed_rule_admission(
        str(run_dir), name="seeds",
        manifest=_manifest({"temperature": 0.7, "seeds": [11, 22, 33]}),
        shard_stamp=json.loads((run_dir / "shard.json").read_text()))


# --- the advisories reach the client's envelopes -------------------------------

@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    monkeypatch.delenv("STEERLAB_ROOT", raising=False)
    monkeypatch.setenv(client_cli.WORKSPACE_ENV, str(root))
    monkeypatch.setattr(cli_envelope, "now", lambda: 1_000.0)
    return root


def _document(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def _verifiable_draft(workspace):
    pairs = workspace / "french-pairs.jsonl"
    pairs.write_text('{"positive": "bonjour", "negative": "hello"}\n',
                     encoding="utf-8")
    assert client_cli.main(["concept", "import", "french", "--file",
                            str(pairs)]) == 0
    assert client_cli.main(["experiment", "create", "demo", "--model", "org/m",
                            "--revision",
                            "0123456789abcdef0123456789abcdef01234567"]) == 0
    assert client_cli.main(["experiment", "attach", "demo", "french"]) == 0
    assert client_cli.main(["experiment", "declare-condition", "demo",
                            "baseline", "--baseline", "--alpha-units",
                            "norm"]) == 0


def test_set_protocol_and_verify_say_when_a_seed_setting_is_not_used(
        workspace, capsys):
    _verifiable_draft(workspace)
    capsys.readouterr()
    assert client_cli.main(["experiment", "set-protocol", "demo",
                            "--set", "seedPolicy=derivedSHA256",
                            "--set", "seeds=[11,22,33]",
                            "--set", "temperature=0.7", "--json"]) == 0
    document = _document(capsys)
    assert document["state"] == "okWithAdvisories"
    [advisory] = document["advisories"]
    assert advisory["code"] == "seedSettingNotUsed"
    assert advisory["detail"] == sampling.unused_seed_settings(
        0.7, 1, "derivedSHA256", [11, 22, 33])[0]

    assert client_cli.main(["experiment", "verify", "demo", "--json"]) == 0
    document = _document(capsys)
    assert document["state"] == "okWithAdvisories"
    assert [a["code"] for a in document["advisories"]] == ["seedSettingNotUsed"]

    # At temperature 0 the note is the greedy one; a fixed-list study gets none.
    assert client_cli.main(["experiment", "set-protocol", "demo",
                            "--set", "seeds=[11]",
                            "--set", "temperature=0", "--json"]) == 0
    [advisory] = _document(capsys)["advisories"]
    assert "greedy" in advisory["detail"]
    assert client_cli.main(["experiment", "set-protocol", "demo",
                            "--set", "seedPolicy=manifestSeeds",
                            "--json"]) == 0
    document = _document(capsys)
    assert document["state"] == "ready"
    assert not document.get("advisories")
