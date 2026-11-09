"""`steerlab agent create` (public issue #17) and an agent's thinking level
(public issue #16).

An agent made by hand previews first and writes nothing; the same command
repeated with the reviewed digest writes the agent file into the library with
a `handmade` record. Every check refuses with a plain reason and a repair.
The file's effort and its legacy thinking switch must agree. Same verb, keys,
and sentences as the Mac (`StudyAgentCreationTests`).
"""
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from steerlab_server import client_cli
from steerlab_server.experiment import (agent_identity, condition_execution, experiment_store as store,
                                        model_variant, prompt_render)
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.steering.vector_store import ConceptVectors, SteeringVectorSidecar, save

MODEL = "test/model"
REVISION = "a" * 40


@pytest.fixture(autouse=True)
def restore_cli_root(monkeypatch):
    if 'STEERLAB_ROOT' in os.environ:
        monkeypatch.setenv('STEERLAB_ROOT', os.environ['STEERLAB_ROOT'])
    else:
        monkeypatch.delenv('STEERLAB_ROOT', raising=False)


def cli(root, capsys, *args):
    code = client_cli.main([*args, '--root', str(root), '--json'])
    return code, json.loads(capsys.readouterr().out)


def sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_vector(root, *, model=MODEL, revision=REVISION, norms=True, name="formality"):
    run = root / "runs" / "20261108T000000000-extract"
    run.mkdir(parents=True, exist_ok=True)
    vectors = ConceptVectors(per_layer=[[1.0, 0.0], [0.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
    sidecar = SteeringVectorSidecar.make(
        model_id=model, concept=name, stimulus_set_hash="h", vectors=vectors,
        revision=revision,
        **({"residual_norm_per_layer": [10.0] * 4, "residual_norm_source": "neutral"}
           if norms else {}))
    save(vectors, sidecar, str(run), name)
    reference = f"runs/{run.name}/{name}"
    return reference, sha(root / f"{reference}.safetensors")


def make_adapter(root, *, model=MODEL, revision=REVISION):
    run = root / "runs" / "20261108T000000000-lora-tone"
    folder = run / "tone"
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    (folder / "adapter_model.safetensors").write_bytes(b"weights")
    (run / "tone.json").write_text(json.dumps({
        "name": "tone", "baseModelID": model, "revision": revision,
        "adapterFormat": "hf-peft-lora",
        "adapterBytesHash": sha(folder / "adapter_model.safetensors"),
        "adapterConfigHash": sha(folder / "adapter_config.json")}))
    return f"runs/{run.name}/tone", sha(folder / "adapter_model.safetensors")


def base_args(name="steered"):
    return ["agent", "create", name, "--model", MODEL, "--revision", REVISION]


def create(root, capsys, *args):
    code, preview = cli(root, capsys, *args)
    assert code == 0, preview
    digest = preview["result"]["reviewSHA256"]
    code, written = cli(root, capsys, *args, "--review-sha256", digest)
    assert code == 0, written
    return preview["result"], written["result"]


# --- preview, then the reviewed write -------------------------------------------


def test_preview_writes_nothing_and_the_reviewed_write_makes_a_library_agent(tmp_path, capsys):
    reference, vector_sha = make_vector(tmp_path)
    adapter, adapter_sha = make_adapter(tmp_path)
    prompt = tmp_path / "persona.txt"
    prompt.write_text("Answer as a careful reviewer.")
    args = base_args() + ["--adapter", adapter, "--adapter-sha256", adapter_sha,
                          "--vector", reference, "--vector-sha256", vector_sha,
                          "--layer", "2", "--strength", "0.12",
                          "--system-prompt-file", str(prompt),
                          "--reason", "the published setting from a prior study"]
    code, preview = cli(tmp_path, capsys, *args)
    assert code == 0, preview
    planned = preview["result"]
    assert planned["written"] is False
    assert not (tmp_path / "runs" / "model-variants").exists()
    assert "--review-sha256" in planned["confirmCommand"]
    assert planned["origin"]["kind"] == "handmade"

    code, written = cli(tmp_path, capsys, *args, "--review-sha256", planned["reviewSHA256"])
    assert code == 0, written
    result = written["result"]
    path = tmp_path / result["path"]
    assert result["path"].startswith("runs/model-variants/")
    assert path.name == "model-variant.json"
    assert sha(path) == result["artifactFileSHA256"]
    assert json.loads((path.parent / "config.json").read_text())["runType"] == "variant-save"
    document = json.loads(path.read_text())
    assert document["baseRevision"] == REVISION
    assert document["alphaInNormUnits"] is True and document["bandWidth"] == 1
    assert document["injections"] == [{"concept": "formality", "vectorArtifactID": reference,
                                       "layer": 2, "alpha": 0.12}]
    assert document["adapters"][0]["adapterHash"] == adapter_sha
    assert document["systemPromptHash"] == hashlib.sha256(
        b"Answer as a careful reviewer.").hexdigest()
    assert "reasoningEffort" not in document and document["qwenThinkingEnabled"] is False
    assert "promotion" not in document
    # The handmade record: the command, its inputs and their hashes, the time
    # it was made, and the stated reason.
    record = document["handmade"]
    assert record["madeWith"] == "agent create"
    assert record["command"][:3] == ["steerlab", "agent", "create"]
    assert "--review-sha256" not in record["command"]
    assert record["reason"] == "the published setting from a prior study"
    assert record["createdAt"] == document["createdAt"]
    assert record["inputs"]["vectors"][0]["artifactSHA256"] == vector_sha
    assert record["inputs"]["adapters"][0]["adapterSHA256"] == adapter_sha
    assert record["inputs"]["systemPromptFile"]["sha256"] == sha(prompt)
    # The engine reads it as the agent it describes.
    variant = model_variant.ModelVariant.from_dict(document)
    assert variant.handmade == record and variant.promotion is None


def test_a_bare_agent_with_no_vectors_is_allowed(tmp_path, capsys):
    _, result = create(tmp_path, capsys, *base_args("plain"))
    document = json.loads((tmp_path / result["path"]).read_text())
    assert document["injections"] == [] and document["adapters"] == []
    assert result["origin"]["summary"] == ("Made by hand with agent create. The researcher "
                                           "chose its settings. No reason was stated.")


def test_a_stale_review_digest_refuses_and_writes_nothing(tmp_path, capsys):
    reference, vector_sha = make_vector(tmp_path)
    args = base_args() + ["--vector", reference, "--vector-sha256", vector_sha,
                          "--layer", "2", "--strength", "0.1"]
    code, preview = cli(tmp_path, capsys, *args)
    stale = preview["result"]["reviewSHA256"]
    code, refused = cli(tmp_path, capsys, *base_args() + [
        "--vector", reference, "--vector-sha256", vector_sha, "--layer", "2",
        "--strength", "0.2", "--review-sha256", stale])
    assert code == 65
    assert refused["error"]["code"] == "staleManifest"
    assert "Preview again" in refused["error"]["repairAction"]
    assert not (tmp_path / "runs" / "model-variants").exists()


# --- each check refuses with its repair -------------------------------------------


@pytest.mark.parametrize("case", [
    "revision", "vectorHash", "vectorModel", "vectorRevision", "layer", "emptyLayer",
    "norms", "adapterHash", "adapterModel", "twoAdapters", "effort", "name"])
def test_each_check_refuses_plainly_with_a_repair(tmp_path, capsys, case):
    reference, vector_sha = make_vector(tmp_path)
    adapter, adapter_sha = make_adapter(tmp_path)
    vector = ["--vector", reference, "--vector-sha256", vector_sha, "--layer", "2",
              "--strength", "0.1"]
    args = base_args() + vector
    expect = {}
    if case == "revision":
        args = ["agent", "create", "steered", "--model", MODEL, "--revision", "main"] + vector
        expect = {"reason": "is not a pinned commit", "repair": "40-character commit hash"}
    elif case == "vectorHash":
        args = base_args() + ["--vector", reference, "--vector-sha256", "0" * 64,
                              "--layer", "2", "--strength", "0.1"]
        expect = {"reason": "not the one you reviewed", "repair": "inspect-artifact"}
    elif case == "vectorModel":
        other, other_sha = make_vector(tmp_path, model="other/model", name="other")
        args = base_args() + ["--vector", other, "--vector-sha256", other_sha,
                              "--layer", "2", "--strength", "0.1"]
        expect = {"reason": "was extracted from other/model", "repair": "base model"}
    elif case == "vectorRevision":
        other, other_sha = make_vector(tmp_path, revision="b" * 40, name="other")
        args = base_args() + ["--vector", other, "--vector-sha256", other_sha,
                              "--layer", "2", "--strength", "0.1"]
        expect = {"reason": "revision " + "b" * 40, "repair": "--revision"}
    elif case == "layer":
        args = base_args() + ["--vector", reference, "--vector-sha256", vector_sha,
                              "--layer", "9", "--strength", "0.1"]
        expect = {"reason": "Layer 9 is not in the vector", "repair": "--layer"}
    elif case == "emptyLayer":
        args = base_args() + ["--vector", reference, "--vector-sha256", vector_sha,
                              "--layer", "1", "--strength", "0.1"]
        expect = {"reason": "is empty at layer 1", "repair": "normsPerLayer"}
    elif case == "norms":
        bare, bare_sha = make_vector(tmp_path, norms=False, name="bare")
        args = base_args() + ["--vector", bare, "--vector-sha256", bare_sha,
                              "--layer", "2", "--strength", "0.1"]
        expect = {"reason": "no residual norm at layer 2", "repair": "backfill-norms"}
    elif case == "adapterHash":
        args = base_args() + ["--adapter", adapter, "--adapter-sha256", "1" * 64]
        expect = {"reason": "not the ones you reviewed", "repair": "shasum -a 256"}
    elif case == "adapterModel":
        record = tmp_path / f"{adapter}.json"
        data = json.loads(record.read_text())
        data["baseModelID"] = "other/model"
        record.write_text(json.dumps(data))
        args = base_args() + ["--adapter", adapter, "--adapter-sha256", adapter_sha]
        expect = {"reason": "was trained on other/model", "repair": "trained on the agent's base model"}
    elif case == "twoAdapters":
        args = base_args() + ["--adapter", adapter, "--adapter-sha256", adapter_sha,
                              "--adapter", adapter, "--adapter-sha256", adapter_sha]
        expect = {"reason": "An agent runs one adapter", "repair": "Give one --adapter"}
    elif case == "effort":
        # The id heuristic: this model has no thinking switch.
        args = base_args() + ["--reasoning-effort", "medium"]
        expect = {"reason": "has no thinking switch", "repair": "--reasoning-effort off"}
    elif case == "name":
        create(tmp_path, capsys, *base_args("taken"))
        args = base_args("taken")
        expect = {"reason": "already exists", "repair": "agent list"}
    code, refused = cli(tmp_path, capsys, *args)
    assert code == 65, refused
    assert expect["reason"] in refused["error"]["reason"], refused["error"]
    assert expect["repair"] in refused["error"]["repairAction"], refused["error"]
    library = tmp_path / "runs" / "model-variants"
    written = list(library.glob("*/model-variant.json")) if library.exists() else []
    assert len(written) == (1 if case == "name" else 0)


def test_there_is_no_temperature_flag(tmp_path, capsys):
    code, refused = cli(tmp_path, capsys, *base_args() + ["--temperature", "0.7"])
    assert code == 64
    assert "--temperature" in json.dumps(refused)


# --- the thinking level on the agent (#16) -------------------------------------------


def qwen_capabilities():
    return prompt_render.capabilities_for("Qwen/Qwen3.8-27B")


def test_a_level_the_template_accepts_is_written_with_its_shadow(tmp_path, capsys):
    args = ["agent", "create", "thinker", "--model", "Qwen/Qwen3.8-27B",
            "--revision", REVISION, "--reasoning-effort", "medium"]
    _, result = create(tmp_path, capsys, *args)
    document = json.loads((tmp_path / result["path"]).read_text())
    assert document["reasoningEffort"] == "medium"
    assert document["qwenThinkingEnabled"] is True
    assert result["origin"]["reasoningEffort"] == "medium"
    assert result["origin"]["reasoningEffortSource"] == "agent"


@pytest.mark.parametrize("effort, thinking", [("medium", False), ("off", True)])
def test_a_file_whose_effort_and_switch_disagree_is_refused(effort, thinking):
    document = {"name": "a", "baseModelID": MODEL, "reasoningEffort": effort,
                "qwenThinkingEnabled": thinking}
    with pytest.raises(ValueError) as refused:
        model_variant.ModelVariant.from_dict(document)
    assert "one says the agent thinks and the other says it does not" in str(refused.value)


def test_an_unknown_effort_and_two_records_are_refused():
    with pytest.raises(ValueError, match="not a level SteerLab knows"):
        model_variant.ModelVariant.from_dict(
            {"name": "a", "baseModelID": MODEL, "reasoningEffort": "hgih"})
    with pytest.raises(ValueError, match="An agent is made one way"):
        model_variant.ModelVariant.from_dict(
            {"name": "a", "baseModelID": MODEL, "promotion": {}, "handmade": {}})


@pytest.mark.parametrize("document, expected", [
    ({"qwenThinkingEnabled": False}, "off"),
    ({"qwenThinkingEnabled": True}, "xhigh"),
    ({}, "off"),
    ({"reasoningEffort": "medium", "qwenThinkingEnabled": True}, "medium"),
    ({"reasoningEffort": "on"}, "on"),
    ({"reasoningEffort": "off", "qwenThinkingEnabled": False}, "off"),
])
def test_the_helper_reads_the_level_or_the_legacy_switch(document, expected):
    variant = model_variant.ModelVariant.from_dict(
        {"name": "a", "baseModelID": MODEL, **document})
    assert agent_identity.variant_reasoning_effort(variant) == expected
    assert variant.qwen_thinking_enabled is (expected != "off")
    # Round trip: the writer keeps the level and writes the agreeing shadow.
    again = model_variant.ModelVariant.from_dict(variant.to_dict())
    assert agent_identity.variant_reasoning_effort(again) == expected


def test_a_legacy_agent_file_round_trips_without_a_new_key():
    legacy = {"name": "a", "baseModelID": MODEL, "qwenThinkingEnabled": True}
    written = model_variant.ModelVariant.from_dict(legacy).to_dict()
    assert "reasoningEffort" not in written and "handmade" not in written
    assert written["qwenThinkingEnabled"] is True


def effective(artifact, **manifest):
    manifest = Manifest.from_dict({"name": "s", "modelID": "Qwen/Qwen3.8-27B", "concepts": [],
                                   "conditions": [], **manifest})
    vc = SimpleNamespace(name="agent", artifact_path="runs/x/a.json", artifact_hash="a" * 64,
                         artifact={"name": "agent", "baseModelID": "Qwen/Qwen3.8-27B",
                                   "alphaInNormUnits": False, **artifact})
    model = SimpleNamespace(model_id="Qwen/Qwen3.8-27B", revision="abc")
    return manifest, condition_execution.effective_variant_condition(
        vc, manifest, model, None, wants_choice=False)


def test_an_ordinary_study_runs_the_agent_at_its_own_level():
    # The agent declares medium; the study itself reasons at xhigh with a
    # budget. The agent's level wins, the study's budget applies.
    manifest, eff = effective({"reasoningEffort": "medium", "qwenThinkingEnabled": True},
                              reasoningEffort="xhigh", reasoningMaxTokens=256)
    assert eff.reasoning_effort == "medium"
    assert eff.qwen_thinking_enabled is True
    assert eff.reasoning_max_tokens == 256
    assert eff.provenance["agentReasoningEffort"] == "medium"
    # A legacy agent reads as it always did.
    _, legacy = effective({"qwenThinkingEnabled": True})
    assert legacy.reasoning_effort == "xhigh"
    assert legacy.provenance["agentReasoningEffort"] == "xhigh"
    # An agent that does not think gets no budget, whatever the study's.
    _, off = effective({"reasoningEffort": "off", "qwenThinkingEnabled": False},
                       reasoningEffort="xhigh", reasoningMaxTokens=256)
    assert off.reasoning_effort == "off" and off.reasoning_max_tokens is None


def test_a_declared_level_reaches_the_renderer_without_a_budget():
    # No study budget: the agent's medium is handed to the template, not
    # dropped back to the legacy boolean's xhigh. Off and xhigh keep the
    # exact historical call.
    assert agent_identity.effort_reaches_renderer("medium", None) is True
    assert agent_identity.effort_reaches_renderer("on", None) is True
    assert agent_identity.effort_reaches_renderer("xhigh", None) is False
    assert agent_identity.effort_reaches_renderer("off", None) is False
    assert agent_identity.effort_reaches_renderer("off", 128) is True
    manifest, _ = effective({"reasoningEffort": "medium"})
    manifest.raw["variantConditions"] = []
    settings = condition_execution.run_arm_render_settings(
        SimpleNamespace(system_prompt=None, prompt_mode="chatAssistant",
                        qwen_thinking_enabled=False, reasoning_effort="off",
                        reasoning_max_tokens=None,
                        variant_conditions=[SimpleNamespace(
                            name="agent", artifact_path="runs/x/a.json",
                            artifact={"name": "agent", "baseModelID": "Qwen/Qwen3.8-27B",
                                      "reasoningEffort": "medium"})]), None)
    assert settings[-1]["reasoning_effort"] == "medium"
    assert settings[-1]["qwen_thinking_enabled"] is True


def test_a_run_hands_the_agents_level_to_generation_and_stamps_it(tmp_path, monkeypatch):
    """End to end through the run loop: an agent at `medium` in a study with
    no reasoning budget generates with reasoning_effort=medium (not the
    boolean's xhigh), a legacy thinking agent keeps its exact historical call,
    and every agent record stamps the effort it ran at."""
    import test_variant_study_sampling as fixture
    from steerlab_server.experiment import tasks
    root = str(tmp_path)
    qwen = "Qwen/Qwen3.8-27B"  # a template with a thinking switch (id heuristic)
    prompts = fixture._study_fixture(
        root, "effort-run",
        agent_a=fixture._artifact("agent-a", reasoningEffort="medium",
                                  qwenThinkingEnabled=True, baseModelID=qwen),
        agent_b=fixture._artifact("agent-b", qwenThinkingEnabled=True, baseModelID=qwen))
    raw = store.load_raw("effort-run", root)
    raw["outcomeInstruments"] = ["sampledText"]
    raw["modelID"] = qwen
    store.save_raw(raw, root)
    calls = []

    def generate(model, prompt, *, model_id=None, max_tokens=0, temperature=0.0,
                 injections=None, prompt_mode=None, system_prompt=None,
                 qwen_thinking_enabled=False, token_ids_out=None, **reasoning):
        calls.append({"thinking": qwen_thinking_enabled, **reasoning})
        return "an answer"

    fixture._patch(monkeypatch, generate, fixture._fake_score_options())
    run_dir = tasks.run("effort-run", prompts, root, model_provider=fixture._fake_model,
                        log=lambda *_: None)
    records = fixture._sampled(fixture._records(run_dir))
    by_condition = {}
    for record, call in zip(records, calls):
        by_condition.setdefault(record["condition"], set()).add(
            (call["thinking"], call.get("reasoning_effort"), record.get("agentReasoningEffort")))
    assert by_condition["agent-a"] == {(True, "medium", "medium")}
    assert by_condition["agent-b"] == {(True, None, "xhigh")}
    assert by_condition["baseline"] == {(False, None, None)}


def test_the_template_rule_is_the_study_rule():
    record = qwen_capabilities()
    ok = model_variant.ModelVariant(name="a", base_model_id="Qwen/Qwen3.8-27B",
                                    reasoning_effort="medium")
    assert agent_identity.variant_effort_violations(ok, record) == []
    none = model_variant.ModelVariant(name="a", base_model_id=MODEL, reasoning_effort="on")
    assert agent_identity.variant_effort_violations(none) == [
        prompt_render.effort_without_thinking_mode_reason("on", MODEL)]


def test_promotion_takes_the_level_the_sweep_ran_at(tmp_path):
    from steerlab_server.experiment import promote
    for run, snapshot in (("declared", {"reasoningEffort": "medium", "reasoningMaxTokens": 64}),
                          ("legacy", {"qwenThinkingEnabled": True}),
                          ("off", {"reasoningEffort": "off"})):
        (tmp_path / "runs" / run).mkdir(parents=True)
        (tmp_path / "runs" / run / "experiment.json").write_text(json.dumps(snapshot))
    level = lambda run: promote._sweep_run_level(run, "s", "c", str(tmp_path))  # noqa: E731
    assert level("declared") == ("medium", True)
    assert level("legacy") == (None, True)       # the bytes such a sweep always promoted
    assert level("off") == (None, False)
    with pytest.raises(promote.PromoteError, match="unknown"):
        level("missing")


# --- swept and handmade, everywhere ------------------------------------------------


def test_inspect_list_and_attach_name_the_kind(tmp_path, capsys):
    _, result = create(tmp_path, capsys, *base_args("plain"))
    code, inspected = cli(tmp_path, capsys, "agent", "inspect", result["path"])
    assert code == 0
    agent = inspected["result"]["agent"]
    assert agent["kind"] == "handmade"
    assert agent["reasoningEffort"] == "off" and agent["reasoningEffortSource"] == "thinkingSwitch"
    code, listed = cli(tmp_path, capsys, "agent", "list")
    assert listed["result"]["agents"][0]["agent"]["kind"] == "handmade"
    swept = dict(json.loads((tmp_path / result["path"]).read_text()))
    swept.pop("handmade")
    swept["promotion"] = {"experiment": "screen", "promotedBy": "criterion"}
    assert agent_identity.agent_origin(swept)["summary"] == (
        "A sweep chose this agent's settings: experiment promote made it from study "
        "screen, using the cell the study's declared rule selected.")

    created = store.create("study", model_id=MODEL, root=str(tmp_path))
    code, attached = cli(tmp_path, capsys, "experiment", "attach-agent", "study",
                         "--artifact", result["path"],
                         "--artifact-sha256", result["artifactFileSHA256"],
                         "--manifest-sha256", created.get("manifestFileSHA256")
                         or _manifest_sha(tmp_path))
    assert code == 0, attached
    assert attached["result"]["agent"]["kind"] == "handmade"


def _manifest_sha(root):
    return hashlib.sha256((root / "experiments" / "study" / "experiment.json")
                          .read_bytes()).hexdigest()


def test_a_handmade_steered_agent_passes_the_same_freeze_gate(tmp_path, capsys):
    from steerlab_server.experiment import freeze_policy
    reference, vector_sha = make_vector(tmp_path)
    _, result = create(tmp_path, capsys, *base_args() + [
        "--vector", reference, "--vector-sha256", vector_sha, "--layer", "2",
        "--strength", "0.1"])
    document = json.loads((tmp_path / result["path"]).read_text())
    study = {"variantConditions": [{"name": "steered", "artifactPath": result["path"],
                                    "artifactHash": result["artifactFileSHA256"],
                                    "artifact": document}]}
    freeze_policy.check_variant_validity("s", study, evidence_grade_variants=set())
    assert "handmade" in document


def test_a_new_policy_version_drops_the_handmade_record(tmp_path, capsys):
    # The record describes the agent it was made for; a new version made by
    # policy-attach is another agent, as it is for a promotion record.
    from steerlab_server.experiment import policy_authoring
    _, result = create(tmp_path, capsys, *base_args("plain"))
    plan = policy_authoring.attachment_review(
        {"agentPath": result["path"], "name": "plain-v2", "policyPaths": []}, tmp_path)
    assert "handmade" not in plan["document"]
    assert "handmade" in json.loads((tmp_path / result["path"]).read_text())
