"""FR6-B round 2: no machine paths in an agent's record, the agent-arm rules
at verify, submission, and run start, the budget for thinking agents, the
revision check at attach, true repairs, and promotion at the sweep's level.

Python twin of the round-2 cases in ``StudyAgentCreationTests``.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from steerlab_server import client_cli
from steerlab_server.experiment import (agent_identity, condition_execution,
                                        experiment_store as store, lifecycle_gates,
                                        model_capabilities as mc, prompt_render,
                                        study_admission)
from steerlab_server.experiment.manifest import Manifest

import test_agent_create as base

MODEL = base.MODEL
QWEN = "Qwen/Qwen3.8-27B"
REVISION = base.REVISION
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def restore_cli_root(monkeypatch):
    if 'STEERLAB_ROOT' in os.environ:
        monkeypatch.setenv('STEERLAB_ROOT', os.environ['STEERLAB_ROOT'])
    else:
        monkeypatch.delenv('STEERLAB_ROOT', raising=False)


# --- B1: no home folder or machine path in a written agent --------------------------


@pytest.mark.parametrize("spelling", ["absolute", "tilde"])
def test_a_file_outside_the_workspace_is_recorded_by_name_and_hash(
        tmp_path, capsys, monkeypatch, spelling):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    names = Path(os.path.expanduser("~/.steerlab/private-names.txt"))
    home = tmp_path / "home" / "researcher"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    prompt = home / "persona.txt"
    prompt.write_text("Answer as a careful reviewer.")
    given = str(prompt) if spelling == "absolute" else "~/persona.txt"
    _, result = base.create(workspace, capsys, *base.base_args() + [
        "--system-prompt-file", given])
    written = (workspace / result["path"]).read_text()
    assert str(home) not in written and str(tmp_path) not in written
    home_root = "/" + "Users" + "/"  # the macOS home root, spelled apart
    assert home_root not in written and "~/" not in written
    record = json.loads(written)["handmade"]
    assert record["inputs"]["systemPromptFile"] == {
        "name": "persona.txt",
        "sha256": hashlib.sha256(b"Answer as a careful reviewer.").hexdigest()}
    assert record["command"][record["command"].index("--system-prompt-file") + 1] \
        == "persona.txt"
    # The same file is clean by the release's own artifact scan.
    env = {k: v for k, v in os.environ.items() if k != "STEERLAB_REQUIRE_PRIVATE_NAMES"}
    if names.is_file():
        env["STEERLAB_PRIVATE_NAMES_FILE"] = str(names)
    scan = subprocess.run(
        [sys.executable, str(REPO / "scripts/ci/artifact_scan.py"),
         "--allow-missing-list", str(workspace / result["path"])],
        capture_output=True, text=True, env=env)
    assert scan.returncode == 0, scan.stdout + scan.stderr


def test_a_file_inside_the_workspace_is_recorded_relative(tmp_path, capsys):
    (tmp_path / "prompts").mkdir()
    prompt = tmp_path / "prompts" / "persona.txt"
    prompt.write_text("Be brief.")
    _, result = base.create(tmp_path, capsys, *base.base_args() + [
        "--system-prompt-file", str(prompt)])
    record = json.loads((tmp_path / result["path"]).read_text())["handmade"]
    assert record["inputs"]["systemPromptFile"]["path"] == "prompts/persona.txt"
    assert str(tmp_path) not in json.dumps(record)


def test_no_temperature_is_recorded_as_chosen(tmp_path, capsys):
    _, result = base.create(tmp_path, capsys, *base.base_args("plain"))
    document = json.loads((tmp_path / result["path"]).read_text())
    assert document["handmade"]["inputs"]["temperature"] is None


# --- agent create: ablation, band, names, repairs, adapters ------------------------


def test_band_width_and_centering_follow_the_editor(tmp_path, capsys):
    reference, digest = base.make_vector(tmp_path)
    code, preview = base.cli(tmp_path, capsys, *base.base_args() + [
        "--vector", reference, "--vector-sha256", digest, "--layer", "2",
        "--strength", "1", "--mode", "ablate", "--band-width", "3"])
    assert code == 0, preview
    planned = preview["result"]
    assert planned["document"]["bandWidth"] == 3
    # A raw ablation is allowed, and the preview says what it risks.
    assert any("ablated without centering" in note for note in planned["advisories"])
    code, refused = base.cli(tmp_path, capsys, *base.base_args() + [
        "--vector", reference, "--vector-sha256", digest, "--layer", "2",
        "--strength", "1", "--mode", "ablate", "--centering", "neutralMean"])
    assert code == 65 and "stores no neutral mean" in refused["error"]["reason"]
    code, refused = base.cli(tmp_path, capsys, *base.base_args() + ["--band-width", "4"])
    assert code == 64


def test_names_are_compared_ignoring_case(tmp_path, capsys):
    base.create(tmp_path, capsys, *base.base_args("taken"))
    code, refused = base.cli(tmp_path, capsys, *base.base_args("TAKEN"))
    assert code == 65 and "ignoring case" in refused["error"]["reason"]


def test_an_adapter_trained_on_a_branch_is_read_by_its_resolved_commit(tmp_path, capsys):
    adapter, digest = base.make_adapter(tmp_path)
    record = tmp_path / f"{adapter}.json"
    data = json.loads(record.read_text())
    data.update(revision="main", revisionResolved=REVISION)
    record.write_text(json.dumps(data))
    code, preview = base.cli(tmp_path, capsys, *base.base_args() + [
        "--adapter", adapter, "--adapter-sha256", digest])
    assert code == 0, preview


def test_the_repairs_name_commands_that_parse(tmp_path, capsys):
    code, refused = base.cli(tmp_path, capsys, "agent", "create", "a", "--model", MODEL,
                             "--revision", "main")
    repair = refused["error"]["repairAction"]
    assert "--runner <url>" in repair
    # The command the repair names is one this client accepts.
    client_cli.parse("model", ["plan", MODEL, "--runner", "http://127.0.0.1:9", "--json"])
    # A heuristic record's repair lists the levels it only assumes.
    capabilities = prompt_render.capabilities_for(QWEN)
    repair = agent_identity.effort_repair(capabilities)
    assert "accepts: " not in repair
    assert "assumed from the model id, with no probed template: low, medium, xhigh" in repair


# --- B2: the agent-arm rules at verify, submission, and run start ------------------


def study_with_arm(root, artifact, *, model=MODEL, revision=REVISION, **settings):
    store.create("study", model_id=model, revision=revision, root=str(root))
    raw = store.load_raw("study", str(root))
    raw["variantConditions"] = [{"name": "arm", "artifactPath": "runs/x/a.json",
                                 "artifactHash": "a" * 64, "artifact": {
                                     "name": "arm", "baseModelID": model,
                                     "alphaInNormUnits": True, **artifact}}]
    raw.update(settings)
    store.save_raw(raw, str(root))
    return Manifest.load("study", str(root))


@pytest.mark.parametrize("case, artifact, settings, phrase", [
    ("effort", {"reasoningEffort": "medium"}, {}, "has no thinking switch"),
    ("fields", {"reasoningEffort": "medium", "qwenThinkingEnabled": False}, {},
     "one says the agent thinks"),
    ("unknown", {"reasoningEffort": 5}, {}, "reasoningEffort 5 is not a level"),
    ("adapters", {"adapters": [{"adapterDirectory": "a"}, {"adapterDirectory": "b"}]}, {},
     "lists 2 adapters"),
    ("instrument", {"qwenThinkingEnabled": True},
     {"outcomeInstruments": ["answerTokenLogprob"]}, "answer-token instrument"),
    ("revision", {"baseRevision": "b" * 40}, {}, "made for revision " + "b" * 40),
])
def test_a_draft_verify_names_each_agent_arm_rule(tmp_path, case, artifact, settings, phrase):
    manifest = study_with_arm(tmp_path, artifact, **settings)
    violations = manifest.verify(str(tmp_path))
    assert any(phrase in v and "agent arm 'arm'" in v for v in violations), violations


def test_a_missing_layer_or_norm_is_named_from_the_vector_sidecar(tmp_path):
    reference, _ = base.make_vector(tmp_path, norms=False, name="bare")
    manifest = study_with_arm(tmp_path, {"injections": [
        {"concept": "bare", "vectorArtifactID": reference, "layer": 9, "alpha": 0.1}]})
    assert any("at layer 9, which its vector does not have" in v
               for v in manifest.verify(str(tmp_path)))
    manifest = study_with_arm(tmp_path / "second", {"injections": [
        {"concept": "bare", "vectorArtifactID": str(tmp_path / reference), "layer": 2,
         "alpha": 0.1}]})
    assert any("records no residual norm at layer 2" in v
               for v in manifest.verify(str(tmp_path / "second")))


def test_an_ignored_level_is_refused_by_the_workspace_record(tmp_path):
    record = mc.heuristic(QWEN, REVISION)
    record["source"] = "probe"
    record["detected"]["effortLevels"] = {"low": "ignored", "medium": "ignored",
                                          "xhigh": "ignored"}
    record["detected"]["effortVariableRead"] = False
    record["advisories"] = []
    mc.write_record(record, str(tmp_path))
    manifest = study_with_arm(tmp_path, {"reasoningEffort": "medium"}, model=QWEN)
    assert any("ignores reasoning_effort" in v for v in manifest.verify(str(tmp_path)))
    # And if such an agent ever ran, its records would say what rendered.
    capabilities = mc.resolve(QWEN, REVISION, str(tmp_path))
    assert agent_identity.rendered_effort("medium", QWEN, capabilities) == "on"
    assert agent_identity.rendered_effort("medium", MODEL) == "off"
    assert agent_identity.rendered_effort("medium", QWEN) == "medium"


def test_a_record_stamps_what_reached_the_renderer(tmp_path):
    # A legacy thinking agent on a template with no thinking switch renders
    # no thinking at all; its records say off, not the switch's xhigh.
    manifest = study_with_arm(tmp_path, {"qwenThinkingEnabled": True})
    vc = SimpleNamespace(name="arm", artifact_path="runs/x/a.json", artifact_hash="a" * 64,
                         artifact=manifest.raw["variantConditions"][0]["artifact"])
    eff = condition_execution.effective_variant_condition(
        vc, manifest, SimpleNamespace(model_id=MODEL, revision=REVISION), str(tmp_path),
        wants_choice=False)
    assert eff.reasoning_effort == "xhigh"
    assert eff.provenance["agentReasoningEffort"] == "off"


def test_a_frozen_study_is_told_at_verify_and_refused_at_submission_and_run_start(tmp_path):
    from steerlab_server.api import submissions
    manifest = study_with_arm(tmp_path, {"qwenThinkingEnabled": True},
                              outcomeInstruments=["answerTokenLogprob"])
    raw = store.load_raw("study", str(tmp_path))
    raw["status"] = "frozen"
    store.save_raw(raw, str(tmp_path))
    manifest = Manifest.load("study", str(tmp_path))
    assert not any("agent arm" in v for v in manifest.verify(str(tmp_path)))
    note = agent_identity.frozen_arm_problems_advisory(manifest, str(tmp_path))
    assert note.startswith("this study will be refused when its run is submitted or started")
    assert "experiment duplicate study study-v2" in note
    refusal = study_admission.settings_refusal(manifest, str(tmp_path))
    assert refusal is not None and refusal.gate == lifecycle_gates.STUDY_DECLARATION
    with pytest.raises(submissions.SubmissionRefusal, match="answer-token instrument"):
        submissions._refuse_settings(manifest, "run", ["run"])
    with pytest.raises(lifecycle_gates.LifecycleError):
        study_admission.refuse_settings(manifest, str(tmp_path))


def test_the_client_verify_reports_the_frozen_note(tmp_path, capsys, monkeypatch):
    study_with_arm(tmp_path, {"qwenThinkingEnabled": True},
                   outcomeInstruments=["answerTokenLogprob"])
    raw = store.load_raw("study", str(tmp_path))
    raw["status"] = "frozen"
    store.save_raw(raw, str(tmp_path))
    # This fixture is not a real freeze, so its pins would not verify; the
    # wiring under test is the advisory a verified frozen study carries.
    monkeypatch.setattr(Manifest, "verify", lambda self, root=None: [])
    code, document = base.cli(tmp_path, capsys, "experiment", "verify", "study")
    notes = [a for a in document.get("advisories", []) if a["code"] == "refusedWhenRun"]
    assert any("answer-token instrument" in a["detail"] for a in notes), document


# --- S4: a budget for the agent arms that think -----------------------------------------


def test_an_off_study_may_declare_a_budget_for_its_thinking_agents(tmp_path):
    from steerlab_server.experiment import draft_protocol_policy
    manifest = study_with_arm(tmp_path, {"reasoningEffort": "on"}, model=QWEN,
                              reasoningEffort="off", reasoningMaxTokens=256)
    assert not any(prompt_render.BUDGET_WITHOUT_EFFORT_REASON in v
                   for v in manifest.verify(str(tmp_path)))
    raw = store.load_raw("study", str(tmp_path))
    draft_protocol_policy.apply_protocol("study", raw, {"reasoningMaxTokens": 128},
                                         capabilities=None)
    assert raw["reasoningMaxTokens"] == 128
    # Without a thinking agent it is refused, as before.
    plain = study_with_arm(tmp_path / "plain", {}, model=QWEN, reasoningEffort="off",
                           reasoningMaxTokens=256)
    assert prompt_render.BUDGET_WITHOUT_EFFORT_REASON in plain.verify(str(tmp_path / "plain"))


def test_the_budget_reaches_only_arms_that_think(tmp_path):
    manifest = study_with_arm(tmp_path, {"reasoningEffort": "on"}, model=QWEN,
                              reasoningEffort="off", reasoningMaxTokens=256)
    model = SimpleNamespace(model_id=QWEN, revision=REVISION)
    vc = SimpleNamespace(name="arm", artifact_path="runs/x/a.json", artifact_hash="a" * 64,
                         artifact=manifest.raw["variantConditions"][0]["artifact"])
    agent = condition_execution.effective_variant_condition(vc, manifest, model,
                                                            str(tmp_path), wants_choice=False)
    assert agent.reasoning_max_tokens == 256
    assert condition_execution.study_arm_budget(manifest) is None
    off_agent = condition_execution.effective_variant_condition(
        SimpleNamespace(name="off", artifact_path="runs/x/b.json", artifact_hash="b" * 64,
                        artifact={"name": "off", "baseModelID": QWEN}),
        manifest, model, str(tmp_path), wants_choice=False)
    assert off_agent.reasoning_max_tokens is None


# --- S7: the revision at attach ---------------------------------------------------------


def test_attach_refuses_an_agent_made_for_another_revision(tmp_path, capsys):
    _, result = base.create(tmp_path, capsys, *base.base_args("plain"))
    created = store.create("study", model_id=MODEL, revision="b" * 40, root=str(tmp_path))
    digest = hashlib.sha256((tmp_path / "experiments/study/experiment.json")
                            .read_bytes()).hexdigest()
    code, refused = base.cli(tmp_path, capsys, "experiment", "attach-agent", "study",
                             "--artifact", result["path"],
                             "--artifact-sha256", result["artifactFileSHA256"],
                             "--manifest-sha256", digest)
    assert code == 65 and "made for revision" in refused["error"]["reason"]
    assert created


# --- S1: promotion at the sweep's level -------------------------------------------------


def _promotable_at(root, effort):
    import test_sweep_promote as sweep
    sweep._promotable(root)
    raw = store.load_raw("pr", root)
    raw.update(reasoningEffort=effort, reasoningMaxTokens=64)
    store.save_raw(raw, root)
    return sweep


def test_minting_writes_the_sweeps_level_and_reuse_refuses_another(tmp_path):
    from steerlab_server.experiment import promote
    root = str(tmp_path)
    _promotable_at(root, "medium")
    minted = promote.promote("pr", "fear", root=root, log=lambda *_: None)
    assert minted["variant"]["reasoningEffort"] == "medium"
    assert json.loads(Path(minted["path"]).read_text())["reasoningEffort"] == "medium"
    # A retry returns the same agent…
    again = promote.promote("pr", "fear", root=root, log=lambda *_: None)
    assert again.get("idempotentReuse") is True
    # …but not one minted before promotion carried the level.
    stale = json.loads(Path(minted["path"]).read_text())
    stale.pop("reasoningEffort")
    stale["qwenThinkingEnabled"] = True
    Path(minted["path"]).write_text(json.dumps(stale))
    with pytest.raises(promote.PromoteError, match="thinks at reasoning effort xhigh") as refused:
        promote.promote("pr", "fear", root=root, log=lambda *_: None)
    assert "--agent-name" in refused.value.repair_action


def test_an_unpinned_sweep_run_must_have_run_at_the_studys_level(tmp_path):
    from steerlab_server.experiment import promote
    root = str(tmp_path)
    sweep = _promotable_at(root, "medium")
    run = sweep.SELECTION_BLOCK["sweepRun"]
    snapshot = Path(root) / "runs" / run / "experiment.json"
    snapshot.unlink()
    with pytest.raises(promote.PromoteError, match="thinking level its cells ran at is unknown"):
        promote.promote("pr", "fear", root=root, log=lambda *_: None, sweep_run=run)
    snapshot.write_text(json.dumps({"reasoningEffort": "off"}))
    with pytest.raises(promote.PromoteError, match="ran its cells at reasoning effort off"):
        promote.promote("pr", "fear", root=root, log=lambda *_: None, sweep_run=run)
    snapshot.write_text(json.dumps({"reasoningEffort": "medium", "reasoningMaxTokens": 64}))
    minted = promote.promote("pr", "fear", root=root, log=lambda *_: None, sweep_run=run)
    assert minted["variant"]["reasoningEffort"] == "medium"
