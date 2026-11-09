"""Panels and agents, wired together (FR6 integration: public issues #15,
#16, and #17).

* A panel seat cast with an agent thinks at the AGENT's level, which reaches
  the renderer by the rule an ordinary study's agent arm follows, and every
  turn stamps the level the template actually rendered.
* One budget predicate: a study whose own effort is off may declare
  ``reasoningMaxTokens`` when an agent arm thinks OR a cast panel seat
  thinks, read from the study's own workspace.
* The agents a panel's seats are cast with meet the agent-file rules (less
  the answer-token and revision rules) at ``panel check``, at verify (a
  violation on a draft, a ``refusedWhenRun`` note on a frozen study), at
  submission, and at run start.

Python twin of ``PanelAgentWiringTests``. Uses a stub generate(): no model.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import (agent_identity, experiment_store as store,
                                        lifecycle_gates, model_capabilities as mc,
                                        model_variant, multi_agent, prompt_render,
                                        study_admission)
from steerlab_server.experiment.manifest import Manifest

#: A model whose chat template has a thinking switch (by the id heuristic).
QWEN = "test/qwen-panel"
#: One whose template has none.
PLAIN = "org/tiny"
REVISION = "a" * 40
AGENT_PATH = "runs/model-variants/seat-agent/model-variant.json"


def _write_agent(root, fields, *, model=QWEN):
    document = {"name": "seat-agent", "baseModelID": model, **fields}
    path = Path(root) / AGENT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scenario(*, model=QWEN, agent_hash=None, cast=True):
    ava = multi_agent.Agent(id="a", name="Ava", base_model_id=model,
                            variant_artifact_path=AGENT_PATH if cast else None,
                            variant_artifact_hash=agent_hash if cast else None)
    ben = multi_agent.Agent(id="b", name="Ben", base_model_id=model)
    return multi_agent.Scenario(
        name="p", base_model_id=model, agents=[ava, ben],
        turns=[multi_agent.Turn(id="t1", title="Ava speaks", speaker_agent_id="a",
                                prompt_template="Speak.", output_label="ava1",
                                routing="all"),
               multi_agent.Turn(id="t2", title="Ben replies", speaker_agent_id="b",
                                prompt_template="Ava said {{outputs.ava1}}",
                                output_label="ben1", routing="all")])


def panel_study(root, agent_fields=None, *, model=QWEN, status="draft", cast=True,
                **settings):
    """A panel study in ``root`` whose seat Ava is cast with an agent file
    carrying ``agent_fields`` (Ben is a bare seat)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    store.create("study", model_id=model, revision=REVISION, root=str(root))
    agent_hash = _write_agent(root, agent_fields or {}, model=model) if cast else None
    panel = root / "prompts" / "panels" / "p.json"
    panel.parent.mkdir(parents=True, exist_ok=True)
    panel.write_text(json.dumps(multi_agent._scenario_to_dict(
        _scenario(model=model, agent_hash=agent_hash, cast=cast))))
    raw = store.load_raw("study", str(root))
    raw.update(studyKind="multiAgent", studyType="multiAgent",
               multiAgentScenarioPath="prompts/panels/p.json",
               multiAgentScenarioHash=hashlib.sha256(panel.read_bytes()).hexdigest(),
               multiAgentSemanticScenarioPath="prompts/panels/semantic/p.json",
               status=status, **settings)
    store.save_raw(raw, str(root))
    return Manifest.load("study", str(root))


def _run(tmp_path, scenario, outputs, **kwargs):
    """Run ``scenario`` with a generate stub; return (generate kwargs, records)."""
    calls = []

    def stub(model, prompt, **kw):
        calls.append(kw)
        return outputs[len(calls) - 1]

    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    original = multi_agent.generate
    multi_agent.generate = stub
    try:
        multi_agent.run_scenario(SimpleNamespace(model_id=QWEN, revision="r"), scenario,
                                 run_dir=str(run_dir), **kwargs)
    finally:
        multi_agent.generate = original
    records = [json.loads(line) for line in
               (run_dir / "turns.jsonl").read_text().splitlines()]
    return calls, records


def _seat_scenario(tmp_path, fields):
    """An ad-hoc scenario whose seat Ava is cast with an agent file (absolute
    path), as ``run_scenario`` reads it."""
    artifact = model_variant.ModelVariant(name="seat-agent", base_model_id=QWEN,
                                          **fields)
    path = tmp_path / "seat-agent.json"
    path.write_text(json.dumps(artifact.to_dict()))
    scenario = _scenario()
    scenario.agents[0].variant_artifact_path = str(path)
    scenario.agents[0].variant_artifact_hash = None
    return scenario


ANSWERED = "weighing it</think>My answer."


# --- 1. the seat's level -------------------------------------------------------


def test_a_seat_agent_thinks_at_its_own_level_without_a_budget(tmp_path):
    """A level the legacy switch cannot say reaches the renderer alone, as an
    ordinary agent arm's does; the turn stamps it."""
    scenario = _seat_scenario(tmp_path, {"reasoning_effort": "medium"})
    calls, records = _run(tmp_path, scenario, [ANSWERED, "ok"], temperature=0.0)
    assert calls[0]["qwen_thinking_enabled"] is True
    assert calls[0]["reasoning_effort"] == "medium"
    assert "reasoning_max_tokens" not in calls[0]
    assert (records[0]["reasoningEffort"], records[0]["reasoningMaxTokens"]) == (
        "medium", None)
    assert records[0]["output"] == "My answer."
    # The bare seat (study effort off) neither thinks nor names a level.
    assert calls[1]["qwen_thinking_enabled"] is False
    assert "reasoning_effort" not in calls[1]
    assert records[1]["reasoningEffort"] == "off"


def test_a_seat_agent_thinks_at_its_own_level_under_the_studys_budget(tmp_path):
    scenario = _seat_scenario(tmp_path, {"reasoning_effort": "low"})
    calls, records = _run(tmp_path, scenario, [ANSWERED, "ok"], temperature=0.0,
                          reasoning_effort="off", reasoning_max_tokens=300)
    assert (calls[0]["reasoning_effort"], calls[0]["reasoning_max_tokens"]) == ("low", 300)
    assert (records[0]["reasoningEffort"], records[0]["reasoningMaxTokens"]) == ("low", 300)
    # The budget never reaches the seat that does not think.
    assert "reasoning_max_tokens" not in calls[1]
    assert records[1]["reasoningMaxTokens"] is None


def test_a_legacy_thinking_seat_keeps_the_call_it_always_had(tmp_path):
    """An agent with only the thinking switch renders at the template's default,
    with no level handed over, exactly as before."""
    scenario = _seat_scenario(tmp_path, {"qwen_thinking_enabled": True})
    calls, records = _run(tmp_path, scenario, [ANSWERED, "ok"], temperature=0.0)
    assert calls[0]["qwen_thinking_enabled"] is True
    assert "reasoning_effort" not in calls[0]
    assert records[0]["reasoningEffort"] == "xhigh"


def test_a_turn_stamps_the_level_the_template_rendered(tmp_path, monkeypatch):
    """A level the template ignores renders as plain ``on``: the generation
    is handed the agent's level, and the record says what reached the
    template."""
    record = mc.heuristic(QWEN, None)
    record["source"] = "probe"
    record["detected"]["effortLevels"] = {"low": "ignored", "medium": "ignored",
                                          "high": "ignored", "xhigh": "ignored"}
    record["detected"]["effortVariableRead"] = False
    record["advisories"] = []
    ignoring = mc.effective(record)
    monkeypatch.setattr(prompt_render, "capabilities_for",
                        lambda model_id, tokenizer=None, capabilities=None, revision=None:
                        capabilities if capabilities is not None else ignoring)
    scenario = _seat_scenario(tmp_path, {"reasoning_effort": "medium"})
    calls, records = _run(tmp_path, scenario, [ANSWERED, "ok"], temperature=0.0)
    assert calls[0]["reasoning_effort"] == "medium"
    assert records[0]["reasoningEffort"] == "on"


def test_the_older_engine_check_reads_the_agents_level(tmp_path):
    """An agent whose file declares off does not think, so an older engine
    plays the seat the same way; one that declares a level does."""
    off = _seat_scenario(tmp_path, {"reasoning_effort": "off"})
    assert multi_agent.panel_rule_changes(off, study_effort="off",
                                          study_temperature=0.0) == []
    (tmp_path / "second").mkdir()
    medium = _seat_scenario(tmp_path / "second", {"reasoning_effort": "medium"})
    assert multi_agent.panel_rule_changes(medium, study_effort="off",
                                          study_temperature=0.0) == [
        ("reasoningPassedOn", "Ava")]


# --- 2. one budget predicate ------------------------------------------------------


def test_an_off_panel_study_may_declare_a_budget_only_when_a_cast_seat_thinks(tmp_path):
    thinking = panel_study(tmp_path / "thinking", {"reasoningEffort": "medium"},
                           reasoningEffort="off", reasoningMaxTokens=256)
    assert prompt_render.BUDGET_WITHOUT_EFFORT_REASON not in thinking.verify(
        str(tmp_path / "thinking"))
    assert agent_identity.budget_permitted_without_effort(
        thinking, str(tmp_path / "thinking")) is True
    plain = panel_study(tmp_path / "plain", {"reasoningEffort": "off"},
                        reasoningEffort="off", reasoningMaxTokens=256)
    assert prompt_render.BUDGET_WITHOUT_EFFORT_REASON in plain.verify(
        str(tmp_path / "plain"))
    bare = panel_study(tmp_path / "bare", cast=False, reasoningEffort="off",
                       reasoningMaxTokens=256)
    assert prompt_render.BUDGET_WITHOUT_EFFORT_REASON in bare.verify(str(tmp_path / "bare"))


def test_set_protocol_reads_the_panel_from_the_drafts_own_workspace(tmp_path, monkeypatch):
    """The panel clause reads the panel and its seats' agent files from the
    workspace the draft is in, not from wherever the process points."""
    workspace = tmp_path / "workspace"
    panel_study(workspace, {"reasoningEffort": "medium"}, reasoningEffort="off")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("STEERLAB_ROOT", str(elsewhere))
    store.set_protocol("study", {"reasoningMaxTokens": 128}, root=str(workspace))
    assert store.load_raw("study", str(workspace))["reasoningMaxTokens"] == 128
    # A panel none of whose seats thinks is still refused one.
    plain = tmp_path / "plain"
    panel_study(plain, {"reasoningEffort": "off"}, reasoningEffort="off")
    with pytest.raises(store.ExperimentStoreError, match="reasoningMaxTokens"):
        store.set_protocol("study", {"reasoningMaxTokens": 128}, root=str(plain))


# --- 3. the agent rules for the agents seats are cast with --------------------------


@pytest.mark.parametrize("fields, model, phrase", [
    ({"reasoningEffort": "medium"}, PLAIN, "has no thinking switch"),
    ({"reasoningEffort": "medium", "qwenThinkingEnabled": False}, QWEN,
     "one says the agent thinks"),
    ({"adapters": [{"adapterDirectory": "a"}, {"adapterDirectory": "b"}]}, QWEN,
     "lists 2 adapters"),
])
def test_a_draft_panel_verify_names_each_seat_agent_rule(tmp_path, fields, model, phrase):
    manifest = panel_study(tmp_path, fields, model=model)
    violations = manifest.verify(str(tmp_path))
    assert any(phrase in v and v.startswith("seat 'Ava': ") for v in violations), violations


def test_a_seat_agent_is_not_held_to_the_revision_or_answer_token_rules(tmp_path):
    """A panel loads each seat at the seat's own revision, and reads its
    generated turns, so neither rule applies to a cast agent."""
    manifest = panel_study(tmp_path, {"baseRevision": "b" * 40,
                                      "qwenThinkingEnabled": True},
                           outcomeInstruments=["answerTokenLogprob"])
    assert agent_identity.manifest_arm_findings(manifest, str(tmp_path)) == []


def test_a_frozen_panel_study_is_told_at_verify_and_refused_at_submission_and_run_start(
        tmp_path):
    from steerlab_server.api import submissions
    from steerlab_server.client import runner as runner_api
    root = str(tmp_path)
    manifest = panel_study(tmp_path, {"reasoningEffort": "medium"}, model=PLAIN,
                           status="frozen")
    assert not any(v.startswith("seat 'Ava'") for v in manifest.verify(root))
    note = agent_identity.frozen_arm_problems_advisory(manifest, root)
    assert note.startswith("this study will be refused when its run is submitted or started, "
                           "because seat 'Ava': ")
    assert "has no thinking switch" in note
    assert "experiment duplicate study study-v2" in note
    assert "panel compile prompts/panels/semantic/p.json --experiment study-v2" in note
    # Run start (settings_refusal is what run_workflow.run and _run_impl ask).
    refusal = study_admission.settings_refusal(manifest, root)
    assert refusal is not None and refusal.gate == lifecycle_gates.STUDY_DECLARATION
    with pytest.raises(lifecycle_gates.LifecycleError, match="seat 'Ava'"):
        study_admission.refuse_settings(manifest, root)
    # The engine's submission preflight, reading the study's own workspace.
    with pytest.raises(submissions.SubmissionRefusal, match="has no thinking switch"):
        submissions._refuse_settings(manifest, "run", ["run"], root)
    # The client's composite run and `runner submit` ask agent_arm_refusal.
    assert study_admission.agent_arm_refusal(manifest, root) is not None
    staged = {"experiment": "study", "experimentContentHash": manifest.content_hash()}
    with pytest.raises(runner_api.RunnerRefusal, match="seat 'Ava'"):
        runner_api.require_greedy_seed_list(staged, "run", root)
    runner_api.require_greedy_seed_list(staged, "analyze", root)


def test_a_file_that_is_not_the_pinned_panel_is_never_judged(tmp_path):
    """A copy at the pinned path whose bytes are not the pinned panel (a stale
    file in an engine's own root, say) never refuses a study for its seats;
    verify refuses the drift itself."""
    manifest = panel_study(tmp_path, {"reasoningEffort": "medium"}, model=PLAIN,
                           status="frozen")
    assert agent_identity.manifest_arm_findings(manifest, str(tmp_path))
    manifest.raw["multiAgentScenarioHash"] = "0" * 64
    assert agent_identity.manifest_arm_findings(manifest, str(tmp_path)) == []
    assert study_admission.settings_refusal(manifest, str(tmp_path)) is None


def test_a_panel_whose_seat_agents_meet_the_rules_is_not_refused(tmp_path):
    manifest = panel_study(tmp_path, {"reasoningEffort": "medium"}, status="frozen")
    assert agent_identity.frozen_arm_problems_advisory(manifest, str(tmp_path)) is None
    assert study_admission.settings_refusal(manifest, str(tmp_path)) is None


def test_panel_check_refuses_a_seat_whose_agent_breaks_a_rule(tmp_path, capsys, monkeypatch):
    from steerlab_server import cli
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    _write_agent(tmp_path, {"reasoningEffort": "medium"}, model=PLAIN)
    path = tmp_path / "prompts" / "panels" / "p.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(multi_agent._scenario_to_dict(_scenario(model=PLAIN))))
    assert cli._panel(["check", str(path)]) == 65
    err = capsys.readouterr().err
    assert "invalid: seat 'Ava': " in err and "has no thinking switch" in err
    assert f"repair: {agent_identity.SEAT_AGENT_REPAIR}" in err
    _write_agent(tmp_path, {"reasoningEffort": "off"}, model=PLAIN)
    assert cli._panel(["check", str(path)]) == 0


def test_a_seat_agent_file_with_a_problem_refuses_when_the_seat_plays(tmp_path):
    """An ad-hoc run (no study to refuse it first) reaches the seat's agent
    file, and its problem refuses there, never runs silently."""
    scenario = _seat_scenario(tmp_path, {})
    Path(scenario.agents[0].variant_artifact_path).write_text(json.dumps(
        {"name": "seat-agent", "baseModelID": QWEN, "reasoningEffort": "medium",
         "qwenThinkingEnabled": False}))
    with pytest.raises(ValueError, match="one says the agent thinks"):
        _run(tmp_path, scenario, [ANSWERED, "ok"], temperature=0.0)
