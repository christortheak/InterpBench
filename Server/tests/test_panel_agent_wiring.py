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
    with pytest.raises(multi_agent.ScenarioError) as refused:
        _run(tmp_path, scenario, [ANSWERED, "ok"], temperature=0.0)
    # The Mac engine's sentence (MultiAgentRunner.seatAgent): the seat by
    # name, then the file's own problem; typed, with the seat repair.
    assert str(refused.value).startswith(
        "seat 'Ava': This agent file says reasoningEffort \"medium\" but "
        "qwenThinkingEnabled false: one says the agent thinks")
    assert lifecycle_gates.gate_of(refused.value) == lifecycle_gates.STUDY_DECLARATION
    assert lifecycle_gates.repair_of(refused.value) == agent_identity.SEAT_AGENT_REPAIR


# --- FR6 integration, round 2 ------------------------------------------------------


def _write_probe(root, model, revision, switch):
    """A probed capability record, as a run's ``ensure_record`` leaves one."""
    record = mc.heuristic(model, revision)
    record["source"] = mc.SOURCE_PROBE
    record["detected"]["thinkingSwitch"] = switch
    record["recordHash"] = mc.record_hash(record)
    path = mc.record_path(model, revision, str(root))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(record))


def test_a_frozen_panel_study_keeps_verifying_after_its_own_run_probes_a_record(tmp_path):
    """A frozen study is read by the rules it froze under: a record its own
    first run probes cannot make its declared budget a violation. A draft
    still reads the probe."""
    frozen = panel_study(tmp_path / "frozen", {"reasoningEffort": "on"}, status="frozen",
                         reasoningEffort="off", reasoningMaxTokens=256)
    root = str(tmp_path / "frozen")
    assert prompt_render.BUDGET_WITHOUT_EFFORT_REASON not in frozen.verify(root)
    _write_probe(root, QWEN, REVISION, mc.UNSUPPORTED)
    assert prompt_render.BUDGET_WITHOUT_EFFORT_REASON not in frozen.verify(root)
    draft = panel_study(tmp_path / "draft", {"reasoningEffort": "on"},
                        reasoningEffort="off", reasoningMaxTokens=256)
    root = str(tmp_path / "draft")
    assert prompt_render.BUDGET_WITHOUT_EFFORT_REASON not in draft.verify(root)
    _write_probe(root, QWEN, REVISION, mc.UNSUPPORTED)
    assert prompt_render.BUDGET_WITHOUT_EFFORT_REASON in draft.verify(root)


def test_a_turn_played_as_thinking_never_records_off(tmp_path):
    """No probe said the seat cannot think, so it is split and budgeted; the
    renderer's record has no thinking switch, so no level reached one. The
    turn records ``on``, never ``off`` beside a budget and a complete split."""
    artifact = model_variant.ModelVariant(name="seat-agent", base_model_id="org/plain",
                                          qwen_thinking_enabled=True)
    path = tmp_path / "seat-agent.json"
    path.write_text(json.dumps(artifact.to_dict()))
    scenario = _scenario(model="org/plain")
    scenario.agents[0].variant_artifact_path = str(path)
    scenario.agents[0].variant_artifact_hash = None
    calls = []

    def stub(model, prompt, **kw):
        calls.append(kw)
        return "weighing it</think>My answer."

    (tmp_path / "run").mkdir()
    original = multi_agent.generate
    multi_agent.generate = stub
    try:
        multi_agent.run_scenario(SimpleNamespace(model_id="org/plain", revision="r"),
                                 scenario, run_dir=str(tmp_path / "run"), temperature=0.0,
                                 reasoning_effort="off", reasoning_max_tokens=256)
    finally:
        multi_agent.generate = original
    first = json.loads((tmp_path / "run" / "turns.jsonl").read_text().splitlines()[0])
    assert calls[0]["reasoning_max_tokens"] == 256
    assert (first["reasoningEffort"], first["reasoningMaxTokens"],
            first["reasoningStatus"]) == ("on", 256, "complete")


#: The make-again lead every seat repair opens with, on the Mac command line.
MAC_MAKE = ("make a new agent for each seat whose agent breaks a rule (seat ID a): "
            "steerlab-cli agent create <new-name> --model <model> --revision <sha> … "
            "(or the app's agent editor), then ")
#: What the Mac study editor does with a casting it recompiles.
MAC_SEATS = ("in the Mac app's study editor (Seats), put each new agent in its seat "
             "and save, which compiles the panel again")
#: The client's copy-and-pin route, for any panel it cannot recompile.
CLIENT_COPY = ("save a copy of the panel prompts/panels/p.json in which each of those "
               "seats' variantArtifactPath and variantArtifactHash name its new agent, "
               "and pin the copy: steerlab experiment set-protocol study --set "
               "multiAgentScenarioPath=<copy-path> --set "
               "multiAgentScenarioHash=<copy-sha256>")


def _repairs(tmp_path, *, semantic, pin="prompts/panels/p.json"):
    """The seat repair for a draft whose seat a's agent breaks a rule, on the
    Mac and on the client, with the panel's semantic path set to
    ``semantic`` (absent when None) and pinned at ``pin``."""
    from steerlab_server.experiment import command_vocabulary as vocabulary
    panel_study(tmp_path, {"reasoningEffort": "medium"}, model=PLAIN)
    raw = store.load_raw("study", str(tmp_path))
    if semantic is None:
        raw.pop("multiAgentSemanticScenarioPath", None)
    else:
        raw["multiAgentSemanticScenarioPath"] = semantic
    if pin != "prompts/panels/p.json":
        target = tmp_path / pin
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((tmp_path / "prompts" / "panels" / "p.json").read_bytes())
        raw["multiAgentScenarioPath"] = pin
    store.save_raw(raw, str(tmp_path))
    manifest = Manifest.load("study", str(tmp_path))
    findings = agent_identity.manifest_arm_findings(manifest, str(tmp_path))
    assert [seat for seat, _, _ in findings] == ["a"]
    out = {}
    for surface in (vocabulary.MAC, vocabulary.CLIENT):
        with vocabulary.speaking_as(surface):
            out[surface] = agent_identity.arm_problems_repair(
                manifest, findings, frozen=False, authoring=vocabulary.authoring,
                root=str(tmp_path))
    return manifest, out[vocabulary.MAC], out[vocabulary.CLIENT]


def test_a_hand_bound_panel_is_repaired_by_pinning_a_copy(tmp_path, capsys):
    """No semantic path and a pin outside prompts/panels/compiled/: the Mac
    study editor reads the panel as hand-bound and pins a new selection as it
    stands. The client's command is run here and clears the refusal."""
    from test_agent_create import cli
    _, mac, client = _repairs(tmp_path, semantic=None)
    assert mac == (MAC_MAKE + "save a copy of the panel prompts/panels/p.json in which "
                   "each of those seats' variantArtifactPath and variantArtifactHash "
                   "name its new agent, and pin the copy: in the Mac app's study editor "
                   "(Setup), which pins a hand-bound panel as it stands")
    assert client.endswith(CLIENT_COPY) and "panel compile" not in client
    fixed = tmp_path / "runs" / "model-variants" / "fixed" / "model-variant.json"
    fixed.parent.mkdir(parents=True)
    fixed.write_text(json.dumps({"name": "fixed", "baseModelID": PLAIN,
                                 "reasoningEffort": "off"}))
    panel = json.loads((tmp_path / "prompts" / "panels" / "p.json").read_text())
    panel["agents"][0]["variantArtifactPath"] = "runs/model-variants/fixed/model-variant.json"
    panel["agents"][0]["variantArtifactHash"] = hashlib.sha256(fixed.read_bytes()).hexdigest()
    copy = tmp_path / "prompts" / "panels" / "p-fixed.json"
    copy.write_text(json.dumps(panel))
    code, document = cli(tmp_path, capsys, "experiment", "set-protocol", "study",
                         "--set", "multiAgentScenarioPath=prompts/panels/p-fixed.json",
                         "--set", "multiAgentScenarioHash="
                         + hashlib.sha256(copy.read_bytes()).hexdigest())
    assert code == 0, document
    repaired = Manifest.load("study", str(tmp_path))
    assert repaired.raw["multiAgentScenarioPath"] == "prompts/panels/p-fixed.json"
    assert agent_identity.manifest_arm_findings(repaired, str(tmp_path)) == []


@pytest.mark.parametrize("semantic, pin", [
    ("", "prompts/panels/p.json"),
    (None, "prompts/panels/compiled/p.json"),
])
def test_a_casting_the_mac_editor_recompiles_is_repaired_in_its_seats(
        tmp_path, semantic, pin):
    """An empty semantic path, or a pin in prompts/panels/compiled/, is a
    casting the Mac study editor recompiles (it ignores a new selection
    there): the Mac repair is its Seats section. The client pins a copy."""
    _, mac, client = _repairs(tmp_path, semantic=semantic, pin=pin)
    assert mac == MAC_MAKE + MAC_SEATS
    assert "save a copy of the panel " + pin in client
    assert client.endswith(CLIENT_COPY.split("prompts/panels/p.json", 1)[1])


def test_the_seat_repair_for_a_compiled_panel_casts_by_seat_id(tmp_path):
    from steerlab_server.experiment import command_vocabulary as vocabulary
    manifest = panel_study(tmp_path, {"reasoningEffort": "medium"}, model=PLAIN)
    findings = agent_identity.manifest_arm_findings(manifest, str(tmp_path))
    with vocabulary.speaking_as(vocabulary.CLIENT):
        repair = agent_identity.arm_problems_repair(
            manifest, findings, frozen=False, authoring=vocabulary.authoring)
    assert "(seat ID a)" in repair and "Ava" not in repair
    assert ("steerlab panel compile prompts/panels/semantic/p.json --experiment study "
            "--casting <file>") in repair
    with vocabulary.speaking_as(vocabulary.MAC):
        mac = agent_identity.arm_problems_repair(
            manifest, findings, frozen=False, authoring=vocabulary.authoring)
    assert mac == (MAC_MAKE + "compile the panel again, casting every seat, with each "
                   "new agent in place of the one its seat holds: steerlab-cli panel "
                   "compile prompts/panels/semantic/p.json --experiment study --seat "
                   "a=<agent-path> --seat <each other seat>=<its agent-path>")


#: The Mac text both engines print for seat IDs a shell or the Mac parser
#: would misread (PanelAgentWiringTests asserts the same strings).
QUOTED_MAC = ("make a new agent for each seat whose agent breaks a rule (seat IDs "
              "judge one, it's): steerlab-cli agent create <new-name> --model <model> "
              "--revision <sha> … (or the app's agent editor), then compile the panel "
              "again, casting every seat, with each new agent in place of the one its "
              "seat holds: steerlab-cli panel compile 'prompts/panels/my panel.json' "
              "--experiment study --seat 'judge one'=<agent-path> --seat "
              "'it'\"'\"'s'=<agent-path> --seat <each other seat>=<its agent-path>")
EQUALS_MAC = ("make a new agent for each seat whose agent breaks a rule (seat ID "
              "a=b): steerlab-cli agent create <new-name> --model <model> --revision "
              "<sha> … (or the app's agent editor), then in the Mac app's study editor "
              "(Seats), put each new agent in its seat and save, which compiles the "
              "panel again (a seat ID with '=' cannot be named on the Mac command line)")


def test_printed_seat_repairs_quote_their_values(tmp_path):
    from steerlab_server.experiment import command_vocabulary as vocabulary
    manifest = panel_study(tmp_path, {}, model=PLAIN)
    manifest.raw["multiAgentSemanticScenarioPath"] = "prompts/panels/my panel.json"
    with vocabulary.speaking_as(vocabulary.MAC):
        quoted = agent_identity.seat_repair(manifest, ["judge one", "it's"], target="study",
                                            authoring=vocabulary.authoring)
        equals = agent_identity.seat_repair(manifest, ["a=b"], target="study",
                                            authoring=vocabulary.authoring)
    assert quoted == QUOTED_MAC
    assert equals == EQUALS_MAC
    with vocabulary.speaking_as(vocabulary.CLIENT):
        client = agent_identity.seat_repair(manifest, ["a=b"], target="study",
                                            authoring=vocabulary.authoring)
    assert "steerlab panel compile 'prompts/panels/my panel.json' --experiment study" in client


def test_a_seat_agent_edited_in_place_is_judged_as_it_is_on_disk(tmp_path):
    """The run loads the agent file as it is now (a drifted seat agent is a
    warning there, never a refusal), so the seat checks judge that file,
    whatever hash the seat pins: a sound agent saved in place with a level
    its template does not take is named by verify and refused at run start."""
    sound = panel_study(tmp_path, {"reasoningEffort": "off"}, model=PLAIN)
    root = str(tmp_path)
    assert agent_identity.manifest_arm_findings(sound, root) == []
    _write_agent(tmp_path, {"reasoningEffort": "medium"}, model=PLAIN)
    edited = Manifest.load("study", root)
    assert [s for s, _, _ in agent_identity.manifest_arm_findings(edited, root)] == ["a"]
    assert any(v.startswith("seat 'Ava': ") for v in edited.verify(root))
    assert study_admission.settings_refusal(edited, root) is not None


def test_a_frozen_study_keeps_its_budget_after_a_seat_agent_is_edited(tmp_path):
    """The study froze with its budget allowed (seat a thought). An agent
    saved in place since, even to off, must never make it stop verifying,
    which would block analyze and evaluate of its runs."""
    frozen = panel_study(tmp_path / "frozen", {"reasoningEffort": "on"}, status="frozen",
                         reasoningEffort="off", reasoningMaxTokens=256)
    root = str(tmp_path / "frozen")
    assert prompt_render.BUDGET_WITHOUT_EFFORT_REASON not in frozen.verify(root)
    _write_agent(root, {"reasoningEffort": "off"})
    after = Manifest.load("study", root)
    assert prompt_render.BUDGET_WITHOUT_EFFORT_REASON not in after.verify(root)
    # A draft reads the file as it is now: nothing in it thinks any more.
    draft = panel_study(tmp_path / "draft", {"reasoningEffort": "on"},
                        reasoningEffort="off", reasoningMaxTokens=256)
    root = str(tmp_path / "draft")
    assert prompt_render.BUDGET_WITHOUT_EFFORT_REASON not in draft.verify(root)
    _write_agent(root, {"reasoningEffort": "off"})
    assert prompt_render.BUDGET_WITHOUT_EFFORT_REASON in Manifest.load(
        "study", root).verify(root)


def _bundle(source, path):
    """A run bundle holding ``source``'s study files, as a staged upload."""
    import tarfile
    with tarfile.open(path, "w:gz") as tar:
        for folder in ("experiments", "prompts", "runs"):
            tar.add(source / folder, arcname=folder)
    return str(path)


def _submit_bundle(tmp_path, monkeypatch, bundle):
    """``submit_run_bundle`` of ``bundle`` for a run, up to the step after the
    settings preflight (a sentinel stands in for the rest)."""
    from steerlab_server.api import submissions

    class PastSettings(Exception):
        pass

    def past(*args, **kwargs):
        raise PastSettings

    monkeypatch.setattr(submissions.bundles, "inspect_bundle",
                        lambda path: {"kind": "runBundle", "experiment": "study"})
    monkeypatch.setattr(submissions, "_refuse_inert_conditions", past)
    try:
        submissions.submit_run_bundle(bundle, verb="run", jobs=None, executor="local",
                                      dry_run=True)
    except PastSettings:
        return None
    raise AssertionError("submission went past the sentinel")


def test_a_bundle_is_judged_by_its_own_seat_agents_never_the_engines(tmp_path, monkeypatch):
    """The engine's root holds a stale copy of the seat's agent that breaks a
    rule; the bundle carries a sound one. The bundle is judged by its own
    files, so it is not refused. A bundle whose own agent breaks the rule is
    refused here, before the queue."""
    from steerlab_server.api import submissions
    sound = tmp_path / "sound"
    panel_study(sound, {"reasoningEffort": "off"}, model=PLAIN, status="frozen")
    engine = tmp_path / "engine"
    panel_study(engine, {"reasoningEffort": "medium"}, model=PLAIN, status="frozen")
    # The engine's copy of the panel is the bundle's, byte for byte.
    (engine / "prompts" / "panels" / "p.json").write_bytes(
        (sound / "prompts" / "panels" / "p.json").read_bytes())
    monkeypatch.setenv("STEERLAB_ROOT", str(engine))
    assert _submit_bundle(tmp_path, monkeypatch,
                          _bundle(sound, tmp_path / "sound.tar.gz")) is None
    broken = tmp_path / "broken"
    panel_study(broken, {"reasoningEffort": "medium"}, model=PLAIN, status="frozen")
    with pytest.raises(submissions.SubmissionRefusal, match="seat 'Ava': "):
        _submit_bundle(tmp_path, monkeypatch, _bundle(broken, tmp_path / "broken.tar.gz"))


def test_one_set_protocol_call_that_pins_a_panel_and_its_budget_is_judged_by_that_panel(
        tmp_path):
    """The budget rule reads the document the call writes, not the panel the
    draft pinned before it."""
    panel_study(tmp_path, {"reasoningEffort": "medium"}, reasoningEffort="off")
    thinking = tmp_path / "prompts" / "panels" / "p.json"
    bare = tmp_path / "prompts" / "panels" / "bare.json"
    bare.write_text(json.dumps(multi_agent._scenario_to_dict(_scenario(cast=False))))
    raw = store.load_raw("study", str(tmp_path))
    raw.update(multiAgentScenarioPath="prompts/panels/bare.json",
               multiAgentScenarioHash=hashlib.sha256(bare.read_bytes()).hexdigest())
    store.save_raw(raw, str(tmp_path))
    store.set_protocol("study", {
        "multiAgentScenarioPath": "prompts/panels/p.json",
        "multiAgentScenarioHash": hashlib.sha256(thinking.read_bytes()).hexdigest(),
        "reasoningMaxTokens": 128}, root=str(tmp_path))
    assert store.load_raw("study", str(tmp_path))["reasoningMaxTokens"] == 128
    with pytest.raises(store.ExperimentStoreError, match="reasoningMaxTokens"):
        store.set_protocol("study", {
            "multiAgentScenarioPath": "prompts/panels/bare.json",
            "multiAgentScenarioHash": hashlib.sha256(bare.read_bytes()).hexdigest(),
            "reasoningMaxTokens": 64}, root=str(tmp_path))


# --- Round 3b: the admission extraction follows the real unpack's limits ------------


def _extracted(bundle):
    """The files the admission helper extracts from ``bundle``, relative."""
    from steerlab_server.api import submissions
    with submissions._bundle_json_members(bundle) as folder:
        base = Path(folder)
        return sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file())


@pytest.mark.parametrize("variable", ["STEERLAB_MAX_BUNDLE_MEMBERS",
                                      "STEERLAB_MAX_BUNDLE_TOTAL_BYTES"])
def test_an_archive_over_the_unpacks_limits_extracts_nothing(tmp_path, monkeypatch,
                                                              variable):
    """More members, or more declared bytes, than the real unpack admits: the
    admission writes nothing, so the bundle is judged at run start, where
    the unpack refuses it."""
    source = tmp_path / "source"
    panel_study(source, {"reasoningEffort": "medium"}, model=PLAIN, status="frozen")
    bundle = _bundle(source, tmp_path / "b.tar.gz")
    assert AGENT_PATH in _extracted(bundle)
    monkeypatch.setenv(variable, "2")
    assert _extracted(bundle) == []
    # Nothing to judge here, so the bundle is not refused at admission.
    (tmp_path / "engine").mkdir()
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "engine"))
    assert _submit_bundle(tmp_path, monkeypatch, bundle) is None


def test_a_json_member_over_the_metadata_limit_is_left_out(tmp_path, monkeypatch):
    """A JSON member larger than the unpack's metadata limit is never written;
    the others are still extracted and judged."""
    from steerlab_server.api import submissions
    source = tmp_path / "source"
    panel_study(source, {"reasoningEffort": "medium"}, model=PLAIN, status="frozen")
    (source / "prompts" / "padding.json").write_text(json.dumps({"pad": "x" * 200_000}))
    bundle = _bundle(source, tmp_path / "b.tar.gz")
    monkeypatch.setenv("STEERLAB_MAX_BUNDLE_METADATA_BYTES", "100000")
    extracted = _extracted(bundle)
    assert "prompts/padding.json" not in extracted
    assert AGENT_PATH in extracted and "prompts/panels/p.json" in extracted
    (tmp_path / "engine").mkdir()
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path / "engine"))
    with pytest.raises(submissions.SubmissionRefusal, match="seat 'Ava': "):
        _submit_bundle(tmp_path, monkeypatch, bundle)
