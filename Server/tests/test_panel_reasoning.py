"""What a panel seat passes on (public issue #15).

A thinking seat's turn is split into its reasoning and its answer. Other seats,
and templates, get the answer alone; the seat's own earlier reasoning comes
back to it only when the panel declares ``includeOwnReasoning``; endpoints are
parsed from the answer; a thinking seat gets the study's reasoning budget; a
bare seat follows the study's effort; and a seat may declare its own
temperature in the panel's design. Uses a stub generate() — no model.
"""

import json
import os
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import (model_variant, multi_agent,
                                        reasoning_split, turn_endpoint)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(REPO, "Tests", "Fixtures", "cross-engine",
                       "reasoning-split.json")

# Ava reasons her way past one answer to another. Written by hand: the
# reasoning tries "Final answer: affirm" out, and the answer gives "reverse".
AVA_TEXT = ("I first lean one way. Final answer: affirm? No — the record "
            "shows the error was not harmless.\n</think>\n\n"
            "Final answer: reverse. The error was not harmless.")
AVA_REASONING = ("I first lean one way. Final answer: affirm? No — the record "
                 "shows the error was not harmless.\n")
AVA_ANSWER = "Final answer: reverse. The error was not harmless."


def _thinking_agent(tmp_path, name="thinker"):
    artifact = model_variant.ModelVariant(
        name=name, base_model_id="m", qwen_thinking_enabled=True)
    path = os.path.join(str(tmp_path), f"{name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact.to_dict(), handle)
    return path


def _panel(tmp_path, *, include_own_reasoning=False, endpoint=True):
    ava = multi_agent.Agent(id="a", name="Ava", base_model_id="m",
                            variant_artifact_path=_thinking_agent(tmp_path))
    ben = multi_agent.Agent(id="b", name="Ben", base_model_id="m")
    vote = (turn_endpoint.TurnEndpoint.from_dict(
        {"name": "vote", "kind": "choice", "marker": "Final answer:",
         "vocabulary": ["affirm", "reverse"]}) if endpoint else None)
    return multi_agent.Scenario(
        name="p", base_model_id="m", include_own_reasoning=include_own_reasoning,
        agents=[ava, ben],
        turns=[
            multi_agent.Turn(id="t1", title="Ava votes", speaker_agent_id="a",
                             prompt_template="Vote.", output_label="ava1",
                             routing="all", endpoint=vote, max_tokens=64),
            multi_agent.Turn(id="t2", title="Ben reads",
                             speaker_agent_id="b",
                             prompt_template="Ava said: {{outputs.ava1}}",
                             output_label="ben1", routing="all"),
            multi_agent.Turn(id="t3", title="Ava again", speaker_agent_id="a",
                             prompt_template="Again.", output_label="ava2",
                             routing="all"),
        ])


def _run(tmp_path, scenario, outputs, **kwargs):
    """Run ``scenario`` with a generate stub returning ``outputs`` in order;
    return (prompts, generate kwargs, turn records)."""
    prompts, calls = [], []

    def stub(model, prompt, **kw):
        prompts.append(prompt)
        calls.append(kw)
        ids = kw.get("token_ids_out")
        text, token_ids = outputs[len(prompts) - 1]
        if ids is not None and token_ids:
            ids.extend(token_ids)
        return text

    run_dir = os.path.join(str(tmp_path), "run")
    os.makedirs(run_dir, exist_ok=True)
    original = multi_agent.generate
    multi_agent.generate = stub
    try:
        multi_agent.run_scenario(kwargs.pop("model", SimpleNamespace(
            model_id="m", revision="r")), scenario, run_dir=run_dir, **kwargs)
    finally:
        multi_agent.generate = original
    with open(os.path.join(run_dir, "turns.jsonl"), encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    return prompts, calls, records


# --- 1. the issue's own test --------------------------------------------------

@pytest.mark.parametrize("include_own_reasoning", [False, True])
def test_a_colleague_reads_only_the_answer_and_the_speaker_follows_the_setting(
        tmp_path, include_own_reasoning):
    scenario = _panel(tmp_path, include_own_reasoning=include_own_reasoning)
    prompts, _calls, records = _run(
        tmp_path, scenario,
        [(AVA_TEXT, []), ("Noted.", []), ("</think>Still reverse.", [])])

    # Ben's prompt: the answer, both as the template's {{outputs.ava1}} and in
    # his routed transcript, and nothing of the reasoning.
    assert AVA_ANSWER in prompts[1]
    assert "affirm" not in prompts[1]
    assert "I first lean" not in prompts[1]
    # Ava's next prompt follows the panel's setting.
    if include_own_reasoning:
        assert multi_agent.own_reasoning_text(AVA_REASONING, AVA_ANSWER) in prompts[2]
        assert "I first lean one way" in prompts[2]
    else:
        assert "I first lean" not in prompts[2]
        assert AVA_ANSWER in prompts[2]
    # Ben's own routed entry never carries Ava's reasoning, either way.
    assert "Your reasoning at the time" not in prompts[1]

    first = records[0]
    assert first["output"] == AVA_ANSWER
    assert first["reasoning"] == AVA_REASONING
    assert first["reasoningStatus"] == "complete"
    # The endpoint is the answer's, and the record says the reasoning tried
    # one out too.
    assert first["endpoint"] == {"name": "vote", "value": "reverse",
                                 "markerInReasoning": True}
    # Ben did not think: his record is unsplit and carries no reasoning keys.
    assert "reasoning" not in records[1] and "reasoningStatus" not in records[1]
    assert records[1]["output"] == "Noted."


def test_a_seat_that_does_not_think_routes_its_text_unchanged(tmp_path):
    """Without a thinking seat nothing is split, so a panel that never thought
    routes the bytes it always did — even text that looks like a tag."""
    scenario = _panel(tmp_path, endpoint=False)
    scenario.agents[0].variant_artifact_path = None
    text = "<think>not a real block</think> but my words"
    prompts, _calls, records = _run(
        tmp_path, scenario, [(text, []), ("ok", []), ("ok", [])])
    assert text in prompts[1]
    assert records[0]["output"] == text
    assert "reasoningStatus" not in records[0]
    assert records[0]["reasoningEffort"] == "off"
    assert records[0]["reasoningMaxTokens"] is None


# --- 2. the reasoning budget ---------------------------------------------------

def test_a_thinking_seat_gets_the_studys_reasoning_budget(tmp_path):
    scenario = _panel(tmp_path, endpoint=False)
    _prompts, calls, records = _run(
        tmp_path, scenario, [(AVA_TEXT, []), ("ok", []), ("x</think>y", [])],
        temperature=0.0, reasoning_effort="off", reasoning_max_tokens=300)
    # Ava (an agent with thinking on) thinks at its own effort, under the
    # study's budget; Ben (bare, study effort off) does not think at all.
    assert calls[0]["qwen_thinking_enabled"] is True
    assert calls[0]["reasoning_max_tokens"] == 300
    assert calls[0]["reasoning_effort"] == "xhigh"
    assert calls[1]["qwen_thinking_enabled"] is False
    assert "reasoning_max_tokens" not in calls[1]
    assert (records[0]["reasoningEffort"], records[0]["reasoningMaxTokens"]) == (
        "xhigh", 300)
    assert (records[1]["reasoningEffort"], records[1]["reasoningMaxTokens"]) == (
        "off", None)


# --- 3. reasoning that never closed ---------------------------------------------

class _Tokenizer:
    """Names ``</think>`` as id 7, so a budgeted decode can be classified."""

    eos_token_id = 0

    @staticmethod
    def convert_tokens_to_ids(token):
        return 7 if token == "</think>" else None


@pytest.mark.parametrize("text,token_ids,budget", [
    # The block opened with a tag and never closed.
    ("<think>weighing affirm against reverse, Final answer: affirm", [], None),
    # Prompt-opened thinking that spent its whole reasoning budget: no tag.
    ("weighing affirm against reverse, Final answer: affirm", [5] * 4, 4),
    # Prompt-opened thinking cut off by the turn's own token limit.
    ("weighing affirm against reverse, Final answer: affirm", [5] * 64, None),
])
def test_reasoning_that_never_closed_passes_nothing_on(tmp_path, text,
                                                       token_ids, budget):
    scenario = _panel(tmp_path)
    model = SimpleNamespace(model_id="m", revision="r", tokenizer=_Tokenizer())
    prompts, _calls, records = _run(
        tmp_path, scenario, [(text, token_ids), ("ok", []), ("ok", [])],
        model=model, reasoning_max_tokens=budget)
    assert "affirm" not in prompts[1]
    assert "weighing" not in prompts[1]
    assert "weighing" not in prompts[2]
    first = records[0]
    assert first["reasoningStatus"] == "cutOff"
    assert first["output"] == ""
    assert first["routedAgentIDs"] == []
    assert "weighing" in first["reasoning"]
    assert first["endpoint"]["unparsed"] is True
    # Ben's template read the label as empty, never as a verbatim placeholder.
    assert "{{outputs.ava1}}" not in prompts[1]


def test_a_thinking_seat_with_no_reasoning_block_that_finished_is_its_answer(
        tmp_path):
    scenario = _panel(tmp_path, endpoint=False)
    prompts, _calls, records = _run(
        tmp_path, scenario, [("Plain answer.", []), ("ok", []), ("ok", [])])
    assert "Plain answer." in prompts[1]
    assert records[0]["reasoningStatus"] == "notFound"
    assert records[0]["reasoning"] is None


# --- 4. a bare seat follows the study's effort ---------------------------------

def test_a_bare_seat_thinks_at_the_studys_effort(tmp_path):
    scenario = _panel(tmp_path, endpoint=False)
    scenario.agents[0].variant_artifact_path = None
    _prompts, calls, records = _run(
        tmp_path, scenario, [(AVA_TEXT, []), ("ok", []), ("x</think>y", [])],
        temperature=0.0, reasoning_effort="on", reasoning_max_tokens=200)
    assert all(call["qwen_thinking_enabled"] is True for call in calls)
    assert all(call["reasoning_effort"] == "on" for call in calls)
    assert records[0]["reasoningEffort"] == "on"
    assert records[0]["output"] == AVA_ANSWER


# --- 5. per-seat temperature ---------------------------------------------------

def test_a_seat_runs_at_its_declared_temperature_and_says_where_it_came_from(
        tmp_path):
    scenario = _panel(tmp_path, endpoint=False)
    scenario.agents[1].temperature = 0.8
    _prompts, calls, records = _run(
        tmp_path, scenario, [(AVA_TEXT, []), ("ok", []), ("x</think>y", [])],
        temperature=0.0, experiment_hash="eh")
    assert [c["temperature"] for c in calls] == [0.0, 0.8, 0.0]
    assert [r["temperatureSource"] for r in records] == [
        "study", "panelSeat", "study"]
    # A greedy turn draws no seed; the warm seat's turn does.
    assert records[0]["seed"] is None and records[2]["seed"] is None
    assert isinstance(records[1]["seed"], int)
    report = json.load(open(os.path.join(str(tmp_path), "run", "report.json")))
    assert report["seedPolicy"] == "derivedSHA256"


def test_an_ad_hoc_run_says_the_panel_file_set_the_temperature(tmp_path):
    scenario = _panel(tmp_path, endpoint=False)
    _prompts, _calls, records = _run(
        tmp_path, scenario, [(AVA_TEXT, []), ("ok", []), ("x</think>y", [])])
    assert {r["temperatureSource"] for r in records} == {"panel"}


def test_a_seat_temperature_round_trips_and_is_validated():
    document = {"name": "p", "agents": [
        {"id": "a", "name": "Ava", "baseModelID": "m", "systemPrompt": "",
         "temperature": 0.6}], "turns": [], "includeOwnReasoning": True}
    scenario = multi_agent.Scenario.from_dict(document)
    assert scenario.agents[0].temperature == 0.6
    assert scenario.include_own_reasoning is True
    written = multi_agent._scenario_to_dict(scenario)
    assert written["agents"][0]["temperature"] == 0.6
    assert written["includeOwnReasoning"] is True
    # A seat with none, and a panel with the setting off, write no key: every
    # panel from before these existed keeps its pinned bytes.
    plain = multi_agent._scenario_to_dict(multi_agent.Scenario.from_dict(
        {"name": "p", "agents": [{"id": "a", "name": "Ava", "baseModelID": "m",
                                  "systemPrompt": ""}], "turns": []}))
    assert "temperature" not in plain["agents"][0]
    assert "includeOwnReasoning" not in plain

    with pytest.raises(multi_agent.ScenarioError, match="must be a number"):
        multi_agent.Scenario.from_dict({**document, "agents": [
            {**document["agents"][0], "temperature": "0.6"}]})
    with pytest.raises(multi_agent.ScenarioError, match="true or false"):
        multi_agent.Scenario.from_dict({**document, "includeOwnReasoning": 1})


def test_a_negative_seat_temperature_refuses_at_panel_check(tmp_path):
    scenario = _panel(tmp_path)
    scenario.agents[1].temperature = -0.5
    with pytest.raises(multi_agent.ScenarioError,
                       match="seat 'Ben' declares temperature -0.5"):
        multi_agent.validate(scenario)


def test_verify_names_a_seat_temperature_that_equals_the_studys(tmp_path):
    scenario = _panel(tmp_path)
    scenario.agents[1].temperature = 0.7
    assert multi_agent.seat_temperature_advisories(scenario, 0.7) == [
        "seat 'Ben' declares temperature 0.7, which is the study's temperature, "
        "so the declaration changes nothing. Remove it from the panel, or give "
        "the seat the different temperature you meant."]
    assert multi_agent.seat_temperature_advisories(scenario, 0.2) == []


def test_the_client_panel_document_keeps_and_checks_the_new_keys():
    from steerlab_server.client import design_panels, panel_documents
    document = {"name": "p", "includeOwnReasoning": True, "agents": [
        {"id": "a", "name": "Ava", "baseModelID": "", "systemPrompt": "",
         "temperature": 0.4}], "turns": []}
    normalized = panel_documents.normalized(document)
    assert normalized["agents"][0]["temperature"] == 0.4
    assert design_panels.semantic(normalized)["agents"][0]["temperature"] == 0.4
    assert design_panels.semantic(normalized)["includeOwnReasoning"] is True
    with pytest.raises(Exception, match="temperature"):
        panel_documents.normalized({**document, "agents": [
            {**document["agents"][0], "temperature": True}]})
    with pytest.raises(Exception, match="includeOwnReasoning"):
        panel_documents.normalized({**document, "includeOwnReasoning": "yes"})


# --- 6. resume never mixes two routing rules ------------------------------------

def _old_engine_turn(turn_id, speaker, output):
    """A turn record as an engine from before issue #15 wrote it."""
    return {"turnID": turn_id, "turnIndex": 1, "title": "Ava votes",
            "speakerAgentID": speaker, "speakerName": "Ava", "prompt": "Vote.",
            "output": output, "outputLabel": "ava1", "temperature": 0.0,
            "seed": None}


def test_a_transcript_an_older_engine_started_with_a_thinking_seat_refuses(
        tmp_path, monkeypatch):
    from steerlab_server.experiment import resume
    scenario = _panel(tmp_path, endpoint=False)
    run_dir = os.path.join(str(tmp_path), "run")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "turns.jsonl"), "w") as handle:
        handle.write(json.dumps(_old_engine_turn("t1", "a", AVA_TEXT)) + "\n")
    generated = []
    monkeypatch.setattr(multi_agent, "generate",
                        lambda *a, **k: generated.append(1) or "never")
    with pytest.raises(resume.ResumeRefusal, match="seat 'Ava' thinks") as caught:
        multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"),
                                 scenario, run_dir=run_dir)
    assert generated == [], "refused before any turn was generated"
    assert caught.value.gate == "manifestEpoch"
    assert "Start a new run" in caught.value.repair_action


def test_the_parked_run_is_refused_before_the_model_loads(tmp_path):
    """The run-level admission reads every transcript of a parked run, with
    no model: the panel twin of the seed rule's admission."""
    from steerlab_server.experiment import resume
    scenario = _panel(tmp_path, endpoint=False)
    root = _workspace_with_panel(tmp_path, scenario)
    run_dir = os.path.join(str(tmp_path), "parked")
    transcript = os.path.join(run_dir, "configured", "replicate-1")
    os.makedirs(transcript)
    with open(os.path.join(transcript, "turns.jsonl"), "w") as handle:
        handle.write(json.dumps(_old_engine_turn("t1", "a", AVA_TEXT)) + "\n")
    problem = multi_agent.run_resume_rule_problem(run_dir, _manifest(), root)
    assert isinstance(problem, resume.ResumeRefusal)
    assert "seat 'Ava' thinks" in str(problem)
    # Turns this engine wrote carry their effort, and the run resumes.
    with open(os.path.join(transcript, "turns.jsonl"), "w") as handle:
        handle.write(json.dumps({**_old_engine_turn("t1", "a", AVA_ANSWER),
                                 "reasoningEffort": "xhigh"}) + "\n")
    assert multi_agent.run_resume_rule_problem(run_dir, _manifest(), root) is None


def test_a_merge_of_shards_played_under_two_routing_rules_refuses(tmp_path):
    from steerlab_server.experiment import sharding
    scenario = _panel(tmp_path, endpoint=False)
    root = _workspace_with_panel(tmp_path, scenario)
    stamps = [{"shardIndex": 0}, {"shardIndex": 1}]
    old = [{"output": AVA_TEXT}]
    new = [{"output": AVA_ANSWER, "reasoningEffort": "xhigh"}]
    with pytest.raises(sharding.ShardMergeRefusal, match="shard\\(s\\) 0 of 's'"):
        sharding._prove_one_panel_rule("s", _manifest(), stamps, [old, new], root)
    # One rule throughout, either one, merges.
    sharding._prove_one_panel_rule("s", _manifest(), stamps, [new, new], root)
    sharding._prove_one_panel_rule("s", _manifest(), stamps, [old, old], root)
    # A panel the change does not touch merges across the two.
    scenario.agents[0].variant_artifact_path = None
    plain_root = _workspace_with_panel(tmp_path / "plain", scenario)
    sharding._prove_one_panel_rule("s", _manifest(), stamps, [old, new], plain_root)


def test_a_transcript_an_older_engine_started_without_thinking_resumes(
        tmp_path, monkeypatch):
    scenario = _panel(tmp_path, endpoint=False)
    scenario.agents[0].variant_artifact_path = None
    run_dir = os.path.join(str(tmp_path), "run")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "turns.jsonl"), "w") as handle:
        handle.write(json.dumps(_old_engine_turn("t1", "a", "plain")) + "\n")
    prompts = []
    monkeypatch.setattr(multi_agent, "generate",
                        lambda model, prompt, **k: prompts.append(prompt) or "ok")
    multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"),
                             scenario, run_dir=run_dir)
    assert len(prompts) == 2 and "plain" in prompts[0]


# --- 7. the shared splitter, pinned across engines -------------------------------

def test_the_reasoning_split_fixture_is_current():
    """The Swift suite reads these bytes; a rule change that is not
    regenerated would leave both suites green over a broken seam."""
    assert os.path.exists(FIXTURE), (
        "reasoning-split.json is missing from Tests/Fixtures/cross-engine/: "
        "restore it from git, or regenerate it with "
        "scripts/regenerate-cross-engine-fixtures.py")
    with open(FIXTURE, encoding="utf-8") as handle:
        fixture = json.load(handle)
    assert len(fixture["split"]) >= 12
    for case in fixture["split"]:
        found = reasoning_split.split(case["text"])
        assert [found.reasoning, found.answer, found.state] == [
            case["reasoning"], case["answer"], case["state"]], case["label"]
    for case in fixture["turnText"]:
        said = multi_agent.turn_text(case["text"], thinking=case["thinking"],
                                     finish=case["finishReason"])
        assert [said.answer, said.reasoning, said.status] == [
            case["answer"], case["reasoning"], case["status"]], case["label"]
    for case in fixture["ownReasoning"]:
        assert multi_agent.own_reasoning_text(
            case["reasoning"], case["answer"]) == case["text"], case["label"]


# --- 8. an older engine is never sent a panel it would run differently ----------

def _workspace_with_panel(tmp_path, scenario):
    root = tmp_path / "ws"
    (root / "prompts" / "panels").mkdir(parents=True)
    (root / "prompts" / "panels" / "p.json").write_text(
        json.dumps(multi_agent._scenario_to_dict(scenario)))
    return str(root)


def _manifest(**fields):
    return SimpleNamespace(study_kind="multiAgent",
                           multi_agent_scenario_path="prompts/panels/p.json",
                           reasoning_effort="off", temperature=0.0,
                           content_hash=lambda: "h", **fields)


def test_a_panel_the_rule_changes_is_named_and_one_it_does_not_is_not(tmp_path):
    scenario = _panel(tmp_path, endpoint=False)
    scenario.agents[0].variant_artifact_path = None
    root = _workspace_with_panel(tmp_path, scenario)
    assert multi_agent.submission_panel_rule_changes(_manifest(), "run", root) == []
    # A thinking study makes its bare seats think.
    thinking = _manifest()
    thinking.reasoning_effort = "on"
    assert multi_agent.submission_panel_rule_changes(thinking, "run", root) == [
        "seat 'Ava' thinks", "seat 'Ben' thinks"]
    # A verb that plays no turns is never refused.
    assert multi_agent.submission_panel_rule_changes(thinking, "analyze", root) == []
    scenario.agents[1].temperature = 0.5
    root = _workspace_with_panel(tmp_path / "second", scenario)
    assert multi_agent.submission_panel_rule_changes(_manifest(), "pipeline", root) == [
        "seat 'Ben' declares its own temperature"]


def test_the_client_refuses_to_send_such_a_panel_to_an_engine_without_the_capability(
        tmp_path, monkeypatch):
    from steerlab_server.client import runner
    from steerlab_server.experiment import manifest as manifest_module
    scenario = _panel(tmp_path, endpoint=False)
    scenario.agents[1].temperature = 0.5
    root = _workspace_with_panel(tmp_path, scenario)
    monkeypatch.setattr(manifest_module.Manifest, "load",
                        classmethod(lambda cls, name, root=None: _manifest()))
    staged = {"experiment": "s", "experimentContentHash": "h"}
    with pytest.raises(runner.RunnerRefusal) as caught:
        runner.require_panel_answer_routing(staged, "run", {}, "http://runner", root)
    assert caught.value.code == "panelAnswerRoutingUnsupported"
    assert "remoteStudy.panelAnswerRouting" in str(caught.value)
    assert "seat 'Ben' declares its own temperature" in str(caught.value)
    # An engine that has it, nested or dotted, is sent the bundle.
    runner.require_panel_answer_routing(
        staged, "run", {"remoteStudy": {"panelAnswerRouting": True}},
        "http://runner", root)
    runner.require_panel_answer_routing(
        staged, "run", {"features": {"remoteStudy.panelAnswerRouting": True}},
        "http://runner", root)


def test_this_engine_advertises_answer_only_panel_routing():
    from steerlab_server.api import profile
    assert profile.capability_snapshot()["remoteStudy"]["panelAnswerRouting"] is True
    assert multi_agent.remote_has_panel_answer_routing(profile.capability_snapshot())


def test_verify_reads_the_pinned_panel_for_the_seat_temperature_advisory(tmp_path):
    scenario = _panel(tmp_path, endpoint=False)
    scenario.agents[1].temperature = 0.0
    root = _workspace_with_panel(tmp_path, scenario)
    notes = multi_agent.study_seat_temperature_advisories(_manifest(), root)
    assert len(notes) == 1 and notes[0].startswith(
        "seat 'Ben' declares temperature 0, which is the study's temperature")


def test_the_results_export_says_which_text_a_thinking_turns_response_is():
    from steerlab_server.client import results_export as rx
    thinking = {"experiment": "s", "condition": "configured", "promptID": "t1",
                "promptIndex": 0, "output": AVA_ANSWER, "reasoning": AVA_REASONING,
                "reasoningStatus": "complete", "speakerName": "Ava",
                "replicateIndex": 0, "turnTitle": "Ava votes"}
    plain = {"experiment": "s", "condition": "configured", "promptID": "t2",
             "promptIndex": 1, "output": "ok"}
    table = rx._responses_table("s", "run", {
        "responses": [thinking, plain],
        "responseFields": set(thinking) | set(plain)})
    meanings = {name: meaning for name, meaning, _ in table.columns}
    assert "reasoning" in meanings and "reasoning_status" in meanings
    assert "the answer the seat gave after its reasoning" in meanings["response"]
    table.fill([thinking, plain])
    row = dict(zip(table.names, table.rows[0]))
    assert row["response"] == rx._cell(AVA_ANSWER)
    assert row["reasoning"] == rx._cell(AVA_REASONING)
    # A study without a thinking turn keeps the sentence it always had.
    table = rx._responses_table("s", "run", {"responses": [plain],
                                             "responseFields": set(plain)})
    meanings = {name: meaning for name, meaning, _ in table.columns}
    assert meanings["response"] == "The text the model generated, as recorded."
    assert "reasoning" not in meanings


@pytest.mark.parametrize("text,expected", [
    # Written by hand, independently of the code under test.
    ("<think>a</think>\n\nb", ("a", "\n\nb", "closed")),
    ("  <THINK id='1'>a</ThInK>b", ("a", "b", "closed")),
    ("a\n</think>\n\nb", ("a\n", "\n\nb", "closed")),
    ("<think>never closes", ("never closes", "", "unclosed")),
    ("no tags", (None, "no tags", "none")),
    ("b <think>aside</think>", (None, "b <think>aside</think>", "none")),
    ("<thinking>x</think>y", ("<thinking>x", "y", "closed")),
])
def test_the_splitter_on_hand_written_cases(text, expected):
    found = reasoning_split.split(text)
    assert (found.reasoning, found.answer, found.state) == expected
