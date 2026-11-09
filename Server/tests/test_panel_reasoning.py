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
        name=name, base_model_id="test/qwen-panel", qwen_thinking_enabled=True)
    path = os.path.join(str(tmp_path), f"{name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact.to_dict(), handle)
    return path


def _panel(tmp_path, *, include_own_reasoning=False, endpoint=True):
    ava = multi_agent.Agent(id="a", name="Ava", base_model_id="test/qwen-panel",
                            variant_artifact_path=_thinking_agent(tmp_path))
    ben = multi_agent.Agent(id="b", name="Ben", base_model_id="test/qwen-panel")
    vote = (turn_endpoint.TurnEndpoint.from_dict(
        {"name": "vote", "kind": "choice", "marker": "Final answer:",
         "vocabulary": ["affirm", "reverse"]}) if endpoint else None)
    return multi_agent.Scenario(
        name="p", base_model_id="test/qwen-panel", include_own_reasoning=include_own_reasoning,
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
            model_id="test/qwen-panel", revision="r")), scenario, run_dir=run_dir, **kwargs)
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


@pytest.mark.parametrize("text,token_ids,budget,status", [
    # The block opened with a tag and the seat stopped before closing it.
    ("<think>weighing affirm against reverse, Final answer: affirm", [], None,
     "unfinished"),
    # Prompt-opened thinking that spent its whole reasoning budget: no tag.
    ("weighing affirm against reverse, Final answer: affirm", [5] * 4, 4,
     "cutOff"),
    # Prompt-opened thinking cut off by the turn's own token limit.
    ("weighing affirm against reverse, Final answer: affirm", [5] * 64, None,
     "cutOff"),
    # Prompt-opened thinking that stopped by itself mid-reasoning (B1 a).
    ("weighing it, Final answer: affirm", [], None, "unfinished"),
])
def test_reasoning_that_never_closed_passes_nothing_on(tmp_path, text,
                                                       token_ids, budget,
                                                       status):
    scenario = _panel(tmp_path)
    model = SimpleNamespace(model_id="test/qwen-panel", revision="r", tokenizer=_Tokenizer())
    prompts, _calls, records = _run(
        tmp_path, scenario, [(text, token_ids), ("ok", []), ("ok", [])],
        model=model, reasoning_max_tokens=budget)
    assert "affirm" not in prompts[1]
    assert "weighing" not in prompts[1]
    assert "weighing" not in prompts[2]
    first = records[0]
    assert first["reasoningStatus"] == status
    assert first["output"] == ""
    assert first["routedAgentIDs"] == []
    assert "weighing" in first["reasoning"]
    assert first["endpoint"]["unparsed"] is True
    # Ben's template read the label as empty, never as a verbatim placeholder.
    assert "{{outputs.ava1}}" not in prompts[1]


def test_a_thinking_seat_with_no_block_is_its_answer_only_where_the_template_says_so(
        tmp_path, monkeypatch):
    # The template leaves the opening tag to the model and no tag appears:
    # the whole text is the answer, and the record has no reasoning key, as
    # Swift writes it.
    monkeypatch.setattr(multi_agent, "seat_template", lambda model: True)
    monkeypatch.setattr(multi_agent, "prompt_opens_reasoning", lambda *a, **k: False)
    scenario = _panel(tmp_path, endpoint=False)
    prompts, _calls, records = _run(
        tmp_path, scenario, [("Plain answer.", []), ("ok", []), ("ok", [])])
    assert "Plain answer." in prompts[1]
    assert records[0]["reasoningStatus"] == "notFound"
    assert "reasoning" not in records[0]
    # The record says what the rendered prompt showed.
    assert records[0]["reasoningOpenedInPrompt"] is False


@pytest.mark.parametrize("opens", [True, None])
def test_with_a_prompt_opened_or_unknown_template_no_close_tag_passes_nothing(
        tmp_path, monkeypatch, opens):
    monkeypatch.setattr(multi_agent, "seat_template", lambda model: True)
    monkeypatch.setattr(multi_agent, "prompt_opens_reasoning", lambda *a, **k: opens)
    scenario = _panel(tmp_path, endpoint=False)
    prompts, _calls, records = _run(
        tmp_path, scenario, [("Plain answer.", []), ("ok", []), ("ok", [])])
    assert "Plain answer." not in prompts[1]
    assert records[0]["reasoningStatus"] == "unfinished"


def test_a_seat_whose_template_cannot_think_is_never_split(tmp_path, monkeypatch):
    """N3: an agent with thinking on, cast on a template without a thinking
    switch, writes plain text: nothing is split or budgeted, and the record
    says the turn did not think."""
    monkeypatch.setattr(multi_agent, "seat_template", lambda model: False)
    scenario = _panel(tmp_path, endpoint=False)
    text = "weighing it</think> my words"
    prompts, calls, records = _run(
        tmp_path, scenario, [(text, []), ("ok", []), ("ok", [])],
        reasoning_max_tokens=300)
    assert text in prompts[1]
    assert calls[0]["qwen_thinking_enabled"] is False
    assert "reasoning_max_tokens" not in calls[0]
    assert records[0]["reasoningEffort"] == "off"
    assert "reasoningStatus" not in records[0]


# Written by hand: the three leaks of review round 2 (B1).
LEAKS = [
    ("weighing it, Final answer: affirm", "stop"),
    ("Should I write a <think> tag? No.\n</think>\n\nFinal answer: reverse.",
     "stop"),
    ("Sure. <think>secret plan</think>Final answer: reverse.", "stop"),
]


@pytest.mark.parametrize("opens", [True, False, None])
def test_a_routed_answer_never_carries_a_reasoning_tag_or_leaked_reasoning(opens):
    corpus = [text for text, _ in LEAKS] + [
        AVA_TEXT, "<think>a</think>b", "a</think>b</think>c", "x <think>y",
        "answer <think>aside</think> more", "a</THINK>b", "<think>a</think>b <think",
        "plain words", "</think>", "<think>", ""]
    for text in corpus:
        for finish in ("stop", "length", "lengthInReasoning"):
            said = multi_agent.turn_text(text, thinking=True, finish=finish,
                                         opens_in_prompt=opens)
            if said.passed_on:
                folded = said.answer.lower()
                assert "<think" not in folded and "</think" not in folded, text
                assert "secret plan" not in said.answer
                if opens is not False:  # a model-opened template: plain text is an answer
                    assert "weighing it" not in said.answer


def test_the_three_leaks_are_closed():
    # (a) prompt-opened, stopped mid-reasoning: nothing passed on.
    said = multi_agent.turn_text(LEAKS[0][0], thinking=True, finish="stop",
                                 opens_in_prompt=True)
    assert (said.passed_on, said.status) == (False, "unfinished")
    # (b) a <think> mention inside prompt-opened reasoning: the first
    # </think> is still the boundary.
    said = multi_agent.turn_text(LEAKS[1][0], thinking=True, finish="stop",
                                 opens_in_prompt=True)
    assert (said.answer, said.status) == ("Final answer: reverse.", "complete")
    assert said.reasoning == "Should I write a <think> tag? No.\n"
    # (c) a tag mid-text with a model-opened template: unclear, withheld.
    said = multi_agent.turn_text(LEAKS[2][0], thinking=True, finish="stop",
                                 opens_in_prompt=False)
    assert (said.passed_on, said.status) == (False, "unclear")


def test_one_per_turn_function_decides_routing_endpoint_and_lint():
    """S4: the live path and the replay path share ``turn_outcome``; this
    drives ``play_turn`` with the raw text and checks every reader of it."""
    scenario = _panel_plain()
    turn = scenario.turns[0]
    outcome = multi_agent.play_turn(scenario, turn, 0, text=AVA_TEXT,
                                    thinking=True, finish="stop",
                                    opens_in_prompt=True)
    assert outcome.label == "ava1" and outcome.routed == ["a", "b"]
    readers = dict(outcome.entries)
    assert AVA_ANSWER in readers["b"] and "I first lean" not in readers["b"]
    assert "I first lean" not in readers["a"]
    assert outcome.fields["output"] == AVA_ANSWER
    assert outcome.fields["endpoint"]["value"] == "reverse"
    # The voice lint read the answer, not the reasoning: a reasoning line
    # spoken as a colleague would otherwise be flagged.
    lint = outcome.fields["voiceLint"]
    assert lint == multi_agent.voice_lint.stamp(
        AVA_ANSWER, speaker="Ava", others=["Ben"])
    # A reasoning line the lint would flag never reaches it.
    raw = "Ben: I agree with Ava.\n</think>\n\nFinal answer: reverse."
    flagged = multi_agent.play_turn(scenario, turn, 0, text=raw, thinking=True,
                                    finish="stop", opens_in_prompt=True)
    answer_only = multi_agent.voice_lint.stamp(
        "Final answer: reverse.", speaker="Ava", others=["Ben"])
    assert multi_agent.voice_lint.stamp(raw, speaker="Ava", others=["Ben"]) != answer_only
    assert flagged.fields["voiceLint"] == answer_only
    # Replaying the record gives the same routing.
    record = {"output": outcome.fields["output"],
              "reasoning": outcome.fields["reasoning"],
              "reasoningStatus": outcome.fields["reasoningStatus"]}
    replayed = multi_agent.turn_outcome(scenario, turn, 0,
                                        multi_agent._record_text(record))
    assert replayed.entries == outcome.entries


def _panel_plain():
    ava = multi_agent.Agent(id="a", name="Ava", base_model_id="test/qwen-panel")
    ben = multi_agent.Agent(id="b", name="Ben", base_model_id="test/qwen-panel")
    vote = turn_endpoint.TurnEndpoint.from_dict(
        {"name": "vote", "kind": "choice", "marker": "Final answer:",
         "vocabulary": ["affirm", "reverse"]})
    return multi_agent.Scenario(
        name="p", base_model_id="test/qwen-panel", agents=[ava, ben],
        turns=[multi_agent.Turn(id="t1", title="Ava votes", speaker_agent_id="a",
                                prompt_template="Vote.", output_label="ava1",
                                routing="all", endpoint=vote)])


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
        {"id": "a", "name": "Ava", "baseModelID": "test/qwen-panel", "systemPrompt": "",
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
        {"name": "p", "agents": [{"id": "a", "name": "Ava", "baseModelID": "test/qwen-panel",
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
    with pytest.raises(resume.ResumeRefusal,
                       match="the reasoning of seat 'Ava' was passed on") as caught:
        multi_agent.run_scenario(SimpleNamespace(model_id="test/qwen-panel", revision="r"),
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
    assert "the reasoning of seat 'Ava' was passed on" in str(problem)
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
    multi_agent.run_scenario(SimpleNamespace(model_id="test/qwen-panel", revision="r"),
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
    assert {case["opensInPrompt"] for case in fixture["turnText"]} == {True, False, None}
    for case in fixture["turnText"]:
        said = multi_agent.turn_text(case["text"], thinking=case["thinking"],
                                     finish=case["finishReason"],
                                     opens_in_prompt=case["opensInPrompt"])
        assert [said.answer, said.reasoning, said.status, said.withheld_text] == [
            case["answer"], case["reasoning"], case["status"],
            case["withheldText"]], case["label"]
    for case in fixture["opensReasoning"]:
        assert reasoning_split.opens_reasoning(case["promptTail"]) == case["opens"], \
            case["label"]
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
        ("seatWouldNotThink", "Ava"), ("seatWouldNotThink", "Ben")]
    assert multi_agent.outdated_panel_engine_reason(
        "http://runner", [("seatWouldNotThink", "Ava")]) == (
        "The engine at http://runner predates answer-only panel routing (its "
        "capabilities do not list remoteStudy.panelAnswerRouting), so on it seat "
        "'Ava' would not think. Nothing was submitted.")
    # A verb that plays no turns is never refused.
    assert multi_agent.submission_panel_rule_changes(thinking, "analyze", root) == []
    scenario.agents[1].temperature = 0.5
    root = _workspace_with_panel(tmp_path / "second", scenario)
    assert multi_agent.submission_panel_rule_changes(_manifest(), "pipeline", root) == [
        ("temperatureIgnored", "Ben")]
    # N2: a seat temperature equal to the study's, and includeOwnReasoning
    # with no thinking seat, change nothing.
    scenario.agents[1].temperature = 0.0
    scenario.include_own_reasoning = True
    root = _workspace_with_panel(tmp_path / "third", scenario)
    assert multi_agent.submission_panel_rule_changes(_manifest(), "run", root) == []
    # A panel that cannot be read here is None: the caller refuses.
    missing = _manifest()
    missing.multi_agent_scenario_path = "prompts/panels/gone.json"
    assert multi_agent.submission_panel_rule_changes(missing, "run", root) is None


def test_a_bundle_requires_answer_routing_only_where_the_panel_needs_it(
        tmp_path, monkeypatch):
    """S5: the requirement is computed from the files the bundle carries,
    so a staged bundle is refused by an older engine whatever the local
    workspace holds."""
    import tarfile
    from steerlab_server.experiment import bundles, instrumentation_contract
    scenario = _panel(tmp_path, endpoint=False)
    scenario.agents[0].variant_artifact_path = None
    scenario.agents[1].temperature = 0.5
    root = _workspace_with_panel(tmp_path, scenario)
    manifest = _manifest()
    manifest.content_hash = lambda: "h"
    manifest.validation_scope_hash = lambda: "v"
    manifest.verify = lambda root=None: []
    monkeypatch.setattr(bundles.Manifest, "load",
                        classmethod(lambda cls, name, root=None: manifest))
    monkeypatch.setattr(bundles, "_experiment_files", lambda *a, **k: [])

    def requirements(root):
        meta = bundles.package_experiment(
            "s", root=root, output_path=os.path.join(root, "b.tar.gz"))
        return meta["runtimeRequirements"]

    assert instrumentation_contract.PANEL_ANSWER_ROUTING in requirements(root)
    scenario.agents[1].temperature = None
    plain = _workspace_with_panel(tmp_path / "plain", scenario)
    assert instrumentation_contract.PANEL_ANSWER_ROUTING not in requirements(plain)
    # An older engine (no such entry in its instrumentation list) refuses it.
    with pytest.raises(ValueError, match="predates answer-only panel routing"):
        instrumentation_contract.require(
            [instrumentation_contract.PANEL_ANSWER_ROUTING],
            {"instrumentation": ["policy-v1"]})
    instrumentation_contract.require(
        [instrumentation_contract.PANEL_ANSWER_ROUTING],
        {"instrumentation": list(instrumentation_contract.SUPPORTED)})
    # Keyed to the verbs that play turns: an analyze bundle needs nothing.
    scenario.agents[1].temperature = 0.5
    changed = _workspace_with_panel(tmp_path / "again", scenario)
    meta = bundles.package_experiment("s", root=changed, verb="analyze",
                                      output_path=os.path.join(changed, "a.tar.gz"))
    assert instrumentation_contract.PANEL_ANSWER_ROUTING not in meta["runtimeRequirements"]


def test_verify_allows_a_budget_only_where_a_cast_seat_thinks(tmp_path):
    """S1: a study whose effort is off may declare reasoningMaxTokens when
    its panel casts an agent that thinks; with nothing thinking it is still
    refused."""
    from steerlab_server.experiment import prompt_render
    scenario = _panel(tmp_path, endpoint=False)
    root = _workspace_with_panel(tmp_path, scenario)
    document = {"studyKind": "multiAgent", "temperature": 0.0,
                "multiAgentScenarioPath": "prompts/panels/p.json"}
    refusal = [prompt_render.BUDGET_WITHOUT_EFFORT_REASON]
    assert multi_agent.permit_panel_budget(refusal, document, root) == []
    scenario.agents[0].variant_artifact_path = None
    plain = _workspace_with_panel(tmp_path / "plain", scenario)
    assert multi_agent.permit_panel_budget(refusal, document, plain) == refusal
    assert multi_agent.permit_panel_budget(
        refusal, {**document, "studyKind": "modelOutput"}, root) == refusal


def test_the_seats_temperatures_decide_a_panels_seed_rule(tmp_path):
    """S3: a greedy study with one warm seat is not a greedy run."""
    from steerlab_server.experiment import panel_workflow, sampling
    scenario = _panel(tmp_path, endpoint=False)
    scenario.agents[1].temperature = 0.7
    root = _workspace_with_panel(tmp_path, scenario)
    manifest = _manifest(samples_per_item=3)
    assert sampling.generating_temperature(manifest, root) == 0.7
    assert sampling.greedy_samples_problem(
        sampling.generating_temperature(manifest, root), 3, "multiAgent") is None
    scenario.agents[1].temperature = None
    cold = _workspace_with_panel(tmp_path / "cold", scenario)
    assert sampling.generating_temperature(manifest, cold) == 0.0
    assert sampling.greedy_samples_problem(
        sampling.generating_temperature(manifest, cold), 3, "multiAgent")
    # Each flattened turn states its own seed facts.
    assert panel_workflow._panel_seed_fields(0.7) == {
        "seedInert": False, "seedPolicy": "derivedSHA256"}
    assert panel_workflow._panel_seed_fields(0.0) == {
        "seedInert": True, "seedPolicy": "greedy"}
    # And the export states a mixed panel's policy as what it is.
    from steerlab_server.client import results_export as rx
    text = rx._seed_policy_text({"seedPolicy": "derivedSHA256"}, {},
                                {"seedPolicy": ["greedy", "derivedSHA256"]},
                                code=lambda v: v)
    assert text == ("seed policy derivedSHA256 for the turns generated above "
                    "temperature 0, and no seed for the greedy turns at "
                    "temperature 0")


def test_a_withheld_turn_exposes_nobody():
    """N6: script-level exposure ignores a turn that passed nothing on."""
    from steerlab_server.experiment import panel_effects
    scenario = _panel_plain()
    scenario.turns.append(multi_agent.Turn(
        id="t2", title="Ben", speaker_agent_id="b", prompt_template="Go.",
        output_label="ben1", routing="all"))
    assert panel_effects.exposure_by_turn(scenario, {"a"})["t2"] is True
    withheld = panel_effects.withheld_turn_ids(
        [{"turnID": "t1", "reasoningStatus": "cutOff"},
         {"turnID": "t2", "reasoningStatus": "complete"}])
    assert withheld == {"t1"}
    assert panel_effects.exposure_by_turn(scenario, {"a"}, withheld)["t2"] is False


@pytest.mark.parametrize("status", ["cutOff", "unfinished", "unclear"])
def test_the_transcript_says_a_withheld_turn_passed_nothing_on(status):
    text = multi_agent._transcript(_panel_plain(), [
        {"turnIndex": 1, "title": "Ava votes", "speakerName": "Ava", "output": "",
         "reasoning": "weighing it", "reasoningStatus": status}])
    assert (f"_Nothing from this turn was passed on (reasoning status {status}). "
            "What it wrote:_") in text
    assert "weighing it" in text


def test_current_engines_accept_their_own_panel_bundles(tmp_path, monkeypatch):
    """R1: the requirement a panel bundle carries is one this engine
    supports, at submission (``_bundle_execute_command``), in the queued
    worker (``instrumented_bundle.main``), and at execution (the check
    ``_execute_run_bundle_inner`` makes)."""
    from steerlab_server import cli, instrumented_bundle
    from steerlab_server.api.submissions import _bundle_execute_command
    from steerlab_server.experiment import instrumentation_contract as contract
    required = [contract.PANEL_ANSWER_ROUTING]
    kwargs = dict(verb="run", target_root=str(tmp_path), dtype="float32",
                  device="cpu", prompts_path=None, source_path=None,
                  package_evidence=False, record_path=str(tmp_path / "record.json"))
    command = _bundle_execute_command("bundle.tar.gz", **kwargs,
                                      runtime_requirements=required)
    assert command[2] == "steerlab_server.instrumented_bundle"
    calls = []
    monkeypatch.setattr(cli, "main", lambda args: calls.append(args) or 0)
    assert instrumented_bundle.main([command[3], *command[4:]]) == 0 and calls
    # The execution step's own check, through the real function: it gets past
    # the requirement and stops only at the stub's deliberate non-bundle.
    from steerlab_server.experiment import bundles
    monkeypatch.setattr(bundles, "inspect_bundle", lambda path: {
        "runtimeRequirements": required, "kind": "notARunBundle"})
    with pytest.raises(bundles.BundleError, match="requires a runBundle"):
        bundles._execute_run_bundle_inner(
            "bundle.tar.gz", verb="run", target_root=str(tmp_path), dtype="float32",
            device=None, prompts_path=None, source_path=None,
            package_evidence_on_complete=False, record_path=None, checkpoint=None,
            shard=None, resume_from=None, captured_logs=[])
    # An engine without it still refuses, with the panel's own sentence.
    with pytest.raises(ValueError, match="predates answer-only panel routing"):
        contract.require(required, {"instrumentation": ["policy-v1"]})


def test_the_rendered_prompt_decides_whether_the_block_opened(monkeypatch):
    """R2: the fact comes from this turn's own rendered prompt."""
    model = SimpleNamespace(model_id="test/qwen-panel", tokenizer=object())
    for tail, opens in (("<|im_start|>assistant\n<think>\n", True),
                        ("<|im_start|>assistant\n<think>\n\n</think>\n\n", False),
                        ("<|im_start|>assistant\n", False)):
        monkeypatch.setattr(multi_agent.prompt_render, "render_for_generation",
                            lambda *a, _t=tail, **k: SimpleNamespace(text="x" + _t))
        assert multi_agent.prompt_opens_reasoning(
            model, "p", prompt_mode="chatAssistant", system_prompt=None,
            reasoning_effort=None) is opens
    def unrenderable(*a, **k):
        raise RuntimeError("no template here")
    monkeypatch.setattr(multi_agent.prompt_render, "render_for_generation", unrenderable)
    assert multi_agent.prompt_opens_reasoning(
        model, "p", prompt_mode="chatAssistant", system_prompt=None,
        reasoning_effort=None) is None


def test_an_agent_that_asks_to_think_is_taken_at_its_word(tmp_path):
    """R3: neither the id heuristic nor another revision's record may
    clear a thinking seat for an older engine; only an exact-revision probed
    record that says the template cannot think does."""
    from steerlab_server.experiment import model_capabilities as mc
    assert multi_agent._template_thinks("acme/plain-model", "r1", str(tmp_path))
    record = mc.heuristic("acme/plain-model", "r1")
    record["source"] = mc.SOURCE_PROBE
    record["detected"]["thinkingSwitch"] = mc.UNSUPPORTED
    mc.write_record(record, str(tmp_path))
    assert not multi_agent._template_thinks("acme/plain-model", "r1", str(tmp_path))
    assert multi_agent._template_thinks("acme/plain-model", "r2", str(tmp_path))
    # A thinking agent on a model whose id names no thinking family.
    scenario = _panel(tmp_path, endpoint=False)
    for seat in scenario.agents:
        seat.base_model_id = "acme/plain-model"
    artifact = model_variant.ModelVariant(name="t", base_model_id="acme/plain-model",
                                          qwen_thinking_enabled=True)
    with open(scenario.agents[0].variant_artifact_path, "w") as handle:
        json.dump(artifact.to_dict(), handle)
    assert multi_agent.panel_rule_changes(
        scenario, study_effort="off", study_temperature=0.0,
        root=str(tmp_path)) == [("reasoningPassedOn", "Ava")]


def test_an_unclear_turn_keeps_its_text_apart_and_only_complete_turns_note_the_marker():
    said = multi_agent.turn_text(LEAKS[2][0], thinking=True, finish="stop",
                                 opens_in_prompt=False)
    assert (said.reasoning, said.withheld_text) == (None, LEAKS[2][0])
    scenario = _panel_plain()
    outcome = multi_agent.turn_outcome(scenario, scenario.turns[0], 0, said)
    assert outcome.fields["withheldText"] == LEAKS[2][0]
    assert "reasoning" not in outcome.fields
    cut = multi_agent.turn_text("Final answer: affirm, weighing", thinking=True,
                                finish="length", opens_in_prompt=True)
    stamp = multi_agent.turn_outcome(scenario, scenario.turns[0], 0, cut).fields["endpoint"]
    assert stamp == {"name": "vote", "value": None, "unparsed": True}


def test_unknown_statuses_fail_closed():
    from steerlab_server.experiment import panel_effects
    assert not multi_agent.TurnText(answer="x", status="aLaterStatus").passed_on
    assert multi_agent.TurnText(answer="x").passed_on
    assert multi_agent.TurnText(answer="x", status="notFound").passed_on
    assert panel_effects.withheld_turn_ids(
        [{"turnID": "t1", "reasoningStatus": "aLaterStatus"}]) == {"t1"}


def test_only_seats_that_speak_count_toward_the_generating_temperature(tmp_path):
    from steerlab_server.experiment import sampling
    scenario = _panel(tmp_path, endpoint=False)
    scenario.agents.append(multi_agent.Agent(id="c", name="Cy",
                                             base_model_id="test/qwen-panel",
                                             temperature=0.9))
    root = _workspace_with_panel(tmp_path, scenario)
    assert sampling.generating_temperature(_manifest(), root) == 0.0


def test_a_heuristic_never_clears_a_seat_that_asks_to_think(tmp_path):
    """Round 4: a thinking agent on a model whose id names no thinking
    family, with no probe of its template, is still split; the prompt rule
    then decides, and its reasoning never reaches a colleague."""
    scenario = _panel(tmp_path, endpoint=False)
    for seat in scenario.agents:
        seat.base_model_id = "acme/plain-model"
    scenario.base_model_id = "acme/plain-model"
    artifact = model_variant.ModelVariant(name="t", base_model_id="acme/plain-model",
                                          qwen_thinking_enabled=True)
    with open(scenario.agents[0].variant_artifact_path, "w") as handle:
        json.dump(artifact.to_dict(), handle)
    prompts, calls, records = _run(
        tmp_path, scenario, [(AVA_TEXT, []), ("ok", []), ("x</think>y", [])],
        model=SimpleNamespace(model_id="acme/plain-model", revision="r"))
    assert calls[0]["qwen_thinking_enabled"] is True
    assert "I first lean" not in prompts[1]
    assert records[0]["reasoningStatus"] == "complete"
    # Its prompt could not be rendered here, so the split read it as opened,
    # and the record says so.
    assert records[0]["reasoningOpenedInPrompt"] is True


def test_the_transcript_withholds_an_unknown_status_too():
    text = multi_agent._transcript(_panel_plain(), [
        {"turnIndex": 1, "title": "Ava votes", "speakerName": "Ava", "output": "",
         "reasoning": "weighing it", "reasoningStatus": "aLaterStatus"}])
    assert "Nothing from this turn was passed on (reasoning status aLaterStatus)" in text


def test_the_judge_reads_thinking_and_attributed_closes_as_text():
    """N4: ``<thinking>`` is not an opening tag and ``</think attr>`` is not
    a closing one, on this engine as on the Swift twin."""
    from steerlab_server.experiment import paired_judge
    body = '{"winner": "A"}'
    assert paired_judge.strip_leading_think_block("<thinking>x</thinking>" + body) == (
        "<thinking>x</thinking>" + body)
    assert paired_judge.strip_leading_think_block("x</think attr>" + body) == (
        "x</think attr>" + body)
    assert paired_judge.strip_leading_think_block("<thinking>x</think>" + body) == body


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
