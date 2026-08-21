"""Panel turn contracts (docs/PANEL-TURN-CONTRACTS-SPEC.md).

Two engines render panel prompts, and a prompt that differs between them is a
study that cannot be pooled — so the load-bearing test here is the fixture
replay against ``prompts/fixtures/panel-render/``, whose worked-example turn is
the spec's §2 example verbatim. The rest pin the schema, the two renderer
changes that affect EXISTING scenarios (reader-aware context entries, fallback
prepend), and each validate error / advisory.
"""

import json
import os
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import multi_agent as ma
from steerlab_server.experiment import scenario_preflight


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(REPO, "prompts", "fixtures", "panel-render")


# --- fixture replay ---------------------------------------------------------

def _fixture_scenario():
    with open(os.path.join(FIXTURE, "scenario.json"), "rb") as handle:
        return ma.Scenario.from_dict(json.loads(handle.read()))


#: The stub outputs README.md publishes, so the Swift twin replays the same
#: transcript. Keyed by output label, in turn order.
STUB_OUTPUTS = {"t1_ava": "AVA DRAFT.", "t1_ben": "BEN DRAFT.",
                "t2_ava": "AVA RESPONSE.", "t3_ava": "AVA RECAP.",
                "t4_ben": "Verdict: yes", "t5_cal": "CAL NOTE."}


def test_the_fixture_panel_replays_byte_for_byte(tmp_path, monkeypatch):
    """The cross-engine claim, exercised through the REAL runner rather than a
    hand-rolled walk: routing, label assignment and per-reader context marking
    all have to be the runner's, or the fixture pins something no run
    produces."""
    scenario = _fixture_scenario()
    labels = [t.output_label for t in scenario.turns]
    served = iter(STUB_OUTPUTS[label] for label in labels)
    monkeypatch.setattr(ma, "generate", lambda *a, **k: next(served))

    # The seats name `fixture/model`, and the runner refuses to substitute the
    # loaded model for a declared one — so the stub has to be that model.
    ma.run_scenario(SimpleNamespace(model_id="fixture/model", revision="r"),
                    scenario, run_dir=str(tmp_path))

    records = [json.loads(line) for line in
               open(os.path.join(tmp_path, "turns.jsonl"), encoding="utf-8")]
    assert [r["outputLabel"] for r in records] == labels
    for record in records:
        path = os.path.join(FIXTURE, "expected", f"{record['outputLabel']}.txt")
        with open(path, encoding="utf-8") as handle:
            expected = handle.read()
        assert record["prompt"] == expected, record["outputLabel"]


def test_the_worked_example_turn_matches_the_spec_verbatim():
    """Spec §2's example is normative. It is reproduced here as a literal so a
    renderer change that silently rewrites BOTH the code and the fixture still
    has to face the spec text."""
    with open(os.path.join(FIXTURE, "expected", "t2_ava.txt"),
              encoding="utf-8") as handle:
        rendered = handle.read()

    assert rendered == (
        "You are Ava, a reviewer. The other participants are Ben and Cal. The "
        "group has exchanged first drafts and you are now writing your "
        "response.\n"
        "\n"
        "===== SHARED MATERIALS =====\n"
        "MATERIAL TEXT\n"
        "===== END OF SHARED MATERIALS =====\n"
        "\n"
        "That was the shared material. Every participant has read it.\n"
        "\n"
        "===== YOUR OWN EARLIER OUTPUT — First draft — Ava =====\n"
        "AVA DRAFT.\n"
        "===== END OF YOUR OWN EARLIER OUTPUT =====\n"
        "\n"
        "That was your own earlier output, written by you, Ava.\n"
        "\n"
        "===== OUTPUT OF Ben — First draft — Ben =====\n"
        "BEN DRAFT.\n"
        "===== END OF OUTPUT OF Ben =====\n"
        "\n"
        "Those were the contributions of the other participants. You have now "
        "read them.\n"
        "\n"
        "===== YOUR TASK =====\n"
        "You are Ava. Write a response memo to your colleagues.\n"
        "\n"
        "Write only your own response, in your own voice, as Ava and no one "
        "else. Do not write, draft, continue, quote at length, or reply on "
        "behalf of Ben or Cal. Their contributions above are finished "
        "documents; you are adding one document of your own.\n"
        "\n"
        "Use exactly this format and nothing else:\n"
        "Verdict: yes OR no\n"
        "\n"
        "Reminder: you are Ava. Respond as Ava and as no one else.")


def test_no_expected_prompt_ends_in_a_newline():
    """A rendered prompt ends at its last character. A stray trailing newline
    in the fixture would make every engine that reads it disagree with the
    runner."""
    for name in sorted(os.listdir(os.path.join(FIXTURE, "expected"))):
        with open(os.path.join(FIXTURE, "expected", name), "rb") as handle:
            assert not handle.read().endswith(b"\n"), name


def test_the_fixture_panel_is_clean():
    """Deliberate: a validate error or advisory here is a regression, not a
    property of the fixture."""
    scenario = _fixture_scenario()
    ma.validate(scenario)
    assert ma.advisories(scenario) == []


def test_the_fixture_scenario_round_trips_byte_for_byte():
    with open(os.path.join(FIXTURE, "scenario.json"), encoding="utf-8") as handle:
        raw = handle.read()
    written = json.dumps(ma._scenario_to_dict(_fixture_scenario()),
                         indent=2, sort_keys=True) + "\n"
    assert written == raw


# --- schema: decode, round-trip, schemaVersion ------------------------------

def _contract_payload(**overrides):
    payload = {
        "schemaVersion": 2, "name": "p", "baseModelID": "m",
        "sharedMaterials": "M", "temperature": 0, "maxTokens": 128,
        "agents": [{"id": "a", "name": "A", "baseModelID": "m",
                    "role": "a reviewer"},
                   {"id": "b", "name": "B", "baseModelID": "m"}],
        "turns": [
            {"id": "t1", "title": "One", "speakerAgentID": "a",
             "promptTemplate": "go", "outputLabel": "one", "routing": "all"},
            {"id": "t2", "title": "Two", "speakerAgentID": "b",
             "promptTemplate": "", "outputLabel": "two", "routing": "all",
             "contract": {"stage": "S", "task": "T", "format": "F",
                          "inputs": ["one"], "ownVoice": True,
                          "materialsTitle": "THE RECORD"}},
        ]}
    payload.update(overrides)
    return payload


def test_a_contract_turn_decodes_every_key():
    scenario = ma.Scenario.from_dict(_contract_payload())
    contract = scenario.turns[1].contract

    assert scenario.agents[0].role == "a reviewer"
    assert scenario.agents[1].role == ""
    assert (contract.stage, contract.task, contract.format) == ("S", "T", "F")
    assert contract.inputs == ("one",)
    assert contract.own_voice is True
    assert contract.materials_title == "THE RECORD"
    assert scenario.turns[0].contract is None


def test_contract_defaults_are_the_spec_defaults():
    payload = _contract_payload()
    payload["turns"][1]["contract"] = {"task": "T"}
    contract = ma.Scenario.from_dict(payload).turns[1].contract

    assert contract.stage == "" and contract.format == ""
    assert contract.inputs == ()
    assert contract.own_voice is True
    assert contract.materials_title == ma.DEFAULT_MATERIALS_TITLE


def test_a_contract_round_trips_through_the_canonical_dialect():
    scenario = ma.Scenario.from_dict(_contract_payload())
    turn = ma._scenario_to_dict(scenario)["turns"][1]

    assert turn["contract"] == {
        "stage": "S", "task": "T", "format": "F", "inputs": ["one"],
        "ownVoice": True, "materialsTitle": "THE RECORD"}


def test_a_scenario_without_the_new_fields_is_unchanged_by_a_re_save():
    """The compatibility floor: every panel written before contracts existed
    must re-save byte for byte, or every pinned panel hash moves at once."""
    payload = {
        "schemaVersion": 1, "name": "p", "baseModelID": "m",
        "description": "", "sharedMaterials": "M", "temperature": 0.0,
        "maxTokens": 128,
        "agents": [{"id": "a", "name": "A", "baseModelID": "",
                    "systemPrompt": "", "variantArtifactPath": None,
                    "variantArtifactHash": None}],
        "turns": [{"id": "t1", "title": "One", "speakerAgentID": "a",
                   "promptTemplate": "go", "outputLabel": "one",
                   "routing": "all", "routedAgentIDs": [],
                   "includeScenarioMaterials": True,
                   "includeSpeakerContext": True, "maxTokens": None}]}
    before = json.dumps(payload, indent=2, sort_keys=True)

    after = json.dumps(ma._scenario_to_dict(ma.Scenario.from_dict(payload)),
                       indent=2, sort_keys=True)

    assert after == before
    assert "contract" not in after and "role" not in after
    assert "acknowledgedInputs" not in after


def test_schema_version_is_derived_from_content():
    """2 when any turn carries a contract, 1 otherwise — computed on both
    engines from the same evidence, never carried over from the file, so the
    two cannot disagree about what a given scenario is."""
    with_contract = ma.Scenario.from_dict(_contract_payload())
    assert ma._scenario_to_dict(with_contract)["schemaVersion"] == 2

    # A file CLAIMING 2 with no contract in it writes 1.
    plain = _contract_payload()
    del plain["turns"][1]["contract"]
    plain["turns"][1]["promptTemplate"] = "go"
    assert ma._scenario_to_dict(ma.Scenario.from_dict(plain))["schemaVersion"] == 1


def test_both_schema_versions_read():
    for version in (1, 2):
        payload = _contract_payload(schemaVersion=version)
        assert ma.Scenario.from_dict(payload).turns[1].contract is not None


def test_a_structurally_impossible_contract_refuses_at_decode():
    """Lenient decode has a floor: an empty task or a dangling input is a
    validate error (there is a half-written thing to hold), but a contract that
    is not an object, or whose inputs are not a list, is nothing at all."""
    for bad in ("not an object", {"task": "T", "inputs": "one"}):
        payload = _contract_payload()
        payload["turns"][1]["contract"] = bad
        with pytest.raises(ma.ScenarioError):
            ma.Scenario.from_dict(payload)


def test_an_empty_task_survives_decode_and_dies_at_validate():
    payload = _contract_payload()
    payload["turns"][1]["contract"] = {"task": "  "}
    scenario = ma.Scenario.from_dict(payload)  # decodes fine

    with pytest.raises(ma.ScenarioError, match="contract with no task"):
        ma.validate(scenario)


# --- validate errors (spec §4, plus the mirrored base-model rule) ----------

def test_every_seat_must_name_its_own_base_model():
    """Mirrors the Swift twin, and closes a silent fallback here: a blank seat
    otherwise inherits whatever model happens to be loaded, so an uncompiled
    semantic panel runs and looks entirely normal."""
    scenario = ma.Scenario(
        name="p",
        agents=[ma.Agent(id="a", name="A", base_model_id="m"),
                ma.Agent(id="b", name="Bare")],
        turns=[ma.Turn(id="t1", title="One", speaker_agent_id="a",
                       prompt_template="go")])

    with pytest.raises(ma.ScenarioError, match="agent 'Bare' needs a base model"):
        ma.validate(scenario)


def test_a_blank_base_model_still_decodes():
    """Lenient decode, strict validate: a draft can hold a seat with no model
    yet, it just cannot run."""
    payload = _contract_payload()
    payload["agents"][1]["baseModelID"] = ""

    scenario = ma.Scenario.from_dict(payload)  # decodes fine

    assert scenario.agents[1].base_model_id == ""
    with pytest.raises(ma.ScenarioError, match="needs a base model"):
        ma.validate(scenario)


# --- contract validate errors ----------------------------------------------

def _contract_scenario(**contract_kwargs):
    contract_kwargs.setdefault("task", "Do the thing.")
    return ma.Scenario(
        name="p", shared_materials="M",
        agents=[ma.Agent(id="a", name="A", base_model_id="m"),
                ma.Agent(id="b", name="B", base_model_id="m")],
        turns=[
            ma.Turn(id="t1", title="One", speaker_agent_id="a",
                    prompt_template="go", output_label="one"),
            ma.Turn(id="t2", title="Two", speaker_agent_id="b",
                    prompt_template="", output_label="two",
                    contract=ma.TurnContract(**contract_kwargs)),
        ])


def test_a_contract_turn_needs_a_task():
    with pytest.raises(ma.ScenarioError, match="contract with no task"):
        ma.validate(_contract_scenario(task=""))


def test_a_contract_plus_a_template_is_never_a_merge():
    scenario = _contract_scenario()
    scenario.turns[1].prompt_template = "also this"

    with pytest.raises(ma.ScenarioError, match="exactly one renderer"):
        ma.validate(scenario)


def test_a_contract_turn_needs_no_template():
    """The other half of the same rule: an empty promptTemplate is REQUIRED of
    a contract turn, so the 'needs a prompt template' check must not fire."""
    ma.validate(_contract_scenario())


@pytest.mark.parametrize("inputs", [("nope",), ("two",), ("three",)])
def test_a_contract_input_must_name_an_earlier_label(inputs):
    """Unknown, self, and forward references all refuse — contract inputs are
    structured data, so a dangling one leaves nothing in the prompt to notice
    (the template form at least ships `{{outputs.X}}` verbatim)."""
    scenario = _contract_scenario(inputs=inputs)
    scenario.turns.append(ma.Turn(id="t3", title="Three", speaker_agent_id="a",
                                  prompt_template="go", output_label="three"))

    with pytest.raises(ma.ScenarioError, match="is not produced by any earlier turn"):
        ma.validate(scenario)


def test_an_earlier_label_is_accepted():
    ma.validate(_contract_scenario(inputs=("one",)))


@pytest.mark.parametrize("field,placeholder", [
    ("stage", "{{scenario.materials}}"),
    ("task", "{{agent.context}}"),
    ("format", "{{outputs.one}}"),
    ("materials_title", "{{scenario.materials}}"),
])
def test_forbidden_placeholders_in_contract_text_refuse(field, placeholder):
    """Those three have canonical slots; the layout is the point."""
    with pytest.raises(ma.ScenarioError, match="its own canonical slot"):
        ma.validate(_contract_scenario(**{field: f"x {placeholder} y"}))


def test_the_refusal_spells_the_outputs_reference_it_found():
    """The author has to find it in their own text, so the message carries the
    label — the bare marker only when there are no closing braces to read one
    from. Mirrors MultiAgentRunner.forbiddenContractPlaceholder."""
    with pytest.raises(ma.ScenarioError, match=r"uses \{\{outputs\.one\}\}, which"):
        ma.validate(_contract_scenario(format="see {{outputs.one}} now"))

    with pytest.raises(ma.ScenarioError, match=r"uses \{\{outputs\., which"):
        ma.validate(_contract_scenario(format="see {{outputs.one"))

    # Declaration order, not text position: the exact placeholders win.
    with pytest.raises(ma.ScenarioError, match=r"uses \{\{agent\.context\}\}"):
        ma.validate(_contract_scenario(
            format="{{outputs.one}} then {{agent.context}}"))


@pytest.mark.parametrize("field", ["stage", "task", "format", "materials_title"])
def test_the_allowed_substitutions_are_not_refused(field):
    ma.validate(_contract_scenario(
        **{field: "{{scenario.name}} {{scenario.description}} "
                  "{{agent.name}} {{turn.title}}"}))


def test_the_allowed_substitutions_reach_the_rendered_text():
    scenario = _contract_scenario(
        task="For {{scenario.name}}, {{agent.name}} on {{turn.title}}.")
    scenario.description = "D"
    scenario.turns[1].contract = ma.TurnContract(
        task=scenario.turns[1].contract.task, stage="{{scenario.description}}")

    prompt = ma._render_prompt(scenario, scenario.turns[1], "", "B", {})

    assert "You are B. The other participants are A. D" in prompt
    assert "You are B. For p, B on Two." in prompt


# --- advisories (spec §4) ---------------------------------------------------

def test_a_private_turn_read_by_name_draws_an_advisory():
    """`{{outputs.X}}` and contract inputs both bypass routing entirely, so a
    speakerOnly note-to-self can be read by everyone and nothing says so."""
    scenario = ma.Scenario(
        name="p",
        agents=[ma.Agent(id="a", name="A", base_model_id="m"),
                ma.Agent(id="b", name="B", base_model_id="m")],
        turns=[
            ma.Turn(id="t1", title="Private note", speaker_agent_id="a",
                    prompt_template="go", output_label="secret",
                    routing="speakerOnly"),
            ma.Turn(id="t2", title="Reader", speaker_agent_id="b",
                    prompt_template="see {{outputs.secret}}", output_label="two"),
        ])

    notes = ma.advisories(scenario)

    assert any("private turn is leaking" in n and "'secret'" in n
               and "never routed to this turn's speaker" in n for n in notes)
    ma.validate(scenario)  # advisory only — a draft is never blocked


def test_a_contract_input_gets_the_same_private_read_advisory():
    scenario = ma.Scenario(
        name="p",
        agents=[ma.Agent(id="a", name="A", base_model_id="m"),
                ma.Agent(id="b", name="B", base_model_id="m")],
        turns=[
            ma.Turn(id="t1", title="Private note", speaker_agent_id="a",
                    prompt_template="go", output_label="secret",
                    routing="speakerOnly"),
            ma.Turn(id="t2", title="Reader", speaker_agent_id="b",
                    prompt_template="", output_label="two",
                    contract=ma.TurnContract(task="T", inputs=("secret",))),
        ])

    assert any("private turn is leaking" in n for n in ma.advisories(scenario))


def test_reading_a_routed_output_draws_nothing():
    scenario = ma.Scenario(
        name="p",
        agents=[ma.Agent(id="a", name="A", base_model_id="m"),
                ma.Agent(id="b", name="B", base_model_id="m")],
        turns=[
            ma.Turn(id="t1", title="Open", speaker_agent_id="a",
                    prompt_template="go", output_label="open", routing="all"),
            ma.Turn(id="t2", title="Reader", speaker_agent_id="b",
                    prompt_template="", output_label="two",
                    contract=ma.TurnContract(task="T", inputs=("open",))),
        ])

    assert ma.advisories(scenario) == []


def test_a_contract_input_reading_a_selected_route_that_includes_it_is_quiet():
    scenario = ma.Scenario(
        name="p",
        agents=[ma.Agent(id="a", name="A", base_model_id="m"),
                ma.Agent(id="b", name="B", base_model_id="m")],
        turns=[
            ma.Turn(id="t1", title="Open", speaker_agent_id="a",
                    prompt_template="go", output_label="open",
                    routing="selected", routed_agent_ids=["b"]),
            ma.Turn(id="t2", title="Reader", speaker_agent_id="b",
                    prompt_template="see {{outputs.open}}", output_label="two"),
        ])

    assert ma.advisories(scenario) == []


def test_fence_marks_in_the_shared_materials_draw_an_advisory():
    scenario = _contract_scenario()
    scenario.shared_materials = "Exhibit A\n===== EXHIBIT B ====="

    assert any("fence marker" in n for n in ma.advisories(scenario))
    ma.validate(scenario)


def test_a_strict_format_turn_with_room_for_an_opinion_draws_an_advisory():
    """The measured failure: a disposition turn with 2048 tokens of room
    appended a whole opinion after its answer."""
    scenario = _contract_scenario()
    scenario.max_tokens = 2048
    scenario.turns[1].endpoint = ma.TurnEndpoint(
        name="verdict", kind="choice", marker="Verdict:", vocabulary=("yes", "no"))

    assert any("format contamination" in n and "2048" in n
               for n in ma.advisories(scenario))

    # Under the budget, the same turn is quiet.
    scenario.turns[1].max_tokens = ma.STRICT_FORMAT_TOKEN_BUDGET
    assert not [n for n in ma.advisories(scenario) if "format contamination" in n]


def test_advisories_come_out_in_the_cross_engine_order():
    """Order is part of the contract — a researcher reads these as a list and
    compares engines. Scenario-level fence first, then per turn in turn order:
    unknown reference, private read, strict-format budget, duplicate label."""
    scenario = ma.Scenario(
        name="p", shared_materials="=====", max_tokens=2048,
        agents=[ma.Agent(id="a", name="A", base_model_id="m"),
                ma.Agent(id="b", name="B", base_model_id="m")],
        turns=[
            ma.Turn(id="t1", title="One", speaker_agent_id="a",
                    prompt_template="go", output_label="secret",
                    routing="speakerOnly"),
            ma.Turn(id="t2", title="Two", speaker_agent_id="b",
                    prompt_template="{{outputs.nope}} {{outputs.secret}}",
                    output_label="secret",
                    endpoint=ma.TurnEndpoint(name="v", kind="choice",
                                             marker="V:", vocabulary=("y",))),
        ])

    notes = ma.advisories(scenario)

    assert len(notes) == 5
    assert notes[0].startswith("shared materials contain the fence marker")
    assert "interpolates {{outputs.nope}}" in notes[1]
    assert "private turn is leaking" in notes[2]
    assert "invites format contamination" in notes[3]
    assert "reuses the output label 'secret'" in notes[4]


def test_repeated_reads_of_one_label_repeat_the_advisory():
    """No dedupe: two reads of a leaking private turn are two places to fix."""
    scenario = ma.Scenario(
        name="p",
        agents=[ma.Agent(id="a", name="A", base_model_id="m"),
                ma.Agent(id="b", name="B", base_model_id="m")],
        turns=[
            ma.Turn(id="t1", title="One", speaker_agent_id="a",
                    prompt_template="go", output_label="secret",
                    routing="speakerOnly"),
            ma.Turn(id="t2", title="Two", speaker_agent_id="b",
                    prompt_template="{{outputs.secret}} and {{outputs.secret}}",
                    output_label="two"),
        ])

    leaks = [n for n in ma.advisories(scenario) if "private turn is leaking" in n]
    assert len(leaks) == 2


def test_unknown_references_are_grouped_before_private_reads():
    """Two separate loops over the references, not one interleaved pass — so
    the author's ordering of the placeholders cannot reorder the advisories."""
    scenario = ma.Scenario(
        name="p",
        agents=[ma.Agent(id="a", name="A", base_model_id="m"),
                ma.Agent(id="b", name="B", base_model_id="m")],
        turns=[
            ma.Turn(id="t1", title="One", speaker_agent_id="a",
                    prompt_template="go", output_label="secret",
                    routing="speakerOnly"),
            # Private read written FIRST, unknown reference second.
            ma.Turn(id="t2", title="Two", speaker_agent_id="b",
                    prompt_template="{{outputs.secret}} {{outputs.nope}}",
                    output_label="two"),
        ])

    notes = ma.advisories(scenario)

    assert "interpolates {{outputs.nope}}" in notes[0]
    assert "private turn is leaking" in notes[1]


def test_a_dangling_contract_input_is_an_error_not_a_verbatim_advisory():
    """The template advisory says the placeholder ships to the model verbatim.
    A contract input does not ship anywhere, so that wording must not appear."""
    scenario = _contract_scenario(inputs=("nope",))

    assert not [n for n in ma.advisories(scenario) if "verbatim" in n]
    with pytest.raises(ma.ScenarioError):
        ma.validate(scenario)


# --- acknowledged cross-routing reads (spec §4.1) ---------------------------

SEATS = ("judge-1", "judge-2", "judge-3")


def _blind_round_scenario(acknowledge: bool) -> ma.Scenario:
    """The real prec-delib shape in miniature: a private round 1, then three
    rounds that read all three round-1 memos by name.

    3 seats × 3 later rounds × 2 COLLEAGUE memos = 18 private reads by design
    (a seat's own memo was routed to it, so it draws nothing). That is the
    number that made the advisory unreadable and is the reason the
    declaration exists."""
    turns = [ma.Turn(id=f"r1-{seat}", title=f"Round 1 — {seat}",
                     speaker_agent_id=seat, prompt_template="",
                     output_label=f"r1_{seat}", routing="speakerOnly",
                     contract=ma.TurnContract(task="Write your memo."))
             for seat in SEATS]
    for round_index in (2, 3, 4):
        for seat in SEATS:
            colleagues = [f"r1_{other}" for other in SEATS if other != seat]
            turns.append(ma.Turn(
                id=f"r{round_index}-{seat}", title=f"Round {round_index} — {seat}",
                speaker_agent_id=seat, prompt_template="",
                output_label=f"r{round_index}_{seat}", routing="all",
                contract=ma.TurnContract(
                    task="Respond.", inputs=tuple(f"r1_{s}" for s in SEATS)),
                acknowledged_inputs=tuple(colleagues) if acknowledge else ()))
    return ma.Scenario(
        name="p", shared_materials="M",
        agents=[ma.Agent(id=seat, name=seat.title(), base_model_id="m")
                for seat in SEATS],
        turns=turns)


def test_a_blind_round_design_is_18_advisories_until_it_declares_its_reads():
    """The whole point: a design that trips the advisory BY DESIGN can say so,
    and the list goes to zero without the advisory being weakened."""
    undeclared = ma.advisories(_blind_round_scenario(acknowledge=False))
    assert len(undeclared) == 18
    assert all("private turn is leaking" in n for n in undeclared)

    declared = _blind_round_scenario(acknowledge=True)
    assert ma.advisories(declared) == []
    ma.validate(declared)


def test_an_undeclared_private_read_still_fires_beside_the_acknowledged_ones():
    """Suppression is per LABEL, never per turn: declaring two reads must not
    buy silence for a third."""
    scenario = _blind_round_scenario(acknowledge=True)
    reader = scenario.turns[3]                       # round 2, judge-1
    # A fourth private memo nobody declared, read by that same turn.
    scenario.turns.insert(3, ma.Turn(
        id="r1-extra", title="Round 1 — extra", speaker_agent_id="judge-2",
        prompt_template="go", output_label="r1_extra", routing="speakerOnly"))
    reader.contract = ma.TurnContract(
        task="Respond.", inputs=reader.contract.inputs + ("r1_extra",))

    notes = ma.advisories(scenario)

    assert len(notes) == 1
    assert "reads the output 'r1_extra'" in notes[0]
    assert notes[0].endswith("check whether a private turn is leaking")


def _acknowledged_scenario(*, labels, producer_routing="speakerOnly",
                           inputs=("secret",)):
    return ma.Scenario(
        name="p",
        agents=[ma.Agent(id="a", name="A", base_model_id="m"),
                ma.Agent(id="b", name="B", base_model_id="m")],
        turns=[
            ma.Turn(id="t1", title="Private note", speaker_agent_id="a",
                    prompt_template="go", output_label="secret",
                    routing=producer_routing),
            ma.Turn(id="t2", title="Reader", speaker_agent_id="b",
                    prompt_template="", output_label="two",
                    contract=ma.TurnContract(task="T", inputs=tuple(inputs)),
                    acknowledged_inputs=tuple(labels)),
        ])


def test_an_acknowledgment_of_a_read_that_is_not_there_is_stale():
    """A silencer left behind after the read it silenced was edited away."""
    scenario = _acknowledged_scenario(labels=("secret", "gone"))

    assert ma.advisories(scenario) == [
        "turn 'Reader' acknowledges the read of 'gone' but does not read it "
        "— remove the stale acknowledgment"]


def test_an_acknowledgment_of_an_unknown_label_is_stale_too():
    """Read, but produced by nothing at all: no advisory could ever have fired
    for it, so the acknowledgment is the same kind of leftover. (The dangling
    input is separately a validate ERROR — this is about the advisory list.)"""
    scenario = _acknowledged_scenario(labels=("secret", "nowhere"),
                                      inputs=("secret", "nowhere"))

    assert ma.advisories(scenario) == [
        "turn 'Reader' acknowledges the read of 'nowhere' but does not read it "
        "— remove the stale acknowledgment"]


def test_an_acknowledgment_of_a_routed_read_is_doing_nothing():
    scenario = _acknowledged_scenario(labels=("secret",), producer_routing="all")

    assert ma.advisories(scenario) == [
        "turn 'Reader' acknowledges the read of 'secret', but its speaker was "
        "routed that output anyway — the acknowledgment is doing nothing"]


def test_acknowledgment_hygiene_sits_between_the_private_reads_and_the_budget():
    """Order is part of the cross-engine contract: unknown reference,
    unacknowledged private read, acknowledgment hygiene, strict-format budget,
    duplicate label."""
    scenario = ma.Scenario(
        name="p", max_tokens=2048,
        agents=[ma.Agent(id="a", name="A", base_model_id="m"),
                ma.Agent(id="b", name="B", base_model_id="m")],
        turns=[
            ma.Turn(id="t1", title="One", speaker_agent_id="a",
                    prompt_template="go", output_label="secret",
                    routing="speakerOnly"),
            ma.Turn(id="t2", title="Two", speaker_agent_id="b",
                    prompt_template="{{outputs.nope}} {{outputs.secret}}",
                    output_label="secret",
                    acknowledged_inputs=("stale",),
                    endpoint=ma.TurnEndpoint(name="v", kind="choice",
                                             marker="V:", vocabulary=("y",))),
        ])

    notes = ma.advisories(scenario)

    assert len(notes) == 5
    assert "interpolates {{outputs.nope}}" in notes[0]
    assert "private turn is leaking" in notes[1]
    assert "remove the stale acknowledgment" in notes[2]
    assert "invites format contamination" in notes[3]
    assert "reuses the output label 'secret'" in notes[4]


def test_acknowledgments_are_reported_in_declaration_order():
    scenario = _acknowledged_scenario(labels=("secret", "zulu", "alpha"))

    assert [n.split("'")[3] for n in ma.advisories(scenario)] == ["zulu", "alpha"]


def test_acknowledgments_round_trip_with_their_order_intact():
    payload = _contract_payload()
    payload["turns"][1]["acknowledgedInputs"] = ["one", "zero"]

    scenario = ma.Scenario.from_dict(payload)
    turn = ma._scenario_to_dict(scenario)["turns"][1]

    assert scenario.turns[1].acknowledged_inputs == ("one", "zero")
    assert turn["acknowledgedInputs"] == ["one", "zero"]
    # Advisory-only: the key does not make a scenario schema 2 on its own.
    assert ma._scenario_to_dict(scenario)["schemaVersion"] == 2
    plain = _contract_payload()
    del plain["turns"][1]["contract"]
    plain["turns"][1]["promptTemplate"] = "go"
    plain["turns"][1]["acknowledgedInputs"] = ["one"]
    assert ma._scenario_to_dict(ma.Scenario.from_dict(plain))["schemaVersion"] == 1


def test_an_acknowledgment_free_turn_emits_no_key():
    """The compatibility floor again: an empty declaration is an ABSENT key,
    so every panel written before this one existed keeps its pinned hash."""
    for value in (None, []):
        payload = _contract_payload()
        if value is not None:
            payload["turns"][1]["acknowledgedInputs"] = value
        written = json.dumps(ma._scenario_to_dict(ma.Scenario.from_dict(payload)),
                             indent=2, sort_keys=True)
        assert "acknowledgedInputs" not in written


def test_a_non_array_acknowledgment_refuses_at_decode():
    """Leniency is about MISSING and EMPTY, never about the wrong shape —
    the same floor the contract's `inputs` has."""
    for bad in ("one", {"labels": ["one"]}, 3):
        payload = _contract_payload()
        payload["turns"][1]["acknowledgedInputs"] = bad
        with pytest.raises(ma.ScenarioError,
                           match="acknowledgedInputs must be an array"):
            ma.Scenario.from_dict(payload)


def test_non_string_elements_refuse_at_decode_like_swift():
    """Element type is part of the shape. Swift's `[String]` decode refuses a
    non-string element; `str(x)` coercion here would let the same file run on
    one engine and refuse on the other."""
    payload = _contract_payload()
    payload["turns"][1]["acknowledgedInputs"] = ["ok", 3]
    with pytest.raises(ma.ScenarioError,
                       match="acknowledgedInputs must be an array"):
        ma.Scenario.from_dict(payload)
    payload = _contract_payload()
    payload["turns"][1]["contract"]["inputs"] = ["ok", None]
    with pytest.raises(ma.ScenarioError,
                       match="contract inputs must be an array"):
        ma.Scenario.from_dict(payload)


# --- §3.1 reader-aware context entries --------------------------------------

def test_an_agents_own_output_is_marked_in_its_own_context():
    assert ma.context_entry("r1", "Round 1", "Ava", "TEXT", own_authored=True) == (
        "[r1] Round 1 — your own earlier output (Ava)\nTEXT")
    assert ma.context_entry("r1", "Round 1", "Ava", "TEXT", own_authored=False) == (
        "[r1] Round 1 — Ava\nTEXT")


def test_the_runner_marks_context_per_receiving_agent(tmp_path, monkeypatch):
    """The entry can no longer be computed once per turn: the same output
    renders one way in the speaker's own context and another in everyone
    else's."""
    outputs = iter(["A-SAID", "B-SAID", "x", "y"])
    monkeypatch.setattr(ma, "generate", lambda *a, **k: next(outputs))
    scenario = ma.Scenario(
        name="p",
        agents=[ma.Agent(id="a", name="A", base_model_id="m"),
                ma.Agent(id="b", name="B", base_model_id="m")],
        turns=[
            ma.Turn(id="t1", title="One", speaker_agent_id="a",
                    prompt_template="go", output_label="one", routing="all"),
            ma.Turn(id="t2", title="Two", speaker_agent_id="b",
                    prompt_template="go", output_label="two", routing="all"),
            ma.Turn(id="t3", title="Three", speaker_agent_id="a",
                    prompt_template="{{agent.context}}", output_label="three"),
            ma.Turn(id="t4", title="Four", speaker_agent_id="b",
                    prompt_template="{{agent.context}}", output_label="four"),
        ])

    ma.run_scenario(SimpleNamespace(model_id="m", revision="r"), scenario,
                    run_dir=str(tmp_path))

    records = [json.loads(line) for line in
               open(os.path.join(tmp_path, "turns.jsonl"), encoding="utf-8")]
    a_reads, b_reads = records[2]["prompt"], records[3]["prompt"]
    assert "[one] One — your own earlier output (A)" in a_reads
    assert "[two] Two — B" in a_reads
    assert "[one] One — A" in b_reads
    assert "[two] Two — your own earlier output (B)" in b_reads


def test_a_resumed_turn_marks_context_the_same_way(tmp_path, monkeypatch):
    """A replayed record must route into context exactly as a generated one
    would, marking included, or the resumed transcript diverges."""
    calls = []

    def die_on_second(model, prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 2:
            raise RuntimeError("walltime")
        return "A-SAID"

    scenario = ma.Scenario(
        name="p", agents=[ma.Agent(id="a", name="A", base_model_id="m")],
        turns=[ma.Turn(id="t1", title="One", speaker_agent_id="a",
                       prompt_template="go", output_label="one", routing="all"),
               ma.Turn(id="t2", title="Two", speaker_agent_id="a",
                       prompt_template="{{agent.context}}", output_label="two")])

    monkeypatch.setattr(ma, "generate", die_on_second)
    with pytest.raises(RuntimeError):
        ma.run_scenario(SimpleNamespace(model_id="m", revision="r"), scenario,
                        run_dir=str(tmp_path))

    calls.clear()
    monkeypatch.setattr(ma, "generate",
                        lambda m, p, **k: (calls.append(p), "out")[1])
    ma.run_scenario(SimpleNamespace(model_id="m", revision="r"), scenario,
                    run_dir=str(tmp_path))

    assert "[one] One — your own earlier output (A)" in calls[0]


# --- §3.2 fallback prepend --------------------------------------------------

def test_the_fallback_prepends_record_then_transcript_then_instruction():
    """The old order put the case record AFTER the task instruction — one of
    the two framework causes of the voice failures."""
    scenario = ma.Scenario(
        name="p", shared_materials="THE RECORD",
        agents=[ma.Agent(id="a", name="A", base_model_id="m")],
        turns=[ma.Turn(id="t1", title="One", speaker_agent_id="a",
                       prompt_template="INSTRUCTION")])

    prompt = ma._render_prompt(scenario, scenario.turns[0], "PRIOR", "A", {})

    assert prompt == ("Shared scenario materials:\nTHE RECORD\n\n"
                      "Visible prior context:\nPRIOR\n\n"
                      "INSTRUCTION")


def test_a_template_positioning_its_own_slots_is_untouched():
    scenario = ma.Scenario(
        name="p", shared_materials="THE RECORD",
        agents=[ma.Agent(id="a", name="A", base_model_id="m")],
        turns=[ma.Turn(id="t1", title="One", speaker_agent_id="a",
                       prompt_template="{{scenario.materials}}|X|{{agent.context}}")])

    assert ma._render_prompt(scenario, scenario.turns[0], "PRIOR", "A", {}) == (
        "THE RECORD|X|PRIOR")


def test_each_fallback_section_still_needs_its_own_condition():
    """Placeholder absent AND content non-empty, exactly as before — only the
    placement moved."""
    scenario = ma.Scenario(
        name="p", shared_materials="THE RECORD",
        agents=[ma.Agent(id="a", name="A", base_model_id="m")],
        turns=[ma.Turn(id="t1", title="One", speaker_agent_id="a",
                       prompt_template="INSTRUCTION",
                       include_scenario_materials=False)])

    assert ma._render_prompt(scenario, scenario.turns[0], "", "A", {}) == "INSTRUCTION"
    assert ma._render_prompt(scenario, scenario.turns[0], "PRIOR", "A", {}) == (
        "Visible prior context:\nPRIOR\n\nINSTRUCTION")


# --- §2 renderer edges ------------------------------------------------------

def _render(scenario, turn_index=1, context="", outputs=None):
    turn = scenario.turns[turn_index]
    name = ma._agent_name(scenario, turn.speaker_agent_id)
    return ma._render_prompt(scenario, turn, context, name, outputs or {})


def test_a_solo_scenario_omits_the_colleagues_sentence_and_the_voice_block():
    scenario = ma.Scenario(
        name="p",
        agents=[ma.Agent(id="a", name="A", base_model_id="m",
                         role="a reviewer")],
        turns=[ma.Turn(id="t1", title="One", speaker_agent_id="a",
                       prompt_template="", output_label="one",
                       contract=ma.TurnContract(task="Do it."))])

    prompt = _render(scenario, turn_index=0)

    assert prompt.startswith("You are A, a reviewer.\n\n")
    assert "other participants" not in prompt
    assert "on behalf of" not in prompt


@pytest.mark.parametrize("names,expected_and,expected_or", [
    (["B"], "B", "B"),
    (["B", "C"], "B and C", "B or C"),
    (["B", "C", "D"], "B, C, and D", "B, C, or D"),
])
def test_colleague_lists_take_the_oxford_comma(names, expected_and, expected_or):
    assert ma._join_names(names, "and") == expected_and
    assert ma._join_names(names, "or") == expected_or


def test_an_empty_stage_leaves_no_trailing_space():
    scenario = _contract_scenario(stage="")
    assert _render(scenario).splitlines()[0] == (
        "You are B. The other participants are A.")


def test_own_voice_off_drops_only_block_six():
    scenario = _contract_scenario(own_voice=False, format="FORMAT")
    prompt = _render(scenario)

    assert "Write only your own response" not in prompt
    assert "FORMAT" in prompt
    assert prompt.endswith("Reminder: you are B. Respond as B and as no one else.")


def test_the_finished_documents_sentence_needs_something_above_it():
    """It refers to material actually shown, so with neither other-authored
    inputs nor a transcript it would point at nothing."""
    bare = _render(_contract_scenario())
    assert "Their contributions above are finished documents" not in bare

    with_context = _render(_contract_scenario(), context="[one] One — A\nX")
    assert "Their contributions above are finished documents" in with_context


def test_an_own_only_input_list_draws_no_other_participants_glue():
    scenario = ma.Scenario(
        name="p", agents=[ma.Agent(id="a", name="A", base_model_id="m"),
                ma.Agent(id="b", name="B", base_model_id="m")],
        turns=[
            ma.Turn(id="t1", title="One", speaker_agent_id="a",
                    prompt_template="go", output_label="one"),
            ma.Turn(id="t2", title="Two", speaker_agent_id="a",
                    prompt_template="", output_label="two",
                    contract=ma.TurnContract(task="T", inputs=("one",))),
        ])

    prompt = _render(scenario, outputs={"one": "MINE"})

    assert "===== YOUR OWN EARLIER OUTPUT — One =====" in prompt
    assert "That was your own earlier output, written by you, A." in prompt
    assert "Those were the contributions of the other participants" not in prompt
    assert "Their contributions above are finished documents" not in prompt


def test_an_unresolved_input_renders_no_empty_fence():
    """The preflight floor and the drafting case. An empty fence would teach
    the model that empty documents are normal here."""
    scenario = _contract_scenario(inputs=("one",))

    prompt = _render(scenario, outputs={})

    assert "=====" in prompt  # the task fence is always there
    assert "OUTPUT OF" not in prompt
    assert "Those were the contributions" not in prompt


def test_materials_are_omitted_entirely_when_the_turn_switches_them_off():
    scenario = _contract_scenario()
    scenario.turns[1].include_scenario_materials = False

    prompt = _render(scenario)

    assert "SHARED MATERIALS" not in prompt
    assert "That was the shared material" not in prompt


def test_no_rendered_prompt_ends_in_whitespace():
    scenario = _contract_scenario(format="FORMAT")
    prompt = _render(scenario, context="X")
    assert prompt == prompt.rstrip()
    assert "\n\n\n" not in prompt


# --- §3.3 provenance --------------------------------------------------------

def test_every_turn_record_stamps_its_renderer(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "generate", lambda *a, **k: "out")
    ma.run_scenario(SimpleNamespace(model_id="m", revision="r"),
                    _contract_scenario(), run_dir=str(tmp_path))

    records = [json.loads(line) for line in
               open(os.path.join(tmp_path, "turns.jsonl"), encoding="utf-8")]

    assert [r["promptRenderer"] for r in records] == [
        ma.TEMPLATE_RENDERER, ma.CONTRACT_RENDERER]


# --- preflight --------------------------------------------------------------

def _preflight(monkeypatch, scenario, window=100_000):
    monkeypatch.setattr(scenario_preflight.token_preflight, "_tokenizer",
                        lambda *a, **k: object())
    monkeypatch.setattr(scenario_preflight.token_preflight, "context_window",
                        lambda *a, **k: window)
    seen = []

    def fake_render(tokenizer, text, **kwargs):
        seen.append(text)
        return type("R", (), {"prompt_token_count": len(text.split())})()

    monkeypatch.setattr(scenario_preflight.prompt_render, "render", fake_render)
    return scenario_preflight.preflight(scenario, model_id="m"), seen


def test_the_preflight_floor_renders_a_contract_turn_for_real(monkeypatch):
    """Empty context, empty outputs — the floor is the SAME renderer, not an
    approximation, and it must not throw on that input."""
    report, rendered = _preflight(monkeypatch, _fixture_scenario())

    assert report["turnCount"] == 6
    assert not report["certainOverflow"]
    # The sandwich scaffold is in the floor, so a contract turn is not
    # under-reported as an empty template would be.
    assert any("===== YOUR TASK =====" in text for text in rendered)
    assert all(text for text in rendered)


def test_contract_inputs_are_charged_to_the_projection(monkeypatch):
    """`inputs` are read exactly the way `{{outputs.X}}` is — straight into the
    prompt, no routing involved — so they must be charged identically."""
    scenario = ma.Scenario(
        name="p", max_tokens=500,
        agents=[ma.Agent(id="a", name="A", base_model_id="m"),
                ma.Agent(id="b", name="B", base_model_id="m")],
        turns=[
            ma.Turn(id="t1", title="One", speaker_agent_id="a",
                    prompt_template="go", output_label="one", routing="none"),
            ma.Turn(id="t2", title="Two", speaker_agent_id="b",
                    prompt_template="", output_label="two",
                    include_speaker_context=False,
                    contract=ma.TurnContract(task="T", inputs=("one",))),
        ])

    report, _ = _preflight(monkeypatch, scenario)
    reader = report["turns"][1]

    assert reader["projectedPromptTokens"] == reader["floorTokens"] + 500
