"""Declared turn endpoints (Wave-2 contract, 2026-08-05).

A panel turn may declare the quantity it is supposed to produce; the runner
parses the generated text at write time and stamps the parse onto the turn
record, and analyze aggregates the stamps. The parse itself is a literal,
case-insensitive marker-then-80-characters scan with NO regex, byte-twinned
with `Sources/ExperimentKit/MultiAgent/TurnEndpoint.swift` — the goldens below
are the committed fixture BOTH engines read
(`prompts/fixtures/panel-endpoints/`), so a divergence fails on both sides.

The refusals matter as much as the parses: the scenario is pinned, reviewed
data, and a typo'd declaration that silently parsed nothing would be
indistinguishable from a panel that never answered.
"""

import csv
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import multi_agent, tasks, turn_endpoint
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.experiment.turn_endpoint import EndpointError, TurnEndpoint


def _fixture(*parts):
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "prompts", "fixtures",
                        "panel-endpoints", *parts)


def _fixture_endpoints():
    scenario, _, _ = multi_agent.read_scenario(_fixture("scenario.json"))
    return {turn.endpoint.name: turn.endpoint for turn in scenario.turns}


def _fixture_cases():
    with open(_fixture("cases.json"), encoding="utf-8") as handle:
        return json.load(handle)["cases"]


# --- the committed goldens --------------------------------------------------

def test_the_fixture_scenario_declares_both_kinds():
    endpoints = _fixture_endpoints()
    assert endpoints["vote"].kind == "choice"
    assert endpoints["vote"].vocabulary == (
        "affirm", "reverse", "vacate", "remand")
    assert endpoints["months"].kind == "number"
    assert (endpoints["months"].minimum, endpoints["months"].maximum) == (0.0, 600.0)


@pytest.mark.parametrize("case", _fixture_cases(),
                         ids=[c["name"] for c in _fixture_cases()])
def test_parser_matches_the_committed_goldens(case):
    endpoint = _fixture_endpoints()[case["endpoint"]]
    stamped = turn_endpoint.stamp(endpoint, case["text"])
    expected = case["expected"]
    if expected["unparsed"]:
        assert stamped == {"name": endpoint.name, "value": None,
                           "unparsed": True}, (
            "unparsed must be recorded as unparsed, never guessed")
    else:
        want = expected["choice"] if "choice" in expected else expected["number"]
        assert stamped == {"name": endpoint.name, "value": want}


def test_choice_value_is_the_declared_member_not_the_matched_text():
    """Counts must not fragment across casings."""
    endpoint = _fixture_endpoints()["vote"]
    assert turn_endpoint.parse(endpoint, "Vote: AFFIRM") == "affirm"


def test_whole_word_boundary_is_letter_adjacency_only():
    endpoint = _fixture_endpoints()["vote"]
    # Punctuation and digits are not letters, so these are whole words.
    assert turn_endpoint.parse(endpoint, "Vote: (affirm)") == "affirm"
    assert turn_endpoint.parse(endpoint, "Vote: affirm/reverse") == "affirm"
    # Letters on either side are not.
    assert turn_endpoint.parse(endpoint, "Vote: xaffirm") is None
    assert turn_endpoint.parse(endpoint, "Vote: affirmx") is None


def test_the_window_is_exactly_eighty_characters():
    endpoint = _fixture_endpoints()["vote"]
    # "Vote:" ends at index 5; the window is [5, 85). "affirm" must END by 85.
    fits = "Vote:" + " " * (85 - 5 - len("affirm")) + "affirm"
    assert turn_endpoint.parse(endpoint, fits) == "affirm"
    assert turn_endpoint.parse(endpoint, fits.replace("Vote:", "Vote: ", 1)) is None


def test_inclusive_bounds():
    endpoint = _fixture_endpoints()["months"]
    assert turn_endpoint.parse(endpoint, "Sentence: 0") == 0.0
    assert turn_endpoint.parse(endpoint, "Sentence: 600") == 600.0
    assert turn_endpoint.parse(endpoint, "Sentence: 600.5") is None


def test_an_out_of_range_first_number_does_not_fall_through_to_a_second():
    """The FIRST number is the answer; a scan that kept looking would report
    the model's most agreeable number rather than its actual one."""
    endpoint = _fixture_endpoints()["months"]
    assert turn_endpoint.parse(endpoint, "Sentence: 900, or 24 at the least") is None


# --- declaration validation -------------------------------------------------

@pytest.mark.parametrize("declaration, fragment", [
    ({"name": "v", "kind": "ordinal", "marker": "V:"}, "unknown kind"),
    ({"name": "v", "kind": "choice", "marker": "  ",
      "vocabulary": ["a"]}, "non-empty marker"),
    ({"name": "", "kind": "choice", "marker": "V:",
      "vocabulary": ["a"]}, "needs a name"),
    ({"name": "v", "kind": "choice", "marker": "V:"}, "non-empty vocabulary"),
    ({"name": "v", "kind": "choice", "marker": "V:",
      "vocabulary": []}, "non-empty vocabulary"),
    ({"name": "v", "kind": "choice", "marker": "V:",
      "vocabulary": ["a", " "]}, "empty vocabulary member"),
    ({"name": "v", "kind": "choice", "marker": "V:", "vocabulary": ["a"],
      "min": 0}, "declares min/max"),
    ({"name": "n", "kind": "number", "marker": "N:",
      "vocabulary": ["a"]}, "declares a vocabulary"),
    ({"name": "n", "kind": "number", "marker": "N:",
      "min": "x"}, "non-numeric min"),
    ({"name": "n", "kind": "number", "marker": "N:",
      "min": 10, "max": 1}, "min > max"),
])
def test_a_malformed_declaration_refuses_loudly(declaration, fragment):
    with pytest.raises(EndpointError) as excinfo:
        TurnEndpoint.from_dict(declaration, turn="T")
    assert fragment in str(excinfo.value), str(excinfo.value)


def test_a_malformed_declaration_refuses_the_whole_scenario():
    """Loading is where it must fail: the scenario is pinned data, and a
    declaration that parses nothing must never reach a measured run."""
    with pytest.raises(multi_agent.ScenarioError) as excinfo:
        multi_agent.Scenario.from_dict({
            "name": "p", "agents": [{"id": "a", "name": "A"}],
            "turns": [{"id": "t", "title": "T", "speakerAgentID": "a",
                       "promptTemplate": "go",
                       "endpoint": {"name": "vote", "kind": "chioce",
                                    "marker": "Vote:",
                                    "vocabulary": ["affirm"]}}]})
    assert "unknown kind" in str(excinfo.value)


def test_validate_re_checks_a_programmatically_built_declaration():
    scenario = multi_agent.Scenario(
        name="p", agents=[multi_agent.Agent(id="a", name="A")],
        turns=[multi_agent.Turn(
            id="t", title="T", speaker_agent_id="a", prompt_template="go",
            endpoint=TurnEndpoint(name="v", kind="choice", marker="V:",
                                  vocabulary=()))])
    with pytest.raises(multi_agent.ScenarioError):
        multi_agent.validate(scenario)


def test_unknown_keys_are_tolerated_and_the_declaration_survives_a_round_trip():
    source = {
        "schemaVersion": 1, "name": "p", "baseModelID": "m",
        "createdAt": "2026-01-01T00:00:00Z",      # a key this engine dropped
        "agents": [{"id": "a", "name": "A", "unknownSeatKey": 1}],
        "turns": [{"id": "t", "title": "T", "speakerAgentID": "a",
                   "promptTemplate": "go", "unknownTurnKey": True,
                   "endpoint": {"name": "vote", "kind": "choice",
                                "marker": "Vote:",
                                "vocabulary": ["affirm", "reverse"]}}]}
    scenario = multi_agent.Scenario.from_dict(source)
    assert scenario.turns[0].endpoint.marker == "Vote:"
    round_tripped = multi_agent._scenario_to_dict(scenario)
    assert round_tripped["turns"][0]["endpoint"] == {
        "name": "vote", "kind": "choice", "marker": "Vote:",
        "vocabulary": ["affirm", "reverse"]}


def test_a_turn_without_a_declaration_emits_no_endpoint_key():
    """Additive and optional: existing panels serialise byte for byte."""
    scenario = multi_agent.Scenario.from_dict({
        "name": "p", "agents": [{"id": "a", "name": "A"}],
        "turns": [{"id": "t", "title": "T", "speakerAgentID": "a",
                   "promptTemplate": "go"}]})
    assert "endpoint" not in multi_agent._scenario_to_dict(scenario)["turns"][0]


# --- the runner stamp -------------------------------------------------------

VOTE_ENDPOINT = {"name": "vote", "kind": "choice", "marker": "Vote:",
                 "vocabulary": ["affirm", "reverse"]}


def _panel_workspace(tmp_path, monkeypatch, *, declare=True, outputs=None):
    """A minimal panel workspace whose two turns declare a vote endpoint, with
    generation stubbed to canned text (same fixture shape as
    ``test_scenario_snapshot``)."""
    root = tmp_path / "ws"
    for sub in ("prompts/panels", "experiments", "runs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    turns = []
    for i in range(2):
        turn = {"id": f"t{i}", "title": f"Turn {i}", "speakerAgentID": "a",
                "promptTemplate": "go", "outputLabel": f"o{i}",
                "routing": "all", "routedAgentIDs": [],
                "includeScenarioMaterials": True,
                "includeSpeakerContext": True, "maxTokens": None}
        if declare:
            turn["endpoint"] = VOTE_ENDPOINT
        turns.append(turn)
    panel = {"schemaVersion": 1, "name": "panel", "baseModelID": "m",
             "description": "", "sharedMaterials": "", "temperature": 0.0,
             "maxTokens": 32,
             "agents": [{"id": "a", "name": "Judge A", "baseModelID": "m",
                         "systemPrompt": ""}],
             "turns": turns}
    scenario_path = root / "prompts/panels/panel.json"
    scenario_path.write_bytes(json.dumps(panel, indent=2).encode("utf-8"))
    spec = {"name": "panel", "modelID": "m", "studyKind": "multiAgent",
            "multiAgentScenarioPath": "prompts/panels/panel.json",
            "multiAgentScenarioHash": hashlib.sha256(
                scenario_path.read_bytes()).hexdigest(),
            "multiAgentIncludeBaseline": True, "samplesPerItem": 1,
            "temperature": 0.0, "seeds": [0]}
    (root / "experiments/panel.json").write_text(json.dumps(spec))

    canned = list(outputs or ["Vote: affirm — the judgment stands.",
                              "I never labelled my line this time."])
    calls = {"n": 0}

    def _generate(*a, **k):
        text = canned[calls["n"] % len(canned)]
        calls["n"] += 1
        return text

    monkeypatch.setattr(multi_agent, "generate", _generate)
    monkeypatch.setattr(tasks, "_advise_cross_substrate", lambda *a, **k: None)
    return root, Manifest.from_dict(spec)


def _turn_records(run_dir, condition="configured"):
    path = os.path.join(run_dir, condition, "turns.jsonl")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_the_runner_stamps_each_declared_turn_at_write_time(
        tmp_path, monkeypatch):
    root, manifest = _panel_workspace(tmp_path, monkeypatch)
    model = SimpleNamespace(model_id="m", revision="r", device="cpu")

    run_dir = tasks._run_multi_agent_study(
        "panel", manifest, model, str(root), log=lambda *_: None)

    turns = _turn_records(run_dir)
    assert [t["endpoint"] for t in turns] == [
        {"name": "vote", "value": "affirm"},
        # Recorded as unparsed, never guessed…
        {"name": "vote", "value": None, "unparsed": True}]
    # …and the full text stays on the record either way.
    assert turns[1]["output"] == "I never labelled my line this time."
    # The stamp reaches the flattened study records analyze reads.
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    assert {json.dumps(r["endpoint"], sort_keys=True) for r in records} == {
        json.dumps({"name": "vote", "value": "affirm"}, sort_keys=True),
        json.dumps({"name": "vote", "value": None, "unparsed": True},
                   sort_keys=True)}


def test_a_turn_declaring_nothing_carries_no_endpoint_key(
        tmp_path, monkeypatch):
    """Existing runs stay byte-identical."""
    root, manifest = _panel_workspace(tmp_path, monkeypatch, declare=False)
    model = SimpleNamespace(model_id="m", revision="r", device="cpu")

    run_dir = tasks._run_multi_agent_study(
        "panel", manifest, model, str(root), log=lambda *_: None)

    assert all("endpoint" not in t for t in _turn_records(run_dir))
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as handle:
        assert all("endpoint" not in json.loads(line)
                   for line in handle if line.strip())


# --- analyze aggregation ----------------------------------------------------

def test_analyze_writes_panel_endpoints(tmp_path, monkeypatch):
    root, manifest = _panel_workspace(
        tmp_path, monkeypatch,
        outputs=["Vote: affirm — the judgment stands.",
                 "Vote: reverse, for the reasons given.",
                 "Vote: affirm — the judgment stands.",
                 "I never labelled my line this time."])
    model = SimpleNamespace(model_id="m", revision="r", device="cpu")
    run_dir = tasks._run_multi_agent_study(
        "panel", manifest, model, str(root), log=lambda *_: None)

    out = tasks.analyze("panel", str(root), source_run=run_dir,
                        log=lambda *_: None)

    with open(os.path.join(out, "panel-endpoints.csv"), encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == turn_endpoint.PANEL_ENDPOINTS_HEADER
    assert len(rows) == 5  # header + 2 turns x 2 conditions
    body = {(r[0], r[3], r[5], r[6]) for r in rows[1:]}
    assert ("configured", "Turn 0", "affirm", "0") in body
    assert ("configured", "Turn 1", "reverse", "0") in body
    # Unparsed rows carry an empty value and are flagged, not dropped.
    assert ("baseline", "Turn 1", "", "1") in body
    assert all(r[2] == "Judge A" for r in rows[1:])

    with open(os.path.join(out, "panel-endpoints.json"), encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["unparsed"] == 1
    seat = report["endpoints"]["vote"]["Judge A"]
    assert seat["configured"]["values"] == {"affirm": 1, "reverse": 1}
    assert seat["baseline"]["values"] == {"affirm": 1}
    assert seat["baseline"]["unparsed"] == 1


def test_analyze_invents_no_section_without_stamps(tmp_path, monkeypatch):
    root, manifest = _panel_workspace(tmp_path, monkeypatch, declare=False)
    model = SimpleNamespace(model_id="m", revision="r", device="cpu")
    run_dir = tasks._run_multi_agent_study(
        "panel", manifest, model, str(root), log=lambda *_: None)

    out = tasks.analyze("panel", str(root), source_run=run_dir,
                        log=lambda *_: None)

    assert not os.path.exists(os.path.join(out, "panel-endpoints.csv"))
    assert not os.path.exists(os.path.join(out, "panel-endpoints.json"))
