"""Two seats, or two turns, of a panel may not share an ID.

A seat's ID (an entry in the panel's ``agents`` list) is what turns, routing,
and records name it by. A turn's ID keys turn-level resume and the per-turn
seed. A repeat fails SILENTLY — the later seat wins the runtime slot; a resumed
transcript replays the first turn's recorded output in place of the second and
never generates it — so it is refused, with the ID named and a repair, wherever
a panel is authored, checked, cast, frozen into a study, or about to run.

It is never refused on DECODE, and never by ``verify()``, so a run directory
that already exists still reads and a study frozen before the rule existed
still answers for its runs.

The two refusal strings are a cross-engine contract. Swift twin:
``Tests/ExperimentKitTests/PanelIdentifierUniquenessTests.swift`` asserts the
same literals.
"""

import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from steerlab_server import client_cli
from steerlab_server.client import design_files, design_panels, study_panels
from steerlab_server.experiment import (experiment_store as es, lifecycle_gates,
                                        multi_agent, panel_workflow)
from steerlab_server.experiment.manifest import Manifest
from steerlab_server.experiment.manifest_errors import ExperimentStoreError


# The two refusals, spelled out in full. Identical on the Mac engine.
DUPLICATE_SEAT = (
    "seats 1 and 3 ('Alice' and 'Carol') share the ID 'a' — each seat needs "
    "its own ID, because turns, routing, and records refer to a seat by its "
    "ID; give one of them a different ID, and update any turn that should "
    "name it")
DUPLICATE_TURN = (
    "turns 1 and 3 ('Alice opens' and 'Alice closes') share the ID 't1' — "
    "each turn needs its own ID, because a run uses the turn ID to pick up "
    "where it stopped and to set each turn's random seed; give one of them a "
    "different ID")


def _scenario():
    return multi_agent.Scenario(
        name="panel", base_model_id="m",
        agents=[multi_agent.Agent(id="a", name="Alice", base_model_id="m"),
                multi_agent.Agent(id="b", name="Bob", base_model_id="m"),
                multi_agent.Agent(id="c", name="Carol", base_model_id="m")],
        turns=[
            multi_agent.Turn(id="t1", title="Alice opens", speaker_agent_id="a",
                             prompt_template="Open.", output_label="one"),
            multi_agent.Turn(id="t2", title="Bob replies", speaker_agent_id="b",
                             prompt_template="Reply.", output_label="two"),
            multi_agent.Turn(id="t3", title="Alice closes", speaker_agent_id="a",
                             prompt_template="Close.", output_label="three"),
        ])


# --- the engine validator -----------------------------------------------------

def test_a_clean_panel_has_no_identity_problem():
    scenario = _scenario()
    assert multi_agent.duplicate_identifier_problem(scenario) is None
    multi_agent.validate(scenario)


def test_a_repeated_seat_id_is_refused_by_name():
    """Parity with the Mac engine, which has always refused this. Here it used
    to run: the later agent silently took the seat."""
    scenario = _scenario()
    scenario.agents[2].id = "a"

    with pytest.raises(multi_agent.ScenarioError) as refusal:
        multi_agent.validate(scenario)

    assert str(refusal.value) == DUPLICATE_SEAT
    # Typed: a rule declined, with a repair — not an operational failure.
    assert lifecycle_gates.gate_of(refusal.value) == lifecycle_gates.MISSING_PREREQUISITE
    assert lifecycle_gates.repair_of(refusal.value) == multi_agent.DUPLICATE_ID_REPAIR
    assert "different ID" in multi_agent.DUPLICATE_ID_REPAIR
    assert "panel check" in multi_agent.DUPLICATE_ID_REPAIR


def test_a_repeated_turn_id_is_refused_by_name():
    scenario = _scenario()
    scenario.turns[2].id = "t1"

    with pytest.raises(multi_agent.ScenarioError) as refusal:
        multi_agent.validate(scenario)

    assert str(refusal.value) == DUPLICATE_TURN
    assert lifecycle_gates.gate_of(refusal.value) == lifecycle_gates.MISSING_PREREQUISITE
    assert lifecycle_gates.repair_of(refusal.value) == multi_agent.DUPLICATE_ID_REPAIR


def test_seats_are_named_before_turns_and_the_first_collision_wins():
    """Refusal ORDER is part of the cross-engine contract: a panel that trips
    two rules must name the same one on both engines."""
    scenario = _scenario()
    scenario.agents[2].id = "a"
    scenario.turns[1].id = "t1"
    scenario.turns[2].id = "t1"

    with pytest.raises(multi_agent.ScenarioError) as refusal:
        multi_agent.validate(scenario)
    assert str(refusal.value) == DUPLICATE_SEAT

    scenario.agents[2].id = "c"
    problem = multi_agent.duplicate_identifier_problem(scenario)
    assert str(problem).startswith(
        "turns 1 and 2 ('Alice opens' and 'Bob replies') share the ID 't1'")


def test_an_untyped_scenario_error_is_what_it_always_was():
    """The typed fields are additive: every other refusal here carries no gate
    and no repair, and renders the message it always did."""
    error = multi_agent.ScenarioError("scenario needs a name")
    assert str(error) == "scenario needs a name"
    assert lifecycle_gates.gate_of(error) is None
    assert lifecycle_gates.repair_of(error) == ""


def test_a_repeated_turn_id_refuses_before_anything_is_generated(tmp_path, monkeypatch):
    """The run path inherits the rule from ``validate``: nothing is generated
    and no transcript is started for a panel whose records could not be told
    apart afterwards."""
    calls = []
    monkeypatch.setattr(multi_agent, "generate",
                        lambda *a, **k: (calls.append(1), "out")[1])
    scenario = _scenario()
    scenario.turns[2].id = "t1"

    with pytest.raises(multi_agent.ScenarioError, match="share the ID 't1'"):
        multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"),
                                 scenario, run_dir=str(tmp_path))

    assert calls == []
    assert not os.path.exists(tmp_path / "turns.jsonl")
    assert not os.path.exists(tmp_path / "scenario.json")


# --- ...but never when an existing run is read --------------------------------

def test_a_panel_with_repeated_ids_still_decodes(tmp_path):
    """Decode is not validation. A run directory's ``scenario.json`` snapshot
    of a panel that ran before this rule existed must keep opening, and the
    readers that work from it must keep working."""
    from steerlab_server.experiment import panel_effects

    scenario = _scenario()
    scenario.agents[2].id = "a"
    scenario.turns[2].id = "t1"
    snapshot = tmp_path / "scenario.json"
    snapshot.write_text(json.dumps(multi_agent._scenario_to_dict(scenario)))

    loaded, digest, raw = multi_agent.read_scenario(str(snapshot))

    assert [a.id for a in loaded.agents] == ["a", "b", "a"]
    assert [t.id for t in loaded.turns] == ["t1", "t2", "t1"]
    assert digest == hashlib.sha256(raw).hexdigest()
    # Authoring advisories and the panel-effects exposure map read it too.
    multi_agent.advisories(loaded)
    panel_effects.exposure_by_turn(loaded, set())


def test_old_turn_records_under_a_repeated_id_still_flatten(tmp_path):
    """An existing transcript whose two turns share an ID flattens to two
    records, in file order — reading never applies the authoring rule."""
    with open(tmp_path / "turns.jsonl", "w", encoding="utf-8") as handle:
        for index, output in enumerate(("first", "second"), start=1):
            handle.write(json.dumps({
                "turnID": "t1", "turnIndex": index, "title": f"Turn {index}",
                "speakerAgentID": "a", "speakerName": "Alice",
                "output": output}) + "\n")

    records = panel_workflow._panel_records_from(
        str(tmp_path), "exp",
        SimpleNamespace(content_hash=lambda: "h", model_id="m", temperature=0.0),
        None, "configured", 0)

    assert [(r["promptID"], r["promptIndex"], r["output"]) for r in records] == [
        ("t1", 0, "first"), ("t1", 1, "second")]


# --- freeze -------------------------------------------------------------------

def _panel_draft(root, panel, name="mas"):
    """A multi-agent draft that pins ``panel`` by hash, in workspace ``root``."""
    path = os.path.join(root, "prompts", "panels", "panel.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = json.dumps(panel, indent=1).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(data)
    es.create(name, model_id="org/m", revision="abc", root=root)
    d = es.load_raw(name, root)
    d["studyKind"] = "multiAgent"
    d["multiAgentScenarioPath"] = "prompts/panels/panel.json"
    d["multiAgentScenarioHash"] = hashlib.sha256(data).hexdigest()
    es.save_raw(d, root)
    return name


def _panel_document():
    return {"name": "panel", "baseModelID": "org/m",
            "agents": [{"id": "j1", "name": "Judge One", "baseModelID": "org/m"},
                       {"id": "j2", "name": "Judge Two", "baseModelID": "org/m"}],
            "turns": [{"id": "t1", "title": "Vote", "speakerAgentID": "j1",
                       "promptTemplate": "Vote."},
                      {"id": "t2", "title": "Reply", "speakerAgentID": "j2",
                       "promptTemplate": "Reply."}]}


def test_freeze_accepts_a_panel_whose_ids_are_all_distinct(tmp_path):
    root = str(tmp_path)
    name = _panel_draft(root, _panel_document())
    assert es.freeze(name, force=False, root=root)["status"] == "frozen"


@pytest.mark.parametrize("force", [False, True])
def test_freeze_refuses_a_pinned_panel_with_a_repeated_turn_id(tmp_path, force):
    """Force included: record identity is the never-skippable class, like the
    pins — no ``forcedGatesSkipped`` stamp can make the records attributable."""
    root = str(tmp_path)
    panel = _panel_document()
    panel["turns"][1]["id"] = "t1"
    name = _panel_draft(root, panel)

    with pytest.raises(ExperimentStoreError) as refusal:
        es.freeze(name, force=force, root=root)

    assert str(refusal.value) == (
        "cannot freeze 'mas': in its pinned panel, turns 1 and 2 ('Vote' and "
        "'Reply') share the ID 't1' — each turn needs its own ID, because a "
        "run uses the turn ID to pick up where it stopped and to set each "
        "turn's random seed; give one of them a different ID")
    assert lifecycle_gates.gate_of(refusal.value) == lifecycle_gates.MISSING_PREREQUISITE
    assert lifecycle_gates.repair_of(refusal.value) == multi_agent.DUPLICATE_ID_REPAIR
    assert es.load_raw(name, root)["status"] == "draft"


def test_freeze_refuses_a_pinned_panel_with_a_repeated_seat_id(tmp_path):
    root = str(tmp_path)
    panel = _panel_document()
    panel["agents"][1]["id"] = "j1"
    panel["turns"][1]["speakerAgentID"] = "j1"
    name = _panel_draft(root, panel)

    with pytest.raises(ExperimentStoreError, match=(
            r"cannot freeze 'mas': in its pinned panel, seats 1 and 2 "
            r"\('Judge One' and 'Judge Two'\) share the ID 'j1'")):
        es.freeze(name, root=root)


def test_the_clients_freeze_verb_answers_the_refusal_with_its_repair(
        tmp_path, capsys, monkeypatch):
    """End to end through ``steerlab experiment freeze``: ``refused`` / 65 with
    the gate as the code — a rule declined, the instrument did not break."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # Set first so monkeypatch restores the environment the CLI exports into.
    monkeypatch.setenv("STEERLAB_ROOT", str(workspace))
    panel = _panel_document()
    panel["turns"][1]["id"] = "t1"
    name = _panel_draft(str(workspace), panel)

    exit_code = client_cli.main(
        ["experiment", "freeze", name, "--root", str(workspace), "--json"])
    envelope = json.loads(capsys.readouterr().out)

    assert exit_code == 65
    assert envelope["state"] == "refused"
    assert envelope["error"]["code"] == "missingPrerequisite"
    assert envelope["error"]["reason"].startswith(
        "cannot freeze 'mas': in its pinned panel, turns 1 and 2")
    assert envelope["error"]["repairAction"] == multi_agent.DUPLICATE_ID_REPAIR
    assert es.load_raw(name, str(workspace))["status"] == "draft"


def test_a_study_over_a_repeated_id_still_verifies(tmp_path):
    """The rule is asked at freeze, not in ``verify()``: verify also admits
    reads of runs that already exist (evaluate, analyze, bundle import), and a
    study frozen before the rule existed must keep answering them."""
    root = str(tmp_path)
    panel = _panel_document()
    panel["turns"][1]["id"] = "t1"
    name = _panel_draft(root, panel)

    assert Manifest.load(name, root=root).verify(root) == []
    assert multi_agent.pinned_panel_identity_problem(
        es.load_raw(name, root), root) is not None


def test_the_freeze_question_is_silent_about_everything_else(tmp_path):
    root = str(tmp_path)
    ask = multi_agent.pinned_panel_identity_problem
    # Not a panel study; a panel study that pins nothing; a path that is not
    # there; a file that is not a panel. verify() and the run report those.
    assert ask({"studyKind": "modelOutput"}, root) is None
    assert ask({"studyKind": "multiAgent"}, root) is None
    assert ask({"studyKind": "multiAgent",
                "multiAgentScenarioPath": "prompts/panels/absent.json"}, root) is None
    os.makedirs(tmp_path / "prompts" / "panels")
    (tmp_path / "prompts/panels/broken.json").write_text("[1, 2")
    assert ask({"studyKind": "multiAgent",
                "multiAgentScenarioPath": "prompts/panels/broken.json"}, root) is None
    (tmp_path / "prompts/panels/list.json").write_text("[]")
    assert ask({"studyKind": "multiAgent",
                "multiAgentScenarioPath": "prompts/panels/list.json"}, root) is None


# --- the Python client's authoring verbs --------------------------------------

def _semantic_panel():
    return {"schemaVersion": 1, "name": "shared-panel", "description": "roles",
            "baseModelID": "", "temperature": 0, "maxTokens": 2048,
            "sharedMaterials": "shared facts",
            "agents": [
                {"id": "first", "name": "First", "baseModelID": "",
                 "systemPrompt": "role one"},
                {"id": "second", "name": "Second", "baseModelID": "",
                 "systemPrompt": "role two"}],
            "turns": [
                {"id": "turn-one", "title": "First response",
                 "speakerAgentID": "first", "promptTemplate": "Read.",
                 "outputLabel": "first-output", "routing": "all",
                 "routedAgentIDs": [], "includeScenarioMaterials": True,
                 "includeSpeakerContext": True},
                {"id": "turn-two", "title": "Second response",
                 "speakerAgentID": "second", "promptTemplate": "Reply.",
                 "outputLabel": "second-output", "routing": "all",
                 "routedAgentIDs": [], "includeScenarioMaterials": True,
                 "includeSpeakerContext": True}]}


def test_the_client_accepts_a_clean_semantic_panel():
    assert study_panels.validate(_semantic_panel())["name"] == "shared-panel"


def test_the_client_refuses_a_repeated_turn_id_with_the_engines_repair():
    panel = _semantic_panel()
    panel["turns"][1]["id"] = "turn-one"

    with pytest.raises(ExperimentStoreError) as refusal:
        study_panels.validate(panel)

    assert str(refusal.value) == (
        "Invalid panel: turns 1 and 2 ('First response' and 'Second "
        "response') share the ID 'turn-one' — each turn needs its own ID, "
        "because a run uses the turn ID to pick up where it stopped and to "
        "set each turn's random seed; give one of them a different ID")
    assert refusal.value.gate == "missingPrerequisite"
    assert refusal.value.repair_action == multi_agent.DUPLICATE_ID_REPAIR


def test_the_client_names_a_repeated_seat_id():
    """This used to answer "A panel needs unique named seats." — true, and no
    help in finding the pair."""
    panel = _semantic_panel()
    panel["agents"][1]["id"] = "first"
    panel["turns"][1]["speakerAgentID"] = "first"

    with pytest.raises(ExperimentStoreError) as refusal:
        study_panels.validate(panel)

    assert str(refusal.value).startswith(
        "Invalid panel: seats 1 and 2 ('First' and 'Second') share the ID "
        "'first' — each seat needs its own ID")
    assert refusal.value.repair_action == multi_agent.DUPLICATE_ID_REPAIR


def test_a_rule_without_its_own_repair_keeps_the_shared_one():
    """Only the rules that know their repair change; the rest refuse exactly
    as they did."""
    panel = _semantic_panel()
    panel["turns"][0]["promptTemplate"] = ""

    with pytest.raises(ExperimentStoreError) as refusal:
        study_panels.validate(panel)

    assert str(refusal.value) == (
        "Invalid panel: turn 'First response' needs a prompt template")
    assert refusal.value.gate == "missingPrerequisite"
    assert refusal.value.repair_action.startswith("Inspect the study and inputs")


def test_loading_a_workspace_panel_names_the_repeated_seat(tmp_path):
    panel = _semantic_panel()
    panel["agents"][1]["id"] = "first"
    panel["turns"][1]["speakerAgentID"] = "first"
    data = design_files.encode(panel)
    path = tmp_path / "prompts" / "panels" / "hand-edited.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    ref = {"path": "prompts/panels/hand-edited.json",
           "hash": hashlib.sha256(data).hexdigest()}

    with pytest.raises(ExperimentStoreError) as refusal:
        design_panels.load(ref, tmp_path)

    assert str(refusal.value).startswith(
        "A panel cannot repeat a seat ID: seats 1 and 2 ('First' and "
        "'Second') share the ID 'first'")
    assert refusal.value.repair_action == multi_agent.DUPLICATE_ID_REPAIR


def test_a_workspace_panel_with_a_repeated_turn_id_can_still_be_inspected(tmp_path):
    """Inspecting and listing are reads. The turn rule belongs to check,
    import and compile, so the file that needs fixing can still be opened."""
    panel = _semantic_panel()
    panel["turns"][1]["id"] = "turn-one"
    data = design_files.encode(panel)
    path = tmp_path / "prompts" / "panels" / "hand-edited.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)

    review = study_panels.inspect("prompts/panels/hand-edited.json", root=tmp_path)

    assert [turn["id"] for turn in review["document"]["turns"]] == [
        "turn-one", "turn-one"]
    assert study_panels.catalog(root=tmp_path)["issues"] == []


def test_panel_check_answers_a_refusal_with_a_repair(tmp_path, capsys, monkeypatch):
    """End to end through the client's own CLI: ``refused`` / 65, the gate as
    the code, the ID in the reason, and a repair the reader can carry out."""
    panel = _semantic_panel()
    panel["turns"][1]["id"] = "turn-one"
    proposed = tmp_path / "proposed.json"
    proposed.write_bytes(design_files.encode(panel))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # The CLI exports the root it resolved into the environment. Setting it
    # here first is what makes monkeypatch put the environment back afterwards
    # (a variable it never touched is not restored), so this temporary
    # workspace cannot leak into whatever test runs next.
    monkeypatch.setenv("STEERLAB_ROOT", str(workspace))

    exit_code = client_cli.main(
        ["panel", "check", str(proposed), "--root", str(workspace), "--json"])
    envelope = json.loads(capsys.readouterr().out)

    assert exit_code == 65
    assert envelope["state"] == "refused"
    assert envelope["error"]["code"] == "missingPrerequisite"
    assert "share the ID 'turn-one'" in envelope["error"]["reason"]
    assert envelope["error"]["repairAction"] == multi_agent.DUPLICATE_ID_REPAIR
