"""The flattened panel record names its seat by ID.

``generations.jsonl`` used to carry ``speakerName`` only. A name is a label:
two seats may share one, and a rename changes it, so a turn could be
attributed to its seat only by joining ``turns.jsonl`` — and a reader that
sees only the root record could not attribute it at all.

``speakerAgentID`` is additive. A record written before it existed has no
such key, and everything that reads records keeps reading them.

Swift twin: ``Tests/ExperimentKitTests/PanelRecordSeatTests.swift``.
"""

import json
import os
from types import SimpleNamespace

from steerlab_server.experiment import (analysis_endpoints, multi_agent,
                                        panel_workflow, resume, voice_lint)
from steerlab_server.experiment.manifest import Manifest


def _scenario():
    """Two seats that deliberately share a DISPLAY name: the ID is the only
    thing that tells them apart, which is the case the record has to survive."""
    return multi_agent.Scenario(
        name="panel", base_model_id="m",
        agents=[multi_agent.Agent(id="seat-a", name="Reviewer", base_model_id="m"),
                multi_agent.Agent(id="seat-b", name="Reviewer", base_model_id="m")],
        turns=[
            multi_agent.Turn(id="t1", title="First", speaker_agent_id="seat-a",
                             prompt_template="Open.", output_label="one"),
            multi_agent.Turn(id="t2", title="Second", speaker_agent_id="seat-b",
                             prompt_template="Reply.", output_label="two"),
        ])


def _manifest_stub():
    return SimpleNamespace(content_hash=lambda: "h", model_id="m",
                           temperature=0.0)


def test_the_flattened_record_names_its_seat_by_id(tmp_path, monkeypatch):
    monkeypatch.setattr(multi_agent, "generate", lambda *a, **k: "out")
    multi_agent.run_scenario(SimpleNamespace(model_id="m", revision="r"),
                             _scenario(), run_dir=str(tmp_path))

    records = panel_workflow._panel_records_from(
        str(tmp_path), "exp", _manifest_stub(), None, "configured", 0)

    # The names cannot tell these two turns apart; the IDs can.
    assert [r["speakerName"] for r in records] == ["Reviewer", "Reviewer"]
    assert [r["speakerAgentID"] for r in records] == ["seat-a", "seat-b"]
    # Carried verbatim from the turn record, not re-derived.
    turns = [json.loads(line) for line in open(tmp_path / "turns.jsonl")]
    assert [t["speakerAgentID"] for t in turns] == ["seat-a", "seat-b"]


def test_the_root_generations_file_carries_the_seat(tmp_path, monkeypatch):
    """Feature level, not component level: the study runner writes the key
    into the run's own ``generations.jsonl``, for both arms."""
    import steerlab_server.experiment.execution_reporting as reporting
    import steerlab_server.experiment.run_artifacts as run_artifacts

    root = tmp_path / "ws"
    for sub in ("prompts/panels", "experiments", "runs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    panel = multi_agent._scenario_to_dict(_scenario())
    (root / "prompts/panels/panel.json").write_text(json.dumps(panel))
    spec = {"name": "panel", "modelID": "m", "studyKind": "multiAgent",
            "multiAgentScenarioPath": "prompts/panels/panel.json",
            "multiAgentIncludeBaseline": True, "samplesPerItem": 1,
            "temperature": 0.0, "seeds": [0]}
    (root / "experiments/panel.json").write_text(json.dumps(spec))
    monkeypatch.setattr(multi_agent, "generate", lambda *a, **k: "out")
    monkeypatch.setattr(reporting, "advise_cross_substrate", lambda *a, **k: None)
    monkeypatch.setattr(run_artifacts, "write_config_snapshot", lambda *a, **k: None)

    run_dir = panel_workflow.run_multi_agent_study(
        "panel", Manifest.from_dict(spec),
        SimpleNamespace(model_id="m", revision="r", device="cpu"), str(root),
        log=lambda *_: None)

    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    assert len(records) == 4  # 2 turns x (configured + baseline)
    assert ([(r["condition"], r["promptID"], r["speakerAgentID"]) for r in records]
            == [("configured", "t1", "seat-a"), ("configured", "t2", "seat-b"),
                ("baseline", "t1", "seat-a"), ("baseline", "t2", "seat-b")])


def test_a_turn_record_without_a_seat_id_flattens_as_it_always_did(tmp_path):
    """Additive means additive: a turn record that carries no ID of its own
    gains no key — absent, never null, never a guess from the name."""
    with open(tmp_path / "turns.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"turnID": "t1", "turnIndex": 1,
                                 "title": "First", "speakerName": "Reviewer",
                                 "output": "old"}) + "\n")
        handle.write(json.dumps({"turnID": "t2", "turnIndex": 2,
                                 "title": "Second", "speakerName": "Reviewer",
                                 "speakerAgentID": None,
                                 "output": "null is not an ID"}) + "\n")

    records = panel_workflow._panel_records_from(
        str(tmp_path), "exp", _manifest_stub(), SimpleNamespace(revision="r"),
        "configured", 0)

    assert [r["promptID"] for r in records] == ["t1", "t2"]
    assert all("speakerAgentID" not in r for r in records)
    assert [r["speakerName"] for r in records] == ["Reviewer", "Reviewer"]


def test_records_written_before_the_key_existed_still_load():
    """Every reader of flattened records keeps working on the old shape, and
    on a file that mixes the two (a resumed or merged run)."""
    old = {"condition": "configured", "promptID": "t1", "promptIndex": 0,
           "sampleIndex": 0, "replicateIndex": 0, "speakerName": "Reviewer",
           "turnTitle": "First", "routedAgentIDs": ["seat-a"], "output": "x",
           "voiceLint": {"version": 1, "speaksForOthers": False,
                         "thirdPersonSelf": 0}}
    new = dict(old, promptID="t2", promptIndex=1, speakerAgentID="seat-b")

    # Merge/resume identity never read the speaker at all.
    assert resume.record_key(old) != resume.record_key(new)
    # Transcript keying for the clustered estimator.
    keyed = analysis_endpoints.key_records_by_transcript([old, new])
    assert [r["promptID"] for r in keyed] == ["t1@0", "t2@0"]
    # The voice-lint roll-up, which groups by speaker.
    rows = voice_lint.csv_rows([old, new])
    assert [row[:3] for row in rows] == [["configured", "Reviewer", 2]]
