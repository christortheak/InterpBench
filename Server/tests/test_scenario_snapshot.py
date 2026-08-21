"""A multi-agent run directory must carry its scenario, not a pointer to it.

`experiment.json` pins `multiAgentScenarioPath` + `multiAgentScenarioHash`, but
the seat→variant attribution — which agent variant sat in which seat — lives in
the scenario's own `agents[]` (`variantArtifactPath`/`variantArtifactHash`). A
reader that sees only the run directory (the Results Explorer's bridge serves
`runs/` and nothing else) therefore could not say what ran. The snapshot is the
fix, and it is only evidence if the bytes are the pinned bytes.
"""

import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from steerlab_server.experiment import multi_agent, tasks
from steerlab_server.experiment.manifest import Manifest


def _panel_workspace(tmp_path, monkeypatch, *, pinned_hash=None):
    """A minimal but REAL panel workspace (same shape as the shard-merge
    fixture): two turns, one seat, generation stubbed."""
    root = tmp_path / "ws"
    for sub in ("prompts/panels", "experiments", "runs/model-variants"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    # A real (un-steered) seat artifact, so the attribution the snapshot
    # preserves points at something that exists.
    variant_path = root / "runs/model-variants/seat-a.json"
    variant_path.write_text(json.dumps(
        {"name": "seat-a", "baseModelID": "m", "injections": [],
         "adapters": [], "promptMode": "chatAssistant"}))
    variant_hash = hashlib.sha256(variant_path.read_bytes()).hexdigest()
    panel = {
        "schemaVersion": 1, "name": "panel", "baseModelID": "m",
        "description": "", "sharedMaterials": "rules",
        "temperature": 0.0, "maxTokens": 32,
        # The seat carries a variant: this is the attribution the snapshot
        # exists to preserve.
        "agents": [{"id": "a", "name": "A", "baseModelID": "m",
                    "systemPrompt": "",
                    "variantArtifactPath": str(variant_path),
                    "variantArtifactHash": variant_hash}],
        "turns": [{"id": f"t{i}", "title": f"Turn {i}", "speakerAgentID": "a",
                   "promptTemplate": "go", "outputLabel": f"o{i}",
                   "routing": "all", "routedAgentIDs": [],
                   "includeScenarioMaterials": True,
                   "includeSpeakerContext": True, "maxTokens": None}
                  for i in range(2)],
    }
    scenario_path = root / "prompts/panels/panel.json"
    # Deliberately NOT canonical JSON (spacing + key order are the author's):
    # a verbatim snapshot must preserve bytes a re-serialisation would lose.
    scenario_path.write_bytes(json.dumps(panel, indent=2).encode("utf-8"))
    spec = {
        "name": "panel", "modelID": "m", "studyKind": "multiAgent",
        "multiAgentScenarioPath": "prompts/panels/panel.json",
        "multiAgentIncludeBaseline": False, "samplesPerItem": 1,
        "temperature": 0.0, "seeds": [0]}
    if pinned_hash is not None:
        spec["multiAgentScenarioHash"] = pinned_hash
    (root / "experiments/panel.json").write_text(json.dumps(spec))
    monkeypatch.setattr(multi_agent, "generate", lambda *a, **k: "out")
    monkeypatch.setattr(tasks, "_advise_cross_substrate", lambda *a, **k: None)
    return root, Manifest.from_dict(spec), scenario_path


def test_run_directory_holds_the_scenario_byte_for_byte(tmp_path, monkeypatch):
    root, manifest, scenario_path = _panel_workspace(tmp_path, monkeypatch)
    source = scenario_path.read_bytes()
    model = SimpleNamespace(model_id="m", revision="r", device="cpu")

    run_dir = tasks._run_multi_agent_study(
        "panel", manifest, model, str(root), log=lambda *_: None)

    snapshot = os.path.join(run_dir, "scenario.json")
    assert os.path.isfile(snapshot), sorted(os.listdir(run_dir))
    assert open(snapshot, "rb").read() == source
    # The point of the snapshot: seat→variant attribution answerable from the
    # run directory alone.
    seat = json.load(open(snapshot))["agents"][0]
    assert seat["variantArtifactPath"].endswith("runs/model-variants/seat-a.json")
    assert seat["variantArtifactHash"]
    # …and it still hashes to what a manifest would pin.
    assert (hashlib.sha256(open(snapshot, "rb").read()).hexdigest()
            == hashlib.sha256(source).hexdigest())


def test_run_directory_holds_the_scenario_the_manifest_pinned(
        tmp_path, monkeypatch):
    """Pin matches: the run proceeds and the snapshot is the pinned bytes."""
    root, manifest, scenario_path = _panel_workspace(tmp_path, monkeypatch)
    digest = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
    manifest.multi_agent_scenario_hash = digest
    model = SimpleNamespace(model_id="m", revision="r", device="cpu")

    run_dir = tasks._run_multi_agent_study(
        "panel", manifest, model, str(root), log=lambda *_: None)

    assert (hashlib.sha256(
        open(os.path.join(run_dir, "scenario.json"), "rb").read()).hexdigest()
        == digest)


def test_a_drifted_scenario_refuses_before_any_generation(
        tmp_path, monkeypatch):
    """Drift in a pinned input is a violation, never a silent copy — and the
    refusal must land before a run directory or a single turn exists."""
    stale = "a" * 64
    root, manifest, scenario_path = _panel_workspace(
        tmp_path, monkeypatch, pinned_hash=stale)
    live = hashlib.sha256(scenario_path.read_bytes()).hexdigest()

    def _no_generation(*a, **k):  # pragma: no cover - must never be reached
        raise AssertionError("generation started despite scenario drift")

    monkeypatch.setattr(multi_agent, "generate", _no_generation)
    model = SimpleNamespace(model_id="m", revision="r", device="cpu")

    with pytest.raises(RuntimeError) as excinfo:
        tasks._run_multi_agent_study(
            "panel", manifest, model, str(root), log=lambda *_: None)

    message = str(excinfo.value)
    assert live[:12] in message and stale[:12] in message, message
    assert "changed since pinning" in message
    # No run directory was created: no half-run to explain away.
    assert [e for e in os.listdir(root / "runs") if e.startswith("2")] == []
