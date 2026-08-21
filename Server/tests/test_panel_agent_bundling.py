"""A panel's SEAT artifacts must travel with the bundle.

Reported from the cluster: a study whose panel had a steered seat died with
FileNotFoundError for runs/model-variants/<...>/model-variant.json after the
queue wait and the model load. The scenario names its seats by path INSIDE the
JSON, and the pin-surface enumeration only ever added the script — the exact
gap the variantConditions branch had already been fixed for.
"""

import json
import os

import pytest

from steerlab_server.experiment import experiment_store


def _workspace(tmp_path, *, with_agent=True, agent_on_disk=True):
    root = tmp_path / "ws"
    (root / "prompts/panels").mkdir(parents=True)
    (root / "runs/model-variants/2026-07-27-neurotic").mkdir(parents=True)
    (root / "runs/vectors").mkdir(parents=True)

    if agent_on_disk:
        (root / "runs/vectors/neurotic.safetensors").write_text("tensor")
        (root / "runs/vectors/neurotic.json").write_text("{}")
        (root / "runs/model-variants/2026-07-27-neurotic/model-variant.json").write_text(
            json.dumps({
                "name": "neurotic", "baseModelID": "google/gemma-3-27b-it",
                "injections": [{"vectorArtifactID": "runs/vectors/neurotic",
                                "layer": 20, "alpha": 1.0}],
                "adapters": [], "promptMode": "chatAssistant"}))

    agent = {"id": "a", "name": "Judge A", "baseModelID": "google/gemma-3-4b-it",
             "systemPrompt": "", "variantArtifactHash": None}
    if with_agent:
        agent["variantArtifactPath"] = \
            "runs/model-variants/2026-07-27-neurotic/model-variant.json"
    else:
        agent["variantArtifactPath"] = None

    (root / "prompts/panels/panel.json").write_text(json.dumps({
        "schemaVersion": 1, "name": "panel", "baseModelID": "google/gemma-3-27b-it",
        "description": "", "sharedMaterials": "", "temperature": 0.0,
        "maxTokens": 64, "agents": [agent],
        "turns": [{"id": "t1", "title": "T", "speakerAgentID": "a",
                   "promptTemplate": "go", "outputLabel": "o", "routing": "all",
                   "routedAgentIDs": [], "includeScenarioMaterials": True,
                   "includeSpeakerContext": True, "maxTokens": None}]}))
    manifest = {"name": "panel", "modelID": "google/gemma-3-4b-it",
                "studyKind": "multiAgent",
                "multiAgentScenarioPath": "prompts/panels/panel.json"}
    return str(root), manifest


def test_a_steered_seats_artifact_is_a_pinned_input(tmp_path):
    """The crash: the bundle carried the script and not the seat."""
    root, manifest = _workspace(tmp_path)

    entries = experiment_store.pinned_input_entries(manifest, root)
    paths = [os.path.relpath(e.path, root) for e in entries]

    assert "prompts/panels/panel.json" in paths
    assert "runs/model-variants/2026-07-27-neurotic/model-variant.json" in paths, \
        "the seat's agent artifact never travelled — this is the reported crash"


def test_the_seats_vectors_travel_too(tmp_path):
    """A seat's artifact names vectors; shipping the JSON alone would fail the
    same way one level deeper."""
    root, manifest = _workspace(tmp_path)

    paths = [os.path.relpath(e.path, root)
             for e in experiment_store.pinned_input_entries(manifest, root)]

    assert "runs/vectors/neurotic.safetensors" in paths
    assert "runs/vectors/neurotic.json" in paths


def test_seat_dependencies_are_required_so_packaging_fails_closed(tmp_path):
    """An unresolvable seat must refuse PACKAGING, where the researcher is
    watching — not on the GPU, after the queue wait."""
    root, manifest = _workspace(tmp_path)

    entry = next(e for e in experiment_store.pinned_input_entries(manifest, root)
                 if e.path.endswith("model-variant.json"))
    assert entry.required


def test_an_unsteered_panel_adds_nothing_extra(tmp_path):
    """Seats without a variant are the common case and must not invent pins."""
    root, manifest = _workspace(tmp_path, with_agent=False)

    paths = [os.path.relpath(e.path, root)
             for e in experiment_store.pinned_input_entries(manifest, root)]

    assert "prompts/panels/panel.json" in paths
    # The snapshot directory is always enumerated; what must NOT appear is any
    # agent artifact invented for a seat that has none.
    assert not any("model-variants" in p for p in paths)


def test_a_missing_seat_artifact_does_not_crash_enumeration(tmp_path):
    """The scenario still enumerates; the missing file is caught by the
    packer's own required-input check, with a message naming it."""
    root, manifest = _workspace(tmp_path, agent_on_disk=False)

    entries = experiment_store.pinned_input_entries(manifest, root)
    assert any(e.path.endswith("model-variant.json") for e in entries)


# --- a panel runs on its SEATS' model, not the manifest's --------------------

def test_a_panel_loads_the_seats_model_not_the_manifests(tmp_path):
    """Reported on MPS: the manifest still said gemma-3-27b-it while every seat
    said gemma-3-4b-it, so the run spent its load fetching a 27B nobody would
    use and died with a huggingface_hub traceback."""
    from steerlab_server.experiment import tasks
    from steerlab_server.experiment.manifest import Manifest

    root, spec = _workspace(tmp_path)
    spec["modelID"] = "google/gemma-3-27b-it"          # stale baseline
    manifest = Manifest.from_dict(spec)                # seats say 4b
    notes = []

    resolved = tasks._panel_load_model(manifest, root, notes.append)

    assert resolved.model_id == "google/gemma-3-4b-it"
    assert any("no turn consults it" in n for n in notes), \
        "loading something other than the declared default must be stated"
    # The original manifest is untouched — provenance still records what was
    # declared.
    assert manifest.model_id == "google/gemma-3-27b-it"


def test_a_matching_manifest_is_left_alone_and_says_nothing(tmp_path):
    from steerlab_server.experiment import tasks
    from steerlab_server.experiment.manifest import Manifest

    root, spec = _workspace(tmp_path)
    spec["modelID"] = "google/gemma-3-4b-it"
    notes = []

    resolved = tasks._panel_load_model(Manifest.from_dict(spec), root, notes.append)

    assert resolved.model_id == "google/gemma-3-4b-it"
    assert notes == []


def test_nothing_refuses_a_multi_model_panel_on_any_path(tmp_path):
    """This used to refuse on the CLI path. Mixed-model panels are a design
    goal — eventually one GPU per seat — so the refusal is gone. Where a
    second model genuinely cannot be served, run_scenario reports it at the
    turn that needs it rather than blocking the study up front."""
    from steerlab_server.experiment import tasks
    from steerlab_server.experiment.manifest import Manifest

    root, spec = _mixed_workspace(tmp_path)

    # No exception, on either path.
    assert tasks._panel_load_model(
        Manifest.from_dict(spec), root, lambda *_: None) is not None


# --- mixed-model panels are legal --------------------------------------------

def _mixed_workspace(tmp_path):
    import json as _json
    root, spec = _workspace(tmp_path)
    panel_path = os.path.join(root, "prompts/panels/panel.json")
    panel = _json.load(open(panel_path))
    panel["agents"].append({
        "id": "b", "name": "Judge B", "baseModelID": "Qwen/Qwen3-4B",
        "systemPrompt": "", "variantArtifactPath": None,
        "variantArtifactHash": None})
    # Two turns for seat a, one for seat b — so the majority is unambiguous.
    panel["turns"].append({
        "id": "t2", "title": "T2", "speakerAgentID": "b", "promptTemplate": "go",
        "outputLabel": "o2", "routing": "all", "routedAgentIDs": [],
        "includeScenarioMaterials": True, "includeSpeakerContext": True,
        "maxTokens": None})
    panel["turns"].append({
        "id": "t3", "title": "T3", "speakerAgentID": "a", "promptTemplate": "go",
        "outputLabel": "o3", "routing": "all", "routedAgentIDs": [],
        "includeScenarioMaterials": True, "includeSpeakerContext": True,
        "maxTokens": None})
    _json.dump(panel, open(panel_path, "w"))
    return root, spec


def test_a_mixed_model_panel_is_not_refused(tmp_path):
    """Seats naming different models is a DESIGN, not an error — eventually one
    GPU per seat. Nothing may refuse it."""
    from steerlab_server.experiment import tasks
    from steerlab_server.experiment.manifest import Manifest

    root, spec = _mixed_workspace(tmp_path)
    notes = []

    resolved = tasks._panel_load_model(Manifest.from_dict(spec), root, notes.append)

    # Loads the model most TURNS need, and says what the mix is.
    assert resolved.model_id == "google/gemma-3-4b-it"
    assert any("mixed-model panel" in n for n in notes)
    assert any("Qwen/Qwen3-4B" in n for n in notes)


def test_the_declared_default_is_never_called_a_claim_about_the_run(tmp_path):
    from steerlab_server.experiment import tasks
    from steerlab_server.experiment.manifest import Manifest

    root, spec = _workspace(tmp_path)
    spec["modelID"] = "google/gemma-3-27b-it"   # declared default only
    notes = []

    resolved = tasks._panel_load_model(Manifest.from_dict(spec), root, notes.append)

    assert resolved.model_id == "google/gemma-3-4b-it"
    assert any("no turn consults it" in n for n in notes)


def test_a_revision_is_never_pinned_onto_a_model_that_was_not_loaded(tmp_path):
    """The corruption: a 4B commit was written into a manifest declaring 27B,
    and it persists into every freeze and bundle."""
    from types import SimpleNamespace

    from steerlab_server.experiment import tasks
    from steerlab_server.experiment.manifest import Manifest

    root, spec = _workspace(tmp_path)
    spec["modelID"] = "google/gemma-3-27b-it"
    spec["status"] = "draft"
    manifest = Manifest.from_dict(spec)
    loaded = SimpleNamespace(model_id="google/gemma-3-4b-it", revision="093f9f38deadbeef")
    notes = []

    out = tasks._pin_model_revision("panel", manifest, loaded, root, notes.append)

    assert out.model_revision is None, "a foreign revision must not be pinned"
    assert any("belongs to" in n for n in notes)


def test_a_matching_model_still_pins_its_revision(tmp_path):
    """The guard must not break ordinary revision pinning."""
    from types import SimpleNamespace

    from steerlab_server.experiment import experiment_store, tasks
    from steerlab_server.experiment.manifest import Manifest

    root, spec = _workspace(tmp_path)
    spec["status"] = "draft"
    os.makedirs(os.path.join(root, "experiments"), exist_ok=True)
    import json as _json
    _json.dump(spec, open(os.path.join(root, "experiments", "panel.json"), "w"))
    loaded = SimpleNamespace(model_id="google/gemma-3-4b-it", revision="abc123def456")

    out = tasks._pin_model_revision(
        "panel", Manifest.from_dict(spec), loaded, root, lambda *_: None)

    assert out.model_revision == "abc123def456"
