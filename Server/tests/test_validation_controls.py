"""C2 — discriminant-validity controls as DECLARED, pinned recipe references.

Swift used to build its cosine matrix from "every other concept on disk",
extracting each with the first pinned paired concept's options. The control
SET was ambient — it changed whenever unrelated work landed in the workspace,
so `worstCosinePair` was not a property of the study and the same manifest
produced different evidence on two machines. The control RECIPE was borrowed,
so a control authored for grand-mean extraction was read at the wrong position
by the wrong method. The server meanwhile had NO controls at all, so the two
engines disagreed about what validate measures.

Mirror of Swift ``ValidationControlTests``.
"""

import json
import os

import pytest

from steerlab_server.experiment import experiment_store as es
from steerlab_server.experiment import tasks
from steerlab_server.experiment.manifest import Manifest


def _concept(root, name, positive="afraid", negative="calm"):
    directory = os.path.join(root, "prompts", "concepts", name)
    os.makedirs(directory, exist_ok=True)
    for side, text in (("positive", positive), ("negative", negative)):
        with open(os.path.join(directory, f"{side}.jsonl"), "w",
                  encoding="utf-8") as handle:
            handle.write(json.dumps({"text": text}) + "\n")
    return directory


def _study(root, name="vc", controls=None):
    _concept(root, "fear")
    es.create(name, model_id="org/m", revision="abc123", root=root)
    es.attach(name, ["fear"], root=root)
    if controls is not None:
        d = es.load_raw(name, root)
        d["validationControls"] = controls
        es.save_raw(d, root)
    return Manifest.load(name, root)


def _stimulus_hash(root, concept):
    from steerlab_server.steering.stimulus_set import StimulusSet
    from steerlab_server.experiment import paths
    return StimulusSet.from_directory(
        paths.concept_directory(concept, root)).hash


# --- nothing disappears silently --------------------------------------------

def test_undeclared_concepts_on_disk_are_named(tmp_path):
    root = str(tmp_path)
    manifest = _study(root)
    _concept(root, "golden-gate", "bridge", "tunnel")
    _concept(root, "hungry", "starving", "full")

    advisories = tasks._undeclared_control_advisories(manifest, root)
    assert len(advisories) == 1
    assert "golden-gate" in advisories[0]
    assert "hungry" in advisories[0]
    # The pinned study concept is not "undeclared".
    assert "fear" not in advisories[0].split(":")[-1]
    # Removing an implicit behaviour must not be invisible to whoever relied
    # on it — say how to keep them.
    assert "validationControls" in advisories[0]


def test_a_declared_control_is_not_reported_as_undeclared(tmp_path):
    root = str(tmp_path)
    _concept(root, "golden-gate", "bridge", "tunnel")
    manifest = _study(root, controls=[
        {"concept": "golden-gate",
         "stimulusSetHash": _stimulus_hash(root, "golden-gate"),
         "options": {"method": "meanDifference"}}])
    assert tasks._undeclared_control_advisories(manifest, root) == []


# --- the pin is enforced -----------------------------------------------------

def test_a_drifted_control_stimulus_set_refuses(tmp_path):
    root = str(tmp_path)
    _concept(root, "golden-gate", "bridge", "tunnel")
    manifest = _study(root, controls=[
        {"concept": "golden-gate",
         "stimulusSetHash": "0" * 64,
         "options": {"method": "meanDifference"}}])

    with pytest.raises(RuntimeError, match="drifted from its pin"):
        tasks._extract_validation_controls(
            object(), manifest, root, lambda *a: None)


def test_a_missing_control_stimulus_set_refuses(tmp_path):
    root = str(tmp_path)
    manifest = _study(root, controls=[
        {"concept": "never-authored", "stimulusSetHash": "0" * 64,
         "options": {"method": "meanDifference"}}])

    with pytest.raises(RuntimeError, match="pinned inputs, not best-effort"):
        tasks._extract_validation_controls(
            object(), manifest, root, lambda *a: None)


def test_a_control_from_a_different_revision_refuses(tmp_path):
    root = str(tmp_path)
    _concept(root, "golden-gate", "bridge", "tunnel")
    manifest = _study(root, controls=[
        {"concept": "golden-gate",
         "stimulusSetHash": _stimulus_hash(root, "golden-gate"),
         "options": {"method": "meanDifference"},
         "modelRevision": "different"}])

    with pytest.raises(RuntimeError, match="not comparable"):
        tasks._extract_validation_controls(
            object(), manifest, root, lambda *a: None)


def test_no_controls_declared_extracts_nothing(tmp_path):
    root = str(tmp_path)
    manifest = _study(root)
    assert tasks._extract_validation_controls(
        object(), manifest, root, lambda *a: None) == {}


# --- the control's OWN recipe ------------------------------------------------

def test_each_control_is_extracted_with_its_own_options(tmp_path, monkeypatch):
    """The fault this closes: Swift borrowed the first paired concept's
    options, so a control authored for one method was read at the position of
    another and the resulting cosine said nothing."""
    root = str(tmp_path)
    _concept(root, "golden-gate", "bridge", "tunnel")
    _concept(root, "hungry", "starving", "full")
    manifest = _study(root, controls=[
        {"concept": "golden-gate",
         "stimulusSetHash": _stimulus_hash(root, "golden-gate"),
         "options": {"method": "lat"}},
        {"concept": "hungry",
         "stimulusSetHash": _stimulus_hash(root, "hungry"),
         "options": {"method": "meanDifference"}}])

    seen = []

    def fake_extract(model, stimuli, options, neutral_texts=None):
        from types import SimpleNamespace
        seen.append(options.method.value)
        return SimpleNamespace(
            vectors=SimpleNamespace(layer_count=2, per_layer=[[1.0], [1.0]]),
            residual_norm_per_layer=[1.0, 1.0],
            residual_norm_source="extraction-stimuli",
            # The convention stamp joined ExtractionResult 2026-08-20; the
            # stub mirrors the real interface (None = legacy/unstamped). The
            # rendering stamp and the reading-position resolution joined it
            # 2026-08-24 on the same absent-is-legacy terms.
            residual_norm_convention=None,
            residual_norm_rendering="raw",
            reading_position_resolution=None)

    monkeypatch.setattr(tasks, "core_extract", fake_extract)
    out = tasks._extract_validation_controls(
        object(), manifest, root, lambda *a: None)

    assert set(out) == {"golden-gate", "hungry"}
    # Each control's OWN declared method, in declaration order.
    assert seen == ["lat", "meanDifference"]


# --- controls are pinned inputs ---------------------------------------------

def test_control_stimuli_join_the_pin_surface(tmp_path):
    root = str(tmp_path)
    _concept(root, "golden-gate", "bridge", "tunnel")
    _study(root, controls=[
        {"concept": "golden-gate",
         "stimulusSetHash": _stimulus_hash(root, "golden-gate"),
         "options": {"method": "meanDifference"}}])
    d = es.load_raw("vc", root)

    entries = es.pinned_input_entries(d, root)
    control_entries = [e for e in entries
                       if "validation control 'golden-gate'" in e.label]
    assert len(control_entries) == 1
    # Required: a bundle missing a declared control's stimuli would produce
    # different discriminant evidence on the far side.
    assert control_entries[0].required
    assert control_entries[0].path.endswith(
        os.path.join("prompts", "concepts", "golden-gate"))


# --- engineer review, 2026-07-26 ---------------------------------------------

def test_a_control_without_a_pinned_hash_refuses(tmp_path):
    """Permitting an absent hash made the Python contract weaker than Swift's
    typed decoding, and weaker than the "complete pinned recipe" claim."""
    root = str(tmp_path)
    _concept(root, "golden-gate", "bridge", "tunnel")
    manifest = _study(root, controls=[
        {"concept": "golden-gate", "options": {"method": "meanDifference"}}])
    with pytest.raises(RuntimeError, match="no stimulusSetHash"):
        tasks._extract_validation_controls(
            object(), manifest, root, lambda *a: None)


def test_a_control_without_options_refuses(tmp_path):
    """Defaulting absent options reads the control at a position it was not
    authored for — the exact fault C2 set out to remove."""
    root = str(tmp_path)
    _concept(root, "golden-gate", "bridge", "tunnel")
    manifest = _study(root, controls=[
        {"concept": "golden-gate",
         "stimulusSetHash": _stimulus_hash(root, "golden-gate")}])
    with pytest.raises(RuntimeError, match="no extraction options"):
        tasks._extract_validation_controls(
            object(), manifest, root, lambda *a: None)


def test_a_per_control_validation_layer_refuses(tmp_path):
    """The matrix compares every cell at ONE layer, so a per-control layer
    could only conflate concept differences with depth differences."""
    root = str(tmp_path)
    _concept(root, "golden-gate", "bridge", "tunnel")
    manifest = _study(root, controls=[
        {"concept": "golden-gate",
         "stimulusSetHash": _stimulus_hash(root, "golden-gate"),
         "options": {"method": "meanDifference"},
         "validationLayer": 12}])
    with pytest.raises(RuntimeError, match="nothing reads"):
        tasks._extract_validation_controls(
            object(), manifest, root, lambda *a: None)


# --- the matrix layer --------------------------------------------------------

def _bundles(layer_count=62):
    from types import SimpleNamespace
    v = SimpleNamespace(layer_count=layer_count,
                        per_layer=[[1.0, 0.0]] * layer_count)
    return {"fear": SimpleNamespace(vectors=v)}


def test_the_matrix_layer_follows_the_declared_study_layer(tmp_path):
    from steerlab_server.experiment.manifest import Manifest
    m = Manifest.from_dict({"name": "x", "modelID": "org/m",
                            "validationLayer": 41})
    assert tasks._matrix_layers(m, _bundles()) == [41]


def test_the_matrix_layer_ignores_per_concept_condition_layers(tmp_path):
    """A per-row layer is what made the matrix asymmetric — (A,B) and (B,A)
    measured at different depths — and an asymmetric matrix has no defined
    reading for the maxCrossConceptCosine gate."""
    from steerlab_server.experiment.manifest import Manifest
    m = Manifest.from_dict({
        "name": "x", "modelID": "org/m",
        "conditions": [{"name": "c", "slots": [
            {"concept": "fear", "layer": 7, "alpha": 0.1}]}]})
    # Mid-network canonical fallback, not the condition's 7.
    assert tasks._matrix_layers(m, _bundles()) == [31]


def test_an_out_of_range_declared_layer_refuses_rather_than_clamping(tmp_path):
    from steerlab_server.experiment import validation_layer as vl
    assert vl.range_refusal(100, 62) is not None
    assert "not silently clamped" in vl.range_refusal(100, 62)
    assert vl.range_refusal(61, 62) is None
    assert vl.range_refusal(None, 62) is None


# --- malformed manifests must refuse at VERIFY, before any allocation -------

@pytest.mark.parametrize("label,controls,expect", [
    # `{}` is FALSY, so `or []` read a dict as "no controls declared" — a
    # malformed manifest verified clean and validated with no discriminant
    # evidence at all.
    ("container-is-object", {}, "must be an array"),
    ("container-is-string", "nope", "must be an array"),
    ("entry-is-string", ["nope"], "not an object"),
    ("entry-is-number", [7], "not an object"),
    ("concept-missing", [{"stimulusSetHash": "h", "options": {}}],
     "no usable 'concept'"),
    ("concept-not-string", [{"concept": 123, "stimulusSetHash": "h",
                             "options": {}}], "no usable 'concept'"),
    ("concept-blank", [{"concept": "   ", "stimulusSetHash": "h",
                        "options": {}}], "no usable 'concept'"),
    ("hash-not-string", [{"concept": "g", "stimulusSetHash": 5,
                          "options": {}}], "must be a non-empty string"),
    # A FALSY non-dict was silently converted into DEFAULT extraction options,
    # reading the control at a position it was never authored for; a truthy
    # one failed only at extraction, after model allocation.
    ("options-is-list", [{"concept": "g", "stimulusSetHash": "h",
                          "options": []}], "must be an object"),
    ("options-is-string", [{"concept": "g", "stimulusSetHash": "h",
                            "options": "meanDifference"}], "must be an object"),
])
def test_malformed_validation_controls_refuse_at_verify(label, controls, expect):
    from steerlab_server.experiment.manifest import Manifest
    m = Manifest.from_dict({"name": "x", "modelID": "org/m",
                            "validationControls": controls})
    violations = [v for v in m.verify() if expect in v]
    assert violations, f"{label}: expected a violation containing {expect!r}"


def test_a_well_formed_control_verifies_clean():
    from steerlab_server.experiment.manifest import Manifest
    m = Manifest.from_dict({
        "name": "x", "modelID": "org/m",
        "validationControls": [{
            "concept": "golden-gate", "stimulusSetHash": "h" * 64,
            "options": {"method": "meanDifference"}}]})
    assert [v for v in m.verify() if "control" in v] == []


def test_verification_catches_it_before_the_model_loads(tmp_path):
    """The point of moving these into verify(): a malformed frozen manifest
    must not consume a cluster allocation to discover a schema problem that
    needs no model."""
    from steerlab_server.experiment.manifest import Manifest
    m = Manifest.from_dict({"name": "x", "modelID": "org/m",
                            "validationControls": [{"concept": "g"}]})
    # verify() takes no model and touches no GPU.
    assert any("control" in v for v in m.verify())
