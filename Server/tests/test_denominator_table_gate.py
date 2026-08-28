"""One out-of-range rule for the residual-norm denominator table (2026-08-28
mathematical-soundness audit, findings F7 + F13) — plus the missing-bundle
refusal that used to drop a condition slot in silence (F8).

Before this landed, ONE artifact with a truncated ``residualNormPerLayer``
behaved four different ways: the condition path substituted ``0.0`` and refused
as ``degenerateData`` (a typed refusal naming the wrong defect), the sweep and
variant paths clamped to the last entry and dosed the deepest layers with a
shallower layer's number, and the Swift condition path clamped too — while an
EMPTY table indexed ``[-1]`` on Swift and crashed. Three of the four were
silent, and none of the four was pinned by a test.

Cross-engine twin: ``Tests/SteeringKitTests/ResidualNormConventionTests.swift``
(the prose) and ``Tests/ExperimentKitTests/DenominatorTableGateTests.swift``
(the per-verb behaviour).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from safetensors.numpy import save_file

from steerlab_server.experiment import model_variant, tasks
from steerlab_server.experiment.manifest import Condition, Slot
from steerlab_server.steering import residual_norm_convention as convention
from steerlab_server.steering import vector_store
from steerlab_server.steering.vector_store import ConceptVectors

HIDDEN = 3
LAYERS = 4


# --- the shared prose -------------------------------------------------------

def test_the_load_time_gate_names_the_artifact_and_both_numbers():
    problem = convention.table_length_problem(
        [1.0, 2.0], layer_count=5, artifact="runs/r/c")
    assert problem == (
        "vector artifact 'runs/r/c' carries 2 residual norms for 5 layers — "
        "a denominator table must cover every layer or none, and a short one "
        "silently doses the layers it does not reach with another layer's "
        "number; re-measure the norms (vectors backfill-norms), or "
        "re-extract the concept")


def test_absent_and_empty_tables_are_legal_at_load_time():
    """The born-without families (OptVec, J-lens, Gemma Scope report imports)
    write no norms at all and acquire them through the backfill. Absent is a
    state the writers deliberately produce; SHORT is a state none of them
    produces, which is why only short refuses here."""
    assert convention.table_length_problem(None, layer_count=5, artifact="a") is None
    assert convention.table_length_problem([], layer_count=5, artifact="a") is None
    assert convention.table_length_problem(
        [1.0] * 5, layer_count=5, artifact="a") is None


def test_the_use_site_gate_names_the_layer_and_the_coverage():
    problem = convention.residual_norm_problem([1.0, 2.0], 4, artifact="fear")
    assert problem == (
        "'fear' has no residual norm at layer 4 — its denominator table "
        "covers 2 layer(s), so an α in residual-norm units cannot be "
        "denominated there; re-measure the norms (vectors backfill-norms), "
        "or switch α to raw units")


def test_the_use_site_gate_refuses_an_empty_table_and_a_negative_layer():
    """The empty table is the case that indexed ``[-1]`` on Swift: a clamp of
    ``min(layer, count - 1)`` against a zero-length table is ``-1``, which is
    a crash there and a silent last-element read in any language that wraps."""
    assert convention.residual_norm_problem([], 0, artifact="fear") is not None
    assert convention.residual_norm_problem(None, 0, artifact="fear") is not None
    assert convention.residual_norm_problem([1.0, 2.0], -1, artifact="fear") \
        is not None
    assert convention.residual_norm_problem([1.0, 2.0], 1, artifact="fear") is None


# --- the load-time gate on real bytes ---------------------------------------

def _write_artifact(directory: str, name: str, *, norms) -> None:
    os.makedirs(directory, exist_ok=True)
    save_file(
        {f"layer_{i}": np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
         for i in range(LAYERS)},
        os.path.join(directory, f"{name}.safetensors"))
    sidecar = {
        "schemaVersion": 2, "modelID": "org/m", "concept": name,
        "stimulusSetHash": "h", "layerCount": LAYERS, "hiddenSize": HIDDEN,
        "normsPerLayer": [1.0] * LAYERS, "extractionDate": "2026-01-01T00:00:00Z",
    }
    if norms is not None:
        sidecar["residualNormPerLayer"] = list(norms)
    with open(os.path.join(directory, f"{name}.json"), "w", encoding="utf-8") as h:
        json.dump(sidecar, h, sort_keys=True)


def test_loading_a_truncated_artifact_refuses(tmp_path):
    _write_artifact(str(tmp_path), "short", norms=[7.0, 7.5])
    with pytest.raises(ValueError, match="carries 2 residual norms for 4 layers"):
        vector_store.load(str(tmp_path), "short")


def test_loading_a_norms_less_or_full_artifact_still_works(tmp_path):
    _write_artifact(str(tmp_path), "bare", norms=None)
    _write_artifact(str(tmp_path), "full", norms=[7.0, 7.5, 8.0, 8.5])
    assert vector_store.load(str(tmp_path), "bare")[1].residualNormPerLayer is None
    assert vector_store.load(str(tmp_path), "full")[0].layer_count == LAYERS


# --- the condition path -----------------------------------------------------

def _bundle(norms) -> tasks.ConceptVectorBundle:
    return tasks.ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[1.0, 0.0, 0.0]] * LAYERS),
        residual_norm_per_layer=list(norms),
        residual_norm_source="neutral-corpus", stimulus_hash="h")


def _condition(layer: int) -> Condition:
    return Condition(name="steered",
                     slots=[Slot(concept="fear", layer=layer, alpha=1.0)],
                     alpha_in_norm_units=True)


def test_a_condition_at_a_covered_layer_still_doses(tmp_path):
    cells = tasks._condition_injections(
        _condition(1), {"fear": _bundle([2.0, 4.0, 6.0, 8.0])})
    assert [c.layer for c in cells] == [1]
    assert cells[0].alpha == pytest.approx(4.0)


def test_a_condition_past_a_truncated_table_refuses():
    with pytest.raises(RuntimeError) as exc:
        tasks._condition_injections(_condition(3), {"fear": _bundle([2.0, 4.0])})
    assert str(exc.value) == (
        "condition 'steered': 'fear' has no residual norm at layer 3 — its "
        "denominator table covers 2 layer(s), so an α in residual-norm units "
        "cannot be denominated there; re-measure the norms (vectors "
        "backfill-norms), or switch α to raw units")


def test_a_condition_against_an_empty_table_refuses():
    """This used to reach ``norm_unit_scale(alpha, 0.0, ‖v‖)`` and refuse as
    ``degenerateData`` — typed, but describing a degenerate VECTOR rather than
    an absent denominator, which sends the reader to re-extract the concept
    instead of measuring its norms."""
    with pytest.raises(RuntimeError, match="covers 0 layer\\(s\\)"):
        tasks._condition_injections(_condition(0), {"fear": _bundle([])})


def test_a_condition_naming_an_unextracted_concept_refuses():
    """F8: this used to ``continue``, so the condition executed weaker — or,
    with one slot, as an unlabelled baseline — under a steered arm's name.
    Byte-identical to the Swift twin's ``ExperimentError.reason``."""
    with pytest.raises(RuntimeError) as exc:
        tasks._condition_injections(_condition(0), {})
    assert str(exc.value) == (
        "condition 'steered' references unextracted concept 'fear'")


# --- the sweep path ---------------------------------------------------------

def test_the_sweep_site_speaks_the_same_sentence_under_its_own_subject():
    """The sweep grid reads the denominator through the SAME accessor the
    condition path uses (``tasks._residual_norm_at``); only the subject
    differs. It used to clamp to the last entry, so the deepest cells of a
    layer sweep were dosed with a shallower layer's number — and, because a
    sweep's whole job is to compare layers, that is the one verb where a
    clamped denominator corrupts the comparison it exists to make."""
    assert tasks._residual_norm_at(
        [2.0, 4.0, 6.0, 8.0], 3, artifact="fear", where="concept 'fear'") == 8.0
    with pytest.raises(RuntimeError) as exc:
        tasks._residual_norm_at([2.0, 4.0], 3, artifact="fear",
                                where="concept 'fear'")
    assert str(exc.value) == (
        "concept 'fear': 'fear' has no residual norm at layer 3 — its "
        "denominator table covers 2 layer(s), so an α in residual-norm units "
        "cannot be denominated there; re-measure the norms (vectors "
        "backfill-norms), or switch α to raw units")
    with pytest.raises(RuntimeError, match="covers 0 layer\\(s\\)"):
        tasks._residual_norm_at([], 0, artifact="fear", where="concept 'fear'")


# --- the variant path -------------------------------------------------------

def _variant(root: str, *, layer: int) -> model_variant.ModelVariant:
    return model_variant.ModelVariant.from_dict({
        "name": "v", "baseModelID": "org/m", "bandWidth": 1,
        "alphaInNormUnits": True,
        "injections": [{"concept": "fear", "layer": layer, "alpha": 1.0,
                        "vectorArtifactID": os.path.join(root, "short")}],
    })


def test_a_variant_past_a_truncated_table_refuses(tmp_path):
    """The variant path clamped to the last entry, so the same manifest that
    refused as a condition ran here with the wrong dose. The truncated table
    is now stopped at load; the use-site rule below is the same sentence when
    a bundle reaches the site by another road."""
    _write_artifact(str(tmp_path), "short", norms=[7.0, 7.5])
    with pytest.raises(ValueError, match="carries 2 residual norms for 4 layers"):
        model_variant.variant_injections(
            _variant(str(tmp_path), layer=3), root=str(tmp_path))


def test_a_variant_at_a_covered_layer_still_doses(tmp_path):
    _write_artifact(str(tmp_path), "short", norms=[2.0, 4.0, 6.0, 8.0])
    cells = model_variant.variant_injections(
        _variant(str(tmp_path), layer=2), root=str(tmp_path))
    assert [c.layer for c in cells] == [2]
    assert cells[0].alpha == pytest.approx(6.0)


# --- F8 at verify time ------------------------------------------------------

def test_verify_reports_a_condition_slot_naming_an_unattached_concept(tmp_path):
    """The run-path refusal above is defence in depth; this is where the state
    should be caught. ``Manifest.verify`` already checked a VARIANT's
    forward-referenced concept and had no equivalent for condition slots, so a
    hand-edited manifest reached the run path — after a run was scheduled and
    a model loaded — to discover it there. Byte-identical to the Swift twin's
    violation (``ExperimentStore.verify``)."""
    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path)
    es.create("slots", model_id="org/m", revision="r1", root=root)
    raw = es.load_raw("slots", root)
    raw["conditions"] = [
        {"name": "steered",
         "slots": [{"concept": "never-attached", "layer": 1, "alpha": 1.0}],
         "alphaInNormUnits": True}]
    es.save_raw(raw, root)

    violations = Manifest.load("slots", root).verify(root)
    assert "condition 'steered': references unattached concept " \
           "'never-attached'" in violations


def test_verify_says_nothing_about_sae_latent_arms(tmp_path):
    """The latent carve-out, kept explicitly: latent arms live in their own
    collection, name FEATURES rather than attached concepts, and are checked
    by ``sae_latent.condition_violations``. The new slot check must not reach
    into them."""
    from steerlab_server.experiment import experiment_store as es
    from steerlab_server.experiment.manifest import Manifest

    root = str(tmp_path)
    es.create("latent", model_id="org/m", revision="r1", root=root)
    raw = es.load_raw("latent", root)
    raw["saeLatentConditions"] = [
        {"name": "latent-a", "mode": "amplify", "dose": 2.0,
         "serverOnly": True, "release": "r", "saeID": "layer_1_width_16k",
         "feature": 7, "layer": 1}]
    es.save_raw(raw, root)

    for violation in Manifest.load("latent", root).verify(root):
        assert "references unattached concept" not in violation
