"""Neutral-mean centering of ablation directions + the mean-alignment preflight.

Why this exists (2026-08-06 collapse study, 4B tier, both families): extracted
concept vectors routinely share a large component with the residual stream's
neutral-corpus mean — on real 27B artifacts, ~74% of each vector's norm lies
along one shared direction, and cross-concept cosines average ~0.52 when
distinct concepts should be near-orthogonal. Ablating such a direction at λ=1
(any layer, every position) collapses generation into single-token repetition;
projecting the neutral mean out of the direction first fully restores
coherence. Qwen3 vectors with |cos| ≤ 0.28 do not collapse — the collapse
tracks mean-alignment, not ablation per se, which is what calibrates the warn
threshold.

The centering fixture numbers here are the CROSS-ENGINE contract — Swift
``MeanCenteringTests`` asserts the identical values.
"""

import warnings

import numpy as np
import pytest

from steerlab_server.steering import vector_math as vm
from steerlab_server.steering import vector_store


# --- pure math (cross-engine fixture values) --------------------------------

def test_mean_centered_removes_exactly_the_mean_component():
    # v = [3,4,0], m̂ = [0,1,0] → v − (v·m̂)m̂ = [3,0,0]. Same numbers in Swift.
    assert vm.mean_centered([3.0, 4.0, 0.0], [0.0, 2.0, 0.0]) == \
        pytest.approx([3.0, 0.0, 0.0])


def test_mean_centered_result_is_orthogonal_to_the_mean():
    centered = vm.mean_centered([1.0, 2.0, 3.0], [0.5, -1.0, 2.0])
    assert vm.dot(centered, [0.5, -1.0, 2.0]) == pytest.approx(0.0, abs=1e-5)


def test_mean_centered_is_idempotent():
    once = vm.mean_centered([1.0, 2.0, 3.0], [0.5, -1.0, 2.0])
    twice = vm.mean_centered(once, [0.5, -1.0, 2.0])
    assert twice == pytest.approx(once, abs=1e-6)


def test_mean_centered_with_zero_mean_is_identity():
    assert vm.mean_centered([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == \
        pytest.approx([1.0, 2.0, 3.0])


def test_mean_alignment_is_abs_cosine_and_degenerate_safe():
    assert vm.mean_alignment([3.0, 4.0, 0.0], [0.0, 2.0, 0.0]) == pytest.approx(0.8)
    assert vm.mean_alignment([3.0, -4.0, 0.0], [0.0, 2.0, 0.0]) == pytest.approx(0.8)
    assert vm.mean_alignment([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]) == 0.0


def test_warn_threshold_is_the_pinned_cross_engine_constant():
    # Calibrated: Qwen3-0.6B coherent at ≤0.28; Gemma-3-4b collapses from
    # ~0.45 mean alignment. Swift pins the same value.
    assert vm.ABLATION_MEAN_ALIGNMENT_WARN_THRESHOLD == pytest.approx(0.35)


# --- artifact round-trip ----------------------------------------------------

def _artifact(tmp_path, *, with_mean, layer_count=3, hidden=3):
    run_dir = tmp_path / "runs" / "20260806T0-extract"
    run_dir.mkdir(parents=True, exist_ok=True)
    sidecar = vector_store.SteeringVectorSidecar(
        modelID="org/m", concept="anger", stimulusSetHash="h" * 64,
        layerCount=layer_count, hiddenSize=hidden,
        normsPerLayer=[5.0] * layer_count,
        extractionDate="2026-08-06T00:00:00Z",
        residualNormPerLayer=[1.0] * layer_count,
        extractionMethod="meanDifference", substrate=vector_store.SUBSTRATE)
    vectors = vector_store.ConceptVectors(per_layer=[[3.0, 4.0, 0.0]] * layer_count)
    mean = [[0.0, 2.0, 0.0]] * layer_count if with_mean else None
    vector_store.save(vectors, sidecar, str(run_dir), "anger",
                      neutral_mean_per_layer=mean)
    return str(run_dir)


def test_save_stamps_neutral_mean_source_and_load_round_trips(tmp_path):
    run_dir = _artifact(tmp_path, with_mean=True)
    _, sidecar = vector_store.load(run_dir, "anger")
    assert sidecar.neutralMeanSource == "neutral-corpus"
    mean = vector_store.load_neutral_mean(run_dir, "anger")
    assert mean is not None
    assert mean[0] == pytest.approx([0.0, 2.0, 0.0])


def test_artifact_without_mean_loads_none_not_zero(tmp_path):
    run_dir = _artifact(tmp_path, with_mean=False)
    _, sidecar = vector_store.load(run_dir, "anger")
    assert sidecar.neutralMeanSource is None
    assert vector_store.load_neutral_mean(run_dir, "anger") is None


def test_stamped_sidecar_with_missing_mean_tensors_refuses(tmp_path):
    run_dir = _artifact(tmp_path, with_mean=False)
    import json, os
    path = os.path.join(run_dir, "anger.json")
    with open(path) as handle:
        d = json.load(handle)
    d["neutralMeanSource"] = "neutral-corpus"
    with open(path, "w") as handle:
        json.dump(d, handle)
    with pytest.raises(ValueError, match="missing neutral_mean_layer_0"):
        vector_store.load_neutral_mean(run_dir, "anger")


def test_layer_count_mismatch_refuses_at_save(tmp_path):
    run_dir = tmp_path / "runs" / "r"
    run_dir.mkdir(parents=True)
    sidecar = vector_store.SteeringVectorSidecar(
        modelID="org/m", concept="c", stimulusSetHash="h" * 64,
        layerCount=2, hiddenSize=3, normsPerLayer=[1.0, 1.0],
        extractionDate="2026-08-06T00:00:00Z")
    with pytest.raises(ValueError, match="neutral mean has 1 layers"):
        vector_store.save(
            vector_store.ConceptVectors(per_layer=[[1.0, 0.0, 0.0]] * 2),
            sidecar, str(run_dir), "c",
            neutral_mean_per_layer=[[0.0, 1.0, 0.0]])


# --- extractor mean computation --------------------------------------------

def test_neutral_mean_per_layer_is_the_rowwise_mean():
    from steerlab_server.steering.extractor import (
        StimulusActivations, _neutral_mean_per_layer)
    neutral = StimulusActivations(
        values=[[[1.0, 0.0], [10.0, 2.0]],   # text 0: layer0, layer1
                [[3.0, 4.0], [30.0, 6.0]]],  # text 1
        residual_norm_per_layer=[1.0, 1.0])
    mean = _neutral_mean_per_layer(neutral, 2)
    assert mean[0] == pytest.approx([2.0, 2.0])
    assert mean[1] == pytest.approx([20.0, 4.0])


# --- variant builder: declared centering + preflight ------------------------

def _variant(run_dir, *, mode="ablate", centering=None, alpha=1.0):
    from steerlab_server.experiment import model_variant
    injection = {"concept": "anger", "vectorArtifactID": f"{run_dir}/anger",
                 "layer": 1, "alpha": alpha, "mode": mode}
    if centering is not None:
        injection["centering"] = centering
    return model_variant.ModelVariant(
        name="v", base_model_id="org/m", injections=[injection], band_width=1)


def test_variant_declared_centering_yields_mean_orthogonal_cells(tmp_path):
    from steerlab_server.experiment import model_variant
    run_dir = _artifact(tmp_path, with_mean=True)
    cells = model_variant.variant_injections(_variant(run_dir, centering="neutralMean"))
    assert cells, "ablation cells expected"
    for cell in cells:
        assert cell.mode == "ablate"
        assert vm.dot(cell.vector, [0.0, 2.0, 0.0]) == pytest.approx(0.0, abs=1e-5)
        assert cell.vector == pytest.approx([3.0, 0.0, 0.0])


def test_variant_declared_centering_without_stored_mean_refuses(tmp_path):
    from steerlab_server.experiment import model_variant
    run_dir = _artifact(tmp_path, with_mean=False)
    with pytest.raises(ValueError, match="no stored neutral mean"):
        model_variant.variant_injections(_variant(run_dir, centering="neutralMean"))


def test_variant_unknown_centering_refuses(tmp_path):
    from steerlab_server.experiment import model_variant
    run_dir = _artifact(tmp_path, with_mean=True)
    with pytest.raises(ValueError, match="unknown centering"):
        model_variant.variant_injections(_variant(run_dir, centering="banana"))


def test_variant_centering_on_a_steering_injection_refuses(tmp_path):
    from steerlab_server.experiment import model_variant
    run_dir = _artifact(tmp_path, with_mean=True)
    with pytest.raises(ValueError, match="STEERING"):
        model_variant.variant_injections(
            _variant(run_dir, mode="add", centering="neutralMean"))


def test_variant_uncentered_mean_aligned_ablation_warns(tmp_path):
    from steerlab_server.experiment import model_variant
    run_dir = _artifact(tmp_path, with_mean=True)  # |cos| = 0.8 > 0.35
    with pytest.warns(UserWarning, match="strongly aligned"):
        cells = model_variant.variant_injections(_variant(run_dir))
    # Diagnostic only: the direction is NOT silently centered.
    assert cells[0].vector == pytest.approx([3.0, 4.0, 0.0])


def test_variant_uncentered_ablation_without_mean_warns_check_impossible(tmp_path):
    from steerlab_server.experiment import model_variant
    run_dir = _artifact(tmp_path, with_mean=False)
    with pytest.warns(UserWarning, match="preflight impossible"):
        model_variant.variant_injections(_variant(run_dir))


def test_variant_steering_emits_no_centering_warning(tmp_path):
    from steerlab_server.experiment import model_variant
    run_dir = _artifact(tmp_path, with_mean=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cells = model_variant.variant_injections(_variant(run_dir, mode="add"))
    assert all(cell.mode == "add" for cell in cells)


# --- API cell resolution ----------------------------------------------------

def test_dto_centering_defaults_to_none():
    from steerlab_server.api import dto
    assert dto.InjectionDTO(layer=0, alpha=1.0).centering == "none"


def test_resolve_injections_centers_artifact_referenced_ablation(tmp_path):
    from fastapi import HTTPException
    from steerlab_server.api import dto
    from steerlab_server.api.routes import _resolve_injections
    run_dir = _artifact(tmp_path, with_mean=True)
    cells = _resolve_injections([dto.InjectionDTO(
        layer=1, alpha=1.0, mode="ablate", concept="anger",
        centering="neutralMean", vectorPath=run_dir, name="anger")])
    assert cells[0].vector == pytest.approx([3.0, 0.0, 0.0])

    with pytest.raises(HTTPException) as refused:
        _resolve_injections([dto.InjectionDTO(
            layer=1, alpha=1.0, mode="ablate", concept="anger",
            centering="neutralMean", vector=[3.0, 4.0, 0.0])])
    assert "centered by the client" in refused.value.detail


def test_resolve_injections_refuses_unknown_or_misplaced_centering(tmp_path):
    from fastapi import HTTPException
    from steerlab_server.api import dto
    from steerlab_server.api.routes import _resolve_injections
    with pytest.raises(HTTPException, match="unknown centering"):
        _resolve_injections([dto.InjectionDTO(
            layer=0, alpha=1.0, mode="ablate", centering="banana",
            vector=[1.0, 0.0])])
    with pytest.raises(HTTPException) as refused:
        _resolve_injections([dto.InjectionDTO(
            layer=0, alpha=1.0, mode="add", centering="neutralMean",
            vector=[1.0, 0.0])])
    assert "ablation-direction transform" in refused.value.detail


def test_resolve_injections_warns_on_uncentered_mean_aligned_artifact(tmp_path):
    from steerlab_server.api import dto
    from steerlab_server.api.routes import _resolve_injections
    run_dir = _artifact(tmp_path, with_mean=True)
    with pytest.warns(UserWarning, match="strongly aligned"):
        cells = _resolve_injections([dto.InjectionDTO(
            layer=1, alpha=1.0, mode="ablate", concept="anger",
            vectorPath=run_dir, name="anger")])
    assert cells[0].vector == pytest.approx([3.0, 4.0, 0.0])


# --- study-run condition preflight ------------------------------------------

def test_condition_builder_warns_on_uncentered_mean_aligned_ablation():
    from steerlab_server.experiment.manifest import Condition, Slot
    from steerlab_server.experiment.tasks import (
        ConceptVectorBundle, _condition_injections)
    from steerlab_server.steering.vector_store import ConceptVectors
    bundle = ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[3.0, 4.0, 0.0]] * 3),
        residual_norm_per_layer=[1.0] * 3,
        residual_norm_source="neutral-corpus", stimulus_hash="h" * 64,
        neutral_mean_per_layer=[[0.0, 2.0, 0.0]] * 3)
    condition = Condition(
        name="no-anger",
        slots=[Slot(concept="anger", layer=1, alpha=1.0, mode="ablate")])
    with pytest.warns(UserWarning, match="strongly aligned"):
        cells = _condition_injections(condition, {"anger": bundle})
    assert sorted(c.layer for c in cells) == [0, 1, 2]
    # Diagnostic only — frozen-manifest semantics unchanged.
    assert cells[0].vector == pytest.approx([3.0, 4.0, 0.0])


def test_condition_builder_warns_check_impossible_without_mean():
    from steerlab_server.experiment.manifest import Condition, Slot
    from steerlab_server.experiment.tasks import (
        ConceptVectorBundle, _condition_injections)
    from steerlab_server.steering.vector_store import ConceptVectors
    bundle = ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[3.0, 4.0, 0.0]] * 3),
        residual_norm_per_layer=[1.0] * 3,
        residual_norm_source="extraction-stimuli", stimulus_hash="h" * 64)
    condition = Condition(
        name="no-anger",
        slots=[Slot(concept="anger", layer=1, alpha=1.0, mode="ablate")])
    with pytest.warns(UserWarning, match="preflight impossible"):
        _condition_injections(condition, {"anger": bundle})


def test_random_direction_ablation_control_is_exempt_from_the_preflight():
    """The control's whole point is an arbitrary direction — warning about
    its alignment would train researchers to ignore the warning."""
    from steerlab_server.experiment.manifest import Condition, Slot
    from steerlab_server.experiment.tasks import (
        ConceptVectorBundle, _condition_injections)
    from steerlab_server.steering.vector_store import ConceptVectors
    bundle = ConceptVectorBundle(
        vectors=ConceptVectors(per_layer=[[3.0, 4.0, 0.0]] * 3),
        residual_norm_per_layer=[1.0] * 3,
        residual_norm_source="neutral-corpus", stimulus_hash="h" * 64,
        neutral_mean_per_layer=[[0.0, 2.0, 0.0]] * 3)
    condition = Condition(
        name="control", control_type="randomDirectionAblation",
        slots=[Slot(concept="anger", layer=1, alpha=1.0, mode="ablate")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _condition_injections(condition, {"anger": bundle})
