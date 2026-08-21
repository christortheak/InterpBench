"""The vector artifact must round-trip and match the cross-engine schema."""

import json
import math
import os

import pytest

from steerlab_server.steering import vector_store as store
from steerlab_server.steering.reading_position import mean_from_token
from steerlab_server.steering.vector_store import ConceptVectors, SteeringVectorSidecar


def test_save_load_roundtrip(tmp_path):
    vectors = ConceptVectors(per_layer=[[0.1, 0.2, 0.3], [-1.0, 0.0, 1.0]])
    sidecar = SteeringVectorSidecar.make(
        model_id="org/model-x", concept="french",
        stimulus_set_hash="abc123", vectors=vectors,
        extraction_method="meanDifference",
        reading_position=mean_from_token(50),
        residual_norm_per_layer=[1.2, 1.3], residual_norm_source="neutral-corpus")
    store.save(vectors, sidecar, str(tmp_path), "french")

    loaded_vectors, loaded_sidecar = store.load(str(tmp_path), "french")
    # float32 storage (matches Swift's MLXArray(Float) round-trip), so compare
    # with tolerance rather than exact equality.
    assert loaded_vectors.per_layer[0] == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
    assert loaded_vectors.layer_count == 2
    assert loaded_vectors.hidden_size == 3
    assert loaded_sidecar.modelID == "org/model-x"
    assert loaded_sidecar.readingPosition == "mean from token 50"
    assert loaded_sidecar.readingMinimumTokenCount == 51
    assert loaded_sidecar.residualNormSource == "neutral-corpus"


def test_sidecar_schema_keys_match_swift(tmp_path):
    vectors = ConceptVectors(per_layer=[[1.0, 2.0]])
    sidecar = SteeringVectorSidecar.make(
        model_id="m", concept="c", stimulus_set_hash="h", vectors=vectors)
    store.save(vectors, sidecar, str(tmp_path), "c")
    with open(os.path.join(tmp_path, "c.json"), encoding="utf-8") as handle:
        data = json.load(handle)
    # Required Codable keys present and camelCase (Swift SteeringVectorSidecar).
    for key in ("schemaVersion", "modelID", "concept", "stimulusSetHash",
                "layerCount", "hiddenSize", "normsPerLayer", "extractionDate"):
        assert key in data, key
    assert data["schemaVersion"] == 2
    assert data["layerCount"] == 1
    assert data["hiddenSize"] == 2
    # norms computed from the vector: ||[1,2]|| = sqrt(5)
    assert abs(data["normsPerLayer"][0] - (5 ** 0.5)) < 1e-5
    # Optional-and-absent stays absent: this sidecar measured no residual
    # norms, so it claims no denominator and no averaging convention. The
    # Swift decoder reads both as nil (SteeringVectorSidecarTests).
    assert "residualNormPerLayer" not in data
    assert "residualNormConvention" not in data


def test_the_convention_stamp_is_a_declared_cross_engine_field():
    """The denominator convention is half of what makes a norm-unit alpha
    mean one dose (the other half is ``residualNormSource``), so the field is
    part of the pinned sidecar shape on BOTH engines — not an ad-hoc extra
    some writer happens to add. Swift twin: the identically-named property on
    ``SteeringVectorSidecar``, asserted in ``ResidualNormConventionTests``."""
    from steerlab_server.steering import residual_norm_convention

    assert "residualNormConvention" in SteeringVectorSidecar.__dataclass_fields__
    assert SteeringVectorSidecar.__dataclass_fields__[
        "residualNormConvention"].default is None
    assert residual_norm_convention.CURRENT == "wholeCorpusMean-v1"


def test_safetensors_keys_are_layer_indexed(tmp_path):
    from safetensors.numpy import load_file
    vectors = ConceptVectors(per_layer=[[1.0], [2.0], [3.0]])
    sidecar = SteeringVectorSidecar.make(
        model_id="m", concept="c", stimulus_set_hash="h", vectors=vectors)
    store.save(vectors, sidecar, str(tmp_path), "c")
    tensors = load_file(os.path.join(tmp_path, "c.safetensors"))
    assert set(tensors.keys()) == {"layer_0", "layer_1", "layer_2"}
    assert tensors["layer_1"].tolist() == [2.0]


def test_save_rejects_nonfinite_vectors_and_norms(tmp_path):
    vectors = ConceptVectors(per_layer=[[1.0, math.nan]])
    sidecar = SteeringVectorSidecar.make(
        model_id="m", concept="bad", stimulus_set_hash="h",
        vectors=ConceptVectors(per_layer=[[1.0, 0.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        store.save(vectors, sidecar, str(tmp_path), "bad-vector")

    good_vectors = ConceptVectors(per_layer=[[1.0, 0.0]])
    bad_sidecar = SteeringVectorSidecar.make(
        model_id="m", concept="bad", stimulus_set_hash="h", vectors=good_vectors,
        residual_norm_per_layer=[math.nan], residual_norm_source="neutral")
    with pytest.raises(ValueError, match="residualNormPerLayer"):
        store.save(good_vectors, bad_sidecar, str(tmp_path), "bad-norm")


# --- substrate stamp (pinned cross-engine contract) ---------------------------

def test_substrate_stamped_and_round_trips(tmp_path):
    vectors = ConceptVectors(per_layer=[[1.0, 0.0]])
    sidecar = SteeringVectorSidecar.make(
        model_id="m", concept="c", stimulus_set_hash="h", vectors=vectors)
    # make() stamps this engine's identity, exactly.
    assert sidecar.substrate == "python-hf-transformers"
    assert sidecar.substrate == store.SUBSTRATE
    store.save(vectors, sidecar, str(tmp_path), "c")
    with open(os.path.join(tmp_path, "c.json"), encoding="utf-8") as handle:
        assert json.load(handle)["substrate"] == "python-hf-transformers"
    _, loaded = store.load(str(tmp_path), "c")
    assert loaded.substrate == "python-hf-transformers"


def test_save_stamps_hand_built_sidecar_but_never_overwrites(tmp_path):
    vectors = ConceptVectors(per_layer=[[1.0]])
    bare = SteeringVectorSidecar(
        modelID="m", concept="c", stimulusSetHash="h", layerCount=1,
        hiddenSize=1, normsPerLayer=[1.0], extractionDate="2026-01-01T00:00:00Z")
    store.save(vectors, bare, str(tmp_path), "bare")
    _, loaded = store.load(str(tmp_path), "bare")
    assert loaded.substrate == store.SUBSTRATE  # written by THIS engine
    # An explicit foreign stamp survives save() untouched (e.g. re-writing a
    # sidecar decoded from another engine's artifact must not relabel it).
    foreign = SteeringVectorSidecar(
        modelID="m", concept="c", stimulusSetHash="h", layerCount=1,
        hiddenSize=1, normsPerLayer=[1.0], extractionDate="2026-01-01T00:00:00Z",
        substrate="swift-mlx")
    store.save(vectors, foreign, str(tmp_path), "foreign")
    _, loaded = store.load(str(tmp_path), "foreign")
    assert loaded.substrate == "swift-mlx"


def test_legacy_sidecar_without_substrate_decodes_to_none():
    legacy = SteeringVectorSidecar.from_dict({
        "modelID": "m", "concept": "c", "stimulusSetHash": "h",
        "layerCount": 1, "hiddenSize": 1, "normsPerLayer": [1.0],
        "extractionDate": "2026-01-01T00:00:00Z"})
    assert legacy.substrate is None  # absent = legacy/unknown, never guessed


def test_substrate_constant_has_one_definition():
    from steerlab_server.steering import repe_reader
    from steerlab_server.experiment import experiment_store, manifest
    assert repe_reader.SUBSTRATE is store.SUBSTRATE
    assert experiment_store._THIS_SUBSTRATE is store.SUBSTRATE
    assert manifest._THIS_SUBSTRATE is store.SUBSTRATE


def test_require_native_substrate_both_directions(tmp_path):
    vectors = ConceptVectors(per_layer=[[1.0]])
    native = SteeringVectorSidecar.make(
        model_id="m", concept="c", stimulus_set_hash="h", vectors=vectors)
    store.require_native_substrate(native, "a/native")     # passes
    legacy = SteeringVectorSidecar.from_dict(
        {k: v for k, v in native.to_dict().items() if k != "substrate"})
    store.require_native_substrate(legacy, "a/legacy")     # unstamped passes
    foreign = SteeringVectorSidecar.from_dict(
        {**native.to_dict(), "substrate": "swift-mlx"})
    with pytest.raises(ValueError, match="re-extract"):
        store.require_native_substrate(foreign, "a/foreign")


# --- injection-time finiteness (legacy pre-guard artifacts) -------------------

def _poison_artifact(tmp_path, name="poison"):
    """Write a non-finite artifact DIRECTLY (bypassing save()'s guard), the way
    a legacy pre-guard extraction (commit 9049f1c) could have left one."""
    import numpy as np
    from safetensors.numpy import save_file
    save_file({"layer_0": np.asarray([1.0, 0.0], dtype=np.float32),
               "layer_1": np.asarray([math.nan, 1.0], dtype=np.float32)},
              os.path.join(tmp_path, f"{name}.safetensors"))
    with open(os.path.join(tmp_path, f"{name}.json"), "w", encoding="utf-8") as h:
        json.dump({"modelID": "m", "concept": "c", "stimulusSetHash": "h",
                   "layerCount": 2, "hiddenSize": 2, "normsPerLayer": [1.0, 1.0],
                   "extractionDate": "2026-01-01T00:00:00Z"}, h)
    return os.path.join(tmp_path, name)


def test_load_refuses_poisoned_artifact_naming_path_and_layer(tmp_path):
    _poison_artifact(str(tmp_path))
    with pytest.raises(ValueError) as err:
        store.load(str(tmp_path), "poison")
    msg = str(err.value)
    assert "poison.safetensors" in msg and "layer_1" in msg
    assert "non-finite" in msg
