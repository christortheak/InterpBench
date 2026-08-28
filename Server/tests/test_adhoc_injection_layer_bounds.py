"""Ad-hoc generate cells bound their injection layer (2026-08-28 audit, F6).

``_resolve_injections`` used to CLAMP the layer it looked the vector row up
with and then build the cell with the RAW request value. Nothing downstream
revalidated: ``VectorInjector.apply`` returns the hidden state untouched when
no injection is registered for a layer, and the hooks dispatch only
0…num_layers−1. So a layer past the end came back as UNSTEERED output with
HTTP 200 — a silent null on the playground surface — and a layer between the
artifact's depth and the model's injected the artifact's LAST row where the
artifact says nothing about the residual stream.

Refusal, not a shared clamp: the frozen-run paths resolve a layer once and use
that value for both purposes, but this is the open playground, and a cell that
quietly moves to a layer nobody asked for is a wrong number wearing a 200.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import HTTPException
from safetensors.numpy import save_file

from steerlab_server.api import dto
from steerlab_server.api.routes import _resolve_injections

HIDDEN = 3
ARTIFACT_LAYERS = 4
MODEL_LAYERS = 6


def _artifact(directory) -> str:
    directory = str(directory)
    os.makedirs(directory, exist_ok=True)
    save_file(
        {f"layer_{i}": np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
         for i in range(ARTIFACT_LAYERS)},
        os.path.join(directory, "fear.safetensors"))
    with open(os.path.join(directory, "fear.json"), "w", encoding="utf-8") as h:
        json.dump({"schemaVersion": 2, "modelID": "org/m", "concept": "fear",
                   "stimulusSetHash": "h", "layerCount": ARTIFACT_LAYERS,
                   "hiddenSize": HIDDEN, "normsPerLayer": [1.0] * ARTIFACT_LAYERS,
                   "extractionDate": "2026-01-01T00:00:00Z"}, h, sort_keys=True)
    return directory


def _artifact_cell(directory, layer: int) -> dto.InjectionDTO:
    return dto.InjectionDTO(layer=layer, alpha=1.0, concept="fear",
                            vectorPath=directory, name="fear")


def _explicit_cell(layer: int) -> dto.InjectionDTO:
    return dto.InjectionDTO(layer=layer, alpha=1.0, vector=[1.0, 0.0, 0.0])


# --- the artifact-referenced branch -----------------------------------------

def test_an_in_range_artifact_cell_still_resolves(tmp_path):
    directory = _artifact(tmp_path)
    cells = _resolve_injections([_artifact_cell(directory, 3)],
                                num_layers=MODEL_LAYERS)
    assert [c.layer for c in cells] == [3]
    assert cells[0].vector == [1.0, 0.0, 0.0]


def test_a_layer_past_the_artifact_refuses_naming_the_range(tmp_path):
    """The mismatched-depth case: the model HAS a layer 4, so nothing
    downstream would have complained — the artifact's last row would simply
    have been injected at a layer the artifact never described."""
    directory = _artifact(tmp_path)
    with pytest.raises(HTTPException) as exc:
        _resolve_injections([_artifact_cell(directory, 4)],
                            num_layers=MODEL_LAYERS)
    assert exc.value.status_code == 400
    assert "injection layer 4 is outside artifact 'fear'" in exc.value.detail
    assert "valid layers are 0…3 (4 layers)" in exc.value.detail


def test_a_negative_artifact_layer_refuses(tmp_path):
    directory = _artifact(tmp_path)
    with pytest.raises(HTTPException) as exc:
        _resolve_injections([_artifact_cell(directory, -1)],
                            num_layers=MODEL_LAYERS)
    assert exc.value.status_code == 400
    assert "injection layer -1 is outside artifact 'fear'" in exc.value.detail


# --- the explicit-vector branch ---------------------------------------------

def test_an_explicit_vector_is_bounded_by_the_loaded_model(tmp_path):
    """An explicit vector carries no depth of its own, so the loaded model is
    the only authority. This is the pure silent-null case: layer 6 of a
    6-layer model dispatched to nothing at all and returned a 200."""
    assert [c.layer for c in _resolve_injections(
        [_explicit_cell(5)], num_layers=MODEL_LAYERS)] == [5]
    with pytest.raises(HTTPException) as exc:
        _resolve_injections([_explicit_cell(6)], num_layers=MODEL_LAYERS)
    assert exc.value.status_code == 400
    assert "outside the loaded model's depth" in exc.value.detail
    assert "valid layers are 0…5 (6 layers)" in exc.value.detail


def test_a_negative_explicit_layer_refuses_even_with_no_model_depth():
    """Callers that cannot supply a depth still get the half of the rule that
    needs no depth — layers are counted from 0."""
    with pytest.raises(HTTPException) as exc:
        _resolve_injections([_explicit_cell(-2)])
    assert exc.value.status_code == 400
    assert "negative" in exc.value.detail


# --- the routes both callers share ------------------------------------------

@pytest.mark.parametrize("route", ["/api/generate", "/api/generate/stream"])
def test_both_ad_hoc_generate_routes_refuse_with_a_400(tmp_path, monkeypatch,
                                                       route):
    """Both ad-hoc callers of ``_resolve_injections`` get the same rule — the
    streaming one is the surface the app's playground actually uses."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from steerlab_server.api.routes import ServiceState, build_router

    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "jobs.sqlite"))
    state = ServiceState()
    state.model = SimpleNamespace(model_id="org/m", revision=None,
                                  num_layers=MODEL_LAYERS)
    app = FastAPI()
    app.include_router(build_router(state))
    response = TestClient(app).post(route, json={
        "text": "hi",
        "injections": [{"layer": 9, "alpha": 1.0, "vector": [1.0, 0.0, 0.0]}]})
    assert response.status_code == 400
    assert "valid layers are 0…5" in response.json()["detail"]
