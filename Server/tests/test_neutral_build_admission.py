"""Model selection is checked before submission and again at execution."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from steerlab_server.api.routes import build_router


@pytest.fixture
def build_client(monkeypatch):
    from steerlab_server.api.app import state

    queued = []

    def submit(kind, work):
        queued.append(work)
        return SimpleNamespace(id="build-test")

    monkeypatch.setattr(state, "model", SimpleNamespace(model_id="org/selected"))
    monkeypatch.setattr(state, "_jobs", SimpleNamespace(submit=submit))
    app = FastAPI()
    app.include_router(build_router(state))
    return TestClient(app), state, queued


def test_mismatched_model_is_rejected_without_a_job(build_client):
    client, _, queued = build_client
    response = client.post("/api/neutral-pcs/build", json={"expectedModelID": "org/other"})
    assert response.status_code == 400
    assert "load the selected model" in response.json()["detail"]
    assert queued == []


@pytest.mark.parametrize("body", [{"expectedModelID": "org/selected"}, {}])
def test_matching_and_legacy_requests_still_submit(build_client, body):
    client, _, queued = build_client
    response = client.post("/api/neutral-pcs/build", json=body)
    assert response.status_code == 200
    assert response.json() == {"jobId": "build-test"}
    assert len(queued) == 1


def test_model_change_after_admission_is_checked_before_measurement(build_client, monkeypatch):
    from steerlab_server.experiment import neutral
    from steerlab_server.steering import extractor

    client, state, queued = build_client
    response = client.post("/api/neutral-pcs/build", json={"expectedModelID": "org/selected"})
    assert response.status_code == 200
    monkeypatch.setattr(neutral, "read_corpus", lambda _: (["sample"] * 4, "a" * 64))

    @contextmanager
    def changed_model():
        yield SimpleNamespace(model_id="org/other")

    def unexpected_measurement(*args, **kwargs):
        pytest.fail("A mismatched model reached activation measurement")

    monkeypatch.setattr(state, "acquire_active", changed_model)
    monkeypatch.setattr(extractor, "neutral_activation_bank", unexpected_measurement)
    with pytest.raises(RuntimeError, match="loaded model changed"):
        queued[0](SimpleNamespace())
