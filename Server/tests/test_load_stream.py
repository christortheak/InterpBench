"""POST /api/load/stream — the SSE model-load route (2026-07-17).

The synchronous /api/load shows a static "loading…" for minutes and races the
client's request timeout on slow-storage cold loads. The stream route reuses
the _locked_sse machinery: a status preamble, elapsed-time heartbeats, in-band
error events, and a terminal `done` carrying the sync route's payload.
"""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from steerlab_server.api import model_registry  # noqa: E402
from steerlab_server.api.routes import ServiceState, build_router  # noqa: E402


def _fake_model():
    return SimpleNamespace(model_id="org/tiny", revision="r1", device="cpu",
                           num_layers=2, hidden_size=8, context_window=128)


def _app(tmp_path, monkeypatch, loader, *, needs_download=False):
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "jobs.sqlite"))
    monkeypatch.setattr(model_registry.model_loader, "available_devices",
                        lambda: ["cpu"])
    monkeypatch.setattr(model_registry.model_loader, "load", loader)
    # Hermetic by default: the fake model reads as locally cached, so the
    # route never consults hub metadata (a network reach) in unit tests.
    monkeypatch.setattr(model_registry.model_loader, "needs_hub_download",
                        lambda model_id, revision=None: needs_download)
    state = ServiceState()
    app = FastAPI()
    app.include_router(build_router(state))
    return TestClient(app), state


def test_load_stream_announces_loads_and_ends_with_the_sync_payload(
        tmp_path, monkeypatch):
    client, state = _app(
        tmp_path, monkeypatch,
        lambda model_id, revision=None, dtype="auto", device=None: _fake_model())
    resp = client.post("/api/load/stream", json={"model": "org/tiny"})
    assert resp.status_code == 200
    # Preamble names the cold load, completion names the device, and the
    # terminal done carries exactly what the sync /api/load returns.
    assert "not loaded yet" in resp.text
    assert "loaded on cpu" in resp.text
    assert '"done": true' in resp.text
    assert '"modelID": "org/tiny"' in resp.text
    assert '"numLayers": 2' in resp.text
    assert resp.text.index("not loaded yet") < resp.text.index('"done"')
    # The model is published exactly as a sync load would have left it.
    assert state.model is not None and state.model.model_id == "org/tiny"

    # Already resident → the preamble says so instead of predicting minutes.
    resp = client.post("/api/load/stream", json={"model": "org/tiny"})
    assert "already resident" in resp.text
    assert '"done": true' in resp.text


def test_load_stream_heartbeats_carry_the_loader_phase(tmp_path, monkeypatch):
    # "43s elapsed" alone told the live researcher nothing about WHICH phase
    # was eating the minutes (2026-07-17: a silent device copy ran 10+ min).
    # Heartbeats must carry the loader's current phase text.
    import time as _time
    from steerlab_server.steering import model_loader as loader_mod

    monkeypatch.setenv("STEERLAB_SSE_HEARTBEAT_SECONDS", "0.05")

    def slow_load(model_id, revision=None, dtype="auto", device=None):
        loader_mod._set_phase("moving 8.0 GiB to cuda:0 — cold pages stream "
                              "off disk here")
        _time.sleep(0.3)
        loader_mod._set_phase(None)
        return _fake_model()

    client, _state = _app(tmp_path, monkeypatch, slow_load)
    resp = client.post("/api/load/stream", json={"model": "org/tiny"})
    assert resp.status_code == 200
    assert "elapsed; moving 8.0 GiB to cuda:0" in resp.text
    assert '"done": true' in resp.text


def test_load_stream_announces_a_download_with_its_size_and_the_cancel_route(
        tmp_path, monkeypatch):
    # Field incident 2026-08-29: selecting an uncached 27B silently began a
    # ~55 GB hub download inside the load, with nothing telling the client
    # what was about to happen or how to stop it. The FIRST event now states
    # the download, its hub-metadata size, and the cancel route.
    from steerlab_server.api import model_install
    monkeypatch.setattr(model_install, "estimated_download_bytes",
                        lambda model_id, revision=None: 55_000_000_000)
    client, _state = _app(
        tmp_path, monkeypatch,
        lambda model_id, revision=None, dtype="auto", device=None: _fake_model(),
        needs_download=True)
    resp = client.post("/api/load/stream", json={"model": "org/tiny"})
    assert resp.status_code == 200
    first_event = json.loads(resp.text.split("\n\n")[0].removeprefix("data: "))
    assert first_event["willDownload"] is True
    assert first_event["downloadBytes"] == 55_000_000_000
    assert "~55.0 GB" in first_event["status"]
    assert "POST /api/models/load/cancel" in first_event["status"]
    assert '"done": true' in resp.text  # the load itself still completes


def test_load_stream_says_unknown_size_when_hub_metadata_is_unreachable(
        tmp_path, monkeypatch):
    # The estimate is best-effort by contract — its absence must neither
    # block the load nor silently drop the download warning.
    from steerlab_server.api import model_install
    monkeypatch.setattr(model_install, "estimated_download_bytes",
                        lambda model_id, revision=None: None)
    client, _state = _app(
        tmp_path, monkeypatch,
        lambda model_id, revision=None, dtype="auto", device=None: _fake_model(),
        needs_download=True)
    resp = client.post("/api/load/stream", json={"model": "org/tiny"})
    first_event = json.loads(resp.text.split("\n\n")[0].removeprefix("data: "))
    assert first_event["willDownload"] is True
    assert first_event["downloadBytes"] is None
    assert "an unknown size" in first_event["status"]


def test_load_stream_reports_failures_in_band(tmp_path, monkeypatch, capsys):
    def boom(model_id, revision=None, dtype="auto", device=None):
        raise model_registry.model_loader.ModelLoadError("no such model")

    client, state = _app(tmp_path, monkeypatch, boom)
    resp = client.post("/api/load/stream", json={"model": "org/missing"})
    assert resp.status_code == 200  # SSE: failures are in-band events
    assert '"error"' in resp.text and "no such model" in resp.text
    assert '"done": true' not in resp.text
    assert state.model is None
    # The failure also leaves a durable stderr traceback (engineer review
    # 2026-07-17): a dropped tunnel must not erase the only diagnostic.
    assert "ModelLoadError" in capsys.readouterr().err


def test_capability_snapshot_advertises_load_stream():
    from steerlab_server.api.profile import capability_snapshot
    assert capability_snapshot()["chat"]["loadStream"] is True
