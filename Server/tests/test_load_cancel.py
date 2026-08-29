"""Cancelling an in-flight model load (field incident 2026-08-29).

A selected-but-uncached 27B silently began a ~55 GB hub download that held
the ONLY resident-model slot: /api/models/unload answered ``unloaded: 0``
(a loading placeholder is never evictable), no cancel route existed, and
the busy-slot refusal said "cancel the work holding it" — a repair that
did not exist. The only way out was SIGTERMing the engine.

These tests pin the repair set: a cooperative cancel seam the registry
points at the loading slot, POST /api/models/load/cancel, an unload
response that names the cancel verb instead of a bare zero, a busy-slot
refusal that names it too, and the needs-download honesty predicate behind
the load stream's announcement.
"""

import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from steerlab_server.api import model_registry  # noqa: E402
from steerlab_server.api.model_registry import ModelRegistry  # noqa: E402
from steerlab_server.api.routes import ServiceState, build_router  # noqa: E402
from steerlab_server.steering import model_loader as loader_mod  # noqa: E402


def _fake_model(model_id, revision=None, device="cpu"):
    return SimpleNamespace(model_id=model_id, revision=revision, device=device,
                           num_layers=2, hidden_size=8, context_window=128,
                           dtype="float32")


def _cancellable_load(entered):
    """A fake load that behaves like the real one: polls the module-level
    cancel seam and aborts with LoadCancelled when it fires."""

    def load(model_id, revision=None, dtype="auto", device=None):
        entered.set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if loader_mod.load_cancel_requested():
                raise loader_mod.LoadCancelled(
                    f"load of '{model_id}' cancelled on request")
            time.sleep(0.01)
        return _fake_model(model_id, revision, device)

    return load


def test_cancel_load_frees_the_slot_and_the_next_load_proceeds(monkeypatch):
    monkeypatch.setattr(model_registry.model_loader, "available_devices",
                        lambda: ["cpu"])
    monkeypatch.setenv("STEERLAB_MAX_LOADED_MODELS", "1")
    entered = threading.Event()
    monkeypatch.setattr(model_registry.model_loader, "load",
                        _cancellable_load(entered))
    reg = ModelRegistry()
    failure: list = []

    def loader():
        try:
            reg.get_or_load("model/big")
        except loader_mod.ModelLoadError as exc:
            failure.append(exc)

    thread = threading.Thread(target=loader, daemon=True)
    thread.start()
    assert entered.wait(timeout=10)
    # The loading snapshot is honest about its state and its cancellability.
    (snap,) = reg.snapshots()
    assert snap["loading"] is True and snap["cancelRequested"] is False

    cancelled = reg.cancel_load()
    assert cancelled == [{"modelID": "model/big", "revision": None,
                          "device": "cpu"}]
    thread.join(timeout=10)
    assert not thread.is_alive()
    # The cancel travelled the failed-load path: typed, slot gone.
    assert isinstance(failure[0], loader_mod.LoadCancelled)
    assert reg.snapshots() == []

    # The one slot is free again — a different model loads normally.
    monkeypatch.setattr(
        model_registry.model_loader, "load",
        lambda model_id, revision=None, dtype="auto", device=None:
        _fake_model(model_id, revision, device))
    assert reg.get_or_load("model/other").model.model_id == "model/other"


def test_cancel_load_matches_by_model_and_never_touches_ready_slots(monkeypatch):
    monkeypatch.setattr(model_registry.model_loader, "available_devices",
                        lambda: ["cpu"])
    monkeypatch.setenv("STEERLAB_MAX_LOADED_MODELS", "2")
    monkeypatch.setattr(
        model_registry.model_loader, "load",
        lambda model_id, revision=None, dtype="auto", device=None:
        _fake_model(model_id, revision, device))
    reg = ModelRegistry()
    reg.get_or_load("model/ready")
    # No load in flight for the named model → an honest empty answer, and
    # the resident model is untouched.
    assert reg.cancel_load("model/ready") == []
    assert reg.cancel_load() == []
    (snap,) = reg.snapshots()
    assert snap["modelID"] == "model/ready" and snap["loading"] is False


def test_full_house_refusal_names_the_cancel_route_for_a_loading_holder(
        monkeypatch):
    monkeypatch.setattr(model_registry.model_loader, "available_devices",
                        lambda: ["cpu"])
    monkeypatch.setenv("STEERLAB_MAX_LOADED_MODELS", "1")
    gate = threading.Event()
    entered = threading.Event()

    def slow_load(model_id, revision=None, dtype="auto", device=None):
        entered.set()
        assert gate.wait(timeout=10)
        return _fake_model(model_id, revision, device)

    monkeypatch.setattr(model_registry.model_loader, "load", slow_load)
    reg = ModelRegistry()
    thread = threading.Thread(target=lambda: reg.get_or_load("model/big"),
                              daemon=True)
    thread.start()
    assert entered.wait(timeout=10)
    # The remedy must be one that exists (the incident's message said
    # "cancel the work holding it" when no cancel did).
    with pytest.raises(loader_mod.ModelLoadError,
                       match="POST /api/models/load/cancel"):
        reg.get_or_load("model/other")
    gate.set()
    thread.join(timeout=10)


def _app(tmp_path, monkeypatch, loader):
    monkeypatch.setenv("STEERLAB_JOBS_DB", str(tmp_path / "jobs.sqlite"))
    monkeypatch.setattr(model_registry.model_loader, "available_devices",
                        lambda: ["cpu"])
    monkeypatch.setattr(model_registry.model_loader, "load", loader)
    state = ServiceState()
    app = FastAPI()
    app.include_router(build_router(state))
    return TestClient(app), state


def test_cancel_route_is_an_idempotent_noop_when_nothing_is_loading(
        tmp_path, monkeypatch):
    client, _state = _app(
        tmp_path, monkeypatch,
        lambda model_id, revision=None, dtype="auto", device=None:
        _fake_model(model_id, revision, device))
    resp = client.post("/api/models/load/cancel", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True and body["cancelRequested"] == []
    assert "nothing to cancel" in body["note"]


def test_cancel_route_stops_an_in_flight_load(tmp_path, monkeypatch):
    entered = threading.Event()
    client, state = _app(tmp_path, monkeypatch, _cancellable_load(entered))

    errors: list = []

    def load_request():
        resp = client.post("/api/load", json={"model": "model/big"})
        errors.append(resp)

    thread = threading.Thread(target=load_request, daemon=True)
    thread.start()
    assert entered.wait(timeout=10)
    resp = client.post("/api/models/load/cancel",
                       json={"modelID": "model/big"})
    body = resp.json()
    assert body["cancelRequested"] == [
        {"modelID": "model/big", "revision": None, "device": "cpu"}]
    assert "cooperative" in body["note"]
    thread.join(timeout=10)
    # The load request answered the typed refusal, and the slot is free.
    assert errors[0].status_code == 400
    assert "cancelled" in errors[0].json()["detail"]
    assert state.registry.snapshots() == []


def test_unload_of_a_loading_model_names_the_cancel_verb(tmp_path, monkeypatch):
    entered = threading.Event()
    client, state = _app(tmp_path, monkeypatch, _cancellable_load(entered))
    thread = threading.Thread(
        target=lambda: client.post("/api/load", json={"model": "model/big"}),
        daemon=True)
    thread.start()
    assert entered.wait(timeout=10)
    # The incident's dead end: unloaded: 0 with no pointer at a repair.
    resp = client.post("/api/models/unload", json={"modelID": "model/big"})
    body = resp.json()
    assert body["unloaded"] == 0
    assert body["loading"] == ["model/big"]
    assert "POST /api/models/load/cancel" in body["hint"]
    state.registry.cancel_load()
    thread.join(timeout=10)


def test_preflight_reports_cached_and_download_size(tmp_path, monkeypatch):
    from steerlab_server.api import model_install
    client, _state = _app(
        tmp_path, monkeypatch,
        lambda model_id, revision=None, dtype="auto", device=None:
        _fake_model(model_id, revision, device))
    monkeypatch.setattr(model_registry.model_loader, "needs_hub_download",
                        lambda model_id, revision=None: True)
    monkeypatch.setattr(model_install, "estimated_download_bytes",
                        lambda model_id, revision=None: 55_000_000_000)
    body = client.get("/api/models/preflight",
                      params={"model": "org/big"}).json()
    assert body == {"modelID": "org/big", "revision": None, "cached": False,
                    "residency": None, "downloadBytes": 55_000_000_000}

    monkeypatch.setattr(model_registry.model_loader, "needs_hub_download",
                        lambda model_id, revision=None: False)
    body = client.get("/api/models/preflight",
                      params={"model": "org/small"}).json()
    # No download coming → no network reach for an estimate.
    assert body["cached"] is True and body["downloadBytes"] is None


def test_capabilities_advertise_load_cancel_and_preflight():
    from steerlab_server.api.profile import capability_snapshot
    chat = capability_snapshot()["chat"]
    assert chat["loadCancel"] is True
    assert chat["loadPreflight"] is True


def _fake_repo(hub, model_id, *, marker=False, incomplete=False):
    repo = os.path.join(hub, "models--" + model_id.replace("/", "--"))
    commit = "c" * 40
    snapshot = os.path.join(repo, "snapshots", commit)
    os.makedirs(snapshot)
    with open(os.path.join(snapshot, "config.json"), "w") as handle:
        handle.write("{}")
    os.makedirs(os.path.join(repo, "refs"), exist_ok=True)
    with open(os.path.join(repo, "refs", "main"), "w") as handle:
        handle.write(commit)
    blobs = os.path.join(repo, "blobs")
    os.makedirs(blobs, exist_ok=True)
    if incomplete:
        with open(os.path.join(blobs, "abc.incomplete"), "w") as handle:
            handle.write("partial")
    if marker:
        with open(os.path.join(repo, loader_mod.INSTALL_COMPLETE_MARKER),
                  "w") as handle:
            handle.write("done")
    return repo


def test_needs_hub_download_predicate(tmp_path, monkeypatch):
    hub = str(tmp_path / "hub")
    os.makedirs(hub)
    monkeypatch.setenv("HF_HUB_CACHE", hub)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    # Nothing cached → a download is coming.
    assert loader_mod.needs_hub_download("org/absent") is True
    # Marker → fully installed, no download.
    _fake_repo(hub, "org/complete", marker=True)
    assert loader_mod.needs_hub_download("org/complete") is False
    # Interrupted download (.incomplete blobs) → the load would RESUME the
    # download (the incident's own aftermath: a SIGTERM mid-download).
    _fake_repo(hub, "org/partial", incomplete=True)
    assert loader_mod.needs_hub_download("org/partial") is True
    # Legacy cache (snapshot present, no marker, no debris) → no download.
    _fake_repo(hub, "org/legacy")
    assert loader_mod.needs_hub_download("org/legacy") is False
    # Offline processes never download, whatever the cache holds.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert loader_mod.needs_hub_download("org/absent") is False


def test_estimated_download_bytes_is_none_offline(monkeypatch):
    from steerlab_server.api import model_install
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert model_install.estimated_download_bytes("org/any") is None


def test_loading_snapshot_carries_the_load_phase(monkeypatch):
    monkeypatch.setattr(model_registry.model_loader, "available_devices",
                        lambda: ["cpu"])
    monkeypatch.setenv("STEERLAB_MAX_LOADED_MODELS", "1")
    entered = threading.Event()
    release = threading.Event()

    def phased_load(model_id, revision=None, dtype="auto", device=None):
        loader_mod._set_phase(
            f"downloading '{model_id}' from the hub — cache holds "
            "5.2 of ~55.4 GB (9%)")
        entered.set()
        assert release.wait(timeout=10)
        loader_mod._set_phase(None)
        return _fake_model(model_id, revision, device)

    monkeypatch.setattr(model_registry.model_loader, "load", phased_load)
    reg = ModelRegistry()
    thread = threading.Thread(target=lambda: reg.get_or_load("model/big"),
                              daemon=True)
    thread.start()
    assert entered.wait(timeout=10)
    (snap,) = reg.snapshots()
    # A client that missed the SSE stream still sees the live phase.
    assert "cache holds 5.2 of ~55.4 GB (9%)" in snap["loadPhase"]
    release.set()
    thread.join(timeout=10)
    loader_mod._set_phase(None)


def test_json_roundtrip_of_cancel_response_is_stable(tmp_path, monkeypatch):
    # The response is part of the client contract — sorted, plain JSON.
    client, _state = _app(
        tmp_path, monkeypatch,
        lambda model_id, revision=None, dtype="auto", device=None:
        _fake_model(model_id, revision, device))
    resp = client.post("/api/models/load/cancel")
    assert json.loads(resp.text)["ok"] is True
