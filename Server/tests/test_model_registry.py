import threading
import time
from types import SimpleNamespace

import pytest

from steerlab_server.api.model_registry import ModelRegistry


def _fake_model(model_id, revision, device):
    return SimpleNamespace(
        model_id=model_id, revision=revision or f"cached:{model_id}", device=device,
        num_layers=2, hidden_size=8, context_window=128)


def test_registry_spreads_models_across_cuda_devices(monkeypatch):
    from steerlab_server.api import model_registry

    monkeypatch.setattr(model_registry.model_loader, "available_devices",
                        lambda: ["cuda:0", "cuda:1", "cpu"])
    loaded = []

    def fake_load(model_id, revision=None, dtype="auto", device=None):
        loaded.append((model_id, device))
        return _fake_model(model_id, revision, device)

    monkeypatch.setattr(model_registry.model_loader, "load", fake_load)
    reg = ModelRegistry()
    a = reg.get_or_load("model/a").model
    b = reg.get_or_load("model/b").model

    assert a.device == "cuda:0"
    assert b.device == "cuda:1"
    assert loaded == [("model/a", "cuda:0"), ("model/b", "cuda:1")]


def test_registry_evicts_on_single_slot(monkeypatch):
    from steerlab_server.api import model_registry

    monkeypatch.setattr(model_registry.model_loader, "available_devices",
                        lambda: ["cuda:0", "cpu"])
    monkeypatch.setenv("STEERLAB_MAX_LOADED_MODELS", "1")
    monkeypatch.setattr(model_registry.model_loader, "load",
                        lambda model_id, revision=None, dtype="auto", device=None:
                        _fake_model(model_id, revision, device))
    monkeypatch.setattr(model_registry.torch.cuda, "is_available", lambda: False)
    reg = ModelRegistry()
    reg.get_or_load("model/a")
    reg.get_or_load("model/b")

    snapshots = reg.snapshots()
    assert len(snapshots) == 1
    assert snapshots[0]["modelID"] == "model/b"


def test_readers_answer_while_a_load_is_in_flight(monkeypatch):
    # The live 2026-07-17 failure: the registry lock was held across the
    # whole multi-minute load, so /api/capabilities (the GPU-session
    # controller's health probe) and /api/state blocked behind it — the
    # controller read a hard-working worker as unreachable and demoted the
    # session to "Starting" through every load. Readers must answer
    # INSTANTLY while a load is in flight, and report it honestly.
    from steerlab_server.api import model_registry

    monkeypatch.setattr(model_registry.model_loader, "available_devices",
                        lambda: ["cpu"])
    gate = threading.Event()
    entered = threading.Event()

    def slow_load(model_id, revision=None, dtype="auto", device=None):
        entered.set()
        assert gate.wait(timeout=10)
        return _fake_model(model_id, revision, device)

    monkeypatch.setattr(model_registry.model_loader, "load", slow_load)
    reg = ModelRegistry()
    loader = threading.Thread(target=lambda: reg.get_or_load("model/slow"),
                              daemon=True)
    loader.start()
    assert entered.wait(timeout=10)

    snaps = reg.snapshots()
    assert len(snaps) == 1
    assert snaps[0]["modelID"] == "model/slow"
    assert snaps[0]["loading"] is True
    assert snaps[0]["busy"] is True  # generations against it would queue
    assert reg.any_busy() is True
    assert reg.residency("model/slow") == "loading"
    assert reg.residency("model/other") is None
    # unload never rips a slot out from under its own in-flight load.
    assert reg.unload("model/slow") == 0

    gate.set()
    loader.join(timeout=10)
    assert not loader.is_alive()
    snaps = reg.snapshots()
    assert snaps[0]["loading"] is False
    assert reg.any_busy() is False
    assert reg.residency("model/slow") == "ready"


def test_same_model_waiters_share_one_load(monkeypatch):
    from steerlab_server.api import model_registry

    monkeypatch.setattr(model_registry.model_loader, "available_devices",
                        lambda: ["cpu"])
    calls = []
    gate = threading.Event()
    entered = threading.Event()

    def slow_load(model_id, revision=None, dtype="auto", device=None):
        calls.append(model_id)
        entered.set()
        assert gate.wait(timeout=10)
        return _fake_model(model_id, revision, device)

    monkeypatch.setattr(model_registry.model_loader, "load", slow_load)
    reg = ModelRegistry()
    results = {}

    def request(name):
        results[name] = reg.get_or_load("model/shared")

    first = threading.Thread(target=request, args=("first",), daemon=True)
    first.start()
    assert entered.wait(timeout=10)
    second = threading.Thread(target=request, args=("second",), daemon=True)
    second.start()
    time.sleep(0.05)  # let the second request reach the wait
    gate.set()
    first.join(timeout=10)
    second.join(timeout=10)

    assert calls == ["model/shared"]  # ONE load, shared
    assert results["first"] is results["second"]
    assert results["first"].model is not None


def test_failed_load_releases_waiters_who_then_retry(monkeypatch):
    from steerlab_server.api import model_registry

    monkeypatch.setattr(model_registry.model_loader, "available_devices",
                        lambda: ["cpu"])
    attempts = []
    release_first = threading.Event()
    first_entered = threading.Event()

    def flaky_load(model_id, revision=None, dtype="auto", device=None):
        attempts.append(model_id)
        if len(attempts) == 1:
            first_entered.set()
            assert release_first.wait(timeout=10)
            raise RuntimeError("shard read failed")
        return _fake_model(model_id, revision, device)

    monkeypatch.setattr(model_registry.model_loader, "load", flaky_load)
    reg = ModelRegistry()
    outcome = {}

    def creator():
        try:
            reg.get_or_load("model/flaky")
        except RuntimeError as exc:
            outcome["creator"] = exc

    def waiter():
        outcome["waiter"] = reg.get_or_load("model/flaky")

    t1 = threading.Thread(target=creator, daemon=True)
    t1.start()
    assert first_entered.wait(timeout=10)
    t2 = threading.Thread(target=waiter, daemon=True)
    t2.start()
    time.sleep(0.05)  # let the waiter park on the slot's ready event
    release_first.set()
    t1.join(timeout=10)
    t2.join(timeout=10)

    # The creator saw the real failure; the waiter was NOT stranded — it
    # retried, became the new loader, and got a model.
    assert isinstance(outcome.get("creator"), RuntimeError)
    assert outcome["waiter"].model is not None
    assert attempts == ["model/flaky", "model/flaky"]
    # The failed placeholder is gone; the retry's slot is the resident one.
    assert reg.residency("model/flaky") == "ready"


def test_acquire_retries_when_its_slot_was_evicted_before_locking(monkeypatch):
    # Engineer review 2026-07-17: unload/_make_room skip only LOCKED slots,
    # so between get_or_load returning and acquire's lock landing the slot
    # can be evicted. acquire must detect the stale slot (model is None —
    # eviction assigns, never `del`s) and reload instead of yielding a dead
    # model.
    from steerlab_server.api import model_registry

    monkeypatch.setattr(model_registry.model_loader, "available_devices",
                        lambda: ["cpu"])
    monkeypatch.setattr(model_registry.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        model_registry.model_loader, "load",
        lambda model_id, revision=None, dtype="auto", device=None:
        _fake_model(model_id, revision, device))
    reg = ModelRegistry()

    stale = reg.get_or_load("model/a")
    assert reg.unload("model/a") == 1
    assert stale.model is None  # eviction leaves a clean, checkable marker

    real_get_or_load = reg.get_or_load
    calls = []

    def racing_get_or_load(model_id, revision=None, *, dtype="auto", device=None):
        calls.append(model_id)
        if len(calls) == 1:
            return stale  # the race: hands back the just-evicted slot
        return real_get_or_load(model_id, revision, dtype=dtype, device=device)

    monkeypatch.setattr(reg, "get_or_load", racing_get_or_load)
    with reg.acquire("model/a") as model:
        assert model is not None
        assert model.model_id == "model/a"
    assert len(calls) == 2  # retried past the stale slot
    assert not stale.lock.locked()  # the stale lock was released, not leaked


def test_full_house_of_loading_slots_refuses_honestly(monkeypatch):
    from steerlab_server.api import model_registry
    from steerlab_server.steering import model_loader as loader_mod

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
    loader = threading.Thread(target=lambda: reg.get_or_load("model/a"),
                              daemon=True)
    loader.start()
    assert entered.wait(timeout=10)
    # A loading slot is never evicted mid-copy: with the one slot occupied
    # by an in-flight load, a different model refuses with the busy message
    # instead of corrupting the load.
    with pytest.raises(loader_mod.ModelLoadError, match="model/a"):
        reg.get_or_load("model/b")
    gate.set()
    loader.join(timeout=10)


def test_revisionless_request_does_not_reuse_explicit_old_revision(monkeypatch):
    from steerlab_server.api import model_registry

    monkeypatch.setattr(model_registry.model_loader, "available_devices", lambda: ["cpu"])
    monkeypatch.setenv("STEERLAB_MAX_LOADED_MODELS", "2")
    calls = []

    def fake_load(model_id, revision=None, dtype="auto", device=None):
        calls.append(revision)
        return _fake_model(model_id, revision, device)

    monkeypatch.setattr(model_registry.model_loader, "load", fake_load)
    reg = ModelRegistry()
    reg.get_or_load("model/a", revision="old")
    reg.get_or_load("model/a", revision=None)

    assert calls == ["old", None]
