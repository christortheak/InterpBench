"""Installed-model listing vs the install verb (live cluster defect,
2026-07-17): a successful ``/api/models/install`` printed its snapshot under
the shared cache, yet the app's installed-models UI (fed by ``/api/state``'s
``models``) said none were installed. The contract under test: every listing
resolves the hub directory EXACTLY the way ``huggingface_hub`` (which the
installer downloads through) resolves it — ``HF_HUB_CACHE`` verbatim when
set, else ``$HF_HOME/hub`` — and the controller role does not skip the scan.

Hermetic: fake cache layouts on disk, no hub/network imports.
"""

import os

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from steerlab_server.api import housekeeping
from steerlab_server.steering import model_loader


def _fake_repo(hub_dir, repo_name, revision="abc123"):
    snap = os.path.join(hub_dir, repo_name, "snapshots", revision)
    os.makedirs(snap, exist_ok=True)
    with open(os.path.join(snap, "config.json"), "w", encoding="utf-8") as fh:
        fh.write("{}")
    refs = os.path.join(hub_dir, repo_name, "refs")
    os.makedirs(refs, exist_ok=True)
    with open(os.path.join(refs, "main"), "w", encoding="utf-8") as fh:
        fh.write(revision)


def test_listing_finds_installed_repo_under_hf_home(tmp_path, monkeypatch):
    # The layout snapshot_download produces under HF_HOME:
    # <HF_HOME>/hub/models--org--name/snapshots/<hash>/config.json
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    _fake_repo(str(tmp_path / "hub"), "models--google--gemma-3-4b-it")
    # Full-precision repos list; MLX twins and SAE artifact repos never do.
    _fake_repo(str(tmp_path / "hub"), "models--mlx-community--gemma-3-4b-it-4bit")
    _fake_repo(str(tmp_path / "hub"), "models--google--gemma-scope-2-4b-pt")
    assert model_loader.local_model_ids() == ["google/gemma-3-4b-it"]
    assert model_loader.cached_revision("google/gemma-3-4b-it") == "abc123"


def test_artifact_repos_without_a_model_config_are_not_offered(tmp_path,
                                                               monkeypatch):
    """Staging a non-model artifact repo in the shared cache must not put it
    in the model picker.

    Live 2026-07-27: the first J-lens staging made
    ``neuronpedia/jacobian-lens`` selectable, and choosing it errored — its
    snapshot holds one directory per model, no ``config.json``, no weights.
    The name-based filters could not see it, so the listing now also requires
    a root ``config.json``. Unlike ``_fake_repo`` above, these fixtures are
    shaped like the real artifact repos: a subdirectory and no config.
    """
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    hub = str(tmp_path / "hub")
    _fake_repo(hub, "models--google--gemma-3-4b-it")
    for repo, payload in (("models--neuronpedia--jacobian-lens", "gemma-3-27b-it"),
                          ("models--google--gemma-scope-2-4b-it", "resid_post")):
        os.makedirs(os.path.join(hub, repo, "snapshots", "deadbee", payload),
                    exist_ok=True)
        os.makedirs(os.path.join(hub, repo, "refs"), exist_ok=True)
        with open(os.path.join(hub, repo, "refs", "main"), "w",
                  encoding="utf-8") as fh:
            fh.write("deadbee")
    assert model_loader.local_model_ids() == ["google/gemma-3-4b-it"]


def test_listing_honors_hf_hub_cache_like_the_installer_does(tmp_path,
                                                             monkeypatch):
    # HF_HUB_CACHE names the hub dir DIRECTLY (no hub/ suffix) and wins over
    # HF_HOME in huggingface_hub — so an install that landed there must be
    # visible to the listing too, or "no models installed" contradicts a
    # printed snapshot path (the live mismatch).
    home = tmp_path / "home-cache"
    shared = tmp_path / "work" / "hf-cache"
    monkeypatch.setenv("HF_HOME", str(home))
    monkeypatch.setenv("HF_HUB_CACHE", str(shared))
    _fake_repo(str(shared), "models--google--gemma-3-4b-it")
    assert model_loader.local_model_ids() == ["google/gemma-3-4b-it"]
    assert model_loader.cached_revision("google/gemma-3-4b-it") == "abc123"
    # An explicit cache_root argument keeps its historical <root>/hub meaning.
    assert model_loader.local_model_ids(cache_root=str(tmp_path / "empty")) == []


def test_state_route_reports_installed_models_on_the_controller(
        tmp_path, monkeypatch):
    # The controller role must not skip the scan: with no live session,
    # /api/state falls through to the controller and its `models` field is
    # what the app's installed-models UI renders.
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("STEERLAB_SERVER_ROLE", "controller")
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path / "meta"))
    _fake_repo(str(tmp_path / "hub"), "models--google--gemma-3-4b-it")
    from steerlab_server.api.app import app
    resp = TestClient(app).get("/api/state")
    assert resp.status_code == 200
    assert resp.json()["models"] == ["google/gemma-3-4b-it"]


def test_housekeeping_inventory_follows_the_same_hub_resolution(
        tmp_path, monkeypatch):
    shared = tmp_path / "work" / "hf-cache"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "home-cache"))
    monkeypatch.setenv("HF_HUB_CACHE", str(shared))
    _fake_repo(str(shared), "models--google--gemma-3-4b-it")
    inventory = housekeeping.scan_hf_cache()
    assert inventory is not None
    assert [m["modelId"] for m in inventory["models"]] == [
        "google/gemma-3-4b-it"]
    info = housekeeping.model_snapshot_info("google/gemma-3-4b-it")
    assert info is not None
    # Explicit cache_root callers keep the historical <root>/hub layout.
    assert housekeeping.scan_hf_cache(cache_root=str(tmp_path / "empty")) is None
