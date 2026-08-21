"""Panel-script HTTP surface (/api/scenario*).

These routes had no test coverage at all. The privilege split is the point:
``/save`` writes a caller-named file under ``prompts/panels/`` and ``/run``
reads a caller-named path *and* runs model compute, which is the combination
that put ``/api/reader/`` behind the token gate. The read-only listing and
single-panel GET stay open, so the prefix carries a trailing slash — the same
shape as ``/api/reader/`` leaving ``GET /api/readers`` open.
"""

import json
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from steerlab_server.api import app as app_module
from steerlab_server.api.app import app

client = TestClient(app)


def _panel(root, name="panel"):
    """Write a minimal valid panel into the canonical location."""
    base = os.path.join(root, "prompts", "panels")
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, f"{name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({
            "schemaVersion": 1, "name": name, "baseModelID": "m",
            "description": "", "sharedMaterials": "", "temperature": 0.0,
            "maxTokens": 128,
            "agents": [{"id": "a", "name": "A", "baseModelID": "m",
                        "systemPrompt": "", "variantArtifactPath": None,
                        "variantArtifactHash": None}],
            "turns": [{"id": "t", "title": "T", "speakerAgentID": "a",
                       "promptTemplate": "go", "outputLabel": "", "routing": "all",
                       "routedAgentIDs": [], "includeScenarioMaterials": True,
                       "includeSpeakerContext": True, "maxTokens": None}],
        }, handle)
    return path


def _slurm_without_token(monkeypatch, root):
    """A deployment where the privileged gate is live but no token is set."""
    monkeypatch.setenv("STEERLAB_ROOT", str(root))
    monkeypatch.setenv("STEERLAB_EXECUTOR", "slurm")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "none")
    monkeypatch.delenv("STEERLAB_AUTH_TOKEN", raising=False)


def test_mutating_scenario_routes_are_privileged(monkeypatch, tmp_path):
    """Same gate as the _PRIVILEGED_PREFIXES peers: on a Slurm deployment these
    need a bearer token even at the default auth mode."""
    _panel(str(tmp_path))
    _slurm_without_token(monkeypatch, tmp_path)

    assert client.post("/api/scenario/save", json={"name": "x"}).status_code == 503
    assert client.post("/api/scenario/run", json={"path": "p"}).status_code == 503


def test_read_only_scenario_routes_stay_open(monkeypatch, tmp_path):
    """The trailing slash on the prefix is load-bearing — a researcher can still
    LIST and READ panels on a gated deployment without a token."""
    path = _panel(str(tmp_path))
    _slurm_without_token(monkeypatch, tmp_path)

    listing = client.get("/api/scenarios")
    assert listing.status_code == 200
    assert [p["name"] for p in listing.json()["scenarios"]] == ["panel"]

    one = client.get("/api/scenario", params={"path": path})
    assert one.status_code == 200
    assert one.json()["name"] == "panel"


def test_scenarios_listing_is_not_swept_up_by_the_prefix():
    """/api/scenarios must not match the /api/scenario/ prefix — the two differ
    only at the character the trailing slash occupies, so this is exactly the
    kind of thing a later edit to the prefix would break silently."""
    prefixes = app_module._PRIVILEGED_PREFIXES
    assert any(p == "/api/scenario/" for p in prefixes)
    assert not any("/api/scenarios".startswith(p) for p in prefixes)
    assert not any("/api/scenario".startswith(p) for p in prefixes)
    assert any("/api/scenario/save".startswith(p) for p in prefixes)
    assert any("/api/scenario/run".startswith(p) for p in prefixes)
