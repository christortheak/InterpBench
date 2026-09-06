"""Runner authority refuses authoring while retaining execution and auth gates."""
import pytest
from starlette.testclient import TestClient
from steerlab_server.api.app import app
from steerlab_server.api.route_roles import CENSUS, Role
from steerlab_server.api.service_authority import refusal


def test_every_workbench_operation_is_refused_by_runner(monkeypatch):
    monkeypatch.setenv("STEERLAB_SERVICE_ROLE", "runner")
    client = TestClient(app)
    for entry in CENSUS:
        if entry.role == Role.WORKBENCH:
            response = client.request(entry.method, entry.path, follow_redirects=False)
            assert response.status_code == 403, entry.key
            assert response.json()["detail"]["code"] == "workbench_required", entry.key


def test_workbench_keeps_all_declared_operations(monkeypatch):
    monkeypatch.setenv("STEERLAB_SERVICE_ROLE", "workbench")
    for entry in CENSUS:
        assert refusal(entry.method, entry.path) is None, entry.key


def test_runner_preserves_execution_and_derived_artifact_declarations(monkeypatch):
    monkeypatch.setenv("STEERLAB_SERVICE_ROLE", "runner")
    for entry in CENSUS:
        if entry.role in {Role.RUNNER, Role.BOTH}:
            assert refusal(entry.method, entry.path) is None, entry.key


def test_invalid_role_refuses_and_runner_cannot_disable_auth(monkeypatch):
    monkeypatch.setenv("STEERLAB_SERVICE_ROLE", "unknown")
    assert refusal("GET", "/healthz")["code"] == "invalid_service_role"
    monkeypatch.setenv("STEERLAB_SERVICE_ROLE", "runner")
    monkeypatch.setenv("STEERLAB_AUTH_MODE", "token")
    monkeypatch.setenv("STEERLAB_AUTH_TOKEN", "fictional-test-token")
    monkeypatch.delenv("STEERLAB_DEV_OPEN_LOOPBACK", raising=False)
    response = TestClient(app).post("/api/studies/submit-bundle", json={"verb": "verify"})
    assert response.status_code == 401


def test_dynamic_authoring_route_is_refused_before_manifest_io(monkeypatch):
    monkeypatch.setenv("STEERLAB_SERVICE_ROLE", "runner")
    response = TestClient(app).put("/api/experiment/example/manifest", json={})
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "workbench_required"
