"""Refuse ambiguous operations and prohibited transfers before side effects."""
import pytest
from starlette.testclient import TestClient
from steerlab_server.api.app import app


@pytest.mark.parametrize("route", ["/api/studies/submit", "/api/studies/submit-bundle"])
@pytest.mark.parametrize("verb", [None, "", " ", 4])
def test_submission_requires_operation(route, verb, monkeypatch):
    monkeypatch.delenv("STEERLAB_SERVER_ROLE", raising=False)
    response = TestClient(app).post(route, json={"verb": verb})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "operation_required"
    assert response.json()["detail"]["repairAction"]


@pytest.mark.parametrize("method,route", [
    ("POST", "/api/bundles/upload"),
    ("GET", "/api/bundles/download?path=missing.tar.gz"),
])
def test_external_transfer_refused_before_reading_or_creating_files(
        method, route, tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_TRANSFER_METHOD", "rsync")
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    response = TestClient(app).request(method, route)
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "external_transfer_required"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("operation", ["upload", "download"])
@pytest.mark.parametrize("policy", [
    {"httpTransfer": False}, {"externalTransferRequired": True},
])
def test_python_client_refuses_policy_before_bulk_request(operation, policy, tmp_path):
    import httpx
    from steerlab_server.client.runner import RunnerClient, RunnerRefusal

    calls = []

    def respond(request):
        calls.append(request.url.path)
        assert request.url.path == "/api/capabilities"
        return httpx.Response(200, json={"remoteStudy": policy})

    http = httpx.Client(transport=httpx.MockTransport(respond))
    with RunnerClient(base_url="http://runner.invalid", http_client=http) as client:
        with pytest.raises(RunnerRefusal) as refused:
            if operation == "upload":
                client.upload_run_bundle(str(tmp_path / "nonexistent.tar.gz"))
            else:
                client.download_bundle(remote_path="/runs/evidence.tar.gz",
                                       expected_sha256="a" * 64,
                                       destination=str(tmp_path / "evidence.tar.gz"))
        assert refused.value.code == "externalTransferRequired"
        assert refused.value.repair_action
    assert calls == ["/api/capabilities"]
    assert not list(tmp_path.iterdir())


def test_engine_cli_requires_operation_before_opening_store(tmp_path, monkeypatch, capsys):
    from steerlab_server import cli
    monkeypatch.setenv("STEERLAB_METADATA_ROOT", str(tmp_path))
    assert cli._study(["submit", "example"]) == 64
    assert "requires --verb" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
