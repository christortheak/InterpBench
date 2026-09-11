"""Transport budgets and controller identity remain honest after a code push."""
import json
from pathlib import Path
import httpx
import pytest
from steerlab_server.client.runner import RunnerClient


def test_export_and_stage_have_long_idle_budget_without_changing_other_calls():
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={})
    with RunnerClient(base_url='http://localhost', http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        client.export_diagnostic('example-job')
        client.stage_diagnostic('/runs/input.tar.gz', 'a'*64)
        client.job('example-job')
    assert [r.extensions['timeout']['read'] for r in requests[:2]] == [3600,3600]
    assert requests[2].extensions['timeout']['read'] != 3600


def test_explicit_export_timeout_is_honored():
    def handle(request):
        assert request.extensions['timeout']['read'] == 12
        return httpx.Response(200, json={})
    with RunnerClient(base_url='http://localhost', timeout=12, http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        client.export_diagnostic('example-job')


def test_deployed_identity_changes_without_relabelling_running_controller(monkeypatch):
    from steerlab_server import build_identity
    from steerlab_server.api import profile
    running = profile.RUNNING_ENGINE_VERSION
    monkeypatch.setattr(build_identity, '_git_identity', lambda: None)
    monkeypatch.setattr(build_identity, '_file_identity', lambda: 'new-deployment')
    monkeypatch.setenv('STEERLAB_BUILD_COMMIT', 'old-process-override')
    assert build_identity.deployed_commit() == 'new-deployment'
    assert profile.RUNNING_ENGINE_VERSION == running



def test_export_timeout_explains_retry_without_resubmission():
    from steerlab_server.client.runner import RunnerUnreachable
    def handle(request): raise httpx.ReadTimeout('idle export', request=request)
    with RunnerClient(base_url='http://localhost', http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        with pytest.raises(RunnerUnreachable) as failure: client.export_diagnostic('example-job')
    assert 'Do not resubmit' in failure.value.repair_action


def test_download_budget_does_not_remove_hash_verification(tmp_path):
    import hashlib
    content = b'complete evidence fixture'
    def handle(request):
        if request.url.path == '/api/capabilities': return httpx.Response(200, json={})
        assert request.extensions['timeout']['read'] == 3600
        return httpx.Response(200, content=content)
    with RunnerClient(base_url='http://localhost', http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        result = client.download_bundle(remote_path='/runs/evidence.tar.gz', expected_sha256=hashlib.sha256(content).hexdigest(), destination=str(tmp_path/'evidence.tar.gz'), request_timeout=client.diagnostic_timeout)
    assert result['verified'] and (tmp_path/'evidence.tar.gz').read_bytes() == content
