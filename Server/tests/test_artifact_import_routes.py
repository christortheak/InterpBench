"""File staging, owner invocation, and workbench authority for custom imports."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from safetensors.numpy import save_file

from steerlab_server.api.artifact_import_routes import build_router


def test_staged_lens_reaches_the_existing_library(tmp_path, monkeypatch):
    monkeypatch.setenv('STEERLAB_ROOT', str(tmp_path))
    jobs = []
    def submit(kind, work):
        jobs.append(work)
        return SimpleNamespace(id='import-job')
    app = FastAPI(); app.include_router(build_router(SimpleNamespace(jobs=SimpleNamespace(submit=submit))))
    client = TestClient(app)
    source = 'a' * 32
    original = tmp_path / 'original.safetensors'
    save_file({'L1': np.eye(2, dtype=np.float32)}, str(original))
    spec = {'schemaVersion': 1, 'kind': 'jlens', 'modelID': 'example/model',
            'hiddenSize': 2, 'layerCount': 3, 'tensorFile': 'nested/weights.safetensors',
            'lens': {'targetLayer': 2, 'layers': {'1': 'L1'}}}
    for name, data in [('import.json', json.dumps(spec).encode()), ('nested/weights.safetensors', original.read_bytes())]:
        response = client.post(f'/api/artifact-imports/stage/{source}/{name}', content=data,
                               headers={'X-Content-SHA256': hashlib.sha256(data).hexdigest()})
        assert response.status_code == 200, response.text
    reference = f'.steerlab/artifact-inputs/{source}/import.json'
    review = client.post('/api/artifact-imports/plan', json={'descriptionFile': reference})
    assert review.status_code == 200, review.text
    request = {'descriptionFile': reference, 'planSHA256': review.json()['planSHA256']}
    response = client.post('/api/artifact-imports/import', json=request)
    assert response.json() == {'jobId': 'import-job'}
    result = jobs[0](SimpleNamespace(log=lambda _: None))
    from steerlab_server.jlens import lens_store
    assert lens_store.resolve(result['lensID'], str(tmp_path)).sourceLayers == [1]
    # No overwrite, even for a repeat upload of the same original bytes.
    duplicate = client.post(f'/api/artifact-imports/stage/{source}/import.json', content=json.dumps(spec).encode(),
                            headers={'X-Content-SHA256': hashlib.sha256(json.dumps(spec).encode()).hexdigest()})
    assert duplicate.status_code == 400


def test_staging_bad_hash_and_external_description_publish_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv('STEERLAB_ROOT', str(tmp_path))
    app = FastAPI(); app.include_router(build_router(SimpleNamespace()))
    client = TestClient(app)
    response = client.post('/api/artifact-imports/stage/' + 'a'*32 + '/bad.json', content=b'changed',
                           headers={'X-Content-SHA256': '0'*64})
    assert response.status_code == 400
    assert not (tmp_path / '.steerlab/artifact-inputs' / ('a'*32) / 'bad.json').exists()
    response = client.post('/api/artifact-imports/plan', json={'descriptionFile': '/outside/import.json'})
    assert response.status_code == 400
    assert 'Stage' in response.text


def test_real_app_protects_the_staging_route(tmp_path, monkeypatch):
    from steerlab_server.api.app import app
    monkeypatch.setenv('STEERLAB_ROOT', str(tmp_path))
    monkeypatch.setenv('STEERLAB_SERVICE_ROLE', 'runner')
    response = TestClient(app).post('/api/artifact-imports/stage/' + 'a'*32 + '/file.json')
    assert response.status_code == 403
    monkeypatch.setenv('STEERLAB_SERVICE_ROLE', 'workbench')
    monkeypatch.setenv('STEERLAB_AUTH_MODE', 'token')
    monkeypatch.setenv('STEERLAB_AUTH_TOKEN', 'fictional-test-token')
    monkeypatch.delenv('STEERLAB_DEV_OPEN_LOOPBACK', raising=False)
    response = TestClient(app).post('/api/artifact-imports/stage/' + 'a'*32 + '/file.json')
    assert response.status_code == 401


def test_external_transfer_policy_is_checked_before_staging(tmp_path, monkeypatch):
    monkeypatch.setenv('STEERLAB_ROOT', str(tmp_path))
    monkeypatch.setenv('STEERLAB_TRANSFER_METHOD', 'external')
    app = FastAPI(); app.include_router(build_router(SimpleNamespace()))
    response = TestClient(app).post('/api/artifact-imports/stage/' + 'a'*32 + '/source.json',
        content=b'{}', headers={'X-Content-SHA256': hashlib.sha256(b'{}').hexdigest()})
    assert response.status_code == 403
    assert response.json()['detail']['code'] == 'external_transfer_required'
    assert not (tmp_path / '.steerlab').exists()
