"""File concurrency metadata never changes scientific manifest bytes."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
from steerlab_server.experiment import manifest_files


def test_two_writers_with_the_same_snapshot_cannot_both_publish(tmp_path):
    path = tmp_path / "experiments" / "example" / "experiment.json"
    path.parent.mkdir(parents=True)
    original = b'{"name":"example", "status":"draft"}\n'
    path.write_bytes(original)
    expected = manifest_files.file_digest(str(path))

    def edit(value):
        try:
            with manifest_files.transaction(str(path), workspace_root=str(tmp_path)):
                manifest_files.require_current(str(path), expected)
                path.write_bytes(value)
            return True
        except manifest_files.StaleManifestError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(edit, [b'{"name":"first"}', b'{"name":"second"}']))
    assert sum(results) == 1


def test_external_digest_does_not_reidentify_or_rewrite_frozen_bytes(tmp_path):
    path = tmp_path / "experiment.json"
    original = b'{ "name": "example", "status":"frozen", "freezeHash":"existing" }\n'
    path.write_bytes(original)
    expected = manifest_files.digest_bytes(original)
    with manifest_files.transaction(str(path), workspace_root=str(tmp_path)):
        manifest_files.require_current(str(path), expected)
    assert path.read_bytes() == original
    assert manifest_files.file_digest(str(path)) == expected


def test_creation_precondition_refuses_an_existing_file(tmp_path):
    path = tmp_path / "experiment.json"
    path.write_bytes(b"{}")
    with pytest.raises(manifest_files.StaleManifestError):
        with manifest_files.transaction(str(path), workspace_root=str(tmp_path)):
            manifest_files.require_current(str(path), None)


def test_lock_is_shared_between_processes_and_reentrant(tmp_path):
    path = tmp_path / "experiment.json"
    marker = tmp_path / "entered"
    program = """\
import sys
from pathlib import Path
from steerlab_server.experiment.manifest_files import transaction
print('ready', flush=True)
with transaction(sys.argv[1], workspace_root=sys.argv[2]):
    Path(sys.argv[3]).write_text('entered')
"""
    process = None
    try:
        with manifest_files.transaction(str(path), workspace_root=str(tmp_path)):
            with manifest_files.transaction(str(path), workspace_root=str(tmp_path)):
                process = subprocess.Popen([sys.executable, "-c", program, str(path), str(tmp_path), str(marker)], stdout=subprocess.PIPE, text=True)
                assert process.stdout.readline().strip() == "ready"
                with pytest.raises(subprocess.TimeoutExpired):
                    process.wait(timeout=0.2)
                assert not marker.exists()
        assert process.wait(timeout=10) == 0
        assert marker.read_text() == "entered"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()


def test_authoring_save_rejects_a_stale_loaded_document(tmp_path):
    import json
    from steerlab_server.experiment import experiment_store as store

    first = store.create("example", model_id="org/model", root=str(tmp_path))
    second = store.load_raw("example", str(tmp_path))
    original_keys = set(first)
    first["experimentDescription"] = "First edit"
    store.save_raw(first, str(tmp_path))
    second["temperature"] = 0.5
    with pytest.raises(manifest_files.StaleManifestError):
        store.save_raw(second, str(tmp_path))
    stored = store.load_raw("example", str(tmp_path))
    assert stored["experimentDescription"] == "First edit"
    assert stored["temperature"] == 0.0
    assert set(json.loads(json.dumps(stored))) == original_keys
    assert "source_digest" not in stored
    assert "source_path" not in stored


def test_plain_document_cannot_overwrite_without_external_precondition(tmp_path):
    from steerlab_server.experiment import experiment_store as store

    original = store.create("example", model_id="org/model", root=str(tmp_path))
    incoming = dict(original)
    incoming["experimentDescription"] = "Reviewed replacement"
    with pytest.raises(manifest_files.StaleManifestError):
        store.save_raw(incoming, str(tmp_path))
    store.save_raw(incoming, str(tmp_path), expected_file_sha256=original.source_digest)
    assert store.load_raw("example", str(tmp_path))["experimentDescription"] == "Reviewed replacement"


def test_http_manifest_etag_guards_replacement_without_a_document_field(tmp_path, monkeypatch):
    from starlette.testclient import TestClient
    from steerlab_server.api.app import app
    from steerlab_server.experiment import experiment_store as store

    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    original = store.create("example", model_id="org/model", root=str(tmp_path))
    client = TestClient(app)
    route = "/api/experiment/example/manifest"
    read = client.get(route)
    path = tmp_path / "experiments" / "example" / "experiment.json"
    assert read.content == path.read_bytes()
    assert read.headers["etag"] == '"' + manifest_files.digest_bytes(read.content) + '"'
    assert client.put(route, json=dict(original)).status_code == 428
    first = dict(original, experimentDescription="first")
    assert client.put(route, json=first, headers={"If-Match": read.headers["etag"]}).status_code == 200
    stale = client.put(route, json=dict(original, temperature=0.8), headers={"If-Match": read.headers["etag"]})
    assert stale.status_code == 412
    assert stale.json()["detail"]["code"] == "staleManifest"
    assert client.get(route).json() == first
    fresh = dict(original, name="fresh")
    assert client.put("/api/experiment/fresh/manifest", json=fresh,
                      headers={"If-None-Match": "*"}).status_code == 200
    assert client.put("/api/experiment/fresh/manifest", json=fresh,
                      headers={"If-None-Match": "*"}).status_code == 412


@pytest.mark.parametrize("copier", [lambda d: d.copy(), deepcopy])
def test_copy_preserves_external_authority_without_json_fields(tmp_path, copier):
    from steerlab_server.experiment import experiment_store as store
    original = store.create("example", model_id="org/model", root=str(tmp_path))
    copied = copier(original)
    assert copied.source_digest == original.source_digest
    assert copied.source_path == original.source_path
    copied["experimentDescription"] = "Reviewed edit"
    saved = store.save_raw(copied, str(tmp_path))
    assert saved["experimentDescription"] == "Reviewed edit"
    assert set(saved) == set(original)
    assert "source_digest" not in saved
