import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from steerlab_server.client.runtime_identity import source_sha256


def test_copied_release_payload_runs_without_checkout_and_refuses_source_drift(tmp_path):
    package = Path(__file__).resolve().parents[1] / 'steerlab_server'
    payload = tmp_path / 'ServerPayload'
    copied = payload / 'steerlab_server'
    shutil.copytree(package, copied, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    expected = source_sha256()
    assert source_sha256(copied) == expected
    document = dict(action='interview', payload=dict(workspaceRoot=str(tmp_path), operation='optvec-gradient'), clientSHA256=expected)

    def run():
        return subprocess.run([sys.executable, '-B', '-s', '-m', 'steerlab_server.client.diagnostic_workspace'],
                              input=json.dumps(document), text=True, capture_output=True, cwd=tmp_path,
                              env={**os.environ, 'PYTHONPATH': str(payload)})

    result = run()
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['result']['id'] == 'optvec-gradient'
    assert not list(payload.rglob('__pycache__'))
    # A source change must refuse BEFORE dispatch/import of the changed owner.
    source = copied / 'experiment/method_authoring.py'
    source.write_text('raise RuntimeError("changed owner must not run")\n' + source.read_text())
    (copied / 'experiment/diagnostic_archives.py').write_text('raise RuntimeError("changed archive owner must not import")\n')
    result = run()
    assert result.returncode == 65
    response = json.loads(result.stdout)
    assert response['ok'] is False and 'sources differ' in response['reason']
    assert 'same reviewed source' in response['repairAction']
    assert 'changed owner must not run' not in response['reason']
    assert not (tmp_path / '.steerlab').exists()


def test_missing_client_identity_refuses_before_dispatch(tmp_path):
    from steerlab_server.client import diagnostic_workspace
    from unittest.mock import patch
    import io
    request = json.dumps(dict(action='import', payload={'workspaceRoot': str(tmp_path / 'missing')}))
    output = io.StringIO()
    with patch('sys.stdin', io.StringIO(request)), patch('sys.stdout', output):
        assert diagnostic_workspace.main() == 65
    assert json.loads(output.getvalue())['ok'] is False
    assert not (tmp_path / 'missing').exists()


def test_compiled_client_identity_matches_current_source():
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, str(root / 'scripts/ci/check-python-client-identity.py')], check=True)
