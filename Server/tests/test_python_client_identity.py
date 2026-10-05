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
    # Both sides, named: what the Mac build expects, what these files are,
    # and where they are. The repair leads with a fresh start, never with
    # installing dependencies, and says no server takes part.
    changed = source_sha256(copied)
    assert expected[:12] in response['reason'] and changed[:12] in response['reason']
    assert response['clientSHA256'] == changed
    assert response['clientRoot'] == str(copied.resolve())
    assert str(copied.resolve()) in response['reason']
    assert response['repairAction'].startswith('Run the command again from a fresh start')
    assert 'No server takes part' in response['repairAction']
    assert 'dependencies' not in response['repairAction']
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


def test_identity_check_confirms_without_importing_any_owner(tmp_path):
    """The bridge's `client-identity` action answers the identity and does
    nothing else: no owner is imported and no workspace is read, so the Mac
    can ask it before a long remote step."""
    package = Path(__file__).resolve().parents[1] / 'steerlab_server'
    payload = tmp_path / 'ServerPayload'
    copied = payload / 'steerlab_server'
    shutil.copytree(package, copied, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # An owner that would fail loudly if the check imported it.
    (copied / 'experiment/diagnostic_archives.py').write_text('raise RuntimeError("owner imported")\n')
    identity = source_sha256(copied)

    def ask(document):
        result = subprocess.run([sys.executable, '-B', '-s', '-m', 'steerlab_server.client.diagnostic_workspace'],
                                input=json.dumps(document), text=True, capture_output=True, cwd=tmp_path,
                                env={**os.environ, 'PYTHONPATH': str(payload)})
        return result.returncode, json.loads(result.stdout)

    code, response = ask(dict(action='client-identity', payload={}, clientSHA256=identity))
    assert code == 0, response
    assert response == {'ok': True, 'clientSHA256': identity, 'clientRoot': str(copied.resolve()),
                        'result': {'changed': False}}
    code, response = ask(dict(action='client-identity', payload={}, clientSHA256='0' * 64))
    assert code == 65 and response['ok'] is False
    assert response['clientSHA256'] == identity and 'sources differ' in response['reason']
    code, response = ask(dict(action='client-identity', payload={'x': 1}, clientSHA256=identity))
    assert code == 65 and 'empty payload' in response['reason']
