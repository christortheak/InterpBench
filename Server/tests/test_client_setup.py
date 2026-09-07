import json
import os
from pathlib import Path
import subprocess
import sys
from steerlab_server.client import setup, workspace_bootstrap


def cli(*args, **environment):
    env = {**os.environ, **environment}
    env.pop('STEERLAB_WORKSPACE', None)
    process = subprocess.run([sys.executable, '-m', 'steerlab_server.client_cli', *args, '--json'], capture_output=True, text=True, env=env)
    return process.returncode, json.loads(process.stdout)


def test_readiness_distinguishes_client_workspace_and_execution(tmp_path):
    rootless = setup.inspect()
    assert rootless['clientReady'] and not rootless['authoringReady']
    assert rootless['workspace'] is None
    assert rootless['execution']['state'] == 'notAssessed'
    workspace_bootstrap.initialize(tmp_path / 'workspace', use_git=False)
    report = setup.inspect(tmp_path / 'workspace')
    assert report['authoringReady']
    assert not report['execution']['requiredForAuthoring']
    (tmp_path / 'workspace/AGENTS.md').unlink()
    assert not setup.inspect(tmp_path / 'workspace')['authoringReady']


def test_cli_uses_explicit_workspace_and_never_checkout_fallback(tmp_path):
    workspace_bootstrap.initialize(tmp_path / 'workspace', use_git=False)
    code, report = cli('setup', 'inspect', '--root', str(tmp_path / 'workspace'))
    assert code == 0 and report['result']['authoringReady'], report
    code, report = cli('setup', 'inspect', STEERLAB_ROOT=str(tmp_path / 'workspace'))
    assert code == 0 and report['result']['workspace'] is None
    missing = tmp_path / 'missing'
    code, report = cli('setup', 'inspect', '--root', str(missing))
    assert code == 66 and not missing.exists()


def test_apply_cannot_skip_review_or_supply_unrecognized_flags(tmp_path):
    code, report = cli('setup', 'apply', '--yes')
    assert code != 0 and not report['changed']
    code, report = cli('setup', 'plan', '--force')
    assert code == 64 and not report['changed']


def test_process_readiness_owner_rejects_extra_fields():
    import pytest
    from steerlab_server.client.diagnostic_commands import workspace_action
    with pytest.raises(ValueError):
        workspace_action('setup-inspect', {'downloadModels': True})
    assert workspace_action('setup-inspect', {})['clientReady']


def test_start_requires_explicit_creation_and_preserves_existing_workspace(tmp_path):
    root = tmp_path / 'workspace'
    code, result = cli('setup', 'start', str(root))
    assert code == 65 and not root.exists()
    assert '--create' in result['error']['repairAction']
    code, result = cli('setup', 'start', str(root), '--create')
    assert code == 0 and result['changed'] and result['result']['readiness']['authoringReady'], result
    before = (root / 'AGENTS.md').read_bytes()
    code, result = cli('setup', 'start', str(root))
    assert code == 0 and not result['changed'], result
    assert result['result']['handoff']['agentGuide'] == str(root / 'AGENTS.md')
    code, result = cli('setup', 'start', str(root), '--create')
    assert code == 65 and (root / 'AGENTS.md').read_bytes() == before
