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


def test_older_environment_reports_missing_corpus_tools_but_can_start(tmp_path, monkeypatch):
    from steerlab_server import client_dependencies as dependencies
    real_run = dependencies.subprocess.run
    def older_probe(command, **kwargs):
        if command[1:3] == ['-I','-c']:
            # Execute the real probe with the newer dependencies made unavailable.
            guard = """
import importlib.abc,sys
class MissingCorpus(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyarrow','huggingface_hub','transformers'}:
            raise ModuleNotFoundError('Not installed in older client: '+fullname)
sys.meta_path.insert(0,MissingCorpus())
"""
            command = [*command[:3], guard + command[3], *command[4:]]
        return real_run(command, **kwargs)
    monkeypatch.setattr(dependencies.subprocess, 'run', older_probe)
    report = setup.inspect()
    assert not report['clientReady'] and report['basicClientReady']
    assert report['capabilities']['basicAuthoring']['ready']
    assert all(not report['capabilities'][name]['ready'] for name in ('parquet','huggingFace','tokenPreview'))
    assert 'Research Setup' in report['repairAction']
    result = setup.start(tmp_path/'workspace', create=True)
    assert result['readiness']['authoringReady'] and not result['readiness']['clientReady']
    assert (tmp_path/'workspace/AGENTS.md').exists()


def test_probe_handles_installed_but_broken_package(tmp_path, monkeypatch):
    from steerlab_server import client_dependencies as dependencies
    real_run = dependencies.subprocess.run
    def broken_probe(command, **kwargs):
        guard = """
import importlib.abc,sys
class BrokenParquet(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pyarrow': raise OSError('Binary extension could not load')
sys.meta_path.insert(0,BrokenParquet())
"""
        return real_run([*command[:3], guard+command[3], *command[4:]], **kwargs)
    monkeypatch.setattr(dependencies.subprocess, 'run', broken_probe)
    report = setup.inspect()
    assert report['dependencies']['pyarrow'] is not None
    assert not report['clientReady'] and report['basicClientReady']
    assert 'Binary extension' in report['capabilities']['parquet']['diagnostic']


def test_probe_timeout_is_a_readiness_report(monkeypatch):
    from steerlab_server import client_dependencies as dependencies
    def timeout(*args, **kwargs): raise subprocess.TimeoutExpired('client-probe',30)
    monkeypatch.setattr(dependencies.subprocess, 'run', timeout)
    report = setup.inspect()
    assert not report['clientReady'] and not report['basicClientReady']
    assert 'Research Setup' in report['repairAction']


def test_the_app_free_handoff_says_what_this_client_does(tmp_path):
    """Release review decision 1: the app-free client authors and submits;
    the Mac app is the supported route to running and reading results."""
    root = tmp_path / 'workspace'
    code, result = cli('setup', 'start', str(root), '--create')
    assert code == 0, result
    scope = result['result']['clientScope']
    assert scope == setup.CLIENT_SCOPE
    for phrase in ('authors studies', 'submits them to a runner', 'does not run models itself', 'results export', 'Mac'):
        assert phrase in scope
    code, result = cli('workspace', 'handoff', '--root', str(root))
    assert code == 0, result
    assert setup.CLIENT_SCOPE_SHORT in result['message']
    # The handoff object itself stays the same on both clients (see
    # WorkspaceBootstrapParityTests), so the statement is not a new key there.
    assert 'clientScope' not in result['result']


def test_an_installer_refusal_keeps_its_typed_code(tmp_path):
    release = tmp_path / 'release'
    release.mkdir()
    (release / 'install-client.sh').write_text(
        'printf \'{"ok":false,"changed":false,"code":"noNetwork","reason":"The download server could not be reached.",'
        '"repairAction":"Check the internet connection, then review a fresh plan and retry."}\\n\'\nexit 70\n')
    code, report = cli('setup', 'plan', '--release', str(release), '--runtime', str(tmp_path / 'client-runtime'))
    assert code == 65 and not report['changed']
    assert report['error']['code'] == 'clientSetupRefused'
    assert report['result']['installerCode'] == 'noNetwork'
    assert 'could not be reached' in report['error']['reason']
