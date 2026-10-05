import json
from pathlib import Path
import subprocess
import sys
import pytest
from steerlab_server.client import workspace_bootstrap as owner
from steerlab_server.experiment.manifest_errors import ExperimentStoreError


def test_cli_creates_complete_seed_and_handoff_without_existing_workspace(tmp_path, capsys, monkeypatch):
    from steerlab_server import client_cli
    monkeypatch.delenv('STEERLAB_WORKSPACE', raising=False)
    root = tmp_path / 'new-workspace'
    assert client_cli.main(['workspace','init',str(root),'--no-git','--json']) == 0
    result = json.loads(capsys.readouterr().out)['result']
    assert result['missingSeedFiles'] == [] and result['changed'] is True
    for name in owner.manifest()['seedFiles']:
        assert (root / name).read_bytes() == (owner.SEED / name).read_bytes()
    assert (root / 'AGENTS.md').read_text() == owner.agent_contents()
    assert client_cli.main(['workspace','handoff','--root',str(root),'--json']) == 0
    report = json.loads(capsys.readouterr().out)['result']
    assert report['agentGuide'] == str(root / 'AGENTS.md')
    expected = Path(sys.executable).with_name('steerlab')
    assert report['executable'] == ([str(expected)] if expected.is_file() else [sys.executable, '-m', 'steerlab_server.client_cli'])
    assert not (root / '.git').exists()


def test_handoff_leads_with_the_interview_then_the_brief_catalog(tmp_path, capsys, monkeypatch):
    """What a coding assistant is handed first. The old handoff sent it to
    `--help` and then to the full 75 KB method catalog, and never mentioned
    the study interview, which is where a new researcher actually starts."""
    from steerlab_server import client_cli
    monkeypatch.delenv('STEERLAB_WORKSPACE', raising=False)
    root = tmp_path / 'workspace'
    assert client_cli.main(['workspace', 'init', str(root), '--no-git', '--json']) == 0
    created = json.loads(capsys.readouterr().out)
    # A new workspace points at the interview, not at `experiment create`.
    assert created['nextAction']['verb'] == 'authoring study <intent>'
    for intent in ('conceptStudy', 'agentComparison', 'multiAgent'):
        assert intent in created['nextAction']['detail']
    assert f'--root {created["result"]["workspaceRoot"]}' in created['nextAction']['detail']

    assert client_cli.main(['workspace', 'handoff', '--root', str(root), '--json']) == 0
    report = json.loads(capsys.readouterr().out)['result']
    command, where = report['executable'], ['--root', report['workspaceRoot'], '--json']
    assert report['discovery'] == [
        command + ['authoring', 'study', '<intent>'] + where,
        command + ['science', 'list', '--brief'] + where,
        command + ['--help']]
    assert [intent['id'] for intent in report['studyIntents']] == ['conceptStudy', 'agentComparison', 'multiAgent']
    assert all(intent['purpose'].endswith('.') for intent in report['studyIntents'])
    instructions = report['instructions']
    assert instructions == owner.HANDOFF_INSTRUCTIONS
    assert "Work at the researcher's level" in instructions
    assert 'Ask before anything that spends compute or money' in instructions
    assert 'study interview' in instructions and 'studyIntents' in instructions
    assert report['nextAction'] == owner.HANDOFF_NEXT_ACTION
    assert set(report) == {'workspaceRoot', 'recognized', 'agentGuidePresent', 'missingSeedFiles', 'seedSchemaVersion', 'changed',
                           'executable', 'agentGuide', 'instructions', 'studyIntents', 'discovery', 'nextAction'}

    # The commands it names are real: each intent's interview is emitted, and
    # the placeholder itself is a usage refusal that names the three intents.
    for intent in report['studyIntents']:
        assert client_cli.main(['authoring', 'study', intent['id'], *where]) == 0
        assert json.loads(capsys.readouterr().out)['result']['intent'] == intent['id']
    assert client_cli.main(['authoring', 'study', '<intent>', *where]) == 64
    assert 'conceptStudy' in json.loads(capsys.readouterr().out)['error']['reason']
    assert client_cli.main(['science', 'list', '--brief', *where]) == 0
    brief = capsys.readouterr().out
    assert json.loads(brief)['result']['brief'] is True and len(brief.encode()) < 12_000

    # `setup start` carries the same block.
    assert client_cli.main(['setup', 'start', str(root), '--json']) == 0
    assert json.loads(capsys.readouterr().out)['result']['handoff'] == report


def test_incomplete_seed_existing_data_and_symlink_destinations_refuse(tmp_path):
    target = tmp_path / 'existing'; target.mkdir(); (target/'keep').write_text('original')
    with pytest.raises(ExperimentStoreError): owner.initialize(target)
    assert (target/'keep').read_text() == 'original'
    link = tmp_path/'link'; link.symlink_to(target)
    with pytest.raises(ExperimentStoreError): owner.initialize(link)
    empty_seed = tmp_path/'empty-seed'; empty_seed.mkdir()
    absent = tmp_path/'absent-parent'/'workspace'
    with pytest.raises(ExperimentStoreError): owner.initialize(absent, seed=empty_seed)
    assert not absent.parent.exists()


def test_empty_destination_supported_and_missing_guide_is_never_rewritten(tmp_path):
    target = tmp_path/'workspace';target.mkdir()
    owner.initialize(target, use_git=False)
    guide = target/'AGENTS.md';guide.write_text('Researcher instructions')
    assert owner.handoff(target)['agentGuidePresent'] is True
    assert guide.read_text() == 'Researcher instructions'
    guide.unlink()
    with pytest.raises(ExperimentStoreError): owner.handoff(target)
    assert not guide.exists()


def test_packaged_seed_gate_and_bootstrap_stay_light():
    root=Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable,str(root/'scripts/ci/check-workspace-bootstrap.py')],check=True)
    code="from steerlab_server.client import workspace_bootstrap; import sys; workspace_bootstrap.manifest(); assert not ({'torch','transformers','fastapi'} & sys.modules.keys())"
    subprocess.run([sys.executable,'-c',code],check=True)


def test_clean_mac_skips_optional_git_without_invoking_installation_shim(monkeypatch):
    from types import SimpleNamespace
    from steerlab_server.client import workspace_bootstrap as owner
    calls = []
    monkeypatch.setattr(owner.sys, 'platform', 'darwin')
    monkeypatch.setattr(owner.shutil, 'which', lambda _: '/usr/bin/git')
    def selected(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(owner.subprocess, 'run', selected)
    assert owner.available_git() is None
    assert calls == [['/usr/bin/xcode-select', '-p']]
