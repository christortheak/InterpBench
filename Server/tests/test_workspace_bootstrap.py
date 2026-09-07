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
    assert report['executable'][0] == sys.executable
    assert not (root / '.git').exists()


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
