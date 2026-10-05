"""`steerlab experiment rename|delete`, `design rename|delete`, `agent delete`.

Each previews by default, applies only with the reviewed digest and `--yes`,
moves deletions into a `.trash-<time>` folder instead of erasing them, and
refuses frozen studies and agents a study uses. Same keys as the Mac.
"""
import json
import os
from pathlib import Path

import pytest

from steerlab_server import client_cli
from steerlab_server.client import housekeeping_commands
from steerlab_server.experiment import experiment_store as store


@pytest.fixture(autouse=True)
def restore_cli_root(monkeypatch):
    if 'STEERLAB_ROOT' in os.environ:
        monkeypatch.setenv('STEERLAB_ROOT', os.environ['STEERLAB_ROOT'])
    else:
        monkeypatch.delenv('STEERLAB_ROOT', raising=False)


def cli(root, capsys, *args):
    code = client_cli.main([*args, '--root', str(root), '--json'])
    return code, json.loads(capsys.readouterr().out)


def make_study(root, name='first'):
    return store.create(name, model_id='test/model', root=str(root))


def set_status(root, name, status):
    path = root / 'experiments' / name / 'experiment.json'
    document = json.loads(path.read_text())
    document['status'] = status
    path.write_text(json.dumps(document))


def make_template(root, name='wave'):
    folder = root / 'templates' / name
    folder.mkdir(parents=True)
    study = {'name': 'source', 'createdAt': '2026-10-05T00:00:00Z', 'modelID': 'test/model',
             'experimentDescription': '', 'status': 'draft', 'concepts': [], 'conditions': [],
             'variantConditions': []}
    (folder / 'template.json').write_text(json.dumps(
        {'schemaVersion': 1, 'name': name, 'createdAt': '2026-10-05T00:00:00Z',
         'templateDescription': 'd', 'study': study}))


def make_agent(root, slug='helper'):
    folder = root / 'runs' / 'model-variants' / slug
    folder.mkdir(parents=True)
    artifact = {'name': slug, 'baseModelID': 'test/model', 'promptMode': 'chatAssistant',
                'qwenThinkingEnabled': False, 'temperature': 0, 'systemPrompt': 'Be brief.',
                'injections': [], 'adapters': [], 'createdAt': '2026-10-05T00:00:00Z'}
    (folder / 'model-variant.json').write_text(json.dumps(artifact))
    return f'runs/model-variants/{slug}/model-variant.json'


def trash(parent):
    return [p for p in parent.iterdir() if p.name.startswith('.trash-')]


# --- studies ----------------------------------------------------------------------


def test_experiment_rename_previews_then_applies_with_the_reviewed_digest(tmp_path, capsys):
    make_study(tmp_path)
    code, preview = cli(tmp_path, capsys, 'experiment', 'rename', 'first', 'Second Name')
    assert code == 0 and preview['changed'] is False
    result = preview['result']
    assert result['applied'] is False and result['newName'] == 'second-name'
    digest = result['manifestFileSHA256']
    assert result['confirmCommand'] == (
        f"steerlab experiment rename first 'Second Name' --manifest-sha256 {digest} --yes")
    assert preview['nextAction']['missingPermissionFlags'] == ['--manifest-sha256', '--yes']
    assert (tmp_path / 'experiments/first/experiment.json').exists()

    code, applied = cli(tmp_path, capsys, 'experiment', 'rename', 'first', 'Second Name',
                        '--manifest-sha256', digest, '--yes')
    assert code == 0 and applied['changed'] is True
    assert applied['result']['applied'] is True
    assert applied['result']['destination'] == 'experiments/second-name'
    assert store.load_raw('second-name', str(tmp_path))['name'] == 'second-name'
    assert not (tmp_path / 'experiments/first').exists()


def test_experiment_rename_refuses_stale_frozen_and_unconfirmed(tmp_path, capsys):
    stale = make_study(tmp_path, 'draft').source_digest
    document = store.load_raw('draft', str(tmp_path))
    document['experimentDescription'] = 'changed elsewhere'
    store.save_raw(document, str(tmp_path))
    code, refused = cli(tmp_path, capsys, 'experiment', 'rename', 'draft', 'other',
                        '--manifest-sha256', stale, '--yes')
    assert code == 65 and refused['error']['code'] == 'staleManifest'
    assert (tmp_path / 'experiments/draft').exists()

    make_study(tmp_path, 'kept')
    set_status(tmp_path, 'kept', 'frozen')
    code, frozen = cli(tmp_path, capsys, 'experiment', 'rename', 'kept', 'renamed')
    assert code == 65 and frozen['error']['gate'] == 'statusImmutable'
    assert 'steerlab experiment duplicate kept kept-v2' in frozen['error']['repairAction']
    assert 'steerlab-cli' not in frozen['error']['repairAction']

    code, unconfirmed = cli(tmp_path, capsys, 'experiment', 'rename', 'draft', 'other', '--yes')
    assert code == 64 and unconfirmed['error']['code'] == 'usage'


def test_experiment_delete_moves_the_draft_to_trash_and_leaves_runs_alone(tmp_path, capsys):
    make_study(tmp_path, 'doomed')
    run = tmp_path / 'runs/20261005-doomed-run'
    run.mkdir(parents=True)
    stamped = json.dumps({'name': 'doomed', 'status': 'draft'}).encode()
    (run / 'experiment.json').write_bytes(stamped)

    code, preview = cli(tmp_path, capsys, 'experiment', 'delete', 'doomed')
    assert code == 0 and preview['result']['runsRecordingName'] == 1
    assert (tmp_path / 'experiments/doomed').exists()

    code, applied = cli(tmp_path, capsys, 'experiment', 'delete', 'doomed', '--manifest-sha256',
                        preview['result']['manifestFileSHA256'], '--yes')
    assert code == 0
    destination = applied['result']['destination']
    assert destination.startswith('experiments/.trash-') and destination.endswith('/doomed')
    assert (tmp_path / destination / 'experiment.json').exists()
    assert not (tmp_path / 'experiments/doomed').exists()
    assert (run / 'experiment.json').read_bytes() == stamped
    # The trash folder is not listed as an unreadable study.
    code, listed = cli(tmp_path, capsys, 'experiment', 'list')
    assert listed['result']['count'] == 0


def test_experiment_delete_refuses_a_complete_study(tmp_path, capsys):
    make_study(tmp_path, 'kept')
    set_status(tmp_path, 'kept', 'complete')
    code, refused = cli(tmp_path, capsys, 'experiment', 'delete', 'kept')
    assert code == 65 and refused['error']['gate'] == 'statusImmutable'
    assert 'cannot be deleted' in refused['error']['reason']
    assert (tmp_path / 'experiments/kept').exists()


# --- templates --------------------------------------------------------------------


def test_design_rename_and_delete_with_preview_and_confirmation(tmp_path, capsys):
    make_template(tmp_path)
    code, preview = cli(tmp_path, capsys, 'design', 'rename', 'wave', 'wave-two')
    assert code == 0 and preview['result']['kind'] == 'template'
    first = preview['result']['designFileSHA256']
    assert (tmp_path / 'templates/wave/template.json').exists()

    code, renamed = cli(tmp_path, capsys, 'design', 'rename', 'wave', 'wave-two',
                        '--file-sha256', first, '--yes')
    assert code == 0
    moved = json.loads((tmp_path / 'templates/wave-two/template.json').read_text())
    assert moved['name'] == 'wave-two'

    code, stale = cli(tmp_path, capsys, 'design', 'delete', 'wave-two', '--file-sha256', first, '--yes')
    assert code == 65 and stale['error']['code'] == 'designChanged'

    code, preview = cli(tmp_path, capsys, 'design', 'delete', 'wave-two')
    code, deleted = cli(tmp_path, capsys, 'design', 'delete', 'wave-two', '--file-sha256',
                        preview['result']['designFileSHA256'], '--yes')
    assert code == 0
    destination = deleted['result']['destination']
    assert destination.startswith('templates/.trash-')
    assert (tmp_path / destination / 'template.json').exists()
    code, listed = cli(tmp_path, capsys, 'design', 'list')
    assert listed['result']['catalog'] == {'entries': [], 'issues': []}

    code, missing = cli(tmp_path, capsys, 'design', 'delete', 'nowhere')
    assert code == 66


# --- agents -----------------------------------------------------------------------


def test_agent_delete_moves_an_unused_agent_and_refuses_a_used_one(tmp_path, capsys):
    path = make_agent(tmp_path)
    make_study(tmp_path, 'user')
    document = store.load_raw('user', str(tmp_path))
    document['variantConditions'] = [{'name': 'helper', 'artifactPath': path, 'artifactHash': 'x',
                                      'artifact': {}}]
    store.save_raw(document, str(tmp_path))

    code, refused = cli(tmp_path, capsys, 'agent', 'delete', path)
    assert code == 65 and refused['error']['code'] == housekeeping_commands_code()
    assert 'user' in refused['error']['reason']
    assert refused['result']['usedBy'] == ['user']
    assert (tmp_path / path).exists()

    document = store.load_raw('user', str(tmp_path))
    document['variantConditions'] = []
    store.save_raw(document, str(tmp_path))
    code, preview = cli(tmp_path, capsys, 'agent', 'delete', path)
    assert code == 0 and preview['result']['usedBy'] == []
    code, applied = cli(tmp_path, capsys, 'agent', 'delete', path, '--artifact-sha256',
                        preview['result']['artifactFileSHA256'], '--yes')
    assert code == 0
    assert applied['result']['destination'].startswith('runs/model-variants/.trash-')
    assert not (tmp_path / path).exists()
    code, listed = cli(tmp_path, capsys, 'agent', 'list')
    assert listed['result']['agents'] == []


def test_an_agent_saved_by_a_run_is_refused_and_its_run_folder_stays(tmp_path, capsys):
    """A run folder is evidence and runs/ is append-only."""
    run = tmp_path / 'runs/20261005-variant-imported'
    run.mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps({'runType': 'variant-save', 'schemaVersion': 2}))
    artifact = json.loads(Path(tmp_path / make_agent(tmp_path, 'spare')).read_text())
    artifact['name'] = 'imported'
    (run / 'imported.json').write_text(json.dumps(artifact))
    path = 'runs/20261005-variant-imported/imported.json'
    before = sorted(p.name for p in run.iterdir())

    code, listed = cli(tmp_path, capsys, 'agent', 'list')
    digest = next(a['artifactFileSHA256'] for a in listed['result']['agents'] if a['path'] == path)
    for extra in ([], ['--artifact-sha256', digest, '--yes']):
        code, refused = cli(tmp_path, capsys, 'agent', 'delete', path, *extra)
        assert code == 65
        assert refused['error']['code'] == 'agentIsRunEvidence'
        assert 'saved by a run' in refused['error']['reason']
    assert sorted(p.name for p in run.iterdir()) == before
    assert trash(tmp_path / 'runs') == []


def test_the_verbs_and_flags_match_the_mac():
    labels = {spec.label: spec for spec in client_cli.CLIENT_VERB_SPECS}
    for label, digest in (('experiment rename', '--manifest-sha256'),
                          ('experiment delete', '--manifest-sha256'),
                          ('design rename', '--file-sha256'), ('design delete', '--file-sha256'),
                          ('agent delete', '--artifact-sha256')):
        spec = labels[label]
        assert spec.boolean_flags == frozenset({'--yes'})
        assert spec.value_flags == frozenset({digest})
        assert not spec.required_flags


def housekeeping_commands_code():
    from steerlab_server.client import housekeeping
    return housekeeping.AGENT_IN_USE_CODE


assert housekeeping_commands.EXPERIMENT_VERBS == frozenset({'rename', 'delete'})
