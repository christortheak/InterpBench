import json
from pathlib import Path
import subprocess
import sys
import pytest
from steerlab_server.experiment import method_authoring as owner, managed_methods, diagnostic_archives as archives
from steerlab_server.client.diagnostic_commands import workspace_action
from test_managed_methods import geometry_request
from test_scientific_execution import setup


def answers(root):
    refs = geometry_request(root)['parameters']['config']['artifacts']
    return dict(purpose='Compare directions', claim='Descriptive geometric separation', controls='Same model and layer',
                selection='Declared before inspection', fields={'artifacts': '\n'.join(refs), 'layer': '1'}, advanced={})


def test_public_interviews_match_registry_and_separate_jspace():
    from steerlab_server.experiment import science_catalog
    assert {o['id'] for o in json.loads(science_catalog.resource('workflows.json'))['operations']} == managed_methods.OPERATIONS
    assert science_catalog.operation('jspace')['method'] == 'jspace'
    assert 'current batch' in owner.interview('jspace')['claimBoundary']
    assert 'optvec-jspace' not in {o['id'] for o in science_catalog.catalog()['operations']}


def test_local_http_owner_and_cli_publish_exact_request_and_refuse_stale_inputs(setup, capsys):
    from steerlab_server import client_cli
    root, _, _ = setup; document = answers(root)
    payload = dict(workspaceRoot=str(root), operation='optvec-geometry', answersText=json.dumps(document))
    plan = workspace_action('draft', payload)
    assert plan == owner.draft('optvec-geometry', document, root)
    source = root/'answers.json'; source.write_text(json.dumps(document))
    assert client_cli.main(['science','draft','optvec-geometry','--answers',str(source),'--root',str(root),'--json']) == 0
    assert json.loads(capsys.readouterr().out)['result']['request'] == plan['request']
    result = workspace_action('publish', {**payload, 'destination':'requests/geometry', 'planSHA256':plan['planSHA256']})
    assert json.loads(Path(result['requestFile']).read_text()) == plan['request']
    assert json.loads(Path(result['reviewFile']).read_text()) == plan
    with pytest.raises(archives.Refusal, match='exists'): owner.publish('optvec-geometry',document,root,'requests/geometry',plan['planSHA256'])
    ref = document['fields']['artifacts'].splitlines()[0]
    with (root/(ref+'.json')).open('a') as f:f.write(' ')
    with pytest.raises(archives.Refusal, match='changed'): owner.publish('optvec-geometry',document,root,'requests/revised',plan['planSHA256'])
    assert not (root/'requests/revised').exists()


def test_unresolved_decisions_advanced_conflicts_and_paths_are_refused(setup):
    root, _, _ = setup; document=answers(root)
    document['claim']=''
    with pytest.raises(archives.Refusal):owner.draft('optvec-geometry',document,root)
    document['claim']='Descriptive';document['advanced']={'layer':2}
    with pytest.raises(archives.Refusal,match='override'):owner.draft('optvec-geometry',document,root)
    document['advanced']={}; plan=owner.draft('optvec-geometry',document,root)
    with pytest.raises(archives.Refusal,match='requests'):owner.publish('optvec-geometry',document,root,'runs/record',plan['planSHA256'])
    document['fields']['artifacts']='../outside'
    with pytest.raises(archives.Refusal):owner.draft('optvec-geometry',document,root)


def test_text_integer_and_document_hash_preserve_exact_bytes(tmp_path):
    assert owner.value({'kind':'integer'},str(2**64-1),tmp_path,[]) == 2**64-1
    path=tmp_path/'input.json';path.write_text('{"seed":18446744073709551615}')
    evidence=[]
    assert owner.value({'kind':'documentFile'},'input.json',tmp_path,evidence)['seed']==2**64-1
    assert evidence[0]['sha256']==archives.file_hash(path)


def test_interview_imports_no_gpu_stack():
    code="from steerlab_server.experiment.method_authoring import interview; import sys; interview('jspace'); assert not ({'torch','transformers'} & sys.modules.keys())"
    assert subprocess.run([sys.executable,'-c',code],capture_output=True).returncode==0


def test_sae_roster_plan_pins_only_unchanged_draft(tmp_path):
    from test_sae_candidates import _study,_seed_roster
    from steerlab_server.experiment import sae_authoring,experiment_store
    _study(str(tmp_path),concept='signal');path=_seed_roster(str(tmp_path))
    # The fixture helper returns the relative roster reference.
    plan=sae_authoring.pin_plan('s',path,str(tmp_path))
    sae_authoring.pin('s',path,str(tmp_path),plan['planSHA256'])
    assert experiment_store.load_raw('s',str(tmp_path))['saeCandidates']['hash']==plan['roster']['sha256']
    with pytest.raises(archives.Refusal,match='changed'):sae_authoring.pin('s',path,str(tmp_path),plan['planSHA256'])

    fresh=sae_authoring.pin_plan('s',path,str(tmp_path))
    assert sae_authoring.pin('s',path,str(tmp_path),fresh['planSHA256'])['changed'] is False


def test_every_interview_field_is_a_key_its_owner_accepts():
    """The form can only publish what the owner's parser accepts. A field id
    the owner does not know (the 2026-09-07 gradient `targetTrain` defect)
    makes every request from that interview unexecutable, so the owner's
    unknown-key refusal must never fire on an interview-shaped config."""
    from steerlab_server.experiment import managed_methods, optvec_campaign, science_catalog
    source = json.loads(science_catalog.resource('workflows.json'))
    placeholders = {'text': 'x', 'integer': 1, 'number': 1.0, 'boolean': True, 'integers': [1], 'numbers': [1.0],
                    'artifact': 'runs/a/v', 'artifacts': ['runs/a/v', 'runs/a/w'], 'file': 'runs/a', 'files': ['runs/a', 'runs/b'],
                    'fileRef': {'path': 'items.jsonl', 'sha256': '0' * 64}, 'documentFile': {}}
    for item in source['operations']:
        if item['id'] not in managed_methods.METHODS and item['id'] != 'optvec-campaign':
            continue
        config = {}
        for field in item['fields']:
            value = placeholders[field['kind']]
            if field['id'] == 'modelID': value = 'example/model'
            if field['id'] == 'revision': value = 'a' * 40
            cursor = config; names = field['id'].split('.')
            for name in names[:-1]: cursor = cursor.setdefault(name, {})
            cursor[names[-1]] = value
        try:
            if item['id'] == 'optvec-campaign': optvec_campaign.OptVecCampaignConfig.from_dict(config)
            else: managed_methods.config_owner(item['id'], config)
        except Exception as exc:  # value refusals are fine; unknown keys are not
            assert 'unknown' not in str(exc).lower(), item['id'] + ': ' + str(exc)
