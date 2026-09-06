"""Actual authoring journeys, refusal preservation and lightweight installation gates."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from steerlab_server import client_cli
from steerlab_server.client import authoring_files, design_files, design_identity, design_panels, study_designs, study_interviews, study_packs
from steerlab_server.experiment import experiment_store as store, manifest_files
from steerlab_server.experiment.manifest_errors import ExperimentStoreError

FIXTURE = Path(__file__).parent / 'fixtures/study-assembly/pack.json'


@pytest.fixture(autouse=True)
def restore_cli_root(monkeypatch):
    # The CLI captures its explicit root in the environment for legacy owners.
    # Ensure in-process CLI journeys restore the test process's prior context.
    if 'STEERLAB_ROOT' in os.environ:
        monkeypatch.setenv('STEERLAB_ROOT', os.environ['STEERLAB_ROOT'])
    else:
        monkeypatch.delenv('STEERLAB_ROOT', raising=False)


def seed(root):
    data = FIXTURE.read_bytes()
    preview = study_packs.preview(data, root=root)
    return study_packs.apply(data, root=root, expected=preview['reviewSHA256'])['study']


def design(root):
    read = seed(root)
    return study_designs.create(read['name'], root=root, expected=read['manifestFileSHA256'], name='shared-design')['design']


def cli(root, capsys, *args):
    exit_code = client_cli.main([*args, '--root', str(root), '--json'])
    return exit_code, json.loads(capsys.readouterr().out)


def test_design_full_reviewed_journey_and_immutable_source(tmp_path, capsys):
    read = seed(tmp_path)
    source = authoring_files.study_path(tmp_path, read['name'])
    original = source.read_bytes()
    code, result = cli(tmp_path, capsys, 'design', 'save', read['name'], '--name', 'shared-design', '--manifest-sha256', read['manifestFileSHA256'])
    assert code == 0
    saved = result['result']['design']
    body = saved['document']['study']
    assert body['conditions'] == [] and body['variantConditions'] == []
    assert not set(design_identity.REMOVED) & body.keys()
    assert source.read_bytes() == original
    casting = tmp_path / 'casting.json'
    casting.write_text('{"agents":[]}')
    code, output = cli(tmp_path, capsys, 'design', 'instantiate', saved['name'], '--casting', str(casting), '--file-sha256', saved['designFileSHA256'], '--study-name', 'sibling')
    assert code == 0
    minted = output['result']
    assert minted['document']['templateProvenance']['templateHash'] == saved['portableContentHash']
    assert minted['document']['templateProvenance']['hashAlgorithm'] == 'portable-v1'
    assert minted['document']['taskPromptsHash'] == read['document']['taskPromptsHash']
    reused = study_designs.create(minted['name'], root=tmp_path, expected=minted['manifestFileSHA256'])
    assert reused['created'] is False and reused['changed'] is False
    before = authoring_files.study_path(tmp_path, 'sibling').read_bytes()
    changed = store.load_raw('sibling', str(tmp_path)); changed['maxTokens'] = 233
    store.save_raw(changed, str(tmp_path))
    revised = study_designs.update(saved['name'], 'sibling', root=tmp_path, expected=saved['designFileSHA256'], source_expected=changed.source_digest)
    assert revised['changed'] and revised['design']['portableContentHash'] != saved['portableContentHash']
    assert authoring_files.study_path(tmp_path, 'sibling').read_bytes() != before
    assert source.read_bytes() == original
    assert revised['design']['document']['templateDescription'] == saved['document']['templateDescription']
    assert revised['design']['document']['createdAt'] == saved['document']['createdAt']
    assert store.load_raw('sibling', str(tmp_path))['templateProvenance']['templateHash'] == saved['portableContentHash']


@pytest.mark.parametrize('operation', ['instantiate', 'describe', 'update'])
def test_stale_design_is_refused_without_publication(tmp_path, operation):
    saved = design(tmp_path)
    path = design_files.path(tmp_path, saved['name'])
    path.write_bytes(path.read_bytes() + b'\n')
    before = path.read_bytes()
    with pytest.raises(ExperimentStoreError):
        if operation == 'instantiate':
            study_designs.instantiate(saved['name'], {'agents': []}, root=tmp_path, expected=saved['designFileSHA256'])
        elif operation == 'describe':
            study_designs.describe(saved['name'], 'new', root=tmp_path, expected=saved['designFileSHA256'])
        else:
            study_designs.update(saved['name'], 'shared-study', root=tmp_path, expected=saved['designFileSHA256'], source_expected='0' * 64)
    assert path.read_bytes() == before
    assert len(list((tmp_path / 'experiments').iterdir())) == 1


def test_source_stale_and_lineage_mismatch_and_frozen_source(tmp_path):
    saved = design(tmp_path)
    study = store.load_raw('shared-study', str(tmp_path))
    with pytest.raises(ExperimentStoreError):
        study_designs.update(saved['name'], 'shared-study', root=tmp_path, expected=saved['designFileSHA256'], source_expected=study.source_digest)
    with pytest.raises(ExperimentStoreError):
        study_designs.create('shared-study', root=tmp_path, expected='0' * 64)
    path = authoring_files.study_path(tmp_path, 'shared-study')
    raw = dict(study); raw.update(status='frozen', freezeHash='retained', frozenAt='original', preregistrationHash='retained')
    path.write_text(json.dumps(raw))
    original = path.read_bytes()
    result = study_designs.create('shared-study', root=tmp_path, expected=manifest_files.digest_bytes(original))
    assert path.read_bytes() == original
    assert result['design']['document']['study']['status'] == 'draft'
    assert 'freezeHash' not in result['design']['document']['study']


def test_design_description_changes_file_review_but_not_identity(tmp_path):
    saved = design(tmp_path)
    updated = study_designs.describe(saved['name'], 'new note', root=tmp_path, expected=saved['designFileSHA256'])
    assert updated['portableContentHash'] == saved['portableContentHash']
    assert updated['designFileSHA256'] != saved['designFileSHA256']
    assert not study_designs.describe(saved['name'], 'new note', root=tmp_path, expected=updated['designFileSHA256'])['changed']


@pytest.mark.parametrize('tree', ['templates', 'experiments', 'prompts'])
def test_symlink_authoring_trees_are_refused(tmp_path, tree):
    root = tmp_path / 'workspace'; root.mkdir()
    if tree != 'templates':
        saved = design(root)
    target = tmp_path / 'protected'; target.mkdir()
    directory = root / tree
    if directory.exists():
        directory.rename(root / (tree + '-old'))
    directory.symlink_to(target, target_is_directory=True)
    with pytest.raises(ExperimentStoreError):
        if tree == 'templates':
            design_files.catalog(root)
        else:
            study_designs.instantiate(saved['name'], {'agents': []}, root=root, expected=saved['designFileSHA256'])
    assert list(target.iterdir()) == []


def test_drifted_task_file_refuses_and_scope_rederives_from_reviewed_bytes(tmp_path):
    saved = design(tmp_path)
    file = design_files.path(tmp_path, saved['name'])
    raw = deepcopy(saved['document'])
    prompt = tmp_path / 'prompts/tasks/scope.jsonl'
    data = b'{"id":"a","text":"choose","responseFormat":"binary-choice","options":["A","B"],"target":"A"}\n'
    # Use the vocabulary's actual label, shared with the run parser.
    data = data.replace(b'binary-choice', b'label')
    prompt.write_bytes(data)
    raw['study'].update(taskPromptsFile='prompts/tasks/scope.jsonl', taskPromptsHash=manifest_files.digest_bytes(data),
                        outcomeInstrumentScope={'responseFormats':['label'], 'itemCount':999, 'itemIDsHash':'stale'})
    file.write_bytes(design_files.encode(raw)); saved = design_files.read(saved['name'], tmp_path)
    result = study_designs.instantiate(saved['name'], {'agents':[]}, root=tmp_path, expected=saved['designFileSHA256'])
    assert result['document']['outcomeInstrumentScope']['itemCount'] == 1
    prompt.write_bytes(data + b' ')
    with pytest.raises(ExperimentStoreError):
        study_designs.instantiate(saved['name'], {'agents':[]}, root=tmp_path, expected=saved['designFileSHA256'])


def panel_design(root):
    saved = design(root)
    panel = {'schemaVersion':1, 'name':'shared-panel', 'description':'roles', 'baseModelID':'', 'temperature':0, 'maxTokens':2048,
             'sharedMaterials':'shared facts', 'agents':[{'id':'first','name':'First','baseModelID':'','systemPrompt':'role one','role':'reviewer'},
                                                        {'id':'second','name':'Second','baseModelID':'','systemPrompt':'role two','role':'reader'}],
             'turns':[{'id':'turn-one','title':'First response','speakerAgentID':'first','promptTemplate':'Read the materials.',
                       'outputLabel':'first-output','routing':'all','routedAgentIDs':[], 'includeScenarioMaterials':True,'includeSpeakerContext':True}]}
    ref = design_panels.pin(panel, root)
    template = saved['document']; template['semanticScenario'] = ref
    template['study'].update(studyType='multiAgent',studyKind='multiAgent')
    template['study'].pop('taskPromptsFile',None); template['study'].pop('taskPromptsHash',None)
    design_files.path(root,saved['name']).write_bytes(design_files.encode(template))
    return design_files.read(saved['name'],root), panel


def agent(root, model='test/model'):
    artifact = {'schemaVersion':1,'name':'shared-agent','baseModelID':model,'createdAt':'2026-01-01T00:00:00Z','adapters':[],
                'injections':[],'bandWidth':1,'alphaInNormUnits':False,'promptMode':'chatAssistant','qwenThinkingEnabled':False,'temperature':0}
    path = root / 'runs/model-variants/shared-agent/model-variant.json'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(design_files.encode(artifact))
    return {'artifactPath':str(path.relative_to(root)), 'artifactFileSHA256':manifest_files.digest_bytes(path.read_bytes())}


def test_panel_casting_keeps_roles_turns_provenance_and_round_trips(tmp_path):
    saved, panel = panel_design(tmp_path)
    ref = agent(tmp_path, saved['document']['study']['modelID'])
    result = study_designs.instantiate(saved['name'], {'seats':{'first':ref,'second':None}}, root=tmp_path, expected=saved['designFileSHA256'])
    study = result['document']; data = (tmp_path / study['multiAgentScenarioPath']).read_bytes()
    assert manifest_files.digest_bytes(data) == study['multiAgentScenarioHash']
    bound = json.loads(data)
    assert bound['turns'] == panel['turns']
    assert bound['agents'][0]['role'] == 'reviewer'
    assert bound['agents'][0]['variantArtifactHash'] == ref['artifactFileSHA256']
    assert 'variantArtifactPath' not in bound['agents'][1]
    assert bound['baseModelID'] == study['modelID']
    assert design_panels.semantic(bound) == panel
    reused = study_designs.create(result['name'], root=tmp_path, expected=result['manifestFileSHA256'])
    assert not reused['created']


@pytest.mark.parametrize('fault',['missing','extra','stale-agent','wrong-model','wrong-kind','panel-drift'])
def test_bad_castings_never_publish_a_draft(tmp_path, fault):
    saved, panel = panel_design(tmp_path)
    ref = agent(tmp_path, saved['document']['study']['modelID'])
    casting = {'seats':{'first':ref,'second':None}}
    if fault == 'missing': del casting['seats']['second']
    elif fault == 'extra': casting['seats']['extra'] = None
    elif fault == 'stale-agent': ref['artifactFileSHA256'] = '0'*64
    elif fault == 'wrong-model': casting['seats']['first'] = agent(tmp_path,'different/model')
    elif fault == 'wrong-kind': casting = {'agents':[]}
    else:
        path = tmp_path / saved['document']['semanticScenario']['path']; path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ExperimentStoreError):
        study_designs.instantiate(saved['name'],casting,root=tmp_path,expected=saved['designFileSHA256'])
    assert len(list((tmp_path / 'experiments').iterdir())) == 1
    assert not (tmp_path/'prompts/panels/compiled').exists()


def test_partial_batch_reports_success_and_refusal_and_does_not_retry(tmp_path, capsys):
    saved = design(tmp_path)
    rows = tmp_path/'rows.json'
    rows.write_text(json.dumps({'rows':[{'casting':{'agents':[]},'studyName':'successful'}, {'casting':{'seats':{'unknown':None}}}]}))
    code, result = cli(tmp_path,capsys,'design','batch',saved['name'],'--rows',str(rows),'--file-sha256',saved['designFileSHA256'])
    assert code == 65 and result['state'] == 'refused' and result['changed']
    assert result['error']['code'] == 'designBatchIncomplete'
    assert result['result']['minted'] == ['successful']
    assert result['result']['results'][1]['issue']['repairAction']
    assert store.load_raw('successful',str(tmp_path))['templateProvenance']['batchGroup'] == result['result']['batchGroup']


def test_lightweight_interviews_and_real_panel_authoring(tmp_path):
    saved, _ = panel_design(tmp_path)
    program = '''
import importlib.abc, json, sys
class NoGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','fastapi','uvicorn','peft'}:
            raise AssertionError('Heavy authoring import: ' + fullname)
sys.meta_path.insert(0, NoGPU())
from pathlib import Path
from steerlab_server.client import study_designs, study_interviews
for intent in ('conceptStudy','agentComparison','multiAgent'):
    assert 'NEVER fabricate hashes' in study_interviews.prompt(intent)
study_designs.instantiate(sys.argv[2], {'seats':{'first':None,'second':None}}, root=Path(sys.argv[1]), expected=sys.argv[3])
'''
    run = subprocess.run([sys.executable,'-c',program,str(tmp_path),saved['name'],saved['designFileSHA256']],capture_output=True,text=True)
    assert run.returncode == 0, run.stderr


@pytest.mark.parametrize('intent',['conceptStudy','agentComparison','multiAgent'])
def test_interviews_are_callable_read_only_and_ship_in_package(tmp_path,capsys,intent):
    code,result=cli(tmp_path,capsys,'authoring','study',intent)
    assert code==0 and result['result']['prompt']==study_interviews.prompt(intent)
    assert 'steerlab pack preview' in result['result']['prompt']
    assert 'steerlab-cli pack preview' in result['result']['prompt']
    assert list(tmp_path.iterdir())==[]


def test_identity_preserves_null_validation_and_opaque_json_and_large_seeds():
    body=json.loads(FIXTURE.read_bytes())['study']
    base={'study':body}
    for key,value in [('pipeline',{'null-is-data':None}),('seeds',[2**63+1]),('temperature',1e-16),('modelID','test/雪')]:
        other=deepcopy(base);other['study'][key]=value
        assert design_identity.content_hash(other)!=design_identity.content_hash(base)
    a=deepcopy(base);a['study']['concepts']=[{'validationHash':None}]
    b=deepcopy(base);b['study']['concepts']=[{}]
    assert design_identity.content_hash(a)!=design_identity.content_hash(b)


def test_panel_contract_defaults_survive_mint_and_design_reuse(tmp_path):
    saved, panel = panel_design(tmp_path)
    panel['turns'][0]['promptTemplate']=''
    panel['turns'][0]['contract']={'task':'Write a careful note.', 'materialsTitle':'  SHARED MATERIALS  '}
    template=saved['document']; template['semanticScenario']=design_panels.pin(panel,tmp_path)
    design_files.path(tmp_path,saved['name']).write_bytes(design_files.encode(template))
    saved=design_files.read(saved['name'],tmp_path)
    study=study_designs.instantiate(saved['name'],{'seats':{'first':None,'second':None}},root=tmp_path,expected=saved['designFileSHA256'])
    bound=json.loads((tmp_path/study['document']['multiAgentScenarioPath']).read_bytes())
    assert bound['schemaVersion']==2
    assert bound['turns'][0]['contract']['task']=='Write a careful note.'
    assert bound['turns'][0]['contract']['materialsTitle']=='SHARED MATERIALS'
    assert bound['agents'][0]['role']=='reviewer'
    # A legacy input with a stale schema label changes at compilation, exactly
    # as on the Mac. Its script stays intact and the save explains new lineage.
    assert bound['turns'][0]['contract']['ownVoice'] is True


def test_batch_unexpected_failure_keeps_successful_outcomes(tmp_path, monkeypatch):
    saved=design(tmp_path)
    original=study_designs.instantiate
    def mint(name,casting,**kwargs):
        if kwargs.get('study_name')=='broken': raise OSError('simulated disk failure')
        return original(name,casting,**kwargs)
    monkeypatch.setattr(study_designs,'instantiate',mint)
    result=study_designs.batch(saved['name'],{'rows':[{'casting':{'agents':[]},'studyName':'kept'}, {'casting':{'agents':[]},'studyName':'broken'}]},root=tmp_path,expected=saved['designFileSHA256'])
    assert result['changed'] and result['minted']==['kept']
    assert result['results'][1]['issue']['state']=='failed'


@pytest.mark.parametrize('args', [('design','list','extra'),('authoring','study','unknown'),('design','instantiate','missing'),('agent','inspect')])
def test_bad_cli_shapes_refuse_with_repair(tmp_path,capsys,args):
    code,result=cli(tmp_path,capsys,*args)
    assert code==64 and result['error']['repairAction']
    assert not list(tmp_path.iterdir())


def test_unreadable_catalog_entry_is_reported_and_symlink_is_not_followed(tmp_path):
    saved=design(tmp_path)
    (tmp_path/'templates/broken').mkdir()
    (tmp_path/'templates/broken/template.json').write_text('[]')
    result=design_files.catalog(tmp_path)['catalog']
    assert len(result['entries'])==1 and len(result['issues'])==1
    (tmp_path/'templates/link').symlink_to(tmp_path/'templates/shared-design',target_is_directory=True)
    result=design_files.catalog(tmp_path)['catalog']
    assert len(result['entries'])==1 and len(result['issues'])==2


def test_portable_hash_covers_the_entire_stored_body_not_only_its_stripped_projection():
    base={'study':design_identity.stripped(json.loads(FIXTURE.read_bytes())['study'])}
    for key,value in [('name','unexpected-instance'),('conditions',[{'name':'changed'}]),('variantConditions',[{'name':'new-arm'}]),('multiAgentScenarioPath','prompts/panels/another.json')]:
        altered=deepcopy(base);altered['study'][key]=value
        assert design_identity.content_hash(altered)!=design_identity.content_hash(base)


def test_interviews_are_declared_in_wheel_data_and_match_maintained_seed():
    import tomllib
    repository=Path(__file__).resolve().parents[2]
    config=tomllib.loads((repository/'Server/pyproject.toml').read_text())
    assert 'seed/prompts/study-interviews/*.md' in config['tool']['setuptools']['package-data']['steerlab_server.experiment']
    for intent in ('conceptStudy','agentComparison','multiAgent'):
        expected=(repository/'WorkspaceSeed/prompts/study-interviews'/f'study-{intent}.md').read_text().rstrip('\n')
        assert study_interviews.prompt(intent)==expected
