"""Independent arithmetic, closed contracts, and immutable cross-surface inventory."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from steerlab_server.experiment import probe_artifacts as a, probe_library as library
from steerlab_server.client.diagnostic_commands import workspace_action
from steerlab_server.api.diagnostic_transport_routes import build_router

FIXTURE = Path(__file__).resolve().parents[2] / 'Tests/Fixtures/cross-engine/probe-artifacts.json'


@pytest.fixture
def documents():
    return json.loads(FIXTURE.read_bytes())


@pytest.fixture
def probes(tmp_path, documents):
    run = tmp_path / 'runs' / 'example'; run.mkdir(parents=True)
    for kind, doc in documents.items():
        name = 'example-probe.json' if kind == 'python' else kind + '.probe.json'
        (run / name).write_text(json.dumps(doc, indent=2))
    return tmp_path


def test_reference_linear_center_scale_orientation_and_strict_threshold(documents):
    d = documents['linear']
    # z = ((5-1)/2, (7-3)/4) = (2,1); 2*2 - 1 + .5 = 3.5.
    assert a.score(d, [5, 7], input_binding=d['input']) == {'score':3.5, 'scoreKind':'logit', 'label':'present'}
    d['output']['threshold'] = 3.5
    assert a.score(d, [5, 7], input_binding=d['input'])['label'] == 'absent'
    assert a.score(documents['mean'], [5, 7], input_binding=d['input'])['scoreKind'] == 'signedMargin'


def test_two_relu_units_represent_xor_without_claiming_probability(documents):
    d = documents['mlp']
    # 2*|x-y|-1 separates XOR; a single affine threshold cannot separate it.
    assert [a.score(d, v, input_binding=d['input'])['score'] for v in ([0,0],[0,1],[1,0],[1,1])] == [-1,1,1,-1]
    assert a.score(d, [0,1], input_binding=d['input'])['scoreKind'] == 'logit'


@pytest.mark.parametrize('key,value', [('schemaVersion',True), ('schemaVersion',2), ('method','unregistered'), ('label','')])
def test_top_level_schema_refuses_ambiguous_inputs(documents,key,value):
    d=documents['linear'];d[key]=value
    with pytest.raises(a.ProbeError): a.validate(d)


@pytest.mark.parametrize('mutate', [
    lambda d: d.update(unknown=True),
    lambda d: d['input'].update(layer=2),
    lambda d: d['input']['site'].update(layer=True),
    lambda d: d['input'].update(hiddenSize=2.0),
    lambda d: d['input'].update(revision='floating-tag'),
    lambda d: d['preprocessing'].update(scale=[0,1]),
    lambda d: d['preprocessing'].update(center=[0]),
    lambda d: d['layers'][0].update(weights=[[1]]),
    lambda d: d['layers'][0].update(bias=[1,2]),
    lambda d: d['layers'][0].update(activation='sigmoid'),
    lambda d: d['output'].update(scoreKind='probability'),
    lambda d: d['output'].update(threshold=float('nan')),
    lambda d: d['output'].update(positiveLabel='absent'),
    lambda d: d['training']['settings'].update(seed=2**64-1),
    lambda d: d['training']['data'][0].update(role='finalTest'),
    lambda d: d['training']['data'].clear(),
])
def test_shape_numerics_and_declared_data_roles_are_validated(documents,mutate):
    d=documents['linear'];mutate(d)
    with pytest.raises(a.ProbeError): a.validate(d)


def test_unknown_pins_are_advisories_and_validation_returns_an_independent_document(documents):
    d=documents['linear'];d['input']['revision']=None;d['input']['tokenizerSHA256']=None
    clean=a.validate(d)
    assert any('revision is unknown' in note for note in a.limitations(clean))
    clean['output']['threshold']=10
    assert d['output']['threshold']==0


@pytest.mark.parametrize('key,value', [('substrate','mlx'),('precision','bfloat16'),('revision','d'*40),('hiddenSize',True)])
def test_reference_scoring_requires_the_actual_full_input_binding(documents,key,value):
    d=documents['linear']; binding=copy.deepcopy(d['input']);binding[key]=value
    with pytest.raises(a.ProbeError,match='binding differs'): a.score(d,[0,0],input_binding=binding)


@pytest.mark.parametrize('activation', [[1], [False,0], [float('inf'),0], [1.79e308,-1.79e308]])
def test_invalid_activations_and_overflow_never_become_scores(documents,activation):
    d=documents['linear']
    with pytest.raises(a.ProbeError): a.score(d,activation,input_binding=d['input'])


@pytest.mark.parametrize('data', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_json_duplicates_and_nonfinite_values_refuse(data):
    with pytest.raises(a.ProbeError): a.read_json(data)


def test_all_formats_discovered_and_original_bytes_unchanged(probes):
    before={p:p.read_bytes() for p in (probes/'runs/example').iterdir()}
    inventory=library.inventory(probes)
    assert inventory['count']==5 and inventory['issues']==[] and inventory['changed'] is False
    assert {p['format'] for p in inventory['probes']}=={'native-reading-probe','python-reading-probe','activation-probe-v1'}
    for record in inventory['probes']:
        detail=library.inspect(record['path'],probes)
        assert detail['sha256']==hashlib.sha256(before[probes/record['path']]).hexdigest()
        assert 'document' not in record
        if detail['format']=='python-reading-probe':
            assert detail['createdAt'] is None and 'recipeName' not in detail['document']
            assert any('layer selection' in note for note in detail['limitations'])
    assert before=={p:p.read_bytes() for p in before}


def test_bad_files_are_visible_and_lens_probe_json_is_not_a_classifier(probes):
    (probes/'runs/example/bad.probe.json').write_text('{}')
    (probes/'runs/example/probe.json').write_text('{}')
    result=library.inventory(probes)
    assert result['count']==5
    assert [i['path'] for i in result['issues']]==['runs/example/bad.probe.json']


def test_paths_stay_inside_an_existing_workspace(probes,tmp_path):
    outside=tmp_path/'external.probe.json';outside.write_text('{}')
    with pytest.raises(a.ProbeError): library.inspect(outside,probes)
    with pytest.raises(a.ProbeError): library.inspect('runs/example/../../../external.probe.json',probes)
    (probes/'runs/example/link.probe.json').symlink_to(outside)
    assert library.inventory(probes)['issues'][0]['path'].endswith('link.probe.json')
    missing=tmp_path/'missing'
    with pytest.raises(a.ProbeError): library.inventory(missing)
    assert not missing.exists()


def test_workspace_action_and_http_call_the_real_library(probes,monkeypatch):
    monkeypatch.setenv('STEERLAB_ROOT',str(probes))
    payload={'workspaceRoot':str(probes)}
    expected=library.inventory(probes)
    assert workspace_action('probe-list',payload)==expected
    app=FastAPI(); app.include_router(build_router(SimpleNamespace()))
    with TestClient(app) as client:
        response=client.post('/api/science/workspace/probe-list',json=payload)
        assert response.status_code==200 and response.json()==expected
        selected=expected['probes'][0]['path']
        response=client.post('/api/science/workspace/probe-inspect',json={**payload,'path':selected})
        assert response.status_code==200 and response.json()==library.inspect(selected,probes)
        assert client.post('/api/science/workspace/probe-list',json={**payload,'workspaceRoot':str(probes/'other')}).status_code==409
        assert client.post('/api/science/workspace/probe-list',json={**payload,'extra':True}).status_code==409


def test_portable_cli_reaches_the_same_owner_without_gpu_imports(probes):
    script='''
import sys
from steerlab_server.client_cli import main
status=main(sys.argv[1:])
assert not any(x in sys.modules for x in ('torch','transformers','mlx'))
raise SystemExit(status)
'''
    result=subprocess.run([sys.executable,'-c',script,'science','probe-list','--root',str(probes),'--json'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    envelope=json.loads(result.stdout)
    assert envelope['result']['count']==5


def test_fixture_tracks_the_current_legacy_writer_and_new_contract(documents,tmp_path):
    from steerlab_server.experiment import probes
    result={'layer':1,'accuracy':0.75,'buildCount':8,'valCount':4,'probe':documents['native']['probe']}
    path=probes.save_artifact('Example reader','example/model',None,result,str(tmp_path))
    assert json.loads(Path(path).read_bytes())==documents['python']
    for name in ('mean','linear','mlp'):
        assert a.validate(documents[name])==documents[name]


def test_missing_probe_gets_a_probe_specific_repair(probes):
    with pytest.raises(a.ProbeError) as caught:
        workspace_action('probe-inspect',{'workspaceRoot':str(probes),'path':'runs/example/missing.probe.json'})
    assert caught.value.code=='probeArtifactRefused'
    assert 'probe' in caught.value.repair_action
    assert 'archive' not in caught.value.repair_action
