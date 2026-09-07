import json
from pathlib import Path
import pytest
from steerlab_server.experiment import diagnostic_archives as archives, diagnostic_inputs, managed_methods
from steerlab_server.api import scientific_execution, diagnostic_transport
from test_scientific_execution import setup


def geometry_request(root):
    from test_battery_run import _vector
    first = _vector(str(root), concept='signal')
    second = _vector(str(root), concept='comparison')
    return {'operation': 'optvec-geometry', 'parameters': {'config': {'artifacts': [first, second], 'layer': 1}}}


def test_managed_geometry_round_trip_executes_real_owner_without_model(setup):
    root, _, profile = setup
    request = geometry_request(root)
    source = diagnostic_inputs.plan(request, root)
    packed = diagnostic_inputs.package(request, root, root/'runs/input.tar.gz', source['planSHA256'])
    staged = diagnostic_transport.stage(packed['bundlePath'], packed['bundleSha256'], profile)
    plan = scientific_execution.plan(staged['request'], profile)
    assert plan['compute'] == 'cpu' and plan['executor'] == 'local'
    packet = root/'packet.json'; packet.write_text(json.dumps(plan)); record = root/'result.json'
    assert scientific_execution.execute_packet(packet, 'example-job', record) == 0
    result = json.loads(record.read_text())['result']
    assert Path(result['runDirectory']).is_relative_to(Path(staged['executionRoot']))
    assert Path(result['runDirectory']).exists()


def test_managed_request_refuses_unknown_shapes_and_unpinned_gpu():
    with pytest.raises(archives.Refusal): managed_methods.request('jspace', {'command': 'anything'})
    with pytest.raises(archives.Refusal): managed_methods.require_pin('model', 'main')
    assert 'jspace' in managed_methods.OPERATIONS and 'optvec-jspace' not in managed_methods.OPERATIONS


def test_absolute_dependencies_and_changed_data_refuse(setup):
    root, _, _ = setup
    request = geometry_request(root)
    request['parameters']['config']['artifacts'][0] = str(root / request['parameters']['config']['artifacts'][0])
    with pytest.raises(archives.Refusal): diagnostic_inputs.plan(request, root)


@pytest.mark.parametrize('operation', ['rescore-style', 'sae-qualification-record'])
def test_managed_cpu_evidence_uses_real_owner_and_keeps_source_immutable(setup, operation):
    root, _, profile = setup
    if operation == 'rescore-style':
        from test_reasoning_style import _analyze_fixture
        _analyze_fixture(root)
        config = {'experiment': 's', 'sourceRun': 'runs/20260101T000000000-exp-s-run'}
    else:
        from test_sae_qualification import _sae_sidecar, _inputs
        config = {'artifact': _sae_sidecar(str(root)), 'inputs': _inputs()}
        for evidence in config['inputs'].get('evidenceRuns', []):
            directory=root/evidence['path'];directory.mkdir(parents=True)
            (directory/'report.json').write_text('{}')
    request = {'operation': operation, 'parameters': {'config': config}}
    source = diagnostic_inputs.plan(request, root)
    packed = diagnostic_inputs.package(request, root, root/'runs/input.tar.gz', source['planSHA256'])
    staged = diagnostic_transport.stage(packed['bundlePath'], packed['bundleSha256'], profile)
    plan = scientific_execution.plan(staged['request'], profile)
    packet=root/'packet.json';packet.write_text(json.dumps(plan)); record=root/'record.json'
    assert scientific_execution.execute_packet(packet,'cpu-evidence',record)==0
    result=json.loads(record.read_text())['result']
    assert Path(result['runDirectory']).parent==Path(staged['executionRoot'])/'runs'
    assert archives.snapshot(root,[e['path'] for e in source['files']])==source['files']


def test_managed_config_typo_refuses_before_any_output(setup):
    root, _, profile=setup;request=geometry_request(root)
    request['parameters']['config']['layers']=1
    with pytest.raises(archives.Refusal,match='unknown'):scientific_execution.plan(request,profile)
    assert not list((root/'runs').glob('*geometry*'))


def test_relocated_lens_uses_captured_tensors_not_original_workspace(setup):
    from test_optvec_jspace import _write_lens, _write_probe, LENS_ID
    from steerlab_server.jlens import lens_store, qualification
    import shutil
    root, _, _=setup
    _write_lens(root,root=str(root))
    config=geometry_request(root)['parameters']['config']
    ref=_write_probe(root/'probe.jsonl').to_dict();ref['path']='probe.jsonl'
    request={'operation':'jspace','parameters':{'config':{'vectorArtifacts':config['artifacts'], 'lensID':LENS_ID,'probeItems':ref,'modelID':'example/model','revision':'a'*40}}}
    source=diagnostic_inputs.plan(request,root)
    relocated=root/'copy';relocated.mkdir()
    for entry in source['files']:
        target=relocated/entry['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/entry['path'],target)
    assert diagnostic_inputs.plan(request,relocated)['files']==source['files']
    record=lens_store.resolve(LENS_ID,str(relocated))
    # Destroy the original converted tensor. A relocated read must still work,
    # even on a machine where the old absolute path remains accessible.
    Path(record.converted.path).write_bytes(b'not tensors')
    assert lens_store.load_layer(record,0,root=str(relocated)).ndim==2
    qualification.verify_converted(record,root=str(relocated))
    assert (relocated/'runs/jlens-lenses'/LENS_ID/'lens.json').read_bytes()==(root/'runs/jlens-lenses'/LENS_ID/'lens.json').read_bytes()
