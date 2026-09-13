"""Study owners, resume, scoped panel turns, and freeze closure use real probe hooks."""
import json
from pathlib import Path
from contextlib import ExitStack, closing
import torch
from steerlab_server.experiment import probe_observation, probe_measurements, prompt_render
from test_probe_measurements import prepared


def observed_generation(model, prompt, *, observers=(), token_ids_out=None, **kwargs):
    with model.hooked.session([]), ExitStack() as sessions:
        for observer in observers:
            sessions.enter_context(observer.observe_session(model, prompt_render.RenderedPrompt(prompt,[1,2,3],3)))
        model.model(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones((1,3)))
    if token_ids_out is not None: token_ids_out.append(4)
    return 'example response'


def test_condition_records_and_resume_retain_measurements(tmp_path,monkeypatch):
    from steerlab_server.experiment import condition_execution as execution, manifest, resume
    config,doc,model=prepared(tmp_path)
    monkeypatch.setattr(execution.generate,'generate',observed_generation)
    study=manifest.Manifest(name='example',model_id='example/model',raw={'probeMeasurements':config},seeds=[0],temperature=0,max_tokens=4)
    eff=execution.EffectiveCondition('baseline',[],{},'rawCompletion',None,False,0)
    directory=tmp_path/'runs'/'study';directory.mkdir()
    kwargs=dict(name='example',manifest=study,experiment_hash='b'*64,wants_choice=False,wants_sampled=True,reader_scorers=[],should_cancel=None,log=lambda _:None,root=tmp_path)
    with closing(resume.GenerationWriter(str(directory))) as writer:
        execution.execute_condition(model,eff,[{'id':'item','prompt':'example'}],writer,**kwargs)
    source=directory/'generations.jsonl';before=source.read_bytes();record=json.loads(before)
    assert record['probeMeasurements']['readings'][0]['status']=='recorded'
    assert record['probeMeasurements']['promptID']=='item'
    with closing(resume.GenerationWriter(str(directory),resume=True)) as writer:
        execution.execute_condition(model,eff,[{'id':'item','prompt':'example'}],writer,**kwargs)
    assert source.read_bytes()==before


def test_panel_seat_filter_and_flattened_results(tmp_path,monkeypatch):
    from steerlab_server.experiment import multi_agent, panel_workflow, manifest
    from test_multi_agent import _scenario
    config,doc,model=prepared(tmp_path)
    config['probes'][0]['agents']=['b']
    scenario=_scenario();scenario.base_model_id='example/model'
    for agent in scenario.agents:
        agent.base_model_id='example/model'
    # Bare panel seats default to chat. This test uses the real binding contract
    # with a template and the existing rendering selection, no artifact coercion.
    model.tokenizer.get_chat_template=lambda:'example template'
    import hashlib
    doc['input']['reading']['rendering']='chatTemplate';doc['input']['templateSHA256']=hashlib.sha256(b'example template').hexdigest()
    source=tmp_path/config['probes'][0]['probe']['path'];source.write_text(json.dumps(doc))
    config['probes'][0]['probe']['sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(multi_agent,'generate',observed_generation)
    directory=tmp_path/'runs'/'panel';directory.mkdir()
    multi_agent.run_scenario(model,scenario,run_dir=str(directory),probe_measurements=config,probe_root=tmp_path,default_revision='a'*40)
    rows=[json.loads(line) for line in (directory/'turns.jsonl').read_text().splitlines()]
    assert 'probeMeasurements' not in rows[0]
    assert rows[1]['probeMeasurements']['agent']=='b'
    study=manifest.Manifest(name='example',model_id='example/model')
    flat=panel_workflow._panel_records_from(str(directory),'example',study,model,'configured',0)
    assert flat[1]['probeMeasurements']==rows[1]['probeMeasurements']


def test_freeze_relocation_preserves_source_and_pins_exact_bytes(tmp_path):
    from steerlab_server.experiment import experiment_store
    config,_,_=prepared(tmp_path)
    ref=config['probes'][0]['probe'];original=tmp_path/ref['path'];before=original.read_bytes()
    document={'name':'example','probeMeasurements':config}
    experiment_store._pin_external_inputs('example',document,tmp_path)
    relocated=document['probeMeasurements']['probes'][0]['probe']
    assert relocated['path'].startswith('experiments/example/pinned/probe-')
    assert original.read_bytes()==before==(tmp_path/relocated['path']).read_bytes()
    assert probe_measurements.load(document['probeMeasurements'],tmp_path)


def test_failure_keeps_partial_evidence_and_no_completed_response(tmp_path):
    import pytest
    config,_,model=prepared(tmp_path);model.revision='b'*40
    observer=probe_observation.create(config,probe_measurements.load(config,tmp_path),condition='baseline',agent='baseline',rendering='rawCompletion',run_directory=tmp_path,context={'promptID':'item'})
    with pytest.raises(ValueError):
        with observer.observe_session(model,prompt_render.RenderedPrompt('',[1],1)): pass
    files=list(tmp_path.glob('probe-failure-*.json'));assert len(files)==1
    result=json.loads(files[0].read_text());assert result['status']=='partial' and result['failures']
    assert not (tmp_path/'generations.jsonl').exists()


def test_http_and_bundle_import_preserve_the_reviewed_configuration(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.diagnostic_transport_routes import build_router
    from steerlab_server.experiment import experiment_store, bundles, manifest
    config,_,_=prepared(tmp_path)
    experiment_store.save_raw({'name':'example','status':'draft','modelID':'example/model','concepts':[],'conditions':[]},tmp_path)
    monkeypatch.setenv('STEERLAB_ROOT',str(tmp_path))
    app=FastAPI();app.include_router(build_router(SimpleNamespace()))
    payload={'workspaceRoot':str(tmp_path),'experiment':'example','settingsText':json.dumps(config)}
    with TestClient(app) as client:
        review=client.post('/api/science/workspace/measurements-review',json=payload)
        assert review.status_code==200
        saved=client.post('/api/science/workspace/measurements-save',json={**payload,'planSHA256':review.json()['planSHA256']})
        assert saved.status_code==200 and saved.json()['changed']
        assert client.post('/api/science/workspace/measurements-review',json={**payload,'workspaceRoot':str(tmp_path/'wrong')}).status_code==409
    packed=bundles.package_experiment('example',root=tmp_path)
    isolated=tmp_path/'isolated';isolated.mkdir()
    bundles.import_bundle(packed['bundlePath'],target_root=str(isolated),expected_sha256=packed['bundleSha256'])
    imported=manifest.Manifest.load('example',isolated)
    assert imported.raw['probeMeasurements']==config
    assert probe_measurements.load(imported.raw['probeMeasurements'],isolated)
