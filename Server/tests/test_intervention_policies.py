"""Independent policy math, publication, timing, state, and conflict acceptance."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from steerlab_server.experiment import policy_artifacts as artifacts, policy_authoring as owner, policy_execution as runtime, prompt_render
from steerlab_server.experiment.model_variant import ModelVariant
from steerlab_server.steering.policy_actions import Action, residual, logits
from steerlab_server.steering.runtime import Reading, Site
from test_probe_measurements import prepared


def setup(root, site='residualPost'):
    _, probe, model = prepared(root, site=site)
    p = {'id':'reader','path':'runs/fit/trained.probe.json'}
    doc = dict(schemaVersion=1,name='example-policy',binding={k:probe['input'][k] for k in ('modelID','revision','tokenizerSHA256','coordinateConvention')},
               site=probe['input']['site'],stages=['prefill','decode'],positions='lastPosition',probes=[p],
               actions=[{'id':'change','kind':'add','bounds':[0,2],'vector':[1,0]}],
               rules=[{'action':'change','kind':'fixed','value':2}],onError='stop',maxEvents=100)
    doc['binding']['rendering']='raw'
    return doc, probe, model


def attached(doc, root):
    raw=owner.encode(owner.materialize(doc,root)); return [{'json':raw.decode(),'sha256':hashlib.sha256(raw).hexdigest()}]


def forward(model, ids):
    return model.model(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids))))


@pytest.mark.parametrize('site,expected', [('residualPre', [23,19]),('residualPost',[19,19])])
def test_fixed_policy_real_block_inputs_outputs_and_sampling_rng(tmp_path,site,expected):
    doc,_,model=setup(tmp_path,site)
    baseline=forward(model,[1,2,3]).clone(); rng=torch.random.get_rng_state().clone()
    policy=runtime.Execution(attached(doc,tmp_path),rendering='rawCompletion')
    with model.hooked.session([]), policy.observe_session(model,prompt_render.RenderedPrompt('',[1,2,3],3)):
        actual=forward(model,[1,2,3])
        assert actual[0,-1].tolist()==expected
        assert torch.equal(actual[0,:2],baseline[0,:2])
    assert torch.equal(torch.random.get_rng_state(),rng)
    assert torch.equal(forward(model,[1,2,3]),baseline)
    assert not model.hooked.runtime.subscriptions and not model.hooked._pre_handles
    assert policy.result([4])['decisions'][0]['inputTokenPosition']==2


def test_threshold_and_affine_scores_independent_reference_and_chunking(tmp_path):
    from steerlab_server.experiment.probe_artifacts import score
    doc,probe,model=setup(tmp_path);doc['positions']='allPositions'
    doc['rules']=[{'action':'change','kind':'threshold','weights':{'reader':1},'threshold':score(probe,[3,5],input_binding=probe['input'])['score'],'below':0,'above':2}]
    p=runtime.Execution(attached(doc,tmp_path),rendering='rawCompletion')
    baseline=forward(model,[1,2,3]).clone()
    with model.hooked.session([]),p.observe_session(model,prompt_render.RenderedPrompt('',[1,2,3],3)):
        chunks=torch.cat([forward(model,[1,2]),forward(model,[3])],dim=1)
    for i,row in enumerate(p.events):
        expected=score(probe,[2*(i+1)+1,2*(i+2)+1],input_binding=probe['input'])['score']
        assert row['scores']['reader']==pytest.approx(expected)
        delta=4 if expected>doc['rules'][0]['threshold'] else 0
        assert chunks[0,i,0]==baseline[0,i,0]+delta
    doc['rules']=[{'action':'change','kind':'affine','weights':{'reader':1},'slope':.5,'intercept':1}]
    p=runtime.Execution(attached(doc,tmp_path),rendering='rawCompletion')
    with model.hooked.session([]),p.observe_session(model,prompt_render.RenderedPrompt('',[1,2,3],3)):forward(model,[1,2,3])
    for row in p.events: assert row['strengths']['change']==pytest.approx(max(0,min(2,row['scores']['reader']*.5+1)))


def test_joint_ablation_then_ordered_additions_and_conflicting_strengths():
    h=torch.tensor([[[3.,4.,5.]]]);value=torch.tensor([1.])
    actions=[Action({'kind':'ablate','vector':[1,0,0]},value),Action({'kind':'ablate','vector':[1,1,0]},value),Action({'kind':'add','vector':[1,0,0]},value*2)]
    assert torch.allclose(residual(h,actions),torch.tensor([[[2.,0.,5.]]]))
    actions[1]=Action(actions[1].specification,value*.5)
    with pytest.raises(ValueError,match='different ablation'):residual(h,actions)
    assert torch.equal(h,torch.tensor([[[3.,4.,5.]]]))


def test_all_decisions_read_same_snapshot_even_when_provider_mutates_its_copy(tmp_path):
    doc,_,model=setup(tmp_path)
    provider='def decide(context, tensor, scores, state, rng, assets):\n    tensor.fill_(100)\n    return [Decision("change", 1)]\n'
    doc['rules']=[];doc['provider']={'sourceText':provider,'sourceSHA256':hashlib.sha256(provider.encode()).hexdigest(),'assets':{}}
    other,_,_=setup(tmp_path);other['name']='other'
    p=runtime.Execution(attached(doc,tmp_path)+attached(other,tmp_path),rendering='rawCompletion')
    readings=[]
    with model.hooked.session([]),p.observe_session(model,prompt_render.RenderedPrompt('',[3],1)),model.hooked.readings([Reading('after',Site('residualPost',0),'postAction',lambda h,c,s:readings.append(h.clone()))]):
        output=forward(model,[3])
    assert output[0,0].tolist()==[21,19] and readings[0][0,0].tolist()==[10,9]


def test_provider_state_rng_assets_failure_budget_and_reset(tmp_path):
    doc,_,model=setup(tmp_path)
    source='def decide(context, tensor, scores, state, rng, assets):\n    state["count"] = state.get("count", 0) + 1\n    if state["count"] > 1: raise ValueError("example failure")\n    assert assets["input"] == b"example"\n    return [Decision("change", rng.random())]\n'
    (tmp_path/'provider.py').write_text(source);(tmp_path/'asset.txt').write_text('example')
    doc.update(rules=[],provider={'sourcePath':'provider.py','assetPaths':{'input':'asset.txt'}},onError='skipPolicy',maxEvents=1)
    results=[]
    before=torch.random.get_rng_state().clone()
    for _ in range(2):
        p=runtime.Execution(attached(doc,tmp_path),rendering='rawCompletion',context={'seed':'1','agent':'first'})
        with model.hooked.session([]),p.observe_session(model,prompt_render.RenderedPrompt('',[1],1)):
            forward(model,[1]);forward(model,[2])
        results.append(p.result([2,3]))
        assert p.closed and not p.states and not p.providers
    assert results[0]==results[1] and results[0]['omittedDecisions']==1 and results[0]['failures']
    assert torch.equal(before,torch.random.get_rng_state())
    doc['onError']='stop';p=runtime.Execution(attached(doc,tmp_path),rendering='rawCompletion',run_directory=tmp_path)
    with pytest.raises(ValueError,match='example failure'):
        with model.hooked.session([]),p.observe_session(model,prompt_render.RenderedPrompt('',[1],1)):
            forward(model,[1]);forward(model,[2])
    assert len(list(tmp_path.glob('policy-failure-*.json')))==1
    assert not model.hooked.runtime.subscriptions


@pytest.mark.parametrize('bad', ['bounds','duplicate','nan','stage','inline'])
def test_invalid_policies_fail_before_execution(tmp_path,bad):
    doc,_,_=setup(tmp_path);doc=owner.materialize(doc,tmp_path)
    if bad=='bounds':doc['rules'][0]['value']=3
    elif bad=='duplicate':doc['actions']*=2
    elif bad=='nan':doc['actions'][0]['vector'][0]=float('nan')
    elif bad=='stage':doc['stages']=[{}]
    else:doc['probes'][0]['json']+=' '
    with pytest.raises(ValueError):artifacts.validate(doc)


def test_logit_bias_mask_force_intersection_and_never_unmask():
    scores=torch.tensor([[1.,2.,-torch.inf,4.]])
    def a(kind,tokens,strength=1):return Action({'kind':kind,'tokens':tokens},torch.tensor([float(strength)]))
    actual=logits(scores,[a('logitBias',[0],5),a('allowTokens',[0,1]),a('forceToken',[0])])
    assert actual[0,0]==6 and torch.isneginf(actual[0,1:]).all()
    for actions in [[a('forceToken',[0]),a('forceToken',[1])],[a('forceToken',[2])],[a('forceToken',[4])]]:
        with pytest.raises(ValueError):logits(scores,actions)
    assert scores[0,0]==1


def test_publication_stale_bytes_new_agent_exact_inline_and_no_execution(tmp_path):
    doc,_,_=setup(tmp_path);plan=owner.review(doc,tmp_path)
    original=tmp_path/'runs/fit/trained.probe.json';original.write_text(original.read_text()+' ')
    with pytest.raises(ValueError,match='changed'):owner.publish(doc,tmp_path,plan['planSHA256'])
    plan=owner.review(doc,tmp_path);saved=owner.publish(doc,tmp_path,plan['planSHA256'])
    source=tmp_path/'runs/fit/agent.json';agent=ModelVariant(name='example',base_model_id='example/model',base_revision='a'*40,prompt_mode='rawCompletion').to_dict();agent['unknownField']={'preserved':True};source.write_text(json.dumps(agent));before=source.read_bytes()
    settings={'agentPath':'runs/fit/agent.json','name':'modified','policyPaths':[saved['path']]}
    review=owner.attachment_review(settings,tmp_path);result=owner.attach(settings,tmp_path,review['planSHA256'])
    raw=json.loads((tmp_path/result['path']).read_bytes());assert raw['unknownField']==agent['unknownField']
    variant=ModelVariant.from_dict(raw);assert variant.intervention_policies[0]['json']==(tmp_path/saved['path']).read_text()
    assert variant.to_dict()['interventionPolicies']==raw['interventionPolicies'] and source.read_bytes()==before
    assert owner.inventory(tmp_path)['count']==1
    with pytest.raises(ValueError,match='does not run intervention policies'):
        from steerlab_server.experiment.model_variant import variant_injections
        variant_injections(variant)
    old=ModelVariant.from_dict(agent).to_dict();assert 'interventionPolicies' not in old


def test_missing_workspace_and_symbolic_inputs_refuse(tmp_path):
    doc,_,_=setup(tmp_path)
    with pytest.raises(ValueError,match='existing workspace'):owner.review(doc,tmp_path/'absent')
    assert not (tmp_path/'absent').exists()
    original=tmp_path/'runs/fit/trained.probe.json'; original.rename(original.with_name('other.json'));original.symlink_to('other.json')
    with pytest.raises(ValueError):owner.review(doc,tmp_path)


def test_policy_session_is_closed_immediately_when_stream_is_superseded(tmp_path):
    doc,_,model=setup(tmp_path);p=runtime.Execution(attached(doc,tmp_path),rendering='rawCompletion')
    old=model.hooked.arm([],abandonable=True)
    scope=p.observe_session(model,prompt_render.RenderedPrompt('',[1],1));scope.__enter__();forward(model,[1])
    new=model.hooked.arm([],abandonable=True)
    assert p.closed and not p.states and not p.providers
    with pytest.raises(ValueError,match='closed'):p.logits_processor(torch.tensor([[1]]),torch.zeros(1,5))
    scope.__exit__(None,None,None);model.hooked.disarm(old);model.hooked.disarm(new)


def test_logits_use_only_current_position_scores_and_constraints(tmp_path,monkeypatch):
    doc,probe,model=setup(tmp_path)
    doc['site']={'kind':'logitsPreSelection'};doc['binding']['tokenizerSHA256']='b'*64
    probe['input']['tokenizerSHA256']='b'*64
    (tmp_path/'runs/fit/trained.probe.json').write_text(json.dumps(probe))
    monkeypatch.setattr(runtime.probe_capture,'tokenizer_identity',lambda _: 'b'*64)
    doc['actions']=[{'id':'change','kind':'forceToken','bounds':[0,1],'tokens':[3]}]
    doc['rules']=[{'action':'change','kind':'threshold','weights':{'reader':1},'threshold':-1e6,'below':0,'above':1}]
    p=runtime.Execution(attached(doc,tmp_path),rendering='rawCompletion')
    with model.hooked.session([]),p.observe_session(model,prompt_render.RenderedPrompt('',[1,2],2)):
        forward(model,[1,2]);out=p.logits_processor(torch.tensor([[1,2]]),torch.zeros(1,5))
        assert torch.isfinite(out[0,3]) and torch.isneginf(out[0,[0,1,2,4]]).all()
        with pytest.raises(ValueError,match='no probe reading'):p.logits_processor(torch.tensor([[1,2,3]]),torch.zeros(1,5))


def test_actual_transformers_generation_applies_constraints_before_sampling(tmp_path):
    from transformers import LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from steerlab_server.steering.hooks import HookedModel
    from steerlab_server.steering.model_loader import SteeredModel
    from steerlab_server.experiment.generate import generate
    backend=Tokenizer(WordLevel({'[PAD]':0,'[EOS]':1,'[UNK]':2,'example':3,'selected':4},unk_token='[UNK]'));backend.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',eos_token='[EOS]',unk_token='[UNK]')
    torch.manual_seed(3)
    lm=LlamaForCausalLM(LlamaConfig(vocab_size=5,hidden_size=8,intermediate_size=16,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=32,pad_token_id=0,eos_token_id=1)).eval()
    model=SteeredModel(model=lm,tokenizer=tokenizer,hooked=HookedModel(lm),model_id='example/model',revision='a'*40)
    doc,_,_=setup(tmp_path)
    doc.update(site={'kind':'logitsPreSelection'},probes=[],actions=[{'id':'change','kind':'forceToken','bounds':[0,1],'tokens':[4]}],rules=[{'action':'change','kind':'fixed','value':1}])
    doc['binding']['tokenizerSHA256']=runtime.probe_capture.tokenizer_identity(tokenizer)
    baseline_ids=[]; generate(model,'example',model_id=model.model_id,prompt_mode='rawCompletion',max_tokens=2,temperature=0,token_ids_out=baseline_ids)
    p=runtime.Execution(attached(doc,tmp_path),rendering='rawCompletion');ids=[]
    generate(model,'example',model_id=model.model_id,prompt_mode='rawCompletion',max_tokens=2,temperature=.8,token_ids_out=ids,observers=[p])
    assert ids==[4,4] and len(p.events)==2 and p.closed
    assert [r['inputTokenPosition'] for r in p.events]==[0,1]
    after=[];generate(model,'example',model_id=model.model_id,prompt_mode='rawCompletion',max_tokens=2,temperature=0,token_ids_out=after)
    assert after==baseline_ids


def test_http_cli_and_isolated_agent_bundle_keep_exact_policy(tmp_path,monkeypatch,capsys):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.diagnostic_transport_routes import build_router
    from steerlab_server.experiment import experiment_store,bundles,manifest
    from steerlab_server.client_cli import main
    doc,_,model=setup(tmp_path);monkeypatch.setenv('STEERLAB_ROOT',str(tmp_path))
    app=FastAPI();app.include_router(build_router(SimpleNamespace()))
    payload={'workspaceRoot':str(tmp_path),'settingsText':json.dumps(doc)}
    with TestClient(app) as client:
        review=client.post('/api/science/workspace/policy-review',json=payload);assert review.status_code==200
        saved=client.post('/api/science/workspace/policy-publish',json={**payload,'planSHA256':review.json()['planSHA256']});assert saved.status_code==200
        assert client.post('/api/science/workspace/policy-review',json={**payload,'workspaceRoot':str(tmp_path/'other')}).status_code==409
    file=tmp_path/'settings.json';file.write_text(json.dumps(doc))
    assert main(['science','policy-review',str(file),'--root',str(tmp_path),'--json'])==0
    envelope=json.loads(capsys.readouterr().out);assert 'planSHA256' in str(envelope)
    source=tmp_path/'runs/fit/source.json';source.write_text(json.dumps(ModelVariant(name='example',base_model_id='example/model',base_revision='a'*40,prompt_mode='rawCompletion').to_dict()))
    settings={'agentPath':'runs/fit/source.json','name':'modified','policyPaths':[saved.json()['path']]}
    plan=owner.attachment_review(settings,tmp_path);new=owner.attach(settings,tmp_path,plan['planSHA256'])
    experiment_store.save_raw({'name':'example','status':'draft','modelID':'example/model','concepts':[],'conditions':[],
        'variantConditions':[{'name':'modified','artifactPath':new['path'],'artifactHash':new['sha256'],'artifact':new['document']}]},tmp_path)
    packed=bundles.package_experiment('example',root=tmp_path);isolated=tmp_path/'isolated';isolated.mkdir()
    bundles.import_bundle(packed['bundlePath'],target_root=str(isolated),expected_sha256=packed['bundleSha256'])
    imported=manifest.Manifest.load('example',isolated);variant=ModelVariant.from_dict(imported.variant_conditions[0].artifact)
    assert variant.intervention_policies==new['document']['interventionPolicies']
    p=runtime.create(variant,rendering='rawCompletion')
    # Remove local inputs: execution consumes only the agent's embedded bytes.
    (tmp_path/'runs/fit/trained.probe.json').unlink();(tmp_path/saved.json()['path']).unlink()
    with model.hooked.session([]),p.observe_session(model,prompt_render.RenderedPrompt('',[1],1)):forward(model,[1])
    assert p.events


def test_real_vector_file_resolves_verified_direction_and_rejects_foreign_substrate(tmp_path):
    import numpy as np
    from safetensors.numpy import save
    from steerlab_server.steering.vector_store import SteeringVectorSidecar
    doc,_,_=setup(tmp_path)
    doc['actions'][0].pop('vector');doc['actions'][0]['vectorArtifactID']='runs/fit/example'
    raw=save({'layer_0':np.array([.5,1.],dtype=np.float32)})
    (tmp_path/'runs/fit/example.safetensors').write_bytes(raw)
    # Real sidecar defaults plus the required core fields.
    import inspect
    fields=inspect.signature(SteeringVectorSidecar).parameters
    meta={k:None for k,v in fields.items() if v.default is inspect.Parameter.empty}
    meta.update(modelID='example/model',concept='example',layerCount=1,hiddenSize=2,substrate='python-hf-transformers')
    path=tmp_path/'runs/fit/example.json';path.write_text(json.dumps(meta))
    plan=owner.review(doc,tmp_path)
    action=plan['document']['actions'][0]
    assert action['vector']==[.5,1.] and action['source']['tensorSHA256']==hashlib.sha256(raw).hexdigest()
    meta['substrate']='swift-mlx';path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='substrate'):owner.review(doc,tmp_path)


def test_sampled_study_records_policy_and_resume_preserves_bytes(tmp_path,monkeypatch):
    from contextlib import closing
    from steerlab_server.experiment import condition_execution as execution,manifest,resume
    from test_probe_measurement_journeys import observed_generation
    doc,_,model=setup(tmp_path);monkeypatch.setattr(execution.generate,'generate',observed_generation)
    variant=ModelVariant(name='modified',base_model_id='example/model',base_revision='a'*40,prompt_mode='rawCompletion',intervention_policies=attached(doc,tmp_path))
    study=manifest.Manifest(name='example',model_id='example/model',seeds=[0],max_tokens=4)
    eff=execution.EffectiveCondition('modified',[],{},'rawCompletion',None,False,0,variant=variant)
    directory=tmp_path/'runs/study';directory.mkdir()
    kwargs=dict(name='example',manifest=study,experiment_hash='b'*64,wants_choice=False,wants_sampled=True,reader_scorers=[],should_cancel=None,log=lambda _:None,root=tmp_path)
    with closing(resume.GenerationWriter(str(directory))) as writer:execution.execute_condition(model,eff,[{'id':'item','prompt':'example'}],writer,**kwargs)
    path=directory/'generations.jsonl';before=path.read_bytes();record=json.loads(before)
    assert record['interventionDecisions']['decisions'][0]['strengths']['change']==2
    with closing(resume.GenerationWriter(str(directory),resume=True)) as writer:execution.execute_condition(model,eff,[{'id':'item','prompt':'example'}],writer,**kwargs)
    assert path.read_bytes()==before
    with closing(resume.GenerationWriter(str(directory),resume=True)) as writer:
        with pytest.raises(ValueError,match='direct choice scoring'):execution.execute_condition(model,eff,[],writer,**{**kwargs,'wants_choice':True})


def test_panel_policy_is_per_seat_and_baseline_strips_it(tmp_path,monkeypatch):
    from steerlab_server.experiment import multi_agent
    from test_multi_agent import _scenario
    from test_probe_measurement_journeys import observed_generation
    doc,_,model=setup(tmp_path);model.tokenizer.get_chat_template=lambda:'example template'
    variant=ModelVariant(name='modified',base_model_id='example/model',base_revision='a'*40,prompt_mode='rawCompletion',intervention_policies=attached(doc,tmp_path))
    path=tmp_path/'runs/fit/agent.json';path.write_text(json.dumps(variant.to_dict()))
    monkeypatch.setenv('STEERLAB_ROOT',str(tmp_path));monkeypatch.setattr(multi_agent,'generate',observed_generation)
    scenario=_scenario()
    # Dataclasses are mutable scenario inputs, and each declared seat points to the same policy agent.
    for agent in scenario.agents:agent.variant_artifact_path=str(path);agent.variant_artifact_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    for strip in (False,True):
        directory=tmp_path/('baseline' if strip else 'configured');directory.mkdir()
        multi_agent.run_scenario(model,scenario,run_dir=str(directory),strip_interventions=strip,log=lambda _:None)
        rows=[json.loads(line) for line in (directory/'turns.jsonl').read_text().splitlines()]
        assert rows and all(('interventionDecisions' in row)==(not strip) for row in rows)
        if not strip:assert len({row['interventionDecisions']['agent'] for row in rows})==2


def test_failure_summaries_are_bounded_independently_of_execution(tmp_path):
    doc,_,_=setup(tmp_path);p=runtime.Execution(attached(doc,tmp_path),rendering='rawCompletion')
    for _ in range(80):p.failure('x'*3000)
    result=p.result([])
    assert result['failureCount']==80 and len(result['failures'])==64
    assert all(len(s)==2048 for s in result['failures']) and result['status']=='partial'


def test_nonlinear_policy_scores_match_independent_portable_reference(tmp_path):
    doc,_,_=setup(tmp_path);p=runtime.Execution(attached(doc,tmp_path),rendering='rawCompletion')
    fixtures=json.loads((Path(__file__).resolve().parents[2]/'Tests/Fixtures/cross-engine/probe-artifacts.json').read_text())
    for probe in fixtures.values():
        if not isinstance(probe,dict) or probe.get('artifactType')!='activation-probe':continue
        width=probe['input']['hiddenSize'];values=[float(i-1) for i in range(width)]
        actual=p.score(torch.tensor([[values]],dtype=torch.float32),probe)[0].item()
        expected=artifacts.probes.score(probe,values,input_binding=probe['input'])['score']
        assert actual==pytest.approx(expected,abs=1e-6)


@pytest.mark.parametrize('value',[None,[],0,'example'])
def test_malformed_provenance_and_agent_paths_have_typed_repairs(tmp_path,value):
    doc,_,_=setup(tmp_path);doc['actions'][0]['source']=value
    with pytest.raises(ValueError,match='provenance'):owner.review(doc,tmp_path)
    with pytest.raises(ValueError):owner.attachment_review({'agentPath':value,'name':'example','policyPaths':[]},tmp_path)
