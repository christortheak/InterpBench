"""Independent policy math, publication, timing, state, and conflict acceptance."""
from contextlib import contextmanager
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


def authored(doc):
    """setup()'s version-1 meaning as new settings: version 2 with each addition's raw multiplier chosen explicitly."""
    doc=copy.deepcopy(doc);doc['schemaVersion']=2
    for action in doc['actions']:
        if action['kind']=='add':action['strengthUnits']='rawMultiplier'
    return doc


def forward(model, ids):
    return model.model(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids))))


class Deltas:
    """A read-only observer: what each forward's action phase changed at one residual site (post minus pre)."""
    def __init__(self, site): self.site=site;self.passes=[];self.before={}
    @contextmanager
    def observe_session(self, model, rendered):
        def before(h,context,state):self.before[context.offset]=h.detach().clone()
        def after(h,context,state):self.passes.append((context.offset,(h.detach()-self.before.pop(context.offset))[0]))
        with model.hooked.readings([Reading('before',self.site,'preAction',before),Reading('after',self.site,'postAction',after)]):yield


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
    for result in results:
        assert result.pop('timing')['decisionHostSeconds'] >= 0
        for row in result['decisions']:
            for outcome in row['actionOutcomes'].values(): assert outcome.pop('hostSeconds') >= 0
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
    doc,_,_=setup(tmp_path);doc=authored(doc);plan=owner.review(doc,tmp_path)
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
    doc,_,_=setup(tmp_path);doc=authored(doc)
    with pytest.raises(ValueError,match='existing workspace'):owner.review(doc,tmp_path/'absent')
    assert not (tmp_path/'absent').exists()
    original=tmp_path/'runs/fit/trained.probe.json'; original.rename(original.with_name('other.json'));original.symlink_to('other.json')
    with pytest.raises(ValueError,match='ordinary'):owner.review(doc,tmp_path)


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
    doc,_,model=setup(tmp_path);doc=authored(doc);monkeypatch.setenv('STEERLAB_ROOT',str(tmp_path))
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


def vector_artifact(root, values, **sidecar):
    """A real one-layer vector artifact: safetensors bytes and a sidecar with the required core fields."""
    import numpy as np
    from safetensors.numpy import save
    raw=save({'layer_0':np.array(values,dtype=np.float32)})
    (root/'runs/fit').mkdir(parents=True,exist_ok=True);(root/'runs/fit/example.safetensors').write_bytes(raw)
    meta={'modelID':'example/model','concept':'example','stimulusSetHash':'c'*64,'layerCount':1,'hiddenSize':len(values),'normsPerLayer':[1.],
          'extractionDate':'2026-01-01T00:00:00Z','substrate':'python-hf-transformers',**sidecar}
    (root/'runs/fit/example.json').write_text(json.dumps(meta))
    return raw, meta


def norm_unit_settings(root, value=.1):
    """New settings as guided authoring writes them: a library direction, units left to their default."""
    doc,_,_=setup(root);doc['schemaVersion']=2
    doc['actions']=[{'id':'change','kind':'add','bounds':[0,.2],'vectorArtifactID':'runs/fit/example'}]
    doc['rules']=[{'action':'change','kind':'fixed','value':value}]
    return doc


def test_real_vector_file_resolves_verified_direction_and_rejects_foreign_substrate(tmp_path):
    from steerlab_server.steering import vector_math as vm
    raw,meta=vector_artifact(tmp_path,[.5,1.],residualNormPerLayer=[7.5],residualNormConvention='perTextMean-v1')
    doc=norm_unit_settings(tmp_path)
    plan=owner.review(doc,tmp_path)
    action=plan['document']['actions'][0]
    assert action['vector']==[.5,1.] and action['source']['tensorSHA256']==hashlib.sha256(raw).hexdigest()
    # Review writes the default units and what execution needs, inside the hashed document.
    assert (action['strengthUnits'],action['residualNorm'],action['residualNormConvention'])==('residualNorm',7.5,'perTextMean-v1')
    assert action['vectorNorm']==vm.l2_norm([.5,1.]) and plan['planSHA256']==hashlib.sha256(owner.encode(plan['document'])).hexdigest()
    assert plan['actionUnits']==[{'id':'change','kind':'add','bounds':[0,.2],'strengthUnits':'residualNorm','residualNorm':7.5,
                                  'residualNormConvention':'perTextMean-v1','vectorNorm':action['vectorNorm']}]
    assert any('residual-norm units' in note for note in plan['limitations']) and not any('directly' in note for note in plan['limitations'])
    saved=owner.publish(doc,tmp_path,plan['planSHA256'])
    assert saved['actionUnits']==plan['actionUnits'] and owner.inspect(saved['path'],tmp_path)['actionUnits']==plan['actionUnits']
    path=tmp_path/'runs/fit/example.json';meta['substrate']='swift-mlx';path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='substrate'):owner.review(doc,tmp_path)


@pytest.mark.parametrize('case', ['absent', 'short', 'zero', 'numbers'])
def test_residual_norm_units_without_a_stamped_norm_refuse_with_a_repair(tmp_path, case, capsys):
    from steerlab_server.client_cli import main
    norms={'absent':{},'short':{'layerCount':2,'residualNormPerLayer':[3.]},'zero':{'residualNormPerLayer':[0.]},'numbers':{'residualNormPerLayer':[7.5]}}[case]
    vector_artifact(tmp_path,[.5,1.],**norms);doc=norm_unit_settings(tmp_path)
    if case=='numbers':doc['actions'][0].pop('vectorArtifactID');doc['actions'][0]['vector']=[.5,1.]
    with pytest.raises(artifacts.NormUnitsUnavailable) as raised:owner.review(doc,tmp_path)
    assert raised.value.code=='policyResidualNormMissing'
    assert 'backfill-norms' in raised.value.repair_action and '"strengthUnits": "rawMultiplier"' in raised.value.repair_action
    file=tmp_path/'settings.json';file.write_text(json.dumps(doc))
    capsys.readouterr();assert main(['science','policy-review',str(file),'--root',str(tmp_path),'--json'])==65
    error=json.loads(capsys.readouterr().out)['error']
    assert error['code']=='policyResidualNormMissing' and error['repairAction']==raised.value.repair_action
    # The repair: an explicit, recorded raw multiplier. Nothing about residual norms is recorded for it.
    doc['actions'][0]['strengthUnits']='rawMultiplier';plan=owner.review(doc,tmp_path)
    assert plan['actionUnits'][0]['strengthUnits']=='rawMultiplier' and not set(artifacts.NORM_FIELDS)&set(plan['document']['actions'][0])
    assert any('not comparable' in note for note in plan['limitations'])


def test_a_raw_multiplier_beside_a_stamped_norm_is_allowed_and_explained(tmp_path):
    vector_artifact(tmp_path,[.5,1.],residualNormPerLayer=[7.5]);doc=norm_unit_settings(tmp_path)
    doc['actions'][0]['strengthUnits']='rawMultiplier'
    notes=owner.review(doc,tmp_path)['limitations']
    assert any('has a stamped residual norm at layer 0' in note for note in notes)


def test_version_one_settings_refuse_at_review_and_published_ones_keep_their_raw_meaning(tmp_path):
    doc,_,model=setup(tmp_path)
    with pytest.raises(owner.VersionOneSettings) as raised:owner.review(doc,tmp_path)
    assert raised.value.code=='policyVersionOneSettings' and '"strengthUnits": "rawMultiplier"' in raised.value.repair_action
    # A published version-1 policy: strength 2 multiplies the stored direction, whatever its norm.
    doc['actions'][0]['vector']=[3.,4.];policy=runtime.Execution(attached(doc,tmp_path),rendering='rawCompletion')
    deltas=Deltas(Site('residualPost',0));rendered=prompt_render.RenderedPrompt('',[1,2,3],3)
    with model.hooked.session([]),deltas.observe_session(model,rendered),policy.observe_session(model,rendered):forward(model,[1,2,3])
    [(_,delta)]=deltas.passes
    assert delta[-1].tolist()==[6.,8.] and not delta[:-1].any()
    [row]=policy.result([4])['decisions']
    assert row['strengthUnits']=={'change':'rawMultiplier'} and row['appliedScales']=={'change':2.} and row['appliedStrengths']=={'change':2.}
    [summary]=artifacts.summaries(attached(doc,tmp_path))
    assert summary['schemaVersion']==1 and summary['actions']==[{'id':'change','kind':'add','bounds':[0,2],'strengthUnits':'rawMultiplier'}]
    # Version 1 has no unit declarations to read; one written into it is refused, not reinterpreted.
    published=owner.materialize(doc,tmp_path)
    for field,value in (('strengthUnits','residualNorm'),('residualNorm',1.)):
        changed=copy.deepcopy(published);changed['actions'][0][field]=value
        with pytest.raises(ValueError):artifacts.validate(changed)


def test_version_two_validation_checks_the_recorded_units(tmp_path):
    vector_artifact(tmp_path,[.5,1.],residualNormPerLayer=[7.5],residualNormConvention='perTextMean-v1')
    published=owner.review(norm_unit_settings(tmp_path),tmp_path)['document']
    artifacts.validate(published)
    for change in ({'strengthUnits':'other'},{'vectorNorm':2.},{'residualNorm':0},{'residualNorm':None},{'residualNormConvention':''}):
        broken=copy.deepcopy(published);broken['actions'][0].update(change)
        with pytest.raises(ValueError):artifacts.validate(broken)
    broken=copy.deepcopy(published);del broken['actions'][0]['residualNorm']
    with pytest.raises(ValueError,match='records residualNorm'):artifacts.validate(broken)
    # A legacy pre-stamp denominator is recorded as such, exactly as static steering reads it.
    vector_artifact(tmp_path,[.5,1.],residualNormPerLayer=[7.5])
    assert owner.review(norm_unit_settings(tmp_path),tmp_path)['document']['actions'][0]['residualNormConvention'] is None
    # Hand-entered norms are not stamped norms; review alone records them.
    settings=norm_unit_settings(tmp_path);settings['actions'][0]['residualNorm']=7.5
    with pytest.raises(ValueError,match='Review records residualNorm'):owner.review(settings,tmp_path)
    # R is only a stamped norm beside the artifact it came from, under a convention a sidecar can carry.
    for change in ({'residualNormConvention':'anything'},{'source':None},{'source':{'vectorArtifactID':'runs/fit/example'}},
                   {'source':{'metadataSHA256':'a'*64}}):
        broken=copy.deepcopy(published);broken['actions'][0].update(change)
        if broken['actions'][0]['source'] is None:del broken['actions'][0]['source']
        with pytest.raises(ValueError):artifacts.validate(broken)
    # The reviewer's reproduction: numbers, no source, an arbitrary R and convention.
    forged=copy.deepcopy(published);forged['actions'][0]=dict(id='change',kind='add',bounds=[0,.2],vector=[.5,1.],strengthUnits='residualNorm',
        residualNorm=12345.,residualNormConvention='anything',vectorNorm=published['actions'][0]['vectorNorm'])
    with pytest.raises(ValueError):artifacts.validate(forged)
    forged['actions'][0]['residualNormConvention']=None
    with pytest.raises(ValueError,match='vectorArtifactID'):artifacts.validate(forged)
    # Only additions declare units: ablation keeps its fraction, and token actions their own scales.
    ablation=norm_unit_settings(tmp_path);ablation['actions'][0].update(kind='ablate',strengthUnits='residualNorm');ablation['rules'][0]['value']=.1
    with pytest.raises(ValueError,match='Only additions'):owner.review(ablation,tmp_path)
    ablation['actions'][0].pop('strengthUnits');assert owner.review(ablation,tmp_path)['actionUnits'][0]['strengthUnits']=='fraction'


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
    doc,_,_=setup(tmp_path);doc=authored(doc);doc['actions'][0]['source']=value
    with pytest.raises(ValueError,match='provenance'):owner.review(doc,tmp_path)
    with pytest.raises(ValueError):owner.attachment_review({'agentPath':value,'name':'example','policyPaths':[]},tmp_path)


def tiny_llama():
    """A real one-layer Llama with no stop token: generate() runs its true background-thread path."""
    from transformers import LlamaConfig,LlamaForCausalLM,PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from steerlab_server.steering.hooks import HookedModel
    from steerlab_server.steering.model_loader import SteeredModel
    backend=Tokenizer(WordLevel({'[PAD]':0,'[EOS]':1,'[UNK]':2,'example':3,'selected':4},unk_token='[UNK]'));backend.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',eos_token='[EOS]',unk_token='[UNK]')
    torch.manual_seed(3)
    lm=LlamaForCausalLM(LlamaConfig(vocab_size=5,hidden_size=2,intermediate_size=4,num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,max_position_embeddings=64,pad_token_id=0,eos_token_id=None)).eval()
    return SteeredModel(model=lm,tokenizer=tokenizer,hooked=HookedModel(lm),model_id='example/model',revision='a'*40)


def conflicting_logits_policy(root, model):
    """The live failure: one fixed allowTokens and one fixed forceToken whose tokens never intersect."""
    doc,_,_=setup(root);doc['binding']['tokenizerSHA256']=runtime.probe_capture.tokenizer_identity(model.tokenizer)
    doc.update(site={'kind':'logitsPreSelection'},probes=[],
               actions=[{'id':'allow','kind':'allowTokens','bounds':[0,1],'tokens':[3]},{'id':'force','kind':'forceToken','bounds':[0,1],'tokens':[4]}],
               rules=[{'action':'allow','kind':'fixed','value':1},{'action':'force','kind':'fixed','value':1}])
    return doc


def bound_measurement(root, model):
    """A study probe measurement whose fitted probe is bound to this model's tokenizer."""
    from steerlab_server.experiment import diagnostic_archives as archives
    config,probe,_=prepared(root);probe['input']['tokenizerSHA256']=runtime.probe_capture.tokenizer_identity(model.tokenizer)
    source=root/config['probes'][0]['probe']['path'];source.write_text(json.dumps(probe));config['probes'][0]['probe']['sha256']=archives.file_hash(source)
    return config


def failing_study(root, model, doc, measurements=None):
    from steerlab_server.experiment import condition_execution as execution,manifest
    variant=ModelVariant(name='modified',base_model_id='example/model',base_revision='a'*40,prompt_mode='rawCompletion',intervention_policies=attached(doc,root))
    study=manifest.Manifest(name='example',model_id='example/model',raw={'probeMeasurements':measurements} if measurements else {},seeds=[0],max_tokens=4)
    eff=execution.EffectiveCondition('modified',[],{},'rawCompletion',None,False,0,variant=variant)
    directory=root/'runs/study';directory.mkdir()
    kwargs=dict(name='example',manifest=study,experiment_hash='b'*64,wants_choice=False,wants_sampled=True,reader_scorers=[],should_cancel=None,log=lambda _:None,root=root)
    return execution,eff,directory,kwargs


def only_sidecar(directory, kind='policy'):
    files=list(directory.glob(kind+'-failure-*.json'));assert len(files)==1
    return json.loads(files[0].read_text())


def no_records(directory):
    path=directory/'generations.jsonl'
    return not path.exists() or not path.read_text().strip()


def test_logits_conflict_during_real_generation_saves_one_sidecar_and_no_record(tmp_path):
    from contextlib import closing
    from steerlab_server.experiment import resume
    model=tiny_llama();execution,eff,directory,kwargs=failing_study(tmp_path,model,conflicting_logits_policy(tmp_path,model))
    with pytest.raises(ValueError,match='Policy token constraints conflict') as raised:
        with closing(resume.GenerationWriter(str(directory))) as writer:execution.execute_condition(model,eff,[{'id':'item','prompt':'example'}],writer,**kwargs)
    assert type(raised.value) is ValueError and no_records(directory)
    left=only_sidecar(directory)
    assert left['status']=='partial' and any('constraints conflict' in reason for reason in left['failures'])
    assert (left['promptID'],left['sampleIndex'],left['condition'])==('item',0,'modified')
    # The first decode step failed: nothing was sampled, and its decision says why.
    [row]=left['decisions'];assert left['outputTokenIDs']==[] and row['status']=='failed'
    assert {k:'constraints conflict' in v['reason'] for k,v in row['actionOutcomes'].items()}=={'allow':True,'force':True}
    assert row['inputTokenID']==left['promptTokenIDs'][-1] and row['predictedTokenID'] is None
    assert not model.hooked.runtime.subscriptions and not model.hooked._pre_handles


def test_policy_failure_beside_a_measurement_leaves_one_sidecar_each(tmp_path):
    from contextlib import closing
    from steerlab_server.experiment import resume
    model=tiny_llama();doc=conflicting_logits_policy(tmp_path,model)
    execution,eff,directory,kwargs=failing_study(tmp_path,model,doc,measurements=bound_measurement(tmp_path,model))
    with pytest.raises(ValueError,match='Policy token constraints conflict'):
        with closing(resume.GenerationWriter(str(directory))) as writer:execution.execute_condition(model,eff,[{'id':'item','prompt':'example'}],writer,**kwargs)
    assert no_records(directory) and only_sidecar(directory)['decisions']
    # The measurement keeps what it read before the policy failed, and names the policy's error as its reason.
    left=only_sidecar(directory,'probe');[row]=left['readings']
    assert left['status']=='partial' and len(left['failures'])==1 and 'constraints conflict' in left['failures'][0]
    assert (row['stage'],row['status'],row['predictedTokenID'])==('prefill','recorded',None) and left['outputTokenIDs']==[]


def test_a_sidecar_that_cannot_be_written_never_replaces_the_generation_error(tmp_path):
    from steerlab_server.experiment import probe_measurements,probe_observation
    from steerlab_server.experiment.generate import generate
    model=tiny_llama();doc=conflicting_logits_policy(tmp_path,model);config=bound_measurement(tmp_path,model)
    absent=tmp_path/'absent'
    measurement=probe_observation.create(config,probe_measurements.load(config,tmp_path),condition='modified',agent='modified',rendering='rawCompletion',run_directory=absent)
    policy=runtime.Execution(attached(doc,tmp_path),rendering='rawCompletion',run_directory=absent)
    with pytest.raises(ValueError,match='Policy token constraints conflict') as raised:
        generate(model,'example',model_id=model.model_id,prompt_mode='rawCompletion',max_tokens=2,temperature=0,observers=[measurement,policy])
    assert type(raised.value) is ValueError and not absent.exists()
    if hasattr(BaseException,'add_note'):
        assert [note.split(':')[0] for note in raised.value.__notes__]==['The policy-failure record could not be written','The probe-failure record could not be written']
    assert not model.hooked.runtime.subscriptions and not model.hooked._pre_handles


def test_residual_decide_failure_mid_decode_keeps_decisions_and_sampled_prefix(tmp_path):
    from contextlib import closing
    from steerlab_server.experiment import resume
    model=tiny_llama();doc,_,_=setup(tmp_path);doc['binding']['tokenizerSHA256']=runtime.probe_capture.tokenizer_identity(model.tokenizer)
    # The prefill decides; the first decode forward raises, under onError stop.
    (tmp_path/'provider.py').write_text('def decide(context, tensor, scores, state, rng, assets):\n    state["calls"] = state.get("calls", 0) + 1\n    if state["calls"] > 1: raise ValueError("example failure")\n    return [Decision("change", 1)]\n')
    doc.update(probes=[],rules=[],provider={'sourcePath':'provider.py','assetPaths':{}})
    execution,eff,directory,kwargs=failing_study(tmp_path,model,doc)
    with pytest.raises(ValueError,match='example failure'):
        with closing(resume.GenerationWriter(str(directory))) as writer:execution.execute_condition(model,eff,[{'id':'item','prompt':'example'}],writer,**kwargs)
    assert no_records(directory)
    left=only_sidecar(directory);assert left['status']=='partial' and 'example-policy: example failure' in left['failures']
    [sampled]=left['outputTokenIDs'];decided,failed=left['decisions']
    assert decided['status']=='applied' and decided['predictedTokenID']==sampled and decided['generatedTokenIndex']==0
    assert failed['status']=='skipped' and failed['reason']=='example failure'
    assert failed['inputTokenID']==sampled and failed['predictedTokenID'] is None
    assert not model.hooked.runtime.subscriptions


def test_panel_turn_generation_failure_saves_sidecar_and_writes_no_turn(tmp_path,monkeypatch):
    from steerlab_server.experiment import multi_agent
    from test_multi_agent import _scenario
    model=tiny_llama();doc=conflicting_logits_policy(tmp_path,model)
    variant=ModelVariant(name='modified',base_model_id='example/model',base_revision='a'*40,prompt_mode='rawCompletion',intervention_policies=attached(doc,tmp_path))
    path=tmp_path/'runs/fit/agent.json';path.write_text(json.dumps(variant.to_dict()))
    monkeypatch.setenv('STEERLAB_ROOT',str(tmp_path))
    scenario=_scenario();scenario.max_tokens=2
    for agent in scenario.agents:agent.variant_artifact_path=str(path);agent.variant_artifact_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    directory=tmp_path/'configured';directory.mkdir()
    with pytest.raises(ValueError,match='Policy token constraints conflict'):
        multi_agent.run_scenario(model,scenario,run_dir=str(directory),log=lambda _:None)
    left=only_sidecar(directory);assert (left['turnID'],left['agent'])==('t1','a') and left['decisions']
    turns=directory/'turns.jsonl';assert not turns.exists() or not turns.read_text().strip()


CHAT=[{'role':'user','content':'example'},{'role':'assistant','content':'example','seeded':True},{'role':'user','content':'example'}]


def test_multi_turn_agent_chat_runs_the_policy_and_returns_its_decisions(tmp_path):
    from steerlab_server.api import variant_chat
    model=tiny_llama();doc,_,_=setup(tmp_path);doc['binding']['tokenizerSHA256']=runtime.probe_capture.tokenizer_identity(model.tokenizer)
    doc.update(site={'kind':'logitsPreSelection'},probes=[],actions=[{'id':'change','kind':'forceToken','bounds':[0,1],'tokens':[4]}],rules=[{'action':'change','kind':'fixed','value':1}])
    variant=ModelVariant(name='modified',base_model_id='example/model',base_revision='a'*40,prompt_mode='rawCompletion',temperature=0,intervention_policies=attached(doc,tmp_path))
    def request(**changes):return variant_chat.VariantChatRequest(variant_path='p',messages=CHAT,max_tokens=2,**changes)
    result=variant_chat.generate_with_variant(model,variant,request())
    decisions=result['interventionDecisions'];prompt=len(decisions['promptTokenIDs'])
    assert decisions['outputTokenIDs']==[4,4] and decisions['status']=='complete' and decisions['mode']=='playground'
    assert [r['inputTokenPosition'] for r in decisions['decisions']]==[prompt-1,prompt] and result['seededMessageIndices']==[1]
    # Streaming executes the same policy; its text stream carries no decision records.
    streamed=''.join(variant_chat.stream_with_variant(model,variant,request()))
    baseline=variant_chat.generate_with_variant(model,variant,request(strip_interventions=True))
    assert streamed==result['output']!=baseline['output'] and 'interventionDecisions' not in baseline
    assert not model.hooked.runtime.subscriptions and not model.hooked._pre_handles


def test_a_norm_unit_policy_adds_what_a_static_agent_adds_at_the_same_positions(tmp_path):
    """The equivalence the units promise, on a real model through generate(): a fixed residual-norm strength α over
    prefill and decode at lastPosition makes the same addition, at the same positions, as a static agent at α."""
    from steerlab_server.experiment.generate import generate
    from steerlab_server.experiment.model_variant import variant_injections
    model=tiny_llama();alpha=.1;site=Site('residualPost',0)
    vector_artifact(tmp_path,[.3,-.4],residualNormPerLayer=[7.5],residualNormConvention='perTextMean-v1',revision='a'*40)
    static=ModelVariant(name='static',base_model_id='example/model',base_revision='a'*40,prompt_mode='rawCompletion',alpha_in_norm_units=True,
                        injections=[{'concept':'example','vectorArtifactID':'runs/fit/example','layer':0,'alpha':alpha}])
    cells=variant_injections(static,root=str(tmp_path))
    settings=norm_unit_settings(tmp_path,alpha);settings.update(probes=[])
    settings['binding']['tokenizerSHA256']=runtime.probe_capture.tokenizer_identity(model.tokenizer)
    raw=owner.encode(owner.review(settings,tmp_path)['document']);attachment=[{'json':raw.decode(),'sha256':hashlib.sha256(raw).hexdigest()}]
    policy=runtime.Execution(attachment,rendering='rawCompletion')
    runs={}
    for name,injections,observers in (('static',cells,[]),('policy',[],[policy])):
        deltas=Deltas(site);ids=[]
        generate(model,'example selected example',model_id=model.model_id,prompt_mode='rawCompletion',max_tokens=3,temperature=0,
                 injections=injections,observers=[deltas,*observers],token_ids_out=ids)
        runs[name]=(deltas.passes,ids)
    (static_passes,static_ids),(policy_passes,policy_ids)=runs['static'],runs['policy']
    assert static_ids==policy_ids and [o for o,_ in static_passes]==[o for o,_ in policy_passes]==[0,3,4]
    for (_,expected),(_,actual) in zip(static_passes,policy_passes):
        changed=expected.abs().sum(-1)>0
        assert torch.equal(changed,actual.abs().sum(-1)>0) and changed[-1] and not changed[:-1].any()
        assert torch.allclose(actual,expected,rtol=1e-6,atol=1e-7)
        # Its norm is α times the residual norm: what "in residual-norm units" means.
        assert actual[-1].norm().item()==pytest.approx(alpha*7.5,rel=1e-5)
    # Evidence states the units, and the absolute scale is, bit for bit, the static agent's raw scalar.
    rows=policy.result(policy_ids)['decisions']
    assert [r['inputTokenPosition'] for r in rows]==[2,3,4]
    for row in rows:
        assert row['strengths']==row['appliedStrengths']=={'change':pytest.approx(alpha)} and row['strengthUnits']=={'change':'residualNorm'}
        assert row['appliedScales']['change']==cells[0].alpha
    [summary]=artifacts.summaries(attachment)
    assert summary['schemaVersion']==2 and summary['actions'][0]['strengthUnits']=='residualNorm' and summary['actions'][0]['residualNorm']==7.5


def test_attachment_review_refuses_a_policy_whose_sidecar_now_gives_another_norm(tmp_path):
    _,meta=vector_artifact(tmp_path,[.5,1.],residualNormPerLayer=[7.5],residualNormConvention='perTextMean-v1')
    doc=norm_unit_settings(tmp_path);saved=owner.publish(doc,tmp_path,owner.review(doc,tmp_path)['planSHA256'])
    (tmp_path/'runs/fit/agent.json').write_text(json.dumps(ModelVariant(name='example',base_model_id='example/model',base_revision='a'*40,prompt_mode='rawCompletion').to_dict()))
    settings={'agentPath':'runs/fit/agent.json','name':'modified','policyPaths':[saved['path']]}
    owner.attachment_review(settings,tmp_path)
    sidecar=tmp_path/'runs/fit/example.json'
    for change in ({'residualNormPerLayer':[8.]},{'residualNormConvention':'wholeCorpusMean-v1'},{'residualNormPerLayer':None}):
        sidecar.write_text(json.dumps({**meta,**change}))
        with pytest.raises(owner.NormsChanged) as raised:owner.attachment_review(settings,tmp_path)
        assert raised.value.code=='policyResidualNormChanged' and 'Review and publish the policy again' in raised.value.repair_action
    # A policy reviewed in another workspace carries its own record; an absent sidecar is not a disagreement.
    sidecar.unlink();plan=owner.attachment_review(settings,tmp_path)
    assert owner.attach(settings,tmp_path,plan['planSHA256'])['changed']


def test_the_scope_sidecar_states_policy_units(tmp_path):
    from steerlab_server.experiment import intervention_scope
    vector_artifact(tmp_path,[.5,1.],residualNormPerLayer=[7.5]);raw=owner.encode(owner.review(norm_unit_settings(tmp_path),tmp_path)['document'])
    variant=ModelVariant(name='modified',base_model_id='example/model',base_revision='a'*40,prompt_mode='rawCompletion',
                         intervention_policies=[{'json':raw.decode(),'sha256':hashlib.sha256(raw).hexdigest()}])
    path=intervention_scope.stamp_run(str(tmp_path),experiment='example',variant_conditions=[SimpleNamespace(name='modified')],
                                      resolve_variant=lambda _:SimpleNamespace(variant=variant,intervention_state={},injections=[]))
    [entry]=json.loads(Path(path).read_text())['conditions']
    [policy]=entry['interventionPolicies']
    assert policy['schemaVersion']==2 and policy['actions'][0]['strengthUnits']=='residualNorm' and policy['actions'][0]['residualNorm']==7.5
