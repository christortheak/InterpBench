import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from steerlab_server.experiment import probe_measurements as owner, probe_observation as runtime, probe_artifacts as artifacts, diagnostic_archives as archives, prompt_render
from steerlab_server.client.diagnostic_commands import workspace_action
from steerlab_server.steering.hooks import HookedModel
from test_probe_capture import Toy, Tokens


def prepared(root, site='residualPost', position='lastNonPadding'):
    doc = json.loads((Path(__file__).resolve().parents[2]/'Tests/Fixtures/cross-engine/probe-artifacts.json').read_text())['linear']
    doc['input'].update(modelID='example/model', revision='a'*40, substrate='pytorch', precision='float32', coordinateConvention='hf-decoder-block-v1/model.layers', tokenizerSHA256=None, templateSHA256=None)
    doc['input']['site'] = {'kind': site, 'layer': 0}
    doc['input']['reading'] = {'rendering':'raw', 'position':position, 'population':'prompt'}
    folder = root/'runs'/'fit'; folder.mkdir(parents=True, exist_ok=True)
    path=folder/'trained.probe.json'; path.write_text(json.dumps(doc))
    item={'id':'reader', 'probe':{'path':path.relative_to(root).as_posix(), 'sha256':archives.file_hash(path)}, 'conditions':[], 'agents':[], 'stages':['prefill','decode'], 'recordingStage':'postAction'}
    config={'schemaVersion':1,'probes':[item],'onError':'recordMissing','maxReadings':100,'retainActivations':False,'maxActivationBytes':1024}
    model=Toy().eval(); loaded=SimpleNamespace(model=model, model_id='example/model', revision='a'*40, tokenizer=Tokens(), hooked=HookedModel(model))
    return config, doc, loaded


@pytest.mark.parametrize('site,activation', [('residualPre',[3,4]), ('residualPost',[7,9])])
def test_real_hook_scores_alignment_and_output_rng_preservation(tmp_path,site,activation):
    config,doc,model=prepared(tmp_path,site)
    observer=runtime.create(config,owner.load(config,tmp_path),condition='baseline',agent='baseline',rendering='rawCompletion')
    rendered=prompt_render.RenderedPrompt('example',[1,2,3],3)
    before=torch.random.get_rng_state().clone()
    encoded=dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones((1,3)))
    expected=model.model(**encoded).clone()
    with model.hooked.session([]), observer.observe_session(model,rendered):
        actual=model.model(**encoded)
        model.model(input_ids=torch.tensor([[4]]),attention_mask=torch.ones((1,1)))
    result=observer.result([4,5])
    assert torch.equal(actual,expected) and torch.equal(before,torch.random.get_rng_state())
    rows=result['readings']; assert [(r['stage'],r['inputTokenPosition'],r['predictedTokenID']) for r in rows]==[('prefill',2,4),('decode',3,5)]
    assert rows[0]['score']==artifacts.score(doc,activation,input_binding=doc['input'])['score']
    assert len(model.model.model.layers[0]._forward_hooks)==1  # original HookedModel only
    assert not model.model.model.layers[0]._forward_pre_hooks


def test_chunked_prefill_scopes_action_order_and_budgets(tmp_path):
    config,doc,model=prepared(tmp_path,position='eachNonPadding')
    config['maxReadings']=3; config['retainActivations']=True; config['maxActivationBytes']=10
    other=copy.deepcopy(config['probes'][0]);other.update(id='before',recordingStage='preAction')
    config['probes'].append(other)
    class Add:
        def apply(self,h,layer,offset): return h+10 if layer==0 else h
    observer=runtime.create(config,owner.load(config,tmp_path),condition='baseline',agent='seat',rendering='rawCompletion')
    with model.hooked.session([Add()]),observer.observe_session(model,prompt_render.RenderedPrompt('',[1,2,3],3)):
        model.model(input_ids=torch.tensor([[1,2]]),attention_mask=torch.ones((1,2)))
        model.model(input_ids=torch.tensor([[3]]),attention_mask=torch.ones((1,1)))
    result=observer.result([4]); assert result['status']=='partial' and result['omittedReadings']==3
    assert result['activationBytes']==10 and sum('activation' in r for r in result['readings'])==1
    before=[r for r in result['readings'] if r['measurementID']=='before']
    after=[r for r in result['readings'] if r['measurementID']=='reader']
    assert before[0]['score']==artifacts.score(doc,[3,5],input_binding=doc['input'])['score']
    assert after[0]['score']==artifacts.score(doc,[13,15],input_binding=doc['input'])['score']
    config['probes'][0]['conditions']=['other'];config['probes'][1]['agents']=['other']
    assert runtime.create(config,owner.load(config,tmp_path),condition='baseline',agent='seat',rendering='rawCompletion') is None


def test_nonfinite_recorded_missing_or_stops_and_hooks_removed(tmp_path):
    config,doc,model=prepared(tmp_path)
    model.model.parameter.data.fill_(float('nan'))
    for policy in ['recordMissing','stop']:
        config['onError']=policy
        observer=runtime.create(config,owner.load(config,tmp_path),condition='baseline',agent='baseline',rendering='rawCompletion')
        def run():
            with model.hooked.session([]),observer.observe_session(model,prompt_render.RenderedPrompt('',[1],1)):
                model.model(input_ids=torch.tensor([[1]]),attention_mask=torch.ones((1,1)))
        if policy=='stop':
            with pytest.raises(artifacts.ProbeError):run()
        else:
            run();row=observer.result([])['readings'][0]
            assert row['status']=='missing' and 'score' not in row
        assert len(model.model.model.layers[0]._forward_hooks)==1


def test_review_save_stale_bytes_frozen_protection_and_dependency_closure(tmp_path):
    from steerlab_server.experiment import experiment_store, manifest as manifest_module, bundles
    config,doc,model=prepared(tmp_path)
    experiment_store.save_raw({'name':'example','status':'draft','modelID':'example/model','concepts':[], 'conditions':[]},tmp_path)
    payload={'workspaceRoot':str(tmp_path),'experiment':'example','settingsText':json.dumps(config)}
    review=workspace_action('measurements-review',payload)
    result=workspace_action('measurements-save',{**payload,'planSHA256':review['planSHA256']})
    assert result['changed'] and result['study']['probeMeasurements']==config
    entries=experiment_store.pinned_input_entries(result['study'],tmp_path)
    assert any(e.path.endswith('trained.probe.json') for e in entries)
    m=manifest_module.Manifest.load('example',tmp_path)
    files=bundles._experiment_files(m,str(tmp_path),tmp_path)
    assert config['probes'][0]['probe']['path'] in {rel for _,rel in files}
    fresh=owner.review('example',config,tmp_path)
    source=tmp_path/config['probes'][0]['probe']['path'];source.write_text(source.read_text()+' ')
    with pytest.raises(artifacts.ProbeError,match='bytes changed'):owner.save('example',config,tmp_path,fresh['planSHA256'])
    assert any('bytes changed' in x for x in manifest_module.Manifest.load('example',tmp_path).verify(tmp_path))


def test_invalid_contract_and_binding_never_silent(tmp_path):
    config,doc,model=prepared(tmp_path)
    for bad in [dict(config,maxReadings=True),dict(config,unknown=1),dict(config,onError='ignore')]:
        with pytest.raises(artifacts.ProbeError):owner.validate(bad)
    model.revision='b'*40
    observer=runtime.create(config,owner.load(config,tmp_path),condition='baseline',agent='baseline',rendering='rawCompletion')
    with pytest.raises(artifacts.ProbeError,match='revision'):
        with observer.observe_session(model,prompt_render.RenderedPrompt('',[1],1)): pass


def test_actual_transformers_generation_preserves_sample_stream_and_attaches_readings(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from steerlab_server.experiment import generate, probe_capture
    from steerlab_server.steering.model_loader import SteeredModel
    config, doc, _ = prepared(tmp_path)
    backend = Tokenizer(WordLevel({'<unk>':0, '<s>':1, '</s>':2, 'one':3, 'two':4, 'three':5}, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>', bos_token='<s>', eos_token='</s>', pad_token='</s>')
    torch.manual_seed(42)
    network = LlamaForCausalLM(LlamaConfig(vocab_size=6, hidden_size=2, intermediate_size=4, num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1, max_position_embeddings=64, eos_token_id=None)).eval()
    model = SteeredModel(network, tokenizer, HookedModel(network), 'example/model', 'a'*40)
    doc['input']['tokenizerSHA256'] = probe_capture.tokenizer_identity(tokenizer)
    source = tmp_path/config['probes'][0]['probe']['path'];source.write_text(json.dumps(doc));config['probes'][0]['probe']['sha256'] = archives.file_hash(source)
    class ExistingReader:
        def __init__(self): self.values=[]
        def apply(self,h,layer,offset):
            if layer == 0: self.values.append(h.detach()[0,-1,:].tolist())
            return h
    existing=ExistingReader()
    outputs=[];ids=[];states=[]
    for enabled in (False,True):
        observer=runtime.create(config,owner.load(config,tmp_path),condition='baseline',agent='baseline',rendering='rawCompletion') if enabled else None
        token_ids=[];torch.manual_seed(123)
        kwargs={'observers':[existing,observer]} if observer else {}
        outputs.append(generate.generate(model,'one two three',prompt_mode='rawCompletion',max_tokens=4,temperature=.7,injections=[generate.CellInjection(0,[.1,-.1],.2)],token_ids_out=token_ids,**kwargs))
        ids.append(token_ids);states.append(torch.random.get_rng_state())
    assert outputs[0]==outputs[1] and ids[0]==ids[1] and torch.equal(states[0],states[1])
    result=observer.result(ids[1]);assert result['status']=='complete'
    assert len(result['readings'])==len(ids[1])
    assert result['readings'][0]['predictedTokenID']==ids[1][0]
    assert result['readings'][0]['score'] == pytest.approx(artifacts.score(doc, existing.values[0], input_binding=doc['input'])['score'])
    assert all(r['inputTokenPosition'] < len(result['promptTokenIDs'])+len(ids[1])-1 for r in result['readings'])
