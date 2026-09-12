import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from steerlab_server.experiment import probe_capture as capture, probe_data as data, probe_artifacts as artifact, diagnostic_archives as archives


class Block(torch.nn.Module):
    def forward(self, hidden_states): return (hidden_states*2+1,)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([Block(),Block()])
        self.parameter=torch.nn.Parameter(torch.tensor(1.))
    def forward(self,input_ids,attention_mask,use_cache=False):
        h=torch.stack([input_ids.float(),input_ids.float()+1],dim=-1)*self.parameter
        for layer in self.model.layers:h=layer(hidden_states=h)[0]
        return h


class Tokens:
    def __call__(self,text,**kwargs):
        n=min(len(text.split()),kwargs['max_length']);ids=torch.arange(1,n+1)[None,:]
        return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}
    def apply_chat_template(self,messages,**kwargs):
        assert kwargs==dict(tokenize=False,add_generation_prompt=True)
        return 'user '+messages[0]['content']+' assistant'


def capture_input(root):
    rows=[{'id':str(i),'group':'group-'+str(i),'label':bool(i%2),'text':f'Example text {i}',
           'split':'fit' if i<4 else 'selection' if i<6 else 'finalTest'} for i in range(8)]
    path=root/'examples.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return {'path':path.name,'sha256':archives.file_hash(path)}


@pytest.mark.parametrize('site,expected',[('residualPre',[3,4]),('residualPost',[7,9])])
def test_hooks_read_exact_sites_remove_handles_and_do_not_mutate_outputs(site,expected):
    model=Toy().eval();block=model.model.layers[0]
    encoded={'input_ids':torch.tensor([[1,2,3,0]]),'attention_mask':torch.tensor([[1,1,1,0]])}
    before=model(**encoded).detach().clone();state=torch.random.get_rng_state().clone()
    cfg=SimpleNamespace(site=site,position='lastNonPadding')
    positions,values,dtype=capture.observe(model,block,encoded,cfg)
    assert positions==[2] and values==[expected] and dtype=='float32'
    assert torch.equal(model(**encoded),before) and torch.equal(torch.random.get_rng_state(),state)
    assert not block._forward_hooks and not block._forward_pre_hooks


def test_capture_split_files_retain_binding_positions_and_can_fit(tmp_path,monkeypatch):
    from steerlab_server.experiment import probe_training as training
    model=Toy().eval()
    monkeypatch.setattr(capture,'load',lambda config,log:(model,Tokens(),{'tokenizerSHA256':'b'*64,'templateSHA256':None}))
    cfg=capture.CaptureConfig.from_dict({'modelID':'example/model','revision':'a'*40,'examples':capture_input(tmp_path),'layer':0,'device':'cpu'})
    seen=[];result=capture.capture(cfg,root=tmp_path,log=lambda _:None,on_run_created=seen.append)
    assert seen==[result['runDirectory']]
    report=json.loads(Path(result['reportPath']).read_text())
    assert report['counts']['fit']['rows']==4 and report['counts']['finalTest']['rows']==2
    assert all(row['positions']==[2] for row in report['alignment'])
    assert report['input']['coordinateConvention']=='hf-decoder-block-v1/model.layers'
    files=result['datasets'];ref=files['fit'];ref['path']=str(Path(ref['path']).relative_to(tmp_path))
    doc=data.dataset(ref,tmp_path);assert doc['rows'][0]['activation']==[7,9]
    trained=training.train(training.TrainConfig.from_dict({'fitData':ref,'label':'Example','steps':2}),root=tmp_path,log=lambda _:None)
    assert Path(trained['artifactPath']).exists()


def test_hook_failure_removes_handle_and_cap_never_silently_drops(tmp_path,monkeypatch):
    model=Toy().eval();block=model.model.layers[0]
    cfg=SimpleNamespace(site='residualPost',position='lastNonPadding')
    with pytest.raises(artifact.ProbeError):capture.observe(model,block,{'input_ids':torch.ones((2,3),dtype=torch.long),'attention_mask':torch.ones((2,3),dtype=torch.long)},cfg)
    assert not block._forward_hooks
    monkeypatch.setattr(capture,'load',lambda config,log:(model,Tokens(),{'tokenizerSHA256':'b'*64,'templateSHA256':None}))
    config=capture.CaptureConfig.from_dict({'modelID':'example/model','revision':'a'*40,'examples':capture_input(tmp_path),'layer':0,'device':'cpu','position':'eachNonPadding','maxRecords':1})
    with pytest.raises(artifact.ProbeError,match='maxRecords'):capture.capture(config,root=tmp_path,log=lambda _:None)
    assert not list((tmp_path/'runs').glob('*/COMPLETED'))


def test_group_hash_order_independent_and_explicit_overlap_refuses(tmp_path):
    rows=[{'id':str(i),'group':str(i//2),'text':str(i),'label':bool(i%2)} for i in range(60)]
    def encoded(rows):return b'\n'.join(archives.encoded(r) for r in rows)
    first=data.text_rows(encoded(rows),'groupHash',7);second=data.text_rows(encoded(list(reversed(rows))),'groupHash',7)
    assert {r['id']:r['split'] for r in first}=={r['id']:r['split'] for r in second}
    assert set(r['split'] for r in first)==set(data.ROLES)
    assert all(len({r['split'] for r in first if r['group']==g})==1 for g in {r['group'] for r in first})
    rows[0]['split']='fit';rows[1]['split']='finalTest'
    with pytest.raises(artifact.ProbeError,match='crosses'):data.text_rows(encoded(rows[:2]),'explicit',0)


def test_unknown_model_path_does_not_guess_modulelist():
    model=torch.nn.Module();model.unrelated=torch.nn.ModuleList([Block(),Block()])
    with pytest.raises(artifact.ProbeError,match='unknown'):capture.decoder_layers(model)


def test_tokenizer_identity_covers_fast_normalization_and_keeps_slow_unknown():
    tokenizer=SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda:'normalization A'),special_tokens_map={})
    first=capture.tokenizer_identity(tokenizer)
    tokenizer.backend_tokenizer.to_str=lambda:'normalization B'
    second=capture.tokenizer_identity(tokenizer)
    assert second!=first
    tokenizer.add_bos_token=True
    assert capture.tokenizer_identity(tokenizer)!=second
    assert capture.tokenizer_identity(SimpleNamespace(get_vocab=lambda:{'same':1})) is None
