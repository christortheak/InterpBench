"""Cross-surface authoring and failure regressions for fitting scaling."""
import json
import sys
import types
import pytest
from steerlab_server.experiment import method_authoring, managed_methods, diagnostic_archives as archives
from steerlab_server.experiment.jlens_kernel_policy import Selection
from test_jlens_fit import fitting, run


def test_kernel_selection_is_explicit_reversible_and_rejects_unknown_fallback(monkeypatch):
    name='transformers.models.scaling_fixture'
    owner=types.ModuleType(name)
    exec('def torch_causal_conv1d_fn(*args): return 1\nclass Layer: pass',owner.__dict__)
    def fused(*args):return 2
    owner.causal_conv1d_fn=fused
    monkeypatch.setitem(sys.modules,name,owner)
    class Model:
        def modules(self):return [owner.Layer()]
    current=Selection(Model(),'current')
    assert owner.causal_conv1d_fn is fused
    current.close()
    fallback=Selection(Model(),'torch')
    assert owner.causal_conv1d_fn is owner.torch_causal_conv1d_fn
    fallback.close()
    assert owner.causal_conv1d_fn is fused
    del owner.torch_causal_conv1d_fn
    with pytest.raises(ValueError,match='no inspectable Torch fallback'):Selection(Model(),'torch')
    assert owner.causal_conv1d_fn is fused


def scaling_interview_fields(fitting):
    from steerlab_server.experiment import artifact_imports
    root,cfg=fitting
    request=root/'pilot.json'
    request.write_text(json.dumps({'operation':'jlens-fit','parameters':{'config':cfg}}))
    fitted=run(root,cfg)
    review=artifact_imports.inspect_source(fitted['artifactDescription'],root)
    lens=artifact_imports.publish(fitted['artifactDescription'],root,review['planSHA256'])
    scenarios={
        'jlens-fit-benchmark':{'modelID':cfg['modelID'],'revision':cfg['revision'],'fittingRequest':'pilot.json'},
        'jlens-fit-round':{'modelID':cfg['modelID'],'revision':cfg['revision'],'fittingRequest':'pilot.json','shards':'2'},
        'jlens-fit-merge':{'fits':str(__import__('pathlib').Path(fitted['runDirectory']).relative_to(root))},
        'jlens-fit-assess':{'modelID':cfg['modelID'],'revision':cfg['revision'],'corpus':'corpus.jsonl',
            'referenceLensID':lens['lensID'],'candidateLensID':lens['lensID'],'device':'cpu','dtype':'float32'},
    }
    return scenarios


def test_new_interviews_draft_through_real_owners(fitting):
    root,_=fitting
    scenarios=scaling_interview_fields(fitting)
    for operation,fields in scenarios.items():
        answer={'purpose':'Compare fitted readouts','claim':'Readout stability on selected text',
                'controls':'Pinned model and corpus','selection':'Declared before assessment',
                'fields':fields,'advanced':{}}
        plan=method_authoring.draft(operation,answer,root)
        config=plan['request']['parameters']['config']
        assert managed_methods.validate(operation,config,root)
        review=plan['operationReview']
        if operation=='jlens-fit-benchmark':assert review['totalRowBudget']==12
        elif operation=='jlens-fit-round':assert review['globalRowBudget']==4
        elif operation=='jlens-fit-merge':assert review['missingRows']==[]
        else:assert review['rows']==4
        from steerlab_server.api.scientific_execution import input_plan
        assert input_plan(plan['request'],root)['operationReview']==review


def test_stopping_checkpoint_cannot_turn_out_of_range_values_into_convergence():
    from steerlab_server.experiment.jlens_stopping import Control, STATISTIC
    rule={'threshold':0.1,'window':2,'minPrompts':3}
    saved={'statistic':STATISTIC,'values':[0.05,0.04],'count':3,'lastValue':0.04}
    assert Control(rule,saved,3).reached
    for changed in ({'values':[0.2,0.04]},{'count':True},{'lastValue':0.03}):
        with pytest.raises(ValueError):Control(rule,{**saved,**changed},3)


def test_merge_detects_source_mutation_during_loading(fitting,monkeypatch):
    from pathlib import Path
    import safetensors
    from steerlab_server.experiment import jlens_merge
    root,cfg=fitting
    fitted=run(root,cfg);directory=Path(fitted['runDirectory'])
    original=safetensors.safe_open
    def changed(path,*args,**kwargs):
        # The merge streams one layer at a time; mutate a sibling once the
        # published mean is opened, after the fingerprints were taken.
        if str(path).endswith('/jacobians.safetensors'):
            report=directory/'fit-report.json';report.write_bytes(report.read_bytes()+b' ')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(safetensors,'safe_open',changed)
    before=set((root/'runs').iterdir())
    config=jlens_merge.MergeConfig.from_dict({'fits':[str(directory.relative_to(root))]})
    with pytest.raises(ValueError,match='changed while being read'):jlens_merge.merge(config,root=root)
    assert set((root/'runs').iterdir())==before


def test_benchmark_records_device_without_host_or_site_identity():
    from types import SimpleNamespace
    from steerlab_server.experiment.jlens_benchmark import hardware
    cuda=SimpleNamespace(is_available=lambda:True,get_device_properties=lambda device:SimpleNamespace(
        name='Fixture GPU',total_memory=80*1024**3,major=9,minor=0))
    result=hardware(SimpleNamespace(cuda=cuda,version=SimpleNamespace(cuda='fixture')),'cuda:0')
    assert result['deviceName']=='Fixture GPU' and result['computeCapability']==[9,0]
    assert result['deviceCapacityBytes']==80*1024**3
    assert set(result)=={'requestedDevice','machine','cudaBuild','deviceName','deviceCapacityBytes','computeCapability'}


def test_benchmark_worker_isolates_compiler_caches(tmp_path,monkeypatch):
    import os
    from types import SimpleNamespace
    from steerlab_server.experiment import jlens_benchmark
    def child(command,*,env,input,stdout,stderr):
        assert env['TORCHINDUCTOR_CACHE_DIR']==str(tmp_path/'inductor-cache')
        assert env['TRITON_CACHE_DIR']==str(tmp_path/'triton-cache')
        (tmp_path/'result.json').write_text('{"fixture":true}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setenv('TORCHINDUCTOR_CACHE_DIR','original-fixture-cache')
    monkeypatch.setattr(jlens_benchmark.subprocess,'run',child)
    assert jlens_benchmark.subprocess_worker({},tmp_path,tmp_path)=={'fixture':True}
    assert os.environ['TORCHINDUCTOR_CACHE_DIR']=='original-fixture-cache'
