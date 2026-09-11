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


def test_stopping_checkpoint_cannot_turn_out_of_range_values_into_convergence():
    from steerlab_server.experiment.jlens_stopping import Control, STATISTIC
    rule={'threshold':0.1,'window':2,'minPrompts':3}
    saved={'statistic':STATISTIC,'values':[0.05,0.04],'count':3,'lastValue':0.04}
    assert Control(rule,saved,3).reached
    for changed in ({'values':[0.2,0.04]},{'count':True},{'lastValue':0.03}):
        with pytest.raises(ValueError):Control(rule,{**saved,**changed},3)


def test_merge_detects_source_mutation_during_loading(fitting,monkeypatch):
    from pathlib import Path
    from safetensors import torch as tensors
    from steerlab_server.experiment import jlens_merge
    root,cfg=fitting
    fitted=run(root,cfg);directory=Path(fitted['runDirectory'])
    original=tensors.load_file
    def changed(path,*args,**kwargs):
        result=original(path,*args,**kwargs)
        if path.endswith('/jacobians.safetensors'):
            report=directory/'fit-report.json';report.write_bytes(report.read_bytes()+b' ')
        return result
    monkeypatch.setattr(tensors,'load_file',changed)
    before=set((root/'runs').iterdir())
    config=jlens_merge.MergeConfig.from_dict({'fits':[str(directory.relative_to(root))]})
    with pytest.raises(ValueError,match='changed while being read'):jlens_merge.merge(config,root=root)
    assert set((root/'runs').iterdir())==before
