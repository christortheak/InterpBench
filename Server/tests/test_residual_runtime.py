"""Independent runtime acceptance, including owners' real tensor arithmetic."""
import pytest
import torch
from torch import nn
from steerlab_server.steering.runtime import Runtime, Reading, Site
from steerlab_server.steering.hooks import HookedModel
from steerlab_server.steering.plan import Edit, Mode, interventions
from steerlab_server.steering.injector import VectorInjector
from steerlab_server.steering.ablator import SubspaceAblator


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_actions_retain_sequential_rounding_gate_and_ablation_units(dtype):
    chain = interventions([Edit(0,[1,0,0],.5,Mode.ABLATE,'axis'),
                          Edit(0,[1,1,0],2,Mode.ADD,'offset'),
                          Edit(0,[0,0,1],.125,Mode.ADD,'other')], prompt_token_count=4)
    runtime = Runtime(); seen=[]
    # Reverse subscription order deliberately; phase order is not registration order.
    runtime.subscribe([Reading('after',Site('residualPost',0),'postAction',lambda h,c,s:seen.append(('after',h.clone(),tuple(c.positions),c.stages))),
                       Reading('before',Site('residualPost',0),'preAction',lambda h,c,s:seen.append(('before',h.clone(),tuple(c.positions),c.stages)))], prompt_token_count=4)
    h = torch.tensor([[[4,3,2],[8,5,6]]],dtype=dtype)
    original=h.clone();rng=torch.random.get_rng_state().clone()
    mid=runtime.apply(h,Site('residualPost',0),0,chain)
    assert torch.equal(mid,torch.tensor([[[2,3,2],[4,5,6]]],dtype=dtype))
    tail=runtime.apply(h,Site('residualPost',0),2,chain)
    assert torch.equal(tail,torch.tensor([[[2,3,2],[6,7,6.125]]],dtype=dtype))
    direct=h
    for action in chain: direct=action.apply(direct,0,2)
    assert torch.equal(tail,direct)
    assert [x[0] for x in seen]==['before','after','before','after']
    assert seen[-1][2:]==((2,3),('prefill','prefill'))
    assert torch.equal(h,original) and torch.equal(torch.random.get_rng_state(),rng)
    off=Runtime(); assert off.apply(h,Site('residualPost',0),0) is h


def test_tensor_gradient_survives_legacy_adapters_and_readings():
    h=torch.tensor([[[4.,3.]]],requires_grad=True)
    runtime=Runtime();runtime.subscribe([Reading('watch',Site('residualPost',0),'preAction',lambda value,c,s:s.update(value=value.detach().sum()))])
    result=runtime.apply(h,Site('residualPost',0),0,[SubspaceAblator.single([0],[1.,0.],.5),VectorInjector.single(0,[0.,1.],2.)])
    result.sum().backward()
    assert torch.equal(result,torch.tensor([[[2.,5.]]]))
    assert torch.equal(h.grad,torch.tensor([[[.5,1.]]]))


class Block(nn.Module):
    def forward(self, hidden_states): return (hidden_states * 2, 'preserved')


class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.model=nn.Module();self.model.layers=nn.ModuleList([Block(),Block()])
    def forward(self,h):
        for block in self.model.layers: h=block(hidden_states=h)[0]
        return h


def test_context_state_seats_nested_sessions_and_failure_cleanup():
    model=Model();hooked=HookedModel(model);events=[]
    def read(h,c,state):
        state['count']=state.get('count',0)+1
        events.append((dict(c.identity)['agent'],state['count'],c.offset,c.site.kind))
    reading=Reading('shared',Site('residualPre',1),'preAction',read)
    with hooked.session([]):
        with hooked.readings([reading],identity={'agent':'first'},prompt_token_count=3) as outer:
            model(torch.ones(1,2,2))
            with hooked.session([]),hooked.readings([reading],identity={'agent':'second'},prompt_token_count=1) as inner:
                model(torch.ones(1,1,2))
            assert inner.closed and not inner.states
            model(torch.ones(1,1,2))
        assert outer.closed and not outer.states
    assert events==[('first',1,0,'residualPre'),('second',1,0,'residualPre'),('first',2,0,'residualPre')]
    assert not hooked._pre_handles
    def fail(*args): raise RuntimeError('provider failed')
    with pytest.raises(RuntimeError,match='provider failed'):
        with hooked.session([]),hooked.readings([Reading('bad',Site('residualPre',0),'postAction',fail)]) as failed:
            model(torch.ones(1,1,2))
    assert failed.closed and not failed.states and not hooked._pre_handles
    assert torch.equal(model(torch.ones(1,1,2)),torch.full((1,1,2),4.))


def test_abandoned_stream_cannot_observe_or_restore_state_into_new_owner():
    model=Model();hooked=HookedModel(model);events=[]
    old=hooked.arm([],abandonable=True)
    scope=hooked.readings([Reading('reader',Site('residualPre',0),'preAction',lambda h,c,s:events.append('old'))])
    subscription=scope.__enter__()
    model(torch.ones(1,1,2))
    new=hooked.arm([],abandonable=True)
    assert subscription.closed
    with hooked.readings([Reading('reader',Site('residualPre',0),'preAction',lambda h,c,s:events.append('new'))]):
        model(torch.ones(1,1,2))
        scope.__exit__(None,None,None)
        assert not hooked.disarm(old)
        model(torch.ones(1,1,2))
    hooked.disarm(new)
    assert events==['old','new','new'] and not hooked._pre_handles


def test_invalid_sites_and_uninterpretable_batch_are_explicit():
    model=Model();hooked=HookedModel(model)
    with pytest.raises(ValueError,match='supported'): Site('logitsPreSelection',0)
    with pytest.raises(ValueError,match='outside'):
        with hooked.readings([Reading('x',Site('residualPre',4),'preAction',lambda *args:None)]): pass
    runtime=Runtime();r=Reading('x',Site('residualPost',0),'preAction',lambda *args:None)
    with pytest.raises(ValueError,match='unique'):runtime.subscribe([r,r])
    runtime.subscribe([r])
    with pytest.raises(ValueError,match='one unpadded'):runtime.apply(torch.ones(2,1,3),r.site,0)


def test_provider_state_can_span_named_sites_but_not_subscriptions():
    events=[]
    def read(h,c,state):
        state['n']=state.get('n',0)+1;events.append(state['n'])
    readings=[Reading('input',Site('residualPre',0),'preAction',read,provider_id='shared'),
              Reading('output',Site('residualPost',0),'postAction',read,provider_id='shared')]
    runtime=Runtime();first=runtime.subscribe(readings)
    h=torch.ones(1,1,2)
    runtime.apply(h,readings[0].site,0);runtime.apply(h,readings[1].site,0)
    runtime.unsubscribe(first);runtime.subscribe(readings)
    runtime.apply(h,readings[0].site,0)
    assert events==[1,2,1] and not first.states


def test_partial_hook_installation_failure_preserves_existing_subscriptions(monkeypatch):
    model=Model();hooked=HookedModel(model)
    reading=Reading('first',Site('residualPre',0),'preAction',lambda *args:None)
    with hooked.session([]),hooked.readings([reading]):
        def broken(*args,**kwargs): raise RuntimeError('registration failed')
        monkeypatch.setattr(model.model.layers[1],'register_forward_pre_hook',broken)
        with pytest.raises(RuntimeError,match='registration failed'):
            with hooked.readings([reading,Reading('second',Site('residualPre',1),'preAction',lambda *args:None)]):pass
        assert hooked._pre_handles[0][1]==1
        assert torch.equal(model(torch.ones(1,1,2)),torch.full((1,1,2),4.))
    assert not hooked._pre_handles


def test_probe_adapter_checks_the_actual_armed_block_map(tmp_path):
    from test_probe_measurements import prepared
    from steerlab_server.experiment import probe_observation, probe_measurements, prompt_render
    from steerlab_server.experiment.probe_artifacts import ProbeError
    config,_,model=prepared(tmp_path)
    observer=probe_observation.create(config,probe_measurements.load(config,tmp_path),condition='baseline',agent='seat',rendering='rawCompletion')
    model.hooked.layers=nn.ModuleList([Block()])
    with pytest.raises(ProbeError,match='armed decoder blocks'):
        with model.hooked.session([]),observer.observe_session(model,prompt_render.RenderedPrompt('',[1],1)):pass
    assert observer.failures and not model.hooked.runtime.subscriptions


def test_same_phase_readings_use_declaration_order_and_cannot_return_actions():
    runtime=Runtime();events=[];site=Site('residualPost',0)
    runtime.subscribe([Reading('second',site,'preAction',lambda *args:events.append('second')),
                       Reading('first',site,'preAction',lambda *args:events.append('first'))])
    h=torch.ones(1,1,2);assert runtime.apply(h,site,0) is h
    assert events==['second','first']
    runtime.subscribe([Reading('wrong',site,'postAction',lambda h,*args:h+1)])
    with pytest.raises(ValueError,match='must return None'): runtime.apply(h,site,0)
