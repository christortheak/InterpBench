"""Placement reaches real child plans and scheduler bundles without changing inputs."""
from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
from test_jlens_rounds import round_job
from test_jlens_fit import fitting
from test_scientific_execution import setup
from steerlab_server.api import jlens_rounds, scientific_execution as engine, science_placement
from steerlab_server.api.executors import SlurmExecutor
from steerlab_server.experiment import diagnostic_archives as archives, runtime_hardware


def slurm(profile, monkeypatch):
    monkeypatch.setenv('STEERLAB_SERVER_ROLE', 'controller')
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'slurm')
    monkeypatch.setenv('STEERLAB_SLURM_GRES', 'gpu:A100:1')
    monkeypatch.setenv('STEERLAB_SLURM_GPU_TYPES', 'A100,H100')
    monkeypatch.setenv('STEERLAB_SLURM_GPU_VRAM', 'A100:40,H100:80')
    return replace(profile, executor='slurm')


@pytest.mark.parametrize('round_job', [2], indirect=True)
def test_mixed_topup_reaches_scheduler_and_keeps_attempted_placement(round_job, monkeypatch):
    parent, jobs, profile = round_job
    profile = slurm(profile, monkeypatch)
    bundles = []
    def submit(self, bundle, **kwargs):
        bundles.append(bundle)
        return str(len(bundles))
    monkeypatch.setattr(SlurmExecutor, 'submit', submit)
    default = jlens_rounds.action(parent.id, 'plan', None, False, jobs, profile)
    review = jlens_rounds.action(parent.id, 'plan', None, False, jobs, profile,
                                gpu_type='A100', shard_gpu_types={'1':'H100'})
    assert review['planSHA256'] != default['planSHA256']
    assert [row['gpuType'] for row in review['shards']] == ['A100','H100']
    for index in ('0','1'):
        assert review['childPlans'][index]['inputSHA256'] == default['childPlans'][index]['inputSHA256']
    with pytest.raises(archives.Refusal, match='changed'):
        jlens_rounds.action(parent.id,'submit',review['planSHA256'],True,jobs,profile,gpu_type='A100')
    assert bundles == []
    result = jlens_rounds.action(parent.id,'submit',review['planSHA256'],True,jobs,profile,
                                gpu_type='A100',shard_gpu_types={'1':'H100'})
    assert len(result['submissions']) == 2
    assert [b.resources.normalized_gres() for b in bundles] == ['gpu:A100:1','gpu:H100:1']
    for b in bundles:
        assert '--gres=' + b.resources.normalized_gres() in open(b.script_path).read()
    status = jlens_rounds.action(parent.id,'status',None,False,jobs,profile)
    assert [row['gpuType'] for row in status['shards']] == ['A100','H100']
    with pytest.raises(archives.Refusal, match='already been attempted'):
        jlens_rounds.action(parent.id,'plan',None,False,jobs,profile,shard_gpu_types={'0':'H100'})
    # A new top-up default never relabels existing allocations.
    next_review = jlens_rounds.action(parent.id,'plan',None,False,jobs,profile,gpu_type='H100')
    assert next_review['submitIndices'] == []
    assert [row['gpuType'] for row in next_review['shards']] == ['A100','H100']


def test_uncertain_placement_survives_without_retry(round_job, monkeypatch):
    parent,jobs,profile=round_job;profile=slurm(profile,monkeypatch)
    def lost(*a, **kw): raise RuntimeError('lost submission reply')
    monkeypatch.setattr(engine,'submit',lost)
    review=jlens_rounds.action(parent.id,'plan',None,False,jobs,profile,gpu_type='H100')
    with pytest.raises(RuntimeError):
        jlens_rounds.action(parent.id,'submit',review['planSHA256'],True,jobs,profile,gpu_type='H100')
    status=jlens_rounds.action(parent.id,'status',None,False,jobs,profile)
    assert status['shards'][0]['status']=='uncertain' and status['shards'][0]['gpuType']=='H100'
    assert status['submitIndices']==[]
    with pytest.raises(archives.Refusal,match='already been attempted'):
        jlens_rounds.action(parent.id,'plan',None,False,jobs,profile,shard_gpu_types={'0':'A100'})


@pytest.mark.parametrize('overrides', [[], {'01':'H100'}, {'2':'H100'}, {'0':'unknown'}, {'1':'H100'}])
def test_override_admission_before_any_attempt(round_job,monkeypatch,overrides):
    parent,jobs,profile=round_job;profile=slurm(profile,monkeypatch)
    with pytest.raises(ValueError):
        jlens_rounds.action(parent.id,'plan',None,False,jobs,profile,shard_gpu_types=overrides)
    assert not jlens_rounds.state_file(parent.id,profile).exists()


def test_selected_capacity_and_pilot_are_honestly_described(setup,monkeypatch):
    _,request,profile=setup;profile=slurm(profile,monkeypatch)
    result=engine.plan(request,profile,gpu_type='H100')
    assert result['gpuReview']['declaredVRAMGB']==80
    assert result['gpuReview']['memoryFit']=='notChecked'
    assert 'gpuReview' not in engine.plan(request,profile)
    caps=science_placement.capabilities(profile)
    assert caps['gpuTypes']==['A100','H100'] and caps['defaultGPUType']=='A100'
    from steerlab_server.api.executors import SlurmResources
    review=science_placement.review(SlurmResources.from_env(),{'pilotMeasurement':{'hardware':{'deviceName':'fixture GPU'}}})
    assert review['pilotHardware']['deviceName']=='fixture GPU' and 'new benchmark' in review['throughputScope']


def test_hardware_is_observation_not_admission():
    props=SimpleNamespace(name='fixture GPU',total_memory=123,major=9,minor=0)
    torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,get_device_properties=lambda _:props),version=SimpleNamespace(cuda='fixture'))
    observed=runtime_hardware.describe(torch,'cuda:0')
    assert observed['deviceName']=='fixture GPU' and observed['computeCapability']==[9,0]
    def broken(_):raise RuntimeError('unavailable')
    torch.cuda.get_device_properties=broken
    assert runtime_hardware.describe(torch,'cuda')['deviceName'] is None
    assert runtime_hardware.describe(torch,'cpu')['computeCapability'] is None


def test_failed_execution_record_keeps_hardware(setup,monkeypatch):
    root,request,profile=setup
    reviewed=engine.plan(request,profile)
    packet=root/'packet.json';packet.write_text(json.dumps(reviewed))
    from steerlab_server.experiment import battery_run
    monkeypatch.setattr(runtime_hardware,'observe',lambda *_:{'deviceName':'observed fixture'})
    def fail(*a,**k):raise RuntimeError('model failed')
    monkeypatch.setattr(battery_run,'execute',fail)
    record=root/'record.json'
    assert engine.execute_packet(str(packet),'test',str(record),reviewed['planSHA256'])==70
    assert json.loads(record.read_text())['result']['runtimeHardware']['deviceName']=='observed fixture'


def test_http_round_gpu_map_is_forwarded(round_job,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.diagnostic_transport_routes import build_router
    parent,jobs,profile=round_job;slurm(profile,monkeypatch)
    app=FastAPI();app.include_router(build_router(SimpleNamespace(jobs=jobs)))
    with TestClient(app) as client:
        base='/api/science/fitting-round/'+parent.id
        response=client.post(base+'/plan',json={'shardGPUTypes':{'0':'H100'}})
        assert response.status_code==200, response.text
        assert response.json()['childPlans']['0']['resources']['gres']=='gpu:H100:1'
        assert client.post(base+'/merge-plan',json={'gpuType':'H100'}).status_code==409
        assert client.post(base+'/plan',json={'gres':'gpu:H100:1'}).status_code==409


def test_hardware_observation_uses_engine_device_resolution(monkeypatch):
    # A missing device follows the engine environment, including MPS; it must
    # not be mislabeled CPU merely because CUDA is absent on a Mac.
    monkeypatch.setenv('STEERLAB_DEVICE', 'mps')
    assert runtime_hardware.observe()['requestedDevice'] == 'mps'
    assert runtime_hardware.observe('auto')['requestedDevice'] == 'mps'
    assert runtime_hardware.observe('cpu')['requestedDevice'] == 'cpu'
