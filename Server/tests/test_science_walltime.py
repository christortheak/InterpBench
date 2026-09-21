"""A per-request walltime rides beside the request like the GPU type, and short
operations get a backfill-sized default from their own reviewed workload."""
from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_jlens_rounds import round_job
from test_jlens_fit import fitting
from test_jlens_multi_candidate_assessment import library, plural
from test_scientific_execution import setup
from steerlab_server.api import jlens_rounds, scientific_execution as engine, science_walltime
from steerlab_server.api.executors import SlurmExecutor, SlurmResources
from steerlab_server.api.jobs import JobManager, DurableJobStore
from steerlab_server.api.scientific_execution_routes import build_scientific_execution_router
from steerlab_server.experiment import diagnostic_archives as archives


def slurm(profile, monkeypatch, walltime='24:00:00'):
    monkeypatch.setenv('STEERLAB_SERVER_ROLE', 'controller')
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'slurm')
    monkeypatch.setenv('STEERLAB_SLURM_GRES', 'gpu:A100:1')
    monkeypatch.setenv('STEERLAB_SLURM_GPU_TYPES', 'A100,H100')
    monkeypatch.setenv('STEERLAB_SLURM_WALLTIME', walltime)
    return replace(profile, executor='slurm')


def test_requested_walltime_is_rendered_and_bound_into_the_plan(setup, monkeypatch):
    root, request, profile = setup
    profile = slurm(profile, monkeypatch)
    default = engine.plan(request, profile)
    assert default['walltimeBasis'] == 'siteDefault' and default['resources']['walltime'] == '24:00:00'
    assert '--time=24:00:00' in default['schedulerPreview'] and 'requestedWalltime' not in default
    assert engine.plan(request, profile, walltime=None) == default
    short = engine.plan(request, profile, walltime='02:30:00')
    assert short['requestedWalltime'] == '02:30:00' and short['walltimeBasis'] == 'requested'
    assert short['resources']['walltime'] == '02:30:00' and '--time=02:30:00' in short['schedulerPreview']
    assert short['walltimeReview']['cap'] == '24:00:00' and short['walltimeReview']['siteDefault'] == '24:00:00'
    assert short['inputSHA256'] == default['inputSHA256'] and short['request'] == default['request']
    days = engine.plan(request, profile, walltime='0-02:30:00')
    assert days['resources']['walltime'] == '02:30:00' and days['planSHA256'] == short['planSHA256']
    other = engine.plan(request, profile, walltime='01:00:00')
    assert len({default['planSHA256'], short['planSHA256'], other['planSHA256']}) == 3
    jobs = JobManager(store=DurableJobStore(str(root / 'jobs.sqlite')), sweep_orphans=False)
    with pytest.raises(engine.ScientificRefusal, match='changed after review'):
        engine.submit(request, short['planSHA256'], profile=profile, jobs=jobs, walltime='01:00:00')
    with pytest.raises(engine.ScientificRefusal, match='changed after review'):
        engine.submit(request, short['planSHA256'], profile=profile, jobs=jobs)  # reviewed with a walltime, submitted without it
    assert jobs.list() == []


def test_submitted_bundle_and_job_record_carry_the_walltime(setup, monkeypatch):
    root, request, profile = setup
    profile = slurm(profile, monkeypatch)
    bundles = []
    def submit(self, bundle, **kwargs):
        bundles.append(bundle)
        return '4242'
    monkeypatch.setattr(SlurmExecutor, 'submit', submit)
    reviewed = engine.plan(request, profile, walltime='02:30:00')
    jobs = JobManager(store=DurableJobStore(str(root / 'jobs.sqlite')), sweep_orphans=False)
    result = engine.submit(request, reviewed['planSHA256'], profile=profile, jobs=jobs, walltime='02:30:00')
    assert result['status'] == 'submitted' and result['scientificPlan']['requestedWalltime'] == '02:30:00'
    assert bundles[0].resources.walltime == '02:30:00' and '--time=02:30:00' in open(bundles[0].script_path).read()
    job = jobs.get(result['jobId'])
    assert job.result['scientificPlan']['resources']['walltime'] == '02:30:00'
    assert job.result['scientificPlan']['walltimeBasis'] == 'requested'


@pytest.mark.parametrize('value', ['25:00:00', '1-00:00:01', '2-00:00:00'])
def test_walltime_above_the_site_cap_refuses(setup, monkeypatch, value):
    _, request, profile = setup
    profile = slurm(profile, monkeypatch)
    with pytest.raises(engine.ScientificRefusal, match='exceeds the site cap of 24:00:00') as refusal:
        engine.plan(request, profile, walltime=value)
    assert refusal.value.code == 'scientificExecutionRefused' and 'at most the site cap' in refusal.value.repair_action
    assert engine.plan(request, profile, walltime='24:00:00')['walltimeBasis'] == 'requested'
    assert engine.plan(request, profile, walltime='1-00:00:00')['resources']['walltime'] == '24:00:00'


@pytest.mark.parametrize('value', ['', '   ', '90', '30:00', '2h', '1:2:3', '01:60:00', '01:00', 'PT1H', '01:00:00 ', 150, None])
def test_malformed_walltime_refuses_before_any_plan(setup, monkeypatch, value):
    _, request, profile = setup
    profile = slurm(profile, monkeypatch)
    if value is None:
        assert engine.plan(request, profile, walltime=None)['walltimeBasis'] == 'siteDefault'
        return
    if value == '01:00:00 ':
        assert engine.plan(request, profile, walltime=value)['requestedWalltime'] == '01:00:00'  # surrounding whitespace only
        return
    with pytest.raises(engine.ScientificRefusal, match='not HH:MM:SS or D-HH:MM:SS') as refusal:
        engine.plan(request, profile, walltime=value)
    assert 'HH:MM:SS' in refusal.value.repair_action


def test_local_executor_records_a_walltime_and_enforces_nothing(setup):
    _, request, profile = setup
    default = engine.plan(request, profile)
    assert 'walltimeBasis' not in default and default['resources'] == {'executor': 'local'}
    recorded = engine.plan(request, profile, walltime='5-00:00:00')  # no cap on a local executor
    assert recorded['requestedWalltime'] == '120:00:00' and recorded['walltimeBasis'] == 'requested'
    assert recorded['resources'] == {'executor': 'local'} and 'enforces no walltime' in recorded['walltimeReview']['summary']
    with pytest.raises(engine.ScientificRefusal, match='not HH:MM:SS'):
        engine.plan(request, profile, walltime='soon')


def test_assessment_default_is_sized_from_its_own_workload(library, monkeypatch):
    root, base, ids, corpora = library
    monkeypatch.setenv('STEERLAB_ROOT', str(root))
    monkeypatch.setenv('STEERLAB_METADATA_ROOT', str(root / '.steerlab'))
    from steerlab_server.api.profile import ServerProfile
    profile = slurm(ServerProfile.from_env(), monkeypatch)
    request = {'operation': 'jlens-fit-assess', 'parameters': {'config': plural(base, ids[1:], corpora).to_dict()}}
    plan = engine.plan(request, profile)
    review = plan['walltimeReview']
    assert plan['walltimeBasis'] == 'estimated' and review['basis'] == 'estimated'
    estimate = review['estimate']
    assert estimate['rows'] == sum(c['rows'] for c in plan['operationReview']['corpora'])
    assert estimate['candidates'] == 2 and estimate['sourceLayers'] == len(plan['operationReview']['sourceLayers'])
    assert estimate['units'] == estimate['rows'] * estimate['sourceLayers'] * 2 and estimate['readoutMultiplier'] == 1
    seconds = science_walltime.parse(plan['resources']['walltime'])
    assert estimate['seconds'] < seconds < science_walltime.parse('24:00:00')
    assert seconds >= 3 * estimate['seconds'] + 30 * 60 and seconds % (15 * 60) == 0 and seconds >= 3600
    assert '--time=' + plan['resources']['walltime'] in plan['schedulerPreview']
    # The estimate is an allowance with a named calibration and limitation, never a promise.
    assert 'not a measurement' in estimate['limitation'] and 'calibration' in estimate
    # A request overrides the estimate; the two plans differ only in execution shape.
    explicit = engine.plan(request, profile, walltime='00:45:00')
    assert explicit['walltimeBasis'] == 'requested' and explicit['inputSHA256'] == plan['inputSHA256']
    assert explicit['planSHA256'] != plan['planSHA256']


def test_default_rule_margins_floor_rounding_and_cap():
    resources = SlurmResources(walltime='24:00:00')
    cap = science_walltime.cap_seconds(resources)
    assert science_walltime.default_seconds(0, cap) == 3600                      # floor
    assert science_walltime.default_seconds(600, cap) == 3600                    # 3×10 min + 30 min = 60 min
    assert science_walltime.default_seconds(601, cap) == 3600 + 15 * 60          # rounds up to the next 15 min
    assert science_walltime.default_seconds(4032, cap) == 4 * 3600               # 3×67.2 min + 30 = 231.6 → 240 min
    assert science_walltime.default_seconds(10 ** 6, cap) == cap                 # capped
    assessment = {'request': {'operation': 'jlens-fit-assess', 'parameters': {'config': {'readoutDtype': 'float32'}}},
                  'operationReview': {'rows': 16, 'sourceLayers': list(range(63))}}
    estimate = science_walltime.estimate(assessment)
    assert estimate['units'] == 16 * 63 * 2 and estimate['readoutMultiplier'] == 2
    resources = SlurmResources(walltime='24:00:00')
    review = science_walltime.review(assessment, resources)
    assert review['basis'] == 'estimated' and resources.walltime == review['walltime'] == '02:15:00'
    huge = {**assessment, 'operationReview': {'rows': 100000, 'sourceLayers': list(range(63))}}
    resources = SlurmResources(walltime='24:00:00')
    assert science_walltime.review(huge, resources)['basis'] == 'siteDefault' and resources.walltime == '24:00:00'
    for plan in ({'request': {'operation': 'jlens-fit'}, 'fittingReview': {'pilotMeasurement': {'extrapolatedHoursAtRowCap': 0.5}}},
                 {'request': {'operation': 'jlens-fit-benchmark'}, 'operationReview': {'rows': 4, 'cases': [{}]}},
                 {'request': {'operation': 'battery'}},
                 {'request': {'operation': 'jlens-fit-assess'}, 'operationReview': {'rows': 'many', 'sourceLayers': [1]}}):
        assert science_walltime.estimate(plan) is None
        resources = SlurmResources(walltime='12:00:00')
        assert science_walltime.review(plan, resources)['basis'] == 'siteDefault' and resources.walltime == '12:00:00'
    # A site that declares its walltime in the executor's shorter vocabulary still yields a cap.
    assert science_walltime.cap_seconds(SlurmResources(walltime='90')) == 5400
    assert science_walltime.render(90061) == '25:01:01' and science_walltime.parse('1-01:01:01') == 90061


@pytest.mark.parametrize('round_job', [2], indirect=True)
def test_fitting_shards_keep_the_site_default_and_take_per_shard_walltimes(round_job, monkeypatch):
    parent, jobs, profile = round_job
    profile = slurm(profile, monkeypatch)
    bundles = []
    def submit(self, bundle, **kwargs):
        bundles.append(bundle)
        return str(len(bundles))
    monkeypatch.setattr(SlurmExecutor, 'submit', submit)
    default = jlens_rounds.action(parent.id, 'plan', None, False, jobs, profile)
    for child in default['childPlans'].values():
        assert child['walltimeBasis'] == 'siteDefault' and child['resources']['walltime'] == '24:00:00'
    assert [(row['walltime'], row['walltimeBasis']) for row in default['shards']] == [('24:00:00', 'siteDefault')] * 2
    assert 'walltimeRequest' not in default
    with pytest.raises(engine.ScientificRefusal, match='exceeds the site cap'):
        jlens_rounds.action(parent.id, 'plan', None, False, jobs, profile, walltime='30:00:00')
    with pytest.raises(archives.Refusal, match='canonical zero-based shard index'):
        jlens_rounds.action(parent.id, 'plan', None, False, jobs, profile, shard_walltimes={'7': '01:00:00'})
    with pytest.raises(archives.Refusal, match='not status or CPU merge'):
        jlens_rounds.action(parent.id, 'status', None, False, jobs, profile, walltime='01:00:00')
    review = jlens_rounds.action(parent.id, 'plan', None, False, jobs, profile, walltime='08:00:00', shard_walltimes={'1': '02:00:00'})
    assert review['planSHA256'] != default['planSHA256']
    assert review['walltimeRequest'] == {'walltime': '08:00:00', 'shardWalltimes': {'1': '02:00:00'}}
    assert [row['walltime'] for row in review['shards']] == ['08:00:00', '02:00:00']
    assert [row['walltimeBasis'] for row in review['shards']] == ['requested', 'requested']
    for index in ('0', '1'):
        assert review['childPlans'][index]['inputSHA256'] == default['childPlans'][index]['inputSHA256']
    with pytest.raises(archives.Refusal, match='changed'):
        jlens_rounds.action(parent.id, 'submit', review['planSHA256'], True, jobs, profile, walltime='08:00:00')
    assert bundles == []
    result = jlens_rounds.action(parent.id, 'submit', review['planSHA256'], True, jobs, profile,
                                 walltime='08:00:00', shard_walltimes={'1': '02:00:00'})
    assert len(result['submissions']) == 2
    assert [b.resources.walltime for b in bundles] == ['08:00:00', '02:00:00']
    for b in bundles:
        assert '--time=' + b.resources.walltime in open(b.script_path).read()
    state = json.loads(jlens_rounds.state_file(parent.id, profile).read_text())
    assert state['walltimes'] == {'0': '08:00:00', '1': '02:00:00'}
    status = jlens_rounds.action(parent.id, 'status', None, False, jobs, profile)
    assert [row['walltime'] for row in status['shards']] == ['08:00:00', '02:00:00']
    with pytest.raises(archives.Refusal, match='already been attempted'):
        jlens_rounds.action(parent.id, 'plan', None, False, jobs, profile, shard_walltimes={'0': '01:00:00'})


def test_http_wrapper_accepts_and_refuses_the_walltime_like_the_gpu_type(setup, monkeypatch):
    root, request, profile = setup
    profile = slurm(profile, monkeypatch)
    jobs = JobManager(store=DurableJobStore(str(root / 'jobs.sqlite')), sweep_orphans=False)
    app = FastAPI(); app.include_router(build_scientific_execution_router(SimpleNamespace(jobs=jobs, registry=None)))
    client = TestClient(app)
    bare = client.post('/api/science/plan', json=request).json()
    assert bare == engine.plan(request, profile) and bare['walltimeBasis'] == 'siteDefault'
    wrapped = client.post('/api/science/plan', json={'request': request, 'walltime': '02:00:00'}).json()
    assert wrapped['requestedWalltime'] == '02:00:00' and wrapped == engine.plan(request, profile, walltime='02:00:00')
    both = client.post('/api/science/plan', json={'request': request, 'gpuType': 'H100', 'walltime': '02:00:00'}).json()
    assert both['requestedGPUType'] == 'H100' and both['resources']['walltime'] == '02:00:00'
    refused = client.post('/api/science/plan', json={'request': request, 'walltime': '36:00:00'})
    assert refused.status_code == 409 and 'exceeds the site cap' in refused.json()['detail']['reason']
    assert 'at most the site cap' in refused.json()['detail']['repairAction']
    malformed = client.post('/api/science/plan', json={'request': request, 'walltime': 'two hours'})
    assert malformed.status_code == 409 and 'HH:MM:SS' in malformed.json()['detail']['reason']
    stale = client.post('/api/science/submit', json={'request': request, 'planSHA256': wrapped['planSHA256']})
    assert stale.status_code == 409 and jobs.list() == []
    unknown = client.post('/api/science/submit', json={'request': request, 'planSHA256': wrapped['planSHA256'], 'time': '02:00:00'})
    assert unknown.status_code == 409 and 'walltime' in unknown.json()['detail']['reason']


def test_http_round_walltime_is_forwarded_per_shard(round_job, monkeypatch):
    from steerlab_server.api.diagnostic_transport_routes import build_router
    parent, jobs, profile = round_job
    slurm(profile, monkeypatch)
    app = FastAPI(); app.include_router(build_router(SimpleNamespace(jobs=jobs)))
    with TestClient(app) as client:
        base = '/api/science/fitting-round/' + parent.id
        response = client.post(base + '/plan', json={'walltime': '06:00:00', 'shardWalltimes': {'0': '03:00:00'}})
        assert response.status_code == 200, response.text
        assert response.json()['childPlans']['0']['resources']['walltime'] == '03:00:00'
        assert response.json()['shards'][0]['walltime'] == '03:00:00'
        assert client.post(base + '/merge-plan', json={'walltime': '06:00:00'}).status_code == 409
        assert client.post(base + '/plan', json={'walltime': '48:00:00'}).status_code == 409


def test_portable_client_sends_the_walltime_beside_the_request(tmp_path, monkeypatch, capsys):
    import httpx
    from steerlab_server import client_cli
    from steerlab_server.client import runner
    bodies = []
    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={'jobId': 'example-job', 'status': 'submitted', 'planSHA256': 'c' * 64})
    original = runner.RunnerClient
    monkeypatch.setattr(runner, 'RunnerClient', lambda **kwargs: original(
        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    request = tmp_path / 'request.json'
    document = {'operation': 'stability', 'parameters': {'experiment': 'example', 'concept': 'signal'}}
    request.write_text(json.dumps(document))
    common = ['--runner', 'https://runner.example.invalid', '--json']
    assert client_cli.main(['runner', 'science-plan', str(request), *common]) == 0
    assert client_cli.main(['runner', 'science-plan', str(request), '--walltime', '02:00:00', *common]) == 0
    assert client_cli.main(['runner', 'science-plan', str(request), '--gpu-type', 'H100', '--walltime', '02:00:00', *common]) == 0
    assert client_cli.main(['runner', 'science-submit', str(request), '--plan-sha256', 'a' * 64, '--walltime', '02:00:00', *common]) == 0
    capsys.readouterr()
    assert bodies == [document,
                      {'request': document, 'walltime': '02:00:00'},
                      {'request': document, 'gpuType': 'H100', 'walltime': '02:00:00'},
                      {'request': document, 'planSHA256': 'a' * 64, 'walltime': '02:00:00'}]
