import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from steerlab_server.api import scientific_execution as owner
from steerlab_server.api.jobs import JobManager, DurableJobStore
from steerlab_server.api.profile import ServerProfile
from steerlab_server.api.scientific_execution_routes import build_scientific_execution_router


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from test_battery_run import _lines
    path = tmp_path / 'prompts/batteries/example.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(_lines())
    monkeypatch.setenv('STEERLAB_ROOT', str(tmp_path))
    monkeypatch.setenv('STEERLAB_METADATA_ROOT', str(tmp_path / '.steerlab'))
    monkeypatch.setenv('STEERLAB_SERVER_ROLE', 'workstation')
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'local')
    request = {'operation': 'battery', 'parameters': {'batteryFile': str(path.relative_to(tmp_path)),
               'agents': ['baseline'], 'modelID': 'example/model', 'revision': 'a' * 40}}
    return tmp_path, request, ServerProfile.from_env()


def test_plan_reads_real_owner_and_binds_bytes_parameters_and_root(setup):
    root, request, profile = setup
    before = sorted(str(p) for p in root.rglob('*'))
    plan = owner.plan(request, profile)
    assert plan['resumable'] is False and plan['models'][0]['revision'] == 'a' * 40
    assert sorted(str(p) for p in root.rglob('*')) == before
    changed = json.loads(json.dumps(request)); changed['parameters']['dtype'] = 'float32'
    assert owner.plan(changed, profile)['planSHA256'] != plan['planSHA256']
    path = root / request['parameters']['batteryFile']; path.write_text(path.read_text().replace('Plain question', 'Revised question'))
    assert owner.plan(request, profile)['inputSHA256'] != plan['inputSHA256']
    with pytest.raises(owner.ScientificRefusal, match='changed'):
        owner.submit(request, plan['planSHA256'], profile=profile, jobs=None)
    assert not (root / 'runs').exists()


def test_request_admission_precedes_publication(setup):
    root, request, profile = setup
    for key, value in [('batteryFile', '../outside'), ('agents', []), ('revision', 'main')]:
        invalid = json.loads(json.dumps(request)); invalid['parameters'][key] = value
        with pytest.raises(ValueError):
            owner.plan(invalid, profile)
    with pytest.raises(ValueError):
        owner.plan({'operation': [], 'parameters': {}}, profile)
    outside = root.parent / (root.name + '-outside.jsonl'); outside.write_text('private')
    (root / 'link').symlink_to(outside)
    with pytest.raises(owner.ScientificRefusal):
        owner.workspace_file(str(root), 'link')


def test_queued_input_drift_refuses_before_execution_and_writes_record(setup, monkeypatch):
    from steerlab_server.experiment import battery_run
    root, request, profile = setup
    plan = owner.plan(request, profile)
    packet = root / 'packet.json'; packet.write_text(json.dumps(plan))
    path = root / request['parameters']['batteryFile']; path.write_text(path.read_text().replace('Plain question', 'Revised question'))
    monkeypatch.setattr(battery_run, 'execute', lambda *a, **k: pytest.fail('must not execute changed input'))
    record = root / 'record.json'
    assert owner.execute_packet(packet, 'example-job', record) == 70
    document = json.loads(record.read_text())
    assert document['status'] == 'failed' and 'changed while queued' in document['error']
    assert document['result']['scientificPlan'] == plan


def test_child_success_keeps_battery_output_type(setup, monkeypatch):
    from steerlab_server.experiment import battery_run
    root, request, profile = setup
    packet = root / 'packet.json'; packet.write_text(json.dumps(owner.plan(request, profile)))
    monkeypatch.setattr(battery_run, 'execute', lambda *a, **k: {'runDirectory': str(root / 'runs/result'), 'agents': []})
    record = root / 'record.json'
    assert owner.execute_packet(packet, 'example-job', record) == 0
    result = json.loads(record.read_text())['result']
    assert result['runDirectory'].endswith('runs/result') and result['resumable'] is False


def test_slurm_intention_is_durable_before_submit_and_uncertainty_is_visible(setup, monkeypatch):
    from dataclasses import replace
    root, request, profile = setup
    monkeypatch.setenv('STEERLAB_SERVER_ROLE', 'controller')
    monkeypatch.setenv('STEERLAB_AUTO_RESUBMIT', '1')
    profile = replace(profile, executor='slurm')
    jobs = JobManager(store=DurableJobStore(str(root / 'jobs.sqlite')), sweep_orphans=False)
    calls = []
    def uncertain(self, bundle, **kwargs):
        calls.append(kwargs)
        assert len(jobs.list()) == 1 and jobs.list()[0].status == 'submitting'
        assert '--export=NONE' in Path(bundle.script_path).read_text()
        raise TimeoutError('scheduler reply lost')
    monkeypatch.setattr(owner.SlurmExecutor, 'submit', uncertain)
    submitted = owner.submit(request, owner.plan(request, profile)['planSHA256'], profile=profile, jobs=jobs)
    job = jobs.get(submitted['jobId'])
    assert len(calls) == 1 and job.status == 'parked'
    assert job.result['schedulerSubmissionName'] == calls[0]['job_name']
    reloaded = DurableJobStore(str(root / 'jobs.sqlite')).load_all()[job.id]
    assert reloaded.result['parked'] is True and reloaded.requested_resources['autoResubmit'] is False
    assert reloaded.requested_resources['auto_resubmit'] is False and reloaded.finished_at is not None


def test_local_failure_preserves_job_context(setup, monkeypatch):
    root, request, profile = setup
    jobs = JobManager(store=DurableJobStore(str(root / 'jobs.sqlite')), sweep_orphans=False)
    def fail(self, command, **kwargs):
        Path(command[-2]).write_text(json.dumps({'result': {'partial': 'retained'}}))
        return SimpleNamespace(returncode=70, stderr='child failed')
    monkeypatch.setattr(owner.LocalExecutor, 'run', fail)
    submitted = owner.submit(request, owner.plan(request, profile)['planSHA256'], profile=profile, jobs=jobs)
    deadline = time.monotonic() + 5
    while jobs.get(submitted['jobId']).status not in {'failed', 'succeeded'} and time.monotonic() < deadline:
        time.sleep(.01)
    job = jobs.get(submitted['jobId'])
    assert job.status == 'failed' and job.result['partial'] == 'retained'
    assert job.result['scientificPlan']['request']['operation'] == 'battery'


def test_http_plan_and_recovery_use_existing_owner_gates(setup):
    root, request, profile = setup
    jobs = JobManager(store=DurableJobStore(str(root / 'jobs.sqlite')), sweep_orphans=False)
    app = FastAPI(); app.include_router(build_scientific_execution_router(SimpleNamespace(jobs=jobs, registry=None)))
    client = TestClient(app)
    assert client.post('/api/science/plan', json=request).json() == owner.plan(request, profile)
    response = client.post('/api/science/submit', json={'request': request, 'planSHA256': 'wrong'})
    assert response.status_code == 409 and response.json()['detail']['repairAction']
    assert jobs.list() == []
    response = client.post('/api/jobs/missing/recover', json={'confirmOwnerExited': True})
    assert response.status_code == 409


def test_real_local_cancellation_retains_child_progress(setup, monkeypatch):
    import subprocess
    import sys
    root, request, profile = setup
    jobs = JobManager(store=DurableJobStore(str(root / 'jobs.sqlite')), sweep_orphans=False)
    real_run = owner.LocalExecutor.run
    def child(self, command, **kwargs):
        script = 'import json,sys,time; from pathlib import Path; p=Path(sys.argv[1]); temp=p.with_suffix(".tmp"); temp.write_text(json.dumps({"result":{"partial":True,"runDirectory":sys.argv[2]}})); temp.replace(p); time.sleep(30)'
        return real_run(self, [sys.executable, '-c', script, command[-2], str(root / 'runs/partial')], **kwargs)
    monkeypatch.setattr(owner.LocalExecutor, 'run', child)
    submitted = owner.submit(request, owner.plan(request, profile)['planSHA256'], profile=profile, jobs=jobs)
    job = jobs.get(submitted['jobId'])
    record = Path(submitted['recordsDirectory']) / (job.id + '.json')
    deadline = time.monotonic() + 5
    while not record.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert record.exists()
    assert jobs.cancel(job.id)
    deadline = time.monotonic() + 10
    while job.status not in {'cancelled', 'cancelledResumable', 'failed'} and time.monotonic() < deadline:
        time.sleep(.01)
    assert job.status == 'cancelled'
    assert job.result['partial'] is True and job.result['runDirectory'].endswith('/partial')


def test_battery_failure_after_creation_keeps_output_location(setup, monkeypatch):
    from steerlab_server.experiment import battery_run
    root, request, profile = setup
    packet = root / 'packet.json'; packet.write_text(json.dumps(owner.plan(request, profile)))
    def execute(*args, on_run_created, **kwargs):
        output = root / 'runs/partial'
        output.mkdir(parents=True)
        on_run_created(str(output))
        raise RuntimeError('generation interrupted')
    monkeypatch.setattr(battery_run, 'execute', execute)
    record = root / 'record.json'
    assert owner.execute_packet(packet, 'example-job', record) == 70
    document = json.loads(record.read_text())
    assert document['result']['partial'] is True
    assert document['result']['runDirectory'].endswith('runs/partial')


def test_portable_adapter_preserves_endpoint_paths_and_posts_once():
    import httpx
    from steerlab_server.client.runner import RunnerClient, RunnerError
    calls = []
    def respond(request):
        calls.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        return httpx.Response(200, json={'jobId': 'job'})
    http = httpx.Client(transport=httpx.MockTransport(respond))
    client = RunnerClient(base_url='https://runner.example.invalid', http_client=http)
    request = {'operation': 'stability', 'parameters': {'experiment': 'example', 'concept': 'signal'}}
    client.scientific_plan(request)
    client.scientific_submit(request, 'a' * 64)
    client.resubmit_job('job', '03:00:00')
    client.job_recovery('job')
    client.recover_job('job', 'review', 'owner exit verified')
    assert [c[:2] for c in calls] == [('POST', '/api/science/plan'), ('POST', '/api/science/submit'),
        ('POST', '/api/jobs/job/resubmit'), ('GET', '/api/jobs/job/recovery'), ('POST', '/api/jobs/job/recover')]
    assert calls[1][2] == {'request': request, 'planSHA256': 'a' * 64}
    assert calls[-1][2]['confirmOwnerExited'] is True
    def timeout(request):
        calls.append(request)
        raise httpx.ReadTimeout('no response', request=request)
    uncertain = RunnerClient(base_url='https://runner.example.invalid', http_client=httpx.Client(transport=httpx.MockTransport(timeout)))
    before = len(calls)
    with pytest.raises(RunnerError):
        uncertain.scientific_submit(request, 'a' * 64)
    assert len(calls) == before + 1


def test_portable_cli_requires_recovery_attestation_and_exposes_new_verbs(capsys, monkeypatch):
    from steerlab_server import client_cli
    from steerlab_server.client.runner import RunnerClient
    def forbidden(*args, **kwargs):
        pytest.fail("Missing attestation must refuse before any HTTP call")
    monkeypatch.setattr(RunnerClient, "_json", forbidden)
    for verb in ('science-plan', 'science-submit', 'resubmit', 'recovery', 'recover'):
        assert client_cli.main(['runner', verb, '--help', '--json']) == 0
        capsys.readouterr()
    assert client_cli.main(['runner', 'recover', 'job', '--runner', 'https://runner.example.invalid',
        '--review-token', 'review', '--reason', 'owner exited', '--json']) == 64
    assert json.loads(capsys.readouterr().out)['error']['repairAction']


def test_stability_job_executes_real_numerical_owner_and_preserves_diagnostic_type(setup, monkeypatch):
    from test_direction_stability import _harness
    root, _, profile = setup
    calls = _harness(root, monkeypatch)
    manifest_path = next((root / 'experiments').rglob('experiment.json'))
    document = json.loads(manifest_path.read_text())
    document['modelRevision'] = 'a' * 40
    manifest_path.write_text(json.dumps(document))
    request = {'operation': 'stability', 'parameters': {'experiment': document['name'], 'concept': document['concepts'][0]['name']}}
    plan = owner.plan(request, profile)
    assert calls == []
    packet = root / 'packet.json'; packet.write_text(json.dumps(plan))
    record = root / 'record.json'
    assert owner.execute_packet(packet, 'example-job', record) == 0
    result = json.loads(record.read_text())['result']
    assert len(calls) == 2 and 'runDirectory' not in result
    assert Path(result['diagnosticDirectory']).parent == root / 'diagnostics'
    assert Path(result['reportPath']).exists() and result['resumable'] is False
    assert json.loads(Path(result['reportPath']).read_text())['modelRevision'] == 'a' * 40


def test_slurm_child_record_retains_reconciliation_and_allocation_identity(setup, monkeypatch):
    from dataclasses import replace
    from steerlab_server.experiment import battery_run
    root, request, profile = setup
    profile = replace(profile, executor='slurm')
    monkeypatch.setenv('SLURM_JOB_ID', '12345')
    plan = owner.plan(request, profile)
    packet = root / 'request.json'; packet.write_text(json.dumps(plan))
    records = root / 'records'; records.mkdir()
    monkeypatch.setattr(battery_run, 'execute', lambda *a, **k: {'runDirectory': str(root / 'runs/result')})
    jobs = JobManager(store=DurableJobStore(str(root / 'jobs.sqlite')), sweep_orphans=False)
    job = jobs.record_external('science:battery', status='parked', executor='slurm', job_id='example-job', result={'recordsDirectory': str(records)})
    assert owner.execute_packet(packet, job.id, records / (job.id + '.json')) == 0
    jobs.reconcile(str(records))
    recovered = jobs.get(job.id)
    assert recovered.executor_job_id == '12345' and recovered.status == 'succeeded'
    assert recovered.result['recordsDirectory'] == str(records)


def test_large_seed_canonicalization_is_exact_and_bool_or_float_refuse():
    def request(seed):
        return {'operation': 'stability', 'parameters': {'experiment': 'example', 'concept': 'signal', 'seed': seed}}
    for seed in (2**53 + 1, 2**64 - 1):
        assert owner.request_document(request(seed))['parameters']['seed'] == str(seed)
        assert owner.request_document(request(str(seed))) == owner.request_document(request(seed))
    for seed in (True, 1.5, -1, 2**64):
        with pytest.raises(owner.ScientificRefusal):
            owner.request_document(request(seed))


def test_changed_submission_packet_refuses_before_the_owner(setup, monkeypatch):
    from steerlab_server.experiment import battery_run
    root, request, profile = setup
    plan = owner.plan(request, profile)
    expected = plan['planSHA256']
    plan['request']['parameters']['dtype'] = 'float32'
    packet = root / 'packet.json'; packet.write_text(json.dumps(plan))
    monkeypatch.setattr(battery_run, 'execute', lambda *a, **k: pytest.fail('changed packet must not execute'))
    record = root / 'record.json'
    assert owner.execute_packet(packet, 'example-job', record, expected) == 70
    assert 'plan changed' in json.loads(record.read_text())['error']


@pytest.mark.parametrize('role', ['controller', 'gpu-session'])
def test_session_workers_and_local_controllers_cannot_start_diagnostic_children(setup, monkeypatch, role):
    root, request, profile = setup
    monkeypatch.setenv('STEERLAB_SERVER_ROLE', role)
    with pytest.raises(owner.ScientificRefusal):
        owner.plan(request, profile)
    assert not (root / 'runs').exists()


def test_plan_is_bound_to_the_reviewed_controller(setup, monkeypatch):
    from steerlab_server.api import job_ownership
    root, request, profile = setup
    first = owner.plan(request, profile)
    monkeypatch.setattr(job_ownership, 'current_owner', lambda: ('different-controller', 123))
    assert owner.plan(request, profile)['planSHA256'] != first['planSHA256']
    with pytest.raises(owner.ScientificRefusal):
        owner.submit(request, first['planSHA256'], profile=profile, jobs=None)
    assert not (root / 'runs').exists()


def test_valid_changed_vector_norm_sidecar_invalidates_battery_plan(setup):
    from test_battery_run import _vector
    root, request, profile = setup
    artifact = _vector(str(root), concept='signal')
    request['parameters']['agents'] = ['baseline', 'signal:1:0.5']
    request['parameters']['modelID'] = 'fake/model'
    first = owner.plan(request, profile)
    sidecar = root / (artifact + '.json')
    document = json.loads(sidecar.read_text())
    document['residualNormPerLayer'] = [2.0] * 20
    sidecar.write_text(json.dumps(document))
    second = owner.plan(request, profile)
    assert second['inputSHA256'] != first['inputSHA256']
    with pytest.raises(owner.ScientificRefusal, match='changed'):
        owner.submit(request, first['planSHA256'], profile=profile, jobs=None)


@pytest.mark.parametrize(('verb', 'extra', 'method', 'route'), [
    ('science-plan', [], 'POST', '/api/science/plan'),
    ('science-submit', ['--plan-sha256', 'a' * 64], 'POST', '/api/science/submit'),
    ('resubmit', ['--walltime', '03:00:00'], 'POST', '/api/jobs/example-job/resubmit'),
    ('reconcile', [], 'POST', '/api/jobs/reconcile'),
    ('recovery', [], 'GET', '/api/jobs/example-job/recovery'),
    ('recover', ['--review-token', 'review', '--reason', 'owner exited', '--confirm-owner-exited'], 'POST', '/api/jobs/example-job/recover'),
])
def test_portable_cli_dispatches_each_remote_workflow(tmp_path, monkeypatch, capsys, verb, extra, method, route):
    import httpx
    from steerlab_server import client_cli
    from steerlab_server.client import runner
    calls = []
    def respond(request):
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json={'jobId': 'example-job', 'status': 'submitted'})
    original = runner.RunnerClient
    monkeypatch.setattr(runner, 'RunnerClient', lambda **kwargs: original(
        **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    request = tmp_path / 'request.json'
    request.write_text(json.dumps({'operation': 'stability', 'parameters': {'experiment': 'example', 'concept': 'signal'}}))
    args = [] if verb == 'reconcile' else [str(request) if verb.startswith('science-') else 'example-job']
    assert client_cli.main(['runner', verb, *args, *extra, '--runner', 'https://runner.example.invalid', '--json']) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope['result']['response']['jobId'] == 'example-job'
    assert calls == [(method, route)]


def test_gpu_type_is_reviewed_validated_and_bound_into_the_plan(setup, monkeypatch):
    from dataclasses import replace
    root, request, profile = setup
    with pytest.raises(owner.ScientificRefusal, match='omit gpuType'):
        owner.plan(request, profile, gpu_type='A100')  # local executor: no placement to choose
    monkeypatch.setenv('STEERLAB_SERVER_ROLE', 'controller')
    monkeypatch.setenv('STEERLAB_SLURM_GRES', 'gpu:A100:1')
    monkeypatch.setenv('STEERLAB_SLURM_GPU_TYPES', 'A100,H100')
    profile = replace(profile, executor='slurm')
    default = owner.plan(request, profile)
    assert owner.plan(request, profile, gpu_type=None) == default and 'requestedGPUType' not in default
    same = owner.plan(request, profile, gpu_type='A100')
    assert same['resources']['gres'] == 'gpu:A100:1' and same['requestedGPUType'] == 'A100'
    other = owner.plan(request, profile, gpu_type='H100')
    assert other['resources']['gres'] == 'gpu:H100:1' and other['requestedGPUType'] == 'H100'
    assert '--gres=gpu:H100:1' in other['schedulerPreview'] and '--gres=gpu:A100:1' in default['schedulerPreview']
    assert len({default['planSHA256'], same['planSHA256'], other['planSHA256']}) == 3
    with pytest.raises(owner.ScientificRefusal, match='not declared for this site: A100, H100') as refusal:
        owner.plan(request, profile, gpu_type='L4')
    assert 'omit gpuType' in refusal.value.repair_action
    jobs = JobManager(store=DurableJobStore(str(root / 'jobs.sqlite')), sweep_orphans=False)
    with pytest.raises(owner.ScientificRefusal, match='changed after review'):
        owner.submit(request, other['planSHA256'], profile=profile, jobs=jobs)  # reviewed for H100, submitted without it
    assert jobs.list() == []


def test_http_plan_and_submit_carry_the_gpu_type_beside_the_request(setup, monkeypatch):
    from dataclasses import replace
    root, request, profile = setup
    monkeypatch.setenv('STEERLAB_SERVER_ROLE', 'controller')
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'slurm')
    monkeypatch.setenv('STEERLAB_SLURM_GRES', 'gpu:A100:1')
    monkeypatch.setenv('STEERLAB_SLURM_GPU_TYPES', 'A100,H100')
    profile = replace(profile, executor='slurm')
    jobs = JobManager(store=DurableJobStore(str(root / 'jobs.sqlite')), sweep_orphans=False)
    app = FastAPI(); app.include_router(build_scientific_execution_router(SimpleNamespace(jobs=jobs, registry=None)))
    client = TestClient(app)
    bare = client.post('/api/science/plan', json=request).json()
    wrapped = client.post('/api/science/plan', json={'request': request, 'gpuType': 'H100'}).json()
    assert bare == owner.plan(request, profile) and wrapped['requestedGPUType'] == 'H100'
    refused = client.post('/api/science/plan', json={'request': request, 'gpuType': 'L4'})
    assert refused.status_code == 409 and 'not declared' in refused.json()['detail']['reason']
    stale = client.post('/api/science/submit', json={'request': request, 'planSHA256': wrapped['planSHA256']})
    assert stale.status_code == 409 and jobs.list() == []
    malformed = client.post('/api/science/submit', json={'request': request, 'planSHA256': wrapped['planSHA256'], 'gres': 'gpu:H100:1'})
    assert malformed.status_code == 409
