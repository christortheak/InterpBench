import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from steerlab_server.experiment import diagnostic_archives as archives, diagnostic_inputs
from steerlab_server.api import diagnostic_transport as transport, diagnostic_cleanup as cleanup, scientific_execution
from steerlab_server.api.diagnostic_transport_routes import build_router
from steerlab_server.api.jobs import JobManager, DurableJobStore
from test_scientific_execution import setup


@pytest.fixture
def completed(setup, monkeypatch, tmp_path):
    # Import requires a deliberately created destination workspace.
    (tmp_path / 'local').mkdir()
    root, request, profile = setup
    reviewed = diagnostic_inputs.plan(request, root)
    archive = root / 'runs/input.tar.gz'
    packed = diagnostic_inputs.package(request, root, archive, reviewed['planSHA256'])
    staged = transport.stage(archive, packed['bundleSha256'], profile)
    plan = scientific_execution.plan(staged['request'], profile)
    output = Path(plan['root']) / 'runs/example-battery'; output.mkdir(parents=True)
    (output / 'battery-report.json').write_text('{"complete":true}')
    (output / 'battery.jsonl').write_text('{"answer":"example"}\n')
    jobs = JobManager(store=DurableJobStore(str(root / 'jobs.sqlite')), sweep_orphans=False)
    job = jobs.record_external('science:battery', status='succeeded', executor='local', job_id='example-job',
        result={'scientificPlan': plan, 'runDirectory': str(output)})
    job.finished_at = time.time() - 3600; jobs.store.update(job)
    policy = root / 'policy.json'
    policy.write_text(json.dumps({'schemaVersion': 1, 'allowDiagnosticOutputRemoval': True, 'minimumRetentionHours': 0, 'source': 'Researcher supplied retention rule for disposable fixture outputs.'}))
    monkeypatch.setenv('STEERLAB_DIAGNOSTIC_CLEANUP_POLICY', str(policy))
    return root, profile, jobs, job, output


def test_round_trip_isolated_inputs_output_and_offline_custody(completed, tmp_path):
    root, profile, jobs, job, output = completed
    assert str(output).startswith(str(root / 'runs/diagnostic-input-'))
    reference = transport.export(job.id, jobs, profile)
    local = tmp_path / 'local'
    imported = archives.import_evidence(reference['bundlePath'], reference['bundleSha256'], local)
    assert imported['changed'] is True and imported['receipt']['context']['jobID'] == job.id
    assert archives.verify(imported['receiptSHA256'], local) == imported['receipt']
    assert archives.import_evidence(reference['bundlePath'], reference['bundleSha256'], local)['changed'] is False
    assert archives.inventory(local)['receipts'][0]['verified'] is True
    (Path(imported['outputDirectory']) / 'battery.jsonl').write_text('changed')
    with pytest.raises(archives.Refusal): archives.verify(imported['receiptSHA256'], local)
    with pytest.raises(archives.Refusal): archives.import_evidence(reference['bundlePath'], reference['bundleSha256'], local)
    assert output.exists()


def test_import_refuses_missing_root_without_creating_parents(completed, tmp_path, capsys):
    from steerlab_server import client_cli
    _, profile, jobs, job, _ = completed
    reference = transport.export(job.id, jobs, profile)
    missing = tmp_path / 'mistyped' / 'workspace'
    with pytest.raises(archives.Refusal, match='existing workspace'):
        archives.import_evidence(reference['bundlePath'], reference['bundleSha256'], missing)
    assert client_cli.main(['science', 'import', reference['bundlePath'],
                           '--sha256', reference['bundleSha256'], '--root', str(missing), '--json']) == 66
    response = json.loads(capsys.readouterr().out)
    assert 'existing workspace' in str(response)
    assert not missing.parent.exists()

    file_root = tmp_path / 'file-root'
    file_root.write_text('unchanged')
    with pytest.raises(archives.Refusal, match='existing workspace'):
        archives.import_evidence(reference['bundlePath'], reference['bundleSha256'], file_root)
    assert file_root.read_text() == 'unchanged'


def test_input_review_and_staged_bytes_cannot_drift(setup):
    root, request, profile = setup
    plan = diagnostic_inputs.plan(request, root)
    battery = root / request['parameters']['batteryFile']
    battery.write_text(battery.read_text() + '\n')
    with pytest.raises(archives.Refusal): diagnostic_inputs.package(request, root, root / 'no.tar.gz', plan['planSHA256'])
    fresh = diagnostic_inputs.plan(request, root)
    packed = diagnostic_inputs.package(request, root, root / 'runs/input.tar.gz', fresh['planSHA256'])
    staged = transport.stage(packed['bundlePath'], packed['bundleSha256'], profile)
    copy = Path(staged['executionRoot']) / request['parameters']['batteryFile']; copy.write_text('changed')
    with pytest.raises(archives.Refusal): scientific_execution.plan(staged['request'], profile)


def test_cleanup_rechecks_custody_policy_dependencies_and_exact_target(completed, tmp_path):
    root, profile, jobs, job, output = completed
    reference = transport.export(job.id, jobs, profile)
    imported = archives.import_evidence(reference['bundlePath'], reference['bundleSha256'], tmp_path / 'local')
    receipt = archives.verify(imported['receiptSHA256'], tmp_path / 'local')
    planned = cleanup.plan(job.id, receipt, jobs, profile)
    assert planned['eligible']
    blocked = jobs.record_external('example', status='checkpointed', executor='local', job_id='other-job')
    assert not cleanup.plan(job.id, receipt, jobs, profile)['eligible']
    with pytest.raises(archives.Refusal): cleanup.apply(job.id, receipt, planned['planSHA256'], jobs, profile, confirmed=True)
    assert output.exists()
    blocked.status = 'succeeded'; jobs.store.update(blocked)
    fresh = cleanup.plan(job.id, receipt, jobs, profile)
    with pytest.raises(archives.Refusal): cleanup.apply(job.id, receipt, fresh['planSHA256'], jobs, profile, confirmed=False)
    result = cleanup.apply(job.id, receipt, fresh['planSHA256'], jobs, profile, confirmed=True)
    assert result['state'] == 'removed' and not output.exists()
    assert Path(reference['bundlePath']).exists()
    assert Path(result['auditPath']).exists()
    assert archives.verify(imported['receiptSHA256'], tmp_path / 'local') == receipt
    assert jobs.get(job.id).status == 'succeeded'


def test_cleanup_refuses_stale_policy_changed_output_and_declared_dependencies(completed, tmp_path):
    root, profile, jobs, job, output = completed
    reference = transport.export(job.id, jobs, profile)
    receipt = archives.import_evidence(reference['bundlePath'], reference['bundleSha256'], tmp_path / 'local')['receipt']
    planned = cleanup.plan(job.id, receipt, jobs, profile)
    policy = root / 'policy.json'; document = json.loads(policy.read_text()); document['source'] += ' revised'; policy.write_text(json.dumps(document))
    with pytest.raises(archives.Refusal): cleanup.apply(job.id, receipt, planned['planSHA256'], jobs, profile, confirmed=True)
    directory = root / 'experiments'; directory.mkdir()
    (directory / 'other.json').write_text(json.dumps({'vectorArtifactID': 'runs/example-battery/vector'}))
    assert not cleanup.plan(job.id, receipt, jobs, profile)['eligible']
    (output / 'battery.jsonl').write_text('different')
    with pytest.raises(archives.Refusal): cleanup.plan(job.id, receipt, jobs, profile)
    assert output.exists()


def test_archive_hash_links_and_wrong_receipt_root_refuse(completed, tmp_path):
    root, profile, jobs, job, output = completed
    reference = transport.export(job.id, jobs, profile)
    local = tmp_path / 'local'
    with pytest.raises(archives.Refusal): archives.import_evidence(reference['bundlePath'], '0'*64, local)
    imported = archives.import_evidence(reference['bundlePath'], reference['bundleSha256'], local)
    with pytest.raises(archives.Refusal): archives.verify_document(imported['receipt'], root)
    (output / 'link').symlink_to(root / 'policy.json')
    with pytest.raises(archives.Refusal): transport.export(job.id, jobs, profile)


def test_http_cleanup_requires_confirmation_and_closed_fields(completed, tmp_path):
    root, profile, jobs, job, output = completed
    app = FastAPI(); app.include_router(build_router(SimpleNamespace(jobs=jobs)))
    client = TestClient(app)
    reference = client.post('/api/science/jobs/example-job/export', json={}).json()
    receipt = archives.import_evidence(reference['bundlePath'], reference['bundleSha256'], tmp_path / 'local')['receipt']
    planned = client.post('/api/science/jobs/example-job/cleanup-plan', json={'custody': receipt}).json()
    response = client.post('/api/science/jobs/example-job/cleanup-apply', json={'custody': receipt, 'planSHA256': planned['planSHA256']})
    assert response.status_code == 409 and output.exists()
    job.status = 'cancelled'; jobs.store.update(job)
    assert client.post('/api/science/jobs/example-job/export').status_code == 409


def test_publication_never_replaces_even_an_empty_directory(tmp_path):
    source = tmp_path / 'source'; source.mkdir(); (source / 'file').write_text('original')
    target = tmp_path / 'target'; target.mkdir()
    with pytest.raises(OSError): archives.publish_directory(source, target)
    assert source.is_dir() and list(target.iterdir()) == []


def test_custody_pins_receipt_bytes_and_full_output_members(completed, tmp_path):
    _, profile, jobs, job, _ = completed
    reference = transport.export(job.id, jobs, profile)
    local = tmp_path / 'local'
    imported = archives.import_evidence(reference['bundlePath'], reference['bundleSha256'], local)
    receipt = local / '.steerlab/diagnostic-custody' / (imported['receiptSHA256'] + '.json')
    original = receipt.read_bytes(); receipt.write_bytes(original + b' ')
    with pytest.raises(archives.Refusal): archives.verify(imported['receiptSHA256'], local)
    receipt.write_bytes(original)
    (Path(imported['outputDirectory']) / 'extra.json').write_text('{}')
    with pytest.raises(archives.Refusal): archives.verify(imported['receiptSHA256'], local)


def test_archive_rejects_duplicate_members_and_symlinks(tmp_path):
    import io
    import tarfile
    for bad in ('duplicate', 'symlink'):
        archive = tmp_path / (bad + '.tar.gz')
        with tarfile.open(archive, 'w:gz') as handle:
            for _ in range(2 if bad == 'duplicate' else 1):
                entry = tarfile.TarInfo('file')
                if bad == 'symlink': entry.type = tarfile.SYMTYPE; entry.linkname = '/outside'
                handle.addfile(entry, io.BytesIO(b''))
        with pytest.raises(archives.Refusal): archives.inspect(archive, archives.file_hash(archive))


def test_wrong_export_origin_refuses_before_local_publication(completed, tmp_path):
    _, profile, jobs, job, _ = completed
    reference = transport.export(job.id, jobs, profile)
    local = tmp_path / 'local'
    with pytest.raises(archives.Refusal): archives.import_evidence(reference['bundlePath'], reference['bundleSha256'], local, expected_context={})
    assert not (local / 'runs').exists()


def test_cli_packaging_and_local_receipts_need_no_gpu_imports(setup, tmp_path):
    import os
    import subprocess
    import sys
    root, request, _ = setup
    request_path = root / 'request.json'; request_path.write_text(json.dumps(request))
    script = '''
import json,sys
from steerlab_server.client.diagnostic_commands import workspace_action
result=workspace_action('input-plan', {'workspaceRoot':sys.argv[1],'requestFile':sys.argv[2]})
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
print(json.dumps(result))
'''
    process = subprocess.run([sys.executable, '-c', script, str(root), str(request_path)], env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])}, cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)['files'][0]['path'] == request['parameters']['batteryFile']


def test_http_client_round_trip_and_cleanup_over_real_loopback(completed, tmp_path):
    import os
    import socket
    import subprocess
    import sys
    import httpx
    from steerlab_server import client_cli
    root, profile, jobs, job, output = completed
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    endpoint = f'http://127.0.0.1:{port}'
    log = open(tmp_path / 'server.log', 'w')
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1]), 'STEERLAB_JOBS_DB': str(root / 'jobs.sqlite'), 'STEERLAB_AUTH_MODE': 'none', 'STEERLAB_BIND': '127.0.0.1'}
    process = subprocess.Popen([sys.executable, '-m', 'steerlab_server.cli', 'serve', '--root', str(root), '--port', str(port)], env=env, stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                if httpx.get(endpoint + '/healthz', timeout=.4).status_code == 200: break
            except httpx.HTTPError: pass
            time.sleep(.1)
        else: pytest.fail('Fixture runner did not start')
        from steerlab_server.client.runner import RunnerClient
        wire = RunnerClient(base_url=endpoint)
        uploaded = wire.upload_run_bundle(str(root / 'runs/input.tar.gz'))
        staged = wire.stage_diagnostic(uploaded['path'], uploaded['sha256'])
        assert staged['request']['inputBundleSHA256'] == uploaded['sha256']
        local = tmp_path / 'local'
        assert client_cli.main(['runner', 'science-fetch', job.id, '--runner', endpoint, '--root', str(local), '--json']) == 0
        receipt = archives.inventory(local)['receipts'][0]['receiptSHA256']
        assert client_cli.main(['runner', 'cleanup-apply', job.id, '--runner', endpoint, '--root', str(local), '--receipt-sha256', receipt, '--plan-sha256', '0'*64, '--json']) != 0
        assert output.exists()
        from steerlab_server.client.runner import RunnerClient
        with RunnerClient(base_url=endpoint) as client:
            custody = archives.verify(receipt, local)
            plan = client.diagnostic_cleanup_plan(job.id, custody)
        assert client_cli.main(['runner', 'cleanup-apply', job.id, '--runner', endpoint, '--root', str(local), '--receipt-sha256', receipt, '--plan-sha256', plan['planSHA256'], '--confirm-removal', '--json']) == 0
        assert not output.exists() and archives.verify(receipt, local)
    finally:
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        log.close()


def test_cleanup_sees_other_controller_jobs_missing_from_manager_cache(completed, tmp_path):
    _, profile, jobs, job, output = completed
    exported = transport.export(job.id, jobs, profile)
    receipt = archives.import_evidence(exported['bundlePath'], exported['bundleSha256'], tmp_path / 'local')['receipt']
    other = JobManager(store=DurableJobStore(jobs.store.path), sweep_orphans=False)
    other.record_external('example', status='submitted', executor='local', job_id='foreign-job')
    assert jobs.get('foreign-job') is None
    plan = cleanup.plan(job.id, receipt, jobs, profile)
    assert not plan['eligible'] and any('foreign-job' in b for b in plan['blockers']) and output.exists()


def test_workspace_actions_reject_extra_fields_and_missing_required_values(tmp_path):
    from steerlab_server.client.diagnostic_commands import workspace_action
    with pytest.raises(archives.Refusal): workspace_action('custody', {'workspaceRoot': str(tmp_path), 'unexpected': True})
    with pytest.raises(archives.Refusal): workspace_action('package', {'workspaceRoot': str(tmp_path), 'requestFile': None})


def test_stability_inputs_reproduce_admission_in_isolated_workspace(setup, monkeypatch):
    from test_direction_stability import _harness
    root, _, profile = setup
    calls = _harness(root, monkeypatch)
    manifest = next((root / 'experiments').rglob('experiment.json'))
    document = json.loads(manifest.read_text()); document['modelRevision'] = 'a' * 40; manifest.write_text(json.dumps(document))
    request = {'operation': 'stability', 'parameters': {'experiment': document['name'], 'concept': document['concepts'][0]['name']}}
    plan = diagnostic_inputs.plan(request, root)
    packed = diagnostic_inputs.package(request, root, root / 'runs/input.tar.gz', plan['planSHA256'])
    staged = transport.stage(packed['bundlePath'], packed['bundleSha256'], profile)
    execution = scientific_execution.plan(staged['request'], profile)
    assert execution['request']['operation'] == 'stability' and calls == []
    assert execution['root'] != str(root)


def test_battery_condition_input_archive_contains_vector_sidecar_and_tensors(setup):
    from test_battery_run import _vector
    root, request, profile = setup
    vector = _vector(str(root), concept='signal')
    request['parameters'].update(agents=['baseline', 'signal:1:0.5'], modelID='fake/model')
    plan = diagnostic_inputs.plan(request, root)
    assert {vector+'.json', vector+'.safetensors'} <= {e['path'] for e in plan['files']}
    packed = diagnostic_inputs.package(request, root, root/'runs/input.tar.gz', plan['planSHA256'])
    stage = transport.stage(packed['bundlePath'], packed['bundleSha256'], profile)
    assert scientific_execution.plan(stage['request'], profile)['request']['parameters']['agents'] == request['parameters']['agents']


def test_adapter_sidecar_and_weight_drift_are_caught_before_queued_execution(setup):
    from test_battery_run import _adapter_directory, _adapter_agent
    root, request, profile = setup
    adapter = _adapter_directory(str(root))
    name = _adapter_agent(str(root), adapter)
    variant = root / 'runs/model-variants' / (name + '.json')
    document = json.loads(variant.read_text()); document['baseRevision'] = 'a' * 40; variant.write_text(json.dumps(document))
    request['parameters'].update(agents=['tuned=runs/model-variants/tuned.json'], modelID='fake/model')
    plan = diagnostic_inputs.plan(request, root)
    assert adapter + '.json' in {e['path'] for e in plan['files']}
    packed = diagnostic_inputs.package(request, root, root / 'runs/input.tar.gz', plan['planSHA256'])
    stage = transport.stage(packed['bundlePath'], packed['bundleSha256'], profile)
    reviewed = scientific_execution.plan(stage['request'], profile)
    packet = root / 'packet.json'; packet.write_text(json.dumps(reviewed))
    (Path(reviewed['root']) / adapter / 'adapter_model.safetensors').write_bytes(b'changed')
    record = root / 'record.json'
    assert scientific_execution.execute_packet(packet, 'example-job', record) == 70
    result = json.loads(record.read_text())
    assert 'Staged input bytes changed' in result['error'] and 'runDirectory' not in result['result']
    document['adapters'][0]['adapterDirectory'] = str(root / adapter); variant.write_text(json.dumps(document))
    with pytest.raises(archives.Refusal): diagnostic_inputs.plan(request, root)


def test_cleanup_default_denies_and_retention_is_an_admission_gate(completed, tmp_path, monkeypatch):
    root, profile, jobs, job, output = completed
    reference = transport.export(job.id, jobs, profile)
    receipt = archives.import_evidence(reference['bundlePath'], reference['bundleSha256'], tmp_path / 'local')['receipt']
    policy = root / 'policy.json'; document = json.loads(policy.read_text())
    document['minimumRetentionHours'] = 24; policy.write_text(json.dumps(document))
    plan = cleanup.plan(job.id, receipt, jobs, profile)
    assert not plan['eligible'] and any('retention' in b for b in plan['blockers'])
    monkeypatch.delenv('STEERLAB_DIAGNOSTIC_CLEANUP_POLICY')
    with pytest.raises(archives.Refusal, match='No diagnostic cleanup policy'): cleanup.plan(job.id, receipt, jobs, profile)
    assert output.exists()


def test_cleanup_quarantined_byte_change_restores_output_and_records_attention(completed, tmp_path, monkeypatch):
    _, profile, jobs, job, output = completed
    reference = transport.export(job.id, jobs, profile)
    receipt = archives.import_evidence(reference['bundlePath'], reference['bundleSha256'], tmp_path / 'local')['receipt']
    plan = cleanup.plan(job.id, receipt, jobs, profile)
    original = archives.publish_directory
    def move(source, destination):
        original(source, destination)
        if destination.name.startswith('.diagnostic-cleanup-'):
            (destination / 'battery.jsonl').write_text('changed by uncoordinated writer')
    monkeypatch.setattr(archives, 'publish_directory', move)
    with pytest.raises(archives.Refusal, match='needs attention'): cleanup.apply(job.id, receipt, plan['planSHA256'], jobs, profile, confirmed=True)
    assert output.exists() and (output / 'battery.jsonl').read_text().startswith('changed')
    audit = next((Path(profile.metadata_root) / 'diagnostic-cleanup').glob('*.json'))
    record = json.loads(audit.read_text())
    assert record['state'] == 'attention' and str(output) in record['retained'] and record['removed'] == []
