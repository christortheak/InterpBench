"""P6: acknowledgements, immutable collection, offline comparisons, and old-engine admission."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import httpx
import pytest
from steerlab_server.experiment import instrumentation_contract as contract, instrumentation_evidence as evidence
from steerlab_server.experiment import policy_execution, prompt_render, bundles
from steerlab_server.steering.policy_actions import Action, apply_with_evidence, logits
from steerlab_server.client.runner import RunnerClient, RunnerRefusal
from test_intervention_policies import setup, attached, forward


def response(root):
    doc, _, model = setup(root)
    execution = policy_execution.Execution(attached(doc, root), rendering='rawCompletion', context={'condition': 'modified', 'agent': 'first'})
    with model.hooked.session([]), execution.observe_session(model, prompt_render.RenderedPrompt('', [1, 2], 2)):
        forward(model, [1, 2])
    return {'condition': 'modified', 'promptID': 'example', 'outputTokenIDs': [3],
            'instrumentationRequirements': ['policy-v1', 'policy-evidence-v2'], 'interventionDecisions': execution.result([3])}


def test_applied_evidence_and_alignment_come_from_real_forward(tmp_path):
    row = response(tmp_path); block = row['interventionDecisions']; event = block['decisions'][0]
    assert event['status'] == 'applied'
    assert event['strengths'] == event['appliedStrengths'] == {'change': 2.0}
    assert event['inputTokenID'] == 2 and event['predictedTokenID'] == 3 and event['generatedTokenIndex'] == 0
    assert block['declarations'][0]['probes'][0]['sha256']
    assert 'vector' not in block['declarations'][0]['actions'][0]
    evidence.validate_record(row)
    summary = evidence.summarize([{'condition': 'baseline'}, row])
    group = summary['groups'][0]
    assert summary['responses'] == 2 and summary['uninstrumentedResponses'] == 1
    assert group['applied']['change'] == {'count': 1, 'mean': 2., 'minimum': 2., 'maximum': 2., 'standardDeviation': None, 'nonzeroCount': 1}


@pytest.mark.parametrize('mutation', ['alignment', 'missing', 'acknowledgement', 'strength', 'condition'])
def test_semantic_corruption_is_not_successful_evidence(tmp_path, mutation):
    row = response(tmp_path); block = row['interventionDecisions']; event = block['decisions'][0]
    if mutation == 'alignment': event['predictedTokenID'] = 4
    if mutation == 'missing': row.pop('interventionDecisions')
    if mutation == 'acknowledgement': event['actionOutcomes'] = {}
    if mutation == 'strength': event['appliedStrengths']['change'] = 1
    if mutation == 'condition': block['condition'] = 'other'
    with pytest.raises(ValueError): evidence.validate_record(row)


def test_failed_site_acknowledges_no_applied_actions():
    import torch
    outcomes = []
    a = Action({'kind': 'forceToken', 'tokens': [1]}, torch.tensor([1.]), lambda *args: outcomes.append(args))
    b = Action({'kind': 'forceToken', 'tokens': [2]}, torch.tensor([1.]), lambda *args: outcomes.append(args))
    with pytest.raises(ValueError, match='conflict'): apply_with_evidence(logits, torch.zeros(1, 4), [a, b])
    assert len(outcomes) == 2 and all(x[0] is False for x in outcomes)


def test_legacy_decisions_are_never_counted_as_applied(tmp_path):
    row = response(tmp_path); block = row['interventionDecisions']; block['schemaVersion'] = 1
    row.pop('instrumentationRequirements')
    for event in block['decisions']:
        event['status'] = 'decided'; event.pop('appliedStrengths'); event.pop('actionOutcomes')
    report = evidence.summarize([row])
    assert report['legacyDecisionResponses'] == 1 and report['groups'][0]['applied'] == {}


def test_collection_validates_before_publishing_and_preserves_exact_bytes(tmp_path):
    source = tmp_path / 'source'; source.mkdir(); row = response(source)
    run = source / 'runs' / 'evidence'; run.mkdir(); file = run / 'generations.jsonl'
    raw = json.dumps(row).encode() + b'\n'; file.write_bytes(raw)
    archive = bundles.package_evidence(str(run), root=str(source))
    destination = tmp_path / 'collected'
    bundles.import_bundle(archive['bundlePath'], target_root=str(destination), expected_sha256=archive['bundleSha256'])
    collected = destination / 'runs/evidence/generations.jsonl'
    assert collected.read_bytes() == raw
    report = evidence.inspect(str(collected), destination)
    assert report['sourceSHA256'] == hashlib.sha256(raw).hexdigest()
    # Repackage a structurally wrong result with valid outer/member hashes: semantic admission still refuses.
    row.pop('interventionDecisions'); file.write_text(json.dumps(row) + '\n')
    corrupt = bundles.package_evidence(str(run), root=str(source), output_path=str(tmp_path / 'bad.tar.gz'))
    target = tmp_path / 'rejected'
    with pytest.raises(ValueError): bundles.import_bundle(corrupt['bundlePath'], target_root=str(target))
    assert not (target / 'runs/evidence').exists()


def test_cli_and_http_share_offline_analysis(tmp_path, monkeypatch, capsys):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.diagnostic_transport_routes import build_router
    from steerlab_server.client_cli import main
    row = response(tmp_path); file = tmp_path / 'runs/fit/generations.jsonl'; file.write_text(json.dumps(row) + '\n')
    monkeypatch.setenv('STEERLAB_ROOT', str(tmp_path))
    assert main(['science', 'evidence-analyze', 'runs/fit', '--root', str(tmp_path), '--json']) == 0
    assert 'applied' in capsys.readouterr().out
    app = FastAPI(); app.include_router(build_router(SimpleNamespace()))
    with TestClient(app) as client:
        result = client.post('/api/science/workspace/evidence-analyze', json={'workspaceRoot': str(tmp_path), 'path': 'runs/fit'})
        assert result.status_code == 200 and result.json()['groups'][0]['condition'] == 'modified'


def test_requirements_include_embedded_panel_and_unknown_runtime():
    document = {'agents': [{'artifact': {'interventionPolicies': [{}]}}], 'probeMeasurements': {'probes': [{}]}}
    assert contract.requirements(document) == list(contract.SUPPORTED)
    contract.require(contract.requirements(document), {'instrumentation': list(contract.SUPPORTED)})
    for offered in ({}, {'instrumentation': ['policy-v1']}):
        with pytest.raises(ValueError, match='restart'): contract.require(contract.requirements(document), offered)
    with pytest.raises(ValueError): contract.require(['future-v3'], {'instrumentation': contract.SUPPORTED})


@pytest.mark.parametrize('offered', [[], ['policy-v1'], list(contract.SUPPORTED)])
def test_old_engine_cannot_receive_policy_submission_or_resubmission(offered):
    calls = []
    meta = {'bundleSha256': 'a' * 64, 'runtimeRequirements': ['policy-v1', 'policy-evidence-v2']}
    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path == '/api/capabilities': return httpx.Response(200, json={'instrumentation': offered})
        if request.url.path == '/api/bundles/inspect': return httpx.Response(200, json=meta)
        if request.url.path == '/api/jobs/example': return httpx.Response(200, json={'result': {'runBundle': meta}})
        return httpx.Response(200, json={'jobId': 'next'})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = RunnerClient(base_url='http://runner.test', http_client=http)
        if offered == list(contract.SUPPORTED):
            assert client.submit_uploaded_bundle(remote_path='runs/input.tar.gz', verb='run', expected_sha256='a' * 64)['jobId'] == 'next'
            assert client.resubmit_job('example')['jobId'] == 'next'
        else:
            with pytest.raises(RunnerRefusal, match='restart'): client.submit_uploaded_bundle(remote_path='runs/input.tar.gz', verb='run', expected_sha256='a' * 64)
            with pytest.raises(RunnerRefusal, match='restart'): client.resubmit_job('example')
            assert ('POST', '/api/studies/submit-bundle') not in calls and ('POST', '/api/jobs/example/resubmit') not in calls


def test_shared_swift_admission_fixture_has_real_applied_actions():
    fixture = Path(__file__).resolve().parents[2] / 'Tests/Fixtures/cross-engine/policy-evidence.json'
    row = json.loads(fixture.read_text()); evidence.validate_record(row)
    assert row['interventionDecisions']['decisions'][0]['appliedStrengths'] == {'change': 2.}


def test_requirements_and_archive_hash_use_the_same_captured_json(tmp_path, monkeypatch):
    import tarfile
    source = tmp_path / 'agent.json'; original = b'{"interventionPolicies":[{}]}'
    source.write_bytes(original); requirement_owner = contract.requirements
    def mutate_after_read(document):
        source.write_text('{}')
        return requirement_owner(document)
    monkeypatch.setattr(contract, 'requirements', mutate_after_read)
    required = set(); destination = tmp_path / 'input.tar.gz'
    with tarfile.open(destination, 'w:gz') as archive:
        entries = bundles._add_files(archive, [(str(source), 'runs/fit/agent.json')], runtime_requirements=required)
    with tarfile.open(destination, 'r:gz') as archive:
        raw = archive.extractfile('runs/fit/agent.json').read()
    assert raw == original and entries[0].sha256 == hashlib.sha256(original).hexdigest()
    assert required == {'policy-v1', 'policy-evidence-v2'} and source.read_text() == '{}'


def test_policy_evidence_fixture_is_current(tmp_path):
    from test_portability_contracts import _write_or_compare
    row = response(tmp_path)
    row['interventionDecisions']['timing']['decisionHostSeconds'] = 0
    for event in row['interventionDecisions']['decisions']:
        for outcome in event['actionOutcomes'].values(): outcome['hostSeconds'] = 0
    _write_or_compare('policy-evidence.json', row)


def test_queued_child_guard_refuses_old_or_incompatible_installations(tmp_path, monkeypatch):
    import subprocess, sys, os
    from steerlab_server.api.submissions import _bundle_execute_command
    kwargs = dict(verb='run', target_root=str(tmp_path), dtype='float32', device='cpu', prompts_path=None, source_path=None, package_evidence=False, record_path=str(tmp_path/'record.json'))
    ordinary = _bundle_execute_command('bundle.tar.gz', **kwargs)
    guarded = _bundle_execute_command('bundle.tar.gz', **kwargs, runtime_requirements=['policy-v1', 'policy-evidence-v2'])
    assert ordinary[2] == 'steerlab_server.cli' and guarded[2] == 'steerlab_server.instrumented_bundle'
    assert guarded[4:] == ordinary[3:]
    # Real separate Python process with a pre-feature package cannot execute the request.
    old = tmp_path/'steerlab_server'; old.mkdir(); (old/'__init__.py').write_text('')
    env = {**os.environ, 'PYTHONPATH': str(tmp_path)}
    result = subprocess.run(guarded, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'instrumented_bundle' in result.stderr
    assert not (tmp_path/'record.json').exists()
    from steerlab_server import instrumented_bundle, cli
    calls = []; monkeypatch.setattr(cli, 'main', lambda args: calls.append(args) or 0)
    assert instrumented_bundle.main(['["future-v3"]', 'bundle', 'execute', 'example']) == 65 and calls == []
    assert instrumented_bundle.main([guarded[3], *guarded[4:]]) == 0 and calls == [ordinary[3:]]
