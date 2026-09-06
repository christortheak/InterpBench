"""Reviewed preparation and authoring journeys, including failure preservation."""
import json
from pathlib import Path

import pytest

from steerlab_server import client_cli
from steerlab_server.client import authoring_files, design_files, study_agents, study_panels, study_pipeline
from steerlab_server.experiment import experiment_store as store, manifest_files
from steerlab_server.experiment.manifest_errors import ExperimentStoreError
from test_client_study_designs import seed, panel_design, agent, restore_cli_root


def test_agent_catalog_attach_preserves_exact_artifact_and_other_study_fields(tmp_path):
    source = seed(tmp_path)
    ref = agent(tmp_path, source['document']['modelID'])
    before = (tmp_path / ref['artifactPath']).read_bytes()
    catalog = study_agents.catalog(root=tmp_path)
    assert [a['path'] for a in catalog['agents']] == [ref['artifactPath']]
    attached = study_agents.attach(source['name'], ref['artifactPath'], root=tmp_path,
        expected=source['manifestFileSHA256'], artifact_sha256=ref['artifactFileSHA256'])
    assert attached['document']['variantConditions'][0]['artifact'] == json.loads(before)
    assert attached['document']['taskPromptsHash'] == source['document']['taskPromptsHash']
    assert (tmp_path / ref['artifactPath']).read_bytes() == before
    with pytest.raises(ExperimentStoreError):
        study_agents.attach(source['name'], ref['artifactPath'], root=tmp_path, expected=source['manifestFileSHA256'], artifact_sha256=ref['artifactFileSHA256'])


def test_panel_authoring_compile_and_pipeline_edit_are_reviewed(tmp_path):
    saved, panel = panel_design(tmp_path)
    data = design_files.encode(panel)
    source = authoring_files.snapshot('shared-study', tmp_path)
    published = study_panels.publish(data, root=tmp_path, expected=manifest_files.digest_bytes(data))
    assert not study_panels.publish(data, root=tmp_path, expected=manifest_files.digest_bytes(data))['changed']
    ref = agent(tmp_path, source['document']['modelID'])
    compiled = study_panels.compile('shared-study', published['path'], {'seats': {'first': ref, 'second': None}},
        root=tmp_path, expected=source['manifestFileSHA256'], file_sha256=published['fileSHA256'])
    assert compiled['document']['multiAgentSemanticScenarioHash'] == published['fileSHA256']
    bound = json.loads((tmp_path / compiled['document']['multiAgentScenarioPath']).read_bytes())
    assert bound['agents'][0]['variantArtifactHash'] == ref['artifactFileSHA256']
    assert bound['turns'] == panel['turns']
    pipeline = {'stages': ['run', 'evaluate', 'analyze']}
    updated = study_pipeline.save('shared-study', pipeline, root=tmp_path, expected=compiled['manifestFileSHA256'])
    assert updated['document']['pipeline'] == pipeline
    assert updated['document']['multiAgentScenarioHash'] == compiled['document']['multiAgentScenarioHash']
    cleared = study_pipeline.save('shared-study', None, root=tmp_path, expected=updated['manifestFileSHA256'])
    assert 'pipeline' not in cleared['document']


@pytest.mark.parametrize('block', [{'stages':['analyze']}, {'stages':['run'], 'typo':True}, {'stages':['run'], 'gates':{'validate':{'minScenarioAccuracy':0.6}}}])
def test_pipeline_invalid_declaration_never_changes_source(tmp_path, block):
    source = seed(tmp_path)
    path = authoring_files.study_path(tmp_path, source['name'])
    before = path.read_bytes()
    with pytest.raises(ExperimentStoreError):
        study_pipeline.save(source['name'], block, root=tmp_path, expected=source['manifestFileSHA256'])
    assert path.read_bytes() == before


def test_panel_refuses_bindings_and_unknown_seats_without_writing(tmp_path):
    _, panel = panel_design(tmp_path)
    panel['baseModelID'] = 'unreviewed/model'
    data = design_files.encode(panel)
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*.json'))
    with pytest.raises(ExperimentStoreError):
        study_panels.publish(data, root=tmp_path, expected=manifest_files.digest_bytes(data))
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*.json')) == before


def test_model_plan_is_offline_and_policy_changes_invalidate_review(tmp_path, monkeypatch):
    from steerlab_server.api import model_preparation
    from steerlab_server.steering import model_loader
    from fastapi import HTTPException
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('HF_HUB_CACHE', str(tmp_path / 'cache'))
    monkeypatch.setenv('STEERLAB_COMPUTE_EGRESS', 'yes')
    monkeypatch.setattr(model_loader, 'snapshot_size_bytes', lambda *a: None)
    plan = model_preparation.plan('test/model')
    assert plan['cacheFileSetPresent'] is False
    assert plan['installationAllowed'] and plan['memoryFit'] == 'notChecked'
    model_preparation.admit_install('test/model', None, plan['planSHA256'])
    with pytest.raises(HTTPException) as stale:
        model_preparation.admit_install('test/other', None, plan['planSHA256'])
    assert stale.value.status_code == 412
    monkeypatch.setenv('STEERLAB_COMPUTE_EGRESS', 'no')
    assert not model_preparation.plan('test/model')['installationAllowed']
    with pytest.raises(HTTPException) as denied:
        model_preparation.admit_install('test/model', None)
    assert denied.value.status_code == 409


def test_model_adapter_submits_once_and_observes_original_job():
    import httpx
    from steerlab_server.client.runner import RunnerClient
    requests = []
    def handler(request):
        requests.append(request)
        if request.url.path.endswith('/plan'):
            return httpx.Response(200, json={'planSHA256':'a'*64})
        if request.url.path.endswith('/install'):
            assert json.loads(request.content)['planSHA256'] == 'a'*64
            return httpx.Response(200, json={'jobId':'install-1'})
        return httpx.Response(200, json={'id':'install-1','kind':'model:install','status':'completed'})
    with httpx.Client(transport=httpx.MockTransport(handler)) as wire:
        with RunnerClient(base_url='http://localhost:8080', http_client=wire) as client:
            plan = client.model_plan('test/model')
            job = client.install_model('test/model', plan_sha256=plan['planSHA256'])
            assert client.job(job['jobId'])['status'] == 'completed'
    assert [r.method for r in requests] == ['GET','POST','GET']


@pytest.mark.parametrize('args', [('agent','list','extra'), ('experiment','set-pipeline','study'), ('panel','compile','path'), ('model','install','test/model')])
def test_new_cli_required_arguments_fail_before_io(tmp_path, capsys, args):
    code = client_cli.main([*args, '--root', str(tmp_path), '--json'])
    result = json.loads(capsys.readouterr().out)
    assert code == 64 and result['error']['repairAction']
    assert not list(tmp_path.iterdir())


def test_expansion_is_distinct_reviewable_and_does_not_mint(tmp_path):
    from steerlab_server.client import design_expansion
    saved, _ = panel_design(tmp_path)
    ref = agent(tmp_path, saved['document']['study']['modelID'])
    before = list((tmp_path / 'experiments').iterdir())
    plan = design_expansion.expand(saved['name'], {'seats':{'first':ref,'second':None}}, 'permutations', root=tmp_path, expected=saved['designFileSHA256'])
    assert plan['count'] == 2
    rows = plan['batch']['rows']
    assert rows[0]['casting']['seats']['first'] is None
    assert rows[1]['casting']['seats']['second'] is None
    sweep = design_expansion.expand(saved['name'], {'agents':[ref]}, 'composition', root=tmp_path, expected=saved['designFileSHA256'])
    assert sweep['count'] == 4
    assert list((tmp_path / 'experiments').iterdir()) == before
    baseline = design_expansion.expand(saved['name'], {'seats':{'first':None,'second':None}}, 'permutations', root=tmp_path, expected=saved['designFileSHA256'])
    assert baseline['count'] == 1


def test_real_model_http_plan_review_policy_and_durable_job(tmp_path, monkeypatch):
    import time
    from fastapi.testclient import TestClient
    from steerlab_server.api.app import app
    from steerlab_server.api import model_install
    from steerlab_server.client.runner import RunnerClient, RunnerHTTPError
    from steerlab_server.steering import model_loader
    monkeypatch.setenv('STEERLAB_COMPUTE_EGRESS', 'yes')
    monkeypatch.setattr(model_loader, 'snapshot_size_bytes', lambda *a: None)
    monkeypatch.setattr(model_install, 'run_install', lambda *a, **kw: str(tmp_path / 'cache'))
    from steerlab_server.experiment import model_capabilities
    monkeypatch.setattr(model_capabilities, 'probe_model', lambda *a: (_ for _ in ()).throw(RuntimeError('no tokenizer fixture')))
    wire = TestClient(app)
    with RunnerClient(base_url='http://testserver', http_client=wire) as client:
        plan = client.model_plan('test/model')
        assert plan['cacheFileSetPresent'] is False
        with pytest.raises(RunnerHTTPError):
            client.install_model('test/changed', plan_sha256=plan['planSHA256'])
        submitted = client.install_model('test/model', plan_sha256=plan['planSHA256'])
        for _ in range(100):
            record = client.job(submitted['jobId'])
            if record['status'] in ('succeeded','failed'):
                break
            time.sleep(0.01)
        assert record['status'] == 'succeeded'
        assert record['result']['path'] == str(tmp_path / 'cache')
        monkeypatch.setenv('STEERLAB_COMPUTE_EGRESS', 'no')
        with pytest.raises(RunnerHTTPError):
            client.install_model('test/model', plan_sha256=plan['planSHA256'])


def test_model_plan_distinguishes_partial_and_complete_sharded_cache(tmp_path, monkeypatch):
    from steerlab_server.api import model_preparation
    import huggingface_hub
    snapshot = tmp_path / 'snapshots' / ('a' * 40)
    snapshot.mkdir(parents=True)
    config = snapshot / 'config.json'; config.write_text('{}')
    monkeypatch.setattr(huggingface_hub, 'try_to_load_from_cache', lambda *a, **kw: str(config))
    assert not model_preparation.cached_file_set('test/model', None)[0]
    (snapshot / 'tokenizer.json').write_text('{}')
    (snapshot / 'tokenizer_config.json').write_text('{}')
    (snapshot / 'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'first':'part-1.safetensors','second':'part-2.safetensors'}}))
    (snapshot / 'part-1.safetensors').write_bytes(b'fixture')
    assert not model_preparation.cached_file_set('test/model', None)[0]
    (snapshot / 'part-2.safetensors').write_bytes(b'fixture')
    assert model_preparation.cached_file_set('test/model', None) == (True, snapshot)


@pytest.mark.parametrize('revision', ['../other', '/absolute', 'a/../../other', '', 'bad\nrevision'])
def test_model_plan_refuses_malformed_revision(revision):
    from fastapi import HTTPException
    from steerlab_server.api.model_preparation import validate
    with pytest.raises(HTTPException) as error:
        validate('test/model', revision)
    assert error.value.status_code == 400


@pytest.mark.parametrize('model_id', ['../repo', 'owner/..', 'owner/repo..name', 'owner/repo--name'])
def test_invalid_hub_model_id_is_a_typed_refusal_before_cache_lookup(monkeypatch, model_id):
    from fastapi import HTTPException
    from steerlab_server.api import model_preparation
    def unexpected_cache(*args):
        pytest.fail('An invalid ID must be refused before inspecting the cache')
    monkeypatch.setattr(model_preparation, 'cached_file_set', unexpected_cache)
    with pytest.raises(HTTPException) as error:
        model_preparation.plan(model_id)
    assert error.value.status_code == 400
    assert error.value.detail['code'] == 'invalidModelID'
    assert error.value.detail['repairAction']


@pytest.mark.parametrize('status, kind, expected_code', [('succeeded', 'model:install', 0), ('failed', 'model:install', 70), ('running', 'experiment:run', 65)])
def test_model_cli_observes_exact_job_and_refuses_foreign_cancellation(tmp_path, capsys, monkeypatch, status, kind, expected_code):
    from steerlab_server.client.runner import RunnerClient
    monkeypatch.setattr(RunnerClient, 'job', lambda self, job_id: {'id': job_id, 'kind': kind, 'status': status})
    def unexpected_cancel(*args):
        pytest.fail('A foreign job must never be cancelled')
    monkeypatch.setattr(RunnerClient, 'cancel_job', unexpected_cancel)
    verb = 'cancel' if kind != 'model:install' else 'status'
    code = client_cli.main(['model', verb, 'install-1', '--runner', 'http://localhost:8080', '--root', str(tmp_path), '--json'])
    envelope = json.loads(capsys.readouterr().out)
    assert code == expected_code
    if code:
        assert envelope['error']['repairAction']
