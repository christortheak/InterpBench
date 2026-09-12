import json
from pathlib import Path
import pytest
from steerlab_server.api import jlens_rounds, scientific_execution as engine, diagnostic_transport
from steerlab_server.api.jobs import JobManager, DurableJobStore
from steerlab_server.experiment import jlens_round, diagnostic_inputs, diagnostic_archives as archives
from test_jlens_fit import fitting
from test_scientific_execution import setup


@pytest.fixture
def round_job(fitting,setup):
    root,cfg=fitting;_,_,profile=setup
    path=root/'fit.json';path.write_text(json.dumps({'operation':'jlens-fit','parameters':{'config':cfg}}))
    config=jlens_round.RoundConfig.from_dict(dict(modelID=cfg['modelID'],revision=cfg['revision'],fittingRequest={'path':'fit.json','sha256':archives.file_hash(path)},shards=2,maxConcurrent=1))
    request={'operation':'jlens-fit-round','parameters':{'config':config.to_dict()}}
    plan=diagnostic_inputs.plan(request,root)
    (root/'runs').mkdir(exist_ok=True)
    archive=diagnostic_inputs.package(request,root,root/'runs/input.tar.gz',plan['planSHA256'])
    staged=diagnostic_transport.stage(archive['bundlePath'],archive['bundleSha256'],profile)
    reviewed=engine.plan(staged['request'],profile)
    output=jlens_round.materialize(config,root=reviewed['root'])
    jobs=JobManager(store=DurableJobStore(str(root/'jobs.sqlite')),sweep_orphans=False)
    parent=jobs.record_external('science:jlens-fit-round',status='succeeded',executor='local',job_id='round-fixture',result={'scientificPlan':reviewed,**output})
    return parent,jobs,profile


def test_reviewed_topups_use_durable_jobs_and_do_not_exceed_capacity(round_job,monkeypatch):
    parent,jobs,profile=round_job
    def submit(request,expected,*,profile,jobs,execution_capsule,round_submission):
        plan=engine.plan(request,profile,execution_capsule=execution_capsule,round_submission=round_submission)
        assert plan['planSHA256']==expected
        job=jobs.record_external('science:jlens-fit',status='submitted',executor='slurm',result={'scientificPlan':plan})
        return {'jobId':job.id,'status':job.status}
    monkeypatch.setattr(engine,'submit',submit)
    review=jlens_rounds.action(parent.id,'plan',None,False,jobs,profile)
    assert review['submitIndices']==[0]
    result=jlens_rounds.action(parent.id,'submit',review['planSHA256'],True,jobs,profile)
    assert len(result['submissions'])==1
    child=jobs.get(result['submissions'][0]['jobId'])
    assert child.result['scientificPlan']['executionCapsuleSHA256']==parent.result['scientificPlan']['inputBundleSHA256']
    next_plan=jlens_rounds.action(parent.id,'plan',None,False,jobs,profile)
    assert next_plan['submitIndices']==[]
    assert next_plan['capacity']['activeJobs'] == [{'jobID':child.id,'kind':child.kind,'status':child.status,'belongsToThisRound':True}]
    child.status='succeeded';jobs.store.update(child)
    final=jlens_rounds.action(parent.id,'plan',None,False,jobs,profile)
    assert final['submitIndices']==[1]
    with pytest.raises(archives.Refusal,match='changed'):
        jlens_rounds.action(parent.id,'submit',review['planSHA256'],True,jobs,profile)


def test_uncertain_submission_is_not_retried(round_job,monkeypatch):
    parent,jobs,profile=round_job
    def lost(*a,**kw):raise RuntimeError('fixture uncertain submission')
    monkeypatch.setattr(engine,'submit',lost)
    review=jlens_rounds.action(parent.id,'plan',None,False,jobs,profile)
    with pytest.raises(RuntimeError):jlens_rounds.action(parent.id,'submit',review['planSHA256'],True,jobs,profile)
    review=jlens_rounds.action(parent.id,'plan',None,False,jobs,profile)
    assert review['shards'][0]['status']=='uncertain'
    assert review['submitIndices']==[] and review['availableSlots']==0
    assert review['capacity']['uncertainShardIndices'] == [0]
    assert review['capacity']['occupiedSlots'] == 1


def test_round_plan_tampering_refuses_before_submission(round_job):
    parent,jobs,profile=round_job
    path=Path(parent.result['runDirectory'])/'round-plan.json'
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(archives.Refusal,match='changed'):
        jlens_rounds.action(parent.id,'plan',None,False,jobs,profile)


def test_http_round_review_and_exact_mutation_body(round_job):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from steerlab_server.api.diagnostic_transport_routes import build_router
    parent,jobs,profile=round_job
    app=FastAPI();app.include_router(build_router(SimpleNamespace(jobs=jobs)))
    with TestClient(app) as client:
        response=client.post('/api/science/fitting-round/'+parent.id+'/plan',json={})
        assert response.status_code==200 and response.json()['submitIndices']==[0]
        response=client.post('/api/science/fitting-round/'+parent.id+'/submit',json={})
        assert response.status_code==409 and 'Supply exactly' in response.json()['detail']['reason']


def test_capacity_explains_unrelated_science_jobs_without_changing_scope(round_job):
    parent, jobs, profile = round_job
    unrelated = jobs.record_external('science:jlens-fit-assess', status='running', executor='slurm',
                                     result={'scientificPlan':{'roundSubmission':None}})
    jobs.record_external('science:jlens-fit', status='succeeded', executor='slurm')
    jobs.record_external('chat', status='running', executor='local')
    review = jlens_rounds.action(parent.id, 'plan', None, False, jobs, profile)
    assert review['availableSlots'] == 0 and review['submitIndices'] == []
    assert review['capacity']['activeJobs'] == [
        {'jobID':unrelated.id,'kind':unrelated.kind,'status':'running','belongsToThisRound':False}]
    assert review['capacity']['occupiedSlots'] == review['capacity']['limit'] == 1
    assert 'Other scientific jobs on this controller count too' in review['capacity']['summary']
    assert jlens_rounds.action(parent.id, 'plan', None, False, jobs, profile)['planSHA256'] == review['planSHA256']
    unrelated.status = 'succeeded'; jobs.store.update(unrelated)
    fresh = jlens_rounds.action(parent.id, 'plan', None, False, jobs, profile)
    assert fresh['capacity']['activeJobs'] == [] and fresh['submitIndices'] == [0]
    with pytest.raises(archives.Refusal, match='changed'):
        jlens_rounds.action(parent.id, 'submit', review['planSHA256'], True, jobs, profile)
