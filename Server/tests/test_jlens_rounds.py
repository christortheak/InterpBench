import json
from pathlib import Path
import pytest
from steerlab_server.api import jlens_rounds, scientific_execution as engine, diagnostic_transport
from steerlab_server.api.jobs import JobManager, DurableJobStore
from steerlab_server.experiment import jlens_round, diagnostic_inputs, diagnostic_archives as archives
from test_jlens_fit import fitting
from test_scientific_execution import setup


@pytest.fixture
def round_job(fitting,setup,request):
    root,cfg=fitting;_,_,profile=setup
    path=root/'fit.json';path.write_text(json.dumps({'operation':'jlens-fit','parameters':{'config':cfg}}))
    config=jlens_round.RoundConfig.from_dict(dict(modelID=cfg['modelID'],revision=cfg['revision'],fittingRequest={'path':'fit.json','sha256':archives.file_hash(path)},shards=2,maxConcurrent=getattr(request, 'param', 1)))
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
    unrelated = jobs.record_external('science:jlens-fit-assess', status='submitted', executor='slurm',
                                     result={'scientificPlan':{'roundSubmission':None}})
    jobs.record_external('science:jlens-fit', status='succeeded', executor='slurm')
    jobs.record_external('chat', status='running', executor='local')
    review = jlens_rounds.action(parent.id, 'plan', None, False, jobs, profile)
    assert review['availableSlots'] == 0 and review['submitIndices'] == []
    assert review['capacity']['activeJobs'] == [
        {'jobID':unrelated.id,'kind':unrelated.kind,'status':'submitted','belongsToThisRound':False}]
    assert review['capacity']['occupiedSlots'] == review['capacity']['limit'] == 1
    assert 'Other scientific jobs on this controller count too' in review['capacity']['summary']
    assert jlens_rounds.action(parent.id, 'plan', None, False, jobs, profile)['planSHA256'] == review['planSHA256']
    unrelated.status = 'running'; jobs.store.update(unrelated)
    status_only = jlens_rounds.action(parent.id, 'plan', None, False, jobs, profile)
    assert status_only['capacity']['activeJobs'][0]['status'] == 'running'
    assert status_only['planSHA256'] == review['planSHA256']
    unrelated.status = 'succeeded'; jobs.store.update(unrelated)
    fresh = jlens_rounds.action(parent.id, 'plan', None, False, jobs, profile)
    assert fresh['capacity']['activeJobs'] == [] and fresh['submitIndices'] == [0]
    with pytest.raises(archives.Refusal, match='changed'):
        jlens_rounds.action(parent.id, 'submit', review['planSHA256'], True, jobs, profile)


# --- Bounded hash reuse inside one fitting-round action --------------------
# One action shares one input_hashes scope across context(), the round plan,
# and the re-plan inside scientific_execution.submit. These tests prove the
# three guarantees: one read per unchanged input within the action, refusal on
# mutation before any attempt is recorded or anything queues, and a fresh scope
# for the queued child and for the next request.

@pytest.fixture
def completed_shard(round_job):
    """A round whose shard 0 finished: a real (tiny) fit under the capsule root."""
    from steerlab_server.experiment import jlens_fit, jlens_fit_execution
    parent,jobs,profile=round_job
    root=Path(parent.result['scientificPlan']['root'])
    plan=json.loads((Path(parent.result['runDirectory'])/'round-plan.json').read_bytes())
    fitted=jlens_fit_execution.execute(jlens_fit.FitConfig.from_dict(plan['shards'][0]['parameters']['config']),root=str(root),log=lambda _:None)
    jobs.record_external('science:jlens-fit',status='succeeded',executor='slurm',
        result={'scientificPlan':{'roundSubmission':{'parentJobID':parent.id,'shardIndex':0}},'runDirectory':fitted['runDirectory']})
    return parent,jobs,profile,root,Path(fitted['runDirectory'])


@pytest.fixture
def byte_reads(monkeypatch):
    """Count full reads of each file by the one hashing primitive input_hashes uses."""
    import hashlib
    from collections import Counter
    counts=Counter();original=hashlib.file_digest
    def counting(stream,*args,**kwargs):
        counts[str(stream.name)]+=1
        return original(stream,*args,**kwargs)
    monkeypatch.setattr(hashlib,'file_digest',counting)
    return counts


def reads(counts,name):
    """Read counts of every hashed file with this basename (a staged corpus and a fit's own copy are distinct files)."""
    found={path:count for path,count in counts.items() if Path(path).name==name}
    assert found,(name,dict(counts))
    return set(found.values())


def stub_local_worker(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(engine.LocalExecutor,'run',lambda self,command,**kwargs:SimpleNamespace(returncode=0,stderr=''))


def settle(jobs,job_id):
    import time
    deadline=time.monotonic()+5
    while jobs.get(job_id).status not in jlens_rounds.TERMINAL and time.monotonic()<deadline:time.sleep(.01)
    return jobs.get(job_id)


def test_merge_submit_reads_each_input_once_across_plan_and_submit_replan(completed_shard,byte_reads,monkeypatch):
    from steerlab_server.experiment import input_hashes
    parent,jobs,profile,root,fit=completed_shard
    stub_local_worker(monkeypatch)
    scopes=[];original=engine.plan
    def observed_plan(request,profile,**kwargs):
        scopes.append((request.get('operation'),input_hashes._active.get() is not None))
        return original(request,profile,**kwargs)
    monkeypatch.setattr(engine,'plan',observed_plan)
    review=jlens_rounds.action(parent.id,'merge-plan',None,False,jobs,profile)
    assert review['mergeRequest']['parameters']['config']['fits']==[fit.relative_to(root).as_posix()]
    byte_reads.clear();scopes.clear()
    result=jlens_rounds.action(parent.id,'merge-submit',review['planSHA256'],True,jobs,profile)
    # The merge closure was planned twice inside the action (review, then the
    # submit re-plan), both inside the action's scope ...
    assert [s for s in scopes if s[0]=='jlens-fit-merge']==[('jlens-fit-merge',True)]*2
    # ... yet every input, including the fitted tensor and the staged archive,
    # was read once.
    assert reads(byte_reads,'jacobians.safetensors')=={1} and reads(byte_reads,'input.tar.gz')=={1}
    assert max(byte_reads.values())==1, dict(byte_reads)
    assert input_hashes._active.get() is None
    assert settle(jobs,result['merge']['jobId']).status=='succeeded'
    assert json.loads(jlens_rounds.state_file(parent.id,profile).read_bytes())['mergeAttempts']==[archives.digest(review['mergeRequest'])]


def test_merge_submit_refuses_a_changed_input_before_recording_an_attempt(completed_shard,monkeypatch):
    parent,jobs,profile,root,fit=completed_shard
    review=jlens_rounds.action(parent.id,'merge-plan',None,False,jobs,profile)
    tensors=fit/'jacobians.safetensors';original=archives.digest;mutated=[]
    def mutate_after_review(value):
        # Runs when the action seals its plan: after the merge closure was
        # hashed, before the mergeAttempts record and the submission.
        if isinstance(value,dict) and 'mergePlan' in value and not mutated:
            tensors.write_bytes(tensors.read_bytes()+b'\0');mutated.append(True)
        return original(value)
    monkeypatch.setattr(archives,'digest',mutate_after_review)
    def never(*args,**kwargs):raise AssertionError('nothing may be submitted')
    monkeypatch.setattr(engine,'submit',never)
    before=[job.id for job in jobs.list()]
    with pytest.raises(archives.Refusal,match='changed before review finished.*Nothing was recorded'):
        jlens_rounds.action(parent.id,'merge-submit',review['planSHA256'],True,jobs,profile)
    assert mutated and [job.id for job in jobs.list()]==before
    state=jlens_rounds.state_file(parent.id,profile)
    assert not state.exists() or json.loads(state.read_bytes()).get('mergeAttempts',[])==[]
    # A fresh review sees the current bytes and carries a new plan hash.
    assert jlens_rounds.action(parent.id,'merge-plan',None,False,jobs,profile)['planSHA256']!=review['planSHA256']


def test_merge_submit_replan_refuses_a_change_after_the_intention_record(completed_shard,monkeypatch):
    # The residual window: a change after mergeAttempts is recorded but before
    # the submit re-plan. The re-plan rechecks at use and nothing queues; the
    # attempt stays recorded, as for any post-record failure.
    parent,jobs,profile,root,fit=completed_shard
    review=jlens_rounds.action(parent.id,'merge-plan',None,False,jobs,profile)
    tensors=fit/'jacobians.safetensors';original=engine.submit
    def mutate_then_submit(request,expected,**kwargs):
        tensors.write_bytes(tensors.read_bytes()+b'\0')
        return original(request,expected,**kwargs)
    monkeypatch.setattr(engine,'submit',mutate_then_submit)
    before=[job.id for job in jobs.list()]
    with pytest.raises(ValueError,match='changed during review'):
        jlens_rounds.action(parent.id,'merge-submit',review['planSHA256'],True,jobs,profile)
    assert [job.id for job in jobs.list()]==before
    assert json.loads(jlens_rounds.state_file(parent.id,profile).read_bytes())['mergeAttempts']==[archives.digest(review['mergeRequest'])]
    with pytest.raises(archives.Refusal,match='already attempted'):
        jlens_rounds.action(parent.id,'merge-submit',jlens_rounds.action(parent.id,'merge-plan',None,False,jobs,profile)['planSHA256'],True,jobs,profile)


def test_queued_child_verifies_in_its_own_scope(completed_shard,byte_reads,monkeypatch):
    from steerlab_server.experiment import input_hashes
    parent,jobs,profile,root,fit=completed_shard
    stub_local_worker(monkeypatch)
    review=jlens_rounds.action(parent.id,'merge-plan',None,False,jobs,profile)
    result=jlens_rounds.action(parent.id,'merge-submit',review['planSHA256'],True,jobs,profile)
    settle(jobs,result['merge']['jobId'])
    assert input_hashes._active.get() is None
    byte_reads.clear()
    packet=Path(result['merge']['submissionDirectory'])/'request.json'
    record=Path(result['merge']['recordsDirectory'])/'child.json'
    reviewed=json.loads(packet.read_bytes())
    # The child's verification (capsule closure, then input_plan) reads the
    # bytes again: nothing from the request's scope is served to it. Its own
    # operations each open their own scope; deduplicating across them is not
    # this change's business.
    assert engine.execute_packet(str(packet),'child',str(record),reviewed['planSHA256'])==0
    assert min(reads(byte_reads,'jacobians.safetensors'))>=1 and min(reads(byte_reads,'input.tar.gz'))>=1
    assert json.loads(record.read_bytes())['status']=='succeeded'


def test_sequential_requests_do_not_share_digests(completed_shard,byte_reads):
    from steerlab_server.experiment import input_hashes
    parent,jobs,profile,root,fit=completed_shard
    first=jlens_rounds.action(parent.id,'merge-plan',None,False,jobs,profile)
    after_first=dict(byte_reads)
    assert reads(after_first,'jacobians.safetensors')=={1} and input_hashes._active.get() is None
    second=jlens_rounds.action(parent.id,'merge-plan',None,False,jobs,profile)
    assert second['planSHA256']==first['planSHA256']
    assert reads(byte_reads,'jacobians.safetensors')=={2} and reads(byte_reads,'input.tar.gz')=={2}
    assert all(byte_reads[path]==2*count for path,count in after_first.items()), dict(byte_reads)


@pytest.mark.parametrize('round_job',[2],indirect=True)
def test_shard_submit_reads_each_input_once_and_rechecks_before_each_attempt(round_job,byte_reads,monkeypatch):
    parent,jobs,profile=round_job
    stub_local_worker(monkeypatch)
    review=jlens_rounds.action(parent.id,'plan',None,False,jobs,profile)
    assert review['submitIndices']==[0,1]
    byte_reads.clear()
    result=jlens_rounds.action(parent.id,'submit',review['planSHA256'],True,jobs,profile)
    assert len(result['submissions'])==2
    assert reads(byte_reads,'corpus.jsonl')=={1} and reads(byte_reads,'input.tar.gz')=={1}
    assert max(byte_reads.values())==1, dict(byte_reads)
    for submission in result['submissions']:settle(jobs,submission['jobId'])


@pytest.mark.parametrize('round_job',[2],indirect=True)
def test_shard_submit_stops_before_recording_a_later_attempt_when_an_input_changes(round_job,monkeypatch):
    parent,jobs,profile=round_job
    root=Path(parent.result['scientificPlan']['root']);corpus=root/'corpus.jsonl'
    stub_local_worker(monkeypatch)
    original=engine.submit
    def submit_then_mutate(request,expected,**kwargs):
        result=original(request,expected,**kwargs)
        corpus.write_bytes(corpus.read_bytes()+b'\n')
        return result
    monkeypatch.setattr(engine,'submit',submit_then_mutate)
    review=jlens_rounds.action(parent.id,'plan',None,False,jobs,profile)
    with pytest.raises(archives.Refusal,match='changed before review finished'):
        jlens_rounds.action(parent.id,'submit',review['planSHA256'],True,jobs,profile)
    state=json.loads(jlens_rounds.state_file(parent.id,profile).read_bytes())
    assert state['attempted']==[0]
    children=[job for job in jobs.list() if job.kind=='science:jlens-fit'];assert len(children)==1
    settle(jobs,children[0].id)  # the local worker fills in result.scientificPlan when it starts
    assert [job.id for job in jlens_rounds.matching_jobs(jobs,parent.id,0)]==[children[0].id]
    assert jlens_rounds.matching_jobs(jobs,parent.id,1)==[]
    # The staged closure itself changed, so every later action refuses at admission.
    with pytest.raises(archives.Refusal,match='Staged input bytes changed'):
        jlens_rounds.action(parent.id,'status',None,False,jobs,profile)


def test_recheck_is_stat_only_and_scoped(tmp_path,byte_reads):
    from steerlab_server.experiment import input_hashes
    path=tmp_path/'input.bin';path.write_bytes(b'x'*1024)
    assert input_hashes.recheck()==0  # nothing reviewed outside a scope
    with input_hashes.session():
        first=input_hashes.file_hash(path)
        assert input_hashes.recheck()==1 and input_hashes.file_hash(path)==first
        assert reads(byte_reads,'input.bin')=={1}
        path.write_bytes(b'y'*1024)
        with pytest.raises(ValueError,match='changed before review finished'):input_hashes.recheck()
        with pytest.raises(ValueError,match='changed during review'):input_hashes.file_hash(path)
        # The scope is not poisoned for other inputs, and nothing was re-read.
        assert reads(byte_reads,'input.bin')=={1}
        input_hashes._active.get().clear()
        gone=tmp_path/'gone.bin';gone.write_bytes(b'z');input_hashes.file_hash(gone);gone.unlink()
        with pytest.raises(OSError):input_hashes.recheck()
        input_hashes._active.get().clear()
    assert input_hashes._active.get() is None and input_hashes.recheck()==0
