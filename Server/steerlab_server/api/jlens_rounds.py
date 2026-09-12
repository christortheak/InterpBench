"""Reviewed queue top-ups for fitting rounds, using ordinary durable science jobs."""
import json
from pathlib import Path
from ..experiment import diagnostic_archives as archives
from . import diagnostic_transport, scientific_execution, workspace_lock

TERMINAL={'succeeded','failed','cancelled'}


def context(job_id,jobs,profile):
    parent=jobs.get(job_id)
    if parent is None or parent.kind!='science:jlens-fit-round' or parent.status!='succeeded':
        raise archives.Refusal('Choose a successfully materialized fitting-round job.')
    result=parent.result or {};source=result.get('scientificPlan',{})
    capsule=source.get('inputBundleSHA256')
    if not capsule:raise archives.Refusal('Remote fitting rounds need isolated staged inputs.')
    _,root=diagnostic_transport.resolve(capsule,profile)
    directory=Path(result.get('runDirectory',''))
    if directory.parent!=root/'runs' or source.get('root')!=str(root):raise archives.Refusal('Round output belongs to another execution root.')
    packet=archives.ordinary(root,(directory/'round-plan.json').relative_to(root).as_posix())
    if archives.file_hash(packet)!=result.get('roundPlanSHA256'):raise archives.Refusal('Round plan changed after materialization.')
    plan=json.loads(packet.read_bytes())
    if archives.digest({k:v for k,v in plan.items() if k!='planSHA256'})!=plan.get('planSHA256'):raise archives.Refusal('Round plan hash differs.')
    return root,capsule,plan


def state_file(job_id,profile):
    if len(archives.parts(job_id))!=1:raise archives.Refusal('Choose a durable job ID.')
    directory=archives.ordinary(Path(profile.metadata_root),'.steerlab/jlens-round-state',missing=True)
    return archives.ordinary(directory,job_id+'.json',missing=True)


def write_state(path,state):
    path.parent.mkdir(parents=True,exist_ok=True)
    scientific_execution.write_json(path,state)


def matching_jobs(jobs,parent,index):
    return [j for j in jobs.list() if (j.result or {}).get('scientificPlan',{}).get('roundSubmission')=={'parentJobID':parent,'shardIndex':index}]


def capacity_review(limit, active, rows, parent):
    uncertain = [row['index'] for row in rows if row['status'] == 'uncertain']
    occupied = len(active) + len(uncertain)
    return {
        'limit': limit,
        'occupiedSlots': occupied,
        'activeJobs': [
            {'jobID': job.id, 'kind': job.kind, 'status': job.status,
             'belongsToThisRound': ((job.result or {}).get('scientificPlan', {}).get('roundSubmission') or {}).get('parentJobID') == parent}
            for job in sorted(active, key=lambda job: job.id)
        ],
        'uncertainShardIndices': uncertain,
        'summary': f'{len(active)} active scientific jobs and {len(uncertain)} uncertain shard submissions '
                   f'occupy {occupied} slots against this round’s limit of {limit}; {max(0, limit-occupied)} slots are available. '
                   'Other scientific jobs on this controller count too. '
                   + ('Wait for jobs to finish, or reconcile uncertain submissions before reviewing another top-up.'
                      if occupied >= limit else 'Review pending shards before submitting a top-up; uncertain submissions are never retried automatically.'),
    }


def action(job_id,action,expected,confirmed,jobs,profile):
    if action not in ('status','plan','submit','cancel','merge-plan','merge-submit'):raise archives.Refusal('Unknown fitting-round action.')
    mutation=action in ('submit','cancel','merge-submit')
    if mutation and confirmed is not True:raise archives.Refusal('Confirm the reviewed fitting-round action.')
    with workspace_lock.submitting():
        root,capsule,round_plan=context(job_id,jobs,profile)
        path=state_file(job_id,profile)
        state=json.loads(path.read_bytes()) if path.exists() else {'attempted':[]}
        if (not isinstance(state,dict) or not isinstance(state.get('attempted'),list)
                or any(type(i) is not int or not 0 <= i < len(round_plan['shards']) for i in state['attempted'])
                or len(set(state['attempted'])) != len(state['attempted'])
                or not isinstance(state.get('mergeAttempts',[]),list)
                or any(not isinstance(value,str) for value in state.get('mergeAttempts',[]))):
            raise archives.Refusal('Round bookkeeping is malformed. Inspect durable jobs before repairing the controller state.')
        rows=[]
        for index,request in enumerate(round_plan['shards']):
            matches=matching_jobs(jobs,job_id,index)
            if len(matches)>1:raise archives.Refusal('More than one allocation matches a shard; inspect durable jobs before any top-up.')
            job=matches[0] if matches else None
            rows.append({'index':index,'jobID':job.id if job else None,
                         'status':job.status if job else ('uncertain' if index in state['attempted'] else 'pending'),
                         'runDirectory':(job.result or {}).get('runDirectory') if job else None})
        active=[j for j in jobs.list() if j.status not in TERMINAL and j.kind.startswith('science:') and j.id!=job_id]
        uncertain=sum(row['status']=='uncertain' for row in rows)
        slots=max(0,round_plan['config']['maxConcurrent']-len(active)-uncertain)
        pending=[row['index'] for row in rows if row['status']=='pending'][:slots]
        selected={}
        for index in pending:
            selected[str(index)]=scientific_execution.plan(round_plan['shards'][index],profile,
                execution_capsule=capsule,round_submission={'parentJobID':job_id,'shardIndex':index})
        plan={'jobID':job_id,'roundPlanSHA256':round_plan['planSHA256'],'state':state,'shards':rows,
              'capacity':capacity_review(round_plan['config']['maxConcurrent'],active,rows,job_id),
              'availableSlots':slots,'submitIndices':pending,'childPlans':selected,'changed':False,
              'scope':'Concurrency counts active scientific jobs on this controller. Set the declared cap within site policy; scheduler policy remains authoritative. Uncertain submissions reserve capacity until reconciled; top-ups never retry them.'}
        if action in ('merge-plan','merge-submit'):
            fits=[]
            for row in rows:
                if row['status']=='succeeded' and row['runDirectory']:
                    fits.append(Path(row['runDirectory']).relative_to(root).as_posix())
            if not fits:raise archives.Refusal('No completed shard is available to merge. Wait for completion or select completed continuation runs explicitly.')
            request={'operation':'jlens-fit-merge','parameters':{'config':{'fits':fits,'allowPartial':True}}}
            plan['mergeRequest']=request
            plan['mergePlan']=scientific_execution.plan(request,profile,execution_capsule=capsule)
        # Capacity detail explains the reviewed slots; unrelated status-only
        # changes must not introduce another submission precondition.
        plan['planSHA256']=archives.digest({key:value for key,value in plan.items() if key!='capacity'})
        if not mutation:return plan
        if expected!=plan['planSHA256']:raise archives.Refusal('Round state, inputs, or queue capacity changed. Review a fresh plan.')
        if action=='cancel':
            results={row['jobID']:jobs.cancel(row['jobID']) for row in rows if row['jobID'] and row['status'] not in TERMINAL}
            return {'changed':any(results.values()),'cancellationRequested':results,'nextAction':'Inspect job status to establish termination; no retry or cleanup is automatic.'}
        if action=='merge-submit':
            merge_hash=archives.digest(plan['mergeRequest'])
            if merge_hash in state.get('mergeAttempts',[]):raise archives.Refusal('This round already attempted a merge; inspect its jobs before authoring another explicit merge request.')
            state.setdefault('mergeAttempts',[]).append(merge_hash);write_state(path,state)
            result=scientific_execution.submit(plan['mergeRequest'],plan['mergePlan']['planSHA256'],profile=profile,jobs=jobs,execution_capsule=capsule)
            return {'changed':True,'merge':result}
        submitted=[]
        for index in pending:
            state['attempted'].append(index);write_state(path,state)
            result=scientific_execution.submit(round_plan['shards'][index],selected[str(index)]['planSHA256'],
                profile=profile,jobs=jobs,execution_capsule=capsule,round_submission={'parentJobID':job_id,'shardIndex':index})
            submitted.append(result)
            if result['status']=='parked':break
        return {'changed':bool(submitted),'submissions':submitted,'nextAction':'Inspect durable shard jobs, then review another top-up when slots become free. Continue a stopped shard from its own checkpoint in a new reviewed run.'}
