"""Standalone execution and controller recovery over existing durable owners."""
from fastapi import APIRouter, HTTPException
from . import scientific_execution
from .profile import ServerProfile


def build_scientific_execution_router(state):
    router = APIRouter()

    def refuse(exc, *, ownership=False):
        return HTTPException(status_code=409, detail={
            'code': 'jobRecoveryRefused' if ownership else getattr(exc, 'code', 'scientificExecutionRefused'), 'reason': str(exc),
            'repairAction': 'Inspect recovery on this endpoint, establish owner exit, and use its current reviewToken with an explicit reason.' if ownership else getattr(exc, 'repair_action', scientific_execution.ScientificRefusal.repair_action)})

    def job_owner():
        if state.jobs is None:
            raise ValueError('This worker has no durable job queue; use its controller.')
        return state.jobs

    @router.post('/api/science/plan')
    def plan(body: dict):
        try:
            return scientific_execution.plan(body, ServerProfile.from_env())
        except (ValueError, OSError) as exc:
            raise refuse(exc) from exc

    @router.post('/api/science/submit')
    def submit(body: dict):
        try:
            if set(body) != {'request', 'planSHA256'} or not isinstance(body['planSHA256'], str):
                raise ValueError('Supply exactly request and planSHA256 from the reviewed plan.')
            if state.jobs is None:
                raise ValueError('Submit on the controller or workstation that owns the job queue.')
            return scientific_execution.submit(body['request'], body['planSHA256'],
                profile=ServerProfile.from_env(), jobs=state.jobs, registry=state.registry)
        except (ValueError, OSError) as exc:
            raise refuse(exc) from exc

    @router.get('/api/jobs/{job_id}/recovery')
    def recovery(job_id: str):
        try:
            return job_owner().store.recovery_report(job_id)
        except ValueError as exc:
            raise refuse(exc, ownership=True) from exc

    @router.post('/api/jobs/{job_id}/recover')
    def recover(job_id: str, body: dict):
        try:
            if (set(body) != {'reviewToken', 'reason', 'confirmOwnerExited'}
                    or body['confirmOwnerExited'] is not True
                    or not isinstance(body['reviewToken'], str)
                    or not isinstance(body['reason'], str) or not body['reason'].strip()):
                raise ValueError('Recovery requires the current reviewToken, a reason, and confirmOwnerExited: true.')
            if not job_owner().recover_orphan(job_id, body['reviewToken'], body['reason']):
                raise ValueError('The job is not eligible for recovery; inspect its current owner and status.')
            return {'jobId': job_id, 'recovered': True}
        except ValueError as exc:
            raise refuse(exc, ownership=True) from exc

    return router
