"""Closed execution-copy, evidence and cleanup adapters."""
from fastapi import APIRouter, HTTPException
from . import diagnostic_transport as transport, diagnostic_cleanup as cleanup
from .profile import ServerProfile
from .workspace_lock import submitting
from ..experiment import diagnostic_archives as archives


def build_router(state):
    router = APIRouter()
    def fields(body, wanted):
        if not isinstance(body, dict) or set(body) != set(wanted):
            raise archives.Refusal('Supply exactly ' + ', '.join(wanted))
    def perform(callback):
        try:
            if state.jobs is None: raise archives.Refusal('Use the controller that owns durable job records.')
            return callback()
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(409, detail={'code': 'diagnosticTransportRefused', 'reason': str(exc), 'repairAction': archives.Refusal.repair_action}) from exc
    @router.post('/api/science/workspace/{action}')
    def workspace(action: str, body: dict):
        from pathlib import Path
        from ..client.diagnostic_commands import workspace_action
        try:
            with submitting():
                root = ServerProfile.from_env().root
                if not isinstance(body.get('workspaceRoot'), str) or Path(body['workspaceRoot']).resolve() != Path(root).resolve():
                    raise archives.Refusal('This workbench is serving another workspace.')
                return workspace_action(action, body)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(409, detail={'code': 'diagnosticTransportRefused', 'reason': str(exc), 'repairAction': archives.Refusal.repair_action}) from exc

    @router.post('/api/science/stage')
    def stage(body: dict):
        def work():
            fields(body, ['bundlePath', 'bundleSHA256'])
            with submitting(): return transport.stage(body['bundlePath'], body['bundleSHA256'], ServerProfile.from_env())
        return perform(work)
    @router.post('/api/science/jobs/{job_id}/export')
    def export(job_id: str):
        def work():
            with submitting(): return transport.export(job_id, state.jobs, ServerProfile.from_env())
        return perform(work)
    @router.post('/api/science/jobs/{job_id}/cleanup-plan')
    def plan(job_id: str, body: dict):
        def work():
            fields(body, ['custody'])
            return cleanup.plan(job_id, body['custody'], state.jobs, ServerProfile.from_env())
        return perform(work)
    @router.post('/api/science/jobs/{job_id}/cleanup-apply')
    def apply(job_id: str, body: dict):
        def work():
            fields(body, ['custody', 'planSHA256', 'confirmRemoval'])
            return cleanup.apply(job_id, body['custody'], body['planSHA256'], state.jobs, ServerProfile.from_env(), confirmed=body['confirmRemoval'])
        return perform(work)
    @router.post('/api/science/campaign/{job_id}/{action}')
    def campaign(job_id: str, action: str, body: dict):
        from . import managed_campaign
        def work():
            fields(body, [] if action in ('status', 'plan') else ['planSHA256', 'confirmAction'])
            return managed_campaign.action(job_id, action, body.get('planSHA256'), body.get('confirmAction'), state.jobs, ServerProfile.from_env())
        return perform(work)
    return router
