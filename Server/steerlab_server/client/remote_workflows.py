"""Portable adapters for reviewed diagnostic submission and explicit recovery."""
import json
from pathlib import Path
from ..cli_envelope import CLIResult


def run(client, invocation, common):
    from ..client_cli import ClientRefusal
    verb = invocation.spec.verb
    if any(flag not in invocation.flags for flag in invocation.spec.required_flags) or any(len(values) != 1 for values in invocation.flags.values()):
        raise ClientRefusal(code='usage', reason='Supply every required flag exactly once, including the explicit recovery attestation.', repair_action=f'steerlab runner {verb} --help')
    if len(invocation.positionals) != (0 if verb == 'reconcile' else 1):
        raise ClientRefusal(code='usage', reason='Supply exactly one request file or job ID.', repair_action=f'steerlab runner {verb} --help')
    value = invocation.positionals[0] if invocation.positionals else None
    if verb.startswith('science-'):
        try:
            request = json.loads(Path(value).read_text())
        except (OSError, ValueError) as exc:
            raise ClientRefusal(code='usage', reason=f'Cannot read diagnostic request: {exc}', repair_action='Supply a JSON document with operation and parameters.') from exc
        gpu_type, walltime = invocation.one('--gpu-type'), invocation.one('--walltime')
        document = (client.scientific_plan(request, gpu_type, walltime) if verb == 'science-plan'
                    else client.scientific_submit(request, invocation.one('--plan-sha256'), gpu_type, walltime))
    elif verb == 'reconcile':
        document = client.reconcile_jobs()
    elif verb == 'resubmit':
        document = client.resubmit_job(value, invocation.one('--walltime'))
    elif verb == 'recovery':
        document = client.job_recovery(value)
    else:
        document = client.recover_job(value, invocation.one('--review-token'), invocation.one('--reason'))
    print(json.dumps(document, indent=2, sort_keys=True))
    if verb in {'science-submit', 'resubmit'} and isinstance(document, dict) \
            and not (verb == 'science-submit' and document.get('status') == 'parked'):
        # A new job record on this runner (a diagnostic, or a resumed run's
        # continuation): record where it went in the named workspace, so the
        # Mac app can act on it without a reconnect. Best effort.
        import sys
        from . import job_origins
        original = (job_origins.origins_for(invocation.workspace_root, value)
                    if verb == 'resubmit' and getattr(invocation, 'workspace_root', None) else [])
        identity = job_origins.server_identity(client.base_url)
        earlier = next((row for row in original if row.get('serverIdentity') == identity), {})
        job_origins.record_quietly(
            getattr(invocation, 'workspace_root', None), job_id=document.get('jobId'),
            endpoint=client.base_url, warn=sys.stderr.write,
            experiment=earlier.get('experiment'), verb=earlier.get('verb'), operation=verb)
    if verb == 'science-submit' and document.get('status') == 'parked':
        return CLIResult(state='failed', code='schedulerSubmissionUncertain', changed=True,
            message='Scheduler reply was uncertain; the durable submission record is retained.',
            repair_action='Inspect this endpoint and schedulerSubmissionName before any retry; reconcile child records when available.',
            payload={**common, 'response': document})
    message = 'Remote workflow request completed; retain this endpoint and job ID for subsequent observation.'
    if verb == 'resubmit' and isinstance(document, dict) and document.get('resumedAfterCancel') and document.get('message'):
        # A cancelled run that a person resumed: the engine's own sentence says
        # what was kept and which job now carries the run.
        message = str(document['message'])
    return CLIResult(message=message,
                     changed=verb not in {'science-plan', 'recovery'}, payload={**common, 'response': document})
