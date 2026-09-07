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
        document = client.scientific_plan(request) if verb == 'science-plan' else client.scientific_submit(request, invocation.one('--plan-sha256'))
    elif verb == 'reconcile':
        document = client.reconcile_jobs()
    elif verb == 'resubmit':
        document = client.resubmit_job(value, invocation.one('--walltime'))
    elif verb == 'recovery':
        document = client.job_recovery(value)
    else:
        document = client.recover_job(value, invocation.one('--review-token'), invocation.one('--reason'))
    print(json.dumps(document, indent=2, sort_keys=True))
    if verb == 'science-submit' and document.get('status') == 'parked':
        return CLIResult(state='failed', code='schedulerSubmissionUncertain', changed=True,
            message='Scheduler reply was uncertain; the durable submission record is retained.',
            repair_action='Inspect this endpoint and schedulerSubmissionName before any retry; reconcile child records when available.',
            payload={**common, 'response': document})
    return CLIResult(message='Remote workflow request completed; retain this endpoint and job ID for subsequent observation.',
                     changed=verb not in {'science-plan', 'recovery'}, payload={**common, 'response': document})
