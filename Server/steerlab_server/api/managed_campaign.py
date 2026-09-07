"""Controller-facing review and lifecycle for an isolated materialized campaign."""
import json
import os
from pathlib import Path
import subprocess
import sys
from ..experiment import diagnostic_archives as archives
from . import diagnostic_transport, workspace_lock


def context(job_id, jobs, profile):
    job = jobs.get(job_id)
    if job is None or job.kind != 'science:optvec-campaign' or job.status != 'succeeded':
        raise archives.Refusal('Choose a successfully materialized campaign job.')
    result = job.result or {}; plan = result.get('scientificPlan', {})
    if not plan.get('inputBundleSHA256'): raise archives.Refusal('Managed campaigns require an isolated staged input bundle.')
    _, root = diagnostic_transport.resolve(plan['inputBundleSHA256'], profile)
    directory = Path(result.get('campaignDirectory', ''))
    if plan.get('root') != str(root) or directory.parent != root / 'runs':
        raise archives.Refusal('Campaign belongs to another execution root.')
    directory = archives.ordinary(root, directory.relative_to(root).as_posix())
    return directory, root, result['managedCampaign']


def action(job_id, action, expected, confirmed, jobs, profile):
    if action not in ('status', 'plan', 'submit', 'cancel'): raise archives.Refusal('Unknown campaign action.')
    if action in ('submit', 'cancel') and confirmed is not True: raise archives.Refusal('Confirm the reviewed campaign action explicitly.')
    with workspace_lock.submitting():
        directory, root, pin = context(job_id, jobs, profile)
        environment = {**os.environ, 'STEERLAB_ROOT': str(root), 'STEERLAB_RUN_ROOT': str(root / 'runs'),
                       'PYTHONPATH': str(Path(__file__).resolve().parents[2])}
        try:
            result = subprocess.run([sys.executable, '-m', 'steerlab_server.api.managed_campaign_engine'],
                input=archives.encoded({'directory': str(directory), 'pin': pin, 'action': action, 'expected': expected}),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=root, env=environment, timeout=300)
        except subprocess.TimeoutExpired:
            raise archives.Refusal('Campaign action timed out with an uncertain outcome; inspect status before another explicit top-up.') from None
        try: response = json.loads(result.stdout)
        except ValueError: raise archives.Refusal('Campaign returned no result; inspect status before retrying.') from None
        if result.returncode or not response.get('ok'): raise archives.Refusal(response.get('reason', 'Campaign action refused.'))
        return {'jobID': job_id, **response['result']}


def require_complete(job_id, jobs, profile):
    report = action(job_id, 'status', None, False, jobs, profile)['status']
    if not report['cellCount'] or report['totals']['completed'] != report['cellCount']:
        raise archives.Refusal('Campaign evidence export requires every cell to complete; retain partial results on the runner.')
    from .executors import SlurmExecutor
    from .job_ownership import TERMINAL_ALLOCATION_STATES
    directory, _, _ = context(job_id, jobs, profile)
    for cell in report['cells']:
        identity = cell['jobID']
        if not identity:
            # A worker may complete after the submit response was lost. Check
            # its exact name before treating the absent local ID as termination.
            bundle = json.loads((directory/'cells'/cell['cellID']/'managed-job/bundle.json').read_bytes())
            try: identity = SlurmExecutor(profile).find_job_by_name(bundle['bundle']['resources']['job_name'])
            except RuntimeError as exc: raise archives.Refusal(str(exc)) from exc
        if identity:
            state, ok = SlurmExecutor(profile).poll_state_detailed(identity)
            if not ok or (state and state not in TERMINAL_ALLOCATION_STATES):
                raise archives.Refusal('A campaign cell is still active or scheduler state is uncertain; retain outputs until termination is established.')
