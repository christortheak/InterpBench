"""Client readiness and adapters to the pre-Python installer; no engine imports."""
import json
import os
from pathlib import Path
import subprocess
import sys
from .runtime_identity import source_sha256
from . import workspace_bootstrap
from .. import client_dependencies


class SetupRefusal(ValueError):
    repair_action = 'Review setup plan from the matching client release; approve setup apply or repair only after reviewing its actions.'


def inspect(root=None):
    dependencies = client_dependencies.versions()
    capabilities = client_dependencies.probe()
    basic_ready = capabilities['basicAuthoring']['ready']
    ready = all(value['ready'] for value in capabilities.values())
    unavailable = [value['label'] for value in capabilities.values() if not value['ready']]
    workspace = workspace_bootstrap.inspect(root) if root is not None else None
    authoring = bool(basic_ready and workspace and workspace['recognized'] and workspace['agentGuidePresent'])
    return {'changed': False, 'clientReady': ready, 'basicClientReady': basic_ready, 'authoringReady': authoring,
            'capabilities': capabilities, 'reason': None if ready else 'Client update needed for: ' + ', '.join(unavailable) + '.',
            'python': sys.version.split()[0], 'interpreter': sys.executable,
            'sourceSHA256': source_sha256(), 'dependencies': dependencies,
            'workspace': workspace, 'execution': {'state': 'notAssessed', 'requiredForAuthoring': False,
                'nextAction': 'When ready to execute, select a local runner or remote profile, check its connection and prepare the model through the existing execution plan.'},
            'repairAction': (client_dependencies.UPGRADE_REPAIR if not ready else 'Create or open a workspace, then read workspace handoff.' if not authoring else 'Choose a method with the researcher.'),
            'diagnostic': '\n'.join(value['diagnostic'] for value in capabilities.values() if value.get('diagnostic')) or None}


def release_directory(explicit=None):
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
    else:
        receipt = Path(sys.prefix) / '.steerlab-client.json'
        if not receipt.is_file():
            raise SetupRefusal('This interpreter has no managed release receipt; pass --release <extracted-client-release>.')
        path = Path(json.loads(receipt.read_bytes())['release'])
    if not path.is_dir() or not (path / 'install-client.sh').is_file():
        raise SetupRefusal('The complete client release is missing.')
    return path


def provision(operation, *, release=None, runtime=None, expected=None, approved=False):
    if operation not in ('plan', 'apply', 'repair'):
        raise SetupRefusal('Unknown client setup operation.')
    command = ['/bin/sh', str(release_directory(release) / 'install-client.sh'), 'install' if operation == 'apply' else operation]
    if runtime is not None: command += ['--runtime', str(runtime)]
    if expected is not None: command += ['--expect', expected]
    if approved: command += ['--yes']
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    try:
        response = json.loads(result.stdout)
    except ValueError as exc:
        raise SetupRefusal('Setup returned no structured result; inspect its stderr diagnostics.') from exc
    if result.returncode or not response.get('ok'):
        failure = SetupRefusal(response.get('reason', 'Client setup did not finish.'))
        failure.repair_action = response.get('repairAction', failure.repair_action)
        raise failure
    return response


def start(directory, *, create=False):
    """One agent-friendly first-run operation after installing the client."""
    readiness = inspect()
    if not readiness['basicClientReady']:
        raise SetupRefusal('Repair the client imports before creating a workspace.')
    root = Path(directory).expanduser().absolute()
    changed = False
    if create:
        workspace_bootstrap.initialize(root)
        changed = True
    elif not root.is_dir():
        failure = SetupRefusal('The workspace is missing; use --create to explicitly create a new workspace.')
        failure.repair_action = 'Choose an existing workspace or run setup start <directory> --create --json.'
        raise failure
    report = inspect(root)
    handoff = workspace_bootstrap.handoff(root)
    return {'changed': changed, 'readiness': report, 'handoff': handoff,
            'nextAction': 'Give this handoff and your research question to an agent, or open the same workspace in the app.'}
