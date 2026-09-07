"""Client readiness and adapters to the pre-Python installer; no engine imports."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from .runtime_identity import source_sha256
from . import workspace_bootstrap


class SetupRefusal(ValueError):
    repair_action = 'Review setup plan from the matching client release; approve setup apply or repair only after reviewing its actions.'


def inspect(root=None):
    dependencies = {}
    for name in ('numpy', 'safetensors', 'httpx'):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    probe = subprocess.run([sys.executable, '-I', '-c', 'import numpy, safetensors, httpx'], capture_output=True, text=True, timeout=30)
    ready = probe.returncode == 0
    workspace = workspace_bootstrap.inspect(root) if root is not None else None
    authoring = bool(ready and workspace and workspace['recognized'] and workspace['agentGuidePresent'])
    return {'changed': False, 'clientReady': ready, 'authoringReady': authoring,
            'python': sys.version.split()[0], 'interpreter': sys.executable,
            'sourceSHA256': source_sha256(), 'dependencies': dependencies,
            'workspace': workspace, 'execution': {'state': 'notAssessed', 'requiredForAuthoring': False,
                'nextAction': 'When ready to execute, select a local runner or remote profile, check its connection and prepare the model through the existing execution plan.'},
            'repairAction': ('Create or open a workspace, then read workspace handoff.' if ready and not authoring else 'Choose a method with the researcher.' if authoring else 'Use the matching release installer to repair the lightweight client environment.'),
            'diagnostic': probe.stderr.strip() if not ready else None}


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
