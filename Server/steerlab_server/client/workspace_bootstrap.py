"""Portable new-workspace publication and agent handoff; no engine imports."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from .. import __version__
from ..experiment import diagnostic_archives as archives
from ..experiment.manifest_errors import ExperimentStoreError

RESOURCES = Path(__file__).parent / 'resources'
SEED = Path(__file__).parents[1] / 'experiment/seed'
HEADER = '<!-- Written by SteerLab workspace seeding. SteerLab keeps this file current for you while this line\'s hash still matches the text under it, and never touches it once you edit that text. sha256:'


def manifest():
    return json.loads((RESOURCES / 'workspace.json').read_bytes())


def agent_contents():
    body = (RESOURCES / 'agent-guide.md').read_text()
    return HEADER + hashlib.sha256(body.encode()).hexdigest() + ' -->\n\n' + body


def refuse(reason):
    raise ExperimentStoreError(reason, gate='workspaceBootstrap', repair='Choose a new or empty workspace directory; keep existing studies and outputs in their current workspace.')


def initialize(directory, *, use_git=True, seed=SEED):
    requested = Path(directory).expanduser().absolute()
    if requested.is_symlink():
        refuse('Workspace creation refuses a symlink destination.')
    root = requested.parent.resolve() / requested.name
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        refuse('The destination is not an empty directory; nothing was replaced.')
    specification = manifest()
    # All assets must be present before any destination directories are made.
    for name in specification['seedFiles']:
        source = archives.ordinary(seed, name, missing=True)
        if not source.is_file():
            refuse('The installed workspace seed is incomplete: ' + name)
    root.parent.mkdir(parents=True, exist_ok=True)
    git_status = 'notRequested' if not use_git else 'unavailable'
    with tempfile.TemporaryDirectory(prefix='.steerlab-workspace-', dir=root.parent) as temp:
        staged = Path(temp) / 'workspace'
        staged.mkdir()
        for name in specification['directories']:
            (staged / name).mkdir(parents=True, exist_ok=True)
        for name in specification['seedFiles']:
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(seed) / name, target)
        marker = (RESOURCES / 'workspace-marker.md').read_text().replace('{{version}}', __version__).replace('{{createdAt}}', datetime.now(timezone.utc).isoformat())
        (staged / specification['markerFile']).write_text(marker)
        (staged / 'AGENTS.md').write_text(agent_contents())
        (staged / '.gitignore').write_text(specification['gitignore'])
        if use_git and shutil.which('git'):
            commands = [['init'], ['-c', 'user.name=SteerLab', '-c', 'user.email=steerlab@localhost', 'add', '-A', '.'],
                        ['-c', 'user.name=SteerLab', '-c', 'user.email=steerlab@localhost', 'commit', '-m', 'Create research workspace']]
            git_status = 'initialized'
            for command in commands:
                result = subprocess.run(['git', *command], cwd=staged, capture_output=True, timeout=30)
                if result.returncode:
                    git_status = 'needsAttention'
                    break
        # rmdir refuses any concurrent addition. Publication never replaces a
        # destination that appeared after this check (including a symlink).
        if root.exists():
            if root.is_symlink():
                refuse('The destination changed during creation.')
            root.rmdir()
        archives.publish_directory(staged, root)
    return {**inspect(root), 'changed': True, 'git': git_status}


def inspect(directory):
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        refuse('Select an existing workspace directory.')
    specification = manifest()
    missing = [name for name in specification['seedFiles'] if not (root / name).is_file()]
    return {'workspaceRoot': str(root), 'recognized': (root / specification['markerFile']).is_file() or (root / 'prompts').is_dir(),
            'agentGuidePresent': (root / 'AGENTS.md').is_file(), 'missingSeedFiles': missing,
            'seedSchemaVersion': specification['schemaVersion'], 'changed': False}


def handoff(directory, *, executable=None):
    report = inspect(directory)
    if not report['recognized'] or not report['agentGuidePresent']:
        refuse('This workspace has no agent guide; initialize a new workspace or restore its instructions explicitly.')
    command = executable or [sys.executable, '-m', 'steerlab_server.client_cli']
    root = report['workspaceRoot']
    return {**report, 'executable': command, 'agentGuide': str(Path(root) / 'AGENTS.md'),
            'instructions': 'Read AGENTS.md before working. Discuss the research question and unresolved scientific choices with the researcher. Use only capabilities reported by this installed client. Workspace data remains local; running hardware receives execution copies.',
            'discovery': [command + ['--help'], command + ['science', 'list', '--root', root, '--json']],
            'nextAction': 'Read the agent guide, then choose a method with the researcher.'}
