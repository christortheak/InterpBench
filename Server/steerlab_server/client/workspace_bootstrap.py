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


#: The line the guide body declares its version on. Bodies written before the
#: line existed carry none and are version 1.
GUIDE_VERSION_PREFIX = 'Guide version: '


def guide_version(body):
    """The guide version a body declares; 1 when it declares none.

    Swift twin: ``AgentContract.guideVersion(of:)``. The version is what makes
    a refresh one-way: the header hash proves a file is unedited, but only the
    version can say whether its text is older or newer than this client's."""
    for line in [candidate for candidate in body.split('\n') if candidate][:12]:
        if line.startswith(GUIDE_VERSION_PREFIX):
            digits = line[len(GUIDE_VERSION_PREFIX):]
            if digits.isascii() and digits.isdigit():
                return int(digits)
    return 1


def refresh_agent_guide(directory):
    """Upgrade an unedited, older ``AGENTS.md`` in place; return a notice or None.

    The Mac command line has always done this (``WorkspaceStore.upkeepAgentContract``);
    this is the same rule for the Python client. It writes in exactly one case:
    the file's first line carries SteerLab's header, that header's SHA-256 is
    the hash of the text under it (so nobody has edited it), and the text is
    not the guide this client ships and does not declare a NEWER guide version.
    A missing guide is never created here, an edited one is never touched, and
    a refresh never downgrades. Any failure leaves the file as it was."""
    path = Path(directory) / 'AGENTS.md'
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return None
    header, newline, rest = text.partition('\n')
    if not newline or not (header.startswith(HEADER) and header.endswith(' -->')):
        return None
    if rest.startswith('\n'):
        rest = rest[1:]
    declared = header[len(HEADER):-len(' -->')]
    if len(declared) != 64 or declared != hashlib.sha256(rest.encode()).hexdigest():
        return None
    shipped = (RESOURCES / 'agent-guide.md').read_text(encoding='utf-8')
    if rest == shipped or guide_version(rest) > guide_version(shipped):
        return None
    staging = path.with_name(f'.AGENTS.md.{os.getpid()}.tmp')
    try:
        staging.write_text(HEADER + hashlib.sha256(shipped.encode()).hexdigest() + ' -->\n\n' + shipped, encoding='utf-8')
        os.replace(staging, path)
    except OSError:
        try:
            staging.unlink()
        except OSError:
            pass
        return None
    return (f'refreshed {path} to the agent guide this client ships — its header hashed the text it '
            'wrote and that hash still matched, so nobody had edited it; nothing else in the workspace was touched')


#: The executable whose commands the packaged guide topics show.
GUIDE_CLIENT = 'steerlab'


def guide_topics():
    """The packaged topic index: names and one-line summaries, in guide order."""
    index = json.loads((RESOURCES / 'agent-guide-topics.json').read_bytes())
    return [{'name': topic['name'], 'summary': topic['summary']} for topic in index['topics']]


def guide_topic(name):
    """One topic's text as this client renders it. The caller checks the name."""
    if name not in {topic['name'] for topic in guide_topics()}:
        raise KeyError(name)
    return (RESOURCES / f'agent-guide-topic-{name}.md').read_text(encoding='utf-8')


def refuse(reason):
    raise ExperimentStoreError(reason, gate='workspaceBootstrap', repair='Choose a new or empty workspace directory; keep existing studies and outputs in their current workspace.')


def available_git():
    executable = shutil.which('git')
    if sys.platform == 'darwin' and executable == '/usr/bin/git':
        # Apple's shim can open a developer-tools installation prompt. Git is
        # optional for workspace creation, so inspect tool selection first.
        selected = subprocess.run(['/usr/bin/xcode-select', '-p'], capture_output=True, timeout=5)
        if selected.returncode:
            return None
    return executable


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
        git = available_git() if use_git else None
        if git:
            commands = [['init'], ['-c', 'user.name=SteerLab', '-c', 'user.email=steerlab@localhost', 'add', '-A', '.'],
                        ['-c', 'user.name=SteerLab', '-c', 'user.email=steerlab@localhost', 'commit', '-m', 'Create research workspace']]
            git_status = 'initialized'
            for command in commands:
                result = subprocess.run([git, *command], cwd=staged, capture_output=True, timeout=30)
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
    if executable is None:
        # Name the same executable the installer returned when this interpreter
        # carries the client's console script; fall back to the module form for
        # a bare interpreter (a checkout venv or a test process).
        script = Path(sys.executable).with_name('steerlab')
        executable = [str(script)] if script.is_file() else [sys.executable, '-m', 'steerlab_server.client_cli']
    command = executable
    root = report['workspaceRoot']
    return {**report, 'executable': command, 'agentGuide': str(Path(root) / 'AGENTS.md'),
            'instructions': 'Read AGENTS.md before working. Discuss the research question and unresolved scientific choices with the researcher. Use only capabilities reported by this installed client. Workspace data remains local; running hardware receives execution copies.',
            'discovery': [command + ['--help'], command + ['science', 'list', '--root', root, '--json']],
            'nextAction': 'Read the agent guide, then choose a method with the researcher.'}
