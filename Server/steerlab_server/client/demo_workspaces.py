"""Demo Workspaces: worked examples this client carries, opened as a verified copy.

A Demo Workspace is a complete study someone can read before downloading a
model: a finished study with its results, and a draft copy ready to run. One
is built per compute backend, because a workspace is bound to one backend and
its vectors and evidence are stamped by it.

The carried tree is never opened in place. ``open_copy`` copies it to a folder
the researcher chooses, checks every copied byte against the original, checks
each study the way ``experiment verify`` does, records the backend's compute
binding, and commits the result once. A build may carry any subset of the
backends, including none.

Swift twin: ``Sources/ExperimentKit/DemoWorkspace.swift``. No engine import at
module load; the study check imports the manifest reader when it runs.
"""
import hashlib
import json
import os
from pathlib import Path

from . import workspace_bootstrap as bootstrap

#: The backends a Demo Workspace can be built for, in the order they are offered.
BACKENDS = ('mlx', 'mps', 'cuda')
SCHEMA_VERSION = 1

#: Each backend in plain words: where its studies run.
BACKEND_TITLES = {
    'mlx': 'This Mac, quick start',
    'mps': 'This Mac, full capabilities',
    'cuda': 'Another machine',
}

#: What a copy records in ``.steerlab/workspace.json``: the compute binding the
#: Mac app writes for the matching choice, so the app opens the copy already
#: set to the engine its vectors and evidence came from.
BINDINGS = {
    'mlx': {'computeSubstrate': 'local-mlx'},
    'mps': {'computeSubstrate': 'cluster', 'computeLocation': 'this-mac'},
    'cuda': {'computeSubstrate': 'cluster', 'computeLocation': 'another-machine'},
}
BINDING_PATH = '.steerlab/workspace.json'

#: Written fresh for every copy by workspace creation, so a carried tree's own
#: copies of them are never used.
GENERATED = ('AGENTS.md', 'WORKSPACE.md')

#: Inside an installed client (the release builder places the trees there;
#: the hyphen keeps the folder from ever being importable), and beside a
#: source checkout.
PACKAGED = Path(__file__).parent / 'demo-workspaces'
CHECKOUT = Path(__file__).parents[3] / 'DemoWorkspaces'


class DemoRefusal(Exception):
    """A demo that cannot be opened, with a stable code and plain words.

    ``repair`` is written for a person; the command layer adds the command to
    run. ``payload`` carries what was learned before the refusal."""

    def __init__(self, code, reason, repair, *, payload=None):
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.repair = repair
        self.payload = dict(payload or {})


REINSTALL = 'Reinstall SteerLab, then open the demo again in a new folder.'


def carried_root():
    """Where this install keeps its Demo Workspaces, or None when it has none."""
    for candidate in (PACKAGED, CHECKOUT):
        if candidate.is_dir():
            return candidate
    return None


def _damaged(backend, detail):
    return DemoRefusal(
        'demoDamaged',
        f'The {backend} Demo Workspace in this copy of SteerLab is incomplete: {detail}',
        REINSTALL, payload={'backend': backend})


def describe(directory):
    """Read and check one carried demo's ``demo.json``; raise ``DemoRefusal``.

    Returns the document as written, so fields a later version adds travel
    through untouched. Checked: the schema version, that the backend matches
    the folder's name, a title and a one-line summary, the model and its
    approximate download size, and that every named study is present."""
    directory = Path(directory)
    backend = directory.name
    try:
        document = json.loads((directory / 'demo.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise _damaged(backend, 'demo.json could not be read.') from None

    def text(value):
        return isinstance(value, str) and bool(value.strip()) and '\n' not in value

    if not isinstance(document, dict):
        raise _damaged(backend, 'demo.json is not an object.')
    if type(document.get('schemaVersion')) is not int or document['schemaVersion'] != SCHEMA_VERSION:
        raise _damaged(backend, f'demo.json needs "schemaVersion": {SCHEMA_VERSION}.')
    if document.get('backend') != backend:
        raise _damaged(backend, f'demo.json names the backend {document.get("backend")!r}, but its folder is {backend!r}.')
    for key in ('title', 'summary'):
        if not text(document.get(key)):
            raise _damaged(backend, f'demo.json needs a one-line "{key}".')
    model = document.get('model')
    size = model.get('approximateDownloadGB') if isinstance(model, dict) else None
    if not isinstance(model, dict) or not text(model.get('id')):
        raise _damaged(backend, 'demo.json needs "model": {"id": …, "approximateDownloadGB": …}.')
    if type(size) not in (int, float) or not size > 0:
        raise _damaged(backend, 'demo.json needs the model\'s "approximateDownloadGB" as a number above zero.')
    studies = document.get('studies')
    if not isinstance(studies, list) or not studies:
        raise _damaged(backend, 'demo.json needs a "studies" list that names at least one study.')
    names = []
    for study in studies:
        name = study.get('name') if isinstance(study, dict) else None
        if not text(name) or '/' in name or name in ('.', '..'):
            raise _damaged(backend, 'every entry in "studies" needs a "name".')
        if 'summary' in study and not text(study['summary']):
            raise _damaged(backend, f'the study {name!r} has a "summary" that is not one line of text.')
        if not ((directory / 'experiments' / name / 'experiment.json').is_file()
                or (directory / 'experiments' / f'{name}.json').is_file()):
            raise _damaged(backend, f'the study {name!r} is named in demo.json but is not under experiments/.')
        names.append(name)
    if len(set(names)) != len(names):
        raise _damaged(backend, 'demo.json names a study twice.')
    readme = directory / 'README.md'
    if not readme.is_file() or not readme.read_text(encoding='utf-8').strip():
        raise _damaged(backend, 'its README.md is missing or empty.')
    return document


def files(directory):
    """The files a copy receives, as sorted workspace-relative paths.

    Left out: anything whose path has a dot-prefixed part (workspace-local
    state, and files an installer may leave), Python bytecode caches, and the
    two files workspace creation writes fresh for every copy. A symlink is
    refused: a copy must be made of this tree's own bytes."""
    directory = Path(directory)
    names = []
    for current, folders, entries in os.walk(directory, followlinks=False):
        here = Path(current)
        for name in list(folders) + entries:
            if (here / name).is_symlink() and not name.startswith('.'):
                raise _damaged(directory.name, f'{(here / name).relative_to(directory).as_posix()} is a symbolic link.')
        folders[:] = [name for name in folders if not name.startswith('.') and name != '__pycache__']
        for name in entries:
            relative = (here / name).relative_to(directory).as_posix()
            if not name.startswith('.') and not name.endswith('.pyc') and relative not in GENERATED:
                names.append(relative)
    return sorted(names)


def available(root=None):
    """The demos this install carries and can open, in ``BACKENDS`` order."""
    root = carried_root() if root is None else Path(root)
    found = []
    for backend in BACKENDS if root is not None else ():
        if (root / backend).is_dir():
            try:
                found.append(describe(root / backend))
            except DemoRefusal:
                continue  # `carried` reports the damage when that backend is asked for
    return found


def carried(backend, root=None):
    """The folder and description of one carried demo, or a plain refusal."""
    root = carried_root() if root is None else Path(root)
    directory = root / backend if root is not None else None
    if directory is None or not directory.is_dir():
        others = [demo['backend'] for demo in available(root)]
        carries = ('It carries one for ' + _joined(others) + '.') if others else 'It carries none.'
        raise DemoRefusal(
            'demoNotCarried',
            f'This copy of SteerLab carries no Demo Workspace for {backend} ({BACKEND_TITLES[backend]}). {carries}',
            'Open a demo this copy carries, or create an ordinary new workspace.',
            payload={'backend': backend, 'carried': others})
    return directory, describe(directory)


def _joined(words):
    """`a`, `a and b`, or `a, b, and c`."""
    if len(words) < 3:
        return ' and '.join(words)
    return ', '.join(words[:-1]) + ', and ' + words[-1]


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def binding_text(backend):
    """The exact bytes of a copy's compute binding: sorted keys, two-space
    indent, and ` : ` between key and value, with no final newline — what the
    Mac app's own writer produces, so both clients record one file."""
    return json.dumps(BINDINGS[backend], indent=2, sort_keys=True, separators=(',', ' : '))


def verify_studies(root, names):
    """Each named study, checked exactly as ``experiment verify`` checks it,
    against the workspace at ``root``."""
    from ..experiment.manifest import Manifest
    results = []
    for name in names:
        try:
            manifest = Manifest.load(name, str(root))
            status, violations = manifest.status, list(manifest.verify(str(root)))
        except (OSError, ValueError, KeyError) as error:
            status, violations = None, [f'the study could not be read ({error})']
        results.append({'name': name, 'status': status, 'verified': not violations, 'violations': violations})
    return results


def open_copy(backend, directory, *, use_git=True, root=None, seed=bootstrap.SEED):
    """Copy one carried demo to ``directory`` and return what was made.

    The copy is staged beside the destination and published only when all of
    it holds: every file is byte-for-byte the carried one, and every study the
    demo names passes verification in the copy. A refusal leaves nothing
    behind. The seed fills in any file a new workspace would have and the demo
    does not carry; it never replaces one the demo does carry."""
    source, description = carried(backend, root)
    names = files(source)
    requested = Path(directory).expanduser().absolute()
    if requested.is_symlink() or (requested.exists() and (not requested.is_dir() or any(requested.iterdir()))):
        raise DemoRefusal(
            'destinationNotEmpty',
            f'The folder {requested} already exists and is not empty, so nothing was copied into it.',
            'Choose a new or empty folder for the copy.', payload={'backend': backend})
    checked = {}

    def place(staged):
        for name in names:
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / name).read_bytes())

    def check(staged):
        differing = [name for name in names if _sha256(source / name) != _sha256(staged / name)]
        if differing:
            raise DemoRefusal(
                'demoCopyUnverified',
                f'The copy of the {backend} Demo Workspace does not match the original in {len(differing)} file(s), '
                'so nothing was created.',
                'Check that the disk has free space, then try again in a new folder.',
                payload={'backend': backend, 'differingFiles': differing})
        studies = verify_studies(staged, [study['name'] for study in description['studies']])
        failed = [study['name'] for study in studies if not study['verified']]
        if failed:
            raise DemoRefusal(
                'demoCopyUnverified',
                f'The {backend} Demo Workspace was copied, but {_joined(failed)} did not pass verification '
                'in the copy, so nothing was created.',
                REINSTALL, payload={'backend': backend, 'studies': studies})
        binding = staged / BINDING_PATH
        binding.parent.mkdir(parents=True, exist_ok=True)
        binding.write_text(binding_text(backend), encoding='utf-8')
        checked.update(files=len(names), bytes=sum((staged / name).stat().st_size for name in names), studies=studies)

    result = bootstrap.initialize(
        directory, use_git=use_git, seed=seed, before_seed=place, before_commit=check,
        commit_message=f'Open the {backend} Demo Workspace')
    return {**result, 'demo': description, 'demoReadme': str(Path(result['workspaceRoot']) / 'README.md'),
            'compute': dict(BINDINGS[backend]), 'verification': {**checked, 'identical': True}}
