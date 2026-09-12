"""Pinned, bounded activation datasets shared by capture, fitting, and evaluation."""
from pathlib import Path
import hashlib
import stat
from . import diagnostic_archives as archives, probe_artifacts as artifact
from .probe_artifacts import ProbeError

MAX_BYTES = 64 * 1024**2
ROLES = ('fit', 'selection', 'finalTest')


def file_ref(value, name):
    artifact.object_fields(value, 'path sha256', name)
    archives.parts(value['path']); artifact.digest(value['sha256'], name + '.sha256')
    return dict(value)


def read(ref, root):
    path = archives.ordinary(root, ref['path'])
    try:
        if not stat.S_ISREG(path.lstat().st_mode): raise ProbeError('Choose an ordinary input file, not a device, pipe, or directory.')
        with path.open('rb') as handle: content = handle.read(MAX_BYTES + 1)
    except OSError as exc: raise ProbeError('Cannot read the selected probe input: ' + str(exc)) from exc
    if len(content) > MAX_BYTES: raise ProbeError('This input exceeds the 64 MiB probe pilot limit. Prepare a smaller reviewed dataset.')
    if hashlib.sha256(content).hexdigest() != ref['sha256']: raise ProbeError('Probe input bytes changed. Review and pin the selected file again.')
    return content


def identity(row):
    return {key: row[key] for key in ('id', 'group', 'sourceSHA256')}


def rows_valid(rows, width):
    if not isinstance(rows, list) or not rows: raise ProbeError('The activation dataset needs labeled rows.')
    seen = set()
    for row in rows:
        artifact.object_fields(row, 'id group sourceSHA256 label activation', 'activation row')
        for key in ('id', 'group'): artifact.text(row[key], key)
        if row['id'] in seen: raise ProbeError('Activation row IDs must be unique.')
        seen.add(row['id']); artifact.digest(row['sourceSHA256'], 'sourceSHA256')
        if type(row['label']) is not bool: raise ProbeError('Activation labels must be JSON true or false.')
        artifact.vector(row['activation'], width, 'activation')
    return rows


def dataset(ref, root):
    d = artifact.read_json(read(ref, root))
    artifact.object_fields(d, 'artifactType schemaVersion input rows provenance', 'activation dataset')
    if d['artifactType'] != 'activation-dataset' or type(d['schemaVersion']) is not int or d['schemaVersion'] != 1:
        raise ProbeError('Use activation-dataset schemaVersion 1.')
    artifact.validate_input(d['input'])
    if not isinstance(d['provenance'], dict): raise ProbeError('Record activation provenance as a JSON object.')
    rows_valid(d['rows'], d['input']['hiddenSize'])
    return d


def overlap(left, right):
    """Report known reuse; absence is not proof of independent authorship."""
    return {key: sorted({r[key] for r in left} & {r[key] for r in right})
            for key in ('id', 'group', 'sourceSHA256')}


def labels(rows):
    positive = sum(row['label'] for row in rows)
    return {'rows': len(rows), 'positive': positive, 'negative': len(rows)-positive,
            'groups': len({r['group'] for r in rows})}


def text_rows(content, policy, seed):
    """Explicit roles or stable whole-group 60/20/20 hashing; never rebalance silently."""
    rows, seen, assignments, sources = [], set(), {}, {}
    for line in content.splitlines():
        if not line.strip(): continue
        row = artifact.read_json(line)
        required = 'id group text label split' if policy == 'explicit' else 'id group text label'
        artifact.object_fields(row, required, 'text row')
        for key in ('id', 'group', 'text'): artifact.text(row[key], key)
        if row['id'] in seen: raise ProbeError('Text row IDs must be unique.')
        seen.add(row['id'])
        if type(row['label']) is not bool: raise ProbeError('Labels must be true or false, not strings.')
        if policy == 'groupHash':
            bucket = int(archives.digest([seed, row['group']])[:16], 16) % 100
            row['split'] = 'fit' if bucket < 60 else 'selection' if bucket < 80 else 'finalTest'
        artifact.choice(row['split'], ROLES, 'split')
        digest = hashlib.sha256(row['text'].encode()).hexdigest()
        for mapping, key in ((assignments, row['group']), (sources, digest)):
            if key in mapping and mapping[key] != row['split']:
                raise ProbeError('A group or identical text crosses data roles. Keep related examples in one role.')
            mapping[key] = row['split']
        rows.append({**row, 'sourceSHA256': digest})
    if not rows: raise ProbeError('Supply labeled text rows before capture.')
    return rows


def new_run(root, operation, callback=None):
    import uuid
    root = Path(root).resolve(strict=True)
    parent = archives.ordinary(root, 'runs', missing=True); parent.mkdir(exist_ok=True)
    run = parent / (operation + '-' + uuid.uuid4().hex); run.mkdir()
    if callback: callback(str(run))
    return run


def save(run, name, value):
    content = archives.encoded(value)
    if len(content) > MAX_BYTES: raise ProbeError('Probe output exceeds the 64 MiB pilot limit. Reduce rows, tokens, or width.')
    with (run/name).open('xb') as handle: handle.write(content)
    return {'path': str(run/name), 'sha256': hashlib.sha256(content).hexdigest()}
