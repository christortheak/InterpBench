"""Read-only workspace probe discovery and inspection for every client surface.

Legacy bytes are retained, never upgraded in place or assigned invented pins.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import stat

from . import probe_artifacts as artifacts
from .probe_artifacts import ProbeError

MAX_BYTES = 64 * 1024 * 1024


def workspace(root):
    root = Path(root).resolve()
    if not root.is_dir(): raise ProbeError('Open an existing workspace folder before inspecting probes.')
    return root


def ordinary(path, root):
    path = Path(path)
    if not path.is_absolute(): path = root / path
    try: parts = path.relative_to(root).parts
    except ValueError as exc: raise ProbeError('Choose a probe inside this workspace’s runs folder.') from exc
    if len(parts) != 3 or parts[0] != 'runs' or not path.name.endswith(('.probe.json', '-probe.json')) or any(p in ('.', '..') for p in parts):
        raise ProbeError('Choose runs/<run>/<name>.probe.json or the legacy <name>-probe.json.')
    current = root
    for index, part in enumerate(parts):
        current = current / part
        mode = current.lstat().st_mode
        if (index == len(parts) - 1 and not stat.S_ISREG(mode)) or (index < len(parts) - 1 and not stat.S_ISDIR(mode)):
            raise ProbeError('Probe paths must use ordinary files and folders, without symbolic links.')
    return current


def load(path):
    with path.open('rb') as handle: data = handle.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES: raise ProbeError('This probe JSON exceeds the 64 MiB inspection limit. Keep large activation datasets separate from the instrument.')
    document = artifacts.read_json(data)
    if not isinstance(document, dict): raise ProbeError('A probe artifact must be a JSON object.')
    return data, document


def legacy(document):
    for key in ('concept', 'modelID'): artifacts.text(document.get(key), key)
    artifacts.integer(document.get('layer'), 'layer')
    probe = document.get('probe')
    if not isinstance(probe, dict): raise ProbeError('A legacy reader requires a scalar probe object.')
    direction = probe.get('direction')
    if not isinstance(direction, list) or not direction: raise ProbeError('The legacy reader has no direction.')
    artifacts.vector(direction, len(direction), 'direction')
    for key in ('projectionCenter', 'projectionScale', 'orientation', 'positiveMean', 'negativeMean'):
        artifacts.number(probe.get(key), key)
    if probe['projectionScale'] <= 0 or probe['orientation'] not in (-1, 1):
        raise ProbeError('The legacy reader needs a positive scale and an orientation of +1 or -1.')
    if probe.get('activationCenter') is not None: artifacts.vector(probe['activationCenter'], len(direction), 'activationCenter')
    notes = ['Legacy reading score, not a calibrated probability.',
             'Activation site, rendering, and coordinate provenance are incomplete. This entry does not establish compatibility with a different backend.']
    if document.get('kind') == 'readingProbe':
        format_name, method = 'python-reading-probe', 'Mean-difference reader (legacy Python)'
        if 'heldOutAccuracy' in document:
            accuracy = artifacts.number(document['heldOutAccuracy'], 'heldOutAccuracy')
            if not 0 <= accuracy <= 1: raise ProbeError('Selection accuracy must be between zero and one.')
            notes.append('heldOutAccuracy was used for layer selection; it is not final-test accuracy.')
    elif 'recipeName' in document and 'createdAt' in document:
        artifacts.text(document['recipeName'], 'recipeName')
        artifacts.text(document['createdAt'], 'createdAt')
        format_name, method = 'native-reading-probe', document['recipeName']
        notes.append('The recipe name alone does not identify the fitted direction algorithm; consult the pinned recipe when available.')
    else: raise ProbeError('Unrecognized legacy reader structure; filenames alone do not define the artifact.')
    return format_name, method, notes


def inspect(path, root):
    root = workspace(root)
    path = ordinary(path, root)
    data, d = load(path)
    if 'artifactType' in d or 'schemaVersion' in d:
        d = artifacts.validate(d)
        format_name, method = 'activation-probe-v1', d['method']
        label, model, layer, created = d['label'], d['input']['modelID'], d['input']['site']['layer'], d['createdAt']
        notes = artifacts.limitations(d)
    else:
        format_name, method, notes = legacy(d)
        label, model, layer, created = d['concept'], d['modelID'], d['layer'], d.get('createdAt')
    return {'path': path.relative_to(root).as_posix(), 'sha256': hashlib.sha256(data).hexdigest(),
            'format': format_name, 'label': label, 'modelID': model, 'layer': layer,
            'method': method, 'createdAt': created, 'limitations': notes, 'document': d}


def inventory(root):
    root = workspace(root)
    runs = root / 'runs'
    records, issues = [], []
    if not runs.exists() and not runs.is_symlink(): return {'probes': records, 'issues': issues, 'count': 0, 'changed': False}
    if not stat.S_ISDIR(runs.lstat().st_mode): raise ProbeError('The runs folder must be an ordinary directory, without symbolic links.')
    for directory in sorted(runs.iterdir()):
        if directory.is_symlink():
            issues.append({'path': directory.relative_to(root).as_posix(), 'reason': 'Symbolic run folders are not scanned; collect the probe evidence into an ordinary workspace run folder.'})
            continue
        if not stat.S_ISDIR(directory.lstat().st_mode): continue
        try: candidates = sorted(directory.iterdir())
        except OSError as exc:
            issues.append({'path': directory.relative_to(root).as_posix(), 'reason': str(exc)})
            continue
        for path in candidates:
            if not path.name.endswith(('.probe.json', '-probe.json')): continue
            try:
                record = inspect(path, root)
                # Discovery is a compact index; full parameters are fetched on selection.
                record.pop('document')
                records.append(record)
            except (OSError, ValueError) as exc:
                issues.append({'path': path.relative_to(root).as_posix(), 'reason': str(exc)})
    return {'probes': records, 'issues': issues, 'count': len(records), 'changed': False}
