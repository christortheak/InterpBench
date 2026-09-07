"""Portable SAE roster inspection and reviewed draft pinning (no model load)."""
import hashlib
from pathlib import Path
from . import diagnostic_archives as archives, sae_candidates, sae_qualification, experiment_store, manifest_files


def inspect(kind, path, root):
    source = archives.ordinary(root, path)
    data = source.read_bytes()
    if kind == 'candidates':
        value = sae_candidates.CandidateManifest.from_bytes(data)
        warnings = sae_qualification.roster_warnings(value)
    elif kind == 'qualification':
        value = sae_qualification.from_bytes(data); warnings = []
    else: raise archives.Refusal('Choose candidates or qualification.')
    return {'path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'summary': value.summary(),
            'warnings': warnings, 'changed': False, 'scientificQualification': 'notEstablishedByInspection'}


def pin_plan(experiment, path, root):
    document = experiment_store.load_raw(experiment, root)
    if document.get('status') != 'draft': raise archives.Refusal('Pin a roster to a draft; duplicate a frozen study first.')
    report = inspect('candidates', path, root)
    result = {'experiment': experiment, 'workspaceRoot': str(Path(root).resolve()), 'roster': report,
              'manifestSHA256': document.source_digest, 'changed': False}
    return {**result, 'planSHA256': archives.digest(result)}


def pin(experiment, path, root, expected):
    document = experiment_store.load_raw(experiment, root)
    with manifest_files.transaction(document.source_path, workspace_root=root):
        with manifest_files.transaction(str(archives.ordinary(root, path)), workspace_root=root):
            fresh = pin_plan(experiment, path, root)
            if fresh['planSHA256'] != expected: raise archives.Refusal('Study or roster changed; review a fresh pin plan.')
            current = experiment_store.load_raw(experiment, root)
            if current.get('saeCandidates') == {'path': path, 'hash': fresh['roster']['sha256']}:
                return {'changed': False, 'study': current, 'review': fresh}
            result = experiment_store.pin_sae_candidates(experiment, path, root)
            return {'changed': True, 'study': result, 'review': fresh}
