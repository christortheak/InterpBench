"""Explicit cleanup of verified, terminal diagnostic outputs under declared policy.

The server validates the client's custody attestation against its own exported
bytes. Only the originating client can re-read its local filesystem; all shipped
apply adapters do so immediately before sending that attestation.
"""
from pathlib import Path
import json
import hashlib
import os
import shutil
import time
import uuid
from types import SimpleNamespace
from ..experiment import diagnostic_archives as archives, manifest_files
from . import diagnostic_transport, workspace_lock

TERMINAL = {'succeeded', 'failed', 'cancelled', 'parked'}


def policy():
    path = os.environ.get('STEERLAB_DIAGNOSTIC_CLEANUP_POLICY')
    if not path: raise archives.Refusal('No diagnostic cleanup policy is configured; remote evidence is retained.')
    source = Path(path)
    if not source.is_file() or source.is_symlink(): raise archives.Refusal('Cleanup policy must be an ordinary configured file.')
    data = source.read_bytes(); document = json.loads(data)
    if (set(document) != {'schemaVersion', 'allowDiagnosticOutputRemoval', 'minimumRetentionHours', 'source'}
        or document['schemaVersion'] != 1 or document['allowDiagnosticOutputRemoval'] is not True
        or type(document['minimumRetentionHours']) not in (int, float)
        or not 0 <= document['minimumRetentionHours'] <= 24 * 36500
        or not isinstance(document['source'], str) or not document['source'].strip()):
        raise archives.Refusal('Cleanup needs an explicit allow rule, nonnegative retention hours and a policy source.')
    return {'document': document, 'fileSHA256': hashlib.sha256(data).hexdigest()}


def contains_reference(value, relative, absolute):
    if isinstance(value, str):
        return (value == relative or value.startswith(relative + '/') or value == absolute or value.startswith(absolute + '/')
                or Path(relative).name in value.split('/'))
    if isinstance(value, list): return any(contains_reference(v, relative, absolute) for v in value)
    if isinstance(value, dict): return any(contains_reference(v, relative, absolute) for v in value.values())
    return False


#: Keys an archive's description carried before it stopped naming the server's
#: folders. A receipt holding them was made from such an archive.
_LEGACY_CONTEXT_ROOTS = ('servingRoot', 'metadataRoot', 'executionRoot')


class LegacyCustody(archives.Refusal):
    repair_action = ('Fetch the evidence again with `science-fetch` (the export is made afresh and its '
                     'import gives a current receipt; the local copy is unchanged), then plan the cleanup '
                     'with the new receipt SHA-256.')


def facts(job_id, custody, jobs, profile):
    return _facts(job_id, custody, jobs, profile)[0]


def _facts(job_id, custody, jobs, profile):
    """The cleanup plan, and the execution root it acts on: a path on this
    server, read here, never from the client's receipt."""
    job = jobs.get(job_id)
    if job is None or job.kind not in ('science:battery', 'science:stability'):
        raise archives.Refusal('This retention policy covers battery and stability copies only; other scientific artifacts are retained.')
    declared = policy()
    # Before the export, which a changed description makes afresh: a receipt
    # that cannot match is refused without packaging the output to find out.
    if (isinstance(custody, dict) and isinstance(custody.get('context'), dict)
            and any(key in custody['context'] for key in _LEGACY_CONTEXT_ROOTS)):
        raise LegacyCustody('This receipt comes from an archive made before archives stopped naming the '
                            "server's folders, so it no longer matches this job's export.")
    reference, execution_root = diagnostic_transport.exported(job_id, jobs, profile)
    context = reference['context']
    if (not isinstance(custody, dict) or custody.get('kind') != 'diagnosticCustody'
        or custody.get('schemaVersion') != 1 or not custody.get('workspaceRoot')
        or custody.get('context') != context or custody.get('archiveSHA256') != reference['bundleSha256']
        or custody.get('files') != reference['entries']):
        raise archives.Refusal('Local custody attestation does not match this job and its exported bytes.')
    serving_root = Path(profile.root).resolve()
    if execution_root == serving_root:
        raise archives.Refusal('Managed cleanup is limited to isolated diagnostic execution copies; retain original workspace outputs.')
    target = execution_root / context['outputRelative']
    blockers, dependencies = [], []
    for job in jobs.list():
        if job.id == job_id: continue
        if job.status not in TERMINAL or job.status == 'parked':
            blockers.append('Job may be active, uncertain or resumable: ' + job.id)
        if contains_reference(job.result, context['outputRelative'], str(target)):
            blockers.append('Another job references this output: ' + job.id)
        dependencies.append({'id': job.id, 'status': job.status, 'resultSHA256': archives.digest(job.result)})
    # Workbench declarations are independent of job state. Malformed documents
    # block rather than silently dropping their dependency evidence.
    for relative in ('experiments', 'runs/model-variants'):
        directory = archives.ordinary(profile.root, relative, missing=True)
        if not directory.exists(): continue
        for path in sorted(directory.rglob('*.json')):
            rel = path.relative_to(Path(profile.root).resolve()).as_posix()
            data = archives.ordinary(profile.root, rel).read_bytes()
            try: value = json.loads(data)
            except ValueError: blockers.append('Unreadable dependency declaration: ' + rel); continue
            if contains_reference(value, context['outputRelative'], str(target)):
                blockers.append('Workspace declaration references this output: ' + rel)
            dependencies.append({'path': rel, 'sha256': archives.file_hash(path)})
    job = jobs.get(job_id)
    retain_until = job.finished_at + declared['document']['minimumRetentionHours'] * 3600
    if time.time() < retain_until: blockers.append('Declared minimum retention period has not elapsed.')
    result = {'schemaVersion': 1, 'kind': 'diagnosticCleanupPlan', 'jobID': job_id,
              'servingRoot': str(serving_root), 'metadataRoot': str(Path(profile.metadata_root).resolve()),
              'target': str(target), 'outputSHA256': context['outputSHA256'],
              'exportSHA256': reference['bundleSha256'], 'custodySHA256': archives.digest(custody),
              'policy': declared, 'retainUntil': retain_until, 'dependencies': dependencies,
              'blockers': sorted(blockers), 'eligible': not blockers,
              'retained': ['staged inputs', 'export archive', 'job records', 'local evidence and receipt']}
    return {**result, 'planSHA256': archives.digest(result)}, execution_root


def plan(job_id, custody, jobs, profile):
    with workspace_lock.switching(), jobs.store.exclusive_snapshot() as snapshot:
        view = SimpleNamespace(get=snapshot.get, list=lambda: list(snapshot.values()))
        return facts(job_id, custody, view, profile)


def apply(job_id, custody, expected, jobs, profile, *, confirmed):
    if confirmed is not True: raise archives.Refusal('Cleanup requires explicit confirmation after review and local custody verification.')
    with workspace_lock.switching(), jobs.store.exclusive_snapshot() as snapshot:
        jobs = SimpleNamespace(get=snapshot.get, list=lambda: list(snapshot.values()))
        with manifest_files.transaction(str(Path(profile.metadata_root) / 'diagnostic-cleanup'), workspace_root=profile.metadata_root):
            reviewed, execution_root = _facts(job_id, custody, jobs, profile)
            if reviewed['planSHA256'] != expected or not reviewed['eligible']:
                raise archives.Refusal('Cleanup facts changed or remain blocked; nothing was removed. Review a new plan.')
            target = Path(reviewed['target'])
            audit_dir = archives.ordinary(profile.metadata_root, 'diagnostic-cleanup', missing=True)
            audit_dir.mkdir(parents=True, exist_ok=True)
            operation = uuid.uuid4().hex
            record = {'operationID': operation, 'plan': reviewed, 'state': 'intended', 'removed': [], 'retained': [str(target)]}
            audit_path = audit_dir / (operation + '.json')
            def save():
                temp = audit_path.with_suffix('.tmp'); temp.write_bytes(archives.encoded(record)); os.replace(temp, audit_path)
            save()
            quarantine = target.with_name('.diagnostic-cleanup-' + operation)
            try:
                archives.publish_directory(target, quarantine)
                relative = quarantine.relative_to(execution_root).as_posix()
                entries = archives.snapshot(execution_root, archives.files_in(execution_root, relative))
                original_prefix = custody['context']['outputRelative']
                for entry in entries: entry['path'] = original_prefix + entry['path'][len(relative):]
                if entries != custody['files']:
                    if not target.exists(): archives.publish_directory(quarantine, target)
                    raise archives.Refusal('Output changed during cleanup admission; retained for inspection.')
                shutil.rmtree(quarantine)
                record.update(state='removed', removed=[str(target)], retained=reviewed['retained'])
            except Exception as exc:
                record.update(state='attention', error=str(exc), retained=[str(p) for p in (target, quarantine) if p.exists()])
                save()
                raise archives.Refusal('Cleanup needs attention; inspect audit record ' + str(audit_path)) from exc
            save()
            return {**record, 'auditPath': str(audit_path), 'changed': True}
