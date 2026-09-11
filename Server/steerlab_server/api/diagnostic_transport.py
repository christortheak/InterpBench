"""Runner execution copies and export references; never workbench authorship."""
from ..experiment import input_hashes
from pathlib import Path
import shutil
import tempfile
from ..experiment import diagnostic_archives as archives, manifest_files
from . import scientific_execution


def storage(profile):
    return Path(profile.run_root or Path(profile.root) / 'runs').resolve()


def stage(path, expected, profile):
    base = storage(profile)
    base.mkdir(parents=True, exist_ok=True)
    source = Path(path)
    if not source.is_absolute(): source = base / source
    # External transport stages under the configured run root, never an
    # arbitrary server path supplied by a remote caller.
    try: relative = source.relative_to(base).as_posix()
    except ValueError as exc:
        error = archives.Refusal('The archive must be transferred beneath the runner run root: ' + str(base))
        error.repair_action = 'Use the site-approved transfer into ' + str(base) + ', then call science-stage with that server path and the archive SHA-256. This is an execution copy; keep local sources.'
        raise error from exc
    source = archives.ordinary(base, relative)
    metadata = archives.inspect(source, expected)
    if metadata['kind'] != 'diagnosticInput': raise archives.Refusal('Expected diagnostic inputs, not evidence.')
    target = archives.ordinary(base, 'diagnostic-input-' + expected, missing=True)
    with manifest_files.transaction(str(target), workspace_root=profile.metadata_root):
        archives.refuse_empty_publication_claim(target)
        if not target.exists():
            with tempfile.TemporaryDirectory(dir=base) as temp:
                capture = Path(temp) / 'input.tar.gz'; shutil.copyfile(source, capture)
                workspace = Path(temp) / 'capsule/workspace'; workspace.mkdir(parents=True)
                archives.inspect(capture, expected, extract_to=workspace)
                # Evaluate the staged bytes through the real cheap admission
                # owner before publication; no model is loaded or job submitted.
                from ..experiment import diagnostic_inputs
                closure = diagnostic_inputs.plan(metadata['context']['request'], str(workspace))
                if closure['files'] != metadata['entries']: raise archives.Refusal('Input bundle does not contain exactly the required diagnostic closure.')
                scientific_execution.input_plan(metadata['context']['request'], str(workspace))
                (Path(temp) / 'capsule/input.json').write_bytes(archives.encoded(metadata))
                shutil.move(capture, Path(temp) / 'capsule/input.tar.gz')
                archives.publish_directory(Path(temp) / 'capsule', target)
        request, root = resolve(expected, profile)
    return {'inputBundleSHA256': expected, 'request': {'inputBundleSHA256': expected},
            'executionRoot': str(root), 'operation': request['operation'],
            'nextAction': 'Plan and submit the returned request object on this controller. Client workspace paths in the original request are not runner paths.'}


def resolve(expected, profile):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise archives.Refusal('Invalid staged input digest.')
    base = storage(profile)
    capsule = archives.ordinary(base, 'diagnostic-input-' + expected)
    root = archives.ordinary(capsule, 'workspace')
    return verify_inputs(root, expected), root


@input_hashes.operation
def verify_inputs(root, expected):
    """Re-read the complete portable closure at admission and on the queued child."""
    from ..experiment import diagnostic_inputs
    root = Path(root)
    if root.name != 'workspace' or root.parent.name != 'diagnostic-input-' + expected:
        raise archives.Refusal('Staged execution root differs from the input digest.')
    metadata = archives.inspect(archives.ordinary(root.parent, 'input.tar.gz'), expected)
    if metadata['kind'] != 'diagnosticInput': raise archives.Refusal('Staged archive has the wrong kind.')
    if archives.snapshot(root, [e['path'] for e in metadata['entries']]) != metadata['entries']:
        raise archives.Refusal('Staged input bytes changed; restage the original archive.')
    closure = diagnostic_inputs.plan(metadata['context']['request'], root)
    if closure['files'] != metadata['entries']:
        raise archives.Refusal('Staged dependency closure changed or escapes the isolated copy.')
    return metadata['context']['request']


def output(job_id, jobs, profile):
    job = jobs.get(job_id)
    if job is None or job.status != 'succeeded' or not job.finished_at or not job.kind.startswith('science:'):
        raise archives.Refusal('Only successfully completed diagnostic jobs are export/cleanup eligible; partial, active and resumable outputs stay protected.')
    result = job.result or {}; plan = result.get('scientificPlan') or {}
    capsule = plan.get('inputBundleSHA256') or plan.get('executionCapsuleSHA256')
    if capsule:
        _, root = resolve(capsule, profile)
    else:
        root = Path(profile.root).resolve()
    if plan.get('root') != str(root): raise archives.Refusal('This job belongs to another serving root.')
    if job.kind == 'science:optvec-campaign':
        from . import managed_campaign
        managed_campaign.require_complete(job_id, jobs, profile)
    directory = result.get('diagnosticDirectory') if job.kind == 'science:stability' else result.get('runDirectory')
    if not directory: raise archives.Refusal('Job has no diagnostic output location.')
    relative = Path(directory).absolute().relative_to(root).as_posix()
    components = archives.parts(relative)
    wanted = 'diagnostics' if job.kind == 'science:stability' else 'runs'
    if len(components) != 2 or components[0] != wanted:
        raise archives.Refusal('Output is outside the bounded diagnostic artifact class.')
    entries = archives.snapshot(root, archives.files_in(root, relative))
    context = {'jobID': job_id, 'servingRoot': str(Path(profile.root).resolve()),
               'metadataRoot': str(Path(profile.metadata_root).resolve()), 'executionRoot': str(root),
               'operation': plan['request']['operation'], 'outputRelative': relative,
               'scientificPlanSHA256': plan['planSHA256'], 'outputSHA256': archives.digest(entries)}
    return root, relative, entries, context


def export(job_id, jobs, profile):
    root, relative, entries, context = output(job_id, jobs, profile)
    base = storage(profile)
    with manifest_files.transaction(str(base / ('diagnostic-export-' + job_id)), workspace_root=profile.metadata_root):
        directory = archives.ordinary(base, 'diagnostic-exports', missing=True); directory.mkdir(parents=True, exist_ok=True)
        archive = archives.ordinary(directory, archives.digest(context) + '.tar.gz', missing=True)
        if archive.exists():
            metadata = archives.inspect(archive, archives.file_hash(archive))
            if metadata['context'] != context or metadata['entries'] != entries:
                raise archives.Refusal('Export archive changed; retain originals and inspect the export store.')
        else:
            metadata = archives.package(root, [e['path'] for e in entries], archive, kind='diagnosticEvidence', context=context, expected_entries=entries)
        if metadata['entries'] != entries: raise archives.Refusal('Output changed during export; do not import it.')
        return {'bundlePath': str(archive), 'bundleSha256': archives.file_hash(archive),
                'context': context, 'entries': entries, 'bytes': archive.stat().st_size}
