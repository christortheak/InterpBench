"""Closed standalone diagnostic jobs over the existing local/Slurm executors.

Inputs must already be staged under the runner root. Planning is read-only;
submission pins that plan and the child rechecks inputs after queueing. Neither
operation is a study stage, nor does either support checkpoint resumption.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

from .executors import LocalExecutor, SlurmExecutor, SlurmResources, JobBundle, render_slurm_script
from .profile import ServerProfile, server_role
from .workspace_lock import submitting


class ScientificRefusal(ValueError):
    code = 'scientificExecutionRefused'
    repair_action = 'Correct or restage the declared inputs, read science-plan on the same runner, then submit its exact planSHA256.'


class ScientificInputUnavailable(ScientificRefusal):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def request_document(raw):
    if not isinstance(raw, dict) or set(raw) != {'operation', 'parameters'}:
        raise ScientificRefusal('Supply exactly operation and parameters.')
    operation, parameters = raw['operation'], raw['parameters']
    from ..experiment import managed_methods
    if isinstance(operation, str) and operation in managed_methods.OPERATIONS:
        return managed_methods.request(operation, parameters)
    allowed = {
        'battery': {'batteryFile', 'agents', 'modelID', 'revision', 'alphaUnits', 'dtype', 'device'},
        'stability': {'experiment', 'concept', 'resamples', 'fraction', 'seed', 'orderShuffles', 'dtype', 'device'},
    }
    if not isinstance(operation, str) or operation not in allowed or not isinstance(parameters, dict) or set(parameters) - allowed[operation]:
        raise ScientificRefusal('Choose a catalogued scientific operation and only its declared parameters.')
    p = dict(parameters)
    for key in ('dtype', 'device', 'modelID', 'revision', 'alphaUnits', 'batteryFile', 'experiment', 'concept'):
        if key in p and (not isinstance(p[key], str) or not p[key].strip()):
            raise ScientificRefusal(f'{key} must be a nonempty string.')
    if operation == 'battery':
        if not p.get('batteryFile') or not isinstance(p.get('agents'), list) or not p['agents'] or not all(isinstance(a, str) and a for a in p['agents']):
            raise ScientificRefusal('A batteryFile and a nonempty agents array are required.')
        p.setdefault('alphaUnits', 'norm')
    else:
        for key in ('experiment', 'concept'):
            if not p.get(key) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', p[key]):
                raise ScientificRefusal(f'{key} must be one workspace name, not a path.')
        for key, default in (('resamples', 32), ('orderShuffles', 8)):
            p.setdefault(key, default)
            if type(p[key]) is not int or p[key] < 0:
                raise ScientificRefusal(f'{key} must be a nonnegative integer.')
        seed = p.get('seed', '0')
        if not ((type(seed) is int and seed >= 0) or (isinstance(seed, str) and re.fullmatch(r'[0-9]{1,20}', seed))):
            raise ScientificRefusal('seed must be a nonnegative UInt64 integer or decimal string.')
        if int(seed) > 2**64 - 1:
            raise ScientificRefusal('seed must fit UInt64.')
        # Decimal text avoids rounding when plans pass through JSON clients
        # whose generic number type is IEEE double. Execution still gets int.
        p['seed'] = str(int(seed))
        p.setdefault('fraction', 0.5)
        if type(p['fraction']) not in (int, float) or not 0 < p['fraction'] <= 1:
            raise ScientificRefusal('fraction must be greater than zero and at most one.')
    return {'operation': operation, 'parameters': p}


def workspace_file(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ScientificRefusal('Input paths must stay relative to the runner workspace.')
    target = Path(root) / path
    if not target.resolve().is_relative_to(Path(root).resolve()) or not target.is_file():
        raise ScientificInputUnavailable(f'Input is missing or outside the runner workspace: {relative}')
    return target


def input_plan(request, root):
    request = request_document(request)
    p = request['parameters']
    from ..experiment import managed_methods, managed_inputs
    if request['operation'] in managed_methods.OPERATIONS:
        from . import managed_validation
        captured = managed_inputs.plan(request, root)
        validated = managed_validation.validate(request, root)
        result = {'request': request, 'root': str(Path(root).resolve()), 'inputSHA256': digest(captured),
                'models': validated['models'], 'effectiveConfig': validated['effectiveConfig'],
                'resumable': False, 'compute': managed_methods.METHODS[request['operation']].compute if request['operation'] in managed_methods.METHODS else 'cpu'}
        if request['operation'] == 'jlens-fit':
            from ..experiment.jlens_fit_review import review
            result['fittingReview'] = review(validated['effectiveConfig'])
        return result
    if request['operation'] == 'battery':
        from ..experiment import battery_run
        workspace_file(root, p['batteryFile'])
        for slot in battery_run.parse_agents(p['agents']):
            if slot.kind == battery_run.KIND_ARTIFACT:
                workspace_file(root, slot.reference)
        spec, agents = battery_run.preflight(p['batteryFile'], p['agents'], root=root,
            model_id=p.get('modelID'), revision=p.get('revision'), alpha_units=p['alphaUnits'])
        identities = [a.identity for a in agents]
        if any(not a.revision or not re.fullmatch(r'[0-9a-fA-F]{40}', a.revision) for a in agents):
            raise ScientificRefusal('Every remote battery agent needs a pinned model revision.')
        # Norm-denominated interventions also depend on sidecar bytes. The
        # battery owner's identities already bind tensors and variant JSON;
        # pin sidecars here so a valid-but-different norm table changes the plan.
        from ..experiment import paths
        references = {identity['vectorArtifactID'] for identity in identities if identity.get('vectorArtifactID')}
        references.update(injection['vectorArtifactID'] for identity in identities
                          for injection in identity.get('injections', []) if injection.get('vectorArtifactID'))
        sidecars = {reference: hashlib.sha256(Path(paths.resolve_artifact(reference, root) + '.json').read_bytes()).hexdigest()
                    for reference in sorted(references)}
        material = {'batterySHA256': spec.digest, 'agents': identities, 'vectorSidecars': sidecars}
        models = [{'modelID': a.model_id, 'revision': a.revision} for a in agents]
    else:
        from ..experiment import extract_stability
        try:
            prepared = extract_stability.preflight(p['experiment'], p['concept'], root=root,
                                                   resamples=p['resamples'], fraction=p['fraction'])
        except extract_stability.ExtractStabilityError as exc:
            raise ScientificRefusal(str(exc)) from exc
        manifest = prepared['manifest']
        if not manifest.model_revision or not re.fullmatch(r'[0-9a-fA-F]{40}', manifest.model_revision):
            raise ScientificRefusal('Pin the model revision before remote diagnostics.')
        material = {'manifest': manifest.raw, 'positive': prepared['positive'],
                    'negative': prepared['negative'], 'stimulus': prepared['stimulus_provenance']}
        models = [{'modelID': manifest.model_id, 'revision': manifest.model_revision}]
    return {'request': request, 'root': str(Path(root).resolve()),
            'inputSHA256': digest(material), 'models': models, 'resumable': False}


def plan(request, profile):
    with submitting():
        if server_role(profile) == 'gpu-session':
            raise ScientificRefusal('Submit through the controller; session workers own no durable job queue.')
        execution_root = profile.root
        staged_digest = None
        if isinstance(request, dict) and set(request) == {'inputBundleSHA256'}:
            from . import diagnostic_transport
            staged_digest = request['inputBundleSHA256']
            request, execution_root = diagnostic_transport.resolve(staged_digest, profile)
        if isinstance(request, dict) and request.get('operation') == 'optvec-campaign' and not staged_digest:
            raise ScientificRefusal('Managed campaigns require an isolated staged input bundle.')
        try:
            result = input_plan(request, execution_root)
        except (ValueError, OSError) as exc:
            from ..experiment import diagnostic_archives
            path_failure = isinstance(exc, (ScientificInputUnavailable, diagnostic_archives.PathRefusal, FileNotFoundError, NotADirectoryError))
            if path_failure and not staged_digest and isinstance(request, dict) and 'parameters' in request:
                error = ScientificRefusal(str(exc) + ' If these inputs were staged from another workspace, plan the staging response request, not the original workspace request.')
                error.repair_action = 'Use the localRequestPath returned by science-stage, or its request object {"inputBundleSHA256": "<staged digest>"}, with science-plan on this controller.'
                raise error from exc
            raise
        if staged_digest:
            result['inputBundleSHA256'] = staged_digest
        from . import job_ownership
        host, pid = job_ownership.current_owner()
        result['controller'] = {'host': host, 'pid': pid, 'metadataRoot': str(Path(profile.metadata_root).resolve())}
        result['memoryFit'] = 'notChecked'
        selected_executor = 'local' if result.get('compute') == 'cpu' else profile.executor
        if server_role(profile) == 'controller' and result.get('compute') != 'cpu' and selected_executor != 'slurm':
            raise ScientificRefusal('A controller must submit this GPU work through Slurm.')
        result['executor'] = selected_executor
        resources = SlurmResources.from_env(job_name='scientific-diagnostic')
        resources.auto_resubmit = False  # These owners have no checkpoint/resume protocol.
        result['resources'] = asdict(resources) if selected_executor == 'slurm' else {'executor': 'local'}
        if selected_executor == 'slurm':
            # Render without writing to apply the same required-header and GRES
            # gates before any submission directory or allocation exists.
            result['schedulerPreview'] = render_slurm_script(JobBundle(
                bundle_dir='<submission>', command=['<diagnostic-child>'], env={},
                resources=SlurmResources(**result['resources']), stdout_path='<submission>/slurm-%j.out',
                stderr_path='<submission>/slurm-%j.err', script_path='<submission>/run.sbatch',
                manifest_path='<submission>/bundle.json'))
        result['changed'] = False
        result['planSHA256'] = digest(result)
        return result


def write_json(path, document):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def submit(request, expected, *, profile, jobs, registry=None):
    with submitting():
        reviewed = plan(request, profile)
        if reviewed['planSHA256'] != expected:
            raise ScientificRefusal('The request, inputs, root or resource plan changed after review; nothing was submitted.')
        from ..experiment import paths
        directory = Path(paths.make_unique_run_directory('scientific-submission', profile.root))
        records = directory / 'records'
        records.mkdir()
        packet = directory / 'request.json'
        write_json(packet, reviewed)
        base = {'scientificPlan': reviewed, 'recordsDirectory': str(records),
                'submissionDirectory': str(directory), 'resumable': False}

        def command(job_id):
            return [sys.executable, '-m', 'steerlab_server.api.scientific_execution',
                    str(packet), job_id, str(records / (job_id + '.json')), reviewed['planSHA256']]

        if reviewed['executor'] == 'local':
            def work(job):
                job.result = dict(base)
                jobs.store.update(job)
                if registry is not None and reviewed.get('compute') != 'cpu':
                    registry.unload_all()
                proc = LocalExecutor().run(command(job.id), log=job.log,
                                            should_cancel=lambda: job.cancelled)
                record = records / (job.id + '.json')
                if record.exists():
                    job.result.update(json.loads(record.read_text()).get('result', {}))
                if job.cancelled:
                    return job.result
                if proc.returncode:
                    raise RuntimeError(proc.stderr.strip() or f'Diagnostic child exited {proc.returncode}; inspect its retained submission record.')
                return job.result
            job = jobs.submit('science:' + reviewed['request']['operation'], work,
                              requested_resources={'executor': 'local', 'resumable': False})
        else:
            job_id = uuid.uuid4().hex[:12]
            executor = SlurmExecutor(profile)
            bundle = executor.create_bundle(str(directory / 'slurm'), command(job_id),
                env={'STEERLAB_JOB_ID': job_id}, resources=SlurmResources(**reviewed['resources']),
                metadata={'kind': 'scientificDiagnostic', 'recordsDirectory': str(records)})
            base['slurmBundle'] = bundle.to_dict()
            # Record the submission intention before sbatch. An uncertain reply
            # stays visible for operator reconciliation and is never retried here.
            base['schedulerSubmissionName'] = 'science-' + job_id
            job = jobs.record_external('science:' + reviewed['request']['operation'],
                status='submitting', executor='slurm', job_id=job_id, result=base,
                requested_resources={**reviewed['resources'], 'autoResubmit': False})
            try:
                job.executor_job_id = executor.submit(bundle, job_name=base['schedulerSubmissionName'])
                job.status = 'submitted'
                job.started_at = time.time()
            except Exception as exc:
                job.status = 'parked'
                job.finished_at = time.time()
                job.result = {**base, 'parked': True, 'reason': 'Scheduler submission outcome requires inspection by schedulerSubmissionName before any retry.', 'submissionError': str(exc)}
            jobs.store.update(job)
        return {'jobId': job.id, **base, 'status': job.status}


def execute_packet(packet, job_id, record, expected_plan_sha256=None):
    started = time.time()
    reviewed = json.loads(Path(packet).read_text())
    root = reviewed['root']
    prior_environment = {key: os.environ.get(key) for key in ('STEERLAB_ROOT', 'STEERLAB_RUN_ROOT')}
    prior_cwd = os.getcwd()
    os.environ['STEERLAB_ROOT'] = root
    os.environ['STEERLAB_RUN_ROOT'] = str(Path(root) / 'runs')
    os.chdir(root)
    directory = Path(packet).parent
    result = {'scientificPlan': reviewed, 'resumable': False,
              'recordsDirectory': str(Path(record).parent), 'submissionDirectory': str(directory)}
    bundle_manifest = directory / 'slurm/bundle.json'
    if bundle_manifest.exists():
        result['slurmBundle'] = json.loads(bundle_manifest.read_text())['bundle']
        result['schedulerSubmissionName'] = 'science-' + job_id
    executor_identity = {'executor': reviewed['executor']}
    if reviewed['executor'] == 'slurm' and os.environ.get('SLURM_JOB_ID'):
        executor_identity['executorJobID'] = os.environ['SLURM_JOB_ID']
    status, error = 'failed', None
    def created(directory):
        result.update(runDirectory=directory, partial=True)
        # No status override: a scheduler cancellation remains cancelled when
        # this in-progress record is folded after the child was killed.
        write_json(Path(record), {'id': job_id, 'result': result, **executor_identity})
    try:
        actual = digest({key: value for key, value in reviewed.items() if key != 'planSHA256'})
        if actual != reviewed['planSHA256'] or (expected_plan_sha256 is not None and actual != expected_plan_sha256):
            raise ScientificRefusal('The recorded diagnostic plan changed after submission; nothing was executed.')
        if reviewed.get('inputBundleSHA256'):
            from . import diagnostic_transport
            staged_request = diagnostic_transport.verify_inputs(root, reviewed['inputBundleSHA256'])
            if staged_request != reviewed['request']:
                raise ScientificRefusal('Staged archive request differs from the queued plan.')
        current = input_plan(reviewed['request'], root)
        if current['inputSHA256'] != reviewed['inputSHA256']:
            raise ScientificRefusal('Inputs changed while queued; diagnostic refused before model loading.')
        p = current['request']['parameters']
        from ..experiment import managed_methods
        if current['request']['operation'] in managed_methods.OPERATIONS:
            # The isolated child owns process-global workspace resolution used
            # by legacy numerical owners. No shared server root is retargeted.
            result.update(partial=True, outputRoot=str(Path(root) / 'runs'))
            write_json(Path(record), {'id': job_id, 'result': result, **executor_identity})
            report = managed_methods.execute(current['request']['operation'], p['config'], root, log=print, on_run_created=created)
            result.pop('partial', None)
            result.update(report)
            if report.get('campaignDirectory'):
                result['runDirectory'] = report['campaignDirectory']
        elif current['request']['operation'] == 'battery':
            from ..experiment import battery_run
            report = battery_run.execute(p['batteryFile'], p['agents'], root=root,
                model_id=p.get('modelID'), revision=p.get('revision'), alpha_units=p['alphaUnits'],
                dtype=p.get('dtype'), device=p.get('device'), log=print, on_run_created=created)
            result.pop('partial', None)
            result.update(runDirectory=report['runDirectory'], reportPath=str(Path(report['runDirectory']) / 'battery-report.json'))
        else:
            from ..experiment import extract_stability
            report = extract_stability.run(p['experiment'], p['concept'], root=root,
                resamples=p['resamples'], fraction=p['fraction'], seed=int(p['seed']), order_shuffles=p['orderShuffles'],
                dtype=p.get('dtype'), device=p.get('device'), log=print)
            result.update(diagnosticDirectory=report['directory'], reportPath=report['path'])
        status = 'succeeded'
    except Exception as exc:
        error = str(exc)
        print(error, file=sys.stderr)
    finally:
        write_json(Path(record), {'id': job_id, 'kind': 'science:' + reviewed['request']['operation'],
            'status': status, 'result': result, 'error': error, 'finishedAt': time.time(), **executor_identity,
            'elapsedSeconds': time.time() - started})
    os.chdir(prior_cwd)
    for key, value in prior_environment.items():
        if value is None: os.environ.pop(key, None)
        else: os.environ[key] = value
    return 0 if status == 'succeeded' else 70


if __name__ == '__main__':
    raise SystemExit(execute_packet(*sys.argv[1:]))
