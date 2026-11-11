"""A lens fitted on a site stays there (public issue #14).

When a fit or a merge ends, the engine converts its output as an import would
and keeps those bytes in its lens library, under the hash the researcher's
later import records. The fit is never at stake: the copy is made after the
fit's own record is written, reads the run folder without writing to it, and a
failure is only reported. A request can then refer to the imported lens, and
the site already holds what it needs.

The site here declares its library, as a real site does, outside the workspace
it serves. Which engines have a library when none is declared is decided in
one place (`ServerProfile.lens_library_root`): a runner that runs its work on
its own machine does, and keeps its fits; an engine that serves a researcher's
own workspace does not, and its fits stay in their run folders only.
"""
import json
import os
import shutil
import time
from pathlib import Path

import pytest
from test_jlens_fit import fitting, run, Tiny, torch  # noqa: F401 - fixture; skips without the fitting extra

from steerlab_server.api import diagnostic_transport as transport, scientific_execution
from steerlab_server.api.profile import ServerProfile
from steerlab_server.experiment import artifact_imports, diagnostic_archives as archives, diagnostic_inputs
from steerlab_server.experiment import jlens_assessment as assessment, jlens_fit_review
from steerlab_server.jlens import lens_library, lens_store
from test_lens_reference_requests import library_client
from test_lens_library import idle, push


@pytest.fixture
def site(fitting, monkeypatch, tmp_path_factory):
    root, config = fitting
    library = tmp_path_factory.mktemp('site-storage') / 'lens-library'
    monkeypatch.setattr(jlens_fit_review, 'cached_config',
                        lambda *args: ({'hidden_size': 2, 'num_hidden_layers': 3}, 'd' * 64))
    monkeypatch.setenv('STEERLAB_ROOT', str(root))
    monkeypatch.setenv('STEERLAB_METADATA_ROOT', str(root / '.steerlab'))
    monkeypatch.setenv('STEERLAB_SERVER_ROLE', 'workstation')
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'local')
    monkeypatch.setenv('STEERLAB_LENS_LIBRARY_ROOT', str(library))
    for name in ('SLURM_JOB_ID', 'STEERLAB_RUN_ROOT', 'STEERLAB_SERVICE_ROLE'):
        monkeypatch.delenv(name, raising=False)
    return root, config, ServerProfile.from_env(), library


def execute(root, plan, name):
    packet, record = root / f'{name}-packet.json', root / f'{name}-record.json'
    packet.write_text(json.dumps(plan))
    return scientific_execution.execute_packet(packet, name, record), json.loads(record.read_bytes())


def fit(root, config, profile, name='fit'):
    request = {'operation': 'jlens-fit', 'parameters': {'config': config}}
    code, output = execute(root, scientific_execution.plan(request, profile), name)
    assert code == 0 and output['status'] == 'succeeded' and output['error'] is None, output
    return output['result']


def files_of(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in sorted(Path(directory).rglob('*')) if path.is_file()}


def imported(description, workspace):
    """What the researcher does with a fit: review its description, then import it."""
    review = artifact_imports.inspect_source(description, workspace)
    return lens_store.resolve(artifact_imports.publish(description, workspace, review['planSHA256'])['lensID'], str(workspace))


def test_a_fit_is_kept_under_the_hash_its_import_records_and_a_request_can_refer_to_it(site, monkeypatch, tmp_path_factory):
    root, config, profile, library = site
    seen = {}
    keep = lens_library.keep_fitted

    def observed(run_directory):
        seen['before'] = files_of(run_directory)         # the run folder as the fit left it
        return keep(run_directory)
    monkeypatch.setattr(lens_library, 'keep_fitted', observed)
    result = fit(root, config, profile)
    assert result['promptsFitted'] == 3
    kept = result['lensLibrary']
    assert kept['kept'] is True and kept['reason'] is None and len(kept['convertedSHA256']) == 64
    fitted = Path(result['runDirectory'])
    assert files_of(fitted) == seen['before']            # nothing in the run folder was written
    entry = json.loads((library / kept['convertedSHA256'] / 'entry.json').read_bytes())
    assert entry['origin'] == 'fitted' and entry['convertedSHA256'] == kept['convertedSHA256']
    assert not (root / 'runs/jlens-lenses').exists()     # bytes only: importing is still the researcher's step

    # On the researcher's machine: the fit's folder comes home and is imported there.
    theirs = tmp_path_factory.mktemp('researcher')
    shutil.copytree(fitted, theirs / 'runs' / fitted.name)
    shutil.copy(root / 'corpus.jsonl', theirs / 'corpus.jsonl')
    reference = imported(theirs / 'runs' / fitted.name / 'artifact-description.json', theirs)
    assert reference.converted.sha256 == kept['convertedSHA256']      # the library's key is the import's hash
    assert (library / kept['convertedSHA256'] / 'jacobians.safetensors').read_bytes() == (theirs / reference.converted.path).read_bytes()

    # A second lens, fitted on the researcher's machine, is sent once.
    candidate = imported(run(theirs, {**config, 'maxPrompts': 1})['artifactDescription'], theirs)
    assert candidate.converted.sha256 != reference.converted.sha256
    assert push(library_client(), theirs / candidate.converted.path)[2].json()['held'] is True
    held_out = theirs / 'assessment.jsonl'
    held_out.write_text(''.join(json.dumps({'id': str(i), 'text': text}) + '\n' for i, text in enumerate(['24', '2', '30', '25'])))
    settings = assessment.AssessmentConfig.from_dict({
        **{key: config[key] for key in ('modelID', 'revision', 'dtype', 'device')},
        'corpus': {'path': held_out.name, 'sha256': archives.file_hash(held_out)},
        'referenceLensID': reference.lensID, 'candidateLensID': candidate.lensID,
        'maxPrompts': 4, 'maxSeqLen': 32, 'skipFirst': 2, 'maxPositionsPerRow': 20, 'topK': 2})
    monkeypatch.setattr(Tiny, 'unembed', lambda self, h: torch.tanh(h / 30) @ torch.tensor(
        [[1., .3, -.2, .7], [-.2, .8, .5, -.1]]), raising=False)
    direct = json.loads(Path(assessment.assess(settings, root=theirs)['reportPath']).read_bytes())

    request = {'operation': 'jlens-fit-assess', 'parameters': {'config': settings.to_dict()}}
    referred = diagnostic_inputs.plan(request, theirs, lens_reference=True)
    assert referred['libraryLenses'] == sorted(
        ({'lensID': lens.lensID, 'convertedSHA256': lens.converted.sha256, 'bytes': (theirs / lens.converted.path).stat().st_size}
         for lens in (reference, candidate)), key=lambda lens: lens['lensID'])
    assert not [e['path'] for e in referred['files'] if e['path'].endswith('jacobians.safetensors')]
    assert sum(e['path'].endswith('/lens.json') for e in referred['files']) == 2
    assert sum(e['path'].endswith('/import-receipt.json') for e in referred['files']) == 2
    packed = diagnostic_inputs.package(request, theirs, theirs / 'referred.tar.gz', referred['planSHA256'], lens_reference=True)
    (root / 'runs').mkdir(exist_ok=True)
    shutil.copy(packed['bundlePath'], root / 'runs/referred.tar.gz')
    staged = transport.stage('referred.tar.gz', packed['bundleSha256'], profile)
    capsule = Path(staged['executionRoot'])
    assert not list(capsule.rglob('jacobians.safetensors'))
    code, output = execute(root, scientific_execution.plan(staged['request'], profile), 'assess')
    assert code == 0 and output['status'] == 'succeeded', output.get('error')
    assert 'lensLibrary' not in output['result']         # only a fit is kept
    report = json.loads(Path(output['result']['reportPath']).read_bytes())
    assert report['layers'] == direct['layers'] and report['layers']['0']['betweenLenses']['positions'] > 0
    assert report['comparisons'] == direct['comparisons']


def test_an_assessment_stops_when_the_library_loses_a_lens_and_never_reads_the_path_its_record_names(
        site, monkeypatch, tmp_path_factory):
    """Review 2, finding 1, as it was reproduced. The candidate's record names an absolute path in the importer's
    layout, as the published importer writes one, and on the engine's host that path holds the REFERENCE lens's
    bytes. An assessment resolves a lens again for every layer it loads. With the candidate's entry lost after
    the job verified its inputs, it used to read that path and report two identical lenses."""
    from steerlab_server.experiment import managed_methods
    from steerlab_server.jlens import artifact_paths
    root, config, profile, library = site
    theirs = tmp_path_factory.mktemp('researcher')
    shutil.copy(root / 'corpus.jsonl', theirs / 'corpus.jsonl')
    reference = imported(run(theirs, config)['artifactDescription'], theirs)
    candidate = imported(run(theirs, {**config, 'maxPrompts': 1})['artifactDescription'], theirs)
    assert candidate.converted.sha256 != reference.converted.sha256
    absolute = theirs / candidate.converted.path
    record = theirs / 'runs/jlens-lenses' / candidate.lensID / 'lens.json'
    document = json.loads(record.read_bytes())
    document['converted']['path'] = str(absolute)
    record.write_text(json.dumps(document))
    for lens in (reference, candidate):
        assert push(library_client(), theirs / 'runs/jlens-lenses' / lens.lensID / 'jacobians.safetensors')[2].json()['held']
    held_out = theirs / 'assessment.jsonl'
    held_out.write_text(''.join(json.dumps({'id': str(i), 'text': text}) + '\n' for i, text in enumerate(['24', '2', '30', '25'])))
    settings = assessment.AssessmentConfig.from_dict({
        **{key: config[key] for key in ('modelID', 'revision', 'dtype', 'device')},
        'corpus': {'path': held_out.name, 'sha256': archives.file_hash(held_out)},
        'referenceLensID': reference.lensID, 'candidateLensID': candidate.lensID,
        'maxPrompts': 4, 'maxSeqLen': 32, 'skipFirst': 2, 'maxPositionsPerRow': 20, 'topK': 2})
    monkeypatch.setattr(Tiny, 'unembed', lambda self, h: torch.tanh(h / 30) @ torch.tensor(
        [[1., .3, -.2, .7], [-.2, .8, .5, -.1]]), raising=False)
    honest = json.loads(Path(assessment.assess(settings, root=theirs)['reportPath']).read_bytes())

    request = {'operation': 'jlens-fit-assess', 'parameters': {'config': settings.to_dict()}}
    referred = diagnostic_inputs.plan(request, theirs, lens_reference=True)
    packed = diagnostic_inputs.package(request, theirs, theirs / 'referred.tar.gz', referred['planSHA256'], lens_reference=True)
    (root / 'runs').mkdir(exist_ok=True)
    shutil.copy(packed['bundlePath'], root / 'runs/referred.tar.gz')
    staged = transport.stage('referred.tar.gz', packed['bundleSha256'], profile)
    # On this host, the path the candidate's record names now holds the reference lens.
    shutil.copy(theirs / 'runs/jlens-lenses' / reference.lensID / 'jacobians.safetensors', absolute)

    plan = scientific_execution.plan(staged['request'], profile)
    code, output = execute(root, plan, 'held')
    assert code == 0 and output['status'] == 'succeeded', output.get('error')
    report = json.loads(Path(output['result']['reportPath']).read_bytes())
    assert report['layers'] == honest['layers'] and report['comparisons'] == honest['comparisons']

    real, seen = managed_methods.execute, {}

    def after_the_job_verified_its_inputs(operation, settings_, execution_root, **kwargs):
        seen['pinned'] = artifact_paths.pinned_library_lenses()
        shutil.rmtree(library / candidate.converted.sha256)          # the library loses the candidate
        return real(operation, settings_, execution_root, **kwargs)
    monkeypatch.setattr(managed_methods, 'execute', after_the_job_verified_its_inputs)
    code, output = execute(root, plan, 'lost')
    assert seen['pinned'] == {reference.lensID: reference.converted.sha256, candidate.lensID: candidate.converted.sha256}
    assert code == 70 and output['status'] == 'failed', output
    assert output['error'] == (
        f"The lens library no longer holds the lens '{candidate.lensID}' (converted SHA-256 "
        f"{candidate.converted.sha256}), which this job's plan took from it. The job stopped, and read the lens from "
        f'nowhere else. Send the lens again with `remote lens-push {candidate.lensID}` (or `runner lens-push`), then '
        'run the request again.')
    assert 'reportPath' not in output['result'] and artifact_paths.pinned_library_lenses() == {}


def test_a_copy_that_cannot_be_made_never_fails_the_fit_or_touches_its_files(site, monkeypatch):
    root, config, profile, library = site
    seen, publish = {}, lens_library._publish

    def full(base, staged, sha, origin):
        seen['run'] = files_of(next(path.parent for path in (root / 'runs').glob('*/COMPLETED')))
        raise OSError(28, 'No space left on device', str(staged))
    monkeypatch.setattr(lens_library, '_publish', full)
    result = fit(root, config, profile, 'full')
    kept = result['lensLibrary']
    assert kept['kept'] is False and kept['convertedSHA256'] is None
    assert "storage is full" in kept['reason'] and 'the fit itself is unaffected' in kept['reason']
    assert str(library) not in kept['reason'] and 'Errno' not in kept['reason']      # no path of the engine's
    fitted = Path(result['runDirectory'])
    assert files_of(fitted) == seen['run'] and (fitted / 'COMPLETED').read_text() == 'jlens-fit\n'
    assert [path.name for path in library.iterdir()] == ['.incoming'] and not list((library / '.incoming').iterdir())
    monkeypatch.setattr(lens_library, '_publish', publish)

    # A fit an import would refuse is not kept: here, one whose description names a tensor it lacks.
    broken = root / 'runs/broken-fit'
    shutil.copytree(fitted, broken)
    description = json.loads((broken / 'artifact-description.json').read_bytes())
    description['lens']['layers']['0'] = 'layer_9'
    (broken / 'artifact-description.json').write_text(json.dumps(description))
    before = files_of(broken)
    refused = lens_library.keep_fitted(broken)
    assert refused['kept'] is False and 'layer_9' in refused['reason'] and files_of(broken) == before

    # Not enough room: said, and nothing is converted or written.
    with monkeypatch.context() as patched:
        patched.setattr(lens_library.shutil, 'disk_usage', lambda path: type('usage', (), {'free': 10})())
        patched.setattr(artifact_imports, 'convert_lens', lambda *args: pytest.fail('converted with no room for it'))
        short = lens_library.keep_fitted(fitted)
    assert short['kept'] is False and '10 bytes free' in short['reason'] and 'the fit itself is unaffected' in short['reason']
    assert not list((library / '.incoming').iterdir())


def test_which_engines_keep_a_fit_and_what_the_others_say(site, monkeypatch, tmp_path_factory):
    root, config, profile, library = site
    request = {'operation': 'jlens-fit', 'parameters': {'config': config}}
    plan = scientific_execution.plan(request, profile)
    default = root / '.steerlab/lens-library'
    none_set = 'No folder is set for a lens library on this engine, so the fitted lens stays in its run folder only.'

    def block(name):
        code, output = execute(root, plan, name)
        assert code == 0 and output['status'] == 'succeeded'
        return output['result']['lensLibrary']

    def reason(name):
        kept = block(name)
        assert kept['kept'] is False and kept['convertedSHA256'] is None
        return kept['reason']
    # No folder is declared from here on. A workbench (the default role) serves a researcher's own workspace,
    # which already holds the fit: it has no library, so no second copy is made, and none is created for it.
    monkeypatch.delenv('STEERLAB_LENS_LIBRARY_ROOT')
    assert reason('workbench') == none_set
    monkeypatch.setenv('STEERLAB_SERVICE_ROLE', 'workbench')
    assert reason('workbench-named') == none_set
    # A runner inside a scheduler job that was given no folder, and one that hands its work to Slurm.
    monkeypatch.setenv('STEERLAB_SERVICE_ROLE', 'runner')
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    assert reason('scheduler') == none_set
    monkeypatch.delenv('SLURM_JOB_ID')
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'slurm')
    assert reason('slurm') == none_set
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'local')
    # Told it has none (as a launcher tells its child): none, even on a runner.
    monkeypatch.setenv('STEERLAB_LENS_LIBRARY_ROOT', '')
    assert reason('told') == none_set
    assert not default.exists() and not library.exists()
    # A runner that runs its work on this machine keeps the fit beside its own records. That folder lies
    # inside the root the fit ran in, and the fit is kept all the same: the role decides, not the location.
    monkeypatch.delenv('STEERLAB_LENS_LIBRARY_ROOT')
    assert ServerProfile.from_env().lens_library_root == str(default)
    kept = block('runner')
    assert kept['kept'] is True and kept['reason'] is None
    assert json.loads((default / kept['convertedSHA256'] / 'entry.json').read_bytes())['origin'] == 'fitted'
    assert not library.exists()
    # A folder is set, and cannot be written from where the fit ran: said as that, on either role.
    blocked = tmp_path_factory.mktemp('blocked') / 'a-file'
    blocked.write_text('not a folder')
    monkeypatch.setenv('STEERLAB_LENS_LIBRARY_ROOT', str(blocked / 'lenses'))
    for role in ('runner', 'workbench'):
        monkeypatch.setenv('STEERLAB_SERVICE_ROLE', role)
        assert reason('unwritable-' + role) == (
            'The lens library folder set for this engine cannot be written from the node this fit ran on, so the '
            'fitted lens stays in its run folder only.')
    # A declared folder is used on a workbench too.
    monkeypatch.setenv('STEERLAB_LENS_LIBRARY_ROOT', str(library))
    declared = block('declared')
    assert declared == {'kept': True, 'convertedSHA256': kept['convertedSHA256'], 'reason': None}
    assert (library / kept['convertedSHA256'] / 'jacobians.safetensors').is_file()


def test_a_relative_library_folder_names_one_folder_wherever_the_job_changes_directory(site, monkeypatch, tmp_path_factory):
    root, config, profile, _ = site
    home = tmp_path_factory.mktemp('engine-directory')
    monkeypatch.chdir(home)
    monkeypatch.setenv('STEERLAB_LENS_LIBRARY_ROOT', 'lenses')          # relative to where the engine was started
    profile = ServerProfile.from_env()
    assert profile.lens_library_root == os.path.join(os.getcwd(), 'lenses')
    # The job changes directory to its execution root; the library it writes to is still the engine's.
    kept = fit(root, config, profile)['lensLibrary']
    assert kept['kept'] is True, kept
    assert (Path(os.getcwd()) / 'lenses' / kept['convertedSHA256'] / 'jacobians.safetensors').is_file()
    assert not (root / 'lenses').exists() and os.getcwd() == str(home.resolve())


def test_a_merge_is_kept_and_a_shard_of_a_round_is_not(site):
    from steerlab_server.experiment import jlens_round
    root, config, profile, library = site
    path = root / 'request.json'
    path.write_text(json.dumps({'operation': 'jlens-fit', 'parameters': {'config': config}}))
    shards = jlens_round.prepare(jlens_round.RoundConfig.from_dict(dict(
        modelID=config['modelID'], revision=config['revision'],
        fittingRequest={'path': 'request.json', 'sha256': archives.file_hash(path)}, shards=2)), root)['shards']
    fits = []
    for index, shard in enumerate(shards):
        plan = scientific_execution.plan(shard, profile, round_submission={'parentJobID': 'round', 'shardIndex': index})
        code, output = execute(root, plan, f'shard-{index}')
        assert code == 0 and output['status'] == 'succeeded'
        kept = output['result']['lensLibrary']
        assert kept['kept'] is False and kept['convertedSHA256'] is None and 'one shard of a fitting round' in kept['reason']
        fits.append(Path(output['result']['runDirectory']).relative_to(root).as_posix())
    assert not library.exists()                           # a slice of a lens is not a lens
    merge = {'operation': 'jlens-fit-merge', 'parameters': {'config': {'fits': fits}}}
    code, output = execute(root, scientific_execution.plan(merge, profile), 'merge')
    assert code == 0 and output['status'] == 'succeeded', output.get('error')
    kept = output['result']['lensLibrary']
    assert kept['kept'] is True
    assert json.loads((library / kept['convertedSHA256'] / 'entry.json').read_bytes())['origin'] == 'fitted'
    merged = imported(output['result']['artifactDescription'], root)
    assert merged.converted.sha256 == kept['convertedSHA256']
    assert sorted(p.name for p in library.iterdir() if p.name != '.incoming') == [kept['convertedSHA256']]


def test_a_purged_entry_is_replaced_when_the_lens_is_fitted_again(site):
    root, config, profile, library = site
    sha = fit(root, config, profile, 'first')['lensLibrary']['convertedSHA256']
    (library / sha / 'jacobians.safetensors').unlink()   # storage purged the large file and left the record
    idle(library / sha, lens_library.STALE_ENTRY_SECONDS + 1)
    again = fit(root, config, profile, 'second')['lensLibrary']
    assert again == {'kept': True, 'convertedSHA256': sha, 'reason': None}
    assert lens_library.held_file(sha) == str((library / sha / 'jacobians.safetensors').resolve())
    # A folder that changed a moment ago may be another publisher's: the fit is told, and is unaffected.
    (library / sha / 'jacobians.safetensors').unlink()
    busy = fit(root, config, profile, 'third')['lensLibrary']
    assert busy['kept'] is False and 'being published right now' in busy['reason'] and 'the fit itself is unaffected' in busy['reason']


def test_a_fit_is_not_kept_over_an_entry_a_newer_engine_wrote(site):
    root, config, profile, library = site
    sha = fit(root, config, profile, 'first')['lensLibrary']['convertedSHA256']
    record = json.loads((library / sha / 'entry.json').read_bytes())
    (library / sha / 'entry.json').write_text(json.dumps({**record, 'schemaVersion': 2}))
    (library / sha / 'jacobians.safetensors').rename(library / sha / 'jacobians.later')
    idle(library / sha, 2 * lens_library.STALE_STAGING_SECONDS)
    before = files_of(library / sha)
    newer = fit(root, config, profile, 'second')['lensLibrary']
    assert newer['kept'] is False and newer['convertedSHA256'] is None
    assert 'written by a newer version of the engine' in newer['reason'] and 'the fit itself is unaffected' in newer['reason']
    assert str(library) not in newer['reason']
    # The newer engine's entry is exactly as it was, and nothing was set aside or left behind.
    assert files_of(library / sha) == before and not list((library / '.incoming').iterdir())


def test_the_kept_lens_reaches_the_job_a_scheduler_ran(site):
    """On a cluster the child's record is folded into the job when the scheduler says it ended. The block that
    says whether the lens was kept arrives with it; and a job killed while the copy was being made, after its
    record said succeeded, is a succeeded job."""
    from steerlab_server.api.jobs import DurableJobStore, JobManager
    root, config, profile, library = site
    request = {'operation': 'jlens-fit', 'parameters': {'config': config}}
    records = root / 'records'
    records.mkdir()
    packet = root / 'packet.json'
    packet.write_text(json.dumps(scientific_execution.plan(request, profile)))
    assert scientific_execution.execute_packet(packet, 'fit-job', records / 'fit-job.json') == 0
    written = json.loads((records / 'fit-job.json').read_text())
    jobs = JobManager(store=DurableJobStore(str(root / 'jobs.sqlite')), sweep_orphans=False)
    jobs.record_external('science:jlens-fit', status='submitted', executor='slurm', job_id='fit-job',
                         result={'recordsDirectory': str(records)})
    jobs.reconcile(str(records), force=True)
    job = jobs.get('fit-job')
    assert job.status == 'succeeded'
    assert job.result['lensLibrary'] == written['result']['lensLibrary']
    assert job.result['lensLibrary']['kept'] is True and (library / job.result['lensLibrary']['convertedSHA256']).is_dir()
    # Killed during the copy: the record on disk is the one written before it, with no block; the
    # scheduler's own verdict on the process does not undo a fit that finished.
    before = {key: value for key, value in written['result'].items() if key != 'lensLibrary'}
    (records / 'killed-job.json').write_text(json.dumps({**written, 'id': 'killed-job', 'result': before}))
    jobs.record_external('science:jlens-fit', status='failed', executor='slurm', job_id='killed-job',
                         result={'recordsDirectory': str(records)})
    jobs.reconcile(str(records), force=True)
    killed = jobs.get('killed-job')
    assert killed.status == 'succeeded' and 'lensLibrary' not in killed.result
    assert killed.result['promptsFitted'] == 3 and time.time() >= killed.finished_at
