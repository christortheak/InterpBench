"""A lens fitted on a site stays there (public issue #14).

When a fit or a merge ends, the engine converts its output as an import would
and keeps those bytes in its lens library, under the hash the researcher's
later import records. The fit is never at stake: the copy is made after the
fit's own record is written, reads the run folder without writing to it, and a
failure is only reported. A request can then refer to the imported lens, and
the site already holds what it needs.
"""
import json
import shutil
from pathlib import Path

import pytest
from test_jlens_fit import fitting, run, Tiny, torch  # noqa: F401 - fixture; skips without the fitting extra

from steerlab_server.api import diagnostic_transport as transport, scientific_execution
from steerlab_server.api.profile import ServerProfile
from steerlab_server.experiment import artifact_imports, diagnostic_archives as archives, diagnostic_inputs
from steerlab_server.experiment import jlens_assessment as assessment, jlens_fit_review
from steerlab_server.jlens import lens_library, lens_store
from test_lens_reference_requests import library_client
from test_lens_library import push


@pytest.fixture
def site(fitting, monkeypatch):
    root, config = fitting
    monkeypatch.setattr(jlens_fit_review, 'cached_config',
                        lambda *args: ({'hidden_size': 2, 'num_hidden_layers': 3}, 'd' * 64))
    monkeypatch.setenv('STEERLAB_ROOT', str(root))
    monkeypatch.setenv('STEERLAB_METADATA_ROOT', str(root / '.steerlab'))
    monkeypatch.setenv('STEERLAB_SERVER_ROLE', 'workstation')
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'local')
    for name in ('STEERLAB_LENS_LIBRARY_ROOT', 'SLURM_JOB_ID', 'STEERLAB_RUN_ROOT'):
        monkeypatch.delenv(name, raising=False)
    return root, config, ServerProfile.from_env()


def execute(root, plan, name):
    packet, record = root / f'{name}-packet.json', root / f'{name}-record.json'
    packet.write_text(json.dumps(plan))
    return scientific_execution.execute_packet(packet, name, record), json.loads(record.read_bytes())


def files_of(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in sorted(Path(directory).rglob('*')) if path.is_file()}


def imported(description, workspace):
    """What the researcher does with a fit: review its description, then import it."""
    review = artifact_imports.inspect_source(description, workspace)
    return lens_store.resolve(artifact_imports.publish(description, workspace, review['planSHA256'])['lensID'], str(workspace))


def test_a_fit_is_kept_under_the_hash_its_import_records_and_a_request_can_refer_to_it(site, monkeypatch, tmp_path_factory):
    root, config, profile = site
    library = root / '.steerlab/lens-library'
    seen = {}
    keep = lens_library.keep_fitted

    def observed(run_directory):
        seen['before'] = files_of(run_directory)         # the run folder as the fit left it
        return keep(run_directory)
    monkeypatch.setattr(lens_library, 'keep_fitted', observed)
    code, output = execute(root, scientific_execution.plan({'operation': 'jlens-fit', 'parameters': {'config': config}}, profile), 'fit')
    assert code == 0 and output['status'] == 'succeeded' and output['result']['promptsFitted'] == 3
    kept = output['result']['lensLibrary']
    assert kept['kept'] is True and kept['reason'] is None and len(kept['convertedSHA256']) == 64
    fitted = Path(output['result']['runDirectory'])
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


def test_a_copy_that_cannot_be_made_never_fails_the_fit_or_touches_its_files(site, monkeypatch):
    root, config, profile = site
    library = root / '.steerlab/lens-library'
    request = {'operation': 'jlens-fit', 'parameters': {'config': config}}
    seen, publish = {}, lens_library._publish

    def full(base, staged, sha, origin):
        seen['run'] = files_of(next(path.parent for path in (root / 'runs').glob('*/COMPLETED')))
        raise OSError(28, 'No space left on device')
    monkeypatch.setattr(lens_library, '_publish', full)
    code, output = execute(root, scientific_execution.plan(request, profile), 'full')
    assert code == 0 and output['status'] == 'succeeded' and output['error'] is None
    kept = output['result']['lensLibrary']
    assert kept['kept'] is False and kept['convertedSHA256'] is None
    assert 'No space left on device' in kept['reason'] and 'the fit itself is unaffected' in kept['reason']
    fitted = Path(output['result']['runDirectory'])
    assert files_of(fitted) == seen['run'] and (fitted / 'COMPLETED').read_text() == 'jlens-fit\n'
    assert [path.name for path in library.iterdir()] == ['.incoming'] and not list((library / '.incoming').iterdir())
    monkeypatch.setattr(lens_library, '_publish', publish)

    # An engine with no library says so, and the fit is as good as ever.
    plan = scientific_execution.plan(request, profile)
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    code, output = execute(root, plan, 'none')
    assert code == 0 and output['status'] == 'succeeded'
    assert output['result']['lensLibrary'] == {
        'kept': False, 'convertedSHA256': None,
        'reason': 'This engine has no lens library, so the fitted lens stays in its run folder only.'}
    monkeypatch.delenv('SLURM_JOB_ID')

    # A fit an import would refuse is not kept: here, one whose description names a tensor it lacks.
    broken = root / 'runs/broken-fit'
    shutil.copytree(fitted, broken)
    description = json.loads((broken / 'artifact-description.json').read_bytes())
    description['lens']['layers']['0'] = 'layer_9'
    (broken / 'artifact-description.json').write_text(json.dumps(description))
    before = files_of(broken)
    refused = lens_library.keep_fitted(broken)
    assert refused['kept'] is False and 'layer_9' in refused['reason'] and files_of(broken) == before


def test_a_merge_is_kept_and_a_shard_of_a_round_is_not(site):
    from steerlab_server.experiment import jlens_round
    root, config, profile = site
    library = root / '.steerlab/lens-library'
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
