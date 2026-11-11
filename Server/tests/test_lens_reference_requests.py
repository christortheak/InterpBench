"""A request that refers to its lens, and a lens that stays where it was fitted
(public issues #13 and #14).

A request used to carry its lens's converted file. That still works, and a
request packaged without `--lens-reference` is, byte for byte, the request it
always was: the first test rebuilds its plan and archive description by hand
and never lets the engine look at a lens library. Packaged with the flag, a
request names each lens by hash, and the engine it is staged on reads the bytes
from its lens library: checked by name at staging, by existence and size at
admission, and by hash once on the queued job.
"""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

torch = pytest.importorskip('torch')

from steerlab_server.api import diagnostic_transport as transport, scientific_execution
from steerlab_server.api import diagnostic_transport_routes, lens_library_routes
from steerlab_server.api.profile import ServerProfile
from steerlab_server.api.scientific_execution_routes import build_scientific_execution_router
from steerlab_server.client.diagnostic_commands import workspace_action
from steerlab_server.experiment import diagnostic_archives as archives, diagnostic_inputs, managed_inputs
from steerlab_server.experiment import jlens_probe_batch as batch
from steerlab_server.jlens import lens_library, lens_store, probe, schemas
from steerlab_server.steering import model_loader
from test_jlens_probe_batch import PROMPTS, config_for, managed, tiny_model, workspace  # noqa: F401 - fixtures
from test_lens_library import UNAVAILABLE_REPAIR, digest, push
from test_optvec_jspace import LENS_ID, _RandomLensSource
from test_scientific_execution import setup  # noqa: F401 - fixture

CONVERTED = f'runs/jlens-lenses/{LENS_ID}/jacobians.safetensors'
RECORD = f'runs/jlens-lenses/{LENS_ID}/lens.json'
PUSH_REPAIR = f'Send the lens once with `remote lens-push {LENS_ID}` (or `runner lens-push`), then stage the request again.'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def entries(root, paths):
    """A closure's entries, written out by hand: path, SHA-256, and size."""
    return [{'path': path, 'sha256': digest((Path(root) / path).read_bytes()), 'bytes': (Path(root) / path).stat().st_size}
            for path in sorted(paths)]


def plan_by_hand(request, root, paths):
    """The plan of a request that carries its lens, as it was before a lens
    library existed: exactly these four fields, and their digest."""
    result = {'schemaVersion': 1, 'request': request, 'sourceRoot': str(Path(root).resolve()), 'files': entries(root, paths)}
    return {**result, 'planSHA256': hashlib.sha256(canonical(result)).hexdigest()}


def description_of(archive):
    with tarfile.open(archive, 'r:gz') as tar:
        return tar.extractfile('steerlab-bundle.json').read(), sorted(tar.getnames())


def probe_request(root, **overrides):
    return {'operation': batch.OPERATION, 'parameters': {'config': config_for(root, **overrides).to_dict()}}


def run_staged(staged, profile, root, monkeypatch, name):
    """Admit and run a staged request on the tiny model; returns its plan and result."""
    plan = scientific_execution.plan(staged['request'], profile)
    monkeypatch.setattr(model_loader, 'load', lambda *a, **k: tiny_model())
    packet, record = root / f'{name}-packet.json', root / f'{name}-record.json'
    packet.write_text(json.dumps(plan))
    code = scientific_execution.execute_packet(packet, name, record)
    return plan, code, json.loads(record.read_text())


def readings(result):
    """Every prompt's top-k and trajectory tables, as bytes: what the lens read."""
    run = Path(result['runDirectory'])
    manifest = json.loads((run / batch.MANIFEST).read_text())
    return {entry['id']: tuple((run / entry['directory'] / name).read_bytes()
                               for name in (probe.TOPK_CSV, probe.TRAJECTORY_CSV))
            for entry in manifest['prompts']}


def library_client():
    app = FastAPI()
    app.include_router(lens_library_routes.build_router(SimpleNamespace(jobs=None)))
    return TestClient(app)


@pytest.fixture
def runner(monkeypatch):
    """The engine is a runner: requests are staged on it, it runs them on its own machine, and with no folder
    declared it keeps a lens library beside its own records. The fixtures borrowed below otherwise build a
    workbench, which has no library unless one is declared. Ask for this fixture first, before the engine's
    profile is read."""
    monkeypatch.setenv('STEERLAB_SERVICE_ROLE', 'runner')
    monkeypatch.delenv('STEERLAB_LENS_LIBRARY_ROOT', raising=False)


# --- a request that carries its lens is the request it always was ----------------

def test_a_request_without_the_flag_is_byte_identical_and_never_asks_for_a_library(managed, monkeypatch):
    root, profile = managed
    monkeypatch.setattr(lens_library, 'root', lambda: pytest.fail('a request that carries its lens consulted the lens library'))
    request = probe_request(root)
    paths = ['prompts/probe-set.jsonl', RECORD, CONVERTED]
    expected = plan_by_hand(request, root, paths)

    plan = diagnostic_inputs.plan(request, root)
    assert canonical(plan) == canonical(expected)
    file = root / 'request.json'
    file.write_text(json.dumps(request))
    for payload in ({}, {'lensReference': False}):
        action = workspace_action('input-plan', {'workspaceRoot': str(root), 'requestFile': str(file), **payload})
        assert canonical(action) == canonical(expected)

    packed = diagnostic_inputs.package(request, root, root / 'runs/carried.tar.gz', plan['planSHA256'])
    metadata = {'schemaVersion': 1, 'kind': 'diagnosticInput', 'entries': expected['files'],
                'context': {'request': request, 'sourcePlanSHA256': expected['planSHA256']}}
    described, names = description_of(packed['bundlePath'])
    assert described == canonical(metadata)
    assert names == sorted(paths + ['steerlab-bundle.json'])

    staged = transport.stage(packed['bundlePath'], packed['bundleSha256'], profile)
    capsule = Path(staged['executionRoot'])
    assert (capsule.parent / 'input.json').read_bytes() == canonical(metadata)
    assert sorted(p.relative_to(capsule).as_posix() for p in capsule.rglob('*') if p.is_file()) == sorted(paths)
    admitted, code, record = run_staged(staged, profile, root, monkeypatch, 'carried')
    # The admitted inputs are the hand-built closure of the execution copy, with no new field.
    assert admitted['inputSHA256'] == hashlib.sha256(canonical(plan_by_hand(request, capsule, paths))).hexdigest()
    assert 'libraryLenses' not in json.dumps(admitted) and 'lensReference' not in json.dumps(admitted)
    assert code == 0 and record['status'] == 'succeeded' and record['result']['prompts'] == len(PROMPTS)
    assert 'lensLibrary' not in record['result']


# --- a request that refers to its lens -------------------------------------------

def test_a_probe_batch_reads_a_lens_the_site_holds_and_the_request_does_not_carry(runner, managed, monkeypatch):
    root, profile = managed
    assert profile.lens_library_root == str(root / '.steerlab/lens-library')
    request = probe_request(root)
    lens = root / CONVERTED
    sha, size = digest(lens.read_bytes()), lens.stat().st_size
    named = [{'lensID': LENS_ID, 'convertedSHA256': sha, 'bytes': size}]

    # The same request, carried: what the lens reads is the reference below.
    carried = diagnostic_inputs.plan(request, root)
    packed = diagnostic_inputs.package(request, root, root / 'runs/carried.tar.gz', carried['planSHA256'])
    _, code, record = run_staged(transport.stage(packed['bundlePath'], packed['bundleSha256'], profile),
                                 profile, root, monkeypatch, 'carried')
    assert code == 0
    carried_readings = readings(record['result'])

    referred = diagnostic_inputs.plan(request, root, lens_reference=True)
    assert [e['path'] for e in referred['files']] == [e['path'] for e in carried['files'] if e['path'] != CONVERTED]
    assert referred['libraryLenses'] == named and referred['planSHA256'] != carried['planSHA256']
    assert set(referred) == set(carried) | {'libraryLenses'}
    # The two plans list different files, so a plan reviewed one way cannot be packaged the other.
    for reviewed, flag in ((carried, True), (referred, False)):
        with pytest.raises(archives.Refusal, match='re-review'):
            diagnostic_inputs.package(request, root, root / 'runs/mixed.tar.gz', reviewed['planSHA256'], lens_reference=flag)
    packed = diagnostic_inputs.package(request, root, root / 'runs/referred.tar.gz', referred['planSHA256'], lens_reference=True)
    assert packed['context'] == {'request': request, 'sourcePlanSHA256': referred['planSHA256'],
                                 'lensReference': True, 'libraryLenses': named}
    described, names = description_of(packed['bundlePath'])
    assert CONVERTED not in names and RECORD in names
    assert Path(packed['bundlePath']).stat().st_size < size        # the archive is smaller than the lens it uses

    # The site does not hold the lens yet: refused by name, with the repair, before anything is unpacked.
    with pytest.raises(lens_library.LensLibraryRefusal) as refused:
        transport.stage(packed['bundlePath'], packed['bundleSha256'], profile)
    assert refused.value.code == 'lensNotInLibrary' and refused.value.repair_action == PUSH_REPAIR
    assert LENS_ID in str(refused.value) and sha in str(refused.value)
    assert not (root / 'runs' / ('diagnostic-input-' + packed['bundleSha256'])).exists()
    app = FastAPI()
    app.include_router(diagnostic_transport_routes.build_router(SimpleNamespace(jobs=object())))
    app.include_router(build_scientific_execution_router(SimpleNamespace(jobs=None, registry=None)))
    http = TestClient(app)
    stage_body = {'bundlePath': packed['bundlePath'], 'bundleSHA256': packed['bundleSha256']}
    response = http.post('/api/science/stage', json=stage_body)
    assert response.status_code == 409
    assert response.json()['detail'] == {'code': 'lensNotInLibrary', 'reason': str(refused.value), 'repairAction': PUSH_REPAIR}

    # Sent once, in parts. The staged copy then holds the record and no lens bytes.
    assert push(library_client(), lens)[2].json() == {'held': True, 'bytes': size}
    staged = http.post('/api/science/stage', json=stage_body).json()
    capsule = Path(staged['executionRoot'])
    assert sorted(p.relative_to(capsule).as_posix() for p in capsule.rglob('*') if p.is_file()) == [
        e['path'] for e in referred['files']]
    # The researcher's own copy is not what the job reads: take it away.
    lens.rename(root / 'moved-away.safetensors')
    admitted, code, record = run_staged(staged, profile, root, monkeypatch, 'referred')
    assert code == 0 and record['status'] == 'succeeded', record.get('error')
    assert record['result']['prompts'] == len(PROMPTS)
    assert readings(record['result']) == carried_readings
    assert Path(record['result']['runDirectory']).parent == capsule / 'runs'
    finished = SimpleNamespace(status='succeeded', finished_at=1.0, kind='science:' + batch.OPERATION,
                               result={**record['result'], 'scientificPlan': admitted})
    jobs = SimpleNamespace(get=lambda _id: finished)
    exported = transport.export('referred', jobs, profile)
    # What was admitted is the closure that names the lens by hash.
    closure = plan_by_hand(request, capsule, [e['path'] for e in referred['files']])
    by_hand = {key: closure[key] for key in ('schemaVersion', 'request', 'sourceRoot', 'files')} | {'libraryLenses': named}
    by_hand['planSHA256'] = hashlib.sha256(canonical(by_hand)).hexdigest()
    assert admitted['inputSHA256'] == hashlib.sha256(canonical(by_hand)).hexdigest()

    # The queued job hashes the library's bytes once: a copy that changed after admission is not read. The
    # whole file was hashed, so the mismatch is definite: the entry is set aside, and the repair needs nobody
    # but the researcher, who sends the lens again.
    held = lens_library.root() / sha / 'jacobians.safetensors'
    original = held.read_bytes()
    damaged = original[:-1] + bytes([original[-1] ^ 1])
    held.write_bytes(damaged)
    assert scientific_execution.plan(staged['request'], profile)['inputSHA256'] == admitted['inputSHA256']
    runs_before = sorted(path.name for path in (capsule / 'runs').iterdir())
    _, code, record = run_staged(staged, profile, root, monkeypatch, 'changed')
    assert code == 70 and record['status'] == 'failed' and 'runDirectory' not in record['result']
    assert record['error'] == (
        f"The lens library's copy of the lens '{LENS_ID}' no longer matches its SHA-256 ({sha}), so it was not "
        'read. It has been set aside, so the lens can be sent again. '
        f'Send the lens again with `remote lens-push {LENS_ID}` (or `runner lens-push`), then try the request again.')
    assert sorted(path.name for path in (capsule / 'runs').iterdir()) == runs_before       # nothing was computed
    aside = list((lens_library.root() / '.incoming').glob(f'damaged-{sha}-*'))
    assert len(aside) == 1 and not held.parent.exists()
    assert (aside[0] / 'jacobians.safetensors').read_bytes() == damaged                  # moved, not deleted
    assert library_client().get(f'/api/lens-library/{sha}').json()['held'] is False
    response = http.post('/api/science/plan', json=staged['request'])                    # and admission says so
    assert response.status_code == 409 and response.json()['detail']['code'] == 'lensNotInLibrary'
    assert response.json()['detail']['repairAction'] == PUSH_REPAIR
    # Sent again from the researcher's copy: the entry is whole, and the same staged request runs.
    assert push(library_client(), root / 'moved-away.safetensors')[2].json() == {'held': True, 'bytes': size}
    assert held.read_bytes() == original
    assert lens_library.reference(LENS_ID, sha, verify=True) == named[0]
    _, code, record = run_staged(staged, profile, root, monkeypatch, 'again')
    assert code == 0 and record['status'] == 'succeeded', record.get('error')
    assert readings(record['result']) == carried_readings
    # A different size, or no entry at all, is caught at admission, by name.
    held.write_bytes(original + b'x')
    response = http.post('/api/science/plan', json=staged['request'])
    assert response.status_code == 409 and response.json()['detail']['code'] == 'lensNotInLibrary'
    assert response.json()['detail']['repairAction'] == PUSH_REPAIR
    shutil.rmtree(held.parent)
    with pytest.raises(lens_library.LensLibraryRefusal) as refused:
        scientific_execution.plan(staged['request'], profile)
    assert refused.value.code == 'lensNotInLibrary'
    # The library is a cache: losing the entry loses no result. The finished job's output still comes home.
    again = transport.export('referred', jobs, profile)
    assert again['bundleSha256'] == exported['bundleSha256'] and again['entries'] == exported['entries']
    assert any(entry['path'].endswith('/' + batch.MANIFEST) for entry in again['entries'])
    # An engine with no library says that instead, with the other repair: a runner inside a scheduler job, a
    # runner that hands its work to Slurm, and a workbench, none of them with a folder declared.
    for name, value in (('SLURM_JOB_ID', '1'), ('STEERLAB_EXECUTOR', 'slurm'), ('STEERLAB_SERVICE_ROLE', 'workbench')):
        with monkeypatch.context() as patched:
            patched.setenv(name, value)
            assert lens_library.root() is None
            assert transport.export('referred', jobs, profile)['bundleSha256'] == exported['bundleSha256']
            for call in (lambda: transport.stage(packed['bundlePath'], packed['bundleSha256'], profile),
                         lambda: scientific_execution.plan(staged['request'], profile)):
                with pytest.raises(lens_library.LensLibraryRefusal) as refused:
                    call()
                assert refused.value.code == 'lensLibraryUnavailable' and refused.value.repair_action == UNAVAILABLE_REPAIR


def test_the_flag_reaches_the_plan_and_the_archive_through_the_workspace_action(managed):
    root, _ = managed
    request = probe_request(root)
    file = root / 'request.json'
    file.write_text(json.dumps(request))
    base = {'workspaceRoot': str(root), 'requestFile': str(file)}
    plan = workspace_action('input-plan', {**base, 'lensReference': True})
    assert canonical(plan) == canonical(diagnostic_inputs.plan(request, root, lens_reference=True))
    assert [lens['lensID'] for lens in plan['libraryLenses']] == [LENS_ID]
    packed = workspace_action('package', {**base, 'archivePath': str(root / 'runs/action.tar.gz'),
                                          'planSHA256': plan['planSHA256'], 'lensReference': True})
    assert packed['context']['lensReference'] is True and packed['context']['libraryLenses'] == plan['libraryLenses']
    for value in ('true', 1, None):
        with pytest.raises(archives.MalformedRequest, match='lensReference'):
            workspace_action('input-plan', {**base, 'lensReference': value})
    # The lens is still checked on the researcher's machine: a reference is only made to bytes that match.
    lens = root / CONVERTED
    lens.write_bytes(lens.read_bytes()[:-1] + b'\x00')
    with pytest.raises(archives.Refusal, match='differ from the imported hash'):
        workspace_action('input-plan', {**base, 'lensReference': True})
    lens.unlink()
    with pytest.raises(archives.PathRefusal, match='missing'):
        workspace_action('input-plan', {**base, 'lensReference': True})


def test_a_request_that_reads_no_lens_names_none_and_needs_no_library(setup, monkeypatch):
    root, request, profile = setup                      # a battery: no lens role at all
    carried = diagnostic_inputs.plan(request, root)
    referred = diagnostic_inputs.plan(request, root, lens_reference=True)
    assert 'libraryLenses' not in carried and referred['libraryLenses'] == []
    assert referred['files'] == carried['files'] and referred['planSHA256'] != carried['planSHA256']
    packed = diagnostic_inputs.package(request, root, root / 'runs/input.tar.gz', referred['planSHA256'], lens_reference=True)
    assert packed['context']['lensReference'] is True and packed['context']['libraryLenses'] == []
    monkeypatch.setattr(lens_library, 'root', lambda: pytest.fail('no lens is referred to, so no library is needed'))
    staged = transport.stage(packed['bundlePath'], packed['bundleSha256'], profile)
    assert scientific_execution.plan(staged['request'], profile)['request']['operation'] == 'battery'


def test_a_local_job_is_told_which_lens_library_its_engine_has(setup, monkeypatch):
    import time
    from steerlab_server.api.jobs import DurableJobStore, JobManager
    root, request, profile = setup
    jobs = JobManager(store=DurableJobStore(str(root / 'jobs.sqlite')), sweep_orphans=False)
    told = []

    def run(self, command, **kwargs):
        told.append(kwargs.get('env'))
        Path(command[-2]).write_text(json.dumps({'result': {}}))
        return SimpleNamespace(returncode=0, stderr='')
    monkeypatch.setattr(scientific_execution.LocalExecutor, 'run', run)

    def submit(profile):
        submitted = scientific_execution.submit(request, scientific_execution.plan(request, profile)['planSHA256'],
                                                profile=profile, jobs=jobs)
        deadline = time.monotonic() + 5
        while jobs.get(submitted['jobId']).status not in {'failed', 'succeeded'} and time.monotonic() < deadline:
            time.sleep(.01)
    # A workbench with no folder declared has no library, and its child is told so rather than left to work
    # it out.
    monkeypatch.delenv('STEERLAB_SERVICE_ROLE', raising=False)
    monkeypatch.delenv('STEERLAB_LENS_LIBRARY_ROOT', raising=False)
    assert profile.lens_library_root is None
    submit(profile)
    assert told == [{'STEERLAB_LENS_LIBRARY_ROOT': ''}]
    # A runner that runs its work on this machine has one beside its own records.
    monkeypatch.setenv('STEERLAB_SERVICE_ROLE', 'runner')
    submit(ServerProfile.from_env())
    assert told[-1] == {'STEERLAB_LENS_LIBRARY_ROOT': str(root / '.steerlab/lens-library')}
    # Inside a scheduler allocation it has none. The local executor drops the scheduler's job id for its
    # child, so the child is told, and cannot guess.
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    submit(ServerProfile.from_env())
    assert told[-1] == {'STEERLAB_LENS_LIBRARY_ROOT': ''}
    # A declared folder is passed on, whatever else is true of the engine.
    monkeypatch.setenv('STEERLAB_LENS_LIBRARY_ROOT', str(root / 'declared'))
    for role in ('runner', 'workbench'):
        monkeypatch.setenv('STEERLAB_SERVICE_ROLE', role)
        submit(ServerProfile.from_env())
        assert told[-1] == {'STEERLAB_LENS_LIBRARY_ROOT': str(root / 'declared')}
    monkeypatch.delenv('SLURM_JOB_ID')
    monkeypatch.setenv('STEERLAB_SERVICE_ROLE', 'runner')
    monkeypatch.setenv('STEERLAB_LENS_LIBRARY_ROOT', '')
    assert ServerProfile.from_env().lens_library_root is None and lens_library.root() is None


def test_an_archive_cannot_claim_a_lens_its_request_does_not_read(runner, managed):
    root, profile = managed
    request = probe_request(root)
    lens = root / CONVERTED
    push(library_client(), lens)
    referred = diagnostic_inputs.plan(request, root, lens_reference=True)
    context = {'request': request, 'sourcePlanSHA256': referred['planSHA256'], 'lensReference': True}
    paths = [e['path'] for e in referred['files']]
    honest = referred['libraryLenses']
    extra = honest + [{'lensID': 'custom-lens-other', 'convertedSHA256': honest[0]['convertedSHA256'], 'bytes': honest[0]['bytes']}]
    for label, lenses, message in [
            ('none', [], 'exactly the required diagnostic closure'),
            ('extra', extra, 'exactly the required diagnostic closure'),
            ('shape', [{'lensID': LENS_ID}], 'does not read'),
            ('flag', None, 'does not read')]:
        described = {**context, 'libraryLenses': lenses} if lenses is not None else {
            'request': request, 'sourcePlanSHA256': referred['planSHA256'], 'libraryLenses': honest}
        packed = archives.package(root, paths, root / f'runs/{label}.tar.gz', kind='diagnosticInput', context=described)
        with pytest.raises(archives.Refusal, match=message):
            transport.stage(packed['bundlePath'], packed['bundleSha256'], profile)
        assert not (root / 'runs' / ('diagnostic-input-' + packed['bundleSha256'])).exists()


# --- a registered lens: the record in the workspace, the bytes in the library ----

def test_a_readout_and_an_unstaged_request_resolve_a_registered_lens(workspace, monkeypatch, tmp_path):
    from test_jlens_readout_gain import _Host
    from steerlab_server.jlens.readout import LensReadout, ReadoutConfig
    from test_optvec_eval import HIDDEN
    site = tmp_path / 'site'
    site.mkdir()
    monkeypatch.setenv('STEERLAB_ROOT', str(site))
    monkeypatch.setenv('STEERLAB_METADATA_ROOT', str(site / '.steerlab'))
    monkeypatch.setenv('STEERLAB_SERVER_ROLE', 'workstation')
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'local')
    # A workbench that serves a workspace of its own, with a lens library declared for it, as a site's is.
    monkeypatch.delenv('STEERLAB_SERVICE_ROLE', raising=False)
    monkeypatch.setenv('STEERLAB_LENS_LIBRARY_ROOT', str(tmp_path / 'site-lenses'))
    profile = ServerProfile.from_env()
    assert lens_library.root() == tmp_path / 'site-lenses'
    client = library_client()
    # Sent from the researcher's workspace: the bytes, then the record as it is on their machine.
    theirs = json.loads((workspace / RECORD).read_text())
    assert Path(theirs['converted']['path']).is_absolute()          # this machine's folder, in the record
    sha = push(client, workspace / CONVERTED)[0]
    assert client.post('/api/lens-library/register', json={'record': theirs}).json() == {
        'lensID': LENS_ID, 'registered': True, 'changed': True}
    assert sorted(p.relative_to(site).as_posix() for p in (site / 'runs').rglob('*') if p.is_file()) == [RECORD]
    assert str(workspace) not in (site / RECORD).read_text()

    record = lens_store.resolve(LENS_ID, str(site))
    assert record.converted.path == CONVERTED and record.converted.sha256 == sha
    source = _RandomLensSource()
    host = _Host(head_weight=torch.randn(11, HIDDEN), norm_weight=torch.zeros(HIDDEN)).eval()
    readout = LensReadout.build(record=record, config=ReadoutConfig(layers=[0, 2], watchlist=[1, 4], topK=2),
                                model=SimpleNamespace(model=host, tokenizer=None, model_id='test/tiny'), root=str(site))
    for layer in (0, 2):
        torch.testing.assert_close(readout.jacobians[layer], source.jacobian(layer), rtol=0, atol=0)

    # A request planned on the serving workspace itself, with nothing staged.
    request = probe_request(site)
    with pytest.raises(archives.PathRefusal, match='missing'):       # asked to carry it, the bytes are not here
        managed_inputs.plan(request, site)
    held = managed_inputs.plan(request, site, managed_inputs.WHERE_HELD)
    assert held['libraryLenses'] == [{'lensID': LENS_ID, 'convertedSHA256': sha, 'bytes': (workspace / CONVERTED).stat().st_size}]
    assert [e['path'] for e in held['files']] == ['prompts/probe-set.jsonl', RECORD]
    plan = scientific_execution.plan(request, profile)
    assert plan['root'] == str(site.resolve()) and 'inputBundleSHA256' not in plan
    monkeypatch.setattr(model_loader, 'load', lambda *a, **k: tiny_model())
    packet, record_file = site / 'packet.json', site / 'record.json'
    packet.write_text(json.dumps(plan))
    assert scientific_execution.execute_packet(packet, 'unstaged', record_file) == 0
    assert json.loads(record_file.read_text())['result']['prompts'] == len(PROMPTS)
    # The registered lens has left the library: said by name, with the repair that works (send it again),
    # where the old answer spoke of staging and offered a repair that could not help.
    shutil.rmtree(lens_library.root() / sha)
    app = FastAPI()
    app.include_router(build_scientific_execution_router(SimpleNamespace(jobs=None, registry=None)))
    response = TestClient(app).post('/api/science/plan', json=request)
    assert response.status_code == 409, response.text
    detail = response.json()['detail']
    assert detail['code'] == 'lensNotInLibrary' and LENS_ID in detail['reason'] and sha in detail['reason']
    assert 'no converted file where this request runs' in detail['reason'] and 'staged' not in detail['reason']
    assert f'`remote lens-push {LENS_ID}` (or `runner lens-push`)' in detail['repairAction']
    assert 'localRequestPath' not in detail['repairAction']
    with pytest.raises(schemas.JLensError, match='missing'):
        lens_store.load_layer(record, 0, root=str(site))
    # On an engine with no library at all (the same workbench, with none declared), a missing file is the
    # missing input it always was.
    monkeypatch.delenv('STEERLAB_LENS_LIBRARY_ROOT')
    assert lens_library.root() is None
    with pytest.raises(archives.PathRefusal, match='Required input or output is missing'):
        managed_inputs.plan(request, site, managed_inputs.WHERE_HELD)


# --- fix round 1: the plan and the reader agree about where a lens's bytes come from ---

def test_a_record_naming_a_file_outside_the_importers_layout_is_never_answered_by_the_library(runner, managed, tmp_path):
    """The crafted archive of review finding 1. Its lens record names a hash the library holds and an absolute
    path, existing on the engine's host, that is not a lens folder. Planned from the library and read from that
    path, the job would compute from bytes nobody hashed; so such a record is never planned from the library."""
    root, profile = managed
    request = probe_request(root)
    lens = root / CONVERTED
    sha, size = push(library_client(), lens)[0], lens.stat().st_size
    outside = tmp_path / 'outside' / 'unhashed.safetensors'
    outside.parent.mkdir()
    outside.write_bytes(bytes(size))                       # the same size, and not the lens
    outside.chmod(0)                                       # anything that opened it would fail loudly
    record = json.loads((root / RECORD).read_text())
    record['converted']['path'] = str(outside)
    crafted = tmp_path / 'crafted'
    (crafted / 'prompts').mkdir(parents=True)
    (crafted / RECORD).parent.mkdir(parents=True)
    shutil.copy(root / 'prompts/probe-set.jsonl', crafted / 'prompts/probe-set.jsonl')
    (crafted / RECORD).write_text(json.dumps(record))
    context = {'request': request, 'sourcePlanSHA256': 'f' * 64, 'lensReference': True,
               'libraryLenses': [{'lensID': LENS_ID, 'convertedSHA256': sha, 'bytes': size}]}
    packed = archives.package(crafted, ['prompts/probe-set.jsonl', RECORD], root / 'runs/crafted.tar.gz',
                              kind='diagnosticInput', context=context)
    old_refusal = 'Managed lenses require converted tensors in the workspace lens library.'
    with pytest.raises(archives.Refusal, match=old_refusal) as refused:
        transport.stage(packed['bundlePath'], packed['bundleSha256'], profile)
    assert not isinstance(refused.value, lens_library.LensLibraryRefusal)
    assert not (root / 'runs' / ('diagnostic-input-' + packed['bundleSha256'])).exists()
    # The same record in the workspace an engine serves: refused in every mode, as before a library existed.
    (root / RECORD).write_text(json.dumps(record))
    for mode in (managed_inputs.CARRIED, managed_inputs.REFERRED, managed_inputs.WHERE_HELD):
        with pytest.raises(archives.Refusal, match=old_refusal):
            managed_inputs.plan(request, root, mode)
    with pytest.raises(archives.Refusal, match=old_refusal):
        scientific_execution.plan(request, profile)
    # The reader agrees: asked for that record's file, it names the path the record names, never the library's.
    from steerlab_server.jlens import artifact_paths
    assert artifact_paths.converted_file(LENS_ID, str(outside), str(root)) == str(outside)
    outside.chmod(0o600)


# --- fix round 1: the engine's library, wherever a job or a child is pointed ------

def test_a_job_reads_the_engines_library_and_a_lost_entry_costs_no_result(runner, managed, monkeypatch, tmp_path):
    import time
    from steerlab_server.api import diagnostic_cleanup as cleanup
    from steerlab_server.api.jobs import DurableJobStore, JobManager
    root, _ = managed
    # The engine keeps its records, and so its library, beside its root; a job is pointed at an execution
    # copy with a root of its own, and must still read the engine's library, not look for one beside the copy.
    monkeypatch.delenv('STEERLAB_METADATA_ROOT')
    profile = ServerProfile.from_env()
    assert profile.lens_library_root == str(root / '.steerlab/lens-library')
    request = probe_request(root)
    referred = diagnostic_inputs.plan(request, root, lens_reference=True)
    packed = diagnostic_inputs.package(request, root, root / 'runs/referred.tar.gz', referred['planSHA256'], lens_reference=True)
    sha = push(library_client(), root / CONVERTED)[0]
    staged = transport.stage(packed['bundlePath'], packed['bundleSha256'], profile)
    admitted, code, record = run_staged(staged, profile, root, monkeypatch, 'referred')
    assert code == 0 and record['status'] == 'succeeded', record.get('error')
    output = Path(record['result']['runDirectory'])
    assert not (Path(staged['executionRoot']) / '.steerlab/lens-library').exists()

    # The library loses the entry. The finished job's output still comes home, through the real job store.
    jobs = JobManager(store=DurableJobStore(str(root / 'jobs.sqlite')), sweep_orphans=False)
    job = jobs.record_external('science:' + batch.OPERATION, status='succeeded', executor='local',
                               job_id='referred-job', result={**record['result'], 'scientificPlan': admitted})
    job.finished_at = time.time() - 3600
    jobs.store.update(job)
    policy = root / 'policy.json'
    policy.write_text(json.dumps({'schemaVersion': 1, 'allowDiagnosticOutputRemoval': True, 'minimumRetentionHours': 0,
                                  'source': 'Researcher supplied retention rule for disposable fixture outputs.'}))
    monkeypatch.setenv('STEERLAB_DIAGNOSTIC_CLEANUP_POLICY', str(policy))
    shutil.rmtree(lens_library.root() / sha)
    with pytest.raises(lens_library.LensLibraryRefusal):             # a new run of it would need the lens again
        scientific_execution.plan(staged['request'], profile)
    reference = transport.export(job.id, jobs, profile)
    local = tmp_path / 'local'
    local.mkdir()
    imported = archives.import_evidence(reference['bundlePath'], reference['bundleSha256'], local)
    assert archives.verify(imported['receiptSHA256'], local) == imported['receipt'] and output.exists()
    # Managed cleanup covers battery and stability copies only, neither of which reads a lens, so no cleanup
    # ever depends on the library: a job that referred to a lens is retained by that policy, lens or no lens.
    with pytest.raises(archives.Refusal, match='covers battery and stability copies only'):
        cleanup.plan(job.id, imported['receipt'], jobs, profile)


def test_a_battery_packaged_with_the_flag_is_exported_and_cleaned_up_with_no_library(setup, monkeypatch, tmp_path):
    """The only jobs managed cleanup removes read no lens. One packaged with `--lens-reference` names none, and
    its export and its cleanup need no library at all."""
    import time
    from steerlab_server.api import diagnostic_cleanup as cleanup
    from steerlab_server.api.jobs import DurableJobStore, JobManager
    root, request, profile = setup
    referred = diagnostic_inputs.plan(request, root, lens_reference=True)
    packed = diagnostic_inputs.package(request, root, root / 'runs/input.tar.gz', referred['planSHA256'], lens_reference=True)
    staged = transport.stage(packed['bundlePath'], packed['bundleSha256'], profile)
    plan = scientific_execution.plan(staged['request'], profile)
    output = Path(plan['root']) / 'runs/example-battery'
    output.mkdir(parents=True)
    (output / 'battery-report.json').write_text('{"complete":true}')
    jobs = JobManager(store=DurableJobStore(str(root / 'jobs.sqlite')), sweep_orphans=False)
    job = jobs.record_external('science:battery', status='succeeded', executor='local', job_id='example-job',
                               result={'scientificPlan': plan, 'runDirectory': str(output)})
    job.finished_at = time.time() - 3600
    jobs.store.update(job)
    policy = root / 'policy.json'
    policy.write_text(json.dumps({'schemaVersion': 1, 'allowDiagnosticOutputRemoval': True, 'minimumRetentionHours': 0,
                                  'source': 'Researcher supplied retention rule for disposable fixture outputs.'}))
    monkeypatch.setenv('STEERLAB_DIAGNOSTIC_CLEANUP_POLICY', str(policy))
    monkeypatch.setattr(lens_library, 'root', lambda: pytest.fail('no lens is referred to, so no library is needed'))
    reference = transport.export(job.id, jobs, profile)
    local = tmp_path / 'local'
    local.mkdir()
    receipt = archives.import_evidence(reference['bundlePath'], reference['bundleSha256'], local)['receipt']
    planned = cleanup.plan(job.id, receipt, jobs, profile)
    assert planned['eligible']
    assert cleanup.apply(job.id, receipt, planned['planSHA256'], jobs, profile, confirmed=True)['state'] == 'removed'
    assert not output.exists()


def test_the_validation_child_is_told_which_library_the_engine_has(runner, managed, monkeypatch):
    from steerlab_server.api import managed_validation
    root, _ = managed
    monkeypatch.delenv('STEERLAB_METADATA_ROOT')           # so a child pointed elsewhere would look elsewhere
    elsewhere = root / 'runs/an-execution-copy'
    elsewhere.mkdir(parents=True)
    request = probe_request(root)
    told = []
    run = managed_validation.subprocess.run

    def child(command, **kwargs):
        told.append((kwargs['env']['STEERLAB_ROOT'], kwargs['env'].get('STEERLAB_LENS_LIBRARY_ROOT')))
        return run(command, **kwargs)
    monkeypatch.setattr(managed_validation.subprocess, 'run', child)
    assert managed_validation.validate(request, elsewhere)['models'] == [{'modelID': 'test/tiny', 'revision': 'a' * 40}]
    assert told == [(str(elsewhere.resolve()), str(root / '.steerlab/lens-library'))]
    # An engine with no library says so to its child: a runner inside a scheduler job, and a workbench.
    for name, value in (('SLURM_JOB_ID', '1'), ('STEERLAB_SERVICE_ROLE', 'workbench')):
        with monkeypatch.context() as patched:
            patched.setenv(name, value)
            managed_validation.validate(request, elsewhere)
            assert told[-1] == (str(elsewhere.resolve()), '')
    # A declared folder is the one the child is told, on a workbench too.
    monkeypatch.setenv('STEERLAB_SERVICE_ROLE', 'workbench')
    monkeypatch.setenv('STEERLAB_LENS_LIBRARY_ROOT', str(root / 'declared'))
    managed_validation.validate(request, elsewhere)
    assert told[-1] == (str(elsewhere.resolve()), str(root / 'declared'))


# --- fix round 1: a finished job stays finished ------------------------------------

@pytest.mark.parametrize('afterwards', ['killed', 'cancelled'])
def test_a_job_whose_record_says_succeeded_is_succeeded_whatever_became_of_its_process(setup, monkeypatch, afterwards):
    """The copy of a fitted lens is made after the fit's record says succeeded. A child killed during it (a
    memory limit on the node), or a cancel that arrives then, must not turn a finished fit into a failed or
    cancelled job. The stand-in child writes that record, then dies hard, or lingers until it is cancelled."""
    import sys
    import time
    from steerlab_server.api.jobs import DurableJobStore, JobManager
    root, request, profile = setup
    jobs = JobManager(store=DurableJobStore(str(root / 'jobs.sqlite')), sweep_orphans=False)
    then = 'os.kill(os.getpid(), 9)' if afterwards == 'killed' else 'time.sleep(60)'
    script = ('import json, os, sys, time; from pathlib import Path; record = Path(sys.argv[1]); '
              'staged = record.with_suffix(".tmp"); '
              'staged.write_text(json.dumps({"status": "succeeded", "result": {"runDirectory": sys.argv[2], "promptsFitted": 3}})); '
              'staged.replace(record); ' + then)
    real_run = scientific_execution.LocalExecutor.run

    def child(self, command, **kwargs):
        return real_run(self, [sys.executable, '-c', script, command[-2], str(root / 'runs/finished')], **kwargs)
    monkeypatch.setattr(scientific_execution.LocalExecutor, 'run', child)
    submitted = scientific_execution.submit(request, scientific_execution.plan(request, profile)['planSHA256'],
                                            profile=profile, jobs=jobs)
    job = jobs.get(submitted['jobId'])
    record = Path(submitted['recordsDirectory']) / (job.id + '.json')
    deadline = time.monotonic() + 20
    while not record.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert record.exists()
    if afterwards == 'cancelled':
        assert jobs.cancel(job.id)
    while job.status not in {'succeeded', 'failed', 'cancelled', 'cancelledResumable'} and time.monotonic() < deadline:
        time.sleep(.01)
    assert job.status == 'succeeded' and job.error is None and not job.cancelled
    assert job.result['promptsFitted'] == 3 and job.result['runDirectory'].endswith('runs/finished')
