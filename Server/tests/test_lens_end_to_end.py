"""The lens library, end to end (public issues #13 and #14): the real Python
client against the real engine.

The engine's half (`test_lens_library.py`, `test_lens_reference_requests.py`)
drives the routes by hand, and the client's half (`test_lens_transfer.py`)
talks to a stand-in written from the contract. Here nothing stands in for
either side. Every command is one a researcher types, run through
`client_cli.main`; every request reaches the engine's own app
(`steerlab_server.api.app.app`, with its token and service-role checks) over
an in-process ASGI transport; and what each side left on disk is then read
back from the disk.

One process holds both sides, and both read their workspace from the
environment, so each side is given its own: a command runs in the
researcher's environment, and each request it sends is served in the
engine's (`Engine.http`), as it would be on another machine.
"""
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from safetensors.numpy import save_file

torch = pytest.importorskip('torch')

from steerlab_server import client_cli
from steerlab_server.api import scientific_execution
from steerlab_server.api.app import app
from steerlab_server.client import runner as runner_api
from steerlab_server.experiment import artifact_imports
from steerlab_server.experiment import jlens_probe_batch as batch
from steerlab_server.jlens import artifact_paths, lens_library, lens_store, schemas
from steerlab_server.jlens import qualification as qualifications
from steerlab_server.steering import model_loader
from test_jlens_probe_batch import MODEL, PROMPTS, REVISION, tiny_model, write_prompts
from test_lens_reference_requests import canonical, description_of, plan_by_hand, readings
from test_optvec_eval import HIDDEN, LAYERS

FIXTURE = json.loads((Path(__file__).resolve().parents[2]
                      / 'prompts/fixtures/lens-library/refusals.json').read_text(encoding='utf-8'))
SERVER = 'http://testserver'
TOKEN = 'fictional-test-token'
#: What decides which engine a process is. None of it may leak from one side to the other.
ENGINE_ONLY = ('STEERLAB_ROOT', 'STEERLAB_RUN_ROOT', 'STEERLAB_METADATA_ROOT', 'STEERLAB_SERVER_ROLE',
               'STEERLAB_SERVICE_ROLE', 'STEERLAB_EXECUTOR', 'STEERLAB_LENS_LIBRARY_ROOT', 'STEERLAB_MAX_UPLOAD_BYTES',
               'STEERLAB_TRANSFER_METHOD', 'STEERLAB_AUTH_MODE', 'STEERLAB_AUTH_TOKEN', 'SLURM_JOB_ID',
               'STEERLAB_WORKSPACE', client_cli.RUNNER_TOKEN_ENV)


def digest(data):
    return hashlib.sha256(data).hexdigest()


@contextmanager
def environment(values):
    """This process's environment is exactly `values` inside, and what it was afterwards. Where it already is
    `values`, nothing is touched: a job of the engine's may be reading it on another thread."""
    saved = dict(os.environ)
    if saved == values:
        yield
        return
    os.environ.clear()
    os.environ.update(values)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


class Engine:
    """One engine: the real app, a workspace of its own, and its own environment.

    `library` is 'declared' (a folder outside the workspace, as a site profile's `lensLibrary` role sets it),
    'empty' (the variable set and empty), or None (nothing declared). `limit` is its upload limit."""

    def __init__(self, base, folder, *, role='workbench', library='declared', limit=8192, transfer=None):
        self.root = folder / 'workspace'
        self.root.mkdir(parents=True)
        self.env = {**base, 'STEERLAB_ROOT': str(self.root), 'STEERLAB_METADATA_ROOT': str(folder / 'records'),
                    'STEERLAB_SERVER_ROLE': 'workstation', 'STEERLAB_EXECUTOR': 'local', 'STEERLAB_AUTH_MODE': 'token',
                    'STEERLAB_AUTH_TOKEN': TOKEN, 'STEERLAB_MAX_UPLOAD_BYTES': str(limit)}
        if role is not None:
            self.env['STEERLAB_SERVICE_ROLE'] = role
        if transfer is not None:
            self.env['STEERLAB_TRANSFER_METHOD'] = transfer
        self.library = None
        if library == 'declared':
            self.library = folder / 'lens-library'
            self.env['STEERLAB_LENS_LIBRARY_ROOT'] = str(self.library)
        elif library == 'empty':
            self.env['STEERLAB_LENS_LIBRARY_ROOT'] = ''
        elif role == 'runner':
            self.library = folder / 'records/lens-library'      # a runner's own, beside its records
        self.calls = []          # (method, path) of every request that reached this engine
        self.lost = []           # (method, pattern, when): one request the network loses, 'before' or 'after' the engine
        engine = self

        class Boundary(TestClient):
            """What a network puts between the two: the request is served in the engine's environment."""

            def request(self, method, url, **options):
                path = httpx.URL(str(url)).path
                found = next((item for item in engine.lost if item[0] == method and re.search(item[1], path)), None)
                if found is not None:
                    engine.lost.remove(found)
                    if found[2] == 'before':
                        raise httpx.ReadError('the connection was lost')
                engine.calls.append((method, path))
                with environment(engine.env):
                    response = super().request(method, url, **options)
                if found is not None:
                    raise httpx.ReadError('the connection was lost')
                return response

        self.http = Boundary(app)

    def sent(self, method, pattern=''):
        return [path for verb, path in self.calls if verb == method and re.search(pattern, path)]

    def get(self, path):
        return self.http.get(path, headers={'Authorization': 'Bearer ' + TOKEN})

    def post(self, path, **options):
        return self.http.post(path, headers={'Authorization': 'Bearer ' + TOKEN, **options.pop('headers', {})}, **options)

    def entry(self, sha):
        return self.library / sha

    def lens_files(self):
        """Every converted lens file anywhere on this engine's side, by its path."""
        return sorted(str(path) for path in self.root.parent.rglob('jacobians.safetensors'))

    def run(self, plan, name, monkeypatch):
        """What the engine's queued job does with an admitted plan, in the engine's environment, on the tiny
        model. Returns its exit code, its record, and the file every read of a lens was sent to."""
        packet, record = self.root.parent / f'{name}-packet.json', self.root.parent / f'{name}-record.json'
        packet.write_text(json.dumps(plan))
        resolved, real = [], artifact_paths.converted_file

        def recording(*arguments, **options):
            path = real(*arguments, **options)
            if options.get('library', True):       # a read; the planner's own question about the workspace is not one
                resolved.append(path)
            return path
        with monkeypatch.context() as patched, environment(self.env):
            patched.setattr(model_loader, 'load', lambda *a, **k: tiny_model())
            patched.setattr(artifact_paths, 'converted_file', recording)
            code = scientific_execution.execute_packet(packet, name, record)
        return code, json.loads(record.read_text()), resolved


class World:
    def __init__(self, tmp_path, monkeypatch, capsys):
        self.tmp_path, self.monkeypatch, self.capsys = tmp_path, monkeypatch, capsys
        self.base = {name: value for name, value in os.environ.items() if name not in ENGINE_ONLY}
        self.client_env = {**self.base, client_cli.RUNNER_TOKEN_ENV: TOKEN}
        self.client_class = runner_api.RunnerClient
        self.count = 0

    def engine(self, **options):
        self.count += 1
        return Engine(self.base, self.tmp_path / f'site-{self.count}', **options)

    def cli(self, engine, *arguments):
        """One command of the real Python client, in the researcher's environment. Its requests go to `engine`
        (None: this command must send none). Returns the exit code and the --json envelope."""
        real = self.client_class

        def connect(**options):
            assert engine is not None, 'this command was not expected to reach an engine'
            options['http_client'] = engine.http
            return real(**options)
        self.monkeypatch.setattr(runner_api, 'RunnerClient', connect)
        self.capsys.readouterr()
        with environment(self.client_env):
            code = client_cli.main([*arguments, '--json'])
        return code, json.loads(self.capsys.readouterr().out)

    def researcher(self, name='researcher', seed=0):
        """A researcher's workspace with one small lens for the tiny model, imported with the client's own
        verbs from a description and a tensor file. Returns the workspace, the lens ID, and the description."""
        root, source = self.tmp_path / name, self.tmp_path / (name + '-source')
        root.mkdir()
        description = lens_source(source, seed)
        code, reviewed = self.cli(None, '--root', str(root), 'science', 'artifact-plan', str(description))
        assert code == 0, reviewed
        code, imported = self.cli(None, '--root', str(root), 'science', 'artifact-import', str(description),
                                  '--plan-sha256', reviewed['result']['planSHA256'])
        assert code == 0 and imported['result']['changed'] is True, imported
        return root, imported['result']['lensID'], description


def lens_source(folder, seed=0):
    """Source files for a lens the tiny model can use: one dense Jacobian for each of its first three layers."""
    folder.mkdir()
    generator = np.random.default_rng(seed)
    save_file({f'J{i}': generator.standard_normal((HIDDEN, HIDDEN)).astype(np.float32) for i in range(LAYERS - 1)},
              str(folder / 'weights.safetensors'))
    description = folder / 'import.json'
    description.write_text(json.dumps({
        'schemaVersion': 1, 'kind': 'jlens', 'modelID': MODEL, 'modelRevision': None, 'hiddenSize': HIDDEN,
        'layerCount': LAYERS, 'tensorFile': 'weights.safetensors',
        'lens': {'targetLayer': LAYERS - 1, 'layers': {str(i): f'J{i}' for i in range(LAYERS - 1)}, 'promptsFitted': 12}}))
    return description


def converted(root, lens_id):
    return Path(root) / 'runs/jlens-lenses' / lens_id / 'jacobians.safetensors'


def record_file(root, lens_id):
    return Path(root) / 'runs/jlens-lenses' / lens_id / 'lens.json'


def strict_loads(data):
    """Parse as a strict JSON reader does: NaN and infinity are errors."""
    def refuse(constant):
        raise ValueError('not JSON: ' + constant)
    return json.loads(data, parse_constant=refuse)


def probe_request(root, lens_id):
    """A probe batch over three prompts that reads the lens: the smallest managed operation with a lens."""
    config = {'modelID': MODEL, 'revision': REVISION, 'lensID': lens_id, 'layers': [0, 2], 'topK': 3, 'pin': ['law'],
              'promptMode': 'rawCompletion', 'dtype': 'float32', 'prompts': write_prompts(root)}
    request = {'operation': batch.OPERATION, 'parameters': {'config': batch.ProbeBatchConfig.from_dict(config).to_dict()}}
    file = Path(root) / 'request.json'
    file.write_text(json.dumps(request))
    return request, file


def members(archive):
    with tarfile.open(archive, 'r:gz') as tar:
        return sorted(tar.getnames())


@pytest.fixture
def world(tmp_path, monkeypatch, capsys):
    return World(tmp_path, monkeypatch, capsys)


def push(world, engine, root, lens_id, *more):
    return world.cli(engine, '--root', str(root), 'runner', 'lens-push', lens_id, '--runner', SERVER, *more)


def status(world, engine, root, lens_id):
    return world.cli(engine, '--root', str(root), 'runner', 'lens-status', lens_id, '--runner', SERVER)


# --- 1. a lens is sent once, and the site has its record -------------------------

def test_a_lens_is_sent_once_in_parts_and_the_site_holds_it_and_its_record(world):
    root, lens_id, _ = world.researcher()
    engine = world.engine()
    file = converted(root, lens_id)
    data = file.read_bytes()
    sha = digest(data)

    code, before = status(world, engine, root, lens_id)
    assert code == 0 and before['changed'] is False
    assert before['message'] == f"The site's lens library does not hold lens {lens_id}; the site does not have its record."
    assert before['result']['response'] == {
        'lensID': lens_id, 'convertedSHA256': sha, 'bytes': len(data), 'libraryAvailable': True, 'held': False,
        'receivedParts': None, 'siteRecord': 'absent', 'changed': False}
    assert not engine.library.exists() and not (engine.root / 'runs').exists()      # a status only reads

    code, pushed = push(world, engine, root, lens_id)
    assert code == 0 and pushed['state'] == 'ready' and pushed['changed'] is True, pushed
    response = pushed['result']['response']
    # The header, then one part for each of the three layers: no part is larger than the engine allows.
    parts = engine.sent('PUT', '/parts/')
    assert parts == [f'/api/lens-library/{sha}/parts/{index}' for index in range(4)]
    assert response == {
        'lensID': lens_id, 'convertedSHA256': sha, 'bytes': len(data), 'held': True, 'alreadyHeld': False,
        'adopted': False, 'uploadedParts': 4, 'uploadedBytes': len(data), 'registered': True, 'recordChanged': True,
        'recordNonFiniteNumbers': 0, 'changed': True}
    assert pushed['message'] == (f"Sent lens {lens_id} to the site's lens library (4 parts, {len(data)} bytes); "
                                 'the site has its record.')
    # The order a push keeps: the capability, what the site holds, the plan of parts, the parts, the whole-file
    # check, the record.
    assert [path.replace(sha, '<sha>') for _, path in engine.calls[-9:]] == [
        '/api/capabilities', '/api/lens-library/<sha>', '/api/lens-library/<sha>/begin',
        *[f'/api/lens-library/<sha>/parts/{index}' for index in range(4)],
        '/api/lens-library/<sha>/complete', '/api/lens-library/register']

    # On the site: the entry is whole, under the file's own hash, outside the workspace.
    entry = engine.entry(sha)
    assert sorted(path.name for path in entry.iterdir()) == ['entry.json', 'jacobians.safetensors']
    assert (entry / 'jacobians.safetensors').read_bytes() == data
    described = json.loads((entry / 'entry.json').read_bytes())
    assert {key: described[key] for key in ('schemaVersion', 'convertedSHA256', 'bytes', 'origin')} == {
        'schemaVersion': 1, 'convertedSHA256': sha, 'bytes': len(data), 'origin': 'pushed'}
    assert not (engine.library / '.incoming' / sha).exists()
    # Its record is in the engine's lens store, naming the file by its place in a workspace and no folder of
    # the researcher's machine; the bytes are in the library only.
    site_record = json.loads(record_file(engine.root, lens_id).read_bytes())
    mine = json.loads(record_file(root, lens_id).read_bytes())
    assert site_record == mine and site_record['converted']['path'] == f'runs/jlens-lenses/{lens_id}/jacobians.safetensors'
    assert str(root) not in json.dumps(site_record)
    assert engine.lens_files() == [str(entry / 'jacobians.safetensors')]
    with environment(engine.env):
        found = lens_store.resolve(lens_id)                        # a study-style lookup, by name, on the site
        assert found.converted.sha256 == sha
        assert torch.equal(lens_store.load_layer(found, 1), lens_store.load_layer(lens_store.resolve(lens_id, str(root)), 1,
                                                                                 root=str(root)))

    code, after = status(world, engine, root, lens_id)
    assert code == 0 and after['changed'] is False
    assert after['message'] == f"The site's lens library holds lens {lens_id}; the site has its record."
    assert after['result']['response'] == {**before['result']['response'], 'held': True, 'siteRecord': 'registered'}

    # Sent again: nothing is uploaded, and nothing on the site changes.
    engine.calls.clear()
    stamp = {path: path.stat().st_mtime_ns for path in (entry / 'jacobians.safetensors', entry / 'entry.json',
                                                         record_file(engine.root, lens_id))}
    code, again = push(world, engine, root, lens_id)
    assert code == 0 and again['changed'] is False
    assert again['result']['response'] == {**response, 'alreadyHeld': True, 'uploadedParts': 0, 'uploadedBytes': 0,
                                           'recordChanged': False, 'changed': False}
    assert again['message'] == (f"The site's lens library already holds lens {lens_id}, so nothing was uploaded; "
                                'the site has its record.')
    assert engine.sent('PUT') == [] and engine.sent('POST', '/begin|/complete') == []
    assert [path.replace(sha, '<sha>') for _, path in engine.calls] == [
        '/api/capabilities', '/api/lens-library/<sha>', '/api/lens-library/register']
    assert {path: path.stat().st_mtime_ns for path in stamp} == stamp


def test_a_runner_keeps_a_library_of_its_own_and_does_not_show_its_records(world):
    """A runner-role engine with nothing declared keeps a library beside its records. It serves no workbench
    route, so lens-status cannot say whether it has the record, and says so."""
    root, lens_id, _ = world.researcher()
    engine = world.engine(role='runner', library=None)
    sha = digest(converted(root, lens_id).read_bytes())
    code, pushed = push(world, engine, root, lens_id)
    assert code == 0 and pushed['result']['response']['uploadedParts'] == 4, pushed
    assert (engine.entry(sha) / 'jacobians.safetensors').read_bytes() == converted(root, lens_id).read_bytes()
    assert json.loads(record_file(engine.root, lens_id).read_bytes())['lensID'] == lens_id
    code, answered = status(world, engine, root, lens_id)
    assert code == 0 and answered['result']['response']['held'] is True
    assert answered['result']['response']['siteRecord'] == 'unknown'
    assert answered['message'] == (f"The site's lens library holds lens {lens_id}; this engine does not show its "
                                   'lens records here, so whether it has the record is not known.')


# --- 2. a request that refers to its lens runs on the library's bytes ------------

def package(world, root, request_file, archive, *flags):
    code, plan = world.cli(None, '--root', str(root), 'science', 'input-plan', str(request_file), *flags)
    assert code == 0, plan
    code, packed = world.cli(None, '--root', str(root), 'science', 'package', str(request_file), '--archive',
                             str(archive), '--plan-sha256', plan['result']['planSHA256'], *flags)
    assert code == 0, packed
    return plan['result'], packed['result']


def upload(world, engine, root, archive):
    return world.cli(engine, '--root', str(root), 'runner', 'upload', str(archive), '--runner', SERVER)


def stage(world, engine, root, uploaded):
    return world.cli(engine, '--root', str(root), 'runner', 'science-stage', uploaded['result']['runnerPath'],
                     '--sha256', uploaded['result']['sha256'], '--runner', SERVER)


def test_a_request_that_refers_to_its_lens_is_small_and_runs_on_the_librarys_bytes(world, monkeypatch):
    root, lens_id, _ = world.researcher()
    engine = world.engine()
    lens = converted(root, lens_id)
    sha, size = digest(lens.read_bytes()), lens.stat().st_size
    request, request_file = probe_request(root, lens_id)
    relative = f'runs/jlens-lenses/{lens_id}/jacobians.safetensors'
    # What the lens reads, computed in the researcher's own workspace: the reference for the job below.
    own = batch.probe_batch(batch.ProbeBatchConfig.from_dict(request['parameters']['config']), root=str(root),
                            log=lambda *_: None, model=tiny_model())
    expected = readings(own)

    # Carried, the request is larger than this engine's upload limit (issue #13), and the refusal says what to do.
    _, carried = package(world, root, request_file, world.tmp_path / 'carried.tar.gz')
    assert relative in members(carried['bundlePath']) and Path(carried['bundlePath']).stat().st_size > 8192
    code, refused = upload(world, engine, root, carried['bundlePath'])
    assert (code, refused['state'], refused['error']['code']) == (65, 'refused', 'runnerRefused')
    assert refused['error']['repairAction'] == FIXTURE['oversize']['pythonRepair']
    assert 'STEERLAB_MAX_UPLOAD_BYTES' in refused['error']['reason']

    plan, packed = package(world, root, request_file, world.tmp_path / 'referred.tar.gz', '--lens-reference')
    named = [{'lensID': lens_id, 'convertedSHA256': sha, 'bytes': size}]
    assert plan['libraryLenses'] == named and relative not in [entry['path'] for entry in plan['files']]
    # The archive holds the lens's record and no converted file, and is smaller than the lens it uses.
    names = members(packed['bundlePath'])
    assert relative not in names and f'runs/jlens-lenses/{lens_id}/lens.json' in names
    assert not any(name.endswith('.safetensors') for name in names)
    assert Path(packed['bundlePath']).stat().st_size < size
    described = json.loads(description_of(packed['bundlePath'])[0])
    assert described['context']['lensReference'] is True and described['context']['libraryLenses'] == named

    # The site does not hold the lens yet. The upload is accepted (the engine has a library), and staging is
    # refused by the engine, by name, with the repair that works.
    code, uploaded = upload(world, engine, root, packed['bundlePath'])
    assert code == 0 and uploaded['result']['sha256'] == packed['bundleSha256'], uploaded
    code, refused = stage(world, engine, root, uploaded)
    assert code == 65 and refused['state'] == 'refused', refused
    assert 'lensNotInLibrary' in refused['error']['reason']
    assert (f"This engine's lens library does not hold the lens '{lens_id}' (converted SHA-256 {sha})."
            in refused['error']['reason'])
    assert refused['error']['repairAction'] == (
        f'Send the lens once with `remote lens-push {lens_id}` (or `runner lens-push`), then stage the request again.')
    assert not list((engine.root / 'runs').glob('diagnostic-input-*'))

    # The repair, carried out: send the lens once, then stage the same upload again.
    assert push(world, engine, root, lens_id)[0] == 0
    code, staged = stage(world, engine, root, uploaded)
    assert code == 0, staged
    capsule = Path(staged['result']['response']['executionRoot'])
    assert sorted(path.relative_to(capsule).as_posix() for path in capsule.rglob('*') if path.is_file()) == [
        entry['path'] for entry in plan['files']]
    code, reviewed = world.cli(engine, '--root', str(root), 'runner', 'science-plan',
                               staged['result']['response']['localRequestPath'], '--runner', SERVER)
    assert code == 0, reviewed
    admitted = reviewed['result']['response']
    # What was admitted is the closure of the execution copy that names the lens by hash, built here by hand.
    closure = plan_by_hand(request, capsule, [entry['path'] for entry in plan['files']])
    by_hand = {key: closure[key] for key in ('schemaVersion', 'request', 'sourceRoot', 'files')} | {'libraryLenses': named}
    by_hand['planSHA256'] = hashlib.sha256(canonical(by_hand)).hexdigest()
    assert admitted['inputSHA256'] == hashlib.sha256(canonical(by_hand)).hexdigest()

    # The only copy of the lens on the engine's side is the library's, and the researcher's own is not what
    # the job reads: it is moved away before the job runs.
    held = engine.entry(sha) / 'jacobians.safetensors'
    assert engine.lens_files() == [str(held)]
    lens.rename(root / 'moved-away.safetensors')
    code, record, resolved = engine.run(admitted, 'referred', monkeypatch)
    assert code == 0 and record['status'] == 'succeeded', record.get('error')
    assert record['result']['prompts'] == len(PROMPTS)
    assert resolved and set(resolved) == {str(held.resolve())}
    assert readings(record['result']) == expected
    assert engine.lens_files() == [str(held)]                      # and the job made no copy of it


# --- 3. a request packaged without the flag is the request it always was ---------

def test_a_request_packaged_without_the_flag_is_what_it_always_was_whether_or_not_the_site_holds_the_lens(
        world, monkeypatch):
    root, lens_id, _ = world.researcher()
    engine = world.engine(limit=1 << 20)
    request, request_file = probe_request(root, lens_id)
    folder = f'runs/jlens-lenses/{lens_id}'
    paths = ['prompts/probe-set.jsonl', f'{folder}/lens.json', f'{folder}/import-receipt.json',
             f'{folder}/jacobians.safetensors']
    expected = plan_by_hand(request, root, paths)       # exactly the four fields a plan had before the library

    plan, packed = package(world, root, request_file, world.tmp_path / 'carried.tar.gz')
    assert canonical({key: value for key, value in plan.items() if key != 'changed'}) == canonical(expected)
    metadata = {'schemaVersion': 1, 'kind': 'diagnosticInput', 'entries': expected['files'],
                'context': {'request': request, 'sourcePlanSHA256': expected['planSHA256']}}
    described, names = description_of(packed['bundlePath'])
    assert described == canonical(metadata) and names == sorted(paths + ['steerlab-bundle.json'])

    def admitted():
        code, reviewed = world.cli(engine, '--root', str(root), 'runner', 'science-plan',
                                   staged['result']['response']['localRequestPath'], '--runner', SERVER)
        assert code == 0, reviewed
        return reviewed['result']['response']

    # On an engine whose library is empty.
    code, uploaded = upload(world, engine, root, packed['bundlePath'])
    assert code == 0, uploaded
    code, staged = stage(world, engine, root, uploaded)
    assert code == 0, staged
    capsule = Path(staged['result']['response']['executionRoot'])
    assert (capsule.parent / 'input.json').read_bytes() == canonical(metadata)
    assert sorted(path.relative_to(capsule).as_posix() for path in capsule.rglob('*') if path.is_file()) == sorted(paths)
    without = admitted()
    assert without['inputSHA256'] == hashlib.sha256(canonical(plan_by_hand(request, capsule, paths))).hexdigest()
    assert 'libraryLenses' not in json.dumps(without) and 'lensReference' not in json.dumps(without)
    assert not engine.library.exists()                  # a carried request never made the engine look for one

    # The same staged request once the site also holds the lens: the same admitted plan, byte for byte but for
    # the engine's own process, and the job still reads the copy the request carried.
    assert push(world, engine, root, lens_id)[0] == 0
    held = admitted()
    assert canonical(held) == canonical(without)
    code, record, resolved = engine.run(held, 'carried', monkeypatch)
    assert code == 0 and record['status'] == 'succeeded', record.get('error')
    assert resolved and set(resolved) == {str((capsule / folder / 'jacobians.safetensors').resolve())}
    assert 'lensLibrary' not in record['result']


# --- 4. an interrupted push, and its resume --------------------------------------

@pytest.mark.parametrize('when', ['before', 'after'])
def test_an_interrupted_push_is_resumed_and_sends_only_what_the_site_lacks(world, when):
    """One part's request is lost: before it reached the engine, or after the engine kept it and before its
    answer came back. Run again, the push sends only what the site still lacks."""
    root, lens_id, _ = world.researcher()
    engine = world.engine()
    data = converted(root, lens_id).read_bytes()
    sha = digest(data)
    engine.lost.append(('PUT', '/parts/2$', when))
    code, broken = push(world, engine, root, lens_id)
    assert (code, broken['state'], broken['error']['code']) == (70, 'failed', 'lensUploadInterrupted')
    assert broken['error']['repairAction'] == FIXTURE['interrupted']['repairAction']
    assert broken['error']['reason'].startswith(f'Sending lens {lens_id} to the site was interrupted before the site '
                                                'confirmed that it holds the whole lens and has its record (')
    # The site keeps what it received, publishes nothing, and has no record yet.
    received = [0, 1] if when == 'before' else [0, 1, 2]
    assert engine.get(f'/api/lens-library/{sha}').json() == {'held': False, 'bytes': len(data), 'receivedParts': received}
    assert not engine.entry(sha).exists() and not record_file(engine.root, lens_id).exists()

    engine.calls.clear()
    code, resumed = push(world, engine, root, lens_id)
    assert code == 0, resumed
    missing = [index for index in range(4) if index not in received]
    assert engine.sent('PUT') == [f'/api/lens-library/{sha}/parts/{index}' for index in missing]
    response = resumed['result']['response']
    assert response['uploadedParts'] == len(missing) and response['alreadyHeld'] is False and response['registered'] is True
    assert response['uploadedBytes'] == HIDDEN * HIDDEN * 4 * len(missing)        # one layer of float32 each
    assert (engine.entry(sha) / 'jacobians.safetensors').read_bytes() == data
    assert json.loads(record_file(engine.root, lens_id).read_bytes())['converted']['sha256'] == sha


# --- 5. a site that moves large files with its own tool --------------------------

def test_a_site_that_forbids_http_transfer_adopts_a_copy_placed_beneath_its_run_root(world):
    root, lens_id, _ = world.researcher()
    engine = world.engine(transfer='globus')
    lens = converted(root, lens_id)
    data = lens.read_bytes()
    sha = digest(data)
    relative = f'runs/jlens-lenses/{lens_id}/jacobians.safetensors'

    # Without --adopt the client sends nothing, and says how the bytes get there.
    code, refused = push(world, engine, root, lens_id)
    assert (code, refused['state'], refused['error']['code']) == (65, 'refused', 'externalTransferRequired')
    assert refused['error']['reason'] == FIXTURE['externalTransfer']['reason'].replace(FIXTURE['server'], SERVER)
    assert refused['error']['repairAction'] == FIXTURE['externalTransfer']['repairAction'].replace(FIXTURE['file'], relative)
    assert engine.sent('POST') == [] and engine.sent('PUT') == []
    # The engine itself refuses a sender that asks anyway, in its own words: a 403 that names a code is the
    # engine's refusal, never a token problem, on the manifest and on a part.
    with world.client_class(base_url=SERVER, token=TOKEN, http_client=engine.http) as direct:
        for call in (lambda: direct.lens_library_begin(sha, len(data), [{'index': 0, 'offset': 0, 'bytes': len(data), 'sha256': sha}]),
                     lambda: direct.lens_library_put_part(sha, {'index': 0, 'offset': 0, 'bytes': len(data), 'sha256': sha}, str(lens))):
            with pytest.raises(runner_api.RunnerRefusal) as caught:
                call()
            assert caught.value.code == 'external_transfer_required'
            assert caught.value.reason == ('This deployment does not permit HTTP artifact transfer, so a lens cannot '
                                           'be sent to it in parts.')
            assert caught.value.repair_action == lens_library.ADOPT_REPAIR
    assert not engine.library.exists()

    # The site's own transfer tool puts the file beneath the engine's run root; the push then names that copy.
    copy = engine.root / 'runs/transfers/jacobians.safetensors'
    copy.parent.mkdir(parents=True)
    copy.write_bytes(data)
    engine.calls.clear()
    code, adopted = push(world, engine, root, lens_id, '--adopt', str(copy))
    assert code == 0, adopted
    response = adopted['result']['response']
    assert (response['adopted'], response['uploadedParts'], response['uploadedBytes'], response['registered']) == (
        True, 0, 0, True)
    assert adopted['message'] == (f'The site checked its copy of lens {lens_id} and keeps it in its lens library; '
                                  'the site has its record.')
    assert engine.sent('PUT') == [] and engine.sent('POST') == ['/api/lens-library/adopt', '/api/lens-library/register']
    entry = engine.entry(sha)
    assert (entry / 'jacobians.safetensors').read_bytes() == data
    assert json.loads((entry / 'entry.json').read_bytes())['origin'] == 'adopted'
    assert copy.read_bytes() == data                                # the copy is left alone
    assert json.loads(record_file(engine.root, lens_id).read_bytes())['converted']['sha256'] == sha
    # A file that is not the lens, or is not beneath the run root, is refused in the engine's words.
    other = world.engine(transfer='globus')
    wrong = other.root / 'runs/transfers/jacobians.safetensors'
    wrong.parent.mkdir(parents=True)
    wrong.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    code, refused = push(world, other, root, lens_id, '--adopt', str(wrong))
    assert (code, refused['error']['code']) == (65, 'lensAdoptMismatch')
    assert refused['error']['reason'] == f'That file does not have the SHA-256 {sha}, so nothing was added to the lens library.'
    code, refused = push(world, other, root, lens_id, '--adopt', str(lens))
    assert (code, refused['error']['code']) == (65, 'lensAdoptPathInvalid')
    assert refused['error']['repairAction'] == lens_library.ADOPT_REPAIR
    assert not other.entry(sha).exists() and not record_file(other.root, lens_id).exists()


# --- 6. an engine with no lens library -------------------------------------------

@pytest.mark.parametrize('options', [
    {'role': 'workbench', 'library': None},      # a workbench with nothing declared: its lenses are its workspace's
    {'role': None, 'library': None},             # no role set is a workbench
    {'role': 'runner', 'library': 'empty'},      # an engine told it has none
], ids=['workbench', 'no role', 'told it has none'])
def test_an_engine_with_no_lens_library_is_sent_nothing(world, options):
    root, lens_id, _ = world.researcher()
    engine = world.engine(limit=1 << 20, **options)
    sha = digest(converted(root, lens_id).read_bytes())
    code, refused = push(world, engine, root, lens_id)
    assert (code, refused['state'], refused['error']['code']) == (65, 'refused', 'lensLibraryUnavailable')
    assert refused['error']['reason'] == FIXTURE['unavailable']['pushReason'].replace(FIXTURE['server'], SERVER)
    assert refused['error']['repairAction'] == FIXTURE['unavailable']['pushRepairAction']
    assert engine.calls == [('GET', '/api/capabilities')]            # the one question, and nothing sent

    engine.calls.clear()
    code, answered = status(world, engine, root, lens_id)
    assert code == 0 and answered['result']['response']['libraryAvailable'] is False
    assert answered['result']['response']['held'] is False
    assert engine.sent('GET', '/api/lens-library/') == []

    request, request_file = probe_request(root, lens_id)
    _, packed = package(world, root, request_file, world.tmp_path / 'referred.tar.gz', '--lens-reference')
    engine.calls.clear()
    code, refused = upload(world, engine, root, packed['bundlePath'])
    assert (code, refused['state'], refused['error']['code']) == (65, 'refused', 'lensLibraryUnavailable')
    assert refused['error']['reason'] == FIXTURE['unavailable']['archiveReason'].replace(FIXTURE['server'], SERVER)
    assert refused['error']['repairAction'] == FIXTURE['unavailable']['archiveRepairAction']
    assert engine.calls == [('GET', '/api/capabilities')]
    assert not (engine.root / 'runs').exists()

    # The same request without the flag carries its lens, and this engine takes it as it always did.
    _, carried = package(world, root, request_file, world.tmp_path / 'carried.tar.gz')
    code, uploaded = upload(world, engine, root, carried['bundlePath'])
    assert code == 0, uploaded
    assert stage(world, engine, root, uploaded)[0] == 0
    # The engine says the same of itself to a sender that asks anyway, with its own code and repair.
    with world.client_class(base_url=SERVER, token=TOKEN, http_client=engine.http) as direct:
        with pytest.raises(runner_api.RunnerRefusal) as caught:
            direct.lens_library_status(sha)
    assert caught.value.code == 'lensLibraryUnavailable'
    assert caught.value.reason == 'This engine has no lens library: no folder is set for one, or the folder cannot be written.'
    assert '`STEERLAB_LENS_LIBRARY_ROOT`' in caught.value.repair_action


def test_a_request_that_names_no_lens_goes_to_any_engine_with_or_without_the_flag(world):
    """Packaged with --lens-reference, a request that reads no lens refers to none (`libraryLenses: []`), and
    an engine with no lens library takes it as it takes any archive. So the client sends it: there is no lens
    for it to say the engine cannot hold."""
    from test_battery_run import _lines
    root = world.tmp_path / 'researcher'
    battery = root / 'prompts/batteries/example.jsonl'
    battery.parent.mkdir(parents=True)
    battery.write_text(_lines())
    request = {'operation': 'battery', 'parameters': {'batteryFile': 'prompts/batteries/example.jsonl',
                                                      'agents': ['baseline'], 'modelID': 'example/model',
                                                      'revision': 'a' * 40}}
    request_file = root / 'request.json'
    request_file.write_text(json.dumps(request))
    plan, packed = package(world, root, request_file, world.tmp_path / 'no-lens.tar.gz', '--lens-reference')
    assert plan['libraryLenses'] == []
    described = json.loads(description_of(packed['bundlePath'])[0])
    assert described['context']['lensReference'] is True and described['context']['libraryLenses'] == []
    for options in ({'role': 'workbench', 'library': None}, {'role': 'runner', 'library': 'empty'}, {}):
        engine = world.engine(limit=1 << 20, **options)
        code, uploaded = upload(world, engine, root, packed['bundlePath'])
        assert code == 0, uploaded
        code, staged = stage(world, engine, root, uploaded)
        assert code == 0, staged
        assert staged['result']['response']['operation'] == 'battery'


# --- 7. a record that holds a number that is not finite --------------------------

def test_a_record_with_a_nan_in_a_qualification_is_sent_and_the_sites_record_holds_null_there(world):
    root, lens_id, _ = world.researcher()
    engine = world.engine()
    sha = digest(converted(root, lens_id).read_bytes())
    # A qualification stores every check's measured values, and one that could not be scored records NaN.
    record = lens_store.resolve(lens_id, str(root))
    qualifications.append_qualification(record, schemas.Qualification(
        qualificationID='q-unscored', modelID=MODEL, revision=REVISION, dtype='float32', passed=False,
        lensSHA256=record.source.tensorSHA256, convertedSHA256=sha, layers=[0, 2],
        checks={'results': {'agreement': {'score': float('nan'), 'threshold': 0.5}}},
        qualifiedAt='2026-11-02T00:00:00Z'), root=str(root))
    assert b'NaN' in record_file(root, lens_id).read_bytes()

    code, pushed = push(world, engine, root, lens_id)
    assert code == 0, pushed
    assert pushed['result']['response']['recordNonFiniteNumbers'] == 1 and pushed['result']['response']['registered'] is True
    assert pushed['message'].endswith('; the site has its record, with 1 number in the record that is not finite '
                                      'stored as null.')
    # The site's record is strict JSON, with null where the workspace has NaN, and everything else as it was.
    site = strict_loads(record_file(engine.root, lens_id).read_bytes())
    assert site['qualifications'][0]['checks'] == {'results': {'agreement': {'score': None, 'threshold': 0.5}}}
    mine = json.loads(record_file(root, lens_id).read_bytes())
    mine['qualifications'][0]['checks']['results']['agreement']['score'] = None
    assert site == mine
    with environment(engine.env):
        assert [q.qualificationID for q in lens_store.resolve(lens_id).qualifications] == ['q-unscored']
    # Sent again, the two records are the same record: nothing is added, and nothing is called different.
    code, again = push(world, engine, root, lens_id)
    assert code == 0 and again['result']['response']['recordChanged'] is False and again['changed'] is False
    assert len(strict_loads(record_file(engine.root, lens_id).read_bytes())['qualifications']) == 1
    code, answered = status(world, engine, root, lens_id)
    assert code == 0 and answered['result']['response']['siteRecord'] == 'registered'
    # A site that holds the NaN itself (the same qualification was made there, and the engine writes one as it
    # is) reads its own record the way the client sends one, so the two are still the same qualification.
    record_file(engine.root, lens_id).write_bytes(record_file(root, lens_id).read_bytes())
    assert b'NaN' in record_file(engine.root, lens_id).read_bytes()
    code, same = push(world, engine, root, lens_id)
    assert code == 0 and same['result']['response']['recordChanged'] is False, same
    assert record_file(engine.root, lens_id).read_bytes() == record_file(root, lens_id).read_bytes()


# --- 8. the engine's refusals reach the researcher as the engine gave them -------

def test_an_engine_refusal_reaches_the_client_with_the_engines_code_sentence_and_repair(world):
    root, lens_id, _ = world.researcher()
    engine = world.engine()
    data = converted(root, lens_id).read_bytes()
    sha = digest(data)
    mine = json.loads(record_file(root, lens_id).read_bytes())
    # The site already has a lens of this name for other bytes.
    other = schemas.JLensRecord.from_dict({**mine, 'source': {**mine['source'], 'tensorSHA256': 'e' * 64},
                                           'converted': {**mine['converted'], 'sha256': 'f' * 64}})
    lens_store.save(other, str(engine.root))
    code, refused = push(world, engine, root, lens_id)
    assert (code, refused['state'], refused['error']['code']) == (65, 'refused', 'lensRecordDiffers')
    assert refused['error']['reason'] == (
        f"The site already has a lens named '{lens_id}' for different bytes. The site's record has converted "
        f"SHA-256 {'f' * 64} and source tensor SHA-256 {'e' * 64}; the record sent has converted SHA-256 {sha} and "
        f"source tensor SHA-256 {mine['source']['tensorSHA256']}. Nothing was changed.")
    assert refused['error']['repairAction'] == (
        'One lens name stands for one set of bytes. Use the lens the site has under that name. A request packaged '
        'with `--lens-reference` carries its own record and does not depend on this registration.')
    assert json.loads(record_file(engine.root, lens_id).read_bytes())['converted']['sha256'] == 'f' * 64
    code, answered = status(world, engine, root, lens_id)
    assert code == 0 and answered['result']['response']['siteRecord'] == 'different'
    assert answered['message'] == (f"The site's lens library holds lens {lens_id}; the site has a different lens "
                                   'under that name.')


def test_an_entry_a_newer_engine_wrote_is_reported_by_status_and_refused_by_name_on_a_push(world):
    """`GET` answers `newerEntry: true` beside `held: false` for an entry whose record names a later schema.
    The client reads that answer as it reads any other, and the push that follows is refused in the engine's
    words."""
    root, lens_id, _ = world.researcher()
    engine = world.engine()
    data = converted(root, lens_id).read_bytes()
    sha = digest(data)
    entry = engine.entry(sha)
    entry.mkdir(parents=True)
    (entry / 'jacobians.safetensors').write_bytes(data)
    (entry / 'entry.json').write_text(json.dumps({'schemaVersion': 2, 'convertedSHA256': sha, 'bytes': len(data),
                                                  'origin': 'pushed', 'addedAt': '2026-11-01T00:00:00Z', 'kept': 'for'}))
    assert engine.get(f'/api/lens-library/{sha}').json() == {'held': False, 'bytes': None, 'receivedParts': None,
                                                             'newerEntry': True}
    code, answered = status(world, engine, root, lens_id)
    assert code == 0 and answered['result']['response']['held'] is False
    code, refused = push(world, engine, root, lens_id)
    assert (code, refused['state'], refused['error']['code']) == (65, 'refused', 'lensLibraryEntryNewer')
    assert refused['error']['reason'] == (
        f"The lens library's entry for {sha} was written by a newer version of the engine (entry schema 2; this "
        'engine reads schema 1), so this engine left it as it is and did not read it.')
    assert refused['error']['repairAction'] == 'Update this engine to the version that wrote the entry, or a later one, then try again.'
    assert engine.sent('PUT') == [] and sorted(path.name for path in entry.iterdir()) == ['entry.json', 'jacobians.safetensors']


def test_a_lens_that_cannot_be_referred_to_is_refused_at_packaging_by_name(world):
    """`lensNotReferable` is the engine-side planner's refusal, raised on the researcher's machine. It reaches
    the command line (and, through the same fields, the Mac's bridge) with its own code and repair."""
    from steerlab_server.client import diagnostic_workspace
    from steerlab_server.client.runtime_identity import source_sha256
    root, lens_id, _ = world.researcher()
    request, request_file = probe_request(root, lens_id)
    folder = Path(root) / 'runs/jlens-lenses' / lens_id
    (folder / 'jacobians.safetensors').rename(folder / 'renamed.safetensors')
    record = json.loads((folder / 'lens.json').read_bytes())
    record['converted']['path'] = f'runs/jlens-lenses/{lens_id}/renamed.safetensors'
    (folder / 'lens.json').write_text(json.dumps(record))
    reason = (f"The lens '{lens_id}' cannot be referred to: its record (`lens.json`) does not name its converted file "
              f'as `jacobians.safetensors` in its own lens folder (`runs/jlens-lenses/{lens_id}/`), where an import '
              'puts it, and a site keeps and serves a lens only under that layout.')
    repair = ('Package the request without `--lens-reference`, so it carries the lens, or import the lens again and '
              'then package with `--lens-reference`.')
    code, refused = world.cli(None, '--root', str(root), 'science', 'input-plan', str(request_file), '--lens-reference')
    assert (code, refused['state'], refused['error']['code']) == (65, 'refused', 'lensNotReferable')
    assert refused['error']['reason'] == reason and refused['error']['repairAction'] == repair
    # What the Mac is handed for the same request, through the bridge it runs.
    payload = {'workspaceRoot': str(root), 'requestFile': str(request_file), 'lensReference': True}
    world.monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps(
        {'action': 'input-plan', 'payload': payload, 'clientSHA256': source_sha256()})))
    world.capsys.readouterr()
    assert diagnostic_workspace.main() == 65
    answer = json.loads(world.capsys.readouterr().out)
    assert (answer['ok'], answer['state'], answer['code'], answer['reason'], answer['repairAction']) == (
        False, 'refused', 'lensNotReferable', reason, repair)
    # Without the flag the same lens is carried, as it always was.
    code, plan = world.cli(None, '--root', str(root), 'science', 'input-plan', str(request_file))
    assert code == 0 and f'runs/jlens-lenses/{lens_id}/renamed.safetensors' in [e['path'] for e in plan['result']['files']]


# --- a lens the site was sent, then imported there --------------------------------

def tree(*folders):
    """Every file beneath these folders, with its size and when it last changed."""
    return {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
            for folder in folders for path in sorted(Path(folder).rglob('*')) if path.is_file()}


def finished(engine, job_id, deadline=30.0):
    """The engine's record of a job once it has stopped, with its worker thread gone."""
    limit = time.monotonic() + deadline
    record = engine.get(f'/api/jobs/{job_id}').json()
    while record.get('status') not in ('succeeded', 'failed', 'cancelled'):
        assert time.monotonic() < limit, record
        time.sleep(0.02)
        record = engine.get(f'/api/jobs/{job_id}').json()
    for thread in threading.enumerate():
        if thread.name.startswith('job-'):
            thread.join(timeout=10.0)
    return record


def test_a_lens_the_site_was_sent_is_found_when_its_files_are_imported_in_the_engines_workspace(world):
    """`register` writes a lens's record with no import receipt, and the bytes stay in the site's lens library.
    Importing the same files in that engine's workspace (the app's "Import my own lens files" on a connected
    workbench) finds that lens: nothing is written, no second copy of the bytes is made, and the site's record,
    with its qualifications, is the one that stays."""
    root, lens_id, description = world.researcher()
    engine = world.engine(limit=1 << 20)        # room to stage the source tensor file whole
    data = converted(root, lens_id).read_bytes()
    sha = digest(data)
    held = engine.entry(sha) / 'jacobians.safetensors'
    assert push(world, engine, root, lens_id)[0] == 0
    folder = engine.root / 'runs/jlens-lenses' / lens_id
    assert sorted(path.name for path in folder.iterdir()) == ['lens.json']      # a record: no receipt, and no bytes

    with environment(engine.env):      # the import is a job of the engine's, and runs after its request returns
        source = 'a' * 32
        for name in ('import.json', 'weights.safetensors'):
            content = (description.parent / name).read_bytes()
            staged = engine.post(f'/api/artifact-imports/stage/{source}/{name}', content=content,
                                 headers={'X-Content-SHA256': digest(content)})
            assert staged.status_code == 200, staged.text
        reference = f'.steerlab/artifact-inputs/{source}/import.json'
        review = engine.post('/api/artifact-imports/plan', json={'descriptionFile': reference})
        assert review.status_code == 200, review.text
        assert artifact_imports.lens_name(review.json()) == lens_id          # one lens, one name, on any machine
        before = tree(engine.root / 'runs', engine.library)
        submitted = engine.post('/api/artifact-imports/import',
                                json={'descriptionFile': reference, 'planSHA256': review.json()['planSHA256']})
        assert submitted.status_code == 200, submitted.text
        job = finished(engine, submitted.json()['jobId'])
    assert job['status'] == 'succeeded', job.get('error')
    result = job['result']
    assert (result['lensID'], result['existing'], result['changed']) == (lens_id, True, False)
    assert 'restored' not in result
    assert result['record'] == json.loads(record_file(engine.root, lens_id).read_bytes())
    # Nothing was written: no receipt, no converted file in the workspace, and the record and the entry as they were.
    assert tree(engine.root / 'runs', engine.library) == before
    assert sorted(path.name for path in folder.iterdir()) == ['lens.json']
    assert engine.lens_files() == [str(held)]

    # A study-style lookup of the lens on the engine: its record by name, and the library's bytes, which are
    # the bytes the record names and the researcher's own.
    with environment(engine.env):
        record = lens_store.resolve(lens_id)
        assert artifact_paths.converted_file(record.lensID, record.converted.path) == str(held.resolve())
        qualifications.verify_converted(record)
        mine = lens_store.resolve(lens_id, str(root))
        for layer in record.sourceLayers:
            assert torch.equal(lens_store.load_layer(record, layer), lens_store.load_layer(mine, layer, root=str(root)))
    code, answered = status(world, engine, root, lens_id)
    assert code == 0 and answered['result']['response']['siteRecord'] == 'registered'

    # The library is a cache. If it loses the entry, the same import puts the lens's file back in the engine's
    # workspace from the reviewed files, and the lookup reads that.
    shutil.rmtree(engine.entry(sha))
    with environment(engine.env):
        submitted = engine.post('/api/artifact-imports/import',
                                json={'descriptionFile': reference, 'planSHA256': review.json()['planSHA256']})
        job = finished(engine, submitted.json()['jobId'])
        assert job['status'] == 'succeeded' and job['result']['restored'] is True, job
        assert (folder / 'jacobians.safetensors').read_bytes() == data
        assert artifact_paths.converted_file(lens_id, record.converted.path) == str((folder / 'jacobians.safetensors').resolve())
    assert sorted(path.name for path in folder.iterdir()) == ['jacobians.safetensors', 'lens.json']


# --- one place says which numbers are not finite ---------------------------------

def test_the_engine_and_the_clients_share_one_reading_of_a_number_that_is_not_finite():
    """The client sends null for such a number, the engine refuses one that arrives anyway and names where it
    is, and the engine reads its own stored record the client's way before comparing. All three are
    `experiment/strict_json.py`."""
    from steerlab_server.client import lens_transfer
    from steerlab_server.experiment import strict_json
    assert lens_library.strict is strict_json.strict is lens_transfer.strict
    assert lens_library.first_non_finite is strict_json.first_non_finite
    assert not hasattr(lens_library, '_finite') and not hasattr(lens_library, '_non_finite')
    value = {'a': [1.5, {'b': float('inf')}], 'c': float('nan'), 'd': (2, float('-inf'))}
    assert strict_json.first_non_finite(value, 'the record') == 'the record.a[1].b'
    assert strict_json.first_non_finite({'c': float('nan')}) == 'the value.c'
    assert strict_json.first_non_finite(float('nan'), 'it') == 'it'
    cleaned, count = strict_json.strict(value)
    assert (cleaned, count) == ({'a': [1.5, {'b': None}], 'c': None, 'd': [2, None]}, 3)
    # Exactly the numbers `strict` replaces are the ones that are found, and nothing else is.
    assert strict_json.first_non_finite(cleaned) is None
    for finite in (0.0, -0.0, 1e308, 10**400, True, None, 'NaN', [], {}):
        assert strict_json.first_non_finite(finite) is None and strict_json.strict(finite) == (finite, 0)
