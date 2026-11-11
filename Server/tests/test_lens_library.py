"""One lens per site, found by its hash (public issues #13 and #14).

The library keeps a lens's converted file once, under its SHA-256, outside any
workspace. These tests drive it the way a client does, over its routes, with
tiny synthetic lenses: where the library is, what it says it can do, an upload
in checked parts (planned here from the safetensors layout alone, as a sender
plans it), a file adopted from the site, a record registered in the serving
workspace, and every way a path or a key could reach outside the library.
"""
import errno
import hashlib
import json
import os
import stat
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from safetensors.numpy import save_file

from steerlab_server.api import lens_library_routes
from steerlab_server.api.profile import ServerProfile, capability_snapshot
from steerlab_server.jlens import artifact_paths, lens_library, lens_store, schemas

LENS_ID = 'custom-lens-example'
CANONICAL = f'runs/jlens-lenses/{LENS_ID}/jacobians.safetensors'
UNAVAILABLE_REPAIR = ('Package the request without `--lens-reference`, or give this engine a lens library: set '
                      '`STEERLAB_LENS_LIBRARY_ROOT` to a folder it can write (a site profile sets it from its '
                      '`lensLibrary` folder), then start the engine again.')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_converted(path, seed=0, layers=3, width=4):
    """A tiny converted lens file: one square float32 tensor per layer."""
    generator = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({f'layer_{i}': generator.standard_normal((width, width)).astype(np.float32)
               for i in range(layers)}, str(path))
    return digest(path.read_bytes())


def part_plan(path, limit):
    """The parts a sender plans, from the safetensors layout alone: the header
    first, then one part per tensor in file order, any part over the limit
    split. Independent of the engine, so it checks the contract and not the
    code that implements it."""
    data = Path(path).read_bytes()
    (length,) = struct.unpack('<Q', data[:8])
    header = json.loads(data[8:8 + length])
    start = 8 + length
    spans = [(0, start)] + sorted((start + tensor['data_offsets'][0], start + tensor['data_offsets'][1])
                                  for name, tensor in header.items() if name != '__metadata__')
    pieces = []
    for low, high in spans:
        while high - low > limit:
            pieces.append((low, low + limit))
            low += limit
        if high > low:
            pieces.append((low, high))
    return {'bytes': len(data), 'parts': [{'index': index, 'offset': low, 'bytes': high - low,
                                           'sha256': digest(data[low:high])}
                                          for index, (low, high) in enumerate(pieces)]}


def send_part(client, sha, plan, data, index, *, body=None, announced=None):
    part = plan['parts'][index]
    return client.put(f'/api/lens-library/{sha}/parts/{index}',
                      content=data[part['offset']:part['offset'] + part['bytes']] if body is None else body,
                      headers={'X-Content-SHA256': announced or part['sha256']})


def push(client, path, *, limit=64):
    """What `lens-push` does with the bytes: ask, begin, send what is missing, complete."""
    data = Path(path).read_bytes()
    sha, plan = digest(data), part_plan(path, limit)
    begun = client.post(f'/api/lens-library/{sha}/begin', json=plan)
    assert begun.status_code == 200, begun.text
    for index in begun.json()['missingParts']:
        sent = send_part(client, sha, plan, data, index)
        assert sent.status_code == 200 and sent.json() == {'received': True}, sent.text
    return sha, begun.json(), client.post(f'/api/lens-library/{sha}/complete', json={})


def refusal(response):
    detail = response.json()['detail']
    assert detail['reason'] == detail['message'] and detail['repairAction']
    return detail


@pytest.fixture
def site(tmp_path, monkeypatch):
    root = tmp_path / 'site'
    root.mkdir()
    monkeypatch.setenv('STEERLAB_ROOT', str(root))
    monkeypatch.setenv('STEERLAB_METADATA_ROOT', str(root / '.steerlab'))
    monkeypatch.setenv('STEERLAB_SERVER_ROLE', 'workstation')
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'local')
    # 64 bytes: one 4 x 4 float32 tensor is exactly one part, and the header is split.
    monkeypatch.setenv('STEERLAB_MAX_UPLOAD_BYTES', '64')
    for name in ('STEERLAB_LENS_LIBRARY_ROOT', 'SLURM_JOB_ID', 'STEERLAB_TRANSFER_METHOD', 'STEERLAB_RUN_ROOT'):
        monkeypatch.delenv(name, raising=False)
    app = FastAPI()
    app.include_router(lens_library_routes.build_router(SimpleNamespace(jobs=None)))
    source = tmp_path / 'researcher' / CANONICAL
    return SimpleNamespace(root=root, library=root / '.steerlab/lens-library', app=app, client=TestClient(app),
                           source=source, sha=write_converted(source))


def record_for(sha, *, path=CANONICAL, lens_id=LENS_ID, tensor='c' * 64, qualifications=()):
    return schemas.JLensRecord(
        lensID=lens_id,
        source=schemas.SourceRef(repo='example/lens', folder='source', tensorFile='tensorFile.safetensors',
                                 configFile='description.json', tensorSHA256=tensor, configSHA256='d' * 64),
        fit=schemas.FitProvenance(modelID='example/model', dtype='float32'),
        sourceLayers=[0, 1, 2], dModel=4, targetLayer=3, nPrompts=5,
        converted=schemas.ConvertedRef(path=path, dtype='float32', sha256=sha, layerCount=3),
        tier='testing', tierSource='custom-artifact', importedAt='2026-11-01T00:00:00Z',
        qualifications=list(qualifications)).to_dict()


def qualification(sha, identifier='q-1', **changes):
    values = dict(qualificationID=identifier, modelID='example/model', revision='a' * 40, dtype='float32',
                  passed=True, lensSHA256='c' * 64, convertedSHA256=sha, layers=[0, 1],
                  checks={'agreement': 0.99}, qualifiedAt='2026-11-02T00:00:00Z')
    values.update(changes)
    return schemas.Qualification(**values)


# --- where the library is, and what the engine says about it --------------------

def test_the_root_is_named_by_the_site_and_never_guessed_on_a_cluster(tmp_path, monkeypatch):
    for name in ('STEERLAB_LENS_LIBRARY_ROOT', 'SLURM_JOB_ID'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('STEERLAB_ROOT', str(tmp_path))
    monkeypatch.setenv('STEERLAB_METADATA_ROOT', str(tmp_path / 'records'))
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'local')
    assert ServerProfile.from_env().lens_library_root == str(tmp_path / 'records/lens-library')
    # A deployment that runs its work through Slurm has no library until its
    # site profile names a folder; so has a job, which cannot see its launcher.
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'slurm')
    assert ServerProfile.from_env().lens_library_root is None
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'local')
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    assert ServerProfile.from_env().lens_library_root is None
    monkeypatch.setenv('STEERLAB_LENS_LIBRARY_ROOT', str(tmp_path / 'lenses'))
    assert ServerProfile.from_env().lens_library_root == str(tmp_path / 'lenses')
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'slurm')
    assert ServerProfile.from_env().lens_library_root == str(tmp_path / 'lenses')
    # Set but empty means "no library" wherever it is read: how an engine tells a local child it has none.
    for executor in ('slurm', 'local'):
        monkeypatch.setenv('STEERLAB_EXECUTOR', executor)
        monkeypatch.setenv('STEERLAB_LENS_LIBRARY_ROOT', '')
        monkeypatch.delenv('SLURM_JOB_ID', raising=False)
        assert ServerProfile.from_env().lens_library_root is None


def test_capabilities_say_whether_lenses_are_kept_and_how_large_a_part_may_be(site, monkeypatch):
    snapshot = capability_snapshot()
    assert snapshot['lensLibrary'] == {'schemaVersion': 1, 'available': True, 'partMaxBytes': 64}
    assert snapshot['workspaceRootPolicy']['lensLibraryRoot'] == str(site.library)
    monkeypatch.delenv('STEERLAB_MAX_UPLOAD_BYTES')
    assert capability_snapshot()['lensLibrary']['partMaxBytes'] == 268435456   # min(4 GiB, 256 MiB)
    monkeypatch.setenv('STEERLAB_MAX_UPLOAD_BYTES', str(1024**3))
    assert capability_snapshot()['lensLibrary']['partMaxBytes'] == 268435456
    # Configured, but a file stands where the folder would be: nothing can be written.
    blocked = site.root / 'blocked'
    blocked.write_text('not a folder')
    monkeypatch.setenv('STEERLAB_LENS_LIBRARY_ROOT', str(blocked / 'lenses'))
    assert capability_snapshot()['lensLibrary']['available'] is False
    # `profile validate` says why, and only for a folder the site declared.
    from steerlab_server.api.profile import validate_profile
    (check,) = [c for c in validate_profile()['checks'] if c['name'] == 'lensLibraryRoot']
    assert check['status'] == 'warn' and 'cannot be written' in check['message']
    monkeypatch.setenv('STEERLAB_LENS_LIBRARY_ROOT', str(site.root / 'not-yet-made' / 'lenses'))
    (check,) = [c for c in validate_profile()['checks'] if c['name'] == 'lensLibraryRoot']
    assert check['status'] == 'ok' and capability_snapshot()['lensLibrary']['available'] is True
    monkeypatch.delenv('STEERLAB_LENS_LIBRARY_ROOT')
    assert not [c for c in validate_profile()['checks'] if c['name'] == 'lensLibraryRoot']
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'slurm')
    monkeypatch.setenv('STEERLAB_SERVER_ROLE', 'controller')
    snapshot = capability_snapshot()
    assert snapshot['lensLibrary']['available'] is False
    assert snapshot['workspaceRootPolicy']['lensLibraryRoot'] is None


def test_an_engine_without_a_library_refuses_every_route_with_the_repair(site, monkeypatch):
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'slurm')
    plan = part_plan(site.source, 64)
    calls = [site.client.get(f'/api/lens-library/{site.sha}'),
             site.client.post(f'/api/lens-library/{site.sha}/begin', json=plan),
             site.client.put(f'/api/lens-library/{site.sha}/parts/0', content=b'x',
                             headers={'X-Content-SHA256': plan['parts'][0]['sha256']}),
             site.client.post(f'/api/lens-library/{site.sha}/complete', json={}),
             site.client.post('/api/lens-library/adopt', json={'path': 'lens.safetensors', 'sha256': site.sha}),
             site.client.post('/api/lens-library/register', json={'record': record_for(site.sha)})]
    for response in calls:
        assert response.status_code == 409, response.text
        detail = refusal(response)
        assert detail['code'] == 'lensLibraryUnavailable'
        # The repair works with or without a site profile: it names the variable and the profile's folder.
        assert detail['repairAction'] == UNAVAILABLE_REPAIR
        assert 'STEERLAB_LENS_LIBRARY_ROOT' in UNAVAILABLE_REPAIR and '`lensLibrary` folder' in UNAVAILABLE_REPAIR
    assert not site.library.exists()


def test_a_job_is_given_the_library_only_when_the_deployment_has_one(tmp_path, monkeypatch):
    from steerlab_server.api.executors import SlurmExecutor, SlurmResources, render_slurm_script
    for name in ('STEERLAB_LENS_LIBRARY_ROOT', 'SLURM_JOB_ID'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('STEERLAB_ROOT', str(tmp_path))
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'slurm')
    resources = SlurmResources(gres='A100', walltime='01:00:00')
    without = SlurmExecutor().create_bundle(str(tmp_path / 'one'), ['python', '-c', 'pass'], resources=resources)
    assert 'STEERLAB_LENS_LIBRARY_ROOT' not in without.env
    assert 'LENS_LIBRARY' not in render_slurm_script(without)
    monkeypatch.setenv('STEERLAB_LENS_LIBRARY_ROOT', str(tmp_path / 'lenses'))
    given = SlurmExecutor().create_bundle(str(tmp_path / 'two'), ['python', '-c', 'pass'], resources=resources)
    assert given.env['STEERLAB_LENS_LIBRARY_ROOT'] == str(tmp_path / 'lenses')
    assert f"export STEERLAB_LENS_LIBRARY_ROOT={tmp_path / 'lenses'}" in render_slurm_script(given)
    # Everything else a job is given is unchanged by the library.
    assert {k: v for k, v in given.env.items() if k != 'STEERLAB_LENS_LIBRARY_ROOT'} == without.env


def test_the_routes_take_a_token_and_serve_the_runner_and_the_workbench(site, monkeypatch):
    from steerlab_server.api.app import app, request_is_privileged
    from steerlab_server.api.service_authority import refusal as role_refusal
    routes = [('GET', f'/api/lens-library/{site.sha}'), ('POST', f'/api/lens-library/{site.sha}/begin'),
              ('PUT', f'/api/lens-library/{site.sha}/parts/0'), ('POST', f'/api/lens-library/{site.sha}/complete'),
              ('POST', '/api/lens-library/register'), ('POST', '/api/lens-library/adopt')]
    for role in ('runner', 'workbench'):
        monkeypatch.setenv('STEERLAB_SERVICE_ROLE', role)
        assert all(role_refusal(method, path) is None for method, path in routes)
    # Everything that writes is privileged wherever the engine is not a local loopback process.
    assert all(request_is_privileged(method, path) for method, path in routes[1:])
    monkeypatch.setenv('STEERLAB_AUTH_MODE', 'token')
    monkeypatch.setenv('STEERLAB_AUTH_TOKEN', 'fictional-test-token')
    client = TestClient(app)
    for method, path in routes:
        assert client.request(method, path).status_code == 401, (method, path)
    answered = client.get(routes[0][1], headers={'authorization': 'Bearer fictional-test-token'})
    assert answered.status_code == 200 and answered.json() == {'held': False, 'bytes': None, 'receivedParts': None}


# --- a lens sent in parts --------------------------------------------------------

def test_a_lens_is_sent_once_in_checked_parts_and_published_whole(site):
    before = site.client.get(f'/api/lens-library/{site.sha}')
    assert before.json() == {'held': False, 'bytes': None, 'receivedParts': None}
    plan = part_plan(site.source, 64)
    # Three one-part tensors, and a header that had to be split: several layers and a forced split.
    assert len(plan['parts']) > 4 and sum(part['bytes'] for part in plan['parts']) == site.source.stat().st_size
    sha, begun, done = push(site.client, site.source)
    assert begun == {'held': False, 'missingParts': [part['index'] for part in plan['parts']]}
    assert done.status_code == 200 and done.json() == {'held': True, 'bytes': site.source.stat().st_size}
    entry = site.library / sha
    assert sorted(path.name for path in entry.iterdir()) == ['entry.json', 'jacobians.safetensors']
    assert (entry / 'jacobians.safetensors').read_bytes() == site.source.read_bytes()
    recorded = json.loads((entry / 'entry.json').read_bytes())
    assert set(recorded) == {'schemaVersion', 'convertedSHA256', 'bytes', 'origin', 'addedAt'}
    assert {k: recorded[k] for k in ('schemaVersion', 'convertedSHA256', 'bytes', 'origin')} == {
        'schemaVersion': 1, 'convertedSHA256': sha, 'bytes': site.source.stat().st_size, 'origin': 'pushed'}
    # The record names no machine, account, or folder.
    assert str(site.root) not in json.dumps(recorded) and os.sep not in json.dumps(recorded)
    assert not (site.library / '.incoming' / sha).exists()
    assert site.client.get(f'/api/lens-library/{sha}').json() == {
        'held': True, 'bytes': site.source.stat().st_size, 'receivedParts': None}
    # Held: a second sender is told so, uploads nothing, and may still complete.
    again = site.client.post(f'/api/lens-library/{sha}/begin', json=plan)
    assert again.json() == {'held': True, 'missingParts': []}
    data = site.source.read_bytes()
    assert send_part(site.client, sha, plan, data, 0).json() == {'received': True}
    assert site.client.post(f'/api/lens-library/{sha}/complete', json={}).json()['held'] is True
    assert not (site.library / '.incoming' / sha).exists()
    assert json.loads((entry / 'entry.json').read_bytes()) == recorded   # never edited


@pytest.mark.parametrize('damage, expected', [
    (lambda plan: plan['parts'][1].update(offset=plan['parts'][1]['offset'] + 1), 'no gap and no overlap'),
    (lambda plan: plan['parts'][1].update(offset=plan['parts'][1]['offset'] - 1), 'no gap and no overlap'),
    (lambda plan: plan['parts'].reverse(), 'numbered 0, 1, 2'),
    (lambda plan: plan['parts'].pop(), 'must cover it exactly'),
    (lambda plan: plan.update(bytes=plan['bytes'] + 1), 'must cover it exactly'),
    (lambda plan: plan['parts'][0].update(bytes=0), 'at least 1 byte'),
    (lambda plan: plan['parts'][0].update(sha256='not a hash'), 'SHA-256'),
    (lambda plan: plan['parts'][0].update(extra=1), 'exactly'),
    (lambda plan: plan.update(parts=[]), 'between 1 and'),
    (lambda plan: plan.update(note='x'), 'exactly `bytes` and `parts`'),
])
def test_parts_that_do_not_tile_the_file_are_refused_before_anything_is_kept(site, damage, expected):
    plan = part_plan(site.source, 64)
    damage(plan)
    response = site.client.post(f'/api/lens-library/{site.sha}/begin', json=plan)
    assert response.status_code == 400, response.text
    detail = refusal(response)
    assert detail['code'] == 'lensUploadPlanInvalid' and expected in detail['reason']
    assert not (site.library / '.incoming' / site.sha).exists()


def test_a_part_larger_than_the_site_allows_is_refused(site):
    plan = part_plan(site.source, 128)
    assert max(part['bytes'] for part in plan['parts']) > 64
    response = site.client.post(f'/api/lens-library/{site.sha}/begin', json=plan)
    assert response.status_code == 400
    assert 'at most 64 bytes' in refusal(response)['reason']


def test_a_corrupt_part_is_refused_and_nothing_of_it_is_kept(site):
    plan, data = part_plan(site.source, 64), site.source.read_bytes()
    site.client.post(f'/api/lens-library/{site.sha}/begin', json=plan)
    size = plan['parts'][2]['bytes']
    flipped = bytes([data[plan['parts'][2]['offset']] ^ 1]) + data[plan['parts'][2]['offset'] + 1:
                                                                  plan['parts'][2]['offset'] + size]
    for body, announced, reason in [
            (flipped, None, 'a SHA-256 that is not the one in the upload plan'),
            (flipped[:-1], None, f'arrived with {size - 1} bytes'),
            (flipped + b'x' * 200, None, f'arrived with more than {size} bytes'),
            (None, digest(flipped), 'announced (X-Content-SHA256)')]:
        response = send_part(site.client, site.sha, plan, data, 2, body=body, announced=announced)
        assert response.status_code == 409, response.text
        detail = refusal(response)
        assert detail['code'] == 'lensUploadPartCorrupt' and reason in detail['reason']
    assert site.client.put(f'/api/lens-library/{site.sha}/parts/2', content=b'x').status_code == 409   # no header
    assert site.client.get(f'/api/lens-library/{site.sha}').json()['receivedParts'] == []
    upload = site.library / '.incoming' / site.sha
    assert not list((upload / 'spool').iterdir())
    assert (upload / 'data').read_bytes() == bytes(len(data))      # no byte of the bad part reached the file
    for index in ('7' * 7, '-1', '01', 'x', str(len(plan['parts']))):
        assert site.client.put(f'/api/lens-library/{site.sha}/parts/{index}', content=b'x',
                               headers={'X-Content-SHA256': plan['parts'][0]['sha256']}).status_code == 400
    incomplete = site.client.post(f'/api/lens-library/{site.sha}/complete', json={})
    assert incomplete.status_code == 409 and refusal(incomplete)['code'] == 'lensUploadIncomplete'


def test_a_part_that_arrives_in_pieces_is_streamed_and_an_overlong_one_is_cut_off(site):
    import asyncio
    import httpx
    plan, data = part_plan(site.source, 64), site.source.read_bytes()
    assert site.client.post(f'/api/lens-library/{site.sha}/begin', json=plan).status_code == 200
    part = plan['parts'][1]
    body = data[part['offset']:part['offset'] + part['bytes']]
    pulled = []

    async def pieces(payload, size):
        for low in range(0, len(payload), size):
            pulled.append(low)
            yield payload[low:low + size]

    async def send(payload, size):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=site.app, client=('127.0.0.1', 50000)),
                                     base_url='http://testserver') as client:
            return await client.put(f'/api/lens-library/{site.sha}/parts/1', content=pieces(payload, size),
                                    headers={'X-Content-SHA256': part['sha256']})
    # Far longer than its plan: refused, and the rest of the body is never read into the upload.
    refused = asyncio.run(send(body + bytes(64 * 50), 16))
    assert refused.status_code == 409 and 'more than' in refusal(refused)['reason']
    assert len(pulled) < 50 and site.client.get(f'/api/lens-library/{site.sha}').json()['receivedParts'] == []
    assert not list((site.library / '.incoming' / site.sha / 'spool').iterdir())
    answered = asyncio.run(send(body, 7))
    assert answered.status_code == 200 and answered.json() == {'received': True}
    assert site.client.get(f'/api/lens-library/{site.sha}').json()['receivedParts'] == [1]
    held = (site.library / '.incoming' / site.sha / 'data').read_bytes()
    assert held[part['offset']:part['offset'] + part['bytes']] == body


def test_an_interrupted_upload_resumes_with_only_the_missing_parts(site):
    plan, data = part_plan(site.source, 64), site.source.read_bytes()
    assert site.client.post(f'/api/lens-library/{site.sha}/begin', json=plan).status_code == 200
    first = [0, 2, 3]
    for index in first:
        assert send_part(site.client, site.sha, plan, data, index).status_code == 200
    # The same part again is accepted and changes nothing.
    assert send_part(site.client, site.sha, plan, data, 2).json() == {'received': True}
    assert site.client.get(f'/api/lens-library/{site.sha}').json() == {
        'held': False, 'bytes': len(data), 'receivedParts': first}
    # A new connection: the same plan, and only what has not arrived is asked for.
    resumed = TestClient(site.app)
    missing = [part['index'] for part in plan['parts'] if part['index'] not in first]
    assert resumed.post(f'/api/lens-library/{site.sha}/begin', json=plan).json() == {
        'held': False, 'missingParts': missing}
    different = part_plan(site.source, 32)
    refused = resumed.post(f'/api/lens-library/{site.sha}/begin', json=different)
    assert refused.status_code == 409 and refusal(refused)['code'] == 'lensUploadPlanDiffers'
    for index in missing:
        assert send_part(resumed, site.sha, plan, data, index).status_code == 200
    done = resumed.post(f'/api/lens-library/{site.sha}/complete', json={})
    assert done.json() == {'held': True, 'bytes': len(data)}
    assert (site.library / site.sha / 'jacobians.safetensors').read_bytes() == data


def test_a_file_that_is_not_what_its_key_says_is_discarded(site, tmp_path):
    # Every part is what its plan says, but the whole is filed under another lens's hash.
    other = write_converted(tmp_path / 'other.safetensors', seed=9)
    plan, data = part_plan(site.source, 64), site.source.read_bytes()
    assert site.client.post(f'/api/lens-library/{other}/begin', json=plan).status_code == 200
    for part in plan['parts']:
        assert send_part(site.client, other, plan, data, part['index']).status_code == 200
    response = site.client.post(f'/api/lens-library/{other}/complete', json={})
    assert response.status_code == 409, response.text
    detail = refusal(response)
    assert detail['code'] == 'lensUploadMismatch' and 'the upload was discarded' in detail['reason']
    assert not (site.library / other).exists() and not (site.library / '.incoming' / other).exists()
    assert site.client.get(f'/api/lens-library/{other}').json() == {'held': False, 'bytes': None, 'receivedParts': None}
    # The right hash for bytes that are no safetensors file is discarded too.
    junk = tmp_path / 'junk.bin'
    junk.write_bytes(b'not a lens ' * 20)
    data, sha = junk.read_bytes(), digest(junk.read_bytes())
    plan = {'bytes': len(data), 'parts': [{'index': index, 'offset': low, 'bytes': len(data[low:low + 64]),
                                           'sha256': digest(data[low:low + 64])}
                                          for index, low in enumerate(range(0, len(data), 64))]}
    assert site.client.post(f'/api/lens-library/{sha}/begin', json=plan).status_code == 200
    for part in plan['parts']:
        assert send_part(site.client, sha, plan, data, part['index']).status_code == 200
    response = site.client.post(f'/api/lens-library/{sha}/complete', json={})
    assert response.status_code == 409 and refusal(response)['code'] == 'lensUploadNotSafetensors'
    assert not (site.library / sha).exists() and not (site.library / '.incoming' / sha).exists()
    # Completing what was never begun says so.
    unbegun = site.client.post(f'/api/lens-library/{site.sha}/complete', json={})
    assert unbegun.status_code == 409 and refusal(unbegun)['code'] == 'lensUploadNotBegun'
    assert site.client.post(f'/api/lens-library/{site.sha}/complete', json={'x': 1}).status_code == 400


def test_two_senders_of_one_lens_both_succeed_and_the_entry_is_whole(site):
    # In step: both begin, both send every part, both complete.
    left, right = TestClient(site.app), TestClient(site.app)
    plan, data = part_plan(site.source, 64), site.source.read_bytes()
    for client in (left, right):
        assert client.post(f'/api/lens-library/{site.sha}/begin', json=plan).json()['held'] is False
    for part in plan['parts']:
        for client in (left, right):
            assert send_part(client, site.sha, plan, data, part['index']).json() == {'received': True}
    for client in (left, right):
        assert client.post(f'/api/lens-library/{site.sha}/complete', json={}).json() == {'held': True, 'bytes': len(data)}
    assert (site.library / site.sha / 'jacobians.safetensors').read_bytes() == data

    # At once: each sender runs the whole exchange on its own thread, many times over.
    for seed in range(1, 7):
        source = site.root.parent / f'concurrent-{seed}.safetensors'
        sha = write_converted(source, seed=seed)
        barrier, outcomes = threading.Barrier(2), []

        def sender():
            try:
                barrier.wait(timeout=10)
                outcomes.append(push(TestClient(site.app), source)[2])
            except Exception as exc:  # noqa: BLE001 - reported below
                outcomes.append(exc)
        threads = [threading.Thread(target=sender) for _ in range(2)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(timeout=30)
        assert len(outcomes) == 2 and all(not isinstance(outcome, Exception) for outcome in outcomes), outcomes
        assert [outcome.status_code for outcome in outcomes] == [200, 200], [o.text for o in outcomes]
        assert digest((site.library / sha / 'jacobians.safetensors').read_bytes()) == sha
        assert json.loads((site.library / sha / 'entry.json').read_bytes())['bytes'] == source.stat().st_size
        assert not (site.library / '.incoming' / sha).exists()


def test_an_entry_is_published_where_the_filesystem_has_no_create_only_rename(site, monkeypatch):
    # Cluster storage (Lustre, NFS) answers EINVAL to the no-replace rename; publication
    # then claims the entry's name with mkdir, and still never replaces an entry.
    import errno
    from steerlab_server.experiment import diagnostic_archives as archives
    monkeypatch.setattr(archives, '_rename_noreplace', lambda source, target: errno.EINVAL)
    sha, _, done = push(site.client, site.source)
    assert done.status_code == 200 and done.json()['held'] is True
    entry = site.library / sha
    assert digest((entry / 'jacobians.safetensors').read_bytes()) == sha
    before = (entry / 'entry.json').read_bytes()
    # A second publisher of the same bytes finds the entry and leaves it as it is.
    staged = site.root / 'runs/again.safetensors'
    staged.parent.mkdir(parents=True)
    staged.write_bytes(site.source.read_bytes())
    with lens_library._staging(site.library.resolve()) as folder:
        (folder / 'jacobians.safetensors').write_bytes(site.source.read_bytes())
        assert lens_library._publish(site.library.resolve(), folder, sha, 'adopted')['bytes'] == staged.stat().st_size
    assert (entry / 'entry.json').read_bytes() == before
    # A name that is taken by something that is not an entry is refused, never replaced.
    other = 'e' * 64
    (site.library / other).mkdir()
    (site.library / other / 'unrelated').write_text('kept')
    with lens_library._staging(site.library.resolve()) as folder:
        (folder / 'jacobians.safetensors').write_bytes(site.source.read_bytes())
        with pytest.raises(lens_library.LensLibraryRefusal) as refused:
            lens_library._publish(site.library.resolve(), folder, other, 'adopted')
    assert refused.value.code == 'lensLibraryEntryBusy'
    assert sorted(path.name for path in (site.library / other).iterdir()) == ['unrelated']
    assert not list((site.library / '.incoming').iterdir())


def test_a_lens_that_cannot_fit_is_refused_before_it_is_begun(site, monkeypatch):
    monkeypatch.setattr(lens_library.shutil, 'disk_usage', lambda path: SimpleNamespace(free=100))
    response = site.client.post(f'/api/lens-library/{site.sha}/begin', json=part_plan(site.source, 64))
    assert response.status_code == 409, response.text
    detail = refusal(response)
    assert detail['code'] == 'lensLibraryNoSpace' and '100 bytes free' in detail['reason']
    assert not (site.library / '.incoming' / site.sha).exists()


def test_a_site_that_moves_files_itself_refuses_an_upload_before_it_begins(site, monkeypatch):
    plan = part_plan(site.source, 64)
    monkeypatch.setenv('STEERLAB_TRANSFER_METHOD', 'globus')
    begun = site.client.post(f'/api/lens-library/{site.sha}/begin', json=plan)
    assert begun.status_code == 403, begun.text
    detail = refusal(begun)
    assert detail['code'] == 'external_transfer_required' and 'a lens cannot be sent to it in parts' in detail['reason']
    # The repair is about a lens (the policy's own speaks of bundles), and names the verb that takes it.
    assert detail['repairAction'] == lens_library.ADOPT_REPAIR and 'lens-push <lensID> --adopt' in detail['repairAction']
    assert not site.library.exists()                      # nothing was created, so nothing is left behind
    # A part obeys the same policy, with the same answer, even for an upload begun before the policy was set.
    monkeypatch.delenv('STEERLAB_TRANSFER_METHOD')
    assert site.client.post(f'/api/lens-library/{site.sha}/begin', json=plan).status_code == 200
    monkeypatch.setenv('STEERLAB_TRANSFER_METHOD', 'globus')
    response = send_part(site.client, site.sha, plan, site.source.read_bytes(), 0)
    assert response.status_code == 403 and refusal(response) == detail
    assert site.client.get(f'/api/lens-library/{site.sha}').json()['receivedParts'] == []


# --- bytes already on the site ---------------------------------------------------

def test_a_file_already_on_the_site_is_adopted_after_it_is_hashed(site):
    staged = site.root / 'runs/transfers/lens.safetensors'
    staged.parent.mkdir(parents=True)
    staged.write_bytes(site.source.read_bytes())
    wrong = site.client.post('/api/lens-library/adopt', json={'path': 'transfers/lens.safetensors', 'sha256': 'e' * 64})
    assert wrong.status_code == 409 and refusal(wrong)['code'] == 'lensAdoptMismatch'
    assert not (site.library / ('e' * 64)).exists()
    for path in ('transfers/lens.safetensors', str(staged)):     # relative to the run root, or absolute beneath it
        adopted = site.client.post('/api/lens-library/adopt', json={'path': path, 'sha256': site.sha})
        assert adopted.status_code == 200 and adopted.json() == {'held': True, 'bytes': staged.stat().st_size}
    entry = json.loads((site.library / site.sha / 'entry.json').read_bytes())
    assert entry['origin'] == 'adopted' and str(site.root) not in json.dumps(entry)
    assert staged.read_bytes() == site.source.read_bytes()          # the source is left alone
    assert sorted(path.name for path in (site.library / '.incoming').iterdir()) == []
    junk = site.root / 'runs/transfers/junk.safetensors'
    junk.write_bytes(b'not a lens')
    refused = site.client.post('/api/lens-library/adopt', json={'path': str(junk), 'sha256': digest(b'not a lens')})
    assert refused.status_code == 409 and refusal(refused)['code'] == 'lensAdoptNotSafetensors'
    # A copy, not a link: a later transfer that writes over the source in place cannot change the entry.
    with open(staged, 'r+b') as stream:
        stream.write(b'overwritten in place')
    assert digest((site.library / site.sha / 'jacobians.safetensors').read_bytes()) == site.sha


# --- nothing reaches outside the library -----------------------------------------

@pytest.mark.parametrize('key', ['A' * 64, 'a' * 63, 'a' * 65, 'g' * 64, 'register', '.incoming', '..', '%2e%2e',
                                 'a' * 63 + '%0a', 'a' * 32 + '%2f' + 'a' * 31])
def test_a_key_that_is_not_a_lowercase_sha256_is_refused_on_every_route(site, key):
    responses = [site.client.get(f'/api/lens-library/{key}'),
                 site.client.post(f'/api/lens-library/{key}/begin', json=part_plan(site.source, 64)),
                 site.client.put(f'/api/lens-library/{key}/parts/0', content=b'x', headers={'X-Content-SHA256': 'a' * 64}),
                 site.client.post(f'/api/lens-library/{key}/complete', json={}),
                 site.client.post('/api/lens-library/adopt', json={'path': 'x', 'sha256': key})]
    for response in responses:
        # 400 with the library's own refusal, or no such route at all; never an answer.
        assert response.status_code in (400, 404, 405), (key, response.status_code, response.text)
        if response.status_code == 400:
            assert refusal(response)['code'] == 'lensLibraryKeyInvalid'
    assert not site.library.exists() or sorted(path.name for path in site.library.iterdir()) == []


def test_no_route_follows_a_symbolic_link_or_leaves_the_run_root(site, tmp_path):
    # A real entry, reached through a link filed under another key: never read.
    sha = push(site.client, site.source)[0]
    alias = 'f' * 64
    (site.library / alias).symlink_to(site.library / sha)
    linked = site.client.get(f'/api/lens-library/{alias}')
    assert linked.status_code == 409 and refusal(linked)['code'] == 'lensLibraryNotOrdinary'
    assert lens_library.held_file(alias) is None
    with pytest.raises(lens_library.LensLibraryRefusal):
        lens_library.require_held([{'lensID': 'x', 'convertedSHA256': alias, 'bytes': 1}])
    # An upload folder that is a link out of the library: nothing is written through it.
    other, outside = write_converted(tmp_path / 'second.safetensors', seed=4), tmp_path / 'outside'
    outside.mkdir()
    (site.library / '.incoming' / other).symlink_to(outside)
    begun = site.client.post(f'/api/lens-library/{other}/begin', json=part_plan(tmp_path / 'second.safetensors', 64))
    assert begun.status_code == 409 and refusal(begun)['code'] == 'lensLibraryNotOrdinary'
    assert list(outside.iterdir()) == []
    # Adoption: beneath the run root, an ordinary file, reached without a link.
    secret = tmp_path / 'elsewhere.safetensors'
    secret.write_bytes(site.source.read_bytes())
    (site.root / 'runs').mkdir(exist_ok=True)
    (site.root / 'runs/link.safetensors').symlink_to(secret)
    (site.root / 'runs/folder').symlink_to(tmp_path)
    for path in (str(secret), '../elsewhere.safetensors', 'link.safetensors', 'folder/elsewhere.safetensors',
                 'missing.safetensors', '', 'runs'):
        refused = site.client.post('/api/lens-library/adopt', json={'path': path, 'sha256': 'b' * 64})
        assert refused.status_code == 400, (path, refused.text)
        assert refusal(refused)['code'] == 'lensAdoptPathInvalid'
    assert not (site.library / ('b' * 64)).exists()


# --- a lens record on the site ---------------------------------------------------

def test_a_record_is_registered_once_and_only_for_bytes_the_library_holds(site):
    record = record_for(site.sha)
    absent = site.client.post('/api/lens-library/register', json={'record': record})
    assert absent.status_code == 409, absent.text
    detail = refusal(absent)
    assert detail['code'] == 'lensNotInLibrary' and LENS_ID in detail['reason'] and site.sha in detail['reason']
    assert 'lens-push' in detail['repairAction'] and 'stage the request again' not in detail['repairAction']
    assert not (site.root / 'runs').exists()
    push(site.client, site.source)
    first = site.client.post('/api/lens-library/register', json={'record': record})
    assert first.status_code == 200 and first.json() == {'lensID': LENS_ID, 'registered': True, 'changed': True}
    stored = site.root / 'runs/jlens-lenses' / LENS_ID
    assert sorted(path.name for path in stored.iterdir()) == ['lens.json']       # the record, and no bytes
    assert lens_store.resolve(LENS_ID, str(site.root)).to_dict() == record
    again = site.client.post('/api/lens-library/register', json={'record': record})
    assert again.json() == {'lensID': LENS_ID, 'registered': True, 'changed': False}
    assert lens_store.resolve(LENS_ID, str(site.root)).to_dict() == record


def test_another_machines_folder_never_reaches_the_site(site):
    push(site.client, site.source)
    foreign = record_for(site.sha, path='/elsewhere/workspace/' + CANONICAL)
    assert site.client.post('/api/lens-library/register', json={'record': foreign}).json()['changed'] is True
    stored = (site.root / 'runs/jlens-lenses' / LENS_ID / 'lens.json').read_text()
    assert '/elsewhere' not in stored
    assert lens_store.resolve(LENS_ID, str(site.root)).converted.path == CANONICAL
    for path in ('/elsewhere/workspace/other.safetensors', 'somewhere/jacobians.safetensors',
                 '/elsewhere/runs/jlens-lenses/another-lens/jacobians.safetensors', '../' + CANONICAL):
        other = record_for(site.sha, path=path, lens_id='custom-lens-second')
        refused = site.client.post('/api/lens-library/register', json={'record': other})
        assert refused.status_code == 400 and refusal(refused)['code'] == 'lensRecordInvalid', (path, refused.text)
    assert not (site.root / 'runs/jlens-lenses/custom-lens-second').exists()
    for name in ('../escape', 'a/b', '.hidden', '', None, 7):
        refused = site.client.post('/api/lens-library/register', json={'record': record_for(site.sha, lens_id=name)})
        assert refused.status_code == 400, (name, refused.text)
    assert site.client.post('/api/lens-library/register', json={'record': 'lens.json'}).status_code == 400
    assert site.client.post('/api/lens-library/register', json={'record': record_for(site.sha), 'x': 1}).status_code == 400
    assert sorted(path.name for path in (site.root / 'runs/jlens-lenses').iterdir()) == [LENS_ID]
    # A lens folder on the site that is a link out of the workspace is never written through.
    outside = site.root.parent / 'outside'
    outside.mkdir()
    (site.root / 'runs/jlens-lenses/custom-lens-linked').symlink_to(outside)
    linked = site.client.post('/api/lens-library/register',
                              json={'record': record_for(site.sha, lens_id='custom-lens-linked',
                                                         path='runs/jlens-lenses/custom-lens-linked/jacobians.safetensors')})
    assert linked.status_code == 409 and refusal(linked)['code'] == 'lensRecordNotOrdinary'
    assert list(outside.iterdir()) == []
    # The same qualification twice in one record is one qualification.
    twice = [qualification(site.sha, 'q-1'), qualification(site.sha, 'q-1')]
    assert site.client.post('/api/lens-library/register',
                            json={'record': record_for(site.sha, qualifications=twice)}).json()['changed'] is True
    assert [q.qualificationID for q in lens_store.resolve(LENS_ID, str(site.root)).qualifications] == ['q-1']


def test_a_registered_record_gains_qualifications_and_never_loses_one(site):
    push(site.client, site.source)
    first = qualification(site.sha, 'q-1')
    site.client.post('/api/lens-library/register', json={'record': record_for(site.sha, qualifications=[first])})
    second = qualification(site.sha, 'q-2', dtype='bfloat16')
    added = site.client.post('/api/lens-library/register',
                             json={'record': record_for(site.sha, qualifications=[first, second])})
    assert added.json() == {'lensID': LENS_ID, 'registered': True, 'changed': True}
    assert [q.qualificationID for q in lens_store.resolve(LENS_ID, str(site.root)).qualifications] == ['q-1', 'q-2']
    # A record that knows fewer qualifications takes none away.
    fewer = site.client.post('/api/lens-library/register', json={'record': record_for(site.sha)})
    assert fewer.json()['changed'] is False
    before = (site.root / 'runs/jlens-lenses' / LENS_ID / 'lens.json').read_bytes()
    conflicting = qualification(site.sha, 'q-1', passed=False)
    third = qualification(site.sha, 'q-3')
    refused = site.client.post('/api/lens-library/register',
                               json={'record': record_for(site.sha, qualifications=[third, conflicting])})
    assert refused.status_code == 409, refused.text
    detail = refusal(refused)
    assert detail['code'] == 'lensQualificationDiffers' and 'q-1' in detail['reason']
    # Refused whole: the new qualification beside the conflicting one was not added either.
    assert (site.root / 'runs/jlens-lenses' / LENS_ID / 'lens.json').read_bytes() == before


def test_one_name_stands_for_one_set_of_bytes(site, tmp_path):
    push(site.client, site.source)
    site.client.post('/api/lens-library/register', json={'record': record_for(site.sha)})
    before = (site.root / 'runs/jlens-lenses' / LENS_ID / 'lens.json').read_bytes()
    write_converted(tmp_path / 'other.safetensors', seed=5)
    other = push(site.client, tmp_path / 'other.safetensors')[0]
    for record, named in [(record_for(other), (site.sha, other)),
                          (record_for(site.sha, tensor='9' * 64), ('c' * 64, '9' * 64))]:
        refused = site.client.post('/api/lens-library/register', json={'record': record})
        assert refused.status_code == 409, refused.text
        detail = refusal(refused)
        assert detail['code'] == 'lensRecordDiffers' and all(value in detail['reason'] for value in named)
    assert (site.root / 'runs/jlens-lenses' / LENS_ID / 'lens.json').read_bytes() == before


# --- resolving a lens: the workspace first, then the library ----------------------

def test_a_workspace_file_always_wins_and_the_library_is_never_hashed(site, monkeypatch):
    push(site.client, site.source)
    site.client.post('/api/lens-library/register', json={'record': record_for(site.sha)})
    monkeypatch.setattr(lens_library, '_hash', lambda path: pytest.fail('resolving a lens must not hash it'))
    library_file = str((site.library / site.sha / 'jacobians.safetensors').resolve())
    canonical = site.root.resolve() / CANONICAL
    for reference in (CANONICAL, '/elsewhere/workspace/' + CANONICAL):
        assert artifact_paths.converted_file(LENS_ID, reference, str(site.root)) == library_file
        # Asked for the workspace only (a request that carries its lens), the library is not consulted.
        assert artifact_paths.converted_file(LENS_ID, reference, str(site.root), library=False) != library_file
    canonical.write_bytes(b'the workspace copy')
    for reference in (CANONICAL, '/elsewhere/workspace/' + CANONICAL):
        assert artifact_paths.converted_file(LENS_ID, reference, str(site.root)) == str(canonical)
    canonical.unlink()
    # A lens the library does not hold resolves as it always did: to the workspace path, absent.
    assert artifact_paths.converted_file('custom-lens-unknown', 'runs/jlens-lenses/custom-lens-unknown/jacobians.safetensors',
                                         str(site.root)) == str(site.root.resolve() / 'runs/jlens-lenses/custom-lens-unknown/jacobians.safetensors')
    # Losing the library loses no record: the lens is still listed, and says its bytes are missing.
    record = lens_store.resolve(LENS_ID, str(site.root))
    monkeypatch.setenv('STEERLAB_EXECUTOR', 'slurm')
    assert artifact_paths.converted_file(LENS_ID, CANONICAL, str(site.root)) == str(canonical)
    with pytest.raises(schemas.JLensError, match='missing'):
        lens_store.load_layer(record, 0, root=str(site.root))
    assert [r.lensID for r in lens_store.list_lenses(str(site.root))] == [LENS_ID]


def test_only_a_record_in_the_importers_layout_is_answered_by_the_library(site, tmp_path):
    """One rule decides whether a record's path may be answered by the library, and the reader obeys it:
    any other path resolves exactly as it did before a library existed, whether or not it exists."""
    push(site.client, site.source)
    site.client.post('/api/lens-library/register', json={'record': record_for(site.sha)})
    library_file = str((site.library / site.sha / 'jacobians.safetensors').resolve())
    outside = tmp_path / 'a-copy-somewhere.safetensors'
    outside.write_bytes(site.source.read_bytes())
    root = str(site.root.resolve())
    for reference in (CANONICAL, '/elsewhere/workspace/' + CANONICAL, 'nested/' + CANONICAL):
        assert artifact_paths.library_eligible(LENS_ID, reference)
        assert artifact_paths.converted_file(LENS_ID, reference, str(site.root)) == library_file
    for reference in (str(outside),                                   # exists, outside the workspace
                      '/elsewhere/absent.safetensors',                 # absolute, and nowhere
                      'somewhere/else.safetensors',                    # relative, and nowhere
                      'runs/jlens-lenses/another-lens/jacobians.safetensors',
                      CANONICAL + '.bak', 'runs/jlens-lenses/' + LENS_ID):
        assert not artifact_paths.library_eligible(LENS_ID, reference)
        expected = reference if os.path.isabs(reference) else os.path.join(root, reference)
        assert artifact_paths.converted_file(LENS_ID, reference, str(site.root)) == expected != library_file
    assert not artifact_paths.library_eligible(LENS_ID, None)


# --- fix round 1: where a record is registered ------------------------------------

def test_a_record_is_registered_where_the_engines_lens_routes_and_runs_read(site, monkeypatch, tmp_path):
    from steerlab_server.api.app import app
    runs = tmp_path / 'scratch-runs'                       # a declared run root that is not <root>/runs
    monkeypatch.setenv('STEERLAB_RUN_ROOT', str(runs))
    push(site.client, site.source)
    answered = site.client.post('/api/lens-library/register', json={'record': record_for(site.sha)})
    assert answered.status_code == 200 and answered.json() == {'lensID': LENS_ID, 'registered': True, 'changed': True}
    assert (runs / 'jlens-lenses' / LENS_ID / 'lens.json').is_file() and not (site.root / 'runs').exists()
    # The engine's own lens routes, and anything that resolves a lens with no root (a study run), find it.
    engine = TestClient(app)
    assert [lens['lensID'] for lens in engine.get('/api/jlens/lenses').json()['lenses']] == [LENS_ID]
    detail = engine.get(f'/api/jlens/lenses/{LENS_ID}')
    assert detail.status_code == 200 and detail.json()['converted'] == {
        'path': CANONICAL, 'dtype': 'float32', 'sha256': site.sha, 'layerCount': 3}
    layer = lens_store.load_layer(lens_store.resolve(LENS_ID), 1)
    assert tuple(layer.shape) == (4, 4)                    # read from the library, through the record found there
    # The same store is where qualifications are added, and where a name already taken is seen.
    added = site.client.post('/api/lens-library/register',
                             json={'record': record_for(site.sha, qualifications=[qualification(site.sha)])})
    assert added.json()['changed'] is True
    assert [q.qualificationID for q in lens_store.resolve(LENS_ID).qualifications] == ['q-1']
    taken = schemas.JLensRecord.from_dict(record_for('e' * 64, lens_id='custom-lens-taken'))
    lens_store.save(taken)                                 # a different lens of that name, imported on the site
    refused = site.client.post('/api/lens-library/register',
                               json={'record': record_for(site.sha, lens_id='custom-lens-taken',
                                                          path='runs/jlens-lenses/custom-lens-taken/jacobians.safetensors')})
    assert refused.status_code == 409, refused.text
    detail = refusal(refused)
    assert detail['code'] == 'lensRecordDiffers' and 'e' * 64 in detail['reason'] and site.sha in detail['reason']
    # The repair no longer sends the reader to import again for a new name.
    assert 'import' not in detail['repairAction'] and '--lens-reference' in detail['repairAction']
    assert lens_store.resolve('custom-lens-taken').converted.sha256 == 'e' * 64


# --- fix round 1: a folder under a hash that is not an entry ----------------------

def idle(path, seconds):
    """Make a folder, and everything in it, look unchanged for this long."""
    stamp = time.time() - seconds
    for directory, folders, files in os.walk(path):
        for name in folders + files:
            os.utime(os.path.join(directory, name), (stamp, stamp))
    os.utime(path, (stamp, stamp))


@pytest.mark.parametrize('claim', [False, True], ids=['create-only rename', 'mkdir claim'])
@pytest.mark.parametrize('left', ['record only', 'nothing'])
def test_a_purged_entry_is_replaced_by_sending_the_lens_again(site, monkeypatch, claim, left):
    from steerlab_server.experiment import diagnostic_archives as archives
    if claim:                                              # cluster storage: no no-replace rename
        monkeypatch.setattr(archives, '_rename_noreplace', lambda source, target: errno.EINVAL)
    sha = push(site.client, site.source)[0]
    entry = site.library / sha
    (entry / 'jacobians.safetensors').unlink()             # storage purged the large, rarely read file
    if left == 'nothing':
        (entry / 'entry.json').unlink()                    # or a publisher died having only claimed the name
    idle(entry, lens_library.STALE_ENTRY_SECONDS + 1)
    plan = part_plan(site.source, 64)
    assert site.client.get(f'/api/lens-library/{sha}').json()['held'] is False
    _, begun, done = push(site.client, site.source)
    assert begun == {'held': False, 'missingParts': [part['index'] for part in plan['parts']]}
    assert done.status_code == 200 and done.json() == {'held': True, 'bytes': site.source.stat().st_size}
    assert digest((entry / 'jacobians.safetensors').read_bytes()) == sha
    # Nobody removed anything: the folder was set aside inside the library, by itself.
    (aside,) = [path.name for path in (site.library / '.incoming').iterdir()]
    assert aside.startswith(f'damaged-{sha}-')
    # The same for a file already on the site.
    (entry / 'jacobians.safetensors').unlink()
    idle(entry, lens_library.STALE_ENTRY_SECONDS + 1)
    staged = site.root / 'runs/lens.safetensors'
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(site.source.read_bytes())
    adopted = site.client.post('/api/lens-library/adopt', json={'path': 'lens.safetensors', 'sha256': sha})
    assert adopted.status_code == 200 and digest((entry / 'jacobians.safetensors').read_bytes()) == sha


def test_an_unfinished_folder_that_is_still_changing_is_refused_before_a_byte_is_sent(site):
    folder = site.library / site.sha
    folder.mkdir(parents=True)                             # a publisher has just claimed the name
    plan = part_plan(site.source, 64)
    staged = site.root / 'runs/lens.safetensors'
    staged.parent.mkdir(parents=True)
    staged.write_bytes(site.source.read_bytes())
    for response in (site.client.post(f'/api/lens-library/{site.sha}/begin', json=plan),
                     site.client.post('/api/lens-library/adopt', json={'path': 'lens.safetensors', 'sha256': site.sha})):
        assert response.status_code == 409, response.text
        detail = refusal(response)
        assert detail['code'] == 'lensLibraryEntryBusy' and 'being published right now' in detail['reason']
        assert detail['repairAction'].startswith('Try again in a few minutes.')
    # Nothing was begun or copied, and the folder was left to whoever is filling it.
    assert not (site.library / '.incoming').exists() and list(folder.iterdir()) == []
    # A little short of idle is still busy; idle, it makes way.
    idle(folder, lens_library.STALE_ENTRY_SECONDS - 5)
    assert site.client.post(f'/api/lens-library/{site.sha}/begin', json=plan).status_code == 409
    idle(folder, lens_library.STALE_ENTRY_SECONDS + 1)
    assert site.client.post(f'/api/lens-library/{site.sha}/begin', json=plan).status_code == 200
    assert lens_library.STALE_ENTRY_SECONDS == 15 * 60


# --- fix round 1: a run root reached through a link -------------------------------

def test_adopt_takes_the_sites_own_spelling_of_its_run_root(site, monkeypatch, tmp_path):
    real = tmp_path / 'scratch'
    (real / 'runs/transfers').mkdir(parents=True)
    link = tmp_path / 'scratch-link'
    link.symlink_to(real)                                  # cluster scratch is often reached through one
    monkeypatch.setenv('STEERLAB_RUN_ROOT', str(link / 'runs'))
    (real / 'runs/transfers/lens.safetensors').write_bytes(site.source.read_bytes())
    for path in (str(link / 'runs/transfers/lens.safetensors'),      # as the site spells it
                 str(real / 'runs/transfers/lens.safetensors'),      # resolved
                 'transfers/lens.safetensors'):                      # relative to the run root
        adopted = site.client.post('/api/lens-library/adopt', json={'path': path, 'sha256': site.sha})
        assert adopted.status_code == 200, (path, adopted.text)
    # Below the root a link is still refused, under either spelling.
    (real / 'runs/alias').symlink_to(real / 'runs/transfers')
    for path in ('alias/lens.safetensors', str(link / 'runs/alias/lens.safetensors'),
                 str(real / 'runs/alias/lens.safetensors'), str(link / 'elsewhere.safetensors')):
        refused = site.client.post('/api/lens-library/adopt', json={'path': path, 'sha256': site.sha})
        assert refused.status_code == 400 and refusal(refused)['code'] == 'lensAdoptPathInvalid', (path, refused.text)
        assert refusal(refused)['repairAction'] == lens_library.ADOPT_REPAIR


# --- fix round 1: nothing large is left for good ----------------------------------

def test_what_a_killed_process_left_is_swept_and_an_idle_upload_expires(site):
    incoming = site.library / '.incoming'
    day, week = lens_library.STALE_STAGING_SECONDS, lens_library.STALE_UPLOAD_SECONDS
    assert (day, week) == (24 * 3600, 7 * 24 * 3600)
    ages = {'publish-killed': day + 60, 'publish-working': 60, 'damaged-' + 'c' * 64 + '-0000': day + 60,
            'begin-killed': day + 60, 'a' * 64: week + 60, 'b' * 64: week - 3600, 'someone-elses-notes': 10 * week}
    for name, seconds in ages.items():
        (incoming / name).mkdir(parents=True)
        (incoming / name / 'jacobians.safetensors').write_bytes(b'x' * 1000)
        (incoming / name / 'manifest.json').write_text('{}')
        idle(incoming / name, seconds)
    push(site.client, site.source)                         # a staging folder is made: the sweep runs
    assert sorted(path.name for path in incoming.iterdir()) == sorted(['publish-working', 'b' * 64, 'someone-elses-notes'])


def test_a_different_plan_replaces_an_upload_that_has_gone_idle(site):
    first, second = part_plan(site.source, 64), part_plan(site.source, 32)
    data = site.source.read_bytes()
    assert site.client.post(f'/api/lens-library/{site.sha}/begin', json=first).status_code == 200
    assert send_part(site.client, site.sha, first, data, 0).status_code == 200
    refused = site.client.post(f'/api/lens-library/{site.sha}/begin', json=second)
    assert refused.status_code == 409, refused.text
    detail = refusal(refused)
    assert detail['code'] == 'lensUploadPlanDiffers'
    assert 'same client and the same part size' in detail['repairAction'] and 'idle for 15 minutes' in detail['repairAction']
    assert 'operator' not in detail['repairAction']
    idle(site.library / '.incoming' / site.sha, lens_library.IDLE_UPLOAD_SECONDS + 1)
    replaced = site.client.post(f'/api/lens-library/{site.sha}/begin', json=second)
    assert replaced.status_code == 200 and replaced.json() == {
        'held': False, 'missingParts': [part['index'] for part in second['parts']]}
    assert site.client.get(f'/api/lens-library/{site.sha}').json()['receivedParts'] == []
    for part in second['parts']:
        assert send_part(site.client, site.sha, second, data, part['index']).status_code == 200
    assert site.client.post(f'/api/lens-library/{site.sha}/complete', json={}).json()['held'] is True


def test_an_upload_folder_with_no_plan_or_no_working_files_starts_clean(site, monkeypatch):
    plan, data = part_plan(site.source, 64), site.source.read_bytes()
    everything = [part['index'] for part in plan['parts']]
    upload = site.library / '.incoming' / site.sha
    # Debris with no plan: nothing in it was ever checked, whatever it claims to have received.
    (upload / 'received').mkdir(parents=True)
    (upload / 'received/0').write_text('')
    (upload / 'data').write_bytes(b'left behind')
    assert site.client.post(f'/api/lens-library/{site.sha}/begin', json=plan).json() == {'held': False, 'missingParts': everything}
    assert (upload / 'data').stat().st_size == len(data) and list((upload / 'received').iterdir()) == []
    # A removal cut short takes the plan first, so what is left reads as no upload, and begins clean.
    assert send_part(site.client, site.sha, plan, data, 0).status_code == 200
    with monkeypatch.context() as patched:
        patched.setattr(lens_library.shutil, 'rmtree', lambda *args, **kwargs: None)      # stopped after its first step
        lens_library._discard(upload)
    assert not (upload / 'manifest.json').exists() and (upload / 'received/0').exists()
    assert site.client.get(f'/api/lens-library/{site.sha}').json() == {'held': False, 'bytes': None, 'receivedParts': None}
    assert site.client.post(f'/api/lens-library/{site.sha}/begin', json=plan).json()['missingParts'] == everything
    # A plan whose working files are gone (a removal cut short the old way) cannot be resumed, and is not.
    assert send_part(site.client, site.sha, plan, data, 0).status_code == 200
    (upload / 'data').unlink()
    assert site.client.post(f'/api/lens-library/{site.sha}/begin', json=plan).json()['missingParts'] == everything
    _, _, done = push(site.client, site.source)
    assert done.status_code == 200 and done.json()['held'] is True


@pytest.mark.filterwarnings('ignore:This process.*is multi-threaded:DeprecationWarning')
def test_two_engine_processes_on_one_library_never_show_an_error_number_or_a_path(site, tmp_path):
    """Two engines on one library folder share no lock. Each child here is one engine process sending the same
    lens over the routes; whichever loses a race is answered plainly, and the entry is whole."""
    plain = {'lensUploadIncomplete', 'lensUploadNotBegun', 'lensUploadBusy', 'lensLibraryEntryBusy'}
    for attempt in range(8):
        source = tmp_path / f'forked-{attempt}.safetensors'
        sha = write_converted(source, seed=100 + attempt)
        data, plan = source.read_bytes(), part_plan(source, 64)
        ready, go = os.pipe()
        children = []
        for number in range(2):
            pid = os.fork()
            if pid == 0:
                status = 1
                try:
                    os.read(ready, 1)                      # both start together
                    client, seen = TestClient(site.app), []
                    begun = client.post(f'/api/lens-library/{sha}/begin', json=plan)
                    seen.append(('begin', begun.status_code, begun.text))
                    for index in (begun.json().get('missingParts', []) if begun.status_code == 200 else []):
                        sent = send_part(client, sha, plan, data, index)
                        seen.append((f'part {index}', sent.status_code, sent.text))
                    done = client.post(f'/api/lens-library/{sha}/complete', json={})
                    seen.append(('complete', done.status_code, done.text))
                    (tmp_path / f'outcome-{attempt}-{number}.json').write_text(json.dumps(seen))
                    status = 0
                finally:
                    os._exit(status)
            children.append(pid)
        os.write(go, b'gg')
        deadline = time.monotonic() + 60
        for pid in children:
            finished, status = os.waitpid(pid, os.WNOHANG)
            while not finished and time.monotonic() < deadline:
                time.sleep(0.01)
                finished, status = os.waitpid(pid, os.WNOHANG)
            if not finished:
                os.kill(pid, 9)
                os.waitpid(pid, 0)
            assert finished and status == 0, 'an engine process did not finish its exchange'
        os.close(ready), os.close(go)
        completed = 0
        for number in range(2):
            seen = json.loads((tmp_path / f'outcome-{attempt}-{number}.json').read_text())
            for step, status, text in seen:
                assert 'Errno' not in text and str(tmp_path) not in text and str(site.root) not in text, (step, text)
                if status != 200:
                    assert status == 409 and json.loads(text)['detail']['code'] in plain, (step, status, text)
            completed += seen[-1][1] == 200
        assert completed >= 1
        entry = site.library / sha
        assert digest((entry / 'jacobians.safetensors').read_bytes()) == sha
        assert json.loads((entry / 'entry.json').read_bytes())['bytes'] == len(data)
        assert not (site.library / '.incoming' / sha).exists()
    assert not list((site.library / '.incoming').iterdir())


# --- fix round 1: the file that is published is the file that was hashed -----------

def test_storage_that_cannot_link_publishes_a_copy_and_the_copy_is_what_was_hashed(site, monkeypatch, tmp_path):
    def cannot_link(source, target, **kwargs):
        raise OSError(errno.EPERM, 'Operation not permitted')
    monkeypatch.setattr(lens_library.os, 'link', cannot_link)
    sha, _, done = push(site.client, site.source)
    assert done.status_code == 200 and (site.library / sha / 'jacobians.safetensors').read_bytes() == site.source.read_bytes()
    # A copy that came out different is caught: the upload was right, the published file would not have been.
    other = tmp_path / 'other.safetensors'
    other_sha = write_converted(other, seed=7)
    copy = lens_library.shutil.copyfile

    def damaged(source, target, **kwargs):
        copy(source, target, **kwargs)
        with open(target, 'r+b') as stream:
            stream.seek(-1, os.SEEK_END)
            stream.write(b'\xff')
    monkeypatch.setattr(lens_library.shutil, 'copyfile', damaged)
    _, _, refused = push(site.client, other)
    assert refused.status_code == 409 and refusal(refused)['code'] == 'lensUploadMismatch'
    assert not (site.library / other_sha).exists()


def test_an_entry_can_be_read_by_whoever_the_umask_allows(site):
    previous = os.umask(0o022)
    try:
        sha = push(site.client, site.source)[0]
        staged = site.root / 'runs/second.safetensors'
        second = write_converted(staged, seed=3)
        assert site.client.post('/api/lens-library/adopt', json={'path': 'second.safetensors', 'sha256': second}).status_code == 200
    finally:
        os.umask(previous)
    for key in (sha, second):
        entry = site.library / key
        assert stat.S_IMODE(entry.stat().st_mode) == 0o755
        assert {stat.S_IMODE(path.stat().st_mode) for path in entry.iterdir()} == {0o644}


# --- fix round 1: what a record may hold, and how records are compared ------------

def test_a_record_says_which_part_of_it_cannot_be_used(site):
    push(site.client, site.source)
    good = record_for(site.sha)

    def without(*path):
        record = json.loads(json.dumps(good))
        block = record
        for name in path[:-1]:
            block = block[name]
        del block[path[-1]]
        return record
    cases = [(without('source'), '`source` block'), (without('converted'), '`converted` block'),
             (without('dModel'), '`sourceLayers`, `dModel`, `targetLayer`, and `nPrompts`'),
             (without('source', 'tensorSHA256'), '`source.tensorSHA256`'),
             ({**good, 'converted': {**good['converted'], 'sha256': 'short'}}, '`converted.sha256`'),
             ({**good, 'converted': {**good['converted'], 'path': 'elsewhere/jacobians.safetensors'}}, '`converted.path`'),
             ({**good, 'qualifications': [{'qualificationID': 'q'}]}, 'qualification 0'),
             ({**good, 'lensID': 'a/b'}, '`lensID`')]
    for record, named in cases:
        response = site.client.post('/api/lens-library/register', json={'record': record})
        assert response.status_code == 400, (named, response.text)
        detail = refusal(response)
        assert detail['code'] == 'lensRecordInvalid' and named in detail['reason'], detail
        # `lens-push` sent this record, so the repair is to make the record whole where it was made.
        assert detail['repairAction'].startswith('Import the lens again on your machine')
    assert not (site.root / 'runs').exists()


def test_a_number_that_is_not_finite_is_refused_when_sent_and_reads_as_null_when_stored(site):
    from steerlab_server.jlens import qualification as qualifications
    push(site.client, site.source)
    endpoint, headers = '/api/lens-library/register', {'content-type': 'application/json'}
    unfinished = qualification(site.sha, 'q-nan', checks={'score': float('nan'), 'margin': float('inf')})
    sent = site.client.post(endpoint, headers=headers,
                            content=json.dumps({'record': record_for(site.sha, qualifications=[unfinished])}))
    assert sent.status_code == 400, sent.text
    detail = refusal(sent)
    assert detail['code'] == 'lensRecordInvalid' and 'not finite' in detail['reason']
    assert 'qualifications[0].checks.score' in detail['reason']
    assert 'Update the client' in detail['repairAction'] and 'null' in detail['repairAction']
    assert not (site.root / 'runs').exists()
    # The site's own record holds the NaN, as the engine's qualification writes one.
    site.client.post(endpoint, json={'record': record_for(site.sha)})
    qualifications.append_qualification(lens_store.resolve(LENS_ID), unfinished)
    assert 'NaN' in (site.root / 'runs/jlens-lenses' / LENS_ID / 'lens.json').read_text()
    # The same qualification, sent as a client sends it (null), is the same qualification.
    as_sent = qualification(site.sha, 'q-nan', checks={'score': None, 'margin': None})
    same = site.client.post(endpoint, json={'record': record_for(site.sha, qualifications=[as_sent])})
    assert same.status_code == 200 and same.json()['changed'] is False, same.text
    # A number that really differs is still a different qualification.
    other = qualification(site.sha, 'q-nan', checks={'score': 0.5, 'margin': None})
    differs = site.client.post(endpoint, json={'record': record_for(site.sha, qualifications=[other])})
    assert differs.status_code == 409 and refusal(differs)['code'] == 'lensQualificationDiffers'


def test_added_qualifications_go_in_date_order_after_what_the_site_has(site):
    push(site.client, site.source)
    held = qualification(site.sha, 'q-site', qualifiedAt='2026-11-09T00:00:00Z')
    site.client.post('/api/lens-library/register', json={'record': record_for(site.sha, qualifications=[held])})
    later = qualification(site.sha, 'q-october-20', qualifiedAt='2026-10-20T00:00:00Z')
    earlier = qualification(site.sha, 'q-october-01', qualifiedAt='2026-10-01T00:00:00Z')
    site.client.post('/api/lens-library/register',
                     json={'record': record_for(site.sha, qualifications=[later, held, earlier])})
    assert [q.qualificationID for q in lens_store.resolve(LENS_ID, str(site.root)).qualifications] == [
        'q-site', 'q-october-01', 'q-october-20']
