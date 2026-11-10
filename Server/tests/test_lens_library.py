"""One lens per site, found by its hash (public issues #13 and #14).

The library keeps a lens's converted file once, under its SHA-256, outside any
workspace. These tests drive it the way a client does, over its routes, with
tiny synthetic lenses: where the library is, what it says it can do, an upload
in checked parts (planned here from the safetensors layout alone, as a sender
plans it), a file adopted from the site, a record registered in the serving
workspace, and every way a path or a key could reach outside the library.
"""
import hashlib
import json
import os
import struct
import threading
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
    monkeypatch.setenv('STEERLAB_LENS_LIBRARY_ROOT', '')
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
        assert detail['repairAction'] == ("Package the request without `--lens-reference`, or set the site "
                                         "profile's `lensLibrary` folder.")
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


def test_parts_obey_the_sites_transfer_policy(site, monkeypatch):
    plan = part_plan(site.source, 64)
    assert site.client.post(f'/api/lens-library/{site.sha}/begin', json=plan).status_code == 200
    monkeypatch.setenv('STEERLAB_TRANSFER_METHOD', 'globus')
    response = send_part(site.client, site.sha, plan, site.source.read_bytes(), 0)
    assert response.status_code == 403 and response.json()['detail']['code'] == 'external_transfer_required'
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
