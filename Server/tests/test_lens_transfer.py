"""The client side of the site lens library (public issues #13 and #14).

A lens is sent once, in parts that follow its layers; a request can then refer
to it instead of carrying it; and a lens can be copied into a second workspace
under the same name. The engine's half (the library, its routes, and a request
that refers to its lens) is tested with the engine. Here the engine is a small
stand-in that speaks the contract's routes, so every expectation below is
about what the CLIENT sends and says.

The sentences both clients say are held to one file,
``prompts/fixtures/lens-library/refusals.json``; the Mac's half is
``Tests/ExperimentKitTests/LensLibraryTests.swift``.
"""
import hashlib
import io
import json
import re
import struct
import sys
import tarfile
from pathlib import Path

import httpx
import numpy as np
import pytest
from safetensors.numpy import save_file

from steerlab_server import client_cli
from steerlab_server.client import diagnostic_commands, diagnostic_workspace, lens_transfer
from steerlab_server.client import runner as runner_api
from steerlab_server.experiment import artifact_imports as imports
from steerlab_server.experiment import diagnostic_archives as archives

FIXTURE = json.loads((Path(__file__).resolve().parents[2]
                      / 'prompts/fixtures/lens-library/refusals.json').read_text(encoding='utf-8'))
SERVER = FIXTURE['server']


# --- a tiny synthetic lens, and a stand-in engine ------------------------------


def make_lens(tmp_path, name='workspace', layers=4, hidden=4, seed=0):
    """A workspace holding one imported lens with ``layers`` fitted layers."""
    folder = tmp_path / (name + '-source')
    folder.mkdir()
    root = tmp_path / name
    root.mkdir()
    rng = np.random.default_rng(seed)
    save_file({f'J{i}': rng.standard_normal((hidden, hidden)).astype(np.float32) for i in range(layers)},
              str(folder / 'weights.safetensors'))
    spec = {'schemaVersion': 1, 'kind': 'jlens', 'modelID': 'example/model', 'modelRevision': None,
            'hiddenSize': hidden, 'layerCount': layers + 1, 'tensorFile': 'weights.safetensors',
            'lens': {'targetLayer': layers, 'layers': {str(i): f'J{i}' for i in range(layers)}, 'promptsFitted': 12}}
    description = folder / 'import.json'
    description.write_text(json.dumps(spec))
    plan = imports.inspect_source(description, root)
    return root.resolve(), imports.publish(description, root, plan['planSHA256'])['lensID']


def converted(root, lens_id):
    return root / 'runs/jlens-lenses' / lens_id / 'jacobians.safetensors'


class Site:
    """An engine's lens library, in memory, speaking the contract's routes."""

    def __init__(self, *, library=True, part_max=1 << 20, external=False, records=True):
        self.library, self.part_max, self.external, self.records = library, part_max, external, records
        self.entries, self.uploads, self.registered = {}, {}, {}
        self.calls, self.fail_part, self.adoptable = [], None, {}

    def sent(self, method, pattern):
        return [path for verb, path in self.calls if verb == method and re.search(pattern, path)]

    def __call__(self, request):
        path, method = request.url.path, request.method
        self.calls.append((method, path))
        body = request.read()
        if path == '/api/capabilities':
            document = {'remoteStudy': {'httpTransfer': not self.external, 'externalTransferRequired': self.external}}
            if self.library is not None:
                document['lensLibrary'] = {'schemaVersion': 1, 'available': self.library, 'partMaxBytes': self.part_max}
            return httpx.Response(200, json=document)
        if path == '/api/bundles/upload':
            return httpx.Response(200, json={'path': '/staged/x.tar.gz', 'filename': 'x.tar.gz', 'bytes': len(body),
                                             'sha256': hashlib.sha256(body).hexdigest(), 'bundle': {}})
        if path.startswith('/api/jlens/lenses/'):
            if not self.records:
                return httpx.Response(403, json={'detail': {'code': 'workbench_required', 'message': 'runner only'}})
            lens_id = path.rsplit('/', 1)[1]
            if lens_id not in self.registered:
                return httpx.Response(404, json={'detail': 'no such lens'})
            return httpx.Response(200, json=self.registered[lens_id])
        if path == '/api/lens-library/register':
            record = json.loads(body)['record']
            assert record['converted']['sha256'] in self.entries, 'register needs the bytes held'
            changed = self.registered.get(record['lensID']) != record
            self.registered[record['lensID']] = record
            return httpx.Response(200, json={'lensID': record['lensID'], 'registered': True, 'changed': changed})
        if path == '/api/lens-library/adopt':
            document = json.loads(body)
            data = self.adoptable[document['path']]
            assert hashlib.sha256(data).hexdigest() == document['sha256']
            self.entries[document['sha256']] = data
            return httpx.Response(200, json={'held': True, 'bytes': len(data)})
        match = re.fullmatch(r'/api/lens-library/([0-9a-f]{64})(?:/(begin|complete|parts/(\d+)))?', path)
        assert match, path
        sha, action, index = match.groups()
        if action is None:
            upload = self.uploads.get(sha)
            return httpx.Response(200, json={
                'held': sha in self.entries, 'bytes': len(self.entries[sha]) if sha in self.entries else None,
                'receivedParts': sorted(upload['parts']) if upload else None})
        if action == 'begin':
            manifest = json.loads(body)
            if sha in self.entries:
                return httpx.Response(200, json={'held': True, 'missingParts': []})
            offset = 0
            for position, part in enumerate(manifest['parts']):
                assert part['index'] == position and part['offset'] == offset and 0 < part['bytes'] <= self.part_max
                offset += part['bytes']
            assert offset == manifest['bytes'], 'the parts must tile the file'
            upload = self.uploads.setdefault(sha, {'manifest': manifest, 'parts': {}})
            if upload['manifest'] != manifest:
                return httpx.Response(409, json={'detail': {'code': 'lensUploadDiffers', 'message': 'another manifest',
                                                            'repairAction': 'wait, then send it again'}})
            return httpx.Response(200, json={
                'held': False, 'missingParts': [p['index'] for p in manifest['parts'] if p['index'] not in upload['parts']]})
        if action == 'complete':
            upload = self.uploads.pop(sha)
            data = b''.join(upload['parts'][p['index']] for p in upload['manifest']['parts'])
            assert hashlib.sha256(data).hexdigest() == sha
            self.entries[sha] = data
            return httpx.Response(200, json={'held': True, 'bytes': len(data)})
        number = int(index)
        if self.fail_part == number:
            self.fail_part = None
            return httpx.Response(500, json={'detail': 'the connection dropped'})
        part = self.uploads[sha]['manifest']['parts'][number]
        assert int(request.headers['content-length']) == part['bytes'] == len(body)
        assert request.headers['x-content-sha256'] == part['sha256'] == hashlib.sha256(body).hexdigest()
        self.uploads[sha]['parts'][number] = body
        return httpx.Response(200, json={'received': True})

    def client(self):
        return runner_api.RunnerClient(base_url=SERVER, http_client=httpx.Client(
            transport=httpx.MockTransport(self), base_url=SERVER))


@pytest.fixture
def wire(monkeypatch):
    """Point the command line's runner verbs at a stand-in engine."""
    real = runner_api.RunnerClient

    def connect(site):
        def factory(**kwargs):
            kwargs['http_client'] = httpx.Client(transport=httpx.MockTransport(site), base_url=SERVER)
            return real(**kwargs)
        monkeypatch.setattr(runner_api, 'RunnerClient', factory)
        monkeypatch.delenv(client_cli.RUNNER_TOKEN_ENV, raising=False)
        monkeypatch.setenv('STEERLAB_ROOT', '')   # restored after the test: the client exports it
    return connect


def cli(capsys, *arguments):
    capsys.readouterr()
    code = client_cli.main([*arguments, '--json'])
    return code, json.loads(capsys.readouterr().out)


# --- the part plan -------------------------------------------------------------


def test_the_parts_follow_the_layers_and_tile_the_file(tmp_path):
    """Hand-derived from the file's own bytes: the header, then one part per
    tensor in file order, with the one part over the limit split in two."""
    root, lens_id = make_lens(tmp_path, layers=4)
    data = converted(root, lens_id).read_bytes()
    header_end = 8 + struct.unpack('<Q', data[:8])[0]
    header = json.loads(data[8:header_end])
    starts = sorted(header_end + entry['data_offsets'][0] for name, entry in header.items() if name != '__metadata__')
    assert len(starts) == 4 and header_end > 64      # four 4x4 float32 tensors of 64 bytes, after a longer header
    limit = header_end - 1                           # so only the header is over the limit: one forced split
    expected = [(0, limit), (limit, 1)] + [(start, 64) for start in starts]

    plan = lens_transfer.push_plan(root, lens_id, limit)

    assert [(part['offset'], part['bytes']) for part in plan['parts']] == expected
    assert [part['index'] for part in plan['parts']] == list(range(6))
    assert sum(part['bytes'] for part in plan['parts']) == plan['bytes'] == len(data)
    for part in plan['parts']:
        assert part['sha256'] == hashlib.sha256(data[part['offset']:part['offset'] + part['bytes']]).hexdigest()
    assert plan['convertedSHA256'] == hashlib.sha256(data).hexdigest()
    assert plan['file'] == f'runs/jlens-lenses/{lens_id}/jacobians.safetensors'
    assert plan['lensID'] == lens_id and plan['changed'] is False
    # The record a site receives names the file by its place in a workspace.
    assert plan['record']['converted']['path'] == plan['file']
    assert plan['record']['lensID'] == lens_id

    # With room for everything, there is no split: the header and four layers.
    assert len(lens_transfer.push_plan(root, lens_id, 1 << 20)['parts']) == 5
    # Every part is at most the limit, however small, and the parts still tile.
    tiny = lens_transfer.push_plan(root, lens_id, 7)['parts']
    assert max(part['bytes'] for part in tiny) == 7
    assert b''.join(data[p['offset']:p['offset'] + p['bytes']] for p in tiny) == data


def test_a_record_with_a_path_on_this_machine_is_sent_without_it(tmp_path):
    root, lens_id = make_lens(tmp_path)
    record_file = root / 'runs/jlens-lenses' / lens_id / 'lens.json'
    record = json.loads(record_file.read_text())
    record['converted']['path'] = str(converted(root, lens_id))     # an older import recorded it absolute
    record_file.write_text(json.dumps(record))
    plan = lens_transfer.push_plan(root, lens_id, 1 << 20)
    assert plan['record']['converted']['path'] == f'runs/jlens-lenses/{lens_id}/jacobians.safetensors'
    assert str(root) not in json.dumps(plan['record'])


def test_bytes_that_differ_from_the_import_are_never_planned(tmp_path):
    root, lens_id = make_lens(tmp_path)
    file = converted(root, lens_id)
    data = bytearray(file.read_bytes())
    data[-1] ^= 1
    file.write_bytes(bytes(data))
    with pytest.raises(lens_transfer.LensRefusal, match='differ from the imported hash'):
        lens_transfer.push_plan(root, lens_id, 1 << 20)
    file.write_bytes(b'not a tensor file')
    with pytest.raises(lens_transfer.LensRefusal, match='not a readable safetensors file'):
        lens_transfer.push_plan(root, lens_id, 1 << 20)
    with pytest.raises(lens_transfer.LensRefusal, match='holds no lens named absent') as caught:
        lens_transfer.push_plan(root, 'absent', 1 << 20)
    assert caught.value.code == 'lensNotFound'
    for wrong in (0, -1, 'many', None, 1.5):
        with pytest.raises(archives.MalformedRequest):
            lens_transfer.push_plan(root, lens_id, wrong)
    with pytest.raises(archives.Refusal):
        lens_transfer.push_plan(root, '../' + lens_id, 1 << 20)


# --- sending a lens ------------------------------------------------------------


def test_a_fresh_push_sends_every_part_once_and_registers_the_record(tmp_path):
    root, lens_id = make_lens(tmp_path)
    data = converted(root, lens_id).read_bytes()
    site = Site(part_max=100)
    with site.client() as client:
        result = lens_transfer.push(client, root, lens_id)
    sha = hashlib.sha256(data).hexdigest()
    assert site.entries == {sha: data}
    parts = len(lens_transfer.push_plan(root, lens_id, 100)['parts'])
    assert result == {'lensID': lens_id, 'convertedSHA256': sha, 'bytes': len(data), 'held': True,
                      'alreadyHeld': False, 'adopted': False, 'uploadedParts': parts, 'uploadedBytes': len(data),
                      'registered': True, 'recordChanged': True, 'changed': True}
    assert site.registered[lens_id]['converted'] == {
        **json.loads((root / 'runs/jlens-lenses' / lens_id / 'lens.json').read_text())['converted'],
        'path': f'runs/jlens-lenses/{lens_id}/jacobians.safetensors'}
    # In the contract's order: the capability, what is held, the manifest, the parts, the whole, the record.
    order = [path.rsplit('/', 1)[-1] if 'parts' not in path else 'part' for _, path in site.calls]
    assert order == ['capabilities', sha, 'begin'] + ['part'] * parts + ['complete', 'register']


def test_a_push_run_again_after_an_interruption_sends_only_what_is_missing(tmp_path):
    root, lens_id = make_lens(tmp_path)
    site = Site(part_max=100)
    site.fail_part = 3
    with site.client() as client:
        with pytest.raises(runner_api.RunnerHTTPError):
            lens_transfer.push(client, root, lens_id)
        assert not site.entries and not site.registered
        received = sorted(next(iter(site.uploads.values()))['parts'])
        assert received == [0, 1, 2]
        site.calls.clear()
        result = lens_transfer.push(client, root, lens_id)
    total = len(lens_transfer.push_plan(root, lens_id, 100)['parts'])
    assert [int(path.rsplit('/', 1)[1]) for path in site.sent('PUT', '/parts/')] == list(range(3, total))
    assert result['uploadedParts'] == total - 3 and result['alreadyHeld'] is False
    assert site.entries[result['convertedSHA256']] == converted(root, lens_id).read_bytes()
    assert lens_id in site.registered


def test_a_lens_the_site_holds_uploads_nothing_and_still_registers(tmp_path):
    root, lens_id = make_lens(tmp_path)
    data = converted(root, lens_id).read_bytes()
    site = Site()
    site.entries[hashlib.sha256(data).hexdigest()] = data        # fitted on the site, or sent from another workspace
    with site.client() as client:
        result = lens_transfer.push(client, root, lens_id)
        assert not site.sent('PUT', '') and not site.sent('POST', 'begin|complete')
        assert result['alreadyHeld'] is True and result['uploadedParts'] == 0 and result['uploadedBytes'] == 0
        assert result['registered'] is True and result['changed'] is True       # the record was new to the site
        assert lens_id in site.registered
        again = lens_transfer.push(client, root, lens_id)
    assert again['changed'] is False and again['recordChanged'] is False
    for case in FIXTURE['pushMessages']:
        assert lens_transfer.push_message(case) == case['message']


@pytest.mark.parametrize('library', [None, False])
def test_an_engine_with_no_lens_library_is_sent_nothing(tmp_path, library):
    """Older (the capability is absent) or unconfigured (it is false): one
    sentence, the same on the Mac, and no request but the capability read."""
    root, lens_id = make_lens(tmp_path)
    site = Site(library=library)
    with site.client() as client, pytest.raises(runner_api.RunnerRefusal) as caught:
        lens_transfer.push(client, root, lens_id)
    assert site.calls == [('GET', '/api/capabilities')]
    assert caught.value.code == FIXTURE['unavailable']['code'] == 'lensLibraryUnavailable'
    assert caught.value.reason == FIXTURE['unavailable']['pushReason']
    assert caught.value.repair_action == FIXTURE['unavailable']['repairAction']
    assert caught.value.state == 'refused'


def test_a_site_that_forbids_http_transfer_adopts_a_copy_already_on_it(tmp_path):
    root, lens_id = make_lens(tmp_path)
    data = converted(root, lens_id).read_bytes()
    site = Site(external=True)
    with site.client() as client:
        with pytest.raises(runner_api.RunnerRefusal) as caught:
            lens_transfer.push(client, root, lens_id)
        assert caught.value.code == 'externalTransferRequired'
        assert not site.sent('POST', '') and not site.sent('PUT', '')
        site.adoptable['/scratch/incoming/jacobians.safetensors'] = data
        result = lens_transfer.push(client, root, lens_id, adopt='/scratch/incoming/jacobians.safetensors')
    assert result['adopted'] is True and result['uploadedParts'] == 0 and result['changed'] is True
    assert site.entries == {hashlib.sha256(data).hexdigest(): data} and lens_id in site.registered
    assert not site.sent('PUT', '') and not site.sent('POST', 'begin')
    assert lens_transfer.external_transfer_reason(SERVER) == FIXTURE['externalTransfer']['reason']
    assert lens_transfer.external_transfer_repair(FIXTURE['file']) == FIXTURE['externalTransfer']['repairAction']


def test_status_says_what_the_site_holds_and_changes_nothing(tmp_path):
    root, lens_id = make_lens(tmp_path)
    site = Site()
    with site.client() as client:
        before = lens_transfer.status(client, root, lens_id)
        assert (before['held'], before['siteRecord'], before['libraryAvailable']) == (False, 'absent', True)
        assert all(method == 'GET' for method, _ in site.calls)
        lens_transfer.push(client, root, lens_id)
        after = lens_transfer.status(client, root, lens_id)
        assert (after['held'], after['siteRecord'], after['changed']) == (True, 'registered', False)
        site.registered[lens_id]['converted']['sha256'] = 'f' * 64
        assert lens_transfer.status(client, root, lens_id)['siteRecord'] == 'different'
        site.records = False                         # a runner that serves no lens records
        assert lens_transfer.status(client, root, lens_id)['siteRecord'] == 'unknown'
    with Site(library=None).client() as client:
        old = lens_transfer.status(client, root, lens_id)
    assert (old['libraryAvailable'], old['held']) == (False, False)
    for case in FIXTURE['statusMessages']:
        assert lens_transfer.status_message(case) == case['message']


def test_the_command_line_pushes_and_reports(tmp_path, wire, capsys):
    root, lens_id = make_lens(tmp_path)
    site = Site(part_max=100)
    wire(site)
    code, document = cli(capsys, '--root', str(root), 'runner', 'lens-push', lens_id, '--runner', SERVER)
    assert code == 0 and document['state'] == 'ready' and document['changed'] is True
    response = document['result']['response']
    assert response['uploadedBytes'] == converted(root, lens_id).stat().st_size and response['registered'] is True
    assert document['message'].startswith(f"Sent lens {lens_id} to the site's lens library")
    code, document = cli(capsys, '--root', str(root), 'runner', 'lens-status', lens_id, '--runner', SERVER)
    assert code == 0 and document['changed'] is False
    assert document['message'] == f"The site's lens library holds lens {lens_id}; the site has its record."
    # A lens the workspace does not hold, and no workspace at all.
    code, document = cli(capsys, '--root', str(root), 'runner', 'lens-push', 'absent', '--runner', SERVER)
    assert (code, document['state'], document['error']['code']) == (65, 'refused', 'lensNotFound')
    code, document = cli(capsys, 'runner', 'lens-status', lens_id, '--runner', SERVER)
    assert (code, document['error']['code']) == (64, 'usage')
    assert '--root' in document['error']['repairAction']
    # An engine with no library: refused with the shared sentence.
    wire(Site(library=None))
    code, document = cli(capsys, '--root', str(root), 'runner', 'lens-push', lens_id, '--runner', SERVER)
    assert (code, document['error']['code']) == (65, 'lensLibraryUnavailable')
    assert document['error']['reason'] == FIXTURE['unavailable']['pushReason']


# --- a request that refers to its lens -----------------------------------------


def request_archive(tmp_path, name, context):
    """A diagnostic input archive with one small member and this context."""
    path = tmp_path / name
    metadata = archives.encoded({'schemaVersion': 1, 'kind': 'diagnosticInput', 'context': context, 'entries': []})
    with tarfile.open(path, 'w:gz') as archive:
        for member, data in (('prompts/request.json', b'{}'), (archives.META, metadata)):
            info = tarfile.TarInfo(member)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def test_a_request_that_refers_to_its_lens_is_not_uploaded_to_an_engine_without_a_library(tmp_path):
    lenses = [{'lensID': 'custom-lens-example', 'convertedSHA256': 'a' * 64, 'bytes': 7}]
    referring = request_archive(tmp_path, 'referring.tar.gz', {'request': {}, 'lensReference': True, 'libraryLenses': lenses})
    carrying = request_archive(tmp_path, 'carrying.tar.gz', {'request': {}})
    assert lens_transfer.archive_library_lenses(referring) == lenses
    assert lens_transfer.archive_library_lenses(carrying) is None
    assert lens_transfer.archive_library_lenses(tmp_path / 'absent.tar.gz') is None
    for library in (None, False):
        site = Site(library=library)
        with site.client() as client:
            with pytest.raises(runner_api.RunnerRefusal) as caught:
                client.upload_run_bundle(str(referring))
            assert not site.sent('POST', '/api/bundles/upload'), 'nothing may be sent'
            assert caught.value.code == 'lensLibraryUnavailable'
            assert caught.value.reason == FIXTURE['unavailable']['archiveReason']
            assert caught.value.repair_action == FIXTURE['unavailable']['repairAction']
            assert caught.value.detail['libraryLenses'] == lenses
            # A request that carries its lens goes to any engine, as before.
            client.upload_run_bundle(str(carrying))
            assert site.sent('POST', '/api/bundles/upload')
    site = Site()
    with site.client() as client:
        client.upload_run_bundle(str(referring))
    assert site.sent('POST', '/api/bundles/upload')


def test_an_upload_over_the_limit_says_a_lens_can_be_referred_to(tmp_path):
    carrying = request_archive(tmp_path, 'carrying.tar.gz', {'request': {}})

    def handler(request):
        if request.url.path == '/api/capabilities':
            return httpx.Response(200, json={})
        request.read()
        return httpx.Response(413, json={'detail': 'upload exceeds 4294967296 bytes (STEERLAB_MAX_UPLOAD_BYTES)'})
    client = runner_api.RunnerClient(base_url=SERVER, http_client=httpx.Client(
        transport=httpx.MockTransport(handler), base_url=SERVER))
    with client, pytest.raises(runner_api.RunnerHTTPError) as caught:
        client.upload_run_bundle(str(carrying))
    assert caught.value.status == 413 and caught.value.code == 'runnerRefused'
    assert caught.value.repair_action == FIXTURE['oversize']['pythonRepair']
    assert FIXTURE['oversize']['advice'] in caught.value.repair_action
    assert lens_transfer.OVERSIZE_LENS_ADVICE == FIXTURE['oversize']['advice']
    assert FIXTURE['oversize']['advice'] in FIXTURE['oversize']['macRepair']


def test_the_lens_reference_flag_reaches_the_planner_and_carry_mode_is_called_as_before(tmp_path, monkeypatch, capsys):
    """The one shared call site: ``--lens-reference`` passes
    ``lens_reference=True`` to the plan and package owners. Without the flag
    they are called exactly as they were before it existed."""
    from steerlab_server.experiment import diagnostic_inputs
    calls = []
    monkeypatch.setattr(diagnostic_inputs, 'plan', lambda *args, **kwargs: calls.append(('plan', args[1:], kwargs)) or {'changed': False})
    monkeypatch.setattr(diagnostic_inputs, 'package', lambda *args, **kwargs: calls.append(('package', args[1:], kwargs)) or {'changed': True})
    root = (tmp_path / 'workspace')
    root.mkdir()
    root = str(root.resolve())
    request = tmp_path / 'request.json'
    request.write_text('{}')
    monkeypatch.setenv('STEERLAB_ROOT', '')
    base = {'workspaceRoot': root, 'requestFile': str(request)}
    packaged = {**base, 'archivePath': str(tmp_path / 'out.tar.gz'), 'planSHA256': 'a' * 64}

    diagnostic_commands.workspace_action('input-plan', base)
    diagnostic_commands.workspace_action('package', packaged)
    assert calls == [('plan', (root,), {}), ('package', (root, packaged['archivePath'], 'a' * 64), {'acknowledge': None})]
    calls.clear()
    diagnostic_commands.workspace_action('input-plan', {**base, 'lensReference': False})
    assert calls == [('plan', (root,), {})]
    calls.clear()

    diagnostic_commands.workspace_action('input-plan', {**base, 'lensReference': True})
    diagnostic_commands.workspace_action('package', {**packaged, 'lensReference': True})
    assert calls == [('plan', (root,), {'lens_reference': True}),
                     ('package', (root, packaged['archivePath'], 'a' * 64), {'acknowledge': None, 'lens_reference': True})]
    calls.clear()
    with pytest.raises(archives.MalformedRequest, match='lensReference is true or false'):
        diagnostic_commands.workspace_action('input-plan', {**base, 'lensReference': 'yes'})

    # …and from the command line's flag.
    assert cli(capsys, '--root', root, 'science', 'input-plan', str(request), '--lens-reference')[0] == 0
    assert cli(capsys, '--root', root, 'science', 'package', str(request), '--archive', packaged['archivePath'],
               '--plan-sha256', 'a' * 64, '--lens-reference')[0] == 0
    assert cli(capsys, '--root', root, 'science', 'input-plan', str(request))[0] == 0
    assert [kwargs for _, _, kwargs in calls] == [{'lens_reference': True}, {'acknowledge': None, 'lens_reference': True}, {}]


# --- the same lens in a second workspace ---------------------------------------


def second_workspace(tmp_path):
    root = tmp_path / 'second'
    root.mkdir()
    return root.resolve()


def tree(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes() for path in sorted(directory.rglob('*')) if path.is_file()}


@pytest.mark.skipif(sys.platform != 'darwin', reason='clonefile is a macOS call')
def test_a_copy_on_macos_is_a_clone_under_the_same_name_and_record(tmp_path):
    origin, lens_id = make_lens(tmp_path, name='first')
    root = second_workspace(tmp_path)
    preview = lens_transfer.copy(root, lens_id, str(origin))
    assert preview['changed'] is False and preview['existing'] is False and preview['action'] == 'copy'
    assert not (root / 'runs').exists(), 'a preview writes nothing'
    assert {entry['path'].rsplit('/', 1)[-1] for entry in preview['files']} >= {'lens.json', 'jacobians.safetensors', 'import-receipt.json'}
    result = lens_transfer.copy(root, lens_id, str(origin), preview['planSHA256'])
    assert result['copyMethod'] == 'clone' and result['changed'] is True
    assert tree(root / 'runs/jlens-lenses' / lens_id) == tree(origin / 'runs/jlens-lenses' / lens_id)
    assert result['outputDirectory'] == str(root / 'runs/jlens-lenses' / lens_id)
    # The copy is a working lens here: its plan is the origin's, byte for byte.
    assert lens_transfer.push_plan(root, lens_id, 1 << 20)['parts'] == lens_transfer.push_plan(origin, lens_id, 1 << 20)['parts']


def test_a_copy_that_cannot_clone_is_a_plain_copy(tmp_path, monkeypatch):
    origin, lens_id = make_lens(tmp_path, name='first')
    root = second_workspace(tmp_path)
    monkeypatch.setattr(lens_transfer, '_clone', lambda source, target: False)
    preview = lens_transfer.copy(root, lens_id, str(origin))
    result = lens_transfer.copy(root, lens_id, str(origin), preview['planSHA256'])
    assert result['copyMethod'] == 'copy' and result['changed'] is True
    assert tree(root / 'runs/jlens-lenses' / lens_id) == tree(origin / 'runs/jlens-lenses' / lens_id)


def test_the_same_lens_already_here_changes_nothing_and_a_different_one_is_refused(tmp_path):
    origin, lens_id = make_lens(tmp_path, name='first')
    root = second_workspace(tmp_path)
    preview = lens_transfer.copy(root, lens_id, str(origin))
    lens_transfer.copy(root, lens_id, str(origin), preview['planSHA256'])
    before = tree(root / 'runs')
    again = lens_transfer.copy(root, lens_id, str(origin))
    assert again['action'] == 'none' and again['existing'] is True and again['changed'] is False
    applied = lens_transfer.copy(root, lens_id, str(origin), again['planSHA256'])
    assert applied['existing'] is True and applied['changed'] is False and 'copyMethod' not in applied
    assert tree(root / 'runs') == before
    # The first review's digest no longer describes this workspace.
    with pytest.raises(lens_transfer.LensRefusal, match='changed after the copy was reviewed'):
        lens_transfer.copy(root, lens_id, str(origin), preview['planSHA256'])

    # A different lens under that name: different bytes, same folder name.
    other, other_id = make_lens(tmp_path, name='third', seed=7)
    (other / 'runs/jlens-lenses' / other_id).rename(other / 'runs/jlens-lenses' / lens_id)
    record_file = other / 'runs/jlens-lenses' / lens_id / 'lens.json'
    record = json.loads(record_file.read_text())
    record['lensID'] = lens_id
    record['converted']['path'] = f'runs/jlens-lenses/{lens_id}/jacobians.safetensors'
    record_file.write_text(json.dumps(record))
    with pytest.raises(lens_transfer.LensRefusal, match='already holds a different lens') as caught:
        lens_transfer.copy(root, lens_id, str(other))
    assert caught.value.code == 'lensNameTaken'
    assert tree(root / 'runs') == before


def test_a_copy_checks_every_file_before_and_after(tmp_path, monkeypatch):
    origin, lens_id = make_lens(tmp_path, name='first')
    root = second_workspace(tmp_path)
    # Before: bytes that differ from the import are not copied.
    file = converted(origin, lens_id)
    good = file.read_bytes()
    file.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
    with pytest.raises(lens_transfer.LensRefusal, match='differ from the imported hash'):
        lens_transfer.copy(root, lens_id, str(origin))
    file.write_bytes(good)
    preview = lens_transfer.copy(root, lens_id, str(origin))

    # After: a copy that did not arrive intact is not published.
    def corrupting(source, target):
        Path(target).write_bytes(Path(source).read_bytes()[:-1] + b'\x00' if str(source).endswith('.safetensors')
                                 else Path(source).read_bytes())
        return True
    monkeypatch.setattr(lens_transfer, '_clone', corrupting)
    with pytest.raises(lens_transfer.LensRefusal, match='changed while it was being copied'):
        lens_transfer.copy(root, lens_id, str(origin), preview['planSHA256'])
    assert not (root / 'runs/jlens-lenses' / lens_id).exists()
    assert not list((root / 'runs/jlens-lenses').iterdir()), 'no staging folder is left behind'

    with pytest.raises(lens_transfer.LensRefusal, match='names this workspace'):
        lens_transfer.copy(origin, lens_id, str(origin))
    with pytest.raises(lens_transfer.LensRefusal, match='no workspace folder'):
        lens_transfer.copy(root, lens_id, str(tmp_path / 'nowhere'))


def test_the_command_line_reviews_then_copies(tmp_path, capsys, monkeypatch):
    origin, lens_id = make_lens(tmp_path, name='first')
    root = second_workspace(tmp_path)
    monkeypatch.setenv('STEERLAB_ROOT', '')
    code, document = cli(capsys, '--root', str(root), 'science', 'lens-copy', lens_id, '--from', str(origin))
    assert code == 0 and document['changed'] is False
    digest = document['result']['planSHA256']
    assert document['message'].endswith(f'Repeat the command with --plan-sha256 {digest} to copy it.')
    code, document = cli(capsys, '--root', str(root), 'science', 'lens-copy', lens_id, '--from', str(origin),
                         '--plan-sha256', digest)
    assert code == 0 and document['changed'] is True and document['result']['copyMethod'] in ('clone', 'copy')
    assert (root / 'runs/jlens-lenses' / lens_id / 'lens.json').is_file()
    code, document = cli(capsys, '--root', str(root), 'science', 'lens-copy', lens_id, '--from', str(origin))
    assert code == 0 and document['message'] == f'This workspace already holds lens {lens_id}; nothing was copied.'
    code, document = cli(capsys, '--root', str(root), 'science', 'lens-copy', 'absent', '--from', str(origin))
    assert (code, document['state'], document['error']['code']) == (65, 'refused', 'lensNotFound')
    for case in FIXTURE['copyMessages']:
        assert lens_transfer.copy_message({**case, 'files': [None] * case['fileCount']}) == case['message']


# --- the bridge the Mac uses ---------------------------------------------------


def bridge(monkeypatch, capsys, action, payload):
    document = {'action': action, 'payload': payload, 'clientSHA256': diagnostic_workspace.source_sha256()}
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(document)))
    capsys.readouterr()
    status = diagnostic_workspace.main()
    return status, json.loads(capsys.readouterr().out)


def test_the_lens_actions_are_the_bridges_and_no_http_route_serves_them(tmp_path, monkeypatch, capsys):
    root, lens_id = make_lens(tmp_path)
    # The Mac sends numbers as JSON numbers; a whole float is the same limit.
    status, answer = bridge(monkeypatch, capsys, 'lens-push-plan',
                            {'workspaceRoot': str(root), 'lensID': lens_id, 'partMaxBytes': 100.0})
    assert status == 0 and answer['ok'] is True
    assert answer['result'] == lens_transfer.push_plan(root, lens_id, 100)
    status, answer = bridge(monkeypatch, capsys, 'lens-record', {'workspaceRoot': str(root), 'lensID': lens_id})
    assert status == 0 and 'parts' not in answer['result']
    assert answer['result']['convertedSHA256'] == hashlib.sha256(converted(root, lens_id).read_bytes()).hexdigest()
    assert answer['result']['bytes'] == converted(root, lens_id).stat().st_size
    status, answer = bridge(monkeypatch, capsys, 'lens-push-plan', {'workspaceRoot': str(root), 'lensID': 'absent', 'partMaxBytes': 100})
    assert (status, answer['ok'], answer['state'], answer['code']) == (65, False, 'refused', 'lensNotFound')
    status, answer = bridge(monkeypatch, capsys, 'lens-copy', {'workspaceRoot': str(root), 'lensID': lens_id})
    assert (answer['ok'], answer['state']) == (False, 'blocked')
    # `workspace_action` is what the workbench's HTTP route serves: a caller
    # there must not be able to name a second folder on the machine.
    for action in diagnostic_commands.LOCAL_ACTIONS:
        with pytest.raises(archives.MalformedRequest, match='Unknown diagnostic workspace operation'):
            diagnostic_commands.workspace_action(action, {'workspaceRoot': str(root), 'lensID': lens_id})
