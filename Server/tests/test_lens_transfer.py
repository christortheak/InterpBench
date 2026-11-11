"""The client side of the site lens library (public issues #13 and #14).

A lens is sent once, in parts that follow its layers; a request can then refer
to it instead of carrying it; and a lens can be copied into a second workspace
under the same name. The engine's half (the library, its routes, and a request
that refers to its lens) is tested with the engine. Here the engine is a small
stand-in that speaks those routes as the engine does, so every expectation
below is about what the CLIENT sends and says.

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
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from safetensors.numpy import save_file

from steerlab_server import client_cli
from steerlab_server.client import diagnostic_commands, diagnostic_workspace, lens_transfer, science_commands
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


def lens_folder(root, lens_id):
    return root / 'runs/jlens-lenses' / lens_id


def converted(root, lens_id):
    return lens_folder(root, lens_id) / 'jacobians.safetensors'


def strict_loads(text):
    """Parse as a strict JSON reader does: NaN and infinity are errors."""
    def refuse(constant):
        raise ValueError('not JSON: ' + constant)
    return json.loads(text, parse_constant=refuse)


def refusal(status, code, sentence, repair):
    """A refusal as the engine's lens-library routes write one."""
    return httpx.Response(status, json={'detail': {'code': code, 'reason': sentence, 'message': sentence,
                                                   'repairAction': repair}})


class Site:
    """An engine's lens library, in memory, speaking its routes."""

    def __init__(self, *, library=True, part_max=1 << 20, external=False, records=True):
        self.library, self.part_max, self.external, self.records = library, part_max, external, records
        self.entries, self.uploads, self.registered = {}, {}, {}
        self.calls, self.adoptable, self.timeouts = [], {}, {}
        self.fail_part = self.drop_part = None        # answer one part with a 500, or with no answer at all
        self.fail_complete = False
        self.answers = {}                             # an answer the contract does not allow, by route name

    def sent(self, method, pattern):
        return [path for verb, path in self.calls if verb == method and re.search(pattern, path)]

    def __call__(self, request):
        path, method = request.url.path, request.method
        self.calls.append((method, path))
        self.timeouts[path] = request.extensions.get('timeout')
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
        if not self.library:
            return refusal(409, 'lensLibraryUnavailable', 'This engine has no lens library.', 'Configure one.')
        if path == '/api/lens-library/register':
            try:
                record = strict_loads(body)['record']
            except ValueError:
                return refusal(400, 'lensRecordMalformed', 'The record is not JSON.', 'Send strict JSON.')
            if record['converted']['sha256'] not in self.entries:
                return refusal(409, 'lensNotInLibrary', 'The library does not hold this lens.', 'Send it first.')
            if 'register' in self.answers:
                return httpx.Response(200, json=self.answers['register'])
            changed = self.registered.get(record['lensID']) != record
            self.registered[record['lensID']] = record
            return httpx.Response(200, json={'lensID': record['lensID'], 'registered': True, 'changed': changed})
        if path == '/api/lens-library/adopt':
            document = json.loads(body)
            if document['path'] not in self.adoptable:
                return refusal(409, 'lensAdoptMissing', 'No file is at that path.', 'Copy the file there first.')
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
            assert set(manifest) == {'bytes', 'parts'}
            if sha in self.entries:
                return httpx.Response(200, json={'held': True, 'missingParts': []})
            offset = 0
            for position, part in enumerate(manifest['parts']):
                assert set(part) == {'index', 'offset', 'bytes', 'sha256'}
                if not (part['index'] == position and part['offset'] == offset and 0 < part['bytes'] <= self.part_max):
                    return refusal(400, 'lensUploadPlanMalformed', 'The parts do not tile the file.', 'Plan it again.')
                offset += part['bytes']
            assert offset == manifest['bytes'], 'the parts must tile the file'
            upload = self.uploads.setdefault(sha, {'manifest': manifest, 'parts': {}})
            if upload['manifest'] != manifest:
                return refusal(409, 'lensUploadPlanDiffers', 'Another plan is in progress for this lens.',
                               'Wait for it to finish, then send the lens again.')
            if 'begin' in self.answers:
                return httpx.Response(200, json=self.answers['begin'])
            return httpx.Response(200, json={
                'held': False, 'missingParts': [p['index'] for p in manifest['parts'] if p['index'] not in upload['parts']]})
        if action == 'complete':
            assert json.loads(body) == {}
            if self.fail_complete:
                self.fail_complete = False
                return httpx.Response(500, json={'detail': 'the engine stopped'})
            if 'complete' in self.answers:
                return httpx.Response(200, json=self.answers['complete'])
            if sha in self.entries:                       # repeating it is safe
                return httpx.Response(200, json={'held': True, 'bytes': len(self.entries[sha])})
            upload = self.uploads.pop(sha)
            data = b''.join(upload['parts'][p['index']] for p in upload['manifest']['parts'])
            assert hashlib.sha256(data).hexdigest() == sha
            self.entries[sha] = data
            return httpx.Response(200, json={'held': True, 'bytes': len(data)})
        number = int(index)
        if self.drop_part == number:
            self.drop_part = None
            raise httpx.ReadTimeout('the read timed out', request=request)
        if self.fail_part == number:
            self.fail_part = None
            return httpx.Response(500, json={'detail': 'the connection dropped'})
        part = self.uploads[sha]['manifest']['parts'][number]
        assert int(request.headers['content-length']) == part['bytes']
        if request.headers['x-content-sha256'] != part['sha256'] or hashlib.sha256(body).hexdigest() != part['sha256']:
            return refusal(409, 'lensUploadPartCorrupt', 'The part differs from its plan.', 'Send it again.')
        self.uploads[sha]['parts'][number] = body
        return httpx.Response(200, json={'received': True})

    def client(self, **options):
        return runner_api.RunnerClient(base_url=SERVER, http_client=httpx.Client(
            transport=httpx.MockTransport(self), base_url=SERVER), **options)


def client_for(handler):
    return runner_api.RunnerClient(base_url=SERVER, http_client=httpx.Client(
        transport=httpx.MockTransport(handler), base_url=SERVER))


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


def not_the_archive_familys(repair):
    """A lens verb's repair is about lenses, never the transport family's default."""
    return repair != archives.Refusal.repair_action and 'diagnostic archive' not in repair and 'custody' not in repair


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
    assert plan['lensID'] == lens_id and plan['changed'] is False and plan['nonFiniteNumbers'] == 0
    # The record a site receives names the file by its place in a workspace.
    assert plan['record']['converted']['path'] == plan['file']
    assert plan['record']['lensID'] == lens_id
    assert strict_loads(plan['recordJSON']) == plan['record']

    # With room for everything, there is no split: the header and four layers.
    assert len(lens_transfer.push_plan(root, lens_id, 1 << 20)['parts']) == 5
    # Every part is at most the limit, however small, and the parts still tile.
    tiny = lens_transfer.push_plan(root, lens_id, 7)['parts']
    assert max(part['bytes'] for part in tiny) == 7
    assert b''.join(data[p['offset']:p['offset'] + p['bytes']] for p in tiny) == data


def test_a_record_with_a_path_on_another_machine_is_sent_without_it_and_a_missing_file_is_called_missing(tmp_path):
    root, lens_id = make_lens(tmp_path)
    record_file = lens_folder(root, lens_id) / 'lens.json'
    record = json.loads(record_file.read_text())
    # An import on another machine recorded the file by its full path there.
    record['converted']['path'] = f'/elsewhere/workspace/runs/jlens-lenses/{lens_id}/jacobians.safetensors'
    record_file.write_text(json.dumps(record))
    plan = lens_transfer.push_plan(root, lens_id, 1 << 20)
    assert plan['record']['converted']['path'] == f'runs/jlens-lenses/{lens_id}/jacobians.safetensors'
    assert '/elsewhere' not in plan['recordJSON'] and str(root) not in plan['recordJSON']

    # The file gone, the lens is not "outside" anything: its file is missing.
    converted(root, lens_id).unlink()
    for call in (lambda: lens_transfer.push_plan(root, lens_id, 1 << 20), lambda: lens_transfer.lens_record(root, lens_id)):
        with pytest.raises(lens_transfer.LensRefusal) as caught:
            call()
        assert str(caught.value) == (f'The converted file of lens {lens_id} is missing from this workspace '
                                     f'(runs/jlens-lenses/{lens_id}/jacobians.safetensors).')
        assert caught.value.code == 'lensFileMissing'
        assert caught.value.repair_action.startswith('Import the lens again from its source files')


def test_bytes_that_differ_from_the_import_are_never_planned(tmp_path):
    root, lens_id = make_lens(tmp_path)
    file = converted(root, lens_id)
    data = bytearray(file.read_bytes())
    data[-1] ^= 1
    file.write_bytes(bytes(data))
    with pytest.raises(lens_transfer.LensRefusal, match='differs from the SHA-256 its record names') as caught:
        lens_transfer.push_plan(root, lens_id, 1 << 20)
    assert caught.value.code == 'lensFileDiffers' and caught.value.repair_action.startswith('Move that file aside')
    file.write_bytes(b'not a tensor file')
    with pytest.raises(lens_transfer.LensRefusal, match='is not a readable safetensors file') as caught:
        lens_transfer.push_plan(root, lens_id, 1 << 20)
    assert caught.value.repair_action.startswith('Move that file aside')
    with pytest.raises(lens_transfer.LensRefusal, match='^This workspace holds no lens named absent') as caught:
        lens_transfer.push_plan(root, 'absent', 1 << 20)
    assert caught.value.code == 'lensNotFound'
    for wrong in (0, -1, 'many', None, 1.5, True):
        with pytest.raises(archives.MalformedRequest) as caught:
            lens_transfer.push_plan(root, lens_id, wrong)
        assert not_the_archive_familys(caught.value.repair_action)
    # A lens ID is one folder name; anything else is said so, with a repair about lenses.
    for wrong in ('../' + lens_id, 'a/b', '', '..', 'a\\b', 7):
        with pytest.raises(archives.MalformedRequest, match='Name one lens by its lensID') as caught:
            lens_transfer.push_plan(root, wrong, 1 << 20)
        assert caught.value.repair_action == 'Give the name of one folder under runs/jlens-lenses/ in the workspace.'


def test_a_symbolic_link_on_the_way_to_a_lens_is_refused_in_words_about_lenses(tmp_path):
    root, lens_id = make_lens(tmp_path)
    elsewhere = tmp_path / 'larger-disk'
    (root / 'runs/jlens-lenses').rename(elsewhere)
    (root / 'runs/jlens-lenses').symlink_to(elsewhere)
    for call in (lambda: lens_transfer.push_plan(root, lens_id, 1 << 20), lambda: lens_transfer.lens_record(root, lens_id)):
        with pytest.raises(lens_transfer.LensRefusal, match='reached through a symbolic link') as caught:
            call()
        assert caught.value.code == 'lensPathNotOrdinary' and not_the_archive_familys(caught.value.repair_action)
    other = tmp_path / 'second'
    other.mkdir()
    with pytest.raises(lens_transfer.LensRefusal, match='reached through a symbolic link') as caught:
        lens_transfer.copy(other, lens_id, str(root))
    assert not_the_archive_familys(caught.value.repair_action)


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
                      'registered': True, 'recordChanged': True, 'recordNonFiniteNumbers': 0, 'changed': True}
    assert site.registered[lens_id]['converted'] == {
        **json.loads((lens_folder(root, lens_id) / 'lens.json').read_text())['converted'],
        'path': f'runs/jlens-lenses/{lens_id}/jacobians.safetensors'}
    # In the contract's order: the capability, what is held, the manifest, the parts, the whole, the record.
    order = [path.rsplit('/', 1)[-1] if 'parts' not in path else 'part' for _, path in site.calls]
    assert order == ['capabilities', sha, 'begin'] + ['part'] * parts + ['complete', 'register']


@pytest.mark.parametrize('failure', ['a 500 on a part', 'no answer to a part', 'a 500 on the whole-file check'])
def test_an_interrupted_push_says_to_run_it_again_and_then_sends_only_what_is_missing(tmp_path, failure):
    root, lens_id = make_lens(tmp_path)
    total = len(lens_transfer.push_plan(root, lens_id, 100)['parts'])
    site = Site(part_max=100)
    if failure == 'a 500 on a part':
        site.fail_part = 3
    elif failure == 'no answer to a part':
        site.drop_part = 3
    else:
        site.fail_complete = True
    with site.client() as client:
        with pytest.raises(runner_api.RunnerError) as caught:
            lens_transfer.push(client, root, lens_id)
        error = caught.value
        # One sentence, the Mac's too: run it again, and only what is missing is sent. Nothing about jobs.
        assert (error.code, error.state) == (FIXTURE['interrupted']['code'], 'failed') == ('lensUploadInterrupted', 'failed')
        assert error.repair_action == FIXTURE['interrupted']['repairAction']
        assert error.reason.startswith(f'Sending lens {lens_id} to the site was interrupted before the site '
                                       'confirmed that it holds the whole lens (')
        assert 'jobs' not in error.reason + error.repair_action
        assert not site.entries and not site.registered
        received = sorted(next(iter(site.uploads.values()))['parts'])
        assert received == ([0, 1, 2] if failure != 'a 500 on the whole-file check' else list(range(total)))
        site.calls.clear()
        result = lens_transfer.push(client, root, lens_id)
    sent = [int(path.rsplit('/', 1)[1]) for path in site.sent('PUT', '/parts/')]
    assert sent == [index for index in range(total) if index not in received]
    assert result['uploadedParts'] == len(sent) and result['alreadyHeld'] is False
    assert site.entries[result['convertedSHA256']] == converted(root, lens_id).read_bytes()
    assert lens_id in site.registered
    if not sent:
        assert lens_transfer.push_message(result).startswith(
            f'The site had already received every part of lens {lens_id} and now holds it')
    assert lens_transfer.interrupted_reason(FIXTURE['interrupted']['lensID'], FIXTURE['interrupted']['cause']) \
        == FIXTURE['interrupted']['reason']


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
    assert len(FIXTURE['pushMessages']) == 6


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
    # A push packages nothing, so its repair does not talk about packaging a request.
    assert caught.value.repair_action == FIXTURE['unavailable']['pushRepairAction']
    assert '--lens-reference' not in caught.value.repair_action
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
    # An interrupted push: a failure (70), with the resume sentence and no talk of jobs.
    broken = Site(part_max=100)
    broken.fail_part = 1
    wire(broken)
    code, document = cli(capsys, '--root', str(root), 'runner', 'lens-push', lens_id, '--runner', SERVER)
    assert (code, document['state'], document['error']['code']) == (70, 'failed', 'lensUploadInterrupted')
    assert document['error']['repairAction'] == FIXTURE['interrupted']['repairAction']


# --- the engine's own refusals -------------------------------------------------


def test_a_lens_library_refusal_is_reported_as_the_engine_gave_it(tmp_path, wire, capsys):
    """Its code, its sentence, and its repair, for every 4xx but 401 and 403:
    what the Mac reports too."""
    root, lens_id = make_lens(tmp_path)
    sha = lens_transfer.lens_record(root, lens_id)['convertedSHA256']

    def answering(response):
        def handler(request):
            request.read()
            if request.url.path == '/api/capabilities':
                return httpx.Response(200, json={'lensLibrary': {'available': True, 'partMaxBytes': 100}})
            return response
        return client_for(handler)

    shaped = refusal(409, 'lensUploadPlanDiffers', 'Another plan is in progress for this lens.',
                     'Wait for it to finish, then send the lens again.')
    with answering(shaped) as client, pytest.raises(runner_api.RunnerRefusal) as caught:
        client.lens_library_begin(sha, 1, [])
    assert (caught.value.code, caught.value.state) == ('lensUploadPlanDiffers', 'refused')
    assert caught.value.reason == 'Another plan is in progress for this lens.'
    assert caught.value.repair_action == 'Wait for it to finish, then send the lens again.'
    assert caught.value.detail['status'] == 409

    # The science routes' shape, with `reason` and no `message`.
    older = httpx.Response(409, json={'detail': {'code': 'lensRecordDiffers', 'reason': 'The site has another record.',
                                                 'repairAction': 'Compare the two records.'}})
    with answering(older) as client, pytest.raises(runner_api.RunnerRefusal) as caught:
        client.lens_library_register({'lensID': lens_id})
    assert (caught.value.code, caught.value.reason, caught.value.repair_action) == (
        'lensRecordDiffers', 'The site has another record.', 'Compare the two records.')

    # A 404 and a 413 are the engine's too: no talk of jobs, and no lens advice for a lens upload.
    for status in (404, 413):
        answer = refusal(status, 'lensAdoptMissing', 'No file is at that path.', 'Copy the file there first.')
        with answering(answer) as client, pytest.raises(runner_api.RunnerRefusal) as caught:
            client.lens_library_adopt('/nowhere', sha)
        assert (caught.value.code, caught.value.repair_action) == ('lensAdoptMissing', 'Copy the file there first.')
    # A bare sentence still arrives as the reason.
    with answering(httpx.Response(413, json={'detail': 'the body is too large'})) as client, \
            pytest.raises(runner_api.RunnerRefusal) as caught:
        client.lens_library_put_part(sha, {'index': 0, 'offset': 0, 'bytes': 8, 'sha256': hashlib.sha256(
            converted(root, lens_id).read_bytes()[:8]).hexdigest()}, str(converted(root, lens_id)))
    assert (caught.value.code, caught.value.reason) == ('lensLibraryRefused', 'the body is too large')
    assert 'lens-push' not in caught.value.repair_action and 'smaller bundle' not in caught.value.repair_action
    # 401 and 403 keep this client's own handling: the token's.
    with answering(httpx.Response(401, json={'detail': 'no token'})) as client, \
            pytest.raises(runner_api.RunnerHTTPError) as caught:
        client.lens_library_status(sha)
    assert caught.value.code == 'runnerUnauthorized'
    # Other routes are untouched: a 413 there has the old sentence, with no lens advice.
    with answering(httpx.Response(413, json={'detail': 'too large'})) as client, \
            pytest.raises(runner_api.RunnerHTTPError) as caught:
        client.stage_diagnostic('/staged/x.tar.gz', 'a' * 64)
    assert caught.value.repair_action == ('the runner refused the body as oversize — raise its '
                                          'STEERLAB_MAX_UPLOAD_BYTES, or package a smaller bundle. Nothing was staged.')

    # On the command line the envelope carries the engine's code.
    site = Site(part_max=100)
    wire(site)
    with site.client() as client:
        lens_transfer.push(client, root, lens_id)       # the site now holds an entry...
    site.entries.clear()                                # ...and loses it, leaving another plan in progress
    site.uploads[sha] = {'manifest': {'bytes': 1, 'parts': []}, 'parts': {}}
    code, document = cli(capsys, '--root', str(root), 'runner', 'lens-push', lens_id, '--runner', SERVER)
    assert (code, document['state'], document['error']['code']) == (65, 'refused', 'lensUploadPlanDiffers')
    assert document['error']['reason'] == 'Another plan is in progress for this lens.'
    assert document['error']['repairAction'] == 'Wait for it to finish, then send the lens again.'


def test_an_entry_of_another_size_and_an_answer_the_contract_does_not_allow(tmp_path):
    root, lens_id = make_lens(tmp_path)
    data = converted(root, lens_id).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    site = Site(part_max=100)
    site.entries[sha] = data + b'!'
    with site.client() as client, pytest.raises(runner_api.RunnerRefusal) as caught:
        lens_transfer.push(client, root, lens_id)
    assert caught.value.code == FIXTURE['entryDiffers']['code'] == 'lensLibraryEntryDiffers'
    assert caught.value.reason == lens_transfer.entry_differs_reason(len(data) + 1, len(data))
    assert caught.value.repair_action == FIXTURE['entryDiffers']['repairAction']
    assert not site.sent('POST', '') and not site.sent('PUT', '')
    assert lens_transfer.entry_differs_reason(FIXTURE['entryDiffers']['heldBytes'], FIXTURE['entryDiffers']['workspaceBytes']) \
        == FIXTURE['entryDiffers']['reason']

    cases = {case['kind']: case for case in FIXTURE['protocol']['cases']}
    assert set(cases) == {'partLimit', 'missingParts', 'notConfirmed', 'otherLens'}
    broken = {'partLimit': dict(part_max=0),
              'missingParts': dict(answers={'begin': {'held': False, 'missingParts': [99]}}),
              'notConfirmed': dict(answers={'complete': {'held': False, 'bytes': None}}),
              'otherLens': dict(answers={'register': {'lensID': 'another-lens', 'registered': True, 'changed': True}})}
    for kind, change in broken.items():
        site = Site(part_max=change.get('part_max', 100))
        site.answers = change.get('answers', {})
        with site.client() as client, pytest.raises(runner_api.RunnerError) as caught:
            lens_transfer.push(client, root, lens_id)
        # A failure (70), not a refusal: nothing declined, the engine misbehaved.
        assert (caught.value.code, caught.value.state) == (FIXTURE['protocol']['code'], 'failed'), kind
        assert caught.value.reason == cases[kind]['reason'], kind
        assert caught.value.repair_action == cases[kind]['repairAction'], kind


def test_the_twin_verb_purposes_are_the_fixtures(tmp_path):
    purposes = FIXTURE['purposes']
    assert science_commands.LENS_REFERENCE_PURPOSE == purposes['lensReference']
    assert science_commands.LENS_COPY_PURPOSE == purposes['lensCopy']
    assert science_commands.LENS_PUSH_PURPOSE == purposes['lensPush']
    assert science_commands.LENS_STATUS_PURPOSE == purposes['lensStatus']
    assert science_commands.ARTIFACT_IMPORT_PURPOSE == purposes['artifactImport']
    spec = client_cli.spec_for
    assert spec('runner', 'lens-push').purpose == purposes['lensPush']
    assert spec('runner', 'lens-status').purpose == purposes['lensStatus']
    assert spec('science', 'lens-copy').purpose == purposes['lensCopy']
    assert spec('science', 'artifact-import').purpose == purposes['artifactImport']
    for verb in ('input-plan', 'package'):
        assert spec('science', verb).purpose.endswith(purposes['lensReference'])
    for constant, key in ((lens_transfer.CHANGED_CODE, 'code'), (lens_transfer.CHANGED_REASON, 'reason'),
                          (lens_transfer.CHANGED_REPAIR, 'repairAction')):
        assert constant == FIXTURE['changed'][key]
    for case in FIXTURE['importMessages']:
        assert lens_transfer.import_message({'kind': 'jlens', 'existing': True, **case}) == case['message']
    assert lens_transfer.import_message({'kind': 'jlens', 'lensID': 'x', 'changed': True}) is None
    assert lens_transfer.import_message({'kind': 'sae-decoder', 'existing': True}) is None


# --- a record with a number that is not finite ---------------------------------


def with_numbers_that_are_not_finite(root, lens_id):
    """A qualification stores every check's measured values, and a check that
    could not be scored records NaN."""
    record_file = lens_folder(root, lens_id) / 'lens.json'
    record = json.loads(record_file.read_text())
    record['qualifications'] = [{'qualificationID': 'q1', 'modelID': 'example/model', 'revision': 'r', 'dtype': 'float32',
                                 'checks': {'causal': {'score': float('nan'), 'margin': float('inf'), 'seed': 2 ** 64 - 1}}}]
    record_file.write_text(json.dumps(record))
    assert 'NaN' in record_file.read_text() and 'Infinity' in record_file.read_text()
    return record


def test_a_number_that_is_not_finite_never_breaks_a_push_or_the_bridge(tmp_path, monkeypatch, capsys):
    root, lens_id = make_lens(tmp_path)
    raw = with_numbers_that_are_not_finite(root, lens_id)
    for described in (lens_transfer.lens_record(root, lens_id), lens_transfer.push_plan(root, lens_id, 100)):
        assert described['nonFiniteNumbers'] == 2
        checks = described['record']['qualifications'][0]['checks']['causal']
        assert checks == {'score': None, 'margin': None, 'seed': 2 ** 64 - 1}
        assert strict_loads(described['recordJSON']) == described['record']
        strict_loads(json.dumps(described))                       # the whole answer is strict JSON

    site = Site(part_max=100)
    with site.client() as client:
        result = lens_transfer.push(client, root, lens_id)        # parts, the whole-file check, and the record
        assert result['recordNonFiniteNumbers'] == 2 and result['registered'] is True
        assert site.registered[lens_id]['qualifications'][0]['checks']['causal']['score'] is None
        assert lens_transfer.push_message(result).endswith(
            '; the site has its record, with 2 numbers in the record that are not finite stored as null.')
        # The route method itself never raises on such a record, whoever calls it.
        assert client.lens_library_register(raw)['lensID'] == lens_id

    # The bridge prints strict JSON for every action, so the Mac can read the answer.
    def bridge_text(action, payload):
        document = {'action': action, 'payload': payload, 'clientSHA256': diagnostic_workspace.source_sha256()}
        monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(document)))
        capsys.readouterr()
        assert diagnostic_workspace.main() == 0
        return capsys.readouterr().out
    base = {'workspaceRoot': str(root), 'lensID': lens_id}
    for action, payload in (('lens-record', base), ('lens-push-plan', {**base, 'partMaxBytes': 100})):
        answer = strict_loads(bridge_text(action, payload))
        assert answer['ok'] is True and answer['result']['nonFiniteNumbers'] == 2
    # A repeated import answers with the lens's record, qualifications and all.
    description = tmp_path / 'workspace-source' / 'import.json'
    plan = imports.inspect_source(description, root)
    found = imports.publish(description, root, plan['planSHA256'])
    assert found['existing'] is True and found['record']['qualifications'][0]['checks']['causal'] == {
        'score': None, 'margin': None, 'seed': 2 ** 64 - 1}
    answer = strict_loads(bridge_text('artifact-import', {'workspaceRoot': str(root), 'descriptionFile': str(description),
                                                          'planSHA256': plan['planSHA256']}))
    assert answer['result']['existing'] is True
    assert answer['result']['record']['qualifications'][0]['checks']['causal']['score'] is None


def test_the_bridge_prints_strict_json_whatever_an_action_answers(monkeypatch, capsys):
    """Not only the lens actions: any answer with a number that is not finite
    reaches the Mac as null, where a bare NaN would read as no answer at all."""
    monkeypatch.setattr(diagnostic_commands, 'bridge_action', lambda action, payload: {
        'changed': False, 'values': [1.5, float('nan'), {'deep': float('-inf')}], 'count': 3})
    document = {'action': 'custody', 'payload': {}, 'clientSHA256': diagnostic_workspace.source_sha256()}
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(document)))
    capsys.readouterr()
    assert diagnostic_workspace.main() == 0
    answer = strict_loads(capsys.readouterr().out)
    assert answer['ok'] is True and answer['result'] == {'changed': False, 'values': [1.5, None, {'deep': None}], 'count': 3}


# --- a lens file that is not the one that was planned ---------------------------


def test_a_lens_file_that_changes_after_the_plan_stops_the_upload(tmp_path, monkeypatch):
    """Python hashes each part as it streams it. A part that is not the plan's
    is stopped before its last block leaves, with the Mac's sentence."""
    root, lens_id = make_lens(tmp_path)
    file = converted(root, lens_id)
    good = file.read_bytes()
    total = len(lens_transfer.push_plan(root, lens_id, 100)['parts'])
    for change in ('altered', 'shortened'):
        file.write_bytes(good)
        site = Site(part_max=100)
        with site.client() as client:
            begin = client.lens_library_begin

            def then_change(sha, size, parts):
                answer = begin(sha, size, parts)
                file.write_bytes(good[:-1] + bytes([good[-1] ^ 1]) if change == 'altered' else good[:-10])
                return answer
            monkeypatch.setattr(client, 'lens_library_begin', then_change)
            with pytest.raises(lens_transfer.LensRefusal) as caught:
                lens_transfer.push(client, root, lens_id)
        assert (caught.value.code, str(caught.value), caught.value.repair_action) == (
            FIXTURE['changed']['code'], FIXTURE['changed']['reason'], FIXTURE['changed']['repairAction'])
        # Every part before the changed one (the last) arrived; that one never did, and nothing was registered.
        assert sorted(next(iter(site.uploads.values()))['parts']) == list(range(total - 1))
        assert not site.entries and not site.registered and not site.sent('POST', 'complete')


def test_a_part_is_read_a_block_at_a_time_and_never_beyond_itself(tmp_path, monkeypatch):
    """Never the whole lens, and never the whole part: every read is at most
    one block, and stays inside the part being sent. The last block of a part
    leaves only after the whole part was hashed."""
    root, lens_id = make_lens(tmp_path)
    plan = lens_transfer.push_plan(root, lens_id, 100)
    reads, real_open = [], open

    class Recording:
        def __init__(self, handle):
            self.handle, self.start, self.position = handle, None, 0

        def seek(self, offset):
            self.start = self.position = offset
            return self.handle.seek(offset)

        def read(self, size=-1):
            data = self.handle.read(size)
            reads.append((self.start, self.position, size, len(data)))
            self.position += len(data)
            return data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.handle.close()

    monkeypatch.setattr(runner_api, 'open', lambda path, mode='r': Recording(real_open(path, mode)), raising=False)
    monkeypatch.setattr(runner_api, 'UPLOAD_CHUNK_BYTES', 16)
    site = Site(part_max=100)
    with site.client() as client:
        lens_transfer.push(client, root, lens_id)
    by_start = {part['offset']: part for part in plan['parts']}
    assert {start for start, *_ in reads} == set(by_start), 'each part is read from its own first byte'
    for start, position, asked, got in reads:
        part = by_start[start]
        assert 0 < asked <= 16, 'never more than one block at once'
        assert part['offset'] <= position and position + asked <= part['offset'] + part['bytes'], 'never beyond the part'
        assert got == asked
    assert sum(got for *_, got in reads) == plan['bytes'], 'every byte was read once'
    assert max(part['bytes'] for part in plan['parts']) > 16, 'a part longer than a block was sent'

    # The last block is held back until the part is known to be the plan's.
    part = max(plan['parts'], key=lambda item: item['bytes'])
    seen = []

    class Collecting(httpx.BaseTransport):
        """Reads the body as it arrives, so what left before a stop is seen."""

        def handle_request(self, request):
            for chunk in request.stream:
                seen.append(chunk)
            return httpx.Response(200, json={'received': True})
    wrong = {**part, 'sha256': '0' * 64}
    client = runner_api.RunnerClient(base_url=SERVER, http_client=httpx.Client(transport=Collecting(), base_url=SERVER))
    with client, pytest.raises(lens_transfer.LensRefusal):
        client.lens_library_put_part(plan['convertedSHA256'], wrong, str(converted(root, lens_id)))
    last = part['bytes'] % 16 or 16
    assert sum(len(chunk) for chunk in seen) == part['bytes'] - last, 'the last block never left'


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
            assert caught.value.repair_action == FIXTURE['unavailable']['archiveRepairAction']
            assert caught.value.detail['libraryLenses'] == lenses
            # A request that carries its lens goes to any engine, as before.
            client.upload_run_bundle(str(carrying))
            assert site.sent('POST', '/api/bundles/upload')
    site = Site()
    with site.client() as client:
        client.upload_run_bundle(str(referring))
    assert site.sent('POST', '/api/bundles/upload')


def test_the_capability_is_asked_first_and_the_archive_is_opened_only_without_a_library(tmp_path, monkeypatch):
    """An up-to-date engine costs an upload nothing extra: one capability
    read, and the archive is never opened to look for a lens reference."""
    carrying = request_archive(tmp_path, 'carrying.tar.gz', {'request': {}})
    opened = []
    inspect = lens_transfer.archive_library_lenses
    monkeypatch.setattr(lens_transfer, 'archive_library_lenses', lambda path: opened.append(path) or inspect(path))
    for library, expected in ((True, 0), (False, 1), (None, 1)):
        opened.clear()
        site = Site(library=library)
        with site.client() as client:
            client.upload_run_bundle(str(carrying))
        assert len(opened) == expected, library
        assert site.calls == [('GET', '/api/capabilities'), ('POST', '/api/bundles/upload')]
        # The engine inspects the archive before it answers: that wait gets an hour.
        assert site.timeouts['/api/bundles/upload']['read'] == 3600
        assert site.timeouts['/api/capabilities']['read'] != 3600, 'the hour is that one request\'s'
    # A timeout the caller chose still decides.
    site = Site()
    with site.client(timeout=5) as client:
        client.upload_run_bundle(str(carrying))
    assert site.timeouts['/api/bundles/upload']['read'] == 5


def test_an_upload_over_the_limit_says_a_lens_can_be_referred_to(tmp_path):
    carrying = request_archive(tmp_path, 'carrying.tar.gz', {'request': {}})

    def handler(request):
        if request.url.path == '/api/capabilities':
            return httpx.Response(200, json={})
        request.read()
        return httpx.Response(413, json={'detail': 'upload exceeds 4294967296 bytes (STEERLAB_MAX_UPLOAD_BYTES)'})
    with client_for(handler) as client, pytest.raises(runner_api.RunnerHTTPError) as caught:
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


def second_workspace(tmp_path, name='second'):
    root = tmp_path / name
    root.mkdir()
    return root.resolve()


def tree(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes() for path in sorted(directory.rglob('*')) if path.is_file()}


def stamps(directory):
    return {path.relative_to(directory).as_posix(): (path.stat().st_mtime_ns, path.read_bytes())
            for path in sorted(directory.rglob('*')) if path.is_file()}


def copied(root, lens_id, origin):
    preview = lens_transfer.copy(root, lens_id, str(origin))
    return lens_transfer.copy(root, lens_id, str(origin), preview['planSHA256'])


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
    assert tree(lens_folder(root, lens_id)) == tree(lens_folder(origin, lens_id))
    assert result['outputDirectory'] == str(lens_folder(root, lens_id))
    # The copy is a working lens here: its plan is the origin's, byte for byte.
    assert lens_transfer.push_plan(root, lens_id, 1 << 20)['parts'] == lens_transfer.push_plan(origin, lens_id, 1 << 20)['parts']


def test_a_copy_that_cannot_clone_is_a_plain_copy(tmp_path, monkeypatch):
    origin, lens_id = make_lens(tmp_path, name='first')
    root = second_workspace(tmp_path)
    monkeypatch.setattr(lens_transfer, '_clone', lambda source, target: False)
    result = copied(root, lens_id, origin)
    assert result['copyMethod'] == 'copy' and result['changed'] is True
    assert tree(lens_folder(root, lens_id)) == tree(lens_folder(origin, lens_id))


def test_the_same_lens_already_here_changes_nothing_and_a_different_one_is_refused(tmp_path):
    origin, lens_id = make_lens(tmp_path, name='first')
    root = second_workspace(tmp_path)
    preview = lens_transfer.copy(root, lens_id, str(origin))
    lens_transfer.copy(root, lens_id, str(origin), preview['planSHA256'])
    before = stamps(root / 'runs')
    again = lens_transfer.copy(root, lens_id, str(origin))
    assert again['action'] == 'none' and again['existing'] is True and again['changed'] is False
    applied = lens_transfer.copy(root, lens_id, str(origin), again['planSHA256'])
    assert applied['existing'] is True and applied['changed'] is False and 'copyMethod' not in applied
    assert stamps(root / 'runs') == before
    # The first review's digest no longer describes this workspace.
    with pytest.raises(lens_transfer.LensRefusal, match='changed after the copy was reviewed'):
        lens_transfer.copy(root, lens_id, str(origin), preview['planSHA256'])

    # A different lens under that name: different bytes, same folder name.
    other, other_id = make_lens(tmp_path, name='third', seed=7)
    lens_folder(other, other_id).rename(lens_folder(other, lens_id))
    record_file = lens_folder(other, lens_id) / 'lens.json'
    record = json.loads(record_file.read_text())
    record['lensID'] = lens_id
    record['converted']['path'] = f'runs/jlens-lenses/{lens_id}/jacobians.safetensors'
    record_file.write_text(json.dumps(record))
    with pytest.raises(lens_transfer.LensRefusal, match='already holds a different lens') as caught:
        lens_transfer.copy(root, lens_id, str(other))
    assert caught.value.code == 'lensNameTaken'
    assert stamps(root / 'runs') == before


def test_a_lens_already_here_is_checked_and_its_missing_file_is_put_back(tmp_path):
    """The destination's lens is not taken on trust: its converted file is
    hashed. Missing, that one file is put back; different, the copy is refused
    and the file is left alone."""
    origin, lens_id = make_lens(tmp_path, name='first')
    root = second_workspace(tmp_path)
    copied(root, lens_id, origin)
    file = converted(root, lens_id)
    good = file.read_bytes()

    file.unlink()
    before = stamps(lens_folder(root, lens_id))
    preview = lens_transfer.copy(root, lens_id, str(origin))
    assert (preview['action'], preview['existing'], preview['changed']) == ('restore', True, False)
    assert not file.exists(), 'a preview writes nothing'
    assert lens_transfer.copy_message(preview) == (
        f'Lens {lens_id} is already in this workspace, but its converted file is missing; nothing was written. '
        f"Repeat the command with --plan-sha256 {preview['planSHA256']} to put that file back.")
    restored = lens_transfer.copy(root, lens_id, str(origin), preview['planSHA256'])
    assert (restored['restored'], restored['existing'], restored['changed']) == (True, True, True)
    assert restored['copyMethod'] in ('clone', 'copy') and file.read_bytes() == good
    after = stamps(lens_folder(root, lens_id))
    assert {name: stamp for name, stamp in after.items() if name != 'jacobians.safetensors'} == before, \
        'nothing else in the folder was touched'
    assert lens_transfer.copy_message(restored).startswith(
        f'Lens {lens_id} was already in this workspace; its missing converted file was put back')
    assert lens_transfer.push_plan(root, lens_id, 1 << 20)['bytes'] == len(good), 'the lens works again'
    assert lens_transfer.copy(root, lens_id, str(origin))['action'] == 'none'

    # A different file under the lens's name is never taken for the lens, and never replaced.
    altered = good[:-1] + bytes([good[-1] ^ 1])
    file.write_bytes(altered)
    with pytest.raises(lens_transfer.LensRefusal, match='differs from the SHA-256 its record names') as caught:
        lens_transfer.copy(root, lens_id, str(origin))
    assert caught.value.code == 'lensFileDiffers'
    assert caught.value.repair_action == ('Move that file aside, then run lens-copy again: it puts back the file '
                                          'from the other workspace.')
    assert file.read_bytes() == altered
    # …and the repair works as written.
    file.rename(file.with_name('jacobians.safetensors.aside'))
    assert copied(root, lens_id, origin)['restored'] is True and file.read_bytes() == good

    # Without cloning, the file is put back as a copy.
    file.unlink()
    import unittest.mock
    with unittest.mock.patch.object(lens_transfer, '_clone', lambda source, target: False):
        assert copied(root, lens_id, origin)['copyMethod'] == 'copy'
    assert file.read_bytes() == good


def test_the_same_lens_under_another_name_is_not_copied_again(tmp_path):
    """One lens, one name: an older import's random name in the other
    workspace, and the same files under the content-derived name here."""
    origin, derived = make_lens(tmp_path, name='first')
    old_name = 'custom-lens-' + '5e' * 16
    lens_folder(origin, derived).rename(lens_folder(origin, old_name))
    record_file = lens_folder(origin, old_name) / 'lens.json'
    record = json.loads(record_file.read_text())
    record['lensID'] = old_name
    record['converted']['path'] = f'runs/jlens-lenses/{old_name}/jacobians.safetensors'
    record_file.write_text(json.dumps(record))

    root = second_workspace(tmp_path)
    description = tmp_path / 'first-source' / 'import.json'
    plan = imports.inspect_source(description, root)
    assert imports.publish(description, root, plan['planSHA256'])['lensID'] == derived
    before = stamps(root / 'runs')
    for expected in (None, 'reviewed'):
        result = lens_transfer.copy(root, old_name, str(origin), expected and lens_transfer.copy(root, old_name, str(origin))['planSHA256'])
        assert (result['existing'], result['changed'], result['action']) == (True, False, 'none')
        assert result['existingLensID'] == derived
        assert lens_transfer.copy_message(result) == f'This workspace already holds this lens as {derived}; nothing was copied.'
    assert stamps(root / 'runs') == before and not lens_folder(root, old_name).exists()

    # Where a receipt is missing, the records' three hashes decide.
    (lens_folder(root, derived) / 'import-receipt.json').unlink()
    assert lens_transfer.copy(root, old_name, str(origin))['existingLensID'] == derived
    # A workspace that holds a different lens copies this one.
    other, _ = make_lens(tmp_path, name='third', seed=7)
    assert lens_transfer.copy(other, old_name, str(origin))['action'] == 'copy'
    for case in FIXTURE['copyMessages']:
        result = {key: value for key, value in case.items() if value is not None}
        assert lens_transfer.copy_message({**result, 'files': [None] * case['fileCount']}) == case['message']
    assert len(FIXTURE['copyMessages']) == 8


def test_a_copy_checks_every_file_before_and_after(tmp_path, monkeypatch):
    origin, lens_id = make_lens(tmp_path, name='first')
    root = second_workspace(tmp_path)
    # Before: bytes that differ from the import are not copied.
    file = converted(origin, lens_id)
    good = file.read_bytes()
    file.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
    with pytest.raises(lens_transfer.LensRefusal, match='in that workspace .* differs from the SHA-256 its record names'):
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
    assert not lens_folder(root, lens_id).exists()
    assert not list((root / 'runs/jlens-lenses').iterdir()), 'no staging folder is left behind'

    with pytest.raises(lens_transfer.LensRefusal, match='names this workspace'):
        lens_transfer.copy(origin, lens_id, str(origin))
    with pytest.raises(lens_transfer.LensRefusal, match='no workspace folder'):
        lens_transfer.copy(root, lens_id, str(tmp_path / 'nowhere'))
    # A lens the OTHER workspace does not hold: the refusal names that workspace.
    with pytest.raises(lens_transfer.LensRefusal) as caught:
        lens_transfer.copy(root, 'absent', str(origin))
    assert str(caught.value) == f'The workspace at {origin} holds no lens named absent (no runs/jlens-lenses/absent/lens.json).'
    assert caught.value.code == 'lensNotFound' and 'that workspace' in caught.value.repair_action
    # A destination that does not exist is refused, never made.
    nowhere = tmp_path / 'not-a-workspace'
    for expected in (None, preview['planSHA256']):
        with pytest.raises(lens_transfer.LensRefusal, match='no workspace folder') as caught:
            lens_transfer.copy(nowhere, lens_id, str(origin), expected)
        assert caught.value.code == 'workspaceNotFound' and not nowhere.exists()


def test_the_command_line_reviews_then_copies(tmp_path, capsys, monkeypatch):
    origin, lens_id = make_lens(tmp_path, name='first')
    root = second_workspace(tmp_path)
    monkeypatch.setenv('STEERLAB_ROOT', '')
    monkeypatch.delenv(client_cli.WORKSPACE_ENV, raising=False)
    code, document = cli(capsys, '--root', str(root), 'science', 'lens-copy', lens_id, '--from', str(origin))
    assert code == 0 and document['changed'] is False
    digest = document['result']['planSHA256']
    assert document['message'].endswith(f'Repeat the command with --plan-sha256 {digest} to copy it.')
    code, document = cli(capsys, '--root', str(root), 'science', 'lens-copy', lens_id, '--from', str(origin),
                         '--plan-sha256', digest)
    assert code == 0 and document['changed'] is True and document['result']['copyMethod'] in ('clone', 'copy')
    assert (lens_folder(root, lens_id) / 'lens.json').is_file()
    code, document = cli(capsys, '--root', str(root), 'science', 'lens-copy', lens_id, '--from', str(origin))
    assert code == 0 and document['message'] == f'This workspace already holds lens {lens_id}; nothing was copied.'
    code, document = cli(capsys, '--root', str(root), 'science', 'lens-copy', 'absent', '--from', str(origin))
    assert (code, document['state'], document['error']['code']) == (65, 'refused', 'lensNotFound')

    # The verb says what is missing, in its own words.
    usage = 'steerlab --root <workspace directory> science lens-copy <lensID> --from <workspace directory>'
    code, document = cli(capsys, '--root', str(root), 'science', 'lens-copy', lens_id)
    assert (code, document['error']['code'], document['error']['repairAction']) == (64, 'usage', usage)
    assert document['error']['reason'] == FIXTURE['copyArguments']['from'] == (
        'science lens-copy needs --from <workspace directory>: the workspace that holds the lens.')
    code, document = cli(capsys, '--root', str(root), 'science', 'lens-copy', '--from', str(origin))
    assert (code, document['error']['code']) == (64, 'usage')
    assert document['error']['reason'] == FIXTURE['copyArguments']['lensID']
    assert document['error']['reason'].startswith('science lens-copy takes one lens ID')
    # With no workspace named it refuses, as lens-push does: the current folder is never written into.
    elsewhere = tmp_path / 'not-a-workspace'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    code, document = cli(capsys, 'science', 'lens-copy', lens_id, '--from', str(origin))
    assert (code, document['error']['code'], document['error']['repairAction']) == (64, 'usage', usage)
    assert document['error']['reason'] == 'science lens-copy copies the lens into a workspace, and none was named.'
    assert not list(elsewhere.iterdir())


# --- the bridge the Mac uses, and the route it must never reach ------------------


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
    assert (answer['ok'], answer['state']) == (False, 'blocked') and not_the_archive_familys(answer['repairAction'])
    # `workspace_action` is what the workbench's HTTP route serves: a caller
    # there must not be able to name a second folder on the machine.
    for action in diagnostic_commands.LOCAL_ACTIONS:
        with pytest.raises(archives.MalformedRequest, match='Unknown diagnostic workspace operation'):
            diagnostic_commands.workspace_action(action, {'workspaceRoot': str(root), 'lensID': lens_id})


def test_the_workbench_route_itself_refuses_the_lens_actions(tmp_path, monkeypatch):
    """Through the real route, not the function behind it: whatever the route
    is wired to, a caller of the served workspace cannot copy a lens out of
    another folder on the machine, or read one."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from steerlab_server.api.diagnostic_transport_routes import build_router
    origin, lens_id = make_lens(tmp_path, name='first')
    root = second_workspace(tmp_path)
    monkeypatch.setenv('STEERLAB_ROOT', str(root))
    app = FastAPI()
    app.include_router(build_router(SimpleNamespace(jobs=None)))
    client = TestClient(app)
    payloads = {'lens-copy': {'workspaceRoot': str(root), 'lensID': lens_id, 'fromWorkspace': str(origin)},
                'lens-record': {'workspaceRoot': str(root), 'lensID': lens_id},
                'lens-push-plan': {'workspaceRoot': str(root), 'lensID': lens_id, 'partMaxBytes': 100}}
    assert set(payloads) == set(diagnostic_commands.LOCAL_ACTIONS)
    for action, payload in payloads.items():
        response = client.post(f'/api/science/workspace/{action}', json=payload)
        assert response.status_code == 409, action
        assert response.json()['detail']['reason'] == 'Unknown diagnostic workspace operation.', action
    assert not (root / 'runs').exists(), 'nothing was copied'
    # The same request through the local bridge's dispatch does copy: the route is what refuses.
    assert diagnostic_commands.bridge_action('lens-copy', payloads['lens-copy'])['action'] == 'copy'
