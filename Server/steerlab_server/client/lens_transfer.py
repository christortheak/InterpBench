"""Send a workspace lens to a site's lens library once, ask what a site holds,
and copy a lens between workspaces on one machine. No model or GPU imports.

The workspace stays the truth. A site's lens library is a cache of converted
lens bytes filed under their SHA-256 (``converted.sha256``): anything in it
can be sent again from a workspace, and losing an entry loses no record.

Three bridge actions live here, reached by the Mac through
``diagnostic_workspace`` and by this client's own verbs:

* ``lens-push-plan``: verify a lens's converted file and cut it into parts
  that follow its layers, each with its own hash;
* ``lens-record``: the same lens's record and hashes, read without hashing
  the file (``lens-status``, and ``lens-push --adopt``);
* ``lens-copy``: the same lens, under the same name, in a second workspace.

The sentences both clients say about an engine with no lens library are here
too. Their Swift twins are in ``LensLibrary.swift``, and the two are held to
the same bytes by ``prompts/fixtures/lens-library/refusals.json``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile

from ..experiment import diagnostic_archives as archives, paths

RECORD_NAME = 'lens.json'
CONVERTED_NAME = 'jacobians.safetensors'
_SHA256 = re.compile('[0-9a-f]{64}')
#: The safetensors format's own bound on its JSON header.
_HEADER_LIMIT = 100_000_000
_BLOCK = 1024 * 1024

#: What both clients answer when an engine has no lens library to use. The
#: engine's own refusal of a staged request carries the same code.
LIBRARY_UNAVAILABLE_CODE = 'lensLibraryUnavailable'
#: Swift twin: ``LensLibrary.unavailableRepair``.
LIBRARY_UNAVAILABLE_REPAIR = (
    "Package the request without --lens-reference so it carries its lens, or set the site profile's "
    'lensLibrary folder to one the engine can write and run the command again. An engine from before '
    'the lens library must be updated first.')
#: Added to the repair for an upload the engine refuses as too large. Swift
#: twin: ``LensLibrary.oversizeAdvice``.
OVERSIZE_LENS_ADVICE = (
    'A request that carries a lens can refer to it instead: send the lens once with lens-push, then '
    'package the request with --lens-reference.')
#: The site forbids HTTP transfer, so a lens's bytes reach it another way.
EXTERNAL_TRANSFER_CODE = 'externalTransferRequired'


def library_unavailable_reason(where, *, archive=False):
    """Why a client sends nothing to an engine with no usable lens library, in
    the same words on both clients (Swift twin ``LensLibrary.unavailableReason``).
    ``archive`` is the upload of a request that refers to its lens."""
    lead = ('This archive refers to its lens instead of carrying it, and the ' if archive else 'The ')
    return (lead + f'engine at {where} does not offer a lens library (its capabilities do not report '
            'lensLibrary.available as true). Nothing was sent.')


def external_transfer_reason(where):
    """Swift twin ``LensLibrary.externalTransferReason``."""
    return (f"The engine at {where} does not accept file transfers over HTTP, so this client cannot "
            "send the lens's bytes. Nothing was sent.")


def external_transfer_repair(file):
    """Swift twin ``LensLibrary.externalTransferRepair``."""
    return (f"Copy {file} to the site with the site's own transfer tool, then run lens-push again "
            'with --adopt <the path of that copy on the site>: the site checks the copy against the '
            "lens's hash and keeps it.")


def library_available(capabilities):
    """Whether a capability snapshot reports a lens library the engine can
    write. Absent on an engine from before the library. Swift twin
    ``ClusterCapabilities.lensLibraryAvailable``."""
    block = capabilities.get('lensLibrary') if isinstance(capabilities, dict) else None
    return isinstance(block, dict) and block.get('available') is True


class LensRefusal(archives.Refusal):
    """A lens this client will not send, describe, or copy as asked."""

    code = 'lensRefused'

    def __init__(self, message, repair_action, code=None):
        super().__init__(message)
        self.repair_action = repair_action
        if code is not None:
            self.code = code


def _lens_name(lens_id):
    if not isinstance(lens_id, str) or not lens_id or '/' in lens_id:
        raise archives.MalformedRequest('Name one lens by its lensID, as its lens.json records it.')
    archives.parts(lens_id)
    return lens_id


def canonical_path(lens_id):
    """Where an imported lens's converted file lives, relative to a workspace."""
    return f'runs/jlens-lenses/{lens_id}/{CONVERTED_NAME}'


def workspace_lens(root, lens_id):
    """``(root, record, file)`` for a lens in a workspace: its record as
    stored, and the workspace-relative path of its converted file. Nothing is
    hashed here."""
    root = Path(root).resolve()
    _lens_name(lens_id)
    directory = Path(paths.jlens_lens_directory(lens_id, str(root))).relative_to(root).as_posix()
    record_file = archives.ordinary(root, directory + '/' + RECORD_NAME, missing=True)
    if not record_file.is_file():
        raise LensRefusal(
            f'This workspace holds no lens named {lens_id} (no {directory}/{RECORD_NAME}).',
            'Name a lens this workspace holds (the folders under runs/jlens-lenses/), or name the '
            'workspace that holds it.', code='lensNotFound')
    try:
        record = json.loads(record_file.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise LensRefusal(f'The record of lens {lens_id} ({directory}/{RECORD_NAME}) is not readable JSON.',
                          'Restore the lens folder from a copy, or import the lens again.') from exc
    converted = record.get('converted') if isinstance(record, dict) else None
    if (not isinstance(record, dict) or record.get('lensID') != lens_id or not isinstance(converted, dict)
            or not isinstance(converted.get('path'), str) or not isinstance(converted.get('sha256'), str)
            or not _SHA256.fullmatch(converted['sha256'])):
        raise LensRefusal(
            f'The record of lens {lens_id} does not name this lens with a converted file and its SHA-256.',
            'Import the lens again from its source files; an import records both.')
    from ..jlens import artifact_paths
    selected = Path(artifact_paths.converted_file(lens_id, converted['path'], str(root)))
    if not selected.is_relative_to(root):
        raise LensRefusal(
            f"Lens {lens_id} keeps its converted file outside this workspace's lens library.",
            'Import the lens again in this workspace, so its converted file is in runs/jlens-lenses/.')
    relative = selected.relative_to(root).as_posix()
    if not archives.ordinary(root, relative, missing=True).is_file():
        raise LensRefusal(
            f'The converted file of lens {lens_id} is missing from this workspace ({relative}).',
            'Restore it from a copy, or import the lens again from its source files.')
    return root, record, relative


def _site_record(record, lens_id):
    """The record as a site receives it: the converted file is named by its
    place in a workspace, never by a path on this machine."""
    return {**record, 'converted': {**record['converted'], 'path': canonical_path(lens_id)}}


def _described(record, lens_id, relative, size):
    """What both lens actions say about a lens. ``recordJSON`` is the record
    again as exact JSON text, for the Mac: its bridge reads every number as a
    double, and a record's integers and decimals must reach the site as they
    were written."""
    site = _site_record(record, lens_id)
    source = record.get('source') if isinstance(record.get('source'), dict) else {}
    return {'lensID': lens_id, 'record': site, 'recordJSON': json.dumps(site, sort_keys=True),
            'convertedSHA256': record['converted']['sha256'], 'tensorSHA256': source.get('tensorSHA256'),
            'bytes': size, 'file': relative, 'changed': False}


def lens_record(root, lens_id):
    """The ``lens-record`` action: a lens's record and hashes, with the size of
    its converted file. The file is not hashed."""
    root, record, relative = workspace_lens(root, lens_id)
    return _described(record, lens_id, relative, (root / relative).stat().st_size)


def _part_limit(value):
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if type(value) is not int or value < 1:
        raise archives.MalformedRequest('partMaxBytes is a whole number of bytes, at least 1.')
    return value


def part_ranges(handle, size, part_max_bytes):
    """``(offset, bytes)`` for each part of a safetensors file, in order.

    The first part is the header, ``[0, 8 + header length)``. Each tensor then
    starts a part of its own, in file order, from the header's
    ``data_offsets``. A part longer than ``part_max_bytes`` is split. The
    parts tile ``[0, size)`` exactly, whatever the header says: a cut is only
    ever added, so bytes no tensor claims stay in the part before them."""
    unreadable = LensRefusal(
        'The converted lens file is not a readable safetensors file.',
        'Import the lens again from its source files; the import writes this file.')
    handle.seek(0)
    head = handle.read(8)
    length = int.from_bytes(head, 'little') if len(head) == 8 else 0
    if not 0 < length <= _HEADER_LIMIT or 8 + length > size:
        raise unreadable
    try:
        header = json.loads(handle.read(length))
    except (ValueError, UnicodeError) as exc:
        raise unreadable from exc
    if not isinstance(header, dict):
        raise unreadable
    start = 8 + length
    cuts = {0, start, size}
    for name, entry in header.items():
        if name == '__metadata__':
            continue
        offsets = entry.get('data_offsets') if isinstance(entry, dict) else None
        if (not isinstance(offsets, list) or len(offsets) != 2 or any(type(item) is not int for item in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= size - start):
            raise unreadable
        cuts.add(start + offsets[0])
    ordered = sorted(cuts)
    ranges = []
    for offset, stop in zip(ordered, ordered[1:]):
        while offset < stop:
            step = min(part_max_bytes, stop - offset)
            ranges.append((offset, step))
            offset += step
    return ranges


def push_plan(root, lens_id, part_max_bytes):
    """The ``lens-push-plan`` action. Reads the converted file once: every
    part is hashed, and the whole must equal the hash the import recorded, so
    a lens is only ever sent, or referred to, as bytes that were checked."""
    limit = _part_limit(part_max_bytes)
    root, record, relative = workspace_lens(root, lens_id)
    changed = LensRefusal('The converted lens file changed while it was being read.',
                          'Run the command again once nothing else is writing the lens folder.')
    whole = hashlib.sha256()
    parts = []
    with open(archives.ordinary(root, relative), 'rb') as handle:
        size = os.fstat(handle.fileno()).st_size
        ranges = part_ranges(handle, size, limit)
        handle.seek(0)
        for index, (offset, length) in enumerate(ranges):
            digest = hashlib.sha256()
            remaining = length
            while remaining:
                block = handle.read(min(_BLOCK, remaining))
                if not block:
                    raise changed
                digest.update(block)
                whole.update(block)
                remaining -= len(block)
            parts.append({'index': index, 'offset': offset, 'bytes': length, 'sha256': digest.hexdigest()})
        if handle.read(1):
            raise changed
    if whole.hexdigest() != record['converted']['sha256']:
        raise LensRefusal('Converted lens bytes differ from the imported hash; nothing was planned.',
                          'Restore the lens folder from a copy, or import the lens again from its source files.')
    return {**_described(record, lens_id, relative, size), 'parts': parts}


# --- a request archive that refers to its lens --------------------------------


def archive_library_lenses(path):
    """The lenses a diagnostic input archive refers to instead of carrying
    (its metadata's ``context.libraryLenses``), or None for any other file.

    An archive this cannot read is not judged here: the engine inspects every
    upload and says what is wrong with it."""
    try:
        with tarfile.open(path, 'r:gz') as archive:
            for member in archive:
                if member.name == archives.META:
                    if not member.isfile() or member.size > archives.MAX_METADATA:
                        return None
                    metadata = json.load(archive.extractfile(member))
                    break
            else:
                return None
    except (tarfile.TarError, OSError, EOFError, ValueError):
        return None
    context = metadata.get('context') if isinstance(metadata, dict) else None
    if not isinstance(context, dict) or context.get('lensReference') is not True:
        return None
    lenses = context.get('libraryLenses')
    return lenses if isinstance(lenses, list) else []


def require_library_for_archive(client, path):
    """Refuse, before anything is sent, to upload a request that refers to its
    lens to an engine with no lens library. Swift twin: the same check in
    ``ClusterClient.uploadBundle``."""
    lenses = archive_library_lenses(path)
    if lenses is None or library_available(client.capabilities()):
        return
    from .runner import RunnerRefusal
    raise RunnerRefusal(library_unavailable_reason(client.base_url, archive=True),
                        code=LIBRARY_UNAVAILABLE_CODE, repair_action=LIBRARY_UNAVAILABLE_REPAIR,
                        detail={'libraryLenses': lenses})


# --- sending a lens, and asking what a site holds -----------------------------


def _answer(document, what):
    from .runner import RunnerError
    if not isinstance(document, dict):
        raise RunnerError(f'The engine answered {what} with something other than an object.',
                          repair_action='Check that --runner names a SteerLab engine with a lens library.')
    return document


def push(client, root, lens_id, *, adopt=None):
    """Send one lens to the site's library and register its record there.

    In order: the capability, what the site holds, the part manifest, only the
    parts the site lacks, the whole-file check, the record. Run again after an
    interruption, it sends only what is still missing; when the site already
    holds the bytes it sends none and still registers the record. With
    ``adopt``, the bytes are a copy already on the site at that path, and the
    site checks it against the lens's hash."""
    from .runner import RunnerError, RunnerRefusal
    capabilities = client.capabilities()
    if not library_available(capabilities):
        raise RunnerRefusal(library_unavailable_reason(client.base_url), code=LIBRARY_UNAVAILABLE_CODE,
                            repair_action=LIBRARY_UNAVAILABLE_REPAIR)
    if adopt is not None:
        plan = lens_record(root, lens_id)
    else:
        try:
            limit = _part_limit(capabilities['lensLibrary'].get('partMaxBytes'))
        except archives.MalformedRequest as exc:
            raise RunnerError(f'The engine at {client.base_url} reports a lens library with no usable partMaxBytes.',
                              repair_action='Update the engine; its capabilities must state the largest part it accepts.') from exc
        plan = push_plan(root, lens_id, limit)
    sha = plan['convertedSHA256']
    status = _answer(client.lens_library_status(sha), 'the lens status')
    already = status.get('held') is True
    uploaded_parts = uploaded_bytes = 0
    adopted = False
    if already and type(status.get('bytes')) is int and status['bytes'] != plan['bytes']:
        raise RunnerRefusal(
            f"The site holds {status['bytes']} bytes under this lens's hash, and the workspace file is "
            f"{plan['bytes']} bytes. Nothing was sent.",
            code='lensLibraryEntryDiffers',
            repair_action="Ask the site's operator to inspect that library entry; the workspace copy is the "
                          'one its record describes.',
            detail={'convertedSHA256': sha})
    if not already and adopt is not None:
        _answer(client.lens_library_adopt(adopt, sha), 'the adoption')
        adopted = True
    elif not already:
        remote = capabilities.get('remoteStudy') or {}
        if remote.get('externalTransferRequired') or remote.get('httpTransfer') is False:
            raise RunnerRefusal(external_transfer_reason(client.base_url), code=EXTERNAL_TRANSFER_CODE,
                                repair_action=external_transfer_repair(plan['file']),
                                detail={'convertedSHA256': sha, 'bytes': plan['bytes']})
        begun = _answer(client.lens_library_begin(sha, plan['bytes'], plan['parts']), 'the upload manifest')
        if begun.get('held') is True:
            already = True
        else:
            missing = begun.get('missingParts')
            indices = {part['index'] for part in plan['parts']}
            if (not isinstance(missing, list) or any(type(item) is not int for item in missing)
                    or not set(missing) <= indices):
                raise RunnerError('The engine named parts to send that are not in the manifest it was given.',
                                  repair_action='Run the command again; if it repeats, update the engine.')
            wanted = set(missing)
            source = str(Path(root).resolve() / plan['file'])
            for part in plan['parts']:
                if part['index'] in wanted:
                    client.lens_library_put_part(sha, part, source)
                    uploaded_parts += 1
                    uploaded_bytes += part['bytes']
            done = _answer(client.lens_library_complete(sha), 'the completed upload')
            if done.get('held') is not True or done.get('bytes') != plan['bytes']:
                raise RunnerError('The engine did not confirm that it holds the whole lens after the upload.',
                                  repair_action='Run the command again: it sends only what the site still lacks.')
    registered = _answer(client.lens_library_register(plan['record']), 'the lens record')
    if registered.get('lensID') != lens_id:
        raise RunnerError('The engine registered a record under another lens name than the one sent.',
                          repair_action='Run lens-status for this lens, and update the engine if it repeats.')
    wrote = uploaded_parts > 0 or adopted or registered.get('changed') is True
    return {'lensID': lens_id, 'convertedSHA256': sha, 'bytes': plan['bytes'], 'held': True,
            'alreadyHeld': already, 'adopted': adopted, 'uploadedParts': uploaded_parts,
            'uploadedBytes': uploaded_bytes, 'registered': registered.get('registered'),
            'recordChanged': registered.get('changed'), 'changed': wrote}


def push_message(result):
    """One true sentence about a finished push. Swift twin
    ``LensLibrary.pushMessage``."""
    lens = result['lensID']
    if result['uploadedParts']:
        sent = (f"Sent lens {lens} to the site's lens library ({result['uploadedParts']} parts, "
                f"{result['uploadedBytes']} bytes)")
    elif result['adopted']:
        sent = f'The site checked its copy of lens {lens} and keeps it in its lens library'
    else:
        sent = f"The site's lens library already holds lens {lens}, so nothing was uploaded"
    return sent + '; the site has its record.'


def status(client, root, lens_id):
    """Whether the site holds a workspace lens's bytes, and whether it has the
    lens's record. Reads only."""
    from .runner import RunnerHTTPError
    local = lens_record(root, lens_id)
    sha = local['convertedSHA256']
    result = {'lensID': lens_id, 'convertedSHA256': sha, 'bytes': local['bytes'],
              'libraryAvailable': library_available(client.capabilities()), 'held': False,
              'receivedParts': None, 'siteRecord': 'unknown', 'changed': False}
    if result['libraryAvailable']:
        held = _answer(client.lens_library_status(sha), 'the lens status')
        result['held'] = held.get('held') is True
        result['receivedParts'] = held.get('receivedParts')
    try:
        site = client.lens_record(lens_id)
    except RunnerHTTPError as exc:
        if exc.status == 404:
            result['siteRecord'] = 'absent'
        elif exc.status != 403:
            raise
    else:
        converted = site.get('converted') if isinstance(site, dict) else None
        source = site.get('source') if isinstance(site, dict) else None
        same = (isinstance(converted, dict) and converted.get('sha256') == sha
                and isinstance(source, dict) and source.get('tensorSHA256') == local['tensorSHA256'])
        result['siteRecord'] = 'registered' if same else 'different'
    return result


#: What ``siteRecord`` says in a sentence. Swift twin ``LensLibrary.recordClauses``.
_RECORD_CLAUSES = {
    'registered': 'the site has its record',
    'absent': 'the site does not have its record',
    'different': 'the site has a different lens under that name',
    'unknown': 'this engine does not show its lens records here, so whether it has the record is not known',
}


def status_message(result):
    """Swift twin ``LensLibrary.statusMessage``."""
    lens = result['lensID']
    if not result['libraryAvailable']:
        lead = f'The engine does not offer a lens library, so lens {lens} is not held in one'
    elif result['held']:
        lead = f"The site's lens library holds lens {lens}"
    else:
        lead = f"The site's lens library does not hold lens {lens}"
    return lead + '; ' + _RECORD_CLAUSES[result['siteRecord']] + '.'


def remote(client, invocation, common):
    """``runner lens-push`` and ``runner lens-status``."""
    from ..cli_envelope import CLIResult
    from ..client_cli import ClientRefusal
    from .diagnostic_commands import refusal_fields
    verb = invocation.spec.verb
    if len(invocation.positionals) != 1:
        raise ClientRefusal(code='usage', reason=f'runner {verb} takes one lens ID.',
                            repair_action=f'steerlab runner {verb} --help')
    if invocation.workspace_root is None:
        raise ClientRefusal(code='usage', reason=f'runner {verb} reads the lens from a workspace, and none was named.',
                            repair_action=f'steerlab --root <workspace directory> runner {verb} <lensID> --runner <url>')
    lens_id = invocation.positionals[0]
    try:
        if verb == 'lens-push':
            result = push(client, invocation.workspace_root, lens_id, adopt=invocation.one('--adopt'))
            message = push_message(result)
        else:
            result = status(client, invocation.workspace_root, lens_id)
            message = status_message(result)
    except archives.Refusal as exc:
        raise ClientRefusal(reason=str(exc), repair_action=exc.repair_action, payload=common,
                            **refusal_fields(exc)) from exc
    print(json.dumps(result, indent=2))
    return CLIResult(message=message, changed=result['changed'], payload={**common, 'response': result})


# --- the same lens in a second workspace --------------------------------------


def _clone(source, target):
    """Clone one file where the filesystem can (APFS on macOS): the copy uses
    no extra disk space until one side changes. False when it cannot."""
    if sys.platform != 'darwin':
        return False
    import ctypes
    try:
        function = ctypes.CDLL(None, use_errno=True).clonefile
    except (AttributeError, OSError):
        return False
    function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    function.restype = ctypes.c_int
    return function(os.fsencode(source), os.fsencode(target), 0) == 0


def _copy_plan(root, lens_id, source_root):
    root = Path(root).resolve()
    _lens_name(lens_id)
    if not isinstance(source_root, str) or not source_root:
        raise archives.MalformedRequest('Name the workspace that holds the lens with --from <workspace directory>.')
    origin = Path(source_root).expanduser()
    if not origin.is_dir():
        raise LensRefusal(f'There is no workspace folder at {source_root}.',
                          'Give --from the folder of the workspace that holds the lens.')
    origin = origin.resolve()
    if origin == root:
        raise LensRefusal('--from names this workspace, which already holds its own lenses.',
                          'Give --from the other workspace, and name this one with --workspace or --root.')
    _, record, relative = workspace_lens(origin, lens_id)
    directory = Path(paths.jlens_lens_directory(lens_id, str(origin))).relative_to(origin).as_posix()
    if relative != canonical_path(lens_id):
        raise LensRefusal(
            f'Lens {lens_id} keeps its converted file outside its own folder in that workspace, so a copy '
            'of the folder would not carry it.',
            'Import the lens again in that workspace, then copy it.')
    entries = archives.pin(origin, archives.files_in(origin, directory))
    by_path = {entry['path']: entry for entry in entries}
    if by_path[relative]['sha256'] != record['converted']['sha256']:
        raise LensRefusal('Converted lens bytes in that workspace differ from the imported hash; nothing was copied.',
                          'Restore that lens folder from a copy, or import the lens again there.')
    action = 'copy'
    target = archives.ordinary(root, directory, missing=True)
    if target.exists():
        try:
            _, present, _ = workspace_lens(root, lens_id)
        except LensRefusal as exc:
            raise LensRefusal(
                f'This workspace already has a folder for lens {lens_id} ({directory}) that holds no '
                'readable lens. Nothing was copied.',
                'Move that folder aside, then run lens-copy again.', code='lensNameTaken') from exc
        ours, theirs = present.get('source') or {}, record.get('source') or {}
        if (present['converted']['sha256'] != record['converted']['sha256']
                or ours.get('tensorSHA256') != theirs.get('tensorSHA256')):
            raise LensRefusal(
                f'This workspace already holds a different lens named {lens_id}: the two records describe '
                'different bytes. Nothing was copied.',
                'Use the lens this workspace holds, or import the other one here from its source files '
                '(science artifact-plan, then science artifact-import), which gives it a name of its own.',
                code='lensNameTaken')
        action = 'none'
    plan = {'schemaVersion': 1, 'lensID': lens_id, 'sourceRoot': str(origin), 'workspaceRoot': str(root),
            'directory': directory, 'files': entries, 'bytes': sum(entry['bytes'] for entry in entries),
            'convertedSHA256': record['converted']['sha256'], 'action': action}
    return {**plan, 'planSHA256': archives.digest(plan)}, origin, root


def copy(root, lens_id, source_root, expected=None):
    """The ``lens-copy`` action: the lens ``lens_id`` of the workspace at
    ``source_root``, under the same name and record, in the workspace at
    ``root``. Without ``expected`` it is a preview that writes nothing; with
    the preview's ``planSHA256`` it copies exactly what was reviewed.

    Every file is hashed before, and again after it is copied. On macOS each
    file is cloned where the filesystem allows; anywhere else, or where a
    clone fails, it is copied. ``copyMethod`` says which."""
    plan, origin, root = _copy_plan(root, lens_id, source_root)
    if expected is None:
        return {**plan, 'changed': False, 'existing': plan['action'] == 'none'}
    if expected != plan['planSHA256']:
        raise LensRefusal('The lens, or this workspace, changed after the copy was reviewed. Nothing was copied.',
                          'Run lens-copy without --plan-sha256 to review it again.')
    if plan['action'] == 'none':
        return {**plan, 'changed': False, 'existing': True}
    parent = archives.ordinary(root, 'runs/jlens-lenses', missing=True)
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / lens_id
    methods = set()
    with tempfile.TemporaryDirectory(prefix='.lens-copy-', dir=parent) as temporary:
        staged = Path(temporary) / 'result'
        for entry in plan['files']:
            source = archives.ordinary(origin, entry['path'])
            copied = staged / Path(entry['path']).relative_to(plan['directory'])
            copied.parent.mkdir(parents=True, exist_ok=True)
            if _clone(source, copied):
                methods.add('clone')
            else:
                shutil.copyfile(source, copied)
                methods.add('copy')
            if archives.file_hash(copied) != entry['sha256']:
                raise LensRefusal('A lens file changed while it was being copied. Nothing was copied.',
                                  'Run lens-copy again once nothing else is writing that lens folder.')
        try:
            archives.publish_directory(staged, target)
        except FileExistsError as exc:
            raise LensRefusal(f'A folder for lens {lens_id} appeared in this workspace during the copy. Nothing was replaced.',
                              'Run lens-copy without --plan-sha256 to review what is there now.') from exc
    return {**plan, 'changed': True, 'existing': False, 'copyMethod': 'clone' if methods == {'clone'} else 'copy',
            'outputDirectory': str(target)}


def copy_message(result):
    """One true sentence about a ``lens-copy`` answer. Swift twin
    ``LensLibrary.copyMessage``."""
    lens = result['lensID']
    if result['existing']:
        return f'This workspace already holds lens {lens}; nothing was copied.'
    if not result['changed']:
        return (f"Reviewed a copy of lens {lens} ({len(result['files'])} files, {result['bytes']} bytes); "
                f"nothing was written. Repeat the command with --plan-sha256 {result['planSHA256']} to copy it.")
    how = 'as a clone that uses no extra disk space' if result['copyMethod'] == 'clone' else 'as a full copy'
    return f'Copied lens {lens} into this workspace {how}, under the same name and record.'
