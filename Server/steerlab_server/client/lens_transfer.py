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

Every sentence both clients say is here too. The Swift twins are in
``LensLibrary.swift``, and the two are held to the same bytes by
``prompts/fixtures/lens-library/refusals.json``.

"Lens library" in this module always means the site's. A workspace's own
lenses are the folders under ``runs/jlens-lenses/``.
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
from ..experiment.strict_json import strict

RECORD_NAME = 'lens.json'
RECEIPT_NAME = 'import-receipt.json'
CONVERTED_NAME = 'jacobians.safetensors'
LENS_FOLDER = 'runs/jlens-lenses'
_SHA256 = re.compile('[0-9a-f]{64}')
#: The safetensors format's own bound on its JSON header.
_HEADER_LIMIT = 100_000_000
_BLOCK = 1024 * 1024


# --- the sentences both clients say -------------------------------------------

#: What both clients answer when an engine has no lens library to use. The
#: engine's own refusal of a staged request carries the same code.
LIBRARY_UNAVAILABLE_CODE = 'lensLibraryUnavailable'
#: The repair for the upload of a request that refers to its lens. Swift twin:
#: ``LensLibrary.archiveUnavailableRepair``.
LIBRARY_UNAVAILABLE_REPAIR = (
    'Package the request without --lens-reference so it carries its lens, or give the engine a lens '
    "library folder it can write (on a cluster, the site profile's lensLibrary folder; on any other "
    'engine, STEERLAB_LENS_LIBRARY_ROOT) and upload the request again. An engine from before the lens '
    'library must be updated first.')
#: The repair for ``lens-push``, which packages nothing. Swift twin:
#: ``LensLibrary.pushUnavailableRepair``.
PUSH_UNAVAILABLE_REPAIR = (
    "Give the engine a lens library folder it can write (on a cluster, the site profile's lensLibrary "
    'folder; on any other engine, STEERLAB_LENS_LIBRARY_ROOT), then run the command again. An engine '
    'from before the lens library must be updated first.')
#: Added to the repair for a bundle upload the engine refuses as too large.
#: Swift twin: ``LensLibrary.oversizeAdvice``.
OVERSIZE_LENS_ADVICE = (
    'A request that carries a lens can refer to it instead: send the lens once with lens-push, then '
    'package the request with --lens-reference.')
#: The site forbids HTTP transfer, so a lens's bytes reach it another way.
EXTERNAL_TRANSFER_CODE = 'externalTransferRequired'
#: The site holds something else under the lens's hash.
ENTRY_DIFFERS_CODE = 'lensLibraryEntryDiffers'
ENTRY_DIFFERS_REPAIR = ("Ask the site's operator to inspect that library entry; the workspace copy is the "
                        'one its record describes.')
#: The engine answered a lens-library route with something the contract does
#: not allow. A failure, not a refusal.
PROTOCOL_CODE = 'lensLibraryProtocol'
#: A part, or the whole-file check, did not get its answer.
INTERRUPTED_CODE = 'lensUploadInterrupted'
RESUME_REPAIR = 'Run the command again: it sends only what the site still lacks.'
#: The lens file is not the one that was planned.
CHANGED_CODE = 'lensRefused'
CHANGED_REASON = 'The converted lens file changed after it was checked. Nothing more was sent.'
CHANGED_REPAIR = ('Run lens-push again once nothing else is writing the lens folder: it sends only what the '
                  'site still lacks.')


#: What ``science lens-copy`` says when an argument is missing, before the
#: family's one sentence for every malformed request. Swift twins:
#: ``LensLibrary.copyNeedsLens`` and ``LensLibrary.copyNeedsFrom``.
COPY_NEEDS_LENS = ("science lens-copy takes one lens ID: the name of the lens's folder under "
                   'runs/jlens-lenses/ in the workspace that holds it.')
COPY_NEEDS_FROM = 'science lens-copy needs --from <workspace directory>: the workspace that holds the lens.'


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


def entry_differs_reason(held, workspace):
    """Swift twin ``LensLibrary.entryDiffersReason``."""
    return (f"The site holds {held} bytes under this lens's hash, and the workspace file is {workspace} "
            'bytes. Nothing was sent.')


def protocol_problem(kind, where=''):
    """``(reason, repair)`` for an answer the engine should never give. Swift
    twin ``LensLibrary.protocolProblem``."""
    return {
        'partLimit': (f'The engine at {where} reports a lens library with no usable partMaxBytes.',
                      'Update the engine; its capabilities must state the largest part it accepts.'),
        'missingParts': ('The engine named parts to send that are not in the manifest it was given.',
                         'Run the command again; if it repeats, update the engine.'),
        'notConfirmed': ('The engine did not confirm that it holds the whole lens after the upload.',
                         RESUME_REPAIR),
        'otherLens': ('The engine registered a record under another lens name than the one sent.',
                      'Run lens-status for this lens, and update the engine if it repeats.'),
    }[kind]


def interrupted_reason(lens_id, cause):
    """A part, or the whole-file check, got no answer (a lost connection, a
    timeout, or a failure on the engine). Swift twin
    ``LensLibrary.interruptedReason``."""
    cause = str(cause).strip().rstrip('.')
    return (f'Sending lens {lens_id} to the site was interrupted before the site confirmed that it holds '
            f'the whole lens ({cause}).')


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


def changed_after_check():
    """The refusal for a lens file that is not the one that was planned."""
    return LensRefusal(CHANGED_REASON, CHANGED_REPAIR, code=CHANGED_CODE)


def _malformed(message, repair):
    """A request in the wrong shape (64), with a repair about lenses and not
    the archive family's default."""
    refusal = archives.MalformedRequest(message)
    refusal.repair_action = repair
    return refusal


# --- a lens in a workspace -----------------------------------------------------


def _lens_name(lens_id):
    if (not isinstance(lens_id, str) or not lens_id or lens_id in ('.', '..')
            or any(mark in lens_id for mark in ('/', '\\', '\0'))):
        raise _malformed(f'Name one lens by its lensID: the name of its folder under {LENS_FOLDER}/.',
                         f'Give the name of one folder under {LENS_FOLDER}/ in the workspace.')
    return lens_id


def canonical_path(lens_id):
    """Where an imported lens's converted file lives, relative to a workspace."""
    return f'{LENS_FOLDER}/{lens_id}/{CONVERTED_NAME}'


def _phrases(root, other):
    """How a sentence names the workspace it is about: the one the command is
    working in, or the other one (``lens-copy --from``)."""
    return (f'The workspace at {root}', 'that workspace') if other else ('This workspace', 'this workspace')


def _ordinary(root, relative, inside, *, missing=False):
    """``archives.ordinary`` for a path on the way to a lens, refused in words
    about lenses: the archive family's own refusal and repair are about
    transport and custody, and nothing in them applies here."""
    try:
        return archives.ordinary(root, relative, missing=missing)
    except archives.Refusal as exc:
        raise LensRefusal(
            f'{relative} in {inside} is reached through a symbolic link, or through something that is not '
            'an ordinary folder. A lens is read only from ordinary files and folders inside its workspace.',
            f'Make {relative} and the folders above it ordinary files and folders inside the workspace (move '
            'the files there in place of a link), then run the command again.', code='lensPathNotOrdinary') from exc


def _move_aside(folder, inside='this workspace'):
    return (f"In {inside}, move {folder} aside, then import the lens again from its source files. That "
            f"folder's {RECORD_NAME} holds the lens's qualifications.")


def read_record(root, lens_id, *, other=False):
    """``(record, folder)``: a lens's record as stored, and its folder relative
    to the workspace. Nothing is hashed, and the converted file is not looked
    for."""
    _lens_name(lens_id)
    holder, inside = _phrases(root, other)
    folder = f'{LENS_FOLDER}/{lens_id}'
    record_file = _ordinary(root, f'{folder}/{RECORD_NAME}', inside, missing=True)
    if not record_file.is_file():
        raise LensRefusal(
            f'{holder} holds no lens named {lens_id} (no {folder}/{RECORD_NAME}).',
            f'Name a lens {inside} holds (the folders under its {LENS_FOLDER}/), or name the '
            'workspace that holds the lens.', code='lensNotFound')
    try:
        record = json.loads(record_file.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise LensRefusal(
            f'The record of lens {lens_id} in {inside} ({folder}/{RECORD_NAME}) is not readable JSON.',
            f'Restore {folder}/{RECORD_NAME} from a copy, or move {folder} aside and import the lens again '
            'from its source files.') from exc
    converted = record.get('converted') if isinstance(record, dict) else None
    if (not isinstance(record, dict) or record.get('lensID') != lens_id or not isinstance(converted, dict)
            or not isinstance(converted.get('path'), str) or not isinstance(converted.get('sha256'), str)
            or not _SHA256.fullmatch(converted['sha256'])):
        raise LensRefusal(
            f'The record of lens {lens_id} in {inside} does not name this lens with a converted file and '
            'its SHA-256.', _move_aside(folder, inside))
    return record, folder


def converted_file(root, lens_id, record, *, other=False):
    """The workspace-relative path of a lens's converted file, which must be
    there. A record that names the lens's own file (as a relative path, or as
    an absolute one from whichever machine imported it) means the file in
    this workspace's lens folder; its absence is said as what it is."""
    _, inside = _phrases(root, other)
    canonical = canonical_path(lens_id)
    named = Path(record['converted']['path'])
    if named.parts[-4:] == Path(canonical).parts:
        relative = canonical
    elif named.is_absolute() or '..' in named.parts or any(mark in str(named) for mark in ('\\', '\0')):
        raise LensRefusal(f'The record of lens {lens_id} names a converted file outside {inside}.',
                          _move_aside(f'{LENS_FOLDER}/{lens_id}', inside))
    else:
        relative = named.as_posix()
    if not _ordinary(root, relative, inside, missing=True).is_file():
        raise LensRefusal(
            f'The converted file of lens {lens_id} is missing from {inside} ({relative}).',
            'Import the lens again from its source files: the import puts that file back. Or restore the '
            'file from a copy.', code='lensFileMissing')
    return relative


def workspace_lens(root, lens_id, *, other=False):
    """``(root, record, file)`` for a lens in a workspace: its record as
    stored, and the workspace-relative path of its converted file. Nothing is
    hashed here. ``other`` says the workspace is not the one the command is
    working in (``lens-copy --from``), so a refusal names it."""
    root = Path(root).resolve()
    record, _ = read_record(root, lens_id, other=other)
    return root, record, converted_file(root, lens_id, record, other=other)


def _site_record(record, lens_id):
    """The record as a site receives it: the converted file is named by its
    place in a workspace, never by a path on this machine."""
    return {**record, 'converted': {**record['converted'], 'path': canonical_path(lens_id)}}


def _described(record, lens_id, relative, size):
    """What both lens actions say about a lens.

    The record is strict JSON: a number in it that is not finite (a
    qualification's measured value that could not be scored) is null, and
    ``nonFiniteNumbers`` counts them, so a record is never refused by a
    reader, or by a site, for a number it could not have read. ``recordJSON``
    is the same record as exact JSON text, for the Mac: its bridge reads
    every number as a double, and a record's integers and decimals must reach
    the site as they were written."""
    site, count = strict(_site_record(record, lens_id))
    source = record.get('source') if isinstance(record.get('source'), dict) else {}
    return {'lensID': lens_id, 'record': site, 'recordJSON': json.dumps(site, sort_keys=True, allow_nan=False),
            'nonFiniteNumbers': count, 'convertedSHA256': record['converted']['sha256'],
            'tensorSHA256': source.get('tensorSHA256'), 'bytes': size, 'file': relative, 'changed': False}


def lens_record(root, lens_id):
    """The ``lens-record`` action: a lens's record and hashes, with the size of
    its converted file. The file is not hashed."""
    root, record, relative = workspace_lens(root, lens_id)
    return _described(record, lens_id, relative, (root / relative).stat().st_size)


def _part_limit(value):
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if type(value) is not int or value < 1:
        raise _malformed('partMaxBytes is a whole number of bytes, at least 1.',
                         "Use the partMaxBytes the engine's capabilities report under lensLibrary.")
    return value


def _not_the_record(lens_id, relative, what):
    return LensRefusal(
        f'The converted file of lens {lens_id} ({relative}) {what}; nothing was planned.',
        'Move that file aside, then import the lens again from its source files: the import puts back the '
        'file the record names. Or restore the file from a copy.', code='lensFileDiffers')


def part_ranges(handle, size, part_max_bytes):
    """``(offset, bytes)`` for each part of a safetensors file, in order, or
    None when the file is not one.

    The first part is the header, ``[0, 8 + header length)``. Each tensor then
    starts a part of its own, in file order, from the header's
    ``data_offsets``. A part longer than ``part_max_bytes`` is split. The
    parts tile ``[0, size)`` exactly, whatever the header says: a cut is only
    ever added, so bytes no tensor claims stay in the part before them."""
    handle.seek(0)
    head = handle.read(8)
    length = int.from_bytes(head, 'little') if len(head) == 8 else 0
    if not 0 < length <= _HEADER_LIMIT or 8 + length > size:
        return None
    try:
        header = json.loads(handle.read(length))
    except (ValueError, UnicodeError):
        return None
    if not isinstance(header, dict):
        return None
    start = 8 + length
    cuts = {0, start, size}
    for name, entry in header.items():
        if name == '__metadata__':
            continue
        offsets = entry.get('data_offsets') if isinstance(entry, dict) else None
        if (not isinstance(offsets, list) or len(offsets) != 2 or any(type(item) is not int for item in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= size - start):
            return None
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
    with open(_ordinary(root, relative, 'this workspace'), 'rb') as handle:
        size = os.fstat(handle.fileno()).st_size
        ranges = part_ranges(handle, size, limit)
        if ranges is None:
            raise _not_the_record(lens_id, relative, 'is not a readable safetensors file')
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
        raise _not_the_record(lens_id, relative, 'differs from the SHA-256 its record names')
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


def require_library_for_archive(where, capabilities, path):
    """Refuse, before anything is sent, to upload a request that refers to its
    lens to an engine with no lens library.

    The capability is asked first, from the snapshot the upload already has:
    an engine with a library takes any archive, so the archive is not opened
    at all. It is read (its metadata is its last member, so that is the whole
    archive) only for an engine without one. Swift twin: the same check in
    ``ClusterClient.uploadBundle``."""
    if library_available(capabilities):
        return
    lenses = archive_library_lenses(path)
    if lenses is None:
        return
    from .runner import RunnerRefusal
    raise RunnerRefusal(library_unavailable_reason(where, archive=True),
                        code=LIBRARY_UNAVAILABLE_CODE, repair_action=LIBRARY_UNAVAILABLE_REPAIR,
                        detail={'libraryLenses': lenses})


# --- sending a lens, and asking what a site holds -----------------------------


def _protocol(kind, where=''):
    from .runner import RunnerError
    reason, repair = protocol_problem(kind, where)
    return RunnerError(reason, repair_action=repair, code=PROTOCOL_CODE)


def _answer(document, what):
    from .runner import RunnerError
    if not isinstance(document, dict):
        raise RunnerError(f'The engine answered {what} with something other than an object.',
                          repair_action='Check that --runner names a SteerLab engine with a lens library.',
                          code=PROTOCOL_CODE)
    return document


def _send(client, lens_id, call):
    """Send one part, or ask for the whole-file check. When that gets no
    answer (the connection, a timeout, or a failure on the engine), say that
    running the command again resumes: what the site received, it keeps."""
    from .runner import RunnerError, RunnerHTTPError, RunnerUnreachable
    try:
        return call()
    except (RunnerUnreachable, RunnerHTTPError) as exc:
        if isinstance(exc, RunnerHTTPError) and exc.status < 500:
            raise
        raise RunnerError(interrupted_reason(lens_id, exc.reason), repair_action=RESUME_REPAIR,
                          code=INTERRUPTED_CODE, detail=exc.detail) from exc


def push(client, root, lens_id, *, adopt=None):
    """Send one lens to the site's library and register its record there.

    In order: the capability, what the site holds, the part manifest, only the
    parts the site lacks, the whole-file check, the record. Run again after an
    interruption, it sends only what is still missing; when the site already
    holds the bytes it sends none and still registers the record. With
    ``adopt``, the bytes are a copy already on the site at that path, and the
    site checks it against the lens's hash."""
    from .runner import RunnerRefusal
    capabilities = client.capabilities()
    if not library_available(capabilities):
        raise RunnerRefusal(library_unavailable_reason(client.base_url), code=LIBRARY_UNAVAILABLE_CODE,
                            repair_action=PUSH_UNAVAILABLE_REPAIR)
    if adopt is not None:
        plan = lens_record(root, lens_id)
    else:
        try:
            limit = _part_limit(capabilities['lensLibrary'].get('partMaxBytes'))
        except archives.MalformedRequest as exc:
            raise _protocol('partLimit', client.base_url) from exc
        plan = push_plan(root, lens_id, limit)
    sha = plan['convertedSHA256']
    status = _answer(client.lens_library_status(sha), 'the lens status')
    already = status.get('held') is True
    uploaded_parts = uploaded_bytes = 0
    adopted = False
    if already and type(status.get('bytes')) is int and status['bytes'] != plan['bytes']:
        raise RunnerRefusal(entry_differs_reason(status['bytes'], plan['bytes']), code=ENTRY_DIFFERS_CODE,
                            repair_action=ENTRY_DIFFERS_REPAIR, detail={'convertedSHA256': sha})
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
                raise _protocol('missingParts')
            wanted = set(missing)
            source = str(Path(root).resolve() / plan['file'])
            for part in plan['parts']:
                if part['index'] in wanted:
                    _send(client, lens_id, lambda part=part: client.lens_library_put_part(sha, part, source))
                    uploaded_parts += 1
                    uploaded_bytes += part['bytes']
            done = _answer(_send(client, lens_id, lambda: client.lens_library_complete(sha)), 'the completed upload')
            if done.get('held') is not True or done.get('bytes') != plan['bytes']:
                raise _protocol('notConfirmed')
    registered = _answer(client.lens_library_register(plan['record']), 'the lens record')
    if registered.get('lensID') != lens_id:
        raise _protocol('otherLens')
    wrote = uploaded_parts > 0 or adopted or registered.get('changed') is True
    return {'lensID': lens_id, 'convertedSHA256': sha, 'bytes': plan['bytes'], 'held': True,
            'alreadyHeld': already, 'adopted': adopted, 'uploadedParts': uploaded_parts,
            'uploadedBytes': uploaded_bytes, 'registered': registered.get('registered'),
            'recordChanged': registered.get('changed'), 'recordNonFiniteNumbers': plan['nonFiniteNumbers'],
            'changed': wrote}


def push_message(result):
    """One true sentence about a finished push. Swift twin
    ``LensLibrary.pushMessage``."""
    lens = result['lensID']
    if result['uploadedParts']:
        sent = (f"Sent lens {lens} to the site's lens library ({result['uploadedParts']} parts, "
                f"{result['uploadedBytes']} bytes)")
    elif result['adopted']:
        sent = f'The site checked its copy of lens {lens} and keeps it in its lens library'
    elif result['alreadyHeld']:
        sent = f"The site's lens library already holds lens {lens}, so nothing was uploaded"
    else:
        # An earlier run sent every part and never got the whole-file check.
        sent = (f'The site had already received every part of lens {lens} and now holds it in its lens '
                'library, so nothing was uploaded')
    count = result.get('recordNonFiniteNumbers') or 0
    nulls = ''
    if count == 1:
        nulls = ', with 1 number in the record that is not finite stored as null'
    elif count > 1:
        nulls = f', with {count} numbers in the record that are not finite stored as null'
    return sent + '; the site has its record' + nulls + '.'


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


# --- a lens an import found ---------------------------------------------------


def import_message(result):
    """What an import says when the workspace already held the lens, or None
    for a new lens or any other instrument (the family's own sentence
    stands). Swift twin ``LensLibrary.importMessage``."""
    if not isinstance(result, dict) or result.get('kind') != 'jlens' or result.get('existing') is not True:
        return None
    lens = result['lensID']
    if result.get('restored') is True:
        return (f'This workspace already holds these files as lens {lens}; its converted file was missing '
                'and was put back.')
    return f'This workspace already holds these files as lens {lens}, so nothing was imported.'


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


def _clone_or_copy(source, target):
    """``'clone'`` or ``'copy'``, whichever put ``source`` at ``target``."""
    if _clone(source, target):
        return 'clone'
    shutil.copyfile(source, target)
    return 'copy'


def _receipt_hashes(root, folder):
    """Role to SHA-256 from a lens folder's import receipt, or None when there
    is no readable one."""
    from ..experiment.artifact_imports import file_hashes
    try:
        receipt = json.loads((Path(root) / folder / RECEIPT_NAME).read_bytes())
    except (OSError, ValueError, UnicodeError):
        return None
    return file_hashes(receipt.get('files')) if isinstance(receipt, dict) else None


def _record_hashes(record):
    converted = record.get('converted') if isinstance(record.get('converted'), dict) else {}
    source = record.get('source') if isinstance(record.get('source'), dict) else {}
    return converted.get('sha256'), source.get('tensorSHA256'), source.get('configSHA256')


def same_lens(origin, origin_folder, origin_record, root, folder, record):
    """Whether two lens folders hold the same lens: the same reviewed files
    by their import receipts, or, where either has no readable receipt, the
    same converted, tensor, and config hashes in their records."""
    ours, theirs = _receipt_hashes(root, folder), _receipt_hashes(origin, origin_folder)
    if ours is not None and theirs is not None:
        return ours == theirs
    return None not in _record_hashes(record)[:2] and _record_hashes(record) == _record_hashes(origin_record)


def _same_lens_elsewhere(origin, origin_folder, origin_record, root, lens_id):
    """The name under which ``root`` already holds the origin's lens, when
    that is another name than ``lens_id``. Folders that cannot be read are
    passed over."""
    try:
        folders = sorted(item.name for item in (root / LENS_FOLDER).iterdir() if item.is_dir() and not item.is_symlink())
    except OSError:
        return None
    for name in folders:
        if name == lens_id or name.startswith('.'):
            continue
        try:
            record = json.loads((root / LENS_FOLDER / name / RECORD_NAME).read_bytes())
        except (OSError, ValueError, UnicodeError):
            continue
        # A folder whose record names another lens is a copy someone set
        # aside (renamed in place), not a lens of this workspace.
        if (isinstance(record, dict) and record.get('lensID') == name
                and same_lens(origin, origin_folder, origin_record, root, f'{LENS_FOLDER}/{name}', record)):
            return name
    return None


def _copy_plan(root, lens_id, source_root):
    _lens_name(lens_id)
    if not isinstance(source_root, str) or not source_root:
        raise _malformed('Name the workspace that holds the lens with --from <workspace directory>.',
                         'science lens-copy <lensID> --from <workspace directory>')
    root = Path(root)
    if not root.is_dir():
        # Never created here: a typed path must not become a workspace.
        raise LensRefusal(f'There is no workspace folder at {root}.',
                          'Name an existing workspace to copy the lens into.', code='workspaceNotFound')
    root = root.resolve()
    origin = Path(source_root).expanduser()
    if not origin.is_dir():
        raise LensRefusal(f'There is no workspace folder at {source_root}.',
                          'Give --from the folder of the workspace that holds the lens.', code='workspaceNotFound')
    origin = origin.resolve()
    if origin == root:
        raise LensRefusal('--from names this workspace, which already holds its own lenses.',
                          'Give --from the other workspace, and name this one with --workspace or --root.')
    record, folder = read_record(origin, lens_id, other=True)
    canonical = canonical_path(lens_id)
    if converted_file(origin, lens_id, record, other=True) != canonical:
        raise LensRefusal(
            f'Lens {lens_id} keeps its converted file outside its own folder in that workspace, so a copy '
            'of the folder would not carry it.', _move_aside(folder, 'that workspace'))
    try:
        entries = archives.pin(origin, archives.files_in(origin, folder))
    except archives.Refusal as exc:
        raise LensRefusal(
            f'The folder of lens {lens_id} in that workspace ({folder}) holds a symbolic link, or something '
            'else that is not an ordinary file or folder, so it was not copied.',
            'Make everything in that folder an ordinary file or folder, then run lens-copy again.',
            code='lensPathNotOrdinary') from exc
    by_path = {entry['path']: entry for entry in entries}
    if by_path[canonical]['sha256'] != record['converted']['sha256']:
        raise LensRefusal(
            f'The converted file of lens {lens_id} in that workspace ({canonical}) differs from the SHA-256 '
            'its record names; nothing was copied.',
            'In that workspace, move that file aside and import the lens again from its source files (the '
            'import puts back the file the record names), then run lens-copy again.', code='lensFileDiffers')
    action, existing_id = 'copy', None
    target = _ordinary(root, folder, 'this workspace', missing=True)
    if target.exists():
        try:
            present, _ = read_record(root, lens_id)
        except LensRefusal as exc:
            raise LensRefusal(
                f'This workspace already has a folder for lens {lens_id} ({folder}) that holds no readable '
                'lens record. Nothing was copied.',
                'Move that folder aside, then run lens-copy again.', code='lensNameTaken') from exc
        if (present['converted']['sha256'] != record['converted']['sha256']
                or not same_lens(origin, folder, record, root, folder, present)):
            raise LensRefusal(
                f'This workspace already holds a different lens named {lens_id}: the two records describe '
                'different bytes. Nothing was copied.',
                'Use the lens this workspace holds, or import the other one here from its source files '
                '(science artifact-plan, then science artifact-import), which gives it a name of its own.',
                code='lensNameTaken')
        if Path(present['converted']['path']).parts[-4:] != Path(canonical).parts:
            raise LensRefusal(
                f'Lens {lens_id} in this workspace records its converted file somewhere other than {canonical}, '
                'so it cannot be checked. Nothing was copied.', _move_aside(folder))
        held = _ordinary(root, canonical, 'this workspace', missing=True)
        if not held.exists():
            action = 'restore'
        elif not held.is_file() or archives.file_hash(held) != record['converted']['sha256']:
            raise LensRefusal(
                f'The converted file of lens {lens_id} in this workspace ({canonical}) differs from the '
                'SHA-256 its record names. Nothing was copied, and that file was left as it is.',
                'Move that file aside, then run lens-copy again: it puts back the file from the other '
                'workspace.', code='lensFileDiffers')
        else:
            action = 'none'
    else:
        existing_id = _same_lens_elsewhere(origin, folder, record, root, lens_id)
        if existing_id is not None:
            action = 'none'
    plan = {'schemaVersion': 1, 'lensID': lens_id, 'sourceRoot': str(origin), 'workspaceRoot': str(root),
            'directory': folder, 'files': entries, 'bytes': sum(entry['bytes'] for entry in entries),
            'convertedSHA256': record['converted']['sha256'], 'action': action}
    if existing_id is not None:
        plan['existingLensID'] = existing_id
    return {**plan, 'planSHA256': archives.digest(plan)}, origin, root


def copy(root, lens_id, source_root, expected=None):
    """The ``lens-copy`` action: the lens ``lens_id`` of the workspace at
    ``source_root``, under the same name and record, in the workspace at
    ``root``. Without ``expected`` it is a preview that writes nothing; with
    the preview's ``planSHA256`` it does exactly what was reviewed.

    Every file is hashed before, and again after it is copied. On macOS each
    file is cloned where the filesystem allows; anywhere else, or where a
    clone fails, it is copied. ``copyMethod`` says which.

    A lens already here is not copied again. Under the same name it is
    checked: intact, nothing changes; its converted file missing, that one
    file is put back (``restored``); its converted file different, the copy
    is refused and the file is left alone. Under another name (the same
    reviewed files), nothing is copied and ``existingLensID`` names it."""
    plan, origin, root = _copy_plan(root, lens_id, source_root)
    existing = plan['action'] != 'copy'
    if expected is None:
        return {**plan, 'changed': False, 'existing': existing}
    if expected != plan['planSHA256']:
        raise LensRefusal('The lens, or this workspace, changed after the copy was reviewed. Nothing was copied.',
                          'Run lens-copy without --plan-sha256 to review it again.')
    if plan['action'] == 'none':
        return {**plan, 'changed': False, 'existing': True}
    changed = LensRefusal('A lens file changed while it was being copied. Nothing was copied.',
                          'Run lens-copy again once nothing else is writing that lens folder.')
    appeared = LensRefusal(f'A file or folder for lens {lens_id} appeared in this workspace during the copy. '
                           'Nothing was replaced.', 'Run lens-copy without --plan-sha256 to review what is there now.')
    parent = _ordinary(root, LENS_FOLDER, 'this workspace', missing=True)
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.lens-copy-', dir=parent) as temporary:
        if plan['action'] == 'restore':
            canonical = canonical_path(lens_id)
            staged = Path(temporary) / CONVERTED_NAME
            method = _clone_or_copy(_ordinary(origin, canonical, 'that workspace'), staged)
            if archives.file_hash(staged) != plan['convertedSHA256']:
                raise changed
            try:
                archives.publish_file(staged, _ordinary(root, canonical, 'this workspace', missing=True))
            except FileExistsError as exc:
                raise appeared from exc
            return {**plan, 'changed': True, 'existing': True, 'restored': True, 'copyMethod': method,
                    'outputDirectory': str(parent / lens_id)}
        methods = set()
        staged = Path(temporary) / 'result'
        for entry in plan['files']:
            copied = staged / Path(entry['path']).relative_to(plan['directory'])
            copied.parent.mkdir(parents=True, exist_ok=True)
            methods.add(_clone_or_copy(_ordinary(origin, entry['path'], 'that workspace'), copied))
            if archives.file_hash(copied) != entry['sha256']:
                raise changed
        try:
            archives.publish_directory(staged, parent / lens_id)
        except FileExistsError as exc:
            raise appeared from exc
    return {**plan, 'changed': True, 'existing': False, 'copyMethod': 'clone' if methods == {'clone'} else 'copy',
            'outputDirectory': str(parent / lens_id)}


def copy_message(result):
    """One true sentence about a ``lens-copy`` answer. Swift twin
    ``LensLibrary.copyMessage``."""
    lens = result['lensID']
    again = f"Repeat the command with --plan-sha256 {result['planSHA256']}"
    how = 'as a clone that uses no extra disk space' if result.get('copyMethod') == 'clone' else 'as a full copy'
    if result.get('existingLensID'):
        return f"This workspace already holds this lens as {result['existingLensID']}; nothing was copied."
    if result.get('restored'):
        return f'Lens {lens} was already in this workspace; its missing converted file was put back {how}.'
    if result.get('action') == 'restore':
        return (f'Lens {lens} is already in this workspace, but its converted file is missing; nothing was '
                f'written. {again} to put that file back.')
    if result['existing']:
        return f'This workspace already holds lens {lens}; nothing was copied.'
    if not result['changed']:
        return (f"Reviewed a copy of lens {lens} ({len(result['files'])} files, {result['bytes']} bytes); "
                f'nothing was written. {again} to copy it.')
    return f'Copied lens {lens} into this workspace {how}, under the same name and record.'
