"""One lens per site, found by its hash (public issues #13 and #14).

A lens's converted file is the same bytes for every study and every workspace
that uses it, and for a 27B model it is several gigabytes. A request used to
carry that file inside its archive, so each request left another copy on the
site, and an archive that large was refused by the upload limit. A site with a
lens library keeps each converted file once, outside any workspace, under the
file's own SHA-256 (a record's ``converted.sha256``):

    <root>/<sha>/jacobians.safetensors      the converted file
    <root>/<sha>/entry.json                 its size, origin, and date
    <root>/.incoming/<sha>/                 an upload in progress

The workspace stays the truth. The library is a cache of bytes: it holds no
lens record, qualification, or result, and anything in it can be sent again
from a workspace that has the lens. So an entry is created once and never
edited, there is no delete verb, and a reader that finds no entry falls back
to exactly what it did before the library existed.

Bytes arrive three ways, recorded as an entry's ``origin``:

- ``pushed``: sent from a researcher's machine in parts, each checked against
  the plan the upload began with, then checked whole (:func:`begin`,
  :func:`commit_part`, :func:`complete`);
- ``adopted``: a file already on the site, checked whole (:func:`adopt`);
- ``fitted``: converted from a fit this engine has just finished
  (:func:`keep_fitted`).

No file is published until its whole SHA-256 equals the key it is filed
under. Readers do not hash here: verification stays where it was, in the
request closure and in ``qualification.verify_converted``.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import tempfile
import threading

from ..experiment import diagnostic_archives as archives

SCHEMA_VERSION = 1
FILE_NAME = 'jacobians.safetensors'
ENTRY_NAME = 'entry.json'
INCOMING = '.incoming'
#: The most one part of an upload may hold, whatever the site's upload limit.
PART_MAX_BYTES = 256 * 1024**2
#: A bound on an upload plan, far above any real lens (a part is at least one
#: byte and, for a real file, one tensor or 256 MiB of one).
MAX_PARTS = 65536
_KEY = re.compile('[0-9a-f]{64}')
_INDEX = re.compile('0|[1-9][0-9]{0,5}')
_PUSH = '`remote lens-push {lens}` (or `runner lens-push`)'

_LOCKS: dict = {}
_LOCKS_GUARD = threading.Lock()


class LensLibraryRefusal(archives.Refusal):
    """A declined lens-library request: a stable code, what is wrong in plain
    words, and a repair the reader can carry out."""

    def __init__(self, code, message, repair_action, *, status=409):
        super().__init__(message)
        self.code = code
        self.repair_action = repair_action
        self.status = status


def unavailable():
    return LensLibraryRefusal(
        'lensLibraryUnavailable',
        'This engine has no lens library: no folder is set for one, or the folder cannot be written.',
        "Package the request without `--lens-reference`, or set the site profile's `lensLibrary` folder.")


def not_held(lens_id, sha, repair=None):
    return LensLibraryRefusal(
        'lensNotInLibrary',
        f"This engine's lens library does not hold the lens '{lens_id}' (converted SHA-256 {sha}).",
        repair or ('Send the lens once with ' + _PUSH.format(lens=lens_id) + ', then stage the request again.'))


# --- where the library is, and what it holds ------------------------------------

def root():
    """The library's folder for this process, or ``None`` when it has none
    (``ServerProfile.lens_library_root`` says how that is decided)."""
    from ..api.profile import ServerProfile
    value = ServerProfile.from_env().lens_library_root
    return Path(value) if value else None


def writable(base):
    """Whether a library folder can take an entry: it exists and can be
    written, or the nearest folder above it that exists can create it."""
    if not base:
        return False
    probe = Path(base)
    while not probe.exists():
        if probe.parent == probe:
            return False
        probe = probe.parent
    return probe.is_dir() and os.access(probe, os.W_OK | os.X_OK)


def part_max_bytes():
    """The largest part this engine takes: its upload limit, and never more
    than :data:`PART_MAX_BYTES`."""
    from ..api.transfer_policy import max_upload_bytes
    return max(1, min(max_upload_bytes(), PART_MAX_BYTES))


def capability(base):
    """The ``lensLibrary`` section of ``GET /api/capabilities``."""
    return {'schemaVersion': SCHEMA_VERSION, 'available': writable(base), 'partMaxBytes': part_max_bytes()}


def _key(sha):
    if not isinstance(sha, str) or not _KEY.fullmatch(sha):
        raise LensLibraryRefusal(
            'lensLibraryKeyInvalid',
            "A lens is filed under its converted file's SHA-256: 64 lowercase hexadecimal characters.",
            'Use the `converted.sha256` value from the lens record (`lens.json`).', status=400)
    return sha


def _under(base, relative):
    """A path inside the library, refusing a symbolic link anywhere below its
    root: nothing here is ever read or written through one."""
    try:
        return archives.ordinary(base, relative, missing=True)
    except archives.Refusal as exc:
        raise LensLibraryRefusal(
            'lensLibraryNotOrdinary',
            f'The lens library holds a symbolic link, or a file where a folder belongs, at {relative}. '
            'Nothing was read or written there.',
            "Ask the site's operator to replace it with an ordinary folder, then try again.") from exc


def _entry(base, sha):
    """The entry for a hash, or ``None``. An existence and size check: the
    file and its record are ordinary files, and the record names this hash
    and the file's size. Nothing is hashed."""
    file, record = _under(base, f'{sha}/{FILE_NAME}'), _under(base, f'{sha}/{ENTRY_NAME}')
    if not file.is_file() or not record.is_file():
        return None
    try:
        document = json.loads(record.read_bytes())
        size = file.stat().st_size
    except (OSError, ValueError):
        return None
    if (not isinstance(document, dict) or document.get('schemaVersion') != SCHEMA_VERSION
            or document.get('convertedSHA256') != sha or document.get('bytes') != size):
        return None
    return {'path': file, 'bytes': size, 'entry': document}


def _reading():
    base = root()
    if base is None:
        raise unavailable()
    return base.resolve()


def _writing():
    base = root()
    if base is None or not writable(base):
        raise unavailable()
    base.mkdir(parents=True, exist_ok=True)
    return base.resolve()


def held_file(sha):
    """The library's file for a hash, or ``None``: for a reader resolving a
    lens whose converted file is not in its workspace
    (``artifact_paths.converted_file``). It never raises and never hashes; a
    library that cannot answer reads as one that holds nothing."""
    try:
        base = root()
        if base is None or not isinstance(sha, str) or not _KEY.fullmatch(sha):
            return None
        found = _entry(base.resolve(), sha)
        return str(found['path']) if found else None
    except (ValueError, OSError):
        return None


def reference(lens_id, sha, *, verify=False):
    """``{lensID, convertedSHA256, bytes}`` for a lens the library holds, or
    ``None``. With ``verify`` (the queued job, once per file) the library's
    file is hashed and must equal its key, as carried bytes are hashed."""
    base = root()
    if base is None or not isinstance(sha, str) or not _KEY.fullmatch(sha):
        return None
    found = _entry(base.resolve(), sha)
    if found is None:
        return None
    if verify and archives.file_hash(found['path']) != sha:
        raise LensLibraryRefusal(
            'lensLibraryEntryCorrupt',
            f"The lens library's copy of the lens '{lens_id}' no longer matches its SHA-256 ({sha}), so it was not read.",
            f"Ask the site's operator to remove the folder {sha} from the lens library, then send the lens again with "
            + _PUSH.format(lens=lens_id) + '.')
    return {'lensID': lens_id, 'convertedSHA256': sha, 'bytes': found['bytes']}


def require_held(lenses):
    """Refuse unless the library holds every lens a request refers to: an
    existence and size check, at staging and at admission."""
    base = root()
    if base is None:
        raise unavailable()
    base = base.resolve()
    for lens in lenses:
        found = _entry(base, _key(lens['convertedSHA256']))
        if found is None or found['bytes'] != lens['bytes']:
            raise not_held(lens['lensID'], lens['convertedSHA256'])


def holds(lenses):
    """Whether the library holds every lens named. It never raises: an engine
    with no library, or one that has lost an entry, holds nothing."""
    try:
        require_held(lenses)
    except (ValueError, OSError):
        return False
    return True


# --- publication: create-only, never replaced -----------------------------------

@contextmanager
def _locked(name):
    """One writer at a time for one hash or lens name in this process. Other
    processes are kept apart by create-only publication."""
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(name, threading.Lock())
    with lock:
        yield


def _hash(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _opens_as_safetensors(path):
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework='numpy') as handle:
            return bool(list(handle.keys()))
    except Exception:  # noqa: BLE001 - any failure to open is the answer
        return False


@contextmanager
def _staging(base):
    """A private folder beside the entries, removed afterwards unless it was
    published (publication renames it away)."""
    incoming = _under(base, INCOMING)
    incoming.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='publish-', dir=incoming))
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _publish(base, staged, sha, origin):
    """Publish a staged, verified file as the entry for its hash.

    The entry appears whole or not at all: its two files are written in a
    private folder, which is then renamed to the entry's name with the
    create-only helper that also works where the filesystem has no
    no-replace rename. A name another writer already holds is never replaced;
    when that writer published the same bytes, their entry is the answer.
    """
    size = (staged / FILE_NAME).stat().st_size
    entry = {'schemaVersion': SCHEMA_VERSION, 'convertedSHA256': sha, 'bytes': size, 'origin': origin,
             'addedAt': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
    (staged / ENTRY_NAME).write_bytes(archives.encoded(entry))
    try:
        archives.publish_directory(staged, _under(base, sha))
    except OSError as exc:
        found = _entry(base, sha)
        if found is not None:
            return found
        raise LensLibraryRefusal(
            'lensLibraryEntryBusy',
            f'The lens library already has a folder for {sha} that is not a complete entry: another copy of this '
            'lens is being published at this moment, or an earlier publication was interrupted.',
            'Send the lens again in a minute. If it is refused again, ask the site\'s operator to remove the '
            f'incomplete folder {sha} from the lens library.') from exc
    return _entry(base, sha)


# --- an upload, in parts ---------------------------------------------------------

def _manifest(sha, body, limit):
    def invalid(message):
        return LensLibraryRefusal(
            'lensUploadPlanInvalid', message,
            'Send the lens again with `lens-push`, which plans the parts from the file and from the part size this '
            'engine reports in its capabilities.', status=400)
    if not isinstance(body, dict) or set(body) != {'bytes', 'parts'}:
        raise invalid('An upload plan gives exactly `bytes` and `parts`.')
    total, parts = body['bytes'], body['parts']
    if type(total) is not int or total <= 0 or not isinstance(parts, list) or not 0 < len(parts) <= MAX_PARTS:
        raise invalid(f'An upload plan gives the file size in bytes and between 1 and {MAX_PARTS} parts.')
    offset, planned = 0, []
    for position, part in enumerate(parts):
        if not isinstance(part, dict) or set(part) != {'index', 'offset', 'bytes', 'sha256'}:
            raise invalid(f'Part {position} must give exactly `index`, `offset`, `bytes`, and `sha256`.')
        if any(type(part[key]) is not int for key in ('index', 'offset', 'bytes')) or part['index'] != position:
            raise invalid(f'The parts must be numbered 0, 1, 2, and so on in order; part {position} is not.')
        if part['offset'] != offset:
            raise invalid('The parts must cover the file exactly, in order, with no gap and no overlap: '
                          f"part {position} starts at byte {part['offset']}, where byte {offset} was expected.")
        if not 0 < part['bytes'] <= limit:
            raise invalid(f"Part {position} is {part['bytes']} bytes. A part holds at least 1 byte and at most "
                          f'{limit} bytes on this engine.')
        if not isinstance(part['sha256'], str) or not _KEY.fullmatch(part['sha256']):
            raise invalid(f'Part {position} needs its SHA-256 as 64 lowercase hexadecimal characters.')
        offset += part['bytes']
        planned.append({key: part[key] for key in ('index', 'offset', 'bytes', 'sha256')})
    if offset != total:
        raise invalid(f'The parts cover {offset} bytes, and the file is {total} bytes; they must cover it exactly.')
    return {'schemaVersion': SCHEMA_VERSION, 'convertedSHA256': sha, 'bytes': total, 'parts': planned}


def _upload(base, sha):
    """The folder and plan of an upload in progress, or ``(folder, None)``."""
    folder = _under(base, f'{INCOMING}/{sha}')
    plan = _under(base, f'{INCOMING}/{sha}/manifest.json')
    if not plan.is_file():
        return folder, None
    try:
        return folder, json.loads(plan.read_bytes())
    except (OSError, ValueError):
        return folder, None


def _received(folder):
    marks = folder / 'received'
    if not marks.is_dir():
        return set()
    return {int(name) for name in os.listdir(marks) if _INDEX.fullmatch(name)}


def _discard(folder):
    shutil.rmtree(folder, ignore_errors=True)


def _not_begun(sha):
    return LensLibraryRefusal(
        'lensUploadNotBegun',
        f'No upload is in progress for {sha}: none was begun, or it was discarded.',
        'Send the lens again with `lens-push`; it begins the upload and sends every part that is missing.')


def status(sha):
    """``GET /api/lens-library/{sha}``: whether the library holds a hash, and
    how far an upload in progress has come."""
    sha = _key(sha)
    base = _reading()
    found = _entry(base, sha)
    if found is not None:
        return {'held': True, 'bytes': found['bytes'], 'receivedParts': None}
    folder, plan = _upload(base, sha)
    if plan is None:
        return {'held': False, 'bytes': None, 'receivedParts': None}
    return {'held': False, 'bytes': plan['bytes'], 'receivedParts': sorted(_received(folder))}


def begin(sha, body):
    """Begin an upload, or resume one: answers which parts are still missing.

    The same plan again changes nothing. A different plan for an upload in
    progress is refused, because the parts already received were checked
    against the first one.
    """
    sha = _key(sha)
    plan = _manifest(sha, body, part_max_bytes())
    base = _writing()
    with _locked(sha):
        folder, current = _upload(base, sha)
        if _entry(base, sha) is not None:
            _discard(folder)
            return {'held': True, 'missingParts': []}
        if current is not None and current != plan:
            raise LensLibraryRefusal(
                'lensUploadPlanDiffers',
                'An upload of this lens is in progress with a different plan of parts, and its received parts were '
                'checked against that plan.',
                'Send the lens with the same client version and the same part size the upload began with. If this '
                "engine's part size has changed since, ask the site's operator to remove the unfinished upload "
                f'folder {INCOMING}/{sha} from the lens library, then run `lens-push` again.')
        received = _received(folder) if current is not None else set()
        missing = [part['index'] for part in plan['parts'] if part['index'] not in received]
        needed = sum(plan['parts'][index]['bytes'] for index in missing) + max(part['bytes'] for part in plan['parts'])
        free = shutil.disk_usage(base).free
        if free < needed:
            raise LensLibraryRefusal(
                'lensLibraryNoSpace',
                f'The lens library has {free} bytes free, and receiving this lens needs {needed}. Nothing was begun.',
                "Free space on the lens library's storage, or ask the site's operator to, then run `lens-push` again.")
        if current is None:
            for name in ('received', 'spool'):
                (folder / name).mkdir(parents=True, exist_ok=True)
            data = folder / 'data'
            if not data.exists():
                with open(data, 'xb') as stream:
                    stream.truncate(plan['bytes'])
            staged = folder / ('manifest-' + secrets.token_hex(8))
            staged.write_bytes(archives.encoded(plan))
            try:
                archives.publish_file(staged, folder / 'manifest.json')
            finally:
                staged.unlink(missing_ok=True)
        return {'held': False, 'missingParts': missing}


def expect_part(sha, index, declared):
    """What this upload's plan says about one part, checked before any byte
    of it is read; ``None`` when the library already holds the lens."""
    sha = _key(sha)
    if not isinstance(index, str) or not _INDEX.fullmatch(index):
        raise LensLibraryRefusal('lensUploadPartInvalid', 'A part is named by its number in the upload plan.',
                                 'Send the lens again with `lens-push`.', status=400)
    base = _writing()
    if _entry(base, sha) is not None:
        return None
    _, plan = _upload(base, sha)
    if plan is None:
        raise _not_begun(sha)
    if int(index) >= len(plan['parts']):
        raise LensLibraryRefusal('lensUploadPartInvalid',
                                 f"The upload plan has {len(plan['parts'])} parts; there is no part {index}.",
                                 'Send the lens again with `lens-push`.', status=400)
    part = plan['parts'][int(index)]
    if declared != part['sha256']:
        raise LensLibraryRefusal(
            'lensUploadPartCorrupt',
            f'Part {index} was announced (X-Content-SHA256) with a SHA-256 other than the one in the upload plan. '
            'Nothing was kept from it.',
            'Send the part again; `lens-push` sends only the parts that are missing.')
    return part


@contextmanager
def spool(sha):
    """A private file for one arriving part. A part is checked there and only
    then copied into the upload, so a damaged part never touches bytes that
    were already checked."""
    folder = _under(_writing(), f'{INCOMING}/{_key(sha)}/spool')
    if not folder.is_dir():
        raise _not_begun(sha)
    path = folder / secrets.token_hex(8)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def commit_part(sha, part, spooled, digest, received_bytes):
    """Keep one part: its size and SHA-256 must be the plan's. The same part
    sent again is accepted and changes nothing."""
    index = part['index']
    problem = None
    if received_bytes != part['bytes']:
        arrived = f"more than {part['bytes']}" if received_bytes > part['bytes'] else str(received_bytes)
        problem = f"arrived with {arrived} bytes, and the upload plan says {part['bytes']}"
    elif digest != part['sha256']:
        problem = 'arrived with a SHA-256 that is not the one in the upload plan'
    if problem:
        raise LensLibraryRefusal(
            'lensUploadPartCorrupt', f'Part {index} {problem}. Nothing was kept from it.',
            'Send the part again; `lens-push` sends only the parts that are missing.')
    sha = _key(sha)
    base = _writing()
    with _locked(sha):
        if _entry(base, sha) is not None:
            return
        folder, plan = _upload(base, sha)
        if plan is None or index >= len(plan['parts']) or plan['parts'][index] != part:
            raise _not_begun(sha)
        mark = folder / 'received' / str(index)
        if mark.exists():
            return
        with open(folder / 'data', 'r+b') as target, open(spooled, 'rb') as source:
            target.seek(part['offset'])
            shutil.copyfileobj(source, target, 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        try:
            with open(mark, 'x'):
                pass
        except FileExistsError:
            pass


def complete(sha):
    """Check the uploaded file whole and publish it.

    Every part was checked on arrival; the whole file is hashed here because
    only that proves the parts are the lens. A file whose hash is not the key
    it was sent under, or that does not open as a safetensors file, is
    discarded with its upload, so the next attempt starts clean.
    """
    sha = _key(sha)
    base = _writing()
    with _locked(sha):
        folder, plan = _upload(base, sha)
        found = _entry(base, sha)
        if found is not None:
            _discard(folder)
            return {'held': True, 'bytes': found['bytes']}
        if plan is None:
            raise _not_begun(sha)
        missing = sorted(set(range(len(plan['parts']))) - _received(folder))
        if missing:
            raise LensLibraryRefusal(
                'lensUploadIncomplete',
                f"{len(missing)} of {len(plan['parts'])} parts have not arrived (the first is part {missing[0]}), "
                'so the lens was not added.',
                'Run `lens-push` again; it sends only the parts that are missing.')
        data = folder / 'data'
        if not data.is_file():
            _discard(folder)
            raise _not_begun(sha)
        if data.stat().st_size != plan['bytes'] or _hash(data) != sha:
            _discard(folder)
            raise LensLibraryRefusal(
                'lensUploadMismatch',
                'The uploaded file does not have the SHA-256 it was sent under, so the upload was discarded and '
                'nothing was added to the lens library.',
                'Run `lens-push` again; it sends the lens from the start. If it is refused the same way, the lens '
                "file on your machine no longer matches its record: restore or re-import the lens first.")
        if not _opens_as_safetensors(data):
            _discard(folder)
            raise LensLibraryRefusal(
                'lensUploadNotSafetensors',
                'The uploaded file does not open as a safetensors file, so the upload was discarded and nothing was '
                'added to the lens library.',
                "Send a lens's converted file (`jacobians.safetensors` in its lens folder) with `lens-push`.")
        with _staging(base) as staged:
            try:
                os.link(data, staged / FILE_NAME)
            except OSError:
                shutil.copyfile(data, staged / FILE_NAME)
            found = _publish(base, staged, sha, 'pushed')
        _discard(folder)
        return {'held': True, 'bytes': found['bytes']}


# --- bytes already on the site ---------------------------------------------------

def adopt(path, sha, run_root):
    """Add a file that is already on the site: for a site that moves large
    files with its own transfer tool, or an operator's copy. The file must be
    an ordinary file beneath the run root. It is copied into the library, the
    copy is hashed, and it must be exactly the lens named.

    Copied, never linked: an entry is never edited, and a link would change
    with its source the next time a transfer wrote over that file in place.
    The source is left alone, and can be removed once the lens is held."""
    sha = _key(sha)
    base = _writing()

    def refused(code, message):
        return LensLibraryRefusal(
            code, message,
            "Put the lens's converted file (`jacobians.safetensors`) beneath this engine's run root with the "
            "site's transfer tool, then name that file and its `converted.sha256`.", status=400)
    if not isinstance(path, str) or not path or '\0' in path:
        raise refused('lensAdoptPathInvalid', 'Name the file to adopt as a path beneath the run root.')
    anchor = Path(run_root).resolve()
    source = Path(path) if os.path.isabs(path) else anchor / path
    try:
        relative = source.relative_to(anchor).as_posix()
        file = archives.ordinary(anchor, relative)
    except (ValueError, archives.Refusal) as exc:
        raise refused('lensAdoptPathInvalid',
                      "The file to adopt must be an ordinary file beneath this engine's run root, reached without "
                      'a symbolic link.') from exc
    if not file.is_file():
        raise refused('lensAdoptPathInvalid', 'There is no ordinary file at that path beneath the run root.')
    with _locked(sha):
        found = _entry(base, sha)
        if found is None:
            needed, free = file.stat().st_size, shutil.disk_usage(base).free
            if free < needed:
                raise LensLibraryRefusal(
                    'lensLibraryNoSpace',
                    f'The lens library has {free} bytes free, and this lens needs {needed}. Nothing was added.',
                    "Free space on the lens library's storage, or ask the site's operator to, then try again.")
            with _staging(base) as staged:
                target = staged / FILE_NAME
                shutil.copyfile(file, target)
                if _hash(target) != sha:
                    raise LensLibraryRefusal(
                        'lensAdoptMismatch',
                        f'That file does not have the SHA-256 {sha}, so nothing was added to the lens library.',
                        "Check the transfer, and name the lens's `converted.sha256` (not its source tensor hash).")
                if not _opens_as_safetensors(target):
                    raise LensLibraryRefusal(
                        'lensAdoptNotSafetensors',
                        'That file does not open as a safetensors file, so nothing was added to the lens library.',
                        "Transfer the lens's converted file (`jacobians.safetensors` in its lens folder).")
                found = _publish(base, staged, sha, 'adopted')
        return {'held': True, 'bytes': found['bytes']}


# --- a lens record on the site ---------------------------------------------------

def register(document, workspace_root):
    """Put a lens record in a workspace's lens store, for a lens whose bytes
    the library holds, so anything that resolves a lens by name finds it.

    A new record is written once, create-only, with its converted path made
    the canonical workspace-relative one: an absolute path from another
    machine never reaches the site. A record the workspace already has, for
    the same bytes, gains any qualification it lacks and is otherwise left as
    it is. The same name for different bytes is refused.
    """
    from . import lens_store, qualification
    from .schemas import JLensError, JLensRecord, write_record

    def invalid(message):
        return LensLibraryRefusal('lensRecordInvalid', message,
                                  'Send the record with `lens-push`, which reads it from the lens folder.', status=400)
    lens_id = document.get('lensID') if isinstance(document, dict) else None
    try:
        if not isinstance(lens_id, str) or len(archives.parts(lens_id)) != 1 or lens_id.startswith('.'):
            raise ValueError
        record = JLensRecord.from_dict(document)
        lens_store.lens_directory(lens_id, str(workspace_root))
    except (ValueError, TypeError, AttributeError, KeyError, JLensError) as exc:
        raise invalid('This is not a lens record (`lens.json`) with a usable lens name.') from exc
    converted = record.converted
    if (converted is None or not isinstance(converted.sha256, str) or not _KEY.fullmatch(converted.sha256)
            or not isinstance(record.source.tensorSHA256, str) or not record.source.tensorSHA256
            or not isinstance(converted.path, str)):
        raise invalid('The lens record must name its converted file, that file\'s SHA-256, and the SHA-256 of the '
                      'source tensor it was imported from.')
    suffix = ('runs', 'jlens-lenses', lens_id, FILE_NAME)
    canonical = '/'.join(suffix)
    if converted.path != canonical:
        given = Path(converted.path)
        if not given.is_absolute() or given.parts[-4:] != suffix:
            raise invalid(f"The record's converted file must be {canonical}; it names another location.")
        converted.path = canonical
    base = root()
    if base is None:
        raise unavailable()
    if _entry(base.resolve(), converted.sha256) is None:
        raise not_held(lens_id, converted.sha256,
                       'Send the lens with ' + _PUSH.format(lens=lens_id)
                       + ': it sends the bytes first, then the record.')
    workspace = Path(workspace_root).resolve()
    try:
        target = archives.ordinary(workspace, canonical.rsplit('/', 1)[0] + '/' + lens_store.RECORD_NAME, missing=True)
    except archives.Refusal as exc:
        raise invalid("The site's lens folder for this name is reached through a symbolic link; nothing was "
                      'written.') from exc
    with _locked('record:' + lens_id):
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.parent / ('.register-' + secrets.token_hex(8) + '.json')
            try:
                write_record(record, str(staged))
                archives.publish_file(staged, target)
                return {'lensID': lens_id, 'registered': True, 'changed': True}
            except FileExistsError:
                pass  # another writer created the record at this moment; compare with it below
            finally:
                staged.unlink(missing_ok=True)
        try:
            existing = lens_store.resolve(lens_id, str(workspace))
        except JLensError as exc:
            raise LensLibraryRefusal(
                'lensRecordUnreadable', f"The site's record for the lens '{lens_id}' cannot be read: {exc}",
                "Ask the site's operator to repair or remove that lens folder, then send the lens again.") from exc
        ours = (converted.sha256, record.source.tensorSHA256)
        theirs = (existing.converted.sha256 if existing.converted else None, existing.source.tensorSHA256)
        if ours != theirs:
            raise LensLibraryRefusal(
                'lensRecordDiffers',
                f"The site already has a lens named '{lens_id}' for different bytes. The site's record has converted "
                f'SHA-256 {theirs[0]} and source tensor SHA-256 {theirs[1]}; the record sent has converted SHA-256 '
                f'{ours[0]} and source tensor SHA-256 {ours[1]}. Nothing was changed.',
                'One lens name stands for one set of bytes. Use the lens the site has, or import your lens again so '
                'it gets its own name, then send it under that name.')
        known = {entry.qualificationID: entry for entry in existing.qualifications}
        added = []
        for entry in record.qualifications:
            if entry.qualificationID not in known:
                added.append(entry)
            elif known[entry.qualificationID] != entry:
                raise LensLibraryRefusal(
                    'lensQualificationDiffers',
                    f"The site's record for the lens '{lens_id}' already has a qualification "
                    f'{entry.qualificationID} with different content. Nothing was changed.',
                    'A qualification is never replaced. Compare the two records; if yours is the newer check, run '
                    'the qualification again so it gets its own ID, then send the lens again.')
        for entry in added:
            qualification.append_qualification(existing, entry, root=str(workspace))
        return {'lensID': lens_id, 'registered': True, 'changed': bool(added)}


# --- a lens fitted here stays here -----------------------------------------------

def keep_fitted(run_directory):
    """After a fit: convert its output as an import would, and keep the
    converted bytes under the hash that import will record.

    Bytes only. No lens record is written: importing a fit is still the
    researcher's reviewed step, and when they take it the site already holds
    what the import produces. Best effort by construction: nothing in the run
    directory is written, and no failure here reaches the fit. Returns the
    ``lensLibrary`` block of the job's result.
    """
    try:
        base = root()
        if base is None or not writable(base):
            return {'kept': False, 'convertedSHA256': None,
                    'reason': 'This engine has no lens library, so the fitted lens stays in its run folder only.'}
        from ..experiment import artifact_imports, artifact_sources
        spec, files = artifact_sources.description(Path(run_directory) / 'artifact-description.json')
        if spec.get('kind') != 'jlens':
            raise ValueError('the run describes no lens to import')
        tensors = artifact_sources.read_tensors(files['tensorFile'], artifact_imports.tensor_keys(spec))
        # The import's own checks: a fit an import would refuse is not kept.
        artifact_imports.lens_details(spec, tensors)
        base.mkdir(parents=True, exist_ok=True)
        base = base.resolve()
        with _staging(base) as staged:
            artifact_imports.convert_lens(spec, tensors, staged / FILE_NAME)
            sha = _hash(staged / FILE_NAME)
            with _locked(sha):
                if _entry(base, sha) is None:
                    _publish(base, staged, sha, 'fitted')
        return {'kept': True, 'convertedSHA256': sha, 'reason': None}
    except Exception as exc:  # noqa: BLE001 - keeping a copy must never fail the fit
        return {'kept': False, 'convertedSHA256': None,
                'reason': f'The fitted lens was not added to the lens library ({exc}); the fit itself is unaffected.'}
