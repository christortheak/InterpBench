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
    <root>/.incoming/{publish,begin,damaged}-*   private folders, swept when idle

The workspace stays the truth. The library is a cache of bytes: it holds no
lens record, qualification, or result, and anything in it can be sent again
from a workspace that has the lens. So an entry is created once and never
edited, there is no delete verb, and a reader that finds no entry falls back
to exactly what it did before the library existed. A folder under a hash that
is not a complete entry (storage purged its large file, or a publisher died)
is set aside once it has been idle, and a complete entry whose file no longer
hashes to its key is set aside when a queued job's check finds that, so
sending the lens again works and nobody has to remove anything by hand.

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
from dataclasses import asdict
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import threading
import time

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
#: A folder under a hash that is not a complete entry is left alone while
#: anything in it changed this recently (a publisher may be at work), and set
#: aside after that.
STALE_ENTRY_SECONDS = 15 * 60
#: An upload that has received nothing for this long may be replaced by one
#: with a different plan of parts.
IDLE_UPLOAD_SECONDS = 15 * 60
#: A private folder a killed process left behind (a publication, the creation
#: of an upload, an entry set aside) is removed once it has been idle a day.
STALE_STAGING_SECONDS = 24 * 3600
#: An upload nobody has added to for seven days is removed.
STALE_UPLOAD_SECONDS = 7 * 24 * 3600
_KEY = re.compile('[0-9a-f]{64}')
_INDEX = re.compile('0|[1-9][0-9]{0,5}')
_PUSH = '`remote lens-push {lens}` (or `runner lens-push`)'
#: What to do with a lens on a site that moves large files with its own tool.
ADOPT_REPAIR = ("Put the lens's `jacobians.safetensors` beneath this engine's run root with the site's transfer "
                'tool, then run `lens-push <lensID> --adopt <the path of that copy>`.')

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
        'Package the request without `--lens-reference`, or give this engine a lens library: set '
        "`STEERLAB_LENS_LIBRARY_ROOT` to a folder it can write (a site profile sets it from its `lensLibrary` "
        'folder), then start the engine again.')


def not_held(lens_id, sha, repair=None):
    return LensLibraryRefusal(
        'lensNotInLibrary',
        f"This engine's lens library does not hold the lens '{lens_id}' (converted SHA-256 {sha}).",
        repair or ('Send the lens once with ' + _PUSH.format(lens=lens_id) + ', then stage the request again.'))


def gone(lens_id, sha):
    """The refusal for a lens whose converted file is in neither place: not
    where the request runs, and not in the library. ``None`` on an engine with
    no library, where a missing file is the missing input it always was."""
    if root() is None or not isinstance(sha, str) or not _KEY.fullmatch(sha):
        return None
    return LensLibraryRefusal(
        'lensNotInLibrary',
        f"The lens '{lens_id}' has no converted file where this request runs, and this engine's lens library does "
        f'not hold it (converted SHA-256 {sha}).',
        'Send the lens again with ' + _PUSH.format(lens=lens_id) + ' from a workspace that has it, then try the '
        'request again. If the lens was imported in the workspace this engine serves, importing it there again '
        'also restores it.')


def storage_refusal(exc):
    """An operating-system failure while the library was at work, said
    without an error number or any of the engine's paths. The storage that
    failed may be the library's folder, the run root a file is adopted from,
    or the lens store a record is written to, so the sentence names none."""
    if getattr(exc, 'errno', None) in (errno.ENOSPC, getattr(errno, 'EDQUOT', None)):
        return LensLibraryRefusal(
            'lensLibraryNoSpace', "The engine's storage is full, so the request was not finished.",
            "Free space on the engine's storage (the lens library's folder, or its run root), or ask the site's "
            'operator to, then try again.')
    reason = getattr(exc, 'strerror', None) or type(exc).__name__
    return LensLibraryRefusal(
        'lensLibraryStorage', f"The engine's storage refused the request ({reason}).",
        "Try again. If it is refused the same way, ask the site's operator to check the lens library's folder, "
        'the file named, and who may read and write them.')


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
    except (FileNotFoundError, ValueError):
        # Gone in this instant, or not a record: not an entry. Any other
        # failure to read is the storage's, and is raised, never taken for
        # "not held": an entry is set aside only when it can be SEEN to be
        # incomplete (`_make_way`).
        return None
    if (not isinstance(document, dict) or document.get('schemaVersion') != SCHEMA_VERSION
            or document.get('convertedSHA256') != sha or document.get('bytes') != size):
        return None
    return {'path': file, 'bytes': size, 'entry': document}


def _read_entry(base, sha):
    """:func:`_entry` for a caller outside the routes (a plan, a staging): a
    failure of the storage is said as a refusal, without the engine's paths."""
    try:
        return _entry(base, sha)
    except OSError as exc:
        raise storage_refusal(exc) from exc


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
    file is hashed and must equal its key, as carried bytes are hashed.

    A file that does not hash to its key is not the lens filed there, and
    that is definite: the whole file was read. The entry is set aside, so
    sending the lens again replaces it and nobody has to remove anything. A
    failure to READ the file is not a mismatch, and sets nothing aside.
    """
    base = root()
    if base is None or not isinstance(sha, str) or not _KEY.fullmatch(sha):
        return None
    base = base.resolve()
    found = _read_entry(base, sha)
    if found is None:
        return None
    if verify:
        try:
            matches = archives.file_hash(found['path']) == sha
        except OSError as exc:
            raise storage_refusal(exc) from exc
        if not matches:
            with _locked(sha):
                left = _set_aside(base, sha)
            said = f"The lens library's copy of the lens '{lens_id}' no longer matches its SHA-256 ({sha}), so it was not read."
            if left is None:
                raise LensLibraryRefusal(
                    'lensLibraryEntryCorrupt', said + ' It has been set aside, so the lens can be sent again.',
                    'Send the lens again with ' + _PUSH.format(lens=lens_id) + ', then try the request again.')
            raise LensLibraryRefusal(
                'lensLibraryEntryCorrupt', said,
                f"Ask the site's operator to remove the folder {sha} from the lens library, then send the lens again "
                'with ' + _PUSH.format(lens=lens_id) + '.')
    return {'lensID': lens_id, 'convertedSHA256': sha, 'bytes': found['bytes']}


def require_held(lenses):
    """Refuse unless the library holds every lens a request refers to: an
    existence and size check, at staging and at admission."""
    base = root()
    if base is None:
        raise unavailable()
    base = base.resolve()
    for lens in lenses:
        found = _read_entry(base, _key(lens['convertedSHA256']))
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


def _idle_seconds(path):
    """How long ago anything in a folder last changed: the newest modification
    time of the folder and of everything in it. A folder that cannot be read
    counts as changed now, so nothing is ever removed on a guess."""
    try:
        newest = os.lstat(path).st_mtime
        for directory, folders, files in os.walk(path):
            for name in folders + files:
                newest = max(newest, os.lstat(os.path.join(directory, name)).st_mtime)
    except OSError:
        return 0.0
    return max(0.0, time.time() - newest)


def _sweep(incoming, keep=None):
    """Remove what a killed process left for good: a private folder idle for
    a day, and an upload nobody has added to for seven. Best effort, and
    never the upload now being worked on (``keep``)."""
    try:
        names = os.listdir(incoming)
    except OSError:
        return
    for name in names:
        if name.startswith(('publish-', 'begin-', 'damaged-')):
            limit = STALE_STAGING_SECONDS
        elif _KEY.fullmatch(name) and name != keep:
            limit = STALE_UPLOAD_SECONDS
        else:
            continue
        path = incoming / name
        if path.is_dir() and not path.is_symlink() and _idle_seconds(path) >= limit:
            _discard(path)


def _private(incoming, prefix):
    """A new private folder. Made with a plain ``mkdir``, so the process's
    umask decides who may read what is later published from it, as it does
    for any folder: a library several accounts share is readable by them."""
    folder = incoming / (prefix + secrets.token_hex(8))
    os.mkdir(folder)
    return folder


@contextmanager
def _staging(base, keep=None):
    """A private folder beside the entries, removed afterwards unless it was
    published (publication renames it away). Making one is also when what
    killed processes left behind is swept."""
    incoming = _under(base, INCOMING)
    incoming.mkdir(parents=True, exist_ok=True)
    _sweep(incoming, keep)
    directory = _private(incoming, 'publish-')
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _busy(sha):
    return LensLibraryRefusal(
        'lensLibraryEntryBusy',
        f'The lens library has an unfinished folder for this lens ({sha}) that changed within the last '
        f'{STALE_ENTRY_SECONDS // 60} minutes: another copy of the lens is being published right now, or one '
        'stopped part-way. Nothing was sent or added.',
        f'Try again in a few minutes. A folder left unfinished is set aside by itself once it has been idle for '
        f'{STALE_ENTRY_SECONDS // 60} minutes.')


def _set_aside(base, sha):
    """Move the folder under a hash aside, inside the library, to
    ``.incoming/damaged-<sha>-<token>``: one rename, swept a day later.

    Returns ``None`` when the name is free afterwards, which it also is when
    another process moved the folder in the same instant. Otherwise returns
    the error that left a folder under the name, for the caller to say."""
    target = _under(base, sha)
    try:
        incoming = _under(base, INCOMING)
        incoming.mkdir(parents=True, exist_ok=True)
        os.rename(target, incoming / f'damaged-{sha}-{secrets.token_hex(8)}')
    except OSError as exc:
        return exc if os.path.lexists(target) else None
    return None


def _make_way(base, sha):
    """Before bytes are sent or copied for a hash: deal with a folder under
    that hash that is not a complete entry.

    Storage that purges old files can remove an entry's large file and leave
    its record; a publisher can die between claiming the name and filling it.
    Either would otherwise cost a whole upload and then need an operator. A
    folder idle for :data:`STALE_ENTRY_SECONDS` is moved aside inside the
    library (one rename; swept later) and the caller carries on. One that
    changed more recently may belong to a publisher at work, so the caller is
    told to try again. A complete entry, or no folder, is left to the caller.
    """
    target = _under(base, sha)
    if not target.exists() or _entry(base, sha) is not None:
        return
    if _idle_seconds(target) < STALE_ENTRY_SECONDS:
        raise _busy(sha)
    left = _set_aside(base, sha)
    if left is None or _entry(base, sha) is not None:
        return                    # moved aside; or another process published in this instant
    if isinstance(left, FileNotFoundError):
        # Another process moved it aside and claimed the name again, in this
        # instant: that folder is a publisher's, at work right now.
        raise _busy(sha)
    raise left                    # the storage would not move it: said as that, not as a busy folder


def _publish(base, staged, sha, origin):
    """Publish a staged, verified file as the entry for its hash.

    The entry appears whole or not at all: its two files are written in a
    private folder, which is then renamed to the entry's name with the
    create-only helper that also works where the filesystem has no
    no-replace rename. A name another writer already holds is never replaced;
    when that writer published the same bytes, their entry is the answer.
    Callers clear an idle, incomplete folder first (:func:`_make_way`); the
    check here is for the race that remains.
    """
    size = (staged / FILE_NAME).stat().st_size
    entry = {'schemaVersion': SCHEMA_VERSION, 'convertedSHA256': sha, 'bytes': size, 'origin': origin,
             'addedAt': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
    # Both files reach storage before the entry gets its name, so a crash
    # cannot leave a published entry whose record or bytes are still in flight.
    with open(staged / FILE_NAME, 'rb') as stream:
        os.fsync(stream.fileno())
    with open(staged / ENTRY_NAME, 'xb') as stream:
        stream.write(archives.encoded(entry))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        archives.publish_directory(staged, _under(base, sha))
    except OSError as exc:
        found = _entry(base, sha)
        if found is not None:
            return found
        if not _under(base, sha).exists():
            raise               # nothing holds the name: the failure is the storage's own, and says so
        raise _busy(sha) from exc
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
    """Remove a private folder. An upload's plan goes first: a removal that is
    cut short then leaves a folder with no plan, which the next ``begin``
    starts clean, never one that answers a resume for parts that are gone."""
    try:
        (Path(folder) / 'manifest.json').unlink(missing_ok=True)
    except OSError:
        pass
    shutil.rmtree(folder, ignore_errors=True)


def _create_upload(base, sha, plan):
    """Create an upload folder whole: built privately, then given its name
    with the create-only rename, so it exists with its plan or not at all.
    False when another engine process created it in the same instant."""
    incoming = _under(base, INCOMING)
    incoming.mkdir(parents=True, exist_ok=True)
    private = _private(incoming, 'begin-')
    try:
        for name in ('received', 'spool'):
            os.mkdir(private / name)
        with open(private / 'data', 'xb') as stream:
            stream.truncate(plan['bytes'])
        (private / 'manifest.json').write_bytes(archives.encoded(plan))
        try:
            archives.publish_directory(private, _under(base, f'{INCOMING}/{sha}'))
        except OSError as exc:
            if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                return False
            raise
        return True
    finally:
        shutil.rmtree(private, ignore_errors=True)


def not_begun(sha):
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

    The same plan again changes nothing. A different plan is refused while
    the upload in progress is still receiving, because its parts were checked
    against the first plan; once that upload has received nothing for
    :data:`IDLE_UPLOAD_SECONDS` it is replaced. An upload folder is created
    whole (:func:`_create_upload`), so two engine processes beginning at once
    end with one upload: the one that loses reads the winner's plan and
    answers as a resume.
    """
    sha = _key(sha)
    plan = _manifest(sha, body, part_max_bytes())
    base = _writing()
    with _locked(sha):
        for _ in range(4):
            folder, current = _upload(base, sha)
            if _entry(base, sha) is not None:
                _discard(folder)
                return {'held': True, 'missingParts': []}
            if current is not None and not ((folder / 'data').is_file() and (folder / 'received').is_dir()
                                            and (folder / 'spool').is_dir()):
                # A plan whose working files are gone cannot be resumed.
                _discard(folder)
                current = None
            # A folder under this hash that is not an entry: set aside when
            # idle, refused when fresh, before a byte is sent.
            _make_way(base, sha)
            if current is not None and current != plan:
                if _idle_seconds(folder) < IDLE_UPLOAD_SECONDS:
                    raise LensLibraryRefusal(
                        'lensUploadPlanDiffers',
                        'An upload of this lens is in progress with a different plan of parts, and its received '
                        'parts were checked against that plan.',
                        'Send the lens with the same client and the same part size that upload began with, or '
                        f'wait until the unfinished upload has been idle for {IDLE_UPLOAD_SECONDS // 60} minutes; '
                        'it is then replaced when you run `lens-push` again.')
                _discard(folder)
                current = None
            received = _received(folder) if current is not None else set()
            missing = [part['index'] for part in plan['parts'] if part['index'] not in received]
            needed = sum(plan['parts'][i]['bytes'] for i in missing) + max(part['bytes'] for part in plan['parts'])
            free = shutil.disk_usage(base).free
            if free < needed:
                raise LensLibraryRefusal(
                    'lensLibraryNoSpace',
                    f'The lens library has {free} bytes free, and receiving this lens needs {needed}. Nothing was '
                    'begun.',
                    "Free space on the lens library's storage, or ask the site's operator to, then run `lens-push` "
                    'again.')
            if current is not None:
                return {'held': False, 'missingParts': missing}
            if folder.exists() and not (folder / 'manifest.json').exists():
                # A folder with no plan: nothing in it was ever checked
                # against one (a removal cut short, or debris). Start clean.
                # An upload is created whole, so a folder that has a plan by
                # now is another process's new upload, and is left alone.
                _discard(folder)
            if _create_upload(base, sha, plan):
                return {'held': False, 'missingParts': missing}
            # Another engine process created the upload in this instant: go
            # round again, read its plan, and answer as a resume.
        raise LensLibraryRefusal(
            'lensUploadBusy',
            'Another sender is beginning an upload of this lens at this moment, and this request could not join it.',
            'Run `lens-push` again; it sends only the parts that are missing.')


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
        raise not_begun(sha)
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
    base = _writing()
    folder = _under(base, f'{INCOMING}/{_key(sha)}/spool')
    if not folder.is_dir():
        raise not_begun(sha)
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
            raise not_begun(sha)
        mark = folder / 'received' / str(index)
        if mark.exists():
            return
        try:
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
        except FileNotFoundError:
            # Another engine process finished this upload and removed its
            # folder in this instant: the lens is held, or the upload is gone.
            if _entry(base, sha) is None:
                _discard(folder)
                raise not_begun(sha) from None


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
            raise not_begun(sha)
        missing = sorted(set(range(len(plan['parts']))) - _received(folder))
        if missing:
            raise LensLibraryRefusal(
                'lensUploadIncomplete',
                f"{len(missing)} of {len(plan['parts'])} parts have not arrived (the first is part {missing[0]}), "
                'so the lens was not added.',
                'Run `lens-push` again; it sends only the parts that are missing.')
        try:
            found = _check_and_publish(base, sha, folder, plan)
        except FileNotFoundError:
            # Another engine process finished this same upload and removed its
            # folder while this one was checking it: the lens is held, or the
            # upload is gone. Never an error number or a path of the engine's.
            found = _entry(base, sha)
            if found is None:
                _discard(folder)
                raise not_begun(sha) from None
        _discard(folder)
        return {'held': True, 'bytes': found['bytes']}


def _check_and_publish(base, sha, folder, plan):
    def discarded(code, what, repair):
        _discard(folder)
        return LensLibraryRefusal(
            code, f'The uploaded file {what}, so the upload was discarded and nothing was added to the lens library.',
            repair)
    mismatch = ('does not have the SHA-256 it was sent under',
                'Run `lens-push` again; it sends the lens from the start. If it is refused the same way, the lens '
                'file on your machine no longer matches its record: restore or re-import the lens first.')
    data = folder / 'data'
    if data.stat().st_size != plan['bytes']:
        raise discarded('lensUploadMismatch', *mismatch)
    _make_way(base, sha)
    with _staging(base, keep=sha) as staged:
        target = staged / FILE_NAME
        try:
            os.link(data, target)
        except FileNotFoundError:
            raise
        except OSError:
            shutil.copyfile(data, target)    # storage that cannot link: publish a copy
        # The file that is published is the file that was hashed: the link or
        # the copy in the staging folder, never the upload it was made from.
        if _hash(target) != sha:
            raise discarded('lensUploadMismatch', *mismatch)
        if not _opens_as_safetensors(target):
            raise discarded('lensUploadNotSafetensors', 'does not open as a safetensors file',
                            "Send a lens's converted file (`jacobians.safetensors` in its lens folder) with "
                            '`lens-push`.')
        return _publish(base, staged, sha, 'pushed')


# --- bytes already on the site ---------------------------------------------------

def adopt(path, sha, run_root):
    """Add a file that is already on the site: for a site that moves large
    files with its own transfer tool, or an operator's copy. The file must be
    an ordinary file beneath the run root. It is copied into the library, the
    copy is hashed, and it must be exactly the lens named.

    Copied, never linked: an entry is never edited, and a link would change
    with its source the next time a transfer wrote over that file in place.
    The source is left alone, and can be removed once the lens is held.

    ``run_root`` is the run root as the site spells it. A site's storage is
    often reached through a symbolic link, so an absolute path is accepted
    beneath that spelling or beneath the resolved one; a symbolic link is
    refused only below the root, as it is for import sources."""
    sha = _key(sha)
    base = _writing()

    def refused(message):
        return LensLibraryRefusal('lensAdoptPathInvalid', message, ADOPT_REPAIR, status=400)
    if not isinstance(path, str) or not path or '\0' in path:
        raise refused('Name the file to adopt as a path beneath the run root.')
    configured = Path(os.path.abspath(run_root))
    anchor = configured.resolve()
    relative = None
    if os.path.isabs(path):
        for spelling in (configured, anchor):
            try:
                relative = Path(path).relative_to(spelling).as_posix()
                break
            except ValueError:
                continue
    else:
        relative = Path(path).as_posix()
    try:
        if relative is None:
            raise ValueError
        file = archives.ordinary(anchor, relative)
    except (ValueError, archives.Refusal) as exc:
        raise refused("The file to adopt must be an ordinary file beneath this engine's run root, with no "
                      'symbolic link below that root.') from exc
    if not file.is_file():
        raise refused('There is no ordinary file at that path beneath the run root.')
    with _locked(sha):
        found = _entry(base, sha)
        if found is None:
            _make_way(base, sha)
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

def _non_finite(value, where='the record'):
    """Where a JSON value first holds a number that is not finite, or None."""
    if isinstance(value, float) and not math.isfinite(value):
        return where
    if isinstance(value, dict):
        for name, item in value.items():
            found = _non_finite(item, f'{where}.{name}')
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for position, item in enumerate(value):
            found = _non_finite(item, f'{where}[{position}]')
            if found:
                return found
    return None


def _finite(value):
    """A JSON value with every number that is not finite replaced by null: how
    a record is sent, so how a stored one is read when the two are compared."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {name: _finite(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _parse_record(document):
    """``(record, None)``, or ``(None, which part of the record is unusable)``."""
    from .schemas import ConvertedRef, FitProvenance, JLensRecord, Qualification, SourceRef
    for name, kind, required in (('source', SourceRef, True), ('fit', FitProvenance, False),
                                 ('converted', ConvertedRef, True)):
        block = document.get(name)
        if block is None and not required:
            continue
        try:
            if not isinstance(block, dict):
                raise TypeError
            kind(**block)
        except TypeError:
            return None, f'its `{name}` block is missing, lacks a field, or has one a lens record does not have'
    listed = document.get('qualifications') or []
    if not isinstance(listed, list):
        return None, 'its `qualifications` are not a list'
    for position, entry in enumerate(listed):
        try:
            if not isinstance(entry, dict):
                raise TypeError
            Qualification(**entry)
        except TypeError:
            return None, (f'qualification {position} in its `qualifications` lacks a field, or has one a '
                          'qualification does not have')
    try:
        return JLensRecord.from_dict(document), None
    except (TypeError, ValueError, AttributeError, KeyError):
        return None, 'it lacks one of `sourceLayers`, `dModel`, `targetLayer`, and `nPrompts`'


def register(document):
    """Put a lens record in this engine's lens store, for a lens whose bytes
    the library holds, so anything that resolves a lens by name finds it.

    The store is the one the engine's own lens importer writes, and its lens
    routes and study runs read: ``lens_store`` with no root, which follows
    ``STEERLAB_RUN_ROOT``. The write, the comparison with a record already
    there, and the appended qualifications all use that one place.

    A new record is written once, create-only, with its converted path made
    the canonical workspace-relative one: an absolute path from another
    machine never reaches the site. A record the store already has, for the
    same bytes, gains any qualification it lacks and is otherwise left as it
    is. The same name for different bytes is refused.
    """
    from . import lens_store, qualification
    from ..experiment import paths
    from .schemas import JLensError, write_record

    def invalid(part, repair=None):
        return LensLibraryRefusal(
            'lensRecordInvalid', f'The lens record cannot be used: {part}.',
            repair or 'Import the lens again on your machine, so that it has a complete record, then send it with '
                      '`lens-push`.', status=400)
    if not isinstance(document, dict):
        raise invalid('it is not a JSON object')
    unfinished = _non_finite(document)
    if unfinished:
        raise invalid(
            f'it holds a number that is not finite (NaN or infinity) at {unfinished}, and a lens record sent to a '
            'site carries no such number',
            'Update the client that sent it: current clients send null in place of such a number.')
    lens_id = document.get('lensID')
    try:
        if not isinstance(lens_id, str) or len(archives.parts(lens_id)) != 1 or lens_id.startswith('.'):
            raise ValueError
    except ValueError as exc:
        raise invalid('its `lensID` is missing, or is not a single lens name') from exc
    record, unusable = _parse_record(document)
    if record is None:
        raise invalid(unusable)
    converted = record.converted
    if not isinstance(converted.sha256, str) or not _KEY.fullmatch(converted.sha256):
        raise invalid('its `converted.sha256` is not a SHA-256 (64 lowercase hexadecimal characters)')
    if not isinstance(record.source.tensorSHA256, str) or not record.source.tensorSHA256:
        raise invalid('its `source.tensorSHA256`, the hash of the tensor it was imported from, is missing')
    suffix = ('runs', 'jlens-lenses', lens_id, FILE_NAME)
    canonical = '/'.join(suffix)
    if converted.path != canonical:
        given = Path(converted.path) if isinstance(converted.path, str) else None
        if given is None or not given.is_absolute() or given.parts[-4:] != suffix:
            raise invalid(f"its `converted.path` does not name the lens's own converted file, {canonical}")
        converted.path = canonical
    base = root()
    if base is None:
        raise unavailable()
    if _entry(base.resolve(), converted.sha256) is None:
        raise not_held(lens_id, converted.sha256,
                       'Send the lens with ' + _PUSH.format(lens=lens_id)
                       + ': it sends the bytes first, then the record.')
    try:
        target = archives.ordinary(paths.jlens_lenses_directory(None), f'{lens_id}/{lens_store.RECORD_NAME}',
                                   missing=True)
    except archives.Refusal as exc:
        raise LensLibraryRefusal(
            'lensRecordNotOrdinary',
            f"The site's folder for the lens '{lens_id}' is reached through a symbolic link, or a file stands where "
            'a folder belongs. Nothing was written.',
            "Ask the site's operator to replace it with an ordinary folder, then send the lens again.") from exc
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
            existing = lens_store.resolve(lens_id)
        except JLensError:
            raise LensLibraryRefusal(
                'lensRecordUnreadable',
                f"The site has a record for a lens named '{lens_id}' that cannot be read, so the two could not be "
                'compared. Nothing was changed.',
                "Ask the site's operator to repair or remove that lens's folder in the engine's lens store, then "
                'send the lens again. A request packaged with `--lens-reference` carries its own record and does '
                'not depend on this registration.') from None
        ours = (converted.sha256, record.source.tensorSHA256)
        theirs = (existing.converted.sha256 if existing.converted else None, existing.source.tensorSHA256)
        if ours != theirs:
            raise LensLibraryRefusal(
                'lensRecordDiffers',
                f"The site already has a lens named '{lens_id}' for different bytes. The site's record has converted "
                f'SHA-256 {theirs[0]} and source tensor SHA-256 {theirs[1]}; the record sent has converted SHA-256 '
                f'{ours[0]} and source tensor SHA-256 {ours[1]}. Nothing was changed.',
                'One lens name stands for one set of bytes. Use the lens the site has under that name. A request '
                'packaged with `--lens-reference` carries its own record and does not depend on this registration.')
        # A stored qualification may hold NaN (a score that could not be
        # computed); a record is sent with null in its place. Both are read
        # that way here, so the same qualification is never called different.
        known = {entry.qualificationID: _finite(asdict(entry)) for entry in existing.qualifications}
        added = []
        for entry in record.qualifications:
            sent = _finite(asdict(entry))
            if entry.qualificationID not in known:
                added.append(entry)
                known[entry.qualificationID] = sent      # the same entry twice in one record is added once
            elif known[entry.qualificationID] != sent:
                raise LensLibraryRefusal(
                    'lensQualificationDiffers',
                    f"The site's record for the lens '{lens_id}' already has a qualification "
                    f'{entry.qualificationID} with different content. Nothing was changed.',
                    'A qualification is never replaced. Compare the two records. If yours is the one to keep, run '
                    'the qualification again so it gets its own ID, then send the lens again.')
        # Appended in date order among themselves. What the site already has
        # keeps its order: `append_qualification` only appends, and is called,
        # not edited.
        for entry in sorted(added, key=lambda entry: str(entry.qualifiedAt)):
            qualification.append_qualification(existing, entry)
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

    An engine with no library keeps nothing, and which engines have one is
    decided in one place (``ServerProfile.lens_library_root``): an engine
    that serves a researcher's own workspace has none unless one is declared,
    so a fit there is not copied a second time. Nothing is kept either when
    the library has no room for it.
    """
    def skipped(reason):
        return {'kept': False, 'convertedSHA256': None, 'reason': reason}
    try:
        base = root()
        if base is None:
            return skipped('No folder is set for a lens library on this engine, so the fitted lens stays in its run '
                           'folder only.')
        if not writable(base):
            return skipped('The lens library folder set for this engine cannot be written from the node this fit '
                           'ran on, so the fitted lens stays in its run folder only.')
        from ..experiment import artifact_imports, artifact_sources
        spec, files = artifact_sources.description(Path(run_directory) / 'artifact-description.json')
        if spec.get('kind') != 'jlens':
            raise ValueError('the run describes no lens to import')
        base.mkdir(parents=True, exist_ok=True)
        base = base.resolve()
        needed, free = files['tensorFile'].stat().st_size, shutil.disk_usage(base).free
        if free < needed:
            return skipped(f'The lens library has {free} bytes free, and this lens needs about {needed}, so it was '
                           'not added; the fit itself is unaffected.')
        tensors = artifact_sources.read_tensors(files['tensorFile'], artifact_imports.tensor_keys(spec))
        # The import's own checks: a fit an import would refuse is not kept.
        artifact_imports.lens_details(spec, tensors)
        with _staging(base) as staged:
            artifact_imports.convert_lens(spec, tensors, staged / FILE_NAME)
            sha = _hash(staged / FILE_NAME)
            with _locked(sha):
                if _entry(base, sha) is None:
                    # The key exists only now, so this is the first moment a
                    # folder under it that is not an entry can be dealt with.
                    _make_way(base, sha)
                    _publish(base, staged, sha, 'fitted')
        return {'kept': True, 'convertedSHA256': sha, 'reason': None}
    except Exception as exc:  # noqa: BLE001 - keeping a copy must never fail the fit
        said = str(storage_refusal(exc) if isinstance(exc, OSError) else exc).rstrip('.')
        return skipped(f'The fitted lens was not added to the lens library ({said}); the fit itself is unaffected.')
