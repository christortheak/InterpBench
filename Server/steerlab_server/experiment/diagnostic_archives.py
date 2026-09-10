"""Bounded, portable diagnostic transport and local byte custody (no GPU imports)."""
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from . import manifest_files

META = 'steerlab-bundle.json'
MAX_FILES = 20000
MAX_BYTES = 16 * 1024**3
MAX_METADATA = 8 * 1024**2


class Refusal(ValueError):
    code = 'diagnosticTransportRefused'
    repair_action = 'Review the diagnostic archive, originating job and exact local workspace; retain remote originals until custody verifies.'


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def parts(relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or '\0' in relative:
        raise Refusal('Invalid portable path.')
    items = relative.split('/')
    if any(p in ('', '.', '..') for p in items):
        raise Refusal('Portable paths must be relative ordinary components.')
    return items


def ordinary(root, relative, *, missing=False):
    root = Path(root).resolve()
    path = root
    for i, component in enumerate(parts(relative)):
        path = path / component
        if not path.exists() and not path.is_symlink() and missing:
            continue
        if path.is_symlink() or (i < len(parts(relative)) - 1 and not path.is_dir()):
            raise Refusal('Transport and custody refuse symlinks or non-directory ancestors: ' + relative)
    return path


def files_in(root, relative):
    target = ordinary(root, relative)
    if target.is_file():
        return [relative]
    if not target.is_dir():
        raise Refusal('Required input or output is missing: ' + relative)
    result = []
    for directory, dirs, files in os.walk(target, followlinks=False):
        for name in dirs + files:
            item = Path(directory) / name
            rel = item.relative_to(Path(root).resolve()).as_posix()
            checked = ordinary(root, rel)
            if not checked.is_file() and not checked.is_dir():
                raise Refusal('Only ordinary files and directories can travel.')
        result.extend((Path(directory) / name).relative_to(Path(root).resolve()).as_posix() for name in files)
    return sorted(result)


def snapshot(root, paths):
    entries = []
    for relative in sorted(set(paths)):
        path = ordinary(root, relative)
        if not path.is_file():
            raise Refusal('A declared member is not an ordinary file: ' + relative)
        entries.append({'path': relative, 'sha256': file_hash(path), 'bytes': path.stat().st_size})
    if not entries or len(entries) > MAX_FILES or sum(e['bytes'] for e in entries) > MAX_BYTES:
        raise Refusal('The diagnostic archive is empty or exceeds transport bounds.')
    return entries


def publish_file(source, target):
    # Hard-link publication cannot replace even a concurrently created target.
    os.link(source, target)


def publish_directory(source, target):
    """Atomic create-only directory rename on the supported Mac/Linux clients.

    Refuse if the platform lacks a create-only primitive; never replace a
    directory another author created between a check and a rename. Where the
    filesystem itself rejects the no-replace rename flag (Lustre and NFS
    answer EINVAL), publication claims the name with `mkdir` first — see
    `_publish_by_claim`.
    """
    error = _rename_noreplace(source, target)
    if error == 0:
        return None
    if error in _NOREPLACE_UNSUPPORTED:
        return _publish_by_claim(source, target)
    raise OSError(error, os.strerror(error), str(target))


def _rename_noreplace(source, target):
    """The native no-replace rename; returns 0 or the errno it failed with."""
    import ctypes
    import sys
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == 'darwin':
        function = library.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = function(os.fsencode(source), os.fsencode(target), 0x4)  # RENAME_EXCL, sys/stdio.h
    elif sys.platform.startswith('linux') and hasattr(library, 'renameat2'):
        function = library.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        source_fd = os.open(Path(source).parent, os.O_RDONLY | os.O_DIRECTORY)
        target_fd = os.open(Path(target).parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            result = function(source_fd, os.fsencode(Path(source).name), target_fd, os.fsencode(Path(target).name), 1)  # RENAME_NOREPLACE
        finally:
            os.close(source_fd); os.close(target_fd)
    else:
        raise Refusal('This platform lacks atomic create-only directory publication.')
    return ctypes.get_errno() if result else 0


# errno values a filesystem returns when it does not implement the
# no-replace rename flag at all (Lustre and NFS answer EINVAL; older kernels
# ENOSYS/ENOTSUP). EEXIST and everything else keep their meaning.
_NOREPLACE_UNSUPPORTED = frozenset(x for x in (
    getattr(errno, 'EINVAL', None), getattr(errno, 'ENOSYS', None),
    getattr(errno, 'ENOTSUP', None), getattr(errno, 'EOPNOTSUPP', None)) if x is not None)


def _publish_by_claim(source, target):
    """Create-only publication where RENAME_NOREPLACE is unsupported.

    `mkdir` claims the target name atomically (EEXIST if any author holds
    it), then the staged directory is renamed over the empty directory this
    call just created — POSIX rename replaces only an EMPTY directory, so a
    claim another process wrote into in between refuses with ENOTEMPTY and
    is left for its author. Nothing created by anyone else is ever replaced.
    """
    os.mkdir(target)  # raises FileExistsError (EEXIST) when the name is taken
    try:
        os.rename(source, target)
    except OSError:
        try:
            os.rmdir(target)  # only succeeds while the claim is still ours and empty
        except OSError:
            pass
        raise


def package(root, paths, destination, *, kind, context, expected_entries=None):
    root, destination = Path(root).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise Refusal('Archive destination already exists; choose a new file.')
    entries = snapshot(root, paths)
    if expected_entries is not None and entries != expected_entries: raise Refusal('Source files changed after review; nothing packaged.')
    metadata = {'schemaVersion': 1, 'kind': kind, 'context': context, 'entries': entries}
    data = encoded(metadata)
    if len(data) > MAX_METADATA:
        raise Refusal('Diagnostic metadata exceeds the transport bound.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        output = Path(staging) / 'archive.tar.gz'
        with tarfile.open(output, 'w:gz') as archive:
            for entry in entries:
                path = ordinary(root, entry['path'])
                info = tarfile.TarInfo(entry['path']); info.size = entry['bytes']; info.mode = 0o600
                with open(path, 'rb') as handle:
                    archive.addfile(info, handle)
            info = tarfile.TarInfo(META); info.size = len(data); info.mode = 0o600
            archive.addfile(info, io.BytesIO(data))
        # Verify captured bytes, not a second reading of mutable source paths.
        inspect(output, file_hash(output))
        publish_file(output, destination)
    return {**metadata, 'bundlePath': str(destination), 'bundleSha256': file_hash(destination)}


def inspect(archive_path, expected, *, extract_to=None):
    try:
        return _inspect(archive_path, expected, extract_to=extract_to)
    except (tarfile.TarError, EOFError, UnicodeError, TypeError, KeyError) as exc:
        raise Refusal('Malformed diagnostic archive; retain originals and inspect the transport source.') from exc


def _inspect(archive_path, expected, *, extract_to=None):
    if not isinstance(expected, str) or len(expected) != 64 or file_hash(archive_path) != expected:
        raise Refusal('Archive SHA-256 mismatch; nothing is imported.')
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = []; total = 0
        for member in archive:
            total += member.size
            if len(members) >= MAX_FILES + 1 or total > MAX_BYTES + MAX_METADATA:
                raise Refusal('Archive members exceed transport bounds.')
            members.append(member)
        names = [m.name for m in members]
        if len(members) > MAX_FILES + 1 or len(set(names)) != len(names):
            raise Refusal('Duplicate or excessive archive members.')
        for member in members:
            parts(member.name)
            if not member.isfile() or member.size < 0:
                raise Refusal('Archives may contain only ordinary files.')
        if META not in names or archive.getmember(META).size > MAX_METADATA:
            raise Refusal('Missing or excessive diagnostic metadata.')
        metadata = json.load(archive.extractfile(META))
        if not isinstance(metadata, dict) or set(metadata) != {'schemaVersion', 'kind', 'context', 'entries'} or metadata['schemaVersion'] != 1 or metadata['kind'] not in ('diagnosticInput', 'diagnosticEvidence'):
            raise Refusal('Unsupported diagnostic archive schema.')
        if not isinstance(metadata['context'], dict): raise Refusal('Invalid diagnostic context.')
        entries = metadata['entries']
        if not isinstance(entries, list) or not entries or len(entries) > MAX_FILES:
            raise Refusal('Invalid diagnostic member inventory.')
        declared = {}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {'path', 'sha256', 'bytes'}:
                raise Refusal('Invalid diagnostic member declaration.')
            parts(entry['path'])
            if not isinstance(entry['sha256'], str) or len(entry['sha256']) != 64: raise Refusal('Invalid member digest.')
            if entry['path'] in declared or type(entry['bytes']) is not int or entry['bytes'] < 0:
                raise Refusal('Duplicate member or invalid byte count.')
            declared[entry['path']] = entry
        if set(names) != set(declared) | {META} or sum(e['bytes'] for e in entries) > MAX_BYTES:
            raise Refusal('Archive closure or aggregate size mismatch.')
        for name, entry in declared.items():
            member = archive.getmember(name)
            if member.size != entry['bytes']:
                raise Refusal('Member byte count mismatch.')
            hashed = hashlib.sha256()
            output = None
            try:
                if extract_to:
                    target = ordinary(extract_to, name, missing=True)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    output = open(target, 'xb')
                with archive.extractfile(member) as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        hashed.update(block)
                        if output: output.write(block)
            finally:
                if output: output.close()
            if hashed.hexdigest() != entry['sha256']:
                raise Refusal('Archived member hash mismatch: ' + name)
    return metadata


def import_evidence(archive, expected, root, *, expected_context=None):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected): raise Refusal('Invalid archive digest.')
    root = Path(root).resolve()
    if not root.is_dir():
        raise Refusal('Evidence import needs an existing workspace directory. Select the intended workspace or explicitly create it before importing; no directories were created.')
    with manifest_files.transaction(str(root / '.steerlab/diagnostic-import'), workspace_root=str(root)):
        cache = ordinary(root, '.steerlab/diagnostic-archives', missing=True); cache.mkdir(parents=True, exist_ok=True)
        retained = ordinary(root, '.steerlab/diagnostic-archives/' + expected + '.tar.gz', missing=True)
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            captured = Path(temp) / 'captured.tar.gz'
            if Path(archive).is_symlink() or not Path(archive).is_file(): raise Refusal('Archive must be an ordinary file.')
            shutil.copyfile(archive, captured)
            staged = Path(temp) / 'expanded'; staged.mkdir()
            meta = inspect(captured, expected, extract_to=staged)
            if meta['kind'] != 'diagnosticEvidence': raise Refusal('This is not diagnostic evidence.')
            context = meta['context']
            if expected_context is not None and context != expected_context: raise Refusal('Archive origin differs from the exporting job; nothing imported.')
            prefix = context['outputRelative']
            components = parts(prefix)
            if len(components) != 2 or components[0] not in ('runs', 'diagnostics'):
                raise Refusal('Evidence must name one diagnostic or battery output directory.')
            if any(not e['path'].startswith(prefix + '/') for e in meta['entries']):
                raise Refusal('Evidence contains files outside its declared output.')
            target = ordinary(root, prefix, missing=True)
            existed = target.exists()
            if existed and snapshot(root, files_in(root, prefix)) != meta['entries']:
                raise Refusal('Existing immutable output differs; choose a separate workspace.')
            if retained.exists():
                if file_hash(retained) != expected: raise Refusal('Retained archive differs from its address.')
            else: publish_file(captured, retained)
            if not existed:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Same lock is used by all adapters of this owner. Directory
                # publication refuses a destination created before this point.
                if target.exists(): raise Refusal('Output destination appeared during import.')
                publish_directory(staged / prefix, target)
            receipt = {'schemaVersion': 1, 'kind': 'diagnosticCustody', 'workspaceRoot': str(root),
                       'archivePath': retained.relative_to(root).as_posix(), 'archiveSHA256': expected,
                       'context': context, 'files': meta['entries']}
            receipt_hash = digest(receipt)
            directory = ordinary(root, '.steerlab/diagnostic-custody', missing=True); directory.mkdir(parents=True, exist_ok=True)
            path = ordinary(root, '.steerlab/diagnostic-custody/' + receipt_hash + '.json', missing=True)
            verify_document(receipt, root)
            if path.exists():
                if path.read_bytes() != encoded(receipt): raise Refusal('Custody receipt changed.')
            else:
                staged_receipt = Path(temp) / 'receipt.json'; staged_receipt.write_bytes(encoded(receipt))
                publish_file(staged_receipt, path)
        return {'receiptSHA256': receipt_hash, 'receipt': receipt, 'changed': not existed, 'outputDirectory': str(target)}


def verify_document(receipt, root):
    root = Path(root).resolve()
    if receipt.get('schemaVersion') != 1 or receipt.get('kind') != 'diagnosticCustody' or receipt.get('workspaceRoot') != str(root):
        raise Refusal('Custody belongs to another workspace or schema.')
    path = '.steerlab/diagnostic-archives/' + receipt['archiveSHA256'] + '.tar.gz'
    if receipt['archivePath'] != path: raise Refusal('Custody archive path mismatch.')
    meta = inspect(ordinary(root, path), receipt['archiveSHA256'])
    if meta['kind'] != 'diagnosticEvidence' or meta['context'] != receipt['context'] or meta['entries'] != receipt['files']:
        raise Refusal('Receipt does not match its retained archive.')
    if snapshot(root, files_in(root, receipt['context']['outputRelative'])) != receipt['files']:
        raise Refusal('Expanded local evidence has changed or is missing.')
    return receipt


def verify(receipt_hash, root):
    if not isinstance(receipt_hash, str) or len(receipt_hash) != 64 or any(c not in '0123456789abcdef' for c in receipt_hash): raise Refusal('Invalid receipt digest.')
    path = ordinary(root, '.steerlab/diagnostic-custody/' + receipt_hash + '.json')
    receipt = json.loads(path.read_bytes())
    if file_hash(path) != receipt_hash: raise Refusal('Receipt content hash mismatch.')
    return verify_document(receipt, root)


def inventory(root):
    directory = ordinary(root, '.steerlab/diagnostic-custody', missing=True)
    receipts, issues = [], []
    if directory.exists():
        for path in sorted(directory.glob('*.json')):
            try:
                receipts.append({'receiptSHA256': path.stem, 'receipt': verify(path.stem, root), 'verified': True})
            except (ValueError, OSError, KeyError, tarfile.TarError) as exc:
                issues.append({'receiptSHA256': path.stem, 'reason': str(exc)})
    return {'receipts': receipts, 'issues': issues}
