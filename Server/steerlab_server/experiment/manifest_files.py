"""External file preconditions and cross-process manifest write serialization.

The lock and expected digest are authoring metadata. Neither is written into
manifest JSON, scientific content hashes, freeze documents or historical runs.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import os
import threading
from .manifest_errors import ExperimentStoreError


class StaleManifestError(ExperimentStoreError):
    code = "staleManifest"
    def __init__(self, message: str):
        super().__init__(message, gate=self.code, repair="Read the current manifest, review the intervening changes, and submit with its file digest.")


_LOCKS: dict[str, threading.RLock] = {}
_REGISTRY_LOCK = threading.Lock()
_HELD = threading.local()


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: str) -> str | None:
    try:
        with open(path, "rb") as handle:
            return digest_bytes(handle.read())
    except FileNotFoundError:
        return None


@contextmanager
def transaction(path: str, *, workspace_root: str):
    """Serialize cooperating writers on a stable sidecar inode.

    Reentrant within a thread, so an authoring transaction can invoke the
    ordinary save primitive without reacquiring an independent flock and
    deadlocking. Swift writers use this same directory and canonical-path key.
    """
    canonical = os.path.realpath(path)
    root = os.path.realpath(workspace_root)
    key = digest_bytes(canonical.encode("utf-8"))
    lock_path = os.path.join(root, ".steerlab", "manifest-locks", key + ".lock")
    with _REGISTRY_LOCK:
        lock = _LOCKS.setdefault(lock_path, threading.RLock())
    with lock:
        held = getattr(_HELD, "paths", None)
        if held is None:
            held = _HELD.paths = set()
        if lock_path in held:
            yield
            return
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        # Runtime synchronization files must not enter workspace freeze commits.
        # Keep this rule local to the metadata directory; never rewrite a
        # researcher's .gitignore or existing scientific files.
        try:
            ignore = os.open(os.path.join(os.path.dirname(lock_path), ".gitignore"),
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(ignore, "w") as handle:
                handle.write("*\n")
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            held.add(lock_path)
            try:
                yield
            finally:
                held.remove(lock_path)
        finally:
            os.close(descriptor)


def require_current(path: str, expected_digest: str | None) -> None:
    """Call under transaction; None explicitly means the file must not exist."""
    if expected_digest is not None:
        if (len(expected_digest) != 64
                or any(character not in "0123456789abcdef" for character in expected_digest)):
            raise ValueError("expected file digest must be a lowercase SHA-256")
    if file_digest(path) != expected_digest:
        raise StaleManifestError("The manifest changed after it was read; no edit was published.")


class Document(dict):
    """Raw JSON plus its external read precondition, never a JSON field.

    Normal dict/JSON iteration exposes only manifest keys. Authoring code can
    preserve unknown keys while carrying the exact bytes it read through to
    save. A plain dict is a new document, not authority to overwrite a file.
    """
    def __init__(self, value: dict, *, source_path: str, source_digest: str):
        super().__init__(value)
        self.source_path = os.path.realpath(source_path)
        self.source_digest = source_digest

    def expected_for(self, path: str) -> str:
        if os.path.realpath(path) != self.source_path:
            raise StaleManifestError("The draft belongs to another manifest path; load the intended draft before editing.")
        return self.source_digest

    def copy(self):
        return Document(self, source_path=self.source_path, source_digest=self.source_digest)
