"""Reuse file hashes only inside one bounded input-review operation.

There is no persistent receipt cache. Every operation starts from bytes again;
identity, size, timestamps, and regular-file status are rechecked on every use
and at scope exit. The queued worker starts its own scope.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import os
from pathlib import Path
import stat

_active = ContextVar('input_hash_review', default=None)


def fingerprint(path):
    value = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('Reviewed input is no longer an ordinary file: ' + str(path))
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_mode)


def file_hash(path):
    cache = _active.get()
    if cache is None:
        with open(path, 'rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    path = str(Path(path).absolute())
    before = fingerprint(path)
    if path in cache:
        prior, digest = cache[path]
        if prior != before:
            raise ValueError('An input changed during review; review the current bytes again: ' + path)
        return digest
    with open(path, 'rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != before[:2]:
            raise ValueError('An input was replaced during review: ' + path)
    if fingerprint(path) != before:
        raise ValueError('An input changed during hashing: ' + path)
    cache[path] = (before, digest)
    return digest


@contextmanager
def session():
    if _active.get() is not None:
        yield
        return
    cache = {}
    token = _active.set(cache)
    try:
        yield
        for path, (stamp, _) in cache.items():
            if fingerprint(path) != stamp:
                raise ValueError('An input changed before review finished: ' + path)
    finally:
        _active.reset(token)


def operation(function):
    @wraps(function)
    def checked(*args, **kwargs):
        with session():
            return function(*args, **kwargs)
    return checked
