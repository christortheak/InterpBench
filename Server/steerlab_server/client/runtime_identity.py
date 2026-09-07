"""Read-only source identity for the Mac's local Python process adapter.

A release number alone cannot distinguish different builds of that release.
Hash shipped Python sources and seed resources; omit caches and installation
metadata so a wheel and the matching app payload have the same identity.
"""
import hashlib
from pathlib import Path


def source_sha256(package_root=None):
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parents[1]
    paths = set(root.rglob('*.py'))
    paths.update(p for p in (root / 'experiment/seed').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    digest = hashlib.sha256(b'steerlab-python-client-source-v1\0')
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix().encode('utf-8')):
        if path.is_symlink():
            raise ValueError('Python client sources must be ordinary installed files.')
        name = path.relative_to(root).as_posix().encode('utf-8')
        data = path.read_bytes()
        digest.update(len(name).to_bytes(8, 'big'))
        digest.update(name)
        digest.update(len(data).to_bytes(8, 'big'))
        digest.update(data)
    return digest.hexdigest()
