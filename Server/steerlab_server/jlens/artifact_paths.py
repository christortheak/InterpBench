"""Resolve the importer-owned converted file when a lens library is relocated.

Records remain byte-identical. Hash verification stays in qualification's
existing once-per-readout gate, and transport checks it before staging. Do not
hash a multi-gigabyte artifact on every layer load.
"""
import json
import os
from pathlib import Path
from ..experiment import paths


def converted_file(lens_id, reference, root=None, *, library=True):
    base = Path(root or paths.project_root()).resolve()
    source = Path(reference)
    # Only the importer's canonical layout is relocatable. Arbitrary external
    # references retain their original standalone behavior and are refused by
    # the managed input owner's containment gate.
    canonical_suffix = ('runs', 'jlens-lenses', lens_id, 'jacobians.safetensors')
    relocatable = source.is_absolute() and source.parts[-4:] == canonical_suffix
    if relocatable:
        candidate = base.joinpath(*canonical_suffix)
        if candidate.is_file(): return str(candidate)
    resolved = paths.resolve(reference, str(base))
    # A site may keep a lens's converted bytes once, outside any workspace, in
    # its lens library (public issues #13 and #14), and give a workspace only
    # the record. A file present in the workspace always wins: the library is
    # asked only when the workspace's canonical file is absent, for the
    # record's `converted.sha256`. A record written on another machine names
    # that machine's folder; when this workspace has no copy, the library's
    # bytes are preferred to whatever that path names here. Nothing is hashed:
    # the callers' existing checks hash whatever is resolved.
    if library and (relocatable or not os.path.exists(resolved)):
        held = _library_file(lens_id, root)
        if held: return held
    return resolved


def _library_file(lens_id, root):
    """The lens library's file for this lens's recorded hash, or None. The
    record is read from where the caller resolved the lens:
    `<root>/runs/jlens-lenses/<lens_id>/lens.json`."""
    try:
        record = json.loads(Path(paths.jlens_lens_directory(lens_id, root), 'lens.json').read_bytes())
        digest = (record.get('converted') or {}).get('sha256')
    except (OSError, ValueError, AttributeError, TypeError):
        return None
    from . import lens_library
    return lens_library.held_file(digest)
