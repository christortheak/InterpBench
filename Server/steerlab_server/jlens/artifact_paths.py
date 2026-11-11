"""Resolve the importer-owned converted file when a lens library is relocated.

Records remain byte-identical. Hash verification stays in qualification's
existing once-per-readout gate, and transport checks it before staging. Do not
hash a multi-gigabyte artifact on every layer load.
"""
import json
from pathlib import Path
from ..experiment import paths


def library_eligible(lens_id, reference):
    """Whether a record's converted path may be answered by the site's lens
    library: only the importer's canonical layout, its last four parts being
    `runs/jlens-lenses/<lensID>/jacobians.safetensors`, relative or absolute.

    This is the ONE rule, shared by the reader below and by the planner
    (`managed_inputs.inventory`), so the two can never disagree about where a
    lens's bytes come from: a plan that names the library's bytes is read
    from the library, and a record that names any other path is planned and
    read exactly as it was before a library existed.
    """
    return isinstance(reference, str) and Path(reference).parts[-4:] == (
        'runs', 'jlens-lenses', lens_id, 'jacobians.safetensors')


def in_workspace(path, root):
    """Whether a resolved converted file is a file inside this workspace. The
    planner's containment test, shared for the same reason as the rule above."""
    path = Path(path)
    return path.is_relative_to(Path(root).resolve()) and path.is_file()


def converted_file(lens_id, reference, root=None, *, library=True):
    base = Path(root or paths.project_root()).resolve()
    source = Path(reference)
    # Only the importer's canonical layout is relocatable. Arbitrary external
    # references retain their original standalone behavior and are refused by
    # the managed input owner's containment gate.
    canonical_suffix = ('runs', 'jlens-lenses', lens_id, 'jacobians.safetensors')
    if source.is_absolute() and source.parts[-4:] == canonical_suffix:
        candidate = base.joinpath(*canonical_suffix)
        if candidate.is_file(): return str(candidate)
    resolved = paths.resolve(reference, str(base))
    # A site may keep a lens's converted bytes once, outside any workspace, in
    # its lens library (public issues #13 and #14), and give a workspace only
    # the record. A file in the workspace always wins. The library is asked
    # only for a record in the canonical layout whose file is not in this
    # workspace, and then for the record's `converted.sha256`; a record
    # written on another machine names that machine's folder, and whatever
    # that path names here is not read. Any other reference resolves as it
    # always did, whether or not its path exists. Nothing is hashed: the
    # callers' existing checks hash whatever is resolved.
    if library and library_eligible(lens_id, reference) and not in_workspace(resolved, base):
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
