"""Resolve the importer-owned converted file when a lens library is relocated.

Records remain byte-identical. Hash verification stays in qualification's
existing once-per-readout gate, and transport checks it before staging. Do not
hash a multi-gigabyte artifact on every layer load.
"""
from pathlib import Path
from ..experiment import paths


def converted_file(lens_id, reference, root=None):
    base = Path(root or paths.project_root()).resolve()
    source = Path(reference)
    # Only the importer's canonical layout is relocatable. Arbitrary external
    # references retain their original standalone behavior and are refused by
    # the managed input owner's containment gate.
    canonical_suffix = ('runs', 'jlens-lenses', lens_id, 'jacobians.safetensors')
    if source.is_absolute() and source.parts[-4:] == canonical_suffix:
        candidate = base.joinpath(*canonical_suffix)
        if candidate.is_file(): return str(candidate)
    return paths.resolve(reference, str(base))
