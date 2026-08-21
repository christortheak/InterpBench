"""List, resolve, and read imported lenses; load one layer at a time.

Lenses live in ``runs/jlens-lenses/<lensID>/`` — a mutable LIBRARY subtree
inside the otherwise-immutable ``runs/`` area, beside ``model-variants`` and
``neutral-pcs``. A lens is a derived instrument, so it does not belong in
``prompts/`` (source data); and it is not a run, so it is not immutable — its
record accumulates qualification entries over time.

There is no delete. A lens can be referenced by a derived vector, a
qualification, a trace, a variant, or a frozen experiment, and removing it
would strand provenance that those artifacts assert (plan §4.2).
"""

from __future__ import annotations

import os

from ..experiment import paths
from .schemas import JLensError, JLensRecord, read_record, write_record

RECORD_NAME = "lens.json"


def _contained(path: str, parent: str) -> bool:
    path, parent = os.path.realpath(path), os.path.realpath(parent)
    return path == parent or path.startswith(parent + os.sep)


def lens_directory(lens_id: str, root: str | None = None) -> str:
    """Resolve a lens id to its directory, refusing traversal.

    ``lensID`` reaches this from API callers, so it is treated as untrusted
    input: a value like ``../../etc`` must not escape the lens library.
    """
    if not lens_id or os.path.isabs(lens_id) or os.sep in lens_id or "\0" in lens_id:
        raise JLensError(f"invalid lens id {lens_id!r}")
    library = paths.jlens_lenses_directory(root)
    directory = os.path.join(library, lens_id)
    if not _contained(directory, library):
        raise JLensError(f"lens id {lens_id!r} escapes the lens library")
    return directory


def record_path(lens_id: str, root: str | None = None) -> str:
    return os.path.join(lens_directory(lens_id, root), RECORD_NAME)


def exists(lens_id: str, root: str | None = None) -> bool:
    try:
        return os.path.exists(record_path(lens_id, root))
    except JLensError:
        return False


def resolve(lens_id: str, root: str | None = None) -> JLensRecord:
    path = record_path(lens_id, root)
    if not os.path.exists(path):
        raise JLensError(
            f"no imported lens '{lens_id}' — import it first "
            f"(steerlab-server jlens import <model>)")
    return read_record(path)


def list_lenses(root: str | None = None) -> list[JLensRecord]:
    """Every imported lens, newest-first by import timestamp."""
    library = paths.jlens_lenses_directory(root)
    out: list[JLensRecord] = []
    if not os.path.isdir(library):
        return out
    for entry in sorted(os.listdir(library)):
        path = os.path.join(library, entry, RECORD_NAME)
        if not os.path.exists(path):
            continue
        try:
            out.append(read_record(path))
        except JLensError:
            continue          # a corrupt record must not hide the healthy ones
    return sorted(out, key=lambda r: r.importedAt, reverse=True)


def save(record: JLensRecord, root: str | None = None) -> str:
    directory = lens_directory(record.lensID, root)
    os.makedirs(directory, exist_ok=True)
    return write_record(record, os.path.join(directory, RECORD_NAME))


def load_layer(record: JLensRecord, layer: int, *, root: str | None = None):
    """One ``J_l`` from the converted artifact — never the whole lens.

    This is the access pattern the conversion exists to enable: at 27B the
    full lens is ~6.6 GiB once promoted, while a single layer is ~58 MB.
    """
    from safetensors import safe_open

    if record.converted is None:
        raise JLensError(
            f"lens '{record.lensID}' has no converted artifact — re-import it")
    if layer not in record.sourceLayers:
        raise JLensError(
            f"layer {layer} is not a fitted source layer of '{record.lensID}' "
            f"(have {record.sourceLayers[0]}..{record.sourceLayers[-1]}; the "
            f"target layer {record.targetLayer} has no Jacobian by construction)")
    path = paths.resolve(record.converted.path, root)
    if not os.path.exists(path):
        raise JLensError(
            f"converted lens artifact missing at '{path}' — it is a derived "
            f"cache, so re-import from the hash-pinned upstream")
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(f"layer_{layer}")


def compatibility(record: JLensRecord, *, model_id: str, revision: str | None,
                  dtype: str | None, num_layers: int, hidden_size: int,
                  quantization: str | None = None) -> dict:
    """Can this lens be used against this exact runtime, and as what?

    Returns a report rather than a bool: callers need to distinguish "wrong
    model" from "right model, not yet qualified", and the second is a normal
    state for exploration while only the first is nonsense.
    """
    problems: list[str] = []
    if record.fit.modelID and record.fit.modelID != model_id:
        problems.append(
            f"lens was fitted on '{record.fit.modelID}', runtime is '{model_id}'")
    if record.dModel != hidden_size:
        problems.append(
            f"lens d_model {record.dModel} != runtime hidden size {hidden_size}")
    if record.targetLayer != num_layers - 1:
        problems.append(
            f"lens target layer {record.targetLayer} != runtime n_layers-1 "
            f"({num_layers - 1}) — the layer mapping does not line up")
    qualified = None
    if revision and dtype:
        q = record.qualification_for(model_id, revision, dtype, quantization)
        qualified = q.qualificationID if q else None
    return {
        "lensID": record.lensID,
        "compatible": not problems,
        "problems": problems,
        "qualified": qualified is not None,
        "qualificationID": qualified,
        # Absent dtype/revision is NOT a match — say so rather than implying
        # the runtime merely lacks a qualification it could inherit.
        "runtimeResolved": bool(revision and dtype),
    }
