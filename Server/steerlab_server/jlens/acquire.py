"""Acquire the published lens bytes into the HF cache.

Acquisition and import are separate operations (plan §11.0.1): this one is
machine-scoped, needs egress, and is repeatable; import is workspace-scoped,
offline, and is the thing under test.

Acquisition extends ``api/model_install.py`` rather than reimplementing it.
That module already owns everything hard about pulling several GB onto a
cluster node — the one verb allowed online, byte progress, a cancel that
terminates the child's process group, token/license/egress triage, a
self-healing Xet retry — and none of it is worth a second copy.

What is NOT reused is the taxonomy. A lens acquired *as a model* would be
advertised as a model, which is precisely the picker defect this project hit on
2026-07-27. Lenses never enter the model catalog; they are found through the
lens store.

Two shapes of acquisition (2026-09-05). A CURATED model's folder is known from
the table and is fetched in one scoped pull. Any OTHER model with a published
lens is fetched in two: first the published ``config.yaml`` files (forty-odd
small files), which say which folder names the model, then that folder. The
folder is never guessed from the model id — upstream folder names are not
model ids — and the pull is scoped either way.
"""

from __future__ import annotations

import glob
import os

from .importer import (CONFIG_FILE, LENS_REPO, PUBLISHED_CONFIG_GLOB, SUPPORTED,
                       TENSOR_SUFFIX, entry_for, is_curated, resolve_published,
                       supported_models)
from .schemas import JLensError


def patterns_for(model_id: str, *, snapshot: str | None = None) -> list[str]:
    """Repo-relative globs for exactly one model's lens folder.

    Load-bearing, not an optimization: ``neuronpedia/jacobian-lens`` holds
    ~40 models totalling tens of GB, and one lens folder is a few GB. An
    unscoped fetch is the difference between a working verb and a filled
    group quota. For an uncurated model the folder comes from the cached
    published configs (``snapshot``), so this is answerable only after
    :func:`acquire`'s first phase — see :func:`acquire`.
    """
    return [entry_for(model_id, snapshot=snapshot)["folder"] + "/*"]


def expected_files(model_id: str, *, snapshot: str | None = None) -> list[str]:
    """The files a landed acquisition must contain, repo-relative.

    A curated row names its tensor; a published folder is verified by its
    config plus exactly one ``*_jacobian_lens.pt`` (:func:`verify_landed`).
    """
    entry = entry_for(model_id, snapshot=snapshot)
    files = [os.path.join(entry["folder"], entry["config"])]
    if entry.get("tensor"):
        files.insert(0, os.path.join(entry["folder"], entry["tensor"]))
    return files


def verify_landed(model_id: str, snapshot: str) -> None:
    """Confirm the requested tensor and config are actually present.

    ``allow_patterns`` matching nothing returns **successfully** over an empty
    cache, so a wrong folder name reads as a clean acquisition. Without this
    check the first sign of trouble would be an import failure much later, with
    nothing pointing at the table row that is wrong.
    """
    entry = entry_for(model_id, snapshot=snapshot)
    folder = os.path.join(snapshot, entry["folder"])
    missing = [rel for rel in expected_files(model_id, snapshot=snapshot)
               if not os.path.exists(os.path.join(snapshot, rel))]
    if missing:
        raise JLensError(
            f"acquisition for '{model_id}' fetched nothing at {missing} — an "
            f"allow_patterns match of zero files returns success, so this is "
            f"almost certainly a wrong folder or filename "
            f"{'in the curated table (§3.1)' if is_curated(model_id) else 'in the published folder'}. "
            f"Verify them against the repository tree.")
    if not is_curated(model_id):
        tensors = sorted(glob.glob(os.path.join(folder, "*" + TENSOR_SUFFIX)))
        if len(tensors) != 1:
            raise JLensError(
                f"acquisition for '{model_id}' landed {len(tensors)} "
                f"'*{TENSOR_SUFFIX}' file(s) in {entry['folder']} "
                f"({[os.path.basename(t) for t in tensors]}) — an import "
                f"needs exactly one, so this folder needs a curated row that "
                f"names the tensor")


def acquire(model_id: str, *, log=None, revision: str | None = None,
            cancelled=None) -> str:
    """Fetch one model's lens into the HF cache; return the snapshot path.

    Public/MIT repository, so unlike the gated Gemma weights this needs no HF
    token — but it does need egress, and the underlying installer already
    reports which failure it was.
    """
    from ..api import model_install

    emit = log or (lambda _m: None)
    if not model_id or "/" not in model_id:
        raise JLensError(
            f"'{model_id}' is not a Hugging Face model id (owner/name) — the "
            f"published configs name models by that id, so nothing else can "
            f"be looked up")
    if is_curated(model_id):
        patterns = patterns_for(model_id)
    else:
        emit(f"{model_id} has no curated row — fetching the published lens "
             f"configs ({PUBLISHED_CONFIG_GLOB}, a few KB each) from "
             f"{LENS_REPO} to find its folder")
        configs = model_install.run_install(
            LENS_REPO, revision, emit, cancelled=cancelled,
            allow_patterns=[PUBLISHED_CONFIG_GLOB])
        entry = resolve_published(model_id, snapshot=configs)
        emit(f"published lens for {model_id}: {entry['folder']} "
             f"(fitted on {entry.get('corpus') or 'an unrecorded corpus'}, "
             f"{entry.get('promptsFitted') or '?'} prompts) — its tier is "
             f"NOT published; declare it at import with --tier")
        patterns = [entry["folder"] + "/*"]
    emit(f"acquiring lens for {model_id} from {LENS_REPO} "
         f"(scoped to {patterns[0]}; the repo is tens of GB unscoped)")
    snapshot = model_install.run_install(
        LENS_REPO, revision, emit, cancelled=cancelled, allow_patterns=patterns)
    verify_landed(model_id, snapshot)
    emit(f"lens bytes present under {snapshot}")
    return snapshot


__all__ = ["acquire", "expected_files", "patterns_for", "verify_landed",
           "supported_models", "SUPPORTED", "CONFIG_FILE"]
