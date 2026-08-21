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
"""

from __future__ import annotations

import os

from .importer import LENS_REPO, entry_for, supported_models
from .schemas import JLensError


def patterns_for(model_id: str) -> list[str]:
    """Repo-relative globs for exactly one model's lens folder.

    Load-bearing, not an optimization: ``neuronpedia/jacobian-lens`` holds 36
    models totalling ~57 GB, and one lens folder is a few GB. An unscoped fetch
    is the difference between a working verb and a filled group quota.
    """
    return [entry_for(model_id)["folder"] + "/*"]


def expected_files(model_id: str) -> list[str]:
    entry = entry_for(model_id)
    return [os.path.join(entry["folder"], entry["tensor"]),
            os.path.join(entry["folder"], entry["config"])]


def verify_landed(model_id: str, snapshot: str) -> None:
    """Confirm the requested tensor and config are actually present.

    ``allow_patterns`` matching nothing returns **successfully** over an empty
    cache, so a wrong folder name reads as a clean acquisition. Without this
    check the first sign of trouble would be an import failure much later, with
    nothing pointing at the table row that is wrong.
    """
    missing = [rel for rel in expected_files(model_id)
               if not os.path.exists(os.path.join(snapshot, rel))]
    if missing:
        raise JLensError(
            f"acquisition for '{model_id}' fetched nothing at {missing} — an "
            f"allow_patterns match of zero files returns success, so this is "
            f"almost certainly a wrong folder or filename in the supported "
            f"table (§3.1). Verify them against the repository tree.")


def acquire(model_id: str, *, log=None, revision: str | None = None,
            cancelled=None) -> str:
    """Fetch one model's lens into the HF cache; return the snapshot path.

    Public/MIT repository, so unlike the gated Gemma weights this needs no HF
    token — but it does need egress, and the underlying installer already
    reports which failure it was.
    """
    from ..api import model_install

    if model_id not in supported_models():
        raise JLensError(
            f"'{model_id}' has no supported lens — Gemma-only, "
            f"{supported_models()}")
    emit = log or (lambda _m: None)
    patterns = patterns_for(model_id)
    emit(f"acquiring lens for {model_id} from {LENS_REPO} "
         f"(scoped to {patterns[0]}; the repo is ~57 GB unscoped)")
    snapshot = model_install.run_install(
        LENS_REPO, revision, emit, cancelled=cancelled, allow_patterns=patterns)
    verify_landed(model_id, snapshot)
    emit(f"lens bytes present under {snapshot}")
    return snapshot
