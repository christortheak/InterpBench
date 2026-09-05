"""Execution provenance stamps and cross-substrate/configuration advisories.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
import hashlib
import json
import os
import sys
import torch
from . import system_prompt as system_prompt_mod
from . import manifest as manifest_module
from . import run_artifacts


def sampling_metadata(model, temperature: float) -> dict:
    """Per-record sampling/substrate fields: the effective temperature/top_p/
    top_k actually used (resolved against the checkpoint), plus dtype/device.
    Stamped on every generation so a reader never has to guess the distribution."""
    from .generate import resolve_sampling
    s = resolve_sampling(model, temperature)
    return {
        "temperature": s["temperature"], "doSample": s["doSample"],
        "topP": s["topP"], "topK": s["topK"],
        "dtype": run_artifacts.model_dtype(model), "device": str(getattr(model, "device", "")),
        "engine": "python-hf-transformers",
    }


def write_substrate(model, run_directory: str, sampling: dict) -> None:
    """Write substrate.json: engine/library/GPU provenance for the run, so cross-
    engine claims can distinguish MLX from HF and pin exact versions (plan §8)."""
    import platform
    import sys
    import transformers
    info = {
        "engine": "python-hf-transformers",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": (torch.version.cuda if torch.cuda.is_available() else None),
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "modelID": model.model_id,
        "modelRevision": model.revision,
        "dtype": sampling.get("dtype"),
        "device": sampling.get("device"),
        "sampling": sampling,
    }
    with open(os.path.join(run_directory, "substrate.json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, sort_keys=True)


def preview_line(text: str, limit: int = 160) -> str:
    """Single-line truncated preview of a generation for the live job log (so
    decoherence is visible while a sweep runs, not after the grid lands).
    Whitespace runs — newlines included — collapse to single spaces; text
    longer than ``limit`` characters is cut there and marked with an ellipsis.
    Slicing a Python ``str`` operates on code points, so multi-byte characters
    are never split mid-character."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "…"


def advise_cross_substrate(manifest: manifest_module.Manifest, run_directory: str | None,
                            root, _log, *, write_file: bool) -> None:
    """WS7.1 loud, non-blocking study-run-start advisory: when this
    experiment's scope-matched validate evidence came from the OTHER engine,
    say so in the run log (and durably in the run directory) instead of
    silently running on a substrate whose vectors were never validated here.
    Never a refusal — freeze already enforces the same-substrate evidence
    gate; this catches the workspace that moved engines after freezing."""
    from . import experiment_store
    try:
        advisory = experiment_store.cross_substrate_validation_advisory(
            manifest.validation_scope_hash(), root)
    except Exception:  # the advisory must never sink a run
        return
    if not advisory:
        return
    _log(f"ADVISORY: {advisory}")
    if not write_file or not run_directory:
        return
    path = os.path.join(run_directory, "advisories.txt")
    if os.path.exists(path):  # resumed run: the creation stamp stands
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(advisory + "\n")


def advise_dependency_lock_drift(run_directory: str | None, _log, *,
                                  write_file: bool) -> None:
    """WP6 R1 loud, non-blocking run-start advisory: the torch/transformers
    this process imported differ from what the committed platform lock pins.

    Advisory by design, never a gate. Refusing here would kill a queued
    cluster job over a resolution difference the researcher may have made
    deliberately (post-submit drift policy: continue loudly + stamp). The
    STAMP is the durable half — config.json's ``pythonEnvironment`` records
    what actually ran regardless of whether anyone read this line. A site
    that installed its own torch build of the LOCKED version does not trip
    this: the comparison ignores the local segment (``+cu128``)."""
    from ..python_environment import lock_drift
    from .run_config import run_platform
    try:
        drift = lock_drift(run_platform())
    except Exception:  # the advisory must never sink a run
        return
    if not drift:
        return
    advisory = ("dependency drift vs the committed platform lock — "
                + "; ".join(drift)
                + ". Not a refusal: the run's config.json stamps what "
                  "actually ran (pythonEnvironment).")
    _log(f"ADVISORY: {advisory}")
    if not write_file or not run_directory:
        return
    # Append, not truncate: the cross-substrate advisory may already own this
    # file for this run. `write_file` is False on a resume, so the creation
    # stamp still does not accumulate a line per restart.
    with open(os.path.join(run_directory, "advisories.txt"), "a",
              encoding="utf-8") as handle:
        handle.write(advisory + "\n")


def advise_system_prompt_divergence(arms, run_directory: str | None, _log, *,
                                     write_file: bool) -> None:
    """Comparability advisory (2026-08-24 ruling): the arms of THIS run are
    not all armed with the same effective system content.

    Same shape as :func:`advise_dependency_lock_drift` — an ``ADVISORY:``
    line at the verb's start, appended to the run directory's
    ``advisories.txt``, never a refusal and never a change to the numbers.

    Silent in the universal case. Before composition landed, every arm of
    every real run shared one ``systemPromptHash`` (empty personas fell back
    to the study frame); after it, they still do unless a researcher gives an
    agent a persona — which is precisely the design decision this line exists
    to make visible rather than to prevent.

    ``arms`` is an ordered ``(name, effective system prompt)`` sequence in the
    run's own emission order, so the advisory reads in the order the run
    executes.
    """
    advisory = system_prompt_mod.divergence_advisory(arms)
    if not advisory:
        return
    _log(f"ADVISORY: {advisory}")
    if not write_file or not run_directory:
        return
    # Append, like the lock-drift and case-family advisories: the
    # cross-substrate advisory may already own this file for this run.
    try:
        with open(os.path.join(run_directory, "advisories.txt"), "a",
                  encoding="utf-8") as handle:
            handle.write(advisory + "\n")
    except OSError:  # the advisory must never sink a run
        pass


def sha256_text(text: str | None) -> str | None:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
