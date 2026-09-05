"""Model acquisition, resident dtype admission and resolved revision pinning.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
from contextlib import contextmanager
from ..steering import model_loader
from . import manifest as _dep_manifest
from . import run_artifacts as _dep_run_artifacts


def _effective_dtype(manifest: _dep_manifest.Manifest, dtype: str) -> str:
    """The dtype a study's model should load at: the MANIFEST PIN when there
    is one, else whatever the caller asked for ("auto" by default).

    The pin wins because it is part of the frozen recipe and the caller's
    flag is not. A caller that asks for something else explicitly is refused
    rather than silently overridden — a run whose precision differs from its
    manifest is exactly the false-provenance case this key exists to close.
    """
    pinned = (manifest.dtype or "").strip()
    if not pinned:
        return dtype
    requested = (dtype or "").strip().lower()
    if requested and requested not in ("auto",):
        if model_loader.normalize_dtype(requested) != \
                model_loader.normalize_dtype(pinned):
            raise RuntimeError(
                f"study '{manifest.name}' pins dtype '{pinned}' but this run "
                f"asked for '{dtype}' — the pin is part of the frozen recipe. "
                "Drop the --dtype flag to honor it, or duplicate the study to "
                "measure at a different precision")
    return pinned


def _load_model(manifest: _dep_manifest.Manifest, dtype: str,
                device: str | None = None) -> model_loader.SteeredModel:
    return model_loader.load(manifest.model_id, revision=manifest.model_revision,
                             dtype=_effective_dtype(manifest, dtype),
                             device=device)


@contextmanager
def _acquire_model(manifest: _dep_manifest.Manifest, dtype: str, device: str | None,
                   model_provider):
    """Yield the pinned model, holding its registry slot lock when a provider is
    given (the API path — one lock authority per model). The CLI path has no
    provider and loads a private copy in a single process, where no other
    forward pass contends for the same object.

    A manifest `dtype` pin is passed through to the provider so a fresh load
    honors it. When the model is ALREADY resident at a different precision
    the acquire refuses: a loaded model's dtype cannot be changed, and
    running the study at the resident precision while the manifest claims
    another is the false pin this key exists to prevent.
    """
    effective = _effective_dtype(manifest, dtype)
    if model_provider is not None:
        kwargs = {"dtype": effective} if manifest.dtype else {}
        with model_provider(manifest.model_id, manifest.model_revision,
                            **kwargs) as model:
            _assert_resident_dtype_matches(manifest, model)
            yield model
    else:
        yield _load_model(manifest, dtype, device)


def _assert_resident_dtype_matches(manifest: _dep_manifest.Manifest, model) -> None:
    """Refuse a study whose pinned dtype is not what the served model runs
    in. Only fires when the manifest pins one — an unpinned study keeps the
    historical behaviour of taking whatever the registry loaded."""
    pinned = (manifest.dtype or "").strip()
    if not pinned:
        return
    actual = _dep_run_artifacts._actual_dtype(model)
    if not actual:
        return
    want = model_loader.normalize_dtype(pinned)
    if want is not None and model_loader.normalize_dtype(actual) != want:
        raise RuntimeError(
            f"study '{manifest.name}' pins dtype '{pinned}' but the served "
            f"model '{manifest.model_id}' is resident as '{actual}' — a "
            "loaded model's precision cannot be changed. Unload it (or "
            "restart the server) so it loads at the pinned precision")


def _pin_model_revision(name: str, manifest: _dep_manifest.Manifest, model, root,
                        _log) -> _dep_manifest.Manifest:
    """Mirror Swift ``ExperimentTasks.loadContainer(pinning:)``: a DRAFT
    manifest with no pinned revision gets the revision the loaded model
    actually resolved written back BEFORE any artifacts persist.

    This is what keeps promote's identity firewall honest instead of
    false-refusing: ``_persist_vectors`` stamps ``model.revision`` (the
    concrete commit ``model_loader.load`` resolves for a revision-less load)
    into every sidecar, while ``recipe_identity.required_identity`` reads the
    MANIFEST's revision — a null manifest pin therefore never matched the
    experiment's own extraction artifacts (the 2026-07-14 sweep→promote bug).
    Frozen manifests are immutable: a legacy one with no pin runs whatever
    the model resolved, loudly (freeze gates on the pin, so such manifests
    predate the gate)."""
    resolved = getattr(model, "revision", None)
    if manifest.model_revision or not resolved:
        return manifest
    # A revision identifies a commit in ONE repository. Writing the loaded
    # model's revision into a manifest declaring a different model produces a
    # pin that is silently wrong — a 4B commit on a 27B id — and it persists
    # into every freeze and bundle. For a mixed-model panel a single
    # manifest-level revision is meaningless anyway; the per-turn records
    # carry each turn's real revision.
    loaded_id = getattr(model, "model_id", None)
    if loaded_id and loaded_id != manifest.model_id:
        _log(f"not pinning revision {resolved[:12]}…: it belongs to "
             f"'{loaded_id}', but '{name}' declares '{manifest.model_id}'. "
             "Each turn records the revision it actually ran on.")
        return manifest
    if manifest.status != "draft":
        _log(f"⚠︎ '{name}' is {manifest.status} without a pinned model "
             f"revision — this run used {resolved[:12]}…")
        return manifest
    from . import experiment_store
    experiment_store.pin_model_revision(name, resolved, root)
    _log(f"pinned model revision {resolved[:12]}… into '{name}' "
         "(resolved at model load)")
    return _dep_manifest.Manifest.load(name, root)
