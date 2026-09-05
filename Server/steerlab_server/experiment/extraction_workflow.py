"""Coordinate extraction, vector persistence and extraction diagnostics in one model scope.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
import json
import os
from typing import Callable
from . import paths
from . import layer_resolution
from . import manifest as manifest_module
from . import model_resources
from . import run_artifacts
from . import study_admission
from . import vector_materialization


def extract(name: str, root: str | None = None, dtype: str = "auto",
            device: str | None = None, *, model_provider=None,
            should_cancel: Callable[[], bool] | None = None, log=None) -> str:
    _log = log or print
    manifest = manifest_module.Manifest.load(name, root)
    study_admission.verify_or_warn(manifest, root)
    _advise_inert_declarations(manifest, _log)
    with model_resources.acquire_model(manifest, dtype, device, model_provider) as model:
        manifest = model_resources.pin_model_revision(name, manifest, model, root, _log)
        bundles = vector_materialization.extract_all(model, manifest, root)
        run_directory = paths.make_unique_run_directory(f"exp-{name}-extract", root)
        run_artifacts.write_config_snapshot(manifest, run_directory, "extract", model=model,
                               root=root, log=_log)
        vector_materialization.persist_vectors(bundles, manifest, model, run_directory)
        _write_reading_position_diagnostics(bundles, run_directory, _log)
        _write_logit_lens_vocabulary(bundles, manifest, model,
                                     run_directory, _log)
    _log(f"extracted {len(bundles)} concept vectors → {run_directory}")
    return run_directory


def _advise_inert_declarations(manifest: manifest_module.Manifest, _log) -> None:
    """Loud, non-blocking extract-time advisory (2026-08-24 field finding):
    say when this manifest's chat-context declarations cannot reach the
    extraction that is about to run.

    Fires only when EVERY extracting concept renders raw — a mixed recipe does
    reach the template somewhere, and the per-concept rendering is already
    stamped. Pinned-artifact concepts materialize bytes rather than extracting,
    so they do not vote.

    Never a gate. The declarations are legal; what was not survivable is
    silence about them: two experiments differing only in the thinking flag
    produced byte-identical vectors, and the comparison read as a null result
    rather than as a measurement that never happened.
    """
    from ..steering import extractor as _extractor

    renderings = [c.options.extraction_rendering for c in manifest.concepts
                  if not c.is_pinned_artifact]
    if not renderings or not all(
            r is None or r.is_raw for r in renderings):
        return
    advisory = _extractor.inert_declaration_advisory(
        None,
        qwen_thinking_enabled=manifest.qwen_thinking_enabled,
        prompt_mode=manifest.prompt_mode,
        system_prompt=manifest.system_prompt,
        reasoning_effort=getattr(manifest, "reasoning_effort", None))
    if advisory:
        _log(f"ADVISORY: {advisory}")


def _write_reading_position_diagnostics(bundles, run_directory: str, _log) -> None:
    """Standing per-concept diagnostic for any DEPARTURE from the legacy
    default recipe (METHODS appendix): the per-layer cosine between this
    recipe's vectors and vectors extracted the legacy way — raw rendering,
    last token.

    Fires for a non-last-token reading position (free: the baseline reads
    from the SAME forward passes via a second recorder in the same hook
    session) AND for any non-raw extraction rendering (not free: a different
    tokenization needs its own passes, which the report flags as
    ``extraForwardPasses``). Either way the justification for a departure is
    the measured gap, not a citation — the two renderings were measured
    cosine ≈ 0.18 apart mid-network while both probed near-perfectly (ledger
    §26), which is exactly the kind of number a METHODS section has to carry.

    Written beside the vectors, never into the sidecar (which is a
    cross-engine artifact contract)."""
    diagnostics = {name: b.reading_position_diagnostic
                   for name, b in bundles.items()
                   if b.reading_position_diagnostic}
    if not diagnostics:
        return
    path = os.path.join(run_directory, "reading-position-diagnostics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True)
    for name, diag in sorted(diagnostics.items()):
        against = diag["comparedTo"]
        if diag.get("comparedToRendering"):
            against += f" / {diag['comparedToRendering']} rendering"
        _log(f"reading-position diagnostic — {name}: cosine vs "
             f"{against} min {diag['min']:.3f} / median "
             f"{diag['median']:.3f} / max {diag['max']:.3f} over "
             f"{len(diag['perLayerCosine'])} layers")


#: How many logit-lens tokens the extract-time vocabulary check records per
#: direction per depth. Pinned to the same 10 ``validate`` uses, so the two
#: reports can be read side by side (see the logit-lens block in
#: ``_validate_impl``).
LOGIT_LENS_VOCABULARY_TOP_K = 10


def _write_logit_lens_vocabulary(bundles, manifest: manifest_module.Manifest, model,
                                 run_directory: str, _log) -> None:
    """Per-direction logit-lens vocabulary, written at EXTRACT time.

    WHY HERE AND NOT ONLY IN VALIDATE (2026-08-25 ruling, the
    doctrine-vs-affect confound). A rendering × reading-position grid produces
    one direction per cell, and the question asked of every cell is what
    VOCABULARY that direction promotes: a cell whose top tokens are the
    subject's doctrinal nouns and a cell whose top tokens are affect words are
    not the same measurement, however similar their probe accuracies look.
    Reading that off the extraction itself means the answer exists before any
    sweep is spent on the cell.

    NEVER A GATE, and never inside the sidecar: the sidecar is a cross-engine
    artifact contract, so this goes beside the vectors in the run directory,
    exactly as the reading-position diagnostic does. A failure to project is
    recorded as a skip string and cannot sink an extraction.

    ENGINE ASYMMETRY, deliberate: the swift-mlx engine writes no equivalent at
    extract time. It has the same logit lens (``LogitLensReadable``) and runs
    it inside ``validate``; the grid this instrument serves runs on the server,
    and a second Swift writer would be an unused surface to keep in parity.

    The layers are the study's OWN declared validation depths
    (:func:`_validation_layer_resolutions`) — the same rule ``validate``
    resolves, so "its layer" means one thing in both reports rather than two.
    """
    from ..steering.extractor import logit_lens

    report: dict = {}
    for concept_name, bundle in sorted(bundles.items()):
        layer_count = bundle.vectors.layer_count
        if not layer_count:
            continue
        try:
            resolutions = layer_resolution.validation_layer_resolutions(
                manifest, concept_name, layer_count)
        except Exception as exc:  # noqa: BLE001 — a diagnostic never gates
            report[concept_name] = f"logit-lens vocabulary skipped: {exc}"
            continue
        depths: list = []
        for resolution in resolutions:
            try:
                lens = logit_lens(model, bundle.vectors, resolution.layer,
                                  top_k=LOGIT_LENS_VOCABULARY_TOP_K)
            except Exception as exc:  # noqa: BLE001 — same rule as validate's
                depths.append(f"logit-lens skipped: {exc}")
                continue
            depths.append({
                "layer": lens.layer,
                "layerResolution": layer_resolution.resolution_block(resolution),
                "topPositive": [{"tokenID": t.token_id, "token": t.token,
                                 "logit": t.logit} for t in lens.top_positive],
                "topNegative": [{"tokenID": t.token_id, "token": t.token,
                                 "logit": t.logit} for t in lens.top_negative],
            })
            top = ", ".join(t.token for t in lens.top_positive[:5])
            _log(f"logit-lens vocabulary — {concept_name} @ L{lens.layer}: {top}")
        if depths:
            report[concept_name] = depths
    if not report:
        return
    path = os.path.join(run_directory, "logit-lens-vocabulary.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
