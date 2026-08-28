"""Experiment task orchestration (parallel to Swift ``ExperimentTasks``):
``extract``, ``validate``, ``sweep``, ``run`` over a pinned manifest.

Each task loads the model **at the pinned revision**, re-derives vectors
deterministically, and writes an immutable ``runs/<stamp>-…`` directory stamped
with the experiment's content hash — so results trace to the exact stimuli that
produced them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Callable

import torch

from .. import memory_diagnostic
from ..steering import model_loader, vector_math as vm
from ..steering import residual_norm_convention as norm_convention
from ..steering.extractor import ExtractionOptions as CoreExtractionOptions, extract as core_extract
from ..steering.stimulus_set import StimulusSet, load_texts
from ..steering import vector_store
from ..steering.vector_store import ConceptVectors, SteeringVectorSidecar, save as save_vectors
from . import judging_custody
from . import judicial, lifecycle_gates, paths, prompt_render, recipe_identity
from . import response_format
from . import resume as resume_mod
from . import run_epoch
from . import sharding as sharding_mod
from . import system_prompt as system_prompt_mod
from . import choice_deltas
from . import turn_endpoint
from .generate import CellInjection, generate
from . import manifest as manifest_mod
from .manifest import JudgeRef, Manifest, VariantCondition
from .run_config import write_run_config
from .run_status import RunStatus, heal_after_completion
from .scoring import MarkerRubric, distinct_bigram_ratio, word_count


@dataclass
class ConceptVectorBundle:
    vectors: ConceptVectors
    residual_norm_per_layer: list[float]
    residual_norm_source: str
    stimulus_hash: str
    # WHICH AVERAGING RULE produced ``residual_norm_per_layer`` — carried to
    # the sidecar's ``residualNormConvention`` stamp. None for a bundle whose
    # norms came from a LEGACY artifact that never recorded one; never
    # invented (see :mod:`steering.residual_norm_convention`).
    residual_norm_convention: str | None = None
    # WHICH RENDERING the denominator corpus was tokenized under — carried to
    # the sidecar's ``residualNormRendering`` stamp. "raw" (or None) is legacy
    # and stamps nothing.
    residual_norm_rendering: str | None = None
    # The requested reading position AND where it resolved, per sequence
    # shape — carried to the sidecar's ``readingPositionResolution``. None
    # when the position's label already implies its index (the legacy pair).
    reading_position_resolution: dict | None = None
    # The rendering block to stamp. None = take the manifest concept's
    # declaration (the ordinary case: this run rendered it). A MATERIALIZED
    # pinned artifact sets it from the source sidecar, because the copy's
    # rendering is the one that produced the bytes, not the one this study
    # would have used.
    extraction_rendering: dict | None = None
    # Grand-mean extractions only: the FULL comparison population actually
    # read (concept name → stories.jsonl SHA-256), stamped into the sidecar
    # so the artifact can prove its recipe identity. None for paired methods.
    grand_mean_population: dict[str, str] | None = None
    # Pooled readings only: the per-layer cosine against last-token vectors
    # from the same passes (the standing reading-position diagnostic —
    # run-dir output, deliberately NOT part of the sidecar contract).
    reading_position_diagnostic: dict | None = None
    # Per-layer neutral-corpus residual mean (the ablation "carrier"
    # estimate) — persisted into the artifact so ablation paths can center
    # against it; None when the manifest pins no neutral corpus.
    neutral_mean_per_layer: list[list[float]] | None = None
    # designatedReference extractions only: {"name", "hash"} of the
    # reference stories actually subtracted (sidecar provenance).
    designated_reference: dict | None = None
    # Artifact-pinned concepts only: the SOURCE artifact this bundle was
    # materialized from — {"path", "sha256TensorHash", "sha256SidecarHash",
    # "sourceMethod", "sourceConcept", "extractionDate"} — stamped into the
    # emitted sidecar as ``pinnedFrom`` so the materialized copy always names
    # the bytes it came from. None for every derived recipe.
    pinned_from: dict | None = None
    # Artifact-pinned concepts only: the neutral corpus the artifact's
    # residual norms were measured on (carried from the source sidecar, since
    # the norms are carried too). None = use the manifest's pin, as before.
    neutral_corpus_hash: str | None = None
    # Artifact-pinned Gemma Scope imports only: the SOURCE sidecar's SAE
    # identity keys — ``gemmascopeConvention`` and its two scaling numbers,
    # plus the ``gemmascopeSource`` provenance block (open-issues #14).
    #
    # These describe the VECTOR, not the recipe that produced this run, so a
    # materialized copy that drops them is claiming less than it knows: every
    # w7 run warned "Gemma Scope SAE import without a gemmascopeConvention
    # stamp: pre-convention import" about copies of imports that ARE
    # post-convention, and `promote` then pinned the unstamped copy into the
    # agent. None for every non-SAE bundle, which keeps their sidecars
    # byte-identical (`to_dict` drops None).
    gemmascope_convention: str | None = None
    raw_decoder_norm: float | None = None
    gemmascope_target_norm: float | None = None
    gemmascope_source: dict | None = None
    # Whether the SOURCE artifact's ``layerCount`` is the MODEL's depth. A
    # materialized copy has the source's rows, so it inherits the source's
    # answer — and a PARTIAL source (a reader-derived direction, zeros below
    # its layer) must not launder itself into a full-looking copy just because
    # the copy's own ``extractionMethod`` is ``pinnedArtifact``. None on every
    # full-depth bundle, which keeps their sidecars byte-identical
    # (``to_dict`` drops None).
    covers_model_depth: bool | None = None
    # Artifact-pinned MIRRORED poles only: the SOURCE sidecar's mirror stamps
    # (``pole_mirror``). ``polesSwappedFromSource`` qualifies the stimulus
    # hash — the copy carries the PARENT's order-sensitive hash, and dropping
    # the qualifier would claim the mirrored concept's own files were read in
    # their own order — and ``negatedFrom`` names the artifact whose tensors
    # were sign-flipped. Same claiming-less-than-it-knows defect class as the
    # SAE keys above; None for every non-mirrored bundle, which keeps their
    # sidecars byte-identical (``to_dict`` drops None).
    poles_swapped_from_source: bool | None = None
    negated_from: dict | None = None


def _effective_dtype(manifest: Manifest, dtype: str) -> str:
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


def _load_model(manifest: Manifest, dtype: str,
                device: str | None = None) -> model_loader.SteeredModel:
    return model_loader.load(manifest.model_id, revision=manifest.model_revision,
                             dtype=_effective_dtype(manifest, dtype),
                             device=device)


@contextmanager
def _acquire_model(manifest: Manifest, dtype: str, device: str | None,
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


def _assert_resident_dtype_matches(manifest: Manifest, model) -> None:
    """Refuse a study whose pinned dtype is not what the served model runs
    in. Only fires when the manifest pins one — an unpinned study keeps the
    historical behaviour of taking whatever the registry loaded."""
    pinned = (manifest.dtype or "").strip()
    if not pinned:
        return
    actual = _actual_dtype(model)
    if not actual:
        return
    want = model_loader.normalize_dtype(pinned)
    if want is not None and model_loader.normalize_dtype(actual) != want:
        raise RuntimeError(
            f"study '{manifest.name}' pins dtype '{pinned}' but the served "
            f"model '{manifest.model_id}' is resident as '{actual}' — a "
            "loaded model's precision cannot be changed. Unload it (or "
            "restart the server) so it loads at the pinned precision")


def _model_dtype(model) -> str:
    try:
        return str(next(model.model.parameters()).dtype)
    except (StopIteration, AttributeError):
        return ""


def _actual_dtype(model) -> str | None:
    """Canonical spelling of the dtype a loaded model ACTUALLY runs in, for
    the run stamp — or None when no model was loaded.

    Prefers the wrapper's own stamp (`model_loader.load` records it at load,
    already `torch.`-stripped); falls back to reading the parameters, which
    also works for the test fakes that never went through `load`.
    """
    if model is None:
        return None
    stamped = getattr(model, "dtype", None)
    if stamped:
        return str(stamped).removeprefix("torch.")
    raw = _model_dtype(model)
    return raw.removeprefix("torch.") or None


def _sampling_metadata(model, temperature: float) -> dict:
    """Per-record sampling/substrate fields: the effective temperature/top_p/
    top_k actually used (resolved against the checkpoint), plus dtype/device.
    Stamped on every generation so a reader never has to guess the distribution."""
    from .generate import resolve_sampling
    s = resolve_sampling(model, temperature)
    return {
        "temperature": s["temperature"], "doSample": s["doSample"],
        "topP": s["topP"], "topK": s["topK"],
        "dtype": _model_dtype(model), "device": str(getattr(model, "device", "")),
        "engine": "python-hf-transformers",
    }


def _write_substrate(model, run_directory: str, sampling: dict) -> None:
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


def _observe_cancel(should_cancel, log, where: str) -> bool:
    """True (and logs where) when a cancellation has been requested. Called at
    loop boundaries so a long task stops promptly instead of only at the end."""
    if should_cancel is not None and should_cancel():
        log(f"cancellation observed at {where}")
        return True
    return False


class TaskCancelled(Exception):
    """Internal sentinel: a cancellation was observed inside a generation loop
    (dev prompts, battery items, choice rows, judge calls). The loop that owns
    partial-output bookkeeping catches it and takes the SAME path as a
    concept/layer-boundary cancel — completed rows kept, partial CSV +
    recommendations written — so the run directory stays consistent."""


def _cancel_checkpoint(should_cancel, log, where: str) -> None:
    """The per-generation twin of ``_observe_cancel``: called between
    generations inside the inner loops so cancel latency is at most one
    generation, and raises ``TaskCancelled`` for the owning loop to catch."""
    if should_cancel is not None and should_cancel():
        if log is not None:
            log(f"cancel observed at {where} — stopping after the current generation")
        raise TaskCancelled(where)


def _preview_line(text: str, limit: int = 160) -> str:
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


def _verify_or_warn(manifest: Manifest, root: str | None) -> None:
    """Warn on any violation; RAISE only for a frozen manifest.

    KNOWN CROSS-ENGINE DIVERGENCE (characterised 2026-07-26, deliberately
    unresolved). Swift's ``loadVerified`` throws unconditionally, so the same
    DRAFT — stimuli drifted from their pins, a variant artifact missing, an
    instrument declared in a configuration that cannot read it — refuses on
    the Mac and runs on the server.

    One of the two engines is wrong and it is not obvious which. Against
    changing this: a draft run is exploratory by construction (freeze is what
    makes a result citable), and refusing every draft whose pins have drifted
    would block iteration that is legitimately messy. For changing it: a
    drifted pin means the run's stamped provenance is false even when the
    numbers are only being eyeballed.

    Measured cost of switching the server to Swift's rule: 50 tests across 11
    files currently depend on the permissive behaviour — nearly all through
    fixture shortcuts (variant artifacts never written to disk, placeholder
    stimulus hashes, studies with neither concepts nor variants). That is
    mechanical to fix, but the BEHAVIOURAL change would also refuse live
    draft runs on the cluster that succeed today, so it is a policy decision
    for the researcher rather than a cleanup.
    """
    violations = manifest.verify(root)
    for v in violations:
        print(f"VERIFY WARNING: {v}")
    if violations and manifest.status == "frozen":
        # WP0 step 8: typed `pinDrift`. The prose is byte-identical to what
        # this site has always raised; the gate id and the runnable repair are
        # strictly additive, and `LifecycleError` IS a `RuntimeError` so every
        # existing catch still catches.
        raise lifecycle_gates.refusing(
            lifecycle_gates.PIN_DRIFT,
            f"experiment '{manifest.name}' is frozen but failed verification: {violations}",
            repair=(f"steerlab-server experiment verify {manifest.name} "
                    "(names every drifted pin) ; then restore the named files "
                    "to their pinned bytes — a frozen pin is never re-pinned: "
                    f"duplicate {manifest.name} on the Mac to change it"))


def _restore_self_pinned_revision(name, manifest, ledger, root, _log,
                                  pipeline_run_directory=None):
    """Repair the pipeline continuation's self-inflicted epoch drift.

    Returns ``(manifest, live_hash, restored)``. The chain's own run stage
    pins the resolved model revision (``_pin_model_revision``) and stamps
    every artifact with the PINNED epoch, but each ``bundle execute``
    re-imports the bundle's unpinned manifest over the live one — so the
    continuation can find a live manifest identical to the ledger's epoch
    except for the missing pin. That is the same manifest, one write behind
    its own machinery: restore the pin (drafts only — the same rule
    ``_pin_model_revision`` applies), persist it, and continue.

    The pin is looked for in the CHAIN's own start snapshot first (the
    pipeline dir's ``experiment.json``, written after the chain's model
    load pinned the revision — it exists for every schema-2 chain), then in
    each completed stage's snapshot. Only an EXACT hash match after
    applying a snapshot revision counts: any other difference is real
    drift and returns ``restored=False`` with the live hash unchanged."""
    if manifest.model_revision:
        return manifest, manifest.content_hash(), False
    candidates = ([pipeline_run_directory] if pipeline_run_directory else [])
    candidates += [
        (entry or {}).get("runDirectory")
        for entry in (ledger.get("stageResults") or {}).values()]
    for run_dir in candidates:
        if not run_dir:
            continue
        snap = run_epoch.snapshot(run_dir)
        revision = getattr(snap, "model_revision", None) if snap else None
        if not revision:
            continue
        raw = dict(manifest.raw)
        raw["modelRevision"] = revision
        try:
            candidate = Manifest.from_dict(raw)
        except Exception:  # noqa: BLE001 - fall through to real-drift handling
            continue
        if candidate.content_hash() == ledger.get("experimentHash"):
            if manifest.status == "draft":
                from . import experiment_store
                experiment_store.pin_model_revision(name, revision, root)
                manifest = Manifest.load(name, root)
            else:
                manifest = candidate
            _log(f"restored model revision pin {revision[:12]}… (the chain "
                 "resolved it at model load; a later bundle import had "
                 "clobbered it) — the live manifest again matches the "
                 "ledger epoch")
            return manifest, manifest.content_hash(), True
    return manifest, manifest.content_hash(), False


def _stamp_pipeline_drift(pipeline_run_directory, ledger, live_hash) -> None:
    """Durable record of a continuation that proceeded past manifest drift
    (never silent): what the ledger expected, what was live, and when.

    Mutates the IN-MEMORY ledger too (merge repair, 2026-08-06): every
    later ``_stage_done`` rewrites the whole ledger from memory, so a stamp
    living only on disk would be clobbered by the first completed stage of
    the very continuation it records."""
    ledger.setdefault("epochDriftAtContinuation", []).append(
        {"ledgerHash": ledger.get("experimentHash"),
         "liveHash": live_hash})
    try:
        _write_pipeline_ledger(pipeline_run_directory, ledger)
    except OSError as exc:
        # The stamp is evidence, not a gate — a failed write must not kill
        # the continuation the policy exists to protect. It IS still logged.
        print(f"warning: could not stamp epochDriftAtContinuation: {exc}")


def _pin_model_revision(name: str, manifest: Manifest, model, root,
                        _log) -> Manifest:
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
    return Manifest.load(name, root)


def _extract_all(model: model_loader.SteeredModel, manifest: Manifest,
                 root: str | None) -> dict[str, ConceptVectorBundle]:
    neutral_texts = None
    if manifest.neutral_corpus_hash:
        try:
            neutral_texts = load_texts(paths.neutral_corpus_path(root)).texts
        except Exception:  # noqa: BLE001
            neutral_texts = None
    bundles: dict[str, ConceptVectorBundle] = {}
    for concept in manifest.concepts:
        if concept.is_pinned_artifact:
            # Not an extraction: a hash-verified materialization of pinned
            # bytes. Everything downstream (validate, sweep, run, promote)
            # sees an ordinary bundle and needs no special case.
            bundles[concept.name] = _materialize_pinned_artifact(
                manifest, concept, root)
            continue
        if concept.options.method.is_designated_reference:
            bundles[concept.name] = _extract_designated_reference(
                model, concept, root, neutral_texts)
            continue
        if concept.options.method.is_grand_mean:
            continue  # grand-mean concepts extract in one corpus pass below
        stimuli = StimulusSet.from_directory(paths.concept_directory(concept.name, root))
        options = CoreExtractionOptions(
            method=concept.options.method,
            reading_position=concept.options.reading_position,
            neutral_pc_count=concept.options.neutral_pc_count,
            extraction_rendering=concept.options.extraction_rendering)
        result = core_extract(model, stimuli, options, neutral_texts=neutral_texts)
        bundles[concept.name] = ConceptVectorBundle(
            vectors=result.vectors,
            residual_norm_per_layer=result.residual_norm_per_layer,
            residual_norm_source=result.residual_norm_source,
            residual_norm_convention=result.residual_norm_convention,
            residual_norm_rendering=result.residual_norm_rendering,
            stimulus_hash=stimuli.hash,
            reading_position_diagnostic=result.reading_position_diagnostic,
            reading_position_resolution=result.reading_position_resolution,
            neutral_mean_per_layer=result.neutral_mean_per_layer)
    bundles.update(_extract_grand_mean_bundles(model, manifest, root, neutral_texts))
    return bundles


def _sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _materialize_pinned_artifact(manifest: Manifest, concept,
                                 root) -> ConceptVectorBundle:
    """Verify an artifact-pinned concept's bytes and load them as a bundle.

    The firewall rule for a recipe concept is "the stimuli you pinned are the
    stimuli that were read"; for an artifact-pinned concept it is "the bytes
    you pinned are the bytes that steer". So BOTH hashes are re-checked here,
    at the moment of use, and a mismatch refuses loudly naming the file, the
    live hash and the pinned hash — the extraction path never stamps pinned
    provenance over drifted bytes (the same rule
    ``_extract_designated_reference`` enforces for stories).

    Also refused: an artifact extracted on another model or revision (a
    direction does not transfer), on another substrate (activations do not
    transfer between engines), or read at a different position than the
    manifest declares (held-out activations must be read where the vector
    was read).
    """
    from . import catalog
    block = concept.vector_artifact or {}
    rel = str(block.get("path") or "")
    if not rel:
        raise RuntimeError(
            f"concept '{concept.name}': method 'pinnedArtifact' with no "
            "vectorArtifact.path — there is nothing to materialize")
    base = paths.project_root() if root is None else root
    directory_rel, name = os.path.split(rel)
    directory = directory_rel if os.path.isabs(directory_rel) \
        else os.path.join(base, directory_rel)
    for suffix, pin_key, label in (
            (".safetensors", "sha256TensorHash", "vectors"),
            (".json", "sha256SidecarHash", "sidecar")):
        path = os.path.join(directory, f"{name}{suffix}")
        pinned = block.get(pin_key)
        if not pinned:
            raise RuntimeError(
                f"concept '{concept.name}': vectorArtifact pin is incomplete "
                f"— no {pin_key} for '{rel}{suffix}' (a half-pin certifies "
                "nothing)")
        if not os.path.isfile(path):
            raise RuntimeError(
                f"concept '{concept.name}': pinned {label} file "
                f"'{rel}{suffix}' is missing — restore the artifact, or "
                "re-attach")
        live = _sha256_file(path)
        if live != pinned:
            raise RuntimeError(
                f"concept '{concept.name}': pinned {label} file "
                f"'{rel}{suffix}' drifted from its pinned hash (have "
                f"{live}, pinned {pinned}) — restore the pinned bytes, or "
                "re-attach the artifact")

    vectors, sidecar = vector_store.load(directory, name)
    vector_store.require_native_substrate(sidecar, f"{rel}.safetensors")
    if sidecar.modelID != manifest.model_id:
        raise RuntimeError(
            f"concept '{concept.name}': pinned artifact '{rel}' was extracted "
            f"on model '{sidecar.modelID}', not this study's "
            f"'{manifest.model_id}' — a direction does not transfer between "
            "models")
    if (manifest.model_revision and sidecar.revision
            and sidecar.revision != manifest.model_revision):
        raise RuntimeError(
            f"concept '{concept.name}': pinned artifact '{rel}' was extracted "
            f"at revision {sidecar.revision}, not this study's pinned "
            f"{manifest.model_revision}")
    declared_reading = concept.options.reading_position.label
    # Reading position is exempt for a direction with NO SOURCE CONCEPT — an
    # OPTVEC vector (plan §6) or an imported Gemma Scope SAE decoder row. The
    # vector was never READ OUT of activations at a stimulus position (it was
    # optimized by backprop against a dataset, or lifted from a published
    # dictionary), so "where the vector was read" has no referent, and the
    # rule this check enforces ("held-out activations must be read where the
    # vector was read") is vacuous for a concept with no held-out activations.
    # Neither writer emits a readingPosition for exactly that reason; the
    # exemption is explicit so a sidecar that carries an incidental one cannot
    # refuse materialization over a value nothing measured.
    if (sidecar.readingPosition and sidecar.readingPosition != declared_reading
            and concept.effective_method.has_source_concept):
        raise RuntimeError(
            f"concept '{concept.name}': pinned artifact '{rel}' was read at "
            f"'{sidecar.readingPosition}' but the manifest declares "
            f"'{declared_reading}' — held-out activations must be read where "
            "the vector was read")
    return ConceptVectorBundle(
        vectors=vectors,
        residual_norm_per_layer=list(sidecar.residualNormPerLayer or []),
        residual_norm_source=sidecar.residualNormSource or "",
        residual_norm_convention=sidecar.residualNormConvention,
        # The materialized copy carries the SOURCE's norm provenance whole:
        # its denominator rendering and where its reading position landed
        # travel with the norms, exactly as the convention stamp does.
        residual_norm_rendering=sidecar.residualNormRendering,
        reading_position_resolution=sidecar.readingPositionResolution,
        extraction_rendering=sidecar.extractionRendering,
        # The artifact's own recorded stimulus identity — the same value
        # attach pinned, so the materialized copy claims exactly what the
        # manifest pins (and nothing the manifest never saw).
        stimulus_hash=sidecar.stimulusSetHash,
        grand_mean_population=sidecar.grandMeanPopulation,
        neutral_mean_per_layer=vector_store.load_neutral_mean(directory, name),
        designated_reference=sidecar.designatedReference,
        neutral_corpus_hash=sidecar.neutralCorpusHash,
        # Mirror stamps travel WITH the vector, exactly like the SAE identity
        # below: the copy's stimulusSetHash is the PARENT's (order-sensitive)
        # hash, and only ``polesSwappedFromSource`` says what that hash means
        # for this concept's role-swapped files.
        poles_swapped_from_source=sidecar.polesSwappedFromSource,
        negated_from=sidecar.negatedFrom,
        # SAE identity travels WITH the vector (open-issues #14): the copy is
        # the same decoder row under the same convention, so it says so.
        gemmascope_convention=sidecar.gemmascopeConvention,
        raw_decoder_norm=sidecar.rawDecoderNorm,
        gemmascope_target_norm=sidecar.gemmascopeTargetNorm,
        gemmascope_source=sidecar.gemmascopeSource,
        # Depth coverage travels with the vector too: the copy has the
        # source's rows, so it may only claim what the source could. The
        # import is local because `catalog` walks the tree.
        covers_model_depth=(
            None if catalog.covers_model_depth(
                covers=sidecar.coversModelDepth,
                extraction_method=sidecar.extractionMethod,
                recipe_method=sidecar.recipeMethod) else False),
        pinned_from={
            "path": rel,
            "sha256TensorHash": block.get("sha256TensorHash"),
            "sha256SidecarHash": block.get("sha256SidecarHash"),
            "sourceMethod": sidecar.extractionMethod,
            "sourceConcept": concept.data_concept,
            "sourceConceptLabel": sidecar.concept,
            "sourceExtractionDate": sidecar.extractionDate,
        })


def _extract_designated_reference(model, concept, root,
                                  neutral_texts) -> ConceptVectorBundle:
    """mean(concept stories) − mean(designated reference stories), both at
    the concept's pinned reading position. The classes go through the SAME
    core extract as paired methods — same math, same reading-position
    diagnostic, same neutral projection plumbing — only the data source
    differs, which is the whole point of the recipe being first-class."""
    from types import SimpleNamespace
    from . import multiconcept
    ref = concept.designated_reference or {}
    ref_name = ref.get("name")
    if not ref_name:
        raise RuntimeError(
            f"designated-reference concept '{concept.name}' has no pinned "
            "reference — re-attach with --reference")
    # Refuse drift (external review 2026-07-31, finding 3): the bundle
    # stamps the PINNED hashes, so the bytes read must BE the pinned bytes —
    # the Swift twin refuses identically, and the alternative (stamping live
    # hashes) would let a draft extraction claim inputs the manifest never
    # pinned.
    live = multiconcept.stories_hash(concept.name, root)
    if live != concept.stimulus_set_hash:
        raise RuntimeError(
            f"concept '{concept.name}': stories drifted from the pinned hash "
            f"(have {(live or 'missing')[:12]}…, pinned "
            f"{concept.stimulus_set_hash[:12]}…) — re-attach, or restore the "
            "pinned bytes")
    live_ref = multiconcept.stories_hash(ref_name, root)
    if live_ref != ref.get("hash"):
        raise RuntimeError(
            f"concept '{concept.name}' reference '{ref_name}' stories drifted "
            f"from the pinned hash (have {(live_ref or 'missing')[:12]}…, "
            f"pinned {(ref.get('hash') or '?')[:12]}…) — re-attach, or "
            "restore the pinned bytes")
    positive = multiconcept.load_stories_texts(concept.name, root)
    negative = multiconcept.load_stories_texts(ref_name, root)
    stimuli = SimpleNamespace(positive=positive, negative=negative)
    options = CoreExtractionOptions(
        method=concept.options.method,
        reading_position=concept.options.reading_position,
        neutral_pc_count=concept.options.neutral_pc_count,
        extraction_rendering=concept.options.extraction_rendering)
    result = core_extract(model, stimuli, options, neutral_texts=neutral_texts)
    return ConceptVectorBundle(
        vectors=result.vectors,
        residual_norm_per_layer=result.residual_norm_per_layer,
        residual_norm_source=result.residual_norm_source,
        residual_norm_convention=result.residual_norm_convention,
        residual_norm_rendering=result.residual_norm_rendering,
        stimulus_hash=concept.stimulus_set_hash,
        reading_position_diagnostic=result.reading_position_diagnostic,
        reading_position_resolution=result.reading_position_resolution,
        neutral_mean_per_layer=result.neutral_mean_per_layer,
        designated_reference={"name": ref_name, "hash": ref.get("hash")})


def _extract_grand_mean_bundles(model, manifest: Manifest, root,
                                neutral_texts) -> dict[str, ConceptVectorBundle]:
    """Grand-mean concepts share one pinned population; extract every target
    that shares (reading position, projection) in a single corpus pass so the
    denominator is computed once, exactly as the recipe defines it."""
    grand = [c for c in manifest.concepts if c.options.method.is_grand_mean]
    if not grand:
        return {}
    from ..steering.extractor import extract_grand_mean
    from . import multiconcept
    if manifest.grand_mean_corpus is None:
        raise RuntimeError(
            "grand-mean concepts attached but no grandMeanCorpus pinned — "
            "re-attach with method emotionGrandMean")
    rows, live_hashes = multiconcept.load_corpus(manifest.grand_mean_corpus.concepts, root)
    if not rows:
        raise RuntimeError("grand-mean corpus is empty on disk")
    groups: dict[tuple, list] = {}
    for concept in grand:
        # The RENDERING joins the grouping key: two concepts that render
        # differently are two different corpus passes with two different
        # denominators, so pooling them would silently give one of them the
        # other's numbers.
        key = (concept.options.reading_position.label,
               concept.options.neutral_pc_count,
               json.dumps(concept.options.extraction_rendering.to_dict(),
                          sort_keys=True))
        groups.setdefault(key, []).append(concept)
    bundles: dict[str, ConceptVectorBundle] = {}
    for (_, pc_count, _), members in groups.items():
        reading = members[0].options.reading_position
        rendering = members[0].options.extraction_rendering
        result = extract_grand_mean(
            model, rows, target_concepts={m.name for m in members},
            reading_position=reading, neutral_texts=neutral_texts,
            neutral_pc_count=pc_count, extraction_rendering=rendering)
        for member in members:
            vectors = result.per_concept.get(member.name)
            if vectors is None:
                raise RuntimeError(
                    f"grand-mean concept '{member.name}' has no rows in the "
                    "pinned corpus")
            bundles[member.name] = ConceptVectorBundle(
                vectors=vectors,
                residual_norm_per_layer=result.residual_norm_per_layer,
                residual_norm_source=result.residual_norm_source,
                residual_norm_convention=result.residual_norm_convention,
                residual_norm_rendering=result.residual_norm_rendering,
                reading_position_resolution=result.reading_position_resolution,
                # Live hashes (like the paired path's stimuli.hash): the
                # sidecar records what was actually read — including the FULL
                # population the grand mean was computed over; verify()
                # reports drift.
                stimulus_hash=live_hashes.get(member.name, member.stimulus_set_hash),
                grand_mean_population=dict(live_hashes),
                neutral_mean_per_layer=result.neutral_mean_per_layer)
    return bundles


def _persist_vectors(bundles: dict[str, ConceptVectorBundle], manifest: Manifest,
                     model: model_loader.SteeredModel, run_directory: str) -> None:
    for name, bundle in bundles.items():
        concept = next(c for c in manifest.concepts if c.name == name)
        pc_count = concept.options.neutral_pc_count or 0
        sidecar = SteeringVectorSidecar.make(
            model_id=manifest.model_id, revision=model.revision, concept=name,
            stimulus_set_hash=bundle.stimulus_hash, vectors=bundle.vectors,
            extraction_method=concept.options.method.value,
            reading_position=concept.options.reading_position,
            residual_norm_per_layer=bundle.residual_norm_per_layer,
            residual_norm_source=bundle.residual_norm_source,
            residual_norm_convention=bundle.residual_norm_convention,
            residual_norm_rendering=bundle.residual_norm_rendering,
            extraction_rendering=(bundle.extraction_rendering
                                  if bundle.pinned_from is not None
                                  else concept.options.extraction_rendering),
            reading_position_resolution=bundle.reading_position_resolution,
            # Record the projection that was actually applied (previously the
            # sidecar said "none" even when the legacy pooled projection ran
            # — an under-recorded recipe).
            neutral_projection=(f"legacy-pooled top-{pc_count} neutral PCs"
                                if pc_count > 0 else None),
            # A materialized artifact carries the SOURCE's residual norms, so
            # it must carry the corpus those norms were measured on: the two
            # are one provenance claim, and splitting them would let α in
            # norm units cite a denominator nothing measured.
            neutral_corpus_hash=(bundle.neutral_corpus_hash
                                 if bundle.pinned_from is not None
                                 else manifest.neutral_corpus_hash))
        sidecar.grandMeanPopulation = bundle.grand_mean_population
        sidecar.designatedReference = bundle.designated_reference
        # Materialized-from-pinned-bytes provenance: the copy in this run
        # directory is an ordinary vector artifact in every way EXCEPT that
        # it names the artifact it was copied from, by path and both hashes.
        sidecar.pinnedFrom = bundle.pinned_from
        # Gemma Scope SAE identity, carried from the SOURCE sidecar
        # (open-issues #14). Without this the sweep's per-cell copies — the
        # very artifacts `promote` pins into an agent — looked like
        # pre-convention imports, so every w7 run warned about a scaling
        # convention its imports had actually declared. `recipe_identity`
        # reads none of these fields, so carrying them cannot move
        # `recipeIdentityHash` and promotion's artifact matcher is unchanged.
        sidecar.gemmascopeConvention = bundle.gemmascope_convention
        sidecar.rawDecoderNorm = bundle.raw_decoder_norm
        sidecar.gemmascopeTargetNorm = bundle.gemmascope_target_norm
        sidecar.gemmascopeSource = bundle.gemmascope_source
        sidecar.coversModelDepth = bundle.covers_model_depth
        # Mirror stamps, carried from the SOURCE sidecar for the same reason:
        # the copy's stimulusSetHash is the parent's, and without the
        # qualifier the copy claims the wrong files were read. Absent on
        # every non-mirrored bundle. `recipe_identity` reads neither field,
        # so carrying them cannot move `recipeIdentityHash`.
        sidecar.polesSwappedFromSource = bundle.poles_swapped_from_source
        sidecar.negatedFrom = bundle.negated_from
        # Stamp the canonical full-recipe identity from the sidecar's own
        # recorded fields — the stamp always describes THIS artifact, and
        # stamping exercises the same reader promotion uses. An extraction
        # writer that cannot prove its own recipe is a writer bug and must
        # fail loudly, never write an unprovable artifact.
        components, missing = recipe_identity.candidate_identity(sidecar.to_dict())
        if components is None:
            raise RuntimeError(
                f"extraction sidecar for '{name}' is missing recipe fields "
                f"[{', '.join(missing)}] — cannot stamp recipeIdentityHash "
                "(writer bug)")
        sidecar.recipeIdentityHash = recipe_identity.identity_hash(components)
        save_vectors(bundle.vectors, sidecar, run_directory, name,
                     neutral_mean_per_layer=bundle.neutral_mean_per_layer)


def _residual_norm_at(norms, layer: int, *, artifact: str, where: str) -> float:
    """The α denominator at ``layer``, or the ONE typed refusal every verb
    shares (2026-08-28 audit, F7/F13).

    ``where`` is the caller's own subject ("condition 'x'", "concept 'y'",
    "variant 'z'") and is the only thing that differs between the sites; the
    sentence after it is byte-identical here, in ``model_variant`` and on the
    Swift engine. Before this existed the condition path substituted 0.0 and
    refused as ``degenerateData`` while the sweep and variant paths clamped to
    the last entry — one artifact, three answers, two of them silent.
    """
    problem = norm_convention.residual_norm_problem(
        norms, layer, artifact=artifact)
    if problem is not None:
        raise RuntimeError(f"{where}: {problem}")
    return float(norms[layer])


def _condition_injections(condition, bundles: dict[str, ConceptVectorBundle]) -> list[CellInjection]:
    """Resolve a condition's slots to per-layer injection cells, applying the
    layer band and norm-unit alpha conversion (parallel to
    ChatService.currentInjections)."""
    injections: list[CellInjection] = []
    for slot in condition.slots:
        bundle = bundles.get(slot.concept)
        if bundle is None:
            # 2026-08-28 audit, F8. This used to `continue`, silently dropping
            # the slot: the condition then executed weaker — or, for a
            # single-slot condition, as an unlabelled baseline — under a
            # steered arm's name, and nothing in the run record said so. The
            # Swift twin (`ExperimentTasks.injections(for:extractions:)`) has
            # always thrown here; the sentence is byte-identical to its
            # `reason`, and `Manifest.verify` now catches the same state at
            # verify time so a run rarely has to.
            raise RuntimeError(
                f"condition '{condition.name}' references unextracted concept "
                f"'{slot.concept}'")
        layer_count = bundle.vectors.layer_count
        center = min(max(0, slot.layer), layer_count - 1)
        half = max(0, condition.band_width // 2)  # Swift uses width / 2
        # Ablation covers the whole network: removing a direction at one layer
        # is usually undone by the layers above it. Steering keeps its declared
        # layer widened by the condition's band.
        is_ablation = slot.is_ablation
        first = 0 if is_ablation else max(0, center - half)
        last = layer_count - 1 if is_ablation else min(layer_count - 1, center + half)
        if is_ablation and condition.control_type != "randomDirectionAblation":
            # Ablation mean-alignment preflight (2026-08-06 collapse study):
            # a direction sharing a large component with the neutral residual
            # mean collapses generation into single-token repetition at λ=1.
            # Diagnostic only here — a frozen manifest's semantics are never
            # changed under it (post-submit drift policy: continue loudly).
            _warn_on_mean_aligned_ablation(
                concept=slot.concept,
                vectors=bundle.vectors.per_layer[first:last + 1],
                first_layer=first,
                neutral_mean=(bundle.neutral_mean_per_layer[first:last + 1]
                              if bundle.neutral_mean_per_layer else None),
                where=f"condition '{condition.name}'")
        for layer in range(first, last + 1):
            vector = bundle.vectors.per_layer[layer]
            vector_norm = vm.l2_norm(vector)
            if vector_norm <= 0:
                continue
            if is_ablation and condition.control_type == "randomDirectionAblation":
                # The ablation analogue of the matched-norm control. Norm
                # matching is meaningless for a projection — the removal is
                # scaled by what the residual stream contains, not by the
                # direction's length — so the control removes a random
                # DIRECTION instead: is the effect specific to this direction,
                # or does removing any rank-1 subspace do it?
                vector = _matched_norm_random(
                    seed_text=f"{condition.name}|{slot.concept}|{layer}",
                    dimension=len(vector), norm=vector_norm)
            elif condition.control_type == "randomMatchedNorm":
                # Magnitude/noise control: a deterministic random direction with
                # the SAME L2 norm as the concept vector at this layer. Seeded
                # from stable identifiers only, so the cell is reproducible and
                # identical across re-runs of the frozen study.
                vector = _matched_norm_random(
                    seed_text=f"{condition.name}|{slot.concept}|{layer}",
                    dimension=len(vector), norm=vector_norm)
            alpha = slot.alpha
            # λ is never scaled by the residual norm: that denominator makes α
            # comparable across concepts and layers, while ablation removes
            # exactly what is present and so already scales itself.
            if condition.alpha_in_norm_units and not is_ablation:
                # ONE out-of-range rule, every verb, both engines (2026-08-28
                # audit, F7/F13). This site used to substitute 0.0 for a layer
                # the table did not reach and refuse as `degenerateData` — a
                # typed refusal, but one that named the wrong defect — while
                # the sweep and variant paths clamped silently.
                residual = _residual_norm_at(
                    bundle.residual_norm_per_layer, layer,
                    artifact=slot.concept,
                    where=f"condition '{condition.name}'")
                alpha = vm.norm_unit_scale(slot.alpha, residual, vector_norm)
            injections.append(CellInjection(
                layer=layer, vector=vector, alpha=alpha,
                mode=slot.effective_mode, concept=slot.concept))
    return injections


def _warn_on_mean_aligned_ablation(*, concept: str, vectors: list[list[float]],
                                   first_layer: int,
                                   neutral_mean: list[list[float]] | None,
                                   where: str) -> None:
    """Loud, non-fatal ablation preflight (parallel to the Swift
    ``ChatService`` ablation advisory; same threshold constant).

    With a mean available, names the worst-aligned layer when it crosses
    :data:`vm.ABLATION_MEAN_ALIGNMENT_WARN_THRESHOLD`; with no mean, says the
    check is impossible — unknown alignment is never reported as safe."""
    import warnings as _warnings
    if neutral_mean is None:
        _warnings.warn(
            f"{where}: ablating '{concept}' with no stored neutral mean — "
            f"mean-alignment preflight impossible (artifact/extraction "
            f"predates the neutral-mean stamp or no neutral corpus was "
            f"pinned). λ=1 ablation of a mean-aligned direction collapses "
            f"generation; re-extract with a neutral corpus to enable the "
            f"check and neutral-mean centering", UserWarning, stacklevel=3)
        return
    worst_layer, worst = -1, 0.0
    for offset, (vector, mean_row) in enumerate(zip(vectors, neutral_mean)):
        alignment = vm.mean_alignment(vector, mean_row)
        if alignment > worst:
            worst_layer, worst = first_layer + offset, alignment
    if worst > vm.ABLATION_MEAN_ALIGNMENT_WARN_THRESHOLD:
        _warnings.warn(
            f"{where}: ablation direction for '{concept}' is strongly aligned "
            f"with the neutral residual mean (|cos| {worst:.2f} at layer "
            f"{worst_layer}; warn threshold "
            f"{vm.ABLATION_MEAN_ALIGNMENT_WARN_THRESHOLD}). Full ablation of "
            f"mean-aligned directions collapses generation into single-token "
            f"repetition — center the direction against the neutral mean "
            f"(variant injections: \"centering\": \"neutralMean\") or expect "
            f"incoherent output", UserWarning, stacklevel=3)


# Canonical identifier for the matched-norm random-control recipe, stamped
# wherever a random control is recorded (both engines emit the same string):
# i.i.d. standard-normal components (an isotropic Gaussian direction), then a
# single rescale to the target L2 norm. Swift twin:
# `SteeringVectorMath.randomVector` (`Sources/SteeringKit/Extraction/
# SeededRandom.swift`). Byte-identical vectors across engines are NOT the
# contract (the RNGs differ per substrate); the distribution and this stamp
# are. Provenance rule for readers: an unstamped random control is legacy —
# on this engine it was already Gaussian; on Swift it was cube-uniform-then-
# rescale (not isotropic).
RANDOM_VECTOR_ALGORITHM = "gaussian-isotropic-v1"


def _matched_norm_random(seed_text: str, dimension: int, norm: float) -> list[float]:
    """Gaussian random direction rescaled to a target L2 norm
    (``RANDOM_VECTOR_ALGORITHM``). Seeded by string (CPython hashes str seeds
    with SHA-512 internally), so deterministic across processes and platforms
    — no torch RNG state involved."""
    rng = random.Random(seed_text)
    direction = [rng.gauss(0.0, 1.0) for _ in range(dimension)]
    scale = norm / vm.l2_norm(direction)
    return [value * scale for value in direction]


def _check_option_lengths(choice, manifest: Manifest, prompt_id: str) -> None:
    """Study-path guard: joint logprobs favor shorter options, so unequal
    scored-option token counts silently bias `selected`. Refuse unless the
    manifest explicitly acknowledges the imbalance. Best practice is short
    canonical labels (A/B) with descriptions outside the scored tokens."""
    counts = {len(score.token_ids) for score in choice.options}
    if len(counts) > 1 and not manifest.acknowledge_unequal_option_lengths:
        detail = ", ".join(f"{s.option!r}={len(s.token_ids)}" for s in choice.options)
        raise RuntimeError(
            f"item '{prompt_id}': scored options have unequal token counts "
            f"({detail}) — joint logprobs favor shorter options. Use canonical "
            "labels of equal length, or set acknowledgeUnequalOptionLengths "
            "in the manifest to accept the bias knowingly")


def _sae_latent_preflight(manifest: Manifest, log) -> list[dict]:
    """Validate declared SAE latent conditions, and refuse the paths that
    cannot execute them. Returns the raw entries (possibly empty).

    Called at run START, before the model loads: a malformed declaration, or a
    study kind whose loop has no place to put a latent arm, costs nothing to
    catch here and a queue wait plus a 27B load to catch later.

    The multi-agent refusal is the load-bearing one. That path's condition
    matrix is ``configured``/``baseline`` per seat, with no slot for a
    residual-stream edit declared at study level — so a panel study carrying
    ``saeLatentConditions`` would run to completion with the declared
    mechanism never armed, and nothing in the transcripts would say so. That
    is exactly the silent-inert failure the separate manifest key exists to
    prevent, arriving one layer lower, so it refuses.
    """
    from . import sae_latent as _sae_latent

    entries = manifest.sae_latent_conditions
    if not entries:
        return []
    # Validate first, so a malformed declaration reports as malformed rather
    # than as anything else.
    violations = _sae_latent.condition_violations(manifest.raw)
    if violations:
        raise RuntimeError(
            "manifest declares invalid SAE latent conditions: "
            + "; ".join(violations))
    names = ", ".join(str(e.get("name") or "?") for e in entries)
    if manifest.study_kind == "multiAgent":
        log(f"refusing run: {len(entries)} SAE latent condition(s) declared "
            f"({names}) on a multi-agent study")
        raise RuntimeError(
            f"experiment '{manifest.name}' is a multi-agent study and declares "
            f"{len(entries)} SAE latent condition(s) ({names}) — the panel loop "
            "runs the scenario's configured/baseline conditions per seat and "
            "has nowhere to arm a study-level residual-stream edit, so those "
            "arms would never execute while the run completed normally. Move "
            "the latent condition to a modelOutput study, or seat the "
            "intervention on an agent variant.")
    return list(entries)


def _advise_sweep_ignores_sae_latent(manifest: Manifest, log) -> None:
    """Say out loud that a sweep does not cover SAE latent conditions.

    Deliberately an ADVISORY, not a refusal — and the distinction is the point.
    A sweep's matrix is derived from the study's CONCEPTS and its own
    layer×alpha grid; it never reads ``conditions``, so it drops nothing that
    was declared and its ``recommendations.json`` makes no claim about a latent
    arm. Refusing would block a legitimate concept sweep on a manifest that
    also declares latent conditions.

    What WOULD be wrong is silence — a researcher assuming the ladder they just
    swept covered the β of their latent arm. There is no dose ladder for a
    latent β today (dose calibration is future work: β is in the feature's own
    activation scale, so a grid over it is not comparable across features), and
    this says so at sweep start.
    """
    entries = manifest.sae_latent_conditions
    if not entries:
        return
    names = ", ".join(str(e.get("name") or "?") for e in entries)
    log(f"note: {len(entries)} SAE latent condition(s) ({names}) are NOT part "
        "of this sweep — a sweep selects a layer×alpha cell per CONCEPT, and a "
        "latent condition declares its own layer and its β in latent units, "
        "which no alpha ladder is comparable to. They execute in `run`; "
        "nothing here selects, tunes or qualifies them.")


def _materialize_sae_latent_conditions(manifest: Manifest, log):
    """Load every declared latent condition's SAE tensors through the loader
    seam. Returns ``[(spec, edit, provenance)]``.

    This is the latent analogue of ``_extract_all``: the manifest pins the
    RECIPE (release, saeID, feature, mode, beta), never the bytes, and the run
    re-derives the intervention from the pinned source — here by reading the
    published dictionary and stamping the exact repository commit it read.
    A failure refuses the run rather than dropping the arm.
    """
    from . import sae_latent as _sae_latent

    specs = _sae_latent.parse(manifest.raw)
    resolved = []
    for spec in specs:
        edit, provenance = _sae_latent.materialize(spec)
        resolved.append((spec, edit, provenance))
        log(f"SAE latent condition '{spec.name}': {spec.release}/{spec.sae_id} "
            f"feature {spec.feature} at layer {provenance['layer']}, "
            f"mode {spec.mode}, beta {spec.beta} (latent units); "
            f"repository {provenance['repository']}@"
            f"{str(provenance['repositoryRevision'])[:12]}")
    return resolved


def _effective_sae_latent_condition(spec, edit, provenance,
                                    manifest) -> EffectiveCondition:
    """A latent condition's resolved execution configuration.

    Executes under the manifest's own prompt/sampling configuration exactly
    like an ordinary condition — the ONE measurement pipeline applies here
    too. What differs is only the model configuration: no vector injections at
    all, and the latent edit carried in its own field.

    A latent arm carries no agent identity, so the system-prompt composition
    degrades to the study frame itself (:func:`_effective_ordinary_condition`
    twin) — byte-identical to what it always rendered.
    """
    from . import sae_latent as _sae_latent

    return EffectiveCondition(
        name=spec.name,
        injections=[],
        latent_edits=[edit],
        intervention_state=_sae_latent.intervention_state(spec, provenance),
        prompt_mode=manifest.prompt_mode,
        system_prompt=system_prompt_mod.compose(None, manifest.system_prompt),
        qwen_thinking_enabled=manifest.qwen_thinking_enabled,
        temperature=manifest.temperature,
        agent_system_prompt=None,
        study_system_prompt=manifest.system_prompt)


def _intervention_state(condition) -> dict:
    """JSON-safe provenance for what was injected under this condition —
    stamped on every record so a reader never reconstructs it from the name."""
    state = {
        "slots": [{"concept": s.concept, "layer": s.layer, "alpha": s.alpha}
                  for s in condition.slots],
        "bandWidth": condition.band_width,
        "alphaInNormUnits": condition.alpha_in_norm_units,
        "controlType": getattr(condition, "control_type", None),
    }
    if state["controlType"] == "randomMatchedNorm":
        # Which random-control recipe generated the injected direction.
        # Records without this key are legacy: Gaussian on this engine,
        # cube-uniform on Swift (see RANDOM_VECTOR_ALGORITHM).
        state["randomVectorAlgorithm"] = RANDOM_VECTOR_ALGORITHM
    return state


def _reader_scorers(manifest: Manifest, root: str | None) -> list[tuple[str, object]]:
    """Load the manifest's pinned RepE reader artifacts for the
    ``repeReaderScore`` outcome instrument: ``[(concept, ReaderArtifact)]``.
    The FULL binding is enforced here as well as in verify() — substrate,
    model, revision, and concept (review 2026-08-02: a draft manifest only
    warns on verification, and a FORCED freeze skips gates by design but
    must never corrupt semantics; without the runtime recheck a forced
    freeze could score one reader while calling it another concept)."""
    from ..steering import repe_reader
    scorers: list[tuple[str, object]] = []
    for ref in manifest.reader_refs:
        path = ref.path if os.path.isabs(ref.path) else os.path.join(
            paths.project_root() if root is None else root, ref.path)
        reader = repe_reader.load_reader(path)
        # The SAME binding helper verify uses — the runtime can never
        # accept a reader verify would flag (review 2026-08-02; a reader
        # with no revision now refuses here too).
        problems = repe_reader.binding_problems(
            reader, ref_concept=ref.concept, model_id=manifest.model_id,
            model_revision=manifest.model_revision)
        if problems:
            raise RuntimeError(
                f"({ref.path}) " + "; ".join(problems))
        scorers.append((ref.concept, reader))
    return scorers


def _reader_scores(model, scorers: list[tuple[str, object]], text: str) -> dict[str, float]:
    """Per-record reader readout: each pinned reader scores the sampled output
    text through its own template + LAT position + training normalization.
    Runs unsteered (the recorder session carries no injectors): the instrument
    measures the *text*, not the steered residual stream that produced it."""
    from ..steering import repe_reader
    return {concept: repe_reader.score_text(model, reader, text)
            for concept, reader in scorers}


def _write_config_snapshot(manifest: Manifest, run_directory: str, task: str,
                           notes: dict | None = None, *, model=None,
                           job_id: str | None = None) -> None:
    with open(os.path.join(run_directory, "experiment.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest.raw, handle, indent=2, sort_keys=True)
    with open(os.path.join(run_directory, "experiment-hash.txt"), "w", encoding="utf-8") as handle:
        handle.write(manifest.content_hash() + "\n")
    with open(os.path.join(run_directory, "task.txt"), "w", encoding="utf-8") as handle:
        handle.write(task + "\n")
    # Canonical cross-engine per-run stamp (additive — never replaces the
    # richer per-task artifacts above). Sampling-policy fields are stamped
    # only for generation-bearing tasks (study runs — incl. the multi-agent
    # study path, which passes task "run" — and sweeps); extract/validate/
    # evaluate/analyze do not sample by the manifest's policy, so they stamp
    # null rather than inventing one.
    generates = task in ("run", "sweep")
    write_run_config(run_directory, task,
                     model_id=manifest.model_id, revision=manifest.model_revision,
                     experiment=manifest.name, experiment_hash=manifest.content_hash(),
                     temperature=manifest.temperature if generates else None,
                     samples_per_item=manifest.samples_per_item if generates else None,
                     seed_policy=manifest.seed_policy if generates else None,
                     # Schema 3: the precision the model ACTUALLY ran in.
                     # Stamped for every task that loaded a model, not just
                     # generation-bearing ones — extraction reads activations
                     # in that precision too, and the residual-norm
                     # denominator that alpha is expressed in comes with it.
                     dtype=_actual_dtype(model),
                     # Explicit for directories created OUTSIDE the producing
                     # job (shard merges, pipeline seeds, assembled on the
                     # controller): without it the env fallback stamped the
                     # CONTROLLER's own Slurm allocation id — four merged
                     # runs shared one stale jobId while their true shard
                     # ids sat in report.json's sharded block (2026-08-06).
                     job_id=job_id,
                     notes=notes)


def _advise_cross_substrate(manifest: Manifest, run_directory: str | None,
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


def _advise_dependency_lock_drift(run_directory: str | None, _log, *,
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


def _advise_system_prompt_divergence(arms, run_directory: str | None, _log, *,
                                     write_file: bool) -> None:
    """Comparability advisory (2026-08-24 ruling): the arms of THIS run are
    not all armed with the same effective system content.

    Same shape as :func:`_advise_dependency_lock_drift` — an ``ADVISORY:``
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


def _advise_implicit_case_family(fires: bool, run_directory: str | None, _log,
                                 *, write_file: bool) -> None:
    """The deprecated ``caseFamily == "sentencing"`` endpoint selection, said
    out loud wherever it actually fires (2026-08-18).

    Same shape as :func:`_advise_dependency_lock_drift` — logged at the verb's
    START, appended to the run directory's ``advisories.txt``, and NEVER a
    refusal. Compatibility is the point: a manifest that already depends on the
    trigger must keep producing the same numbers, so this changes nothing about
    the run except that the run says how its endpoint was chosen.

    ``fires`` is computed by the CALLER rather than re-derived here, because
    the sites do not share one predicate: the record-parse and analyze-rescue
    paths let a declared ``numericParser`` win
    (:func:`manifest.implicit_case_family_endpoint`), while the multi-agent
    panel-effects endpoint reads ``case_family`` alone. Deriving one predicate
    for all three would make the advisory lie at one of them.
    """
    if not fires:
        return
    from .manifest import IMPLICIT_CASE_FAMILY_ADVISORY
    _log(f"ADVISORY: {IMPLICIT_CASE_FAMILY_ADVISORY}")
    if not write_file or not run_directory:
        return
    # Append, like the lock-drift advisory: the cross-substrate advisory may
    # already own this file for this run.
    try:
        with open(os.path.join(run_directory, "advisories.txt"), "a",
                  encoding="utf-8") as handle:
            handle.write(IMPLICIT_CASE_FAMILY_ADVISORY + "\n")
    except OSError:  # the advisory must never sink a run
        pass


# --- tasks -----------------------------------------------------------------

def extract(name: str, root: str | None = None, dtype: str = "auto",
            device: str | None = None, *, model_provider=None,
            should_cancel: Callable[[], bool] | None = None, log=None) -> str:
    _log = log or print
    manifest = Manifest.load(name, root)
    _verify_or_warn(manifest, root)
    _advise_inert_declarations(manifest, _log)
    with _acquire_model(manifest, dtype, device, model_provider) as model:
        manifest = _pin_model_revision(name, manifest, model, root, _log)
        bundles = _extract_all(model, manifest, root)
        run_directory = paths.make_unique_run_directory(f"exp-{name}-extract", root)
        _write_config_snapshot(manifest, run_directory, "extract", model=model)
        _persist_vectors(bundles, manifest, model, run_directory)
        _write_reading_position_diagnostics(bundles, run_directory, _log)
        _write_logit_lens_vocabulary(bundles, manifest, model,
                                     run_directory, _log)
    _log(f"extracted {len(bundles)} concept vectors → {run_directory}")
    return run_directory


def _advise_inert_declarations(manifest: Manifest, _log) -> None:
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
        system_prompt=manifest.system_prompt)
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


def _write_logit_lens_vocabulary(bundles, manifest: Manifest, model,
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
            resolutions = _validation_layer_resolutions(
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
                "layerResolution": _resolution_block(resolution),
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


def validate(name: str, root: str | None = None, dtype: str = "auto",
             device: str | None = None, *, model_provider=None,
             should_cancel: Callable[[], bool] | None = None, log=None) -> str:
    """Cross-concept cosine matrix + held-out probe accuracy (validation.jsonl).

    Convergent validity: the vector classifies its own held-out scenarios.
    Discriminant validity: distinct concepts are not collapsed into one
    direction (reported as a cosine matrix CSV).
    """
    manifest = Manifest.load(name, root)
    manifest = _autopin_capability_battery(name, manifest, root, log or print)
    _verify_or_warn(manifest, root)
    with _acquire_model(manifest, dtype, device, model_provider) as model:
        manifest = _pin_model_revision(name, manifest, model, root, log or print)
        return _validate_impl(name, manifest, model, root, log or print)


def _autopin_capability_battery(name: str, manifest: Manifest, root,
                                _log) -> Manifest:
    """Pin the DEFAULT battery into an unpinned variant-study DRAFT before
    validation (mirrors Swift): the validation scope hash composes the PIN,
    so evidence produced against an implicit default would never match a
    manifest that freeze later pins — the manifest must name its battery
    before the evidence is minted. Frozen manifests are immutable and are
    never touched (legacy frozen studies keep the live-default comparison)."""
    from . import battery as battery_mod
    from . import experiment_store

    if (not manifest.variant_conditions or manifest.capability_battery_hash
            or manifest.status != "draft"):
        return manifest
    digest = battery_mod.live_hash(battery_mod.DEFAULT_BATTERY_FILE, root)
    if digest is None:
        return manifest
    experiment_store.set_protocol(
        name, {"capabilityBatteryFile": battery_mod.DEFAULT_BATTERY_FILE,
               "capabilityBatteryHash": digest}, root)
    _log(f"pinned default capability battery {battery_mod.DEFAULT_BATTERY_FILE} "
         f"@ {digest[:12]}… into '{name}' (variant study, no explicit pin)")
    return Manifest.load(name, root)


def _validate_impl(name: str, manifest: Manifest, model, root, _log) -> str:
    bundles = _extract_all(model, manifest, root)

    # Depth and range checks BEFORE the run directory exists. Extraction is
    # what reveals model depth, and everything after this point WRITES:
    # config.json, validation-evidence.json, the persisted vectors. Checking
    # later left a half-populated run directory behind on refusal — not
    # freeze-acceptable (no report), but litter that looks like a validation
    # run and has to be reasoned about.
    from . import validation_layer as _vl
    _depth = _require_uniform_depth(bundles)
    _range_refusal = _vl.range_refusal(
        manifest.raw.get("validationLayer"), _depth)
    if _range_refusal:
        raise RuntimeError(_range_refusal)
    # A declared depth LIST can refuse too (out-of-range entry, or two
    # entries resolving to one layer) — same rule: before anything writes.
    _vl.resolve_all(
        declared_layers=manifest.raw.get("validationLayers"),
        declared_fractions=manifest.raw.get("validationLayerFractions"),
        declared_layer=manifest.raw.get("validationLayer"),
        declared_fraction=manifest.raw.get("validationLayerFraction"),
        condition_layer=None, layer_count=max(_depth, 1))

    run_directory = paths.make_unique_run_directory(f"exp-{name}-validate", root)
    _write_config_snapshot(manifest, run_directory, "validate", model=model)
    # Validation-evidence contract (parallel to Swift ``isCompleteValidationRun``):
    # a freeze accepts this run only if validation-evidence.json names task
    # "validate" with the matching scope hash AND a validation report exists.
    # ``substrate`` makes the same-substrate requirement EXPLICIT: CUDA/HF
    # activations do not match MLX/Metal, so evidence produced on one engine
    # must never certify a freeze on the other (previously prevented only by
    # accidental filename/hash divergence between the engines).
    evidence = {"schemaVersion": 1, "task": "validate", "experiment": name,
                "substrate": vector_store.SUBSTRATE,
                "reportFile": "validation-report.json",
                "validationScopeHash": manifest.validation_scope_hash()}
    if manifest.variant_conditions:
        # Capability-battery-as-evidence: run the pinned battery under EVERY
        # variant condition and baseline; freeze's variant gate requires these
        # per-condition results in the scope-matched evidence.
        evidence["batteryResults"] = _battery_results(manifest, model, root, _log)
    with open(os.path.join(run_directory, "validation-evidence.json"), "w",
              encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)
    _persist_vectors(bundles, manifest, model, run_directory)

    # Declared discriminant-validity CONTROLS (C2). The server had none at
    # all, while Swift swept in "every other concept on disk" extracted with
    # a borrowed recipe — so the two engines disagreed about what validate
    # even measures, and Swift's answer depended on unrelated workspace
    # contents. Controls are now declared, fully pinned, and extracted with
    # their OWN options on both engines.
    control_bundles = _extract_validation_controls(model, manifest, root, _log)
    matrix_bundles = {**bundles, **control_bundles}
    for advisory in _undeclared_control_advisories(manifest, root):
        _log(advisory)

    names = list(matrix_bundles)
    # Cross-concept cosine matrix at ONE layer for the whole matrix.
    #
    # This used to be hardcoded to mid-network, so a study declaring
    # validationLayer 41 got its convergent accuracy at 41 and its
    # discriminant matrix at 31 — two depths in one report, and different
    # from Swift, which resolved a layer PER ROW and could therefore produce
    # an asymmetric matrix whose (A,B) and (B,A) cells were measured at
    # different depths.
    #
    # One layer for the entire matrix is the only form the cap can be read
    # against: the residual stream drifts with depth (the same concept 4
    # layers apart can be near-orthogonal to itself), so a cosine measured
    # across two depths conflates "different concepts" with "different
    # depths". The layer is recorded in the CSV and the report so the number
    # always carries the depth it was measured at.
    # Controls can still violate the depth invariant even when the study's
    # own bundles agree (checked before the run directory was created).
    _require_uniform_depth(matrix_bundles)
    matrix_layer_list = _matrix_layers(manifest, matrix_bundles)
    matrix_layer = matrix_layer_list[0]
    for index, one_layer in enumerate(matrix_layer_list):
        # One complete matrix per declared depth. The first keeps the
        # historical filename so every existing consumer still finds it;
        # additional depths are suffixed with the layer they were read at.
        filename = ("cosine-matrix.csv" if index == 0
                    else f"cosine-matrix-L{one_layer}.csv")
        matrix_path = os.path.join(run_directory, filename)
        with open(matrix_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["concept", "layer"] + names)
            for a in names:
                # No per-artifact clamp: depth is uniform (checked above), so
                # every cell in this matrix is read at exactly `one_layer`.
                va = matrix_bundles[a].vectors.per_layer[one_layer]
                row = [a, str(one_layer)]
                for b in names:
                    vb = matrix_bundles[b].vectors.per_layer[one_layer]
                    try:
                        row.append(f"{vm.cosine_similarity(va, vb):.4f}")
                    except vm.SteeringVectorError:
                        row.append("nan")
                writer.writerow(row)

    # Convergent validity: true held-out accuracy on LABELED validation.jsonl
    # scenarios (matches Swift scenarioAccuracy — the firewall's convergent half).
    from ..steering.extractor import activations
    from ..steering.stimulus_set import load_validation
    from . import concept_stats, multiconcept, scenario_diagnostics
    report = {"experiment": name, "concepts": {},
              "cosineMatrixLayer": matrix_layer,
              "cosineMatrixLayers": matrix_layer_list}
    # Vacuity ledger (2026-08-17 firewall repair). Every pinned concept that
    # OWES a held-out probe starts here and is struck off the moment one is
    # actually SCORED. What survives is what this run did not measure: a
    # missing/empty/unlabeled validation.jsonl, or a concept that produced no
    # bundle at all. The survivors are stamped into validation-evidence.json,
    # where freeze's validateEvidence gate refuses them — a validate run with
    # nothing to probe used to satisfy that gate silently, on the DEFAULT
    # path (a seeded workspace has no validation.jsonl for any concept),
    # while `data check` called the same missing file a blocker.
    from .manifest import held_out_probe_relpath, owes_held_out_probe
    vacuous_concepts = {c.name: held_out_probe_relpath(c)
                        for c in manifest.concepts if owes_held_out_probe(c)}
    corpus_cache: dict[str, tuple[list[str], list]] = {}

    def _corpus_activations(reading, rendering):
        """(concept labels, pooled activations) for the pinned grand-mean
        corpus at one reading position and rendering — computed once per
        (position, rendering) pair. The rendering joins the cache key because
        it changes the token sequence, hence the activations."""
        key = f"{reading.label} | {rendering.label}"
        if key not in corpus_cache:
            rows, _ = multiconcept.load_corpus(manifest.grand_mean_corpus.concepts, root)
            from ..steering.extractor import _screen_short
            rows = _screen_short(model, rows, reading, 1.0, rendering)
            values = activations(model, [t for _, t in rows], reading,
                                 rendering).values
            corpus_cache[key] = ([c for c, _ in rows], values)
        return corpus_cache[key]

    for concept_name, bundle in bundles.items():
        concept = next(c for c in manifest.concepts if c.name == concept_name)
        # Branch on what validation MEANS: contrastive (two class means)
        # vs population (grand mean). A method that is neither refuses
        # loudly instead of falling into whichever branch is syntactically
        # last (review 2026-07-31 round 2, finding 2).
        #
        # An artifact-pinned concept asks its SOURCE method (its vector was
        # materialized, not derived, but the held-out probe is unchanged:
        # the same class means, the same scoring, read at the artifact's own
        # reading position) and reads its DATA concept's files — a post-hoc
        # derived direction is renamed ("crit" → "crit-gm") but keeps the
        # base concept's held-out set.
        method = concept.effective_method
        data_concept = concept.data_concept
        if not method.has_source_concept:
            # Nothing to validate. An optvec direction's evidence lives in its
            # eval.json (OptVec plan §6); an imported Gemma Scope SAE decoder
            # row's lives in the pinned candidate roster's discovery snapshot
            # + qualification artifact (proposal r2 §4/§6). Neither has
            # stimuli, class means or a held-out validation.jsonl, so there is
            # no probe to run. Skipped rather than refused so a MIXED study
            # still validates its ordinary concepts.
            continue
        if not (method.uses_contrastive_validation or method.is_grand_mean):
            raise RuntimeError(
                f"concept '{concept_name}': method "
                f"'{method.value}' declares no validation "
                "semantics (neither contrastive nor grand-mean)")
        grand_mean = method.is_grand_mean
        # Dual-root lookup (2026-08-19): the recipe's canonical home first,
        # the OTHER recipe's home as a fallback. A set filed under the wrong
        # root used to be read as absent — no probe scored, no pin, no error
        # — so the fallback is LOUD: the advisory names where it was found
        # and where it belongs, and the probe still runs.
        from .manifest import resolve_validation_file, validation_lookup_advisory
        location = resolve_validation_file(
            data_concept, paired=not method.uses_story_corpus, root=root)
        advisory = validation_lookup_advisory(concept_name, location)
        if advisory:
            _log(f"advisory: {advisory}")
        if location is None:
            continue
        val_path = location.path
        scenarios = load_validation(val_path)
        if not scenarios:
            continue
        reading = concept.options.reading_position
        # Held-out activations must be read where AND rendered how the vector
        # was: a probe score is a projection onto that direction, and a
        # raw-tokenized scenario is a sample from a different distribution
        # than a chat-template-rendered one (ledger §26).
        #
        # VALIDATION IS FRAME-FREE, deliberately: the study's
        # `manifest.systemPrompt` — and, since the 2026-08-24 composition
        # ruling, any agent persona composed with it — governs GENERATION
        # arming and nothing else. It must never reach a held-out read, or the
        # probe would score a distribution the vector was not extracted from
        # and the accuracy would move with a run-time deployment choice. The
        # ONE sanctioned channel for persona- or template-conditioned
        # validation is the recipe's own pinned `extractionRendering
        # .systemPrompt`, resolved here — it is part of recipe identity, so
        # extraction and validation cannot silently disagree about it.
        # (`test_system_prompt_composition.py` asserts this by test; Swift
        # twin: the `resolvedExtractionRendering` sites in ExperimentTasks.)
        rendering = concept.options.extraction_rendering
        resolutions = _validation_layer_resolutions(
            manifest, concept_name, bundle.vectors.layer_count)
        # Activations are captured ONCE for all layers — extra declared
        # depths cost per-layer arithmetic, not forward passes. That is why
        # a depth list is one run, not N runs.
        scen = activations(model, [s["text"] for s in scenarios], reading,
                           rendering).values
        labeled = all("expresses" in s for s in scenarios)
        entry: dict = {"scenarioCount": len(scenarios), "labeled": labeled}
        if grand_mean:
            labels_by_row, values = _corpus_activations(reading, rendering)
        else:
            if method.is_designated_reference:
                from types import SimpleNamespace
                ref_pin = concept.designated_reference or {}
                stimuli = SimpleNamespace(
                    positive=multiconcept.load_stories_texts(data_concept, root),
                    negative=multiconcept.load_stories_texts(
                        ref_pin.get("name", ""), root))
            else:
                stimuli = StimulusSet.from_directory(
                    paths.concept_directory(data_concept, root))
            pos = activations(model, stimuli.positive, reading, rendering).values
            neg = activations(model, stimuli.negative, reading, rendering).values

        depth_entries: list[dict] = []
        for resolution in resolutions:
            layer = resolution.layer
            direction = bundle.vectors.per_layer[layer]
            sub: dict = {"layer": layer,
                         "layerResolution": _resolution_block(resolution)}
            if grand_mean:
                concept_rows = [values[i][layer]
                                for i, c in enumerate(labels_by_row)
                                if c == data_concept]
                population_rows = [v[layer] for v in values]
                class_means = {
                    "concept": vm.dot(direction, vm.mean(concept_rows)),
                    "population": vm.dot(direction, vm.mean(population_rows))}
            else:
                class_means = {
                    "positive": vm.dot(direction,
                                       vm.mean([r[layer] for r in pos])),
                    "negative": vm.dot(direction,
                                       vm.mean([r[layer] for r in neg]))}
            midpoint = sum(class_means.values()) / 2
            if labeled:
                if grand_mean:
                    sub["scenarioAccuracy"] = \
                        concept_stats.scenario_accuracy_grand_mean(
                            direction=direction, concept=concept_rows,
                            population=population_rows,
                            scenarios=[a[layer] for a in scen],
                            labels=[s["expresses"] for s in scenarios])
                else:
                    sub["scenarioAccuracy"] = concept_stats.scenario_accuracy(
                        direction=direction,
                        positive=[r[layer] for r in pos],
                        negative=[r[layer] for r in neg],
                        scenarios=[a[layer] for a in scen],
                        labels=[s["expresses"] for s in scenarios])
                # D1: keep the working. The accuracy above is computed FROM
                # per-row projections and a midpoint that were previously
                # discarded — leaving no way to tell "does not read the
                # concept" from "ranks correctly, thresholds badly".
                sub["diagnostics"] = scenario_diagnostics.diagnostics(
                    direction=direction, scenarios=scenarios,
                    projections=[vm.dot(direction, a[layer]) for a in scen],
                    labels=[bool(s["expresses"]) for s in scenarios],
                    threshold=midpoint,
                    class_means=class_means,
                    layer=layer,
                    direction_norm=vm.l2_norm(direction))
            else:
                # Legacy unlabeled file: can't score convergent accuracy.
                # Report the midpoint-side fraction and flag that labels are
                # required.
                above = sum(1 for a in scen
                            if vm.dot(direction, a[layer]) > midpoint)
                sub["fractionAboveMidpoint"] = above / len(scenarios)
                sub["note"] = ("validation.jsonl is unlabeled; add "
                               "'expresses' for true accuracy")
            depth_entries.append(sub)
            _log(f"{concept_name}: validation read at {resolution.summary}")
        # `depths` is the canonical shape; the flat single-depth mirror is
        # kept EXACTLY when one depth resolves, so every pre-list consumer
        # (and report) reads unchanged. With several depths there is no flat
        # mirror — nothing may silently read depth[0] as "the" accuracy.
        entry["depths"] = depth_entries
        if len(depth_entries) == 1:
            entry.update(depth_entries[0])
        report["concepts"][concept_name] = entry
        # A SCORED probe strikes the concept off the vacuity ledger. An
        # unlabeled legacy file does not: it yields fractionAboveMidpoint and
        # a "add 'expresses' for true accuracy" note, never an accuracy — and
        # Swift's loader refuses such a file outright, so counting it as
        # evidence would make the two engines disagree about what validated.
        if any("scenarioAccuracy" in sub for sub in depth_entries):
            vacuous_concepts.pop(concept_name, None)

    # Logit lens (C3): project each concept direction through the model's
    # unembedding head and record the tokens it most promotes/suppresses.
    # Not a gate — a cheap read that catches dead or obviously confounded
    # vectors before expensive steering runs.
    #
    # `logit_lens` has existed in extractor.py since the reader work landed
    # and was never called from anywhere: Swift ran its equivalent inside
    # validate, the server did not, so the same study produced a
    # `logitLens` block on one engine and nothing on the other. This is the
    # missing call site, not a new instrument.
    #
    # topK is pinned to 10 to match the Swift call (ExperimentTasks.swift's
    # `topK: 10`), NOT to `logit_lens`'s own default of 12 — the two reports
    # are meant to be read side by side.
    lens_report: dict = {}
    for concept_name, bundle in bundles.items():
        per_depth: list = []
        for resolution in _validation_layer_resolutions(
                manifest, concept_name, bundle.vectors.layer_count):
            layer = resolution.layer
            try:
                from ..steering.extractor import logit_lens
                lens = logit_lens(model, bundle.vectors, layer, top_k=10)
                per_depth.append({
                    "layer": lens.layer,
                    "topPositive": [
                        {"tokenID": t.token_id, "token": t.token,
                         "logit": t.logit} for t in lens.top_positive],
                    "topNegative": [
                        {"tokenID": t.token_id, "token": t.token,
                         "logit": t.logit} for t in lens.top_negative],
                })
                top = ", ".join(t.token for t in lens.top_positive[:5])
                _log(f"{concept_name}: logit-lens top tokens @ L{layer}: {top}")
            except Exception as exc:  # noqa: BLE001 — a diagnostic must never
                # fail a validation run; Swift records the same skip string.
                per_depth.append(f"logit-lens skipped: {exc}")
        # Single depth keeps the historical flat shape; a list of depths is
        # a list of the same blocks.
        lens_report[concept_name] = (
            per_depth[0] if len(per_depth) == 1 else per_depth)
    report["logitLens"] = lens_report

    # The vacuity verdict rides the REPORT (so a run directory says on its own
    # face what it did not measure) and the EVIDENCE file (so freeze can read
    # it without re-deriving anything). The evidence file was written before
    # the probe loop — the stamp is added by rewriting it here, once the
    # verdict exists. ALWAYS stamped, possibly empty: an absent key means
    # legacy evidence, which keeps satisfying the gate exactly as it did.
    # Swift twin: ``ExperimentTasks.validate`` / ``writeValidationEvidence``.
    report["vacuousConcepts"] = sorted(vacuous_concepts)
    evidence["vacuousConcepts"] = sorted(vacuous_concepts)
    with open(os.path.join(run_directory, "validation-evidence.json"), "w",
              encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)

    with open(os.path.join(run_directory, "validation-report.json"), "w", encoding="utf-8") as h:
        json.dump(report, h, indent=2, sort_keys=True)
    if vacuous_concepts:
        _log("WARNING: VACUOUS validation — no held-out probe was scored for "
             f"concept(s) {', '.join(sorted(vacuous_concepts))}. Author the "
             "never-named scenarios ("
             + ", ".join(p for _c, p in sorted(vacuous_concepts.items()) if p)
             + ") as {\"text\": …, \"expresses\": true|false} rows and re-run "
             "validate; this run is stamped vacuous and will NOT satisfy "
             "freeze's validateEvidence gate")
    _log(f"validation → {run_directory}")
    return run_directory


def _extract_validation_controls(model, manifest: Manifest, root, _log
                                 ) -> dict[str, ConceptVectorBundle]:
    """Extract each DECLARED control with its OWN pinned recipe.

    A control is a complete pinned recipe reference: which concept, which
    stimulus bytes, and its own extraction options. Borrowing a study
    concept's options — as Swift used to — reads a control authored for one
    method at the position of another, and the resulting cosine says nothing.
    Swift twin: the control loop in ``ExperimentTasks.validate``."""
    controls = manifest.raw.get("validationControls") or []
    if not controls:
        return {}
    neutral_texts = None
    if manifest.neutral_corpus_hash:
        try:
            neutral_texts = load_texts(paths.neutral_corpus_path(root)).texts
        except Exception:  # noqa: BLE001
            neutral_texts = None
    from .manifest import ExtractionOptions

    out: dict[str, ConceptVectorBundle] = {}
    for control in controls:
        concept = (control or {}).get("concept")
        if not concept:
            raise RuntimeError("validationControls entry has no 'concept'")
        declared_revision = control.get("modelRevision")
        if declared_revision and manifest.model_revision \
                and declared_revision != manifest.model_revision:
            raise RuntimeError(
                f"validation control '{concept}' pins model revision "
                f"{declared_revision}, but the study pins "
                f"{manifest.model_revision} — a control extracted from a "
                "different revision is not comparable to the study's "
                "directions")
        # A control is a COMPLETE pinned recipe reference or it is not a
        # control. Permitting an absent hash, or defaulting absent options,
        # made the Python contract weaker than Swift's typed decoding — and
        # weaker than the "complete pinned recipe" this feature claims. A
        # control extracted under defaulted options is not the direction the
        # researcher declared, and a control with no pinned hash cannot be
        # shown to be the bytes they compared against.
        pinned = control.get("stimulusSetHash")
        if not pinned:
            raise RuntimeError(
                f"validation control '{concept}' declares no stimulusSetHash "
                "— a control is a complete pinned recipe reference, so its "
                "stimulus bytes must be pinned like any other input")
        if control.get("options") is None:
            raise RuntimeError(
                f"validation control '{concept}' declares no extraction "
                "options — a control must carry its OWN recipe; inheriting or "
                "defaulting one reads it at a position it was not authored for")
        directory = paths.concept_directory(concept, root)
        try:
            stimuli = StimulusSet.from_directory(directory)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"validation control '{concept}' has no readable stimulus set "
                f"at {directory} — declared controls are pinned inputs, not "
                f"best-effort extras ({exc})") from exc
        if stimuli.hash != pinned:
            raise RuntimeError(
                f"validation control '{concept}' stimulus set drifted from "
                f"its pin ({pinned[:12]}… → {stimuli.hash[:12]}…) — re-pin the "
                "control or restore the file")
        if control.get("validationLayer") is not None:
            # Inoperative on both engines; see Swift
            # ValidationControl.validationLayer for why it can never be
            # honoured.
            raise RuntimeError(
                f"validation control '{concept}' declares validationLayer, "
                "which nothing reads: the cosine matrix compares every cell "
                "at ONE layer, so a per-control layer would conflate concept "
                "differences with depth differences. Remove the field; "
                "declare the study-wide validationLayer instead")
        options = ExtractionOptions.from_json(control.get("options") or {})
        _log(f"extracting control '{concept}' with its own recipe…")
        result = core_extract(
            model, stimuli,
            CoreExtractionOptions(
                method=options.method,
                reading_position=options.reading_position,
                neutral_pc_count=options.neutral_pc_count,
                extraction_rendering=options.extraction_rendering),
            neutral_texts=neutral_texts)
        out[concept] = ConceptVectorBundle(
            vectors=result.vectors,
            residual_norm_per_layer=result.residual_norm_per_layer,
            residual_norm_source=result.residual_norm_source,
            residual_norm_convention=result.residual_norm_convention,
            residual_norm_rendering=result.residual_norm_rendering,
            reading_position_resolution=result.reading_position_resolution,
            stimulus_hash=stimuli.hash)
    return out


def _undeclared_control_advisories(manifest: Manifest, root) -> list[str]:
    """Name the concepts on disk that are NOT declared controls.

    Swift's old ambient rule folded these into the matrix silently; removing
    that must not itself be silent. Advisory only — an undeclared concept is
    not an error, it is simply not evidence. Swift twin:
    ``ExperimentStore.undeclaredControlAdvisories``."""
    declared = {(c or {}).get("concept")
                for c in (manifest.raw.get("validationControls") or [])}
    pinned = {c.name for c in manifest.concepts}
    concepts_root = os.path.join(paths.project_root() if root is None else root,
                                 "prompts", "concepts")
    try:
        available = sorted(
            d for d in os.listdir(concepts_root)
            if os.path.isdir(os.path.join(concepts_root, d)))
    except OSError:
        return []
    undeclared = [d for d in available
                  if d not in declared and d not in pinned]
    if not undeclared:
        return []
    return [
        f"note: {len(undeclared)} concept(s) in this workspace are not "
        "declared as validation controls and are NOT in the cosine matrix: "
        f"{', '.join(undeclared)}. Discriminant evidence covers declared "
        "inputs only — add them to validationControls (each with its own "
        "stimulus hash and extraction options) to measure against them."]


def _require_uniform_depth(bundles) -> int:
    """Every bundle must report the SAME layer count, or refuse.

    All bundles belong to one model at one revision, so differing depths mean
    a corrupt or mismatched artifact — not something to paper over. Clamping
    each row to its own depth is what let the matrix become asymmetric again
    through the back door: two rows at different layers produce (A,B) and
    (B,A) measured at different depths, which is exactly the state the
    one-layer invariant exists to prevent."""
    depths = {name: b.vectors.layer_count for name, b in bundles.items()}
    distinct = set(depths.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{n}={d}" for n, d in sorted(depths.items()))
        raise RuntimeError(
            "validation artifacts disagree about model depth "
            f"({detail}) — every vector in one study belongs to the same "
            "model revision, so this is a corrupt or mismatched artifact. "
            "Re-extract before validating; clamping each to its own depth "
            "would make the cosine matrix asymmetric")
    return next(iter(distinct), 0)


def _matrix_layers(manifest, bundles) -> list[int]:
    """The layer(s) the cross-concept cosine matrix is computed at — one
    matrix PER declared depth, each internally single-layer.

    A study-wide declaration governs; otherwise mid-network is the documented
    canonical fallback. Deliberately NOT the per-concept legacy rule: a
    per-row layer makes a matrix asymmetric — (A,B) and (B,A) measured at
    different depths — and an asymmetric matrix has no defined reading for
    the ``maxCrossConceptCosine`` gate. A depth LIST yields one complete
    matrix per entry, never a mixed one.

    Swift twin: ``ExperimentTasks.matrixLayers``."""
    from . import validation_layer as vl
    depth = _require_uniform_depth(bundles)
    if depth <= 0:
        return [0]
    resolutions = vl.resolve_all(
        declared_layers=manifest.raw.get("validationLayers"),
        declared_fractions=manifest.raw.get("validationLayerFractions"),
        declared_layer=manifest.raw.get("validationLayer"),
        declared_fraction=manifest.raw.get("validationLayerFraction"),
        # No condition fallback: the matrix needs a STUDY layer, and a
        # per-concept one is exactly what makes it asymmetric.
        condition_layer=None,
        layer_count=depth)
    return [r.layer for r in resolutions]


def _validation_layer_resolutions(manifest: Manifest, concept_name: str,
                                  layer_count: int) -> list:
    """Every read layer AND why it is that layer (D4). The legacy rule —
    inherit from a steering condition, else mid-network — is preserved as the
    fallback, so existing manifests keep their single resolution; a declared
    scalar takes precedence, and a declared LIST yields one resolution per
    entry (the validate-at-the-sweep-layers policy: one run measures every
    depth). Swift twin: ``validationLayerResolutions``."""
    from . import validation_layer as vl
    condition_layer = None
    for condition in manifest.conditions:
        for slot in condition.slots:
            if slot.concept == concept_name:
                condition_layer = slot.layer
                break
        if condition_layer is not None:
            break
    refusal = vl.range_refusal(
        manifest.raw.get("validationLayer"), layer_count)
    if refusal:
        raise RuntimeError(refusal)
    return vl.resolve_all(
        declared_layers=manifest.raw.get("validationLayers"),
        declared_fractions=manifest.raw.get("validationLayerFractions"),
        declared_layer=manifest.raw.get("validationLayer"),
        declared_fraction=manifest.raw.get("validationLayerFraction"),
        condition_layer=condition_layer,
        layer_count=layer_count)


def _resolution_block(resolution) -> dict:
    """The cross-engine ``layerResolution`` report block for one depth."""
    return {
        "layer": resolution.layer,
        "layerCount": resolution.layer_count,
        "depthFraction": resolution.depth_fraction,
        "source": resolution.source,
    }


def _battery_results(manifest: Manifest, model, root, _log) -> list[dict]:
    """Run the pinned capability battery under baseline + every variant
    condition (greedy, ``BATTERY_MAX_TOKENS`` — mirrors Swift
    ``VariantRobustness``), scored with the pure exact/normalized matcher.
    Returns the per-condition evidence rows freeze's variant gate consumes:
    ``{condition, batteryHash, total, correct, accuracy}`` (an unloadable
    variant contributes an ``error`` row — the gate then names it missing).

    Arming follows the battery's FORMAT (2026-08-13 repair): a format-2
    battery declares its own rendering context and every condition — baseline
    included — is scored under it; a legacy battery keeps the historical
    behaviour (the manifest's context for baseline, the variant artifact's for
    each variant) so its pinned hash keeps its historical meaning, and earns a
    loud contamination advisory when a study system prompt is in play.

    SAE latent conditions cannot reach here, by construction rather than by
    omission: this path exists only for the freeze VARIANT gate, runs only
    when ``manifest.variant_conditions`` is non-empty, and enumerates baseline
    + variants. A latent arm is neither, and no freeze gate consumes
    validate-time battery evidence for one — its capability control is the
    RUN's battery (``_run_capability_battery``, which does score it). Adding a
    latent branch here would be dead code guarding nothing."""
    from . import battery as battery_mod, model_variant
    battery_file = battery_mod.battery_file(manifest)
    spec = battery_mod.load_spec(battery_file, root)
    items, digest = spec.items, spec.digest
    if manifest.capability_battery_hash and digest != manifest.capability_battery_hash:
        raise RuntimeError(
            f"capability battery '{battery_file}' drifted from the pinned hash "
            f"(have {digest[:12]}…, pinned {manifest.capability_battery_hash[:12]}…)")

    def _score(injections, prompt_mode, system_prompt, thinking, *,
               agent_system_prompt=None) -> int:
        # ``system_prompt`` is the FORMAT-1 caller context (unchanged, so a
        # legacy battery's pinned hash keeps its meaning);
        # ``agent_system_prompt`` is the format-2 persona, composed ahead of
        # the battery's own declared arming. The study frame reaches neither
        # on a format-2 reading — that is the isolation (2026-08-24 ruling).
        arming = battery_mod.resolve_arming(
            spec, prompt_mode=prompt_mode, system_prompt=system_prompt,
            qwen_thinking_enabled=thinking,
            agent_system_prompt=agent_system_prompt)
        advisory = battery_mod.contamination_advisory(spec, arming)
        if advisory:
            _log(f"WARNING: {advisory}")
        generate_fn, choice_fn = _battery_backends(
            model, manifest.model_id, injections)
        correct = 0
        for item in items:
            fields = battery_mod.score_item(
                spec, item, arming, generate_fn=generate_fn,
                choice_fn=choice_fn)
            if fields["correct"]:
                correct += 1
        return correct

    def _row(condition: str, correct: int) -> dict:
        return {"condition": condition, "batteryHash": digest,
                "total": len(items), "correct": correct,
                "accuracy": correct / len(items),
                "batteryFormat": spec.format_version,
                "armingIsolated": spec.isolated}

    results = [_row("baseline", _score([], manifest.prompt_mode,
                                       manifest.system_prompt,
                                       manifest.qwen_thinking_enabled))]
    for vc in manifest.variant_conditions:
        try:
            if vc.from_promotion:
                # Validate-time battery for a forward-referenced condition
                # works only once its agent is promoted; before that the
                # resolution raises and the row records why (the freeze
                # battery gate exempts forward refs for exactly this
                # reason — their battery evidence is the RUN's).
                vc, _ = _resolve_forward_variant(vc, manifest, root, _log)
            variant = (model_variant.ModelVariant.from_dict(vc.artifact)
                       if vc.artifact else model_variant.ModelVariant.from_file(
                           paths.resolve(vc.artifact_path, root)))
            injections = model_variant.variant_injections(variant)
            adapter = model_variant.apply_adapter(model, variant, root=root)
        except (OSError, KeyError, ValueError, RuntimeError) as exc:
            _log(f"battery: variant '{vc.name}' skipped: {exc}")
            results.append({"condition": vc.name, "batteryHash": digest,
                            "error": str(exc)})
            continue
        try:
            correct = _score(injections, variant.prompt_mode,
                             variant.system_prompt,
                             variant.qwen_thinking_enabled,
                             agent_system_prompt=variant.system_prompt)
        finally:
            model_variant.remove_adapter(model, adapter)
        results.append(_row(vc.name, correct))
    _log(f"capability battery ({len(items)} items × {len(results)} conditions)")
    return results


# Default sweep grid (recalibrated 2026-07-14, researcher decision,
# live-testing): stronger alphas routinely push models into wasteful
# incoherence, and the live optimum sits late in the network — L28/α0.08 on
# gemma-3-4b (≈0.82 depth) lies inside this grid. Depth fractions resolve
# against the model's layer count at sweep time (`resolve_sweep_layers`);
# alphas are residual-norm units. MUST stay identical to the Swift declare
# default (`ExperimentManifest.SweepSpec`). Explicit grids in a manifest's
# sweep spec always override these.
DEFAULT_SWEEP_LAYER_FRACTIONS = (0.5, 0.7, 0.85)
DEFAULT_SWEEP_ALPHAS = (0.05, 0.08, 0.1, 0.13)


#: Depth-fraction → block-index resolution for the spec'd sweep grid. Defined
#: in ``manifest`` and re-exported here, where it was written and where every
#: run-loop caller still reads it: the AUTHORING side has to resolve a grid too
#: (``experiment_store.set_sweep_grid`` converts absolute layers against it),
#: and that side must never import this module's torch-bearing run loop.
resolve_sweep_layers = manifest_mod.resolve_sweep_layers


def concept_sweep_layers(concept, vectors, layers: list[int], log) -> list[int]:
    """The layers a sweep may actually vary for one concept.

    Identity for every ordinary concept: a CAA / grand-mean / LAT direction
    exists at every depth, so a depth-fraction grid is a real axis and the
    declared grid stands untouched.

    For an imported **Gemma Scope SAE** decoder row it is not an axis at all.
    The artifact is full-depth zeros with the row placed at the SAE's own
    layer, because an SAE's dictionary lives at exactly one layer — the import
    verb already refuses a requested layer that disagrees with it ("a
    mis-specified import, not a choice"). Sweeping the declared fractions
    would inject a ZERO vector at every other cell: every such cell generates
    baseline text under a steered label, the selection rule then compares
    baseline against baseline, and the winner is noise. Silent, and expensive
    — the grid costs full generations per cell.

    So the axis COLLAPSES to the artifact's own nonzero layer, loudly logged.
    Chosen over refusing-until-declared for two reasons. (a) Consistency: the
    engine's rule for SAE features is already that the layer is a property of
    the artifact rather than a study choice, and the collapse states that rule
    where it bites instead of asking the researcher to restate it. Proposal r2
    §6 says the same thing in the study's own words — "the layer axis
    collapses to the dictionary layer, so the grid is an α ladder in both
    signs at one layer". (b) A refusal would demand the single layer be
    expressed as a DEPTH FRACTION, and `int(layer_count * f)` makes "layer 40
    of 62" a brittle arithmetic puzzle whose answer changes with the model —
    a gate no honest grid could reliably pass. Nothing is hidden: the log line
    names the collapse, the sweep's rows carry the one layer, and the
    recommendation's winningCell records it.

    Refuses (never guesses) when the artifact does not have exactly one
    nonzero layer: all-zero bytes are nothing to sweep, and several nonzero
    layers mean this is not a single decoder row, so "the" layer would be a
    coin flip.
    """
    if not concept.effective_method.is_gemma_scope_sae:
        return layers
    nonzero = [i for i in range(vectors.layer_count) if vectors.norm(i) > 0]
    if len(nonzero) != 1:
        raise RuntimeError(
            f"concept '{concept.name}' pins a Gemma Scope SAE feature, whose "
            f"vector must occupy exactly one layer (the SAE's own), but the "
            f"artifact is nonzero at {len(nonzero)} layer(s)"
            + (f" {nonzero}" if nonzero else "")
            + " — an all-zero artifact has nothing to sweep and a multi-layer "
            "one is not a single decoder row; re-import the feature")
    layer = nonzero[0]
    if layers != [layer]:
        log(f"{concept.name}: Gemma Scope SAE feature — collapsing the sweep's "
            f"layer grid {layers} to the dictionary's own layer {layer} "
            f"(every other cell would inject a zero vector); the grid is an "
            f"alpha ladder at L{layer}")
    return [layer]


def sweep(name: str, root: str | None = None, dtype: str = "auto",
          device: str | None = None, layer_fractions=(0.3, 0.4, 0.5, 0.6, 0.7),
          alphas=(-4.0, 2.0, 4.0, 8.0), prompt: str | None = None, *,
          model_provider=None, max_loaded: int | None = None,
          should_cancel: Callable[[], bool] | None = None,
          checkpoint=None, run_directory: str | None = None,
          on_run_directory=None,
          log=None) -> str:
    """Layer×alpha dose-response for picking per-concept settings on a dev split.

    When the manifest carries a ``sweep`` spec (the Swift-authored
    ``SweepSpec`` shape: grid, dev-prompts file, battery file, max tokens,
    optional ``selection`` criterion), the sweep converges with the Swift
    engine's: baseline cell + dev-prompt grid + capability battery per cell,
    then a selection step that appends ``<concept>-recommended`` (with full
    provenance) to a DRAFT manifest and writes ``recommendations.json``.
    Without a spec, the legacy single-prompt grid over the function arguments
    runs unchanged (backward compatibility for existing callers).

    ``max_loaded`` is the serving registry's resident-model capacity (the API
    route passes ``state.registry.max_loaded``); a judgeScore sweep whose
    local judge needs a SECOND resident model refuses at start on a one-slot
    server. ``None`` (the CLI/bundle path, which loads private in-process
    copies with no registry) skips that capacity check."""
    from . import sweep_selection
    manifest = Manifest.load(name, root)
    _verify_or_warn(manifest, root)
    _advise_sweep_ignores_sae_latent(manifest, log or print)
    spec = manifest.raw.get("sweep")
    spec = spec if isinstance(spec, dict) else None
    # Resolve the selection criterion AND its objective's instrument config
    # BEFORE loading the model: an unknown metric, a missing/empty choice
    # file, missing judge pins, or a credential-less Claude judge must fail
    # at sweep start, never after minutes of generation.
    criterion = objective = None
    if spec is not None:
        selection = spec.get("selection")
        criterion = sweep_selection.resolve_selection(selection)
        sel_objective = (selection or {}).get("objective") or {}
        declared_choice = sel_objective.get("choicePromptsFile")
        declared_choice_map = sel_objective.get("choicePromptsFiles")
        objective = sweep_selection.resolve_objective(
            criterion, selection,
            choice_path=(paths.resolve(declared_choice, root)
                         if declared_choice else None),
            choice_paths=({c: paths.resolve(p, root)
                           for c, p in declared_choice_map.items()
                           if isinstance(p, str)}
                          if isinstance(declared_choice_map, dict) else None),
            concepts=tuple(c.name for c in manifest.concepts),
            judge_rubric_file=manifest.judge_rubric_file,
            judge_rubric_hash=manifest.judge_rubric_hash,
            judge_refs=manifest.judges,
            judges_raw=(manifest.raw.get("judges") or []))
        if criterion.metric == "judgeScore":
            _judge_preflight(manifest, max_loaded, log or print)
    # The judge stack spans the WHOLE sweep, so a foreign local judge loads
    # once rather than once per comparison (external review round 4, finding
    # 4 — the same fix evaluate got in 10adf47d8). Nested INSIDE the study
    # model's acquire so the judge's slot is released first: it is the
    # second resident model, and it should not outlive the work it serves.
    with _acquire_model(manifest, dtype, device, model_provider) as model, \
            ExitStack() as judge_stack:
        manifest = _pin_model_revision(name, manifest, model, root, log or print)
        if spec is not None:
            return _sweep_with_spec(name, manifest, model, root, spec,
                                    criterion, objective, model_provider,
                                    should_cancel, log or print,
                                    judge_stack=judge_stack,
                                    checkpoint=checkpoint,
                                    resume_directory=run_directory,
                                    on_run_directory=on_run_directory)
        return _sweep_impl(name, manifest, model, root, layer_fractions, alphas,
                           prompt, should_cancel, log or print)


def _score_choice(model, manifest, prompt: str, options, injections):
    """One choice-instrument readout, armed exactly as the study path arms it
    (`_run_impl`): same renderer, same manifest prompt config, same injector
    gating. Split out so tests can fake the scoring boundary."""
    from . import logprob
    return logprob.score_options(
        model, prompt, list(options), model_id=manifest.model_id,
        injections=injections, prompt_mode=manifest.prompt_mode,
        system_prompt=manifest.system_prompt,
        qwen_thinking_enabled=manifest.qwen_thinking_enabled)


def _battery_backends(model, model_id: str, injections, latent_edits=None):
    """The two scoring back-ends a battery item can use, bound to one
    condition's arming (2026-08-13 battery repair).

    ``generate_fn`` and ``choice_fn`` both take the battery's own
    :class:`battery.BatteryArming` — NOT the study manifest's rendering
    context — so a format-2 battery is scored identically under baseline,
    steering, and variant conditions, and the intervention is the only thing
    that differs. ``choice_fn`` goes through the answer-token logprob
    instrument (the stepped KV-cache path a study's categorical outcome
    endpoints use), so
    nothing is generated and the score cannot move with response length or
    format compliance. Split out so tests can fake both boundaries.

    ``latent_edits`` arms a TRUE SAE latent intervention for the condition
    being scored (2026-08-13 review finding 2). It is a SEPARATE mechanism
    from ``injections`` for the same reason it is separate everywhere else,
    and — exactly like ``_execute_condition``'s ``latent_kwargs`` — it is
    threaded as a keyword ONLY when non-empty, so every non-latent
    condition's ``generate``/``score_options`` call is byte-for-byte the call
    it was before this argument existed. Without it a latent arm's battery
    scored the UNSTEERED model and filed the result under the latent
    condition's name: a capability control that measured nothing."""
    from . import logprob

    latent_kwargs = {"latent_edits": list(latent_edits)} if latent_edits else {}

    def generate_fn(prompt: str, arming) -> str:
        return generate(model, prompt, model_id=model_id,
                        max_tokens=arming.max_tokens, temperature=0.0,
                        injections=injections, **latent_kwargs,
                        prompt_mode=arming.prompt_mode,
                        system_prompt=arming.system_prompt,
                        qwen_thinking_enabled=arming.qwen_thinking_enabled)

    def choice_fn(prompt: str, options, arming):
        result = logprob.score_options(
            model, prompt, list(options), model_id=model_id,
            injections=injections, **latent_kwargs,
            prompt_mode=arming.prompt_mode,
            system_prompt=arming.system_prompt,
            qwen_thinking_enabled=arming.qwen_thinking_enabled)
        return result.selected, result.probability

    return generate_fn, choice_fn


def _choice_target_logprobs(model, manifest, choice_rows, injections, *,
                            should_cancel=None, log=None) -> dict[str, float]:
    """Per-row joint logprob of the TARGET option under the given injections
    (``{row id: logP(target)}``). The option-length guard runs per row — the
    baseline pass calls this before any grid generation, so a guard failure
    aborts the sweep at start, never mid-grid. A cancel is observed between
    rows (``TaskCancelled``)."""
    out: dict[str, float] = {}
    total = len(choice_rows)
    for i, row in enumerate(choice_rows, start=1):
        _cancel_checkpoint(should_cancel, log, f"choice row {i}/{total}")
        choice = _score_choice(model, manifest, row.prompt, row.options, injections)
        _check_option_lengths(choice, manifest, row.id)
        out[row.id] = next(
            s.logprob for s in choice.options if s.option == row.target)
    return out


def _mean_logprob_shift(cell: dict[str, float], baseline: dict[str, float]) -> float:
    """logprobShift objective: mean over choice rows of
    logP(target | cell) − logP(target | baseline). 0 for the baseline cell
    by construction."""
    if not baseline:
        return 0.0
    return sum(cell.get(k, 0.0) - v for k, v in baseline.items()) / len(baseline)


def _judge_preference(judge_panel, rubric: str, condition: str,
                      cell_texts, baseline_texts, *, prompts=None,
                      should_cancel=None, log=None) -> float:
    """judgeScore objective: mean paired-judge preference of the CELL text vs
    the same-prompt BASELINE text, blinded A/B (deterministic per item, the
    same seed-free convention as ``paired_judge._baseline_first``), mapped to
    [0, 1] with 0.5 = tie. ``judge_panel`` is ``[(name, judge_fn, model)]``
    with the ``paired_judge`` judge signature, so a local model or a test
    fake drives it without the network. A cancel is observed between judge
    calls (``TaskCancelled``)."""
    from . import paired_judge
    scores: list[float] = []
    for _name, judge_fn, judge_model in judge_panel:
        for i, (cell_text, base_text) in enumerate(zip(cell_texts, baseline_texts)):
            _cancel_checkpoint(should_cancel, log,
                               f"judge '{_name}' item {i + 1}/{len(cell_texts)}")
            baseline_is_a = paired_judge._baseline_first(f"dev-{i + 1}", condition)
            a, b = ((base_text, cell_text) if baseline_is_a
                    else (cell_text, base_text))
            # Canonical contract: the judge sees the task prompt the
            # responses answered — the SAME information set the deferred
            # (Mac) judging path gives it (engineer review 2026-07-18).
            # The verdict's winner is VALIDATED (retry once, then refuse —
            # `paired_judge.valid_verdict`, invalid-verdict closure
            # 2026-07-20): an out-of-vocabulary winner was silently scored
            # as an invented 0.5 tie, corrupting the preference mean.
            task_prompt = prompts[i] if prompts else None
            verdict = paired_judge.valid_verdict(
                judge_fn, judge_model, rubric, a, b, None,
                task_prompt=task_prompt,
                judge_label=f"'{_name}'",
                item_label=f"dev item {i + 1} of condition '{condition}'")
            winner = verdict.get("winner")
            if winner == "tie":
                scores.append(0.5)
            else:
                scores.append(0.0 if (winner == "A") == baseline_is_a else 1.0)
    return sum(scores) / len(scores) if scores else 0.5


def _judge_preflight(manifest: Manifest, max_loaded: int | None, _log) -> None:
    """judgeScore judge-panel preflight, run at sweep START — before the study
    model loads, so an instrument problem aborts the sweep at start, never
    mid-grid. Logs every judge's RESOLVED model (the cross-engine rule: a
    LOCAL judge with no/empty ``model`` judges with the STUDY model), and
    refuses when the panel needs more models resident SIMULTANEOUSLY than
    the registry allows. The count is over distinct
    (model, revision, canonical dtype) identities including the study model
    the sweep holds for the whole grid — two judges naming the same identity
    collapse into one load, two different ones do not (external review round
    5, finding 3). ``max_loaded`` is None on the CLI/bundle path (private
    in-process copies, no registry), where the co-residency check falls to
    the loader's GPU-capacity guard instead."""
    from . import paired_judge, sweep_selection
    # Every model that must be resident SIMULTANEOUSLY, as
    # (model, revision, canonical dtype). Seeded with the study model, which
    # the sweep holds for the whole grid.
    resident: set[tuple] = {(
        manifest.model_id,
        (manifest.model_revision or "").strip() or None,
        model_loader.normalize_dtype(manifest.dtype))}
    foreign_judges: list[str] = []
    # Same two announcements as evaluate, at sweep START — the sweep is
    # where a surprise costs the most, since it surfaces mid-grid after the
    # study model has loaded and cells have been generated.
    log_judging_custody(manifest.judges, _log)
    _preflight_openrouter_judges(manifest.judges, _log)
    for ref in manifest.judges:
        if ref.kind == "openrouter":
            _log(f"judge '{ref.name}': openrouter model '{ref.model}' via "
                 f"pinned provider '{getattr(ref, 'provider', None)}'")
            continue
        if ref.kind != "local":
            _log(f"judge '{ref.name}': claude model "
                 f"'{ref.model or paired_judge.DEFAULT_JUDGE_MODEL}'")
            continue
        resolved = sweep_selection.resolve_local_judge_model(
            ref.model, manifest.model_id)
        if resolved == manifest.model_id:
            # A study-model judge cannot pin a DIFFERENT identity: the sweep
            # judges with the already-held weights and never loads anything
            # else, so a divergent revision/dtype would be silently ignored
            # (external review round 5, finding 1). Freeze refuses this, but
            # a forced freeze or a hand-edited manifest can still arrive
            # here — refuse at START rather than judging with something the
            # manifest does not describe.
            declared_revision = (getattr(ref, "revision", None) or "").strip()
            if declared_revision and declared_revision != (
                    manifest.model_revision or "").strip():
                raise RuntimeError(
                    f"judge '{ref.name}' resolves to the STUDY model but "
                    f"pins revision '{declared_revision}', while the study "
                    f"runs '{manifest.model_revision or 'unpinned'}' — a "
                    "sweep judges with the held weights and cannot load a "
                    "second revision. Drop the judge's revision pin, or "
                    "name a different model")
            declared_dtype = (getattr(ref, "dtype", None) or "").strip()
            if declared_dtype and model_loader.normalize_dtype(declared_dtype) \
                    != model_loader.normalize_dtype(manifest.dtype or ""):
                raise RuntimeError(
                    f"judge '{ref.name}' resolves to the STUDY model but "
                    f"pins dtype '{declared_dtype}', while the study pins "
                    f"'{manifest.dtype or 'none (the device decides)'}' — a "
                    "sweep judges with the held weights and cannot load a "
                    "second precision. Drop the judge's dtype pin, or name "
                    "a different model")
            if (ref.model or "").strip():
                _log(f"judge '{ref.name}': local model '{resolved}' (the "
                     "study model — reuses the sweep's held model) at "
                     f"revision {manifest.model_revision or 'unpinned'}")
            else:
                _log(f"judge '{ref.name}': no model set — using the study "
                     f"model {manifest.model_id} at revision "
                     f"{manifest.model_revision or 'unpinned'}")
            continue
        _log(f"judge '{ref.name}': local model '{resolved}'")
        # Distinct RESIDENT IDENTITIES, not a per-judge yes/no (external
        # review round 5, finding 3). Two judges naming the same
        # model+revision+dtype collapse into ONE load (the same grouping
        # `evaluate_fanout_judge_models` uses), while two DIFFERENT foreign
        # judges need two slots on top of the study model's. Asking only
        # "is capacity >= 2" per judge passed a three-model panel on a
        # two-slot server, which then died partway through the grid.
        resident.add((
            resolved,
            (getattr(ref, "revision", None) or "").strip() or None,
            model_loader.normalize_dtype(getattr(ref, "dtype", None))))
        foreign_judges.append(f"'{ref.name}' ({resolved})")
    if max_loaded is not None and len(resident) > max_loaded:
        raise lifecycle_gates.refusing(
            lifecycle_gates.SWEEP_JUDGE_CAPACITY,
            f"judgeScore needs {len(resident)} models resident at once — "
            f"the study model '{manifest.model_id}' plus "
            f"{len(resident) - 1} distinct local judge model(s): "
            + ", ".join(foreign_judges)
            + f" — but this server keeps STEERLAB_MAX_LOADED_MODELS="
            f"{max_loaded}. The sweep holds its own slot for the whole "
            "grid, so every judge model needs one beside it: use the study "
            "model as judge, pin external judges, or raise the limit",
            repair=("set STEERLAB_MAX_LOADED_MODELS to at least "
                    f"{len(resident)} on this server, or re-pin the panel on "
                    "the Mac so every local judge resolves to the study model "
                    "(steerlab-cli experiment pin-rubric <name> <rubric> "
                    "--judges <name>:local)"))


def _assert_study_model_judge_matches_held(ref, manifest, model) -> None:
    """Refuse a study-model judge whose declared pins differ from the weights
    actually held (external review round 5, finding 1).

    A study-model judge has no independent identity — the sweep judges with
    the held model and never loads a second one — so a divergent pin would
    be silently ignored while staying in the criterion provenance. Freeze
    refuses it and the sweep preflight refuses it against the manifest; this
    is the check against reality, which is the only one an unpinned "let the
    device decide" dtype can be measured against.
    """
    declared_revision = (getattr(ref, "revision", None) or "").strip()
    held_revision = (getattr(model, "revision", None) or "").strip()
    if declared_revision and held_revision and declared_revision != held_revision:
        raise RuntimeError(
            f"judge '{ref.name}' pins revision '{declared_revision}' but the "
            f"sweep holds '{manifest.model_id}' at '{held_revision}' — a "
            "study-model judge judges with the held weights and cannot load "
            "a second revision. Drop the pin, or name a different model")
    declared_dtype = (getattr(ref, "dtype", None) or "").strip()
    held_dtype = _actual_dtype(model)
    if declared_dtype and held_dtype and \
            model_loader.normalize_dtype(declared_dtype) != \
            model_loader.normalize_dtype(held_dtype):
        raise RuntimeError(
            f"judge '{ref.name}' pins dtype '{declared_dtype}' but the sweep "
            f"holds '{manifest.model_id}' as '{held_dtype}' — a study-model "
            "judge judges with the held weights and cannot load a second "
            "precision. Drop the pin, or name a different model")


def _sweep_judge_panel(manifest, model, model_provider, root, _log, *,
                       judge_stack=None):
    """(rubric text, [(name, judge_fn, model)]) for a judgeScore sweep. The
    rubric comes from the manifest PINS (resolve_objective already required
    them; ``_resolve_rubric`` drift-checks the file at read time).

    Judge models resolve by the cross-engine SWEEP rule
    (``sweep_selection.resolve_local_judge_model``): a LOCAL judge with
    no/empty ``model`` judges with the STUDY model. A local judge whose
    resolved model IS the study model reuses the sweep's already-HELD
    ``model`` object directly — never a second registry acquire, which on a
    one-slot server would find the only slot locked by the sweep itself (and
    with the same slot key would self-deadlock on the non-reentrant slot
    lock). Revision compatibility is by construction: the held model IS the
    manifest-pinned one.

    Different-model local judges keep the provider path (valid on multi-slot
    servers; the sweep-start preflight already refused them when capacity is
    1), and now hold their slot on ``judge_stack`` for the WHOLE sweep
    (external review round 4, finding 4). Without it each comparison entered
    and exited the provider itself, and `model_loader.load` has no cache — a
    12B judge across a grid of cells reloaded on every pair."""
    from . import paired_judge, response_coding, sweep_selection
    rubric, _live_hash, _file = _resolve_rubric(manifest, root, _log)
    # A coding rubric declares no preference — the sweep's judgeScore
    # objective would force the judge to improvise a winner (2026-08-04).
    response_coding.refuse_if_coding(
        rubric, context="the sweep's judgeScore objective",
        rubric_file=_file)
    panel = []
    for ref in manifest.judges:
        if ref.kind == "local":
            resolved = sweep_selection.resolve_local_judge_model(
                ref.model, manifest.model_id)
            if resolved == manifest.model_id:
                # Last check against the model actually in hand (external
                # review round 5, finding 1). Freeze refuses a divergent
                # study-model judge pin and `_judge_preflight` refuses it
                # again at sweep start against the MANIFEST; this compares
                # against the loaded weights, which is the only place the
                # real identity is knowable — an unpinned study dtype
                # resolves per device, so only the load knows what it became.
                _assert_study_model_judge_matches_held(ref, manifest, model)

                def _gen(prompt: str) -> str:
                    # JUDGE_MAX_TOKENS, never a smaller ad-hoc cap: the
                    # 2026-07-22 incident cap (512) truncated a legible
                    # verdict mid-reasoning and the run refused it.
                    return generate(model, prompt, model_id=manifest.model_id,
                                    max_tokens=paired_judge.JUDGE_MAX_TOKENS,
                                    temperature=0.0,
                                    prompt_mode=prompt_render.CHAT_ASSISTANT)
                panel.append((ref.name, paired_judge.make_local_judge(_gen),
                              resolved))
                continue
        judge_fn, requested_model, _holder = _judge_callable(
            ref, model_provider, study_model=manifest.model_id,
            study_revision=manifest.model_revision, stack=judge_stack)
        panel.append((ref.name, judge_fn, requested_model))
    return rubric, panel


def _emit_judging_packets(packets, packet_map, concept, kind, layer, alpha,
                          prompts, steered_texts, baseline_texts,
                          rubric_hash) -> None:
    """One blinded comparison packet per dev item for a deferred sweep cell
    (or its matched-norm control). The A/B orientation uses EXACTLY the
    inline convention (``paired_judge._baseline_first`` over the same
    condition tag), so a deferred selection is bit-comparable to what the
    inline path would have judged. The judge-visible packet carries ONLY
    prompt + responses; cell identity and orientation live in the separate
    map the judging client never consumes."""
    from . import paired_judge
    tag = (f"sweep:{concept}:L{layer}:a{alpha:g}" if kind == "cell"
           else f"sweep-control:{concept}:L{layer}:a{alpha:g}")
    for i, (steered, base) in enumerate(zip(steered_texts, baseline_texts)):
        item_id = f"dev-{i + 1}"
        baseline_is_a = paired_judge._baseline_first(item_id, tag)
        a, b = (base, steered) if baseline_is_a else (steered, base)
        packet_id = hashlib.sha256(
            f"{tag}|{item_id}|{rubric_hash}|{a}|{b}".encode("utf-8")).hexdigest()
        packets.append({"packetID": packet_id, "prompt": prompts[i],
                        "responseA": a, "responseB": b})
        packet_map[packet_id] = {
            "concept": concept, "kind": kind, "layer": layer, "alpha": alpha,
            "item": item_id, "baselineIsA": baseline_is_a,
            "conditionTag": tag}


def _normalized_judge_entries(raw, study_model: str | None = None) -> list[dict]:
    """Judge entries with kind and model RESOLVED (engineer review
    2026-07-18, second pass): kind defaults to claude, and an empty model
    pins the server's DEFAULT_JUDGE_MODEL at emission — the judging client
    must never resolve against its own ambient default. OpenRouter judges
    (2026-07-19) have NO defaults to fill: an explicit model slug and a
    pinned provider are required — normalization refuses rather than
    inventing either. LOCAL judges (judge fan-out, 2026-07-23): a blank
    model resolves to ``study_model`` when the caller provides it — the
    cross-engine local-judge rule — so worker judgments verify against the
    emission pin."""
    from . import paired_judge
    out: list[dict] = []
    for j in raw or ():
        entry = dict(j or {})
        entry["kind"] = entry.get("kind") or "claude"
        if entry["kind"] == "local" and study_model:
            entry["model"] = (str(entry.get("model") or "").strip()
                              or study_model)
            out.append(entry)
            continue
        if entry["kind"] == "openrouter":
            model = str(entry.get("model") or "").strip()
            provider = str(entry.get("provider") or "").strip()
            if not model:
                raise ValueError(
                    f"openrouter judge '{entry.get('name')}' has no model "
                    "slug — there is no default to pin at emission")
            if not provider:
                raise ValueError(
                    f"openrouter judge '{entry.get('name')}' has no pinned "
                    "provider — an unpinned provider is not a pinned judge")
            entry["model"], entry["provider"] = model, provider
            out.append(entry)
            continue
        entry["model"] = (str(entry.get("model") or "").strip()
                          or paired_judge.DEFAULT_JUDGE_MODEL)
        out.append(entry)
    return out


def _write_deferred_judging(run_directory, name, manifest, criterion,
                            objective, packets, packet_map, selection_ctx,
                            dev_hash, rubric_text, rubric_hash, _log) -> None:
    """The awaiting-judgment artifact set, written into the (still-being-
    created) sweep run directory: judge-visible packets (hash-pinned),
    the identity/orientation map, the selection context the completion verb
    replays, and a manifest binding it all to the experiment EPOCH."""
    def _sha256_of(path: str) -> str:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    packets_path = os.path.join(run_directory, "judging-packets.jsonl")
    with open(packets_path, "w", encoding="utf-8") as handle:
        for row in packets:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    packets_sha = _sha256_of(packets_path)
    map_path = os.path.join(run_directory, "judging-map.json")
    with open(map_path, "w", encoding="utf-8") as handle:
        json.dump({"packets": packet_map}, handle, indent=2, sort_keys=True)
    ctx_path = os.path.join(run_directory, "deferred-selection.json")
    with open(ctx_path, "w", encoding="utf-8") as handle:
        json.dump({"criterion": criterion.to_dict(objective),
                   "devPromptsHash": dev_hash,
                   "manifestStatus": manifest.status,
                   "concepts": selection_ctx}, handle, indent=2,
                  sort_keys=True)
    # EVERY interpretation artifact is pinned, not just the packets
    # (engineer review 2026-07-18): the map decides orientation and cell
    # identity, the selection context decides constraints — tampering with
    # either could flip the selected agent while the packet hash and epoch
    # still pass.
    judging_manifest = {
        "kind": "sweep",  # vs "evaluate" — scanners filter by this
        "experiment": name,
        "experimentHash": manifest.content_hash(),
        "sweepRun": os.path.basename(run_directory),
        "rubricFile": manifest.judge_rubric_file,
        "rubricHash": rubric_hash,
        "rubric": rubric_text,
        # The hash of the EXACT text embedded above — what the Mac's judging
        # preflight verifies (the file pin can differ from the text hash
        # across newline conventions; the judge reads the text).
        "rubricTextSha256": hashlib.sha256(
            rubric_text.encode("utf-8")).hexdigest(),
        "judges": _normalized_judge_entries(objective.judges),
        "packetsFile": "judging-packets.jsonl",
        "packetsSha256": packets_sha,
        "mapSha256": _sha256_of(map_path),
        "selectionContextSha256": _sha256_of(ctx_path),
        "packetCount": len(packets),
        "devPromptsHash": dev_hash,
    }
    with open(os.path.join(run_directory, "judging-manifest.json"), "w",
              encoding="utf-8") as handle:
        json.dump(judging_manifest, handle, indent=2, sort_keys=True)
    with open(os.path.join(run_directory, "awaiting-judgment.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"packetCount": len(packets),
                   "judgingManifest": "judging-manifest.json"},
                  handle, indent=2, sort_keys=True)
    _log(f"awaiting judgment: {len(packets)} blinded packets emitted — "
         "judge on the Mac, then complete-judgment computes the selection")


def list_awaiting_judgment(name: str, root: str | None = None) -> list[dict]:
    """Sweep runs for ``name`` that emitted judging packets and have no
    completion run referencing them yet — what the app's "judge on this Mac"
    affordance lists."""
    runs_root = paths.runs_directory(root)
    try:
        entries = sorted(os.listdir(runs_root))
    except OSError:
        return []
    completed: set[str] = set()
    for entry in entries:
        src = os.path.join(runs_root, entry, "judgment-source.json")
        if not os.path.exists(src):
            continue
        try:
            with open(src, encoding="utf-8") as handle:
                marker = json.load(handle)
        except (OSError, ValueError):
            continue
        # Evaluate-judgment markers have their own scanner (kind-keyed);
        # a sweep marker carries no kind (pre-2026-07-19) or "sweep".
        if marker.get("kind") not in (None, "sweep"):
            continue
        sweep_ref = str(marker.get("sweepRun"))
        try:
            # Only a VERIFIED completion record suppresses "awaiting" —
            # a bare/altered marker must not hide judgable work (engineer
            # review 2026-07-18, third pass).
            _verify_judgment_marker(
                os.path.join(runs_root, entry), marker, name=name,
                sweep_jm=_sweep_judging_manifest(runs_root, sweep_ref))
        except ValueError:
            continue
        completed.add(sweep_ref)
    out: list[dict] = []
    for entry in entries:
        jm_path = os.path.join(runs_root, entry, "judging-manifest.json")
        if not os.path.exists(jm_path):
            continue
        try:
            with open(jm_path, encoding="utf-8") as handle:
                jm = json.load(handle)
        except (OSError, ValueError):
            continue
        if jm.get("experiment") != name or entry in completed:
            continue
        # Evaluate-awaiting runs share the artifact shape but have their own
        # scanner + completion verb; legacy manifests without a kind are
        # sweeps (the only kind that existed before 2026-07-19).
        if jm.get("kind", "sweep") != "sweep":
            continue
        out.append({
            "run": entry,
            "packetCount": jm.get("packetCount"),
            "judges": jm.get("judges"),
            "rubricFile": jm.get("rubricFile"),
            "rubricHash": jm.get("rubricHash"),
            "rubricTextSha256": jm.get("rubricTextSha256"),
            "rubric": jm.get("rubric"),
            "experimentHash": jm.get("experimentHash"),
            "packetsFile": jm.get("packetsFile"),
            "packetsSha256": jm.get("packetsSha256"),
        })
    return out


JUDGMENT_MARKER_SCHEMA = 1


def _verify_judgment_marker(run_dir: str, marker: dict, *, name: str,
                            sweep_jm: dict | None) -> None:
    """Strict verification of a completion record (engineer review
    2026-07-18, third pass): a marker is CANONICAL — it suppresses
    "awaiting" and projects conditions into the manifest — so a bare or
    altered one must never be trusted. Schema-versioned, bound to the
    experiment identity and the sweep's packet pin, and hash-linked to the
    run's own judgment artifacts. Raises ValueError naming the run."""
    run = os.path.basename(run_dir)

    def _refuse(why: str):
        raise ValueError(
            f"completion record in '{run}' failed verification ({why}) — "
            "an unverified judgment run is never canonical; inspect it by "
            "hand (a genuine one re-heals by re-running completion)")

    if marker.get("schema") != JUDGMENT_MARKER_SCHEMA:
        _refuse(f"schema {marker.get('schema')!r}, expected "
                f"{JUDGMENT_MARKER_SCHEMA}")
    if marker.get("experiment") != name:
        _refuse(f"experiment {marker.get('experiment')!r}, expected {name!r}")
    # FAIL CLOSED on missing sweep evidence (engineer review 2026-07-18,
    # fourth pass): no readable judging manifest means no pin to verify
    # against — an unverifiable record is never canonical.
    if sweep_jm is None:
        _refuse("the sweep run's judging manifest cannot be read — no "
                "packet pin to verify against")
    if marker.get("packetsSha256") != sweep_jm.get("packetsSha256"):
        _refuse("packet pin does not match the sweep's judging manifest")
    if marker.get("experimentHashAtJudgment") != sweep_jm.get("experimentHash"):
        _refuse("judgment epoch does not match the sweep's experiment hash")
    for artifact, key in (("judgments.jsonl", "judgmentsSha256"),
                          ("recommendations.json", "recommendationsSha256")):
        stamped = marker.get(key)
        if not stamped:
            _refuse(f"no {key} stamp")
        path = os.path.join(run_dir, artifact)
        try:
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            _refuse(f"{artifact} is missing")
        if digest != stamped:
            _refuse(f"{artifact} does not hash to its stamp")


def _sweep_judging_manifest(runs_root: str, sweep_run: str) -> dict | None:
    """``sweep_run``'s judging manifest, or None when unreadable — in which
    case the marker verifier REFUSES (fail closed): a completion record
    with no recoverable sweep evidence is never canonical."""
    try:
        with open(os.path.join(runs_root, sweep_run,
                               "judging-manifest.json"),
                  encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _find_judgment_run(name: str, sweep_run: str,
                       root: str | None) -> tuple[str, dict] | None:
    """``(run_dir, VERIFIED marker)`` of the completed judgment run for
    ``sweep_run``, else None. A candidate that fails verification RAISES —
    silently ignoring a corrupt canonical record would re-run judging on
    top of it."""
    runs_root = paths.runs_directory(root)
    try:
        entries = sorted(os.listdir(runs_root))
    except OSError:
        return None
    for entry in entries:
        src = os.path.join(runs_root, entry, "judgment-source.json")
        if not os.path.exists(src):
            continue
        try:
            with open(src, encoding="utf-8") as handle:
                marker = json.load(handle)
        except (OSError, ValueError):
            # An unreadable marker cannot even say which sweep it refers
            # to: skip it (our own writer is atomic, so this is external
            # damage) rather than letting one damaged run anywhere under
            # runs/ block every unrelated completion (engineer review
            # 2026-07-18, fourth pass). Immutability means we never
            # overwrite it; a re-judge creates a fresh run.
            continue
        if marker.get("sweepRun") != sweep_run:
            continue
        run_dir = os.path.join(runs_root, entry)
        _verify_judgment_marker(
            run_dir, marker, name=name,
            sweep_jm=_sweep_judging_manifest(runs_root, sweep_run))
        return run_dir, marker
    return None


def _conditions_from_recommendations(run_dir: str) -> list[dict]:
    """The projection payload, RECONSTRUCTED from the judgment run's
    hash-verified ``recommendations.json`` (engineer review 2026-07-18,
    fourth pass): a marker-carried conditions list was an independent claim
    the verifier never bound to the judged artifacts — deriving conditions
    from the verified selection blocks makes them a pure function of the
    evidence. String entries (failures / gate refusals) project nothing."""
    with open(os.path.join(run_dir, "recommendations.json"),
              encoding="utf-8") as handle:
        recommendations = json.load(handle)
    out: list[dict] = []
    for concept, block in sorted((recommendations or {}).items()):
        if not isinstance(block, dict):
            continue
        cell = block.get("winningCell") or {}
        out.append({"name": f"{concept}-recommended",
                    "slots": [{"concept": concept,
                               "layer": cell.get("layer"),
                               "alpha": cell.get("alpha")}],
                    "bandWidth": 1, "alphaInNormUnits": True,
                    "selection": block})
    return out


def _project_judgment_conditions(name, manifest, run_dir, root, _log) -> None:
    """Apply a completed judgment run's recommended conditions to the DRAFT
    manifest, IDEMPOTENTLY (engineer review 2026-07-18, second pass:
    marker-last ordering alone left a crash window where the manifest had
    changed while the run still read "awaiting" — and a partial multi-
    concept append had no recovery). Rules: a same-name condition from THIS
    judgment run is already projected (skip); a missing one appends; a
    same-name condition from any OTHER source is a real conflict (refuse,
    by hand). Safe to re-run any number of times; the payload comes from
    the verified recommendations, never from the marker."""
    from . import experiment_store
    conditions = _conditions_from_recommendations(run_dir)
    if not conditions:
        return
    if manifest.status != "draft":
        _log(f"manifest is {manifest.status} — recommendations reported "
             "only (no conditions projected)")
        return
    existing = {str(c.get("name")): c
                for c in (manifest.raw.get("conditions") or [])}
    for condition in conditions:
        cname = str(condition.get("name"))
        current = existing.get(cname)
        if current is not None:
            mine = (condition.get("selection") or {}).get("judgmentRun")
            theirs = (current.get("selection") or {}).get("judgmentRun")
            if mine == theirs:
                continue  # already projected — idempotent
            raise ValueError(
                f"manifest already carries condition '{cname}' from a "
                "different source — resolve by hand before re-projecting")
        experiment_store.add_condition(name, condition, root)
        _log(f"recommended condition '{cname}' written into draft manifest")


def complete_sweep_judgment(name: str, sweep_run: str, judgments: list,
                            root: str | None = None, log=None) -> str:
    """Phase 2 of a two-phase (deferred, Claude-judged) sweep: verify the
    judgment set against the sweep's pinned packets and the experiment
    EPOCH, replay the judgeScore selection exactly as the inline path
    computes it, write an immutable judgment run directory, and append
    ``<concept>-recommended`` to a DRAFT manifest.

    CPU-only by construction (no model, no credential) — it runs on the
    controller. Refusals are exhaustive because judgments arrive from
    outside the run's own process: unknown packets, unpinned judges,
    duplicate or missing (packet × judge) pairs, packet-file drift, and
    manifest-epoch drift each name themselves."""
    from . import experiment_store, sweep_selection as sel
    _log = log or print
    if not sweep_run or "/" in sweep_run or os.sep in sweep_run \
            or sweep_run in (".", ".."):
        raise ValueError(f"bad sweep run name {sweep_run!r}")
    manifest = Manifest.load(name, root)
    _verify_or_warn(manifest, root)
    # IDEMPOTENT: if this sweep's judgment run already exists, the judging
    # is done — re-running completion only heals the manifest PROJECTION
    # (recovery from a crash between the marker and the appends). The epoch
    # gate below is deliberately skipped here: the appends themselves
    # change the manifest hash, and the epoch was proven before the first
    # write (stamped in the marker for audit).
    existing = _find_judgment_run(name, sweep_run, root)
    if existing is not None:
        run_directory, _marker = existing
        _project_judgment_conditions(name, manifest, run_directory, root,
                                     _log)
        _log(f"sweep judgment already completed → {run_directory} "
             "(projection verified)")
        return run_directory
    runs_root = paths.runs_directory(root)
    sweep_dir = os.path.join(runs_root, sweep_run)
    jm_path = os.path.join(sweep_dir, "judging-manifest.json")
    if not os.path.exists(jm_path):
        raise ValueError(
            f"run '{sweep_run}' has no judging-manifest.json — not an "
            "awaiting-judgment sweep run")
    with open(jm_path, encoding="utf-8") as handle:
        jm = json.load(handle)
    if jm.get("experiment") != name:
        raise ValueError(
            f"run '{sweep_run}' belongs to experiment "
            f"'{jm.get('experiment')}', not '{name}'")
    # The manifest must name the run directory it lives in (engineer review
    # 2026-07-18, emission-pin pass) — a copied/renamed awaiting dir must
    # not complete under another run's identity.
    if jm.get("sweepRun") != sweep_run:
        raise ValueError(
            f"the judging manifest names sweep run '{jm.get('sweepRun')}' "
            f"but lives in '{sweep_run}' — refusing a relocated awaiting "
            "run")
    live_hash = manifest.content_hash()
    if jm.get("experimentHash") != live_hash:
        raise ValueError(
            f"experiment epoch mismatch: the sweep ran under manifest hash "
            f"{str(jm.get('experimentHash'))[:12]}… but '{name}' now hashes "
            f"{live_hash[:12]}… — judgments cannot complete a selection for "
            "a drifted manifest (duplicate the experiment and re-sweep)")
    def _verify_pinned_file(path: str, stamped: str | None,
                            what: str) -> bytes:
        if not stamped:
            raise ValueError(
                f"the judging manifest carries no hash for {what} — this "
                "awaiting run predates full artifact pinning; re-sweep")
        with open(path, "rb") as handle:
            data = handle.read()
        digest_ = hashlib.sha256(data).hexdigest()
        if digest_ != stamped:
            raise ValueError(
                f"{what} drifted since emission (sha256 mismatch) — the "
                "sweep run directory must be immutable; re-sweep")
        return data

    # EVERY interpretation artifact verifies against its emission pin
    # (engineer review 2026-07-18): packets alone were not enough — the map
    # decides orientation/cell identity and the selection context decides
    # constraints, so either could flip the selected agent silently.
    packets_path = os.path.join(
        sweep_dir, jm.get("packetsFile") or "judging-packets.jsonl")
    _verify_pinned_file(packets_path, jm.get("packetsSha256"),
                        "the judging packets file")
    with open(packets_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    map_bytes = _verify_pinned_file(
        os.path.join(sweep_dir, "judging-map.json"), jm.get("mapSha256"),
        "the judging map")
    packet_map = json.loads(map_bytes.decode("utf-8"))["packets"]
    ctx_bytes = _verify_pinned_file(
        os.path.join(sweep_dir, "deferred-selection.json"),
        jm.get("selectionContextSha256"), "the deferred selection context")
    ctx = json.loads(ctx_bytes.decode("utf-8"))

    # The rubric and the judge panel come from the LIVE manifest (the epoch
    # check above proved it identical to sweep time) — the judging
    # manifest's copies must agree, or something was edited in place.
    if jm.get("rubricHash") != manifest.judge_rubric_hash:
        raise ValueError(
            "the judging manifest's rubric hash does not match the "
            "manifest's pinned judgeRubricHash — re-sweep")
    _resolve_rubric(manifest, root, lambda *_: None)  # file-drift check
    # The EMITTED panel is authoritative for models (engineer review
    # 2026-07-18, third pass): re-normalizing the live manifest here would
    # resolve empty models against the server's CURRENT env default
    # (STEERLAB_JUDGE_MODEL), refusing valid judgments if it changed
    # between sweep and completion. The epoch gate already proved the live
    # manifest byte-identical to sweep time, so only structure is compared;
    # an EXPLICIT live model must match the pin, an empty one accepts the
    # value emission resolved.
    live_raw = [dict(j or {}) for j in (manifest.raw.get("judges") or [])]
    jm_judges = [dict(j or {}) for j in (jm.get("judges") or [])]
    if len(live_raw) != len(jm_judges):
        raise ValueError(
            "the judging manifest's judge panel does not match the live "
            "manifest's pinned judges — re-sweep")
    pinned_model_by_judge: dict[str, str] = {}
    pinned_provider_by_judge: dict[str, str | None] = {}
    for raw_j, pinned in zip(live_raw, jm_judges):
        if (str(raw_j.get("name")) != str(pinned.get("name"))
                or (raw_j.get("kind") or "claude")
                != (pinned.get("kind") or "claude")):
            raise ValueError(
                "the judging manifest's judge panel does not match the "
                "live manifest's pinned judges — re-sweep")
        declared = str(raw_j.get("model") or "").strip()
        if declared and declared != str(pinned.get("model")):
            raise ValueError(
                f"judge '{raw_j.get('name')}' declares model '{declared}' "
                f"but the sweep pinned '{pinned.get('model')}' — re-sweep")
        declared_provider = str(raw_j.get("provider") or "").strip()
        if declared_provider and declared_provider != str(
                pinned.get("provider") or ""):
            raise ValueError(
                f"judge '{raw_j.get('name')}' declares provider "
                f"'{declared_provider}' but the sweep pinned "
                f"'{pinned.get('provider')}' — re-sweep")
        pinned_model_by_judge[str(pinned.get("name"))] = \
            str(pinned.get("model"))
        pinned_provider_by_judge[str(pinned.get("name"))] = (
            str(pinned.get("provider") or "").strip() or None
            if (pinned.get("kind") or "claude") == "openrouter" else None)

    spec = manifest.raw.get("sweep") or {}
    criterion = sel.resolve_selection(spec.get("selection"))
    if criterion.metric != "judgeScore":
        raise ValueError(
            f"live selection criterion is '{criterion.metric}', but the "
            "awaiting run judged 'judgeScore' — the manifest drifted")
    judge_names = set(pinned_model_by_judge)

    seen: dict[tuple[str, str], str] = {}
    for row in judgments:
        row = row or {}
        pid = str(row.get("packetID") or "")
        judge = str(row.get("judge") or "")
        winner = row.get("winner")
        if pid not in packet_map:
            raise ValueError(f"judgment for unknown packet '{pid[:16]}…'")
        if judge not in judge_names:
            raise ValueError(
                f"judgment by unpinned judge {judge!r} — the sweep pinned "
                f"{sorted(judge_names)}")
        if winner not in ("A", "B", "tie"):
            raise ValueError(
                f"judgment winner must be 'A', 'B', or 'tie', got {winner!r}")
        # The model is pinned at EMISSION; every judgment must carry it and
        # match it (engineer review 2026-07-18, second pass — a recorded
        # string is provenance only if it is verified).
        model = str(row.get("model") or "").strip()
        if not model:
            raise ValueError(
                f"judgment by {judge!r} carries no model — the judging "
                "client must stamp the pinned Claude model (update the app)")
        if model != pinned_model_by_judge.get(judge):
            raise ValueError(
                f"judge {judge!r} judged with model '{model}' but the sweep "
                f"pinned '{pinned_model_by_judge.get(judge)}' — refusing")
        provider = _verify_judgment_provider(
            row, judge, pinned_provider_by_judge.get(judge))
        confidence, verdict = _verified_judgment_payload(row, judge, winner)
        key = (pid, judge)
        if key in seen:
            raise ValueError(
                f"duplicate judgment for packet '{pid[:16]}…' by {judge!r}")
        seen[key] = (winner, row.get("model"), provider, confidence, verdict)
    expected = len(packet_map) * len(judge_names)
    if len(seen) != expected:
        raise ValueError(
            f"incomplete judgments: {len(seen)} of {expected} "
            "(packet × judge) pairs — every pinned judge must judge every "
            "packet")

    # Scores per (concept, kind, layer, alpha) — the inline mapping exactly:
    # tie 0.5; baseline wins 0.0; the steered/control text wins 1.0.
    buckets: dict[tuple, list[float]] = {}
    for (pid, judge), (winner, _model, _provider, _conf, _verdict) \
            in seen.items():
        meta = packet_map[pid]
        if winner == "tie":
            score = 0.5
        else:
            score = 0.0 if (winner == "A") == bool(meta["baselineIsA"]) else 1.0
        key = (meta["concept"], meta["kind"], int(meta["layer"]),
               float(meta["alpha"]))
        buckets.setdefault(key, []).append(score)

    objective_stub = sel.ResolvedObjective(
        metric="judgeScore", judge_rubric_file=jm.get("rubricFile"),
        judge_rubric_hash=jm.get("rubricHash"),
        judges=tuple(jm.get("judges") or ()))
    # PHASE A — compute every concept's outcome IN MEMORY (engineer review
    # 2026-07-18: completion must be transactional — a failure on the last
    # concept must not leave earlier conditions appended or a completion
    # marker hiding an unfinished run). No filesystem or manifest write
    # happens until every concept has an answer.
    recommendations: dict = {}
    for concept_name, cinfo in sorted((ctx.get("concepts") or {}).items()):
        base_info = cinfo["baseline"]
        baseline = sel.BaselineCell(
            metric=sel.baseline_metric("judgeScore",
                                       float(base_info["markerDensity"])),
            distinct2=float(base_info["distinct2"]),
            battery_accuracy=float(base_info["batteryAccuracy"]))
        cells: list[sel.SweepCell] = []
        for c in cinfo["cells"]:
            scores = buckets.get(
                (concept_name, "cell", int(c["layer"]), float(c["alpha"])))
            if not scores:
                raise ValueError(
                    f"no judgments for cell {concept_name} L{c['layer']} "
                    f"α{c['alpha']:g}")
            cells.append(sel.SweepCell(
                layer=int(c["layer"]), alpha=float(c["alpha"]),
                metric=sum(scores) / len(scores),
                distinct2=float(c["distinct2"]),
                battery_accuracy=float(c["batteryAccuracy"]),
                # A deferred context written before the words field leaves
                # the length unrecorded — the winner's lengthInflated stamp
                # is then absent rather than invented.
                words=(float(c["words"])
                       if c.get("words") is not None else None)))
        best = sel.select_cell(cells, baseline, criterion)
        if best is None:
            # Say WHICH gate refused. "Capability/coherence" is one of two
            # possible reasons and often the wrong one — a grid whose cells
            # are all eligible but none of which beats the baseline objective
            # is a different result entirely, and reporting it as a gate
            # failure sends the researcher to loosen a tolerance that was
            # never binding.
            recommendations[concept_name] = sel.no_selection_reason(
                cells, baseline, criterion)
            _append_progress({"kind": "recommendation",
                              "concept": concept_name,
                              "block": recommendations[concept_name]})
            continue
        control_info = None
        if criterion.matched_norm_random_margin is not None:
            cscores = buckets.get(
                (concept_name, "control", best.layer, best.alpha))
            if not cscores:
                raise ValueError(
                    f"missing control judgments for the winning cell "
                    f"{concept_name} L{best.layer} α{best.alpha:g}")
            control_metric = sum(cscores) / len(cscores)
            control_info = {"type": "randomMatchedNorm",
                            "metricValue": control_metric,
                            "margin": criterion.matched_norm_random_margin,
                            "randomVectorAlgorithm": RANDOM_VECTOR_ALGORITHM}
            if not sel.control_passes(best.metric, control_metric,
                                      criterion.matched_norm_random_margin):
                message = sel.control_failure_message(
                    best.metric, control_metric,
                    criterion.matched_norm_random_margin)
                recommendations[concept_name] = message
                _log(f"{concept_name}: {message}")
                continue
        selection_block: dict = {
            "sweepRun": sweep_run,
            "judgedOn": "client",
            "packetsSha256": digest,
            "criterion": criterion.to_dict(objective_stub),
            "devPromptsHash": ctx.get("devPromptsHash"),
            "winningCell": {"layer": best.layer, "alpha": best.alpha},
            "metrics": {"judgeScore": best.metric,
                        "baselineJudgeScore": baseline.metric,
                        "distinct2": best.distinct2,
                        "batteryAccuracy": best.battery_accuracy,
                        "baselineBatteryAccuracy": baseline.battery_accuracy,
                        # Same report pair the inline path stamps — the
                        # coherence gate's own evidence, in the metrics the
                        # promotion certificate copies.
                        **sel.selection_report_metrics(
                            best.distinct2, baseline.distinct2, best.words,
                            (float(base_info["words"])
                             if base_info.get("words") is not None
                             else None))},
        }
        if control_info is not None:
            selection_block["control"] = control_info
        recommendations[concept_name] = selection_block

    # PHASE B — writes, completion marker LAST: the awaiting-run scanner
    # keys on judgment-source.json, so it must exist only once everything
    # else (run artifacts + manifest appends) has landed. A crash anywhere
    # earlier leaves the run honestly "awaiting" and re-judgeable.
    run_directory = paths.make_unique_run_directory(
        f"exp-{name}-sweep-judgment", root)
    judgment_run = os.path.basename(run_directory)
    for block in recommendations.values():
        if isinstance(block, dict):
            block["judgmentRun"] = judgment_run
    _write_config_snapshot(manifest, run_directory, "sweep-judgment")
    with open(os.path.join(run_directory, "judgments.jsonl"), "w",
              encoding="utf-8") as handle:
        for (pid, judge), (winner, judge_model, judge_provider,
                           confidence, verdict) in sorted(seen.items()):
            meta = packet_map[pid]
            record = {"packetID": pid, "judge": judge, "winner": winner,
                      **{k: meta[k] for k in ("concept", "kind", "layer",
                                              "alpha", "item",
                                              "baselineIsA")}}
            if judge_model:
                # The RESOLVED Anthropic model the Mac judged with —
                # provenance, so a defaulted model is a recorded fact.
                record["judgeModel"] = str(judge_model)
            if judge_provider:
                # The VERIFIED serving provider (openrouter judges) — the
                # per-judgment stamp completion just checked against the
                # emission pin.
                record["judgeProvider"] = str(judge_provider)
            if confidence is not None:
                record["confidence"] = confidence
            if verdict is not None:
                # The judge's FULL verdict (winner-only closure 2026-07-20):
                # the same object the inline path records as "judgment",
                # winner-consistency-verified above. Absent on rows from
                # older judging clients — winner remains the selection input.
                record["judgment"] = verdict
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with open(os.path.join(run_directory, "recommendations.json"), "w",
              encoding="utf-8") as handle:
        json.dump(recommendations, handle, indent=2, sort_keys=True)
    # The marker lands BEFORE the manifest appends and CARRIES the full
    # projection payload: the judgment run is canonical, and the manifest
    # conditions are a recoverable projection of it — a crash mid-append
    # heals by re-POSTing completion (idempotent path above).
    def _artifact_sha(filename: str) -> str:
        with open(os.path.join(run_directory, filename), "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    marker = {"schema": JUDGMENT_MARKER_SCHEMA,
              "experiment": name,
              "sweepRun": sweep_run, "packetsSha256": digest,
              "experimentHashAtJudgment": live_hash,
              "judgmentsSha256": _artifact_sha("judgments.jsonl"),
              "recommendationsSha256": _artifact_sha("recommendations.json")}
    marker_path = os.path.join(run_directory, "judgment-source.json")
    tmp_marker = marker_path + ".tmp"
    with open(tmp_marker, "w", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)
    os.replace(tmp_marker, marker_path)  # atomic: canonical or absent
    _project_judgment_conditions(name, manifest, run_directory, root, _log)
    _log(f"sweep judgment completed → {run_directory}")
    return run_directory


def _sweep_progress_path(run_directory: str) -> str:
    return os.path.join(run_directory, "sweep-progress.jsonl")


#: The sweep's qualitative record: one JSON line per dev-prompt generation
#: ({kind, concept, layer, alpha, promptIndex, text}), appended durably as
#: each text is generated. Before this file existed the only prose evidence a
#: sweep left behind was the 160-char log previews — an entire dose ladder's
#: generations were unreadable after the fact. Swift twin:
#: ``SweepRunCatalog.devGenerationsFile``.
DEV_GENERATIONS_FILE = "dev-generations.jsonl"

#: Per-record bound on the persisted text. Dev generations are short by
#: construction (the sweep spec's maxTokens, default 80), so this is a
#: safety rail against a decohered cell looping forever, not a working
#: limit; a capped record carries ``truncated: true``.
DEV_GENERATION_TEXT_LIMIT = 20_000


def _dev_generations_path(run_directory: str) -> str:
    return os.path.join(run_directory, DEV_GENERATIONS_FILE)


def _dev_generation_key(record: dict) -> tuple:
    return (record.get("kind"), record.get("concept"),
            int(record["layer"]), float(record["alpha"]),
            int(record["promptIndex"]))


def _load_dev_generation_keys(run_directory: str) -> set:
    """Keys already durable in ``dev-generations.jsonl`` — a resumed sweep
    regenerates some texts it already recorded (a judgeScore resume even
    regenerates the baseline), and the record must not duplicate them.
    Malformed lines (a kill mid-write) are skipped, never fatal: this file
    is a prose record, not a ledger anything resumes from."""
    keys: set = set()
    try:
        with open(_dev_generations_path(run_directory),
                  encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    keys.add(_dev_generation_key(entry))
                except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                    continue
    except OSError:
        pass
    return keys


def _append_dev_generation(run_directory: str, *, kind: str, concept,
                           layer: int, alpha: float, prompt_index: int,
                           text: str, seen: set | None = None) -> None:
    """Durably append one dev generation (flush + fsync, like the progress
    journal): the texts ARE the sweep's qualitative evidence, and a walltime
    kill must not reduce a dose ladder's prose record to log previews."""
    record = {"kind": kind, "concept": concept, "layer": int(layer),
              "alpha": float(alpha), "promptIndex": int(prompt_index),
              "text": text}
    if len(text) > DEV_GENERATION_TEXT_LIMIT:
        record["text"] = text[:DEV_GENERATION_TEXT_LIMIT]
        record["truncated"] = True
    if seen is not None:
        key = _dev_generation_key(record)
        if key in seen:
            return
        seen.add(key)
    with open(_dev_generations_path(run_directory), "a",
              encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_sweep_progress(run_directory: str) -> tuple[list[dict], dict]:
    """(completed grid rows, completed per-concept recommendation blocks)
    from a checkpointed sweep's durable progress log. Torn trailing lines
    (a kill mid-write) are dropped — every complete line was flushed before
    the work it records was counted."""
    rows: list[dict] = []
    recommendations: dict = {}
    try:
        with open(_sweep_progress_path(run_directory), encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    break  # torn tail — everything after is unaccounted work
                if entry.get("kind") == "row":
                    rows.append(entry["row"])
                elif entry.get("kind") == "recommendation":
                    recommendations[entry["concept"]] = entry["block"]
    except OSError:
        pass
    return rows, recommendations


def _sweep_with_spec(name, manifest, model, root, spec, criterion, objective,
                     model_provider, should_cancel, _log, *,
                     judge_stack=None, checkpoint=None,
                     resume_directory: str | None = None,
                     on_run_directory=None) -> str:
    """Manifest-spec'd sweep (Swift parity): per concept, a baseline cell over
    the dev prompts (marker density, distinct-2, battery accuracy), the
    layer×alpha grid in RESIDUAL-NORM units, then the data-declared selection
    criterion; the winner is appended to a draft manifest as
    ``<concept>-recommended`` with a ``selection`` provenance block, and
    ``recommendations.json`` carries the same provenance (or the reason no
    recommendation was made). Greedy single-sample throughout.

    The OBJECTIVE is data too: markerDensity reads the dev texts' marker
    rubric; logprobShift reads the answer-token instrument on the criterion's
    pinned choice rows; judgeScore reads paired-judge preference of each
    cell's dev texts against the baseline's (reusing the texts generated for
    the constraints — never generating twice). Whatever the objective, the
    capability/coherence constraints are computed from the generated dev
    texts and are never bypassed.

    Cancellation is observed between individual generations (dev prompts,
    battery items, choice rows, judge calls), not only at concept/layer
    boundaries, so cancel latency is at most one generation; the outcome is
    the same partial CSV + recommendations a boundary cancel writes. Every
    dev generation logs a one-line preview so decoherence is visible live."""
    from . import battery as battery_mod, experiment_store, sweep_selection as sel
    from .manifest import Condition, Slot

    # Fallback grid alphas are gentle by design (residual-norm units): hot
    # defaults decohere long generations before the researcher sees a single
    # dev text. See DEFAULT_SWEEP_* for the recalibration rationale; an
    # explicit grid in the spec always wins.
    layer_fractions = [float(f) for f in (spec.get("layerFractions")
                                          or DEFAULT_SWEEP_LAYER_FRACTIONS)]
    alphas = [float(a) for a in (spec.get("alphas") or DEFAULT_SWEEP_ALPHAS)]
    dev_prompts_file = spec.get("devPromptsFile") or "prompts/dev/dev-prompts.jsonl"
    battery_file = spec.get("batteryFile") or battery_mod.DEFAULT_BATTERY_FILE
    max_tokens = int(spec.get("maxTokens") or 80)

    dev = load_texts(paths.resolve(dev_prompts_file, root))
    if not dev.texts:
        raise RuntimeError(f"dev prompts '{dev_prompts_file}' has no rows")
    battery_spec = battery_mod.load_spec(battery_file, root)
    battery_items, battery_hash = battery_spec.items, battery_spec.digest
    # The sweep's capability constraint is armed by the battery's FORMAT, not
    # by the study's prompt config (2026-08-13 repair). A legacy battery is
    # still armed from the manifest — which is exactly how a cell can win by
    # writing in the study's format rather than by keeping capability — so it
    # says so out loud.
    battery_arming = battery_mod.resolve_arming(
        battery_spec, prompt_mode=manifest.prompt_mode,
        system_prompt=manifest.system_prompt,
        qwen_thinking_enabled=manifest.qwen_thinking_enabled)
    _battery_advisory = battery_mod.contamination_advisory(
        battery_spec, battery_arming)
    if _battery_advisory:
        _log(f"WARNING: {_battery_advisory}")
    # C4: say what the declared tolerance can actually gate on, BEFORE the
    # grid runs. Battery accuracy moves in steps of 1/N, so a tolerance
    # between steps is not the tolerance that operates — and a sweep that
    # reports "tolerance 0.15" while gating at 0.2 is reporting a number that
    # did not decide anything.
    _resolution = sel.battery_resolution(
        len(battery_items), criterion.capability_tolerance)
    if _resolution is not None:
        _log(("⚠︎ " if _resolution.is_coarse else "") + _resolution.summary)
    # Coherence-length guard (c18 lesson): the distinct-2 floor is only as
    # good as the generation length it is measured at. Advisory, before the
    # grid spends anything; the length is stamped in selection provenance.
    _coherence_advisory = sel.coherence_length_advisory(
        max_tokens, manifest.max_tokens,
        declared=bool(spec.get("maxTokens")))
    if _coherence_advisory:
        _log(f"WARNING: {_coherence_advisory}")
    # Sweep-input pin enforcement at the moment of use (firewall closure
    # 2026-07-20, Swift twin: ExperimentTasks.sweep): a pinned dev-prompts /
    # battery hash must match the bytes the sweep is ABOUT to select on.
    # The refusal is what keeps the ex-post provenance stamp
    # (selection.devPromptsHash = dev.hash) and the manifest pin in
    # agreement — a mismatch refuses, never silently overwrites either.
    from .manifest import sweep_choice_pin_entries
    # Choice-instrument pins are enforced when the objective READS them —
    # a markerDensity/judgeScore sweep leaves a declared choice file inert,
    # so its pin has nothing live to disagree with.
    choice_pin_checks = []
    if objective is not None and objective.metric == "logprobShift":
        for concept, rel, pinned, label in sweep_choice_pin_entries(spec):
            if pinned:
                live = (objective.choice_set_for(concept).hash
                        if concept is not None
                        else (objective.choice_prompts_hash or ""))
                choice_pin_checks.append((live, pinned, rel, label))
    for live_hash, pinned, rel, label in (
            (dev.hash, spec.get("devPromptsHash"), dev_prompts_file,
             "sweep dev prompts"),
            (battery_hash, spec.get("batteryHash"), battery_file,
             "sweep capability battery"),
            *choice_pin_checks):
        if pinned and live_hash != pinned:
            # WP0 step 8: typed `sweepInputDrift` (same prose, same code).
            raise lifecycle_gates.refusing(
                lifecycle_gates.SWEEP_INPUT_DRIFT,
                f"{label} '{rel}' do not match the manifest's pinned hash "
                f"(have {live_hash[:12]}…, pinned {str(pinned)[:12]}…) — the "
                "sweep would select on data the study did not pin; restore "
                "the pinned file, or duplicate the study and re-declare the "
                "sweep",
                repair=(f"restore {rel} to its pinned bytes ; then "
                        "steerlab-server experiment sweep <name>  (a frozen "
                        "pin is never re-pinned: duplicate the study on the "
                        "Mac to change it)"))

    # Objective instruments arm BEFORE any grid work: the choice baseline
    # pass (and its option-length guard) and the judge panel resolve here,
    # so an instrument problem aborts the sweep at start, never mid-grid.
    # A cancel during arming lets TaskCancelled propagate: no run directory
    # exists yet, so aborting the whole sweep IS the consistent outcome (the
    # job runner stamps the cancel).
    # Per-concept instruments (choicePromptsFiles, 2026-08-02): each
    # concept's cells are scored on ITS OWN rows, so the baseline pass runs
    # once per DISTINCT file (keyed by content hash — the singular
    # declaration therefore keeps its single shared baseline pass).
    choice_baseline_by_hash: dict[str, dict[str, float]] = {}
    if objective.metric == "logprobShift":
        for concept_ref in manifest.concepts:
            chosen = objective.choice_set_for(concept_ref.name)
            if chosen.hash in choice_baseline_by_hash:
                continue
            _log(f"choice baseline ('{chosen.file}', "
                 f"{len(chosen.rows)} rows)…")
            choice_baseline_by_hash[chosen.hash] = _choice_target_logprobs(
                model, manifest, chosen.rows, [],
                should_cancel=should_cancel, log=_log)
    judge_rubric, judge_panel = "", []
    judge_rubric_hash = None
    deferred_judging = (objective.metric == "judgeScore"
                        and objective.defer_judging)
    if deferred_judging and criterion.control_apply_to == "topK":
        raise RuntimeError(
            "topK control selection is not supported with deferred judging — "
            "the completion verb applies the winner-only control; use "
            "controls.applyTo 'winner', or push a judge key so judging runs "
            "inline")
    if objective.metric == "judgeScore" and not deferred_judging:
        judge_rubric, judge_panel = _sweep_judge_panel(
            manifest, model, model_provider, root, _log,
            judge_stack=judge_stack)
    elif deferred_judging:
        # Two-phase Claude-judged sweep (key-custody design 2026-07-18):
        # this server has no Anthropic credential BY POLICY. The sweep still
        # resolves the pinned rubric — its TEXT rides in the judging packets
        # to the Mac verbatim, its hash pins them.
        judge_rubric, judge_rubric_hash, _rubric_file = _resolve_rubric(
            manifest, root, _log)
        from . import response_coding
        response_coding.refuse_if_coding(
            judge_rubric, context="the sweep's judgeScore objective",
            rubric_file=_rubric_file)
        _log("judgeScore panel is claude-only and this server holds no "
             "credential (by design): generating everything and emitting "
             "blinded judging packets — judge them on the Mac, then "
             "complete-judgment computes the selection")

    bundles = _extract_all(model, manifest, root)
    resumed_rows: list[dict] = []
    resumed_recommendations: dict = {}
    if resume_directory is not None:
        # Resuming a walltime-checkpointed sweep (2026-08-03): the same gate
        # every resumable verb passes — refuses complete directories
        # (recommendations.json is the sweep's marker), never-checkpointed
        # directories, and checkpoints from a different verb.
        from . import resume as resume_mod
        resume_mod.require_resumable(resume_directory, verb="sweep")
        # And the manifest must be the SAME manifest — a checkpoint from one
        # epoch must not silently continue under edited data.
        hash_file = os.path.join(resume_directory, "experiment-hash.txt")
        try:
            with open(hash_file, encoding="utf-8") as handle:
                checkpointed_hash = handle.read().strip()
        except OSError:
            checkpointed_hash = ""
        if checkpointed_hash and checkpointed_hash != manifest.content_hash():
            raise RuntimeError(
                f"cannot resume sweep in {resume_directory}: the manifest "
                "changed since the checkpoint (content hash "
                f"{checkpointed_hash[:12]}… → "
                f"{manifest.content_hash()[:12]}…) — re-run the sweep fresh")
        run_directory = resume_directory
        resumed_rows, resumed_recommendations = _load_sweep_progress(
            run_directory)
        _log(f"resuming checkpointed sweep: {len(resumed_rows)} completed "
             f"cell row(s), {len(resumed_recommendations)} concept(s) "
             "already selected")
    else:
        run_directory = paths.make_unique_run_directory(
            f"exp-{name}-sweep", root)
        _write_config_snapshot(manifest, run_directory, "sweep", model=model)
        # Persist the sweep's re-derived vectors as first-class extraction
        # artifacts (same helper as extract/validate — never a parallel
        # writer): the sweep run itself then carries recipe-matching
        # sidecars, so "sweep then promote" needs no separate extract run
        # for promote's artifact matcher to find.
        _persist_vectors(bundles, manifest, model, run_directory)
    if on_run_directory is not None:
        on_run_directory(run_directory)
    # Dev generations already recorded (resume dedupe) — fresh runs start
    # empty, a resumed directory seeds from its own durable record.
    dev_generation_keys = _load_dev_generation_keys(run_directory)

    def _append_progress(entry: dict) -> None:
        """Durably append one progress line (flush + fsync): a checkpoint
        may only count work whose record is already on disk."""
        with open(_sweep_progress_path(run_directory), "a",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _checkpoint_if_requested(where: str) -> None:
        """Exit-85 checkpoint between units of work (deferred-judging sweeps
        excluded: their packets live in memory until the end, so parking
        mid-grid would lose them — they run to completion or fail)."""
        if checkpoint is None or not getattr(checkpoint, "requested", False):
            return
        if deferred_judging:
            return
        from . import resume as resume_mod
        completed = len(rows)
        resume_mod.write_state(
            run_directory, run_id=os.path.basename(run_directory),
            verb="sweep", completed_records=completed, reason="signal")
        _log(f"sweep checkpoint at {where}: {completed} cell row(s) durable "
             f"→ exit {resume_mod.CHECKPOINT_EXIT_CODE}")
        raise resume_mod.CheckpointRequested(
            run_directory, "sweep", completed)

    def _gen(prompt_text: str, injections, tokens: int) -> str:
        return generate(model, prompt_text, model_id=manifest.model_id,
                        max_tokens=tokens, temperature=0.0, injections=injections,
                        prompt_mode=manifest.prompt_mode,
                        system_prompt=manifest.system_prompt,
                        qwen_thinking_enabled=manifest.qwen_thinking_enabled)

    def _battery_accuracy(injections, label: str) -> float:
        """Battery accuracy under the given injections, armed by the battery
        (format 2) or by the study manifest (legacy). A cancel is observed
        between items (``TaskCancelled``); item texts are NOT previewed —
        volume would drown the dev previews that matter."""
        total = len(battery_items)
        correct = 0
        generate_fn, choice_fn = _battery_backends(
            model, manifest.model_id, injections)
        for i, item in enumerate(battery_items, start=1):
            _cancel_checkpoint(should_cancel, _log,
                               f"{label} battery {i}/{total}")
            if battery_mod.score_item(battery_spec, item, battery_arming,
                                      generate_fn=generate_fn,
                                      choice_fn=choice_fn)["correct"]:
                correct += 1
        return correct / total

    def _dev_texts(injections, label: str, record=None) -> list[str]:
        """Dev-prompt generations under the given injections. A cancel is
        observed between prompts (``TaskCancelled``), every generation logs
        a one-line preview so decoherence is visible live, and ``record``
        — ``(kind, concept, layer, alpha)`` — names the cell each text is
        durably appended to ``dev-generations.jsonl`` under, AS GENERATED,
        so the prose evidence survives a later kill."""
        texts: list[str] = []
        total = len(dev.texts)
        for i, prompt_text in enumerate(dev.texts, start=1):
            _cancel_checkpoint(should_cancel, _log, f"{label} dev {i}/{total}")
            text = _gen(prompt_text, injections, max_tokens)
            _log(f'{label} dev {i}/{total}: "{_preview_line(text)}"')
            if record is not None:
                kind, concept, layer, alpha = record
                _append_dev_generation(
                    run_directory, kind=kind, concept=concept, layer=layer,
                    alpha=alpha, prompt_index=i - 1, text=text,
                    seen=dev_generation_keys)
            texts.append(text)
        return texts

    def _text_stats(texts, rubric) -> tuple[float, float, float]:
        """(mean marker density, mean distinct-2, mean word count) over
        already-generated dev texts."""
        n = len(texts)
        return (sum(rubric.density(t) if rubric else 0.0 for t in texts) / n,
                sum(distinct_bigram_ratio(t) for t in texts) / n,
                sum(word_count(t) for t in texts) / n)

    def _cell_objective(concept_name, condition_tag, injections, density,
                        texts, baseline_texts) -> float:
        """The declared objective's value for one cell — the dev texts are
        the SAME ones the constraints were computed from, and the choice
        rows are the CONCEPT's own instrument."""
        if objective.metric == "logprobShift":
            chosen = objective.choice_set_for(concept_name)
            return _mean_logprob_shift(
                _choice_target_logprobs(model, manifest, chosen.rows,
                                        injections,
                                        should_cancel=should_cancel, log=_log),
                choice_baseline_by_hash[chosen.hash])
        if objective.metric == "judgeScore":
            if deferred_judging:
                return None  # judged on the Mac; complete-judgment selects
            return _judge_preference(judge_panel, judge_rubric, condition_tag,
                                     texts, baseline_texts, prompts=dev.texts,
                                     should_cancel=should_cancel, log=_log)
        return density

    extra_metric = objective.metric != "markerDensity"
    # Column order is now IDENTICAL on both engines (the Swift twin's
    # `SweepRunCatalog.csvHeader`). `distinct2Ratio` is written for every cell
    # whichever coherence rule is in force — it is the number the
    # baseline-relative floor gates on — and `lengthInflated` flags a cell whose
    # mean output ran more than 1.5× the baseline's. The flag is REPORTED, never
    # gated on.
    fieldnames = ["concept", "layer", "alpha", "markerDensity", "distinct2",
                  "distinct2Ratio", "words", "lengthInflated",
                  "batteryAccuracy"]
    if extra_metric:
        fieldnames.append("objective")

    rows: list[dict] = list(resumed_rows)
    recommendations: dict = dict(resumed_recommendations)
    deferred_packets: list[dict] = []
    deferred_map: dict[str, dict] = {}
    deferred_selection: dict = {}
    cancelled = False
    # (baseline texts, baseline battery accuracy) — generated once, shared
    # by every concept (see the baseline comment inside the loop).
    shared_baseline: tuple[list[str], float] | None = None
    for concept_name, bundle in bundles.items():
        if concept_name in recommendations:
            _log(f"{concept_name}: already selected before the checkpoint — "
                 "skipping")
            continue
        # Completed work from a checkpointed run, keyed exactly as the grid
        # iterates (baseline is layer -1, alpha 0).
        resumed_cells = {
            (int(r["layer"]), float(r["alpha"])): r
            for r in resumed_rows if r.get("concept") == concept_name}
        _checkpoint_if_requested(f"concept={concept_name}")
        if _observe_cancel(should_cancel, _log, f"concept={concept_name}"):
            cancelled = True
            break
        rubric = MarkerRubric.from_directory(paths.concept_directory(concept_name, root))
        if rubric is None:
            _log(f"{concept_name}: no markers.json — expression scores will be 0")
        layer_count = bundle.vectors.layer_count
        layers = concept_sweep_layers(
            next(c for c in manifest.concepts if c.name == concept_name),
            bundle.vectors,
            resolve_sweep_layers(layer_count, layer_fractions), _log)
        residual_norms = bundle.residual_norm_per_layer or []
        if not residual_norms:
            raise RuntimeError(
                f"sweep spec alphas are in residual-norm units but concept "
                f"'{concept_name}' has no residual norms — pin a neutral corpus "
                "and re-extract")

        # Baseline cell: no injection (layer -1, alpha 0 — Swift parity).
        # Baseline texts are kept: judgeScore pairs every cell against them.
        # The GENERATIONS are concept-independent (no injection, same dev
        # prompts, same battery) and are generated ONCE for the whole sweep
        # (review 2026-08-02, P2 — a multi-concept sweep regenerated them
        # per concept); only the marker DENSITY is per-concept, rescored
        # from the cached texts with this concept's rubric.
        baseline_prev = resumed_cells.get((-1, 0.0))
        if baseline_prev is not None and objective.metric != "judgeScore":
            # Resume: this concept's baseline row is already durable, and a
            # non-judge objective never reads the baseline TEXTS — reuse the
            # recorded stats instead of regenerating (judgeScore pairs cells
            # against baseline texts, so it regenerates them).
            baseline_texts = []
            baseline_accuracy = float(baseline_prev["batteryAccuracy"])
            baseline_density = float(baseline_prev["markerDensity"])
            baseline_distinct = float(baseline_prev["distinct2"])
            baseline_words = float(baseline_prev["words"])
        else:
            try:
                if shared_baseline is None:
                    shared_baseline = (
                        _dev_texts([], "baseline",
                                   record=("baseline", None, -1, 0.0)),
                        _battery_accuracy([], "baseline"))
            except TaskCancelled:
                cancelled = True
                break
            baseline_texts, baseline_accuracy = shared_baseline
            baseline_density, baseline_distinct, baseline_words = _text_stats(
                baseline_texts, rubric)
        baseline_objective = sel.baseline_metric(objective.metric, baseline_density)
        baseline = sel.BaselineCell(metric=baseline_objective,
                                    distinct2=baseline_distinct,
                                    battery_accuracy=baseline_accuracy)
        if baseline_prev is None:
            # The baseline is its own reference, so its ratio is 1 and it is
            # never length-inflated.
            baseline_row = {"concept": concept_name, "layer": -1, "alpha": 0,
                            "markerDensity": baseline_density,
                            "distinct2": baseline_distinct,
                            "distinct2Ratio": 1.0,
                            "words": baseline_words,
                            "lengthInflated": False,
                            "batteryAccuracy": baseline_accuracy}
            if extra_metric:
                baseline_row["objective"] = baseline_objective
            rows.append(baseline_row)
            _append_progress({"kind": "row", "row": baseline_row})
        _log(f"{concept_name} baseline: density {baseline_density:.4g}, "
             f"distinct2 {baseline_distinct:.4g}, battery {baseline_accuracy:.4g}")

        cells: list[sel.SweepCell] = []
        for layer in layers:
            if _observe_cancel(should_cancel, _log,
                               f"concept={concept_name} layer={layer}"):
                cancelled = True
                break
            vector = bundle.vectors.per_layer[layer]
            vector_norm = vm.l2_norm(vector)
            # Same rule as the condition and variant paths (2026-08-28 audit,
            # F7/F13): a layer the denominator table does not reach refuses,
            # where this site used to clamp to the last entry and dose the
            # deepest sweep cells with a shallower layer's number.
            residual = _residual_norm_at(
                residual_norms, layer, artifact=concept_name,
                where=f"concept '{concept_name}'")
            for alpha in alphas:
                done = resumed_cells.get((int(layer), float(alpha)))
                if done is not None:
                    # Completed before the checkpoint: rebuild the selection
                    # cell from the durable row — no regeneration.
                    cells.append(sel.SweepCell(
                        layer=layer, alpha=alpha,
                        metric=(float(done["objective"]) if extra_metric
                                else float(done["markerDensity"])),
                        distinct2=float(done["distinct2"]),
                        battery_accuracy=float(done["batteryAccuracy"]),
                        # A journal row from before the words column leaves
                        # the length unrecorded — the winner's lengthInflated
                        # stamp is then absent rather than invented.
                        words=(float(done["words"])
                               if done.get("words") not in (None, "")
                               else None)))
                    continue
                _checkpoint_if_requested(
                    f"concept={concept_name} L{layer} α{alpha:g}")
                if _observe_cancel(should_cancel, _log,
                                   f"concept={concept_name} layer={layer} "
                                   f"alpha={alpha:g}"):
                    cancelled = True
                    break
                raw_alpha = vm.norm_unit_scale(alpha, residual, vector_norm)
                cell = [CellInjection(layer=layer, vector=vector, alpha=raw_alpha)]
                try:
                    texts = _dev_texts(cell, f"L{layer} α{alpha:g}",
                                       record=("cell", concept_name, layer,
                                               alpha))
                    density, distinct, words = _text_stats(texts, rubric)
                    accuracy = _battery_accuracy(cell, f"L{layer} α{alpha:g}")
                    metric_value = _cell_objective(
                        concept_name,
                        f"sweep:{concept_name}:L{layer}:a{alpha:g}", cell,
                        density, texts, baseline_texts)
                except TaskCancelled:
                    # Mid-cell cancel: the incomplete cell is dropped; rows
                    # for completed cells keep today's partial-CSV behavior.
                    cancelled = True
                    break
                ratio = sel.distinct2_ratio(distinct, baseline_distinct)
                inflated = sel.length_inflated(words, baseline_words)
                row = {"concept": concept_name, "layer": layer,
                       "alpha": alpha, "markerDensity": density,
                       "distinct2": distinct,
                       "distinct2Ratio": "" if ratio is None else ratio,
                       "words": words, "lengthInflated": inflated,
                       "batteryAccuracy": accuracy}
                if extra_metric:
                    row["objective"] = metric_value
                rows.append(row)
                _append_progress({"kind": "row", "row": row})
                cells.append(sel.SweepCell(layer=layer, alpha=alpha,
                                           metric=metric_value, distinct2=distinct,
                                           battery_accuracy=accuracy,
                                           words=words))
                if deferred_judging:
                    _emit_judging_packets(
                        deferred_packets, deferred_map, concept_name, "cell",
                        layer, alpha, dev.texts, texts, baseline_texts,
                        judge_rubric_hash)
                    if criterion.matched_norm_random_margin is not None:
                        # The control belongs to the WINNER — unknown until
                        # the Mac judges — so a deferred sweep with a margin
                        # generates the control for EVERY cell now, while
                        # the GPU is still allocated (bounded, known cost);
                        # completion consumes only the winning cell's.
                        control_condition = Condition(
                            name=f"{concept_name}-recommended",
                            slots=[Slot(concept=concept_name, layer=layer,
                                        alpha=alpha)],
                            band_width=1, alpha_in_norm_units=True,
                            control_type="randomMatchedNorm")
                        control_injections = _condition_injections(
                            control_condition, bundles)
                        try:
                            control_texts = _dev_texts(
                                control_injections,
                                f"control L{layer} α{alpha:g}",
                                record=("control", concept_name, layer,
                                        alpha))
                        except TaskCancelled:
                            cancelled = True
                            break
                        _emit_judging_packets(
                            deferred_packets, deferred_map, concept_name,
                            "control", layer, alpha, dev.texts,
                            control_texts, baseline_texts, judge_rubric_hash)
                _log(f"{concept_name} L{layer} α{alpha:g}: density {density:.4g}, "
                     f"distinct2 {distinct:.4g}"
                     + ("" if ratio is None else f" ({ratio:.4g}× baseline)")
                     + f", battery {accuracy:.4g}"
                     + (f", ⚠︎ output {words:.4g} words vs baseline "
                        f"{baseline_words:.4g}" if inflated else "")
                     + (f", {objective.metric} {metric_value:.4g}"
                        if extra_metric and metric_value is not None else ""))
            if cancelled:
                break
        if cancelled:
            break  # partial grid — never select from incomplete evidence

        if deferred_judging:
            deferred_selection[concept_name] = {
                "baseline": {"markerDensity": baseline_density,
                             "distinct2": baseline_distinct,
                             "words": baseline_words,
                             "batteryAccuracy": baseline_accuracy},
                "cells": [{"layer": c.layer, "alpha": c.alpha,
                           "distinct2": c.distinct2,
                           "words": c.words,
                           "batteryAccuracy": c.battery_accuracy}
                          for c in cells],
            }
            recommendations[concept_name] = (
                "awaiting judgment — blinded packets emitted; judge them on "
                "the Mac, then complete-judgment computes the selection")
            continue

        best = sel.select_cell(cells, baseline, criterion)
        if best is None:
            # Say WHICH gate refused. "Capability/coherence" is one of two
            # possible reasons and often the wrong one — a grid whose cells
            # are all eligible but none of which beats the baseline objective
            # is a different result entirely, and reporting it as a gate
            # failure sends the researcher to loosen a tolerance that was
            # never binding.
            recommendations[concept_name] = sel.no_selection_reason(
                cells, baseline, criterion)
            continue

        control_info = None
        controls_evaluated: list[dict] = []
        if criterion.matched_norm_random_margin is not None:
            # Control cells: a deterministic random direction norm-matched
            # to the concept vector at the candidate's layer, same alpha,
            # same dev prompts. Built through _condition_injections with the
            # SAME condition shape a `controlType: randomMatchedNorm` study
            # cell uses, so seeding/norm-matching are identical and
            # reproducible. The control evaluates the SAME declared
            # objective as the grid.
            #
            # applyTo "winner" (historical): the argmax cell alone. "topK"
            # (2026-08-03): walk the top K promotable cells and promote the
            # FIRST that beats its own control — one disruption-artifact
            # corner can no longer veto a grid containing a legitimate
            # winner (observed live in the first stances sweep).
            _checkpoint_if_requested(f"concept={concept_name} controls")
            margin = criterion.matched_norm_random_margin
            candidates = (
                sel.ranked_candidates(cells, baseline, criterion,
                                      criterion.control_top_k)
                if criterion.control_apply_to == "topK" else [best])
            promoted = None
            for candidate in candidates:
                control_condition = Condition(
                    name=f"{concept_name}-recommended",
                    slots=[Slot(concept=concept_name, layer=candidate.layer,
                                alpha=candidate.alpha)],
                    band_width=1, alpha_in_norm_units=True,
                    control_type="randomMatchedNorm")
                control_injections = _condition_injections(
                    control_condition, bundles)
                try:
                    if objective.metric == "logprobShift":
                        chosen = objective.choice_set_for(concept_name)
                        control_metric = _mean_logprob_shift(
                            _choice_target_logprobs(
                                model, manifest, chosen.rows,
                                control_injections,
                                should_cancel=should_cancel, log=_log),
                            choice_baseline_by_hash[chosen.hash])
                    else:
                        control_texts = _dev_texts(
                            control_injections,
                            f"control L{candidate.layer} α{candidate.alpha:g}",
                            record=("control", concept_name, candidate.layer,
                                    candidate.alpha))
                        control_density, _, _ = _text_stats(
                            control_texts, rubric)
                        control_metric = control_density
                        if objective.metric == "judgeScore":
                            control_metric = _judge_preference(
                                judge_panel, judge_rubric,
                                f"sweep-control:{concept_name}:"
                                f"L{candidate.layer}:a{candidate.alpha:g}",
                                control_texts, baseline_texts,
                                prompts=dev.texts,
                                should_cancel=should_cancel, log=_log)
                except TaskCancelled:
                    # A winner without its verified control is incomplete
                    # evidence — no recommendation for this concept.
                    cancelled = True
                    break
                passed = sel.control_passes(
                    candidate.metric, control_metric, margin)
                controls_evaluated.append({
                    "layer": candidate.layer, "alpha": candidate.alpha,
                    "metricValue": candidate.metric,
                    "controlMetricValue": control_metric,
                    "passed": passed})
                _log(f"{concept_name} control L{candidate.layer} "
                     f"α{candidate.alpha:g}: cell {candidate.metric:g} vs "
                     f"control {control_metric:g} → "
                     + ("passes" if passed else "fails"))
                if passed:
                    promoted = candidate
                    control_info = {
                        "type": "randomMatchedNorm",
                        "metricValue": control_metric,
                        "margin": margin,
                        # Recipe stamp (cross-engine contract string);
                        # unstamped = legacy (see RANDOM_VECTOR_ALGORITHM).
                        "randomVectorAlgorithm": RANDOM_VECTOR_ALGORITHM}
                    break
            if cancelled:
                break
            if promoted is None:
                if criterion.control_apply_to == "topK":
                    message = sel.top_k_control_failure_message(
                        controls_evaluated, margin)
                else:
                    message = sel.control_failure_message(
                        best.metric,
                        controls_evaluated[0]["controlMetricValue"], margin)
                recommendations[concept_name] = message
                _append_progress({"kind": "recommendation",
                                  "concept": concept_name, "block": message})
                _log(f"{concept_name}: {message}")
                continue
            # The PROMOTED cell may not be the argmax under topK — every
            # downstream stamp (metrics, winningCell, the minted condition)
            # describes the cell that actually passed its control.
            best = promoted

        if objective.metric == "markerDensity":
            metrics_block = {"markerDensity": best.metric,
                             "baselineDensity": baseline_density}
        else:
            baseline_key = ("baselineJudgeScore"
                            if objective.metric == "judgeScore"
                            else "baselineLogprobShift")
            metrics_block = {objective.metric: best.metric,
                             baseline_key: baseline_objective}
        metrics_block.update({"distinct2": best.distinct2,
                              "batteryAccuracy": best.battery_accuracy,
                              "baselineBatteryAccuracy": baseline_accuracy})
        # The coherence gate's own evidence travels WITH the metrics it
        # adjudicated — previously the ratio and the length flag lived only
        # in sweep.csv, so a promotion certificate inheriting this block
        # could not show what its own floor gated on.
        metrics_block.update(sel.selection_report_metrics(
            best.distinct2, baseline_distinct, best.words, baseline_words))
        selection_block: dict = {
            "sweepRun": os.path.basename(run_directory),
            # Per-concept instruments: the provenance block pins the choice
            # file THIS concept's cells were scored on.
            "criterion": criterion.to_dict(objective, concept=concept_name),
            "devPromptsHash": dev.hash,
            # The length the coherence floor was measured at (c18 lesson):
            # a reader comparing this against the manifest's maxTokens can
            # tell whether the winning cell's distinct-2 is study-relevant
            # evidence or short-generation evidence.
            "devMaxTokens": max_tokens,
            "winningCell": {"layer": best.layer, "alpha": best.alpha},
            "metrics": metrics_block,
        }
        if objective is not None and objective.metric == "judgeScore":
            # Where the judging ran and through which credential is
            # RECORDED provenance, not ambient fact (mirrors the deferred
            # path's "judgedOn": "client"). External judges on this branch
            # judged inline on the server.
            selection_block["judgedOn"] = "server"
            if any(ref.kind != "local" for ref in manifest.judges):
                from . import judge_credentials
                try:
                    credential = judge_credentials.resolve()
                except ValueError:
                    credential = None
                if credential is not None:
                    selection_block["judgeCredential"] = {
                        "kind": credential.kind, "source": credential.source}
        if control_info is not None:
            selection_block["control"] = control_info
        if criterion.control_apply_to == "topK":
            # Which cells were controlled and how each fared — the argmax
            # being rejected by its own control is provenance, not trivia.
            selection_block["controlsEvaluated"] = controls_evaluated
        recommendations[concept_name] = selection_block
        _append_progress({"kind": "recommendation", "concept": concept_name,
                          "block": selection_block})

    csv_path = os.path.join(run_directory, "sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    from . import resume as resume_mod
    if cancelled:
        # A cancelled sweep is an explicit PARTIAL, never "complete"
        # (review 2026-08-03 round 2, P1): recommendations.json is the
        # sweep's completion marker, so writing it here would freeze a
        # partial grid as done. A non-deferred sweep parks exactly like a
        # cancelled study run — the progress journal already holds every
        # completed row and recommendation, so the ordinary resume path
        # finishes the grid (its partial recommendations live only in the
        # journal until then, which is correct: a partial screen must not
        # feed promote's newest-sweep fallback). Deferred sweeps are
        # outside the checkpoint system by design; a cancel leaves them a
        # plain partial to rerun fresh.
        if not deferred_judging:
            resume_mod.write_state(
                run_directory, run_id=os.path.basename(run_directory),
                verb="sweep", completed_records=len(rows), reason="cancel")
        _log(f"sweep cancelled after {len(rows)} cell row(s) — partial, "
             + ("resumable" if not deferred_judging else "rerun it fresh")
             + f" → {run_directory}")
        return run_directory
    if deferred_judging and deferred_packets:
        _write_deferred_judging(
            run_directory, name, manifest, criterion, objective,
            deferred_packets, deferred_map, deferred_selection, dev.hash,
            judge_rubric, judge_rubric_hash, _log)
    # Clear the resume pointer BEFORE projection: projection mutates the
    # manifest, so a crash after it would leave a "resumable" directory the
    # epoch guard must refuse — cleared first, that crash reads as an honest
    # partial instead (fresh rerun; add_condition is idempotent).
    resume_mod.clear_state(run_directory)
    # Draft-condition projection happens ONLY here, after the sweep's whole
    # checkpointable life (review 2026-08-03, P1): projecting per concept
    # mid-sweep mutated the manifest between checkpoints, so a multi-concept
    # draft sweep's own recommendations changed content_hash() and the
    # resume epoch guard refused its checkpoint. Resumed concepts
    # (journal-recovered blocks) project here too, and add_condition
    # replaces by name, so re-projection is idempotent.
    if manifest.status == "draft":
        projected = [
            {"name": f"{rec_concept}-recommended",
             "slots": [{"concept": rec_concept,
                        "layer": block["winningCell"]["layer"],
                        "alpha": block["winningCell"]["alpha"]}],
             "bandWidth": 1, "alphaInNormUnits": True, "selection": block}
            for rec_concept, block in recommendations.items()
            if isinstance(block, dict)
            and isinstance(block.get("winningCell"), dict)
        ]  # failure/awaiting strings project nothing
        if projected:
            # One load, one save — a partial projection cannot exist
            # (review 2026-08-03 round 3, P2).
            experiment_store.add_conditions(name, projected, root)
            for entry in projected:
                _log(f"recommended condition '{entry['name']}' written "
                     "into draft manifest")
    elif any(isinstance(b, dict) for b in recommendations.values()):
        _log(f"manifest is {manifest.status} — recommendations reported only")
    # recommendations.json is the sweep's COMPLETION MARKER
    # (resume.completion_file_for): written atomically LAST, so its presence
    # guarantees every required artifact — csv, deferred packets, draft
    # projections — already exists, and the complete-pointer path can trust
    # it without reconciliation (review 2026-08-03 round 2, P1).
    marker_tmp = os.path.join(run_directory, "recommendations.json.tmp")
    with open(marker_tmp, "w", encoding="utf-8") as handle:
        json.dump(recommendations, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(marker_tmp,
               os.path.join(run_directory, "recommendations.json"))
    _log(f"sweep ({len(rows)} cells) → {run_directory}")
    return run_directory


def _sweep_impl(name, manifest, model, root, layer_fractions, alphas, prompt,
                should_cancel, _log) -> str:
    bundles = _extract_all(model, manifest, root)
    run_directory = paths.make_unique_run_directory(f"exp-{name}-sweep", root)
    _write_config_snapshot(manifest, run_directory, "sweep", model=model)
    sweep_prompt = prompt or (manifest.task_description or "Write a short paragraph.")

    rows = []
    cancelled = False
    for concept_name, bundle in bundles.items():
        if _observe_cancel(should_cancel, _log, f"concept={concept_name}"):
            cancelled = True
            break
        rubric = MarkerRubric.from_directory(paths.concept_directory(concept_name, root))
        layer_count = bundle.vectors.layer_count
        # The legacy grid carries the same zero-injection hazard as the spec'd
        # one, so it asks the same question (see concept_sweep_layers). Its own
        # fraction→layer resolution is kept verbatim (declared order, duplicates
        # and all) so a non-SAE concept sweeps exactly the cells it always did.
        legacy_layers = [min(layer_count - 1, int(layer_count * fraction))
                         for fraction in layer_fractions]
        for layer in concept_sweep_layers(
                next(c for c in manifest.concepts if c.name == concept_name),
                bundle.vectors, legacy_layers, _log):
            for alpha in alphas:
                cell = CellInjection(layer=layer, vector=bundle.vectors.per_layer[layer], alpha=alpha)
                text = generate(model, sweep_prompt, model_id=manifest.model_id,
                                max_tokens=manifest.max_tokens, temperature=0.0,
                                injections=[cell], prompt_mode=manifest.prompt_mode,
                                system_prompt=manifest.system_prompt,
                                qwen_thinking_enabled=manifest.qwen_thinking_enabled)
                # Same qualitative-record rule as the spec'd sweep: the text
                # this row was scored on is evidence, not disposable.
                _append_dev_generation(
                    run_directory, kind="cell", concept=concept_name,
                    layer=layer, alpha=alpha, prompt_index=0, text=text)
                rows.append({
                    "concept": concept_name, "layer": layer, "alpha": alpha,
                    "markerDensity": rubric.density(text) if rubric else "",
                    "distinct2": distinct_bigram_ratio(text), "words": word_count(text),
                })
    csv_path = os.path.join(run_directory, "sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["concept", "layer", "alpha",
                                                    "markerDensity", "distinct2", "words"])
        writer.writeheader()
        writer.writerows(rows)
    _log(f"sweep ({len(rows)} cells{', cancelled early' if cancelled else ''}) → {run_directory}")
    return run_directory


def _load_prompts(manifest: Manifest, prompts_file: str | None, root: str | None) -> list[dict]:
    """Load task prompts with the drift checks the firewall promises.

    The pinned file must match its pinned hash at RUN time, not only at
    freeze; an override (--prompts) on a frozen study must be byte-identical
    to the pin (frozen means frozen — dev iteration belongs on a duplicate);
    and a frozen study cannot run on unpinned prompts at all.
    """
    path = prompts_file or manifest.task_prompts_file
    if not path:
        raise RuntimeError("no task prompts file specified")
    if not os.path.isabs(path):
        path = os.path.join(paths.project_root() if root is None else root, path)
    try:
        with open(path, "rb") as handle:
            live_hash = hashlib.sha256(handle.read()).hexdigest()
    except FileNotFoundError:
        # Swift's twin (`ExperimentTasks.loadTaskPrompts`) has refused this as
        # a typed `missingPrerequisite` since step 7; this engine raised a bare
        # FileNotFoundError, so the same input answered `refused`/65 with a
        # runnable repair on the Mac and a traceback + 1 here.
        raise lifecycle_gates.refusing(
            lifecycle_gates.MISSING_PREREQUISITE,
            missing_task_prompts_refusal(str(path)),
            repair=missing_task_prompts_repair(
                manifest.name,
                prompts_file or manifest.task_prompts_file or "")) from None
    frozen = manifest.status == "frozen"
    if prompts_file is None:
        if manifest.task_prompts_hash and live_hash != manifest.task_prompts_hash:
            raise lifecycle_gates.refusing(
                lifecycle_gates.PIN_DRIFT,
                f"task prompts '{manifest.task_prompts_file}' drifted from the "
                f"pinned hash (have {live_hash[:12]}…, pinned "
                f"{manifest.task_prompts_hash[:12]}…)",
                repair=(f"restore {manifest.task_prompts_file} to its pinned "
                        "bytes ; then re-run this verb (a frozen pin is never "
                        "re-pinned: duplicate the study on the Mac to change "
                        "it)"))
        if frozen and not manifest.task_prompts_hash:
            raise lifecycle_gates.refusing(
                lifecycle_gates.MISSING_PREREQUISITE,
                "frozen study has no pinned task prompts — duplicate, pin a "
                "prompt set, and re-freeze",
                repair=(f"steerlab-cli experiment duplicate {manifest.name} "
                        f"{manifest.name}-v2 && steerlab-cli experiment "
                        f"pin-prompts {manifest.name}-v2 prompts/…/file.jsonl "
                        f"&& steerlab-cli experiment freeze {manifest.name}-v2 "
                        "(authoring is Mac-authority)"))
    elif frozen and live_hash != manifest.task_prompts_hash:
        raise lifecycle_gates.refusing(
            lifecycle_gates.PIN_DRIFT,
            "prompt override on a FROZEN study must match the pinned prompt "
            "set byte-for-byte — duplicate the experiment to iterate",
            repair=(f"steerlab-cli experiment duplicate {manifest.name} "
                    f"{manifest.name}-v2 && steerlab-cli experiment "
                    f"pin-prompts {manifest.name}-v2 <the override file>"))
    prompts = []
    seen_ids: dict[str, int] = {}  # id → 1-based item ordinal
    with open(path, encoding="utf-8") as handle:
        for i, raw in enumerate(handle):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Auto-id parity with Swift `parseTaskPrompts` (2026-07-26).
            # This used to be the 0-based FILE LINE index, blank lines
            # included, while Swift used the 1-based ordinal of PARSED
            # prompts. For a file whose rows carry no explicit `id`, the two
            # engines therefore produced `prompt-0…N-1` and `prompt-1…N`.
            # Paired statistics key on promptID, so the intersection mapped
            # one engine's item k+1 onto the other's item k: it did not fail
            # to join, it joined the WRONG items and dropped one at each end.
            # A blank line anywhere shifted it further.
            # Explicit ids must be non-empty STRINGS; null and absent both
            # take the shared prompt-<ordinal> fallback (review 2026-08-03,
            # P2: `id: null` used to survive as None here while Swift fell
            # back — a divergence in the exact vocabulary the scope pin and
            # paired statistics key on). Message string is the cross-engine
            # contract (Swift twin: parseTaskPrompts).
            raw_id = obj.get("id")
            if raw_id is None:
                raw_id = f"prompt-{len(prompts) + 1}"
            elif not isinstance(raw_id, str) or not raw_id.strip():
                raise RuntimeError(
                    f"task prompts: item {len(prompts) + 1} declares an "
                    "empty or non-string 'id' — declare a non-empty string, "
                    "or omit the key for the prompt-<ordinal> fallback")
            entry = {"id": raw_id,
                     "prompt": obj.get("prompt") or obj.get("text", "")}
            # Duplicate item ids silently corrupt pairing on BOTH engines
            # (choice readouts and paired statistics key on promptID), so the
            # file refuses at LOAD — run/validate/sweep/logprob all inherit
            # the gate. Checked BEFORE the per-item transcript validation
            # (cross-engine ordering contract); the message string is the
            # cross-engine contract (Swift twin:
            # ExperimentTasks.parseTaskPrompts; fixture:
            # prompts/fixtures/task-prompts-validation/cases.json).
            item = len(prompts) + 1
            first = seen_ids.get(entry["id"])
            if first is not None:
                raise RuntimeError(
                    f"task prompts: duplicate item id '{entry['id']}' "
                    f"(items {first} and {item}) — ids must be unique for "
                    "pairing and reporting")
            seen_ids[entry["id"]] = item
            # Scripted transcript (the metacognition-study instrument): a
            # pinned multi-turn conversation — researcher-authored assistant
            # turns included — whose final user turn the model answers.
            # Schema-validated at LOAD on both engines (identical messages);
            # `text`/`prompt` becomes optional (display text derives from the
            # final user turn). Normalized to {role, content} so records are
            # cross-engine identical.
            if "transcript" in obj:
                violation = prompt_render.transcript_schema_violation(
                    obj["transcript"], entry["id"])
                if violation:
                    raise RuntimeError(violation)
                entry["transcript"] = prompt_render.normalize_transcript(
                    obj["transcript"])
                if not entry["prompt"]:
                    entry["prompt"] = prompt_render.transcript_display_text(
                        entry["transcript"])
            # Per-item attention check (the exclusion instrument's first
            # user): {"expected": …, "grading": <battery grading mode>?},
            # graded at ANALYSIS time against the record's output with the
            # capability battery's grading vocabulary. Validated at LOAD with
            # plain-language, cross-engine-identical messages; items without
            # a check are untouched (legacy files load unchanged).
            if "attentionCheck" in obj:
                from . import exclusions
                check_violation = exclusions.attention_check_violation(
                    obj["attentionCheck"], entry["id"])
                if check_violation:
                    raise RuntimeError(check_violation)
                entry["attentionCheck"] = exclusions.normalized_check(
                    obj["attentionCheck"])
            # Science-layer item metadata (all optional, carried into records):
            # answer options + which one the endpoint tracks, the presented
            # anchor and offense severity (Case 3), and the doctrine arm (Case 1).
            for key in ("options", "target", "anchorMonths", "severity", "arm", "caseID"):
                if key in obj:
                    entry[key] = obj[key]
            # What the prompt asks the model to EMIT — decides whether the
            # answer-token instruments can read this item at all. Closed
            # vocabulary, validated at LOAD: an unrecognised value refuses
            # rather than degrading to "unspecified", which would re-open the
            # hole `response_format` closes. Twin of Swift parseTaskPrompts.
            if obj.get("responseFormat") is not None:
                try:
                    entry["responseFormat"] = response_format.parse(
                        obj["responseFormat"])
                except ValueError as exc:
                    raise RuntimeError(
                        f"task prompt '{entry['id']}': {exc}") from exc
            # Factorial-design cell metadata (factor name → level name, the
            # generator's `factors` object): validated as a flat
            # string-to-string map at LOAD (identical message on both
            # engines) and carried into every record the item produces so
            # analysis can stratify by declared factors without rejoining
            # the input file. An empty object is treated as absent.
            if "factors" in obj:
                factors = obj["factors"]
                if (not isinstance(factors, dict) or not all(
                        isinstance(k, str) and isinstance(v, str)
                        for k, v in factors.items())):
                    raise RuntimeError(
                        f"task prompts: item '{entry['id']}' has a "
                        "'factors' value that is not a flat "
                        "string-to-string object — factor names and level "
                        "names must both be strings")
                if factors:
                    entry["factors"] = dict(factors)
            prompts.append(entry)
    return prompts


def _check_response_formats(manifest, prompts: list[dict]) -> None:
    """Run-start gate: the declared option-consuming instruments must be able
    to read the items they will be pointed at, and any declared applicability
    scope must still select the items it was pinned to. Swift twin:
    ``ExperimentTasks.checkResponseFormats``.

    The vocabulary gate comes FIRST and is item-independent: a declaration
    this engine cannot dispatch is a worse failure than one it dispatches at
    the wrong items, because it produces no error at all — the study runs and
    measures nothing. Same gate as the rest of this function:
    ``responseFormat`` is the one gate whose subject is ``outcomeInstruments``
    and whose repair is ``set-instruments``, and its existing family is
    exactly "a declared instrument that would silently produce zero records"
    (the zero-item rules, 2026-08-06)."""
    from . import experiment_store  # local: the module imports tasks back
    unknown = experiment_store.unknown_outcome_instrument_problem(manifest.raw)
    if unknown:
        raise lifecycle_gates.refusing(
            lifecycle_gates.RESPONSE_FORMAT, unknown,
            repair=experiment_store.unknown_outcome_instrument_repair(
                manifest.name))
    items = response_format.items_of(prompts)
    scope = manifest.raw.get("outcomeInstrumentScope")
    drift = response_format.scope_drift_refusal(scope, items)
    if drift:
        # WP0 step 8: typed `responseFormat` (same prose, same exit code).
        raise lifecycle_gates.refusing(
            lifecycle_gates.RESPONSE_FORMAT, drift,
            repair=("steerlab-cli experiment set-instruments <name> "
                    "sampledText, or re-author the items so the pinned scope "
                    "selects them again and re-pin with steerlab-cli "
                    "experiment pin-prompts <name> <file>"))
    refusal = response_format.refusal(
        items, manifest.raw.get("outcomeInstruments"), scope)
    if refusal:
        raise lifecycle_gates.refusing(
            lifecycle_gates.RESPONSE_FORMAT, refusal,
            repair=("steerlab-cli experiment set-instruments <name> "
                    "sampledText, or re-author the items with "
                    '"responseFormat": "label" and re-pin with steerlab-cli '
                    "experiment pin-prompts <name> <file>"))


def _check_transcript_prompts(manifest: Manifest, prompts: list[dict]) -> None:
    """Run-START refusal for scripted-transcript items (never a mid-run
    template error): rawCompletion cannot render a transcript, and every
    transcript must satisfy the study model family's chat-template
    constraints (Gemma's user-first strict alternation). Message strings are
    the cross-engine contract (Swift twin:
    ``ExperimentTasks.checkTranscriptPrompts``)."""
    transcripted = [p for p in prompts if p.get("transcript")]
    if not transcripted:
        return
    if manifest.prompt_mode == prompt_render.RAW_COMPLETION:
        raise RuntimeError(prompt_render.TRANSCRIPT_RAW_COMPLETION_MESSAGE)
    violations = []
    for prompt in transcripted:
        violation = prompt_render.transcript_family_violation(
            prompt["transcript"], prompt["id"], manifest.model_id)
        if violation:
            violations.append(violation)
    if violations:
        raise RuntimeError(
            f"scripted transcripts are incompatible with {manifest.model_id}'s "
            "chat template: " + "; ".join(violations))


# Item metadata copied verbatim onto every record the item produces —
# sampled generations AND deterministic instrument readouts (Swift twin:
# the science-layer fields of GenerationRecord/ChoiceRecord). ``factors``
# (2026-07-20) is the factorial generator's cell metadata: factor name →
# level name, present only on items that declare it.
_PROMPT_META_KEYS = ("target", "anchorMonths", "severity", "arm", "caseID",
                     "factors")

# Declared instrument ids that dispatch the answer-token choice scoring path
# (one deterministic readout per condition × prompt). ``ordinalScale`` rides
# the same machinery — it only adds the ordinal aggregation fields to the
# record. Swift twin: ``ExperimentTasks.choiceInstruments``.
CHOICE_INSTRUMENTS = frozenset(
    {"answerTokenLogprob", "choiceProbability", "ordinalScale"})


def _resolve_ordinal_aggregation(manifest: Manifest) -> str | None:
    """The manifest's declared ``ordinalAggregation`` when the ordinalScale
    instrument is declared, else None. Refuses (RuntimeError) a declared
    ordinalScale with a missing or unknown aggregation — the
    instrument-design choice is declared, never silently defaulted."""
    if "ordinalScale" not in manifest.outcome_instruments:
        return None
    from .logprob import ORDINAL_AGGREGATIONS
    aggregation = manifest.raw.get("ordinalAggregation")
    if aggregation not in ORDINAL_AGGREGATIONS:
        raise RuntimeError(
            "outcomeInstruments includes ordinalScale but ordinalAggregation "
            f"is {aggregation!r} — declare one of "
            + ", ".join(ORDINAL_AGGREGATIONS))
    return aggregation


def derive_seed(experiment_hash: str, condition: str, prompt_id: str,
                sample_index: int) -> int:
    """Deterministic per-record seed for samplesPerItem runs (seedPolicy
    'derivedSHA256'): any (condition, prompt, sampleIndex) cell reproduces its
    exact sample stream on this substrate without a seeds table.

    Policy (asserted by tests, do not change silently): the derivation
    includes CONDITION identity, so baseline and every saved-agent/steered
    condition draw DISTINCT random streams for the same (prompt,
    sampleIndex) — the design is paired at the prompt level, not a
    common-random-numbers design across conditions. Anything that PAIRS
    records across conditions must therefore join on (promptID,
    sampleIndex), never the seed (``paired_judge._pair_generations``)."""
    blob = f"{experiment_hash}|{condition}|{prompt_id}|{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") % (2 ** 63)


def _sha256_text(text: str | None) -> str | None:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@contextmanager
def _seeded_generation(temperature: float, seed: int):
    """Generation-local per-record seeding (RNG isolation, 2026-07-13).

    ``torch.manual_seed`` mutates PROCESS-GLOBAL RNG state, so two concurrent
    seeded studies interleaving records would corrupt each other's sample
    streams. Fork the global RNG state (CPU + every visible CUDA device) for
    the duration of one record's generation and seed INSIDE the fork: the
    record draws exactly the stream its seed names, and the global state is
    restored on exit — interleaved seeded records draw identically to serial
    ones. Greedy records (``temperature <= 0``) never touch the RNG at all
    (preserves the resume byte-equality contract unchanged).
    """
    if not temperature or temperature <= 0:
        yield
        return
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def run(name: str, prompts_file: str | None = None, root: str | None = None,
        dtype: str = "auto", device: str | None = None, *, model_provider=None,
        should_cancel: Callable[[], bool] | None = None, log=None,
        checkpoint: resume_mod.CheckpointFlag | None = None,
        run_directory: str | None = None,
        on_run_directory: Callable[[str], None] | None = None,
        forward_resolutions: dict | None = None,
        shard: "sharding_mod.ShardSpec | None" = None) -> str:
    """Main condition matrix: every prompt × condition × seed, paired by prompt.
    Writes ``generations.jsonl`` + ``report.json`` (parallel to Swift run).

    Reliability hooks (headless paths only; the API passes none of them):
    ``checkpoint`` is polled between records — when a signal set it, the JSONL
    is fsynced, ``resume-state.json`` written, and ``CheckpointRequested``
    raised for the exit-85 path. ``run_directory`` resumes a checkpointed,
    incomplete run (record-level skip; refuses complete directories).
    ``on_run_directory`` is called once with the chosen run directory so the
    submitter can persist its resume pointer before generation starts."""
    _log = log or print
    manifest = Manifest.load(name, root)
    _verify_or_warn(manifest, root)
    # SAE latent conditions (proposal r2 §8 P2-9): validate the declaration and
    # refuse the study kinds whose loop cannot arm one, BEFORE the model loads.
    # The modelOutput path materializes them inside _run_impl.
    _sae_latent_preflight(manifest, _log)

    # API multi-agent runs can use the server's resident model registry. CLI
    # runs keep the older single-loaded-model behavior by falling through.
    if manifest.study_kind == "multiAgent":
        # E1. The old blanket refusal was right about ONE axis and applied to
        # the whole family. Turns within a transcript are ordered and cannot be
        # split across workers — that refusal stands, below. Replicates are
        # independent play-throughs sharing no state, which is exactly the
        # property sharding needs, so the transcript is the shardable unit.
        if shard is not None and manifest.samples_per_item < 2:
            raise RuntimeError(
                "cannot shard a single-transcript panel study: turns within one "
                "transcript are ordered — turn k is conditioned on turns "
                "1..k-1 — so they cannot be split across workers. Replicates "
                "CAN be: raise samplesPerItem and shard across them, e.g. "
                "samplesPerItem 8 with --shard 0/4.")
        # Resume is turn-level, not record-level, but auto-resubmit still
        # hands back the SAME run directory and expects the run to continue
        # in it. Refusing that made requeued panel jobs start somewhere else
        # (or refuse outright) and abandon the partial transcripts the runner
        # had carefully been flushing.
        # Preflight BEFORE dispatching to either path. It used to sit after
        # this branch, so the resident-model/API route — the one a cluster
        # submission actually takes — skipped the check entirely.
        # A panel's turns run on the models its SEATS name, not on the
        # manifest's baseline model. Resolve that here, before anything is
        # acquired: a study whose manifest still said 27B while every seat
        # said 4B spent the load attempting to fetch a 27B nobody would use,
        # and surfaced as a huggingface_hub traceback rather than as the
        # mismatch it was.
        manifest = _panel_load_model(manifest, root, _log)
        # Preflight AFTER resolving the model, not before: it used to run
        # against the manifest's declared default, which no turn uses. On a
        # study whose default was a gated 27B that produced a 401 and a
        # skipped preflight — so the one check that could have sized the run
        # never looked at the 4B that actually ran.
        _scenario_preflight_or_warn(manifest, root, _log)
        _artifact_preflight(manifest, root, _log)
        if model_provider is not None:
            return _run_multi_agent_study(
                name, manifest, None, root, model_provider=model_provider,
                log=_log, shard=shard, run_directory=run_directory,
                on_run_directory=on_run_directory,
                should_cancel=should_cancel, checkpoint=checkpoint)

    # Exact token preflight BEFORE the model load (C1). On a cluster the
    # alternative is the queue wait plus a multi-minute 27B load, followed by
    # a death on whichever oversized item the run reached first — telling you
    # about that one item and none of the others. This is weights-free
    # (AutoTokenizer + AutoConfig), so it costs nothing but a file read.
    #
    # A preflight that cannot RUN never blocks the run: the in-generation
    # ContextBudgetError remains the backstop, and turning a diagnostic into
    # a new way to fail would be a poor trade. Only a successfully computed
    # overflow refuses.
    if manifest.study_kind != "multiAgent":
        # A declared-but-empty agent comparison carrying injection
        # conditions would measure baseline only while every declared arm
        # silently vanished (observed live 2026-08-11: a 4-shard fan-out
        # produced 12 baseline records instead of 96). verify() reports the
        # same violation; drafts only warn there, so the run refuses here —
        # before any queue wait or model load, and identically on every
        # shard of a fan-out.
        from .manifest import (inert_conditions_problem,
                               no_measured_conditions_problem)
        inert_problem = inert_conditions_problem(manifest.raw)
        if inert_problem:
            # WP0 step 8: typed `inertConditions` — the 2026-08-11 incident's
            # rule (a 4-shard fan-out that produced 12 baseline records
            # instead of 96), now machine-readable on both engines.
            raise lifecycle_gates.refusing(
                lifecycle_gates.INERT_CONDITIONS, inert_problem,
                repair=("declare arms the study's studyType actually runs, on "
                        "the Mac: steerlab-cli experiment declare-condition "
                        f"{name} <arm> --slots <concept>:<layer>:<alpha>"))
        # The other road to a silent baseline-only run (WP0 dry run #0,
        # P0-2): a concept study whose arms were never declared at all. Same
        # place, same reason — before any queue wait or model load.
        nothing_to_measure = no_measured_conditions_problem(manifest.raw)
        if nothing_to_measure:
            raise lifecycle_gates.refusing(
                lifecycle_gates.INERT_CONDITIONS, nothing_to_measure,
                repair=("steerlab-cli experiment declare-condition "
                        f"{name} <arm> --slots <concept>:<layer>:<alpha>  "
                        "(authoring is Mac-authority)"))
        _token_preflight_or_warn(manifest, prompts_file, root, _log)
        _artifact_preflight(manifest, root, _log)
        _instrument_preflight(manifest, prompts_file, root, _log)
        _response_format_preflight(manifest, prompts_file, root)

    # THE RESUME/SHARD GATE, BEFORE THE MODEL (open-issues §16 repair 2).
    # Its inputs are three file reads on the run directory, and its refusals
    # are the ones a resumed cluster job hits first — but they used to fire
    # from inside `_run_impl`/`_run_multi_agent_study`, i.e. after staging and
    # loading 51 GiB of weights onto the device. Four GPU allocations died
    # that way on 2026-08-18 at the shard-identity check alone.
    #
    # The epoch comparison is deferred here for an UNPINNED DRAFT only:
    # `_pin_model_revision` (below) can still write the resolved revision into
    # such a manifest and change its content hash, and a bundle execute that
    # re-imported the unpinned manifest depends on that repair. Both inner
    # paths re-run the full gate under the post-pinning manifest, so this
    # never weakens it.
    if run_directory is not None:
        _epoch_stable = bool(manifest.model_revision)
        if manifest.study_kind == "multiAgent":
            _panel_resume_admission(run_directory, manifest=manifest,
                                    shard=shard,
                                    check_experiment_hash=_epoch_stable)
        else:
            _resume_admission(run_directory, name=name, manifest=manifest,
                              shard=shard,
                              check_experiment_hash=_epoch_stable)

    with _acquire_model(manifest, dtype, device, model_provider) as model:
        manifest = _pin_model_revision(name, manifest, model, root, _log)
        # Multi-agent studies branch to the scenario runner (parallel to Swift
        # runMultiAgentStudy) — task prompts and concept generations don't apply.
        if manifest.study_kind == "multiAgent":
            return _run_multi_agent_study(
                name, manifest, model, root, log=_log, shard=shard,
                run_directory=run_directory, on_run_directory=on_run_directory,
                should_cancel=should_cancel, checkpoint=checkpoint)
        return _run_impl(name, manifest, model, root, prompts_file, should_cancel,
                         _log, checkpoint=checkpoint, run_directory=run_directory,
                         on_run_directory=on_run_directory,
                         forward_resolutions=forward_resolutions, shard=shard)


def _token_preflight_or_warn(manifest, prompts_file, root, _log) -> None:
    """Refuse a study whose prompts cannot fit, naming EVERY offending item.

    Never drops or truncates: silently excluding items would change the
    measured sample without recording it. A preflight that cannot run (no
    cached tokenizer, unreadable config) warns and yields to the
    in-generation check rather than blocking the run."""
    from . import token_preflight
    try:
        prompts = _load_prompts(manifest, prompts_file, root)
        report = token_preflight.preflight(
            prompts,
            model_id=manifest.model_id,
            revision=manifest.model_revision,
            prompt_mode=manifest.prompt_mode,
            system_prompt=manifest.system_prompt,
            qwen_thinking_enabled=manifest.qwen_thinking_enabled,
            max_tokens=manifest.max_tokens)
    except token_preflight.PreflightError as exc:
        _log(f"token preflight unavailable ({exc}) — the in-generation "
             "context check remains the backstop")
        return
    except Exception as exc:  # noqa: BLE001 — see the docstring
        _log(f"token preflight unavailable ({type(exc).__name__}: {exc}) — "
             "the in-generation context check remains the backstop")
        return
    refusal = token_preflight.refusal(report)
    if refusal:
        raise RuntimeError(refusal)
    if report.get("promptBudget"):
        worst = max((i["promptTokens"] for i in report["items"]), default=0)
        _log(f"token preflight: {report['itemCount']} prompts, longest "
             f"{worst} tokens, budget {report['promptBudget']} — all fit")


def _instrument_preflight(manifest, prompts_file, root, _log) -> None:
    """Instrument/exclusion coherence BEFORE the model loads (2026-08-06).

    The same gates run again inside ``_run_impl`` (the backstop for callers
    that reach it directly), but on a cluster the difference is the queue
    wait plus a multi-minute 27B load: a declaration that can never fire —
    an option-consuming instrument declared when no in-scope item carries
    ``options`` — silently produced zero records while the sampled arm
    burned the whole GPU allocation. Like ``_artifact_preflight`` this is a
    pure file read, decidable now and unable to change mid-run, so a miss
    REFUSES rather than warns. Also logs the ladder-window advisories: a
    declared outOfRange keep-window whose bounds cannot bind the scale the
    items' options imply (min 0 / max 100 on a 1–7 ladder) is legal but
    inert, and inert-by-declaration is worth a loud line before compute."""
    from . import exclusions as _exclusions
    prompts = _load_prompts(manifest, prompts_file, root)
    _check_response_formats(manifest, prompts)
    _exclusions.preflight(manifest.raw, prompts)
    for warning in _exclusions.ladder_warnings(
            list(manifest.raw.get("exclusionRules") or []), prompts):
        _log(f"warning: {warning}")


def _response_format_preflight(manifest, prompts_file, root) -> None:
    """The response-format/scope-drift gate BEFORE the model loads.

    The identical ``_check_response_formats`` still runs at run start inside
    ``_run_impl`` — this is the same rule moved ahead of the expensive part,
    not a second rule. Field incident 2026-08-06: a 4-shard Slurm run of a
    scope-drifted study staged and loaded gemma-3-27b-it (~2.5 minutes per
    shard, all four wasted) before the run-stage gate refused. The refusal
    was correct; the ordering was not. Like the artifact preflight — and
    unlike the token preflight — nothing here depends on the node's caches:
    the inputs are the manifest and the prompt file, read through the exact
    loader the run uses, so any error raised now (drifted scope, unreadable
    instrument, or a prompt file the loader refuses) is one the run would
    raise after the load. Refusing is strictly earlier, never new."""
    prompts = _load_prompts(manifest, prompts_file, root)
    _check_response_formats(manifest, prompts)


def _panel_load_model(manifest, root, _log):
    """Which model the CLI path should LOAD for a panel, before acquiring it.

    Every turn runs on its seat's own base model; the manifest's ``modelID``
    is the default for seats that name none, and no turn otherwise consults
    it. Loading it regardless is at best wasted and at worst fatal — a
    manifest left at 27B while every seat said 4B spent the load fetching a
    27B nobody would use, and surfaced as a huggingface_hub traceback.

    Returns a manifest whose ``model_id`` is what to load. It NEVER refuses:
    seats that disagree are a legal mixed-model panel, and the API path serves
    them from the registry. On the CLI, where only one model can be resident,
    the majority seat model is loaded and ``run_scenario`` reports the first
    turn that needs another — a capability limit, surfaced where it bites.
    """
    from collections import Counter

    from . import multi_agent
    spath = manifest.multi_agent_scenario_path
    if not spath:
        return manifest
    try:
        if not os.path.isabs(spath):
            spath = os.path.join(paths.project_root() if root is None else root, spath)
        scenario, _ = multi_agent.load_scenario(spath)
    except (OSError, ValueError, KeyError):
        return manifest  # the runner's own load reports this properly
    # Weight by TURNS, not seats: the model most turns need is the one worth
    # holding resident.
    by_turn = Counter()
    seats = {a.id: a.base_model_id for a in scenario.agents}
    for turn in scenario.turns:
        model = seats.get(turn.speaker_agent_id) or manifest.model_id
        if model:
            by_turn[model] += 1
    if not by_turn:
        return manifest
    chosen = by_turn.most_common(1)[0][0]
    if len(by_turn) > 1:
        _log("mixed-model panel: seats name "
             + ", ".join(f"{m} ({n} turn{'s' if n != 1 else ''})"
                         for m, n in by_turn.most_common())
             + f". Loading {chosen}; the server's registry serves the rest, "
               "and each turn records the model it actually ran on.")
    if chosen != manifest.model_id:
        _log(f"panel runs on '{chosen}'; the study's declared default is "
             f"'{manifest.model_id}'. The declared value is kept for "
             "provenance — no turn consults it.")
        import copy
        manifest = copy.copy(manifest)
        manifest.model_id = chosen
    return manifest


def _artifact_preflight(manifest, root, _log) -> None:
    """Refuse BEFORE the model loads when a steering-artifact reference does
    not resolve on this host — naming EVERY dangling reference, not the
    first.

    Unlike the token preflights, a miss here REFUSES rather than warns: the
    check is a file stat through the exact resolution generation uses
    (``model_variant.missing_artifacts`` → ``paths.resolve_artifact``), so a
    reference that does not resolve now cannot resolve mid-run. Observed
    live 2026-08-04: six app-promoted agents carried absolute Mac paths; the
    panel run allocated a GPU, loaded 27B weights to 51 GiB, and died on
    turn 2's `fear.json` — and the agentComparison twin burned its whole
    baseline arm before erroring every variant condition. Seconds versus
    GPU-hours.

    The per-condition error-record machinery in the run loop remains the
    backstop for failures existence cannot predict (unreadable tensors,
    foreign substrate, missing norms)."""
    from . import model_variant, multi_agent
    problems: list[str] = []

    def check(label: str, variant) -> None:
        for miss in model_variant.missing_artifacts(variant, root):
            problems.append(
                f"{label}: {miss['kind']} reference "
                f"'{miss['reference']}' does not resolve")

    for vc in manifest.variant_conditions:
        try:
            variant = (model_variant.ModelVariant.from_dict(vc.artifact)
                       if vc.artifact else
                       model_variant.ModelVariant.from_file(
                           paths.resolve_artifact(vc.artifact_path, root)))
        except (OSError, KeyError, ValueError) as exc:
            problems.append(f"variant condition '{vc.name}': cannot read "
                            f"variant artifact ({exc})")
            continue
        check(f"variant condition '{vc.name}'", variant)

    spath = manifest.multi_agent_scenario_path
    if manifest.study_kind == "multiAgent" and spath:
        scenario = None
        try:
            if not os.path.isabs(spath):
                spath = os.path.join(
                    paths.project_root() if root is None else root, spath)
            scenario, _ = multi_agent.load_scenario(spath)
        except (OSError, ValueError, KeyError):
            pass  # the runner's own load reports this properly
        for agent in (scenario.agents if scenario is not None else []):
            if not agent.variant_artifact_path:
                continue
            try:
                variant = model_variant.ModelVariant.from_file(
                    paths.resolve_artifact(agent.variant_artifact_path))
            except (OSError, KeyError, ValueError) as exc:
                problems.append(
                    f"agent '{agent.name}': cannot read variant artifact "
                    f"'{agent.variant_artifact_path}' ({exc})")
                continue
            check(f"agent '{agent.name}'", variant)

    if problems:
        raise RuntimeError(
            f"artifact preflight: {len(problems)} steering-artifact "
            "reference(s) do not resolve on this host — refusing before the "
            "model loads:\n  - " + "\n  - ".join(problems)
            + "\nA reference recorded on another machine rebases "
            "automatically when the artifact exists under this workspace's "
            "runs/, experiments/, adapters/, or prompts/ — these do not, so "
            "the artifacts themselves are absent (or the references are "
            "wrong). Re-promote/re-import the agents or fix the paths, then "
            "resubmit.")


def _scenario_preflight_or_warn(manifest, root, _log) -> None:
    """Panel twin of ``_token_preflight_or_warn`` (plan A4).

    Refuses only on a turn whose FLOOR — its own template plus shared
    materials, before any deliberation accumulates — already exceeds the
    budget, because that cannot fit whatever the model writes. A worst-case
    projection over the accumulating context warns instead: it charges every
    prior routed turn its full Max tokens, so it is a bound rather than a
    prediction, and blocking on it would refuse runs that fit.

    Same escape hatch as its twin: a preflight that cannot RUN never blocks a
    run — the in-generation context check remains the backstop."""
    from . import multi_agent, scenario_preflight
    spath = manifest.multi_agent_scenario_path
    if not spath:
        return
    try:
        if not os.path.isabs(spath):
            spath = os.path.join(paths.project_root() if root is None else root, spath)
        scenario, _ = multi_agent.load_scenario(spath)
        report = scenario_preflight.preflight(
            scenario, model_id=manifest.model_id, revision=manifest.model_revision)
    except scenario_preflight.token_preflight.PreflightError as exc:
        _log(f"token preflight unavailable ({exc}) — the in-generation "
             "context check remains the backstop")
        return
    except Exception as exc:  # noqa: BLE001 — see the docstring
        _log(f"token preflight unavailable ({type(exc).__name__}: {exc}) — "
             "the in-generation context check remains the backstop")
        return
    refusal = scenario_preflight.refusal(report, scenario)
    if refusal:
        raise RuntimeError(refusal)
    advisory = scenario_preflight.advisory(report, scenario)
    if advisory:
        _log(advisory)
    elif report.get("contextWindow"):
        worst = max((t["projectedPromptTokens"] for t in report["turns"]), default=0)
        _log(f"token preflight: {report['turnCount']} turns, worst-case "
             f"projection {worst} tokens on a {report['contextWindow']}-token "
             "window — all fit")
    _memory_preflight_or_stay_silent(report, manifest, _log)


def _memory_preflight_or_stay_silent(scenario_report, manifest, _log) -> None:
    """Peak-memory advisory beside the token one — MPS only, never blocking.

    The context window was never the binding constraint on a Mac: a panel
    passed token preflight at 14,731/131,072 and died at ~turn 21 on memory.
    Same escape hatch as every preflight here: an estimator that cannot run
    never blocks a run."""
    try:
        from ..steering import model_loader
        from . import memory_preflight

        device = model_loader.resolve_device(None)
        if not device.startswith("mps"):
            return  # CUDA/CPU: see memory_preflight's module docstring
        from transformers import AutoConfig
        kwargs = {"revision": manifest.model_revision} if manifest.model_revision else {}
        config = AutoConfig.from_pretrained(manifest.model_id, **kwargs)
        size = model_loader.snapshot_size_bytes(
            manifest.model_id, manifest.model_revision)
        model = memory_preflight.model_from_config(
            manifest.model_id, config,
            weights_gib=(size / memory_preflight.GIB) if size else None)
        from . import generate as generate_mod
        mem = memory_preflight.report(
            scenario_report, model, device=device,
            budget=memory_preflight.budget_gib(),
            # Which turns will chunk their prefill — without this the
            # estimate assumes single-pass and over-warns on exactly the
            # long turns chunking exists to save.
            prefill_chunk_for=lambda n: generate_mod.prefill_chunk_size(device, n))
        line = memory_preflight.advisory(mem) or memory_preflight.summary(mem)
        if line:
            _log(line)
    except Exception as exc:  # noqa: BLE001 — advisory-only, see docstring
        _log(f"memory preflight unavailable ({type(exc).__name__}: {exc}) — "
             "the run proceeds without a peak-memory estimate")


def _resume_admission(run_directory, *, name, manifest, shard,
                      check_experiment_hash: bool = True) -> None:
    """Admit (or refuse) a supplied run directory for a standard study run.

    Pure file I/O on the run directory — ``require_resumable`` reads the
    directory's completion/park state, the epoch stamp is a JSON read, and the
    shard stamp is another. That is why ``run`` calls it BEFORE acquiring the
    model (open-issues §16): the observed failure order on the cluster was
    stage + 51 GiB weight load + device copy (~4 minutes of a GPU allocation),
    and only THEN the shard-identity ``ResumeError``. Four allocations died
    that way on 2026-08-18.

    ``check_experiment_hash=False`` defers only the epoch comparison — the one
    input that is not stable across the model load, because
    ``_pin_model_revision`` writes a resolved revision into an unpinned DRAFT
    manifest and changes its content hash. ``_run_impl`` runs the full gate
    again afterwards, so nothing is skipped, only ordered.
    """
    resume_mod.require_resumable(run_directory, verb="run")
    if check_experiment_hash:
        stamped = _stamped_experiment_hash(run_directory)
        if stamped is not None and stamped != manifest.content_hash():
            raise resume_mod.ResumeError(
                f"run directory {run_directory} was checkpointed by experiment "
                f"content hash {stamped[:12]}…, but '{name}' now hashes "
                f"{manifest.content_hash()[:12]}… — refusing to mix")
    # Shard identity is part of a partial's resume identity: resuming a
    # shard partial without its exact --shard k/K (or vice versa) would
    # silently generate the wrong record subset into the same file.
    shard_stamp = sharding_mod.read_shard_stamp(run_directory)
    if shard is None and shard_stamp is not None:
        raise resume_mod.ResumeError(
            f"run directory {run_directory} is shard "
            f"{shard_stamp.get('shardIndex')}/{shard_stamp.get('shardCount')} "
            "of a sharded run — resume it with the same --shard k/K")
    if shard is not None:
        if shard_stamp is None:
            raise resume_mod.ResumeError(
                f"run directory {run_directory} is not a shard partial "
                "(no shard.json) — it cannot be resumed under --shard")
        if (int(shard_stamp.get("shardIndex", -1)),
                int(shard_stamp.get("shardCount", -1))) \
                != (shard.index, shard.count):
            raise resume_mod.ResumeError(
                f"run directory {run_directory} was checkpointed as shard "
                f"{shard_stamp.get('shardIndex')}/"
                f"{shard_stamp.get('shardCount')}, not {shard.label} — "
                "refusing to mix shard ranges")


def _panel_resume_admission(run_directory, *, manifest, shard,
                            check_experiment_hash: bool = True) -> None:
    """The panel path's twin of :func:`_resume_admission`.

    Same admissions, same order, different sentences (a panel refusal names
    the directory's basename and speaks of turns) — the wording is the
    contract a field operator reads, so the two stay separate rather than
    being merged into one parameterised message.
    """
    resume_mod.require_resumable(run_directory, verb="run")
    if check_experiment_hash:
        stamped = _stamped_experiment_hash(run_directory)
        if stamped is not None and stamped != manifest.content_hash():
            raise resume_mod.ResumeError(
                f"refusing to resume {os.path.basename(run_directory)}: it was "
                f"produced under experiment hash {stamped[:12]}… but the live "
                f"manifest is {manifest.content_hash()[:12]}… — appending turns "
                "across a manifest epoch would mix two experiments into one "
                "artifact. Duplicate the study to iterate.")
    existing_stamp = sharding_mod.read_shard_stamp(run_directory)
    if shard is None and existing_stamp is not None:
        raise resume_mod.ResumeError(
            f"{os.path.basename(run_directory)} is shard "
            f"{existing_stamp.get('shardIndex')}/"
            f"{existing_stamp.get('shardCount')} of a sharded panel run — "
            "resume it with the same --shard k/K")
    if shard is not None:
        if existing_stamp is None:
            # The case the first guard missed: an unsharded partial
            # resumed under --shard was accepted, restamped as a shard,
            # and "completed" holding only that shard's subset.
            raise resume_mod.ResumeError(
                f"{os.path.basename(run_directory)} is not a shard partial "
                "(no shard.json) — it cannot be resumed under --shard")
        if (int(existing_stamp.get("shardIndex", -1)),
                int(existing_stamp.get("shardCount", -1))) != (shard.index,
                                                               shard.count):
            raise resume_mod.ResumeError(
                f"{os.path.basename(run_directory)} was checkpointed as "
                f"shard {existing_stamp.get('shardIndex')}/"
                f"{existing_stamp.get('shardCount')}, not {shard.label} — "
                "refusing to mix shard ranges")


def _run_impl(name, manifest, model, root, prompts_file, should_cancel, _log,
              checkpoint=None, run_directory=None, on_run_directory=None,
              forward_resolutions=None, shard=None) -> str:
    # Greedy temp 0 ignores seeds — reject the redundant/foot-gun combo (Swift).
    if manifest.temperature == 0 and len(manifest.seeds) > 1:
        raise RuntimeError(
            "temperature 0 is greedy and ignores seeds — use exactly one seed, or "
            "raise the temperature")
    if manifest.samples_per_item > 1 and manifest.temperature <= 0:
        raise RuntimeError(
            "samplesPerItem > 1 requires temperature > 0 — greedy decoding makes "
            "every sample identical")

    # Resume gate BEFORE any model work: a complete directory refuses loudly
    # (immutable runs), and the checkpointed manifest must be the SAME frozen
    # content — appending records derived from drifted pins would silently mix
    # two experiments into one artifact.
    #
    # `run` runs the SAME gate before it acquires the model (§16 repair 2);
    # this call is what keeps a direct `_run_impl` caller admitted identically,
    # and it re-checks the hash under the post-pinning manifest.
    resuming = run_directory is not None
    if resuming:
        _resume_admission(run_directory, name=name, manifest=manifest,
                          shard=shard)

    # Reasoning-style scoring rides on sampled text (no model access): loaded
    # up front so a drifted/broken taxonomy fails BEFORE generation, scored
    # at metrics/report time from the generated text.
    from . import reasoning_style
    style = reasoning_style.load_pinned(manifest, root)

    # Prompts load + transcript gates BEFORE extraction: a schema-invalid or
    # family-incompatible transcript (or rawCompletion+transcript) must fail
    # at run START, not after minutes of vector re-derivation.
    prompts = _load_prompts(manifest, prompts_file, root)
    _check_transcript_prompts(manifest, prompts)
    # Exclusion-rule preflight at run START (same rule as transcripts): a
    # malformed rule declaration, or failedAttentionCheck with no checked
    # items, refuses before any generation compute — the rules are joined at
    # analyze, but a run whose analysis is doomed should not spend GPU time.
    from . import exclusions as _exclusions
    _exclusions.preflight(manifest.raw, prompts)
    # Response-format gate: an answer-token instrument pointed at rows that
    # ask for a JSON object scores the opening brace's position, not the
    # choice. That is a silently WRONG measurement rather than a failure,
    # which is why it refuses rather than warns. Swift twin:
    # ExperimentTasks.checkResponseFormats.
    _check_response_formats(manifest, prompts)
    # Inert concept machinery is inert at RUN time too (2026-07-19): a
    # compare-agents study never re-derives carried concepts' vectors.
    from .manifest import concept_machinery_operative, inert_machinery_note
    machinery = concept_machinery_operative(manifest.raw)
    bundles = _extract_all(model, manifest, root) if machinery else {}
    # LOUD when inert (2026-08-11): a legal agent-comparison run that
    # carries concepts/conditions it will not execute must say so at start
    # and stamp it — a baseline-only result that looks completed and
    # ordinary cost two full GPU rounds before anyone saw why. The
    # no-agent-arms shape refuses earlier (inert_conditions_problem);
    # this is the shape that rightfully proceeds.
    inert_note = inert_machinery_note(manifest.raw)
    if inert_note is not None:
        _log("WARNING: declared studyType "
             f"'{inert_note['declaredStudyType']}' keeps the concept "
             f"machinery INERT — {len(inert_note['inertConditions'])} "
             "carried injection condition(s) "
             f"({', '.join(inert_note['inertConditions']) or 'none'}) and "
             f"{len(inert_note['inertConcepts'])} concept(s) will NOT "
             "execute; this run measures baseline and agent (variant) "
             "conditions only")
    # SAE latent conditions re-derive here, alongside the concept vectors and
    # for the same reason: the manifest pins the RECIPE, the run re-derives the
    # intervention from the pinned source. Done BEFORE the run directory is
    # stamped so the pinned SAE repository commit can ride config.json.
    _sae_latent_preflight(manifest, _log)
    latent_conditions = _materialize_sae_latent_conditions(manifest, _log)
    if run_directory is None:
        # Shard partials are visibly partial by name; the merge assembles the
        # plain exp-<name>-run directory from them.
        slug = (f"exp-{name}-run" if shard is None
                else f"exp-{name}-run-shard{shard.index}of{shard.count}")
        run_directory = paths.make_unique_run_directory(slug, root)
    if on_run_directory is not None:
        on_run_directory(run_directory)
    sampling = _sampling_metadata(model, manifest.temperature)
    if not resuming:
        # A resumed run keeps its original stamps (config.json's createdAt is
        # the run's birth, and vectors re-derive to identical bytes anyway).
        # The inertness note rides config.json's open `notes` dict so a
        # baseline-only run is self-describing without its log.
        #
        # The SAE latent provenance rides the SAME open dict, for the same
        # reason and by the same rule: config.json's top-level key set is
        # CLOSED (run_config.RUN_CONFIG_KEYS — changing it is a cross-engine
        # schema bump), and engine-specific extras belong in `notes`. What is
        # stamped is the pinned SOURCE of each latent arm: repository +
        # resolved commit, SAE config hash, encoder/decoder row hashes,
        # activation family, and the declared mode/beta in latent units. A run
        # that steers on a published dictionary must record which published
        # bytes it read, or the arm cannot be reproduced from its own record.
        run_notes: dict = {}
        if inert_note is not None:
            run_notes["inertConceptMachinery"] = inert_note
        if latent_conditions:
            run_notes["saeLatentConditions"] = [
                provenance for _spec, _edit, provenance in latent_conditions]
        _write_config_snapshot(manifest, run_directory, "run", model=model,
                               notes=run_notes or None)
        _persist_vectors(bundles, manifest, model, run_directory)
        _write_substrate(model, run_directory, sampling)
    # WS7.1: study-run-start cross-substrate check — logged every start
    # (resumes included); the durable advisories.txt is a creation stamp.
    _advise_cross_substrate(manifest, run_directory, root, _log,
                            write_file=not resuming)
    # WP6 R1: same shape, different question — is the stack underneath us the
    # one the committed platform lock pins?
    _advise_dependency_lock_drift(run_directory, _log, write_file=not resuming)

    # Stage 4: forward-referenced conditions resolve HERE — after the run
    # directory exists (the resolution record is run evidence), before any
    # condition executes. An unresolvable reference refuses the whole run.
    _resolve_manifest_forward_refs(manifest, run_directory, root, _log,
                                   ledger_pins=forward_resolutions)

    # The SAME resolver verify() validates and sharding enumerates against —
    # one definition of what a run executes, so the three cannot disagree
    # (external review round 11). Variant comparison runs baseline + variants
    # only (carried injection conditions are inert, Swift runVariantComparison
    # twin); otherwise the concept machinery decides, and a baseline is
    # prepended when absent so paired judging has baseline pairs.
    from .manifest import effective_conditions
    conditions = effective_conditions(manifest)
    cancelled = False
    experiment_hash = manifest.content_hash()
    instruments = set(manifest.outcome_instruments)
    wants_choice = bool(instruments & CHOICE_INSTRUMENTS)
    # Fail fast (before any model compute): an ordinalScale study must have
    # DECLARED a known aggregation — the instrument-design choice is never
    # silently defaulted. verify() reports the same violation; drafts only
    # warn there, so the run refuses here.
    _resolve_ordinal_aggregation(manifest)
    # Declared numeric-answer parser (registry data, not code): resolved once
    # here — a missing/malformed registry or a drifted pin refuses the run at
    # START, never mid-generation. None = the historical caseFamily path.
    numeric_parser = None
    if manifest.numeric_parser:
        from . import parser_registry
        numeric_parser = parser_registry.resolve(manifest.numeric_parser, root)
        if (manifest.parser_registry_hash
                and numeric_parser.registry_hash != manifest.parser_registry_hash):
            raise RuntimeError(
                f"parser registry '{parser_registry.REGISTRY_FILE}' drifted "
                f"from the pinned hash (have "
                f"{numeric_parser.registry_hash[:12]}…, pinned "
                f"{manifest.parser_registry_hash[:12]}…)")
    # …and when nothing was declared, the DEPRECATED caseFamily trigger is what
    # chose this run's numeric endpoint. Said at start, where the rest of the
    # run's configuration is reported, not once per record.
    from .manifest import implicit_case_family_endpoint
    _advise_implicit_case_family(
        numeric_parser is None and implicit_case_family_endpoint(manifest),
        run_directory, _log, write_file=not resuming)
    # E1: one resolver decides what a study does; the run loop, the routing
    # rules and the UI all read the same answer.
    from . import execution_plan
    _plan = execution_plan.resolve(instruments)
    wants_sampled = _plan.generates_sampled_text
    # A declared sampling setting this plan will never read. Advisory, not a
    # refusal: the run is well-defined and its result unaffected, but a
    # temperature that decides nothing is a design mistake worth saying.
    _inert = execution_plan.inert_sampling_advisory(
        instruments, manifest.temperature, manifest.samples_per_item)
    if _inert:
        _log(f"note: {_inert}")
    # RepE reader scoring rides on sampled text: each output is re-read through
    # every pinned reader and stamped as readerScores on the record.
    reader_scorers = _reader_scorers(manifest, root) \
        if ("repeReaderScore" in instruments and manifest.reader_refs) else []
    # Shard plan (multi-GPU fan-out): enumerate the run's FULL expected
    # record-key list in the executor's own emission order and take this
    # shard's contiguous slice. The sample count mirrors _execute_condition;
    # variant conditions are enumerated under the MANIFEST's sampling policy,
    # which _require_manifest_sampling_policy below enforces before any
    # generation. Sharding is execution logistics: nothing here touches the
    # manifest or its content hash.
    plan = None
    if shard is not None:
        # Order IS the contract: this list must match the executor's emission
        # order below (ordinary, then variants, then latent), or the shard's
        # contiguous key slice names records a different worker will emit.
        # Latent conditions are APPENDED so adding one never shifts an existing
        # condition's start position in the key list.
        condition_names = ([c.name for c in conditions]
                           + [vc.name for vc in manifest.variant_conditions]
                           + [spec.name for spec, _e, _p in latent_conditions])
        sample_count = effective_sample_count(manifest)
        all_keys = sharding_mod.expected_record_keys(
            condition_names=condition_names, prompts=prompts,
            wants_choice=wants_choice, wants_sampled=wants_sampled,
            sample_count=sample_count,
            # The SAME scope rule _execute_condition applies at emission —
            # a scope-blind plan expects instrument readouts for items the
            # executor rightly declines, and the merge then refuses a run
            # whose shards all succeeded (2026-08-04).
            instrument_scope=manifest.raw.get("outcomeInstrumentScope"))
        plan = sharding_mod.plan_shard(shard, all_keys=all_keys,
                                       condition_names=condition_names)
        if not resuming:
            sharding_mod.write_shard_stamp(run_directory, plan,
                                           experiment_hash)
        _log(f"shard {shard.label}: records "
             f"[{plan.record_range[0]}, {plan.record_range[1]}) of "
             f"{plan.total_records}; battery/condition ownership: "
             + (", ".join(plan.owned_conditions) or "none"))
    writer = resume_mod.GenerationWriter(
        run_directory, verb="run", checkpoint=checkpoint, resume=resuming,
        log=_log, allowed_keys=(plan.allowed_keys if plan is not None else None))
    if resuming:
        _log(f"resuming run in {run_directory}: {writer.resumed_count} records "
             "already complete — skipping them")
    # J-lens readout, resolved at RUN START. A manifest that DECLARES a readout
    # and silently produces none is the exact failure this instrument exists to
    # prevent, so an unusable declaration refuses here rather than being
    # skipped: before this, the freeze gates pinned a config that nothing ever
    # armed.
    # The WHOLE-STUDY generation count, for the readout budget. Computed from
    # the full matrix, not this shard's slice: a per-shard bound would multiply
    # the effective ceiling by the shard count, which is exactly what the
    # ceiling exists to prevent.
    jlens_expected_generations = (
        (len(conditions) + len(manifest.variant_conditions)
         + len(latent_conditions))
        * len(prompts) * effective_sample_count(manifest))
    jlens_trace = _open_jlens_trace(
        manifest, model, root, run_directory=run_directory,
        checkpoint=checkpoint, resuming=resuming, log=_log,
        allowed_keys=(plan.allowed_keys if plan is not None else None),
        expected_generations=jlens_expected_generations,
        generates_sampled_text=wants_sampled)
    measurement = dict(name=name, manifest=manifest,
                       experiment_hash=experiment_hash,
                       wants_choice=wants_choice, wants_sampled=wants_sampled,
                       reader_scorers=reader_scorers,
                       numeric_parser=numeric_parser,
                       should_cancel=should_cancel, log=_log,
                       jlens_trace=jlens_trace)
    # Study-owned sampling guard (2026-07-21, defense in depth): refuse
    # BEFORE any generation compute if a condition would execute under a
    # sampling policy different from the manifest's declared one.
    _require_manifest_sampling_policy(manifest, model, root,
                                      wants_choice=wants_choice)
    # …and the comparability question the same resolution answers (2026-08-24):
    # are all the arms of this run armed with the SAME effective system
    # content? Advisory, never a refusal, and silent unless they diverge.
    # Computed over the FULL matrix, not this shard's slice: divergence is a
    # property of the design, and every shard should say the same thing.
    _advise_system_prompt_divergence(
        _run_arm_system_prompts(manifest, conditions, latent_conditions, root),
        run_directory, _log, write_file=not resuming)
    try:
        for condition in conditions:
            if plan is not None and not plan.condition_participates(condition.name):
                continue  # another shard owns every record of this condition
            if _observe_cancel(should_cancel, _log, f"condition={condition.name}"):
                cancelled = True
                break
            eff = _effective_ordinary_condition(condition, bundles, manifest)
            if _execute_condition(model, eff, prompts, writer, **measurement):
                cancelled = True
                break

        # Variant conditions run through the SAME executor as baseline and
        # concept conditions (one measurement pipeline — the 2026-07-13
        # unification): the variant resolves to an effective configuration
        # (stored injections + adapter + prompt settings + provenance) and
        # every requested measurement applies identically. Only the model
        # configuration is condition-specific, never the measurements.
        if not cancelled:
            from . import model_variant
            for vc in manifest.variant_conditions:
                if plan is not None and not plan.condition_participates(vc.name):
                    continue  # another shard owns every record of this condition
                if _observe_cancel(should_cancel, _log, f"condition={vc.name}"):
                    cancelled = True
                    break
                try:
                    eff = _effective_variant_condition(
                        vc, manifest, model, root, wants_choice=wants_choice)
                    # Bind verification to the bytes that are ABOUT to load.
                    # The run-start preflight fails fast, but it can run long
                    # before this point — a baseline arm may take hours — and
                    # files are not immutable for the life of a run. Verifying
                    # here means the identity stamped on every row of this
                    # condition describes the adapter that actually shaped it
                    # (external review round 10). Scoped to armed readouts:
                    # that is where the identity becomes a claim.
                    if jlens_trace is not None:
                        eff.verified_identity = _verified_identity_for(
                            eff.variant, root, label=vc.name)
                    # root=root, or verification and loading can resolve the
                    # same relative path in DIFFERENT workspaces: verified
                    # here, loaded from STEERLAB_ROOT/cwd. _adapter_directory
                    # has warned about this since round 3; no call site passed
                    # it (external review round 11).
                    adapter = model_variant.apply_adapter(
                        model, eff.variant, root=root)
                except (OSError, KeyError, ValueError, RuntimeError) as exc:
                    _log(f"variant '{vc.name}' skipped: {exc}")
                    # Under sharding, exactly ONE shard (the condition's
                    # owner) emits the error record — the merge would refuse
                    # duplicated cells, and a single-job run emits it once.
                    if plan is None or plan.owns_condition(vc.name):
                        writer.emit({"experiment": name, "condition": vc.name,
                                     "error": str(exc),
                                     "variantArtifactPath": vc.artifact_path})
                    continue
                try:
                    if _execute_condition(model, eff, prompts, writer,
                                          adapter_active=adapter is not None,
                                          **measurement):
                        cancelled = True
                finally:
                    model_variant.remove_adapter(model, adapter)
                if cancelled:
                    break

        # SAE latent conditions run through the SAME executor as every other
        # condition (proposal r2 §8 P2-9): the mechanism is different, the
        # measurements are identical — prompt-metadata copy, the answer-token
        # instrument, sampled generation, categorical parsing, reader scores,
        # and the same resume/checkpoint semantics. Condition-specific code
        # configures the model; it never redefines how outcomes are measured.
        # Last in the matrix, matching the shard key order built above.
        if not cancelled:
            for spec, edit, provenance in latent_conditions:
                if plan is not None and not plan.condition_participates(spec.name):
                    continue  # another shard owns every record of this condition
                if _observe_cancel(should_cancel, _log, f"condition={spec.name}"):
                    cancelled = True
                    break
                eff = _effective_sae_latent_condition(
                    spec, edit, provenance, manifest)
                if _execute_condition(model, eff, prompts, writer, **measurement):
                    cancelled = True
                    break

        if cancelled:
            # Cooperative cancel parks the run exactly like a checkpoint signal
            # (reason "cancel"): the directory stays resumable and report.json
            # — the completion artifact — is NOT written.
            writer.interrupt(reason="cancel")
    finally:
        writer.close()
        jlens_summary = (jlens_trace.close(expected_records=None)
                         if jlens_trace is not None else None)
        if jlens_summary is not None:
            _log(f"J-lens readout: {jlens_summary['traceRows']} traced "
                 f"generation(s), {jlens_summary['traceObservations']} "
                 f"observation(s), complete={jlens_summary['complete']}"
                 + ("" if jlens_summary["complete"] else
                    f" ({jlens_summary['incompleteRecords']} incomplete — NOT "
                    f"usable as a readout)"))

    records = writer.records
    # Capability battery inside the run (2026-07-13): when the manifest PINS
    # a battery, score it under every condition of the matrix — after the
    # main loop so generations.jsonl bytes are untouched, with its own
    # resume-skippable battery.jsonl stream. Runs only when the main loop
    # completed; a cancel/checkpoint during the battery parks the run
    # resumable exactly like one during generation.
    battery = None
    if (not cancelled and manifest.capability_battery_file
            and manifest.capability_battery_hash):
        # Under sharding, each condition's battery is owned by exactly one
        # shard (the one holding the condition's first record) so batteries
        # run exactly once across the fleet; the merge recombines them.
        battery, battery_cancelled = _run_capability_battery(
            model, name, manifest, bundles, conditions, root, run_directory,
            should_cancel, _log, checkpoint, resuming,
            condition_filter=(set(plan.owned_conditions)
                              if plan is not None else None),
            latent_conditions=latent_conditions)
        if battery_cancelled:
            cancelled = True
            battery = None
    _write_metrics_csv(records, run_directory, style=style)
    _write_summaries_csv(records, run_directory)
    if not cancelled:
        _write_report(name, manifest, records, run_directory, battery=battery,
                      style=style, numeric_parser=numeric_parser)
        resume_mod.clear_state(run_directory)
    _log(f"run ({len(records)} generations"
         f"{f', shard {shard.label}' if shard is not None else ''}"
         f"{', cancelled early — directory is resumable' if cancelled else ''}"
         f"{f', {writer.resumed_count} resumed' if resuming else ''}) "
         f"→ {run_directory}")
    return run_directory


def _stamped_experiment_hash(run_directory: str) -> str | None:
    """The manifest-epoch stamp of a run directory. Delegates to the shared
    :mod:`run_epoch` reader so ``promote``'s epoch guard and this one cannot
    drift apart."""
    return run_epoch.stamped_experiment_hash(run_directory)


def _require_source_epoch(verb: str, name: str, manifest: Manifest,
                          run_dir: str, *, allow_unverified_epoch: bool
                          ) -> tuple[bool, str | None]:
    """Epoch guard for evaluate/analyze source runs (2026-07-13): a source run
    is eligible ONLY if its stamped experiment hash equals the LIVE manifest's
    content hash — otherwise a pre-edit draft run could be judged/analyzed
    under a frozen manifest and stamped with the frozen hash. Legacy runs with
    NO stamp refuse unless ``allow_unverified_epoch`` explicitly accepts them;
    the caller must then stamp ``epochUnverified: true`` into its output.

    Returns ``(unverified, measurement_drift)``. Every caller of this guard
    is a MEASUREMENT verb (evaluate/analyze/rescore-style), so drift confined
    to measurement-side fields (``run_epoch.MEASUREMENT_FIELDS`` — judges,
    evaluation, pipeline) is tolerated rather than refused: those fields
    cannot have affected a byte of the source run's generations, and refusing
    them forced a full GPU re-run to swap a judge whose model had died at its
    provider (2026-08-05). The caller must LOG the returned drift and stamp
    it (``measurementDrift``) into its output — tolerated is never silent.
    The rule itself lives in :mod:`run_epoch`, shared with ``promote`` (which
    stays strict: a judge swap changes what a judged sweep's evidence
    means)."""
    # A source run that is NOT THERE is a path fault, and it gets its own
    # typed refusal with its own repair (ledger 2026-08-21). Before this split
    # the missing directory fell through to "carries no experiment-hash stamp
    # … or pass allowUnverifiedEpoch" — the same sentence a genuinely legacy
    # run gets, on a run that was correctly stamped all along, with a repair
    # that invites the operator to switch the epoch firewall off. The two are
    # different failures and must read as different failures.
    unreadable = run_epoch.unreadable_source_refusal(verb, run_dir)
    if unreadable:
        raise lifecycle_gates.refusing(
            lifecycle_gates.MISSING_PREREQUISITE, unreadable,
            repair=run_epoch.unreadable_source_repair(verb, name))
    refusal, unverified, drift = run_epoch.epoch_refusal(
        verb, name, manifest.content_hash(), run_dir,
        allow_unverified=allow_unverified_epoch, live_manifest=manifest,
        tolerate_measurement_drift=True,
        # The whole family reads the source run's RECORDS, so a run from the
        # other engine is refused here rather than measured into a result
        # this engine's pairing keys cannot have produced (WP0 dry run #2,
        # P0 — found on the Swift side, identical hole here).
        refuse_foreign_substrate=True)
    if refusal:
        # WP0 step 8: typed `manifestEpoch`. Same prose, same exit code; the
        # gate id is what tells an agent this is a REFUSAL against a healthy
        # system rather than the bare `RuntimeError` a genuine defect raises.
        #
        # A foreign run's repair is not "re-run" and is certainly not
        # ``--allow-unverified-epoch`` (which forgives a missing stamp, and
        # would leave this run just as unreadable) — it is the same verb on
        # the engine that wrote the records.
        #
        # THE ASSUMPTION, stated so a third engine cannot inherit it silently
        # (2026-08-18, WP0 residual (d)): there are exactly TWO substrates
        # (CLAUDE.md), so "foreign" means ``swift-mlx``, whose CLI is
        # ``steerlab-cli``; and every verb that reaches this guard —
        # ``analyze``, ``evaluate``, ``rescore-style`` — is TWINNED, i.e. it
        # exists on that CLI under the same spelling. Both halves are pinned
        # by ``test_foreign_substrate_repair_names_an_engine_that_has_the_verb``
        # (the Mac verb roster it checks against is the generated
        # ``swift-*`` region of ``docs/CLI-REFERENCE.md``). This is
        # deliberately a pinned assumption and not engine-capability
        # negotiation: a third substrate must come back here and decide what
        # it can compose, and the test is what will stop it from not
        # noticing.
        if run_epoch.foreign_substrate(run_dir) is not None:
            repair = (f"steerlab-cli experiment {verb} {name}  (on the engine "
                      "that produced the run; this engine reads its results, "
                      "it does not re-measure them)")
        else:
            repair = (f"steerlab-server experiment run {name}  (a run of the "
                      "CURRENT manifest), or re-read the older run with "
                      f"steerlab-server experiment {verb} {name} "
                      "--allow-unverified-epoch")
        raise lifecycle_gates.refusing(
            lifecycle_gates.MANIFEST_EPOCH, refusal, repair=repair)
    return unverified, drift


def _panel_transcript_directory(run_directory: str, condition: str,
                                replicate: int, replicates: int) -> str:
    """Where ONE panel transcript's artifacts live.

    Single-replicate runs keep the historical ``<run>/<condition>/`` layout so
    existing consumers (``panel_effects``, the Runs browser) are untouched;
    replicates nest one level deeper. Shared by the writer loop and the
    run-end completeness check on purpose: a check that derived the layout
    independently could pass while looking in the wrong place.
    """
    return (os.path.join(run_directory, condition) if replicates == 1
            else os.path.join(run_directory, condition,
                              f"replicate-{replicate}"))


#: Artifacts a COMPLETE panel transcript tree carries. ``turns.jsonl`` is the
#: per-transcript record the root ``generations.jsonl`` is flattened from;
#: ``transcript.md`` is the human-readable layer. Both are written by
#: ``multi_agent.run_scenario``; either one missing means the writer did not
#: finish that transcript.
_PANEL_TRANSCRIPT_ARTIFACTS = ("turns.jsonl", "transcript.md")


def panel_transcript_completeness(run_directory: str,
                                  planned: list[tuple[str, int]],
                                  replicates: int) -> list[str]:
    """Which PLANNED transcript trees are not on disk, named one per entry.

    ``planned`` is the (condition, replicate) list this run was responsible
    for — for a shard, only the transcripts that shard owns, so the check
    cannot cry wolf about work another shard did. Returns an empty list when
    every planned tree is present and carries its artifacts.

    Deliberately a DISK check rather than a tally kept by the loop: a tally
    proves the loop believed it wrote something, and the failure this exists
    to catch (2026-08-20 ledger: a complete ``generations.jsonl``, a complete
    ``baseline/`` tree, and an empty ``configured/``) is precisely the case
    where that belief and the filesystem disagree.
    """
    problems: list[str] = []
    for condition, replicate in planned:
        sub = _panel_transcript_directory(run_directory, condition, replicate,
                                          replicates)
        label = (condition if replicates == 1
                 else f"{condition}/replicate-{replicate}")
        if not os.path.isdir(sub):
            problems.append(f"{label}: no transcript directory")
            continue
        absent = [artifact for artifact in _PANEL_TRANSCRIPT_ARTIFACTS
                  if not os.path.isfile(os.path.join(sub, artifact))]
        if absent:
            problems.append(f"{label}: missing " + ", ".join(absent))
    return problems


def _advise_panel_transcripts(run_directory: str, problems: list[str],
                              notes: list[str], _log) -> None:
    """Run-end LOUD, non-blocking transcript-completeness advisory.

    Same shape as the other run-directory advisories
    (:func:`_advise_dependency_lock_drift`): an ``ADVISORY:`` line in the run
    log and an appended line in ``advisories.txt``, and NEVER a change to the
    exit code. ``generations.jsonl`` is the authoritative record and it is
    written before this runs; ``transcript.md``/``turns.jsonl`` are the
    human-readable layer, so their absence is a thing a later reader of the
    run directory must be TOLD about, not a thing that fails a finished run.

    ``notes`` are per-transcript problems the writer already observed while
    running (an artifact write that raised, a transcript that flattened to
    zero records) — carried here with their exception text so the advisory
    says what happened as well as what is missing.
    """
    if not problems and not notes:
        return
    parts = []
    if problems:
        parts.append(f"{len(problems)} transcript tree(s) missing or "
                     "incomplete: " + "; ".join(problems))
    if notes:
        parts.append("writer reported: " + "; ".join(notes))
    advisory = (
        "panel transcripts incomplete — " + ". ".join(parts)
        + ". generations.jsonl is the authoritative record and is complete "
          "for this run; the human-readable transcript layer is what is "
          "missing. Not a refusal.")
    _log(f"ADVISORY: {advisory}")
    # Append, like the lock-drift and case-family advisories: the
    # cross-substrate advisory may already own this file for this run.
    try:
        with open(os.path.join(run_directory, "advisories.txt"), "a",
                  encoding="utf-8") as handle:
            handle.write(advisory + "\n")
    except OSError:  # the advisory must never sink a run
        pass


def _panel_records_from(sub: str, name: str, manifest, model, condition: str,
                        replicate: int) -> list[dict]:
    """Flatten one transcript's turns.jsonl into study generation records.

    Shared by the normal path and the mid-transcript checkpoint handler, which
    must fold an INTERRUPTED transcript's completed turns into the root view
    before parking — otherwise the resume state under-reports what is durably
    on disk.
    """
    out: list[dict] = []
    path = os.path.join(sub, "turns.jsonl")
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                turn = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail; the runner truncates it on resume
            output = turn.get("output", "")
            # Declared-endpoint parse, carried verbatim from the turn record
            # (the runner stamped it at write time; nothing re-parses here —
            # one parse, one place). Absent key when the turn declared no
            # endpoint, so panels without declarations flatten byte for byte
            # as before.
            endpoint = turn.get("endpoint")
            # Voice-lint stamp, likewise carried verbatim (the runner stamped
            # it at write time). Absent on turns written before the lint
            # existed, so old runs flatten byte for byte as before.
            lint = turn.get("voiceLint")
            out.append({
                "experiment": name, "experimentHash": manifest.content_hash(),
                "modelID": turn.get("modelID", manifest.model_id),
                "modelRevision": turn.get(
                    "modelRevision", model.revision if model is not None else None),
                "condition": condition, "seed": turn.get("seed") or 0,
                "promptID": turn.get("turnID"),
                "promptIndex": int(turn.get("turnIndex", 1)) - 1,
                "replicateIndex": replicate,
                # Merge/resume identity (resume.record_key) is the 5-tuple
                # (condition, promptIndex, promptID, sampleIndex, kind). A
                # panel's replicate IS its sample axis — samplesPerItem drives
                # it — so it rides in sampleIndex and the shared contract needs
                # no panel special case. Without this every replicate of a turn
                # collapses to one key and the merge refuses the partials as
                # duplicated cells.
                "sampleIndex": replicate,
                "temperature": turn.get("temperature", manifest.temperature),
                "prompt": turn.get("prompt", ""), "output": output,
                "speakerName": turn.get("speakerName"),
                "turnTitle": turn.get("title"),
                "routedAgentIDs": turn.get("routedAgentIDs"),
                "device": turn.get("device"),
                "wordCount": word_count(output),
                "distinct2": distinct_bigram_ratio(output),
                **({"endpoint": endpoint} if endpoint else {}),
                **({"voiceLint": lint} if lint else {})})
    return out


def _park_panel_run(run_directory: str, records: list[dict], log,
                    *, reason: str) -> None:
    """Land a partial panel run durably: generations.jsonl + resume-state.json.

    The checkpoint contract is that by the time ``CheckpointRequested``
    propagates, everything the requeue needs is already on disk — the caller's
    only remaining job is to exit 85. Per-turn transcripts are already flushed
    by the runner; this adds the root-level view and the resume pointer."""
    with open(os.path.join(run_directory, "generations.jsonl"), "w",
              encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    resume_mod.write_state(
        run_directory, run_id=os.path.basename(run_directory), verb="run",
        completed_records=len(records), reason=reason)
    log(f"checkpoint ({reason}): {len(records)} turn record(s) flushed; "
        f"resume-state written → exit {resume_mod.CHECKPOINT_EXIT_CODE}")


def _run_multi_agent_study(name, manifest, model, root, model_provider=None,
                           log=print, shard=None, run_directory=None,
                           on_run_directory=None, should_cancel=None,
                           checkpoint=None) -> str:
    """Run a multi-agent study's scenario (configured + optional baseline) into
    one run directory (parallel to Swift runMultiAgentStudy)."""
    from . import multi_agent
    spath = manifest.multi_agent_scenario_path
    if not spath:
        raise RuntimeError(f"multi-agent study '{name}' has no pinned scenario")
    if not os.path.isabs(spath):
        spath = os.path.join(paths.project_root() if root is None else root, spath)
    scenario, shash, scenario_bytes = multi_agent.read_scenario(spath)
    # Drift in a pinned input is a violation, never a silent copy: the
    # snapshot below is only evidence if the bytes it preserves are the bytes
    # the study pinned. Checked here, before the run directory exists, so a
    # drifted scenario produces no half-run to explain away.
    pinned_hash = manifest.multi_agent_scenario_hash
    if pinned_hash and pinned_hash != shash:
        raise RuntimeError(
            f"multi-agent scenario '{manifest.multi_agent_scenario_path}' "
            f"changed since pinning (have {shash[:12]}…, pinned "
            f"{pinned_hash[:12]}…) — refusing to run "
            f"'{name}' against an input its manifest does not describe")
    # A panel's seats may legitimately name DIFFERENT base models — mixed-model
    # panels are a design goal (eventually one GPU per seat), not an error. The
    # manifest's modelID is the DEFAULT for seats that name none, never a claim
    # about what ran; the per-turn records carry the truth. So nothing refuses
    # here. Where a second model genuinely cannot be served — the CLI path has
    # one resident model and no registry — run_scenario says so at the turn
    # that needs it, naming the remedy.
    # Reasoning-style scoring is study-kind-agnostic: a pinned taxonomy
    # scores each flattened turn (drift fails up front).
    from . import reasoning_style
    style = reasoning_style.load_pinned(manifest, root)
    resuming = run_directory is not None
    if not resuming:
        run_directory = paths.make_unique_run_directory(f"exp-{name}-run", root)
        _write_config_snapshot(manifest, run_directory, "run", model=model)
    else:
        # Same admission guards the ordinary resume path enforces. Accepting
        # any supplied directory would permit mutating a COMPLETE run, or
        # appending turns derived from a different manifest epoch or a
        # different shard range into someone else's partial. `run` runs this
        # same gate before acquiring the model (§16 repair 2); the call here
        # is what admits a direct caller identically.
        _panel_resume_admission(run_directory, manifest=manifest, shard=shard)
        log(f"resuming panel run in {os.path.basename(run_directory)}")
    # Snapshot the scenario VERBATIM beside experiment.json, before a single
    # turn is generated. experiment.json only POINTS at the scenario, but the
    # seat→variant attribution (agents[].variantArtifactPath/Hash) lives in
    # the scenario itself — so without this a finished run cannot answer
    # "which seat carried which agent variant" without the live workspace
    # file, and readers that see only the run directory (the Results Explorer
    # bridge serves runs/) cannot answer it at all.
    snapshot = os.path.join(run_directory, "scenario.json")
    if not os.path.exists(snapshot):
        with open(snapshot, "wb") as handle:
            handle.write(scenario_bytes)
    # Auto-resubmit needs the directory BEFORE the work starts, so a requeue
    # can hand the same one back.
    if on_run_directory is not None:
        on_run_directory(run_directory)
    # WS7.1: same study-run-start cross-substrate check as the standard path.
    _advise_cross_substrate(manifest, run_directory, root, log, write_file=True)
    _advise_dependency_lock_drift(run_directory, log, write_file=True)
    # The panel-effects decomposition below adds a built-in "months" endpoint
    # on the DEPRECATED caseFamily trigger. `implicit_case_family_endpoint`
    # knows this path reads case_family ALONE — a declared numericParser does
    # not displace it here, as it does on the record-parse path — so the
    # advisory fires whenever the trigger does, which is the whole contract.
    from .manifest import implicit_case_family_endpoint
    _advise_implicit_case_family(
        implicit_case_family_endpoint(manifest), run_directory, log,
        write_file=True)
    # No system-prompt divergence advisory here, deliberately (2026-08-24
    # casting ruling). This path HAS the channel the standard path writes to —
    # advisories.txt, three owners above — but the advisory's unit is the ARM,
    # and a panel's two arms cannot diverge in effective system content: the
    # split is `strip_interventions`, which drops injections and the adapter
    # and never touches a seat's persona or role text
    # (`multi_agent._runtime_settings`). A line that can only ever be silent is
    # noise in the source instead of noise in the log. Seats WITHIN a
    # transcript are armed differently on purpose — that is what casting is —
    # so per-seat divergence is the design, not a finding; the per-turn
    # `systemPromptComposition` stamp is what records it.
    conditions = [("configured", False)]
    if manifest.multi_agent_include_baseline:
        conditions.append(("baseline", True))
    # Also emit a root-level generations.jsonl + report.json (flattening each
    # turn into a study generation) so /evaluate and the Runs browser treat
    # multi-agent output as ordinary study results (parallel to Swift).
    records: list[dict] = []
    cancelled = False
    # The transcripts THIS run is responsible for, appended as the loop admits
    # each one, and the per-transcript problems the writer saw while running.
    # Both feed the run-end completeness advisory: `planned` is what the check
    # looks for on disk (a shard is responsible only for the transcripts it
    # owns), `writer_notes` is what the writer already knows went wrong.
    planned: list[tuple[str, int]] = []
    writer_notes: list[str] = []
    # Replicates: the STUDY manifest owns measured-run sampling policy, so its
    # temperature overrides the scenario's authoring value and samplesPerItem
    # is the replicate count. Each replicate is an independent play-through —
    # that independence is what makes replicates (not turns) the clusterable
    # and shardable unit.
    replicates = max(1, manifest.samples_per_item)
    # E1: shard over TRANSCRIPTS — (condition, replicate) pairs. Every turn of
    # a transcript stays with its transcript, which is what keeps the ordered
    # dependence intact while still parallelising the fan-out.
    owned: set | None = None
    if shard is not None:
        plan = sharding_mod.plan_panel_shard(
            shard, condition_names=[cond for cond, _ in conditions],
            replicates=replicates,
            turn_ids=[turn.id for turn in scenario.turns])
        # Transcript ownership, derived from the record keys the merge will
        # check — the shard splits on transcripts but STAMPS records, so the
        # two views cannot drift apart.
        owned = {(key[0], key[3]) for key in plan.keys}
        sharding_mod.write_shard_stamp(
            run_directory, plan, manifest.content_hash())
        log(f"shard {shard.label}: {len(owned)} transcript(s), "
            f"{len(plan.keys)} record(s)")
    for cond, strip in conditions:
        if cancelled:
            break
        for replicate in range(replicates):
            if owned is not None and (cond, replicate) not in owned:
                continue
            # Cancellation unit is the transcript: stopping mid-transcript
            # would leave a partial that turn-level resume can finish, but a
            # requeue is the right actor for that, not a half-written arm.
            # Walltime checkpoint (SIGUSR1/SIGTERM). Checked at the
            # transcript boundary for the same reason cancellation is: a
            # half-written arm is not a measurement, and turn-level resume can
            # finish the transcript on the next attempt.
            if checkpoint is not None and checkpoint.requested:
                _park_panel_run(run_directory, records, log, reason="signal")
                raise resume_mod.CheckpointRequested(
                    run_directory, "run", len(records), reason="signal")
            if should_cancel is not None and should_cancel():
                log("cancelled before transcript "
                    f"{cond}/replicate-{replicate} — partial run kept for resume")
                cancelled = True
                break
            sub = _panel_transcript_directory(run_directory, cond, replicate,
                                              replicates)
            # Recorded BEFORE the write, so a transcript this run admitted and
            # then failed to write is still something the run-end check looks
            # for. (Appending after a successful write would make the check
            # blind to exactly the skip it exists to catch.)
            planned.append((cond, replicate))
            os.makedirs(sub, exist_ok=True)
            try:
                multi_agent.run_scenario(
                    model, scenario, run_dir=sub, condition_name=cond,
                    strip_interventions=strip, scenario_hash=shash,
                    model_provider=model_provider,
                    default_revision=manifest.model_revision,
                    temperature=manifest.temperature,
                    replicate_index=replicate,
                    experiment_hash=manifest.content_hash(),
                    checkpoint=checkpoint,
                    # Per-transcript summary-artifact failures land here with
                    # their exception text instead of vanishing (or sinking a
                    # run whose turns are already durable); the run-end
                    # advisory says them out loud.
                    artifact_problems=writer_notes,
                    # Forward the task's logger. Without this, run_scenario
                    # fell back to its default no-op and every per-turn line
                    # — progress ✓s, the accelerator-memory probe, turn
                    # warnings — vanished. A 144-turn field run produced
                    # ZERO turn lines while its component test proved the
                    # probe "reaches the run log" by passing its own
                    # callback: component-not-feature, again. The probe
                    # existed precisely to replace guessing during the 75 GB
                    # OOM investigation, and this dropped kwarg is why the
                    # investigation had to guess anyway.
                    log=log)
            except resume_mod.CheckpointRequested:
                # Signalled MID-transcript. The completed turns are already
                # fsynced; fold them into the root view before parking, or the
                # requeue sees a resume state that under-reports what exists.
                records.extend(_panel_records_from(
                    sub, name, manifest, model, cond, replicate))
                _park_panel_run(run_directory, records, log, reason="signal")
                # Re-raise naming the ROOT run directory. run_scenario knows
                # only its transcript sub-directory, and the contract is that
                # CheckpointRequested.run_directory is what the requeue
                # resumes — pointing that at a transcript folder would send
                # the next attempt somewhere it can do nothing useful.
                raise resume_mod.CheckpointRequested(
                    run_directory, "run", len(records), reason="signal") from None
            flattened = _panel_records_from(
                sub, name, manifest, model, cond, replicate)
            if not flattened:
                # `_panel_records_from` returns [] for an absent or unreadable
                # turns.jsonl — the one place a whole transcript can leave the
                # run's record without anything being raised. Say so here
                # rather than letting the arm evaporate silently.
                writer_notes.append(
                    f"{cond}/replicate-{replicate} flattened to zero turn "
                    "records (turns.jsonl absent or unreadable)")
                log(f"WARNING: {writer_notes[-1]}")
            records.extend(flattened)
    with open(os.path.join(run_directory, "generations.jsonl"), "w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r) + "\n")
    if cancelled:
        # Park it as RESUMABLE rather than writing report.json + panel effects
        # over a partial matrix — those imply a completeness this run does not
        # have, and analyze/evaluate would happily read them as measurement.
        # Auto-resubmit finds resume-state.json and hands the same directory
        # back; each transcript then continues from its last completed turn.
        resume_mod.write_state(
            run_directory, run_id=os.path.basename(run_directory), verb="run",
            completed_records=len(records), reason="cancel")
        log(f"panel run interrupted: {len(records)} turn record(s) kept; "
            "directory is resumable")
        return run_directory
    # Run-end completeness of the TRANSCRIPT layer (2026-08-20 ledger): the
    # trees on disk must match the transcripts this run was responsible for.
    # A mismatch is a loud advisory naming exactly which condition/replicates
    # are missing — never an exit code, never a failed run, because
    # generations.jsonl (written above) is the authoritative record and it is
    # complete. Runs only on the COMPLETION path: a cancelled or checkpointed
    # run legitimately has trees it has not written yet, and returned above.
    _advise_panel_transcripts(
        run_directory,
        panel_transcript_completeness(run_directory, planned, replicates),
        writer_notes, log)
    _write_metrics_csv(records, run_directory, style=style)
    _write_report(name, manifest, records, run_directory, style=style)
    # Voice lint, aggregated per (speaker × condition) into its OWN artifact
    # rather than into panel-effects.csv. Two reasons, both structural:
    # panel-effects.csv is one row per ENDPOINT of a paired configured/baseline
    # decomposition with a fixed twelve-column header that both engines assert
    # byte for byte — a speaker×condition rate is a different grain and would
    # have to break that header — and it is written only for single-replicate
    # runs that carry both arms, which is exactly the shape the panel runs that
    # exposed these failures do NOT have (five replicates, so no
    # panel-effects.csv at all). A stamp nobody can read is not a measurement.
    from . import voice_lint as voice_lint_mod
    lint_rows = voice_lint_mod.csv_rows(records)
    if lint_rows:
        voice_lint_mod.write_csv(
            os.path.join(run_directory, voice_lint_mod.VOICE_LINT_FILENAME),
            lint_rows)
        log(f"voice lint: {len(lint_rows)} speaker×condition cell(s) "
            "→ panel-voice-lint.csv")
    # A finished run is not resumable. Leaving the pointer behind violates the
    # artifact contract and invites a later caller to "resume" a complete run.
    resume_mod.clear_state(run_directory)
    # Panel-effect decomposition needs both arms of the pair.
    if manifest.multi_agent_include_baseline and replicates == 1:
        from . import panel_effects
        endpoints: dict = {"wordCount": lambda text: float(word_count(text))}
        if manifest.case_family == "sentencing":
            endpoints["months"] = judicial.parse_months
        rows = panel_effects.write_panel_effects(run_directory, scenario,
                                                 endpoints=endpoints)
        if rows:
            print(f"panel effects: {len(rows)} endpoint(s) → panel-effects.csv")
    elif manifest.multi_agent_include_baseline:
        # No silent caps: say what was skipped and why. The decomposition
        # pairs one configured transcript against one baseline transcript;
        # with replicates there are N of each, and how to pool them IS the
        # unit-of-analysis question (cluster by transcript) that has to be
        # settled before this can report an honest number.
        log(f"panel effects: skipped — {replicates} replicates per condition, and "
            "pooling across replicates needs the clustered estimator "
            "(MULTI-AGENT-SUBSTRATE-PARITY-PLAN D1). Re-run with "
            "samplesPerItem 1 for the paired single-transcript decomposition.")
    print(f"multi-agent run ({len(conditions)} conditions, {replicates} replicate(s), "
          f"{len(records)} turns) → {run_directory}")
    return run_directory


@dataclass
class EffectiveCondition:
    """The resolved execution configuration of ONE study condition.

    Every condition of the run matrix — implicit baseline, steered/control
    concept conditions, and ModelVariant-backed variant conditions — reduces
    to this shape, and ONE shared per-item executor
    (:func:`_execute_condition`) performs the same requested measurements for
    all of them. Condition-specific code (the resolvers below) configures the
    model; it never redefines how outcomes are measured. This closes the
    2026-07-13 measurement-asymmetry finding: variant conditions previously
    ran a separate, poorer pipeline (no prompt-metadata copy, no parsedChoice,
    no answer-token instrument, no intervention/seed-policy metadata), so
    summaries carried blank choice-rate fields for a no-intervention variant
    whose raw outputs were identical to baseline."""

    name: str
    injections: list[CellInjection]
    intervention_state: dict
    prompt_mode: str
    #: The EFFECTIVE system prompt this arm generates under — the composition
    #: of ``agent_system_prompt`` and ``study_system_prompt``
    #: (:func:`system_prompt.compose`), never one of them alone. This is what
    #: reaches the renderer and what ``systemPromptHash`` stamps.
    system_prompt: str | None
    qwen_thinking_enabled: bool
    temperature: float
    #: The two LEVELS the effective prompt was composed from, kept beside it so
    #: every record can stamp which level contributed what
    #: (``systemPromptComposition``) and so the run-start comparability
    #: advisory can name the arms. ``agent_system_prompt`` is None for every
    #: arm that is not agent-backed — which is baseline, every steering
    #: condition, and every SAE latent condition.
    agent_system_prompt: str | None = None
    study_system_prompt: str | None = None
    # Extra provenance keys stamped on EVERY record of this condition
    # (variantArtifactPath/variantArtifactHash/agentPlaygroundTemperature
    # for variant conditions — the last is the artifact's stored Playground
    # temperature, provenance only; "temperature" remains the single source
    # of what governed generation).
    # Empty for ordinary conditions — their record bytes are unchanged.
    provenance: dict = field(default_factory=dict)
    # The loaded ModelVariant for variant conditions (the run loop applies /
    # removes its PEFT adapter around the condition); None otherwise.
    variant: object | None = None
    # TRUE SAE latent edits (steering.sae_latent.SAELatentEdit) for
    # saeLatentConditions arms. A SEPARATE field from `injections`, never
    # folded into it: a latent edit is state-dependent and dosed in latent
    # units, so one list holding both mechanisms would make every downstream
    # len(injections) and every provenance stamp misdescribe what ran. Empty
    # for every other condition type, so their record bytes are unchanged.
    latent_edits: list = field(default_factory=list)
    # The VERIFIED adapter identity of this condition, attached immediately
    # before its adapter is loaded. Carried on the object that actually runs
    # rather than looked up by display name: names are not guaranteed unique,
    # and a name-keyed cache would stamp one agent's rows with another's
    # identity (external review round 10).
    verified_identity: dict | None = None


def _effective_ordinary_condition(condition, bundles, manifest) -> EffectiveCondition:
    """Baseline and concept conditions execute under the manifest's own
    prompt/sampling configuration; injections and intervention provenance
    resolve from the condition's slots exactly as before (matched-norm random
    controls included).

    These arms carry no agent identity, so the composition
    (:func:`system_prompt.compose`) degrades to the study frame itself — the
    same object, byte for byte what these conditions have always rendered and
    stamped."""
    return EffectiveCondition(
        name=condition.name,
        injections=_condition_injections(condition, bundles),
        intervention_state=_intervention_state(condition),
        prompt_mode=manifest.prompt_mode,
        system_prompt=system_prompt_mod.compose(None, manifest.system_prompt),
        qwen_thinking_enabled=manifest.qwen_thinking_enabled,
        temperature=manifest.temperature,
        agent_system_prompt=None,
        study_system_prompt=manifest.system_prompt)


def _variant_intervention_state(vc, variant) -> dict:
    """The variant twin of :func:`_intervention_state`: the SAME cross-engine
    keys (slots/bandWidth/alphaInNormUnits/controlType) so readers parse one
    shape for every condition, plus the variant identity and its non-injection
    components — a reader never reconstructs what a variant condition applied
    from its artifact path."""
    return {
        "slots": [{"concept": inj.get("concept"), "layer": inj.get("layer"),
                   "alpha": inj.get("alpha")} for inj in variant.injections],
        "bandWidth": variant.band_width,
        "alphaInNormUnits": variant.alpha_in_norm_units,
        "controlType": None,
        "variant": vc.name,
        "adapters": [{"adapterDirectory": a.get("adapterDirectory"),
                      "adapterHash": a.get("adapterHash")}
                     for a in variant.adapters],
    }


def _effective_variant_condition(vc, manifest, model, root, *,
                                 wants_choice: bool) -> EffectiveCondition:
    """Resolve a variant condition to its effective configuration: the stored
    artifact's injections + prompt settings, plus provenance stamped on every
    record. Raises on an invalid variant (the run loop turns that into the
    historical error record and continues).

    System-prompt COMPOSITION (maintainer ruling, 2026-08-24): an agent arm
    generates under its own persona AND the study's frame — persona first
    (:func:`system_prompt.compose`) — not under the persona with the frame
    silently dropped, which is what replacement semantics did and what made
    an agent arm incomparable to the baseline it is contrasted with. An agent
    with no persona (every agent artifact in the workspace today, and every
    newborn agent since `promote` stopped inheriting) composes to the frame
    alone, byte-identically to the historical behaviour."""
    from . import model_variant
    variant = (model_variant.ModelVariant.from_dict(vc.artifact) if vc.artifact
               else model_variant.ModelVariant.from_file(
                   paths.resolve_artifact(vc.artifact_path, root)))
    # The variant must run the study's model (else prompt-family detection and
    # the generation would silently use the wrong model).
    if variant.base_model_id and variant.base_model_id != manifest.model_id:
        raise ValueError(
            f"variant '{vc.name}' base model {variant.base_model_id} != study "
            f"model {manifest.model_id}")
    if variant.base_model_id and variant.base_model_id != model.model_id:
        raise ValueError(
            f"variant '{vc.name}' needs {variant.base_model_id} but {model.model_id} "
            f"is loaded")
    if wants_choice and variant.qwen_thinking_enabled:
        # Same rule the manifest enforces for its own arms: thinking-mode
        # answers are marginals over sampled reasoning paths, so the
        # answer-token instrument (conditional on NO reasoning prefix) is
        # invalid under a thinking-enabled variant.
        #
        # WP0 step 8: typed `thinkingModeConflict`. This per-variant refusal is
        # PYTHON-ONLY (audit §2.4's divergence 5) — the manifest-level rule
        # exists on both engines, this one does not, and the shared gate id
        # must not be read as implying it does.
        raise lifecycle_gates.refusing_value(
            lifecycle_gates.THINKING_MODE_CONFLICT,
            f"variant '{vc.name}' enables thinking mode but the study declares "
            "an answer-token instrument — disable thinking on the variant or "
            "drop the instrument",
            repair=("steerlab-cli experiment set-instruments <name> "
                    "sampledText, or re-promote the agent with thinking "
                    "disabled"))
    return EffectiveCondition(
        name=vc.name,
        injections=model_variant.variant_injections(variant),
        intervention_state=_variant_intervention_state(vc, variant),
        prompt_mode=variant.prompt_mode,
        system_prompt=system_prompt_mod.compose(
            variant.system_prompt, manifest.system_prompt),
        qwen_thinking_enabled=variant.qwen_thinking_enabled,
        # Study-owned sampling (2026-07-21): the MANIFEST owns the measured-
        # run sampling policy for EVERY condition; an agent artifact's stored
        # temperature is a Playground convenience, never a measured-run
        # setting. The effective temperature therefore comes from the
        # manifest — matching _effective_ordinary_condition — so a
        # stochastic study gives its saved agents the same samplesPerItem
        # draws as baseline. (The historical bug: variants required and ran
        # artifact temperature 0, one greedy path against a 20-30-draw
        # baseline — an unbalanced design.) The artifact temperature
        # survives as provenance only: "agentPlaygroundTemperature" below,
        # stamped on every record of the condition; the record's
        # "temperature" field remains the single source of what governed
        # generation.
        temperature=manifest.temperature,
        agent_system_prompt=variant.system_prompt,
        study_system_prompt=manifest.system_prompt,
        provenance={"variantArtifactPath": vc.artifact_path,
                    "variantArtifactHash": vc.artifact_hash,
                    "agentPlaygroundTemperature": variant.temperature},
        variant=variant)


def _run_arm_system_prompts(manifest, conditions, latent_conditions,
                            root) -> list:
    """``(arm name, effective system prompt)`` for every arm of the run
    matrix, in the executor's own emission order (ordinary, then variants,
    then latent).

    Built from the same :func:`system_prompt.compose` rule the three
    ``_effective_*_condition`` resolvers apply, so the advisory can never
    describe an arming the run does not execute. It deliberately does NOT go
    through those resolvers: they also build injection cells (loading every
    vector from disk), and a run-start advisory must not pay for a third
    resolution of the whole matrix. Only the agent artifact's persona is
    read here — the one input the composition takes that this function does
    not already have.

    A variant whose artifact will not load is skipped rather than raised on:
    the condition loop owns that failure (it becomes the historical error
    record), and an advisory must never be the thing that sinks a run.
    """
    from . import model_variant
    frame = manifest.system_prompt
    ordinary = system_prompt_mod.compose(None, frame)
    arms = [(c.name, ordinary) for c in conditions]
    for vc in manifest.variant_conditions:
        try:
            variant = (model_variant.ModelVariant.from_dict(vc.artifact)
                       if vc.artifact
                       else model_variant.ModelVariant.from_file(
                           paths.resolve_artifact(vc.artifact_path, root)))
        except (OSError, KeyError, ValueError, RuntimeError):
            continue  # surfaces as the loop's per-condition error record
        arms.append((vc.name,
                     system_prompt_mod.compose(variant.system_prompt, frame)))
    arms.extend((spec.name, ordinary) for spec, _edit, _p in latent_conditions)
    return arms


def _require_manifest_sampling_policy(manifest, model, root, *,
                                      wants_choice: bool) -> None:
    """Defense in depth for study-owned sampling (2026-07-21): every
    condition of the run matrix must execute under the MANIFEST's sampling
    policy. Ordinary conditions take the manifest temperature structurally
    (:func:`_effective_ordinary_condition` copies it), so the check resolves
    the variant conditions — the resolver that historically diverged (agent
    artifact temperature, forced to 0) and produced an unbalanced design:
    one greedy path per agent against a samplesPerItem-draw baseline.

    Runs BEFORE the condition loop, so a divergent resolver refuses the run
    before any generation compute. A variant that fails to resolve at all is
    skipped here — the condition loop turns it into the historical error
    record and keeps measuring the other conditions."""
    for vc in manifest.variant_conditions:
        try:
            eff = _effective_variant_condition(vc, manifest, model, root,
                                               wants_choice=wants_choice)
        except (OSError, KeyError, ValueError, RuntimeError):
            continue  # surfaces as the loop's per-condition error record
        if eff.temperature != manifest.temperature:
            raise RuntimeError(
                f"condition '{eff.name}' would generate at temperature "
                f"{eff.temperature}, but the study manifest declares "
                f"{manifest.temperature}. The study manifest owns the "
                "sampling policy for EVERY condition — baseline and saved "
                "agents alike — anything else is an unbalanced design; "
                "refusing before any generation runs")


@contextmanager
def _adapters_suspended(model, active: bool):
    """Readers measure TEXT through the base model: suspend an applied PEFT
    adapter for the duration of a reader readout so the instrument is the same
    model under every condition (reading through the condition's own adapter
    would fold the intervention into the instrument)."""
    if not active:
        yield
        return
    lm = model.model
    if hasattr(lm, "disable_adapters"):
        lm.disable_adapters()
    try:
        yield
    finally:
        if hasattr(lm, "enable_adapters"):
            lm.enable_adapters()


def effective_sample_count(manifest) -> int:
    """Generations per (condition, prompt) — the ONE resolver.

    Not ``samplesPerItem``: with ``samplesPerItem == 1`` the run loop emits one
    generation per declared SEED, so a temperature study with three seeds
    produces three. Sharding has always computed it this way; the J-lens budget
    briefly used ``max(1, samplesPerItem)`` and therefore priced such a study at
    a third of its real size (external review round 2). Two call sites, one
    rule, so they cannot disagree again.
    """
    if manifest.samples_per_item > 1 and manifest.temperature > 0:
        return manifest.samples_per_item
    return max(1, len(manifest.seeds))


def _verified_identity_for(variant, root, *, label: str) -> dict:
    """The verified identity block for one agent, or a typed refusal."""
    from . import model_variant as mv

    try:
        adapters = mv.verified_adapter_identity(variant, root)
    except mv.AdapterIdentityError as exc:
        raise RuntimeError(
            f"jlensReadout is armed and condition '{label}' cannot be "
            f"verified: {exc}") from exc
    return {"variantName": getattr(variant, "name", None), "adapters": adapters}


def _require_verified_variant_identities(manifest, root) -> dict:
    """Fail fast at run start: every agent is the agent it declares.

    A declared hash is a claim ABOUT an adapter, not a measurement of the one
    that will load. Legacy agents with no pin are fine (they downgrade the
    row's claim); what refuses is a pin that DISAGREES with the bytes.

    Resolves BOTH artifact forms. Only the embedded `artifact` was handled
    before, so a path-backed condition — an equally supported form the run
    loop resolves the same way — preflighted to nothing and fell through to
    lazy per-row verification, reintroducing both of the previous rounds'
    bugs for exactly that path (external review round 10).

    This is a PREFLIGHT. It is not the binding verification: the authoritative
    one happens immediately before each adapter loads and is attached to the
    EffectiveCondition, because these files can change in between.
    """
    from . import model_variant as mv

    identities: dict[str, dict] = {}
    for vc in getattr(manifest, "variant_conditions", None) or []:
        artifact = getattr(vc, "artifact", None)
        path = getattr(vc, "artifact_path", None)
        if artifact:
            variant = mv.ModelVariant.from_dict(artifact)
        elif path:
            resolved = paths.resolve_artifact(path, root)
            if not resolved or not os.path.isfile(resolved):
                # Absence is the run loop's error to report, per condition,
                # in its own vocabulary; the preflight does not pre-empt it.
                continue
            variant = mv.ModelVariant.from_file(resolved)
        else:
            continue
        identities[getattr(vc, "name", "?")] = _verified_identity_for(
            variant, root, label=getattr(vc, "name", "?"))
    return identities


def _open_jlens_trace(manifest, model, root, *, run_directory, checkpoint,
                      resuming, log, allowed_keys=None,
                      expected_generations: int | None = None,
                      generates_sampled_text: bool | None = None):
    """Resolve a manifest's ``jlensReadout`` into a live trace session, or None.

    Refuses rather than degrades. A declared readout that cannot be armed —
    missing lens, layer the lens never fitted, empty configuration — must stop
    the run at the start, before the model slot is spent, because the
    alternative is a study that completes, reports normally, and contains no
    readout. Nothing downstream could tell that apart from a study that never
    asked for one.

    Server-only by rule (CLAUDE.md): imported lens artifacts are PyTorch/HF
    native, so this path exists on this engine alone.
    """
    block = getattr(manifest, "jlens_readout", None)
    if not block:
        return None
    if not isinstance(block, dict):
        raise RuntimeError(
            f"jlensReadout is present but is a {type(block).__name__}, not an "
            f"object — a readout block is a JSON object of pinned "
            f"declarations, and a malformed one must refuse by name rather "
            f"than fail somewhere downstream")
    from ..jlens.readout import Budget, ReadoutConfig, preflight
    from ..jlens.schemas import JLensError
    from ..jlens.trace import TraceSession

    # Adapter identity, verified at RUN START — before the model slot, the
    # lens load, or the session (external review round 8). The claim stamped
    # on each row comes from the same verifier, but a MISMATCH is not a weaker
    # claim: it means the bytes about to shape generation are not the bytes
    # the agent was pinned with. Discovering that after a multi-hour GPU job
    # has generated is discovering it too late, and this function's whole
    # contract is to refuse at the start rather than degrade.
    verified_identities = _require_verified_variant_identities(manifest, root)

    # A readout only exists where sampled text is generated: the recorder is a
    # read-only observer armed on the GENERATION path, and a deterministic
    # choice/logprob study runs no generations at all. Declaring one there
    # armed the trace, priced the budget, ran nothing, and closed — the run
    # otherwise finishing normally (external review round 3). Refuse before the
    # model slot is spent, in the same spirit as every other run-start refusal
    # here: a study that DECLARES a readout and silently produces none is the
    # exact failure this instrument exists to prevent.
    if generates_sampled_text is False:
        raise RuntimeError(
            "jlensReadout is declared but this study's execution plan "
            "generates no sampled text — the readout rides along on "
            "generation, so it would record nothing and close with an empty "
            "trace. Add a sampled-text instrument, or drop the readout block")

    lens_id = block.get("lensID")
    if not lens_id:
        raise RuntimeError(
            "manifest declares jlensReadout without a lensID — refusing to run "
            "a study whose readout cannot be resolved")
    config = ReadoutConfig(
        layers=[int(x) for x in (block.get("layers") or [])],
        watchlist=[int(x) for x in (block.get("watchlist") or [])],
        topK=int(block.get("topK") or 0),
        topKLayers=[int(x) for x in (block.get("topKLayers") or [])],
        logitLensCompanion=bool(block.get("logitLensCompanion", True)))
    budget = Budget(**{k: v for k, v in (block.get("budget") or {}).items()
                       if k in Budget.__dataclass_fields__})
    # ENFORCE it, before the model slot is spent on a study that cannot fit.
    # The budget was constructed and logged and never consulted, which made a
    # declared ceiling decoration (external review 2026-08-16). The bound is a
    # WHOLE-STUDY quantity by design: computing it per shard would multiply
    # the effective ceiling by the shard count.
    if expected_generations:
        estimate = preflight(config, generations=expected_generations,
                             max_new_tokens=manifest.max_tokens,
                             budget=budget)
        if not estimate["withinBudget"]:
            raise RuntimeError(
                "jlensReadout is over its declared budget and would not fit: "
                + "; ".join(estimate["problems"])
                + f" (projected over {expected_generations} generation(s) × "
                  f"{manifest.max_tokens} tokens). Raise the manifest's budget "
                  f"block as a DECLARED choice, arm fewer top-k layers, or use "
                  f"the watchlist, which needs no full projection at all")
        log(f"J-lens budget: {estimate['observations']} observation(s), "
            f"{estimate['fullVocabProjections']} full-vocab projection(s), "
            f"~{estimate['projectedTraceBytes'] / 1e9:.2f} GB projected trace")
    # DERIVE the tier and the qualification from the runtime that is actually
    # loaded — never inherit the manifest's claim (external review 2026-08-16,
    # P0). Freeze enforces this, but only for server-side freezes: a
    # Swift-frozen manifest never met the gate, --force skips it, and an
    # unfrozen run never reaches it. Every trace row stamps evidenceTier and
    # qualificationID, so a self-declared stamp would travel into the artifact
    # as though it had been checked.
    from ..jlens import importer as jlens_importer
    from ..jlens import lens_store as jlens_store
    from ..jlens.qualification import resolve_runtime

    try:
        record = jlens_store.resolve(lens_id, root)
    except JLensError as exc:
        # Same wording as the arming refusal below: "missing lens" has always
        # been one of the things that cannot be armed, and a second phrase for
        # it would be a second contract.
        raise RuntimeError(
            f"jlensReadout is declared but cannot be armed: {exc}") from exc
    resolved_tier = (jlens_importer.SUPPORTED.get(manifest.model_id or "")
                     or {}).get("tier") or "unknown"
    try:
        runtime_dtype, quantization = resolve_runtime(model)
    except JLensError:
        runtime_dtype, quantization = None, None
    qualification = None
    if runtime_dtype and getattr(model, "revision", None):
        qualification = record.qualification_for(
            manifest.model_id, model.revision, runtime_dtype, quantization,
            layers=config.layers,
            qualification_id=block.get("qualificationID") or None)

    claimed_tier = block.get("evidenceTier")
    if claimed_tier and claimed_tier != resolved_tier:
        raise RuntimeError(
            f"jlensReadout claims evidenceTier {claimed_tier!r} but "
            f"'{manifest.model_id}' resolves to {resolved_tier!r} — a tier is "
            f"a property of the model, not a declaration, and every trace row "
            f"would have carried the claim")
    claimed_qualification = block.get("qualificationID")
    resolved_id = qualification.qualificationID if qualification else None
    if claimed_qualification and resolved_id is None:
        raise RuntimeError(
            f"jlensReadout pins qualification {claimed_qualification!r}, which "
            f"does not resolve for {manifest.model_id}@"
            f"{(getattr(model, 'revision', None) or '?')[:12]}…/"
            f"{runtime_dtype or '?'} over layers {config.layers} — the pin "
            f"must name a passing record bound to these lens bytes and "
            f"covering these layers; refusing to stamp one this runtime does "
            f"not hold")
    if qualification is None:
        log(f"WARNING: J-lens readout is armed on a runtime with NO passing "
            f"qualification ({manifest.model_id}@"
            f"{(getattr(model, 'revision', None) or '?')[:12]}…/"
            f"{runtime_dtype or 'unresolved dtype'}) — every trace row is "
            f"stamped unqualified, and this run's readout is exploratory. "
            f"'steerlab-server jlens qualify' produces the evidence.")

    try:
        session = TraceSession.open(
            run_directory=run_directory, lens_id=lens_id, config=config,
            model=model, root=root, checkpoint=checkpoint, resume=resuming,
            log=log, evidence_tier=resolved_tier,
            qualification_id=resolved_id, budget=budget,
            condition_identities=verified_identities)
    except JLensError as exc:
        raise RuntimeError(
            f"jlensReadout is declared but cannot be armed: {exc}") from exc

    pinned = block.get("configHash")
    if pinned and pinned != session.configHash:
        # The pin is the researcher's declared choices; a mismatch means the
        # block was edited after freeze, which is a firewall violation and not
        # a thing to run through.
        raise RuntimeError(
            f"jlensReadout configuration drifted from its pinned hash "
            f"(have {session.configHash[:12]}…, pinned {pinned[:12]}…) — "
            f"re-freeze, or run the manifest it was frozen from")

    log(f"J-lens readout armed: lens {lens_id}, layers {config.layers}, "
        f"{len(config.watchlist)} watched token(s), topK {config.topK} on "
        f"{config.armed_topk_layers()}, logit-lens companion "
        f"{'on' if config.logitLensCompanion else 'off'} "
        f"(budget: max {budget.maxArmedLayers} layers, "
        f"{budget.maxFullVocabProjections} full-vocab projections)")
    return session


def _execute_condition(model, eff: EffectiveCondition, prompts, writer, *,
                       name, manifest, experiment_hash, wants_choice,
                       wants_sampled, reader_scorers, should_cancel, log,
                       numeric_parser=None, adapter_active: bool = False,
                       jlens_trace=None) -> bool:
    """The shared per-item executor — ONE measurement pipeline for every
    condition: prompt-metadata copy, the declared deterministic instruments,
    sampled generation, categorical parsing, reader scores, and consistent
    record emission all happen here, from the condition's resolved effective
    configuration, with identical resume/checkpoint semantics through the
    run's ``GenerationWriter``. Returns True when a cancel was observed (the
    caller parks the run resumable).

    Record construction is byte-identical to the historical ordinary-condition
    loop for ordinary conditions (``eff`` mirrors the manifest and
    ``provenance`` is empty), which preserves the interrupted+resumed ==
    uninterrupted byte-equality contract.

    Scripted-transcript items run through this SAME executor for every
    condition type: the item's ``transcript`` is threaded into
    ``generate``/``score_options`` (which render it via
    ``prompt_render.render_transcript``), and their records carry
    ``scriptedTranscript: true`` plus the transcript itself — the transcript
    is the stimulus, and records are the rebuild-without-rerun archive."""
    # Condition-scoped transcript gate: variant conditions carry their OWN
    # prompt mode, so a rawCompletion variant over transcript items must
    # refuse at condition start — never as a mid-run template error.
    if eff.prompt_mode == prompt_render.RAW_COMPLETION and any(
            p.get("transcript") for p in prompts):
        raise RuntimeError(
            f"condition '{eff.name}': task prompts include scripted "
            "transcripts but the condition's promptMode is rawCompletion — "
            "transcript items render through the chat template by "
            "definition; use chatAssistant")
    sampling = _sampling_metadata(model, eff.temperature)
    # Threaded as kwargs ONLY for a latent condition, exactly as
    # transcript_kwargs and readout_kwargs are: every other condition's
    # generate/score_options call signature stays byte-for-byte unchanged, so
    # the monkeypatched test fakes built against it keep working.
    latent_kwargs = ({"latent_edits": eff.latent_edits}
                     if eff.latent_edits else {})
    common = {
        "experiment": name, "experimentHash": experiment_hash,
        "modelID": manifest.model_id, "modelRevision": model.revision,
        "promptMode": eff.prompt_mode,
        "systemPromptHash": _sha256_text(eff.system_prompt),
        # …and WHICH LEVELS produced that effective hash (2026-08-24 ruling).
        # Additive provenance beside the effective hash, always present with
        # explicit nulls: the effective hash alone cannot tell a reader
        # whether an arm carried a persona, and after composition landed
        # "no persona" is a fact about the arm rather than the default.
        # Cross-engine spelling: Swift `SystemPromptCompositionStamp`.
        "systemPromptComposition": system_prompt_mod.composition(
            eff.agent_system_prompt, eff.study_system_prompt),
        **eff.provenance,
    }
    for prompt_index, prompt in enumerate(prompts):
        if _observe_cancel(should_cancel, log,
                           f"condition={eff.name} prompt={prompt_index}"):
            return True
        prompt_meta = {k: prompt[k] for k in _PROMPT_META_KEYS if k in prompt}
        transcript = prompt.get("transcript")
        # Threaded as kwargs only when the item carries a transcript, so the
        # plain-item call signature (and every monkeypatched test fake built
        # against it) is byte-for-byte unchanged.
        transcript_kwargs = {"transcript": transcript} if transcript else {}
        transcript_fields = (
            {"scriptedTranscript": True, "transcript": transcript}
            if transcript else {})

        # Answer-token logprob instrument: one deterministic, temperature-free
        # readout per (condition, prompt) — the primary categorical endpoint.
        # Runs regardless of the study's sampling temperature (it never
        # samples), under the condition's injections; a variant's PEFT adapter
        # is already applied to ``model`` by the run loop, so the stepped
        # KV-cache scoring sees the identical forward pass generation would.
        # A declared applicability scope is honored HERE, not merely validated
        # at run start: declining to measure out-of-scope rows is the whole
        # point of declaring one (response_format.scope_includes).
        in_scope = response_format.scope_includes(
            manifest.raw.get("outcomeInstrumentScope"),
            {"id": prompt.get("id"), "hasOptions": True,
             "format": prompt.get("responseFormat")})
        if wants_choice and prompt.get("options") and in_scope and not writer.skip(
                eff.name, prompt_index, prompt["id"], None,
                resume_mod.KIND_INSTRUMENT):
            from . import logprob
            choice = logprob.score_options(
                model, prompt["prompt"], list(prompt["options"]),
                model_id=manifest.model_id, injections=eff.injections,
                **latent_kwargs,
                prompt_mode=eff.prompt_mode,
                system_prompt=eff.system_prompt,
                qwen_thinking_enabled=eff.qwen_thinking_enabled,
                **transcript_kwargs)
            _check_option_lengths(choice, manifest, prompt["id"])
            # Ordinal-scale fields (cross-engine contract keys
            # "ordinalPosition"/"ordinalDistribution"): the option
            # probabilities renormalized over the item's declared ladder (in
            # ladder order) and the 1-based position under the manifest's
            # declared aggregation. Keys absent when ordinalScale is not
            # declared (Swift ChoiceRecord twin).
            ordinal_fields = {}
            aggregation = _resolve_ordinal_aggregation(manifest)
            if aggregation is not None:
                distribution = logprob.ordinal_distribution(
                    choice.ordered_probabilities)
                ordinal_fields = {
                    "ordinalDistribution": distribution,
                    "ordinalPosition": logprob.ordinal_position(
                        distribution, aggregation),
                }
            record = {
                **common,
                "condition": eff.name,
                "promptIndex": prompt_index,
                "promptID": prompt["id"], "prompt": prompt["prompt"],
                # DECLARED target only (open-issues #6): defaulting to
                # options[0] stamped every ordinalScale record with the scale
                # minimum as its "target", and analyze then emitted a
                # choiceLogOdds endpoint nobody declared (log-odds of rating
                # token "1" — pole movement confounded with distribution
                # sharpening). A None target is a fact about the item, and
                # analyze emits choiceLogOdds only for declared ones.
                "target": prompt.get("target"),
                "targetSource": "declared" if prompt.get("target") else None,
                "interventionState": eff.intervention_state,
                **prompt_meta,
                **transcript_fields,
                **choice.as_record_fields(),
                **ordinal_fields,
                # Instrument readouts are temperature-free by construction.
                **{**sampling, "temperature": 0.0, "doSample": False,
                   "topP": None, "topK": None},
            }
            writer.emit(record)
            memory_diagnostic.observe(writer, model, eff)

        if not wants_sampled:
            continue
        if manifest.samples_per_item > 1 and eff.temperature > 0:
            samples = [(i, derive_seed(experiment_hash, eff.name,
                                       prompt["id"], i))
                       for i in range(manifest.samples_per_item)]
            seed_policy = "derivedSHA256"
        else:
            samples = list(enumerate(manifest.seeds))
            seed_policy = "manifestSeeds"
        for sample_index, seed in samples:
            if writer.skip(eff.name, prompt_index, prompt["id"],
                           sample_index, resume_mod.KIND_SAMPLED):
                continue
            # Seeded PER RECORD, so skipping completed records cannot
            # shift the sample stream of the ones still to generate —
            # and generation-LOCAL (fork_rng), so concurrent seeded
            # studies cannot corrupt each other's streams.
            # J-lens readout rides along READ-ONLY when declared: a recorder
            # per generation (never shared), armed after the injectors so it
            # observes the post-intervention residual, and the sampled ids
            # retained so the trace can be prediction-aligned to what was
            # actually emitted rather than to streamed text.
            recorder, token_ids = None, None
            # Threaded as kwargs ONLY when something needs them, exactly as
            # transcript_kwargs is: the plain call signature stays byte-for-byte
            # unchanged, so every monkeypatched test fake built against it keeps
            # working. Passing observers=None still widens the call.
            #
            # Two independent reasons to capture the sampled ids: the J-lens
            # recorder needs them to align its observations (it always has),
            # and `recordTokenIDs` retains them on the record so a completed
            # run stays exactly replayable (see Manifest.record_token_ids).
            readout_kwargs = {}
            if jlens_trace is not None:
                recorder = jlens_trace.recorder_for(prompt)
                if recorder is not None:
                    token_ids = []
                    readout_kwargs = {"observers": [recorder],
                                      "token_ids_out": token_ids}
            if manifest.record_token_ids and token_ids is None:
                token_ids = []
                readout_kwargs["token_ids_out"] = token_ids
            with _seeded_generation(eff.temperature, seed):
                text = generate(
                    model, prompt["prompt"], model_id=manifest.model_id,
                    max_tokens=manifest.max_tokens, temperature=eff.temperature,
                    injections=eff.injections, **latent_kwargs,
                    prompt_mode=eff.prompt_mode,
                    system_prompt=eff.system_prompt,
                    qwen_thinking_enabled=eff.qwen_thinking_enabled,
                    **readout_kwargs, **transcript_kwargs)
            record = {
                **common,
                "condition": eff.name, "seed": seed,
                "seedPolicy": seed_policy,
                "sampleIndex": sample_index,
                "promptIndex": prompt_index,
                "promptID": prompt["id"], "prompt": prompt["prompt"],
                "interventionState": eff.intervention_state,
                **prompt_meta,
                **transcript_fields,
                "output": text, "wordCount": word_count(text),
                "distinct2": distinct_bigram_ratio(text),
                **sampling,
            }
            # The exact sampled sequence, for replay. Written only when the
            # study DECLARED it: a J-lens run captures ids for alignment
            # regardless, and persisting those as a side effect would make the
            # record shape depend on an unrelated block. Omitted rather than
            # written empty when nothing was captured — an empty list would
            # read as "this generation emitted nothing", a different claim
            # from "the ids were not retained".
            if manifest.record_token_ids and token_ids:
                record["outputTokenIDs"] = [int(t) for t in token_ids]
            # Numeric endpoint parse: a DECLARED registry parser wins (the
            # study's own unit grammar, workspace data); otherwise the
            # historical case-family rule (sentencing -> built-in months
            # parser) applies unchanged.
            if numeric_parser is not None:
                record["parsedMonths"] = numeric_parser.parse(text)
            elif manifest.case_family == "sentencing":
                record["parsedMonths"] = judicial.parse_months(text)
            if prompt.get("options"):
                record["parsedChoice"] = judicial.parse_choice(
                    text, list(prompt["options"]))
            if reader_scorers:
                with _adapters_suspended(model, adapter_active):
                    record["readerScores"] = _reader_scores(
                        model, reader_scorers, text)
            if recorder is not None:
                # generations.jsonl gets a REFERENCE and small summaries; the
                # per-step observations live in jlens-readout.jsonl.
                record["jlensReadout"] = jlens_trace.record_generation(
                    recorder, eff, prompt, prompt_index, sample_index,
                    model=model, manifest=manifest,
                    generated_ids=token_ids or [])
            writer.emit(record)
            memory_diagnostic.observe(writer, model, eff)
    return False


def _run_capability_battery(model, name, manifest, bundles, conditions, root,
                            run_directory, should_cancel, _log, checkpoint,
                            resuming, condition_filter=None,
                            latent_conditions=()) -> tuple[dict, bool]:
    """Score the manifest's PINNED capability battery under EVERY condition of
    the run matrix — baseline, concept/steering, variant AND SAE latent arms —
    with each condition's intervention applied exactly as for that condition's
    generations (2026-07-13; Swift identical for the arms Swift has).

    "Every condition" is a claim this docstring once made while the loop ran
    only ``conditions`` + ``manifest.variant_conditions``: a declared SAE
    latent arm got NO battery row at all, so the one arm whose mechanism is
    least understood was also the only arm with no capability control
    (review finding 2, repaired 2026-08-13). Latent arms are now scored last
    — the same order the executor emits them in, which is the order the shard
    plan's key list assumes — with their ``latent_edits`` armed on both
    scoring back-ends, and their rows carry the condition's
    ``interventionState`` so a battery.jsonl row says which mechanism was
    live rather than being trusted to have had one.

    Battery generations are GATE EVIDENCE, not outcomes: they stream to a
    separate ``battery.jsonl`` (same append/skip/checkpoint discipline as
    generations.jsonl, so checkpoint/resume covers them record-for-record)
    and never enter generations.jsonl. Returns
    ``({condition: {"accuracy", "itemCount", "batteryHash"}}, cancelled)`` —
    the per-condition block ``_write_report`` stamps into report.json under
    the cross-engine key ``capabilityBattery``.

    Arming follows the battery's FORMAT (2026-08-13 repair). A format-2
    battery declares its own promptMode/systemPrompt/maxTokens and EVERY
    condition is scored under that same arming, so the intervention is the
    only thing that varies; its items are scored by answer-token logprob, so
    a condition cannot gain or lose accuracy by changing how much it writes.
    A legacy (format-1) battery is armed exactly as before — the manifest's
    context for baseline/steering conditions, the variant artifact's for
    variant conditions — so its pinned hash keeps its historical meaning; the
    asymmetry that produced the c19 "steering improves capability" reading is
    now logged as a contamination advisory rather than passing silently.
    """
    from . import battery as battery_mod, model_variant
    battery_file = manifest.capability_battery_file
    spec = battery_mod.load_spec(battery_file, root)
    items, digest = spec.items, spec.digest
    if digest != manifest.capability_battery_hash:
        raise RuntimeError(
            f"capability battery '{battery_file}' drifted from the pinned hash "
            f"(have {digest[:12]}…, pinned {manifest.capability_battery_hash[:12]}…)")
    writer = resume_mod.GenerationWriter(
        run_directory, verb="run", checkpoint=checkpoint, resume=resuming,
        log=_log, filename="battery.jsonl")
    cancelled = False
    advised: set[str] = set()

    def _score_condition(condition_name, injections, prompt_mode, system_prompt,
                         thinking, latent_edits=None,
                         intervention_state=None,
                         agent_system_prompt=None) -> bool:
        """Emit one battery record per item under the given arming. Returns
        False when a cancel was observed (partial condition kept, resumable).

        ``latent_edits``/``intervention_state`` are the SAE latent arm's
        mechanism and its provenance stamp. Both default to off and neither
        touches a non-latent condition's record bytes.

        ``system_prompt`` is the FORMAT-1 caller context — the study manifest's
        for baseline/steering/latent arms, the artifact's for a variant arm —
        preserved byte for byte so a legacy battery's pinned hash keeps its
        historical meaning. ``agent_system_prompt`` is the FORMAT-2 persona:
        composed ahead of the battery file's own declared arming, while the
        STUDY frame reaches a format-2 reading through neither channel
        (2026-08-24 battery-isolation ruling). Baseline therefore reads the
        battery bare, which is what makes it the control for the agent arms.
        """
        arming = battery_mod.resolve_arming(
            spec, prompt_mode=prompt_mode, system_prompt=system_prompt,
            qwen_thinking_enabled=thinking,
            agent_system_prompt=agent_system_prompt)
        advisory = battery_mod.contamination_advisory(spec, arming)
        if advisory and advisory not in advised:
            advised.add(advisory)
            _log(f"WARNING: {advisory}")
        generate_fn, choice_fn = _battery_backends(
            model, manifest.model_id, injections, latent_edits=latent_edits)
        for index, item in enumerate(items):
            if _observe_cancel(should_cancel, _log,
                               f"battery condition={condition_name} item={index}"):
                return False
            # ONE prompt id per item, used for both the resume probe and the
            # emitted record. ``resume.record_key`` keys on promptID, so a
            # skip() probe that disagreed with what emit() stores would match
            # nothing on resume: every completed item's forward pass would be
            # re-run and only emit()'s dedupe would suppress the duplicate row
            # — i.e. resume would silently stop saving the expensive half.
            prompt_id = battery_mod.item_prompt_id(item, index)
            if writer.skip(condition_name, index, prompt_id, 0,
                           resume_mod.KIND_SAMPLED):
                continue
            fields = battery_mod.score_item(
                spec, item, arming, generate_fn=generate_fn,
                choice_fn=choice_fn)
            record = {
                "condition": condition_name, "promptIndex": index,
                "promptID": prompt_id,
                "sampleIndex": 0,
                "prompt": item["prompt"], "answer": item["answer"],
            }
            if intervention_state is not None:
                # Latent arms only (2026-08-13): the SAME block generations.
                # jsonl carries, so one reader parses both files and a battery
                # row is self-describing about what was armed — release, SAE,
                # feature, layer, mode, β in latent units, and the pinned
                # repository commit. Absent for every other condition, whose
                # record bytes are therefore unchanged.
                record["interventionState"] = intervention_state
            if spec.isolated:
                # Format-2 provenance: what the reading was armed with, so a
                # stored battery.jsonl is self-describing.
                record["batteryFormat"] = spec.format_version
                record.update(arming.as_record_fields())
                record["batteryHash"] = digest
                record.update(fields)
            else:
                # Legacy records keep their exact historical key set and
                # order — an old battery.jsonl and a new one must diff clean.
                record["output"] = fields["output"]
                record["batteryHash"] = digest
                record["correct"] = fields["correct"]
            writer.emit(record)
        return True

    try:
        for condition in conditions:
            if condition_filter is not None \
                    and condition.name not in condition_filter:
                continue  # another shard owns this condition's battery
            injections = _condition_injections(condition, bundles)
            if not _score_condition(condition.name, injections,
                                    manifest.prompt_mode, manifest.system_prompt,
                                    manifest.qwen_thinking_enabled):
                cancelled = True
                break
        if not cancelled:
            for vc in manifest.variant_conditions:
                if condition_filter is not None \
                        and vc.name not in condition_filter:
                    continue  # another shard owns this condition's battery
                try:
                    if vc.from_promotion:
                        vc, _ = _resolve_forward_variant(
                            vc, manifest, root, _log)
                    variant = (model_variant.ModelVariant.from_dict(vc.artifact)
                               if vc.artifact else model_variant.ModelVariant.from_file(
                                   paths.resolve(vc.artifact_path, root)))
                    injections = model_variant.variant_injections(variant)
                    adapter = model_variant.apply_adapter(model, variant, root=root)
                except (OSError, KeyError, ValueError, RuntimeError) as exc:
                    _log(f"battery: variant '{vc.name}' skipped: {exc}")
                    continue
                try:
                    completed = _score_condition(
                        vc.name, injections, variant.prompt_mode,
                        variant.system_prompt, variant.qwen_thinking_enabled,
                        agent_system_prompt=variant.system_prompt)
                finally:
                    model_variant.remove_adapter(model, adapter)
                if not completed:
                    cancelled = True
                    break
        if not cancelled:
            # SAE latent arms, last — the executor's own emission order, which
            # the shard plan's key list mirrors. The effective condition is
            # built by the SAME helper the run loop uses, so the battery can
            # never be armed differently from the generations it is the
            # control for.
            for spec_l, edit, provenance in latent_conditions:
                if condition_filter is not None \
                        and spec_l.name not in condition_filter:
                    continue  # another shard owns this condition's battery
                eff = _effective_sae_latent_condition(
                    spec_l, edit, provenance, manifest)
                if not _score_condition(
                        eff.name, eff.injections, eff.prompt_mode,
                        eff.system_prompt, eff.qwen_thinking_enabled,
                        latent_edits=eff.latent_edits,
                        intervention_state=eff.intervention_state):
                    cancelled = True
                    break
        if cancelled:
            writer.interrupt(reason="cancel")
    finally:
        writer.close()

    summary: dict[str, dict] = {}
    if not cancelled:
        correct: dict[str, int] = {}
        seen: set[str] = set()
        for record in writer.records:
            seen.add(record["condition"])
            if record.get("correct"):
                correct[record["condition"]] = correct.get(record["condition"], 0) + 1
        summary = {condition: {
            "accuracy": correct.get(condition, 0) / len(items),
            "itemCount": len(items),
            "batteryHash": digest,
        } for condition in sorted(seen)}
        _log(f"capability battery ({len(items)} items × {len(summary)} "
             f"condition(s)) → battery.jsonl")
    return summary, cancelled


def no_rubric_refusal(name: str) -> str:
    """The refusal both engines give an evaluation with no rubric at all.

    Byte-identical to Swift's ``JudgeRubricStore.noRubricRefusal``, and it
    names ``steerlab-cli`` on BOTH engines on purpose: authoring is
    Mac-authority (WP0 §10.x), and this CLI has no ``pin-rubric`` verb to
    name."""
    return (f"study '{name}' has no judge rubric — pin one: "
            f"'steerlab-cli experiment pin-rubric {name} "
            "prompts/rubrics/default-paired-v1.md' (any file under "
            "prompts/rubrics/; inline draft text is draft-only and cannot "
            "freeze)")


def no_rubric_repair(name: str) -> str:
    """The repair for :func:`no_rubric_refusal` on THIS engine: pin on the
    Mac (the only CLI with a ``pin-rubric`` verb), then re-run here."""
    return (f"steerlab-cli experiment pin-rubric {name} "
            "prompts/rubrics/default-paired-v1.md  (authoring is "
            f"Mac-authority) ; then steerlab-server experiment evaluate {name}")


def missing_rubric_refusal(path: str) -> str:
    """The refusal for a named rubric path that is not on disk.

    Byte-identical to Swift's ``JudgeRubricStore.missingRubricRefusal``. Both
    engines used to let the filesystem error out unhandled here — Swift printed
    an ``NSCocoaErrorDomain`` dump, this engine a ``FileNotFoundError``
    traceback and exit 1 (observed 2026-08-18)."""
    return f"judge rubric file not found: {path}"


def missing_rubric_repair(name: str, relative: str) -> str:
    """The repair on THIS engine: author the file under the convention
    directory, pin it on the Mac (authoring is Mac-authority — this CLI has no
    ``pin-rubric`` verb), then re-run here. The Swift twin
    (``JudgeRubricStore.missingRubricRepair``) is the same sentence without the
    Mac-authority note and the server re-run, exactly as
    :func:`no_rubric_repair` differs from Swift's."""
    default = "prompts/rubrics/default-paired-v1.md"
    # When the ABSENT path IS the shipped default, naming it as an example
    # would be the repair pointing at itself (Swift does the same).
    example = (f" (a seeded workspace ships {default})" if relative == default
               else f" (the shipped {default} is one)")
    return (f"author {relative} under prompts/rubrics/{example}, then "
            f"steerlab-cli experiment pin-rubric {name} {relative}  "
            f"(authoring is Mac-authority) ; then steerlab-server experiment "
            f"evaluate {name}")


def missing_task_prompts_refusal(path: str) -> str:
    """Twin of Swift's ``ExperimentTasks.loadTaskPrompts`` /
    ``ExperimentStore.pinTaskPrompts`` sentence for a prompt set that is not on
    disk. Swift has typed this since step 7; this engine raised a bare
    ``FileNotFoundError``."""
    return f"task prompt file not found: {path}"


def missing_task_prompts_repair(name: str, relative: str) -> str:
    return (f"author {relative} as {{\"id\": …, \"prompt\": …}} JSONL rows, "
            f"then steerlab-cli experiment pin-prompts {name} {relative}  "
            f"(authoring is Mac-authority) ; then steerlab-server experiment "
            f"run {name}")


def _resolve_rubric(manifest: Manifest, root, _log) -> tuple[str, str | None, str | None]:
    """The rubric text evaluate judges with: ``(text, sha256|None, file|None)``.

    Frozen studies MUST judge from the pinned rubric file (never an unpinned
    inline string); a draft may fall back to the manifest's inline
    ``evaluation.judgePrompt`` with a loud warning. Drift between the file and
    its pin is refused at read time, like task prompts.

    An EMPTY inline fallback is refused, not warned about (WP0 dry run #2's
    skipped check, Christian-flagged): a draft study with an evaluation block
    and no rubric anywhere used to reach the judges with the empty string as
    its rubric — this engine emitted twelve blinded judging packets built on
    nothing and exited 0. Swift has always refused exactly this input
    (``JudgeRubricStore.resolveRubric``); the sentence is now the same one."""
    if manifest.judge_rubric_file:
        path = paths.resolve(manifest.judge_rubric_file, root)
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except FileNotFoundError:
            raise lifecycle_gates.refusing(
                lifecycle_gates.MISSING_PREREQUISITE,
                missing_rubric_refusal(str(path)),
                repair=missing_rubric_repair(
                    manifest.name, manifest.judge_rubric_file)) from None
        live = hashlib.sha256(data).hexdigest()
        if manifest.judge_rubric_hash and live != manifest.judge_rubric_hash:
            raise RuntimeError(
                f"judge rubric '{manifest.judge_rubric_file}' drifted from the "
                f"pinned hash (have {live[:12]}…, pinned "
                f"{manifest.judge_rubric_hash[:12]}…)")
        if not manifest.judge_rubric_hash and manifest.status == "frozen":
            raise RuntimeError(
                "frozen study names a judge rubric file without judgeRubricHash "
                "— duplicate, pin both, and re-freeze")
        return data.decode("utf-8"), live, manifest.judge_rubric_file
    if manifest.status == "frozen":
        raise RuntimeError(
            f"frozen study '{manifest.name}' has no pinned judge rubric — a "
            "frozen evaluation must judge from a hashed rubric FILE "
            "(judgeRubricFile + judgeRubricHash; see prompts/rubrics/). "
            "Duplicate, pin one, and re-freeze")
    inline = (manifest.evaluation.judge_prompt if manifest.evaluation
              else "") or ""
    if not inline.strip():
        raise lifecycle_gates.refusing(
            lifecycle_gates.MISSING_PREREQUISITE,
            no_rubric_refusal(manifest.name),
            repair=no_rubric_repair(manifest.name))
    _log("WARNING: judging with an UNPINNED inline rubric (draft study) — pin "
         "a rubric file under prompts/rubrics/ before freezing")
    return inline, None, None


# --- pipeline (chain runner, stage 3 of the seamless pipeline) ----------------

#: pipeline.json schema. 1 = stage ledger with per-stage run dirs, bare-path
#: promote records, disposition completed|aborted, abort record. 2 = promote
#: records are FULL pins ({path, hash, sweepRun, winningCell}) the run stage
#: consumes verbatim — a schema-1 ledger predates the pin contract, and
#: resuming it would fall back to ambient catalog resolution, so resume
#: REFUSES it (the chain re-runs fresh under the current contract).
#: Schema 2 grew two ADDITIVE keys 2026-08-06 (no schema bump — absent
#: means "none", every reader tolerates absence): ``epochDriftAtContinuation``
#: (the loud-and-stamped record of a resume that proceeded past manifest
#: drift) and ``parked`` (the startup reconciler's orphan stamp — completed
#: stages, null disposition, no live job; cleared by the next successful
#: resume).
PIPELINE_LEDGER_SCHEMA = 2


def _pipeline_ledger_path(run_directory: str) -> str:
    return os.path.join(run_directory, "pipeline.json")


def _write_pipeline_ledger(run_directory: str, ledger: dict) -> None:
    """Atomic (tmp + rename): the ledger is what a Slurm requeue resumes
    from, so it must always be canonical-or-previous, never torn. Every
    write stamps ``updatedAt`` — a chain with no terminal disposition is
    "unfinished", and the timestamp is what lets a reader judge whether it
    is plausibly still running or was abandoned (sixth round)."""
    ledger["updatedAt"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    path = _pipeline_ledger_path(run_directory)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _read_pipeline_ledger(run_directory: str) -> dict:
    with open(_pipeline_ledger_path(run_directory), encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("schema") != PIPELINE_LEDGER_SCHEMA:
        found = data.get("schema") if isinstance(data, dict) else None
        raise ValueError(
            f"'{run_directory}' is not a resumable pipeline run: ledger "
            f"schema {found!r} != {PIPELINE_LEDGER_SCHEMA} (or malformed). "
            "A schema-1 ledger predates the promote-pin contract — resuming "
            "it would fall back to ambient catalog resolution; start a "
            "fresh pipeline instead")
    return data


def list_pipeline_runs(name: str | None = None,
                       root: str | None = None) -> list[dict]:
    """Pipeline runs for ``name`` — or EVERY experiment when ``name`` is
    None (the Compute panel's awaiting-import affordance, 2026-08-06) —
    newest first. Tolerant where RESUME is strict: a schema-1 ledger is
    listable (display is not resume) and unreadable dirs are skipped. Each
    row: run id, experiment, schema, disposition (``completed`` |
    ``aborted`` | null = in flight/awaiting), an ordered stage summary,
    the abort record when present, the promoted-agent pins, and — when
    stamped — the ``parked`` block and ``epochDriftAtContinuation``."""
    runs_root = paths.runs_directory(root)
    try:
        entries = sorted(os.listdir(runs_root), reverse=True)
    except OSError:
        return []
    out: list[dict] = []
    for dirname in entries:
        ledger_path = os.path.join(runs_root, dirname, "pipeline.json")
        if not os.path.isfile(ledger_path):
            continue
        try:
            with open(ledger_path, encoding="utf-8") as handle:
                ledger = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(ledger, dict):
            continue
        if name is not None and ledger.get("experiment") != name:
            continue
        stage_results = ledger.get("stageResults") or {}
        stages = []
        for stage in ledger.get("stages") or []:
            entry = stage_results.get(stage) or {}
            stage_run = entry.get("runDirectory")
            stages.append({
                "stage": stage,
                "status": entry.get("status") or "pending",
                "runID": (os.path.basename(stage_run)
                          if isinstance(stage_run, str) and stage_run
                          else None)})
        row: dict = {"run": dirname, "schema": ledger.get("schema"),
                     "experiment": ledger.get("experiment"),
                     "disposition": ledger.get("disposition"),
                     "experimentHash": ledger.get("experimentHash"),
                     "updatedAt": ledger.get("updatedAt"),
                     "manifestStatus": ledger.get("manifestStatus"),
                     "stages": stages}
        if isinstance(ledger.get("parked"), dict):
            # The startup reconciler's orphan stamp (2026-08-06): a chain
            # with completed stages, no terminal disposition, and no live
            # job. The row keeps disposition null — parked is a STATE OF
            # BEING UNFINISHED, not a terminal outcome — and the app offers
            # resume/import from this block.
            row["parked"] = ledger["parked"]
        if isinstance(ledger.get("epochDriftAtContinuation"), list) \
                and ledger["epochDriftAtContinuation"]:
            row["epochDriftAtContinuation"] = \
                ledger["epochDriftAtContinuation"]
        abort = ledger.get("abort")
        if isinstance(abort, dict):
            evidence = abort.get("evidenceRunDirectory")
            row["abort"] = {
                "stage": abort.get("stage"),
                "gates": abort.get("gates") or [],
                "evidenceRunID": (os.path.basename(evidence)
                                  if isinstance(evidence, str) and evidence
                                  else None)}
        concepts = (stage_results.get("promote") or {}).get("concepts") or {}
        if concepts:
            agents: dict = {}
            for concept, pin in concepts.items():
                if isinstance(pin, dict):
                    agents[concept] = {
                        "artifact": os.path.basename(str(pin.get("path")
                                                         or "")),
                        "hash": pin.get("hash"),
                        "sweepRun": pin.get("sweepRun"),
                        "winningCell": pin.get("winningCell")}
                else:  # legacy schema-1 bare path
                    agents[concept] = {"artifact": os.path.basename(str(pin))}
            row["promotedAgents"] = agents
        out.append(row)
    return out


def _pipeline_will_judge(manifest: Manifest, stages: list[str]) -> bool:
    """Whether any REMAINING stage of the chain will call judges: evaluate
    always does; a sweep does only under the judgeScore objective."""
    if "evaluate" in stages:
        return True
    if "sweep" in stages:
        sweep_block = manifest.raw.get("sweep")
        selection = (sweep_block.get("selection")
                     if isinstance(sweep_block, dict) else None)
        objective = (selection.get("objective")
                     if isinstance(selection, dict) else None)
        return (isinstance(objective, dict)
                and objective.get("metric") == "judgeScore")
    return False


def _pipeline_inline_judging_preflight(manifest: Manifest,
                                       stages: list[str]) -> None:
    """The chain requires an IN-JOB-resolvable judging path (design contract,
    2026-07-18): deferral hands the selection/evaluation to a later Mac
    session, which cannot be a job dependency — so a sweep or evaluate stage
    that would defer refuses HERE, before the model loads. The manual
    two-phase flow (sweep → Mac judgment → promote → run) remains available
    outside the chain. ``stages`` is the REMAINING stage list: a resumed
    chain whose judged sweep already completed must not refuse because a
    transient judge key has since been cleared."""
    from . import sweep_selection
    # Finding 1 guard, fan-out era (2026-07-23): the chain holds ONE model,
    # so a judged SWEEP whose LOCAL judge declares a different model still
    # refuses HERE, before the model loads — the judge fan-out exists for
    # the post-generation EVALUATE stage only (a sweep's judging is
    # interleaved with the selection, not a separable post-stage). An
    # evaluate stage with such judges no longer refuses: it emits blinded
    # packets, the controller fans one worker job per distinct judge model,
    # and the merge resumes the chain.
    if "sweep" in stages:
        sweep_block = manifest.raw.get("sweep")
        selection_block = (sweep_block.get("selection")
                           if isinstance(sweep_block, dict) else None)
        objective_block = (selection_block.get("objective")
                           if isinstance(selection_block, dict) else None)
        if isinstance(objective_block, dict) \
                and objective_block.get("metric") == "judgeScore":
            offenders = []
            for j in (manifest.raw.get("judges") or []):
                if not (isinstance(j, dict) and j.get("name")):
                    continue
                if (str(j.get("kind") or "claude").strip()
                        or "claude") != "local":
                    continue
                declared = str(j.get("model") or "").strip()
                if declared and declared != manifest.model_id:
                    offenders.append(f"'{j['name']}' (model '{declared}')")
            if offenders:
                raise RuntimeError(
                    "the pipeline's sweep stage holds ONE model — the "
                    f"study model '{manifest.model_id}' — but local "
                    "judge(s) " + ", ".join(offenders) + " resolve to a "
                    "different model, which cannot load inside the chain "
                    "(the judge fan-out covers the evaluate stage only). "
                    "Leave a local judge's model empty to judge with the "
                    "study model, pin claude/openrouter judges, or select "
                    "on logprobShift")
    if "sweep" in stages:
        sweep_spec = manifest.raw.get("sweep")
        if isinstance(sweep_spec, dict):
            selection = sweep_spec.get("selection")
            criterion = sweep_selection.resolve_selection(selection)
            if criterion.metric == "judgeScore":
                objective = sweep_selection.resolve_objective(
                    criterion, selection,
                    judge_rubric_file=manifest.judge_rubric_file,
                    judge_rubric_hash=manifest.judge_rubric_hash,
                    judge_refs=manifest.judges,
                    judges_raw=(manifest.raw.get("judges") or []))
                if objective.defer_judging:
                    raise RuntimeError(
                        "the pipeline requires INLINE judging: the sweep's "
                        "judgeScore selection would DEFER (no credential "
                        "for its external judges on this host) and a chain "
                        "cannot wait for a Mac session — push a judge key "
                        "(app: Compute → External judge key), pin a local "
                        "judge, or select on logprobShift")
    if "evaluate" in stages:
        from .manifest import EVALUATE_WITHOUT_JUDGING_MESSAGE
        # The same effective-evaluation rule the evaluate stage resolves
        # with (2026-07-22): judges + a pinned rubric file count; a chain
        # with neither an evaluation block nor that pin pair refuses HERE,
        # with the verify gate's exact wording, before the model loads.
        spec, _source = manifest.effective_evaluation()
        if spec is None or spec.kind != "pairedJudge":
            raise RuntimeError(EVALUATE_WITHOUT_JUDGING_MESSAGE)
        roster = _judge_roster(manifest, spec)
        missing = _missing_external_credentials(roster)
        if missing:
            raise RuntimeError(
                "the pipeline requires INLINE judging: evaluate would "
                f"DEFER (no credential for {'/'.join(sorted(missing))} "
                "judges on this host) and a chain cannot wait for a Mac "
                "session — push a judge key (app: Compute → External judge "
                "key), pin a local panel, or drop the evaluate stage")


def _pipeline_needs_model(remaining: list[str], manifest: Manifest) -> bool:
    """Whether any remaining stage holds the GPU model. ``evaluate`` needs
    it only for LOCAL judges that judge INLINE (all resolving to the study
    model) — a fan-out evaluate only EMITS packets (CPU), and its judging
    runs in per-judge-model worker jobs (2026-07-23); ``promote``/
    ``analyze`` are CPU-only, so an all-CPU remainder (a requeue that died
    between promote and analyze) never loads the model at all."""
    from . import pipeline_spec as pspec
    if any(stage in pspec.GPU_STAGES for stage in remaining):
        return True
    if "evaluate" in remaining:
        from .manifest import EvaluationSpec
        if evaluate_fanout_judge_models(manifest):
            return False
        roster = _judge_roster(manifest,
                               manifest.evaluation or EvaluationSpec())
        return any(ref.kind == "local" for ref in roster)
    return False


def _expected_promotion_identity(name: str, concept: str, root: str | None,
                                 *, chain_sweep_run: str | None = None):
    """The SELECTION identity a criterion promotion of this concept would
    stamp right now: ``(sweepRun, winningCell)`` from the same evidence
    ``promote`` reads — the manifest's ``<concept>-recommended`` condition,
    else the newest sweep run's recommendations entry. When
    ``chain_sweep_run`` names a chain's own sweep stage, the evidence must
    come from that exact sweep run (a frozen manifest keeps one hash across
    many sweeps, so the hash alone cannot distinguish this chain's
    selection from an earlier one). Returns None when no unambiguous
    identity exists."""
    from . import promote as promote_lib
    manifest = Manifest.load(name, root)
    selection = None
    recommended = next(
        (c for c in manifest.raw.get("conditions", [])
         if c.get("name") == f"{concept}-recommended"
         and isinstance(c.get("selection"), dict)), None)
    if recommended is not None:
        selection = recommended["selection"]
    else:
        evidence = promote_lib._newest_sweep_evidence(name, concept, root)
        if evidence is not None and isinstance(evidence[1], dict):
            selection = evidence[1]
    if not isinstance(selection, dict):
        return None
    sweep_run = selection.get("sweepRun")
    cell = selection.get("winningCell")
    if not sweep_run or not isinstance(cell, dict):
        return None
    if chain_sweep_run and chain_sweep_run != sweep_run:
        return None  # the evidence is not this chain's sweep — never adopt
    return str(sweep_run), {"layer": int(cell.get("layer")),
                            "alpha": float(cell.get("alpha"))}


def _find_minted_agent(name: str, concept: str, ledger: dict,
                       root: str | None) -> str | None:
    """Crash-recovery for the promote stage (RECOVERY ONLY — the caller
    gates on the ledger's promote stage being mid-flight): a variant
    artifact minted by a criterion promotion of this concept whose FULL
    selection identity matches what a promotion would stamp right now —
    experiment, manifest epoch, sweep run, and winning cell. The epoch
    alone is not identity: a frozen manifest keeps one hash across many
    sweeps, so an earlier pipeline's agent (different cell, same hash)
    must never be adopted. Returns the artifact path, or None."""
    chain_sweep = (ledger.get("stageResults", {}).get("sweep") or {}).get(
        "runDirectory")
    expected = _expected_promotion_identity(
        name, concept, root,
        chain_sweep_run=os.path.basename(chain_sweep) if chain_sweep
        else None)
    if expected is None:
        return None
    return _minted_agent_matching(name, concept, root, expected=expected,
                                  live_hash=ledger.get("experimentHash"))


def _minted_agent_matching(name: str, concept: str, root: str | None, *,
                           expected, live_hash) -> str | None:
    """The newest variant artifact whose promotion birth certificate
    matches the FULL selection identity: experiment, manifest epoch,
    criterion promotion, sweep run, winning cell, and concept."""
    expected_sweep, expected_cell = expected
    runs = paths.runs_directory(root)
    try:
        entries = sorted(os.listdir(runs), reverse=True)  # newest first
    except OSError:
        return None
    for dirname in entries:
        if "-variant-" not in f"-{dirname}":
            continue
        run_dir = os.path.join(runs, dirname)
        if not os.path.isdir(run_dir):
            continue
        for fname in os.listdir(run_dir):
            if not fname.endswith(".json") or fname == "config.json":
                continue
            try:
                with open(os.path.join(run_dir, fname),
                          encoding="utf-8") as handle:
                    d = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            promotion = d.get("promotion")
            if not isinstance(promotion, dict):
                continue
            cell = promotion.get("winningCell") or {}
            injections = d.get("injections") or []
            if (promotion.get("experiment") == name
                    and promotion.get("experimentHash") == live_hash
                    and promotion.get("promotedBy") == "criterion"
                    and promotion.get("sweepRun") == expected_sweep
                    and isinstance(cell, dict)
                    and cell.get("layer") == expected_cell["layer"]
                    and float(cell.get("alpha", "nan"))
                    == expected_cell["alpha"]
                    and any(inj.get("concept") == concept
                            for inj in injections)):
                return os.path.join(run_dir, fname)
    return None


def _verify_agent_concrete(vc_name: str, artifact: dict, concept: str,
                           cell: dict | None, manifest: Manifest, *,
                           strict: bool = False) -> None:
    """The artifact must EMBODY its claimed identity (engineer review
    2026-07-18, fourth round): the birth certificate's winningCell is a
    claim — the concrete intervention must equal it. Exactly one injection
    for the concept, at exactly the winning cell's layer and alpha, on the
    study's model (and pinned revision, when both are stamped). ``strict``
    (ledger/resume pins, fifth round) additionally REQUIRES the artifact
    to state its base model — and its revision when the study pins one —
    rather than verifying them only when present."""
    base = artifact.get("baseModelID")
    if strict and not base:
        raise ValueError(
            f"variant '{vc_name}': the pinned agent states no baseModelID "
            "— an unattributed artifact cannot be a pinned arm")
    if base and base != manifest.model_id:
        raise ValueError(
            f"variant '{vc_name}': the promoted agent runs {base}, not the "
            f"study model {manifest.model_id}")
    revision = artifact.get("baseRevision")
    if strict and manifest.model_revision and not revision:
        raise ValueError(
            f"variant '{vc_name}': the study pins revision "
            f"{manifest.model_revision} but the pinned agent states none")
    if (revision and manifest.model_revision
            and revision != manifest.model_revision):
        raise ValueError(
            f"variant '{vc_name}': the promoted agent pins revision "
            f"{revision}, not the study's {manifest.model_revision}")
    matches = [inj for inj in (artifact.get("injections") or [])
               if inj.get("concept") == concept]
    if len(matches) != 1:
        raise ValueError(
            f"variant '{vc_name}': the promoted agent carries "
            f"{len(matches)} injections for '{concept}' — a criterion "
            "promotion mints exactly one")
    injection = matches[0]
    if cell is not None and (
            int(injection.get("layer", -1)) != int(cell["layer"])
            or float(injection.get("alpha", "nan")) != float(cell["alpha"])):
        raise ValueError(
            f"variant '{vc_name}': the agent's concrete injection "
            f"L{injection.get('layer')} α{injection.get('alpha')} does not "
            f"equal its claimed winning cell L{cell['layer']} "
            f"α{cell['alpha']:g} — the artifact does not embody its birth "
            "certificate")


def _concrete_from_pin(vc, concept: str, pin: dict, manifest: Manifest,
                       root: str | None, log,
                       source: str) -> tuple[VariantCondition, dict]:
    """A forward reference resolved from an EXACT pin — the pipeline
    ledger's promote record, or a prior run's forward-resolutions.json —
    never from ambient catalog state. Every pin field is REQUIRED (fifth
    round: a "full pin" with optional halves is a partial pin): path,
    hash, sweepRun, and a finite winningCell. The artifact bytes must
    still hash to the pin, live under the runs root, and the concrete
    intervention must equal the pinned winning cell."""
    import math as _math
    raw_path = str(pin.get("path") or pin.get("artifactPath") or "")
    pinned_hash = str(pin.get("hash") or pin.get("artifactHash") or "")
    if not raw_path or not pinned_hash:
        raise ValueError(
            f"variant '{vc.name}': the {source} pin for '{concept}' carries "
            "no path/hash — malformed record")
    if not str(pin.get("sweepRun") or "").strip():
        raise ValueError(
            f"variant '{vc.name}': the {source} pin for '{concept}' names "
            "no sweepRun — a pin without its selection identity is not a "
            "pin")
    cell_claim = pin.get("winningCell")
    try:
        cell_layer = int(cell_claim["layer"])
        cell_alpha = float(cell_claim["alpha"])
        if not _math.isfinite(cell_alpha):
            raise ValueError
    except (TypeError, KeyError, ValueError):
        raise ValueError(
            f"variant '{vc.name}': the {source} pin for '{concept}' has no "
            f"finite winningCell (got {cell_claim!r}) — layer/alpha "
            "verification cannot be skipped") from None
    artifact_path = raw_path if os.path.isabs(raw_path) else os.path.join(
        paths.project_root() if root is None else root, raw_path)
    runs_root = os.path.realpath(paths.runs_directory(root))
    if not os.path.realpath(artifact_path).startswith(runs_root + os.sep):
        raise ValueError(
            f"variant '{vc.name}': the {source} pin for '{concept}' points "
            f"outside the runs root ({raw_path}) — refusing to follow a "
            "tampered record")
    try:
        with open(artifact_path, "rb") as handle:
            blob = handle.read()
    except OSError as exc:
        raise ValueError(
            f"variant '{vc.name}': the {source}-pinned agent for "
            f"'{concept}' is gone ({raw_path}: {type(exc).__name__}) — the "
            "evidence chain is broken") from exc
    digest = hashlib.sha256(blob).hexdigest()
    if digest != pinned_hash:
        raise ValueError(
            f"variant '{vc.name}': the {source}-pinned agent for "
            f"'{concept}' changed since it was pinned (have "
            f"{digest[:12]}…, pinned {pinned_hash[:12]}…) — refusing a "
            "drifted arm")
    artifact = json.loads(blob.decode("utf-8"))
    _verify_agent_concrete(vc.name, artifact, concept,
                           {"layer": cell_layer, "alpha": cell_alpha},
                           manifest, strict=True)
    # The BIRTH CERTIFICATE must agree with the pin (sixth round): hash
    # equality proves the bytes are the pinned bytes, but if path and hash
    # were swapped TOGETHER, only the certificate's own claims — this
    # experiment, this epoch, criterion promotion, this sweep, this cell —
    # catch a substituted agent.
    promotion = artifact.get("promotion")
    if not isinstance(promotion, dict):
        raise ValueError(
            f"variant '{vc.name}': the pinned agent carries no promotion "
            "birth certificate — a hand-created artifact cannot fill a "
            "forward-referenced arm")
    expected_sweep = str(pin.get("sweepRun")).strip()
    cert_cell = promotion.get("winningCell") or {}
    checks = [
        (promotion.get("experiment") == manifest.name,
         f"names experiment '{promotion.get('experiment')}', not "
         f"'{manifest.name}'"),
        (promotion.get("promotedBy") == "criterion",
         f"was promoted by '{promotion.get('promotedBy')}', not by the "
         "declared criterion"),
        (str(promotion.get("sweepRun") or "") == expected_sweep,
         f"names sweep run '{promotion.get('sweepRun')}', not the pinned "
         f"'{expected_sweep}'"),
        (isinstance(cert_cell, dict)
         and cert_cell.get("layer") == cell_layer
         and float(cert_cell.get("alpha", "nan")) == cell_alpha,
         f"claims winning cell {cert_cell!r}, not the pinned "
         f"L{cell_layer} α{cell_alpha:g}"),
        (promotion.get("experimentHash") == manifest.content_hash(),
         "was minted under a different manifest epoch"),
    ]
    for passed, reason in checks:
        if not passed:
            raise ValueError(
                f"variant '{vc.name}': the pinned agent's birth "
                f"certificate {reason} — refusing a substituted arm")
    base = paths.project_root() if root is None else root
    relative = (os.path.relpath(artifact_path, base)
                if os.path.isabs(raw_path) else raw_path)
    log(f"variant '{vc.name}' ← {source}-pinned agent for '{concept}': "
        f"{relative} ({digest[:12]}…)")
    resolved = VariantCondition(
        name=vc.name, artifact_path=relative, artifact_hash=digest,
        artifact=artifact)
    provenance = {"condition": vc.name, "concept": concept,
                  "artifactPath": relative, "artifactHash": digest,
                  "resolvedFrom": source,
                  "experimentHash": manifest.content_hash()}
    for key in ("sweepRun", "winningCell"):
        if pin.get(key) is not None:
            provenance[key] = pin[key]
    return resolved, provenance


def _resolve_forward_variant(vc, manifest: Manifest, root: str | None,
                             log) -> tuple[VariantCondition, dict]:
    """Resolve a forward-referenced variant condition — "the agent this
    experiment's sweep promotes for CONCEPT under the declared criterion"
    (stage 4) — to a concrete artifact. The promotion birth certificate is
    the pin: the artifact must be a CRITERION promotion of the concept
    whose selection identity (sweep run + winning cell, from the
    manifest's current selection evidence) matches under the current
    manifest epoch. Raises ``ValueError`` naming the remedy when no such
    agent exists — the study never silently runs without a declared arm.
    Returns ``(concrete VariantCondition, resolution-provenance dict)``."""
    concept = str((vc.from_promotion or {}).get("concept") or "")
    if not concept:
        raise ValueError(
            f"variant '{vc.name}' fromPromotion names no concept")
    expected = _expected_promotion_identity(manifest.name, concept, root)
    if expected is None:
        raise ValueError(
            f"variant '{vc.name}' forward-references the promoted agent "
            f"for '{concept}', but no sweep selection evidence exists — "
            "sweep and promote first (the pipeline's sweep/promote stages "
            "do this in-chain)")
    live_hash = manifest.content_hash()
    artifact_path = _minted_agent_matching(
        manifest.name, concept, root, expected=expected,
        live_hash=live_hash)
    if artifact_path is None:
        raise ValueError(
            f"variant '{vc.name}': no promoted agent matches the current "
            f"selection for '{concept}' (sweep run {expected[0]}, cell "
            f"L{expected[1]['layer']} α{expected[1]['alpha']:g}) under "
            "this manifest epoch — run 'experiment promote' first")
    with open(artifact_path, "rb") as handle:
        blob = handle.read()
    digest = hashlib.sha256(blob).hexdigest()
    artifact = json.loads(blob.decode("utf-8"))
    _verify_agent_concrete(vc.name, artifact, concept, expected[1], manifest)
    base = paths.project_root() if root is None else root
    relative = os.path.relpath(artifact_path, base)
    log(f"variant '{vc.name}' ← promoted agent for '{concept}': "
        f"{relative} ({digest[:12]}…)")
    resolved = VariantCondition(
        name=vc.name, artifact_path=relative, artifact_hash=digest,
        artifact=artifact)
    provenance = {"condition": vc.name, "concept": concept,
                  "artifactPath": relative, "artifactHash": digest,
                  "resolvedFrom": "catalog",
                  "sweepRun": expected[0], "winningCell": expected[1],
                  "experimentHash": live_hash}
    return resolved, provenance


def _resolve_manifest_forward_refs(manifest: Manifest, run_directory: str,
                                   root: str | None, log, *,
                                   ledger_pins: dict | None = None) -> None:
    """Resolve every forward-referenced variant condition IN MEMORY at run
    start (after the run directory exists, before any condition executes)
    and record the resolutions as run evidence
    (``forward-resolutions.json``). The manifest FILE is never touched —
    a frozen manifest stays frozen; the run directory carries the pins.
    Raises if any reference cannot resolve (a missing arm is a refusal,
    never a silently smaller study).

    Resolution AUTHORITY, in order (engineer review 2026-07-18, fourth
    round):

    1. An existing ``forward-resolutions.json`` (RESUME): the record is
       authoritative — every artifact is re-verified against its recorded
       hash and reused. Fresh catalog resolution never runs on resume, so
       a mid-run requeue can never silently switch agents while the
       evidence file still names the old one.
    2. ``ledger_pins`` (the CHAIN): the exact ``{path, hash, sweepRun,
       winningCell}`` this pipeline's promote stage recorded — never
       ambient catalog state, which a concurrent sweep could change.
    3. Catalog scan by promotion birth certificate (standalone runs).

    Every path verifies the artifact CONCRETELY: bytes hash to the pin,
    and the injection equals the claimed winning cell on the study model.
    """
    if not any(vc.from_promotion for vc in manifest.variant_conditions):
        return
    record_path = os.path.join(run_directory, "forward-resolutions.json")
    if os.path.exists(record_path):
        with open(record_path, encoding="utf-8") as handle:
            record = json.load(handle)
        # Structural fail-closed (fifth round): hash verification protects
        # the artifact bytes; these checks protect the PROVENANCE — a
        # malformed or foreign record must never quietly steer resolution.
        if not isinstance(record, dict) or record.get("schema") != 1:
            raise ValueError(
                "forward-resolutions.json is malformed (schema != 1) — "
                "the resume evidence is unreadable")
        if record.get("experiment") != manifest.name:
            raise ValueError(
                "forward-resolutions.json belongs to experiment "
                f"'{record.get('experiment')}', not '{manifest.name}'")
        rows = record.get("resolutions")
        if not isinstance(rows, list) or not all(
                isinstance(r, dict) for r in rows):
            raise ValueError(
                "forward-resolutions.json resolutions are malformed")
        names = [str(r.get("condition")) for r in rows]
        if len(set(names)) != len(names):
            raise ValueError(
                "forward-resolutions.json carries duplicate condition "
                "rows — the resume evidence is inconsistent")
        by_condition = {str(r.get("condition")): r for r in rows}
        resolved_list: list[VariantCondition] = []
        for vc in manifest.variant_conditions:
            if not vc.from_promotion:
                resolved_list.append(vc)
                continue
            concept = str(vc.from_promotion.get("concept") or "")
            pin = by_condition.get(vc.name)
            if not isinstance(pin, dict):
                raise ValueError(
                    f"variant '{vc.name}': this run's "
                    "forward-resolutions.json carries no record for it — "
                    "the resume evidence is inconsistent")
            if str(pin.get("concept") or "") != concept:
                raise ValueError(
                    f"variant '{vc.name}': the resume record resolves "
                    f"concept '{pin.get('concept')}' but the manifest "
                    f"declares '{concept}' — the resume evidence is "
                    "inconsistent")
            resolved, _ = _concrete_from_pin(
                vc, concept, pin, manifest, root, log,
                source="resume-record")
            resolved_list.append(resolved)
        manifest.variant_conditions = resolved_list
        return
    resolved_list = []
    resolutions: list[dict] = []
    for vc in manifest.variant_conditions:
        if vc.from_promotion:
            concept = str(vc.from_promotion.get("concept") or "")
            if ledger_pins is not None:
                # CHAIN mode is fail-closed (fifth round): the ledger is
                # the authority, so an absent or malformed pin is an
                # inconsistency, never a license to consult ambient
                # catalog state.
                pin = ledger_pins.get(concept)
                if not isinstance(pin, dict):
                    raise ValueError(
                        f"variant '{vc.name}': the pipeline ledger carries "
                        f"no promote pin for '{concept}' — the chain is "
                        "inconsistent (is 'promote' in the stage list?)")
                resolved, provenance = _concrete_from_pin(
                    vc, concept, pin, manifest, root, log, source="ledger")
            else:
                # Standalone run: catalog resolution by the promotion
                # birth certificate.
                resolved, provenance = _resolve_forward_variant(
                    vc, manifest, root, log)
            resolved_list.append(resolved)
            resolutions.append(provenance)
        else:
            resolved_list.append(vc)
    manifest.variant_conditions = resolved_list
    with open(record_path, "w", encoding="utf-8") as handle:
        json.dump({"schema": 1, "experiment": manifest.name,
                   "resolutions": resolutions}, handle, indent=2,
                  sort_keys=True)


def pipeline(name: str, root: str | None = None, dtype: str = "auto",
             device: str | None = None, *, model_provider=None,
             should_cancel: Callable[[], bool] | None = None, log=None,
             checkpoint: "resume_mod.CheckpointFlag | None" = None,
             pipeline_run_directory: str | None = None,
             on_pipeline_directory: Callable[[str], None] | None = None,
             model_release=None) -> str:
    """The chain runner: one submission runs the manifest's declared stage
    list (default ``extract → validate → sweep → promote → run``) with ONE
    model load, evaluating declared gates between stages.

    - **Gates**: manifest DATA (``pipeline.gates``), evaluated as pure
      functions over the produced artifacts (:mod:`pipeline_spec`). A gate
      failure writes ``pipeline-abort.json`` and sets ``disposition:
      "aborted"`` — a successful scientific determination, NOT a job
      failure: the process exits 0 and a requeue returns idempotently.
    - **One model load**: the pinned model is acquired once and every GPU
      stage's existing task receives a provider that re-yields the held
      object. A stage requesting a DIFFERENT model (e.g. a different-model
      local judge) refuses loudly. A remainder with only CPU stages never
      loads the model. The chain's held model is LOCKED for the chain's
      duration, so the evaluate stage's release seam (``model_release``,
      2026-08-28) can never take it away mid-chain — that seam's work here
      is freeing containers the chain did NOT load (a leftover interactive
      model) before a judge column starts, and the generation→judging
      question is answered by ``_pipeline_needs_model`` over the stages
      that remain AFTER evaluate.
    - **Requeue/resume**: ``pipeline.json`` records completed stages and
      their run dirs; ``pipeline_run_directory`` reopens it, skipping
      completed stages (an interrupted extract/validate/sweep re-runs from
      scratch into a FRESH stage run dir — immutability), and the ``run``
      stage resumes RECORD-level through its own checkpoint machinery
      (``CheckpointRequested`` propagates to the exit-85 path). Resuming
      refuses if the manifest drifted since the last completed stage
      (epoch guard).
    - **Promote**: per-concept, recorded per concept in the ledger so a
      crash mid-promote never re-mints finished concepts. A concept whose
      sweep selected no cell aborts the chain (the sweep gate makes that
      explicit and earlier).
    """
    from . import pipeline_spec as pspec
    from . import promote as promote_lib
    _log = log or print
    manifest = Manifest.load(name, root)
    _verify_or_warn(manifest, root)
    raw_block = manifest.raw.get("pipeline")
    if raw_block is None:
        # The chain is preregistered DATA, not a default (engineer review
        # 2026-07-18, second round): an implicit five-stage chain with no
        # gates is a footgun, and stage 5's abort UI does not exist yet.
        raise RuntimeError(
            f"experiment '{name}' declares no pipeline block — add "
            '"pipeline": {"stages": [...], "gates": {...}} to the manifest '
            "(gates optional but strongly recommended) so the chain is a "
            "declared object, then re-run")
    spec = pspec.resolve_pipeline(raw_block)
    if spec.validate_gate is None and spec.sweep_gate is None:
        # Gates exist for validate and sweep only (pipeline_spec's gate
        # vocabulary). A chain that contains neither stage HAS no gate to
        # declare — calm information, not a warning (aligned with the
        # app's 2026-07-21 copy); the warning stays for chains where a
        # gateable stage runs ungated.
        if any(stage in ("validate", "sweep") for stage in spec.stages):
            _log("WARNING: pipeline declares no gates — every stage will run "
                 "to completion with no scientific stop conditions; declare "
                 "pipeline.gates for evidence-grade chains")
        else:
            _log("pipeline declares no gates — a "
                 + " → ".join(spec.stages)
                 + " chain has none to declare")

    # Resume-side checks FIRST (engineer review 2026-07-18, second round):
    # a doomed resume must refuse BEFORE minutes of model staging, and a
    # completed/aborted chain must return before any preflight runs.
    ledger: dict | None = None
    if pipeline_run_directory is not None:
        ledger = _read_pipeline_ledger(pipeline_run_directory)
        if ledger.get("experiment") != name:
            raise ValueError(
                f"'{pipeline_run_directory}' belongs to experiment "
                f"'{ledger.get('experiment')}', not '{name}'")
        if ledger.get("disposition"):
            _log(f"pipeline already {ledger['disposition']} → "
                 f"{pipeline_run_directory} (idempotent)")
            return pipeline_run_directory
        if list(spec.stages) != ledger.get("stages"):
            raise ValueError(
                "pipeline resume refused: the declared stage list changed "
                "since this pipeline started — start a fresh pipeline")
        # Epoch guard, PRE-model. The commonest "drift" here is SELF-INFLICTED
        # (a 2026-08-05 replication run): the run stage pins the resolved
        # model revision into the live manifest, but every `bundle execute`
        # re-imports the bundle's UNPINNED manifest with allow_overwrite — the
        # continuation job clobbers the pin moments before reading the ledger,
        # then refuses the epoch the chain itself created, after hours of GPU
        # work. When the live manifest differs from the ledger ONLY by the
        # missing revision pin, restore the pin and continue.
        live_hash = manifest.content_hash()
        if ledger.get("experimentHash") != live_hash:
            manifest, live_hash, restored = _restore_self_pinned_revision(
                name, manifest, ledger, root, _log,
                pipeline_run_directory=pipeline_run_directory)
            if not restored and ledger.get("experimentHash") != live_hash:
                # Real drift. POLICY (2026-08-05, Christian): a submitted
                # chain must never die on pinning drift — the completed
                # stages' evidence is intact and the remaining stages carry
                # their own per-source epoch guards (measurement-side drift
                # tolerated and stamped; generation-side drift refuses only
                # the cheap continuation stage, never the run). Continue
                # LOUDLY and stamp the ledger so the drift is a checkable
                # fact, not a memory.
                _log("WARNING: the manifest drifted since the last completed "
                     f"stage (ledger {str(ledger.get('experimentHash'))[:12]}…, "
                     f"live {live_hash[:12]}…) — continuing under the LIVE "
                     "manifest; the drift is stamped epochDriftAtContinuation "
                     "in the pipeline ledger, and each remaining stage's own "
                     "epoch guard decides what it may measure")
                _stamp_pipeline_drift(pipeline_run_directory, ledger,
                                      live_hash)
        if ledger.pop("parked", None) is not None:
            # A parked chain that is being resumed is no longer parked —
            # the stamp must not outlive the state it describes
            # (startup-reconcile orphan handling, 2026-08-06).
            _write_pipeline_ledger(pipeline_run_directory, ledger)
            _log("pipeline resume: cleared the parked stamp — the chain "
                 "is live again")

    completed = (ledger or {}).get("stageResults") or {}
    remaining = [s for s in spec.stages
                 if (completed.get(s) or {}).get("status") != "completed"]
    # Preflight ONLY the remaining stages: a resumed chain whose judged
    # sweep already completed must not refuse for a since-cleared key.
    _pipeline_inline_judging_preflight(manifest, remaining)
    # OpenRouter judge pins are checked at CHAIN start, pre-model
    # (2026-08-04): the sweep/evaluate stages preflight too, but inside the
    # chain they fire only after the model load — and evaluate's fires after
    # the run stage has already burned its GPU hours. A judge model the
    # catalogue positively does not list must refuse before stage 1
    # (`test-compare-2`: 240 records generated, then the first judge call
    # 404'd on a judge nobody could ever have reached).
    if _pipeline_will_judge(manifest, remaining) and manifest.judges:
        _preflight_openrouter_judges(manifest.judges, _log)
    needs_model = _pipeline_needs_model(remaining, manifest)

    from contextlib import nullcontext
    context = (_acquire_model(manifest, dtype, device, model_provider)
               if needs_model else nullcontext(None))
    with context as model:
        if model is not None:
            manifest = _pin_model_revision(name, manifest, model, root, _log)

            @contextmanager
            def _held(model_id, revision=None, dtype=None):
                if model_id != manifest.model_id:
                    raise RuntimeError(
                        f"the pipeline holds '{manifest.model_id}' but a "
                        f"stage requested '{model_id}' — the chain runs one "
                        "model load; use the study model or run that stage "
                        "outside the chain")
                # `dtype` is accepted and ignored ON PURPOSE. A stage
                # acquires through `_acquire_model`, which forwards the
                # manifest's pinned dtype as a keyword whenever one is
                # pinned — so a dtype-pinned study reaches here with the
                # keyword set. The chain already loaded ONE model through
                # that same helper, which ran `_assert_resident_dtype_matches`
                # against this same manifest; a stage therefore cannot ask
                # for a precision the held model does not already have, and
                # re-checking here would be unreachable code.
                # Without this parameter a dtype-pinned study could not run
                # as a pipeline at all: the first stage raised TypeError
                # before touching any data (cluster, 2026-07-26).
                yield model

            stage_provider = _held
        else:
            stage_provider = None

        if ledger is None:
            # Fresh chain: the ledger hash is stamped AFTER revision pinning
            # (above, inside the model context) so it stays stable across
            # requeues. It then advances stage by stage — the sweep
            # legitimately appends recommended conditions mid-chain.
            live_hash = Manifest.load(name, root).content_hash()
            run_directory = paths.make_unique_run_directory(
                f"exp-{name}-pipeline", root)
            _write_config_snapshot(manifest, run_directory, "pipeline")
            ledger = {"schema": PIPELINE_LEDGER_SCHEMA, "experiment": name,
                      "experimentHash": live_hash,
                      # Draft chains are legal (exploratory); the stamp
                      # makes the distinction durable provenance, and the
                      # app labels Run Draft Pipeline vs Run Frozen
                      # Pipeline from it (seventh round).
                      "manifestStatus": manifest.status,
                      "stages": list(spec.stages), "stageResults": {},
                      "disposition": None}
            _write_pipeline_ledger(run_directory, ledger)
        else:
            run_directory = pipeline_run_directory
        if on_pipeline_directory is not None:
            on_pipeline_directory(run_directory)

        def _stage_done(stage: str, stage_dir: str | None,
                        extra: dict | None = None) -> None:
            entry = {"status": "completed"}
            if stage_dir is not None:
                entry["runDirectory"] = stage_dir
            if extra:
                entry.update(extra)
            ledger["stageResults"][stage] = entry
            ledger["experimentHash"] = Manifest.load(name, root).content_hash()
            _write_pipeline_ledger(run_directory, ledger)

        def _abort(stage: str, failures: list, evidence_dir: str | None) -> str:
            record = {
                "schema": 1, "experiment": name,
                "experimentHash": ledger["experimentHash"],
                "stage": stage,
                "gates": [f.to_dict() for f in failures],
                "evidenceRunDirectory": evidence_dir,
            }
            with open(os.path.join(run_directory, "pipeline-abort.json"),
                      "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
            ledger["disposition"] = "aborted"
            ledger["abort"] = record
            _write_pipeline_ledger(run_directory, ledger)
            for failure in failures:
                _log(f"pipeline gate failed [{stage}]: {failure.detail}")
            _log(f"pipeline ABORTED at '{stage}' — a scientific "
                 "determination, recorded in pipeline-abort.json; nothing "
                 f"after '{stage}' ran")
            return run_directory

        for stage in spec.stages:
            prior = ledger["stageResults"].get(stage) or {}
            if prior.get("status") == "completed":
                _log(f"pipeline: '{stage}' already completed "
                     f"({prior.get('runDirectory', 'no dir')}) — skipping")
                continue
            if _observe_cancel(should_cancel, _log,
                               f"pipeline before '{stage}'"):
                return run_directory
            _log(f"pipeline: stage '{stage}' starting")
            failures: list = []
            stage_dir: str | None = None

            if stage == "extract":
                stage_dir = extract(name, root, dtype, device,
                                    model_provider=stage_provider,
                                    should_cancel=should_cancel, log=_log)
            elif stage == "validate":
                stage_dir = validate(name, root, dtype, device,
                                     model_provider=stage_provider,
                                     should_cancel=should_cancel, log=_log)
                if spec.validate_gate is not None:
                    with open(os.path.join(stage_dir,
                                           "validation-report.json"),
                              encoding="utf-8") as handle:
                        report = json.load(handle)
                    with open(os.path.join(stage_dir, "cosine-matrix.csv"),
                              encoding="utf-8", newline="") as handle:
                        cosine_rows = list(csv.reader(handle))
                    # One matrix per declared validation depth: the cap is
                    # applied to every one (distinctness must hold at every
                    # depth the study declared it would measure).
                    extra_matrices = []
                    for extra_name in sorted(os.listdir(stage_dir)):
                        if extra_name.startswith("cosine-matrix-L") and \
                                extra_name.endswith(".csv"):
                            with open(os.path.join(stage_dir, extra_name),
                                      encoding="utf-8", newline="") as handle:
                                extra_matrices.append(
                                    (extra_name, list(csv.reader(handle))))
                    concepts = [c.name for c in
                                Manifest.load(name, root).concepts]
                    failures = [r for r in pspec.evaluate_validate_gate(
                        spec.validate_gate, concepts, report, cosine_rows,
                        extra_cosine_matrices=extra_matrices)
                        if not r.passed]
            elif stage == "sweep":
                stage_dir = sweep(
                    name, root, dtype, device,
                    model_provider=stage_provider,
                    max_loaded=(1 if stage_provider is not None else None),
                    should_cancel=should_cancel, log=_log)
                if spec.sweep_gate is not None:
                    with open(os.path.join(stage_dir, "recommendations.json"),
                              encoding="utf-8") as handle:
                        recommendations = json.load(handle)
                    concepts = [c.name for c in
                                Manifest.load(name, root).concepts]
                    failures = [r for r in pspec.evaluate_sweep_gate(
                        spec.sweep_gate, concepts, recommendations)
                        if not r.passed]
            elif stage == "promote":
                # ADOPTION is recovery-only (engineer review 2026-07-18,
                # third round): a fresh chain must always MINT — cross-
                # pipeline dedup by epoch alone would adopt an earlier
                # chain's agent on a frozen manifest. The stage is marked
                # started BEFORE the first mint so a crash anywhere in the
                # save→record window is recognizable as recovery.
                recovering = prior.get("status") == "started"
                minted = dict(prior.get("concepts") or {})
                ledger["stageResults"][stage] = {
                    "status": "started", "concepts": minted}
                _write_pipeline_ledger(run_directory, ledger)
                concepts = [c.name for c in Manifest.load(name, root).concepts]
                for concept in concepts:
                    if concept in minted:
                        _log(f"pipeline: '{concept}' already promoted "
                             f"({minted[concept]}) — skipping")
                        continue
                    # Crash window closure (save_variant → ledger write):
                    # during RECOVERY, adopt the agent whose full selection
                    # identity (epoch + sweep run + winning cell) matches
                    # what a promotion would stamp right now.
                    existing = (_find_minted_agent(name, concept, ledger,
                                                   root)
                                if recovering else None)
                    if existing is not None:
                        _log(f"pipeline: '{concept}' agent already minted "
                             f"by this chain's selection ({existing}) — "
                             "adopting")
                        with open(existing, "rb") as handle:
                            blob = handle.read()
                        adopted = json.loads(blob.decode("utf-8"))
                        promotion = adopted.get("promotion") or {}
                        minted[concept] = {
                            "path": existing,
                            "hash": hashlib.sha256(blob).hexdigest(),
                            "sweepRun": promotion.get("sweepRun"),
                            "winningCell": promotion.get("winningCell")}
                        ledger["stageResults"][stage] = {
                            "status": "started", "concepts": minted}
                        _write_pipeline_ledger(run_directory, ledger)
                        continue
                    # The chain promotes ITS OWN sweep's winner: the ledger's
                    # sweep run is the ONLY evidence source (a frozen
                    # manifest can carry a stale -recommended condition
                    # forever, and "newest run" is ambient state).
                    chain_sweep_dir = (ledger["stageResults"].get("sweep")
                                       or {}).get("runDirectory")
                    chain_sweep_run = (os.path.basename(chain_sweep_dir)
                                       if chain_sweep_dir else None)
                    # Under the PINNED contract (B2) when the chain has its
                    # own sweep: the epoch guard fires if the sweep belongs
                    # to a different manifest epoch, and the promotion key
                    # makes a retried stage return the existing agent
                    # instead of minting a competing duplicate.
                    pins = (promote_lib.PromotionPins(
                        sweep_run=chain_sweep_run,
                        experiment_hash=ledger.get("experimentHash"))
                        if chain_sweep_run else None)
                    try:
                        outcome = promote_lib.promote(
                            name, concept, root=root, log=_log,
                            sweep_run=None if pins else chain_sweep_run,
                            pins=pins)
                    except promote_lib.PromoteError as exc:
                        # The sweep selected no cell (or evidence is
                        # missing): an explicit scientific stop with the
                        # promote refusal as the gate detail — never a
                        # stack trace masquerading as a job failure.
                        ledger["stageResults"][stage] = {
                            "status": "started", "concepts": minted}
                        _write_pipeline_ledger(run_directory, ledger)
                        return _abort(stage, [pspec.GateResult(
                            passed=False, stage=stage, gate="promotable",
                            detail=str(exc))], None)
                    promotion = (outcome.get("variant") or {}).get(
                        "promotion") or {}
                    # The FULL pin — path, content hash, and selection
                    # identity — so the run stage uses THIS exact agent,
                    # never ambient catalog state a later sweep could shift.
                    minted[concept] = {
                        "path": outcome["path"], "hash": outcome["hash"],
                        "sweepRun": promotion.get("sweepRun"),
                        "winningCell": promotion.get("winningCell")}
                    # Per-concept crash safety: a requeue re-mints nothing.
                    ledger["stageResults"][stage] = {
                        "status": "started", "concepts": minted}
                    _write_pipeline_ledger(run_directory, ledger)
                _stage_done(stage, None, {"concepts": minted})
                continue
            elif stage == "run":
                recorded = prior.get("runDirectory")
                resume_dir = None
                if recorded and os.path.isdir(recorded):
                    if resume_mod.is_complete(recorded):
                        # Crash between run completion and the ledger
                        # update: adopt, don't re-run.
                        _stage_done(stage, recorded)
                        continue
                    if resume_mod.is_resumable(recorded):
                        resume_dir = recorded

                def _record_run_dir(created: str) -> None:
                    ledger["stageResults"]["run"] = {
                        "status": "started", "runDirectory": created}
                    _write_pipeline_ledger(run_directory, ledger)

                # The exact agents THIS chain's promote stage minted are
                # the run's forward-reference pins — never ambient catalog
                # state a concurrent sweep could change. The dict is passed
                # even when EMPTY (fifth round): inside a chain, a forward
                # reference without a ledger pin is an inconsistency that
                # refuses, never a license to fall back to the catalog.
                promote_pins = {
                    concept: pin
                    for concept, pin in ((ledger["stageResults"].get(
                        "promote") or {}).get("concepts") or {}).items()
                    if isinstance(pin, dict)}
                stage_dir = run(name, None, root, dtype, device,
                                model_provider=stage_provider,
                                should_cancel=should_cancel, log=_log,
                                checkpoint=checkpoint,
                                run_directory=resume_dir,
                                on_run_directory=_record_run_dir,
                                forward_resolutions=promote_pins)
            elif stage in ("evaluate", "analyze"):
                # The chain's OWN run is the only legal source (the
                # resolver refused evaluate/analyze without run), and
                # evaluate/analyze take the source as a DIRECTORY PATH —
                # a basename would resolve against the process cwd.
                source = (ledger["stageResults"].get("run") or {}).get(
                    "runDirectory")
                if not source:
                    raise RuntimeError(
                        f"pipeline stage '{stage}' has no recorded run "
                        "stage directory — the ledger is inconsistent")
                if stage == "evaluate":
                    # A resumed chain whose evaluate emitted fan-out packets
                    # earlier: adopt the completed judgment run (the
                    # controller merge, or a Mac judging session, produced
                    # it); still waiting → refuse loudly rather than
                    # re-emitting a second packet set.
                    if prior.get("status") == "awaitingJudgment":
                        awaiting = str(prior.get("runDirectory") or "")
                        found = _find_evaluate_judgment_run(
                            name, os.path.basename(awaiting), root)
                        if found is None:
                            raise RuntimeError(
                                "pipeline resume: the evaluate stage is "
                                "awaiting judgments for "
                                f"'{os.path.basename(awaiting)}' — the "
                                "judge fan-out (or a Mac judging session) "
                                "has not completed them yet; resume after "
                                "the merge")
                        stage_dir = found[0]
                        _log("pipeline: adopting completed judgment run "
                             f"→ {stage_dir}")
                        _stage_done(stage, stage_dir,
                                    {"judgedVia": "deferredJudgment"})
                        continue
                    # The generation→judging seam's conservative half
                    # (2026-08-28): does anything AFTER evaluate still hold
                    # the GPU model? On today's stage vocabulary nothing
                    # after evaluate generates (analyze is CPU-side), but
                    # the question is asked of the spec rather than assumed.
                    after_evaluate = spec.stages[
                        spec.stages.index(stage) + 1:]
                    study_model_generates_later = _pipeline_needs_model(
                        after_evaluate, manifest)
                    # Local judges needing models OTHER than the held study
                    # model: the chain cannot judge them inline (one model
                    # load) — EMIT blinded packets and STOP; the controller
                    # fans one worker job per distinct judge model and
                    # merges, then the final continuation resumes here
                    # (2026-07-23; the commit-1 refusal is replaced by this
                    # routing).
                    if evaluate_fanout_judge_models(manifest):
                        stage_dir = evaluate(
                            name, root, source_run=source,
                            model_provider=stage_provider,
                            should_cancel=should_cancel, log=_log,
                            defer_local_judges=True,
                            model_release=model_release,
                            study_model_generates_later=(
                                study_model_generates_later))
                        ledger["stageResults"][stage] = {
                            "status": "awaitingJudgment",
                            "runDirectory": stage_dir}
                        _write_pipeline_ledger(run_directory, ledger)
                        request = write_judge_fanout_request(
                            run_directory, name, manifest, stage_dir)
                        _log("pipeline: evaluate emitted "
                             f"{request.get('packetCount')} blinded "
                             "packet(s) for the judge fan-out ("
                             + ", ".join(m["model"] for m
                                         in request["judgeModels"])
                             + ") — the chain stops here; judge workers "
                             "and the merge continue it")
                        return run_directory
                    stage_dir = evaluate(
                        name, root, source_run=source,
                        model_provider=stage_provider,
                        should_cancel=should_cancel, log=_log,
                        model_release=model_release,
                        study_model_generates_later=(
                            study_model_generates_later))
                else:
                    stage_dir = analyze(
                        name, root, source_run=source,
                        should_cancel=should_cancel, log=_log)

            _stage_done(stage, stage_dir)
            if failures:
                return _abort(stage, failures, stage_dir)
            _log(f"pipeline: stage '{stage}' completed"
                 + (f" → {stage_dir}" if stage_dir else ""))

        ledger["disposition"] = "completed"
        _write_pipeline_ledger(run_directory, ledger)
        # A resumed pipeline may be completing a directory whose FIRST
        # attempt left a failed run-status.json and FAILED.md; the ledger is
        # the authority and now says completed, so the failure-era record is
        # rewritten to match reality (history preserved as supersededError).
        if heal_after_completion(run_directory):
            _log("pipeline: healed failure-era status artifacts "
                 "(run-status.json / FAILED.md) left by a prior failed "
                 "attempt of this now-completed pipeline")
        _log(f"pipeline completed ({len(spec.stages)} stage(s)) → "
             f"{run_directory}")
        return run_directory


def _judge_roster(manifest: Manifest, spec) -> list[JudgeRef]:
    """The manifest's pinned judge panel, else the legacy single judge derived
    from ``evaluation.judgeModel`` (draft convenience — freeze requires >= 2)."""
    from . import paired_judge
    if manifest.judges:
        return list(manifest.judges)
    model = spec.judge_model or paired_judge.DEFAULT_JUDGE_MODEL
    kind = "claude" if paired_judge.is_claude_model(model) else "local"
    return [JudgeRef(name=model, kind=kind, model=model)]


def judge_models_still_needed(remaining_roster, *, study_model: str,
                              study_model_generates_later: bool) -> set[str]:
    """The models the REMAINDER of a judged run still needs.

    The still-needed rule, exactly (maintainer's ruling, 2026-08-28: "any
    runs that require two models will need to unload and load models in
    order not to OOM. We need to ensure this happens"):

        still-needed = {resolved model of every LOCAL judge in
                        ``remaining_roster``}
                       ∪ ({study model} if a later stage of this run or
                          pipeline GENERATES)

    ``remaining_roster`` is ``roster[i:]`` at the boundary before judge
    ``i`` — so the judge about to load is itself in the set, which is what
    keeps two consecutive same-model columns warm instead of
    releasing-and-reloading the very weights the next column needs.
    External judges (claude/openrouter) hold no device memory and
    contribute nothing. A local judge with an empty model resolves to the
    study model by the cross-engine rule, so a study-model judge keeps the
    study model resident without any special case.

    ``study_model_generates_later`` is the conservative half: evaluate is
    terminal for generation on its own (analyze/rescore are CPU-side), but
    a caller that cannot PROVE the study model is finished passes True and
    the model is kept — the capacity gate then speaks, as before.
    """
    from . import sweep_selection
    needed = {
        sweep_selection.resolve_local_judge_model(ref.model, study_model)
        for ref in remaining_roster if ref.kind == "local"}
    if study_model_generates_later:
        needed.add(study_model)
    return needed


def _release_models_for_judge(model_release, roster, index: int, *,
                              study_model: str,
                              study_model_generates_later: bool,
                              _log) -> None:
    """The model-slot release seam: free every container the remainder of
    this run will not use, BEFORE judge ``index``'s model loads.

    The guarantee this buys: a run whose models are needed SEQUENTIALLY
    never fails for co-residency. Peak device memory becomes the MAX of any
    one still-needed model instead of the SUM of the panel's weights — the
    two-judge calibration on 2026-08-28 refused at ~22.7 GiB needed against
    23.3 GiB free on an 80 GiB A100 purely because the FINISHED judge's
    container was still resident and nothing in the run would ever use it
    again.

    Candidates are this run's OWN models — the study model and every local
    judge's resolved model — never an unrelated resident (a chat model, a
    variant-generate model): the release is a run seam, not a new global
    eviction heuristic, and the interactive cache policy is unchanged.
    Whatever ``judge_models_still_needed`` names is subtracted, so the seam
    is a no-op for a single-judge run and for a same-model column boundary.

    Deliberate cross-engine divergence (documented so neither side reads as
    an oversight): the Mac answers the same problem by REFUSING up front —
    ``SweepObjectives.localJudgeSlotProblem`` declines a two-model sweep
    before any work starts, which is right for MLX's memory model, where a
    unified-memory container is not cheaply reclaimable mid-run. CUDA can
    hand the weights back, so this engine releases BETWEEN columns instead
    of refusing.

    A failing release never fails the run: it is logged and the capacity
    gate remains the backstop.
    """
    from . import sweep_selection
    ref = roster[index]
    keep = judge_models_still_needed(
        roster[index:], study_model=study_model,
        study_model_generates_later=study_model_generates_later)
    candidates = {study_model} | {
        sweep_selection.resolve_local_judge_model(r.model, study_model)
        for r in roster if r.kind == "local"}
    stale = sorted(candidates - keep)
    next_model = (
        sweep_selection.resolve_local_judge_model(ref.model, study_model)
        if ref.kind == "local" else None)
    need_text = (f"next judge '{ref.name}' needs '{next_model}'"
                 if next_model else
                 f"next judge '{ref.name}' needs no local model")
    where = ("generation complete" if index == 0
             else f"column '{roster[index - 1].name}' complete")
    if model_release is not None and stale:
        try:
            released = model_release(stale) or []
        except Exception as exc:  # noqa: BLE001 - never fail a run on cleanup
            _log(f"WARNING: could not release model slot(s) "
                 f"{', '.join(stale)} before judge '{ref.name}' ({exc}) — "
                 "continuing; the load capacity gate remains the backstop")
            return
        for record in released:
            size = record.get("bytes")
            size_text = f" (~{size / (1 << 30):.1f} GiB)" if size else ""
            _log(f"released '{record['modelID']}'{size_text} from "
                 f"{record.get('device')} — {where}, {need_text}")
        return
    if model_release is None and index > 0:
        # CLI/bundle path (the Slurm path): there is no registry, so the
        # previous column's PRIVATE in-process copy is what has to go. Its
        # last reference was dropped when that column's ExitStack closed
        # (``_local_judge_generation`` registers the drop); the allocator
        # still holds the blocks until they are collected and trimmed, and
        # `cuda.mem_get_info` — what the capacity gate reads — counts them
        # as used until then.
        model_loader.free_device_memory()
        _log(f"released the private model copy of {where} — {need_text}")


def _judge_callable(ref: JudgeRef, model_provider, *, study_model: str,
                    study_revision: str | None = None, stack=None):
    """(judge_fn, requested_model, actual_model_holder) for one judge. Claude
    judges call the Anthropic API; local judges acquire the served model
    through ``model_provider`` (the registry slot lock — no forward pass ever
    shares a model object unlocked). The CLI path has no provider and
    synthesizes one that loads a private copy in-process.

    A LOCAL judge's model resolves by the cross-engine rule
    (``sweep_selection.resolve_local_judge_model``, unified for sweep AND
    evaluate 2026-07-22): the declared ``model`` when non-empty, else the
    STUDY model at its pinned revision — acquired through the normal
    provider/registry slot, so an already-resident study model is reused,
    never loaded twice. The judge's NAME is a label, never a model id (the
    dead ``model or name`` fallback sent 'judge-1' to HuggingFace as a model
    id on an offline compute node)."""
    from . import paired_judge, sweep_selection
    if ref.kind == "openrouter":
        model = (ref.model or "").strip()
        provider = (getattr(ref, "provider", None) or "").strip()
        if not model or not provider:
            raise RuntimeError(
                f"openrouter judge '{ref.name}' needs an explicit model "
                "slug AND a pinned provider — neither has a server default")
        return (paired_judge.make_openrouter_judge(model, provider),
                model, {"actual": model})
    if ref.kind != "local":
        model = ref.model or paired_judge.DEFAULT_JUDGE_MODEL
        return paired_judge.judge_pair, model, {"actual": model}
    gen, judge_model, holder = _local_judge_generation(
        ref, model_provider, study_model=study_model,
        study_revision=study_revision, stack=stack)
    return paired_judge.make_local_judge(gen), judge_model, holder


def _coder_callable(ref: JudgeRef, model_provider, *, study_model: str,
                    study_revision: str | None = None, stack=None):
    """``(complete_fn, requested_model, holder)`` for one judge on the
    per-response coding path — the raw-completion sibling of
    ``_judge_callable`` (same resolution rules, same slot machinery, same
    holder provenance); ``complete_fn(prompt) -> (text, provider|None)``."""
    from . import paired_judge
    if ref.kind == "openrouter":
        model = (ref.model or "").strip()
        provider = (getattr(ref, "provider", None) or "").strip()
        if not model or not provider:
            raise RuntimeError(
                f"openrouter judge '{ref.name}' needs an explicit model "
                "slug AND a pinned provider — neither has a server default")

        def _openrouter(prompt: str):
            return paired_judge.openrouter_complete(
                model, prompt, provider=provider)
        return _openrouter, model, {"actual": model}
    if ref.kind != "local":
        model = ref.model or paired_judge.DEFAULT_JUDGE_MODEL

        def _claude(prompt: str):
            return paired_judge.claude_complete(model, prompt), None
        return _claude, model, {"actual": model}
    gen, judge_model, holder = _local_judge_generation(
        ref, model_provider, study_model=study_model,
        study_revision=study_revision, stack=stack)

    def _local(prompt: str):
        return gen(prompt), None
    return _local, judge_model, holder


def _local_judge_generation(ref: JudgeRef, model_provider, *,
                            study_model: str,
                            study_revision: str | None = None, stack=None):
    """``(gen_fn, judge_model, holder)`` for one LOCAL judge — the shared
    model-resolution/slot core of ``_judge_callable`` and
    ``_coder_callable`` (extracted 2026-08-04 for the coding instrument;
    behavior unchanged)."""
    from . import paired_judge, sweep_selection
    judge_model = sweep_selection.resolve_local_judge_model(
        ref.model, study_model)
    # The judge's own pinned revision wins (JudgeRef.revision, 2026-07-23);
    # a study-model judge falls back to the study pin — the same bytes that
    # generated the outputs.
    judge_revision = (getattr(ref, "revision", None)
                      or (study_revision if judge_model == study_model
                          else None))
    # The judge's declared dtype is a PIN, not a hint (external review round
    # 3, finding 2): freeze now requires it for a foreign local judge, and a
    # manifest that claims a dtype the load ignored is a false pin. Passed
    # only when declared, so providers that predate the parameter keep
    # working for judges that declare none.
    judge_dtype = (getattr(ref, "dtype", None) or "").strip() or None
    provider = model_provider
    if provider is None:
        @contextmanager
        def _cli_provider(model_id, revision=None, dtype=None):
            yield model_loader.load(model_id, revision=revision, dtype=dtype)
        provider = _cli_provider
    holder = {"actual": judge_model, "revision": judge_revision,
              "requestedDtype": judge_dtype}
    held: dict = {}

    @contextmanager
    def _slot():
        """The judge's model, loaded ONCE per column when the caller
        supplies an ``ExitStack`` (external review round 3, finding 3c).

        ``model_loader.load`` has no cache, and this context used to be
        entered inside the per-pair generation call — so on the CLI/bundle
        path (which is the Slurm path) a foreign local judge reloaded on
        EVERY judgment. A 12B judge across a hundred pairs is a hundred
        full weight loads: not slow, a walltime kill that presents as a
        hang.

        Without a stack the behaviour is exactly as before (acquire and
        release per call), so callers opt in rather than silently having
        their slot-holding semantics changed underneath them.
        """
        args = (judge_model, judge_revision)
        kwargs = {"dtype": judge_dtype} if judge_dtype else {}
        if stack is None:
            with provider(*args, **kwargs) as slot:
                yield slot
            return
        if "slot" not in held:
            held["slot"] = stack.enter_context(provider(*args, **kwargs))
            # Drop the column's reference when the stack closes, BEFORE the
            # provider context exits (callbacks unwind LIFO). Without this
            # the CLI/bundle path's private copy stayed pinned by this
            # closure until the NEXT column's callable replaced it — which
            # happens AFTER the next model has loaded, i.e. exactly the
            # co-residency the column boundary exists to prevent
            # (2026-08-28). The registry path is unaffected: there the
            # container's owner is the slot, not this reference.
            stack.callback(held.pop, "slot", None)
        yield held["slot"]

    def _gen(prompt: str) -> str:
        try:
            with _slot() as slot:
                holder["actual"] = slot.model_id
                # The dtype the model ACTUALLY runs in, read off the loaded
                # parameters — not the dtype the manifest asked for
                # (external review round 4, finding 2). Recording only the
                # request let an artifact claim a dtype the load never
                # honored. Stamped for EVERY local judge, including one
                # using the study model, because evaluate may run in a
                # separate job on a different device.
                actual_dtype = getattr(slot, "dtype", None)
                if actual_dtype:
                    holder["actualDtype"] = str(actual_dtype)
                # JUDGE_MAX_TOKENS — the one cross-engine judge cap (the
                # 2026-07-22 incident: 512 truncated a legible verdict).
                return generate(slot, prompt, model_id=slot.model_id,
                                max_tokens=paired_judge.JUDGE_MAX_TOKENS,
                                temperature=0.0,
                                prompt_mode=prompt_render.CHAT_ASSISTANT)
        except model_loader.ModelLoadError as exc:
            if judge_model == study_model:
                raise
            # Name the judge — and when the loader's refusal already carries
            # complete advice (its own typed sentences: device capacity,
            # co-residency headroom, VRAM class), CARRY IT rather than
            # replace it. The old static "install the model on the server"
            # overwrote all of them, and when the real cause was
            # co-residency it sent an agent to install a model that was
            # already there (observed 2026-08-28, the two-judge
            # calibration). Raw hub/network dumps stay summarized away —
            # "check your internet connection" is misleading on an
            # air-gapped compute node. The Swift twin
            # (`localJudgeLoadFailureMessage`) legitimately keeps static
            # install advice: it decides on a PRESENCE check before the
            # loader is asked, so "not installed" is the one possible cause
            # there.
            if getattr(exc, "advice_complete", False):
                raise model_loader.ModelLoadError(
                    f"local judge '{ref.name}' declares model "
                    f"'{judge_model}', which could not be loaded on this "
                    f"server — {exc} (a local judge with an EMPTY model "
                    "judges with the study model)",
                    advice_complete=True) from exc
            raise model_loader.ModelLoadError(
                f"local judge '{ref.name}' declares model '{judge_model}', "
                "which could not be loaded on this server — install the "
                "model on the server, or leave the judge's model empty to "
                "judge with the study model") from exc
    return _gen, judge_model, holder


def _judgment_key(row: dict) -> tuple[str, str, str]:
    """Pair-cell identity for aligning outcomes across judges (and against a
    human-labeled subset): the same id-keyed shape judge output rows carry.
    Keyed on ``sampleIndex`` (absent normalizes to 0) — the pairing join key
    — never the seed, which differs between the two sides of a pair under
    derived seeding."""
    return (str(row.get("promptID")), str(row.get("sampleIndex") or 0),
            str(row.get("condition")))


def _verify_judgment_provider(row: dict, judge: str,
                              pinned_provider: str | None) -> str | None:
    """Per-judgment provider verification for BOTH completion verbs
    (engineer review 2026-07-18, provider-evidence pass): an openrouter
    judge's serving provider is emission-pinned exactly like its model, so
    every judgment must carry it and match it — a recorded string is
    provenance only if verified. A non-openrouter judgment claiming a
    provider is a mix-up, refused rather than shrugged past."""
    from . import paired_judge
    provider = str(row.get("provider") or "").strip()
    if pinned_provider is not None:
        if not provider:
            raise ValueError(
                f"judgment by {judge!r} carries no provider — the judging "
                "client must stamp the emission-pinned OpenRouter provider "
                "it verified against the response (update the app)")
        if (paired_judge.canonical_openrouter_provider(provider)
                != paired_judge.canonical_openrouter_provider(pinned_provider)):
            raise ValueError(
                f"judge {judge!r} judged via provider '{provider}' but the "
                f"emission pinned '{pinned_provider}' — refusing off-pin "
                "judgments")
        return paired_judge.canonical_openrouter_provider(provider)
    if provider:
        raise ValueError(
            f"judgment by {judge!r} carries provider '{provider}' but "
            "that judge is not openrouter-kind — only provider-pinned "
            "judges stamp one")
    return None


def _verified_judgment_payload(row: dict, judge: str,
                               winner: str) -> tuple[float | None,
                                                     dict | None]:
    """``(confidence, full verdict payload)`` from one client judgment row —
    the winner-only closure (2026-07-20, both completion verbs): deferred
    judgments may now carry the judge's FULL verdict (scores, structured
    fields, confidence, brief reason — the same object the inline path
    records as ``judgment``), so deferred artifacts stop being
    winner-only. The payload is OPTIONAL (older judging clients send
    winner-only rows; completion still succeeds and records
    ``judgment: null``), but a present payload must be a JSON object whose
    own ``winner`` agrees with the row's verified winner — a verdict that
    contradicts the winner it supposedly explains is refused, never
    recorded (a recorded payload is provenance only if verified)."""
    def _as_confidence(value) -> float | None:
        return (float(value)
                if isinstance(value, (int, float))
                and not isinstance(value, bool) else None)

    confidence = _as_confidence(row.get("confidence"))
    payload = row.get("judgment")
    if payload is None:
        return confidence, None
    if not isinstance(payload, dict):
        raise ValueError(
            f"judgment by {judge!r} carries a non-object verdict payload — "
            "refusing")
    if payload.get("winner") != winner:
        raise ValueError(
            f"judgment by {judge!r} carries a verdict payload whose winner "
            f"({payload.get('winner')!r}) contradicts the judged winner "
            f"({winner!r}) — refusing an inconsistent verdict")
    if confidence is None:
        confidence = _as_confidence(payload.get("confidence"))
    return confidence, payload


def _agreement_entries(labeled: list[tuple[str, dict]]) -> list[dict]:
    """Percent agreement + Cohen's kappa for every judge pair, over the pair
    cells both judged. ``labeled`` is ``[(judge_name, {key: outcome})]``."""
    from . import study_stats
    entries: list[dict] = []
    for i in range(len(labeled)):
        for j in range(i + 1, len(labeled)):
            name_a, map_a = labeled[i]
            name_b, map_b = labeled[j]
            keys = sorted(set(map_a) & set(map_b))
            if not keys:
                continue
            a = [map_a[key] for key in keys]
            b = [map_b[key] for key in keys]
            entries.append({"judges": [name_a, name_b], "n": len(keys),
                            "percentAgreement": study_stats.percent_agreement(a, b),
                            "kappa": study_stats.cohens_kappa(a, b)})
    return entries


def _load_human_validation(
        manifest: Manifest, root) -> dict[tuple[str, str | None, str], str]:
    """The pinned human-labeled subset, hash-checked then parsed through the
    ONE row parser (``human_validation.parse_rows`` — the same rules
    ``Manifest.verify`` applies, review 2026-08-02). See that module for the
    cross-engine row semantics; Swift twin:
    ``ExperimentTasks.parseHumanValidation`` + ``humanAgreement``."""
    from . import human_validation
    hv = manifest.human_validation
    path = paths.resolve(hv.path, root)
    with open(path, "rb") as handle:
        data = handle.read()
    live = hashlib.sha256(data).hexdigest()
    if live != hv.hash:
        raise RuntimeError(
            f"human validation '{hv.path}' drifted from the pinned hash "
            f"(have {live[:12]}…, pinned {hv.hash[:12]}…)")
    return human_validation.parse_rows(data, hv.path)


def _materialize_human_validation(
        human: dict[tuple[str, str | None, str], str],
        outcome_maps: list[tuple[str, dict]]) -> dict[tuple[str, str, str], str]:
    """Resolve wildcard rows against the cells the judges actually judged,
    exact rows first — the agreement join is a key-set intersection, so the
    wildcard must be expanded to concrete keys before it can match."""
    resolved: dict[tuple[str, str, str], str] = {
        key: outcome for key, outcome in human.items() if key[1] is not None}
    judged: set[tuple[str, str, str]] = set()
    for _, outcome_map in outcome_maps:
        judged.update(outcome_map)
    for key in judged:
        prompt_id, _, condition = key
        if key not in resolved and (prompt_id, None, condition) in human:
            resolved[key] = human[(prompt_id, None, condition)]
    return resolved


#: What a partial evaluate run was judged UNDER. Written before the first
#: judge call so a later targeted retry can prove it is completing the same
#: evaluation rather than merging two different ones.
JUDGING_CONTEXT_FILENAME = "judging-context.json"


def _judging_context(manifest, spec, run_dir: str, rubric_hash: str | None,
                     rubric_file: str | None, roster) -> dict:
    """The pins a resumed evaluation must match to reuse judgments.

    Every field here is something that, if it changed, would make an
    earlier verdict an answer to a DIFFERENT question. The bar is the
    deferred path's, which pins the source generations by content
    (``sourceGenerationsSha256``) — retry pinning less than deferred would
    be backwards, since retry is precisely the case where time has passed
    (external review 2026-07-24, finding 1).

    Judges are recorded by RESOLVED identity, not as spelled: a local judge
    with a blank model resolves to the study model, so comparing the raw
    spelling would refuse a resume that is in fact identical — and, worse,
    would ACCEPT one where the study model changed underneath a blank.
    ``revision`` and ``dtype`` ride along because a foreign local judge
    reloaded at a different revision is a different judge, whatever its
    name says.
    """
    from . import paired_judge, sweep_selection
    structured = getattr(spec, "structured_prompt", None)
    generations = os.path.join(run_dir, "generations.jsonl")
    try:
        with open(generations, "rb") as handle:
            source_sha = hashlib.sha256(handle.read()).hexdigest()
    except OSError as exc:
        # Refuse, never pin None: the run this context describes must exist
        # to be judged, and a None hash made the resume pin VACUOUS — the
        # resume equality check compared None == None and passed, so two
        # evaluations of two unreadable (possibly different) source runs
        # "matched". The deferred path already refuses by raising on open;
        # retry must not pin less than deferred.
        raise RuntimeError(
            f"cannot judge run '{os.path.basename(run_dir.rstrip(os.sep))}': "
            f"its generations.jsonl cannot be read ({type(exc).__name__}: "
            f"{exc}) — the source generations must exist to be judged, and "
            "without their hash this evaluation could never prove what it "
            "judged") from exc
    judges = []
    for r in roster:
        resolved = r.model
        revision = r.revision
        if r.kind == "local":
            resolved = sweep_selection.resolve_local_judge_model(
                r.model, manifest.model_id)
            # A study-model judge inherits the study's pinned revision —
            # the same fallback the loader applies.
            if not revision and resolved == manifest.model_id:
                revision = manifest.model_revision
        judges.append({
            "name": r.name, "kind": r.kind, "model": resolved,
            "revision": revision if r.kind == "local" else None,
            "dtype": r.dtype if r.kind == "local" else None,
            "provider": (paired_judge.canonical_openrouter_provider(r.provider)
                         if r.kind == "openrouter" else None),
        })
    return {
        "schemaVersion": 2,
        "experiment": manifest.name,
        "experimentHash": manifest.content_hash(),
        "sourceRun": os.path.basename(run_dir.rstrip(os.sep)),
        # The directory NAME only says which run; the hash says which
        # BYTES. Runs are immutable by convention, but retention now writes
        # into run directories, so "nobody touches a run" is no longer a
        # thing to rest an evidence claim on.
        "sourceGenerationsSha256": source_sha,
        "rubricFile": rubric_file,
        "rubricHash": rubric_hash,
        "structuredPromptSha256": (
            hashlib.sha256(structured.encode("utf-8")).hexdigest()
            if structured else None),
        "judges": judges,
    }


def _load_resumable_judgments(name: str, resume_from: str, root: str | None,
                              context: dict, log) -> dict:
    """Judgments from a partial evaluate run, keyed ``(judge, cell)``.

    Every pin is checked before a single row is reused. Reuse is the whole
    point of a targeted retry — judging is the expensive, non-deterministic
    step — but reusing a verdict produced under a DIFFERENT rubric, a
    different manifest epoch, a different source run, or by a differently
    configured judge would silently merge two experiments into one table.
    Each of those is a refusal, naming what differs.
    """
    runs = paths.runs_directory(root)
    if not resume_from or "/" in resume_from or os.sep in resume_from \
            or resume_from in (".", ".."):
        raise RuntimeError(f"invalid resume run id: {resume_from!r}")
    partial = os.path.join(runs, resume_from)
    if not os.path.isdir(partial):
        raise RuntimeError(f"no run directory '{resume_from}' to resume from")
    if os.path.exists(os.path.join(partial, "judge-report.json")):
        raise RuntimeError(
            f"'{resume_from}' is a COMPLETED evaluation (it has a "
            "judge-report.json) — there is nothing to retry. Analyze it, or "
            "run a fresh evaluate")
    context_path = os.path.join(partial, JUDGING_CONTEXT_FILENAME)
    try:
        with open(context_path, encoding="utf-8") as handle:
            before = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"'{resume_from}' carries no readable {JUDGING_CONTEXT_FILENAME} "
            f"({type(exc).__name__}) — it predates targeted retry, so what it "
            "was judged under cannot be proven. Re-run evaluate from the "
            "source run instead of resuming") from exc

    if int(before.get("schemaVersion") or 1) < 2:
        # A schema-1 context pinned the source run by NAME only and carried
        # no judge revision/dtype, so it cannot prove the things a resume
        # now has to prove. Refuse rather than silently applying weaker
        # rules to older evidence.
        raise RuntimeError(
            f"cannot resume '{resume_from}': its {JUDGING_CONTEXT_FILENAME} "
            "predates the strengthened pins (no source-generations hash, no "
            "judge revision) — it cannot prove those judgments answer the "
            "same question. Run a fresh evaluate")
    for key, label in (("experimentHash", "experiment epoch"),
                       ("sourceRun", "source run"),
                       ("sourceGenerationsSha256", "source generations"),
                       ("rubricHash", "rubric"),
                       ("structuredPromptSha256", "structured prompt")):
        if before.get(key) != context.get(key):
            raise RuntimeError(
                f"cannot resume '{resume_from}': its {label} differs from "
                f"the current evaluation ({before.get(key)!r} vs "
                f"{context.get(key)!r}) — those judgments answered a "
                "different question. Run a fresh evaluate")
    before_judges = {j["name"]: j for j in before.get("judges") or []}
    for judge in context["judges"]:
        prior = before_judges.get(judge["name"])
        if prior is None:
            continue  # a judge added since: it simply has nothing to reuse
        if prior != judge:
            raise RuntimeError(
                f"cannot resume '{resume_from}': judge '{judge['name']}' was "
                f"configured differently ({prior} vs {judge}) — its earlier "
                "verdicts came from a different judge. Run a fresh evaluate")

    rows: dict[tuple[str, tuple[str, str, str]], dict] = {}
    path = os.path.join(partial, "judgments.jsonl")
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                judge = str(row.get("judge") or "")
                if not judge:
                    continue
                rows[(judge, _judgment_key(row))] = row
    except FileNotFoundError:
        rows = {}
    except json.JSONDecodeError as exc:
        # A torn tail is normal for a killed writer; everything before it is
        # still good. Refusing the whole file would throw away exactly the
        # data this feature exists to save.
        log(f"WARNING: '{resume_from}' judgments.jsonl has an unreadable "
            f"tail ({exc}) — reusing the {len(rows)} complete row(s) before it")
    log(f"resuming from '{resume_from}': {len(rows)} judgment(s) available "
        "for reuse")
    return rows


def _preflight_openrouter_judges(roster, log, *, transport=None) -> None:
    """Check every openrouter judge's provider pin BEFORE work starts.

    A wrong provider used to surface at the first judge call — after
    generation had finished and GPU hours were spent. The catalogue this
    checks against is public and keyless, so the check costs nothing and
    needs no credential.

    Refuses only on POSITIVE evidence (the catalogue answered and the pin
    is not among the serving endpoints). An unreachable catalogue warns and
    proceeds: compute nodes routinely have no outbound network, and a study
    must not become unrunnable because a metadata endpoint was down. The
    call-time off-pin refusal is still there either way — this moves the
    discovery earlier, it does not replace the guarantee.

    ``STEERLAB_SKIP_PROVIDER_PREFLIGHT=1`` disables the catalogue lookup.
    It exists for air-gapped sites that would rather not wait out a network
    timeout at every start, and it is what the test suite sets so no test
    reaches the real internet. Skipping is LOGGED — an unverified pin must
    never look like a verified one.
    """
    from . import paired_judge
    if os.environ.get("STEERLAB_SKIP_PROVIDER_PREFLIGHT", "").strip().lower() \
            in ("1", "true", "yes"):
        if any(getattr(r, "kind", "") == "openrouter" for r in roster):
            log("provider preflight SKIPPED "
                "(STEERLAB_SKIP_PROVIDER_PREFLIGHT) — openrouter provider "
                "pins stay unverified until the first judge call")
        return
    for ref in roster:
        if getattr(ref, "kind", "") != "openrouter":
            continue
        result = paired_judge.preflight_openrouter_provider(
            ref.model, ref.provider or "", transport=transport)
        for warning in result["warnings"]:
            log(f"WARNING: judge '{ref.name}': {warning}")
        if result["problem"]:
            raise RuntimeError(f"judge '{ref.name}': {result['problem']}")
        if result["checked"]:
            log(f"judge '{ref.name}': provider pin "
                f"'{paired_judge.canonical_openrouter_provider(ref.provider)}' "
                f"verified against OpenRouter's catalogue for '{ref.model}'")


def _judgment_stamp_judge(judgment: dict, ref) -> None:
    """Stamp a judgment with the judge that produced it, in place.

    Split out of the evaluate loop when judgments began being written as
    they are produced (2026-07-24): the stamp has to happen BEFORE the row
    reaches disk, not in a post-pass over the returned list."""
    from . import paired_judge
    judgment["judge"] = ref.name
    if ref.kind == "openrouter":
        # The VERIFIED serving provider from the verdict (the client refused
        # an unattributed or off-pin response) — the same per-judgment stamp
        # the deferred completion writes, so inline and deferred reports
        # carry one provenance shape.
        judgment["judgeProvider"] = paired_judge.canonical_openrouter_provider(
            (judgment.get("judgment") or {}).get("provider") or ref.provider)


def _evaluate_response_coding(name: str, manifest: Manifest, schema,
                              run_dir: str, generations: list[dict],
                              rubric_hash: str | None,
                              rubric_file: str | None, roster,
                              root: str | None, epoch_unverified: bool,
                              measurement_drift: str | None,
                              evaluation_source: str | None,
                              exclusion_stamp: dict | None,
                              model_provider, _log, *, model_release=None,
                              study_model_generates_later: bool = False) -> str:
    """The per-response coding instrument's evaluate body (2026-08-04;
    Swift twin: ``ExperimentTasks.runResponseCoding``).

    Every sampled-text record — baseline INCLUDED — goes to every judge
    individually and blinded (the coder sees the task prompt and one
    response, never the condition, never a second response). Codes are
    validated against the rubric's declared schema (retry once, then refuse
    — invented data is never recorded), streamed row-by-row to
    ``codings.jsonl``, and summarized in ``coding-report.json`` with
    per-condition per-field aggregates, engine-computed word counts, and
    per-field inter-judge agreement (percent + Cohen's kappa for
    categorical fields — the same statistic the K&Z paper used to validate
    its coders). There is no pairing and no winner anywhere on this path.
    """
    from . import response_coding
    codeable = [g for g in generations
                if "instrument" not in g and "error" not in g
                and "output" in g]
    if not codeable:
        raise RuntimeError(response_coding.NO_CODEABLE_MESSAGE)
    out = paths.make_unique_run_directory(f"exp-{name}-evaluate", root)
    write_run_config(out, "evaluate", model_id=manifest.model_id,
                     revision=manifest.model_revision, experiment=name,
                     experiment_hash=manifest.content_hash(),
                     notes={"epochUnverified": True} if epoch_unverified else None)
    if exclusion_stamp is not None:
        with open(os.path.join(out, "exclusions.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(exclusion_stamp, handle, indent=2, sort_keys=True)
    status = RunStatus(out, stage="evaluate", experiment=name,
                       source_run=os.path.basename(run_dir),
                       expected=[ref.name for ref in roster],
                       item_label="coding")
    status.write()
    _log(f"coding {len(codeable)} record(s) × {len(roster)} judge(s) "
         f"under perResponseCoding rubric "
         f"'{rubric_file or '(inline draft)'}'")

    rows: list[dict] = []
    judge_details: list[dict] = []
    codings_path = os.path.join(out, "codings.jsonl")
    try:
        with open(codings_path, "w", encoding="utf-8") as codings_handle:
            for index, ref in enumerate(roster):
                # One model load per judge COLUMN (same rule as paired
                # evaluate): the stack closes at the end of each iteration,
                # and the release seam below frees the finished column's
                # container BEFORE this one loads, so two judge models are
                # never resident at once.
                _release_models_for_judge(
                    model_release, roster, index,
                    study_model=manifest.model_id,
                    study_model_generates_later=study_model_generates_later,
                    _log=_log)
                with ExitStack() as judge_stack:
                    complete_fn, requested_model, holder = _coder_callable(
                        ref, model_provider, study_model=manifest.model_id,
                        study_revision=manifest.model_revision,
                        stack=judge_stack)
                    judge_noncompliant = 0
                    for g in codeable:
                        sample_index = g.get("sampleIndex") or 0
                        try:
                            result = response_coding.valid_codes(
                                complete_fn, schema, g.get("output", ""),
                                g.get("prompt") or None,
                                judge_label=f"'{ref.name}'",
                                item_label=(f"record {g.get('condition')}/"
                                            f"{g.get('promptID')}"
                                            f"[{sample_index}]"),
                                on_invalid=status.note_invalid_response)
                        except response_coding.paired_judge.JudgeNoncompliant as exc:
                            # Same policy as the paired judge (Christian,
                            # 2026-08-09): one record the coder ANSWERS but
                            # will not code becomes a recorded, classifiable
                            # ROW — codes: None, excluded from aggregates
                            # and agreement — instead of aborting hours of
                            # completed work. Transport errors still fail
                            # the session (resume machinery); systemic
                            # noncompliance still fails too: see the cap
                            # check after this judge's column.
                            judge_noncompliant += 1
                            row = {
                                "experiment": name,
                                "condition": g.get("condition"),
                                "promptID": g.get("promptID"),
                                "sampleIndex": sample_index,
                                "seed": g.get("seed"),
                                "codes": None,
                                "noncompliant": True,
                                "noncomplianceReason": str(exc)[:2000],
                                "judge": ref.name,
                                "judgeKind": ref.kind,
                                "judgeModel": requested_model,
                            }
                            rows.append(row)
                            codings_handle.write(json.dumps(row) + "\n")
                            codings_handle.flush()
                            status.note_item()
                            continue
                        row = {
                            "experiment": name,
                            "condition": g.get("condition"),
                            "promptID": g.get("promptID"),
                            "sampleIndex": sample_index,
                            "seed": g.get("seed"),
                            "wordCount": response_coding.word_count(
                                g.get("output", "")),
                            "codes": result["codes"],
                            "briefReason": result["briefReason"],
                            "judge": ref.name,
                            "judgeKind": ref.kind,
                            "judgeModel": requested_model,
                        }
                        # Keys the coder invented, kept verbatim and kept OUT
                        # of the measurement — absent from the row entirely
                        # when it invented nothing. Nothing aggregates this.
                        if result.get("undeclaredCodes"):
                            row["undeclaredCodes"] = result["undeclaredCodes"]
                        if result.get("provider"):
                            row["judgeProvider"] = result["provider"]
                        if ref.kind == "local" and holder.get("revision"):
                            row["judgeRevision"] = holder["revision"]
                        rows.append(row)
                        # Flushed per row: a killed worker still leaves
                        # every coding it finished.
                        codings_handle.write(json.dumps(row) + "\n")
                        codings_handle.flush()
                        status.note_item()
                    from . import paired_judge as _pj
                    if (judge_noncompliant and codeable
                            and judge_noncompliant / len(codeable)
                            > _pj.NONCOMPLIANCE_CAP):
                        raise RuntimeError(
                            f"judge '{ref.name}' was noncompliant on "
                            f"{judge_noncompliant} of {len(codeable)} "
                            f"record(s) (> "
                            f"{_pj.NONCOMPLIANCE_CAP:.0%} cap) — "
                            "systemic coder failure, not flakiness. Every "
                            "row (including the noncompliant ones, with "
                            "raw reasons) is in codings.jsonl; fix or swap "
                            "the judge, then re-run evaluate")
                    detail = {"name": ref.name, "kind": ref.kind,
                              "requestedModel": requested_model,
                              "actualModel": holder.get("actual")}
                    if holder.get("revision"):
                        detail["revision"] = holder["revision"]
                    if holder.get("actualDtype"):
                        detail["actualDtype"] = holder["actualDtype"]
                    if judge_noncompliant:
                        detail["noncompliantCodings"] = judge_noncompliant
                    judge_details.append(detail)
                    status.note_judge_complete(ref.name)
                    _log(f"judge '{ref.name}' coded "
                         f"{len(codeable) - judge_noncompliant} record(s)"
                         + (f" ({judge_noncompliant} noncompliant, kept as "
                            "rows for review)" if judge_noncompliant
                            else ""))
    except BaseException as exc:
        status.fail(exc)
        _log(f"coding evaluate FAILED after {status.judgment_count} "
             f"coding(s) — rows kept in {os.path.basename(out)}; no "
             "coding report written")
        raise

    report = {
        "mode": "perResponseCoding",
        "experiment": name,
        "experimentHash": manifest.content_hash(),
        "sourceRun": os.path.basename(run_dir),
        "judges": [ref.name for ref in roster],
        "judgeModel": ", ".join(d["requestedModel"] for d in judge_details),
        "judgeDetails": judge_details,
        "judgeRubricFile": rubric_file,
        "judgeRubricHash": rubric_hash,
        "fields": [
            {"name": f.name, "type": f.type, "optional": f.optional,
             **({"values": list(f.values)} if f.values else {})}
            for f in schema.fields
        ],
        "codings": len(rows),
        "conditions": response_coding.aggregate_conditions(rows, schema),
        "evaluationSource": evaluation_source,
    }
    # Absent-with-reason rather than empty when ONE coder coded the run:
    # there is no pair to compare, which is not the same fact as a pair that
    # agreed about nothing. Swift twin: `CodingReport.fieldAgreement` is
    # optional and omitted on the same rule.
    if len(roster) >= 2:
        report["fieldAgreement"] = response_coding.field_agreement(
            rows, schema, [ref.name for ref in roster])
    else:
        report["fieldAgreementAbsentReason"] = \
            response_coding.SINGLE_CODER_AGREEMENT_ABSENT_REASON
    noncompliant_total = sum(1 for r in rows if r.get("noncompliant"))
    if noncompliant_total:
        # Nonzero-only, like the paired judge's noncompliantJudgments: these
        # rows carry no codes and sit outside every aggregate — the report
        # must say the columns are incomplete and by how much.
        report["noncompliantCodings"] = noncompliant_total
    if epoch_unverified:
        report["epochUnverified"] = True
    if measurement_drift:
        # Tolerated measurement-side drift is never silent: which fields
        # differed from the source run's epoch, verbatim.
        report["measurementDrift"] = measurement_drift
    if exclusion_stamp is not None:
        report["exclusions"] = exclusion_stamp
    with open(os.path.join(out, "coding-report.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    status.complete()
    _log(f"coding evaluation artifacts: {out}")
    return out


def evaluate(name: str, root: str | None = None, source_run: str | None = None,
             *, model_provider=None, should_cancel: Callable[[], bool] | None = None,
             log=None, allow_unverified_epoch: bool = False,
             max_loaded: int | None = None,
             defer_local_judges: bool = False,
             resume_from: str | None = None,
             model_release=None,
             study_model_generates_later: bool = False) -> str:
    """Paired-judge evaluation over a prior run's generations (parallel to the
    Swift evaluate task). Judges with the manifest's PINNED rubric file (drafts
    may fall back to the inline prompt, loudly) and its pinned judge panel:
    every judge scores every pair, each judgment is stamped with the judge's
    name, and the report carries per-judge tallies plus inter-judge agreement
    (percent + Cohen's kappa per judge pair) — and judge-vs-human agreement
    when a ``humanValidation`` subset is pinned. Claude judges need
    ANTHROPIC_API_KEY; local-model judges run through ``model_provider``.

    Local judges resolve by the sweep's cross-engine rule (unified
    2026-07-22): an empty/absent ``model`` means the STUDY model at its
    pinned revision — never the judge's name as a model id. A local judge
    naming a DIFFERENT model needs a second resident slot: ``max_loaded``
    (the registry capacity, passed by the API path) < 2 refuses at evaluate
    start; None (the CLI/bundle path, private in-process copies) skips the
    capacity check, exactly like the sweep preflight.

    Model residency across the panel (2026-08-28): judges are needed one
    COLUMN at a time, so the peak is the MAX of any one still-needed model,
    never the SUM. ``model_release`` (the API path passes the registry's
    explicit release) is called at every judge boundary with the models the
    remainder of the run no longer needs — see
    ``_release_models_for_judge``. ``study_model_generates_later`` is the
    conservative flag for the generation→judging seam: False (evaluate is
    terminal for generation) lets the study model go when no judge uses it;
    a pipeline stage that still generates passes True and it is kept.

    Epoch guard: the source run's stamped experiment hash must equal the live
    manifest's content hash; legacy unstamped runs need
    ``allow_unverified_epoch`` and are stamped ``epochUnverified: true``."""
    from . import paired_judge
    _log = log or print
    manifest = Manifest.load(name, root)
    # Effective evaluation (2026-07-22 incident): an explicit block wins;
    # with none, pinned judges + a pinned rubric file ARE the paired-judge
    # declaration and the spec is synthesized from those pins (the app's
    # rubric-file path historically never wrote the block, so frozen
    # studies died HERE after generation). The judge report stamps where
    # the spec came from (cross-engine key "evaluationSource":
    # "manifest" | "pinnedRubric").
    spec, evaluation_source = manifest.effective_evaluation()
    if spec is None or spec.kind != "pairedJudge":
        # WP0 dry run #2's skipped check: right refusal, untyped
        # (`verbFailed`/70 with the boilerplate repair). The verb needs
        # something the study never declared — missingPrerequisite, with a
        # repair that names the CLI that can actually pin it. The reason
        # string is unchanged.
        raise lifecycle_gates.refusing(
            lifecycle_gates.MISSING_PREREQUISITE,
            f"experiment '{name}' has no pairedJudge evaluation configured "
            "— pin at least one judge and a rubric, or declare an "
            "evaluation block",
            repair=no_rubric_repair(name))
    if evaluation_source == "pinnedRubric":
        _log("evaluation: no explicit evaluation block — judging from the "
             "pinned judges + rubric file (evaluationSource: pinnedRubric)")

    rubric, rubric_hash, rubric_file = _resolve_rubric(manifest, root, _log)
    human = _load_human_validation(manifest, root) if manifest.human_validation else None

    run_dir = source_run or _latest_run(name, root)
    if not run_dir:
        raise lifecycle_gates.refusing(
            lifecycle_gates.MISSING_PREREQUISITE,
            f"no prior run with generations found for '{name}' — run it first",
            repair=(f"steerlab-server experiment run {name} && "
                    f"steerlab-server experiment analyze {name}"))
    epoch_unverified, measurement_drift = _require_source_epoch(
        "evaluate", name, manifest, run_dir,
        allow_unverified_epoch=allow_unverified_epoch)
    if measurement_drift:
        _log(f"WARNING: '{name}' drifted from source run "
             f"'{os.path.basename(run_dir)}' in MEASUREMENT-side fields only "
             f"({measurement_drift}) — the generations are unaffected; "
             "judging proceeds under the LIVE settings and the output is "
             "stamped measurementDrift")
    if epoch_unverified:
        _log(f"WARNING: source run '{os.path.basename(run_dir)}' carries no "
             "experiment-hash stamp — judging it under allowUnverifiedEpoch; "
             "the report is stamped epochUnverified")
    generations = []
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                generations.append(json.loads(line))

    # Declared exclusion rules join HERE — BEFORE judging, so no judge call
    # (or judging packet, on the deferred path) is ever spent on an
    # excluded record. Pairwise deletion matches analyze: filtering the
    # baseline record removes its item from every condition's pairs (the
    # baseline join in paired_judge simply finds no partner). The stamp
    # lands in judge-report.json + exclusions.json; excluded records stay
    # in the source run's generations.jsonl (runs are immutable). No rules
    # declared = today's behavior byte-for-byte. Swift twin:
    # ``evaluatePairedJudge``.
    from . import exclusions as exclusions_mod
    exclusion_rules = exclusions_mod.declared_rules(manifest.raw)
    exclusion_stamp = None
    if exclusion_rules:
        checks: dict[str, dict] = {}
        if exclusions_mod.needs_checks(exclusion_rules):
            if not manifest.task_prompts_hash:
                # WP0 step 8: the deferred cross-engine-twinned message from
                # c86ce53 gets its id on BOTH engines. The STRING is unchanged
                # and stays byte-identical to Swift's
                # `ExclusionEngine.pinRequiredMessage` (asserted on both
                # sides); only the gate id and the repair are new.
                raise lifecycle_gates.refusing(
                    lifecycle_gates.MISSING_PREREQUISITE,
                    exclusions_mod.PIN_REQUIRED_MESSAGE,
                    repair=exclusions_mod.PIN_REQUIRED_REPAIR)
            checks = exclusions_mod.attention_checks(
                _load_prompts(manifest, None, root))
            if not checks:
                raise RuntimeError(exclusions_mod.NO_CHECKS_MESSAGE)
        generations, exclusion_stamp = exclusions_mod.apply(
            generations, exclusion_rules, checks,
            note=exclusions_mod.EVALUATE_NOTE,
            scope=exclusions_mod.SCOPE_SAMPLED_RECORDS)
        _log(f"exclusions: {exclusion_stamp['excludedRecords']} record(s) "
             f"excluded before judging by {len(exclusion_rules)} declared "
             "rule(s); surviving N per condition: "
             + ", ".join(f"{c}={n}" for c, n
                         in exclusion_stamp["survivingN"].items()))

    roster = _judge_roster(manifest, spec)
    # Local-judge resolution, logged at evaluate START (cross-engine rule,
    # unified with the sweep 2026-07-22 — the judge's NAME is a label, never
    # a model id): empty/absent model → the study model; a different-model
    # local judge refuses here when a second resident model is impossible,
    # never mid-panel — UNLESS the caller declared the judge fan-out
    # (``defer_local_judges``, 2026-07-23): the panel's judging then becomes
    # blinded packets for per-judge-model worker jobs instead of an inline
    # second load.
    from . import sweep_selection
    foreign_local: list = []
    for ref in roster:
        if ref.kind != "local":
            continue
        resolved = sweep_selection.resolve_local_judge_model(
            ref.model, manifest.model_id)
        if resolved == manifest.model_id:
            _log(f"local judge '{ref.name}' resolves to the study model "
                 f"{manifest.model_id}")
            continue
        _log(f"local judge '{ref.name}' judges with local model '{resolved}'")
        foreign_local.append(ref)
        if not defer_local_judges and max_loaded is not None and max_loaded < 2:
            raise RuntimeError(
                f"evaluate with local judge '{ref.name}' (model "
                f"'{resolved}') needs a second resident model alongside "
                f"the study model '{manifest.model_id}' — this server "
                f"keeps STEERLAB_MAX_LOADED_MODELS={max_loaded}; use the "
                "study model as judge (leave the judge's model empty), a "
                "claude judge, or raise the limit")
    # Where judging will happen, announced BEFORE it happens (2026-07-24):
    # the inline/deferred fork used to be discoverable only from the
    # artifacts afterwards, and a mixed panel deferring despite a pushed key
    # is the case that surprises people.
    log_judging_custody(roster, _log)
    # Provider pins are checked against OpenRouter's public catalogue BEFORE
    # any judging starts (2026-07-24) — a wrong provider used to surface at
    # the first judge call, after generation had already been paid for.
    _preflight_openrouter_judges(roster, _log)
    # Per-response coding fork (2026-08-04): a rubric whose frontmatter
    # declares `mode: perResponseCoding` runs the coding instrument — every
    # sampled-text record coded individually, blinded, no pairing and no
    # winner. Everything above (epoch guard, exclusions, roster resolution,
    # provider preflight) is shared; everything below is paired-only.
    from . import response_coding
    coding_schema = response_coding.parse_rubric(rubric)
    if coding_schema is not None:
        if (spec.structured_prompt or "").strip():
            raise RuntimeError(
                "the study declares a paired structured-comparison prompt "
                "but the pinned rubric is perResponseCoding — the two "
                "contracts cannot combine; clear the structured prompt or "
                "pin a paired rubric")
        if manifest.human_validation:
            raise RuntimeError(
                "judge-vs-human agreement for per-response coding is not "
                "implemented yet — unpin humanValidation (the paired-shape "
                "baseline|variant|tie labels do not describe per-response "
                "codes)")
        if resume_from:
            raise RuntimeError(
                "per-response coding does not support --resume-from yet — "
                "re-run the evaluation in one session")
        if (defer_local_judges and foreign_local) \
                or _missing_external_credentials(roster):
            raise RuntimeError(
                "per-response coding judges inline only for now — deferred "
                "judging packets for the coding instrument are not "
                "implemented yet. Push a judge key from the app for inline "
                "external coding, or pin local judges (resolving to the "
                "study model on a single-slot server)")
        return _evaluate_response_coding(
            name, manifest, coding_schema, run_dir, generations,
            rubric_hash, rubric_file, roster, root, epoch_unverified,
            measurement_drift, evaluation_source, exclusion_stamp,
            model_provider, _log, model_release=model_release,
            study_model_generates_later=study_model_generates_later)
    # P0 guard (external review 2026-07-22): a pairedJudge evaluate always
    # has a judge panel configured, so zero surviving pairs must refuse HERE
    # — a "pairs: 0" judge-report looked like a successful evaluation while
    # judging nothing (the exact failure mode of the old (promptID, seed)
    # join under per-condition derived seeds). Fires before the custody fork
    # so inline and deferred paths refuse identically.
    if not paired_judge._pair_generations(generations):
        raise RuntimeError(paired_judge.NO_PAIRS_MESSAGE)
    # Judge fan-out deferral (2026-07-23): the caller (the pipeline chain,
    # which holds ONE model) declared that local judges needing OTHER models
    # must not judge inline. The whole LOCAL panel becomes blinded,
    # hash-pinned packets — per-judge-model worker jobs judge them and the
    # controller merges through `complete_evaluate_judgment` (full-coverage
    # refusals included). Mixed local+external panels refuse for now: the
    # external half would judge at a different evidence time than the
    # workers — pin an all-local or all-external panel (the honest refusal
    # this commit ships instead of a two-clock merge).
    if defer_local_judges and foreign_local:
        external_kinds = sorted({r.kind for r in roster if r.kind != "local"})
        if external_kinds:
            raise RuntimeError(
                "evaluate panel mixes local judges needing the judge "
                f"fan-out ({', '.join(r.name for r in foreign_local)}) with "
                f"{'/'.join(external_kinds)} judges — a mixed panel cannot "
                "merge one report from two judging clocks yet. Pin an "
                "all-local panel (fans out per judge model) or an "
                "all-external panel (judges inline/deferred), or run "
                "evaluate outside the pipeline")
        return _emit_evaluate_judging(
            name, manifest, spec, run_dir, generations, rubric, rubric_hash,
            rubric_file, root, epoch_unverified, measurement_drift, _log,
            exclusion_stamp=exclusion_stamp,
            evaluation_source=evaluation_source)
    # Custody fork (seamless pipeline stage 2, 2026-07-19): external judges
    # without a credential DEFER — the run's paired generations become
    # blinded, hash-pinned packets the Mac judges; complete-judgment
    # verifies and aggregates. Split local/external panels refuse (two
    # evidence times for one report), exactly the sweep rule.
    missing = _missing_external_credentials(roster)
    if missing:
        if any(ref.kind == "local" for ref in roster):
            raise RuntimeError(
                "evaluate panel mixes local and external "
                f"({'/'.join(sorted(r.kind for r in roster if r.kind != 'local'))}) "
                "judges but this server has no credential for "
                f"{'/'.join(sorted(missing))} (keyless is the default "
                "custody posture): a split panel cannot defer coherently. "
                "Pin an all-local or all-external panel, or push a judge "
                "key from the app for inline external judging")
        return _emit_evaluate_judging(
            name, manifest, spec, run_dir, generations, rubric, rubric_hash,
            rubric_file, root, epoch_unverified, measurement_drift, _log,
            exclusion_stamp=exclusion_stamp,
            evaluation_source=evaluation_source)
    # Retention (2026-07-24): the evaluate run directory is created BEFORE
    # the first judge call, and every judgment is written as it is produced
    # — the Swift path's shape, ported. The old code created the directory
    # after the whole panel finished, so an invalid verdict on the last pair
    # of the last judge destroyed every successful judgment before it and
    # left the researcher with an error message and nothing on disk. A
    # partial directory is marked by run-status.json and the ABSENCE of
    # judge-report.json — a partial panel is never summarized as a report.
    context = _judging_context(manifest, spec, run_dir, rubric_hash,
                               rubric_file, roster)
    resumable: dict = {}
    if resume_from:
        resumable = _load_resumable_judgments(
            name, resume_from, root, context, _log)
        if resumable:
            # Honest about WHEN, not just what (2026-07-24). A resumed
            # evaluation is judged across two sessions, and an external
            # judge's model can change between them — a `claude-opus-4-8`
            # or an OpenRouter endpoint is not revision-pinned the way a
            # local judge is. The reader has to be able to see that.
            external = sorted({r.name for r in roster if r.kind != "local"})
            if external:
                _log("WARNING: resuming judgment reuses verdicts from an "
                     f"earlier session while judge(s) {', '.join(external)} "
                     "are external (not revision-pinned) — the provider may "
                     "have changed the model between sessions. The report "
                     "stamps this as a multi-session evaluation")

    out = paths.make_unique_run_directory(f"exp-{name}-evaluate", root)
    write_run_config(out, "evaluate", model_id=manifest.model_id,
                     revision=manifest.model_revision, experiment=name,
                     experiment_hash=manifest.content_hash(),
                     notes={"epochUnverified": True} if epoch_unverified else None)
    # The pins THIS run judges under, written before the first judge call so
    # a later targeted retry can prove it is completing the same evaluation.
    with open(os.path.join(out, JUDGING_CONTEXT_FILENAME), "w",
              encoding="utf-8") as handle:
        json.dump(context, handle, indent=2, sort_keys=True)
    if exclusion_stamp is not None:
        # The stamp file the analyze path also writes — one artifact name
        # to look for on either engine.
        with open(os.path.join(out, "exclusions.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(exclusion_stamp, handle, indent=2, sort_keys=True)
    status = RunStatus(out, stage="evaluate", experiment=name,
                       source_run=os.path.basename(run_dir),
                       expected=[ref.name for ref in roster])
    status.write()

    all_judgments: list[dict] = []
    judge_blocks: list[dict] = []
    outcome_maps: list[tuple[str, dict]] = []
    judgments_path = os.path.join(out, "judgments.jsonl")
    try:
        with open(judgments_path, "w", encoding="utf-8") as judgments_handle:
            for index, ref in enumerate(roster):
                # One model load per judge COLUMN (external review round 3,
                # finding 3c). The stack closes at the END of each
                # iteration, and the release seam below drops the finished
                # column's container BEFORE this one loads, so two judge
                # models are never resident at once — the guarantee the
                # single-slot rule used to buy by refusing instead.
                _release_models_for_judge(
                    model_release, roster, index,
                    study_model=manifest.model_id,
                    study_model_generates_later=study_model_generates_later,
                    _log=_log)
                with ExitStack() as judge_stack:
                    judge_fn, requested_model, holder = _judge_callable(
                        ref, model_provider, study_model=manifest.model_id,
                        study_revision=manifest.model_revision,
                        stack=judge_stack)

                    def _persist(judgment, _ref=ref, _handle=judgments_handle):
                        _judgment_stamp_judge(judgment, _ref)
                        # Flushed per row: a killed worker (SIGKILL, node
                        # eviction) still leaves every judgment it finished.
                        _handle.write(json.dumps(judgment) + "\n")
                        _handle.flush()
                        status.note_judgment()

                    # Cells this judge already decided in the resumed session:
                    # reused verbatim, never re-judged.
                    reusable = {cell: row for (judge_name, cell), row
                                in resumable.items() if judge_name == ref.name}
                    judgments, judge_report = paired_judge.evaluate(
                        generations, judge_model=requested_model,
                        judge_rubric=rubric,
                        structured_prompt=spec.structured_prompt, judge=judge_fn,
                        on_judgment=_persist,
                        on_invalid=status.note_invalid_response,
                        existing=reusable or None)
                    if judge_report.get("reusedJudgments"):
                        _log(f"judge '{ref.name}': reused "
                             f"{judge_report['reusedJudgments']} judgment(s) from "
                             f"'{resume_from}', judged "
                             f"{judge_report['freshJudgments']} fresh")
                    all_judgments.extend(judgments)
                    outcome_maps.append(
                        (ref.name,
                         {_judgment_key(j): j["outcome"] for j in judgments
                          if not j.get("noncompliant")}))
                    status.note_judge_complete(ref.name)
                    block = {
                        "name": ref.name, "kind": ref.kind,
                        "requestedModel": requested_model,
                        "actualModel": holder["actual"],
                        "conditions": judge_report["conditions"],
                        "pairs": judge_report["pairs"],
                    }
                    for token_key in ("completionTokens", "reasoningTokens"):
                        # Token transparency (2026-08-06): what this judge
                        # actually spent, summed over its column. Provenance
                        # only — no gate reads it. Present just for judges
                        # whose transport reports usage (OpenRouter today).
                        if token_key in judge_report:
                            block[token_key] = judge_report[token_key]
                    if judge_report.get("salvagedVerdicts"):
                        # Salvage visibility (2026-08-06): verdicts in this
                        # column whose winner was regex-rescued from
                        # truncated JSON rather than cleanly parsed. Loud in
                        # the log AND stamped in the report — a column that
                        # is mostly salvage is weaker evidence than a clean
                        # one, and 22/36 salvaged rows on a 2026-08 run
                        # were invisible until the raw JSONL was reread.
                        block["salvagedVerdicts"] = \
                            judge_report["salvagedVerdicts"]
                        _log(f"WARNING: judge '{ref.name}' produced "
                             f"{judge_report['salvagedVerdicts']} of "
                             f"{judge_report['pairs']} verdict(s) via "
                             "truncation salvage (winner legible, reasoning "
                             "cut) — see verdictSalvaged rows in "
                             "judgments.jsonl")
                    if judge_report.get("noncompliantJudgments"):
                        # Noncompliance visibility (2026-08-09): pairs in
                        # this column with NO verdict — recorded as rows,
                        # excluded from every tally and agreement entry.
                        # Nonzero-only, like salvagedVerdicts, and stamped
                        # on the WRITTEN report: a reader must see the
                        # column is incomplete without re-reading
                        # judgments.jsonl.
                        block["noncompliantJudgments"] = \
                            judge_report["noncompliantJudgments"]
                        _log(f"WARNING: judge '{ref.name}' was noncompliant "
                             f"on {judge_report['noncompliantJudgments']} of "
                             f"{judge_report['pairs']} pair(s) — those cells "
                             "carry no verdict (see noncompliant rows in "
                             "judgments.jsonl)")
                    if resume_from:
                        # Per-judge session composition, stamped for EVERY judge
                        # in a resumed run — including judges that reused
                        # nothing, whose whole column is fresh. Deriving both
                        # from the pair count rather than only stamping judges
                        # that happened to reuse keeps the report's totals
                        # equal to the judgments actually written.
                        reused_here = judge_report.get("reusedJudgments", 0)
                        block["reusedJudgments"] = reused_here
                        block["freshJudgments"] = judge_report["pairs"] - reused_here
                    if ref.kind == "local" and holder.get("revision"):
                        # The judge model's pinned revision (JudgeRef.revision /
                        # study-pin fallback, 2026-07-23) — judgment artifacts
                        # name the exact judge bytes.
                        block["revision"] = holder["revision"]
                    if ref.kind == "local" and holder.get("requestedDtype"):
                        # The dtype the manifest PINNED and this load
                        # requested. Stamped so the artifact records what
                        # was asked for, not just what the server happened
                        # to default to (finding 2).
                        block["requestedDtype"] = holder["requestedDtype"]
                    if ref.kind == "local" and holder.get("actualDtype"):
                        # The dtype the judge model actually ran in. The
                        # loader now refuses an unhonored pin outright, so
                        # these agree whenever a pin exists — but an
                        # UNPINNED judge (study-model judges need no pin)
                        # has only this, and it is what makes two judging
                        # sessions on different devices comparable after
                        # the fact (external review round 4, finding 2).
                        block["dtype"] = holder["actualDtype"]
                    if ref.kind == "openrouter":
                        block["provider"] = \
                            paired_judge.canonical_openrouter_provider(ref.provider)
                    # Per-judge credential provenance (engineer review
                    # 2026-07-19): a mixed claude/openrouter panel can judge
                    # through DIFFERENT credential sources, so each external
                    # judge records its own (the report-level judgeCredential
                    # stays as the legacy single-source stamp).
                    if ref.kind != "local":
                        from . import judge_credentials
                        credential = judge_credentials.credential_for(
                            "openrouter" if ref.kind == "openrouter" else "claude")
                        if credential is not None:
                            block["credential"] = {"kind": credential.kind,
                                                   "source": credential.source}
                    judge_blocks.append(block)
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        # The refusal stands; what changes is that the successful judgments
        # SURVIVE it, named and hash-able, with the failing judge and the
        # raw malformed responses beside them. No judge-report.json is
        # written — this directory is a failure record, never a result.
        status.fail(exc)
        _log(f"evaluate FAILED after {status.judgment_count} judgment(s) — "
             f"partial evidence kept in {os.path.basename(out)} "
             f"({type(exc).__name__}: {exc})")
        raise

    report: dict = {
        "experiment": name,
        "sourceRun": os.path.basename(run_dir),
        "rubricFile": rubric_file, "rubricHash": rubric_hash,
        "judges": judge_blocks,
        "agreement": _agreement_entries(outcome_maps),
        "pairs": judge_blocks[0]["pairs"] if judge_blocks else 0,
        # Legacy single-judge keys (first judge) so existing readers keep
        # working; per-judge truth lives in "judges".
        "conditions": judge_blocks[0]["conditions"] if judge_blocks else {},
        "judgeModel": judge_blocks[0]["requestedModel"] if judge_blocks else None,
        "requestedJudgeModel": judge_blocks[0]["requestedModel"] if judge_blocks else None,
        "actualJudgeModel": judge_blocks[0]["actualModel"] if judge_blocks else None,
    }
    noncompliant_total = sum(
        b.get("noncompliantJudgments", 0) for b in judge_blocks)
    if noncompliant_total:
        # Panel total, nonzero-only — the paired-judge sibling of the coding
        # report's noncompliantCodings (and the cross-engine report-level
        # key): per-judge truth lives in "judges", but the report itself
        # must say the evaluation has holes and how many.
        report["noncompliantJudgments"] = noncompliant_total
    # Where the judging ran + through which credential is RECORDED
    # provenance (2026-07-19), mirroring the sweep selection blocks.
    report["judgedOn"] = "server"
    # Where the effective SPEC came from (2026-07-22): "manifest" = explicit
    # evaluation block; "pinnedRubric" = synthesized from the pinned judges
    # + rubric file. Cross-engine key — the Swift report stamps the same.
    report["evaluationSource"] = evaluation_source
    if resume_from:
        # A multi-session evaluation is a DIFFERENT evidentiary object from
        # one judged in a single sitting, and the report says so rather than
        # leaving the reader to infer it from per-judge counts.
        reused = sum(b.get("reusedJudgments", 0) for b in judge_blocks)
        report["judgingSessions"] = {
            "resumedFrom": resume_from,
            "reusedJudgments": reused,
            "freshJudgments": sum(b.get("freshJudgments", 0)
                                  for b in judge_blocks),
            # Named because it is the reason this needs a stamp: these
            # judges are not revision-pinned, so "the same judge" across
            # two sessions is an assumption, not a fact.
            "unpinnedExternalJudges": sorted(
                {r.name for r in roster if r.kind != "local"}),
        }
        _log(f"evaluate completed by resuming '{resume_from}': {reused} "
             "judgment(s) reused")
    if exclusion_stamp is not None:
        # The identical stamp shape analyze writes; excluded records were
        # filtered BEFORE judging, so their judge calls never happened.
        report["exclusions"] = exclusion_stamp
    if any(ref.kind != "local" for ref in roster):
        from . import judge_credentials
        try:
            credential = judge_credentials.resolve()
        except ValueError:
            credential = None
        if credential is not None:
            report["judgeCredential"] = {"kind": credential.kind,
                                         "source": credential.source}
    if epoch_unverified:
        report["epochUnverified"] = True
    if measurement_drift:
        # Tolerated measurement-side drift is never silent: which fields
        # differed from the source run's epoch, verbatim.
        report["measurementDrift"] = measurement_drift
    if human is not None:
        report["humanValidation"] = {"path": manifest.human_validation.path,
                                     "hash": manifest.human_validation.hash,
                                     "rows": len(human)}
        report["humanAgreement"] = [
            {"judge": entry["judges"][1], "n": entry["n"],
             "percentAgreement": entry["percentAgreement"],
             "kappa": entry["kappa"]}
            for entry in _agreement_entries(
                [("human", _materialize_human_validation(human, outcome_maps))]
                + outcome_maps)
            if entry["judges"][0] == "human"]

    # judgments.jsonl was written row-by-row as the panel judged; the report
    # is what makes this directory a RESULT rather than a failure record.
    with open(os.path.join(out, "judge-report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    status.complete()
    _log(f"evaluate ({len(all_judgments)} judgments, {len(roster)} judge(s)) → {out}")
    return out


#: The custody decision itself lives in the torch-free
#: :mod:`judging_custody` (2026-08-20) so the freeze advisory and the
#: submission preflight can ask it without importing this module. These are
#: the historical spellings, unchanged for every caller and test here.
_missing_external_credentials = judging_custody.missing_external_credentials
judging_custody_plan = judging_custody.custody_plan


def log_judging_custody(roster, log) -> dict:
    """Announce the custody plan at stage start. One line, always — a
    researcher should never learn where judging ran by inspecting the
    artifacts afterwards."""
    plan = judging_custody_plan(roster)
    log(f"judging custody: {plan['disposition'].upper()} — {plan['reason']}")
    return plan


def _emit_evaluate_judging(name, manifest, spec, source_run_dir, generations,
                           rubric, rubric_hash, rubric_file, root,
                           epoch_unverified, measurement_drift, _log, *,
                           exclusion_stamp=None,
                           evaluation_source=None) -> str:
    """Deferred evaluate, phase 1: pair the source run's generations with
    their baselines, blind them with EXACTLY the inline convention
    (``paired_judge._baseline_first`` over (promptID, condition)), and emit
    hash-pinned judging packets for the Mac. The judge-visible packets
    carry only prompt + responses; identity and orientation live in the map
    the judging client never consumes. ``generations`` arrive already
    filtered by any declared exclusion rules (no packet is emitted for an
    excluded record); the caller's stamp is recorded as exclusions.json."""
    from . import paired_judge
    pairs = paired_judge._pair_generations(generations)
    if not pairs:
        # Belt-and-braces: evaluate() already refused before the custody
        # fork; the shared message keeps the two paths in one family.
        raise RuntimeError(paired_judge.NO_PAIRS_MESSAGE)
    frozen_needs = manifest.status != "draft"
    if frozen_needs and not (manifest.judge_rubric_file
                             and manifest.judge_rubric_hash):
        # _resolve_rubric already enforced this upstream; belt-and-braces.
        raise RuntimeError("frozen evaluation must judge from a pinned rubric")
    packets: list[dict] = []
    packet_map: dict[str, dict] = {}
    for pair in pairs:
        baseline_is_a = paired_judge._baseline_first(
            str(pair["promptID"]), pair["condition"])
        a, b = ((pair["baseline"], pair["variant"]) if baseline_is_a
                else (pair["variant"], pair["baseline"]))
        packet_id = hashlib.sha256(
            f"evaluate:{pair['condition']}|{pair['promptID']}|"
            f"{pair['sampleIndex']}|{rubric_hash}|{a}|{b}".encode("utf-8")
        ).hexdigest()
        packets.append({"packetID": packet_id,
                        "prompt": pair.get("prompt", ""),
                        "responseA": a, "responseB": b})
        packet_map[packet_id] = {
            "promptID": pair["promptID"],
            "sampleIndex": pair["sampleIndex"],
            # Seed provenance for both sides (cross-engine keys) — under
            # derived seeding the two sides never share a seed, so a single
            # "seed" field cannot exist on a pair.
            "baselineSeed": pair["baselineSeed"],
            "variantSeed": pair["variantSeed"],
            "condition": pair["condition"], "baselineIsA": baseline_is_a}

    run_directory = paths.make_unique_run_directory(
        f"exp-{name}-evaluate", root)
    write_run_config(run_directory, "evaluate-awaiting",
                     model_id=manifest.model_id,
                     revision=manifest.model_revision, experiment=name,
                     experiment_hash=manifest.content_hash(),
                     notes=({**({"epochUnverified": True}
                                if epoch_unverified else {}),
                             **({"measurementDrift": measurement_drift}
                                if measurement_drift else {})} or None))
    if exclusion_stamp is not None:
        # Recorded at emission time: which records never became packets,
        # and why (the same stamp shape the inline path and analyze write).
        with open(os.path.join(run_directory, "exclusions.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(exclusion_stamp, handle, indent=2, sort_keys=True)
    packets_path = os.path.join(run_directory, "judging-packets.jsonl")
    with open(packets_path, "w", encoding="utf-8") as handle:
        for packet in packets:
            handle.write(json.dumps(packet, sort_keys=True) + "\n")
    map_path = os.path.join(run_directory, "judging-map.json")
    with open(map_path, "w", encoding="utf-8") as handle:
        json.dump({"packets": packet_map}, handle, indent=2, sort_keys=True)

    def _sha256_of(path: str) -> str:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    structured = (spec.structured_prompt or "").strip() or None
    judge_entries = _normalized_judge_entries(manifest.raw.get("judges") or [],
                                              study_model=manifest.model_id)
    # The agent-facing framing is an ENGINE artifact, not a per-campaign
    # hand-written prompt (Cowork judging pipeline, 2026-08-11): rendered
    # from the same pinned inputs as the packets, hashed into the emission
    # record, and verified back at complete-judgment intake. It sees the
    # rubric and the pinned panel — never the map's contents.
    from . import judging_instructions
    instructions_path = os.path.join(
        run_directory, judging_instructions.INSTRUCTIONS_FILENAME)
    with open(instructions_path, "w", encoding="utf-8") as handle:
        handle.write(judging_instructions.render(
            experiment=name, evaluate_run=os.path.basename(run_directory),
            packets_file="judging-packets.jsonl", packet_count=len(packets),
            rubric=rubric, structured_prompt=structured,
            judges=judge_entries))
    judging_manifest = {
        "kind": "evaluate",
        "experiment": name,
        "experimentHash": manifest.content_hash(),
        "evaluateRun": os.path.basename(run_directory),
        "sourceRun": os.path.basename(source_run_dir),
        "sourceGenerationsSha256": _sha256_of(
            os.path.join(source_run_dir, "generations.jsonl")),
        "rubricFile": rubric_file,
        "rubricHash": rubric_hash,
        "rubric": rubric,
        "rubricTextSha256": hashlib.sha256(
            rubric.encode("utf-8")).hexdigest(),
        "judges": judge_entries,
        "packetsFile": "judging-packets.jsonl",
        "packetsSha256": _sha256_of(packets_path),
        "mapSha256": _sha256_of(map_path),
        "instructionsFile": judging_instructions.INSTRUCTIONS_FILENAME,
        "instructionsSha256": _sha256_of(instructions_path),
        "packetCount": len(packets),
    }
    if evaluation_source is not None:
        # Spec provenance rides with the packets so the COMPLETED report
        # carries the same "evaluationSource" stamp the inline path writes.
        judging_manifest["evaluationSource"] = evaluation_source
    if structured is not None:
        # The structured prompt is part of the evaluation criterion — the
        # Mac judges with it and verifies its pin like the rubric's.
        judging_manifest["structuredPrompt"] = structured
        judging_manifest["structuredPromptSha256"] = hashlib.sha256(
            structured.encode("utf-8")).hexdigest()
    if epoch_unverified:
        judging_manifest["epochUnverified"] = True
    if measurement_drift:
        # Travels with the packets so the fan-out merge stamps the final
        # report the same way the inline path does.
        judging_manifest["measurementDrift"] = measurement_drift
    with open(os.path.join(run_directory, "judging-manifest.json"), "w",
              encoding="utf-8") as handle:
        json.dump(judging_manifest, handle, indent=2, sort_keys=True)
    with open(os.path.join(run_directory, "awaiting-judgment.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"packetCount": len(packets),
                   "judgingManifest": "judging-manifest.json"},
                  handle, indent=2, sort_keys=True)
    _log(f"no credential for the pinned external judges — emitted "
         f"{len(packets)} blinded judging packets → {run_directory} "
         "(judge on the Mac, hand judging-instructions.md to an agent "
         "orchestrator, or push a judge key for inline judging)")
    return run_directory


JUDGE_FANOUT_REQUEST_FILE = "judge-fanout-request.json"


def evaluate_fanout_judge_models(manifest: Manifest) -> list[dict]:
    """The distinct LOCAL judge models an evaluate fan-out needs, grouped by
    resolved model — ``[{model, revision, dtype, judges: [names]}]`` — or []
    when no local judge resolves to a model other than the study model (the
    inline path then suffices). When ANY local judge needs the fan-out, ALL
    local judges fan out (a study-model worker included), so the merged
    report has ONE judging clock. Revisions: the judge's own pin, else the
    study pin for study-model judges (JudgeRef contract, 2026-07-23)."""
    from . import sweep_selection
    locals_ = [ref for ref in (manifest.judges or []) if ref.kind == "local"]
    if not locals_:
        return []
    resolved = [(ref, sweep_selection.resolve_local_judge_model(
        ref.model, manifest.model_id)) for ref in locals_]
    if all(model == manifest.model_id for _ref, model in resolved):
        return []
    grouped: dict[tuple, dict] = {}
    for ref, model in resolved:
        revision = (ref.revision
                    or (manifest.model_revision if model == manifest.model_id
                        else None))
        dtype = ref.dtype
        key = (model, revision, dtype)
        entry = grouped.setdefault(key, {"model": model,
                                         "revision": revision,
                                         "dtype": dtype, "judges": []})
        entry["judges"].append(ref.name)
    return [grouped[key] for key in sorted(grouped, key=lambda k:
                                           (k[0], k[1] or "", k[2] or ""))]


def write_judge_fanout_request(pipeline_directory: str, name: str,
                               manifest: Manifest, awaiting_dir: str) -> dict:
    """The controller-facing fan-out request, written into the PIPELINE
    directory when the evaluate stage emitted packets for local judge
    workers: which awaiting run, which distinct judge models (with pinned
    revisions/dtypes and the judge names each covers), and the packet pin.
    The reconciler reads it off the continuation's child record and submits
    one worker job per entry."""
    with open(os.path.join(awaiting_dir, "judging-manifest.json"),
              encoding="utf-8") as handle:
        jm = json.load(handle)
    request = {
        "schema": 1,
        "experiment": name,
        "evaluateRun": os.path.basename(awaiting_dir),
        "pipelineDirectory": pipeline_directory,
        "packetsSha256": jm.get("packetsSha256"),
        "packetCount": jm.get("packetCount"),
        "judgeModels": evaluate_fanout_judge_models(manifest),
    }
    with open(os.path.join(pipeline_directory, JUDGE_FANOUT_REQUEST_FILE),
              "w", encoding="utf-8") as handle:
        json.dump(request, handle, indent=2, sort_keys=True)
    return request


def read_judge_fanout_request(pipeline_directory: str) -> dict | None:
    """The pipeline directory's fan-out request, or None."""
    path = os.path.join(pipeline_directory, JUDGE_FANOUT_REQUEST_FILE)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def judge_worker(name: str, awaiting_run: str, model: str,
                 revision: str | None = None, dtype: str = "auto",
                 device: str | None = None, out_path: str | None = None,
                 root: str | None = None, log=None,
                 generate_fn=None) -> dict:
    """One judge-model worker of the post-generation judge fan-out
    (2026-07-23): load THIS judge model (at its pinned revision), judge
    EVERY packet of the awaiting evaluate run for every pinned local judge
    that resolves to this model, and write one hash-pinned judgment
    artifact with deterministic bytes. The controller merges the artifacts
    through ``complete_evaluate_judgment`` only when every judge ×
    response-pair cell appears exactly once.

    ``generate_fn`` is the test seam: ``generate_fn(prompt) -> text``
    replaces the model load + generate (fake judge models in tests). The
    artifact rows carry exactly the fields the completion verb verifies
    (packetID, judge, model, winner, confidence, judgment payload)."""
    from . import paired_judge, sweep_selection
    _log = log or print
    manifest = Manifest.load(name, root)
    runs_root = paths.runs_directory(root)
    eval_dir = os.path.join(runs_root, awaiting_run)
    jm = _evaluate_judging_manifest(runs_root, awaiting_run)
    if jm is None:
        raise ValueError(
            f"run '{awaiting_run}' has no evaluate judging manifest — not "
            "an awaiting-judgment evaluate run")
    if jm.get("experiment") != name:
        raise ValueError(
            f"run '{awaiting_run}' belongs to experiment "
            f"'{jm.get('experiment')}', not '{name}'")
    packets_path = os.path.join(eval_dir, jm.get("packetsFile")
                                or "judging-packets.jsonl")
    with open(packets_path, "rb") as handle:
        packet_bytes = handle.read()
    packets_sha = hashlib.sha256(packet_bytes).hexdigest()
    if jm.get("packetsSha256") and packets_sha != jm["packetsSha256"]:
        raise ValueError(
            "the judging packets drifted since emission (sha256 mismatch) — "
            "the awaiting run directory must be immutable")
    packets = [json.loads(line) for line
               in packet_bytes.decode("utf-8").splitlines() if line.strip()]
    # The judges THIS worker covers: pinned local judges whose resolved
    # model is this worker's model.
    judge_names = []
    for entry in jm.get("judges") or []:
        if (entry.get("kind") or "claude") != "local":
            continue
        resolved = sweep_selection.resolve_local_judge_model(
            entry.get("model"), manifest.model_id)
        if resolved == model:
            judge_names.append(str(entry.get("name")))
    if not judge_names:
        raise ValueError(
            f"no pinned local judge resolves to model '{model}' — nothing "
            "for this worker to judge")
    rubric = str(jm.get("rubric") or "")
    structured = jm.get("structuredPrompt")
    if generate_fn is None:
        slot = model_loader.load(model, revision=revision, dtype=dtype,
                                 device=device)

        def generate_fn(prompt: str) -> str:  # noqa: PLR0913 - closure
            return generate(slot, prompt, model_id=model,
                            max_tokens=paired_judge.JUDGE_MAX_TOKENS,
                            temperature=0.0,
                            prompt_mode=prompt_render.CHAT_ASSISTANT)

    judge_fn = paired_judge.make_local_judge(generate_fn)
    rows: list[dict] = []
    for judge_name in sorted(judge_names):
        for packet in sorted(packets, key=lambda p: str(p.get("packetID"))):
            verdict = paired_judge.valid_verdict(
                judge_fn, model, rubric, packet.get("responseA", ""),
                packet.get("responseB", ""), structured,
                task_prompt=packet.get("prompt") or None,
                judge_label=f"'{judge_name}' (model '{model}')",
                item_label=f"packet {str(packet.get('packetID'))[:16]}…")
            rows.append({
                "packetID": packet.get("packetID"),
                "judge": judge_name,
                "model": model,
                "winner": verdict.get("winner"),
                "confidence": verdict.get("confidence"),
                "judgment": verdict,
            })
    artifact = {
        "schema": 1,
        "kind": "judgeWorkerJudgments",
        "experiment": name,
        "evaluateRun": awaiting_run,
        "packetsSha256": packets_sha,
        "judgeModel": model,
        **({"revision": revision} if revision else {}),
        **({"dtype": dtype} if dtype and dtype != "auto" else {}),
        "judges": sorted(judge_names),
        "rows": rows,
    }
    payload = json.dumps(artifact, indent=2, sort_keys=True)
    result = {"experiment": name, "evaluateRun": awaiting_run,
              "judgeModel": model, "judges": sorted(judge_names),
              "judgments": len(rows),
              "artifactSha256": hashlib.sha256(
                  payload.encode("utf-8")).hexdigest()}
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
        result["artifactPath"] = out_path
        _log(f"judge worker ({model}): {len(rows)} judgments by "
             f"{', '.join(sorted(judge_names))} → {out_path}")
    result["rows"] = rows
    return result


def read_judge_worker_artifact(path: str, *, expected_packets_sha:
                               str | None = None) -> dict:
    """A worker's judgment artifact, verified against the awaiting run's
    packet pin — a worker that judged DIFFERENT packets contributes
    nothing mergeable."""
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    if artifact.get("kind") != "judgeWorkerJudgments":
        raise ValueError(f"'{path}' is not a judge-worker judgment artifact")
    if (expected_packets_sha
            and artifact.get("packetsSha256") != expected_packets_sha):
        raise ValueError(
            f"judge-worker artifact '{path}' judged packets with a "
            "different hash than the awaiting run's pin — refusing to merge")
    return artifact


def list_awaiting_evaluate_judgment(name: str,
                                    root: str | None = None) -> list[dict]:
    """Evaluate runs for ``name`` that emitted judging packets and have no
    VERIFIED completion run referencing them yet — same suppression rule as
    the sweep scanner (only a verified record hides judgable work)."""
    runs_root = paths.runs_directory(root)
    try:
        entries = sorted(os.listdir(runs_root))
    except OSError:
        return []
    completed: set[str] = set()
    for entry in entries:
        src = os.path.join(runs_root, entry, "judgment-source.json")
        if not os.path.exists(src):
            continue
        try:
            with open(src, encoding="utf-8") as handle:
                marker = json.load(handle)
        except (OSError, ValueError):
            continue
        if marker.get("kind") != "evaluate":
            continue
        ref = str(marker.get("evaluateRun"))
        try:
            _verify_evaluate_marker(
                os.path.join(runs_root, entry), marker, name=name,
                eval_jm=_evaluate_judging_manifest(runs_root, ref))
        except ValueError:
            continue
        completed.add(ref)
    out: list[dict] = []
    for entry in entries:
        jm_path = os.path.join(runs_root, entry, "judging-manifest.json")
        if not os.path.exists(jm_path):
            continue
        try:
            with open(jm_path, encoding="utf-8") as handle:
                jm = json.load(handle)
        except (OSError, ValueError):
            continue
        if (jm.get("experiment") != name or jm.get("kind") != "evaluate"
                or entry in completed):
            continue
        out.append({
            "run": entry,
            "kind": "evaluate",
            "sourceRun": jm.get("sourceRun"),
            "packetCount": jm.get("packetCount"),
            "judges": jm.get("judges"),
            "rubricFile": jm.get("rubricFile"),
            "rubricHash": jm.get("rubricHash"),
            "rubricTextSha256": jm.get("rubricTextSha256"),
            "rubric": jm.get("rubric"),
            "structuredPrompt": jm.get("structuredPrompt"),
            "structuredPromptSha256": jm.get("structuredPromptSha256"),
            "experimentHash": jm.get("experimentHash"),
            "packetsFile": jm.get("packetsFile"),
            "packetsSha256": jm.get("packetsSha256"),
            "instructionsFile": jm.get("instructionsFile"),
            "instructionsSha256": jm.get("instructionsSha256"),
        })
    return out


def _evaluate_judging_manifest(runs_root: str,
                               evaluate_run: str) -> dict | None:
    """``evaluate_run``'s judging manifest, or None when unreadable — the
    verifier then REFUSES (fail closed, same rule as sweeps)."""
    try:
        with open(os.path.join(runs_root, evaluate_run,
                               "judging-manifest.json"),
                  encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(loaded, dict) or loaded.get("kind") != "evaluate":
        return None
    return loaded


def _verify_evaluate_marker(run_dir: str, marker: dict, *, name: str,
                            eval_jm: dict | None) -> None:
    """Strict verification of an evaluate-judgment completion record —
    the same discipline as `_verify_judgment_marker`: schema-versioned,
    bound to the experiment and the emission's packet pin + epoch, and
    hash-linked to the run's own artifacts. Raises ValueError."""
    run = os.path.basename(run_dir)

    def _refuse(why: str):
        raise ValueError(
            f"evaluate completion record in '{run}' failed verification "
            f"({why}) — an unverified judgment run is never canonical; "
            "inspect it by hand (a genuine one re-heals by re-running "
            "completion)")

    if marker.get("schema") != JUDGMENT_MARKER_SCHEMA:
        _refuse(f"schema {marker.get('schema')!r}, expected "
                f"{JUDGMENT_MARKER_SCHEMA}")
    if marker.get("kind") != "evaluate":
        _refuse(f"kind {marker.get('kind')!r}, expected 'evaluate'")
    if marker.get("experiment") != name:
        _refuse(f"experiment {marker.get('experiment')!r}, expected {name!r}")
    if eval_jm is None:
        _refuse("the evaluate run's judging manifest cannot be read — no "
                "packet pin to verify against")
    if marker.get("packetsSha256") != eval_jm.get("packetsSha256"):
        _refuse("packet pin does not match the evaluate run's judging "
                "manifest")
    if marker.get("experimentHashAtJudgment") != eval_jm.get("experimentHash"):
        _refuse("judgment epoch does not match the evaluate run's "
                "experiment hash")
    for artifact, key in (("judgments.jsonl", "judgmentsSha256"),
                          ("judge-report.json", "judgeReportSha256")):
        stamped = marker.get(key)
        if not stamped:
            _refuse(f"no {key} stamp")
        path = os.path.join(run_dir, artifact)
        try:
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            _refuse(f"{artifact} is missing")
        if digest != stamped:
            _refuse(f"{artifact} does not hash to its stamp")


def _find_evaluate_judgment_run(name: str, evaluate_run: str,
                                root: str | None) -> tuple[str, dict] | None:
    """``(run_dir, VERIFIED marker)`` of the completed judgment run for
    ``evaluate_run``, else None — unreadable unrelated markers skip, a
    matching candidate that fails verification RAISES (sweep rule)."""
    runs_root = paths.runs_directory(root)
    try:
        entries = sorted(os.listdir(runs_root))
    except OSError:
        return None
    for entry in entries:
        src = os.path.join(runs_root, entry, "judgment-source.json")
        if not os.path.exists(src):
            continue
        try:
            with open(src, encoding="utf-8") as handle:
                marker = json.load(handle)
        except (OSError, ValueError):
            continue
        if (marker.get("kind") != "evaluate"
                or marker.get("evaluateRun") != evaluate_run):
            continue
        run_dir = os.path.join(runs_root, entry)
        _verify_evaluate_marker(
            run_dir, marker, name=name,
            eval_jm=_evaluate_judging_manifest(runs_root, evaluate_run))
        return run_dir, marker
    return None


def _instructions_intake_stamp(eval_dir: str, jm: dict,
                               claimed: str | None, log) -> dict | None:
    """The ``judgingInstructions`` stamp for a completed judgment report
    (Cowork judging pipeline, 2026-08-11): verify the instructions hash the
    judging client CLAIMS against the emission's ``instructionsSha256`` and
    against the live file bytes. Any mismatch is a LOUD warning stamped
    into the report — never a refusal (post-submit drift policy: verdicts
    already produced are evidence about what the campaign actually did; the
    stamp makes the framing question checkable after the fact instead of
    silently unanswerable). Returns None only when neither side has
    anything to say (legacy emission AND no claim) — legacy behavior stays
    byte-identical."""
    emitted = str(jm.get("instructionsSha256") or "").strip().lower() or None
    claimed = str(claimed or "").strip().lower() or None
    if emitted is None and claimed is None:
        return None
    stamp: dict = {
        "file": jm.get("instructionsFile") or "judging-instructions.md",
        "emittedSha256": emitted,
        "claimedSha256": claimed,
        "verified": bool(emitted and claimed and claimed == emitted),
    }
    if emitted:
        try:
            with open(os.path.join(eval_dir, stamp["file"]), "rb") as handle:
                live = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            live = None
        if live != emitted:
            stamp["fileDrifted"] = True
            log(f"WARNING: '{stamp['file']}' in the awaiting run "
                "drifted (or went missing) since emission — the emission "
                "stamp remains the pin the claim is verified against")
    if emitted and claimed and claimed != emitted:
        log(f"WARNING: the judgments claim instructions "
            f"{claimed[:12]}… but the emission stamped {emitted[:12]}… — "
            "the judging campaign read DIFFERENT instructions than this "
            "run emitted; completing anyway (post-submit drift policy) "
            "and stamping judgingInstructions.verified: false")
    elif claimed and not emitted:
        log("WARNING: the judgments claim an instructions hash but this "
            "awaiting run's emission stamped none (legacy emission) — "
            "recorded unverified")
    elif emitted and not claimed:
        log("note: the judging client claimed no instructions hash — "
            "stamped judgingInstructions.verified: false (the Mac app "
            "client does not read the instructions file; agent-"
            "orchestrated campaigns should claim it)")
    return stamp


def complete_evaluate_judgment(name: str, evaluate_run: str, judgments: list,
                               root: str | None = None, log=None,
                               instructions_sha256: str | None = None) -> str:
    """Phase 2 of a deferred evaluate: verify the Mac's judgments against
    the emission pins (packets hash, judge panel, full coverage, experiment
    epoch), unblind through the map, and aggregate into the SAME
    judgments.jsonl + judge-report.json shapes the inline path writes —
    per-judge tallies, inter-judge agreement, and judge-vs-human agreement
    when a humanValidation subset is pinned. CPU-only; idempotent (a
    completed run returns itself).

    ``instructions_sha256`` is the judging client's claim of which
    ``judging-instructions.md`` its campaign judged under — verified
    against the emission stamp and recorded as the report's
    ``judgingInstructions`` block (mismatch warns loudly, never refuses)."""
    from . import paired_judge
    _log = log or print
    if not evaluate_run or "/" in evaluate_run or os.sep in evaluate_run \
            or evaluate_run in (".", ".."):
        raise ValueError(f"bad evaluate run name {evaluate_run!r}")
    manifest = Manifest.load(name, root)
    _verify_or_warn(manifest, root)
    existing = _find_evaluate_judgment_run(name, evaluate_run, root)
    if existing is not None:
        run_directory, _marker = existing
        _log(f"evaluate judgment already completed → {run_directory}")
        return run_directory
    runs_root = paths.runs_directory(root)
    eval_dir = os.path.join(runs_root, evaluate_run)
    jm = _evaluate_judging_manifest(runs_root, evaluate_run)
    if jm is None:
        raise ValueError(
            f"run '{evaluate_run}' has no evaluate judging manifest — not "
            "an awaiting-judgment evaluate run")
    if jm.get("experiment") != name:
        raise ValueError(
            f"run '{evaluate_run}' belongs to experiment "
            f"'{jm.get('experiment')}', not '{name}'")
    # The manifest must name the run directory it lives in (engineer review
    # 2026-07-18, emission-pin pass) — a copied/renamed awaiting dir must
    # not complete under another run's identity.
    if jm.get("evaluateRun") != evaluate_run:
        raise ValueError(
            f"the judging manifest names evaluate run "
            f"'{jm.get('evaluateRun')}' but lives in '{evaluate_run}' — "
            "refusing a relocated awaiting run")
    live_hash = manifest.content_hash()
    if jm.get("experimentHash") != live_hash:
        raise ValueError(
            f"experiment epoch mismatch: the packets were emitted under "
            f"manifest hash {str(jm.get('experimentHash'))[:12]}… but "
            f"'{name}' now hashes {live_hash[:12]}… — judgments cannot "
            "complete a report for a drifted manifest (re-run evaluate)")

    def _verify_pinned_file(path: str, stamped: str | None,
                            what: str) -> bytes:
        if not stamped:
            raise ValueError(
                f"the judging manifest carries no hash for {what} — "
                "re-run evaluate")
        with open(path, "rb") as handle:
            data = handle.read()
        digest_ = hashlib.sha256(data).hexdigest()
        if digest_ != stamped:
            raise ValueError(
                f"{what} drifted since emission (sha256 mismatch) — the "
                "evaluate run directory must be immutable; re-run evaluate")
        return data

    _verify_pinned_file(
        os.path.join(eval_dir, jm.get("packetsFile")
                     or "judging-packets.jsonl"),
        jm.get("packetsSha256"), "the judging packets file")
    map_bytes = _verify_pinned_file(
        os.path.join(eval_dir, "judging-map.json"), jm.get("mapSha256"),
        "the judging map")
    packet_map = json.loads(map_bytes.decode("utf-8"))["packets"]

    # Which instructions the judging campaign judged under — verified and
    # stamped, never refused (see _instructions_intake_stamp).
    instructions_stamp = _instructions_intake_stamp(
        eval_dir, jm, instructions_sha256, _log)

    # The SOURCE generations the packets were cut from must still hash to
    # their emission pin (engineer review 2026-07-18, emission-pin pass) —
    # a judgment set is evidence about THOSE generations, and a drifted or
    # missing source run breaks the chain from report back to raw outputs.
    source_run = str(jm.get("sourceRun") or "")
    source_generations = os.path.join(runs_root, source_run,
                                      "generations.jsonl")
    if not os.path.exists(source_generations):
        raise ValueError(
            f"source run '{source_run}' has no generations.jsonl on this "
            "server — the judged generations must remain present and "
            "immutable for the completion to bind evidence to them")
    _verify_pinned_file(source_generations,
                        jm.get("sourceGenerationsSha256"),
                        "the source run's generations")

    # The structured prompt is part of the evaluation criterion: the text
    # must hash to its own emission pin (what the Mac judged with), and must
    # still be the LIVE manifest's structured prompt (the epoch gate makes
    # this belt-and-braces, but the criterion deserves its own loud check).
    emitted_structured = jm.get("structuredPrompt")
    if emitted_structured is not None:
        stamped = jm.get("structuredPromptSha256")
        digest_ = hashlib.sha256(
            str(emitted_structured).encode("utf-8")).hexdigest()
        if not stamped or digest_ != stamped:
            raise ValueError(
                "the judging manifest's structured prompt does not hash to "
                "its emission pin — re-run evaluate")
    live_spec, _live_source = manifest.effective_evaluation()
    live_structured = ((live_spec.structured_prompt or "").strip()
                       or None) if live_spec else None
    if (emitted_structured or None) != live_structured:
        raise ValueError(
            "the emitted structured prompt does not match the live "
            "manifest's evaluation.structuredPrompt — re-run evaluate")

    # Rubric agreement with the live manifest (epoch above proved identity).
    if jm.get("rubricHash") != manifest.judge_rubric_hash:
        raise ValueError(
            "the judging manifest's rubric hash does not match the "
            "manifest's pinned judgeRubricHash — re-run evaluate")
    if manifest.judge_rubric_file:
        _resolve_rubric(manifest, root, lambda *_: None)  # file-drift check
    # The EMITTED panel is authoritative for models (sweep rule): compare
    # structure to the live manifest, accept emission-resolved defaults.
    live_raw = [dict(j or {}) for j in (manifest.raw.get("judges") or [])]
    jm_judges = [dict(j or {}) for j in (jm.get("judges") or [])]
    if live_raw:
        if len(live_raw) != len(jm_judges):
            raise ValueError(
                "the judging manifest's judge panel does not match the "
                "live manifest's pinned judges — re-run evaluate")
        for raw_j, pinned in zip(live_raw, jm_judges):
            if (str(raw_j.get("name")) != str(pinned.get("name"))
                    or (raw_j.get("kind") or "claude")
                    != (pinned.get("kind") or "claude")):
                raise ValueError(
                    "the judging manifest's judge panel does not match the "
                    "live manifest's pinned judges — re-run evaluate")
            declared = str(raw_j.get("model") or "").strip()
            if declared and declared != str(pinned.get("model")):
                raise ValueError(
                    f"judge '{raw_j.get('name')}' declares model "
                    f"'{declared}' but the emission pinned "
                    f"'{pinned.get('model')}' — re-run evaluate")
            declared_provider = str(raw_j.get("provider") or "").strip()
            if declared_provider and declared_provider != str(
                    pinned.get("provider") or ""):
                raise ValueError(
                    f"judge '{raw_j.get('name')}' declares provider "
                    f"'{declared_provider}' but the emission pinned "
                    f"'{pinned.get('provider')}' — re-run evaluate")
    pinned_model_by_judge = {str(j.get("name")): str(j.get("model"))
                             for j in jm_judges}
    pinned_provider_by_judge = {
        str(j.get("name")): (str(j.get("provider") or "").strip() or None
                             if (j.get("kind") or "claude") == "openrouter"
                             else None)
        for j in jm_judges}
    judge_names = set(pinned_model_by_judge)

    seen: dict[tuple[str, str],
               tuple[str, str, float | None, str | None, dict | None,
                     str | None]] = {}
    for row in judgments:
        row = row or {}
        pid = str(row.get("packetID") or "")
        judge = str(row.get("judge") or "")
        winner = row.get("winner")
        if pid not in packet_map:
            raise ValueError(f"judgment for unknown packet '{pid[:16]}…'")
        if judge not in judge_names:
            raise ValueError(
                f"judgment by unpinned judge {judge!r} — the emission "
                f"pinned {sorted(judge_names)}")
        if winner not in ("A", "B", "tie"):
            raise ValueError(
                f"judgment winner must be 'A', 'B', or 'tie', got {winner!r}")
        model = str(row.get("model") or "").strip()
        if not model:
            raise ValueError(
                f"judgment by '{judge}' carries no model — the judging "
                "client must stamp the emission-pinned model it used")
        if model != pinned_model_by_judge[judge]:
            raise ValueError(
                f"judgment by '{judge}' used model '{model}' but the "
                f"emission pinned '{pinned_model_by_judge[judge]}' — "
                "refusing off-pin judgments")
        provider = _verify_judgment_provider(
            row, judge, pinned_provider_by_judge.get(judge))
        # The model the judging AGENT itself ran on (Cowork judging
        # pipeline, 2026-08-11) — provenance distinct from the pinned
        # judge model, recorded per judgment so cross-model annotation
        # agreement stays computable. Optional; a present value must be a
        # non-empty string (a recorded field is provenance only if it
        # says something).
        annotator = row.get("annotatorModel")
        if annotator is not None and (not isinstance(annotator, str)
                                      or not annotator.strip()):
            raise ValueError(
                f"judgment by {judge!r} carries a non-string or empty "
                "annotatorModel — omit the field or record the actual "
                "model string the judging agent ran on")
        key = (pid, judge)
        if key in seen:
            raise ValueError(
                f"duplicate judgment for packet '{pid[:16]}…' by '{judge}'")
        confidence, verdict = _verified_judgment_payload(
            row, judge, str(winner))
        seen[key] = (str(winner), model, confidence, provider, verdict,
                     annotator.strip() if annotator else None)
    expected = {(pid, judge) for pid in packet_map for judge in judge_names}
    missing_pairs = expected - set(seen)
    if missing_pairs:
        raise ValueError(
            f"incomplete judgment set: {len(missing_pairs)} of "
            f"{len(expected)} (packet × judge) pairs missing — the "
            "completion verb requires full coverage")

    # PHASE A — aggregate in memory (inline shapes exactly).
    all_judgments: list[dict] = []
    judge_blocks: list[dict] = []
    outcome_maps: list[tuple[str, dict]] = []
    for judge in sorted(judge_names):
        tally: dict[str, dict] = {}
        judge_rows: list[dict] = []
        for pid, meta in packet_map.items():
            (winner, model, confidence, provider, verdict,
             annotator_model) = seen[(pid, judge)]
            baseline_is_a = bool(meta["baselineIsA"])
            if winner == "tie":
                outcome = "tie"
            else:
                baseline_won = (winner == "A") == baseline_is_a
                outcome = "baseline" if baseline_won else "variant"
            judge_row = {
                "promptID": meta["promptID"],
                # Legacy awaiting runs (pre sample-cell join) mapped by
                # "seed" only; their completed rows normalize to cell 0 and
                # carry no per-side seed provenance.
                "sampleIndex": meta.get("sampleIndex", 0),
                "condition": meta["condition"],
                "baselineWas": "A" if baseline_is_a else "B",
                "outcome": outcome, "confidence": confidence,
                # The judge's FULL verdict when the judging client sent it
                # (winner-only closure 2026-07-20): the same object the
                # inline path records, winner-consistency-verified at
                # intake. None on rows from older clients.
                "judgment": verdict, "judge": judge, "judgeModel": model,
            }
            for seed_key in ("baselineSeed", "variantSeed"):
                # Both sides' seed provenance, exactly as emitted into the
                # map (cross-engine keys; absent only on legacy maps).
                if seed_key in meta:
                    judge_row[seed_key] = meta[seed_key]
            if annotator_model:
                # The model the judging agent ran on, as claimed by the
                # client — cross-model agreement provenance, distinct from
                # the pin-verified judgeModel.
                judge_row["annotatorModel"] = annotator_model
            if provider:
                # The VERIFIED serving provider (openrouter judges) — the
                # per-judgment stamp completion just checked against the
                # emission pin.
                judge_row["judgeProvider"] = provider
            judge_rows.append(judge_row)
            agg = tally.setdefault(
                meta["condition"],
                {"baselineWins": 0, "variantWins": 0, "ties": 0, "n": 0})
            agg["n"] += 1
            agg[{"baseline": "baselineWins", "variant": "variantWins",
                 "tie": "ties"}[outcome]] += 1
        judge_rows.sort(key=_judgment_key)
        all_judgments.extend(judge_rows)
        outcome_maps.append(
            (judge, {_judgment_key(j): j["outcome"] for j in judge_rows}))
        kind = next((str(j.get("kind")) for j in jm_judges
                     if str(j.get("name")) == judge), "claude")
        raw_block_provider = pinned_provider_by_judge.get(judge)
        block_provider = (
            paired_judge.canonical_openrouter_provider(raw_block_provider)
            if raw_block_provider else None)
        judge_blocks.append({
            **({"provider": block_provider} if block_provider else {}),
            "name": judge, "kind": kind,
            "requestedModel": pinned_model_by_judge[judge],
            "actualModel": pinned_model_by_judge[judge],
            "conditions": tally, "pairs": len(packet_map),
        })

    human = (_load_human_validation(manifest, root)
             if manifest.human_validation else None)
    report: dict = {
        "experiment": name,
        "sourceRun": jm.get("sourceRun"),
        "evaluateRun": evaluate_run,
        "rubricFile": jm.get("rubricFile"), "rubricHash": jm.get("rubricHash"),
        "judges": judge_blocks,
        "agreement": _agreement_entries(outcome_maps),
        "pairs": judge_blocks[0]["pairs"] if judge_blocks else 0,
        "conditions": judge_blocks[0]["conditions"] if judge_blocks else {},
        "judgeModel": judge_blocks[0]["requestedModel"] if judge_blocks else None,
        "requestedJudgeModel": judge_blocks[0]["requestedModel"] if judge_blocks else None,
        "actualJudgeModel": judge_blocks[0]["actualModel"] if judge_blocks else None,
        "judgedOn": "client",
    }
    if jm.get("evaluationSource"):
        # Same stamp the inline path writes ("manifest" | "pinnedRubric"),
        # carried through the packets; legacy emissions simply omit it.
        report["evaluationSource"] = jm["evaluationSource"]
    if instructions_stamp is not None:
        # Which judging-instructions.md the campaign claims it judged
        # under, verified against the emission stamp (mismatch already
        # warned loudly at intake — recorded here, never refused).
        report["judgingInstructions"] = instructions_stamp
    if jm.get("epochUnverified"):
        report["epochUnverified"] = True
    if jm.get("measurementDrift"):
        report["measurementDrift"] = jm["measurementDrift"]
    # Exclusions were applied at EMISSION (excluded records never became
    # packets); carry the emission stamp into the completed report so the
    # final artifact set matches the inline path's.
    emission_exclusions = None
    emission_exclusions_path = os.path.join(eval_dir, "exclusions.json")
    if os.path.exists(emission_exclusions_path):
        with open(emission_exclusions_path, encoding="utf-8") as handle:
            emission_exclusions = json.load(handle)
        report["exclusions"] = emission_exclusions
    if human is not None:
        report["humanValidation"] = {"path": manifest.human_validation.path,
                                     "hash": manifest.human_validation.hash,
                                     "rows": len(human)}
        report["humanAgreement"] = [
            {"judge": entry["judges"][1], "n": entry["n"],
             "percentAgreement": entry["percentAgreement"],
             "kappa": entry["kappa"]}
            for entry in _agreement_entries(
                [("human", _materialize_human_validation(human, outcome_maps))]
                + outcome_maps)
            if entry["judges"][0] == "human"]

    # PHASE B — writes, atomic marker LAST (canonical or absent).
    out = paths.make_unique_run_directory(f"exp-{name}-evaluate-judgment",
                                          root)
    write_run_config(out, "evaluate-judgment", model_id=manifest.model_id,
                     revision=manifest.model_revision, experiment=name,
                     experiment_hash=live_hash)
    if emission_exclusions is not None:
        with open(os.path.join(out, "exclusions.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(emission_exclusions, handle, indent=2, sort_keys=True)
    with open(os.path.join(out, "judgments.jsonl"), "w",
              encoding="utf-8") as handle:
        for j in all_judgments:
            handle.write(json.dumps(j, sort_keys=True) + "\n")
    with open(os.path.join(out, "judge-report.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    def _artifact_sha(filename: str) -> str:
        with open(os.path.join(out, filename), "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    marker = {"schema": JUDGMENT_MARKER_SCHEMA,
              "kind": "evaluate",
              "experiment": name,
              "evaluateRun": evaluate_run,
              "packetsSha256": jm.get("packetsSha256"),
              "experimentHashAtJudgment": live_hash,
              "judgmentsSha256": _artifact_sha("judgments.jsonl"),
              "judgeReportSha256": _artifact_sha("judge-report.json")}
    marker_path = os.path.join(out, "judgment-source.json")
    tmp_marker = marker_path + ".tmp"
    with open(tmp_marker, "w", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)
    os.replace(tmp_marker, marker_path)  # atomic: canonical or absent
    _log(f"evaluate judgment completed ({len(all_judgments)} judgments, "
         f"{len(judge_names)} judge(s)) → {out}")
    return out


def analyze(name: str, root: str | None = None, source_run: str | None = None,
            *, model_provider=None, should_cancel: Callable[[], bool] | None = None,
            log=None, allow_unverified_epoch: bool = False,
            adjudicated_endpoint: str | None = None) -> str:
    """Statistics + reporting over a prior run: paired effect sizes with
    bootstrap CIs and Wilcoxon (effect-sizes.csv), the phase's multiple-
    comparison correction (BH-FDR for screen, Holm for confirm), alien-stance
    residuals against the pinned human baseline (alien-residuals.csv), the
    per-item paired choice deltas of the answer-token instrument
    (choice-deltas.csv — the citable version of the per-item Δ a viewer would
    otherwise derive), and the promoted-movers funnel artifact for
    screen-phase studies. Pure CPU — reads the immutable run directory,
    writes a new analyze run directory. Records whose run-time numeric parse
    was null are re-parsed under the manifest's pinned grammar first (the
    null-only endpoint rescue, stamped in endpoint-reparse.json).

    Epoch guard: the source run's stamped experiment hash must equal the live
    manifest's content hash; legacy unstamped runs need
    ``allow_unverified_epoch`` and are stamped ``epochUnverified: true``.

    ``adjudicated_endpoint`` names an external extraction campaign's
    per-record values for one numeric endpoint (see
    ``experiment/adjudication.py``). Verified against THIS source run —
    generations hash, full coverage, verbatim quote custody — then
    substituted in memory AFTER the rescue and BEFORE exclusions, and
    stamped separately (``adjudicated-endpoint.json`` +
    ``adjudication-divergence.csv``); ``endpoint-reparse.json`` is never
    touched. ``source_run`` is required with it — the CLI refuses without,
    because an adjudication is evidence about one specific run."""
    from . import (adjudication as adjudication_mod,
                   promotion as promotion_mod, reasoning_style,
                   residuals as residuals_mod, study_stats)
    _log = log or print
    manifest = Manifest.load(name, root)
    # Reasoning-style values are derived, not stored: recompute them from
    # each record's output through the pinned (hash-checked) taxonomy so
    # rs_<featureID> joins the same paired effect-size machinery.
    style = reasoning_style.load_pinned(manifest, root)
    run_dir = source_run or _latest_run(name, root)
    if not run_dir:
        raise lifecycle_gates.refusing(
            lifecycle_gates.MISSING_PREREQUISITE,
            f"no prior run with generations found for '{name}' — run it first",
            repair=(f"steerlab-server experiment run {name} && "
                    f"steerlab-server experiment analyze {name}"))
    epoch_unverified, measurement_drift = _require_source_epoch(
        "analyze", name, manifest, run_dir,
        allow_unverified_epoch=allow_unverified_epoch)
    if measurement_drift:
        _log(f"WARNING: '{name}' drifted from source run "
             f"'{os.path.basename(run_dir)}' in MEASUREMENT-side fields only "
             f"({measurement_drift}) — the generations are unaffected; "
             "analyzing proceeds under the LIVE settings and the output is "
             "stamped measurementDrift")
    if epoch_unverified:
        _log(f"WARNING: source run '{os.path.basename(run_dir)}' carries no "
             "experiment-hash stamp — analyzing it under allowUnverifiedEpoch; "
             "the output is stamped epochUnverified")
    records = []
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    # Records exist, but every one of them is the baseline: the paired
    # statistics have no contrast to compute, so this analysis is
    # structurally empty however many generations it read. A warning, not a
    # refusal — the run's own artifacts are still legitimate material — but
    # never silent (WP0 dry run #0, P0-2). Stderr, like the freeze-gate
    # warnings, so it survives a caller that only reads stdout for results.
    # Swift twin: the same line in ``ExperimentTasks.analyze``.
    if records and not any(r.get("condition") != "baseline" for r in records):
        print(f"WARNING: run '{os.path.basename(run_dir)}' contains only "
              "BASELINE records — there is no non-baseline condition to pair "
              "against, so this analysis will produce no effect sizes. Check "
              "the study's conditions before citing it.", file=sys.stderr)

    # Endpoint rescue (2026-08-10, an anchoring run): a record whose
    # run-time numeric parse came back null is re-parsed from its stored
    # output under the manifest's pinned grammar as THIS engine implements
    # it — a grammar fix (e.g. accepting a spelled-out formal register, "ten
    # years and six months") reaches finished runs without regenerating
    # them. Null-only, deliberately: run-time parses stay authoritative
    # (re-parsing everything would let a NEW first match retroactively
    # overwrite a correct old one — verified on real records, where a
    # spelled-out threshold quoted earlier in the text would beat the
    # actual answer), and a record no grammar parses stays unparsed,
    # never guessed. generations.jsonl is untouched (runs are immutable);
    # rescued values feed this analysis' exclusions and endpoints only, and
    # the analyze output stamps what happened (endpoint-reparse.json).
    # Resolution mirrors the run path: a declared registry parser (drifted
    # pin refuses) wins; otherwise caseFamily sentencing uses the built-in.
    numeric_parser = None
    if manifest.numeric_parser:
        from . import parser_registry
        numeric_parser = parser_registry.resolve(manifest.numeric_parser, root)
        if (manifest.parser_registry_hash
                and numeric_parser.registry_hash != manifest.parser_registry_hash):
            raise RuntimeError(
                f"parser registry '{parser_registry.REGISTRY_FILE}' drifted "
                f"from the pinned hash (have "
                f"{numeric_parser.registry_hash[:12]}…, pinned "
                f"{manifest.parser_registry_hash[:12]}…)")
    rescue_parse, rescue_parser_stamp = None, None
    if numeric_parser is not None:
        rescue_parse = numeric_parser.parse
        rescue_parser_stamp = numeric_parser.provenance()
    elif manifest.case_family == "sentencing":
        rescue_parse = judicial.parse_months
        rescue_parser_stamp = {"name": "builtin:sentencing",
                               "kind": "durationMonths"}
        # The DEPRECATED trigger picked the rescue grammar. Logged only here:
        # the analyze run directory does not exist yet at this point, and the
        # durable half of this analysis' record is `endpoint-reparse.json`,
        # which already stamps `builtin:sentencing` as the parser it used.
        _advise_implicit_case_family(True, None, _log, write_file=False)
    reparse_stamp = None
    if rescue_parse is not None:
        unparsed = [r for r in records
                    if "error" not in r and "instrument" not in r
                    and "output" in r and r.get("parsedMonths", 0.0) is None]
        rescued = 0
        for record in unparsed:
            fresh = rescue_parse(record.get("output") or "")
            if fresh is not None:
                record["parsedMonths"] = fresh
                rescued += 1
        reparse_stamp = {
            "endpoint": "parsedMonths",
            "parser": rescue_parser_stamp,
            "unparsedRecords": len(unparsed),
            "rescuedRecords": rescued,
            "stillUnparsed": len(unparsed) - rescued,
            "note": ("Null-only endpoint rescue: records whose run-time "
                     "parse was null were re-parsed from their stored "
                     "output under the manifest's pinned grammar as this "
                     "engine version implements it. Run-time parses stay "
                     "authoritative, generations.jsonl is untouched, and a "
                     "record no grammar parses stays unparsed — never "
                     "guessed. Rescued values feed this analysis only."),
        }
        if unparsed:
            _log(f"endpoint rescue: {rescued}/{len(unparsed)} previously "
                 f"unparsed parsedMonths record(s) re-parsed under "
                 f"'{rescue_parser_stamp['name']}'"
                 + (f"; {len(unparsed) - rescued} still unparsed"
                    if rescued < len(unparsed) else ""))

    # Adjudicated-endpoint intake (open-issues §10, 2026-08-18): an external
    # extraction campaign's per-record values REPLACE the endpoint the run
    # parsed, once they have been verified against this exact run. Separate
    # pass, separate stamp, deliberately: the rescue above is null-only
    # precisely because a run-time parse must never be silently overwritten,
    # and an adjudication overwrites by design — so it carries its own
    # custody ladder (generations hash, per-row validity, verbatim quote
    # containment, exhaustive coverage) and its own divergence accounting
    # against the value analyze would otherwise have used. It runs AFTER the
    # rescue (a record both rescued and adjudicated is accounted against its
    # rescued value) and BEFORE exclusions and _endpoint_values, so both see
    # adjudicated values. generations.jsonl is untouched.
    adjudication_stamp = None
    adjudication_rows: list = []
    if adjudicated_endpoint:
        adjudication_document, adjudication_sha = adjudication_mod.load(
            adjudicated_endpoint)
        records, adjudication_stamp, adjudication_rows = adjudication_mod.apply(
            records, adjudication_document, file_sha256=adjudication_sha,
            run_dir=run_dir,
            instructions_dir=os.path.dirname(
                os.path.abspath(adjudicated_endpoint)),
            log=_log)

    # Declared exclusion rules join HERE — records are dropped from the
    # paired statistics only (pairwise deletion falls out of the promptID
    # join below), never from generations.jsonl, and the stamp
    # (exclusions.json) records what was active, what each rule excluded
    # per condition, the surviving N, and the declared scope. Scope is
    # allRecordTypes (the apply default): deterministic instrument
    # readouts are considered too — endpoint rules read endpoints the
    # record itself carries (e.g. ordinalPosition), and a cell whose every
    # sampled record failed its attention check drops its instrument
    # readouts (choiceLogOdds / ordinalPosition endpoints) with it. No
    # rules declared = today's behavior byte-for-byte.
    from . import exclusions as exclusions_mod
    exclusion_rules = exclusions_mod.declared_rules(manifest.raw)
    exclusion_stamp = None
    if exclusion_rules:
        checks: dict[str, dict] = {}
        if exclusions_mod.needs_checks(exclusion_rules):
            if not manifest.task_prompts_hash:
                # WP0 step 8: the deferred cross-engine-twinned message from
                # c86ce53 gets its id on BOTH engines. The STRING is unchanged
                # and stays byte-identical to Swift's
                # `ExclusionEngine.pinRequiredMessage` (asserted on both
                # sides); only the gate id and the repair are new.
                raise lifecycle_gates.refusing(
                    lifecycle_gates.MISSING_PREREQUISITE,
                    exclusions_mod.PIN_REQUIRED_MESSAGE,
                    repair=exclusions_mod.PIN_REQUIRED_REPAIR)
            checks = exclusions_mod.attention_checks(
                _load_prompts(manifest, None, root))
            if not checks:
                raise RuntimeError(exclusions_mod.NO_CHECKS_MESSAGE)
        records, exclusion_stamp = exclusions_mod.apply(
            records, exclusion_rules, checks)
        _log(f"exclusions: {exclusion_stamp['excludedRecords']} record(s) "
             f"excluded by {len(exclusion_rules)} declared rule(s); "
             "surviving N per condition: "
             + ", ".join(f"{c}={n}" for c, n
                         in exclusion_stamp["survivingN"].items()))

    # D1, multi-agent only: turns are NOT independent observations — turn k is
    # conditioned on turns 1..k-1, and after turn 1 the arms diverge, so what
    # pairs across conditions is script POSITION, not matched input. Carry the
    # replicate in the pairing key (otherwise the same turn id from different
    # transcripts collides and gets averaged into one cell), then aggregate
    # each transcript to its mean difference before any test runs.
    clustered = manifest.study_kind == "multiAgent"
    # Declared panel endpoints (Wave-2). Collected BEFORE the transcript
    # re-keying below, which rewrites promptID — the endpoint aggregation
    # reads seat/condition/replicate, and should not depend on a key another
    # concern owns. Nothing is parsed here: these are the runner's write-time
    # stamps, verbatim. Free-text mining never happens downstream of the run.
    endpoint_rows = turn_endpoint.csv_rows(records)
    endpoint_counts = turn_endpoint.counts(records)
    # Per-item choice deltas (Phase 3 of the results-explorer plan). Collected
    # here for the same reason as the endpoint stamps: the baseline join is by
    # promptID, and the transcript re-keying below rewrites it. Exclusions have
    # already been applied, so a dropped readout drops from this table too.
    # Declared-target map from the PINNED task file (open-issues #6): the
    # exact record of which items declared a choice target, so the
    # choiceLogOdds endpoint and the choice-deltas artifact never read a
    # run-time-synthesized target (an ordinalScale item's scale minimum) as
    # a declared endpoint — and, symmetrically, so a mixed instrument like
    # s4-framings (declared A/B target AND an ordinal readout on the same
    # record) keeps its legitimate endpoint. Unloadable prompts fall back to
    # the per-record heuristic inside the consumers.
    declared_targets = None
    try:
        if manifest.task_prompts_hash:
            declared_targets = {str(p.get("id")): bool(p.get("target"))
                                for p in _load_prompts(manifest, None, root)}
    except Exception as exc:
        _log(f"declared-target map unavailable ({exc}); choice endpoints "
             "fall back to per-record classification")
    choice_delta_rows, choice_delta_summary = choice_deltas.rows(
        records, declared_targets=declared_targets)
    if clustered:
        records = _key_records_by_transcript(records)
    # Parser kind for endpoint-label honesty (see _endpoint_values), from
    # the parser already resolved for the endpoint rescue above.
    numeric_parser_kind = (numeric_parser.kind
                           if numeric_parser is not None else None)
    endpoints = _endpoint_values(records, style=style,
                                 numeric_parser_kind=numeric_parser_kind,
                                 declared_targets=declared_targets)
    baseline = {endpoint: cells.get("baseline", {}) for endpoint, cells in endpoints.items()}
    modalities = _condition_modalities(manifest, root)
    rows: list[study_stats.EffectRow] = []
    diffs_index: dict[tuple[str, str], list[float]] = {}
    skipped_for_replication = False
    for endpoint, cells in endpoints.items():
        for condition, values in cells.items():
            if condition == "baseline":
                continue
            base = baseline.get(endpoint, {})
            if clustered:
                diffs = _transcript_level_diffs(values, base)
                # One transcript per arm is a point estimate, not an interval.
                if len(diffs) < 2:
                    skipped_for_replication = True
                    continue
            else:
                diffs = [values[pid] - base[pid] for pid in sorted(values) if pid in base]
            if not diffs:
                continue
            row = study_stats.effect_row(condition, endpoint, diffs)
            row.modality = modalities.get(condition, "")
            rows.append(row)
            diffs_index[(condition, endpoint)] = diffs
    method = "holm" if manifest.phase == "confirm" else "bh"
    by_endpoint: dict[str, list[study_stats.EffectRow]] = {}
    for row in rows:
        by_endpoint.setdefault(row.endpoint, []).append(row)
    for family in by_endpoint.values():
        study_stats.apply_correction(family, method=method)
    # Per-cell strata beside the pooled rows (same CSV, extra rows): pooling
    # across items has both hidden a real single-cell effect behind saturated
    # cells and manufactured pooled effects from one cell's parse garbage.
    # Pooled rows keep their exact semantics and correction family; each
    # stratified family (promptID, declared factors, their cross) is
    # corrected independently. Clustered multi-agent runs skip stratification
    # — their promptIDs were re-keyed to transcript positions above and the
    # transcript aggregation is the honest unit there.
    stratified_rows: list = []
    if not clustered:
        stratified_rows = _stratified_effect_rows(
            records, endpoints, style=style, method=method,
            modalities=modalities)

    out = paths.make_unique_run_directory(f"exp-{name}-analyze", root)
    _write_config_snapshot(manifest, out, "analyze",
                           notes=({**({"epochUnverified": True}
                                      if epoch_unverified else {}),
                                   **({"measurementDrift": measurement_drift}
                                      if measurement_drift else {}),
                                   # The canonical per-run stamp says the
                                   # endpoint values were substituted, so a
                                   # reader of config.json alone cannot miss
                                   # it (notes is the established extension
                                   # point).
                                   **(adjudication_mod.notes_block(
                                       adjudication_stamp)
                                      if adjudication_stamp else {})}
                                  or None))
    with open(os.path.join(out, "source-run.txt"), "w", encoding="utf-8") as handle:
        handle.write(os.path.basename(run_dir) + "\n")
    if clustered:
        # Say what an effect row averages over. `n` counts TRANSCRIPTS here,
        # not turns, and a reader cannot tell which from the number alone.
        with open(os.path.join(out, "unit-of-analysis.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"unitOfAnalysis": "transcript",
                       "reason": "turns within a transcript are dependent; "
                                 "each transcript is reduced to its mean paired "
                                 "difference before testing",
                       "skippedForSingleTranscript": skipped_for_replication},
                      handle, indent=2, sort_keys=True)
        if skipped_for_replication:
            _log("effect sizes: some endpoints skipped — 1 transcript per "
                 "condition supports a point estimate but no interval. "
                 "Re-run with samplesPerItem > 1.")
    if endpoint_rows:
        # Zero stamps ⇒ no file and no section: an analyze over a panel that
        # declared nothing must not grow an empty table implying it did.
        turn_endpoint.write_csv(
            os.path.join(out, "panel-endpoints.csv"), endpoint_rows)
        with open(os.path.join(out, "panel-endpoints.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"endpoints": endpoint_counts,
                       "records": len(endpoint_rows),
                       "unparsed": sum(row[-1] for row in endpoint_rows)},
                      handle, indent=2, sort_keys=True)
        _log(f"panel endpoints: {len(endpoint_rows)} stamped turn(s), "
             f"{sum(row[-1] for row in endpoint_rows)} unparsed → "
             "panel-endpoints.csv")
    if choice_delta_summary["conditions"]:
        # Same rule as the panel endpoints above: a run with no non-baseline
        # choice readouts grows no table implying it had some. When there ARE
        # readouts the file is written even if every one of them was skipped —
        # the skip counts are the finding in that case.
        choice_deltas.write_csv(
            os.path.join(out, "choice-deltas.csv"), choice_delta_rows)
        with open(os.path.join(out, "choice-deltas.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(choice_delta_summary, handle, indent=2, sort_keys=True)
        _log(f"choice deltas: {len(choice_delta_rows)} paired item(s) across "
             f"{len(choice_delta_summary['conditions'])} condition(s), "
             f"{sum(block['flipped'] for block in choice_delta_summary['conditions'].values())} "
             f"flip(s), {choice_delta_summary['skippedNoBaseline']} skipped "
             "(no baseline partner) → choice-deltas.csv")
        if choice_delta_summary["skippedNoTargetValue"]:
            _log(f"choice deltas: {choice_delta_summary['skippedNoTargetValue']} "
                 "readout(s) skipped — no log-odds entry for the item's own "
                 "target option")
    if exclusion_stamp is not None:
        # The exclusion stamp (cross-engine shape; Swift embeds the same
        # object in analysis.json AND writes this file): active rules with
        # plain-language descriptions, per-condition per-rule exclusion
        # counts, surviving N, and the pairwise-deletion note.
        with open(os.path.join(out, "exclusions.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(exclusion_stamp, handle, indent=2, sort_keys=True)
    if reparse_stamp is not None:
        # The endpoint-rescue stamp: which grammar re-parsed the null
        # records, how many were rescued, how many stayed unparsed. Written
        # whenever a numeric grammar applied — zero rescues included, so an
        # analysis that changed nothing says so explicitly.
        with open(os.path.join(out, "endpoint-reparse.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(reparse_stamp, handle, indent=2, sort_keys=True)
    if adjudication_stamp is not None:
        # The adjudication's own two artifacts, never mixed into the
        # rescue's: the summary stamp (file hash, instructions block,
        # divergence counts, per-condition breakdown) and the FULL row-level
        # list of records whose adjudicated value differs from the value
        # analyze would otherwise have used — that list is analysis
        # evidence, not a sample. The CSV is written even when nothing
        # diverged (header only), like the rescue's zero-rescue stamp.
        adjudication_mod.write_stamp(out, adjudication_stamp)
        adjudication_mod.write_divergence_csv(out, adjudication_rows)
        _log(f"adjudicated endpoint: {len(adjudication_rows)} divergent "
             f"record(s) → {adjudication_mod.DIVERGENCE_FILENAME}")
    if epoch_unverified:
        with open(os.path.join(out, "epoch-unverified.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"epochUnverified": True,
                       "sourceRun": os.path.basename(run_dir)}, handle,
                      indent=2, sort_keys=True)
    if measurement_drift:
        # Tolerated measurement-side drift is never silent (twin of the
        # epoch-unverified sidecar): which fields differed, verbatim.
        with open(os.path.join(out, "measurement-drift.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"measurementDrift": measurement_drift,
                       "sourceRun": os.path.basename(run_dir)}, handle,
                      indent=2, sort_keys=True)
    # D3: distance-from-boundary diagnostics, per condition. A large
    # joint-logprob margin means the FLIP RATE has poor sensitivity there —
    # an intervention can move the log-odds a long way without flipping any
    # item — while the log-odds itself keeps moving continuously. Reporting
    # that as "saturation" invites the wrong conclusion; true numerical
    # saturation is the separately counted clamp incidence.
    from . import choice_margin
    margin_report = {}
    by_condition_for_margins: dict[str, list[dict]] = {}
    for record in records:
        if record.get("optionLogprobs"):
            by_condition_for_margins.setdefault(
                record.get("condition", ""), []).append(record)
    for condition, items in by_condition_for_margins.items():
        block = choice_margin.diagnostics(items)
        if block.get("scoredItems"):
            margin_report[condition] = block
    if margin_report:
        with open(os.path.join(out, "choice-margins.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(margin_report, handle, indent=2, sort_keys=True)
        for condition, block in sorted(margin_report.items()):
            _log(f"{condition}: {block['interpretation']}")

    with open(os.path.join(out, "effect-sizes.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(study_stats.EFFECT_SIZES_HEADER)
        for row in rows:
            writer.writerow(row.as_csv_row())
        # Stratified rows ride after every pooled row; readers that group by
        # (condition, endpoint) alone must filter on stratifyBy == "pooled"
        # (the semantic viewers do). Each stratified row also declares its
        # ``estimand`` and what its p-values are licensed for (``inference``);
        # a within-item-sample row is diagnostic and carries no adjusted p.
        for row in stratified_rows:
            writer.writerow(row.as_csv_row())

    # Behavioral-fingerprint table: the same effect rows reshaped (condition ×
    # endpoint, with modality) for cross-condition comparison — no new stats.
    with open(os.path.join(out, "fingerprints.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(study_stats.FINGERPRINTS_HEADER)
        for csv_row in study_stats.fingerprint_csv_rows(rows):
            writer.writerow(csv_row)

    if manifest.human_baseline is not None:
        base_path = manifest.human_baseline.path
        if not os.path.isabs(base_path):
            base_path = os.path.join(paths.project_root() if root is None else root,
                                     base_path)
        human = residuals_mod.load_human_baseline(base_path, manifest.human_baseline.hash)
        residual_rows = residuals_mod.residual_rows(rows, human)
        residuals_mod.write_alien_residuals_csv(
            os.path.join(out, "alien-residuals.csv"), residual_rows)
        _log(f"alien residuals: {len(residual_rows)} rows")

    if manifest.promotion_rule is not None and manifest.phase == "screen":
        decisions = _promotion_decisions(manifest, rows, run_dir, promotion_mod)
        promotion_mod.write_promoted_movers(
            os.path.join(out, "promoted-movers.json"), decisions,
            experiment=name, experiment_hash=manifest.content_hash(),
            rule=manifest.promotion_rule)
        _log(f"promotion: {sum(1 for d in decisions if d.promoted)}/{len(decisions)} promoted")

    diagnostic_strata = sum(1 for row in stratified_rows
                            if row.inference == "diagnostic")
    _log(f"analyze ({len(rows)} effect rows"
         + (f" + {len(stratified_rows)} stratified" if stratified_rows else "")
         + (f", {diagnostic_strata} of them within-item diagnostics "
            "(uncorrected — a single item's samples support no cross-item "
            "claim)" if diagnostic_strata else "")
         + f") → {out}")
    return out


def rescore_style(name: str, root: str | None = None, source_run: str | None = None,
                  *, allow_unverified_epoch: bool = False, log=None) -> str:
    """Post-hoc reasoning-style scoring (Swift twin ``rescoreStyle``):
    recompute ``rs_<featureID>`` values for an EXISTING completed run's
    sampled generations from the manifest's pinned taxonomy — pure CPU, no
    model — writing ``reasoning-style.csv`` + ``reasoning-style.json`` into a
    NEW immutable rescore run directory. The source run is NEVER mutated (run
    immutability), and the epoch guard applies exactly as for analyze."""
    from . import reasoning_style
    _log = log or print
    manifest = Manifest.load(name, root)
    _verify_or_warn(manifest, root)
    style = reasoning_style.load_pinned(manifest, root)
    if style is None:
        raise RuntimeError(
            f"experiment '{name}' pins no reasoning-style taxonomy — pin one "
            "first (reasoningStyleTaxonomyPath + reasoningStyleTaxonomyHash; "
            "see prompts/templates/reasoning-style/)")
    run_dir = source_run or _latest_run(name, root)
    if not run_dir:
        raise lifecycle_gates.refusing(
            lifecycle_gates.MISSING_PREREQUISITE,
            f"no prior run with generations found for '{name}' — run it first",
            repair=(f"steerlab-server experiment run {name} && "
                    f"steerlab-server experiment rescore-style {name}"))
    epoch_unverified, measurement_drift = _require_source_epoch(
        "rescore-style", name, manifest, run_dir,
        allow_unverified_epoch=allow_unverified_epoch)
    if measurement_drift:
        _log(f"WARNING: '{name}' drifted from source run "
             f"'{os.path.basename(run_dir)}' in MEASUREMENT-side fields only "
             f"({measurement_drift}) — the generations are unaffected; "
             "rescoring proceeds under the LIVE settings and the output is "
             "stamped measurementDrift")
    if epoch_unverified:
        _log(f"WARNING: source run '{os.path.basename(run_dir)}' carries no "
             "experiment-hash stamp — rescoring it under allowUnverifiedEpoch; "
             "the output is stamped epochUnverified")
    records: list[dict] = []
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    sampled = [r for r in records
               if "error" not in r and "instrument" not in r and "output" in r]
    if not sampled:
        raise RuntimeError(
            f"run '{os.path.basename(run_dir)}' has no sampled generations to rescore")

    # NEW immutable artifacts only — never a byte into the source run.
    out = paths.make_unique_run_directory(f"exp-{name}-rescore-style", root)
    _write_config_snapshot(manifest, out, "rescore-style",
                           notes=({**({"epochUnverified": True}
                                      if epoch_unverified else {}),
                                   **({"measurementDrift": measurement_drift}
                                      if measurement_drift else {})} or None))
    with open(os.path.join(out, "source-run.txt"), "w", encoding="utf-8") as handle:
        handle.write(os.path.basename(run_dir) + "\n")
    feature_ids = style.taxonomy.feature_ids
    with open(os.path.join(out, "reasoning-style.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "seed", "promptIndex", "promptID"]
                        + [f"rs_{fid}" for fid in feature_ids])
        for index, record in enumerate(sampled):
            values = style.taxonomy.score(record.get("output", ""))
            writer.writerow([
                record.get("condition", ""),
                record.get("seed", 0),
                record.get("promptIndex", index),
                record.get("promptID", ""),
            ] + [values.get(fid, 0.0) for fid in feature_ids])
    by_condition: dict[str, list[dict]] = {}
    for record in sampled:
        by_condition.setdefault(record.get("condition", ""), []).append(record)
    report = {
        "experiment": name,
        "experimentHash": manifest.content_hash(),
        "sourceRun": os.path.basename(run_dir),
        # The shared reader already falls back to config.json's stamp.
        "sourceRunExperimentHash": _stamped_experiment_hash(run_dir),
        "taxonomy": style.taxonomy.name,
        "taxonomyHash": style.hash,
        "taxonomyFile": style.path,
        # Same status stamp as report.json's per-condition block: style
        # features are a diagnostic/manipulation check, never an outcome
        # endpoint (docs/METHODS.md).
        "diagnosticOnly": True,
        "conditions": {
            cond: {"features": _reasoning_style_block(items, style)["features"]}
            for cond, items in by_condition.items()
        },
    }
    if epoch_unverified:
        report["epochUnverified"] = True
    if measurement_drift:
        # Tolerated measurement-side drift is never silent: which fields
        # differed from the source run's epoch, verbatim.
        report["measurementDrift"] = measurement_drift
    with open(os.path.join(out, "reasoning-style.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    _log(f"rescore-style ({len(sampled)} generations, {len(feature_ids)} "
         f"feature(s), {len(by_condition)} condition(s)) → {out}")
    return out


def _condition_modalities(manifest: Manifest, root: str | None = None) -> dict[str, str]:
    """Intervention modality per condition, derived from the manifest
    (RESULTS-ARCHITECTURE: modality is a design axis — injection / adapter /
    systemPrompt / stacked; baseline = none).

    Steering-slot conditions are pure residual-stream work → "injection"
    (matched-norm random controls included: same modality, different content).
    Variant conditions are read off the variant artifact's components —
    adapters, injections, and a variant-level systemPrompt — with any
    combination of two or more → "stacked". Conditions that cannot be found in
    the manifest map to "" (never guessed).

    SAE latent conditions get their OWN modality, ``"saeLatent"`` — not
    "injection". They are residual-stream work, but the edit is state-dependent
    and dosed in latent units rather than residual-norm units, so pooling them
    with vector additions under one label would let a reader compare doses that
    are not comparable.
    """
    modalities: dict[str, str] = {"baseline": "none"}
    for condition in manifest.conditions:
        modalities[condition.name] = "injection"
    for entry in manifest.sae_latent_conditions:
        if entry.get("name"):
            modalities[str(entry["name"])] = "saeLatent"
    for vc in manifest.variant_conditions:
        artifact = vc.artifact or {}
        if not artifact and vc.artifact_path:
            # Older manifests pin path+hash without embedding the artifact.
            try:
                with open(paths.resolve(vc.artifact_path, root), encoding="utf-8") as handle:
                    artifact = json.load(handle)
            except (OSError, json.JSONDecodeError):
                artifact = {}
        has_injection = bool(artifact.get("injections"))
        has_adapter = bool(artifact.get("adapters"))
        has_system_prompt = bool(str(artifact.get("systemPrompt") or "").strip())
        components = [name for name, present in (
            ("injection", has_injection), ("adapter", has_adapter),
            ("systemPrompt", has_system_prompt)) if present]
        if len(components) > 1:
            modalities[vc.name] = "stacked"
        elif components:
            modalities[vc.name] = components[0]
        else:
            modalities[vc.name] = "none"
    return modalities


def _key_records_by_transcript(records: list[dict]) -> list[dict]:
    """Rewrite promptID to ``<turnID>@<replicateIndex>`` so the pairing join
    distinguishes the same script position in DIFFERENT play-throughs.

    Without this the sample-axis averaging in ``_endpoint_values`` pools every
    replicate of a turn into one cell, which silently discards exactly the
    between-transcript variation the clustered estimator exists to measure."""
    out = []
    for record in records:
        replicate = record.get("replicateIndex", 0) or 0
        clone = dict(record)
        clone["promptID"] = f"{record.get('promptID', '')}@{replicate}"
        out.append(clone)
    return out


def _transcript_level_diffs(values: dict[str, float],
                            base: dict[str, float]) -> list[float]:
    """Turn-level paired differences aggregated to ONE value per transcript.

    Clusters are balanced by construction — every transcript plays the same
    turn script — which is exactly the case where cluster-level aggregation is
    the correct estimator and needs no new statistics: the existing bootstrap
    and Wilcoxon then run over transcript-level values, so the reported ``n``
    is the number of transcripts. Cross-engine twin of Swift
    ``ExperimentTasks.transcriptEffectSizes``."""
    by_transcript: dict[str, list[float]] = {}
    for pid in sorted(values):
        if pid not in base:
            continue
        _, _, replicate = pid.rpartition("@")
        by_transcript.setdefault(replicate, []).append(values[pid] - base[pid])
    return [sum(v) / len(v) for _, v in sorted(by_transcript.items()) if v]


def _endpoint_values(records: list[dict], style=None,
                     numeric_parser_kind: str | None = None,
                     declared_targets: dict[str, bool] | None = None,
                     ) -> dict[str, dict[str, dict[str, float]]]:
    """endpoint → condition → promptID → value, from a run's records.

    Endpoints: ``choiceLogOdds`` (instrument log-odds of the item's target
    option), ``ordinalPosition`` (the ordinalScale instrument's ladder
    position — stamped on the instrument record only when the manifest
    declared ordinalScale; cross-engine endpoint name pinned against Swift's
    ``effectSizes``), ``choiceRate`` (sampled parsed-choice rate of the
    target), ``meanMonths`` and ``monthsSpread`` (Case 3 mean and stdev over
    the sample axis), ``readerScore:<concept>`` (mean RepE reader score of the
    sampled outputs, one endpoint per pinned reader), and — when ``style``
    pins a reasoning-style taxonomy — ``rs_<featureID>`` (mean per-generation
    feature value over the sample axis, recomputed from the output text).
    Same-item pairing happens downstream by promptID.

    Endpoint-label honesty (2026-08-06): the ``parsedMonths`` record key is
    written by ANY declared registry parser — percentage and 1–7 scale
    parsers included — so ``meanMonths``/``monthsSpread`` were verified
    misleading on real runs whose parser never produced months. When
    ``numeric_parser_kind`` names a registry kind other than
    ``durationMonths``, the same values are ADDITIONALLY emitted under the
    neutral ``parsedValueMean``/``parsedValueSpread`` endpoints; the months
    labels are retained as deprecated aliases so existing readers (residual
    human-baseline joins, ``_PRIMARY_ENDPOINT_ORDER``, the explorer) keep
    working. New readers should prefer the parsedValue names."""
    endpoints: dict[str, dict[str, dict[str, float]]] = {}

    def put(endpoint: str, condition: str, prompt_id: str, value: float) -> None:
        endpoints.setdefault(endpoint, {}).setdefault(condition, {})[prompt_id] = value

    cells: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        if "error" in record:
            continue
        key = (record.get("condition", ""), str(record.get("promptID", "")))
        cells.setdefault(key, []).append(record)
    # Prose endpoints. Every generation already carries these and Swift has
    # always used them, but this collector never emitted them — so a panel
    # study (prose turns, no choice instrument, often no pinned taxonomy)
    # produced an EMPTY endpoint set and therefore no effect sizes at all,
    # silently. They are means over the sample axis, like every other endpoint
    # here.
    for (condition, prompt_id), items in cells.items():
        for endpoint, field in (("wordCount", "wordCount"),
                                ("distinct2", "distinct2")):
            values = [float(r[field]) for r in items
                      if isinstance(r.get(field), (int, float))]
            if values:
                put(endpoint, condition, prompt_id, sum(values) / len(values))
    for (condition, prompt_id), items in cells.items():
        months: list[float | None] = []
        target_hits: list[bool] = []
        reader_values: dict[str, list[float]] = {}
        style_values: dict[str, list[float]] = {}
        for record in items:
            if record.get("instrument") == "answerTokenLogprob":
                # choiceLogOdds is a DECLARED endpoint (open-issues #6). The
                # authority ladder: the pinned task file's per-item map when
                # the caller could load it (exact — handles mixed
                # instruments like s4-framings, declared target + ordinal
                # readout on one record); else the record's own targetSource
                # stamp (new writers); else the observed historical failure
                # class — a record whose only "target" was synthesized rides
                # an ordinalScale readout, so ordinalPosition marks it.
                target = record.get("target")
                if declared_targets is not None and prompt_id in declared_targets:
                    declared = declared_targets[prompt_id]
                elif "targetSource" in record:
                    declared = record.get("targetSource") == "declared"
                else:
                    declared = record.get("ordinalPosition") is None
                odds = record.get("logOdds", {}).get(target)
                if odds is not None and declared:
                    put("choiceLogOdds", condition, prompt_id, float(odds))
                # ordinalScale rides the same instrument record: the ladder
                # position is one more per-item numeric endpoint through the
                # SAME paired machinery (no new statistics). Key present only
                # when the manifest declared ordinalScale at run time.
                position = record.get("ordinalPosition")
                if position is not None:
                    put("ordinalPosition", condition, prompt_id,
                        float(position))
                continue
            if "parsedMonths" in record:
                months.append(record["parsedMonths"])
            if "parsedChoice" in record and record.get("target") is not None:
                if record["parsedChoice"] is not None:
                    target_hits.append(record["parsedChoice"] == record["target"])
            for concept, score in (record.get("readerScores") or {}).items():
                reader_values.setdefault(str(concept), []).append(float(score))
            if style is not None and "output" in record:
                scored = style.taxonomy.score(record.get("output", ""))
                for fid in style.taxonomy.feature_ids:
                    style_values.setdefault(fid, []).append(scored.get(fid, 0.0))
        for concept, values in reader_values.items():
            put(f"readerScore:{concept}", condition, prompt_id,
                sum(values) / len(values))
        for fid, values in style_values.items():
            put(f"rs_{fid}", condition, prompt_id, sum(values) / len(values))
        summary = judicial.summarize([m for m in months])
        if summary is not None:
            put("meanMonths", condition, prompt_id, summary.mean)
            if summary.count > 1:
                put("monthsSpread", condition, prompt_id, summary.stdev)
            # Honest twin labels for non-months parsers (see docstring):
            # additive, so every months-keyed reader keeps working.
            if numeric_parser_kind not in (None, "durationMonths"):
                put("parsedValueMean", condition, prompt_id, summary.mean)
                if summary.count > 1:
                    put("parsedValueSpread", condition, prompt_id,
                        summary.stdev)
        if target_hits:
            put("choiceRate", condition, prompt_id,
                sum(target_hits) / len(target_hits))
    return endpoints


# --- stratified effect rows ---------------------------------------------------
#
# Pooling every task item into one paired test can BOTH hide a real per-cell
# effect behind saturated cells (11 of 12 items at a 0/1 choice-rate floor
# contribute only denominator) AND manufacture a pooled effect out of one
# cell's parse garbage. The stratified rows below keep the pooled rows intact
# (their semantics and correction family are untouched) and ADD per-stratum
# rows — split by promptID always, and by declared factor metadata (top-level
# ``arm``/``caseID`` plus every key of the factorial ``factors`` object) when
# items carry it — each family corrected independently (same phase method,
# applied per endpoint WITHIN the family). Swift twin:
# ``ExperimentTasks.stratifiedEffectSizes`` (identical CSV column vocabulary:
# stratifyBy, stratum, unit).

_STRATUM_JOIN = "×"


def _item_factor_levels(records: list[dict]) -> dict[str, dict[str, str]]:
    """promptID → {factorKey: level} from the record-carried prompt metadata
    (``_PROMPT_META_KEYS`` stamps ``arm``/``caseID``/``factors`` verbatim on
    sampled AND instrument records, so no rejoin of the input file). First
    record per item wins; items stamp their metadata identically by
    construction."""
    meta: dict[str, dict[str, str]] = {}
    for record in records:
        prompt_id = str(record.get("promptID", ""))
        if not prompt_id or prompt_id in meta:
            continue
        levels: dict[str, str] = {}
        for key in ("arm", "caseID"):
            value = record.get(key)
            if isinstance(value, str) and value:
                levels[key] = value
        factors = record.get("factors")
        if isinstance(factors, dict):
            for key, value in factors.items():
                if isinstance(value, str) and value:
                    levels[str(key)] = value
        meta[prompt_id] = levels
    return meta


def _stratification_families(
        factors_by_item: dict[str, dict[str, str]],
        items: set[str]) -> list[tuple[str, dict[str, set[str]]]]:
    """The stratification families for a run, in a fixed order: promptID
    (always), each factor key with ≥2 observed levels (marginals), and the
    full cross of ALL factor keys when there are ≥2 of them (the per-cell
    view — e.g. ``arm×caseID`` → ``notLegal×loan``). Keys sort
    alphabetically; a constant factor is skipped (its one stratum would just
    duplicate the pooled row under another name)."""
    families: list[tuple[str, dict[str, set[str]]]] = [
        ("promptID", {prompt_id: {prompt_id} for prompt_id in sorted(items)})]
    keys = sorted({key for levels in factors_by_item.values() for key in levels})
    for key in keys:
        strata: dict[str, set[str]] = {}
        for prompt_id in sorted(items):
            level = factors_by_item.get(prompt_id, {}).get(key)
            if level is not None:
                strata.setdefault(level, set()).add(prompt_id)
        if len(strata) >= 2:
            families.append((key, strata))
    if len(keys) >= 2:
        cells: dict[str, set[str]] = {}
        for prompt_id in sorted(items):
            levels = factors_by_item.get(prompt_id, {})
            if all(key in levels for key in keys):
                label = _STRATUM_JOIN.join(levels[key] for key in keys)
                cells.setdefault(label, set()).add(prompt_id)
        if len(cells) >= 2:
            families.append((_STRATUM_JOIN.join(keys), cells))
    return families


def _endpoint_sample_values(
        records: list[dict],
        style=None) -> dict[str, dict[str, dict[str, dict[int, float]]]]:
    """endpoint → condition → promptID → {sampleIndex: value}, for endpoints
    that have a per-sample reading (``wordCount``, ``distinct2``,
    ``choiceRate`` as the 0/1 target hit, ``meanMonths`` as the per-sample
    parse, ``readerScore:<concept>``, ``rs_<featureID>``). Deterministic
    instrument readouts (choiceLogOdds, ordinalPosition) and cross-sample
    aggregates (monthsSpread) have no sample axis and are absent. This is the
    single-item stratum's resolution: within one item, treatment sample k
    pairs to baseline sample k."""
    out: dict[str, dict[str, dict[str, dict[int, float]]]] = {}

    def put(endpoint: str, condition: str, prompt_id: str, sample: int,
            value: float) -> None:
        out.setdefault(endpoint, {}).setdefault(condition, {}) \
           .setdefault(prompt_id, {})[sample] = value

    for record in records:
        if "error" in record or record.get("instrument"):
            continue
        condition = record.get("condition", "")
        prompt_id = str(record.get("promptID", ""))
        sample = int(record.get("sampleIndex") or 0)
        for endpoint, field in (("wordCount", "wordCount"),
                                ("distinct2", "distinct2")):
            value = record.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                put(endpoint, condition, prompt_id, sample, float(value))
        if record.get("parsedMonths") is not None:
            put("meanMonths", condition, prompt_id, sample,
                float(record["parsedMonths"]))
        if "parsedChoice" in record and record.get("target") is not None \
                and record["parsedChoice"] is not None:
            put("choiceRate", condition, prompt_id, sample,
                1.0 if record["parsedChoice"] == record["target"] else 0.0)
        for concept, score in (record.get("readerScores") or {}).items():
            put(f"readerScore:{concept}", condition, prompt_id, sample,
                float(score))
        if style is not None and "output" in record:
            scored = style.taxonomy.score(record.get("output", ""))
            for fid in style.taxonomy.feature_ids:
                put(f"rs_{fid}", condition, prompt_id, sample,
                    scored.get(fid, 0.0))
    return out


def _stratified_effect_rows(records: list[dict], endpoints, *, style, method,
                            modalities) -> list:
    """The stratified companion rows to analyze's pooled effect rows.

    Within a stratum the unit of analysis is the ITEM whenever the stratum
    has ≥2 items pairing to a baseline (the pooled machinery restricted to
    the stratum); a single-item stratum drops to the SAMPLE axis (per-sample
    pairs by sampleIndex) when one exists — that resolution is the whole
    point, since a single saturating item's signal is exactly what pooling
    averages away.

    But the two are NOT the same estimand, and treating them as one was a
    real defect (review 2026-08-06). An item-level row estimates a quantity
    that generalizes over the items in its stratum. A within-item row
    estimates how ONE prompt's generations moved — a prompt-specific
    stochastic quantity, over samples that are exchangeable draws rather than
    a designed pairing. Running the two through one correction family let a
    within-item row emerge with a corrected p that reads exactly like a
    cross-item finding, which it can never be. So each row is now STAMPED
    with its estimand, and within-item rows are marked ``diagnostic`` and
    HELD OUT of the correction: they keep their estimate, interval and raw
    Wilcoxon p as a locator for which cell moved, and carry no adjusted p.
    (Held out, not modeled: pooling the two levels properly is a hierarchical
    model, and inventing one silently inside a CSV writer would be worse than
    the defect.)

    Corrections run per endpoint WITHIN each family, over that family's
    item-level rows only — never across families and never joined to the
    pooled family."""
    from . import study_stats
    item_ids = {prompt_id
                for cells in endpoints.values()
                for values in cells.values()
                for prompt_id in values}
    if not item_ids:
        return []
    families = _stratification_families(_item_factor_levels(records), item_ids)
    samples = _endpoint_sample_values(records, style=style)
    rows: list = []
    for family_name, strata in families:
        family_rows: dict[str, list] = {}
        for endpoint in sorted(endpoints):
            cells = endpoints[endpoint]
            base = cells.get("baseline", {})
            if not base:
                continue
            for stratum_label in sorted(strata):
                members = strata[stratum_label]
                for condition in sorted(cells):
                    if condition == "baseline":
                        continue
                    values = {prompt_id: value
                              for prompt_id, value in cells[condition].items()
                              if prompt_id in members}
                    paired = [prompt_id for prompt_id in sorted(values)
                              if prompt_id in base]
                    diffs = [values[prompt_id] - base[prompt_id]
                             for prompt_id in paired]
                    unit = "item"
                    if len(paired) == 1:
                        by_condition = samples.get(endpoint, {})
                        cond_samples = by_condition.get(condition, {}) \
                                                   .get(paired[0], {})
                        base_samples = by_condition.get("baseline", {}) \
                                                   .get(paired[0], {})
                        sample_diffs = [cond_samples[k] - base_samples[k]
                                        for k in sorted(cond_samples)
                                        if k in base_samples]
                        if len(sample_diffs) >= 2:
                            diffs = sample_diffs
                            unit = "sample"
                    if not diffs:
                        continue
                    row = study_stats.effect_row(condition, endpoint, diffs)
                    row.modality = modalities.get(condition, "")
                    row.stratify_by = family_name
                    row.stratum = stratum_label
                    row.unit = unit
                    row.estimand = ("withinItemSamples" if unit == "sample"
                                    else "itemLevel")
                    row.inference = ("diagnostic" if unit == "sample"
                                     else "corrected")
                    family_rows.setdefault(endpoint, []).append(row)
        for endpoint_rows in family_rows.values():
            # Only the item-level rows form a correction family. The
            # diagnostic within-item rows keep adjustedP and correction empty
            # — a blank cell reads as "not tested" to every strict reader on
            # both engines, which is exactly the claim.
            study_stats.apply_correction(
                [row for row in endpoint_rows if row.inference == "corrected"],
                method=method)
        for endpoint in sorted(family_rows):
            rows.extend(family_rows[endpoint])
    return rows


_PRIMARY_ENDPOINT_ORDER = ("choiceLogOdds", "meanMonths", "choiceRate")


def _promotion_decisions(manifest, rows, run_dir, promotion_mod):
    """Assemble per-concept screening evidence from single-concept conditions:
    the primary-endpoint effect, the dose-response over that concept's alpha
    grid, and the matched-norm random floor."""
    from . import study_stats

    def condition_meta(name: str):
        for condition in manifest.conditions:
            if condition.name == name and len(condition.slots) == 1:
                return (condition.slots[0].concept, condition.slots[0].alpha,
                        condition.control_type)
        return (None, None, None)

    primary = next((e for e in _PRIMARY_ENDPOINT_ORDER
                    if any(r.endpoint == e for r in rows)), None)
    if primary is None:
        return []
    primary_rows = [r for r in rows if r.endpoint == primary]
    random_floor: dict[str, float] = {}
    by_concept: dict[str, list[tuple[float, study_stats.EffectRow]]] = {}
    for row in primary_rows:
        concept, alpha, control_type = condition_meta(row.condition)
        if concept is None:
            continue
        if control_type == "randomMatchedNorm":
            previous = random_floor.get(concept)
            magnitude = abs(row.ci.mean)
            random_floor[concept] = max(previous, magnitude) if previous is not None \
                else magnitude
            continue
        by_concept.setdefault(concept, []).append((alpha, row))

    decisions = []
    for concept, dosed in sorted(by_concept.items()):
        dosed.sort(key=lambda pair: pair[0])
        # The concept's headline row: largest-|alpha| treatment cell.
        headline = max(dosed, key=lambda pair: abs(pair[0]))[1]
        dose = None
        if len(dosed) >= 2:
            dose = study_stats.dose_monotonicity(
                [alpha for alpha, _ in dosed], [row.ci.mean for _, row in dosed])
        candidate = promotion_mod.PromotionCandidate(
            concept=concept, condition=headline.condition, endpoint=primary,
            effect=headline, dose=dose,
            random_floor_effect=random_floor.get(concept),
            capability_passed=None,
            provenance={"sourceRun": os.path.basename(run_dir)})
        decisions.append(promotion_mod.decide(candidate, manifest.promotion_rule))
    return decisions


def _latest_run(name: str, root: str | None) -> str | None:
    runs = paths.runs_directory(root)
    if not os.path.isdir(runs):
        return None
    candidates = sorted(
        (e for e in os.listdir(runs)
         if e.endswith(f"-exp-{name}-run")
         and os.path.isfile(os.path.join(runs, e, "generations.jsonl"))),
        reverse=True)
    return os.path.join(runs, candidates[0]) if candidates else None


def _write_metrics_csv(records: list[dict], run_directory: str,
                       style=None) -> None:
    """Write Swift-compatible run metrics for result browsers and notebooks.

    ``style`` (a ``reasoning_style.PinnedStyle``) adds one ``rs_<featureID>``
    column per taxonomy feature, in declared taxonomy order (cross-engine
    contract) — values recomputed from each record's output text. Records
    carrying factorial ``factors`` metadata add one ``factor_<name>`` column
    per factor name (sorted union, appended last — cross-engine contract);
    a factor-less run's header and rows are byte-identical to before."""
    path = os.path.join(run_directory, "metrics.csv")
    feature_ids = style.taxonomy.feature_ids if style else []
    sampled = [r for r in records
               if "error" not in r and "instrument" not in r]
    factor_names = sorted({name for r in sampled
                           for name in (r.get("factors") or {})})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["condition", "seed", "promptIndex", "promptID", "wordCount", "distinct2"]
            + [f"rs_{fid}" for fid in feature_ids]
            + [f"factor_{name}" for name in factor_names])
        for index, record in enumerate(records):
            if "error" in record or "instrument" in record:
                continue  # instrument readouts have no sampled text to score
            row = [
                record.get("condition", ""),
                record.get("seed", 0),
                record.get("promptIndex", index),
                record.get("promptID", ""),
                record.get("wordCount", 0),
                record.get("distinct2", 0.0),
            ]
            if style:
                values = style.taxonomy.score(record.get("output", ""))
                row += [values.get(fid, 0.0) for fid in feature_ids]
            row += [(record.get("factors") or {}).get(name, "")
                    for name in factor_names]
            writer.writerow(row)


def _write_summaries_csv(records: list[dict], run_directory: str) -> None:
    """Per-(condition, prompt) distributional summaries — Case 3's mean AND
    spread endpoints over the sample axis, plus parse-failure and choice rates.
    One row per item cell; the analysis step consumes this alongside the raw
    JSONL."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        if "error" in record:
            continue
        key = (record.get("condition", ""), str(record.get("promptID", "")))
        groups.setdefault(key, []).append(record)
    if not groups:
        return
    header = ["condition", "promptID", "samples",
              "monthsParseFailureRate", "monthsMean", "monthsStdev", "monthsMin",
              "monthsQ25", "monthsMedian", "monthsQ75", "monthsMax",
              "choiceRates", "selectedOption", "targetProbability", "targetLogOdds"]
    path = os.path.join(run_directory, "summaries.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for (condition, prompt_id), items in sorted(groups.items()):
            sampled = [r for r in items if "output" in r]
            row: dict = {"condition": condition, "promptID": prompt_id,
                         "samples": len(sampled)}
            months = [r["parsedMonths"] for r in sampled if "parsedMonths" in r]
            if months:
                row["monthsParseFailureRate"] = f"{judicial.parse_failure_rate(months):.6g}"
                summary = judicial.summarize(months)
                if summary is not None:
                    row.update({
                        "monthsMean": f"{summary.mean:.6g}",
                        "monthsStdev": f"{summary.stdev:.6g}",
                        "monthsMin": f"{summary.minimum:.6g}",
                        "monthsQ25": f"{summary.q25:.6g}",
                        "monthsMedian": f"{summary.median:.6g}",
                        "monthsQ75": f"{summary.q75:.6g}",
                        "monthsMax": f"{summary.maximum:.6g}",
                    })
            choices = [r["parsedChoice"] for r in sampled if "parsedChoice" in r]
            if choices:
                parsed = [c for c in choices if c is not None]
                rates = {option: parsed.count(option) / len(parsed)
                         for option in sorted(set(parsed))} if parsed else {}
                row["choiceRates"] = json.dumps(rates, sort_keys=True)
            for record in items:
                if record.get("instrument") == "answerTokenLogprob":
                    target = record.get("target")
                    row["selectedOption"] = record.get("selected", "")
                    if target is not None:
                        probability = record.get("choiceProbability", {}).get(target)
                        odds = record.get("logOdds", {}).get(target)
                        if probability is not None:
                            row["targetProbability"] = f"{probability:.6g}"
                        if odds is not None:
                            row["targetLogOdds"] = f"{odds:.6g}"
                    break
            writer.writerow(row)


def _reasoning_style_block(sampled: list[dict], style) -> dict | None:
    """Per-condition ``reasoningStyle`` report block (cross-engine contract:
    {"taxonomy", "taxonomyHash", "taxonomyFile", "diagnosticOnly",
    "features": {id: {"mean", "n"}}}): the mean of each feature's
    per-generation values over this condition's sampled outputs — the same
    values, in each feature's own declared normalization units, as the
    ``rs_<featureID>`` metrics.csv columns. ``taxonomyFile`` names the pinned
    taxonomy file (beside its hash) so the report is self-describing, and
    ``diagnosticOnly`` marks these as surface style features — a
    diagnostic/manipulation check reported beside outcome endpoints, never an
    outcome endpoint itself (docs/METHODS.md). None when no taxonomy is
    pinned or nothing was sampled."""
    if style is None or not sampled:
        return None
    scored = [style.taxonomy.score(record.get("output", "")) for record in sampled]
    return {
        "taxonomy": style.taxonomy.name,
        "taxonomyHash": style.hash,
        "taxonomyFile": style.path,
        "diagnosticOnly": True,
        "features": {
            fid: {
                "mean": sum(values.get(fid, 0.0) for values in scored) / len(scored),
                "n": len(scored),
            }
            for fid in style.taxonomy.feature_ids
        },
    }


def _choice_readouts(items: list[dict]) -> dict[tuple, str]:
    """A condition's categorical choices keyed by
    ``(promptID, sampleIndex, source)``: the sampled parse (source
    ``"parsed"``) and the deterministic instrument's selected option (source
    ``"instrument"``). ``None`` parses are excluded — an unparseable output
    can neither agree nor disagree."""
    out: dict[tuple, str] = {}
    for record in items:
        prompt_id = str(record.get("promptID", ""))
        if record.get("instrument") == "answerTokenLogprob":
            selected = record.get("selected")
            if selected is not None:
                out[(prompt_id, record.get("sampleIndex"), "instrument")] = selected
        elif record.get("parsedChoice") is not None:
            out[(prompt_id, record.get("sampleIndex"), "parsed")] = \
                record["parsedChoice"]
    return out


def _write_report(name: str, manifest: Manifest, records: list[dict],
                  run_directory: str, battery: dict | None = None,
                  style=None, numeric_parser=None, sharded=None) -> None:
    """report.json. Choice-bearing runs additionally stamp, per condition:
    ``choiceRate`` (fraction of parseable sampled outputs choosing the item's
    target — same rule as the ``choiceRate`` analysis endpoint) and, for every
    non-baseline condition, the cross-engine parity summary
    ``agreementWithBaseline`` = ``{"n", "agreement"}``: over the choice
    readouts (sampled parses + instrument selections) paired item-for-item
    with baseline's, the fraction that chose the SAME option. A
    no-intervention variant at temperature 0 should sit at agreement 1.0 —
    a cheap, computable baseline-parity check."""
    by_condition: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}
    for r in records:
        if "error" in r:  # a failed variant condition produced no generations
            errors[r["condition"]] = r["error"]
            continue
        by_condition.setdefault(r["condition"], []).append(r)
    baseline_choices = _choice_readouts(by_condition.get("baseline", []))
    conditions = {}
    for cond, error in errors.items():
        conditions[cond] = {"generations": 0, "error": error}
    for cond, items in by_condition.items():
        sampled = [i for i in items if "instrument" not in i]
        n = len(sampled)
        conditions[cond] = {
            "generations": n,
            "meanWordCount": sum(i["wordCount"] for i in sampled) / n if n else 0.0,
            "meanDistinct2": sum(i["distinct2"] for i in sampled) / n if n else 0.0,
        }
        instrument_readouts = sum(1 for i in items if "instrument" in i)
        if instrument_readouts:
            conditions[cond]["choiceReadouts"] = instrument_readouts
        # Ordinal-scale summary (cross-engine contract keys
        # "ordinalMean"/"ordinalSD"; population SD — defined for a single
        # readout). Swift twin: ExperimentTasks.report.
        positions = [i["ordinalPosition"] for i in items
                     if "ordinalPosition" in i]
        if positions:
            mean_position = sum(positions) / len(positions)
            conditions[cond]["ordinalMean"] = mean_position
            conditions[cond]["ordinalSD"] = (
                sum((p - mean_position) ** 2 for p in positions)
                / len(positions)) ** 0.5
        target_hits = [i["parsedChoice"] == i["target"] for i in sampled
                       if i.get("target") is not None
                       and i.get("parsedChoice") is not None]
        if target_hits:
            conditions[cond]["choiceRate"] = sum(target_hits) / len(target_hits)
        if cond != "baseline" and baseline_choices:
            mine = _choice_readouts(items)
            shared = set(mine) & set(baseline_choices)
            if shared:
                agree = sum(1 for key in shared
                            if mine[key] == baseline_choices[key])
                conditions[cond]["agreementWithBaseline"] = {
                    "n": len(shared), "agreement": agree / len(shared)}
        reasoning_style_block = _reasoning_style_block(sampled, style)
        if reasoning_style_block is not None:
            conditions[cond]["reasoningStyle"] = reasoning_style_block
    # Per-condition capability-battery results (cross-engine contract key
    # "capabilityBattery": {"accuracy", "itemCount", "batteryHash"}).
    for cond, block in (battery or {}).items():
        conditions.setdefault(cond, {"generations": 0})["capabilityBattery"] = block
    report = {
        "experiment": name, "experimentHash": manifest.content_hash(),
        "promptMode": manifest.prompt_mode, "conditionCount": len(by_condition),
        "conditions": conditions,
    }
    # Registry-parser provenance (cross-engine contract key "numericParser":
    # {"name", "kind", "registryFile", "registryHash"}): stamped only when a
    # declared parser actually parsed this run's numeric outcome — legacy
    # runs' report bytes are unchanged.
    if numeric_parser is not None:
        report["numericParser"] = numeric_parser.provenance()
    # Shard provenance of a MERGED run ({"shardCount", "shardRuns",
    # "shardJobIDs"?}, deterministic — no timestamps). Only the merge passes
    # it; single-job runs and shard partials never carry the key, and it
    # lives here, never in config.json (closed schema-2 contract) or the
    # manifest.
    if sharded is not None:
        report["sharded"] = sharded
    # What actually RAN, as a set. The manifest's modelID is a declared
    # default; a panel's seats may each name their own, so a single scalar
    # cannot describe the run. config.json's key set is a closed cross-engine
    # contract, so the honest multi-model view lives here.
    used = sorted({r.get("modelID") for r in records if r.get("modelID")})
    if used:
        report["modelsUsed"] = used
        report["declaredModelID"] = manifest.model_id
        if len(used) > 1 or used[0] != manifest.model_id:
            # Per-seat, so a reader can attribute a turn to its model without
            # re-reading generations.jsonl.
            by_seat: dict[str, str] = {}
            for r in records:
                seat, model = r.get("speakerName"), r.get("modelID")
                if seat and model:
                    by_seat[seat] = model
            if by_seat:
                report["modelBySeat"] = by_seat
    with open(os.path.join(run_directory, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
