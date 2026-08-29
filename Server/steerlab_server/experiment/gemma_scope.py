"""Gemma Scope SAE cross-check of a concept vector (parallel to Swift
``GemmaScopeAnalysis`` + ``scripts/gemmascope_analyze.py``), plus the direct
feature-ID import path.

Gemma 3's unique asset: Gemma Scope 2 publishes SAEs/transcoders for every layer
of every size, so a CAA direction can be ranked against independently identified
SAE features. This loads an SAE, cosine-ranks its decoder rows (``W_dec``)
against the steering vector at a layer, and returns the top concept-aligned
features — a port of the existing standalone script onto our
:mod:`vector_store` artifacts. Requires ``sae_lens`` only when ``analyze`` is
called — importing this module works without it (the ``gemmascope`` extra:
``pip install 'steerlab-server[gemmascope]'``; included in ``[all]``).

Two import paths, two conventions, never mixed
----------------------------------------------
1. :func:`import_feature` — REPORT-based. The feature must appear in a cosine
   report, and the decoder row is rescaled to the analyzed concept vector's
   norm (:data:`IMPORT_CONVENTION`, ``"analyzed-vector-norm-match"``).
2. :func:`import_feature_by_id` — DIRECT. A feature chosen on semantic grounds
   (Neuronpedia, an auto-interp label, a prior paper) needs no cosine
   relationship to any CAA direction, so requiring a report would force a
   fictitious pairing. The row is rescaled to the RESIDUAL-STREAM norm at the
   SAE's layer, read from a *calibration donor* artifact
   (:data:`RESIDUAL_NORM_MATCH_CONVENTION`, ``"residual-norm-match"``; see
   ``docs/SAE-VECTOR-INTERVENTION-PROPOSAL-2026-08-13-r2.md`` §5).

Both emit the same immutable ``gemmaScopeSAE`` full-depth artifact (the row at
its layer, zeros elsewhere). The convention stamp says which transform ran; the
two are distinct stamps and are never silently mixed.

Source identity: resolve the commit, THEN read the bytes
--------------------------------------------------------
Every SAE load in this module — the by-id path and :func:`analyze` alike —
resolves the repository and its exact commit sha BEFORE anything is fetched,
and threads that sha into each download (:class:`PinnedSAESource`). The
revision an artifact is stamped with is therefore the revision its weights
came from by construction — not a second metadata query afterwards, which a
re-tagged repository or a stale HF cache can make disagree with the bytes in
hand. Reports record it (``saeRepository``/``saeRevision``) and both import
paths stamp it, with the raw decoder-row hash, into ``gemmascopeSource``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Callable

from ..steering import vector_store


# Gemma Scope 2 residual SAE availability per size/tuning (parallel to Swift
# GemmaScopeCatalog.availableResidualLayers).
_RESIDUAL_LAYERS = {("4b", "it"): [9, 17, 22, 29], ("12b", "it"): [12, 24, 31, 41],
                    ("27b", "it"): [16, 31, 40, 53]}


def _gemma3_size(model_id: str) -> str | None:
    lid = model_id.lower()
    for size in ("4b", "12b", "27b"):
        if "gemma-3-" + size in lid or "gemma3-" + size in lid or f"-{size}-" in lid:
            return size
    return None


def scope_info(model_id: str, *, layer_count: int | None = None,
               preferred_layer: int | None = None) -> dict:
    """Recommended Gemma Scope 2 SAE for a Gemma 3 model (parallel to Swift
    ``GemmaScopeCatalog.info``). Returns ``available: False`` for non-Gemma-3."""
    size = _gemma3_size(model_id)
    if size is None:
        return {"available": False,
                "reason": "Gemma Scope SAEs exist only for Gemma 3 (4b/12b/27b)."}
    tuning = "it"
    layers = _RESIDUAL_LAYERS.get((size, tuning), [])
    if preferred_layer is not None:
        requested = max(0, preferred_layer)
    elif layer_count:
        requested = max(0, layer_count // 2)
    else:
        requested = 12
    layer = min(layers, key=lambda l: abs(l - requested)) if layers else requested
    return {
        "available": True, "size": size, "tuning": tuning,
        "repository": f"google/gemma-scope-2-{size}-{tuning}",
        "release": f"gemma-scope-2-{size}-{tuning}-res",
        "saeID": f"layer_{layer}_width_16k_l0_medium",
        "site": "resid_post", "layer": layer, "availableLayers": layers,
    }


def coerce_layer(value: object, *, label: str = "layer") -> int:
    """A layer index out of untrusted JSON, or :class:`ValueError`.

    Only a JSON integer (``2``) or a finite integral float (``2.0``) names a
    layer; ``2.5``, ``true``, ``"2"``, NaN and Infinity do not. Bare ``int()``
    accepted all of those — ``2.5`` truncated to ``2`` and ``true`` became
    ``1`` — and a truncated layer can then AGREE with an ``saeID`` naming
    layer 2, so the decoder row is imported at a depth nobody asked for while
    every downstream layer/SAE-agreement check passes. That is a silent
    scientific-correctness failure, not a typing nit.

    This brings the Python engine in line with the Swift decoder, where
    ``GemmaScopeReportVector.layer`` is an ``Int``
    (``Sources/ExperimentKit/GemmaScopeReports.swift``) and ``JSONDecoder``
    already refuses a fractional or boolean layer outright.
    """
    ok = (not isinstance(value, bool)) and isinstance(value, (int, float))
    if ok and isinstance(value, float):
        ok = math.isfinite(value) and value.is_integer()
    if not ok:
        raise ValueError(
            f"{label} must be an integer, got {value!r} — a layer is a depth "
            "index, so a fractional, boolean, string or non-finite value "
            "cannot name one, and truncating it would place the decoder row "
            "at a depth nothing analyzed")
    return int(value)


@dataclass
class FeatureRow:
    feature: int
    cosine: float
    sparsity: float | None = None
    decoder_values: list[float] | None = None
    # The decoder row's own L2 norm BEFORE any import rescale. Descriptive
    # geometry (SAE decoder norms vary by an order of magnitude across a
    # dictionary) and the number the residual-norm-match convention divides
    # out, so a reader can reconstruct the transform from the report alone.
    raw_decoder_norm: float | None = None
    # True on rows present because the caller asked for the feature by id
    # (``requestedFeatureIDs``), not because it ranked. Set on the rows in the
    # ``requested`` bucket; a requested feature that ALSO ranks appears in both
    # buckets, flagged only in this one.
    requested: bool = False


@dataclass
class GemmaScopeReport:
    release: str
    sae_id: str
    layer: int
    decoder_shape: list[int]
    top_positive: list[FeatureRow]
    top_negative: list[FeatureRow]
    top_absolute: list[FeatureRow]
    # Rows for caller-named feature ids, included whether or not they rank —
    # so ONE report carries both the CAA-aligned top-k and a semantically
    # chosen shortlist (proposal §8 P0-3). Empty on every report that did not
    # ask for any, and then absent from the JSON entirely: existing reports
    # keep their exact shape.
    requested: list[FeatureRow] = field(default_factory=list)
    # Analyzed-vector context (parallel to the Swift report's ``vector`` block
    # + embedded ``artifactSidecar``): the concept vector's L2 norm at the
    # analyzed layer is the TARGET of the import convention's rescale, and the
    # source sidecar supplies layer count / residual norms / model identity so
    # an imported feature is a full peer artifact. ``None`` only on legacy
    # reports (predating WS7.2), which ``import_feature`` refuses.
    vector_norm: float | None = None
    vector_concept: str | None = None
    artifact_sidecar: dict | None = None
    # The SAE's own source identity: the repository the decoder rows were read
    # from and the EXACT commit resolved before the fetch (the pinned path's
    # rule). Additive keys — ``None`` on legacy reports, which then import
    # without SAE source identity rather than refusing.
    sae_repository: str | None = None
    sae_revision: str | None = None

    def to_dict(self) -> dict:
        def rows(rs):
            out = []
            for r in rs:
                row = {"feature": r.feature, "cosine": r.cosine,
                       "sparsity": r.sparsity, "decoderValues": r.decoder_values}
                # Additive keys, emitted only when they carry information, so
                # every existing consumer (Swift ``GemmaScopeFeatureRow``, the
                # web table) keeps decoding these rows unchanged.
                if r.raw_decoder_norm is not None:
                    row["rawDecoderNorm"] = r.raw_decoder_norm
                if r.requested:
                    row["requested"] = True
                out.append(row)
            return out
        out = {
            "release": self.release, "saeID": self.sae_id, "layer": self.layer,
            "decoderShape": self.decoder_shape,
            "topPositive": rows(self.top_positive),
            "topNegative": rows(self.top_negative),
            "topAbsolute": rows(self.top_absolute),
        }
        if self.requested:
            out["requested"] = rows(self.requested)
        if self.vector_norm is not None:
            out["vectorNorm"] = self.vector_norm
        if self.vector_concept is not None:
            out["vectorConcept"] = self.vector_concept
        if self.artifact_sidecar is not None:
            out["artifactSidecar"] = self.artifact_sidecar
        if self.sae_repository is not None:
            out["saeRepository"] = self.sae_repository
        if self.sae_revision is not None:
            out["saeRevision"] = self.sae_revision
        return out


def resolve_requested_features(requested: list[int] | None,
                               feature_count: int) -> list[int]:
    """De-duplicated, order-preserving requested feature ids, bounds-checked.

    Pure (no SAE, no torch) so the requested-ID contract is testable offline.
    An out-of-range id REFUSES rather than being dropped: a typo'd feature
    number that silently vanishes from the report would read as "the feature
    we asked about did not rank", which is exactly the wrong conclusion.
    """
    if not requested:
        return []
    seen: list[int] = []
    for raw in requested:
        value = int(raw)
        if value < 0 or value >= feature_count:
            raise ValueError(
                f"requested feature {value} is outside this SAE's dictionary "
                f"(0..{feature_count - 1}) — check the feature id against the "
                f"release/SAE it was found in")
        if value not in seen:
            seen.append(value)
    return seen


def _row_values(row) -> list[float]:
    """Decoder row as plain floats, from a torch tensor / numpy array / list."""
    values = row.tolist() if hasattr(row, "tolist") else list(row)
    return [float(x) for x in values]


def feature_rows(indices, *, cosines, decoder_rows, sparsity=None,
                 requested: bool = False) -> list[FeatureRow]:
    """Build report rows for ``indices``. Pure and framework-agnostic —
    ``cosines``/``decoder_rows``/``sparsity`` need only be indexable — so both
    the top-k and requested buckets come from one tested code path."""
    import numpy as np
    out: list[FeatureRow] = []
    for i in indices:
        index = int(i)
        values = _row_values(decoder_rows[index])
        norm = float(np.sqrt(np.square(np.asarray(values, dtype=np.float32))
                             .sum(dtype=np.float32)))
        out.append(FeatureRow(
            feature=index, cosine=float(cosines[index]),
            sparsity=(float(sparsity[index]) if sparsity is not None else None),
            decoder_values=values, raw_decoder_norm=norm, requested=requested))
    return out


def analyze(vector_directory: str, name: str, *, layer: int, release: str,
            sae_id: str, top_k: int = 25, output_path: str | None = None,
            requested_feature_ids: list[int] | None = None,
            resolver: RepoRevisionResolver | None = None,
            builder: PinnedSAELoader | None = None) -> GemmaScopeReport:
    """Cosine-rank an SAE's decoder rows against a stored concept vector.

    ``requested_feature_ids`` adds rows for caller-named features regardless of
    rank (proposal §8 P0-3), so one report can carry both the CAA-aligned top-k
    and a semantically chosen shortlist. Requested rows are flagged, never
    merged into the ranked buckets: what ranked and what was asked for must stay
    distinguishable downstream.

    Refuses (never clamps, never guesses): a ``layer`` outside the analyzed
    artifact's depth, and a ``layer`` that disagrees with the SAE's own layer
    (its id grammar, or its published config after the load) — an SAE's
    dictionary lives at exactly one layer, and ``import_feature`` places the
    decoder row at the REPORT layer, so a mismatched analysis would import a
    steering artifact at a depth the SAE does not describe.

    The SAE loads through the pinned resolve-then-fetch path (the module-note
    rule every other loader here follows): the repository commit is resolved
    before any byte is read, and the report records it (``saeRepository`` /
    ``saeRevision``) so report-path imports carry the same source identity as
    by-id imports.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "Gemma Scope analysis needs sae-lens and torch: "
            "pip install 'steerlab-server[gemmascope]' (or: pip install sae-lens)") from exc

    vectors, source_sidecar = vector_store.load(vector_directory, name)
    if not (0 <= layer < vectors.layer_count):
        raise ValueError(
            f"layer {layer} is outside the analyzed artifact "
            f"({vectors.layer_count} layers) — analyze at a layer the concept "
            "vector actually has")
    parsed_layer = parse_sae_id(sae_id).get("layer")
    if parsed_layer is not None and parsed_layer != layer:
        raise ValueError(
            f"analysis layer {layer} disagrees with the SAE's own layer "
            f"{parsed_layer} ({sae_id}) — an SAE's dictionary lives at exactly "
            "one layer, and the report's feature rows import at the report "
            "layer; analyze at the SAE's layer or pick an SAE at this layer")
    vector = torch.tensor(vectors.per_layer[layer], dtype=torch.float32)

    source = resolve_pinned_source(release, sae_id, resolver=resolver)
    sae, cfg, sparsity = (builder or load_pinned_sae)(source)
    config_layer = _config_layer(cfg if isinstance(cfg, dict) else {})
    if config_layer is not None and config_layer != layer:
        raise ValueError(
            f"analysis layer {layer} disagrees with the loaded SAE's published "
            f"config, which places it at layer {config_layer} ({sae_id}) — "
            "an SAE's dictionary lives at exactly one layer; analyze at the "
            "SAE's layer or pick an SAE at this layer")
    decoder = getattr(sae, "W_dec", None)
    if decoder is None:
        raise RuntimeError("loaded SAE does not expose W_dec")
    decoder = decoder.detach().float().cpu()
    if decoder.ndim != 2:
        raise RuntimeError(f"expected a 2-D decoder, got {tuple(decoder.shape)}")
    if decoder.shape[1] != vector.numel() and decoder.shape[0] == vector.numel():
        decoder = decoder.T
    if decoder.shape[1] != vector.numel():
        raise RuntimeError(
            f"decoder/vector mismatch: W_dec={tuple(decoder.shape)}, vector={vector.numel()}")

    vector = vector / vector.norm().clamp_min(1e-12)
    decoder = decoder / decoder.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scores = decoder @ vector

    k = min(top_k, scores.numel())
    sparsity_list = None
    if sparsity is not None:
        try:
            sparsity_list = sparsity.detach().float().cpu().tolist()
        except Exception:  # noqa: BLE001
            sparsity_list = None

    # The (model-hidden-dim) decoder direction per feature, so a feature can be
    # imported as a steering vector later (parallel to Swift importFeature).
    raw_decoder = getattr(sae, "W_dec").detach().float().cpu()
    if raw_decoder.shape[1] != vector.numel() and raw_decoder.shape[0] == vector.numel():
        raw_decoder = raw_decoder.T

    def rows(indices, *, requested=False):
        return feature_rows(indices, cosines=scores, decoder_rows=raw_decoder,
                            sparsity=sparsity_list, requested=requested)

    requested_ids = resolve_requested_features(requested_feature_ids,
                                               int(scores.numel()))

    report = GemmaScopeReport(
        release=release, sae_id=sae_id, layer=layer,
        decoder_shape=list(decoder.shape),
        top_positive=rows(torch.topk(scores, k=k).indices.tolist()),
        top_negative=rows(torch.topk(-scores, k=k).indices.tolist()),
        top_absolute=rows(torch.topk(scores.abs(), k=k).indices.tolist()),
        requested=rows(requested_ids, requested=True),
        vector_norm=vectors.norm(layer),
        vector_concept=source_sidecar.concept,
        artifact_sidecar=source_sidecar.to_dict(),
        sae_repository=source.repo_id,
        sae_revision=source.revision)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
        print(f"wrote {output_path}")
    return report


# The decided cross-engine Gemma Scope import convention (WS7.2) — the Swift
# app's historical behavior, now shared: the SAE decoder row is rescaled AT
# IMPORT so its L2 norm equals the analyzed concept vector's L2 norm at the
# report layer (norm-matched, like LAT is norm-matched to the mean difference,
# so α at a given layer produces the same perturbation magnitude for the SAE
# feature as for the concept vector it was ranked against). Same constant
# string as Swift ``GemmaScopeReportCatalog.importConvention``.
IMPORT_CONVENTION = "analyzed-vector-norm-match"


def _convention_rescale(values: list[float], target_norm: float) -> tuple[list[float], float]:
    """Rescale a decoder row to ``target_norm`` (float32 math, mirroring the
    Swift ``scale(_:toNorm:)`` exactly, including its degenerate guard: a
    zero-norm row or non-positive target is returned unscaled). Returns
    ``(scaled_values, raw_decoder_norm)``."""
    import numpy as np
    arr = np.asarray(values, dtype=np.float32)
    raw = float(np.sqrt(np.square(arr).sum(dtype=np.float32)))
    if raw <= 0 or target_norm <= 0:
        return [float(x) for x in arr.tolist()], raw
    factor = np.float32(target_norm) / np.float32(raw)
    return [float(x) for x in (arr * factor).astype(np.float32).tolist()], raw


def import_feature(report_path: str, feature: int, *, model_id: str,
                   run_directory: str) -> str:
    """Import an SAE feature's decoder direction as a steering-vector artifact
    (parallel to Swift ``GemmaScopeReportCatalog.importFeature``, and now the
    SAME transform at the SAME stage): the decoder row is rescaled at import
    per :data:`IMPORT_CONVENTION` and placed at the report's layer inside a
    full-depth zero artifact, carrying the analyzed artifact's model identity
    and residual-norm calibration so norm-unit alphas keep meaning.

    Refuses reports that predate the convention (no ``vectorNorm`` /
    ``artifactSidecar``): re-run the analysis, then import. Also refuses a
    report whose ``layer`` disagrees with its own ``saeID`` — ``analyze`` now
    refuses to produce one, but a pre-guard report would otherwise import a
    decoder row at a depth the SAE does not describe. ``model_id`` is a
    fallback only — the embedded sidecar's model identity wins.
    """
    from ..build_identity import engine_version
    from ..steering.vector_store import ConceptVectors, SteeringVectorSidecar, save
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    # Refuse BEFORE anything is written: a non-integer report layer is not a
    # layer, and truncating it would import at a depth the analysis never
    # touched (see :func:`coerce_layer`).
    try:
        layer = coerce_layer(report.get("layer"), label="report layer")
    except ValueError as exc:
        raise ValueError(
            f"{exc}; re-run the Gemma Scope analysis, then import") from exc
    # The FEATURE has the same predicate (round-11 follow-up): the row match
    # below used int() on both sides, so "7" and 7.5 both selected a row —
    # and a truncated feature imports the wrong dictionary entry outright.
    feature = coerce_layer(feature, label="feature")
    release = report.get("release", "")
    sae_id = report.get("saeID", "")
    sae_layer = parse_sae_id(sae_id).get("layer")
    if sae_layer is not None and sae_layer != layer:
        raise ValueError(
            f"report layer {layer} disagrees with the SAE's own layer "
            f"{sae_layer} ({sae_id}) — an SAE's dictionary lives at exactly "
            "one layer, so this report would import the decoder row at a depth "
            "the SAE does not describe; re-run the Gemma Scope analysis at the "
            "SAE's layer, then import")
    values = None
    for bucket in ("topAbsolute", "topPositive", "topNegative"):
        for row in report.get(bucket, []):
            # BOTH sides of the match, not just the request (external review
            # round 12, finding 6): the requested id got the exact-integer
            # predicate above, but the row side still ran through ``int()``,
            # so a report row whose feature is ``7.5`` truncated to 7 and
            # matched a request for feature 7, and ``true`` became 1. A row
            # that cannot NAME a dictionary entry is not that entry's row —
            # it is skipped, and if nothing else matches the not-found
            # refusal below fires, which is the honest answer.
            try:
                row_feature = coerce_layer(row.get("feature"),
                                           label="report row feature")
            except ValueError:
                continue
            if row_feature == feature and row.get("decoderValues"):
                values = row["decoderValues"]
                break
        if values:
            break
    if values is None:
        raise ValueError(f"feature {feature} not found in report (or no decoder values)")

    target_norm = report.get("vectorNorm")
    source = report.get("artifactSidecar")
    if target_norm is None or not isinstance(source, dict):
        raise ValueError(
            "report predates the aligned Gemma Scope import convention "
            f"({IMPORT_CONVENTION!r}: no analyzed-vector norm / source sidecar) "
            "— re-run the Gemma Scope analysis, then import the feature")
    src = SteeringVectorSidecar.from_dict(source)
    if len(values) != src.hiddenSize:
        raise ValueError(
            f"feature {feature} decoder has dimension {len(values)}, "
            f"expected {src.hiddenSize}")
    if not (0 <= layer < src.layerCount):
        raise ValueError(
            f"report layer {layer} is outside the analyzed artifact "
            f"({src.layerCount} layers)")

    raw_values = [float(x) for x in values]
    scaled, raw_norm = _convention_rescale(raw_values, float(target_norm))
    # The applied/skipped marker, same shape as the by-id path's: when the
    # degenerate guard returned the row RAW (zero-norm row, non-positive
    # target), the artifact says so instead of stamping a transform that
    # never ran.
    rescale: dict = {"convention": IMPORT_CONVENTION,
                     "rawDecoderNorm": raw_norm,
                     "targetNorm": float(target_norm), "applied": True}
    if raw_norm <= 0 or float(target_norm) <= 0:
        rescale["applied"] = False
        rescale["skippedReason"] = (
            "zero-norm decoder row" if raw_norm <= 0 else
            "non-positive analyzed-vector norm")
    per_layer = [[0.0] * src.hiddenSize for _ in range(src.layerCount)]
    per_layer[layer] = scaled
    vectors = ConceptVectors(per_layer=per_layer)
    name = f"sae-feature-{feature}"
    # The report path's source identity, mirroring the by-id path's stamp so
    # the identity chain (repository commit + raw-row hash) covers both kinds
    # of import. Repository/revision exist only on reports from the pinned
    # ``analyze``; a legacy report imports without them rather than refusing.
    source_stamp: dict = {
        "importPath": "cosine-report",
        "release": release, "saeID": sae_id, "feature": int(feature),
        "layer": layer,
        "decoderRowHash": decoder_row_hash(raw_values),
        "rescale": rescale,
        "reportPath": os.path.basename(report_path),
        "importedBy": engine_version(),
        "importedAt": _now_iso8601(),
        "substrate": vector_store.SUBSTRATE,
    }
    if report.get("saeRepository"):
        source_stamp["repository"] = str(report["saeRepository"])
    if report.get("saeRevision"):
        source_stamp["repositoryRevision"] = str(report["saeRevision"])
    sidecar = SteeringVectorSidecar(
        modelID=src.modelID or model_id,
        concept=f"sae:{src.concept}:L{layer}:F{int(feature)}",
        stimulusSetHash=f"gemmascope:{release}:{sae_id}:{int(feature)}",
        layerCount=vectors.layer_count, hiddenSize=vectors.hidden_size,
        normsPerLayer=[vectors.norm(i) for i in range(vectors.layer_count)],
        extractionDate=_now_iso8601(),
        revision=src.revision,
        extractionMethod="gemmaScopeSAE",
        coversModelDepth=_inherited_depth_coverage(src),
        residualNormPerLayer=src.residualNormPerLayer,
        residualNormSource=src.residualNormSource,
        recipeHash=f"{release}|{sae_id}|feature:{int(feature)}",
        recipeName=(f"Gemma Scope SAE feature {int(feature)} from "
                    f"{os.path.basename(report_path)}, scaled to analyzed vector norm"),
        gemmascopeConvention=IMPORT_CONVENTION,
        rawDecoderNorm=raw_norm,
        gemmascopeTargetNorm=float(target_norm),
        gemmascopeSource=source_stamp)
    save(vectors, sidecar, run_directory, name)
    return os.path.join(run_directory, name)


def _inherited_depth_coverage(source) -> bool | None:
    """An SAE import writes one row per layer of the artifact it was calibrated
    against, so it states the model's depth exactly when THAT artifact did.

    Absent (the common case — the donor is a full-depth concept vector) leaves
    the import reading as full, the same as every other extraction family. An
    explicit ``false`` is what stops a PARTIAL donor's row count from
    laundering itself into a full-looking import: the donor's own method stamp
    does not survive the copy, so the fact has to be carried forward here."""
    from . import catalog
    covers = catalog.covers_model_depth(
        covers=source.coversModelDepth,
        extraction_method=source.extractionMethod,
        recipe_method=source.recipeMethod)
    return None if covers else False


def _now_iso8601() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Direct feature-ID import (proposal r2 §5 / §8 P0-1, P0-2)
# --------------------------------------------------------------------------

# The convention for a feature chosen by ID rather than by cosine rank: the
# decoder row is rescaled AT IMPORT so its L2 norm equals the RESIDUAL-STREAM
# norm at the SAE's layer, taken from a calibration-donor artifact's
# ``residualNormPerLayer``. Distinct from :data:`IMPORT_CONVENTION` and never
# silently mixed with it — the stamps name different transforms with different
# denominators, and only the stamp can tell two stored rows apart after the
# fact. (Storage scale is cosmetic under norm-unit alphas, which re-normalize
# at injection; it makes RAW-α surfaces like chat behave sanely — raw α then
# reads as a fraction of the typical residual magnitude.)
RESIDUAL_NORM_MATCH_CONVENTION = "residual-norm-match"


# --------------------------------------------------------------------------
# Pinned source: the revision is decided BEFORE any bytes are read
# --------------------------------------------------------------------------
#
# The defect this closes (review finding 3a): the loaders used to call
# ``SAE.from_pretrained`` FIRST and ask the Hub what the repository's current
# commit was AFTERWARDS. Two ways that stamps a revision which never described
# the loaded bytes:
#
#   1. ``from_pretrained`` resolves the floating ``main`` ref (and can serve
#      an older blob straight from the HF cache), while the later metadata
#      call reports whatever ``main`` points at NOW;
#   2. even with a live fetch, the repository can be re-tagged between the two
#      calls — a plain TOCTOU window.
#
# So the order is inverted and the pin is threaded through: resolve
# repository + exact commit sha first, then hand that sha to every file fetch.
# The stamped ``repositoryRevision`` is the revision the bytes came from BY
# CONSTRUCTION, not by a second query that could disagree.


@dataclass(frozen=True)
class PinnedSAESource:
    """The exact published bytes an import is permitted to read."""

    release: str
    sae_id: str
    #: The Hugging Face repository holding the SAE.
    repo_id: str
    #: The repository-relative folder holding this SAE's files.
    folder_name: str
    #: The EXACT commit sha. Resolved before the first byte is fetched, and
    #: passed to every fetch — a floating ref is not provenance.
    revision: str


#: ``repo_id -> commit sha``. The Hub-metadata boundary, injectable so the
#: resolve-then-fetch ORDER is testable with no network.
RepoRevisionResolver = Callable[[str], str]

#: ``(repo_id, filename, revision) -> local path``. The download boundary,
#: injectable for the same reason. Every call carries a revision: there is no
#: code path in this module that fetches SAE bytes without one.
PinnedFileFetcher = Callable[[str, str, str], str]

#: The published Gemma Scope layout: one folder per SAE, holding these files.
SAE_PARAMS_FILENAME = "params.safetensors"
SAE_CONFIG_FILENAME = "config.json"

#: ``params.safetensors`` key -> the sae_lens state-dict name. Same mapping
#: sae_lens's own ``gemma_3_sae_huggingface_loader`` applies; reproduced here
#: (rather than called) only so the DOWNLOAD can be pinned — sae_lens's loader
#: fetches through ``hf_hub_download`` with no revision.
_GEMMA_SCOPE_TENSORS = {"w_enc": "W_enc", "w_dec": "W_dec", "b_enc": "b_enc",
                        "b_dec": "b_dec", "threshold": "threshold"}


def _sae_with_cfg_and_sparsity(sae_cls, release: str, sae_id: str,
                               **kwargs) -> tuple:
    """``(sae, cfg_dict, sparsity)`` across sae_lens majors.

    sae_lens 6.x returns the SAE alone from ``from_pretrained`` and moved the
    triple to ``from_pretrained_with_cfg_and_sparsity``; 5.x returned the
    triple from ``from_pretrained``. Both are handled explicitly because the
    difference is silent at import time and a TypeError deep inside a cluster
    job is a bad way to discover it.
    """
    triple = getattr(sae_cls, "from_pretrained_with_cfg_and_sparsity", None)
    if triple is not None:
        return triple(release=release, sae_id=sae_id, **kwargs)
    return sae_cls.from_pretrained(release=release, sae_id=sae_id, **kwargs)


def _sae_lens_repo_and_folder(release: str,
                              sae_id: str) -> tuple[str | None, str | None]:
    """``(repo_id, folder)`` from sae_lens's pretrained directory, or
    ``(None, None)`` when sae_lens is absent or does not know the release.

    A ``ValueError`` from sae_lens (an sae_id that is not in a known release)
    propagates: that is a mis-specified import, not a reason to fall back to a
    guessed repository.
    """
    import importlib
    for module in ("sae_lens.loading.pretrained_sae_loaders", "sae_lens"):
        try:
            resolver = getattr(importlib.import_module(module),
                               "get_repo_id_and_folder_name")
        except (ImportError, AttributeError):  # pragma: no cover - optional dep
            continue
        repo_id, folder = resolver(release, sae_id)
        return (str(repo_id) if repo_id else None,
                str(folder) if folder else None)
    return (None, None)


def resolve_pinned_source(release: str, sae_id: str, *,
                          resolver: RepoRevisionResolver | None = None
                          ) -> PinnedSAESource:
    """Resolve repository, folder and EXACT commit — before anything loads.

    Refuses rather than guessing when the commit cannot be resolved: an
    artifact stamped with a floating ref would claim provenance it does not
    have.
    """
    repo_id, folder = _sae_lens_repo_and_folder(release, sae_id)
    if not repo_id:
        repo_id = repository_for_release(release)
    if not repo_id:
        raise ValueError(
            f"cannot resolve the Hugging Face repository for release "
            f"{release!r} — pass a loader that knows it")
    revision = (resolver or _resolve_repo_revision)(repo_id)
    if not revision:
        raise ValueError(f"'{repo_id}' returned no commit sha to pin")
    return PinnedSAESource(release=release, sae_id=sae_id, repo_id=repo_id,
                           folder_name=(folder or sae_id),
                           revision=str(revision))


def _hf_fetch(repo_id: str, filename: str,
              revision: str) -> str:  # pragma: no cover - network
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo_id, filename=filename,
                           revision=revision)


def _published_config(source: PinnedSAESource, folder: str,
                      fetch: PinnedFileFetcher) -> dict:
    """The SAE's published ``config.json`` at the pinned revision, or ``{}``.

    Absent is legal (some Gemma Scope folders publish none) and is NOT an
    error: everything load-bearing — dimensions, layer, site — is derived from
    the weights and the folder name, which are pinned too.
    """
    try:
        path = fetch(source.repo_id, f"{folder}/{SAE_CONFIG_FILENAME}",
                     source.revision)
    except Exception:  # noqa: BLE001 - a missing optional file is not an error
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _gemma_scope_hook_name(folder: str, layer: int) -> str:
    """The sae_lens hook name for a Gemma Scope folder (same rule as
    ``get_gemma_3_config_from_hf``). Transcoder/CLT folders REFUSE: they carry
    a second hook and a summed decoder, and a single-feature residual import
    has no meaning for them."""
    if "transcoder" in folder or "clt" in folder:
        raise RuntimeError(
            f"{folder!r} is a transcoder/CLT dictionary — the pinned "
            "single-feature import path covers residual/attention/MLP SAEs, "
            "whose decoder row is a direction in the residual stream")
    if "resid_post" in folder:
        return f"blocks.{layer}.hook_resid_post"
    if "attn_out" in folder:
        return f"blocks.{layer}.attn.hook_z"
    if "mlp_out" in folder:
        return f"blocks.{layer}.hook_mlp_out"
    raise RuntimeError(
        f"cannot tell which site {folder!r} reads — the Gemma Scope layout "
        "names it in the folder (resid_post / attn_out / mlp_out)")


def pinned_gemma_scope_converter(source: PinnedSAESource, *,
                                 fetch: PinnedFileFetcher | None = None):
    """An sae_lens ``converter`` that reads ONLY ``source.revision``.

    ``SAE.from_pretrained(converter=…)`` is sae_lens's own seam for supplying
    the (config, weights, sparsity) triple, so this keeps sae_lens building the
    SAE object — same class, same config defaulting, same dtype/device
    handling — and changes exactly one thing: where the bytes come from. It is
    the least invasive mechanism that makes the pin real, because sae_lens's
    published loaders take no revision argument at any level.
    """
    fetcher = fetch or _hf_fetch

    def convert(*, repo_id: str, folder_name: str, device: str = "cpu",
                force_download: bool = False,
                cfg_overrides: dict | None = None):
        if repo_id and repo_id != source.repo_id:
            # The revision was resolved for a DIFFERENT repository, so it
            # cannot describe these bytes. Refuse rather than stamp it.
            raise RuntimeError(
                f"sae_lens resolved repository {repo_id!r} for release "
                f"{source.release!r}, but the revision {source.revision!r} was "
                f"pinned for {source.repo_id!r} — refusing to load bytes whose "
                "commit was never resolved")
        folder = folder_name or source.folder_name
        params_path = fetcher(source.repo_id,
                              f"{folder}/{SAE_PARAMS_FILENAME}",
                              source.revision)
        published = _published_config(source, folder, fetcher)

        from safetensors.torch import load_file
        raw = load_file(params_path, device=device)
        missing = [key for key in _GEMMA_SCOPE_TENSORS if key not in raw]
        if missing:
            raise RuntimeError(
                f"{source.repo_id}/{folder}/{SAE_PARAMS_FILENAME} at "
                f"{source.revision[:12]}… is missing {', '.join(missing)} — "
                "this is not the published Gemma Scope layout, so the pinned "
                "loader refuses rather than guessing at the weights")
        state_dict = {name: raw[key]
                      for key, name in _GEMMA_SCOPE_TENSORS.items()}

        layer = parse_sae_id(folder).get("layer")
        if layer is None:
            layer = parse_sae_id(source.sae_id).get("layer")
        if layer is None:
            raise RuntimeError(
                f"cannot extract the layer from {folder!r} — the Gemma Scope "
                "layout names it ('layer_<n>_…')")
        d_in, d_sae = (int(x) for x in raw["w_enc"].shape)
        model_name = str(published.get("model_name")
                         or source.repo_id.replace("gemma-scope-2", "gemma-3"))
        if "/" not in model_name:
            model_name = "google/" + model_name
        model_name = model_name.replace("-v3", "-3")
        # Same cfg keys sae_lens's gemma_3 loader produces; from_pretrained
        # runs handle_config_defaulting over them afterwards.
        cfg_dict = {
            "architecture": "jumprelu",
            "d_in": d_in, "d_sae": d_sae, "dtype": "float32",
            "model_name": model_name,
            "hook_name": _gemma_scope_hook_name(folder, int(layer)),
            "hook_head_index": None,
            "finetuning_scaling_factor": False,
            "sae_lens_training_version": None,
            "prepend_bos": True,
            "dataset_path": "monology/pile-uncopyrighted",
            "context_size": 1024,
            "apply_b_dec_to_input": bool(
                published.get("apply_b_dec_to_input", False)),
            "normalize_activations": "none",
            "reshape_activations": "none",
            "hf_hook_name": published.get("hf_hook_point_in"),
            "device": device,
        }
        if cfg_overrides:
            cfg_dict.update(cfg_overrides)
        # Gemma Scope publishes no per-feature sparsity tensor.
        return cfg_dict, state_dict, None

    return convert


def load_pinned_sae(source: PinnedSAESource, *,
                    fetch: PinnedFileFetcher | None = None) -> tuple:
    """``(sae, cfg_dict, sparsity)`` for ``source``, every byte fetched at
    ``source.revision``."""
    try:
        from sae_lens import SAE
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "Gemma Scope import needs sae-lens and torch: "
            "pip install 'steerlab-server[gemmascope]'") from exc
    return _sae_with_cfg_and_sparsity(
        SAE, source.release, source.sae_id,
        converter=pinned_gemma_scope_converter(source, fetch=fetch))


#: ``PinnedSAESource -> (sae, cfg_dict, sparsity)``. The whole load boundary
#: as ONE seam, so a test can assert that what a loader stamps is what it
#: read, without sae_lens, torch, or the network.
PinnedSAELoader = Callable[[PinnedSAESource], tuple]


def _pinned_sae(release: str, sae_id: str, *,
                resolver: RepoRevisionResolver | None,
                builder: PinnedSAELoader | None) -> tuple:
    """``(source, sae, cfg, sparsity)`` — resolve FIRST, then load pinned."""
    source = resolve_pinned_source(release, sae_id, resolver=resolver)
    sae, cfg, sparsity = (builder or load_pinned_sae)(source)
    return source, sae, cfg, sparsity


@dataclass
class LoadedSAEFeature:
    """One decoder row plus the provenance needed to pin where it came from.

    Produced by an *injectable* loader (:data:`SAEFeatureLoader`) so the import
    logic — validation, rescale, stamping — is testable with no network, no HF
    cache, and no ``sae_lens``.
    """

    decoder_row: list[float]
    #: The Hugging Face repository the SAE was read from (e.g.
    #: ``google/gemma-scope-2-27b-it``).
    repo_id: str
    #: The EXACT commit sha resolved at import time. A floating ref (``main``)
    #: is not provenance: the repository can be re-tagged under it.
    repo_revision: str
    #: The SAE's own config as published, verbatim (hashed into the sidecar).
    config: dict = field(default_factory=dict)
    #: This feature's published sparsity/L0 statistic, when the release has one.
    sparsity: float | None = None


#: ``(release, sae_id, feature) -> LoadedSAEFeature``.
SAEFeatureLoader = Callable[[str, str, int], LoadedSAEFeature]


def repository_for_release(release: str) -> str | None:
    """Best-effort ``google/gemma-scope-2-<size>-<tuning>`` for a release id.

    Only a FALLBACK for the default loader — ``sae_lens``'s pretrained
    directory is authoritative when it knows the release.
    """
    name = (release or "").strip()
    if not name:
        return None
    if "/" in name:  # already a repo id
        return name
    for suffix in ("-res", "-att", "-mlp", "-transcoder"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return f"google/{name}"


def parse_sae_id(sae_id: str) -> dict:
    """``layer_40_width_65k_l0_medium`` → layer/width/L0 target.

    The Gemma Scope id grammar is the only place some of this appears for
    releases whose config omits it; config values win where both exist.
    """
    text = sae_id or ""
    layer = re.search(r"layer_(\d+)", text)
    width = re.search(r"width_([0-9]+[kKmM]?)", text)
    l0 = re.search(r"l0_([A-Za-z0-9]+)", text)
    return {"layer": int(layer.group(1)) if layer else None,
            "width": width.group(1) if width else None,
            "l0Target": l0.group(1) if l0 else None}


def _config_layer(config: dict) -> int | None:
    for key in ("hook_layer", "layer"):
        value = config.get(key)
        if isinstance(value, int):
            return value
    hook = config.get("hook_name") or config.get("hook_point") or ""
    match = re.search(r"(?:blocks|layers)\.(\d+)", str(hook))
    return int(match.group(1)) if match else None


def _config_site(config: dict) -> str | None:
    hook = str(config.get("hook_name") or config.get("hook_point") or "")
    match = re.search(r"hook_(\w+)$", hook)
    if match:
        return match.group(1)
    site = config.get("site")
    return str(site) if site else None


def _sha256_hex(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sae_config_hash(config: dict) -> str:
    """Hash of the SAE config as published — canonical JSON, so the same config
    hashes identically on both engines and across library versions."""
    text = json.dumps(config or {}, sort_keys=True, separators=(",", ":"),
                      default=str)
    return _sha256_hex(text.encode("utf-8"))


def decoder_row_hash(values: list[float]) -> str:
    """Hash of the RAW decoder row's float32 little-endian bytes.

    Pinned pre-rescale: the stored bytes depend on the calibration donor, the
    published direction does not, so this is what identifies "the same feature
    from the same dictionary" across imports.
    """
    import numpy as np
    return _sha256_hex(np.asarray(values, dtype="<f4").tobytes())


def load_sae_feature(release: str, sae_id: str, feature: int, *,
                     resolver: RepoRevisionResolver | None = None,
                     builder: PinnedSAELoader | None = None
                     ) -> LoadedSAEFeature:
    """Default (online) loader: pin the repository commit, THEN fetch the SAE.

    The one function in this path that touches the network / HF cache, and the
    seam every test substitutes. The commit is resolved before any byte is
    read and threaded into the fetch, so ``repo_revision`` describes the bytes
    in hand by construction. Refuses rather than guessing when the exact
    commit cannot be resolved — an artifact stamped with a floating ref would
    claim provenance it does not have.
    """
    source, sae, cfg, sparsity = _pinned_sae(release, sae_id,
                                             resolver=resolver,
                                             builder=builder)
    decoder = getattr(sae, "W_dec", None)
    if decoder is None:
        raise RuntimeError("loaded SAE does not expose W_dec")
    decoder = decoder.detach().float().cpu()
    if decoder.ndim != 2:
        raise RuntimeError(f"expected a 2-D decoder, got {tuple(decoder.shape)}")
    if not (0 <= int(feature) < decoder.shape[0]):
        raise ValueError(
            f"feature {feature} is outside this SAE's dictionary "
            f"(0..{decoder.shape[0] - 1}) for {release}/{sae_id}")
    row = [float(x) for x in decoder[int(feature)].tolist()]

    config = dict(cfg) if isinstance(cfg, dict) else {}
    feature_sparsity = None
    if sparsity is not None:
        try:
            feature_sparsity = float(sparsity.detach().float().cpu()[int(feature)])
        except Exception:  # noqa: BLE001 - a missing statistic is not an error
            feature_sparsity = None
    return LoadedSAEFeature(decoder_row=row, repo_id=source.repo_id,
                            repo_revision=source.revision, config=config,
                            sparsity=feature_sparsity)


def _resolve_repo_revision(repo_id: str) -> str:  # pragma: no cover - network
    try:
        from huggingface_hub import HfApi
        sha = HfApi().repo_info(repo_id).sha
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"could not resolve the exact commit of '{repo_id}' ({exc}) — the "
            "import pins the repository revision, so a metadata call (or a "
            "loader that supplies the sha) is required") from exc
    if not sha:
        raise ValueError(f"'{repo_id}' returned no commit sha to pin")
    return str(sha)


def import_feature_by_id(
        *, model_id: str, release: str, sae_id: str, feature: int, label: str,
        residual_norm_artifact: str, run_directory: str,
        layer: int | None = None, neuronpedia_url: str | None = None,
        name: str | None = None,
        loader: SAEFeatureLoader | None = None) -> str:
    """Import one SAE feature BY ID as a steering-vector artifact.

    Unlike :func:`import_feature` this needs no cosine report: features chosen
    on semantic grounds (an auto-interp label, a Neuronpedia search, a prior
    paper) have no cosine relationship to any CAA direction, and inventing one
    to satisfy the importer would fabricate a pairing the study does not claim.

    ``residual_norm_artifact`` is a **calibration donor only**. It supplies the
    residual-stream norm table, its source label, and the model identity +
    revision the calibration was measured under. It is NOT a reference vector,
    and no semantic relationship to the feature is implied — which is why the
    only thing checked about it is that its calibration is applicable (same
    model, same depth, same hidden size).

    Refuses (never clobbers, never guesses): a donor with no residual norms at
    the SAE layer, a donor for a different model, a decoder row whose dimension
    is not the model's hidden size, a layer outside the model's depth, and an
    artifact that already exists at the destination.
    """
    from ..build_identity import engine_version
    from ..steering.vector_store import ConceptVectors, SteeringVectorSidecar, save
    from . import paths

    feature = coerce_layer(feature, label="feature")
    label = (label or "").strip()
    if not label:
        raise ValueError(
            "a construct label is required — it is the researcher's name for "
            "what the feature is being used as, and it becomes the artifact's "
            "concept string")
    if ":" in label:
        raise ValueError(
            f"construct label {label!r} may not contain ':' — the concept "
            "string is 'sae:<label>:L<layer>:F<id>' and a colon makes it "
            "ambiguous to parse")
    if not (release or "").strip() or not (sae_id or "").strip():
        raise ValueError("release and saeID are required")
    model_id = (model_id or "").strip()
    if not model_id:
        raise ValueError("model id is required")

    # --- the calibration donor ------------------------------------------
    reference = (residual_norm_artifact or "").strip()
    if not reference:
        raise ValueError(
            "residualNormArtifact is required: the import denominates the "
            "stored row in residual-stream norm units, and that denominator "
            "is measured, never assumed")
    resolved = paths.resolve_artifact(reference)
    donor_dir, donor_name = os.path.split(resolved)
    if not donor_name or not os.path.isfile(os.path.join(donor_dir, donor_name + ".json")):
        raise ValueError(
            f"residualNormArtifact {reference!r} names no vector artifact "
            f"(expected {resolved}.json + .safetensors — the base path, no "
            f"extension)")
    _donor_vectors, donor = vector_store.load(donor_dir, donor_name)

    donor_model = (donor.modelID or "").strip()
    if not donor_model:
        raise ValueError(
            f"residualNormArtifact {reference!r} carries no model identity — "
            "a residual-norm table is a per-model measurement and cannot be "
            "applied to an unnamed model")
    if donor_model.casefold() != model_id.casefold():
        raise ValueError(
            f"residualNormArtifact {reference!r} was measured on "
            f"{donor_model!r} but the import names {model_id!r} — residual-norm "
            "calibration does not transfer between models; use a donor from "
            "this model")

    # --- the decoder row -------------------------------------------------
    loaded = (loader or load_sae_feature)(release, sae_id, feature)
    if not loaded.repo_id or not loaded.repo_revision:
        raise ValueError(
            "the SAE loader returned no repository id/revision — the import "
            "pins the exact published bytes it read, so unpinned sources are "
            "refused")
    row = [float(x) for x in loaded.decoder_row]
    if not row:
        raise ValueError(f"feature {feature} has an empty decoder row")

    parsed = parse_sae_id(sae_id)
    config_layer = _config_layer(loaded.config or {})
    derived_layer = config_layer if config_layer is not None else parsed["layer"]
    if layer is not None and derived_layer is not None and int(layer) != derived_layer:
        raise ValueError(
            f"requested layer {int(layer)} disagrees with the SAE's own layer "
            f"{derived_layer} ({sae_id}) — an SAE's dictionary lives at exactly "
            "one layer, so this is a mis-specified import, not a choice")
    sae_layer = int(layer) if layer is not None else derived_layer
    if sae_layer is None:
        raise ValueError(
            f"cannot determine the layer for SAE {sae_id!r}: its config names "
            "no hook layer and the id does not follow the 'layer_<n>_…' "
            "grammar — pass the layer explicitly")

    if len(row) != donor.hiddenSize:
        raise ValueError(
            f"feature {feature} decoder has dimension {len(row)}, expected "
            f"{donor.hiddenSize} (the model's hidden size, per the calibration "
            f"donor) — wrong SAE for this model")
    if not (0 <= sae_layer < donor.layerCount):
        raise ValueError(
            f"SAE layer {sae_layer} is outside the model's depth "
            f"({donor.layerCount} layers, per the calibration donor)")

    residual_norms = donor.residualNormPerLayer
    if not residual_norms or sae_layer >= len(residual_norms):
        raise ValueError(
            f"residualNormArtifact {reference!r} has no residualNormPerLayer at "
            f"layer {sae_layer} — measure the residual norms on the pinned "
            "neutral corpus first (vectors backfill-norms)")
    target_norm = float(residual_norms[sae_layer])

    scaled, raw_norm = _convention_rescale(row, target_norm)
    rescale: dict = {"convention": RESIDUAL_NORM_MATCH_CONVENTION,
                     "rawDecoderNorm": raw_norm, "targetNorm": target_norm,
                     "applied": True}
    if raw_norm <= 0 or target_norm <= 0:
        # Degenerate guard, identical in spirit to the report path's: the row
        # is stored RAW and says so, rather than being silently scaled by a
        # meaningless factor.
        rescale["applied"] = False
        rescale["skippedReason"] = (
            "zero-norm decoder row" if raw_norm <= 0 else
            f"non-positive residual norm at layer {sae_layer}")

    # --- write (immutably) -----------------------------------------------
    artifact_name = name or f"sae-feature-{feature}"
    if ("/" in artifact_name or os.sep in artifact_name
            or artifact_name in (".", "..")):
        # Containment, same rule as the backfill verb's --output-name: the
        # artifact name is a file-name component, never a path.
        raise ValueError(
            f"artifact name {artifact_name!r} must be a plain file-name "
            "component")
    for extension in (".json", ".safetensors"):
        existing = os.path.join(run_directory, artifact_name + extension)
        if os.path.exists(existing):
            raise ValueError(
                f"refusing to overwrite existing artifact {existing} — vector "
                "artifacts are immutable; import into a fresh run directory")

    per_layer = [[0.0] * donor.hiddenSize for _ in range(donor.layerCount)]
    per_layer[sae_layer] = scaled
    vectors = ConceptVectors(per_layer=per_layer)

    source: dict = {
        "importPath": "direct-feature-id",
        "release": release, "saeID": sae_id, "feature": feature,
        "repository": loaded.repo_id, "repositoryRevision": loaded.repo_revision,
        "layer": sae_layer,
        "saeConfigHash": sae_config_hash(loaded.config or {}),
        "decoderRowHash": decoder_row_hash(row),
        "rescale": rescale,
        "constructLabel": label,
        # Calibration provenance: what supplied the denominator, and what that
        # denominator itself was measured on.
        "residualNormArtifact": reference,
        "residualNormSource": donor.residualNormSource,
        "importedBy": engine_version(),
        "importedAt": _now_iso8601(),
        "substrate": vector_store.SUBSTRATE,
    }
    if parsed["width"]:
        source["width"] = parsed["width"]
    if isinstance((loaded.config or {}).get("d_sae"), int):
        source["dSAE"] = int(loaded.config["d_sae"])
    if parsed["l0Target"]:
        source["l0Target"] = parsed["l0Target"]
    site = _config_site(loaded.config or {})
    if site:
        source["site"] = site
    if loaded.sparsity is not None:
        source["featureSparsity"] = float(loaded.sparsity)
    if neuronpedia_url:
        # DISCOVERY provenance, deliberately labelled as such: Neuronpedia is
        # where a candidate was found, never a runtime dependency for evidence
        # (auto-interp labels are regenerated; the URL is a pointer, not data).
        source["discovery"] = {"neuronpediaURL": str(neuronpedia_url)}

    sidecar = SteeringVectorSidecar(
        modelID=donor.modelID,
        concept=f"sae:{label}:L{sae_layer}:F{feature}",
        stimulusSetHash=f"gemmascope:{release}:{sae_id}:{feature}",
        layerCount=vectors.layer_count, hiddenSize=vectors.hidden_size,
        normsPerLayer=[vectors.norm(i) for i in range(vectors.layer_count)],
        extractionDate=_now_iso8601(),
        revision=donor.revision,
        extractionMethod="gemmaScopeSAE",
        coversModelDepth=_inherited_depth_coverage(donor),
        residualNormPerLayer=[float(n) for n in residual_norms[:donor.layerCount]],
        residualNormSource=donor.residualNormSource,
        # The calibration identity travels as one unit: a neutral-corpus norm
        # source without the corpus hash cannot stamp recipeIdentityHash
        # downstream (sweep refuses at start — "writer bug", and it was).
        neutralCorpusHash=donor.neutralCorpusHash,
        recipeHash=f"{release}|{sae_id}|feature:{feature}",
        recipeName=(f"Gemma Scope SAE feature {feature} from {release}/{sae_id}, "
                    f"scaled to the residual-stream norm at layer {sae_layer}"),
        gemmascopeConvention=RESIDUAL_NORM_MATCH_CONVENTION,
        rawDecoderNorm=raw_norm,
        gemmascopeTargetNorm=target_norm,
        gemmascopeSource=source)
    save(vectors, sidecar, run_directory, artifact_name)
    return os.path.join(run_directory, artifact_name)


# --------------------------------------------------------------------------
# TRUE LATENT intervention: the four tensors of ONE feature (proposal r2 §8 P2-9)
# --------------------------------------------------------------------------
#
# A THIRD path, and the one that is not a steering vector at all. The two
# import paths above both produce a decoder DIRECTION stored as a vector
# artifact, to be added to the residual stream. A latent intervention instead
# runs the residual stream through the SAE encoder, edits this feature's latent
# activation, and decodes only the induced delta
# (:mod:`steerlab_server.steering.sae_latent`). That needs the ENCODER side as
# well, so it needs its own loader — and it deliberately writes NO vector
# artifact, because storing one would invite exactly the conflation the
# proposal forbids.


@dataclass
class LoadedSAELatentFeature:
    """One feature's encoder column, decoder row, bias and JumpReLU threshold,
    plus the provenance needed to pin where they came from.

    Produced by an *injectable* loader (:data:`SAELatentFeatureLoader`) — the
    same seam discipline as :data:`SAEFeatureLoader`, so every validation and
    stamping test runs with no network, no HF cache and no ``sae_lens``.
    """

    #: ``W_enc[:, f]`` — the encoder direction (length = model hidden size).
    encoder_row: list[float]
    #: ``W_dec[f]`` — the decoder direction (length = model hidden size).
    decoder_row: list[float]
    #: ``b_enc[f]``, with the ``b_dec``-applied-to-input correction FOLDED IN
    #: when the SAE uses one (see :attr:`b_dec_folded`).
    encoder_bias: float
    #: ``threshold[f]`` — the per-feature JumpReLU threshold.
    threshold: float
    #: The HF repository the SAE was read from.
    repo_id: str
    #: The EXACT commit sha resolved at load time. A floating ref is not
    #: provenance: the repository can be re-tagged under it.
    repo_revision: str
    #: The SAE's own config as published, verbatim (hashed into provenance).
    config: dict = field(default_factory=dict)
    #: This feature's published sparsity/L0 statistic, when the release has one.
    sparsity: float | None = None
    #: Whether ``−b_dec · W_enc[:, f]`` was folded into :attr:`encoder_bias`.
    #: Recorded, never inferred: it shifts every pre-activation this feature
    #: will ever produce.
    b_dec_folded: bool = False
    #: The activation family the threshold came from (``"jumprelu"`` or
    #: ``"relu"``), so a reader never has to work out which gate ran.
    activation: str = "jumprelu"


#: ``(release, sae_id, feature) -> LoadedSAELatentFeature``.
SAELatentFeatureLoader = Callable[[str, str, int], LoadedSAELatentFeature]


def _sae_dimensions(sae, config: dict) -> tuple[int, int]:
    """``(d_in, d_sae)`` for a loaded SAE, VERIFIED rather than assumed.

    Orientation is the one thing here that must not be guessed: transposing
    ``W_enc`` silently substitutes a decoder row for an encoder column, and
    every downstream number would still look plausible. The BIASES pin it
    without ambiguity — ``b_enc`` has length ``d_sae``, ``b_dec`` has length
    ``d_in`` — and the published config's ``d_in``/``d_sae`` are cross-checked
    against them. Only when neither source exists do we fall back to sae_lens's
    documented ``W_dec: [d_sae, d_in]`` convention, and even then the
    degenerate ``d_in == d_sae`` case refuses rather than picking.
    """
    def _length(name: str) -> int | None:
        tensor = getattr(sae, name, None)
        if tensor is None:
            return None
        try:
            return int(tensor.detach().shape[-1])
        except Exception:  # noqa: BLE001
            return None

    d_in = _length("b_dec")
    d_sae = _length("b_enc")
    cfg_in = config.get("d_in") if isinstance(config.get("d_in"), int) else None
    cfg_sae = config.get("d_sae") if isinstance(config.get("d_sae"), int) else None
    if d_in is not None and cfg_in is not None and d_in != cfg_in:
        raise RuntimeError(
            f"SAE config says d_in={cfg_in} but b_dec has length {d_in} — "
            "refusing to guess the orientation of an inconsistent SAE")
    if d_sae is not None and cfg_sae is not None and d_sae != cfg_sae:
        raise RuntimeError(
            f"SAE config says d_sae={cfg_sae} but b_enc has length {d_sae} — "
            "refusing to guess the orientation of an inconsistent SAE")
    d_in = d_in if d_in is not None else cfg_in
    d_sae = d_sae if d_sae is not None else cfg_sae
    if d_in is not None and d_sae is not None:
        return int(d_in), int(d_sae)

    decoder = getattr(sae, "W_dec", None)
    if decoder is None or getattr(decoder, "ndim", 0) != 2:
        raise RuntimeError(
            "cannot determine this SAE's dimensions: no b_enc/b_dec, no "
            "d_in/d_sae in the config, and no 2-D W_dec")
    rows, cols = (int(x) for x in decoder.shape)
    if rows == cols:
        raise RuntimeError(
            f"SAE has a square W_dec ({rows}x{cols}) and publishes neither "
            "biases nor d_in/d_sae — the encoder/decoder orientation is "
            "genuinely ambiguous; pass a loader that knows it")
    # sae_lens convention: W_dec is [d_sae, d_in], and a dictionary expands.
    return (cols, rows) if rows > cols else (rows, cols)


def _oriented_row(tensor, *, feature: int, d_in: int, d_sae: int,
                  which: str) -> list[float]:
    """The feature's row/column of a 2-D SAE weight, whichever way it is stored.

    ``which`` is ``"encoder"`` (want ``W_enc[:, f]``, canonically stored
    ``[d_in, d_sae]``) or ``"decoder"`` (want ``W_dec[f]``, canonically stored
    ``[d_sae, d_in]``). Both accept the transposed storage and refuse anything
    else; the returned row is always length ``d_in``.
    """
    if tensor is None:
        raise RuntimeError(f"loaded SAE does not expose its {which} weight")
    matrix = tensor.detach().float().cpu()
    if matrix.ndim != 2:
        raise RuntimeError(
            f"expected a 2-D {which} weight, got {tuple(matrix.shape)}")
    shape = tuple(int(x) for x in matrix.shape)
    if which == "encoder":
        if shape == (d_in, d_sae):
            row = matrix[:, feature]
        elif shape == (d_sae, d_in):
            row = matrix[feature, :]
        else:
            raise RuntimeError(
                f"encoder weight {shape} matches neither [d_in, d_sae] = "
                f"[{d_in}, {d_sae}] nor its transpose")
    else:
        if shape == (d_sae, d_in):
            row = matrix[feature, :]
        elif shape == (d_in, d_sae):
            row = matrix[:, feature]
        else:
            raise RuntimeError(
                f"decoder weight {shape} matches neither [d_sae, d_in] = "
                f"[{d_sae}, {d_in}] nor its transpose")
    return [float(x) for x in row.tolist()]


def load_sae_latent_feature(release: str, sae_id: str, feature: int, *,
                            resolver: RepoRevisionResolver | None = None,
                            builder: PinnedSAELoader | None = None
                            ) -> LoadedSAELatentFeature:
    """Default (online) loader for a latent intervention: the four tensors of
    ONE feature, read at a commit pinned BEFORE the fetch.

    The one function in this path that touches the network / HF cache, and the
    seam every test substitutes. Refuses rather than guessing: an unresolvable
    commit, an ambiguous encoder orientation, or an activation family whose
    single-feature rule is neither JumpReLU nor ReLU all raise — an
    intervention that silently used the wrong gate would produce numbers no
    reader could catch.
    """
    source, sae, cfg, sparsity = _pinned_sae(release, sae_id,
                                             resolver=resolver,
                                             builder=builder)
    config = dict(cfg) if isinstance(cfg, dict) else {}
    d_in, d_sae = _sae_dimensions(sae, config)
    feature = coerce_layer(feature, label="feature")
    if not (0 <= feature < d_sae):
        raise ValueError(
            f"feature {feature} is outside this SAE's dictionary "
            f"(0..{d_sae - 1}) for {release}/{sae_id}")

    encoder_row = _oriented_row(getattr(sae, "W_enc", None), feature=feature,
                                d_in=d_in, d_sae=d_sae, which="encoder")
    decoder_row = _oriented_row(getattr(sae, "W_dec", None), feature=feature,
                                d_in=d_in, d_sae=d_sae, which="decoder")

    b_enc = getattr(sae, "b_enc", None)
    bias = float(b_enc.detach().float().cpu()[feature]) if b_enc is not None else 0.0

    # b_dec applied to the INPUT folds into the bias exactly:
    #   pre = (h − b_dec)·w + b_enc = h·w + (b_enc − b_dec·w)
    # so the intervention never carries a second hidden-size vector and the
    # arithmetic stays a dot product plus a scalar.
    b_dec_folded = False
    if bool(config.get("apply_b_dec_to_input", False)):
        b_dec = getattr(sae, "b_dec", None)
        if b_dec is not None:
            values = [float(x) for x in b_dec.detach().float().cpu().tolist()]
            bias -= sum(v * w for v, w in zip(values, encoder_row))
            b_dec_folded = True

    threshold_tensor = getattr(sae, "threshold", None)
    activation = str(config.get("activation_fn_str")
                     or config.get("activation_fn") or "").lower()
    if threshold_tensor is not None:
        threshold = float(threshold_tensor.detach().float().cpu()[feature])
        family = "jumprelu"
    elif activation in ("relu", ""):
        # A plain-ReLU SAE gates at zero. Stamped explicitly so a reader never
        # has to work out whether a zero threshold was measured or assumed.
        threshold, family = 0.0, "relu"
    else:
        raise RuntimeError(
            f"SAE {release}/{sae_id} uses activation {activation!r} and "
            "publishes no per-feature threshold — a single-feature latent edit "
            "needs the exact activation rule, and this one is neither JumpReLU "
            "nor ReLU")

    feature_sparsity = None
    if sparsity is not None:
        try:
            feature_sparsity = float(sparsity.detach().float().cpu()[feature])
        except Exception:  # noqa: BLE001 - a missing statistic is not an error
            feature_sparsity = None
    return LoadedSAELatentFeature(
        encoder_row=encoder_row, decoder_row=decoder_row, encoder_bias=bias,
        threshold=threshold, repo_id=source.repo_id,
        repo_revision=source.revision,
        config=config, sparsity=feature_sparsity, b_dec_folded=b_dec_folded,
        activation=family)
