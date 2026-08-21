"""Residual-norm BACKFILL for vector artifacts that predate norm-unit alphas.

Norm-unit alphas (the default steering denomination) need
``residualNormPerLayer`` in the vector sidecar, measured on a pinned neutral
corpus. Three artifact classes lack it: legacy pre-capture vectors, Gemma Scope
SAE imports, and RepE reader-derived steering vectors. Backfill measures the
per-layer typical residual norms on a neutral corpus and writes a **new**
artifact (runs are immutable — the original is never mutated).

The denominator convention is load-bearing: this module calls the *same*
measurement path extraction uses for its neutral corpus —
:func:`steerlab_server.steering.extractor.activations` — at the reading
position stamped in the artifact's own sidecar, and averages exactly the way
``extract()`` does, so a backfilled α means the same thing as an
extraction-time α.

Backfill is also THE opt-in migration onto the current residual-norm
DENOMINATOR CONVENTION (:mod:`residual_norm_convention`). Existing sidecars
are never rewritten in place and never recomputed: an artifact with no
``residualNormConvention`` stamp is LEGACY and keeps meaning exactly what it
always meant. Running this verb re-measures the denominator under the current
convention and stamps it, which is how a researcher chooses to move a specific
artifact forward.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone

from . import extractor, stimulus_set, vector_store
from .reading_position import from_label
from .residual_norm_convention import CURRENT as RESIDUAL_NORM_CONVENTION

# Must match the string ``extractor.extract`` / ``extract_grand_mean`` write
# when the norms come from a neutral corpus (extractor.py: ``norm_source =
# "neutral-corpus"``) — the sidecar convention the catalog and Swift decode.
RESIDUAL_NORM_SOURCE_NEUTRAL_CORPUS = "neutral-corpus"

# Extraction only accepts a neutral corpus as the norm denominator when it has
# at least this many texts (``extract(...)``'s ``len(neutral_texts) >= 4``
# gate); a backfilled denominator must clear the same bar.
MINIMUM_CORPUS_TEXTS = 4


@dataclass
class BackfillResult:
    run_directory: str
    artifact_id: str                # "<run_directory>/<name>" (catalog id)
    vectors_path: str
    sidecar_path: str
    residual_norm_per_layer: list[float]
    residual_norm_source: str
    residual_norm_convention: str
    layer_count: int
    neutral_corpus_hash: str


def backfill_norms(model, vector_dir: str, name: str, corpus_path: str,
                   run_directory: str, *, output_name: str | None = None,
                   redenominate: bool = False) -> BackfillResult:
    """Measure per-layer residual norms for an existing vector artifact on a
    pinned neutral corpus and write a NEW artifact into ``run_directory``.

    The vectors bytes are copied verbatim; the sidecar is copied with only
    ``residualNormPerLayer`` / ``residualNormSource`` /
    ``residualNormConvention`` / ``neutralCorpusHash`` updated plus a
    ``normBackfill`` provenance stamp — every other field
    (extraction method, stimulus hashes, reader provenance, …) is preserved
    byte-for-byte, including fields this engine does not know about.

    ``redenominate`` (loud, never silent — the same override pattern as
    ``freeze --force``): re-measures an artifact whose EXISTING norms were
    taken on a non-neutral distribution (``extraction-stimuli``, the legacy
    concept-dependent denominator, under which "α = 1 norm units" does not
    mean what the methods policy promises). Refused when the artifact is
    already neutral-corpus denominated; the provenance stamp records the
    replaced source. The original artifact is never mutated either way.
    """
    vectors, sidecar = vector_store.load(vector_dir, name)

    replaced_norm_source: str | None = None
    if sidecar.residualNormPerLayer:
        source = sidecar.residualNormSource or "unrecorded"
        if not redenominate:
            raise ValueError(
                f"artifact already has residual norms (source: {source}) — "
                "backfill never overwrites; pass redenominate to write a NEW "
                "artifact denominated on the neutral corpus")
        if source.startswith("neutral-corpus"):
            raise ValueError(
                f"artifact is already neutral-corpus denominated ({source}) — "
                "nothing to redenominate")
        replaced_norm_source = source

    # HARD guard: a norm table is a per-model measurement (same contract as
    # the reader model guard).
    if sidecar.modelID != model.model_id:
        raise ValueError(
            f"vectors were extracted on model {sidecar.modelID!r}; the loaded model is "
            f"{model.model_id!r} — a residual-norm table is a per-model measurement, "
            "backfill it on the artifact's own model")

    corpus = stimulus_set.load_texts(corpus_path)
    if len(corpus.texts) < MINIMUM_CORPUS_TEXTS:
        raise ValueError(
            f"neutral corpus has {len(corpus.texts)} texts; extraction requires at "
            f"least {MINIMUM_CORPUS_TEXTS} for a norm denominator")

    # Same reading-position semantics as extraction: the position stamped in
    # the sidecar, resolved by the same ``from_label`` helper routes use.
    position = from_label(sidecar.readingPosition or "")
    measured = extractor.activations(model, corpus.texts, position)
    # A non-finite norm ANYWHERE in the measurement means the forward pass
    # overflowed (fp16 activation blow-up, the Gemma-3 failure mode) — the
    # whole measurement is untrustworthy as a denominator, even at layers
    # before the overflow. Same rule as the Swift twin's alignedNorms.
    if not all(math.isfinite(n) for n in measured.residual_norm_per_layer):
        raise ValueError(
            "measured residual norms contain non-finite values — activation "
            "overflow; re-measure with a float32-capable dtype (Gemma 3 needs "
            "float32 on MPS)")

    # Equality-or-prefix alignment (mirrors the Swift twin's alignedNorms):
    # reader-/SAE-derived artifacts deliberately carry vectors only up to
    # their injection layer (zeros-below convention), so the measurement may
    # legitimately cover MORE layers than the artifact — take the block-index
    # prefix. Fewer measured layers than vectors means the wrong model.
    norms = measured.residual_norm_per_layer
    if len(norms) < vectors.layer_count:
        raise ValueError(
            f"measured {len(norms)} per-layer norms but the artifact has "
            f"{vectors.layer_count} layers — wrong model family for this artifact")
    norms = norms[: vectors.layer_count]

    # Copy the artifact as raw bytes/raw JSON, not through the dataclass, so
    # sidecar fields this engine does not model survive untouched.
    source_vectors_path = os.path.join(vector_dir, f"{name}.safetensors")
    source_sidecar_path = os.path.join(vector_dir, f"{name}.json")
    with open(source_vectors_path, "rb") as handle:
        vector_bytes = handle.read()
    with open(source_sidecar_path, encoding="utf-8") as handle:
        sidecar_dict = json.load(handle)

    sidecar_dict["residualNormPerLayer"] = list(norms)
    sidecar_dict["residualNormSource"] = RESIDUAL_NORM_SOURCE_NEUTRAL_CORPUS
    # Backfill IS the opt-in migration to the current denominator convention:
    # these norms were measured just now, by this code, under
    # ``residual_norm_convention.CURRENT``. Legacy artifacts are never stamped
    # in place — running backfill is how a researcher chooses to move one onto
    # the stamped convention.
    sidecar_dict["residualNormConvention"] = RESIDUAL_NORM_CONVENTION
    sidecar_dict["neutralCorpusHash"] = corpus.hash
    # Pinned provenance contract (Swift decodes this exact shape;
    # replacedNormSource is optional and present only on redenomination).
    sidecar_dict["normBackfill"] = {
        "sourceArtifact": os.path.join(vector_dir, name),
        "sourceVectorsHash": hashlib.sha256(vector_bytes).hexdigest(),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **({"replacedNormSource": replaced_norm_source}
           if replaced_norm_source else {}),
    }

    out_name = output_name or name
    os.makedirs(run_directory, exist_ok=True)
    vectors_path = os.path.join(run_directory, f"{out_name}.safetensors")
    with open(vectors_path, "wb") as handle:
        handle.write(vector_bytes)
    sidecar_path = os.path.join(run_directory, f"{out_name}.json")
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(sidecar_dict, handle, sort_keys=True, indent=2)

    return BackfillResult(
        run_directory=run_directory,
        artifact_id=os.path.join(run_directory, out_name),
        vectors_path=vectors_path, sidecar_path=sidecar_path,
        residual_norm_per_layer=list(norms),
        residual_norm_source=RESIDUAL_NORM_SOURCE_NEUTRAL_CORPUS,
        residual_norm_convention=RESIDUAL_NORM_CONVENTION,
        layer_count=vectors.layer_count,
        neutral_corpus_hash=corpus.hash)
