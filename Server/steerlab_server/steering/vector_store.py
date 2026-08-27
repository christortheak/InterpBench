"""Persist concept vectors as ``<name>.safetensors`` + ``<name>.json`` sidecar
(parallel to Swift ``SteeringVectorStore`` + ``SteeringVectorSidecar``).

The on-disk format is the cross-engine contract:
- ``<name>.safetensors`` holds one float32 tensor per layer, keyed
  ``layer_0 … layer_N`` (exactly what Swift ``MLX.save`` writes).
- ``<name>.json`` is the provenance sidecar (``schemaVersion: 2``). The Swift
  loader only JSON-*decodes* the sidecar, so **schema** compatibility — not
  byte-identical formatting — is what's required. We write sorted, indented
  JSON; Swift reads it back fine. Hashes that gate the freeze firewall are over
  the *stimulus* files, not this sidecar, so formatting drift here is harmless.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from safetensors.numpy import load_file, save_file

from . import vector_math
from .reading_position import ReadingPosition

# The single definition of this engine's substrate identity, stamped into every
# artifact this engine writes (vector sidecars here; reader artifacts and
# validation evidence re-export/import it). The Swift engine stamps
# ``"swift-mlx"``. An absent stamp means legacy/unknown — never guessed.
SUBSTRATE = "python-hf-transformers"


@dataclass
class ConceptVectors:
    """Per-layer steering directions (parallel to Swift ``ConceptVectors``)."""

    per_layer: list[list[float]]

    @property
    def layer_count(self) -> int:
        return len(self.per_layer)

    @property
    def hidden_size(self) -> int:
        return len(self.per_layer[0]) if self.per_layer else 0

    def norm(self, layer: int) -> float:
        v = self.per_layer[layer]
        return float(math.sqrt(sum(x * x for x in v)))


@dataclass
class SteeringVectorSidecar:
    """Provenance sidecar. Field names match the Swift Codable JSON keys."""

    modelID: str
    concept: str
    stimulusSetHash: str
    layerCount: int
    hiddenSize: int
    normsPerLayer: list[float]
    extractionDate: str
    schemaVersion: int = 2
    revision: str | None = None
    # Which engine extracted these vectors (:data:`SUBSTRATE` here,
    # ``"swift-mlx"`` for the Mac app). Activations do not transfer across
    # engines, so injection paths refuse foreign-substrate artifacts; ``None``
    # is a legacy/unknown artifact (pre-stamp), which loads with no way to know.
    substrate: str | None = None
    # Whether ``layerCount`` above is the MODEL's depth or only this artifact's
    # own row count. Reader-derived directions are PARTIAL by construction
    # (zeros below the reader's layer, then one row), so a workspace that reads
    # a model's depth off one of them gets the reader's layer plus one — and
    # then converts absolute sweep layers against a network that does not
    # exist. Stamped ``false`` forward by every writer that knows its artifact
    # is partial. **Absent is read by method**: a reader-derived artifact
    # written before the stamp is partial anyway (the whole family is), and
    # everything else — CAA, LAT, grand-mean, designated reference, OptVec,
    # J-lens, and the artifacts old enough to carry no ``extractionMethod`` at
    # all — is full depth. The rule lives in one place,
    # ``catalog.covers_model_depth``. Pinned cross-engine contract: same JSON
    # key on the Swift ``SteeringVectorSidecar``.
    coversModelDepth: bool | None = None
    extractionMethod: str | None = None
    readingPosition: str | None = None
    confoundProjection: str | None = None
    residualNormPerLayer: list[float] | None = None
    residualNormSource: str | None = None
    # WHICH AVERAGING RULE produced ``residualNormPerLayer`` — the denominator
    # convention (``residual_norm_convention.CURRENT``,
    # ``"wholeCorpusMean-v1"``). ``residualNormSource`` says which corpus was
    # measured; this says how its positions were averaged, and the two
    # together are what make an alpha in norm units mean one dose.
    #
    # Pinned cross-engine contract: same JSON key and same value string on the
    # Swift ``SteeringVectorSidecar``. Stamped only by a FRESH measurement
    # (extraction, or ``vectors backfill-norms``) and propagated verbatim when
    # an artifact's norms are copied. **Absent is LEGACY and is never
    # retro-filled** — a pre-stamp artifact's rule genuinely depends on which
    # engine wrote it and whether its corpus was downsampled, so guessing
    # would be worse than the honest gap. Legacy artifacts read exactly as
    # they always have; ``backfill-norms`` is the opt-in path to a stamped
    # denominator.
    residualNormConvention: str | None = None
    # WHICH RENDERING the denominator corpus was tokenized under — the third
    # member of the residualNorm* provenance family (source = which corpus,
    # convention = how its positions were averaged, rendering = how its texts
    # reached the model). α in norm units only means one dose if the
    # denominator was measured on the same distribution the vector was read
    # from, so the denominator always follows the extraction's rendering and
    # the artifact says which one that was. Values: "raw" | "chatTemplate".
    # **Absent is LEGACY RAW** — every pre-2026-08-24 artifact was raw, and no
    # artifact is retro-stamped.
    residualNormRendering: str | None = None
    # HOW the stimulus strings reached the model during extraction — the
    # declared rendering block ({"mode": "raw"} or {"mode": "chatTemplate",
    # "addGenerationPrompt": …, "qwenThinkingEnabled": …, optional
    # "systemPrompt"}). Pinned cross-engine contract: same JSON key and inner
    # key names on the Swift ``SteeringVectorSidecar``. Additive and
    # **absent = legacy raw**, never retro-filled: the same discipline
    # residualNormConvention follows.
    extractionRendering: dict | None = None
    # The requested reading position AND what it RESOLVED to, per sequence
    # shape — {"requested", "mode", "parameter"?, "rendering", "source",
    # "shapes": [{"offsetFromEnd", "sequenceCount", "exampleIndex",
    # "exampleEndIndex", "exampleTokenCount", and for a CONTENT-MASKED pool
    # only, "examplePooledTokenCount"/"exampleMaskedTokenCount"}]}. Those two
    # are additive and absent everywhere else, so no other position's stamped
    # bytes moved when they landed. `readingPosition` says what
    # was ASKED for; this says where that landed, so a reader never has to
    # re-derive a template's internals to know what was read. Stamped only
    # when the label does not already imply the index (a template-aware role,
    # an explicit offset, or any non-raw rendering) — so legacy artifacts keep
    # byte-identical sidecars. Additive cross-engine contract.
    readingPositionResolution: dict | None = None
    recipeMethod: str | None = None
    recipeHash: str | None = None
    recipeName: str | None = None
    neutralProjection: str | None = None
    neutralCorpusHash: str | None = None
    readingMinimumTokenCount: int | None = None
    sourceStimulusCount: int | None = None
    includedStimulusCount: int | None = None
    excludedShortStimulusCount: int | None = None
    neutralSourceStimulusCount: int | None = None
    neutralIncludedStimulusCount: int | None = None
    neutralExcludedShortStimulusCount: int | None = None
    comparisonConcepts: list[str] | None = None
    selectedTopics: list[str] | None = None
    selectedSplits: list[str] | None = None
    # Derived-artifact provenance (e.g. a vector converted from a RepE reader:
    # source "repe-reader-lat", the reader file id + SHA-256, and the honest
    # control-mode label "reading-vector activation addition").
    source: str | None = None
    readerID: str | None = None
    readerHash: str | None = None
    controlMode: str | None = None
    # The reader's layer, template pin, contrast construction and sign rule,
    # copied onto the derived vector so an attached reader-derived concept can
    # be verified against the instrument it came from without re-opening the
    # reader file (which may not travel with a bundle). Absent on every
    # non-reader artifact and on reader-derived vectors written before
    # 2026-08-27. Pinned cross-engine contract: same keys on the Swift
    # ``SteeringVectorSidecar``.
    readerLayer: int | None = None
    readerTemplateID: str | None = None
    readerTemplateHash: str | None = None
    readerContrastMode: str | None = None
    readerSignConvention: str | None = None
    # The reader probe's ``orientation`` at derive time (+1 or −1) — the sign
    # the TRAIN class means imply. Under ``trainMajority`` the derived BYTES
    # have it folded in (a reader with orientation −1 stores a direction
    # pointing away from the concept, and the conversion negates it); under
    # ``heldOutPairAgreement`` the fitted direction already carries the
    # held-out-chosen sign and ships unflipped. Either way this stamp is what
    # makes the conversion recoverable from the artifact alone.
    readerProbeOrientation: float | None = None
    # Whether the TRAIN split would have signed a held-out-signed direction the
    # other way (``readerProbeOrientation == −1`` under
    # ``signConvention: "heldOutPairAgreement"``). Diagnostic, not corrective:
    # the held-out sign stands, and this records that the two splits did not
    # agree about it, which is a fact about the direction's stability. Present
    # (true or false) only when held-out did the signing; absent under
    # train-majority and on every non-reader artifact. Pinned cross-engine
    # contract: same JSON key on the Swift ``SteeringVectorSidecar``.
    trainHeldOutSignDisagreement: bool | None = None
    # HOW this direction's sign was fixed — "heldOutPairAgreement" (the RepE
    # paper's step 4) or "trainMajority" (the reference implementation's
    # get_signs). Stamped by every family whose direction has a sign to choose,
    # so the difference between "held out decided this" and "the training
    # labels decided this" is visible on the artifact instead of being a fact
    # about which code path produced it. Absent = legacy train-majority.
    signConvention: str | None = None
    # Residual-norm backfill provenance (norm_backfill.backfill_norms): the
    # pinned {"sourceArtifact", "sourceVectorsHash", "date"} shape stamped on
    # a NEW artifact whose norms were measured after the fact on a neutral
    # corpus. Swift decodes this exact shape — do not rename keys.
    normBackfill: dict | None = None
    # Gemma Scope import convention (pinned cross-engine contract, WS7.2).
    # Both engines now import an SAE feature the way the Swift app always
    # has ("analyzed-vector-norm-match": decoder row rescaled at import to
    # the analyzed concept vector's L2 norm at the report layer). The stamp
    # names the convention; ``rawDecoderNorm`` is the decoder row's
    # pre-transform L2 norm and ``gemmascopeTargetNorm`` the norm it was
    # rescaled to (the transform is fully recoverable from the pair).
    # Absent on a Gemma-Scope-sourced sidecar = pre-convention import —
    # loading warns; re-import before evidence use.
    # Second convention since 2026-08-13 (proposal r2 §5): the DIRECT
    # feature-ID import stamps "residual-norm-match" — the row rescaled to
    # the residual-stream norm at the SAE's layer, taken from a calibration
    # donor. Same three keys, different transform; the stamp is the only
    # thing that tells two stored rows apart, so the two are never mixed.
    gemmascopeConvention: str | None = None
    rawDecoderNorm: float | None = None
    gemmascopeTargetNorm: float | None = None
    # Direct feature-ID imports only (gemma_scope.import_feature_by_id):
    # the complete source provenance of an imported SAE direction — Gemma
    # Scope repository + EXACT resolved commit, release/saeID/feature,
    # layer/width/L0/site, SAE-config hash, raw-decoder-row hash, the
    # rescale block, the calibration donor that supplied the residual-norm
    # denominator, optional Neuronpedia DISCOVERY pointer, and the importing
    # build identity + date. One nested block rather than a dozen top-level
    # keys, following ``normBackfill``/``pinnedFrom``. Additive and optional:
    # artifacts without it load exactly as before, and no existing hash
    # covers it. NOTE (2026-08-13): the Swift ``SteeringVectorSidecar`` has
    # no twin field yet, so a Swift-side decode→re-encode (NormBackfill)
    # would DROP this block — see the SAE worklist report.
    gemmascopeSource: dict | None = None
    # Grand-mean artifacts only: the FULL comparison population the grand
    # mean was computed over — every corpus member's concept name mapped to
    # the SHA-256 of its stories.jsonl as actually read at extraction.
    # Pinned cross-engine contract (same JSON key on the Swift
    # ``SteeringVectorSidecar``); additive — absent on paired methods and on
    # artifacts predating the stamp.
    grandMeanPopulation: dict | None = None
    # Designated-reference artifacts only: {"name", "hash"} of the reference
    # stories corpus actually subtracted. Same cross-engine additive contract
    # as grandMeanPopulation.
    designatedReference: dict | None = None
    # Set when the artifact's .safetensors additionally carries the per-layer
    # NEUTRAL RESIDUAL MEAN (keys ``neutral_mean_layer_<i>``) measured on the
    # extraction's neutral corpus (``neutralCorpusHash``) at the reading
    # position. The mean enables neutral-mean CENTERING of ablation
    # directions and the ablation mean-alignment preflight — extracted
    # concept vectors routinely share a large component with this mean, and
    # ablating an uncentered direction at λ=1 collapses generation. Value is
    # the estimator description (currently always "neutral-corpus"). Pinned
    # cross-engine additive contract (same JSON key + tensor keys on the
    # Swift ``SteeringVectorSidecar``); absent on artifacts predating the
    # stamp or extracted without a neutral corpus.
    neutralMeanSource: str | None = None
    # Artifact-pinned concepts only (manifest method "pinnedArtifact"): the
    # SOURCE artifact these bytes were materialized from —
    # {"path", "sha256TensorHash", "sha256SidecarHash", "sourceMethod",
    # "sourceConcept", "sourceConceptLabel", "sourceExtractionDate"}. The
    # copy in a run directory is an ordinary vector artifact in every other
    # way; this stamp is what makes "where did this direction come from?"
    # answerable when no stimulus recipe can re-derive it (post-hoc derived
    # directions — e.g. family-grand-mean centring). Pinned cross-engine
    # additive contract; absent on every derived-from-stimuli artifact.
    pinnedFrom: dict | None = None
    # Canonical full-recipe identity hash (see experiment/recipe_identity.py
    # for the pinned canonical form). Stamped by the experiment extraction
    # writers on both engines so promotion can match artifacts by complete
    # recipe, not a six-field subset. Additive — absent on legacy artifacts,
    # which must then prove every recipe field from their other sidecar
    # fields or be refused for promotion.
    recipeIdentityHash: str | None = None
    # Set when this artifact was produced by POLE MIRRORING (``pole_mirror``;
    # Swift twin ``PoleMirror``): the source artifact's tensors multiplied by
    # −1 at every layer — a bit-exact sign flip, never a re-encode — under a
    # NEW concept name. A CAA direction points from its negative file's pole
    # toward its positive file's pole, so its negation points at the other
    # pole, which is a different concept label; that is why the mint requires
    # one. Shape follows ``pinnedFrom``'s idiom: {"path", "sha256TensorHash",
    # "sha256SidecarHash", "concept", "date"}. Pinned cross-engine additive
    # contract (same JSON key on the Swift ``SteeringVectorSidecar``).
    negatedFrom: dict | None = None
    # Present (and always True) on a mirrored artifact, qualifying
    # ``stimulusSetHash``: the mirrored pole's stimuli are the SAME two files
    # as the source's with the positive/negative ROLES swapped. A fresh hash
    # would claim different bytes were read; the source's hash carried
    # silently would claim the same recipe. The hash travels AND this stamp
    # says what changed about its meaning. Absent on every non-mirrored
    # artifact; never False.
    polesSwappedFromSource: bool | None = None

    @classmethod
    def make(cls, *, model_id: str, concept: str, stimulus_set_hash: str,
             vectors: ConceptVectors, revision: str | None = None,
             extraction_method: str | None = None,
             reading_position: ReadingPosition | None = None,
             residual_norm_per_layer: list[float] | None = None,
             residual_norm_source: str | None = None,
             residual_norm_convention: str | None = None,
             residual_norm_rendering: str | None = None,
             extraction_rendering=None,
             reading_position_resolution: dict | None = None,
             neutral_projection: str | None = None,
             neutral_corpus_hash: str | None = None,
             source_stimulus_count: int | None = None,
             included_stimulus_count: int | None = None,
             excluded_short_stimulus_count: int | None = None,
             extraction_date: datetime | None = None) -> "SteeringVectorSidecar":
        date = (extraction_date or datetime.now(timezone.utc))
        return cls(
            modelID=model_id, concept=concept, stimulusSetHash=stimulus_set_hash,
            layerCount=vectors.layer_count, hiddenSize=vectors.hidden_size,
            normsPerLayer=[vectors.norm(i) for i in range(vectors.layer_count)],
            extractionDate=_iso8601(date), revision=revision,
            substrate=SUBSTRATE,
            extractionMethod=extraction_method,
            readingPosition=(reading_position.label if reading_position else None),
            residualNormPerLayer=residual_norm_per_layer,
            residualNormSource=residual_norm_source,
            # Never invented: a caller that measured norms passes the
            # convention it measured under; one that copies or omits norms
            # passes None, and the artifact stays honestly unstamped (legacy).
            residualNormConvention=residual_norm_convention,
            # Absent-not-null, exactly like the convention stamp: a raw
            # extraction writes NOTHING, so its sidecar bytes are identical to
            # what this engine has always written. Only a declared
            # chat-template rendering stamps.
            residualNormRendering=(
                residual_norm_rendering
                if residual_norm_rendering and residual_norm_rendering != "raw"
                else None),
            extractionRendering=_rendering_block(extraction_rendering),
            readingPositionResolution=reading_position_resolution,
            neutralProjection=neutral_projection or "none",
            neutralCorpusHash=neutral_corpus_hash,
            readingMinimumTokenCount=(reading_position.minimum_token_count
                                      if reading_position else None),
            sourceStimulusCount=source_stimulus_count,
            includedStimulusCount=included_stimulus_count,
            excludedShortStimulusCount=excluded_short_stimulus_count)

    def to_dict(self) -> dict:
        # Drop None values so the JSON matches the Swift encoder, which omits
        # absent optionals.
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "SteeringVectorSidecar":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def _rendering_block(rendering) -> dict | None:
    """The ``extractionRendering`` block to stamp, or ``None`` to omit it.

    Accepts an :class:`~steerlab_server.steering.extraction_rendering.
    ExtractionRendering` (a fresh extraction), an already-shaped dict (a
    materialized copy carrying its source's block), or ``None``. RAW stamps
    NOTHING — absent is the legacy meaning and a raw artifact's sidecar bytes
    must stay byte-identical to what this engine has always written.
    """
    if rendering is None:
        return None
    if isinstance(rendering, dict):
        return None if rendering.get("mode", "raw") == "raw" else dict(rendering)
    return None if rendering.is_raw else rendering.to_dict()


def stamp_grand_mean_provenance(
        sidecar: SteeringVectorSidecar,
        population: dict[str, str]) -> SteeringVectorSidecar:
    """Stamp the comparison population that defines a grand-mean vector.

    The vector bytes can be injected without this information, but an
    optimizer cannot re-derive their recipe. Keep the convenience concept
    list and the stronger concept-to-stories-hash identity in lockstep.
    """
    normalized = {
        str(concept): str(digest)
        for concept, digest in sorted(population.items())
        if str(concept) and str(digest)
    }
    if sidecar.concept not in normalized:
        raise ValueError(
            f"grand-mean population does not contain target {sidecar.concept!r}")
    sidecar.recipeMethod = "emotionGrandMean"
    sidecar.comparisonConcepts = list(normalized)
    sidecar.grandMeanPopulation = normalized
    return sidecar


def _iso8601(date: datetime) -> str:
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    return date.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save(vectors: ConceptVectors, sidecar: SteeringVectorSidecar,
         directory: str, name: str,
         neutral_mean_per_layer: list[list[float]] | None = None) -> tuple[str, str]:
    """Write ``<name>.safetensors`` (keys ``layer_<i>``) and ``<name>.json``.

    ``neutral_mean_per_layer``, when supplied, is stored alongside as
    ``neutral_mean_layer_<i>`` tensors and stamped ``neutralMeanSource:
    "neutral-corpus"`` — additive keys the Swift loader ignores by name, so
    old readers are unaffected."""
    os.makedirs(directory, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    for i, v in enumerate(vectors.per_layer):
        tensor = np.asarray(v, dtype=np.float32)
        if not np.isfinite(tensor).all():
            raise ValueError(f"vector layer_{i} contains non-finite values")
        tensors[f"layer_{i}"] = tensor
    if neutral_mean_per_layer is not None:
        if len(neutral_mean_per_layer) != vectors.layer_count:
            raise ValueError(
                f"neutral mean has {len(neutral_mean_per_layer)} layers, "
                f"vectors have {vectors.layer_count}")
        for i, m in enumerate(neutral_mean_per_layer):
            tensor = np.asarray(m, dtype=np.float32)
            if not np.isfinite(tensor).all():
                raise ValueError(
                    f"neutral_mean_layer_{i} contains non-finite values")
            tensors[f"neutral_mean_layer_{i}"] = tensor
        sidecar.neutralMeanSource = "neutral-corpus"
    for label, values in (
        ("normsPerLayer", sidecar.normsPerLayer),
        ("residualNormPerLayer", sidecar.residualNormPerLayer),
    ):
        if values is not None and not np.isfinite(np.asarray(values, dtype=np.float32)).all():
            raise ValueError(f"{label} contains non-finite values")
    vectors_url = os.path.join(directory, f"{name}.safetensors")
    save_file(tensors, vectors_url)

    # Everything written through save() was produced by THIS engine: stamp the
    # substrate when the caller built the sidecar by hand (make() already
    # stamps). Norm-backfill deliberately bypasses save() with raw JSON copies,
    # so a legacy artifact's unknown origin is never overwritten with a guess.
    if sidecar.substrate is None:
        sidecar.substrate = SUBSTRATE

    # The paired-difference-PCA family's sign comes from TRAIN-label majority —
    # it has no held-out split to run the RepE paper's step-4 rule on, unlike a
    # reader. Stamping it here (rather than at each sidecar constructor) means
    # the difference between "held-out data decided this direction's sign" and
    # "the training labels did" is readable off any artifact, without a reader
    # having to know which code path wrote it. A sidecar that already carries a
    # convention — a reader-derived vector does — is left alone. Swift twin:
    # ``SteeringVectorStore.stamped``.
    if sidecar.signConvention is None and (
            sidecar.extractionMethod
            == vector_math.ExtractionMethod.PAIRED_DIFFERENCE_PCA.value
            or sidecar.recipeMethod == "repeLAT"):
        sidecar.signConvention = "trainMajority"

    sidecar_url = os.path.join(directory, f"{name}.json")
    with open(sidecar_url, "w", encoding="utf-8") as handle:
        json.dump(sidecar.to_dict(), handle, sort_keys=True, indent=2)
    return vectors_url, sidecar_url


def gemmascope_pre_convention_warning(sidecar: SteeringVectorSidecar,
                                      artifact_path: str) -> str | None:
    """The loud pre-convention advisory for Gemma-Scope-sourced artifacts.

    Server imports made before WS7.2 saved the RAW SAE decoder row; the
    decided convention (the Swift app's, "analyzed-vector-norm-match")
    rescales the row at import. A Gemma-Scope-sourced sidecar without a
    ``gemmascopeConvention`` stamp therefore has unknown/legacy scaling —
    existing artifacts are never migrated in place; re-import instead.
    Returns the warning text, or ``None`` for non-Gemma-Scope or stamped
    artifacts.
    """
    from_gemma_scope = (sidecar.extractionMethod == "gemmaScopeSAE"
                        or (sidecar.stimulusSetHash or "").startswith("gemmascope:"))
    if not from_gemma_scope or sidecar.gemmascopeConvention is not None:
        return None
    return (f"vector artifact '{artifact_path}' is a Gemma Scope SAE import "
            f"without a gemmascopeConvention stamp: pre-convention import — "
            f"re-import the feature before evidence use (the decided "
            f"conventions are 'analyzed-vector-norm-match' for report-ranked "
            f"imports and 'residual-norm-match' for direct feature-ID "
            f"imports; raw-decoder-row imports are not comparable to either)")


def load(directory: str, name: str) -> tuple[ConceptVectors, SteeringVectorSidecar]:
    sidecar_url = os.path.join(directory, f"{name}.json")
    with open(sidecar_url, encoding="utf-8") as handle:
        sidecar = SteeringVectorSidecar.from_dict(json.load(handle))
    advisory = gemmascope_pre_convention_warning(sidecar, sidecar_url)
    if advisory is not None:
        # Loud but non-fatal: every experiment/validation path funnels through
        # this loader, so the advisory reaches sweeps, variants, and studies
        # without each caller opting in. stacklevel points at the caller.
        warnings.warn(advisory, UserWarning, stacklevel=2)

    vectors_url = os.path.join(directory, f"{name}.safetensors")
    tensors = load_file(vectors_url)
    per_layer: list[list[float]] = []
    for i in range(sidecar.layerCount):
        key = f"layer_{i}"
        if key not in tensors:
            raise FileNotFoundError(f"{vectors_url} missing {key}")
        values = np.asarray(tensors[key], dtype=np.float32)
        # Artifacts saved before the save-time guard (or written by other
        # tools) can carry non-finite bytes; injecting NaN/Inf poisons the
        # whole residual stream, so refuse loudly with the artifact named.
        if not np.isfinite(values).all():
            raise ValueError(
                f"vector artifact '{vectors_url}' layer_{i} contains non-finite "
                f"values — corrupt or pre-guard legacy artifact; re-extract the "
                f"concept before using it")
        per_layer.append(values.tolist())
    return ConceptVectors(per_layer=per_layer), sidecar


def load_neutral_mean(directory: str, name: str) -> list[list[float]] | None:
    """The per-layer neutral residual mean stored with an artifact, or None.

    ``None`` means the artifact predates the neutral-mean stamp or was
    extracted without a neutral corpus — callers must then treat the ablation
    mean-alignment as UNKNOWN, never as zero."""
    sidecar_url = os.path.join(directory, f"{name}.json")
    with open(sidecar_url, encoding="utf-8") as handle:
        sidecar = SteeringVectorSidecar.from_dict(json.load(handle))
    if sidecar.neutralMeanSource is None:
        return None
    tensors = load_file(os.path.join(directory, f"{name}.safetensors"))
    per_layer: list[list[float]] = []
    for i in range(sidecar.layerCount):
        key = f"neutral_mean_layer_{i}"
        if key not in tensors:
            raise ValueError(
                f"artifact '{directory}/{name}' stamps neutralMeanSource but "
                f"its safetensors is missing {key} — corrupt artifact")
        values = np.asarray(tensors[key], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(
                f"artifact '{directory}/{name}' {key} contains non-finite values")
        per_layer.append(values.tolist())
    return per_layer


def require_native_substrate(sidecar: SteeringVectorSidecar, artifact_path: str) -> None:
    """Refuse a vector artifact stamped by a different engine.

    Steering directions live in a substrate-specific activation basis
    (CUDA/MPS-HF vs MLX/Metal activations do not match), so injecting a
    foreign-substrate vector silently produces meaningless steering. Unstamped
    (``substrate`` absent) artifacts pass: they are legacy files with no way to
    know their origin — absent is never treated as a guessable value.
    """
    stamped = sidecar.substrate
    if stamped is not None and stamped != SUBSTRATE:
        raise ValueError(
            f"vector artifact '{artifact_path}' was extracted on substrate "
            f"{stamped!r}; this engine is {SUBSTRATE!r} — steering vectors do "
            f"not transfer across engines, re-extract the concept on this "
            f"substrate")
