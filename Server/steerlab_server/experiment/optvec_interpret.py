"""OptVec interpretation — what, semantically, is IN an optimized vector, and
what the solution FAMILY looks like (plan of record:
``docs/OPTVEC-OPTIMIZED-INJECTION-VECTORS-PLAN.md``, WP7).

Training (WP2) discovers a direction sufficient to move judgments; eval (WP4)
reads its held-out behavior. This verb asks the inverted pipeline's last
question — *behavior → vector → concept* — by pointing every reading
instrument the engine already has at one artifact, and then summarizing across
a family of them.

Six stages, all inference-only, all optional except the first:

1. **Logit lens** — ±v through the model's own final norm + unembedding
   (:meth:`SteeredModel.logits_for_residual_vector`, via the existing
   :func:`steering.extractor.logit_lens`): top-K promoted and suppressed
   tokens for each sign.
2. **SAE decomposition** — the existing Gemma Scope machinery
   (:func:`experiment.gemma_scope.analyze`, ``analyzed-vector-norm-match``
   convention), top-k features by decoder-direction cosine, plus the encoder
   activation of the direction.
3. **J-lens static support** — the existing :func:`jlens.decompose.decompose`,
   whose report is stored VERBATIM so its mandatory ``nullEnergyFraction``
   travels with every energy figure it states.
4. **Steered generations** — greedy generations on a pinned probe battery at
   α=0 and each declared multiple, through the deployed injection path.
5. **Self-explanation** (patchscope/SelfIE-style) — the model asked what the
   direction means, while the direction is injected. Stamped
   ``{"suggestive": true}``: a model's self-report under injection is a
   hypothesis generator, never evidence (plan §7).
6. **Library cosines** — the same-layer cosine table with random-direction null
   percentiles, reused wholesale from :mod:`optvec_eval`.

**Claim grammar (binding, plan §1): SUFFICIENCY.** Nothing here is a claim
that the direction is, or is not, "the" concept it resembles. And per plan §1
the family-level result is first-class: *"different solutions match different
known concepts"* and *"solutions match nothing in the library"* are both
reportable findings — the second is the ALIEN result, so
:func:`family_summary` gives it its own row (``no-library-match``) rather than
treating an unmatched solution as a failure or an omission.

Optional stages that are not configured are recorded as
``{"skipped": <reason>}`` — never silently absent. A stage that IS configured
but cannot run (no ``sae_lens`` installed, no such lens in the workspace, the
SAE download fails on a compute node) is recorded the same way with
``"attempted": true`` and the underlying error: an interpretation run is a
reading, not a gate, and losing five readings because the sixth is
unavailable would be the wrong trade. The reason is loud in ``interpret.json``
and in the run's ``config.json`` notes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Callable

from . import gemma_scope, paths, prompt_render, scoring, truncation_gate
from .generate import CellInjection, generate
from .optvec_eval import (FileRef, LoadedArtifact, OptVecArtifactError,
                          OptVecEvalDataError, OptVecTarget, cell_injection,
                          library_comparison, load_artifact,
                          load_neutral_texts, require_optvec, verify_model)
from .run_config import write_run_config

#: Run types. Each verb owns its own immutable run directory: an interpretation
#: is a separate measurement from the training run it reads, and may be
#: repeated (a second SAE, a new lens) without touching a finished record.
RUN_TYPE = "optvec-interpret"
FAMILY_RUN_TYPE = "optvec-family"

INTERPRET_JSON = "interpret.json"
GENERATIONS = "generations.jsonl"
FAMILY_JSON = "family.json"

SCHEMA_VERSION = 1

DEFAULT_ALPHA_MULTIPLES = (0.5, 1.0, 2.0)
DEFAULT_MAX_TOKENS = 128
DEFAULT_LOGIT_LENS_TOP_K = 50
DEFAULT_SAE_TOP_K = 25
DEFAULT_NULL_SAMPLES = 1000

DEFAULT_SELF_EXPLANATION_TEMPLATE = (
    "What concept or disposition does this direction represent? "
    "Describe it in one sentence.")

#: The stamp on every self-explanation output. Kept as one constant so the
#: block cannot be written without it.
SUGGESTIVE_STAMP = {
    "suggestive": True,
    "note": "self-report under injection — not evidence",
}

#: The family's own category for a solution whose best library cosine does not
#: clear its random-direction null. The plan calls this the alien finding, so
#: it is a first-class row of the distribution table, never an error or a gap.
NO_LIBRARY_MATCH = "no-library-match"

#: A library cosine counts as a MATCH only above this percentile of the
#: matched random-direction null the eval's library table already carries.
#: (Same rule the J-lens support readout enforces for energy: a magnitude is
#: not readable without its null.)
DEFAULT_MATCH_NULL_PERCENTILE = 99.0

#: How many entries the concentration statistics put in their numerator.
CONCENTRATION_TOP_N = 5

CONCENTRATION_FORMULA = (
    "concentration = sum of the |value| of the top 5 entries / sum of the "
    "|value| of all reported top-K entries (1.0 = the whole reported mass "
    "sits in 5 entries; K/5 entries of equal magnitude give 5/K)")

CLAIM = "sufficiency"

CLAIM_NOTE = (
    "OptVec certifies SUFFICIENCY only: a direction sufficient to shift the "
    "declared judgments within the declared budgets. An interpretation stage "
    "describes what the direction resembles — it never certifies that the "
    "vector IS that concept, that the concept is necessary, or that the "
    "solution is unique.")


class OptVecInterpretConfigError(ValueError):
    """An interpretation config that cannot be run as written."""


class OptVecInterpretDataError(OptVecEvalDataError):
    """A pinned input file that cannot be read as configured (hash drift…).

    Subclasses the eval's error because the probe battery is loaded by the
    eval's own pinned-file loader — a caller that catches one catches both.
    """


class OptVecFamilyError(ValueError):
    """A family summary that cannot be assembled from the given runs."""


# ------------------------------------------------------------------- config


@dataclass
class LogitLensConfig:
    top_k: int = DEFAULT_LOGIT_LENS_TOP_K

    def __post_init__(self) -> None:
        self.top_k = int(self.top_k)
        if self.top_k < 1:
            raise OptVecInterpretConfigError("logitLens.topK must be >= 1")

    @classmethod
    def from_dict(cls, payload: dict) -> "LogitLensConfig":
        _require_object(payload, "logitLens")
        _refuse_unknown(payload, {"topK"}, "logitLens")
        return cls(top_k=payload.get("topK", DEFAULT_LOGIT_LENS_TOP_K))

    def to_dict(self) -> dict:
        return {"topK": self.top_k}


@dataclass
class SAEConfig:
    """Where the imported Gemma Scope SAE lives. The identifiers are the
    EXISTING machinery's own (``gemma_scope.scope_info`` names ``release`` and
    ``saeID``; ``analyze`` takes them straight through), so an SAE that is
    analyzable by the current verbs is nameable here without translation."""

    release: str
    sae_id: str
    #: Defaults to the artifact's own optvec layer — the only layer at which an
    #: optvec artifact says anything.
    layer: int | None = None
    top_k: int = DEFAULT_SAE_TOP_K

    def __post_init__(self) -> None:
        if not (self.release or "").strip():
            raise OptVecInterpretConfigError("sae.release is required")
        if not (self.sae_id or "").strip():
            raise OptVecInterpretConfigError("sae.saeID is required")
        self.top_k = int(self.top_k)
        if self.top_k < 1:
            raise OptVecInterpretConfigError("sae.topK must be >= 1")
        if self.layer is not None:
            self.layer = int(self.layer)
            if self.layer < 0:
                raise OptVecInterpretConfigError("sae.layer must be >= 0")

    @classmethod
    def from_dict(cls, payload: dict) -> "SAEConfig":
        _require_object(payload, "sae")
        _refuse_unknown(payload, {"release", "saeID", "layer", "topK"}, "sae")
        return cls(release=payload.get("release", ""),
                   sae_id=payload.get("saeID", ""),
                   layer=payload.get("layer"),
                   top_k=payload.get("topK", DEFAULT_SAE_TOP_K))

    def to_dict(self) -> dict:
        return {"release": self.release, "saeID": self.sae_id,
                "layer": self.layer, "topK": self.top_k}


@dataclass
class SelfExplanationConfig:
    enabled: bool = True
    template: str = DEFAULT_SELF_EXPLANATION_TEMPLATE
    alpha_multiple: float = 1.0

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        if not (self.template or "").strip():
            raise OptVecInterpretConfigError(
                "selfExplanation.template must be a prompt")
        self.alpha_multiple = float(self.alpha_multiple)
        if self.alpha_multiple <= 0:
            raise OptVecInterpretConfigError(
                "selfExplanation.alphaMultiple must be positive — the whole "
                "point is to read the model WHILE the direction is present")

    @classmethod
    def from_dict(cls, payload: dict) -> "SelfExplanationConfig":
        _require_object(payload, "selfExplanation")
        _refuse_unknown(payload, {"enabled", "template", "alphaMultiple"},
                        "selfExplanation")
        return cls(enabled=payload.get("enabled", True),
                   template=payload.get("template",
                                        DEFAULT_SELF_EXPLANATION_TEMPLATE),
                   alpha_multiple=payload.get("alphaMultiple", 1.0))

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "template": self.template,
                "alphaMultiple": self.alpha_multiple}


@dataclass
class LibraryConfig:
    vector_paths: list[str] = field(default_factory=list)
    null_samples: int = DEFAULT_NULL_SAMPLES

    def __post_init__(self) -> None:
        self.vector_paths = [str(p) for p in self.vector_paths]
        self.null_samples = int(self.null_samples)
        if self.null_samples < 1:
            raise OptVecInterpretConfigError(
                "library.nullSamples must be at least 1 — a cosine against a "
                "library vector is never reported without its null")

    @classmethod
    def from_dict(cls, payload: dict) -> "LibraryConfig":
        _require_object(payload, "library")
        _refuse_unknown(payload, {"vectorPaths", "nullSamples"}, "library")
        return cls(vector_paths=list(payload.get("vectorPaths", [])),
                   null_samples=payload.get("nullSamples",
                                            DEFAULT_NULL_SAMPLES))

    def to_dict(self) -> dict:
        return {"vectorPaths": list(self.vector_paths),
                "nullSamples": self.null_samples}


def _require_object(payload, label: str) -> None:
    if not isinstance(payload, dict):
        raise OptVecInterpretConfigError(
            f"'{label}' must be a JSON object (got {payload!r})")


def _refuse_unknown(payload: dict, known: set, label: str) -> None:
    unknown = sorted(set(payload) - known)
    if unknown:
        raise OptVecInterpretConfigError(
            f"unknown {label} key(s): " + ", ".join(unknown))


_CONFIG_KEYS = {
    "vectorArtifact": "vector_artifact",
    "modelID": "model_id", "revision": "revision", "name": "name",
    "device": "device", "dtype": "dtype",
    "alphaMultiples": "alpha_multiples",
    "maxTokens": "max_tokens", "seed": "seed",
    "jlensLensID": "jlens_lens_id",
    "promptMode": "prompt_mode", "systemPrompt": "system_prompt",
    "qwenThinkingEnabled": "qwen_thinking_enabled",
}

_BLOCK_KEYS = {"probePrompts", "logitLens", "sae", "selfExplanation",
               "library"}


@dataclass
class OptVecInterpretConfig:
    vector_artifact: str
    probe_prompts: FileRef | None = None
    model_id: str | None = None
    revision: str | None = None
    name: str | None = None
    device: str | None = None
    dtype: str = "auto"

    alpha_multiples: tuple[float, ...] = DEFAULT_ALPHA_MULTIPLES
    max_tokens: int = DEFAULT_MAX_TOKENS
    seed: int = 0

    logit_lens: LogitLensConfig = field(default_factory=LogitLensConfig)
    sae: SAEConfig | None = None
    jlens_lens_id: str | None = None
    self_explanation: SelfExplanationConfig = field(
        default_factory=SelfExplanationConfig)
    library: LibraryConfig = field(default_factory=LibraryConfig)

    prompt_mode: str = prompt_render.CHAT_ASSISTANT
    system_prompt: str | None = None
    qwen_thinking_enabled: bool = False

    def __post_init__(self) -> None:
        if not (self.vector_artifact or "").strip():
            raise OptVecInterpretConfigError(
                "'vectorArtifact' is required — the extension-less path of the "
                "optvec artifact to interpret")
        multiples = tuple(float(m) for m in self.alpha_multiples)
        if not multiples:
            raise OptVecInterpretConfigError(
                "alphaMultiples must name at least one dose")
        if any(m <= 0 for m in multiples):
            raise OptVecInterpretConfigError(
                "alphaMultiples must be positive — α=0 is always generated as "
                "the in-run baseline and is not a declared dose")
        self.alpha_multiples = multiples
        self.max_tokens = int(self.max_tokens)
        if self.max_tokens < 1:
            raise OptVecInterpretConfigError("maxTokens must be at least 1")
        self.seed = int(self.seed)
        if self.jlens_lens_id is not None:
            self.jlens_lens_id = str(self.jlens_lens_id).strip() or None

    @classmethod
    def from_dict(cls, payload: dict) -> "OptVecInterpretConfig":
        if not isinstance(payload, dict):
            raise OptVecInterpretConfigError(
                "the OptVec interpret config must be a JSON object")
        unknown = sorted(set(payload) - set(_CONFIG_KEYS) - _BLOCK_KEYS)
        if unknown:
            raise OptVecInterpretConfigError(
                "unknown OptVec interpret config key(s): " + ", ".join(unknown))
        kwargs = {name: payload[key]
                  for key, name in _CONFIG_KEYS.items() if key in payload}
        if "vector_artifact" not in kwargs:
            raise OptVecInterpretConfigError("'vectorArtifact' is required")
        if "probePrompts" in payload:
            kwargs["probe_prompts"] = FileRef.from_dict(
                payload["probePrompts"], "probePrompts")
        if "logitLens" in payload:
            kwargs["logit_lens"] = LogitLensConfig.from_dict(
                payload["logitLens"])
        if "sae" in payload:
            kwargs["sae"] = SAEConfig.from_dict(payload["sae"])
        if "selfExplanation" in payload:
            kwargs["self_explanation"] = SelfExplanationConfig.from_dict(
                payload["selfExplanation"])
        if "library" in payload:
            kwargs["library"] = LibraryConfig.from_dict(payload["library"])
        return cls(**kwargs)

    def to_dict(self) -> dict:
        out = {key: getattr(self, name)
               for key, name in _CONFIG_KEYS.items()}
        out["alphaMultiples"] = list(self.alpha_multiples)
        out["logitLens"] = self.logit_lens.to_dict()
        out["sae"] = self.sae.to_dict() if self.sae is not None else None
        out["selfExplanation"] = self.self_explanation.to_dict()
        out["library"] = self.library.to_dict()
        if self.probe_prompts is not None:
            out["probePrompts"] = self.probe_prompts.to_dict()
        return out


def load_config(path: str) -> OptVecInterpretConfig:
    with open(path, encoding="utf-8") as handle:
        return OptVecInterpretConfig.from_dict(json.load(handle))


# ------------------------------------------------------------ shared helpers


def sha256_text(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def concentration(values: list[float], top_n: int = CONCENTRATION_TOP_N
                  ) -> float | None:
    """Share of the reported magnitude carried by the ``top_n`` largest
    entries — the one number that says "this readout is a few things" vs.
    "this readout is smeared". See :data:`CONCENTRATION_FORMULA`; the formula
    string is written into every artifact that reports the statistic, because a
    concentration figure with an unstated denominator is unreadable."""
    magnitudes = sorted((abs(float(v)) for v in values), reverse=True)
    total = sum(magnitudes)
    if not magnitudes or total <= 0:
        return None
    return float(sum(magnitudes[:top_n]) / total)


def stage_skip_reason(stage: dict | None) -> str | None:
    """The reason a stage did not run, or ``None`` when it ran.

    Deliberately typed: the library table has its OWN ``skipped`` key — a LIST
    of library artifacts that could not be compared — and a stage that ran but
    skipped two of ten comparisons is not a skipped stage. Only a string
    ``skipped`` is a stage-level skip.
    """
    if not isinstance(stage, dict):
        return None
    reason = stage.get("skipped")
    return reason if isinstance(reason, str) else None


def _mean(values: list[float]) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    return float(sum(numbers) / len(numbers)) if numbers else None


def condition_of(optvec_block: dict | None) -> str:
    """The S0/S1/S2/S3 condition an artifact was born in, read from its
    ``optvec`` sidecar block (plan §2).

    The λ's and the shuffle flag live in the training driver's ``objective``
    sub-block; older/hand-built blocks may carry them at the top level, so both
    are read. Rule order is the plan's, and it matters:

    * ``shuffleTargetLabels`` → **s0** (the shuffled-target null is S0 whatever
      its λ's are — it is S2's optimization pointed at permuted labels);
    * ``lambdaAnchor == 0 and lambdaCap == 0`` → **s1** (shift-only);
    * ``lambdaOrth > 0`` → **s3** (multiplicity enumeration);
    * otherwise → **s2** (the composite primary).

    An artifact that states none of these is ``"unknown"`` — never guessed into
    a condition, because the condition label is what the family tables group by.
    """
    if not isinstance(optvec_block, dict):
        return "unknown"
    objective = optvec_block.get("objective")
    objective = objective if isinstance(objective, dict) else {}

    def read(key):
        for source in (optvec_block, objective):
            if key in source:
                return source[key]
        return None

    if read("shuffleTargetLabels") is True:
        return "s0"
    anchor = _number(read("lambdaAnchor"))
    cap = _number(read("lambdaCap"))
    orth = _number(read("lambdaOrth"))
    if anchor is None and cap is None and orth is None:
        return "unknown"
    if (anchor or 0.0) == 0.0 and (cap or 0.0) == 0.0:
        return "s1"
    if (orth or 0.0) > 0.0:
        return "s3"
    return "s2"


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def load_probe_prompts(ref: FileRef) -> list[str]:
    """The pinned probe battery — one prompt per line, through the eval's own
    hashed line-oriented loader (so a battery that loads there loads here)."""
    try:
        return load_neutral_texts(ref)
    except OptVecEvalDataError as exc:
        raise OptVecInterpretDataError(str(exc)) from exc


# ------------------------------------------------------------- stage 1: lens


def logit_lens_stage(model, target: OptVecTarget, top_k: int) -> dict:
    """±v read through the model's own final norm and unembedding.

    Delegates to :func:`steering.extractor.logit_lens`, which is the engine's
    single logit-lens implementation (it calls
    ``SteeredModel.logits_for_residual_vector`` — final norm THEN head, never
    the head alone). Called twice: once for the artifact, once for its
    negation, because a direction's suppressed tokens are as much of its
    content as its promoted ones and the two signs are what a reader compares.
    """
    from ..steering import extractor
    from ..steering.vector_store import ConceptVectors

    vectors = target.artifact.vectors
    negated = ConceptVectors(per_layer=[[-x for x in row]
                                        for row in vectors.per_layer])

    def sign_report(concept_vectors) -> dict:
        report = extractor.logit_lens(model, concept_vectors, target.layer,
                                      top_k=top_k)
        promoted = [_lens_token(t) for t in report.top_positive]
        suppressed = [_lens_token(t) for t in report.top_negative]
        return {
            "promoted": promoted,
            "suppressed": suppressed,
            "promotedConcentration": concentration(
                [row["logit"] for row in promoted]),
        }

    return {
        "layer": target.layer,
        "topK": int(top_k),
        "readout": "model final norm + unembedding "
                   "(SteeredModel.logits_for_residual_vector)",
        "concentrationFormula": CONCENTRATION_FORMULA,
        "positive": sign_report(vectors),
        "negative": sign_report(negated),
    }


def _lens_token(token) -> dict:
    return {"tokenID": int(token.token_id), "piece": token.token,
            "logit": float(token.logit)}


# -------------------------------------------------------------- stage 2: SAE


def _analyze_gemma_scope(directory: str, name: str, *, layer: int,
                         release: str, sae_id: str, top_k: int):
    """Seam over the EXISTING Gemma Scope analysis (decoder-cosine ranking,
    ``analyzed-vector-norm-match`` context). A function of its own so a test
    can stand in for the download without reimplementing the ranking."""
    return gemma_scope.analyze(directory, name, layer=layer, release=release,
                               sae_id=sae_id, top_k=top_k)


def _load_sae(release: str, sae_id: str):
    """The SAE object itself, for the ENCODER half (which
    :func:`gemma_scope.analyze` does not expose — it returns decoder rows).
    Same loader, same identifiers; see the module note about the second load."""
    from sae_lens import SAE

    sae, _config, _sparsity = SAE.from_pretrained(release=release,
                                                  sae_id=sae_id)
    return sae


def sae_stage(target: OptVecTarget, config: SAEConfig | None) -> dict:
    """Top features by decoder-direction cosine and by encoder activation.

    The decoder half is the existing machinery, called not copied: the same
    ``analyze`` that produces the reports ``gemma_scope.import_feature`` reads,
    so a feature named here can be imported as a peer steering artifact under
    the ``analyzed-vector-norm-match`` convention with no extra step.

    The encoder half loads the SAE a second time, because ``analyze`` neither
    accepts a preloaded SAE nor returns activations. That is a known cost, not
    a second implementation — no SAE math is written here.
    """
    if config is None:
        return {"skipped": "no 'sae' block declared — SAE decomposition needs "
                           "an imported Gemma Scope SAE (release + saeID)"}
    layer = config.layer if config.layer is not None else target.layer
    directory = os.path.dirname(target.artifact.path) or "."
    out: dict = {"release": config.release, "saeID": config.sae_id,
                 "layer": layer, "topK": config.top_k,
                 "convention": gemma_scope.IMPORT_CONVENTION,
                 "conventionNote":
                     "feature decoder rows are rescaled to the analyzed "
                     "vector's L2 norm AT IMPORT (gemma_scope.import_feature); "
                     "the cosines below are scale-free and unaffected",
                 "concentrationFormula": CONCENTRATION_FORMULA}
    try:
        report = _analyze_gemma_scope(directory, target.artifact.name,
                                      layer=layer, release=config.release,
                                      sae_id=config.sae_id,
                                      top_k=config.top_k)
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        return {**out, "attempted": True,
                "skipped": f"Gemma Scope analysis unavailable: {exc}"}

    def rows(feature_rows) -> list[dict]:
        return [{"feature": int(row.feature), "cosine": float(row.cosine),
                 "sparsity": row.sparsity} for row in feature_rows]

    out.update({
        "analyzedLayer": report.layer,
        "vectorNorm": report.vector_norm,
        "decoderShape": list(report.decoder_shape),
        "topByDecoderCosine": rows(report.top_absolute),
        "topPositiveByDecoderCosine": rows(report.top_positive),
        "topNegativeByDecoderCosine": rows(report.top_negative),
    })
    out["decoderCosineConcentration"] = concentration(
        [row["cosine"] for row in out["topByDecoderCosine"]])
    out["encoderActivation"] = _encoder_activation(target, config)
    return out


def _encoder_activation(target: OptVecTarget, config: SAEConfig) -> dict:
    """Top features by the SAE's own encoder applied to the direction.

    Caveat stated in the artifact, not just here: an SAE encoder is fitted on
    residual STATES, and a bare direction is off that distribution. The ranking
    is a reading aid beside the decoder cosines, not a reconstruction.
    """
    try:
        import torch

        sae = _load_sae(config.release, config.sae_id)
        encode = getattr(sae, "encode", None)
        if encode is None:
            return {"skipped": "the loaded SAE exposes no encode()"}
        row = target.direction
        activations = encode(torch.tensor(row, dtype=torch.float32))
        activations = activations.detach().float().reshape(-1).cpu()
        k = min(int(config.top_k), int(activations.numel()))
        top = torch.topk(activations, k=k)
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        return {"attempted": True,
                "skipped": f"SAE encoder unavailable: {exc}"}
    features = [{"feature": int(i), "activation": float(v)}
                for v, i in zip(top.values.tolist(), top.indices.tolist())]
    return {
        "topFeatures": features,
        "encodedVectorNorm": float(sum(x * x for x in target.direction) ** 0.5),
        "note": "the stored row (norm = optvec alphaAbsolute) encoded as-is; "
                "an SAE encoder is fitted on residual states, so a bare "
                "direction is off-distribution — a reading aid, not a "
                "reconstruction",
        "concentration": concentration([f["activation"] for f in features]),
    }


# ---------------------------------------------------------- stage 3: J-lens


def jlens_stage(target: OptVecTarget, lens_id: str | None,
                root: str | None = None) -> dict:
    """The existing J-lens static support readout, stored verbatim.

    Zero new math: :func:`jlens.decompose.decompose` already refuses to state
    an ``energyFraction`` without the matched-norm ``nullEnergyFraction`` beside
    it, so the whole report is copied through rather than summarized — a
    summary is exactly where that pairing gets lost.
    """
    if not lens_id:
        return {"skipped": "no 'jlensLensID' declared — the J-lens support "
                           "readout needs an imported lens in the workspace"}
    try:
        from ..jlens import decompose as decompose_mod

        report = decompose_mod.decompose(
            lens_id=lens_id,
            vector_directory=os.path.dirname(target.artifact.path) or ".",
            vector_name=target.artifact.name,
            layers=[target.layer], root=root)
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        return {"attempted": True, "lensID": lens_id,
                "skipped": f"J-lens support unavailable: {exc}"}
    return {"lensID": lens_id, "layer": target.layer,
            "nullRule": "energy is never reported without its matched-norm "
                        "null (jlens.decompose): every layer entry carries "
                        "nullEnergyFraction",
            "report": report}


# ----------------------------------------------------- stage 4: generations


def probe_generations(model, target: OptVecTarget, prompts: list[str],
                      config: OptVecInterpretConfig, *,
                      log: Callable[[str], None] | None = None
                      ) -> tuple[list[dict], dict]:
    """Greedy generations on every probe prompt at α=0 and each multiple,
    through the DEPLOYED injection path (``generate`` + ``CellInjection``) —
    the same call the study runner makes, so a transcript here is commensurable
    with a transcript in a run.

    α=0 passes ``injections=None``: the baseline arm is the unsteered model,
    not "the steered path with a zero" (which would still build an injector
    chain and is a different code path to attribute a difference to).
    """
    records: list[dict] = []
    per_dose: list[dict] = []
    common = {
        "modelID": target.artifact.sidecar.modelID,
        "modelRevision": getattr(model, "revision", None),
        "promptMode": config.prompt_mode,
        "systemPromptHash": sha256_text(config.system_prompt),
        "artifactReference": target.artifact.reference,
        "artifactTensorSHA256": target.artifact.tensor_sha256,
        "layer": target.layer,
        "maxTokens": config.max_tokens,
        # Greedy by construction: an interpretation battery that sampled would
        # make every transcript a draw rather than a reading of the vector.
        "temperature": 0.0, "doSample": False, "topP": None, "topK": None,
    }
    for multiple in (0.0, *config.alpha_multiples):
        injections = (None if multiple == 0.0
                      else [cell_injection(target, multiple)])
        condition = ("alpha-0" if multiple == 0.0
                     else f"alpha-{_condition_suffix(multiple)}")
        outputs: list[str] = []
        for index, prompt in enumerate(prompts):
            token_ids: list = []
            text = generate(
                model, prompt, model_id=target.artifact.sidecar.modelID,
                max_tokens=config.max_tokens, temperature=0.0,
                injections=injections, prompt_mode=config.prompt_mode,
                system_prompt=config.system_prompt,
                qwen_thinking_enabled=config.qwen_thinking_enabled,
                token_ids_out=token_ids)
            outputs.append(text)
            records.append({
                **common,
                "condition": condition,
                "interventionState": ("baseline" if multiple == 0.0
                                      else "steered"),
                "alphaMultiple": float(multiple),
                "alphaAbsolute": float(multiple) * target.alpha_absolute,
                "promptIndex": index,
                "promptID": f"probe-{index + 1}",
                "prompt": prompt,
                "output": text,
                "wordCount": scoring.word_count(text),
                "distinct2": scoring.distinct_bigram_ratio(text),
                # A probe transcript that hit the cap is a cut-off reading of
                # the vector, not a short one — and at high α that is exactly
                # the regime where truncation gets common. Stamped like every
                # other generation record's.
                truncation_gate.RECORD_KEY: truncation_gate.finish_reason(
                    token_ids, max_tokens=config.max_tokens,
                    stop_ids=truncation_gate.stop_token_ids(model)),
            })
        per_dose.append({
            "alphaMultiple": float(multiple),
            "alphaAbsolute": float(multiple) * target.alpha_absolute,
            "condition": condition,
            "promptCount": len(prompts),
            "meanWordCount": _mean([scoring.word_count(t) for t in outputs]),
            "meanDistinct2": _mean([scoring.distinct_bigram_ratio(t)
                                    for t in outputs]),
            # A blunt, honest movement statistic: how many prompts produced a
            # different transcript from the α=0 arm. Not an effect size —
            # effect sizes come from eval and the confirm study.
            "changedFromBaselineCount": None,
        })
        if log is not None:
            log(f"probe battery at {multiple}×: {len(prompts)} generation(s)")

    baseline = {r["promptID"]: r["output"] for r in records
                if r["alphaMultiple"] == 0.0}
    for entry in per_dose:
        rows = [r for r in records
                if r["alphaMultiple"] == entry["alphaMultiple"]]
        entry["changedFromBaselineCount"] = sum(
            1 for r in rows if r["output"] != baseline.get(r["promptID"]))
    return records, {"promptCount": len(prompts),
                     "recordCount": len(records),
                     "instrument": "greedy generation through the deployed "
                                   "CellInjection path (experiment.generate)",
                     "doses": per_dose}


def _condition_suffix(multiple: float) -> str:
    text = f"{float(multiple):g}"
    return text.replace(".", "p").replace("-", "neg")


# ------------------------------------------------- stage 5: self-explanation


def self_explanation_stage(model, target: OptVecTarget,
                           config: OptVecInterpretConfig) -> dict:
    """Ask the model what the direction means, with the direction injected.

    Patchscope/SelfIE-shaped, and stamped :data:`SUGGESTIVE_STAMP` without an
    option to omit it. A model's verbal self-report under injection is
    generated BY the thing being measured: it is a hypothesis generator for the
    other five stages, and the plan (§7) admits it only marked suggestive,
    never evidential. The unsteered answer to the same prompt is generated
    beside it so a reader can see what the injection changed.
    """
    settings = config.self_explanation
    if not settings.enabled:
        return {**SUGGESTIVE_STAMP,
                "skipped": "selfExplanation.enabled is false"}
    injections = [cell_injection(target, settings.alpha_multiple)]
    steered = generate(
        model, settings.template, model_id=target.artifact.sidecar.modelID,
        max_tokens=config.max_tokens, temperature=0.0, injections=injections,
        prompt_mode=config.prompt_mode, system_prompt=config.system_prompt,
        qwen_thinking_enabled=config.qwen_thinking_enabled)
    baseline = generate(
        model, settings.template, model_id=target.artifact.sidecar.modelID,
        max_tokens=config.max_tokens, temperature=0.0, injections=None,
        prompt_mode=config.prompt_mode, system_prompt=config.system_prompt,
        qwen_thinking_enabled=config.qwen_thinking_enabled)
    return {
        **SUGGESTIVE_STAMP,
        "template": settings.template,
        "templateSHA256": sha256_text(settings.template),
        "alphaMultiple": settings.alpha_multiple,
        "alphaAbsolute": settings.alpha_multiple * target.alpha_absolute,
        "layer": target.layer,
        "output": steered,
        "baselineOutput": baseline,
        "changedFromBaseline": steered != baseline,
        "maxTokens": config.max_tokens,
        "temperature": 0.0,
    }


# ------------------------------------------------------------------ driver


def interpret(config: OptVecInterpretConfig, *, model=None,
              log: Callable[[str], None] | None = None,
              root: str | None = None) -> dict:
    """Run every configured stage against one optvec artifact and write the
    interpretation's own immutable run directory. Returns its summary dict."""
    def emit(message: str) -> None:
        if log is not None:
            log(message)

    artifact = load_artifact(config.vector_artifact)
    target = require_optvec(artifact)
    if config.model_id and artifact.sidecar.modelID \
            and config.model_id != artifact.sidecar.modelID:
        raise OptVecInterpretConfigError(
            f"config modelID {config.model_id!r} disagrees with the artifact's "
            f"{artifact.sidecar.modelID!r} — the artifact's model is the one "
            "its direction means anything in")

    prompts: list[str] = []
    if config.probe_prompts is not None:
        prompts = load_probe_prompts(config.probe_prompts)

    name = (config.name or "").strip() or artifact.name
    run_directory = paths.make_unique_run_directory(
        f"optvec-interpret-{name}", root)
    condition = condition_of(artifact.optvec)
    notes: dict = {"stage": "starting", "claim": CLAIM,
                   "artifact": artifact.identity(),
                   "condition": condition,
                   "optvec": {"layer": target.layer,
                              "alphaAbsolute": target.alpha_absolute},
                   "probePrompts": (config.probe_prompts.to_dict()
                                    if config.probe_prompts else None),
                   "promptCount": len(prompts),
                   "alphaMultiples": list(config.alpha_multiples)}
    dtype_name: str | None = None
    stages: dict = {}

    try:
        if model is None:
            from ..steering import model_loader
            device = model_loader.resolve_device(config.device)
            emit(f"loading {artifact.sidecar.modelID} on {device}")
            model = model_loader.load(
                config.model_id or artifact.sidecar.modelID,
                config.revision or artifact.sidecar.revision,
                dtype=config.dtype, device=device)
        dtype_name = getattr(model, "dtype", None)
        notes["model"] = verify_model(target, model)

        notes["stage"] = "logitLens"
        stages["logitLens"] = logit_lens_stage(model, target,
                                               config.logit_lens.top_k)
        emit("logit lens: ±v read through the final norm + unembedding")

        notes["stage"] = "sae"
        stages["sae"] = sae_stage(target, config.sae)

        notes["stage"] = "jlensSupport"
        stages["jlensSupport"] = jlens_stage(target, config.jlens_lens_id,
                                             root)

        notes["stage"] = "generations"
        if prompts:
            records, summary = probe_generations(model, target, prompts,
                                                 config, log=log)
            with open(os.path.join(run_directory, GENERATIONS), "w",
                      encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            stages["generations"] = summary
        else:
            stages["generations"] = {
                "skipped": "no 'probePrompts' declared — steered generations "
                           "need a pinned probe battery"}

        notes["stage"] = "selfExplanation"
        stages["selfExplanation"] = self_explanation_stage(model, target,
                                                           config)

        notes["stage"] = "library"
        stages["library"] = library_comparison(
            target, config.library.vector_paths,
            samples=config.library.null_samples, seed=config.seed)

        readout = {
            "schemaVersion": SCHEMA_VERSION,
            "runType": RUN_TYPE,
            "runID": os.path.basename(os.path.normpath(run_directory)),
            "claim": CLAIM,
            "claimNote": CLAIM_NOTE,
            "condition": condition,
            "artifact": artifact.identity(),
            "optvec": {"layer": target.layer,
                       "alphaAbsolute": target.alpha_absolute,
                       "sidecarBlock": artifact.optvec},
            "model": {**notes["model"], "dtype": dtype_name,
                      "device": str(getattr(model, "device", ""))},
            "probePrompts": (config.probe_prompts.to_dict()
                             if config.probe_prompts else None),
            "promptCount": len(prompts),
            "alphaMultiples": list(config.alpha_multiples),
            "config": config.to_dict(),
            "stages": stages,
        }
        with open(os.path.join(run_directory, INTERPRET_JSON), "w",
                  encoding="utf-8") as handle:
            json.dump(readout, handle, indent=2, sort_keys=True)
            handle.write("\n")

        skips = {key: stage_skip_reason(value)
                 for key, value in stages.items()}
        notes["stages"] = {key: ("skipped" if reason else "ran")
                           for key, reason in skips.items()}
        notes["skipped"] = {key: reason for key, reason in skips.items()
                            if reason}
        notes["stage"] = "complete"
        emit(f"optvec interpret → {run_directory}")
        return {"runDirectory": run_directory,
                "artifact": artifact.identity(),
                "condition": condition,
                "layer": target.layer,
                "alphaAbsolute": target.alpha_absolute,
                "stages": stages}
    finally:
        # Written LAST so the notes are complete, and in `finally` so a crashed
        # interpretation still records the stage it died in (run directories
        # are immutable and write_run_config never rewrites).
        write_run_config(run_directory, RUN_TYPE,
                         model_id=(config.model_id
                                   or artifact.sidecar.modelID),
                         revision=(getattr(model, "revision", None)
                                   or config.revision
                                   or artifact.sidecar.revision),
                         dtype=dtype_name, temperature=0.0, notes=notes)


# ----------------------------------------------------------- family summary


@dataclass
class FamilySummaryConfig:
    """Which interpretations to summarize. Entries are interpretation RUN
    DIRECTORIES (or a direct path to an ``interpret.json``) — the family table
    is a table of readings, so it is assembled from readings, not re-derived
    from artifacts."""

    interpret_runs: list[str] = field(default_factory=list)
    name: str | None = None
    match_null_percentile: float = DEFAULT_MATCH_NULL_PERCENTILE

    def __post_init__(self) -> None:
        self.interpret_runs = [str(r) for r in self.interpret_runs]
        if len(self.interpret_runs) < 2:
            raise OptVecFamilyError(
                "a family summary needs at least 2 interpretation runs — the "
                "distribution of matches over one solution is not a family")
        self.match_null_percentile = float(self.match_null_percentile)
        if not 0.0 <= self.match_null_percentile <= 100.0:
            raise OptVecFamilyError(
                "matchNullPercentile must be a percentile in [0, 100]")

    @classmethod
    def from_dict(cls, payload: dict) -> "FamilySummaryConfig":
        _require_object(payload, "the OptVec family config")
        _refuse_unknown(payload,
                        {"interpretRuns", "name", "matchNullPercentile"},
                        "OptVec family config")
        return cls(interpret_runs=list(payload.get("interpretRuns", [])),
                   name=payload.get("name"),
                   match_null_percentile=payload.get(
                       "matchNullPercentile", DEFAULT_MATCH_NULL_PERCENTILE))

    def to_dict(self) -> dict:
        return {"interpretRuns": list(self.interpret_runs),
                "name": self.name,
                "matchNullPercentile": self.match_null_percentile}


def interpret_json_path(reference: str, root: str | None = None) -> str:
    """Accept a run directory or the ``interpret.json`` itself."""
    resolved = paths.resolve(reference, root)
    if os.path.isdir(resolved):
        return os.path.join(resolved, INTERPRET_JSON)
    return resolved


def load_interpret(reference: str, root: str | None = None) -> dict:
    path = interpret_json_path(reference, root)
    if not os.path.exists(path):
        raise OptVecFamilyError(
            f"no {INTERPRET_JSON} at '{path}' — a family summarizes finished "
            "interpretation runs")
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except ValueError as exc:
        raise OptVecFamilyError(f"'{path}' is not readable JSON: {exc}") from exc
    found = payload.get("runType") if isinstance(payload, dict) else None
    if found != RUN_TYPE:
        raise OptVecFamilyError(
            f"'{path}' is not an {RUN_TYPE} readout (runType {found!r})")
    payload["_path"] = path
    return payload


def top_library_match(interpret_payload: dict) -> dict | None:
    """The best-|cosine| library row of one interpretation, or ``None`` when it
    compared against nothing."""
    library = (interpret_payload.get("stages") or {}).get("library") or {}
    entries = [e for e in (library.get("entries") or [])
               if isinstance(e, dict) and e.get("cosine") is not None]
    if not entries:
        return None
    return max(entries, key=lambda e: abs(float(e["cosine"])))


def library_compared_count(interpret_payload: dict) -> int:
    """How many library vectors this interpretation actually compared against.

    Read separately from the match itself because "matched nothing" and "was
    asked about nothing" are different facts, and only the first is the plan's
    alien finding.
    """
    library = (interpret_payload.get("stages") or {}).get("library") or {}
    compared = library.get("comparedCount")
    if isinstance(compared, int) and not isinstance(compared, bool):
        return compared
    return len([e for e in (library.get("entries") or [])
                if isinstance(e, dict)])


def _solution_row(payload: dict, *, match_percentile: float) -> dict:
    stages = payload.get("stages") or {}
    lens = stages.get("logitLens") or {}
    positive = ({} if stage_skip_reason(lens)
                else (lens.get("positive") or {}))
    promoted = list(positive.get("promoted") or [])
    sae = stages.get("sae") or {}
    sae_top = ([] if stage_skip_reason(sae)
               else list(sae.get("topByDecoderCosine") or []))

    top = top_library_match(payload)
    percentile = (None if top is None else top.get("nullPercentile"))
    matched = (top is not None and percentile is not None
               and float(percentile) >= match_percentile)
    return {
        "runID": payload.get("runID"),
        "interpretPath": payload.get("_path"),
        "artifact": {
            "reference": (payload.get("artifact") or {}).get("reference"),
            "name": (payload.get("artifact") or {}).get("name"),
            "tensorSHA256": (payload.get("artifact") or {}).get("tensorSHA256"),
        },
        "condition": payload.get("condition")
        or condition_of((payload.get("optvec") or {}).get("sidecarBlock")),
        "layer": (payload.get("optvec") or {}).get("layer"),
        "topLibraryMatch": (None if top is None else {
            "reference": top.get("reference"),
            "name": top.get("name"),
            "concept": top.get("concept"),
            "cosine": top.get("cosine"),
            "absCosine": abs(float(top["cosine"])),
            "nullPercentile": percentile,
            "clearsNull": matched}),
        "libraryMatchCategory": (top.get("reference") if matched
                                 else NO_LIBRARY_MATCH),
        "libraryComparedCount": library_compared_count(payload),
        "topSAEFeatures": ([f.get("feature") for f in sae_top[:5]]
                           if sae_top else None),
        "saeConcentration": (concentration([f.get("cosine") for f in sae_top
                                            if f.get("cosine") is not None])
                             if sae_top else None),
        "logitLensTopTokens": [row.get("piece")
                               for row in promoted[:CONCENTRATION_TOP_N]],
        "logitLensConcentration": (
            concentration([row["logit"] for row in promoted
                           if row.get("logit") is not None])
            if promoted else None),
    }


def _distribution(rows: list[dict]) -> list[dict]:
    """How many distinct concepts does this family match? — with
    ``no-library-match`` as an ORDINARY row of the same table (plan §1: that
    solutions match nothing in the library is the alien finding, a result)."""
    buckets: dict[str, list[str]] = {}
    for row in rows:
        buckets.setdefault(row["libraryMatchCategory"], []).append(
            row.get("runID") or row["artifact"].get("name") or "?")
    out = [{"match": key, "count": len(value), "solutions": value,
            "isNoMatchCategory": key == NO_LIBRARY_MATCH}
           for key, value in buckets.items()]
    out.sort(key=lambda entry: (-entry["count"], entry["match"]))
    return out


def _group_stats(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "meanAbsTopLibraryCosine": _mean(
            [row["topLibraryMatch"]["absCosine"] for row in rows
             if row["topLibraryMatch"] is not None]),
        "meanLogitLensConcentration": _mean(
            [row["logitLensConcentration"] for row in rows]),
        "meanSAEConcentration": _mean([row["saeConcentration"]
                                       for row in rows]),
        "noLibraryMatchCount": sum(
            1 for row in rows
            if row["libraryMatchCategory"] == NO_LIBRARY_MATCH),
    }


def s1_s2_contrast(rows: list[dict]) -> dict:
    """The plan's primary interpretability contrast (§1, §7): preservation
    constraints (S2) vs. none (S1). Numbers and labels only — whether a
    difference here means "more interpretable" is a claim, and this artifact
    makes none."""
    s1 = [row for row in rows if row["condition"] == "s1"]
    s2 = [row for row in rows if row["condition"] == "s2"]
    if not s1 or not s2:
        present = sorted({row["condition"] for row in rows})
        return {"skipped": "the contrast needs both s1 and s2 solutions "
                           f"(present: {', '.join(present) or 'none'})"}
    stats1, stats2 = _group_stats(s1), _group_stats(s2)
    deltas = {}
    for key in ("meanAbsTopLibraryCosine", "meanLogitLensConcentration",
                "meanSAEConcentration"):
        a, b = stats1[key], stats2[key]
        deltas[key] = None if a is None or b is None else float(a - b)
    return {"s1": stats1, "s2": stats2, "deltaS1MinusS2": deltas,
            "concentrationFormula": CONCENTRATION_FORMULA}


def family_summary(config: FamilySummaryConfig, *, root: str | None = None,
                   log: Callable[[str], None] | None = None) -> dict:
    """Summarize N interpretations as one family table and write its own
    immutable run directory."""
    payloads = [load_interpret(reference, root)
                for reference in config.interpret_runs]
    rows = [_solution_row(payload,
                          match_percentile=config.match_null_percentile)
            for payload in payloads]
    distribution = _distribution(rows)
    conditions: dict[str, int] = {}
    for row in rows:
        conditions[row["condition"]] = conditions.get(row["condition"], 0) + 1

    report = {
        "schemaVersion": SCHEMA_VERSION,
        "runType": FAMILY_RUN_TYPE,
        "claim": CLAIM,
        "claimNote": CLAIM_NOTE,
        "count": len(rows),
        "matchRule": (
            "a library cosine counts as a MATCH when its |cosine| percentile "
            f"in the run's own random-direction null is >= "
            f"{config.match_null_percentile}; otherwise the solution falls in "
            f"the '{NO_LIBRARY_MATCH}' category, which is a RESULT (plan §1: "
            "solutions matching nothing in the library is the alien finding), "
            "not a missing value. A solution whose interpretation compared "
            "against ZERO library vectors also lands there and is counted in "
            "noLibraryComparedCount — read that first, since 'was asked about "
            "nothing' is not evidence of alienness"),
        "matchNullPercentile": config.match_null_percentile,
        "concentrationFormula": CONCENTRATION_FORMULA,
        "solutions": rows,
        "libraryMatchDistribution": distribution,
        "distinctLibraryMatches": sum(
            1 for entry in distribution if not entry["isNoMatchCategory"]),
        "noLibraryMatchCount": sum(
            entry["count"] for entry in distribution
            if entry["isNoMatchCategory"]),
        "noLibraryComparedCount": sum(
            1 for row in rows if row["libraryComparedCount"] == 0),
        "conditions": conditions,
        "contrastS1S2": s1_s2_contrast(rows),
        "sources": [payload.get("_path") for payload in payloads],
    }

    name = (config.name or "").strip() or f"{len(rows)}-solutions"
    run_directory = paths.make_unique_run_directory(
        f"optvec-family-{name}", root)
    report["runID"] = os.path.basename(os.path.normpath(run_directory))
    with open(os.path.join(run_directory, FAMILY_JSON), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    model_ids = {(payload.get("model") or {}).get("artifactModelID")
                 for payload in payloads}
    model_ids.discard(None)
    write_run_config(
        run_directory, FAMILY_RUN_TYPE,
        model_id=next(iter(model_ids)) if len(model_ids) == 1 else None,
        notes={"claim": CLAIM, "count": len(rows),
               "conditions": conditions,
               "distinctLibraryMatches": report["distinctLibraryMatches"],
               "noLibraryMatchCount": report["noLibraryMatchCount"],
               "noLibraryComparedCount": report["noLibraryComparedCount"],
               "matchNullPercentile": config.match_null_percentile,
               "sources": report["sources"],
               "modelIDs": sorted(model_ids)})
    if log is not None:
        log(f"optvec family → {run_directory}")
    return {"runDirectory": run_directory, **report}


__all__ = [
    "CLAIM", "CONCENTRATION_FORMULA", "DEFAULT_ALPHA_MULTIPLES",
    "DEFAULT_MATCH_NULL_PERCENTILE", "FAMILY_JSON", "FAMILY_RUN_TYPE",
    "GENERATIONS", "INTERPRET_JSON", "NO_LIBRARY_MATCH", "RUN_TYPE",
    "SUGGESTIVE_STAMP", "FamilySummaryConfig", "LibraryConfig",
    "LogitLensConfig", "OptVecFamilyError", "OptVecInterpretConfig",
    "OptVecInterpretConfigError", "OptVecInterpretDataError", "SAEConfig",
    "SelfExplanationConfig", "concentration", "condition_of",
    "family_summary", "interpret", "jlens_stage", "library_compared_count",
    "load_config",
    "load_interpret", "logit_lens_stage", "probe_generations",
    "sae_stage", "self_explanation_stage", "s1_s2_contrast",
    "stage_skip_reason", "top_library_match",
]
