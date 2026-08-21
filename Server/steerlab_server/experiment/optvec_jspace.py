"""OptVec propagated J-space decomposition — what the model COMPUTES from a
vector, beyond what the vector contains (plan of record:
``docs/OPTVEC-OPTIMIZED-INJECTION-VECTORS-PLAN.md``, WP8).

The question WP7's static readouts cannot ask: an injected direction is one
thing at the layer it enters, and something else by the time the network has
run twelve more blocks on it. The J-lens is a LINEAR map ``J_ℓ`` from a
source-layer residual into the final pre-normalization basis, so the steered−
baseline delta at any observation layer ``ℓ ≥ L`` decomposes exactly::

    J_ℓ(h_steered − h_base) = J_ℓ(α·v) + J_ℓ(Δ_computed)
                              └ direct ┘   └── emergent ──┘

The **direct** term is statically computable (it is the vector itself, dosed);
subtracting it isolates the **emergent** term, which is the quantity of
interest: two vectors with similar direct terms and different emergent terms
are doing different things downstream.

Teacher-forced pairing, and why nothing else works
--------------------------------------------------
Baseline and steered passes must run over IDENTICAL token sequences, or
per-position pairing is meaningless — free generation diverges after the first
steered token and position *p* of the two runs is no longer the same place in
the same sentence. So this verb runs forced-choice probe items teacher-forced:
one full-sequence pass per condition over the item's rendered PROMPT, with the
residual read at the item's ANSWER POSITION (the position whose next-token
logits are the answer distribution). Injection is
:class:`~steerlab_server.steering.trainable_injector.TrainableVectorInjector`
in ``position_mode="from_response"`` under ``torch.no_grad`` — the same
position-exact, decode-identical argument the training driver documents
(attention is causal, so injecting at position *p* of layer *L*'s output in one
full pass produces exactly the state a stepped decode writes into the KV cache
at that position).

Energy is never reported alone (hard rule, mirrored from ``jlens/decompose.py``)
-------------------------------------------------------------------------------
A transported delta's L2 energy is uninterpretable on its own: the same
pipeline run with a matched-norm RANDOM direction also moves the residual
stream, also transports through ``J_ℓ``, and also produces an emergent term
(the network computes from noise too). So every energy in this readout carries
its matched-norm-random sibling and their ratio, and
:func:`validate_energy_pairing` refuses to write a report where any energy key
lacks its null — the schema cannot express one without the other.

Multiplicity: shallow vs deep (family mode)
-------------------------------------------
With ≥2 vectors at the same layer, the family is described twice per
observation layer: the pairwise cosine matrix and participation ratio of the
RAW directions, and the same two statistics over the mean PROPAGATED deltas.
Raw-scattered but propagated-collapsed is shallow multiplicity (one mechanism,
differently parameterized); separated in both is deep multiplicity. The verb
reports the two tables and makes no claim about which one the numbers show.

Tier (mandatory, plan §7 WP8 caveat)
------------------------------------
Every readout this module writes is stamped with the lens's own evidence tier
AND with its qualification state against the runtime that actually ran
(:func:`qualification_state`). Unqualified is the default and stays
exploratory, not citable evidence; a runtime with a passing ``jlens qualify``
record is stamped with that record's id instead. The stamp is written by the
engine, not by the researcher remembering.

Gemma-only / server-only by rule (CLAUDE.md, hard requirement): imported lens
artifacts are PyTorch/HF-native and activations do not transfer across
substrates. The gate is mechanical — the lens must resolve, and its fit model
must be the running model — because lenses exist only for the Gemma models in
``jlens/importer.py``'s supported table.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable

import torch

from ..steering import vector_math as vm
from ..steering.intervention import LayerIntervention
from ..steering.trainable_injector import TrainableVectorInjector
from . import paths, prompt_render
from .optvec_eval import (EvalItem, FileRef, OptVecArtifactError, OptVecTarget,
                          load_artifact, load_dataset, prepare_items,
                          require_optvec, verify_model)
from .optvec_geometry import cosine_matrix, participation_ratio
from .optvec_train import _batch_tensors, _logits_keep_kwargs
from .run_config import write_run_config

#: Run type stamped into ``config.json``. Its own immutable run directory: the
#: training and eval runs are finished records, and a J-space readout is a
#: separate measurement that may be repeated against a different lens.
RUN_TYPE = "optvec-jspace"

JSPACE_JSON = "jspace.json"
JSPACE_RECORDS = "jspace-records.jsonl"

SCHEMA_VERSION = 1

#: Fixed default seed for the matched-norm random null. Config-overridable and
#: always recorded, so the null is reproducible from the artifact alone.
NULL_SEED = 20260810

#: The null's distribution, stamped so a reader never has to infer it. Same
#: family and same stamp as the study path's random control
#: (``tasks.RANDOM_VECTOR_ALGORITHM``); a test pins the two strings equal.
NULL_ALGORITHM = "gaussian-isotropic-v1"

#: Spacing of the default observation-layer ladder above the injection layer.
DEFAULT_LAYER_STRIDE = 4

#: Default top-k width for the token readouts.
DEFAULT_TOP_K = 10

#: Mandatory on every readout this module writes (plan §7 WP8). Verbatim — a
#: test asserts the exact string, because the whole point is that it cannot be
#: softened by paraphrase later.
#:
#: Reworded 2026-08-15, when Stage 4 (``jlens qualify``) landed: the stamp used
#: to say qualification was UNIMPLEMENTED, which was true and is now false, and
#: a stamp that misdescribes why a readout is uncitable is worse than none. The
#: default is still "unqualified", because most runtimes are — but it is now a
#: statement about THIS runtime with a remedy attached, not about the engine.
QUALIFICATION_STAMP = ("unqualified for this runtime — no passing jlens "
                       "qualification (steerlab-server jlens qualify); "
                       "exploratory analysis, not citable evidence")

#: How :func:`observation_layers` picks its default. Stated in the artifact as
#: well as here so a reader never has to guess which layers were read and why.
LAYER_RULE = (
    "default: the injection layer L, then a ladder L+4, L+8, … while the layer "
    "is a fitted source layer of the lens, plus the lens's deepest fitted "
    "source layer if the stride missed it. Explicit observationLayers must all "
    "be >= L (transport below the injection sees no delta by construction) and "
    "must all be fitted source layers of the lens (the target layer has no "
    "Jacobian).")

#: Energy is the squared L2 norm of the transported vector. Named in the
#: artifact because "energy" is otherwise ambiguous between ‖x‖ and ‖x‖².
ENERGY_DEFINITION = "energy = ||J_l(x)||_2^2 (squared L2 of the transported vector)"

OBSERVATION_CONVENTION = "postInterventionBlockOutput at the item's answer position"


class OptVecJSpaceConfigError(ValueError):
    """A J-space config that cannot be run as written."""


class OptVecJSpaceError(RuntimeError):
    """A J-space readout that cannot be produced (lens, capture, schema)."""


# ------------------------------------------------------------------- config


_CONFIG_KEYS = {
    "vectorArtifacts": "vector_artifacts",
    "lensID": "lens_id",
    "observationLayers": "observation_layers",
    "alphaMultiple": "alpha_multiple",
    "nullSeed": "null_seed",
    "seed": "seed",
    "topK": "top_k",
    "microbatchSize": "microbatch_size",
    "modelID": "model_id", "revision": "revision", "name": "name",
    "device": "device", "dtype": "dtype",
    "promptMode": "prompt_mode", "systemPrompt": "system_prompt",
    "qwenThinkingEnabled": "qwen_thinking_enabled",
}


@dataclass
class OptVecJSpaceConfig:
    """Strict, closed-key configuration. Unknown keys refuse rather than being
    ignored: a typo'd ``observationLayer`` that silently fell back to the
    default ladder would produce a readout answering a different question."""

    vector_artifacts: list[str]
    lens_id: str
    probe_items: FileRef
    observation_layers: list[int] | None = None
    alpha_multiple: float = 1.0
    null_seed: int = NULL_SEED
    seed: int = 0
    top_k: int = DEFAULT_TOP_K
    microbatch_size: int = 4
    model_id: str | None = None
    revision: str | None = None
    name: str | None = None
    device: str | None = None
    dtype: str = "auto"
    prompt_mode: str = prompt_render.CHAT_ASSISTANT
    system_prompt: str | None = None
    qwen_thinking_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.vector_artifacts, list) or not all(
                isinstance(a, str) and a.strip() for a in self.vector_artifacts):
            raise OptVecJSpaceConfigError(
                "'vectorArtifacts' must be a list of extension-less artifact "
                "paths")
        self.vector_artifacts = [a.strip() for a in self.vector_artifacts]
        if not self.vector_artifacts:
            raise OptVecJSpaceConfigError(
                "'vectorArtifacts' needs at least one artifact — with two or "
                "more the run additionally reports the family tables")
        if len(set(self.vector_artifacts)) != len(self.vector_artifacts):
            raise OptVecJSpaceConfigError(
                "'vectorArtifacts' names the same artifact twice — a family "
                "cosine matrix with a duplicated row is not a statistic")
        if not (self.lens_id or "").strip():
            raise OptVecJSpaceConfigError(
                "'lensID' is required — the propagated readout IS the lens, "
                "and there is no default one")
        self.lens_id = self.lens_id.strip()
        if not isinstance(self.probe_items, FileRef):
            raise OptVecJSpaceConfigError(
                "'probeItems' must be an object with 'path' and 'sha256'")
        if self.observation_layers is not None:
            if not isinstance(self.observation_layers, list) or not all(
                    isinstance(x, int) and not isinstance(x, bool)
                    for x in self.observation_layers):
                raise OptVecJSpaceConfigError(
                    "'observationLayers' must be a list of integers")
            if not self.observation_layers:
                raise OptVecJSpaceConfigError(
                    "'observationLayers' is empty — omit it for the default "
                    "ladder rather than declaring no readout")
            self.observation_layers = sorted(set(int(x) for x in
                                                 self.observation_layers))
        self.alpha_multiple = float(self.alpha_multiple)
        if not self.alpha_multiple > 0:
            raise OptVecJSpaceConfigError(
                "alphaMultiple must be positive — at 0 the steered pass IS the "
                "baseline and every delta is exactly zero")
        if self.microbatch_size < 1:
            raise OptVecJSpaceConfigError("microbatchSize must be at least 1")
        if self.top_k < 1:
            raise OptVecJSpaceConfigError(
                "topK must be at least 1 — the token readout is what makes a "
                "transported delta readable")
        self.null_seed = int(self.null_seed)
        self.seed = int(self.seed)

    @classmethod
    def from_dict(cls, payload: dict) -> "OptVecJSpaceConfig":
        if not isinstance(payload, dict):
            raise OptVecJSpaceConfigError(
                "the OptVec J-space config must be a JSON object")
        unknown = sorted(set(payload) - set(_CONFIG_KEYS) - {"probeItems"})
        if unknown:
            raise OptVecJSpaceConfigError(
                "unknown OptVec J-space config key(s): " + ", ".join(unknown))
        if "probeItems" not in payload:
            raise OptVecJSpaceConfigError(
                "'probeItems' is required — a hashed choice-row JSONL; the "
                "readout is taken at each item's answer position")
        try:
            probe = FileRef.from_dict(payload["probeItems"], "probeItems")
        except ValueError as exc:
            raise OptVecJSpaceConfigError(str(exc)) from exc
        kwargs = {field_name: payload[key]
                  for key, field_name in _CONFIG_KEYS.items() if key in payload}
        for required in ("vector_artifacts", "lens_id"):
            if required not in kwargs:
                raise OptVecJSpaceConfigError(
                    f"'{'vectorArtifacts' if required == 'vector_artifacts' else 'lensID'}'"
                    " is required")
        return cls(probe_items=probe, **kwargs)

    def to_dict(self) -> dict:
        out = {json_key: getattr(self, field_name)
               for json_key, field_name in _CONFIG_KEYS.items()}
        out["vectorArtifacts"] = list(self.vector_artifacts)
        out["probeItems"] = self.probe_items.to_dict()
        if self.observation_layers is not None:
            out["observationLayers"] = list(self.observation_layers)
        return out


def load_config(path: str) -> OptVecJSpaceConfig:
    with open(path, encoding="utf-8") as handle:
        return OptVecJSpaceConfig.from_dict(json.load(handle))


# --------------------------------------------------------------------- lens


def resolve_lens(lens_id: str, root: str | None = None):
    """The imported lens record, or a refusal that names BOTH acquisition steps.

    Acquisition (bytes into the HF cache) and import (conversion + record into
    the workspace) are separate operations, and a researcher who has done
    neither cannot tell from "no imported lens" which one they are missing.
    """
    from ..jlens import lens_store
    from ..jlens.schemas import JLensError

    try:
        return lens_store.resolve(lens_id, root)
    except JLensError as exc:
        raise OptVecJSpaceError(
            f"J-space readout needs an imported lens, and '{lens_id}' is not "
            f"in this workspace's lens library ({exc}). Acquire the published "
            f"bytes first (steerlab-server jlens acquire <model>), then import "
            f"them (steerlab-server jlens import <model>) — acquisition and "
            f"import are separate steps and this is the second one."
        ) from exc


def evidence_tier_for(record) -> str:
    """The lens's own tier, from the supported-lens table keyed by the model it
    was FITTED on.

    The tier is this project's scope decision about where evidence comes from
    (27B is 'evidence', smaller Gemmas are 'testing'), so it belongs to the
    lens, not to the run asking for it. A lens whose fit model is not in the
    table — a hand-built record, a future row — is stamped ``unknown`` rather
    than defaulted into a tier it never earned.
    """
    from ..jlens.importer import SUPPORTED

    entry = SUPPORTED.get(record.fit.modelID or "")
    return (entry or {}).get("tier") or "unknown"


def qualification_state(record, model) -> tuple[str, str]:
    """``(stamp, claim)`` for a lens against the runtime actually loaded.

    Resolved from the record's own qualifications by exact runtime key
    (model + revision + dtype + quantization), which is the same match the
    freeze gate makes. **Absent is not a match**: a runtime whose revision or
    dtype could not be established gets the unqualified stamp, never an
    inherited one, because a qualification is only about the numerics it was
    measured against.
    """
    model_id = getattr(model, "model_id", None)
    revision = getattr(model, "revision", None)
    dtype = getattr(model, "dtype", None)
    if not (model_id and revision and dtype):
        return QUALIFICATION_STAMP, "exploratory"
    qualification = record.qualification_for(model_id, revision, dtype, None)
    if qualification is None:
        return QUALIFICATION_STAMP, "exploratory"
    return (f"qualified — {qualification.qualificationID} for "
            f"{model_id}@{revision[:12]}…/{dtype} "
            f"({qualification.qualifiedAt})", "qualified")


def observation_layers(layer: int, record,
                       declared: list[int] | None = None) -> list[int]:
    """Which ``ℓ`` the propagated delta is read at. See :data:`LAYER_RULE`.

    Below ``L`` there is nothing to read — the injection is downstream of those
    blocks, so every delta is exactly zero and the report would carry rows of
    structural noise. Above the lens's fitted range there is no ``J_ℓ`` at all.
    """
    fitted = sorted(int(x) for x in record.sourceLayers)
    if not fitted:
        raise OptVecJSpaceError(
            f"lens '{record.lensID}' declares no fitted source layers")
    if layer not in fitted:
        raise OptVecJSpaceError(
            f"the vectors' injection layer {layer} is not a fitted source "
            f"layer of lens '{record.lensID}' (have {fitted[0]}..{fitted[-1]}; "
            f"the target layer {record.targetLayer} has no Jacobian by "
            f"construction) — the direct term could not be transported there")
    if declared is not None:
        below = [x for x in declared if x < layer]
        if below:
            raise OptVecJSpaceError(
                f"observationLayers {below} are below the injection layer "
                f"{layer} — the delta there is exactly zero by construction, "
                f"so those rows would measure nothing")
        missing = [x for x in declared if x not in fitted]
        if missing:
            raise OptVecJSpaceError(
                f"observationLayers {missing} are not fitted source layers of "
                f"'{record.lensID}' (have {fitted[0]}..{fitted[-1]}; the "
                f"target layer {record.targetLayer} has no Jacobian by "
                f"construction)")
        return sorted(set(int(x) for x in declared))
    deepest = fitted[-1]
    ladder = [x for x in range(layer, deepest + 1, DEFAULT_LAYER_STRIDE)
              if x in fitted]
    if deepest not in ladder:
        ladder.append(deepest)
    return sorted(set(ladder))


# ------------------------------------------------------------------ capture


@dataclass(frozen=True)
class ProbeItem:
    """One probe item, rendered and tokenized.

    Field names match the training driver's ``PreparedItem`` on purpose: the
    right-padded batch builder (and its assertion that no answer position lands
    on a pad) is reused verbatim rather than reimplemented.
    """

    id: str
    prompt: str
    prompt_text: str
    input_ids: tuple[int, ...]
    options: tuple[str, ...]
    target: str

    @property
    def answer_position(self) -> int:
        """Index whose next-token logits are the answer distribution — the
        position the residual is read at, and the position injected."""
        return len(self.input_ids) - 1


def prepare_probe_items(model, items: list[EvalItem],
                        config: OptVecJSpaceConfig) -> list[ProbeItem]:
    """Render + tokenize eval items for the teacher-forced passes.

    Rendering goes through :mod:`prompt_render` with the same arguments every
    behavioral study uses, so a probe prompt is byte-identical to the prompt
    the same row would produce in an eval run. Unlike training, options are
    never tokenized here: this readout scores no options — it reads the
    residual — so multi-token options are perfectly legal probes.
    """
    prepared: list[ProbeItem] = []
    for item in items:
        rendered = prompt_render.render(
            model.tokenizer, item.prompt, model_id=model.model_id,
            prompt_mode=config.prompt_mode, system_prompt=config.system_prompt,
            qwen_thinking_enabled=config.qwen_thinking_enabled)
        if len(rendered.input_ids) < 1:
            raise OptVecJSpaceError(
                f"probe item '{item.id}' rendered to zero tokens — there is no "
                "answer position to read")
        prepared.append(ProbeItem(
            id=item.id, prompt=item.prompt, prompt_text=rendered.text,
            input_ids=tuple(rendered.input_ids), options=tuple(item.options),
            target=item.target))
    return prepared


class AnswerPositionCapture(LayerIntervention):
    """Read-only capture of the residual at each item's answer position.

    Returns the hidden state unchanged and never mutates in place, so arming it
    cannot alter what the model computes — the same contract
    ``JLensReadoutRecorder`` holds. Armed AFTER the injector, so at the
    injection layer it observes the POST-intervention residual (the convention
    every other observer in this engine uses), which is what makes the delta at
    ``ℓ = L`` exactly ``α·v``.

    Rows are cast to float32 and moved to CPU inside the pass: a bf16 residual
    would round the subtraction that isolates the emergent term, and keeping
    device tensors alive across microbatches is how capture paths OOM.
    """

    def __init__(self, layers):
        self.layers = {int(x) for x in layers}
        self.positions: torch.Tensor | None = None
        self.rows: dict[int, torch.Tensor] = {}

    def set_batch(self, positions: torch.Tensor) -> None:
        self.positions = positions.reshape(-1).long()
        self.rows = {}

    def clear_batch(self) -> None:
        self.positions = None

    def apply(self, h: torch.Tensor, layer: int, offset: int) -> torch.Tensor:
        if layer not in self.layers:
            return h
        if self.positions is None:
            raise OptVecJSpaceError(
                "capture was armed without answer positions — call set_batch() "
                "before the forward pass")
        if h.dim() != 3:
            raise OptVecJSpaceError(
                f"expected a [batch, seq, hidden] residual stream, got "
                f"{tuple(h.shape)}")
        if layer in self.rows:
            # One full-sequence pass per batch is the whole design (no KV
            # cache, no chunked prefill). A second firing would mean the driver
            # is doing something else, and the second write would silently win.
            raise OptVecJSpaceError(
                f"layer {layer} fired twice in one capture pass — this driver "
                "runs exactly one full-sequence pass per microbatch")
        batch = h.shape[0]
        positions = self.positions.to(h.device)
        if positions.numel() != batch:
            raise OptVecJSpaceError(
                f"{positions.numel()} answer positions for a batch of {batch}")
        picked = h[torch.arange(batch, device=h.device), positions]
        self.rows[layer] = picked.detach().to(torch.float32).cpu()
        return h


@torch.no_grad()
def capture_answer_residuals(model, items: list[ProbeItem], *,
                             layers: list[int],
                             injector: TrainableVectorInjector | None = None,
                             microbatch_size: int = 4) -> dict:
    """``{item id: {layer: float32 CPU row}}`` for one condition.

    Teacher-forced: the token sequence is the item's rendered prompt and is
    IDENTICAL across conditions by construction, which is what makes the
    baseline/steered pairing per position meaningful at all.
    """
    captured: dict[str, dict[int, torch.Tensor]] = {}
    for start in range(0, len(items), microbatch_size):
        batch = items[start:start + microbatch_size]
        input_ids, attention_mask, answer_positions = _batch_tensors(model, batch)
        capture = AnswerPositionCapture(layers)
        capture.set_batch(answer_positions)
        interventions: list[LayerIntervention] = []
        if injector is not None:
            injector.set_batch(answer_positions=answer_positions,
                               attention_mask=attention_mask)
            interventions.append(injector)
        # The observer runs LAST, so at the injection layer it sees the
        # post-intervention residual.
        interventions.append(capture)
        # Only the residual stream is read here; the vocabulary projection is
        # pure cost at 27B, so ask for the minimum the model will honor.
        kwargs = _logits_keep_kwargs(model.model, 1)
        try:
            model.hooked.reset_offsets()
            with model.hooked.session(interventions):
                model.model(input_ids=input_ids, attention_mask=attention_mask,
                            use_cache=False, **kwargs)
        finally:
            if injector is not None:
                injector.clear_batch()
            capture.clear_batch()
        missing = [layer for layer in layers if layer not in capture.rows]
        if missing:
            raise OptVecJSpaceError(
                f"observation layer(s) {missing} never fired — the model has "
                f"{getattr(model, 'num_layers', '?')} blocks and the lens "
                "names layers it does not have")
        for index, item in enumerate(batch):
            captured[item.id] = {layer: capture.rows[layer][index].clone()
                                 for layer in layers}
    return captured


def injector_for(model, layer: int, direction: list[float],
                 norm: float) -> TrainableVectorInjector:
    """A position-exact injector carrying exactly ``norm · direction/‖direction‖``.

    The constructor projects ``u`` onto the sphere of radius
    ``alpha_absolute``, so passing the row as ``u`` with the desired norm
    reproduces the dosed vector exactly — and makes the matched-norm null
    matched by construction rather than by arithmetic that could drift.
    """
    return TrainableVectorInjector(
        layer=layer, hidden_size=len(direction), alpha_absolute=float(norm),
        position_mode="from_response",
        u=torch.tensor(direction, dtype=torch.float32),
        device=getattr(model, "device", None))


# ------------------------------------------------------------------ readout


def _transport(readout, vector: torch.Tensor, layer: int) -> torch.Tensor:
    """``J_ℓ x`` through the existing lens helper (never a re-derived g-fold)."""
    return readout.transported(
        vector.to(device=readout.device, dtype=readout.dtype), layer)


def energy(transported: torch.Tensor) -> float:
    """:data:`ENERGY_DEFINITION`."""
    return float(transported.dot(transported))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    """Cosine, or ``None`` when either side is the zero vector — at ``ℓ = L``
    the emergent term IS zero, and 0.0 would read as 'orthogonal' rather than
    'undefined'."""
    na, nb = float(a.norm()), float(b.norm())
    if na == 0.0 or nb == 0.0:
        return None
    return float(a.dot(b) / (na * nb))


def _ratio(value: float, null_value: float) -> float | None:
    if null_value <= 0.0:
        return None
    return float(value / null_value)


def _mean(values: list[float]) -> float | None:
    live = [v for v in values if v is not None]
    return float(sum(live) / len(live)) if live else None


def _mean_energy(values: list[float]) -> float:
    """Mean of a per-item energy list. The probe set is non-empty by the time
    this runs (``analyze`` refuses an empty one), so an energy is always a
    number — never a null that a consumer would have to special-case."""
    if not values:
        raise OptVecJSpaceError(
            "no probe items contributed an energy — an empty readout is not a "
            "measurement")
    return float(sum(values) / len(values))


def energy_block(name: str, value: float, null_value: float) -> dict:
    """One energy and its matched-norm-random sibling.

    ``null_value`` is a REQUIRED argument for the same reason
    ``jlens/decompose._layer_report`` requires ``nullEnergyFraction``: the
    schema must have no way to express an energy without its control.
    """
    return {f"{name}Energy": float(value),
            f"{name}EnergyNull": float(null_value),
            f"{name}EnergyNullRatio": _ratio(float(value), float(null_value))}


def validate_energy_pairing(payload) -> None:
    """Refuse any structure carrying an energy without its null and ratio.

    Applied to the whole report (and every per-item record) immediately before
    writing, so the rule is enforced by the writer rather than by whoever adds
    the next metric remembering it.
    """
    if isinstance(payload, list):
        for entry in payload:
            validate_energy_pairing(entry)
        return
    if not isinstance(payload, dict):
        return
    for key in payload:
        if key.endswith("Energy"):
            for suffix in ("Null", "NullRatio"):
                if key + suffix not in payload:
                    raise OptVecJSpaceError(
                        f"'{key}' is reported without '{key}{suffix}' — energy "
                        "is never reportable without its matched-norm-random "
                        "null (jlens/decompose.py's rule, mirrored here)")
    for value in payload.values():
        validate_energy_pairing(value)


def _pieces(tokenizer, ids: list[int]) -> list:
    """Token pieces for the top-k table. A display field: a tokenizer without
    ``convert_ids_to_tokens`` records nulls rather than failing the readout."""
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is None:
        return [None] * len(ids)
    try:
        return list(convert(ids))
    except Exception:  # noqa: BLE001 — a display field never fails a readout
        return [None] * len(ids)


@torch.no_grad()
def _topk_table(readout, tokenizer, vector: torch.Tensor, layer: int,
                k: int) -> list[dict]:
    """Top-k tokens of a RESIDUAL-space vector under the canonical readout.

    The vector is handed in untransported: :meth:`LensReadout.topk` applies
    ``J_ℓ``, the RMSNorm, the gain-folded head and the softcap itself, which is
    the canonical convention — re-deriving any of it here would be a second,
    divergent readout.
    """
    if float(vector.norm()) == 0.0:
        return []
    ids, values = readout.topk(
        vector.to(device=readout.device, dtype=readout.dtype), layer, k)
    id_list = [int(x) for x in ids.cpu()]
    return [{"tokenID": token, "piece": piece, "logit": float(value)}
            for token, piece, value in zip(id_list, _pieces(tokenizer, id_list),
                                           [float(v) for v in values.cpu()])]


@dataclass
class VectorPass:
    """One artifact's captured conditions plus the direct term it was dosed at."""

    target: OptVecTarget
    direct: torch.Tensor                    # α·v in residual space (float32 CPU)
    null_direct: torch.Tensor               # matched-norm random, same norm
    null_seed: int
    steered: dict = field(default_factory=dict)
    steered_null: dict = field(default_factory=dict)


@torch.no_grad()
def decompose_vector(readout, tokenizer, *, passes: VectorPass,
                     baseline: dict, items: list[ProbeItem],
                     layers: list[int], top_k: int) -> tuple[list[dict], list[dict], dict]:
    """The decomposition for one artifact.

    Returns ``(per-layer aggregates, per-item records, {layer: mean propagated
    delta})`` — the third is what the family tables cluster on.
    """
    aggregates: list[dict] = []
    records: list[dict] = []
    mean_propagated: dict[int, torch.Tensor] = {}
    alpha_absolute = float(passes.direct.norm())

    for layer in layers:
        direct_t = _transport(readout, passes.direct, layer)
        null_direct_t = _transport(readout, passes.null_direct, layer)
        direct_energy = energy(direct_t)
        null_direct_energy = energy(null_direct_t)

        delta_energies: list[float] = []
        emergent_energies: list[float] = []
        null_delta_energies: list[float] = []
        null_emergent_energies: list[float] = []
        cosines: list[float] = []
        null_cosines: list[float] = []
        linearity: list[float] = []
        delta_sum: torch.Tensor | None = None
        emergent_sum: torch.Tensor | None = None
        propagated_sum: torch.Tensor | None = None

        for item in items:
            base_row = baseline[item.id][layer]
            delta = passes.steered[item.id][layer] - base_row
            emergent = delta - passes.direct
            delta_t = _transport(readout, delta, layer)
            emergent_t = _transport(readout, emergent, layer)
            null_delta = passes.steered_null[item.id][layer] - base_row
            null_emergent = null_delta - passes.null_direct
            null_delta_t = _transport(readout, null_delta, layer)
            null_emergent_t = _transport(readout, null_emergent, layer)

            item_delta_energy = energy(delta_t)
            item_emergent_energy = energy(emergent_t)
            item_null_delta_energy = energy(null_delta_t)
            item_null_emergent_energy = energy(null_emergent_t)
            item_cosine = cosine(delta_t, direct_t)
            item_null_cosine = cosine(null_delta_t, null_direct_t)
            # The identity the whole decomposition rests on, measured rather
            # than asserted: J(delta) − [J(direct) + J(emergent)] is zero to
            # float tolerance because J is linear.
            residual = float((delta_t - (direct_t + emergent_t)).norm())

            delta_energies.append(item_delta_energy)
            emergent_energies.append(item_emergent_energy)
            null_delta_energies.append(item_null_delta_energy)
            null_emergent_energies.append(item_null_emergent_energy)
            if item_cosine is not None:
                cosines.append(item_cosine)
            if item_null_cosine is not None:
                null_cosines.append(item_null_cosine)
            linearity.append(residual)

            delta_sum = delta if delta_sum is None else delta_sum + delta
            emergent_sum = (emergent if emergent_sum is None
                            else emergent_sum + emergent)
            propagated_sum = (delta_t.detach().cpu() if propagated_sum is None
                              else propagated_sum + delta_t.detach().cpu())

            record = {
                "vectorReference": passes.target.artifact.reference,
                "itemID": item.id,
                "layer": layer,
                "injectionLayer": passes.target.layer,
                "alphaAbsolute": alpha_absolute,
                "nullSeed": passes.null_seed,
                **energy_block("delta", item_delta_energy,
                               item_null_delta_energy),
                **energy_block("direct", direct_energy, null_direct_energy),
                **energy_block("emergent", item_emergent_energy,
                               item_null_emergent_energy),
                "cosineDeltaDirect": item_cosine,
                "cosineDeltaDirectNull": item_null_cosine,
                "linearityResidualL2": residual,
            }
            records.append(record)

        count = max(1, len(items))
        mean_delta = delta_sum / count if delta_sum is not None else None
        mean_emergent = (emergent_sum / count if emergent_sum is not None
                         else None)
        mean_propagated[layer] = (propagated_sum / count
                                  if propagated_sum is not None
                                  else torch.zeros_like(direct_t.cpu()))

        aggregate = {
            "layer": layer,
            "itemCount": len(items),
            "isInjectionLayer": layer == passes.target.layer,
            **energy_block("meanDelta", _mean_energy(delta_energies),
                           _mean_energy(null_delta_energies)),
            **energy_block("direct", direct_energy, null_direct_energy),
            **energy_block("meanEmergent", _mean_energy(emergent_energies),
                           _mean_energy(null_emergent_energies)),
            "meanCosineDeltaDirect": _mean(cosines),
            "meanCosineDeltaDirectNull": _mean(null_cosines),
            "maxLinearityResidualL2": max(linearity) if linearity else 0.0,
            "topKDelta": ([] if mean_delta is None else
                          _topk_table(readout, tokenizer, mean_delta, layer,
                                      top_k)),
            "topKEmergent": ([] if mean_emergent is None else
                             _topk_table(readout, tokenizer, mean_emergent,
                                         layer, top_k)),
        }
        aggregates.append(aggregate)
    return aggregates, records, mean_propagated


# ------------------------------------------------------------------- family


def _stacked_statistics(rows: list[list[float]]) -> dict:
    """Cosine matrix + participation ratio, raw and unit-normalized.

    Both are reported for the same reason ``optvec_geometry`` reports both: the
    raw PR is dominated by the longest rows, and 'how many directions is this
    family?' is the unit-normalized one.
    """
    import numpy as np

    matrix = np.asarray(rows, dtype=np.float64)
    singular = [float(s) for s in np.linalg.svd(matrix, compute_uv=False)]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.where(norms == 0.0, 1.0, norms)
    singular_unit = [float(s) for s in np.linalg.svd(unit, compute_uv=False)]
    return {
        "cosineMatrix": cosine_matrix(rows),
        "participationRatio": participation_ratio(singular),
        "participationRatioUnitNormalized": participation_ratio(singular_unit),
        "singularValues": singular,
    }


def family_tables(entries: list[dict], raw_rows: list[list[float]],
                  propagated: dict[int, list[list[float]]],
                  *, layer: int) -> dict:
    """The shallow-vs-deep multiplicity table: the SAME two statistics over the
    raw directions and over the mean propagated deltas, per observation layer.

    No verdict is computed. Raw-scattered/propagated-collapsed and
    separated-in-both are readings of these numbers, and which one a family
    shows is a claim the researcher makes, not a field this engine fills in.
    """
    return {
        "layer": layer,
        "count": len(entries),
        "entries": entries,
        "raw": _stacked_statistics(raw_rows),
        "byObservationLayer": [
            {"layer": observation, **_stacked_statistics(rows)}
            for observation, rows in sorted(propagated.items())],
        "statistics": ("pairwise cosine matrix and participation ratio, "
                       "computed identically for the raw directions at layer "
                       "L and for the mean propagated J_l(delta) at each "
                       "observation layer"),
    }


# -------------------------------------------------------------------- driver


def _resolve_targets(references: list[str]) -> list[OptVecTarget]:
    targets = [require_optvec(load_artifact(reference))
               for reference in references]
    layers = {target.layer for target in targets}
    if len(layers) > 1:
        detail = "; ".join(f"{t.artifact.reference}: layer {t.layer}"
                           for t in targets)
        raise OptVecJSpaceError(
            "the artifacts name different optvec layers (" + detail + ") — "
            "one propagated decomposition is one injection layer read through "
            "one lens; run one per layer")
    widths = {target.artifact.vectors.hidden_size for target in targets}
    if len(widths) > 1:
        raise OptVecJSpaceError(
            f"the artifacts have different hidden sizes {sorted(widths)} — "
            "different models' bases")
    return targets


def _lens_compatibility(record, model) -> dict:
    """Refuse a lens that cannot be used against this runtime.

    Reuses ``lens_store.compatibility`` rather than restating its checks, so
    "wrong model" and "right model, not yet qualified" stay distinguishable —
    the second is the NORMAL state here (Stage 4 is unimplemented) and must not
    be treated as an error.
    """
    from ..jlens import lens_store

    report = lens_store.compatibility(
        record, model_id=getattr(model, "model_id", None),
        revision=getattr(model, "revision", None),
        dtype=getattr(model, "dtype", None),
        num_layers=int(getattr(model, "num_layers", 0) or 0),
        hidden_size=int(getattr(model, "hidden_size", 0) or 0))
    if not report["compatible"]:
        raise OptVecJSpaceError(
            f"lens '{record.lensID}' cannot be used against this runtime: "
            + "; ".join(report["problems"]))
    return report


def analyze(config: OptVecJSpaceConfig, *, model=None, root: str | None = None,
            log: Callable[[str], None] | None = None) -> dict:
    """Run the propagated J-space decomposition and write its own run directory."""
    def emit(message: str) -> None:
        if log is not None:
            log(message)

    targets = _resolve_targets(config.vector_artifacts)
    layer = targets[0].layer
    record = resolve_lens(config.lens_id, root)
    layers = observation_layers(layer, record, config.observation_layers)
    rows = load_dataset(config.probe_items, "probeItems")
    eval_items = prepare_items(rows, role="probe", dataset="probeItems")
    if not eval_items:
        raise OptVecJSpaceError(
            f"probe set '{config.probe_items.path}' parsed to zero items — "
            "the propagated decomposition is an average over a pinned probe "
            "set, and there is nothing to average")

    name = (config.name or "").strip() or targets[0].artifact.name
    run_directory = paths.make_unique_run_directory(f"optvec-jspace-{name}",
                                                    root)
    tier = evidence_tier_for(record)
    notes: dict = {
        "stage": "starting",
        "evidenceTier": tier,
        "qualification": QUALIFICATION_STAMP,
        "lensID": record.lensID,
        "injectionLayer": layer,
        "observationLayers": list(layers),
        "artifacts": [t.artifact.reference for t in targets],
        "probeItems": config.probe_items.to_dict(),
        "alphaMultiple": config.alpha_multiple,
        "nullSeed": config.null_seed,
    }
    dtype_name: str | None = None

    try:
        if model is None:
            from ..steering import model_loader
            device = model_loader.resolve_device(config.device)
            model_id = config.model_id or targets[0].artifact.sidecar.modelID
            emit(f"loading {model_id} on {device}")
            model = model_loader.load(
                model_id, config.revision or targets[0].artifact.sidecar.revision,
                dtype=config.dtype, device=device)
        dtype_name = getattr(model, "dtype", None)
        notes["model"] = {t.artifact.reference: verify_model(t, model)
                          for t in targets}
        notes["lens"] = _lens_compatibility(record, model)
        # Resolved only now, because it is a fact about the RUNTIME and the
        # runtime did not exist at run-directory creation. config.json is
        # written last (see the `finally`), so the notes carry the resolved
        # value rather than the placeholder the dict was seeded with.
        notes["qualification"], claim = qualification_state(record, model)

        from ..jlens.readout import LensReadout, ReadoutConfig
        from ..jlens.schemas import JLensError

        readout_config = ReadoutConfig(layers=list(layers), topK=config.top_k,
                                       topKLayers=list(layers),
                                       logitLensCompanion=False)
        try:
            readout = LensReadout.build(record=record, config=readout_config,
                                        model=model, root=root)
        except JLensError as exc:
            raise OptVecJSpaceError(
                f"could not arm lens '{record.lensID}': {exc}") from exc

        notes["stage"] = "capture"
        items = prepare_probe_items(model, eval_items, config)
        emit(f"capturing baseline over {len(items)} probe item(s) at layers "
             f"{layers}")
        baseline = capture_answer_residuals(
            model, items, layers=layers, injector=None,
            microbatch_size=config.microbatch_size)

        vector_reports: list[dict] = []
        records: list[dict] = []
        propagated: dict[int, list[list[float]]] = {ell: [] for ell in layers}
        raw_rows: list[list[float]] = []
        entries: list[dict] = []

        for index, target in enumerate(targets):
            direction = target.direction
            dosed_norm = config.alpha_multiple * vm.l2_norm(direction)
            # Independent draws per artifact (base seed + ordinal), so a family
            # of eight solutions is not compared against one shared null
            # direction that would make every null row identical.
            null_seed = config.null_seed + index
            null_direction = vm.random_vector(len(direction), dosed_norm,
                                              seed=null_seed)
            direct = torch.tensor(direction, dtype=torch.float32)
            direct = direct * (dosed_norm / float(direct.norm()))
            null_direct = torch.tensor(null_direction, dtype=torch.float32)

            emit(f"capturing steered + null for {target.artifact.reference}")
            steered = capture_answer_residuals(
                model, items, layers=layers,
                injector=injector_for(model, layer, direction, dosed_norm),
                microbatch_size=config.microbatch_size)
            steered_null = capture_answer_residuals(
                model, items, layers=layers,
                injector=injector_for(model, layer, null_direction, dosed_norm),
                microbatch_size=config.microbatch_size)

            passes = VectorPass(target=target, direct=direct,
                                null_direct=null_direct, null_seed=null_seed,
                                steered=steered, steered_null=steered_null)
            aggregates, item_records, mean_propagated = decompose_vector(
                readout, model.tokenizer, passes=passes, baseline=baseline,
                items=items, layers=layers, top_k=config.top_k)
            records.extend(item_records)
            vector_reports.append({
                "artifact": target.artifact.identity(),
                "optvec": {"layer": target.layer,
                           "alphaAbsolute": target.alpha_absolute},
                "alphaMultiple": config.alpha_multiple,
                "appliedNorm": dosed_norm,
                "null": {"seed": null_seed, "algorithm": NULL_ALGORITHM,
                         "matchedNorm": dosed_norm},
                "layers": aggregates,
            })
            raw_rows.append(list(direction))
            entries.append({
                "reference": target.artifact.reference,
                "name": target.artifact.name,
                "seed": (target.artifact.optvec or {}).get("seed"),
                "alphaAbsolute": target.alpha_absolute,
                "tensorSHA256": target.artifact.tensor_sha256})
            for ell in layers:
                propagated[ell].append(
                    [float(x) for x in mean_propagated[ell]])

        family = None
        if len(targets) >= 2:
            family = family_tables(entries, raw_rows, propagated, layer=layer)

        report = {
            "schemaVersion": SCHEMA_VERSION,
            "runType": RUN_TYPE,
            "runID": os.path.basename(os.path.normpath(run_directory)),
            # Mandatory, plan §7 WP8. Written by the engine on every readout,
            # resolved against the runtime that actually ran.
            "evidenceTier": tier,
            "qualification": notes["qualification"],
            "claim": claim,
            "lens": {
                "lensID": record.lensID,
                "fitModelID": record.fit.modelID,
                "fitRevisionKnown": record.fit.revisionKnown,
                "sourceLensSHA256": record.source.tensorSHA256,
                "sourceLayers": [record.sourceLayers[0],
                                 record.sourceLayers[-1]],
                "targetLayer": record.targetLayer,
                "readoutConvention": record.readoutConvention,
                "directionConvention": record.directionConvention,
                "compatibility": notes["lens"],
            },
            "model": {"modelID": getattr(model, "model_id", None),
                      "revision": getattr(model, "revision", None),
                      "dtype": dtype_name,
                      "device": str(getattr(model, "device", "")),
                      "perArtifact": notes["model"]},
            "instrument": {
                "pairing": "teacher-forced paired passes over identical token "
                           "sequences (baseline vs steered vs matched-norm "
                           "random), read at each item's answer position",
                "injection": "TrainableVectorInjector position_mode="
                             "'from_response' under torch.no_grad — position-"
                             "exact, decode-identical for a single answer "
                             "position",
                "observation": OBSERVATION_CONVENTION,
                "energy": ENERGY_DEFINITION,
                "decomposition": "J_l(h_steered - h_base) = J_l(alpha.v) + "
                                 "J_l(delta_computed); direct is the first "
                                 "term, emergent the second",
                "promptMode": config.prompt_mode,
                "qwenThinkingEnabled": config.qwen_thinking_enabled,
                "topK": config.top_k,
                "topKSource": "the MEAN residual delta (and mean emergent "
                              "term) over the probe set, transported and read "
                              "by LensReadout.topk — one table per layer, not "
                              "per item",
            },
            # Every pass here is a deterministic teacher-forced forward with no
            # sampling, so the run seed moves nothing. Recorded for provenance
            # and stamped inert rather than left to look causal (the same rule
            # local generation records carry).
            "seed": {"seed": config.seed, "seedInert": True,
                     "note": "no sampling occurs in this verb; the only RNG is "
                             "the null draw, which uses nullSeed"},
            "injectionLayer": layer,
            "observationLayers": list(layers),
            "observationLayerRule": LAYER_RULE,
            "probeItems": {**config.probe_items.to_dict(),
                           "itemCount": len(items)},
            "null": {"seedBase": config.null_seed,
                     "algorithm": NULL_ALGORITHM,
                     "note": "one independent matched-norm random direction "
                             "per artifact (seedBase + artifact ordinal), "
                             "pushed through the identical pipeline; every "
                             "energy is reported only beside its null"},
            "vectors": vector_reports,
            "family": family,
        }
        # The writer enforces the rule, not whoever adds the next metric.
        validate_energy_pairing(report)
        validate_energy_pairing(records)

        with open(os.path.join(run_directory, JSPACE_JSON), "w",
                  encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with open(os.path.join(run_directory, JSPACE_RECORDS), "w",
                  encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        notes["recordCount"] = len(records)
        notes["summary"] = _summary(vector_reports)
        notes["stage"] = "complete"
        emit(f"optvec jspace → {run_directory}")
        return {"runDirectory": run_directory, **report,
                "recordCount": len(records)}
    finally:
        # Written LAST so the notes are complete, and in `finally` so a crashed
        # readout still records the stage it died in (run directories are
        # immutable and write_run_config never rewrites).
        write_run_config(
            run_directory, RUN_TYPE,
            model_id=(config.model_id or targets[0].artifact.sidecar.modelID),
            revision=(getattr(model, "revision", None) or config.revision
                      or targets[0].artifact.sidecar.revision),
            dtype=dtype_name, notes=notes)


def _summary(vector_reports: list[dict]) -> list[dict]:
    """The compact per-(artifact, layer) table that lands in ``config.json``
    notes — enough to read the run's shape without opening ``jspace.json``.
    Energies keep their nulls here too."""
    out: list[dict] = []
    for report in vector_reports:
        for entry in report["layers"]:
            out.append({
                "reference": report["artifact"]["reference"],
                "layer": entry["layer"],
                "meanDeltaEnergy": entry["meanDeltaEnergy"],
                "meanDeltaEnergyNull": entry["meanDeltaEnergyNull"],
                "meanDeltaEnergyNullRatio": entry["meanDeltaEnergyNullRatio"],
                "meanEmergentEnergy": entry["meanEmergentEnergy"],
                "meanEmergentEnergyNull": entry["meanEmergentEnergyNull"],
                "meanEmergentEnergyNullRatio":
                    entry["meanEmergentEnergyNullRatio"],
                "meanCosineDeltaDirect": entry["meanCosineDeltaDirect"],
            })
    return out


__all__ = [
    "AnswerPositionCapture", "ENERGY_DEFINITION", "JSPACE_JSON",
    "JSPACE_RECORDS", "LAYER_RULE", "NULL_ALGORITHM", "NULL_SEED",
    "OptVecJSpaceConfig", "OptVecJSpaceConfigError", "OptVecJSpaceError",
    "OptVecArtifactError", "QUALIFICATION_STAMP", "RUN_TYPE", "ProbeItem",
    "VectorPass", "analyze", "capture_answer_residuals", "cosine",
    "decompose_vector", "energy", "energy_block", "evidence_tier_for",
    "family_tables", "injector_for", "load_config", "observation_layers",
    "prepare_probe_items", "qualification_state", "resolve_lens",
    "validate_energy_pairing",
]
