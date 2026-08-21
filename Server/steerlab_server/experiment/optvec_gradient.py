"""OptVec gradient survey — the α→0 fast path (WP-S4a).

Training (:mod:`optvec_train`) searches a sphere in R^d for a direction that
moves a whole item set. This module asks the cheapest possible version of the
same question, one item at a time:

    at α = 0, which way does an injection at this item's answer position have
    to point for the margin to rise fastest, and how far does the margin
    actually follow that direction before the linear reading breaks?

The answer is one forward+backward per item. The margin is the SAME quantity
the training loop's shift term hinges on — ``logp(target) − logp(contrast)``
over the restricted choice softmax, with the contrast fixed by the same rule
(the baseline's own selection, or the strongest competitor when the baseline
already picks the target) — so the gradient read here is the derivative of the
objective that would be optimized, not of a lookalike.

Claim grammar (binding, plan §1, and NARROWER than training's): what a
gradient direction certifies is **local sensitivity** — "at this item, at
α → 0, the margin's steepest-ascent direction is this". It is not a sufficiency
claim: nothing here shows the direction still works at a usable dose, on
another item, or without wrecking anchors and capability. The dose ladder is
what makes the gap visible rather than assumed; a minted direction carries the
ladder's linearity summary in its own sidecar and must still go through
backfill → eval → interpret like any other vector.

Deliberately reused, never re-implemented:

* item rendering/tokenization and the single-token option restriction
  (:func:`optvec_train.prepare_items`),
* the unsteered baseline through the deployed instrument
  (:func:`optvec_train.baseline_prepass`),
* the batched answer-position forward (:func:`optvec_train.answer_logit_rows`)
  and the position machinery it drives
  (:class:`~steerlab_server.steering.trainable_injector.PositionedDelta`),
* the α denominator (:func:`optvec_train.resolve_alpha`),
* the dose scoring path — ``logprob.score_options`` with a ``CellInjection``,
  which is exactly what ``optvec_eval`` and every behavioral study use, so a
  ladder margin is decode-identical to what a run would measure.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Callable

import torch

from ..steering import vector_store
from ..steering.trainable_injector import AdditiveDeltaProbe
from ..steering.vector_store import ConceptVectors, SteeringVectorSidecar
from . import logprob, optvec_train, paths, prompt_render
from .generate import CellInjection
from .optvec_train import (DatasetRef, OptVecConfigError, OptVecDataError,
                           PreparedItem)
from .run_config import write_run_config

#: Run type of the survey's immutable directory. ONE directory per survey, not
#: one per item: the survey is a single measurement over a set, and a
#: per-item run directory would multiply the record without adding a fact.
RUN_TYPE = "optvec-gradient"

#: Run type of a MINT — the directory that holds one exported direction as an
#: ordinary vector artifact. Separate from the survey run because run
#: directories are immutable: minting later must not write into a finished
#: record, and two mints of the same item must not collide.
MINT_RUN_TYPE = "optvec-gradient-mint"

#: ``optvec.method`` on a minted artifact. The value is what tells a reader
#: (and the interpret/eval verbs) that no optimization produced these bytes.
GRADIENT_METHOD = "gradientDirection"

#: Sidecar ``extractionMethod``. Deliberately the SAME as a trained OptVec
#: vector: the lifecycle (attach → freeze → run, norm backfill, eval,
#: interpret) keys off this value, and a gradient direction goes through the
#: identical lifecycle. What it is NOT is hidden in plain sight — ``optvec``
#: block's ``method``, ``claim`` and ``optimized`` fields say so.
EXTRACTION_METHOD = optvec_train.EXTRACTION_METHOD

#: ``stimulusSetHash`` prefix. Distinct from the training driver's ``optvec:``
#: composite so a minted direction can never be read as a vector trained on
#: those dataset splits.
STIMULUS_PREFIX = "optvec-gradient:"

#: Fractions of the resolved absolute α at which the linear reading is checked.
DEFAULT_DOSE_LADDER = (0.25, 0.5, 1.0, 1.5, 2.0)

GRADIENTS_FILE = "gradients.safetensors"
SURVEY_JSON = "gradient-survey.json"

SCHEMA_VERSION = 1


class OptVecGradientConfigError(ValueError):
    """A gradient-survey config that cannot be run as written."""


class OptVecGradientDataError(ValueError):
    """A dataset or survey artifact that cannot be read as declared."""


# ------------------------------------------------------------------- config


_CONFIG_KEYS = {
    "modelID": "model_id", "revision": "revision", "layer": "layer",
    "name": "name", "device": "device", "dtype": "dtype",
    "alphaNormFactor": "alpha_norm_factor", "alphaAbsolute": "alpha_absolute",
    "residualNorm": "residual_norm",
    "residualNormArtifact": "residual_norm_artifact",
    "positionMode": "position_mode",
    "doseLadder": "dose_ladder",
    "itemFilter": "item_filter",
    "promptMode": "prompt_mode", "systemPrompt": "system_prompt",
    "qwenThinkingEnabled": "qwen_thinking_enabled",
}


@dataclass
class OptVecGradientConfig:
    """Strict camelCase config, unknown keys refused — the training config's
    idiom, and a deliberate subset of its fields.

    Field NAMES match :class:`optvec_train.OptVecTrainConfig` wherever the
    meaning is the same (``layer``, the four α fields, ``prompt_mode`` …), which
    is what lets this object be handed straight to that module's
    :func:`resolve_alpha`, :func:`prepare_items` and :func:`baseline_prepass`
    instead of copies of them.
    """

    model_id: str
    layer: int
    target_train: DatasetRef
    revision: str | None = None
    name: str | None = None
    device: str | None = None
    dtype: str = "auto"

    alpha_norm_factor: float = 0.10
    alpha_absolute: float | None = None
    residual_norm: float | None = None
    residual_norm_artifact: str | None = None

    position_mode: str = "from_response"
    dose_ladder: tuple[float, ...] = DEFAULT_DOSE_LADDER
    item_filter: list[str] | None = None

    prompt_mode: str = prompt_render.CHAT_ASSISTANT
    system_prompt: str | None = None
    qwen_thinking_enabled: bool = False

    def __post_init__(self) -> None:
        if self.position_mode != "from_response":
            raise OptVecGradientConfigError(
                f"positionMode {self.position_mode!r} — the gradient survey is "
                "'from_response' only: the derivative it reports is taken at "
                "the ONE position whose logits are the answer distribution, "
                "which is also the only position a deployed injector can "
                "affect that answer from, and an 'all'-position delta would "
                "differentiate an intervention no study runs")
        if self.layer < 0:
            raise OptVecGradientConfigError(
                f"layer must be >= 0 (got {self.layer})")
        ladder = tuple(float(x) for x in self.dose_ladder)
        if not ladder:
            raise OptVecGradientConfigError(
                "doseLadder must name at least one dose — the ladder IS the "
                "evidence that the α→0 reading survives a usable α")
        if any(x <= 0 for x in ladder):
            raise OptVecGradientConfigError(
                "doseLadder entries must be positive fractions of the "
                "resolved α; α=0 is the baseline and is always measured")
        self.dose_ladder = ladder
        if self.alpha_absolute is None and self.residual_norm is None \
                and not self.residual_norm_artifact:
            raise OptVecGradientConfigError(
                "α needs a denominator: give alphaAbsolute directly, or "
                "residualNorm (or residualNormArtifact, a vector artifact "
                "whose sidecar carries residualNormPerLayer) so "
                "alphaNormFactor can be converted to absolute units")
        if self.item_filter is not None:
            if not isinstance(self.item_filter, (list, tuple)) \
                    or not all(isinstance(x, str) and x.strip()
                               for x in self.item_filter):
                raise OptVecGradientConfigError(
                    "itemFilter must be a list of item-id strings")
            if not self.item_filter:
                raise OptVecGradientConfigError(
                    "itemFilter is empty — omit the key to survey every item")
            self.item_filter = [str(x) for x in self.item_filter]

    @classmethod
    def from_dict(cls, payload: dict) -> "OptVecGradientConfig":
        if not isinstance(payload, dict):
            raise OptVecGradientConfigError(
                "the OptVec gradient config must be a JSON object")
        unknown = sorted(set(payload) - set(_CONFIG_KEYS) - {"datasets"})
        if unknown:
            raise OptVecGradientConfigError(
                "unknown OptVec gradient config key(s): " + ", ".join(unknown))
        raw_datasets = payload.get("datasets")
        if not isinstance(raw_datasets, dict):
            raise OptVecGradientConfigError(
                "the config needs a 'datasets' object with 'targetTrain'")
        unknown_sets = sorted(set(raw_datasets) - {"targetTrain"})
        if unknown_sets:
            raise OptVecGradientConfigError(
                "unknown datasets key(s): " + ", ".join(unknown_sets)
                + " — the survey reads the TRAIN split only. It takes a "
                "derivative, not a held-out measurement; val and test are "
                "read by the training and eval verbs")
        if "targetTrain" not in raw_datasets:
            raise OptVecGradientConfigError("datasets.targetTrain is required")
        try:
            target_train = DatasetRef.from_dict(raw_datasets["targetTrain"],
                                                "targetTrain")
        except OptVecConfigError as exc:
            raise OptVecGradientConfigError(str(exc)) from exc
        kwargs = {field_name: payload[key]
                  for key, field_name in _CONFIG_KEYS.items() if key in payload}
        for required in ("model_id", "layer"):
            if required not in kwargs:
                json_key = next(k for k, v in _CONFIG_KEYS.items()
                                if v == required)
                raise OptVecGradientConfigError(f"'{json_key}' is required")
        return cls(target_train=target_train, **kwargs)

    def to_dict(self) -> dict:
        payload = {json_key: getattr(self, field_name)
                   for json_key, field_name in _CONFIG_KEYS.items()}
        payload["doseLadder"] = list(self.dose_ladder)
        payload["datasets"] = {"targetTrain": self.target_train.to_dict()}
        if self.item_filter is None:
            payload.pop("itemFilter", None)
        else:
            payload["itemFilter"] = list(self.item_filter)
        return payload


def load_config(path: str) -> OptVecGradientConfig:
    with open(path, encoding="utf-8") as handle:
        return OptVecGradientConfig.from_dict(json.load(handle))


# -------------------------------------------------------------- the probe


def freeze_model(model) -> None:
    """Every parameter frozen, model in eval mode.

    Same configuration the training loop runs in, and for the same two
    reasons: backward is truncated at the injection layer (nothing below it
    requires grad, so no tape is built there), and a stray gradient can never
    land on a model parameter of a registry-resident model shared with
    inference paths.
    """
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    model.model.eval()


def _margin(logprobs: torch.Tensor, target_index: int,
            contrast_index: int) -> torch.Tensor:
    return logprobs[target_index] - logprobs[contrast_index]


def item_gradient(model, item: PreparedItem, contrast_index: int, *,
                  layer: int, position_mode: str = "from_response"
                  ) -> tuple[torch.Tensor, float]:
    """``(∂margin/∂delta, margin)`` at ``delta = 0`` for one item.

    One forward and one backward. The delta is exactly zero on the forward, so
    the margin returned here IS the unsteered margin — the same pass supplies
    the operating point and its derivative.

    This pass needs the same backward-time recomputation the training loop
    needs, for the same reason: without gradient checkpointing a single
    memo-length item at 27B holds every activation above the injection layer
    for backward and OOMs an 80 GB card. The training loop disarms
    checkpointing in its ``finally``, and the first cell of a 2026-08
    hard-item campaign died exactly here (2026-08-11): 300/300 steps trained,
    then the post-training cosine diagnostic ran this function bare and lost
    the run. Arming lives
    HERE — the one choke point every gradient caller (the training
    diagnostic, the gradient survey verb, fracture reference rows) shares —
    on CUDA only, skipped when a caller already armed, and undone before
    returning so the model leaves in the state it arrived.
    """
    probe = AdditiveDeltaProbe(layer=layer, hidden_size=model.hidden_size,
                               position_mode=position_mode,
                               device=model.device)
    already_armed = bool(getattr(model.model, "is_gradient_checkpointing",
                                 False))
    armed = (not already_armed and str(model.device).startswith("cuda")
             and optvec_train._enable_gradient_checkpointing(
                 model, lambda _message: None))
    try:
        # backward() inside the armed session: if gradient checkpointing is
        # active on this model, the backward pass re-runs the block forwards
        # and must re-fire the probe's hook (see
        # optvec_train.answer_logit_session).
        with optvec_train.answer_logit_session(model, [item], probe) as rows:
            logits = optvec_train.choice_logits(rows[0], item)
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            margin = _margin(logprobs, item.target_index, contrast_index)
            margin.backward()
    finally:
        if armed:
            optvec_train._disable_gradient_checkpointing(model)
    return probe.gradient(), float(margin.detach())


@torch.no_grad()
def margin_at_delta(model, item: PreparedItem, contrast_index: int,
                    delta, *, layer: int,
                    position_mode: str = "from_response") -> float:
    """The margin when a FIXED ``delta`` is added at the injection position.

    The finite-difference counterpart of :func:`item_gradient`: same forward,
    same position, a preset delta instead of a differentiated zero. Used by the
    property test that pins the gradient against directional derivatives, and
    available to any caller that wants a residual-space (rather than
    deployed-injection) reading.
    """
    probe = AdditiveDeltaProbe(layer=layer, hidden_size=model.hidden_size,
                               position_mode=position_mode, delta=delta,
                               device=model.device, requires_grad=False)
    rows = optvec_train.answer_logit_rows(model, [item], probe)
    logits = optvec_train.choice_logits(rows[0], item)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    return float(_margin(logprobs, item.target_index, contrast_index))


def unit(vector) -> list[float] | None:
    """``v/‖v‖`` as a plain float list, or ``None`` for a zero vector (which
    has no direction — never a silently normalized zero)."""
    values = [float(x) for x in vector]
    norm = math.sqrt(sum(x * x for x in values))
    if norm == 0.0 or not math.isfinite(norm):
        return None
    return [x / norm for x in values]


def contrast_index_for(item: PreparedItem, baselines: dict) -> int | None:
    """The margin's reference option index for one item, from the cached
    baseline pre-pass — the training loop's rule, verbatim."""
    record = baselines.get(item.id)
    if record is None or record.contrast is None:
        return None
    return item.options.index(record.contrast)


def mean_margin_gradient(model, items: list[PreparedItem], baselines: dict, *,
                         layer: int, position_mode: str = "from_response"
                         ) -> list[float] | None:
    """Unit direction of ``∂(mean margin)/∂delta`` over ``items``.

    Per-item gradients are computed one pass at a time and averaged, which is
    exactly the gradient of the mean: the delta is the same vector at every
    item's own answer position, so the derivative is linear in the items.
    ``None`` when no item has a usable contrast, or the mean gradient is zero.
    """
    freeze_model(model)
    total: torch.Tensor | None = None
    used = 0
    for item in items:
        contrast = contrast_index_for(item, baselines)
        if contrast is None:
            continue
        gradient, _margin_value = item_gradient(
            model, item, contrast, layer=layer, position_mode=position_mode)
        total = gradient if total is None else total + gradient
        used += 1
    if total is None or used == 0:
        return None
    return unit(total / used)


# --------------------------------------------------------------- dose ladder


def _option_logprobs(result) -> dict:
    return {score.option: score.logprob for score in result.options}


def log_odds(options, probabilities) -> dict:
    """``ln(p/(1−p))`` per option, clamped away from ±inf — the same formula
    (and the same eps) :class:`logprob.ChoiceResult.log_odds` uses, so a survey
    record's log-odds are commensurable with every behavioral study's."""
    eps = 1e-12
    return {option: math.log(max(p, eps) / max(1.0 - p, eps))
            for option, p in zip(options, probabilities)}


def dose_margin(model, item: PreparedItem, config: OptVecGradientConfig,
                direction: list[float], alpha: float, contrast: str) -> dict:
    """Score one item at one dose through the DEPLOYED instrument.

    ``VectorInjector`` adds ``alpha · vector`` without renormalizing and
    ``direction`` is a unit vector, so ``alpha`` IS the injected L2 norm — the
    same convention ``optvec_eval`` uses (its stored row already has norm
    ``alphaAbsolute``, dosed by a multiple).
    """
    result = logprob.score_options(
        model, item.prompt, list(item.options),
        injections=[CellInjection(layer=config.layer, vector=list(direction),
                                  alpha=float(alpha))],
        prompt_mode=config.prompt_mode, system_prompt=config.system_prompt,
        qwen_thinking_enabled=config.qwen_thinking_enabled)
    logprobs = _option_logprobs(result)
    probabilities = list(result.ordered_probabilities)
    return {"margin": float(logprobs[item.target] - logprobs[contrast]),
            "selected": result.selected,
            "probabilities": probabilities,
            "logOdds": log_odds(item.options, probabilities),
            "optionLogprobs": logprobs}


def linearity_summary(baseline_margin: float, gradient_norm: float,
                      doses: list[dict]) -> dict:
    """How far the α→0 reading survives the ladder.

    The direction surveyed is ``g/‖g‖``, so the directional derivative of the
    margin along it is ``g · g/‖g‖ = ‖g‖``, and the linear extrapolation to an
    injected norm ``a`` is

        ``predicted(a) = margin(0) + a · ‖g‖``

    with ``a`` in ABSOLUTE units (the dose fraction times the resolved α).
    Reported:

    * ``maxAbsDeviation`` — ``max over the ladder of |observed(a) −
      predicted(a)|``;
    * ``maxRelativeDeviation`` — that maximum divided by the largest predicted
      MOVEMENT ``max_a |predicted(a) − margin(0)|`` (i.e. by
      ``a_max · ‖g‖``), so it reads as "the linear model is wrong by this
      fraction of what it promised". ``None`` when the promise is zero.

    Both are computed against the baseline margin measured on the SAME
    instrument as the dose margins; the probe pass's own margin is reported
    separately in the item record rather than mixed in here.
    """
    rows = []
    worst = 0.0
    promise = 0.0
    for entry in doses:
        alpha = float(entry["alphaAbsolute"])
        predicted = baseline_margin + alpha * gradient_norm
        deviation = float(entry["margin"]) - predicted
        worst = max(worst, abs(deviation))
        promise = max(promise, abs(predicted - baseline_margin))
        rows.append({"alphaFraction": entry["alphaFraction"],
                     "alphaAbsolute": alpha,
                     "predictedMargin": predicted,
                     "observedMargin": float(entry["margin"]),
                     "deviation": deviation})
    return {
        "formula": "predicted(a) = margin(0) + a·‖g‖ for an injection of "
                   "a·(g/‖g‖) at the answer position; deviation = observed − "
                   "predicted; maxAbsDeviation = max|deviation| over the "
                   "ladder; maxRelativeDeviation = maxAbsDeviation / "
                   "max|predicted − margin(0)|",
        "baselineMargin": float(baseline_margin),
        "gradientNorm": float(gradient_norm),
        "maxAbsDeviation": float(worst),
        "maxRelativeDeviation": (None if promise <= 0.0
                                 else float(worst / promise)),
        "perDose": rows,
    }


# ------------------------------------------------------------------- survey


def _prepare(model, config: OptVecGradientConfig) -> tuple[list[PreparedItem],
                                                           dict]:
    """Load, hash-verify, filter, render and tokenize the train split."""
    try:
        rows = optvec_train.load_dataset(config.target_train, "targetTrain")
    except OptVecDataError as exc:
        raise OptVecGradientDataError(str(exc)) from exc
    notes: dict = {"files": {"targetTrain": config.target_train.to_dict()}}
    if config.item_filter is not None:
        try:
            rows = optvec_train.apply_item_filter(rows, config.item_filter,
                                                  "targetTrain")
        except OptVecDataError as exc:
            raise OptVecGradientDataError(str(exc)) from exc
        # Same shape as the training driver's record: a plain list of ids,
        # with the application facts alongside it.
        notes["itemFilter"] = list(config.item_filter)
        notes["itemFilterApplication"] = {"appliedTo": "targetTrain",
                                          "selectedCount": len(rows)}
    try:
        items = optvec_train.prepare_items(model, rows, role="target",
                                           split="train", config=config,
                                           declared="targetTrain")
    except OptVecDataError as exc:
        raise OptVecGradientDataError(str(exc)) from exc
    notes["counts"] = {"targetTrain": len(items)}
    notes["compositeHash"] = optvec_train.composite_dataset_hash(
        [config.target_train.sha256])
    return items, notes


def survey(config: OptVecGradientConfig, *, model=None,
           log: Callable[[str], None] | None = None) -> dict:
    """Read the α→0 gradient for every item and write the survey's run
    directory. Returns the run's summary dict."""
    def emit(message: str) -> None:
        if log is not None:
            log(message)

    name = (config.name or "").strip() or \
        f"{os.path.basename(config.model_id)}-L{config.layer}"
    run_directory = paths.make_unique_run_directory(f"optvec-gradient-{name}")
    notes: dict = {"stage": "starting", "claim": "localSensitivity",
                   "split": "train",
                   "positionMode": config.position_mode,
                   "doseLadder": list(config.dose_ladder)}
    dtype_name: str | None = None

    try:
        if model is None:
            from ..steering import model_loader
            from .lora_train import resolve_training_dtype
            device = model_loader.resolve_device(config.device)
            dtype_name = resolve_training_dtype(
                config.dtype, device, config.model_id, warn=emit)
            emit(f"loading {config.model_id} on {device} ({dtype_name})")
            model = model_loader.load(config.model_id, config.revision,
                                      dtype=dtype_name, device=device)
        dtype_name = getattr(model, "dtype", None) or dtype_name

        if config.layer >= model.num_layers:
            raise OptVecGradientConfigError(
                f"layer {config.layer} is out of range for a "
                f"{model.num_layers}-layer model")

        try:
            alpha_absolute, alpha_notes, _residual_provenance = \
                optvec_train.resolve_alpha(config)
        except OptVecConfigError as exc:
            raise OptVecGradientConfigError(str(exc)) from exc
        notes["alpha"] = alpha_notes

        items, dataset_notes = _prepare(model, config)
        notes["datasets"] = dataset_notes
        notes["stage"] = "baseline"

        baselines = optvec_train.baseline_prepass(model, items, config)
        emit(f"baseline scored for {len(baselines)} item(s)")
        notes["stage"] = "gradients"

        freeze_model(model)
        directions: dict[str, list[float]] = {}
        records: list[dict] = []
        skipped: list[dict] = []
        for item in items:
            baseline = baselines[item.id]
            contrast_index = contrast_index_for(item, baselines)
            if contrast_index is None:
                # No reference option ⇒ no margin ⇒ no derivative. Recorded,
                # never silently dropped: a surveyed set whose count shrinks
                # without saying why is a set nobody can audit.
                skipped.append({"id": item.id,
                                "reason": "no contrast option — the baseline "
                                          "pre-pass found no option to measure "
                                          "the target against"})
                continue
            gradient, probe_margin = item_gradient(
                model, item, contrast_index, layer=config.layer,
                position_mode=config.position_mode)
            gradient_norm = float(torch.linalg.vector_norm(gradient))
            direction = unit(gradient)
            option_logprobs = dict(baseline.option_logprobs)
            baseline_margin = float(option_logprobs[item.target]
                                    - option_logprobs[baseline.contrast])
            record = {
                "id": item.id,
                "options": list(item.options),
                "target": item.target,
                "contrast": baseline.contrast,
                "baselineSelected": baseline.selected,
                "baselineProbabilities": list(baseline.probabilities),
                "baselineOptionLogprobs": option_logprobs,
                "baselineLogOdds": log_odds(item.options,
                                            baseline.probabilities),
                "baselineMargin": baseline_margin,
                # The same margin as the probe pass computes it. The two paths
                # (stepped KV-cache instrument vs one teacher-forced pass) are
                # the same arithmetic on the same logits and should agree to
                # float32; the gap is recorded rather than asserted away.
                "probePassMargin": probe_margin,
                "probePassMarginDelta": probe_margin - baseline_margin,
                "gradientNorm": gradient_norm,
            }
            if direction is None:
                record["direction"] = None
                record["doses"] = []
                record["linearity"] = None
                record["note"] = ("the gradient is exactly zero — this item's "
                                  "margin is flat at α=0 in every direction, "
                                  "so there is no direction to mint")
                records.append(record)
                continue
            directions[item.id] = direction
            doses: list[dict] = []
            for fraction in config.dose_ladder:
                alpha = float(fraction) * alpha_absolute
                scored = dose_margin(model, item, config, direction, alpha,
                                     baseline.contrast)
                baseline_log_odds = record["baselineLogOdds"][item.target]
                doses.append({"alphaFraction": float(fraction),
                              "alphaAbsolute": alpha,
                              "margin": scored["margin"],
                              "marginDelta": scored["margin"] - baseline_margin,
                              "selected": scored["selected"],
                              "selectedTarget":
                                  scored["selected"] == item.target,
                              "probabilities": scored["probabilities"],
                              "logOddsTarget": scored["logOdds"][item.target],
                              "deltaLogOddsTarget":
                                  scored["logOdds"][item.target]
                                  - baseline_log_odds})
            record["doses"] = doses
            record["linearity"] = linearity_summary(baseline_margin,
                                                    gradient_norm, doses)
            records.append(record)
            emit(f"item {item.id}: ‖g‖ {gradient_norm:.4f}, "
                 f"max|deviation| {record['linearity']['maxAbsDeviation']:.4f}")

        notes["stage"] = "writing"
        _write_gradients(os.path.join(run_directory, GRADIENTS_FILE),
                         directions)
        readout = {
            "schemaVersion": SCHEMA_VERSION,
            "runType": RUN_TYPE,
            "runID": os.path.basename(os.path.normpath(run_directory)),
            "claim": "localSensitivity",
            "claimNote": "a per-item α→0 steepest-ascent direction for the "
                         "training margin. NOT a sufficiency claim: no "
                         "optimization ran, no anchor or capability budget was "
                         "enforced, and nothing was held out. The dose ladder "
                         "is the only behavioral evidence here.",
            "split": "train",
            "model": {"modelID": config.model_id,
                      "revision": (getattr(model, "revision", None)
                                   or config.revision),
                      "dtype": dtype_name,
                      "device": str(getattr(model, "device", "")),
                      "numLayers": int(model.num_layers),
                      "hiddenSize": int(model.hidden_size),
                      "substrate": vector_store.SUBSTRATE},
            "layer": config.layer,
            "alpha": alpha_notes,
            "alphaAbsolute": float(alpha_absolute),
            "doseLadder": list(config.dose_ladder),
            "positionMode": config.position_mode,
            "instrument": {
                "gradient": "one teacher-forced full-sequence pass with a "
                            "zero-initialized additive delta at the answer "
                            "position; margin = logp(target) − logp(contrast) "
                            "over the restricted choice softmax",
                "dose": "answerTokenLogprob (stepped KV-cache score_options "
                        "with the deployed CellInjection path)",
                "promptMode": config.prompt_mode,
                "qwenThinkingEnabled": config.qwen_thinking_enabled},
            "datasets": dataset_notes,
            "config": config.to_dict(),
            "items": records,
            "skipped": skipped,
        }
        with open(os.path.join(run_directory, SURVEY_JSON), "w",
                  encoding="utf-8") as handle:
            json.dump(readout, handle, indent=2, sort_keys=True)
            handle.write("\n")

        notes["itemCount"] = len(records)
        notes["directionCount"] = len(directions)
        notes["skippedCount"] = len(skipped)
        notes["summary"] = [
            {"id": r["id"], "gradientNorm": r["gradientNorm"],
             "baselineMargin": r["baselineMargin"],
             "maxAbsDeviation": (r["linearity"] or {}).get("maxAbsDeviation")}
            for r in records]
        notes["stage"] = "complete"
        emit(f"optvec gradient survey → {run_directory}")
        return {"runDirectory": run_directory,
                "layer": config.layer,
                "alphaAbsolute": float(alpha_absolute),
                "itemCount": len(records),
                "directionCount": len(directions),
                "items": records}
    finally:
        # Written LAST so the notes are complete, and in `finally` so a crashed
        # survey still records the stage it died in (run directories are
        # immutable and write_run_config never rewrites).
        write_run_config(run_directory, RUN_TYPE, model_id=config.model_id,
                         revision=getattr(model, "revision", None)
                         or config.revision,
                         dtype=dtype_name, notes=notes)


def _write_gradients(path: str, directions: dict) -> str:
    """``gradients.safetensors``: one float32 UNIT direction per item id."""
    import numpy as np
    from safetensors.numpy import save_file

    tensors = {item_id: np.asarray(values, dtype=np.float32)
               for item_id, values in directions.items()}
    if not tensors:
        # safetensors refuses an empty file; an empty survey still needs a
        # readable artifact set, so record the emptiness explicitly.
        tensors = {"__none__": np.zeros(0, dtype=np.float32)}
    save_file(tensors, path)
    return path


def load_survey(run_directory: str) -> tuple[dict, dict]:
    """``(survey readout, {item id: unit direction})`` from a survey run."""
    from safetensors.numpy import load_file

    survey_path = os.path.join(run_directory, SURVEY_JSON)
    if not os.path.exists(survey_path):
        raise OptVecGradientDataError(
            f"'{run_directory}' has no {SURVEY_JSON} — not an optvec-gradient "
            "survey run directory")
    with open(survey_path, encoding="utf-8") as handle:
        readout = json.load(handle)
    tensors = load_file(os.path.join(run_directory, GRADIENTS_FILE))
    directions = {key: [float(x) for x in value]
                  for key, value in tensors.items() if key != "__none__"}
    return readout, directions


# --------------------------------------------------------------------- mint


def mint(run_directory: str, item_id: str, name: str | None = None, *,
         root: str | None = None) -> dict:
    """Export one surveyed item's direction as a STANDARD vector artifact.

    The bytes are ``alphaAbsolute · (g/‖g‖)`` at the survey's layer, zeros at
    every other layer — the same full-depth convention the training driver
    writes, for the same reason (the injection path selects a layer by index,
    and a nonzero row at an unoptimized layer would be a direction nothing
    certifies).

    Written into its OWN immutable run directory: the survey run is a finished
    record and must not gain files afterwards, and two mints of the same item
    must not collide.

    The additive ``optvec`` sidecar block is what makes the artifact evaluable
    (``optvec_eval.require_optvec`` needs an integer ``layer`` and a positive
    ``alphaAbsolute``) — and what keeps it honest: ``method:
    "gradientDirection"``, ``optimized: false``, ``claim: "localSensitivity"``,
    plus the source survey, the item, and that item's linearity summary. The
    ordinary lifecycle then applies unchanged: norm backfill → eval →
    interpret.
    """
    readout, directions = load_survey(run_directory)
    record = next((r for r in readout.get("items", [])
                   if r.get("id") == item_id), None)
    if record is None:
        known = ", ".join(sorted(r.get("id", "?")
                                 for r in readout.get("items", []))) or "none"
        raise OptVecGradientDataError(
            f"survey '{run_directory}' has no item '{item_id}' "
            f"(surveyed: {known})")
    direction = directions.get(item_id)
    if direction is None:
        raise OptVecGradientDataError(
            f"item '{item_id}' has no direction in {GRADIENTS_FILE} — its "
            f"gradient was zero, so there is nothing to mint "
            f"({record.get('note') or 'see the survey record'})")

    leaf = (name or "").strip() or f"gradient-{item_id}"
    if os.sep in leaf or (os.altsep and os.altsep in leaf) or leaf in (".", ".."):
        raise OptVecGradientDataError(
            f"artifact name {leaf!r} is not a single path component — pass an "
            "explicit 'name' when the item id contains a path separator")

    model_block = readout.get("model") or {}
    layer = int(readout["layer"])
    alpha_absolute = float(readout["alphaAbsolute"])
    layer_count = int(model_block.get("numLayers") or 0)
    hidden = int(model_block.get("hiddenSize") or len(direction))
    if layer_count <= layer:
        raise OptVecGradientDataError(
            f"survey '{run_directory}' records {layer_count} model layer(s) "
            f"but its own layer is {layer} — the survey record is unusable "
            "for minting a full-depth artifact")
    if len(direction) != hidden:
        raise OptVecGradientDataError(
            f"item '{item_id}' direction is {len(direction)}-dimensional but "
            f"the survey records hidden size {hidden}")

    per_layer = [[0.0] * hidden for _ in range(layer_count)]
    per_layer[layer] = [alpha_absolute * x for x in direction]
    vectors = ConceptVectors(per_layer=per_layer)

    mint_directory = paths.make_unique_run_directory(
        f"optvec-gradient-vector-{leaf}", root)
    from datetime import datetime, timezone
    sidecar = SteeringVectorSidecar(
        modelID=model_block.get("modelID") or "",
        concept=leaf,
        stimulusSetHash=_mint_stimulus_hash(readout, item_id),
        layerCount=vectors.layer_count,
        hiddenSize=vectors.hidden_size,
        normsPerLayer=[vectors.norm(i) for i in range(vectors.layer_count)],
        extractionDate=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        revision=model_block.get("revision"),
        substrate=vector_store.SUBSTRATE,
        extractionMethod=EXTRACTION_METHOD,
        recipeMethod=EXTRACTION_METHOD,
        # Born WITHOUT residual norms, exactly like a trained OptVec vector:
        # run the norm backfill against the pinned neutral corpus before any
        # alphaInNormUnits condition uses this direction.
        residualNormPerLayer=None,
    )
    vector_store.save(vectors, sidecar, mint_directory, leaf)

    sidecar_path = os.path.join(mint_directory, f"{leaf}.json")
    with open(sidecar_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["optvec"] = {
        "layer": layer,
        "alphaAbsolute": alpha_absolute,
        "method": GRADIENT_METHOD,
        "optimized": False,
        "claim": "localSensitivity",
        "claimNote": "α→0 steepest-ascent direction of ONE item's choice "
                     "margin, scaled to the survey's α. No optimization, no "
                     "checkpoint selection, no anchor/capability budget, and "
                     "no held-out split were involved — 'sufficiency' is NOT "
                     "certified. The dose ladder's linearity summary below is "
                     "the whole of this artifact's behavioral evidence.",
        "sourceItem": item_id,
        # The flat per-item marker the geometry/fracture readouts look for
        # (``optvec_geometry.item_of``): a gradient direction is per-item by
        # construction, and an artifact that cannot say which item it belongs
        # to is ungroupable there.
        "item": item_id,
        "sourceSurveyRun": readout.get("runID"),
        "sourceSurveyRunDirectory": os.path.basename(
            os.path.normpath(run_directory)),
        "positionMode": readout.get("positionMode"),
        "doseLadder": list(readout.get("doseLadder") or []),
        "alpha": readout.get("alpha"),
        "datasets": readout.get("datasets"),
        "gradientNorm": record.get("gradientNorm"),
        "baselineMargin": record.get("baselineMargin"),
        "linearity": record.get("linearity"),
        "runID": os.path.basename(os.path.normpath(mint_directory)),
        "substrate": vector_store.SUBSTRATE,
        "gitSHA": optvec_train._git_sha(),
    }
    tmp = sidecar_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, sidecar_path)

    write_run_config(
        mint_directory, MINT_RUN_TYPE,
        model_id=sidecar.modelID or None, revision=sidecar.revision,
        dtype=(model_block.get("dtype") or None),
        notes={"stage": "complete", "claim": "localSensitivity",
               "sourceSurveyRun": readout.get("runID"),
               "sourceItem": item_id, "layer": layer,
               "alphaAbsolute": alpha_absolute,
               "artifact": {"name": leaf}})
    return {"runDirectory": mint_directory, "name": leaf,
            "vectorArtifactID": os.path.join(mint_directory, leaf),
            "layer": layer, "alphaAbsolute": alpha_absolute,
            "sourceItem": item_id,
            "sourceSurveyRun": readout.get("runID")}


def _mint_stimulus_hash(readout: dict, item_id: str) -> str:
    """``optvec-gradient:<sha256 over the survey's dataset identity + item>``.

    An item's direction is identified by the bytes it was read from AND which
    row of them: two items of one file are different directions, and the file
    hash alone would call them the same provenance.
    """
    composite = ((readout.get("datasets") or {}).get("compositeHash")
                 or readout.get("runID") or "")
    digest = hashlib.sha256(
        f"{composite}\n{item_id}".encode("utf-8")).hexdigest()
    return STIMULUS_PREFIX + digest


__all__ = [
    "DEFAULT_DOSE_LADDER", "EXTRACTION_METHOD", "GRADIENTS_FILE",
    "GRADIENT_METHOD", "MINT_RUN_TYPE", "OptVecGradientConfig",
    "OptVecGradientConfigError", "OptVecGradientDataError", "RUN_TYPE",
    "STIMULUS_PREFIX", "SURVEY_JSON", "contrast_index_for", "dose_margin",
    "freeze_model", "item_gradient", "linearity_summary", "load_config",
    "load_survey", "margin_at_delta", "mean_margin_gradient", "mint",
    "survey", "unit",
]
