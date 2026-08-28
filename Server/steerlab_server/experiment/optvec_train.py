"""OptVec — optimize a residual-stream injection vector by backprop through the
frozen model (plan of record: ``docs/OPTVEC-OPTIMIZED-INJECTION-VECTORS-PLAN.md``).

Where every other recipe runs concept → vector → behavior, this one inverts the
pipeline: a direction is SEARCHED for such that forced-choice judgments on a
target set move, while designated-unchanged judgments and objective capability
are preserved. Everything issue-specific enters as hashed choice-row JSONL
data; nothing here knows what the items are about.

Modeled on :mod:`steerlab_server.experiment.lora_train`, the engine's other
backprop loop: same training-dtype policy (never fp16), same
own-run-type/immutable-run-directory discipline, same provenance-sidecar habit.
Two things it deliberately does NOT reuse: every scoring/generation entry point
in this package is ``@torch.no_grad()``-decorated at function level, so the
driver owns its forward pass; and the deployed ``VectorInjector`` has no
autograd path, so training arms
:class:`~steerlab_server.steering.trainable_injector.TrainableVectorInjector`
instead (the exported vector is deployed through the standard injector).

Claim grammar (binding, plan §1): what this certifies is SUFFICIENCY — "a
direction sufficient to shift judgments on issue X within stated anchor and
capability budgets". Never necessity, never uniqueness.

Firewall (plan §3): ``train`` sees the *train* split; the *val* split carries
early stopping and checkpoint selection, so selection pressure lands there by
design. The *test* split is untouched here — it is first read by the eval verb
(WP4), and only a confirm-style behavioral study produces citable numbers.

Per-item runs (WP-S4a)
----------------------
Two additive, opt-in config keys turn the same loop into the per-item
instrument the fracture/family work needs. Both are absent from every config
written before them, and a config without them behaves exactly as it did:

* ``itemFilter`` — train on named ``targetTrain`` items only. The file's hash
  pin is untouched (the filter is config data, recorded in the run's notes and
  in the artifact's sidecar); it applies to the TRAIN split only, for the
  reason set out in :func:`apply_item_filter`. A filtered run additionally
  stamps ``training.cosineToGradient``: how much of the optimized direction is
  the α→0 steepest-ascent direction of the very margin it optimized
  (:mod:`optvec_gradient`), a diagnostic and never a gate.
* omitting ``datasets.targetVal`` — **fixed-steps mode**. One or two training
  items do not have a meaningful val split, and pretending otherwise would
  select checkpoints on noise. Nothing is evaluated, nothing is selected, the
  loop runs exactly ``stepsMax`` steps, and the final step is the artifact —
  stamped ``chosenCheckpoint.selection = "finalStep"`` and
  ``selectionScope = "none-finalStep"`` so no reader can mistake it for a
  val-selected vector. Half a val split refuses (``anchorVal`` without
  ``targetVal``), as do the val-only knobs (``valEvery``,
  ``earlyStopPatience``) when there is nothing to select on.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import torch

from ..steering import vector_store
from ..steering.trainable_injector import (POSITION_MODES, PositionedDelta,
                                           TrainableVectorInjector)
from ..steering.vector_store import ConceptVectors, SteeringVectorSidecar
from . import logprob, paths, prompt_render
from .lora_train import resolve_training_dtype
from .run_config import write_run_config
from .sweep_selection import ChoiceRow, load_choice_rows

#: Run type stamped into ``config.json`` (peer of ``lora-train``,
#: ``reader-fit``).
RUN_TYPE = "optvec-train"

#: Sidecar ``extractionMethod`` / ``recipeMethod`` for an optimized vector.
EXTRACTION_METHOD = "optvec"

#: ``stimulusSetHash`` prefix — an OptVec vector has no stimulus set, it has a
#: dataset bundle, and the composite below is what identifies it (same
#: namespacing idiom as the J-lens and Gemma Scope artifacts).
STIMULUS_PREFIX = "optvec:"

#: The three item roles. A role is what an item CONTRIBUTES to the objective,
#: not what it is about.
ROLES = ("target", "anchor", "capability")


class OptVecConfigError(ValueError):
    """A training config that cannot be run as written."""


class OptVecDataError(ValueError):
    """A dataset that cannot be trained on (hash drift, multi-token option…)."""


# ---------------------------------------------------------------- config


@dataclass(frozen=True)
class DatasetRef:
    """One hashed choice-row JSONL. The hash is the pin: split membership and
    item content are DATA, and drift in either must refuse rather than
    silently retrain on a different set."""

    path: str
    sha256: str

    @classmethod
    def from_dict(cls, value: dict, label: str) -> "DatasetRef":
        if not isinstance(value, dict):
            raise OptVecConfigError(
                f"datasets.{label} must be an object with 'path' and "
                f"'sha256' (got {value!r})")
        path = value.get("path")
        digest = value.get("sha256")
        if not isinstance(path, str) or not path.strip():
            raise OptVecConfigError(f"datasets.{label}.path must be a file path")
        if not isinstance(digest, str) or not digest.strip():
            raise OptVecConfigError(
                f"datasets.{label}.sha256 must be the SHA-256 of the file's "
                "raw bytes — an unpinned dataset makes the run's own record "
                "unable to say what it trained on")
        return cls(path=path, sha256=digest.strip().lower())

    def to_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class OptVecDatasets:
    target_train: DatasetRef
    #: Optional since WP-S4a. Present ⇒ the historical path (val evaluations,
    #: early stopping, val-selected checkpoint). ABSENT ⇒ fixed-steps mode: no
    #: val evaluation exists at all, training runs exactly ``stepsMax`` steps,
    #: and the FINAL step is the artifact (stamped as such — see
    #: :func:`optimize`). The per-item regime the mode was built for has one
    #: or two train items, where a val split is not small, it is meaningless.
    target_val: DatasetRef | None = None
    anchor_train: DatasetRef | None = None
    anchor_val: DatasetRef | None = None
    capability_train: DatasetRef | None = None

    @property
    def selects_on_val(self) -> bool:
        return self.target_val is not None

    def to_dict(self) -> dict:
        out = {"targetTrain": self.target_train.to_dict()}
        for key, ref in (("targetVal", self.target_val),
                         ("anchorTrain", self.anchor_train),
                         ("anchorVal", self.anchor_val),
                         ("capabilityTrain", self.capability_train)):
            if ref is not None:
                out[key] = ref.to_dict()
        return out


_DATASET_KEYS = {
    "targetTrain": "target_train", "targetVal": "target_val",
    "anchorTrain": "anchor_train", "anchorVal": "anchor_val",
    "capabilityTrain": "capability_train",
}

# JSON key → dataclass field for the scalar config. The JSON is camelCase like
# every other authored file in the workspace; unknown keys REFUSE (a silently
# ignored "lambdaAnchor" typo would run S1 while the record claims S2).
_CONFIG_KEYS = {
    "modelID": "model_id", "revision": "revision", "layer": "layer",
    "name": "name", "device": "device", "dtype": "dtype",
    "alphaNormFactor": "alpha_norm_factor", "alphaAbsolute": "alpha_absolute",
    "residualNorm": "residual_norm",
    "residualNormArtifact": "residual_norm_artifact",
    "positionMode": "position_mode",
    "lambdaShift": "lambda_shift", "lambdaAnchor": "lambda_anchor",
    "lambdaCap": "lambda_cap", "lambdaOrth": "lambda_orth",
    "priorVectorPaths": "prior_vector_paths",
    "hingeMarginNats": "hinge_margin_nats",
    "lr": "lr", "lrSchedule": "lr_schedule", "stepsMax": "steps_max",
    "earlyStopPatience": "early_stop_patience", "valEvery": "val_every",
    "microbatchSize": "microbatch_size",
    "gradAccumToEffective": "grad_accum_to_effective",
    "mixRatio": "mix_ratio", "seed": "seed",
    "anchorKLBudget": "anchor_kl_budget",
    "shuffleTargetLabels": "shuffle_target_labels",
    "checkpointEvery": "checkpoint_every",
    "gradientCheckpointing": "gradient_checkpointing",
    "promptMode": "prompt_mode", "systemPrompt": "system_prompt",
    "qwenThinkingEnabled": "qwen_thinking_enabled",
    # WP-S4a, additive and LAST so the canonical key order of every existing
    # config is unchanged (campaign cells hash their canonical payload).
    "itemFilter": "item_filter",
}

#: Config keys that only mean something when a val split exists.
_VAL_ONLY_KEYS = ("earlyStopPatience", "valEvery")


@dataclass
class OptVecTrainConfig:
    model_id: str
    layer: int
    datasets: OptVecDatasets
    revision: str | None = None
    name: str | None = None
    device: str | None = None
    dtype: str = "auto"

    # α is denominated in units of the residual-stream norm at the layer (the
    # library-wide convention); the absolute value is the product.
    alpha_norm_factor: float = 0.10
    alpha_absolute: float | None = None
    residual_norm: float | None = None
    residual_norm_artifact: str | None = None

    position_mode: str = "from_response"

    lambda_shift: float = 1.0
    lambda_anchor: float = 1.0
    lambda_cap: float = 1.0
    lambda_orth: float = 0.0
    prior_vector_paths: list[str] = field(default_factory=list)
    hinge_margin_nats: float | None = 4.0

    lr: float = 1e-2
    lr_schedule: str = "cosine"
    steps_max: int = 500
    early_stop_patience: int = 50
    val_every: int = 10
    microbatch_size: int = 4
    grad_accum_to_effective: int = 16
    mix_ratio: tuple[int, int, int] = (2, 1, 1)
    seed: int = 0
    anchor_kl_budget: float | None = None
    shuffle_target_labels: bool = False
    checkpoint_every: int = 50
    #: None resolves to "on for CUDA, off elsewhere" — recomputation is a
    #: memory trade that only pays where memory is the binding constraint,
    #: and it must never slow the CPU test path down.
    gradient_checkpointing: bool | None = None

    prompt_mode: str = prompt_render.CHAT_ASSISTANT
    system_prompt: str | None = None
    qwen_thinking_enabled: bool = False

    #: WP-S4a — restrict TRAINING to these item ids. ``None`` (the default) is
    #: the historical whole-file behavior. See :func:`apply_item_filter` for
    #: why it applies to ``targetTrain`` only.
    item_filter: list[str] | None = None

    def __post_init__(self) -> None:
        if self.position_mode not in POSITION_MODES:
            raise OptVecConfigError(
                f"positionMode {self.position_mode!r} — one of "
                + ", ".join(POSITION_MODES))
        if self.layer < 0:
            raise OptVecConfigError(f"layer must be >= 0 (got {self.layer})")
        if self.steps_max < 1:
            raise OptVecConfigError("stepsMax must be at least 1")
        if self.microbatch_size < 1:
            raise OptVecConfigError("microbatchSize must be at least 1")
        if self.grad_accum_to_effective < 1:
            raise OptVecConfigError("gradAccumToEffective must be at least 1")
        if self.val_every < 1:
            raise OptVecConfigError("valEvery must be at least 1")
        if len(tuple(self.mix_ratio)) != 3 or any(
                r < 0 for r in self.mix_ratio):
            raise OptVecConfigError(
                "mixRatio must be three non-negative numbers "
                "[target, anchor, capability]")
        self.mix_ratio = tuple(float(r) for r in self.mix_ratio)
        if self.lambda_anchor > 0 and self.datasets.anchor_train is None:
            raise OptVecConfigError(
                "lambdaAnchor > 0 needs datasets.anchorTrain — the anchor term "
                "is the preservation constraint and has nothing to preserve "
                "without its own items")
        if self.lambda_cap > 0 and self.datasets.capability_train is None:
            raise OptVecConfigError(
                "lambdaCap > 0 needs datasets.capabilityTrain")
        if self.lambda_orth > 0 and not self.prior_vector_paths:
            raise OptVecConfigError(
                "lambdaOrth > 0 needs priorVectorPaths — the orthogonality "
                "penalty is measured against accepted solutions")
        if self.alpha_absolute is None and self.residual_norm is None \
                and not self.residual_norm_artifact:
            raise OptVecConfigError(
                "α needs a denominator: give alphaAbsolute directly, or "
                "residualNorm (or residualNormArtifact, a vector artifact "
                "whose sidecar carries residualNormPerLayer) so "
                "alphaNormFactor can be converted to absolute units")
        if self.datasets.target_val is None and \
                self.datasets.anchor_val is not None:
            # Half a val split is not a mode: anchorVal exists only to enter
            # the val composite, which no longer exists without targetVal, so
            # it would be loaded, baseline-scored, and then silently ignored.
            raise OptVecConfigError(
                "datasets.anchorVal is declared without datasets.targetVal — "
                "val evaluation either exists or it does not. Declare "
                "targetVal (the selected-checkpoint path) or drop anchorVal "
                "(fixed-steps mode, where the final step IS the artifact)")
        if self.item_filter is not None:
            if not isinstance(self.item_filter, (list, tuple)) \
                    or not all(isinstance(x, str) and x.strip()
                               for x in self.item_filter):
                raise OptVecConfigError(
                    "itemFilter must be a list of item-id strings")
            if not self.item_filter:
                raise OptVecConfigError(
                    "itemFilter is empty — omit the key to train on the whole "
                    "targetTrain file; an empty list would select no items")
            self.item_filter = [str(x) for x in self.item_filter]

    @classmethod
    def from_dict(cls, payload: dict) -> "OptVecTrainConfig":
        if not isinstance(payload, dict):
            raise OptVecConfigError("the OptVec config must be a JSON object")
        unknown = sorted(set(payload) - set(_CONFIG_KEYS) - {"datasets"})
        if unknown:
            raise OptVecConfigError(
                "unknown OptVec config key(s): " + ", ".join(unknown))
        raw_datasets = payload.get("datasets")
        if not isinstance(raw_datasets, dict):
            raise OptVecConfigError(
                "the config needs a 'datasets' object (targetTrain, targetVal, "
                "and optionally anchorTrain/anchorVal/capabilityTrain)")
        unknown_sets = sorted(set(raw_datasets) - set(_DATASET_KEYS))
        if unknown_sets:
            raise OptVecConfigError(
                "unknown datasets key(s): " + ", ".join(unknown_sets))
        if "targetTrain" not in raw_datasets:
            raise OptVecConfigError("datasets.targetTrain is required")
        if "targetVal" not in raw_datasets:
            # Fixed-steps mode. The val-only knobs are refused HERE rather
            # than ignored: a config that sets earlyStopPatience alongside no
            # val split is describing a selection policy the run cannot
            # execute, and a silently ignored key lets the record lie about
            # how its artifact was chosen.
            declared = [key for key in _VAL_ONLY_KEYS if key in payload]
            if declared:
                raise OptVecConfigError(
                    ", ".join(declared) + " set without datasets.targetVal — "
                    "there is nothing to select on. Fixed-steps mode runs "
                    "exactly stepsMax steps and takes the FINAL checkpoint; "
                    "drop these keys, or declare targetVal")
        datasets = OptVecDatasets(**{
            field_name: DatasetRef.from_dict(raw_datasets[key], key)
            for key, field_name in _DATASET_KEYS.items() if key in raw_datasets})
        kwargs = {field_name: payload[key]
                  for key, field_name in _CONFIG_KEYS.items() if key in payload}
        for required in ("model_id", "layer"):
            if required not in kwargs:
                json_key = next(k for k, v in _CONFIG_KEYS.items()
                                if v == required)
                raise OptVecConfigError(f"'{json_key}' is required")
        return cls(datasets=datasets, **kwargs)

    def to_dict(self) -> dict:
        payload = {json_key: getattr(self, field_name)
                   for json_key, field_name in _CONFIG_KEYS.items()}
        payload["mixRatio"] = list(self.mix_ratio)
        payload["priorVectorPaths"] = list(self.prior_vector_paths)
        payload["datasets"] = self.datasets.to_dict()
        # Absent, not null, when unused: this payload is canonicalized and
        # HASHED as a campaign cell's identity, so a config written before
        # itemFilter existed must still canonicalize to the same bytes.
        if self.item_filter is None:
            payload.pop("itemFilter", None)
        else:
            payload["itemFilter"] = list(self.item_filter)
        if not self.datasets.selects_on_val:
            # Fixed-steps mode (S4): the val-only knobs have nothing to act
            # on and from_dict REFUSES them when declared — but they carry
            # dataclass defaults, so emitting them unconditionally made every
            # campaign-materialized fixed-steps cell refuse its own canonical
            # config (observed live 2026-08-11: all 15 cells of a campaign).
            # Same absent-not-null discipline as itemFilter; val-bearing
            # configs are byte-unchanged, so existing campaign cell hashes
            # (rve-1, seeds-ext) are unaffected.
            for key in _VAL_ONLY_KEYS:
                payload.pop(key, None)
        return payload


def load_config(path: str) -> OptVecTrainConfig:
    with open(path, encoding="utf-8") as handle:
        return OptVecTrainConfig.from_dict(json.load(handle))


def composite_dataset_hash(hashes: list[str]) -> str:
    """The dataset bundle's identity: SHA-256 over the sorted per-file hashes
    joined with newlines (the ``response_format.ids_hash`` idiom). Order-free,
    so the same five files identify the same bundle however the config lists
    them."""
    joined = "\n".join(sorted(hashes))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- datasets


@dataclass(frozen=True)
class PreparedItem:
    """One training item: the rendered prompt, its single-token option ids, and
    the role it plays in the objective."""

    id: str
    role: str
    split: str
    prompt: str
    prompt_text: str            # the RENDERED prompt (what the model sees)
    input_ids: tuple[int, ...]
    options: tuple[str, ...]
    option_token_ids: tuple[int, ...]
    target: str

    @property
    def answer_position(self) -> int:
        """Index whose next-token logits are the answer distribution."""
        return len(self.input_ids) - 1

    @property
    def target_index(self) -> int:
        return self.options.index(self.target)


def prepare_items(model, rows: tuple[ChoiceRow, ...], *, role: str, split: str,
                  config: OptVecTrainConfig,
                  declared: str) -> list[PreparedItem]:
    """Render + tokenize choice rows for the TRAINING path.

    Rendering goes through :mod:`prompt_render` with the same arguments the
    logprob instrument uses, so a training prompt is byte-identical to the
    eval-time prompt for the same row.

    Training is restricted to SINGLE-token options and refuses otherwise. Eval
    supports multi-token options (joint logprob over stepped option positions);
    training does not, because the whole design is one forward pass per item
    reading the full choice distribution at ONE position — a multi-token option
    would need its own option-step injections and a per-option pass, which is
    a different (and far costlier) computation than the one being
    differentiated.
    """
    items: list[PreparedItem] = []
    for row in rows:
        rendered = prompt_render.render(
            model.tokenizer, row.prompt, model_id=model.model_id,
            prompt_mode=config.prompt_mode, system_prompt=config.system_prompt,
            qwen_thinking_enabled=config.qwen_thinking_enabled)
        token_ids: list[int] = []
        for option in row.options:
            ids = list(model.tokenizer(
                option, add_special_tokens=False).input_ids)
            if len(ids) != 1:
                raise OptVecDataError(
                    f"'{declared}' item '{row.id}': option {option!r} "
                    f"tokenizes to {len(ids)} tokens — OptVec TRAINING needs "
                    "single-token options (one forward pass per item reads the "
                    "whole choice distribution at one position). Use short "
                    "canonical single-token labels, or score this set with the "
                    "eval instrument, which does support multi-token options")
            token_ids.append(ids[0])
        if len(set(token_ids)) != len(token_ids):
            raise OptVecDataError(
                f"'{declared}' item '{row.id}': two options share a token id — "
                "their probabilities would be indistinguishable")
        items.append(PreparedItem(
            id=row.id, role=role, split=split, prompt=row.prompt,
            prompt_text=rendered.text, input_ids=tuple(rendered.input_ids),
            options=tuple(row.options), option_token_ids=tuple(token_ids),
            target=row.target))
    return items


def resolve_data_path(path: str) -> str:
    """A dataset ref is WORKSPACE data: resolve a relative path against
    ``STEERLAB_ROOT``, never the process cwd. Observed live 2026-08-11
    (rve-1-campaign-1): campaign cell jobs run with cwd = the CELL directory
    (their sbatch bundle ``cd``s there for the slurm logs), so a bare
    ``os.path.exists("prompts/optvec/…")`` looked under the cell dir and all
    six cells died "file not found" with the files sitting in the workspace
    the whole time."""
    return path if os.path.isabs(path) else \
        os.path.join(paths.project_root(), path)


def load_dataset(ref: DatasetRef, label: str) -> tuple[ChoiceRow, ...]:
    """Load a hashed choice-row file, refusing on hash drift.

    The strict loader is ``sweep_selection.load_choice_rows`` — the same rules
    the sweep's logprobShift instrument enforces, cross-engine pinned.
    """
    rows, digest = load_choice_rows(resolve_data_path(ref.path), label)
    if digest != ref.sha256:
        raise OptVecDataError(
            f"dataset '{label}' ({ref.path}) hashes {digest} but the config "
            f"pins {ref.sha256} — the file changed since the run was "
            "configured; re-pin it deliberately or restore the pinned bytes")
    return rows


def apply_item_filter(rows: tuple[ChoiceRow, ...], ids: list[str],
                      label: str) -> tuple[ChoiceRow, ...]:
    """Keep only ``ids`` from a loaded choice-row file.

    **Applies to ``targetTrain`` only, deliberately.** Item ids are unique
    across the WHOLE bundle (``_prepare_all`` refuses an id that appears in two
    splits — baselines are keyed by id, and overlapping splits are a firewall
    breach besides), so val, test and anchor ids are disjoint from train ids by
    construction. Filtering any of them by a list of TRAIN ids would therefore
    select nothing and empty the split. A per-item run restricts what gradients
    are taken on; it does not, and cannot, restrict what they are evaluated
    against.

    The file's own hash pin is untouched: the filter is CONFIG data (recorded
    in the run's notes and sidecar), not an edit to the pinned bytes. Two runs
    filtering the same file to different items share a dataset identity and are
    told apart by their recorded filter.
    """
    wanted = list(dict.fromkeys(ids))          # de-duplicated, order preserved
    known = {row.id for row in rows}
    unknown = [item_id for item_id in wanted if item_id not in known]
    if unknown:
        raise OptVecDataError(
            f"itemFilter names item id(s) not in '{label}': "
            + ", ".join(unknown)
            + f" — the file declares {len(known)} item(s). A filter that "
            "names a missing item is a typo or a stale id, and training on "
            "the remainder would silently run a different experiment than "
            "the config describes")
    selected = set(wanted)
    kept = tuple(row for row in rows if row.id in selected)
    if not kept:
        # Defensive: the config layer refuses an empty itemFilter and the
        # unknown-id check above catches every other way to select nothing, so
        # reaching here means a caller filtered by hand. Refuse rather than
        # optimize an empty objective.
        raise OptVecDataError(
            f"itemFilter selected no items from '{label}' — there is nothing "
            "to take a gradient on")
    return kept


def shuffled_target_labels(items: list[PreparedItem],
                           seed: int) -> tuple[list[PreparedItem], list[int]]:
    """S0's shuffled-target null: permute the target labels across items.

    Returns the relabeled items and the permutation (``new[i] = old[perm[i]]``)
    so the run's record can reproduce the assignment. A permutation that
    happens to reproduce the original labels is redrawn — the null must
    actually be a null, not the target condition under another name — and when
    no differing assignment exists (one item, or every item declaring the same
    label), the identity is returned and the caller's record shows it.
    """
    if not items:
        return [], []
    rng = random.Random(seed)
    order = list(range(len(items)))
    labels = [item.target for item in items]
    can_differ = len(set(labels)) > 1
    for _ in range(100):
        rng.shuffle(order)
        if not can_differ or any(labels[order[i]] != labels[i]
                                 for i in range(len(items))):
            break
    relabeled = []
    for i, item in enumerate(items):
        label = labels[order[i]]
        if label not in item.options:
            # Options differ across items, so a borrowed label may not exist
            # here; fall back to this item's first non-target option, which is
            # still label-independent of its baseline target.
            label = next((o for o in item.options if o != item.target),
                         item.target)
        relabeled.append(PreparedItem(
            id=item.id, role=item.role, split=item.split, prompt=item.prompt,
            prompt_text=item.prompt_text, input_ids=item.input_ids,
            options=item.options, option_token_ids=item.option_token_ids,
            target=label))
    return relabeled, order


# ------------------------------------------------------------ loss terms


def shift_loss(choice_logits: torch.Tensor, target_index: int,
               contrast_index: int, hinge_margin: float | None) -> torch.Tensor:
    """``hinge(m − margin)``, margin = logp(target) − logp(contrast).

    The two logprobs share a normalizer, so the margin is the raw logit gap and
    is identical whether taken over the full vocabulary or restricted to the
    option tokens. ``hinge_margin=None`` disables the hinge and returns
    ``−margin`` (an unbounded objective that keeps spending budget on items it
    has already flipped — the reason the hinge is on by default).
    """
    logprobs = torch.log_softmax(choice_logits.float(), dim=-1)
    margin = logprobs[target_index] - logprobs[contrast_index]
    if hinge_margin is None:
        return -margin
    return torch.clamp(hinge_margin - margin, min=0.0)


def anchor_kl(choice_logits: torch.Tensor,
              baseline_probabilities: torch.Tensor) -> torch.Tensor:
    """``KL(p0 ‖ p_v)`` over the choice-token softmax — direction matters.

    ``p0`` is the BASELINE distribution (the reference); ``p_v`` is the steered
    one. This penalizes moving probability mass away from where the unsteered
    model put it, and it preserves CONFIDENCE, not merely the argmax.
    """
    p0 = baseline_probabilities.float()
    log_p0 = torch.log(p0.clamp_min(1e-12))
    log_pv = torch.log_softmax(choice_logits.float(), dim=-1)
    return torch.sum(p0 * (log_p0 - log_pv))


def capability_ce(choice_logits: torch.Tensor,
                  correct_index: int) -> torch.Tensor:
    """Cross-entropy toward the objectively correct answer token."""
    return -torch.log_softmax(choice_logits.float(), dim=-1)[correct_index]


def orthogonality_penalty(vector: torch.Tensor,
                          priors: list[torch.Tensor]) -> torch.Tensor:
    """``Σ_j cos²(v, v_j)`` against already-accepted solutions (plan S3)."""
    if not priors:
        return vector.new_zeros(())
    total = vector.new_zeros(())
    v_norm = vector.norm().clamp_min(1e-12)
    for prior in priors:
        cosine = torch.dot(vector, prior) / (v_norm * prior.norm().clamp_min(1e-12))
        total = total + cosine ** 2
    return total


# ------------------------------------------------------------ forward path


def _logits_keep_kwargs(model_module, keep: int) -> dict:
    """``{"logits_to_keep": keep}`` when the forward supports it, else the 4.x
    spelling, else nothing (test fakes and exotic wrappers).

    Materializing logits for every position is what makes this loop expensive:
    at Gemma's 262K vocabulary a 1K-token microbatch row is hundreds of MB of
    tensor to read one row from. The batch is length-sorted so ``keep`` stays
    small.
    """
    import inspect
    target = getattr(model_module, "forward", model_module)
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return {}
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return {name: int(keep)}
    return {}


def _pad_id(tokenizer) -> int:
    return (getattr(tokenizer, "pad_token_id", None)
            or getattr(tokenizer, "eos_token_id", None) or 0)


def _batch_tensors(model, items: list[PreparedItem]):
    """Right-padded ``(input_ids, attention_mask, answer_positions)``.

    RIGHT padding is the convention here and it is asserted once, at the only
    place padding is constructed: every item's prompt occupies ``[0, len)``, so
    its answer position is ``len − 1`` and default position ids are correct for
    the non-pad prefix. Left padding would silently shift every answer position.
    """
    device = model.device
    width = max(len(item.input_ids) for item in items)
    pad = _pad_id(model.tokenizer)
    input_ids = torch.tensor(
        [list(item.input_ids) + [pad] * (width - len(item.input_ids))
         for item in items], dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        [[1] * len(item.input_ids) + [0] * (width - len(item.input_ids))
         for item in items], dtype=torch.long, device=device)
    answer_positions = torch.tensor(
        [item.answer_position for item in items], dtype=torch.long,
        device=device)
    assert bool((attention_mask.gather(
        1, answer_positions.unsqueeze(1)) == 1).all()), \
        "right-padding convention violated: an answer position landed on a pad"
    return input_ids, attention_mask, answer_positions


@contextmanager
def answer_logit_session(model, items: list[PreparedItem],
                         injector: PositionedDelta | None):
    """``[batch, vocab]`` answer-position logit rows, yielded with the hook
    session and the injector's batch state STILL ARMED.

    Any ``backward()`` through these rows must run inside this block whenever
    gradient checkpointing may be active: the backward pass RE-RUNS each
    checkpointed block's forward (recompute), which re-fires the residual
    hooks. A backward taken after the session closed recomputes an UNSTEERED
    block — torch detects the mismatched graph and raises (observed while
    profiling a hard-item campaign OOM, 2026-08-11); without that guard it
    would be a silently wrong gradient. No-grad readers use
    :func:`answer_logit_rows` instead.
    """
    input_ids, attention_mask, answer_positions = _batch_tensors(model, items)
    seq_len = input_ids.shape[1]
    keep = seq_len - int(answer_positions.min())
    kwargs = _logits_keep_kwargs(model.model, keep)
    interventions = []
    if injector is not None:
        injector.set_batch(answer_positions=answer_positions,
                           attention_mask=attention_mask)
        interventions.append(injector)
    try:
        with model.hooked.session(interventions):
            out = model.model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              use_cache=False, **kwargs)
            logits = out.logits
            # A model that honored `keep` returned only the tail; one that
            # ignored it returned everything. Derive the offset from the SHAPE
            # rather than the request, so both are correct.
            offset = seq_len - logits.shape[1]
            rows = logits[torch.arange(len(items), device=logits.device),
                          answer_positions - offset]
            yield rows
    finally:
        if injector is not None:
            injector.clear_batch()


def answer_logit_rows(model, items: list[PreparedItem],
                      injector: PositionedDelta | None) -> torch.Tensor:
    """``[batch, vocab]`` next-token logits at each item's answer position.

    Forward-only convenience over :func:`answer_logit_session` — safe for
    no-grad evaluation, and for backward ONLY when gradient checkpointing is
    off (see the session's docstring)."""
    with answer_logit_session(model, items, injector) as rows:
        return rows


def choice_logits(row: torch.Tensor, item: PreparedItem) -> torch.Tensor:
    index = torch.tensor(item.option_token_ids, dtype=torch.long,
                         device=row.device)
    return row.index_select(0, index)


# --------------------------------------------------------- baseline pre-pass


BASELINE_CACHE = "baseline-cache.jsonl"


@dataclass(frozen=True)
class BaselineRecord:
    """What the UNSTEERED model does on one item. Computed once, cached, and
    read from the cache thereafter — the optimization loop never runs an
    unsteered forward (plan §5)."""

    id: str
    role: str
    split: str
    selected: str                       # baseline argmax over the option set
    option_logprobs: dict               # option → joint logprob
    probabilities: list                 # p0 in DECLARED option order
    contrast: str | None = None         # target items: margin's reference option

    def to_dict(self) -> dict:
        return {"id": self.id, "role": self.role, "split": self.split,
                "selected": self.selected, "optionLogprobs": self.option_logprobs,
                "probabilities": self.probabilities, "contrast": self.contrast}

    @classmethod
    def from_dict(cls, payload: dict) -> "BaselineRecord":
        return cls(id=payload["id"], role=payload["role"],
                   split=payload["split"], selected=payload["selected"],
                   option_logprobs=payload["optionLogprobs"],
                   probabilities=payload["probabilities"],
                   contrast=payload.get("contrast"))


def baseline_prepass(model, items: list[PreparedItem],
                     config: OptVecTrainConfig) -> dict:
    """Score every item unsteered through the standard logprob instrument.

    Deliberately the SAME instrument the eval verb and every behavioral study
    use, not a bespoke forward: the baseline the objective is defined against
    must be the baseline the study would report.
    """
    records: dict[str, BaselineRecord] = {}
    for item in items:
        result = logprob.score_options(
            model, item.prompt, list(item.options),
            prompt_mode=config.prompt_mode, system_prompt=config.system_prompt,
            qwen_thinking_enabled=config.qwen_thinking_enabled)
        option_logprobs = {s.option: s.logprob for s in result.options}
        contrast: str | None = None
        if item.role == "target":
            # The margin needs a reference the target is measured AGAINST. The
            # baseline choice is that reference — except when the baseline
            # already picks the target, where logp(target) − logp(target) = 0
            # is a constant with no gradient; the strongest competing option is
            # then the honest reference and the item stays trainable.
            if result.selected != item.target:
                contrast = result.selected
            else:
                competitors = [o for o in item.options if o != item.target]
                contrast = max(competitors, key=lambda o: option_logprobs[o]) \
                    if competitors else None
        records[item.id] = BaselineRecord(
            id=item.id, role=item.role, split=item.split,
            selected=result.selected, option_logprobs=option_logprobs,
            probabilities=list(result.ordered_probabilities), contrast=contrast)
    return records


def write_baseline_cache(path: str, records: dict) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records.values():
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return path


def load_baseline_cache(path: str) -> dict:
    records: dict[str, BaselineRecord] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = BaselineRecord.from_dict(json.loads(line))
                records[record.id] = record
    return records


# ------------------------------------------------------------- batch plan


def _apportion(total: int, weights: list[float]) -> list[int]:
    """Largest-remainder apportionment of ``total`` over ``weights``."""
    positive = sum(w for w in weights if w > 0)
    if positive <= 0:
        return [0] * len(weights)
    exact = [total * w / positive if w > 0 else 0.0 for w in weights]
    counts = [int(math.floor(e)) for e in exact]
    remaining = total - sum(counts)
    order = sorted(range(len(weights)),
                   key=lambda i: (exact[i] - counts[i], weights[i]), reverse=True)
    for i in range(remaining):
        counts[order[i % len(order)]] += 1
    return counts


class _RolePool:
    """A reshuffled-on-exhaustion stream over one role's items."""

    def __init__(self, items: list[PreparedItem], rng: random.Random):
        self._items = list(items)
        self._rng = rng
        self._queue: list[PreparedItem] = []

    def __bool__(self) -> bool:
        return bool(self._items)

    def take(self, n: int) -> list[PreparedItem]:
        out: list[PreparedItem] = []
        while len(out) < n and self._items:
            if not self._queue:
                self._queue = list(self._items)
                self._rng.shuffle(self._queue)
            out.append(self._queue.pop())
        return out


# ------------------------------------------------------------- optimization


@dataclass
class ValMetrics:
    shift_rate: float
    mean_margin: float
    anchor_kl: float
    composite: float

    def to_dict(self) -> dict:
        return {"valShiftRate": self.shift_rate,
                "valMeanMargin": self.mean_margin,
                "valAnchorKL": self.anchor_kl,
                "valComposite": self.composite}


@torch.no_grad()
def evaluate_split(model, injector: TrainableVectorInjector | None,
                   target_items: list[PreparedItem],
                   anchor_items: list[PreparedItem], baselines: dict,
                   microbatch_size: int,
                   anchor_kl_budget: float | None) -> ValMetrics:
    """Val-split readout: shift rate + mean margin on targets, mean anchor KL.

    The composite is the shift rate with any anchor-KL overrun subtracted, so a
    checkpoint cannot win selection by trading the preservation constraint away.
    """
    shifts: list[float] = []
    margins: list[float] = []
    kls: list[float] = []
    for batch in _length_sorted_microbatches(target_items, microbatch_size):
        rows = answer_logit_rows(model, batch, injector)
        for i, item in enumerate(batch):
            logits = choice_logits(rows[i], item)
            baseline = baselines[item.id]
            target_index = item.target_index
            shifts.append(1.0 if int(torch.argmax(logits)) == target_index
                          else 0.0)
            if baseline.contrast is not None:
                contrast_index = item.options.index(baseline.contrast)
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                margins.append(float(logprobs[target_index]
                                     - logprobs[contrast_index]))
    for batch in _length_sorted_microbatches(anchor_items, microbatch_size):
        rows = answer_logit_rows(model, batch, injector)
        for i, item in enumerate(batch):
            p0 = torch.tensor(baselines[item.id].probabilities,
                              dtype=torch.float32, device=rows.device)
            kls.append(float(anchor_kl(choice_logits(rows[i], item), p0)))
    shift_rate = sum(shifts) / len(shifts) if shifts else 0.0
    mean_margin = sum(margins) / len(margins) if margins else 0.0
    mean_kl = sum(kls) / len(kls) if kls else 0.0
    overrun = 0.0 if anchor_kl_budget is None \
        else max(0.0, mean_kl - anchor_kl_budget)
    return ValMetrics(shift_rate=shift_rate, mean_margin=mean_margin,
                      anchor_kl=mean_kl, composite=shift_rate - overrun)


def _is_better(candidate: ValMetrics, best: dict | None) -> bool:
    """Checkpoint selection: composite first, mean margin as a TIE-BREAK only.

    The composite is a rate over a small val split, so it saturates and ties
    constantly — on a set the vector already flips, every later checkpoint
    would look identical to the first and early stopping would fire on a
    plateau that is really still improving. The margin breaks exact ties and
    nothing else: a checkpoint can never win on margin while losing shift or
    blowing the anchor budget.
    """
    if best is None:
        return True
    if candidate.composite > best["valComposite"] + 1e-12:
        return True
    if candidate.composite < best["valComposite"] - 1e-12:
        return False
    return candidate.mean_margin > best["valMeanMargin"] + 1e-12


def _length_sorted_microbatches(items: list[PreparedItem],
                                size: int) -> list[list[PreparedItem]]:
    """Length-bucketed microbatches: sorting by prompt length keeps padding —
    and the kept-logits tail — small."""
    ordered = sorted(items, key=lambda i: len(i.input_ids))
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]


#: Checkpoint file names. ``best-val`` is written only when a val split exists;
#: ``final-step`` only when one does not — the file name alone says which
#: selection produced the artifact.
BEST_VAL_CHECKPOINT = "best-val.safetensors"
FINAL_STEP_CHECKPOINT = "final-step.safetensors"


def _save_vector_checkpoint(path: str, vector: list) -> str:
    from safetensors.numpy import save_file
    import numpy as np
    save_file({"vector": np.asarray(vector, dtype=np.float32)}, path)
    return path


def optimize(model, config: OptVecTrainConfig, *, pools: dict,
             val_target: list[PreparedItem], val_anchor: list[PreparedItem],
             baselines: dict, alpha_absolute: float, run_directory: str,
             log: Callable[[str], None] | None = None) -> dict:
    """Run the optimization loop and return its provenance dict.

    ``pools`` maps role → training items. ``baselines`` is the CACHED pre-pass
    (this function never scores an unsteered forward). The returned dict names
    the selected checkpoint and its val metrics.

    Two modes, decided by the CONFIG (``datasets.targetVal``), never by an
    argument — so a run's mode is a property of its pinned recipe:

    * **val-selected** (targetVal declared): the historical path — periodic val
      evaluations, early stopping on patience, the best-composite checkpoint is
      the artifact, and ``chosenCheckpoint`` carries its val metrics.
    * **fixed-steps** (no targetVal): no val evaluation runs at all, no early
      stop can fire, the loop runs exactly ``stepsMax`` steps and the FINAL
      step is the artifact. ``chosenCheckpoint`` is then
      ``{"step": stepsMax, "selection": "finalStep"}`` — no ``val*`` metrics
      anywhere, so a reader cannot mistake it for a selected checkpoint.
    """
    def emit(message: str) -> None:
        if log is not None:
            log(message)

    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    model.model.eval()

    generator = torch.Generator().manual_seed(config.seed)
    injector = TrainableVectorInjector(
        layer=config.layer, hidden_size=model.hidden_size,
        alpha_absolute=alpha_absolute, position_mode=config.position_mode,
        device=model.device, generator=generator)
    priors = _load_prior_vectors(config, model.hidden_size,
                                 device=injector.u.device)

    checkpointing = _use_gradient_checkpointing(config, model)
    armed = _enable_gradient_checkpointing(model, emit) if checkpointing \
        else False

    optimizer = torch.optim.Adam([injector.u], lr=config.lr)
    scheduler = None
    if (config.lr_schedule or "").lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.steps_max)

    rng = random.Random(config.seed)
    role_pools = {role: _RolePool(pools.get(role, []), rng) for role in ROLES}
    weights = [config.mix_ratio[i] if role_pools[role] else 0.0
               for i, role in enumerate(ROLES)]
    lambdas = {"target": config.lambda_shift, "anchor": config.lambda_anchor,
               "capability": config.lambda_cap}
    for i, role in enumerate(ROLES):
        if lambdas[role] <= 0:
            weights[i] = 0.0
    if sum(weights) <= 0:
        raise OptVecConfigError(
            "no role contributes to the objective — every λ is zero or its "
            "pool is empty")

    select_on_val = config.datasets.selects_on_val
    if not select_on_val:
        emit(f"fixed-steps mode: no targetVal split, so no val evaluation "
             f"runs — training {config.steps_max} step(s) and taking the "
             "final checkpoint")

    metrics_path = os.path.join(run_directory, "metrics.jsonl")
    checkpoint_dir = os.path.join(run_directory, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best: dict | None = None
    last_improved_step = 0
    completed = 0
    stopped_early = False

    try:
        with open(metrics_path, "a", encoding="utf-8") as metrics:
            def record(payload: dict) -> None:
                metrics.write(json.dumps(payload, sort_keys=True) + "\n")
                metrics.flush()   # a killed job's partial curve is still readable

            for step in range(1, config.steps_max + 1):
                counts = _apportion(config.grad_accum_to_effective, weights)
                batch: list[PreparedItem] = []
                for i, role in enumerate(ROLES):
                    batch.extend(role_pools[role].take(counts[i]))
                per_role = {role: sum(1 for x in batch if x.role == role)
                            for role in ROLES}
                optimizer.zero_grad(set_to_none=True)
                totals = {role: 0.0 for role in ROLES}
                for micro in _length_sorted_microbatches(batch,
                                                         config.microbatch_size):
                    # backward() runs INSIDE the armed session: under gradient
                    # checkpointing it re-runs each block's forward, which must
                    # re-fire the injection hook (see answer_logit_session).
                    with answer_logit_session(model, micro, injector) as rows:
                        loss = rows.new_zeros((), dtype=torch.float32)
                        for i, item in enumerate(micro):
                            logits = choice_logits(rows[i], item)
                            baseline = baselines[item.id]
                            if item.role == "target":
                                if baseline.contrast is None:
                                    continue
                                term = shift_loss(
                                    logits, item.target_index,
                                    item.options.index(baseline.contrast),
                                    config.hinge_margin_nats)
                                weight = config.lambda_shift / max(1, per_role["target"])
                            elif item.role == "anchor":
                                p0 = torch.tensor(baseline.probabilities,
                                                  dtype=torch.float32,
                                                  device=logits.device)
                                term = anchor_kl(logits, p0)
                                weight = config.lambda_anchor / max(1, per_role["anchor"])
                            else:
                                term = capability_ce(logits, item.target_index)
                                weight = config.lambda_cap / max(1, per_role["capability"])
                            totals[item.role] += float(term.detach())
                            loss = loss + weight * term
                        if float(loss.detach()) != 0.0 or loss.requires_grad:
                            loss.backward()
                orth_value = 0.0
                if config.lambda_orth > 0 and priors:
                    orth = config.lambda_orth * orthogonality_penalty(
                        injector.vector(), priors)
                    orth_value = float(orth.detach())
                    orth.backward()
                grad_norm = float(injector.u.grad.norm()) \
                    if injector.u.grad is not None else 0.0
                lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                completed = step

                means = {role: (totals[role] / per_role[role]
                                if per_role[role] else 0.0) for role in ROLES}
                row = {"step": step, "lr": lr, "gradNormU": grad_norm,
                       "lossShift": means["target"], "lossAnchor": means["anchor"],
                       "lossCap": means["capability"], "lossOrth": orth_value,
                       "loss": (config.lambda_shift * means["target"]
                                + config.lambda_anchor * means["anchor"]
                                + config.lambda_cap * means["capability"]
                                + orth_value),
                       "items": per_role}

                if select_on_val and (step % config.val_every == 0
                                      or step == config.steps_max):
                    val = evaluate_split(model, injector, val_target, val_anchor,
                                         baselines, config.microbatch_size,
                                         config.anchor_kl_budget)
                    row.update(val.to_dict())
                    if _is_better(val, best):
                        best = {"step": step, **val.to_dict()}
                        last_improved_step = step
                        _save_vector_checkpoint(
                            os.path.join(checkpoint_dir, BEST_VAL_CHECKPOINT),
                            injector.constrained_vector())
                record(row)
                emit(f"step {step}/{config.steps_max}: loss {row['loss']:.4f} "
                     f"(shift {means['target']:.4f}) ‖∇u‖ {grad_norm:.4f}")

                if step % config.checkpoint_every == 0:
                    _save_vector_checkpoint(
                        os.path.join(checkpoint_dir, f"step-{step}.safetensors"),
                        injector.constrained_vector())

                gap = step - last_improved_step
                if select_on_val and best is not None \
                        and gap >= config.early_stop_patience:
                    stopped_early = True
                    emit(f"early stop at step {step}: no val improvement in "
                         f"{gap} steps")
                    break
    finally:
        if checkpointing:
            # The model may be registry-resident and shared with inference
            # paths; recomputation, the layer training flags, and the
            # input-grads hook must not outlive this loop.
            _disable_gradient_checkpointing(model)

    if select_on_val:
        artifact_path = os.path.join(checkpoint_dir, BEST_VAL_CHECKPOINT)
        chosen = best
        if best is None or not os.path.exists(artifact_path):
            # No val evaluation ever ran (impossible with val_every ≤ steps_max,
            # but the artifact must never be an unselected checkpoint by
            # accident).
            raise OptVecConfigError(
                "no validation evaluation ran, so no checkpoint was selected — "
                "valEvery must be ≤ stepsMax")
    else:
        # Fixed-steps: the last state of `u` IS the artifact. Written here, not
        # inside the loop, so the file that exists is the one that was chosen.
        artifact_path = _save_vector_checkpoint(
            os.path.join(checkpoint_dir, FINAL_STEP_CHECKPOINT),
            injector.constrained_vector())
        chosen = {"step": completed, "selection": "finalStep"}
    from safetensors.numpy import load_file
    vector = load_file(artifact_path)["vector"].astype("float32").tolist()
    return {"steps": completed, "stoppedEarly": stopped_early,
            "chosenCheckpoint": chosen, "vector": vector,
            "alphaAbsolute": alpha_absolute,
            "gradientCheckpointingArmed": armed,
            "metricsFile": os.path.basename(metrics_path)}


def _use_gradient_checkpointing(config: OptVecTrainConfig, model) -> bool:
    if config.gradient_checkpointing is not None:
        return bool(config.gradient_checkpointing)
    return str(model.device).startswith("cuda")


def _enable_gradient_checkpointing(model, emit: Callable[[str], None]) -> bool:
    """Arm recomputation for the training loop; returns whether it is ARMED
    (stamped into the run's record — a reader must be able to tell a
    recomputing run from one that silently kept every activation).

    Best-effort: recomputation is a memory optimization, never a correctness
    requirement, so a model that cannot do it trains without it.

    ``use_reentrant=False`` is required here: with every model parameter frozen
    the reentrant implementation sees no input requiring grad and silently
    drops the recomputed graph — the exact configuration this loop runs in.

    Two repairs after HF's ``gradient_checkpointing_enable`` (both observed
    live on a hard-item campaign, 2026-08-11 — 78 GB allocated on an 80 GB
    A100 by a single ~3.5k-token item):

    * **The layer gate is ``self.gradient_checkpointing and self.training``**,
      and this loop deliberately runs the model in ``eval()``. Flip the
      TRAINING FLAG of the decoder layers alone — the flag, never
      ``.train()``, so every submodule keeps eval-mode numerics (the
      training-gated dropout paths inside attention/MLP stay off) while the
      checkpointing gate opens. Without this, enable() is a silent no-op:
      nothing recomputes and every activation of every layer above the
      injection is held for backward, quadratic-in-sequence attention state
      included.
    * **HF's enable() also marks the input embeddings' output as requiring
      grad** (a PEFT-era side effect, unconditional for ``input_ids`` models
      in transformers 5.x). This loop trains no model parameter, and that
      hook extends the autograd tape from layer 0 instead of the injection
      layer — undoing the truncated backward that makes 27B affordable
      (``test_backward_is_truncated_at_the_injection_layer``). Undo it.
    """
    enable = getattr(model.model, "gradient_checkpointing_enable", None)
    if enable is None:
        emit("gradient checkpointing unavailable on this model — continuing")
        return False
    try:
        enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        try:
            enable()
        except Exception as exc:  # noqa: BLE001 - never blocks training
            emit(f"gradient checkpointing could not be enabled ({exc})")
            return False
    except Exception as exc:  # noqa: BLE001 - never blocks training
        emit(f"gradient checkpointing could not be enabled ({exc})")
        return False
    disable_input_grads = getattr(model.model, "disable_input_require_grads",
                                  None)
    if disable_input_grads is not None:
        try:
            disable_input_grads()
        except (AttributeError, RuntimeError):
            # Older transformers never installed the hook; nothing to undo.
            pass
    for layer in model.hooked.layers:
        layer.training = True
    return True


def _disable_gradient_checkpointing(model) -> None:
    """Undo everything :func:`_enable_gradient_checkpointing` armed. The model
    may be registry-resident and shared with inference paths, so the layer
    training flags and recomputation must not outlive the training loop."""
    disable = getattr(model.model, "gradient_checkpointing_disable", None)
    if disable is not None:
        disable()
    model.model.eval()   # restores every module's training flag in one sweep


def _load_prior_vectors(config: OptVecTrainConfig, hidden_size: int,
                        device: torch.device | None = None
                        ) -> list[torch.Tensor]:
    """The S3 priors, each one the row at ``config.layer``, on ``device``.

    ``device`` must be where the injector's vector lives: the penalty is
    ``dot(v, prior)`` and a CPU prior against a CUDA/MPS ``v`` is a step-1
    RuntimeError. And the row must be NONZERO — OptVec artifacts are zeros
    everywhere except their own layer, so a prior trained at another layer
    would contribute cos = 0 to every step: the run record would claim
    orthogonality pressure that never existed.
    """
    priors: list[torch.Tensor] = []
    for reference in config.prior_vector_paths:
        resolved = paths.resolve_artifact(reference)
        vectors, _sidecar = vector_store.load(os.path.dirname(resolved),
                                              os.path.basename(resolved))
        if config.layer >= vectors.layer_count:
            raise OptVecConfigError(
                f"prior vector '{reference}' has {vectors.layer_count} layers; "
                f"this run's layer is {config.layer}")
        row = vectors.per_layer[config.layer]
        if len(row) != hidden_size:
            raise OptVecConfigError(
                f"prior vector '{reference}' is {len(row)}-dimensional, model "
                f"hidden size is {hidden_size}")
        if not any(row):
            raise OptVecConfigError(
                f"prior vector '{reference}' is all zeros at layer "
                f"{config.layer} — it would contribute exactly zero "
                "orthogonality pressure while lambdaOrth stamps the run as "
                "S3. An OptVec artifact is nonzero only at its own "
                "optimization layer; use a prior trained at this run's layer, "
                "or drop it from priorVectorPaths")
        priors.append(torch.tensor(row, dtype=torch.float32, device=device))
    return priors


# ------------------------------------------------------------- calibration


def resolve_alpha(config: OptVecTrainConfig) -> tuple[float, dict, dict]:
    """``(alpha_absolute, provenance, residual_norm_provenance)``.

    α is denominated in units of the residual-stream norm at the layer — the
    convention the whole vector library uses — so the driver needs that
    denominator. It is either given (``residualNorm``) or read from an existing
    artifact's ``residualNormPerLayer`` (``residualNormArtifact``); no new
    calibration artifact type is created here. ``alphaAbsolute`` bypasses the
    conversion entirely and records that it did.

    The third element is the DONOR's denominator provenance — its per-layer
    residual-norm table and the corpus/convention stamps behind it — so
    :func:`_save_artifact` can put it in the artifact instead of leaving dose
    comparability to process trust (open-issues §24). Empty when there is no
    donor: an ``alphaAbsolute`` run has no denominator at all, and a bare
    ``residualNorm`` scalar has a denominator whose corpus nobody recorded.
    """
    if config.alpha_absolute is not None:
        return float(config.alpha_absolute), {
            "alphaAbsolute": float(config.alpha_absolute),
            "denomination": "absolute",
            "note": "alphaAbsolute given directly; not denominated in norm units"}, {}
    norm = config.residual_norm
    source = "config.residualNorm"
    residual_provenance: dict = {}
    if norm is None:
        resolved = paths.resolve_artifact(config.residual_norm_artifact or "")
        _vectors, sidecar = vector_store.load(os.path.dirname(resolved),
                                              os.path.basename(resolved))
        per_layer = sidecar.residualNormPerLayer
        if not per_layer or config.layer >= len(per_layer):
            raise OptVecConfigError(
                f"residualNormArtifact '{config.residual_norm_artifact}' has no "
                f"residualNormPerLayer at layer {config.layer} — run the "
                "norm backfill against the pinned neutral corpus first")
        norm = float(per_layer[config.layer])
        source = (f"{config.residual_norm_artifact}"
                  f"#residualNormPerLayer[{config.layer}]"
                  f" ({sidecar.residualNormSource or 'unrecorded source'})")
        # Absent-not-null throughout, matching the sidecar family convention:
        # a donor that never recorded a corpus hash or a convention leaves the
        # key out rather than writing a null anyone could mistake for a value.
        residual_provenance = {"residualNormPerLayer": [float(n) for n in per_layer],
                               "residualNormArtifact": config.residual_norm_artifact}
        if sidecar.residualNormSource:
            residual_provenance["residualNormSource"] = sidecar.residualNormSource
        if sidecar.neutralCorpusHash:
            residual_provenance["neutralCorpusHash"] = sidecar.neutralCorpusHash
        if sidecar.residualNormConvention:
            residual_provenance["residualNormConvention"] = sidecar.residualNormConvention
    if not norm or norm <= 0:
        raise OptVecConfigError(
            f"residual norm at layer {config.layer} is {norm} — α cannot be "
            "denominated against a non-positive norm")
    return float(config.alpha_norm_factor) * float(norm), {
        "alphaNormFactor": float(config.alpha_norm_factor),
        "residualNorm": float(norm), "residualNormSource": source,
        "alphaAbsolute": float(config.alpha_norm_factor) * float(norm),
        "denomination": "residual-norm-units"}, residual_provenance


# ------------------------------------------------------------------- driver


def train(config: OptVecTrainConfig, *, model=None,
          log: Callable[[str], None] | None = None) -> dict:
    """Optimize one vector and write its immutable run directory.

    ``model`` may be an already-loaded :class:`SteeredModel` (the server passes
    its registry-resident model rather than paying a second load); ``None``
    loads per the config. Returns the run's provenance dict.
    """
    def emit(message: str) -> None:
        if log is not None:
            log(message)

    name = (config.name or "").strip() or \
        f"{os.path.basename(config.model_id)}-L{config.layer}"
    run_directory = paths.make_unique_run_directory(f"optvec-{name}")
    notes: dict = {"stage": "starting", "objective": _objective_notes(config)}
    dtype_name: str | None = None

    try:
        if model is None:
            from ..steering import model_loader
            device = model_loader.resolve_device(config.device)
            dtype_name = resolve_training_dtype(
                config.dtype, device, config.model_id, warn=emit)
            emit(f"loading {config.model_id} on {device} ({dtype_name})")
            model = model_loader.load(config.model_id, config.revision,
                                      dtype=dtype_name, device=device)
        dtype_name = getattr(model, "dtype", None) or dtype_name

        alpha_absolute, alpha_notes, residual_norm_provenance = resolve_alpha(config)
        notes["alpha"] = alpha_notes
        notes["residualNormProvenance"] = residual_norm_provenance

        prepared, dataset_notes = _prepare_all(model, config)
        notes["datasets"] = dataset_notes
        degenerate = [
            label for label, fraction in
            (dataset_notes.get("shuffleEffectiveChangeFraction") or {}).items()
            if fraction == 0.0]
        if degenerate:
            emit("WARNING: the S0 shuffled-target null changed ZERO labels "
                 f"on {', '.join(sorted(degenerate))} — every item declares "
                 "the same target, so this 'null' optimizes the treatment "
                 "objective under a null's name. The run proceeds and stamps "
                 "shuffleEffectiveChangeFraction, but it certifies nothing "
                 "as a null; author a dataset with both labels present")
        if not config.datasets.selects_on_val:
            emit("NOTE: no targetVal split — fixed-steps mode. Nothing is "
                 "selected on behavior here: the artifact is the final step, "
                 "stamped chosenCheckpoint.selection = 'finalStep'. Its "
                 "evidence is the test-side eval, not a val curve")
        elif config.lambda_anchor > 0 and not prepared.get("anchorVal"):
            emit("WARNING: lambdaAnchor > 0 but the config declares no "
                 "anchorVal split — checkpoint selection sees no anchor-KL "
                 "constraint and is TARGET-ONLY; the chosen checkpoint is "
                 "target-selected, awaiting test-side qualification "
                 "(stamped as training.selectionScope)")
        notes["stage"] = "baseline"

        baseline_items = [item for group in prepared.values() for item in group]
        baselines = baseline_prepass(model, baseline_items, config)
        write_baseline_cache(os.path.join(run_directory, BASELINE_CACHE),
                             baselines)
        emit(f"baseline pre-pass cached for {len(baselines)} item(s)")
        notes["stage"] = "optimizing"

        result = optimize(
            model, config,
            pools={"target": prepared["targetTrain"],
                   "anchor": prepared.get("anchorTrain", []),
                   "capability": prepared.get("capabilityTrain", [])},
            val_target=prepared.get("targetVal", []),
            val_anchor=prepared.get("anchorVal", []),
            baselines=baselines, alpha_absolute=alpha_absolute,
            run_directory=run_directory, log=log)

        notes["training"] = {
            "steps": result["steps"], "stepsMax": config.steps_max,
            "stoppedEarly": result["stoppedEarly"],
            "chosenCheckpoint": result["chosenCheckpoint"],
            # What the checkpoint was SELECTED on (review finding
            # 2026-08-10): capability never joins selection (the reader
            # firewall keeps capability data examiner-side), and without an
            # anchorVal split the anchor constraint drops out of the
            # composite too. Describe checkpoints accordingly —
            # "target(+anchor)-selected, awaiting test-side qualification".
            "selectionScope": (
                # Fixed-steps selects on NOTHING; say so in the same field
                # rather than letting "targetVal-only" imply a val curve
                # that never ran.
                "none-finalStep" if not config.datasets.selects_on_val
                else ("targetVal+anchorVal" if prepared.get("anchorVal")
                      else "targetVal-only")),
            "lr": config.lr, "lrSchedule": config.lr_schedule,
            "microbatchSize": config.microbatch_size,
            "gradAccumToEffective": config.grad_accum_to_effective,
            "mixRatio": list(config.mix_ratio), "seed": config.seed,
            # What actually happened, not the policy request: False when the
            # policy declined it OR when arming failed — a run that kept every
            # activation must not record itself as a recomputing one.
            "gradientCheckpointing": result["gradientCheckpointingArmed"],
        }
        if config.item_filter is not None:
            # Per-item runs only (WP-S4a). The α→0 gradient direction is the
            # cheap answer to "which way does this item's margin actually
            # want the residual to move?"; the cosine says how much of the
            # optimized direction is that answer and how much is everything
            # else the optimization found on the way. Computing it costs one
            # forward+backward per filtered item, which is affordable exactly
            # because the filter made the set small — hence the gate. It is a
            # DIAGNOSTIC, not a gate: nothing refuses on its value.
            notes["training"]["cosineToGradient"] = _cosine_to_gradient(
                model, config, prepared["targetTrain"], baselines,
                result["vector"], emit)
        notes["stage"] = "saving"

        artifact = _save_artifact(model, config, result, notes, name,
                                  run_directory)
        notes["artifact"] = artifact
        notes["stage"] = "complete"
        emit(f"optvec vector → {artifact['vectorArtifactID']}")
        return {"runDirectory": run_directory, **artifact,
                "chosenCheckpoint": result["chosenCheckpoint"],
                "steps": result["steps"]}
    finally:
        # Written LAST so the notes are complete, and in `finally` so a crashed
        # run still records what it was doing (the run directory is immutable
        # and write_run_config never rewrites, so the success path's call —
        # this one — is the only one).
        write_run_config(run_directory, RUN_TYPE, model_id=config.model_id,
                         revision=getattr(model, "revision", None)
                         or config.revision,
                         dtype=dtype_name, notes=notes)


def _cosine_to_gradient(model, config: OptVecTrainConfig,
                        items: list[PreparedItem], baselines: dict,
                        vector: list, emit: Callable[[str], None]) -> float | None:
    """Cosine between the trained direction and the α→0 mean-margin gradient
    over the filtered training items (``None`` when the gradient is degenerate
    — a zero gradient has no direction, and 0.0 would read as orthogonality).

    Imported lazily: :mod:`optvec_gradient` imports this module for its item
    preparation and baseline machinery, so a module-level import here would be
    a cycle.
    """
    from .optvec_gradient import mean_margin_gradient

    gradient = mean_margin_gradient(model, items, baselines,
                                    layer=config.layer,
                                    position_mode=config.position_mode)
    if gradient is None:
        emit("cosineToGradient: the mean-margin gradient is zero (or no "
             "filtered item has a usable contrast) — no direction to compare")
        return None
    dot = sum(a * b for a, b in zip(vector, gradient))
    norm_v = math.sqrt(sum(x * x for x in vector))
    norm_g = math.sqrt(sum(x * x for x in gradient))
    if norm_v == 0.0 or norm_g == 0.0:
        return None
    value = float(dot / (norm_v * norm_g))
    emit(f"cosineToGradient: {value:.4f} (trained direction vs the α→0 "
         f"mean-margin gradient over {len(items)} filtered item(s))")
    return value


def _objective_notes(config: OptVecTrainConfig) -> dict:
    return {"lambdaShift": config.lambda_shift,
            "lambdaAnchor": config.lambda_anchor,
            "lambdaCap": config.lambda_cap,
            "lambdaOrth": config.lambda_orth,
            "hingeMarginNats": config.hinge_margin_nats,
            "positionMode": config.position_mode,
            "anchorKLBudget": config.anchor_kl_budget,
            "layer": config.layer,
            "priorVectorPaths": list(config.prior_vector_paths),
            "shuffleTargetLabels": config.shuffle_target_labels,
            "claim": "sufficiency"}


def _prepare_all(model, config: OptVecTrainConfig) -> tuple[dict, dict]:
    """Load, hash-verify, render and tokenize every declared split."""
    specs = [("targetTrain", config.datasets.target_train, "target", "train"),
             ("targetVal", config.datasets.target_val, "target", "val"),
             ("anchorTrain", config.datasets.anchor_train, "anchor", "train"),
             ("anchorVal", config.datasets.anchor_val, "anchor", "val"),
             ("capabilityTrain", config.datasets.capability_train,
              "capability", "train")]
    prepared: dict[str, list[PreparedItem]] = {}
    notes: dict = {"files": {}, "counts": {}}
    hashes: list[str] = []
    seen_ids: dict[str, str] = {}
    for label, ref, role, split in specs:
        if ref is None:
            continue
        rows = load_dataset(ref, label)
        if config.item_filter is not None and label == "targetTrain":
            rows = apply_item_filter(rows, config.item_filter, label)
            # A plain LIST of ids, deliberately: this block is copied verbatim
            # into the artifact's sidecar, and the per-item readers downstream
            # (optvec_geometry.item_of) read "the item this vector was trained
            # on" out of a single-id itemFilter list. The application facts
            # ride alongside rather than wrapping it.
            notes["itemFilter"] = list(config.item_filter)
            notes["itemFilterApplication"] = {
                "appliedTo": "targetTrain",
                "selectedCount": len(rows),
                "note": "gradients are taken on these items only; every other "
                        "declared split is unfiltered (item ids are unique "
                        "bundle-wide, so val/anchor/capability ids are "
                        "disjoint from train ids by construction). The "
                        "file's SHA-256 pin is unchanged — the filter is "
                        "config data, not an edit to the pinned bytes"}
        items = prepare_items(model, rows, role=role, split=split,
                              config=config, declared=label)
        for item in items:
            if item.id in seen_ids:
                raise OptVecDataError(
                    f"item id '{item.id}' appears in both '{seen_ids[item.id]}' "
                    f"and '{label}' — baselines are keyed by id, and a split "
                    "that overlaps another is a firewall breach besides")
            seen_ids[item.id] = label
        prepared[label] = items
        notes["files"][label] = ref.to_dict()
        notes["counts"][label] = len(items)
        hashes.append(ref.sha256)

    if config.shuffle_target_labels:
        permutations = {}
        changed = {}
        # Only the splits that exist: fixed-steps mode has no targetVal.
        for label in [x for x in ("targetTrain", "targetVal") if x in prepared]:
            original = [item.target for item in prepared[label]]
            relabeled, order = shuffled_target_labels(prepared[label],
                                                      config.seed)
            prepared[label] = relabeled
            permutations[label] = order
            # How much of a null the null actually is: the fraction of items
            # whose effective target moved. Identity (all items declaring
            # one label) means S0 optimizes the TREATMENT objective under a
            # null's name — stamped here, warned about by the caller.
            changed[label] = (
                sum(1 for before, after in zip(original, relabeled)
                    if after.target != before) / len(original)
                if original else 0.0)
        notes["shuffleTargetLabels"] = True
        notes["targetLabelPermutation"] = permutations
        notes["shuffleEffectiveChangeFraction"] = changed

    notes["compositeHash"] = composite_dataset_hash(hashes)
    return prepared, notes


def _save_artifact(model, config: OptVecTrainConfig, result: dict, notes: dict,
                   name: str, run_directory: str) -> dict:
    """Write the selected vector as an ordinary vector artifact + sidecar.

    The artifact is full-depth (one row per decoder layer) with the optimized
    direction at ``config.layer`` and ZEROS elsewhere: the injection path
    selects a layer by index, so a short artifact would either mis-index or be
    refused, and a nonzero row at an unoptimized layer would be a direction
    nothing certifies. ``optvec.layer`` in the sidecar names the only layer
    this artifact means anything at.
    """
    layer_count = model.num_layers
    if config.layer >= layer_count:
        raise OptVecConfigError(
            f"layer {config.layer} is out of range for a {layer_count}-layer "
            f"model")
    hidden = model.hidden_size
    per_layer = [[0.0] * hidden for _ in range(layer_count)]
    per_layer[config.layer] = list(result["vector"])
    vectors = ConceptVectors(per_layer=per_layer)
    from datetime import datetime, timezone
    sidecar = SteeringVectorSidecar(
        modelID=config.model_id,
        concept=name,
        stimulusSetHash=STIMULUS_PREFIX + notes["datasets"]["compositeHash"],
        layerCount=vectors.layer_count,
        hiddenSize=vectors.hidden_size,
        normsPerLayer=[vectors.norm(i) for i in range(vectors.layer_count)],
        extractionDate=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        revision=getattr(model, "revision", None) or config.revision,
        substrate=vector_store.SUBSTRATE,
        extractionMethod=EXTRACTION_METHOD,
        recipeMethod=EXTRACTION_METHOD,
        # Born WITHOUT residual norms, like J-lens directions and Gemma Scope
        # imports: run the existing norm backfill against the pinned neutral
        # corpus before any alphaInNormUnits condition uses this vector.
        residualNormPerLayer=None,
    )
    vector_store.save(vectors, sidecar, run_directory, name)

    # Additive provenance beside the ordinary sidecar fields (the jlens/derive
    # idiom): written after save() so the base contract is exactly what every
    # other artifact writes and this is a strict extension of it.
    sidecar_path = os.path.join(run_directory, f"{name}.json")
    with open(sidecar_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    alpha_notes = notes.get("alpha") or {}
    payload["optvec"] = {
        "layer": config.layer,
        "alphaAbsolute": result["alphaAbsolute"],
        "alpha": alpha_notes,
        # --- §24: the dose is now CHECKABLE from the artifact ---------------
        #
        # ``vectorPackaging`` is a WARNING, not a label. This family stores
        # the vector PRE-SCALED to its full trained magnitude: the row at
        # ``layer`` already has ``alphaAbsolute`` baked into it, and injecting
        # it means injecting it AS IS, at coefficient 1.
        #
        # THE TRAP: every other artifact family stores a direction that a
        # consumer scales by its own α, using ``residualNormPerLayer`` as the
        # denominator. A consumer that treats an OptVec artifact the same way
        # — reading a norms slot as a denominator, or re-normalizing the row
        # and applying its own α — changes the dose by ORDERS OF MAGNITUDE
        # (the trained magnitude at L43 of the rve-1 vectors is ~4935 against
        # a residual norm of ~49351; α=1 norm units against a re-normalized
        # row is a 10× overdose, and α applied on top of the pre-scaled row is
        # a 4935× one). Read this marker before scaling anything.
        "vectorPackaging": "preScaledFullMagnitude",
        # The α block promoted to where a reader looks first (§24: it used to
        # live only in eval-run records, so cross-family dose comparability
        # rested on process trust). ``alphaNormFactor`` is absent on an
        # ``alphaAbsolute`` run — there was no denominator, so there is no
        # factor, and absent says that more honestly than 0 would.
        **({"alphaNormFactor": alpha_notes["alphaNormFactor"]}
           if "alphaNormFactor" in alpha_notes else {}),
        # The DONOR's denominator, carried whole: its per-layer residual-norm
        # table, the neutral corpus it was measured on, and the averaging
        # CONVENTION behind it (``wholeCorpusMean-v1``). Deliberately nested
        # here rather than written to the top-level ``residualNormPerLayer``:
        # that key means "denominator for MY direction" everywhere else, and
        # on a pre-scaled artifact it would invite exactly the trap above.
        # Absent when α was given absolutely or as a bare scalar — no donor,
        # nothing to attribute.
        **({"residualNorm": notes["residualNormProvenance"]}
           if notes.get("residualNormProvenance") else {}),
        "objective": notes.get("objective"),
        "datasets": notes.get("datasets"),
        "training": notes.get("training"),
        "seed": config.seed,
        "runID": os.path.basename(os.path.normpath(run_directory)),
        "substrate": vector_store.SUBSTRATE,
        "gitSHA": _git_sha(),
        "claim": "sufficiency",
    }
    tmp = sidecar_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, sidecar_path)
    return {"name": name,
            "vectorArtifactID": os.path.join(run_directory, name),
            "layer": config.layer,
            "stimulusSetHash": sidecar.stimulusSetHash,
            "residualNormsPresent": False}


def _git_sha() -> str | None:
    """The engine commit, when this tree is a git checkout. Best effort — a
    cluster deployment from a tarball legitimately has none."""
    import subprocess
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha or None
