"""OptVec eval — the TEST-split behavioral readout for an optimized vector
(plan of record: ``docs/OPTVEC-OPTIMIZED-INJECTION-VECTORS-PLAN.md``, WP4).

Training (WP2) sees the *train* split and selects checkpoints on the *val*
split. This verb reads the *test* split, which nothing has selected on — the
firewall rule OptVec runs under is the funnel's, not the pre-registration one:
**selection on behavior is allowed; citing the data you selected on is not**
(plan §3). So the config here CANNOT NAME a train or val file: a key like
``targetTrain`` refuses with the firewall named, rather than quietly producing
numbers that are the maximum of many noisy val evaluations.

What it produces, per α multiple of the vector's own α (dose-response at
{0.25, 0.5, 1, 1.5, 2}× by default, plan §4):

* **target** — shift rate (argmax flips to the declared target) and mean
  log-odds movement toward it, each paired to the IN-RUN α=0 baseline;
* **anchor** — flip rate and mean ``KL(p0 ‖ p_α)`` over the choice
  distribution (confidence, not merely argmax);
* **capability** — accuracy against the declared correct option;
* **fluency** — mean per-token logprob of a pinned neutral text set under
  steering;
* **library comparison** — cosine against every named library vector at the
  same layer, each with a percentile against a seeded random-direction null at
  this ``d_model``.

Everything behavioral is scored through the EXISTING
:func:`steerlab_server.experiment.logprob.score_options` with the deployed
``CellInjection`` path — stepped KV-cache scoring, injection at the true prompt
end and every option step. The records therefore carry the same fields every
other behavioral study writes and are directly commensurable with them.

Why the fluency guard injects at EVERY position while the behavioral arms do
not
--------------------------------------------------------------------------
The deployed ``VectorInjector`` fires at the last position of each pass: the
true prompt end, then one decode step at a time. Over a *generated* text that
means every token of the generation was produced with the vector present —
each was, in its turn, the last position of its own pass. A teacher-forced
single pass over an already-written text has no decode steps, so injecting only
at its final position would steer nothing that is actually scored (the only
logits it changes predict the token AFTER the text). All-position injection is
what reproduces "the vector was present at every position that was generated",
and it is therefore the decode-identical semantics for a fluency readout — the
opposite of the behavioral arms, where the answer token is read at one position
and the deployed gate is exactly right. Hence
:class:`~steerlab_server.steering.trainable_injector.TrainableVectorInjector`
in ``position_mode="all"`` under ``torch.no_grad`` (it is the only injector in
the engine that can apply across positions; its constructor renormalizes to a
declared absolute norm, which reproduces the scaled vector exactly).

Claim grammar (binding, plan §1): SUFFICIENCY. Nothing here certifies that the
direction is necessary or unique, and ``eval.json`` is screen-grade evidence —
citable numbers come from a confirm-style behavioral study with the vector
attached as a pinned artifact.
"""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable

import torch

from ..steering import vector_store
from ..steering.trainable_injector import TrainableVectorInjector
from . import logprob, paths, prompt_render
from .generate import CellInjection
from .run_config import write_run_config
from .sweep_selection import ChoiceRow, load_choice_rows
from .optvec_train import resolve_data_path as optvec_train_resolve

#: Run type stamped into ``config.json``. Its OWN immutable run directory — the
#: training run is never written into (it is a finished, immutable record, and
#: an eval is a separate measurement that may be repeated).
RUN_TYPE = "optvec-eval"

#: Dose grid, plan §4: the 0.10 norm-unit factor is not swept inside the
#: optimization, it is read post-hoc at these multiples of the vector's own α.
DEFAULT_ALPHA_MULTIPLES = (0.25, 0.5, 1.0, 1.5, 2.0)

#: Random directions drawn for the library-cosine null. A cosine against a
#: library vector means nothing without the distribution of cosines a random
#: direction at this ``d_model`` achieves.
DEFAULT_NULL_SAMPLES = 1000

#: Rows of the library table reported by ``|cosine|``.
LIBRARY_TOP_K = 10

#: Item roles. A role is what an item MEASURES, not what it is about.
ROLES = ("target", "anchor", "capability")

EVAL_JSON = "eval.json"
EVAL_RECORDS = "eval-records.jsonl"

SCHEMA_VERSION = 1


class OptVecEvalConfigError(ValueError):
    """An eval config that cannot be run as written."""


class OptVecEvalDataError(ValueError):
    """A dataset or neutral text file that cannot be evaluated (hash drift…)."""


class OptVecArtifactError(ValueError):
    """The named artifact is not an evaluable OptVec vector."""


# ------------------------------------------------------------------- config


@dataclass(frozen=True)
class FileRef:
    """One hashed input file. The hash is the pin: an eval whose record cannot
    say which bytes it read is not evidence of anything."""

    path: str
    sha256: str

    @classmethod
    def from_dict(cls, value: dict, label: str) -> "FileRef":
        if not isinstance(value, dict):
            raise OptVecEvalConfigError(
                f"{label} must be an object with 'path' and 'sha256' "
                f"(got {value!r})")
        unknown = sorted(set(value) - {"path", "sha256"})
        if unknown:
            raise OptVecEvalConfigError(
                f"unknown {label} key(s): " + ", ".join(unknown))
        path = value.get("path")
        digest = value.get("sha256")
        if not isinstance(path, str) or not path.strip():
            raise OptVecEvalConfigError(f"{label}.path must be a file path")
        if not isinstance(digest, str) or not digest.strip():
            raise OptVecEvalConfigError(
                f"{label}.sha256 must be the SHA-256 of the file's raw bytes")
        return cls(path=path, sha256=digest.strip().lower())

    def to_dict(self) -> dict:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class EvalDatasets:
    """Test-split item sets. ``targetTest`` is required; the other two are the
    preservation and capability budgets, absent when a run declares none."""

    target_test: FileRef
    anchor_test: FileRef | None = None
    capability_eval: FileRef | None = None

    def to_dict(self) -> dict:
        out = {"targetTest": self.target_test.to_dict()}
        for key, ref in (("anchorTest", self.anchor_test),
                         ("capabilityEval", self.capability_eval)):
            if ref is not None:
                out[key] = ref.to_dict()
        return out


_DATASET_KEYS = {"targetTest": "target_test", "anchorTest": "anchor_test",
                 "capabilityEval": "capability_eval"}

#: Which objective a declared set MEASURES. All three are ordinary choice-row
#: JSONL — the schema's ``target`` field means "the option to shift onto" for a
#: target item and "the objectively correct option" for a capability item, and
#: is unused for an anchor (whose whole claim is about its own baseline). One
#: loader, three readings, no new schema (plan §4).
_DATASET_ROLES = {"targetTest": "target", "anchorTest": "anchor",
                  "capabilityEval": "capability"}

_CONFIG_KEYS = {
    "vectorArtifact": "vector_artifact",
    "modelID": "model_id", "revision": "revision", "name": "name",
    "device": "device", "dtype": "dtype",
    "alphaMultiples": "alpha_multiples",
    "libraryVectorPaths": "library_vector_paths",
    "nullSamples": "null_samples", "seed": "seed",
    "microbatchSize": "microbatch_size",
    "promptMode": "prompt_mode", "systemPrompt": "system_prompt",
    "qwenThinkingEnabled": "qwen_thinking_enabled",
}


def _refuse_selection_data_keys(keys, where: str) -> None:
    """The firewall, enforced at the config boundary.

    A train or val file named anywhere in an eval config is refused with the
    rule quoted, not silently ignored: gradients ran on train and checkpoint
    selection ran on val, so an "effect size" measured on either is the maximum
    of many noisy evaluations rather than a held-out number.
    """
    offenders = sorted(k for k in keys
                       if k.endswith("Train") or k.endswith("Val")
                       or k in ("train", "val"))
    if not offenders:
        return
    raise OptVecEvalConfigError(
        f"{where}: " + ", ".join(offenders) + " — OptVec eval reads the TEST "
        "split ONLY and must be unable to see the data selection ran on "
        "(plan §3: selection on behavior is allowed; citing the data you "
        "selected on is not — train contributed gradients, val carried "
        "checkpoint selection). Point this config at the held-out test files "
        "(targetTest / anchorTest / capabilityEval).")


@dataclass
class OptVecEvalConfig:
    vector_artifact: str
    datasets: EvalDatasets
    neutral_texts: FileRef | None = None
    model_id: str | None = None
    revision: str | None = None
    name: str | None = None
    device: str | None = None
    dtype: str = "auto"

    alpha_multiples: tuple[float, ...] = DEFAULT_ALPHA_MULTIPLES
    library_vector_paths: list[str] = field(default_factory=list)
    null_samples: int = DEFAULT_NULL_SAMPLES
    seed: int = 0
    microbatch_size: int = 4

    prompt_mode: str = prompt_render.CHAT_ASSISTANT
    system_prompt: str | None = None
    qwen_thinking_enabled: bool = False

    def __post_init__(self) -> None:
        if not (self.vector_artifact or "").strip():
            raise OptVecEvalConfigError(
                "'vectorArtifact' is required — the extension-less path of the "
                "optvec artifact to evaluate")
        multiples = tuple(float(m) for m in self.alpha_multiples)
        if not multiples:
            raise OptVecEvalConfigError(
                "alphaMultiples must name at least one dose")
        if any(m <= 0 for m in multiples):
            raise OptVecEvalConfigError(
                "alphaMultiples must be positive — α=0 is always evaluated as "
                "the in-run baseline and is not a declared dose")
        self.alpha_multiples = multiples
        if self.microbatch_size < 1:
            raise OptVecEvalConfigError("microbatchSize must be at least 1")
        if self.null_samples < 1:
            raise OptVecEvalConfigError(
                "nullSamples must be at least 1 — a library cosine is never "
                "reported without its random-direction null")
        self.library_vector_paths = list(self.library_vector_paths)

    @classmethod
    def from_dict(cls, payload: dict) -> "OptVecEvalConfig":
        if not isinstance(payload, dict):
            raise OptVecEvalConfigError(
                "the OptVec eval config must be a JSON object")
        _refuse_selection_data_keys(payload, "selection-split config key(s)")
        unknown = sorted(set(payload) - set(_CONFIG_KEYS)
                         - {"datasets", "neutralTexts"})
        if unknown:
            raise OptVecEvalConfigError(
                "unknown OptVec eval config key(s): " + ", ".join(unknown))
        raw_datasets = payload.get("datasets")
        if not isinstance(raw_datasets, dict):
            raise OptVecEvalConfigError(
                "the config needs a 'datasets' object (targetTest, and "
                "optionally anchorTest/capabilityEval)")
        _refuse_selection_data_keys(raw_datasets, "selection-split datasets key(s)")
        unknown_sets = sorted(set(raw_datasets) - set(_DATASET_KEYS))
        if unknown_sets:
            raise OptVecEvalConfigError(
                "unknown datasets key(s): " + ", ".join(unknown_sets))
        if "targetTest" not in raw_datasets:
            raise OptVecEvalConfigError("datasets.targetTest is required")
        datasets = EvalDatasets(**{
            field_name: FileRef.from_dict(raw_datasets[key], f"datasets.{key}")
            for key, field_name in _DATASET_KEYS.items() if key in raw_datasets})
        kwargs = {field_name: payload[key]
                  for key, field_name in _CONFIG_KEYS.items() if key in payload}
        if "neutralTexts" in payload:
            kwargs["neutral_texts"] = FileRef.from_dict(payload["neutralTexts"],
                                                        "neutralTexts")
        if "vector_artifact" not in kwargs:
            raise OptVecEvalConfigError("'vectorArtifact' is required")
        return cls(datasets=datasets, **kwargs)

    def to_dict(self) -> dict:
        out = {json_key: getattr(self, field_name)
               for json_key, field_name in _CONFIG_KEYS.items()}
        out["alphaMultiples"] = list(self.alpha_multiples)
        out["libraryVectorPaths"] = list(self.library_vector_paths)
        out["datasets"] = self.datasets.to_dict()
        if self.neutral_texts is not None:
            out["neutralTexts"] = self.neutral_texts.to_dict()
        return out


def load_config(path: str) -> OptVecEvalConfig:
    with open(path, encoding="utf-8") as handle:
        return OptVecEvalConfig.from_dict(json.load(handle))


# ----------------------------------------------------------------- artifacts


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LoadedArtifact:
    """A vector artifact plus the identity of the exact bytes read.

    ``optvec`` is the additive sidecar block the training driver writes; it is
    read from the RAW sidecar JSON because
    :meth:`SteeringVectorSidecar.from_dict` keeps only the fields it declares
    and would drop it.
    """

    reference: str
    path: str
    name: str
    vectors: vector_store.ConceptVectors
    sidecar: vector_store.SteeringVectorSidecar
    raw_sidecar: dict
    tensor_sha256: str
    sidecar_sha256: str

    @property
    def optvec(self) -> dict | None:
        block = self.raw_sidecar.get("optvec")
        return block if isinstance(block, dict) else None

    @property
    def is_optvec(self) -> bool:
        return self.optvec is not None

    def row(self, layer: int) -> list[float] | None:
        """The direction at ``layer``, or ``None`` when the artifact has no
        such layer or its row there is exactly zero (an optvec artifact is
        full-depth with zeros everywhere but its own layer, so a zero row is
        'this artifact says nothing here', not a direction)."""
        if layer < 0 or layer >= self.vectors.layer_count:
            return None
        if self.vectors.norm(layer) == 0.0:
            return None
        return list(self.vectors.per_layer[layer])

    def identity(self) -> dict:
        return {"reference": self.reference, "path": self.path,
                "name": self.name,
                "extractionMethod": self.sidecar.extractionMethod,
                "modelID": self.sidecar.modelID,
                "revision": self.sidecar.revision,
                "stimulusSetHash": self.sidecar.stimulusSetHash,
                "layerCount": self.vectors.layer_count,
                "hiddenSize": self.vectors.hidden_size,
                "tensorSHA256": self.tensor_sha256,
                "sidecarSHA256": self.sidecar_sha256}


def load_artifact(reference: str) -> LoadedArtifact:
    """Load an extension-less artifact reference and hash both of its files.

    Every failure mode — missing sidecar, missing tensors, non-finite bytes —
    surfaces as one typed error naming the reference, because callers here
    (the library table especially) need to record WHICH artifact could not be
    read, not just that something could not be.
    """
    resolved = paths.resolve_artifact(reference)
    directory = os.path.dirname(resolved) or "."
    name = os.path.basename(resolved)
    try:
        vectors, sidecar = vector_store.load(directory, name)
        with open(os.path.join(directory, f"{name}.json"),
                  encoding="utf-8") as handle:
            raw = json.load(handle)
        tensor_hash = sha256_file(os.path.join(directory,
                                               f"{name}.safetensors"))
        sidecar_hash = sha256_file(os.path.join(directory, f"{name}.json"))
    except Exception as exc:  # noqa: BLE001 - re-raised as one typed error
        raise OptVecArtifactError(
            f"vector artifact '{reference}' could not be loaded: {exc}") from exc
    return LoadedArtifact(
        reference=reference, path=resolved, name=name, vectors=vectors,
        sidecar=sidecar, raw_sidecar=raw, tensor_sha256=tensor_hash,
        sidecar_sha256=sidecar_hash)


@dataclass(frozen=True)
class OptVecTarget:
    """The artifact under evaluation, with its own layer and α resolved."""

    artifact: LoadedArtifact
    layer: int
    alpha_absolute: float

    @property
    def direction(self) -> list[float]:
        return list(self.artifact.vectors.per_layer[self.layer])


def require_optvec(artifact: LoadedArtifact) -> OptVecTarget:
    """Refuse anything without the additive ``optvec`` sidecar block.

    The block is what makes the artifact evaluable: the layer it means anything
    at, and the absolute α its direction was optimized (and normalized) at. An
    ordinary extracted vector has neither, and a dose grid stated in multiples
    of an α that does not exist is not a measurement.
    """
    block = artifact.optvec
    if block is None:
        raise OptVecArtifactError(
            f"vector artifact '{artifact.reference}' carries no 'optvec' "
            f"sidecar block (extractionMethod "
            f"{artifact.sidecar.extractionMethod!r}) — this verb evaluates "
            "OPTIMIZED vectors, whose sidecar names the layer they were "
            "optimized at and the absolute α they carry. Nothing else can be "
            "dosed in multiples of its own α.")
    layer = block.get("layer")
    alpha = block.get("alphaAbsolute")
    if not isinstance(layer, int) or isinstance(layer, bool):
        raise OptVecArtifactError(
            f"'{artifact.reference}' optvec.layer must be an integer "
            f"(got {layer!r})")
    if layer < 0 or layer >= artifact.vectors.layer_count:
        raise OptVecArtifactError(
            f"'{artifact.reference}' optvec.layer {layer} is outside its "
            f"{artifact.vectors.layer_count}-layer artifact")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) \
            or not alpha > 0:
        raise OptVecArtifactError(
            f"'{artifact.reference}' optvec.alphaAbsolute must be a positive "
            f"number (got {alpha!r})")
    if artifact.vectors.norm(layer) == 0.0:
        raise OptVecArtifactError(
            f"'{artifact.reference}' has a zero row at its own optvec layer "
            f"{layer} — there is no direction to evaluate")
    return OptVecTarget(artifact=artifact, layer=layer,
                        alpha_absolute=float(alpha))


def verify_model(target: OptVecTarget, model) -> dict:
    """Refuse a vector/model mismatch, and refuse a foreign substrate.

    Activations do not transfer across engines or across models, so a vector
    scored against the wrong one produces confident nonsense. The revision is
    checked only when BOTH sides pin one: an unpinned side is unknown, never a
    guessed match.
    """
    vector_store.require_native_substrate(target.artifact.sidecar,
                                          target.artifact.path)
    stamped = target.artifact.sidecar.modelID
    loaded = getattr(model, "model_id", None)
    if stamped and loaded and stamped != loaded:
        raise OptVecArtifactError(
            f"vector artifact '{target.artifact.reference}' was optimized on "
            f"model {stamped!r}; this eval is running {loaded!r} — a direction "
            "is a direction in ONE model's residual basis")
    pinned = target.artifact.sidecar.revision
    running = getattr(model, "revision", None)
    if pinned and running and pinned != running:
        raise OptVecArtifactError(
            f"vector artifact '{target.artifact.reference}' pins revision "
            f"{pinned!r}; this eval is running {running!r} — re-optimize, or "
            "run the eval at the pinned revision")
    return {"artifactModelID": stamped, "runningModelID": loaded,
            "artifactRevision": pinned, "runningRevision": running,
            "revisionChecked": bool(pinned and running)}


# ------------------------------------------------------------------ datasets


def load_dataset(ref: FileRef, label: str) -> tuple[ChoiceRow, ...]:
    """Load a hashed choice-row file, refusing on hash drift (same strict
    loader — ``sweep_selection.load_choice_rows`` — the sweep and the training
    driver use, so a set that loads for training loads here identically)."""
    try:
        # Workspace-relative refs resolve against STEERLAB_ROOT, never the
        # process cwd (the campaign-cell trap — see
        # ``optvec_train.resolve_data_path``).
        rows, digest = load_choice_rows(
            optvec_train_resolve(ref.path), label)
    except ValueError as exc:
        raise OptVecEvalDataError(str(exc)) from exc
    if digest != ref.sha256:
        raise OptVecEvalDataError(
            f"dataset '{label}' ({ref.path}) hashes {digest} but the config "
            f"pins {ref.sha256} — the file changed since the eval was "
            "configured; re-pin it deliberately or restore the pinned bytes")
    return rows


def parse_neutral_texts(content: str) -> list[str]:
    """One text per line: a JSON object line contributes its ``text`` field,
    any other non-blank line IS the text.

    Deliberately not ``neutral.parse_texts``, which treats a plain file as
    blank-line-separated PARAGRAPHS — the fluency guard wants a known count of
    short independent passages, and a line-oriented rule makes the file's
    line count and its text count the same number.
    """
    texts: list[str] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except ValueError:
                texts.append(line)
                continue
            value = obj.get("text") if isinstance(obj, dict) else None
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
            continue
        texts.append(line)
    return texts


def load_neutral_texts(ref: FileRef) -> list[str]:
    resolved = optvec_train_resolve(ref.path)
    if not os.path.exists(resolved):
        raise OptVecEvalDataError(
            f"neutral text file not found: {ref.path}"
        ) from FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                 resolved)  # a CLI reads this as notFound
    with open(resolved, "rb") as handle:
        data = handle.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != ref.sha256:
        raise OptVecEvalDataError(
            f"neutral texts ({ref.path}) hash {digest} but the config pins "
            f"{ref.sha256} — the fluency denominator changed since the eval "
            "was configured")
    texts = parse_neutral_texts(data.decode("utf-8"))
    if not texts:
        raise OptVecEvalDataError(
            f"neutral texts ({ref.path}) parsed to zero texts")
    return texts


# ------------------------------------------------------------------- metrics


def kl_divergence(p0: list[float], p: list[float]) -> float:
    """``KL(p0 ‖ p)`` over two choice distributions in the SAME declared option
    order. Direction matters: ``p0`` (the baseline) is the reference, so this
    measures probability mass moved away from where the unsteered model put it
    — the same direction the training objective's anchor term uses."""
    if len(p0) != len(p):
        raise ValueError(
            f"distributions of different length ({len(p0)} vs {len(p)})")
    total = 0.0
    for a, b in zip(p0, p):
        if a <= 0:
            continue
        total += a * (math.log(a) - math.log(max(b, 1e-12)))
    return float(total)


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _pad_id(tokenizer) -> int:
    return (getattr(tokenizer, "pad_token_id", None)
            or getattr(tokenizer, "eos_token_id", None) or 0)


# --------------------------------------------------------- behavioral arms


@dataclass(frozen=True)
class EvalItem:
    """One test item: the raw prompt (rendering happens inside the instrument,
    exactly as in every behavioral study), its options, and the option this
    item's role measures against."""

    id: str
    role: str
    dataset: str
    prompt: str
    options: tuple[str, ...]
    target: str


def prepare_items(rows: tuple[ChoiceRow, ...], *, role: str,
                  dataset: str) -> list[EvalItem]:
    return [EvalItem(id=row.id, role=role, dataset=dataset, prompt=row.prompt,
                     options=tuple(row.options), target=row.target)
            for row in rows]


def score_item(model, item: EvalItem, config: OptVecEvalConfig, *,
               injections: list[CellInjection] | None):
    """One item under one dose, through the deployed instrument."""
    return logprob.score_options(
        model, item.prompt, list(item.options), injections=injections,
        prompt_mode=config.prompt_mode, system_prompt=config.system_prompt,
        qwen_thinking_enabled=config.qwen_thinking_enabled)


def cell_injection(target: OptVecTarget, multiple: float) -> CellInjection:
    """The dose, in the deployed injection path's own units.

    ``VectorInjector`` adds ``alpha · vector`` — it does NOT renormalize — and
    the stored row already has L2 norm ``alphaAbsolute`` at this layer (the
    training injector projected onto that sphere and exported the constrained
    vector). So ``alpha = multiple`` injects a vector of norm
    ``multiple × alphaAbsolute``, which is what the dose grid means. Anything
    else here would silently re-denominate α.
    """
    return CellInjection(layer=target.layer, vector=target.direction,
                         alpha=float(multiple))


def _role_metrics(role: str, rows: list[dict]) -> dict:
    """Per-role summary for one dose. Every rate is paired to the α=0 baseline
    row for the same item — the baseline is measured IN THIS RUN, not read from
    the training run's cache, so a metric never straddles two model loads."""
    if not rows:
        return {"itemCount": 0}
    summary: dict = {"itemCount": len(rows),
                     "flipRate": _mean([1.0 if r["flipped"] else 0.0
                                        for r in rows]),
                     "meanKLFromBaseline": _mean([r["klFromBaseline"]
                                                  for r in rows])}
    if role == "target":
        summary["targetSelectedRate"] = _mean(
            [1.0 if r["selectedTarget"] else 0.0 for r in rows])
        # The headline shift rate is over the items that COULD flip — an item
        # whose unsteered argmax is already the target cannot be shifted onto
        # it, and counting it as a success would report the model's prior as
        # the vector's effect.
        flippable = [r for r in rows if not r["baselineSelectedTarget"]]
        summary["flippableItemCount"] = len(flippable)
        summary["shiftRate"] = _mean(
            [1.0 if r["selectedTarget"] else 0.0 for r in flippable])
        summary["meanLogOddsMovement"] = _mean(
            [r["deltaLogOddsTarget"] for r in rows])
    elif role == "capability":
        summary["accuracy"] = _mean([1.0 if r["correct"] else 0.0
                                     for r in rows])
        summary["baselineAccuracy"] = _mean(
            [1.0 if r["baselineCorrect"] else 0.0 for r in rows])
        summary["accuracyDelta"] = (summary["accuracy"]
                                    - summary["baselineAccuracy"])
    return summary


# ----------------------------------------------------------- fluency guard


@torch.no_grad()
def mean_token_logprobs(model, texts: list[str], *,
                        injector: TrainableVectorInjector | None = None,
                        microbatch_size: int = 4) -> list[dict]:
    """Teacher-forced mean per-token logprob for each text.

    Right-padded batches (the same convention the training driver asserts), so
    every text occupies ``[0, len)`` and its scored positions are
    ``0 … len − 2`` predicting tokens ``1 … len − 1``. A text of fewer than two
    tokens has nothing to predict and is recorded with a null score rather than
    silently dropped from the denominator.
    """
    device = model.device
    pad = _pad_id(model.tokenizer)
    encoded = [list(model.tokenizer(text).input_ids) for text in texts]
    out: list[dict] = [{"index": i, "tokenCount": len(ids),
                        "meanTokenLogprob": None}
                       for i, ids in enumerate(encoded)]
    scorable = [i for i, ids in enumerate(encoded) if len(ids) >= 2]
    for start in range(0, len(scorable), microbatch_size):
        indices = scorable[start:start + microbatch_size]
        width = max(len(encoded[i]) for i in indices)
        input_ids = torch.tensor(
            [encoded[i] + [pad] * (width - len(encoded[i])) for i in indices],
            dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            [[1] * len(encoded[i]) + [0] * (width - len(encoded[i]))
             for i in indices], dtype=torch.long, device=device)
        interventions = []
        if injector is not None:
            injector.set_batch(attention_mask=attention_mask)
            interventions.append(injector)
        model.hooked.reset_offsets()
        try:
            with model.hooked.session(interventions):
                result = model.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     use_cache=False)
        finally:
            if injector is not None:
                injector.clear_batch()
        logprobs = torch.log_softmax(result.logits.float(), dim=-1)
        for position, index in enumerate(indices):
            length = len(encoded[index])
            targets = input_ids[position, 1:length]
            rows = logprobs[position, :length - 1]
            picked = rows.gather(1, targets.unsqueeze(1)).squeeze(1)
            out[index]["meanTokenLogprob"] = float(picked.mean())
    return out


def fluency_injector(model, target: OptVecTarget,
                     multiple: float) -> TrainableVectorInjector:
    """All-position injector carrying exactly ``multiple × direction``.

    The constructor renormalizes ``u`` onto the sphere of radius
    ``alpha_absolute``; passing the stored row as ``u`` with
    ``alpha_absolute = multiple · ‖row‖`` therefore reproduces the scaled
    vector exactly (``multiple·‖row‖ · row/‖row‖ = multiple · row``), which is
    the same vector the behavioral arms inject through ``CellInjection`` at
    this dose.
    """
    row = target.direction
    norm = math.sqrt(sum(x * x for x in row))
    return TrainableVectorInjector(
        layer=target.layer, hidden_size=len(row),
        alpha_absolute=float(multiple) * norm, position_mode="all",
        u=torch.tensor(row, dtype=torch.float32), device=model.device)


# ------------------------------------------------------- library comparison


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


def null_cosines(direction: list[float], samples: int, seed: int) -> list[float]:
    """Cosines of ``samples`` seeded random unit directions against
    ``direction``. Random directions in high dimension are near-orthogonal to
    everything, so this distribution is what makes a library cosine readable —
    the same rule the J-lens support readout enforces for energy."""
    import numpy as np

    rng = np.random.default_rng(seed)
    v = np.asarray(direction, dtype=np.float64)
    v = v / (np.linalg.norm(v) or 1.0)
    draws = rng.standard_normal((int(samples), v.shape[0]))
    draws /= np.linalg.norm(draws, axis=1, keepdims=True)
    return [float(x) for x in draws @ v]


def library_comparison(target: OptVecTarget, references: list[str], *,
                       samples: int, seed: int) -> dict:
    """Cosine against each named library vector at the OptVec layer, each with
    its percentile in the random-direction null."""
    direction = target.direction
    nulls = null_cosines(direction, samples, seed)
    absolute_nulls = sorted(abs(c) for c in nulls)

    def percentile(value: float) -> float:
        below = sum(1 for c in absolute_nulls if c <= abs(value))
        return 100.0 * below / len(absolute_nulls)

    entries: list[dict] = []
    skipped: list[dict] = []
    for reference in references:
        try:
            artifact = load_artifact(reference)
        except OptVecArtifactError as exc:
            skipped.append({"reference": reference, "reason": str(exc)})
            continue
        if artifact.vectors.hidden_size != len(direction):
            skipped.append({
                "reference": reference,
                "reason": f"hidden size {artifact.vectors.hidden_size} != "
                          f"{len(direction)} — a different model's basis"})
            continue
        row = artifact.row(target.layer)
        if row is None:
            skipped.append({
                "reference": reference,
                "reason": f"no nonzero row at layer {target.layer} "
                          f"({artifact.vectors.layer_count} layer(s) present)"})
            continue
        value = cosine(direction, row)
        entries.append({
            "reference": reference, "name": artifact.name,
            "extractionMethod": artifact.sidecar.extractionMethod,
            "concept": artifact.sidecar.concept,
            "isOptVec": artifact.is_optvec,
            "layer": target.layer,
            "cosine": value, "absCosine": abs(value),
            "nullPercentile": percentile(value),
            "tensorSHA256": artifact.tensor_sha256})
    top = sorted(entries, key=lambda e: e["absCosine"],
                 reverse=True)[:LIBRARY_TOP_K]
    return {
        "layer": target.layer,
        "comparedCount": len(entries),
        "entries": entries,
        "skipped": skipped,
        "topK": [{"reference": e["reference"], "cosine": e["cosine"],
                  "nullPercentile": e["nullPercentile"],
                  "extractionMethod": e["extractionMethod"]} for e in top],
        "topKSize": LIBRARY_TOP_K,
        "null": {
            "samples": len(nulls), "seed": seed,
            "note": "cosines of seeded random unit directions at this d_model "
                    "against the optvec direction; percentiles are over |cos|",
            "absCosineP50": _quantile(absolute_nulls, 0.50),
            "absCosineP95": _quantile(absolute_nulls, 0.95),
            "absCosineP99": _quantile(absolute_nulls, 0.99),
            "absCosineMax": absolute_nulls[-1] if absolute_nulls else None},
    }


def _quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1,
                max(0, int(round(q * (len(sorted_values) - 1)))))
    return float(sorted_values[index])


# -------------------------------------------------------------------- driver


def evaluate(config: OptVecEvalConfig, *, model=None,
             log: Callable[[str], None] | None = None) -> dict:
    """Score the test split at every dose and write the eval's own run
    directory. Returns the run's summary dict."""
    def emit(message: str) -> None:
        if log is not None:
            log(message)

    artifact = load_artifact(config.vector_artifact)
    target = require_optvec(artifact)
    if config.model_id and artifact.sidecar.modelID \
            and config.model_id != artifact.sidecar.modelID:
        raise OptVecEvalConfigError(
            f"config modelID {config.model_id!r} disagrees with the artifact's "
            f"{artifact.sidecar.modelID!r} — the artifact's model is the one "
            "its direction means anything in")

    items: list[EvalItem] = []
    dataset_notes: dict = {"files": {}, "counts": {}}
    seen: dict[str, str] = {}
    for key, field_name in _DATASET_KEYS.items():
        ref = getattr(config.datasets, field_name)
        if ref is None:
            continue
        rows = load_dataset(ref, key)
        prepared = prepare_items(rows, role=_DATASET_ROLES[key], dataset=key)
        for item in prepared:
            if item.id in seen:
                raise OptVecEvalDataError(
                    f"item id '{item.id}' appears in both '{seen[item.id]}' and "
                    f"'{key}' — records are keyed by id, and overlapping sets "
                    "would double-count one item under two roles")
            seen[item.id] = key
        items.extend(prepared)
        dataset_notes["files"][key] = ref.to_dict()
        dataset_notes["counts"][key] = len(prepared)

    neutral_texts: list[str] = []
    if config.neutral_texts is not None:
        neutral_texts = load_neutral_texts(config.neutral_texts)
        dataset_notes["files"]["neutralTexts"] = config.neutral_texts.to_dict()
        dataset_notes["counts"]["neutralTexts"] = len(neutral_texts)

    name = (config.name or "").strip() or artifact.name
    run_directory = paths.make_unique_run_directory(f"optvec-eval-{name}")
    notes: dict = {"stage": "starting", "claim": "sufficiency",
                   "split": "test",
                   "artifact": artifact.identity(),
                   "optvec": {"layer": target.layer,
                              "alphaAbsolute": target.alpha_absolute},
                   "datasets": dataset_notes,
                   "alphaMultiples": list(config.alpha_multiples)}
    dtype_name: str | None = None

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
        notes["stage"] = "scoring"

        records: list[dict] = []
        doses = [0.0, *config.alpha_multiples]
        baseline: dict[str, dict] = {}
        dose_response: list[dict] = []

        for multiple in doses:
            injections = None if multiple == 0.0 \
                else [cell_injection(target, multiple)]
            per_role: dict[str, list[dict]] = {role: [] for role in ROLES}
            for item in items:
                result = score_item(model, item, config, injections=injections)
                probabilities = list(result.ordered_probabilities)
                log_odds = result.log_odds
                if multiple == 0.0:
                    baseline[item.id] = {
                        "selected": result.selected,
                        "probabilities": probabilities,
                        "logOdds": dict(log_odds)}
                base = baseline[item.id]
                row = {
                    "id": item.id, "role": item.role, "dataset": item.dataset,
                    "alphaMultiple": float(multiple),
                    "alphaAbsolute": float(multiple) * target.alpha_absolute,
                    "layer": target.layer,
                    "targetOption": item.target,
                    "baselineSelected": base["selected"],
                    "selectedTarget": result.selected == item.target,
                    "baselineSelectedTarget": base["selected"] == item.target,
                    "flipped": result.selected != base["selected"],
                    "logOddsTarget": log_odds.get(item.target),
                    "deltaLogOddsTarget": (log_odds.get(item.target, 0.0)
                                           - base["logOdds"].get(item.target,
                                                                 0.0)),
                    "klFromBaseline": kl_divergence(base["probabilities"],
                                                    probabilities),
                    "correct": result.selected == item.target,
                    "baselineCorrect": base["selected"] == item.target,
                    **result.as_record_fields(),
                }
                records.append(row)
                per_role[item.role].append(row)

            entry = {"alphaMultiple": float(multiple),
                     "alphaAbsolute": float(multiple) * target.alpha_absolute,
                     "isBaseline": multiple == 0.0}
            for role in ROLES:
                entry[role] = _role_metrics(role, per_role[role])
            dose_response.append(entry)
            emit(f"dose {multiple}×: scored {len(items)} item(s)")

        notes["stage"] = "fluency"
        if neutral_texts:
            baseline_fluency = mean_token_logprobs(
                model, neutral_texts, injector=None,
                microbatch_size=config.microbatch_size)
            baseline_mean = _mean([r["meanTokenLogprob"]
                                   for r in baseline_fluency
                                   if r["meanTokenLogprob"] is not None])
            for entry in dose_response:
                multiple = entry["alphaMultiple"]
                per_text = baseline_fluency if multiple == 0.0 else \
                    mean_token_logprobs(
                        model, neutral_texts,
                        injector=fluency_injector(model, target, multiple),
                        microbatch_size=config.microbatch_size)
                scored = [r["meanTokenLogprob"] for r in per_text
                          if r["meanTokenLogprob"] is not None]
                mean_value = _mean(scored)
                entry["fluency"] = {
                    "textCount": len(per_text),
                    "scoredTextCount": len(scored),
                    "meanTokenLogprob": mean_value,
                    "deltaFromBaseline": (None if mean_value is None
                                          or baseline_mean is None
                                          else mean_value - baseline_mean),
                    "positionMode": "all"}
                for record in per_text:
                    records.append({
                        "id": f"neutral-{record['index'] + 1}",
                        "role": "neutral", "dataset": "neutralTexts",
                        "alphaMultiple": float(multiple),
                        "alphaAbsolute": (float(multiple)
                                          * target.alpha_absolute),
                        "layer": target.layer,
                        "instrument": "meanTokenLogprob",
                        "positionMode": "all",
                        "tokenCount": record["tokenCount"],
                        "meanTokenLogprob": record["meanTokenLogprob"]})
        else:
            for entry in dose_response:
                entry["fluency"] = {
                    "textCount": 0, "scoredTextCount": 0,
                    "meanTokenLogprob": None, "deltaFromBaseline": None,
                    "note": "no neutralTexts declared — no fluency guard ran"}

        notes["stage"] = "library"
        library = library_comparison(target, config.library_vector_paths,
                                     samples=config.null_samples,
                                     seed=config.seed)

        readout = {
            "schemaVersion": SCHEMA_VERSION,
            "runType": RUN_TYPE,
            "runID": os.path.basename(os.path.normpath(run_directory)),
            "claim": "sufficiency",
            "split": "test",
            "firewall": "test split only: gradients ran on train, checkpoint "
                        "selection on val; neither is readable from this "
                        "config. eval.json is screen-grade evidence — citable "
                        "numbers come from a confirm-style behavioral study "
                        "with this vector attached as a pinned artifact.",
            "artifact": artifact.identity(),
            "optvec": {"layer": target.layer,
                       "alphaAbsolute": target.alpha_absolute,
                       "sidecarBlock": artifact.optvec},
            "model": {**notes["model"], "dtype": dtype_name,
                      "device": str(getattr(model, "device", ""))},
            "datasets": dataset_notes,
            "instrument": {
                "behavioral": "answerTokenLogprob (stepped KV-cache "
                              "score_options with the deployed CellInjection "
                              "path; injection at the prompt end and every "
                              "option step)",
                "fluency": "teacher-forced mean per-token logprob under "
                           "ALL-position injection (see module docstring: "
                           "every position of a generated text is a decode "
                           "position)",
                "promptMode": config.prompt_mode,
                "qwenThinkingEnabled": config.qwen_thinking_enabled},
            "alphaMultiples": list(config.alpha_multiples),
            "doseResponse": dose_response,
            "library": library,
        }
        with open(os.path.join(run_directory, EVAL_JSON), "w",
                  encoding="utf-8") as handle:
            json.dump(readout, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with open(os.path.join(run_directory, EVAL_RECORDS), "w",
                  encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        notes["metrics"] = _metrics_summary(dose_response)
        notes["library"] = {"comparedCount": library["comparedCount"],
                            "skippedCount": len(library["skipped"]),
                            "topK": library["topK"]}
        notes["recordCount"] = len(records)
        notes["stage"] = "complete"
        emit(f"optvec eval → {run_directory}")
        return {"runDirectory": run_directory,
                "artifact": artifact.identity(),
                "layer": target.layer,
                "alphaAbsolute": target.alpha_absolute,
                "doseResponse": dose_response,
                "library": library,
                "recordCount": len(records)}
    finally:
        # Written LAST so the notes are complete, and in `finally` so a crashed
        # eval still records the stage it died in (run directories are
        # immutable and write_run_config never rewrites).
        write_run_config(run_directory, RUN_TYPE,
                         model_id=(config.model_id
                                   or artifact.sidecar.modelID),
                         revision=(getattr(model, "revision", None)
                                   or config.revision
                                   or artifact.sidecar.revision),
                         dtype=dtype_name, notes=notes)


def _metrics_summary(dose_response: list[dict]) -> list[dict]:
    """The compact per-dose table that lands in ``config.json`` notes — enough
    to read the run's verdict without opening ``eval.json``."""
    out: list[dict] = []
    for entry in dose_response:
        out.append({
            "alphaMultiple": entry["alphaMultiple"],
            "targetShiftRate": entry["target"].get("shiftRate"),
            "targetMeanLogOddsMovement":
                entry["target"].get("meanLogOddsMovement"),
            "anchorFlipRate": entry["anchor"].get("flipRate"),
            "anchorMeanKL": entry["anchor"].get("meanKLFromBaseline"),
            "capabilityAccuracy": entry["capability"].get("accuracy"),
            "fluencyDelta": entry.get("fluency", {}).get("deltaFromBaseline"),
        })
    return out
