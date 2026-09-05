"""LoRA adapter training (parallel to Swift ``FineTuneTrainer``), via HF PEFT.

Weight-level disposition as a contrast intervention to activation-level
injection. Two cluster-specific swaps from the Swift path: **HF PEFT** replaces
MLX ``LoRATrain``, and **pypdf** replaces PDFKit for PDF ingestion (a hard Linux
blocker on the Swift side). ``peft`` and ``pypdf`` are optional deps
(``pip install -e .[lora]``) and imported lazily.

Adapter-format note: PEFT writes ``adapter_model.safetensors`` +
``adapter_config.json`` (its standard layout), which is **not** byte-compatible
with the MLX ``adapters.safetensors`` the Swift app loads. Cluster adapters are
a separate, model-specific artifact — re-train on the cluster; do not transplant
(see CHANGES-TO-SWIFT-SIDE.md).

Three training modes (``docs/CLUSTER-LORA-READINESS.md`` §2.1–2.6):

- ``legacy_inline`` — the historical exploratory path, preserved verbatim: the
  uploaded documents are concatenated, cut into fixed token windows, and the
  trailing fraction is called "validation" without ever being evaluated. It is
  a smoke test, not evidence, and the sidecar now says so on its face
  (``evidenceGrade: false``, ``trainingMode: "legacy_inline"``).
- ``document`` / ``instruction_chat`` — the evidence-grade path, whose data
  discipline lives entirely in :mod:`lora_data` (researcher-authored splits,
  row-level identity, assistant-only loss, never a concatenated boundary) and
  whose *training* discipline lives here: a declared epoch/step schedule, a
  seeded deterministic per-epoch shuffle, gradient accumulation, warmup +
  linear decay, gradient clipping, full-validation-set evaluation at declared
  intervals AND every epoch boundary, and **best-checkpoint selection by a
  preregistered metric** — the final adapter directory holds the SELECTED
  checkpoint's weights, not merely the last optimizer step's (plan §2.5).

Why the selection rule is data, not a default (plan §0.5): "take the last
step" and "take the best validation loss" can disagree by more than the
intervention being measured. An evidence-grade run whose manifest declares no
metric refuses at start rather than silently picking one.

**The objective** (``TRAINING_OBJECTIVE``, external review 2026-09-05). One
optimizer step minimizes the token-average cross-entropy over ALL supervised
next-token targets in that step's group of micro-batches::

    L = sum(token losses in the group) / count(supervised targets in the group)

A *supervised target* is a position where ``labels != IGNORE_LABEL`` AFTER the
causal shift — i.e. ``labels[:, 1:]``. Prompt tokens (masked ``-100`` by
:mod:`lora_data`) and padding are not targets, and the first position of a
sequence is never one because nothing predicts it.

The denominator is a property of the OPTIMIZER STEP, never of how the step was
cut into micro-batches. The trainer therefore counts the group's targets before
the first backward pass and scales each micro-batch's SUMMED loss by
``1/group_targets``. The obvious-looking alternative — take HF's per-micro-batch
mean and divide by ``gradient_accumulation`` — is a mean of means weighted by
the partitioning: it makes ``batch_size=2, accumulation=1`` and
``batch_size=1, accumulation=2`` produce different gradients from identical
data (measured: gradient cosine 0.968, relative difference 0.300 on the
two-layer test model), and it scales an incomplete final group down by the
accumulation factor it never filled (measured: exactly half). Both were real
defects here before 2026-09-05; ``tests/test_lora_train_objective.py`` is what
keeps them fixed.

The same definition is what ``_evaluate`` reports, so a training row and a
validation row in ``training-history.json`` are comparable numbers rather than
two differently-weighted averages. Every history row names its denominator.

Checkpoint/resume reuses :mod:`resume` verbatim — exit code 85, SIGUSR1/SIGTERM
via :class:`resume.CheckpointFlag`, tmp+fsync+rename atomicity — so a preempted
27B training job continues to a run directory indistinguishable from an
uninterrupted one (plan §2.6). Resume verifies the dataset, model revision, and
config fingerprint against the checkpoint before touching a weight: continuing
one experiment's optimizer state into another experiment's data is a silent
provenance forgery, so it is a typed refusal.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime, timezone
from typing import Callable

from . import lora_data, paths, resume
# LoRADataError is re-exported deliberately: a caller catching training
# refusals needs the dataset refusals too (they abort the same verb).
from .lora_data import DatasetFile, LoRADataError, LoRADatasetSpec  # noqa: F401
from ..steering.vector_store import SUBSTRATE

# This engine's adapter format, stamped into every adapter sidecar it writes
# (the pinned cross-engine contract, parallel to vector_store.SUBSTRATE). The
# Swift engine writes ``"mlx-lora"``. PEFT cannot load an MLX adapter (and vice
# versa), so the stamp is what lets the loading side refuse loudly instead of
# failing weirdly. Absent stamp = legacy/unknown — attempted as today.
ADAPTER_FORMAT = "hf-peft-lora"

#: How this engine's ``alpha`` becomes the multiplier actually applied to the
#: LoRA update. PEFT scales the adapter delta by ``lora_alpha / r``, so ``alpha``
#: is a NUMERATOR, meaningless without the rank beside it. The Swift/MLX path's
#: adapter ``scale`` is a DIRECT multiplier with no rank in it, so two runs whose
#: numeric fields agree are not the same treatment. Naming the convention (and
#: stamping the resolved multiplier) is what stops a reader equating them
#: (external review, REM-06).
ADAPTER_SCALE_CONVENTION = "peft:lora_alpha/r"

#: Sidecar schema. v1 keys are all still written; v2 adds the provenance the
#: plan's §2.8 requires (dataset identity, revisions, schedule, selection,
#: package/GPU/Slurm identity, resume lineage, adapter byte hashes).
SIDECAR_SCHEMA_VERSION = 2

#: The exploratory mode: inline text, implicit fractional split, no evaluation.
LEGACY_INLINE = "legacy_inline"
TRAINING_MODES = (lora_data.DOCUMENT, lora_data.INSTRUCTION_CHAT, LEGACY_INLINE)

#: The only checkpoint-selection metric implemented today. Manifest DATA —
#: absent is legal for exploratory runs (last step wins) and a refusal for
#: evidence-grade ones (plan §0.5).
VALIDATION_LOSS = "validationLoss"
SELECTION_METRICS = (VALIDATION_LOSS,)

#: The training OBJECTIVE, stamped into the schedule provenance and into the
#: config fingerprint a resumed run must still agree with. See "The objective"
#: in this module's docstring: token-average cross-entropy over every
#: supervised target in one optimizer step's group. Named (rather than left
#: implicit in the loop) because "the loss" is ambiguous until its denominator
#: is: a mean of micro-batch means is a DIFFERENT objective, and an adapter
#: trained under one cannot be compared to an adapter trained under the other
#: (external review, 2026-09-05).
TRAINING_OBJECTIVE = "tokenMeanPerOptimizerStep"

#: The verb name in ``resume-state.json`` — a study run's state and a LoRA
#: training run's state must never be mistaken for each other.
RESUME_VERB = "lora-train"

CHECKPOINTS_DIRNAME = "checkpoints"
TRAINER_STATE_FILENAME = "trainer-state.json"
HISTORY_FILENAME = "training-history.json"

_FULL_SHA_LENGTH = 40
_FP16_NAMES = {"float16", "fp16"}

#: Control-arm kinds (plan §0.2). ``shuffledAssistantPairing`` is the S0-analog
#: the trainer can *measure* (it stamps how many rows actually changed);
#: ``declaredNeutralizedDataset`` is a researcher declaration about a separate
#: dataset, stamped as given with a null fraction — the trainer cannot verify a
#: neutralization it never saw.
SHUFFLED_ASSISTANT_PAIRING = "shuffledAssistantPairing"
DECLARED_NEUTRALIZED_DATASET = "declaredNeutralizedDataset"
CONTROL_ARM_KINDS = (SHUFFLED_ASSISTANT_PAIRING, DECLARED_NEUTRALIZED_DATASET)


class LoRATrainError(RuntimeError):
    """A training refusal: an evidence-grade run missing a pin the evidence
    depends on, an unusable mode/metric, or a run directory that is already
    finalized. Loud and specific — never a silent downgrade."""


class LoRAResumeError(LoRATrainError):
    """A checkpoint that must not be adopted: no checkpoint to resume, or one
    whose dataset / model revision / config fingerprint disagrees with the live
    configuration. Continuing across such a mismatch would produce an adapter
    whose sidecar describes data that never trained it."""


# --- legacy ingestion (exploratory mode only) --------------------------------


def ingest_documents(paths_in: list[str]) -> tuple[list[str], list[dict]]:
    """Read txt / pdf / jsonl documents to plain text, with per-file SHA-256s
    (the dataset is hashed into the adapter sidecar, like vectors are).

    Exploratory path only: this concatenates whole files and loses row
    identity. Evidence-grade splits go through :mod:`lora_data`.
    """
    texts: list[str] = []
    provenance: list[dict] = []
    for path in paths_in:
        with open(path, "rb") as handle:
            raw = handle.read()
        digest = hashlib.sha256(raw).hexdigest()
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            content = _read_pdf(path)
        elif ext == ".jsonl":
            content = "\n\n".join(
                json.loads(line)["text"]
                for line in raw.decode("utf-8").splitlines() if line.strip())
        else:
            content = raw.decode("utf-8", errors="replace")
        if content.strip():
            texts.append(content)
            provenance.append({"path": path, "hash": digest, "bytes": len(raw)})
    return texts, provenance


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError("PDF ingestion needs pypdf: pip install -e .[lora]") from exc
    reader = PdfReader(path)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _chunk(token_ids: list[int], size: int) -> list[list[int]]:
    return [token_ids[i:i + size] for i in range(0, len(token_ids), size)
            if len(token_ids[i:i + size]) >= 8]


# --- configuration -----------------------------------------------------------


@dataclass
class LoRAConfig:
    """Everything a training run is: the legacy exploratory fields (kept so the
    daemon route's historical call keeps working byte-for-byte) plus the v2
    fields the evidence-grade path resolves as manifest data.

    ``max_sequence_tokens`` supersedes ``max_chunk_tokens``; the two are kept in
    sync so a caller using either spelling gets the same window.
    """

    base_model_id: str
    document_paths: list[str] = field(default_factory=list)
    revision: str | None = None
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    learning_rate: float = 1e-4
    iterations: int = 200
    batch_size: int = 2
    max_chunk_tokens: int = 512
    validation_fraction: float = 0.1
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    dtype: str = "auto"
    device: str | None = None
    output_name: str | None = None

    # --- v2 (contract §5) ---------------------------------------------------
    training_mode: str = LEGACY_INLINE
    train_paths: list[str] = field(default_factory=list)
    validation_paths: list[str] = field(default_factory=list)
    expected_hashes: dict[str, str] = field(default_factory=dict)
    dataset_bundle_id: str | None = None
    dataset_manifest_path: str | None = None
    dataset_manifest_hash: str | None = None
    reserved_evaluation_hashes: list[str] | None = None
    #: Root for workspace-relative split paths (``None`` = the project root).
    dataset_root: str | None = None
    seed: int = 0
    epochs: int = 1
    max_steps: int | None = None
    gradient_accumulation: int = 1
    warmup_steps: int = 0
    lr_schedule: str = "linear"          # "linear" | "constant"
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    #: ``None`` defers to ``max_chunk_tokens`` (the legacy spelling); the two
    #: are reconciled in ``__post_init__`` and always agree afterwards.
    max_sequence_tokens: int | None = None
    long_document_policy: str = lora_data.SPLIT
    chunk_overlap_tokens: int = 64
    eval_interval_steps: int | None = None
    checkpoint_interval_steps: int | None = None
    selection_metric: str | None = None
    evidence_grade: bool = False
    control_arm: dict | None = None
    #: PROVENANCE, not a setting: the direct multiplier a request declared as
    #: ``adapterScale`` (the Swift/MLX ``scale`` convention — no rank in it),
    #: which the wire layer resolved into ``alpha = adapter_scale × rank``
    #: before this config was built. ``None`` = the request declared
    #: ``alpha`` itself. Deliberately outside :meth:`fingerprint`: it changes
    #: nothing about what trains, only records how ``alpha`` was arrived at,
    #: so a resumed run's fingerprint is unaffected.
    requested_adapter_scale: float | None = None

    def __post_init__(self) -> None:
        if self.max_sequence_tokens is None:
            self.max_sequence_tokens = self.max_chunk_tokens
        else:
            self.max_chunk_tokens = self.max_sequence_tokens
        # A stamp that disagrees with the number it explains is a forged (or
        # hand-edited) provenance record: refuse it wherever a config is built,
        # including from a job directory's ``finetune-config.json``.
        if self.requested_adapter_scale is not None:
            resolved = float(self.requested_adapter_scale) * float(self.rank)
            if not math.isclose(float(self.alpha), resolved,
                                rel_tol=1e-9, abs_tol=1e-12):
                raise LoRATrainError(
                    f"requested_adapter_scale {self.requested_adapter_scale} "
                    f"at rank {self.rank} resolves to alpha {resolved}, but "
                    f"alpha is {self.alpha} — the provenance stamp and the "
                    "number it explains disagree; declare exactly one of them")

    def dataset_spec(self) -> LoRADatasetSpec:
        """The frozen :mod:`lora_data` description of this run's dataset.

        Refuses in ``legacy_inline`` mode: that path has no explicit splits to
        describe, which is precisely why it is not evidence.
        """
        if self.training_mode == LEGACY_INLINE:
            raise LoRATrainError(
                "legacy_inline training has no explicit train/validation "
                "split to describe — it is the exploratory path (plan §2.1). "
                "Set trainingMode to 'document' or 'instruction_chat'.")
        if self.training_mode not in TRAINING_MODES:
            raise LoRATrainError(
                f"unknown trainingMode {self.training_mode!r} — expected one "
                f"of {', '.join(TRAINING_MODES)}")
        hashes = {k: v for k, v in (self.expected_hashes or {}).items()}
        return LoRADatasetSpec(
            training_mode=self.training_mode,
            train_files=[DatasetFile(path, hashes.get(path), "train")
                         for path in self.train_paths],
            validation_files=[DatasetFile(path, hashes.get(path), "validation")
                              for path in self.validation_paths],
            max_sequence_tokens=int(self.max_sequence_tokens or 512),
            long_document_policy=self.long_document_policy,
            chunk_overlap_tokens=self.chunk_overlap_tokens,
            bundle_id=self.dataset_bundle_id,
            manifest_path=self.dataset_manifest_path,
            manifest_hash=self.dataset_manifest_hash,
        )

    def fingerprint(self) -> str:
        """SHA-256 over the training-relevant settings — what a resumed run
        must still agree with. Deliberately excludes cosmetics (output name,
        device, log-only fields): resuming on a different GPU is fine,
        resuming under a different learning rate is not."""
        payload = {
            "baseModelID": self.base_model_id,
            "trainingMode": self.training_mode,
            "rank": self.rank, "alpha": self.alpha, "dropout": self.dropout,
            "targetModules": sorted(self.target_modules),
            "learningRate": self.learning_rate,
            "weightDecay": self.weight_decay,
            "maxGradNorm": self.max_grad_norm,
            "lrSchedule": self.lr_schedule,
            "warmupSteps": self.warmup_steps,
            "epochs": self.epochs, "maxSteps": self.max_steps,
            "batchSize": self.batch_size,
            "gradientAccumulation": self.gradient_accumulation,
            # The objective belongs in the fingerprint for the same reason the
            # learning rate does: a checkpoint trained under a different
            # weighting of its targets is a different experiment, and adopting
            # it mid-run would be a silent provenance forgery. Checkpoints
            # written before 2026-09-05 therefore refuse to resume under this
            # code — correctly (external review, 2026-09-05).
            "objective": TRAINING_OBJECTIVE,
            "seed": self.seed,
            "maxSequenceTokens": self.max_sequence_tokens,
            "longDocumentPolicy": self.long_document_policy,
            "chunkOverlapTokens": self.chunk_overlap_tokens,
            "selectionMetric": self.selection_metric,
            "evidenceGrade": self.evidence_grade,
            "controlArm": self.control_arm,
        }
        return _sha256_text(lora_data.canonical_json(payload))


def config_from_dict(payload: dict) -> LoRAConfig:
    """Build a config from a snake_case mapping, ignoring unknown keys.

    The wire layer (Agent C) owns camelCase→snake_case translation; this is the
    last mile, so a route cannot construct a config with a misspelled field
    that silently defaults.
    """
    known = {f.name for f in dataclass_fields(LoRAConfig)}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise LoRATrainError(
            f"unknown LoRA config field(s): {', '.join(unknown)}")
    return LoRAConfig(**{k: v for k, v in payload.items() if k in known})


# --- dtype policy ------------------------------------------------------------


def resolve_training_dtype(config_dtype: str | None, device: str, model_id: str,
                           warn: Callable[[str], None]) -> str:
    """The dtype TRAINING runs in. ``auto`` (or absent) takes the training
    policy (:func:`model_loader.training_dtype` — never fp16); an explicit
    user-provided dtype still wins, with a loud warning when it is float16."""
    from ..steering.model_loader import training_dtype
    if config_dtype and config_dtype.lower() != "auto":
        if config_dtype.lower() in _FP16_NAMES:
            warn("WARNING: explicit float16 training dtype requested — fp16 "
                 "AdamW training is numerically unstable and is the classic "
                 "NaN-adapter factory; prefer bfloat16 (or float32)")
        return config_dtype
    return training_dtype(device, model_id)


def _torch_dtype(dtype_name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16,
            "float32": torch.float32}.get(dtype_name.lower(), torch.float32)


# --- small helpers -----------------------------------------------------------


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str | None:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_full_sha(revision: str | None) -> bool:
    if not revision or len(revision) != _FULL_SHA_LENGTH:
        return False
    return all(character in "0123456789abcdefABCDEF" for character in revision)


def execution_path() -> str:
    """``"slurm"`` under a Slurm allocation, else ``"daemon"``. Evidence-grade
    training belongs on the Slurm path (plan §0.3); the stamp is what makes a
    daemon-resident training run visible as such after the fact."""
    return "slurm" if os.environ.get("SLURM_JOB_ID") else "daemon"


def _slurm_identity() -> dict | None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    return {"jobID": job_id, "jobName": os.environ.get("SLURM_JOB_NAME")}


def _build_identity() -> dict:
    from ..build_identity import build_commit
    token = build_commit()
    if not token:
        return {"commit": None, "dirty": False}
    dirty = token.endswith("-dirty")
    return {"commit": token[:-len("-dirty")] if dirty else token,
            "dirty": dirty}


def _package_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    versions: dict = {}
    for package in ("torch", "transformers", "peft", "accelerate"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    try:
        import torch
        versions["cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001 - torch absent or built without CUDA info
        versions["cuda"] = None
    return versions


def _gpu_identity() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"types": [], "count": 0}
        count = torch.cuda.device_count()
        return {"types": sorted({torch.cuda.get_device_name(index)
                                 for index in range(count)}),
                "count": count}
    except Exception:  # noqa: BLE001 - no torch / no driver
        return {"types": [], "count": 0}


def epoch_seed(seed: int, epoch: int) -> int:
    """The per-epoch shuffle seed. Derived by SHA-256 rather than ``seed +
    epoch`` so two runs whose seeds differ by one do not share every ordering
    but the first, and so the derivation is identical across processes and
    Python builds (``hash()`` on strings is salted; this is not)."""
    return int(_sha256_text(f"{seed}:{epoch}")[:16], 16)


def epoch_order(count: int, seed: int, epoch: int) -> list[int]:
    """The deterministic example ORDER for one epoch, as indices.

    Indices, not rows: file order defines row identity (``rowsRoot`` is
    order-sensitive), so shuffling the rows themselves would change the dataset
    the sidecar claims trained. The sampler permutes the reading order and
    leaves identity alone.
    """
    order = list(range(count))
    random.Random(epoch_seed(seed, epoch)).shuffle(order)
    return order


# --- sidecar -----------------------------------------------------------------


#: The convention a request's ``adapterScale`` speaks: the multiplier itself,
#: no rank in it — the Swift/MLX path's ``scale``. Stamped as
#: ``requestedAdapterScaleConvention`` beside ``requestedAdapterScale`` so a
#: sidecar says both what was asked for and what it became.
DIRECT_ADAPTER_SCALE_CONVENTION = "direct"


def effective_adapter_scale(config: LoRAConfig) -> float | None:
    """The multiplier PEFT actually applies under :data:`ADAPTER_SCALE_CONVENTION`
    — ``alpha / rank``. ``rank`` is a positive integer in every configuration
    PEFT will build; ``None`` rather than a raised ZeroDivisionError is the
    honest answer if it ever is not."""
    return float(config.alpha) / float(config.rank) if config.rank else None


def adapter_sidecar_dict(config: LoRAConfig, *, name: str, provenance: list[dict],
                         train_chunk_count: int, final_loss: float | None,
                         training_dtype_name: str,
                         extra: dict | None = None) -> dict:
    """The adapter's provenance sidecar (written next to the adapter directory).

    ``substrate`` + ``adapterFormat`` are the pinned cross-engine contract:
    this engine stamps ``"python-hf-transformers"`` / ``"hf-peft-lora"``, the
    Swift engine stamps ``"swift-mlx"`` / ``"mlx-lora"``. Loading paths refuse
    an explicit foreign stamp (adapters do not transplant across engines);
    absent = legacy/unknown, attempted as before stamping existed.

    Every v1 key is still written (contract §7). ``extra`` carries the v2
    provenance block the evidence-grade path assembles; the legacy path passes
    none and gets the v1 payload plus the four stamps that let a reader tell an
    exploratory adapter from an evidence-grade one on sight.
    """
    sidecar = {
        "name": name, "baseModelID": config.base_model_id, "revision": config.revision,
        "rank": config.rank, "alpha": config.alpha, "learningRate": config.learning_rate,
        "iterations": config.iterations, "maxChunkTokens": config.max_chunk_tokens,
        "targetModules": config.target_modules, "trainChunks": train_chunk_count,
        "finalLoss": final_loss, "documents": provenance,
        "adapterFormat": ADAPTER_FORMAT,
        "substrate": SUBSTRATE,
        "trainingDtype": training_dtype_name,
        # ``alpha`` above is PEFT's ``lora_alpha`` — a numerator, not the
        # multiplier. Stamped beside it: the convention that turns it into one,
        # and the resolved multiplier itself. Without these a reader compares
        # this adapter's ``alpha`` to the Swift/MLX path's ``scale``, which is a
        # DIRECT multiplier carrying no rank — equal numbers there are different
        # treatments (external review, REM-06). ``rank`` is a positive integer
        # in every configuration PEFT will build; ``None`` rather than a raised
        # ZeroDivisionError is the honest answer if it ever is not.
        "adapterScaleConvention": ADAPTER_SCALE_CONVENTION,
        "effectiveAdapterScale": (float(config.alpha) / float(config.rank)
                                  if config.rank else None),
        # v2 stamps that every mode carries, so "is this evidence?" is
        # answerable from the artifact alone.
        "schemaVersion": SIDECAR_SCHEMA_VERSION,
        "trainingMode": config.training_mode,
        "evidenceGrade": bool(config.evidence_grade),
        "executionPath": execution_path(),
        "dropout": config.dropout,
        "device": config.device,
        # The request's own spelling of the knob when it was not ``alpha``: a
        # direct multiplier (the Swift/MLX ``scale`` convention) that the wire
        # layer resolved into the ``alpha`` above. ``None`` = the request
        # declared ``alpha`` itself. With ``effectiveAdapterScale`` beside it,
        # the sidecar shows the translation on its face: the two agree when
        # the request was honored.
        "requestedAdapterScale": config.requested_adapter_scale,
        "requestedAdapterScaleConvention": (
            DIRECT_ADAPTER_SCALE_CONVENTION
            if config.requested_adapter_scale is not None else None),
    }
    if extra:
        sidecar.update(extra)
    return sidecar


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_json_atomic(path: str, payload: dict) -> None:
    """tmp + fsync + rename (``resume.write_state`` discipline): a crash
    mid-write must never leave a torn state file marking the run."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# --- evidence gates ----------------------------------------------------------


def evidence_refusals(config: LoRAConfig, *, resolved_dtype: str | None = None
                      ) -> list[str]:
    """Every reason this configuration cannot produce evidence, in order.

    Returned as a list (rather than raised one at a time) so a submission path
    can show the researcher the whole gap instead of one refusal per round
    trip. :func:`train` raises on the first.
    """
    if not config.evidence_grade:
        return []
    reasons: list[str] = []
    if config.training_mode == LEGACY_INLINE:
        reasons.append(
            "trainingMode is 'legacy_inline' — the inline-corpus path "
            "concatenates documents across boundaries and never evaluates its "
            "nominal validation split (plan §2.1); evidence-grade training "
            "requires 'document' or 'instruction_chat'")
    if not _is_full_sha(config.revision):
        reasons.append(
            f"revision {config.revision!r} is not a full 40-character commit "
            "sha — a floating revision means the base bytes that trained this "
            "adapter cannot be recovered (plan §2.4)")
    explicit = (config.dtype or "").lower()
    if explicit in _FP16_NAMES or (resolved_dtype or "").lower() in _FP16_NAMES:
        reasons.append(
            "training dtype is float16 — fp16 AdamW training is the classic "
            "NaN-adapter factory and is a loud refusal for evidence "
            "(plan §2.5); use bfloat16")
    if not config.validation_paths:
        reasons.append(
            "no validation files declared — the legacy fractional split is "
            "exploratory only, and a checkpoint cannot be selected on a "
            "metric that was never measured (plan §2.1)")
    if config.selection_metric is None:
        reasons.append(
            "no checkpoint-selection metric declared — the metric is manifest "
            "DATA (plan §0.5), and trainer defaults are for exploratory runs "
            f"only; declare one of {', '.join(SELECTION_METRICS)}")
    elif config.selection_metric not in SELECTION_METRICS:
        reasons.append(
            f"unknown selectionMetric {config.selection_metric!r} — expected "
            f"one of {', '.join(SELECTION_METRICS)}")
    return reasons


def _gate_evidence(config: LoRAConfig, *, resolved_dtype: str | None) -> None:
    reasons = evidence_refusals(config, resolved_dtype=resolved_dtype)
    if reasons:
        raise LoRATrainError(
            "evidence-grade LoRA training refused:\n  - "
            + "\n  - ".join(reasons))


# --- control arm -------------------------------------------------------------


def _apply_control_arm(config: LoRAConfig, loaded: lora_data.LoadedRows,
                       emit: Callable[[str], None]
                       ) -> tuple[lora_data.LoadedRows, dict | None]:
    """Apply the declared S0-analog control to the TRAIN split only.

    Validation is untouched on purpose: the control arm asks "does the same
    schedule on the same volume of construct-free data move the endpoint?", and
    it is answered against the same held-out set as the treatment arm.
    """
    declared = config.control_arm
    if not declared:
        return loaded, None
    kind = declared.get("kind")
    if kind not in CONTROL_ARM_KINDS:
        raise LoRATrainError(
            f"unknown controlArm kind {kind!r} — expected one of "
            f"{', '.join(CONTROL_ARM_KINDS)}")
    stamp = {"kind": kind, "declaredAgainst": declared.get("declaredAgainst"),
             "shuffleEffectiveChangeFraction": None}
    if kind == DECLARED_NEUTRALIZED_DATASET:
        # A declaration about a dataset prepared elsewhere. The trainer stamps
        # it as given and measures nothing: a fraction it did not compute would
        # be a claim it cannot support.
        emit(f"control arm: {kind} (declared against "
             f"{stamp['declaredAgainst']!r}; no trainer-side measure)")
        return loaded, stamp
    if config.training_mode != lora_data.INSTRUCTION_CHAT:
        raise LoRATrainError(
            f"controlArm {SHUFFLED_ASSISTANT_PAIRING!r} re-pairs prompts with "
            f"other rows' assistant replies — it is an "
            f"{lora_data.INSTRUCTION_CHAT} control, and this run is "
            f"{config.training_mode!r}")

    shuffled, fraction = lora_data.shuffle_assistant_pairing(
        loaded.train_rows, config.seed)
    stamp["shuffleEffectiveChangeFraction"] = fraction
    # Redistribute back into the declared files so per-file provenance keeps
    # its shape; each file's rowsRoot is recomputed from the SHUFFLED rows, so
    # the sidecar's roots describe what actually trained (and can never collide
    # with the treatment arm's).
    rebuilt: list[lora_data.LoadedFile] = []
    cursor = 0
    for file in loaded.train_files:
        rows = tuple(shuffled[cursor:cursor + len(file.rows)])
        cursor += len(file.rows)
        rebuilt.append(lora_data.LoadedFile(
            path=file.path, role=file.role, sha256=file.sha256, rows=rows,
            rows_root=lora_data.rows_root([row.row_hash for row in rows])))
    emit(f"control arm: {kind} — {fraction:.1%} of training rows carry a "
         "different assistant reply")
    if fraction < 1.0:
        emit("NOTE: the shuffle left some rows unchanged (duplicate assistant "
             "texts); a low effective-change fraction certifies nothing and is "
             "stamped on the adapter for exactly that reason")
    return (lora_data.LoadedRows(spec=loaded.spec,
                                 evidence_grade=loaded.evidence_grade,
                                 train_files=tuple(rebuilt),
                                 validation_files=loaded.validation_files),
            stamp)


# --- schedule ----------------------------------------------------------------


@dataclass(frozen=True)
class Schedule:
    epochs: int
    max_steps: int | None
    total_steps: int
    warmup_steps: int
    lr_schedule: str
    gradient_accumulation: int
    batch_size: int
    effective_batch_size: int
    seed: int
    eval_interval_steps: int | None
    checkpoint_interval_steps: int | None
    micro_batches_per_epoch: int
    #: What ``gradientAccumulation`` and ``batchSize`` accumulate INTO. Carried
    #: beside them so a sidecar reader never has to infer the denominator from
    #: the code that happened to be deployed (external review, 2026-09-05).
    objective: str = TRAINING_OBJECTIVE

    def to_dict(self) -> dict:
        return {"epochs": self.epochs, "maxSteps": self.max_steps,
                "totalSteps": self.total_steps,
                "warmupSteps": self.warmup_steps,
                "lrSchedule": self.lr_schedule,
                "gradientAccumulation": self.gradient_accumulation,
                "objective": self.objective,
                "batchSize": self.batch_size,
                "effectiveBatchSize": self.effective_batch_size,
                "seed": self.seed,
                "evalIntervalSteps": self.eval_interval_steps,
                "checkpointIntervalSteps": self.checkpoint_interval_steps}


def plan_schedule(config: LoRAConfig, example_count: int) -> Schedule:
    """Resolve the declared epoch/step policy into concrete step counts.

    A "step" is one OPTIMIZER step (``gradient_accumulation`` micro-batches),
    which is what the LR schedule, evaluation interval, and checkpoint interval
    all count — reporting steps in micro-batches would make the same schedule
    read differently under a different accumulation factor.
    """
    if example_count < 1:
        raise LoRATrainError("no training examples")
    batch = max(1, int(config.batch_size))
    accumulation = max(1, int(config.gradient_accumulation))
    micro_per_epoch = math.ceil(example_count / batch)
    steps_per_epoch = math.ceil(micro_per_epoch / accumulation)
    epochs = max(1, int(config.epochs))
    total = steps_per_epoch * epochs
    if config.max_steps is not None:
        total = min(total, max(1, int(config.max_steps)))
    return Schedule(
        epochs=epochs, max_steps=config.max_steps, total_steps=total,
        warmup_steps=max(0, int(config.warmup_steps)),
        lr_schedule=config.lr_schedule, gradient_accumulation=accumulation,
        batch_size=batch, effective_batch_size=batch * accumulation,
        seed=int(config.seed),
        eval_interval_steps=config.eval_interval_steps,
        checkpoint_interval_steps=config.checkpoint_interval_steps,
        micro_batches_per_epoch=micro_per_epoch)


def _lr_lambda(schedule: Schedule) -> Callable[[int], float]:
    warmup, total = schedule.warmup_steps, schedule.total_steps
    constant = schedule.lr_schedule == "constant"

    def factor(current: int) -> float:
        if warmup and current < warmup:
            return current / max(1, warmup)
        if constant:
            return 1.0
        return max(0.0, (total - current) / max(1, total - warmup))

    return factor


# --- checkpoints -------------------------------------------------------------


def checkpoints_directory(run_directory: str) -> str:
    return os.path.join(run_directory, CHECKPOINTS_DIRNAME)


def checkpoint_directory(run_directory: str, step: int) -> str:
    return os.path.join(checkpoints_directory(run_directory), f"step-{step}")


def _complete_checkpoints(run_directory: str) -> list[int]:
    """Steps whose checkpoint directory carries a trainer state — i.e. the
    rename completed. A ``.tmp-step-*`` directory is by construction never
    adopted."""
    root = checkpoints_directory(run_directory)
    steps: list[int] = []
    for entry in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if not entry.startswith("step-"):
            continue
        if not os.path.isfile(os.path.join(root, entry, TRAINER_STATE_FILENAME)):
            continue
        try:
            steps.append(int(entry[len("step-"):]))
        except ValueError:
            continue
    return sorted(steps)


def _prune_checkpoints(run_directory: str, keep: set[int]) -> None:
    """Bounded checkpoint retention: the SELECTED-so-far and the latest. A
    27B adapter checkpoint every N steps fills a /scratch quota fast, and the
    only two a resume or a finalization can need are these."""
    root = checkpoints_directory(run_directory)
    for step in _complete_checkpoints(run_directory):
        if step in keep:
            continue
        shutil.rmtree(os.path.join(root, f"step-{step}"), ignore_errors=True)


def _write_resume_state(run_directory: str, *, step: int, checkpoint: str) -> None:
    """The run-root pointer at the newest COMPLETE checkpoint (contract §8).

    Deliberately not :func:`resume.write_state`: a study run's state counts
    records, a training run's counts optimizer steps, and one shape must not be
    read as the other. The atomicity discipline is the same.
    """
    _write_json_atomic(resume.state_path(run_directory), {
        "runId": os.path.basename(os.path.normpath(run_directory)),
        "verb": RESUME_VERB,
        "step": int(step),
        "checkpoint": checkpoint,
        "updatedAt": _utc_now(),
    })


# --- the trainer -------------------------------------------------------------


def train(config: LoRAConfig, log: Callable[[str], None] | None = None, *,
          run_directory: str | None = None, resume: bool = False,
          checkpoint_flag=None) -> str:
    """Train a LoRA adapter and write it (+ sidecar) to a run directory.

    ``run_directory=None`` mints a fresh immutable run directory, exactly as
    the daemon path always has. The Slurm execute verb passes an EXISTING
    directory plus ``resume=True`` and an installed ``resume.CheckpointFlag``,
    so a requeued job continues the checkpoint it left instead of starting a
    second run.

    (The ``resume`` PARAMETER shadows the module of the same name inside this
    function only; the checkpoint machinery it names lives in the helpers.)

    Returns the run directory. Raises ``resume.CheckpointRequested`` when
    ``checkpoint_flag`` fires (the caller exits with
    ``resume.CHECKPOINT_EXIT_CODE``).
    """
    resuming = bool(resume)

    def emit(message: str) -> None:
        if log is not None:
            log(message)
        print(message, flush=True)

    if config.training_mode not in TRAINING_MODES:
        raise LoRATrainError(
            f"unknown trainingMode {config.training_mode!r} — expected one of "
            f"{', '.join(TRAINING_MODES)}")

    from ..steering.model_loader import resolve_device
    device = resolve_device(config.device)
    # Training dtype policy, NOT the inference default: default_dtype() answers
    # fp16 for non-Gemma models on MPS, which trains straight into NaN adapters.
    dtype_name = resolve_training_dtype(
        config.dtype, device, config.base_model_id, warn=emit)
    _gate_evidence(config, resolved_dtype=dtype_name)

    if config.training_mode == LEGACY_INLINE:
        if resuming:
            raise LoRAResumeError(
                "legacy_inline training is not resumable — it writes no "
                "checkpoints (plan §2.6). Re-run it, or use an evidence-grade "
                "mode.")
        return _train_legacy_inline(config, emit, device=device,
                                    dtype_name=dtype_name,
                                    run_directory=run_directory)
    return _train_split_mode(config, emit, device=device, dtype_name=dtype_name,
                             run_directory=run_directory, resuming=resuming,
                             checkpoint_flag=checkpoint_flag)


def _adapter_name(config: LoRAConfig) -> str:
    return config.output_name or f"lora-{os.path.basename(config.base_model_id)}"


def _refuse_refinalization(run_directory: str, name: str) -> str:
    """At-most-once finalization (plan §2.6): a run directory whose sidecar
    exists is a COMPLETE adapter, and completed runs are immutable."""
    sidecar_path = os.path.join(run_directory, f"{name}.json")
    if os.path.exists(sidecar_path):
        raise LoRATrainError(
            f"{sidecar_path} already exists — this run directory holds a "
            "finalized adapter, and finished runs are immutable. Train into a "
            "fresh directory.")
    return sidecar_path


# --- legacy inline mode ------------------------------------------------------


def _train_legacy_inline(config: LoRAConfig, emit: Callable[[str], None], *,
                         device: str, dtype_name: str,
                         run_directory: str | None) -> str:
    """The historical exploratory path, preserved verbatim.

    Concatenated documents, fixed windows, an unevaluated trailing "validation"
    fraction, no schedule, no checkpoints. It is kept because a smoke test on
    a 4B model should stay a one-liner; it is stamped so nobody can mistake
    its output for evidence.
    """
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError("LoRA training needs peft: pip install -e .[lora]") from exc

    from ..steering.model_loader import hf_dtype_kwargs
    dtype = _torch_dtype(dtype_name)
    name = _adapter_name(config)
    if run_directory is not None:
        _refuse_refinalization(run_directory, name)
    emit(f"loading base model {config.base_model_id} on {device} ({dtype_name})")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, revision=config.revision)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id, revision=config.revision, **hf_dtype_kwargs(dtype))
    model.to(device)
    emit("base model loaded")
    model = get_peft_model(model, LoraConfig(
        r=config.rank, lora_alpha=config.alpha, lora_dropout=config.dropout,
        target_modules=config.target_modules, task_type="CAUSAL_LM"))
    model.train()
    device = next(model.parameters()).device

    texts, provenance = ingest_documents(config.document_paths)
    chunks: list[list[int]] = []
    for text in texts:
        chunks.extend(_chunk(tokenizer(text, add_special_tokens=True).input_ids,
                             config.max_chunk_tokens))
    if not chunks:
        raise RuntimeError("no trainable chunks produced from the documents")
    split = max(1, int(len(chunks) * (1 - config.validation_fraction)))
    train_chunks = chunks[:split]
    emit(
        f"prepared {len(texts)} document(s), {len(train_chunks)} train chunk(s), "
        f"{len(chunks) - len(train_chunks)} validation chunk(s), "
        f"max {config.max_chunk_tokens} tokens/chunk")
    emit(f"starting LoRA training loop for {config.iterations} iterations")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)

    losses: list[float] = []
    step = 0
    while step < config.iterations:
        for start in range(0, len(train_chunks), config.batch_size):
            if step >= config.iterations:
                break
            batch = train_chunks[start:start + config.batch_size]
            width = max(len(c) for c in batch)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            input_ids = torch.tensor(
                [c + [pad_id] * (width - len(c)) for c in batch], device=device)
            labels = input_ids.clone()
            labels[input_ids == pad_id] = -100
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
            if step % 20 == 0 or step == config.iterations:
                emit(f"iteration {step}/{config.iterations}: loss {losses[-1]:.4f}")

    if run_directory is None:
        run_directory = paths.make_unique_run_directory(f"lora-{name}")
    from .run_config import write_run_config
    write_run_config(run_directory, "lora-train", model_id=config.base_model_id,
                     revision=getattr(config, "revision", None),
                     dtype=dtype_name)
    adapter_directory = os.path.join(run_directory, name)
    model.save_pretrained(adapter_directory)  # adapter_model.safetensors + config

    sidecar = adapter_sidecar_dict(
        config, name=name, provenance=provenance, train_chunk_count=len(train_chunks),
        final_loss=losses[-1] if losses else None, training_dtype_name=dtype_name,
        extra={
            "adapterBytesHash": _sha256_file(
                os.path.join(adapter_directory, "adapter_model.safetensors")),
            "adapterConfigHash": _sha256_file(
                os.path.join(adapter_directory, "adapter_config.json")),
            "buildIdentity": _build_identity(),
        })
    _write_json(os.path.join(run_directory, f"{name}.json"), sidecar)
    emit(f"LoRA adapter → {adapter_directory}")
    return run_directory


# --- evidence-grade modes ----------------------------------------------------


def _batch_tensors(examples, *, pad_token_id: int, device):
    import torch
    padded = lora_data.pad_batch(examples, pad_token_id=pad_token_id)
    return (torch.tensor(padded["input_ids"], dtype=torch.long, device=device),
            torch.tensor(padded["labels"], dtype=torch.long, device=device),
            torch.tensor(padded["attention_mask"], dtype=torch.long, device=device))


def _supervised_positions(labels) -> int:
    """Positions that actually contribute to the causal-LM loss: HF shifts
    labels by one, so position 0 never does — and padding is ``-100`` by
    construction (:func:`lora_data.pad_batch`), so padding never does either."""
    return int((labels[:, 1:] != lora_data.IGNORE_LABEL).sum().item())


def _supervised_loss_sum(model, *, input_ids, labels, attention):
    """Cross-entropy SUMMED (not averaged) over this batch's supervised targets.

    Computed here rather than taken from ``model(..., labels=...).loss`` because
    the sum is the quantity the objective composes: the caller divides ONE
    total by ONE denominator (the whole optimizer-step group's target count),
    which is what makes the gradient independent of the micro-batch cut. Taking
    HF's mean and multiplying it back by this batch's target count would give
    the same number today — ``test_lora_train_objective`` pins that identity for
    the installed transformers — but it would silently re-weight every run the
    day HF changes its denominator (it already has one: ``num_items_in_batch``).

    The shift, the ``-100`` ignore index and the float upcast mirror
    :func:`transformers.loss.loss_utils.ForCausalLMLoss` exactly, so the numbers
    stay comparable to anything computed the HF way, and the memory profile is
    unchanged: HF materializes the same upcast logits to compute its own loss.
    Labels are not passed to the model, so that loss is not computed twice.
    (external review, 2026-09-05)
    """
    import torch
    logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
    # Shift the LABELS (as HF does) rather than slicing ``logits[:, :-1]``:
    # slicing would force a copy of the whole (batch × sequence × vocab)
    # tensor, which at a 150k vocab is the largest allocation in the step.
    # The appended ignore column is the position nothing predicts.
    shifted = torch.nn.functional.pad(
        labels, (0, 1), value=lora_data.IGNORE_LABEL)[..., 1:]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), shifted.reshape(-1),
        ignore_index=lora_data.IGNORE_LABEL, reduction="sum")


def _evaluate(model, examples, *, pad_token_id: int, device, batch_size: int
              ) -> tuple[float | None, int]:
    """Full-validation-set loss and the denominator it was divided by.

    The SAME objective the training loop optimizes (see "The objective" in the
    module docstring): the summed token losses over every supervised target in
    the split, divided by their count. Weighted by supervised token count
    rather than averaged over batches, so the number is a property of the
    validation set and not of how it happened to be batched — and directly
    comparable to a training row, which is the point of returning the count
    alongside it (external review, 2026-09-05).
    """
    import torch
    if not examples:
        return None, 0
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            input_ids, labels, attention = _batch_tensors(
                batch, pad_token_id=pad_token_id, device=device)
            tokens = _supervised_positions(labels)
            if tokens == 0:
                continue
            loss_sum = _supervised_loss_sum(model, input_ids=input_ids,
                                            labels=labels, attention=attention)
            total_loss += float(loss_sum.detach().cpu())
            total_tokens += tokens
    if was_training:
        model.train()
    if not total_tokens:
        return None, 0
    return total_loss / total_tokens, total_tokens


def _train_split_mode(config: LoRAConfig, emit: Callable[[str], None], *,
                      device: str, dtype_name: str, run_directory: str | None,
                      resuming: bool, checkpoint_flag) -> str:
    try:
        import torch
        from peft import (LoraConfig, get_peft_model,
                          set_peft_model_state_dict)
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError("LoRA training needs peft: pip install -e .[lora]") from exc

    from ..steering.model_loader import cached_revision, hf_dtype_kwargs
    started_at = _utc_now()
    name = _adapter_name(config)

    # --- revision ------------------------------------------------------------
    requested_revision = config.revision
    resolved_revision = config.revision or cached_revision(config.base_model_id)
    if not config.revision and resolved_revision:
        emit(f"no revision requested — pinning the locally cached snapshot "
             f"{resolved_revision} (exploratory)")

    # --- dataset (before the model: a dataset refusal must cost no GPU) -------
    spec = config.dataset_spec()
    dataset_root = config.dataset_root or paths.project_root()
    loaded = lora_data.load_split_rows(spec, evidence_grade=config.evidence_grade,
                                       root=dataset_root)
    loaded, control_stamp = _apply_control_arm(config, loaded, emit)

    emit(f"loading tokenizer {config.base_model_id} "
         f"({resolved_revision or 'unpinned'})")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id,
                                              revision=config.revision)
    tokenized = lora_data.tokenize_rows(loaded, tokenizer,
                                        model_id=config.base_model_id, spec=spec)
    train_examples = list(tokenized.train)
    validation_examples = list(tokenized.validation)
    if not train_examples:
        raise LoRATrainError("the training split tokenized to no examples")
    schedule = plan_schedule(config, len(train_examples))
    emit(f"prepared {len(train_examples)} train example(s), "
         f"{len(validation_examples)} validation example(s); "
         f"{schedule.total_steps} optimizer step(s) over {schedule.epochs} "
         f"epoch(s) at effective batch {schedule.effective_batch_size}")

    dataset_block = lora_data.dataset_manifest_dict(loaded, tokenized)
    template_block = dataset_block.pop("template", None)
    dataset_block.pop("trainingMode", None)
    dataset_block["reservedEvaluationHashes"] = (
        list(config.reserved_evaluation_hashes)
        if config.reserved_evaluation_hashes is not None else None)
    dataset_files = {file.path: file.sha256
                     for file in loaded.train_files + loaded.validation_files}
    dataset_fingerprint = _sha256_text(lora_data.canonical_json(
        [{"path": file.path, "sha256": file.sha256, "rowsRoot": file.rows_root}
         for file in loaded.train_files + loaded.validation_files]))
    config_hash = config.fingerprint()

    # --- run directory -------------------------------------------------------
    if run_directory is None:
        if resuming:
            raise LoRAResumeError(
                "resume=True needs the run directory to resume INTO — a fresh "
                "directory has nothing to adopt")
        run_directory = paths.make_unique_run_directory(f"lora-{name}")
    else:
        os.makedirs(run_directory, exist_ok=True)
    sidecar_path = _refuse_refinalization(run_directory, name)
    os.makedirs(checkpoints_directory(run_directory), exist_ok=True)

    adopted: dict | None = None
    adopted_directory: str | None = None
    if resuming:
        adopted, adopted_directory = _adopt_checkpoint(
            run_directory, dataset_fingerprint=dataset_fingerprint,
            model_id=config.base_model_id, revision=resolved_revision,
            config_hash=config_hash)
        emit(f"resuming from {adopted_directory} at step {adopted['step']}")

    # --- model ---------------------------------------------------------------
    torch.manual_seed(int(config.seed))
    dtype = _torch_dtype(dtype_name)
    emit(f"loading base model {config.base_model_id} on {device} ({dtype_name})")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id, revision=config.revision, **hf_dtype_kwargs(dtype))
    model_class = type(base_model).__name__
    model_config_hash = _sha256_text(base_model.config.to_json_string())
    base_model.to(device)
    emit("base model loaded")
    model = get_peft_model(base_model, LoraConfig(
        r=config.rank, lora_alpha=config.alpha, lora_dropout=config.dropout,
        target_modules=config.target_modules, task_type="CAUSAL_LM"))
    model.train()
    torch_device = next(model.parameters()).device
    pad_token_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                    else (tokenizer.eos_token_id or 0))

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  _lr_lambda(schedule))

    # --- state ---------------------------------------------------------------
    # ``objective`` names what every "loss" in this file was divided by, so the
    # history is readable without knowing which build wrote it (external
    # review, 2026-09-05).
    history: dict = {"objective": TRAINING_OBJECTIVE,
                     "train": [], "validation": [], "checkpoints": []}
    evaluated_steps: set[int] = set()
    best: dict | None = None
    resume_lineage: list[dict] = []
    step = 0
    start_epoch = 0
    start_position = 0

    if adopted is not None:
        set_peft_model_state_dict(model, _load_adapter_state(adopted_directory))
        optimizer.load_state_dict(
            torch.load(os.path.join(adopted_directory, "optimizer.pt"),
                       map_location=torch_device, weights_only=False))
        scheduler.load_state_dict(
            torch.load(os.path.join(adopted_directory, "scheduler.pt"),
                       map_location="cpu", weights_only=False))
        rng_state = torch.load(os.path.join(adopted_directory, "rng-state.pt"),
                               map_location="cpu", weights_only=False)
        _restore_rng(rng_state)
        history = adopted.get("history") or history
        history.setdefault("objective", TRAINING_OBJECTIVE)
        evaluated_steps = set(adopted.get("evaluatedSteps") or [])
        best = adopted.get("best")
        resume_lineage = list(adopted.get("resumeLineage") or [])
        step = int(adopted["step"])
        start_epoch = int(adopted["epoch"])
        start_position = int(adopted["positionInEpoch"])
        resume_lineage.append({"resumedAtStep": step,
                               "fromCheckpoint": os.path.relpath(
                                   adopted_directory, run_directory),
                               "timestamp": _utc_now()})

    shuffle_rng_state = random.Random(epoch_seed(schedule.seed, start_epoch)).getstate()

    def snapshot_state(current_step: int, epoch: int, position: int) -> dict:
        return {
            "schemaVersion": 1,
            "verb": RESUME_VERB,
            "step": current_step,
            "epoch": epoch,
            "positionInEpoch": position,
            "totalSteps": schedule.total_steps,
            "best": best,
            "history": history,
            "evaluatedSteps": sorted(evaluated_steps),
            "resumeLineage": resume_lineage,
            "datasetFingerprint": dataset_fingerprint,
            "datasetFiles": dataset_files,
            "modelID": config.base_model_id,
            "revisionResolved": resolved_revision,
            "configHash": config_hash,
            "trainingDtype": dtype_name,
            "updatedAt": _utc_now(),
        }

    def write_checkpoint(current_step: int, epoch: int, position: int) -> str:
        directory = _write_checkpoint(
            model, optimizer, scheduler, run_directory=run_directory,
            step=current_step, state=snapshot_state(current_step, epoch, position),
            shuffle_rng_state=shuffle_rng_state)
        entry = {
            "step": current_step,
            "path": os.path.relpath(directory, run_directory),
            "metric": config.selection_metric,
            "value": (best or {}).get("value")
            if (best or {}).get("step") == current_step else None,
        }
        # One entry per step: a new-best evaluation and the epoch boundary can
        # both land on the same step, and a duplicated history row would read
        # as two checkpoints where there is one directory.
        existing = next((index for index, item in enumerate(history["checkpoints"])
                         if item["step"] == current_step), None)
        if existing is None:
            history["checkpoints"].append(entry)
        else:
            history["checkpoints"][existing] = entry
        keep = {current_step}
        if best is not None:
            keep.add(int(best["step"]))
        _prune_checkpoints(run_directory, keep)
        _write_resume_state(run_directory, step=current_step,
                            checkpoint=os.path.relpath(directory, run_directory))
        return directory

    def run_evaluation(current_step: int, epoch: int, position: int) -> None:
        nonlocal best
        if current_step in evaluated_steps or not validation_examples:
            return
        value, tokens = _evaluate(model, validation_examples,
                                  pad_token_id=pad_token_id,
                                  device=torch_device,
                                  batch_size=schedule.batch_size)
        evaluated_steps.add(current_step)
        if value is None:
            return
        history["validation"].append({
            "step": current_step, "loss": value,
            "supervisedTokens": tokens,
            "lossDenominator": "supervisedTokens"})
        # Strict improvement, so a tie keeps the EARLIER step: later steps have
        # trained longer on the same data, and preferring them on a tie is a
        # silent bias toward overfitting.
        improved = best is None or value < float(best["value"])
        emit(f"step {current_step}: validation loss {value:.5f}"
             + (" (best)" if improved else ""))
        if improved:
            best = {"step": current_step, "value": value}
            # A selected checkpoint must EXIST: evaluation and checkpointing
            # are co-scheduled at every new best so the final adapter can carry
            # the selected weights even when the interval cadence would not
            # have written one here.
            write_checkpoint(current_step, epoch, position)

    # --- the loop ------------------------------------------------------------
    accumulation = schedule.gradient_accumulation
    finished = False
    for epoch in range(start_epoch, schedule.epochs):
        rng = random.Random(epoch_seed(schedule.seed, epoch))
        order = list(range(len(train_examples)))
        rng.shuffle(order)
        shuffle_rng_state = rng.getstate()
        micro_index = start_position if epoch == start_epoch else 0
        while micro_index < schedule.micro_batches_per_epoch:
            optimizer.zero_grad(set_to_none=True)
            # The objective's denominator is fixed BEFORE the first backward
            # pass. The group's micro-batches are known from ``order`` without
            # any forward, and ``_batch_tensors`` needs only the labels, so the
            # group's supervised-target count can be totalled up front — which
            # is what lets each micro-batch contribute its SUM scaled by one
            # shared 1/N. Dividing per micro-batch instead (by its own mean, or
            # by the nominal ``accumulation``) weights the objective by the
            # partitioning and under-scales a partial final group
            # (external review, 2026-09-05).
            #
            # Only the integer index/label/mask tensors are held across the
            # group — activations are still freed by each backward, so gradient
            # accumulation keeps the memory profile it exists for.
            group: list[tuple] = []
            group_targets = 0
            group_sequence_tokens = 0
            for _ in range(accumulation):
                if micro_index >= schedule.micro_batches_per_epoch:
                    break
                indices = order[micro_index * schedule.batch_size:
                                (micro_index + 1) * schedule.batch_size]
                batch = [train_examples[index] for index in indices]
                micro_index += 1
                if not batch:
                    continue
                tensors = _batch_tensors(batch, pad_token_id=pad_token_id,
                                         device=torch_device)
                targets = _supervised_positions(tensors[1])
                if targets == 0:
                    # Nothing to learn from and no denominator to contribute:
                    # forwarding it would divide by zero inside the loss.
                    continue
                group.append(tensors)
                group_targets += targets
                group_sequence_tokens += int(tensors[2].sum().item())
            group_loss_sum = 0.0
            for input_ids, labels, attention in group:
                loss_sum = _supervised_loss_sum(
                    model, input_ids=input_ids, labels=labels,
                    attention=attention)
                (loss_sum / group_targets).backward()
                group_loss_sum += float(loss_sum.detach().cpu())
            # Clipping sees the FULLY accumulated gradient — the norm of one
            # optimizer step's update, not of a fraction of it.
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config.max_grad_norm)
            current_lr = float(scheduler.get_last_lr()[0])
            # Exactly one optimizer and one scheduler step per logical group,
            # empty group included: the step count IS the schedule contract
            # that checkpoint/resume and the LR curve are pinned to.
            optimizer.step()
            scheduler.step()
            step += 1
            history["train"].append({
                "step": step,
                # The objective itself: this group's summed token losses over
                # this group's supervised-target count. Named denominators,
                # because the old "loss"/"tokens" pair reported a mean of
                # micro-batch means beside a count of every non-padding
                # position, prompts included — two different denominators,
                # neither of them stated.
                "loss": (group_loss_sum / group_targets
                         if group_targets else None),
                "lr": current_lr,
                "supervisedTokens": group_targets,
                "lossDenominator": "supervisedTokens",
                # NOT the denominator: every non-padding position in the group,
                # prompt tokens included. Kept for throughput accounting only.
                "sequenceTokens": group_sequence_tokens,
            })
            if schedule.eval_interval_steps and \
                    step % schedule.eval_interval_steps == 0:
                run_evaluation(step, epoch, micro_index)
            if schedule.checkpoint_interval_steps and \
                    step % schedule.checkpoint_interval_steps == 0:
                write_checkpoint(step, epoch, micro_index)
            if checkpoint_flag is not None and checkpoint_flag.requested:
                # BETWEEN optimizer steps: gradients are consumed, the state is
                # coherent, and the resumed run continues at exactly this
                # micro-batch.
                directory = write_checkpoint(step, epoch, micro_index)
                emit(f"checkpoint (signal): step {step} → {directory}; exit "
                     f"{resume.CHECKPOINT_EXIT_CODE}")
                raise resume.CheckpointRequested(run_directory, RESUME_VERB,
                                                 step, reason="signal")
            if step >= schedule.total_steps:
                finished = True
                break
        # Epoch boundary: evaluate and checkpoint unconditionally (plan §2.5).
        boundary_position = min(micro_index, schedule.micro_batches_per_epoch)
        next_epoch = epoch + 1 if boundary_position >= schedule.micro_batches_per_epoch else epoch
        next_position = 0 if next_epoch != epoch else boundary_position
        run_evaluation(step, next_epoch, next_position)
        write_checkpoint(step, next_epoch, next_position)
        if finished:
            break
        start_position = 0

    # A final evaluation at the last step, when the schedule stopped somewhere
    # the interval did not land on.
    run_evaluation(step, schedule.epochs, 0)

    # --- selection -----------------------------------------------------------
    if config.selection_metric == VALIDATION_LOSS and best is not None:
        selected = {"step": int(best["step"]), "metric": VALIDATION_LOSS,
                    "value": float(best["value"]), "reason": "minValidationLoss"}
        if int(best["step"]) != step:
            source = checkpoint_directory(run_directory, int(best["step"]))
            emit(f"selected checkpoint step {best['step']} (validation loss "
                 f"{best['value']:.5f}) — reloading its weights for the final "
                 f"adapter")
            set_peft_model_state_dict(model, _load_adapter_state(source))
    else:
        last_value = history["validation"][-1]["loss"] if history["validation"] else None
        selected = {"step": step, "metric": config.selection_metric,
                    "value": last_value, "reason": "lastStep(exploratory)"}
        emit(f"selected the last step ({step}) — no validation-loss selection "
             "metric was declared")

    # --- finalize ------------------------------------------------------------
    from .run_config import write_run_config
    write_run_config(run_directory, "lora-train", model_id=config.base_model_id,
                     revision=resolved_revision, dtype=dtype_name)
    adapter_directory = os.path.join(run_directory, name)
    model.save_pretrained(adapter_directory)

    _write_json(os.path.join(run_directory, HISTORY_FILENAME), history)
    final_train_loss = next(
        (entry["loss"] for entry in reversed(history["train"])
         if entry["loss"] is not None), None)
    sidecar = adapter_sidecar_dict(
        config, name=name,
        provenance=[{"path": file.path, "hash": file.sha256, "rows": len(file.rows)}
                    for file in loaded.train_files + loaded.validation_files],
        train_chunk_count=len(train_examples), final_loss=final_train_loss,
        training_dtype_name=dtype_name,
        extra={
            "dataset": dataset_block,
            "template": template_block,
            "revisionRequested": requested_revision,
            "revisionResolved": resolved_revision,
            "tokenizerSource": config.base_model_id,
            "tokenizerRevision": resolved_revision,
            "modelClass": model_class,
            "modelConfigHash": model_config_hash,
            "buildIdentity": _build_identity(),
            "optimizerSettings": {
                "optimizer": "adamw", "learningRate": config.learning_rate,
                "weightDecay": config.weight_decay,
                "maxGradNorm": config.max_grad_norm},
            "schedule": schedule.to_dict(),
            "selectedCheckpoint": selected,
            "historyFile": HISTORY_FILENAME,
            "packageVersions": _package_versions(),
            "gpu": _gpu_identity(),
            "slurm": _slurm_identity(),
            "timestamps": {"start": started_at, "end": _utc_now()},
            "resumeLineage": resume_lineage,
            "adapterBytesHash": _sha256_file(
                os.path.join(adapter_directory, "adapter_model.safetensors")),
            "adapterConfigHash": _sha256_file(
                os.path.join(adapter_directory, "adapter_config.json")),
            "controlArm": control_stamp,
        })
    _write_json(sidecar_path, sidecar)
    # The run is complete: clear the resume pointer so a stray requeue cannot
    # adopt a finished run (the sidecar refusal is the second line of defense).
    resume.clear_state(run_directory)
    emit(f"LoRA adapter → {adapter_directory}")
    return run_directory


# --- checkpoint I/O ----------------------------------------------------------


def _load_adapter_state(directory: str) -> dict:
    from safetensors.torch import load_file
    path = os.path.join(directory, "adapter_model.safetensors")
    if not os.path.isfile(path):
        raise LoRAResumeError(
            f"{path} is missing — the checkpoint has no adapter weights")
    return load_file(path)


def _rng_state() -> dict:
    import torch
    return {"torchCPU": torch.get_rng_state(),
            "torchCUDA": (torch.cuda.get_rng_state_all()
                          if torch.cuda.is_available() else None)}


def _restore_rng(state: dict) -> None:
    import torch
    cpu = state.get("torchCPU")
    if cpu is not None:
        torch.set_rng_state(cpu.cpu() if hasattr(cpu, "cpu") else cpu)
    cuda = state.get("torchCUDA")
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda)


def _write_checkpoint(model, optimizer, scheduler, *, run_directory: str,
                      step: int, state: dict, shuffle_rng_state) -> str:
    """Write ``checkpoints/step-<n>/`` atomically: everything lands in a
    ``.tmp-step-<n>`` directory that is renamed into place only once complete,
    so an interrupted checkpoint is never adoptable (the adoption scan requires
    ``trainer-state.json``, which is written last)."""
    import torch
    root = checkpoints_directory(run_directory)
    os.makedirs(root, exist_ok=True)
    tmp = os.path.join(root, f".tmp-step-{step}")
    final = os.path.join(root, f"step-{step}")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    model.save_pretrained(tmp)
    torch.save(optimizer.state_dict(), os.path.join(tmp, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(tmp, "scheduler.pt"))
    rng = _rng_state()
    rng["shuffleRandom"] = shuffle_rng_state
    torch.save(rng, os.path.join(tmp, "rng-state.pt"))
    _write_json_atomic(os.path.join(tmp, TRAINER_STATE_FILENAME), state)
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    return final


def read_trainer_state(directory: str) -> dict:
    path = os.path.join(directory, TRAINER_STATE_FILENAME)
    try:
        with open(path, encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise LoRAResumeError(f"{path}: unreadable trainer state: {exc}") from exc
    if not isinstance(state, dict):
        raise LoRAResumeError(f"{path}: trainer state is not an object")
    return state


def _adopt_checkpoint(run_directory: str, *, dataset_fingerprint: str,
                      model_id: str, revision: str | None,
                      config_hash: str) -> tuple[dict, str]:
    """Adopt the newest COMPLETE checkpoint, or refuse.

    The four verifications are the whole point of the mechanism: an adapter
    finished from a checkpoint trained on different data, a different base
    revision, or a different schedule would carry a sidecar describing a run
    that never happened.
    """
    steps = _complete_checkpoints(run_directory)
    pointed = resume.read_state(run_directory) or {}
    if pointed.get("checkpoint"):
        candidate = os.path.join(run_directory, pointed["checkpoint"])
        if os.path.isfile(os.path.join(candidate, TRAINER_STATE_FILENAME)):
            directory = candidate
        else:
            directory = None
    else:
        directory = None
    if directory is None:
        if not steps:
            raise LoRAResumeError(
                f"{run_directory} has no complete checkpoint to resume — "
                "start a fresh run directory instead")
        directory = checkpoint_directory(run_directory, steps[-1])

    state = read_trainer_state(directory)
    mismatches: list[str] = []
    if state.get("datasetFingerprint") != dataset_fingerprint:
        mismatches.append(
            f"dataset fingerprint {state.get('datasetFingerprint')} != "
            f"{dataset_fingerprint} (the split files or their row order "
            "changed since the checkpoint)")
    if state.get("modelID") != model_id:
        mismatches.append(
            f"base model {state.get('modelID')!r} != {model_id!r}")
    if state.get("revisionResolved") != revision:
        mismatches.append(
            f"resolved revision {state.get('revisionResolved')!r} != "
            f"{revision!r} (different base bytes)")
    if state.get("configHash") != config_hash:
        mismatches.append(
            f"config fingerprint {state.get('configHash')} != {config_hash} "
            "(a training setting changed since the checkpoint)")
    if mismatches:
        raise LoRAResumeError(
            f"refusing to resume {directory}:\n  - " + "\n  - ".join(mismatches))
    for required in ("optimizer.pt", "scheduler.pt", "rng-state.pt"):
        if not os.path.isfile(os.path.join(directory, required)):
            raise LoRAResumeError(
                f"{directory}: incomplete checkpoint — {required} is missing")
    return state, directory
