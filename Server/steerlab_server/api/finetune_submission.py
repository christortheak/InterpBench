"""The v2 fine-tune wire layer: request parsing, the training PLAN, the
LoRA-shaped preflight, and Slurm submission.

Three routes share this module (implementation contract §6):

- ``POST /api/finetune/plan``   — normalize a request into a plan + planHash.
  No side effects, no tokenizer, no model: the plan is what a researcher
  confirms BEFORE a queue allocation is spent.
- ``POST /api/finetune/train``  — the daemon-resident job (exploratory only).
- ``POST /api/finetune/submit`` — the EVIDENCE-GRADE path: a Slurm job with
  bundle, preflight, checkpoint/resume and auto-resubmit, exactly like study
  submission (readiness plan §0 amendment 3; a daemon-resident training job
  bypasses cluster survivability and occupies the controller).

Everything here is wire-shaped: camelCase in, camelCase out, snake_case
inside. Refusals are typed (:class:`FineTuneRequestError`), loud, and name the
offending key/file — a silently defaulted hyperparameter is a training run
nobody can defend afterwards.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from ..experiment import lora_data, lora_train, paths
from . import housekeeping
from .executors import SlurmExecutor, SlurmResources, _parse_walltime
from .profile import ServerProfile
from .submissions import (PREFLIGHT_WALLTIME_MARGIN,
                          PREFLIGHT_WALLTIME_WARN_FRACTION, PreflightRejection,
                          _check, _check_gpu_request, _check_maintenance,
                          _gate_on_preflight, _gb, _gres_gpu_count,
                          _resources_from_dict, _stamped_resources,
                          _root_stays_put, _verdict_of)


class FineTuneRequestError(ValueError):
    """A malformed or refused fine-tune request (HTTP 400 at the route)."""


#: The job kind recorded for a Slurm fine-tune (mirrors ``study-submit``).
JOB_KIND = "finetune-train"

#: The v2 request body's top-level keys. ``resources``/``force`` are accepted
#: only by the submit route (the client flattens the request and appends them).
_REQUEST_KEYS = {
    "schemaVersion", "baseModelID", "revision", "name", "trainingMode",
    "evidenceGrade", "dataset", "hyperparameters", "selectionMetric",
    "controlArm", "expectedPlanHash",
}
_SUBMIT_ONLY_KEYS = {"resources", "force", "dryRun"}
_DATASET_KEYS = {"bundleID", "manifestPath", "manifestHash", "files"}
_DATASET_FILE_KEYS = {"role", "path", "sha256", "content"}
_CONTROL_ARM_KEYS = {"kind", "declaredAgainst"}

#: Wire hyperparameter name → ``LoRAConfig`` field. The key SET is the
#: contract's (the original 20 plus ``adapterScale``); an unknown key is a
#: refusal, never a default. ``alpha`` and ``adapterScale`` are two
#: conventions for ONE knob — see :func:`resolve_adapter_scale`, which turns
#: whichever was declared into the trainer's ``alpha`` and refuses both.
_HYPERPARAMETERS = {
    "rank": ("rank", int),
    "alpha": ("alpha", float),
    "adapterScale": ("requested_adapter_scale", float),
    "dropout": ("dropout", float),
    "learningRate": ("learning_rate", float),
    "epochs": ("epochs", int),
    "maxSteps": ("max_steps", int),
    "batchSize": ("batch_size", int),
    "gradientAccumulation": ("gradient_accumulation", int),
    "warmupSteps": ("warmup_steps", int),
    "lrSchedule": ("lr_schedule", str),
    "maxGradNorm": ("max_grad_norm", float),
    "weightDecay": ("weight_decay", float),
    "seed": ("seed", int),
    "maxSequenceTokens": ("max_sequence_tokens", int),
    "longDocumentPolicy": ("long_document_policy", str),
    "chunkOverlapTokens": ("chunk_overlap_tokens", int),
    "evalIntervalSteps": ("eval_interval_steps", int),
    "checkpointIntervalSteps": ("checkpoint_interval_steps", int),
    "targetModules": ("target_modules", list),
    "dtype": ("dtype", str),
}

#: Wire ``trainingMode`` spellings → the python/artifact spelling. The wire is
#: camelCase (``instructionChat``); the artifact spelling is accepted too so a
#: config file can be POSTed verbatim.
_TRAINING_MODES = {
    "document": lora_data.DOCUMENT,
    "instructionChat": lora_data.INSTRUCTION_CHAT,
    "instruction_chat": lora_data.INSTRUCTION_CHAT,
}

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

#: Where a submission stages the dataset bytes and the training run inside its
#: job directory. Both are relative to the submission directory so the whole
#: job is one movable, self-contained tree.
DATASET_DIRNAME = "dataset"
RUN_DIRNAME = "run"
CONFIG_FILENAME = "finetune-config.json"
PLAN_FILENAME = "plan.json"


# --- request parsing ---------------------------------------------------------


@dataclass
class ParsedRequest:
    """A validated v2 body: the trainer config plus the wire-only extras."""

    config: lora_train.LoRAConfig
    name: str | None
    evidence_grade: bool
    expected_plan_hash: str | None
    #: (declared relative path, sha256, role, inline content or None)
    files: list[tuple[str, str, str, str | None]]
    resources: dict
    force: bool
    dry_run: bool


def _require_dict(value: object, *, key: str) -> dict:
    if not isinstance(value, dict):
        raise FineTuneRequestError(
            f"{key} must be a JSON object (got "
            f"{'null' if value is None else type(value).__name__})")
    return value


def _refuse_unknown(payload: dict, allowed: set[str], *, where: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise FineTuneRequestError(
            f"unknown {where} key(s): {', '.join(unknown)} — expected only "
            f"{', '.join(sorted(allowed))}")


def _coerce(value: object, kind: type, *, key: str):
    """Wire value → python, refusing loudly. ``None`` never reaches here (a
    present-null key is dropped upstream, so it reads exactly like absent)."""
    if kind is str:
        if not isinstance(value, str):
            raise FineTuneRequestError(f"{key} must be a string")
        return value
    if kind is list:
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise FineTuneRequestError(f"{key} must be a list of strings")
        return list(value)
    if kind is int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FineTuneRequestError(f"{key} must be a number")
        if isinstance(value, float) and value != int(value):
            raise FineTuneRequestError(f"{key} must be a whole number")
        return int(value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FineTuneRequestError(f"{key} must be a number")
    return float(value)


def resolve_adapter_scale(*, alpha: object, adapter_scale: object,
                          rank: int) -> tuple[float | None, float | None]:
    """The LoRA strength knob has two conventions, and a request may use ONE.

    ``alpha`` is PEFT's ``lora_alpha`` — a numerator the trainer divides by
    ``rank`` (:data:`lora_train.ADAPTER_SCALE_CONVENTION`). ``adapterScale``
    is the multiplier itself — the Swift/MLX ``scale`` convention, no rank in
    it. Returns ``(alpha, requested_adapter_scale)``: the ``alpha`` the trainer
    will use (``None`` = neither was declared, so the dataclass default
    applies) and the direct value it was resolved from (``None`` = the request
    spoke ``alpha``). Both present is refused, not reconciled: two spellings
    of one knob that could disagree are exactly the ambiguity the second
    spelling exists to remove.
    """
    if alpha is not None and adapter_scale is not None:
        raise FineTuneRequestError(
            "hyperparameters declares both alpha and adapterScale — they are "
            "two conventions for one knob (alpha is PEFT's lora_alpha, a "
            "numerator the trainer divides by rank; adapterScale is the "
            "multiplier itself, the Swift/MLX scale); declare exactly one")
    if adapter_scale is None:
        return alpha, None
    try:
        value = float(adapter_scale)
    except (TypeError, ValueError):
        raise FineTuneRequestError(
            "hyperparameters.adapterScale must be a number") from None
    if not value > 0:  # also catches NaN
        raise FineTuneRequestError(
            f"hyperparameters.adapterScale must be positive (got "
            f"{adapter_scale!r}) — it is the multiplier applied to the "
            "adapter update, and zero trains nothing")
    return value * float(rank), value


def adapter_scale_block(config: lora_train.LoRAConfig) -> dict:
    """What the LoRA strength knob resolves to, on the plan's face — so the
    researcher confirms a MULTIPLIER, not a numerator whose meaning depends on
    knowing PEFT's convention. Same spellings as the adapter sidecar's stamps,
    so plan and artifact read alike."""
    requested = config.requested_adapter_scale
    return {
        "rank": config.rank,
        "alpha": config.alpha,
        "adapterScaleConvention": lora_train.ADAPTER_SCALE_CONVENTION,
        "effectiveAdapterScale": lora_train.effective_adapter_scale(config),
        "requestedAdapterScale": requested,
        "requestedAdapterScaleConvention": (
            lora_train.DIRECT_ADAPTER_SCALE_CONVENTION
            if requested is not None else None),
    }


def _relative_dataset_path(path: object) -> str:
    """Dataset files are WORKSPACE-RELATIVE by contract (the Mac workspace is
    the source of truth, and a job directory has to be movable). An absolute
    or escaping path is refused rather than silently rewritten."""
    if not isinstance(path, str) or not path.strip():
        raise FineTuneRequestError("dataset file 'path' must be a non-empty string")
    if "\0" in path:
        raise FineTuneRequestError(f"dataset file path {path!r} contains a NUL")
    if os.path.isabs(path):
        raise FineTuneRequestError(
            f"dataset file path {path!r} is absolute — dataset paths are "
            "workspace-relative (e.g. 'adapters/<package>/training/train.jsonl') "
            "so the pinned bundle reads the same on the Mac and on the cluster")
    normalized = os.path.normpath(path)
    if normalized == "." or normalized.startswith(".." + os.sep) or normalized == "..":
        raise FineTuneRequestError(
            f"dataset file path {path!r} escapes the dataset root")
    return normalized


def parse_request(body: dict, *, allow_submit_keys: bool = False) -> ParsedRequest:
    """Validate a v2 body and build the trainer config.

    Every optional key encodes as an explicit JSON ``null`` on the wire; a
    present-null is dropped here, so it reads exactly like an absent key and
    the dataclass default applies.
    """
    body = _require_dict(body, key="request body")
    allowed = set(_REQUEST_KEYS)
    if allow_submit_keys:
        allowed |= _SUBMIT_ONLY_KEYS
    _refuse_unknown(body, allowed, where="request")

    schema = body.get("schemaVersion")
    if schema is not None and int(schema) != 2:
        raise FineTuneRequestError(
            f"unsupported schemaVersion {schema!r} — this server speaks "
            "fine-tune schema 2")

    base_model_id = body.get("baseModelID")
    if not isinstance(base_model_id, str) or not base_model_id.strip():
        raise FineTuneRequestError("baseModelID is required")

    raw_mode = body.get("trainingMode")
    if raw_mode not in _TRAINING_MODES:
        raise FineTuneRequestError(
            f"unknown trainingMode {raw_mode!r} — expected 'document' or "
            "'instructionChat'")
    training_mode = _TRAINING_MODES[raw_mode]

    evidence_grade = bool(body.get("evidenceGrade") or False)

    settings: dict[str, Any] = {
        "base_model_id": base_model_id,
        "training_mode": training_mode,
        "evidence_grade": evidence_grade,
    }
    if body.get("revision") is not None:
        settings["revision"] = _coerce(body["revision"], str, key="revision")
    name = body.get("name")
    if name is not None:
        name = _coerce(name, str, key="name")
        settings["output_name"] = name
    if body.get("selectionMetric") is not None:
        settings["selection_metric"] = _coerce(
            body["selectionMetric"], str, key="selectionMetric")

    control_arm = body.get("controlArm")
    if control_arm is not None:
        control_arm = _require_dict(control_arm, key="controlArm")
        _refuse_unknown(control_arm, _CONTROL_ARM_KEYS, where="controlArm")
        if not isinstance(control_arm.get("kind"), str):
            raise FineTuneRequestError("controlArm.kind is required")
        settings["control_arm"] = {
            k: v for k, v in control_arm.items() if v is not None}

    hyper = body.get("hyperparameters")
    if hyper is not None:
        hyper = _require_dict(hyper, key="hyperparameters")
        _refuse_unknown(hyper, set(_HYPERPARAMETERS), where="hyperparameters")
        for wire_key, (field, kind) in _HYPERPARAMETERS.items():
            if hyper.get(wire_key) is None:
                continue
            settings[field] = _coerce(hyper[wire_key], kind,
                                      key=f"hyperparameters.{wire_key}")
    # One knob, two conventions: whichever the request declared becomes the
    # trainer's ``alpha`` here, and the direct value (if that is what was
    # declared) rides along as provenance the sidecar and plan stamp.
    alpha, requested_adapter_scale = resolve_adapter_scale(
        alpha=settings.get("alpha"),
        adapter_scale=settings.pop("requested_adapter_scale", None),
        rank=int(settings.get("rank", lora_train.LoRAConfig.rank)))
    if alpha is not None:
        settings["alpha"] = alpha
    if requested_adapter_scale is not None:
        settings["requested_adapter_scale"] = requested_adapter_scale

    dataset = _require_dict(body.get("dataset"), key="dataset")
    _refuse_unknown(dataset, _DATASET_KEYS, where="dataset")
    for wire_key, field in (("bundleID", "dataset_bundle_id"),
                            ("manifestPath", "dataset_manifest_path"),
                            ("manifestHash", "dataset_manifest_hash")):
        if dataset.get(wire_key) is not None:
            settings[field] = _coerce(dataset[wire_key], str,
                                      key=f"dataset.{wire_key}")

    raw_files = dataset.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise FineTuneRequestError(
            "dataset.files must be a non-empty list of split files")
    files: list[tuple[str, str, str, str | None]] = []
    train_paths: list[str] = []
    validation_paths: list[str] = []
    expected: dict[str, str] = {}
    for index, entry in enumerate(raw_files):
        entry = _require_dict(entry, key=f"dataset.files[{index}]")
        _refuse_unknown(entry, _DATASET_FILE_KEYS,
                        where=f"dataset.files[{index}]")
        role = entry.get("role")
        if role not in ("train", "validation"):
            raise FineTuneRequestError(
                f"dataset.files[{index}].role must be 'train' or 'validation' "
                f"(got {role!r})")
        path = _relative_dataset_path(entry.get("path"))
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not _SHA256.match(digest.lower()):
            raise FineTuneRequestError(
                f"{path}: sha256 must be a 64-character hex digest of the raw "
                "file bytes")
        digest = digest.lower()
        content = entry.get("content")
        if content is not None and not isinstance(content, str):
            raise FineTuneRequestError(
                f"{path}: content must be the file's UTF-8 text, or null for a "
                "server-resident file")
        if path in expected:
            raise FineTuneRequestError(
                f"{path}: declared twice in dataset.files — a file cannot be "
                "in both splits, and listing it twice double-counts its rows")
        expected[path] = digest
        files.append((path, digest, role, content))
        (train_paths if role == "train" else validation_paths).append(path)

    settings["train_paths"] = train_paths
    settings["validation_paths"] = validation_paths
    settings["expected_hashes"] = expected

    expected_plan_hash = body.get("expectedPlanHash")
    if expected_plan_hash is not None:
        expected_plan_hash = _coerce(expected_plan_hash, str,
                                     key="expectedPlanHash")

    resources = body.get("resources") or {}
    if not isinstance(resources, dict):
        raise FineTuneRequestError("resources must be a JSON object")

    try:
        config = lora_train.config_from_dict(settings)
    except lora_train.LoRATrainError as exc:
        raise FineTuneRequestError(str(exc)) from exc

    return ParsedRequest(
        config=config, name=name, evidence_grade=evidence_grade,
        expected_plan_hash=expected_plan_hash, files=files,
        resources=resources, force=bool(body.get("force") or False),
        dry_run=bool(body.get("dryRun") or False))


# --- dataset staging ---------------------------------------------------------


def stage_dataset(parsed: ParsedRequest, directory: str, resolver) -> str:
    """Materialize every declared split file under ``directory`` at its
    DECLARED relative path, verifying SHA-256 in both directions:

    - ``content`` non-null: an inline upload — ``sha256(content utf-8)`` must
      equal the declared digest before a byte is written.
    - ``content`` null: a server-resident file — resolved through the path
      resolver (containment) and hash-verified against the declared digest.

    Returns the staging root (the config's ``dataset_root``). The declared
    path is what provenance names, so a plan computed from inline bytes and
    one computed from the resident file are the same plan.
    """
    os.makedirs(directory, exist_ok=True)
    for path, digest, _role, content in parsed.files:
        target = os.path.join(directory, path)
        os.makedirs(os.path.dirname(target) or directory, exist_ok=True)
        if content is not None:
            raw = content.encode("utf-8")
            actual = hashlib.sha256(raw).hexdigest()
            if actual != digest:
                raise FineTuneRequestError(
                    f"{path}: inline content hashes to {actual} but the "
                    f"request declares {digest} — the upload and its pin "
                    "disagree, so neither can be trusted")
            with open(target, "wb") as handle:
                handle.write(raw)
            continue
        resolved = resolver.require_file(path)
        actual = lora_data.sha256_file(resolved)
        if actual != digest:
            raise FineTuneRequestError(
                f"{path}: the server-resident file hashes to {actual} but the "
                f"request declares {digest} — the training data drifted after "
                "it was pinned")
        shutil.copyfile(resolved, target)
    return directory


# --- the plan ----------------------------------------------------------------


def canonical_plan_hash(plan: dict) -> str:
    """SHA-256 of the plan's canonical JSON (contract §6). The submission
    echoes this as ``expectedPlanHash``, so the plan a researcher confirmed
    and the plan that runs are provably the same one."""
    blob = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _plan_device(profile: ServerProfile) -> str:
    """The device the plan is written FOR. A Slurm-executor server plans for
    the GPU node it submits to, not for the controller it runs on: planning a
    27B bf16 job as ``float32`` because the login node has no CUDA would be a
    plan nobody could act on."""
    if profile.executor == "slurm":
        return "cuda"
    from ..steering.model_loader import resolve_device
    return resolve_device(None)


def resolve_plan_dtype(config: lora_train.LoRAConfig,
                       profile: ServerProfile) -> str:
    device = config.device or _plan_device(profile)
    try:
        return lora_train.resolve_training_dtype(
            config.dtype, device, config.base_model_id, warn=lambda _m: None)
    except Exception:  # noqa: BLE001 - the plan degrades to the declared dtype
        return config.dtype or "auto"


def resolve_revision(config: lora_train.LoRAConfig) -> str | None:
    """Full-SHA passthrough, else the locally cached ``refs/main`` commit,
    else None. Never invents a revision — a null here is exactly the floating
    pin the evidence gate refuses."""
    revision = (config.revision or "").strip().lower()
    if _FULL_SHA.match(revision):
        return revision
    try:
        from ..steering.model_loader import cached_revision
        return cached_revision(config.base_model_id)
    except Exception:  # noqa: BLE001
        return None


def build_plan(parsed: ParsedRequest, *, dataset_root: str,
               profile: ServerProfile) -> tuple[dict, str]:
    """The normalized training plan + its hash. No tokenizer, no model: row
    counts and per-file hashes come from :mod:`lora_data`'s loader, and the
    schedule is counted in ROWS (see ``planNotes``)."""
    config = parsed.config
    try:
        loaded = lora_data.load_split_rows(
            config.dataset_spec(), evidence_grade=config.evidence_grade,
            root=dataset_root)
    except (lora_data.LoRADataError, lora_train.LoRATrainError) as exc:
        raise FineTuneRequestError(str(exc)) from exc
    manifest = lora_data.dataset_manifest_dict(loaded)

    row_count = len(loaded.train_rows)
    try:
        schedule = lora_train.plan_schedule(config, row_count)
    except lora_train.LoRATrainError as exc:
        raise FineTuneRequestError(str(exc)) from exc

    files = [
        {"role": file.role, "path": file.path, "sha256": file.sha256,
         "rowsRoot": file.rows_root, "rows": len(file.rows)}
        for file in tuple(loaded.train_files) + tuple(loaded.validation_files)
    ]
    notes = [
        "schedule step counts are derived from ROWS, not tokenized examples: "
        "the plan endpoint loads no tokenizer by design. In document mode a "
        "row longer than maxSequenceTokens is split under longDocumentPolicy, "
        "so the executed run may take MORE steps than planned; instruction/"
        "chat mode is one example per row and the counts are exact.",
    ]
    if config.evidence_grade and not _FULL_SHA.match((config.revision or "").lower()):
        notes.append(
            "revision is not a pinned full commit sha — evidence-grade "
            "submission will refuse this request")

    plan = {
        "resolvedRevision": resolve_revision(config),
        "dtype": resolve_plan_dtype(config, profile),
        "trainingMode": config.training_mode,
        "evidenceGrade": config.evidence_grade,
        "selectionMetric": config.selection_metric,
        "controlArm": config.control_arm,
        "adapterScale": adapter_scale_block(config),
        "schedule": {
            "totalSteps": schedule.total_steps,
            "epochs": schedule.epochs,
            "effectiveBatchSize": schedule.effective_batch_size,
            "warmupSteps": schedule.warmup_steps,
            "lrSchedule": schedule.lr_schedule,
        },
        "dataset": {
            "bundleID": manifest["bundleID"],
            "manifestPath": manifest["manifestPath"],
            "manifestHash": manifest["manifestHash"],
            "files": files,
            "counts": manifest["counts"],
        },
        "planNotes": notes,
    }
    return plan, canonical_plan_hash(plan)


def plan_response(body: dict, *, resolver,
                  profile: ServerProfile | None = None) -> dict:
    """``POST /api/finetune/plan``. Inline content is hash-verified but never
    persisted: the staging directory is a temp tree that is removed before the
    response is returned."""
    profile = profile or ServerProfile.from_env()
    parsed = parse_request(body)
    with tempfile.TemporaryDirectory(prefix="steerlab-finetune-plan-") as tmp:
        stage_dataset(parsed, tmp, resolver)
        plan, plan_hash = build_plan(parsed, dataset_root=tmp, profile=profile)
    return {
        "plan": plan,
        "planHash": plan_hash,
        # Informational, deliberately OUTSIDE the hashed plan: the whole gap
        # at once, so a researcher fixes it in one round trip instead of one
        # refusal per submission attempt.
        "evidenceRefusals": lora_train.evidence_refusals(
            parsed.config, resolved_dtype=plan["dtype"]),
    }


# --- LoRA preflight ----------------------------------------------------------
#
# Same discipline as the study preflight (submissions.py): every builder
# DEGRADES to "warn" with an honest message when its inputs are unavailable
# and never raises. A "fail" blocks the submission unless the caller forces
# (recorded loudly on the job).

#: AdamW keeps two fp32 moments per trainable parameter.
LORA_OPTIMIZER_BYTES_PER_PARAM = 8
#: Retained-for-backward activation tensors per transformer block, per token.
#: Deliberately generous: the exact count is implementation-specific and the
#: conservative direction is the safe one for an OOM gate.
LORA_ACTIVATION_TENSORS_PER_LAYER = 8
#: Final multiplier on the whole estimate (fragmentation, workspace, cuBLAS).
LORA_MEMORY_HEADROOM = 1.20
#: Achieved (not peak) bf16 training throughput per GPU type, TFLOP/s. A
#: DECLARED conservative table — there is no observed training-throughput
#: history yet; when one appears (a ``stepsPerHour`` entry in the housekeeping
#: throughput table for this model+GPU) it wins over these constants.
LORA_TRAINING_TFLOPS = {
    "H100": 400.0, "A100": 150.0, "V100": 40.0, "L4": 40.0, "P100": 8.0,
}
#: FLOPs per parameter per token for a forward+backward pass. LoRA freezes the
#: base weights but still backpropagates THROUGH them, so the standard 6·N·T
#: training estimate is the right order (and errs high, which is what a
#: walltime gate wants).
LORA_FLOPS_PER_PARAM_TOKEN = 6.0


def _text_config(config: dict | None) -> dict:
    """Gemma 3 nests the text tower under ``text_config``; flatten it the same
    way the study memory-fit check does."""
    cfg = dict(config or {})
    inner = cfg.get("text_config")
    if isinstance(inner, dict):
        merged = dict(inner)
        for key in ("num_hidden_layers", "hidden_size", "torch_dtype"):
            if key not in merged and key in cfg:
                merged[key] = cfg[key]
        cfg = merged
    return cfg


_DTYPE_BYTES = {"float32": 4, "fp32": 4, "float": 4,
                "bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2, "half": 2}


def _dtype_bytes(name: str | None, default: int = 2) -> int:
    return _DTYPE_BYTES.get((name or "").lower().replace("torch.", ""), default)


def _check_lora_memory_fit(config: lora_train.LoRAConfig, *, dtype: str,
                           revision: str | None,
                           resources: SlurmResources) -> dict:
    check_id = "loraMemoryFit"
    info = housekeeping.model_snapshot_info(config.base_model_id, revision)
    if info is None or info["weightsBytes"] <= 0:
        return _check(check_id, "warn",
                      f"{config.base_model_id} has no weights in the HF cache "
                      f"({housekeeping.hf_cache_root()}) — install the model "
                      "first; LoRA memory fit cannot be checked")
    model_config = _text_config(info.get("config"))
    layers = model_config.get("num_hidden_layers")
    hidden = model_config.get("hidden_size")
    stored_bytes = _dtype_bytes(model_config.get("torch_dtype"))
    runtime_bytes = _dtype_bytes(dtype)
    params = info["weightsBytes"] / stored_bytes
    weights = params * runtime_bytes
    data: dict[str, Any] = {
        "modelId": config.base_model_id,
        "snapshotRevision": info.get("revision"),
        "weightsBytes": info["weightsBytes"],
        "baseParameters": int(params),
        "trainingDtype": dtype,
        "weightsRuntimeBytes": int(weights),
        "batchSize": config.batch_size,
        "maxSequenceTokens": config.max_sequence_tokens,
    }
    if not (isinstance(layers, int) and isinstance(hidden, int)
            and layers > 0 and hidden > 0):
        return _check(check_id, "warn",
                      f"snapshot {info['snapshotPath']} has no readable "
                      "config.json geometry (num_hidden_layers/hidden_size) — "
                      "adapter and activation memory cannot be estimated "
                      f"(base weights alone: {_gb(weights):.1f} GB)", data)
    # LoRA A+B on one d×d projection is r·(d_in + d_out) ≈ 2·r·hidden.
    adapter_params = (layers * max(1, len(config.target_modules))
                      * 2 * config.rank * hidden)
    # weight + gradient in the training dtype, two fp32 AdamW moments.
    adapter_state = adapter_params * (2 * runtime_bytes
                                      + LORA_OPTIMIZER_BYTES_PER_PARAM)
    activations = (config.batch_size * int(config.max_sequence_tokens or 512)
                   * hidden * layers * runtime_bytes
                   * LORA_ACTIVATION_TENSORS_PER_LAYER)
    estimate = int((weights + adapter_state + activations) * LORA_MEMORY_HEADROOM)
    # Two retained checkpoints (selected-so-far + latest) of adapter weights
    # plus their optimizer state: disk, not VRAM — reported, never gated.
    checkpoint_disk = int(2 * adapter_params
                          * (runtime_bytes + LORA_OPTIMIZER_BYTES_PER_PARAM))
    data.update({
        "layers": layers, "hiddenSize": hidden,
        "adapterParameters": adapter_params,
        "adapterStateBytes": int(adapter_state),
        "activationBytes": int(activations),
        "estimateBytes": estimate,
        "checkpointDiskBytes": checkpoint_disk,
    })
    gpu_type = housekeeping.parse_gpu_type(resources.gres)
    gpu_count = _gres_gpu_count(resources)
    data["gpuType"] = gpu_type
    data["gpuCount"] = gpu_count
    if gpu_type is None:
        return _check(check_id, "warn",
                      f"needs ≈{_gb(estimate):.1f} GB but no GPU type is "
                      "requested (gres unset) — cannot compare against VRAM",
                      data)
    vram_gb = (resources.gpu_vram_gb or {}).get(gpu_type)
    if vram_gb is None:
        return _check(check_id, "warn",
                      f"needs ≈{_gb(estimate):.1f} GB but there is no VRAM "
                      f"entry for {gpu_type} — set STEERLAB_SLURM_GPU_VRAM "
                      "(e.g. \"A100:80,H100:80,L4:24,P100:16\") or pass "
                      "resources.gpuVram", data)
    # THE TRAINER IS SINGLE-DEVICE. `lora_train` does `model.to(device)` —
    # there is no DDP/FSDP/`device_map` sharding and no bitsandbytes/QLoRA
    # path anywhere in it. Budgeting `vram_gb * gpu_count` (the arithmetic
    # here until 2026-08-13) therefore passed gpu:A100:2 for a model that
    # then loaded ENTIRELY onto cuda:0 and OOM'd — after the queue wait.
    # The honest comparison is against ONE GPU's VRAM, whatever the request
    # asks for; `vramPerGpuBytes` + `gpuCount` say so in the data block.
    budget = int(vram_gb * 1024**3)
    data["vramPerGpuBytes"] = budget
    data["vramBytes"] = budget
    breakdown = (f"weights {_gb(weights):.1f} + adapter/optimizer "
                 f"{_gb(adapter_state):.2f} + activations "
                 f"{_gb(activations):.1f} + 20% headroom")
    # Extra GPUs are not capacity — they are idle allocation. Say so on the
    # pass as well as the failure; readiness plan §2.7 forbids the silent
    # fall from a declared multi-GPU mode to one GPU, and the mode it would
    # fall FROM does not exist yet.
    idle = ""
    if gpu_count > 1:
        idle = (f" WARNING: gpu:{gpu_type}:{gpu_count} requests {gpu_count} "
                f"GPUs but the trainer is single-device (model.to(device), no "
                f"FSDP/DDP/device_map) — {gpu_count - 1} GPU(s) will sit idle "
                f"and the extra allocation is wasted; request gpu:{gpu_type}:1 "
                "unless a sharded training mode lands")
    if estimate > budget:
        return _check(check_id, "fail",
                      f"needs ≈{_gb(estimate):.1f} GB ({breakdown}) but one "
                      f"{gpu_type} has {vram_gb} GB and the trainer runs on a "
                      "single GPU — lower batchSize or maxSequenceTokens, or "
                      "request a larger GPU (readiness plan §2.7 refuses a "
                      "silent fallback; no multi-GPU or QLoRA training mode "
                      "is implemented)" + idle, data)
    return _check(check_id, "ok",
                  f"fits: ≈{_gb(estimate):.1f} GB of {vram_gb} GB on one "
                  f"{gpu_type} ({breakdown}); checkpoints "
                  f"≈{_gb(checkpoint_disk):.2f} GB of disk" + idle, data)


def _check_lora_walltime(config: lora_train.LoRAConfig, *, total_steps: int,
                         dtype: str, revision: str | None,
                         resources: SlurmResources,
                         profile: ServerProfile) -> dict:
    """Readiness plan §0 amendment 4: memory-fit alone saves the OOM; the
    walltime check saves the queue allocation."""
    check_id = "loraWalltime"
    gpu_type = housekeeping.parse_gpu_type(resources.gres)
    tokens_per_step = (max(1, config.batch_size)
                       * max(1, config.gradient_accumulation)
                       * int(config.max_sequence_tokens or 512))
    data: dict[str, Any] = {
        "totalSteps": total_steps,
        "tokensPerStep": tokens_per_step,
        "gpuType": gpu_type,
    }
    seconds_per_step: float | None = None
    basis = None
    observed = None
    try:
        observed = housekeeping.throughput_lookup(config.base_model_id,
                                                  gpu_type, profile)
    except Exception:  # noqa: BLE001 - the table is advisory
        observed = None
    steps_per_hour = float((observed or {}).get("stepsPerHour") or 0)
    if steps_per_hour > 0:
        seconds_per_step = 3600.0 / steps_per_hour
        basis = "observed"
        data["stepsPerHour"] = steps_per_hour
    else:
        tflops = LORA_TRAINING_TFLOPS.get(gpu_type or "")
        info = housekeeping.model_snapshot_info(config.base_model_id, revision)
        if info is None or info["weightsBytes"] <= 0:
            return _check(check_id, "warn",
                          f"{config.base_model_id} is not in the HF cache — "
                          "the step-time estimate needs the model's parameter "
                          "count; walltime cannot be estimated", data)
        stored_bytes = _dtype_bytes(
            _text_config(info.get("config")).get("torch_dtype"))
        params = info["weightsBytes"] / stored_bytes
        data["baseParameters"] = int(params)
        if not tflops:
            return _check(check_id, "warn",
                          f"no training-throughput constant for GPU type "
                          f"{gpu_type or '(unset gres)'} — walltime cannot be "
                          "estimated; request a known GPU type "
                          f"({', '.join(sorted(LORA_TRAINING_TFLOPS))}) or "
                          "raise the walltime deliberately", data)
        seconds_per_step = (LORA_FLOPS_PER_PARAM_TOKEN * params
                            * tokens_per_step) / (tflops * 1e12)
        basis = "declaredThroughput"
        data["achievedTFLOPs"] = tflops
    data["secondsPerStep"] = round(seconds_per_step, 3)
    data["basis"] = basis
    estimated_hours = (total_steps * seconds_per_step / 3600.0
                       * PREFLIGHT_WALLTIME_MARGIN)
    data["estimatedHours"] = round(estimated_hours, 2)
    try:
        requested_hours = _parse_walltime(resources.walltime).total_seconds() / 3600.0
    except (ValueError, TypeError):
        return _check(check_id, "warn",
                      f"requested walltime {resources.walltime!r} is not "
                      f"parseable — cannot compare against the "
                      f"≈{estimated_hours:.1f} h estimate", data)
    data["requestedHours"] = round(requested_hours, 2)
    if requested_hours <= 0:
        return _check(check_id, "warn",
                      f"requested walltime {resources.walltime!r} is zero — "
                      "cannot compare", data)
    if estimated_hours > requested_hours:
        return _check(check_id, "fail",
                      f"estimated ≈{estimated_hours:.1f} h ({total_steps} "
                      f"steps × {seconds_per_step:.2f} s/step × "
                      f"{PREFLIGHT_WALLTIME_MARGIN} margin) exceeds the "
                      f"requested walltime {resources.walltime} — raise the "
                      "walltime, cut epochs, or shorten the sequence", data)
    if estimated_hours > PREFLIGHT_WALLTIME_WARN_FRACTION * requested_hours:
        return _check(check_id, "warn",
                      f"estimated ≈{estimated_hours:.1f} h is "
                      f"{estimated_hours / requested_hours:.0%} of the "
                      f"requested walltime {resources.walltime} — tight; "
                      "a checkpointed requeue will finish it, but consider "
                      "more headroom", data)
    return _check(check_id, "ok",
                  f"estimated ≈{estimated_hours:.1f} h of "
                  f"{resources.walltime} requested ({basis})", data)


def lora_preflight(config: lora_train.LoRAConfig, *, total_steps: int,
                   dtype: str, revision: str | None,
                   resources: SlurmResources,
                   profile: ServerProfile) -> dict:
    checks: list[dict] = []
    builders: list[tuple[str, Callable[[], dict]]] = [
        ("gpuRequest", lambda: _check_gpu_request(resources)),
        ("loraMemoryFit", lambda: _check_lora_memory_fit(
            config, dtype=dtype, revision=revision, resources=resources)),
        ("loraWalltime", lambda: _check_lora_walltime(
            config, total_steps=total_steps, dtype=dtype, revision=revision,
            resources=resources, profile=profile)),
        ("maintenanceWindow", lambda: _check_maintenance(resources, profile)),
    ]
    for check_id, builder in builders:
        try:
            checks.append(builder())
        except Exception as exc:  # noqa: BLE001 - degrade, never crash a submit
            checks.append(_check(
                check_id, "warn",
                f"check could not run ({type(exc).__name__}: {exc})"))
    return {"checks": checks, "verdict": _verdict_of(checks)}


# --- Slurm submission --------------------------------------------------------


def _execute_command(job_directory: str, record_path: str) -> list[str]:
    import sys
    python = os.environ.get("STEERLAB_PYTHON") or sys.executable or "python"
    return [python, "-m", "steerlab_server.cli", "finetune", "execute",
            job_directory, "--record", record_path]


@_root_stays_put
def submit_finetune(body: dict, *, jobs, resolver, root: str | None = None,
                    profile: ServerProfile | None = None,
                    env: dict | None = None) -> dict:
    """``POST /api/finetune/submit`` — the evidence-grade path.

    Materializes a self-contained job directory (dataset bytes, resolved
    config, plan + planHash), runs the LoRA preflight, renders the sbatch
    bundle around ``steerlab-server finetune execute <dir>``, and records the
    external job with the same stamps ``study submit`` uses — so the
    reconciler's checkpoint mapping (exit 85 → ``checkpointed``) and
    auto-resubmit-on-checkpoint work here unchanged.
    """
    profile = profile or ServerProfile.from_env()
    parsed = parse_request(body, allow_submit_keys=True)
    config = parsed.config
    if profile.executor != "slurm" and not parsed.dry_run:
        raise FineTuneRequestError(
            "Slurm fine-tune submission requires STEERLAB_EXECUTOR=slurm — "
            f"this server's profile declares executor {profile.executor!r}. "
            "Exploratory training runs on the daemon route "
            "(/api/finetune/train); evidence-grade training needs a scheduler.")

    name = parsed.name or f"lora-{os.path.basename(config.base_model_id)}"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "adapter"

    # Validate against a THROWAWAY staging tree first: a refusal must not
    # leave a submission directory behind (the plan depends only on declared
    # paths and file hashes, so the tree it was staged in cannot change it).
    with tempfile.TemporaryDirectory(prefix="steerlab-finetune-check-") as tmp:
        stage_dataset(parsed, tmp, resolver)
        config.dataset_root = tmp
        plan, plan_hash = build_plan(parsed, dataset_root=tmp, profile=profile)
    refusals = lora_train.evidence_refusals(config, resolved_dtype=plan["dtype"])
    if refusals:
        raise FineTuneRequestError(
            "evidence-grade LoRA submission refused:\n  - "
            + "\n  - ".join(refusals))
    if config.evidence_grade:
        if not parsed.expected_plan_hash:
            raise FineTuneRequestError(
                "evidence-grade submission requires expectedPlanHash — fetch "
                "/api/finetune/plan, confirm the plan, and echo its planHash. "
                f"This request's plan hashes to {plan_hash}")
        if parsed.expected_plan_hash.lower() != plan_hash:
            raise FineTuneRequestError(
                "plan drift: the submission pins expectedPlanHash "
                f"{parsed.expected_plan_hash} but this request's plan hashes "
                f"to {plan_hash} — re-fetch /api/finetune/plan and confirm the "
                "plan that would actually run")

    submission_dir = paths.make_unique_run_directory(f"submit-finetune-{slug}",
                                                     root)
    records_dir = os.path.join(submission_dir, "records")
    os.makedirs(records_dir, exist_ok=True)
    dataset_root = stage_dataset(
        parsed, os.path.join(submission_dir, DATASET_DIRNAME), resolver)
    config.dataset_root = dataset_root

    resources = _resources_from_dict(parsed.resources, slug, "finetune")
    preflight = lora_preflight(
        config, total_steps=int(plan["schedule"]["totalSteps"]),
        dtype=plan["dtype"], revision=plan["resolvedRevision"],
        resources=resources, profile=profile)
    overridden = _gate_on_preflight(preflight, dry_run=parsed.dry_run,
                                    force=parsed.force)

    run_directory = os.path.join(submission_dir, RUN_DIRNAME)
    # The trainer's run directory is INSIDE the job so a requeue of the same
    # sbatch script adopts the same checkpoint tree.
    stored = _config_to_dict(config)
    stored["run_directory"] = run_directory
    with open(os.path.join(submission_dir, CONFIG_FILENAME), "w",
              encoding="utf-8") as handle:
        json.dump(stored, handle, indent=2, sort_keys=True)
    with open(os.path.join(submission_dir, PLAN_FILENAME), "w",
              encoding="utf-8") as handle:
        json.dump({"plan": plan, "planHash": plan_hash,
                   "preflight": preflight}, handle, indent=2, sort_keys=True)

    job_id = uuid.uuid4().hex[:12]
    record_path = os.path.join(records_dir, f"{job_id}.json")
    command = _execute_command(submission_dir, record_path)
    job_env = dict(env or {})
    job_env["STEERLAB_JOB_ID"] = job_id
    bundle = SlurmExecutor(profile).create_bundle(
        os.path.join(submission_dir, "slurm"), command, env=job_env,
        resources=resources,
        metadata={"kind": "fineTuneSubmission", "adapter": name,
                  "modelID": config.base_model_id, "planHash": plan_hash,
                  "recordsDirectory": records_dir})

    slurm_id: str | None = None
    status = "prepared" if parsed.dry_run else "submitted"
    log = f"prepared Slurm fine-tune submission {name}"
    if not parsed.dry_run:
        slurm_id = SlurmExecutor(profile).submit(bundle)
        log = f"submitted fine-tune {name} as Slurm job {slurm_id}"
    if overridden:
        log += " (PREFLIGHT OVERRIDDEN: verdict fail, forced by caller)"

    stamped = _stamped_resources(bundle, None, preflight, overridden,
                                 records_dir=records_dir)
    stamped["modelID"] = config.base_model_id
    result = {
        "adapter": name,
        "modelID": config.base_model_id,
        "submissionDirectory": submission_dir,
        "runDirectory": run_directory,
        "datasetRoot": dataset_root,
        "slurmBundle": bundle.to_dict(),
        "command": command,
        "recordsDirectory": records_dir,
        "plan": plan,
        "planHash": plan_hash,
        "preflight": preflight,
        **({"preflightOverridden": True} if overridden else {}),
    }
    job = jobs.record_external(
        JOB_KIND, status=status, executor="slurm", executor_job_id=slurm_id,
        requested_resources=stamped, job_id=job_id, result=result, log=log)
    return {"jobId": job.id, "plan": plan, "planHash": plan_hash,
            "preflight": preflight, "slurmJobID": slurm_id,
            "submissionDirectory": submission_dir, "dryRun": parsed.dry_run}


def _config_to_dict(config: lora_train.LoRAConfig) -> dict:
    from dataclasses import asdict
    return asdict(config)


# --- the execute side (CLI child) --------------------------------------------


def load_job_config(job_directory: str) -> tuple[lora_train.LoRAConfig, str]:
    """Read a submitted job directory back into (config, run directory)."""
    path = os.path.join(job_directory, CONFIG_FILENAME)
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        raise FineTuneRequestError(
            f"{path}: not a readable fine-tune job config ({exc})") from exc
    if not isinstance(payload, dict):
        raise FineTuneRequestError(f"{path}: job config must be a JSON object")
    run_directory = payload.pop("run_directory", None) \
        or os.path.join(job_directory, RUN_DIRNAME)
    try:
        return lora_train.config_from_dict(payload), run_directory
    except lora_train.LoRATrainError as exc:
        raise FineTuneRequestError(f"{path}: {exc}") from exc


def is_finished(run_directory: str, config: lora_train.LoRAConfig) -> bool:
    """At-most-once finalization: a run whose sidecar exists is complete, and
    a requeue that lands on it must be idempotent, not mint a second run."""
    name = config.output_name or \
        f"lora-{os.path.basename(config.base_model_id)}"
    return os.path.isfile(os.path.join(run_directory, f"{name}.json"))


__all__ = [
    "FineTuneRequestError", "JOB_KIND", "ParsedRequest", "PreflightRejection",
    "adapter_scale_block", "build_plan", "canonical_plan_hash", "is_finished",
    "load_job_config", "lora_preflight", "parse_request", "plan_response",
    "resolve_adapter_scale", "resolve_plan_dtype", "resolve_revision",
    "stage_dataset", "submit_finetune",
]
