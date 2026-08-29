"""Load a hookable HF causal LM (parallel to Swift ``SteeredContainerLoader`` +
``SteeredModels`` + the vendored model files).

Where the Swift side had to *vendor* Qwen3/Gemma3 to insert hook points, here a
stock ``AutoModelForCausalLM`` plus :class:`HookedModel` gives the same residual
-stream interventions on every block — so there is nothing model-specific to
vendor. Weights download to the standard HF cache (``HF_HOME`` /
``~/.cache/huggingface``), never into the project tree.

The cluster is a **fresh extraction substrate**: bf16/fp16 activations on CUDA
will not byte-match MLX-8bit-on-Metal, so vectors extracted here are a new,
separately-validated model variant — not a transplant of the Mac results.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import __version__ as _TRANSFORMERS_VERSION

from .hooks import HookedModel


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

#: The canonical spelling each accepted alias resolves to. This IS the closed
#: dtype vocabulary — Swift's `SteeringDtype` carries the same set, and a
#: manifest pin is checked against it on both engines.
_DTYPE_CANONICAL = {
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float16": "float16", "fp16": "float16",
    "float32": "float32", "fp32": "float32",
}

#: Canonical names only, for error messages and UI pickers.
DTYPE_VOCABULARY = ("bfloat16", "float16", "float32")


def normalize_dtype(value: str | None) -> str | None:
    """The canonical spelling of a dtype alias, or None if unrecognized.

    ``"auto"`` and empty resolve to None — they mean "let the device decide",
    which is resolution rather than a pin.
    """
    name = (value or "").strip().lower()
    if not name or name == "auto":
        return None
    return _DTYPE_CANONICAL.get(name)


def _config_int(config: object, *names: str) -> int:
    """Read an integer model-dimension field from a HF config.

    Newer multimodal checkpoints such as Gemma 3 keep text-transformer
    dimensions under ``config.text_config`` while older decoder-only configs
    expose them at the top level. Treat both layouts as the same text model for
    SteerLab's residual-stream work.
    """
    candidates = [config]
    for nested in ("text_config", "llm_config", "language_config", "decoder_config"):
        child = getattr(config, nested, None)
        if child is not None:
            candidates.append(child)
    for candidate in candidates:
        for name in names:
            value = getattr(candidate, name, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
    return 0


def resolve_device(device: str | None = None) -> str:
    """Pick a compute device: explicit arg, else ``STEERLAB_DEVICE`` env, else
    auto-detect CUDA → Apple MPS (Metal) → CPU.

    On a Mac the "graphics card" is the Apple-Silicon GPU via PyTorch's **MPS**
    backend (``mps``). Note this is the development/local path — the cluster
    target is CUDA — and that 8-bit quantization (bitsandbytes) is CUDA-only, so
    on MPS you load a **full-precision HF repo** (e.g. ``Qwen/Qwen3-4B``), not an
    MLX-quantized repo. For any op MPS hasn't implemented, set
    ``PYTORCH_ENABLE_MPS_FALLBACK=1`` to fall back to CPU per-op.
    """
    if device and device != "auto":
        return device
    env = os.environ.get("STEERLAB_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def available_devices() -> list[str]:
    """Devices this torch build can actually use, best first. On an
    Apple-Silicon Mac this is ``["mps", "cpu"]`` — CUDA is absent, so the UI
    shouldn't offer it (the cause of the 'Torch not compiled with CUDA' error)."""
    devices: list[str] = []
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        devices.extend(f"cuda:{i}" for i in range(count))
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def _is_gemma3_repo(model_id: str | None) -> bool:
    if not model_id:
        return False
    lid = model_id.lower()
    return "gemma-3" in lid or "gemma3" in lid


def hf_dtype_kwargs(dtype: torch.dtype) -> dict:
    """Keyword spelling for HF model dtype across Transformers versions.

    Transformers 5 warns that ``torch_dtype`` is deprecated in favor of
    ``dtype``. Older cluster modules may still expect the old spelling, so keep
    the choice centralized and version-gated.
    """
    try:
        major = int(_TRANSFORMERS_VERSION.split(".", 1)[0])
    except (TypeError, ValueError):
        major = 0
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


_MPS_BF16_PROBE: bool | None = None


def _mps_supports_bfloat16() -> bool:
    """One-time live probe: recent torch runs bf16 on Metal; older builds do
    not. A tiny matmul is the truthful check (version gating lies across
    macOS/torch combinations)."""
    global _MPS_BF16_PROBE
    if _MPS_BF16_PROBE is None:
        try:
            a = torch.ones(2, 2, dtype=torch.bfloat16, device="mps")
            _MPS_BF16_PROBE = bool(torch.isfinite((a @ a).sum()).item())
        except Exception:  # noqa: BLE001 - any failure means "no bf16 here"
            _MPS_BF16_PROBE = False
    return _MPS_BF16_PROBE


def default_dtype(device: str, model_id: str | None = None) -> str:
    """Sensible activation dtype per device.

    CUDA gets bf16; CPU gets fp32. Apple MPS usually takes fp16, but Gemma 3
    produces non-finite activations under fp16 (its residual norms overflow
    fp16's range), so it gets bfloat16 — fp32's exponent range at roughly
    fp16 speed — falling back to float32 only on torch builds without MPS
    bf16 (fp32 is correct for extraction but far too slow for interactive
    chat on a 4B).
    """
    if device.startswith("cuda"):
        return "bfloat16"
    if device == "mps":
        if _is_gemma3_repo(model_id):
            return "bfloat16" if _mps_supports_bfloat16() else "float32"
        return "float16"
    return "float32"


def attention_implementation(device: str) -> str | None:
    """Per-device attention kernel: ``"eager"`` on MPS, ``None`` — meaning
    keep HF's default — everywhere else.

    Measured 2026-07-28 (``Server/scripts/memory_growth_repro.py``, numbers
    in its docstring): torch's MPS sdpa leaves a per-prompt-shape workspace
    in MPSGraph's cache that ``empty_cache()`` cannot reclaim, so any run
    whose prompt length varies — every panel, every study — ratchets the
    driver reservation until the watermark kills it (observed: 74 GB and
    death at a 10K-token context that eager completes flat at 15 GB, ~10%
    faster). Eager's transient lives in torch's own allocator and the
    between-turn trim returns all of it.

    CUDA is deliberately untouched: ``None`` leaves sdpa, which dispatches
    real flash/mem-efficient kernels there — tiled, linear-memory, and the
    thing eager would pessimize. The science arm never sees this branch.
    """
    return "eager" if (device or "").startswith("mps") else None


def training_dtype(device: str, model_id: str | None = None) -> str:
    """Training dtype policy — **never float16**.

    :func:`default_dtype` may answer fp16 (non-Gemma models on MPS), which is
    acceptable for inference but not for training: fp16 AdamW training is the
    classic NaN-adapter factory (fp16's 5-bit exponent underflows gradients and
    overflows the optimizer's second moment). Training gets its own policy:
    bfloat16 on CUDA; bfloat16 on MPS when the torch build supports it, else
    float32; float32 on CPU.
    """
    if device.startswith("cuda"):
        return "bfloat16"
    if device == "mps":
        return "bfloat16" if _mps_supports_bfloat16() else "float32"
    return "float32"


class ModelLoadError(Exception):
    """A user-actionable load failure (bad repo, MLX format, OOM hint).

    ``advice_complete`` marks a refusal whose PROSE already names the right
    remedy — this engine's own typed sentences (device capacity,
    co-residency headroom, VRAM class). A wrapper adding caller context
    (e.g. which judge asked for the load) must CARRY that prose rather than
    replace it: the static "install the model" replacement sent an agent to
    install a model that was already there when the true cause was another
    model resident (observed 2026-08-28). False for raw hub/network dumps,
    which wrappers may still summarize away — "check your internet
    connection" is misleading on an air-gapped compute node."""

    def __init__(self, message: str, *, advice_complete: bool = False):
        super().__init__(message)
        self.advice_complete = advice_complete


class LoadCancelled(ModelLoadError):
    """An in-flight load interrupted by an explicit cancel request
    (POST /api/models/load/cancel — field incident 2026-08-29: a silent
    ~55 GB hub download held the only resident-model slot with no way to
    free it short of SIGTERMing the engine). A subclass of
    :class:`ModelLoadError` so every existing surface (400 conversion,
    SSE in-band error events, registry cleanup) treats it as a failed
    load; typed so callers that ASKED for the cancel can tell it from a
    genuine failure. The prose is the whole remedy: nothing is broken.
    """

    def __init__(self, message: str):
        super().__init__(message, advice_complete=True)


def _is_mlx_repo(model_id: str) -> bool:
    lid = model_id.lower()
    return "mlx" in lid or "-mlx-" in lid


@dataclass
class SteeredModel:
    model: torch.nn.Module
    tokenizer: object
    hooked: HookedModel
    model_id: str
    revision: str | None
    # Actual parameter dtype the model runs in (e.g. "bfloat16"), stamped at
    # load — the answer to "why is this model slow?" (fp32 fallback) in
    # /api/state. None only for legacy/fake wrappers built without load().
    dtype: str | None = None
    # Effective attention kernel ("eager"/"sdpa"…), read back from the loaded
    # config rather than echoing the request. Provenance for the per-device
    # choice below — kernels differ at float level, so a run's numbers should
    # be attributable to the kernel that actually produced them.
    attn_implementation: str | None = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def num_layers(self) -> int:
        return self.hooked.num_layers

    @property
    def hidden_size(self) -> int:
        hidden = _config_int(self.model.config, "hidden_size", "d_model", "n_embd")
        if hidden <= 0:
            raise AttributeError("could not determine text hidden size from model config")
        return hidden

    @property
    def context_window(self) -> int:
        """Configured max context (``max_position_embeddings``), to reject
        impossible prompt+generation budgets before the runtime asserts."""
        return _config_int(
            self.model.config,
            "max_position_embeddings",
            "n_positions",
            "seq_length",
            "max_sequence_length",
        )

    @torch.no_grad()
    def logits_for_residual_vector(self, vector: list[float]) -> list[float]:
        """Read a residual-stream direction through the output head (logit-lens
        validation of concept vectors; parallel to ``LogitLensReadable``).

        Applies the model's OWN final-norm module, then the unembedding —
        never the head alone: the model always normalizes before the head, so
        an un-normed readout would rank tokens the model never computes (the
        Swift twin applies the same two steps).
        """
        device = self.device
        h = torch.tensor(vector, device=device, dtype=torch.float32)
        norm = _final_norm(self.model)
        if norm is None:
            raise RuntimeError(
                "could not locate the model's final norm for the logit lens — "
                "an un-normed readout would not be what the model computes")
        head = _lm_head(self.model)
        param_dtype = next(head.parameters()).dtype if any(True for _ in head.parameters()) \
            else torch.float32
        h = h.to(param_dtype)
        h = norm(h)
        logits = head(h)
        return logits.to(torch.float32).cpu().tolist()


def _assert_cuda_kernels_for_device(dev: str) -> None:
    """Refuse a GPU the installed torch has NO kernels for (e.g. cu128 wheels
    ship sm_75+; Pascal P100 is sm_60) — the alternative is minutes of shard
    loading followed by a wedge or 'no kernel image' at first op. Best-effort:
    any introspection failure lets the load proceed (torch's own error is the
    fallback)."""
    try:
        index = int(dev.split(":")[1]) if ":" in dev else 0
        major, minor = torch.cuda.get_device_capability(index)
        arch_list = torch.cuda.get_arch_list()
        supported = {
            arch.removeprefix("sm_") for arch in arch_list if arch.startswith("sm_")
        }
        if not supported:
            return
        device_sm = f"{major}{minor}"
        # A binary built for sm_XY runs on hardware >= XY within the same
        # major generation; torch's arch list names the floors. Simplest
        # sound check: the device must be >= the lowest shipped arch.
        floor = min(int(s) for s in supported)
        if int(device_sm) < floor:
            name = torch.cuda.get_device_name(index)
            raise ModelLoadError(
                f"this node's GPU ({name}, sm_{device_sm}) is not supported by "
                f"the installed PyTorch build (kernels for sm_{floor}+ only) — "
                "loading would hang after minutes of shard reads. Start the "
                "session on a newer GPU type (this site's vocabulary is in "
                "STEERLAB_SLURM_GPU_TYPES), "
                "and drop unsupported types from STEERLAB_SLURM_GPU_TYPES so "
                "they are never offered.")
    except ModelLoadError:
        raise
    except Exception:  # noqa: BLE001 - introspection only, never blocks a load
        return


# The load's current PHASE, readable across threads (the SSE heartbeat
# thread reports it to the client): "elapsed seconds" alone told the live
# researcher nothing about WHICH phase was eating the minutes (2026-07-17).
# One loader runs at a time (the registry's load lock), so a single slot
# suffices.
_PHASE_LOCK = threading.Lock()
_CURRENT_PHASE: str | None = None


def _set_phase(text: str | None) -> None:
    global _CURRENT_PHASE
    with _PHASE_LOCK:
        _CURRENT_PHASE = text


def current_load_phase() -> str | None:
    with _PHASE_LOCK:
        return _CURRENT_PHASE


# The in-flight load's cancel probe, installed by the registry around each
# ``load`` call (one loader at a time — the registry's load lock — so a
# single slot suffices, exactly like the phase slot above). A module-level
# seam rather than a ``load`` parameter so the many existing callers and
# test fakes keep their signatures.
_CANCEL_CHECK_LOCK = threading.Lock()
_CANCEL_CHECK: "Callable[[], bool] | None" = None


def set_cancel_check(check: "Callable[[], bool] | None") -> None:
    """Install (or clear, with None) the probe :func:`load` polls to learn
    that its slot's cancel was requested. The registry owns this: it points
    the probe at the loading slot's cancel event for the duration of the
    load and always clears it after."""
    global _CANCEL_CHECK
    with _CANCEL_CHECK_LOCK:
        _CANCEL_CHECK = check


def load_cancel_requested() -> bool:
    with _CANCEL_CHECK_LOCK:
        check = _CANCEL_CHECK
    try:
        return bool(check()) if check is not None else False
    except Exception:  # pragma: no cover - a probe must never break a load
        return False


def _log_line(text: str) -> None:
    print(f"model_loader: {text}", file=sys.stderr, flush=True)


def _stage_root() -> str | None:
    """Node-local staging directory, or None when staging is off. The value
    may carry env placeholders expanded HERE on the compute node (e.g.
    ``/lscratch/$SLURM_JOB_ID`` — single-quoted through the bootstrap env
    file and the sbatch render, so neither the controller nor the script
    expands it with the wrong job id)."""
    raw = os.environ.get("STEERLAB_NODE_STAGE_DIR", "").strip()
    if not raw:
        return None
    return os.path.expandvars(raw)


# A revision/ref/commit may be used as a SINGLE path component under both
# cache roots — including the staging DESTINATION, whose stale-cleanup
# rmtree's it (engineer review 2026-07-18: an absolute or traversal revision
# could escape containment and delete outside the cache). HF commits are
# 40-hex and ref names are branch-like; anything else declines staging (the
# shared-cache fallback then decides whether the load itself fails).
_SAFE_CACHE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _safe_cache_component(value: str) -> bool:
    return bool(_SAFE_CACHE_COMPONENT_RE.match(value)) and ".." not in value


def _resolve_staging_commit(src_repo: str,
                            revision: str | None) -> tuple[str, str | None] | None:
    """``(commit, ref_name)`` for the snapshot a load of ``revision`` would
    use from the shared cache, or None when it cannot be determined. An
    explicit revision naming an existing snapshot wins; otherwise the ref
    file (``refs/main`` for revision-less loads) resolves it — in which case
    ``ref_name`` is returned so the staged copy can carry the same ref.

    Containment: the caller's raw revision AND the commit read from a ref
    file are validated as safe single path components before any filesystem
    join — never trust either as a path fragment."""
    if revision is not None and not _safe_cache_component(revision):
        return None
    if revision and os.path.isdir(os.path.join(src_repo, "snapshots", revision)):
        return revision, None
    ref_name = revision or "main"
    ref_path = os.path.join(src_repo, "refs", ref_name)
    try:
        with open(ref_path, encoding="utf-8") as handle:
            commit = handle.read().strip()
    except OSError:
        return None
    if not commit or not _safe_cache_component(commit):
        return None
    if os.path.isdir(os.path.join(src_repo, "snapshots", commit)):
        return commit, ref_name
    return None


def snapshot_size_bytes(model_id: str, revision: str | None = None) -> int | None:
    """Total bytes of the resolved cached snapshot (symlinks followed), or
    None when the model/revision is not in the shared cache. Weights
    dominate, so this is the load's device-memory floor at matching dtype
    — used by the GPU-capacity preflight and the client-facing size list."""
    repo = "models--" + model_id.replace("/", "--")
    src = os.path.join(hf_hub_dir(), repo)
    if not os.path.isdir(src):
        return None
    resolved = _resolve_staging_commit(src, revision)
    if resolved is None:
        return None
    commit, _ref = resolved
    total = 0
    for dirpath, _dirs, files in os.walk(os.path.join(src, "snapshots", commit)):
        for name in files:
            try:
                total += os.stat(
                    os.path.realpath(os.path.join(dirpath, name))).st_size
            except OSError:
                continue
    return total or None


#: Written by the install child (model_install.CHILD_SOURCE) after
#: ``snapshot_download`` returns — "this repo is fully downloaded".
INSTALL_COMPLETE_MARKER = ".steerlab-install-complete"


def hub_offline() -> bool:
    """Is this process pinned offline (cluster hermeticity)? Matches the
    rendered env file's spelling: ``HF_HUB_OFFLINE=1`` or omitted."""
    return (os.environ.get("HF_HUB_OFFLINE") or "").strip() == "1"


def needs_hub_download(model_id: str, revision: str | None = None) -> bool:
    """Would loading this model reach for the network first?

    The honesty seam behind the load stream's download warning and the
    load path's cancellable-download routing (field incident 2026-08-29:
    ``from_pretrained`` began a silent ~55 GB download inside the load,
    uncancellable and unannounced). Three local signals, no network:

    - no resolvable cached snapshot → a download is coming;
    - the install-complete marker → the repo is fully here, no download;
    - ``blobs/*.incomplete`` → an interrupted download would RESUME
      inside the load, so it counts as a download.

    A legacy cache (snapshot present, no marker — downloaded by
    ``from_pretrained`` before the marker existed) answers False: almost
    always right, and the price of a rare wrong False is only the old
    inline behavior. Offline processes always answer False — nothing
    downloads there, the load itself refuses if bytes are missing.
    """
    if hub_offline():
        return False
    if snapshot_size_bytes(model_id, revision) is None:
        return True
    repo_dir = os.path.join(hf_hub_dir(),
                            "models--" + model_id.replace("/", "--"))
    if os.path.exists(os.path.join(repo_dir, INSTALL_COMPLETE_MARKER)):
        return False
    try:
        return any(name.endswith(".incomplete")
                   for name in os.listdir(os.path.join(repo_dir, "blobs")))
    except OSError:
        return False


_SIZES_CACHE: tuple[tuple, str, dict[str, int], float] | None = None
_SIZES_CACHE_LOCK = threading.Lock()
# TTL backstop (engineer review 2026-07-18, second pass): the completion
# marker below only exists for installs OUR verb performed — a snapshot
# dropped in by hand (huggingface-cli) mid-download can still fingerprint
# identically while partial, so staleness must self-correct in bounded time.
_SIZES_CACHE_TTL_SECONDS = 60.0


def invalidate_model_size_cache() -> None:
    """Drop the size cache NOW (same-process fast path after an install —
    the fingerprint key below already catches cross-process changes)."""
    global _SIZES_CACHE
    with _SIZES_CACHE_LOCK:
        _SIZES_CACHE = None


def _hub_fingerprint(hub: str) -> tuple:
    """Cheap change-detector for the shared HF cache: repo names, their
    ``refs/main`` mtimes, and their install-completion markers. A handful
    of stats per call — nothing next to walking every snapshot — and
    CROSS-PROCESS correct (engineer review 2026-07-18: a TTL cache
    invalidated on the CONTROLLER stayed stale on the session WORKER, the
    process that actually answers a proxied /api/state).

    The completion marker matters (second-pass review, reproduced): the hub
    writes ``refs/main`` BEFORE the snapshot files download, so refs alone
    can fingerprint a PARTIAL snapshot identically to the finished one — a
    poll in that window would cache the partial size. The install child
    writes ``.steerlab-install-complete`` atomically after
    ``snapshot_download`` returns, changing the fingerprint exactly when
    the bytes are all present."""
    entries = []
    try:
        names = sorted(os.listdir(hub))
    except OSError:
        return ()
    for name in names:
        if not name.startswith("models--"):
            continue
        repo = os.path.join(hub, name)
        try:
            mtime = os.stat(os.path.join(repo, "refs", "main")).st_mtime_ns
        except OSError:
            try:
                mtime = os.stat(repo).st_mtime_ns
            except OSError:
                continue
        try:
            marker = os.stat(
                os.path.join(repo, ".steerlab-install-complete")).st_mtime_ns
        except OSError:
            marker = None
        entries.append((name, mtime, marker))
    return tuple(entries)


def local_model_sizes(cache_root: str | None = None) -> dict[str, int]:
    """``{model_id: snapshot bytes}`` for every loadable cached model — the
    client uses it to gray out models that cannot fit the session's GPU.
    Cached by HUB FINGERPRINT (repo names + refs + completion markers) with
    a TTL backstop: /api/state is polled and re-walking every snapshot per
    request is wasteful on shared storage; any process sees our installs
    the moment their completion marker lands, and foreign mid-download
    snapshots self-correct within the TTL.

    Deliberately does NOT require a root ``config.json`` the way
    :func:`local_model_ids` does: this is byte accounting, and a repo mid-
    download has blobs before its config lands. Filtering on shape here would
    report nothing for an install in progress — exactly when the size matters
    most."""
    global _SIZES_CACHE
    hub = hf_hub_dir(cache_root)
    fingerprint = _hub_fingerprint(hub)
    with _SIZES_CACHE_LOCK:
        if (_SIZES_CACHE is not None and _SIZES_CACHE[1] == hub
                and _SIZES_CACHE[0] == fingerprint
                and time.monotonic() - _SIZES_CACHE[3] < _SIZES_CACHE_TTL_SECONDS):
            return dict(_SIZES_CACHE[2])
    sizes: dict[str, int] = {}
    for model_id in _cached_repo_ids(cache_root, require_model_config=False):
        size = snapshot_size_bytes(model_id)
        if size:
            sizes[model_id] = size
    with _SIZES_CACHE_LOCK:
        _SIZES_CACHE = (fingerprint, hub, dict(sizes), time.monotonic())
    return sizes


# Weight bytes alone are a FLOOR: the CUDA context, KV cache, and activation
# workspace all come on top. A load whose weights leave less than this much
# free is refused before any staging or reading happens.
_GPU_LOAD_HEADROOM_BYTES = 1 << 30

# GPU-less loads above this checkpoint size REFUSE instead of warn (live
# cluster failure 2026-07-21: a 16 GB CPU-only controller job printed the
# CPU-fallback warning, staged 8 GiB of gemma-3-4b, loaded weights to ~58%
# in float32, and died — the warning was truthful and useless). Small models
# stay zero-friction: CPU smoke tests are legitimate. Deliberate large CPU
# runs set STEERLAB_ALLOW_CPU_LOAD=1 to get the old warn-and-proceed.
_CPU_LOAD_REFUSAL_BYTES = 2 << 30


def _allow_cpu_load() -> bool:
    return os.environ.get("STEERLAB_ALLOW_CPU_LOAD", "").strip().lower() in {
        "1", "true", "yes"}


def _assert_cpu_capacity(model_id: str, revision: str | None,
                         dtype: str) -> None:
    """Refuse a GPU-less load of a large model BEFORE anything stages or
    downloads (2026-07-21 incident, part 2). Uses the same cached-snapshot
    size estimate as the GPU capacity gate; an unknown size (model not in
    the cache yet) keeps the historical warn-and-proceed — the estimate,
    not a guess, is what earns a refusal."""
    if _allow_cpu_load():
        return
    size = snapshot_size_bytes(model_id, revision)
    if size is None or size <= _CPU_LOAD_REFUSAL_BYTES:
        return
    raise ModelLoadError(
        f"refusing to load '{model_id}' on CPU: no GPU is visible to torch "
        f"(CUDA/MPS unavailable) and the checkpoint is "
        f"~{size / (1 << 30):.1f} GiB — in {dtype} on a CPU host this load "
        "will crawl and likely exhaust the job's memory before finishing. "
        "On a Slurm site this usually means the job requested no GPU "
        "(empty gres/gpus): resubmit with the site's GPU resources "
        "(Remote options → GPU gres). For a deliberate CPU run of a large "
        "model, set STEERLAB_ALLOW_CPU_LOAD=1.")


def _assert_gpu_capacity(dev: str, model_id: str, revision: str | None,
                         resolved_dtype=None) -> None:
    """Refuse a CUDA load whose weights cannot fit the device BEFORE staging
    or reading anything (live 2026-07-18: a 22.7 GiB model staged for 15
    minutes onto a 22.05 GiB L4, then OOM'd in the device copy — and its
    debris OOM'd the next model's first generation). Weights-fit is
    NECESSARY, not sufficient — generation still needs KV/activation room —
    so the margin is deliberately thin: this gate catches impossibility,
    not tightness. Unknown sizes (model not cached yet) skip the check.

    dtype-aware (engineer review 2026-07-18): a float32 request needs ~2×
    the checkpoint's bytes (modern checkpoints ship bf16, 2 bytes/param —
    the assumed floor), so the estimate scales by the requested element
    size over 2, never below 1×."""
    try:
        index = int(dev.split(":")[1]) if ":" in dev else 0
        total = torch.cuda.get_device_properties(index).total_memory
        name = torch.cuda.get_device_name(index)
    except Exception:  # noqa: BLE001 - a failed probe never blocks a load
        return
    # FREE memory, not just total (external review round 5, finding 3): a
    # second model loading BESIDE a resident one — a foreign judge next to
    # the study model a sweep holds — was measured against a device that
    # looked empty. Nothing anywhere checked whether two models fit
    # TOGETHER, which is the configuration the judge-lifetime fix made
    # routine. Probed SEPARATELY so its failure degrades to the total-only
    # check rather than disabling the guard entirely.
    free: int | None = None
    try:
        with torch.cuda.device(index):
            free, _ = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - estimate only
        free = None
    size = snapshot_size_bytes(model_id, revision)
    if size is None:
        return
    factor = 1.0
    if resolved_dtype is not None:
        try:
            factor = max(
                1.0, torch.tensor([], dtype=resolved_dtype).element_size() / 2)
        except Exception:  # noqa: BLE001 - estimate only
            factor = 1.0
    needed = int(size * factor)
    if needed + _GPU_LOAD_HEADROOM_BYTES > total:
        at_dtype = (f" at {resolved_dtype}".replace("torch.", "")
                    if factor > 1.0 else "")
        raise ModelLoadError(
            f"'{model_id}' needs ~{needed / (1 << 30):.1f} GiB of "
            f"weights{at_dtype} but {dev} ({name}) has "
            f"{total / (1 << 30):.1f} GiB total — it cannot fit even before "
            "KV cache and activations. Start the session on a larger GPU "
            "type — see STEERLAB_SLURM_GPU_VRAM for what this site offers — "
            "or load a smaller model.")
    # Fits the device, but not what is LEFT of it. Distinguished from the
    # absolute refusal above because the remedy is different: unload
    # something, or judge with the study model — not "get a bigger GPU".
    #
    # Who can still reach this after the judge-column release seam landed
    # (2026-08-28): judged run/evaluate paths release every finished judge's
    # container before the next judge loads, so a SEQUENTIAL panel no longer
    # arrives here at all. What remains is genuine co-residency — a
    # judgeScore SWEEP panel, whose foreign judge models are held for the
    # whole grid because judging is interleaved with the selection — and
    # ad-hoc loads: /api/load, chat, variant generate, reader fit, and any
    # second process sharing the device. The remedy therefore still names
    # manual unloading, but says WHERE it applies rather than implying the
    # judged paths need a human.
    if free is not None and needed + _GPU_LOAD_HEADROOM_BYTES > free:
        at_dtype = (f" at {resolved_dtype}".replace("torch.", "")
                    if factor > 1.0 else "")
        raise ModelLoadError(
            f"'{model_id}' needs ~{needed / (1 << 30):.1f} GiB of "
            f"weights{at_dtype} but only {free / (1 << 30):.1f} GiB of "
            f"{dev} ({name})'s {total / (1 << 30):.1f} GiB is free — "
            "another model is already resident. Two models fit only if the "
            "device has room for both. A judged run or evaluate releases "
            "each finished judge's model before the next one loads, so this "
            "refusal means models genuinely needed AT ONCE: a judgeScore "
            "sweep panel (every judge's model is held for the whole grid), "
            "an interactive/API load beside a resident model, or another "
            "process on this device. Unload the other model, use the study "
            "model as judge, or pin an external judge.",
            advice_complete=True)


def free_device_memory(device: str | None = None) -> None:
    """Reclaim what a just-dropped model left behind on the accelerator.

    The ONE implementation of "the container is gone, now make the device
    believe it" (2026-08-28). Dropping the last Python reference to a
    ``SteeredModel`` returns its blocks to torch's caching allocator, not to
    the driver — the next ``cuda.mem_get_info`` still reports them as used,
    which is precisely what the capacity gate reads. A cycle collection
    (module trees are cyclic) followed by an allocator trim is what actually
    moves the free-memory number.

    Callers: ``ModelRegistry._evict`` (the registry's eviction path) and the
    judge-column release seam on the CLI/bundle path, which holds a private
    in-process copy with no registry to evict it. ``device`` is advisory —
    the CUDA trim is device-global anyway, and an MPS trim is skipped for a
    CUDA caller.
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if (device is None or device == "mps") and _mps_trim_is_safe():
        try:
            torch.mps.empty_cache()
        except Exception:  # pragma: no cover - best-effort cache trim
            pass


def _mps_trim_is_safe() -> bool:
    """True only when ``torch.mps.empty_cache()`` may actually be CALLED.

    The module existing is NOT the same question as the backend being usable,
    and the difference is not catchable: on a torch build whose ``torch.mps``
    is present while ``torch.backends.mps.is_available()`` is False, the trim
    reaches a backend that was never initialized and takes the process down
    with SIGSEGV (exit 139) — no ``except`` runs, because there is no Python
    exception. So availability is asked FIRST, and the answer gates the call.
    ``torch.backends.mps`` itself is hasattr-guarded for a torch old enough to
    predate the backend.
    """
    backends = getattr(torch, "backends", None)
    mps_backend = getattr(backends, "mps", None) if backends is not None else None
    is_available = getattr(mps_backend, "is_available", None)
    if not callable(is_available):
        return False
    try:
        if not is_available():
            return False
    except Exception:  # pragma: no cover - a probe that throws is a no
        return False
    return hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache")


def _stage_model_locally(model_id: str, revision: str | None = None) -> str | None:
    """Copy the SNAPSHOT this load resolves to onto node-local disk and
    return the staged hub directory (a valid HF cache holding ``models--…``),
    or None when staging is off or impossible.

    Why (measured live on a cluster, 2026-07-17): a cold load against slow shared
    storage reads the full weight file TWICE at mmap-fault granularity
    (checkpoint materialization, then the device copy) — /scratch delivered
    ~12 MB/s that way against ~42 MB/s sequential. Staging is ONE sequential
    pass at the filesystem's best rate; the load then runs at node-NVMe
    speed, and repeat loads on this node reuse the staged copy for free.

    Revision-scoped (engineer review 2026-07-17): only the resolved
    snapshot is copied and the reuse marker names its commit — a repo-level
    marker both over-copied multi-snapshot repos and could satisfy reuse
    while MISSING a newly requested revision, which under offline mode +
    ``cache_dir`` would fail a load the shared cache could serve. A second
    revision stages alongside the first in the same staged repo.

    Never raises: any failure (missing dir, insufficient space, copy error)
    logs one line and falls back to the shared cache. Snapshot symlinks are
    dereferenced into real files (no ``blobs/`` tree), and the size
    preflight counts exactly the bytes the copy will write.
    """
    root = _stage_root()
    if not root:
        return None
    import shutil
    repo = "models--" + model_id.replace("/", "--")
    src = os.path.join(hf_hub_dir(), repo)
    if not os.path.isdir(src):
        return None
    resolved = _resolve_staging_commit(src, revision)
    if resolved is None:
        _log_line(f"staging skipped: cannot resolve revision "
                  f"{revision or 'main'!r} for '{model_id}' in the shared "
                  "cache — loading from the shared cache")
        return None
    commit, ref_name = resolved
    # Belt and braces over _resolve_staging_commit's own checks: everything
    # below joins these into paths that get created, written, and rmtree'd.
    if not _safe_cache_component(commit) or (
            ref_name is not None and not _safe_cache_component(ref_name)):
        _log_line(f"staging skipped: unsafe revision component for "
                  f"'{model_id}' — loading from the shared cache")
        return None
    src_snapshot = os.path.join(src, "snapshots", commit)
    # A snapshot dir that is itself a symlink out of the cache must not be
    # dereferenced into the staged copy.
    snapshots_root = os.path.realpath(os.path.join(src, "snapshots"))
    if not os.path.realpath(src_snapshot).startswith(snapshots_root + os.sep):
        _log_line(f"staging skipped: snapshot for '{model_id}' resolves "
                  "outside the shared cache — loading from the shared cache")
        return None
    staged_hub = os.path.join(root, "steerlab-hf-hub")
    dst_repo = os.path.join(staged_hub, repo)
    dst_snapshot = os.path.join(dst_repo, "snapshots", commit)
    marker = os.path.join(dst_repo, f".steerlab-staged-{commit}")

    def _write_ref() -> None:
        if ref_name is None:
            return
        refs_dir = os.path.join(dst_repo, "refs")
        os.makedirs(refs_dir, exist_ok=True)
        with open(os.path.join(refs_dir, ref_name), "w",
                  encoding="utf-8") as handle:
            handle.write(commit)

    if os.path.exists(marker) and os.path.isdir(dst_snapshot):
        try:
            # Keep the ref pointing at THIS commit: the shared ref may have
            # moved since an earlier staging of a different snapshot.
            _write_ref()
        except OSError:
            return None
        _log_line(f"reusing staged '{model_id}'@{commit[:12]} at {dst_repo}")
        return staged_hub
    try:
        os.makedirs(os.path.join(dst_repo, "snapshots"), exist_ok=True)
        # Exactly the bytes the dereferencing copy will write: every entry
        # resolved, NO dedup (copytree copies each entry separately).
        total = 0
        for dirpath, _dirs, files in os.walk(src_snapshot):
            for name in files:
                total += os.stat(
                    os.path.realpath(os.path.join(dirpath, name))).st_size
        gib = total / (1 << 30)
        free = shutil.disk_usage(dst_repo).free
        if total + (1 << 30) > free:
            _log_line(
                f"staging skipped: '{model_id}'@{commit[:12]} needs "
                f"{gib:.1f} GiB (+1 GiB margin) but {root} has "
                f"{free / (1 << 30):.1f} GiB free — loading from the shared "
                "cache")
            return None
        _set_phase(f"staging '{model_id}' ({gib:.1f} GiB) to node-local disk")
        _log_line(f"staging '{model_id}'@{commit[:12]} ({gib:.1f} GiB) to "
                  f"{dst_snapshot}")
        started = time.monotonic()
        tmp = f"{dst_snapshot}.tmp-{os.getpid()}"
        for stale in (tmp, dst_snapshot):
            if os.path.exists(stale):
                shutil.rmtree(stale)

        # Live progress (researcher request 2026-07-18: a 15-minute staging
        # copy showed a static caption): a watcher sums the bytes landed so
        # far and updates the loader phase, which the SSE heartbeat relays
        # — "staging … 3.2/8.0 GiB (40%)" ticking in the app.
        stop_watch = threading.Event()

        def _watch_staging() -> None:
            ticks = 0
            while not stop_watch.wait(5.0):
                ticks += 1
                done = 0
                for dirpath, _dirs, files in os.walk(tmp):
                    for name in files:
                        try:
                            done += os.path.getsize(
                                os.path.join(dirpath, name))
                        except OSError:
                            continue
                pct = min(100, round(100 * done / total)) if total else 0
                text = (f"staging '{model_id}' — {done / (1 << 30):.1f}/"
                        f"{gib:.1f} GiB ({pct}%) to node-local disk")
                _set_phase(text)
                if ticks % 12 == 0:
                    # Once a minute the err file gets a durable line too —
                    # progress must not live only in a connected client.
                    _log_line(text)

        watcher = threading.Thread(target=_watch_staging, daemon=True)
        watcher.start()
        try:
            shutil.copytree(src_snapshot, tmp, symlinks=False)
        finally:
            stop_watch.set()
            watcher.join(timeout=1.0)
        os.rename(tmp, dst_snapshot)
        _write_ref()
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write(f"{model_id} {commit}\n")
        elapsed = time.monotonic() - started
        rate = total / max(elapsed, 1e-6) / (1 << 20)
        _log_line(f"staged '{model_id}'@{commit[:12]} in {elapsed:.1f}s "
                  f"({rate:.0f} MB/s); loading from node-local disk")
        return staged_hub
    except Exception as exc:  # noqa: BLE001 - staging must never block a load
        _log_line(f"staging failed ({exc}); loading from the shared cache")
        return None


def _download_into_cache(model_id: str, revision: str | None, log) -> None:
    """Fetch the snapshot through the install verb's cancellable child
    process BEFORE ``from_pretrained`` can begin an inline, uncancellable
    one (field incident 2026-08-29: that inline download held the only
    resident-model slot for ~55 GB with no way to free it). The child's
    progress lines feed both the stderr log and the load PHASE, so SSE
    heartbeats carry live download progress. A failure with a usable local
    snapshot falls back to the shared cache (same posture as staging); a
    failure with no local bytes is the load's failure, spoken with the
    install verb's actionable remedy. A cancel observed by the child
    terminates its whole process group and surfaces as LoadCancelled."""
    # model_install lives in the API layer but is deliberately
    # self-contained (stdlib only) — one downloader implementation beats a
    # clean layering diagram here. Imported lazily to keep pure-engine
    # imports framework-free.
    from ..api import model_install

    def _progress(text: str) -> None:
        log(text)
        _set_phase(f"downloading '{model_id}' from the hub — {text}")

    _set_phase(f"downloading '{model_id}' from the hub")
    try:
        model_install.run_install(model_id, revision, _progress,
                                  cancelled=load_cancel_requested)
    except RuntimeError as exc:
        if load_cancel_requested():
            _set_phase(None)
            raise LoadCancelled(
                f"load of '{model_id}' cancelled during its hub download — "
                "partial blobs stay in the cache and a re-load resumes from "
                "them; the resident slot is free again") from exc
        if snapshot_size_bytes(model_id, revision) is not None:
            log(f"hub download failed ({exc}); trying the cached snapshot")
            return
        raise ModelLoadError(str(exc), advice_complete=True) from exc


def load(model_id: str, revision: str | None = None, *,
         dtype: str | None = None, device: str | None = None,
         device_map: str | None = None) -> SteeredModel:
    """Load a hookable model on the chosen device.

    ``device`` / ``dtype`` default to :func:`resolve_device` /
    :func:`default_dtype` (``None`` or ``"auto"`` means auto). On a single
    device (MPS, CPU, or one CUDA GPU) the model is moved with ``.to(device)``;
    ``device_map`` (accelerate sharding) is used only for multi-GPU CUDA.
    """
    dev = resolve_device(device)
    requested_dtype = (dtype or "").strip() or None
    if not dtype or dtype.lower() == "auto":
        dtype = default_dtype(dev, model_id)
    # An UNKNOWN dtype used to fall back to float32 SILENTLY (external review
    # round 4, finding 2). Combined with the freeze gate that requires a
    # judge dtype pin, that made the pin a false claim: the manifest could
    # say "bfloat16" — or "banana" — while the load ran fp32, and the
    # judgment artifact recorded the request rather than the reality. A
    # closed vocabulary with a refusal is the only version of that pin worth
    # having.
    canonical = normalize_dtype(dtype)
    if canonical is None:
        raise ModelLoadError(
            f"unknown dtype '{dtype}' for '{model_id}' — this engine loads "
            f"only {', '.join(DTYPE_VOCABULARY)} (aliases bf16/fp16/fp32, or "
            "'auto' to let the device decide). An unrecognized value used to "
            "load float32 silently, so a pinned dtype could be a false claim")
    dtype = canonical
    resolved_dtype = _DTYPES[canonical]

    # A silent CPU fallback is the worst failure mode on a cluster (live
    # shakedown 2026-07-19): a Slurm job submitted without GPU resources
    # sees no CUDA, quietly loads a multi-GB model in float32 on the
    # controller/compute host's CPU, crawls through the weight shards, and
    # dies mid-load with no explanation. Large models now REFUSE up front
    # (2026-07-21: the warning alone did not save a 16 GB controller job
    # from an 8 GiB gemma-3-4b float32 load — it died at ~58% of the
    # weights); small models keep the warn-and-proceed for CPU smoke tests,
    # and STEERLAB_ALLOW_CPU_LOAD=1 restores it deliberately for large ones.
    # Either way, say what is happening BEFORE the first byte stages.
    if dev == "cpu":
        _assert_cpu_capacity(model_id, revision, dtype)
        print(
            "model_loader: ⚠ NO GPU VISIBLE to torch (CUDA/MPS unavailable) "
            f"— loading '{model_id}' on CPU in {dtype}. A multi-billion-"
            "parameter model will be very slow and may exhaust host memory. "
            "On a Slurm site this usually means the job requested no GPU "
            "(empty gres/gpus) — resubmit with the site's GPU resources "
            "(Remote options → GPU gres), or use a small model for "
            "CPU-only smoke tests.",
            file=sys.stderr, flush=True)

    # BEFORE any weights move: does the installed torch even carry kernels
    # for this GPU? Measured live 2026-07-17: a P100 (sm_60) session read 8 GB
    # of shards for 7½ minutes and then wedged, because the cu128 wheel ships
    # sm_75+ only — fail in milliseconds with the remedy instead.
    if dev.startswith("cuda"):
        _assert_cuda_kernels_for_device(dev)
        # And does the device have room for the weights at all? Refuse in
        # milliseconds instead of staging tens of GiB first (2026-07-18).
        _assert_gpu_capacity(dev, model_id, revision, resolved_dtype)

    if _is_mlx_repo(model_id):
        raise ModelLoadError(
            f"'{model_id}' is an MLX-quantized repo, which the PyTorch/HF engine "
            f"cannot load — MLX's weight format is Apple-only and unrelated to HF "
            f"safetensors. Use a full-precision HF repo instead (e.g. 'Qwen/Qwen3-4B' "
            f"rather than 'Qwen/Qwen3-4B-MLX-4bit'). The cluster is a separate "
            f"extraction substrate by design; see CHANGES-TO-SWIFT-SIDE.md › WS7.")

    # Phase-timed load logging (live cluster run 2026-07-17, round 2): the HF
    # "Loading weights" bar can complete near-instantly (mmap'd safetensors,
    # lazy pages) while the REAL byte movement happens silently inside
    # `.to(device)` — an err file that goes dark exactly where the minutes go
    # reads as a hang. Each phase prints its wall time and size to stderr.
    def _log(text: str) -> None:
        print(f"model_loader: {text}", file=sys.stderr, flush=True)

    def _check_cancel() -> None:
        # Polled at every phase boundary (and continuously through a hub
        # download): a cancel request interrupts a download within seconds
        # and a weight copy at the next boundary — never mid-tensor.
        if load_cancel_requested():
            _set_phase(None)
            raise LoadCancelled(
                f"load of '{model_id}' cancelled on request — the resident "
                "slot is free again; load a model to continue")

    started = time.monotonic()
    _check_cancel()
    try:
        # Cold-cache downloads go through the SAME cancellable child
        # process the install verb uses (field incident 2026-08-29: a
        # ~55 GB download inside ``from_pretrained`` held the only
        # resident-model slot with no cancel short of SIGTERMing the
        # engine). The child's progress lines become the load PHASE, so
        # SSE heartbeats carry live download progress to the client.
        if needs_hub_download(model_id, revision):
            _download_into_cache(model_id, revision, _log)
            _check_cancel()
        # Node-local staging (when configured): one sequential copy to fast
        # local disk, then every read below hits NVMe instead of shared
        # storage. Falls back silently to the shared cache on any failure.
        staged_hub = _stage_model_locally(model_id, revision)
        _check_cancel()
        cache_kwargs = {"cache_dir": staged_hub} if staged_hub else {}
        _set_phase(f"reading '{model_id}' checkpoint from the cache")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, **cache_kwargs)
        # Per-device kernel choice (see attention_implementation). None means
        # "say nothing" — HF keeps its own default, so CUDA loads are
        # byte-for-byte the call they were before this existed.
        attn = attention_implementation(dev)
        attn_kwargs = {"attn_implementation": attn} if attn else {}
        _log(f"loading '{model_id}' (dtype {dtype}, target {dev}"
             + (f", attention {attn}" if attn else "") + ")")
        phase = time.monotonic()
        if dev.startswith("cuda") and device_map is not None:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, device_map=device_map,
                **hf_dtype_kwargs(resolved_dtype), **cache_kwargs)
            _log(f"weights materialized (device_map={device_map}) in "
                 f"{time.monotonic() - phase:.1f}s")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, **hf_dtype_kwargs(resolved_dtype),
                **attn_kwargs, **cache_kwargs)
            gib = sum(p.numel() * p.element_size()
                      for p in model.parameters()) / (1 << 30)
            _check_cancel()
            _log(f"weights materialized ({gib:.1f} GiB) in "
                 f"{time.monotonic() - phase:.1f}s; moving to {dev} — this is "
                 "where cold-cache bytes actually stream off disk")
            _set_phase(f"moving {gib:.1f} GiB to {dev} — cold pages stream "
                       "off disk here")
            phase = time.monotonic()
            model.to(dev)
            _log(f"device copy to {dev} finished in "
                 f"{time.monotonic() - phase:.1f}s")
        _set_phase("finalizing (hooks + wrapper)")
    except ModelLoadError:
        _set_phase(None)
        # A cancel observed AFTER weights materialized leaves a multi-GiB
        # module tree bound in this frame — drop it before unwinding so the
        # cancel actually returns the memory (same debris rationale as the
        # generic handler below; no device sweep needed, the copy to the
        # device has not started at any cancel checkpoint).
        import gc
        try:
            del model  # noqa: F821 - bound only if from_pretrained returned
        except NameError:
            pass
        gc.collect()
        raise
    except Exception as exc:  # noqa: BLE001 - convert to a user-actionable message
        _set_phase(None)
        msg = str(exc)
        # Free the debris HERE, in the frame that owns it (engineer review
        # 2026-07-18): the exception's traceback pins this frame's `model`
        # local AND torch's .to() frames (each referencing the module tree),
        # so any cleanup that runs after unwinding sweeps nothing and the
        # partially-moved weights squat on the device. Print the real
        # traceback now (durable in the err file), sever the frames, drop
        # the local, and sweep.
        import gc
        import traceback
        traceback.print_exc()
        exc.__traceback__ = None
        try:
            del model  # noqa: F821 - bound only if from_pretrained returned
        except NameError:
            pass
        gc.collect()
        if dev.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if "quant_method" in msg or "quantization" in msg.lower():
            raise ModelLoadError(
                f"'{model_id}' appears to be quantized in a format HF can't load "
                f"(likely MLX). Use a full-precision HF repo.") from exc
        raise ModelLoadError(f"could not load '{model_id}': {msg}") from exc
    model.eval()
    hooked = HookedModel(model)
    try:
        # Stamp the dtype the model ACTUALLY runs in (first parameter), not the
        # request — "auto" and alias spellings resolve to one truthful name.
        actual_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    except StopIteration:  # pragma: no cover - a parameterless model
        actual_dtype = dtype
    # A PIN that the load did not honor is a false pin, so say so rather than
    # returning a model whose recorded dtype contradicts the manifest. With
    # the vocabulary closed above this is near-unreachable today — it exists
    # so that a future device fallback (or a checkpoint that forces its own
    # dtype) fails loudly instead of silently, which is exactly the failure
    # mode this finding was about.
    pinned = normalize_dtype(requested_dtype)
    if pinned is not None and normalize_dtype(actual_dtype) != pinned:
        raise ModelLoadError(
            f"'{model_id}' was pinned to dtype '{pinned}' but loaded as "
            f"'{actual_dtype}' — the pin cannot be honored on this device, "
            "so any artifact naming it would be a false claim. Pin the dtype "
            "the device can actually run, or leave it unset for 'auto'")
    steered = SteeredModel(model=model, tokenizer=tokenizer, hooked=hooked,
                           model_id=model_id, revision=revision or cached_revision(model_id),
                           dtype=actual_dtype,
                           # Read back, not echoed: the truthful kernel even
                           # when the request said nothing (CUDA → "sdpa").
                           attn_implementation=getattr(
                               model.config, "_attn_implementation", None))
    # The completion marker means HOOKABLE AND PUBLISHABLE — printed after
    # HookedModel and the wrapper exist, not merely after the device copy
    # (engineer review 2026-07-17).
    _log(f"load complete: '{model_id}' ({actual_dtype}) on {dev} in "
         f"{time.monotonic() - started:.1f}s total")
    _set_phase(None)
    return steered


def cached_revision(model_id: str) -> str | None:
    """Commit hash of the locally cached snapshot (HF cache ``refs/main``).

    This is what a revision-less load actually runs, so it is the value to pin
    for reproducibility. Mirrors Swift ``cachedRevision``.
    """
    repo = "models--" + model_id.replace("/", "--")
    ref = os.path.join(hf_hub_dir(), repo, "refs", "main")
    try:
        with open(ref, encoding="utf-8") as handle:
            commit = handle.read().strip()
    except OSError:
        return None
    return commit or None


def local_model_ids(cache_root: str | None = None) -> list[str]:
    """Cached repos THIS engine can actually load as a causal LM.

    The HF cache is shared: on a Mac with the MLX app, and on every substrate
    with non-model artifact repos we deliberately stage there (Gemma Scope SAEs,
    J-lens Jacobians). Advertising those in /api/state offers the client models
    that 400 on load; filter them here, keeping the loud rejection in load() for
    direct requests.

    Two filters, because neither subsumes the other:

    * **Shape** — a cached repo with no ``config.json`` in any snapshot is not
      a loadable causal LM. This is the general, honest predicate, and it
      covers every artifact repo without naming any of them: Gemma Scope
      (whose snapshot holds ``resid_post/``), the J-lens repo (whose snapshot
      holds one directory per model), and whatever gets cached next. It exists
      because the name-based checks below could only ever catch the artifact
      class someone had already been bitten by: staging the first lens put
      ``neuronpedia/jacobian-lens`` straight into the model picker, where it
      errored on selection (live 2026-07-27).
    * **MLX** — MLX-quantized repos DO ship a ``config.json``, so shape cannot
      see them; their weights are simply an Apple-only format this engine
      cannot read. This check is not redundant and must stay.
    * **Gemma Scope** — every release we hold lacks ``config.json``, so shape
      already excludes them. Kept as a cheap explicit guard in case a future
      release ships one: a redundant filter costs a substring compare, while
      removing a working one to tidy a docstring costs a regression.

    A repo fetched with ``allow_patterns`` that excluded ``config.json`` also
    drops off this list. That is correct: it could not be loaded either.
    """
    return _cached_repo_ids(cache_root, require_model_config=True)


def _cached_repo_ids(cache_root: str | None = None, *,
                     require_model_config: bool) -> list[str]:
    """The shared cache walk behind :func:`local_model_ids` and
    :func:`local_model_sizes`.

    ``require_model_config`` is the difference between the two callers, and it
    is not cosmetic. ADVERTISING a model asks "can this load?", which needs the
    shape test. ACCOUNTING for bytes asks "what is on disk?", and a repo being
    downloaded right now legitimately has blobs before its ``config.json``
    lands — requiring one there hides in-flight installs from the size/progress
    view (caught by the completion-marker staging test, 2026-07-27).
    """
    hub = hf_hub_dir(cache_root)
    try:
        entries = os.listdir(hub)
    except OSError:
        return []
    ids: list[str] = []
    for name in entries:
        if not name.startswith("models--"):
            continue
        path = os.path.join(hub, name)
        has_marker = os.path.exists(os.path.join(path, "refs", "main")) \
            or os.path.exists(os.path.join(path, "snapshots"))
        if not has_marker:
            continue
        encoded = name[len("models--"):]
        parts = encoded.split("--")
        model_id = parts[0] + "/" + "--".join(parts[1:]) if len(parts) >= 2 else encoded
        if _is_mlx_repo(model_id) or "gemma-scope" in model_id.lower():
            continue
        if require_model_config and not _has_causal_lm_config(path):
            continue
        ids.append(model_id)
    return sorted(ids)


def _has_causal_lm_config(repo_dir: str) -> bool:
    """True when any snapshot of this cached repo has a root ``config.json``.

    The minimal thing ``from_pretrained`` needs before it can even decide what
    architecture it is loading. Checked across all snapshots rather than the
    ``refs/main`` one alone, so a repo pinned to a non-default revision still
    lists.
    """
    snapshots = os.path.join(repo_dir, "snapshots")
    try:
        revisions = os.listdir(snapshots)
    except OSError:
        return False
    return any(os.path.exists(os.path.join(snapshots, rev, "config.json"))
               for rev in revisions)


def _hf_cache_root() -> str:
    return os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface")


def hf_hub_dir(cache_root: str | None = None) -> str:
    """The hub directory holding ``models--org--name`` repos, resolved the
    way ``huggingface_hub`` itself resolves it: an explicit ``cache_root``
    argument (tests/callers) means ``<cache_root>/hub``; else ``HF_HUB_CACHE``
    verbatim (that variable names the hub directory DIRECTLY, no ``hub/``
    suffix); else ``$HF_HOME/hub``; else ``~/.cache/huggingface/hub``.

    The install verb downloads through ``huggingface_hub`` (which honors
    ``HF_HUB_CACHE``), so every LISTING must follow the same resolution —
    hand-rolling ``$HF_HOME/hub`` here while ``snapshot_download`` honored
    ``HF_HUB_CACHE`` made a successful install invisible to ``/api/state``
    ("no models installed" against a freshly installed snapshot, live
    2026-07-17)."""
    if cache_root:
        return os.path.join(cache_root, "hub")
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return hub
    return os.path.join(_hf_cache_root(), "hub")


def _final_norm(model: torch.nn.Module):
    """The model's own final-norm module (RMSNorm/LayerNorm before the head).

    Logit-lens readouts MUST go through this before the unembedding — the
    lens question is "what would the model compute from this residual state",
    and the model always normalizes before the head. Located generically from
    the HF module structure: decoder-only (``model.norm``), Gemma 3
    multimodal with the text model nested under ``model.language_model``
    (both the current ``model.language_model.norm`` and the older
    ``language_model.model.norm`` layouts), and GPT-2-style ``transformer.ln_f``.
    """
    for accessor in (lambda m: m.model.norm,
                     lambda m: m.model.language_model.norm,
                     lambda m: m.language_model.model.norm,
                     lambda m: m.transformer.ln_f):
        try:
            return accessor(model)
        except AttributeError:
            continue
    return None


def _lm_head(model: torch.nn.Module):
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
        return model.get_output_embeddings()
    raise RuntimeError("could not locate the output head for logit lens")
