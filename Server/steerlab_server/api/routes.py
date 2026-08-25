"""Route handlers (parallel to the ``WebServer`` route table).

A subset is wired to real compute now — load, generate (+ SSE stream), extract,
and the experiment job verbs — and the long GPU verbs go through the async job
model. Handlers are built against a small ``ServiceState`` so the app wires the
singletons in one place.
"""

from __future__ import annotations

import json
import queue
import tempfile
import threading
import tarfile

import base64
import hashlib
import os
from contextlib import contextmanager
from dataclasses import asdict
from types import SimpleNamespace

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import (FileResponse, JSONResponse, Response,
                               StreamingResponse)

from .. import build_identity
from ..experiment import catalog, paths, tasks
from ..experiment import resume as resume_mod
from ..experiment.generate import CellInjection, stream_generate
from ..experiment.manifest import Manifest
from ..steering import model_loader, vector_math, vector_store
from . import dto, gpu_session, workspace_lock
from .executors import SlurmExecutor, SlurmResources
from .jobs import ResubmitRefused
from .model_registry import ModelRegistry
from .profile import (ServerProfile, capability_snapshot, server_role,
                      validate_profile, workspace_switch_policy)
from .safe_paths import SafePathResolver, is_contained, safe_name
from .submissions import PreflightRejection, submit_run_bundle, submit_study
from . import variant_chat


def _controller_blocks_resident_models() -> bool:
    """Whether THIS process is a controller that must never hold a resident
    model — loading a 27B model into a 1-core allocation is never right;
    GPU work is a child job or a GPU session. Keyed to the ROLE
    (GPU-SESSION-PLAN §2.2; the legacy cluster+slurm combination derives to
    role=controller, so existing deployments keep this gate).
    STEERLAB_ALLOW_RESIDENT_MODELS=1 stays the documented manual escape
    hatch for a server run by hand INSIDE an interactive GPU allocation
    (the `interact` recipe)."""
    return (server_role() == "controller"
            and os.environ.get("STEERLAB_ALLOW_RESIDENT_MODELS", "") not in
            ("1", "true", "yes"))


def _registered_session_state() -> str | None:
    """State of a non-terminal GPU-session record on this controller, else
    None. This is the WORDING seam, deliberately looser than
    ``_delegation_target`` (which additionally requires a dialable worker):
    an honest refusal must never say "no GPU session running" while one is
    registered, even mid-start."""
    if server_role() != "controller":
        return None
    from . import gpu_session
    record = gpu_session.read_session_record()
    if record is None or record.get("state") in \
            gpu_session.TERMINAL_SESSION_STATES:
        return None
    return str(record.get("state") or "unknown")


def _refuse_controller_resident_load() -> None:
    """The role gate for resident models on a controller. The detail is
    SESSION-AWARE (live 2026-07-21: with a session serving, a verb that was
    not delegated to it still claimed "no GPU session running" — a false
    statement that sends the researcher chasing a session that already
    exists): each variant states what is actually true and names the
    working remedy."""
    if not _controller_blocks_resident_models():
        return
    session_state = _registered_session_state()
    if session_state is None:
        raise HTTPException(
            status_code=409,
            detail="model loading is disabled on the controller "
                   "(role: controller) — no GPU session running: start "
                   "one (POST /api/session/start or the app's GPU "
                   "Session control) and this same route will proxy to "
                   "it; batch studies go through POST "
                   "/api/studies/submit. STEERLAB_ALLOW_RESIDENT_MODELS=1"
                   " remains the manual escape hatch inside an "
                   "interactive GPU allocation")
    if _delegation_target() is None:
        # Registered but not dialable (queued/starting/unknown, or the
        # worker went silent): never claim absence — name the wait.
        raise HTTPException(
            status_code=409,
            detail="model loading is disabled on the controller "
                   f"(role: controller) — a GPU session is registered "
                   f"(state: {session_state}) but its worker is not "
                   "reachable yet: check GET /api/session and retry once "
                   "it reports ready; batch studies go through POST "
                   "/api/studies/submit")
    raise HTTPException(
        status_code=409,
        detail="model loading is disabled on the controller "
               f"(role: controller) — a GPU session IS running "
               f"(state: {session_state}), but this route runs "
               "controller-side and is not delegated to it: interactive "
               "routes proxy to the session automatically, and delegated "
               "verbs reach it through controller jobs; batch-scale work "
               "goes through POST /api/studies/submit. "
               "STEERLAB_ALLOW_RESIDENT_MODELS=1 remains the manual "
               "escape hatch inside an interactive GPU allocation")


class ServiceState:
    """Holds resident model slots and serializes forward passes per model.

    Every forward pass on a registry-managed model — chat, concept authoring,
    experiment jobs, variant chat, multi-agent — must run under that model's
    slot lock. Two requests touching one ``HookedModel`` concurrently would
    cross-contaminate the injection list, so there is exactly one lock authority
    per model (the registry slot). ``exclusive_gpu`` is a separate coarse lock
    used only by LoRA training, which loads its own model outside the registry.
    """

    def __init__(self):
        self.model: model_loader.SteeredModel | None = None
        self.profile = ServerProfile.from_env()
        self.registry = ModelRegistry()
        # Coarse serializer for GPU work that does NOT go through the registry
        # (LoRA training loads its own model via peft). Registry-managed forward
        # passes use per-slot locks instead, so multi-GPU serving is preserved.
        self.exclusive_gpu = threading.Lock()
        self.default_dtype = "auto"
        self.default_device: str | None = None
        # Single-writer rule (GPU-SESSION-PLAN §2.4, acceptance criterion 2):
        # a gpu-session worker constructs NO job subsystem — no JobManager, no
        # reconciler, no durable SQLite store. Both processes would otherwise
        # open the same jobs.sqlite under the shared metadata root, and the
        # worker's reconciler would adjudicate the controller's Slurm records.
        # The controller is the only writer; jobs routes on a worker answer a
        # clear 409 (the `jobs` property below).
        if server_role() == "gpu-session":
            self._jobs = None
        else:
            from .jobs import JobManager
            self._jobs = JobManager(
                capability_provider=lambda: capability_snapshot(self.registry),
                # The daemon's startup reconcile (2026-08-06): orphaned
                # pipeline ledgers are resumed in-process or parked loudly,
                # never left silently invisible.
                reconcile_pipelines=True)

    @property
    def jobs(self):
        """The JobManager, or a 409 on a gpu-session worker. A property (not
        a None attribute) so every job-using route answers an actionable
        refusal instead of a 500 when it is hit on a worker."""
        if self._jobs is None or server_role() == "gpu-session":
            raise HTTPException(
                status_code=409,
                detail="this server runs no job subsystem (role: gpu-session)"
                       " — jobs, submissions, and reconciliation live on the "
                       "controller")
        return self._jobs

    @property
    def job_manager_or_none(self):
        """Non-raising accessor for read paths that degrade honestly on a
        worker (e.g. /api/state reports an empty jobs list)."""
        if server_role() == "gpu-session":
            return None
        return self._jobs

    @property
    def resolver(self) -> SafePathResolver:
        return SafePathResolver(ServerProfile.from_env())

    def load(self, model: str, revision: str | None, dtype: str,
             device: str | None = None) -> None:
        self.default_dtype = dtype
        self.default_device = device
        self.model = self.registry.get_or_load(
            model, revision=revision, dtype=dtype, device=device).model

    def require_model(self) -> model_loader.SteeredModel:
        if self.model is None:
            # On a controller, "load a model first" is a dead end (loads are
            # role-refused there) — the actionable answer is the GPU-session
            # flow, same as /api/load's own refusal (live 2026-07-18: a
            # no-session extraction 409'd with the misleading text).
            _refuse_controller_resident_load()
            raise HTTPException(status_code=409, detail="no model loaded; POST /api/load first")
        return self.model

    def is_busy(self) -> bool:
        return self.registry.any_busy() or self.exclusive_gpu.locked()

    @contextmanager
    def acquire_active(self):
        """Acquire the active UI model's slot lock for the duration of a forward
        pass. Every route that generates/reads activations on ``state.model``
        goes through here so no two share a model object unlocked."""
        m = self.require_model()
        with self.registry.acquire(
                m.model_id, revision=m.revision, dtype=self.default_dtype,
                device=self.default_device) as model:
            yield model

    def acquire_model(self, model_id: str, revision: str | None = None,
                      dtype: str | None = None):
        # Belt-and-braces role guard at the ONE chokepoint every in-process
        # model acquisition funnels through (variant generate, reader
        # fit/score, norm backfill, experiment verbs): a controller is a
        # 1-core submit/poll allocation, and a route that slipped past the
        # session proxy must refuse like /api/load does — not attempt an
        # in-process load that hangs the caller and OOMs the job (live
        # 2026-07-17: a variant-generate with an unreachable session worker
        # fell through and spun forever).
        _refuse_controller_resident_load()
        # A caller-declared dtype wins over the server default (external
        # review round 3, finding 2): a judge that PINS its dtype must be
        # loaded at that dtype, or the manifest claims a pin the load
        # ignored. bf16 and fp16 are different judges.
        return self.registry.acquire(
            model_id, revision=revision, dtype=dtype or self.default_dtype,
            device=self.default_device)


def _max_upload_bytes() -> int:
    """Upper bound on a single uploaded artifact (bundle or variant asset). A
    remote client can otherwise exhaust the staging disk. Read per-request (not a
    module constant) so deployments can tune it without a restart. Override with
    STEERLAB_MAX_UPLOAD_BYTES; default 4 GiB."""
    return int(os.environ.get("STEERLAB_MAX_UPLOAD_BYTES", str(4 * 1024**3)))


def _safe_name(name: str) -> None:
    """Reject path traversal in user-supplied path components."""
    safe_name(name)


def _resolve_optional_dir(resolver: SafePathResolver, ref, root: str):
    """Contain an optional client-supplied directory under ``root`` (no existence
    requirement — an import target may not exist yet). Returns None when absent."""
    if not ref:
        return None
    return resolver.resolve_under(ref, root, allow_local_absolute=True)


def _resolve_optional_file(resolver: SafePathResolver, ref):
    """Contain an optional client-supplied input file under the workspace root."""
    if not ref:
        return None
    return resolver.require_file(ref, allow_local_absolute=True)


def _resolve_injections(items: list[dto.InjectionDTO]) -> list[CellInjection]:
    cells: list[CellInjection] = []
    for item in items:
        centering = item.centering or "none"
        if centering not in ("none", "neutralMean"):
            raise HTTPException(
                status_code=400,
                detail=f"unknown centering '{centering}' — this engine "
                       f"implements 'none' and 'neutralMean'")
        if centering != "none" and item.mode != "ablate":
            raise HTTPException(
                status_code=400,
                detail="centering is an ablation-direction transform — set "
                       "mode 'ablate' or drop it")
        if item.vector is not None:
            if centering != "none":
                # The server has no mean to apply to raw bytes: a client that
                # owns the vector must center it client-side (the Swift app
                # does) — accepting the flag here would silently no-op.
                raise HTTPException(
                    status_code=400,
                    detail="centering requires an artifact reference "
                           "(vectorPath+name); explicit-vector cells must be "
                           "centered by the client")
            vector = item.vector
        elif item.vectorPath and item.name:
            path = SafePathResolver().require_dir(item.vectorPath, allow_local_absolute=True)
            try:
                vectors, sidecar = vector_store.load(path, item.name)
                # Refuse non-finite bytes (load) and foreign-substrate vectors
                # (e.g. swift-mlx artifacts on a shared tree) before injecting.
                vector_store.require_native_substrate(
                    sidecar, os.path.join(path, item.name))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            layer = min(max(0, item.layer), vectors.layer_count - 1)
            vector = vectors.per_layer[layer]
            if item.mode == "ablate":
                try:
                    neutral_mean = vector_store.load_neutral_mean(path, item.name)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc))
                if centering == "neutralMean":
                    if neutral_mean is None:
                        raise HTTPException(
                            status_code=400,
                            detail=f"artifact '{item.name}' carries no stored "
                                   f"neutral mean — re-extract the concept "
                                   f"with a neutral corpus to enable "
                                   f"neutral-mean centering")
                    vector = vector_math.mean_centered(vector, neutral_mean[layer])
                else:
                    _warn_uncentered_ablation_cell(item, layer, vector, neutral_mean)
        else:
            raise HTTPException(status_code=400,
                                detail="injection needs either 'vector' or 'vectorPath'+'name'")
        cells.append(CellInjection(
            layer=item.layer, vector=vector, alpha=item.alpha,
            mode=item.mode, concept=item.concept))
    return cells


def _warn_uncentered_ablation_cell(item: dto.InjectionDTO, layer: int,
                                   vector: list[float],
                                   neutral_mean: list[list[float]] | None) -> None:
    """Ablation mean-alignment preflight for ad-hoc API cells — warning only
    (the request still runs; same threshold as the study/variant preflights)."""
    import warnings as _warnings
    label = item.concept or item.name or "?"
    if neutral_mean is None:
        _warnings.warn(
            f"ablating '{label}' with no stored neutral mean — mean-alignment "
            f"preflight impossible; λ=1 ablation of a mean-aligned direction "
            f"collapses generation (re-extract with a neutral corpus to "
            f"enable the check and centering)", UserWarning, stacklevel=3)
        return
    alignment = vector_math.mean_alignment(vector, neutral_mean[layer])
    if alignment > vector_math.ABLATION_MEAN_ALIGNMENT_WARN_THRESHOLD:
        _warnings.warn(
            f"ablation direction '{label}' at layer {layer} is strongly "
            f"aligned with the neutral residual mean (|cos| {alignment:.2f}; "
            f"threshold {vector_math.ABLATION_MEAN_ALIGNMENT_WARN_THRESHOLD}) "
            f"— pass centering 'neutralMean' or expect single-token collapse",
            UserWarning, stacklevel=3)


def _missing_variant_artifacts(variant) -> list[dict]:
    """Referenced vector/adapter/neutral-basis artifacts that do not resolve
    on this host. Delegates to ``model_variant.missing_artifacts`` — the SAME
    check the run/panel artifact preflight applies, through the same
    resolution (rebase fallback included) the generation path uses, so what
    this route reports is exactly what a run would fail to open."""
    from ..experiment import model_variant
    return model_variant.missing_artifacts(variant)


_SSE_DONE = object()


def _locked_sse(produce, done_extra=None, preamble=None):
    """Run a chunk generator (which holds a model slot lock) in a background
    thread that feeds an UNBOUNDED queue, and stream the queue to the client.

    The lock is held only while ``produce`` is generating, never while the
    client consumes: a slow or stalled SSE reader can never keep the GPU lock,
    because the producer never blocks on the queue. Buffered output is bounded
    by the generation's own max_tokens, not by client behavior.

    ``done_extra`` is a zero-arg callable returning extra fields to merge into
    the terminal ``done`` event (e.g. resolved model id / variant metadata).

    ``preamble`` is an optional event dict emitted BEFORE the first produced
    chunk — the honesty channel for work that happens before any token can
    exist (live 2026-07-17: a chat against a not-yet-resident model performed
    a multi-minute cold load inside the stream with zero bytes sent, and the
    client had nothing to show but a spinner). While a preamble is pending
    resolution (no chunk/done/error yet), the stream emits a status HEARTBEAT
    every ``STEERLAB_SSE_HEARTBEAT_SECONDS`` (default 15) with the elapsed
    time, so the client's caption ticks and a silent minutes-long load is
    distinguishable from a dead stream (second live round, 2026-07-17: an
    "instant" weight bar hid a silent device copy and the one preamble line
    sat frozen for minutes). The heartbeats also keep the proxy/tunnel path
    warm.

    ``produce`` may yield strings (wrapped as ``{"chunk": …}``) or ready-made
    event dicts (passed through verbatim — e.g. a ``{"status": …}``
    transition once the model is acquired and generation begins).
    """
    q: "queue.Queue" = queue.Queue()
    if preamble:
        q.put(dict(preamble))
    # Cooperative cancellation: set when the SSE consumer goes away (client
    # disconnect closes event_stream → finally). `produce` threads it into the
    # generation loop as a stopping criterion, so an abandoned chat cannot
    # squat on the model-slot lock until max_tokens.
    stop = threading.Event()

    def worker():
        try:
            for chunk in produce(stop):
                q.put(chunk if isinstance(chunk, dict) else {"chunk": chunk})
            done = {"done": True}
            if done_extra is not None:
                done.update(done_extra())
            q.put(done)
        except Exception as exc:  # noqa: BLE001 - reported to the client in-band
            # ALSO to stderr (the worker's err file): the SSE error event is
            # the only other copy, and a dropped tunnel/proxy loses it — a
            # streamed failure must leave a durable diagnostic record
            # (engineer review 2026-07-17).
            import traceback
            traceback.print_exc()
            error_event = {"error": str(exc)}
            # Chat-template constraint refusals carry the same structured
            # detail the non-stream routes return as a 400, so streaming
            # clients can act on {message, constraint, modelID} too.
            from ..experiment.prompt_render import ChatTemplateConstraintError
            if isinstance(exc, ChatTemplateConstraintError):
                error_event["detail"] = exc.detail()
            q.put(error_event)
        finally:
            q.put(_SSE_DONE)

    threading.Thread(target=worker, daemon=True).start()

    def event_stream():
        import time as _time
        try:
            interval = float(os.environ.get(
                "STEERLAB_SSE_HEARTBEAT_SECONDS", "15"))
        except ValueError:
            interval = 15.0
        started = _time.monotonic()
        base_status = (preamble or {}).get("status")
        heartbeats = bool(base_status)
        try:
            while True:
                if heartbeats:
                    try:
                        item = q.get(timeout=max(0.05, interval))
                    except queue.Empty:
                        elapsed = int(_time.monotonic() - started)
                        # Include the loader's live phase when one is running:
                        # "43s elapsed" alone told the researcher nothing
                        # about WHICH phase was eating the minutes.
                        phase = model_loader.current_load_phase()
                        text = f"{base_status} ({elapsed}s elapsed" + (
                            f"; {phase}" if phase else "") + ")"
                        yield f"data: {json.dumps({'status': text})}\n\n"
                        continue
                else:
                    item = q.get()
                if item is _SSE_DONE:
                    return
                # Any substantive event (chunk/done/error) ends the wait;
                # status-only events keep the heartbeat ticking through a
                # long acquire + prefill, and later heartbeats repeat the
                # LATEST status ("ready — generating"), not the stale one.
                if isinstance(item, dict) and set(item.keys()) == {"status"}:
                    base_status = item["status"]
                else:
                    heartbeats = False
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            stop.set()

    return event_stream()


def _refuse_worker_submission() -> None:
    """Studies and raw sbatch submissions never target the GPU session worker
    (GPU-SESSION-PLAN acceptance criterion 4) — it runs no job subsystem and
    its allocation is sized for interactive chat, not a study matrix."""
    if server_role() == "gpu-session":
        raise HTTPException(
            status_code=409,
            detail="studies go through the controller — this GPU session "
                   "worker runs no job subsystem (role: gpu-session)")


def _delegation_target():
    """(node, port) of a live session worker THIS controller can delegate
    synchronous compute to, else None. Non-controller roles never delegate
    (a workstation runs its own jobs; a worker never re-delegates)."""
    if server_role() != "controller":
        return None
    from . import gpu_session
    return gpu_session.worker_sync_target()


def _require_model_or_worker(state: ServiceState) -> None:
    """Fail-fast model guard for job-backed authoring compute: on a
    controller with a live session the WORKER holds the model (its own
    require_model enforces at delegation time), so refuse only when there is
    nowhere to delegate."""
    if _delegation_target() is not None:
        return
    state.require_model()


# Job-backed EXPERIMENT verbs with the exact _run_or_submit shape (live
# 2026-07-21: with a session serving, the app's Validate Study still 409'd
# "no GPU session running" — the 2026-07-18 seam wired only the authoring
# verbs): an unconditional model need, minutes-scale compute, and a run
# directory written into the shared workspace by whichever process holds the
# model (single-writer rule for JOBS is untouched — the worker writes runs/,
# never jobs.sqlite).
_DELEGATED_EXPERIMENT_VERBS = frozenset({"extract", "validate"})

# Batch-scale experiment verbs a controller must neither run in-process NOR
# delegate to the interactive GPU session: a study run, a layer×alpha sweep,
# or a chained pipeline can outlive the session's idle timer and walltime.
# They go through the Slurm study path; the refusal below says so
# immediately instead of queuing a job that dies on the role gate.
_BATCH_ONLY_ON_CONTROLLER = frozenset({"run", "sweep", "pipeline"})


def _refuse_batch_only_verb(verb: str) -> None:
    """Immediate, honest 409 for a batch-scale experiment verb on a
    controller. Never claims "no GPU session running" when one is
    registered — the session simply is not where this work belongs."""
    session_state = _registered_session_state()
    if session_state is not None:
        session_clause = (
            f" — a GPU session is registered (state: {session_state}), but "
            f"{verb} is batch-scale work and is deliberately not delegated "
            "to the interactive session (it could outlive the session's "
            "idle timer and walltime)")
    else:
        session_clause = f" — {verb} is batch-scale work"
    raise HTTPException(
        status_code=409,
        detail=f"experiment {verb} needs a resident model, which the "
               "controller never holds (role: controller)" + session_clause
               + ": submit it as a batch study instead (POST "
               "/api/studies/submit or the app's Submit Study). "
               "STEERLAB_ALLOW_RESIDENT_MODELS=1 remains the manual escape "
               "hatch for a server running inside an interactive GPU "
               "allocation")


def _retain_worker_partial(kind: str, exc: BaseException, log) -> dict | None:
    """Package what a GPU-session worker produced before failing.

    The worker runs tasks synchronously outside the durable-job layer, so
    nothing else can do this: the controller sees only an HTTP error, and
    its own context variable is empty because the directory was created in
    this process. Returns the bundle metadata to relay, or None when
    nothing was produced.

    Best-effort. A packaging failure is logged and swallowed — the caller
    still needs the real error, not the packager's.
    """
    try:
        from ..experiment import bundles, run_status
        directory = run_status.partial_run_directory(exc)
        if not directory:
            return None
        verb = str(kind or "").split(":")[-1] or "job"
        # Derived from the DIRECTORY, not from the job's verb: those differ
        # whenever the failing stage is not the job's own kind (a pipeline
        # job's `run` stage), and the hand-rolled trim then kept the stage
        # in the name.
        name = bundles.experiment_name_of_run_directory(directory, verb)
        try:
            bundles._mark_partial_run(directory, verb=verb, name=name, exc=exc)
        except Exception:  # noqa: BLE001
            pass
        meta = bundles.package_evidence(
            directory,
            failure={"error": f"{type(exc).__name__}: {exc}",
                     "errorType": type(exc).__name__, "verb": verb,
                     "experiment": name, "role": "gpu-session"})
        log(f"partial evidence packaged on the worker from "
            f"{os.path.basename(directory)} — it is on the SHARED workspace, "
            "so the controller can retrieve it")
        return {"bundlePath": meta.get("bundlePath"),
                "bundleSha256": meta.get("bundleSha256"),
                "runDirectory": directory,
                "partialRunID": os.path.basename(directory.rstrip(os.sep)),
                "experiment": name, "verb": verb}
    except Exception as pack_exc:  # noqa: BLE001
        try:
            log(f"partial-evidence packaging failed on the worker: "
                f"{type(pack_exc).__name__}: {pack_exc}")
        except Exception:  # noqa: BLE001
            pass
        return None


def _run_or_submit(state: ServiceState, kind: str, work, *,
                   path: str, body: dict):
    """Execution seam for the job-backed verbs that need a model in-process:
    the authoring verbs (concept extract, grand-mean, reader fit — engineer
    review 2026-07-18: proxying them to the GPU-session worker answered 409
    "no job subsystem") and the experiment verbs in
    _DELEGATED_EXPERIMENT_VERBS (2026-07-21).

    - workstation (JobManager + model in-process): a durable job, as ever.
    - gpu-session worker: NO job subsystem — run ``work`` SYNCHRONOUSLY and
      answer ``{"ok", "sync", "result", "logs"}``; the worker's in-flight
      request counter keeps the idle timer honest for the duration.
    - controller with a live session: submit a durable DELEGATING job that
      re-POSTs the caller's body to the same route on the worker and relays
      its result + log lines — client contract (jobId + job log), durability,
      and the single-writer rule all preserved. Cancelling the controller job
      abandons the wait; the worker finishes its computation regardless (its
      artifacts land in the shared workspace either way).
    - controller with NO session: the actionable session-flow refusal, now —
      never a queued job that fails on the role gate minutes later.
    """
    if server_role() == "gpu-session":
        logs: list[str] = []
        job_like = SimpleNamespace(log=logs.append, cancelled=False)
        # Scope the retention context to THIS request, exactly as the job
        # runner does: the worker serves many requests on shared threads.
        from ..experiment.run_status import current_run_directory
        run_dir_token = current_run_directory.set(None)
        try:
            result = work(job_like)
        except HTTPException:
            raise
        except FileNotFoundError as exc:
            # A missing experiment/input on the shared workspace: a 404 the
            # delegating controller job relays verbatim, never a raw 500.
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - retained, then re-raised
            # Retention on the WORKER (external review round 2, finding 2;
            # widened round 3, finding 4). Catching only ValueError /
            # RuntimeError / ModelLoadError let an ordinary TypeError or
            # assertion strand the worker's output — and the durable
            # JobManager path already retains on any Exception, so the two
            # paths disagreed about which failures are worth keeping.
            # This branch runs the task in-process against a SimpleNamespace
            # job, not a JobManager job, so the durable-job layer's
            # retention never sees it — and the delegating controller only
            # ever receives this HTTP error, with no worker-side context
            # variable to consult. The worker holds the run directory on
            # shared storage, so it must package the partial itself and
            # report where it landed.
            partial = _retain_worker_partial(kind, exc, logs.append)
            detail: dict | str = str(exc)
            if partial:
                detail = {"message": str(exc), "partialEvidence": partial}
            # 400 for the failure classes a caller can act on (bad input,
            # a model that will not load); 500 for anything unexpected, so
            # widening retention does not also start reporting genuine
            # bugs as client errors.
            expected = (ValueError, RuntimeError, model_loader.ModelLoadError)
            raise HTTPException(
                status_code=400 if isinstance(exc, expected) else 500,
                detail=detail)
        finally:
            current_run_directory.reset(run_dir_token)
        return {"ok": True, "sync": True, "result": result, "logs": logs}
    target = _delegation_target()
    if target is not None:
        from . import gpu_session

        def delegated(job):
            job.log(f"delegating {kind} to the GPU session worker at "
                    f"{target[0]}:{target[1]}")
            payload = gpu_session.call_worker_sync(path, body)
            for line in payload.get("logs") or []:
                job.log(f"[worker] {line}")
            return payload.get("result")

        return {"jobId": state.jobs.submit(kind, delegated).id}
    # No session on a controller: refuse NOW with the session-flow message
    # instead of queuing a job that fails on the role gate.
    _refuse_controller_resident_load()
    return {"jobId": state.jobs.submit(kind, work).id}


def _jlens_supported() -> list[dict]:
    """The supported-lens table as data for clients.

    Exposed so a picker can show what is acquirable and at which evidence tier
    without hardcoding a second copy of the table — and so a testing-tier entry
    is unmistakable in the UI rather than inferable from a model id.
    """
    from ..jlens import importer

    return [{"modelID": model_id, "tier": entry["tier"],
             "folder": entry["folder"], "tensor": entry["tensor"]}
            for model_id, entry in sorted(importer.SUPPORTED.items())]


def build_router(state: ServiceState) -> APIRouter:
    router = APIRouter()

    @router.get("/healthz")
    def healthz():
        return {"ok": True}

    @router.get("/api/capabilities")
    def capabilities():
        return capability_snapshot(state.registry)

    @router.get("/api/profile/validate")
    def profile_validate():
        return validate_profile()

    @router.get("/api/state", response_model=dto.StateDTO)
    def get_state():
        m = state.model
        manager = state.job_manager_or_none
        return dto.StateDTO(
            models=model_loader.local_model_ids(),
            modelSizesBytes=model_loader.local_model_sizes(),
            loadedModel=(m.model_id if m else None),
            loadedRevision=(m.revision if m else None),
            device=(str(m.device) if m else None),
            dtype=(getattr(m, "dtype", None) if m else None),
            attnImplementation=(
                getattr(m, "attn_implementation", None) if m else None),
            numLayers=(m.num_layers if m else None),
            hiddenSize=(m.hidden_size if m else None),
            contextWindow=(m.context_window if m else None),
            isBusy=state.is_busy(),
            loadedModels=state.registry.snapshots(),
            jobs=([dto.JobDTO(**j.to_dict()) for j in manager.list()]
                  if manager is not None else []))

    @router.post("/api/load")
    def load(body: dto.LoadRequest):
        # Role gate shared with ServiceState.acquire_model (the in-process
        # acquisition chokepoint) — see _refuse_controller_resident_load for
        # the full rationale and the escape hatch.
        _refuse_controller_resident_load()
        try:
            state.load(body.model, body.revision, body.dtype, body.device)
        except model_loader.ModelLoadError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        m = state.require_model()
        return {"ok": True, "modelID": m.model_id, "revision": m.revision,
                "numLayers": m.num_layers, "hiddenSize": m.hidden_size,
                "device": str(m.device)}

    @router.post("/api/load/stream")
    def load_stream(body: dto.LoadRequest):
        """SSE variant of /api/load (engineer review 2026-07-17): the
        synchronous POST shows a static "loading…" for minutes and races the
        client's request timeout on slow-storage cold loads (live: 10+ min
        off /work). This route reuses the _locked_sse heartbeat machinery —
        preamble, elapsed-time heartbeats, in-band error events, and a
        terminal `done` carrying the same payload the sync route returns.
        The sync /api/load stays for compatibility."""
        _refuse_controller_resident_load()
        residency = state.registry.residency(
            body.model, body.revision,
            dtype=body.dtype or "auto", device=body.device)
        preamble = {"status": (
            f"model {body.model} is already resident — activating"
            if residency == "ready" else
            f"model {body.model} "
            + ("is still loading — waiting for it"
               if residency == "loading" else
               "is not loaded yet — loading it now; cold loads can take minutes"))}

        def produce(stop):
            state.load(body.model, body.revision, body.dtype, body.device)
            m = state.require_model()
            yield {"status": f"model {m.model_id} loaded on {m.device}"}

        def done_extra():
            m = state.model
            if m is None:  # pragma: no cover - produce failed before publish
                return {"ok": False}
            return {"ok": True, "modelID": m.model_id, "revision": m.revision,
                    "numLayers": m.num_layers, "hiddenSize": m.hidden_size,
                    "device": str(m.device)}

        return StreamingResponse(_locked_sse(produce, done_extra, preamble),
                                 media_type="text/event-stream")

    @router.post("/api/models/unload")
    def unload_model(body: dict | None = None):
        body = body or {}
        model_id = body.get("modelID") or body.get("model")
        if model_id:
            removed = state.registry.unload(model_id, revision=body.get("revision"))
            if state.model and state.model.model_id == model_id:
                state.model = None
        else:
            removed = state.registry.unload_all()
            state.model = None
        return {"ok": True, "unloaded": removed}

    @router.post("/api/generate", response_model=dto.GenerateResponse)
    def generate_route(body: dto.GenerateRequest):
        state.require_model()
        cells = _resolve_injections(body.injections)
        from ..experiment.generate import NonFiniteLogitsError
        from ..experiment.generate import generate as run_generate
        from ..experiment.generate import generate_messages
        from ..experiment import prompt_render
        if body.continueFinalMessage and not body.messages:
            raise HTTPException(
                status_code=400,
                detail="continueFinalMessage requires a messages transcript "
                       "ending in the assistant prefix to continue")
        try:
            with state.acquire_active() as model:
                if body.messages:
                    text = generate_messages(
                        model, body.messages, max_tokens=body.maxTokens,
                        temperature=body.temperature, injections=cells,
                        prompt_mode=body.promptMode, system_prompt=body.systemPrompt,
                        qwen_thinking_enabled=body.qwenThinkingEnabled,
                        continue_final_message=body.continueFinalMessage)
                    rendered = prompt_render.render_messages(
                        model.tokenizer, body.messages, model_id=model.model_id,
                        prompt_mode=body.promptMode, system_prompt=body.systemPrompt,
                        qwen_thinking_enabled=body.qwenThinkingEnabled,
                        continue_final_message=body.continueFinalMessage)
                else:
                    text = run_generate(
                        model, body.text, max_tokens=body.maxTokens, temperature=body.temperature,
                        injections=cells, prompt_mode=body.promptMode,
                        system_prompt=body.systemPrompt, qwen_thinking_enabled=body.qwenThinkingEnabled)
                    rendered = prompt_render.render(
                        model.tokenizer, body.text, model_id=model.model_id,
                        prompt_mode=body.promptMode, system_prompt=body.systemPrompt,
                        qwen_thinking_enabled=body.qwenThinkingEnabled)
                model_id = model.model_id
        except NonFiniteLogitsError as exc:
            # Self-naming 400: the message carries step/dtype/steering/adapter
            # context plus the likely causes — never a raw torch 500.
            raise HTTPException(status_code=400, detail=str(exc))
        except prompt_render.ChatTemplateConstraintError as exc:
            # The chat template refused the conversation structure (e.g.
            # Gemma's 'Conversation roles must alternate…'): a structured,
            # actionable 400 ({message, constraint, modelID}) — never a raw
            # jinja TemplateError 500.
            raise HTTPException(status_code=400, detail=exc.detail())
        except ValueError as exc:
            # Self-naming render refusals (e.g. continueFinalMessage without a
            # final assistant turn, or a chat template that transforms the
            # continued content) — a 400, never a raw 500.
            raise HTTPException(status_code=400, detail=str(exc))
        return dto.GenerateResponse(output=text, promptTokens=rendered.prompt_token_count,
                                    modelID=model_id)

    @router.post("/api/generate/stream")
    def generate_stream(body: dto.GenerateRequest):
        state.require_model()
        cells = _resolve_injections(body.injections)
        from ..experiment.generate import stream_generate_messages

        if body.continueFinalMessage and not body.messages:
            raise HTTPException(
                status_code=400,
                detail="continueFinalMessage requires a messages transcript "
                       "ending in the assistant prefix to continue")

        def produce(stop):
            with state.acquire_active() as model:
                if body.messages:
                    yield from stream_generate_messages(
                        model, body.messages, max_tokens=body.maxTokens,
                        temperature=body.temperature, injections=cells,
                        prompt_mode=body.promptMode, system_prompt=body.systemPrompt,
                        qwen_thinking_enabled=body.qwenThinkingEnabled,
                        continue_final_message=body.continueFinalMessage,
                        should_stop=stop)
                else:
                    yield from stream_generate(
                        model, body.text, max_tokens=body.maxTokens,
                        temperature=body.temperature, injections=cells,
                        prompt_mode=body.promptMode, system_prompt=body.systemPrompt,
                        qwen_thinking_enabled=body.qwenThinkingEnabled,
                        should_stop=stop)

        return StreamingResponse(_locked_sse(produce), media_type="text/event-stream")

    @router.post("/api/extract")
    def extract_route(body: dto.ExtractRequest):
        state.require_model()
        # `outputName` becomes the artifact name AND the run-directory slug
        # (`api-extract-<name>`); refuse a path component synchronously, so a
        # traversal attempt is a 400 rather than a job that dies later.
        if body.outputName:
            _safe_name(str(body.outputName))
        from ..steering.extractor import ExtractionOptions, extract
        from ..steering.reading_position import from_label
        from ..steering.stimulus_set import StimulusSet, load_texts
        from ..steering.vector_math import ExtractionMethod
        from ..steering.vector_store import SteeringVectorSidecar, save

        def work(job):
            stimuli = StimulusSet.from_directory(body.conceptDirectory)
            neutral = load_texts(body.neutralCorpusPath).texts if body.neutralCorpusPath else None
            options = ExtractionOptions(method=ExtractionMethod(body.method),
                                        reading_position=from_label(body.readingPosition))
            with state.acquire_active() as model:
                result = extract(model, stimuli, options, neutral_texts=neutral)
                name = body.outputName or stimuli.name
                run_dir = paths.make_unique_run_directory(f"api-extract-{name}")
                from ..experiment.run_config import write_run_config
                write_run_config(run_dir, "extract", model_id=model.model_id,
                                 revision=model.revision)
                sidecar = SteeringVectorSidecar.make(
                    model_id=model.model_id, revision=model.revision, concept=name,
                    stimulus_set_hash=stimuli.hash, vectors=result.vectors,
                    extraction_method=body.method, reading_position=options.reading_position,
                    residual_norm_per_layer=result.residual_norm_per_layer,
                    residual_norm_source=result.residual_norm_source,
                    residual_norm_convention=result.residual_norm_convention)
                save(result.vectors, sidecar, run_dir, name,
                     neutral_mean_per_layer=result.neutral_mean_per_layer)
            job.log(f"extracted {result.vectors.layer_count} layers → {run_dir}")
            return {"runDirectory": run_dir, "name": name}

        job = state.jobs.submit("extract", work)
        return {"jobId": job.id}

    @router.get("/api/experiment/{name}/sweep/awaiting")
    def sweep_awaiting(name: str):
        """Deferred (Claude-judged) sweep runs awaiting Mac-side judgment:
        packet counts, pinned judges, the rubric text, and the hash pins the
        client echoes back through complete-judgment."""
        _safe_name(name)
        return {"awaiting": tasks.list_awaiting_judgment(name)}

    @router.post("/api/experiment/{name}/sweep/complete-judgment")
    def sweep_complete_judgment(name: str, body: dict):
        """Phase 2 of a two-phase judgeScore sweep (key-custody design
        2026-07-18): accept the Mac's judgments over the sweep's blinded
        packets, verify every pin (packets hash, judge names, coverage,
        experiment epoch), replay the selection, and append the
        recommendation to a draft manifest. CPU-only — runs on the
        controller."""
        _safe_name(name)
        sweep_run = str(body.get("sweepRun") or "")
        judgments = body.get("judgments")
        if not sweep_run or not isinstance(judgments, list) or not judgments:
            raise HTTPException(
                status_code=400,
                detail="body needs sweepRun + a non-empty judgments list "
                       "([{packetID, judge, winner: A|B|tie}, …])")
        try:
            run_dir = tasks.complete_sweep_judgment(name, sweep_run, judgments)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"ok": True, "runDirectory": run_dir}

    @router.get("/api/experiment/{name}/pipelines")
    def pipeline_runs(name: str):
        """Pipeline (chain-runner) runs for this experiment, newest first —
        the app's awaiting/aborted affordance (stage 5): per-stage status,
        disposition, the abort record with gate details, and the
        promoted-agent pins."""
        _safe_name(name)
        return {"pipelines": tasks.list_pipeline_runs(name)}

    @router.get("/api/pipelines")
    def all_pipeline_runs():
        """Every experiment's pipeline runs, newest first (2026-08-06) —
        the Compute panel's orphan/awaiting-import surface: rows carry
        ``experiment``, per-stage status, disposition, and the ``parked``
        stamp when the startup reconcile parked a dead chain. Read-only,
        same tolerance as the per-experiment route."""
        return {"pipelines": tasks.list_pipeline_runs(None)}

    @router.get("/api/experiment/{name}/evaluate/awaiting")
    def evaluate_awaiting(name: str):
        """Deferred evaluate runs awaiting Mac-side judgment (seamless
        pipeline stage 2, 2026-07-19): same shape as sweep/awaiting plus
        the source run and the optional structured-prompt pin."""
        _safe_name(name)
        return {"awaiting": tasks.list_awaiting_evaluate_judgment(name)}

    @router.post("/api/experiment/{name}/evaluate/complete-judgment")
    def evaluate_complete_judgment(name: str, body: dict):
        """Phase 2 of a deferred evaluate: accept the Mac's judgments over
        the run's blinded packets, verify every pin (packets hash, judge
        panel, coverage, experiment epoch), and aggregate the same
        judgments.jsonl + judge-report.json the inline path writes.
        CPU-only — runs on the controller."""
        _safe_name(name)
        evaluate_run = str(body.get("evaluateRun") or "")
        judgments = body.get("judgments")
        if not evaluate_run or not isinstance(judgments, list) or not judgments:
            raise HTTPException(
                status_code=400,
                detail="body needs evaluateRun + a non-empty judgments list "
                       "([{packetID, judge, winner: A|B|tie, model}, …])")
        try:
            run_dir = tasks.complete_evaluate_judgment(
                name, evaluate_run, judgments,
                instructions_sha256=body.get("instructionsSha256"))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"ok": True, "runDirectory": run_dir}

    @router.post("/api/experiment/{name}/promote")
    def experiment_promote(name: str, body: dict):
        """Mint an agent (variant artifact) from a sweep-selected cell —
        synchronous and CPU-only (no model, no job). Registered BEFORE the
        generic verb route so it is not swallowed by ``{verb}``."""
        from ..experiment import promote as promote_mod
        _safe_name(name)
        concept = str(body.get("concept") or "").strip()
        if not concept:
            raise HTTPException(status_code=400, detail="concept is required")
        cell = None
        raw_cell = body.get("cell")
        if raw_cell is not None:
            try:
                cell = (int(raw_cell["layer"]), float(raw_cell["alpha"]))
            except (TypeError, KeyError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail="cell must be {\"layer\": int, \"alpha\": number}")
        # The PINNED promotion contract (B2). When `pins.sweepRun` is
        # present, nothing is resolved by recency: the named sweep run is the
        # only evidence, the manifest epoch is checked, a pinned cell must
        # agree with the sweep's, a pinned artifact is selected by identity,
        # and a retry of the same request returns the existing agent.
        pins = None
        raw_pins = body.get("pins")
        if raw_pins is not None:
            if not isinstance(raw_pins, dict) or not raw_pins.get("sweepRun"):
                raise HTTPException(
                    status_code=400,
                    detail="pins must be an object with a sweepRun")
            pinned_cell = raw_pins.get("winningCell")
            if pinned_cell is not None:
                try:
                    pinned_cell = (int(pinned_cell["layer"]),
                                   float(pinned_cell["alpha"]))
                except (TypeError, KeyError, ValueError):
                    raise HTTPException(
                        status_code=400,
                        detail="pins.winningCell must be "
                               "{\"layer\": int, \"alpha\": number}")
            pins = promote_mod.PromotionPins(
                sweep_run=str(raw_pins["sweepRun"]),
                experiment_hash=raw_pins.get("experimentHash"),
                winning_cell=pinned_cell,
                vector_artifact_id=raw_pins.get("vectorArtifactID"),
                vector_artifact_hash=raw_pins.get("vectorArtifactHash"))
        try:
            return promote_mod.promote(
                name, concept, agent_name=body.get("agentName"), cell=cell,
                override_reason=body.get("overrideReason"),
                sweep_run=body.get("sweepRun"), pins=pins,
                # Optional CITED evidence (SAE proposal r2 §6): a workspace
                # path to a sae-feature-qualification.json. Absent = the
                # promotion behaves exactly as before.
                qualification=body.get("qualification"))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no experiment {name!r}")
        except promote_mod.PromoteError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/api/experiment/{name}/confirm")
    def experiment_confirm(name: str, body: dict):
        """Attach a confirmation-study perturbation policy to a DRAFT
        manifest — synchronous and CPU-only (no model, no job). Registered
        BEFORE the generic verb route so it is not swallowed by ``{verb}``."""
        from ..experiment import confirmation
        _safe_name(name)
        agent = str(body.get("agent") or "").strip()
        if not agent:
            raise HTTPException(status_code=400, detail="agent is required")
        raw_deltas = body.get("deltas")
        if raw_deltas is None:
            deltas = (0.2,)
        else:
            try:
                deltas = tuple(float(x) for x in raw_deltas)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400, detail="deltas must be a list of numbers")
        try:
            return confirmation.attach_perturbations(
                name, agent, deltas=deltas,
                include_control=bool(body.get("includeControl", True)))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no experiment {name!r}")
        except confirmation.ConfirmationError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/api/experiment/{name}/{verb}")
    def experiment_job(name: str, verb: str, body: dict | None = None):
        _safe_name(name)
        if verb not in {"extract", "validate", "sweep", "run", "evaluate",
                        "analyze", "pipeline"}:
            raise HTTPException(status_code=400, detail=f"unknown experiment verb {verb!r}")
        fn = getattr(tasks, verb)
        # Epoch-guard escape hatch for evaluate/analyze (cross-engine body key
        # "allowUnverifiedEpoch"): accept a legacy source run with no
        # experiment-hash stamp; the output is then stamped epochUnverified.
        allow_unverified_epoch = bool((body or {}).get("allowUnverifiedEpoch"))
        # Targeted retry (2026-07-24, cross-engine body key "resumeFrom"):
        # finish a FAILED evaluation by judging only its undecided cells,
        # reusing the verdicts it already produced. Every pin of the
        # partial run is verified before a single row is reused.
        resume_from = (body or {}).get("resumeFrom")
        if resume_from is not None:
            resume_from = str(resume_from).strip() or None
            if resume_from and verb != "evaluate":
                raise HTTPException(
                    status_code=400,
                    detail="resumeFrom applies to the 'evaluate' verb only")

        def work(job):
            job.log(f"experiment {name}: {verb} starting")
            should_cancel = lambda: job.cancelled  # noqa: E731 - polled in task loops
            # Every verb runs the pinned model through the registry (one slot,
            # one lock authority), instead of loading a second private copy.
            if verb == "evaluate":
                # The registry's resident-model capacity travels with
                # evaluate (as with sweep) so a local judge naming a second
                # model refuses at start on a one-slot server.
                run_dir = tasks.evaluate(
                    name, model_provider=state.acquire_model, log=job.log,
                    allow_unverified_epoch=allow_unverified_epoch,
                    max_loaded=state.registry.max_loaded,
                    resume_from=resume_from)
            elif verb == "analyze":
                # Pure-CPU statistics/reporting pass — no model acquisition.
                run_dir = tasks.analyze(
                    name, log=job.log,
                    allow_unverified_epoch=allow_unverified_epoch)
            elif verb == "run":
                run_dir = tasks.run(name, model_provider=state.acquire_model,
                                    should_cancel=should_cancel, log=job.log)
            elif verb == "sweep":
                # The registry's resident-model capacity travels with the
                # sweep so a judgeScore panel needing a SECOND model refuses
                # at start on a one-slot server, never mid-grid.
                run_dir = tasks.sweep(name, model_provider=state.acquire_model,
                                      max_loaded=state.registry.max_loaded,
                                      should_cancel=should_cancel, log=job.log)
            elif verb == "pipeline":
                # The chain runner: declared stages, one model acquisition,
                # gate-aborted between stages (pipeline-abort.json). In-
                # process jobs have no Slurm requeue; cancel + resubmit
                # resumes via the bundle path instead.
                run_dir = tasks.pipeline(
                    name, model_provider=state.acquire_model,
                    should_cancel=should_cancel, log=job.log)
            else:
                run_dir = fn(name, model_provider=state.acquire_model,
                             should_cancel=should_cancel, log=job.log)
            job.log(f"experiment {name}: {verb} → {run_dir}")
            # "experiment" rides along so a cancelledResumable job's Resume
            # can re-dispatch the verb against the parked directory without
            # parsing the experiment name back out of the run id.
            return {"runDirectory": run_dir, "experiment": name}

        # Delegation contract (2026-07-21, closing the live validate failure):
        # controller + live session → a durable DELEGATING job (jobId + job-log
        # contract unchanged; the WORKER writes the run into the shared
        # workspace); gpu-session worker → synchronous (no job subsystem);
        # controller + no session → the actionable refusal NOW, not a queued
        # job dying on the role gate; workstation → the durable in-process
        # job, exactly as before.
        if verb in _DELEGATED_EXPERIMENT_VERBS or (
                verb == "evaluate" and server_role() == "gpu-session"):
            # The second clause receives the controller's delegated evaluate
            # re-POST on the worker (synchronous branch) — without it the
            # worker's jobs property would answer 409 "no job subsystem".
            return _run_or_submit(state, f"experiment:{verb}", work,
                                  path=f"/api/experiment/{name}/{verb}",
                                  body=body or {})
        if verb == "evaluate" and server_role() == "controller":
            # Evaluate needs the session's GPU only when the pinned panel
            # includes a LOCAL judge. Claude/OpenRouter panels are CPU+network
            # work that stays a plain controller job (compute nodes may have
            # no outbound HTTP), and a judge-less/CPU evaluate must keep
            # working on a controller with no session at all — so the wiring
            # is conditional, unlike extract/validate.
            try:
                needs_local_judge = any(
                    j.kind == "local" for j in Manifest.load(name).judges)
            except Exception:  # noqa: BLE001 - missing/bad manifest: fall
                # through to the plain controller job DELIBERATELY — its own
                # failure (which experiment/file is broken) is more
                # informative than a routing refusal would be here.
                needs_local_judge = False
            if needs_local_judge:
                if _delegation_target() is not None:
                    return _run_or_submit(state, f"experiment:{verb}", work,
                                          path=f"/api/experiment/{name}/{verb}",
                                          body=body or {})
                # Local judge + controller + NO dialable session (external
                # review 2026-07-22): the plain job would queue, then die
                # minutes later at acquire_model on the role gate. Refuse
                # NOW with the session-flow remedy instead — a no-op when
                # STEERLAB_ALLOW_RESIDENT_MODELS opts the controller in, in
                # which case the in-process job below is legitimate.
                _refuse_controller_resident_load()
        if verb in _BATCH_ONLY_ON_CONTROLLER \
                and _controller_blocks_resident_models():
            _refuse_batch_only_verb(verb)

        job = state.jobs.submit(f"experiment:{verb}", work)
        return {"jobId": job.id}

    # --- catalog (read-only discovery for the UI) ---------------------------

    def _info_payload():
        # REALPATH (symlinks resolved) so the Mac app can compare this root
        # against its own workspace path reliably for substrate pairing.
        # rootLooksLikeSourceCheckout mirrors the serve-time warning: True
        # when the artifact root appears to be the code checkout rather than
        # a data workspace, so clients can badge the pairing state instead of
        # discovering it via a run refusal.
        # `engineVersion` is the SAME string every run config, frozen
        # manifest, and promotion this deployment stamps
        # (`build_identity.engine_version`): `steerlab-server <version>+<sha8>`,
        # or the bare version where no build identity is resolvable. Site
        # identity used to be spread across four surfaces with the build
        # identity on none of them, so a client could see which root a server
        # served but not which CODE was serving it — and `site qualify`'s
        # buildIdentity check answers that question only for whoever is
        # logged into the node.
        # `controllerChain` is this daemon's answer to "can I resubmit
        # myself?" (open-issues §1 field report, 2026-08-20). The walltime
        # self-chain is shell code in the controller-job TEMPLATE, and the
        # RENDERED copy an operator sbatches is not refreshed by a deploy — so
        # a controller can run the fix's code under a launching script that
        # predates it, which is what cost job 47564632 its successor. State
        # `notApplicable` everywhere but a Slurm daemon-in-a-job controller.
        from .. import controller_render
        from .profile import ServerProfile as _Profile
        try:
            chain = controller_render.boot_chain_status(_Profile.from_env())
        except Exception:      # noqa: BLE001 - info must never 500
            chain = {"state": "unknown", "marker": None,
                     "templateSha256": None, "warnings": []}
        return {"service": "steerlab-server",
                "engineVersion": build_identity.engine_version(),
                "root": os.path.realpath(paths.project_root()),
                "rootLooksLikeSourceCheckout": paths.looks_like_source_checkout(),
                "controllerChain": chain,
                "device": (str(state.model.device) if state.model else None),
                "devices": model_loader.available_devices(),
                "loadedModels": state.registry.snapshots(),
                "capabilities": capability_snapshot(state.registry)}

    @router.get("/api/info")
    def info():
        return _info_payload()

    # --- runtime workspace switching -----------------------------------------

    @router.get("/api/workspace")
    def workspace():
        """The serving root plus the live switch policy, so clients can gate
        their affordances on THIS deployment's verdict instead of trying and
        decoding a refusal."""
        policy = workspace_switch_policy()
        return {"root": os.path.realpath(paths.project_root()),
                "rootLooksLikeSourceCheckout": paths.looks_like_source_checkout(),
                "switchable": policy["switchable"],
                "parent": policy["parent"],
                "reason": policy["reason"]}

    @router.post("/api/workspace/switch")
    def workspace_switch(body: dict):
        """Repoint the serving root at a different workspace without a server
        restart (privileged — listed in ``_PRIVILEGED_PREFIXES``).

        Root flow: every path helper (``paths.project_root``), profile
        (``ServerProfile.from_env``), and resolver (``state.resolver``) reads
        ``STEERLAB_ROOT`` per call, so flipping the process env IS the switch;
        nothing snapshots the root across requests except the pieces that are
        deliberately instance-scoped (below).

        Deliberately NOT moved: the metadata root (jobs DB, housekeeping
        snapshots, maintenance calendar). Job history is a record of what THIS
        server instance ran — jobs from the previous workspace must survive
        the switch, and the DurableJobStore latched its SQLite path at process
        start anyway. When ``STEERLAB_METADATA_ROOT`` is unset (its default
        derives from the root), it is pinned to the pre-switch value first so
        every metadata consumer keeps pointing at the same tree the jobs DB
        lives in, instead of the metadata splitting across workspaces.

        Refused while any non-terminal job exists: in-flight work resolved its
        inputs against the old root, and relative artifact paths written into
        results must not silently re-resolve against the new one mid-run.
        (This also covers checkpointed Slurm jobs — non-terminal — so the
        reconciler's cached executor never builds a resubmission under a root
        the job was not submitted with.)

        That refusal is only as good as its atomicity, which is what
        ``workspace_lock`` supplies: the active-job scan and the mutation run
        inside ONE exclusive critical section, and every submission path holds
        the shared side while it resolves roots and registers. Without it the
        scan and the mutation were two steps with a window between them, and a
        submission landing in that window was neither refused nor contained —
        its early paths resolved against the old root and its later ones
        against the new (review finding, 2026-08-21).
        """
        raw = str(body.get("root") or "").strip()
        if not raw:
            raise HTTPException(status_code=400, detail="root is required")
        if "\0" in raw:
            raise HTTPException(status_code=400, detail="invalid path")
        if not os.path.isabs(raw):
            raise HTTPException(
                status_code=400,
                detail="root must be an absolute path on the server")
        target = os.path.realpath(raw)

        policy = workspace_switch_policy()
        if policy["parent"] is not None:
            if not is_contained(target, policy["parent"]):
                raise HTTPException(
                    status_code=403,
                    detail=(f"workspace root must be under the configured "
                            f"parent {policy['parent']} "
                            "(STEERLAB_WORKSPACE_PARENT)"))
        elif not policy["switchable"]:
            raise HTTPException(status_code=403, detail=policy["reason"])

        from .jobs import TERMINAL
        # EXCLUSIVE for the scan AND the mutation. No submission can register
        # a job (or resolve a submission root) between the two.
        with workspace_lock.switching():
            manager = state.job_manager_or_none
            active = ([j for j in manager.list() if j.status not in TERMINAL]
                      if manager is not None else [])
            if active:
                names = ", ".join(f"{j.id} ({j.kind}: {j.status})" for j in active[:8])
                more = f" and {len(active) - 8} more" if len(active) > 8 else ""
                raise HTTPException(
                    status_code=409,
                    detail=(f"cannot switch workspace while {len(active)} job(s) "
                            f"are not terminal: {names}{more} — wait or cancel "
                            "them first"))

            if os.path.exists(target):
                # An existing directory is served AS-IS — never seeded. Writing
                # the WORKSPACE.md marker into an arbitrary existing tree would,
                # among other things, silently reclassify a source checkout as a
                # data workspace and defeat the pairing warning.
                if not os.path.isdir(target):
                    raise HTTPException(
                        status_code=400, detail=f"{target} is not a directory")
            elif os.path.isdir(os.path.dirname(target)):
                # Parent exists: create and seed a fresh workspace skeleton, the
                # same shape the Mac app seeds (prompts/experiments/runs +
                # WORKSPACE.md marker).
                paths.seed_workspace(target)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=(f"neither {target} nor its parent directory "
                            "exists on the server"))

            # Pin the metadata tree BEFORE the root moves (see docstring).
            os.environ.setdefault(
                "STEERLAB_METADATA_ROOT", ServerProfile.from_env().metadata_root)
            os.environ["STEERLAB_ROOT"] = target
            # ServiceState snapshotted a profile at construction; refresh it so
            # nothing ever reads a stale root through the snapshot.
            state.profile = ServerProfile.from_env()
        return {"switched": True, **_info_payload()}

    @router.post("/api/generation-prompt")
    def generation_prompt(body: dict):
        from ..experiment import authoring
        try:
            prompt = authoring.template_prompt(
                body.get("template", "caa"), concept=body.get("concept", ""),
                count=int(body.get("count", 20)), guidance=body.get("guidance", ""),
                neutral_concepts=body.get("neutralConcepts", ""),
                matched_domains=body.get("matchedDomains", ""),
                avoid_settings=body.get("avoidSettings", ""))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"prompt": prompt}

    @router.get("/api/battery/generation-prompt")
    def battery_generation_prompt(count: int = 0, avoid: str = ""):
        """The capability-battery authoring brief — the twin of
        GET /api/concept/{name}/prompt for the other artifact a study pins.

        A GET on purpose: it renders text from the battery contract and the
        linter's thresholds, reads no caller-named path, runs no model, and
        writes nothing, so it needs no place in the mutating-route allowlist.
        """
        from ..experiment import battery_brief
        if count <= 0:
            count = battery_brief.DEFAULT_ITEM_COUNT
        return {"prompt": battery_brief.generation_prompt(count, avoid=avoid),
                "count": count}

    @router.get("/api/vectors")
    def vectors():
        return {"vectors": [asdict(v) | {"id": v.id} for v in catalog.list_vectors()]}

    @router.get("/api/readers")
    def readers():
        """RepE reader artifacts (measurement instruments — separate from
        steering vectors)."""
        return {"readers": [asdict(r) | {"id": r.id} for r in catalog.list_readers()]}

    @router.post("/api/reader/score")
    def reader_score(body: dict):
        """Exact RepE reader inference over the loaded model: render the
        reader's own template around each text, capture the LAT token at the
        reader's layer, score through the stored training normalization."""
        from ..steering import repe_reader
        reader_ref = str(body.get("readerID") or "").strip()
        texts = body.get("texts")
        if not reader_ref:
            raise HTTPException(status_code=400, detail="readerID is required")
        if (not isinstance(texts, list) or not texts
                or not all(isinstance(t, str) for t in texts)):
            raise HTTPException(status_code=400,
                                detail="texts must be a non-empty list of strings")
        state.require_model()
        path = state.resolver.require_file(reader_ref, allow_local_absolute=True)
        try:
            reader = repe_reader.load_reader(path)
        except repe_reader.RepeReaderError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        with state.acquire_active() as model:
            try:
                scores = repe_reader.score_texts(model, reader, list(texts))
            except (repe_reader.RepeReaderError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        return {"readerID": reader_ref, "concept": reader.concept,
                "layer": reader.layer, "templateID": reader.template.id,
                "scores": scores}

    @router.post("/api/reader/fit")
    def reader_fit(body: dict):
        """Fit a faithful RepE reader (template + LAT token + PC1 + persisted
        probe) as a DURABLE JOB, on the model pinned at submission (explicit
        ``modelID``/``revision``, else a synchronous snapshot of the active
        model). Artifacts land in a fresh immutable run directory (discovered
        by GET /api/readers); an uploaded pairs file is validated in memory
        first and only then atomically written to its canonical
        git-versionable home under ``prompts/``, so the pinned bytes exist
        outside the run and a rejected upload leaves nothing behind."""
        from ..steering import repe_reader

        concept = str(body.get("concept") or "").strip()
        if not concept:
            raise HTTPException(status_code=400, detail="concept is required")
        _safe_name(concept)

        # Exactly one template source: a registry id, or one-off raw JSON whose
        # bytes are persisted into the run directory (hash-over-raw-bytes).
        template_id = body.get("templateID")
        template_json = body.get("templateJSON")
        if (template_id is None) == (template_json is None):
            raise HTTPException(
                status_code=400,
                detail="exactly one of templateID/templateJSON is required")
        template = None
        template_bytes: bytes | None = None
        custom_template_file = None
        if template_id is not None:
            template_id = str(template_id).strip()
            _safe_name(template_id)
            try:
                template = repe_reader.load_template(
                    paths.template_path(template_id))
            except repe_reader.RepeReaderError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        else:
            template_bytes = str(template_json).encode("utf-8")
            try:
                parsed = json.loads(template_bytes.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400,
                                    detail=f"invalid templateJSON: {exc}")
            for key in ("id", "text"):
                if not isinstance(parsed, dict) or key not in parsed:
                    raise HTTPException(status_code=400,
                                        detail=f"templateJSON missing {key!r}")
            custom_id = str(parsed["id"]).strip()
            _safe_name(custom_id)
            custom_template_file = f"{custom_id}.json"

        pairs_jsonl = body.get("pairsJSONL")
        pairs_ref = body.get("pairsPath")
        if (pairs_jsonl is None) == (pairs_ref is None):
            raise HTTPException(
                status_code=400,
                detail="exactly one of pairsJSONL/pairsPath is required")

        layers = body.get("layers")
        if layers is not None:
            if (not isinstance(layers, list) or not layers
                    or not all(isinstance(l, int) and not isinstance(l, bool)
                               for l in layers)):
                raise HTTPException(status_code=400,
                                    detail="layers must be a non-empty list of ints")
            layers = list(layers)

        name = str(body.get("outputName") or f"reader-{concept}").strip()
        _safe_name(name)

        # Model pin at SUBMISSION time: an explicit modelID (+ optional
        # revision) wins; otherwise the ACTIVE model's id/revision are
        # snapshotted synchronously here. The queued job then acquires exactly
        # that model — never "whatever is active later", which could silently
        # fit the reader on a different model than the caller saw.
        fit_model_id = str(body.get("modelID") or "").strip() or None
        fit_revision = body.get("revision")
        if fit_revision is not None:
            fit_revision = str(fit_revision)
        if fit_model_id is None:
            if _delegation_target() is not None:
                # Delegated fit: the WORKER's route snapshots ITS active
                # model — this controller holds none by design.
                pass
            else:
                active = state.require_model()
                fit_model_id, fit_revision = active.model_id, active.revision

        # Pairs are fully validated in memory BEFORE any canonical write: a
        # malformed or wrong-concept upload must not leave bad bytes behind at
        # (or clobber) the pinned prompts/readers/<concept>/pairs.jsonl.
        pairs_bytes: bytes | None = None
        if pairs_jsonl is not None:
            pairs_bytes = str(pairs_jsonl).encode("utf-8")
            try:
                dataset = repe_reader.parse_pairs(pairs_bytes,
                                                  source="uploaded pairsJSONL")
            except repe_reader.RepeReaderError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        else:
            # A path reference is contained the same way /api/reader/score
            # contains reader references.
            pairs_file = state.resolver.require_file(str(pairs_ref),
                                                     allow_local_absolute=True)
            try:
                dataset = repe_reader.load_pairs(pairs_file)
            except repe_reader.RepeReaderError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        if dataset.concept != concept:
            raise HTTPException(
                status_code=400,
                detail=f"pairs file is for concept {dataset.concept!r}, "
                       f"not {concept!r}")
        # Template pin: every row must pin the template this fit uses. The fit
        # re-checks, but failing synchronously keeps a mismatched upload out of
        # the canonical location (and returns a 400 instead of a failed job).
        fit_template_id = template.id if template is not None else custom_id
        mismatched = sorted({p.template_id for p in dataset.pairs
                             if p.template_id != fit_template_id})
        if mismatched:
            raise HTTPException(
                status_code=400,
                detail=f"pairs pin template(s) {mismatched}, but this fit uses "
                       f"{fit_template_id!r}")

        # All synchronous validation passed — only now does an uploaded corpus
        # reach its canonical git-versionable home under ``prompts/`` (a fixed
        # path; ``concept`` is safe-named, so no caller-controlled traversal).
        # Write-temp + os.replace in the same directory so a pre-existing good
        # pairs.jsonl is replaced atomically, never left half-written.
        if pairs_bytes is not None:
            pairs_file = paths.reader_pairs_path(concept)
            pairs_dir = os.path.dirname(pairs_file)
            os.makedirs(pairs_dir, exist_ok=True)
            fd, tmp_pairs = tempfile.mkstemp(dir=pairs_dir, prefix=".pairs-",
                                             suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(pairs_bytes)
                os.replace(tmp_pairs, pairs_file)
            except OSError:
                try:
                    os.unlink(tmp_pairs)
                except OSError:
                    pass
                raise

        def work(job):
            from ..experiment.run_config import write_run_config
            run_dir = paths.make_unique_run_directory(f"api-reader-fit-{name}")
            write_run_config(run_dir, "reader-fit", model_id=fit_model_id,
                             revision=fit_revision)
            if template_bytes is not None:
                custom_path = os.path.join(run_dir, custom_template_file)
                with open(custom_path, "wb") as handle:
                    handle.write(template_bytes)
                fit_template = repe_reader.load_template(custom_path)
            else:
                fit_template = template
            job.log(f"fitting reader '{concept}' via template "
                    f"'{fit_template.id}' ({len(dataset.train)} train / "
                    f"{len(dataset.held_out)} held-out pairs) on "
                    f"{fit_model_id}"
                    + (f"@{fit_revision}" if fit_revision else ""))
            with state.acquire_model(fit_model_id, fit_revision) as model:
                artifacts = repe_reader.fit(model, dataset, fit_template,
                                            layers=layers)
                artifact_paths = repe_reader.save_readers(artifacts, run_dir)
            job.log(f"fitted {len(artifacts)} layer reader(s) → {run_dir}")
            return {"runDirectory": run_dir,
                    "artifacts": artifact_paths,
                    "layerScores": [{"layer": a.layer,
                                     "trainAccuracy": a.train_accuracy,
                                     "heldOutAccuracy": a.held_out_accuracy,
                                     "pc1ExplainedVariance": a.pc1_explained_variance}
                                    for a in artifacts]}

        return _run_or_submit(state, "reader:fit", work,
                               path="/api/reader/fit", body=body)

    # --- residual-norm backfill ---------------------------------------------

    @router.post("/api/vectors/backfill-norms")
    def vectors_backfill_norms(body: dict):
        """Backfill ``residualNormPerLayer`` onto a vector artifact that lacks
        it (legacy pre-capture vectors, Gemma Scope SAE imports, reader-derived
        steering vectors) as a DURABLE JOB, on the model pinned at submission
        (explicit ``modelID``/``revision``, else a synchronous snapshot of the
        active model — same pattern as /api/reader/fit). Norms are measured on
        a pinned neutral corpus with the exact extraction-time measurement path
        and land in a NEW artifact in a fresh run directory — runs are
        immutable, the original is never mutated."""
        from ..steering import norm_backfill

        vector_ref = str(body.get("vectorID") or "").strip()
        if not vector_ref:
            raise HTTPException(status_code=400, detail="vectorID is required")
        corpus_ref = str(body.get("neutralCorpusPath") or "").strip()
        if not corpus_ref:
            raise HTTPException(status_code=400, detail="neutralCorpusPath is required")

        # vectorID is "<runDirectory>/<name>" (the catalog id). Contain the
        # directory exactly as other vector reads do; the name is a plain
        # path component.
        vector_dir = state.resolver.require_dir(os.path.dirname(vector_ref),
                                                allow_local_absolute=True)
        name = os.path.basename(vector_ref)
        _safe_name(name)
        sidecar_path = os.path.join(vector_dir, f"{name}.json")
        vectors_path = os.path.join(vector_dir, f"{name}.safetensors")
        if not (os.path.isfile(sidecar_path) and os.path.isfile(vectors_path)):
            raise HTTPException(status_code=400,
                                detail=f"no vector artifact at {vector_ref!r}")
        try:
            with open(sidecar_path, encoding="utf-8") as handle:
                sidecar = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"unreadable sidecar: {exc}")
        if "modelID" not in sidecar or "layerCount" not in sidecar:
            raise HTTPException(status_code=400,
                                detail=f"{vector_ref!r} is not a steering-vector artifact")
        redenominate = bool(body.get("redenominate", False))
        if sidecar.get("residualNormPerLayer") and not redenominate:
            raise HTTPException(
                status_code=400,
                detail="artifact already has residual norms (source: "
                       f"{sidecar.get('residualNormSource') or 'unrecorded'}) — "
                       "backfill never overwrites; pass redenominate: true to "
                       "write a NEW neutral-corpus-denominated artifact")

        corpus_file = state.resolver.require_file(corpus_ref, allow_local_absolute=True)

        output_name = str(body.get("outputName") or name).strip()
        _safe_name(output_name)

        # Model pin at SUBMISSION time (see /api/reader/fit): an explicit
        # modelID (+ optional revision) wins; otherwise the ACTIVE model is
        # snapshotted synchronously, so the queued job never measures on
        # "whatever is active later".
        bf_model_id = str(body.get("modelID") or "").strip() or None
        bf_revision = body.get("revision")
        if bf_revision is not None:
            bf_revision = str(bf_revision)
        if bf_model_id is None:
            if _delegation_target() is not None:
                # Delegated backfill: the WORKER's route snapshots ITS active
                # model — this controller holds none by design (live
                # 2026-08-06: the unconditional snapshot 409'd the whole
                # route on a controller even with an idle session serving).
                pass
            else:
                active = state.require_model()
                bf_model_id, bf_revision = active.model_id, active.revision
        # A norm table is a per-model measurement — reject a wrong-model
        # submission synchronously when the pin is known here (the delegated
        # worker re-runs this same check against its own snapshot;
        # backfill_norms re-checks against the model it actually acquired —
        # that guard stays authoritative).
        if bf_model_id is not None and str(sidecar["modelID"]) != bf_model_id:
            raise HTTPException(
                status_code=400,
                detail=f"vectors were extracted on model {sidecar['modelID']!r}, "
                       f"not {bf_model_id!r} — a residual-norm table is a "
                       "per-model measurement")

        def work(job):
            from ..experiment.run_config import write_run_config
            run_dir = paths.make_unique_run_directory(f"api-backfill-norms-{output_name}")
            write_run_config(run_dir, "norm-backfill", model_id=bf_model_id,
                             revision=bf_revision)
            job.log(f"backfilling residual norms for {vector_ref} on "
                    f"{bf_model_id}" + (f"@{bf_revision}" if bf_revision else "")
                    + f" (corpus {corpus_ref})")
            with state.acquire_model(bf_model_id, bf_revision) as model:
                result = norm_backfill.backfill_norms(
                    model, vector_dir, name, corpus_file, run_dir,
                    output_name=output_name, redenominate=redenominate)
            job.log(f"measured {result.layer_count}-layer residual norms "
                    f"({result.residual_norm_source}) → {run_dir}")
            return {"runDirectory": run_dir,
                    "artifact": result.artifact_id,
                    "residualNormSource": result.residual_norm_source,
                    "layerCount": result.layer_count}

        # The _run_or_submit seam (2026-08-06 field failure): the plain
        # jobs.submit here sat in NEITHER cluster dispatch class — not
        # proxied like interactive verbs, not delegated like authoring jobs —
        # so on a controller the backfill 409'd even with an idle GPU session
        # serving, and the "backfill norms" remedy every norm-units refusal
        # names was a dead end on exactly the deployment that runs studies.
        return _run_or_submit(state, "vector:backfill-norms", work,
                              path="/api/vectors/backfill-norms", body=body)

    @router.post("/api/models/install")
    def install_model(body: dict):
        """Prefetch a HF repo into this substrate's cache as a DURABLE JOB —
        a 30 GB download must not be a silent side effect of a blocking load
        call. Auth-gated on non-local profiles (privileged prefix)."""
        import re as _re
        model_id = str(body.get("modelID", "")).strip()
        revision = body.get("revision")
        if not _re.match(r"^[\w.\-]+/[\w.\-]+$", model_id):
            raise HTTPException(status_code=400, detail=f"invalid model id {model_id!r}")
        if model_loader._is_mlx_repo(model_id):
            raise HTTPException(
                status_code=400,
                detail=f"{model_id!r} is an MLX-quantized repo — this engine "
                       "loads full-precision HF repos; install the family twin "
                       "instead (e.g. Qwen/Qwen3-4B, google/gemma-3-4b-it)")

        def work(job):
            # Downloads run in an ONLINE child process even when this server
            # is HF_HUB_OFFLINE=1 (cluster hermeticity applies to runs, not to
            # the install verb) — see model_install.py for the policy split
            # and the failure→remedy mapping.
            from . import model_install
            job.log(f"installing {model_id}"
                    + (f" @ {revision}" if revision else "") + " into the HF cache")
            local_path = model_install.run_install(
                model_id, revision, job.log, cancelled=lambda: job.cancelled)
            # The size list feeds the client's memory-fit gating: a fresh
            # install must be gated immediately, not after the TTL
            # (engineer review 2026-07-18).
            model_loader.invalidate_model_size_cache()
            return {"modelID": model_id, "path": local_path}

        job = state.jobs.submit("model:install", work)
        return {"jobId": job.id}

    # --- J-lens reading instruments (server-only, Gemma-only) ----------------
    # Acquisition and import are separate verbs because they have different
    # prerequisites and different failure modes: acquire needs egress and puts
    # bytes in the machine's HF cache; import is offline and writes the
    # workspace lens store. Both are DURABLE JOBS — at 27B the fetch is 3.53 GB
    # and the conversion reads all of it, so neither belongs in a request.

    @router.get("/api/jlens/lenses")
    def jlens_lenses():
        """Imported lenses. Read-only and open, like the vector catalog."""
        from ..jlens import lens_store

        return {"lenses": [r.to_dict() for r in lens_store.list_lenses()],
                "supported": _jlens_supported()}

    @router.get("/api/jlens/lenses/{lens_id}")
    def jlens_lens_detail(lens_id: str):
        from ..jlens import lens_store
        from ..jlens.schemas import JLensError

        try:
            return lens_store.resolve(lens_id).to_dict()
        except JLensError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @router.post("/api/jlens/lenses/acquire")
    def jlens_acquire(body: dict):
        """Fetch one model's lens into the HF cache (privileged: goes online).

        Scoped by construction — the repository holds 36 models totalling
        ~57 GB and one lens folder is a few GB, so the verb never accepts a
        caller-supplied pattern.
        """
        from ..jlens import acquire as acquire_mod
        from ..jlens.schemas import JLensError

        model_id = str(body.get("modelID", "")).strip()
        try:
            acquire_mod.patterns_for(model_id)      # validates against the table
        except JLensError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        def work(job):
            snapshot = acquire_mod.acquire(
                model_id, log=job.log, revision=body.get("revision"),
                cancelled=lambda: job.cancelled)
            return {"modelID": model_id, "snapshot": snapshot}

        return {"jobId": state.jobs.submit("jlens-acquire", work).id}

    @router.post("/api/jlens/lenses/import")
    def jlens_import(body: dict):
        """Convert a cached lens into the workspace store (privileged: writes).

        Offline by construction: import must not silently pull a different
        snapshot than the one acquisition pinned.
        """
        from ..jlens import importer
        from ..jlens.schemas import JLensError

        model_id = str(body.get("modelID", "")).strip()
        try:
            importer.entry_for(model_id)
        except JLensError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        def work(job):
            job.log(f"importing the cached lens for {model_id} "
                    f"(converting once to per-layer safetensors)")
            record = importer.import_lens(model_id)
            job.log(f"lens {record.lensID}: layers {record.sourceLayers[0]}.."
                    f"{record.sourceLayers[-1]}, target {record.targetLayer}, "
                    f"converted {record.converted.layerCount} layers as "
                    f"{record.converted.dtype}")
            return {"lensID": record.lensID,
                    "sourceLayers": record.sourceLayers,
                    "targetLayer": record.targetLayer,
                    "dModel": record.dModel,
                    "converted": record.converted.path if record.converted else None}

        return {"jobId": state.jobs.submit("jlens-import", work).id}

    @router.post("/api/jlens/token-options")
    def jlens_token_options(body: dict):
        """Exact token candidates for a typed string — never a chosen one.

        Read-only and cheap (tokenizer only, no model), so it is not privileged.
        """
        from ..jlens import token_options
        from ..jlens.schemas import JLensError

        model_id = str(body.get("modelID", "")).strip()
        text = body.get("text")
        if not model_id:
            raise HTTPException(status_code=400, detail="modelID is required")
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            return token_options.options_for(
                tokenizer, text,
                include_case_variants=bool(body.get("includeCaseVariants")),
                max_matches=int(body.get("maxMatches",
                                         token_options.DEFAULT_MAX_MATCHES)))
        except JLensError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except OSError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"could not load a tokenizer for {model_id!r}: {exc}")

    @router.post("/api/jlens/decode-tokens")
    def jlens_decode_tokens(body: dict):
        """Token IDs → their pieces, for rendering a trace.

        A trace stores IDs and only IDs — that is the durable identity, and
        resolving them at write time would bake one tokenizer's answer into the
        record. But a table of bare integers cannot be read, so the viewer
        decodes on demand. Tokenizer only, no model, no privilege.
        """
        model_id = str(body.get("modelID", "")).strip()
        ids = body.get("tokenIDs") or []
        if not model_id:
            raise HTTPException(status_code=400, detail="modelID is required")
        if not isinstance(ids, list) or len(ids) > 4096:
            raise HTTPException(
                status_code=400,
                detail="tokenIDs must be a list of at most 4096 ids")
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except OSError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"could not load a tokenizer for {model_id!r}: {exc}")
        out = {}
        for raw in ids:
            try:
                token_id = int(raw)
            except (TypeError, ValueError):
                continue
            try:
                out[str(token_id)] = tokenizer.decode([token_id])
            except Exception:  # noqa: BLE001 - an undecodable id is reported absent
                continue
        return {"modelID": model_id, "pieces": out}

    @router.post("/api/jlens/directions/derive")
    def jlens_derive(body: dict):
        """Derive a model-depth direction for one exact token (privileged).

        A durable job, though a cheap one: derivation reads ~10 KB from the
        cached snapshot rather than loading the model, so this needs no GPU and
        no resident weights. It is a job for provenance and cancellation, not
        for duration.
        """
        from ..jlens import derive as derive_mod
        from ..jlens.schemas import JLensError

        lens_id = str(body.get("lensID", "")).strip()
        model_id = str(body.get("modelID", "")).strip()
        token_id = body.get("tokenID")
        if not lens_id or not model_id:
            raise HTTPException(status_code=400,
                                detail="lensID and modelID are required")
        if not isinstance(token_id, int):
            # A word here would have to be resolved to a token, which is
            # exactly the silent mis-selection token-options exists to prevent.
            raise HTTPException(
                status_code=400,
                detail="tokenID must be an integer — resolve a word through "
                       "/api/jlens/token-options and send the selected id")

        def work(job):
            job.log(f"deriving J-lens direction for token {token_id} "
                    f"from {lens_id} on {model_id}")
            result = derive_mod.derive_direction(
                lens_id, int(token_id), model_id=model_id,
                revision=body.get("revision"), name=body.get("name"),
                piece=body.get("piece"), concept=body.get("concept"))
            job.log(f"wrote {result['name']} covering "
                    f"{len(result['definedLayers'])} layers → "
                    f"{result['runDirectory']}")
            return result

        try:
            derive_mod.lens_store.resolve(lens_id)
        except JLensError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"jobId": state.jobs.submit("jlens-derive", work).id}

    @router.post("/api/jlens/qualify")
    def jlens_qualify(body: dict):
        """Stage 4: accept a lens against ONE exact runtime (privileged).

        A durable GPU job — it loads the model, because the checks it runs are
        about the numerics the model actually presents, and geometry alone
        cannot see those. The job's return value is the verdict; a FAILED
        qualification completes normally and stores its failure, because "we
        tested this runtime and it did not pass" is evidence and losing it
        would leave the absence looking like an untested runtime.
        """
        from ..jlens import qualification
        from ..jlens.schemas import JLensError

        lens_id = str(body.get("lensID", "")).strip()
        model_id = str(body.get("modelID", "")).strip()
        if not lens_id or not model_id:
            raise HTTPException(status_code=400,
                                detail="lensID and modelID are required")
        for key in ("layers", "watchlist"):
            value = body.get(key)
            if value is not None and not (
                    isinstance(value, list)
                    and all(isinstance(x, int) for x in value)):
                raise HTTPException(
                    status_code=400,
                    detail=f"{key} must be a list of integers")
        alpha = body.get("alphaRange")
        if alpha is not None and not (
                isinstance(alpha, list)
                and all(isinstance(x, (int, float)) for x in alpha)):
            raise HTTPException(status_code=400,
                                detail="alphaRange must be a list of numbers")

        # Caller-named file inputs are CONTAINED before the job is submitted:
        # both are opened by the qualification path, and a bearer-token holder
        # must not be able to make the server read arbitrary host files
        # (external review round 2; CLAUDE.md security posture — on a shared
        # node assume other local users can reach 127.0.0.1).
        prompts_path = (state.resolver.require_file(str(body["prompts"]))
                        if body.get("prompts") else None)
        battery_path = (state.resolver.require_file(str(body["battery"]))
                        if body.get("battery") else None)

        def work(job):
            job.log(f"qualifying {lens_id} against {model_id}")
            result = qualification.qualify(
                lens_id, model_id, revision=body.get("revision"),
                layers=body.get("layers"), watchlist=body.get("watchlist"),
                alpha_range=tuple(alpha) if alpha else None,
                token_id=body.get("tokenID"),
                prompts_path=prompts_path, battery_path=battery_path,
                device=body.get("device"), dtype=body.get("dtype"),
                log=job.log)
            job.log(f"{result['qualificationID']}: "
                    + ("PASSED" if result["passed"]
                       else f"FAILED ({', '.join(result['blockingFailures'])})")
                    + f" [tier {result['tier']}]")
            return result

        # Resolve BEFORE submitting: a lens that is not imported is a 404 the
        # caller can act on, not a job that fails minutes later on a GPU.
        from ..jlens import lens_store

        try:
            lens_store.resolve(lens_id)
        except JLensError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"jobId": state.jobs.submit("jlens-qualify", work).id}

    @router.post("/api/jlens/g0")
    def jlens_g0(body: dict):
        """The G0 1b feasibility gate (privileged: loads the model, generates).

        Returns the report with the two arms verdicted SEPARATELY. A failed
        arm is an outcome, not an error — a study may steer without readout
        claims, or read without steering — so the job succeeds and the report
        carries the verdicts.
        """
        from ..jlens import g0
        from ..jlens.schemas import JLensError

        model_id = str(body.get("modelID", "")).strip()
        if not model_id:
            raise HTTPException(status_code=400, detail="modelID is required")
        endpoint = body.get("endpoint")
        if endpoint is not None:
            # A caller-named workspace path: contained exactly like every
            # other route that reads one.
            endpoint = state.resolver.require_file(str(endpoint))

        def work(job):
            job.log(f"G0 1b gate on {model_id}")
            report = g0.run(
                model_id, lens_id=body.get("lensID"),
                revision=body.get("revision"), layers=body.get("layers"),
                watchlist=body.get("watchlist"), token_id=body.get("tokenID"),
                piece=body.get("piece"), endpoint_path=endpoint,
                alpha_range=(tuple(body["alphaRange"])
                             if body.get("alphaRange") else None),
                top_k=int(body.get("topK") or 10),
                device=body.get("device"), dtype=body.get("dtype"),
                log=job.log)
            job.log(f"mechanical {report['mechanical']['verdict']}; "
                    f"steering arm {report['arms']['steering']['verdict']}; "
                    f"readout arm {report['arms']['readout']['verdict']}")
            return report

        from ..jlens import importer, lens_store

        try:
            lens_store.resolve(str(body.get("lensID") or "").strip()
                               or importer.lens_id_for(model_id))
        except JLensError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"jobId": state.jobs.submit("jlens-g0", work).id}

    @router.post("/api/jlens/probe")
    def jlens_probe(body: dict):
        """Position-resolved readout over ONE prompt (privileged).

        A durable job: it loads the model. Bounded by construction (one prompt
        × armed layers × positions), so it sits outside the study trace budget
        and refuses on its own projection ceiling rather than wedging a node.
        """
        from ..jlens import importer, lens_store, probe as probe_mod
        from ..jlens.schemas import JLensError

        model_id = str(body.get("modelID", "")).strip()
        prompt = str(body.get("prompt", "")).strip()
        if not model_id or not prompt:
            raise HTTPException(status_code=400,
                                detail="modelID and prompt are required")
        for key in ("layers", "pinIDs"):
            value = body.get(key)
            if value is not None and not (
                    isinstance(value, list)
                    and all(isinstance(x, int) for x in value)):
                raise HTTPException(
                    status_code=400,
                    detail=f"{key} must be a list of integers")
        # Caller-named artifact locators are contained exactly like the
        # support route's vectorID.
        directions = []
        for reference in (body.get("directions") or []):
            state.resolver.require_dir(os.path.dirname(str(reference)),
                                       allow_local_absolute=True)
            directions.append(str(reference))

        def work(job):
            job.log(f"probing {model_id} over {len(prompt)} char(s)")
            report = probe_mod.probe(
                model_id, prompt=prompt, lens_id=body.get("lensID"),
                revision=body.get("revision"), layers=body.get("layers"),
                top_k=int(body.get("topK") or 10),
                pin_words=body.get("pin") or (),
                pin_ids=body.get("pinIDs") or (),
                directions=directions,
                position_stride=int(
                    body.get("stride") or probe_mod.DEFAULT_POSITION_STRIDE),
                max_tokens=body.get("maxTokens"),
                device=body.get("device"), dtype=body.get("dtype"),
                log=job.log)
            job.log(f"{report['claim']} [{report['evidenceTier']}] → "
                    f"{report['runDirectory']}")
            return report

        try:
            lens_store.resolve(str(body.get("lensID") or "").strip()
                               or importer.lens_id_for(model_id))
        except JLensError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"jobId": state.jobs.submit("jlens-probe", work).id}

    @router.post("/api/jlens/report")
    def jlens_report(body: dict):
        """Roll a completed run's J-lens trace up into its report (privileged:
        reads and writes a caller-named run directory).

        Synchronous: it is pure CPU over one JSONL file, so a durable job
        would add ceremony without adding recoverability.
        """
        from ..jlens import report as report_mod
        from ..jlens.schemas import JLensError

        run_dir = state.resolver.require_dir(str(body.get("runDirectory", "")),
                                             allow_local_absolute=True)
        try:
            return report_mod.report(
                run_dir,
                baseline=str(body.get("baseline")
                             or report_mod.BASELINE_CONDITION),
                band=body.get("band"),
                bands=int(body.get("bands") or report_mod.DEFAULT_BANDS),
                token_sets=body.get("tokenSets"))
        except JLensError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/api/jlens/support")
    def jlens_support(body: dict):
        """Read an existing concept vector back as J-lens token atoms (privileged).

        A durable job: it reads a caller-named artifact and the gain-scaled head
        (~2.7 GB at 4B, ~5.6 GB at 27B), so it is gated and contained like every
        other route that touches paths the caller supplies.

        The response is the readout AND a run directory — the readout is evidence
        about a vector, so it is persisted rather than only returned.
        """
        from ..jlens import decompose
        from ..jlens.schemas import JLensError

        lens_id = str(body.get("lensID", "")).strip()
        vector_ref = str(body.get("vectorID", "")).strip()
        if not lens_id or not vector_ref:
            raise HTTPException(status_code=400,
                                detail="lensID and vectorID are required")
        # vectorID is "<runDirectory>/<name>" — the catalog id, contained exactly
        # as the norm-backfill route contains it.
        vector_dir = state.resolver.require_dir(os.path.dirname(vector_ref),
                                                allow_local_absolute=True)
        name = os.path.basename(vector_ref)
        _safe_name(name)

        layers = body.get("layers")
        if layers is not None:
            if not isinstance(layers, list) or not all(isinstance(x, int) for x in layers):
                raise HTTPException(status_code=400,
                                    detail="layers must be a list of integers")
        budget = body.get("budget", decompose.DEFAULT_BUDGET)
        if not isinstance(budget, int) or budget < 1:
            raise HTTPException(status_code=400, detail="budget must be a positive integer")

        def work(job):
            job.log(f"decomposing {name} over {lens_id} at budget {budget}")
            readout = decompose.decompose(
                lens_id=lens_id, vector_directory=vector_dir, vector_name=name,
                layers=layers, budget=budget, progress=job.log)
            run_dir = decompose.write_readout(readout)
            best = max((r["energyOverNull"] for r in readout["layers"]), default=0.0)
            job.log(f"read {len(readout['layers'])} layer(s); best margin over the "
                    f"matched-norm null {best:+.1%} → {run_dir}")
            return {**readout, "runDirectory": run_dir}

        try:
            decompose.lens_store.resolve(lens_id)
        except JLensError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"jobId": state.jobs.submit("jlens-support", work).id}

    @router.get("/api/concepts")
    def concepts():
        return {"concepts": [asdict(c) for c in catalog.list_concepts()]}

    @router.get("/api/concept/{name}")
    def concept_detail(name: str):
        _safe_name(name)
        return catalog.concept_preview(name)

    # --- concept authoring (Concept Lab) ------------------------------------

    @router.get("/api/concept/{name}/full")
    def concept_full(name: str):
        _safe_name(name)
        from ..experiment import authoring
        return authoring.read_concept(name)

    @router.post("/api/concepts/create")
    def concept_create(body: dict):
        from ..experiment import authoring
        try:
            authoring.create_concept(body.get("name", ""))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"ok": True}

    @router.post("/api/concept/{name}/save")
    def concept_save(name: str, body: dict):
        _safe_name(name)
        from ..experiment import authoring
        try:
            return authoring.save_concept(name, body.get("positive", []),
                                          body.get("negative", []))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/api/concept/import")
    def concept_import(body: dict):
        from ..experiment import authoring
        return authoring.parse_import(body.get("content", ""), body.get("filename", ""))

    @router.post("/api/concept/{name}/delete")
    def concept_delete(name: str):
        _safe_name(name)
        from ..experiment import authoring
        return authoring.delete_concept(name)

    @router.get("/api/proposals/available")
    def proposals_available():
        from ..experiment import proposals
        return {"available": proposals.available()}

    @router.post("/api/concept/{name}/proposals")
    def concept_proposals(name: str, body: dict | None = None):
        _safe_name(name)
        from ..experiment import proposals
        body = body or {}
        try:
            pairs = proposals.generate_pairs(
                name, count=int(body.get("count", 10)), guidance=body.get("guidance", ""),
                examples_positive=body.get("examplesPositive"),
                examples_negative=body.get("examplesNegative"))
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"pairs": pairs}

    @router.get("/api/concept/{name}/prompt")
    def concept_prompt(name: str, count: int = 20):
        _safe_name(name)
        from ..experiment import authoring
        return {"prompt": authoring.generation_prompt(name, count)}

    @router.post("/api/concept/{name}/extract")
    def concept_extract(name: str, body: dict | None = None):
        _safe_name(name)
        _require_model_or_worker(state)
        from ..steering.extractor import ExtractionOptions, extract
        from ..steering.reading_position import LAST_TOKEN, mean_from_token
        from ..steering.stimulus_set import StimulusSet
        from ..steering.vector_math import ExtractionMethod
        from ..steering.vector_store import SteeringVectorSidecar, save as save_vec

        body = body or {}
        try:
            method = ExtractionMethod(body.get("method", "meanDifference"))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"bad method {body.get('method')!r}")
        pool = body.get("poolFromToken")
        reading = mean_from_token(int(pool)) if pool not in (None, "", 0, "0") else LAST_TOKEN
        options = ExtractionOptions(method=method, reading_position=reading)

        def work(job):
            stimuli = StimulusSet.from_directory(paths.concept_directory(name))
            with state.acquire_active() as model:
                result = extract(model, stimuli, options)
                run_dir = paths.make_unique_run_directory(f"concept-{name}")
                from ..experiment.run_config import write_run_config
                write_run_config(run_dir, "extract", model_id=model.model_id,
                                 revision=model.revision)
                sidecar = SteeringVectorSidecar.make(
                    model_id=model.model_id, revision=model.revision, concept=name,
                    stimulus_set_hash=stimuli.hash, vectors=result.vectors,
                    extraction_method=method.value, reading_position=reading,
                    residual_norm_per_layer=result.residual_norm_per_layer,
                    residual_norm_source=result.residual_norm_source,
                    residual_norm_convention=result.residual_norm_convention)
                save_vec(result.vectors, sidecar, run_dir, name)
            job.log(f"built {result.vectors.layer_count}-layer {method.value} vector "
                    f"({reading.label}) → {run_dir}")
            return {"runDirectory": run_dir, "name": name}

        return _run_or_submit(state, f"concept-extract:{name}", work,
                               path=f"/api/concept/{name}/extract",
                               body=body)

    # --- grand-mean (emotion multi-concept) ---------------------------------

    @router.get("/api/multiconcept/concepts")
    def mc_concepts():
        from ..experiment import multiconcept
        return {"concepts": multiconcept.list_story_concepts()}

    @router.get("/api/multiconcept/{concept}/stories")
    def mc_stories(concept: str):
        _safe_name(concept)
        from ..experiment import multiconcept
        return {"concept": concept, "rows": multiconcept.read_stories(concept)}

    @router.post("/api/multiconcept/{concept}/stories")
    def mc_save_stories(concept: str, body: dict):
        _safe_name(concept)
        from ..experiment import multiconcept
        return multiconcept.save_stories(concept, body.get("rows", []))

    @router.post("/api/multiconcept/extract")
    def mc_extract(body: dict):
        """Build grand-mean vectors: each target concept's direction is
        mean(its stories) − mean(all stories in the loaded corpus)."""
        _require_model_or_worker(state)
        from ..experiment import multiconcept
        from ..steering.extractor import extract_grand_mean
        from ..steering.reading_position import mean_from_token
        from ..steering.vector_store import (
            SteeringVectorSidecar,
            save as save_vec,
            stamp_grand_mean_provenance,
        )

        corpus_concepts = body.get("concepts")  # None = all
        targets = set(body.get("targets") or []) or None
        pool = int(body.get("poolFromToken", 50) or 50)
        reading = mean_from_token(pool)

        def work(job):
            rows, hashes = multiconcept.load_corpus(corpus_concepts)
            if not rows:
                raise RuntimeError("no stories in prompts/emotions/*/stories.jsonl")
            job.log(f"corpus: {len(rows)} stories across {len(hashes)} concepts")
            with state.acquire_active() as model:
                result = extract_grand_mean(model, rows, target_concepts=targets,
                                            reading_position=reading)
                run_dir = paths.make_unique_run_directory("grandmean")
                from ..experiment.run_config import write_run_config
                write_run_config(run_dir, "extract", model_id=model.model_id,
                                 revision=model.revision)
                for concept, vectors in result.per_concept.items():
                    sidecar = SteeringVectorSidecar.make(
                        model_id=model.model_id, revision=model.revision, concept=concept,
                        stimulus_set_hash=hashes.get(concept, ""), vectors=vectors,
                        reading_position=reading,
                        residual_norm_per_layer=result.residual_norm_per_layer,
                        residual_norm_source=result.residual_norm_source,
                        residual_norm_convention=result.residual_norm_convention,
                        source_stimulus_count=len(rows),
                        included_stimulus_count=result.included,
                        excluded_short_stimulus_count=len(rows) - result.included)
                    # A grand-mean direction is defined by the whole
                    # comparison population, not only the target's stories.
                    # Playground can inject the bytes without this stamp, but
                    # the optimization composer must be able to re-derive the
                    # exact recipe from source data. Keep the older
                    # comparisonConcepts field for Swift authoring UI and
                    # stamp the stronger concept→file-hash identity consumed
                    # by both engines' recipe firewall.
                    stamp_grand_mean_provenance(sidecar, hashes)
                    save_vec(vectors, sidecar, run_dir, concept)
            job.log(f"built {len(result.per_concept)} grand-mean vectors "
                    f"({result.included} stories pooled) → {run_dir}")
            return {"runDirectory": run_dir, "concepts": list(result.per_concept)}

        return _run_or_submit(state, "grandmean-extract", work,
                               path="/api/multiconcept/extract", body=body)

    @router.post("/api/concept/{name}/stats")
    def concept_stats_route(name: str, body: dict | None = None):
        _safe_name(name)
        state.require_model()
        from ..experiment import concept_stats
        from ..steering.extractor import activations
        from ..steering.reading_position import LAST_TOKEN, mean_from_token
        from ..steering.stimulus_set import StimulusSet
        from ..steering.vector_math import ExtractionMethod
        from ..steering.vector_store import load as load_vec

        body = body or {}
        try:
            method = ExtractionMethod(body.get("method", "meanDifference"))
        except ValueError:
            raise HTTPException(status_code=400, detail="bad method")
        pool = body.get("poolFromToken")
        reading = mean_from_token(int(pool)) if pool not in (None, "", 0, "0") else LAST_TOKEN
        stimuli = StimulusSet.from_directory(paths.concept_directory(name))
        with state.acquire_active() as model:
            pos = activations(model, stimuli.positive, reading).values
            neg = activations(model, stimuli.negative, reading).values
            model_id = model.model_id
        stats_layer = (len(pos[0]) // 2) if pos else 0
        controls: dict[str, list[float]] = {}
        for v in catalog.list_vectors():
            if v.modelID == model_id and v.concept != name and v.concept not in controls:
                try:
                    vecs, _ = load_vec(v.runDirectory, v.name)
                    controls[v.concept] = vecs.per_layer[min(stats_layer, vecs.layer_count - 1)]
                except (OSError, KeyError):
                    pass
        return concept_stats.compute(positive_by_layer=pos, negative_by_layer=neg,
                                     method=method, control_vectors=controls)

    # --- scalar reading probes ----------------------------------------------

    @router.get("/api/concept/{name}/probe-items")
    def probe_items(name: str):
        _safe_name(name)
        from ..experiment import probes
        rows = probes.read_items(name)
        return {"concept": name, "rows": rows,
                "positive": sum(1 for r in rows if r.get("expresses")),
                "negative": sum(1 for r in rows if not r.get("expresses"))}

    @router.post("/api/concept/{name}/probe-items")
    def probe_save(name: str, body: dict):
        _safe_name(name)
        from ..experiment import probes
        return probes.save_items(name, body.get("rows", []))

    @router.post("/api/concept/{name}/probe-import")
    def probe_import(name: str, body: dict):
        _safe_name(name)
        from ..experiment import probes
        return {"rows": probes.parse_items(body.get("content", ""))}

    @router.post("/api/concept/{name}/probe-train")
    def probe_train(name: str):
        _safe_name(name)
        state.require_model()
        from ..experiment import probes
        from ..steering.extractor import activations
        from ..steering.reading_position import LAST_TOKEN

        def work(job):
            items = probes.read_items(name)
            if not items:
                raise RuntimeError(f"no probe items in prompts/probes/{name}/items.jsonl")
            with state.acquire_active() as model:
                acts = activations(model, [it["text"] for it in items], LAST_TOKEN).values
                result = probes.train(items=items, activations_by_layer=acts)
                run_dir = paths.make_unique_run_directory(f"probe-{name}")
                from ..experiment.run_config import write_run_config
                write_run_config(run_dir, "probe-train", model_id=model.model_id,
                                 revision=model.revision)
                probes.save_artifact(name, model.model_id, model.revision, result, run_dir)
            job.log(f"probe layer {result['layer']} · held-out acc "
                    f"{result['accuracy']:.2f} → {run_dir}")
            return {"runDirectory": run_dir, "layer": result["layer"],
                    "accuracy": result["accuracy"]}

        return {"jobId": state.jobs.submit(f"probe-train:{name}", work).id}

    # --- neutral corpus + neutral-PC basis ----------------------------------

    @router.get("/api/neutral/corpora")
    def neutral_corpora():
        from ..experiment import neutral
        return {"corpora": neutral.list_corpora(), "bases": neutral.list_bases()}

    @router.post("/api/neutral/import")
    def neutral_import(body: dict):
        from ..experiment import neutral
        texts = neutral.parse_texts(body.get("content", ""))
        return neutral.save_corpus(texts, body.get("name") or None)

    @router.post("/api/neutral-pcs/build")
    def neutral_pcs_build(body: dict | None = None):
        state.require_model()
        from ..experiment import neutral
        from ..steering.extractor import neutral_activation_bank
        from ..steering.reading_position import mean_from_token

        body = body or {}
        corpus_name = body.get("corpus") or None
        min_variance = float(body.get("minVariance", 0.5))
        from ..steering.extractor import DEFAULT_MAX_NEUTRAL_PCS, DEFAULT_MAX_TOKEN_ROWS
        max_token_rows = int(body.get("maxTokenRows", DEFAULT_MAX_TOKEN_ROWS))
        # Hard ceiling alongside the variance target: the target still governs
        # below it, but it can no longer run to `rows − 1` PCs per layer.
        max_components = int(body.get("maxComponents", DEFAULT_MAX_NEUTRAL_PCS))

        def work(job):
            from ..experiment.run_config import write_run_config
            texts, corpus_hash = neutral.read_corpus(corpus_name)
            if len(texts) < 4:
                raise RuntimeError("neutral corpus needs ≥4 texts to build PCs")
            # Deterministic downsample seed FROM THE CORPUS HASH: the same
            # corpus always banks the same token positions, so the basis is
            # reproducible without a seeds table.
            seed = int((corpus_hash or "0" * 16)[:16], 16) % (2 ** 63)
            with state.acquire_active() as model:
                bank = neutral_activation_bank(model, texts,
                                               reading_position=mean_from_token(50),
                                               max_token_rows=max_token_rows,
                                               downsample_seed=seed)
                components = bank.components_by_layer(min_variance=min_variance,
                                                      maximum_count=max_components)
                by_layer = dict(zip(bank.layers, components))
                run_dir = paths.make_unique_run_directory("neutral-pcs")
                write_run_config(run_dir, "neutral-pcs", model_id=model.model_id,
                                 revision=model.revision)
                neutral.save_basis(model_id=model.model_id, revision=model.revision,
                                   corpus_name=corpus_name or "norm", corpus_hash=corpus_hash,
                                   components_by_layer=by_layer,
                                   residual_norm_per_layer=bank.residual_norm_per_layer,
                                   token_rows=bank.token_row_count, run_directory=run_dir,
                                   token_positions_total=bank.positions_total,
                                   token_positions_kept=bank.positions_kept,
                                   downsample_seed=bank.downsample_seed,
                                   row_cap_per_layer=max_token_rows,
                                   minimum_explained_variance=min_variance,
                                   maximum_component_count=max_components)
            total = sum(len(v) for v in by_layer.values())
            job.log(f"built {total} neutral PCs across {len(by_layer)} layers "
                    f"({bank.positions_kept}/{bank.positions_total} token "
                    f"positions banked) → {run_dir}")
            return {"runDirectory": run_dir, "totalComponents": total}

        return {"jobId": state.jobs.submit("neutral-pcs", work).id}

    @router.get("/api/experiments")
    def experiments():
        return {"experiments": catalog.list_experiments()}

    @router.get("/api/experiment/{name}")
    def experiment_detail(name: str):
        _safe_name(name)
        try:
            return catalog.experiment_detail(Manifest.load(name))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no experiment {name!r}")

    @router.get("/api/experiment/{name}/manifest")
    def experiment_manifest(name: str):
        """Raw manifest document — the server-resident experiment.json,
        verbatim (parsed, not summarized like ``experiment_detail``).

        Additive read route for the client-side remote-freeze identity
        check: before asking THIS server to freeze a same-named study, the
        app fetches this body and compares it — same-engine canonicalized,
        volatile freeze stamps excluded — against the manifest it displays,
        so a freeze can never silently act on a document the researcher is
        not looking at (the unpaired-workspace divergence).
        """
        _safe_name(name)
        directory = paths.experiments_directory()
        path = os.path.join(directory, name, "experiment.json")
        if not os.path.exists(path):
            path = os.path.join(directory, f"{name}.json")
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no experiment {name!r}")

    @router.put("/api/experiment/{name}/manifest")
    def experiment_manifest_replace(name: str, body: dict):
        """One-click server-draft sync (2026-07-21 incident, part 3): the
        write half of the remote-freeze identity check. When that check
        blocks ("the server's copy is NOT the manifest you are looking
        at"), the app's "Update the server's copy" affordance PUTs the
        displayed manifest here; the server installs it as its draft copy
        and answers the stored status + canonical body hash. Frozen server
        copies and non-draft incoming documents refuse in
        ``experiment_store.replace_draft_manifest`` (freeze firewall).
        Mutating authoring surface — token-gated exactly like POST
        /api/experiment/* (the middleware treats PUT the same)."""
        _safe_name(name)
        from ..experiment import experiment_store
        try:
            return experiment_store.replace_draft_manifest(name, body)
        except experiment_store.ExperimentStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/api/experiment/{name}/verify")
    def experiment_verify(name: str):
        _safe_name(name)
        return {"violations": Manifest.load(name).verify()}

    @router.post("/api/experiment/{name}/token-preflight")
    def experiment_token_preflight(name: str, body: dict | None = None):
        """EXACT prompt-token counts against the pinned model revision (C1).

        Weights-free: `AutoTokenizer` + `AutoConfig` only, so this costs no
        GPU, no model slot and no queue wait. That is the whole point — a
        Mac cannot settle whether a prompt fits a server-only model, because
        token counts belong to a specific tokenizer at a specific revision,
        so the exact answer must be computed where that revision lives.

        Reports EVERY over-budget item rather than the first, and never
        drops any: silently excluding items would change the measured sample
        without recording it. The refusal string is returned as data — the
        caller decides whether an over-budget study is fatal.

        POST under /api/experiment/ so the auth middleware treats it as
        privileged (it reads caller-named workspace files).
        """
        from ..experiment import token_preflight, tasks as tasks_mod
        _safe_name(name)
        body = body or {}
        try:
            manifest = Manifest.load(name)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no experiment {name!r}")
        try:
            prompts = tasks_mod._load_prompts(
                manifest, body.get("promptsFile"), None)
            report = token_preflight.preflight(
                prompts,
                model_id=manifest.model_id,
                revision=manifest.model_revision,
                prompt_mode=manifest.prompt_mode,
                system_prompt=manifest.system_prompt,
                qwen_thinking_enabled=manifest.qwen_thinking_enabled,
                max_tokens=int(body.get("maxTokens") or manifest.max_tokens or 0))
        except token_preflight.PreflightError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        report["refusal"] = token_preflight.refusal(report)
        return report

    @router.post("/api/geometry")
    def geometry(body: dict):
        """Pairwise cosine matrix across selected vectors at a layer (parallel to
        the Swift Geometry panel's cross-concept structure).

        Vector addressing accepts BOTH conventions: ``vectorPath`` as the run
        DIRECTORY plus a separate ``name`` (the server workbench's form), and
        ``vectorPath`` as the catalog id / inline-spec form
        ``<runDirectory>/<name>`` (what ``GET /api/vectors`` returns as ``id``
        and what variant injections reference) — split exactly like
        ``variant_injections`` does.

        Silent-skip contract (a 0×0 "success" is a lie a client cannot
        distinguish from an empty selection): unloadable entries are reported
        per-vector in ``skipped: [{name, reason}]``; when vectors were
        requested and NONE loaded, the route answers 400 naming every entry
        instead of returning an empty matrix.
        """
        from ..steering import vector_math as vm
        items = body.get("vectors", [])
        layer = body.get("layer")
        loaded = []
        skipped: list[dict] = []
        for it in items:
            ref = str(it.get("vectorPath") or "")
            name = str(it.get("name") or os.path.basename(ref))
            label = str(it.get("label") or name or ref or "?")
            try:
                try:
                    vdir = state.resolver.require_dir(ref, allow_local_absolute=True)
                except HTTPException:
                    # Catalog-id form: the last path component is the vector
                    # NAME, not a directory — resolve its parent instead.
                    vdir = state.resolver.require_dir(
                        os.path.dirname(ref), allow_local_absolute=True)
                    name = os.path.basename(ref)
                vecs, _ = vector_store.load(vdir, name)
            except HTTPException as exc:
                skipped.append({"name": label, "reason": str(exc.detail)})
                continue
            except (OSError, KeyError, ValueError) as exc:
                # ValueError: non-finite legacy artifact refused by load().
                skipped.append({"name": label, "reason": str(exc)})
                continue
            idx = (vecs.layer_count // 2) if layer is None else min(max(0, int(layer)), vecs.layer_count - 1)
            loaded.append((label, vecs.per_layer[idx], idx))
        if items and not loaded:
            detail = "; ".join(f"{s['name']}: {s['reason']}" for s in skipped)
            raise HTTPException(
                status_code=400,
                detail=f"no geometry vectors could be loaded — {detail}")
        labels = [l for l, _, _ in loaded]
        matrix = []
        for _, a, _ in loaded:
            row = []
            for _, b, _ in loaded:
                try:
                    row.append(round(vm.cosine_similarity(a, b), 4))
                except vm.SteeringVectorError:
                    row.append(None)
            matrix.append(row)
        return {"labels": labels, "matrix": matrix,
                "layers": [idx for _, _, idx in loaded],
                "skipped": skipped}

    @router.get("/api/runs")
    def runs():
        return {"runs": [asdict(r) for r in catalog.list_runs()]}

    # Server-side ceiling on a bounded run-file read (``head=`` below). The
    # remote Results browser previews with heads well under this; the cap
    # exists so a buggy/hostile client cannot turn "bounded" back into a
    # whole-file slurp of a multi-hundred-MB generations.jsonl. HARD
    # REQUIREMENT: this must stay >= the Swift client's report head cap
    # (``ExperimentPanel.remoteReportByteLimit``, 4 MiB) — a server cap below
    # the client's request silently truncated big report.json fetches, which
    # then failed to decode and dropped run-level metrics.
    RUN_FILE_HEAD_CAP = 8_388_608  # 8 MiB

    # Head-response metadata headers: the ACTUAL on-disk file size and
    # whether the returned bytes are a truncated head of it. The client
    # prefers these over listed sizes (a 0/absent listed size means
    # "unknown", never "empty") when deciding whether a head is complete.
    RUN_FILE_SIZE_HEADER = "X-SteerLab-File-Size"
    RUN_FILE_TRUNCATED_HEADER = "X-SteerLab-Truncated"

    @router.get("/api/runs/{run_id}/file")
    def run_file(run_id: str, name: str, head: int | None = None):
        _safe_name(run_id)
        _safe_name(name)
        path = os.path.join(paths.runs_directory(), run_id, name)
        runs_root = os.path.realpath(paths.runs_directory())
        if os.path.commonpath([os.path.realpath(path), runs_root]) != runs_root \
                or not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="no such run file")
        if head is not None:
            if head <= 0:
                raise HTTPException(status_code=400,
                                    detail="head must be a positive byte count")
            limit = min(head, RUN_FILE_HEAD_CAP)
            try:
                size = os.path.getsize(path)
                with open(path, "rb") as handle:
                    data = handle.read(limit)
            except OSError as exc:
                raise HTTPException(status_code=404,
                                    detail="no such run file") from exc
            return Response(
                content=data, media_type="application/octet-stream",
                headers={
                    RUN_FILE_SIZE_HEADER: str(size),
                    RUN_FILE_TRUNCATED_HEADER:
                        "true" if len(data) < size else "false",
                })
        return FileResponse(path)

    # --- jobs ---------------------------------------------------------------

    @router.get("/api/jobs")
    def list_jobs():
        jobs = state.jobs.list()
        store = getattr(state.jobs, "store", None)
        if store is None:
            return {"jobs": [j.to_dict() for j in jobs]}
        # One batched query for every tail: per-job reads each open a fresh
        # connection against an NFS-resident sqlite file, which made this
        # endpoint take minutes at a few hundred jobs (the app timed out).
        tails = store.log_tails(limit=50)
        return {"jobs": [j.to_dict(log_tail=tails.get(j.id, [])) for j in jobs]}

    @router.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job.to_dict()

    @router.get("/api/jobs/{job_id}/stream")
    def stream_job(job_id: str):
        if state.jobs.get(job_id) is None:
            raise HTTPException(status_code=404, detail="no such job")

        def gen():
            for line in state.jobs.stream(job_id):
                yield f"data: {json.dumps({'line': line})}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @router.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        if not state.jobs.cancel(job_id):
            # cancel() returns achieved-or-honestly-accepted; False here means
            # a Slurm scancel failed — answering ok would hide a possibly
            # still-running billed allocation (live shakedown 2026-07-16).
            raise HTTPException(
                status_code=502,
                detail=(f"scancel {job.executor_job_id} did not confirm — "
                        "the Slurm allocation may still be running; retry "
                        f"the cancel, or run `scancel {job.executor_job_id}` "
                        f"on the cluster and check "
                        f"`sacct -j {job.executor_job_id}`"))
        return {"ok": True}

    def _resume_cancelled_in_process(job) -> dict:
        """Resume a ``cancelledResumable`` IN-PROCESS job (review 2026-08-03
        round 3, P1): a cancel-parked sweep/run has no ``run.sbatch`` to
        re-execute, so Resume re-dispatches the verb against the parked run
        directory as a NEW in-process job. Manual only — auto-resubmit never
        follows a user's cancel.

        Concurrency discipline (round 4, P1): the same durable SQLite
        compare-and-set claim the Slurm resubmit core uses — two
        simultaneous clicks can never both submit, and a crash between
        submission and stamping is repaired from the continuation's own
        ``resubmitOf`` marker rather than double-submitting."""
        result = job.result or {}
        if result.get("resubmittedAs"):
            raise HTTPException(
                status_code=409,
                detail=f"job {job.id} already resumed as "
                       f"{result['resubmittedAs']} — that continuation is "
                       "carrying the run")
        # Submit-then-stamp crash repair BEFORE claiming: a continuation
        # written durably by a dead claimant means the resume already
        # happened exactly once — restore the stamp, never submit again.
        existing = state.jobs._existing_resubmission(job)
        if existing is not None:
            job.result = {**{k: v for k, v in result.items()
                             if k != "resubmitClaim"},
                          "resubmittedAs": existing.id,
                          "manualResubmit": True}
            job.log(f"resume found existing continuation {existing.id} — "
                    "repaired the resubmittedAs stamp (no new submission)")
            state.jobs.store.update(job)
            return {"ok": True, "resubmitOf": job.id, "jobId": existing.id,
                    "manualResubmit": True}
        run_dir = result.get("runDirectory")
        name = result.get("experiment")
        resume_state = resume_mod.read_state(run_dir) if run_dir else None
        verb = (resume_state or {}).get("verb")
        if not (run_dir and name and verb in ("sweep", "run")
                and resume_mod.is_resumable(run_dir, verb)):
            raise HTTPException(
                status_code=409,
                detail=(f"job {job.id} is cancelledResumable but its parked "
                        "run directory is missing, already complete, or "
                        "carries no resume state — submit the study verb "
                        "again for a fresh run"))
        _safe_name(str(name))
        claimed, result_now = state.jobs.store.claim_resubmit(
            job.id, claimant="manual-inprocess")
        if (not claimed
                and not (result_now or {}).get("resubmittedAs")
                and state.jobs.store.claim_is_stale(result_now)
                and state.jobs._existing_resubmission(job) is None):
            # Claim-then-crash recovery (round 5, P2): the claimant died
            # between claiming and the continuation's durable insertion —
            # the stale window has passed AND no resubmitOf child exists in
            # the store (submit inserts durably before work starts), so the
            # dead claimant provably never submitted. A manual click may
            # take the claim over; blind takeover of a LIVE claim stays
            # forbidden.
            claimed, result_now = state.jobs.store.takeover_stale_claim(
                job.id, "manual-inprocess")
            if claimed:
                job.log("stale resume claim taken over (claimant died "
                        "before submitting) — resuming")
        if not claimed:
            already = (result_now or {}).get("resubmittedAs")
            raise HTTPException(
                status_code=409,
                detail=(f"job {job.id} already resumed as {already} — that "
                        "continuation is carrying the run" if already else
                        f"job {job.id} has a live resume claim — another "
                        "Resume is in flight; check the job list for its "
                        "continuation"))

        def work(continuation):
            continuation.log(
                f"experiment {name}: {verb} resuming cancelled job "
                f"{job.id} from {run_dir}")
            should_cancel = lambda: continuation.cancelled  # noqa: E731
            if verb == "sweep":
                out = tasks.sweep(name, model_provider=state.acquire_model,
                                  max_loaded=state.registry.max_loaded,
                                  should_cancel=should_cancel,
                                  run_directory=run_dir,
                                  log=continuation.log)
            else:
                out = tasks.run(name, model_provider=state.acquire_model,
                                should_cancel=should_cancel,
                                run_directory=run_dir,
                                log=continuation.log)
            continuation.log(f"experiment {name}: {verb} → {out}")
            return {"runDirectory": out, "experiment": name}

        try:
            # resubmitOf makes the continuation durably discoverable — the
            # crash-repair path above (and the store's
            # find_resubmissions_of) key on it.
            continuation = state.jobs.submit(
                f"experiment:{verb}", work,
                requested_resources={"resubmitOf": job.id})
        except Exception:
            # Submission failed — release so a later click can retry.
            state.jobs.store.release_resubmit_claim(job.id, "manual-inprocess")
            raise
        stamped = {k: v for k, v in result_now.items()
                   if k != "resubmitClaim"}
        stamped["resubmittedAs"] = continuation.id
        stamped["manualResubmit"] = True
        job.result = stamped
        job.log(f"resumed manually as in-process job {continuation.id} — "
                "continuing from the parked run directory")
        state.jobs.store.update(job)
        return {"ok": True, "resubmitOf": job.id, "jobId": continuation.id,
                "manualResubmit": True}

    @router.post("/api/jobs/{job_id}/resubmit")
    def resubmit_job(job_id: str):
        """Manual resume of a checkpointed job (the app's Resume button).
        Valid in the ``checkpointed`` state (re-sbatches the job's OWN
        ``run.sbatch`` through the same implementation auto-resubmit uses)
        and the ``cancelledResumable`` state (re-dispatches the in-process
        verb against the parked directory). Privileged
        (app._PRIVILEGED_PREFIXES clause): this route runs sbatch."""
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        if job.status == "cancelledResumable":
            return _resume_cancelled_in_process(job)
        try:
            return state.jobs.resubmit(job_id)
        except ResubmitRefused as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except Exception as exc:  # noqa: BLE001 - sbatch and kin; run stays parked
            raise HTTPException(
                status_code=502,
                detail=(f"resubmit did not go through: {exc} — the "
                        "checkpointed run is unchanged; retry, or sbatch the "
                        "job's own run.sbatch on the cluster by hand"))

    @router.post("/api/jobs/reconcile")
    def reconcile_jobs(body: dict | None = None):
        """Fold child records, then run the shard-merge pass.

        **Open-issues §2b.** The endpoint used to require ``recordsDirectory``
        (bare POST → 422, ``{}`` → 400) and folded records ONLY — it never
        touched the merge, so the one thing an operator reached for when a
        sharded run stalled could not substitute for a daemon cycle. Two
        changes, no removals:

        - **No body, or ``{}``, means "everything"**: every records directory
          the store knows about. A stalled daemon is exactly the situation in
          which the operator cannot say which submission is stuck.
        - **The merge pass always runs afterwards**, for both modes, and the
          response reports what was reconciled AND what was merged.

        A malformed non-empty body is still a 400 — an operator who typed a key
        wrong must not silently get the reconcile-everything behaviour."""
        body = {} if body is None else body
        if not isinstance(body, dict):
            raise HTTPException(
                status_code=400,
                detail="body must be a JSON object (or empty for "
                       "\"reconcile everything, then merge\")")
        unknown = sorted(set(body) - {"recordsDirectory"})
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown field(s) {', '.join(unknown)} — this endpoint "
                       "takes recordsDirectory, or an empty body for "
                       "\"reconcile everything, then merge\"")
        records_dir = body.get("recordsDirectory")
        if records_dir is not None and (not isinstance(records_dir, str)
                                        or not records_dir.strip()):
            raise HTTPException(
                status_code=400,
                detail="recordsDirectory must be a non-empty string — omit it "
                       "entirely to reconcile every known records directory")
        if records_dir:
            directory = state.resolver.require_dir(
                records_dir, allow_local_absolute=True)
            reconciled = state.jobs.reconcile(directory)
            visited = [str(directory)]
            scope = "recordsDirectory"
        else:
            reconciled, visited = state.jobs.reconcile_all()
            scope = "all"
        merged = state.jobs.run_merge_pass()
        return {
            "reconciled": reconciled,
            "scope": scope,
            "recordsDirectories": visited,
            "merged": merged,
            "monitor": state.jobs.monitor_health(),
        }

    # --- GPU session lifecycle (GPU-SESSION-PLAN Wave 1) ----------------------
    # Controller-owned; token-gated via app._PRIVILEGED_PREFIXES. The proxy
    # itself lives in app.py middleware — these verbs manage the session and
    # are never proxied.

    @router.post("/api/session/start")
    def session_start(body: dict | None = None):
        role = server_role()
        if role == "gpu-session":
            # No recursion: a worker acquiring workers would chain billed
            # allocations behind a single chat client.
            raise HTTPException(
                status_code=409,
                detail="a GPU session worker cannot start another GPU "
                       "session (no recursion) — session lifecycle lives on "
                       "the controller")
        if role != "controller":
            raise HTTPException(
                status_code=409,
                detail="GPU sessions are managed by a controller-role server "
                       "(STEERLAB_SERVER_ROLE=controller, the daemon-in-a-job"
                       " topology) — this workstation loads models directly "
                       "(POST /api/load)")
        try:
            record = gpu_session.start_session(body or {}, state.jobs)
        except gpu_session.SessionConflict as exc:
            # Idempotent double-start: the live record rides in the 409 body
            # so the caller can adopt the running session instead of racing
            # a second worker into existence.
            return JSONResponse(
                {"detail": "a GPU session already exists — reuse it, or end "
                           "it first (DELETE /api/session)",
                 "session": exc.session},
                status_code=409)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"session": record}

    @router.get("/api/session")
    def session_get():
        role = server_role()
        if role == "gpu-session":
            # The worker's own status: its generation, busy flag, and idle
            # countdown — what the controller's reconcile folds into the
            # session record.
            return gpu_session.ensure_worker().status()
        if role != "controller":
            return {"session": None}
        return {"session": gpu_session.reconcile_session()}

    @router.delete("/api/session")
    def session_delete(force: bool = False, body: dict | None = None):
        # ``force`` rides either the query string (?force=1) or a JSON body
        # ({"force": true}) — both clients exist (Swift sends the body, curl
        # and the web workbench favor the query). Force is the OPERATOR CLEAR
        # for a session stuck "unknown"/"ending": best-effort scancel, then
        # stamp ended — only after checking the job by hand (sacct/squeue).
        if server_role() != "controller":
            raise HTTPException(
                status_code=409,
                detail="GPU sessions are managed by a controller-role server"
                       " — nothing to end here")
        force = force or bool((body or {}).get("force"))
        return {"ok": True,
                "session": gpu_session.end_session(state.jobs, force=force)}

    @router.post("/api/session/keepalive")
    async def session_keepalive(request: Request):
        role = server_role()
        if role == "gpu-session":
            # Counts as activity twice over: the idle middleware brackets the
            # request, and the explicit touch covers direct (unproxied) pings.
            worker = gpu_session.ensure_worker()
            worker.timer.touch()
            return {"ok": True, **worker.status()}
        if role != "controller":
            raise HTTPException(status_code=409,
                                detail=gpu_session.NO_SESSION_DETAIL)
        return await gpu_session.forward_keepalive(request)

    # --- housekeeping (WS3) --------------------------------------------------
    # status is read-only and unprivileged: it serves the last tick's persisted
    # snapshot plus live df stats — never an expensive scan. refresh (forced
    # rescan) and maintenance (writes the calendar the executor enforces) are
    # privileged via app._PRIVILEGED_PREFIXES.

    @router.get("/api/housekeeping/status")
    def housekeeping_status():
        from . import housekeeping
        return housekeeping.status()

    @router.post("/api/housekeeping/refresh")
    def housekeeping_refresh():
        from . import housekeeping
        return housekeeping.refresh(jobs=state.jobs)

    @router.post("/api/housekeeping/maintenance")
    def housekeeping_maintenance(body: dict):
        from . import housekeeping
        try:
            return housekeeping.write_maintenance(body.get("windows"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/api/slurm/bundle")
    def slurm_bundle(body: dict):
        command = body.get("command")
        if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
            raise HTTPException(status_code=400, detail="command must be a list of strings")
        bundle_dir = body.get("bundleDirectory")
        if not bundle_dir:
            bundle_dir = paths.make_unique_run_directory("slurm-bundle")
        elif not os.path.isabs(bundle_dir):
            bundle_dir = state.resolver.resolve_workspace(bundle_dir)
        elif ServerProfile.from_env().profile != "local":
            raise HTTPException(status_code=400, detail="absolute bundleDirectory rejected")
        res_body = body.get("resources") or {}
        try:
            # The site's own facts (WP5 Step 8): QOS, node features,
            # reservation, required headers, GPU vocabulary and the scheduler
            # binaries come from what the site declared, and the request body
            # supplies only the job's own shape. This route used to build a
            # bundle that knew none of them.
            site = SlurmResources.from_env()
            resources = SlurmResources(
                job_name=res_body.get("jobName", "steerlab"),
                partition=res_body.get("partition"),
                gres=res_body.get("gres"),
                gpus=int(res_body.get("gpus", 1)),
                memory=res_body.get("memory"),
                walltime=res_body.get("walltime", "04:00:00"),
                cpus_per_task=int(res_body.get("cpusPerTask", 4)),
                signal_seconds=int(res_body.get("signalSeconds", 600)),
                signal_target=res_body.get("signalTarget", "step"),
                use_srun=bool(res_body.get("useSrun", True)),
                export_none=bool(res_body.get("exportNone", True)),
                account=res_body.get("account"),
                qos=site.qos,
                constraints=list(site.constraints),
                reservation=site.reservation,
                required_headers=list(site.required_headers),
                gpu_types=list(site.gpu_types),
                gpu_vram_gb=dict(site.gpu_vram_gb),
                extra_sbatch=list(res_body.get("extraSbatch", [])) or list(
                    site.extra_sbatch),
            )
            bundle = SlurmExecutor().create_bundle(
                bundle_dir, command, env=body.get("env") or {},
                resources=resources, metadata=body.get("metadata") or {})
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"bundle": bundle.to_dict()}

    @router.post("/api/slurm/submit")
    def slurm_submit(body: dict):
        _refuse_worker_submission()
        command = body.get("command")
        if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
            raise HTTPException(status_code=400, detail="command must be a list of strings")
        # Shared workspace-root hold: the bundle directory below is resolved
        # against the current root and the job record is written after sbatch
        # returns — a switch between the two would file the record under a
        # different workspace than the bundle it names.
        with workspace_lock.submitting():
            return _slurm_submit(body, command)

    def _slurm_submit(body: dict, command: list[str]):
        bundle_dir = body.get("bundleDirectory") or paths.make_unique_run_directory("slurm-submit")
        if not os.path.isabs(bundle_dir):
            bundle_dir = state.resolver.resolve_workspace(bundle_dir)
        try:
            resources = SlurmResources.from_env(job_name=body.get("jobName", "steerlab"))
            if body.get("resources"):
                tmp = body["resources"]
                resources.gres = tmp.get("gres", resources.gres)
                resources.partition = tmp.get("partition", resources.partition)
                resources.walltime = tmp.get("walltime", resources.walltime)
                resources.memory = tmp.get("memory", resources.memory)
            bundle = SlurmExecutor().create_bundle(
                bundle_dir, command, env=body.get("env") or {},
                resources=resources, metadata=body.get("metadata") or {})
            if body.get("dryRun", False):
                return {"dryRun": True, "bundle": bundle.to_dict()}
            # Gate on reality, not just declared env: actually running sbatch
            # requires the Slurm executor to be declared, which in turn makes
            # this route token-gated by the auth middleware.
            if ServerProfile.from_env().executor != "slurm":
                raise HTTPException(
                    status_code=403,
                    detail="sbatch submission requires STEERLAB_EXECUTOR=slurm")
            slurm_id = SlurmExecutor().submit(bundle)
            job = state.jobs.record_external(
                "slurm-submit", status="submitted", executor="slurm",
                executor_job_id=slurm_id,
                requested_resources=bundle.resources.__dict__,
                result={"bundle": bundle.to_dict()},
                log=f"submitted Slurm job {slurm_id}")
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"jobId": job.id, "slurmJobID": slurm_id, "bundle": bundle.to_dict()}

    @router.post("/api/studies/submit")
    def submit_study_route(body: dict):
        _refuse_worker_submission()
        name = body.get("experiment")
        if not name:
            raise HTTPException(status_code=400, detail="experiment required")
        _safe_name(name)
        resolver = state.resolver
        target_root = _resolve_optional_dir(resolver, body.get("targetRoot"),
                                            resolver.roots.workspace)
        prompts_path = _resolve_optional_file(resolver, body.get("promptsPath"))
        source_path = _resolve_optional_dir(resolver, body.get("sourcePath"),
                                            resolver.roots.runs)
        try:
            submission = submit_study(
                name, verb=body.get("verb", "run"), jobs=state.jobs,
                executor=body.get("executor"), dry_run=bool(body.get("dryRun", False)),
                dtype=body.get("dtype", "auto"), device=body.get("device"),
                target_root=target_root, prompts_path=prompts_path,
                source_path=source_path,
                package_evidence=bool(body.get("packageEvidence", True)),
                resources=body.get("resources") or {}, env=body.get("env") or {},
                force=bool(body.get("force", False)),
                # Targeted retry through the SLURM/bundle path (2026-07-24,
                # cross-engine body key "resumeFrom"): finish a failed
                # evaluation by judging only its undecided cells. This is
                # the submission path a cluster study actually uses, and
                # the one where redoing a judged evaluate costs most.
                resume_from=(str(body.get("resumeFrom")).strip() or None
                             if body.get("resumeFrom") else None),
                registry=state.registry)
        except PreflightRejection as exc:
            raise HTTPException(
                status_code=400,
                detail={"message": str(exc), "preflight": exc.preflight})
        except (OSError, RuntimeError, ValueError, FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return submission.to_dict()

    @router.post("/api/studies/submit-bundle")
    def submit_bundle_route(body: dict):
        _refuse_worker_submission()
        bundle_path = body.get("bundlePath")
        if not bundle_path:
            raise HTTPException(status_code=400, detail="bundlePath required")
        resolver = state.resolver
        bundle_path = resolver.require_file(
            bundle_path, root=resolver.roots.runs, allow_local_absolute=True)
        # Contain the WRITE target and read inputs the same way as the bundle
        # itself: unresolved, these reach import_bundle(allow_overwrite=True) and
        # tasks' prompt/source loaders, escaping the workspace root.
        target_root = _resolve_optional_dir(resolver, body.get("targetRoot"),
                                            resolver.roots.workspace)
        prompts_path = _resolve_optional_file(resolver, body.get("promptsPath"))
        source_path = _resolve_optional_dir(resolver, body.get("sourcePath"),
                                            resolver.roots.runs)
        try:
            submission = submit_run_bundle(
                bundle_path, verb=body.get("verb", "run"), jobs=state.jobs,
                executor=body.get("executor"), dry_run=bool(body.get("dryRun", False)),
                dtype=body.get("dtype", "auto"), device=body.get("device"),
                target_root=target_root, prompts_path=prompts_path,
                source_path=source_path,
                package_evidence=bool(body.get("packageEvidence", True)),
                resources=body.get("resources") or {}, env=body.get("env") or {},
                force=bool(body.get("force", False)),
                # Multi-GPU fan-out (default 1 = the historical single-job
                # path, byte-identical): K sibling shard jobs + a merging
                # parent. Execution logistics only — never in the manifest.
                parallel_jobs=int(body.get("parallelJobs", 1) or 1))
        except PreflightRejection as exc:
            raise HTTPException(
                status_code=400,
                detail={"message": str(exc), "preflight": exc.preflight})
        except (OSError, RuntimeError, ValueError, FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return submission.to_dict()

    @router.post("/api/bundles/run")
    def make_run_bundle(body: dict):
        from ..experiment import bundles
        name = body.get("experiment")
        if not name:
            raise HTTPException(status_code=400, detail="experiment required")
        _safe_name(name)
        output = body.get("outputPath")
        if output:
            if os.path.isabs(output) and ServerProfile.from_env().profile != "local":
                raise HTTPException(status_code=400, detail="absolute outputPath rejected")
            if not os.path.isabs(output):
                output = state.resolver.resolve_workspace(output)
        try:
            return bundles.package_experiment(name, output_path=output)
        except (OSError, bundles.BundleError, FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/api/bundles/evidence")
    def make_evidence_bundle(body: dict):
        from ..experiment import bundles
        run_dir = body.get("runDirectory")
        if not run_dir:
            raise HTTPException(status_code=400, detail="runDirectory required")
        run_dir = state.resolver.require_dir(run_dir, root=paths.runs_directory(),
                                             allow_local_absolute=True)
        output = body.get("outputPath")
        if output:
            if os.path.isabs(output) and ServerProfile.from_env().profile != "local":
                raise HTTPException(status_code=400, detail="absolute outputPath rejected")
            if not os.path.isabs(output):
                output = state.resolver.resolve_run(output)
        # A ledger-only failure record (a refused continuation's pipeline
        # dir) cannot produce an evidence bundle — answer a structured
        # skip so a bulk import notes it and keeps going (2026-08-11,
        # a factorial memo study). Real packaging errors below stay loud.
        skip = bundles.failure_record_skip(run_dir)
        if skip is not None:
            return skip
        try:
            return bundles.package_evidence(run_dir, output_path=output)
        except (OSError, bundles.BundleError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/api/bundles/upload")
    async def upload_bundle(request: Request):
        """Upload a bundle-like artifact into a server-managed staging run dir.

        The route intentionally accepts a raw body instead of FastAPI's
        ``UploadFile`` so the core server does not require the optional
        python-multipart package. Clients should send ``application/octet-stream``
        plus ``X-SteerLab-Filename``.

        **Staged, then committed** (external review, 2026-08-24). The body used
        to be written straight to its final name, so a client that disconnected
        mid-upload — or any exception out of ``request.stream()`` — left a
        truncated file sitting at the destination inside the runner's runs
        tree, indistinguishable from a complete one to anything that later
        listed the directory. The three explicit refusals cleaned up after
        themselves; the one nobody raises did not. Now the bytes land in a temp
        file beside the destination and are renamed into place only after the
        archive has been inspected, and a ``finally`` removes the temp — and
        the staging directory, when it is empty — on EVERY unsuccessful exit.
        The success path, and the response shape, are unchanged.
        """
        from ..experiment import bundles
        filename = request.headers.get("x-steerlab-filename", "upload.bundle")
        if "/" in filename or "\\" in filename or "\0" in filename:
            raise HTTPException(status_code=400, detail="invalid upload filename")
        run_dir = paths.make_unique_run_directory("uploaded-bundle")
        dest = os.path.join(run_dir, filename)
        handle_fd, temp = tempfile.mkstemp(prefix=".upload-", suffix=".part",
                                           dir=run_dir)
        os.close(handle_fd)
        committed = False
        try:
            total = 0
            hasher = hashlib.sha256()
            oversize = False
            cap = _max_upload_bytes()
            with open(temp, "wb") as handle:
                async for chunk in request.stream():
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > cap:
                        oversize = True
                        break
                    hasher.update(chunk)
                    handle.write(chunk)
            if oversize:
                raise HTTPException(
                    status_code=413,
                    detail=f"upload exceeds {cap} bytes (STEERLAB_MAX_UPLOAD_BYTES)")
            if total == 0:
                raise HTTPException(status_code=400, detail="empty upload")
            try:
                inspected = bundles.inspect_bundle(temp)
            except (OSError, bundles.BundleError, tarfile.TarError,
                    json.JSONDecodeError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"not a SteerLab bundle: {exc}")
            os.replace(temp, dest)
            committed = True
            # `inspect_bundle` stamps the path it was handed; the document
            # must name where the archive actually IS, not where it waited.
            inspected["bundlePath"] = dest
            executable = inspected.get("kind") == "runBundle"
            return {
                "path": dest,
                "filename": filename,
                "sha256": hasher.hexdigest(),
                "bytes": total,
                "bundle": inspected,
                "executable": executable,
                "stagingDirectory": run_dir,
            }
        finally:
            if not committed:
                try:
                    os.remove(temp)
                except OSError:
                    pass
                try:
                    # Only when empty — a staging directory that somehow holds
                    # something else is not this handler's to remove.
                    os.rmdir(run_dir)
                except OSError:
                    pass

    @router.get("/api/bundles/download")
    def download_bundle(path: str):
        safe_path = state.resolver.require_file(
            path, root=state.resolver.roots.runs, allow_local_absolute=True)
        name = os.path.basename(safe_path)
        # .safetensors joined 2026-08-05: the app localizes server-side vector
        # artifacts (sidecar + tensor pair) into the Mac workspace on picker
        # selection — the workspace is the source of truth; the server only
        # caches. Still read-only and contained to the runs root.
        if not (name.endswith(".tar.gz") or name.endswith(".json")
                or name.endswith(".jsonl") or name.endswith(".safetensors")):
            raise HTTPException(status_code=400, detail="unsupported downloadable artifact")
        return FileResponse(safe_path, filename=name)

    @router.post("/api/bundles/inspect")
    def inspect_bundle(body: dict):
        from ..experiment import bundles
        bundle_path = body.get("bundlePath")
        if not bundle_path:
            raise HTTPException(status_code=400, detail="bundlePath required")
        bundle_path = state.resolver.require_file(bundle_path, allow_local_absolute=True)
        try:
            return bundles.inspect_bundle(bundle_path)
        except (OSError, bundles.BundleError, tarfile.TarError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/api/bundles/import")
    def import_bundle(body: dict):
        from ..experiment import bundles
        bundle_path = body.get("bundlePath")
        if not bundle_path:
            raise HTTPException(status_code=400, detail="bundlePath required")
        bundle_path = state.resolver.require_file(bundle_path, allow_local_absolute=True)
        target = body.get("targetRoot")
        if target:
            if os.path.isabs(target) and ServerProfile.from_env().profile != "local":
                raise HTTPException(status_code=400, detail="absolute targetRoot rejected")
            if not os.path.isabs(target):
                target = state.resolver.resolve_workspace(target)
        try:
            # `expectedSha256` (additive, G3): the out-of-band outer pin,
            # verified BEFORE extraction. Absent = today's behaviour; the
            # cross-platform client always sends it.
            return bundles.import_bundle(
                bundle_path, target_root=target,
                allow_overwrite=bool(body.get("allowOverwrite", False)),
                expected_sha256=body.get("expectedSha256"))
        except (OSError, bundles.BundleError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # --- model variants + LoRA fine-tuning ----------------------------------

    @router.get("/api/variants")
    def list_variants():
        from ..experiment import model_variant
        return {"variants": model_variant.list_variants()}

    @router.post("/api/variants/upload")
    def upload_variant(body: dict):
        from ..experiment import model_variant
        payload = dict(body.get("variant") if isinstance(body.get("variant"), dict) else body)
        try:
            variant = model_variant.ModelVariant.from_dict(payload)
            saved = model_variant.save_variant(variant)
        except (KeyError, OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        reference_map: dict[str, str] = {}
        uploaded_bytes = 0
        for artifact in body.get("artifacts") or []:
            ref = str(artifact.get("reference") or "")
            rel = str(artifact.get("relativePath") or "")
            encoded = artifact.get("dataBase64")
            if not ref or not rel or not encoded:
                raise HTTPException(status_code=400, detail="artifact needs reference, relativePath, dataBase64")
            if os.path.isabs(rel) or "\0" in rel or ".." in rel.split("/"):
                raise HTTPException(status_code=400, detail="unsafe artifact relativePath")
            uploaded_bytes += (len(encoded) * 3) // 4  # base64 → decoded byte estimate
            if uploaded_bytes > _max_upload_bytes():
                raise HTTPException(
                    status_code=413,
                    detail=f"variant artifacts exceed {_max_upload_bytes()} bytes (STEERLAB_MAX_UPLOAD_BYTES)")
            target = os.path.realpath(os.path.join(saved["runDirectory"], "artifacts", rel))
            artifact_root = os.path.realpath(os.path.join(saved["runDirectory"], "artifacts"))
            if os.path.commonpath([target, artifact_root]) != artifact_root:
                raise HTTPException(status_code=400, detail="artifact escapes staging root")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            try:
                data = base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"bad artifact encoding: {exc}")
            with open(target, "wb") as handle:
                handle.write(data)
            if artifact.get("kind") == "vector":
                base, ext = os.path.splitext(target)
                if ext in {".json", ".safetensors"}:
                    reference_map[ref] = base
            elif artifact.get("kind") == "adapter":
                parts = rel.split("/")
                if len(parts) >= 2 and parts[0] == "adapters":
                    reference_map[ref] = os.path.realpath(
                        os.path.join(saved["runDirectory"], "artifacts", "adapters", parts[1]))
                else:
                    reference_map[ref] = os.path.dirname(target)

        if reference_map:
            for inj in payload.get("injections", []):
                ref = inj.get("vectorArtifactID")
                if ref in reference_map:
                    inj["vectorArtifactID"] = reference_map[ref]
            for adapter in payload.get("adapters", []):
                for key in ("adapterDirectory", "artifactPath"):
                    ref = adapter.get(key)
                    if ref in reference_map:
                        adapter[key] = reference_map[ref]
            if payload.get("neutralPCBasisPath") in reference_map:
                payload["neutralPCBasisPath"] = reference_map[payload["neutralPCBasisPath"]]
            try:
                variant = model_variant.ModelVariant.from_dict(payload)
                blob = json.dumps(variant.to_dict(), indent=2, sort_keys=True)
                with open(saved["path"], "w", encoding="utf-8") as handle:
                    handle.write(blob)
                saved["hash"] = hashlib.sha256(blob.encode()).hexdigest()
            except (KeyError, OSError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        missing = _missing_variant_artifacts(variant)
        models = set(model_loader.local_model_ids())
        compatible = not models or variant.base_model_id in models
        return {
            "path": saved["path"],
            "hash": saved["hash"],
            "runDirectory": saved["runDirectory"],
            "variant": variant.to_dict(),
            "missingArtifacts": missing,
            "compatibleWithServerModels": compatible,
        }

    @router.post("/api/model-variant/save")
    def save_variant(body: dict):
        from ..experiment import model_variant
        try:
            variant = model_variant.ModelVariant.from_dict(body)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=f"missing field {exc}")
        return model_variant.save_variant(variant)

    def _variant_from_body(body: dict):
        """Resolve the request's variant from EXACTLY ONE of two forms:

        - a stored artifact path (``variantPath``/``variantArtifactPath``/
          ``path`` + optional ``variantHash`` pin) — existing behavior, or
        - ``variant``: an inline spec dict (same schema as the
          /api/variants/upload payload) for ephemeral chat exploration.

        Returns ``(variant, request, model_id, metadata)``; the metadata is the
        honest provenance stamp for the response — inline specs stamp
        ``source: "inline"`` and never carry a path/hash, so an inline chat
        generation can never masquerade as a stored hash-pinned variant.
        """
        vpath = body.get("variantPath") or body.get("variantArtifactPath") or body.get("path")
        inline = body.get("variant")
        if inline is not None and not isinstance(inline, dict):
            raise HTTPException(status_code=400,
                                detail="variant must be an inline spec object")
        if (inline is None) == (not vpath):
            raise HTTPException(
                status_code=400,
                detail="exactly one of variantPath (stored artifact) or "
                       "variant (inline spec) is required")
        if inline is not None:
            try:
                variant = variant_chat.variant_from_dict(inline)
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(status_code=400,
                                    detail=f"invalid inline variant spec: {exc}")
            missing = _missing_variant_artifacts(variant)
            if missing:
                refs = ", ".join(
                    f"{m['kind']} {m['reference']!r} ({m['reason']})" for m in missing)
                raise HTTPException(
                    status_code=400,
                    detail=f"inline variant references missing artifacts: {refs}")
            request = variant_chat.request_from_body(body, resolved_variant_path="")
            metadata = variant_chat.inline_variant_metadata(variant)
        else:
            path = state.resolver.require_file(vpath, allow_local_absolute=True)
            try:
                variant, resolved = variant_chat.load_variant(path, body.get("variantHash"))
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            request = variant_chat.request_from_body(body, resolved_variant_path=resolved)
            metadata = variant_chat.variant_metadata(request.variant_path, variant)
        model_id = variant.base_model_id or body.get("baseModelID") or (
            state.model.model_id if state.model else None)
        if not model_id:
            raise HTTPException(status_code=400, detail="variant has no baseModelID")
        return variant, request, model_id, metadata

    @router.get("/api/variant/detail")
    def variant_detail(path: str):
        """Full stored variant spec + SHA-256 of the artifact file bytes, so a
        client can seed its steering controls from a stored variant (and later
        pin it via ``variantHash``). Read-only; the path is contained exactly
        like other variant reads."""
        from ..experiment import model_variant
        resolved = state.resolver.require_file(path, allow_local_absolute=True)
        with open(resolved, "rb") as handle:
            data = handle.read()
        try:
            spec = json.loads(data.decode("utf-8"))
            if not isinstance(spec, dict):
                raise ValueError("not a JSON object")
            model_variant.ModelVariant.from_dict(spec)  # must parse as a variant
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError,
                TypeError, ValueError) as exc:
            raise HTTPException(status_code=400,
                                detail=f"not a variant artifact: {exc}")
        return {"variant": spec, "hash": hashlib.sha256(data).hexdigest()}

    @router.post("/api/variant/generate")
    def variant_generate(body: dict):
        from ..experiment.prompt_render import ChatTemplateConstraintError
        variant, request, model_id, metadata = _variant_from_body(body)
        try:
            with state.acquire_model(model_id, variant.base_revision) as active_model:
                result = variant_chat.generate_with_variant(active_model, variant, request)
        except ChatTemplateConstraintError as exc:
            # Structured template-refusal 400 (see /api/generate).
            raise HTTPException(status_code=400, detail=exc.detail())
        except (ValueError, RuntimeError, model_loader.ModelLoadError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        result["variantMetadata"] = metadata
        return result

    @router.post("/api/variant/battery")
    def variant_battery(body: dict):
        """Score a pinned capability battery against a variant (privileged:
        loads and runs the model).

        The app's server-workspace robustness path used to take its battery
        readings as N ``/api/variant/generate`` calls, which forced two
        approximations a format-2 battery cannot survive: that wire generates
        text (it cannot read the model's distribution over an item's declared
        options) and it renders under the VARIANT's prompt mode and system
        prompt (``systemPrompt: null`` means "use the spec's", so "none" is
        not expressible). Both are the arming defect format 2 exists to fix,
        so the app refused v2 by name — and every seed battery is now v2
        (open issues §23). This route is the fix: one request per SIDE of the
        comparison, scored by ``battery.evaluate`` through the same back-ends
        ``experiment validate|run`` uses.

        Request (variant vocabulary identical to ``/api/variant/generate``):

        - exactly one of ``variantPath`` (+ optional ``variantHash``) or an
          inline ``variant`` spec;
        - ``stripInterventions`` — ``true`` IS the baseline side: the same
          spec with its injections/adapter removed, exactly as the generate
          wire's baseline arm. There is no separate no-variant form, because
          "the model without this intervention" is the comparison being made;
        - ``battery`` — workspace-relative path, contained like every other
          caller-named file input;
        - ``batteryHash`` — REQUIRED. The battery is a pinned input; a
          reading taken against a file that differs from the caller's bytes
          is not the reading it will report, so drift refuses here rather
          than silently measuring the server's copy.

        Response: per-item records in ``battery.jsonl``'s field vocabulary,
        the ``accuracy``/``itemCount``/``batteryHash`` summary, the arming
        provenance, and the contamination advisory slot.

        Format 1 is refused (see the refusal text): its arming is by
        definition the surrounding instrument's — the study manifest's, or
        the variant artifact's inside a run — and a bare variant reference is
        not an instrument. Choosing one here would invent a measurement that
        neither engine's run path performs. The app keeps scoring legacy
        batteries on the generate wire, unchanged.
        """
        from ..experiment import battery as battery_mod
        from ..experiment.prompt_render import ChatTemplateConstraintError

        variant, request, model_id, metadata = _variant_from_body(body)
        ref = str(body.get("battery") or "").strip()
        expected = str(body.get("batteryHash") or "").strip()
        if not ref:
            raise HTTPException(status_code=400,
                                detail="battery (workspace-relative path) is required")
        if not expected:
            raise HTTPException(
                status_code=400,
                detail="batteryHash is required — a capability reading taken "
                       "against an unpinned battery is not evidence")
        resolved = state.resolver.require_file(ref)
        try:
            spec = battery_mod.load_spec(resolved)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if spec.digest != expected:
            raise HTTPException(
                status_code=400,
                detail=f"capability battery '{ref}' drifted from the pinned "
                       f"hash (server has {spec.digest[:12]}…, caller pinned "
                       f"{expected[:12]}…) — the two sides are not scoring "
                       "the same file")
        if not spec.isolated:
            raise HTTPException(
                status_code=400,
                detail=f"capability battery '{ref}' is format "
                       f"{spec.format_version} (legacy): its arming is the "
                       "SURROUNDING INSTRUMENT's — a study manifest's or a "
                       "variant artifact's — and this wire has no instrument "
                       "to take it from, so scoring it here would be a "
                       "different measurement under the same pinned hash. "
                       "Score a legacy battery through /api/variant/generate "
                       "(the app's existing path), or inside "
                       "`steerlab-server experiment validate|run`. Re-pin a "
                       "format-2 battery (steerlab-server battery lint "
                       "<path>) to use this route.")
        try:
            with state.acquire_model(model_id, variant.base_revision) as active_model:
                result = variant_chat.evaluate_battery_with_variant(
                    active_model, variant, spec,
                    strip_interventions=request.strip_interventions)
        except ChatTemplateConstraintError as exc:
            raise HTTPException(status_code=400, detail=exc.detail())
        except (ValueError, RuntimeError, model_loader.ModelLoadError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        result["battery"] = ref
        result["variantMetadata"] = metadata
        return result

    @router.post("/api/variant/generate/stream")
    def variant_generate_stream(body: dict):
        variant, request, model_id, metadata = _variant_from_body(body)

        resolved_model_id = model_id

        # Say so BEFORE the silence: when this chat's base model is not
        # resident, acquire_model inside `produce` performs (or queues behind)
        # a cold load that takes minutes with no tokens to show. The preamble
        # status event is the client's only signal that the wait is a load,
        # not a hang (live 2026-07-17), and _locked_sse heartbeats it with
        # elapsed time until the first token.
        residency = state.registry.residency(
            model_id, variant.base_revision,
            dtype=state.default_dtype, device=state.default_device)
        preamble = None
        if residency != "ready":
            phase = ("is still loading — waiting for it"
                     if residency == "loading"
                     else "is not loaded yet — loading it now; cold loads "
                          "can take minutes")
            preamble = {"status": f"model {model_id} {phase}"}

        def produce(stop):
            nonlocal resolved_model_id
            with state.acquire_model(model_id, variant.base_revision) as active_model:
                resolved_model_id = active_model.model_id
                if preamble is not None:
                    # The load (or the wait behind one) is over — mark the
                    # transition so the client's caption moves from "loading"
                    # to "generating" and a stall in EITHER phase is
                    # attributable (second live round, 2026-07-17).
                    yield {"status": f"model {active_model.model_id} ready — "
                                     "generating"}
                yield from variant_chat.stream_with_variant(
                    active_model, variant, request, should_stop=stop)

        def done_extra():
            return {"modelID": resolved_model_id, "variant": metadata}

        return StreamingResponse(_locked_sse(produce, done_extra, preamble),
                                 media_type="text/event-stream")

    @router.get("/api/adapters")
    def list_adapters():
        # LoRA adapters are run dirs containing an adapter_model.safetensors.
        import os as _os
        from ..experiment import model_variant as _model_variant
        runs = paths.runs_directory()
        out = []
        if _os.path.isdir(runs):
            for entry in sorted(_os.listdir(runs), reverse=True):
                for root_dir, _dirs, files in _os.walk(_os.path.join(runs, entry)):
                    if "adapter_model.safetensors" in files or "adapter_config.json" in files:
                        item = {"name": _os.path.basename(root_dir), "adapterDirectory": root_dir}
                        # Cross-engine contract stamps from the sidecar next to
                        # the adapter dir (absent on legacy adapters).
                        sidecar = _model_variant.adapter_sidecar(root_dir) or {}
                        # v1 keys, then the v2 provenance a variant attachment
                        # and the freeze pin surface need (readiness contract
                        # §7): what tier the adapter is, what it trained on,
                        # and the byte hashes that make "this adapter" checkable.
                        for key in ("substrate", "adapterFormat", "schemaVersion",
                                    "evidenceGrade", "trainingMode",
                                    "adapterBytesHash", "adapterConfigHash"):
                            if sidecar.get(key) is not None:
                                item[key] = sidecar[key]
                        dataset = sidecar.get("dataset")
                        if isinstance(dataset, dict) and dataset.get("manifestHash"):
                            item["datasetManifestHash"] = dataset["manifestHash"]
                        selected = sidecar.get("selectedCheckpoint")
                        if isinstance(selected, dict) and selected.get("step") is not None:
                            item["selectedCheckpoint"] = {"step": selected["step"]}
                        out.append(item)
                        break
        return {"adapters": out}

    def _finetune_is_v2(body: dict) -> bool:
        """A v2 body announces itself (``schemaVersion``) or carries the
        versioned dataset bundle. Everything else is the historical v1 inline
        corpus, whose behavior is preserved byte-for-byte."""
        return body.get("schemaVersion") is not None or "dataset" in body

    @router.post("/api/finetune/plan")
    def finetune_plan(body: dict):
        """Normalize a v2 request into the training PLAN a researcher confirms
        before a queue allocation is spent (readiness plan §3). No side
        effects, no tokenizer, no model load: inline dataset bytes are
        hash-verified in a temp tree and never persisted."""
        from . import finetune_submission as _ft
        try:
            return _ft.plan_response(body, resolver=state.resolver)
        except _ft.FineTuneRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/api/finetune/submit")
    def finetune_submit(body: dict):
        """The EVIDENCE-GRADE fine-tune path: a Slurm job with bundle,
        LoRA preflight (memory AND walltime), checkpoint/resume and
        auto-resubmit (readiness plan §0 amendment 3)."""
        from . import finetune_submission as _ft
        try:
            return _ft.submit_finetune(body, jobs=state.jobs,
                                       resolver=state.resolver)
        except _ft.PreflightRejection as exc:
            raise HTTPException(status_code=400,
                                detail={"message": str(exc),
                                        "preflight": exc.preflight})
        except _ft.FineTuneRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/api/finetune/train")
    def finetune_train(body: dict):
        from ..experiment import lora_train
        import os as _os
        if _finetune_is_v2(body):
            return _finetune_train_v2(body)
        # Training documents are caller-named files: contain them like every
        # other artifact reference instead of opening raw host paths.
        documents = [
            state.resolver.require_file(p, allow_local_absolute=True)
            for p in (body.get("documentPaths") or [])
        ]
        if body.get("text"):  # inline corpus → write to a temp doc in a run dir
            run_dir = paths.make_unique_run_directory("lora-docs")
            doc = _os.path.join(run_dir, "corpus.txt")
            with open(doc, "w", encoding="utf-8") as handle:
                handle.write(body["text"])
            documents.append(doc)
        if not documents:
            raise HTTPException(status_code=400, detail="provide 'text' or 'documentPaths'")
        if not body.get("baseModelID"):
            raise HTTPException(status_code=400, detail="baseModelID required")
        config = lora_train.LoRAConfig(
            base_model_id=body["baseModelID"], document_paths=documents,
            rank=int(body.get("rank", 8)), alpha=float(body.get("alpha", 16.0)),
            iterations=int(body.get("iterations", 200)),
            learning_rate=float(body.get("learningRate", 1e-4)),
            output_name=body.get("name"))

        def work(job):
            # LoRA loads its own model via peft (not a registry slot), so it uses
            # the coarse GPU lock rather than a slot lock.
            with state.exclusive_gpu:
                run_dir = lora_train.train(config, log=job.log)
            job.log(f"LoRA adapter → {run_dir}")
            return {"runDirectory": run_dir}

        return {"jobId": state.jobs.submit("lora-train", work).id}

    def _finetune_train_v2(body: dict):
        """The v2 body on the DAEMON route — exploratory only.

        Evidence-grade training is refused here on purpose: a daemon-resident
        job has no checkpoint/resume, no auto-resubmit, and occupies the
        controller for the length of the run (readiness plan §0 amendment 3).
        """
        from . import finetune_submission as _ft
        from ..experiment import lora_train
        try:
            parsed = _ft.parse_request(body)
        except _ft.FineTuneRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if parsed.evidence_grade:
            refusals = lora_train.evidence_refusals(parsed.config)
            detail = (
                "evidence-grade LoRA training may not run on the "
                "daemon-resident route (/api/finetune/train): it has no "
                "checkpoint/resume, no auto-resubmit, and holds the "
                "controller for the whole run — submit it as a Slurm job via "
                "POST /api/finetune/submit")
            if refusals:
                detail += ("; this configuration would additionally be "
                           "refused for: " + "; ".join(refusals))
            raise HTTPException(status_code=400, detail=detail)
        dataset_root = paths.make_unique_run_directory("lora-dataset")
        try:
            _ft.stage_dataset(parsed, dataset_root, state.resolver)
        except _ft.FineTuneRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        config = parsed.config
        config.dataset_root = dataset_root

        def work(job):
            with state.exclusive_gpu:
                run_dir = lora_train.train(config, log=job.log)
            job.log(f"LoRA adapter → {run_dir}")
            return {"runDirectory": run_dir, "datasetRoot": dataset_root}

        return {"jobId": state.jobs.submit("lora-train", work).id}

    # --- Gemma Scope (SAE cross-check) --------------------------------------

    @router.get("/api/gemmascope/info")
    def gemmascope_info(model: str | None = None):
        from ..experiment import gemma_scope
        mid = model or (state.model.model_id if state.model else "")
        lc = state.model.num_layers if state.model else None
        return gemma_scope.scope_info(mid, layer_count=lc)

    @router.post("/api/gemmascope/run")
    def gemmascope_run(body: dict):
        from ..experiment import gemma_scope
        vp, nm = body.get("vectorPath"), body.get("name")
        if not vp or not nm:
            raise HTTPException(status_code=400, detail="vectorPath + name required")
        # `nm` becomes a run-directory slug below (`gemmascope-<nm>`); a
        # separator in it would place the run outside runs/.
        _safe_name(str(nm))
        vp = state.resolver.require_dir(vp, allow_local_absolute=True)
        info = gemma_scope.scope_info(body.get("modelID", ""),
                                      preferred_layer=body.get("layer"))
        release = body.get("release") or info.get("release")
        sae_id = body.get("saeID") or info.get("saeID")
        layer = int(body.get("layer", info.get("layer", 0)))
        if not release or not sae_id:
            raise HTTPException(status_code=400, detail="not a Gemma 3 model / no SAE available")

        def work(job):
            gs_dir = paths.make_unique_run_directory(f"gemmascope-{nm}")
            from ..experiment.run_config import write_run_config
            write_run_config(gs_dir, "gemmascope-analysis",
                             model_id=body.get("modelID") or None)
            out = os.path.join(gs_dir, "gemmascope-report.json")
            report = gemma_scope.analyze(vp, nm, layer=layer, release=release,
                                         sae_id=sae_id, output_path=out)
            job.log(f"top feature {report.top_absolute[0].feature} "
                    f"cos {report.top_absolute[0].cosine:.3f} → {out}")
            return {"reportPath": out,
                    "topAbsolute": [r.feature for r in report.top_absolute[:10]]}

        return {"jobId": state.jobs.submit("gemmascope", work).id}

    @router.post("/api/gemmascope/import")
    def gemmascope_import(body: dict):
        from ..experiment import gemma_scope
        model = state.require_model()
        report_path, feature = body.get("reportPath"), body.get("feature")
        if not report_path or feature is None:
            raise HTTPException(status_code=400, detail="reportPath + feature required")
        report_path = state.resolver.require_file(report_path, allow_local_absolute=True)
        try:
            feature = int(feature)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="feature must be an integer")
        run_dir = paths.make_unique_run_directory(f"sae-feature-{feature}")
        from ..experiment.run_config import write_run_config
        write_run_config(run_dir, "sae-feature-import", model_id=model.model_id,
                         revision=model.revision)
        try:
            path = gemma_scope.import_feature(report_path, int(feature),
                                              model_id=model.model_id, run_directory=run_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"vectorPath": run_dir, "name": os.path.basename(path)}

    @router.post("/api/gemmascope/import-id")
    def gemmascope_import_id(body: dict):
        """Import a Gemma Scope feature BY ID — no cosine report required
        (proposal r2 §8 P0-1). CLI twin: ``steerlab-server gemmascope
        import-id``.

        Privileged (``app._PRIVILEGED_PREFIXES``): it goes ONLINE to Hugging
        Face like ``/api/jlens/lenses/acquire``, reads a caller-named artifact,
        and writes under ``runs/``.

        The heavy part (SAE download) runs as a job; everything cheap and
        refusable is checked synchronously so a mis-specified import fails as a
        400 instead of a job that dies minutes later.
        """
        from ..experiment import gemma_scope
        required = ("model", "release", "saeID", "feature", "label",
                    "residualNormArtifact")
        missing = [k for k in required if body.get(k) in (None, "")]
        if missing:
            raise HTTPException(status_code=400,
                                detail=f"missing: {', '.join(missing)}")
        try:
            feature = int(body["feature"])
            layer = int(body["layer"]) if body.get("layer") is not None else None
        except (TypeError, ValueError):
            raise HTTPException(status_code=400,
                                detail="feature/layer must be integers")
        reference = str(body["residualNormArtifact"])
        resolved = paths.resolve_artifact(reference)
        if not os.path.isfile(resolved + ".json"):
            raise HTTPException(
                status_code=400,
                detail=(f"residualNormArtifact {reference!r} names no vector "
                        f"artifact (expected {resolved}.json) — pass the base "
                        f"path <runDir>/<name>, no extension"))

        def work(job):
            run_dir = paths.make_unique_run_directory(f"sae-feature-{feature}")
            from ..experiment.run_config import write_run_config
            write_run_config(run_dir, "sae-feature-import",
                             model_id=str(body["model"]), job_id=job.id)
            path = gemma_scope.import_feature_by_id(
                model_id=str(body["model"]), release=str(body["release"]),
                sae_id=str(body["saeID"]), feature=feature,
                label=str(body["label"]), residual_norm_artifact=reference,
                run_directory=run_dir, layer=layer,
                neuronpedia_url=body.get("neuronpediaURL"),
                name=body.get("name"))
            job.log(f"SAE feature {feature} → {path} "
                    f"({gemma_scope.RESIDUAL_NORM_MATCH_CONVENTION})")
            return {"vectorPath": run_dir, "name": os.path.basename(path),
                    "artifact": path}

        return {"jobId": state.jobs.submit("gemmascope-import-id", work).id}

    # --- multi-agent scenarios ----------------------------------------------

    @router.get("/api/scenarios")
    def scenarios():
        from ..experiment import multi_agent
        return {"scenarios": multi_agent.list_scenarios()}

    @router.get("/api/scenario")
    def scenario_get(path: str):
        from ..experiment import multi_agent
        path = state.resolver.require_file(path, allow_local_absolute=True)
        scenario, _ = multi_agent.load_scenario(path)
        return multi_agent._scenario_to_dict(scenario)

    @router.post("/api/scenario/save")
    def scenario_save(body: dict):
        from ..experiment import multi_agent
        try:
            scenario = multi_agent.Scenario.from_dict(body)
            multi_agent.validate(scenario)
        except (KeyError, multi_agent.ScenarioError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return multi_agent.save_scenario(scenario)

    @router.post("/api/scenario/run")
    def scenario_run(body: dict):
        from ..experiment import multi_agent
        path = body.get("path")
        strip = bool(body.get("stripInterventions", False))
        if not path:
            raise HTTPException(status_code=400, detail="scenario 'path' required")
        path = state.resolver.require_file(path, allow_local_absolute=True)

        base_model = state.require_model()

        def work(job):
            scenario, shash = multi_agent.load_scenario(path)
            run_dir = paths.make_unique_run_directory(
                f"multi-agent-{scenario.name}-{'baseline' if strip else 'configured'}")
            from ..experiment.run_config import write_run_config
            write_run_config(run_dir, "multi-agent", model_id=base_model.model_id,
                             revision=base_model.revision)
            # Each turn acquires its own model's slot lock through the provider;
            # base_model is only the fallback id for turns without an explicit
            # base. No coarse lock is held across turns.
            multi_agent.run_scenario(
                base_model, scenario, run_dir=run_dir,
                condition_name="baseline" if strip else "configured",
                strip_interventions=strip, scenario_hash=shash, log=job.log,
                model_provider=state.acquire_model)
            return {"runDirectory": run_dir, "turns": len(scenario.turns)}

        return {"jobId": state.jobs.submit("multi-agent", work).id}

    # --- experiment authoring (firewall WRITE side) -------------------------
    # The server is now an authoring authority too: it stamps frozen manifests
    # with frozenBy:"server" + a content hash; cross-engine verification rests on
    # the stimulus/corpus SHA-256s (identical across engines). Frozen manifests
    # are read-only here — duplicate to iterate.

    @router.post("/api/authoring/create")
    def authoring_create(body: dict):
        from ..experiment import experiment_store
        if not body.get("name") or not body.get("modelID"):
            raise HTTPException(status_code=400, detail="name + modelID required")
        try:
            return experiment_store.create(
                body["name"], model_id=body["modelID"], revision=body.get("revision"),
                description=body.get("description", ""))
        except experiment_store.ExperimentStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    def _authoring(fn, name, *args):
        from ..experiment.experiment_store import ExperimentStoreError
        try:
            return fn(name, *args)
        except ExperimentStoreError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @router.post("/api/authoring/{name}/attach")
    def authoring_attach(name: str, body: dict):
        _safe_name(name)
        from ..experiment import experiment_store
        # method "pinnedArtifact" takes ONE concept plus "vectorArtifact"
        # (the extension-less locator, e.g. a catalog row's
        # workspaceRelativeID) and optional "sourceConcept": the artifact's
        # sidecar supplies everything else, and both file hashes are computed
        # and pinned here.
        #
        # "extractionRendering" is additive and optional: the object the
        # schema documents ({"mode": "chatTemplate", …}) or a bare mode
        # string. Absent — and an explicit {"mode": "raw"} — write nothing,
        # so a body that never mentions it produces the manifest bytes it
        # always did. A malformed declaration is the store's typed refusal,
        # which `_authoring` already turns into a 400 carrying the repair.
        from ..steering.extraction_rendering import ExtractionRenderingError
        try:
            return _authoring(
                lambda n: experiment_store.attach(
                    n, body.get("concepts", []),
                    method=body.get("method", "meanDifference"),
                    pool_from_token=body.get("poolFromToken"),
                    corpus_concepts=body.get("corpusConcepts"),
                    reference=body.get("reference"),
                    vector_artifact=body.get("vectorArtifact"),
                    source_concept=body.get("sourceConcept"),
                    eval_run=body.get("evalRun"),
                    extraction_rendering=body.get("extractionRendering")), name)
        except ExtractionRenderingError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/api/authoring/{name}/protocol")
    def authoring_protocol(name: str, body: dict):
        _safe_name(name)
        from ..experiment import experiment_store
        return _authoring(lambda n: experiment_store.set_protocol(n, body), name)

    @router.post("/api/authoring/{name}/condition")
    def authoring_condition(name: str, body: dict):
        _safe_name(name)
        from ..experiment import experiment_store
        return _authoring(lambda n: experiment_store.add_condition(n, body), name)

    @router.post("/api/authoring/{name}/condition/remove")
    def authoring_condition_remove(name: str, body: dict):
        _safe_name(name)
        from ..experiment import experiment_store
        return _authoring(lambda n: experiment_store.remove_condition(n, body["name"]), name)

    @router.post("/api/authoring/{name}/duplicate")
    def authoring_duplicate(name: str, body: dict):
        _safe_name(name)
        from ..experiment import experiment_store
        return _authoring(lambda n: experiment_store.duplicate(n, body["newName"]), name)

    @router.post("/api/authoring/{name}/freeze")
    def authoring_freeze(name: str, body: dict | None = None):
        _safe_name(name)
        from ..experiment import experiment_store

        def run(n):
            frozen = experiment_store.freeze(
                n, force=bool((body or {}).get("force", False)),
                cached_revision=model_loader.cached_revision)
            # Additive response key only — advisories are non-blocking and
            # never persist into the manifest (stderr already carries them
            # for CLI parity; this surfaces them to HTTP clients).
            advisories = experiment_store.freeze_advisories(frozen)
            return {**frozen, "advisories": advisories} if advisories else frozen

        return _authoring(run, name)

    # --- contract stubs -----------------------------------------------------
    # Only legacy aliases remain stubbed.
    for path in ("/api/concept/save",):
        def _stub(path=path):
            def handler():
                raise HTTPException(
                    status_code=501,
                    detail=f"{path} not wired on the cluster yet; "
                           "manifest/concept authoring stays Swift/UI-side for now")
            return handler
        router.add_api_route(path, _stub(), methods=["POST"])

    return router
