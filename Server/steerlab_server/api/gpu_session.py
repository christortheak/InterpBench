"""GPU-session lifecycle (docs/GPU-SESSION-PLAN.md, Wave 1 — server side).

One sentence of architecture (plan §2): the CONTROLLER stays the single front
door — it acquires a role-scoped, self-expiring GPU worker job through the
ordinary JobManager/SlurmExecutor machinery (so maintenance-window preflight,
poll seams, scancel, and durable job records all come free) and
reverse-proxies the interactive route families to it. One client URL, one
token; the browser workbench needs no second tunnel.

This module holds all three sides of that contract:

- the controller-owned SESSION RECORD (``<metadata_root>/session.json``,
  atomic tmp+mv) plus the lifecycle verbs the routes call
  (``start_session`` / ``reconcile_session`` / ``end_session``);
- the WORKER side: the discovery record
  (``<metadata_root>/session-worker.json``), the idle timer that only counts
  REAL work (plan §2.3 — acceptance criterion 1), and graceful self-expiry;
- the STREAMING reverse proxy with its explicit allowlist (plan §2.1).

The metadata root is shared (/home-side), so both processes see the records;
the ``sessionGeneration`` UUID guards every cross-process read against
yesterday's allocation (plan §2.5).
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import socket
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .. import __version__
from ..experiment import paths
from .executors import SlurmExecutor, SlurmResources, _parse_walltime
from .profile import ServerProfile


SESSION_JOB_KIND = "session:gpu"

# The full cross-engine session-record contract (camelCase, closed keys —
# tested by a key-list test like config.json schema 2). Response-only fields
# (idleRemainingSeconds, busy) are deliberately NOT in the record.
# 2026-07-16 (first live shakedown): added "submittedAt" (anchors the
# accounting-visibility grace window — a site's sacct can lag freshly submitted
# jobs) and "stateDetail" (actionable text for failed/ending states, naming
# the Slurm job id so the operator can check by hand; null otherwise).
SESSION_RECORD_KEYS = (
    "sessionGeneration", "slurmJobID", "node", "port", "state",
    "workspaceRoot", "gpuType", "gres", "partition", "walltime",
    "idleMinutes", "submittedAt", "startedAt", "expiresAt", "serverVersion",
    "role", "stateDetail",
)

# "unknown" (2026-07-16 follow-up review): the scheduler conversation broke
# down — sacct/squeue stopped answering for the job (or never answered past
# the grace window) and the worker is unreachable. That is evidence about the
# SCHEDULER CONVERSATION, not the allocation, so it is deliberately
# NON-terminal: it keeps the double-start guard occupied (a billed orphan
# must never hide behind a fresh session). Exits: a positive terminal report
# from the scheduler, the worker answering again (recovery), or an explicit
# operator clear (DELETE /api/session with force after a by-hand check).
SESSION_STATES = {"queued", "starting", "ready", "busy", "idle", "ending",
                  "unknown", "ended", "failed"}
TERMINAL_SESSION_STATES = {"ended", "failed"}

# Session-shape fallbacks — the LAST link of each chain, never the first.
#
# WP5 Step 9 (audit c20): these were the only source, which made one
# institution's session shape every site's. A session now resolves each field
# as  request parameter → this site's `scheduler.gpuSession` declaration (the
# STEERLAB_SESSION_* keys the site-profile renderer emits into the env file
# this controller sources) → the site's study-job value where one exists
# (STEERLAB_SLURM_GRES / _MEMORY / _PARTITION, via SlurmResources.from_env) →
# the constant below. Only a site that declared nothing at all reaches these.
#
# Dispositions, matching the renderer's LegacyDefaults table:
#   * DEFAULT_CPUS — ENGINE-GENERIC. Eight cores feed one GPU's dataloader
#     anywhere; it is not a claim about anyone's hardware.
#   * DEFAULT_IDLE_MINUTES — ENGINE-GENERIC. A chat session's patience is a
#     product decision, not a site fact.
#   * DEFAULT_WALLTIME / DEFAULT_MEMORY / DEFAULT_GRES — v1-legacy, i.e.
#     one-site-shaped. They stay because a GPU session must ask for SOMETHING
#     and refusing here would break every site that has not yet declared a
#     session block; a site says otherwise in `scheduler.gpuSession` (or, for
#     memory/gres, in its study defaults, which already win over these).
DEFAULT_IDLE_MINUTES = 30
DEFAULT_WALLTIME = "02:00:00"
DEFAULT_MEMORY = "64G"
DEFAULT_CPUS = 8
# DEFAULT_PORT is only the fallback when a port must be materialized OUTSIDE
# a Slurm allocation (manual/dev worker runs with no SLURM_JOB_ID). Session
# starts default to "auto" — see derive_session_port below.
DEFAULT_PORT = 8081
DEFAULT_GRES = "gpu:A100:1"

# COLLISION-RESISTANT (not collision-proof) worker port: a FIXED port
# collides the moment two session jobs land on the same multi-GPU node — the
# second worker fails to bind, or the controller dials the wrong allocation's
# worker. With STEERLAB_SESSION_PORT "auto" (or unset) the WORKER derives its
# port from its own SLURM_JOB_ID and writes the ACTUAL port into the
# discovery record, which reconcile and the proxy already prefer over the
# session record's port. Residual risk, accepted for now (engineer review
# 2026-07-17): jobs exactly SESSION_PORT_SPAN ids apart, or an unrelated
# process squatting the derived port, still collide — the durable fix is
# bind-with-retry advertising the bound port, which the discovery-record
# plumbing already supports when someone implements it.
SESSION_PORT_BASE = 10000
SESSION_PORT_SPAN = 50000


def derive_session_port() -> int:
    """``10000 + (SLURM_JOB_ID % 50000)`` — the allocation-unique worker
    port. Outside Slurm (no/garbled SLURM_JOB_ID: manual dev runs) falls back
    to DEFAULT_PORT."""
    raw = (os.environ.get("SLURM_JOB_ID") or "").strip()
    if not raw.isdigit():
        return DEFAULT_PORT
    return SESSION_PORT_BASE + (int(raw) % SESSION_PORT_SPAN)

# Accounting-visibility grace (first live shakedown, 2026-07-16): that site's
# sacct/squeue can lag freshly submitted jobs, so "the scheduler doesn't know
# this job" is NOT proof it failed — reading it that way "ended" two PENDING
# sessions that then ran orphaned on billed GPUs. Within this window of
# submittedAt an unknown job stays "queued"; only past it does unknown mean
# failed (with a stateDetail naming the job id for a by-hand check).
#
# SITE DATA since WP5 Step 9 (audit c22): the window is how long THIS site's
# accounting may lag, declared as `scheduler.accountingVisibilityGraceSeconds`
# and rendered into the env file below as
# STEERLAB_SESSION_VISIBILITY_GRACE_SECONDS. 600 is the v1-legacy value this
# number was measured at, kept as the no-declaration fallback.
DEFAULT_VISIBILITY_GRACE_SECONDS = 600


def _session_env(key: str) -> str | None:
    """One declared `scheduler.gpuSession` field, from the rendered site env."""
    raw = (os.environ.get(key) or "").strip()
    return raw or None


def _session_int_env(key: str, low: int, high: int) -> int | None:
    """A declared numeric session field, bounded by the same limits the request
    parameter obeys. A site value outside them is an authoring error the
    operator must see, not a number to silently clamp."""
    raw = _session_env(key)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise _bad_param(f"{key} {raw!r} is not an integer")
    if not low <= value <= high:
        raise _bad_param(f"{key} {value} is outside {low}-{high}")
    return value


def _visibility_grace_seconds() -> float:
    raw = os.environ.get("STEERLAB_SESSION_VISIBILITY_GRACE_SECONDS", "")
    try:
        return float(raw) if raw.strip() else float(DEFAULT_VISIBILITY_GRACE_SECONDS)
    except ValueError:
        return float(DEFAULT_VISIBILITY_GRACE_SECONDS)


def _within_visibility_grace(submitted_at_iso: str | None) -> bool:
    """True while a scheduler-unknown job might still be accounting lag.
    A missing/garbled submittedAt (legacy records) grants NO grace — that
    keeps the pre-2026-07-16 semantics for records that predate the field."""
    if not submitted_at_iso:
        return False
    elapsed = _elapsed_since(submitted_at_iso)
    return elapsed is not None and elapsed < _visibility_grace_seconds()


def _elapsed_since(iso_timestamp: str | None) -> float | None:
    if not iso_timestamp:
        return None
    try:
        then = datetime.fromisoformat(
            str(iso_timestamp).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return (datetime.now(timezone.utc) - then).total_seconds()


# Probe-miss hysteresis (live 2026-07-17): a worker under heavy GPU work (a
# cold model load, a long decode) can miss a single short-timeout probe while
# being perfectly alive, and demoting the session to "Starting" on one miss
# made the state flap Ready↔Starting through every load. A session that
# answered a probe within this window keeps its last reachable state on a
# miss; only a sustained silence demotes it.
DEFAULT_PROBE_LAPSE_SECONDS = 180


def _probe_lapse_seconds() -> float:
    raw = os.environ.get("STEERLAB_SESSION_PROBE_LAPSE_SECONDS", "")
    try:
        return (float(raw) if raw.strip()
                else float(DEFAULT_PROBE_LAPSE_SECONDS))
    except ValueError:
        return float(DEFAULT_PROBE_LAPSE_SECONDS)


def _within_probe_lapse(last_reachable_iso: str | None) -> bool:
    """True while a probe miss may still be load-induced latency rather than a
    dead worker. No lastReachableAt (never yet reached) grants NO grace."""
    elapsed = _elapsed_since(last_reachable_iso)
    return elapsed is not None and elapsed < _probe_lapse_seconds()


# Running-but-unreachable diagnosis (2026-07-16 follow-up review): compute→
# compute HTTP reachability is exactly the UNVERIFIED assumption of the live
# shakedown (plan §5). A job that Slurm reports running while the worker
# probe keeps failing would otherwise read a bare "starting" forever — after
# this many seconds the state stays "starting" (nonterminal) but stateDetail
# spells out the manual check.
DEFAULT_STARTING_DIAGNOSIS_SECONDS = 180


def _starting_diagnosis_seconds() -> float:
    raw = os.environ.get("STEERLAB_SESSION_STARTING_DIAGNOSIS_SECONDS", "")
    try:
        return (float(raw) if raw.strip()
                else float(DEFAULT_STARTING_DIAGNOSIS_SECONDS))
    except ValueError:
        return float(DEFAULT_STARTING_DIAGNOSIS_SECONDS)


def _starting_diagnosis(record: dict) -> str | None:
    """stateDetail for a job running longer than the diagnosis threshold with
    the worker still unreachable, else None (which also clears stale text)."""
    elapsed = _elapsed_since(record.get("startedAt"))
    if elapsed is None or elapsed < _starting_diagnosis_seconds():
        return None
    job_id = record.get("slurmJobID")
    node = record.get("node")
    port = record.get("port")
    if node and port:
        return (f"Slurm reports job {job_id} running for {int(elapsed)}s but "
                f"the worker at {node}:{port} is not answering — from a "
                f"login shell try `curl http://{node}:{port}/api/"
                f"capabilities`; if this never answers, compute-to-compute "
                "HTTP may be blocked on this cluster (the proxy-reachability "
                "assumption in GPU-SESSION-PLAN §5). Also check the job's "
                "slurm-*.err log in its bundle directory")
    return (f"Slurm reports job {job_id} running for {int(elapsed)}s but the "
            "worker has not written its discovery record — check the job's "
            "slurm-*.err log in its bundle directory (the worker may have "
            "failed at startup)")


def _auth_mismatch_detail(record: dict) -> str:
    """stateDetail for a worker that ANSWERED the probe with 401/403 (live
    2026-07-17: a healthy, token-enforcing worker was read as "not answering"
    because the unauthenticated probe was rejected). This is a CONFIG bug the
    operator must see — the remedy is aligning tokens, not waiting."""
    node = record.get("node")
    port = record.get("port")
    return (f"worker at {node}:{port} is up but rejected the controller's "
            "bearer token — controller and worker must share "
            "STEERLAB_AUTH_TOKEN (token file mismatch?). Both processes "
            "hydrate from STEERLAB_AUTH_TOKEN_FILE (default "
            "~/.steerlab-token); restart whichever side holds the stale "
            "token")


def _unknown_detail(record: dict) -> str:
    job_id = record.get("slurmJobID")
    return (f"lost sight of Slurm job {job_id}: the scheduler does not "
            f"report it and the worker is not answering — the allocation "
            f"may still be running. Check `sacct -j {job_id}` (and "
            f"`squeue -j {job_id}`) by hand; the session recovers by itself "
            "if the worker or scheduler answers again, or clear it with "
            "DELETE /api/session?force=1 after verifying the job is dead "
            f"(`scancel {job_id}` first if it is still running)")

# The worker reports "ready" right after activity and "idle" once it has
# been quiet for a while — pure presentation semantics for the session
# control ("Ready" vs "Idle 18m"); the countdown itself is the honest number.
_READY_GRACE_SECONDS = 30.0

WORKER_DOWN_DETAIL = ("GPU session worker not answering — it may have "
                      "expired; check GET /api/session")

NO_SESSION_DETAIL = ("no GPU session running — start one "
                     "(POST /api/session/start or the app's GPU Session "
                     "control)")


class SessionConflict(Exception):
    """A live session already exists; carries its record so the route can
    return it in the 409 body (idempotent double-start, plan §2.5)."""

    def __init__(self, session: dict):
        super().__init__("a GPU session already exists")
        self.session = session


# --- record I/O (atomic, generation-guarded) ---------------------------------


def _metadata_root() -> str:
    return ServerProfile.from_env().metadata_root


def session_record_path() -> str:
    return os.path.join(_metadata_root(), "session.json")


def worker_record_path() -> str:
    return os.path.join(_metadata_root(), "session-worker.json")


def _write_json_atomic(path: str, record: dict) -> None:
    """tmp+mv like every other record we write: a reader (the other process,
    over NFS) must never see a half-written JSON document."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".session-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def read_session_record() -> dict | None:
    return _read_json(session_record_path())


def write_session_record(record: dict) -> None:
    _write_json_atomic(session_record_path(), record)


def read_worker_record(expected_generation: str | None) -> dict | None:
    """The worker's discovery record, ONLY when its generation matches the
    session record's — a leftover record from yesterday's allocation must
    never be trusted for node/port (stale-record guard, plan §2.5)."""
    record = _read_json(worker_record_path())
    if record is None or not expected_generation:
        return None
    if str(record.get("sessionGeneration") or "") != str(expected_generation):
        return None
    return record


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _lifecycle_lock():
    """Cross-PROCESS mutual exclusion for the session lifecycle's
    check-then-submit and end verbs (fcntl.flock on a lockfile in the
    metadata root, blocking). An in-process threading lock is not enough: a
    restarted controller's successor, a second uvicorn worker, or a CLI call
    would all race the same session.json — and two concurrent starts that
    both pass the "no live session" check submit TWO billed GPU workers."""
    root = _metadata_root()
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "session.lock"), "w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# --- worker side: idle lifecycle (plan §2.3) ---------------------------------

# Requests that count as REAL activity on the worker. Load-bearing: the app
# polls /api/capabilities every ~15 seconds, so "any API traffic" as the
# signal would hold a billed GPU to walltime forever. Health checks,
# capability/status/state polling, and catalog GETs are deliberately absent.
_ACTIVITY_EXACT = frozenset({
    "/api/load", "/api/load/stream", "/api/models/unload",
    "/api/generate", "/api/generate/stream",
    "/api/variant/generate", "/api/variant/generate/stream",
    # Scores a whole capability battery against a variant — the app's
    # robustness path for a format-2 battery. Real GPU work, so it must reset
    # the idle timer exactly as the generate wire it replaced does.
    "/api/variant/battery",
    "/api/reader/score", "/api/reader/fit",
    "/api/extract", "/api/multiconcept/extract",
    "/api/session/keepalive",
})


def is_activity(path: str) -> bool:
    if path in _ACTIVITY_EXACT:
        return True
    # Concept-scoped compute verbs (stats/probe-train/extract run the model);
    # concept catalog/authoring GETs and saves do not hold a GPU.
    if path.startswith("/api/concept/") and path.endswith(
            ("/stats", "/probe-train", "/extract")):
        return True
    return False


class IdleTimer:
    """Idle expiry that only real work resets (acceptance criterion 1).

    ``begin``/``end`` bracket every allowlisted request — including the full
    duration of a streamed generation — via an in-flight counter, and the
    timer can NEVER fire while that counter is above zero: an active
    generation blocks expiry no matter how long it runs. ``touch`` is the
    explicit "Keep session" ping. ``idle_seconds <= 0`` disables the timer.

    The class is pure (injectable clock, ``check()`` drives it) so tests can
    exercise the arithmetic without threads or sleeps; ``WorkerSession`` owns
    the polling thread.
    """

    def __init__(self, idle_seconds: float, on_expire, *,
                 clock=time.monotonic):
        self.idle_seconds = max(0.0, float(idle_seconds))
        self._on_expire = on_expire
        self._clock = clock
        self._lock = threading.Lock()
        self._inflight = 0
        self._last_activity = clock()
        self._expired = False

    @property
    def enabled(self) -> bool:
        return self.idle_seconds > 0

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._inflight > 0

    @property
    def last_activity(self) -> float:
        with self._lock:
            return self._last_activity

    def begin(self) -> None:
        with self._lock:
            self._inflight += 1
            self._last_activity = self._clock()

    def end(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)
            self._last_activity = self._clock()

    def touch(self) -> None:
        with self._lock:
            self._last_activity = self._clock()

    def idle_elapsed(self) -> float:
        with self._lock:
            if self._inflight > 0:
                return 0.0
            return self._clock() - self._last_activity

    def remaining_seconds(self) -> int | None:
        """Countdown for the UI ("Idle 18m"), None when disabled. While a
        request is in flight the countdown reads full — honest, because the
        clock only starts when the last in-flight request ends."""
        if not self.enabled:
            return None
        return max(0, int(round(self.idle_seconds - self.idle_elapsed())))

    def check(self) -> bool:
        """Fire ``on_expire`` exactly once when idle for the full window with
        nothing in flight. Returns True on the firing call."""
        if not self.enabled:
            return False
        with self._lock:
            if self._expired or self._inflight > 0:
                return False
            if self._clock() - self._last_activity < self.idle_seconds:
                return False
            self._expired = True
        self._on_expire()
        return True


class WorkerSession:
    """The gpu-session process's own state: discovery record + idle timer."""

    def __init__(self, *, generation: str, port: int, idle_minutes: float,
                 node: str | None = None, on_expire=None,
                 poll_interval: float = 1.0):
        self.generation = generation
        self.node = node or socket.gethostname()
        self.port = int(port)
        self.started_at = _utc_now_iso()
        self.timer = IdleTimer(idle_minutes * 60.0, on_expire or self._expire)
        self._poll_interval = poll_interval
        self._stop = threading.Event()

    def start(self) -> None:
        self.write_record("ready")
        if not self.timer.enabled:
            return

        def loop():
            while not self._stop.wait(self._poll_interval):
                try:
                    if self.timer.check():
                        return
                except Exception:  # noqa: BLE001 - the watchdog must not die
                    pass

        threading.Thread(target=loop, name="session-idle-timer",
                         daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def write_record(self, state: str, reason: str | None = None) -> None:
        record = {
            "sessionGeneration": self.generation,
            "node": self.node,
            "port": self.port,
            "startedAt": self.started_at,
            "state": state,
        }
        if reason:
            record["endedReason"] = reason
        _write_json_atomic(worker_record_path(), record)

    def status(self) -> dict:
        """What GET /api/session returns ON THE WORKER (the controller folds
        this into the session record it serves clients)."""
        busy = self.timer.busy
        if busy:
            state = "busy"
        elif self.timer.idle_elapsed() < _READY_GRACE_SECONDS:
            state = "ready"
        else:
            state = "idle"
        payload = {
            "sessionGeneration": self.generation,
            "state": state,
            "busy": busy,
            "idleRemainingSeconds": self.timer.remaining_seconds(),
        }
        total = _gpu_total_memory_bytes()
        if total:
            # ACTUAL device capacity — CUDA's usable total (an "L4 24 GB"
            # reports 22.05 GiB), the number the client's memory-fit gating
            # must compare against, never the marketing GB from a profile
            # (engineer review 2026-07-18: the GiB/GB confusion passed the
            # exact 12B-on-L4 case the gate was built for).
            payload["gpuTotalMemoryBytes"] = total
        return payload

    def _expire(self) -> None:
        # Loud, stamped, and clean: write the shared record FIRST so the
        # controller's next reconcile reads "ended" instead of inferring a
        # dead socket, then SIGTERM our own pid so uvicorn drains connections
        # and exits — the Slurm job then completes and the controller's
        # session record transitions without the app ever seeing a hang.
        print(f"gpu-session: idle for {self.timer.idle_seconds / 60:.0f} "
              "minutes with no in-flight work — expiring the session and "
              "shutting down (the controller stays up; start a new session "
              "with POST /api/session/start)", file=sys.stderr, flush=True)
        try:
            self.write_record("ended", reason="idle-expired")
        except OSError:
            pass
        os.kill(os.getpid(), signal.SIGTERM)


_worker_lock = threading.Lock()
_worker: WorkerSession | None = None


def ensure_worker() -> WorkerSession:
    """Lazily create the process-wide worker state (role=gpu-session only).
    Lazy + idempotent so both the lifespan hook and the first request can
    initialize it, whichever comes first."""
    global _worker
    with _worker_lock:
        if _worker is None:
            idle_raw = os.environ.get("STEERLAB_SESSION_IDLE_MINUTES", "0")
            try:
                idle_minutes = float(idle_raw or "0")
            except ValueError:
                idle_minutes = 0.0
            port_raw = (os.environ.get("STEERLAB_SESSION_PORT") or "").strip()
            if port_raw.isdigit():
                port = int(port_raw)
            else:
                # "auto"/unset/garbled: allocation-derived, same rule as
                # cli._serve (which stamps the resolved number into the env
                # for scheduled runs; this branch covers direct uvicorn
                # launches).
                port = derive_session_port()
            _worker = WorkerSession(
                generation=os.environ.get("STEERLAB_SESSION_GENERATION", ""),
                port=port, idle_minutes=idle_minutes)
            _worker.start()
        return _worker


def reset_worker_for_tests() -> None:
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.stop()
        _worker = None


# --- controller side: session lifecycle ---------------------------------------


def _gpu_type_of(gres: str | None) -> str | None:
    from . import housekeeping
    return housekeeping.parse_gpu_type(gres)


def _walltime_expiry(started_at_iso: str, walltime: str | None) -> str | None:
    """startedAt + walltime — the scheduler's hard backstop, surfaced so the
    UI can show remaining walltime without asking Slurm."""
    if not walltime:
        return None
    try:
        started = datetime.fromisoformat(started_at_iso.replace("Z", "+00:00"))
        return (started + _parse_walltime(walltime)).isoformat()
    except (ValueError, TypeError):
        return None


# --- start-parameter validation ------------------------------------------------
# The Swift form bounds these client-side, but the API is also reachable from
# the web workbench and curl — and idleMinutes <= 0 DISABLES the idle timer,
# letting a direct caller burn a GPU to walltime (plan acceptance criterion 1).
# The 0-disables behavior survives for unit tests only; it is unreachable here.

_WALLTIME_RE = re.compile(r"^\d{1,3}:[0-5]\d:[0-5]\d$")
_MEMORY_RE = re.compile(r"^[1-9]\d*[KMGTP]?$")
# gres/partition are rendered into #SBATCH lines: restrict to the safe token
# alphabet so a caller can never smuggle a directive or shell metacharacter.
_SLURM_TOKEN_RE = re.compile(r"^[A-Za-z0-9_:.\-]+$")


def _bad_param(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _int_param(body: dict, key: str, low: int, high: int) -> int | None:
    """An optional integer field, bounds-checked with an actionable 400.
    Accepts int or an integer string (form clients); bool/float are refused —
    JSON true would otherwise coerce to 1."""
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise _bad_param(f"{key} must be an integer between {low} and {high} "
                         f"(got {value!r})")
    if isinstance(value, str):
        text = value.strip()
        if not (text.lstrip("-").isdigit() and text.lstrip("-")):
            raise _bad_param(f"{key} must be an integer between {low} and "
                             f"{high} (got {value!r})")
        value = int(text)
    if not low <= value <= high:
        raise _bad_param(f"{key} must be between {low} and {high} "
                         f"(got {value})")
    return value


def _pattern_param(body: dict, key: str, pattern: re.Pattern,
                   example: str) -> str | None:
    value = body.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not pattern.match(value):
        raise _bad_param(f"{key} must match {pattern.pattern} "
                         f"(e.g. {example!r}; got {value!r})")
    return value


def validate_start_params(body: dict) -> dict:
    """Server-side bounds for POST /api/session/start. Returns the validated
    values (None where the field was absent — defaults apply downstream)."""
    return {
        # 1, not 0: idleMinutes=0 disables the idle timer entirely, which on
        # an API surface means a forgotten curl can hold a GPU to walltime.
        "idleMinutes": _int_param(body, "idleMinutes", 1, 240),
        "port": _int_param(body, "port", 1024, 65535),
        "cpus": _int_param(body, "cpus", 1, 64),
        "walltime": _pattern_param(body, "walltime", _WALLTIME_RE, "02:00:00"),
        "memory": _pattern_param(body, "memory", _MEMORY_RE, "64G"),
        "gres": _pattern_param(body, "gres", _SLURM_TOKEN_RE, "gpu:A100:1"),
        "partition": _pattern_param(body, "partition", _SLURM_TOKEN_RE,
                                    "gpu_p"),
    }


def start_session(body: dict, jobs) -> dict:
    """Submit the GPU worker job and write the session record (state
    "queued"). Raises ``SessionConflict`` (carrying the live record) when a
    non-terminal session exists — never two workers. The check-then-submit
    runs under the cross-process lifecycle lock so concurrent starts cannot
    both pass the check and race two billed workers into existence."""
    profile = ServerProfile.from_env()
    if profile.executor != "slurm":
        raise HTTPException(
            status_code=403,
            detail="GPU sessions submit a Slurm worker job — this controller "
                   "does not declare the Slurm executor "
                   "(STEERLAB_EXECUTOR=slurm)")

    params = validate_start_params(body)

    with _lifecycle_lock():
        return _start_session_locked(body, params, jobs, profile)


def _start_session_locked(body: dict, params: dict, jobs,
                          profile: ServerProfile) -> dict:
    existing = reconcile_session()
    if existing is not None and existing.get("state") not in TERMINAL_SESSION_STATES:
        # "ending" counts as occupied on purpose: a stop whose scancel FAILED
        # leaves the record "ending" with the allocation possibly still
        # running — a fresh start here would hide that billed orphan behind
        # a new session (the live-shakedown failure mode).
        raise SessionConflict(
            {key: existing.get(key) for key in SESSION_RECORD_KEYS})

    generation = str(uuid.uuid4())
    # None = "auto": the WORKER derives an allocation-unique port from its
    # own SLURM_JOB_ID (a fixed default collides when two session jobs share
    # a multi-GPU node — engineer review 2026-07-17) and advertises the real
    # one via its discovery record, which reconcile folds back in. An
    # explicit port in the start request still wins, validated as before.
    port = params["port"]
    if port is None:
        # A site MAY pin its session port (`scheduler.gpuSession.port`). It is
        # the site's call and it behaves exactly like an explicit request port
        # — including the collision risk the "auto" derivation exists to avoid,
        # which is why nothing but a declaration turns it on.
        port = _session_int_env("STEERLAB_SESSION_DEFAULT_PORT", 1024, 65535)
    idle_minutes = params["idleMinutes"]
    if idle_minutes is None:
        idle_minutes = _session_int_env("STEERLAB_SESSION_IDLE_MINUTES", 1, 240)
    if idle_minutes is None:
        idle_minutes = DEFAULT_IDLE_MINUTES
    cpus = params["cpus"]
    if cpus is None:
        cpus = _session_int_env("STEERLAB_SESSION_CPUS", 1, 64)
    if cpus is None:
        cpus = DEFAULT_CPUS

    # Every field: request → this site's declared session block → its study
    # defaults where they exist → the engine fallback (WP5 Step 9, audit c20).
    resources = SlurmResources.from_env(job_name="steerlab-gpu-session")
    resources.partition = (
        params["partition"] or _session_env("STEERLAB_SESSION_PARTITION")
        or resources.partition)
    resources.gres = (
        params["gres"] or _session_env("STEERLAB_SESSION_GRES") or resources.gres
        or DEFAULT_GRES)
    resources.memory = (
        params["memory"] or _session_env("STEERLAB_SESSION_MEMORY")
        or resources.memory or DEFAULT_MEMORY)
    resources.walltime = (
        params["walltime"] or _session_env("STEERLAB_SESSION_WALLTIME")
        or DEFAULT_WALLTIME)
    # The class's own #SBATCH lines, after the site-wide ones — the order
    # `compose_scheduler_headers` gives `class_extra_sbatch`, reached here by
    # appending because the study path's `SlurmResources` has one extras slot.
    resources.extra_sbatch = list(resources.extra_sbatch) + [
        item for item in (os.environ.get("STEERLAB_SESSION_EXTRA_SBATCH") or "").split()
    ]
    resources.cpus_per_task = cpus
    resources.gpus = 1
    # HARD RULE (plan §2.3): an interactive session must never silently
    # respawn on a billed allocation — auto-resubmit and requeue are forced
    # off regardless of env/site defaults, and the stamped False on the job
    # record is what the reconciler's gate reads.
    resources.requeue = False
    resources.auto_resubmit = False
    # No --signal directive (live shakedown 2026-07-16): the SlurmResources
    # default is the STUDY checkpoint signal (USR1 at T-minus-10-minutes),
    # but the session worker has no USR1 handler — inheriting it kills the
    # session ~10 minutes before the walltime the app displays.
    resources.signal_seconds = 0

    python = os.environ.get("STEERLAB_PYTHON") or sys.executable or "python"
    command = [python, "-m", "steerlab_server.cli", "serve"]
    if port is not None:
        # Explicit request only; under "auto" the worker resolves the port
        # itself (cli._serve → derive_session_port).
        command += ["--port", str(port)]
    env = {
        "STEERLAB_SERVER_ROLE": "gpu-session",
        # Same bind decision as controller-job.sbatch.template: the
        # controller's proxy dials the compute node's EXTERNAL interface, so
        # loopback would block the only path in. Compensating controls:
        # token mode on every route, and the worker REFUSES to start without
        # a token (enforced in cli._serve, in Python, not just in bash).
        "STEERLAB_BIND": "0.0.0.0",
        "STEERLAB_AUTH_MODE": "token",
        "STEERLAB_SESSION_GENERATION": generation,
        "STEERLAB_SESSION_IDLE_MINUTES": str(idle_minutes),
        "STEERLAB_SESSION_PORT": "auto" if port is None else str(port),
        # Both processes must resolve the SAME records tree (/home-side).
        "STEERLAB_METADATA_ROOT": profile.metadata_root,
        # The bearer token itself must NEVER land in the bundle — bundle.json
        # and run.sbatch are durable on-disk artifacts. The worker hydrates
        # STEERLAB_AUTH_TOKEN from this FILE at startup (cli._serve),
        # defaulting to the bootstrap-written ~/.steerlab-token.
        "STEERLAB_AUTH_TOKEN_FILE": (
            os.environ.get("STEERLAB_AUTH_TOKEN_FILE") or "~/.steerlab-token"),
    }
    # Non-secret runtime reconstruction the rendered sbatch script consumes
    # under --export=NONE (modules/conda/venv), plus profile facts the worker
    # should inherit from the controller's site env. HF_HOME/HF_HUB_OFFLINE
    # ride along explicitly (belt-and-suspenders over the same inheritance in
    # create_bundle): without them the worker looks at ~/.cache, misses every
    # model installed into the shared cache, and re-downloads on a billed GPU
    # — this is what broke the first live chat (shakedown 2026-07-16).
    # STEERLAB_NODE_STAGE_DIR rides along VERBATIM (e.g. the literal
    # '/lscratch/$SLURM_JOB_ID'): the sbatch render shell-quotes env values,
    # and the loader expands the placeholder on the compute node — so the
    # path lands under the WORKER's job id, never the controller's.
    # STEERLAB_JUDGE_KEY_FILE is a PATH, not a secret (the key file itself
    # lives mode-600 in $HOME, pushed by the app) — the path riding along
    # lets a worker's inline judging find a non-default location.
    for key in ("STEERLAB_SERVER_PROFILE", "STEERLAB_MODULES",
                "STEERLAB_CONDA_SH", "STEERLAB_CONDA_ENV", "STEERLAB_VENV",
                "STEERLAB_MAX_LOADED_MODELS", "HF_HOME", "HF_HUB_OFFLINE",
                "STEERLAB_NODE_STAGE_DIR", "STEERLAB_JUDGE_KEY_FILE"):
        if os.environ.get(key):
            env.setdefault(key, os.environ[key])

    submission_dir = paths.make_unique_run_directory("session-gpu")
    executor = SlurmExecutor(profile)
    bundle = executor.create_bundle(
        os.path.join(submission_dir, "slurm"), command, env=env,
        resources=resources,
        metadata={"kind": "gpuSession", "sessionGeneration": generation,
                  "port": port, "idleMinutes": idle_minutes})
    # submit() runs the maintenance-window preflight — a session that would
    # die mid-chat in a maintenance window refuses here, like any other job.
    slurm_id = str(executor.submit(bundle))

    try:
        stamped = dict(resources.__dict__)
        stamped.update({"sessionGeneration": generation,
                        "scriptPath": bundle.script_path,
                        "port": port, "idleMinutes": idle_minutes})
        jobs.record_external(
            SESSION_JOB_KIND, status="submitted", executor="slurm",
            executor_job_id=slurm_id, requested_resources=stamped,
            log=(f"submitted GPU session worker as Slurm job {slurm_id} "
                 f"(generation {generation}; idle timeout {idle_minutes} min; "
                 "auto-resubmit off by design — a chat session never respawns "
                 "on a billed allocation)"))

        record = {
            "sessionGeneration": generation,
            "slurmJobID": slurm_id,
            "node": None,
            "port": port,
            "state": "queued",
            "workspaceRoot": profile.root,
            "gpuType": _gpu_type_of(resources.gres),
            "gres": resources.gres,
            "partition": resources.partition,
            "walltime": resources.walltime,
            "idleMinutes": idle_minutes,
            "submittedAt": _utc_now_iso(),
            "startedAt": None,
            "expiresAt": None,
            "serverVersion": __version__,
            "role": "gpu-session",
            "stateDetail": None,
        }
        write_session_record(record)
    except BaseException:
        # sbatch succeeded but the record didn't land: an allocation the app
        # cannot see is the WORST outcome (live 2026-07-16: orphaned PENDING
        # jobs billed until manual scancel) — kill the fresh job before
        # re-raising, and say so if even that fails.
        try:
            rolled_back = executor.cancel(slurm_id)
        except Exception:  # noqa: BLE001 - the original error must surface
            rolled_back = False
        if not rolled_back:
            print(f"gpu-session: failed to persist the session record AND "
                  f"failed to scancel the fresh worker job {slurm_id} — "
                  f"run `scancel {slurm_id}` on the cluster to avoid a "
                  "billed orphan", file=sys.stderr, flush=True)
        raise
    return record


def _controller_bearer_token() -> str | None:
    """The controller's OWN token for controller→worker HTTP (the probe, the
    keepalive forward, the /api/state composition): STEERLAB_AUTH_TOKEN from
    the env, falling back to the SAME token file both processes hydrate from
    (cli._serve reads it for the worker; a controller started before the env
    was exported must still be able to authenticate its probes — live
    2026-07-17, fix 1)."""
    token = os.environ.get("STEERLAB_AUTH_TOKEN")
    if token:
        return token
    token_file = os.path.expanduser(
        (os.environ.get("STEERLAB_AUTH_TOKEN_FILE") or
         "~/.steerlab-token").strip())
    try:
        with open(token_file, encoding="utf-8") as handle:
            return handle.read().strip() or None
    except OSError:
        return None


def _with_controller_auth(headers: dict) -> dict:
    """Ensure an outgoing controller→worker request carries a bearer token:
    a client-supplied Authorization forwards verbatim (it already passed the
    controller's auth middleware); only when the client sent none does the
    controller's own token fill in — the worker runs token mode on EVERY
    route, so an unauthenticated forward would 401 against a healthy worker
    (the live 2026-07-17 probe failure, generalized)."""
    if any(k.lower() == "authorization" for k in headers):
        return headers
    token = _controller_bearer_token()
    if token:
        headers = dict(headers)
        headers["Authorization"] = f"Bearer {token}"
    return headers


class _ProbeAuthRejected:
    """Sentinel probe result: the worker ANSWERED — HTTP 401/403 — but
    refused the controller's bearer token. Distinct from unreachable (None)
    on purpose: an answering socket is positive evidence the allocation is
    up, and the remedy is a token-file fix, not waiting or a by-hand curl
    (live 2026-07-17: this exact case read as "not answering" and parked the
    session at Starting forever)."""

    def __repr__(self) -> str:  # pragma: no cover - debugging nicety
        return "PROBE_AUTH_REJECTED"


PROBE_AUTH_REJECTED = _ProbeAuthRejected()


_GPU_TOTAL_MEMORY_CACHE: int | None | str = "unset"


def _gpu_total_memory_bytes() -> int | None:
    """CUDA device 0's total memory in bytes, probed once (torch import is
    heavy; the value never changes for a worker's lifetime). None off-GPU."""
    global _GPU_TOTAL_MEMORY_CACHE
    if _GPU_TOTAL_MEMORY_CACHE == "unset":
        try:
            import torch
            _GPU_TOTAL_MEMORY_CACHE = (
                int(torch.cuda.get_device_properties(0).total_memory)
                if torch.cuda.is_available() else None)
        except Exception:  # noqa: BLE001 - status must never fail on a probe
            _GPU_TOTAL_MEMORY_CACHE = None
    return _GPU_TOTAL_MEMORY_CACHE  # type: ignore[return-value]


def probe_worker(node: str | None, port: int | None, timeout: float = 2.0):
    """GET the worker's own /api/session (its state/busy/countdown),
    AUTHENTICATED with the controller's own bearer token (the worker enforces
    token mode on every route — an unauthenticated probe 401s against a
    perfectly healthy worker; live 2026-07-17). A module seam so tests can
    stub reachability. Classified result:

    - dict — 2xx, parsed worker status (reachable);
    - ``PROBE_AUTH_REJECTED`` — 401/403: the worker is UP but the tokens
      mismatch (a config bug the operator must see, never "not answering");
    - None — connection refused/timeout/garbage: unreachable; reconcile
      degrades to job-level truth. Never raises.
    """
    if not node or not port:
        return None
    import httpx
    headers = {}
    token = _controller_bearer_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.get(f"http://{node}:{port}/api/session",
                             headers=headers, timeout=timeout)
    except Exception:  # noqa: BLE001 - unreachable IS the answer
        return None
    if response.status_code in (401, 403):
        return PROBE_AUTH_REJECTED
    if response.status_code != 200:
        return None
    try:
        data = response.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _poll_job_state(slurm_job_id: str | None) -> tuple[str | None, bool]:
    """``(state, query_ok)`` — see ``SlurmExecutor.poll_state_detailed``.
    ``query_ok`` True means a scheduler query SUCCEEDED, so ``state`` None is
    POSITIVE absence ("the scheduler does not know this job"); False means
    the query itself failed — silence about the CONVERSATION, which must
    never be read as the job being gone (2026-07-17 review, finding 1: an
    "ending" session closed on scheduler silence and could release the
    double-start slot with the allocation still billing)."""
    if not slurm_job_id:
        return None, False
    try:
        return SlurmExecutor().poll_state_detailed(str(slurm_job_id))
    except Exception:  # noqa: BLE001 - a flaky sacct must not break GET
        return None, False


def reconcile_session() -> dict | None:
    """Converge the session record against the durable truth: the worker's
    discovery record, the scheduler, and the worker's own status endpoint.

    Deliberately a pure function of ON-DISK state plus the scheduler — no
    in-memory session object — so a restarted controller (its own walltime
    ended, resubmit successor took over) rediscovers a still-running worker
    from the record alone and resumes proxying (plan §2.1). Returns the
    record plus the response-only ``idleRemainingSeconds``/``busy`` fields,
    or None when no session was ever started.
    """
    record = read_session_record()
    if record is None:
        return None
    extras = {"idleRemainingSeconds": None, "busy": None}
    if record.get("state") in TERMINAL_SESSION_STATES:
        return {**record, **extras}
    original = dict(record)

    worker = read_worker_record(record.get("sessionGeneration"))
    if worker is not None:
        record["node"] = worker.get("node") or record.get("node")
        # The worker's discovery record carries the port it ACTUALLY bound
        # (allocation-derived under port "auto") — fold it in so the record,
        # the diagnosis texts, and the UI name the real port.
        record["port"] = worker.get("port") or record.get("port")
        if worker.get("startedAt") and \
                record.get("startedAt") != worker["startedAt"]:
            # The worker's own startedAt is the finest truth and OVERRIDES a
            # scheduler-observed stamp (below): the fallback exists precisely
            # for workers that never manage to write this record.
            record["startedAt"] = worker["startedAt"]
            record["expiresAt"] = _walltime_expiry(
                record["startedAt"], record.get("walltime"))

    job_state, query_ok = _poll_job_state(record.get("slurmJobID"))
    worker_ended = worker is not None and worker.get("state") == "ended"
    ending_requested = original.get("state") == "ending"
    probe = None
    if (not ending_requested and worker is not None and not worker_ended
            and job_state in {"running", None}):
        probe = probe_worker(record.get("node"),
                             worker.get("port") or record.get("port"))
    probe_auth_rejected = probe is PROBE_AUTH_REJECTED
    if probe_auth_rejected:
        probe = None

    if ending_requested:
        # A stop was requested (end_session stamped "ending"). The record
        # leaves "ending" only on a POSITIVE terminal state, or positive
        # absence from a SUCCESSFUL scheduler query past the visibility
        # grace — a still-answering worker must never resurrect it,
        # scheduler-unknown within grace is accounting lag, and a FAILED
        # query proves nothing about the allocation (2026-07-17 review:
        # closing on silence could release the double-start slot while the
        # billed job kept running).
        if job_state in {"succeeded", "cancelled", "failed", "checkpointed"}:
            record["state"] = "ended"
            record["stateDetail"] = None
        elif (job_state is None and query_ok
                and not _within_visibility_grace(record.get("submittedAt"))):
            # The scheduler answered and positively does not know the job.
            record["state"] = "ended"
            record["stateDetail"] = None
        elif job_state is None and not query_ok:
            # sacct/squeue themselves are failing: stay "ending" (still
            # occupying the slot) and say WHY the stop cannot confirm. The
            # operator force-clear remains the escape.
            record["stateDetail"] = (
                f"stop requested for job {record.get('slurmJobID')} but the "
                "scheduler query is failing — cannot confirm the allocation "
                "stopped; will keep retrying. Check `sacct -j "
                f"{record.get('slurmJobID')}` by hand, then clear with "
                "DELETE /api/session?force=1 once the job is verified dead")
        # else: still submitted/running (or unknown within grace) — the stop
        # is not yet confirmed; stay "ending" and keep double-start blocked.
    elif probe is not None:
        # A reachable worker is the strongest evidence there is — it also
        # RECOVERS a session parked "unknown", and clears any stale
        # unknown/diagnosis text.
        state = probe.get("state")
        if state not in {"ready", "busy", "idle"}:
            state = "busy" if probe.get("busy") else "ready"
        record["state"] = state
        record["stateDetail"] = None
        record["lastReachableAt"] = _utc_now_iso()
        if probe.get("gpuTotalMemoryBytes"):
            # ACTUAL usable VRAM (CUDA total_memory) — persisted so the
            # client's memory-fit gating survives probe gaps; never the
            # marketing GB from the site profile.
            record["gpuTotalMemoryBytes"] = int(probe["gpuTotalMemoryBytes"])
        extras["idleRemainingSeconds"] = probe.get("idleRemainingSeconds")
        extras["busy"] = bool(probe.get("busy"))
    elif probe_auth_rejected:
        # The worker ANSWERED (401/403): positive evidence the allocation is
        # up, but the controller's token was refused — an auth-config bug,
        # never "not answering" (live 2026-07-17: this case sat at a bare
        # Starting past the diagnosis window while the researcher's curl
        # proved the worker healthy). Nonterminal "starting" with the
        # distinct token-mismatch diagnosis; recovery is fixing the token,
        # after which the next probe succeeds and reconcile converges.
        record["state"] = "starting"
        if not record.get("startedAt"):
            record["startedAt"] = _utc_now_iso()
            record["expiresAt"] = _walltime_expiry(
                record["startedAt"], record.get("walltime"))
        record["stateDetail"] = _auth_mismatch_detail(record)
    elif worker_ended:
        # The worker stamped its own exit (idle expiry); "ending" while the
        # scheduler still shows the job draining, "ended" once it is gone.
        record["state"] = "ending" if job_state == "running" else "ended"
        record["stateDetail"] = None
    elif ((job_state == "running"
           or (job_state is None and record.get("startedAt")))
          and _recent_proxied_traffic()):
        # Bytes beat probes: the controller itself streamed proxied traffic
        # to/from the worker within the last minute (load heartbeats, chat
        # chunks). That is DIRECT evidence the worker is alive and working —
        # a probe miss here means its CPUs are busy with the request, not
        # that compute-to-compute HTTP is blocked (live 2026-07-17: the
        # "try curl" diagnosis fired mid-load while heartbeats flowed
        # through this very process).
        record["state"] = "busy"
        record["lastReachableAt"] = _utc_now_iso()
        record["stateDetail"] = (
            "worker is serving a proxied request (stream bytes within the "
            "last minute); health probe deferred while it works")
        if not record.get("startedAt"):
            record["startedAt"] = _utc_now_iso()
            record["expiresAt"] = _walltime_expiry(
                record["startedAt"], record.get("walltime"))
        extras["busy"] = True
    elif job_state == "submitted":
        record["state"] = "queued"
        record["stateDetail"] = None
    elif (job_state == "running"
            and original.get("state") in {"ready", "busy", "idle"}
            and _within_probe_lapse(record.get("lastReachableAt"))):
        # Probe-miss hysteresis: the worker answered a probe moments ago and
        # the job is still running — one missed short-timeout probe under
        # heavy GPU work (cold load, long decode) is latency, not death.
        # Keep the last reachable state; a sustained silence (past the lapse
        # window) falls through to the demotion below on a later reconcile.
        record["state"] = original["state"]
        record["stateDetail"] = (
            "worker missed a health probe (heavy load can delay answers); "
            "treating the session as still up")
    elif job_state == "running":
        record["state"] = "starting"
        if not record.get("startedAt"):
            # Scheduler-observed start: the sbatch script execs the worker
            # within seconds of RUNNING, so this anchors expiresAt and the
            # unreachable-worker diagnosis even when the worker never writes
            # its discovery record (which would carry the finer timestamp).
            record["startedAt"] = _utc_now_iso()
            record["expiresAt"] = _walltime_expiry(
                record["startedAt"], record.get("walltime"))
        # Running with the probe failing: bare "starting" forever explains
        # nothing — past the diagnosis threshold, say exactly what to check
        # (compute→compute reachability is the plan-§5 open assumption).
        record["stateDetail"] = _starting_diagnosis(record)
    elif job_state in {"succeeded", "cancelled", "checkpointed"}:
        record["state"] = "ended"
        record["stateDetail"] = None
    elif job_state == "failed":
        record["state"] = "failed"
        record["stateDetail"] = (
            f"Slurm reported job {record.get('slurmJobID')} FAILED — check "
            f"`sacct -j {record.get('slurmJobID')}` and the slurm-*.err log "
            "in its bundle directory")
    elif record.get("startedAt"):
        # Once-started worker with scheduler AND probe silent: that is a
        # breakdown of the scheduler CONVERSATION, not proof the allocation
        # is gone — stamping a terminal state here reopens the orphan door
        # (2026-07-16 follow-up review). Nonterminal "unknown" keeps the
        # double-start guard occupied until evidence or an operator clear.
        record["state"] = "unknown"
        record["stateDetail"] = _unknown_detail(record)
    elif _within_visibility_grace(record.get("submittedAt")):
        # Freshly submitted and the scheduler doesn't know it YET: the observed
        # sacct/squeue lag freshly submitted jobs (live shakedown 2026-07-16
        # — reading this as failure orphaned two PENDING sessions on billed
        # GPUs). Visibility lag must NEVER become "failed"; stay queued.
        record["state"] = "queued"
    else:
        # Grace expired with no positive scheduler answer: same rule — the
        # silence is about the conversation, so nonterminal "unknown", never
        # "failed" (which would let a new start bury a possibly-running job).
        record["state"] = "unknown"
        record["stateDetail"] = _unknown_detail(record)

    if record != original:
        write_session_record(record)
    return {**record, **extras}


def end_session(jobs, force: bool = False) -> dict:
    """scancel the worker through the job machinery; record → "ending", and
    RECONCILIATION moves ending → ended once the scheduler confirms the job
    terminal/gone. Never stamps a success it cannot prove: a failed scancel
    surfaces as an actionable 502 with the session left "ending" (still
    occupying the double-start guard — a billed orphan must not hide behind
    a fresh session). Raises 409 when there is nothing live to end. Retrying
    DELETE on an "ending" session re-attempts the cancel.

    ``force`` is the OPERATOR CLEAR (2026-07-16 follow-up review) — the only
    manual exit from "unknown" (or a stuck "ending"): scancel is attempted
    best-effort, then the record is stamped "ended" with a stateDetail that
    records the human took responsibility. Use it only AFTER verifying the
    job by hand (sacct/squeue) — that is what the stamp asserts."""
    with _lifecycle_lock():
        record = read_session_record()
        if record is None or record.get("state") in TERMINAL_SESSION_STATES:
            raise HTTPException(
                status_code=409,
                detail="no live GPU session to end — GET /api/session shows "
                       "none running")
        job_id = str(record.get("slurmJobID"))
        record["state"] = "ending"
        record["stateDetail"] = None
        write_session_record(record)

        job = next((j for j in jobs.list()
                    if j.kind == SESSION_JOB_KIND
                    and j.executor_job_id == record.get("slurmJobID")), None)
        if job is not None:
            # cancel() reports achieved-or-honestly-accepted; for a Slurm job
            # that means scancel exited 0 (or the scheduler had already
            # reported it terminal) — the proof of cancellation we require.
            cancelled = jobs.cancel(job.id)
        else:
            # Controller restarted with a fresh store (or the record predates
            # it): cancel through the scheduler directly rather than
            # stranding the allocation behind a missing bookkeeping row.
            try:
                cancelled = SlurmExecutor().cancel(job_id)
            except Exception:  # noqa: BLE001 - surfaced as the 502 below
                cancelled = False
        if force:
            # Best-effort scancel already happened above; the operator clear
            # stamps terminal REGARDLESS and records who decided.
            record["state"] = "ended"
            record["stateDetail"] = ("cleared by operator after manual "
                                     f"verification (job {job_id})")
            write_session_record(record)
            return record
        if not cancelled:
            detail = (f"scancel {job_id} did not confirm — the GPU "
                      f"allocation may still be running; retry "
                      f"DELETE /api/session, or run `scancel {job_id}` "
                      f"(or `scancel --name steerlab-gpu-session`) on the "
                      f"cluster and check `sacct -j {job_id}`, then clear "
                      "with DELETE /api/session?force=1")
            record["stateDetail"] = detail
            write_session_record(record)
            raise HTTPException(status_code=502, detail=detail)
        return record


# --- the streaming reverse proxy (plan §2.1) ----------------------------------

# EXPLICIT allowlist — exactly the interactive route families the Swift
# ChatService / remote-generation path calls (ClusterClient: /api/state,
# /api/load, /api/models/unload, /api/generate/stream, /api/variant/generate
# [+/stream], /api/variant/battery, /api/reader/score) plus the genuinely
# SYNCHRONOUS authoring
# compute (concept stats). The JOB-BACKED authoring verbs (concept extract,
# grand-mean, reader fit — and, since 2026-07-21, experiment extract/
# validate, plus evaluate when its panel pins a local judge) are
# deliberately NOT proxied (engineer review 2026-07-18: the worker runs no
# job subsystem, so proxying them answered 409 "no job subsystem") — they
# stay controller-owned, and the controller's job DELEGATES to the worker's
# synchronous route via call_worker_sync.
# Session management, jobs, experiments, submissions, workspace,
# housekeeping, and model installs are controller-owned and NEVER proxied —
# Studies must not accidentally target the interactive server (acceptance
# criterion 4). Batch-scale experiment verbs (run/sweep/pipeline) are
# neither proxied nor delegated: a controller answers them with an
# immediate refusal naming POST /api/studies/submit.
PROXIED_ROUTES = frozenset({
    "/api/load",
    "/api/load/stream",
    "/api/models/unload",
    "/api/generate",
    "/api/generate/stream",
    "/api/variant/generate",
    "/api/variant/generate/stream",
    # The robustness path's format-2 half: synchronous in-process compute the
    # app orchestrates turn by turn, exactly like the generate wire beside it
    # (and NOT a job — the worker runs no job subsystem).
    "/api/variant/battery",
    "/api/reader/score",
    "/api/state",
})

# Concept-scoped SYNCHRONOUS compute verbs: /api/concept/<name>/stats runs
# the model in-process and returns inline. extract/probe-train are absent —
# job-backed (see above).
_PROXIED_CONCEPT_SUFFIXES = ("/stats",)


def should_proxy(path: str) -> bool:
    if path in PROXIED_ROUTES:
        return True
    return (path.startswith("/api/concept/")
            and path.endswith(_PROXIED_CONCEPT_SUFFIXES))

# Direct reachability evidence from the proxy itself (live 2026-07-17): a
# worker whose CPUs are pegged by the very load/generation it is serving can
# miss the short-timeout health probe for minutes — while its heartbeat and
# chunk bytes flow through THIS process. Bytes beat probes: reconcile treats
# recent proxied traffic as proof the worker is alive and busy, instead of
# demoting the session to "starting" with the compute-to-compute-blocked
# diagnosis mid-request.
_LAST_PROXIED_TRAFFIC_MONOTONIC = 0.0


def note_proxied_traffic() -> None:
    global _LAST_PROXIED_TRAFFIC_MONOTONIC
    _LAST_PROXIED_TRAFFIC_MONOTONIC = time.monotonic()


def _recent_proxied_traffic(within: float = 60.0) -> bool:
    return (_LAST_PROXIED_TRAFFIC_MONOTONIC > 0
            and time.monotonic() - _LAST_PROXIED_TRAFFIC_MONOTONIC < within)


_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host",
    "content-length",
})

# Test seam: an httpx transport override (e.g. ASGITransport) so proxy tests
# can run hermetically; None means the real network.
_TRANSPORT = None


def _proxy_target() -> tuple[str, int] | None:
    """(node, port) of a live, DISCOVERED worker, else None — in which case
    every route behaves exactly as it does on a session-less controller
    (zero regression for non-session deployments)."""
    record = read_session_record()
    if record is None or record.get("state") in TERMINAL_SESSION_STATES \
            or record.get("state") == "ending":
        return None
    worker = read_worker_record(record.get("sessionGeneration"))
    if worker is None or worker.get("state") == "ended":
        return None
    node = worker.get("node")
    port = worker.get("port") or record.get("port")
    if not node or not port:
        return None
    return str(node), int(port)


def worker_sync_target() -> tuple[str, int] | None:
    """(node, port) of the live session worker for DIRECT synchronous
    compute calls — the controller-job delegation seam (engineer review
    2026-07-18: job-backed authoring verbs cannot run on the worker, which
    has no job subsystem; instead the controller's job calls the worker's
    route synchronously). Same dialability rules as the proxy."""
    return _proxy_target()


#: Attribute carrying a worker-packaged partial-evidence pointer out
#: through the RuntimeError the controller's job will catch.
WORKER_PARTIAL_ATTR = "steerlab_worker_partial_evidence"


def worker_partial_evidence(exc: BaseException) -> dict | None:
    """The partial-evidence pointer a session worker attached to a failure,
    if any. The controller cannot discover this for itself: the run
    directory was created in the worker's process."""
    value = getattr(exc, WORKER_PARTIAL_ATTR, None)
    return value if isinstance(value, dict) else None


def call_worker_sync(path: str, body: dict) -> dict:
    """POST an authoring-compute route DIRECTLY on the session worker and
    return its parsed JSON (the worker answers these routes synchronously —
    ``{"ok", "result", "logs", "sync"}``). For controller JOB work
    functions: the job stays durable in the controller's store (single-
    writer rule) while the model math runs where the model lives. Blocking
    (job threads); no read timeout — extraction legitimately runs minutes.
    Raises RuntimeError with an actionable message when no dialable worker
    exists or the worker answers an error."""
    target = worker_sync_target()
    if target is None:
        raise RuntimeError(
            "no GPU session worker to run this on — start one (POST "
            "/api/session/start or the app's GPU Session control) and retry")
    node, port = target
    import asyncio

    import httpx
    headers = _with_controller_auth({})

    async def _post():
        # No read/write timeout: model compute streams no bytes until done.
        async with httpx.AsyncClient(
                transport=_TRANSPORT,
                timeout=httpx.Timeout(5.0, read=None, write=None,
                                      pool=None)) as client:
            return await client.post(f"http://{node}:{port}{path}",
                                     json=body, headers=headers)

    try:
        response = asyncio.run(_post())
    except Exception as exc:  # noqa: BLE001 - transport failure = worker gone
        raise RuntimeError(
            f"session worker at {node}:{port} did not answer {path} — the "
            f"session may have ended mid-compute ({exc})") from exc
    note_proxied_traffic()  # direct worker traffic is reachability evidence
    if response.status_code != 200:
        try:
            detail = response.json().get("detail")
        except Exception:  # noqa: BLE001 - non-JSON error body
            detail = response.text[:500]
        # A worker that packaged partial evidence reports WHERE (external
        # review round 2, finding 2). The bundle is on the SHARED
        # workspace, so the controller can hand it to the app without
        # re-fetching anything from the worker — but only if the pointer
        # survives the error path, hence the structured detail.
        partial = None
        if isinstance(detail, dict):
            partial = detail.get("partialEvidence")
            detail = detail.get("message", detail)
        error = RuntimeError(
            f"session worker refused {path} ({response.status_code}): "
            f"{detail}")
        if isinstance(partial, dict):
            setattr(error, WORKER_PARTIAL_ATTR, partial)
        raise error
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"session worker answered {path} with a "
                           "non-object body")
    return payload


def session_unavailable_detail(record: dict,
                               worker: dict | None = None) -> str:
    """Actionable 503 detail for an allowlisted route while a NON-TERMINAL
    session exists but its worker cannot be dialed (no discovery record, or
    the record says the worker ended). Derived from the reconciled session
    state so the message names the wait ("queued"), the diagnosis
    ("starting" + stateDetail), or the operator action ("unknown")."""
    state = record.get("state")
    job_id = record.get("slurmJobID")
    detail = record.get("stateDetail")
    if worker is not None and worker.get("state") == "ended":
        return (f"GPU session worker has ended (job {job_id}) — start a new "
                "session (POST /api/session/start or the app's GPU Session "
                "control)")
    if state == "queued":
        return f"GPU session is queued (job {job_id}) — wait for it to start"
    if state == "starting":
        base = f"GPU session worker is starting (job {job_id})"
        return f"{base} — {detail}" if detail else base
    if state == "unknown":
        return detail or _unknown_detail(record)
    if state == "ending":
        base = (f"GPU session is ending (job {job_id}) — wait for it to "
                "finish, then start a new one")
        return f"{base} — {detail}" if detail else base
    base = (f"GPU session (job {job_id}, state {state}) exists but its "
            "worker is not reachable from the controller — check "
            "GET /api/session")
    return f"{base} — {detail}" if detail else base


def _blocked_session_response() -> JSONResponse | None:
    """503 for allowlisted routes while a live session's worker is
    undialable; None when NO session (or a terminal one) exists — only then
    may the request fall through to the controller's own handling. Falling
    through with a live-but-unreachable session was the live failure
    (2026-07-17): /api/load answered "no GPU session running: start one"
    right after a session WAS started, and a variant-generate fell through
    to a controller-side load attempt that hung the chat spinner forever."""
    record = read_session_record()
    if record is None or record.get("state") in TERMINAL_SESSION_STATES:
        return None
    worker = read_worker_record(record.get("sessionGeneration"))
    return JSONResponse(
        {"detail": session_unavailable_detail(record, worker)},
        status_code=503)


async def _proxy_state(request: Request, target: tuple[str, int] | None,
                       controller_jobs):
    """/api/state is the Swift connection HANDSHAKE and must never answer a
    blocked-session 503 (engineer review 2026-07-17, fix 2: reconnecting
    during a queued/starting session failed against a healthy controller).
    COMPOSE instead:

    - no dialable worker → None: fall through, the controller answers its
      own state (its jobs list, no loaded models);
    - reachable worker → the worker's state with the CONTROLLER's jobs
      array overlaid (the worker's is always empty by the single-writer
      rule — this also fixes the jobs badge reading zero during a live
      session);
    - worker errors/vanishes mid-request → fall through to the controller's
      own state rather than 502: a degraded answer beats a failed handshake.

    Load/generate/score keep the blocked-503 path exactly as before.
    """
    if target is None:
        return None
    import httpx
    node, port = target
    headers = _with_controller_auth(
        {k: v for k, v in request.headers.items()
         if k.lower() not in _HOP_BY_HOP})
    try:
        async with httpx.AsyncClient(transport=_TRANSPORT,
                                     timeout=10.0) as client:
            response = await client.get(f"http://{node}:{port}/api/state",
                                        headers=headers)
        payload = response.json()
    except Exception:  # noqa: BLE001 - degrade to the controller's own state
        return None
    if response.status_code != 200 or not isinstance(payload, dict):
        return None
    if controller_jobs is not None:
        payload["jobs"] = controller_jobs()
    return JSONResponse(payload)


async def maybe_proxy(request: Request, controller_jobs=None):
    """Controller-side dispatch: a StreamingResponse forwarding to the live
    worker, a 502 when the worker vanished mid-session, a 503 when a live
    session's worker is not yet (or no longer) dialable, or None to fall
    through to the controller's own routes (no session / terminal session
    only). ``controller_jobs`` is a callable returning the controller's job
    dicts, used to overlay the jobs array on a proxied /api/state.

    Exception to the blocked-503 rule: /api/state (see _proxy_state) — the
    handshake route composes and never blocks."""
    if not should_proxy(request.url.path):
        return None
    target = _proxy_target()
    if request.url.path == "/api/state":
        return await _proxy_state(request, target, controller_jobs)
    if target is None:
        return _blocked_session_response()
    import httpx
    node, port = target
    url = f"http://{node}:{port}{request.url.path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    # Authorization and body forward VERBATIM; hop-by-hop headers do not.
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in _HOP_BY_HOP}
    body = await request.body()
    # No read/write timeout: generations stream for minutes. Connect stays
    # short so a dead worker maps to the actionable 502 quickly.
    client = httpx.AsyncClient(
        transport=_TRANSPORT,
        timeout=httpx.Timeout(5.0, read=None, write=None, pool=None))
    try:
        upstream = await client.send(
            client.build_request(request.method, url, headers=headers,
                                 content=body),
            stream=True)
    except Exception:  # noqa: BLE001 - any transport failure = worker gone
        await client.aclose()
        return JSONResponse({"detail": WORKER_DOWN_DETAIL}, status_code=502)
    note_proxied_traffic()

    response_headers = {k: v for k, v in upstream.headers.items()
                        if k.lower() not in _HOP_BY_HOP}

    async def stream_body():
        # SSE requirement: chunks pass through AS THEY ARRIVE — buffering a
        # whole generation would freeze the chat UI until the last token.
        try:
            async for chunk in upstream.aiter_raw():
                note_proxied_traffic()
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(stream_body(), status_code=upstream.status_code,
                             headers=response_headers)


async def forward_keepalive(request: Request):
    """POST /api/session/keepalive on the controller: proxied to the worker,
    where it counts as activity. 409 when no session is running."""
    record = read_session_record()
    worker = (read_worker_record(record.get("sessionGeneration"))
              if record is not None else None)
    if (record is None or record.get("state") in TERMINAL_SESSION_STATES
            or worker is None or worker.get("state") == "ended"):
        raise HTTPException(status_code=409, detail=NO_SESSION_DETAIL)
    node = worker.get("node")
    port = worker.get("port") or record.get("port")
    import httpx
    # Controller auth fills in when the client sent none: the worker runs
    # token mode on every route, so an unauthenticated forward would 401
    # against a healthy worker (the live 2026-07-17 probe failure,
    # generalized to every controller→worker call).
    headers = _with_controller_auth(
        {k: v for k, v in request.headers.items()
         if k.lower() not in _HOP_BY_HOP})
    client = httpx.AsyncClient(transport=_TRANSPORT, timeout=10.0)
    try:
        response = await client.post(
            f"http://{node}:{port}/api/session/keepalive", headers=headers)
    except Exception:  # noqa: BLE001 - worker vanished
        return JSONResponse({"detail": WORKER_DOWN_DETAIL}, status_code=502)
    finally:
        await client.aclose()
    try:
        payload = response.json()
    except ValueError:
        payload = {"ok": False}
    return JSONResponse(payload, status_code=response.status_code)
