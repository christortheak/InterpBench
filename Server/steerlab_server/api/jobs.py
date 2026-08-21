"""Durable job model for long server tasks.

Local development still runs callables in daemon threads. The difference from
the first-pass in-memory manager is that status and logs are also written to a
SQLite store under ``STEERLAB_METADATA_ROOT``/``.steerlab`` so clients can
reconnect by job id after a server restart.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .profile import ServerProfile, capability_snapshot


# "prepared" is the completed outcome of a dry-run submission: the bundle was
# staged and nothing executed. It is deliberately NOT rebranded "succeeded"
# (clients must be able to tell "ran" from "only prepared"), but it is terminal
# — nothing will ever advance a prepared job, so SSE streams must close on it.
#
# "cancelling" is deliberately NOT terminal: it means a cancel was requested on
# a RUNNING local job whose worker thread has not yet observed the flag — the
# work is still executing. The terminal "cancelled" stamp happens only where
# the worker acknowledges (its runner's return/raise path), so job status never
# claims the GPU is idle while a generation loop is still burning. SSE streams
# stay open through "cancelling" and close on the acknowledged "cancelled".
#
# "checkpointed" (a Slurm child that exited with the checkpoint code, 85) is
# also NOT terminal: the run directory is durably parked and resumable, and a
# requeue/resubmit continues it — the reconciler must keep following the job
# rather than declaring it dead. When auto-resubmit is enabled for the job
# (stamped requestedResources, else STEERLAB_AUTO_RESUBMIT), the reconciler
# resubmits the SAME sbatch script itself — bounded by the resubmit-chain cap,
# at most once per checkpoint, and never for a job whose cancellation was
# requested (cancelled beats checkpointed).
#
# "parked" is the completed outcome of a worker that STOPPED SHORT on purpose:
# the work could not continue here, the state is durable on disk, and a
# recovery action (named in `result["reason"]`) is waiting for the researcher.
# It is terminal — nothing will advance a parked job — but it is deliberately
# NOT "succeeded" (2026-08-06 review round 2, P1). The orphan-pipeline
# reconciler's `_park` returns normally, so before this the generic runner
# stamped it succeeded and every client painted it green: a chain that needs
# a human read as a finished one, which is the 2026-08-06 replication-run
# failure mode wearing a checkmark. Clients classify it as its own attention
# state.
TERMINAL = {"succeeded", "failed", "cancelled", "cancelledResumable",
            "prepared", "parked"}


def is_parked_result(result) -> bool:
    """Whether a worker's return value says it PARKED rather than completed.

    The marker is the ``parked: true`` key the parking helpers already stamp
    on both the returned result and the on-disk ledger, so the job status and
    the ledger can never disagree about what happened."""
    return isinstance(result, dict) and result.get("parked") is True

# "merging" is the sharded parent's post-shards state: every shard child
# succeeded and the reconciler is assembling the merged run directory. It is
# NOT terminal (the merge can refuse) and, like "running", it means work is
# in flight.


def derive_shard_parent_state(child_states: list[str]) -> str:
    """The sharded parent's derived state from its shards' EFFECTIVE states
    (resubmit chains already collapsed by the caller). Pure and unit-tested:

    - any terminal non-success (failed / cancelled / prepared-after-start
      anomalies) → "failed" — the fan-out cannot complete;
    - a cancelled shard reports "cancelled" (whole-fan-out stop);
    - all succeeded → "succeeded" (the caller then merges);
    - any checkpointed (and none failed) → "checkpointed" (resumable);
    - any running → "running"; else "submitted".
    """
    states = [s or "" for s in child_states]
    if any(s == "cancelled" for s in states):
        return "cancelled"
    if any(s == "failed" for s in states):
        return "failed"
    if states and all(s == "succeeded" for s in states):
        return "succeeded"
    if any(s == "checkpointed" for s in states):
        return "checkpointed"
    if any(s in ("running", "cancelling") for s in states):
        return "running"
    return "submitted"


def _resumable_directory_in(result: dict) -> str | None:
    """The run directory a cancelled worker's result points at, when that
    directory actually parked with resume state (its own stamped verb is the
    authority) — else None. Decides "cancelled" vs "cancelledResumable"."""
    run_dir = result.get("runDirectory")
    if not isinstance(run_dir, str) or not run_dir:
        return None
    from ..experiment import resume as resume_mod
    state = resume_mod.read_state(run_dir)
    verb = state.get("verb") if isinstance(state, dict) else None
    if verb and resume_mod.is_resumable(run_dir, verb):
        return run_dir
    return None


class ResubmitRefused(Exception):
    """A manual resume request refused with a plain-language reason.

    Raised only by ``JobManager.resubmit`` (the Resume button's verb) for
    states that are not a resumable checkpoint — already running, already
    finished, already resubmitted, cancelled, or missing its sbatch script.
    The message is the actionable text; the API maps it to HTTP 409."""


class DurableJobStore:
    def __init__(self, path: str | None = None):
        profile = ServerProfile.from_env()
        self.path = path or os.environ.get(
            "STEERLAB_JOBS_DB",
            os.path.join(profile.metadata_root, "jobs.sqlite"),
        )
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.RLock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    result_json TEXT,
                    error TEXT,
                    requested_resources_json TEXT,
                    output_artifacts_json TEXT,
                    executor TEXT,
                    executor_job_id TEXT,
                    cancellation_requested INTEGER NOT NULL DEFAULT 0,
                    capability_snapshot_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS job_logs (
                    job_id TEXT NOT NULL,
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    message TEXT NOT NULL
                )
            """)
            # Without this, every per-job log read scans the whole table —
            # and the jobs list reads a tail for EVERY job, over a database
            # that typically lives on NFS.
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_job_logs_job_seq
                ON job_logs(job_id, seq)
            """)

    def insert(self, job: "Job") -> None:
        with self._lock, self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO jobs (
                    id, kind, status, created_at, started_at, finished_at,
                    result_json, error, requested_resources_json,
                    output_artifacts_json, executor, executor_job_id,
                    cancellation_requested, capability_snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.id, job.kind, job.status, job.created_at, job.started_at,
                job.finished_at, _dumps(job.result), job.error,
                _dumps(job.requested_resources), _dumps(job.output_artifacts),
                job.executor, job.executor_job_id, int(job.cancelled),
                _dumps(job.capability_snapshot),
            ))

    def update(self, job: "Job") -> None:
        with self._lock, self._connect() as conn:
            conn.execute("""
                UPDATE jobs SET
                    status = ?, started_at = ?, finished_at = ?,
                    result_json = ?, error = ?, requested_resources_json = ?,
                    output_artifacts_json = ?, executor = ?, executor_job_id = ?,
                    cancellation_requested = ?, capability_snapshot_json = ?
                WHERE id = ?
            """, (
                job.status, job.started_at, job.finished_at, _dumps(job.result),
                job.error, _dumps(job.requested_resources), _dumps(job.output_artifacts),
                job.executor, job.executor_job_id, int(job.cancelled),
                _dumps(job.capability_snapshot), job.id,
            ))

    def append_log(self, job_id: str, message: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO job_logs (job_id, timestamp, message) VALUES (?, ?, ?)",
                (job_id, time.time(), message),
            )

    def log_tails(self, limit: int = 50) -> dict[str, list[str]]:
        """The last ``limit`` log lines of EVERY job, one query, one connection.

        The jobs list needs a tail for every job at once; calling ``logs()``
        per job opens a fresh connection each time — over an NFS-resident
        database that turned GET /api/jobs into a minutes-long request
        (344 jobs ≈ 100 s measured on the cluster, 2026-08-09)."""
        sql = ("SELECT job_id, message FROM ("
               "SELECT job_id, message, seq, ROW_NUMBER() OVER ("
               "PARTITION BY job_id ORDER BY seq DESC) AS rn FROM job_logs) "
               "WHERE rn <= ? ORDER BY job_id, seq")
        tails: dict[str, list[str]] = {}
        with self._lock, self._connect() as conn:
            for row in conn.execute(sql, (limit,)):
                tails.setdefault(row["job_id"], []).append(row["message"])
        return tails

    def logs(self, job_id: str, limit: int | None = None) -> list[str]:
        sql = "SELECT message FROM job_logs WHERE job_id = ? ORDER BY seq"
        params: tuple = (job_id,)
        if limit is not None:
            sql = ("SELECT message FROM job_logs WHERE job_id = ? "
                   "ORDER BY seq DESC LIMIT ?")
            params = (job_id, limit)
        with self._lock, self._connect() as conn:
            rows = [row["message"] for row in conn.execute(sql, params)]
        return list(reversed(rows)) if limit is not None else rows

    def mark_cancel_requested(self, job_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE jobs SET cancellation_requested = 1 WHERE id = ?", (job_id,))
            return cur.rowcount > 0

    #: A resubmission claim older than this is treated as abandoned (the
    #: claimant crashed between claiming and sbatch). Long enough that no
    #: live sbatch call outlasts it; short enough that a crash does not
    #: park a resumable run for good.
    RESUBMIT_CLAIM_STALE_SECONDS = 600.0

    def claim_resubmit(self, job_id: str, claimant: str,
                       token: str | None = None) -> tuple[bool, dict]:
        """Atomically claim the right to resubmit ``job_id`` (external review
        2026-07-22, finding 1: the manual Resume verb and the auto-resubmit
        tick both ran check-then-sbatch with no shared claim, so a double
        click or a manual-overlapping-auto could launch two Slurm processes
        appending to the same run directory).

        The claim is a durable compare-and-set on the job record's
        ``result_json``: a ``resubmitClaim`` stamp lands only when (a) no
        ``resubmittedAs`` stamp exists, (b) no claim exists at all,
        and (c) the row's bytes are exactly the bytes just read — so two
        claimants (threads OR processes over the same store) can never both
        win. Returns ``(claimed, result_after)``; on a lost race
        ``result_after`` is the current result, whose ``resubmittedAs`` (if
        already stamped) is the honest "already resumed as" answer.

        ``token`` is the durable, scheduler-findable submission token the
        winner will sbatch under (``--job-name``). It lands IN the claim
        BEFORE any sbatch, so a claimant that dies between sbatch and its
        continuation record leaves a searchable trail.

        A STALE claim is deliberately NOT taken over here (resume
        crash-safety, 2026-07-23): the claimant may have died AFTER sbatch
        with the continuation record unwritten, and blind takeover would
        double-submit. The manager resolves stale claims by searching the
        scheduler for the claim's token first
        (``JobManager._resolve_stale_claim``); only a proven-absent
        submission reclaims (``takeover_stale_claim``)."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT result_json FROM jobs WHERE id = ?",
                               (job_id,)).fetchone()
            if row is None:
                return False, {}
            current = row["result_json"]
            result = _loads(current)
            result = result if isinstance(result, dict) else {}
            if result.get("resubmittedAs"):
                return False, result
            if isinstance(result.get("resubmitClaim"), dict):
                return False, result
            new_result = {k: v for k, v in result.items() if k != "resubmitClaim"}
            claim: dict = {"claimant": claimant, "at": time.time()}
            if token:
                claim["token"] = token
            new_result["resubmitClaim"] = claim
            cur = conn.execute(
                "UPDATE jobs SET result_json = ? WHERE id = ? "
                "AND result_json IS ?",
                (_dumps(new_result), job_id, current))
            if cur.rowcount != 1:
                # A concurrent writer changed the row between read and CAS —
                # treated as a lost race; the caller re-reads on its side.
                return False, result
            return True, new_result

    @classmethod
    def claim_is_stale(cls, result: dict) -> bool:
        """Whether ``result`` carries a resubmission claim older than the
        stale window — the claimant is presumed dead, and the claim needs
        RESOLUTION (scheduler search), never blind takeover."""
        claim = (result or {}).get("resubmitClaim")
        if not isinstance(claim, dict):
            return False
        age = time.time() - float(claim.get("at") or 0)
        return age >= cls.RESUBMIT_CLAIM_STALE_SECONDS

    def takeover_stale_claim(self, job_id: str, claimant: str,
                             token: str | None = None) -> tuple[bool, dict]:
        """Replace a STALE claim with ``claimant``'s own — legal only after
        the manager proved (scheduler search by the old claim's token, or a
        human's explicit Resume consent where no token exists) that the dead
        claimant's sbatch never happened. Same CAS discipline as
        ``claim_resubmit``; refuses when the claim is live, absent, or a
        ``resubmittedAs`` stamp landed meanwhile."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT result_json FROM jobs WHERE id = ?",
                               (job_id,)).fetchone()
            if row is None:
                return False, {}
            current = row["result_json"]
            result = _loads(current)
            result = result if isinstance(result, dict) else {}
            if result.get("resubmittedAs"):
                return False, result
            if not self.claim_is_stale(result):
                return False, result
            new_result = {k: v for k, v in result.items() if k != "resubmitClaim"}
            claim: dict = {"claimant": claimant, "at": time.time()}
            if token:
                claim["token"] = token
            new_result["resubmitClaim"] = claim
            cur = conn.execute(
                "UPDATE jobs SET result_json = ? WHERE id = ? "
                "AND result_json IS ?",
                (_dumps(new_result), job_id, current))
            if cur.rowcount != 1:
                return False, result
            return True, new_result

    def stamp_claim_reconciliation_required(self, job_id: str,
                                            message: str) -> dict | None:
        """Durably mark a stale claim ``reconciliationRequired`` with a
        plain-language operator message: the scheduler could not be searched
        for the claim's submission token, so NOTHING may be resubmitted
        automatically (at-most-once beats liveness). The claim keeps its
        claimant/at/token so a later tick — or an operator — can still
        resolve it once the scheduler answers. Idempotent; returns the
        result after stamping, or None when nothing changed."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT result_json FROM jobs WHERE id = ?",
                               (job_id,)).fetchone()
            if row is None:
                return None
            result = _loads(row["result_json"])
            if not isinstance(result, dict):
                return None
            claim = result.get("resubmitClaim")
            if not isinstance(claim, dict):
                return None
            if (claim.get("reconciliationRequired")
                    and claim.get("operatorMessage") == message):
                return result
            new_claim = dict(claim)
            new_claim["reconciliationRequired"] = True
            new_claim["operatorMessage"] = message
            new_result = {**result, "resubmitClaim": new_claim}
            conn.execute(
                "UPDATE jobs SET result_json = ? WHERE id = ? "
                "AND result_json IS ?",
                (_dumps(new_result), job_id, row["result_json"]))
            return new_result

    def read_result(self, job_id: str) -> dict:
        """The DURABLE result for ``job_id``, straight from SQLite — what
        another process's writes look like. The in-memory catalog only sees
        this process's own updates (finding: the manual loser-wait polled
        memory and missed a sibling process's continuation)."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT result_json FROM jobs WHERE id = ?",
                               (job_id,)).fetchone()
        if row is None:
            return {}
        result = _loads(row["result_json"])
        return result if isinstance(result, dict) else {}

    def find_resubmissions_of(self, job_id: str) -> list["Job"]:
        """Every job in the DURABLE store born as a resubmission of
        ``job_id`` (``requestedResources.resubmitOf``), loaded fresh from
        SQLite so continuations written by another process are visible."""
        with self._lock, self._connect() as conn:
            rows = list(conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC"))
        out: list[Job] = []
        for row in rows:
            if row["id"] == job_id:
                continue
            rr = _loads(row["requested_resources_json"])
            if isinstance(rr, dict) and rr.get("resubmitOf") == job_id:
                out.append(self._from_row(row))
        return out

    def release_resubmit_claim(self, job_id: str, claimant: str) -> dict | None:
        """Drop ``claimant``'s resubmission claim (sbatch failed — a later
        tick or click must be able to retry). Only the claim's own claimant
        may release it. Returns the result after release, or None when
        nothing changed."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT result_json FROM jobs WHERE id = ?",
                               (job_id,)).fetchone()
            if row is None:
                return None
            result = _loads(row["result_json"])
            if not isinstance(result, dict):
                return None
            claim = result.get("resubmitClaim")
            if not (isinstance(claim, dict) and claim.get("claimant") == claimant):
                return None
            new_result = {k: v for k, v in result.items() if k != "resubmitClaim"}
            conn.execute(
                "UPDATE jobs SET result_json = ? WHERE id = ? "
                "AND result_json IS ?",
                (_dumps(new_result), job_id, row["result_json"]))
            return new_result

    def load_all(self) -> dict[str, "Job"]:
        with self._lock, self._connect() as conn:
            rows = list(conn.execute("SELECT * FROM jobs ORDER BY created_at DESC"))
        return {row["id"]: self._from_row(row) for row in rows}

    def _from_row(self, row: sqlite3.Row) -> "Job":
        job = Job(
            id=row["id"],
            kind=row["kind"],
            status=row["status"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            result=_loads(row["result_json"]),
            error=row["error"],
            requested_resources=_loads(row["requested_resources_json"]) or {},
            output_artifacts=_loads(row["output_artifacts_json"]) or [],
            executor=row["executor"] or "local",
            executor_job_id=row["executor_job_id"],
            capability_snapshot=_loads(row["capability_snapshot_json"]) or {},
            _store=self,
        )
        if row["cancellation_requested"]:
            job._cancel.set()
        for line in self.logs(job.id, limit=2000):
            job._logs.append(line)
        return job


def _dumps(value: object) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def _loads(value: str | None) -> object | None:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


@dataclass
class Job:
    id: str
    kind: str
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: str | None = None
    requested_resources: dict = field(default_factory=dict)
    output_artifacts: list[dict] = field(default_factory=list)
    executor: str = "local"
    executor_job_id: str | None = None
    capability_snapshot: dict = field(default_factory=dict)
    _logs: deque = field(default_factory=lambda: deque(maxlen=2000))
    _cancel: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _store: DurableJobStore | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def log(self, message: str) -> None:
        with self._lock:
            self._logs.append(message)
        if self._store is not None:
            self._store.append_log(self.id, message)

    def log_tail(self, n: int = 50) -> list[str]:
        if self._store is not None:
            return self._store.logs(self.id, limit=n)
        with self._lock:
            return list(self._logs)[-n:]

    def all_logs(self) -> list[str]:
        if self._store is not None:
            return self._store.logs(self.id)
        with self._lock:
            return list(self._logs)

    def to_dict(self, *, log_tail: list[str] | None = None) -> dict:
        # ``log_tail`` lets a bulk caller (the jobs list) supply tails it
        # fetched in one batched query instead of one store round-trip per job.
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "createdAt": self.created_at,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "result": self.result,
            "error": self.error,
            "logTail": self.log_tail() if log_tail is None else log_tail,
            "requestedResources": self.requested_resources,
            "outputArtifacts": self.output_artifacts,
            "executor": self.executor,
            "executorJobID": self.executor_job_id,
            "cancellationRequested": self.cancelled,
            "capabilitySnapshot": self.capability_snapshot,
        }


def _retain_partial_evidence(job, exc: BaseException) -> None:
    """Mark and package whatever a failed in-process job produced.

    The bundle executor does this for BUNDLED work; this is the same
    contract for jobs that run tasks in-process — which is how the direct
    ``/api/experiment/{name}/{verb}`` route executes, and how a paired
    server runs everything the app asks of it.

    Folds the bundle into ``job.result`` with ``partialEvidence: True``,
    because that is exactly what the Mac's retrieval affordance keys on: a
    failed job with a packaged bundle becomes a "Retrieve partial data"
    row instead of a dead end.

    Best-effort throughout. A job that produced no run directory records
    nothing (honest — there is nothing to retrieve), and a packaging
    failure is recorded without displacing the real error, which the
    researcher actually needs to see.
    """
    try:
        from ..experiment import bundles, run_status
        # A DELEGATED failure carries the worker's own pointer. The run
        # directory was created in the WORKER's process, so this controller
        # can neither observe nor package it — but the bundle already sits
        # on the shared workspace, so relaying the pointer is all the app
        # needs to retrieve it.
        from .gpu_session import worker_partial_evidence
        delegated = worker_partial_evidence(exc)
        if delegated:
            result = job.result if isinstance(job.result, dict) else {}
            result["evidenceBundle"] = {
                "bundlePath": delegated.get("bundlePath"),
                "bundleSha256": delegated.get("bundleSha256"),
                "evidenceComplete": False,
            }
            result["partialEvidence"] = True
            result["runDirectory"] = delegated.get("runDirectory")
            result["experiment"] = delegated.get("experiment")
            result["verb"] = delegated.get("verb")
            result["partialRunID"] = delegated.get("partialRunID")
            job.result = result
            job.log(
                "partial evidence packaged by the GPU session worker at "
                f"{delegated.get('partialRunID')} — on the shared workspace, "
                "retrievable from the app")
            return
        directory = run_status.partial_run_directory(exc)
        if not directory:
            return
        verb = str(job.kind or "").split(":")[-1] or "job"
        # Derived from the DIRECTORY, not from the job's verb: those differ
        # whenever the failing stage is not the job's own kind (a pipeline
        # job's `run` stage), and the hand-rolled trim then kept the stage
        # in the name.
        name = bundles.experiment_name_of_run_directory(directory, verb)
        try:
            bundles._mark_partial_run(directory, verb=verb, name=name, exc=exc)
        except Exception:  # noqa: BLE001 - a marker must not mask the error
            pass
        meta = bundles.package_evidence(
            directory,
            failure={"error": f"{type(exc).__name__}: {exc}",
                     "errorType": type(exc).__name__,
                     "verb": verb, "jobID": job.id,
                     "traceback": traceback.format_exc()})
        result = job.result if isinstance(job.result, dict) else {}
        result["evidenceBundle"] = meta
        result["partialEvidence"] = True
        result["runDirectory"] = directory
        # Enough for a client to offer a targeted RETRY without parsing a
        # run id: which experiment, which verb, and which partial to resume.
        result["experiment"] = name
        result["verb"] = verb
        result["partialRunID"] = os.path.basename(directory.rstrip(os.sep))
        job.result = result
        job.log(
            f"partial evidence packaged from {os.path.basename(directory)} "
            "— retrievable from the app as a failure record")
    except Exception as pack_exc:  # noqa: BLE001
        try:
            job.log(f"partial-evidence packaging failed: "
                    f"{type(pack_exc).__name__}: {pack_exc}")
        except Exception:  # noqa: BLE001
            pass


class JobManager:
    def __init__(self, store: DurableJobStore | None = None, capability_provider=None,
                 *, sweep_orphans: bool = True, slurm_executor=None,
                 reconcile_pipelines: bool = False):
        self.store = store or DurableJobStore()
        self._jobs: dict[str, Job] = self.store.load_all()
        self._lock = threading.Lock()
        self._capability_provider = capability_provider
        self._slurm_executor = slurm_executor
        self._monitor_stop = threading.Event()
        # --- monitor liveness (open-issues §2) -------------------------------
        # The shard merge has exactly ONE driver: `_reconcile_shard_parents`,
        # reached from `poll_slurm`, reached from the monitor thread. There is
        # no separate boot-time merge — "a fresh boot merges in seconds" is the
        # SAME code on a new thread in a new process. So a merge that never
        # happens means the monitor thread stopped ticking, and nothing here
        # used to notice: the loop swallowed every exception with a bare
        # `pass`, and every scheduler call underneath it was an unbounded
        # `subprocess.run` that a wedged sacct/squeue could block forever.
        # These three fields make the tick OBSERVABLE and the watchdog below
        # makes a stall recoverable without a daemon cycle.
        self._monitor_started_at: float | None = None
        self._monitor_last_tick: float | None = None
        self._monitor_ticks = 0
        self._monitor_stalls = 0
        self._monitor_last_error: str | None = None
        #: Serialises the shard-merge pass so the watchdog (and the reconcile
        #: endpoint) can never run it alongside the monitor thread.
        self._merge_lock = threading.Lock()
        # Per-job auto-resubmit note dedup (in-memory only): a deferred or
        # refused resubmit is logged loudly ONCE per distinct situation, not
        # once per 15 s reconcile tick — a 4 h maintenance window must not
        # write a thousand identical log lines. Restarts may re-log once.
        self._resubmit_notes: dict[str, tuple[str, str]] = {}
        # This manager's identity on durable resubmission claims (finding 1):
        # distinguishes claims across restarts/processes so a claim is only
        # ever released by the actor that took it.
        self._claimant = uuid.uuid4().hex[:8]
        if sweep_orphans:
            self._sweep_orphans()
        if reconcile_pipelines:
            # Opt-in (the daemon's ServiceState passes True): a scan of the
            # ambient runs root would surprise bare test constructions. A
            # failing scan must never take the daemon down with it.
            try:
                self._reconcile_orphaned_pipelines()
            except Exception as exc:  # noqa: BLE001
                print("[jobs] startup pipeline reconcile failed: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)

    def _slurm(self):
        if self._slurm_executor is None:
            from .executors import SlurmExecutor
            self._slurm_executor = SlurmExecutor()
        return self._slurm_executor

    def _sweep_orphans(self) -> int:
        """Fail local jobs left mid-flight by a crashed/restarted server. Their
        worker threads are gone, so ``running``/``pending`` is a lie. Slurm jobs
        are owned by the scheduler and left for the poller to reconcile."""
        swept = 0
        for job in list(self._jobs.values()):
            # A sharded parent still "pending" was orphaned mid fan-out (the
            # submit loop creates the parent first, attaches shards as they
            # submit, and flips the status itself — finding 3): the loop's
            # thread is gone, so the fan-out will never complete. Honest
            # terminal failure — and the already-submitted shards are
            # actively CANCELLED here (fan-out cleanup honesty, 2026-07-23),
            # with per-shard outcomes recorded; any shard whose cancel did
            # not go through is stamped ``cleanupIncomplete`` so the
            # reconciler keeps retrying until the scheduler confirms.
            if (job.executor == "slurm" and job.status == "pending"
                    and (job.requested_resources or {}).get("parallelJobs")):
                attached = [str(c) for c in
                            (job.requested_resources or {}).get("shardChildren")
                            or []]
                job.status = "failed"
                job.error = (
                    "sharded submission orphaned by server restart mid "
                    "fan-out — "
                    + (self.cancel_shards_honestly(job, attached)
                       if attached else
                       "no shard had been submitted; submit the study again"))
                job.finished_at = time.time()
                job.log(job.error)
                self.store.update(job)
                swept += 1
                continue
            if job.executor != "local" or job.status not in {"running", "pending",
                                                             "cancelling"}:
                continue
            if job.status == "cancelling":
                # The worker died before acknowledging, but the work IS
                # stopped (the thread is gone) and a cancel was requested —
                # "cancelled" is the honest terminal state.
                job.status = "cancelled"
                job.log("marked cancelled: cancel was pending when the server restarted")
            else:
                job.status = "failed"
                job.error = "orphaned by server restart"
                job.log("marked failed: worker thread lost on server restart")
            job.finished_at = time.time()
            self.store.update(job)
            swept += 1
        return swept

    def _live_pipeline_directories(self) -> set[str]:
        """Realpaths of every pipeline/run directory a NON-terminal job is
        following — job results (``pipelineDirectory``/``runDirectory``) plus
        every resume pointer in a live job's records directory. Biased WIDE
        on purpose: an extra path here only makes the orphan scan skip a
        ledger, never park a live one."""
        from ..experiment import resume as resume_mod
        live: set[str] = set()
        for job in list(self._jobs.values()):
            if job.status in TERMINAL:
                continue
            result = job.result if isinstance(job.result, dict) else {}
            for key in ("pipelineDirectory", "runDirectory",
                        "mergedRunDirectory"):
                value = result.get(key)
                if isinstance(value, str) and value:
                    live.add(os.path.realpath(value))
            resources = job.requested_resources or {}
            requested_pipeline = resources.get("pipelineDirectory")
            if isinstance(requested_pipeline, str) and requested_pipeline:
                live.add(os.path.realpath(requested_pipeline))
            records_dir = resources.get("recordsDirectory")
            if isinstance(records_dir, str) and os.path.isdir(records_dir):
                for entry in os.listdir(records_dir):
                    if not entry.endswith(resume_mod.POINTER_SUFFIX):
                        continue
                    pointer = resume_mod.read_pointer(
                        os.path.join(records_dir, entry))
                    pointed = (pointer or {}).get("runDirectory")
                    if isinstance(pointed, str) and pointed:
                        live.add(os.path.realpath(pointed))
        return live

    def _reconcile_orphaned_pipelines(self) -> int:
        """Startup pass (2026-08-06 replication-run incident): find
        pipeline ledgers with completed stages, a null disposition, and no
        live job following them, then RESUME each in-process when its
        remainder needs no model load, else stamp it ``parked`` with a
        reason the app can show. Never leaves a ledger silently orphaned —
        the failure mode where a controller restart made finished cluster
        results invisible in every UI."""
        from ..experiment import paths, pipeline_reconcile
        orphans = pipeline_reconcile.scan(
            paths.runs_directory(None), self._live_pipeline_directories())
        for orphan in orphans:
            if not orphan.resumable_schema:
                parked = pipeline_reconcile.park(
                    orphan.run_directory,
                    reason=(f"ledger schema {orphan.schema!r} predates the "
                            "resume contract — not resumable; import the "
                            "completed stage runs (evidence bundle) and "
                            "start a fresh pipeline for the rest"))
                print(f"[jobs] parked orphaned pipeline "
                      f"{os.path.basename(orphan.run_directory)} "
                      f"(schema {orphan.schema!r})"
                      if parked is not None else
                      f"[jobs] orphaned pipeline "
                      f"{os.path.basename(orphan.run_directory)} has an "
                      "unreadable ledger — could not park it",
                      file=sys.stderr)
                continue
            self._dispatch_orphaned_pipeline(orphan)
        return len(orphans)

    def _dispatch_orphaned_pipeline(self, orphan) -> Job:
        """One orphan → one durable job (visible in the app's job list, with
        logs): resume the remainder in-process when it needs no model and no
        judge fan-out, else park with the reason. A refused resume (e.g.
        genuine manifest drift) parks too — the refusal string IS the
        parked reason, so the outcome is loud either way. So does a resume
        that COMPLETED but could not package its evidence: reaching the
        researcher is the invariant, and a plain success with no bundle is
        the original incident wearing a green checkmark.

        Every one of those parks returns ``{"parked": True, …}``, which the
        generic runner stamps as the terminal ``parked`` STATUS — not
        ``succeeded`` (2026-08-06 review round 2). The job row and the
        ledger then say the same thing, and neither says "done"."""
        name = orphan.experiment
        pipeline_dir = orphan.run_directory
        remaining = list(orphan.remaining)

        def work(job: Job) -> dict:
            from ..experiment import pipeline_reconcile
            from ..experiment import tasks as tasks_mod
            from ..experiment.manifest import Manifest

            def _park(reason: str, **extra) -> dict:
                parked = pipeline_reconcile.park(pipeline_dir, reason=reason)
                job.log(f"parked orphaned pipeline "
                        f"{os.path.basename(pipeline_dir)}: {reason}"
                        if parked is not None else
                        f"could not stamp parked on {pipeline_dir} — "
                        f"ledger unreadable; reason was: {reason}")
                return {"parked": True, "reason": reason,
                        "pipelineDirectory": pipeline_dir,
                        "experiment": name, "verb": "pipeline", **extra}

            job.log(f"orphaned pipeline {os.path.basename(pipeline_dir)} "
                    f"(experiment '{name}'): completed stage(s) with no "
                    "live job — reconciling")
            try:
                manifest = Manifest.load(name)
            except Exception as exc:  # noqa: BLE001
                return _park("experiment manifest could not be loaded "
                             f"({type(exc).__name__}: {exc}) — the workspace "
                             "moved or the study was deleted; import the "
                             "completed stage runs if they are still needed")
            try:
                needs_model = tasks_mod._pipeline_needs_model(
                    remaining, manifest)
                evaluate_fresh = (
                    "evaluate" in remaining
                    and orphan.stage_status.get("evaluate")
                    != "awaitingJudgment")
                needs_fanout = bool(
                    evaluate_fresh
                    and tasks_mod.evaluate_fanout_judge_models(manifest))
            except Exception as exc:  # noqa: BLE001
                return _park("could not classify the remaining stages "
                             f"({type(exc).__name__}: {exc})")
            if needs_model or needs_fanout:
                return _park(
                    "remaining stage(s) " + ", ".join(remaining)
                    + (" need the study model"
                       if needs_model else
                       " need the judge fan-out (worker jobs)")
                    + " — resubmit the pipeline to continue, or import the "
                    "completed stage runs (evidence bundle)")
            try:
                out = tasks_mod.pipeline(
                    name, pipeline_run_directory=pipeline_dir,
                    should_cancel=lambda: job.cancelled, log=job.log)
            except Exception as exc:  # noqa: BLE001
                return _park(f"in-process resume refused: {exc}")
            job.log(f"orphaned pipeline resumed to completion → {out}")
            result = {"runDirectory": out, "experiment": name,
                      "verb": "pipeline"}
            # Package evidence like the bundle-execute path would have —
            # the whole point of the reconcile is that the finished chain's
            # results REACH the researcher.
            #
            # Packaging is a SUCCESS INVARIANT, not a nice-to-have (2026-08-06
            # review, P1): a healed ledger whose evidence never travels is
            # exactly the 2026-08-06 replication-run failure this feature
            # exists to close — completed cluster evidence sitting on the
            # server, unreachable from the Mac, with a job reporting plain
            # success.
            # A packaging failure therefore PARKS the chain, with the run
            # directory and the recovery action in the reason: the app's
            # awaiting-import surface reads `parked.reason` off the ledger and
            # offers the one-click import that packages on demand. The result
            # is additionally stamped ``evidenceMissing`` so the job row says
            # the same thing the ledger does.
            try:
                from ..experiment import bundles as bundles_mod
                result["evidenceBundle"] = bundles_mod.package_evidence(out)
                job.log("evidence bundle packaged → "
                        f"{result['evidenceBundle'].get('bundlePath')}")
            except Exception as pack_exc:  # noqa: BLE001
                return _park(
                    "the chain finished but its evidence could not be "
                    f"packaged ({type(pack_exc).__name__}: {pack_exc}) — the "
                    f"completed stage runs are on this server under {out}; "
                    "use Import evidence to package and bring them home, or "
                    "package by hand with POST /api/bundles/evidence",
                    evidenceMissing=True,
                    runDirectory=out,
                    evidencePackagingError=f"{type(pack_exc).__name__}: "
                                           f"{pack_exc}")
            return result

        return self.submit(
            "pipeline-orphan-reconcile", work,
            requested_resources={"pipelineDirectory": pipeline_dir,
                                 "experiment": name,
                                 "remainingStages": remaining})

    def submit(self, kind: str, fn: Callable[[Job], dict | None], *,
               requested_resources: dict | None = None,
               executor: str = "local") -> Job:
        snapshot = (
            self._capability_provider()
            if self._capability_provider is not None
            else capability_snapshot()
        )
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            requested_resources=requested_resources or {},
            executor=executor,
            capability_snapshot=snapshot,
            _store=self.store,
        )
        with self._lock:
            self._jobs[job.id] = job
            self.store.insert(job)

        def runner():
            if job.cancelled:
                # Cancelled before the worker picked it up: the work never
                # ran, so cancel()'s immediate terminal stamp was honest.
                job.status = "cancelled"
                job.finished_at = job.finished_at or time.time()
                job.log("cancelled before start — work never ran")
                self.store.update(job)
                return
            job.status = "running"
            job.started_at = time.time()
            self.store.update(job)
            # Any run directory stamped by this in-process worker carries the
            # job id (Slurm children get theirs from SLURM_JOB_ID instead).
            from ..experiment.run_config import current_job_id
            from ..experiment.run_status import current_run_directory
            token = current_job_id.set(job.id)
            # Scoped per job, like the job id: a worker thread is reused
            # across jobs in some paths, and inheriting a previous job's
            # run directory would package the wrong run on failure.
            run_dir_token = current_run_directory.set(None)
            try:
                result = fn(job)
                if job.cancelled:
                    # The one place a live worker's cancel becomes terminal:
                    # the work has actually returned, so "cancelled" is true.
                    # The RESULT is retained (review 2026-08-03 round 3, P1:
                    # dropping it orphaned cancel-parked sweep directories
                    # from the app — resumable on disk, invisible in the
                    # UI), and a run directory that parked with resume state
                    # stamps the resumable variant so the Resume affordance
                    # can find it. Auto-resubmit never touches this status —
                    # the user asked the run to stop; resuming is a
                    # deliberate act.
                    job.result = (result if isinstance(result, dict)
                                  else {"value": result})
                    parked = _resumable_directory_in(job.result)
                    if parked is not None:
                        job.status = "cancelledResumable"
                        job.log("cancellation observed — the run parked "
                                f"resumably at {parked}; Resume continues it")
                    else:
                        job.status = "cancelled"
                        job.log("cancellation observed")
                else:
                    job.result = result if isinstance(result, dict) else {"value": result}
                    # A worker that PARKED returned normally, but it did not
                    # finish the work — it stopped with durable state and a
                    # recovery action. Stamping that "succeeded" is how a
                    # chain needing a human came to render green in every
                    # client (2026-08-06 review round 2, P1).
                    if is_parked_result(job.result):
                        job.status = "parked"
                        job.log("parked — terminal, but NOT a completion: "
                                + str(job.result.get("reason")
                                      or "no reason recorded"))
                    else:
                        job.status = "succeeded"
                    if isinstance(job.result, dict):
                        artifacts = job.result.get("outputArtifacts")
                        if isinstance(artifacts, list):
                            job.output_artifacts = artifacts
            except Exception as exc:  # noqa: BLE001 - surfaced to the job
                if job.cancelled:
                    # Work aborted by (or during) a cancel: the user asked for
                    # this stop — report it as the cancel it is, keeping the
                    # abort reason on record instead of masquerading as a bug.
                    job.status = "cancelled"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.log(f"cancellation observed (work stopped: {job.error})")
                else:
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.log(f"ERROR: {job.error}")
                    # Retention for DIRECT jobs (external review
                    # 2026-07-24, finding 2). `execute_run_bundle` already
                    # packages failed BUNDLED work, but the direct
                    # /api/experiment/{name}/{verb} route runs tasks
                    # inline right here — so a failed server-resident
                    # evaluate/sweep/run recorded the error and stranded
                    # its output. Centralised in the durable-job layer, so
                    # every job kind is covered rather than one route.
                    #
                    # Runs BEFORE the terminal status flip: "failed" is the
                    # signal clients poll for, so the result must already be
                    # final when it appears — packaging after it left a
                    # window where a fast poller read the job without its
                    # partialEvidence stamp (visible whenever the handler's
                    # imports were cold).
                    _retain_partial_evidence(job, exc)
                    job.status = "failed"
            finally:
                current_job_id.reset(token)
                current_run_directory.reset(run_dir_token)
                job.finished_at = time.time()
                self.store.update(job)

        threading.Thread(target=runner, name=f"job-{job.id}", daemon=True).start()
        return job

    def record_external(self, kind: str, *, status: str, executor: str,
                        executor_job_id: str | None = None,
                        requested_resources: dict | None = None,
                        result: dict | None = None,
                        log: str | None = None,
                        job_id: str | None = None) -> Job:
        snapshot = (
            self._capability_provider()
            if self._capability_provider is not None
            else capability_snapshot()
        )
        now = time.time()
        job = Job(
            id=job_id or uuid.uuid4().hex[:12],
            kind=kind,
            status=status,
            created_at=now,
            started_at=(now if status in {"submitted", "running"} else None),
            # A job recorded terminal-at-birth (e.g. "prepared" dry runs) is
            # already finished; stamping finished_at keeps clients that key on
            # it (fallback polls, snapshots) from treating it as in-flight.
            finished_at=(now if status in TERMINAL else None),
            result=result,
            requested_resources=requested_resources or {},
            executor=executor,
            executor_job_id=executor_job_id,
            capability_snapshot=snapshot,
            _store=self.store,
        )
        with self._lock:
            self._jobs[job.id] = job
            self.store.insert(job)
        if log:
            job.log(log)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def cancel(self, job_id: str) -> bool:
        """Cancel a job (and its auto-resubmit descendants). Returns whether
        the cancellation was ACHIEVED or honestly accepted — a cooperative
        local-flag cancel counts (the worker observes it), but a Slurm scancel
        that exited nonzero returns False: the allocation may still be running
        and the caller must not report success (live shakedown 2026-07-16 —
        an ok-on-found answer hid exactly that). False also when the job id is
        unknown; callers who need to distinguish check ``get()`` first."""
        job = self.get(job_id)
        if job is None:
            return False
        # Sharded parent: cancelling it must stop the WHOLE fleet — every
        # shard child (each with its own resubmit chain) plus any pipeline
        # continuation. The parent itself owns no scheduler allocation, so
        # its terminal stamp is honest exactly when every child cancel was.
        rr = job.requested_resources or {}
        shard_children = rr.get("shardChildren") or []
        if shard_children:
            job._cancel.set()
            self.store.mark_cancel_requested(job.id)
            job.log("cancel requested for sharded parent — cancelling every "
                    "shard job")
            achieved = True
            targets = [str(cid) for cid in shard_children]
            if rr.get("continuationJob"):
                targets.append(str(rr["continuationJob"]))
            # Judge fan-out workers (2026-07-23) are siblings too: stopping
            # the fan-out must stop every judging allocation.
            for worker in (rr.get("judgeFanout") or {}).get("workers") or []:
                if isinstance(worker, dict) and worker.get("jobId"):
                    targets.append(str(worker["jobId"]))
            unconfirmed: list[str] = []
            for cid in targets:
                child = self.get(cid)
                if child is not None and child.status not in TERMINAL:
                    one = self.cancel(cid)
                    achieved = one and achieved
                    if not one:
                        unconfirmed.append(cid)
            if achieved and job.status not in TERMINAL:
                job.status = "cancelled"
                job.finished_at = time.time()
            elif not achieved:
                # Uncertainty is STAMPED (finding 3, 2026-07-23): the
                # reconciler retries these cancellations every tick until
                # the scheduler confirms — the parent never quietly forgets
                # a possibly-still-billing shard.
                if unconfirmed:
                    job.result = {**(job.result or {}),
                                  "cleanupIncomplete": {
                                      "pendingCancel": unconfirmed}}
                job.log("shard cancel did not go through for "
                        f"{', '.join(unconfirmed) or 'one or more shards'} — "
                        "the parent stays non-terminal; the reconciler "
                        "retries until confirmed (or retry the cancel)")
            self.store.update(job)
            return achieved
        achieved = self._cancel_one(job)
        # Auto-resubmit chains (WS2): cancelling the ORIGINAL job must not
        # leave its resubmitted continuation running. Walk result.resubmittedAs
        # to the newest descendant and cancel every non-terminal link — each
        # gets the persisted cancel flag too, so none of them can ever be
        # auto-resubmitted after this. A cycle in corrupt data is bounded by
        # ``seen``. One failed scancel anywhere in the chain fails the verb:
        # "ok" must mean the WHOLE chain is stopped.
        seen = {job.id}
        current = job
        while True:
            next_id = (current.result or {}).get("resubmittedAs")
            if not next_id or next_id in seen:
                break
            child = self.get(str(next_id))
            if child is None:
                break
            seen.add(child.id)
            if child.status not in TERMINAL:
                child.log(f"cancelled with ancestor {job.id} (auto-resubmit chain)")
                achieved = self._cancel_one(child) and achieved
            current = child
        return achieved

    def _cancel_one(self, job: Job) -> bool:
        job._cancel.set()
        self.store.mark_cancel_requested(job.id)
        job.log("cancel requested")
        achieved = True
        # Scheduler-owned work must be cancelled through the scheduler; setting a
        # cooperative flag alone would leave the Slurm allocation running.
        if job.executor == "slurm" and job.executor_job_id and job.status not in TERMINAL:
            try:
                ok = self._slurm().cancel(job.executor_job_id)
            except Exception as exc:  # noqa: BLE001 - report, don't crash the API
                job.log(f"scancel failed: {exc}")
                ok = False
            if ok:
                job.log(f"scancel {job.executor_job_id}")
                # scancel kills the allocation itself, so terminal is honest here.
                job.status = "cancelled"
                job.finished_at = time.time()
            else:
                # A nonzero scancel is NOT a cancellation: the allocation may
                # still be running/queued (live shakedown 2026-07-16 — a stop
                # that stamps terminal here hides a billed orphan). Status
                # stays non-terminal so the reconciler keeps following the
                # job and the caller can retry or scancel by hand.
                job.log(f"scancel {job.executor_job_id} exited nonzero — the "
                        "allocation may still be running; retry the cancel or "
                        f"run `scancel {job.executor_job_id}` on the cluster")
                achieved = False
        elif job.status == "pending":
            # The worker never started this job; the runner's started-check
            # guarantees the work will not run, so terminal is honest.
            job.status = "cancelled"
            job.finished_at = time.time()
        elif job.status == "running":
            # Honest lifecycle: the worker thread is still executing and only
            # polls the flag cooperatively. Non-terminal "cancelling" keeps
            # SSE streams open and the UI truthful until the worker actually
            # exits and its runner stamps the terminal "cancelled". This IS
            # an accepted cancel: the flag is persisted and the worker will
            # observe it — unlike a refused scancel, nothing was lost.
            job.status = "cancelling"
        self.store.update(job)
        return achieved

    def poll_slurm(self) -> int:
        """Reconcile non-terminal Slurm jobs against the scheduler. Updates each
        job's status from ``sacct``/``squeue`` and, on reaching a terminal state,
        folds any child JSON records (the run directory + artifacts) into the
        store. Checkpointed jobs (whether observed via the sacct exit-code
        mapping or via a folded child record stamped "checkpointed") then pass
        through the auto-resubmit gate every tick — that is what lets a resubmit
        deferred by a maintenance window fire on a later tick, and what repairs
        a missing resubmittedAs stamp after a crash/restart. Returns the number
        of jobs whose status changed or that were resubmitted/repaired."""
        changed = 0
        pending = [j for j in self.list()
                   if j.executor == "slurm" and j.executor_job_id and j.status not in TERMINAL]
        for job in pending:
            try:
                state = self._slurm().poll_state(job.executor_job_id)
            except Exception:  # noqa: BLE001 - a flaky sacct must not kill the loop
                continue
            if state is not None and state != job.status:
                job.status = state
                job.log(f"slurm {job.executor_job_id} → {state}")
                if state == "checkpointed" or state in TERMINAL:
                    if state == "checkpointed":
                        # Checkpoint exit (85): NOT a failure and NOT finished —
                        # the run directory holds a flushed generations.jsonl
                        # plus resume-state.json, and a requeue/resubmit of the
                        # same submission continues it record-by-record. Fold
                        # the child record now so the resumable run directory
                        # is visible.
                        job.log("child exited with the checkpoint code (85): run is "
                                "resumable — press Resume in the app (POST "
                                f"/api/jobs/{job.id}/resubmit) to continue it, or "
                                "auto-resume (autoResubmit) re-submits it "
                                "automatically when enabled")
                    else:
                        job.finished_at = time.time()
                    # The child record replaces job.result on fold, so later
                    # transitions fall back to the stamped requestedResources
                    # copy of the records directory.
                    records_dir = ((job.result or {}).get("recordsDirectory")
                                   or (job.requested_resources or {}).get("recordsDirectory"))
                    if records_dir:
                        self.reconcile(records_dir)
                    # The fold may have CORRECTED the status from the child
                    # record — e.g. a checkpoint the wrapper reported as plain
                    # FAILED, or a completed continuation folding back over a
                    # checkpointed original. Keep the non-terminal/finished_at
                    # invariant honest either way.
                    if job.status == "checkpointed":
                        job.finished_at = None
                    elif job.status in TERMINAL and job.finished_at is None:
                        job.finished_at = time.time()
                    # A terminal FAILURE whose child wrote NO record died
                    # before it could say why (walltime kill, preemption,
                    # OOM) — the app then showed a bare "failed" with no
                    # reason at all (field incident 2026-08-02: a sweep hit
                    # its 30-minute walltime invisibly). Ask sacct HOW the
                    # scheduler ended it and stamp that as the error.
                    if (job.status in TERMINAL and job.status != "succeeded"
                            and not (job.result or {}).get("error")):
                        try:
                            detail = self._slurm().death_detail(
                                job.executor_job_id)
                        except Exception:  # noqa: BLE001 - best-effort forensics
                            detail = None
                        if detail:
                            message = (
                                f"{detail} — the job wrote no completion "
                                "record, so it was killed before finishing; "
                                "if elapsed matches the walltime limit, "
                                "raise the submission's walltime")
                            job.result = {**(job.result or {}),
                                          "error": message}
                            job.log(message)
                self.store.update(job)
                changed += 1
            if job.status == "checkpointed":
                try:
                    if self._maybe_auto_resubmit(job):
                        changed += 1
                except Exception as exc:  # noqa: BLE001 - one job must not kill the loop
                    self._resubmit_note(
                        job, "error",
                        f"auto-resubmit error: {type(exc).__name__}: {exc}")
        # Sharded parents derive their state from their shard children (and
        # run the merge when every shard has succeeded) on the same tick.
        try:
            changed += self.run_merge_pass()
        except Exception as exc:  # noqa: BLE001 - never let the monitor die
            # …but never let it die SILENTLY either (open-issues §2): a
            # swallowed exception here is indistinguishable from a healthy
            # tick that found nothing to merge.
            self._monitor_note(f"shard-merge pass failed: "
                               f"{type(exc).__name__}: {exc}")
        # Unconfirmed fan-out cleanups (finding 3, 2026-07-23): keep
        # retrying shard cancellations until the scheduler confirms them.
        try:
            changed += self._retry_incomplete_cleanups()
        except Exception as exc:  # noqa: BLE001 - never let the monitor die
            self._monitor_note(f"cleanup retry pass failed: "
                               f"{type(exc).__name__}: {exc}")
        return changed

    # --- fan-out cleanup honesty (external review, 2026-07-23) ----------------

    def cancel_shards_honestly(self, parent: "Job",
                               shard_ids: list[str]) -> str:
        """Attempt to cancel every shard in ``shard_ids``, RECORDING the
        per-shard outcome instead of assuming success (finding 3: the abort
        path reported "were cancelled" without reading ``cancel()``'s
        boolean, so a failed scancel read as a clean stop while the
        allocation kept billing). Returns the honest plain-language account
        for the parent's error message; any shard whose cancellation is
        uncertain is stamped ``cleanupIncomplete`` on the parent's result,
        which the reconciler retries every tick until the scheduler
        confirms (``_retry_incomplete_cleanups``)."""
        cancelled: list[str] = []
        failed: list[tuple[str, str | None]] = []
        for cid in [str(c) for c in shard_ids]:
            ok = False
            try:
                ok = self.cancel(cid)
            except Exception:  # noqa: BLE001 - an errored cancel is a failed cancel
                ok = False
            if ok:
                cancelled.append(cid)
            else:
                child = self.get(cid)
                failed.append((cid, child.executor_job_id if child else None))
        if failed:
            parent.result = {**(parent.result or {}),
                             "cleanupIncomplete": {
                                 "pendingCancel": [cid for cid, _ in failed]}}
        parts: list[str] = []
        if cancelled:
            n = len(cancelled)
            parts.append(f"shard job(s) {', '.join(cancelled)} "
                         f"{'were' if n != 1 else 'was'} cancelled (confirmed)")
        if failed:
            named = ", ".join(
                f"{cid} (Slurm {slurm_id or 'id unknown'})"
                for cid, slurm_id in failed)
            parts.append(
                f"cancel FAILED for shard job(s) {named} — their "
                "allocations may still be running; the reconciler keeps "
                "retrying the cancellation until the scheduler confirms, "
                "or run scancel by hand")
        return "; ".join(parts) if parts else "no shards had been submitted"

    def _retry_incomplete_cleanups(self) -> int:
        """Reconciler pass over every job stamped ``cleanupIncomplete``:
        retry the pending shard cancellations (a parent may already be
        terminal — its own state never changes here) and clear the stamp
        only when every shard's stop is scheduler-confirmed. Returns the
        number of jobs whose cleanup state changed."""
        changed = 0
        for job in self.list():
            stamp = (job.result or {}).get("cleanupIncomplete")
            pending = (stamp or {}).get("pendingCancel")
            if not pending:
                continue
            still: list[str] = []
            for cid in [str(c) for c in pending]:
                child = self.get(cid)
                if child is not None and child.status in TERMINAL:
                    continue  # confirmed stopped (or never ran)
                ok = False
                try:
                    ok = self.cancel(cid)
                except Exception:  # noqa: BLE001 - retried next tick
                    ok = False
                child = self.get(cid)
                if ok and (child is None or child.status in TERMINAL):
                    continue
                still.append(cid)
            if still == [str(c) for c in pending]:
                continue  # nothing moved this tick
            if still:
                job.result = {**(job.result or {}),
                              "cleanupIncomplete": {"pendingCancel": still}}
                job.log("cleanup retry: shard cancellation still unconfirmed "
                        f"for {', '.join(still)} — will retry next tick")
            else:
                job.result = {k: v for k, v in (job.result or {}).items()
                              if k != "cleanupIncomplete"}
                job.log("cleanup complete: every previously-unconfirmed shard "
                        "cancellation is now scheduler-confirmed")
            self.store.update(job)
            changed += 1
        return changed

    # --- sharded parents (multi-GPU fan-out, 2026-07-22) ----------------------

    def _shard_chain_tail(self, job: "Job") -> "Job":
        """Follow a shard child's auto/manual-resubmit chain to the newest
        continuation record (cycles in corrupt data bounded by ``seen``)."""
        seen = {job.id}
        current = job
        while True:
            next_id = (current.result or {}).get("resubmittedAs")
            if not next_id or next_id in seen:
                return current
            child = self.get(str(next_id))
            if child is None:
                return current
            seen.add(child.id)
            current = child

    def _shard_effective(self, child: "Job") -> tuple[str, str | None]:
        """One shard's effective ``(state, run_directory)`` with its resubmit
        chain collapsed: a succeeded link anywhere wins (the child record
        folds back onto the ORIGINAL id, so success can appear on either
        end); otherwise the newest link speaks."""
        seen = {child.id}
        links = [child]
        current = child
        while True:
            next_id = (current.result or {}).get("resubmittedAs")
            if not next_id or next_id in seen:
                break
            nxt = self.get(str(next_id))
            if nxt is None:
                break
            seen.add(nxt.id)
            links.append(nxt)
            current = nxt
        for link in links:
            if link.status == "succeeded":
                run_dir = (link.result or {}).get("runDirectory")
                if not run_dir:
                    inner = (link.result or {}).get("result")
                    if isinstance(inner, dict):
                        run_dir = inner.get("runDirectory")
                return "succeeded", run_dir
        return links[-1].status, None

    def _continuation_evidence(self, continuation: "Job") -> dict | None:
        """The evidence bundle the pipeline continuation's bundle-execute
        child packaged over the finished pipeline directory (folded into its
        result by the reconciler, in either record shape), walking its
        resubmit chain to the succeeded link exactly like
        ``_shard_effective``."""
        seen: set[str] = set()
        current: Job | None = continuation
        while current is not None and current.id not in seen:
            seen.add(current.id)
            if current.status == "succeeded":
                result = current.result or {}
                evidence = result.get("evidenceBundle")
                if not isinstance(evidence, dict):
                    inner = result.get("result")
                    evidence = (inner.get("evidenceBundle")
                                if isinstance(inner, dict) else None)
                return evidence if isinstance(evidence, dict) else None
            next_id = (current.result or {}).get("resubmittedAs")
            current = self.get(str(next_id)) if next_id else None
        return None

    # --- judge fan-out (post-generation judging stage, 2026-07-23) ------------

    def _continuation_fanout_request(self, continuation: "Job") -> dict | None:
        """The judge fan-out request the pipeline continuation's child
        record carried (the chain stopped at the evaluate stage after
        emitting packets), walking the resubmit chain to the succeeded link
        exactly like ``_continuation_evidence``."""
        seen: set[str] = set()
        current: Job | None = continuation
        while current is not None and current.id not in seen:
            seen.add(current.id)
            if current.status == "succeeded":
                result = current.result or {}
                request = result.get("awaitingJudgeFanout")
                if not isinstance(request, dict):
                    inner = result.get("result")
                    request = (inner.get("awaitingJudgeFanout")
                               if isinstance(inner, dict) else None)
                return request if isinstance(request, dict) else None
            next_id = (current.result or {}).get("resubmittedAs")
            current = self.get(str(next_id)) if next_id else None
        return None

    @staticmethod
    def _judge_model_slug(model: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_." else "-"
                       for ch in model)

    def _start_judge_fanout(self, parent: "Job", continuation_id: str,
                            request: dict) -> bool:
        """Submit ONE worker job per distinct local judge model (sibling
        jobs under the pipeline parent, exactly like shard siblings —
        records, resume, cancel, and honest states all reuse the sibling
        machinery). The generation continuation is retired into the
        ``judgeFanout`` block; the merge submits the FINAL continuation."""
        rr = parent.requested_resources or {}
        merge_cfg = rr.get("shardMerge") or {}
        name = str(request.get("experiment") or merge_cfg.get("experiment")
                   or "")
        evaluate_run = str(request.get("evaluateRun") or "")
        judge_models = [m for m in (request.get("judgeModels") or [])
                        if isinstance(m, dict) and m.get("model")]
        if not (name and evaluate_run and judge_models):
            parent.status = "failed"
            parent.error = ("judge fan-out request from continuation "
                            f"{continuation_id} is malformed (experiment/"
                            "evaluateRun/judgeModels) — judge manually via "
                            "the deferred-judgment flow and resume")
            parent.finished_at = time.time()
            parent.log(parent.error)
            self.store.update(parent)
            return True
        from .executors import SlurmResources
        records_dir = str(rr.get("recordsDirectory") or "")
        submission_dir = str(merge_cfg.get("submissionDirectory")
                             or os.path.dirname(records_dir))
        artifacts_dir = os.path.join(submission_dir, "judge-artifacts")
        resource_fields = {f.name for f in
                           SlurmResources.__dataclass_fields__.values()}
        resources = SlurmResources(**{k: v for k, v in rr.items()
                                      if k in resource_fields})
        target = str(merge_cfg.get("targetRoot") or "")
        python = os.environ.get("STEERLAB_PYTHON") or sys.executable or "python"
        workers: list[dict] = []
        for index, entry in enumerate(judge_models):
            model = str(entry["model"])
            slug = self._judge_model_slug(model)
            child_id = uuid.uuid4().hex[:12]
            record_path = os.path.join(records_dir, f"{child_id}.json")
            artifact_path = os.path.join(artifacts_dir, f"{slug}.json")
            command = [python, "-m", "steerlab_server.cli", "experiment",
                       "judge-worker", name,
                       "--awaiting-run", evaluate_run,
                       "--model", model,
                       "--out", artifact_path,
                       "--record", record_path,
                       "--root", target]
            if entry.get("revision"):
                command += ["--revision", str(entry["revision"])]
            if entry.get("dtype"):
                command += ["--dtype", str(entry["dtype"])]
            if merge_cfg.get("device"):
                command += ["--device", str(merge_cfg["device"])]
            bundle = self._slurm().create_bundle(
                os.path.join(submission_dir, f"slurm-judge-{index}-{slug}"),
                command, env={"STEERLAB_JOB_ID": child_id},
                resources=resources,
                metadata={"kind": "judgeWorker", "experiment": name,
                          "evaluateRun": evaluate_run, "judgeModel": model,
                          "parentJob": parent.id,
                          "recordsDirectory": records_dir})
            slurm_id = self._slurm().submit(bundle)
            stamped = dict(rr)
            for drop in ("shardChildren", "shardMerge", "parallelJobs",
                         "continuationJob", "judgeFanout"):
                stamped.pop(drop, None)
            stamped.update({"scriptPath": bundle.script_path,
                            "recordsDirectory": records_dir,
                            "parentJob": parent.id,
                            "judgeWorker": {"model": model,
                                            "revision": entry.get("revision"),
                                            "dtype": entry.get("dtype"),
                                            "judges": entry.get("judges"),
                                            "artifactPath": artifact_path}})
            worker = self.record_external(
                "study-judge-worker", status="submitted", executor="slurm",
                executor_job_id=str(slurm_id), job_id=child_id,
                requested_resources=stamped,
                result={"recordsDirectory": records_dir,
                        "parentJob": parent.id},
                log=(f"judge worker for model '{model}' (judges "
                     f"{', '.join(str(j) for j in entry.get('judges') or [])}) "
                     f"over awaiting run {evaluate_run}, as Slurm job "
                     f"{slurm_id}"))
            workers.append({"jobId": worker.id, "model": model,
                            "artifactPath": artifact_path})
        new_rr = dict(rr)
        new_rr.pop("continuationJob", None)
        new_rr["judgeFanout"] = {
            "evaluateRun": evaluate_run,
            "experiment": name,
            "packetsSha256": request.get("packetsSha256"),
            "generationContinuation": continuation_id,
            "workers": workers,
            "merged": False,
        }
        parent.requested_resources = new_rr
        parent.status = "running"
        parent.log(
            f"generation finished; evaluate emitted "
            f"{request.get('packetCount')} blinded packet(s) — judging fans "
            f"out to {len(workers)} judge-model worker job(s): "
            + ", ".join(f"{w['model']} ({w['jobId']})" for w in workers))
        self.store.update(parent)
        return True

    def _reconcile_judge_fanout(self, parent: "Job") -> bool:
        """Derive the judge fan-out's state from its worker jobs (resubmit
        chains collapsed exactly like shard children); when EVERY worker
        succeeded, merge their judgment artifacts through
        ``complete_evaluate_judgment`` — which refuses unless every judge ×
        response-pair cell appears exactly once — and submit the FINAL
        pipeline continuation for the remaining stages."""
        rr = parent.requested_resources or {}
        fanout = rr.get("judgeFanout") or {}
        workers = fanout.get("workers") or []
        records = [self.get(str(w.get("jobId"))) for w in workers]
        if any(record is None for record in records):
            missing = [str(w.get("jobId")) for w, record
                       in zip(workers, records) if record is None]
            if parent.status != "failed":
                parent.status = "failed"
                parent.error = ("judge fan-out lost track of worker job "
                                f"record(s) {missing} — cannot derive state")
                parent.finished_at = time.time()
                parent.log(parent.error)
                self.store.update(parent)
                return True
            return False
        effective = [self._shard_effective(record) for record in records]
        derived = derive_shard_parent_state([state for state, _ in effective])
        if derived == "succeeded":
            return self._merge_judge_fanout(parent, fanout, workers, records)
        if derived != parent.status:
            parent.status = derived
            if derived in TERMINAL:
                parent.finished_at = time.time()
                bad = [record.id for record, (state, _)
                       in zip(records, effective)
                       if state in ("failed", "cancelled")]
                parent.error = (
                    f"judge worker job(s) {', '.join(bad)} ended without "
                    "success — the judge fan-out cannot complete; finished "
                    "workers' judgment artifacts are intact" if bad
                    else parent.error)
                parent.log(f"judge fan-out {derived}"
                           + (f": {parent.error}" if parent.error else ""))
            elif derived == "checkpointed":
                parent.log("one or more judge workers checkpointed "
                           "(resumable) — Resume on this parent resumes "
                           "them")
            else:
                parent.log(f"judge workers → {derived}")
            self.store.update(parent)
            return True
        return False

    def _merge_judge_fanout(self, parent: "Job", fanout: dict,
                            workers: list[dict],
                            records: list["Job"]) -> bool:
        from ..experiment import tasks as tasks_mod
        rr = parent.requested_resources or {}
        merge_cfg = rr.get("shardMerge") or {}
        name = str(fanout.get("experiment") or "")
        evaluate_run = str(fanout.get("evaluateRun") or "")
        target = str(merge_cfg.get("targetRoot") or "")
        parent.log(f"all {len(workers)} judge worker(s) succeeded — merging "
                   "judgment artifacts")
        try:
            rows: list[dict] = []
            for worker in workers:
                artifact = tasks_mod.read_judge_worker_artifact(
                    str(worker.get("artifactPath")),
                    expected_packets_sha=fanout.get("packetsSha256"))
                rows.extend(artifact.get("rows") or [])
            completion_dir = tasks_mod.complete_evaluate_judgment(
                name, evaluate_run, rows, root=target, log=parent.log)
            # Mark the pipeline ledger's evaluate stage completed so the
            # final continuation resumes straight into the remaining
            # CPU-only stages without reloading the model.
            pipeline_dir = ((parent.result or {}).get("pipelineDirectory")
                            or "")
            if pipeline_dir:
                self._mark_pipeline_evaluate_completed(
                    pipeline_dir, completion_dir, log=parent.log)
            new_rr = dict(parent.requested_resources or {})
            new_rr["judgeFanout"] = {**fanout, "merged": True,
                                     "completionRun": completion_dir}
            parent.requested_resources = new_rr
            continuation = self._submit_pipeline_continuation(
                parent, pipeline_dir, name, merge_cfg,
                bundle_subdir="slurm-continuation-postjudge",
                log_line=(f"post-judging continuation of {parent.id}: "
                          "remaining pipeline stage(s) after the judge "
                          "merge"))
            parent.status = "running"
            parent.log(f"judge merge complete → {completion_dir}; final "
                       f"pipeline continuation submitted as "
                       f"{continuation.id} (Slurm "
                       f"{continuation.executor_job_id})")
            self.store.update(parent)
            return True
        except Exception as exc:  # noqa: BLE001 - surfaced on the parent
            parent.status = "failed"
            parent.error = (f"judge fan-out merge refused: "
                            f"{type(exc).__name__}: {exc} — worker judgment "
                            "artifacts are intact under the submission "
                            "directory")
            parent.finished_at = time.time()
            parent.log(parent.error)
            self.store.update(parent)
            return True

    @staticmethod
    def _mark_pipeline_evaluate_completed(pipeline_dir: str,
                                          completion_dir: str, log) -> None:
        """Fold the judge merge's completion run into the pipeline ledger:
        evaluate → completed. Best-effort — the pipeline's own resume also
        adopts a completed judgment run, so a failed write only costs one
        redundant model-free adoption pass."""
        path = os.path.join(pipeline_dir, "pipeline.json")
        try:
            with open(path, encoding="utf-8") as handle:
                ledger = json.load(handle)
            stage = (ledger.get("stageResults") or {}).get("evaluate") or {}
            if stage.get("status") == "awaitingJudgment":
                ledger["stageResults"]["evaluate"] = {
                    "status": "completed", "runDirectory": completion_dir,
                    "judgedVia": "judgeWorkerFanout"}
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as handle:
                    json.dump(ledger, handle, indent=2, sort_keys=True)
                os.replace(tmp, path)
        except (OSError, ValueError) as exc:
            log(f"could not fold the judge merge into the pipeline ledger "
                f"({exc}) — the continuation adopts the completed judgment "
                "run itself")

    def _reconcile_shard_parents(self) -> int:
        """Derive every sharded parent's state from its shard children; when
        every shard has succeeded, run the merge (completeness-refusing) and,
        for a run-first pipeline, submit the continuation job. Returns the
        number of parents whose state changed."""
        changed = 0
        # "pending" parents are still ATTACHING their shard children (the
        # fan-out submit loop creates the parent first, finding 3) — deriving
        # state from a partial child list would try to merge a half-submitted
        # fan-out. The submit loop flips them to submitted/failed itself; a
        # crash mid-loop is swept at restart (_sweep_orphans).
        parents = [j for j in self.list()
                   if j.status not in TERMINAL and j.status != "pending"
                   and (j.requested_resources or {}).get("shardChildren")]
        for parent in parents:
            try:
                if self._reconcile_one_shard_parent(parent):
                    changed += 1
            except Exception as exc:  # noqa: BLE001 - one parent must not kill the loop
                parent.log(f"shard reconcile error: {type(exc).__name__}: {exc}")
        return changed

    def _reconcile_one_shard_parent(self, parent: "Job") -> bool:
        rr = parent.requested_resources or {}
        fanout = rr.get("judgeFanout")
        if isinstance(fanout, dict) and not fanout.get("merged"):
            # Judge fan-out phase (2026-07-23): worker jobs are judging;
            # the merge (and the final continuation) happen here.
            return self._reconcile_judge_fanout(parent)
        continuation_id = rr.get("continuationJob")
        if continuation_id:
            return self._follow_continuation(parent, str(continuation_id))
        children = [self.get(str(cid)) for cid in rr.get("shardChildren") or []]
        if any(child is None for child in children):
            missing = [cid for cid, child
                       in zip(rr.get("shardChildren") or [], children)
                       if child is None]
            if parent.status != "failed":
                parent.status = "failed"
                parent.error = ("sharded parent lost track of shard job "
                                f"record(s) {missing} — cannot derive state")
                parent.finished_at = time.time()
                parent.log(parent.error)
                self.store.update(parent)
                return True
            return False
        effective = [self._shard_effective(child) for child in children]
        derived = derive_shard_parent_state([state for state, _ in effective])
        if derived == "succeeded":
            run_dirs = [run_dir for _, run_dir in effective]
            return self._merge_shard_parent(parent, children, run_dirs)
        if derived != parent.status:
            parent.status = derived
            if derived in TERMINAL:
                parent.finished_at = time.time()
                bad = [child.id for child, (state, _) in zip(children, effective)
                       if state in ("failed", "cancelled")]
                parent.error = (
                    f"shard job(s) {', '.join(bad)} ended without success — "
                    "the fan-out cannot complete; the other shards' partials "
                    "are intact" if bad else parent.error)
                parent.log(f"sharded run {derived}"
                           + (f": {parent.error}" if parent.error else ""))
            elif derived == "checkpointed":
                parent.log(
                    "one or more shards checkpointed (resumable) — Resume on "
                    "this parent resumes every checkpointed shard, or "
                    "auto-resume continues them when enabled")
            else:
                parent.log(f"shards → {derived}")
            self.store.update(parent)
            return True
        return False

    def _follow_continuation(self, parent: "Job", continuation_id: str) -> bool:
        """A run-first pipeline's post-merge continuation job carries the
        remaining stages; the parent adopts its terminal outcome — including
        the continuation's evidence bundle (external review 2026-07-22,
        finding 2): the app imports the PARENT's bundle, so the parent must
        carry the final pipeline evidence (judge outputs, analysis, ledger),
        not a run-only snapshot from merge time."""
        continuation = self.get(continuation_id)
        if continuation is None:
            return False
        state, run_dir = self._shard_effective(continuation)
        mapped = state if state in ("succeeded", "failed", "cancelled",
                                    "checkpointed") else "running"
        if mapped == "succeeded":
            # Judge fan-out handshake (2026-07-23): a pipeline continuation
            # that STOPPED at the evaluate stage carries the fan-out request
            # in its child record — the parent does not succeed; it submits
            # one worker job per distinct judge model and waits for the
            # merge.
            request = self._continuation_fanout_request(continuation)
            if request is not None:
                return self._start_judge_fanout(parent, continuation_id,
                                                request)
            result = dict(parent.result or {})
            evidence = self._continuation_evidence(continuation)
            merge_cfg = (parent.requested_resources or {}).get("shardMerge") or {}
            if merge_cfg.get("packageEvidence", True) and evidence is None:
                # Evidence-as-success invariant (finding 4, 2026-07-23):
                # evidence packaging was REQUESTED, and the app imports the
                # PARENT's bundle — a parent stamped "succeeded" with no
                # evidence block looks imported while the pipeline outputs
                # stay on the cluster. That is a failed contract, reported
                # as one, with the recovery spelled out.
                pipeline_dir = (result.get("pipelineDirectory") or run_dir
                                or result.get("mergedRunDirectory")
                                or "<pipeline directory unknown>")
                parent.status = "failed"
                parent.error = (
                    "evidence_missing: the pipeline continuation "
                    f"{continuation_id} finished but returned no evidence "
                    "bundle, and evidence packaging was requested for this "
                    "submission. The pipeline outputs are intact on the "
                    f"cluster at {pipeline_dir} — package and import them "
                    "manually: on the server run `steerlab-server bundle "
                    f"evidence {pipeline_dir}`, then import the produced "
                    "bundle from the app (Runs → import evidence bundle)")
                parent.finished_at = time.time()
                result["continuationJob"] = continuation_id
                result["evidenceMissing"] = True
                if run_dir:
                    result["runDirectory"] = run_dir
                parent.result = result
                parent.log(parent.error)
                self.store.update(parent)
                return True
            parent.status = "succeeded"
            parent.finished_at = time.time()
            if run_dir:
                result["runDirectory"] = run_dir
            if evidence is not None:
                result["evidenceBundle"] = evidence
            result["continuationJob"] = continuation_id
            parent.result = result
            parent.log(f"pipeline continuation {continuation_id} succeeded"
                       + (f" → {run_dir}" if run_dir else "")
                       + (" — pipeline evidence bundle folded onto this parent"
                          if evidence is not None else ""))
            self.store.update(parent)
            return True
        if mapped != parent.status:
            parent.status = mapped
            if mapped in TERMINAL:
                parent.finished_at = time.time()
                parent.error = continuation.error or parent.error
            parent.log(f"pipeline continuation {continuation_id} → {mapped}")
            self.store.update(parent)
            return True
        return False

    def _merge_shard_parent(self, parent: "Job", children: list["Job"],
                            run_dirs: list[str | None]) -> bool:
        """All shards succeeded: assemble the merged run. The merge REFUSES
        (loudly, partials intact) on any missing/duplicated cell; a refusal
        fails the parent with the merge's own words."""
        from ..experiment import sharding
        rr = parent.requested_resources or {}
        merge_cfg = rr.get("shardMerge") or {}
        if (parent.result or {}).get("runDirectory"):
            return False  # already merged (idempotency across ticks)
        parent.status = "merging"
        parent.log(f"all {len(children)} shard(s) succeeded — merging partials")
        self.store.update(parent)
        try:
            missing = [child.id for child, run_dir
                       in zip(children, run_dirs) if not run_dir]
            if missing:
                raise sharding.ShardMergeError(
                    "shard merge refused: shard job(s) "
                    f"{', '.join(missing)} recorded no run directory — "
                    "reconcile their child records first")
            merged = sharding.merge_shard_runs(
                str(merge_cfg.get("experiment") or ""),
                [str(d) for d in run_dirs],
                root=str(merge_cfg.get("targetRoot") or ""),
                shard_job_ids=[child.id for child in children],
                # The parent record's id, explicitly: the merge runs on the
                # controller's monitor thread, where the env fallback would
                # stamp the CONTROLLER's own Slurm allocation into every
                # merged run's config.json (the stale-jobId bug, 2026-08-06).
                job_id=parent.id,
                log=parent.log)
            if merge_cfg.get("verb") == "pipeline":
                # Evidence packaging is DEFERRED for pipelines (external
                # review 2026-07-22, finding 2): the bundle the app imports
                # must be the FINAL pipeline evidence (judge outputs,
                # analysis, ledger), which exists only after the remaining
                # stages complete — a merged-RUN bundle stored here would
                # look imported while the judgments stayed on the cluster.
                return self._start_pipeline_continuation(
                    parent, merged, merge_cfg)
            evidence = None
            if merge_cfg.get("packageEvidence", True):
                from ..experiment import bundles as bundles_mod
                evidence = bundles_mod.package_evidence(merged)
                parent.log(f"evidence bundle packaged for the merged run "
                           f"→ {evidence.get('bundlePath')}")
            parent.status = "succeeded"
            parent.finished_at = time.time()
            result = dict(parent.result or {})
            result["runDirectory"] = merged
            if evidence is not None:
                result["evidenceBundle"] = evidence
            parent.result = result
            parent.log(f"sharded run complete → {merged}")
            self.store.update(parent)
            return True
        except sharding.ShardMergeError as exc:
            parent.status = "failed"
            parent.error = str(exc)
            parent.finished_at = time.time()
            parent.log(str(exc))
            self.store.update(parent)
            return True
        except Exception as exc:  # noqa: BLE001 - surfaced on the parent
            parent.status = "failed"
            parent.error = f"shard merge failed: {type(exc).__name__}: {exc}"
            parent.finished_at = time.time()
            parent.log(parent.error)
            self.store.update(parent)
            return True

    def _start_pipeline_continuation(self, parent: "Job", merged: str,
                                     merge_cfg: dict) -> bool:
        """Seed a pipeline directory whose run stage is the merged run and
        submit ONE ordinary continuation job (`bundle execute --verb
        pipeline`) for the remaining stages — the pipeline's own resume
        machinery skips the completed run stage. No remaining stages: the
        seeded pipeline directory IS the final artifact, so its evidence is
        packaged here and the parent succeeds. With remaining stages the
        evidence is packaged by the continuation over the finished pipeline
        directory and folded onto this parent when it completes
        (``_follow_continuation``) — never the merged run's partial story."""
        from ..experiment import resume as resume_mod
        from ..experiment import sharding
        from .executors import SlurmResources
        from .submissions import _bundle_execute_command
        name = str(merge_cfg.get("experiment") or "")
        target = str(merge_cfg.get("targetRoot") or "")
        pipeline_dir, remaining = sharding.seed_pipeline_directory(
            name, target, merged, job_id=parent.id, log=parent.log)
        result = dict(parent.result or {})
        result["mergedRunDirectory"] = merged
        # The pipeline directory is stamped up front (finding 4, 2026-07-23)
        # so an evidence_missing failure can name the exact path an operator
        # packages by hand.
        result["pipelineDirectory"] = pipeline_dir
        if not remaining:
            if merge_cfg.get("packageEvidence", True):
                from ..experiment import bundles as bundles_mod
                evidence = bundles_mod.package_evidence(pipeline_dir)
                result["evidenceBundle"] = evidence
                parent.log(f"evidence bundle packaged for the pipeline "
                           f"directory → {evidence.get('bundlePath')}")
            parent.status = "succeeded"
            parent.finished_at = time.time()
            result["runDirectory"] = pipeline_dir
            parent.result = result
            parent.log(f"sharded pipeline complete (run only) → {pipeline_dir}")
            self.store.update(parent)
            return True
        continuation = self._submit_pipeline_continuation(
            parent, pipeline_dir, name, merge_cfg,
            bundle_subdir="slurm-continuation",
            log_line=(f"pipeline continuation of sharded run {parent.id}: "
                      f"remaining stage(s) {', '.join(remaining)} over the "
                      "merged run"))
        parent.status = "running"
        parent.result = result
        parent.log(f"merge complete → {merged}; remaining pipeline stage(s) "
                   f"{', '.join(remaining)} submitted as {continuation.id} "
                   f"(Slurm {continuation.executor_job_id})")
        self.store.update(parent)
        return True

    def _submit_pipeline_continuation(self, parent: "Job", pipeline_dir: str,
                                      name: str, merge_cfg: dict, *,
                                      bundle_subdir: str,
                                      log_line: str) -> "Job":
        """Submit ONE ordinary `bundle execute --verb pipeline` continuation
        for ``pipeline_dir``'s remaining stages, attach it as the parent's
        ``continuationJob``, and return its record. Shared by the
        post-merge continuation and the post-judge-merge continuation."""
        from ..experiment import resume as resume_mod
        from .executors import SlurmResources
        from .submissions import _bundle_execute_command
        target = str(merge_cfg.get("targetRoot") or "")
        child_id = uuid.uuid4().hex[:12]
        rr = parent.requested_resources or {}
        records_dir = str(rr.get("recordsDirectory") or "")
        record_path = os.path.join(records_dir, f"{child_id}.json")
        resume_mod.write_pointer(
            resume_mod.pointer_path_for_record(record_path), pipeline_dir,
            verb="pipeline", experiment=name)
        command = _bundle_execute_command(
            str(merge_cfg.get("bundlePath") or ""), verb="pipeline",
            target_root=target, dtype=str(merge_cfg.get("dtype") or "auto"),
            device=merge_cfg.get("device"),
            prompts_path=merge_cfg.get("promptsPath"), source_path=None,
            package_evidence=bool(merge_cfg.get("packageEvidence", True)),
            record_path=record_path)
        resource_fields = {f.name for f in
                           SlurmResources.__dataclass_fields__.values()}
        resources = SlurmResources(**{k: v for k, v in rr.items()
                                      if k in resource_fields})
        bundle = self._slurm().create_bundle(
            os.path.join(str(merge_cfg.get("submissionDirectory")
                             or os.path.dirname(records_dir)),
                         bundle_subdir),
            command, env={"STEERLAB_JOB_ID": child_id}, resources=resources,
            metadata={"kind": "studyBundleSubmission", "experiment": name,
                      "verb": "pipeline", "parentJob": parent.id,
                      "continuationOf": "shardedRun",
                      "recordsDirectory": records_dir})
        slurm_id = self._slurm().submit(bundle)
        stamped = dict(rr)
        stamped.pop("shardChildren", None)
        stamped.pop("shardMerge", None)
        stamped.pop("parallelJobs", None)
        stamped.pop("judgeFanout", None)
        stamped.update({"scriptPath": bundle.script_path,
                        "recordsDirectory": records_dir,
                        "parentJob": parent.id})
        continuation = self.record_external(
            "study-submit-bundle-continuation", status="submitted",
            executor="slurm", executor_job_id=str(slurm_id),
            job_id=child_id, requested_resources=stamped,
            result={"recordsDirectory": records_dir, "parentJob": parent.id},
            log=f"{log_line}, as Slurm job {slurm_id}")
        parent.requested_resources = {**(parent.requested_resources or {}),
                                      "continuationJob": continuation.id}
        return continuation

    # --- auto-resubmit on checkpoint (WS2) -----------------------------------

    def _maybe_auto_resubmit(self, job: Job) -> bool:
        """Resubmit a checkpointed Slurm job's OWN sbatch script so the resume
        pointer continues the run — at most once per checkpoint, bounded by the
        chain cap, never for cancelled jobs, and deferred (not dropped) across
        maintenance windows. Returns True when it submitted or repaired.

        Crash-safety ordering: sbatch FIRST, stamp second. Stamping first and
        crashing before sbatch would mark the run continued when nothing runs;
        the reverse crash (submitted, not yet stamped) is repaired here on a
        later tick by finding the live child whose resubmitOf points at us."""
        if job.status != "checkpointed":
            return False
        if job.cancelled:
            # Cancelled beats checkpointed: a user-cancelled job is never
            # auto-resubmitted, even if a late child-record fold resurrected a
            # "checkpointed" status around the cancel stamp.
            self._resubmit_note(
                job, "cancelled",
                "not auto-resubmitting: cancellation was requested "
                "(cancelled beats checkpointed)")
            return False
        result = job.result or {}
        if result.get("resubmittedAs"):
            return False
        existing = self._existing_resubmission(job)
        if existing is not None:
            # Submit-then-stamp crash repair: the child exists in the store but
            # the marker is missing (crash/restart between sbatch and stamp, or
            # a fold replaced the result). Restore the stamp; NEVER submit a
            # second time.
            job.result = {**{k: v for k, v in result.items()
                             if k != "resubmitClaim"},
                          "resubmittedAs": existing.id}
            job.log(f"auto-resubmit: found existing resubmission {existing.id} "
                    "— repaired the resubmittedAs stamp (no new submission)")
            self.store.update(job)
            self._resubmit_notes.pop(job.id, None)
            return True
        from .executors import _env_truthy, first_crossing_window
        rr = job.requested_resources or {}
        enabled = rr.get("auto_resubmit", rr.get("autoResubmit"))
        if enabled is None:
            # No stamp on the record (legacy/external jobs): the env decides,
            # read at reconcile time.
            enabled = _env_truthy(os.environ.get("STEERLAB_AUTO_RESUBMIT"))
        if not enabled:
            return False
        limit = self._resubmit_limit(rr)
        count = int(rr.get("resubmitCount") or 0)
        if count >= limit:
            self._resubmit_note(
                job, "limit",
                f"auto-resubmit limit reached ({limit}) — resubmit manually")
            return False
        script = self._resubmit_script_candidate(job)
        if not script or not os.path.isfile(script):
            self._resubmit_note(
                job, "script",
                "auto-resubmit unavailable: the submission's sbatch script is "
                f"not on record or not on disk ({script!r}) — resubmit manually")
            return False
        walltime = str(rr.get("walltime") or "04:00:00")
        executor = self._slurm()
        calendar = getattr(getattr(executor, "profile", None),
                           "maintenance_calendar_path", None)
        window = first_crossing_window(walltime, calendar)
        if window is not None:
            # Deferred, not dropped: the job stays checkpointed and this gate
            # re-runs every reconcile tick until the window clears.
            self._resubmit_note(
                job, "window",
                f"auto-resubmit deferred: walltime {walltime} crosses the "
                f"maintenance window {window['start']} – {window['end']} — "
                "will retry on the next reconcile tick after it clears")
            return False
        # Atomic per-job claim (finding 1, 2026-07-22): the manual Resume
        # verb and this tick both land in _perform_resubmit — the durable
        # claim guarantees only one of them ever reaches sbatch. The claim
        # carries a scheduler-findable submission token BEFORE any sbatch
        # (resume crash-safety, 2026-07-23).
        claimant = f"auto:{self._claimant}"
        token = self._new_resubmit_token(job)
        claimed, result_now = self.store.claim_resubmit(job.id, claimant,
                                                        token=token)
        if not claimed:
            stamped = result_now.get("resubmittedAs")
            if stamped:
                # Someone else (a manual Resume, or a sibling process over
                # the same store) already resumed it: sync the in-memory
                # record — the honest outcome, not an error.
                job.result = {**(job.result or {}), **result_now}
                self._resubmit_notes.pop(job.id, None)
                return False
            if DurableJobStore.claim_is_stale(result_now):
                # The claimant died mid-resubmit — possibly AFTER sbatch.
                # Resolve by scheduler search (adopt / reclaim /
                # reconciliationRequired); never blind takeover.
                outcome, _payload = self._resolve_stale_claim(
                    job, result_now, limit=limit, manual=False)
                return outcome in ("adopted", "resubmitted")
            self._resubmit_note(
                job, "claim",
                "auto-resubmit skipped: another resubmission of this job "
                "is in flight (claimed) — will re-check next tick")
            return False
        job.result = {**(job.result or {}), **result_now}
        try:
            self._perform_resubmit(job, limit=limit, token=token)
        except Exception as exc:  # noqa: BLE001 - surfaced on the job, retried next tick
            released = self.store.release_resubmit_claim(job.id, claimant)
            if released is not None:
                job.result = released
            self._resubmit_note(
                job, "submit",
                f"auto-resubmit failed: {type(exc).__name__}: {exc} — will "
                "retry on the next reconcile tick")
            return False
        return True

    def _new_resubmit_token(self, job: "Job") -> str:
        """A fresh durable submission token: unique per resubmission attempt,
        prefixed so an operator can recognize it in ``squeue`` output, and
        carrying the job id so a found submission names its original."""
        return f"slre-{job.id}-{uuid.uuid4().hex[:8]}"

    def _resolve_stale_claim(self, job: "Job", result_now: dict, *,
                             limit: int, manual: bool) -> tuple[str, dict]:
        """Resolve a STALE resubmission claim (resume crash-safety,
        2026-07-23): the claimant died between claiming and stamping, and
        the crash window includes "sbatch succeeded, continuation record
        unwritten" — so resubmitting without looking would double-run the
        checkpointed run's directory. Returns ``(outcome, payload)``:

        - ``("adopted", {…})`` — the scheduler KNOWS the claim's token: the
          continuation is real; its missing job record is written now and
          ``resubmittedAs`` repaired. Nothing resubmitted.
        - ``("resubmitted", {…})`` — the scheduler positively does NOT know
          the token: the claimant died before sbatch; the claim is taken
          over and exactly one new submission happens.
        - ``("blocked", {"message": …})`` — the scheduler could not be
          searched (unreachable, or a legacy claim with no token on the
          auto path): the claim is durably stamped ``reconciliationRequired``
          with a plain-language operator message and NOTHING is resubmitted
          — at-most-once beats liveness. A later tick (or Resume click)
          retries the resolution; it heals on its own once the scheduler
          answers.

        On the MANUAL path a token-less claim may still be reclaimed: a
        human clicking Resume after checking the queue is explicit consent,
        logged loudly (there is no token to search for — pre-token claims
        only)."""
        claim = (result_now or {}).get("resubmitClaim") or {}
        stale_token = claim.get("token")
        # Cheap repair first: the continuation record may exist in the
        # DURABLE store (written by the dead claimant, or another process).
        existing = self._existing_resubmission(job)
        if existing is not None:
            job.result = {**{k: v for k, v in (job.result or {}).items()
                             if k != "resubmitClaim"},
                          "resubmittedAs": existing.id}
            job.log(f"stale resubmission claim resolved: found existing "
                    f"resubmission {existing.id} — repaired the "
                    "resubmittedAs stamp (no new submission)")
            self.store.update(job)
            self._resubmit_notes.pop(job.id, None)
            return "adopted", {"jobId": existing.id}
        finder = getattr(self._slurm(), "find_job_by_name", None)
        if stale_token and finder is not None:
            try:
                found = finder(stale_token)
            except Exception as exc:  # noqa: BLE001 - silence is not absence
                message = (
                    "resume reconciliation required: a resubmission claim "
                    f"for job {job.id} went stale (claimant "
                    f"{claim.get('claimant')!r} died mid-resubmit) and the "
                    "scheduler could not be searched for its submission "
                    f"token '{stale_token}' ({type(exc).__name__}: {exc}). "
                    "Nothing was resubmitted automatically — a blind "
                    "resubmit could double-run the checkpointed run. "
                    f"Operator: run `squeue --name {stale_token}` or `sacct "
                    f"--name {stale_token}` on the cluster; the reconciler "
                    "retries this search every tick and resolves it on its "
                    "own once the scheduler answers")
                updated = self.store.stamp_claim_reconciliation_required(
                    job.id, message)
                if updated is not None:
                    job.result = updated
                self._resubmit_note(job, "reconciliationRequired", message)
                return "blocked", {"message": message}
            if found:
                child = self._adopt_found_resubmission(job, claim, str(found),
                                                       limit=limit)
                return "adopted", {"jobId": child.id,
                                   "slurmJobID": child.executor_job_id}
        elif not (manual and stale_token is None):
            # Auto path with no token (legacy claim) or no scheduler search
            # available: absence cannot be proven — reconciliation required.
            missing = ("carries no scheduler-findable submission token "
                       "(written before tokens existed)"
                       if stale_token is None else
                       "cannot be searched on this executor")
            message = (
                "resume reconciliation required: a resubmission claim for "
                f"job {job.id} went stale (claimant "
                f"{claim.get('claimant')!r} died mid-resubmit) and it "
                f"{missing}. Nothing was resubmitted automatically — a "
                "blind resubmit could double-run the checkpointed run. "
                "Operator: check the cluster queue for a continuation of "
                "this submission by hand; if none is running, press Resume "
                "in the app — a human's click is explicit consent to "
                "reclaim")
            updated = self.store.stamp_claim_reconciliation_required(
                job.id, message)
            if updated is not None:
                job.result = updated
            self._resubmit_note(job, "reconciliationRequired", message)
            return "blocked", {"message": message}
        # Positive absence (or a human's explicit consent on a token-less
        # manual Resume): the dead claimant never reached sbatch — reclaim.
        new_claimant = (f"manual:{self._claimant}:{uuid.uuid4().hex[:6]}"
                        if manual else f"auto:{self._claimant}")
        new_token = self._new_resubmit_token(job)
        taken, result_after = self.store.takeover_stale_claim(
            job.id, new_claimant, token=new_token)
        if not taken:
            # Someone else resolved it between our search and the CAS.
            job.result = {**(job.result or {}), **result_after}
            return "blocked", {"message": "the stale claim was resolved by "
                               "another actor — re-check the job"}
        if manual and stale_token is None:
            job.log("stale token-less resubmission claim reclaimed by a "
                    "manual Resume — a human's click is explicit consent "
                    "(no scheduler token existed to search for)")
        else:
            job.log(f"stale resubmission claim resolved: the scheduler does "
                    f"not know token '{stale_token}' (positive absence) — "
                    "the dead claimant never submitted; reclaiming")
        job.result = {**(job.result or {}), **result_after}
        try:
            child = self._perform_resubmit(job, limit=limit, manual=manual,
                                           token=new_token)
        except Exception as exc:  # noqa: BLE001 - surfaced; retried next tick
            released = self.store.release_resubmit_claim(job.id, new_claimant)
            if released is not None:
                job.result = released
            if manual:
                raise
            self._resubmit_note(
                job, "submit",
                f"auto-resubmit failed after stale-claim takeover: "
                f"{type(exc).__name__}: {exc} — will retry next tick")
            return "blocked", {"message": str(exc)}
        return "resubmitted", {"jobId": child.id,
                               "slurmJobID": child.executor_job_id}

    def _adopt_found_resubmission(self, job: "Job", claim: dict,
                                  slurm_id: str, *, limit: int) -> "Job":
        """The crash window's repair (sbatch succeeded, record unwritten):
        the scheduler KNOWS the stale claim's token, so the continuation is
        already running — write the job record the dead claimant never got
        to, stamp ``resubmittedAs``, and submit NOTHING."""
        rr = job.requested_resources or {}
        count = int(rr.get("resubmitCount") or 0)
        next_count = count + 1
        chain = [str(x) for x in (rr.get("resubmitChain") or [])] + [job.id]
        child_resources = dict(rr)
        child_resources.update({"resubmitOf": job.id, "resubmitChain": chain,
                                "resubmitCount": next_count,
                                "submissionToken": claim.get("token"),
                                "adoptedFromToken": True})
        records_dir = (rr.get("recordsDirectory")
                       or (job.result or {}).get("recordsDirectory"))
        child = self.record_external(
            job.kind, status="submitted", executor="slurm",
            executor_job_id=str(slurm_id),
            requested_resources=child_resources,
            result=({"recordsDirectory": records_dir} if records_dir else None),
            log=(f"adopted a live resubmission of {job.id} found by its "
                 f"submission token {claim.get('token')!r} (Slurm job "
                 f"{slurm_id}): the original claimant died before writing "
                 "the continuation record — record repaired, NOTHING "
                 "resubmitted"))
        job.result = {**{k: v for k, v in (job.result or {}).items()
                         if k != "resubmitClaim"},
                      "resubmittedAs": child.id}
        job.log(f"checkpointed — resubmission adopted as {child.id} "
                f"({next_count}/{limit}), already running as Slurm job "
                f"{slurm_id} (crash-window repair, no second submission)")
        self.store.update(job)
        self._resubmit_notes.pop(job.id, None)
        return child

    @staticmethod
    def _resubmit_script_candidate(job: "Job") -> str | None:
        """The submission's own ``run.sbatch`` path, from the durable
        requestedResources stamp (survives child-record folds) or, failing
        that, the original result's bundle blocks. Not existence-checked —
        callers validate and word their own refusal."""
        rr = job.requested_resources or {}
        result = job.result or {}
        return (rr.get("scriptPath")
                or (result.get("slurmBundle") or {}).get("script_path")
                or (result.get("bundle") or {}).get("script_path"))

    @staticmethod
    def _resubmit_limit(rr: dict) -> int:
        """The resolved auto-resubmit chain cap for a job record: per-request
        stamp first, env second, shipped default last."""
        from .executors import _parse_resubmit_limit
        limit = rr.get("auto_resubmit_limit", rr.get("autoResubmitLimit"))
        if limit is None:
            limit = _parse_resubmit_limit(
                os.environ.get("STEERLAB_AUTO_RESUBMIT_LIMIT"))
        return int(limit)

    def _perform_resubmit(self, job: Job, *, limit: int,
                          manual: bool = False,
                          token: str | None = None) -> Job:
        """THE SAME sbatch script, verbatim, through the executor's normal
        submit path (which re-checks the maintenance window itself): the
        script re-executes and the resume pointer continues the parked run.
        The ONE resubmit implementation — the reconciler's auto-resubmit and
        the manual Resume verb both land here, differing only in their gates
        and the ``manualResubmit`` stamp on the continuation record.

        Crash-safety ordering: sbatch FIRST, stamp second (the reverse crash
        is repaired by ``_existing_resubmission`` on a later tick — and,
        cross-process, by the scheduler search for ``token``, which the
        durable claim already carries and which sbatch runs under as the
        job name). Raises on sbatch failure; callers own turning that into
        a reconcile note (auto) or an HTTP error (manual)."""
        import inspect
        from .executors import JobBundle, SlurmResources
        rr = job.requested_resources or {}
        script = self._resubmit_script_candidate(job)
        walltime = str(rr.get("walltime") or "04:00:00")
        bundle_dir = os.path.dirname(script)
        bundle = JobBundle(
            bundle_dir=bundle_dir, command=[], env={},
            resources=SlurmResources(job_name=str(rr.get("job_name") or "steerlab"),
                                     walltime=walltime),
            stdout_path=os.path.join(bundle_dir, "slurm-%j.out"),
            stderr_path=os.path.join(bundle_dir, "slurm-%j.err"),
            script_path=script,
            manifest_path=os.path.join(bundle_dir, "bundle.json"))
        executor = self._slurm()
        supports_name = False
        if token:
            try:
                supports_name = ("job_name"
                                 in inspect.signature(executor.submit).parameters)
            except (TypeError, ValueError):
                supports_name = False
        slurm_id = (executor.submit(bundle, job_name=token)
                    if supports_name else executor.submit(bundle))
        count = int(rr.get("resubmitCount") or 0)
        next_count = count + 1
        chain = [str(x) for x in (rr.get("resubmitChain") or [])] + [job.id]
        child_resources = dict(rr)
        child_resources.update({"resubmitOf": job.id, "resubmitChain": chain,
                                "resubmitCount": next_count})
        if token:
            # The durable token the claim carried and sbatch ran under —
            # provenance for the adopt-on-crash search.
            child_resources["submissionToken"] = token
        if manual:
            # A human clicking Resume is explicit consent — recorded on the
            # continuation's own resubmit entry, distinct from reconciler acts.
            child_resources["manualResubmit"] = True
        records_dir = (rr.get("recordsDirectory")
                       or (job.result or {}).get("recordsDirectory"))
        how = "manual resubmission" if manual else "auto-resubmission"
        child = self.record_external(
            job.kind, status="submitted", executor="slurm",
            executor_job_id=str(slurm_id),
            requested_resources=child_resources,
            result=({"recordsDirectory": records_dir} if records_dir else None),
            log=(f"{how} of {job.id} ({next_count}/{limit}): the same "
                 f"sbatch script re-executes as Slurm job {slurm_id} and resumes "
                 "the checkpointed run via its pointer"))
        # The stamp CONSUMES the resubmission claim: resubmittedAs is now the
        # durable "this checkpoint was continued exactly once" marker.
        job.result = {**{k: v for k, v in (job.result or {}).items()
                         if k != "resubmitClaim"},
                      "resubmittedAs": child.id}
        if manual:
            job.log(f"checkpointed — manually resubmitted as {child.id} "
                    f"({next_count}/{limit}) — continuing as Slurm job {slurm_id}")
        else:
            job.log(f"checkpointed — auto-resubmitted as {child.id} ({next_count}/{limit})")
        self.store.update(job)
        self._resubmit_notes.pop(job.id, None)
        return child

    def resubmit(self, job_id: str) -> dict:
        """Manual resume of a CHECKPOINTED job — the app's Resume button
        (``POST /api/jobs/{id}/resubmit``). Live incident 2026-07-22: a
        frozen pipeline checkpointed cleanly at the walltime margin (exit
        85), the app relayed "requeue or resubmit to continue it" — and
        offered neither. This verb is the resubmit half.

        Reuses ``_perform_resubmit`` (never a forked implementation): the
        job's OWN ``run.sbatch`` re-executes and the resume pointer
        continues the parked run. Differences from the reconciler's auto
        path are exactly the human's click: the auto-resubmit toggle is
        ignored, the chain cap may be EXCEEDED (explicit consent, logged),
        and the continuation record is stamped ``manualResubmit``. Every
        non-resumable state refuses with a plain-language
        ``ResubmitRefused``; sbatch failures raise through untouched."""
        job = self.get(job_id)
        if job is None:
            raise ResubmitRefused(
                f"no job {job_id!r} on this server — check the id")
        # Sharded parent: Resume fans out to every checkpointed,
        # not-yet-resubmitted shard child — and judge fan-out worker
        # (2026-07-23) — each through the one shared resubmit
        # implementation. The parent itself has no sbatch script.
        shard_children = (job.requested_resources or {}).get("shardChildren")
        if shard_children:
            if job.status in TERMINAL:
                raise ResubmitRefused(
                    f"sharded run {job_id} already finished ({job.status}) — "
                    "nothing to resume")
            resumed: list[dict] = []
            refusals: list[str] = []
            resumable = [str(c) for c in shard_children]
            for worker in ((job.requested_resources or {})
                           .get("judgeFanout") or {}).get("workers") or []:
                if isinstance(worker, dict) and worker.get("jobId"):
                    resumable.append(str(worker["jobId"]))
            for cid in resumable:
                child = self.get(cid)
                if child is None or child.status != "checkpointed":
                    continue
                try:
                    resumed.append(self.resubmit(cid))
                except ResubmitRefused as exc:
                    refusals.append(f"{cid}: {exc}")
            if not resumed:
                raise ResubmitRefused(
                    "no shard of this run is resumable right now — "
                    + ("; ".join(refusals) if refusals
                       else "no shard is checkpointed (they are running, "
                            "finished, or already resubmitted)"))
            job.log(f"resume fanned out to {len(resumed)} checkpointed "
                    "shard(s)"
                    + (f"; refused: {'; '.join(refusals)}" if refusals else ""))
            self.store.update(job)
            return {"ok": True, "resubmitOf": job.id,
                    "resumedShards": resumed, "manualResubmit": True}
        if job.status in TERMINAL:
            raise ResubmitRefused(
                f"job {job_id} already finished ({job.status}) — a terminal "
                "job has nothing to resume; submit the study again for a "
                "fresh run")
        if job.status != "checkpointed":
            raise ResubmitRefused(
                f"job {job_id} is {job.status} — resume applies only to a "
                "checkpointed job; this one is still with the scheduler "
                "(wait for it to finish or checkpoint, or cancel it)")
        if job.cancelled:
            raise ResubmitRefused(
                f"job {job_id} has a cancellation on record — cancelled "
                "beats checkpointed, so it will not be resumed; submit the "
                "study again if you want it to run")
        stamped = (job.result or {}).get("resubmittedAs")
        if not stamped:
            prior = self._existing_resubmission(job)
            if prior is not None:
                # Same submit-then-stamp repair the reconciler performs:
                # restore the marker, NEVER submit a second time.
                job.result = {**{k: v for k, v in (job.result or {}).items()
                                 if k != "resubmitClaim"},
                              "resubmittedAs": prior.id}
                job.log(f"resume request: found existing resubmission "
                        f"{prior.id} — repaired the resubmittedAs stamp "
                        "(no new submission)")
                self.store.update(job)
                stamped = prior.id
        if stamped:
            raise ResubmitRefused(
                f"job {job_id} was already resubmitted as {stamped} — that "
                "continuation is carrying the run; follow it instead")
        script = self._resubmit_script_candidate(job)
        if not script or not os.path.isfile(script):
            raise ResubmitRefused(
                "the submission's sbatch script is not on record or not on "
                f"disk ({script!r}) — this job cannot be resumed from here; "
                "submit the study again")
        from .executors import first_crossing_window
        walltime = str((job.requested_resources or {}).get("walltime")
                       or "04:00:00")
        calendar = getattr(getattr(self._slurm(), "profile", None),
                           "maintenance_calendar_path", None)
        window = first_crossing_window(walltime, calendar)
        if window is not None:
            raise ResubmitRefused(
                f"resume refused for now: walltime {walltime} crosses the "
                f"maintenance window {window['start']} – {window['end']} — "
                "try again after it clears (auto-resume, when enabled, "
                "retries on its own)")
        limit = self._resubmit_limit(job.requested_resources or {})
        count = int((job.requested_resources or {}).get("resubmitCount") or 0)
        if count >= limit:
            job.log(f"manual resubmit past the auto-resubmit limit "
                    f"({count} of {limit} used) — allowed: a human clicking "
                    "Resume is explicit consent")
        # Atomic per-job claim (finding 1, 2026-07-22): a double click, or a
        # click overlapping the auto-resubmit tick, must produce exactly ONE
        # sbatch. The loser of the race waits for the winner's stamp and
        # returns the honest "already resumed as <id>" outcome, not an error.
        claimant = f"manual:{self._claimant}:{uuid.uuid4().hex[:6]}"
        token = self._new_resubmit_token(job)
        claimed, result_now = self.store.claim_resubmit(job.id, claimant,
                                                        token=token)
        if not claimed:
            if DurableJobStore.claim_is_stale(result_now):
                # A dead claimant's stale claim: resolve it (scheduler
                # search → adopt / reclaim / refuse), never blind takeover
                # and never a spurious wait on a claimant that no longer
                # exists (resume crash-safety, 2026-07-23).
                outcome, payload = self._resolve_stale_claim(
                    job, result_now, limit=limit, manual=True)
                if outcome == "blocked":
                    raise ResubmitRefused(str(payload.get("message")))
                child = self.get(str(payload.get("jobId") or ""))
                return {
                    "ok": True,
                    "jobId": payload.get("jobId"),
                    "resubmitOf": job.id,
                    "slurmJobID": (payload.get("slurmJobID")
                                   or (child.executor_job_id if child else None)),
                    "resubmitCount": int(
                        ((child.requested_resources if child else None) or {})
                        .get("resubmitCount") or 0),
                    "autoResubmitLimit": limit,
                    "manualResubmit": True,
                    **({"alreadyResumed": True} if outcome == "adopted" else {}),
                }
            return self._await_concurrent_resubmission(job, result_now)
        job.result = {**(job.result or {}), **result_now}
        try:
            child = self._perform_resubmit(job, limit=limit, manual=True,
                                           token=token)
        except Exception:
            released = self.store.release_resubmit_claim(job.id, claimant)
            if released is not None:
                job.result = released
            raise
        return {
            "ok": True,
            "jobId": child.id,
            "resubmitOf": job.id,
            "slurmJobID": child.executor_job_id,
            "resubmitCount": int(
                (child.requested_resources or {}).get("resubmitCount") or 0),
            "autoResubmitLimit": limit,
            "manualResubmit": True,
        }

    def _await_concurrent_resubmission(self, job: Job, result_now: dict,
                                       timeout: float = 10.0) -> dict:
        """The race loser's coherent answer: another actor holds (or just
        consumed) the resubmission claim for ``job``. Wait briefly for the
        winner's ``resubmittedAs`` stamp — sbatch is seconds, the claim is
        already ours to lose — and report the continuation it launched. Only
        if the winner is still mid-flight after the wait does this refuse,
        and then with the situation, not a spurious failure.

        The wait re-reads the DURABLE store every pass (resume crash-safety,
        2026-07-23): the winner may be a sibling PROCESS over the same
        SQLite store, whose stamp and continuation record never appear in
        this process's in-memory catalog."""
        deadline = time.time() + timeout
        while True:
            durable = self.store.read_result(job.id)
            stamped = (durable.get("resubmittedAs")
                       or (job.result or {}).get("resubmittedAs")
                       or result_now.get("resubmittedAs"))
            if stamped:
                # Sync the in-memory record with the durable truth — WITHOUT
                # resurrecting the claim the stamp consumed. The hazard (CI
                # caught it on a slow runner, 2026-08-20): this durable read
                # can predate the same-process winner's stamp WRITE while the
                # shared in-memory job already carries the stamp, so `durable`
                # still holds the claim and a plain merge produces
                # {resubmittedAs, resubmitClaim} — a poisoned record that any
                # later store.update() would persist. Once a resubmittedAs
                # stamp exists ANYWHERE (durable, in-memory, or the CAS
                # answer), the claim is consumed by definition.
                merged = {**(job.result or {}), **durable}
                merged.pop("resubmitClaim", None)
                job.result = merged
            if not stamped:
                existing = self._existing_resubmission(job)
                if existing is not None:
                    stamped = existing.id
            if stamped:
                child = self.get(str(stamped))
                job.log(f"resume request raced a concurrent resubmission — "
                        f"already resumed as {stamped} (no second submission)")
                return {
                    "ok": True,
                    "jobId": str(stamped),
                    "resubmitOf": job.id,
                    "slurmJobID": child.executor_job_id if child else None,
                    "resubmitCount": int(
                        ((child.requested_resources if child else None) or {})
                        .get("resubmitCount") or 0),
                    "autoResubmitLimit": self._resubmit_limit(
                        job.requested_resources or {}),
                    "manualResubmit": True,
                    "alreadyResumed": True,
                }
            if time.time() >= deadline:
                raise ResubmitRefused(
                    f"job {job.id} is being resumed by another request right "
                    "now — nothing was double-submitted; check the job list "
                    "in a moment for the continuation")
            time.sleep(0.05)

    def _existing_resubmission(self, job: Job) -> "Job | None":
        """The newest job born as a resubmission of ``job``, or None.
        Existence alone counts — live or finished — because a resubmit
        happened exactly once regardless of how its continuation fared.

        Reads the DURABLE store, not the in-memory catalog (resume
        crash-safety, 2026-07-23): a sibling process over the same SQLite
        file may have written the continuation, and missing it here was
        exactly the double-submission window. Rows found durably are folded
        into this process's catalog (the already-live in-memory instance
        wins — it carries worker/cancel state)."""
        candidates: list[Job] = []
        for fresh in self.store.find_resubmissions_of(job.id):
            with self._lock:
                current = self._jobs.get(fresh.id)
                if current is None:
                    self._jobs[fresh.id] = fresh
                    current = fresh
            candidates.append(current)
        if not candidates:
            return None
        return max(candidates, key=lambda j: j.created_at)

    def _resubmit_note(self, job: Job, key: str, message: str) -> None:
        """Log an auto-resubmit situation loudly but once: repeat ticks with the
        identical situation stay silent; a CHANGED situation logs again."""
        if self._resubmit_notes.get(job.id) == (key, message):
            return
        self._resubmit_notes[job.id] = (key, message)
        job.log(message)

    def start_monitor(self, interval: float = 15.0,
                      housekeeping_interval: float = 3600.0,
                      watchdog_interval: float = 60.0,
                      stall_after: float | None = None) -> None:
        """Background reconciler: polls Slurm state so clients reconnecting by job
        id see live status and SSE streams close when the scheduler job ends.

        A second, low-frequency housekeeping tick (WS3) rides the same
        stop-event: it refreshes the expensive scans (purge risk, HF-cache
        inventory, evidence bundles) and folds throughput from terminal job
        records, persisting the snapshot so restarts serve stale-but-honest
        data. It never touches the reconcile cadence. Pass
        ``housekeeping_interval=0`` to disable it.

        **Open-issues §2.** Three changes make a stalled reconciler survivable:
        the stop event is CLEARED (it was never cleared, so a manager whose
        monitor had been stopped once could never be restarted — the new thread
        exited on its first ``wait``); every tick stamps ``_monitor_last_tick``
        so a stall is observable; and a watchdog thread re-runs the merge pass
        when the monitor has gone quiet for ``stall_after`` seconds. Pass
        ``watchdog_interval=0`` to disable the watchdog."""
        self._monitor_stop.clear()
        self._monitor_started_at = time.time()
        self._monitor_last_tick = self._monitor_started_at

        def loop():
            while not self._monitor_stop.wait(interval):
                # Stamp BEFORE the poll: the field means "the tick that is
                # running now started here", so a poll wedged inside an
                # unbounded scheduler call ages it just like a dead thread.
                self._monitor_last_tick = time.time()
                try:
                    self.poll_slurm()
                except Exception as exc:  # noqa: BLE001 - never let the monitor die
                    self._monitor_note(
                        f"poll failed: {type(exc).__name__}: {exc}")
                self._monitor_ticks += 1
        threading.Thread(target=loop, name="slurm-monitor", daemon=True).start()
        if watchdog_interval and watchdog_interval > 0:
            threading.Thread(
                target=lambda: self._watchdog_loop(
                    watchdog_interval, stall_after or max(10 * interval, 600.0)),
                name="slurm-monitor-watchdog", daemon=True).start()
        if housekeeping_interval and housekeeping_interval > 0:
            def housekeeping_loop():
                while True:
                    try:
                        from .housekeeping import refresh
                        refresh(jobs=self)
                    except Exception:  # noqa: BLE001 - never let the tick die
                        pass
                    if self._monitor_stop.wait(housekeeping_interval):
                        return
            threading.Thread(target=housekeeping_loop, name="housekeeping-tick",
                             daemon=True).start()

    def stop_monitor(self) -> None:
        self._monitor_stop.set()

    # --- monitor liveness, watchdog, and the operator's merge pass (§2) -------

    def _monitor_note(self, message: str) -> None:
        """One loud line on stderr, remembered for ``monitor_health``. The
        historical bare ``except … pass`` made a broken reconciler and an idle
        one look identical from outside the process."""
        self._monitor_last_error = message
        print(f"[jobs] monitor: {message}", file=sys.stderr, flush=True)

    def monitor_health(self) -> dict:
        """What the reconciler is doing, for the operator and for tests.

        ``stalledSeconds`` is the age of the tick currently in flight — the
        number that stayed at 0 in a healthy daemon and grew without bound in
        the two observed stalls."""
        now = time.time()
        last = self._monitor_last_tick
        return {
            "running": self._monitor_started_at is not None
                       and not self._monitor_stop.is_set(),
            "startedAt": self._monitor_started_at,
            "lastTickAt": last,
            "stalledSeconds": None if last is None else max(0.0, now - last),
            "ticks": self._monitor_ticks,
            "stalls": self._monitor_stalls,
            "lastError": self._monitor_last_error,
        }

    def _watchdog_loop(self, interval: float, stall_after: float) -> None:
        """Detect a monitor that has stopped ticking and run the merge pass on
        this thread instead.

        This is a WATCHDOG, not a cure. The root cause we can name and have
        fixed is the unbounded scheduler subprocess (``executors.py`` now
        passes a timeout everywhere); the watchdog covers the residual — any
        other way the single monitor thread can wedge — so that completed
        shards merge without a daemon cycle. It deliberately does NOT restart
        the monitor thread: a wedged thread is stuck in a syscall we cannot
        interrupt, and spawning a second poller would double-submit
        resubmissions. It runs the merge pass only, which is idempotent
        (``_merge_shard_parent`` returns early once ``runDirectory`` is
        stamped) and guarded by ``_merge_lock``."""
        while not self._monitor_stop.wait(interval):
            last = self._monitor_last_tick
            if last is None:
                continue
            stalled = time.time() - last
            if stalled < stall_after:
                continue
            self._monitor_stalls += 1
            self._monitor_note(
                f"WATCHDOG: no reconcile tick for {int(stalled)}s "
                f"(threshold {int(stall_after)}s) — the poll loop is wedged, "
                "very likely inside a scheduler command. Running the "
                "shard-merge pass here so completed shards still merge; "
                "cycle serverd to restore full reconciliation.")
            try:
                merged = self.run_merge_pass(blocking=False)
            except Exception as exc:  # noqa: BLE001 - the watchdog outlives it
                self._monitor_note(
                    f"WATCHDOG merge pass failed: {type(exc).__name__}: {exc}")
                continue
            self._monitor_note(f"WATCHDOG merge pass changed {merged} parent(s)")

    def run_merge_pass(self, *, blocking: bool = True) -> int:
        """The shard-parent reconciliation + merge pass, serialised.

        The one entry point to ``_reconcile_shard_parents`` for every caller —
        monitor tick, watchdog, and ``POST /api/jobs/reconcile`` — so two
        threads can never merge the same parent concurrently. A non-blocking
        caller that loses the race returns 0 rather than queueing behind a
        wedged holder."""
        if not self._merge_lock.acquire(blocking=blocking):
            return 0
        try:
            return self._reconcile_shard_parents()
        finally:
            self._merge_lock.release()

    def known_records_directories(self) -> list[str]:
        """Every records directory this store knows about, deduplicated and in
        a stable order. Both places a job can carry one (the live ``result``
        and the submit-time ``requestedResources`` stamp) are read, exactly as
        ``poll_slurm`` reads them."""
        seen: dict[str, None] = {}
        for job in self.list():
            for source in (job.result or {}, job.requested_resources or {}):
                value = source.get("recordsDirectory")
                if isinstance(value, str) and value.strip():
                    seen.setdefault(value, None)
        return list(seen)

    def reconcile_all(self) -> tuple[int, list[str]]:
        """Fold child records from every known records directory.

        This is what an operator means by "reconcile everything": the
        single-directory verb requires knowing which submission is stuck,
        which is precisely what a stalled daemon makes hard to see."""
        total = 0
        visited: list[str] = []
        for directory in self.known_records_directories():
            if not os.path.isdir(directory):
                continue
            visited.append(directory)
            try:
                total += self.reconcile(directory)
            except Exception as exc:  # noqa: BLE001 - one bad dir is not the set
                self._monitor_note(
                    f"reconcile of {directory} failed: "
                    f"{type(exc).__name__}: {exc}")
        return total, visited

    def reconcile(self, records_dir: str) -> int:
        """Fold child-job JSON records into the single-writer store.

        Slurm child processes should write one JSON record per job into their
        scratch directory instead of opening ``jobs.sqlite`` directly.
        """
        count = 0
        if not os.path.isdir(records_dir):
            return 0
        for name in sorted(os.listdir(records_dir)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(records_dir, name)
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            job_id = str(data.get("id") or data.get("jobID") or "")
            if not job_id:
                continue
            job = self.get(job_id) or Job(id=job_id, kind=data.get("kind", "slurm-child"),
                                          _store=self.store)
            # Reconciler bookkeeping the child cannot know must survive the
            # result replacement below: losing resubmittedAs would make the
            # auto-resubmit gate fall back to the (idempotent, but noisier)
            # repair scan on every subsequent tick; losing parentJob/shard
            # would orphan a shard child from its sharded parent's display
            # grouping (the child stamps its own "shard" when it ran with
            # --shard, but parentJob only the submitter knows).
            preserved = {key: (job.result or {}).get(key)
                         for key in ("resubmittedAs", "parentJob", "shard")}
            job.status = data.get("status", job.status)
            job.executor = data.get("executor", job.executor)
            job.executor_job_id = data.get("executorJobID", job.executor_job_id)
            job.result = data.get("result", job.result)
            # Throughput/provenance stamps from the WS2 child-record contract;
            # folded into the result so preflight estimators can read them
            # from the job store without re-opening scratch directories.
            for key in ("elapsedSeconds", "recordCount"):
                if key in data:
                    job.result = {**(job.result or {}), key: data[key]}
            for key, value in preserved.items():
                if value and not (job.result or {}).get(key):
                    job.result = {**(job.result or {}), key: value}
            job.error = data.get("error", job.error)
            job.output_artifacts = data.get("outputArtifacts", job.output_artifacts)
            job.finished_at = data.get("finishedAt", job.finished_at)
            with self._lock:
                self._jobs[job.id] = job
            self.store.insert(job)
            for line in data.get("logs", []):
                job.log(str(line))
            count += 1
        return count

    def stream(self, job_id: str):
        job = self.get(job_id)
        if job is None:
            return
        sent = 0
        while True:
            logs = job.all_logs()
            for line in logs[sent:]:
                yield line
            sent = len(logs)
            if job.status in TERMINAL:
                yield f"[{job.status}]"
                return
            time.sleep(0.2)
