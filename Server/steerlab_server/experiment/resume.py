"""Record-level resume + cooperative checkpoint for headless study runs.

The reliability contract (TURNKEY-CLUSTER-PLAN WS2): a run killed by the
scheduler — walltime warning, drain, preemption, scancel — must be
continuable to a run directory indistinguishable from an uninterrupted one.
Two mechanisms compose:

1. **Checkpoint on signal.** The headless runner (``bundle execute`` /
   ``experiment run`` — never the FastAPI serve path) installs
   :class:`CheckpointFlag` on SIGUSR1/SIGTERM. The generation loop checks the
   flag between records; when set it flushes + fsyncs ``generations.jsonl``,
   writes ``resume-state.json``, and the process exits with
   :data:`CHECKPOINT_EXIT_CODE` (85).
2. **Skip-completed on restart.** A run started against a directory that
   carries ``resume-state.json`` (and no completion artifact) loads the
   existing records, skips every completed ``(condition, promptIndex,
   promptID, sampleIndex, kind)`` key, and appends only the missing records.
   Greedy and derived-seed generation are deterministic per record, so the
   union is byte-identical to a fresh, uninterrupted run.

Fixed cross-wave contract:

- checkpoint exit code is **85**;
- ``resume-state.json`` = ``{"runId", "verb", "completedRecords",
  "updatedAt", "reason": "signal"|"cancel"}`` — presence marks a resumable,
  incomplete run; it is deleted when the run completes normally;
- ``report.json`` marks a COMPLETE study run, and complete runs refuse
  resume (immutable-runs invariant).

This module is stdlib-only so the subprocess signal tests (and any child
runner) can import it without paying the torch/transformers import tax.
"""

from __future__ import annotations

import json
import os
import signal as signal_mod
import threading
from datetime import datetime, timezone
from typing import Callable

CHECKPOINT_EXIT_CODE = 85
RESUME_STATE_FILENAME = "resume-state.json"
COMPLETION_FILENAME = "report.json"

# generations.jsonl record kinds (the key's disambiguator: a choice-instrument
# readout and a sampled generation legally share (condition, promptID)).
KIND_SAMPLED = "sampled"
KIND_INSTRUMENT = "instrument"
KIND_ERROR = "error"


class ResumeError(RuntimeError):
    """A run directory that must not be resumed: already complete, never
    checkpointed, or checkpointed by a different verb/experiment."""


class CheckpointRequested(Exception):
    """Raised between records once a checkpoint signal has been observed.

    By the time this propagates, ``generations.jsonl`` is flushed + fsynced
    and ``resume-state.json`` is on disk — the caller's only job is to exit
    the process with :data:`CHECKPOINT_EXIT_CODE`.
    """

    def __init__(self, run_directory: str, verb: str, completed_records: int,
                 reason: str = "signal"):
        super().__init__(
            f"checkpoint ({reason}): {completed_records} records flushed → "
            f"{run_directory}")
        self.run_directory = run_directory
        self.verb = verb
        self.completed_records = completed_records
        self.reason = reason


class CheckpointFlag:
    """Thread-safe "checkpoint requested" flag settable from a signal handler.

    ``install()`` binds SIGUSR1 and SIGTERM (main thread only — exactly the
    headless CLI paths; the FastAPI serve path never installs it). The object
    itself is also usable directly as a test double: call :meth:`request`.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def request(self, signum: int | None = None, frame=None) -> None:  # noqa: ARG002 - signal handler signature
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()

    def install(self) -> "CheckpointFlag":
        signal_mod.signal(signal_mod.SIGUSR1, self.request)
        signal_mod.signal(signal_mod.SIGTERM, self.request)
        return self


# --- record identity ---------------------------------------------------------

def record_kind(record: dict) -> str:
    if "error" in record:
        return KIND_ERROR
    if "instrument" in record:
        return KIND_INSTRUMENT
    return KIND_SAMPLED


def record_key(record: dict) -> tuple:
    """Canonical identity of one generations.jsonl record.

    ``promptIndex`` rides along so duplicate user-supplied prompt ids cannot
    alias two distinct records into one key; ``kind`` separates the choice
    instrument's readout from the sampled generation of the same item.
    """
    return (record.get("condition"), record.get("promptIndex"),
            record.get("promptID"), record.get("sampleIndex"),
            record_kind(record))


def make_key(condition: str, prompt_index: int | None, prompt_id: str | None,
             sample_index: int | None, kind: str) -> tuple:
    return (condition, prompt_index, prompt_id, sample_index, kind)


# --- resume-state.json -------------------------------------------------------

def state_path(run_directory: str) -> str:
    return os.path.join(run_directory, RESUME_STATE_FILENAME)


def completion_path(run_directory: str) -> str:
    return os.path.join(run_directory, COMPLETION_FILENAME)


def completion_file_for(verb: str | None) -> str:
    """The artifact whose existence marks a run of ``verb`` finished: the
    study report for runs, ``recommendations.json`` for sweeps (which never
    write a report.json — sweep checkpoint/resume, 2026-08-03)."""
    return "recommendations.json" if verb == "sweep" else COMPLETION_FILENAME


def is_complete(run_directory: str, verb: str | None = None) -> bool:
    return os.path.isfile(
        os.path.join(run_directory, completion_file_for(verb)))


def is_resumable(run_directory: str, verb: str | None = None) -> bool:
    return (os.path.isfile(state_path(run_directory))
            and not is_complete(run_directory, verb))


def read_state(run_directory: str) -> dict | None:
    try:
        with open(state_path(run_directory), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_state(run_directory: str, *, run_id: str, verb: str,
                completed_records: int, reason: str) -> str:
    """Atomically write ``resume-state.json`` (tmp + fsync + rename), so a
    crash mid-write can never leave a torn state file marking the run."""
    if reason not in ("signal", "cancel"):
        raise ValueError(f"resume-state reason must be 'signal' or 'cancel', got {reason!r}")
    payload = {
        "runId": run_id,
        "verb": verb,
        "completedRecords": int(completed_records),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    final = state_path(run_directory)
    tmp = final + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, final)
    return final


def clear_state(run_directory: str) -> None:
    try:
        os.remove(state_path(run_directory))
    except FileNotFoundError:
        pass


def require_resumable(run_directory: str, *, verb: str) -> dict:
    """The resume gate: returns the parsed resume state, or raises
    :class:`ResumeError` with the reason resume is refused."""
    if not os.path.isdir(run_directory):
        raise ResumeError(f"resume target is not a directory: {run_directory}")
    if is_complete(run_directory, verb):
        raise ResumeError(
            f"run directory {run_directory} is complete "
            f"({completion_file_for(verb)} present) — complete runs are "
            "immutable and never resumed")
    state = read_state(run_directory)
    if state is None:
        raise ResumeError(
            f"run directory {run_directory} has no {RESUME_STATE_FILENAME} — "
            "only checkpointed runs are resumable")
    stated_verb = state.get("verb")
    if stated_verb and stated_verb != verb:
        raise ResumeError(
            f"run directory {run_directory} was checkpointed by verb "
            f"{stated_verb!r}, not {verb!r}")
    return state


# --- generations.jsonl loading -----------------------------------------------

def load_completed(generations_path: str) -> tuple[list[dict], set[tuple]]:
    """Parse an interrupted ``generations.jsonl`` into (records, keys).

    Tolerates a torn tail (a hard kill between ``write`` and the page hitting
    disk): parsing stops at the first line that is incomplete or fails to
    decode, and the file is truncated back to the end of the last complete
    record so the append-mode writer continues from a clean boundary.
    """
    if not os.path.exists(generations_path):
        return [], set()
    with open(generations_path, "rb") as handle:
        blob = handle.read()
    records: list[dict] = []
    keys: set[tuple] = set()
    valid_end = 0
    start = 0
    while True:
        newline = blob.find(b"\n", start)
        if newline == -1:
            break  # no terminator: a torn tail (or empty remainder)
        line = blob[start:newline]
        if line.strip():
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                break  # torn/corrupt line: truncate from here
            records.append(record)
            keys.add(record_key(record))
        valid_end = newline + 1
        start = newline + 1
    if valid_end < len(blob):
        with open(generations_path, "r+b") as handle:
            handle.truncate(valid_end)
    return records, keys


# --- the writer ----------------------------------------------------------------

class GenerationWriter:
    """Append-discipline ``generations.jsonl`` writer with a skip set and
    between-records checkpointing.

    Serialization matches the historical inline writer byte-for-byte
    (``json.dumps(record)`` + newline, flushed per record), and resumed runs
    append in the same deterministic loop order — so interrupted + resumed
    equals uninterrupted, byte for byte.
    """

    def __init__(self, run_directory: str, *, verb: str = "run",
                 checkpoint: CheckpointFlag | None = None,
                 resume: bool = False,
                 log: Callable[[str], None] | None = None,
                 filename: str = "generations.jsonl",
                 allowed_keys: set | None = None):
        # ``filename`` lets sidecar record streams (e.g. the capability
        # battery's battery.jsonl — gate evidence, never outcomes) reuse the
        # same append/skip/checkpoint discipline without touching
        # generations.jsonl.
        #
        # ``allowed_keys`` is the shard filter (multi-GPU fan-out): when set,
        # ``skip`` also answers True for any record key OUTSIDE the shard's
        # contiguous range, so the run loop generates exactly the shard's
        # records with zero changes to its iteration order — which is what
        # keeps the merged concatenation byte-identical to a single-job run.
        self.run_directory = run_directory
        self.verb = verb
        self.checkpoint = checkpoint
        self.log = log
        self.allowed_keys = allowed_keys
        self.path = os.path.join(run_directory, filename)
        if resume:
            self.records, self.completed_keys = load_completed(self.path)
        else:
            self.records, self.completed_keys = [], set()
        self.resumed_count = len(self.records)
        self._handle = open(self.path, "a" if resume else "w", encoding="utf-8")
        self._closed = False

    # -- loop hooks -----------------------------------------------------------

    def skip(self, condition: str, prompt_index: int | None, prompt_id: str | None,
             sample_index: int | None, kind: str = KIND_SAMPLED) -> bool:
        """True when the record for this cell already exists — or, under a
        shard filter, belongs to a different shard (pre-generation check, so
        the expensive forward pass is skipped, not just the write)."""
        key = make_key(condition, prompt_index, prompt_id, sample_index, kind)
        if self.allowed_keys is not None and key not in self.allowed_keys:
            return True
        return key in self.completed_keys

    def emit(self, record: dict) -> None:
        """Append one record (idempotent on key), flush, then honor a pending
        checkpoint request — the "between records" boundary of the contract."""
        key = record_key(record)
        if key in self.completed_keys:
            return
        self.records.append(record)
        self.completed_keys.add(key)
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()
        if self.checkpoint is not None and self.checkpoint.requested:
            self.checkpoint_now(reason="signal")

    # -- interruption ----------------------------------------------------------

    def checkpoint_now(self, reason: str = "signal") -> None:
        """Durably park the run (fsync + resume-state) and raise
        :class:`CheckpointRequested` for the process-exit path."""
        self._sync_to_disk(reason)
        if self.log is not None:
            self.log(f"checkpoint ({reason}): {len(self.records)} records "
                     f"flushed; resume-state written → exit {CHECKPOINT_EXIT_CODE}")
        raise CheckpointRequested(self.run_directory, self.verb,
                                  len(self.records), reason=reason)

    def interrupt(self, reason: str = "cancel") -> None:
        """Durably park the run without raising (the cooperative-cancel path:
        the caller finishes its partial-artifact bookkeeping and returns)."""
        self._sync_to_disk(reason)
        if self.log is not None:
            self.log(f"run interrupted ({reason}): {len(self.records)} records "
                     "kept; directory is resumable")

    def _sync_to_disk(self, reason: str) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        write_state(self.run_directory, run_id=os.path.basename(self.run_directory),
                    verb=self.verb, completed_records=len(self.records),
                    reason=reason)

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True


# --- bundle-execute resume pointer ---------------------------------------------

# The pointer ties a submission's job identity (its child-record path, stable
# across a Slurm requeue because the SAME sbatch script re-executes verbatim)
# to the run directory it started. Deliberately NOT ``*.json``: the records
# directory is scanned by the job reconciler, which folds every ``*.json``
# file as a child-job record.
POINTER_SUFFIX = ".resume"


def pointer_path_for_record(record_path: str) -> str:
    base, _ext = os.path.splitext(record_path)
    return base + POINTER_SUFFIX


def write_pointer(pointer_path: str, run_directory: str, *, verb: str,
                  experiment: str | None = None) -> None:
    os.makedirs(os.path.dirname(pointer_path), exist_ok=True)
    payload = {
        "schemaVersion": 1,
        "runDirectory": run_directory,
        "verb": verb,
        "experiment": experiment,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }
    tmp = pointer_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, pointer_path)


def read_pointer(pointer_path: str) -> dict | None:
    try:
        with open(pointer_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def resolve_pointer(pointer_path: str, *, verb: str) -> tuple[str, str | None]:
    """Classify a prior execution of the same job: ``(disposition, run_dir)``.

    - ``("resume", dir)`` — checkpointed and incomplete: continue it.
    - ``("complete", dir)`` — finished: re-executing (a requeue race) must be
      idempotent, not mint a second run.
    - ``("fresh", None)`` — no usable prior run (never started, hard-killed
      before any checkpoint, verb mismatch): start a new run directory.
    """
    data = read_pointer(pointer_path)
    if not data:
        return "fresh", None
    run_directory = data.get("runDirectory")
    if not run_directory or not os.path.isdir(run_directory):
        return "fresh", None
    if data.get("verb") not in (None, verb):
        return "fresh", None
    if is_complete(run_directory, verb):
        return "complete", run_directory
    if is_resumable(run_directory, verb):
        return "resume", run_directory
    # Started but never checkpointed (hard kill): its JSONL has no durability
    # guarantee, so honesty demands a fresh directory, not a guess.
    return "fresh", None
