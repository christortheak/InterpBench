"""Record-level resume + cooperative checkpoint for headless study runs.

The reliability contract (TURNKEY-CLUSTER-PLAN WS2): a run killed by the
scheduler — walltime warning, drain, preemption, scancel — must be
continuable to a run directory indistinguishable from an uninterrupted one.
Three mechanisms compose:

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
3. **Resume from the records alone** (2026-10-04). A run killed in the
   MIDDLE of a response never reaches mechanism 1: the response in progress
   outlasts the grace period, the process is killed, and no
   ``resume-state.json`` is written. Its run directory still holds every
   completed response, one flushed line each, and nothing in the state file
   is needed to continue: the state carries the run id (the directory's
   name), the verb (also in ``task.txt``), a record count (the number of
   lines), and a timestamp and reason nobody reads back. Sampling keeps no
   global state either: every record is generated under its own seed.
   So :func:`records_basis` says whether a directory can be continued from
   its records, :func:`require_resumable` admits one that can, and
   :func:`adopt_records` saves its place after the fact.

   One rule makes that safe: **a response is complete only when its record
   line is.** A line cut off in mid-write is dropped
   (:func:`load_completed`), and a row a cut-off response left in a side
   stream before its record line is removed
   (:func:`reconcile_side_streams`), so the response is generated again
   and nothing of the first attempt is counted.

   What the state file did imply, and the records cannot, is that the
   process that wrote them has ENDED. Nothing here can establish that, so
   the automatic path (:func:`resolve_pointer`, a scheduler requeue) still
   starts a fresh directory for a run with no saved place. A resume from
   records happens only when a caller that knows the writer has ended asks
   for it: the controller's resume of a cancelled job, after the scheduler
   confirms the job stopped, or an operator naming the directory.

Fixed cross-wave contract:

- checkpoint exit code is **85**;
- ``resume-state.json`` = ``{"runId", "verb", "completedRecords",
  "updatedAt", "reason": "signal"|"cancel"|"records"}`` — presence marks a
  resumable, incomplete run; it is deleted when the run completes normally.
  ``"records"`` (additive, 2026-10-04) means the run did not save this
  place itself: it was saved afterwards, from the record lines on disk;
- ``report.json`` marks a COMPLETE study run, and complete runs refuse
  resume (immutable-runs invariant).

This module imports only the standard library and the equally light
``run_status`` at load, so the subprocess signal tests (and any child runner)
can import it without paying the torch/transformers import tax.
"""

from __future__ import annotations

import json
import os
import signal as signal_mod
import threading
from datetime import datetime, timezone
from typing import Callable

from . import run_status

CHECKPOINT_EXIT_CODE = 85
RESUME_STATE_FILENAME = "resume-state.json"
COMPLETION_FILENAME = "report.json"
GENERATIONS_FILENAME = "generations.jsonl"
BATTERY_FILENAME = "battery.jsonl"
#: One per panel transcript: ``<run>/<condition>/turns.jsonl``, or
#: ``<run>/<condition>/replicate-<n>/turns.jsonl`` with replicates.
PANEL_TURNS_FILENAME = "turns.jsonl"

#: ``resume-state.json`` reasons. The first two are written by the run as it
#: parks. ``records`` is written for it afterwards (:func:`adopt_records`).
RECORDS_REASON = "records"
STATE_REASONS = ("signal", "cancel", RECORDS_REASON)

#: Streams a sampled response appends a row to BEFORE its record line, keyed
#: like the record. A response cut off between the two leaves a row that
#: belongs to no completed response; :func:`reconcile_side_streams` removes
#: it. The J-lens trace is the only such stream (``jlens.trace.TRACE_FILENAME``;
#: named here by value because this module imports nothing heavier than the
#: standard library). Policy decisions and probe readings are not side
#: streams: they ride inside the record line itself.
RESPONSE_SIDE_STREAMS = ("jlens-readout.jsonl",)

# generations.jsonl record kinds (the key's disambiguator: a choice-instrument
# readout and a sampled generation legally share (condition, promptID)).
KIND_SAMPLED = "sampled"
KIND_INSTRUMENT = "instrument"
KIND_ERROR = "error"


class ResumeError(RuntimeError):
    """A run directory that must not be resumed: already complete, with
    neither a saved place nor records to continue from, or checkpointed by a
    different verb/experiment."""


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
    if reason not in STATE_REASONS:
        raise ValueError(
            "resume-state reason must be 'signal', 'cancel', or 'records', "
            f"got {reason!r}")
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
    """The resume gate: returns the resume state, or raises
    :class:`ResumeError` with the reason resume is refused.

    A directory with no ``resume-state.json`` is admitted when it is a study
    run that can be continued from its records (:func:`records_basis`); the
    state returned for it is built from those records and marked
    ``"reason": "records"``. This gate is reached only with a directory a
    caller chose to hand over, so the caller answers for the run's earlier
    writer having ended. Sweeps keep the old rule: no state file, no resume.
    """
    if not os.path.isdir(run_directory):
        raise ResumeError(f"resume target is not a directory: {run_directory}")
    if is_complete(run_directory, verb):
        raise ResumeError(
            f"run directory {run_directory} is complete "
            f"({completion_file_for(verb)} present) — complete runs are "
            "immutable and never resumed")
    state = read_state(run_directory)
    if state is None:
        if verb != "run":
            raise ResumeError(
                f"run directory {run_directory} has no {RESUME_STATE_FILENAME} — "
                "only checkpointed runs are resumable")
        basis = records_basis(run_directory, verb=verb)
        if not basis["qualifies"]:
            raise ResumeError(
                f"run directory {run_directory} has no {RESUME_STATE_FILENAME} "
                "and cannot be continued from its response records: "
                f"{basis['reason']}")
        return {"runId": os.path.basename(os.path.normpath(run_directory)),
                "verb": verb,
                "completedRecords": basis["completedRecords"],
                "reason": RECORDS_REASON}
    stated_verb = state.get("verb")
    if stated_verb and stated_verb != verb:
        raise ResumeError(
            f"run directory {run_directory} was checkpointed by verb "
            f"{stated_verb!r}, not {verb!r}")
    return state


# --- generations.jsonl loading -----------------------------------------------

class _CompleteLines:
    """The complete record lines of a JSONL stream, in file order. Reads only.

    A line is complete when it ends in a newline and decodes to a JSON
    object. The writer emits a record as one string that ENDS in the newline,
    so a terminated line was written whole, and a line cut off in mid-write
    has no terminator. Iteration stops at the first line that is not
    complete; ``valid_end`` is then the byte offset just past the last
    complete line and ``torn`` says whether anything lies beyond it.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.valid_end = 0
        self.size = 0

    def __iter__(self):
        self.valid_end = 0
        try:
            self.size = os.path.getsize(self.path)
            handle = open(self.path, "rb")
        except OSError:
            self.size = 0
            return
        with handle:
            for raw in handle:
                if not raw.endswith(b"\n"):
                    break  # no terminator: a torn tail
                record = None
                if raw.strip():
                    try:
                        record = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        break  # torn/corrupt line: nothing after it counts
                    if not isinstance(record, dict):
                        break
                self.valid_end += len(raw)
                if record is not None:
                    yield record, raw

    @property
    def torn(self) -> bool:
        return self.valid_end < self.size


def load_completed(generations_path: str) -> tuple[list[dict], set[tuple]]:
    """Parse an interrupted ``generations.jsonl`` into (records, keys).

    Tolerates a torn tail (a hard kill between ``write`` and the page hitting
    disk): parsing stops at the first line that is incomplete or fails to
    decode, and the file is truncated back to the end of the last complete
    record so the append-mode writer continues from a clean boundary.
    """
    if not os.path.exists(generations_path):
        return [], set()
    scan = _CompleteLines(generations_path)
    records: list[dict] = []
    keys: set[tuple] = set()
    for record, _raw in scan:
        records.append(record)
        keys.add(record_key(record))
    if scan.torn:
        _cut(generations_path, scan.valid_end)
    return records, keys


def _cut(path: str, length: int) -> None:
    """Truncate ``path`` to ``length`` bytes, durably."""
    with open(path, "r+b") as handle:
        handle.truncate(length)
        handle.flush()
        os.fsync(handle.fileno())


def _settled_keys(path: str) -> set[tuple]:
    """The keys of a record stream's complete lines, with a torn tail cut
    off. :func:`load_completed` without holding the records: a stream can be
    large, and the caller here only needs to know which responses exist."""
    keys: set[tuple] = set()
    if not os.path.exists(path):
        return keys
    scan = _CompleteLines(path)
    for record, _raw in scan:
        keys.add(record_key(record))
    if scan.torn:
        _cut(path, scan.valid_end)
    return keys


# --- resume from the records alone ---------------------------------------------

def panel_turn_files(run_directory: str) -> list[str]:
    """Every panel transcript's ``turns.jsonl`` under a run directory, in a
    stable order. Empty for an ordinary (non-panel) run, whose records are
    the root ``generations.jsonl``."""
    found: list[str] = []
    try:
        entries = sorted(os.listdir(run_directory))
    except OSError:
        return found
    for entry in entries:
        condition = os.path.join(run_directory, entry)
        if not os.path.isdir(condition):
            continue
        direct = os.path.join(condition, PANEL_TURNS_FILENAME)
        if os.path.isfile(direct):
            found.append(direct)
        try:
            inner = sorted(os.listdir(condition))
        except OSError:
            continue
        for name in inner:
            nested = os.path.join(condition, name, PANEL_TURNS_FILENAME)
            if name.startswith("replicate-") and os.path.isfile(nested):
                found.append(nested)
    return found


def completed_turn_count(turns_path: str) -> int:
    """How many turns of one panel transcript are complete.

    The same reading the panel runner applies when it resumes a transcript
    (``multi_agent._completed_turns``): a turn is complete when its line
    decodes and names a ``turnID``. A line cut off in mid-write decodes to
    nothing, so the turn in progress when the run stopped is never counted.
    """
    seen: set = set()
    try:
        with open(turns_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get("turnID"):
                    seen.add(record["turnID"])
    except (OSError, UnicodeDecodeError):
        return 0
    return len(seen)


def _recorded_task(run_directory: str) -> str | None:
    try:
        with open(os.path.join(run_directory, "task.txt"),
                  encoding="utf-8") as handle:
            return handle.read().strip() or None
    except OSError:
        return None


def _is_failure_record(run_directory: str) -> bool:
    status = run_status.read_status(run_directory)
    if isinstance(status, dict) and status.get("status") == "failed":
        return True
    return os.path.exists(
        os.path.join(run_directory, run_status.FAILURE_NOTE_FILENAME))


def _no_basis(code: str, reason: str) -> dict:
    return {"qualifies": False, "code": code, "reason": reason,
            "completedRecords": 0}


def records_basis(run_directory: str, *, verb: str = "run") -> dict:
    """Whether a run directory can be continued from its records alone, and
    from how many. Reads only; safe to call on a directory in any state.

    Returns ``{"qualifies", "code", "reason", "completedRecords"}`` and, when
    it qualifies, ``"unit"`` (``response`` or ``turn``), ``"tornTail"`` (a
    last line cut off in mid-write, which a resume drops), and the
    ``"experimentHash"`` the directory is stamped with. ``reason`` is a plain
    clause about "its run folder", written to follow "the run was stopped
    before it could save its place, and …".

    A directory qualifies when all of these hold:

    - it is a study run (verb ``run``, and ``task.txt`` does not say
      otherwise), still on disk, and not complete;
    - it is not a failure record: a run that stopped on an ERROR was
      packaged and reported as failed, and continuing it in place would
      rewrite that record;
    - it says which version of the study wrote it (the manifest-epoch
      stamp), and no response in it was written under another version. The
      run's own admission then compares that stamp with the live manifest;
    - it holds at least one complete record: a line of ``generations.jsonl``
      or, for a panel, of a transcript's ``turns.jsonl``. One line is enough
      to know the run's creation stamps exist, because every one of them is
      written before the first response is generated.

    Whether the process that wrote the records has ENDED is not something a
    directory can say. The caller establishes it.
    """
    if verb != "run":
        return _no_basis(
            "notAStudyRun",
            f"it is a {verb}, which cannot be continued from its response "
            "records")
    if not os.path.isdir(run_directory):
        return _no_basis("missing", "its run folder is no longer on disk")
    if is_complete(run_directory, verb):
        return _no_basis("complete", "the run had already finished")
    task = _recorded_task(run_directory)
    if task is not None and task != "run":
        return _no_basis(
            "notAStudyRun",
            f"its run folder was written by a {task}, not by a study run")
    if _is_failure_record(run_directory):
        return _no_basis(
            "failedRun",
            "its run folder records a run that stopped on an error, which "
            "is kept as a failure record")
    from . import run_epoch  # numpy-weight import, so not at module load
    stamp = run_epoch.stamped_experiment_hash(run_directory)
    turn_files = panel_turn_files(run_directory)
    torn = False
    other_study = False
    if turn_files:
        unit = "turn"
        count = sum(completed_turn_count(path) for path in turn_files)
    else:
        unit = "response"
        count = 0
        scan = _CompleteLines(os.path.join(run_directory, GENERATIONS_FILENAME))
        for record, _raw in scan:
            written_under = record.get("experimentHash")
            if stamp and isinstance(written_under, str) \
                    and written_under != stamp:
                other_study = True
            count += 1
        torn = scan.torn
    if count == 0:
        return _no_basis("noRecords",
                         "its run folder holds no completed response")
    if not stamp:
        return _no_basis(
            "noStudyStamp",
            "its run folder does not say which version of the study wrote "
            "it")
    if other_study:
        return _no_basis(
            "studyHashMismatch",
            "a response in its run folder was written under a different "
            "version of the study than the folder is stamped with")
    return {"qualifies": True, "code": None, "reason": None,
            "completedRecords": count, "unit": unit, "tornTail": torn,
            "experimentHash": stamp}


def reconcile_side_streams(run_directory: str, completed_keys: set, *,
                           log: Callable[[str], None] | None = None
                           ) -> dict[str, int]:
    """Remove from every per-response side stream each row that belongs to a
    response with no record line. Returns ``{filename: rows removed}`` for
    the streams that lost any.

    ``completed_keys`` are the keys of the complete lines of
    ``generations.jsonl``. A side row is written BEFORE its response's
    record line, so a run stopped between the two leaves a row for a
    response that never completed. Left in place it would be read as that
    response's evidence, and the writer's own idempotence would keep it and
    drop the row the regenerated response produces. Removed, the response is
    generated again and writes both afresh.

    A last row cut off in mid-write goes with it. Kept rows keep their bytes
    and their order, and a stream with nothing to remove is not touched.
    The rows are streamed, never held: a trace can run to gigabytes. In the
    shape a kill actually leaves (the cut-off response's row is the last
    one) the file is simply cut at that row; only rows with kept rows after
    them need the file rewritten, which is done beside it and swapped in
    atomically.
    """
    removed: dict[str, int] = {}
    for filename in RESPONSE_SIDE_STREAMS:
        path = os.path.join(run_directory, filename)
        if not os.path.isfile(path):
            continue
        scan = _CompleteLines(path)
        orphans = 0
        first_orphan_at: int | None = None
        kept_after_an_orphan = False
        for record, raw in scan:
            if record_key(record) in completed_keys:
                kept_after_an_orphan = (kept_after_an_orphan
                                        or first_orphan_at is not None)
            else:
                orphans += 1
                if first_orphan_at is None:
                    first_orphan_at = scan.valid_end - len(raw)
        if not orphans:
            if scan.torn:
                _cut(path, scan.valid_end)
            continue
        if not kept_after_an_orphan:
            _cut(path, first_orphan_at)
        else:
            tmp = path + ".tmp"
            with open(tmp, "wb") as handle:
                for record, raw in _CompleteLines(path):
                    if record_key(record) in completed_keys:
                        handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        removed[filename] = orphans
        if log is not None:
            log(f"resume: removed {orphans} row(s) from {filename} that a "
                "response left before it was cut off; a response is complete "
                "only when its record line is, so that response is generated "
                "again")
    return removed


def adopt_records(run_directory: str, *, verb: str = "run",
                  log: Callable[[str], None] | None = None) -> dict:
    """Save the place of a run that was stopped before it could save its own.

    For a caller that KNOWS the run's writer has ended (the controller, once
    the scheduler confirms a cancelled job stopped). Leaves the directory as
    a clean park would have: a last line cut off in mid-write is dropped,
    side-stream rows of the cut-off response are removed, and
    ``resume-state.json`` is written with ``"reason": "records"``. From then
    on every existing path that continues a parked run continues this one:
    the re-executed script's pointer, and a pipeline's run stage.

    Returns the :func:`records_basis` it adopted. Raises :class:`ResumeError`
    when the directory does not qualify. A directory the run parked itself
    is left exactly as it is; a place saved here earlier is refreshed, since
    a continuation may have added records and been stopped the same way.
    """
    if is_complete(run_directory, verb):
        raise ResumeError(
            f"run directory {run_directory} is complete — complete runs are "
            "immutable and never resumed")
    state = read_state(run_directory)
    if state is not None and state.get("reason") != RECORDS_REASON:
        return {"qualifies": True, "code": None, "reason": None,
                "completedRecords": state.get("completedRecords"),
                "parkedByRun": True}
    basis = records_basis(run_directory, verb=verb)
    if not basis["qualifies"]:
        raise ResumeError(
            f"run directory {run_directory} cannot be continued from its "
            f"response records: {basis['reason']}")
    if basis["unit"] == "response":
        # A panel's transcripts are repaired by the panel runner itself as it
        # resumes each one; an ordinary run's streams are settled here.
        keys = _settled_keys(os.path.join(run_directory, GENERATIONS_FILENAME))
        _settled_keys(os.path.join(run_directory, BATTERY_FILENAME))
        basis = {**basis, "removedSideRows": reconcile_side_streams(
            run_directory, keys, log=log)}
    write_state(run_directory,
                run_id=os.path.basename(os.path.normpath(run_directory)),
                verb=verb, completed_records=basis["completedRecords"],
                reason=RECORDS_REASON)
    return basis


# --- the writer ----------------------------------------------------------------

def _size_or_zero(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


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
        #: Side-stream rows removed on resume because their response had no
        #: record line (``{filename: count}``); empty for a fresh run.
        self.removed_side_rows: dict[str, int] = {}
        if resume:
            size_before = _size_or_zero(self.path)
            self.records, self.completed_keys = load_completed(self.path)
            if log is not None and _size_or_zero(self.path) < size_before:
                log(f"resume: dropped a last line of {filename} that was cut "
                    "off in mid-write; the response it belonged to is "
                    "generated again")
            if filename == GENERATIONS_FILENAME:
                # THE RUN'S RECORD STREAM decides which responses are
                # complete, so this is where everything else is held to it:
                # whatever a cut-off response left in a side stream goes
                # before any other writer of this run loads that stream.
                self.removed_side_rows = reconcile_side_streams(
                    run_directory, self.completed_keys, log=log)
                state = read_state(run_directory)
                if log is not None and (
                        state is None
                        or state.get("reason") == RECORDS_REASON):
                    log("resume: this run was stopped before it could save "
                        f"its place; continuing from its {len(self.records)} "
                        "completed record(s). A response that was still "
                        "being generated when the run stopped is generated "
                        "again")
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

    - ``("resume", dir)`` — incomplete with a saved place: continue it. The
      place was saved by the run as it parked, or for it afterwards by
      :func:`adopt_records`.
    - ``("complete", dir)`` — finished: re-executing (a requeue race) must be
      idempotent, not mint a second run.
    - ``("fresh", None)`` — no usable prior run (never started, hard-killed
      with no saved place, verb mismatch): start a new run directory.
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
    # Started, and no place saved (a hard kill). Its records may well be
    # complete and resumable (``records_basis``), but this is the AUTOMATIC
    # path: a scheduler requeue re-executes the script with nobody having
    # established that the earlier process is gone, and after a node failure
    # it may not be. Two writers appending to one run directory would count
    # responses twice, so the requeue starts a fresh directory. The records
    # are continued only once something that knows the writer ended saves
    # their place (``adopt_records``), which makes the directory resumable
    # above.
    return "fresh", None
