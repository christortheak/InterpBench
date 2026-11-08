"""A job's status and its log tail, in fixed fields (issue #7).

``GET /api/jobs/{id}/status`` answers "is it pending, and why? running, and
how far along? ended, and how?" in one call that returns.
``GET /api/jobs/{id}/log-tail`` returns the last lines of the job's
controller log and of its own scheduler output files, and exits.

Both are reads. They never change a job record, and they reach the scheduler
only for a job SteerLab submitted: the request names a SteerLab job id, which
must exist in the job store, and the scheduler id and every file path come
from that record, never from the request. Scheduler commands go through
``SlurmExecutor.status_observation`` (site-declared binary names, argv lists,
no shell). The fields are fixed; nothing the scheduler prints is passed
through unparsed.
"""

from __future__ import annotations

import json
import os
import threading
import time

STATUS_SCHEMA_VERSION = 1
LOG_TAIL_SCHEMA_VERSION = 1

#: Controller lines a log tail returns when the caller names no count, and
#: the most it will return.
DEFAULT_TAIL_LINES = 50
MAX_TAIL_LINES = 1000

#: The most bytes read from the end of each scheduler output file. A job's
#: stderr can run to gigabytes (a warning printed per token); the tail never
#: reads more than this, whatever the line count asked for.
MAX_TAIL_BYTES = 256 * 1024


class LogTailRequestError(ValueError):
    """A log-tail request whose line count is out of range."""


def tail_line_count(raw) -> int:
    """The requested line count, validated: 1 to ``MAX_TAIL_LINES``."""
    if raw is None or raw == "":
        return DEFAULT_TAIL_LINES
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        raise LogTailRequestError(
            f"lines must be a whole number from 1 to {MAX_TAIL_LINES}, "
            f"not {raw!r}") from None
    if not 1 <= value <= MAX_TAIL_LINES:
        raise LogTailRequestError(
            f"lines must be from 1 to {MAX_TAIL_LINES}, not {value}")
    return value


# --- status ------------------------------------------------------------------

#: How long one status read may spend asking the scheduler, in all. Both
#: clients give up on a request after 60 seconds, so the read answers well
#: before that: a part whose scheduler did not answer in time says so, and
#: the rest of the document is still returned.
STATUS_READ_DEADLINE_SECONDS = 30.0

#: The most parts of a split run asked about at once.
STATUS_READ_WORKERS = 8


def job_status(manager, job, *,
               deadline_seconds: float = STATUS_READ_DEADLINE_SECONDS) -> dict:
    """The status document for ``job``: its own record, the scheduler's view
    of it, its records written so far, and, for a job that ended, the
    scheduler's sentence on how. A sharded parent adds one entry per shard
    (each shard's current continuation, when it was resumed) and progress
    summed across them. The parts are asked about at once, and the whole
    read is bounded by ``deadline_seconds``."""
    from concurrent.futures import ThreadPoolExecutor, wait

    observed_at = time.time()
    deadline = time.monotonic() + deadline_seconds
    document = {"schemaVersion": STATUS_SCHEMA_VERSION,
                "observedAt": observed_at,
                **_describe(manager, job, deadline)}
    shard_ids = [str(x) for x in
                 ((job.requested_resources or {}).get("shardChildren") or [])]
    if shard_ids:
        shards: list[dict | None] = [None] * len(shard_ids)
        carriers = {}
        for index, child_id in enumerate(shard_ids):
            child = manager.get(child_id)
            if child is None:
                shards[index] = {
                    "jobId": child_id, "shardIndex": index, "found": False,
                    "detail": ("this part's job record is not in the "
                               "server's job store")}
                continue
            carriers[index] = (child, manager._shard_chain_tail(child))
        pool = ThreadPoolExecutor(
            max_workers=max(1, min(STATUS_READ_WORKERS, len(carriers))))
        try:
            futures = {index: pool.submit(_describe, manager, carrier,
                                          deadline)
                       for index, (_child, carrier) in carriers.items()}
            wait(list(futures.values()),
                 timeout=max(0.0, deadline - time.monotonic()) + 2.0)
            for index, future in futures.items():
                child, carrier = carriers[index]
                if future.done() and future.exception() is None:
                    entry = future.result()
                else:
                    entry = _unanswered(carrier, deadline_seconds)
                entry["found"] = True
                rr = carrier.requested_resources or {}
                entry["shardIndex"] = rr.get("shardIndex", index)
                entry["shardCount"] = rr.get("shardCount", len(shard_ids))
                if carrier.id != child.id:
                    # The part was resumed: this entry is the job carrying
                    # it now, and ``shardJob`` the part's original record.
                    entry["shardJob"] = child.id
                shards[index] = entry
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        document["shards"] = shards
        held = [entry["jobId"] for entry in shards if entry.get("held")]
        if held:
            document["notes"].append(held_parts_note(len(held), len(shards)))
        late = [entry["jobId"] for entry in shards
                if (entry.get("scheduler") or {}).get("timedOut")]
        if late:
            document["notes"].append(
                f"{len(late)} of {len(shards)} part(s) did not hear from the "
                f"scheduler within {deadline_seconds:g} seconds "
                f"({', '.join(late)}); ask again in a moment")
        combined = _combined_progress(shards)
        if combined["recordsExpected"] is None:
            planned = _preflight_planned_records(job)
            if planned is not None:
                combined["recordsExpected"] = planned
                combined["recordsExpectedSource"] = "preflightEstimate"
                combined["detail"] = None
        document["progress"] = combined
    return document


def _unanswered(job, deadline_seconds: float) -> dict:
    """A part whose whole description did not finish within the read's
    time: its record, and a scheduler view that says so."""
    return {
        "jobId": job.id, "kind": job.kind, "status": job.status,
        "executor": job.executor, "executorJobID": job.executor_job_id,
        "createdAt": job.created_at, "startedAt": job.started_at,
        "finishedAt": job.finished_at,
        "cancellationRequested": job.cancelled,
        "continuedAs": (job.result or {}).get("resubmittedAs")
        if isinstance(job.result, dict) else None,
        "scheduler": {"queryOk": False, "timedOut": True,
                      "detail": ("this part was not described within "
                                 f"{deadline_seconds:g} seconds; ask again "
                                 "in a moment")},
        "schedulerOutcome": None, "schedulerOutcomeNote": None,
        "deathDetail": None, "progress": None, "notes": [],
        **_held_fields(job),
    }


def _describe(manager, job, deadline: float | None = None) -> dict:
    result = job.result if isinstance(job.result, dict) else {}
    rr = job.requested_resources or {}
    entry = {
        "jobId": job.id,
        "kind": job.kind,
        "status": job.status,
        "executor": job.executor,
        "executorJobID": job.executor_job_id,
        "createdAt": job.created_at,
        "startedAt": job.started_at,
        "finishedAt": job.finished_at,
        "cancellationRequested": job.cancelled,
        "continuedAs": result.get("resubmittedAs"),
        "scheduler": None,
        # Never beside a success: that word describes an earlier attempt.
        "schedulerOutcome": (result.get("schedulerOutcome")
                             if job.status != "succeeded" else None),
        "schedulerOutcomeNote": None,
        "deathDetail": None,
        "progress": job_progress(job),
        "notes": [],
        **_held_fields(job),
    }
    notes = entry["notes"]
    if result.get("resubmittedAs"):
        notes.append(f"this job's run continues as job "
                     f"{result['resubmittedAs']}; ask for that job's status "
                     "to follow it")
    if rr.get("shardChildren"):
        # A parent holds no scheduler job of its own; its parts do.
        notes.append("this run is split across shard jobs; each one's "
                     "scheduler view is under shards")
        return _with_outcome_note(entry, waiting=False)
    if job.executor != "slurm":
        notes.append("this job runs on the server itself, not through the "
                     "scheduler, so there is no scheduler view")
        return _with_outcome_note(entry, waiting=False)
    slurm_id = job.executor_job_id
    if not slurm_id:
        if not entry["held"]:
            notes.append("no scheduler job id is on record yet (the "
                         "submission may still be in progress, or it was a "
                         "dry run)")
        return _with_outcome_note(entry, waiting=False)
    from .executors import is_scheduler_job_id
    if not is_scheduler_job_id(str(slurm_id)):
        notes.append("the scheduler job id on record is not one the scheduler "
                     "could have issued, so the scheduler was not asked")
        return _with_outcome_note(entry, waiting=False)
    executor = manager._slurm()
    observe = getattr(executor, "status_observation", None)
    if not callable(observe):
        notes.append("this server's scheduler connection cannot report a "
                     "job's scheduler view")
        return _with_outcome_note(entry, waiting=False)
    try:
        view = observe(str(slurm_id), deadline=deadline)
    except Exception as exc:  # noqa: BLE001 - a status read never raises
        entry["scheduler"] = {"queryOk": False, "timedOut": False,
                              "detail": f"{type(exc).__name__}: {exc}"}
        return _with_outcome_note(entry, waiting=False)
    entry["scheduler"] = view.to_dict()
    if view.outcome and not entry["schedulerOutcome"] and job.status != "succeeded":
        entry["schedulerOutcome"] = view.outcome
    entry["deathDetail"] = view.death_detail
    from .executors import _SLURM_WAITING_STATES, _slurm_state_token
    token = _slurm_state_token(view.raw_state)
    if token == "SUSPENDED":
        notes.append("the scheduler has suspended this job: it is paused, "
                     "and its time limit is not counting down")
    return _with_outcome_note(entry, waiting=token in _SLURM_WAITING_STATES)


#: What each scheduler word means, in a sentence that follows "The scheduler
#: recorded <WORD>:".
_OUTCOME_MEANINGS = {
    "TIMEOUT": "the job reached its time limit",
    "OUT_OF_MEMORY": "the job used more memory than it was allocated",
    "NODE_FAIL": "a node the job ran on failed",
    "BOOT_FAIL": "the node the job was given failed to start",
    "DEADLINE": "the job reached its deadline",
}


def outcome_note(outcome: str | None, *, waiting: bool,
                 cancel_requested: bool) -> str | None:
    """One sentence on the scheduler's word for how the job stopped, which
    came from outside the job's code. Never "ended" for a job that waits to
    run again, and a cancel says whether SteerLab asked for it."""
    if not outcome:
        return None
    if outcome == "CANCELLED":
        meaning = ("the job was cancelled through SteerLab" if cancel_requested
                   else "the job was cancelled outside SteerLab (by a person "
                        "or an administrator)")
    elif outcome == "PREEMPTED":
        meaning = ("the scheduler preempted the job, which now waits to run "
                   "again" if waiting else "the scheduler preempted the job")
    else:
        meaning = _OUTCOME_MEANINGS.get(outcome, "")
    return f"The scheduler recorded {outcome}" + (f": {meaning}." if meaning
                                                  else ".")


def _with_outcome_note(entry: dict, *, waiting: bool) -> dict:
    entry["schedulerOutcomeNote"] = outcome_note(
        entry.get("schedulerOutcome"), waiting=waiting,
        cancel_requested=bool(entry.get("cancellationRequested")))
    return entry


# --- held parts (issue #12) ---------------------------------------------------

def _held_fields(job) -> dict:
    """``held`` (the record's own stamp: ``reason``, ``since``,
    ``schedulerMessage``, and the last refusal) and ``heldNote`` (a sentence)
    for a part SteerLab holds because the cluster's job limit is reached;
    both None otherwise. Beside the status, never a status value: a held
    part reads ``pending``, which every client already treats as waiting."""
    from .jobs import HELD_PART_NOTE, JobManager
    if not JobManager.is_held(job):
        return {"held": None, "heldNote": None}
    stamp = job.result["held"]
    held = {key: stamp.get(key) for key in
            ("reason", "since", "schedulerMessage", "lastAttemptAt",
             "lastRefusal") if stamp.get(key) is not None}
    return {"held": held, "heldNote": HELD_PART_NOTE}


def held_parts_note(held: int, parts: int) -> str:
    """A split run's note when some of its parts are held."""
    return (f"{held} of {parts} part{'' if parts == 1 else 's'} "
            f"{'is' if held == 1 else 'are'} held by SteerLab: the cluster's "
            "job limit is reached. The controller submits each held part, in "
            "order, when a slot frees; while the controller is not running "
            "they wait, and the run is merged only when every part has "
            "finished")


# --- progress ----------------------------------------------------------------

#: Earlier counts of a record file, so a status read parses only the lines
#: written since: ``path -> (device, inode, validEnd, completedRecords)``.
#: ``resume.written_records`` checks that the boundary still ends a line.
_COUNTS: dict[str, tuple[int, int, int, int]] = {}
_COUNTS_LOCK = threading.Lock()
_COUNTS_LIMIT = 512


def _written(directory: str) -> dict:
    from ..experiment import resume as resume_mod
    path = os.path.join(directory, resume_mod.GENERATIONS_FILENAME)
    try:
        st = os.stat(path)
        identity = (st.st_dev, st.st_ino)
    except OSError:
        identity = None
    known = None
    with _COUNTS_LOCK:
        cached = _COUNTS.get(path)
    if identity is not None and cached is not None and cached[:2] == identity:
        known = (cached[2], cached[3])
    written = resume_mod.written_records(directory, known=known)
    if identity is not None and written.get("validEnd") is not None:
        with _COUNTS_LOCK:
            if len(_COUNTS) >= _COUNTS_LIMIT and path not in _COUNTS:
                _COUNTS.clear()
            _COUNTS[path] = (*identity, written["validEnd"],
                             written["completedRecords"])
    return written


def job_progress(job) -> dict:
    """Records written so far, out of the records this job owns, and when the
    last one was written. Read from the run directory the job's resume
    pointer names (the same pointer a resume follows), and counted the way a
    resume counts them (``resume.written_records``).

    ``recordsExpected`` is exact for a shard of a split run (its
    ``shard.json``). For a whole run it is the submission's planning figure
    (``preflightEstimate``), which can slightly exceed the real count when
    only some prompts carry options. None when neither was recorded."""
    progress = {
        "unit": None, "recordsWritten": None, "recordsExpected": None,
        "recordsExpectedSource": None, "lastRecordAt": None,
        "tornTail": False, "runID": None, "stage": None, "detail": None,
    }
    located = _run_directory(job)
    if located.get("detail"):
        progress["detail"] = located["detail"]
    directory = located.get("runDirectory")
    progress["stage"] = located.get("stage")
    if directory is None:
        _fill_planned_estimate(progress, job)
        return progress
    progress["runID"] = os.path.basename(os.path.normpath(directory))
    if not os.path.isdir(directory):
        progress["detail"] = ("the run folder on record is not on the "
                              "server's disk")
        return progress
    from ..experiment import sharding as sharding_mod
    written = _written(directory)
    progress.update({
        "unit": written["unit"],
        "recordsWritten": written["completedRecords"],
        "lastRecordAt": written["lastWrittenAt"],
        "tornTail": bool(written["tornTail"]),
    })
    stamp = sharding_mod.read_shard_stamp(directory)
    if stamp is not None and isinstance(stamp.get("expectedRecords"), int):
        progress["recordsExpected"] = stamp["expectedRecords"]
        progress["recordsExpectedSource"] = "shardStamp"
    else:
        _fill_planned_estimate(progress, job)
    if written["lastWrittenAt"] is None and not progress["detail"]:
        progress["detail"] = "no record has been written yet"
    return progress


def _fill_planned_estimate(progress: dict, job) -> None:
    """The submission's planning figure as ``recordsExpected``, for a whole
    study run only. A part of a split run never borrows it: the figure is
    the whole study's, and a part's own share is in its shard stamp once
    it starts."""
    rr = job.requested_resources or {}
    if rr.get("shardCount") is not None or rr.get("parentJob"):
        if not progress["detail"]:
            progress["detail"] = ("this part's share of the records is known "
                                  "once it starts (its shard stamp)")
        return
    if rr.get("shardChildren") or _job_verb(job) != "run":
        return
    expected = _preflight_planned_records(job)
    if expected is not None:
        progress["recordsExpected"] = expected
        progress["recordsExpectedSource"] = "preflightEstimate"


def _job_verb(job) -> str | None:
    """The study verb the job executes, from its record: the folded result's
    own ``verb``, else the ``--verb`` of the command it was submitted with."""
    result = job.result if isinstance(job.result, dict) else {}
    for candidate in (result, result.get("result")):
        if isinstance(candidate, dict) and isinstance(candidate.get("verb"),
                                                      str):
            return candidate["verb"]
    bundle = result.get("slurmBundle")
    for command in (result.get("command"),
                    bundle.get("command") if isinstance(bundle, dict) else None):
        if isinstance(command, list) and "--verb" in command:
            index = command.index("--verb")
            if index + 1 < len(command) and isinstance(command[index + 1], str):
                return command[index + 1]
    return None


def _run_directory(job) -> dict:
    """The run directory the job writes its records to, from its resume
    pointer (``<records>/<chain root>.resume``, written the moment the run
    directory exists), then from its folded result. A pipeline's records are
    its run stage's, named in its ledger."""
    from ..experiment import resume as resume_mod
    rr = job.requested_resources or {}
    result = job.result if isinstance(job.result, dict) else {}
    records_dir = rr.get("recordsDirectory") or result.get("recordsDirectory")
    directory = verb = None
    if isinstance(records_dir, str) and records_dir:
        chain = [str(x) for x in (rr.get("resubmitChain") or [])]
        root_id = chain[0] if chain else job.id
        pointer = resume_mod.read_pointer(resume_mod.pointer_path_for_record(
            os.path.join(records_dir, f"{root_id}.json")))
        if pointer and isinstance(pointer.get("runDirectory"), str):
            directory = pointer["runDirectory"]
            verb = str(pointer.get("verb") or "run")
    if directory is None:
        inner = result.get("result") if isinstance(result.get("result"),
                                                   dict) else {}
        for value in (result.get("runDirectory"), inner.get("runDirectory")):
            if isinstance(value, str) and value:
                directory = value
                verb = str(result.get("verb") or inner.get("verb") or "run")
                break
    if directory is None:
        return {"detail": ("no run folder is on record yet (the job has "
                           "not started its run, or it writes none)")}
    if verb == "pipeline":
        return _pipeline_run_stage(directory)
    if verb != "run":
        return {"runDirectory": None, "stage": verb,
                "detail": (f"this job is a {verb}; records written are "
                           "counted for study runs only")}
    return {"runDirectory": directory, "stage": "run"}


def _pipeline_run_stage(pipeline_directory: str) -> dict:
    from ..experiment.pipeline_ledger import resolve_reference
    try:
        with open(os.path.join(pipeline_directory, "pipeline.json"),
                  encoding="utf-8") as handle:
            ledger = json.load(handle)
    except (OSError, ValueError):
        ledger = None
    if not isinstance(ledger, dict):
        return {"runDirectory": None, "stage": "pipeline",
                "detail": "the pipeline has not written its ledger yet"}
    stages = ledger.get("stageResults")
    stages = stages if isinstance(stages, dict) else {}
    started = [name for name, entry in stages.items()
               if isinstance(entry, dict) and entry.get("status") == "started"]
    stage = started[-1] if started else None
    run_stage = stages.get("run") if isinstance(stages.get("run"), dict) else {}
    run_dir = resolve_reference(run_stage.get("runDirectory"),
                                pipeline_directory)
    if isinstance(run_dir, str) and run_dir:
        return {"runDirectory": run_dir, "stage": stage or "run"}
    return {"runDirectory": None, "stage": stage,
            "detail": ("the pipeline has not reached its run stage; records "
                       "written are counted for that stage only")}


def _preflight_planned_records(job) -> int | None:
    preflight = (job.requested_resources or {}).get("preflight")
    if not isinstance(preflight, dict):
        return None
    for check in preflight.get("checks") or []:
        if not isinstance(check, dict) or check.get("id") != "walltime":
            continue
        data = check.get("data")
        planned = data.get("plannedRecords") if isinstance(data, dict) else None
        if isinstance(planned, int) and not isinstance(planned, bool):
            return planned
    return None


def _combined_progress(shards: list[dict]) -> dict:
    """A split run's progress: the parts' records summed. ``recordsExpected``
    is summed only when every part's is known and exact (its shard stamp),
    and the newest part's last write is the run's."""
    found = [entry.get("progress") or {} for entry in shards
             if entry.get("found")]
    written = [p.get("recordsWritten") for p in found]
    expected = [p.get("recordsExpected") for p in found]
    sources = {p.get("recordsExpectedSource") for p in found}
    times = [p.get("lastRecordAt") for p in found
             if p.get("lastRecordAt") is not None]
    complete = bool(found) and len(found) == len(shards)
    units = {p.get("unit") for p in found if p.get("unit")}
    combined = {
        "unit": units.pop() if len(units) == 1 else None,
        "recordsWritten": (sum(w for w in written if isinstance(w, int))
                           if any(isinstance(w, int) for w in written)
                           else None),
        "recordsExpected": (sum(expected)
                            if complete and sources == {"shardStamp"}
                            and all(isinstance(e, int) for e in expected)
                            else None),
        "recordsExpectedSource": ("shardStamp"
                                  if complete and sources == {"shardStamp"}
                                  else None),
        "lastRecordAt": max(times) if times else None,
        "tornTail": any(p.get("tornTail") for p in found),
        "runID": None,
        "stage": None,
        "detail": None,
    }
    if combined["recordsExpected"] is None:
        combined["detail"] = ("the total is summed only when every part has "
                              "started and recorded its own share")
    return combined


# --- log tail ----------------------------------------------------------------

def log_tail(manager, job, lines: int) -> dict:
    """The last ``lines`` controller log lines of ``job`` and, for a scheduler
    job, the last ``lines`` lines of its own ``slurm-<id>.out`` and ``.err``
    (each read from at most ``MAX_TAIL_BYTES`` at the end of the file)."""
    fetched = job.log_tail(lines + 1)
    shown = fetched[-lines:]
    document = {
        "schemaVersion": LOG_TAIL_SCHEMA_VERSION,
        "jobId": job.id,
        "status": job.status,
        "observedAt": time.time(),
        "lines": shown,
        "lineCount": len(shown),
        "truncated": len(fetched) > lines,
        "schedulerOutput": None,
        "notes": [],
    }
    rr = job.requested_resources or {}
    if rr.get("shardChildren"):
        shard_ids = [str(x) for x in rr["shardChildren"]]
        document["notes"].append(
            "this run is split across shard jobs, each with its own scheduler "
            "output; read a part's tail by its job id: " + ", ".join(shard_ids))
        return document
    if job.executor != "slurm":
        document["notes"].append(
            "this job runs on the server itself, so its output is in the "
            "lines above; it has no scheduler output files")
        return document
    document["schedulerOutput"] = scheduler_output(job, lines)
    return document


def scheduler_output(job, lines: int) -> dict:
    """The tails of the job's own scheduler output files. The directory comes
    from the job's record (its Slurm bundle, or the directory of the script
    it was submitted from) and the file names from the scheduler id on
    record, which must look like one; nothing comes from the request."""
    from .executors import is_scheduler_job_id
    slurm_id = str(job.executor_job_id or "")
    if not is_scheduler_job_id(slurm_id):
        return {"available": False,
                "detail": ("no scheduler job id is on record, so there is no "
                           "scheduler output to read")}
    bundle_dir = _bundle_directory(job)
    if bundle_dir is None:
        return {"available": False,
                "detail": ("the job's record does not say where its "
                           "scheduler output is written")}
    files = {}
    for key, suffix in (("stdout", "out"), ("stderr", "err")):
        files[key] = _file_tail(
            os.path.join(bundle_dir, f"slurm-{slurm_id}.{suffix}"),
            lines, waiting=job.status in ("submitted", "pending"))
    return {"available": True, **files}


def _bundle_directory(job) -> str | None:
    result = job.result if isinstance(job.result, dict) else {}
    bundle = result.get("slurmBundle")
    if isinstance(bundle, dict) and isinstance(bundle.get("bundle_dir"), str):
        return bundle["bundle_dir"]
    script = (job.requested_resources or {}).get("scriptPath")
    if isinstance(script, str) and script:
        return os.path.dirname(script)
    return None


def _file_tail(path: str, lines: int, *, waiting: bool) -> dict:
    name = os.path.basename(path)
    entry = {"name": name, "exists": False, "lines": [], "lineCount": 0,
             "truncated": False, "sizeBytes": None, "bytesRead": 0,
             "detail": None}
    try:
        size = os.path.getsize(path)
    except OSError:
        entry["detail"] = (
            f"{name} does not exist yet: the scheduler creates it when the "
            "job starts" if waiting else
            f"{name} does not exist (the job may not have started yet, or "
            "the file was removed)")
        return entry
    start = max(0, size - MAX_TAIL_BYTES)
    try:
        with open(path, "rb") as handle:
            handle.seek(start)
            data = handle.read(MAX_TAIL_BYTES)
    except OSError as exc:
        entry.update({"exists": True, "sizeBytes": size,
                      "detail": f"{name} could not be read ({exc.strerror or exc})"})
        return entry
    text = data.decode("utf-8", errors="replace")
    parts = text.splitlines()
    if start > 0 and parts:
        parts = parts[1:]   # the first line began before the bytes read
    shown = parts[-lines:]
    entry.update({
        "exists": True, "sizeBytes": size, "bytesRead": len(data),
        "lines": shown, "lineCount": len(shown),
        "truncated": start > 0 or len(parts) > lines,
    })
    if size == 0:
        entry["detail"] = f"{name} exists and is empty so far"
    return entry
