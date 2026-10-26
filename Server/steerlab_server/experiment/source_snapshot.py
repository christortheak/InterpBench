"""Which records of its source run an analysis read.

A run's ``generations.jsonl`` is append-only, and a run parked at a
checkpoint gains records when it is continued. An analysis names its source
run (``source-run.txt``), but a name alone cannot tell an analysis of the
finished run from one made while it was paused: after the continuation both
name the same directory. Review finding (October 23, 2026): a checkpoint's
analysis (6 over two pairs) was exported as the completed run's (3 over four
pairs), with nothing to say so.

So an analysis records the snapshot it read (``source-snapshot.json``): the
source's status at that moment, how many records it held, and the size and
SHA-256 of the exact ``generations.jsonl`` bytes it parsed. A reader compares
that with the run as it stands (:func:`compare`) and says when they differ
instead of presenting the analysis as the run's. An analysis made before the
stamp existed is ``notRecorded``: never assumed to match, and flagged only
when the run itself shows it was paused and continued.

Swift twin: ``SourceSnapshot`` (Sources/ExperimentKit/SourceSnapshot.swift),
the same keys, statuses, verdicts, and sentences, held to one contract by
``Tests/Fixtures/cross-engine/source-snapshot.json``. Nothing here changes a
statistic: the analysis reads the same records it always read.
"""
from __future__ import annotations

import hashlib
import io
import json
import os

# Light on purpose: the app-free client's export reads snapshots too.
from . import paths, run_status

STAMP_FILE = "source-snapshot.json"
SCHEMA_VERSION = 1

#: A source run's state as a snapshot records it. ``completed`` means its
#: records are final: the completion report is written, and no status says
#: the run failed or cannot be read. A park left beside a report
#: (``checkpointed`` or ``inProgress``, never settled because the run was
#: continued before 0.9.16 settled parks) is completed too, since a run writes
#: its report only once every record is in. Every other value names why the
#: run is not complete.
COMPLETED = "completed"
NO_COMPLETION_REPORT = "noCompletionReport"
UNREADABLE = "unreadable"

#: How a status reads inside a sentence.
_STATUS_WORDS = {
    "checkpointed": "paused at a checkpoint",
    "inProgress": "still running",
    "failed": "marked failed",
    UNREADABLE: "in an unknown state (its status file could not be read)",
    NO_COMPLETION_REPORT: "without a completion report",
}


def status(run_dir: str) -> str:
    """The run's state now: ``completed``, the status its ``run-status.json``
    records (``checkpointed``, ``inProgress``, ``failed``, …), ``unreadable``
    for a torn status file, or ``noCompletionReport``."""
    recorded = run_status.read_status(run_dir)
    if recorded is run_status.UNREADABLE:
        return UNREADABLE
    reported = os.path.isfile(os.path.join(run_dir, "report.json"))
    if isinstance(recorded, dict):
        value = recorded.get("status")
        if isinstance(value, str) and value and value != COMPLETED and not (
                reported and value in run_status.PARKED_STATUSES):
            return value
    return COMPLETED if reported else NO_COMPLETION_REPORT


def was_continued(run_dir: str) -> bool:
    """Whether the run was parked and then continued: its status records a
    settled park (``resumedFrom``, written by
    ``run_status.settle_after_resumed_completion``), or a park was left beside
    the completion report (continued before parks were settled)."""
    recorded = run_status.read_status(run_dir)
    if not isinstance(recorded, dict):
        return False
    return bool(recorded.get("resumedFrom")) or (
        recorded.get("status") in run_status.PARKED_STATUSES
        and os.path.isfile(os.path.join(run_dir, "report.json")))


def _parse(data: bytes) -> list:
    """The records exactly as the analysis has always read them: text lines
    (universal newlines), stripped, blank lines skipped."""
    records = []
    for line in io.TextIOWrapper(io.BytesIO(data), encoding="utf-8"):
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _count(data: bytes) -> int:
    """Non-blank lines: the record count a snapshot states, read the same way
    on both engines (no JSON decode, so a reader can count any run), one line
    at a time rather than as copies of the whole file."""
    lines = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8", errors="replace")
    return sum(1 for line in lines if line.strip())


def describe(data: bytes) -> dict:
    """Records, size, and SHA-256 of a ``generations.jsonl``'s bytes."""
    return {"records": _count(data), "generationsBytes": len(data),
            "generationsSHA256": hashlib.sha256(data).hexdigest()}


def read_generations(run_dir: str) -> tuple[list, dict]:
    """The source's records and the snapshot they came from, read in ONE
    pass, so the stamp names exactly the bytes that were parsed even while a
    running source keeps appending. The status is read FIRST: a source that
    completes between the two reads is then stamped unfinished, which a
    reader says, never completed over records that were not all there."""
    state = status(run_dir)
    with open(os.path.join(run_dir, "generations.jsonl"), "rb") as handle:
        data = handle.read()
    stamp = {"schemaVersion": SCHEMA_VERSION,
             "sourceRun": os.path.basename(os.path.normpath(run_dir)),
             "status": state, **describe(data)}
    return _parse(data), stamp


def write(out: str, stamp: dict) -> None:
    with open(os.path.join(out, STAMP_FILE), "w", encoding="utf-8") as handle:
        json.dump(stamp, handle, indent=2, sort_keys=True)
        handle.write("\n")


def latest_completed_run(name: str, root: str | None) -> str | None:
    """The newest run of the study that has completed (a completion report,
    and no status saying otherwise)."""
    runs = paths.runs_directory(root)
    if not os.path.isdir(runs):
        return None
    for entry in sorted(os.listdir(runs), reverse=True):
        directory = os.path.join(runs, entry)
        if (entry.endswith(f"-exp-{name}-run")
                and os.path.isfile(os.path.join(directory, "generations.jsonl"))
                and status(directory) == COMPLETED):
            return directory
    return None


def choose_source(name: str, root: str | None, source_run: str | None, *, log) -> str | None:
    """The run an analysis reads: the one named, else the newest COMPLETED
    run, else (none has completed) the newest run with records. An
    unfinished source is analyzed as it stands, never refused, and never
    silently: the warning says so, and the snapshot stamp keeps saying so
    after the run is continued."""
    from .run_artifacts import latest_run
    run_dir = source_run or latest_completed_run(name, root) or latest_run(name, root)
    # A named directory with no records is left to the reader below, which
    # says so; there is no snapshot to warn about.
    if run_dir and os.path.isfile(os.path.join(run_dir, "generations.jsonl")):
        state = status(run_dir)
        if state != COMPLETED:
            why = "" if source_run else f" (no run of '{name}' has completed)"
            log(f"WARNING: source run '{os.path.basename(os.path.normpath(run_dir))}' "
                f"is {status_words(state)}{why} — this analysis reads the records it "
                f"holds now and records that snapshot in {STAMP_FILE}, so it is never "
                "presented as the analysis of the finished run")
    return run_dir


def status_words(state: str) -> str:
    return _STATUS_WORDS.get(state, f"not complete (status: {state})")


def read_stamp(analysis_dir: str) -> dict | None:
    try:
        with open(os.path.join(analysis_dir, STAMP_FILE), encoding="utf-8") as handle:
            stamp = json.load(handle)
    except (OSError, ValueError):
        return None
    # Strict like the Swift twin's decoder: every key, of its type. (That
    # decoder also takes an integral 8.0 where an integer is due; neither
    # engine writes one.)
    typed = (("schemaVersion", int), ("sourceRun", str), ("status", str),
             ("records", int), ("generationsBytes", int), ("generationsSHA256", str))
    if not isinstance(stamp, dict) or any(
            not isinstance(stamp.get(key), kind) or isinstance(stamp.get(key), bool)
            for key, kind in typed):
        return None
    return stamp


def current(run_dir: str) -> dict | None:
    """The run's ``generations.jsonl`` as it stands, described; None when it
    cannot be read."""
    try:
        with open(os.path.join(run_dir, "generations.jsonl"), "rb") as handle:
            return describe(handle.read())
    except OSError:
        return None


#: Verdicts. Only the last three are said to a reader.
MATCHES = "matches"
NOT_RECORDED = "notRecorded"
UNFINISHED_SOURCE = "unfinishedSource"
CHANGED = "changed"
NOT_RECORDED_RUN_CONTINUED = "notRecordedRunContinued"
NOTEWORTHY = (UNFINISHED_SOURCE, CHANGED, NOT_RECORDED_RUN_CONTINUED)


def verdict(stamp: dict | None, now: dict | None, *, continued: bool) -> dict:
    """The comparison of an analysis's stamp with its run as it stands.

    ``now`` is :func:`current` of the run (None: unreadable, so only the
    stamp speaks); ``continued`` is :func:`was_continued`. Pure, so both
    engines answer the shared fixture's cases from the same inputs."""
    if stamp is None:
        return {"state": NOT_RECORDED_RUN_CONTINUED if continued else NOT_RECORDED}
    found = {"analyzedRecords": stamp.get("records"),
             "analyzedStatus": stamp.get("status")}
    if now is not None:
        found["runRecords"] = now["records"]
    if stamp.get("status") != COMPLETED:
        return {"state": UNFINISHED_SOURCE, **found}
    if now is not None and now["generationsSHA256"] != stamp.get("generationsSHA256"):
        return {"state": CHANGED, **found}
    return {"state": MATCHES, **found}


def compare(analysis_dir: str, run_dir: str) -> dict:
    """:func:`verdict` for an analysis directory and the run it names."""
    return verdict(read_stamp(analysis_dir), current(run_dir),
                   continued=was_continued(run_dir))


def sentence(found: dict, run: str) -> str | None:
    """What a reader is told, or None when there is nothing to say."""
    state = found.get("state")
    analyzed, now = found.get("analyzedRecords"), found.get("runRecords")
    if state == UNFINISHED_SOURCE:
        held = (f", and the run now holds {now}"
                if isinstance(now, int) and now != analyzed else "")
        return (f"This analysis read run {run} while it was "
                f"{status_words(str(found.get('analyzedStatus')))}, with {analyzed} "
                f"records{held}. Re-analyze the run for effects over all its records.")
    if state == CHANGED:
        return (f"This analysis read {analyzed} records of run {run}, and the run's "
                f"records have changed since (it now holds {now}). Re-analyze the run "
                "for effects over its current records.")
    if state == NOT_RECORDED_RUN_CONTINUED:
        return (f"This analysis does not record which of run {run}'s records it read, "
                "and the run was paused and continued, so it may predate the "
                "continuation. Re-analyze the run to be sure its effects cover every "
                "record.")
    return None
