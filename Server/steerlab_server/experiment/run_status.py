"""Run-directory status: the honest record of a stage that did not finish.

Retention principle (2026-07-24, external review
``docs/CLUSTER-SHARDING-JUDGING-REVIEW-2026-07-23.md``):

    Failure must prevent an evidentiary success claim, but it must not
    prevent the researcher from retrieving the data and diagnostic record
    that were actually produced.

Before this module a failing stage produced an exception and nothing on
disk — "the data still exists somewhere under /scratch" was the retention
policy. A stage that writes a ``RunStatus`` instead leaves a directory that
is retrievable, self-describing, and — critically — *not mistakable for a
result*: the completion artifact (``judge-report.json`` for evaluate) is
written only on success, and ``status`` says plainly what happened.

The two states are deliberately asymmetric. ``completed`` means every
expected unit of work landed. Anything else is ``failed`` (or still
``inProgress`` if the process died without unwinding), carries
``evidenceComplete: false``, and is barred from evidence-grade gates by
``is_partial``. There is no "mostly done" that reads as done.
"""

from __future__ import annotations

import contextvars
import json
import os
import time
import traceback

#: The run directory the CURRENT unit of work most recently created.
#:
#: Set by ``paths.make_unique_run_directory``, so every stage registers its
#: directory without any per-task plumbing. This is what lets a failure be
#: retained on paths where the stage never reports its directory — the
#: direct ``/api/experiment/{name}/{verb}`` route runs ``tasks.sweep`` and
#: friends inline, and those return their directory only on SUCCESS.
#:
#: A context variable rather than a global: each job thread gets its own
#: context, so two concurrent jobs cannot read each other's directory. Same
#: pattern as ``run_config.current_job_id``.
#:
#: Holds the LAST directory created, which for a multi-stage verb (pipeline)
#: is the stage that was running when it failed — the useful one.
current_run_directory: contextvars.ContextVar[str | None] = \
    contextvars.ContextVar("steerlab_current_run_directory", default=None)

#: Machine-readable status, written to every run directory a status-aware
#: stage creates. Cross-engine filename.
STATUS_FILENAME = "run-status.json"

#: Human-readable failure note, written beside it. The researcher opening an
#: imported run directory should not have to parse JSON to learn it failed.
FAILURE_NOTE_FILENAME = "FAILED.md"

#: Raw malformed judge responses, one JSON object per line. The 2026-07-23
#: incident ("winner None") left the researcher with a parser complaint and
#: no way to see what the judge actually wrote.
INVALID_RESPONSES_FILENAME = "judge-failures.jsonl"

STATUS_SCHEMA = 1

#: Attribute a failing stage attaches to its exception, naming the run
#: directory that holds its partial evidence. The bundle layer needs to
#: package that directory, and an exception is the only thing that crosses
#: the boundary between a stage that created a directory internally and the
#: caller that has to ship it home.
PARTIAL_RUN_ATTR = "steerlab_partial_run_directory"


def partial_run_directory(exc: BaseException) -> str | None:
    """The partial run directory for a failed unit of work, or None when
    nothing was produced (which is different from losing it).

    Two sources, in order of authority:

    1. what the failing stage ATTACHED to the exception — a stage that
       tracks its own status names its directory exactly;
    2. the directory most recently CREATED in this context — every stage
       registers that automatically, which is what covers verbs like
       ``sweep`` and ``extract`` that report their directory only on
       success and would otherwise leave a failure with nothing to
       package.
    """
    path = getattr(exc, PARTIAL_RUN_ATTR, None)
    if isinstance(path, str) and os.path.isdir(path):
        return path
    current = current_run_directory.get()
    return current if isinstance(current, str) and os.path.isdir(current) else None


class RunStatus:
    """Tracks and persists one stage's progress inside its run directory.

    Every mutator rewrites ``run-status.json`` in full: the file is small,
    and a crash between a counter bump and its write would otherwise make
    the record lie about work that did happen. Writes are best-effort — a
    status-file failure must never take down a run that is otherwise fine,
    and must never replace a real error with an I/O error.
    """

    def __init__(self, run_directory: str, *, stage: str,
                 experiment: str | None = None,
                 source_run: str | None = None,
                 expected: list[str] | None = None,
                 item_label: str = "item"):
        self.run_directory = run_directory
        self.stage = stage
        self.experiment = experiment
        self.source_run = source_run
        self.expected = list(expected or [])
        self.completed: list[str] = []
        #: What this stage produces one of, singular ("judgment", "record",
        #: "cell", "concept"). Stages differ in what they emit but not in
        #: what a reader needs to know — how much survived — so the COUNT
        #: key is shared and only the label varies.
        self.item_label = item_label
        self.item_count = 0
        self.invalid_count = 0
        self.status = "inProgress"
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.error: str | None = None
        self.error_type: str | None = None

    @property
    def judgment_count(self) -> int:
        """Back-compat alias for the evaluate path's original name."""
        return self.item_count

    # -- snapshot ---------------------------------------------------------

    def to_dict(self) -> dict:
        pending = [name for name in self.expected if name not in self.completed]
        data = {
            "schemaVersion": STATUS_SCHEMA,
            "stage": self.stage,
            "status": self.status,
            "startedAt": self.started_at,
            # Only a completed stage claims complete evidence. Everything
            # else — including a stage still running — is explicitly not.
            "evidenceComplete": self.status == "completed",
            "itemLabel": self.item_label,
            "itemsWritten": self.item_count,
            "invalidResponses": self.invalid_count,
        }
        if self.experiment:
            data["experiment"] = self.experiment
        if self.source_run:
            data["sourceRun"] = self.source_run
        if self.expected:
            data["expectedUnits"] = list(self.expected)
            data["completedUnits"] = list(self.completed)
            # What is MISSING is named, never left to be inferred from a
            # diff the reader has to compute.
            data["pendingUnits"] = pending
        if self.finished_at is not None:
            data["finishedAt"] = self.finished_at
        if self.error:
            data["error"] = self.error
            data["errorType"] = self.error_type
        return data

    def write(self) -> None:
        """Replace the status file ATOMICALLY.

        Write-in-place (truncate, then write) leaves a torn file if the
        process dies mid-write — and this file is written precisely when
        processes are dying. Combined with a reader that treated malformed
        JSON as "no status", the crash that made a run partial could also
        make it look complete. ``os.replace`` is atomic on POSIX, so a
        reader sees either the whole previous status or the whole new one.
        """
        path = os.path.join(self.run_directory, STATUS_FILENAME)
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except OSError:
            # Best-effort by design (a status write must not kill a run),
            # but never leave the temp file behind to be mistaken for data.
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # -- progress ---------------------------------------------------------

    def note_item(self, count: int = 1) -> None:
        """One (or ``count``) more units of this stage's output landed."""
        self.item_count += count
        self.write()

    #: The evaluate path's original spelling.
    note_judgment = note_item

    def note_judge_complete(self, name: str) -> None:
        if name not in self.completed:
            self.completed.append(name)
        self.write()

    def note_invalid_response(self, record: dict) -> None:
        """Persist one malformed judge response verbatim.

        Appended, never rewritten: a retry that later succeeds still leaves
        its failed attempt on disk, because "the judge needed two tries"
        is exactly the kind of fact a quiet retry erases.
        """
        self.invalid_count += 1
        try:
            path = os.path.join(self.run_directory,
                                INVALID_RESPONSES_FILENAME)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({**record, "at": time.time()},
                                        sort_keys=True) + "\n")
        except OSError:
            pass
        self.write()

    # -- terminal ---------------------------------------------------------

    def complete(self) -> None:
        self.status = "completed"
        self.finished_at = time.time()
        self.write()

    def checkpointed(self, reason: str | None = None) -> None:
        """A durably PARKED stage, not a failed one.

        Checkpoint-on-signal is the reliability path working as designed:
        the run stopped on purpose and is resumable. It still is not
        complete, so `is_partial` remains true and no evidence-grade gate
        will take it — but it gets no FAILED.md, because calling a
        successful requeue a failure would train the researcher to ignore
        the marker that matters.
        """
        self.status = "checkpointed"
        self.finished_at = time.time()
        if reason:
            self.error = reason
            self.error_type = "CheckpointRequested"
        self.write()

    def fail(self, exc: BaseException) -> None:
        self.status = "failed"
        self.finished_at = time.time()
        self.error = str(exc)
        self.error_type = type(exc).__name__
        self.write()
        self._write_failure_note(exc)
        try:
            # Tell whoever catches this where the surviving evidence is —
            # the bundle layer cannot otherwise know that a stage which
            # raised had already created a directory worth shipping home.
            setattr(exc, PARTIAL_RUN_ATTR, self.run_directory)
        except (AttributeError, TypeError):
            pass

    def _write_failure_note(self, exc: BaseException) -> None:
        pending = [name for name in self.expected if name not in self.completed]
        lines = [
            f"# {self.stage} FAILED",
            "",
            "This run directory is a **failure record**, not a result. The "
            "data below is real and was produced before the failure; it is "
            "preserved so it can be inspected and retried, and it must not "
            "be cited as a completed "
            f"{self.stage}.",
            "",
            f"- **Stage:** {self.stage}",
        ]
        if self.experiment:
            lines.append(f"- **Experiment:** {self.experiment}")
        if self.source_run:
            lines.append(f"- **Source run:** {self.source_run}")
        lines += [
            f"- **Error:** `{type(exc).__name__}: {exc}`",
            f"- **{self.item_label.capitalize()}s written before the "
            f"failure:** {self.item_count}",
        ]
        if self.invalid_count:
            lines.append(
                f"- **Malformed judge responses:** {self.invalid_count} "
                f"(raw text in `{INVALID_RESPONSES_FILENAME}`)")
        if self.completed:
            lines.append(f"- **Completed:** {', '.join(self.completed)}")
        if pending:
            lines.append(f"- **Did not run / incomplete:** {', '.join(pending)}")
        lines += [
            "",
            "## Traceback",
            "",
            "```",
            "".join(traceback.format_exception(
                type(exc), exc, exc.__traceback__)).rstrip(),
            "```",
            "",
        ]
        try:
            path = os.path.join(self.run_directory, FAILURE_NOTE_FILENAME)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines))
        except OSError:
            pass


# -- readers ---------------------------------------------------------------


#: `read_status` result for a status file that EXISTS but cannot be parsed.
#: Distinct from None (no file at all) because the two mean opposite things
#: about whether the run may be trusted.
UNREADABLE = "unreadable"


def heal_after_completion(run_directory: str) -> bool:
    """A continuation COMPLETED a run whose directory still carries a
    failure-era record: rewrite the record to match reality.

    Observed live (2026-08-11, memo-study campaigns c18/c19): a pipeline that
    failed on a judge 402, was resumed, and ran to ``disposition: completed``
    still had the original attempt's ``run-status.json`` (``status: failed``)
    and ``FAILED.md`` in its directory — anything reading those (agents, the
    Results Explorer, ``is_partial``) saw a failed run that must not be
    cited, contradicting the ledger. Resume updated the ledger and nothing
    else.

    Healing preserves history rather than erasing it: the old error moves to
    ``supersededError`` with a ``healedAt`` stamp, ``status`` becomes
    ``completed``, and ``FAILED.md`` is removed (its claim — "not a result,
    must not be cited" — is now false, and a false failure marker trains the
    researcher to ignore the marker that matters). Returns whether anything
    changed. Best-effort like every status write: healing must never take
    down the completion it records.
    """
    healed = False
    path = os.path.join(run_directory, STATUS_FILENAME)
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict) and payload.get("status") != "completed":
        history = {key: payload.get(key) for key in ("error", "errorType")
                   if payload.get(key)}
        payload["status"] = "completed"
        payload["evidenceComplete"] = True
        payload["error"] = None
        payload["errorType"] = None
        if history:
            payload["supersededError"] = history
        payload["healedAt"] = time.time()
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            healed = True
        except OSError:
            pass
    note = os.path.join(run_directory, FAILURE_NOTE_FILENAME)
    if os.path.exists(note):
        try:
            os.remove(note)
            healed = True
        except OSError:
            pass
    return healed


def read_status(run_directory: str):
    """``dict`` | ``None`` | ``UNREADABLE``.

    - ``None``       — no status file. A LEGACY run: unannotated, not
                       incomplete. Its own completion artifacts govern.
    - ``UNREADABLE`` — a status file is present but malformed. Something
                       wrote it and did not finish, so the run is suspect.
    - ``dict``       — the record.

    Collapsing the last two into None was a fail-OPEN (external review
    2026-07-24, finding 4): a process killed mid-write left a torn file,
    which then read as "no status", which read as "legacy", which read as
    citable. The crash that made a run partial made it look complete.
    """
    path = os.path.join(run_directory, STATUS_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError:
        # Present but unopenable (permissions, I/O error) — not a legacy
        # run, and not something to wave through.
        return UNREADABLE
    except json.JSONDecodeError:
        return UNREADABLE
    return data if isinstance(data, dict) else UNREADABLE


def has_readable_status(run_directory: str) -> bool:
    """Whether this directory already carries a status a reader can trust.

    The question every WRITER should ask before deciding not to overwrite:
    a TORN status is not an account of anything, so replacing it with a
    real one is an improvement, where clobbering a stage's own detailed
    record would not be.
    """
    return isinstance(read_status(run_directory), dict)


def is_partial(run_directory: str) -> bool:
    """Whether this directory is a known-incomplete record.

    An ABSENT status file is not partial — legacy runs carry none and are
    governed by their completion artifacts; treating them as incomplete
    would retroactively invalidate real results.

    An UNREADABLE one IS partial. It fails closed: a torn status file is
    evidence that a writer died, which is exactly the situation the marker
    exists to record, and the safe reading of "I cannot tell" is "not
    citable".

    A readable status that does not say ``completed`` is partial, including
    ``inProgress`` (a process killed without unwinding never got to write
    ``failed``, and the work is no more complete for it).
    """
    status = read_status(run_directory)
    if status is None:
        return False
    if status is UNREADABLE:
        return True
    return str(status.get("status") or "") != "completed"
