"""A study submission that is sent twice lands once (public issue #11).

A large study's submission is answered only after the server has measured
every task prompt, which can take minutes. A client that gave up waiting, or
whose connection dropped, cannot know whether a job was created; the server
went on and created it. Submitted again, the same study became a second job.

Each client therefore sends a **submission key** with the request
(``submissionKey``): the SHA-256 of the request itself, with the staged
bundle's path replaced by the bundle's CONTENT identity (its files' digests,
never its packaging time), so the same request for the same study carries
the same key however often it is repackaged, and any change to what is
submitted (the verb, the resources, the number of parts, the target, a dry
run) changes it. This module answers a repeat:

- while a submission with that key is still being prepared, a repeat waits
  for it and is answered with the job it created;
- while that job has not ended (``jobs.TERMINAL``), a repeat is answered with
  it, plainly marked ``alreadySubmitted``, and nothing is submitted again;
- once it has ended (failed, cancelled, succeeded, or a dry run's
  ``prepared``), a repeat is a new submission, as before.

A request without a key (every older client) is submitted exactly as before.
The key is persisted in the job record's requested resources
(``submissionKey``) when the record is created, and a compact copy of the
first answer (``submissionAnswer``) is kept beside it, so a repeat after a
server restart is still recognised. A repeat is answered with what is true
NOW: the job's status, the job carrying it, and a split run's held parts.
Capability: ``cluster.submissionKey``.

A key is only as good as its derivation, and each client derives it from
what it sends: the same command with the same flags, from the same client,
carries the same key; a request from the other client is a different body
and carries a different one.
"""

from __future__ import annotations

import contextvars
import re
import threading

from .jobs import TERMINAL

#: The body key both clients send.
BODY_KEY = "submissionKey"
#: The requested-resources keys this module writes on a job record.
RECORD_KEY = "submissionKey"
ANSWER_KEY = "submissionAnswer"
#: A key is a lowercase SHA-256 hex digest.
_KEY_SHAPE = re.compile(r"[0-9a-f]{64}")
#: How far a resumed job's ``resubmittedAs`` chain is followed.
_CHAIN_LIMIT = 64


class SubmissionKeyError(ValueError):
    """A key in the wrong shape, or one reused for a different request."""

    def __init__(self, message: str, *, code: str, repair_action: str,
                 status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.repair_action = repair_action
        self.status = status

    def detail(self) -> dict:
        return {"code": self.code, "message": str(self),
                "repairAction": self.repair_action}


def from_body(body: dict) -> str | None:
    """The request's key, or None when it carries none (an older client).

    A key in the wrong shape is refused rather than ignored: ignoring it would
    quietly take away the protection the client asked for."""
    raw = body.get(BODY_KEY)
    if raw is None:
        return None
    if not isinstance(raw, str) or not _KEY_SHAPE.fullmatch(raw):
        raise SubmissionKeyError(
            "submissionKey must be a lowercase SHA-256 hex digest (64 "
            "characters); nothing was submitted",
            code="submissionKeyMalformed",
            repair_action=(
                "send the key the client derives from the request, or leave "
                "submissionKey out to submit without repeat protection"))
    return raw


class _Preparing:
    """One submission with a key, while it is being prepared."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.job_id: str | None = None


_GUARD = threading.Lock()
_PREPARING: dict[str, _Preparing] = {}

#: The key of the submission being prepared on this thread, so the job record
#: carries it from the moment it is created (``keyed``), not from a second
#: write after the submission returns.
CURRENT_KEY: contextvars.ContextVar = contextvars.ContextVar(
    "steerlab_submission_key", default=None)


def keyed(requested_resources: dict) -> dict:
    """``requested_resources`` with this submission's key, when it has one.
    Called where a submission creates its own job record (never a part's)."""
    key = CURRENT_KEY.get()
    if key is None:
        return requested_resources
    return {**requested_resources, RECORD_KEY: key}


def _chain_tail(jobs, job):
    """The job carrying a resumed submission now: a checkpointed or cancelled
    record continued by a resubmission names it in ``result.resubmittedAs``."""
    seen = {job.id}
    current = job
    for _ in range(_CHAIN_LIMIT):
        follower = (current.result or {}).get("resubmittedAs")
        if not isinstance(follower, str) or follower in seen:
            break
        nxt = jobs.get(follower)
        if nxt is None:
            break
        seen.add(follower)
        current = nxt
    return current


def live_submission(jobs, key: str):
    """``(job, current)`` for the submission with this key that has not ended,
    or None. ``job`` is the original record (a resumed job's continuation
    copies the key, so only records with no ``resubmitChain`` count as the
    submission itself); ``current`` is the job carrying it now."""
    for job in jobs.list():
        resources = job.requested_resources or {}
        if resources.get(RECORD_KEY) != key or resources.get("resubmitChain"):
            continue
        current = _chain_tail(jobs, job)
        if current.status not in TERMINAL:
            return job, current
    return None


def _compact_answer(answer: dict) -> dict:
    """What a repeat is answered with, kept on the record: the first answer
    without the preflight report (the record already holds it) and with the
    bundle reduced to its identity."""
    compact = {k: v for k, v in answer.items() if k != "preflight"}
    bundle = answer.get("runBundle")
    if isinstance(bundle, dict):
        compact["runBundle"] = {
            k: bundle[k] for k in ("kind", "experiment", "bundlePath",
                                   "bundleSha256", "experimentContentHash")
            if k in bundle}
    return compact


def _stamp(jobs, answer: dict, key: str) -> None:
    """Keep the first answer beside the key the record already carries. A
    failure here never turns a job that landed into an error: the key was
    written with the record, and a repeat is answered from the record."""
    job = jobs.get(answer.get("jobId"))
    if job is None:
        return
    try:
        job.requested_resources = {**(job.requested_resources or {}),
                                   RECORD_KEY: key,
                                   ANSWER_KEY: _compact_answer(answer)}
        jobs.store.update(job)
    except Exception as exc:  # noqa: BLE001 - the job landed; say so, answer
        try:
            job.log(f"the first answer to this submission could not be kept "
                    f"beside its key ({type(exc).__name__}: {exc}); a repeat "
                    "is answered from the job record")
        except Exception:  # noqa: BLE001
            pass


def _answer_from_record(job, *, verb: str, dry_run: bool) -> dict:
    """A repeat's answer when the first one was not kept: the same fields,
    read from the job record."""
    result = job.result if isinstance(job.result, dict) else {}
    resources = job.requested_resources or {}
    bundle = result.get("runBundle") if isinstance(result.get("runBundle"), dict) else {}
    merge = resources.get("shardMerge") if isinstance(resources.get("shardMerge"), dict) else {}
    children = resources.get("shardChildren")
    return {"jobId": job.id,
            "experiment": bundle.get("experiment") or merge.get("experiment") or "",
            "verb": verb, "executor": job.executor, "dryRun": bool(dry_run),
            "runBundle": bundle, "slurmBundle": result.get("slurmBundle"),
            "slurmJobID": job.executor_job_id,
            "command": result.get("command") or [],
            "recordsDirectory": (result.get("recordsDirectory")
                                 or resources.get("recordsDirectory") or ""),
            "submissionDirectory": result.get("submissionDirectory") or "",
            "shardJobIDs": list(children) if isinstance(children, list) else None}


def _held_now(jobs, job) -> dict | None:
    """The split run's held parts as they are NOW (issue #12): the first
    answer's count goes stale as soon as the controller submits one."""
    resources = job.requested_resources or {}
    children = resources.get("shardChildren")
    if not isinstance(children, list) or not children:
        return None
    is_held = getattr(jobs, "is_held", None)
    if is_held is None:
        return None
    held = []
    stamp = None
    for cid in children:
        child = jobs.get(str(cid))
        if child is None:
            continue
        tail = _chain_tail(jobs, child)
        if is_held(tail):
            held.append(tail.id)
            stamp = stamp or (tail.result or {}).get("held")
    if not held:
        return None
    from .submissions import _held_parts_summary
    parts = int(resources.get("parallelJobs") or len(children))
    summary = _held_parts_summary(stamp, held, parts)
    count = len(held)
    summary["message"] = (
        f"{parts - count} of {parts} parts have been submitted; {count} "
        f"{'is' if count == 1 else 'are'} still held by SteerLab because the "
        "cluster's job limit is reached. The controller submits each held "
        "part, in order, when a slot frees; while the controller is not "
        "running they wait. The run is merged when every part has finished.")
    return summary


def _already_submitted(jobs, job, current, *, verb: str, dry_run: bool,
                       arrived_while_preparing: bool) -> dict:
    resources = job.requested_resources or {}
    kept = resources.get(ANSWER_KEY)
    answer = (dict(kept) if isinstance(kept, dict)
              else _answer_from_record(job, verb=verb, dry_run=dry_run))
    if answer.get("verb") not in (None, verb) \
            or bool(answer.get("dryRun")) != bool(dry_run):
        raise SubmissionKeyError(
            f"submissionKey names job {job.id}, a different request "
            f"({answer.get('verb')}"
            f"{', dry run' if answer.get('dryRun') else ''}); nothing was "
            "submitted",
            code="submissionKeyConflict", status=409,
            repair_action=(
                "derive the key from this request (each client does), or "
                "leave submissionKey out"))
    answer.setdefault("preflight", resources.get("preflight"))
    # What changes after the first answer is answered as it is now: the
    # job's status, the job carrying it, and a split run's held parts.
    answer["shardsHeld"] = _held_now(jobs, job)
    status = current.status
    if arrived_while_preparing:
        state = "was being prepared when this request arrived"
    elif current.id != job.id:
        state = f"continues as job {current.id}, which is {status}"
    else:
        state = f"is {status}"
    note = (f"Already submitted: this request was submitted before as job "
            f"{job.id}, which {state}. Nothing was submitted again.")
    # Worded for either client: each names its own status and resume verbs.
    if status == "checkpointed":
        note += (f" It stopped at a checkpoint: resume job {current.id} with "
                 "your client's resubmit verb, or cancel it before submitting "
                 "again.")
    elif status not in TERMINAL:
        note += (" Follow it in the server's job list, or cancel it before "
                 "submitting again.")
    held = answer.get("shardsHeld")
    if isinstance(held, dict) and held.get("message"):
        note += " " + held["message"]
    answer.update({"alreadySubmitted": True, "alreadySubmittedNote": note,
                   "currentJobID": current.id, "currentStatus": status})
    return answer


def submit_once(key: str | None, *, jobs, verb: str, dry_run: bool,
                submit) -> dict:
    """Run ``submit()`` (which returns a ``StudySubmission``) unless a
    submission with ``key`` is being prepared or has not ended; answer with
    the existing job then. ``key`` None is today's path, unchanged."""
    if key is None:
        return submit().to_dict()
    while True:
        with _GUARD:
            waiting_on = _PREPARING.get(key)
            if waiting_on is None:
                live = live_submission(jobs, key)
                if live is not None:
                    return _already_submitted(
                        jobs, *live, verb=verb, dry_run=dry_run,
                        arrived_while_preparing=False)
                mine = _Preparing()
                _PREPARING[key] = mine
        if waiting_on is not None:
            # Two requests with one key: the second waits for the first, and
            # is answered with the job the first created — even one that is
            # terminal at birth (a dry run's "prepared"). A first that created
            # nothing (refused, or failed before a job existed) leaves the
            # second to submit for itself.
            waiting_on.done.wait()
            if waiting_on.job_id is not None:
                job = jobs.get(waiting_on.job_id)
                if job is not None:
                    return _already_submitted(
                        jobs, job, _chain_tail(jobs, job), verb=verb,
                        dry_run=dry_run, arrived_while_preparing=True)
            continue
        token = CURRENT_KEY.set(key)
        try:
            submission = submit()
            mine.job_id = submission.job_id
            answer = submission.to_dict()
            _stamp(jobs, answer, key)
            return answer
        finally:
            CURRENT_KEY.reset(token)
            with _GUARD:
                if _PREPARING.get(key) is mine:
                    del _PREPARING[key]
            mine.done.set()
