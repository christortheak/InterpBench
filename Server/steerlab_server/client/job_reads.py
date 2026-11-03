"""``runner status`` and ``runner logs --tail`` (issue #7): reads that return.

Twins of the Mac client's ``remote status`` and ``remote logs --tail``
(``RemoteJobStatusReport.swift``). The runner answers both in fixed fields
(``api/job_status.py``); the envelope carries its document as it arrived, and
the lines printed here are a summary for a person.
"""
from __future__ import annotations

import re
import time

from ..cli_envelope import CLIResult

#: The most lines ``--tail`` asks for; the runner's own cap.
MAX_TAIL_LINES = 1000

#: A job id: letters, digits, ``.``, ``_``, and ``-``, at most 128
#: characters. Twin of ``RemoteJobStatusText.isJobID``.
_JOB_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

LACKS_JOB_READS_CODE = "jobStatusUnsupported"


def _refusal(**kwargs):
    from ..client_cli import ClientRefusal
    return ClientRefusal(**kwargs)


def job_id(value: str, verb: str, common: dict) -> str:
    if not _JOB_ID.match(value or "") or value in (".", ".."):
        raise _refusal(
            code="usage",
            reason=(f"{value!r} is not a job id: a job id is letters, digits, "
                    "'.', '_', and '-' only"),
            repair_action=("steerlab runner jobs --runner <url> lists the "
                           f"runner's job records; run runner {verb} with an "
                           "id from it"),
            payload=dict(common))
    return value


def tail_count(raw: str, common: dict) -> int:
    try:
        value = int(str(raw).strip())
    except ValueError:
        value = 0
    if not 1 <= value <= MAX_TAIL_LINES:
        raise _refusal(
            code="usage",
            reason=(f"--tail expects a whole number from 1 to {MAX_TAIL_LINES}"
                    f" — got {raw!r}"),
            repair_action="--tail 50", payload=dict(common))
    return value


def require(client, flag: str, verb: str, common: dict) -> None:
    """Refuse, with an update repair, when the runner predates the read: its
    route would answer 404, which reads like "no such job"."""
    capabilities = client.capabilities()
    cluster = capabilities.get("cluster") if isinstance(capabilities, dict) else None
    if not (isinstance(cluster, dict) and cluster.get(flag) is True):
        raise _refusal(
            code=LACKS_JOB_READS_CODE, state="refused",
            reason=(f"this runner's engine predates runner {verb}: it cannot "
                    "report a job's scheduler view or log tail"),
            repair_action=("ask whoever operates the runner to update its "
                           "engine; meanwhile `steerlab runner jobs <job-id> "
                           "--runner <url>` reads the job's record"),
            payload=dict(common))


def status(client, invocation, common: dict) -> CLIResult:
    args = invocation.positionals
    if len(args) != 1:
        raise _refusal(code="usage", reason="runner status needs one job id",
                       repair_action="steerlab runner status <job-id> --runner <url>",
                       payload=dict(common))
    asked = job_id(args[0], "status", common)
    require(client, "jobStatus", "status", common)
    document = client.job_status(asked)
    for line in status_lines(document):
        print(line)
    return CLIResult(message=status_message(document),
                     payload={**common, "jobId": asked, "response": document})


def log_tail(client, invocation, common: dict) -> CLIResult:
    args = invocation.positionals
    asked = job_id(args[0], "logs", common)
    if invocation.has("--follow"):
        raise _refusal(
            code="usage",
            reason="--tail reads the last lines and returns; --follow streams "
                   "to the end. Choose one",
            repair_action=f"steerlab runner logs {asked} --tail 50 --runner <url>",
            payload=dict(common))
    lines = tail_count(invocation.one("--tail"), common)
    require(client, "jobLogTail", "logs --tail", common)
    document = client.job_log_tail(asked, lines)
    for line in tail_lines(document):
        print(line)
    shown = document.get("lines") if isinstance(document, dict) else None
    return CLIResult(
        message=(f"last {len(shown or [])} log line(s) of job {asked} "
                 f"[{(document or {}).get('status')}]"),
        payload={**common, "jobId": asked, "tailLines": lines,
                 "response": document})


# --- plain text ------------------------------------------------------------------

def status_message(document: dict) -> str:
    text = f"job {document.get('jobId')} is {document.get('status') or 'unknown'}"
    if document.get("schedulerOutcome"):
        text += f" (ended by the scheduler: {document['schedulerOutcome']})"
    records = _records_phrase(document.get("progress"))
    return text + (f"; {records}" if records else "")


def status_lines(document: dict) -> list[str]:
    head = f"job {document.get('jobId')} [{document.get('status') or 'unknown'}]"
    if document.get("executorJobID"):
        head += f" — scheduler job {document['executorJobID']}"
    lines = [head, *_body(document, "  ")]
    for shard in document.get("shards") or []:
        index = shard.get("shardIndex")
        count = shard.get("shardCount")
        line = (f"  part {index + 1 if isinstance(index, int) else '?'}"
                + (f" of {count}" if count else "")
                + f": job {shard.get('jobId')}")
        if shard.get("found") is False:
            lines.append(f"{line} — {shard.get('detail') or 'not found'}")
            continue
        line += f" [{shard.get('status') or 'unknown'}]"
        if shard.get("executorJobID"):
            line += f" — scheduler job {shard['executorJobID']}"
        if shard.get("shardJob"):
            line += f" (continues job {shard['shardJob']})"
        lines.append(line)
        lines.extend(_body(shard, "    "))
    return lines


def tail_lines(document: dict) -> list[str]:
    lines = list(document.get("lines") or [])
    if document.get("truncated"):
        lines.insert(0, "(earlier controller lines not shown)")
    output = document.get("schedulerOutput")
    if isinstance(output, dict):
        if output.get("available") is False:
            lines.append(f"--- scheduler output: "
                         f"{output.get('detail') or 'not available'} ---")
        for key in ("stdout", "stderr"):
            entry = output.get(key)
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or "scheduler output"
            if not entry.get("exists"):
                lines.append(f"--- {name}: {entry.get('detail') or 'does not exist'} ---")
                continue
            shown = entry.get("lines") or []
            heading = f"--- {name}: last {len(shown)} line(s)"
            if entry.get("truncated"):
                heading += ", earlier lines not shown"
            lines.append(heading + " ---")
            lines.extend(shown)
            if entry.get("detail"):
                lines.append(f"({entry['detail']})")
    lines.extend(f"note: {note}" for note in document.get("notes") or [])
    return lines


def _body(entry: dict, indent: str) -> list[str]:
    lines = []
    view = entry.get("scheduler")
    if isinstance(view, dict):
        lines.append(f"{indent}scheduler: {_scheduler_phrase(view)}")
    if entry.get("schedulerOutcome"):
        lines.append(f"{indent}ended by the scheduler: {entry['schedulerOutcome']}")
    progress = entry.get("progress") or {}
    records = _records_phrase(progress)
    if records:
        if progress.get("lastRecordAt") is not None:
            records += f"; last written {_stamp(progress['lastRecordAt'])}"
        lines.append(f"{indent}records: {records}")
    elif progress.get("detail"):
        lines.append(f"{indent}records: {progress['detail']}")
    if entry.get("deathDetail"):
        lines.append(f"{indent}{entry['deathDetail']}")
    if entry.get("continuedAs"):
        lines.append(f"{indent}continued as job {entry['continuedAs']}")
    for note in entry.get("notes") or []:
        if not str(note).startswith("this job's run continues as"):
            lines.append(f"{indent}note: {note}")
    return lines


def _scheduler_phrase(view: dict) -> str:
    if view.get("queryOk") is False:
        return "did not answer" + (f" ({view['detail']})" if view.get("detail") else "")
    raw = view.get("rawState")
    if not raw:
        return view.get("detail") or "has no record of this job"
    text = str(raw)
    if view.get("reason"):
        text += f" ({view['reason']})"
    if view.get("expectedStartAt") is not None:
        text += f"; expected to start {_stamp(view['expectedStartAt'])}"
    if view.get("nodeList"):
        text += f" on {view['nodeList']}"
    if view.get("startedAt") is not None:
        text += f"; started {_stamp(view['startedAt'])}"
    if view.get("elapsed"):
        text += f"; elapsed {view['elapsed']}"
        if view.get("timeLimit"):
            text += f" of {view['timeLimit']}"
    elif view.get("timeLimit"):
        text += f"; time limit {view['timeLimit']}"
    remaining = view.get("walltimeRemainingSeconds")
    if isinstance(remaining, (int, float)):
        text += f" ({_duration(remaining)} left)"
    if view.get("endedAt") is not None:
        text += f"; ended {_stamp(view['endedAt'])}"
        if view.get("exitCode"):
            text += f"; exit {view['exitCode']}"
    return text


def _records_phrase(progress) -> str | None:
    if not isinstance(progress, dict):
        return None
    written = progress.get("recordsWritten")
    if not isinstance(written, int):
        return None
    unit = "turn" if progress.get("unit") == "turn" else "record"
    expected = progress.get("recordsExpected")
    if isinstance(expected, int):
        text = f"{written} of {expected} {unit}s written"
        if progress.get("recordsExpectedSource") == "preflightEstimate":
            text += " (the total is the submission's planning estimate)"
    else:
        text = f"{written} {unit}{'' if written == 1 else 's'} written"
    if progress.get("tornTail"):
        text += ", plus one incomplete line that is not counted"
    return text


def _stamp(epoch) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(float(epoch)))
    except (TypeError, ValueError, OverflowError):
        return str(epoch)


def _duration(seconds) -> str:
    total = max(0, int(round(float(seconds))))
    hours, minutes = total // 3600, (total % 3600) // 60
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m"
    return f"{total}s"
