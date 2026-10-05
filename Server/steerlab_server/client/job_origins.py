"""Remote job origins recorded in the workspace.

When this client submits a job to a runner, it writes where the job went —
the server, its serving root, and the workspace it came from — into the
workspace itself, so the Mac app can import the job's evidence or act on it
without the researcher reconnecting by job ID. The Mac command line writes the
same record (``Sources/ExperimentKit/WorkspaceJobOrigins.swift``); the layout
is shared::

    <workspace>/.steerlab/job-origins/origins.json
    {"schemaVersion": 1, "jobs": {"<job id>": [<origin>, ...]}}

A job ID maps to a LIST because job IDs are unique only per server: two
servers can hand out the same one, and a reader must then refuse rather than
guess. A writer replaces only the row for the same job on the same server and
carries every other row through untouched, including rows a newer client
wrote. Only references and paths are recorded — never a token.

Writers hold an advisory ``flock`` on ``origins.lock`` for the whole
read-modify-write and publish with an atomic rename, so concurrent writers in
different processes (this client, the Mac command line, two coding
assistants) never leave a torn or half-merged file.

Standard library only: this module is imported by the runner verbs, inside
the client's light-install import set.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from urllib.parse import urlsplit

try:  # POSIX advisory locks; absent on Windows, where the rename still holds
    import fcntl
except ImportError:  # pragma: no cover - exercised only off POSIX
    fcntl = None  # type: ignore[assignment]

SCHEMA_VERSION = 1

#: The folder, relative to the workspace root. Its own ``.gitignore`` keeps
#: the record out of a workspace's git history: it names servers and paths
#: that mean something only on this machine.
RELATIVE_DIRECTORY = (".steerlab", "job-origins")
ORIGINS_FILE = "origins.json"
LOCK_FILE = "origins.lock"

#: ``submittedBy`` for this client, and for the Mac command line.
PYTHON_CLIENT = "steerlab"
MAC_COMMAND_LINE = "steerlab-cli"


def directory(workspace_root: str) -> str:
    return os.path.join(workspace_root, *RELATIVE_DIRECTORY)


def file_path(workspace_root: str) -> str:
    return os.path.join(directory(workspace_root), ORIGINS_FILE)


def server_identity(base_url: str) -> str:
    """The durable identity of a server reached by URL.

    The same normalization the Mac app's registry applies to a direct server
    (``ClusterConnectionStore.normalizedEndpointKey``): lowercased
    ``scheme://host:port`` with the scheme's default port filled in, so a
    record this client writes matches the server the app is connected to.
    """
    trimmed = (base_url or "").strip()
    candidate = trimmed if "://" in trimmed else f"http://{trimmed}"
    try:
        parts = urlsplit(candidate)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return trimmed.lower()
    if not host:
        return trimmed.lower()
    scheme = (parts.scheme or "http").lower()
    if port is None:
        port = 443 if scheme == "https" else 80
    return f"{scheme}://{host}:{port}"


def timestamp(now: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ",
                         time.gmtime(time.time() if now is None else now))


def load(workspace_root: str) -> dict:
    """Every origin by job ID. An absent or unreadable file is an empty
    record, never an error."""
    try:
        with open(file_path(workspace_root), encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return {}
    jobs = document.get("jobs") if isinstance(document, dict) else None
    if not isinstance(jobs, dict):
        return {}
    return {job_id: [row for row in rows if isinstance(row, dict)
                     and row.get("serverIdentity")]
            for job_id, rows in jobs.items() if isinstance(rows, list)}


def origins_for(workspace_root: str, job_id: str) -> list:
    return load(workspace_root).get(job_id, [])


def record(workspace_root: str, *, job_id: str, endpoint: str,
           submitted_by: str = PYTHON_CLIENT,
           identity: str | None = None, site_id: str | None = None,
           serving_root: str | None = None, experiment: str | None = None,
           verb: str | None = None, operation: str | None = None,
           now: float | None = None) -> dict:
    """Record (or refresh) one job's origin and return the row written.

    Raises ``OSError``/``ValueError`` on failure; callers treat a failure as a
    warning, because the job exists whatever happens here.
    """
    job_id = str(job_id or "").strip()
    if not job_id:
        raise ValueError("a job origin needs a job ID")
    row = {
        "serverIdentity": identity or server_identity(endpoint),
        "endpoint": endpoint,
        "siteID": site_id,
        "servingRoot": serving_root,
        "workspaceRoot": os.path.realpath(workspace_root),
        "submittedBy": submitted_by,
        "recordedAt": timestamp(now),
        "experiment": experiment,
        "verb": verb,
        "operation": operation,
    }
    # Absent rather than null, as the Mac writer encodes an optional.
    row = {key: value for key, value in row.items() if value is not None}

    folder = directory(workspace_root)
    os.makedirs(folder, exist_ok=True)
    _write_ignore_rule(folder)
    lock = os.open(os.path.join(folder, LOCK_FILE), os.O_CREAT | os.O_RDWR,
                   0o600)
    try:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            _merge_and_publish(workspace_root, job_id, row)
        finally:
            if fcntl is not None:
                fcntl.flock(lock, fcntl.LOCK_UN)
    finally:
        os.close(lock)
    return row


def _write_ignore_rule(folder: str) -> None:
    try:
        descriptor = os.open(os.path.join(folder, ".gitignore"),
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return
    try:
        os.write(descriptor, b"*\n")
    finally:
        os.close(descriptor)


def _merge_and_publish(workspace_root: str, job_id: str, row: dict) -> None:
    path = file_path(workspace_root)
    document: dict = {}
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError:
        text = None
    if text is not None:
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("jobs", {}), dict):
            document = parsed
        else:
            # Never destroy what cannot be read: set it aside and start a
            # fresh record beside it.
            os.replace(path, f"{path}.unreadable-{int(time.time())}")
    jobs = document.get("jobs") if isinstance(document.get("jobs"), dict) else {}
    rows = jobs.get(job_id) if isinstance(jobs.get(job_id), list) else []
    rows = [existing for existing in rows
            if not (isinstance(existing, dict)
                    and existing.get("serverIdentity") == row["serverIdentity"])]
    rows.append(row)
    jobs[job_id] = rows
    document["jobs"] = jobs
    existing_version = document.get("schemaVersion")
    document["schemaVersion"] = max(
        existing_version if isinstance(existing_version, int) else 0,
        SCHEMA_VERSION)
    folder = os.path.dirname(path)
    handle, temp_path = tempfile.mkstemp(prefix=".origins-", suffix=".json",
                                         dir=folder)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def record_quietly(workspace_root: str | None, *, job_id, endpoint: str,
                   warn, **fields) -> dict | None:
    """``record``, for a verb that has already submitted: a failure becomes
    one warning line through ``warn`` and the verb carries on. No workspace
    named means nothing to record."""
    if not workspace_root or not job_id:
        return None
    try:
        return record(workspace_root, job_id=str(job_id), endpoint=endpoint,
                      **fields)
    except (OSError, ValueError) as exc:
        warn(f"warning: job {job_id} was submitted, but its origin could not "
             f"be recorded in this workspace ({exc}) — the Mac app will ask "
             "you to reconnect to it by job ID before importing its "
             "evidence\n")
        return None
