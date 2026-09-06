"""Controller ownership and recovery evidence, outside scientific artifacts.

Only an absent local PID or the exact controller allocation's terminal
accounting record proves exit. Missing/ambiguous accounting is uncertainty.
Explicit recovery records an operator's assertion, never disguising it as proof.
"""
from __future__ import annotations

from functools import lru_cache
import json
import os
import re
import socket
import sqlite3
import time
import uuid

TERMINAL_ALLOCATION_STATES = frozenset({
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "TIMEOUT",
})


def current_owner() -> tuple[str, int]:
    return socket.gethostname(), os.getpid()


def process_state(host: str, pid: int) -> str:
    if host != socket.gethostname() or pid <= 0:
        return "unknown"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "exited"
    except OSError:
        return "unknown"
    return "live"


def owner_has_exited(host: str, pid: int) -> bool:
    return process_state(host, pid) == "exited"


def allocation_rows(job_id: str, cluster: str) -> list[dict]:
    """Read allocation records, including reused IDs, through configured wrappers.

    DBIndex + submit/start distinguish allocations and requeues. UTC and a fixed
    time format make the identity independent of the observing node's locale.
    This function MUST be called outside a database write transaction.
    """
    from .executors import scheduler_poll_commands, scheduler_run
    if not re.fullmatch(r"[0-9]+", job_id) or (cluster and not re.fullmatch(
            r"[A-Za-z0-9_-]+", cluster)):
        return []
    command = [scheduler_poll_commands()[0], "-n", "-X", "-P", "--duplicates",
               "-j", job_id, "-o", "DBIndex,JobIDRaw,Cluster,Submit,Start,State%64"]
    if cluster:
        command += ["-M", cluster]
    try:
        result = scheduler_run(command, capture_output=True, text=True,
                               env=dict(os.environ, TZ="UTC",
                                        SLURM_TIME_FORMAT="standard"))
    except OSError:
        return []
    if result.returncode:
        return []
    rows = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split("|")]
        if len(fields) != 6:
            continue
        index, raw_id, actual_cluster, submitted, started, state = fields
        if (raw_id != job_id or not index.isdigit() or int(index) <= 0
                or not actual_cluster or (cluster and actual_cluster != cluster)
                or not all(re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", t)
                           for t in (submitted, started)) or not state):
            continue
        rows.append(dict(dbIndex=index, jobID=raw_id, cluster=actual_cluster,
                         submitted=submitted, started=started, state=state.split()[0]))
    return rows


@lru_cache(maxsize=8)
def capture_allocation(pid: int, job_id: str, cluster: str) -> dict | None:
    """Capture this process's controller allocation once, before storing jobs.

    An unavailable record leaves ownership incomplete and requires later
    operator review. It is never filled in from a potentially reused ID later.
    """
    if not job_id:
        return None
    rows = [row for row in allocation_rows(job_id, cluster) if row["state"] == "RUNNING"]
    if len(rows) != 1:
        return None
    return {key: value for key, value in rows[0].items() if key != "state"}


def current_allocation() -> dict | None:
    return capture_allocation(os.getpid(), os.environ.get("SLURM_JOB_ID", ""),
                              os.environ.get("SLURM_CLUSTER_NAME", ""))


def owner_state(owner: dict | None, allocation_states: dict | None = None) -> str:
    if owner is None:
        return "unknown"
    local = process_state(owner["host"], owner["pid"])
    if local != "unknown":
        return local
    allocation = json.loads(owner["allocation_json"]) if owner["allocation_json"] else None
    if not allocation:
        return "unknown"
    key = json.dumps(allocation, sort_keys=True)
    if allocation_states is not None and key in allocation_states:
        return allocation_states[key]
    matches = [row for row in allocation_rows(allocation["jobID"], allocation["cluster"])
               if all(row.get(field) == value for field, value in allocation.items())]
    result = "unknown"
    if len(matches) == 1:
        state = matches[0]["state"]
        if state in TERMINAL_ALLOCATION_STATES:
            result = "exited"
        elif state in {"RUNNING", "SUSPENDED", "CONFIGURING", "COMPLETING"}:
            result = "live"
    if allocation_states is not None:
        allocation_states[key] = result
    return result


def initialize(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS job_owners (
        job_id TEXT PRIMARY KEY, host TEXT NOT NULL, pid INTEGER NOT NULL,
        allocation_json TEXT, instance TEXT)""")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(job_owners)")}
    for column in ("allocation_json", "instance"):
        if column not in columns:
            conn.execute(f"ALTER TABLE job_owners ADD COLUMN {column} TEXT")
    conn.execute("""CREATE TABLE IF NOT EXISTS job_recoveries (
        seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
        timestamp REAL NOT NULL, actor_host TEXT NOT NULL, actor_pid INTEGER NOT NULL,
        previous_owner_json TEXT NOT NULL, evidence TEXT NOT NULL,
        reason TEXT NOT NULL, review_token TEXT NOT NULL)""")


def read(conn: sqlite3.Connection, job_id: str) -> dict | None:
    row = conn.execute("SELECT job_id, host, pid, allocation_json, instance "
                       "FROM job_owners WHERE job_id = ?", (job_id,)).fetchone()
    return dict(zip(("job_id", "host", "pid", "allocation_json", "instance"), row)) if row else None


def record(conn: sqlite3.Connection, job_id: str, allocation: dict | None = None) -> None:
    conn.execute("INSERT OR REPLACE INTO job_owners "
                 "(job_id, host, pid, allocation_json, instance) VALUES (?, ?, ?, ?, ?)",
                 (job_id, *current_owner(), json.dumps(allocation) if allocation else None,
                  str(uuid.uuid4())))


def audit(conn: sqlite3.Connection, job_id: str, owner: dict | None,
          evidence: str, reason: str, token: str) -> None:
    conn.execute("INSERT INTO job_recoveries (job_id, timestamp, actor_host, actor_pid, "
                 "previous_owner_json, evidence, reason, review_token) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                 (job_id, time.time(), *current_owner(), json.dumps(owner, sort_keys=True),
                  evidence, reason, token))
