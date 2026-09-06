"""Conservative process ownership for controller recovery, outside study data.

A hostname and PID prove death only on this host, when the kernel reports
that PID absent. Foreign hosts, legacy records, permissions and PID reuse
cannot establish death and must never trigger destructive startup recovery.
"""
from __future__ import annotations

import os
import socket
import sqlite3


def current_owner() -> tuple[str, int]:
    return socket.gethostname(), os.getpid()


def owner_has_exited(host: str, pid: int) -> bool:
    if host != socket.gethostname() or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def initialize(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_owners (
            job_id TEXT PRIMARY KEY,
            host TEXT NOT NULL,
            pid INTEGER NOT NULL
        )
    """)


def record(conn: sqlite3.Connection, job_id: str) -> None:
    conn.execute("INSERT OR REPLACE INTO job_owners VALUES (?, ?, ?)",
                 (job_id, *current_owner()))


def claim_exited(conn: sqlite3.Connection, job_id: str) -> bool:
    """Caller holds an IMMEDIATE transaction through claim and job snapshot.

    Transferring ownership serializes competing recoverers. A recoverer that
    crashes can itself be recovered; no wall-clock lease expiry is used as
    evidence that a slow or disconnected process has died.
    """
    owner = conn.execute("SELECT host, pid FROM job_owners WHERE job_id = ?",
                         (job_id,)).fetchone()
    if owner is None or not owner_has_exited(owner[0], owner[1]):
        return False
    record(conn, job_id)
    return True
