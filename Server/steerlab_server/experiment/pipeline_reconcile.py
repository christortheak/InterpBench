"""Orphaned-pipeline detection and the ``parked`` stamp (2026-08-06).

The incident this closes (the 2026-08-06 replication run): the Slurm
controller daemon restarted between a chain's ``run`` and ``analyze`` stages.
The ledger showed a completed run stage with a null disposition, the new
daemon had no job
following the chain, no ``FAILED.md`` was written, nothing was packaged —
the completed results were invisible in every UI and were recovered by hand.

The rule this module carries: **a pipeline ledger with completed stages, no
terminal disposition, and no live job is never left silent.** At JobManager
startup the ledger is either resumed in-process (when the remainder needs no
model load — the daemon can finish promote/analyze/adopt-judgment tails
itself) or stamped ``parked`` with a reason the app can show, from which the
researcher can resubmit or import the finished stage runs.

Stdlib-only (like :mod:`resume`) so the jobs layer imports it without the
torch tax; the resume ACTION lives with the caller, which imports
:mod:`tasks` lazily inside the worker thread.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

#: Ledger schema this module can hand to the resume machinery (matches
#: ``tasks.PIPELINE_LEDGER_SCHEMA``; duplicated as a literal to keep this
#: module import-light — the cross-module agreement is pinned by test).
RESUMABLE_LEDGER_SCHEMA = 2


@dataclass
class OrphanedPipeline:
    """One ledger the startup reconcile must act on — never skip."""
    run_directory: str
    experiment: str
    schema: object
    stages: list[str] = field(default_factory=list)
    stage_status: dict = field(default_factory=dict)
    remaining: list[str] = field(default_factory=list)

    @property
    def resumable_schema(self) -> bool:
        return self.schema == RESUMABLE_LEDGER_SCHEMA


def _read_ledger(run_directory: str) -> dict | None:
    path = os.path.join(run_directory, "pipeline.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def scan(runs_root: str, live_directories: set[str]) -> list[OrphanedPipeline]:
    """Every orphaned pipeline ledger under ``runs_root``: completed
    stage(s), null disposition, not already parked, and not followed by any
    live job (``live_directories`` — realpaths a non-terminal job record or
    resume pointer references; the caller assembles it, biased WIDE: an
    extra live path only means an extra skip, never a wrong park).

    Tolerant like ``list_pipeline_runs``, deliberately: an unreadable or
    schema-foreign ledger is still RETURNED (with its schema as found) so
    the caller can park it loudly — silence is the failure mode this module
    exists to remove."""
    try:
        entries = sorted(os.listdir(runs_root), reverse=True)
    except OSError:
        return []
    orphans: list[OrphanedPipeline] = []
    for dirname in entries:
        run_directory = os.path.join(runs_root, dirname)
        ledger = _read_ledger(run_directory)
        if ledger is None:
            continue
        if ledger.get("disposition"):
            continue  # terminal: completed or aborted — nothing orphaned
        if isinstance(ledger.get("parked"), dict):
            continue  # already loud — the app shows it; nothing silent here
        stage_results = ledger.get("stageResults") or {}
        completed = [s for s, e in stage_results.items()
                     if isinstance(e, dict) and e.get("status") == "completed"]
        if not completed:
            # A chain that never finished a stage has nothing to import and
            # nothing to resume-skip; the ordinary job-failure surfaces
            # (partial evidence, FAILED.md) own that case.
            continue
        if os.path.realpath(run_directory) in live_directories:
            continue
        stages = [s for s in (ledger.get("stages") or []) if isinstance(s, str)]
        status = {s: ((stage_results.get(s) or {}).get("status") or "pending")
                  for s in stages}
        orphans.append(OrphanedPipeline(
            run_directory=run_directory,
            experiment=str(ledger.get("experiment") or ""),
            schema=ledger.get("schema"),
            stages=stages,
            stage_status=status,
            remaining=[s for s in stages if status.get(s) != "completed"]))
    return orphans


def park(run_directory: str, *, reason: str,
         by: str = "startup-reconcile") -> dict | None:
    """Stamp the ledger ``parked`` — loud, durable, and additive (the
    disposition stays null: parked is a state of being unfinished, not a
    terminal outcome, so ``package_evidence``'s completed-chain refusal and
    every disposition reader keep their meaning). Atomic tmp+rename with an
    ``updatedAt`` stamp, mirroring ``tasks._write_pipeline_ledger``. The
    next successful resume clears the stamp. Returns the parked block, or
    None when the ledger cannot be read (the caller logs — a ledger too
    broken to stamp must at least be named in the daemon log)."""
    ledger = _read_ledger(run_directory)
    if ledger is None:
        return None
    stage_results = ledger.get("stageResults") or {}
    stages = [s for s in (ledger.get("stages") or []) if isinstance(s, str)]
    parked = {
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "by": by,
        "reason": reason,
        "completedStages": [
            s for s in stages
            if (stage_results.get(s) or {}).get("status") == "completed"],
        "remainingStages": [
            s for s in stages
            if (stage_results.get(s) or {}).get("status") != "completed"],
    }
    ledger["parked"] = parked
    ledger["updatedAt"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    path = os.path.join(run_directory, "pipeline.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return parked
