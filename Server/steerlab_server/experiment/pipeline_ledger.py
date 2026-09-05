"""Durable pipeline ledger encoding shared by execution and reconciliation."""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone

PIPELINE_LEDGER_SCHEMA = 2

def pipeline_ledger_path(run_directory: str) -> str:
    return os.path.join(run_directory, "pipeline.json")



def write_pipeline_ledger(run_directory: str, ledger: dict) -> None:
    """Atomic (tmp + rename): the ledger is what a Slurm requeue resumes
    from, so it must always be canonical-or-previous, never torn. Every
    write stamps ``updatedAt`` — a chain with no terminal disposition is
    "unfinished", and the timestamp is what lets a reader judge whether it
    is plausibly still running or was abandoned (sixth round)."""
    ledger["updatedAt"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    path = pipeline_ledger_path(run_directory)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)



def read_pipeline_ledger(run_directory: str) -> dict:
    with open(pipeline_ledger_path(run_directory), encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("schema") != PIPELINE_LEDGER_SCHEMA:
        found = data.get("schema") if isinstance(data, dict) else None
        raise ValueError(
            f"'{run_directory}' is not a resumable pipeline run: ledger "
            f"schema {found!r} != {PIPELINE_LEDGER_SCHEMA} (or malformed). "
            "A schema-1 ledger predates the promote-pin contract — resuming "
            "it would fall back to ambient catalog resolution; start a "
            "fresh pipeline instead")
    return data

