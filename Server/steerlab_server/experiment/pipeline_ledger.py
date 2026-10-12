"""Durable pipeline ledger encoding shared by execution and reconciliation.

A ledger names other runs: each stage's run directory, each promoted agent's
file, and the run an abort cites as its evidence. It names them relative to
the folder the chain's own directory sits in, spelled as the workspace spells
it (``runs/<run ID>``, ``runs/<run ID>/<file>``): every run of a chain is a
sibling there, on the machine that ran it and in the workspace an evidence
bundle is imported into, whatever that folder is called on disk or however it
is reached (a symlinked ``runs``, a site's own run root). The ledger is a
member of the chain's run directory, so it travels in every evidence bundle,
and an absolute path there named the account that ran the chain (its home or
scratch folder). Ledgers written before this change hold absolute paths; both
spellings read back as a path on this machine, so those chains still resume.

In memory a ledger always holds absolute paths, because resume opens them:
:func:`read_pipeline_ledger` resolves on the way in and
:func:`write_pipeline_ledger` writes the workspace spelling on the way out.
A reader that loads ``pipeline.json`` itself and opens a reference resolves it
with :func:`resolve_ledger` or :func:`resolve_reference`.
"""
from __future__ import annotations
import copy
import json
import os
from datetime import datetime, timezone

PIPELINE_LEDGER_SCHEMA = 2

def pipeline_ledger_path(run_directory: str) -> str:
    return os.path.join(run_directory, "pipeline.json")


#: How a stored reference names the folder the chain sits in.
_RUNS = "runs"


def _runs_folder(run_directory: str) -> str:
    """The folder holding a chain's run directory, and so all of its runs."""
    return os.path.dirname(os.path.abspath(run_directory))


def workspace_reference(path, run_directory: str):
    """``path`` as a ledger stores it: ``runs/…`` when it lies in the folder
    that holds ``run_directory``, otherwise unchanged (a reference outside it
    cannot be made portable, and packaging names it missing). A relative path
    is a path from the working directory, as everywhere a ledger is used, and
    is made absolute first; anything that is not a path is returned as is."""
    if not isinstance(path, str) or not path:
        return path
    runs = _runs_folder(run_directory)
    # As written first, then resolved: a folder reached through a link (or a
    # caller that resolved the chain's directory) must still find its runs.
    for base in dict.fromkeys((runs, os.path.realpath(runs))):
        for candidate in dict.fromkeys((os.path.abspath(path),
                                        os.path.realpath(path))):
            parts = os.path.relpath(candidate, base).split(os.sep)
            if parts[0] not in (os.curdir, os.pardir) \
                    and os.pardir not in parts:
                return "/".join([_RUNS, *parts])
    return path


def resolve_reference(value, run_directory: str):
    """A stored reference as a path on this machine: ``runs/…`` is joined to
    the folder that holds ``run_directory``; an absolute path (a ledger
    written before references were portable) is returned as is, and so is any
    other relative path, which such a ledger meant from the working
    directory."""
    if not isinstance(value, str) or not value or os.path.isabs(value):
        return value
    parts = value.split("/")
    if len(parts) < 2 or parts[0] != _RUNS:
        return value
    return os.path.join(_runs_folder(run_directory), *parts[1:])


def _map_references(ledger: dict, convert) -> dict:
    """A copy of ``ledger`` with ``convert`` applied to every run reference:
    each stage's ``runDirectory``, each promoted agent's ``path`` (a bare
    string on schema-1 ledgers), and the abort's ``evidenceRunDirectory``."""
    out = copy.deepcopy(ledger)
    stage_results = out.get("stageResults")
    if isinstance(stage_results, dict):
        for entry in stage_results.values():
            if not isinstance(entry, dict):
                continue
            if "runDirectory" in entry:
                entry["runDirectory"] = convert(entry["runDirectory"])
            concepts = entry.get("concepts")
            if isinstance(concepts, dict):
                for concept, pin in concepts.items():
                    if isinstance(pin, dict):
                        if "path" in pin:
                            pin["path"] = convert(pin["path"])
                    else:
                        concepts[concept] = convert(pin)
    abort = out.get("abort")
    if isinstance(abort, dict) and "evidenceRunDirectory" in abort:
        abort["evidenceRunDirectory"] = convert(abort["evidenceRunDirectory"])
    return out


def portable_ledger(ledger: dict, run_directory: str) -> dict:
    """``ledger`` as it is written: every run reference workspace-relative."""
    return _map_references(
        ledger, lambda value: workspace_reference(value, run_directory))


def resolve_ledger(ledger: dict, run_directory: str) -> dict:
    """``ledger`` as it is used: every run reference a path on this machine."""
    return _map_references(
        ledger, lambda value: resolve_reference(value, run_directory))


def write_pipeline_ledger(run_directory: str, ledger: dict) -> None:
    """Atomic (tmp + rename): the ledger is what a Slurm requeue resumes
    from, so it must always be canonical-or-previous, never torn. Every
    write stamps ``updatedAt`` — a chain with no terminal disposition is
    "unfinished", and the timestamp is what lets a reader judge whether it
    is plausibly still running or was abandoned (sixth round). The caller's
    ledger keeps its absolute paths; the file gets the portable spelling."""
    ledger["updatedAt"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    path = pipeline_ledger_path(run_directory)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(portable_ledger(ledger, run_directory), handle, indent=2,
                  sort_keys=True)
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
    return resolve_ledger(data, run_directory)
