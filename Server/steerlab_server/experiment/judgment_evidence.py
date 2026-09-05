"""Read and verify sweep/evaluate judgment completion evidence.

A matching but invalid completion remains a refusal, never an absent result.
"""
from __future__ import annotations
import hashlib
import json
import os
from . import paths

JUDGMENT_MARKER_SCHEMA = 1

def verify_judgment_marker(run_dir: str, marker: dict, *, name: str,
                            sweep_jm: dict | None) -> None:
    """Strict verification of a completion record (engineer review
    2026-07-18, third pass): a marker is CANONICAL — it suppresses
    "awaiting" and projects conditions into the manifest — so a bare or
    altered one must never be trusted. Schema-versioned, bound to the
    experiment identity and the sweep's packet pin, and hash-linked to the
    run's own judgment artifacts. Raises ValueError naming the run."""
    run = os.path.basename(run_dir)

    def _refuse(why: str):
        raise ValueError(
            f"completion record in '{run}' failed verification ({why}) — "
            "an unverified judgment run is never canonical; inspect it by "
            "hand (a genuine one re-heals by re-running completion)")

    if marker.get("schema") != JUDGMENT_MARKER_SCHEMA:
        _refuse(f"schema {marker.get('schema')!r}, expected "
                f"{JUDGMENT_MARKER_SCHEMA}")
    if marker.get("experiment") != name:
        _refuse(f"experiment {marker.get('experiment')!r}, expected {name!r}")
    # FAIL CLOSED on missing sweep evidence (engineer review 2026-07-18,
    # fourth pass): no readable judging manifest means no pin to verify
    # against — an unverifiable record is never canonical.
    if sweep_jm is None:
        _refuse("the sweep run's judging manifest cannot be read — no "
                "packet pin to verify against")
    if marker.get("packetsSha256") != sweep_jm.get("packetsSha256"):
        _refuse("packet pin does not match the sweep's judging manifest")
    if marker.get("experimentHashAtJudgment") != sweep_jm.get("experimentHash"):
        _refuse("judgment epoch does not match the sweep's experiment hash")
    for artifact, key in (("judgments.jsonl", "judgmentsSha256"),
                          ("recommendations.json", "recommendationsSha256")):
        stamped = marker.get(key)
        if not stamped:
            _refuse(f"no {key} stamp")
        path = os.path.join(run_dir, artifact)
        try:
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            _refuse(f"{artifact} is missing")
        if digest != stamped:
            _refuse(f"{artifact} does not hash to its stamp")



def sweep_judging_manifest(runs_root: str, sweep_run: str) -> dict | None:
    """``sweep_run``'s judging manifest, or None when unreadable — in which
    case the marker verifier REFUSES (fail closed): a completion record
    with no recoverable sweep evidence is never canonical."""
    try:
        with open(os.path.join(runs_root, sweep_run,
                               "judging-manifest.json"),
                  encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None



def find_judgment_run(name: str, sweep_run: str,
                       root: str | None) -> tuple[str, dict] | None:
    """``(run_dir, VERIFIED marker)`` of the completed judgment run for
    ``sweep_run``, else None. A candidate that fails verification RAISES —
    silently ignoring a corrupt canonical record would re-run judging on
    top of it."""
    runs_root = paths.runs_directory(root)
    try:
        entries = sorted(os.listdir(runs_root))
    except OSError:
        return None
    for entry in entries:
        src = os.path.join(runs_root, entry, "judgment-source.json")
        if not os.path.exists(src):
            continue
        try:
            with open(src, encoding="utf-8") as handle:
                marker = json.load(handle)
        except (OSError, ValueError):
            # An unreadable marker cannot even say which sweep it refers
            # to: skip it (our own writer is atomic, so this is external
            # damage) rather than letting one damaged run anywhere under
            # runs/ block every unrelated completion (engineer review
            # 2026-07-18, fourth pass). Immutability means we never
            # overwrite it; a re-judge creates a fresh run.
            continue
        if marker.get("sweepRun") != sweep_run:
            continue
        run_dir = os.path.join(runs_root, entry)
        verify_judgment_marker(
            run_dir, marker, name=name,
            sweep_jm=sweep_judging_manifest(runs_root, sweep_run))
        return run_dir, marker
    return None



def evaluate_judging_manifest(runs_root: str,
                               evaluate_run: str) -> dict | None:
    """``evaluate_run``'s judging manifest, or None when unreadable — the
    verifier then REFUSES (fail closed, same rule as sweeps)."""
    try:
        with open(os.path.join(runs_root, evaluate_run,
                               "judging-manifest.json"),
                  encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(loaded, dict) or loaded.get("kind") != "evaluate":
        return None
    return loaded



def verify_evaluate_marker(run_dir: str, marker: dict, *, name: str,
                            eval_jm: dict | None) -> None:
    """Strict verification of an evaluate-judgment completion record —
    the same discipline as `verify_judgment_marker`: schema-versioned,
    bound to the experiment and the emission's packet pin + epoch, and
    hash-linked to the run's own artifacts. Raises ValueError."""
    run = os.path.basename(run_dir)

    def _refuse(why: str):
        raise ValueError(
            f"evaluate completion record in '{run}' failed verification "
            f"({why}) — an unverified judgment run is never canonical; "
            "inspect it by hand (a genuine one re-heals by re-running "
            "completion)")

    if marker.get("schema") != JUDGMENT_MARKER_SCHEMA:
        _refuse(f"schema {marker.get('schema')!r}, expected "
                f"{JUDGMENT_MARKER_SCHEMA}")
    if marker.get("kind") != "evaluate":
        _refuse(f"kind {marker.get('kind')!r}, expected 'evaluate'")
    if marker.get("experiment") != name:
        _refuse(f"experiment {marker.get('experiment')!r}, expected {name!r}")
    if eval_jm is None:
        _refuse("the evaluate run's judging manifest cannot be read — no "
                "packet pin to verify against")
    if marker.get("packetsSha256") != eval_jm.get("packetsSha256"):
        _refuse("packet pin does not match the evaluate run's judging "
                "manifest")
    if marker.get("experimentHashAtJudgment") != eval_jm.get("experimentHash"):
        _refuse("judgment epoch does not match the evaluate run's "
                "experiment hash")
    for artifact, key in (("judgments.jsonl", "judgmentsSha256"),
                          ("judge-report.json", "judgeReportSha256")):
        stamped = marker.get(key)
        if not stamped:
            _refuse(f"no {key} stamp")
        path = os.path.join(run_dir, artifact)
        try:
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            _refuse(f"{artifact} is missing")
        if digest != stamped:
            _refuse(f"{artifact} does not hash to its stamp")



def find_evaluate_judgment_run(name: str, evaluate_run: str,
                                root: str | None) -> tuple[str, dict] | None:
    """``(run_dir, VERIFIED marker)`` of the completed judgment run for
    ``evaluate_run``, else None — unreadable unrelated markers skip, a
    matching candidate that fails verification RAISES (sweep rule)."""
    runs_root = paths.runs_directory(root)
    try:
        entries = sorted(os.listdir(runs_root))
    except OSError:
        return None
    for entry in entries:
        src = os.path.join(runs_root, entry, "judgment-source.json")
        if not os.path.exists(src):
            continue
        try:
            with open(src, encoding="utf-8") as handle:
                marker = json.load(handle)
        except (OSError, ValueError):
            continue
        if (marker.get("kind") != "evaluate"
                or marker.get("evaluateRun") != evaluate_run):
            continue
        run_dir = os.path.join(runs_root, entry)
        verify_evaluate_marker(
            run_dir, marker, name=name,
            eval_jm=evaluate_judging_manifest(runs_root, evaluate_run))
        return run_dir, marker
    return None
