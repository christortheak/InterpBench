"""Pipeline continuation identity, promotion custody and listing.

Consumes explicit workspace roots and the existing run-epoch/ledger contracts.
"""
from __future__ import annotations
import json
import os
from . import paths, run_epoch
from .manifest import Manifest
from .pipeline_ledger import write_pipeline_ledger as _write_pipeline_ledger



def _restore_self_pinned_revision(name, manifest, ledger, root, _log,
                                  pipeline_run_directory=None):
    """Repair the pipeline continuation's self-inflicted epoch drift.

    Returns ``(manifest, live_hash, restored)``. The chain's own run stage
    pins the resolved model revision (``_pin_model_revision``) and stamps
    every artifact with the PINNED epoch, but each ``bundle execute``
    re-imports the bundle's unpinned manifest over the live one — so the
    continuation can find a live manifest identical to the ledger's epoch
    except for the missing pin. That is the same manifest, one write behind
    its own machinery: restore the pin (drafts only — the same rule
    ``_pin_model_revision`` applies), persist it, and continue.

    The pin is looked for in the CHAIN's own start snapshot first (the
    pipeline dir's ``experiment.json``, written after the chain's model
    load pinned the revision — it exists for every schema-2 chain), then in
    each completed stage's snapshot. Only an EXACT hash match after
    applying a snapshot revision counts: any other difference is real
    drift and returns ``restored=False`` with the live hash unchanged."""
    if manifest.model_revision:
        return manifest, manifest.content_hash(), False
    candidates = ([pipeline_run_directory] if pipeline_run_directory else [])
    candidates += [
        (entry or {}).get("runDirectory")
        for entry in (ledger.get("stageResults") or {}).values()]
    for run_dir in candidates:
        if not run_dir:
            continue
        snap = run_epoch.snapshot(run_dir)
        revision = getattr(snap, "model_revision", None) if snap else None
        if not revision:
            continue
        raw = dict(manifest.raw)
        raw["modelRevision"] = revision
        try:
            candidate = Manifest.from_dict(raw)
        except Exception:  # noqa: BLE001 - fall through to real-drift handling
            continue
        if candidate.content_hash() == ledger.get("experimentHash"):
            if manifest.status == "draft":
                from . import experiment_store
                experiment_store.pin_model_revision(name, revision, root)
                manifest = Manifest.load(name, root)
            else:
                manifest = candidate
            _log(f"restored model revision pin {revision[:12]}… (the chain "
                 "resolved it at model load; a later bundle import had "
                 "clobbered it) — the live manifest again matches the "
                 "ledger epoch")
            return manifest, manifest.content_hash(), True
    return manifest, manifest.content_hash(), False



def _stamp_pipeline_drift(pipeline_run_directory, ledger, live_hash) -> None:
    """Durable record of a continuation that proceeded past manifest drift
    (never silent): what the ledger expected, what was live, and when.

    Mutates the IN-MEMORY ledger too (merge repair, 2026-08-06): every
    later ``_stage_done`` rewrites the whole ledger from memory, so a stamp
    living only on disk would be clobbered by the first completed stage of
    the very continuation it records."""
    ledger.setdefault("epochDriftAtContinuation", []).append(
        {"ledgerHash": ledger.get("experimentHash"),
         "liveHash": live_hash})
    try:
        _write_pipeline_ledger(pipeline_run_directory, ledger)
    except OSError as exc:
        # The stamp is evidence, not a gate — a failed write must not kill
        # the continuation the policy exists to protect. It IS still logged.
        print(f"warning: could not stamp epochDriftAtContinuation: {exc}")



def list_pipeline_runs(name: str | None = None,
                       root: str | None = None) -> list[dict]:
    """Pipeline runs for ``name`` — or EVERY experiment when ``name`` is
    None (the Compute panel's awaiting-import affordance, 2026-08-06) —
    newest first. Tolerant where RESUME is strict: a schema-1 ledger is
    listable (display is not resume) and unreadable dirs are skipped. Each
    row: run id, experiment, schema, disposition (``completed`` |
    ``aborted`` | null = in flight/awaiting), an ordered stage summary,
    the abort record when present, the promoted-agent pins, and — when
    stamped — the ``parked`` block and ``epochDriftAtContinuation``."""
    runs_root = paths.runs_directory(root)
    try:
        entries = sorted(os.listdir(runs_root), reverse=True)
    except OSError:
        return []
    out: list[dict] = []
    for dirname in entries:
        ledger_path = os.path.join(runs_root, dirname, "pipeline.json")
        if not os.path.isfile(ledger_path):
            continue
        try:
            with open(ledger_path, encoding="utf-8") as handle:
                ledger = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(ledger, dict):
            continue
        if name is not None and ledger.get("experiment") != name:
            continue
        stage_results = ledger.get("stageResults") or {}
        stages = []
        for stage in ledger.get("stages") or []:
            entry = stage_results.get(stage) or {}
            stage_run = entry.get("runDirectory")
            stages.append({
                "stage": stage,
                "status": entry.get("status") or "pending",
                "runID": (os.path.basename(stage_run)
                          if isinstance(stage_run, str) and stage_run
                          else None)})
        row: dict = {"run": dirname, "schema": ledger.get("schema"),
                     "experiment": ledger.get("experiment"),
                     "disposition": ledger.get("disposition"),
                     "experimentHash": ledger.get("experimentHash"),
                     "updatedAt": ledger.get("updatedAt"),
                     "manifestStatus": ledger.get("manifestStatus"),
                     "stages": stages}
        if isinstance(ledger.get("parked"), dict):
            # The startup reconciler's orphan stamp (2026-08-06): a chain
            # with completed stages, no terminal disposition, and no live
            # job. The row keeps disposition null — parked is a STATE OF
            # BEING UNFINISHED, not a terminal outcome — and the app offers
            # resume/import from this block.
            row["parked"] = ledger["parked"]
        if isinstance(ledger.get("epochDriftAtContinuation"), list) \
                and ledger["epochDriftAtContinuation"]:
            row["epochDriftAtContinuation"] = \
                ledger["epochDriftAtContinuation"]
        abort = ledger.get("abort")
        if isinstance(abort, dict):
            evidence = abort.get("evidenceRunDirectory")
            row["abort"] = {
                "stage": abort.get("stage"),
                "gates": abort.get("gates") or [],
                "evidenceRunID": (os.path.basename(evidence)
                                  if isinstance(evidence, str) and evidence
                                  else None)}
        concepts = (stage_results.get("promote") or {}).get("concepts") or {}
        if concepts:
            agents: dict = {}
            for concept, pin in concepts.items():
                if isinstance(pin, dict):
                    agents[concept] = {
                        "artifact": os.path.basename(str(pin.get("path")
                                                         or "")),
                        "hash": pin.get("hash"),
                        "sweepRun": pin.get("sweepRun"),
                        "winningCell": pin.get("winningCell")}
                else:  # legacy schema-1 bare path
                    agents[concept] = {"artifact": os.path.basename(str(pin))}
            row["promotedAgents"] = agents
        out.append(row)
    return out



def _expected_promotion_identity(name: str, concept: str, root: str | None,
                                 *, chain_sweep_run: str | None = None):
    """The SELECTION identity a criterion promotion of this concept would
    stamp right now: ``(sweepRun, winningCell)`` from the same evidence
    ``promote`` reads — the manifest's ``<concept>-recommended`` condition,
    else the newest sweep run's recommendations entry. When
    ``chain_sweep_run`` names a chain's own sweep stage, the evidence must
    come from that exact sweep run (a frozen manifest keeps one hash across
    many sweeps, so the hash alone cannot distinguish this chain's
    selection from an earlier one). Returns None when no unambiguous
    identity exists."""
    from . import promote as promote_lib
    manifest = Manifest.load(name, root)
    selection = None
    recommended = next(
        (c for c in manifest.raw.get("conditions", [])
         if c.get("name") == f"{concept}-recommended"
         and isinstance(c.get("selection"), dict)), None)
    if recommended is not None:
        selection = recommended["selection"]
    else:
        evidence = promote_lib._newest_sweep_evidence(name, concept, root)
        if evidence is not None and isinstance(evidence[1], dict):
            selection = evidence[1]
    if not isinstance(selection, dict):
        return None
    sweep_run = selection.get("sweepRun")
    cell = selection.get("winningCell")
    if not sweep_run or not isinstance(cell, dict):
        return None
    if chain_sweep_run and chain_sweep_run != sweep_run:
        return None  # the evidence is not this chain's sweep — never adopt
    return str(sweep_run), {"layer": int(cell.get("layer")),
                            "alpha": float(cell.get("alpha"))}



def _find_minted_agent(name: str, concept: str, ledger: dict,
                       root: str | None) -> str | None:
    """Crash-recovery for the promote stage (RECOVERY ONLY — the caller
    gates on the ledger's promote stage being mid-flight): a variant
    artifact minted by a criterion promotion of this concept whose FULL
    selection identity matches what a promotion would stamp right now —
    experiment, manifest epoch, sweep run, and winning cell. The epoch
    alone is not identity: a frozen manifest keeps one hash across many
    sweeps, so an earlier pipeline's agent (different cell, same hash)
    must never be adopted. Returns the artifact path, or None."""
    chain_sweep = (ledger.get("stageResults", {}).get("sweep") or {}).get(
        "runDirectory")
    expected = _expected_promotion_identity(
        name, concept, root,
        chain_sweep_run=os.path.basename(chain_sweep) if chain_sweep
        else None)
    if expected is None:
        return None
    return _minted_agent_matching(name, concept, root, expected=expected,
                                  live_hash=ledger.get("experimentHash"))



def _minted_agent_matching(name: str, concept: str, root: str | None, *,
                           expected, live_hash) -> str | None:
    """The newest variant artifact whose promotion birth certificate
    matches the FULL selection identity: experiment, manifest epoch,
    criterion promotion, sweep run, winning cell, and concept."""
    expected_sweep, expected_cell = expected
    runs = paths.runs_directory(root)
    try:
        entries = sorted(os.listdir(runs), reverse=True)  # newest first
    except OSError:
        return None
    for dirname in entries:
        if "-variant-" not in f"-{dirname}":
            continue
        run_dir = os.path.join(runs, dirname)
        if not os.path.isdir(run_dir):
            continue
        for fname in os.listdir(run_dir):
            if not fname.endswith(".json") or fname == "config.json":
                continue
            try:
                with open(os.path.join(run_dir, fname),
                          encoding="utf-8") as handle:
                    d = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            promotion = d.get("promotion")
            if not isinstance(promotion, dict):
                continue
            cell = promotion.get("winningCell") or {}
            injections = d.get("injections") or []
            if (promotion.get("experiment") == name
                    and promotion.get("experimentHash") == live_hash
                    and promotion.get("promotedBy") == "criterion"
                    and promotion.get("sweepRun") == expected_sweep
                    and isinstance(cell, dict)
                    and cell.get("layer") == expected_cell["layer"]
                    and float(cell.get("alpha", "nan"))
                    == expected_cell["alpha"]
                    and any(inj.get("concept") == concept
                            for inj in injections)):
                return os.path.join(run_dir, fname)
    return None
