"""What an agent-path verb LEARNED, read back out of the run directory it
wrote (WP0 step 8).

Swift twin: the payload helpers at the bottom of
``Sources/ExperimentKit/ExperimentCLIRunner.swift`` (``validationScores``,
``sweepPayload``, ``analysisPayload``, ``allEffectSizesAreZero``).

**The ENVELOPE keys are the parity contract, not the report keys.** The two
engines record the same science under different file names and different keys —
Swift's validate report nests per-concept blocks under ``validation`` with an
``accuracy``; the server's nests them under ``concepts`` with a
``scenarioAccuracy``; Swift's analyze writes ``analysis.json``, the server
writes only ``effect-sizes.csv``. Those are genuine idiom differences (audit
§3.2). What an agent reads — ``result.validation[].accuracy``,
``result.recommendations[].winningCell``, ``result.effectSizeCount`` — is
identical, which is the parity that matters: the same question is answerable
with the same key on either engine.

Every reader here is TOLERANT: a missing or unreadable artifact yields an empty
payload rather than an exception. A verb that did its work must not fail
because the document it produced could not be summarised.
"""

from __future__ import annotations

import csv
import json
import os

#: The chance floor for a two-class held-out probe. Swift twin:
#: ``ExperimentCLIRunner.ValidationScore.isAtOrBelowChance``.
CHANCE_FLOOR = 0.5


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


# --- validate ------------------------------------------------------------------


def validation_scores(run_directory: str) -> list[dict]:
    """One entry per concept's held-out probe, in the envelope's own shape
    (punch list #1, P4: a chance-level probe froze and ran with no machine
    signal at all, because the envelope reported only that validate happened).

    The report's canonical shape is a ``depths`` list with a flat mirror only
    when exactly one depth resolves, so the FIRST depth is the one reported —
    the deepest-scoring reading is never assumed. A concept with no scored
    probe is skipped; the vacuity ledger already names it.
    """
    for name in ("validation-report.json", "report.json"):
        report = _read_json(os.path.join(run_directory, name))
        if not isinstance(report, dict):
            continue
        concepts = report.get("concepts")
        if not isinstance(concepts, dict):
            continue
        scores: list[dict] = []
        for concept in sorted(concepts):
            entry = concepts[concept]
            if not isinstance(entry, dict):
                continue
            depths = entry.get("depths")
            depth = depths[0] if isinstance(depths, list) and depths else entry
            if not isinstance(depth, dict):
                continue
            diagnostics = depth.get("diagnostics")
            diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
            accuracy = depth.get("scenarioAccuracy")
            if accuracy is None:
                accuracy = diagnostics.get("accuracy")
            balanced = diagnostics.get("balancedAccuracy")
            one_sided = bool(diagnostics.get("oneSidedPredictions", False))
            score = {
                "concept": concept,
                "oneSidedPredictions": one_sided,
                "atOrBelowChance": _at_or_below_chance(
                    accuracy, balanced, one_sided),
            }
            if isinstance(depth.get("layer"), int):
                score["layer"] = depth["layer"]
            if isinstance(accuracy, (int, float)):
                score["accuracy"] = float(accuracy)
            if isinstance(balanced, (int, float)):
                score["balancedAccuracy"] = float(balanced)
            if isinstance(diagnostics.get("auc"), (int, float)):
                score["auc"] = float(diagnostics["auc"])
            if isinstance(entry.get("scenarioCount"), int):
                score["scenarios"] = entry["scenarioCount"]
            scores.append(score)
        if scores:
            return scores
    return []


def _at_or_below_chance(accuracy, balanced, one_sided: bool) -> bool:
    """Balanced accuracy is preferred where it exists — a probe on an
    unbalanced set can read 0.7 while separating nothing — and a threshold that
    put every item on one side is at the floor by construction, whatever the
    accuracy says."""
    if one_sided:
        return True
    value = balanced if isinstance(balanced, (int, float)) else accuracy
    if not isinstance(value, (int, float)):
        return False
    return float(value) <= CHANCE_FLOOR + 1e-9


def probe_advisory_detail(score: dict) -> str:
    """The at-chance advisory's prose. Byte-identical to the Swift twin
    (``ValidationScore.advisoryDetail``) — an agent reading either engine's
    document gets the same sentence, and the sentence is the one that says the
    gate will pass anyway."""
    value = score.get("balancedAccuracy", score.get("accuracy"))
    read = f"{float(value):.2f}" if isinstance(value, (int, float)) else "unscored"
    tail = (
        " and the transfer threshold put EVERY item on one side, so the number "
        "is measuring the threshold, not the vector"
        if score.get("oneSidedPredictions")
        else " — at or below the 0.50 chance floor for a two-class probe")
    return (
        f"'{score['concept']}' scores {read} on its held-out probe" + tail
        + ". This evidence still SATISFIES freeze's validateEvidence gate "
          "(the gate asks whether a probe was scored, not how well), so a "
          "non-discriminating direction can be frozen and run. Author more or "
          "better never-named scenarios, or sweep the reading layer, before "
          "treating this concept as measured.")


def vacuous_concepts(run_directory: str):
    """The vacuity ledger a validate run stamped into its own report. ``None``
    when the run predates the stamp (legacy evidence, which keeps satisfying
    the gate exactly as it did)."""
    for name in ("validation-report.json", "report.json"):
        report = _read_json(os.path.join(run_directory, name))
        if isinstance(report, dict) and isinstance(
                report.get("vacuousConcepts"), list):
            return sorted(str(x) for x in report["vacuousConcepts"])
    return None


def validation_payload(experiment: str, run_directory: str) -> dict:
    vacuous = vacuous_concepts(run_directory)
    payload = {
        "experiment": experiment,
        "runDirectory": run_directory,
        "vacuous": bool(vacuous),
    }
    if vacuous is not None:
        payload["vacuousConcepts"] = vacuous
    scores = validation_scores(run_directory)
    if scores:
        payload["validation"] = scores
    return payload


def validation_summary(scores: list[dict]) -> str:
    """The human-readable score list the envelope's message carries."""
    parts = []
    for score in scores:
        accuracy = score.get("accuracy")
        percent = f"{round(float(accuracy) * 100)}%" if isinstance(
            accuracy, (int, float)) else "—"
        auc = score.get("auc")
        tail = f" (AUC {float(auc):.2f})" if isinstance(auc, (int, float)) else ""
        parts.append(f"{score['concept']} {percent}{tail}")
    return ", ".join(parts)


# --- sweep ---------------------------------------------------------------------


def sweep_payload(experiment: str, run_directory: str, *, manifest_status: str,
                  criterion: str, dev_prompts_hash: str = "") -> dict:
    """The sweep's DECISION, machine-readable (punch list #1, P2: the sweep's
    ``--json`` result carried no run directory, no winning cell, no criterion
    and no metrics).

    ``recommendations.json`` maps concept → selection block, or → a string when
    the concept selected nothing (a failure reason, or "awaiting judgment").
    Both are reported: a failure entry is evidence a sweep ran, and ``promote``
    reads it as such.
    """
    recommendations = _read_json(
        os.path.join(run_directory, "recommendations.json"))
    entries: list[dict] = []
    if isinstance(recommendations, dict):
        for concept in sorted(recommendations):
            block = recommendations[concept]
            entry: dict = {"concept": concept, "selected": False}
            if isinstance(block, dict):
                cell = block.get("winningCell")
                if isinstance(cell, dict) and "layer" in cell:
                    entry["selected"] = True
                    entry["winningCell"] = {
                        "layer": cell.get("layer"), "alpha": cell.get("alpha")}
                rule = block.get("criterion")
                if isinstance(rule, dict) and isinstance(
                        rule.get("objective"), dict):
                    metric = rule["objective"].get("metric")
                    if metric:
                        entry["criterion"] = metric
                metrics = block.get("metrics")
                if isinstance(metrics, dict):
                    entry["metrics"] = {
                        key: value for key, value in sorted(metrics.items())
                        if isinstance(value, (int, float))}
                if block.get("control") is not None:
                    entry["control"] = json.dumps(
                        block["control"], sort_keys=True)
                if not dev_prompts_hash and isinstance(
                        block.get("devPromptsHash"), str):
                    dev_prompts_hash = block["devPromptsHash"]
            elif isinstance(block, str):
                entry["failure"] = block
            entries.append(entry)
    return {
        "experiment": experiment,
        "runDirectory": run_directory,
        "manifestStatus": manifest_status,
        # A sweep on a non-draft manifest writes NO `<concept>-recommended`
        # condition — it reports into the run directory only.
        "recommendationsOnly": manifest_status != "draft",
        "criterion": criterion,
        # FULL hash: the dev split is what the selection is provenance FOR.
        "devPromptsHash": dev_prompts_hash,
        "recommendations": entries,
    }


# --- analyze -------------------------------------------------------------------


#: The name of THIS engine's ``effect-sizes.csv`` column set, reported as
#: ``result.effectSizesSchema``. Swift's is ``metric-meanDiff``. Recorded as a
#: known cross-engine difference in WP0-AGENT-SURFACE-AUDIT §3.2/§12: the
#: numbers mean the same thing, the column names do not, and the CSVs are not
#: being unified because existing runs are immutable.
EFFECT_SIZES_SCHEMA = "endpoint-deltaMean"

#: The historical ``emptyAnalysis`` detail: the ONE cause the advisory used to
#: claim unconditionally. Still exactly right when it is what happened. Swift
#: twin: ``ExperimentTasks.emptyAnalysisNoContrast``.
EMPTY_ANALYSIS_NO_CONTRAST = (
    "0 effect-size entries — the source run has no non-baseline condition "
    "to pair against")


def empty_analysis_detail(run_name: str, record_count: int | None,
                          conditions) -> str:
    """The ``emptyAnalysis`` advisory's DETAIL: what an analysis with zero
    effect-size entries actually observed.

    Until WP0 dry run #2 this said one thing unconditionally, and on the run
    that produced the finding it was simply false — the run carried two
    conditions and 24 records, and what failed was READING and PAIRING them.
    An advisory that names the wrong cause is worse than a vague one: it
    sends the reader to audit a study design that is fine while the real
    fault (foreign-engine artifacts, or a record-schema mismatch) goes
    unlooked-at.

    Pure, so every branch is testable without a run directory. Swift twin:
    ``ExperimentTasks.emptyAnalysisDetail``."""
    if record_count is None:
        return (f"0 effect-size entries — the records of source run "
                f"'{run_name}' could not be read here, so this analysis "
                "cannot say what it measured nothing over")
    if record_count == 0:
        return (f"0 effect-size entries — source run '{run_name}' holds no "
                "records at all")
    conditions = set(conditions)
    if not any(c != "baseline" for c in conditions):
        return EMPTY_ANALYSIS_NO_CONTRAST
    named = ", ".join(sorted(conditions))
    return (f"0 effect-size entries — source run '{run_name}' holds "
            f"{record_count} record{'' if record_count == 1 else 's'} across "
            f"condition{'' if len(conditions) == 1 else 's'} {named}, so it "
            "HAS a contrast: analyze could not read or pair those records. "
            "Likeliest cause: artifacts produced on the other engine (record "
            "schemas and pairing keys are per-engine), or a record-schema "
            "mismatch")


def source_run_records(run_directory: str):
    """``(record_count, conditions)`` read straight off a source run's
    generations, for :func:`empty_analysis_detail`. A None count means the
    file could not be read at all — a different fact from "it was empty".
    Swift twin: ``ExperimentTasks.analysisSourceRecords``."""
    path = os.path.join(run_directory, "generations.jsonl")
    try:
        with open(path, encoding="utf-8") as handle:
            lines = [line for line in handle.read().split("\n") if line]
    except OSError:
        return None, set()
    conditions = set()
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and isinstance(record.get("condition"),
                                                   str):
            conditions.add(record["condition"])
    return len(lines), conditions


def analysis_payload(run_directory: str) -> dict:
    """A compact summary of an analyze run's ``effect-sizes.csv`` — the effect
    size LEDGER, not the full table, which stays in the CSV beside it.

    Swift reads the same numbers out of ``analysis.json``; the server never
    wrote one, and inventing one here would change an artifact contract for a
    summary. The envelope's keys are identical either way.
    """
    path = os.path.join(run_directory, "effect-sizes.csv")
    rows = _effect_rows(path)
    pooled = [r for r in rows if (r.get("stratifyBy") or "pooled") == "pooled"]
    conditions = sorted({r["condition"] for r in pooled if r.get("condition")})
    metrics = sorted({r["endpoint"] for r in pooled if r.get("endpoint")})
    significant = 0
    for row in pooled:
        adjusted = _number(row.get("adjustedP"))
        if adjusted is not None and adjusted < 0.05:
            significant += 1
    payload = {
        "effectSizeCount": len(pooled),
        "conditions": conditions,
        "metrics": metrics,
        "significantAtAdjusted05": significant,
        "effectSizesCSV": path,
        # WHICH dialect that CSV is in. The two engines' effect-size columns
        # differ by idiom (audit §3.2) and always have: this engine writes
        # `condition,endpoint,n,deltaMean,…`, Swift writes
        # `condition,metric,n,meanDiff,…`. The columns are NOT being
        # unified — existing run directories are immutable and both engines'
        # readers depend on their own — so the envelope NAMES the dialect
        # instead, which is what a cross-engine reader actually needs. Any
        # future unification is a schema-versioned change announced by this
        # field, never a silent rewrite. Swift twin:
        # `ExperimentCLIRunner.analysisPayload`.
        "effectSizesSchema": EFFECT_SIZES_SCHEMA,
    }
    if len(rows) != len(pooled):
        payload["stratifiedRowCount"] = len(rows) - len(pooled)
    source = read_text(os.path.join(run_directory, "source-run.txt"))
    if source:
        payload["sourceRun"] = source
    config = _read_json(os.path.join(run_directory, "config.json"))
    if isinstance(config, dict):
        if isinstance(config.get("experimentHash"), str):
            payload["experimentHash"] = config["experimentHash"]
        notes = config.get("notes")
        if isinstance(notes, dict) and notes.get("epochUnverified"):
            payload["epochUnverified"] = True
        # Endpoint values that came from an external adjudication rather than
        # from the run's own parse are the single most consequential thing
        # about an analysis, and a reader of the SUMMARY must not have to
        # open config.json to learn it (open-issues §10).
        if isinstance(notes, dict) and isinstance(
                notes.get("adjudicatedEndpoint"), dict):
            block = notes["adjudicatedEndpoint"]
            payload["adjudicated"] = True
            payload["adjudication"] = {
                k: block.get(k) for k in ("fileSha256", "divergence")
                if block.get(k) is not None}
    return payload


def all_effect_sizes_are_zero(run_directory: str) -> bool:
    """True when the analysis produced entries and EVERY paired mean difference
    is exactly zero (punch list #1, P14). Not the same as no entries: the
    pairing worked and the intervention moved nothing, which is either a real
    null or an arm that was declared and never actually injected."""
    rows = [r for r in _effect_rows(
        os.path.join(run_directory, "effect-sizes.csv"))
        if (r.get("stratifyBy") or "pooled") == "pooled"]
    if not rows:
        return False
    for row in rows:
        value = _number(row.get("deltaMean"))
        if value is None or value != 0:
            return False
    return True


def _effect_rows(path: str) -> list[dict]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, ValueError):
        return []


def _number(raw):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def read_json(path: str):
    """A run artifact read back for the envelope, or ``None``.

    Tolerant on purpose, exactly like :func:`read_text`: an envelope field
    lifted out of an artifact the verb has already written must never be the
    thing that turns a completed verb into a failure. An absent or unreadable
    file simply contributes nothing."""
    import json as _json
    try:
        with open(path, encoding="utf-8") as handle:
            return _json.load(handle)
    except (OSError, ValueError):
        return None
