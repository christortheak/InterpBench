"""Per-item paired deltas for the answer-token choice instrument.

``analyze`` writes ``choice-deltas.csv`` — the CITABLE, engine-computed
version of the per-item Δ the results explorer otherwise derives client-side.
A number a paper quotes should come out of the engine, under the epoch and
measurement-drift guards, not out of a viewer's arithmetic.

One row per NON-baseline instrument record, joined to the same item's baseline
record::

    condition, promptID, target,
    baselineTargetLogOdds, conditionTargetLogOdds, deltaTargetLogOdds,
    baselineSelected, conditionSelected, flipped,
    baselineTargetProbability, conditionTargetProbability,
    deltaTargetProbability

Pinned decisions (the Swift twin ``ChoiceDeltas.swift`` implements each one
identically):

* **The join key is ``promptID`` alone.** The answer-token instrument emits
  exactly ONE deterministic readout per (condition, prompt) — it never
  samples (``tasks.py`` emits it outside the ``samplesPerItem`` loop and the
  resume ledger keys it with ``sampleIndex = None``), so instrument records
  carry no ``sampleIndex`` to join on. Should one ever appear, it joins as
  part of the key rather than silently collapsing replicates.
* **A non-baseline record with no baseline partner is skipped AND COUNTED**
  (``skippedNoBaseline``). Silent truncation reads as coverage: a table of 8
  rows where 12 items were measured must say so.
* **The row's quantity is the TARGET option's log-odds**, the same endpoint
  the paired effect sizes use (``choiceLogOdds``). A record whose ``logOdds``
  has no entry for its own target is unreadable, not zero: it is skipped and
  counted (``skippedNoTargetValue``).
* **Probabilities are a convenience column, not the row's reason to exist.**
  When ``choiceProbability`` is missing the three probability cells are
  empty and the log-odds row still stands.
* **``flipped`` needs both sides' ``selected``.** With either missing the
  cell is 0 and the two selected columns are empty — a flip is a claim about
  two observed choices, never an inference from one.
* **Rows sort by (condition, promptID)**, stably over run order, so the same
  run in produces a byte-identical file out.

The per-condition summary (``choice-deltas.json``) reuses the existing paired
bootstrap in :mod:`study_stats` at its default conventions (10 000
replicates, seed 0) — the same machinery ``effect-sizes.csv`` runs on. No new
statistics live here.
"""

from __future__ import annotations

import csv
import math

CHOICE_DELTAS_HEADER = [
    "condition", "promptID", "target",
    "baselineTargetLogOdds", "conditionTargetLogOdds", "deltaTargetLogOdds",
    "baselineSelected", "conditionSelected", "flipped",
    "baselineTargetProbability", "conditionTargetProbability",
    "deltaTargetProbability",
]

#: The instrument id whose records this reads. ``choiceProbability`` and
#: ``ordinalScale`` ride the SAME record (``logprob.ChoiceResult
#: .as_record_fields`` stamps ``answerTokenLogprob`` for all three), so one id
#: covers the whole choice family.
INSTRUMENT = "answerTokenLogprob"

BASELINE = "baseline"


def _fmt(value) -> str:
    """CSV rendering, matching ``study_stats._fmt``: 6 significant digits,
    empty for absent/NaN. Six digits is what effect-sizes.csv already
    publishes — one dialect for engine-computed numbers."""
    if value is None:
        return ""
    number = float(value)
    return "" if math.isnan(number) else f"{number:.6g}"


def _number(mapping, key):
    """``mapping[key]`` as a finite float, or None when it is absent or not a
    real number (a JSON ``null``/string cannot be a measurement)."""
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _key(record: dict) -> tuple[str, str]:
    """The item identity a condition record pairs to its baseline by.

    ``sampleIndex`` joins only when the record actually carries one; today's
    instrument path never writes it (see the module docstring), and absent
    normalizes to "" rather than 0 so a future sampled instrument cannot pair
    a sample-0 record with an unsampled one.
    """
    sample = record.get("sampleIndex")
    return (str(record.get("promptID", "")),
            "" if sample is None else str(sample))


def is_instrument(record: dict) -> bool:
    return record.get("instrument") == INSTRUMENT and "error" not in record


def rows(records: list[dict], declared_targets: dict[str, bool] | None = None,
         ) -> tuple[list[list], dict]:
    """``(csv_rows, summary)`` over a run's records.

    ``summary`` is the ``choice-deltas.json`` payload: per-condition n, mean Δ
    target log-odds with a paired bootstrap 95% CI, flip count, and the skip
    counts. Empty ``csv_rows`` AND an empty ``conditions`` map means the run
    had nothing for this artifact to be about — the caller writes nothing.
    """
    from . import study_stats

    # Declared targets only (open-issues #6): the run-time synthesized
    # options[0] target gave every ordinalScale record a fake "target" (the
    # scale minimum), and this artifact would then tabulate pole-token
    # log-odds as if they were a choice endpoint. Authority ladder shared
    # with ``_endpoint_values``: the pinned task file's per-item map when
    # the caller supplies it (exact; keeps mixed instruments like
    # s4-framings), else the record's own ``targetSource`` stamp, else the
    # historical backstop — an unstamped record riding an ordinalScale
    # readout is the failure class.
    def _declared(r: dict) -> bool:
        pid = str(r.get("promptID", ""))
        if declared_targets is not None and pid in declared_targets:
            return declared_targets[pid]
        if "targetSource" in r:
            return r.get("targetSource") == "declared"
        return r.get("ordinalPosition") is None

    instrument_records = [r for r in records if is_instrument(r) and _declared(r)]
    baselines: dict[tuple[str, str], dict] = {}
    for record in instrument_records:
        if record.get("condition") == BASELINE:
            baselines.setdefault(_key(record), record)

    out: list[list] = []
    diffs: dict[str, list[float]] = {}
    flips: dict[str, int] = {}
    skipped_no_baseline: dict[str, int] = {}
    skipped_no_target: dict[str, int] = {}
    conditions: set[str] = set()
    for record in instrument_records:
        condition = str(record.get("condition", ""))
        if condition == BASELINE:
            continue
        conditions.add(condition)
        base = baselines.get(_key(record))
        if base is None:
            skipped_no_baseline[condition] = skipped_no_baseline.get(condition, 0) + 1
            continue
        target = record.get("target")
        target = "" if target is None else str(target)
        condition_odds = _number(record.get("logOdds"), target)
        baseline_odds = _number(base.get("logOdds"), target)
        if condition_odds is None or baseline_odds is None:
            skipped_no_target[condition] = skipped_no_target.get(condition, 0) + 1
            continue
        delta = condition_odds - baseline_odds
        diffs.setdefault(condition, []).append(delta)

        condition_p = _number(record.get("choiceProbability"), target)
        baseline_p = _number(base.get("choiceProbability"), target)
        delta_p = (None if condition_p is None or baseline_p is None
                   else condition_p - baseline_p)

        base_selected = base.get("selected")
        condition_selected = record.get("selected")
        both_selected = (isinstance(base_selected, str)
                         and isinstance(condition_selected, str)
                         and base_selected != "" and condition_selected != "")
        flipped = 1 if both_selected and base_selected != condition_selected else 0
        if flipped:
            flips[condition] = flips.get(condition, 0) + 1

        out.append([
            condition,
            str(record.get("promptID", "")),
            target,
            _fmt(baseline_odds),
            _fmt(condition_odds),
            _fmt(delta),
            base_selected if both_selected else "",
            condition_selected if both_selected else "",
            flipped,
            _fmt(baseline_p),
            _fmt(condition_p),
            _fmt(delta_p),
        ])

    # Sorted so the same run in gives a byte-identical file out. Stable, so
    # any residual duplicate key keeps run order rather than an arbitrary one.
    out.sort(key=lambda row: (row[0], row[1]))

    summary_conditions: dict[str, dict] = {}
    for condition in sorted(conditions):
        values = diffs.get(condition, [])
        block: dict = {
            "n": len(values),
            "flipped": flips.get(condition, 0),
            "skippedNoBaseline": skipped_no_baseline.get(condition, 0),
            "skippedNoTargetValue": skipped_no_target.get(condition, 0),
        }
        if values:
            ci = study_stats.paired_bootstrap_ci(values)
            block.update({
                "deltaTargetLogOddsMean": ci.mean,
                "ciLower": ci.ci_lower,
                "ciUpper": ci.ci_upper,
                "replicates": ci.replicates,
                "seed": ci.seed,
            })
        summary_conditions[condition] = block

    summary = {
        "conditions": summary_conditions,
        "records": len(out),
        "skippedNoBaseline": sum(skipped_no_baseline.values()),
        "skippedNoTargetValue": sum(skipped_no_target.values()),
    }
    return out, summary


def write_csv(path: str, csv_rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CHOICE_DELTAS_HEADER)
        for row in csv_rows:
            writer.writerow(row)
