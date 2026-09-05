"""Run report and CSV contracts shared by execution and shard merge.

Consumes retained records; never acquires a model or imports task orchestration.
"""
from __future__ import annotations
import csv
import json
import os
from . import judicial, truncation_gate
from .manifest import Manifest


def write_metrics_csv(records: list[dict], run_directory: str,
                       style=None) -> None:
    """Write Swift-compatible run metrics for result browsers and notebooks.

    ``style`` (a ``reasoning_style.PinnedStyle``) adds one ``rs_<featureID>``
    column per taxonomy feature, in declared taxonomy order (cross-engine
    contract) — values recomputed from each record's output text. Records
    carrying factorial ``factors`` metadata add one ``factor_<name>`` column
    per factor name (sorted union, appended last — cross-engine contract);
    a factor-less run's header and rows are byte-identical to before."""
    path = os.path.join(run_directory, "metrics.csv")
    feature_ids = style.taxonomy.feature_ids if style else []
    sampled = [r for r in records
               if "error" not in r and "instrument" not in r]
    factor_names = sorted({name for r in sampled
                           for name in (r.get("factors") or {})})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["condition", "seed", "promptIndex", "promptID", "wordCount", "distinct2"]
            + [f"rs_{fid}" for fid in feature_ids]
            + [f"factor_{name}" for name in factor_names])
        for index, record in enumerate(records):
            if "error" in record or "instrument" in record:
                continue  # instrument readouts have no sampled text to score
            row = [
                record.get("condition", ""),
                record.get("seed", 0),
                record.get("promptIndex", index),
                record.get("promptID", ""),
                record.get("wordCount", 0),
                record.get("distinct2", 0.0),
            ]
            if style:
                values = style.taxonomy.score(record.get("output", ""))
                row += [values.get(fid, 0.0) for fid in feature_ids]
            row += [(record.get("factors") or {}).get(name, "")
                    for name in factor_names]
            writer.writerow(row)



def write_summaries_csv(records: list[dict], run_directory: str) -> None:
    """Per-(condition, prompt) distributional summaries — Case 3's mean AND
    spread endpoints over the sample axis, plus parse-failure and choice rates.
    One row per item cell; the analysis step consumes this alongside the raw
    JSONL."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        if "error" in record:
            continue
        key = (record.get("condition", ""), str(record.get("promptID", "")))
        groups.setdefault(key, []).append(record)
    if not groups:
        return
    header = ["condition", "promptID", "samples",
              "monthsParseFailureRate", "monthsMean", "monthsStdev", "monthsMin",
              "monthsQ25", "monthsMedian", "monthsQ75", "monthsMax",
              "choiceRates", "selectedOption", "targetProbability", "targetLogOdds",
              # Truncation, per cell and unconditionally — appended last, so a
              # reader of the historical columns is unaffected. Reported even
              # with the gate off (see truncation_gate): the 2026-08-30
              # incident was invisible because nothing counted, and a reader
              # who was never told to look for truncation should still meet it
              # here beside the other per-cell aggregates. Blank on a cell that
              # classified nothing — no generation of it carried a
              # `finishReason` — which is a different claim from zero.
              # `lengthStoppedInReasoning` is the subset that never closed
              # its reasoning block (always 0 without a reasoning budget).
              "lengthStopped", "lengthStoppedInReasoning",
              "lengthStoppedFraction"]
    path = os.path.join(run_directory, "summaries.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for (condition, prompt_id), items in sorted(groups.items()):
            sampled = [r for r in items if "output" in r]
            row: dict = {"condition": condition, "promptID": prompt_id,
                         "samples": len(sampled)}
            months = [r["parsedMonths"] for r in sampled if "parsedMonths" in r]
            if months:
                row["monthsParseFailureRate"] = f"{judicial.parse_failure_rate(months):.6g}"
                summary = judicial.summarize(months)
                if summary is not None:
                    row.update({
                        "monthsMean": f"{summary.mean:.6g}",
                        "monthsStdev": f"{summary.stdev:.6g}",
                        "monthsMin": f"{summary.minimum:.6g}",
                        "monthsQ25": f"{summary.q25:.6g}",
                        "monthsMedian": f"{summary.median:.6g}",
                        "monthsQ75": f"{summary.q75:.6g}",
                        "monthsMax": f"{summary.maximum:.6g}",
                    })
            classified = [r for r in sampled
                          if isinstance(r.get(truncation_gate.RECORD_KEY), str)]
            if classified:
                stopped = sum(
                    1 for r in classified
                    if r[truncation_gate.RECORD_KEY]
                    in truncation_gate.CUT_OFF_REASONS)
                in_reasoning = sum(
                    1 for r in classified
                    if r[truncation_gate.RECORD_KEY]
                    == truncation_gate.FINISH_LENGTH_IN_REASONING)
                row["lengthStopped"] = stopped
                row["lengthStoppedInReasoning"] = in_reasoning
                row["lengthStoppedFraction"] = \
                    f"{stopped / len(classified):.6g}"
            choices = [r["parsedChoice"] for r in sampled if "parsedChoice" in r]
            if choices:
                parsed = [c for c in choices if c is not None]
                rates = {option: parsed.count(option) / len(parsed)
                         for option in sorted(set(parsed))} if parsed else {}
                row["choiceRates"] = json.dumps(rates, sort_keys=True)
            for record in items:
                if record.get("instrument") == "answerTokenLogprob":
                    target = record.get("target")
                    row["selectedOption"] = record.get("selected", "")
                    if target is not None:
                        probability = record.get("choiceProbability", {}).get(target)
                        odds = record.get("logOdds", {}).get(target)
                        if probability is not None:
                            row["targetProbability"] = f"{probability:.6g}"
                        if odds is not None:
                            row["targetLogOdds"] = f"{odds:.6g}"
                    break
            writer.writerow(row)



def reasoning_style_summary(sampled: list[dict], style) -> dict | None:
    """Per-condition ``reasoningStyle`` report block (cross-engine contract:
    {"taxonomy", "taxonomyHash", "taxonomyFile", "diagnosticOnly",
    "features": {id: {"mean", "n"}}}): the mean of each feature's
    per-generation values over this condition's sampled outputs — the same
    values, in each feature's own declared normalization units, as the
    ``rs_<featureID>`` metrics.csv columns. ``taxonomyFile`` names the pinned
    taxonomy file (beside its hash) so the report is self-describing, and
    ``diagnosticOnly`` marks these as surface style features — a
    diagnostic/manipulation check reported beside outcome endpoints, never an
    outcome endpoint itself (docs/METHODS.md). None when no taxonomy is
    pinned or nothing was sampled."""
    if style is None or not sampled:
        return None
    scored = [style.taxonomy.score(record.get("output", "")) for record in sampled]
    return {
        "taxonomy": style.taxonomy.name,
        "taxonomyHash": style.hash,
        "taxonomyFile": style.path,
        "diagnosticOnly": True,
        "features": {
            fid: {
                "mean": sum(values.get(fid, 0.0) for values in scored) / len(scored),
                "n": len(scored),
            }
            for fid in style.taxonomy.feature_ids
        },
    }



def choice_readouts(items: list[dict]) -> dict[tuple, str]:
    """A condition's categorical choices keyed by
    ``(promptID, sampleIndex, source)``: the sampled parse (source
    ``"parsed"``) and the deterministic instrument's selected option (source
    ``"instrument"``). ``None`` parses are excluded — an unparseable output
    can neither agree nor disagree."""
    out: dict[tuple, str] = {}
    for record in items:
        prompt_id = str(record.get("promptID", ""))
        if record.get("instrument") == "answerTokenLogprob":
            selected = record.get("selected")
            if selected is not None:
                out[(prompt_id, record.get("sampleIndex"), "instrument")] = selected
        elif record.get("parsedChoice") is not None:
            out[(prompt_id, record.get("sampleIndex"), "parsed")] = \
                record["parsedChoice"]
    return out



def write_report(name: str, manifest: Manifest, records: list[dict],
                  run_directory: str, battery: dict | None = None,
                  style=None, numeric_parser=None, sharded=None) -> None:
    """report.json. Choice-bearing runs additionally stamp, per condition:
    ``choiceRate`` (fraction of parseable sampled outputs choosing the item's
    target — same rule as the ``choiceRate`` analysis endpoint) and, for every
    non-baseline condition, the cross-engine parity summary
    ``agreementWithBaseline`` = ``{"n", "agreement"}``: over the choice
    readouts (sampled parses + instrument selections) paired item-for-item
    with baseline's, the fraction that chose the SAME option. A
    no-intervention variant at temperature 0 should sit at agreement 1.0 —
    a cheap, computable baseline-parity check."""
    by_condition: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}
    for r in records:
        if "error" in r:  # a failed variant condition produced no generations
            errors[r["condition"]] = r["error"]
            continue
        by_condition.setdefault(r["condition"], []).append(r)
    baseline_choices = choice_readouts(by_condition.get("baseline", []))
    conditions = {}
    for cond, error in errors.items():
        conditions[cond] = {"generations": 0, "error": error}
    for cond, items in by_condition.items():
        sampled = [i for i in items if "instrument" not in i]
        n = len(sampled)
        conditions[cond] = {
            "generations": n,
            "meanWordCount": sum(i["wordCount"] for i in sampled) / n if n else 0.0,
            "meanDistinct2": sum(i["distinct2"] for i in sampled) / n if n else 0.0,
        }
        instrument_readouts = sum(1 for i in items if "instrument" in i)
        if instrument_readouts:
            conditions[cond]["choiceReadouts"] = instrument_readouts
        # Ordinal-scale summary (cross-engine contract keys
        # "ordinalMean"/"ordinalSD"; population SD — defined for a single
        # readout). Swift twin: ExperimentTasks.report.
        positions = [i["ordinalPosition"] for i in items
                     if "ordinalPosition" in i]
        if positions:
            mean_position = sum(positions) / len(positions)
            conditions[cond]["ordinalMean"] = mean_position
            conditions[cond]["ordinalSD"] = (
                sum((p - mean_position) ** 2 for p in positions)
                / len(positions)) ** 0.5
        target_hits = [i["parsedChoice"] == i["target"] for i in sampled
                       if i.get("target") is not None
                       and i.get("parsedChoice") is not None]
        if target_hits:
            conditions[cond]["choiceRate"] = sum(target_hits) / len(target_hits)
        if cond != "baseline" and baseline_choices:
            mine = choice_readouts(items)
            shared = set(mine) & set(baseline_choices)
            if shared:
                agree = sum(1 for key in shared
                            if mine[key] == baseline_choices[key])
                conditions[cond]["agreementWithBaseline"] = {
                    "n": len(shared), "agreement": agree / len(shared)}
        reasoning_style_block = reasoning_style_summary(sampled, style)
        if reasoning_style_block is not None:
            conditions[cond]["reasoningStyle"] = reasoning_style_block
    # Per-condition capability-battery results (cross-engine contract key
    # "capabilityBattery": {"accuracy", "itemCount", "batteryHash"}).
    for cond, block in (battery or {}).items():
        conditions.setdefault(cond, {"generations": 0})["capabilityBattery"] = block
    report = {
        "experiment": name, "experimentHash": manifest.content_hash(),
        "promptMode": manifest.prompt_mode, "conditionCount": len(by_condition),
        "conditions": conditions,
        # Truncation (cross-engine contract key "truncation": {"threshold",
        # "classified", "lengthStopped", "lengthStoppedFraction", "cells"}).
        # Written whether or not the study declared a ceiling — the reading
        # is evidence about the run, and the gate is only what a study chose
        # to do about it. Per CELL, because a run-wide fraction is exactly
        # what hid the 2026-08-30 incident: one arm was fine, the other was
        # not, and the pooled number looked unremarkable.
        "truncation": truncation_gate.report(
            records, threshold=manifest.max_length_stopped_fraction),
    }
    # Registry-parser provenance (cross-engine contract key "numericParser":
    # {"name", "kind", "registryFile", "registryHash"}): stamped only when a
    # declared parser actually parsed this run's numeric outcome — legacy
    # runs' report bytes are unchanged.
    if numeric_parser is not None:
        report["numericParser"] = numeric_parser.provenance()
    # Shard provenance of a MERGED run ({"shardCount", "shardRuns",
    # "shardJobIDs"?}, deterministic — no timestamps). Only the merge passes
    # it; single-job runs and shard partials never carry the key, and it
    # lives here, never in config.json (closed schema-2 contract) or the
    # manifest.
    if sharded is not None:
        report["sharded"] = sharded
    # What actually RAN, as a set. The manifest's modelID is a declared
    # default; a panel's seats may each name their own, so a single scalar
    # cannot describe the run. config.json's key set is a closed cross-engine
    # contract, so the honest multi-model view lives here.
    used = sorted({r.get("modelID") for r in records if r.get("modelID")})
    if used:
        report["modelsUsed"] = used
        report["declaredModelID"] = manifest.model_id
        if len(used) > 1 or used[0] != manifest.model_id:
            # Per-seat, so a reader can attribute a turn to its model without
            # re-reading generations.jsonl.
            by_seat: dict[str, str] = {}
            for r in records:
                seat, model = r.get("speakerName"), r.get("modelID")
                if seat and model:
                    by_seat[seat] = model
            if by_seat:
                report["modelBySeat"] = by_seat
    with open(os.path.join(run_directory, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

