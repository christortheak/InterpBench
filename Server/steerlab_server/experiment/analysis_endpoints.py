"""Endpoint reduction, stratification and promotion summaries over retained evidence."""
from __future__ import annotations
import json
import os
from . import judicial, paths
from .manifest import Manifest
_STRATUM_JOIN = "×"
_PRIMARY_ENDPOINT_ORDER = ("choiceLogOdds", "meanMonths", "choiceRate")

def condition_modalities(manifest: Manifest, root: str | None = None) -> dict[str, str]:
    """Intervention modality per condition, derived from the manifest
    (RESULTS-ARCHITECTURE: modality is a design axis — injection / adapter /
    systemPrompt / stacked; baseline = none).

    Steering-slot conditions are pure residual-stream work → "injection"
    (matched-norm random controls included: same modality, different content).
    Variant conditions are read off the variant artifact's components —
    adapters, injections, and a variant-level systemPrompt — with any
    combination of two or more → "stacked". Conditions that cannot be found in
    the manifest map to "" (never guessed).

    SAE latent conditions get their OWN modality, ``"saeLatent"`` — not
    "injection". They are residual-stream work, but the edit is state-dependent
    and dosed in latent units rather than residual-norm units, so pooling them
    with vector additions under one label would let a reader compare doses that
    are not comparable.
    """
    modalities: dict[str, str] = {"baseline": "none"}
    for condition in manifest.conditions:
        modalities[condition.name] = "injection"
    for entry in manifest.sae_latent_conditions:
        if entry.get("name"):
            modalities[str(entry["name"])] = "saeLatent"
    for vc in manifest.variant_conditions:
        artifact = vc.artifact or {}
        if not artifact and vc.artifact_path:
            # Older manifests pin path+hash without embedding the artifact.
            try:
                with open(paths.resolve(vc.artifact_path, root), encoding="utf-8") as handle:
                    artifact = json.load(handle)
            except (OSError, json.JSONDecodeError):
                artifact = {}
        has_injection = bool(artifact.get("injections"))
        has_adapter = bool(artifact.get("adapters"))
        has_system_prompt = bool(str(artifact.get("systemPrompt") or "").strip())
        components = [name for name, present in (
            ("injection", has_injection), ("adapter", has_adapter),
            ("systemPrompt", has_system_prompt)) if present]
        if len(components) > 1:
            modalities[vc.name] = "stacked"
        elif components:
            modalities[vc.name] = components[0]
        else:
            modalities[vc.name] = "none"
    return modalities



def key_records_by_transcript(records: list[dict]) -> list[dict]:
    """Rewrite promptID to ``<turnID>@<replicateIndex>`` so the pairing join
    distinguishes the same script position in DIFFERENT play-throughs.

    Without this the sample-axis averaging in ``endpoint_values`` pools every
    replicate of a turn into one cell, which silently discards exactly the
    between-transcript variation the clustered estimator exists to measure."""
    out = []
    for record in records:
        replicate = record.get("replicateIndex", 0) or 0
        clone = dict(record)
        clone["promptID"] = f"{record.get('promptID', '')}@{replicate}"
        out.append(clone)
    return out



def transcript_level_diffs(values: dict[str, float],
                            base: dict[str, float]) -> list[float]:
    """Turn-level paired differences aggregated to ONE value per transcript.

    Clusters are balanced by construction — every transcript plays the same
    turn script — which is exactly the case where cluster-level aggregation is
    the correct estimator and needs no new statistics: the existing bootstrap
    and Wilcoxon then run over transcript-level values, so the reported ``n``
    is the number of transcripts. Cross-engine twin of Swift
    ``ExperimentTasks.transcriptEffectSizes``."""
    by_transcript: dict[str, list[float]] = {}
    for pid in sorted(values):
        if pid not in base:
            continue
        _, _, replicate = pid.rpartition("@")
        by_transcript.setdefault(replicate, []).append(values[pid] - base[pid])
    return [sum(v) / len(v) for _, v in sorted(by_transcript.items()) if v]



def endpoint_values(records: list[dict], style=None,
                     numeric_parser_kind: str | None = None,
                     declared_targets: dict[str, bool] | None = None,
                     ) -> dict[str, dict[str, dict[str, float]]]:
    """endpoint → condition → promptID → value, from a run's records.

    Endpoints: ``choiceLogOdds`` (instrument log-odds of the item's target
    option), ``ordinalPosition`` (the ordinalScale instrument's ladder
    position — stamped on the instrument record only when the manifest
    declared ordinalScale; cross-engine endpoint name pinned against Swift's
    ``effectSizes``), ``choiceRate`` (sampled parsed-choice rate of the
    target), ``meanMonths`` and ``monthsSpread`` (Case 3 mean and stdev over
    the sample axis), ``readerScore:<concept>`` (mean RepE reader score of the
    sampled outputs, one endpoint per pinned reader), and — when ``style``
    pins a reasoning-style taxonomy — ``rs_<featureID>`` (mean per-generation
    feature value over the sample axis, recomputed from the output text).
    Same-item pairing happens downstream by promptID.

    Endpoint-label honesty (2026-08-06): the ``parsedMonths`` record key is
    written by ANY declared registry parser — percentage and 1–7 scale
    parsers included — so ``meanMonths``/``monthsSpread`` were verified
    misleading on real runs whose parser never produced months. When
    ``numeric_parser_kind`` names a registry kind other than
    ``durationMonths``, the same values are ADDITIONALLY emitted under the
    neutral ``parsedValueMean``/``parsedValueSpread`` endpoints; the months
    labels are retained as deprecated aliases so existing readers (residual
    human-baseline joins, ``_PRIMARY_ENDPOINT_ORDER``, the explorer) keep
    working. New readers should prefer the parsedValue names."""
    endpoints: dict[str, dict[str, dict[str, float]]] = {}

    def put(endpoint: str, condition: str, prompt_id: str, value: float) -> None:
        endpoints.setdefault(endpoint, {}).setdefault(condition, {})[prompt_id] = value

    cells: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        if "error" in record:
            continue
        key = (record.get("condition", ""), str(record.get("promptID", "")))
        cells.setdefault(key, []).append(record)
    # Prose endpoints. Every generation already carries these and Swift has
    # always used them, but this collector never emitted them — so a panel
    # study (prose turns, no choice instrument, often no pinned taxonomy)
    # produced an EMPTY endpoint set and therefore no effect sizes at all,
    # silently. They are means over the sample axis, like every other endpoint
    # here.
    for (condition, prompt_id), items in cells.items():
        for endpoint, field in (("wordCount", "wordCount"),
                                ("distinct2", "distinct2")):
            values = [float(r[field]) for r in items
                      if isinstance(r.get(field), (int, float))]
            if values:
                put(endpoint, condition, prompt_id, sum(values) / len(values))
    for (condition, prompt_id), items in cells.items():
        months: list[float | None] = []
        target_hits: list[bool] = []
        reader_values: dict[str, list[float]] = {}
        style_values: dict[str, list[float]] = {}
        for record in items:
            if record.get("instrument") == "answerTokenLogprob":
                # choiceLogOdds is a DECLARED endpoint (open-issues #6). The
                # authority ladder: the pinned task file's per-item map when
                # the caller could load it (exact — handles mixed
                # instruments like s4-framings, declared target + ordinal
                # readout on one record); else the record's own targetSource
                # stamp (new writers); else the observed historical failure
                # class — a record whose only "target" was synthesized rides
                # an ordinalScale readout, so ordinalPosition marks it.
                target = record.get("target")
                if declared_targets is not None and prompt_id in declared_targets:
                    declared = declared_targets[prompt_id]
                elif "targetSource" in record:
                    declared = record.get("targetSource") == "declared"
                else:
                    declared = record.get("ordinalPosition") is None
                odds = record.get("logOdds", {}).get(target)
                if odds is not None and declared:
                    put("choiceLogOdds", condition, prompt_id, float(odds))
                # ordinalScale rides the same instrument record: the ladder
                # position is one more per-item numeric endpoint through the
                # SAME paired machinery (no new statistics). Key present only
                # when the manifest declared ordinalScale at run time.
                position = record.get("ordinalPosition")
                if position is not None:
                    put("ordinalPosition", condition, prompt_id,
                        float(position))
                continue
            if "parsedMonths" in record:
                months.append(record["parsedMonths"])
            if "parsedChoice" in record and record.get("target") is not None:
                if record["parsedChoice"] is not None:
                    target_hits.append(record["parsedChoice"] == record["target"])
            for concept, score in (record.get("readerScores") or {}).items():
                reader_values.setdefault(str(concept), []).append(float(score))
            if style is not None and "output" in record:
                scored = style.taxonomy.score(record.get("output", ""))
                for fid in style.taxonomy.feature_ids:
                    style_values.setdefault(fid, []).append(scored.get(fid, 0.0))
        for concept, values in reader_values.items():
            put(f"readerScore:{concept}", condition, prompt_id,
                sum(values) / len(values))
        for fid, values in style_values.items():
            put(f"rs_{fid}", condition, prompt_id, sum(values) / len(values))
        summary = judicial.summarize([m for m in months])
        if summary is not None:
            put("meanMonths", condition, prompt_id, summary.mean)
            if summary.count > 1:
                put("monthsSpread", condition, prompt_id, summary.stdev)
            # Honest twin labels for non-months parsers (see docstring):
            # additive, so every months-keyed reader keeps working.
            if numeric_parser_kind not in (None, "durationMonths"):
                put("parsedValueMean", condition, prompt_id, summary.mean)
                if summary.count > 1:
                    put("parsedValueSpread", condition, prompt_id,
                        summary.stdev)
        if target_hits:
            put("choiceRate", condition, prompt_id,
                sum(target_hits) / len(target_hits))
    return endpoints



def _item_factor_levels(records: list[dict]) -> dict[str, dict[str, str]]:
    """promptID → {factorKey: level} from the record-carried prompt metadata
    (``_PROMPT_META_KEYS`` stamps ``arm``/``caseID``/``factors`` verbatim on
    sampled AND instrument records, so no rejoin of the input file). First
    record per item wins; items stamp their metadata identically by
    construction."""
    meta: dict[str, dict[str, str]] = {}
    for record in records:
        prompt_id = str(record.get("promptID", ""))
        if not prompt_id or prompt_id in meta:
            continue
        levels: dict[str, str] = {}
        for key in ("arm", "caseID"):
            value = record.get(key)
            if isinstance(value, str) and value:
                levels[key] = value
        factors = record.get("factors")
        if isinstance(factors, dict):
            for key, value in factors.items():
                if isinstance(value, str) and value:
                    levels[str(key)] = value
        meta[prompt_id] = levels
    return meta



def _stratification_families(
        factors_by_item: dict[str, dict[str, str]],
        items: set[str]) -> list[tuple[str, dict[str, set[str]]]]:
    """The stratification families for a run, in a fixed order: promptID
    (always), each factor key with ≥2 observed levels (marginals), and the
    full cross of ALL factor keys when there are ≥2 of them (the per-cell
    view — e.g. ``arm×caseID`` → ``notLegal×loan``). Keys sort
    alphabetically; a constant factor is skipped (its one stratum would just
    duplicate the pooled row under another name)."""
    families: list[tuple[str, dict[str, set[str]]]] = [
        ("promptID", {prompt_id: {prompt_id} for prompt_id in sorted(items)})]
    keys = sorted({key for levels in factors_by_item.values() for key in levels})
    for key in keys:
        strata: dict[str, set[str]] = {}
        for prompt_id in sorted(items):
            level = factors_by_item.get(prompt_id, {}).get(key)
            if level is not None:
                strata.setdefault(level, set()).add(prompt_id)
        if len(strata) >= 2:
            families.append((key, strata))
    if len(keys) >= 2:
        cells: dict[str, set[str]] = {}
        for prompt_id in sorted(items):
            levels = factors_by_item.get(prompt_id, {})
            if all(key in levels for key in keys):
                label = _STRATUM_JOIN.join(levels[key] for key in keys)
                cells.setdefault(label, set()).add(prompt_id)
        if len(cells) >= 2:
            families.append((_STRATUM_JOIN.join(keys), cells))
    return families



def _endpoint_sample_values(
        records: list[dict],
        style=None) -> dict[str, dict[str, dict[str, dict[int, float]]]]:
    """endpoint → condition → promptID → {sampleIndex: value}, for endpoints
    that have a per-sample reading (``wordCount``, ``distinct2``,
    ``choiceRate`` as the 0/1 target hit, ``meanMonths`` as the per-sample
    parse, ``readerScore:<concept>``, ``rs_<featureID>``). Deterministic
    instrument readouts (choiceLogOdds, ordinalPosition) and cross-sample
    aggregates (monthsSpread) have no sample axis and are absent. This is the
    single-item stratum's resolution: within one item, treatment sample k
    pairs to baseline sample k."""
    out: dict[str, dict[str, dict[str, dict[int, float]]]] = {}

    def put(endpoint: str, condition: str, prompt_id: str, sample: int,
            value: float) -> None:
        out.setdefault(endpoint, {}).setdefault(condition, {}) \
           .setdefault(prompt_id, {})[sample] = value

    for record in records:
        if "error" in record or record.get("instrument"):
            continue
        condition = record.get("condition", "")
        prompt_id = str(record.get("promptID", ""))
        sample = int(record.get("sampleIndex") or 0)
        for endpoint, field in (("wordCount", "wordCount"),
                                ("distinct2", "distinct2")):
            value = record.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                put(endpoint, condition, prompt_id, sample, float(value))
        if record.get("parsedMonths") is not None:
            put("meanMonths", condition, prompt_id, sample,
                float(record["parsedMonths"]))
        if "parsedChoice" in record and record.get("target") is not None \
                and record["parsedChoice"] is not None:
            put("choiceRate", condition, prompt_id, sample,
                1.0 if record["parsedChoice"] == record["target"] else 0.0)
        for concept, score in (record.get("readerScores") or {}).items():
            put(f"readerScore:{concept}", condition, prompt_id, sample,
                float(score))
        if style is not None and "output" in record:
            scored = style.taxonomy.score(record.get("output", ""))
            for fid in style.taxonomy.feature_ids:
                put(f"rs_{fid}", condition, prompt_id, sample,
                    scored.get(fid, 0.0))
    return out



def stratified_effect_rows(records: list[dict], endpoints, *, style, method,
                            modalities) -> list:
    """The stratified companion rows to analyze's pooled effect rows.

    Within a stratum the unit of analysis is the ITEM whenever the stratum
    has ≥2 items pairing to a baseline (the pooled machinery restricted to
    the stratum); a single-item stratum drops to the SAMPLE axis (per-sample
    pairs by sampleIndex) when one exists — that resolution is the whole
    point, since a single saturating item's signal is exactly what pooling
    averages away.

    But the two are NOT the same estimand, and treating them as one was a
    real defect (review 2026-08-06). An item-level row estimates a quantity
    that generalizes over the items in its stratum. A within-item row
    estimates how ONE prompt's generations moved — a prompt-specific
    stochastic quantity, over samples that are exchangeable draws rather than
    a designed pairing. Running the two through one correction family let a
    within-item row emerge with a corrected p that reads exactly like a
    cross-item finding, which it can never be. So each row is now STAMPED
    with its estimand, and within-item rows are marked ``diagnostic`` and
    HELD OUT of the correction: they keep their estimate, interval and raw
    Wilcoxon p as a locator for which cell moved, and carry no adjusted p.
    (Held out, not modeled: pooling the two levels properly is a hierarchical
    model, and inventing one silently inside a CSV writer would be worse than
    the defect.)

    Corrections run per endpoint WITHIN each family, over that family's
    item-level rows only — never across families and never joined to the
    pooled family."""
    from . import study_stats
    item_ids = {prompt_id
                for cells in endpoints.values()
                for values in cells.values()
                for prompt_id in values}
    if not item_ids:
        return []
    families = _stratification_families(_item_factor_levels(records), item_ids)
    samples = _endpoint_sample_values(records, style=style)
    rows: list = []
    for family_name, strata in families:
        family_rows: dict[str, list] = {}
        for endpoint in sorted(endpoints):
            cells = endpoints[endpoint]
            base = cells.get("baseline", {})
            if not base:
                continue
            for stratum_label in sorted(strata):
                members = strata[stratum_label]
                for condition in sorted(cells):
                    if condition == "baseline":
                        continue
                    values = {prompt_id: value
                              for prompt_id, value in cells[condition].items()
                              if prompt_id in members}
                    paired = [prompt_id for prompt_id in sorted(values)
                              if prompt_id in base]
                    diffs = [values[prompt_id] - base[prompt_id]
                             for prompt_id in paired]
                    unit = "item"
                    if len(paired) == 1:
                        by_condition = samples.get(endpoint, {})
                        cond_samples = by_condition.get(condition, {}) \
                                                   .get(paired[0], {})
                        base_samples = by_condition.get("baseline", {}) \
                                                   .get(paired[0], {})
                        sample_diffs = [cond_samples[k] - base_samples[k]
                                        for k in sorted(cond_samples)
                                        if k in base_samples]
                        if len(sample_diffs) >= 2:
                            diffs = sample_diffs
                            unit = "sample"
                    if not diffs:
                        continue
                    row = study_stats.effect_row(condition, endpoint, diffs)
                    row.modality = modalities.get(condition, "")
                    row.stratify_by = family_name
                    row.stratum = stratum_label
                    row.unit = unit
                    row.estimand = ("withinItemSamples" if unit == "sample"
                                    else "itemLevel")
                    row.inference = ("diagnostic" if unit == "sample"
                                     else "corrected")
                    family_rows.setdefault(endpoint, []).append(row)
        for endpoint_rows in family_rows.values():
            # Only the item-level rows form a correction family. The
            # diagnostic within-item rows keep adjustedP and correction empty
            # — a blank cell reads as "not tested" to every strict reader on
            # both engines, which is exactly the claim.
            study_stats.apply_correction(
                [row for row in endpoint_rows if row.inference == "corrected"],
                method=method)
        for endpoint in sorted(family_rows):
            rows.extend(family_rows[endpoint])
    return rows



def promotion_decisions(manifest, rows, run_dir, promotion_mod):
    """Assemble per-concept screening evidence from single-concept conditions:
    the primary-endpoint effect, the dose-response over that concept's alpha
    grid, and the matched-norm random floor."""
    from . import study_stats

    def condition_meta(name: str):
        for condition in manifest.conditions:
            if condition.name == name and len(condition.slots) == 1:
                return (condition.slots[0].concept, condition.slots[0].alpha,
                        condition.control_type)
        return (None, None, None)

    primary = next((e for e in _PRIMARY_ENDPOINT_ORDER
                    if any(r.endpoint == e for r in rows)), None)
    if primary is None:
        return []
    primary_rows = [r for r in rows if r.endpoint == primary]
    random_floor: dict[str, float] = {}
    by_concept: dict[str, list[tuple[float, study_stats.EffectRow]]] = {}
    for row in primary_rows:
        concept, alpha, control_type = condition_meta(row.condition)
        if concept is None:
            continue
        if control_type == "randomMatchedNorm":
            previous = random_floor.get(concept)
            magnitude = abs(row.ci.mean)
            random_floor[concept] = max(previous, magnitude) if previous is not None \
                else magnitude
            continue
        by_concept.setdefault(concept, []).append((alpha, row))

    decisions = []
    for concept, dosed in sorted(by_concept.items()):
        dosed.sort(key=lambda pair: pair[0])
        # The concept's headline row: largest-|alpha| treatment cell.
        headline = max(dosed, key=lambda pair: abs(pair[0]))[1]
        dose = None
        if len(dosed) >= 2:
            dose = study_stats.dose_monotonicity(
                [alpha for alpha, _ in dosed], [row.ci.mean for _, row in dosed])
        candidate = promotion_mod.PromotionCandidate(
            concept=concept, condition=headline.condition, endpoint=primary,
            effect=headline, dose=dose,
            random_floor_effect=random_floor.get(concept),
            capability_passed=None,
            provenance={"sourceRun": os.path.basename(run_dir)})
        decisions.append(promotion_mod.decide(candidate, manifest.promotion_rule))
    return decisions
