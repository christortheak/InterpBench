"""Offline analysis and style rescoring. Never acquires a generation model.

Compatibility entry points in tasks.py delegate here; this module depends only
on evidence readers, admission policy, endpoint calculations and output writers.
"""
from __future__ import annotations
import csv
import json
import os
import sys
from typing import Callable
from . import choice_deltas, judicial, lifecycle_gates, paths, turn_endpoint
from .manifest import Manifest
from .task_inputs import load_prompts as _load_prompts
from .run_reporting import reasoning_style_summary as _reasoning_style_block
from .run_artifacts import latest_run, write_config_snapshot
from .study_admission import (advise_implicit_case_family, require_source_epoch,
                              stamped_experiment_hash, verify_or_warn)
from .analysis_endpoints import (condition_modalities, endpoint_values,
    key_records_by_transcript, promotion_decisions, stratified_effect_rows,
    transcript_level_diffs)


def analyze(name: str, root: str | None = None, source_run: str | None = None,
            *, model_provider=None, should_cancel: Callable[[], bool] | None = None,
            log=None, allow_unverified_epoch: bool = False,
            adjudicated_endpoint: str | None = None) -> str:
    """Statistics + reporting over a prior run: paired effect sizes with
    bootstrap CIs and Wilcoxon (effect-sizes.csv), the phase's multiple-
    comparison correction (BH-FDR for screen, Holm for confirm), alien-stance
    residuals against the pinned human baseline (alien-residuals.csv), the
    per-item paired choice deltas of the answer-token instrument
    (choice-deltas.csv — the citable version of the per-item Δ a viewer would
    otherwise derive), and the promoted-movers funnel artifact for
    screen-phase studies. Pure CPU — reads the immutable run directory,
    writes a new analyze run directory. Records whose run-time numeric parse
    was null are re-parsed under the manifest's pinned grammar first (the
    null-only endpoint rescue, stamped in endpoint-reparse.json).

    Epoch guard: the source run's stamped experiment hash must equal the live
    manifest's content hash; legacy unstamped runs need
    ``allow_unverified_epoch`` and are stamped ``epochUnverified: true``.

    ``adjudicated_endpoint`` names an external extraction campaign's
    per-record values for one numeric endpoint (see
    ``experiment/adjudication.py``). Verified against THIS source run —
    generations hash, full coverage, verbatim quote custody — then
    substituted in memory AFTER the rescue and BEFORE exclusions, and
    stamped separately (``adjudicated-endpoint.json`` +
    ``adjudication-divergence.csv``); ``endpoint-reparse.json`` is never
    touched. ``source_run`` is required with it — the CLI refuses without,
    because an adjudication is evidence about one specific run."""
    from . import (adjudication as adjudication_mod,
                   promotion as promotion_mod, reasoning_style,
                   residuals as residuals_mod, study_stats)
    _log = log or print
    manifest = Manifest.load(name, root)
    # Reasoning-style values are derived, not stored: recompute them from
    # each record's output through the pinned (hash-checked) taxonomy so
    # rs_<featureID> joins the same paired effect-size machinery.
    style = reasoning_style.load_pinned(manifest, root)
    run_dir = source_run or latest_run(name, root)
    if not run_dir:
        raise lifecycle_gates.refusing(
            lifecycle_gates.MISSING_PREREQUISITE,
            f"no prior run with generations found for '{name}' — run it first",
            repair=(f"steerlab-server experiment run {name} && "
                    f"steerlab-server experiment analyze {name}"))
    epoch_unverified, measurement_drift = require_source_epoch(
        "analyze", name, manifest, run_dir,
        allow_unverified_epoch=allow_unverified_epoch)
    if measurement_drift:
        _log(f"WARNING: '{name}' drifted from source run "
             f"'{os.path.basename(run_dir)}' in MEASUREMENT-side fields only "
             f"({measurement_drift}) — the generations are unaffected; "
             "analyzing proceeds under the LIVE settings and the output is "
             "stamped measurementDrift")
    if epoch_unverified:
        _log(f"WARNING: source run '{os.path.basename(run_dir)}' carries no "
             "experiment-hash stamp — analyzing it under allowUnverifiedEpoch; "
             "the output is stamped epochUnverified")
    records = []
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    # Records exist, but every one of them is the baseline: the paired
    # statistics have no contrast to compute, so this analysis is
    # structurally empty however many generations it read. A warning, not a
    # refusal — the run's own artifacts are still legitimate material — but
    # never silent (WP0 dry run #0, P0-2). Stderr, like the freeze-gate
    # warnings, so it survives a caller that only reads stdout for results.
    # Swift twin: the same line in ``ExperimentTasks.analyze``.
    if records and not any(r.get("condition") != "baseline" for r in records):
        print(f"WARNING: run '{os.path.basename(run_dir)}' contains only "
              "BASELINE records — there is no non-baseline condition to pair "
              "against, so this analysis will produce no effect sizes. Check "
              "the study's conditions before citing it.", file=sys.stderr)

    # Endpoint rescue (2026-08-10, an anchoring run): a record whose
    # run-time numeric parse came back null is re-parsed from its stored
    # output under the manifest's pinned grammar as THIS engine implements
    # it — a grammar fix (e.g. accepting a spelled-out formal register, "ten
    # years and six months") reaches finished runs without regenerating
    # them. Null-only, deliberately: run-time parses stay authoritative
    # (re-parsing everything would let a NEW first match retroactively
    # overwrite a correct old one — verified on real records, where a
    # spelled-out threshold quoted earlier in the text would beat the
    # actual answer), and a record no grammar parses stays unparsed,
    # never guessed. generations.jsonl is untouched (runs are immutable);
    # rescued values feed this analysis' exclusions and endpoints only, and
    # the analyze output stamps what happened (endpoint-reparse.json).
    # Resolution mirrors the run path: a declared registry parser (drifted
    # pin refuses) wins; otherwise caseFamily sentencing uses the built-in.
    numeric_parser = None
    if manifest.numeric_parser:
        from . import parser_registry
        numeric_parser = parser_registry.resolve(manifest.numeric_parser, root)
        if (manifest.parser_registry_hash
                and numeric_parser.registry_hash != manifest.parser_registry_hash):
            raise RuntimeError(
                f"parser registry '{parser_registry.REGISTRY_FILE}' drifted "
                f"from the pinned hash (have "
                f"{numeric_parser.registry_hash[:12]}…, pinned "
                f"{manifest.parser_registry_hash[:12]}…)")
    rescue_parse, rescue_parser_stamp = None, None
    if numeric_parser is not None:
        rescue_parse = numeric_parser.parse
        rescue_parser_stamp = numeric_parser.provenance()
    elif manifest.case_family == "sentencing":
        rescue_parse = judicial.parse_months
        rescue_parser_stamp = {"name": "builtin:sentencing",
                               "kind": "durationMonths"}
        # The DEPRECATED trigger picked the rescue grammar. Logged only here:
        # the analyze run directory does not exist yet at this point, and the
        # durable half of this analysis' record is `endpoint-reparse.json`,
        # which already stamps `builtin:sentencing` as the parser it used.
        advise_implicit_case_family(True, None, _log, write_file=False)
    reparse_stamp = None
    if rescue_parse is not None:
        unparsed = [r for r in records
                    if "error" not in r and "instrument" not in r
                    and "output" in r and r.get("parsedMonths", 0.0) is None]
        rescued = 0
        for record in unparsed:
            fresh = rescue_parse(record.get("output") or "")
            if fresh is not None:
                record["parsedMonths"] = fresh
                rescued += 1
        reparse_stamp = {
            "endpoint": "parsedMonths",
            "parser": rescue_parser_stamp,
            "unparsedRecords": len(unparsed),
            "rescuedRecords": rescued,
            "stillUnparsed": len(unparsed) - rescued,
            "note": ("Null-only endpoint rescue: records whose run-time "
                     "parse was null were re-parsed from their stored "
                     "output under the manifest's pinned grammar as this "
                     "engine version implements it. Run-time parses stay "
                     "authoritative, generations.jsonl is untouched, and a "
                     "record no grammar parses stays unparsed — never "
                     "guessed. Rescued values feed this analysis only."),
        }
        if unparsed:
            _log(f"endpoint rescue: {rescued}/{len(unparsed)} previously "
                 f"unparsed parsedMonths record(s) re-parsed under "
                 f"'{rescue_parser_stamp['name']}'"
                 + (f"; {len(unparsed) - rescued} still unparsed"
                    if rescued < len(unparsed) else ""))

    # Adjudicated-endpoint intake (open-issues §10, 2026-08-18): an external
    # extraction campaign's per-record values REPLACE the endpoint the run
    # parsed, once they have been verified against this exact run. Separate
    # pass, separate stamp, deliberately: the rescue above is null-only
    # precisely because a run-time parse must never be silently overwritten,
    # and an adjudication overwrites by design — so it carries its own
    # custody ladder (generations hash, per-row validity, verbatim quote
    # containment, exhaustive coverage) and its own divergence accounting
    # against the value analyze would otherwise have used. It runs AFTER the
    # rescue (a record both rescued and adjudicated is accounted against its
    # rescued value) and BEFORE exclusions and _endpoint_values, so both see
    # adjudicated values. generations.jsonl is untouched.
    adjudication_stamp = None
    adjudication_rows: list = []
    if adjudicated_endpoint:
        adjudication_document, adjudication_sha = adjudication_mod.load(
            adjudicated_endpoint)
        records, adjudication_stamp, adjudication_rows = adjudication_mod.apply(
            records, adjudication_document, file_sha256=adjudication_sha,
            run_dir=run_dir,
            instructions_dir=os.path.dirname(
                os.path.abspath(adjudicated_endpoint)),
            log=_log)

    # Declared exclusion rules join HERE — records are dropped from the
    # paired statistics only (pairwise deletion falls out of the promptID
    # join below), never from generations.jsonl, and the stamp
    # (exclusions.json) records what was active, what each rule excluded
    # per condition, the surviving N, and the declared scope. Scope is
    # allRecordTypes (the apply default): deterministic instrument
    # readouts are considered too — endpoint rules read endpoints the
    # record itself carries (e.g. ordinalPosition), and a cell whose every
    # sampled record failed its attention check drops its instrument
    # readouts (choiceLogOdds / ordinalPosition endpoints) with it. No
    # rules declared = today's behavior byte-for-byte.
    from . import exclusions as exclusions_mod
    exclusion_rules = exclusions_mod.declared_rules(manifest.raw)
    exclusion_stamp = None
    if exclusion_rules:
        checks: dict[str, dict] = {}
        if exclusions_mod.needs_checks(exclusion_rules):
            if not manifest.task_prompts_hash:
                # WP0 step 8: the deferred cross-engine-twinned message from
                # c86ce53 gets its id on BOTH engines. The STRING is unchanged
                # and stays byte-identical to Swift's
                # `ExclusionEngine.pinRequiredMessage` (asserted on both
                # sides); only the gate id and the repair are new.
                raise lifecycle_gates.refusing(
                    lifecycle_gates.MISSING_PREREQUISITE,
                    exclusions_mod.PIN_REQUIRED_MESSAGE,
                    repair=exclusions_mod.PIN_REQUIRED_REPAIR)
            checks = exclusions_mod.attention_checks(
                _load_prompts(manifest, None, root))
            if not checks:
                raise RuntimeError(exclusions_mod.NO_CHECKS_MESSAGE)
        records, exclusion_stamp = exclusions_mod.apply(
            records, exclusion_rules, checks)
        _log(f"exclusions: {exclusion_stamp['excludedRecords']} record(s) "
             f"excluded by {len(exclusion_rules)} declared rule(s); "
             "surviving N per condition: "
             + ", ".join(f"{c}={n}" for c, n
                         in exclusion_stamp["survivingN"].items()))

    # D1, multi-agent only: turns are NOT independent observations — turn k is
    # conditioned on turns 1..k-1, and after turn 1 the arms diverge, so what
    # pairs across conditions is script POSITION, not matched input. Carry the
    # replicate in the pairing key (otherwise the same turn id from different
    # transcripts collides and gets averaged into one cell), then aggregate
    # each transcript to its mean difference before any test runs.
    clustered = manifest.study_kind == "multiAgent"
    # Declared panel endpoints (Wave-2). Collected BEFORE the transcript
    # re-keying below, which rewrites promptID — the endpoint aggregation
    # reads seat/condition/replicate, and should not depend on a key another
    # concern owns. Nothing is parsed here: these are the runner's write-time
    # stamps, verbatim. Free-text mining never happens downstream of the run.
    endpoint_rows = turn_endpoint.csv_rows(records)
    endpoint_counts = turn_endpoint.counts(records)
    # Per-item choice deltas (Phase 3 of the results-explorer plan). Collected
    # here for the same reason as the endpoint stamps: the baseline join is by
    # promptID, and the transcript re-keying below rewrites it. Exclusions have
    # already been applied, so a dropped readout drops from this table too.
    # Declared-target map from the PINNED task file (open-issues #6): the
    # exact record of which items declared a choice target, so the
    # choiceLogOdds endpoint and the choice-deltas artifact never read a
    # run-time-synthesized target (an ordinalScale item's scale minimum) as
    # a declared endpoint — and, symmetrically, so a mixed instrument like
    # s4-framings (declared A/B target AND an ordinal readout on the same
    # record) keeps its legitimate endpoint. Unloadable prompts fall back to
    # the per-record heuristic inside the consumers.
    declared_targets = None
    try:
        if manifest.task_prompts_hash:
            declared_targets = {str(p.get("id")): bool(p.get("target"))
                                for p in _load_prompts(manifest, None, root)}
    except Exception as exc:
        _log(f"declared-target map unavailable ({exc}); choice endpoints "
             "fall back to per-record classification")
    choice_delta_rows, choice_delta_summary = choice_deltas.rows(
        records, declared_targets=declared_targets)
    if clustered:
        records = key_records_by_transcript(records)
    # Parser kind for endpoint-label honesty (see _endpoint_values), from
    # the parser already resolved for the endpoint rescue above.
    numeric_parser_kind = (numeric_parser.kind
                           if numeric_parser is not None else None)
    endpoints = endpoint_values(records, style=style,
                                 numeric_parser_kind=numeric_parser_kind,
                                 declared_targets=declared_targets)
    baseline = {endpoint: cells.get("baseline", {}) for endpoint, cells in endpoints.items()}
    modalities = condition_modalities(manifest, root)
    rows: list[study_stats.EffectRow] = []
    diffs_index: dict[tuple[str, str], list[float]] = {}
    skipped_for_replication = False
    for endpoint, cells in endpoints.items():
        for condition, values in cells.items():
            if condition == "baseline":
                continue
            base = baseline.get(endpoint, {})
            if clustered:
                diffs = transcript_level_diffs(values, base)
                # One transcript per arm is a point estimate, not an interval.
                if len(diffs) < 2:
                    skipped_for_replication = True
                    continue
            else:
                diffs = [values[pid] - base[pid] for pid in sorted(values) if pid in base]
            if not diffs:
                continue
            row = study_stats.effect_row(condition, endpoint, diffs)
            row.modality = modalities.get(condition, "")
            rows.append(row)
            diffs_index[(condition, endpoint)] = diffs
    method = "holm" if manifest.phase == "confirm" else "bh"
    by_endpoint: dict[str, list[study_stats.EffectRow]] = {}
    for row in rows:
        by_endpoint.setdefault(row.endpoint, []).append(row)
    for family in by_endpoint.values():
        study_stats.apply_correction(family, method=method)
    # Per-cell strata beside the pooled rows (same CSV, extra rows): pooling
    # across items has both hidden a real single-cell effect behind saturated
    # cells and manufactured pooled effects from one cell's parse garbage.
    # Pooled rows keep their exact semantics and correction family; each
    # stratified family (promptID, declared factors, their cross) is
    # corrected independently. Clustered multi-agent runs skip stratification
    # — their promptIDs were re-keyed to transcript positions above and the
    # transcript aggregation is the honest unit there.
    stratified_rows: list = []
    if not clustered:
        stratified_rows = stratified_effect_rows(
            records, endpoints, style=style, method=method,
            modalities=modalities)

    out = paths.make_unique_run_directory(f"exp-{name}-analyze", root)
    write_config_snapshot(manifest, out, "analyze",
                           notes=({**({"epochUnverified": True}
                                      if epoch_unverified else {}),
                                   **({"measurementDrift": measurement_drift}
                                      if measurement_drift else {}),
                                   # The canonical per-run stamp says the
                                   # endpoint values were substituted, so a
                                   # reader of config.json alone cannot miss
                                   # it (notes is the established extension
                                   # point).
                                   **(adjudication_mod.notes_block(
                                       adjudication_stamp)
                                      if adjudication_stamp else {})}
                                  or None))
    with open(os.path.join(out, "source-run.txt"), "w", encoding="utf-8") as handle:
        handle.write(os.path.basename(run_dir) + "\n")
    if clustered:
        # Say what an effect row averages over. `n` counts TRANSCRIPTS here,
        # not turns, and a reader cannot tell which from the number alone.
        with open(os.path.join(out, "unit-of-analysis.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"unitOfAnalysis": "transcript",
                       "reason": "turns within a transcript are dependent; "
                                 "each transcript is reduced to its mean paired "
                                 "difference before testing",
                       "skippedForSingleTranscript": skipped_for_replication},
                      handle, indent=2, sort_keys=True)
        if skipped_for_replication:
            _log("effect sizes: some endpoints skipped — 1 transcript per "
                 "condition supports a point estimate but no interval. "
                 "Re-run with samplesPerItem > 1.")
    if endpoint_rows:
        # Zero stamps ⇒ no file and no section: an analyze over a panel that
        # declared nothing must not grow an empty table implying it did.
        turn_endpoint.write_csv(
            os.path.join(out, "panel-endpoints.csv"), endpoint_rows)
        with open(os.path.join(out, "panel-endpoints.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"endpoints": endpoint_counts,
                       "records": len(endpoint_rows),
                       "unparsed": sum(row[-1] for row in endpoint_rows)},
                      handle, indent=2, sort_keys=True)
        _log(f"panel endpoints: {len(endpoint_rows)} stamped turn(s), "
             f"{sum(row[-1] for row in endpoint_rows)} unparsed → "
             "panel-endpoints.csv")
    if choice_delta_summary["conditions"]:
        # Same rule as the panel endpoints above: a run with no non-baseline
        # choice readouts grows no table implying it had some. When there ARE
        # readouts the file is written even if every one of them was skipped —
        # the skip counts are the finding in that case.
        choice_deltas.write_csv(
            os.path.join(out, "choice-deltas.csv"), choice_delta_rows)
        with open(os.path.join(out, "choice-deltas.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(choice_delta_summary, handle, indent=2, sort_keys=True)
        _log(f"choice deltas: {len(choice_delta_rows)} paired item(s) across "
             f"{len(choice_delta_summary['conditions'])} condition(s), "
             f"{sum(block['flipped'] for block in choice_delta_summary['conditions'].values())} "
             f"flip(s), {choice_delta_summary['skippedNoBaseline']} skipped "
             "(no baseline partner) → choice-deltas.csv")
        if choice_delta_summary["skippedNoTargetValue"]:
            _log(f"choice deltas: {choice_delta_summary['skippedNoTargetValue']} "
                 "readout(s) skipped — no log-odds entry for the item's own "
                 "target option")
    if exclusion_stamp is not None:
        # The exclusion stamp (cross-engine shape; Swift embeds the same
        # object in analysis.json AND writes this file): active rules with
        # plain-language descriptions, per-condition per-rule exclusion
        # counts, surviving N, and the pairwise-deletion note.
        with open(os.path.join(out, "exclusions.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(exclusion_stamp, handle, indent=2, sort_keys=True)
    if reparse_stamp is not None:
        # The endpoint-rescue stamp: which grammar re-parsed the null
        # records, how many were rescued, how many stayed unparsed. Written
        # whenever a numeric grammar applied — zero rescues included, so an
        # analysis that changed nothing says so explicitly.
        with open(os.path.join(out, "endpoint-reparse.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(reparse_stamp, handle, indent=2, sort_keys=True)
    if adjudication_stamp is not None:
        # The adjudication's own two artifacts, never mixed into the
        # rescue's: the summary stamp (file hash, instructions block,
        # divergence counts, per-condition breakdown) and the FULL row-level
        # list of records whose adjudicated value differs from the value
        # analyze would otherwise have used — that list is analysis
        # evidence, not a sample. The CSV is written even when nothing
        # diverged (header only), like the rescue's zero-rescue stamp.
        adjudication_mod.write_stamp(out, adjudication_stamp)
        adjudication_mod.write_divergence_csv(out, adjudication_rows)
        _log(f"adjudicated endpoint: {len(adjudication_rows)} divergent "
             f"record(s) → {adjudication_mod.DIVERGENCE_FILENAME}")
    if epoch_unverified:
        with open(os.path.join(out, "epoch-unverified.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"epochUnverified": True,
                       "sourceRun": os.path.basename(run_dir)}, handle,
                      indent=2, sort_keys=True)
    if measurement_drift:
        # Tolerated measurement-side drift is never silent (twin of the
        # epoch-unverified sidecar): which fields differed, verbatim.
        with open(os.path.join(out, "measurement-drift.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"measurementDrift": measurement_drift,
                       "sourceRun": os.path.basename(run_dir)}, handle,
                      indent=2, sort_keys=True)
    # D3: distance-from-boundary diagnostics, per condition. A large
    # joint-logprob margin means the FLIP RATE has poor sensitivity there —
    # an intervention can move the log-odds a long way without flipping any
    # item — while the log-odds itself keeps moving continuously. Reporting
    # that as "saturation" invites the wrong conclusion; true numerical
    # saturation is the separately counted clamp incidence.
    from . import choice_margin
    margin_report = {}
    by_condition_for_margins: dict[str, list[dict]] = {}
    for record in records:
        if record.get("optionLogprobs"):
            by_condition_for_margins.setdefault(
                record.get("condition", ""), []).append(record)
    for condition, items in by_condition_for_margins.items():
        block = choice_margin.diagnostics(items)
        if block.get("scoredItems"):
            margin_report[condition] = block
    if margin_report:
        with open(os.path.join(out, "choice-margins.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(margin_report, handle, indent=2, sort_keys=True)
        for condition, block in sorted(margin_report.items()):
            _log(f"{condition}: {block['interpretation']}")

    from . import instrumentation_evidence
    if any('probeMeasurements' in r or 'interventionDecisions' in r for r in records):
        with open(os.path.join(out, 'instrumentation-summary.json'), 'w', encoding='utf-8') as handle:
            json.dump(instrumentation_evidence.summarize(records), handle, indent=2, sort_keys=True, allow_nan=False)

    with open(os.path.join(out, "effect-sizes.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(study_stats.EFFECT_SIZES_HEADER)
        for row in rows:
            writer.writerow(row.as_csv_row())
        # Stratified rows ride after every pooled row; readers that group by
        # (condition, endpoint) alone must filter on stratifyBy == "pooled"
        # (the semantic viewers do). Each stratified row also declares its
        # ``estimand`` and what its p-values are licensed for (``inference``);
        # a within-item-sample row is diagnostic and carries no adjusted p.
        for row in stratified_rows:
            writer.writerow(row.as_csv_row())

    # Behavioral-fingerprint table: the same effect rows reshaped (condition ×
    # endpoint, with modality) for cross-condition comparison — no new stats.
    with open(os.path.join(out, "fingerprints.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(study_stats.FINGERPRINTS_HEADER)
        for csv_row in study_stats.fingerprint_csv_rows(rows):
            writer.writerow(csv_row)

    if manifest.human_baseline is not None:
        base_path = manifest.human_baseline.path
        if not os.path.isabs(base_path):
            base_path = os.path.join(paths.project_root() if root is None else root,
                                     base_path)
        human = residuals_mod.load_human_baseline(base_path, manifest.human_baseline.hash)
        residual_rows = residuals_mod.residual_rows(rows, human)
        residuals_mod.write_alien_residuals_csv(
            os.path.join(out, "alien-residuals.csv"), residual_rows)
        _log(f"alien residuals: {len(residual_rows)} rows")

    if manifest.promotion_rule is not None and manifest.phase == "screen":
        decisions = promotion_decisions(manifest, rows, run_dir, promotion_mod)
        promotion_mod.write_promoted_movers(
            os.path.join(out, "promoted-movers.json"), decisions,
            experiment=name, experiment_hash=manifest.content_hash(),
            rule=manifest.promotion_rule)
        _log(f"promotion: {sum(1 for d in decisions if d.promoted)}/{len(decisions)} promoted")

    diagnostic_strata = sum(1 for row in stratified_rows
                            if row.inference == "diagnostic")
    _log(f"analyze ({len(rows)} effect rows"
         + (f" + {len(stratified_rows)} stratified" if stratified_rows else "")
         + (f", {diagnostic_strata} of them within-item diagnostics "
            "(uncorrected — a single item's samples support no cross-item "
            "claim)" if diagnostic_strata else "")
         + f") → {out}")
    return out



def rescore_style(name: str, root: str | None = None, source_run: str | None = None,
                  *, allow_unverified_epoch: bool = False, log=None) -> str:
    """Post-hoc reasoning-style scoring (Swift twin ``rescoreStyle``):
    recompute ``rs_<featureID>`` values for an EXISTING completed run's
    sampled generations from the manifest's pinned taxonomy — pure CPU, no
    model — writing ``reasoning-style.csv`` + ``reasoning-style.json`` into a
    NEW immutable rescore run directory. The source run is NEVER mutated (run
    immutability), and the epoch guard applies exactly as for analyze."""
    from . import reasoning_style
    _log = log or print
    manifest = Manifest.load(name, root)
    verify_or_warn(manifest, root)
    style = reasoning_style.load_pinned(manifest, root)
    if style is None:
        raise RuntimeError(
            f"experiment '{name}' pins no reasoning-style taxonomy — pin one "
            "first (reasoningStyleTaxonomyPath + reasoningStyleTaxonomyHash; "
            "see prompts/templates/reasoning-style/)")
    run_dir = source_run or latest_run(name, root)
    if not run_dir:
        raise lifecycle_gates.refusing(
            lifecycle_gates.MISSING_PREREQUISITE,
            f"no prior run with generations found for '{name}' — run it first",
            repair=(f"steerlab-server experiment run {name} && "
                    f"steerlab-server experiment rescore-style {name}"))
    epoch_unverified, measurement_drift = require_source_epoch(
        "rescore-style", name, manifest, run_dir,
        allow_unverified_epoch=allow_unverified_epoch)
    if measurement_drift:
        _log(f"WARNING: '{name}' drifted from source run "
             f"'{os.path.basename(run_dir)}' in MEASUREMENT-side fields only "
             f"({measurement_drift}) — the generations are unaffected; "
             "rescoring proceeds under the LIVE settings and the output is "
             "stamped measurementDrift")
    if epoch_unverified:
        _log(f"WARNING: source run '{os.path.basename(run_dir)}' carries no "
             "experiment-hash stamp — rescoring it under allowUnverifiedEpoch; "
             "the output is stamped epochUnverified")
    records: list[dict] = []
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    sampled = [r for r in records
               if "error" not in r and "instrument" not in r and "output" in r]
    if not sampled:
        raise RuntimeError(
            f"run '{os.path.basename(run_dir)}' has no sampled generations to rescore")

    # NEW immutable artifacts only — never a byte into the source run.
    out = paths.make_unique_run_directory(f"exp-{name}-rescore-style", root)
    write_config_snapshot(manifest, out, "rescore-style",
                           notes=({**({"epochUnverified": True}
                                      if epoch_unverified else {}),
                                   **({"measurementDrift": measurement_drift}
                                      if measurement_drift else {})} or None))
    with open(os.path.join(out, "source-run.txt"), "w", encoding="utf-8") as handle:
        handle.write(os.path.basename(run_dir) + "\n")
    feature_ids = style.taxonomy.feature_ids
    with open(os.path.join(out, "reasoning-style.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "seed", "promptIndex", "promptID"]
                        + [f"rs_{fid}" for fid in feature_ids])
        for index, record in enumerate(sampled):
            values = style.taxonomy.score(record.get("output", ""))
            writer.writerow([
                record.get("condition", ""),
                record.get("seed", 0),
                record.get("promptIndex", index),
                record.get("promptID", ""),
            ] + [values.get(fid, 0.0) for fid in feature_ids])
    by_condition: dict[str, list[dict]] = {}
    for record in sampled:
        by_condition.setdefault(record.get("condition", ""), []).append(record)
    report = {
        "experiment": name,
        "experimentHash": manifest.content_hash(),
        "sourceRun": os.path.basename(run_dir),
        # The shared reader already falls back to config.json's stamp.
        "sourceRunExperimentHash": stamped_experiment_hash(run_dir),
        "taxonomy": style.taxonomy.name,
        "taxonomyHash": style.hash,
        "taxonomyFile": style.path,
        # Same status stamp as report.json's per-condition block: style
        # features are a diagnostic/manipulation check, never an outcome
        # endpoint (docs/METHODS.md).
        "diagnosticOnly": True,
        "conditions": {
            cond: {"features": _reasoning_style_block(items, style)["features"]}
            for cond, items in by_condition.items()
        },
    }
    if epoch_unverified:
        report["epochUnverified"] = True
    if measurement_drift:
        # Tolerated measurement-side drift is never silent: which fields
        # differed from the source run's epoch, verbatim.
        report["measurementDrift"] = measurement_drift
    with open(os.path.join(out, "reasoning-style.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    _log(f"rescore-style ({len(sampled)} generations, {len(feature_ids)} "
         f"feature(s), {len(by_condition)} condition(s)) → {out}")
    return out
