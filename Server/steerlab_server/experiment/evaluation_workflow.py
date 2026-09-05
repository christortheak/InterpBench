"""Coordinate paired judging and response coding with explicit provider/release callbacks.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
import json
import os
from contextlib import ExitStack
from typing import Callable
from . import lifecycle_gates, paths
from . import deferred_evaluation
from . import evaluation_evidence
from . import judge_dispatch
from . import judge_resources
from . import manifest as manifest_module
from . import rubric_inputs
from . import run_artifacts
from . import run_config
from . import run_status
from . import study_admission
from . import task_inputs


def _evaluate_response_coding(name: str, manifest: manifest_module.Manifest, schema,
                              run_dir: str, generations: list[dict],
                              rubric_hash: str | None,
                              rubric_file: str | None, roster,
                              root: str | None, epoch_unverified: bool,
                              measurement_drift: str | None,
                              evaluation_source: str | None,
                              exclusion_stamp: dict | None,
                              model_provider, _log, *, model_release=None,
                              study_model_generates_later: bool = False,
                              subsample=None) -> str:
    """The per-response coding instrument's evaluate body (2026-08-04;
    Swift twin: ``ExperimentTasks.runResponseCoding``).

    Every sampled-text record — baseline INCLUDED — goes to every judge
    individually and blinded (the coder sees the task prompt and one
    response, never the condition, never a second response). Codes are
    validated against the rubric's declared schema (retry once, then refuse
    — invented data is never recorded), streamed row-by-row to
    ``codings.jsonl``, and summarized in ``coding-report.json`` with
    per-condition per-field aggregates, engine-computed word counts, and
    per-field inter-judge agreement (percent + Cohen's kappa for
    categorical fields — the same statistic the K&Z paper used to validate
    its coders). There is no pairing and no winner anywhere on this path.

    ``subsample`` (an ``evaluate_subsample.SubsampleRequest``, 2026-08-29)
    codes a seeded, stratified DRAW from the source run instead of all of it
    — the preregistered design the instrument previously had no spelling for.
    The draw runs BEFORE the run directory is minted, so its refusals (an
    over-ask against a condition's population) write nothing, and its stamp
    rides in the run config and the coding report while every human line
    says ``N of M (seeded subsample)``.
    """
    from . import evaluate_subsample, response_coding
    codeable = [g for g in generations
                if "instrument" not in g and "error" not in g
                and "output" in g]
    if not codeable:
        raise RuntimeError(response_coding.NO_CODEABLE_MESSAGE)
    sampling: dict | None = None
    source_total = len(codeable)
    if subsample is not None:
        # Before `make_unique_run_directory`, deliberately: an over-ask is a
        # power-computation error the caller must see, and a refusal that
        # leaves a run directory behind has already broken the rule that
        # refusals never write.
        codeable, sampling = evaluate_subsample.select(
            codeable, subsample, program="steerlab-server")
    coded_phrase = evaluate_subsample.coded_phrase(sampling, source_total)
    out = paths.make_unique_run_directory(f"exp-{name}-evaluate", root)
    notes: dict = {}
    if epoch_unverified:
        notes["epochUnverified"] = True
    if sampling is not None:
        notes["sampling"] = sampling
    run_config.write_run_config(out, "evaluate", model_id=manifest.model_id,
                     revision=manifest.model_revision, experiment=name,
                     experiment_hash=manifest.content_hash(),
                     notes=notes or None)
    if exclusion_stamp is not None:
        with open(os.path.join(out, "exclusions.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(exclusion_stamp, handle, indent=2, sort_keys=True)
    status = run_status.RunStatus(out, stage="evaluate", experiment=name,
                       source_run=os.path.basename(run_dir),
                       expected=[ref.name for ref in roster],
                       item_label="coding")
    status.write()
    _log(f"coding {coded_phrase} × {len(roster)} judge(s) "
         f"under perResponseCoding rubric "
         f"'{rubric_file or '(inline draft)'}'")
    if sampling is not None:
        _log(f"seeded subsample: {sampling['samplePerCondition']} record(s) "
             f"per condition at seed {sampling['sampleSeed']} — "
             f"{sampling['rule']}")

    rows: list[dict] = []
    judge_details: list[dict] = []
    codings_path = os.path.join(out, "codings.jsonl")
    try:
        with open(codings_path, "w", encoding="utf-8") as codings_handle:
            for index, ref in enumerate(roster):
                # One model load per judge COLUMN (same rule as paired
                # evaluate): the stack closes at the end of each iteration,
                # and the release seam below frees the finished column's
                # container BEFORE this one loads, so two judge models are
                # never resident at once.
                judge_resources.release_models_for_judge(
                    model_release, roster, index,
                    study_model=manifest.model_id,
                    study_revision=manifest.model_revision,
                    study_dtype=manifest.dtype,
                    study_model_generates_later=study_model_generates_later,
                    _log=_log)
                with ExitStack() as judge_stack:
                    complete_fn, requested_model, holder = judge_resources.coder_callable(
                        ref, model_provider, study_model=manifest.model_id,
                        study_revision=manifest.model_revision,
                        stack=judge_stack)
                    judge_noncompliant = 0
                    for g in codeable:
                        sample_index = g.get("sampleIndex") or 0
                        try:
                            result = response_coding.valid_codes(
                                complete_fn, schema, g.get("output", ""),
                                g.get("prompt") or None,
                                judge_label=f"'{ref.name}'",
                                item_label=(f"record {g.get('condition')}/"
                                            f"{g.get('promptID')}"
                                            f"[{sample_index}]"),
                                on_invalid=status.note_invalid_response)
                        except response_coding.paired_judge.JudgeNoncompliant as exc:
                            # Same policy as the paired judge (Christian,
                            # 2026-08-09): one record the coder ANSWERS but
                            # will not code becomes a recorded, classifiable
                            # ROW — codes: None, excluded from aggregates
                            # and agreement — instead of aborting hours of
                            # completed work. Transport errors still fail
                            # the session (resume machinery); systemic
                            # noncompliance still fails too: see the cap
                            # check after this judge's column.
                            judge_noncompliant += 1
                            row = {
                                "experiment": name,
                                "condition": g.get("condition"),
                                "promptID": g.get("promptID"),
                                "sampleIndex": sample_index,
                                "seed": g.get("seed"),
                                "codes": None,
                                "noncompliant": True,
                                "noncomplianceReason": str(exc)[:2000],
                                "judge": ref.name,
                                "judgeKind": ref.kind,
                                "judgeModel": requested_model,
                            }
                            rows.append(row)
                            codings_handle.write(json.dumps(row) + "\n")
                            codings_handle.flush()
                            status.note_item()
                            continue
                        row = {
                            "experiment": name,
                            "condition": g.get("condition"),
                            "promptID": g.get("promptID"),
                            "sampleIndex": sample_index,
                            "seed": g.get("seed"),
                            "wordCount": response_coding.word_count(
                                g.get("output", "")),
                            "codes": result["codes"],
                            "briefReason": result["briefReason"],
                            "judge": ref.name,
                            "judgeKind": ref.kind,
                            "judgeModel": requested_model,
                        }
                        # Keys the coder invented, kept verbatim and kept OUT
                        # of the measurement — absent from the row entirely
                        # when it invented nothing. Nothing aggregates this.
                        if result.get("undeclaredCodes"):
                            row["undeclaredCodes"] = result["undeclaredCodes"]
                        if result.get("provider"):
                            row["judgeProvider"] = result["provider"]
                        if ref.kind == "local" and holder.get("revision"):
                            row["judgeRevision"] = holder["revision"]
                        rows.append(row)
                        # Flushed per row: a killed worker still leaves
                        # every coding it finished.
                        codings_handle.write(json.dumps(row) + "\n")
                        codings_handle.flush()
                        status.note_item()
                    from . import paired_judge as _pj
                    if (judge_noncompliant and codeable
                            and judge_noncompliant / len(codeable)
                            > _pj.NONCOMPLIANCE_CAP):
                        raise RuntimeError(
                            f"judge '{ref.name}' was noncompliant on "
                            f"{judge_noncompliant} of {len(codeable)} "
                            f"record(s) (> "
                            f"{_pj.NONCOMPLIANCE_CAP:.0%} cap) — "
                            "systemic coder failure, not flakiness. Every "
                            "row (including the noncompliant ones, with "
                            "raw reasons) is in codings.jsonl; fix or swap "
                            "the judge, then re-run evaluate")
                    detail = {"name": ref.name, "kind": ref.kind,
                              "requestedModel": requested_model,
                              "actualModel": holder.get("actual")}
                    if holder.get("revision"):
                        detail["revision"] = holder["revision"]
                    if holder.get("actualDtype"):
                        detail["actualDtype"] = holder["actualDtype"]
                    if judge_noncompliant:
                        detail["noncompliantCodings"] = judge_noncompliant
                    judge_details.append(detail)
                    status.note_judge_complete(ref.name)
                    coded_count = len(codeable) - judge_noncompliant
                    _log(f"judge '{ref.name}' coded "
                         + (f"{coded_count} of {source_total} record(s) "
                            "(seeded subsample)" if sampling
                            else f"{coded_count} record(s)")
                         + (f" ({judge_noncompliant} noncompliant, kept as "
                            "rows for review)" if judge_noncompliant
                            else ""))
    except BaseException as exc:
        status.fail(exc)
        _log(f"coding evaluate FAILED after {status.judgment_count} "
             f"coding(s) — rows kept in {os.path.basename(out)}; no "
             "coding report written")
        raise

    report = {
        "mode": "perResponseCoding",
        "experiment": name,
        "experimentHash": manifest.content_hash(),
        "sourceRun": os.path.basename(run_dir),
        "judges": [ref.name for ref in roster],
        "judgeModel": ", ".join(d["requestedModel"] for d in judge_details),
        "judgeDetails": judge_details,
        "judgeRubricFile": rubric_file,
        "judgeRubricHash": rubric_hash,
        "fields": [
            {"name": f.name, "type": f.type, "optional": f.optional,
             **({"values": list(f.values)} if f.values else {})}
            for f in schema.fields
        ],
        "codings": len(rows),
        "conditions": response_coding.aggregate_conditions(rows, schema),
        "evaluationSource": evaluation_source,
    }
    # Absent-with-reason rather than empty when ONE coder coded the run:
    # there is no pair to compare, which is not the same fact as a pair that
    # agreed about nothing. Swift twin: `CodingReport.fieldAgreement` is
    # optional and omitted on the same rule.
    if len(roster) >= 2:
        report["fieldAgreement"] = response_coding.field_agreement(
            rows, schema, [ref.name for ref in roster])
    else:
        report["fieldAgreementAbsentReason"] = \
            response_coding.SINGLE_CODER_AGREEMENT_ABSENT_REASON
    noncompliant_total = sum(1 for r in rows if r.get("noncompliant"))
    if noncompliant_total:
        # Nonzero-only, like the paired judge's noncompliantJudgments: these
        # rows carry no codes and sit outside every aggregate — the report
        # must say the columns are incomplete and by how much.
        report["noncompliantCodings"] = noncompliant_total
    if epoch_unverified:
        report["epochUnverified"] = True
    if measurement_drift:
        # Tolerated measurement-side drift is never silent: which fields
        # differed from the source run's epoch, verbatim.
        report["measurementDrift"] = measurement_drift
    if exclusion_stamp is not None:
        report["exclusions"] = exclusion_stamp
    if sampling is not None:
        # Additive and LOUD: absent means the full corpus was coded, so every
        # report written before this existed reads back unchanged, and a
        # report that carries the block cannot be mistaken for a full-corpus
        # coding by a reader who only looks at `codings`.
        report["sampling"] = sampling
    with open(os.path.join(out, "coding-report.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    status.complete()
    if sampling is not None:
        _log(f"coded {coded_phrase} at seed {sampling['sampleSeed']} — this "
             "report covers a SUBSAMPLE, not the full corpus")
    _log(f"coding evaluation artifacts: {out}")
    return out


def evaluate(name: str, root: str | None = None, source_run: str | None = None,
             *, model_provider=None, should_cancel: Callable[[], bool] | None = None,
             log=None, allow_unverified_epoch: bool = False,
             max_loaded: int | None = None,
             defer_local_judges: bool = False,
             resume_from: str | None = None,
             model_release=None,
             study_model_generates_later: bool = False,
             sample_per_condition=None,
             sample_seed=None) -> str:
    """Paired-judge evaluation over a prior run's generations (parallel to the
    Swift evaluate task). Judges with the manifest's PINNED rubric file (drafts
    may fall back to the inline prompt, loudly) and its pinned judge panel:
    every judge scores every pair, each judgment is stamped with the judge's
    name, and the report carries per-judge tallies plus inter-judge agreement
    (percent + Cohen's kappa per judge pair) — and judge-vs-human agreement
    when a ``humanValidation`` subset is pinned. Claude judges need
    ANTHROPIC_API_KEY; local-model judges run through ``model_provider``.

    Local judges resolve by the sweep's cross-engine rule (unified
    2026-07-22): an empty/absent ``model`` means the STUDY model at its
    pinned revision — never the judge's name as a model id. The capacity
    guard then asks ``judge_slots_required`` for the panel's PEAK residency
    — the largest single moment under the column release seam, not the
    panel's size — and refuses at evaluate start when ``max_loaded`` (the
    registry capacity, passed by the API path) is smaller. ``max_loaded``
    None (the CLI/bundle path, private in-process copies) skips the
    capacity check, exactly like the sweep preflight.

    Model residency across the panel (2026-08-28): judges are needed one
    COLUMN at a time, so the peak is the MAX of any one still-needed model,
    never the SUM. ``model_release`` (the API path passes the registry's
    explicit release) is called at every judge boundary with the models the
    remainder of the run no longer needs — see
    ``_release_models_for_judge``. ``study_model_generates_later`` is the
    conservative flag for the generation→judging seam: False (evaluate is
    terminal for generation) lets the study model go when no judge uses it;
    a pipeline stage that still generates passes True and it is kept.

    Epoch guard: the source run's stamped experiment hash must equal the live
    manifest's content hash; legacy unstamped runs need
    ``allow_unverified_epoch`` and are stamped ``epochUnverified: true``.

    ``sample_per_condition`` + ``sample_seed`` (2026-08-29) draw a seeded,
    stratified subsample of the source run instead of coding all of it. Both
    or neither: the pair is validated HERE as well as at the CLI edge, so a
    library caller cannot reach the draw with half a request. Per-response
    coding only — see ``evaluate_subsample.paired_refusal``.

    The study's own ``evaluationSampling`` DECLARATION (2026-08-29, review
    round 12) outranks both: when the manifest declares a design, the draw
    follows it with no arguments at all, and arguments that disagree with it
    refuse rather than override (``evaluate_subsample.reconcile``). The
    reconciliation happens HERE rather than at a CLI edge deliberately — this
    is the first point that holds the manifest, so the CLI, the bundle-execute
    path and the submitted argv all get the same cross-check on the same
    bytes."""
    from . import evaluate_subsample, paired_judge
    _log = log or print
    # First thing, before the manifest is even read: a malformed sample ask
    # must refuse without touching the workspace.
    subsample = evaluate_subsample.resolve_request(
        sample_per_condition, sample_seed, program="steerlab-server")
    manifest = manifest_module.Manifest.load(name, root)
    # …then the DECLARATION, which the wire fields and the flags are both
    # checked against. `program` is the CLIENT's: this engine executes a
    # design, it never authors one, so a repair that named `steerlab-server
    # experiment set-evaluation-sampling` would be a command nobody can run.
    subsample = evaluate_subsample.reconcile(
        subsample,
        evaluate_subsample.declared_request(
            manifest.raw.get(evaluate_subsample.DECLARATION_KEY),
            experiment=name, program="steerlab"),
        program="steerlab")
    if subsample is not None and subsample.declared:
        _log(f"evaluate: the study declares a sampling design — "
             f"{subsample.sample_per_condition} record(s) per condition at "
             f"seed {subsample.seed_text} (evaluationSampling)")
    # Effective evaluation (2026-07-22 incident): an explicit block wins;
    # with none, pinned judges + a pinned rubric file ARE the paired-judge
    # declaration and the spec is synthesized from those pins (the app's
    # rubric-file path historically never wrote the block, so frozen
    # studies died HERE after generation). The judge report stamps where
    # the spec came from (cross-engine key "evaluationSource":
    # "manifest" | "pinnedRubric").
    spec, evaluation_source = manifest.effective_evaluation()
    if spec is None or spec.kind != "pairedJudge":
        # WP0 dry run #2's skipped check: right refusal, untyped
        # (`verbFailed`/70 with the boilerplate repair). The verb needs
        # something the study never declared — missingPrerequisite, with a
        # repair that names the CLI that can actually pin it. The reason
        # string is unchanged.
        raise lifecycle_gates.refusing(
            lifecycle_gates.MISSING_PREREQUISITE,
            f"experiment '{name}' has no pairedJudge evaluation configured "
            "— pin at least one judge and a rubric, or declare an "
            "evaluation block",
            repair=rubric_inputs.no_rubric_repair(name))
    if evaluation_source == "pinnedRubric":
        _log("evaluation: no explicit evaluation block — judging from the "
             "pinned judges + rubric file (evaluationSource: pinnedRubric)")

    rubric, rubric_hash, rubric_file = rubric_inputs.resolve_rubric(manifest, root, _log)
    human = evaluation_evidence.load_human_validation(manifest, root) if manifest.human_validation else None

    run_dir = source_run or run_artifacts.latest_run(name, root)
    if not run_dir:
        raise lifecycle_gates.refusing(
            lifecycle_gates.MISSING_PREREQUISITE,
            f"no prior run with generations found for '{name}' — run it first",
            repair=(f"steerlab-server experiment run {name} && "
                    f"steerlab-server experiment analyze {name}"))
    epoch_unverified, measurement_drift = study_admission.require_source_epoch(
        "evaluate", name, manifest, run_dir,
        allow_unverified_epoch=allow_unverified_epoch)
    if measurement_drift:
        _log(f"WARNING: '{name}' drifted from source run "
             f"'{os.path.basename(run_dir)}' in MEASUREMENT-side fields only "
             f"({measurement_drift}) — the generations are unaffected; "
             "judging proceeds under the LIVE settings and the output is "
             "stamped measurementDrift")
    if epoch_unverified:
        _log(f"WARNING: source run '{os.path.basename(run_dir)}' carries no "
             "experiment-hash stamp — judging it under allowUnverifiedEpoch; "
             "the report is stamped epochUnverified")
    generations = []
    with open(os.path.join(run_dir, "generations.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                generations.append(json.loads(line))

    # Declared exclusion rules join HERE — BEFORE judging, so no judge call
    # (or judging packet, on the deferred path) is ever spent on an
    # excluded record. Pairwise deletion matches analyze: filtering the
    # baseline record removes its item from every condition's pairs (the
    # baseline join in paired_judge simply finds no partner). The stamp
    # lands in judge-report.json + exclusions.json; excluded records stay
    # in the source run's generations.jsonl (runs are immutable). No rules
    # declared = today's behavior byte-for-byte. Swift twin:
    # ``evaluatePairedJudge``.
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
                task_inputs.load_prompts(manifest, None, root))
            if not checks:
                raise RuntimeError(exclusions_mod.NO_CHECKS_MESSAGE)
        generations, exclusion_stamp = exclusions_mod.apply(
            generations, exclusion_rules, checks,
            note=exclusions_mod.EVALUATE_NOTE,
            scope=exclusions_mod.SCOPE_SAMPLED_RECORDS)
        _log(f"exclusions: {exclusion_stamp['excludedRecords']} record(s) "
             f"excluded before judging by {len(exclusion_rules)} declared "
             "rule(s); surviving N per condition: "
             + ", ".join(f"{c}={n}" for c, n
                         in exclusion_stamp["survivingN"].items()))

    roster = judge_dispatch.judge_roster(manifest, spec)
    # Local-judge resolution, logged at evaluate START (cross-engine rule,
    # unified with the sweep 2026-07-22 — the judge's NAME is a label, never
    # a model id): empty/absent model → the study model; a different-model
    # local judge refuses here when the panel's PEAK residency is impossible,
    # never mid-panel — UNLESS the caller declared the judge fan-out
    # (``defer_local_judges``, 2026-07-23): the panel's judging then becomes
    # blinded packets for per-judge-model worker jobs instead of an inline
    # second load.
    from . import sweep_selection
    foreign_local: list = []
    for ref in roster:
        if ref.kind != "local":
            continue
        resolved = sweep_selection.resolve_local_judge_model(
            ref.model, manifest.model_id)
        if resolved == manifest.model_id:
            _log(f"local judge '{ref.name}' resolves to the study model "
                 f"{manifest.model_id}")
            continue
        _log(f"local judge '{ref.name}' judges with local model '{resolved}'")
        foreign_local.append(ref)
    # The capacity guard, run ONCE over the whole panel and aware of the
    # custody the run actually has (external review round 12, finding 2a).
    # It used to refuse any foreign local judge on a one-slot server — a
    # per-judge `max_loaded < 2` inside the loop above, decided BEFORE the
    # judge-column release seam ever ran, so the seam that made sequential
    # judging possible could never be reached on the one-slot registry it
    # was built for. The ask is now the run's largest SINGLE MOMENT.
    #
    # Deliberately NOT the sweep's rule: ``_judge_preflight`` requires
    # SIMULTANEOUS residency because a judgeScore sweep interleaves judging
    # with selection and holds every judge's model for the whole grid. An
    # evaluate judges column-outer, so its peak is smaller. Two instruments,
    # two honest arithmetics.
    sequential_custody = model_release is not None
    if not defer_local_judges and max_loaded is not None and foreign_local:
        required = judge_resources.judge_slots_required(
            roster, study_model=manifest.model_id,
            study_revision=manifest.model_revision,
            study_dtype=manifest.dtype,
            study_model_generates_later=study_model_generates_later,
            sequential=sequential_custody)
        if required > max_loaded:
            named = ", ".join(
                f"'{ref.name}' ("
                + judge_resources.identity_text(judge_resources.judge_model_identity(
                    ref, study_model=manifest.model_id,
                    study_revision=manifest.model_revision,
                    study_dtype=manifest.dtype), quoted=False)
                + ")" for ref in foreign_local)
            custody = (
                "judges run as SEQUENTIAL columns here — each finished "
                "judge's model is released before the next one loads — so "
                "the ask is the run's largest single moment, not the "
                f"panel's {len(foreign_local)} foreign local judge(s)"
                if sequential_custody else
                "this caller supplies no model-release seam, so there is no "
                "sequential custody: every local judge's model must be "
                "resident at once")
            raise RuntimeError(
                f"evaluate needs {required} model(s) resident AT ONCE, but "
                f"this server keeps STEERLAB_MAX_LOADED_MODELS={max_loaded} "
                f"— {custody}. The foreign local judge(s): {named}. Set "
                f"STEERLAB_MAX_LOADED_MODELS to at least {required} on this "
                "server, use the study model as judge (leave the judge's "
                "model empty), or pin claude/openrouter judges")
    # Where judging will happen, announced BEFORE it happens (2026-07-24):
    # the inline/deferred fork used to be discoverable only from the
    # artifacts afterwards, and a mixed panel deferring despite a pushed key
    # is the case that surprises people.
    judge_resources.log_judging_custody(roster, _log)
    # Provider pins are checked against OpenRouter's public catalogue BEFORE
    # any judging starts (2026-07-24) — a wrong provider used to surface at
    # the first judge call, after generation had already been paid for.
    judge_dispatch.preflight_openrouter_judges(roster, _log)
    # Per-response coding fork (2026-08-04): a rubric whose frontmatter
    # declares `mode: perResponseCoding` runs the coding instrument — every
    # sampled-text record coded individually, blinded, no pairing and no
    # winner. Everything above (epoch guard, exclusions, roster resolution,
    # provider preflight) is shared; everything below is paired-only.
    from . import response_coding
    coding_schema = response_coding.parse_rubric(rubric)
    if coding_schema is not None:
        if (spec.structured_prompt or "").strip():
            raise RuntimeError(
                "the study declares a paired structured-comparison prompt "
                "but the pinned rubric is perResponseCoding — the two "
                "contracts cannot combine; clear the structured prompt or "
                "pin a paired rubric")
        if manifest.human_validation:
            raise RuntimeError(
                "judge-vs-human agreement for per-response coding is not "
                "implemented yet — unpin humanValidation (the paired-shape "
                "baseline|variant|tie labels do not describe per-response "
                "codes)")
        if resume_from:
            raise RuntimeError(
                "per-response coding does not support --resume-from yet — "
                "re-run the evaluation in one session")
        if (defer_local_judges and foreign_local) \
                or judge_resources.missing_external_credentials(roster):
            raise RuntimeError(
                "per-response coding judges inline only for now — deferred "
                "judging packets for the coding instrument are not "
                "implemented yet. Push a judge key from the app for inline "
                "external coding, or pin local judges (resolving to the "
                "study model on a single-slot server)")
        return _evaluate_response_coding(
            name, manifest, coding_schema, run_dir, generations,
            rubric_hash, rubric_file, roster, root, epoch_unverified,
            measurement_drift, evaluation_source, exclusion_stamp,
            model_provider, _log, model_release=model_release,
            study_model_generates_later=study_model_generates_later,
            subsample=subsample)
    if subsample is not None:
        # Reached only on the paired path: the sample flags name a design the
        # paired judge's unit of analysis cannot express. Refusing beats
        # half-executing a correct-looking command line (the `--shard` rule).
        raise evaluate_subsample.paired_refusal("steerlab-server")
    # P0 guard (external review 2026-07-22): a pairedJudge evaluate always
    # has a judge panel configured, so zero surviving pairs must refuse HERE
    # — a "pairs: 0" judge-report looked like a successful evaluation while
    # judging nothing (the exact failure mode of the old (promptID, seed)
    # join under per-condition derived seeds). Fires before the custody fork
    # so inline and deferred paths refuse identically.
    if not paired_judge._pair_generations(generations):
        raise RuntimeError(paired_judge.NO_PAIRS_MESSAGE)
    # Judge fan-out deferral (2026-07-23): the caller (the pipeline chain,
    # which holds ONE model) declared that local judges needing OTHER models
    # must not judge inline. The whole LOCAL panel becomes blinded,
    # hash-pinned packets — per-judge-model worker jobs judge them and the
    # controller merges through `complete_evaluate_judgment` (full-coverage
    # refusals included). Mixed local+external panels refuse for now: the
    # external half would judge at a different evidence time than the
    # workers — pin an all-local or all-external panel (the honest refusal
    # this commit ships instead of a two-clock merge).
    if defer_local_judges and foreign_local:
        external_kinds = sorted({r.kind for r in roster if r.kind != "local"})
        if external_kinds:
            raise RuntimeError(
                "evaluate panel mixes local judges needing the judge "
                f"fan-out ({', '.join(r.name for r in foreign_local)}) with "
                f"{'/'.join(external_kinds)} judges — a mixed panel cannot "
                "merge one report from two judging clocks yet. Pin an "
                "all-local panel (fans out per judge model) or an "
                "all-external panel (judges inline/deferred), or run "
                "evaluate outside the pipeline")
        return deferred_evaluation.emit_evaluate_judging(
            name, manifest, spec, run_dir, generations, rubric, rubric_hash,
            rubric_file, root, epoch_unverified, measurement_drift, _log,
            exclusion_stamp=exclusion_stamp,
            evaluation_source=evaluation_source)
    # Custody fork (seamless pipeline stage 2, 2026-07-19): external judges
    # without a credential DEFER — the run's paired generations become
    # blinded, hash-pinned packets the Mac judges; complete-judgment
    # verifies and aggregates. Split local/external panels refuse (two
    # evidence times for one report), exactly the sweep rule.
    missing = judge_resources.missing_external_credentials(roster)
    if missing:
        if any(ref.kind == "local" for ref in roster):
            raise RuntimeError(
                "evaluate panel mixes local and external "
                f"({'/'.join(sorted(r.kind for r in roster if r.kind != 'local'))}) "
                "judges but this server has no credential for "
                f"{'/'.join(sorted(missing))} (keyless is the default "
                "custody posture): a split panel cannot defer coherently. "
                "Pin an all-local or all-external panel, or push a judge "
                "key from the app for inline external judging")
        return deferred_evaluation.emit_evaluate_judging(
            name, manifest, spec, run_dir, generations, rubric, rubric_hash,
            rubric_file, root, epoch_unverified, measurement_drift, _log,
            exclusion_stamp=exclusion_stamp,
            evaluation_source=evaluation_source)
    # Retention (2026-07-24): the evaluate run directory is created BEFORE
    # the first judge call, and every judgment is written as it is produced
    # — the Swift path's shape, ported. The old code created the directory
    # after the whole panel finished, so an invalid verdict on the last pair
    # of the last judge destroyed every successful judgment before it and
    # left the researcher with an error message and nothing on disk. A
    # partial directory is marked by run-status.json and the ABSENCE of
    # judge-report.json — a partial panel is never summarized as a report.
    context = evaluation_evidence.judging_context(manifest, spec, run_dir, rubric_hash,
                               rubric_file, roster)
    resumable: dict = {}
    if resume_from:
        resumable = evaluation_evidence.load_resumable_judgments(
            name, resume_from, root, context, _log)
        if resumable:
            # Honest about WHEN, not just what (2026-07-24). A resumed
            # evaluation is judged across two sessions, and an external
            # judge's model can change between them — a `claude-opus-4-8`
            # or an OpenRouter endpoint is not revision-pinned the way a
            # local judge is. The reader has to be able to see that.
            external = sorted({r.name for r in roster if r.kind != "local"})
            if external:
                _log("WARNING: resuming judgment reuses verdicts from an "
                     f"earlier session while judge(s) {', '.join(external)} "
                     "are external (not revision-pinned) — the provider may "
                     "have changed the model between sessions. The report "
                     "stamps this as a multi-session evaluation")

    out = paths.make_unique_run_directory(f"exp-{name}-evaluate", root)
    run_config.write_run_config(out, "evaluate", model_id=manifest.model_id,
                     revision=manifest.model_revision, experiment=name,
                     experiment_hash=manifest.content_hash(),
                     notes={"epochUnverified": True} if epoch_unverified else None)
    # The pins THIS run judges under, written before the first judge call so
    # a later targeted retry can prove it is completing the same evaluation.
    with open(os.path.join(out, evaluation_evidence.JUDGING_CONTEXT_FILENAME), "w",
              encoding="utf-8") as handle:
        json.dump(context, handle, indent=2, sort_keys=True)
    if exclusion_stamp is not None:
        # The stamp file the analyze path also writes — one artifact name
        # to look for on either engine.
        with open(os.path.join(out, "exclusions.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(exclusion_stamp, handle, indent=2, sort_keys=True)
    status = run_status.RunStatus(out, stage="evaluate", experiment=name,
                       source_run=os.path.basename(run_dir),
                       expected=[ref.name for ref in roster])
    status.write()

    all_judgments: list[dict] = []
    judge_blocks: list[dict] = []
    outcome_maps: list[tuple[str, dict]] = []
    judgments_path = os.path.join(out, "judgments.jsonl")
    try:
        with open(judgments_path, "w", encoding="utf-8") as judgments_handle:
            for index, ref in enumerate(roster):
                # One model load per judge COLUMN (external review round 3,
                # finding 3c). The stack closes at the END of each
                # iteration, and the release seam below drops the finished
                # column's container BEFORE this one loads, so two judge
                # models are never resident at once — the guarantee the
                # single-slot rule used to buy by refusing instead.
                judge_resources.release_models_for_judge(
                    model_release, roster, index,
                    study_model=manifest.model_id,
                    study_revision=manifest.model_revision,
                    study_dtype=manifest.dtype,
                    study_model_generates_later=study_model_generates_later,
                    _log=_log)
                with ExitStack() as judge_stack:
                    judge_fn, requested_model, holder = judge_resources.judge_callable(
                        ref, model_provider, study_model=manifest.model_id,
                        study_revision=manifest.model_revision,
                        stack=judge_stack)

                    def _persist(judgment, _ref=ref, _handle=judgments_handle):
                        evaluation_evidence.judgment_stamp_judge(judgment, _ref)
                        # Flushed per row: a killed worker (SIGKILL, node
                        # eviction) still leaves every judgment it finished.
                        _handle.write(json.dumps(judgment) + "\n")
                        _handle.flush()
                        status.note_judgment()

                    # Cells this judge already decided in the resumed session:
                    # reused verbatim, never re-judged.
                    reusable = {cell: row for (judge_name, cell), row
                                in resumable.items() if judge_name == ref.name}
                    judgments, judge_report = paired_judge.evaluate(
                        generations, judge_model=requested_model,
                        judge_rubric=rubric,
                        structured_prompt=spec.structured_prompt, judge=judge_fn,
                        on_judgment=_persist,
                        on_invalid=status.note_invalid_response,
                        existing=reusable or None)
                    if judge_report.get("reusedJudgments"):
                        _log(f"judge '{ref.name}': reused "
                             f"{judge_report['reusedJudgments']} judgment(s) from "
                             f"'{resume_from}', judged "
                             f"{judge_report['freshJudgments']} fresh")
                    all_judgments.extend(judgments)
                    outcome_maps.append(
                        (ref.name,
                         {evaluation_evidence.judgment_key(j): j["outcome"] for j in judgments
                          if not j.get("noncompliant")}))
                    status.note_judge_complete(ref.name)
                    block = {
                        "name": ref.name, "kind": ref.kind,
                        "requestedModel": requested_model,
                        "actualModel": holder["actual"],
                        "conditions": judge_report["conditions"],
                        "pairs": judge_report["pairs"],
                    }
                    for token_key in ("completionTokens", "reasoningTokens"):
                        # Token transparency (2026-08-06): what this judge
                        # actually spent, summed over its column. Provenance
                        # only — no gate reads it. Present just for judges
                        # whose transport reports usage (OpenRouter today).
                        if token_key in judge_report:
                            block[token_key] = judge_report[token_key]
                    if judge_report.get("salvagedVerdicts"):
                        # Salvage visibility (2026-08-06): verdicts in this
                        # column whose winner was regex-rescued from
                        # truncated JSON rather than cleanly parsed. Loud in
                        # the log AND stamped in the report — a column that
                        # is mostly salvage is weaker evidence than a clean
                        # one, and 22/36 salvaged rows on a 2026-08 run
                        # were invisible until the raw JSONL was reread.
                        block["salvagedVerdicts"] = \
                            judge_report["salvagedVerdicts"]
                        _log(f"WARNING: judge '{ref.name}' produced "
                             f"{judge_report['salvagedVerdicts']} of "
                             f"{judge_report['pairs']} verdict(s) via "
                             "truncation salvage (winner legible, reasoning "
                             "cut) — see verdictSalvaged rows in "
                             "judgments.jsonl")
                    if judge_report.get("noncompliantJudgments"):
                        # Noncompliance visibility (2026-08-09): pairs in
                        # this column with NO verdict — recorded as rows,
                        # excluded from every tally and agreement entry.
                        # Nonzero-only, like salvagedVerdicts, and stamped
                        # on the WRITTEN report: a reader must see the
                        # column is incomplete without re-reading
                        # judgments.jsonl.
                        block["noncompliantJudgments"] = \
                            judge_report["noncompliantJudgments"]
                        _log(f"WARNING: judge '{ref.name}' was noncompliant "
                             f"on {judge_report['noncompliantJudgments']} of "
                             f"{judge_report['pairs']} pair(s) — those cells "
                             "carry no verdict (see noncompliant rows in "
                             "judgments.jsonl)")
                    if resume_from:
                        # Per-judge session composition, stamped for EVERY judge
                        # in a resumed run — including judges that reused
                        # nothing, whose whole column is fresh. Deriving both
                        # from the pair count rather than only stamping judges
                        # that happened to reuse keeps the report's totals
                        # equal to the judgments actually written.
                        reused_here = judge_report.get("reusedJudgments", 0)
                        block["reusedJudgments"] = reused_here
                        block["freshJudgments"] = judge_report["pairs"] - reused_here
                    if ref.kind == "local" and holder.get("revision"):
                        # The judge model's pinned revision (JudgeRef.revision /
                        # study-pin fallback, 2026-07-23) — judgment artifacts
                        # name the exact judge bytes.
                        block["revision"] = holder["revision"]
                    if ref.kind == "local" and holder.get("requestedDtype"):
                        # The dtype the manifest PINNED and this load
                        # requested. Stamped so the artifact records what
                        # was asked for, not just what the server happened
                        # to default to (finding 2).
                        block["requestedDtype"] = holder["requestedDtype"]
                    if ref.kind == "local" and holder.get("actualDtype"):
                        # The dtype the judge model actually ran in. The
                        # loader now refuses an unhonored pin outright, so
                        # these agree whenever a pin exists — but an
                        # UNPINNED judge (study-model judges need no pin)
                        # has only this, and it is what makes two judging
                        # sessions on different devices comparable after
                        # the fact (external review round 4, finding 2).
                        block["dtype"] = holder["actualDtype"]
                    if ref.kind == "openrouter":
                        block["provider"] = \
                            paired_judge.canonical_openrouter_provider(ref.provider)
                    # Per-judge credential provenance (engineer review
                    # 2026-07-19): a mixed claude/openrouter panel can judge
                    # through DIFFERENT credential sources, so each external
                    # judge records its own (the report-level judgeCredential
                    # stays as the legacy single-source stamp).
                    if ref.kind != "local":
                        from . import judge_credentials
                        credential = judge_credentials.credential_for(
                            "openrouter" if ref.kind == "openrouter" else "claude")
                        if credential is not None:
                            block["credential"] = {"kind": credential.kind,
                                                   "source": credential.source}
                    judge_blocks.append(block)
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        # The refusal stands; what changes is that the successful judgments
        # SURVIVE it, named and hash-able, with the failing judge and the
        # raw malformed responses beside them. No judge-report.json is
        # written — this directory is a failure record, never a result.
        status.fail(exc)
        _log(f"evaluate FAILED after {status.judgment_count} judgment(s) — "
             f"partial evidence kept in {os.path.basename(out)} "
             f"({type(exc).__name__}: {exc})")
        raise

    report: dict = {
        "experiment": name,
        "sourceRun": os.path.basename(run_dir),
        "rubricFile": rubric_file, "rubricHash": rubric_hash,
        "judges": judge_blocks,
        "agreement": evaluation_evidence.agreement_entries(outcome_maps),
        "pairs": judge_blocks[0]["pairs"] if judge_blocks else 0,
        # Legacy single-judge keys (first judge) so existing readers keep
        # working; per-judge truth lives in "judges".
        "conditions": judge_blocks[0]["conditions"] if judge_blocks else {},
        "judgeModel": judge_blocks[0]["requestedModel"] if judge_blocks else None,
        "requestedJudgeModel": judge_blocks[0]["requestedModel"] if judge_blocks else None,
        "actualJudgeModel": judge_blocks[0]["actualModel"] if judge_blocks else None,
    }
    noncompliant_total = sum(
        b.get("noncompliantJudgments", 0) for b in judge_blocks)
    if noncompliant_total:
        # Panel total, nonzero-only — the paired-judge sibling of the coding
        # report's noncompliantCodings (and the cross-engine report-level
        # key): per-judge truth lives in "judges", but the report itself
        # must say the evaluation has holes and how many.
        report["noncompliantJudgments"] = noncompliant_total
    # Where the judging ran + through which credential is RECORDED
    # provenance (2026-07-19), mirroring the sweep selection blocks.
    report["judgedOn"] = "server"
    # Where the effective SPEC came from (2026-07-22): "manifest" = explicit
    # evaluation block; "pinnedRubric" = synthesized from the pinned judges
    # + rubric file. Cross-engine key — the Swift report stamps the same.
    report["evaluationSource"] = evaluation_source
    if resume_from:
        # A multi-session evaluation is a DIFFERENT evidentiary object from
        # one judged in a single sitting, and the report says so rather than
        # leaving the reader to infer it from per-judge counts.
        reused = sum(b.get("reusedJudgments", 0) for b in judge_blocks)
        report["judgingSessions"] = {
            "resumedFrom": resume_from,
            "reusedJudgments": reused,
            "freshJudgments": sum(b.get("freshJudgments", 0)
                                  for b in judge_blocks),
            # Named because it is the reason this needs a stamp: these
            # judges are not revision-pinned, so "the same judge" across
            # two sessions is an assumption, not a fact.
            "unpinnedExternalJudges": sorted(
                {r.name for r in roster if r.kind != "local"}),
        }
        _log(f"evaluate completed by resuming '{resume_from}': {reused} "
             "judgment(s) reused")
    if exclusion_stamp is not None:
        # The identical stamp shape analyze writes; excluded records were
        # filtered BEFORE judging, so their judge calls never happened.
        report["exclusions"] = exclusion_stamp
    if any(ref.kind != "local" for ref in roster):
        from . import judge_credentials
        try:
            credential = judge_credentials.resolve()
        except ValueError:
            credential = None
        if credential is not None:
            report["judgeCredential"] = {"kind": credential.kind,
                                         "source": credential.source}
    if epoch_unverified:
        report["epochUnverified"] = True
    if measurement_drift:
        # Tolerated measurement-side drift is never silent: which fields
        # differed from the source run's epoch, verbatim.
        report["measurementDrift"] = measurement_drift
    if human is not None:
        report["humanValidation"] = {"path": manifest.human_validation.path,
                                     "hash": manifest.human_validation.hash,
                                     "rows": len(human)}
        report["humanAgreement"] = [
            {"judge": entry["judges"][1], "n": entry["n"],
             "percentAgreement": entry["percentAgreement"],
             "kappa": entry["kappa"]}
            for entry in evaluation_evidence.agreement_entries(
                [("human", evaluation_evidence.materialize_human_validation(human, outcome_maps))]
                + outcome_maps)
            if entry["judges"][0] == "human"]

    # judgments.jsonl was written row-by-row as the panel judged; the report
    # is what makes this directory a RESULT rather than a failure record.
    with open(os.path.join(out, "judge-report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    status.complete()
    _log(f"evaluate ({len(all_judgments)} judgments, {len(roster)} judge(s)) → {out}")
    return out
