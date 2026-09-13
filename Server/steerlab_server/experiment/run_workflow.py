"""Coordinate run admission, source resolution, condition execution and completion.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
from typing import Callable
from . import lifecycle_gates, paths
from . import resume as resume_mod
from . import sharding as sharding_mod
from . import cancellation
from . import choice_scoring
from . import condition_execution
from . import execution_reporting
from . import forward_resolution
from . import manifest as manifest_module
from . import model_resources
from . import panel_workflow
from . import run_artifacts
from . import run_preflight
from . import run_readouts
from . import run_reporting
from . import study_admission
from . import task_inputs
from . import vector_materialization


def run(name: str, prompts_file: str | None = None, root: str | None = None,
        dtype: str = "auto", device: str | None = None, *, model_provider=None,
        should_cancel: Callable[[], bool] | None = None, log=None,
        checkpoint: resume_mod.CheckpointFlag | None = None,
        run_directory: str | None = None,
        on_run_directory: Callable[[str], None] | None = None,
        forward_resolutions: dict | None = None,
        shard: "sharding_mod.ShardSpec | None" = None) -> str:
    """Main condition matrix: every prompt × condition × seed, paired by prompt.
    Writes ``generations.jsonl`` + ``report.json`` (parallel to Swift run).

    Reliability hooks (headless paths only; the API passes none of them):
    ``checkpoint`` is polled between records — when a signal set it, the JSONL
    is fsynced, ``resume-state.json`` written, and ``CheckpointRequested``
    raised for the exit-85 path. ``run_directory`` resumes a checkpointed,
    incomplete run (record-level skip; refuses complete directories).
    ``on_run_directory`` is called once with the chosen run directory so the
    submitter can persist its resume pointer before generation starts."""
    _log = log or print
    manifest = manifest_module.Manifest.load(name, root)
    study_admission.verify_or_warn(manifest, root)
    # SAE latent conditions (proposal r2 §8 P2-9): validate the declaration and
    # refuse the study kinds whose loop cannot arm one, BEFORE the model loads.
    # The modelOutput path materializes them inside _run_impl.
    condition_execution.sae_latent_preflight(manifest, _log)

    # API multi-agent runs can use the server's resident model registry. CLI
    # runs keep the older single-loaded-model behavior by falling through.
    if manifest.study_kind == "multiAgent":
        # E1. The old blanket refusal was right about ONE axis and applied to
        # the whole family. Turns within a transcript are ordered and cannot be
        # split across workers — that refusal stands, below. Replicates are
        # independent play-throughs sharing no state, which is exactly the
        # property sharding needs, so the transcript is the shardable unit.
        if shard is not None and manifest.samples_per_item < 2:
            raise RuntimeError(
                "cannot shard a single-transcript panel study: turns within one "
                "transcript are ordered — turn k is conditioned on turns "
                "1..k-1 — so they cannot be split across workers. Replicates "
                "CAN be: raise samplesPerItem and shard across them, e.g. "
                "samplesPerItem 8 with --shard 0/4.")
        # Resume is turn-level, not record-level, but auto-resubmit still
        # hands back the SAME run directory and expects the run to continue
        # in it. Refusing that made requeued panel jobs start somewhere else
        # (or refuse outright) and abandon the partial transcripts the runner
        # had carefully been flushing.
        # Preflight BEFORE dispatching to either path. It used to sit after
        # this branch, so the resident-model/API route — the one a cluster
        # submission actually takes — skipped the check entirely.
        # A panel's turns run on the models its SEATS name, not on the
        # manifest's baseline model. Resolve that here, before anything is
        # acquired: a study whose manifest still said 27B while every seat
        # said 4B spent the load attempting to fetch a 27B nobody would use,
        # and surfaced as a huggingface_hub traceback rather than as the
        # mismatch it was.
        manifest = run_preflight.panel_load_model(manifest, root, _log)
        # Preflight AFTER resolving the model, not before: it used to run
        # against the manifest's declared default, which no turn uses. On a
        # study whose default was a gated 27B that produced a 401 and a
        # skipped preflight — so the one check that could have sized the run
        # never looked at the 4B that actually ran.
        run_preflight.scenario_preflight_or_warn(manifest, root, _log)
        run_preflight.artifact_preflight(manifest, root, _log)
        if model_provider is not None:
            return panel_workflow.run_multi_agent_study(
                name, manifest, None, root, model_provider=model_provider,
                log=_log, shard=shard, run_directory=run_directory,
                on_run_directory=on_run_directory,
                should_cancel=should_cancel, checkpoint=checkpoint)

    # Exact token preflight BEFORE the model load (C1). On a cluster the
    # alternative is the queue wait plus a multi-minute 27B load, followed by
    # a death on whichever oversized item the run reached first — telling you
    # about that one item and none of the others. This is weights-free
    # (AutoTokenizer + AutoConfig), so it costs nothing but a file read.
    #
    # A preflight that cannot RUN never blocks the run: the in-generation
    # ContextBudgetError remains the backstop, and turning a diagnostic into
    # a new way to fail would be a poor trade. Only a successfully computed
    # overflow refuses.
    if manifest.study_kind != "multiAgent":
        # A declared-but-empty agent comparison carrying injection
        # conditions would measure baseline only while every declared arm
        # silently vanished (observed live 2026-08-11: a 4-shard fan-out
        # produced 12 baseline records instead of 96). verify() reports the
        # same violation; drafts only warn there, so the run refuses here —
        # before any queue wait or model load, and identically on every
        # shard of a fan-out.
        from .manifest import (inert_conditions_problem,
                               no_measured_conditions_problem)
        inert_problem = inert_conditions_problem(manifest.raw)
        if inert_problem:
            # WP0 step 8: typed `inertConditions` — the 2026-08-11 incident's
            # rule (a 4-shard fan-out that produced 12 baseline records
            # instead of 96), now machine-readable on both engines.
            raise lifecycle_gates.refusing(
                lifecycle_gates.INERT_CONDITIONS, inert_problem,
                repair=("declare arms the study's studyType actually runs, on "
                        "the Mac: steerlab-cli experiment declare-condition "
                        f"{name} <arm> --slots <concept>:<layer>:<alpha>"))
        # The other road to a silent baseline-only run (WP0 dry run #0,
        # P0-2): a concept study whose arms were never declared at all. Same
        # place, same reason — before any queue wait or model load.
        nothing_to_measure = no_measured_conditions_problem(manifest.raw)
        if nothing_to_measure:
            raise lifecycle_gates.refusing(
                lifecycle_gates.INERT_CONDITIONS, nothing_to_measure,
                repair=("steerlab-cli experiment declare-condition "
                        f"{name} <arm> --slots <concept>:<layer>:<alpha>  "
                        "(authoring is Mac-authority)"))
        run_preflight.token_preflight_or_warn(manifest, prompts_file, root, _log)
        run_preflight.artifact_preflight(manifest, root, _log)
        run_preflight.instrument_preflight(manifest, prompts_file, root, _log)
        run_preflight.response_format_preflight(manifest, prompts_file, root)

    # THE RESUME/SHARD GATE, BEFORE THE MODEL (open-issues §16 repair 2).
    # Its inputs are three file reads on the run directory, and its refusals
    # are the ones a resumed cluster job hits first — but they used to fire
    # from inside `_run_impl`/`_run_multi_agent_study`, i.e. after staging and
    # loading 51 GiB of weights onto the device. Four GPU allocations died
    # that way on 2026-08-18 at the shard-identity check alone.
    #
    # The epoch comparison is deferred here for an UNPINNED DRAFT only:
    # `_pin_model_revision` (below) can still write the resolved revision into
    # such a manifest and change its content hash, and a bundle execute that
    # re-imported the unpinned manifest depends on that repair. Both inner
    # paths re-run the full gate under the post-pinning manifest, so this
    # never weakens it.
    if run_directory is not None:
        _epoch_stable = bool(manifest.model_revision)
        if manifest.study_kind == "multiAgent":
            run_preflight.panel_resume_admission(run_directory, manifest=manifest,
                                    shard=shard,
                                    check_experiment_hash=_epoch_stable)
        else:
            run_preflight.resume_admission(run_directory, name=name, manifest=manifest,
                              shard=shard,
                              check_experiment_hash=_epoch_stable)

    with model_resources.acquire_model(manifest, dtype, device, model_provider) as model:
        manifest = model_resources.pin_model_revision(name, manifest, model, root, _log)
        # Multi-agent studies branch to the scenario runner (parallel to Swift
        # runMultiAgentStudy) — task prompts and concept generations don't apply.
        if manifest.study_kind == "multiAgent":
            return panel_workflow.run_multi_agent_study(
                name, manifest, model, root, log=_log, shard=shard,
                run_directory=run_directory, on_run_directory=on_run_directory,
                should_cancel=should_cancel, checkpoint=checkpoint)
        return _run_impl(name, manifest, model, root, prompts_file, should_cancel,
                         _log, checkpoint=checkpoint, run_directory=run_directory,
                         on_run_directory=on_run_directory,
                         forward_resolutions=forward_resolutions, shard=shard)


def _run_impl(name, manifest, model, root, prompts_file, should_cancel, _log,
              checkpoint=None, run_directory=None, on_run_directory=None,
              forward_resolutions=None, shard=None) -> str:
    # Greedy temp 0 ignores seeds — reject the redundant/foot-gun combo (Swift).
    if manifest.temperature == 0 and len(manifest.seeds) > 1:
        raise RuntimeError(
            "temperature 0 is greedy and ignores seeds — use exactly one seed, or "
            "raise the temperature")
    if manifest.samples_per_item > 1 and manifest.temperature <= 0:
        raise RuntimeError(
            "samplesPerItem > 1 requires temperature > 0 — greedy decoding makes "
            "every sample identical")

    # Resume gate BEFORE any model work: a complete directory refuses loudly
    # (immutable runs), and the checkpointed manifest must be the SAME frozen
    # content — appending records derived from drifted pins would silently mix
    # two experiments into one artifact.
    #
    # `run` runs the SAME gate before it acquires the model (§16 repair 2);
    # this call is what keeps a direct `_run_impl` caller admitted identically,
    # and it re-checks the hash under the post-pinning manifest.
    resuming = run_directory is not None
    if resuming:
        run_preflight.resume_admission(run_directory, name=name, manifest=manifest,
                          shard=shard)

    # Reasoning-style scoring rides on sampled text (no model access): loaded
    # up front so a drifted/broken taxonomy fails BEFORE generation, scored
    # at metrics/report time from the generated text.
    from . import reasoning_style
    style = reasoning_style.load_pinned(manifest, root)

    # Prompts load + transcript gates BEFORE extraction: a schema-invalid or
    # family-incompatible transcript (or rawCompletion+transcript) must fail
    # at run START, not after minutes of vector re-derivation.
    prompts = task_inputs.load_prompts(manifest, prompts_file, root)
    task_inputs.check_transcript_prompts(manifest, prompts)
    # Exclusion-rule preflight at run START (same rule as transcripts): a
    # malformed rule declaration, or failedAttentionCheck with no checked
    # items, refuses before any generation compute — the rules are joined at
    # analyze, but a run whose analysis is doomed should not spend GPU time.
    from . import exclusions as _exclusions
    _exclusions.preflight(manifest.raw, prompts)
    # Response-format gate: an answer-token instrument pointed at rows that
    # ask for a JSON object scores the opening brace's position, not the
    # choice. That is a silently WRONG measurement rather than a failure,
    # which is why it refuses rather than warns. Swift twin:
    # ExperimentTasks.checkResponseFormats.
    task_inputs.check_response_formats(manifest, prompts)
    # Inert concept machinery is inert at RUN time too (2026-07-19): a
    # compare-agents study never re-derives carried concepts' vectors.
    from .manifest import concept_machinery_operative, inert_machinery_note
    machinery = concept_machinery_operative(manifest.raw)
    bundles = vector_materialization.extract_all(model, manifest, root) if machinery else {}
    # LOUD when inert (2026-08-11): a legal agent-comparison run that
    # carries concepts/conditions it will not execute must say so at start
    # and stamp it — a baseline-only result that looks completed and
    # ordinary cost two full GPU rounds before anyone saw why. The
    # no-agent-arms shape refuses earlier (inert_conditions_problem);
    # this is the shape that rightfully proceeds.
    inert_note = inert_machinery_note(manifest.raw)
    if inert_note is not None:
        _log("WARNING: declared studyType "
             f"'{inert_note['declaredStudyType']}' keeps the concept "
             f"machinery INERT — {len(inert_note['inertConditions'])} "
             "carried injection condition(s) "
             f"({', '.join(inert_note['inertConditions']) or 'none'}) and "
             f"{len(inert_note['inertConcepts'])} concept(s) will NOT "
             "execute; this run measures baseline and agent (variant) "
             "conditions only")
    # SAE latent conditions re-derive here, alongside the concept vectors and
    # for the same reason: the manifest pins the RECIPE, the run re-derives the
    # intervention from the pinned source. Done BEFORE the run directory is
    # stamped so the pinned SAE repository commit can ride config.json.
    condition_execution.sae_latent_preflight(manifest, _log)
    latent_conditions = condition_execution.materialize_sae_latent_conditions(manifest, _log)
    if run_directory is None:
        # Shard partials are visibly partial by name; the merge assembles the
        # plain exp-<name>-run directory from them.
        slug = (f"exp-{name}-run" if shard is None
                else f"exp-{name}-run-shard{shard.index}of{shard.count}")
        run_directory = paths.make_unique_run_directory(slug, root)
    if on_run_directory is not None:
        on_run_directory(run_directory)
    sampling = execution_reporting.sampling_metadata(model, manifest.temperature)
    if not resuming:
        # A resumed run keeps its original stamps (config.json's createdAt is
        # the run's birth, and vectors re-derive to identical bytes anyway).
        # The inertness note rides config.json's open `notes` dict so a
        # baseline-only run is self-describing without its log.
        #
        # The SAE latent provenance rides the SAME open dict, for the same
        # reason and by the same rule: config.json's top-level key set is
        # CLOSED (run_config.RUN_CONFIG_KEYS — changing it is a cross-engine
        # schema bump), and engine-specific extras belong in `notes`. What is
        # stamped is the pinned SOURCE of each latent arm: repository +
        # resolved commit, SAE config hash, encoder/decoder row hashes,
        # activation family, and the declared mode/beta in latent units. A run
        # that steers on a published dictionary must record which published
        # bytes it read, or the arm cannot be reproduced from its own record.
        run_notes: dict = {}
        if inert_note is not None:
            run_notes["inertConceptMachinery"] = inert_note
        if latent_conditions:
            run_notes["saeLatentConditions"] = [
                provenance for _spec, _edit, provenance in latent_conditions]
        run_artifacts.write_config_snapshot(manifest, run_directory, "run", model=model,
                               notes=run_notes or None, root=root, log=_log)
        vector_materialization.persist_vectors(bundles, manifest, model, run_directory)
        execution_reporting.write_substrate(model, run_directory, sampling)
    # WS7.1: study-run-start cross-substrate check — logged every start
    # (resumes included); the durable advisories.txt is a creation stamp.
    execution_reporting.advise_cross_substrate(manifest, run_directory, root, _log,
                            write_file=not resuming)
    # WP6 R1: same shape, different question — is the stack underneath us the
    # one the committed platform lock pins?
    execution_reporting.advise_dependency_lock_drift(run_directory, _log, write_file=not resuming)

    # Stage 4: forward-referenced conditions resolve HERE — after the run
    # directory exists (the resolution record is run evidence), before any
    # condition executes. An unresolvable reference refuses the whole run.
    forward_resolution.resolve_manifest_forward_refs(manifest, run_directory, root, _log,
                                   ledger_pins=forward_resolutions)

    # The SAME resolver verify() validates and sharding enumerates against —
    # one definition of what a run executes, so the three cannot disagree
    # (external review round 11). Variant comparison runs baseline + variants
    # only (carried injection conditions are inert, Swift runVariantComparison
    # twin); otherwise the concept machinery decides, and a baseline is
    # prepended when absent so paired judging has baseline pairs.
    from .manifest import effective_conditions
    conditions = effective_conditions(manifest)
    cancelled = False
    experiment_hash = manifest.content_hash()
    instruments = set(manifest.outcome_instruments)
    wants_choice = bool(instruments & condition_execution.CHOICE_INSTRUMENTS)
    # Fail fast (before any model compute): an ordinalScale study must have
    # DECLARED a known aggregation — the instrument-design choice is never
    # silently defaulted. verify() reports the same violation; drafts only
    # warn there, so the run refuses here.
    task_inputs.resolve_ordinal_aggregation(manifest)
    # Declared numeric-answer parser (registry data, not code): resolved once
    # here — a missing/malformed registry or a drifted pin refuses the run at
    # START, never mid-generation. None = the historical caseFamily path.
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
    # …and when nothing was declared, the DEPRECATED caseFamily trigger is what
    # chose this run's numeric endpoint. Said at start, where the rest of the
    # run's configuration is reported, not once per record.
    from .manifest import implicit_case_family_endpoint
    study_admission.advise_implicit_case_family(
        numeric_parser is None and implicit_case_family_endpoint(manifest),
        run_directory, _log, write_file=not resuming)
    # E1: one resolver decides what a study does; the run loop, the routing
    # rules and the UI all read the same answer.
    from . import execution_plan
    _plan = execution_plan.resolve(instruments)
    wants_sampled = _plan.generates_sampled_text
    # A declared sampling setting this plan will never read. Advisory, not a
    # refusal: the run is well-defined and its result unaffected, but a
    # temperature that decides nothing is a design mistake worth saying.
    _inert = execution_plan.inert_sampling_advisory(
        instruments, manifest.temperature, manifest.samples_per_item)
    if _inert:
        _log(f"note: {_inert}")
    # RepE reader scoring rides on sampled text: each output is re-read through
    # every pinned reader and stamped as readerScores on the record.
    reader_scorers = condition_execution.reader_scorers(manifest, root) \
        if ("repeReaderScore" in instruments and manifest.reader_refs) else []
    # Shard plan (multi-GPU fan-out): enumerate the run's FULL expected
    # record-key list in the executor's own emission order and take this
    # shard's contiguous slice. The sample count mirrors _execute_condition;
    # variant conditions are enumerated under the MANIFEST's sampling policy,
    # which _require_manifest_sampling_policy below enforces before any
    # generation. Sharding is execution logistics: nothing here touches the
    # manifest or its content hash.
    plan = None
    if shard is not None:
        # Order IS the contract: this list must match the executor's emission
        # order below (ordinary, then variants, then latent), or the shard's
        # contiguous key slice names records a different worker will emit.
        # Latent conditions are APPENDED so adding one never shifts an existing
        # condition's start position in the key list.
        condition_names = ([c.name for c in conditions]
                           + [vc.name for vc in manifest.variant_conditions]
                           + [spec.name for spec, _e, _p in latent_conditions])
        sample_count = condition_execution.effective_sample_count(manifest)
        all_keys = sharding_mod.expected_record_keys(
            condition_names=condition_names, prompts=prompts,
            wants_choice=wants_choice, wants_sampled=wants_sampled,
            sample_count=sample_count,
            # The SAME scope rule _execute_condition applies at emission —
            # a scope-blind plan expects instrument readouts for items the
            # executor rightly declines, and the merge then refuses a run
            # whose shards all succeeded (2026-08-04).
            instrument_scope=manifest.raw.get("outcomeInstrumentScope"))
        plan = sharding_mod.plan_shard(shard, all_keys=all_keys,
                                       condition_names=condition_names)
        if not resuming:
            sharding_mod.write_shard_stamp(run_directory, plan,
                                           experiment_hash)
        _log(f"shard {shard.label}: records "
             f"[{plan.record_range[0]}, {plan.record_range[1]}) of "
             f"{plan.total_records}; battery/condition ownership: "
             + (", ".join(plan.owned_conditions) or "none"))
    writer = resume_mod.GenerationWriter(
        run_directory, verb="run", checkpoint=checkpoint, resume=resuming,
        log=_log, allowed_keys=(plan.allowed_keys if plan is not None else None))
    if resuming:
        _log(f"resuming run in {run_directory}: {writer.resumed_count} records "
             "already complete — skipping them")
    # J-lens readout, resolved at RUN START. A manifest that DECLARES a readout
    # and silently produces none is the exact failure this instrument exists to
    # prevent, so an unusable declaration refuses here rather than being
    # skipped: before this, the freeze gates pinned a config that nothing ever
    # armed.
    # The WHOLE-STUDY generation count, for the readout budget. Computed from
    # the full matrix, not this shard's slice: a per-shard bound would multiply
    # the effective ceiling by the shard count, which is exactly what the
    # ceiling exists to prevent.
    jlens_expected_generations = (
        (len(conditions) + len(manifest.variant_conditions)
         + len(latent_conditions))
        * len(prompts) * condition_execution.effective_sample_count(manifest))
    jlens_trace = run_readouts.open_jlens_trace(
        manifest, model, root, run_directory=run_directory,
        checkpoint=checkpoint, resuming=resuming, log=_log,
        allowed_keys=(plan.allowed_keys if plan is not None else None),
        expected_generations=jlens_expected_generations,
        generates_sampled_text=wants_sampled)
    measurement = dict(name=name, manifest=manifest, root=root,
                       experiment_hash=experiment_hash,
                       wants_choice=wants_choice, wants_sampled=wants_sampled,
                       reader_scorers=reader_scorers,
                       numeric_parser=numeric_parser,
                       should_cancel=should_cancel, log=_log,
                       jlens_trace=jlens_trace)
    # Study-owned sampling guard (2026-07-21, defense in depth): refuse
    # BEFORE any generation compute if a condition would execute under a
    # sampling policy different from the manifest's declared one.
    condition_execution.require_manifest_sampling_policy(manifest, model, root,
                                      wants_choice=wants_choice)
    # …and the comparability question the same resolution answers (2026-08-24):
    # are all the arms of this run armed with the SAME effective system
    # content? Advisory, never a refusal, and silent unless they diverge.
    # Computed over the FULL matrix, not this shard's slice: divergence is a
    # property of the design, and every shard should say the same thing.
    execution_reporting.advise_system_prompt_divergence(
        condition_execution.run_arm_system_prompts(manifest, conditions, latent_conditions, root),
        run_directory, _log, write_file=not resuming)
    # What each condition's intervention actually CHANGES — token positions,
    # prefill/decode behaviour, dose units, the matched control, the claim
    # limits — stamped once, beside the run's other provenance, before any
    # generation compute. The declared half (`interventionState`) already rides
    # every record; it does not say where in the token stream an edit lands,
    # and the four paths a condition can arm do not land in the same places.
    # Sidecar, not a config.json key: that key set is the closed cross-engine
    # schema. Full matrix like the advisory above, so every shard writes the
    # same bytes and the merge carries one. See `.intervention_scope`.
    if not resuming:
        from . import intervention_scope as _intervention_scope
        _intervention_scope.stamp_run(
            run_directory, experiment=name, conditions=conditions,
            resolve_ordinary=lambda c: (
                condition_execution.intervention_state(c),
                condition_execution.condition_injections(c, bundles, preflight=False)),
            variant_conditions=manifest.variant_conditions,
            resolve_variant=lambda vc: condition_execution.effective_variant_condition(
                vc, manifest, model, root, wants_choice=wants_choice),
            latent_conditions=latent_conditions, log=_log)
    try:
        for condition in conditions:
            if plan is not None and not plan.condition_participates(condition.name):
                continue  # another shard owns every record of this condition
            if cancellation.observe_cancel(should_cancel, _log, f"condition={condition.name}"):
                cancelled = True
                break
            eff = condition_execution.effective_ordinary_condition(condition, bundles, manifest)
            if condition_execution.execute_condition(model, eff, prompts, writer, **measurement):
                cancelled = True
                break

        # Variant conditions run through the SAME executor as baseline and
        # concept conditions (one measurement pipeline — the 2026-07-13
        # unification): the variant resolves to an effective configuration
        # (stored injections + adapter + prompt settings + provenance) and
        # every requested measurement applies identically. Only the model
        # configuration is condition-specific, never the measurements.
        if not cancelled:
            from . import model_variant
            for vc in manifest.variant_conditions:
                if plan is not None and not plan.condition_participates(vc.name):
                    continue  # another shard owns every record of this condition
                if cancellation.observe_cancel(should_cancel, _log, f"condition={vc.name}"):
                    cancelled = True
                    break
                try:
                    eff = condition_execution.effective_variant_condition(
                        vc, manifest, model, root, wants_choice=wants_choice)
                    # Bind verification to the bytes that are ABOUT to load.
                    # The run-start preflight fails fast, but it can run long
                    # before this point — a baseline arm may take hours — and
                    # files are not immutable for the life of a run. Verifying
                    # here means the identity stamped on every row of this
                    # condition describes the adapter that actually shaped it
                    # (external review round 10). Scoped to armed readouts:
                    # that is where the identity becomes a claim.
                    if jlens_trace is not None:
                        eff.verified_identity = run_readouts.verified_identity_for(
                            eff.variant, root, label=vc.name)
                    # root=root, or verification and loading can resolve the
                    # same relative path in DIFFERENT workspaces: verified
                    # here, loaded from STEERLAB_ROOT/cwd. _adapter_directory
                    # has warned about this since round 3; no call site passed
                    # it (external review round 11).
                    adapter = model_variant.apply_adapter(
                        model, eff.variant, root=root)
                except (OSError, KeyError, ValueError, RuntimeError) as exc:
                    _log(f"variant '{vc.name}' skipped: {exc}")
                    # Under sharding, exactly ONE shard (the condition's
                    # owner) emits the error record — the merge would refuse
                    # duplicated cells, and a single-job run emits it once.
                    if plan is None or plan.owns_condition(vc.name):
                        writer.emit({"experiment": name, "condition": vc.name,
                                     "error": str(exc),
                                     "variantArtifactPath": vc.artifact_path})
                    continue
                try:
                    if condition_execution.execute_condition(model, eff, prompts, writer,
                                          adapter_active=adapter is not None,
                                          **measurement):
                        cancelled = True
                finally:
                    model_variant.remove_adapter(model, adapter)
                if cancelled:
                    break

        # SAE latent conditions run through the SAME executor as every other
        # condition (proposal r2 §8 P2-9): the mechanism is different, the
        # measurements are identical — prompt-metadata copy, the answer-token
        # instrument, sampled generation, categorical parsing, reader scores,
        # and the same resume/checkpoint semantics. Condition-specific code
        # configures the model; it never redefines how outcomes are measured.
        # Last in the matrix, matching the shard key order built above.
        if not cancelled:
            for spec, edit, provenance in latent_conditions:
                if plan is not None and not plan.condition_participates(spec.name):
                    continue  # another shard owns every record of this condition
                if cancellation.observe_cancel(should_cancel, _log, f"condition={spec.name}"):
                    cancelled = True
                    break
                eff = condition_execution.effective_sae_latent_condition(
                    spec, edit, provenance, manifest)
                if condition_execution.execute_condition(model, eff, prompts, writer, **measurement):
                    cancelled = True
                    break

        if cancelled:
            # Cooperative cancel parks the run exactly like a checkpoint signal
            # (reason "cancel"): the directory stays resumable and report.json
            # — the completion artifact — is NOT written.
            writer.interrupt(reason="cancel")
    finally:
        writer.close()
        jlens_summary = (jlens_trace.close(expected_records=None)
                         if jlens_trace is not None else None)
        if jlens_summary is not None:
            _log(f"J-lens readout: {jlens_summary['traceRows']} traced "
                 f"generation(s), {jlens_summary['traceObservations']} "
                 f"observation(s), complete={jlens_summary['complete']}"
                 + ("" if jlens_summary["complete"] else
                    f" ({jlens_summary['incompleteRecords']} incomplete — NOT "
                    f"usable as a readout)"))

    records = writer.records
    # Capability battery inside the run (2026-07-13): when the manifest PINS
    # a battery, score it under every condition of the matrix — after the
    # main loop so generations.jsonl bytes are untouched, with its own
    # resume-skippable battery.jsonl stream. Runs only when the main loop
    # completed; a cancel/checkpoint during the battery parks the run
    # resumable exactly like one during generation.
    battery = None
    if (not cancelled and manifest.capability_battery_file
            and manifest.capability_battery_hash):
        # Under sharding, each condition's battery is owned by exactly one
        # shard (the one holding the condition's first record) so batteries
        # run exactly once across the fleet; the merge recombines them.
        battery, battery_cancelled = _run_capability_battery(
            model, name, manifest, bundles, conditions, root, run_directory,
            should_cancel, _log, checkpoint, resuming,
            condition_filter=(set(plan.owned_conditions)
                              if plan is not None else None),
            latent_conditions=latent_conditions)
        if battery_cancelled:
            cancelled = True
            battery = None
    run_reporting.write_metrics_csv(records, run_directory, style=style)
    run_reporting.write_summaries_csv(records, run_directory)
    if not cancelled:
        run_reporting.write_report(name, manifest, records, run_directory, battery=battery,
                      style=style, numeric_parser=numeric_parser)
        resume_mod.clear_state(run_directory)
    _log(f"run ({len(records)} generations"
         f"{f', shard {shard.label}' if shard is not None else ''}"
         f"{', cancelled early — directory is resumable' if cancelled else ''}"
         f"{f', {writer.resumed_count} resumed' if resuming else ''}) "
         f"→ {run_directory}")
    return run_directory


def _run_capability_battery(model, name, manifest, bundles, conditions, root,
                            run_directory, should_cancel, _log, checkpoint,
                            resuming, condition_filter=None,
                            latent_conditions=()) -> tuple[dict, bool]:
    """Score the manifest's PINNED capability battery under EVERY condition of
    the run matrix — baseline, concept/steering, variant AND SAE latent arms —
    with each condition's intervention applied exactly as for that condition's
    generations (2026-07-13; Swift identical for the arms Swift has).

    "Every condition" is a claim this docstring once made while the loop ran
    only ``conditions`` + ``manifest.variant_conditions``: a declared SAE
    latent arm got NO battery row at all, so the one arm whose mechanism is
    least understood was also the only arm with no capability control
    (review finding 2, repaired 2026-08-13). Latent arms are now scored last
    — the same order the executor emits them in, which is the order the shard
    plan's key list assumes — with their ``latent_edits`` armed on both
    scoring back-ends, and their rows carry the condition's
    ``interventionState`` so a battery.jsonl row says which mechanism was
    live rather than being trusted to have had one.

    Battery generations are GATE EVIDENCE, not outcomes: they stream to a
    separate ``battery.jsonl`` (same append/skip/checkpoint discipline as
    generations.jsonl, so checkpoint/resume covers them record-for-record)
    and never enter generations.jsonl. Returns
    ``({condition: {"accuracy", "itemCount", "batteryHash"}}, cancelled)`` —
    the per-condition block ``_write_report`` stamps into report.json under
    the cross-engine key ``capabilityBattery``.

    Arming follows the battery's FORMAT (2026-08-13 repair). A format-2
    battery declares its own promptMode/systemPrompt/maxTokens and EVERY
    condition is scored under that same arming, so the intervention is the
    only thing that varies; its items are scored by answer-token logprob, so
    a condition cannot gain or lose accuracy by changing how much it writes.
    A legacy (format-1) battery is armed exactly as before — the manifest's
    context for baseline/steering conditions, the variant artifact's for
    variant conditions — so its pinned hash keeps its historical meaning; the
    asymmetry that produced the c19 "steering improves capability" reading is
    now logged as a contamination advisory rather than passing silently.
    """
    from . import battery as battery_mod, model_variant
    battery_file = manifest.capability_battery_file
    spec = battery_mod.load_spec(battery_file, root)
    # The pin gate, checked before the drift check because it is the more
    # fundamental complaint: an unpinnable format is wrong whatever it hashes
    # to. `experiment verify` and the Swift pin validator refuse it earlier;
    # this is the backstop for a manifest that was pinned before the format
    # existed, and it fires with the model already resident, which is exactly
    # why the earlier gates are worth having.
    unpinnable = battery_mod.pinnability_problem(spec)
    if unpinnable is not None:
        raise RuntimeError(unpinnable)
    items, digest = spec.items, spec.digest
    if digest != manifest.capability_battery_hash:
        raise RuntimeError(
            f"capability battery '{battery_file}' drifted from the pinned hash "
            f"(have {digest[:12]}…, pinned {manifest.capability_battery_hash[:12]}…)")
    writer = resume_mod.GenerationWriter(
        run_directory, verb="run", checkpoint=checkpoint, resume=resuming,
        log=_log, filename="battery.jsonl")
    cancelled = False
    advised: set[str] = set()

    def _score_condition(condition_name, injections, prompt_mode, system_prompt,
                         thinking, latent_edits=None,
                         intervention_state=None,
                         agent_system_prompt=None) -> bool:
        """Emit one battery record per item under the given arming. Returns
        False when a cancel was observed (partial condition kept, resumable).

        ``latent_edits``/``intervention_state`` are the SAE latent arm's
        mechanism and its provenance stamp. Both default to off and neither
        touches a non-latent condition's record bytes.

        ``system_prompt`` is the FORMAT-1 caller context — the study manifest's
        for baseline/steering/latent arms, the artifact's for a variant arm —
        preserved byte for byte so a legacy battery's pinned hash keeps its
        historical meaning. ``agent_system_prompt`` is the FORMAT-2 persona:
        composed ahead of the battery file's own declared arming, while the
        STUDY frame reaches a format-2 reading through neither channel
        (2026-08-24 battery-isolation ruling). Baseline therefore reads the
        battery bare, which is what makes it the control for the agent arms.
        """
        arming = battery_mod.resolve_arming(
            spec, prompt_mode=prompt_mode, system_prompt=system_prompt,
            qwen_thinking_enabled=thinking,
            agent_system_prompt=agent_system_prompt)
        advisory = battery_mod.contamination_advisory(spec, arming)
        if advisory and advisory not in advised:
            advised.add(advisory)
            _log(f"WARNING: {advisory}")
        generate_fn, choice_fn = choice_scoring.battery_backends(
            model, manifest.model_id, injections, latent_edits=latent_edits)
        for index, item in enumerate(items):
            if cancellation.observe_cancel(should_cancel, _log,
                               f"battery condition={condition_name} item={index}"):
                return False
            # ONE prompt id per item, used for both the resume probe and the
            # emitted record. ``resume.record_key`` keys on promptID, so a
            # skip() probe that disagreed with what emit() stores would match
            # nothing on resume: every completed item's forward pass would be
            # re-run and only emit()'s dedupe would suppress the duplicate row
            # — i.e. resume would silently stop saving the expensive half.
            prompt_id = battery_mod.item_prompt_id(item, index)
            if writer.skip(condition_name, index, prompt_id, 0,
                           resume_mod.KIND_SAMPLED):
                continue
            fields = battery_mod.score_item(
                spec, item, arming, generate_fn=generate_fn,
                choice_fn=choice_fn)
            record = {
                "condition": condition_name, "promptIndex": index,
                "promptID": prompt_id,
                "sampleIndex": 0,
                "prompt": item["prompt"], "answer": item["answer"],
            }
            if intervention_state is not None:
                # Latent arms only (2026-08-13): the SAME block generations.
                # jsonl carries, so one reader parses both files and a battery
                # row is self-describing about what was armed — release, SAE,
                # feature, layer, mode, β in latent units, and the pinned
                # repository commit. Absent for every other condition, whose
                # record bytes are therefore unchanged.
                record["interventionState"] = intervention_state
            if spec.isolated:
                # Format-2 provenance: what the reading was armed with, so a
                # stored battery.jsonl is self-describing.
                record["batteryFormat"] = spec.format_version
                record.update(arming.as_record_fields())
                record["batteryHash"] = digest
                record.update(fields)
            else:
                # Legacy records keep their exact historical key set and
                # order — an old battery.jsonl and a new one must diff clean.
                record["output"] = fields["output"]
                record["batteryHash"] = digest
                record["correct"] = fields["correct"]
            writer.emit(record)
        return True

    try:
        for condition in conditions:
            if condition_filter is not None \
                    and condition.name not in condition_filter:
                continue  # another shard owns this condition's battery
            injections = condition_execution.condition_injections(condition, bundles)
            if not _score_condition(condition.name, injections,
                                    manifest.prompt_mode, manifest.system_prompt,
                                    manifest.qwen_thinking_enabled):
                cancelled = True
                break
        if not cancelled:
            for vc in manifest.variant_conditions:
                if condition_filter is not None \
                        and vc.name not in condition_filter:
                    continue  # another shard owns this condition's battery
                try:
                    if vc.from_promotion:
                        vc, _ = forward_resolution.resolve_forward_variant(
                            vc, manifest, root, _log)
                    variant = (model_variant.ModelVariant.from_dict(vc.artifact)
                               if vc.artifact else model_variant.ModelVariant.from_file(
                                   paths.resolve(vc.artifact_path, root)))
                    injections = model_variant.variant_injections(variant)
                    adapter = model_variant.apply_adapter(model, variant, root=root)
                except (OSError, KeyError, ValueError, RuntimeError) as exc:
                    _log(f"battery: variant '{vc.name}' skipped: {exc}")
                    continue
                try:
                    completed = _score_condition(
                        vc.name, injections, variant.prompt_mode,
                        variant.system_prompt, variant.qwen_thinking_enabled,
                        agent_system_prompt=variant.system_prompt)
                finally:
                    model_variant.remove_adapter(model, adapter)
                if not completed:
                    cancelled = True
                    break
        if not cancelled:
            # SAE latent arms, last — the executor's own emission order, which
            # the shard plan's key list mirrors. The effective condition is
            # built by the SAME helper the run loop uses, so the battery can
            # never be armed differently from the generations it is the
            # control for.
            for spec_l, edit, provenance in latent_conditions:
                if condition_filter is not None \
                        and spec_l.name not in condition_filter:
                    continue  # another shard owns this condition's battery
                eff = condition_execution.effective_sae_latent_condition(
                    spec_l, edit, provenance, manifest)
                if not _score_condition(
                        eff.name, eff.injections, eff.prompt_mode,
                        eff.system_prompt, eff.qwen_thinking_enabled,
                        latent_edits=eff.latent_edits,
                        intervention_state=eff.intervention_state):
                    cancelled = True
                    break
        if cancelled:
            writer.interrupt(reason="cancel")
    finally:
        writer.close()

    summary: dict[str, dict] = {}
    if not cancelled:
        correct: dict[str, int] = {}
        seen: set[str] = set()
        for record in writer.records:
            seen.add(record["condition"])
            if record.get("correct"):
                correct[record["condition"]] = correct.get(record["condition"], 0) + 1
        summary = {condition: {
            "accuracy": correct.get(condition, 0) / len(items),
            "itemCount": len(items),
            "batteryHash": digest,
        } for condition in sorted(seen)}
        _log(f"capability battery ({len(items)} items × {len(summary)} "
             f"condition(s)) → battery.jsonl")
    return summary, cancelled
