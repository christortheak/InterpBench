"""Coordinate declared and legacy sweeps, preserving checkpoint and judge lifetimes.

This owner never imports the task compatibility facade.
"""
from __future__ import annotations
import csv
import json
import os
from contextlib import ExitStack
from typing import Callable
from ..steering import vector_math as vm
from . import lifecycle_gates, paths
from . import resume as resume_mod
from . import cancellation
from . import choice_scoring
from . import condition_execution
from . import execution_reporting
from . import generate
from . import layer_resolution
from . import manifest as manifest_module
from . import model_resources
from ..steering import stimulus_set
from . import rubric_inputs
from . import run_artifacts
from . import scoring
from . import study_admission
from . import sweep_evidence
from . import sweep_judging
from . import vector_materialization


def sweep(name: str, root: str | None = None, dtype: str = "auto",
          device: str | None = None, layer_fractions=(0.3, 0.4, 0.5, 0.6, 0.7),
          alphas=(-4.0, 2.0, 4.0, 8.0), prompt: str | None = None, *,
          model_provider=None, max_loaded: int | None = None,
          should_cancel: Callable[[], bool] | None = None,
          checkpoint=None, run_directory: str | None = None,
          on_run_directory=None,
          log=None) -> str:
    """Layer×alpha dose-response for picking per-concept settings on a dev split.

    When the manifest carries a ``sweep`` spec (the Swift-authored
    ``SweepSpec`` shape: grid, dev-prompts file, battery file, max tokens,
    optional ``selection`` criterion), the sweep converges with the Swift
    engine's: baseline cell + dev-prompt grid + capability battery per cell,
    then a selection step that appends ``<concept>-recommended`` (with full
    provenance) to a DRAFT manifest and writes ``recommendations.json``.
    Without a spec, the legacy single-prompt grid over the function arguments
    runs unchanged (backward compatibility for existing callers).

    ``max_loaded`` is the serving registry's resident-model capacity (the API
    route passes ``state.registry.max_loaded``); a judgeScore sweep whose
    local judge needs a SECOND resident model refuses at start on a one-slot
    server. ``None`` (the CLI/bundle path, which loads private in-process
    copies with no registry) skips that capacity check."""
    from . import sweep_selection
    manifest = manifest_module.Manifest.load(name, root)
    study_admission.verify_or_warn(manifest, root)
    condition_execution.advise_sweep_ignores_sae_latent(manifest, log or print)
    spec = manifest.raw.get("sweep")
    spec = spec if isinstance(spec, dict) else None
    # Resolve the selection criterion AND its objective's instrument config
    # BEFORE loading the model: an unknown metric, a missing/empty choice
    # file, missing judge pins, or a credential-less Claude judge must fail
    # at sweep start, never after minutes of generation.
    criterion = objective = None
    if spec is not None:
        selection = spec.get("selection")
        criterion = sweep_selection.resolve_selection(selection)
        sel_objective = (selection or {}).get("objective") or {}
        declared_choice = sel_objective.get("choicePromptsFile")
        declared_choice_map = sel_objective.get("choicePromptsFiles")
        objective = sweep_selection.resolve_objective(
            criterion, selection,
            choice_path=(paths.resolve(declared_choice, root)
                         if declared_choice else None),
            choice_paths=({c: paths.resolve(p, root)
                           for c, p in declared_choice_map.items()
                           if isinstance(p, str)}
                          if isinstance(declared_choice_map, dict) else None),
            concepts=tuple(c.name for c in manifest.concepts),
            judge_rubric_file=manifest.judge_rubric_file,
            judge_rubric_hash=manifest.judge_rubric_hash,
            judge_refs=manifest.judges,
            judges_raw=(manifest.raw.get("judges") or []))
        if criterion.metric == "judgeScore":
            sweep_judging.judge_preflight(manifest, max_loaded, log or print)
    # The judge stack spans the WHOLE sweep, so a foreign local judge loads
    # once rather than once per comparison (external review round 4, finding
    # 4 — the same fix evaluate got in 10adf47d8). Nested INSIDE the study
    # model's acquire so the judge's slot is released first: it is the
    # second resident model, and it should not outlive the work it serves.
    with model_resources.acquire_model(manifest, dtype, device, model_provider) as model, \
            ExitStack() as judge_stack:
        manifest = model_resources.pin_model_revision(name, manifest, model, root, log or print)
        if spec is not None:
            return _sweep_with_spec(name, manifest, model, root, spec,
                                    criterion, objective, model_provider,
                                    should_cancel, log or print,
                                    judge_stack=judge_stack,
                                    checkpoint=checkpoint,
                                    resume_directory=run_directory,
                                    on_run_directory=on_run_directory)
        return _sweep_impl(name, manifest, model, root, layer_fractions, alphas,
                           prompt, should_cancel, log or print)


def _sweep_with_spec(name, manifest, model, root, spec, criterion, objective,
                     model_provider, should_cancel, _log, *,
                     judge_stack=None, checkpoint=None,
                     resume_directory: str | None = None,
                     on_run_directory=None) -> str:
    """Manifest-spec'd sweep (Swift parity): per concept, a baseline cell over
    the dev prompts (marker density, distinct-2, battery accuracy), the
    layer×alpha grid in RESIDUAL-NORM units, then the data-declared selection
    criterion; the winner is appended to a draft manifest as
    ``<concept>-recommended`` with a ``selection`` provenance block, and
    ``recommendations.json`` carries the same provenance (or the reason no
    recommendation was made). Greedy single-sample throughout.

    The OBJECTIVE is data too: markerDensity reads the dev texts' marker
    rubric; logprobShift reads the answer-token instrument on the criterion's
    pinned choice rows; judgeScore reads paired-judge preference of each
    cell's dev texts against the baseline's (reusing the texts generated for
    the constraints — never generating twice). Whatever the objective, the
    capability/coherence constraints are computed from the generated dev
    texts and are never bypassed.

    Cancellation is observed between individual generations (dev prompts,
    battery items, choice rows, judge calls), not only at concept/layer
    boundaries, so cancel latency is at most one generation; the outcome is
    the same partial CSV + recommendations a boundary cancel writes. Every
    dev generation logs a one-line preview so decoherence is visible live."""
    from . import battery as battery_mod, experiment_store, sweep_selection as sel
    from .manifest import Condition, Slot

    # Fallback grid alphas are gentle by design (residual-norm units): hot
    # defaults decohere long generations before the researcher sees a single
    # dev text. See DEFAULT_SWEEP_* for the recalibration rationale; an
    # explicit grid in the spec always wins.
    layer_fractions = [float(f) for f in (spec.get("layerFractions")
                                          or layer_resolution.DEFAULT_SWEEP_LAYER_FRACTIONS)]
    alphas = [float(a) for a in (spec.get("alphas") or layer_resolution.DEFAULT_SWEEP_ALPHAS)]
    dev_prompts_file = spec.get("devPromptsFile") or "prompts/dev/dev-prompts.jsonl"
    battery_file = spec.get("batteryFile") or battery_mod.DEFAULT_BATTERY_FILE
    max_tokens = int(spec.get("maxTokens") or 80)

    dev = stimulus_set.load_texts(paths.resolve(dev_prompts_file, root))
    if not dev.texts:
        raise RuntimeError(f"dev prompts '{dev_prompts_file}' has no rows")
    battery_spec = battery_mod.load_spec(battery_file, root)
    battery_items, battery_hash = battery_spec.items, battery_spec.digest
    # The sweep's capability constraint is armed by the battery's FORMAT, not
    # by the study's prompt config (2026-08-13 repair). A legacy battery is
    # still armed from the manifest — which is exactly how a cell can win by
    # writing in the study's format rather than by keeping capability — so it
    # says so out loud.
    battery_arming = battery_mod.resolve_arming(
        battery_spec, prompt_mode=manifest.prompt_mode,
        system_prompt=manifest.system_prompt,
        qwen_thinking_enabled=manifest.qwen_thinking_enabled)
    _battery_advisory = battery_mod.contamination_advisory(
        battery_spec, battery_arming)
    if _battery_advisory:
        _log(f"WARNING: {_battery_advisory}")
    # C4: say what the declared tolerance can actually gate on, BEFORE the
    # grid runs. Battery accuracy moves in steps of 1/N, so a tolerance
    # between steps is not the tolerance that operates — and a sweep that
    # reports "tolerance 0.15" while gating at 0.2 is reporting a number that
    # did not decide anything.
    _resolution = sel.battery_resolution(
        len(battery_items), criterion.capability_tolerance)
    if _resolution is not None:
        _log(("⚠︎ " if _resolution.is_coarse else "") + _resolution.summary)
    # Coherence-length guard (c18 lesson): the distinct-2 floor is only as
    # good as the generation length it is measured at. Advisory, before the
    # grid spends anything; the length is stamped in selection provenance.
    _coherence_advisory = sel.coherence_length_advisory(
        max_tokens, manifest.max_tokens,
        declared=bool(spec.get("maxTokens")))
    if _coherence_advisory:
        _log(f"WARNING: {_coherence_advisory}")
    # Sweep-input pin enforcement at the moment of use (firewall closure
    # 2026-07-20, Swift twin: ExperimentTasks.sweep): a pinned dev-prompts /
    # battery hash must match the bytes the sweep is ABOUT to select on.
    # The refusal is what keeps the ex-post provenance stamp
    # (selection.devPromptsHash = dev.hash) and the manifest pin in
    # agreement — a mismatch refuses, never silently overwrites either.
    from .manifest import sweep_choice_pin_entries
    # Choice-instrument pins are enforced when the objective READS them —
    # a markerDensity/judgeScore sweep leaves a declared choice file inert,
    # so its pin has nothing live to disagree with.
    choice_pin_checks = []
    if objective is not None and objective.metric == "logprobShift":
        for concept, rel, pinned, label in sweep_choice_pin_entries(spec):
            if pinned:
                live = (objective.choice_set_for(concept).hash
                        if concept is not None
                        else (objective.choice_prompts_hash or ""))
                choice_pin_checks.append((live, pinned, rel, label))
    for live_hash, pinned, rel, label in (
            (dev.hash, spec.get("devPromptsHash"), dev_prompts_file,
             "sweep dev prompts"),
            (battery_hash, spec.get("batteryHash"), battery_file,
             "sweep capability battery"),
            *choice_pin_checks):
        if pinned and live_hash != pinned:
            # WP0 step 8: typed `sweepInputDrift` (same prose, same code).
            raise lifecycle_gates.refusing(
                lifecycle_gates.SWEEP_INPUT_DRIFT,
                f"{label} '{rel}' do not match the manifest's pinned hash "
                f"(have {live_hash[:12]}…, pinned {str(pinned)[:12]}…) — the "
                "sweep would select on data the study did not pin; restore "
                "the pinned file, or duplicate the study and re-declare the "
                "sweep",
                repair=(f"restore {rel} to its pinned bytes ; then "
                        "steerlab-server experiment sweep <name>  (a frozen "
                        "pin is never re-pinned: duplicate the study on the "
                        "Mac to change it)"))

    # Objective instruments arm BEFORE any grid work: the choice baseline
    # pass (and its option-length guard) and the judge panel resolve here,
    # so an instrument problem aborts the sweep at start, never mid-grid.
    # A cancel during arming lets TaskCancelled propagate: no run directory
    # exists yet, so aborting the whole sweep IS the consistent outcome (the
    # job runner stamps the cancel).
    # Per-concept instruments (choicePromptsFiles, 2026-08-02): each
    # concept's cells are scored on ITS OWN rows, so the baseline pass runs
    # once per DISTINCT file (keyed by content hash — the singular
    # declaration therefore keeps its single shared baseline pass).
    choice_baseline_by_hash: dict[str, dict[str, float]] = {}
    if objective.metric == "logprobShift":
        for concept_ref in manifest.concepts:
            chosen = objective.choice_set_for(concept_ref.name)
            if chosen.hash in choice_baseline_by_hash:
                continue
            _log(f"choice baseline ('{chosen.file}', "
                 f"{len(chosen.rows)} rows)…")
            choice_baseline_by_hash[chosen.hash] = choice_scoring.choice_target_logprobs(
                model, manifest, chosen.rows, [],
                should_cancel=should_cancel, log=_log)
    judge_rubric, judge_panel = "", []
    judge_rubric_hash = None
    deferred_judging = (objective.metric == "judgeScore"
                        and objective.defer_judging)
    if deferred_judging and criterion.control_apply_to == "topK":
        raise RuntimeError(
            "topK control selection is not supported with deferred judging — "
            "the completion verb applies the winner-only control; use "
            "controls.applyTo 'winner', or push a judge key so judging runs "
            "inline")
    if objective.metric == "judgeScore" and not deferred_judging:
        judge_rubric, judge_panel = sweep_judging.sweep_judge_panel(
            manifest, model, model_provider, root, _log,
            judge_stack=judge_stack)
    elif deferred_judging:
        # Two-phase Claude-judged sweep (key-custody design 2026-07-18):
        # this server has no Anthropic credential BY POLICY. The sweep still
        # resolves the pinned rubric — its TEXT rides in the judging packets
        # to the Mac verbatim, its hash pins them.
        judge_rubric, judge_rubric_hash, _rubric_file = rubric_inputs.resolve_rubric(
            manifest, root, _log)
        from . import response_coding
        response_coding.refuse_if_coding(
            judge_rubric, context="the sweep's judgeScore objective",
            rubric_file=_rubric_file)
        _log("judgeScore panel is claude-only and this server holds no "
             "credential (by design): generating everything and emitting "
             "blinded judging packets — judge them on the Mac, then "
             "complete-judgment computes the selection")

    bundles = vector_materialization.extract_all(model, manifest, root)
    resumed_rows: list[dict] = []
    resumed_recommendations: dict = {}
    if resume_directory is not None:
        # Resuming a walltime-checkpointed sweep (2026-08-03): the same gate
        # every resumable verb passes — refuses complete directories
        # (recommendations.json is the sweep's marker), never-checkpointed
        # directories, and checkpoints from a different verb.
        from . import resume as resume_mod
        resume_mod.require_resumable(resume_directory, verb="sweep")
        # And the manifest must be the SAME manifest — a checkpoint from one
        # epoch must not silently continue under edited data.
        hash_file = os.path.join(resume_directory, "experiment-hash.txt")
        try:
            with open(hash_file, encoding="utf-8") as handle:
                checkpointed_hash = handle.read().strip()
        except OSError:
            checkpointed_hash = ""
        if checkpointed_hash and checkpointed_hash != manifest.content_hash():
            raise RuntimeError(
                f"cannot resume sweep in {resume_directory}: the manifest "
                "changed since the checkpoint (content hash "
                f"{checkpointed_hash[:12]}… → "
                f"{manifest.content_hash()[:12]}…) — re-run the sweep fresh")
        run_directory = resume_directory
        resumed_rows, resumed_recommendations = sweep_evidence.load_sweep_progress(
            run_directory)
        _log(f"resuming checkpointed sweep: {len(resumed_rows)} completed "
             f"cell row(s), {len(resumed_recommendations)} concept(s) "
             "already selected")
    else:
        run_directory = paths.make_unique_run_directory(
            f"exp-{name}-sweep", root)
        run_artifacts.write_config_snapshot(manifest, run_directory, "sweep", model=model,
                               root=root, log=_log)
        # Persist the sweep's re-derived vectors as first-class extraction
        # artifacts (same helper as extract/validate — never a parallel
        # writer): the sweep run itself then carries recipe-matching
        # sidecars, so "sweep then promote" needs no separate extract run
        # for promote's artifact matcher to find.
        vector_materialization.persist_vectors(bundles, manifest, model, run_directory)
    if on_run_directory is not None:
        on_run_directory(run_directory)
    # Dev generations already recorded (resume dedupe) — fresh runs start
    # empty, a resumed directory seeds from its own durable record.
    dev_generation_keys = sweep_evidence.load_dev_generation_keys(run_directory)

    def _append_progress(entry: dict) -> None:
        """Durably append one progress line (flush + fsync): a checkpoint
        may only count work whose record is already on disk."""
        with open(sweep_evidence.sweep_progress_path(run_directory), "a",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _checkpoint_if_requested(where: str) -> None:
        """Exit-85 checkpoint between units of work (deferred-judging sweeps
        excluded: their packets live in memory until the end, so parking
        mid-grid would lose them — they run to completion or fail)."""
        if checkpoint is None or not getattr(checkpoint, "requested", False):
            return
        if deferred_judging:
            return
        from . import resume as resume_mod
        completed = len(rows)
        resume_mod.write_state(
            run_directory, run_id=os.path.basename(run_directory),
            verb="sweep", completed_records=completed, reason="signal")
        _log(f"sweep checkpoint at {where}: {completed} cell row(s) durable "
             f"→ exit {resume_mod.CHECKPOINT_EXIT_CODE}")
        raise resume_mod.CheckpointRequested(
            run_directory, "sweep", completed)

    def _gen(prompt_text: str, injections, tokens: int) -> str:
        return generate.generate(model, prompt_text, model_id=manifest.model_id,
                        max_tokens=tokens, temperature=0.0, injections=injections,
                        prompt_mode=manifest.prompt_mode,
                        system_prompt=manifest.system_prompt,
                        qwen_thinking_enabled=manifest.qwen_thinking_enabled)

    def _battery_accuracy(injections, label: str) -> float:
        """Battery accuracy under the given injections, armed by the battery
        (format 2) or by the study manifest (legacy). A cancel is observed
        between items (``TaskCancelled``); item texts are NOT previewed —
        volume would drown the dev previews that matter."""
        total = len(battery_items)
        correct = 0
        generate_fn, choice_fn = choice_scoring.battery_backends(
            model, manifest.model_id, injections)
        for i, item in enumerate(battery_items, start=1):
            cancellation.cancel_checkpoint(should_cancel, _log,
                               f"{label} battery {i}/{total}")
            if battery_mod.score_item(battery_spec, item, battery_arming,
                                      generate_fn=generate_fn,
                                      choice_fn=choice_fn)["correct"]:
                correct += 1
        return correct / total

    def _dev_texts(injections, label: str, record=None) -> list[str]:
        """Dev-prompt generations under the given injections. A cancel is
        observed between prompts (``TaskCancelled``), every generation logs
        a one-line preview so decoherence is visible live, and ``record``
        — ``(kind, concept, layer, alpha)`` — names the cell each text is
        durably appended to ``dev-generations.jsonl`` under, AS GENERATED,
        so the prose evidence survives a later kill."""
        texts: list[str] = []
        total = len(dev.texts)
        for i, prompt_text in enumerate(dev.texts, start=1):
            cancellation.cancel_checkpoint(should_cancel, _log, f"{label} dev {i}/{total}")
            text = _gen(prompt_text, injections, max_tokens)
            _log(f'{label} dev {i}/{total}: "{execution_reporting.preview_line(text)}"')
            if record is not None:
                kind, concept, layer, alpha = record
                sweep_evidence.append_dev_generation(
                    run_directory, kind=kind, concept=concept, layer=layer,
                    alpha=alpha, prompt_index=i - 1, text=text,
                    seen=dev_generation_keys)
            texts.append(text)
        return texts

    def _text_stats(texts, rubric) -> tuple[float, float, float]:
        """(mean marker density, mean distinct-2, mean word count) over
        already-generated dev texts."""
        n = len(texts)
        return (sum(rubric.density(t) if rubric else 0.0 for t in texts) / n,
                sum(scoring.distinct_bigram_ratio(t) for t in texts) / n,
                sum(scoring.word_count(t) for t in texts) / n)

    def _cell_objective(concept_name, condition_tag, injections, density,
                        texts, baseline_texts) -> float:
        """The declared objective's value for one cell — the dev texts are
        the SAME ones the constraints were computed from, and the choice
        rows are the CONCEPT's own instrument."""
        if objective.metric == "logprobShift":
            chosen = objective.choice_set_for(concept_name)
            return choice_scoring.mean_logprob_shift(
                choice_scoring.choice_target_logprobs(model, manifest, chosen.rows,
                                        injections,
                                        should_cancel=should_cancel, log=_log),
                choice_baseline_by_hash[chosen.hash])
        if objective.metric == "judgeScore":
            if deferred_judging:
                return None  # judged on the Mac; complete-judgment selects
            return sweep_judging.judge_preference(judge_panel, judge_rubric, condition_tag,
                                     texts, baseline_texts, prompts=dev.texts,
                                     should_cancel=should_cancel, log=_log)
        return density

    extra_metric = objective.metric != "markerDensity"
    # Column order is now IDENTICAL on both engines (the Swift twin's
    # `SweepRunCatalog.csvHeader`). `distinct2Ratio` is written for every cell
    # whichever coherence rule is in force — it is the number the
    # baseline-relative floor gates on — and `lengthInflated` flags a cell whose
    # mean output ran more than 1.5× the baseline's. The flag is REPORTED, never
    # gated on.
    fieldnames = ["concept", "layer", "alpha", "markerDensity", "distinct2",
                  "distinct2Ratio", "words", "lengthInflated",
                  "batteryAccuracy"]
    if extra_metric:
        fieldnames.append("objective")

    rows: list[dict] = list(resumed_rows)
    recommendations: dict = dict(resumed_recommendations)
    deferred_packets: list[dict] = []
    deferred_map: dict[str, dict] = {}
    deferred_selection: dict = {}
    cancelled = False
    # (baseline texts, baseline battery accuracy) — generated once, shared
    # by every concept (see the baseline comment inside the loop).
    shared_baseline: tuple[list[str], float] | None = None
    for concept_name, bundle in bundles.items():
        if concept_name in recommendations:
            _log(f"{concept_name}: already selected before the checkpoint — "
                 "skipping")
            continue
        # Completed work from a checkpointed run, keyed exactly as the grid
        # iterates (baseline is layer -1, alpha 0).
        resumed_cells = {
            (int(r["layer"]), float(r["alpha"])): r
            for r in resumed_rows if r.get("concept") == concept_name}
        _checkpoint_if_requested(f"concept={concept_name}")
        if cancellation.observe_cancel(should_cancel, _log, f"concept={concept_name}"):
            cancelled = True
            break
        rubric = scoring.MarkerRubric.from_directory(paths.concept_directory(concept_name, root))
        if rubric is None:
            _log(f"{concept_name}: no markers.json — expression scores will be 0")
        layer_count = bundle.vectors.layer_count
        layers = layer_resolution.concept_sweep_layers(
            next(c for c in manifest.concepts if c.name == concept_name),
            bundle.vectors,
            layer_resolution.resolve_sweep_layers(layer_count, layer_fractions), _log)
        residual_norms = bundle.residual_norm_per_layer or []
        if not residual_norms:
            raise RuntimeError(
                f"sweep spec alphas are in residual-norm units but concept "
                f"'{concept_name}' has no residual norms — pin a neutral corpus "
                "and re-extract")

        # Baseline cell: no injection (layer -1, alpha 0 — Swift parity).
        # Baseline texts are kept: judgeScore pairs every cell against them.
        # The GENERATIONS are concept-independent (no injection, same dev
        # prompts, same battery) and are generated ONCE for the whole sweep
        # (review 2026-08-02, P2 — a multi-concept sweep regenerated them
        # per concept); only the marker DENSITY is per-concept, rescored
        # from the cached texts with this concept's rubric.
        baseline_prev = resumed_cells.get((-1, 0.0))
        if baseline_prev is not None and objective.metric != "judgeScore":
            # Resume: this concept's baseline row is already durable, and a
            # non-judge objective never reads the baseline TEXTS — reuse the
            # recorded stats instead of regenerating (judgeScore pairs cells
            # against baseline texts, so it regenerates them).
            baseline_texts = []
            baseline_accuracy = float(baseline_prev["batteryAccuracy"])
            baseline_density = float(baseline_prev["markerDensity"])
            baseline_distinct = float(baseline_prev["distinct2"])
            baseline_words = float(baseline_prev["words"])
        else:
            try:
                if shared_baseline is None:
                    shared_baseline = (
                        _dev_texts([], "baseline",
                                   record=("baseline", None, -1, 0.0)),
                        _battery_accuracy([], "baseline"))
            except cancellation.TaskCancelled:
                cancelled = True
                break
            baseline_texts, baseline_accuracy = shared_baseline
            baseline_density, baseline_distinct, baseline_words = _text_stats(
                baseline_texts, rubric)
        baseline_objective = sel.baseline_metric(objective.metric, baseline_density)
        baseline = sel.BaselineCell(metric=baseline_objective,
                                    distinct2=baseline_distinct,
                                    battery_accuracy=baseline_accuracy)
        if baseline_prev is None:
            # The baseline is its own reference, so its ratio is 1 and it is
            # never length-inflated.
            baseline_row = {"concept": concept_name, "layer": -1, "alpha": 0,
                            "markerDensity": baseline_density,
                            "distinct2": baseline_distinct,
                            "distinct2Ratio": 1.0,
                            "words": baseline_words,
                            "lengthInflated": False,
                            "batteryAccuracy": baseline_accuracy}
            if extra_metric:
                baseline_row["objective"] = baseline_objective
            rows.append(baseline_row)
            _append_progress({"kind": "row", "row": baseline_row})
        _log(f"{concept_name} baseline: density {baseline_density:.4g}, "
             f"distinct2 {baseline_distinct:.4g}, battery {baseline_accuracy:.4g}")

        cells: list[sel.SweepCell] = []
        for layer in layers:
            if cancellation.observe_cancel(should_cancel, _log,
                               f"concept={concept_name} layer={layer}"):
                cancelled = True
                break
            vector = bundle.vectors.per_layer[layer]
            vector_norm = vm.l2_norm(vector)
            # Same rule as the condition and variant paths (2026-08-28 audit,
            # F7/F13): a layer the denominator table does not reach refuses,
            # where this site used to clamp to the last entry and dose the
            # deepest sweep cells with a shallower layer's number.
            residual = condition_execution.residual_norm_at(
                residual_norms, layer, artifact=concept_name,
                where=f"concept '{concept_name}'")
            for alpha in alphas:
                done = resumed_cells.get((int(layer), float(alpha)))
                if done is not None:
                    # Completed before the checkpoint: rebuild the selection
                    # cell from the durable row — no regeneration.
                    cells.append(sel.SweepCell(
                        layer=layer, alpha=alpha,
                        metric=(float(done["objective"]) if extra_metric
                                else float(done["markerDensity"])),
                        distinct2=float(done["distinct2"]),
                        battery_accuracy=float(done["batteryAccuracy"]),
                        # A journal row from before the words column leaves
                        # the length unrecorded — the winner's lengthInflated
                        # stamp is then absent rather than invented.
                        words=(float(done["words"])
                               if done.get("words") not in (None, "")
                               else None)))
                    continue
                _checkpoint_if_requested(
                    f"concept={concept_name} L{layer} α{alpha:g}")
                if cancellation.observe_cancel(should_cancel, _log,
                                   f"concept={concept_name} layer={layer} "
                                   f"alpha={alpha:g}"):
                    cancelled = True
                    break
                raw_alpha = vm.norm_unit_scale(alpha, residual, vector_norm)
                cell = [generate.CellInjection(layer=layer, vector=vector, alpha=raw_alpha)]
                try:
                    texts = _dev_texts(cell, f"L{layer} α{alpha:g}",
                                       record=("cell", concept_name, layer,
                                               alpha))
                    density, distinct, words = _text_stats(texts, rubric)
                    accuracy = _battery_accuracy(cell, f"L{layer} α{alpha:g}")
                    metric_value = _cell_objective(
                        concept_name,
                        f"sweep:{concept_name}:L{layer}:a{alpha:g}", cell,
                        density, texts, baseline_texts)
                except cancellation.TaskCancelled:
                    # Mid-cell cancel: the incomplete cell is dropped; rows
                    # for completed cells keep today's partial-CSV behavior.
                    cancelled = True
                    break
                ratio = sel.distinct2_ratio(distinct, baseline_distinct)
                inflated = sel.length_inflated(words, baseline_words)
                row = {"concept": concept_name, "layer": layer,
                       "alpha": alpha, "markerDensity": density,
                       "distinct2": distinct,
                       "distinct2Ratio": "" if ratio is None else ratio,
                       "words": words, "lengthInflated": inflated,
                       "batteryAccuracy": accuracy}
                if extra_metric:
                    row["objective"] = metric_value
                rows.append(row)
                _append_progress({"kind": "row", "row": row})
                cells.append(sel.SweepCell(layer=layer, alpha=alpha,
                                           metric=metric_value, distinct2=distinct,
                                           battery_accuracy=accuracy,
                                           words=words))
                if deferred_judging:
                    sweep_judging.emit_judging_packets(
                        deferred_packets, deferred_map, concept_name, "cell",
                        layer, alpha, dev.texts, texts, baseline_texts,
                        judge_rubric_hash)
                    if criterion.matched_norm_random_margin is not None:
                        # The control belongs to the WINNER — unknown until
                        # the Mac judges — so a deferred sweep with a margin
                        # generates the control for EVERY cell now, while
                        # the GPU is still allocated (bounded, known cost);
                        # completion consumes only the winning cell's.
                        control_condition = Condition(
                            name=f"{concept_name}-recommended",
                            slots=[Slot(concept=concept_name, layer=layer,
                                        alpha=alpha)],
                            band_width=1, alpha_in_norm_units=True,
                            control_type="randomMatchedNorm")
                        control_injections = condition_execution.condition_injections(
                            control_condition, bundles)
                        try:
                            control_texts = _dev_texts(
                                control_injections,
                                f"control L{layer} α{alpha:g}",
                                record=("control", concept_name, layer,
                                        alpha))
                        except cancellation.TaskCancelled:
                            cancelled = True
                            break
                        sweep_judging.emit_judging_packets(
                            deferred_packets, deferred_map, concept_name,
                            "control", layer, alpha, dev.texts,
                            control_texts, baseline_texts, judge_rubric_hash)
                _log(f"{concept_name} L{layer} α{alpha:g}: density {density:.4g}, "
                     f"distinct2 {distinct:.4g}"
                     + ("" if ratio is None else f" ({ratio:.4g}× baseline)")
                     + f", battery {accuracy:.4g}"
                     + (f", ⚠︎ output {words:.4g} words vs baseline "
                        f"{baseline_words:.4g}" if inflated else "")
                     + (f", {objective.metric} {metric_value:.4g}"
                        if extra_metric and metric_value is not None else ""))
            if cancelled:
                break
        if cancelled:
            break  # partial grid — never select from incomplete evidence

        if deferred_judging:
            deferred_selection[concept_name] = {
                "baseline": {"markerDensity": baseline_density,
                             "distinct2": baseline_distinct,
                             "words": baseline_words,
                             "batteryAccuracy": baseline_accuracy},
                "cells": [{"layer": c.layer, "alpha": c.alpha,
                           "distinct2": c.distinct2,
                           "words": c.words,
                           "batteryAccuracy": c.battery_accuracy}
                          for c in cells],
            }
            recommendations[concept_name] = (
                "awaiting judgment — blinded packets emitted; judge them on "
                "the Mac, then complete-judgment computes the selection")
            continue

        best = sel.select_cell(cells, baseline, criterion)
        if best is None:
            # Say WHICH gate refused. "Capability/coherence" is one of two
            # possible reasons and often the wrong one — a grid whose cells
            # are all eligible but none of which beats the baseline objective
            # is a different result entirely, and reporting it as a gate
            # failure sends the researcher to loosen a tolerance that was
            # never binding.
            recommendations[concept_name] = sel.no_selection_reason(
                cells, baseline, criterion)
            continue

        control_info = None
        controls_evaluated: list[dict] = []
        if criterion.matched_norm_random_margin is not None:
            # Control cells: a deterministic random direction norm-matched
            # to the concept vector at the candidate's layer, same alpha,
            # same dev prompts. Built through _condition_injections with the
            # SAME condition shape a `controlType: randomMatchedNorm` study
            # cell uses, so seeding/norm-matching are identical and
            # reproducible. The control evaluates the SAME declared
            # objective as the grid.
            #
            # applyTo "winner" (historical): the argmax cell alone. "topK"
            # (2026-08-03): walk the top K promotable cells and promote the
            # FIRST that beats its own control — one disruption-artifact
            # corner can no longer veto a grid containing a legitimate
            # winner (observed live in the first stances sweep).
            _checkpoint_if_requested(f"concept={concept_name} controls")
            margin = criterion.matched_norm_random_margin
            candidates = (
                sel.ranked_candidates(cells, baseline, criterion,
                                      criterion.control_top_k)
                if criterion.control_apply_to == "topK" else [best])
            promoted = None
            for candidate in candidates:
                control_condition = Condition(
                    name=f"{concept_name}-recommended",
                    slots=[Slot(concept=concept_name, layer=candidate.layer,
                                alpha=candidate.alpha)],
                    band_width=1, alpha_in_norm_units=True,
                    control_type="randomMatchedNorm")
                control_injections = condition_execution.condition_injections(
                    control_condition, bundles)
                try:
                    if objective.metric == "logprobShift":
                        chosen = objective.choice_set_for(concept_name)
                        control_metric = choice_scoring.mean_logprob_shift(
                            choice_scoring.choice_target_logprobs(
                                model, manifest, chosen.rows,
                                control_injections,
                                should_cancel=should_cancel, log=_log),
                            choice_baseline_by_hash[chosen.hash])
                    else:
                        control_texts = _dev_texts(
                            control_injections,
                            f"control L{candidate.layer} α{candidate.alpha:g}",
                            record=("control", concept_name, candidate.layer,
                                    candidate.alpha))
                        control_density, _, _ = _text_stats(
                            control_texts, rubric)
                        control_metric = control_density
                        if objective.metric == "judgeScore":
                            control_metric = sweep_judging.judge_preference(
                                judge_panel, judge_rubric,
                                f"sweep-control:{concept_name}:"
                                f"L{candidate.layer}:a{candidate.alpha:g}",
                                control_texts, baseline_texts,
                                prompts=dev.texts,
                                should_cancel=should_cancel, log=_log)
                except cancellation.TaskCancelled:
                    # A winner without its verified control is incomplete
                    # evidence — no recommendation for this concept.
                    cancelled = True
                    break
                passed = sel.control_passes(
                    candidate.metric, control_metric, margin)
                controls_evaluated.append({
                    "layer": candidate.layer, "alpha": candidate.alpha,
                    "metricValue": candidate.metric,
                    "controlMetricValue": control_metric,
                    "passed": passed})
                _log(f"{concept_name} control L{candidate.layer} "
                     f"α{candidate.alpha:g}: cell {candidate.metric:g} vs "
                     f"control {control_metric:g} → "
                     + ("passes" if passed else "fails"))
                if passed:
                    promoted = candidate
                    control_info = {
                        "type": "randomMatchedNorm",
                        "metricValue": control_metric,
                        "margin": margin,
                        # Recipe stamp (cross-engine contract string);
                        # unstamped = legacy (see RANDOM_VECTOR_ALGORITHM).
                        "randomVectorAlgorithm": condition_execution.RANDOM_VECTOR_ALGORITHM}
                    break
            if cancelled:
                break
            if promoted is None:
                if criterion.control_apply_to == "topK":
                    message = sel.top_k_control_failure_message(
                        controls_evaluated, margin)
                else:
                    message = sel.control_failure_message(
                        best.metric,
                        controls_evaluated[0]["controlMetricValue"], margin)
                recommendations[concept_name] = message
                _append_progress({"kind": "recommendation",
                                  "concept": concept_name, "block": message})
                _log(f"{concept_name}: {message}")
                continue
            # The PROMOTED cell may not be the argmax under topK — every
            # downstream stamp (metrics, winningCell, the minted condition)
            # describes the cell that actually passed its control.
            best = promoted

        if objective.metric == "markerDensity":
            metrics_block = {"markerDensity": best.metric,
                             "baselineDensity": baseline_density}
        else:
            baseline_key = ("baselineJudgeScore"
                            if objective.metric == "judgeScore"
                            else "baselineLogprobShift")
            metrics_block = {objective.metric: best.metric,
                             baseline_key: baseline_objective}
        metrics_block.update({"distinct2": best.distinct2,
                              "batteryAccuracy": best.battery_accuracy,
                              "baselineBatteryAccuracy": baseline_accuracy})
        # The coherence gate's own evidence travels WITH the metrics it
        # adjudicated — previously the ratio and the length flag lived only
        # in sweep.csv, so a promotion certificate inheriting this block
        # could not show what its own floor gated on.
        metrics_block.update(sel.selection_report_metrics(
            best.distinct2, baseline_distinct, best.words, baseline_words))
        selection_block: dict = {
            "sweepRun": os.path.basename(run_directory),
            # Per-concept instruments: the provenance block pins the choice
            # file THIS concept's cells were scored on.
            "criterion": criterion.to_dict(objective, concept=concept_name),
            "devPromptsHash": dev.hash,
            # The length the coherence floor was measured at (c18 lesson):
            # a reader comparing this against the manifest's maxTokens can
            # tell whether the winning cell's distinct-2 is study-relevant
            # evidence or short-generation evidence.
            "devMaxTokens": max_tokens,
            "winningCell": {"layer": best.layer, "alpha": best.alpha},
            "metrics": metrics_block,
        }
        if objective is not None and objective.metric == "judgeScore":
            # Where the judging ran and through which credential is
            # RECORDED provenance, not ambient fact (mirrors the deferred
            # path's "judgedOn": "client"). External judges on this branch
            # judged inline on the server.
            selection_block["judgedOn"] = "server"
            if any(ref.kind != "local" for ref in manifest.judges):
                from . import judge_credentials
                try:
                    credential = judge_credentials.resolve()
                except ValueError:
                    credential = None
                if credential is not None:
                    selection_block["judgeCredential"] = {
                        "kind": credential.kind, "source": credential.source}
        if control_info is not None:
            selection_block["control"] = control_info
        if criterion.control_apply_to == "topK":
            # Which cells were controlled and how each fared — the argmax
            # being rejected by its own control is provenance, not trivia.
            selection_block["controlsEvaluated"] = controls_evaluated
        recommendations[concept_name] = selection_block
        _append_progress({"kind": "recommendation", "concept": concept_name,
                          "block": selection_block})

    csv_path = os.path.join(run_directory, "sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    from . import resume as resume_mod
    if cancelled:
        # A cancelled sweep is an explicit PARTIAL, never "complete"
        # (review 2026-08-03 round 2, P1): recommendations.json is the
        # sweep's completion marker, so writing it here would freeze a
        # partial grid as done. A non-deferred sweep parks exactly like a
        # cancelled study run — the progress journal already holds every
        # completed row and recommendation, so the ordinary resume path
        # finishes the grid (its partial recommendations live only in the
        # journal until then, which is correct: a partial screen must not
        # feed promote's newest-sweep fallback). Deferred sweeps are
        # outside the checkpoint system by design; a cancel leaves them a
        # plain partial to rerun fresh.
        if not deferred_judging:
            resume_mod.write_state(
                run_directory, run_id=os.path.basename(run_directory),
                verb="sweep", completed_records=len(rows), reason="cancel")
        _log(f"sweep cancelled after {len(rows)} cell row(s) — partial, "
             + ("resumable" if not deferred_judging else "rerun it fresh")
             + f" → {run_directory}")
        return run_directory
    if deferred_judging and deferred_packets:
        sweep_judging.write_deferred_judging(
            run_directory, name, manifest, criterion, objective,
            deferred_packets, deferred_map, deferred_selection, dev.hash,
            judge_rubric, judge_rubric_hash, _log)
    # Clear the resume pointer BEFORE projection: projection mutates the
    # manifest, so a crash after it would leave a "resumable" directory the
    # epoch guard must refuse — cleared first, that crash reads as an honest
    # partial instead (fresh rerun; add_condition is idempotent).
    resume_mod.clear_state(run_directory)
    # Draft-condition projection happens ONLY here, after the sweep's whole
    # checkpointable life (review 2026-08-03, P1): projecting per concept
    # mid-sweep mutated the manifest between checkpoints, so a multi-concept
    # draft sweep's own recommendations changed content_hash() and the
    # resume epoch guard refused its checkpoint. Resumed concepts
    # (journal-recovered blocks) project here too, and add_condition
    # replaces by name, so re-projection is idempotent.
    if manifest.status == "draft":
        projected = [
            {"name": f"{rec_concept}-recommended",
             "slots": [{"concept": rec_concept,
                        "layer": block["winningCell"]["layer"],
                        "alpha": block["winningCell"]["alpha"]}],
             "bandWidth": 1, "alphaInNormUnits": True, "selection": block}
            for rec_concept, block in recommendations.items()
            if isinstance(block, dict)
            and isinstance(block.get("winningCell"), dict)
        ]  # failure/awaiting strings project nothing
        if projected:
            # One load, one save — a partial projection cannot exist
            # (review 2026-08-03 round 3, P2).
            experiment_store.add_conditions(name, projected, root)
            for entry in projected:
                _log(f"recommended condition '{entry['name']}' written "
                     "into draft manifest")
    elif any(isinstance(b, dict) for b in recommendations.values()):
        _log(f"manifest is {manifest.status} — recommendations reported only")
    # recommendations.json is the sweep's COMPLETION MARKER
    # (resume.completion_file_for): written atomically LAST, so its presence
    # guarantees every required artifact — csv, deferred packets, draft
    # projections — already exists, and the complete-pointer path can trust
    # it without reconciliation (review 2026-08-03 round 2, P1).
    marker_tmp = os.path.join(run_directory, "recommendations.json.tmp")
    with open(marker_tmp, "w", encoding="utf-8") as handle:
        json.dump(recommendations, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(marker_tmp,
               os.path.join(run_directory, "recommendations.json"))
    _log(f"sweep ({len(rows)} cells) → {run_directory}")
    return run_directory


def _sweep_impl(name, manifest, model, root, layer_fractions, alphas, prompt,
                should_cancel, _log) -> str:
    bundles = vector_materialization.extract_all(model, manifest, root)
    run_directory = paths.make_unique_run_directory(f"exp-{name}-sweep", root)
    run_artifacts.write_config_snapshot(manifest, run_directory, "sweep", model=model,
                           root=root, log=_log)
    sweep_prompt = prompt or (manifest.task_description or "Write a short paragraph.")

    rows = []
    cancelled = False
    for concept_name, bundle in bundles.items():
        if cancellation.observe_cancel(should_cancel, _log, f"concept={concept_name}"):
            cancelled = True
            break
        rubric = scoring.MarkerRubric.from_directory(paths.concept_directory(concept_name, root))
        layer_count = bundle.vectors.layer_count
        # The legacy grid carries the same zero-injection hazard as the spec'd
        # one, so it asks the same question (see concept_sweep_layers). Its own
        # fraction→layer resolution is kept verbatim (declared order, duplicates
        # and all) so a non-SAE concept sweeps exactly the cells it always did.
        legacy_layers = [min(layer_count - 1, int(layer_count * fraction))
                         for fraction in layer_fractions]
        for layer in layer_resolution.concept_sweep_layers(
                next(c for c in manifest.concepts if c.name == concept_name),
                bundle.vectors, legacy_layers, _log):
            for alpha in alphas:
                cell = generate.CellInjection(layer=layer, vector=bundle.vectors.per_layer[layer], alpha=alpha)
                text = generate.generate(model, sweep_prompt, model_id=manifest.model_id,
                                max_tokens=manifest.max_tokens, temperature=0.0,
                                injections=[cell], prompt_mode=manifest.prompt_mode,
                                system_prompt=manifest.system_prompt,
                                qwen_thinking_enabled=manifest.qwen_thinking_enabled)
                # Same qualitative-record rule as the spec'd sweep: the text
                # this row was scored on is evidence, not disposable.
                sweep_evidence.append_dev_generation(
                    run_directory, kind="cell", concept=concept_name,
                    layer=layer, alpha=alpha, prompt_index=0, text=text)
                rows.append({
                    "concept": concept_name, "layer": layer, "alpha": alpha,
                    "markerDensity": rubric.density(text) if rubric else "",
                    "distinct2": scoring.distinct_bigram_ratio(text), "words": scoring.word_count(text),
                })
    csv_path = os.path.join(run_directory, "sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["concept", "layer", "alpha",
                                                    "markerDensity", "distinct2", "words"])
        writer.writeheader()
        writer.writerows(rows)
    _log(f"sweep ({len(rows)} cells{', cancelled early' if cancelled else ''}) → {run_directory}")
    return run_directory
