"""Admission and resource requirements for the remaining pipeline stages.

Credential-dependent admission retains judging_custody's existing policy; stage
execution and model acquisition live outside this module.
"""
from __future__ import annotations
from .manifest import Manifest
from .judge_dispatch import judge_roster, evaluate_fanout_judge_models
from .judging_custody import missing_external_credentials as _missing_external_credentials



def pipeline_will_judge(manifest: Manifest, stages: list[str]) -> bool:
    """Whether any REMAINING stage of the chain will call judges: evaluate
    always does; a sweep does only under the judgeScore objective."""
    if "evaluate" in stages:
        return True
    if "sweep" in stages:
        sweep_block = manifest.raw.get("sweep")
        selection = (sweep_block.get("selection")
                     if isinstance(sweep_block, dict) else None)
        objective = (selection.get("objective")
                     if isinstance(selection, dict) else None)
        return (isinstance(objective, dict)
                and objective.get("metric") == "judgeScore")
    return False



def pipeline_inline_judging_preflight(manifest: Manifest,
                                       stages: list[str]) -> None:
    """The chain requires an IN-JOB-resolvable judging path (design contract,
    2026-07-18): deferral hands the selection/evaluation to a later Mac
    session, which cannot be a job dependency — so a sweep or evaluate stage
    that would defer refuses HERE, before the model loads. The manual
    two-phase flow (sweep → Mac judgment → promote → run) remains available
    outside the chain. ``stages`` is the REMAINING stage list: a resumed
    chain whose judged sweep already completed must not refuse because a
    transient judge key has since been cleared."""
    from . import sweep_selection
    # Finding 1 guard, fan-out era (2026-07-23): the chain holds ONE model,
    # so a judged SWEEP whose LOCAL judge declares a different model still
    # refuses HERE, before the model loads — the judge fan-out exists for
    # the post-generation EVALUATE stage only (a sweep's judging is
    # interleaved with the selection, not a separable post-stage). An
    # evaluate stage with such judges no longer refuses: it emits blinded
    # packets, the controller fans one worker job per distinct judge model,
    # and the merge resumes the chain.
    if "sweep" in stages:
        sweep_block = manifest.raw.get("sweep")
        selection_block = (sweep_block.get("selection")
                           if isinstance(sweep_block, dict) else None)
        objective_block = (selection_block.get("objective")
                           if isinstance(selection_block, dict) else None)
        if isinstance(objective_block, dict) \
                and objective_block.get("metric") == "judgeScore":
            offenders = []
            for j in (manifest.raw.get("judges") or []):
                if not (isinstance(j, dict) and j.get("name")):
                    continue
                if (str(j.get("kind") or "claude").strip()
                        or "claude") != "local":
                    continue
                declared = str(j.get("model") or "").strip()
                if declared and declared != manifest.model_id:
                    offenders.append(f"'{j['name']}' (model '{declared}')")
            if offenders:
                raise RuntimeError(
                    "the pipeline's sweep stage holds ONE model — the "
                    f"study model '{manifest.model_id}' — but local "
                    "judge(s) " + ", ".join(offenders) + " resolve to a "
                    "different model, which cannot load inside the chain "
                    "(the judge fan-out covers the evaluate stage only). "
                    "Leave a local judge's model empty to judge with the "
                    "study model, pin claude/openrouter judges, or select "
                    "on logprobShift")
    if "sweep" in stages:
        sweep_spec = manifest.raw.get("sweep")
        if isinstance(sweep_spec, dict):
            selection = sweep_spec.get("selection")
            criterion = sweep_selection.resolve_selection(selection)
            if criterion.metric == "judgeScore":
                objective = sweep_selection.resolve_objective(
                    criterion, selection,
                    judge_rubric_file=manifest.judge_rubric_file,
                    judge_rubric_hash=manifest.judge_rubric_hash,
                    judge_refs=manifest.judges,
                    judges_raw=(manifest.raw.get("judges") or []))
                if objective.defer_judging:
                    raise RuntimeError(
                        "the pipeline requires INLINE judging: the sweep's "
                        "judgeScore selection would DEFER (no credential "
                        "for its external judges on this host) and a chain "
                        "cannot wait for a Mac session — push a judge key "
                        "(app: Compute → External judge key), pin a local "
                        "judge, or select on logprobShift")
    if "evaluate" in stages:
        from .manifest import EVALUATE_WITHOUT_JUDGING_MESSAGE
        # The same effective-evaluation rule the evaluate stage resolves
        # with (2026-07-22): judges + a pinned rubric file count; a chain
        # with neither an evaluation block nor that pin pair refuses HERE,
        # with the verify gate's exact wording, before the model loads.
        spec, _source = manifest.effective_evaluation()
        if spec is None or spec.kind != "pairedJudge":
            raise RuntimeError(EVALUATE_WITHOUT_JUDGING_MESSAGE)
        roster = judge_roster(manifest, spec)
        missing = _missing_external_credentials(roster)
        if missing:
            raise RuntimeError(
                "the pipeline requires INLINE judging: evaluate would "
                f"DEFER (no credential for {'/'.join(sorted(missing))} "
                "judges on this host) and a chain cannot wait for a Mac "
                "session — push a judge key (app: Compute → External judge "
                "key), pin a local panel, or drop the evaluate stage")



def pipeline_needs_model(remaining: list[str], manifest: Manifest) -> bool:
    """Whether any remaining stage holds the GPU model. ``evaluate`` needs
    it only for LOCAL judges that judge INLINE (all resolving to the study
    model) — a fan-out evaluate only EMITS packets (CPU), and its judging
    runs in per-judge-model worker jobs (2026-07-23); ``promote``/
    ``analyze`` are CPU-only, so an all-CPU remainder (a requeue that died
    between promote and analyze) never loads the model at all."""
    from . import pipeline_spec as pspec
    if any(stage in pspec.GPU_STAGES for stage in remaining):
        return True
    if "evaluate" in remaining:
        from .manifest import EvaluationSpec
        if evaluate_fanout_judge_models(manifest):
            return False
        roster = judge_roster(manifest,
                               manifest.evaluation or EvaluationSpec())
        return any(ref.kind == "local" for ref in roster)
    return False
