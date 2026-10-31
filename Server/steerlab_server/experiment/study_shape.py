"""A study that measures the baseline agent alone (public issue #4).

Researchers replicating a published study on an unmodified model want a study
whose only subject is the baseline agent: the pinned base model, with no
vectors or adapters, at the study's sampling settings. The sanctioned spelling
is a declared ``studyType: agentComparison`` with no agent, injection, or SAE
latent arm. Its run executes one arm, ``baseline``, and its analysis has no
effects by design.

Standard library only, so the results export and the results page, which run
on an install that carries no engine, can ask the same question the engine
does. :mod:`.manifest` re-exports everything here. Swift twins:
``ExperimentStore.isBaselineOnlyStudy``, ``baselineOnlyStudyAdvisory``, and
``baselineOnlySummary``.
"""

from __future__ import annotations

#: The ``baselineOnlyStudy`` advisory's detail, said at verify, at freeze,
#: and by an analysis of such a study's run. Swift twin:
#: ``ExperimentStore.baselineOnlyStudyAdvisory``, byte for byte.
BASELINE_ONLY_STUDY_ADVISORY = (
    "This study measures the baseline agent alone: the pinned base model, "
    "with no vectors or adapters, at the study's sampling settings. With no "
    "comparison arm, analyze computes no effects; the run's summaries "
    "describe the baseline's responses. If the study is judged, use a rubric "
    "that codes each response (perResponseCoding), because paired judging "
    "compares a condition with the baseline and has nothing to pair here.")

#: The sentence the settings summary (``preregistration.md``) adds under
#: Conditions for a baseline-only study. Swift twin:
#: ``ExperimentStore.baselineOnlySummary``, byte for byte.
BASELINE_ONLY_SUMMARY = (
    "Baseline only: this study measures the baseline agent alone (the pinned "
    "base model, with no vectors or adapters). It has no comparison arm, so "
    "its analysis reports no effects.")


def baseline_only_study(d: dict) -> bool:
    """Whether a model-output study measures the BASELINE AGENT ALONE, by
    declaration.

    The shape: a declared ``studyType`` that keeps the concept machinery
    inert, no agent (variant) condition, no SAE latent arm, and no injection
    condition other than the canonical baseline (named ``baseline``, no
    slots). Its run executes exactly one arm, ``baseline``
    (``manifest.effective_conditions``). Concepts may be carried, as any agent
    comparison may carry them; they are inert and never execute.

    With no agent conditions, the only declaration that keeps the machinery
    inert is ``agentComparison`` (``manifest.concept_machinery_operative``):
    every other declared or derived type is a concept study. So the rule reads
    the declaration directly, and needs nothing beyond the standard library.

    ``verify`` accepts this shape with nothing attached, where an undeclared
    draft or a declared concept study with nothing attached is still the
    empty study: there the emptiness is the mistake it catches. A declared
    agentComparison that carries injection conditions and no agents is not
    this shape either; ``manifest.inert_conditions_problem`` refuses it."""
    if not isinstance(d, dict):
        return False
    if (d.get("studyKind") or "modelOutput") != "modelOutput":
        return False
    if d.get("studyType") != "agentComparison":
        return False
    if d.get("variantConditions") or d.get("saeLatentConditions"):
        return False
    for condition in d.get("conditions") or []:
        condition = condition if isinstance(condition, dict) else {}
        if condition.get("name") != "baseline" or condition.get("slots"):
            return False
    return True
