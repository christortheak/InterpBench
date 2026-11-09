"""What an agent file says about itself beyond its injections: how hard the
agent thinks (public issue #16) and how it was made (public issue #17).

Kept beside :mod:`model_variant` rather than in it: that module's runtime
bodies are frozen by ``scripts/ci/audit-design-lazy-imports.py``, and nothing
here changes how a variant's injections are built. ``ModelVariant`` reads and
writes the two new fields (``reasoningEffort``, ``handmade``); every reader of
what they MEAN reads it here.

Swift twin: the ``ModelVariantArtifact`` extension in ``ModelVariantStore.swift``.
"""

from __future__ import annotations

import json

from . import prompt_render

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .model_variant import ModelVariant


#: The two ways an agent comes to exist, as every surface names them.
#: ``swept``: the file carries a ``promotion`` record, so a sweep selected its
#: settings and ``experiment promote`` minted it. ``handmade``: it carries
#: none, so a researcher chose its settings, with ``agent create`` (which
#: leaves a ``handmade`` record) or in the app's agent editor (which leaves
#: none). Swift twin: ``ModelVariantArtifact.Kind``.
AGENT_KIND_SWEPT = "swept"
AGENT_KIND_HANDMADE = "handmade"


def agent_file_problem(d: dict) -> str | None:
    """Why an agent file cannot be read as one agent, or ``None``.

    Two rules, both about fields that must say one thing:

    - ``reasoningEffort`` is in the study's vocabulary, and when the file also
      spells the legacy ``qwenThinkingEnabled`` the two agree (true exactly
      when the effort is not off). A disagreeing pair says the agent both
      thinks and does not, and which one a run honoured would depend on the
      reader.
    - ``promotion`` and ``handmade`` are two accounts of how the agent was
      made; an agent has at most one.

    Swift twin: ``ModelVariantArtifact.fileProblem``, with the same
    sentences."""
    effort = d.get(prompt_render.REASONING_EFFORT_KEY)
    if effort is not None:
        if not isinstance(effort, str) or effort not in prompt_render.REASONING_EFFORTS:
            return (f"This agent file's reasoningEffort {json.dumps(effort)} is not a "
                    "level SteerLab knows (off, on, low, medium, high, xhigh). Make "
                    "the agent again with agent create --reasoning-effort, or in "
                    "the app's agent editor.")
        thinking = d.get(prompt_render.LEGACY_THINKING_KEY)
        if thinking is not None and thinking is not (effort != prompt_render.REASONING_OFF):
            return (f"This agent file says reasoningEffort \"{effort}\" but "
                    f"qwenThinkingEnabled {json.dumps(thinking)}: one says the "
                    "agent thinks and the other says it does not. Make the agent "
                    "again with agent create --reasoning-effort, or in the app's "
                    "agent editor, so both fields say the same thing.")
    if isinstance(d.get("promotion"), dict) and isinstance(d.get("handmade"), dict):
        return ("This agent file carries both a promotion record (a sweep chose "
                "its settings) and a handmade record (a researcher chose them). "
                "An agent is made one way: make it again with agent create or "
                "experiment promote.")
    return None


def variant_reasoning_effort(variant: ModelVariant) -> str:
    """How hard this agent thinks, wherever it is cast: the ONE reading.

    The agent's declared ``reasoningEffort`` when it has one; otherwise the
    legacy boolean's meaning (true: the template's default effort, xhigh;
    false: off), exactly as every agent file was read before the field
    existed. An ordinary study calls it in
    ``condition_execution.effective_variant_condition``; a panel seat backed
    by an agent calls it where the panel runtime resolves the seat's
    thinking. Swift twin: ``ModelVariantArtifact.resolvedReasoningEffort``."""
    if variant.reasoning_effort is not None:
        return variant.reasoning_effort
    return prompt_render.resolve_reasoning_effort(None, variant.qwen_thinking_enabled)


def effort_reaches_renderer(effort: str, reasoning_max_tokens: int | None) -> bool:
    """Whether a run must hand ``effort`` to the renderer explicitly.

    Always with a reasoning budget (the declared protocol). Without one, only
    when the effort differs from what the legacy boolean renders: ``off`` and
    ``xhigh`` render identically either way, so every arm that ran before an
    agent could declare a level keeps its exact call, while an agent that
    declares ``medium`` (or ``on``) is rendered at ``medium`` and not at the
    boolean's xhigh."""
    if reasoning_max_tokens:
        return True
    return effort not in (prompt_render.REASONING_OFF,
                          prompt_render.LEGACY_THINKING_EFFORT)


def variant_effort_violations(variant: ModelVariant, capabilities=None) -> list[str]:
    """The template rule a study's ``set-sampling --reasoning-effort``
    applies, for this agent's declared effort: a non-off effort needs a
    thinking switch, and a LEVEL must be one the model's chat template
    accepts. The reasoning BUDGET is not checked here: it belongs to the
    study that casts the agent.

    ``capabilities`` is the model's record (``model_capabilities.resolve``);
    absent, the id heuristic answers, as everywhere else. A legacy agent with
    no declared effort is not judged anew: it runs as it always has.
    Swift twin: ``ModelVariantArtifact.effortViolations``."""
    effort = variant.reasoning_effort
    if effort is None or effort == prompt_render.REASONING_OFF:
        return []
    record = prompt_render.capabilities_for(variant.base_model_id,
                                            capabilities=capabilities)
    if not record.has_thinking_switch:
        return [prompt_render.effort_without_thinking_mode_reason(
            effort, variant.base_model_id)]
    if effort == prompt_render.REASONING_ON:
        return []
    return prompt_render.effort_level_violations(effort, variant.base_model_id, record)


def agent_kind(artifact: dict) -> str:
    """``swept`` or ``handmade``; see :data:`AGENT_KIND_SWEPT`."""
    return (AGENT_KIND_SWEPT if isinstance(artifact.get("promotion"), dict)
            else AGENT_KIND_HANDMADE)


def agent_origin(artifact: dict) -> dict:
    """What every surface says about an agent, read from the file alone: how
    it was made (``kind`` and the sentence a reader is shown) and how hard it
    thinks. The records behind the sentence stay in the file (``promotion``,
    ``handmade``). Swift twin: ``ModelVariantArtifact.origin``,
    with the same keys and sentences."""
    effort = artifact.get(prompt_render.REASONING_EFFORT_KEY)
    declared = isinstance(effort, str) and effort in prompt_render.REASONING_EFFORTS
    thinking = {"reasoningEffort": effort if declared else prompt_render.resolve_reasoning_effort(
                    None, bool(artifact.get(prompt_render.LEGACY_THINKING_KEY, False))),
                "reasoningEffortSource": ("agent" if declared else "thinkingSwitch")}
    return {**_agent_made(artifact), **thinking}


def _agent_made(artifact: dict) -> dict:
    kind = agent_kind(artifact)
    if kind == AGENT_KIND_SWEPT:
        promotion = artifact["promotion"]
        by = promotion.get("promotedBy")
        study = promotion.get("experiment")
        return {
            "kind": kind,
            "summary": ("A sweep chose this agent's settings: experiment promote "
                        "made it" + (f" from study {study}" if study else "")
                        + {"manualOverride": ", using a cell the researcher picked "
                                             "within the sweep.",
                           "criterion": ", using the cell the study's declared "
                                        "rule selected."}.get(
                                            by if isinstance(by, str) else "", "."))}
    record = artifact.get("handmade") if isinstance(artifact.get("handmade"), dict) else None
    steered = bool(artifact.get("injections")) or bool(artifact.get("interventionPolicies"))
    chose = ("The researcher chose its steering settings; no sweep selected them."
             if steered else "The researcher chose its settings.")
    if record is None:
        return {"kind": kind,
                "summary": ("This agent carries no promotion record, so nothing "
                            "records that a sweep selected its settings, and no "
                            "record of how it was made (the app's agent editor and "
                            "new agent versions leave none).")}
    reason = record.get("reason") if isinstance(record.get("reason"), str) else None
    made_with = record.get("madeWith") if isinstance(record.get("madeWith"), str) else None
    return {"kind": kind,
            "summary": ((f"Made by hand with {made_with}. " if made_with
                         else "Made by hand. ") + chose
                        + (f" Stated reason: {reason}" if reason
                           else " No reason was stated."))}


