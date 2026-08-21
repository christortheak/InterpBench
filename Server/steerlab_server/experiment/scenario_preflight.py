"""Scenario-aware token preflight for multi-agent (panel) studies.

The standard ``token_preflight`` measures independent task prompts, so it does
not apply here and the multi-agent path skipped it. But a panel is the study
kind MOST likely to overflow: with ``routing: all`` every agent's visible
context accumulates every prior turn, and the built-in three-judge template is
sixteen turns ending in two 2048-token opinion drafts. Without a preflight the
only backstop is the in-generation context check, which means dying part-way
through turn 14 after a queue wait and a multi-minute 27B load — and telling
you about that one turn and none of the others.

What a preflight can honestly know
----------------------------------
It cannot know what the model will WRITE, and the written text is most of what
accumulates. So this reports two different things and treats them differently:

* **Floor** — the turn's rendered template plus the shared materials, with no
  prior context and no interpolated outputs. This is the smallest that turn can
  possibly be. A floor over budget is CERTAIN to fail whatever the model
  generates, so it refuses.
* **Projection** — the floor plus a worst-case allowance for everything that
  accumulates: each prior turn routed to this speaker contributes its full
  ``maxTokens``, as does every ``{{outputs.<label>}}`` the template
  interpolates. A projection over budget MIGHT fail, so it advises and names
  the levers rather than blocking a run that may well fit.

That split follows the plan's minimise-refusals rule: refuse only on what is
computed and certain, advise on what is bounded and likely. Accounting is in
tokens rather than by synthesising fake transcript text, so the projection
costs one tokenizer pass per turn and no generation at all.

Cross-engine note: the walk reuses ``multi_agent._render_prompt`` and
``_routed_ids`` directly rather than reimplementing them, so it cannot drift
from what the runner actually does.
"""

from __future__ import annotations

import re

from . import multi_agent, prompt_render, token_preflight

#: Per-context-entry framing the runner adds around each routed output
#: (``[label] title — speaker``, or the longer own-authored form, plus blank
#: lines). Small, but it is charged once per accumulated turn, so at sixteen
#: turns it is not nothing. A flat allowance covers both forms: the difference
#: between them is a handful of tokens on an entry that reserves hundreds.
CONTEXT_ENTRY_OVERHEAD = 24

_OUTPUT_REF = re.compile(r"\{\{outputs\.([^}]+)\}\}")


def preflight(scenario, *, model_id: str, revision: str | None = None,
              prompt_mode: str = prompt_render.CHAT_ASSISTANT) -> dict:
    """Per-turn token accounting for one play-through of ``scenario``.

    Raises ``token_preflight.PreflightError`` only when it genuinely cannot
    read the tokenizer or config — never for an oversized turn, which is the
    caller's decision.
    """
    tokenizer = token_preflight._tokenizer(model_id, revision)
    window = token_preflight.context_window(model_id, revision)

    agents = {a.id: a for a in scenario.agents}
    # Worst-case tokens accumulated in each agent's visible context so far.
    context_tokens: dict[str, int] = {a.id: 0 for a in scenario.agents}
    # Reserved output size per output label, for {{outputs.*}} interpolation.
    label_tokens: dict[str, int] = {}

    turns: list[dict] = []
    for index, turn in enumerate(scenario.turns):
        speaker = agents.get(turn.speaker_agent_id)
        if speaker is None:  # validate() catches this; do not crash here
            continue
        reserve = max(1, turn.max_tokens or scenario.max_tokens)
        budget = (window - reserve - token_preflight.CONTEXT_BUDGET_RESERVE
                  if window else None)

        # Floor: the turn as authored, with NOTHING accumulated. Contract
        # turns walk the SAME renderer (empty context, empty outputs) rather
        # than an approximation of it — the sandwich's scaffold wording is
        # hundreds of tokens on a wide panel, and a floor that omitted it
        # would under-report exactly the turns most at risk.
        floor_text = multi_agent._render_prompt(
            scenario, turn, "", speaker.name, {})
        floor = prompt_render.render(
            tokenizer, floor_text, model_id=model_id, prompt_mode=prompt_mode,
            system_prompt=speaker.system_prompt or None).prompt_token_count

        # Projection: floor + what will accumulate into it. A contract's
        # `inputs` are read exactly the way `{{outputs.X}}` is — straight into
        # the prompt, no routing involved — so they are charged identically.
        projected = floor
        if turn.include_speaker_context:
            projected += context_tokens.get(speaker.id, 0)
        read_labels = list(_OUTPUT_REF.findall(turn.prompt_template))
        if turn.contract is not None:
            read_labels += list(turn.contract.inputs)
        for label in read_labels:
            projected += label_tokens.get(label, 0)

        entry = {
            "turnID": turn.id,
            "turnIndex": index + 1,
            "title": turn.title,
            "speaker": speaker.name,
            "reservedTokens": reserve,
            "floorTokens": floor,
            "projectedPromptTokens": projected,
            "promptBudget": budget,
        }
        if budget is not None:
            entry["floorOverBy"] = max(0, floor - budget)
            entry["projectedOverBy"] = max(0, projected - budget)
        turns.append(entry)

        # This turn's output now accumulates for everyone it routes to.
        label = turn.output_label.strip() or f"turn_{index + 1}"
        label_tokens[label] = reserve
        for agent_id in multi_agent._routed_ids(turn, scenario.agents):
            context_tokens[agent_id] = (
                context_tokens.get(agent_id, 0) + reserve + CONTEXT_ENTRY_OVERHEAD)

    return {
        "scenarioName": scenario.name,
        "modelID": model_id,
        "revision": revision,
        "contextWindow": window,
        "turnCount": len(turns),
        "turns": turns,
        "certainOverflow": [t for t in turns if t.get("floorOverBy")],
        "projectedOverflow": [
            t for t in turns if t.get("projectedOverBy") and not t.get("floorOverBy")],
    }


def _levers(scenario) -> str:
    return ("Lower that turn's Max tokens, narrow its routing so fewer turns "
            "accumulate into it, trim the shared materials, or move to a model "
            "with a larger context window.")


def refusal(report: dict, scenario=None) -> str | None:
    """Refusal for turns that cannot fit WHATEVER the model generates."""
    over = report.get("certainOverflow") or []
    if not over:
        return None
    window = report.get("contextWindow")
    lines = "; ".join(
        f"turn {t['turnIndex']} '{t['title']}' ({t['speaker']}) needs "
        f"{t['floorTokens']} tokens against a {t['promptBudget']} budget — "
        f"{t['floorOverBy']} over"
        for t in over)
    return (
        f"{len(over)} of {report.get('turnCount')} turns in panel "
        f"'{report.get('scenarioName')}' cannot fit on "
        f"{report.get('modelID')}: the context window is {window} tokens, and "
        f"each turn reserves its own Max tokens for generation. This is the "
        f"turn's own template and shared materials alone, before ANY "
        f"deliberation accumulates, so it will not fit whatever the model "
        f"writes. {lines}. {_levers(scenario)}")


def advisory(report: dict, scenario=None) -> str | None:
    """Warning for turns that fit today but are projected to overflow once the
    deliberation accumulates. Never blocks: the projection charges every prior
    routed turn its FULL Max tokens, and real turns usually write less."""
    over = report.get("projectedOverflow") or []
    if not over:
        return None
    lines = "; ".join(
        f"turn {t['turnIndex']} '{t['title']}' ({t['speaker']}) projects "
        f"{t['projectedPromptTokens']} tokens against a {t['promptBudget']} "
        f"budget — {t['projectedOverBy']} over"
        for t in over)
    return (
        f"token preflight: {len(over)} turn(s) may overflow once the "
        f"deliberation accumulates. This is a worst-case projection — every "
        f"prior routed turn is charged its full Max tokens, and real turns "
        f"usually write less — so the run is NOT blocked. {lines}. "
        f"{_levers(scenario)}")
