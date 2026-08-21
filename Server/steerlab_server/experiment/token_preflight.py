"""Exact prompt-token preflight against the PINNED model revision, without
loading weights (C1).

Before this, nothing checked a study's prompts ahead of time. The context
budget was enforced only inside generation — with the model resident, one item
at a time (``generate.check_context_budget``). On a cluster that means the
queue wait and a multi-minute 27B load happen first, and the run then dies on
whichever oversized item it reached, telling you about that one item and not
the other forty.

Two properties matter here and are easy to get wrong:

**Exactness.** Token counts belong to a specific tokenizer at a specific model
revision. A character-based estimate computed on the Mac cannot settle whether
a prompt fits a server-only model — it can only say "probably fine" or
"probably not". So the exact answer is computed where the pinned revision
lives, using ``AutoTokenizer``/``AutoConfig`` alone: no weights, no GPU, no
model slot.

**Completeness.** Report EVERY failing row, then refuse. A preflight that
stops at the first overflow, or that silently drops oversized items, converts
a data problem into either a slow guessing game or an unrecorded change to the
measured sample. If screening is ever offered it must be declared, and the
exclusions plus the surviving N stamped — never inferred.
"""

from __future__ import annotations

from . import prompt_render

# Matches generate.CONTEXT_BUDGET_RESERVE and Swift contextBudgetReserve.
CONTEXT_BUDGET_RESERVE = 16


class PreflightError(RuntimeError):
    pass


def _tokenizer(model_id: str, revision: str | None):
    """The pinned revision's tokenizer — weights are never touched."""
    from transformers import AutoTokenizer
    kwargs = {"revision": revision} if revision else {}
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def context_window(model_id: str, revision: str | None) -> int | None:
    """``max_position_embeddings`` (or its aliases) from the HF CONFIG alone.

    ``SteeredModel.context_window`` reads the same fields off a loaded
    model's config; this reads the config directly so the check can run
    before anything is loaded. None when the config declares none — the
    caller must then say the budget is unknown rather than assume one.
    """
    from transformers import AutoConfig
    kwargs = {"revision": revision} if revision else {}
    try:
        config = AutoConfig.from_pretrained(model_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 — surfaced as a readable refusal
        raise PreflightError(
            f"could not read the model config for '{model_id}'"
            f"{f' at revision {revision}' if revision else ''}: {exc}") from exc
    # Some multimodal configs nest the text config.
    candidates = [config, getattr(config, "text_config", None)]
    for source in candidates:
        if source is None:
            continue
        for field in ("max_position_embeddings", "n_positions", "seq_length",
                      "max_sequence_length"):
            value = getattr(source, field, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def preflight(prompts: list[dict], *, model_id: str, revision: str | None,
              prompt_mode: str = prompt_render.CHAT_ASSISTANT,
              system_prompt: str | None = None,
              qwen_thinking_enabled: bool = False,
              max_tokens: int = 0) -> dict:
    """Exact per-item token counts and the context verdict.

    Returns a report dict; it never raises for oversized items (that is the
    CALLER's decision to refuse), only for inputs it genuinely cannot read.
    """
    tokenizer = _tokenizer(model_id, revision)
    window = context_window(model_id, revision)
    budget = (window - max_tokens - CONTEXT_BUDGET_RESERVE
              if window else None)

    items: list[dict] = []
    for index, prompt in enumerate(prompts):
        text = prompt.get("prompt") or prompt.get("text") or ""
        if not text and prompt.get("transcript"):
            rendered = prompt_render.render_transcript(
                tokenizer, prompt["transcript"], model_id=model_id,
                prompt_mode=prompt_mode, system_prompt=system_prompt,
                qwen_thinking_enabled=qwen_thinking_enabled)
        else:
            rendered = prompt_render.render(
                tokenizer, text, model_id=model_id, prompt_mode=prompt_mode,
                system_prompt=system_prompt,
                qwen_thinking_enabled=qwen_thinking_enabled)
        count = rendered.prompt_token_count
        entry = {"id": prompt.get("id") or f"prompt-{index}",
                 "promptTokens": count}
        if budget is not None:
            entry["fits"] = count <= budget
            if count > budget:
                entry["overflow"] = count - budget
        items.append(entry)

    over = [i for i in items if i.get("fits") is False]
    return {
        "modelID": model_id,
        "revision": revision,
        "contextWindow": window,
        "maxTokens": max_tokens,
        "reservedTokens": CONTEXT_BUDGET_RESERVE,
        "promptBudget": budget,
        "itemCount": len(items),
        "items": items,
        "overflowCount": len(over),
        # EVERY failing row, not the first — see the module docstring.
        "overflowItems": over,
        "exact": True,
    }


def refusal(report: dict) -> str | None:
    """The complete refusal for a report with overflows, or None."""
    over = report.get("overflowItems") or []
    if not over:
        return None
    window = report.get("contextWindow")
    budget = report.get("promptBudget")
    worst = max(i["promptTokens"] for i in over)
    lines = ", ".join(
        f"{i['id']} ({i['promptTokens']} tokens, {i['overflow']} over)"
        for i in over)
    revision = report.get("revision")
    at_revision = f" at revision {revision[:12]}…" if revision else ""
    return (
        f"{len(over)} of {report.get('itemCount')} task prompts exceed the "
        f"context budget for {report.get('modelID')}{at_revision}"
        f": the model's context window is {window} tokens, and reserving "
        f"{report.get('maxTokens')} for generation plus "
        f"{report.get('reservedTokens')} leaves {budget} for the prompt. "
        f"Over-budget items: {lines}. The longest needs {worst} tokens. "
        "Shorten those items, lower Max tokens, or use a model with a larger "
        "context window — nothing is dropped automatically, because silently "
        "excluding items would change the measured sample without recording it.")
