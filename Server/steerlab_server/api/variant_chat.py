"""Interactive generation against saved model variants."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from ..experiment import model_variant, paths
from ..experiment.generate import (
    generate,
    generate_messages,
    stream_generate,
    stream_generate_messages,
)


@dataclass
class VariantChatRequest:
    variant_path: str
    text: str | None = None
    # Message dicts ({role, content, seeded?, edited?}): `seeded: true` marks
    # a researcher-authored assistant turn (send-as-assistant); `edited: true`
    # marks a generated turn the researcher altered afterwards. Provenance for
    # the record only — rendering ignores both (a seeded/edited turn must be
    # byte-identical to a real one in the prompt the model sees).
    messages: list[dict] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    system_prompt: str | None = None
    strip_interventions: bool = False
    prompt_mode: str | None = None
    # Assistant-prefix continuation ("prefill"): requires `messages` ending in
    # an assistant turn; generation continues that turn mid-turn.
    continue_final_message: bool = False


def load_variant(path: str, expected_hash: str | None = None) -> tuple[model_variant.ModelVariant, str]:
    if expected_hash:
        with open(path, "rb") as handle:
            actual = hashlib.sha256(handle.read()).hexdigest()
        if actual != expected_hash:
            raise ValueError(
                f"variant artifact drifted (have {actual[:12]}, expected {expected_hash[:12]})")
    variant = model_variant.ModelVariant.from_file(path)
    return variant, path


def variant_from_dict(spec: dict) -> model_variant.ModelVariant:
    """Inline-spec seam: an ephemeral variant built from a request body dict —
    exactly the schema a stored artifact (and the /api/variants/upload payload)
    uses. Raises ``KeyError``/``ValueError``/``TypeError`` on a bad spec
    (``baseModelID`` is required), same as loading a stored artifact."""
    return model_variant.ModelVariant.from_dict(spec)


def _adapter_cache(model) -> model_variant.ChatAdapterCache:
    """The chat adapter cache for this model slot. Stored on the model wrapper
    itself so unloading the model (registry eviction) drops the cache — and the
    loaded adapter weights — with it."""
    cache = getattr(model, "_steerlab_chat_adapter_cache", None)
    if cache is None:
        cache = model_variant.ChatAdapterCache()
        model._steerlab_chat_adapter_cache = cache
    return cache


@contextmanager
def prepared_variant(model, variant: model_variant.ModelVariant,
                     *, strip_interventions: bool = False):
    injections = [] if strip_interventions else model_variant.variant_injections(variant)
    cache = _adapter_cache(model)
    # Chat-only adapter reuse: activate() loads the adapter once per
    # (directory, content-hash) and reuses it across turns; deactivate() parks
    # it disabled so a plain / stripped / different-variant generation on this
    # model is bit-identical to a never-adapted model. Measured experiment
    # paths keep their strict apply_adapter/remove_adapter lifecycle.
    if strip_interventions:
        cache.deactivate(model)
    else:
        cache.activate(model, variant)
    try:
        yield injections
    finally:
        cache.deactivate(model)


def render_chat_text(text: str | None, messages: list[dict] | None) -> str:
    """Human-readable transcript for the RESPONSE RECORD (never fed to the
    model when `messages` are present). Seeded/edited assistant turns are
    labeled so a researcher-authored (or researcher-altered) turn is never
    indistinguishable from a real generation in the saved record."""
    if text is not None:
        return text
    if not messages:
        return ""
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip() or "user"
        if role == "assistant":
            flags = [flag for flag in ("seeded", "edited")
                     if bool(message.get(flag))]
            if flags:
                role = f"assistant ({', '.join(flags)})"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _require_messages_for_continuation(request: VariantChatRequest) -> None:
    if request.continue_final_message and request.messages is None:
        raise ValueError(
            "continueFinalMessage requires a messages transcript ending in "
            "the assistant prefix to continue")


def generate_with_variant(model, variant: model_variant.ModelVariant,
                          request: VariantChatRequest,
                          *, on_chunk=None) -> dict:
    _require_messages_for_continuation(request)
    prompt = render_chat_text(request.text, request.messages)
    with prepared_variant(model, variant, strip_interventions=request.strip_interventions) as injections:
        kwargs = dict(
            model_id=model.model_id,
            max_tokens=request.max_tokens or 512,
            temperature=variant.temperature if request.temperature is None else request.temperature,
            injections=injections,
            prompt_mode=request.prompt_mode or variant.prompt_mode,
            system_prompt=(request.system_prompt if request.system_prompt is not None
                           else variant.system_prompt),
            qwen_thinking_enabled=variant.qwen_thinking_enabled,
            on_chunk=on_chunk,
        )
        if request.messages is not None:
            output = generate_messages(
                model, request.messages,
                continue_final_message=request.continue_final_message, **kwargs)
        else:
            output = generate(model, prompt, **kwargs)
    result = {
        "output": output,
        "modelID": model.model_id,
        "modelRevision": model.revision,
        "variant": variant.to_dict(),
        "stripInterventions": request.strip_interventions,
        "prompt": prompt,
    }
    # Provenance in the response record: which transcript turns were
    # researcher-authored (seeded) or researcher-altered after generation
    # (edited), and whether this generation continued a seeded assistant
    # prefix — a seeded or edited turn must never be indistinguishable from a
    # real generation in any saved record.
    if request.messages is not None:
        seeded = [i for i, m in enumerate(request.messages)
                  if isinstance(m, dict) and bool(m.get("seeded"))]
        if seeded:
            result["seededMessageIndices"] = seeded
        edited = [i for i, m in enumerate(request.messages)
                  if isinstance(m, dict) and bool(m.get("edited"))]
        if edited:
            result["editedMessageIndices"] = edited
        if request.continue_final_message:
            result["continuedFinalMessage"] = True
    return result


def evaluate_battery_with_variant(model, variant: model_variant.ModelVariant,
                                  spec, *, strip_interventions: bool = False) -> dict:
    """Score a capability battery against a variant — the app's server-workspace
    robustness wire (open issues §23).

    The point of this function is what it does NOT take: no prompt mode, no
    system prompt, no token cap. A format-2 battery's arming comes from the
    battery file and nothing else (``battery.resolve_arming`` with no
    instrument context), so the variant's own ``systemPrompt``/``promptMode``
    — which :func:`generate_with_variant` would apply — never reach the
    battery prompts. The variant contributes exactly ONE thing: its
    intervention (injections + adapter), which ``prepared_variant`` arms and
    ``strip_interventions=True`` removes for the baseline side. That is the
    arming isolation the generate wire could not express, and it is why this
    wire exists rather than a new mode on that one.

    Scoring itself is ``battery.evaluate`` — the same loader, the same arming
    resolver, the same ``score_item``, and the same two back-ends
    (``tasks._battery_backends``) ``experiment validate|run`` uses, so a
    reading taken here and a reading taken inside a study run are the same
    measurement.
    """
    from ..experiment import battery as battery_mod
    from ..experiment.tasks import _battery_backends

    arming = battery_mod.resolve_arming(spec)
    with prepared_variant(model, variant,
                          strip_interventions=strip_interventions) as injections:
        generate_fn, choice_fn = _battery_backends(model, model.model_id, injections)
        result = battery_mod.evaluate(spec, arming, generate_fn=generate_fn,
                                      choice_fn=choice_fn)
    result.update({
        "batteryFormat": spec.format_version,
        # The digest the reading was actually taken against — the caller's
        # pin, verified by the route before this ran.
        "batteryHash": spec.digest,
        "modelID": model.model_id,
        "modelRevision": model.revision,
        "stripInterventions": strip_interventions,
        # Structurally None while format 1 is refused upstream (a v2 reading
        # is isolated by definition). Emitted anyway so the response shape
        # does not change if the wire ever learns legacy arming.
        "advisory": battery_mod.contamination_advisory(spec, arming),
    })
    result.update(arming.as_record_fields())
    return result


def stream_with_variant(model, variant: model_variant.ModelVariant,
                        request: VariantChatRequest,
                        should_stop=None) -> Iterator[str]:
    _require_messages_for_continuation(request)
    prompt = render_chat_text(request.text, request.messages)
    with prepared_variant(model, variant, strip_interventions=request.strip_interventions) as injections:
        kwargs = dict(
            model_id=model.model_id,
            max_tokens=request.max_tokens or 512,
            temperature=variant.temperature if request.temperature is None else request.temperature,
            injections=injections,
            prompt_mode=request.prompt_mode or variant.prompt_mode,
            system_prompt=(request.system_prompt if request.system_prompt is not None
                           else variant.system_prompt),
            qwen_thinking_enabled=variant.qwen_thinking_enabled,
            should_stop=should_stop,
        )
        if request.messages is not None:
            yield from stream_generate_messages(
                model, request.messages,
                continue_final_message=request.continue_final_message, **kwargs)
        else:
            yield from stream_generate(model, prompt, **kwargs)


def request_from_body(body: dict, *, resolved_variant_path: str) -> VariantChatRequest:
    return VariantChatRequest(
        variant_path=resolved_variant_path,
        text=body.get("text"),
        messages=body.get("messages") if isinstance(body.get("messages"), list) else None,
        max_tokens=int(body.get("maxTokens", 512)),
        temperature=(float(body["temperature"]) if body.get("temperature") is not None else None),
        system_prompt=body.get("systemPrompt"),
        strip_interventions=bool(body.get("stripInterventions", False)),
        prompt_mode=body.get("promptMode"),
        continue_final_message=bool(body.get("continueFinalMessage", False)),
    )


def inline_variant_metadata(variant: model_variant.ModelVariant) -> dict:
    """Provenance stamp for an ephemeral inline spec: ``source: "inline"`` and
    deliberately NO path/hash — an inline chat generation must never masquerade
    as a stored hash-pinned variant (chat is exploration, not measurement)."""
    return {
        "source": "inline",
        "name": variant.name,
        "baseModelID": variant.base_model_id,
        "baseRevision": variant.base_revision,
        "injections": len(variant.injections),
        "adapters": len(variant.adapters),
        "promptMode": variant.prompt_mode,
        "temperature": variant.temperature,
    }


def variant_metadata(variant_path: str, variant: model_variant.ModelVariant) -> dict:
    try:
        rel = variant_path if variant_path.startswith("/") else paths.resolve(variant_path)
    except Exception:
        rel = variant_path
    return {
        "path": variant_path,
        "resolvedPath": rel,
        "name": variant.name,
        "baseModelID": variant.base_model_id,
        "baseRevision": variant.base_revision,
        "injections": len(variant.injections),
        "adapters": len(variant.adapters),
        "promptMode": variant.prompt_mode,
        "temperature": variant.temperature,
    }
