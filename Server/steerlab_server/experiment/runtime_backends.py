"""Bind battery scoring to a condition and explicit execution capabilities.

Backends load lazily; importing this module does not load a model runtime.
Injected callables use the same interface as generate and score_options.
"""
from __future__ import annotations

def battery_backends(model, model_id: str, injections, latent_edits=None, *,
                     generate_text=None, score_choices=None):
    """The two scoring back-ends a battery item can use, bound to one
    condition's arming (2026-08-13 battery repair).

    ``generate_fn`` and ``choice_fn`` both take the battery's own
    :class:`battery.BatteryArming` — NOT the study manifest's rendering
    context — so a format-2 battery is scored identically under baseline,
    steering, and variant conditions, and the intervention is the only thing
    that differs. ``choice_fn`` goes through the answer-token logprob
    instrument (the stepped KV-cache path a study's categorical outcome
    endpoints use), so
    nothing is generated and the score cannot move with response length or
    format compliance. Split out so tests can fake both boundaries.

    ``latent_edits`` arms a TRUE SAE latent intervention for the condition
    being scored (2026-08-13 review finding 2). It is a SEPARATE mechanism
    from ``injections`` for the same reason it is separate everywhere else,
    and — exactly like ``_execute_condition``'s ``latent_kwargs`` — it is
    threaded as a keyword ONLY when non-empty, so every non-latent
    condition's ``generate``/``score_options`` call is byte-for-byte the call
    it was before this argument existed. Without it a latent arm's battery
    scored the UNSTEERED model and filed the result under the latent
    condition's name: a capability control that measured nothing."""
    if generate_text is None:
        from .generate import generate as generate_text
    if score_choices is None:
        from .logprob import score_options as score_choices

    latent_kwargs = {"latent_edits": list(latent_edits)} if latent_edits else {}

    def generate_fn(prompt: str, arming) -> str:
        return generate_text(model, prompt, model_id=model_id,
                        max_tokens=arming.max_tokens, temperature=0.0,
                        injections=injections, **latent_kwargs,
                        prompt_mode=arming.prompt_mode,
                        system_prompt=arming.system_prompt,
                        qwen_thinking_enabled=arming.qwen_thinking_enabled)

    def choice_fn(prompt: str, options, arming):
        result = score_choices(
            model, prompt, list(options), model_id=model_id,
            injections=injections, **latent_kwargs,
            prompt_mode=arming.prompt_mode,
            system_prompt=arming.system_prompt,
            qwen_thinking_enabled=arming.qwen_thinking_enabled)
        return result.selected, result.probability

    return generate_fn, choice_fn
