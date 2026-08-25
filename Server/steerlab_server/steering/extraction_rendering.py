"""HOW a stimulus string reaches the model during extraction — the rendering.

Until this module existed the answer was an undeclared implementation detail:
this engine's extraction tokenized the RAW stimulus string
(``extractor._encode`` ≈ bare ``tokenizer(text)``), while measured generation
rendered the same text through the model's chat template with
``add_generation_prompt=True`` and ``add_special_tokens=False``
(:mod:`steerlab_server.experiment.prompt_render`). Extracting one concept both
ways yields two DIFFERENT directions — cosine ≈ 0.18 mid-network, per-layer
norms diverging ~10× with the ordering inverting across depth — that both
separate the classes near-perfectly on a probe (ledger §26, measured
2026-08-23). Which direction a study got, and what the α denominator equalled,
therefore depended on something nobody declared. That is exactly what the
freeze discipline exists to forbid, so the choice is now explicit, pinned, and
sweepable.

**Absent is LEGACY RAW, and is never retro-applied.** A recipe with no
``extractionRendering`` renders exactly as it always has, its manifest bytes
are byte-identical to before, and its recipe-identity hash is unchanged — the
same discipline ``residualNormConvention: wholeCorpusMean-v1`` follows. Only an
explicitly declared CHAT-TEMPLATE rendering changes a recipe's identity; an
explicitly declared ``raw`` is the legacy semantics said out loud, and
canonicalizes away (see :func:`canonical_identity_fragment`).

Cross-engine contract — the Swift twin is
``Sources/SteeringKit/Extraction/ExtractionRendering.swift``. Same JSON key
(``extractionRendering``), same mode spellings, same inner key names, same
defaults:

.. code-block:: json

    {"mode": "raw"}
    {"mode": "chatTemplate", "addGenerationPrompt": true,
     "qwenThinkingEnabled": false, "systemPrompt": null}

``mode`` is the only required key. ``addGenerationPrompt`` defaults to true,
``qwenThinkingEnabled`` to false, ``systemPrompt`` to absent — the parameters
the MEASUREMENT renderer (``prompt_render.render``) actually varies, and no
more. Family handling (Gemma has no system role; Qwen's ``enable_thinking``)
is derived from the pinned model id by that same renderer, so it is not
re-declared here: there is ONE rendering definition, and extraction calls it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

#: Today's behavior, and what an absent declaration means.
RAW = "raw"
#: The model family's chat template, rendered by
#: :func:`steerlab_server.experiment.prompt_render.render`.
CHAT_TEMPLATE = "chatTemplate"

MODES = (RAW, CHAT_TEMPLATE)

#: This engine's name in refusals about an unsupportable rendering form.
ENGINE = "python-hf-transformers"

#: The flag both engines' CLIs write a declaration with, and the name every
#: refusal that asks for one must use. Swift twin:
#: ``ExtractionRendering.declarationFlag``.
DECLARATION_FLAG = "--extraction-rendering"


class ExtractionRenderingError(ValueError):
    """A rendering declaration this engine cannot honor. Typed and carrying a
    repair, per house style — never a silent fallback to raw, because a silent
    fallback is precisely the ambiguity this module removes."""


@dataclass(frozen=True)
class ExtractionRendering:
    """A declared extraction rendering. Construct via :func:`from_json`."""

    mode: str = RAW
    add_generation_prompt: bool = True
    qwen_thinking_enabled: bool = False
    system_prompt: str | None = None

    @property
    def is_raw(self) -> bool:
        return self.mode == RAW

    @property
    def label(self) -> str:
        """Short human label for logs and refusal messages."""
        if self.is_raw:
            return "raw"
        bits = [f"addGenerationPrompt={str(self.add_generation_prompt).lower()}"]
        if self.qwen_thinking_enabled:
            bits.append("qwenThinkingEnabled=true")
        if self.system_prompt:
            bits.append("systemPrompt=set")
        return "chatTemplate (" + ", ".join(bits) + ")"

    def to_dict(self) -> dict:
        """The stamped/declared JSON block. Defaults are written EXPLICITLY on
        a chat-template rendering (the artifact must say what it did without a
        reader knowing this module's defaults); ``systemPrompt`` is omitted
        when absent, matching the sidecar's drop-None convention."""
        if self.is_raw:
            return {"mode": RAW}
        block = {
            "mode": CHAT_TEMPLATE,
            "addGenerationPrompt": self.add_generation_prompt,
            "qwenThinkingEnabled": self.qwen_thinking_enabled,
        }
        if self.system_prompt is not None:
            block["systemPrompt"] = self.system_prompt
        return block


RAW_RENDERING = ExtractionRendering()


def from_json(value) -> ExtractionRendering:
    """Parse a manifest/sidecar ``extractionRendering`` block.

    ``None`` (the key absent) is LEGACY RAW. A bare string is accepted as the
    mode alone, so a hand-written manifest may say ``"extractionRendering":
    "chatTemplate"``. An unknown mode is a typed refusal naming this engine —
    never a fallback, because falling back to raw would reproduce the exact
    silent-default bug this option exists to close.
    """
    if value is None:
        return RAW_RENDERING
    if isinstance(value, str):
        value = {"mode": value}
    if not isinstance(value, dict):
        raise ExtractionRenderingError(
            f"extractionRendering must be an object or a mode string, got "
            f"{type(value).__name__} — repair: declare "
            f'{{"mode": "raw"}} or {{"mode": "chatTemplate"}}')
    mode = value.get("mode", RAW)
    if mode not in MODES:
        raise ExtractionRenderingError(
            f"extraction rendering '{mode}' is not supported by the {ENGINE} "
            f"engine — repair: declare one of {', '.join(MODES)} "
            f"(absent means legacy raw)")
    if mode == RAW:
        # A raw rendering takes no parameters: accepting them silently would
        # let a manifest look like it declared something it cannot get.
        extra = sorted(k for k in value
                       if k not in ("mode",) and value.get(k) is not None)
        if extra:
            raise ExtractionRenderingError(
                f"extractionRendering mode 'raw' takes no parameters but "
                f"declares {', '.join(extra)} — repair: drop the parameters, "
                f"or declare mode 'chatTemplate' if you meant the template "
                f"rendering")
        return RAW_RENDERING
    add_generation_prompt = value.get("addGenerationPrompt")
    qwen_thinking = value.get("qwenThinkingEnabled")
    system_prompt = value.get("systemPrompt")
    if add_generation_prompt is not None and not isinstance(add_generation_prompt, bool):
        raise ExtractionRenderingError(
            "extractionRendering.addGenerationPrompt must be a boolean — "
            "repair: use true or false")
    if qwen_thinking is not None and not isinstance(qwen_thinking, bool):
        raise ExtractionRenderingError(
            "extractionRendering.qwenThinkingEnabled must be a boolean — "
            "repair: use true or false")
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise ExtractionRenderingError(
            "extractionRendering.systemPrompt must be a string or absent — "
            "repair: drop the key when the render carries no system prompt")
    return ExtractionRendering(
        mode=CHAT_TEMPLATE,
        add_generation_prompt=(True if add_generation_prompt is None
                               else add_generation_prompt),
        qwen_thinking_enabled=(False if qwen_thinking is None
                               else qwen_thinking),
        system_prompt=system_prompt)


def parse_declaration(value) -> ExtractionRendering | None:
    """Parse an `extractionRendering` DECLARATION — a CLI flag value or a
    route body field — into what a manifest should store.

    The option shipped 2026-08-24 with every consumer live and no writer at
    all: the refusals said "declare extractionRendering" and could name no
    command to declare it with. This is the writer's parser, and it validates
    at DECLARATION time rather than at extraction, so a malformed or
    unsupportable rendering is answered while the person is still typing.

    Accepts the JSON object the schema documents, a bare mode word
    (``chatTemplate``) as the shell-friendly form of the same thing, or an
    already-decoded object (the route's case). Every rule is
    :func:`from_json`'s — one definition of what a rendering may say.

    **Returns ``None`` for a RAW declaration, and that is the hash contract.**
    An explicit ``{"mode": "raw"}`` is the legacy semantics said out loud, so
    it must write exactly what saying nothing writes: absent. The same rule
    :func:`canonical_identity_fragment` already applies to the identity hash.
    Swift twin: ``ExtractionRendering.declared(_:)``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ExtractionRenderingError(
                f"{DECLARATION_FLAG} was given no value — repair: declare a "
                f'rendering, e.g. {DECLARATION_FLAG} \'{{"mode": '
                f'"chatTemplate"}}\' — or omit the flag entirely, which means '
                "the legacy raw rendering")
        if text.startswith("{") or text.startswith('"'):
            try:
                value = json.loads(text)
            except ValueError:
                raise ExtractionRenderingError(
                    f"{DECLARATION_FLAG} is not valid JSON: {text} — repair: "
                    f"quote the whole object in the shell, e.g. "
                    f'{DECLARATION_FLAG} \'{{"mode": "chatTemplate"}}\'')
        else:
            value = {"mode": text}
    rendering = from_json(value)
    return None if rendering.is_raw else rendering


def canonical_identity_fragment(rendering: "ExtractionRendering | None") -> dict | None:
    """The recipe-identity payload for a rendering, or ``None`` to OMIT the key.

    ``None`` for absent AND for an explicitly declared ``raw``: both mean the
    legacy rendering, so both must hash exactly as every pre-existing recipe
    does. Only a chat-template rendering contributes, and it contributes ALL
    its resolved parameters explicitly (the identity may not depend on a
    default that a later version could change). Swift twin:
    ``ExtractionRendering.canonicalIdentityFragment``.
    """
    if rendering is None or rendering.is_raw:
        return None
    return {
        "addGenerationPrompt": rendering.add_generation_prompt,
        "mode": CHAT_TEMPLATE,
        "qwenThinkingEnabled": rendering.qwen_thinking_enabled,
        "systemPrompt": rendering.system_prompt,
    }


def rendered_token_ids(model, text: str,
                       rendering: ExtractionRendering) -> list[int]:
    """Token ids for one stimulus under a CHAT-TEMPLATE rendering.

    Calls the MEASUREMENT renderer
    (:func:`steerlab_server.experiment.prompt_render.render`) — the same
    function the run loop renders generation prompts with — so there is one
    rendering definition, not a second copy that could drift from it. The raw
    branch stays in ``extractor._encode``, byte-identical to what it always
    was.
    """
    if rendering.is_raw:  # pragma: no cover - callers branch before this
        raise ExtractionRenderingError(
            "rendered_token_ids is the chat-template path; a raw rendering "
            "tokenizes through extractor._encode")
    # Imported lazily: `steering` is the concept-agnostic core and must not
    # take a module-import dependency on the `experiment` layer.
    from ..experiment import prompt_render
    rendered = prompt_render.render(
        model.tokenizer, text,
        model_id=getattr(model, "model_id", "") or "",
        prompt_mode=prompt_render.CHAT_ASSISTANT,
        system_prompt=rendering.system_prompt,
        qwen_thinking_enabled=rendering.qwen_thinking_enabled,
        add_generation_prompt=rendering.add_generation_prompt)
    return list(rendered.input_ids)
