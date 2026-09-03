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
same discipline ``residualNormConvention`` follows. Only an
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
     "reasoningEffort": "off", "systemPrompt": null}

``mode`` is the only required key. ``addGenerationPrompt`` defaults to true,
``reasoningEffort`` to ``off``, ``systemPrompt`` to absent — the parameters
the MEASUREMENT renderer (``prompt_render.render``) actually varies, and no
more. Family handling (Gemma has no system role; Qwen's ``enable_thinking``)
is derived from the pinned model id by that same renderer, so it is not
re-declared here: there is ONE rendering definition, and extraction calls it.

THE REASONING EFFORT (2026-09-03)
---------------------------------

``reasoningEffort`` ∈ off | low | medium | xhigh replaced the Qwen-specific
boolean ``qwenThinkingEnabled``. The old key is still READ — ``false`` is
``off`` and ``true`` is ``xhigh``, because a recipe declared with thinking on
rendered under the chat template's DEFAULT effort, which on Qwen3.8 is xhigh
— so every existing declaration, sidecar stamp, and recipe identity keeps
its meaning and its bytes; a block that spells both keys is refused as two
declarations of one parameter. A new declaration writes ``reasoningEffort``
only. The recipe-identity fragment (:func:`canonical_identity_fragment`)
keeps the boolean spelling for off/xhigh — the hashes of every frozen recipe
depend on it — and adds ``reasoningEffort`` only for the two values the
boolean cannot express. A non-off effort on a family without a thinking mode
is refused where the model id is known (attach and the extraction routes),
never silently rendered without it.

THE VOICE (2026-08-25, maintainer ruling)
-----------------------------------------

``voice`` names WHOSE TURN the stimulus is rendered as:

.. code-block:: json

    {"mode": "chatTemplate", "voice": "assistant"}

- ``"user"`` — the stimulus is what the model READS. This is the legacy and
  default behavior, and **absent ≡ "user" ≡ today's bytes**: an explicit
  ``"voice": "user"`` canonicalizes away exactly as an explicit
  ``{"mode": "raw"}`` does, so the manifest, the sidecar, the freeze hash and
  the recipe identity of every existing recipe are untouched.
- ``"assistant"`` — the stimulus is rendered as the model's OWN OUTPUT: the
  template's assistant-turn markers wrap it, with NO preceding user content
  (see :func:`prompt_render.render_assistant_turn` for the exact construction
  and why the user turn is subtracted rather than injected).

Two parameters are MEANINGLESS under the assistant voice and are typed
refusals at declaration time rather than silent ignores — ``addGenerationPrompt``
(the stimulus IS the generation) and ``systemPrompt`` (the construction carries
no preceding turn to put it in). Both refusal texts are shared with the Swift
twin verbatim.

**Engine asymmetry:** the swift-mlx engine refuses the assistant voice
outright — MLXLMCommon's tokenizer bridge exposes only the generation-prompt
form of ``applyChatTemplate``, so a completed assistant turn cannot be rendered
there without family-specific arithmetic that holds for Gemma and breaks for
Qwen3's thinking scaffold. The refusal names this engine, exactly as the
``addGenerationPrompt: false`` refusal does.
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

#: The legacy voice, and what an absent ``voice`` means: the stimulus is what
#: the model READS.
VOICE_USER = "user"
#: The stimulus is rendered as the model's OWN OUTPUT — an assistant turn with
#: no preceding user content.
VOICE_ASSISTANT = "assistant"

VOICES = (VOICE_USER, VOICE_ASSISTANT)

#: This engine's name in refusals about an unsupportable rendering form.
ENGINE = "python-hf-transformers"

#: Refusal text shared VERBATIM with the Swift twin
#: (``ExtractionRendering.assistantVoiceGenerationPromptReason``). Under the
#: assistant voice the stimulus IS the generation, so a generation prompt is
#: not a choice that exists — and a declaration that looks like a recipe axis
#: but reaches nothing is the exact failure this module was built to end.
ASSISTANT_VOICE_GENERATION_PROMPT_REASON = (
    "extractionRendering declares voice 'assistant' together with "
    "addGenerationPrompt: under the assistant voice the stimulus IS the "
    "generation, so there is no generation prompt to add or withhold and the "
    "key would reach nothing — repair: drop addGenerationPrompt, or declare "
    "voice 'user' if you meant the model READING the stimulus")

#: Refusal text shared VERBATIM with the Swift twin
#: (``ExtractionRendering.assistantVoiceSystemPromptReason``).
ASSISTANT_VOICE_SYSTEM_PROMPT_REASON = (
    "extractionRendering declares voice 'assistant' together with a "
    "systemPrompt: the assistant-voice construction renders the assistant "
    "turn ALONE, with no preceding turn for system text to live in (injected "
    "context would confound the very contrast the voice exists to isolate) — "
    "repair: drop systemPrompt, or declare voice 'user' if the model is meant "
    "to read the stimulus under that system prompt")

#: The flag both engines' CLIs write a declaration with, and the name every
#: refusal that asks for one must use. Swift twin:
#: ``ExtractionRendering.declarationFlag``.
DECLARATION_FLAG = "--extraction-rendering"

#: EVERY key a ``chatTemplate`` rendering may carry, sorted — the vocabulary a
#: refusal about a stranger names, and the list :func:`from_json` measures an
#: object against. Swift twin: ``ExtractionRendering.chatTemplateKeys``.
CHAT_TEMPLATE_KEYS = ("addGenerationPrompt", "mode", "qwenThinkingEnabled",
                      "reasoningEffort", "systemPrompt", "voice")

#: The reasoning-effort vocabulary, in the fixed cross-engine order — the
#: same closed set the study protocol's ``reasoningEffort`` takes
#: (``experiment.prompt_render.REASONING_EFFORTS``; duplicated here rather
#: than imported because ``steering`` may not depend on ``experiment`` at
#: import time). Swift twin: ``ReasoningEffort.allCases``.
REASONING_OFF = "off"
REASONING_EFFORTS = (REASONING_OFF, "low", "medium", "xhigh")
#: What a legacy ``qwenThinkingEnabled: true`` declaration meant: the chat
#: template's default effort.
LEGACY_THINKING_EFFORT = "xhigh"

#: Refusal text shared VERBATIM with the Swift twin
#: (``ExtractionRendering.bothThinkingKeysReason``).
BOTH_THINKING_KEYS_REASON = (
    "extractionRendering declares both qwenThinkingEnabled and "
    "reasoningEffort — two spellings of one parameter; qwenThinkingEnabled "
    "is the legacy boolean (false ≡ off, true ≡ xhigh) and reasoningEffort "
    "replaces it — repair: keep reasoningEffort and drop qwenThinkingEnabled")


def unknown_effort_reason(value) -> str:
    """Refusal text shared VERBATIM with the Swift twin
    (``ExtractionRendering.unknownEffortReason``)."""
    return (f"extractionRendering.reasoningEffort '{value}' is not in the "
            f"closed vocabulary — repair: declare one of "
            f"{', '.join(REASONING_EFFORTS)} (absent means off)")


def effort_without_thinking_mode_reason(effort: str, model_id: str) -> str:
    """Refusal for a non-off effort on a family whose chat template has no
    thinking mode, raised where the model id is known. Shared VERBATIM with
    the Swift twin (``ExtractionRendering.effortWithoutThinkingModeReason``)."""
    return (f"extractionRendering declares reasoningEffort '{effort}' for "
            f"{model_id}, whose family has no thinking mode — the chat "
            "template would ignore it and the recipe would look as if it "
            "rendered a reasoning scaffold when it did not — repair: declare "
            "reasoningEffort off, or pin a model with a thinking mode")


def has_thinking_mode(model_id: str) -> bool:
    """Whether the pinned family's chat template has a thinking mode. The
    one family rule, restated here so the attach path can ask it without a
    ``steering`` → ``experiment`` import; ``prompt_render.has_thinking_mode``
    is the renderer's own copy and the two are pinned equal by test."""
    return "qwen" in (model_id or "").lower()


def unknown_chat_template_keys(value: dict) -> list[str]:
    """The non-null keys of ``value`` this engine does not read, sorted.

    NON-null, exactly as the raw branch's own extra-parameter check: an
    explicit ``null`` says "this parameter is absent", which is what an absent
    key already says, so it declares nothing and refusing it would break
    manifests that spell absence out.
    """
    return sorted(k for k in value
                  if k not in CHAT_TEMPLATE_KEYS and value.get(k) is not None)


def unknown_chat_template_key_reason(keys) -> str:
    """Refusal text shared VERBATIM with the Swift twin
    (``ExtractionRendering.unknownChatTemplateKeyReason``).

    The raw branch has always refused strangers; the chat-template branch used
    to read its five known keys and IGNORE everything else, so
    ``"addGenerationPromt": false`` — one transposed letter — silently kept the
    default ``true`` and a frozen study measured something other than what its
    manifest appears to declare. A key that reaches nothing is the same failure
    this module exists to end, so it is answered out loud (review 2026-08-26).
    """
    return (
        f"extractionRendering mode 'chatTemplate' does not accept "
        f"{', '.join(keys)}: a key this engine does not read reaches nothing, "
        f"so a misspelling (addGenerationPromt) silently leaves the default in "
        f"place and changes what a frozen study measured — repair: correct or "
        f"drop the key; the accepted parameters are "
        f"{', '.join(CHAT_TEMPLATE_KEYS)}")


def raw_parameters(value: dict) -> list[str]:
    """The non-null keys a ``raw`` block carries beyond ``mode``, sorted.

    NON-null, for the same reason :func:`unknown_chat_template_keys` is: an
    explicit ``null`` says "this parameter is absent", which is what an absent
    key already says. Swift twin: ``ExtractionRendering.rawParameters(in:)``.
    """
    return sorted(k for k in value
                  if k != "mode" and value.get(k) is not None)


def raw_parameters_reason(keys) -> str:
    """Refusal text shared VERBATIM with the Swift twin
    (``ExtractionRendering.rawParametersError``, as its
    ``DeclarationError.message``).

    A raw rendering takes no parameters: accepting them silently would let a
    manifest look like it declared something it cannot get — an
    ``addGenerationPrompt`` under ``raw`` reaches no template at all.
    """
    return (
        f"extractionRendering mode 'raw' takes no parameters but declares "
        f"{', '.join(keys)} — repair: drop the parameters, or declare mode "
        f"'chatTemplate' if you meant the template rendering")


class ExtractionRenderingError(ValueError):
    """A rendering declaration this engine cannot honor. Typed and carrying a
    repair, per house style — never a silent fallback to raw, because a silent
    fallback is precisely the ambiguity this module removes."""


@dataclass(frozen=True)
class ExtractionRendering:
    """A declared extraction rendering. Construct via :func:`from_json`."""

    mode: str = RAW
    add_generation_prompt: bool = True
    #: The declared reasoning effort (``reasoningEffort``); ``off`` is the
    #: legacy ``qwenThinkingEnabled: false`` and what an absent key means.
    reasoning_effort: str = REASONING_OFF
    system_prompt: str | None = None
    #: WHOSE TURN the stimulus is rendered as. ``VOICE_USER`` is the legacy
    #: value and what an absent key means; see the module docstring.
    voice: str = VOICE_USER

    @property
    def is_raw(self) -> bool:
        return self.mode == RAW

    @property
    def is_assistant_voice(self) -> bool:
        return self.voice == VOICE_ASSISTANT

    @property
    def qwen_thinking_enabled(self) -> bool:
        """Whether the render carries a thinking scaffold at all — the
        boolean every pre-effort reader asked for. Derived from the effort."""
        return self.reasoning_effort != REASONING_OFF

    @property
    def label(self) -> str:
        """Short human label for logs and refusal messages."""
        if self.is_raw:
            return "raw"
        if self.is_assistant_voice:
            # addGenerationPrompt is not a knob under this voice, so naming it
            # in a human label would invite the reader to look for it.
            bits = ["voice=assistant"]
            if self.qwen_thinking_enabled:
                bits.append(f"reasoningEffort={self.reasoning_effort}")
            return "chatTemplate (" + ", ".join(bits) + ")"
        bits = [f"addGenerationPrompt={str(self.add_generation_prompt).lower()}"]
        if self.qwen_thinking_enabled:
            bits.append(f"reasoningEffort={self.reasoning_effort}")
        if self.system_prompt:
            bits.append("systemPrompt=set")
        return "chatTemplate (" + ", ".join(bits) + ")"

    def to_dict(self) -> dict:
        """The stamped/declared JSON block. Defaults are written EXPLICITLY on
        a chat-template rendering (the artifact must say what it did without a
        reader knowing this module's defaults); ``systemPrompt`` is omitted
        when absent, matching the sidecar's drop-None convention.

        The USER voice writes no ``voice`` key at all — absent ≡ "user", and a
        recipe that never heard of the voice must keep its bytes. The ASSISTANT
        voice writes the key AND omits ``addGenerationPrompt``: that parameter
        is refused at declaration time as meaningless there, so stamping the
        internal ``false`` would be an artifact claiming a choice nobody made.

        Writes ``reasoningEffort`` (never the legacy boolean): this is what a
        NEW stamp or declaration says. A block READ under the old spelling is
        never re-serialized into a frozen file — attach writes into a draft,
        extraction stamps a new sidecar — so no frozen bytes move.
        """
        if self.is_raw:
            return {"mode": RAW}
        if self.is_assistant_voice:
            return {
                "mode": CHAT_TEMPLATE,
                "reasoningEffort": self.reasoning_effort,
                "voice": VOICE_ASSISTANT,
            }
        block = {
            "mode": CHAT_TEMPLATE,
            "addGenerationPrompt": self.add_generation_prompt,
            "reasoningEffort": self.reasoning_effort,
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

    **Reading is as strict as declaring**, and deliberately so: this function
    serves the DECLARATION path (through :func:`parse_declaration`) and the
    manifest/sidecar READ path alike, and an unknown key in a recorded block
    can only mean a newer engine wrote a field this one does not understand.
    Refusing loudly beats misreading the recipe — the callers that read
    artifacts turn the refusal into their own named violation
    (``manifest._verify_vector_artifact_pins``,
    ``experiment_store.attach_artifact``, ``recipe_identity``'s unprovable
    field).
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
        extra = raw_parameters(value)
        if extra:
            raise ExtractionRenderingError(raw_parameters_reason(extra))
        return RAW_RENDERING
    # …and neither does a chat-template rendering, beyond the five parameters
    # it actually varies. Checked FIRST, before any of them is read, so a
    # stranger is answered before a declaration that also carries one of the
    # meaningless-under-this-voice combinations below — one deterministic
    # order, twinned in Swift's ``declared(object:)``.
    unknown = unknown_chat_template_keys(value)
    if unknown:
        raise ExtractionRenderingError(unknown_chat_template_key_reason(unknown))
    add_generation_prompt = value.get("addGenerationPrompt")
    qwen_thinking = value.get("qwenThinkingEnabled")
    declared_effort = value.get("reasoningEffort")
    system_prompt = value.get("systemPrompt")
    voice = value.get("voice")
    if qwen_thinking is not None and declared_effort is not None:
        raise ExtractionRenderingError(BOTH_THINKING_KEYS_REASON)
    if voice is not None and (not isinstance(voice, str) or voice not in VOICES):
        raise ExtractionRenderingError(
            f"extraction rendering voice '{voice}' is not supported by the "
            f"{ENGINE} engine — repair: declare one of {', '.join(VOICES)} "
            f"(absent means '{VOICE_USER}', the legacy voice: the stimulus is "
            f"what the model reads)")
    if voice == VOICE_ASSISTANT:
        # Both refusals are shared VERBATIM with the Swift twin, and both fire
        # here rather than at extraction: a parameter that reaches nothing must
        # be answered while the person is still typing, never hours later on a
        # GPU (and never by silently ignoring it).
        if add_generation_prompt is not None:
            raise ExtractionRenderingError(
                ASSISTANT_VOICE_GENERATION_PROMPT_REASON)
        if system_prompt is not None:
            raise ExtractionRenderingError(ASSISTANT_VOICE_SYSTEM_PROMPT_REASON)
    if add_generation_prompt is not None and not isinstance(add_generation_prompt, bool):
        raise ExtractionRenderingError(
            "extractionRendering.addGenerationPrompt must be a boolean — "
            "repair: use true or false")
    if qwen_thinking is not None and not isinstance(qwen_thinking, bool):
        raise ExtractionRenderingError(
            "extractionRendering.qwenThinkingEnabled must be a boolean — "
            "repair: use true or false")
    if declared_effort is not None and (
            not isinstance(declared_effort, str)
            or declared_effort not in REASONING_EFFORTS):
        raise ExtractionRenderingError(unknown_effort_reason(declared_effort))
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise ExtractionRenderingError(
            "extractionRendering.systemPrompt must be a string or absent — "
            "repair: drop the key when the render carries no system prompt")
    if declared_effort is not None:
        effort = declared_effort
    else:
        # The legacy boolean's meaning: true ran under the template's default
        # effort. Absent is off, exactly as it always was.
        effort = LEGACY_THINKING_EFFORT if qwen_thinking else REASONING_OFF
    return ExtractionRendering(
        mode=CHAT_TEMPLATE,
        # The assistant voice renders a COMPLETED assistant turn, which is the
        # `add_generation_prompt=False` form of the template call; the field
        # records what the construction actually does, and `to_dict` omits it
        # because the declaration may not name it.
        add_generation_prompt=(
            False if voice == VOICE_ASSISTANT
            else (True if add_generation_prompt is None
                  else add_generation_prompt)),
        reasoning_effort=effort,
        system_prompt=system_prompt,
        voice=voice or VOICE_USER)


def thinking_mode_problem(rendering: "ExtractionRendering | None",
                          model_id: str) -> str | None:
    """The refusal for a non-off effort on a family without a thinking mode,
    or None — asked wherever a declaration meets the pinned model id (attach,
    the extraction routes), because :func:`from_json` itself never sees one.
    Swift twin: ``ExtractionRendering.thinkingModeProblem(modelID:)``."""
    if rendering is None or rendering.is_raw:
        return None
    if rendering.reasoning_effort == REASONING_OFF:
        return None
    if has_thinking_mode(model_id):
        return None
    return effort_without_thinking_mode_reason(
        rendering.reasoning_effort, model_id)


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

    ``voice`` is the SECOND optional key, and for the same reason the first
    one exists: every chat-template recipe written before the voice existed
    rendered the user voice, so the key appears ONLY for the assistant voice
    and an explicit ``"voice": "user"`` canonicalizes away. In sorted-key order
    it lands last (after ``systemPrompt``), which is where the Swift twin's
    hand-built JSON appends it.
    """
    if rendering is None or rendering.is_raw:
        return None
    fragment = {
        "addGenerationPrompt": rendering.add_generation_prompt,
        "mode": CHAT_TEMPLATE,
        # THE BOOLEAN SPELLING, DELIBERATELY. Every recipe hashed before the
        # effort existed hashed `qwenThinkingEnabled`, and a recipe declared
        # `reasoningEffort: "xhigh"` today renders the identical scaffold a
        # `qwenThinkingEnabled: true` recipe rendered — so both must hash the
        # same. The effort key joins the fragment ONLY for the two values the
        # boolean cannot express (sorted position: after qwenThinkingEnabled,
        # before systemPrompt — where the Swift twin's hand-built JSON puts
        # it).
        "qwenThinkingEnabled": rendering.qwen_thinking_enabled,
    }
    if rendering.reasoning_effort not in (REASONING_OFF, LEGACY_THINKING_EFFORT):
        fragment["reasoningEffort"] = rendering.reasoning_effort
    fragment["systemPrompt"] = rendering.system_prompt
    if rendering.is_assistant_voice:
        fragment["voice"] = VOICE_ASSISTANT
    return fragment


def rendered_token_ids(model, text: str,
                       rendering: ExtractionRendering) -> list[int]:
    """Token ids for one stimulus under a CHAT-TEMPLATE rendering.

    Calls the MEASUREMENT renderer
    (:func:`steerlab_server.experiment.prompt_render.render`) — the same
    function the run loop renders generation prompts with — so there is one
    rendering definition, not a second copy that could drift from it. The raw
    branch stays in ``extractor._encode``, byte-identical to what it always
    was.

    Under the ASSISTANT voice the call goes to
    :func:`prompt_render.render_assistant_turn` instead — still that module,
    still the family's own chat template, and still one definition.
    """
    if rendering.is_raw:  # pragma: no cover - callers branch before this
        raise ExtractionRenderingError(
            "rendered_token_ids is the chat-template path; a raw rendering "
            "tokenizes through extractor._encode")
    # Imported lazily: `steering` is the concept-agnostic core and must not
    # take a module-import dependency on the `experiment` layer.
    from ..experiment import prompt_render
    if rendering.is_assistant_voice:
        try:
            rendered = prompt_render.render_assistant_turn(
                model.tokenizer, text,
                model_id=getattr(model, "model_id", "") or "",
                reasoning_effort=rendering.reasoning_effort)
        except prompt_render.AssistantVoiceUnsupported as exc:
            # Typed and re-homed, so every caller of this module sees one
            # error type — never a silent fallback to the user voice, which
            # would answer a different question than the one declared.
            raise ExtractionRenderingError(str(exc)) from exc
        return list(rendered.input_ids)
    rendered = prompt_render.render(
        model.tokenizer, text,
        model_id=getattr(model, "model_id", "") or "",
        prompt_mode=prompt_render.CHAT_ASSISTANT,
        system_prompt=rendering.system_prompt,
        reasoning_effort=rendering.reasoning_effort,
        add_generation_prompt=rendering.add_generation_prompt)
    return list(rendered.input_ids)
