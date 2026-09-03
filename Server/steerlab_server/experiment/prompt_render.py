"""Render a prompt to token ids with family-specific handling (parallel to
Swift ``ExperimentTasks.userInput`` / ``qwenContext``).

**Chat-template parity with the Swift side is the #1 correctness risk** (see the
plan). The rules reproduced here, family-detected from the model id exactly as
Swift does:

- **Gemma 3** has *no system role*: system text is prepended to the first user
  turn (``system + "\\n\\n" + user``), and we rely on the HF chat template + a
  single ``add_special_tokens=False`` tokenize to avoid a double BOS.
- **Qwen3** disables thinking mode for studies: ``enable_thinking=False`` is
  passed to the chat template (chat mode), or `` /no_think`` is appended (raw
  completion), unless a non-off ``reasoningEffort`` is declared — then
  ``enable_thinking=True`` AND ``reasoning_effort=<value>`` go to the template
  (Qwen3.8's template reads the effort; Qwen3's ignores the extra variable,
  which is what makes the declared study reproducible on either).

Two prompt modes mirror ``ExperimentManifest.PromptMode``:
``chatAssistant`` (apply chat template, add generation prompt) and
``rawCompletion`` (no template; the bare/scaffolded text).

THE REASONING EFFORT (2026-09-03). The Qwen-specific boolean
``qwenThinkingEnabled`` was replaced by ``reasoningEffort`` ∈ {off, low,
medium, xhigh}, on the study protocol and on a concept's
``extractionRendering`` alike. Every renderer here takes the effort; the old
boolean survives as a keyword alias (``qwen_thinking_enabled``) that means
``off``/``xhigh`` — because a manifest frozen under the old spelling with
thinking ON ran under the template's DEFAULT effort, which on Qwen3.8 is
``xhigh``. Callers that pass the effort pass it explicitly; callers that
never heard of it keep rendering exactly what they always did.
"""

from __future__ import annotations

from dataclasses import dataclass

CHAT_ASSISTANT = "chatAssistant"
RAW_COMPLETION = "rawCompletion"

#: The declared reasoning effort that means NO thinking block: today's
#: ``enable_thinking=False`` path, byte for byte.
REASONING_OFF = "off"
#: The closed vocabulary of ``reasoningEffort``, in the fixed cross-engine
#: order. The non-off values are the ones the Qwen3.8 chat template accepts.
#: Swift twin: ``ReasoningEffort`` (SteeringKit).
REASONING_EFFORTS: tuple[str, ...] = (REASONING_OFF, "low", "medium", "xhigh")
#: What a legacy ``qwenThinkingEnabled: true`` means: the template's own
#: default effort, which is what every such study actually ran under.
LEGACY_THINKING_EFFORT = "xhigh"


@dataclass
class RenderedPrompt:
    text: str           # the rendered string (for logging/inspection)
    input_ids: list[int]  # token ids fed to the model
    prompt_token_count: int


def _is_gemma(model_id: str) -> bool:
    return "gemma" in model_id.lower()


def _is_qwen(model_id: str) -> bool:
    return "qwen" in model_id.lower()


def has_thinking_mode(model_id: str) -> bool:
    """Whether this family's chat template HAS a thinking mode to declare an
    effort for. Qwen3/Qwen3.8 do; Gemma 3 does not — a non-off effort on it
    would reach nothing, so the declaration gates refuse it by name rather
    than letting a study look as if it reasoned. Swift twin:
    ``PromptRendering.hasThinkingMode``."""
    return _is_qwen(model_id)


def resolve_reasoning_effort(reasoning_effort: str | None,
                             qwen_thinking_enabled: bool = False) -> str:
    """The effort a renderer applies: the declared value when one was passed,
    else the legacy boolean's meaning (``off`` / ``xhigh``)."""
    if reasoning_effort is None:
        return LEGACY_THINKING_EFFORT if qwen_thinking_enabled else REASONING_OFF
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(
            f"unknown reasoningEffort {reasoning_effort!r} — known: "
            + ", ".join(REASONING_EFFORTS))
    return reasoning_effort


def thinking_template_kwargs(model_id: str, effort: str) -> dict:
    """The chat-template variables the effort becomes, for this family.

    Nothing for a family without a thinking mode. For Qwen, ``off`` is
    exactly the kwargs every study has always rendered with
    (``enable_thinking=False``), so an off study's prompt bytes cannot move;
    a non-off effort adds ``reasoning_effort`` beside ``enable_thinking=True``.
    Swift twin: ``PromptRendering.thinkingContext``.
    """
    if not has_thinking_mode(model_id):
        return {}
    if effort == REASONING_OFF:
        return {"enable_thinking": False}
    return {"enable_thinking": True, "reasoning_effort": effort}


# --- the declared reasoning protocol (manifest keys and their rules) ---------

#: Manifest / protocol key of the effort. Swift twin: `CodingKeys.reasoningEffort`.
REASONING_EFFORT_KEY = "reasoningEffort"
#: Manifest / protocol key of the reasoning block's own token cap.
REASONING_MAX_TOKENS_KEY = "reasoningMaxTokens"
#: The key this vocabulary replaced. READ (as off/xhigh) wherever no
#: ``reasoningEffort`` is present, so every manifest frozen under it keeps
#: loading and verifying unchanged; never written by a new declaration.
LEGACY_THINKING_KEY = "qwenThinkingEnabled"


def read_reasoning_effort(d: dict) -> str:
    """The effort a manifest dict declares.

    A NON-OFF ``reasoningEffort`` wins outright. Otherwise a legacy
    ``qwenThinkingEnabled: true`` still means the template's default effort
    (xhigh) — even beside an explicit ``off``, which every manifest created
    since the effort existed carries by default: the boolean is the OLDER,
    more deliberate declaration on such a document (a hand edit, or a
    pre-effort toggle written onto a fresh manifest), and reading it as off
    would silently drop a thinking mode the study asked for. The writers never
    leave both keys on a draft (``set_protocol`` drops the boolean). Then
    ``reasoningEffort`` as spelled, else off.

    A malformed value comes back VERBATIM (as a string) rather than being
    corrected: :func:`verify` names it, and silently reading ``"hgih"`` as off
    would be the exact misspelling-changes-the-measurement failure the closed
    vocabulary exists to prevent. Swift twin:
    ``ExperimentManifest.resolvedReasoningEffort``.
    """
    effort = d.get(REASONING_EFFORT_KEY)
    if effort is not None and str(effort) != REASONING_OFF:
        return str(effort)
    if bool(d.get(LEGACY_THINKING_KEY, False)):
        return LEGACY_THINKING_EFFORT
    return REASONING_OFF if effort is None else str(effort)


def read_reasoning_max_tokens(d: dict) -> int | None:
    """The declared reasoning budget, or None. A non-integer (or a bool, or
    a non-positive number) reads as None here and is named by
    :func:`reasoning_protocol_violations` — the same split
    ``truncation_gate.declared_threshold`` makes."""
    value = d.get(REASONING_MAX_TOKENS_KEY)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


#: Refusal sentences shared VERBATIM with the Swift twin
#: (``ExperimentStore.setSamplingProtocol`` / ``modelOutputPinViolations``).
BUDGET_WITHOUT_EFFORT_REASON = (
    f"{REASONING_MAX_TOKENS_KEY} is declared but {REASONING_EFFORT_KEY} is "
    "off — the model generates no reasoning block to cap; declare a non-off "
    f"{REASONING_EFFORT_KEY} or drop the budget")


def unknown_effort_reason(value) -> str:
    return (f"unknown {REASONING_EFFORT_KEY} {value!r} — known: "
            + ", ".join(REASONING_EFFORTS))


def effort_without_budget_reason(effort: str) -> str:
    return (f"{REASONING_EFFORT_KEY} '{effort}' needs a "
            f"{REASONING_MAX_TOKENS_KEY} — the reasoning block's own token "
            "cap, declared and never defaulted; maxTokens stays the answer "
            "budget, counted from the token after </think>")


def effort_without_thinking_mode_reason(effort: str, model_id: str) -> str:
    return (f"{REASONING_EFFORT_KEY} '{effort}' declared for {model_id}, "
            "whose family has no thinking mode — the chat template would "
            "ignore it and the study would look as if it reasoned when it "
            f"did not; declare {REASONING_EFFORT_KEY} off")


def malformed_budget_reason(value) -> str:
    return (f"{REASONING_MAX_TOKENS_KEY} must be a positive integer — got "
            f"{value!r}")


def reasoning_protocol_violations(*, effort, reasoning_max_tokens,
                                  model_id: str) -> list[str]:
    """Every rule a declared reasoning protocol must satisfy, as the
    sentences both engines refuse with. ``effort`` and
    ``reasoning_max_tokens`` are the RAW declared values (None = absent).

    - the effort is in the closed vocabulary;
    - a non-off effort names a family with a thinking mode;
    - a non-off effort carries a positive-integer reasoning budget;
    - an off effort carries none.
    """
    problems: list[str] = []
    if effort is None:
        effort = REASONING_OFF
    if not isinstance(effort, str) or effort not in REASONING_EFFORTS:
        problems.append(unknown_effort_reason(effort))
        return problems
    if reasoning_max_tokens is not None and (
            isinstance(reasoning_max_tokens, bool)
            or not isinstance(reasoning_max_tokens, int)
            or reasoning_max_tokens < 1):
        problems.append(malformed_budget_reason(reasoning_max_tokens))
        return problems
    if effort == REASONING_OFF:
        if reasoning_max_tokens is not None:
            problems.append(BUDGET_WITHOUT_EFFORT_REASON)
        return problems
    if not has_thinking_mode(model_id):
        problems.append(effort_without_thinking_mode_reason(effort, model_id))
    if reasoning_max_tokens is None:
        problems.append(effort_without_budget_reason(effort))
    return problems


def has_system_role(model_id: str) -> bool:
    """Whether this family's chat template has a SYSTEM ROLE.

    The capability :func:`render` branches on, named so authoring surfaces can
    ask it without re-deriving the family rule: True means a declared system
    prompt is inserted as a genuine system turn, False means the SAME text is
    prepended to the first user turn (``system + "\\n\\n" + user``) because the
    template has nowhere else to put it. Either way it reaches the model —
    which is the property ``set-system-prompt`` reports. Gemma 3 is today's
    only False. Swift twin: ``PromptRendering.hasSystemRole``.
    """
    return not _is_gemma(model_id)


# --- Conversation-structure constraints (per vendored family) -----------------

@dataclass(frozen=True)
class ConversationConstraints:
    """What a family's chat template refuses at render time.

    Determined EMPIRICALLY against the pinned tokenizers (2026-07-13, all
    eight cached models — see ``test_seeded_turns.py``'s live-template check,
    which fails loudly if a re-vendored template drifts from this table):

    - **Gemma 3** (``google/gemma-3-4b-it`` + all mlx-community 4b/12b/27b):
      the template's role loop raises ``jinja2 TemplateError('Conversation
      roles must alternate user/assistant/user/assistant/...')`` for an
      assistant-first history AND for any consecutive same-role pair.
    - **Qwen3** (0.6B/4B/14B/32B): fully permissive — assistant-first and
      consecutive same-role histories all render.

    Swift twin: ``ExperimentTasks.conversationConstraints``.
    """
    requires_leading_user_turn: bool
    forbids_consecutive_same_role: bool

    def to_dict(self) -> dict:
        return {
            "requiresLeadingUserTurn": self.requires_leading_user_turn,
            "forbidsConsecutiveSameRole": self.forbids_consecutive_same_role,
        }


def conversation_constraints(model_id: str) -> ConversationConstraints:
    """Family constraints table (static per vendored family — both are pinned)."""
    if _is_gemma(model_id):
        return ConversationConstraints(
            requires_leading_user_turn=True,
            forbids_consecutive_same_role=True)
    return ConversationConstraints(
        requires_leading_user_turn=False,
        forbids_consecutive_same_role=False)


def family_label(model_id: str) -> str:
    """Short family label for constraint messages ("gemma-3", "qwen3", or the
    model id itself when the family is unrecognized). Swift twin:
    ``ExperimentTasks.familyLabel``."""
    lowered = model_id.lower()
    if "gemma" in lowered:
        return "gemma-3"
    if "qwen" in lowered:
        return "qwen3"
    return model_id


# --- Scripted study transcripts (the metacognition-study instrument) ----------
#
# A task-prompt item MAY carry a `transcript` — a scripted multi-turn
# conversation (researcher-authored assistant turns included: "words in the
# model's mouth") pinned as hashed stimulus data via the ordinary
# taskPromptsHash. The helpers below are the cross-engine data contract:
# validation rules AND their exact message strings must match the Swift twins
# in `ExperimentTasks` (pinned by the committed fixture
# `prompts/fixtures/transcript-validation/cases.json`).

TRANSCRIPT_ROLES = ("system", "user", "assistant")

TRANSCRIPT_RAW_COMPLETION_MESSAGE = (
    "task prompts include scripted transcripts but promptMode is "
    "rawCompletion — transcript items render through the chat template by "
    "definition; use chatAssistant")


def normalize_transcript(turns: list) -> list[dict]:
    """Normalized `{role, content}` turns (extra keys — e.g. a Playground
    export's `seeded` flag — are dropped): rendering must ignore them, and the
    RECORD copy of the transcript is identical across engines."""
    normalized = []
    for turn in turns or []:
        if not isinstance(turn, dict):
            normalized.append({"role": "", "content": ""})
            continue
        role = turn.get("role")
        content = turn.get("content")
        normalized.append({
            "role": role if isinstance(role, str) else "",
            "content": content if isinstance(content, str) else "",
        })
    return normalized


def transcript_schema_violation(turns: list, item_id: str) -> str | None:
    """First schema violation of one item's scripted transcript, or None.

    The rules (validated identically at load on BOTH engines): non-empty;
    roles from {system, user, assistant}; non-empty content per turn; at most
    one system turn, and only first; the FINAL turn must be `user` —
    generation produces the assistant's reply (a trailing assistant turn
    would be assistant-prefix continuation, out of scope v1).
    """
    turns = normalize_transcript(turns)
    if not turns:
        return (f"item '{item_id}': transcript is empty — a scripted "
                "transcript needs at least a final user turn")
    for index, turn in enumerate(turns):
        if turn["role"] not in TRANSCRIPT_ROLES:
            return (f"item '{item_id}': transcript turn {index + 1} has role "
                    f"'{turn['role']}' — allowed roles are system, user, "
                    "assistant")
        if not turn["content"].strip():
            return (f"item '{item_id}': transcript turn {index + 1} has "
                    "empty content")
    if any(t["role"] == "system" for t in turns[1:]):
        return (f"item '{item_id}': transcript may carry at most one system "
                "turn, and it must be first")
    final = turns[-1]["role"]
    if final == "assistant":
        return (f"item '{item_id}': transcript ends with an assistant turn — "
                "generation produces the assistant's reply to a final user "
                "turn; assistant-prefix continuation is out of scope for "
                "scripted-transcript studies (v1)")
    if final != "user":
        return (f"item '{item_id}': transcript must end with a user turn "
                "(generation produces the assistant's reply to it)")
    return None


def transcript_family_violation(turns: list, item_id: str,
                                model_id: str) -> str | None:
    """First family chat-template constraint the transcript violates, or None.

    Checked over the NON-SYSTEM turns (a leading system turn composes with —
    replaces — the study system prompt and is handled by the renderer's own
    family convention). Positions in messages are 1-based indices into the
    FULL transcript so the researcher can find the offending turns.
    """
    constraints = conversation_constraints(model_id)
    family = family_label(model_id)
    indexed = [(i, t) for i, t in enumerate(normalize_transcript(turns))
               if t["role"] != "system"]
    if constraints.requires_leading_user_turn and indexed \
            and indexed[0][1]["role"] == "assistant":
        return (f"item '{item_id}': {family}'s chat template requires the "
                "conversation to start with a user turn — this transcript "
                "starts with an assistant turn")
    if constraints.forbids_consecutive_same_role:
        for (i, a), (j, b) in zip(indexed, indexed[1:]):
            if a["role"] == b["role"]:
                return (f"item '{item_id}': {family}'s chat template requires "
                        "strict user/assistant alternation — transcript turns "
                        f"{i + 1} and {j + 1} are consecutive {a['role']} "
                        "turns")
    return None


def transcript_display_text(turns: list) -> str:
    """The record's display text for a transcript item without its own
    `text`/`prompt`: the final user turn (schema-validated to exist)."""
    turns = normalize_transcript(turns)
    return turns[-1]["content"] if turns else ""


def render_transcript(tokenizer, transcript: list, *, model_id: str,
                      prompt_mode: str = CHAT_ASSISTANT,
                      system_prompt: str | None = None,
                      qwen_thinking_enabled: bool = False,
                      reasoning_effort: str | None = None) -> RenderedPrompt:
    """Render one scripted-transcript study item through the model family's
    REAL chat template — the same :func:`render_messages` path the interactive
    Playground verified byte-parity for, so pinned stimulus bytes and the
    interactive instrument can never drift.

    System-prompt composition RULE (cross-engine): a transcript's own system
    turn REPLACES the study-level system prompt for that item — the transcript
    is the more specific declaration. (Swift twin:
    ``ExperimentTasks.transcriptMessages``.)
    """
    if prompt_mode == RAW_COMPLETION:
        raise ValueError("scripted transcripts require chatAssistant prompt mode")
    turns = normalize_transcript(transcript)
    system = system_prompt
    if turns and turns[0]["role"] == "system":
        system = turns[0]["content"]
        turns = turns[1:]
    return render_messages(
        tokenizer, turns, model_id=model_id, prompt_mode=CHAT_ASSISTANT,
        system_prompt=system, qwen_thinking_enabled=qwen_thinking_enabled,
        reasoning_effort=reasoning_effort)


class ChatTemplateConstraintError(ValueError):
    """A chat template refused the conversation at render time (e.g. Gemma's
    'Conversation roles must alternate…'). Subclasses ``ValueError`` so
    existing 400 translation still catches it; routes that catch it FIRST can
    return the structured ``detail()`` instead of a bare string."""

    def __init__(self, message: str, *, model_id: str, constraint: str = "roleOrder"):
        super().__init__(message)
        self.message = message
        self.model_id = model_id
        self.constraint = constraint

    def detail(self) -> dict:
        return {
            "message": self.message,
            "constraint": self.constraint,
            "modelID": self.model_id,
        }


def _constraint_kind(message: str) -> str:
    lowered = message.lower()
    if "role" in lowered or "alternate" in lowered or "user turn" in lowered:
        return "roleOrder"
    return "template"


def _apply_chat_template(tokenizer, chat, *, model_id: str, **kwargs) -> str:
    """``apply_chat_template`` with template-refusal translation: a jinja
    ``TemplateError`` (how templates' ``raise_exception`` surfaces) — or an
    alternation ``ValueError`` raised inside a template — becomes a
    :class:`ChatTemplateConstraintError`, never a raw 500 at the route."""
    try:
        import jinja2
        template_error: type[Exception] = jinja2.exceptions.TemplateError
    except ImportError:  # pragma: no cover - transformers depends on jinja2
        template_error = ()
    try:
        return tokenizer.apply_chat_template(chat, **kwargs)
    except ChatTemplateConstraintError:
        raise
    except template_error as exc:
        raise ChatTemplateConstraintError(
            str(exc), model_id=model_id,
            constraint=_constraint_kind(str(exc))) from exc
    except ValueError as exc:
        # Some templates/tokenizers raise the alternation refusal as a bare
        # ValueError; only translate messages that name a role-order problem —
        # anything else propagates unchanged.
        if _constraint_kind(str(exc)) == "roleOrder":
            raise ChatTemplateConstraintError(
                str(exc), model_id=model_id, constraint="roleOrder") from exc
        raise


def render(tokenizer, prompt: str, *, model_id: str,
           prompt_mode: str = CHAT_ASSISTANT, system_prompt: str | None = None,
           qwen_thinking_enabled: bool = False,
           add_generation_prompt: bool = True,
           reasoning_effort: str | None = None) -> RenderedPrompt:
    """Render one prompt to token ids.

    ``add_generation_prompt`` exists for the EXTRACTION rendering path
    (``steering.extraction_rendering``), which may want the conversation
    rendered without the trailing generation prompt. Generation callers never
    pass it: the default reproduces the measured-generation render exactly,
    so this parameter cannot change what any existing caller does.

    ``reasoning_effort`` is the declared effort (see the module docstring);
    when None the legacy ``qwen_thinking_enabled`` boolean decides.
    """
    effort = resolve_reasoning_effort(reasoning_effort, qwen_thinking_enabled)
    system = (system_prompt or "").strip()
    has_system = bool(system)

    if prompt_mode == RAW_COMPLETION:
        text = prompt
        if has_system:
            text = system + "\n\n" + text
        if _is_qwen(model_id):
            text += " /no_think" if effort == REASONING_OFF else " /think"
        # Raw completion: tokenize as-is (special tokens per tokenizer default,
        # matching the Swift raw-text path which adds them once).
        ids = tokenizer(text, add_special_tokens=True).input_ids
        return RenderedPrompt(text=text, input_ids=list(ids), prompt_token_count=len(ids))

    # chatAssistant
    user = prompt
    messages: list[dict] = []
    if has_system:
        if not has_system_role(model_id):
            user = system + "\n\n" + user  # Gemma has no system role
        else:
            messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    template_kwargs = thinking_template_kwargs(model_id, effort)

    text = _apply_chat_template(
        tokenizer, messages, model_id=model_id,
        tokenize=False, add_generation_prompt=add_generation_prompt,
        **template_kwargs)
    # The template already emits BOS/special markers; tokenize without adding a
    # second set (double-BOS guard, esp. for Gemma).
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return RenderedPrompt(text=text, input_ids=list(ids), prompt_token_count=len(ids))


# --- the ASSISTANT VOICE (extraction only) -----------------------------------

#: The throwaway user turn the assistant-voice construction renders WITH and
#: then subtracts. It exists only because chat templates are written around a
#: conversation (Gemma 3's raises outright on an assistant-first history — see
#: :func:`conversation_constraints`), and it never reaches the model: the
#: rendered prefix that contains it is removed by string subtraction below.
#: A single ASCII letter, so no family's template can transform it.
ASSISTANT_VOICE_PROBE_TURN = "x"


class AssistantVoiceUnsupported(ValueError):
    """This family's chat template cannot express the assistant-voice render.

    Typed and carrying a repair, per house style — never a silent fallback to
    the user voice, which would answer a different question than the one the
    recipe declared.
    """


def render_assistant_turn(tokenizer, text: str, *, model_id: str,
                          qwen_thinking_enabled: bool = False,
                          reasoning_effort: str | None = None) -> RenderedPrompt:
    """Render one stimulus as the model's OWN OUTPUT — the assistant voice.

    THE EXACT RENDERED FORM. On Gemma 3 this produces, and nothing else::

        <bos><start_of_turn>model\\n{stimulus}<end_of_turn>\\n

    On a ChatML family (Qwen 3)::

        <|im_start|>assistant\\n{stimulus}<|im_end|>\\n

    — the template's own assistant-turn markers wrapping the stimulus, with NO
    preceding user content, and the family's BOS added exactly once by the
    tokenizer's defaults (the template's own BOS lives in the prefix this
    construction subtracts, so ``add_special_tokens=True`` restores one and
    only one).

    HOW, and why not otherwise. Chat templates are functions of a
    CONVERSATION, and most of them refuse an assistant-first one outright
    (Gemma 3 raises ``Conversation roles must alternate…``). Hand-writing the
    markers per family would be a SECOND rendering definition, which is the
    exact drift ``extractionRendering`` exists to make impossible. So the
    turn's markers are obtained from the template itself, by subtraction:

    1. render ``[user(probe)]`` with no generation prompt → ``prefix``;
    2. render ``[user(probe), assistant(stimulus)]`` with no generation
       prompt → ``full``;
    3. ``full`` begins with ``prefix`` — everything after it is precisely the
       assistant turn the template writes, markers included.

    Injecting real user content instead of subtracting the probe would
    confound the contrast the voice exists to isolate: the vector would carry
    whatever the user turn said. The probe never reaches the model.

    The Swift twin REFUSES this voice (``PromptRendering.assistantVoiceReason``):
    MLXLMCommon's tokenizer bridge exposes only the generation-prompt form of
    ``applyChatTemplate``, so step 1 is unavailable there and reconstructing it
    by token arithmetic holds for Gemma but breaks on Qwen3's thinking
    scaffold. A study that needs this voice extracts here.
    """
    template_kwargs = thinking_template_kwargs(
        model_id, resolve_reasoning_effort(reasoning_effort, qwen_thinking_enabled))
    probe = [{"role": "user", "content": ASSISTANT_VOICE_PROBE_TURN}]
    prefix = _apply_chat_template(
        tokenizer, probe, model_id=model_id, tokenize=False,
        add_generation_prompt=False, **template_kwargs)
    full = _apply_chat_template(
        tokenizer, probe + [{"role": "assistant", "content": text}],
        model_id=model_id, tokenize=False, add_generation_prompt=False,
        **template_kwargs)
    if not isinstance(prefix, str) or not isinstance(full, str) \
            or not full.startswith(prefix) or len(full) == len(prefix):
        raise AssistantVoiceUnsupported(
            f"the chat template for '{model_id}' does not render an assistant "
            "turn as a suffix of the conversation before it, so the "
            "assistant-voice construction cannot isolate the turn's own "
            "markers — repair: extract this concept with voice 'user', or "
            "with mode 'raw', and say in METHODS which voice the direction "
            "came from (never let a voice fall back silently: the two voices "
            "are different directions)")
    rendered = full[len(prefix):]
    # add_special_tokens=True, DELIBERATELY, and only here: the subtraction
    # above removed the template's own leading BOS along with the probe turn,
    # so the tokenizer's default re-adds exactly one. The user-voice path
    # keeps add_special_tokens=False because there the template's BOS survives.
    ids = tokenizer(rendered, add_special_tokens=True).input_ids
    return RenderedPrompt(text=rendered, input_ids=list(ids),
                          prompt_token_count=len(ids))


# --- RepE reader scaffolds ---------------------------------------------------

READER_RENDERING_CONVENTION = (
    "rawCompletion scaffold: no chat template, no system role, no family "
    "thinking suffix; tokenized by the extraction path (extractor.activations) "
    "with the tokenizer's default special tokens — single BOS added by the "
    "tokenizer, LAT token = final scaffold token")

#: The convention stamped into a CHAT-TEMPLATE-rendered reader artifact. This
#: is the reference implementation's ``user_tag``/``assistant_tag``
#: construction: the scaffold is the user turn's content and the family
#: template supplies every special token, so the final token is the generation
#: prompt's tail — exactly what ``rep_token=-1`` reads in
#: ``f"{user_tag} {instruction} {scenario} {assistant_tag}"``. Swift twin:
#: ``RepEReader.chatTemplateRenderingConvention``.
READER_CHAT_TEMPLATE_RENDERING_CONVENTION = (
    "chatTemplate scaffold: the rendered scaffold is the USER TURN's content "
    "and the family chat template supplies every special token (single BOS, "
    "turn markers, generation prompt); tokenized by the extraction path "
    "(extractor.activations) through PromptRendering — LAT token = final token "
    "of the generation prompt, the repo's assistant_tag position")

# Chat-template / special-token markers that must never appear in a reader
# scaffold. Under a RAW rendering the scaffold IS the whole token sequence, so
# an embedded BOS or turn marker is the double-BOS / hand-tokenized-template
# hazard; under a CHAT-TEMPLATE rendering the scaffold is the user turn's
# CONTENT, so an embedded marker forges a turn boundary inside a turn. Same
# list, two different reasons — see :func:`render_reader`.
_READER_FORBIDDEN_MARKERS = (
    "<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>",   # Gemma
    "<|im_start|>", "<|im_end|>", "<|endoftext|>",          # Qwen/ChatML
    "</s>",
)


def reader_rendering_convention(rendering=None) -> str:
    """The convention string for a declared reader rendering. Raw resolves to
    the byte-frozen :data:`READER_RENDERING_CONVENTION`, so a raw fit is
    byte-identical to every reader written before the option existed."""
    if rendering is None or getattr(rendering, "is_raw", True):
        return READER_RENDERING_CONVENTION
    return READER_CHAT_TEMPLATE_RENDERING_CONVENTION


def render_reader(text: str, *, model_id: str, rendering=None) -> str:
    """Family-aware pass for a RepE reader scaffold (measurement, not chat).

    **Under ``raw``** (the default, and every reader fitted before the
    rendering became declarable) reader scaffolds are **rawCompletion-style
    plain text**: the RepE LAT reader reads the hidden state at the scaffold's
    final token, so the rendered string must reach the model through exactly
    the tokenizer conventions the extractor uses (``extractor.activations``
    tokenizes with the tokenizer defaults — single BOS, no chat template).
    Family notes:

    - **Gemma**: no system role exists and the HF tokenizer adds BOS itself; a
      scaffold that embeds ``<bos>``/turn markers would double them.
    - **Qwen**: no `` /no_think`` suffix is appended (unlike the rawCompletion
      *generation* path) — nothing is generated, and appending it would move
      the LAT token off the scaffold's final position.

    **Under ``chatTemplate``** the family template IS the marker mechanism — it
    supplies BOS, the turn markers and the generation prompt exactly once, and
    the scaffold becomes the user turn's content. The double-BOS rationale
    above therefore does not apply and its refusal must not fire. What survives
    is a narrower hazard: a marker embedded in the CONTENT forges a turn
    boundary inside the turn, splitting the scaffold the model was meant to
    read as one unit and moving the LAT token off the generation prompt's tail.
    The manual-``<s>`` check is dropped there, because under a template a
    leading ``<s>`` in content is ordinary text, not a hand-written BOS.

    The convention actually applied is stamped into every reader artifact as
    ``renderingConvention`` (see :func:`reader_rendering_convention`).
    """
    del model_id  # both families share the plain-text contract today
    is_raw = rendering is None or getattr(rendering, "is_raw", True)
    for marker in _READER_FORBIDDEN_MARKERS:
        if marker in text:
            if is_raw:
                raise ValueError(
                    f"reader scaffold embeds special/chat-template marker {marker!r} — "
                    "scaffolds must be plain text; the tokenizer adds special tokens "
                    "exactly once (double-BOS / hand-tokenized-template hazard)")
            raise ValueError(
                f"reader scaffold embeds turn marker {marker!r} inside the user "
                "turn's content — under a chatTemplate rendering the template "
                "supplies the markers, so an embedded one forges a turn boundary "
                "and moves the LAT token off the generation prompt. Repair: write "
                "the scaffold as plain content and let the template do the framing")
    if is_raw and text.lstrip().startswith("<s>"):
        raise ValueError(
            "reader scaffold embeds a manual '<s>' BOS — the tokenizer adds it")
    return text


def render_messages(tokenizer, messages: list[dict], *, model_id: str,
                    prompt_mode: str = CHAT_ASSISTANT,
                    system_prompt: str | None = None,
                    qwen_thinking_enabled: bool = False,
                    continue_final_message: bool = False,
                    reasoning_effort: str | None = None) -> RenderedPrompt:
    """Render a multi-turn chat transcript for interactive server chat.

    Study runs intentionally stay on ``render``'s single-prompt path. This
    helper gives the remote Steering tab turn-level parity without pretending
    the whole transcript is one user message.

    Message dicts may carry extra provenance keys — notably ``seeded: true``
    on researcher-authored assistant turns (the metacognition instrument:
    "words in the model's mouth") and ``edited: true`` on generated turns the
    researcher altered afterwards. Rendering deliberately IGNORES them: a
    seeded or edited assistant turn must be byte-identical to a real one in
    the prompt the model sees; the flags live in the request/transcript
    RECORD only.

    With ``continue_final_message=True`` the final message must be an
    assistant turn; it is rendered as an INCOMPLETE turn (no end-of-turn
    marker, no new generation prompt) via transformers' own
    ``continue_final_message`` so generation continues it mid-turn.

    The scripted-transcript STUDY instrument renders through THIS function
    too (:func:`render_transcript`, reached from the unified run executor via
    ``generate``/``score_options``) — the same role-tagged path interactive
    chat uses — so pinned stimulus bytes and the interactive instrument can
    never drift.
    """
    cleaned: list[dict] = []
    for message in messages:
        role = str(message.get("role", "user")).strip() or "user"
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        cleaned.append({"role": role, "content": content})
    if continue_final_message:
        if not cleaned or cleaned[-1]["role"] != "assistant":
            raise ValueError(
                "continueFinalMessage requires the final message to be a "
                "non-empty assistant turn (the prefix the model continues)")
        if prompt_mode == RAW_COMPLETION:
            raise ValueError(
                "continueFinalMessage requires chatAssistant prompt mode — "
                "raw completion has no assistant turn to continue")
    if not cleaned:
        return render(tokenizer, "", model_id=model_id, prompt_mode=prompt_mode,
                      system_prompt=system_prompt,
                      qwen_thinking_enabled=qwen_thinking_enabled,
                      reasoning_effort=reasoning_effort)
    if prompt_mode == RAW_COMPLETION:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in cleaned)
        return render(tokenizer, text, model_id=model_id, prompt_mode=prompt_mode,
                      system_prompt=system_prompt,
                      qwen_thinking_enabled=qwen_thinking_enabled,
                      reasoning_effort=reasoning_effort)

    system = (system_prompt or "").strip()
    chat = list(cleaned)
    if system:
        if _is_gemma(model_id):
            for idx, message in enumerate(chat):
                if message["role"] == "user":
                    chat[idx] = {"role": "user", "content": system + "\n\n" + message["content"]}
                    break
            else:
                chat.insert(0, {"role": "user", "content": system})
        elif chat[0]["role"] != "system":
            chat.insert(0, {"role": "system", "content": system})

    template_kwargs = thinking_template_kwargs(
        model_id, resolve_reasoning_effort(reasoning_effort, qwen_thinking_enabled))
    if continue_final_message:
        # transformers renders the final assistant turn WITHOUT its
        # end-of-turn suffix and adds no generation prompt (sentinel-tag
        # algorithm; raises if the template transforms the final content
        # so the continuation would be dishonest).
        text = _apply_chat_template(
            tokenizer, chat, model_id=model_id,
            tokenize=False, add_generation_prompt=False,
            continue_final_message=True, **template_kwargs)
    else:
        text = _apply_chat_template(
            tokenizer, chat, model_id=model_id,
            tokenize=False, add_generation_prompt=True, **template_kwargs)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return RenderedPrompt(text=text, input_ids=list(ids), prompt_token_count=len(ids))
