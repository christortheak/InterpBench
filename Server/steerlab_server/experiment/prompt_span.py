"""Resolve a declared PROMPT SPAN to token positions of the exact prompt the
engine feeds the model (2026-10-31).

A span names one part of the study prompt — an instruction sentence, say — so
an intervention can act while the model READS that part and nowhere else.
This module is the resolver alone: pure, engine-side, and stamped. Policies
do not read it yet; the integration slice turns its ranges into a prefill
strength mask.

THE INPUTS. Exactly what :func:`prompt_render.render` takes for one study
prompt — the tokenizer, the user text (``prompt``), the model id, the prompt
mode (chat template or raw completion), the system prompt, the reasoning
effort (or the legacy thinking boolean) and the capability record — plus a
span declaration anchored on the USER text. The resolver renders through
that same function, so the text and ids it resolves against are the ones the
engine feeds the model; a caller that already rendered passes its
``RenderedPrompt`` as ``rendered`` and the resolver refuses unless a fresh
render reproduces it.

THE DECLARATION GRAMMAR (a closed JSON object; unknown keys are refused)::

    {"kind": "substring", "text": "<exact>", "occurrence": 2}
    {"kind": "anchors", "start": "<exact>", "end": "<exact>",
     "startOccurrence": 1, "endOccurrence": 1}
    {"kind": "userMessage"}
    {"kind": "allPrompt"}

- Matching is EXACT over the user text as given: case, punctuation,
  whitespace and Unicode form all count. No normalization is applied to
  either side; a near miss (it would match after NFC, ignoring case, or with
  different whitespace) is named in the refusal's repair.
- Occurrences are 1-BASED and count every START position, left to right,
  overlapping ones included (``"aa"`` occurs twice in ``"aaa"``). An omitted
  occurrence means "the only one": the text must then match exactly once,
  and a second match is refused as ambiguous. A declared occurrence past the
  last match is refused as missing. The evidence records the match count, so
  "occurrence 1 of 3" is visible in every stamp.
- ``substring`` selects the characters of the chosen match.
- ``anchors`` selects from the first character of the start match THROUGH
  the last character of the end match: both anchors are INCLUSIVE. The end
  anchor is searched only from the end of the chosen start match onward, and
  ``endOccurrence`` counts matches there (``endOccurrence: 1`` is the next
  one); an end anchor overlapping or preceding the start is missing.
- An anchor text must contain a non-whitespace character: a template may
  strip or collapse whitespace, so a whitespace-only anchor could name
  characters the model never reads.
- ``userMessage`` selects every character of the user text — the study
  prompt itself, NEVER the system prompt, even on a template that folds the
  system prompt into the user turn (Gemma 3): the folded text and its
  separator are delivery framing, and selecting the same characters on every
  family keeps a span comparable across them.
- ``allPrompt`` selects every position of the rendered ids — BOS, template
  markers, system prompt and generation prompt included. It needs no
  character offsets, so it resolves on any tokenizer.

Named template regions (the tokens that open the assistant turn, the system
turn, a thinking block) are out of scope here; they would be further kinds
resolved from the same framing, later.

THE OUTPUT (:class:`ResolvedSpan`). ``token_ranges`` — half-open
``[start, end)`` index ranges into the rendered prompt's ids (one range for
every kind today; a tuple so later kinds may select several) — and
``evidence``, the JSON object a run stamps per response:

- ``resolverVersion`` (:data:`RESOLVER_VERSION`) and the canonical
  ``declaration``;
- ``promptMode``, ``systemPromptDelivery`` (None without a system prompt)
  and ``addSpecialTokens`` (the tokenize flag the renderer used);
- ``tokenRanges``, ``tokenCount``, ``promptTokenCount``;
- ``userCharRange`` (None for ``allPrompt``) and ``renderedCharRange`` —
  the declared characters in user-text and in rendered-text coordinates,
  half-open; ``unrenderedCharacters`` counts declared characters the
  template dropped (stripped or collapsed whitespace), and
  ``uncoveredCharacters`` rendered characters of the range that no
  token's offsets cover (both None for ``allPrompt``);
- ``match`` — the chosen occurrence and match count (per anchor for
  ``anchors``), or None;
- ``selectedText`` (the selected ids, decoded with special tokens kept) and
  ``selectedRenderedText`` (the rendered characters those tokens cover);
- ``boundaryTokens`` — ``first`` and ``last``, each with ``index``,
  ``tokenID``, ``text``, ``charRange``, ``straddles`` and ``overhang`` (the
  token's characters outside the declared range on that side);
- ``userTextLocation`` — how the user text was found (below), or None;
- ``renderedTextSHA256`` (UTF-8 bytes of the rendered text) and
  ``promptTokensSHA256`` (``json.dumps`` of the id list, the encoding
  ``outputTokensSHA256`` already uses).

MID-TOKEN EDGES. One rule: a token is selected when its character span
OVERLAPS the declared rendered range (``start < hi and end > lo``), and the
selection is the contiguous range from the first such token to the last, so
an interior token with an empty span (rare) stays in. A boundary token that
reaches past the declared edge ``straddles`` it, and its ``overhang`` says by
exactly which characters — a byte-level token's leading space, a
SentencePiece merge across a word boundary, or template text adjacent to the
user turn. A zero-width token (a BOS the tokenizer adds at offset ``(0, 0)``)
overlaps nothing, so it is never selected by a character span.

LOCATING THE USER TEXT. The user text is never searched for in the rendered
prompt — a short text can also occur in the system prompt or the template's
own markers. Instead the resolver renders the SAME inputs twice more with the
user text replaced by two sentinels of different lengths; each must appear
exactly once, and the text before and after it (the FRAMING: BOS, turn
markers, a folded system prompt and its separator, the generation prompt, a
raw-mode ``/no_think``) must be identical for both. The real rendered text
must then begin with that prefix and end with that suffix; what lies between
is the window the template wrote the user text into. The user text is
aligned to the window character by character: non-whitespace characters
must match one for one and in order, while whitespace may be dropped or
inserted — which absorbs Gemma's ``| trim`` and any template that
normalizes line endings or collapses runs of spaces. Anything else is
refused as unlocated.

Limits, each refused rather than guessed: a template that rewrites the
content beyond whitespace (case, escaping, Unicode normalization), a
template whose framing depends on the content, and a system prompt that
contains a sentinel. Tokenizer-level normalization (Qwen's NFC, Gemma's
``▁``) is not a limit: fast-tokenizer offsets refer to the original text.
One effect of it is recorded instead: a character the normalizer folds
into its neighbour (a combining accent NFC composes) is covered by no
token's offsets, so it is counted in ``uncoveredCharacters`` and cannot
select a token on its own.
Scripted transcripts (several user turns) are not handled; the resolver
renders through the single-prompt path only.

CHARACTERS TO TOKENS. The rendered text is re-tokenized with
``return_offsets_mapping=True`` under the renderer's own special-token flag
(:func:`prompt_render.tokenizes_with_special_tokens`: raw completion adds
the tokenizer's BOS, a chat render does not because the template wrote it).
The ids of that call must equal the renderer's ids exactly, or nothing
resolves.

REFUSALS (:class:`SpanRefusal`, a ``ValueError`` carrying ``code`` and
``repair_action``): :data:`DECLARATION_INVALID`, :data:`ANCHOR_MISSING`,
:data:`ANCHOR_AMBIGUOUS`, :data:`OFFSETS_UNAVAILABLE`,
:data:`USER_TEXT_UNLOCATED`, :data:`RENDER_MISMATCH`,
:data:`IDS_NOT_REPRODUCED` and :data:`EMPTY_SPAN`.

Import discipline: stdlib and :mod:`prompt_render` only; the tokenizer
arrives as an argument.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass

from . import prompt_render

#: Stamped into every evidence object; bump when a rule above changes what a
#: declaration selects.
RESOLVER_VERSION = "prompt-span-resolver/1"

SUBSTRING = "substring"
ANCHORS = "anchors"
USER_MESSAGE = "userMessage"
ALL_PROMPT = "allPrompt"
SPAN_KINDS: tuple[str, ...] = (SUBSTRING, ANCHORS, USER_MESSAGE, ALL_PROMPT)

#: Per kind: the keys a declaration must carry, and the ones it may.
_REQUIRED_KEYS = {SUBSTRING: {"kind", "text"}, ANCHORS: {"kind", "start", "end"},
                  USER_MESSAGE: {"kind"}, ALL_PROMPT: {"kind"}}
_OPTIONAL_KEYS = {SUBSTRING: {"occurrence"},
                  ANCHORS: {"startOccurrence", "endOccurrence"},
                  USER_MESSAGE: set(), ALL_PROMPT: set()}

DECLARATION_INVALID = "spanDeclarationInvalid"
ANCHOR_MISSING = "spanAnchorMissing"
ANCHOR_AMBIGUOUS = "spanAnchorAmbiguous"
OFFSETS_UNAVAILABLE = "spanOffsetsUnavailable"
USER_TEXT_UNLOCATED = "spanUserTextUnlocated"
RENDER_MISMATCH = "spanRenderMismatch"
IDS_NOT_REPRODUCED = "spanIdsNotReproduced"
EMPTY_SPAN = "spanEmpty"
REFUSAL_CODES: tuple[str, ...] = (
    DECLARATION_INVALID, ANCHOR_MISSING, ANCHOR_AMBIGUOUS, OFFSETS_UNAVAILABLE,
    USER_TEXT_UNLOCATED, RENDER_MISMATCH, IDS_NOT_REPRODUCED, EMPTY_SPAN)

#: The two stand-in user texts the framing is read from: ASCII letters no
#: template transforms, of different lengths so a template whose framing
#: depends on the content's length shows it.
SENTINELS: tuple[str, str] = ("steerlabspansentinel", "steerlabspansentinelsecondform")

LOCATION_METHOD = "sentinelFraming"


class SpanRefusal(ValueError):
    """A typed refusal carrying its machine code and a practical repair — the
    shape ``battery_run.BatteryRunRefusal`` carries. Nothing is resolved
    when one is raised."""

    def __init__(self, reason: str, *, code: str, repair: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code
        self.repair_action = repair

    def detail(self) -> dict:
        return {"code": self.code, "reason": self.reason,
                "repairAction": self.repair_action}


def _shown(text: str, limit: int = 80) -> str:
    return repr(text if len(text) <= limit else text[:limit] + "…")


# --- the declaration ---------------------------------------------------------

@dataclass(frozen=True)
class SpanDeclaration:
    """One parsed span declaration (see the module docstring's grammar)."""
    kind: str
    text: str | None = None
    occurrence: int | None = None
    start: str | None = None
    end: str | None = None
    start_occurrence: int | None = None
    end_occurrence: int | None = None

    def to_dict(self) -> dict:
        """The canonical wire form: only the keys this kind carries, an
        omitted occurrence left out (it means "the only one")."""
        out: dict = {"kind": self.kind}
        if self.kind == SUBSTRING:
            out["text"] = self.text
            if self.occurrence is not None:
                out["occurrence"] = self.occurrence
        elif self.kind == ANCHORS:
            out["start"] = self.start
            out["end"] = self.end
            if self.start_occurrence is not None:
                out["startOccurrence"] = self.start_occurrence
            if self.end_occurrence is not None:
                out["endOccurrence"] = self.end_occurrence
        return out


def _invalid(reason: str, repair: str = "") -> SpanRefusal:
    return SpanRefusal(reason, code=DECLARATION_INVALID, repair=repair or (
        "declare one of: {\"kind\": \"substring\", \"text\": ...}, "
        "{\"kind\": \"anchors\", \"start\": ..., \"end\": ...}, "
        "{\"kind\": \"userMessage\"} or {\"kind\": \"allPrompt\"}"))


def _anchor_text(raw: dict, key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _invalid(
            f"span {key!r} must be a string with a non-whitespace character — "
            f"got {value!r}",
            f"copy the exact {key!r} text from the user prompt; a template may "
            "strip or collapse whitespace, so whitespace alone cannot anchor a span")
    return value


def _occurrence(raw: dict, key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _invalid(
            f"span {key!r} must be a positive integer (1 is the first match) — "
            f"got {value!r}",
            f"declare {key!r} as 1, 2, … counting matches from the left, or "
            "omit it when the text occurs exactly once")
    return value


def parse_declaration(raw) -> SpanDeclaration:
    """Validate one declaration object; refuses with
    :data:`DECLARATION_INVALID`. Needs no tokenizer, so an authoring surface
    can check the shape before anything is rendered. A
    :class:`SpanDeclaration` built by hand is validated through its wire
    form like any other."""
    if isinstance(raw, SpanDeclaration):
        raw = raw.to_dict()
    if not isinstance(raw, dict):
        raise _invalid(f"a span declaration must be an object — got {type(raw).__name__}")
    kind = raw.get("kind")
    if kind not in SPAN_KINDS:
        raise _invalid(f"unknown span kind {kind!r} — known: " + ", ".join(SPAN_KINDS))
    keys = set(raw)
    missing = sorted(_REQUIRED_KEYS[kind] - keys)
    unknown = sorted(keys - _REQUIRED_KEYS[kind] - _OPTIONAL_KEYS[kind])
    if missing or unknown:
        allowed = sorted(_REQUIRED_KEYS[kind] | _OPTIONAL_KEYS[kind])
        raise _invalid(
            f"a {kind} span " + "; ".join(
                part for part in (
                    ("is missing " + ", ".join(missing)) if missing else "",
                    ("does not take " + ", ".join(unknown)) if unknown else "")
                if part),
            f"a {kind} span takes exactly: " + ", ".join(allowed))
    if kind == SUBSTRING:
        return SpanDeclaration(kind, text=_anchor_text(raw, "text"),
                               occurrence=_occurrence(raw, "occurrence"))
    if kind == ANCHORS:
        return SpanDeclaration(kind, start=_anchor_text(raw, "start"),
                               end=_anchor_text(raw, "end"),
                               start_occurrence=_occurrence(raw, "startOccurrence"),
                               end_occurrence=_occurrence(raw, "endOccurrence"))
    return SpanDeclaration(kind)


# --- the declared characters, in user-text coordinates -----------------------

def _starts(text: str, needle: str, begin: int = 0) -> list[int]:
    """Every start position of ``needle`` at or after ``begin``, overlapping
    matches included."""
    found = []
    at = text.find(needle, begin)
    while at >= 0:
        found.append(at)
        at = text.find(needle, at + 1)
    return found


def _near_miss(text: str, needle: str) -> str:
    """Why an exact match may have failed, for the repair; empty when none of
    the usual near misses applies."""
    if unicodedata.normalize("NFC", needle) in unicodedata.normalize("NFC", text):
        return (" It matches only after Unicode normalization (NFC): the prompt "
                "spells some character with a different code-point sequence.")
    if needle.casefold() in text.casefold():
        return " It matches only when case is ignored."
    if " ".join(needle.split()) in " ".join(text.split()):
        return " It matches only with different whitespace."
    return ""


def _choose(text: str, needle: str, occurrence: int | None, label: str,
            begin: int = 0) -> tuple[int, dict]:
    """The chosen match's start in ``text`` and its evidence."""
    found = _starts(text, needle, begin)
    where = "the user prompt" if begin == 0 else "the user prompt after the start anchor"
    if not found:
        hint = _near_miss(text[begin:], needle)
        if begin and needle in text:
            hint += " It occurs only before (or overlapping) the start anchor."
        raise SpanRefusal(
            f"span {label} {_shown(needle)} does not occur in {where}",
            code=ANCHOR_MISSING,
            repair=("copy the text exactly as the prompt spells it (matching is "
                    "exact: case, punctuation, whitespace and Unicode form)." + hint))
    if occurrence is None:
        if len(found) > 1:
            raise SpanRefusal(
                f"span {label} {_shown(needle)} occurs {len(found)} times in "
                f"{where}, and no occurrence is declared",
                code=ANCHOR_AMBIGUOUS,
                repair=(f"declare which match to use (1 to {len(found)}, counted "
                        "from the left), or lengthen the text until it occurs once"))
        occurrence = 1
    elif occurrence > len(found):
        raise SpanRefusal(
            f"span {label} {_shown(needle)} occurrence {occurrence} was declared, "
            f"but it occurs only {len(found)} time(s) in {where}",
            code=ANCHOR_MISSING,
            repair=f"declare an occurrence from 1 to {len(found)}")
    at = found[occurrence - 1]
    return at, {"occurrence": occurrence, "matchCount": len(found),
                "userCharRange": [at, at + len(needle)]}


def declared_user_range(declaration, prompt: str) -> tuple[tuple[int, int] | None, dict | None]:
    """The declared characters as a half-open range of the user text, and the
    match evidence; ``(None, None)`` for ``allPrompt``. Pure over the text."""
    spec = parse_declaration(declaration)
    if spec.kind == ALL_PROMPT:
        return None, None
    if spec.kind == USER_MESSAGE:
        if not prompt.strip():
            raise SpanRefusal(
                "a userMessage span was declared, but the user prompt is empty "
                "or whitespace only", code=EMPTY_SPAN,
                repair="declare allPrompt, or give the item a non-empty prompt")
        return (0, len(prompt)), None
    if spec.kind == SUBSTRING:
        at, match = _choose(prompt, spec.text, spec.occurrence, "text")
        return (at, at + len(spec.text)), match
    start_at, start = _choose(prompt, spec.start, spec.start_occurrence, "start anchor")
    end_at, end = _choose(prompt, spec.end, spec.end_occurrence, "end anchor",
                          begin=start_at + len(spec.start))
    return (start_at, end_at + len(spec.end)), {"start": start, "end": end}


# --- the user text, in rendered-text coordinates -----------------------------

def _align(user: str, window: str) -> list[int | None] | None:
    """Each user character's index in ``window`` (None for whitespace the
    template dropped), or None when the two differ beyond whitespace.
    Non-whitespace characters match one for one, in order; whitespace on
    either side may be skipped, and whitespace the template rewrote as other
    whitespace (a tab as a space) maps to what it became."""
    if user == window:
        return list(range(len(user)))
    mapping: list[int | None] = [None] * len(user)
    i = j = 0
    while i < len(user) and j < len(window):
        if user[i] == window[j] or (user[i].isspace() and window[j].isspace()):
            mapping[i] = j
            i += 1
            j += 1
        elif user[i].isspace():
            i += 1
        elif window[j].isspace():
            j += 1
        else:
            return None
    if user[i:].strip() or window[j:].strip():
        return None
    return mapping


def _unlocated(reason: str) -> SpanRefusal:
    return SpanRefusal(
        reason, code=USER_TEXT_UNLOCATED,
        repair=("this rendering does not place the user text where a span can be "
                "mapped onto it; declare allPrompt, or render the study in "
                "rawCompletion mode, and say in METHODS which positions were "
                "steered"))


def locate_user_text(tokenizer, prompt: str, rendered_text: str,
                     **render_kwargs) -> tuple[tuple[int, int], list[int | None], str]:
    """Where the template wrote the user text: the window as a half-open
    range of the rendered text, each user character's index within the
    window (None where the template dropped it), and the normalization seen
    (``identical`` or ``whitespace``). ``render_kwargs`` are
    :func:`prompt_render.render`'s keyword arguments, exactly as the real
    prompt was rendered."""
    framings = []
    for sentinel in SENTINELS:
        try:
            probe = prompt_render.render(tokenizer, sentinel, **render_kwargs).text
        except ValueError as exc:
            raise _unlocated(f"the template refuses a plain stand-in user text: {exc}") from exc
        at = probe.find(sentinel)
        if at < 0 or probe.find(sentinel, at + 1) >= 0:
            raise _unlocated(
                "the template does not write a plain user text verbatim exactly "
                "once (it transforms it, drops it, or the system prompt contains "
                "the stand-in)")
        framings.append((probe[:at], probe[at + len(sentinel):]))
    (prefix, suffix), other = framings
    if other != (prefix, suffix):
        raise _unlocated("the template frames the user text differently "
                         "depending on what the text says")
    if (len(rendered_text) < len(prefix) + len(suffix)
            or not rendered_text.startswith(prefix)
            or not rendered_text.endswith(suffix)):
        raise _unlocated("the rendered prompt is not framed the way the template "
                         "frames a plain user text")
    window_end = len(rendered_text) - len(suffix)
    window = rendered_text[len(prefix):window_end]
    mapping = _align(prompt, window)
    if mapping is None:
        raise _unlocated("the template changes the user text beyond whitespace, "
                         "so its characters cannot be matched one for one")
    return ((len(prefix), window_end), mapping,
            "identical" if window == prompt else "whitespace")


# --- characters to tokens -----------------------------------------------------

def _offsets(tokenizer, text: str, ids: list[int], prompt_mode: str) -> list[tuple[int, int]]:
    """Character offsets for the renderer's ids, re-tokenized under the
    renderer's own special-token flag; refuses when the tokenizer gives none
    or the ids differ."""
    repair = ("load the model's fast tokenizer (its tokenizer.json, through "
              "AutoTokenizer with use_fast=True); only a fast tokenizer maps "
              "tokens to characters. Or declare allPrompt, which needs no offsets")
    if getattr(tokenizer, "is_fast", None) is False:
        raise SpanRefusal("this tokenizer is not a fast tokenizer, so it gives no "
                          "character offsets", code=OFFSETS_UNAVAILABLE, repair=repair)
    try:
        encoded = tokenizer(text, add_special_tokens=prompt_render.tokenizes_with_special_tokens(
            prompt_mode), return_offsets_mapping=True)
        offset_ids = list(encoded["input_ids"])
        offsets = [(int(s), int(e)) for s, e in encoded["offset_mapping"]]
    except (NotImplementedError, TypeError, KeyError, ValueError) as exc:
        raise SpanRefusal(f"the tokenizer gives no character offsets: {exc}",
                          code=OFFSETS_UNAVAILABLE, repair=repair) from exc
    if len(offsets) != len(offset_ids) or any(
            not 0 <= s <= e <= len(text) for s, e in offsets):
        raise SpanRefusal("the tokenizer's offsets do not describe the rendered "
                          "text", code=OFFSETS_UNAVAILABLE, repair=repair)
    if offset_ids != ids:
        raise _ids_differ("re-tokenizing the rendered text with offsets")
    return offsets


def _ids_differ(how: str) -> SpanRefusal:
    return SpanRefusal(
        f"{how} does not reproduce the ids the renderer feeds the model",
        code=IDS_NOT_REPRODUCED,
        repair=("resolve with the same tokenizer object that rendered the prompt, "
                "and pass the RenderedPrompt exactly as rendered (no truncation or "
                "added tokens); if they already match, report the tokenizer — its "
                "offset path disagrees with its plain one"))


def _decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)


def _boundary(tokenizer, ids, offsets, index: int, text: str,
              edge: int | None, side: str) -> dict:
    char_range = list(offsets[index]) if offsets is not None else None
    straddles, overhang = False, ""
    if char_range is not None:
        start, end = char_range
        if side == "first" and start < edge:
            straddles, overhang = True, text[start:edge]
        elif side == "last" and end > edge:
            straddles, overhang = True, text[edge:end]
    return {"index": index, "tokenID": ids[index],
            "text": _decode(tokenizer, [ids[index]]), "charRange": char_range,
            "straddles": straddles, "overhang": overhang}


# --- the resolver -------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedSpan:
    """A span resolved against one rendered prompt: half-open index ranges
    into its ids, and the evidence a run stamps per response."""
    token_ranges: tuple[tuple[int, int], ...]
    evidence: dict

    @property
    def token_count(self) -> int:
        return sum(end - start for start, end in self.token_ranges)

    def positions(self) -> list[int]:
        """Every selected prompt position, ascending."""
        return [p for start, end in self.token_ranges for p in range(start, end)]


def resolve(tokenizer, declaration, *, prompt: str, model_id: str,
            prompt_mode: str = prompt_render.CHAT_ASSISTANT,
            system_prompt: str | None = None,
            qwen_thinking_enabled: bool = False,
            reasoning_effort: str | None = None,
            capabilities=None,
            rendered=None) -> ResolvedSpan:
    """Resolve ``declaration`` (a dict or :class:`SpanDeclaration`) against
    the prompt :func:`prompt_render.render` produces for these inputs.

    ``rendered``, when given, is the ``RenderedPrompt`` the engine will feed
    the model; it must be byte-identical to a fresh render of the same inputs
    (:data:`RENDER_MISMATCH`) with identical ids (:data:`IDS_NOT_REPRODUCED`).
    Raises :class:`SpanRefusal`; render-time errors of the prompt itself
    (an unrenderable effort, a refused system prompt) propagate unchanged.
    """
    spec = parse_declaration(declaration)
    record = prompt_render.capabilities_for(model_id, tokenizer, capabilities)
    render_kwargs = {"model_id": model_id, "prompt_mode": prompt_mode,
                     "system_prompt": system_prompt,
                     "qwen_thinking_enabled": qwen_thinking_enabled,
                     "reasoning_effort": reasoning_effort, "capabilities": record}
    fresh = prompt_render.render(tokenizer, prompt, **render_kwargs)
    if rendered is not None and rendered.text != fresh.text:
        raise SpanRefusal(
            "the rendered prompt given to the resolver is not what these rendering "
            "inputs produce", code=RENDER_MISMATCH,
            repair=("resolve with exactly the prompt, prompt mode, system prompt, "
                    "reasoning effort and capability record the prompt was "
                    "rendered with"))
    target = rendered if rendered is not None else fresh
    text, ids = target.text, [int(i) for i in target.input_ids]
    if [int(i) for i in fresh.input_ids] != ids:
        raise _ids_differ("tokenizing the rendered text")
    if not ids:
        raise SpanRefusal("the rendered prompt has no tokens", code=EMPTY_SPAN,
                          repair="give the item a non-empty prompt")

    user_range, match = declared_user_range(spec, prompt)
    evidence: dict = {
        "resolverVersion": RESOLVER_VERSION,
        "declaration": spec.to_dict(),
        "promptMode": prompt_mode,
        "systemPromptDelivery": (prompt_render.system_prompt_delivery(
            model_id, prompt_mode, record) if (system_prompt or "").strip() else None),
        "addSpecialTokens": prompt_render.tokenizes_with_special_tokens(prompt_mode),
        "promptTokenCount": len(ids),
        "renderedTextSHA256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "promptTokensSHA256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "userCharRange": list(user_range) if user_range else None,
        "match": match,
    }

    if spec.kind == ALL_PROMPT:
        first, last, offsets = 0, len(ids) - 1, None
        lo, hi = 0, len(text)
        evidence.update(renderedCharRange=[lo, hi], unrenderedCharacters=None,
                        uncoveredCharacters=None, userTextLocation=None,
                        selectedRenderedText=text)
    else:
        offsets = _offsets(tokenizer, text, ids, prompt_mode)
        (window_start, window_end), mapping, normalization = locate_user_text(
            tokenizer, prompt, text, **render_kwargs)
        declared = mapping[user_range[0]:user_range[1]]
        mapped = [j for j in declared if j is not None]
        if not mapped:
            raise SpanRefusal(
                "every declared character is whitespace the template drops, so "
                "the model never reads it", code=EMPTY_SPAN,
                repair="anchor the span on visible text")
        lo, hi = window_start + mapped[0], window_start + mapped[-1] + 1
        selected = [i for i, (s, e) in enumerate(offsets) if e > s and s < hi and e > lo]
        if not selected:
            raise SpanRefusal("no token covers the declared characters",
                              code=EMPTY_SPAN, repair="anchor the span on visible text")
        first, last = selected[0], selected[-1]
        covered = set()
        for s, e in (offsets[i] for i in selected):
            covered.update(range(max(s, lo), min(e, hi)))
        evidence.update(
            renderedCharRange=[lo, hi],
            unrenderedCharacters=len(declared) - len(mapped),
            uncoveredCharacters=hi - lo - len(covered),
            userTextLocation={"method": LOCATION_METHOD, "normalization": normalization,
                              "windowCharRange": [window_start, window_end]},
            selectedRenderedText=text[min(offsets[i][0] for i in selected):
                                      max(offsets[i][1] for i in selected)])

    ranges = ((first, last + 1),)
    evidence.update(
        tokenRanges=[list(r) for r in ranges],
        tokenCount=last + 1 - first,
        selectedText=_decode(tokenizer, ids[first:last + 1]),
        boundaryTokens={
            "first": _boundary(tokenizer, ids, offsets, first, text, lo, "first"),
            "last": _boundary(tokenizer, ids, offsets, last, text, hi, "last")})
    return ResolvedSpan(token_ranges=ranges, evidence=evidence)
