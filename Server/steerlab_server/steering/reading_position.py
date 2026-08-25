"""Where in a stimulus the residual stream is read during extraction.

Parallel to Swift ``ReadingPosition`` (``Capture/ActivationRecorder.swift``).
The string ``label`` is part of the sidecar contract — it must match the Swift
labels byte-for-byte ("last token", "mean from token k", "offset from end k",
"last content token", "turn close token", "post-instruction i") so artifacts
written by either engine describe their reading position identically.

MAINTAINER RULING (recorded here because it is the whole point of the named
roles): **raw negative indices are the MECHANISM, named roles are the PORTABLE
FORM.** ``offsetFromEnd(3)`` says "three back from the end" and nothing more —
on Gemma 3 under the chat template that token is ``<start_of_turn>``, on
Llama-3 it is something else entirely, and on a raw stimulus it is a word. A
study should therefore prefer :class:`LastContentToken`,
:class:`TurnCloseToken`, and :class:`PostInstruction`, which name the thing
being read and resolve to whatever concrete index the model family's template
puts it at. ``offsetFromEnd`` exists so an arbitrary offset is DECLARABLE
rather than smuggled in, not because it travels.

Two axes, deliberately separate:

- **shape-only positions** (``lastToken``, ``meanFromToken``,
  ``offsetFromEnd``) resolve from the sequence LENGTH alone and mean the same
  thing under any rendering;
- **template-aware roles** (``lastContentToken``, ``turnCloseToken``,
  ``postInstruction``) resolve against the TOKEN IDS and the tokenizer's own
  special-token inventory. They refuse under raw rendering, because a raw
  stimulus has no turn-close token to find (see
  :class:`ReadingPositionError`).

Resolution is stamped, never re-derived by readers: every artifact records the
REQUESTED position (name + parameter) and the RESOLVED index per sequence
shape (:class:`ResolvedReadingPosition` → ``resolution_report``), mirroring the
way ``layerResolution`` records both the layer and its depth fraction.
"""

from __future__ import annotations

from dataclasses import dataclass

from .extraction_rendering import DECLARATION_FLAG

#: This engine's name in refusals about an unresolvable position.
ENGINE = "python-hf-transformers"


class ReadingPositionError(ValueError):
    """A reading position that cannot be honored for this sequence.

    Typed and carrying a repair, per house style. Never clamped: silently
    reading token 0 because the caller asked for "7 back from the end" of a
    5-token stimulus is the class of quiet substitution the freeze discipline
    exists to forbid.
    """


class ReadingPosition:
    """Base class. Construct via the concrete types or :func:`from_label`."""

    @property
    def label(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def minimum_token_count(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def requested_start_index(self) -> int | None:  # pragma: no cover
        raise NotImplementedError

    # --- recipe-identity projection (see experiment/recipe_identity.py) ---

    @property
    def identity_mode(self) -> str:  # pragma: no cover - overridden
        """The canonical mode token this position contributes to the recipe
        identity."""
        raise NotImplementedError

    @property
    def identity_parameter(self) -> int | None:  # pragma: no cover
        raise NotImplementedError

    # --- rendering dependency ---

    @property
    def requires_templated_rendering(self) -> bool:
        """True for roles that only exist inside a rendered chat turn."""
        return False

    # --- resolution ---

    def resolve(self, token_ids, *, tokenizer=None,
                rendering_is_raw: bool = True) -> "ResolvedReadingPosition":
        """The concrete read window for one tokenized stimulus.

        ``token_ids`` is the sequence actually fed to the model — already
        rendered, so a templated stimulus resolves against the template's own
        tokens. Raises :class:`ReadingPositionError` (typed, with a repair)
        rather than clamping.
        """
        raise NotImplementedError  # pragma: no cover - overridden


@dataclass(frozen=True)
class ResolvedReadingPosition:
    """What was ACTUALLY read, for one sequence.

    ``start_index``/``end_index`` are a half-open window into the sequence, so
    a single-token read is ``(i, i + 1)`` and a pooled read is ``(k, n)``. The
    artifact stamps this alongside the requested position so a reader can see
    what happened without re-deriving template internals.
    """

    requested: str
    mode: str
    parameter: int | None
    start_index: int
    end_index: int
    token_count: int
    #: How the index was derived, in words ("sequence end", "last content
    #: token + 2", …). The provenance half of the stamp.
    source: str

    @property
    def is_pooled(self) -> bool:
        return self.end_index - self.start_index > 1

    @property
    def offset_from_end(self) -> int | None:
        """Distance of the read position from the sequence end, or ``None``
        for a pooled read (which has no single position). This is the
        SEQUENCE-SHAPE invariant: under a fixed template the offset is
        constant while the absolute index moves with stimulus length."""
        if self.is_pooled:
            return None
        return self.token_count - 1 - self.start_index


# --- shape-only positions ----------------------------------------------------


@dataclass(frozen=True)
class LastToken(ReadingPosition):
    """Hidden state of the final token (RepE convention; Phase 0 default)."""

    @property
    def label(self) -> str:
        return "last token"

    @property
    def minimum_token_count(self) -> int:
        return 1

    @property
    def requested_start_index(self) -> int | None:
        return None

    @property
    def identity_mode(self) -> str:
        return "lastToken"

    @property
    def identity_parameter(self) -> int | None:
        return None

    def resolve(self, token_ids, *, tokenizer=None,
                rendering_is_raw: bool = True) -> ResolvedReadingPosition:
        n = len(token_ids)
        _require_length(n, 1, self.label)
        return ResolvedReadingPosition(
            requested=self.label, mode=self.identity_mode, parameter=None,
            start_index=n - 1, end_index=n, token_count=n,
            source="sequence end")


@dataclass(frozen=True)
class MeanFromToken(ReadingPosition):
    """Mean over token positions from ``k`` onward.

    The emotion paper pools from token 50 of paragraph stories. Extraction
    callers must enforce that token ``k`` exists; otherwise this reading
    position is not valid (see :attr:`minimum_token_count`).
    """

    k: int

    @property
    def label(self) -> str:
        return f"mean from token {self.k}"

    @property
    def minimum_token_count(self) -> int:
        return max(1, self.k + 1)

    @property
    def requested_start_index(self) -> int | None:
        return self.k

    @property
    def identity_mode(self) -> str:
        return "meanFromToken"

    @property
    def identity_parameter(self) -> int | None:
        return self.k

    def resolve(self, token_ids, *, tokenizer=None,
                rendering_is_raw: bool = True) -> ResolvedReadingPosition:
        n = len(token_ids)
        _require_length(n, self.minimum_token_count, self.label)
        # Historical clamp preserved EXACTLY: this position has always pooled
        # from min(max(0, k), n - 1); the length guard above is the same one
        # the extraction driver already enforced, so nothing changes.
        start = min(max(0, self.k), n - 1)
        return ResolvedReadingPosition(
            requested=self.label, mode=self.identity_mode, parameter=self.k,
            start_index=start, end_index=n, token_count=n,
            source=f"token {self.k} through sequence end")


@dataclass(frozen=True)
class OffsetFromEnd(ReadingPosition):
    """The single token at index ``last − k``, ``k ≥ 0``.

    ``k = 0`` is exactly :class:`LastToken` — same index, same value — and the
    recipe identity canonicalizes it as such so declaring the offset form can
    never split an identity away from an otherwise-identical last-token
    recipe.

    This is the MECHANISM, not the portable form: ``offsetFromEnd(3)`` names a
    distance, and what lives at that distance depends entirely on the model
    family and the rendering. Prefer a named role when one fits.
    """

    k: int

    def __post_init__(self):
        if self.k < 0:
            raise ReadingPositionError(
                f"offsetFromEnd needs k ≥ 0, got {self.k} — repair: offsets "
                "count BACKWARD from the last token, so k=0 is the last "
                "token and k=3 is three before it")

    @property
    def label(self) -> str:
        return f"offset from end {self.k}"

    @property
    def minimum_token_count(self) -> int:
        return self.k + 1

    @property
    def requested_start_index(self) -> int | None:
        # Not a pooled read: the neutral token bank's "bank from position N
        # onward" question has no answer here, and answering it with k would
        # bank the WRONG rows (k counts from the end).
        return None

    @property
    def identity_mode(self) -> str:
        return "offsetFromEnd"

    @property
    def identity_parameter(self) -> int | None:
        return self.k

    def resolve(self, token_ids, *, tokenizer=None,
                rendering_is_raw: bool = True) -> ResolvedReadingPosition:
        n = len(token_ids)
        if n < self.minimum_token_count:
            raise ReadingPositionError(
                f"sequence is too short for '{self.label}': {n} tokens, needs "
                f"{self.minimum_token_count} — repair: lower k, or drop the "
                f"short stimuli (this position is never clamped, because "
                f"reading token 0 instead would silently change the recipe)")
        index = n - 1 - self.k
        return ResolvedReadingPosition(
            requested=self.label, mode=self.identity_mode, parameter=self.k,
            start_index=index, end_index=index + 1, token_count=n,
            source=f"sequence end − {self.k}")


# --- template-aware named roles ----------------------------------------------


class _TemplateRole(ReadingPosition):
    """Shared plumbing for roles that need a rendered chat turn."""

    @property
    def minimum_token_count(self) -> int:
        # The real bound is not a length: it is "the template put this anchor
        # in the sequence". Resolution enforces that against the token ids.
        return 1

    @property
    def requested_start_index(self) -> int | None:
        return None

    @property
    def identity_parameter(self) -> int | None:
        return None

    @property
    def requires_templated_rendering(self) -> bool:
        return True

    def _require_templated(self, rendering_is_raw: bool) -> None:
        if rendering_is_raw:
            raise ReadingPositionError(
                f"reading position '{self.label}' needs templated rendering: "
                f"a raw stimulus has no chat turn, so there is no "
                f"{self._anchor_noun} to read at — repair: re-attach the "
                f"concept with {DECLARATION_FLAG} "
                f'\'{{"mode": "chatTemplate"}}\', or choose a '
                f"rendering-independent position "
                f"('last token', 'offset from end k', 'mean from token k')")

    @property
    def _anchor_noun(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass(frozen=True)
class LastContentToken(_TemplateRole):
    """The last token of the stimulus's own CONTENT — the token immediately
    before the marker that closes its turn.

    Under ``add_generation_prompt=True`` the render ends with the turn-close
    marker plus the assistant's opening scaffold, so the sequence's own last
    token is template, not content (on Gemma 3 it is a bare newline, which no
    special-token test would catch). This role names the content boundary,
    which is what a study usually means by "read the end of the prompt".
    """

    @property
    def label(self) -> str:
        return "last content token"

    @property
    def identity_mode(self) -> str:
        return "lastContentToken"

    @property
    def _anchor_noun(self) -> str:
        return "content/turn boundary"

    def resolve(self, token_ids, *, tokenizer=None,
                rendering_is_raw: bool = True) -> ResolvedReadingPosition:
        self._require_templated(rendering_is_raw)
        n = len(token_ids)
        index = last_content_index(token_ids, tokenizer, self.label)
        return ResolvedReadingPosition(
            requested=self.label, mode=self.identity_mode, parameter=None,
            start_index=index, end_index=index + 1, token_count=n,
            source="token before the last end-of-turn marker")


@dataclass(frozen=True)
class TurnCloseToken(_TemplateRole):
    """The template's end-of-turn token — Gemma 3's ``<end_of_turn>``, ChatML's
    ``<|im_end|>``, Llama-3's ``<|eot_id|>``.

    Resolved as the LAST end-of-turn marker in the sequence, so under
    ``add_generation_prompt=True`` it is the marker closing the user turn (the
    assistant turn the generation prompt opens is not closed), and under
    ``add_generation_prompt=False`` it is the marker closing the final turn.
    """

    @property
    def label(self) -> str:
        return "turn close token"

    @property
    def identity_mode(self) -> str:
        return "turnCloseToken"

    @property
    def _anchor_noun(self) -> str:
        return "turn-close token"

    def resolve(self, token_ids, *, tokenizer=None,
                rendering_is_raw: bool = True) -> ResolvedReadingPosition:
        self._require_templated(rendering_is_raw)
        n = len(token_ids)
        index = turn_close_index(token_ids, tokenizer, self.label)
        return ResolvedReadingPosition(
            requested=self.label, mode=self.identity_mode,
            parameter=None, start_index=index, end_index=index + 1,
            token_count=n, source="last end-of-turn marker")


@dataclass(frozen=True)
class PostInstruction(_TemplateRole):
    """Arditi's post-instruction convention: the ``i``-th token AFTER the end
    of the instruction content, ``i ∈ 1..5``.

    Those positions are the template's own trailing tokens — for Gemma 3 with
    a generation prompt, ``<end_of_turn>``, ``\\n``, ``<start_of_turn>``,
    ``model``, ``\\n``. Reading there is the refusal-direction literature's
    convention, and naming it as a role is what makes it portable: the same
    declaration lands on the analogous tokens of any family's template instead
    of on a hard-coded ``−5``.
    """

    i: int

    def __post_init__(self):
        if not 1 <= self.i <= 5:
            raise ReadingPositionError(
                f"postInstruction needs i in 1..5, got {self.i} — repair: the "
                "convention covers the five template tokens that follow the "
                "instruction; for anything further out declare an explicit "
                "'offset from end k'")

    @property
    def label(self) -> str:
        return f"post-instruction {self.i}"

    @property
    def identity_mode(self) -> str:
        return "postInstruction"

    @property
    def identity_parameter(self) -> int | None:
        return self.i

    @property
    def _anchor_noun(self) -> str:
        return "instruction boundary"

    def resolve(self, token_ids, *, tokenizer=None,
                rendering_is_raw: bool = True) -> ResolvedReadingPosition:
        self._require_templated(rendering_is_raw)
        n = len(token_ids)
        content = last_content_index(token_ids, tokenizer, self.label)
        index = content + self.i
        if index >= n:
            raise ReadingPositionError(
                f"reading position '{self.label}' did not resolve: the "
                f"instruction ends at token {content} of {n}, so there is no "
                f"token {self.i} after it — repair: lower i, or render with "
                f"addGenerationPrompt true so the template's trailing tokens "
                f"are present")
        return ResolvedReadingPosition(
            requested=self.label, mode=self.identity_mode, parameter=self.i,
            start_index=index, end_index=index + 1, token_count=n,
            source=f"last content token + {self.i}")


# --- template anchors --------------------------------------------------------

#: End-of-turn markers looked up by NAME in the tokenizer's vocabulary, in
#: addition to whatever it declares as its eos. Cross-engine contract: the
#: Swift twin (``ReadingPosition.endOfTurnTokens``) probes the same strings in
#: the same order.
END_OF_TURN_TOKENS = (
    "<end_of_turn>",    # Gemma 3
    "<|im_end|>",       # Qwen / ChatML
    "<|eot_id|>",       # Llama 3
    "<|end|>",          # Phi
    "</s>",             # Llama 2 / Mistral
)


def end_of_turn_ids(tokenizer) -> set[int]:
    """Ids of this tokenizer's end-of-turn markers (see
    :data:`END_OF_TURN_TOKENS`), plus its declared eos."""
    ids: set[int] = set()
    unk = getattr(tokenizer, "unk_token_id", None)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        for token in END_OF_TURN_TOKENS:
            try:
                value = convert(token)
            except Exception:  # pragma: no cover - defensive
                continue
            if isinstance(value, int) and value >= 0 and value != unk:
                ids.add(value)
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int) and eos >= 0:
        ids.add(eos)
    return ids


def turn_close_index(token_ids, tokenizer, label: str) -> int:
    """Index of the LAST end-of-turn marker in the sequence.

    ONE mechanism for all three named roles, and one both engines have: a
    marker lookup by name plus the tokenizer's declared eos. A "last token
    that is not special" rule would have been engine-specific AND wrong — on
    Gemma 3 the generation prompt ends in a bare newline, an ordinary content
    token, so a special-token scan lands past the instruction rather than
    inside it. The Swift twin (``ReadingPosition.turnCloseIndex``) probes the
    same markers in the same order.
    """
    markers = end_of_turn_ids(tokenizer)
    if not markers:
        raise ReadingPositionError(
            f"reading position '{label}' did not resolve on the {ENGINE} "
            f"engine: this tokenizer declares no end-of-turn token (looked "
            f"for {', '.join(END_OF_TURN_TOKENS)} and the tokenizer's own "
            f"eos) — repair: read at an explicit 'offset from end k', which "
            f"needs no template anchor")
    for index in range(len(token_ids) - 1, -1, -1):
        if int(token_ids[index]) in markers:
            return index
    raise ReadingPositionError(
        f"reading position '{label}' did not resolve: the rendered sequence "
        f"({len(token_ids)} tokens) contains no end-of-turn marker — repair: "
        f"check that extractionRendering really is 'chatTemplate' for this "
        f"concept, or read at an explicit 'offset from end k'")


def last_content_index(token_ids, tokenizer, label: str) -> int:
    """Index of the last token of the stimulus's own content: the token
    immediately before the turn-close marker."""
    close = turn_close_index(token_ids, tokenizer, label)
    if close == 0:
        raise ReadingPositionError(
            f"reading position '{label}' did not resolve: the turn closes at "
            f"token 0, so the stimulus contributed no content — repair: check "
            f"the stimulus is non-empty and that the rendering is the one you "
            f"meant")
    return close - 1


def _require_length(n: int, minimum: int, label: str) -> None:
    if n < minimum:
        raise ReadingPositionError(
            f"sequence is too short for '{label}': {n} tokens, needs "
            f"{minimum} — repair: drop the short stimuli, or choose a "
            f"position that fits them")


# --- constructors + label round trip -----------------------------------------


def mean_from_token(k: int) -> MeanFromToken:
    return MeanFromToken(k=k)


def offset_from_end(k: int) -> OffsetFromEnd:
    return OffsetFromEnd(k=k)


def post_instruction(i: int) -> PostInstruction:
    return PostInstruction(i=i)


LAST_TOKEN = LastToken()
LAST_CONTENT_TOKEN = LastContentToken()
TURN_CLOSE_TOKEN = TurnCloseToken()

_PARAMETERIZED = (
    ("mean from token ", mean_from_token),
    ("offset from end ", offset_from_end),
    ("post-instruction ", post_instruction),
)

_BARE = {
    "last token": LAST_TOKEN,
    "last content token": LAST_CONTENT_TOKEN,
    "turn close token": TURN_CLOSE_TOKEN,
}


def from_label(label: str) -> ReadingPosition:
    """Parse a sidecar/manifest ``label`` back into a reading position.

    Inverse of :attr:`ReadingPosition.label`. Unknown labels fall back to
    last-token, matching the Swift side's tolerance for older artifacts. Use
    :func:`parse_label_strict` where a wrong guess would be laundered into an
    identity.
    """
    return parse_label_strict(label) or LAST_TOKEN


def parse_label_strict(label) -> ReadingPosition | None:
    """STRICT label parse: ``None`` for anything unrecognized, so a caller
    that must not guess (recipe identity) can refuse instead."""
    if not isinstance(label, str):
        return None
    text = label.strip().lower()
    bare = _BARE.get(text)
    if bare is not None:
        return bare
    for prefix, make in _PARAMETERIZED:
        if text.startswith(prefix):
            rest = text[len(prefix):].strip()
            if not rest.isdigit():
                return None
            try:
                return make(int(rest))
            except ReadingPositionError:
                return None
    return None
