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
  ``postInstruction``, ``contentOffset``) resolve against the TOKEN IDS and
  the tokenizer's own special-token inventory. They refuse under raw
  rendering, because a raw stimulus has no turn-close token to find (see
  :class:`ReadingPositionError`).

:class:`MeanContentFromToken` is deliberately in NEITHER camp: it pools over
CONTENT tokens only, which under a template needs the template map and under
raw is simply every token — so it resolves under both renderings and says so
in its stamp rather than refusing (under raw it is exactly
``meanFromToken(n)``; the identity still distinguishes the two, because the
DECLARATIONS differ and a recipe identity records what was declared).

Resolution is stamped, never re-derived by readers: every artifact records the
REQUESTED position (name + parameter) and the RESOLVED index per sequence
shape (:class:`ResolvedReadingPosition` → ``resolution_report``), mirroring the
way ``layerResolution`` records both the layer and its depth fraction.
"""

from __future__ import annotations

from dataclasses import dataclass

from .extraction_rendering import DECLARATION_FLAG as RENDERING_FLAG

#: This engine's name in refusals about an unresolvable position.
ENGINE = "python-hf-transformers"

#: The flag both engines' CLIs pin a reading position with, and the name every
#: refusal that asks for one must use. Swift twin:
#: ``ReadingPosition.declarationFlag``.
DECLARATION_FLAG = "--reading-position"

#: The LEGACY spelling for one specific position (``mean from token k``). It
#: predates the label vocabulary and stays, because studies and scripts type
#: it — but the two may never be declared together (see
#: :func:`declaration_conflict`). Swift twin: ``ReadingPosition.poolFromFlag``.
POOL_FROM_FLAG = "--pool-from"

#: Every spelling :func:`parse_declaration` accepts, as a person would type it.
#: This IS the cross-engine vocabulary (the committed fixture
#: ``extraction-rendering-and-positions.json`` pins the concrete labels), and
#: it is what an unknown-label refusal lists. Swift twin:
#: ``ReadingPosition.declarableLabels``.
DECLARABLE_LABELS = (
    "last token",
    "mean from token <k>",
    "offset from end <k>",
    "last content token",
    "turn close token",
    "post-instruction <i>  (i in 1..5)",
    "content offset <k>",
    "mean content from token <n>",
)


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
    #: A CONTENT-MASKED pool's exact positions, or ``None`` for the contiguous
    #: ``[start_index, end_index)`` window every other position reads. The
    #: window still bounds it (first pooled index → last pooled index + 1), so
    #: every consumer that only understands windows still sees a truthful
    #: span; the recorder reads exactly these positions.
    pooled_indices: tuple[int, ...] | None = None
    #: How many positions of the whole sequence THE MASK called structure —
    #: template markers, the turn's role tag, the trailing generation scaffold
    #: — or ``None`` for a read that masks nothing at all. Zero is meaningful
    #: (under raw rendering every token is content, so the mask excludes
    #: nothing), so ``None`` and ``0`` say different things. Deliberately NOT
    #: "everything not pooled": the tokens a ``meanContentFromToken(n)`` skips
    #: before its n-th content token were excluded by the REQUEST, not by the
    #: mask, and reporting them together would hide how much structure the
    #: template actually contributed.
    masked_token_count: int | None = None

    @property
    def is_pooled(self) -> bool:
        return self.end_index - self.start_index > 1

    @property
    def pooled_token_count(self) -> int:
        """How many positions were actually averaged."""
        if self.pooled_indices is not None:
            return len(self.pooled_indices)
        return self.end_index - self.start_index

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
            raise ReadingPositionError(templated_rendering_refusal(self) or "")

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
class ContentOffset(_TemplateRole):
    """The single token at ``lastContentToken − k``, ``k ≥ 0``.

    The CONTENT-coordinate sibling of :class:`OffsetFromEnd`, and the reason
    it exists: ``offsetFromEnd`` counts from the end of the SEQUENCE, so under
    a generation prompt every one of its first few positions is template
    scaffolding, and the count of scaffold tokens is a fact about the family.
    ``contentOffset(2)`` says "two tokens back into what the stimulus itself
    said", which lands in the same place on every family.

    ``k = 0`` is exactly :class:`LastContentToken` — same index, same value —
    and the recipe identity canonicalizes it as such, so declaring the offset
    form can never split an identity away from an otherwise-identical
    last-content-token recipe (the ``offsetFromEnd(0) ≡ lastToken``
    precedent).

    Underflow REFUSES: asking for 40 tokens back into a 12-token stimulus
    reads nothing meaningful, and clamping to the turn's first token would be
    the quiet substitution the freeze discipline forbids.
    """

    k: int

    def __post_init__(self):
        if self.k < 0:
            raise ReadingPositionError(
                f"contentOffset needs k ≥ 0, got {self.k} — repair: offsets "
                "count BACKWARD from the last content token, so k=0 is the "
                "last content token and k=3 is three before it")

    @property
    def label(self) -> str:
        return f"content offset {self.k}"

    @property
    def identity_mode(self) -> str:
        return "contentOffset"

    @property
    def identity_parameter(self) -> int | None:
        return self.k

    @property
    def _anchor_noun(self) -> str:
        return "content/turn boundary"

    def resolve(self, token_ids, *, tokenizer=None,
                rendering_is_raw: bool = True) -> ResolvedReadingPosition:
        self._require_templated(rendering_is_raw)
        n = len(token_ids)
        content = last_content_index(token_ids, tokenizer, self.label)
        index = content - self.k
        if index < 0:
            raise ReadingPositionError(
                f"reading position '{self.label}' did not resolve: the "
                f"stimulus's content ends at token {content} of {n}, so there "
                f"is no token {self.k} before it — repair: lower k, or drop "
                f"the short stimuli (this position is never clamped, because "
                f"reading the turn's first token instead would silently "
                f"change the recipe)")
        return ResolvedReadingPosition(
            requested=self.label, mode=self.identity_mode, parameter=self.k,
            start_index=index, end_index=index + 1, token_count=n,
            source=f"last content token − {self.k}")


@dataclass(frozen=True)
class MeanContentFromToken(ReadingPosition):
    """Mean over CONTENT token positions from the ``n``-th content token on.

    The mask sibling of :class:`MeanFromToken`. ``meanFromToken(4)`` pools
    every position from index 4 to the sequence end, which under a chat
    template means the pool is part stimulus and part scaffolding — and, with
    a generation prompt, ends INSIDE the template's trailing tokens. This
    position pools the stimulus's own tokens and nothing else, counting ``n``
    in CONTENT coordinates so the same declaration means the same thing under
    every family's template.

    **Under RAW rendering every token is content**, so this resolves exactly as
    ``meanFromToken(n)`` does — same window, same numbers, nothing masked —
    rather than refusing. Refusing would be pedantry: the position is
    perfectly meaningful there, it simply has nothing to exclude. The stamp
    says so (``source``, and a masked count of 0), and the recipe identity
    still distinguishes the two declarations, because an identity records what
    the recipe DECLARED, not what a particular rendering made of it.

    The mask itself is :func:`content_indices` — the same template map the
    named roles resolve against, extended symmetrically with turn-OPEN
    markers. One definition.
    """

    n: int

    def __post_init__(self):
        if self.n < 0:
            raise ReadingPositionError(
                f"meanContentFromToken needs n ≥ 0, got {self.n} — repair: n "
                "counts FORWARD from the first content token, so n=0 pools "
                "the whole content")

    @property
    def label(self) -> str:
        return f"mean content from token {self.n}"

    @property
    def minimum_token_count(self) -> int:
        # The honest bound under RAW rendering, and a necessary (not
        # sufficient) one under a template, where the real requirement is
        # "the content is longer than n" — enforced in `resolve`, which
        # refuses rather than clamps.
        return max(1, self.n + 1)

    @property
    def requested_start_index(self) -> int | None:
        # DELIBERATELY None. `n` counts in CONTENT coordinates, and the
        # neutral token bank's "bank from raw position N onward" question has
        # no answer in those coordinates — answering with n would bank rows
        # from the wrong place under any template. Banking from 0 (what None
        # means) is the honest superset.
        return None

    @property
    def identity_mode(self) -> str:
        return "meanContentFromToken"

    @property
    def identity_parameter(self) -> int | None:
        return self.n

    def resolve(self, token_ids, *, tokenizer=None,
                rendering_is_raw: bool = True) -> ResolvedReadingPosition:
        length = len(token_ids)
        if rendering_is_raw:
            _require_length(length, self.minimum_token_count, self.label)
            # The MeanFromToken clamp, preserved exactly: under raw the two
            # positions must read identical rows, or "equivalent to
            # meanFromToken(n)" would be a claim the numbers contradict.
            start = min(max(0, self.n), length - 1)
            return ResolvedReadingPosition(
                requested=self.label, mode=self.identity_mode,
                parameter=self.n, start_index=start, end_index=length,
                token_count=length,
                source=(f"token {self.n} through sequence end "
                        "(raw rendering: every token is content)"),
                pooled_indices=None, masked_token_count=0)
        content = content_indices(token_ids, tokenizer, self.label)
        if len(content) <= self.n:
            raise ReadingPositionError(
                f"reading position '{self.label}' did not resolve: the "
                f"rendered turn carries {len(content)} content tokens, so "
                f"there is no content token {self.n} to pool from — repair: "
                f"lower n, or drop the short stimuli (this position is never "
                f"clamped, because pooling from the turn's first token "
                f"instead would silently change the recipe)")
        pooled = tuple(content[self.n:])
        start, end = pooled[0], pooled[-1] + 1
        return ResolvedReadingPosition(
            requested=self.label, mode=self.identity_mode, parameter=self.n,
            start_index=start, end_index=end, token_count=length,
            source=f"content token {self.n} through the last content token",
            pooled_indices=pooled,
            masked_token_count=length - len(content))


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
    ids = _ids_for_tokens(tokenizer, END_OF_TURN_TOKENS)
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int) and eos >= 0:
        ids.add(eos)
    return ids


#: Turn-OPEN markers, the symmetric half of :data:`END_OF_TURN_TOKENS` and the
#: other half of the one template map. Probed by NAME in the same way and the
#: same order on both engines (Swift twin:
#: ``ReadingPosition.turnOpenTokens``). Llama 3 contributes both header
#: markers: the role tag lives BETWEEN them, so both are structure.
TURN_OPEN_TOKENS = (
    "<start_of_turn>",      # Gemma 3
    "<|im_start|>",         # Qwen / ChatML
    "<|start_header_id|>",  # Llama 3
    "<|end_header_id|>",    # Llama 3
    "<|user|>",             # Phi
    "<|assistant|>",        # Phi
)

#: Sequence-level markers that are structure wherever they appear (Swift twin:
#: ``ReadingPosition.structuralTokens``).
STRUCTURAL_TOKENS = (
    "<bos>",                # Gemma 3
    "<|begin_of_text|>",    # Llama 3
    "<s>",                  # Llama 2 / Mistral
    "<|endoftext|>",        # GPT-2 / Qwen padding
)

#: How far past a turn-open marker the ROLE TAG may run before the mask gives
#: up looking for its newline. Every vendored family writes
#: ``<open><role>\n`` in at most three tokens; the cap keeps a tokenizer that
#: never emits a newline token from eating the stimulus.
ROLE_TAG_MAX_TOKENS = 4


def _ids_for_tokens(tokenizer, names) -> set[int]:
    """Ids for a name-probed marker list — the one lookup both marker
    inventories use."""
    ids: set[int] = set()
    unk = getattr(tokenizer, "unk_token_id", None)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return ids
    for token in names:
        try:
            value = convert(token)
        except Exception:  # pragma: no cover - defensive
            continue
        if isinstance(value, int) and value >= 0 and value != unk:
            ids.add(value)
    return ids


def turn_open_ids(tokenizer) -> set[int]:
    """Ids of this tokenizer's turn-open markers (see
    :data:`TURN_OPEN_TOKENS`)."""
    return _ids_for_tokens(tokenizer, TURN_OPEN_TOKENS)


def structural_ids(tokenizer) -> set[int]:
    """Every id THE TEMPLATE MAP calls structure: turn-open markers, turn-close
    markers (including the tokenizer's declared eos), and the sequence-level
    markers. This is the same inventory the named roles resolve against, read
    as a set instead of scanned for a boundary — one definition, two uses."""
    return (turn_open_ids(tokenizer) | end_of_turn_ids(tokenizer)
            | _ids_for_tokens(tokenizer, STRUCTURAL_TOKENS))


def _token_piece(tokenizer, token_id: int) -> str:
    """The vocabulary PIECE for one id, with the two BPE surface conventions
    normalized (``Ċ`` is a newline, ``Ġ``/``▁`` are spaces). Used only to find
    the newline that ends a template's role tag; a tokenizer that answers
    nothing yields "", and the caller then leaves the tag unmasked rather than
    guessing."""
    for attribute, call in (("convert_ids_to_tokens", lambda f: f(token_id)),
                            ("decode", lambda f: f([token_id]))):
        function = getattr(tokenizer, attribute, None)
        if not callable(function):
            continue
        try:
            piece = call(function)
        except Exception:  # pragma: no cover - defensive
            continue
        if isinstance(piece, str):
            return piece.replace("Ċ", "\n").replace("Ġ", " ").replace("▁", " ")
    return ""


def content_indices(token_ids, tokenizer, label: str) -> list[int]:
    """The positions of the stimulus's OWN CONTENT in a rendered sequence.

    THE MASK, defined once and used by every content-coordinate position. It
    derives entirely from the template map the named roles already use:

    1. the read turn CLOSES at :func:`turn_close_index` — the last end-of-turn
       marker — and content cannot follow it, which is what excludes the
       trailing generation scaffold (on Gemma 3 that scaffold ends in a bare
       newline, an ordinary token no special-token test would catch);
    2. the read turn OPENS at the last :data:`TURN_OPEN_TOKENS` marker before
       that close (position 0 when a template writes none);
    3. the template's ROLE TAG — ``<start_of_turn>`` ``model`` ``\\n``, or
       ChatML's ``<|im_start|>assistant\\n`` — is structure, and every vendored
       family terminates it with a newline, so the tokens up to and including
       the first newline piece after the open marker are dropped (bounded by
       :data:`ROLE_TAG_MAX_TOKENS`; a tokenizer that reports no newline keeps
       its tag rather than losing stimulus tokens to a guess);
    4. anything still inside that the map calls structure
       (:func:`structural_ids`) is dropped.

    Returns the surviving positions in order. Refuses (typed, with a repair)
    when nothing survives, because a mean over no positions is not a reading.
    """
    close = turn_close_index(token_ids, tokenizer, label)
    if close == 0:
        raise ReadingPositionError(
            f"reading position '{label}' did not resolve: the turn closes at "
            f"token 0, so the stimulus contributed no content — repair: check "
            f"the stimulus is non-empty and that the rendering is the one you "
            f"meant")
    opens = turn_open_ids(tokenizer)
    start = 0
    for index in range(close - 1, -1, -1):
        if int(token_ids[index]) in opens:
            start = index + 1
            # The role tag runs from here to its newline, inclusive.
            for offset in range(min(ROLE_TAG_MAX_TOKENS, close - start)):
                if "\n" in _token_piece(tokenizer, int(token_ids[start + offset])):
                    start = start + offset + 1
                    break
            break
    structural = structural_ids(tokenizer)
    content = [i for i in range(start, close)
               if int(token_ids[i]) not in structural]
    if not content:
        raise ReadingPositionError(
            f"reading position '{label}' did not resolve: the rendered turn "
            f"({len(token_ids)} tokens) carries no content tokens — every "
            f"position between the turn's opening and its close is template "
            f"structure — repair: check the stimulus is non-empty and that "
            f"the rendering is the one you meant")
    return content


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


def templated_rendering_refusal(position) -> str | None:
    """The refusal for a template-aware role under RAW rendering, or ``None``
    when the position has no such dependency.

    ONE sentence, TWO moments. It was written for resolution time, where a
    role that needs a chat turn meets a raw stimulus; since 2026-08-25 the
    ATTACH path asks the same question at DECLARATION time — a pin that could
    never resolve is answered while the person is typing, not hours later on a
    GPU (the ``addGenerationPrompt: false`` precedent). Hoisting it here is
    what keeps the two moments from drifting into two explanations. Swift
    twin: ``ReadingPosition.templatedRenderingRefusal``.
    """
    if not position.requires_templated_rendering:
        return None
    anchor = getattr(position, "_anchor_noun", "content/turn boundary")
    return (
        f"reading position '{position.label}' needs templated rendering: "
        f"a raw stimulus has no chat turn, so there is no "
        f"{anchor} to read at — repair: re-attach the "
        f"concept with {RENDERING_FLAG} "
        f'\'{{"mode": "chatTemplate"}}\', or choose a '
        f"rendering-independent position "
        f"('last token', 'offset from end k', 'mean from token k')")


def declaration_conflict(declaration, pool_from_token) -> str | None:
    """The refusal for declaring a reading position in BOTH spellings, or
    ``None``.

    ``--pool-from K`` is exactly ``--reading-position 'mean from token K'``;
    accepting both would mean silently picking one, and which one it picked
    would be the recipe. Twin text on both engines (Swift:
    ``ReadingPosition.declarationConflict``).
    """
    if declaration is None or pool_from_token is None:
        return None
    return (
        f"reading position declared twice: {DECLARATION_FLAG} "
        f"'{declaration}' and {POOL_FROM_FLAG} {pool_from_token} name two "
        f"recipes, and a concept pins exactly one — repair: drop "
        f"{POOL_FROM_FLAG} ({DECLARATION_FLAG} 'mean from token "
        f"{pool_from_token}' is its spelling), or drop {DECLARATION_FLAG}")


def parse_declaration(value) -> "ReadingPosition | None":
    """Parse a reading-position DECLARATION — a CLI flag value or a route body
    field — into the position a manifest should pin.

    ``None`` (the flag absent) means the manifest keeps its default, which is
    what every study written before this writer existed pinned: nothing new is
    written, and the recipe identity does not move. An identity moves ONLY for
    an explicitly declared non-default position, which is correct — the
    reading position IS part of the recipe.

    STRICT: an unrecognized label is a typed refusal naming the vocabulary,
    never a fall back to last-token. :func:`from_label`'s tolerance exists for
    READING old artifacts; a WRITER that guessed would pin a recipe nobody
    asked for. Swift twin: ``ReadingPosition.declared(_:)``.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ReadingPositionError(
            f"{DECLARATION_FLAG} was given no value — repair: name a reading "
            f"position, e.g. {DECLARATION_FLAG} 'last content token' — or "
            f"omit the flag, which keeps the recipe's default")
    position = parse_label_strict(value)
    if position is None:
        raise ReadingPositionError(
            f"reading position '{value.strip()}' is not one the {ENGINE} "
            f"engine knows — repair: declare one of "
            f"{'; '.join(DECLARABLE_LABELS)}")
    return position


def mean_from_token(k: int) -> MeanFromToken:
    return MeanFromToken(k=k)


def offset_from_end(k: int) -> OffsetFromEnd:
    return OffsetFromEnd(k=k)


def post_instruction(i: int) -> PostInstruction:
    return PostInstruction(i=i)


def content_offset(k: int) -> ContentOffset:
    return ContentOffset(k=k)


def mean_content_from_token(n: int) -> MeanContentFromToken:
    return MeanContentFromToken(n=n)


LAST_TOKEN = LastToken()
LAST_CONTENT_TOKEN = LastContentToken()
TURN_CLOSE_TOKEN = TurnCloseToken()

#: Prefix → constructor. Order matters only for readability: no prefix here is
#: a prefix of another ("mean content from token " and "mean from token "
#: diverge at their second word).
_PARAMETERIZED = (
    ("mean content from token ", mean_content_from_token),
    ("mean from token ", mean_from_token),
    ("offset from end ", offset_from_end),
    ("post-instruction ", post_instruction),
    ("content offset ", content_offset),
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
