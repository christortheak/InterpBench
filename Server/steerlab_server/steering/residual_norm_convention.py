"""The residual-norm DENOMINATOR CONVENTION — the fixed rule by which the
per-layer number that denominates a norm-unit alpha is computed.

Alpha is reported in units of the residual-stream norm at the injection layer
(CLAUDE.md > "Report steering strength in units of the residual-stream
norm..."). That only makes alpha comparable across concepts if every engine
computes the denominator the same way. Until 2026-08-20 the two engines did
NOT: Swift's neutral token bank averaged over every corpus position, while
THIS engine averaged over the BANKED positions only (the deterministic
row-cap draw) — see the standing NOTE that used to sit in
``extractor.neutral_activation_bank``. On a downsampled corpus those are
different numbers, so "alpha = 1 norm units" meant two different doses.

**The ruling (2026-08-20): a token bank's denominator is the WHOLE-CORPUS
average.** Every measured position counts, banked or not — the denominator
describes the corpus, not the draw. This engine was fixed to match; the Swift
twin is ``Sources/SteeringKit/Extraction/ResidualNormConvention.swift``.

The convention is STAMPED, never retro-applied. Every measurement writes a
``residualNormConvention`` stamp naming the rule it applied; artifacts without
a stamp are LEGACY and are read exactly as before — no migration, no
recompute, no warning. ``vectors backfill-norms`` re-measures under the current
convention and stamps it, which is the researcher's opt-in migration path.

**Two rules, two stamps (2026-08-28 audit, F1).** Until this landed there was
one stamp string and two averaging rules behind it. ``ResidualNormTally``
below averages POSITIONS; the ``activations``-based measurement that actually
writes vector sidecars (``extract``/``extract_grand_mean``/``norm_backfill``,
and their Swift twins) averages TEXTS — each capture already carries the mean
norm over its own reading window, and those per-text numbers are then averaged
with equal weight per text. The two coincide wherever every text contributes
one position (``lastToken`` and every other single-position reading) or where
every window has the same length, and they diverge at a POOLED reading
position (``meanFromToken``, ``meanContentFromToken``) over variable-length
texts, where per-text weighting gives a long text's positions less weight each
than a short text's. One string could not say which number an artifact holds,
which is precisely what the "bump the version when the averaging RULE changes"
contract exists to prevent — so the second rule now has its own string and
every writer stamps the rule it applied.
"""

from __future__ import annotations

#: Rule A — the PER-POSITION mean: every measured position in the corpus counts
#: once, banked or not (:class:`ResidualNormTally` is the only implementation
#: on either engine). Pinned cross-engine contract: same JSON key
#: (``residualNormConvention``), same value string. Bump the version suffix
#: only when this averaging RULE changes — never for an unrelated sidecar edit.
WHOLE_CORPUS_MEAN = "wholeCorpusMean-v1"

#: Rule B — the MEAN OF PER-TEXT WINDOW-MEANS: each text contributes exactly
#: one number per layer, the mean residual-stream L2 norm over that text's
#: reading window (``recorder.Capture.residual_norm``), and those per-text
#: numbers are averaged with equal weight per TEXT. This is what
#: ``extractor.activations`` measures and therefore what ``extract``,
#: ``extract_grand_mean`` and ``norm_backfill`` write into vector sidecars, on
#: both engines.
#:
#: Equal to :data:`WHOLE_CORPUS_MEAN` when every text's reading window is a
#: single position (``lastToken`` and friends) or when every window is the same
#: length; different at pooled readings over variable-length texts.
#:
#: One documented sub-case: the paired ``extract`` fallback with NO neutral
#: corpus averages the positive and negative CLASS means (``(p + n) / 2``),
#: which is this rule exactly when the two classes hold the same number of
#: texts — the paired-stimulus contract — and class-balanced rather than
#: text-flat when they do not. The class-balanced reading is the intended one
#: for a contrastive pair; the stamp is honest either way because both are
#: means of per-text window-means.
PER_TEXT_MEAN = "perTextMean-v1"

#: GRANDFATHERING (reader-side, documented, never a rewrite). Artifacts on disk
#: carry ``wholeCorpusMean-v1`` written by the per-text writers before the two
#: rules were separated. Frozen bytes are never rewritten, so those stamps stay
#: exactly as they are and keep meaning exactly what they always meant: the
#: number the per-text rule produced. A reader comparing doses across such an
#: artifact and a freshly stamped ``perTextMean-v1`` one may treat the two as
#: the same rule — they are, and at a single-position reading (which is this
#: engine's default and what the whole-corpus rule would have produced anyway)
#: the numbers are identical. What a reader may NOT do is credit a legacy
#: ``wholeCorpusMean-v1`` with the per-position rule: no writer has ever
#: produced a sidecar under it. Nothing in either engine GATES on the stamp —
#: it is provenance and display only (``display_label``, the α-default
#: convention note, the OptVec packaging advisory) — so a new-stamp and an
#: old-stamp artifact cannot refuse against each other.
GRANDFATHERED_PER_TEXT_STAMP = WHOLE_CORPUS_MEAN

#: Human-readable label for an artifact that predates the stamp. Surfaces that
#: already show norm provenance may render this; nothing may treat it as
#: evidence of a particular rule, because a pre-stamp artifact's rule is
#: genuinely unknown (it depends on which engine wrote it and whether its
#: corpus was downsampled).
LEGACY_LABEL = "legacy (pre-stamp)"


def display_label(residual_norm_per_layer, stamp: str | None) -> str | None:
    """The convention label to display for a sidecar's stamp.

    ``None`` means "no norms recorded at all" — there is no denominator to
    describe. Swift twin: ``ResidualNormConvention.displayLabel``.
    """
    if not residual_norm_per_layer:
        return None
    return stamp or LEGACY_LABEL


def table_length_problem(residual_norm_per_layer, *, layer_count: int,
                         artifact: str) -> str | None:
    """LOAD-TIME gate on the shape of a denominator table (2026-08-28 audit,
    F7/F13). Returns the refusal prose, or ``None`` when the table is usable.

    A denominator table must cover every layer of the artifact it travels
    with, or none of them. **Absent (or empty) is legal and stays legal**: the
    OptVec, J-lens and Gemma-Scope-report families are born with no norms at
    all and acquire them through ``vectors backfill-norms``, so an absent
    table is a state the writers deliberately produce and the per-verb
    refusals already name. A table that is present but SHORTER than the
    artifact's depth is the state no writer produces — extraction measures one
    norm per layer, backfill writes exactly ``layer_count`` of them, and the
    SAE by-id import slices the donor to its layer count — so a short table is
    a malformed or hand-edited artifact, and every verb that reads it with a
    layer index would otherwise dose the layers past the end with some other
    layer's number.

    Swift twin: ``ResidualNormConvention.tableLengthProblem`` (byte-identical
    prose, pinned by ``ResidualNormConventionTests`` /
    ``test_residual_norm_convention.py``).
    """
    norms = list(residual_norm_per_layer or [])
    if not norms or len(norms) == int(layer_count):
        return None
    return (f"vector artifact '{artifact}' carries {len(norms)} residual "
            f"norms for {int(layer_count)} layers — a denominator table must "
            f"cover every layer or none, and a short one silently doses the "
            f"layers it does not reach with another layer's number; "
            f"re-measure the norms (vectors backfill-norms), or re-extract "
            f"the concept")


def residual_norm_problem(residual_norm_per_layer, layer: int, *,
                          artifact: str) -> str | None:
    """USE-SITE gate — the ONE out-of-range rule every verb applies (condition,
    sweep, variant), on both engines. Returns the refusal prose, or ``None``
    when the layer has a denominator.

    Before this landed the same truncated table produced four different
    outcomes for one artifact: the condition path substituted ``0.0`` and
    refused as ``degenerateData``, the sweep and variant paths clamped to the
    last entry and dosed the deepest layers with a shallower layer's number,
    and the Swift condition path clamped as well — while an EMPTY table
    indexed ``[-1]`` on Swift and crashed. One artifact, four behaviours, and
    three of them silent. The rule is now the house one everywhere: refuse,
    and say which layer had no denominator.

    Swift twin: ``ResidualNormConvention.residualNormProblem`` (byte-identical
    prose).
    """
    norms = list(residual_norm_per_layer or [])
    if 0 <= int(layer) < len(norms):
        return None
    return (f"'{artifact}' has no residual norm at layer {int(layer)} — its "
            f"denominator table covers {len(norms)} layer(s), so an α in "
            f"residual-norm units cannot be denominated there; re-measure the "
            f"norms (vectors backfill-norms), or switch α to raw units")


class ResidualNormTally:
    """Per-layer running mean of residual norms under :data:`WHOLE_CORPUS_MEAN`
    — the PER-POSITION rule, and the only implementation of it.

    Both the banked rows and the rows the token-bank cap EXCLUDED are added
    here; :meth:`mean` is then the corpus mean, not the draw's mean. It feeds
    the neutral token bank (and through it the neutral-PC basis) and nothing
    else; a denominator written into a VECTOR SIDECAR comes from
    ``extractor.activations`` and is stamped :data:`PER_TEXT_MEAN`. The Swift
    ``ResidualNormConvention.Tally`` is the line-for-line twin, and the two are
    pinned against one shared fixture (``test_residual_norm_convention.py`` /
    ``ResidualNormConventionTests``).
    """

    __slots__ = ("_sums", "_counts")

    def __init__(self) -> None:
        self._sums: dict[int, float] = {}
        self._counts: dict[int, int] = {}

    def add(self, layer: int, norm: float) -> None:
        """One measured position at one layer. Call for EVERY position the
        forward pass produced — dropping the unbanked ones is exactly the bug
        this convention closes."""
        self._sums[layer] = self._sums.get(layer, 0.0) + float(norm)
        self._counts[layer] = self._counts.get(layer, 0) + 1

    @property
    def layers(self) -> list[int]:
        return sorted(self._counts)

    def count(self, layer: int) -> int:
        return self._counts.get(layer, 0)

    @property
    def total_count(self) -> int:
        """Total measured positions across every layer — the bank's
        ``token_row_count`` (positions counted, not rows retained)."""
        return sum(self._counts.values())

    def mean(self, layer: int) -> float:
        """The whole-corpus mean at ``layer``; 0.0 when nothing was measured
        there (an unmeasured layer has no denominator, and 0 is refused
        downstream by the ``residual_norm > 0`` guards rather than silently
        steering)."""
        count = self._counts.get(layer, 0)
        if count <= 0:
            return 0.0
        return self._sums.get(layer, 0.0) / count
