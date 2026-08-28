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

**The ruling (2026-08-20): the convention is the WHOLE-CORPUS average.** Every
measured position counts toward the denominator, banked or not — the
denominator describes the corpus, not the draw. This engine was fixed to
match; the Swift twin is
``Sources/SteeringKit/Extraction/ResidualNormConvention.swift``.

The convention is STAMPED, never retro-applied. New measurements write
``residualNormConvention: "wholeCorpusMean-v1"`` into the sidecar; artifacts
without the stamp are LEGACY and are read exactly as before — no migration, no
recompute, no warning. ``vectors backfill-norms`` re-measures under the current
convention and stamps it, which is the researcher's opt-in migration path.
"""

from __future__ import annotations

#: The stamp written by every fresh measurement on BOTH engines. Pinned
#: cross-engine contract: same JSON key (``residualNormConvention``), same
#: value string. Bump the version suffix only when the averaging RULE changes
#: — never for an unrelated sidecar edit.
CURRENT = "wholeCorpusMean-v1"

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
    """Per-layer running mean of residual norms under the whole-corpus rule.

    Both the banked rows and the rows the token-bank cap EXCLUDED are added
    here; :meth:`mean` is then the corpus mean, not the draw's mean. The Swift
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
