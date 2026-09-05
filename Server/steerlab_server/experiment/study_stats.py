"""Statistics for reported studies (server twin of Swift ``StudyStatistics``).

Everything the reporting policy in CLAUDE.md requires, implemented pure-Python
and deterministic so both substrates can share fixture tests and a re-run of a
report reproduces byte-identical numbers:

- paired bootstrap CIs on per-item (treatment − same-case-baseline) differences,
- Wilcoxon signed-rank (robustness companion to the bootstrap),
- Benjamini–Hochberg FDR adjustment for Phase-1 screening across concepts,
- Holm step-down adjustment for the pre-registered confirm family,
- dose-monotonicity over an alpha grid (promotion-rule criterion).

The module deliberately avoids numpy/scipy: effect tables must be rebuildable
anywhere the run directory lands, including boxes without the ML stack.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class BootstrapCI:
    n: int
    mean: float
    ci_lower: float
    ci_upper: float
    replicates: int
    seed: int


def paired_bootstrap_ci(diffs: list[float], *, replicates: int = 10_000,
                        alpha: float = 0.05, seed: int = 0) -> BootstrapCI:
    """Percentile bootstrap CI on the mean of paired per-item differences.

    ``diffs`` must already be treatment − baseline for the SAME item — the
    pairing is the caller's job and the reason every study keeps a same-case
    baseline arm. Deterministic for a given seed.
    """
    if not diffs:
        raise ValueError("paired bootstrap needs at least one difference")
    n = len(diffs)
    mean = sum(diffs) / n
    rng = random.Random(seed)
    means = sorted(
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(replicates))
    lower = _percentile(means, alpha / 2)
    upper = _percentile(means, 1 - alpha / 2)
    return BootstrapCI(n=n, mean=mean, ci_lower=lower, ci_upper=upper,
                       replicates=replicates, seed=seed)


def _percentile(ordered: list[float], q: float) -> float:
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def wilcoxon_signed_rank(diffs: list[float]) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Zeros are dropped (standard treatment); ties get average ranks; the p-value
    uses the normal approximation with tie correction and continuity
    correction. Adequate at screening/confirm sample sizes (n ≥ ~10); at
    smaller n the test is reported but the bootstrap CI is the primary
    quantity. Returns (W, p) where W = min(W+, W−); (NaN, NaN) when every
    difference is zero.
    """
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return float("nan"), float("nan")
    by_magnitude = sorted(range(n), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n
    tie_correction = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nonzero[by_magnitude[j + 1]]) == abs(nonzero[by_magnitude[i]]):
            j += 1
        average_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[by_magnitude[k]] = average_rank
        t = j - i + 1
        tie_correction += t ** 3 - t
        i = j + 1
    w_plus = sum(rank for rank, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(rank for rank, d in zip(ranks, nonzero) if d < 0)
    w = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24 - tie_correction / 48
    if variance <= 0:
        return w, float("nan")
    z = (w - mean_w + 0.5) / math.sqrt(variance)  # continuity-corrected
    p = 2 * _normal_cdf(z)
    return w, min(1.0, max(0.0, p))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini–Hochberg adjusted p-values (q-values), input order preserved.
    The Phase-1 screening correction, applied across concepts."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_min = 1.0
    for position in range(m - 1, -1, -1):
        index = order[position]
        candidate = p_values[index] * m / (position + 1)
        running_min = min(running_min, candidate)
        adjusted[index] = min(1.0, running_min)
    return adjusted


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, input order preserved. The confirm-
    phase familywise correction for the pre-registered hypothesis family."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for position, index in enumerate(order):
        candidate = p_values[index] * (m - position)
        running_max = max(running_max, candidate)
        adjusted[index] = min(1.0, running_max)
    return adjusted


def percent_agreement(a: list[str], b: list[str]) -> float:
    """Fraction of positions where two label sequences agree (judge-agreement
    companion to :func:`cohens_kappa`; same input contract)."""
    if len(a) != len(b):
        raise ValueError("percent agreement needs equal-length label sequences")
    if not a:
        raise ValueError("percent agreement needs at least one paired label")
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa between two raters' label sequences (chance-corrected
    agreement for the judge-pair and judge-vs-human reports).

    ``a[i]`` and ``b[i]`` are the two raters' labels for the SAME item — the
    alignment is the caller's job (evaluation aligns on
    ``(promptID, sampleIndex, condition)``). Labels are arbitrary strings (here
    ``baseline | variant | tie``). Deterministic, pure, no dependencies.

    kappa = (p_o − p_e) / (1 − p_e), with p_e from each rater's own marginal
    label distribution. When p_e == 1 both raters are constant on the SAME
    label, which forces p_o == 1: returns 1.0 rather than 0/0.
    """
    if len(a) != len(b):
        raise ValueError("Cohen's kappa needs equal-length label sequences")
    n = len(a)
    if n == 0:
        raise ValueError("Cohen's kappa needs at least one paired label")
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    p_e = sum((a.count(label) / n) * (b.count(label) / n) for label in labels)
    if 1 - p_e == 0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


@dataclass
class DoseResponse:
    spearman_rho: float
    is_monotone: bool


def dose_monotonicity(alphas: list[float], effects: list[float], *,
                      tolerance: float = 0.0) -> DoseResponse:
    """Is the effect monotone in dose? (Promotion-rule criterion.)

    Sorts by alpha, reports Spearman's rho, and checks that consecutive
    effects never move against the overall direction by more than
    ``tolerance × effect range``. A flat or inverted dose-response fails
    promotion no matter how large a single-alpha effect looks.

    Nondecreasing GEOMETRY is not the ACCEPTANCE criterion, and conflating
    the two is what let a flat ladder through (external review, 2026-09-05,
    SCI-04): identical effects satisfy every consecutive step trivially
    (0 ≥ -slack) while carrying no dose information at all, and their
    Spearman rho is undefined. So a ZERO effect range (max == min) is not
    monotone. The refusal is exactly zero variation, not a minimum effect
    size — an increase of 1e-9 still passes, and this helper must not
    quietly become an effect-size gate.

    The mirror image is refused too: a ZERO DOSE range — every alpha the
    same — is not monotone either (owner's ruling, 2026-09-05). With no dose
    variation the (alpha, effect) tie-break sorts the effects by themselves,
    so any "ladder" the step check then sees was manufactured by the sort,
    and rho is undefined for the same zero-rank-variance reason. The Swift
    narrative already refused to build a ladder from fewer than two distinct
    strengths; the promote verb had no such guard, and this is where it now
    lives for both.

    DIRECTION IS OBSERVED, NOT PRESPECIFIED. The sign comes from the sorted
    endpoints (last vs first), so a consistently DOWNWARD ladder is monotone
    here, and the random-floor criterion in ``promotion.decide`` compares
    magnitudes (``abs``) by the same design. A study that wants to reject
    effects running opposite its hypothesis needs a declared expected
    direction, which this helper does not take.

    Input policy, matched on the Swift twin except where noted:

    * fewer than two points → not monotone (rho needs two ranks anyway);
    * unequal input lengths → ``ValueError``. A mismatch is a pairing bug in
      the caller, and ``zip`` would silently truncate and score a ladder
      nobody measured. (``StudyStatistics.doseMonotonicity`` cannot throw
      from this position and answers (NaN, false) instead, the refusal shape
      that file already uses for a mismatched pairing.)
    * any nonfinite alpha or effect → not monotone, rho NaN, decided BEFORE
      the sort. NaN is not orderable, and every comparison against it is
      false, so an unguarded NaN can smuggle a true verdict out;
    * PARTIAL ties in ``alphas`` (some doses repeated, at least two distinct)
      are kept, not merged — deliberately unchanged behaviour, pinned by the
      cross-engine fixture. Pairs sort by (alpha, effect), so rows sharing a
      dose are ordered by their own effect and a repeat never manufactures a
      step violation; Spearman gives the tied doses their average rank, which
      holds |rho| below 1 even for a perfect ladder. ALL doses tied is the
      zero-dose-range refusal above.
    """
    if len(alphas) != len(effects):
        raise ValueError(
            "dose monotonicity needs one effect per alpha (got "
            f"{len(alphas)} alphas and {len(effects)} effects)")
    if not all(math.isfinite(value) for value in alphas) \
            or not all(math.isfinite(value) for value in effects):
        return DoseResponse(spearman_rho=float("nan"), is_monotone=False)
    pairs = sorted(zip(alphas, effects))
    ordered = [effect for _, effect in pairs]
    rho = _spearman([a for a, _ in pairs], ordered)
    if len(ordered) < 2:
        return DoseResponse(spearman_rho=rho, is_monotone=False)
    if pairs[-1][0] == pairs[0][0]:
        # Zero DOSE range (pairs are sorted by alpha, so equal ends mean every
        # alpha is equal): the effects were ordered by the tie-break, not by
        # dose, so the step check below would grade a ladder the sort built.
        # rho is NaN here too — zero rank variance in alpha.
        return DoseResponse(spearman_rho=rho, is_monotone=False)
    effect_range = max(ordered) - min(ordered)
    if effect_range == 0:
        # SCI-04: flat is not monotone. rho is already NaN here (zero rank
        # variance), so the verdict and the correlation agree that there is
        # nothing to see.
        return DoseResponse(spearman_rho=rho, is_monotone=False)
    direction = 1.0 if ordered[-1] >= ordered[0] else -1.0
    slack = tolerance * effect_range
    monotone = all(
        direction * (ordered[i + 1] - ordered[i]) >= -slack
        for i in range(len(ordered) - 1))
    return DoseResponse(spearman_rho=rho, is_monotone=monotone)


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    rank_x = _ranks(xs)
    rank_y = _ranks(ys)
    mean_x = sum(rank_x) / len(rank_x)
    mean_y = sum(rank_y) / len(rank_y)
    var_x = sum((r - mean_x) ** 2 for r in rank_x)
    var_y = sum((r - mean_y) ** 2 for r in rank_y)
    if var_x == 0 or var_y == 0:
        return float("nan")
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y))
    return cov / math.sqrt(var_x * var_y)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


# --- effect-sizes.csv ---------------------------------------------------------

EFFECT_SIZES_HEADER = [
    "condition", "endpoint", "n", "deltaMean", "ciLower", "ciUpper",
    "wilcoxonW", "wilcoxonP", "adjustedP", "correction", "modality",
    "stratifyBy", "stratum", "unit",
    # Appended 2026-08-06 (ADDITIVE — existing readers key by name and are
    # untouched; every historical column keeps its position).
    "estimand", "inference",
]


def _fmt(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.6g}"


@dataclass
class EffectRow:
    condition: str
    endpoint: str
    ci: BootstrapCI
    wilcoxon_w: float
    wilcoxon_p: float
    adjusted_p: float = float("nan")
    correction: str = ""
    # Intervention modality of the condition (RESULTS-ARCHITECTURE design
    # axis): injection | adapter | systemPrompt | stacked; baseline = none;
    # "" = underivable (condition absent from the manifest).
    modality: str = ""
    # Stratified-analysis provenance (2026-08-06, cross-engine column
    # vocabulary). "pooled" rows are the historical all-items rows — their
    # statistics and correction family are unchanged. Stratified rows carry
    # the family ("promptID", a factor key, or "×"-joined crossed keys) in
    # ``stratify_by``, the cell label in ``stratum``, and what one paired
    # difference IS in ``unit``: "item" (per-item, sample-averaged — the
    # pooled semantics restricted to the stratum) or "sample" (per-sample
    # pairs by sampleIndex, used when the stratum is a single item). Pooled
    # rows leave ``unit`` empty — their unit is the run's default (item, or
    # transcript for clustered multi-agent runs; see unit-of-analysis.json).
    stratify_by: str = "pooled"
    stratum: str = ""
    unit: str = ""
    # WHAT QUANTITY THIS ROW ESTIMATES, and what its p-values are licensed for
    # (2026-08-06). ``unit`` says what one difference is; these two say what
    # the row therefore means, because the two are not the same claim:
    #
    #   estimand = "itemLevel"          one difference per ITEM. The mean
    #                                   generalizes over the items in the
    #                                   stratum — the same estimand the pooled
    #                                   rows carry, restricted.
    #   estimand = "withinItemSamples"  one difference per SAMPLE INDEX inside
    #                                   ONE item. This is a prompt-SPECIFIC
    #                                   stochastic quantity: it says how that
    #                                   single item's generations moved, and
    #                                   nothing whatever about any other item.
    #                                   It cannot support a cross-item claim,
    #                                   so it must never be corrected together
    #                                   with — or read as — an item-level row.
    #
    # ``inference`` is the consequence: "corrected" rows carry an ``adjustedP``
    # from a within-family correction; "diagnostic" rows are DELIBERATELY left
    # out of every correction family and carry no adjusted p at all — their
    # raw Wilcoxon p stays as a descriptive locator (which cell moved), not as
    # a test. Pooled rows leave both fields empty, as they leave ``unit``
    # empty: their estimand is the run's declared unit of analysis (item, or
    # transcript for clustered runs; see unit-of-analysis.json).
    estimand: str = ""
    inference: str = ""

    def as_csv_row(self) -> list[str]:
        return [self.condition, self.endpoint, str(self.ci.n),
                _fmt(self.ci.mean), _fmt(self.ci.ci_lower), _fmt(self.ci.ci_upper),
                _fmt(self.wilcoxon_w), _fmt(self.wilcoxon_p),
                _fmt(self.adjusted_p), self.correction, self.modality,
                self.stratify_by, self.stratum, self.unit,
                self.estimand, self.inference]


def effect_row(condition: str, endpoint: str, diffs: list[float], *,
               replicates: int = 10_000, seed: int = 0) -> EffectRow:
    """One paired effect estimate: bootstrap CI + Wilcoxon on the same diffs."""
    ci = paired_bootstrap_ci(diffs, replicates=replicates, seed=seed)
    w, p = wilcoxon_signed_rank(diffs)
    return EffectRow(condition=condition, endpoint=endpoint, ci=ci,
                     wilcoxon_w=w, wilcoxon_p=p)


# --- fingerprints.csv ---------------------------------------------------------

FINGERPRINTS_HEADER = ["condition", "modality", "endpoint", "effect",
                       "ciLow", "ciHigh", "n"]


def fingerprint_csv_rows(rows: list[EffectRow]) -> list[list[str]]:
    """The behavioral-fingerprint table: one row per condition × endpoint with
    the paired effect size — a tidy pivot of the effect rows already computed
    (no new statistics), sorted by endpoint then condition so conditions'
    fingerprints line up for cross-condition (and cross-modality) comparison."""
    ordered = sorted(rows, key=lambda r: (r.endpoint, r.condition))
    return [[row.condition, row.modality, row.endpoint, _fmt(row.ci.mean),
             _fmt(row.ci.ci_lower), _fmt(row.ci.ci_upper), str(row.ci.n)]
            for row in ordered]


def apply_correction(rows: list[EffectRow], *, method: str) -> None:
    """Fill ``adjusted_p`` in place across a family of effect rows.
    ``method``: "bh" (screen, across concepts) or "holm" (confirm family)."""
    p_values = [row.wilcoxon_p for row in rows]
    usable = [not math.isnan(p) for p in p_values]
    dense = [p for p, ok in zip(p_values, usable) if ok]
    adjusted = bh_fdr(dense) if method == "bh" else holm(dense)
    it = iter(adjusted)
    for row, ok in zip(rows, usable):
        row.adjusted_p = next(it) if ok else float("nan")
        row.correction = method
