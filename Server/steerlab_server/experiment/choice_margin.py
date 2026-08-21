"""Distance-from-boundary diagnostics for the answer-token instrument (D3).

The logprob arm of one categorical case family was previously described as
showing a "ceiling effect" or "saturation". That was the wrong word, and the
wrong mental model.

What was actually observed was a large **joint-logprob margin**: the gap
between the winning option and the runner-up. A large margin means the item
sits far from the decision boundary, so the *flip rate* — how often the argmax
changes — has poor sensitivity: an intervention can move the log-odds a long
way without ever flipping a single item. But the log-odds itself keeps moving
continuously, and remains a perfectly good continuous readout. Reporting "the
instrument saturated" invites the conclusion that the measurement failed, when
the correct conclusion is that the *flip rate* was the wrong statistic to read.

TRUE numerical saturation is a different thing and does occur: `log_odds`
clamps probabilities away from 0 and 1 by ``eps`` before taking the ratio, so
a degenerate softmax yields a clamp ARTEFACT rather than a measurement. That
is worth counting separately, and is counted here.

**Every threshold below is a declared, versioned DIAGNOSTIC band.** None of
them gates anything. A silent numeric cutoff that decides which items "count"
is exactly the kind of unrecorded analytic choice the firewall exists to
prevent, so the band edges travel with the report and carry a version string.

Cross-engine twin: ``Sources/ExperimentKit/ChoiceMarginDiagnostics.swift``.
"""

from __future__ import annotations

import math

# Versioned so a report says which band edges produced its counts. Bumping
# these MUST bump the version — a comparison across two versions of the bands
# is not a comparison.
BANDS_VERSION = "d3-2026-07-26"

#: Joint-logprob margin bands, in nats. Diagnostic only.
MARGIN_BANDS = (1.0, 2.5, 5.0, 10.0)

#: The clamp `ChoiceResult.log_odds` applies before the ratio. Must match the
#: instrument's own epsilon, or clamp incidence is counted against the wrong
#: boundary.
LOG_ODDS_EPSILON = 1e-12


def margin(option_logprobs: dict) -> float | None:
    """Gap between the best and second-best option, in nats.

    None when fewer than two options were scored: with one option there is no
    boundary, so there is no distance from it."""
    totals = sorted((v for v in option_logprobs.values()
                     if isinstance(v, (int, float)) and math.isfinite(v)),
                    reverse=True)
    if len(totals) < 2:
        return None
    return totals[0] - totals[1]


def is_clamped(option_logprobs: dict, eps: float = LOG_ODDS_EPSILON) -> bool:
    """True when the softmax over these logprobs is degenerate enough that
    `log_odds` returns a clamp artefact rather than a measured value."""
    totals = [v for v in option_logprobs.values()
              if isinstance(v, (int, float)) and math.isfinite(v)]
    if len(totals) < 2:
        return False
    peak = max(totals)
    weights = [math.exp(v - peak) for v in totals]
    z = sum(weights)
    if z <= 0:
        return True
    for w in weights:
        p = w / z
        if p <= eps or (1.0 - p) <= eps:
            return True
    return False


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default 'linear' method), so the
    two engines agree on the median of an even-length sample."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[int(position)]
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def diagnostics(records: list[dict]) -> dict:
    """Margin distribution, band counts and clamp incidence over choice
    records (each carrying ``optionLogprobs``)."""
    margins: list[float] = []
    clamped = 0
    scored = 0
    for record in records:
        logprobs = record.get("optionLogprobs") or {}
        value = margin(logprobs)
        if value is None:
            continue
        scored += 1
        margins.append(value)
        if is_clamped(logprobs):
            clamped += 1

    if not margins:
        return {"bandsVersion": BANDS_VERSION, "scoredItems": 0,
                "marginBands": list(MARGIN_BANDS), "clampEpsilon": LOG_ODDS_EPSILON}

    ordered = sorted(margins)
    above = {f"above{band:g}Nats": sum(1 for m in ordered if m > band)
             for band in MARGIN_BANDS}
    return {
        "bandsVersion": BANDS_VERSION,
        "scoredItems": scored,
        "marginBands": list(MARGIN_BANDS),
        "clampEpsilon": LOG_ODDS_EPSILON,
        "marginMin": ordered[0],
        "marginP25": _quantile(ordered, 0.25),
        "marginMedian": _quantile(ordered, 0.5),
        "marginP75": _quantile(ordered, 0.75),
        "marginMax": ordered[-1],
        "marginMean": sum(ordered) / len(ordered),
        # How many items sit further than each band from the boundary.
        "itemsBeyondBand": above,
        # TRUE saturation: log-odds is a clamp artefact for these items.
        "clampedItems": clamped,
        "interpretation": interpretation(
            scored=scored, ordered=ordered, clamped=clamped),
    }


def interpretation(*, scored: int, ordered: list[float], clamped: int) -> str:
    """Say what the numbers mean, in the words that are actually correct."""
    median = _quantile(ordered, 0.5)
    far = sum(1 for m in ordered if m > MARGIN_BANDS[-1])
    parts = [
        f"median joint-logprob margin {median:.2f} nats over {scored} scored "
        f"item{'' if scored == 1 else 's'}"
    ]
    if far:
        parts.append(
            f"{far} item{'' if far == 1 else 's'} sit more than "
            f"{MARGIN_BANDS[-1]:g} nats from the decision boundary, where the "
            "FLIP RATE has poor sensitivity — an intervention can move the "
            "log-odds a long way without flipping any item. Read the "
            "continuous log-odds shift, not the flip rate; the instrument has "
            "not saturated")
    if clamped:
        parts.append(
            f"{clamped} item{'' if clamped == 1 else 's'} produced a "
            f"probability at the {LOG_ODDS_EPSILON:g} clamp, so their log-odds "
            "is a clamp ARTEFACT rather than a measurement — this is true "
            "numerical saturation and those items carry no usable magnitude")
    return "; ".join(parts)
