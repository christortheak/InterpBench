"""Turn a stored trace into an interpretable one.

A raw watchlist score is not a finding. Three things stand between it and one,
and all three are cheap:

* **A control set.** The paper's validated concept score is mean target
  log-probability minus mean control, not raw loading: the raw number carries
  whatever the layer's overall scale happens to be, and differences in that
  scale across layers or conditions look exactly like differences in loading.
* **A mention mask.** A target token present in the stimulus, or already
  emitted, sits near ceiling for reasons that have nothing to do with the
  model's state (plan §8.2). Mentioned steps stay in the trace and are excluded
  from the aggregate rather than silently averaged into it.
* **A null, for the right claim.** The position-permutation null tests
  POSITION-SPECIFIC statistics only, and refuses the permutation-invariant ones
  by name. A shuffled-position null for a band-averaged mean returns the
  observed value on every draw and reports p = 1.0 exactly — which reads like a
  strong negative result and is really a null that cannot find anything. Found
  by running it on a real trace, where nullMean equalled the score to the
  digit. Nulls are computed from the LIVE captured scores and never replay the
  model.

Every function reports the counts behind its number. An aggregate over two
unmentioned steps and one over two hundred are not the same claim, and a
consumer that cannot see which it has will treat them alike.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field, asdict

from .schemas import JLensError

#: The paper-validated form (plan §8.2): mean over targets minus mean over
#: controls, averaged across the measured band. Stamped on every aggregate so a
#: consumer never has to infer which score it is holding.
SCORE_CONVENTION = "meanTargetMinusMeanControl"
RAW_CONVENTION = "meanTargetRaw"


@dataclass
class TokenSet:
    """A pinned target set and its controls.

    Controls are what make the score a difference rather than a level. Absent
    controls is a legal, declared choice — the aggregate then reports the raw
    convention and says so, instead of quietly presenting a level as a
    contrast.
    """

    targets: list[int] = field(default_factory=list)
    controls: list[int] = field(default_factory=list)
    name: str = ""

    def hash(self) -> str:
        payload = f"{self.name}|{sorted(self.targets)}|{sorted(self.controls)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self, watchlist: list[int]) -> None:
        if not self.targets:
            raise JLensError("a token set needs at least one target token")
        overlap = sorted(set(self.targets) & set(self.controls))
        if overlap:
            raise JLensError(
                f"tokens {overlap} are both target and control — the contrast "
                f"would partly cancel itself")
        missing = sorted((set(self.targets) | set(self.controls))
                         - set(watchlist))
        if missing:
            raise JLensError(
                f"token set names {missing}, which the readout never watched — "
                f"the trace contains no scores for them")


def _index(watchlist: list[int]) -> dict[int, int]:
    return {int(t): i for i, t in enumerate(watchlist)}


def concept_score(observation: dict, token_set: TokenSet, watchlist: list[int],
                  *, mention_mask: dict | None = None,
                  use_logit_lens: bool = False) -> dict | None:
    """The aggregate for ONE observation, or None when nothing is usable.

    ``mention_mask`` excludes primed tokens from the aggregate. Returning None
    rather than 0.0 matters: zero is a value, absence is not, and averaging a
    fabricated zero into a condition mean would bias it toward no effect.
    """
    key = "watchedLogitLens" if use_logit_lens else "watched"
    scores = observation.get(key) or []
    if not scores:
        return None
    at = _index(watchlist)

    def usable(token: int) -> bool:
        if token not in at or at[token] >= len(scores):
            return False
        if mention_mask is not None and mention_mask.get(str(token),
                                                        mention_mask.get(token)):
            return False
        return True

    targets = [scores[at[t]] for t in token_set.targets if usable(t)]
    controls = [scores[at[c]] for c in token_set.controls if usable(c)]
    if not targets:
        return None
    raw = sum(targets) / len(targets)
    if token_set.controls and controls:
        return {"score": raw - (sum(controls) / len(controls)),
                "raw": raw, "convention": SCORE_CONVENTION,
                "targetCount": len(targets), "controlCount": len(controls)}
    # Declared-absent controls, or every control primed out: report the level
    # and label it a level.
    return {"score": raw, "raw": raw, "convention": RAW_CONVENTION,
            "targetCount": len(targets), "controlCount": len(controls)}


def aggregate_over_band(trace_row: dict, token_set: TokenSet,
                        watchlist: list[int], *, band: list[int] | None = None,
                        use_logit_lens: bool = False) -> dict:
    """Average the per-observation score across the measured band.

    ``band`` restricts to layers whose readouts are trusted (the paper's
    workspace band). Absent = every armed layer, which is honest for
    exploration and should be declared for evidence.
    """
    mask = trace_row.get("mentionMask") or {}
    per_layer: dict[int, list[float]] = {}
    excluded = 0
    for obs in trace_row.get("observations") or []:
        layer = obs.get("layer")
        if band is not None and layer not in band:
            continue
        step_mask = mask.get(str(obs.get("predictedIndex")))
        scored = concept_score(obs, token_set, watchlist,
                               mention_mask=step_mask,
                               use_logit_lens=use_logit_lens)
        if scored is None:
            excluded += 1
            continue
        per_layer.setdefault(layer, []).append(scored["score"])
    layer_means = {l: sum(v) / len(v) for l, v in per_layer.items() if v}
    values = list(layer_means.values())
    return {
        "score": (sum(values) / len(values)) if values else None,
        "perLayer": layer_means,
        "band": sorted(band) if band is not None else sorted(layer_means),
        "convention": (SCORE_CONVENTION if token_set.controls
                       else RAW_CONVENTION),
        "tokenSetHash": token_set.hash(),
        # Counts travel with the number: an aggregate over two steps and one
        # over two hundred are different claims.
        "scoredObservations": sum(len(v) for v in per_layer.values()),
        "excludedObservations": excluded,
        "usedLogitLens": use_logit_lens,
    }


def step_series(trace_row: dict, token_set: TokenSet, watchlist: list[int],
                *, band: list[int] | None = None,
                use_logit_lens: bool = False) -> dict[int, float]:
    """``{predictedIndex: band-averaged score}`` — the position-resolved series.

    This is what a position-specific claim is about, and what the permutation
    null below operates on.
    """
    mask = trace_row.get("mentionMask") or {}
    per_step: dict[int, list[float]] = {}
    for obs in trace_row.get("observations") or []:
        if band is not None and obs.get("layer") not in band:
            continue
        index = obs.get("predictedIndex")
        scored = concept_score(obs, token_set, watchlist,
                               mention_mask=mask.get(str(index)),
                               use_logit_lens=use_logit_lens)
        if scored is not None:
            per_step.setdefault(index, []).append(scored["score"])
    return {i: sum(v) / len(v) for i, v in per_step.items() if v}


#: Statistics a position-permutation null can actually speak to. Each depends on
#: WHICH step holds which score, so shuffling the assignment moves it.
#:
#: The bar is higher than "sounds positional". ``max`` sounds positional and is
#: not: the largest value in a set does not change when you shuffle which
#: position holds it. Neither does the mean, or the sum. Anything that reduces
#: the series to an order-independent summary is invariant, and a
#: position-permutation null for it reports p = 1.0 while looking like a real
#: negative result.
STATISTICS = {
    # Front-loaded vs late: does the model load the concept early and release
    # it, or build toward it?
    "earlyMinusLate": lambda series: _early_minus_late(series),
    # Do adjacent steps resemble each other? Structure over the sequence, which
    # is what the paper's top-1 autocorrelation metric is getting at.
    "lag1Autocorrelation": lambda series: _lag1(series),
}

#: Refused by name. Every one of these is a summary of the VALUES alone, so a
#: position-shuffled null cannot test it. ``peak``/``max`` are listed because
#: they are the tempting mistake — this module made it once (see the module
#: docstring) and then made it again by defaulting to it.
PERMUTATION_INVARIANT = ("mean", "meanOverPositions", "sum", "peak", "max",
                         "min", "median", "range")


def _lag1(series: dict[int, float]) -> float:
    """Lag-1 autocorrelation over the step order, or 0.0 when undefined."""
    order = sorted(series)
    values = [series[i] for i in order]
    n = len(values)
    if n < 3:
        return 0.0
    mean = sum(values) / n
    centered = [v - mean for v in values]
    denom = sum(c * c for c in centered)
    if denom <= 0:
        return 0.0
    return sum(centered[i] * centered[i + 1] for i in range(n - 1)) / denom


def _early_minus_late(series: dict[int, float]) -> float:
    order = sorted(series)
    half = len(order) // 2 or 1
    early = [series[i] for i in order[:half]]
    late = [series[i] for i in order[half:]] or early
    return sum(early) / len(early) - sum(late) / len(late)


def permutation_null(trace_row: dict, token_set: TokenSet,
                     watchlist: list[int], *, band: list[int] | None = None,
                     statistic: str = "earlyMinusLate", permutations: int = 200,
                     seed: int | None = None,
                     use_logit_lens: bool = False) -> dict:
    """A position-shuffled null for a POSITION-SPECIFIC statistic.

    Scoped deliberately, because the obvious version of this is vacuous. A
    position-permutation null shuffles which step each score is attributed to,
    so it can only test a statistic that depends on that assignment. The
    band-averaged mean does not: averaging is permutation-invariant, so a
    "null" for it returns the observed value every time and p = 1.0 exactly.
    That is not a null finding, it is a null that cannot find anything — and
    reporting it as p = 1.0 would look like a real, strong negative result.

    So a permutation-invariant statistic is REFUSED by name, and the supported
    ones (:data:`STATISTICS`) are the position-sensitive ones the plan scoped
    this to.

    Never replays the model: the draws come from the live captured scores, so
    the null cannot invent activations the model never had — the same objection
    that makes post-hoc replay a non-goal for the whole feature.

    Deterministically seeded from the trace's own content when no seed is
    given, so the same trace yields the same null anywhere without a seeds
    table.
    """
    if permutations <= 0:
        raise JLensError("permutation_null needs at least one permutation")
    if statistic in PERMUTATION_INVARIANT:
        raise JLensError(
            f"'{statistic}' is a summary of the VALUES alone, so it is "
            f"invariant under position permutation and a "
            f"position-shuffled null cannot test it — every draw would equal "
            f"the observed value and report p = 1.0, which reads like a strong "
            f"negative result and is not one. Use a position-specific "
            f"statistic ({sorted(STATISTICS)}) or a different null (a paired "
            f"comparison against the same item's unsteered condition is the "
            f"right control for a level).")
    if statistic not in STATISTICS:
        raise JLensError(
            f"unknown statistic '{statistic}' — supported: {sorted(STATISTICS)}")

    series = step_series(trace_row, token_set, watchlist, band=band,
                         use_logit_lens=use_logit_lens)
    if len(series) < 2:
        return {"statistic": statistic, "observed": None, "null": None,
                "reason": f"{len(series)} scorable step(s); a position null "
                          f"needs at least 2"}
    observed = STATISTICS[statistic](series)

    if seed is None:
        material = (f"{token_set.hash()}|{trace_row.get('run')}|"
                    f"{trace_row.get('condition')}|{trace_row.get('promptID')}|"
                    f"{trace_row.get('sampleIndex')}|{statistic}|{permutations}")
        seed = int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)

    order = sorted(series)
    values = [series[i] for i in order]
    draws: list[float] = []
    for _ in range(permutations):
        shuffled = list(values)
        rng.shuffle(shuffled)
        draws.append(STATISTICS[statistic](dict(zip(order, shuffled))))

    draws.sort()
    at_or_above = sum(1 for d in draws if d >= observed)
    return {
        "statistic": statistic,
        "observed": observed,
        "steps": len(series),
        "band": sorted(band) if band is not None else None,
        "tokenSetHash": token_set.hash(),
        "usedLogitLens": use_logit_lens,
        "null": {
            # (k+1)/(n+1): an empirical null of n draws cannot license a claim
            # stronger than 1/(n+1).
            "p": (at_or_above + 1) / (len(draws) + 1),
            "permutations": len(draws),
            "seed": seed,
            "mean": sum(draws) / len(draws),
            "median": draws[len(draws) // 2],
            "convention": "positionPermutationWithinBand",
            "replayed": False,
        },
    }


def token_set_from_block(block: dict) -> TokenSet:
    """Read a manifest's declared target/control sets."""
    return TokenSet(
        targets=[int(x) for x in (block.get("targets") or [])],
        controls=[int(x) for x in (block.get("controls") or [])],
        name=str(block.get("name") or ""))
