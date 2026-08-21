"""Per-scenario validation diagnostics (D1).

``scenario_accuracy`` and ``scenario_accuracy_grand_mean`` compute every
projection and the class means that define the midpoint, then throw all of it
away and return one number. When nine virtues all score near chance, that
number cannot distinguish the two explanations a researcher actually needs to
tell apart:

- the direction does not read the concept at all, or
- it ranks scenarios correctly but the MIDPOINT sits in the wrong place.

The second is a threshold problem, not a vector problem, and it is invisible
to accuracy. So this module keeps the working: per-row projections and margins,
the class means and threshold that produced them, and a threshold-free
ranking statistic (``auc``) that separates the two cases.

The 27B validates of 2026-08-01 turned the hypothetical into an incident:
every ``designatedReference`` concept thresholded its held-out scenarios at
the midpoint of two STORY-corpus projections, and the scenario distribution
sat entirely on one side of it (``fair``: tp=20 fp=20 tn=0 fn=0, accuracy
0.50, AUC 0.855). Two additions carry that lesson:

- ``heldOutCalibration`` — the same confusion arithmetic at the held-out
  items' OWN class-mean midpoint, answering "does the direction separate the
  validation classes at their natural boundary". It spends the held-out labels
  on exactly one scalar (the midpoint), which is a disclosed calibration, not
  a fit; and it is deliberately NOT sign-oriented, so an inverted direction
  reads below 0.5 here exactly as it does in AUC.
- ``oneSidedPredictions`` — true when the transfer threshold put every item
  on one side, the signature that ``accuracy`` is measuring the threshold,
  not the vector.

``accuracy`` keeps its historical meaning — does the EXTRACTION-derived
threshold transfer to held-out text — because that failure is worth seeing,
just not worth confusing with a dead direction.

**AUC and the calibration are DIAGNOSTICS.** No freeze gate reads them, and
none should start to without an explicit policy decision — they are reported
so a near-chance accuracy can be interpreted, not so they can be optimised
against.

Statistics here are deliberately DESCRIPTIVE (D2): confusion matrix, balanced
accuracy, class counts, Wilson intervals. No p-values. The binomial null
assumes independence, and these scenario sets are generated per concept by one
agent over shared topics with deliberately matched pairs — matched pairs and
topic clustering both induce correlation, so a binomial p-value would
understate variance and read optimistically. The inferential design (unit of
analysis, blocking, correction family) is a decision to be made and declared,
not a default to be shipped.

Cross-engine twin: ``Sources/ExperimentKit/ScenarioDiagnostics.swift``.
Committed fixture: ``Tests/Fixtures/cross-engine/scenario-diagnostics.json``.
"""

from __future__ import annotations

import hashlib
import json
import math


def row_hash(text: str, label: bool) -> str:
    """Identity of a scenario ROW, independent of its position in the file —
    so a diagnostic record still names the right item after a re-order.

    Hashes the canonical ROW, not the text alone: two scenarios with identical
    text and opposite ``expresses`` labels are different rows, and one
    identity between them makes a diagnostic record ambiguous about which it
    describes. Swift twin: ``ScenarioDiagnostics.rowHash``."""
    canonical = json.dumps({"expresses": bool(label), "text": text},
                           sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054
                    ) -> tuple[float, float] | None:
    """NAIVE item-level Wilson score interval (95% by default).

    "Naive" is not modesty — it is the assumption. Wilson assumes independent
    Bernoulli trials, and these scenario sets are generated per concept by one
    agent over shared topics with deliberately matched pairs. Matched pairs
    and topic clustering both induce correlation, so the TRUE interval is
    wider than this one. Calling the output "descriptive" does not repair
    that; an interval is an inferential object whatever it is labelled, so the
    label states the assumption instead.

    Kept rather than dropped because small-N imprecision is real and worth
    seeing: 20/20 and 200/200 are different evidence. A cluster-aware
    bootstrap would be the honest replacement, and it needs the blocking
    structure that is still undeclared (D2).

    Wilson rather than normal-approximation because these sets are small
    (40 items) and often near 0 or 1, where the normal interval runs past the
    ends of the scale and reports impossible bounds."""
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
              / denominator)
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def auc(projections: list[float], labels: list[bool]) -> float | None:
    """Threshold-free separation: the probability that a randomly chosen
    positive scenario projects above a randomly chosen negative one.

    Mann–Whitney U over midranks, so TIES CONTRIBUTE 0.5 — an all-ties
    direction scores exactly 0.5 rather than 0 or 1 depending on comparison
    order. Returns None when either class is empty: with one class there is
    nothing to separate, and any number would be an artefact.
    """
    positives = [p for p, lab in zip(projections, labels) if lab]
    negatives = [p for p, lab in zip(projections, labels) if not lab]
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def held_out_calibration(projections: list[float], labels: list[bool]
                         ) -> dict | None:
    """Confusion arithmetic at the held-out items' OWN class-mean midpoint.

    The transfer threshold comes from the extraction classes, and when those
    are a different text family (designatedReference's story corpora vs
    scenario validation items) the whole held-out distribution can land on one
    side of it. This measures separation at the boundary the held-out classes
    themselves define — one scalar spent on the labels, disclosed as such.

    NOT sign-oriented on purpose: an inverted direction scores below 0.5 here,
    agreeing with AUC, instead of being silently flipped into respectability.
    Returns None when either class is empty — with one class there is no
    boundary to define.

    Plain ``sum/len`` means rather than ``statistics.fmean``: the Swift twin
    reduces the same way, and the committed fixture pins the bytes.
    """
    positives = [p for p, lab in zip(projections, labels) if lab]
    negatives = [p for p, lab in zip(projections, labels) if not lab]
    if not positives or not negatives:
        return None
    class_means = {"positive": sum(positives) / len(positives),
                   "negative": sum(negatives) / len(negatives)}
    threshold = (class_means["positive"] + class_means["negative"]) / 2
    tp = sum(1 for p in positives if p > threshold)
    fn = len(positives) - tp
    fp = sum(1 for n in negatives if n > threshold)
    tn = len(negatives) - fp
    sensitivity = tp / len(positives)
    specificity = tn / len(negatives)
    return {
        "threshold": threshold,
        "classMeans": class_means,
        "accuracy": (tp + tn) / len(projections),
        "balancedAccuracy": (sensitivity + specificity) / 2,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def diagnostics(*, direction: list[float], scenarios: list[dict],
                projections: list[float], labels: list[bool],
                threshold: float, class_means: dict, layer: int,
                direction_norm: float) -> dict:
    """The rich record: everything the accuracy number was computed FROM.

    ``scenarios`` supplies each row's identity (``id``/``text``); the caller
    has already projected them, because how activations are obtained differs
    between the paired and grand-mean paths while the arithmetic below does
    not.
    """
    # Refuse rather than truncate. `zip` silently dropped the tail of a
    # longer array while Swift invented `false` for a missing label — the
    # same malformed input produced two different answers, neither flagged.
    counts = {"scenarios": len(scenarios), "projections": len(projections),
              "labels": len(labels)}
    if len(set(counts.values())) > 1:
        raise ValueError(
            "scenario diagnostics received unequal inputs "
            + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
            + " — a padded or truncated row set would silently change which "
            "items were scored")

    rows = []
    correct = 0
    tp = fp = tn = fn = 0
    for index, (scenario, projection, label) in enumerate(
            zip(scenarios, projections, labels)):
        predicted = projection > threshold
        is_correct = predicted == label
        correct += int(is_correct)
        if label and predicted:
            tp += 1
        elif label and not predicted:
            fn += 1
        elif not label and predicted:
            fp += 1
        else:
            tn += 1
        text = scenario.get("text", "")
        rows.append({
            # Position AND identity: a line number alone stops meaning
            # anything the moment the file is re-ordered.
            "index": index,
            "id": scenario.get("id") or f"scenario-{index + 1}",
            "rowHash": row_hash(text, label),
            "label": bool(label),
            "projection": projection,
            "predicted": predicted,
            # Signed distance from the decision boundary. Near zero means the
            # item barely decided either way — the rows to read first when an
            # accuracy is disappointing.
            "margin": projection - threshold,
            "correct": is_correct,
        })

    total = len(rows)
    positives = tp + fn
    negatives = tn + fp
    sensitivity = tp / positives if positives else None
    specificity = tn / negatives if negatives else None
    balanced = ((sensitivity + specificity) / 2
                if sensitivity is not None and specificity is not None else None)
    interval = wilson_interval(correct, total)

    return {
        "layer": layer,
        "threshold": threshold,
        "classMeans": class_means,
        "directionNorm": direction_norm,
        "scenarioCount": total,
        "classCounts": {"positive": positives, "negative": negatives},
        "accuracy": correct / total if total else None,
        # NAIVE item-level — assumes independent items, which these sets
        # violate. See `wilson_interval`.
        "naiveItemLevelInterval95": (list(interval) if interval else None),
        "balancedAccuracy": balanced,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        # DIAGNOSTIC ONLY — see the module docstring.
        "auc": auc(projections, labels),
        # The transfer threshold put every item on one side: `accuracy` is
        # measuring the threshold, not the vector. Read the calibration.
        "oneSidedPredictions": total > 0 and (tp + fp == total
                                              or tn + fn == total),
        # DIAGNOSTIC ONLY — separation at the held-out classes' own midpoint.
        "heldOutCalibration": held_out_calibration(projections, labels),
        "rows": rows,
    }
