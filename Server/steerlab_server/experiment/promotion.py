"""Screen→confirm funnel artifacts (study guide Phase 1 → Phase 2).

A Phase-1 screen produces a ``promoted-movers.json`` naming exactly which
concepts enter the confirm phase and why. The promotion rule is data (pinned in
the manifest), evaluated here, and the artifact carries full provenance so the
confirm study can reference it immutably. Disjoint-pool enforcement lives in
``Manifest.verify`` (confirm phase must pin the screen pool hash it is held out
from); this module supplies the artifact both sides reference.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from .manifest import PromotionRule
from .study_stats import DoseResponse, EffectRow


@dataclass
class PromotionCandidate:
    """One concept's screening evidence, assembled by the analysis step."""

    concept: str
    condition: str
    endpoint: str
    effect: EffectRow
    dose: DoseResponse | None = None
    random_floor_effect: float | None = None   # |Δ| of the matched-norm random arm
    capability_passed: bool | None = None
    provenance: dict = field(default_factory=dict)


@dataclass
class PromotionDecision:
    candidate: PromotionCandidate
    promoted: bool
    reasons: list[str]

    def as_json(self) -> dict:
        effect = self.candidate.effect
        payload = {
            "concept": self.candidate.concept,
            "condition": self.candidate.condition,
            "endpoint": self.candidate.endpoint,
            "effectEstimate": effect.ci.mean,
            "effectCILower": effect.ci.ci_lower,
            "effectCIUpper": effect.ci.ci_upper,
            "wilcoxonP": None if math.isnan(effect.wilcoxon_p) else effect.wilcoxon_p,
            "adjustedP": None if math.isnan(effect.adjusted_p) else effect.adjusted_p,
            "correction": effect.correction,
            "doseMonotone": self.candidate.dose.is_monotone if self.candidate.dose else None,
            # An UNDEFINED rho (flat ladder, or tied doses — zero rank
            # variance) must stay distinguishable from a valid zero, so it
            # serializes as null and never as 0.0. This is load-bearing, not
            # tidiness: json.dump would otherwise emit the bare token NaN,
            # which is not JSON and which a reader comparing `>= 0` would
            # misread as "no correlation" rather than "no answer" (external
            # review, 2026-09-05, SCI-04).
            "doseSpearmanRho": (None if self.candidate.dose is None
                                or math.isnan(self.candidate.dose.spearman_rho)
                                else self.candidate.dose.spearman_rho),
            "randomFloorEffect": self.candidate.random_floor_effect,
            "capabilityPassed": self.candidate.capability_passed,
            "promoted": self.promoted,
            "reasons": self.reasons,
        }
        payload.update(self.candidate.provenance)
        return payload


def decide(candidate: PromotionCandidate, rule: PromotionRule) -> PromotionDecision:
    """Apply the pinned promotion rule to one concept's screening evidence.
    Every criterion failure is named — a silent gate is unauditable."""
    reasons: list[str] = []
    p = candidate.effect.adjusted_p
    if math.isnan(p):
        reasons.append("no adjusted p-value (screen analysis incomplete)")
    elif p > rule.fdr_threshold:
        reasons.append(f"adjusted p {p:.4g} exceeds FDR threshold {rule.fdr_threshold}")
    if rule.dose_monotone:
        if candidate.dose is None:
            reasons.append("no dose-response evidence (alpha grid missing)")
        elif not candidate.dose.is_monotone:
            reasons.append("dose-response is not monotone")
    if rule.exceeds_random_floor:
        if candidate.random_floor_effect is None:
            reasons.append("no matched-norm random floor measured")
        elif abs(candidate.effect.ci.mean) <= abs(candidate.random_floor_effect):
            reasons.append(
                f"effect {candidate.effect.ci.mean:.4g} does not exceed the "
                f"random floor {candidate.random_floor_effect:.4g}")
    if rule.capability_gate is not None:
        if candidate.capability_passed is None:
            reasons.append(f"capability gate '{rule.capability_gate}' not evaluated")
        elif not candidate.capability_passed:
            reasons.append(f"capability gate '{rule.capability_gate}' failed")
    return PromotionDecision(candidate=candidate, promoted=not reasons, reasons=reasons)


def write_promoted_movers(path: str, decisions: list[PromotionDecision], *,
                          experiment: str, experiment_hash: str,
                          rule: PromotionRule) -> None:
    """The frozen Phase-1 output the confirm study must import. Includes the
    non-promoted concepts and their failure reasons — the funnel is only
    defensible if rejections are as documented as promotions."""
    payload = {
        "experiment": experiment,
        "experimentHash": experiment_hash,
        "promotionRule": {
            "fdrThreshold": rule.fdr_threshold,
            "doseMonotone": rule.dose_monotone,
            "exceedsRandomFloor": rule.exceeds_random_floor,
            "capabilityGate": rule.capability_gate,
        },
        "promoted": [d.as_json() for d in decisions if d.promoted],
        "rejected": [d.as_json() for d in decisions if not d.promoted],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
