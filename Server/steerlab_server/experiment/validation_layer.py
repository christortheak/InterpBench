"""Which layer convergent validity is read at — declared, not inferred (D4).

The historical rule was "the layer a condition steers this concept at, else
mid-network". That made the validation layer a SIDE EFFECT of the injection
conditions: to revalidate at a different depth you had to add or edit a
steering condition, which is a different decision entirely. On 2026-07-26 a
researcher testing a readout-layer hypothesis had to hand-write nine
conditions at layer 41 to move the validation read, because the manifest
offered no way to say the thing they meant.

Replacing that with a hardcoded default would swap one hidden choice for
another. The layer is a measurement decision, so it is declared:

1. ``validationLayer`` — an absolute index.
2. ``validationLayerFraction`` — a depth fraction, for studies that should
   read at the same relative depth across model sizes.
3. Legacy: the first condition slot steering this concept (preserved exactly,
   so existing manifests keep their numbers and their content hash).
4. Mid-network.

Both the resolved index AND the fraction are reported: an index means nothing
without the depth it represents, and a depth means nothing without the model
it resolved against.

Cross-engine twin: ``Sources/ExperimentKit/ValidationLayerRule.swift``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DECLARED_INDEX = "declaredIndex"
DECLARED_FRACTION = "declaredFraction"
STEERING_CONDITION = "steeringCondition"
MID_NETWORK = "midNetwork"

_BECAUSE = {
    DECLARED_INDEX: "declared validationLayer",
    DECLARED_FRACTION: "declared validationLayerFraction",
    STEERING_CONDITION: ("inherited from a steering condition (legacy rule — "
                         "declare validationLayer to say it directly)"),
    MID_NETWORK: ("mid-network default (legacy rule — declare validationLayer "
                  "to say it directly)"),
}


@dataclass(frozen=True)
class Resolution:
    layer: int
    layer_count: int
    source: str

    @property
    def depth_fraction(self) -> float:
        if self.layer_count <= 1:
            return 0.0
        return self.layer / (self.layer_count - 1)

    @property
    def summary(self) -> str:
        depth = f"{self.depth_fraction:.2f}".rstrip("0").rstrip(".") or "0"
        return (f"layer {self.layer} of {self.layer_count}, {depth} depth — "
                f"{_BECAUSE[self.source]}")


def resolve(*, declared_layer: int | None, declared_fraction: float | None,
            condition_layer: int | None, layer_count: int) -> Resolution:
    max_index = max(0, layer_count - 1)

    def clamp(value: int) -> int:
        return min(max(0, value), max_index)

    if declared_layer is not None:
        return Resolution(clamp(int(declared_layer)), layer_count, DECLARED_INDEX)
    if declared_fraction is not None and math.isfinite(declared_fraction):
        # Truncating, clamped — the same rule `resolve_sweep_layers` applies,
        # so a fraction means one thing across the engine.
        return Resolution(clamp(int(declared_fraction * layer_count)),
                          layer_count, DECLARED_FRACTION)
    if condition_layer is not None:
        return Resolution(clamp(int(condition_layer)), layer_count,
                          STEERING_CONDITION)
    return Resolution(clamp(layer_count // 2), layer_count, MID_NETWORK)


def resolve_all(*, declared_layers=None, declared_fractions=None,
                declared_layer=None, declared_fraction=None,
                condition_layer=None, layer_count: int) -> list[Resolution]:
    """Every layer convergent validity reads at, in declared order.

    The depth LIST exists for the validate-at-the-sweep-layers policy
    (2026-08-01): the steering sweep spans a band of layers, and the reading
    certificate should cover every layer the sweep may promote — measured in
    ONE run, since the scenario activations are captured once for all layers.

    Declaring a list and a scalar together is refused by ``violation``; here
    a declared list wins, else the scalar path resolves as before (so every
    existing manifest keeps its single resolution). Two declared entries that
    resolve to the SAME layer refuse: the report keys per-depth entries by
    layer, and silently collapsing "0.60 and 0.61 of 6 layers" into one row
    would misreport what was declared.
    """
    max_index = max(0, layer_count - 1)

    def clamp(value: int) -> int:
        return min(max(0, value), max_index)

    resolutions: list[Resolution] = []
    if declared_layers:
        for layer in declared_layers:
            refusal = range_refusal(layer, layer_count)
            if refusal:
                raise RuntimeError(refusal)
            resolutions.append(
                Resolution(clamp(int(layer)), layer_count, DECLARED_INDEX))
    elif declared_fractions:
        for fraction in declared_fractions:
            resolutions.append(
                Resolution(clamp(int(fraction * layer_count)), layer_count,
                           DECLARED_FRACTION))
    else:
        return [resolve(declared_layer=declared_layer,
                        declared_fraction=declared_fraction,
                        condition_layer=condition_layer,
                        layer_count=layer_count)]
    seen: dict[int, object] = {}
    for declared, res in zip(declared_layers or declared_fractions,
                             resolutions):
        if res.layer in seen:
            field = ("validationLayers" if declared_layers
                     else "validationLayerFractions")
            raise RuntimeError(
                f"{field} entries {seen[res.layer]!r} and {declared!r} both "
                f"resolve to layer {res.layer} on this {layer_count}-layer "
                "model — declared depths must stay distinct once resolved, "
                "or the per-depth report would silently collapse them")
        seen[res.layer] = declared
    return resolutions


def range_refusal(declared_layer, layer_count: int) -> str | None:
    """An EXPLICIT declaration outside the model's depth, once depth is known.

    ``resolve`` clamps, which is right for the LEGACY paths — a condition
    layer inherited from a study built for a deeper model, or the mid-network
    fallback, should bend rather than break. It is wrong for an explicit
    ``validationLayer: 100`` on a 62-layer model: that is a scientific
    declaration the researcher wrote down, and silently reading layer 61
    instead answers a different question without saying so.

    Separate from ``violation`` because depth is not known at verify time —
    no model is loaded there — so this is checked where the layer is actually
    resolved. Swift twin: ``ValidationLayerRule.rangeRefusal``."""
    if declared_layer is None or layer_count <= 0:
        return None
    if declared_layer < layer_count:
        return None
    return (f"validationLayer {declared_layer} is outside this model's depth "
            f"({layer_count} layers, so the last index is {layer_count - 1}) — "
            "an explicit declaration is not silently clamped; declare a layer "
            "that exists, or a validationLayerFraction if the study should "
            "follow depth across model sizes")


def _index_problem(field: str, value) -> str | None:
    # bool is an int subtype in Python and a layer index of `true` is a typo,
    # not a declaration — same rule as the pipeline thresholds (2026-08-02).
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return (f"{field} must be a non-negative layer index "
                f"(got {value!r})")
    return None


def _fraction_problem(field: str, value) -> str | None:
    ok = (isinstance(value, (int, float)) and not isinstance(value, bool)
          and math.isfinite(value) and 0 <= value <= 1)
    if not ok:
        return f"{field} must be a number in [0, 1] (got {value!r})"
    return None


def violation(declared_layer, declared_fraction,
              declared_layers=None, declared_fractions=None) -> str | None:
    """Declaring more than one depth field is ambiguous: an index and a
    fraction disagree on every model whose depth is not exactly what the
    author assumed, and a scalar next to a list leaves which one the run
    reads unsaid. Exactly one of the four may be declared."""
    declared = [name for name, value in (
        ("validationLayer", declared_layer),
        ("validationLayerFraction", declared_fraction),
        ("validationLayers", declared_layers),
        ("validationLayerFractions", declared_fractions),
    ) if value is not None]
    if len(declared) > 1:
        if declared == ["validationLayer", "validationLayerFraction"]:
            return ("validationLayer and validationLayerFraction are both "
                    "declared — they disagree on any model whose depth differs "
                    "from the one assumed; declare exactly one")
        return (" and ".join(declared) + " are declared together — the run "
                "cannot know which depth declaration to read; declare exactly "
                "one")
    if declared_layer is not None:
        if (problem := _index_problem("validationLayer", declared_layer)):
            return problem
    if declared_fraction is not None:
        if (problem := _fraction_problem("validationLayerFraction",
                                         declared_fraction)):
            return problem
    for field, values, check in (
            ("validationLayers", declared_layers, _index_problem),
            ("validationLayerFractions", declared_fractions,
             _fraction_problem)):
        if values is None:
            continue
        if not isinstance(values, list) or not values:
            return (f"{field} must be a non-empty list — an empty list "
                    "declares nothing; remove the field to use the legacy "
                    "rule")
        for value in values:
            if (problem := check(f"{field} entries", value)):
                return problem
        if len(set(values)) != len(values):
            return (f"{field} contains duplicate entries — each declared "
                    "depth is measured once")
    return None
