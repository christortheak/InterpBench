"""Ablation is a projection, and the properties that make it one are the point.

Mirror of Swift ``Tests/SteeringKitTests/SubspaceAblatorTests.swift`` — same
claims, same numbers, so a divergence between the engines shows up as one
side's test failing rather than as two studies that quietly disagree.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.steering.ablator import (  # noqa: E402
    Ablation, SubspaceAblator, orthonormalized)
from steerlab_server.steering.injector import Injection, VectorInjector  # noqa: E402


def _hidden(rows):
    return torch.tensor([rows], dtype=torch.float32)  # [1, seq, hidden]


def _rows(h):
    return h[0].tolist()


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(v):
    n = _norm(v)
    return [x / n for x in v]


# --- orthonormalization ----------------------------------------------------

def test_the_basis_is_orthonormal():
    basis = orthonormalized([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]])
    assert len(basis) == 3
    for row in basis:
        assert abs(_norm(row) - 1) < 1e-9
    for i in range(len(basis)):
        for j in range(i + 1, len(basis)):
            assert abs(_dot(basis[i], basis[j])) < 1e-9


def test_dependent_directions_are_dropped_not_normalized_from_noise():
    """A direction already spanned by the earlier ones is DROPPED rather than
    normalized from numerical noise — the rank falls, which is the honest
    report of what was asked for."""
    basis = orthonormalized([[1, 0, 0], [2, 0, 0], [0, 1, 0]])
    assert len(basis) == 2


def test_zero_directions_produce_an_empty_basis():
    assert orthonormalized([[0, 0, 0]]) == []
    assert orthonormalized([]) == []


def test_the_basis_matches_the_swift_twin_bit_for_bit_enough():
    """Both engines run modified Gram-Schmidt in 64-bit over the same order,
    so the bases agree to float precision — the reason the ablation the
    Playground shows is the ablation the cluster runs."""
    basis = orthonormalized([[3, 4, 0]])
    assert basis == [[0.6, 0.8, 0.0]]


# --- the defining properties ----------------------------------------------

def test_full_ablation_zeroes_the_projection():
    ablator = SubspaceAblator.single([0], [3, 4, 0])
    out = _rows(ablator.apply(_hidden([[1, 2, 3], [-5, 0.5, 2]]), 0, 0))
    unit = _unit([3, 4, 0])
    for row in out:
        assert abs(_dot(row, unit)) < 1e-5


def test_the_norm_falls_by_exactly_the_removed_component():
    """‖h'‖² = ‖h‖² − c²: the norm falls by the removed part, not by the
    vector's own norm (which is what a fixed −α·v would cost)."""
    ablator = SubspaceAblator.single([0], [1, 1, 0])
    source = [2, -1, 4]
    out = _rows(ablator.apply(_hidden([source]), 0, 0))[0]
    c = _dot(source, _unit([1, 1, 0]))
    assert abs(_norm(out) - math.sqrt(_dot(source, source) - c * c)) < 1e-4


def test_full_ablation_is_idempotent():
    ablator = SubspaceAblator.single([0], [1, 2, 3])
    once = ablator.apply(_hidden([[4, -1, 0.5]]), 0, 0)
    twice = ablator.apply(once, 0, 0)
    assert torch.allclose(once, twice, atol=1e-5)


def test_lambda_two_reflects_and_preserves_the_norm():
    """λ=2 reflects the component through the hyperplane: the projection flips
    sign and the NORM IS PRESERVED — the well-behaved "reverse the concept"
    that a large negative α cannot give, since its norm change is
    uncontrolled."""
    ablator = SubspaceAblator.single([0], [0, 1, 0], strength=2.0)
    source = [3, 5, -1]
    out = _rows(ablator.apply(_hidden([source]), 0, 0))[0]
    assert abs(_norm(out) - _norm(source)) < 1e-4
    unit = _unit([0, 1, 0])
    assert abs(_dot(out, unit) + _dot(source, unit)) < 1e-4


def test_partial_ablation_removes_part_of_the_component():
    ablator = SubspaceAblator.single([0], [1, 0, 0], strength=0.25)
    out = _rows(ablator.apply(_hidden([[4, 1, 1]]), 0, 0))[0]
    assert abs(out[0] - 3) < 1e-4  # 4 − 0.25·4
    assert abs(out[1] - 1) < 1e-5


# --- the subspace invariant ------------------------------------------------

def test_correlated_directions_are_removed_jointly_not_twice():
    """THE reason all directions at a layer share one ablator.

    Subtracting two correlated directions' projections separately
    double-counts what they share: neither ends up removed, and the shared
    component is driven NEGATIVE by an uncontrolled amount — silent negative
    steering behind a control labelled "ablation"."""
    v, w, source = [1, 0, 0], [1, 0.2, 0], [1, 1, 1]
    unit_v, unit_w = _unit(v), _unit(w)

    # Wrong: two independent projections off the SAME input.
    c_v, c_w = _dot(source, unit_v), _dot(source, unit_w)
    doubled = [source[i] - c_v * unit_v[i] - c_w * unit_w[i] for i in range(3)]
    assert _dot(doubled, unit_v) < -0.1, (
        "the double-subtraction should overshoot into negative steering — if "
        "it does not, this test no longer demonstrates the hazard")

    # Right: one ablator over the orthonormalized span.
    basis = orthonormalized([v, w])
    assert len(basis) == 2
    ablator = SubspaceAblator(
        {0: Ablation(basis=tuple(tuple(r) for r in basis), strength=1.0)})
    out = _rows(ablator.apply(_hidden([source]), 0, 0))[0]
    assert abs(_dot(out, unit_v)) < 1e-5
    assert abs(_dot(out, unit_w)) < 1e-5


def test_an_ablator_first_makes_the_chain_order_independent():
    """With the ablator FIRST, the per-layer result is `h₀ − λ·P h₀ + Σαᵢvᵢ`
    however the injectors are ordered — the property that lets the ablator
    live in a sequential chain."""
    ablator = SubspaceAblator.single([0], [1, 0, 0])
    first = VectorInjector({0: Injection(vector=[0, 1, 0], alpha=2.0)})
    second = VectorInjector({0: Injection(vector=[0, 0, 1], alpha=3.0)})
    source = _hidden([[5, 1, 1]])

    def chain(interventions):
        current = source
        for intervention in interventions:
            current = intervention.apply(current, 0, 0)
        return _rows(current)[0]

    a = chain([ablator, first, second])
    b = chain([ablator, second, first])
    assert all(abs(x - y) < 1e-5 for x, y in zip(a, b))
    assert abs(a[0]) < 1e-5
    assert abs(a[1] - 3) < 1e-5  # 1 + 2
    assert abs(a[2] - 4) < 1e-5  # 1 + 3


def test_a_non_orthogonal_injection_reintroduces_the_ablated_direction():
    """The documented semantics, made a test so it cannot drift into folklore:
    ablation removes what the MODEL produced, not what you injected."""
    ablator = SubspaceAblator.single([0], [1, 0, 0])
    injector = VectorInjector({0: Injection(vector=[1, 1, 0], alpha=1.0)})
    current = ablator.apply(_hidden([[4, 0, 0]]), 0, 0)
    current = injector.apply(current, 0, 0)
    assert abs(_rows(current)[0][0] - 1) < 1e-5


# --- positions -------------------------------------------------------------

def test_every_position_is_ablated_including_mid_prompt_chunks():
    """Ablation fires at EVERY position — the inverse of the injector's
    chunked-prefill gate, which deliberately leaves mid-prompt chunk tails
    untouched. Here the whole prompt must be stripped."""
    ablator = SubspaceAblator.single([0], [1, 0, 0])
    out = _rows(ablator.apply(_hidden([[9, 1, 0], [8, 2, 0], [7, 3, 0]]), 0, 0))
    assert len(out) == 3
    assert all(abs(row[0]) < 1e-5 for row in out)
    # The orthogonal content of every position survives untouched.
    assert [row[1] for row in out] == pytest.approx([1, 2, 3])


def test_unconfigured_layers_are_untouched():
    ablator = SubspaceAblator.single([3], [1, 0, 0])
    out = _rows(ablator.apply(_hidden([[9, 1, 0]]), 0, 0))[0]
    assert out == pytest.approx([9, 1, 0])


def test_a_bf16_residual_stream_is_ablated_in_float32():
    """The residual stream is bf16/fp16 on real models. A reduction over
    `hidden` terms in bf16 drifts enough that the engines' differing reduction
    strategies would surface as a different ablation for the same config, so
    both compute in float32 and narrow only the result."""
    ablator = SubspaceAblator.single([0], [3, 4, 0])
    h = _hidden([[1, 2, 3]]).to(torch.bfloat16)
    out = ablator.apply(h, 0, 0)
    assert out.dtype == torch.bfloat16
    # Accurate to bf16's own resolution, not to a compounded reduction error.
    row = out.to(torch.float32)[0].tolist()[0]
    assert abs(_dot(row, _unit([3, 4, 0]))) < 0.02


# --- ablation as a study condition -----------------------------------------

def test_a_legacy_slot_decodes_as_steering():
    """Manifest bytes are the content hash, so `mode` must be absent-means-add
    or every frozen study in the workspace would re-identify."""
    from steerlab_server.experiment.manifest import Manifest
    manifest = Manifest.from_dict({
        "name": "s", "modelID": "org/m", "concepts": [],
        "conditions": [{"name": "c", "slots": [
            {"concept": "fear", "layer": 14, "alpha": 0.5}]}]})
    slot = manifest.conditions[0].slots[0]
    assert slot.mode is None
    assert slot.effective_mode == "add"
    assert not slot.is_ablation


def test_an_ablation_slot_parses():
    from steerlab_server.experiment.manifest import Manifest
    manifest = Manifest.from_dict({
        "name": "s", "modelID": "org/m", "concepts": [],
        "conditions": [{"name": "c", "slots": [
            {"concept": "fear", "layer": 0, "alpha": 1.0, "mode": "ablate"}]}]})
    assert manifest.conditions[0].slots[0].is_ablation


def test_the_ablation_control_matrix_is_not_the_steering_one():
    """No layer, no norm units, no negative cell — λ=2 reflects, whereas a
    negative λ would simply add the concept back — and a random DIRECTION
    rather than a norm-matched random vector."""
    from steerlab_server.experiment.control_matrix import (
        ablation_control_conditions)
    cells = ablation_control_conditions("fear")
    assert [c["name"] for c in cells] == [
        "fear-ablate-l0p5", "fear-ablate-l1", "fear-ablate-random"]
    assert all(c["alphaInNormUnits"] is False for c in cells)
    assert all(c["slots"][0]["mode"] == "ablate" for c in cells)
    assert all("neg" not in c["name"] for c in cells)
    assert cells[-1]["controlType"] == "randomDirectionAblation"
    # The control ablates at the same λ — only the DIRECTION differs.
    assert cells[-1]["slots"][0]["alpha"] == 1.0


def test_lambda_zero_refuses_with_the_reason_and_the_alternatives():
    from steerlab_server.experiment.control_matrix import (
        ablation_control_conditions)
    with pytest.raises(ValueError) as excinfo:
        ablation_control_conditions("fear", lambdas=[0])
    message = str(excinfo.value)
    assert "baseline already covers it" in message
    assert "0.5" in message and "reflect" in message
