"""The builder is the only place a chain is assembled.

Mirror of Swift ``Tests/SteeringKitTests/InterventionPlanTests.swift``. The
ablator's two requirements — read ``h₀``, remove a layer's directions as one
subspace — are satisfied by one construction and violated by any call site that
improvises, so the construction lives here and the tests pin it on both sides.
"""

import pytest

torch = pytest.importorskip("torch")

from steerlab_server.steering.ablator import SubspaceAblator  # noqa: E402
from steerlab_server.steering.injector import VectorInjector  # noqa: E402
from steerlab_server.steering import plan as plan_mod  # noqa: E402
from steerlab_server.steering.plan import (  # noqa: E402
    AmbiguousStrength, Edit, Mode)


def _edit(concept, vector, *, layer=0, strength=1.0, mode=Mode.ABLATE):
    return Edit(layer=layer, vector=vector, strength=strength, mode=mode,
                concept=concept)


def _out(chain, row):
    current = torch.tensor([[row]], dtype=torch.float32)
    for intervention in chain:
        current = intervention.apply(current, 0, 0)
    return current[0][0].tolist()


def test_a_pure_steering_condition_is_unchanged_from_before():
    chain = plan_mod.interventions([
        _edit("a", [0, 1, 0], strength=2.0, mode=Mode.ADD),
        _edit("b", [0, 0, 1], strength=3.0, mode=Mode.ADD),
    ])
    assert len(chain) == 2
    assert all(isinstance(item, VectorInjector) for item in chain)
    assert plan_mod.satisfies_ordering_invariant(chain)


def test_every_chain_has_at_most_one_ablator_and_it_is_first():
    chain = plan_mod.interventions([
        _edit("z", [1, 0, 0], layer=0),
        _edit("a", [0, 1, 0], layer=1),
        _edit("m", [0, 1, 0], layer=0),
        _edit("q", [0, 0, 1], layer=0, strength=2.0, mode=Mode.ADD),
    ])
    assert sum(isinstance(i, SubspaceAblator) for i in chain) == 1
    assert isinstance(chain[0], SubspaceAblator)
    assert plan_mod.satisfies_ordering_invariant(chain)


def test_concentric_ablations_at_a_layer_become_one_subspace():
    """Two concepts at one layer become one rank-2 subspace, not two
    sequential removals — the double-subtraction hazard closed by construction
    rather than by a comment."""
    ablator = plan_mod.ablator([
        _edit("fear", [1, 0, 0]), _edit("anger", [1, 0.2, 0])])
    assert ablator.ablation(0).rank == 2
    result = _out([ablator], [1, 1, 1])
    assert abs(result[0]) < 1e-5
    assert abs(result[1]) < 1e-5
    assert abs(result[2] - 1) < 1e-5


def test_the_basis_is_ordered_by_concept_not_by_caller_order():
    """Otherwise two runs of the same condition ablate subtly different
    subspaces, since Gram-Schmidt is order-dependent."""
    forward = plan_mod.ablator([
        _edit("anger", [1, 0.2, 0]), _edit("fear", [1, 0, 0])])
    reverse = plan_mod.ablator([
        _edit("fear", [1, 0, 0]), _edit("anger", [1, 0.2, 0])])
    assert forward.ablation(0).basis == reverse.ablation(0).basis


def test_two_strengths_at_one_layer_refuse():
    """λ scales the removal of a subspace whose orthonormalized rows no longer
    correspond to the concepts that produced them, so two λ at one layer has no
    defensible meaning and is refused rather than guessed."""
    with pytest.raises(AmbiguousStrength):
        plan_mod.ablator([
            _edit("a", [1, 0, 0], strength=1.0),
            _edit("b", [0, 1, 0], strength=0.5)])


def test_different_layers_may_carry_different_strengths():
    ablator = plan_mod.ablator([
        _edit("a", [1, 0, 0], layer=0, strength=1.0),
        _edit("b", [0, 1, 0], layer=1, strength=0.5)])
    assert ablator.ablation(0).strength == 1.0
    assert ablator.ablation(1).strength == 0.5


def test_a_condition_that_ablates_nothing_builds_no_ablator():
    assert plan_mod.ablator([]) is None
    assert plan_mod.ablator([_edit("a", [1, 0, 0], mode=Mode.ADD)]) is None
    # An all-zero vector cannot define a direction: no ablator rather than a
    # basis of noise.
    assert plan_mod.ablator([_edit("a", [0, 0, 0])]) is None


def test_the_built_chain_is_order_independent():
    """The whole point of the ordering: the chain computes
    `h₀ − λ·P h₀ + Σαᵢvᵢ` regardless of how the injectors are listed."""
    edits = [
        _edit("x", [1, 0, 0]),
        _edit("y", [0, 1, 0], strength=2.0, mode=Mode.ADD),
        _edit("z", [0, 0, 1], strength=3.0, mode=Mode.ADD),
    ]
    a = _out(plan_mod.interventions(edits), [5, 1, 1])
    b = _out(plan_mod.interventions(list(reversed(edits))), [5, 1, 1])
    assert all(abs(x - y) < 1e-5 for x, y in zip(a, b))
    assert abs(a[0]) < 1e-5
    assert abs(a[1] - 3) < 1e-5
    assert abs(a[2] - 4) < 1e-5


def test_the_invariant_check_rejects_a_misordered_chain():
    ablator = SubspaceAblator.single([0], [1, 0, 0])
    injector = VectorInjector.single(0, [0, 1, 0], 1.0)
    assert not plan_mod.satisfies_ordering_invariant([injector, ablator])
    assert not plan_mod.satisfies_ordering_invariant([ablator, ablator])
    assert plan_mod.satisfies_ordering_invariant([ablator, injector])
    assert plan_mod.satisfies_ordering_invariant([injector])


def test_both_engines_build_the_same_basis():
    """The Swift twin asserts this exact value; if the two orthonormalizations
    ever diverge, one side's test fails instead of two studies quietly
    disagreeing."""
    ablator = plan_mod.ablator([
        _edit("fear", [1, 0, 0]), _edit("anger", [1, 0.2, 0])])
    basis = ablator.ablation(0).basis
    # "anger" sorts before "fear", so ANGER seeds the basis — the concept-name
    # ordering is what makes this reproducible at all, and pinning the literal
    # is what makes it comparable to the other engine.
    assert basis[0] == pytest.approx(
        (0.9805806756909201, 0.19611613513818402, 0.0), abs=1e-9)
    assert basis[1] == pytest.approx(
        (0.19611613513818446, -0.98058067569092, 0.0), abs=1e-9)
