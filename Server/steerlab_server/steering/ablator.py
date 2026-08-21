"""Removes concept directions from the residual stream by projection
(parallel to Swift ``SubspaceAblator``).

``h' = h − λ·P h``, where ``P`` projects onto the span of the ablated
directions — "the model cannot represent this here" rather than "push against
it". Contrast :class:`~steerlab_server.steering.injector.VectorInjector`, which
adds a FIXED offset ``α·v`` whatever is present: negative steering overshoots
where the concept is weakly represented (driving the projection negative, i.e.
asserting the anti-concept) and undershoots where it is strong. Ablation
removes exactly what is there, so it takes no strength parameter and no
residual-norm denominator — ``α``'s whole purpose is comparability, and here
there is nothing to make comparable.

Properties the tests pin: ``h'·v̂ = 0`` for every ablated direction at λ=1;
``‖h'‖² = ‖h‖² − Σcₖ²``; idempotent at λ=1; norm-PRESERVING at λ=2, which
reflects the component through the hyperplane (a better-behaved "reverse the
concept" than a large negative α, whose norm change is uncontrolled).

**Two invariants that are really one.** All directions at a layer live in ONE
ablator: subtracting each direction's projection separately double-counts what
they share, leaving ``h'·v̂ = −c_w(ŵ·v̂)`` for non-orthogonal ``v̂, ŵ`` — neither
removed, and correlated concepts driven negative by an uncontrolled amount. And
the ablator runs BEFORE any injector, because every edit must be computed from
the block's unmodified output ``h₀`` for the per-layer result to be the
order-independent ``h₀ − λ·P h₀ + Σ αᵢvᵢ``. Interventions are a sequential
chain, so "reads ``h₀``" means "runs first"; pure additions commute among
themselves, so one ablator in front suffices — which is also exactly what the
subspace invariant demands.

The semantics: ablation removes what the MODEL produced, not what you
injected. Steering a non-orthogonal vector therefore reintroduces some of the
ablated direction, which is the intended reading.

**Positions.** Applies at EVERY position, unlike injection: the claim is that
the direction is unavailable, and letting the model read the whole prompt with
the concept intact and stripping it only while writing is a different, muddier
intervention. This deliberately inverts the injector's chunked-prefill gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .intervention import LayerIntervention

#: Directions below this relative norm after orthogonalization are already
#: inside the span of the earlier ones and are dropped: keeping them would
#: divide by a near-zero norm and yield a basis vector of numerical noise.
#: Identical constant in Swift ``SubspaceAblator.dependenceTolerance``.
DEPENDENCE_TOLERANCE = 1e-6


def orthonormalized(directions: list[list[float]]) -> list[list[float]]:
    """Modified Gram-Schmidt over the supplied directions, in float64.

    Deterministic and engine-shared: the CALLER fixes the order (by concept
    name), the algorithm is modified — not classical — Gram-Schmidt for
    numerical stability, and accumulation is float64 regardless of the residual
    stream's dtype. Swift's ``SubspaceAblator.orthonormalized`` does the
    identical thing, so the engines agree to float precision rather than to
    whatever their reduction strategies happen to share.

    Zero and dependent directions are dropped, so the result may be shorter
    than the input; an all-zero input yields an empty basis, which makes
    :meth:`SubspaceAblator.apply` a no-op rather than a division by zero.
    """
    basis: list[list[float]] = []
    for direction in directions:
        residual = [float(value) for value in direction]
        original_norm = _norm(residual)
        if original_norm <= 0:
            continue
        # Modified: subtract each accepted basis vector's component from the
        # RUNNING residual, not from the original.
        for accepted in basis:
            coefficient = _dot(residual, accepted)
            for index in range(len(residual)):
                residual[index] -= coefficient * accepted[index]
        remaining = _norm(residual)
        if remaining <= DEPENDENCE_TOLERANCE * original_norm:
            continue  # already spanned by the earlier directions
        basis.append([value / remaining for value in residual])
    return basis


def _dot(a: list[float], b: list[float]) -> float:
    total = 0.0
    for index in range(len(a)):
        total += a[index] * b[index]
    return total


def _norm(a: list[float]) -> float:
    return math.sqrt(_dot(a, a))


@dataclass(frozen=True)
class Ablation:
    """One layer's ablated subspace."""

    #: Orthonormal rows spanning the subspace. Built by :func:`orthonormalized`,
    #: never supplied raw.
    basis: tuple[tuple[float, ...], ...]
    #: λ. 1 = full ablation, (0,1) = partial, 2 = reflection.
    strength: float = 1.0

    @property
    def rank(self) -> int:
        """Rank of the removed subspace — below the number of directions
        supplied when some were linearly dependent on the others."""
        return len(self.basis)


class SubspaceAblator(LayerIntervention):
    def __init__(self, ablations: dict[int, Ablation]):
        self._ablations = dict(ablations)
        # Materialized lazily per (device, layer); always float32.
        self._tensor_cache: dict[tuple, torch.Tensor] = {}

    @classmethod
    def single(cls, layers: list[int], vector: list[float],
               strength: float = 1.0) -> "SubspaceAblator":
        """One concept ablated across a set of layers — the common case."""
        basis = tuple(tuple(row) for row in orthonormalized([vector]))
        return cls({layer: Ablation(basis=basis, strength=strength)
                    for layer in layers})

    def ablation(self, layer: int) -> Ablation | None:
        return self._ablations.get(layer)

    def apply(self, h: torch.Tensor, layer: int, offset: int) -> torch.Tensor:
        ablation = self._ablations.get(layer)
        if ablation is None or not ablation.basis:
            return h

        key = (h.device, layer)
        basis = self._tensor_cache.get(key)
        if basis is None:
            basis = torch.tensor([list(row) for row in ablation.basis],
                                 dtype=torch.float32, device=h.device)
            self._tensor_cache[key] = basis

        # Float32 throughout: the residual stream is bf16/fp16, and a reduction
        # over `hidden` terms in bf16 drifts enough that the two engines'
        # differing reduction strategies would surface as a different ablation
        # for the same configuration. Only the stored result is narrowed.
        original_dtype = h.dtype
        h32 = h.to(torch.float32)
        # coefficients [batch, seq, rank]; removed [batch, seq, hidden].
        # Every position, no gating.
        coefficients = h32 @ basis.T
        removed = coefficients @ basis
        return (h32 - removed * ablation.strength).to(original_dtype)
