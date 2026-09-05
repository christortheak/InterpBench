"""Turns a condition's declared edits into the intervention chain, and is the
ONLY place allowed to build one (parallel to Swift ``InterventionPlan``).

Steering was assembled inline at each call site, which was safe because
additions commute — no site could order them wrongly. Ablation removes that
safety: it must read the block's unmodified output ``h₀``, and in a sequential
chain that means running first, and all of a layer's ablated directions must be
removed as ONE subspace or their shared component is subtracted twice.

Both requirements are satisfied by the same construction — a single ablator at
the head of the chain — so they are built here once rather than documented at
five call sites and eventually gotten wrong at one of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from . import intervention as scope_vocabulary
from .ablator import Ablation, SubspaceAblator, orthonormalized
from .injector import Injection, VectorInjector
from .intervention import InterventionScope, LayerIntervention
from .sae_latent import SAELatentEdit, SAELatentIntervention, group_edits


class Mode(str, Enum):
    """How an edit acts on the residual stream."""

    #: ``h + α·v`` — a fixed offset, whatever is present.
    ADD = "add"
    #: ``h − λ·(h·v̂)v̂`` — removes what is present.
    ABLATE = "ablate"


@dataclass(frozen=True)
class Edit:
    """One declared edit, with its vector already resolved."""

    layer: int
    vector: list[float]
    #: α for ``add``, λ for ``ablate``.
    strength: float
    mode: Mode
    #: Names the concept. Used ONLY to order the ablation basis: modified
    #: Gram-Schmidt depends on input order, so a stable, data-derived order is
    #: what makes the removed subspace reproducible across runs and engines
    #: rather than a function of dict iteration.
    concept: str
    #: The centering convention this edit's DIRECTION was expressed in
    #: ("none" or "neutralMean"). Declared upstream, where the vector is
    #: resolved, and carried here only so the chain can DESCRIBE itself
    #: (:func:`scope_inventory`) — nothing in the arithmetic reads it, and no
    #: caller has to supply it, so every existing construction is unchanged.
    centering: str = scope_vocabulary.CENTERING_NONE


class AmbiguousStrength(Exception):
    def __init__(self, layer: int, strengths: list[float]):
        self.layer = layer
        self.strengths = strengths
        super().__init__(
            f"layer {layer} declares ablations with different strengths "
            f"({', '.join(str(s) for s in strengths)}) — λ scales the removal "
            "of a SUBSPACE, and once the directions are orthonormalized its "
            "rows no longer correspond to the concepts that produced them, so "
            "there is no defensible way to apply two. Use one λ per layer")


def ablator(edits: list[Edit]) -> SubspaceAblator | None:
    """The single ablator covering every ablated layer, or None when the
    condition ablates nothing."""
    ablations = [edit for edit in edits if edit.mode == Mode.ABLATE]
    if not ablations:
        return None

    by_layer: dict[int, list[Edit]] = {}
    for edit in ablations:
        by_layer.setdefault(edit.layer, []).append(edit)

    per_layer: dict[int, Ablation] = {}
    for layer, group in by_layer.items():
        strengths = sorted({edit.strength for edit in group})
        if len(strengths) != 1:
            raise AmbiguousStrength(layer, strengths)
        # Sorted by concept: Gram-Schmidt is order-dependent, so the basis must
        # not depend on how the caller happened to build its list. Ties (the
        # same concept twice at a layer) are dropped as linearly dependent.
        ordered = sorted(group, key=lambda edit: edit.concept)
        basis = orthonormalized([edit.vector for edit in ordered])
        if not basis:
            continue
        per_layer[layer] = Ablation(
            basis=tuple(tuple(row) for row in basis), strength=strengths[0],
            # Carried, not honored per row: the orthonormalized basis no longer
            # corresponds one-to-one with the concepts that produced it, so a
            # layer whose edits declare different conventions can only be
            # REPORTED as mixed. Not a refusal like `AmbiguousStrength`,
            # because centering changes nothing about how the removal is
            # computed — it changes what the removed direction MEANS, which is
            # a reader's problem and belongs in the record.
            centering=scope_vocabulary.centering_summary(
                edit.centering for edit in ordered))
    return SubspaceAblator(per_layer) if per_layer else None


def interventions(edits: list[Edit],
                  prompt_token_count: int | None = None,
                  latent_edits: list[SAELatentEdit] | None = None
                  ) -> list[LayerIntervention]:
    """The chain: ablator, then SAE latent edits, then vector additions.

    ``prompt_token_count`` gates the injectors' and the latent intervention's
    chunked-prefill behaviour, and is deliberately not passed to the ablator,
    which applies at every position.

    **Additions consolidate when — and only when — their layers are distinct.**
    The historical shape was one ``VectorInjector`` per edit, and the hook loop
    runs EVERY armed intervention at EVERY layer: an 11-cell band on a 62-layer
    model therefore made 682 ``apply`` calls per decode step, 671 of which were
    dict misses returning ``h`` untouched. When every ADD edit targets a
    different layer, one injector holding all of them makes 62 calls for the
    same 11 effective injections. This is a bit-identity-preserving change, not
    an approximation: each layer's ``apply`` runs the same clone-and-add on the
    same inputs it ran before, and a layer reached by exactly one edit cannot
    tell which object dispatched it. Only the Python call count changes.

    When two or more edits SHARE a layer, that layer's edits keep the historical
    per-edit chain — deliberately, on two grounds. Numerically, ``(h + v₁) + v₂``
    is not bit-identical to ``h + (v₁ + v₂)`` in floating point, so folding a
    shared layer's vectors would silently change a condition's outputs. And a
    linear mix is a first-class hashed condition in this project (CLAUDE.md:
    "multiple at once ARE the linear mix ``h + Σ αᵢ·vᵢ``"), so its arithmetic is
    part of what a run's hash promises, not an implementation detail. Mixed
    conditions are rare; correctness there costs a few dispatches.

    Either way the composition contract is unchanged: additions commute in
    exact arithmetic, every injector still carries the same gate, and the chain
    still computes ``h + Δ_latent(h) + Σ αᵢ·vᵢ``.

    **Why the latent intervention sits before the injectors.** It is
    state-dependent: it encodes the residual stream through the SAE and its
    JumpReLU gate decides whether the feature is active at all. Running it
    ahead of the pure offsets means it reads the residual the MODEL would carry
    (post-ablation, pre-offset), so an experimenter's ``α·v`` can never move an
    SAE gate — and, as a direct consequence, the composed result is exactly
    ``h + Δ_latent(h) + Σ αᵢ·vᵢ``, the sum of each intervention's individual
    delta. Reversed, that additivity is lost and a mixed condition would mean
    something the manifest does not say. All of a layer's latent edits go into
    ONE intervention (one read of ``h``, deltas summed) for the same reason the
    ablator removes a layer's directions as one subspace.
    """
    chain: list[LayerIntervention] = []
    built = ablator(edits)
    if built is not None:
        chain.append(built)
    grouped = group_edits(list(latent_edits or []))
    if grouped:
        chain.append(SAELatentIntervention(
            grouped, prompt_token_count=prompt_token_count))
    additions = [edit for edit in edits if edit.mode == Mode.ADD]
    layers = [edit.layer for edit in additions]
    if additions and len(set(layers)) == len(layers):
        # Every ADD edit at its own layer — the common case (a sweep-promoted
        # band is one cell per layer). ONE injector, insertion-ordered, same
        # arithmetic per layer.
        chain.append(VectorInjector(
            {edit.layer: Injection(vector=edit.vector, alpha=edit.strength)
             for edit in additions},
            prompt_token_count=prompt_token_count))
    else:
        # A layer carries more than one edit: keep the historical per-edit
        # chain so the shared layer's additions apply in sequence, in the
        # caller's order, exactly as every previously run mixed condition did.
        for edit in additions:
            chain.append(VectorInjector(
                {edit.layer: Injection(vector=edit.vector, alpha=edit.strength)},
                prompt_token_count=prompt_token_count))
    return chain


def scope_inventory(edits: list[Edit],
                    prompt_token_count: int | None = None,
                    latent_edits: list[SAELatentEdit] | None = None
                    ) -> list[InterventionScope]:
    """The scope descriptors for the chain :func:`interventions` would build,
    in chain order (ablator first, then latent, then the injectors).

    Built BY building the chain and asking each member, never by re-deriving
    the description from the edits: a description assembled independently is a
    second implementation of the plan, and the two would eventually disagree
    about the thing this exists to state. The cost is one Gram-Schmidt over the
    ablated directions, paid once at run start.

    A condition that both ablates and adds yields TWO descriptors, in that
    order, saying two different things about token positions — which is the
    whole point: those two interventions are not one intervention with two
    doses, and a run that arms both cannot be described by either row alone.
    """
    return [item.scope() for item in interventions(
        edits, prompt_token_count=prompt_token_count,
        latent_edits=latent_edits)]


def satisfies_ordering_invariant(chain: list[LayerIntervention]) -> bool:
    """Whether a chain has at most one ablator, first. Exposed so tests can
    assert it of chains built anywhere, not only this module's output."""
    positions = [index for index, item in enumerate(chain)
                 if isinstance(item, SubspaceAblator)]
    return positions in ([], [0])


def satisfies_latent_ordering_invariant(chain: list[LayerIntervention]) -> bool:
    """Whether a chain has at most one latent intervention, placed after any
    ablator and before every vector injector.

    The additivity contract above depends on this ordering, so it is assertable
    of chains built anywhere, not only this module's output.
    """
    latent = [index for index, item in enumerate(chain)
              if isinstance(item, SAELatentIntervention)]
    if not latent:
        return True
    if len(latent) != 1:
        return False
    position = latent[0]
    ablators = [index for index, item in enumerate(chain)
                if isinstance(item, SubspaceAblator)]
    injectors = [index for index, item in enumerate(chain)
                 if isinstance(item, VectorInjector)]
    return (all(index < position for index in ablators)
            and all(index > position for index in injectors))
