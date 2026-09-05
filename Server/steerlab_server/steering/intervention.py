"""The residual-stream intervention protocol (parallel to Swift
``LayerIntervention`` + ``InterventionHookable``).

A vendored MLX model called ``apply(h, layer, offset)`` between transformer
blocks. Here, the equivalent is a forward hook registered on each HF decoder
block (see :mod:`steerlab_server.steering.hooks`), which invokes every active
intervention in order. Two conforming types cover everything, exactly as on the
Swift side: :class:`~steerlab_server.steering.injector.VectorInjector` (adds
``alpha·v``) and :class:`~steerlab_server.steering.recorder.ActivationRecorder`
(captures for extraction). Injectors compose additively, so a list of them *is*
the linear mix ``h + Σ αᵢ·vᵢ``.

Scope descriptors
-----------------

The protocol is one method, which is exactly why the four intervention PATHS
this engine can arm look interchangeable from outside and are not: the additive
path edits the final prompt position and each decode step's last position, the
ablation path edits EVERY position at its layers, and the training-time
injector edits one teacher-forced pass's answer positions (or all of them). One
sentence cannot describe all four, and a report that uses one sentence for all
four is claiming something no run performed.

:class:`InterventionScope` is that description, as data: a frozen record of
site, layers, token positions, prefill/decode behaviour, centering convention,
dose units, the matched control this codebase actually builds for the path, and
the path's claim limits — every field a plain string a report can print
verbatim. Each intervention class fills one from its OWN configuration
(:meth:`VectorInjector.scope`, :meth:`SubspaceAblator.scope`,
:meth:`PositionedDelta.scope`, :meth:`SAELatentIntervention.scope`), so a
descriptor cannot drift from the object that runs; the position claims are
pinned to behaviour by the tests each ``scope`` docstring names, not by these
strings agreeing with each other. ``steering.plan.scope_inventory`` describes a
whole condition's chain in chain order, and the run driver stamps that
inventory into ``intervention-scope.json`` beside the run's other provenance.

Prose for readers is ``docs/INTERVENTION-SCOPE.md``; these constants are its
machine half, and the two are meant to be edited together.

**Swift parity owed.** The BEHAVIOUR these descriptors describe is already
mirrored — ``Sources/SteeringKit/Injection/VectorInjector.swift`` (same
last-position firing and the same ``shouldInject`` gate),
``SubspaceAblator.swift`` (same every-position removal, same dependence
tolerance) and ``InterventionPlan`` (same ablator-first chain). The DESCRIPTOR
is Python-only today: Swift has no ``InterventionScope``, so a Swift-executed
run stamps no ``intervention-scope.json``. Porting these constants verbatim is
the parity item; until it lands, a reader must not take the sidecar's absence
as a claim about what a Mac run did.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# --- path vocabulary -------------------------------------------------------

#: ``h + α·v`` under stepped decode — :class:`VectorInjector`.
ADDITIVE = "additive"
#: ``h − λ·P h`` at every position — :class:`SubspaceAblator`.
ABLATION = "ablation"
#: The training-time injector / gradient probe — one teacher-forced pass.
TRAINABLE_ADDITIVE = "trainableAdditive"
#: Encode → edit one latent → decode only the induced delta.
SAE_LATENT = "saeLatent"
PATHS = (ADDITIVE, ABLATION, TRAINABLE_ADDITIVE, SAE_LATENT)

# --- centering vocabulary --------------------------------------------------

#: The declaration sites spell an absent centering exactly this way
#: (``str(inj.get("centering") or "none")`` in ``experiment.model_variant`` and
#: ``api.routes``), so absent-means-``none`` here too rather than
#: absent-means-unknown.
CENTERING_NONE = "none"
CENTERING_NEUTRAL_MEAN = "neutralMean"
#: Centering is an ablation-DIRECTION transform; this engine refuses it on a
#: steering injection, so the additive paths report neither "none" (which would
#: imply the choice existed and was declined) nor a value.
CENTERING_NOT_APPLICABLE = "notApplicable"

# --- site ------------------------------------------------------------------

#: Every path in this package edits the same tensor. Stated once so no
#: descriptor can imply an attention- or MLP-internal hook that nothing here
#: registers.
SITE_BLOCK_OUTPUT = (
    "residual stream at the block output of the named layer(s) — the tensor a "
    "forward hook receives from decoder block L, before block L+1 reads it")

# --- position statements ---------------------------------------------------

POSITIONS_LAST_GATED = (
    "final prompt position on the last prefill chunk, then the last position "
    "of every decode pass")
POSITIONS_LAST_UNGATED = (
    "the last position of every forward pass — no prompt length was supplied, "
    "so a chunked prefill would also fire at each intermediate chunk's tail")
POSITIONS_EVERY = "every position, prefill and decode"
POSITIONS_ANSWER_TEACHER_FORCED = (
    "each item's answer position in one teacher-forced full-sequence pass")
POSITIONS_ALL_TEACHER_FORCED = (
    "every non-pad position in one teacher-forced pass")

PREFILL_GATED = (
    "fires once, at the true prompt end; suppressed on every prefill chunk "
    "whose last position is mid-prompt")
PREFILL_UNGATED = (
    "fires at the last position of every prefill chunk — correct only when "
    "prefill is guaranteed single-chunk")
PREFILL_EVERY_POSITION = (
    "applies at every position of every prefill chunk — deliberately ungated, "
    "the inverse of the additive path's chunked-prefill gate")
PREFILL_TEACHER_FORCED = (
    "no prefill/decode split: one full-sequence pass with no KV cache")

DECODE_LAST_POSITION = (
    "fires on every decode pass, at that pass's single new position")
DECODE_EVERY_POSITION = "applies at every position of every decode pass"
DECODE_NOT_RUN = (
    "never runs under stepped decode — training only; the finished vector "
    "deploys through VectorInjector, whose additive row is what a measured run "
    "executes")

# --- dose units ------------------------------------------------------------

DOSE_UNITS_ALPHA = (
    "α at the layer, applied here as an absolute offset; a study declaring "
    "alphaInNormUnits converts its declared α through that layer's "
    "residual-norm denominator — the study's declared denominator convention — "
    "before the vector reaches this class")
DOSE_UNITS_LAMBDA = (
    "λ, a dimensionless fraction of the projected component (1 = full removal, "
    "2 = reflection); no residual-norm denominator, because the removal is "
    "already scaled by whatever the stream carries")
DOSE_UNITS_ALPHA_ABSOLUTE = (
    "α as an absolute L2 norm at the layer — the direction is projected onto "
    "that sphere inside apply(); converting a norm-unit α into absolute units "
    "is the training driver's job")
DOSE_UNITS_BETA_LATENT = (
    "β in LATENT units — this feature's own activation scale, set by the "
    "dictionary's normalization. Not α, not residual-norm units, and not "
    "comparable across features")
DOSE_UNITS_PROBE = (
    "none — the delta is zero-initialized; this is a local sensitivity probe, "
    "not a dose")

# --- controls (what this codebase actually builds, by name) ----------------

CONTROL_RANDOM_MATCHED_NORM = (
    "randomMatchedNorm: the same layers at the same α, with a seeded isotropic-"
    "Gaussian direction rescaled to the concept vector's L2 norm at each layer "
    "(algorithm stamp 'gaussian-isotropic-v1'); scaffolded by "
    "experiment.control_matrix.control_matrix_conditions, substituted by "
    "experiment.tasks._condition_injections")
CONTROL_RANDOM_DIRECTION_ABLATION = (
    "randomDirectionAblation: the same layers at the same λ with a random "
    "DIRECTION removed instead. Norm matching means nothing to a projection, "
    "so the control varies the direction and asks whether removing any rank-1 "
    "subspace does this (experiment.control_matrix.ablation_control_conditions)")
CONTROL_TRAINABLE = (
    "declared by the confirm study, never synthesized at run time: the S0 "
    "shuffled-target null (the identical optimization against permuted labels) "
    "alongside the ordinary randomMatchedNorm cell "
    "(experiment.control_matrix.optvec_confirm_conditions)")
CONTROL_SAE_LATENT = (
    "none synthesized by this engine — a latent arm's contrast is the baseline "
    "arm, and clamp β=0 is the per-feature removal cell; dose calibration "
    "against a pinned corpus is declared future work")

# --- claim limits ----------------------------------------------------------

CLAIM_LIMITS_ADDITIVE = (
    "Adds a fixed offset at these positions and nowhere else. A behavioural "
    "change shows the offset moved this model's output here; it does not "
    "locate the concept, and equal residual-norm doses are not equal potency "
    "across layers or models.")
CLAIM_LIMITS_ABLATION = (
    "Removes the component along the ablated subspace at this site only. It "
    "does not show the model cannot represent the concept — in another basis, "
    "at another site, or reconstructed by the layers above — and λ is not "
    "comparable to α.")
CLAIM_LIMITS_TRAINABLE = (
    "A direction selected ON behaviour is a screen, not a result: it is one "
    "sample from an equivalence class, so only loadings stable across seeds "
    "are interpretable, and the citable numbers come from a separate confirm "
    "study run through the deployed additive path.")
CLAIM_LIMITS_PROBE = (
    "A local sensitivity reading at one position and one dose — the derivative "
    "of the readout with respect to an offset there. Never a certified vector, "
    "and never evidence about behaviour at any other dose.")
CLAIM_LIMITS_SAE_LATENT = (
    "Edits one dictionary feature's latent activation and decodes only the "
    "induced delta at this site. It shows what THIS dictionary's feature does "
    "here; β is in latent units, comparable neither across features nor to α.")


def centering_summary(declarations) -> str:
    """One printable centering string for a set of per-edit declarations.

    A layer's ablated directions are orthonormalized into a basis whose rows no
    longer correspond to the concepts that produced them, so "this row was
    centered and that one was not" is not a statement the ablator can act on —
    only one it can REPORT. A uniform declaration prints itself; a mixed one
    prints as mixed rather than silently picking a winner, which is the shape
    a reader has to see to know the run's directions were not all expressed in
    one convention.
    """
    values = sorted({str(value) for value in declarations})
    if not values:
        return CENTERING_NONE
    if len(values) == 1:
        return values[0]
    return "mixed(" + ",".join(values) + ")"


@dataclass(frozen=True)
class InterventionScope:
    """What one armed intervention actually changes — printable verbatim.

    Every field is a plain string (or a plain list of layer indices) so a
    report, a manifest reader, or a methods section can quote it without
    interpreting anything. The class holds no defaults for the prose fields on
    purpose: a descriptor is filled by the object that runs, and a default
    would let a new path inherit another path's claims.

    ``detail`` carries the path-specific facts that are numbers rather than
    prose (per-layer α, per-layer rank and λ, the gate's state). It is always
    present, possibly empty, so a reader never has to distinguish "no detail"
    from "an older writer".
    """

    path: str
    site: str
    layers: tuple[int, ...]
    positions: str
    prefill: str
    decode: str
    centering: str
    dose_units: str
    control: str
    claim_limits: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-safe, camelCase — the shape stamped into a run's sidecar."""
        return {
            "path": self.path,
            "site": self.site,
            "layers": [int(layer) for layer in self.layers],
            "positions": self.positions,
            "prefill": self.prefill,
            "decode": self.decode,
            "centering": self.centering,
            "doseUnits": self.dose_units,
            "control": self.control,
            "claimLimits": self.claim_limits,
            "detail": dict(self.detail),
        }


class LayerIntervention:
    """Base class for a residual-stream hook.

    Subclasses override :meth:`apply`, and every subclass that EDITS the
    residual stream also overrides :meth:`scope` — a recorder does not, because
    it changes nothing and has nothing to scope.

    Args:
        h: hidden state ``[batch, seq, hidden]``.
        layer: index of the transformer block whose output this is.
        offset: KV-cache position of the first token of this forward pass
            (0 = prefill). Decode steps have ``offset = prompt_len + step``.
    Returns:
        The (possibly modified) hidden state. Recorders return ``h`` unchanged.
    """

    def apply(self, h: torch.Tensor, layer: int, offset: int) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def scope(self) -> InterventionScope:  # pragma: no cover
        """This intervention's scope descriptor, filled from its own state.

        Raising is deliberate: a chain member that cannot say what it changes
        must not be described by a neighbour's sentence.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not describe its intervention scope")
