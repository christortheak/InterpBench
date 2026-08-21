"""A residual-stream injector whose direction is a TRAINABLE parameter.

The deployed :class:`~steerlab_server.steering.injector.VectorInjector` rebuilds
its direction from a ``list[float]`` into a memoized leaf tensor — by design,
there is no autograd path through it. This class is its training-time twin: it
holds an unconstrained parameter ``u`` and computes the constrained direction

    v = alpha_absolute · u / ‖u‖

**inside** :meth:`apply`, so the projection onto the sphere of radius
``alpha_absolute`` is on the autograd tape and the gradient the optimizer sees
is already the sphere-tangent one. ``alpha_absolute`` is in ABSOLUTE units at
the layer; converting a norm-unit factor into absolute units is the driver's
job (the residual-norm denominator is a per-model calibration, not a property
of the injector).

**Training-only.** Nothing outside the optimizer should use this class: a
finished vector is exported with :meth:`constrained_vector` and deployed
through the standard ``VectorInjector``, which is what every measured run
already uses.

Equivalence argument (why a single teacher-forced full-sequence pass is a
faithful stand-in for stepped KV-cache decode)
---------------------------------------------------------------------------
Attention is causal, so the hidden state at position *p* of layer *L+1*
depends only on layer-*L* states at positions ``≤ p``. Injecting ``α·v`` into
the layer-*L* output at a chosen SET of positions in one full-sequence pass
therefore produces exactly the states a stepped decode would produce if the
same positions were injected as they were written into the KV cache — the
cache holds per-position states, and a full pass computes those same
per-position states in one go. Two consequences:

* ``position_mode="from_response"`` injects at each item's last non-pad PROMPT
  position — the position whose logits predict the answer token. In stepped
  decode the deployed injector fires at the true prompt end (and at every
  subsequent option step), and for a SINGLE-token forced-choice readout the
  prompt-end firing is the only injection that can affect the answer token's
  logits. So the training-time gradient is taken through exactly the
  computation the eval-time instrument scores. (Multi-token options would also
  need the option-step injections, which one teacher-forced pass over the
  prompt alone does not have — hence the driver's single-token training
  restriction.)
* ``position_mode="all"`` injects at every non-pad position. That is a
  deliberately DIFFERENT intervention from the deployed semantics — an
  explicit variant, not the default, because a vector optimized under ``all``
  and deployed under prompt-end semantics is a train/deploy mismatch.

This class does not gate on ``prompt_token_count`` the way ``VectorInjector``
does: it is built for single full-sequence passes with no KV cache, where the
driver knows every item's answer position exactly and supplies them.

The α→0 twin
------------
:class:`AdditiveDeltaProbe` shares this class's position machinery (the base
:class:`PositionedDelta`) but adds an UNCONSTRAINED, zero-initialized delta
instead of a sphere-projected direction. Adding exact zeros leaves the forward
pass numerically identical to an unsteered one while still building the tape,
so one forward+backward yields both the baseline readout and the gradient of
that readout with respect to an injection at the same position the deployed
injector fires at. That is the gradient survey's fast path
(:mod:`steerlab_server.experiment.optvec_gradient`) — a local sensitivity
reading, never a certified vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .intervention import LayerIntervention

#: The two position modes. ``from_response`` is the default and the one that
#: matches deployed injection semantics for a single-token readout.
POSITION_MODES = ("from_response", "all")


class TrainableInjectorError(RuntimeError):
    """A misuse of the training injector (unset batch, shape mismatch, …)."""


class PositionedDelta(LayerIntervention):
    """Shared machinery for the two training-time interventions: which layer,
    which positions, and the add itself.

    Subclasses supply the delta vector through :meth:`delta_vector`; everything
    about WHERE it lands — the two position modes, the per-batch declaration,
    the shape checks — lives here so the sphere-projected injector and the
    zero-initialized gradient probe cannot drift on injection semantics. (They
    must not: the probe reads the derivative of the very quantity the injector
    optimizes, at the same position.)
    """

    def __init__(self, *, layer: int, hidden_size: int,
                 position_mode: str = "from_response"):
        if position_mode not in POSITION_MODES:
            raise TrainableInjectorError(
                f"unknown position mode {position_mode!r} — one of "
                + ", ".join(POSITION_MODES))
        if hidden_size <= 0:
            raise TrainableInjectorError(
                f"hidden size must be positive (got {hidden_size})")
        self.layer = int(layer)
        self.hidden_size = int(hidden_size)
        self.position_mode = position_mode
        self._answer_positions: torch.Tensor | None = None
        self._attention_mask: torch.Tensor | None = None

    # ---------------------------------------------------------------- batch

    def set_batch(self, *, answer_positions: torch.Tensor | None = None,
                  attention_mask: torch.Tensor | None = None) -> None:
        """Declare the current forward pass's per-item positions.

        ``answer_positions`` is a ``[batch]`` LongTensor of the index each item
        is injected at under ``from_response``; ``attention_mask`` is the
        ``[batch, seq]`` 0/1 mask used under ``all``. The driver sets these
        immediately before each pass — the ``apply`` signature is fixed by the
        intervention protocol, so per-pass data cannot ride in as arguments.
        """
        self._answer_positions = (None if answer_positions is None
                                  else answer_positions.reshape(-1).long())
        self._attention_mask = attention_mask

    def clear_batch(self) -> None:
        self._answer_positions = None
        self._attention_mask = None

    # --------------------------------------------------------------- vector

    def delta_vector(self) -> torch.Tensor:
        """The ``[hidden]`` float32 vector added at the selected positions."""
        raise NotImplementedError

    # ---------------------------------------------------------------- apply

    def apply(self, h: torch.Tensor, layer: int, offset: int) -> torch.Tensor:
        if layer != self.layer:
            return h
        if h.dim() != 3:
            raise TrainableInjectorError(
                f"expected a [batch, seq, hidden] residual stream, got shape "
                f"{tuple(h.shape)}")
        batch, seq_len, hidden = h.shape
        if hidden != self.hidden_size:
            raise TrainableInjectorError(
                f"residual stream width {hidden} != injector hidden size "
                f"{self.hidden_size}")
        mask = self._position_mask(batch, seq_len, h.device)
        v = self.delta_vector()
        # float32 throughout, cast once at the add: the residual stream may be
        # bf16, but the parameter and its gradient never are.
        delta = mask.unsqueeze(-1) * v
        return h + delta.to(h.dtype)

    def _position_mask(self, batch: int, seq_len: int,
                       device: torch.device) -> torch.Tensor:
        if self.position_mode == "all":
            if self._attention_mask is None:
                raise TrainableInjectorError(
                    "position mode 'all' needs an attention mask — call "
                    "set_batch(attention_mask=…) before the forward pass")
            mask = self._attention_mask.to(device=device, dtype=torch.float32)
            if mask.shape != (batch, seq_len):
                raise TrainableInjectorError(
                    f"attention mask shape {tuple(mask.shape)} does not match "
                    f"the pass's [batch, seq] = ({batch}, {seq_len})")
            return mask
        if self._answer_positions is None:
            raise TrainableInjectorError(
                "position mode 'from_response' needs per-item answer "
                "positions — call set_batch(answer_positions=…) before the "
                "forward pass")
        positions = self._answer_positions.to(device)
        if positions.numel() != batch:
            raise TrainableInjectorError(
                f"{positions.numel()} answer positions for a batch of {batch}")
        if int(positions.min()) < 0 or int(positions.max()) >= seq_len:
            raise TrainableInjectorError(
                f"answer positions {positions.tolist()} fall outside the "
                f"pass's {seq_len} positions")
        mask = torch.zeros((batch, seq_len), dtype=torch.float32, device=device)
        mask[torch.arange(batch, device=device), positions] = 1.0
        return mask


class TrainableVectorInjector(PositionedDelta):
    """Adds ``alpha_absolute · u/‖u‖`` at selected positions of one layer.

    Args:
        layer: the decoder block whose OUTPUT is steered (residual after the
            block, matching ``VectorInjector`` and the hook manager).
        hidden_size: ``d_model``.
        alpha_absolute: the constrained direction's L2 norm, in absolute units
            at this layer.
        position_mode: one of :data:`POSITION_MODES`.
        u: optional initial unconstrained parameter (any nonzero vector); a
            seeded random init is used when absent.
        generator: RNG for the default init, so a run's seed reaches it.
    """

    def __init__(self, *, layer: int, hidden_size: int, alpha_absolute: float,
                 position_mode: str = "from_response",
                 u: torch.Tensor | None = None,
                 device: torch.device | str | None = None,
                 generator: torch.Generator | None = None):
        super().__init__(layer=layer, hidden_size=hidden_size,
                         position_mode=position_mode)
        hidden_size = self.hidden_size
        if not (alpha_absolute > 0):
            raise TrainableInjectorError(
                f"alpha_absolute must be positive (got {alpha_absolute}) — it "
                "is the L2 norm the direction is projected onto")
        self.alpha_absolute = float(alpha_absolute)
        if u is None:
            init = torch.randn(hidden_size, dtype=torch.float32,
                               generator=generator)
        else:
            init = torch.as_tensor(u, dtype=torch.float32).reshape(-1).clone()
            if init.numel() != hidden_size:
                raise TrainableInjectorError(
                    f"initial u has {init.numel()} elements, hidden size is "
                    f"{hidden_size}")
        if float(init.norm()) == 0.0:
            raise TrainableInjectorError(
                "initial u is all zeros — u/‖u‖ is undefined there")
        if device is not None:
            init = init.to(device)
        # Always float32: the sphere projection and the optimizer state are
        # numerically delicate next to a bf16 residual stream, and the cast to
        # the stream's dtype happens at the add.
        self.u = nn.Parameter(init)

    # --------------------------------------------------------------- vector

    def vector(self) -> torch.Tensor:
        """The constrained direction ``alpha · u/‖u‖`` (float32, on the tape)."""
        return self.alpha_absolute * self.u / self.u.norm()

    def constrained_vector(self) -> list[float]:
        """The constrained direction as a plain float32 list — the handoff to
        ``VectorInjector`` / ``vector_store``, detached from the tape."""
        with torch.no_grad():
            return self.vector().detach().to(torch.float32).cpu().tolist()

    def delta_vector(self) -> torch.Tensor:
        return self.vector()


class AdditiveDeltaProbe(PositionedDelta):
    """Adds an UNCONSTRAINED delta (zero by default) at the same positions.

    The α→0 reading instrument. Two properties make it what the gradient survey
    needs:

    * **Zero init is numerically inert.** ``h + 0`` is exactly ``h`` in IEEE
      arithmetic, so the pass that carries the tape is bit-identical to an
      unsteered pass — the baseline readout and the gradient of that readout
      come out of ONE forward+backward, with no separate baseline pass to drift
      against.
    * **No sphere projection.** :class:`TrainableVectorInjector` differentiates
      through ``α·u/‖u‖``, whose gradient is the sphere-TANGENT one (the radial
      component is projected out, by design — the optimizer moves on a fixed-α
      sphere). A local sensitivity reading wants the raw
      ``∂margin/∂h`` at the injection position, which is what an unconstrained
      additive delta gives. ``u`` is undefined at zero anyway.

    ``delta`` may be preset to a fixed vector, which turns the same object into
    an exact evaluator of "what does the margin do if I add THIS at the
    injection position" — the finite-difference check's other half.
    """

    def __init__(self, *, layer: int, hidden_size: int,
                 position_mode: str = "from_response",
                 delta: torch.Tensor | None = None,
                 device: torch.device | str | None = None,
                 requires_grad: bool = True):
        super().__init__(layer=layer, hidden_size=hidden_size,
                         position_mode=position_mode)
        if delta is None:
            init = torch.zeros(self.hidden_size, dtype=torch.float32)
        else:
            init = torch.as_tensor(delta, dtype=torch.float32).reshape(-1).clone()
            if init.numel() != self.hidden_size:
                raise TrainableInjectorError(
                    f"delta has {init.numel()} elements, hidden size is "
                    f"{self.hidden_size}")
        if device is not None:
            init = init.to(device)
        self.delta = nn.Parameter(init, requires_grad=requires_grad)

    def delta_vector(self) -> torch.Tensor:
        return self.delta

    def gradient(self) -> torch.Tensor:
        """``∂(whatever was backwarded)/∂delta`` as a detached CPU float32
        row, or zeros when no backward has run (never ``None``: a caller
        normalizing a gradient must be able to see that it is zero)."""
        if self.delta.grad is None:
            return torch.zeros(self.hidden_size, dtype=torch.float32)
        return self.delta.grad.detach().to(torch.float32).cpu().clone()
