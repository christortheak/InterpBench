"""True SAE **latent** intervention — encode, edit one latent, decode ONLY the
induced delta (SAE-VECTOR-INTERVENTION proposal r2 §8 P2-9).

This is a different mechanism from steering along an SAE decoder direction, and
the two are never conflated. Decoder-direction addition is ``h + α·W_dec[f]``:
an offset the model's own sparse code never sees. A latent intervention runs the
model's residual state through the SAE encoder, edits *that feature's latent
activation*, and puts back only what the edit induced::

    z_f  = act(W_enc[:, f] · h + b_enc[f])     # this feature's latent activation
    z'_f = clamp_or_add(z_f, β)                # the edit, in LATENT units
    h'   = h + (z'_f − z_f) · W_dec[f]         # decode ONLY the induced delta

**Only the delta is decoded, so the SAE's reconstruction error never enters the
residual stream.** A full encode→decode round trip would replace ``h`` with
``h_hat = z·W_dec + b_dec``, silently adding ``h − h_hat`` (everything the
dictionary fails to explain, which at Gemma-Scope sparsities is a large fraction
of the norm) to every steered position. That error would be a second,
unmeasured intervention riding along with the one the study declares. The delta
form makes the edit exactly what the condition says it is: when ``β`` induces no
change in the latent, the residual stream is bit-identical to baseline.

**Gemma Scope 2 SAEs are JumpReLU.** The per-feature activation is

    act(pre) = pre  if  pre > threshold[f]  else  0

with a *per-feature* threshold published alongside the weights (implemented here
as ``mask · relu(pre)``, the upstream Gemma Scope form, so a hypothetical
negative threshold behaves identically to the reference implementation).

Mode semantics (exact, and both tested — see ``Server/tests/test_sae_latent.py``)
--------------------------------------------------------------------------------

``β`` is in **LATENT units** — the feature's own activation scale, set by the
dictionary's normalization. It is NOT ``α``, it is NOT in residual-norm units,
and it is not comparable across features. Provenance stamps ``latentUnits: true``
so no reader mistakes it for a dose in norm units. (Dose CALIBRATION for latent
interventions — measuring each feature's activation distribution on a pinned
corpus so ``β`` can be expressed as a percentile — is deliberately future work;
nothing here reads a corpus.)

``"add"`` — the edit is applied to the **pre-activation** and the SAE's own gate
is then re-evaluated: ``z'_f = act(pre_f + β)``. Consequences, all deliberate:

* an already-active feature (``pre > θ``) that stays active gains exactly ``β``,
  so the induced delta is ``β·W_dec[f]`` — in that regime, and only in that
  regime, ``add`` coincides with decoder-direction addition at raw ``α = β``;
* a **dormant** feature (``pre ≤ θ``) is activated only if ``β`` clears the gate
  (``pre + β > θ``), in which case the delta is the whole ``(pre+β)·W_dec[f]``,
  not ``β·W_dec[f]``. A feature dormant by more than ``β`` is a **no-op**: the
  edit changes nothing, and the record shows a zero delta rather than a
  fabricated activation;
* a negative ``β`` can push an active feature below the gate, giving
  ``z' = 0`` and a delta of ``−z_f`` — full removal of the feature's
  contribution.

Post-activation addition (``z' = z + β``) was rejected: it is *identically*
``h + β·W_dec[f]`` for every state, dormant or not, which is exactly the
decoder-direction addition this condition type must never be conflated with. It
would have shipped two names for one edit.

``"clamp"`` — ``z'_f = β``, unconditionally, whatever the state:

* ``z_f = 0`` (dormant) → delta ``β·W_dec[f]``: clamp **activates** a dormant
  feature. This is the mode's whole point, and the reason it is not ``add``;
* ``β = 0`` → delta ``−z_f·W_dec[f]``: removes exactly this feature's
  contribution wherever it is present (per-feature ablation);
* ``β < 0`` is REFUSED at manifest validation: a JumpReLU latent is
  non-negative by construction, so a negative clamp target is outside the
  dictionary's range and the decoded delta would not correspond to any state
  the SAE can represent.

Firing rule (the standing hard requirement)
-------------------------------------------

Identical to :class:`~steerlab_server.steering.injector.VectorInjector` in every
behavioural respect — it reuses that class's gate arithmetic rather than
restating it: fires at the final prompt position and at the last position of
every decode pass, suppressed on prefill chunks whose tail is mid-prompt.
Steering only during prefill silently produces near-null results; this is the
classic bug both classes are shaped to prevent.

Composition
-----------

**All of a layer's latent edits are computed from ONE read of the residual
stream and their deltas summed** — the same rule, for the same reason, as the
ablator's "remove a layer's directions as one subspace". A cascade (feature *f*
edited, then feature *g* encoded against the already-edited state) would make a
multi-feature latent condition order-dependent and would let one feature's
decoded delta silently move another feature's gate.

:mod:`steerlab_server.steering.plan` places the latent intervention BEFORE the
vector injectors, which makes the chain exactly additive: the latent edit reads
the residual stream the model itself would carry (post-ablation, pre-offset), and
the injectors' ``Σ αᵢ·vᵢ`` is a pure offset on top. Composed result is
``h + Δ_latent(h) + Σ αᵢ·vᵢ`` = the sum of each intervention's individual delta,
for arbitrary vectors. That ordering is a contract, not an accident (reversing it
would let an injected offset move the SAE gate), and it is asserted by test.

The residual stream is float (bf16/fp16) even for quantized models: the encode /
edit / decode arithmetic runs in float32 and only the final delta is cast back to
``h.dtype``. Quantized weights are never touched.

Schema half
-----------

The **declared** surface — ``ADD`` / ``CLAMP`` / ``MODES`` and the
``SAELatentFeature`` / ``SAELatentEdit`` data classes — lives in
:mod:`steerlab_server.steering.sae_latent_schema`, which imports no torch, and
is re-exported here by name. Manifest validation needs that vocabulary and
nothing else; this file needs tensors. See the schema module's docstring for
why the two are separated (portability gap G7).
"""

from __future__ import annotations

import torch

from .injector import VectorInjector
from .intervention import LayerIntervention
# The declared surface — mode vocabulary and the two data classes — lives in a
# torch-free module so that manifest VALIDATION (`experiment.sae_latent`, and
# through it `Manifest.verify`) can reach it without importing the execution
# stack this file sits on. Re-exported here by name, so every existing
# `from .sae_latent import SAELatentEdit` (and the experiment-side imports)
# keeps resolving exactly as before. The arrow points execution → schema and
# never back: `sae_latent_schema` imports nothing from this module.
from .sae_latent_schema import (ADD, CLAMP, MODES, SAELatentEdit,
                                SAELatentFeature)

__all__ = ["ADD", "CLAMP", "MODES", "SAELatentEdit", "SAELatentFeature",
           "SAELatentIntervention", "group_edits"]


class SAELatentIntervention(LayerIntervention):
    """Applies every declared latent edit of a layer from ONE read of ``h``.

    Behavioural twin of :class:`VectorInjector`: same firing gate (reused, not
    reimplemented), same last-position-only application, same per-(device,
    dtype, layer) tensor cache, same "clone, never mutate" rule.
    """

    def __init__(self, edits: dict[int, list[SAELatentEdit]],
                 prompt_token_count: int | None = None):
        """Layer index → the edits applied to that block's output.

        ``prompt_token_count`` (when known) suppresses the edit on prefill
        chunks whose last position is mid-prompt. ``None`` preserves
        last-position-of-every-pass behaviour, correct only when prefill is
        guaranteed single-chunk.
        """
        self._edits: dict[int, list[SAELatentEdit]] = {
            int(layer): list(group) for layer, group in edits.items() if group}
        self._prompt_token_count = prompt_token_count
        # Materialized lazily per (device, dtype, layer) the first time the
        # hook fires, exactly like VectorInjector. The cached tensors are
        # float32 (the arithmetic runs there whatever h's dtype); dtype stays
        # in the key for parity with the injector's cache shape, and because a
        # pass in a second dtype is a different pass, not a cache hit.
        self._tensor_cache: dict[tuple, tuple[torch.Tensor, ...]] = {}

    @classmethod
    def single(cls, edit: SAELatentEdit,
               prompt_token_count: int | None = None) -> "SAELatentIntervention":
        return cls({edit.layer: [edit]}, prompt_token_count=prompt_token_count)

    @staticmethod
    def should_inject(offset: int, seq_len: int,
                      prompt_token_count: int | None) -> bool:
        """The chunked-prefill gate — literally
        :meth:`VectorInjector.should_inject`.

        Delegated rather than duplicated: two copies of this arithmetic are two
        chances for the prefill-only bug to come back on one of them.
        """
        return VectorInjector.should_inject(offset, seq_len, prompt_token_count)

    def _tensors(self, layer: int, group: list[SAELatentEdit],
                 h: torch.Tensor) -> tuple[torch.Tensor, ...]:
        key = (h.device, h.dtype, layer)
        cached = self._tensor_cache.get(key)
        if cached is not None:
            return cached
        hidden = h.shape[-1]
        for edit in group:
            if edit.feature.hidden_size != hidden:
                raise ValueError(
                    f"SAE latent edit at layer {layer} (feature "
                    f"{edit.feature_id}) has dimension "
                    f"{edit.feature.hidden_size} but the residual stream is "
                    f"{hidden}-dimensional — wrong SAE for this model")
        f32 = torch.float32
        # [hidden, n_edits] / [n_edits, hidden]: one matmul covers the whole
        # group, and the group is encoded from a single read of h.
        w_enc = torch.tensor([list(e.feature.encoder_row) for e in group],
                             device=h.device, dtype=f32).T.contiguous()
        w_dec = torch.tensor([list(e.feature.decoder_row) for e in group],
                             device=h.device, dtype=f32)
        bias = torch.tensor([e.feature.encoder_bias for e in group],
                            device=h.device, dtype=f32)
        threshold = torch.tensor([e.feature.threshold for e in group],
                                 device=h.device, dtype=f32)
        beta = torch.tensor([e.beta for e in group], device=h.device, dtype=f32)
        is_clamp = torch.tensor([e.mode == CLAMP for e in group],
                                device=h.device, dtype=torch.bool)
        built = (w_enc, w_dec, bias, threshold, beta, is_clamp)
        self._tensor_cache[key] = built
        return built

    @staticmethod
    def _jumprelu(pre: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """``mask · relu(pre)`` — the upstream Gemma Scope JumpReLU, verbatim."""
        return torch.where(pre > threshold, torch.relu(pre),
                           torch.zeros_like(pre))

    def apply(self, h: torch.Tensor, layer: int, offset: int) -> torch.Tensor:
        group = self._edits.get(layer)
        if not group:
            return h
        seq_len = h.shape[1]
        if not self.should_inject(offset, seq_len, self._prompt_token_count):
            return h  # intermediate prefill chunk: its tail is mid-prompt

        w_enc, w_dec, bias, threshold, beta, is_clamp = self._tensors(
            layer, group, h)
        last_index = seq_len - 1
        # ONE read of the residual stream for the whole group (see the module
        # docstring): every feature's pre-activation is computed against the
        # same state, so the deltas sum instead of cascading.
        state = h[:, last_index, :].to(torch.float32)      # [batch, hidden]
        pre = state @ w_enc + bias                         # [batch, n_edits]
        z = self._jumprelu(pre, threshold)
        # add: edit the PRE-activation and re-evaluate the gate.
        # clamp: set the POST-activation latent outright.
        z_add = self._jumprelu(pre + beta, threshold)
        z_new = torch.where(is_clamp, beta.expand_as(z), z_add)
        delta = (z_new - z) @ w_dec                        # [batch, hidden]

        # Clone so we never mutate a tensor the model may reuse.
        out = h.clone()
        out[:, last_index, :] = out[:, last_index, :] + delta.to(h.dtype)
        return out

    # --- diagnostics ------------------------------------------------------

    def latent_readout(self, h: torch.Tensor, layer: int) -> dict:
        """Pre-activation, gated latent, edited latent and induced delta norm
        at the last position — read-only, for tests and diagnostics.

        Never called from the hook path: a run's provenance records the
        DECLARED parameters, and measuring per-token activations is a separate
        instrument, not a side effect of steering.
        """
        group = self._edits.get(layer)
        if not group:
            return {}
        w_enc, w_dec, bias, threshold, beta, is_clamp = self._tensors(
            layer, group, h)
        state = h[:, h.shape[1] - 1, :].to(torch.float32)
        pre = state @ w_enc + bias
        z = self._jumprelu(pre, threshold)
        z_new = torch.where(is_clamp, beta.expand_as(z),
                            self._jumprelu(pre + beta, threshold))
        delta = (z_new - z) @ w_dec
        return {"preActivation": pre.tolist(), "latent": z.tolist(),
                "editedLatent": z_new.tolist(),
                "deltaNorm": float(delta.norm().item())}


def group_edits(edits: list[SAELatentEdit]) -> dict[int, list[SAELatentEdit]]:
    """Group edits by layer, ordered by ``(feature_id, label)``.

    A stable, data-derived order matters for the same reason the ablator sorts
    its basis: the deltas sum, so the RESULT is order-free, but float32 addition
    is not associative and a reproducible run should not depend on how a caller
    happened to build its list.
    """
    grouped: dict[int, list[SAELatentEdit]] = {}
    for edit in edits:
        grouped.setdefault(int(edit.layer), []).append(edit)
    for layer in grouped:
        grouped[layer] = sorted(grouped[layer],
                                key=lambda e: (e.feature_id, e.label))
    return grouped
