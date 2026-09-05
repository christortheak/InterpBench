"""Adds ``alpha · vector`` to the residual stream (parallel to Swift
``VectorInjector``).

Injection fires at the final prompt position and the last position of every
decode pass. **Firing on every decode pass is load-bearing**: steering only
during prefill silently produces near-null results — the classic bug this
design guards against. The converse bug is firing too often: a chunked prefill
makes "last position of the pass" a mid-prompt token for every chunk but the
final one, so injection is gated on ``prompt_token_count`` when the prompt
length is known.

The residual stream is float (bf16/fp16) even for quantized models — we cast
the vector to ``h.dtype`` and never touch quantized weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import intervention as scope_vocabulary
from .intervention import InterventionScope, LayerIntervention


@dataclass(frozen=True)
class Injection:
    """One steering direction + strength applied at a layer."""

    vector: list[float]  # length == model hidden size
    alpha: float  # report in units of the typical residual-stream norm


class VectorInjector(LayerIntervention):
    def __init__(self, injections: dict[int, Injection],
                 prompt_token_count: int | None = None):
        """Layer index → injection applied to that block's output.

        The dict may hold ONE layer or every distinct layer a condition
        steers: :func:`steerlab_server.steering.plan.interventions` consolidates
        a condition's ADD edits into a single injector whenever their layers are
        distinct, because the per-layer arithmetic is unchanged and the hook
        loop then makes one ``apply`` call per layer instead of one per cell per
        layer. Two edits at the SAME layer must stay separate injectors — see
        that function for why.

        ``prompt_token_count`` (when known) suppresses injection on prefill
        chunks whose last position is mid-prompt. ``None`` preserves
        last-position-of-every-pass behavior, correct only when prefill is
        guaranteed single-chunk.
        """
        self._injections = dict(injections)
        self._prompt_token_count = prompt_token_count
        # Materialized lazily per (device, dtype) the first time the hook fires.
        self._tensor_cache: dict[tuple, torch.Tensor] = {}

    @classmethod
    def single(cls, layer: int, vector: list[float], alpha: float,
               prompt_token_count: int | None = None) -> "VectorInjector":
        return cls({layer: Injection(vector=vector, alpha=alpha)},
                   prompt_token_count=prompt_token_count)

    @staticmethod
    def should_inject(offset: int, seq_len: int, prompt_token_count: int | None) -> bool:
        """Whether a pass over absolute positions ``offset ..< offset+seq_len``
        should be injected at its last position. Pure, so the chunked-prefill
        gating is unit-testable without a model. Mirrors Swift
        ``VectorInjector.shouldInject``.
        """
        if prompt_token_count is None:
            return True
        return offset + seq_len - 1 >= prompt_token_count - 1

    def scope(self) -> InterventionScope:
        """This injector's scope descriptor (see ``docs/INTERVENTION-SCOPE.md``).

        The position claims are the module docstring's, restated as data — and
        they are pinned to BEHAVIOUR, not to this method, by tests that already
        exist: ``tests/test_injection_fires_per_token.py``
        ``::test_apply_adds_alpha_v_at_last_position_only`` (only the last
        position moves), ``::test_should_not_inject_on_mid_prompt_prefill_chunk``
        and ``::test_should_inject_single_chunk_prefill_then_decode`` (the gate),
        ``::test_injection_fires_on_every_decode_step_of_a_sampled_generation``
        (the decode row). ``tests/test_intervention_scope.py`` asserts that
        those behaviours and these strings still describe each other.

        The gate's STATE is part of the description because the same class
        means two different things with and without it: ``prompt_token_count``
        absent is the legacy shape, correct only when prefill is single-chunk,
        and a descriptor that hid that would let a chunked-prefill run be
        reported as a prompt-end injection.
        """
        gated = self._prompt_token_count is not None
        return InterventionScope(
            path=scope_vocabulary.ADDITIVE,
            site=scope_vocabulary.SITE_BLOCK_OUTPUT,
            layers=tuple(sorted(self._injections)),
            positions=(scope_vocabulary.POSITIONS_LAST_GATED if gated
                       else scope_vocabulary.POSITIONS_LAST_UNGATED),
            prefill=(scope_vocabulary.PREFILL_GATED if gated
                     else scope_vocabulary.PREFILL_UNGATED),
            decode=scope_vocabulary.DECODE_LAST_POSITION,
            centering=scope_vocabulary.CENTERING_NOT_APPLICABLE,
            dose_units=scope_vocabulary.DOSE_UNITS_ALPHA,
            control=scope_vocabulary.CONTROL_RANDOM_MATCHED_NORM,
            claim_limits=scope_vocabulary.CLAIM_LIMITS_ADDITIVE,
            detail={
                "chunkedPrefillGate": "promptTokenCount" if gated else "none",
                "promptTokenCount": self._prompt_token_count,
                "alphaPerLayer": {str(layer): float(self._injections[layer].alpha)
                                  for layer in sorted(self._injections)},
            })

    def apply(self, h: torch.Tensor, layer: int, offset: int) -> torch.Tensor:
        injection = self._injections.get(layer)
        if injection is None:
            return h
        seq_len = h.shape[1]
        if not self.should_inject(offset, seq_len, self._prompt_token_count):
            return h  # intermediate prefill chunk: its tail is mid-prompt

        key = (h.device, h.dtype, layer)
        v = self._tensor_cache.get(key)
        if v is None:
            v = (torch.tensor(injection.vector, device=h.device, dtype=torch.float32)
                 * float(injection.alpha)).to(h.dtype)
            self._tensor_cache[key] = v

        last_index = seq_len - 1
        # Apply at the last position only — the true prompt end on the final
        # prefill chunk, or the single position of a decode step. Clone so we
        # never mutate a tensor the model may reuse.
        #
        # The addition is IN-PLACE ON THE FRESH CLONE (never on ``h``): the
        # historical ``out[…] = out[…] + v`` materialized a second tensor for
        # the sum and then copied it back, so every injected layer of every
        # decode step allocated twice — the clone plus a temporary of the same
        # row shape. At 27B (hidden 5376, bf16) that temporary is ~21.5 KB per
        # armed layer per step, ~231 KB/step across an 11-cell band, which is
        # the churn cadence open-issues §15 is chasing. ``add_`` on the view
        # runs the identical elementwise kernel on the identical inputs, so the
        # RESULT is bit-identical (pinned in
        # ``tests/test_injector_equivalence.py`` for fp32/fp16/bf16); only the
        # temporary disappears.
        out = h.clone()
        out[:, last_index, :].add_(v)
        return out
