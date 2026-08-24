"""Records the residual stream during a forward pass (parallel to Swift
``ActivationRecorder`` and ``ActivationBankRecorder``).

Values are detached and copied to CPU as float32 inside the hook — capturing
live GPU tensors for every layer/token explodes memory. Both recorders assume
batch size 1, matching the Swift contract and the extraction driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .intervention import LayerIntervention
from .reading_position import LAST_TOKEN, MeanFromToken, ReadingPosition


@dataclass
class Capture:
    """One pooled readout (parallel to Swift ``ActivationRecorder.Capture``)."""

    layer: int          # transformer block whose output was recorded
    offset: int         # KV-cache position of the first token of the pass
    token_count: int    # number of token positions in this forward pass
    start_index: int    # first position included in the pooled readout
    values: list[float] # hidden vector at the reading position (CPU float32)
    residual_norm: float  # mean L2 norm of the residual stream over the window


class ActivationRecorder(LayerIntervention):
    """Pooled vector per stimulus, at configured layers.

    Appropriate for extraction passes (one prefill per stimulus, a handful of
    layers), not per-token capture over long generations.
    """

    def __init__(self, layers, position: ReadingPosition = LAST_TOKEN):
        self._layers = set(layers)
        self._position = position
        self._captures: list[Capture] = []
        self._window: tuple[int, int] | None = None

    @property
    def captures(self) -> list[Capture]:
        return list(self._captures)

    def reset(self) -> None:
        self._captures.clear()

    def set_window(self, start: int, end: int | None) -> None:
        """Pin the half-open read window for the NEXT forward pass.

        Template-aware reading positions ("turn close token",
        "post-instruction 2") resolve against the TOKEN IDS, which only the
        extraction driver holds — the hook sees hidden states. So the driver
        resolves and pins the window here, and this recorder stays free of
        template knowledge. Unset (the default) keeps the historical
        length-only behavior for the two shape-only positions, byte-for-byte.
        """
        self._window = (start, end if end is not None else -1)

    def clear_window(self) -> None:
        self._window = None

    def apply(self, h: torch.Tensor, layer: int, offset: int) -> torch.Tensor:
        if layer not in self._layers:
            return h
        length = h.shape[1]
        if self._window is not None:
            start_index, pinned_end = self._window
            end_index = length if pinned_end < 0 else min(pinned_end, length)
        elif isinstance(self._position, MeanFromToken):
            start_index = min(max(0, self._position.k), length - 1)
            end_index = length
        else:  # last token
            start_index = length - 1
            end_index = length

        # `h[0, start:length, :]` is the same tensor `h[0, start:, :]` was, so
        # the pooled-reading numbers are unchanged for every legacy recipe.
        rows = h[0, start_index:end_index, :].to(torch.float32)   # [positions, hidden]
        pooled = rows.mean(dim=0)                                  # [hidden]
        norms = torch.sqrt(rows.square().sum(dim=-1)).mean()      # scalar
        self._captures.append(Capture(
            layer=layer, offset=offset, token_count=length, start_index=start_index,
            values=pooled.detach().cpu().tolist(),
            residual_norm=float(norms.detach().cpu().item())))
        return h


@dataclass
class HookFire:
    layer: int
    offset: int
    seq_len: int


class HookFireCounter(LayerIntervention):
    """Records every hook firing (parallel to Swift ``HookFireCounter``).

    Used by the smoke test to prove the hook fires on every layer of every
    forward pass — prefill and per-token decode alike — which is the invariant
    that guards against the prefill-only steering bug.
    """

    def __init__(self):
        self._fires: list[HookFire] = []

    @property
    def fires(self) -> list[HookFire]:
        return list(self._fires)

    def reset(self) -> None:
        self._fires.clear()

    def apply(self, h: torch.Tensor, layer: int, offset: int) -> torch.Tensor:
        self._fires.append(HookFire(layer=layer, offset=offset, seq_len=h.shape[1]))
        return h


@dataclass
class BankRow:
    layer: int
    offset: int
    token_index: int
    values: list[float]
    residual_norm: float


@dataclass
class SkippedNorm:
    """Residual norm of a position the row cap excluded. Folded into the
    per-layer norm average so the denominator still describes the corpus."""

    layer: int
    residual_norm: float


class ActivationBankRecorder(LayerIntervention):
    """One residual-stream row per token position, for calibration banks
    (Anthropic-style neutral-transcript PCA).

    **Ingestion is bounded, not post-hoc trimmed.** The row cap is applied
    HERE, as rows arrive, via ``selected_row_indices`` — a precomputed set of
    global bank positions (see ``extractor.deterministic_row_selection``). Only
    the selected positions are gathered off the GPU, so the transient CPU
    float32 copy is ``kept × hidden``, never ``tokens × hidden``. Before
    2026-08-18 this recorder materialized every position of every layer of one
    text as Python floats before the driver's filter ran — for a long text at
    27B that is several GB of transient per text, which fails inside the
    allocator rather than raising (Swift twin: ``ActivationBankRecorder``).

    Positions are counted per layer in ARRIVAL order — text order, then token
    order — which is exactly the order the driver's filter indexed, so the
    banked rows are unchanged. The cursor therefore survives ``reset()`` (which
    drains rows only); use ``reset_all()`` to start a fresh bank.

    Token banks are still large even when bounded; restrict layers and corpus
    size deliberately.
    """

    def __init__(self, layers, start_index: int = 0,
                 selected_row_indices: set[int] | None = None):
        self._layers = set(layers)
        self._start_index = max(0, start_index)
        self._selected = selected_row_indices
        self._rows: list[BankRow] = []
        self._skipped_norms: list[SkippedNorm] = []
        self._next_position: dict[int, int] = {}
        self._peak_row_count = 0

    @property
    def rows(self) -> list[BankRow]:
        return list(self._rows)

    @property
    def skipped_norms(self) -> list[SkippedNorm]:
        return list(self._skipped_norms)

    @property
    def peak_retained_row_count(self) -> int:
        """High-water mark of retained rows — the bounded-ingestion instrument."""
        return self._peak_row_count

    def reset(self) -> None:
        """Drains rows and skipped norms. The per-layer cursor deliberately
        survives, so a driver can drain after every pass while the global
        selection stays aligned."""
        self._rows.clear()
        self._skipped_norms.clear()

    def reset_all(self) -> None:
        """Full reset, cursors included — a new bank over a new corpus."""
        self.reset()
        self._next_position.clear()
        self._peak_row_count = 0

    def apply(self, h: torch.Tensor, layer: int, offset: int) -> torch.Tensor:
        if layer not in self._layers:
            return h
        length = h.shape[1]
        if self._start_index >= length:
            return h
        position_count = length - self._start_index

        # Claim this pass's slice of the global position space first, so the
        # selection is stable regardless of pass order.
        base = self._next_position.get(layer, 0)
        self._next_position[layer] = base + position_count

        if self._selected is None:
            kept = list(range(position_count))
        else:
            kept = [p for p in range(position_count) if base + p in self._selected]

        window = h[0, self._start_index:, :].to(torch.float32)
        # Norms over EVERY position: one float per token, cheap even when
        # nothing in this pass is banked.
        residual_norms = torch.sqrt(window.square().sum(dim=-1)).detach().cpu().tolist()

        # Gather ONLY the kept rows on-device before the CPU copy. This is the
        # bound: ``values`` is kept × hidden, not position_count × hidden.
        if kept:
            if len(kept) == position_count:
                gathered = window
            else:
                index = torch.tensor(kept, device=window.device, dtype=torch.long)
                gathered = window.index_select(0, index)
            values = gathered.detach().cpu().tolist()
            for slot, position in enumerate(kept):
                self._rows.append(BankRow(
                    layer=layer, offset=offset,
                    token_index=self._start_index + position,
                    values=values[slot],
                    residual_norm=float(residual_norms[position])))
        kept_set = set(kept)
        for position in range(position_count):
            if position not in kept_set:
                self._skipped_norms.append(
                    SkippedNorm(layer=layer,
                                residual_norm=float(residual_norms[position])))
        self._peak_row_count = max(self._peak_row_count, len(self._rows))
        return h
