"""The SAE latent intervention's **declared surface** — vocabulary and data,
with no execution stack under it.

This module is the schema half of :mod:`steerlab_server.steering.sae_latent`.
It holds the mode vocabulary (:data:`MODES`) and the two frozen dataclasses
that carry one feature's slice of an SAE and one declared edit over it. It
imports ``math`` and ``dataclasses`` and **nothing else**: no torch, no
injector, no model loader. That is the whole point of its existence.

Why the split (portability gap G7)
----------------------------------

``Manifest.verify`` — reached by ``experiment verify``, ``experiment freeze``
and ``bundle package`` — validates ``saeLatentConditions`` through
:mod:`steerlab_server.experiment.sae_latent`, whose own docstring says
"everything here validates OFFLINE: no SAE weights, no HuggingFace, no
network". That was true of what it *does* and false of what it *imported*:
four names (``CLAMP``, ``MODES``, ``SAELatentEdit``, ``SAELatentFeature``)
came from the execution module, which sits on
:mod:`steerlab_server.steering.injector` and therefore on ``import torch``. A
pure metadata check on a study with zero latent conditions paid for the entire
execution stack, and the cross-platform client could not verify or freeze on a
bare install.

Re-homing the four names here ends that chain. The dependency arrow points
**execution → schema and never back**: :mod:`steerlab_server.steering.sae_latent`
imports this module (and re-exports every name from it, so
``from .sae_latent import SAELatentEdit`` keeps working verbatim), and this
module imports nothing of the execution half.

Nothing here is new. Every constant, dataclass, field, docstring and refusal
string was moved from the execution module unchanged — the mode vocabulary is
a closed key set that manifest validation refuses against, and its refusal
texts are read by both engines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["ADD", "CLAMP", "MODES", "SAELatentEdit", "SAELatentFeature"]

#: Edit the pre-activation and re-evaluate the SAE's gate.
ADD = "add"
#: Set the post-activation latent to β, whatever the state.
CLAMP = "clamp"
#: The closed vocabulary. An unknown mode is refused at manifest validation —
#: never defaulted, because every default here is a different experiment.
MODES = (ADD, CLAMP)


@dataclass(frozen=True)
class SAELatentFeature:
    """One feature's slice of an SAE: everything needed to encode and decode it.

    Four numbers-and-vectors, no dictionary: a single-feature latent edit needs
    the encoder column, its bias, its JumpReLU threshold, and the decoder row.
    Loading them is a separate concern (the injectable loader seam in
    :mod:`steerlab_server.experiment.gemma_scope`), so this class is pure data
    and every test builds one by hand.
    """

    #: ``W_enc[:, f]`` — the encoder direction, length = model hidden size.
    encoder_row: tuple[float, ...]
    #: ``W_dec[f]`` — the decoder direction, length = model hidden size.
    decoder_row: tuple[float, ...]
    #: ``b_enc[f]``. When the SAE applies ``b_dec`` to its input, the exact
    #: correction ``−b_dec · W_enc[:, f]`` is FOLDED IN here by the loader
    #: (``pre = (h − b_dec)·w + b_enc`` = ``h·w + (b_enc − b_dec·w)``), so this
    #: class never needs ``b_dec`` and the arithmetic stays a dot product plus
    #: a scalar.
    encoder_bias: float = 0.0
    #: ``threshold[f]`` — the per-feature JumpReLU threshold (0.0 for a plain
    #: ReLU SAE, which the loader stamps explicitly rather than assuming).
    threshold: float = 0.0

    def __post_init__(self) -> None:
        # Any float sequence is accepted and normalized to a tuple: the rows
        # come from a loader (list) or a test (list), and a frozen dataclass
        # holding a mutable list is only nominally immutable.
        object.__setattr__(self, "encoder_row",
                           tuple(float(x) for x in self.encoder_row))
        object.__setattr__(self, "decoder_row",
                           tuple(float(x) for x in self.decoder_row))
        object.__setattr__(self, "encoder_bias", float(self.encoder_bias))
        object.__setattr__(self, "threshold", float(self.threshold))
        if len(self.encoder_row) != len(self.decoder_row):
            raise ValueError(
                f"encoder row ({len(self.encoder_row)}) and decoder row "
                f"({len(self.decoder_row)}) must both have the model's hidden "
                f"size — this is not one feature of one SAE")
        if not self.encoder_row:
            raise ValueError("empty encoder/decoder rows")

    @property
    def hidden_size(self) -> int:
        return len(self.encoder_row)


@dataclass(frozen=True)
class SAELatentEdit:
    """One declared latent edit, with its feature already resolved.

    ``label`` / ``source`` carry identity for error messages and record
    provenance only; nothing in the arithmetic reads them.
    """

    layer: int
    feature: SAELatentFeature
    #: ``"add"`` or ``"clamp"`` — see :mod:`steerlab_server.steering.sae_latent`
    #: for exact semantics.
    mode: str
    #: The edit magnitude, in LATENT units (never α, never norm units).
    beta: float
    #: The feature's id in its dictionary (provenance / messages).
    feature_id: int = -1
    #: The researcher's construct label (provenance / messages).
    label: str = ""

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(
                f"unknown SAE latent mode {self.mode!r} — expected one of "
                f"{', '.join(MODES)}")
        if not math.isfinite(self.beta):
            raise ValueError(f"β must be finite, got {self.beta!r}")
        if self.mode == CLAMP and self.beta < 0:
            raise ValueError(
                f"clamp β must be ≥ 0 (got {self.beta}): a JumpReLU latent is "
                "non-negative by construction, so a negative clamp target is "
                "outside the dictionary's range")
