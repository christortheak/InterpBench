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
"""

from __future__ import annotations

import torch


class LayerIntervention:
    """Base class for a residual-stream hook.

    Subclasses override :meth:`apply`.

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
