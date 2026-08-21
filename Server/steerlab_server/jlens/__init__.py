"""J-lens reading instruments — server-only, Gemma-only (CLAUDE.md).

Imported lens artifacts are PyTorch/HF-native and activations do not transfer
across substrates, so nothing here is valid for the MLX engine: the Mac app may
render lens metadata, traces, and results, but it may not produce them.

Plan of record:
``docs/GEMMA3-27B-JLENS-VECTORS-AND-ONLINE-READOUT-PLAN.md``.

Nothing at import time pulls in the optional ``jlens`` reference package — the
submodules import it lazily so the server runs, and its tests pass, without the
extra installed.
"""

from .schemas import (ARTIFACT_TYPE, CANONICAL_READOUT, DIRECTION_CONVENTION,
                      JLensError, JLensRecord, Qualification, SUBSTRATE)

__all__ = [
    "ARTIFACT_TYPE",
    "CANONICAL_READOUT",
    "DIRECTION_CONVENTION",
    "JLensError",
    "JLensRecord",
    "Qualification",
    "SUBSTRATE",
]
