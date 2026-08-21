"""Thin adapter over the pinned ``jlens`` reference package.

Two reasons this is an adapter and not a direct dependency at every call site:

1. **The dependency is optional and lazily imported.** It lives behind the
   ``jlens`` extra, which is deliberately NOT in ``all`` (it floors
   ``transformers>=5.5`` and installs from a git URL). The server imports and
   runs without it; only J-lens verbs need it.
2. **The reference loader cannot serve our access pattern.**
   ``JacobianLens.__init__`` eagerly promotes every layer to float32, so the
   published fp16 tensor becomes ~2x resident, all layers, whether one is armed
   or all of them. Import converts once to per-layer safetensors (plan §4.1)
   and generation reads a single layer from there.

:class:`StubBackend` lets the store, importer, routes, and CLI be tested with
no reference package, no model, and no GPU.
"""

from __future__ import annotations

from typing import Protocol

from .schemas import JLensError

#: The pinned reference (see Server/pyproject.toml [jlens] extra). Recorded on
#: every imported record; a test asserts the installed package still matches.
REFERENCE_PACKAGE = "jlens"
REFERENCE_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"


class LensSource(Protocol):
    """What the importer needs from a raw lens checkpoint."""

    @property
    def d_model(self) -> int: ...

    @property
    def n_prompts(self) -> int: ...

    @property
    def source_layers(self) -> list[int]: ...

    def jacobian(self, layer: int):
        """The raw ``J_l`` at its STORED dtype — never promoted."""


class CheckpointLensSource:
    """Reads the published ``.pt`` directly, preserving the stored dtype.

    The checkpoint is a plain mapping written by ``JacobianLens.save``::

        {"J": {layer: Tensor}, "n_prompts": int,
         "source_layers": [...], "d_model": int}

    Loaded with ``weights_only=True``. That is not merely defensive: the file
    is a pickle-based container fetched from a public repository, and the
    reference's own loader passes the same flag, so the safe path costs
    nothing.

    Deliberately does NOT go through ``JacobianLens``: that constructor's
    ``J.float()`` is exactly the promotion this conversion exists to avoid.
    The published tensors are **fp16** (chosen upstream because entries are
    O(1), so fp16's extra mantissa bits beat bf16) — preserve that rather than
    round-tripping through another width.
    """

    def __init__(self, path: str):
        import torch

        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 — surfaced verbatim, typed
            raise JLensError(f"could not read lens checkpoint '{path}': {exc}") from exc
        if not isinstance(checkpoint, dict) or "J" not in checkpoint:
            found = sorted(checkpoint) if isinstance(checkpoint, dict) else type(checkpoint).__name__
            raise JLensError(
                f"'{path}' is not a JacobianLens file (found {found!r}) — a raw "
                f"fit() checkpoint, or the wrong file from the repository")
        self._j = {int(k): v for k, v in checkpoint["J"].items()}
        self._d_model = int(checkpoint.get("d_model") or 0)
        self._n_prompts = int(checkpoint.get("n_prompts") or 0)
        declared = checkpoint.get("source_layers")
        self._source_layers = sorted(int(x) for x in declared) if declared else sorted(self._j)
        missing = [l for l in self._source_layers if l not in self._j]
        if missing:
            raise JLensError(
                f"'{path}' declares source layers {missing} with no matching "
                f"matrix — refusing an ambiguous layer mapping")

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def n_prompts(self) -> int:
        return self._n_prompts

    @property
    def source_layers(self) -> list[int]:
        return list(self._source_layers)

    def jacobian(self, layer: int):
        try:
            return self._j[layer]
        except KeyError:
            raise JLensError(
                f"layer {layer} is not a fitted source layer "
                f"(have {self._source_layers[0]}..{self._source_layers[-1]})") from None


class StubBackend:
    """Deterministic in-memory lens for tests. No torch beyond tensor creation."""

    def __init__(self, *, d_model: int = 8, source_layers: list[int] | None = None,
                 n_prompts: int = 3, dtype: str = "float16"):
        import torch

        self._d_model = d_model
        self._source_layers = sorted(source_layers if source_layers is not None else [0, 1, 2])
        self._n_prompts = n_prompts
        td = getattr(torch, dtype)
        # Distinct per layer and reproducible, so a mis-mapped layer is visible
        # rather than silently plausible.
        self._j = {
            l: (torch.eye(d_model) * (l + 1)).to(td) for l in self._source_layers
        }

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def n_prompts(self) -> int:
        return self._n_prompts

    @property
    def source_layers(self) -> list[int]:
        return list(self._source_layers)

    def jacobian(self, layer: int):
        try:
            return self._j[layer]
        except KeyError:
            raise JLensError(f"layer {layer} is not a fitted source layer") from None

    def save_checkpoint(self, path: str) -> str:
        """Write this stub in the published ``.pt`` layout, for round-trip tests."""
        import torch

        torch.save({"J": self._j, "n_prompts": self._n_prompts,
                    "source_layers": self._source_layers,
                    "d_model": self._d_model}, path)
        return path


def reference_version() -> str | None:
    """Installed reference package version, or None when the extra is absent."""
    try:
        from importlib.metadata import version

        return version(REFERENCE_PACKAGE)
    except Exception:  # noqa: BLE001 — absence is a normal state, not an error
        return None


def require_reference():
    """Import the reference package or raise a typed, actionable error."""
    try:
        import jlens  # noqa: F401
    except ImportError as exc:
        raise JLensError(
            "the J-lens reference package is not installed — it lives behind an "
            "optional extra that is deliberately not in `all` (it floors "
            "transformers>=5.5 and installs from a git URL). Install it with: "
            "pip install -e 'Server[jlens]'") from exc
    return __import__("jlens")
