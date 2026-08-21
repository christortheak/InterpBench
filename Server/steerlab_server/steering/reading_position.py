"""Where in a stimulus the residual stream is read during extraction.

Parallel to Swift ``ReadingPosition`` (``Capture/ActivationRecorder.swift``).
The string ``label`` is part of the sidecar contract — it must match the Swift
labels byte-for-byte ("last token", "mean from token k") so artifacts written
by either engine describe their reading position identically.
"""

from __future__ import annotations

from dataclasses import dataclass


class ReadingPosition:
    """Base class. Use :class:`LastToken` or :func:`mean_from_token`."""

    @property
    def label(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def minimum_token_count(self) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def requested_start_index(self) -> int | None:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class LastToken(ReadingPosition):
    """Hidden state of the final token (RepE convention; Phase 0 default)."""

    @property
    def label(self) -> str:
        return "last token"

    @property
    def minimum_token_count(self) -> int:
        return 1

    @property
    def requested_start_index(self) -> int | None:
        return None


@dataclass(frozen=True)
class MeanFromToken(ReadingPosition):
    """Mean over token positions from ``k`` onward.

    The emotion paper pools from token 50 of paragraph stories. Extraction
    callers must enforce that token ``k`` exists; otherwise this reading
    position is not valid (see :attr:`minimum_token_count`).
    """

    k: int

    @property
    def label(self) -> str:
        return f"mean from token {self.k}"

    @property
    def minimum_token_count(self) -> int:
        return max(1, self.k + 1)

    @property
    def requested_start_index(self) -> int | None:
        return self.k


def mean_from_token(k: int) -> MeanFromToken:
    return MeanFromToken(k=k)


LAST_TOKEN = LastToken()


def from_label(label: str) -> ReadingPosition:
    """Parse a sidecar/manifest ``label`` back into a reading position.

    Inverse of :attr:`ReadingPosition.label`. Unknown labels fall back to
    last-token, matching the Swift side's tolerance for older artifacts.
    """
    label = (label or "").strip().lower()
    prefix = "mean from token "
    if label.startswith(prefix):
        try:
            return mean_from_token(int(label[len(prefix):].strip()))
        except ValueError:
            return LAST_TOKEN
    return LAST_TOKEN
