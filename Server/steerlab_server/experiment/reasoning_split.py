"""Split a generated text into the reasoning a model wrote and the answer after it.

One rule, shared by every reader that has to tell the two apart: the paired
judge (which reads a verdict out of the answer) and the panel runner (which
passes a seat's answer, and never its reasoning, on to its colleagues). Swift
twin: ``ReasoningSplit`` in ``Sources/ExperimentKit/ReasoningSplit.swift``,
pinned to this module by ``Tests/Fixtures/cross-engine/reasoning-split.json``.

The text carries one of three shapes:

* ``closed``: a reasoning block that ends. Either it opens with a leading
  ``<think>`` tag (whitespace allowed before it, attributes allowed inside it)
  and a later ``</think>`` closes it, or the first ``</think>`` comes before
  any opening tag. The second shape is what a chat template that opens the
  block in the PROMPT leaves behind: the generation starts inside the block,
  and its first ``</think>`` is the only boundary the text carries. The
  reasoning is the text inside the block and the answer is everything after
  the closing tag, byte for byte.
* ``unclosed``: a leading opening tag with no closing tag after it. Everything
  after the opening tag is reasoning, and there is no answer.
* ``none``: neither shape. There is no reasoning block, and the whole text is
  the answer. A ``<think>`` aside AFTER some answer text is content, not a
  block, and so is a closing tag that follows an opening tag which is not at
  the start.

Reasoning cut off before its closing tag, in a generation that started inside
the block, carries no tag at all. The text alone cannot tell it from an
answer, so it reads as ``none`` here; a caller that knows the generation was
thinking and was cut off says so itself (``multi_agent``).

The scan is literal and works on Unicode code points: the tags are matched
with ASCII case folding only, and "whitespace" is ``str.isspace``. Both
engines apply the same rule to the same code points, so they cannot disagree
about where a block ends.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The closing tag. Matched case-insensitively (ASCII only), exactly as
#: written: a closing tag carries no attributes.
CLOSE_TAG = "</think>"
#: The opening tag's name, after ``<``. A tag is ``<think>`` or ``<think``
#: followed by whitespace, then anything up to the first ``>``.
_OPEN_NAME = "<think"

#: The three shapes, as the ``state`` of a :class:`ReasoningSplit`.
CLOSED = "closed"
UNCLOSED = "unclosed"
NONE = "none"


@dataclass(frozen=True)
class ReasoningSplit:
    """What :func:`split` found. ``reasoning`` is None exactly when ``state``
    is ``none``; ``answer`` is empty when ``state`` is ``unclosed``."""

    reasoning: str | None
    answer: str
    state: str


def _ascii_lower(text: str) -> str:
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in text)


def _matches_at(text: str, index: int, needle: str) -> bool:
    return _ascii_lower(text[index:index + len(needle)]) == needle


def _open_tag_end(text: str, index: int) -> int | None:
    """The index just past an opening tag that starts at ``index``, or None
    when no opening tag starts there."""
    if not _matches_at(text, index, _OPEN_NAME):
        return None
    after = index + len(_OPEN_NAME)
    if after >= len(text):
        return None
    if text[after] == ">":
        return after + 1
    if not text[after].isspace():
        return None  # ``<thinking>`` is not a reasoning tag
    close = text.find(">", after)
    return None if close < 0 else close + 1


def _first_close(text: str, start: int = 0) -> int:
    index = text.find("<", start)
    while index >= 0:
        if _matches_at(text, index, CLOSE_TAG):
            return index
        index = text.find("<", index + 1)
    return -1


def _first_open(text: str) -> int:
    index = text.find("<")
    while index >= 0:
        if _open_tag_end(text, index) is not None:
            return index
        index = text.find("<", index + 1)
    return -1


def split(text: str) -> ReasoningSplit:
    """The reasoning block in ``text``, and the answer after it."""
    text = text or ""
    start = 0
    while start < len(text) and text[start].isspace():
        start += 1
    open_end = _open_tag_end(text, start)
    if open_end is not None:
        close = _first_close(text, open_end)
        if close < 0:
            return ReasoningSplit(text[open_end:], "", UNCLOSED)
        return ReasoningSplit(text[open_end:close],
                              text[close + len(CLOSE_TAG):], CLOSED)
    close = _first_close(text)
    if close >= 0:
        first_open = _first_open(text)
        if first_open < 0 or close < first_open:
            return ReasoningSplit(text[:close], text[close + len(CLOSE_TAG):],
                                  CLOSED)
    return ReasoningSplit(None, text, NONE)


def leading_whitespace_removed(text: str) -> str:
    """``text`` without its leading whitespace (``str.isspace``), the form a
    panel routes an answer in: the newlines a model writes after
    ``</think>`` are layout, not part of what it said. Swift twin:
    ``ReasoningSplit.leadingWhitespaceRemoved``."""
    index = 0
    while index < len(text) and text[index].isspace():
        index += 1
    return text[index:]
