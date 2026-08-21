"""Output metrics as data (parallel to Swift ``Scoring.swift``).

Concept expression is scored by a keyword/character ``markers.json`` rubric
first (model-graded second, never model-graded only — CLAUDE.md). Degeneration
is the distinct-bigram ratio. The capability battery grades short
concept-unrelated probes so a "bias" finding can be separated from sheer
capability loss under steering. Adding a concept requires no code.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass

_LETTERS = re.compile(r"[^a-z]+")


@dataclass
class MarkerRubric:
    words: set[str]
    characters: set[str]

    @classmethod
    def from_directory(cls, directory: str) -> "MarkerRubric | None":
        path = os.path.join(directory, "markers.json")
        try:
            with open(path, encoding="utf-8") as handle:
                file = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        words = {w.lower() for w in (file.get("words") or [])}
        characters = set(file.get("characters") or "")
        if not words and not characters:
            return None
        return cls(words=words, characters=characters)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [t for t in _LETTERS.split(text.lower()) if t]

    def count(self, text: str) -> int:
        tokens = self._tokens(text)
        word_hits = sum(1 for t in tokens if t in self.words)
        char_hits = sum(1 for c in text if c in self.characters)
        return word_hits + char_hits

    def density(self, text: str) -> float:
        n = len(self._tokens(text))
        return self.count(text) / n if n else 0.0


def distinct_bigram_ratio(text: str) -> float:
    """Distinct bigrams / total bigrams over whitespace tokens. Repetition
    collapse (high-alpha failure mode) drives this toward 0."""
    tokens = text.split()
    if len(tokens) < 3:
        return 0.0
    bigrams = set()
    total = 0
    for i in range(len(tokens) - 1):
        bigrams.add(f"{tokens[i]} {tokens[i + 1]}")
        total += 1
    return len(bigrams) / total if total else 0.0


def word_count(text: str) -> int:
    return len(text.split())


# --- capability battery ----------------------------------------------------

_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
_STANDALONE_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?![A-Za-z0-9])")
_LABELED_RE = re.compile(
    r"(?i)(?:final\s+answer|answer|equals?|=|therefore)\s*(?:is|:)?\s*"
    r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))")


def _normalized_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    tokens = re.split(r"[^0-9a-z]+", stripped.lower())
    return " ".join(t for t in tokens if t)


def _normalized_tokens(text: str) -> list[str]:
    return _normalized_text(text).split()


def _yes_no(text: str) -> bool | None:
    return {"yes": True, "y": True, "true": True,
            "no": False, "n": False, "false": False}.get(_normalized_text(text))


def _parse_number(text: str) -> float | None:
    t = text.strip()
    return float(t) if _NUMBER_RE.match(t) else None


def _final_answer_number(text: str) -> float | None:
    matches = _LABELED_RE.findall(text)
    if matches:
        return float(matches[-1])
    values = _STANDALONE_NUMBER_RE.findall(text)
    if len(values) == 1:
        return float(values[0])
    return None


def _infer_mode(answer: str) -> str:
    if _parse_number(answer) is not None:
        return "exact_number"
    if _yes_no(answer) is not None:
        return "yes_no"
    if len(_normalized_tokens(answer)) == 1:
        return "token_exact"
    return "exact_normalized"


def is_correct(response: str, answer: str, grading: str | None = None) -> bool:
    """Grade a capability-battery response (parallel to Swift
    ``CapabilityBattery.isCorrect``)."""
    mode = grading or _infer_mode(answer)
    if mode == "exact_number":
        expected = _parse_number(answer)
        actual = _final_answer_number(response)
        return expected is not None and actual is not None and abs(actual - expected) < 1e-9
    if mode == "yes_no":
        expected = _yes_no(answer)
        if expected is None:
            return False
        for token in _normalized_tokens(response):
            value = _yes_no(token)
            if value is not None:
                return value == expected
        return False
    if mode == "token_exact":
        expected = _normalized_text(answer)
        return bool(expected) and expected in _normalized_tokens(response)
    if mode == "exact_normalized":
        return _normalized_text(response) == _normalized_text(answer)
    if mode == "regex":
        try:
            return re.search(answer, response, re.IGNORECASE) is not None
        except re.error:
            return False
    return False
