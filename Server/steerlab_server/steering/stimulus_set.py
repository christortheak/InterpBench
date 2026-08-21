"""Contrastive stimulus sets + the hashing that pins them (parallel to Swift
``StimulusSet.swift``).

**Hash parity is load-bearing.** The freeze/circularity firewall verifies a
``stimulusSetHash`` recorded in the manifest against the live files; if the
Python and Swift hashes disagree, every cross-engine experiment fails to verify.
The Swift hash is ``SHA-256`` over the **raw bytes** of ``positive.jsonl`` then
``negative.jsonl`` (``StimulusSet.swift:46``), and a single-file corpus hashes
its raw bytes. We reproduce both exactly — hashing the bytes on disk, never a
re-serialization.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass


class StimulusSetError(Exception):
    pass


def _sha256_hex(*chunks: bytes) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _parse_jsonl_text(data: bytes, file: str) -> list[str]:
    """Parse a ``{"text": ...}``-per-line JSONL file. Blank lines are skipped,
    matching the Swift parser's tolerance."""
    texts: list[str] = []
    for raw in data.decode("utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StimulusSetError(f"{file}: invalid JSON line: {exc}") from exc
        if "text" not in obj:
            raise StimulusSetError(f"{file}: line missing 'text' field")
        texts.append(obj["text"])
    return texts


@dataclass
class StimulusSet:
    """Concept stimuli from ``prompts/concepts/<name>/{positive,negative}.jsonl``."""

    name: str
    positive: list[str]
    negative: list[str]
    hash: str

    @classmethod
    def from_directory(cls, directory: str) -> "StimulusSet":
        name = os.path.basename(os.path.normpath(directory))
        positive_path = os.path.join(directory, "positive.jsonl")
        negative_path = os.path.join(directory, "negative.jsonl")
        positive_data = _read(positive_path)
        negative_data = _read(negative_path)
        positive = _parse_jsonl_text(positive_data, "positive.jsonl")
        negative = _parse_jsonl_text(negative_data, "negative.jsonl")
        if not positive or not negative:
            raise StimulusSetError(f"empty stimulus set: {name}")
        # SHA-256 over positive bytes then negative bytes — matches Swift.
        return cls(name=name, positive=positive, negative=negative,
                   hash=_sha256_hex(positive_data, negative_data))


@dataclass
class LoadedTexts:
    texts: list[str]
    hash: str


def load_texts(path: str) -> LoadedTexts:
    """Plain ``{"text": …}``-per-line file (dev prompts, neutral corpus) plus
    the SHA-256 of its raw bytes for pinning. Mirrors Swift ``loadTexts``."""
    data = _read(path)
    texts = _parse_jsonl_text(data, os.path.basename(path))
    return LoadedTexts(texts=texts, hash=_sha256_hex(data))


def load_validation(path: str) -> list[dict]:
    """Load ``validation.jsonl`` as labeled held-out scenarios — one
    ``{"text": …, "expresses": bool}`` per line (parallel to Swift
    ``StimulusSet.loadValidation``). Rows missing ``expresses`` are returned
    without a label so callers can detect a legacy/unlabeled file.

    STRICT since 2026-08-01 (review: the Swift twin has always thrown while
    this loader silently skipped malformed rows and coerced labels — the
    same file could validate over different scenario sets per engine, and
    ``bool("no")`` is ``True``, so a string label silently INVERTED its
    scenario). A malformed row refuses with its line number; ``expresses``
    must be a real JSON boolean."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StimulusSetError(
                    f"malformed validation.jsonl line {i + 1}: {exc}") from exc
            if not isinstance(obj, dict) or not isinstance(obj.get("text"), str) \
                    or not obj["text"]:
                raise StimulusSetError(
                    f"validation.jsonl line {i + 1}: need a JSON object with "
                    "a non-empty 'text'")
            row = {"text": obj["text"]}
            if "expresses" in obj:
                if not isinstance(obj["expresses"], bool):
                    raise StimulusSetError(
                        f"validation.jsonl line {i + 1}: 'expresses' must be "
                        f"a JSON boolean, got {obj['expresses']!r} — a "
                        "coerced truthy string would silently invert the "
                        "scenario")
                row["expresses"] = obj["expresses"]
            rows.append(row)
    return rows


@dataclass
class PairedStimulus:
    positive: str
    negative: str
    id: str | None = None
    topic: str | None = None
    split: str | None = None


def load_pairs(path: str) -> tuple[list[PairedStimulus], str]:
    """RepE/LAT paired data; hash is SHA-256 over raw bytes (Swift parity)."""
    data = _read(path)
    pairs: list[PairedStimulus] = []
    for raw in data.decode("utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        pairs.append(PairedStimulus(
            positive=obj["positive"], negative=obj["negative"],
            id=obj.get("id"), topic=obj.get("topic"), split=obj.get("split")))
    return pairs, _sha256_hex(data)


def _read(path: str) -> bytes:
    if not os.path.exists(path):
        raise StimulusSetError(f"missing file: {path}")
    with open(path, "rb") as handle:
        return handle.read()
