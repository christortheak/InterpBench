"""Multi-concept (emotion grand-mean) story corpora (parallel to Swift
``StimulusSet.MultiConceptStimulus`` + ``prompts/emotions/<concept>/stories.jsonl``).

The emotion-vector method extracts each concept's direction as
``mean(its rows) − mean(every row in the whole corpus)``, pooled from token 50.
Stories live one JSON object per line — ``{id?, concept, topic?, text, split?,
source?, notes?}`` — under ``prompts/emotions/<concept>/stories.jsonl``. This
module reads/writes that tree and assembles the combined corpus the grand-mean
extractor consumes, hashing each file's raw bytes the same way the firewall does.
"""

from __future__ import annotations

import hashlib
import json
import os

from . import paths


def _emotions_dir(root: str | None) -> str:
    return os.path.join(paths.project_root() if root is None else root, "prompts", "emotions")


def _stories_path(concept: str, root: str | None) -> str:
    return os.path.join(_emotions_dir(root), concept, "stories.jsonl")


def stories_path(concept: str, root: str | None = None) -> str:
    """Public path accessor for a concept's stories.jsonl."""
    return _stories_path(concept, root)


def stories_hash(concept: str, root: str | None = None) -> str | None:
    """SHA-256 of a concept's stories.jsonl raw bytes (the firewall pin), or
    None when the file does not exist."""
    path = _stories_path(concept, root)
    if not os.path.isfile(path):
        return None
    return _sha256_file(path)


def validation_path(concept: str, root: str | None = None) -> str:
    """Held-out labeled scenarios for a grand-mean concept (never-named texts,
    ``{"text": …, "expresses": bool}`` per line) — beside its stories.jsonl."""
    return os.path.join(_emotions_dir(root), concept, "validation.jsonl")


def list_story_concepts(root: str | None = None) -> list[dict]:
    """Concepts that have a stories.jsonl, with row counts."""
    base = _emotions_dir(root)
    out: list[dict] = []
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        path = _stories_path(name, root)
        if os.path.isfile(path):
            out.append({"concept": name, "stories": _count_rows(path)})
    return out


def read_stories(concept: str, root: str | None = None) -> list[dict]:
    """All MultiConceptStimulus rows for one concept (full fields)."""
    path = _stories_path(concept, root)
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return rows


def save_stories(concept: str, rows: list[dict], root: str | None = None) -> dict:
    """Overwrite one concept's stories.jsonl. Each row keeps its concept label."""
    path = _stories_path(concept, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cleaned: list[dict] = []
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        obj = {"concept": row.get("concept") or concept, "text": text}
        for key in ("id", "topic", "split", "source", "notes"):
            if row.get(key):
                obj[key] = row[key]
        cleaned.append(obj)
    with open(path, "w", encoding="utf-8") as handle:
        for obj in cleaned:
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return {"concept": concept, "stories": len(cleaned),
            "hash": _sha256_file(path) if cleaned else None}


def load_corpus(concepts: list[str] | None = None,
                root: str | None = None) -> tuple[list[tuple[str, str]], dict]:
    """Assemble ``[(concept, text)]`` across the chosen concepts (all, if None),
    plus a per-concept SHA-256 of the raw bytes (for pinning)."""
    names = concepts if concepts is not None else [c["concept"] for c in list_story_concepts(root)]
    rows: list[tuple[str, str]] = []
    hashes: dict[str, str] = {}
    for name in names:
        path = _stories_path(name, root)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as handle:
            data = handle.read()
        hashes[name] = hashlib.sha256(data).hexdigest()
        for line in data.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (obj.get("text") or "").strip()
            if text:
                rows.append((obj.get("concept", name), text))
    return rows, hashes


def load_stories_texts(concept: str, root: str | None = None) -> list[str]:
    """The text rows of ONE concept's stories.jsonl, in file order — the
    class loader for designated-reference extraction. Raises when the file
    is missing or empty: a silent empty class would extract a garbage mean."""
    rows, _hashes = load_corpus([concept], root)
    texts = [text for _name, text in rows]
    if not texts:
        raise ValueError(
            f"no usable stories for '{concept}' under prompts/emotions/ — "
            "designated-reference extraction needs a non-empty stories.jsonl")
    return texts


def _count_rows(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()
