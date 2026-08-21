"""Concept authoring — create/edit/import stimulus sets and emit LLM helper
prompts (parallel to Swift ``ConceptBuilder`` + the ``/api/concept/*`` writes).

Writes the cluster-canonical ``prompts/concepts/<name>/{positive,negative}.jsonl``
in the same ``{"text": …}``-per-line format the rest of the engine reads, so a
concept authored here extracts and freezes identically to one authored in the
Swift app. Editing a concept changes its SHA-256 — which is exactly what the
freeze firewall is meant to detect.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re

from . import paths
from .scoring import distinct_bigram_ratio  # noqa: F401 (kept for parity imports)


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]*$")


def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not _NAME_RE.match(name) or "/" in name or "\\" in name:
        raise ValueError(f"invalid concept name {name!r}")
    return name


def create_concept(name: str, root: str | None = None) -> str:
    name = _validate_name(name)
    directory = paths.concept_directory(name, root)
    os.makedirs(directory, exist_ok=True)
    for fname in ("positive.jsonl", "negative.jsonl"):
        path = os.path.join(directory, fname)
        if not os.path.exists(path):
            open(path, "w", encoding="utf-8").close()
    return directory


def delete_concept(name: str, root: str | None = None) -> dict:
    """Remove a concept's editable datasets (concepts/emotions/probes/repe).
    Immutable vector artifacts in runs/ are intentionally left untouched."""
    import shutil
    name = _validate_name(name)
    base = paths.project_root() if root is None else root
    removed = []
    for sub in ("concepts", "emotions", "probes", "repe"):
        directory = os.path.join(base, "prompts", sub, name)
        if os.path.isdir(directory):
            shutil.rmtree(directory)
            removed.append(sub)
    return {"name": name, "removed": removed}


def stimulus_content_hash(positive: list[str], negative: list[str]) -> str | None:
    """Cross-engine CONTENT hash of a paired stimulus set: SHA-256 over the
    compact, key-sorted JSON ``{"negative":[…],"positive":[…]}`` (UTF-8,
    non-ASCII unescaped, slashes unescaped).

    Raw file hashes (``StimulusSet.hash``) are engine-formatting-sensitive —
    this engine writes ``{"text": …}`` with a space, Swift writes
    ``{"text":…}`` — so the same texts pushed through the save API hash
    differently at the byte level on each tree. Drift detection in the Swift
    Concept Lab therefore compares THIS hash (mirrored byte-for-byte by
    ``ConceptBuilder.stimulusContentHash``; golden-fixture tested on both
    engines). Nil when both sides are empty. The freeze firewall still pins
    the raw byte hash of the executing substrate's files — this hash is a UI
    drift comparator, never a manifest pin.
    """
    if not positive and not negative:
        return None
    payload = json.dumps({"negative": negative, "positive": positive},
                         ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_concept(name: str, root: str | None = None) -> dict:
    _validate_name(name)
    directory = paths.concept_directory(name, root)
    positive = _read_texts(os.path.join(directory, "positive.jsonl"))
    negative = _read_texts(os.path.join(directory, "negative.jsonl"))
    return {
        "name": name,
        "positive": positive,
        "negative": negative,
        "hasValidation": os.path.exists(os.path.join(directory, "validation.jsonl")),
        "hasMarkers": os.path.exists(os.path.join(directory, "markers.json")),
        "contentHash": stimulus_content_hash(positive, negative),
    }


def save_concept(name: str, positive: list[str], negative: list[str],
                 root: str | None = None) -> dict:
    """Overwrite the stimulus files with the given texts, returning the new
    hash + counts. Editing intentionally changes the stimulus hash."""
    name = _validate_name(name)
    directory = paths.concept_directory(name, root)
    os.makedirs(directory, exist_ok=True)
    positive = [t for t in (s.strip() for s in positive) if t]
    negative = [t for t in (s.strip() for s in negative) if t]
    _write_texts(os.path.join(directory, "positive.jsonl"), positive)
    _write_texts(os.path.join(directory, "negative.jsonl"), negative)

    # Recompute the hash the way StimulusSet does (only when both sides exist).
    info: dict = {"name": name, "positiveCount": len(positive),
                  "negativeCount": len(negative),
                  "contentHash": stimulus_content_hash(positive, negative)}
    if positive and negative:
        from ..steering.stimulus_set import StimulusSet
        info["hash"] = StimulusSet.from_directory(directory).hash
    return info


def parse_import(content: str, filename: str = "") -> dict:
    """Parse pasted/uploaded stimuli into pairs and/or single texts.

    Supports: JSONL with ``{"positive","negative"}`` (pairs) or ``{"text"}``
    (single side); CSV with two columns (positive,negative); and plain lines
    (treated as single texts). The UI decides which side single texts join.
    """
    pairs: list[dict] = []
    texts: list[str] = []
    name = (filename or "").lower()
    stripped = content.strip()

    if name.endswith(".jsonl") or stripped.startswith("{"):
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                texts.append(line)
                continue
            if "positive" in obj and "negative" in obj:
                pairs.append({"positive": obj["positive"], "negative": obj["negative"]})
            elif "text" in obj:
                texts.append(obj["text"])
    elif name.endswith(".csv") or ("," in stripped and "\n" in stripped):
        reader = csv.reader(io.StringIO(content))
        for row in reader:
            if len(row) >= 2 and (row[0].strip() or row[1].strip()):
                pairs.append({"positive": row[0].strip(), "negative": row[1].strip()})
            elif row and row[0].strip():
                texts.append(row[0].strip())
    else:
        texts = [l.strip() for l in stripped.splitlines() if l.strip()]
    return {"pairs": pairs, "texts": texts}


# Recipe-family generation templates — the SAME files the Swift app uses, so
# the "copy LLM prompt" helpers are byte-identical across engines (parallel to
# ConceptBuilder.generationPrompt / anthropicStyleNeutralDialoguePrompt).
TEMPLATE_FILES = {
    "caa": "caa-paired-stimuli.md",
    "lat": "repe-paired-reader-data.md",
    "grandmean": "emotion-grand-mean-stories.md",
    "cowork": "grand-mean-cowork-agent.md",
    "probe": "probe-validation-items.md",
    "neutral": "neutral-norm-corpus.md",
    "anthropic-dialogue": "neutral-dialogues-anthropic-style.md",
}


def template_prompt(template: str, *, concept: str = "", count: int = 20,
                    guidance: str = "", neutral_concepts: str = "",
                    matched_domains: str = "", avoid_settings: str = "",
                    root: str | None = None) -> str:
    """Render a generation template from ``prompts/generation/`` with the same
    ``{{key}}`` substitutions the Swift Concept Lab uses."""
    filename = TEMPLATE_FILES.get(template)
    if filename is None:
        raise ValueError(f"unknown template {template!r}")
    path = os.path.join(paths.project_root() if root is None else root,
                        "prompts", "generation", filename)
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise ValueError(f"template not found: {filename}") from exc

    repl = _template_replacements(template, concept, count, guidance,
                                  neutral_concepts, matched_domains, avoid_settings)
    for key, value in repl.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def _template_replacements(template, concept, count, guidance,
                           neutral_concepts, matched_domains, avoid_settings) -> dict:
    if template == "caa":
        return {"concept": concept, "count": count}
    if template == "lat":
        return {"concept": concept, "count": count,
                "template_or_scaffold": guidance or
                "Use matched prompt fragments or an answer scaffold appropriate to the concept."}
    if template == "grandmean":
        return {"concepts": concept, "concept": concept, "topic": "topic",
                "stories_per_concept_topic": max(1, count),
                "topics": guidance or "ordinary non-study topics matched across concepts"}
    if template == "cowork":
        return {"concepts": concept, "split": "build",
                "stories_per_concept_topic": max(12, count),
                "expected_rows": "concepts × chosen topics × stories",
                "topics": guidance or
                "choose 6-10 ordinary, study-neutral topics and use the same topics for every concept"}
    if template == "probe":
        return {"concept": concept, "count": max(20, count)}
    if template == "neutral":
        return {"count": max(200, count * 20), "minimum_words": "90"}
    if template == "anthropic-dialogue":
        return {
            "count": max(200, count * 20), "minimum_words": "120",
            "neutral_concepts": neutral_concepts or
            "the concept set currently under study; ask the user to provide concepts before generating final data",
            "matched_domains": matched_domains or
            ("coding, factual questions, math, geography, workplace summarization, practical "
             "how-to, formatting, classification, list generation, simple planning, household "
             "tasks, inventory, scheduling, and technical explanation"),
            "avoid_settings": avoid_settings or
            ("danger, illness, violence, conflict, romance, death, law, politics, religion, "
             "identity groups, moral judgment, persuasion, experiments, AI safety, or model behavior"),
        }
    return {}


def generation_prompt(name: str, count: int = 20) -> str:
    """An LLM prompt the user can paste elsewhere to draft contrastive stimuli
    (parallel to the Swift 'Copy LLM prompt' helper)."""
    return (
        f"You are helping build a contrastive stimulus set for the concept "
        f"\"{name}\" for activation-steering research.\n\n"
        f"Produce {count} contrastive PAIRS. Each pair has a `positive` sentence "
        f"that strongly expresses \"{name}\" and a `negative` sentence that is "
        f"matched in length, topic, and style but does NOT express it. Crucially, "
        f"the pairs must be INDEPENDENT of any downstream task the concept will be "
        f"tested on (the circularity constraint): keep them to ordinary, "
        f"everyday scenarios and do not mention the task domain the study's "
        f"own items are set in.\n\n"
        f"Return JSONL, one object per line:\n"
        f'{{"positive": "...", "negative": "..."}}\n')


def _read_texts(path: str) -> list[str]:
    out: list[str] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line).get("text", ""))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


def _write_texts(path: str, texts: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
