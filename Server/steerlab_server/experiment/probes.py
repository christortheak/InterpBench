"""Scalar reading probes (parallel to Swift ``ReadingProbeArtifact`` +
``ConceptBuilder.trainReadingProbe`` / probe item storage).

A reading probe is a diagnostic *reader*, not an injection: it scores how much a
text expresses a concept. Examples live one JSON object per line in
``prompts/probes/<concept>/items.jsonl`` — ``{id?, text, expresses(bool),
topic?, split?}`` — split into ``build`` (train) and ``validation`` (held-out).
Training records activations, fits a calibrated :class:`ScalarProbe` at every
layer (direction = mean(expresses) − mean(control)), and picks the layer with
the best held-out accuracy. The artifact is saved as a JSON sidecar in a fresh
run directory.
"""

from __future__ import annotations

import hashlib
import json
import os

from . import paths
from ..steering import vector_math as vm


def _items_path(concept: str, root: str | None) -> str:
    return os.path.join(paths.project_root() if root is None else root,
                        "prompts", "probes", concept, "items.jsonl")


def read_items(concept: str, root: str | None = None) -> list[dict]:
    rows: list[dict] = []
    try:
        with open(_items_path(concept, root), encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        pass
    return rows


def save_items(concept: str, rows: list[dict], root: str | None = None) -> dict:
    path = _items_path(concept, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cleaned = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        obj = {"text": text, "expresses": bool(r.get("expresses"))}
        for k in ("id", "topic", "split"):
            if r.get(k):
                obj[k] = r[k]
        cleaned.append(obj)
    with open(path, "w", encoding="utf-8") as handle:
        for obj in cleaned:
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
    pos = sum(1 for r in cleaned if r["expresses"])
    return {"concept": concept, "items": len(cleaned), "positive": pos,
            "negative": len(cleaned) - pos}


def parse_items(content: str) -> list[dict]:
    """Parse pasted JSONL/JSON-array probe examples."""
    content = content.strip()
    rows: list[dict] = []
    if content.startswith("["):
        try:
            rows = [r for r in json.loads(content) if isinstance(r, dict)]
        except json.JSONDecodeError:
            rows = []
    else:
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return [r for r in rows if r.get("text")]


def _content_hash_split(items: list[dict],
                        build_idx: list[int]) -> tuple[list[int], list[int]]:
    """Deterministic content-hash validation split for untagged probe pools
    (cross-engine contract, 2026-07-13 — Swift implements the identical rule).

    Rule: sort the untagged items ascending by the lowercase SHA-256 hex
    digest of the item's UTF-8 ``text``; every 5th of that sorted order
    (0-based sorted index % 5 == 4) is validation, the rest build. Membership
    depends only on the texts themselves — never on file order — so both
    engines (and any re-shuffled items.jsonl) hold out the same examples.
    Returns ``(val_idx, build_idx)`` in original item order. Pools smaller
    than 5 untagged items get no validation split (index 4 never occurs).
    """
    ordered = sorted(
        build_idx,
        key=lambda i: hashlib.sha256(
            str(items[i].get("text") or "").encode("utf-8")).hexdigest())
    held = {i for position, i in enumerate(ordered) if position % 5 == 4}
    return ([i for i in build_idx if i in held],
            [i for i in build_idx if i not in held])


def train(*, items: list[dict], activations_by_layer: list[list[list[float]]]) -> dict:
    """Fit a probe per layer and select the best by held-out accuracy.

    ``activations_by_layer`` is ``[item][layer][hidden]`` aligned with ``items``.
    Items with split ``validation`` are held out; the rest train. Returns the
    chosen layer, accuracy, and the calibrated probe dict.
    """
    if not items or not activations_by_layer:
        raise ValueError("no probe items")
    layer_count = len(activations_by_layer[0])
    build_idx = [i for i, it in enumerate(items) if (it.get("split") or "build") != "validation"]
    val_idx = [i for i, it in enumerate(items) if (it.get("split") or "build") == "validation"]
    if not val_idx:
        val_idx, build_idx = _content_hash_split(items, build_idx)

    labels = [bool(items[i].get("expresses")) for i in range(len(items))]
    if not any(labels[i] for i in build_idx) or not any(not labels[i] for i in build_idx):
        raise ValueError("need both expressing and control examples in the build split")

    best = None
    for layer in range(layer_count):
        pos = [activations_by_layer[i][layer] for i in build_idx if labels[i]]
        neg = [activations_by_layer[i][layer] for i in build_idx if not labels[i]]
        if len(pos) < 2 or len(neg) < 2:
            continue
        direction = vm.mean_difference(pos, neg)
        try:
            probe = vm.scalar_probe(direction, pos, neg)
        except vm.SteeringVectorError:
            continue
        if val_idx:
            correct = sum(1 for i in val_idx
                          if probe.classifies_positive(activations_by_layer[i][layer]) == labels[i])
            acc = correct / len(val_idx)
        else:
            acc = 0.0
        if best is None or acc > best["accuracy"]:
            best = {"layer": layer, "accuracy": acc, "probe": probe.to_dict(),
                    "buildCount": len(build_idx), "valCount": len(val_idx)}
    if best is None:
        raise ValueError("could not fit a probe at any layer")
    return best


def save_artifact(concept: str, model_id: str, revision: str | None,
                  result: dict, run_directory: str) -> str:
    """Write the probe sidecar (direction + calibration + provenance)."""
    artifact = {
        "kind": "readingProbe", "concept": concept, "modelID": model_id,
        "revision": revision, "layer": result["layer"],
        "heldOutAccuracy": result["accuracy"], "probe": result["probe"],
        "buildCount": result["buildCount"], "valCount": result["valCount"],
    }
    path = os.path.join(run_directory, f"{concept}-probe.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
    return path
