"""Neutral corpus + neutral-PC basis (parallel to Swift ``NeutralCorpusStore`` /
``NeutralPCStore`` and ``ChatService.buildNeutralPCBasis``).

The neutral corpus is the keystone for three things: norm-unit alphas,
cross-layer sweeps, and **nuisance projection**. The norm corpus lives at
``prompts/neutral/corpus.jsonl``; named projection corpora at
``prompts/neutral/projection/<name>/corpus.jsonl`` (one ``{"text": …}`` per
line). A neutral-PC basis is the top principal components of a **token-position
activation bank** over that corpus, per layer — saved as a JSON sidecar so a
steering run can project those directions out of a concept vector.
"""

from __future__ import annotations

import hashlib
import json
import os

from . import paths


def norm_corpus_path(root: str | None = None) -> str:
    return os.path.join(paths.project_root() if root is None else root,
                        "prompts", "neutral", "corpus.jsonl")


def projection_corpus_path(name: str, root: str | None = None) -> str:
    return os.path.join(paths.project_root() if root is None else root,
                        "prompts", "neutral", "projection", name, "corpus.jsonl")


def _corpus_path(name: str | None, root: str | None) -> str:
    return norm_corpus_path(root) if not name or name == "norm" \
        else projection_corpus_path(name, root)


def parse_texts(content: str) -> list[str]:
    """Parse JSONL ``{"text":…}``, a JSON array, or blank-line-separated prose."""
    content = content.strip()
    if content.startswith("["):
        try:
            return [str(x.get("text", x) if isinstance(x, dict) else x)
                    for x in json.loads(content)]
        except json.JSONDecodeError:
            pass
    out: list[str] = []
    looks_jsonl = content.lstrip().startswith("{")
    if looks_jsonl:
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line).get("text", ""))
            except json.JSONDecodeError:
                continue
        return [t for t in out if t]
    # plain prose: blank-line separated paragraphs
    return [p.strip() for p in content.split("\n\n") if p.strip()]


def save_corpus(texts: list[str], name: str | None = None, root: str | None = None) -> dict:
    path = _corpus_path(name, root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    texts = [t.strip() for t in texts if t.strip()]
    with open(path, "w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return {"name": name or "norm", "count": len(texts),
            "hash": _sha256_file(path) if texts else None}


def read_corpus(name: str | None = None, root: str | None = None) -> tuple[list[str], str | None]:
    path = _corpus_path(name, root)
    texts: list[str] = []
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError:
        return [], None
    for line in data.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            texts.append(json.loads(line).get("text", ""))
        except json.JSONDecodeError:
            continue
    return [t for t in texts if t], hashlib.sha256(data).hexdigest()


def list_corpora(root: str | None = None) -> list[dict]:
    out = []
    texts, h = read_corpus(None, root)
    out.append({"name": "norm", "count": len(texts), "hash": h})
    base = os.path.join(paths.project_root() if root is None else root,
                        "prompts", "neutral", "projection")
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            t, hh = read_corpus(name, root)
            out.append({"name": name, "count": len(t), "hash": hh})
    return out


# --- neutral-PC basis ------------------------------------------------------

def _basis_dir(root: str | None) -> str:
    return os.path.join(paths.runs_directory(root))


def save_basis(*, model_id: str, revision: str | None, corpus_name: str,
               corpus_hash: str | None, components_by_layer: dict[int, list[list[float]]],
               residual_norm_per_layer: list[float], token_rows: int,
               run_directory: str, token_positions_total: int | None = None,
               token_positions_kept: int | None = None,
               downsample_seed: int | None = None,
               row_cap_per_layer: int | None = None,
               minimum_explained_variance: float | None = None,
               maximum_component_count: int | None = None) -> str:
    artifact = {
        "kind": "neutralPCBasis", "modelID": model_id, "revision": revision,
        "corpusName": corpus_name, "corpusHash": corpus_hash,
        "readingPosition": "mean from token 50", "tokenRows": token_rows,
        # Token-bank cap provenance: corpus positions vs. positions actually
        # banked, and the deterministic downsample seed (null = no cap hit).
        "tokenPositionsTotal": token_positions_total,
        "tokenPositionsKept": token_positions_kept,
        "downsampleSeed": downsample_seed,
        # The per-layer row cap that bounded ingestion, plus BOTH selection
        # controls — a variance target alone does not determine the component
        # count, because the hard ceiling can bind (Swift twin stamps the same
        # three on ``NeutralPCBasis``).
        "rowCapPerLayer": row_cap_per_layer,
        "minimumExplainedVariance": minimum_explained_variance,
        "maximumComponentCount": maximum_component_count,
        "componentsByLayer": {str(k): v for k, v in components_by_layer.items()},
        "residualNormPerLayer": residual_norm_per_layer,
        "totalComponents": sum(len(v) for v in components_by_layer.values()),
    }
    path = os.path.join(run_directory, "neutral-pc-basis.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
    return path


def load_basis(path: str) -> dict[int, list[list[float]]]:
    """Load a neutral-PC basis sidecar → ``{layer: [components]}`` for projection.
    Accepts a directory (looks for neutral-pc-basis.json) or the file itself."""
    if os.path.isdir(path):
        path = os.path.join(path, "neutral-pc-basis.json")
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    return {int(k): v for k, v in (artifact.get("componentsByLayer") or {}).items()}


def list_bases(root: str | None = None) -> list[dict]:
    runs = paths.runs_directory(root)
    out = []
    if not os.path.isdir(runs):
        return out
    for entry in sorted(os.listdir(runs), reverse=True):
        path = os.path.join(runs, entry, "neutral-pc-basis.json")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    a = json.load(handle)
                out.append({"runDirectory": os.path.join(runs, entry),
                            "modelID": a.get("modelID"), "corpusName": a.get("corpusName"),
                            "totalComponents": a.get("totalComponents"),
                            "tokenRows": a.get("tokenRows")})
            except (OSError, json.JSONDecodeError):
                continue
    return out


def _sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()
