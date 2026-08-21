#!/usr/bin/env python3
"""Rank Gemma Scope SAE decoder directions against a SteerLab vector export."""

import argparse
import json
from pathlib import Path


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def feature_rows(indices, scores, decoder, sparsity=None):
    rows = []
    for index in indices:
        item = {
            "feature": int(index),
            "cosine": float(scores[index]),
            "decoderValues": decoder[index].tolist(),
        }
        if sparsity is not None:
            try:
                item["sparsity"] = float(sparsity[index])
            except Exception:
                pass
        rows.append(item)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="Path to gemmascope-job.json")
    args = parser.parse_args()

    job = load_json(args.job)
    vector_export = load_json(job["vectorFile"])
    scope = job["gemmaScope"]

    try:
        import torch
        from sae_lens import SAE
    except Exception as exc:
        raise SystemExit(
            "This analysis requires Python packages: sae-lens and torch. "
            "Install them in your research Python environment, then rerun this command. "
            f"Import error: {exc}"
        ) from exc

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=scope["recommendedRelease"],
        sae_id=scope["recommendedSAEID"],
    )

    decoder = getattr(sae, "W_dec", None)
    if decoder is None:
        raise SystemExit("Loaded SAE does not expose W_dec; inspect the SAELens object manually.")

    decoder = decoder.detach().float().cpu()
    vector = torch.tensor(vector_export["values"], dtype=torch.float32)
    if decoder.ndim != 2:
        raise SystemExit(f"Expected a 2D decoder matrix, got shape {tuple(decoder.shape)}.")
    if decoder.shape[1] != vector.numel() and decoder.shape[0] == vector.numel():
        decoder = decoder.T
    if decoder.shape[1] != vector.numel():
        raise SystemExit(
            "Decoder/vector dimensionality mismatch: "
            f"W_dec={tuple(decoder.shape)}, vector={vector.numel()}."
        )

    vector = vector / vector.norm().clamp_min(1e-12)
    decoder = decoder / decoder.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scores = decoder @ vector

    top_k = min(int(job.get("topK", 25)), scores.numel())
    positive = torch.topk(scores, k=top_k).indices.tolist()
    negative = torch.topk(-scores, k=top_k).indices.tolist()
    absolute = torch.topk(scores.abs(), k=top_k).indices.tolist()

    sparsity_list = None
    if sparsity is not None:
        try:
            sparsity_list = sparsity.detach().float().cpu().tolist()
        except Exception:
            sparsity_list = None

    report = {
        "jobFile": str(Path(args.job).resolve()),
        "vector": {
            "concept": vector_export["concept"],
            "modelID": vector_export["modelID"],
            "layer": vector_export["layer"],
            "hiddenSize": vector_export["hiddenSize"],
            "norm": vector_export["norm"],
        },
        "gemmaScope": scope,
        "artifactSidecar": job["artifactSidecar"],
        "saeConfig": cfg_dict,
        "decoderShape": list(decoder.shape),
        "topPositive": feature_rows(positive, scores, decoder, sparsity_list),
        "topNegative": feature_rows(negative, scores, decoder, sparsity_list),
        "topAbsolute": feature_rows(absolute, scores, decoder, sparsity_list),
    }

    report_path = Path(job["reportFile"])
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
