"""Regenerates the committed cross-engine parity fixtures in this directory.

Tiny HAND-AUTHORED vector-artifact pairs (4 layers x 8 dims, exact small
integers, float32) exercising the `vectors compare` parity harness on BOTH
engines (`steerlab-server vectors compare` / `steerlab-cli vectors compare`):

- identical-a / identical-b : same values            -> cosine 1.0, ratio 1.0
- orthogonal-a / orthogonal-b: disjoint one-hots      -> cosine 0.0, ratio 1.0
- scaled-a / scaled-b        : b = 2*a                -> cosine 1.0, ratio 2.0
- truncated-a (3 layers)     : identical-a minus its last layer, paired with
  identical-b to pin the layer-count-mismatch behavior (intersection compared,
  `layerCountMismatch: true`).
- narrow-a (4 dims)          : identical-a at HALF the hidden size, paired with
  identical-b to pin the COULD-NOT-COMPARE outcome (hidden-size mismatch —
  `vectors compare` exit 2 / `notFound` 66, never a failed comparison). It has
  no golden by construction: there is no report.

`golden-<pair>.json` files are the Python engine's `vectors compare` output at
the default threshold; the Swift and Python test suites BOTH assert their own
engine's report against these exact numbers (1e-6) and key sets. The copies
under `Tests/ExperimentKitTests/Fixtures/parity/` must stay byte-identical to
this directory — rerun this script, then re-copy.

These integer fixtures are SCAFFOLDING for the harness itself. The real
MLX-vs-CUDA french-vector fixture pair (same concept extracted on both
substrates, per-layer cosine ~= 1.00 expected as a STRUCTURE claim, not byte
identity) gets added here after the first cluster session produces one.

Run from the repo root:
    Server/.venv.nosync/bin/python Server/tests/fixtures/parity/generate.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))

LAYERS, HIDDEN = 4, 8


def _identical_layer(layer: int) -> list[float]:
    return [float(layer + 1 + i) for i in range(HIDDEN)]


def _one_hot(index: int, value: float) -> list[float]:
    row = [0.0] * HIDDEN
    row[index] = value
    return row


ARTIFACTS: dict[str, dict] = {
    # substrate stamps document that parity comparison is CROSS-substrate by
    # design ("a" swift-mlx, "b" python-hf-transformers); the compare verb
    # must not refuse foreign-substrate artifacts (unlike injection paths).
    "identical-a": {
        "substrate": "swift-mlx",
        "perLayer": [_identical_layer(l) for l in range(LAYERS)],
    },
    "identical-b": {
        "substrate": "python-hf-transformers",
        "perLayer": [_identical_layer(l) for l in range(LAYERS)],
    },
    "orthogonal-a": {
        "substrate": "swift-mlx",
        "perLayer": [_one_hot(l, float(l + 1)) for l in range(LAYERS)],
    },
    "orthogonal-b": {
        "substrate": "python-hf-transformers",
        "perLayer": [_one_hot(l + 4, float(l + 1)) for l in range(LAYERS)],
    },
    "scaled-a": {
        "substrate": "swift-mlx",
        "perLayer": [[float(l + 1) * (i + 1) for i in range(HIDDEN)]
                     for l in range(LAYERS)],
    },
    "scaled-b": {
        "substrate": "python-hf-transformers",
        "perLayer": [[2.0 * (l + 1) * (i + 1) for i in range(HIDDEN)]
                     for l in range(LAYERS)],
    },
    "truncated-a": {
        "substrate": "swift-mlx",
        "perLayer": [_identical_layer(l) for l in range(LAYERS - 1)],
    },
    # Half the hidden size: the only fixture pair that CANNOT be compared at
    # all. Its sidecar declares hiddenSize 4, which is what makes it a
    # different-model artifact rather than a truncated one.
    "narrow-a": {
        "substrate": "swift-mlx",
        "hiddenSize": HIDDEN // 2,
        "perLayer": [_identical_layer(l)[: HIDDEN // 2] for l in range(LAYERS)],
    },
}

PAIRS = {
    "identical": ("identical-a", "identical-b"),
    "orthogonal": ("orthogonal-a", "orthogonal-b"),
    "scaled": ("scaled-a", "scaled-b"),
    "truncated": ("truncated-a", "identical-b"),
}


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in values))


def write_artifact(name: str, spec: dict) -> None:
    per_layer = spec["perLayer"]
    tensors = {f"layer_{i}": np.asarray(v, dtype=np.float32)
               for i, v in enumerate(per_layer)}
    save_file(tensors, os.path.join(HERE, f"{name}.safetensors"))
    sidecar = {
        "schemaVersion": 2,
        "modelID": "steerlab/parity-fixture",
        "concept": f"parity-{name}",
        "stimulusSetHash": "parity-fixture-v1",
        "layerCount": len(per_layer),
        "hiddenSize": spec.get("hiddenSize", HIDDEN),
        "normsPerLayer": [_norm(v) for v in per_layer],
        "extractionDate": "2026-07-12T00:00:00Z",
        "substrate": spec["substrate"],
        "recipeName": "hand-authored cross-engine parity fixture (WS7.3)",
    }
    with open(os.path.join(HERE, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> int:
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    from steerlab_server.steering import vector_parity

    for name, spec in ARTIFACTS.items():
        write_artifact(name, spec)
    for pair, (a, b) in PAIRS.items():
        report = vector_parity.compare_paths(
            os.path.join(HERE, f"{a}.safetensors"),
            os.path.join(HERE, f"{b}.safetensors"))
        golden = os.path.join(HERE, f"golden-{pair}.json")
        with open(golden, "w", encoding="utf-8") as fh:
            fh.write(report.json_text())
        print(f"golden-{pair}.json: minCosine={report.min_cosine} "
              f"pass={report.passed}")
    print(f"wrote {len(ARTIFACTS)} artifacts + {len(PAIRS)} goldens in {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
