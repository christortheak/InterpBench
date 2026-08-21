"""Cross-engine vector-artifact parity harness (WS7.3).

Compares two per-layer steering-vector artifacts (``<name>.safetensors`` +
``<name>.json`` sidecar — the cross-engine on-disk contract of
:mod:`vector_store`) layer by layer: cosine similarity, norm ratio, and a
min/mean summary, with a threshold gate for CI. The Swift twin
(``ExperimentKit/VectorParity.swift``, ``steerlab-cli vectors compare``)
emits a KEY-IDENTICAL JSON report computed with the same double-precision
sequential arithmetic, so `diff`ing the two engines' outputs is trivial and
both test suites can assert the same committed golden files
(``Server/tests/fixtures/parity/`` /
``Tests/ExperimentKitTests/Fixtures/parity/``).

Semantics (pinned, mirror any change in BOTH engines):
- Layer-count mismatch is tolerated: the intersection ``0..<min(countA,
  countB)`` is compared and ``layerCountMismatch`` says so.
- A layer where either vector has zero norm has no defined cosine: its
  ``cosine`` is null, it is EXCLUDED from min/mean cosine, and it is counted
  in ``summary.skippedZeroNormLayers``. ``normRatio`` is ``normB / normA``
  (B in units of A) and is null when ``normA`` is zero.
- ``pass`` is true iff a min cosine exists and is ≥ ``threshold`` — an
  empty/fully-skipped comparison can never pass.
- Hidden-size mismatch is an error, not a report: the artifacts are not
  comparable at all. It is the CLI's THIRD outcome (could-not-compare,
  ``notFound``/66 in JSON and exit 2 in human mode), never a comparison that
  failed — as against layer-count mismatch, which IS a comparison.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

from . import vector_store

DEFAULT_THRESHOLD = 0.98


@dataclass
class LayerComparison:
    layer: int
    norm_a: float
    norm_b: float
    cosine: float | None
    norm_ratio: float | None


@dataclass
class ArtifactSummary:
    name: str
    layer_count: int
    hidden_size: int


@dataclass
class ParityReport:
    artifact_a: ArtifactSummary
    artifact_b: ArtifactSummary
    per_layer: list[LayerComparison]
    threshold: float

    @property
    def compared_layer_count(self) -> int:
        return len(self.per_layer)

    @property
    def layer_count_mismatch(self) -> bool:
        return self.artifact_a.layer_count != self.artifact_b.layer_count

    @property
    def cosines(self) -> list[float]:
        return [c.cosine for c in self.per_layer if c.cosine is not None]

    @property
    def min_cosine(self) -> float | None:
        return min(self.cosines) if self.cosines else None

    @property
    def mean_cosine(self) -> float | None:
        return _naive_mean(self.cosines)

    @property
    def mean_norm_ratio(self) -> float | None:
        return _naive_mean(
            [c.norm_ratio for c in self.per_layer if c.norm_ratio is not None])

    @property
    def skipped_zero_norm_layers(self) -> int:
        return sum(1 for c in self.per_layer if c.cosine is None)

    @property
    def passed(self) -> bool:
        m = self.min_cosine
        return m is not None and m >= self.threshold

    def to_dict(self) -> dict:
        """The pinned cross-engine JSON shape — keys must stay identical to
        Swift ``VectorParity.Report`` (goldens in the parity fixture dirs
        assert this key set on both engines)."""
        def summary(a: ArtifactSummary) -> dict:
            return {"hiddenSize": a.hidden_size, "layerCount": a.layer_count,
                    "name": a.name}
        return {
            "artifactA": summary(self.artifact_a),
            "artifactB": summary(self.artifact_b),
            "comparedLayerCount": self.compared_layer_count,
            "layerCountMismatch": self.layer_count_mismatch,
            "pass": self.passed,
            "perLayer": [
                {"cosine": c.cosine, "layer": c.layer, "normA": c.norm_a,
                 "normB": c.norm_b, "normRatio": c.norm_ratio}
                for c in self.per_layer
            ],
            "summary": {
                "meanCosine": self.mean_cosine,
                "meanNormRatio": self.mean_norm_ratio,
                "minCosine": self.min_cosine,
                "skippedZeroNormLayers": self.skipped_zero_norm_layers,
            },
            "threshold": self.threshold,
        }

    def json_text(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _naive_mean(values: list[float]) -> float | None:
    # Plain sequential accumulation, NOT the builtin ``sum`` — CPython ≥ 3.12
    # gives float ``sum`` Neumaier compensation, which the Swift twin's
    # ``reduce(0, +)`` does not do, and the two diverge in the last ulp
    # (byte-different goldens). Naive-sequential is the pinned convention.
    if not values:
        return None
    total = 0.0
    for x in values:
        total += x
    return total / len(values)


def _norm(values: list[float]) -> float:
    # Sequential double-precision accumulation — the Swift twin uses the
    # identical loop order, so tiny fixtures agree far below the 1e-6 bar.
    total = 0.0
    for x in values:
        total += float(x) * float(x)
    return math.sqrt(total)


def _dot(a: list[float], b: list[float]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        total += float(x) * float(y)
    return total


def compare_vectors(name_a: str, per_layer_a: list[list[float]],
                    name_b: str, per_layer_b: list[list[float]],
                    *, threshold: float = DEFAULT_THRESHOLD) -> ParityReport:
    """Pure comparison over in-memory per-layer vectors (unit-test surface)."""
    hidden_a = len(per_layer_a[0]) if per_layer_a else 0
    hidden_b = len(per_layer_b[0]) if per_layer_b else 0
    if hidden_a != hidden_b:
        raise ValueError(
            f"hidden-size mismatch: {name_a} is {hidden_a}-dim, {name_b} is "
            f"{hidden_b}-dim — these artifacts are not comparable")
    compared = min(len(per_layer_a), len(per_layer_b))
    rows: list[LayerComparison] = []
    for layer in range(compared):
        va, vb = per_layer_a[layer], per_layer_b[layer]
        if len(va) != len(vb):
            raise ValueError(
                f"hidden-size mismatch at layer_{layer}: "
                f"{len(va)} vs {len(vb)}")
        na, nb = _norm(va), _norm(vb)
        cosine = (_dot(va, vb) / (na * nb)) if na > 0 and nb > 0 else None
        ratio = (nb / na) if na > 0 else None
        rows.append(LayerComparison(layer=layer, norm_a=na, norm_b=nb,
                                    cosine=cosine, norm_ratio=ratio))
    return ParityReport(
        artifact_a=ArtifactSummary(name=name_a, layer_count=len(per_layer_a),
                                   hidden_size=hidden_a),
        artifact_b=ArtifactSummary(name=name_b, layer_count=len(per_layer_b),
                                   hidden_size=hidden_b),
        per_layer=rows, threshold=threshold)


def _artifact_base(path: str) -> tuple[str, str]:
    """``…/foo.safetensors`` (or ``…/foo.json`` or extension-less ``…/foo``)
    → ``(directory, name)`` for :func:`vector_store.load`."""
    base = path
    for suffix in (".safetensors", ".json"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return os.path.dirname(base) or ".", os.path.basename(base)


def compare_paths(path_a: str, path_b: str,
                  *, threshold: float = DEFAULT_THRESHOLD) -> ParityReport:
    dir_a, name_a = _artifact_base(path_a)
    dir_b, name_b = _artifact_base(path_b)
    vectors_a, _ = vector_store.load(dir_a, name_a)
    vectors_b, _ = vector_store.load(dir_b, name_b)
    return compare_vectors(name_a, vectors_a.per_layer,
                           name_b, vectors_b.per_layer, threshold=threshold)
