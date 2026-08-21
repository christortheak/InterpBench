"""OptVec family geometry — the cross-run shape of a set of solutions
(plan of record: ``docs/OPTVEC-OPTIMIZED-INJECTION-VECTORS-PLAN.md``, WP4).

OptVec certifies SUFFICIENCY, and multiple sufficient directions are expected;
the plan makes the **solution family a first-class result** rather than a
robustness caveat (§1). This verb reads a set of finished artifacts — the same
layer across seeds, conditions, or comparison vectors from the ordinary library
— and reports the three things that describe a family:

* the pairwise **cosine matrix** (symmetric, unit diagonal);
* the **participation ratio** of the singular values of the stacked
  ``[N, d_model]`` matrix,

      PR = (Σ σ_i²)² / Σ σ_i⁴

  the effective-rank statistic: 1 when every solution is the same direction, N
  when N equal-norm solutions are mutually orthogonal, and something in between
  for a family that occupies a low-dimensional subspace;
* the **per-vector norms**, so a reader can see whether PR is being carried by
  one long vector (which is why the unit-normalized PR is reported beside it).

Comparison vectors that are not OptVec artifacts are welcome — a CAA or
grand-mean direction at the same layer is exactly the contrast the family
should be read against — as long as they have a nonzero row there; each entry
records its ``extractionMethod`` so the table never conflates the two.

Inference-free: this reads artifacts off disk and loads no model.

The module's second verb — :func:`fracture` — reads the same artifacts through
the same helpers, but grouped: PER ITEM and PER DOSE, over the restarts of a
per-item campaign (``docs/OPTVEC-…-PLAN.md`` §S4). Where :func:`geometry`
answers "how many directions is this family?", :func:`fracture` answers "how
many BASINS does this item's optimization land in, and at what dose does that
number leave 1?" — the fracture-α readout. Both are counts; neither carries a
verdict.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from . import paths
from .optvec_eval import (LoadedArtifact, OptVecArtifactError, load_artifact,
                          sha256_file)
from .run_config import write_run_config

RUN_TYPE = "optvec-geometry"
GEOMETRY_JSON = "geometry.json"
SCHEMA_VERSION = 1

#: The fracture verb's own immutable run directory and artifact.
FRACTURE_RUN_TYPE = "optvec-fracture"
FRACTURE_JSON = "fracture.json"

#: Two solutions are in the same basin when their cosine is at least this.
#: A parameter, not a constant of nature — it is stated in the report beside
#: every count it produced, because a cluster count without its threshold is
#: not a number anyone can read.
DEFAULT_CLUSTER_THRESHOLD = 0.9

#: How the threshold builds clusters, stated in the artifact for the same
#: reason ``PR_FORMULA`` is.
CLUSTER_RULE = (
    "single-linkage over the graph on solutions with an edge wherever the "
    "SIGNED cosine at this layer is >= threshold (transitive closure: a chain "
    "of near-duplicates is one basin); clusters are ordered by their smallest "
    "member index, members by input order")

#: Stated in the artifact as well as here, so a reader never has to guess which
#: of the several "effective rank" definitions produced the number.
PR_FORMULA = "PR = (sum sigma_i^2)^2 / sum sigma_i^4 over the singular values of the stacked [N, d_model] matrix of layer rows"


class OptVecGeometryError(ValueError):
    """A set of artifacts whose geometry cannot be computed as asked."""


_CONFIG_KEYS = {"name": "name", "artifacts": "artifacts", "layer": "layer"}


@dataclass
class OptVecGeometryConfig:
    artifacts: list[str] = field(default_factory=list)
    name: str | None = None
    #: Optional explicit layer. When given it must AGREE with every optvec
    #: artifact's own layer — it selects the shared layer for a set of plain
    #: comparison vectors, it never overrides a solution's own.
    layer: int | None = None

    def __post_init__(self) -> None:
        self.artifacts = [str(a) for a in self.artifacts]
        if len(self.artifacts) < 2:
            raise OptVecGeometryError(
                "geometry needs at least 2 artifacts — a cosine matrix and a "
                "participation ratio over one vector are not statistics")
        if self.layer is not None and int(self.layer) < 0:
            raise OptVecGeometryError("layer must be >= 0")

    @classmethod
    def from_dict(cls, payload: dict) -> "OptVecGeometryConfig":
        if not isinstance(payload, dict):
            raise OptVecGeometryError(
                "the OptVec geometry config must be a JSON object")
        unknown = sorted(set(payload) - set(_CONFIG_KEYS))
        if unknown:
            raise OptVecGeometryError(
                "unknown OptVec geometry config key(s): " + ", ".join(unknown))
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list) or not all(
                isinstance(a, str) for a in artifacts):
            raise OptVecGeometryError(
                "'artifacts' must be a list of artifact paths "
                "(extension-less vector locators)")
        return cls(**{field_name: payload[key]
                      for key, field_name in _CONFIG_KEYS.items()
                      if key in payload})


def load_config(path: str) -> OptVecGeometryConfig:
    with open(path, encoding="utf-8") as handle:
        return OptVecGeometryConfig.from_dict(json.load(handle))


# ------------------------------------------------------------------- math


def participation_ratio(singular_values) -> float:
    """``(Σ σ²)² / Σ σ⁴`` — the effective number of directions the stacked
    matrix occupies. Degenerate (all-zero) input returns 0.0."""
    squares = [float(s) ** 2 for s in singular_values]
    numerator = sum(squares) ** 2
    denominator = sum(s ** 2 for s in squares)
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def cosine_matrix(rows: list[list[float]]) -> list[list[float]]:
    """Symmetric matrix with a unit diagonal. Computed from the normalized rows
    once, so ``M[i][j]`` and ``M[j][i]`` are the SAME float rather than two
    round-offs of one quantity."""
    import numpy as np

    matrix = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / norms
    gram = unit @ unit.T
    gram = (gram + gram.T) / 2.0
    np.fill_diagonal(gram, 1.0)
    return [[float(x) for x in row] for row in gram]


def _singular_values(rows: list[list[float]], *, normalize: bool) -> list[float]:
    import numpy as np

    matrix = np.asarray(rows, dtype=np.float64)
    if normalize:
        matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return [float(s) for s in np.linalg.svd(matrix, compute_uv=False)]


# ------------------------------------------------------------------ driver


def _entry_layer(artifact: LoadedArtifact) -> int | None:
    block = artifact.optvec
    if block is None:
        return None
    layer = block.get("layer")
    if not isinstance(layer, int) or isinstance(layer, bool):
        raise OptVecGeometryError(
            f"'{artifact.reference}' has an optvec block whose layer is "
            f"{layer!r}, not an integer")
    return layer


def resolve_layer(artifacts: list[LoadedArtifact],
                  declared: int | None) -> int:
    """The one layer the family is read at.

    Directions at different layers live in different bases; a cosine between
    them is a number with no meaning, so a mixed-layer set refuses rather than
    averaging over a category error.
    """
    layers = {}
    for artifact in artifacts:
        layer = _entry_layer(artifact)
        if layer is not None:
            layers.setdefault(layer, []).append(artifact.reference)
    if len(layers) > 1:
        detail = "; ".join(f"layer {layer}: " + ", ".join(refs)
                           for layer, refs in sorted(layers.items()))
        raise OptVecGeometryError(
            "artifacts name different optvec layers (" + detail + ") — a "
            "cosine between directions at different layers compares two "
            "different bases; run one geometry per layer")
    if layers:
        only = next(iter(layers))
        if declared is not None and int(declared) != only:
            raise OptVecGeometryError(
                f"declared layer {declared} disagrees with the artifacts' own "
                f"optvec layer {only}")
        return only
    if declared is None:
        raise OptVecGeometryError(
            "no artifact carries an optvec block naming a layer, and the "
            "config declares none — say which layer this set is read at")
    return int(declared)


def geometry(config: OptVecGeometryConfig, root: str | None = None) -> dict:
    """Compute the family geometry and write its own immutable run directory."""
    artifacts = [load_artifact(reference) for reference in config.artifacts]
    layer = resolve_layer(artifacts, config.layer)

    rows: list[list[float]] = []
    entries: list[dict] = []
    width: int | None = None
    for artifact in artifacts:
        row = artifact.row(layer)
        if row is None:
            raise OptVecGeometryError(
                f"'{artifact.reference}' has no nonzero row at layer {layer} "
                f"({artifact.vectors.layer_count} layer(s) present) — it "
                "contributes no direction to this family")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise OptVecGeometryError(
                f"'{artifact.reference}' is {len(row)}-dimensional; the first "
                f"artifact is {width}-dimensional — different models' bases")
        rows.append(row)
        entries.append({
            "reference": artifact.reference,
            "name": artifact.name,
            "extractionMethod": artifact.sidecar.extractionMethod,
            "concept": artifact.sidecar.concept,
            "modelID": artifact.sidecar.modelID,
            "revision": artifact.sidecar.revision,
            "isOptVec": artifact.is_optvec,
            "optvecLayer": _entry_layer(artifact),
            "seed": (artifact.optvec or {}).get("seed"),
            "alphaAbsolute": (artifact.optvec or {}).get("alphaAbsolute"),
            "norm": artifact.vectors.norm(layer),
            "tensorSHA256": artifact.tensor_sha256,
            "sidecarSHA256": artifact.sidecar_sha256})

    singular = _singular_values(rows, normalize=False)
    singular_unit = _singular_values(rows, normalize=True)
    report = {
        "schemaVersion": SCHEMA_VERSION,
        "runType": RUN_TYPE,
        "layer": layer,
        "count": len(rows),
        "hiddenSize": width,
        "claim": "sufficiency",
        "formula": PR_FORMULA,
        "entries": entries,
        "cosineMatrix": cosine_matrix(rows),
        "singularValues": singular,
        "participationRatio": participation_ratio(singular),
        # The raw PR is dominated by the longest vectors when norms differ
        # (an artifact library mixes α conventions), so the unit-normalized
        # twin — every direction weighted equally — is reported beside it and
        # is the one to read for "how many directions is this family?".
        "singularValuesUnitNormalized": singular_unit,
        "participationRatioUnitNormalized": participation_ratio(singular_unit),
        "perVectorNorms": [entry["norm"] for entry in entries],
    }

    name = (config.name or "").strip() or f"L{layer}-{len(rows)}"
    run_directory = paths.make_unique_run_directory(
        f"optvec-geometry-{name}", root)
    report["runID"] = os.path.basename(os.path.normpath(run_directory))
    with open(os.path.join(run_directory, GEOMETRY_JSON), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    model_ids = {e["modelID"] for e in entries if e["modelID"]}
    write_run_config(
        run_directory, RUN_TYPE,
        model_id=next(iter(model_ids)) if len(model_ids) == 1 else None,
        notes={"layer": layer, "count": len(rows),
               "formula": PR_FORMULA,
               "participationRatio": report["participationRatio"],
               "participationRatioUnitNormalized":
                   report["participationRatioUnitNormalized"],
               "artifacts": [e["reference"] for e in entries],
               "modelIDs": sorted(model_ids),
               "claim": "sufficiency"})
    return {"runDirectory": run_directory, **report}


# ---------------------------------------------------------------- fracture


_FRACTURE_CONFIG_KEYS = {"name": "name", "artifacts": "artifacts",
                         "layer": "layer", "items": "items",
                         "gradients": "gradients", "threshold": "threshold"}


@dataclass
class OptVecFractureConfig:
    """A set of per-item solutions to count basins over.

    ``items`` is the explicit grouping argument: artifact reference → item id,
    for artifacts whose sidecar carries no item marker. Nothing here infers an
    item from a file name or a run id — an ungrouped artifact refuses, because
    a wrongly grouped solution silently changes a cluster count.
    """

    artifacts: list[str] = field(default_factory=list)
    name: str | None = None
    layer: int | None = None
    items: dict = field(default_factory=dict)
    #: Optional ``gradients.safetensors`` from a gradient-survey run. Present →
    #: every cluster reports its mean cosine to that item's gradient row;
    #: absent → the field is absent, never a placeholder.
    gradients: str | None = None
    threshold: float = DEFAULT_CLUSTER_THRESHOLD

    def __post_init__(self) -> None:
        self.artifacts = [str(a) for a in self.artifacts]
        if len(self.artifacts) < 2:
            raise OptVecGeometryError(
                "fracture needs at least 2 solution artifacts — a basin count "
                "over one restart is not a multiplicity statistic")
        if len(set(self.artifacts)) != len(self.artifacts):
            raise OptVecGeometryError(
                "fracture artifacts repeat a reference — the same solution "
                "counted twice inflates a basin's frequency")
        if not isinstance(self.items, dict):
            raise OptVecGeometryError(
                "'items' must be an object mapping artifact reference → item "
                "id")
        for reference, item in self.items.items():
            if not isinstance(item, str) or not item.strip():
                raise OptVecGeometryError(
                    f"items[{reference!r}] must be a non-empty item id")
            if reference not in self.artifacts:
                raise OptVecGeometryError(
                    f"items names {reference!r}, which is not one of the "
                    "artifacts — a mapping key that matches nothing would "
                    "silently leave its artifact ungrouped")
        self.threshold = float(self.threshold)
        if not -1.0 <= self.threshold <= 1.0:
            raise OptVecGeometryError(
                "threshold is a cosine: it must lie in [-1, 1]")
        if self.layer is not None and int(self.layer) < 0:
            raise OptVecGeometryError("layer must be >= 0")

    @classmethod
    def from_dict(cls, payload: dict) -> "OptVecFractureConfig":
        if not isinstance(payload, dict):
            raise OptVecGeometryError(
                "the OptVec fracture config must be a JSON object")
        unknown = sorted(set(payload) - set(_FRACTURE_CONFIG_KEYS))
        if unknown:
            raise OptVecGeometryError(
                "unknown OptVec fracture config key(s): " + ", ".join(unknown))
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list) or not all(
                isinstance(a, str) for a in artifacts):
            raise OptVecGeometryError(
                "'artifacts' must be a list of artifact paths "
                "(extension-less vector locators)")
        return cls(**{field_name: payload[key]
                      for key, field_name in _FRACTURE_CONFIG_KEYS.items()
                      if key in payload})


def load_fracture_config(path: str) -> OptVecFractureConfig:
    with open(path, encoding="utf-8") as handle:
        return OptVecFractureConfig.from_dict(json.load(handle))


#: Where an item marker may live in an ``optvec`` sidecar block, in the order
#: read. The training driver's block nests its dataset facts, and hand-built
#: or older blocks carry them flat, so both are read — the ``condition_of``
#: idiom in ``optvec_interpret``.
_ITEM_MARKER_BLOCKS = ("datasets", "training", "objective")


def item_of(optvec_block: dict | None) -> str | None:
    """The single item an artifact was trained on, read from its sidecar.

    Two spellings are accepted: ``item`` (a string) and ``itemFilter`` (the
    campaign's per-item key — a list of exactly ONE id). A filter naming
    several items is not a per-item vector and returns ``None`` rather than a
    guess at which of them "the" item is; so does a block with no marker at
    all. Callers supply the item explicitly in that case.
    """
    if not isinstance(optvec_block, dict):
        return None
    sources = [optvec_block]
    for key in _ITEM_MARKER_BLOCKS:
        nested = optvec_block.get(key)
        if isinstance(nested, dict):
            sources.append(nested)
    for source in sources:
        value = source.get("item")
        if isinstance(value, str) and value.strip():
            return value
        value = source.get("itemFilter")
        if (isinstance(value, list) and len(value) == 1
                and isinstance(value[0], str) and value[0].strip()):
            return value[0]
    return None


def cluster_by_cosine(rows: list[list[float]],
                      threshold: float) -> list[list[int]]:
    """Single-linkage clustering of directions at one threshold
    (:data:`CLUSTER_RULE`); returns member-index lists.

    Built on :func:`cosine_matrix` rather than a second normalization pass, so
    the numbers a cluster is built from are the numbers the report prints.
    """
    if not rows:
        return []
    matrix = cosine_matrix(rows)
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if matrix[i][j] >= threshold:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[max(root_i, root_j)] = min(root_i, root_j)

    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    return [groups[root] for root in sorted(groups)]


def _cosine(a: list[float], b: list[float]) -> float:
    return cosine_matrix([a, b])[0][1]


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def load_gradient_rows(path: str) -> tuple[dict, dict]:
    """``(item id → gradient row, provenance)`` from a gradient-survey file.

    Two on-disk forms are read, and nothing else is inferred:

    * **one tensor per item**, keyed by the item id (1-D, or ``[1, d]``);
    * **one stacked ``[N, d]`` tensor** plus an item index — either the
      safetensors header's ``items`` metadata (a JSON array) or a sibling
      ``<name>.json`` / ``gradients.json`` carrying ``{"items": [...]}``.
      Row order is item order.

    A file in neither form refuses: a gradient reference that cannot say which
    row belongs to which item would otherwise become a cosine against an
    arbitrary vector.
    """
    from safetensors import safe_open

    resolved = paths.resolve(path)
    if not os.path.exists(resolved):
        raise OptVecGeometryError(
            f"gradient reference '{path}' names no file")
    try:
        with safe_open(resolved, framework="numpy") as handle:
            keys = list(handle.keys())
            metadata = handle.metadata() or {}
            tensors = {key: handle.get_tensor(key) for key in keys}
    except OptVecGeometryError:
        raise
    except Exception as exc:  # noqa: BLE001 - one typed error names the file
        raise OptVecGeometryError(
            f"gradient reference '{path}' could not be read: {exc}") from exc

    provenance = {"path": resolved, "sha256": sha256_file(resolved)}

    index = _gradient_item_index(resolved, metadata)
    if index is not None and len(keys) == 1:
        stacked = tensors[keys[0]]
        if getattr(stacked, "ndim", 0) != 2 or len(stacked) != len(index):
            raise OptVecGeometryError(
                f"gradient reference '{path}' names {len(index)} items but its "
                f"'{keys[0]}' tensor has shape {getattr(stacked, 'shape', None)}"
                " — the index and the rows disagree")
        rows = {item: [float(x) for x in stacked[i]]
                for i, item in enumerate(index)}
        provenance.update({"form": "stacked", "tensor": keys[0],
                           "itemCount": len(rows)})
        return rows, provenance

    rows = {}
    for key, tensor in tensors.items():
        array = tensor
        if getattr(array, "ndim", 0) == 2 and len(array) == 1:
            array = array[0]
        if getattr(array, "ndim", 0) != 1:
            raise OptVecGeometryError(
                f"gradient reference '{path}' tensor {key!r} has shape "
                f"{getattr(tensor, 'shape', None)} — a per-item gradient file "
                "holds one 1-D row per item id, or one stacked [N, d] tensor "
                "with an item index")
        rows[key] = [float(x) for x in array]
    provenance.update({"form": "perItem", "itemCount": len(rows)})
    return rows, provenance


def _gradient_item_index(resolved: str, metadata: dict) -> list | None:
    """The item order of a stacked gradient tensor, from the safetensors
    header or a sibling JSON. ``None`` when neither states one."""
    raw = metadata.get("items") if isinstance(metadata, dict) else None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, list) and all(isinstance(i, str) for i in parsed):
            return parsed
    stem = resolved[:-len(".safetensors")] if resolved.endswith(
        ".safetensors") else resolved
    candidates = [stem + ".json",
                  os.path.join(os.path.dirname(resolved) or ".",
                               "gradients.json")]
    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        items = payload.get("items") if isinstance(payload, dict) else None
        if isinstance(items, list) and all(isinstance(i, str) for i in items):
            return items
    return None


def _fracture_entry(artifact: LoadedArtifact, config: OptVecFractureConfig,
                    layer: int) -> dict:
    """One solution's identity, group keys and direction."""
    block = artifact.optvec or {}
    marked = item_of(block)
    mapped = config.items.get(artifact.reference)
    if marked and mapped and marked != mapped:
        raise OptVecGeometryError(
            f"'{artifact.reference}' is marked as item {marked!r} in its "
            f"sidecar but mapped to {mapped!r} by the config — one of them is "
            "wrong, and guessing which would move a solution between basins")
    item = mapped or marked
    if not item:
        raise OptVecGeometryError(
            f"'{artifact.reference}' names no item: its optvec block carries "
            "neither 'item' nor a single-id 'itemFilter', and the config's "
            "'items' mapping does not cover it. Fracture groups per item; an "
            "ungrouped solution is never guessed into a group")
    dose = block.get("alphaAbsolute")
    if isinstance(dose, bool) or not isinstance(dose, (int, float)):
        raise OptVecGeometryError(
            f"'{artifact.reference}' has no numeric optvec.alphaAbsolute — the "
            "dose axis is read from the artifact, never assumed")
    row = artifact.row(layer)
    if row is None:
        raise OptVecGeometryError(
            f"'{artifact.reference}' has no nonzero row at layer {layer} "
            f"({artifact.vectors.layer_count} layer(s) present) — it "
            "contributes no direction to this readout")
    return {"reference": artifact.reference, "name": artifact.name,
            "item": item, "itemSource": "config" if mapped else "sidecar",
            "dose": float(dose), "seed": block.get("seed"),
            "isOptVec": artifact.is_optvec,
            "extractionMethod": artifact.sidecar.extractionMethod,
            "modelID": artifact.sidecar.modelID,
            "revision": artifact.sidecar.revision,
            "norm": artifact.vectors.norm(layer),
            "tensorSHA256": artifact.tensor_sha256,
            "sidecarSHA256": artifact.sidecar_sha256,
            "row": row}


def fracture(config: OptVecFractureConfig, root: str | None = None) -> dict:
    """Per-item, per-dose basin counts over a set of solutions; writes its own
    immutable run directory.

    The report is counts, frequencies and cosines. What a rise in cluster
    count MEANS — that the optimization has left the dose range where the
    objective is effectively linear in the injection — is the plan's argument,
    not this artifact's, and no word here adjudicates it.
    """
    artifacts = [load_artifact(reference) for reference in config.artifacts]
    layer = resolve_layer(artifacts, config.layer)
    entries = [_fracture_entry(artifact, config, layer)
               for artifact in artifacts]

    width = len(entries[0]["row"])
    for entry in entries:
        if len(entry["row"]) != width:
            raise OptVecGeometryError(
                f"'{entry['reference']}' is {len(entry['row'])}-dimensional; "
                f"the first artifact is {width}-dimensional — different "
                "models' bases")

    gradient_rows: dict = {}
    gradient_provenance: dict | None = None
    advisories: list[str] = []
    if config.gradients:
        gradient_rows, gradient_provenance = load_gradient_rows(config.gradients)
        for item, row in gradient_rows.items():
            if len(row) != width:
                raise OptVecGeometryError(
                    f"gradient row for item {item!r} is {len(row)}-dimensional; "
                    f"the solutions are {width}-dimensional")
            if not any(row):
                raise OptVecGeometryError(
                    f"gradient row for item {item!r} is exactly zero — a "
                    "cosine to it is undefined, and reporting one would "
                    "invent a direction the survey did not find")

    groups: dict[tuple, list[dict]] = {}
    for entry in entries:
        groups.setdefault((entry["item"], entry["dose"]), []).append(entry)

    missing_gradient: list[str] = []
    group_reports: list[dict] = []
    for (item, dose) in sorted(groups, key=lambda key: (key[0], key[1])):
        members = groups[(item, dose)]
        rows = [member["row"] for member in members]
        matrix = cosine_matrix(rows)
        clusters = cluster_by_cosine(rows, config.threshold)
        gradient = gradient_rows.get(item) if config.gradients else None
        if config.gradients and gradient is None and item not in missing_gradient:
            missing_gradient.append(item)

        cluster_reports = []
        for index, indices in enumerate(clusters):
            within = [matrix[i][j] for i in indices for j in indices if i < j]
            cluster: dict = {
                "cluster": index,
                "size": len(indices),
                # The basin frequency: the fraction of this (item, dose) cell's
                # RESTARTS that landed here.
                "frequency": len(indices) / len(members),
                "members": [{"reference": members[i]["reference"],
                             "seed": members[i]["seed"],
                             "norm": members[i]["norm"]}
                            for i in indices],
                "meanWithinClusterCosine": _mean(within) if within else 1.0,
            }
            if gradient is not None:
                cosines = [_cosine(rows[i], gradient) for i in indices]
                for member, cosine in zip(cluster["members"], cosines):
                    member["gradientCosine"] = cosine
                cluster["meanGradientCosine"] = _mean(cosines)
            cluster_reports.append(cluster)

        singular_unit = _singular_values(rows, normalize=True)
        group_reports.append({
            "item": item,
            "dose": dose,
            "restarts": len(members),
            "clusterCount": len(clusters),
            "seeds": [member["seed"] for member in members],
            "references": [member["reference"] for member in members],
            "clusters": cluster_reports,
            "cosineMatrix": matrix,
            "participationRatioUnitNormalized":
                participation_ratio(singular_unit),
            "gradientReferencePresent": gradient is not None,
        })

    for item in missing_gradient:
        advisories.append(
            f"gradient reference carries no row for item {item!r} — its "
            "clusters report no gradient cosine (an absent row is absent, not "
            "zero)")

    by_item: dict[str, list[dict]] = {}
    for group in group_reports:
        by_item.setdefault(group["item"], []).append(
            {"dose": group["dose"], "restarts": group["restarts"],
             "clusterCount": group["clusterCount"]})
    fracture_table = [
        {"item": item,
         "doses": sorted(rows_, key=lambda row: row["dose"])}
        for item, rows_ in sorted(by_item.items())]

    report = {
        "schemaVersion": SCHEMA_VERSION,
        "runType": FRACTURE_RUN_TYPE,
        "layer": layer,
        "count": len(entries),
        "hiddenSize": width,
        "claim": "sufficiency",
        "threshold": config.threshold,
        "clusterRule": CLUSTER_RULE,
        "formula": PR_FORMULA,
        "itemCount": len(by_item),
        "entries": [{key: value for key, value in entry.items() if key != "row"}
                    for entry in entries],
        "groups": group_reports,
        "fractureTable": fracture_table,
        "gradientReference": gradient_provenance,
        "advisories": advisories,
    }

    name = (config.name or "").strip() or f"L{layer}-{len(by_item)}items"
    run_directory = paths.make_unique_run_directory(
        f"optvec-fracture-{name}", root)
    report["runID"] = os.path.basename(os.path.normpath(run_directory))
    with open(os.path.join(run_directory, FRACTURE_JSON), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    model_ids = {e["modelID"] for e in entries if e["modelID"]}
    write_run_config(
        run_directory, FRACTURE_RUN_TYPE,
        model_id=next(iter(model_ids)) if len(model_ids) == 1 else None,
        notes={"layer": layer, "count": len(entries),
               "threshold": config.threshold,
               "clusterRule": CLUSTER_RULE,
               "items": sorted(by_item),
               "doses": sorted({group["dose"] for group in group_reports}),
               "clusterCounts": {
                   f"{group['item']}@{group['dose']}": group["clusterCount"]
                   for group in group_reports},
               "gradientReference": gradient_provenance,
               "artifacts": [e["reference"] for e in entries],
               "modelIDs": sorted(model_ids),
               "advisories": advisories,
               "claim": "sufficiency"})
    return {"runDirectory": run_directory, **report}


__all__ = ["OptVecGeometryConfig", "OptVecGeometryError", "RUN_TYPE",
           "GEOMETRY_JSON", "PR_FORMULA", "cosine_matrix", "geometry",
           "load_config", "participation_ratio", "resolve_layer",
           "OptVecArtifactError",
           "OptVecFractureConfig", "FRACTURE_RUN_TYPE", "FRACTURE_JSON",
           "CLUSTER_RULE", "DEFAULT_CLUSTER_THRESHOLD", "cluster_by_cosine",
           "fracture", "item_of", "load_fracture_config", "load_gradient_rows"]
