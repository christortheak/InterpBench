"""OptVec family geometry: the cosine matrix, the participation-ratio
effective-rank statistic, the mixed-layer refusal, and the run directory.

Hand-built artifacts only — no model, no inference. Every expected number is
linear algebra with a closed form.
"""

import json
import math
import os

import pytest

from steerlab_server.experiment import optvec_geometry
from steerlab_server.experiment.optvec_geometry import (OptVecGeometryConfig,
                                                        OptVecGeometryError)
from tests.test_optvec_eval import (HIDDEN, LAYERS, OPTVEC_LAYER,
                                    write_artifact, write_optvec_artifact)


def _basis(index: int, scale: float = 1.0) -> list[float]:
    row = [0.0] * HIDDEN
    row[index] = scale
    return row


# ------------------------------------------------------------------ pure math


def test_cosine_matrix_is_symmetric_with_a_unit_diagonal():
    rows = [_basis(0), _basis(1, 3.0), [0.6, 0.8] + [0.0] * (HIDDEN - 2)]
    matrix = optvec_geometry.cosine_matrix(rows)
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    for i in range(3):
        assert matrix[i][i] == pytest.approx(1.0, abs=1e-12)
        for j in range(3):
            # Byte-identical, not merely close: the two halves are one number.
            assert matrix[i][j] == matrix[j][i]
    assert matrix[0][1] == pytest.approx(0.0, abs=1e-12)
    assert matrix[0][2] == pytest.approx(0.6, abs=1e-9)
    assert matrix[1][2] == pytest.approx(0.8, abs=1e-9)
    # Scale-free: a longer vector has the same cosines.
    scaled = optvec_geometry.cosine_matrix(
        [_basis(0, 17.0), _basis(1, 3.0), [0.6, 0.8] + [0.0] * (HIDDEN - 2)])
    assert scaled[0][2] == pytest.approx(matrix[0][2], abs=1e-9)


def test_participation_ratio_is_one_for_identical_and_n_for_orthogonal():
    """PR = (Σσ²)²/Σσ⁴ — 1 direction however many copies, N for N orthogonal
    equal-norm directions."""
    import numpy as np

    identical = [_basis(0), _basis(0), _basis(0)]
    singular = np.linalg.svd(np.asarray(identical), compute_uv=False)
    assert optvec_geometry.participation_ratio(singular) == pytest.approx(1.0)

    for n in (2, 3, 5):
        rows = [_basis(i) for i in range(n)]
        singular = np.linalg.svd(np.asarray(rows), compute_uv=False)
        assert optvec_geometry.participation_ratio(singular) == \
            pytest.approx(float(n), rel=1e-9)

    # Unequal norms pull the raw PR down toward the dominant direction —
    # exactly why the report carries the unit-normalized twin as well.
    rows = [_basis(0, 10.0), _basis(1, 0.1)]
    singular = np.linalg.svd(np.asarray(rows), compute_uv=False)
    assert optvec_geometry.participation_ratio(singular) < 1.01
    assert optvec_geometry.participation_ratio([0.0, 0.0]) == 0.0


# --------------------------------------------------------------- config rules


def test_config_needs_two_artifacts_and_refuses_unknown_keys():
    with pytest.raises(OptVecGeometryError):
        OptVecGeometryConfig(artifacts=["only/one"])
    with pytest.raises(OptVecGeometryError) as exc:
        OptVecGeometryConfig.from_dict({"artifacts": ["a", "b"],
                                        "layers": 2})
    assert "layers" in str(exc.value)
    with pytest.raises(OptVecGeometryError):
        OptVecGeometryConfig.from_dict({"artifacts": "a"})
    assert OptVecGeometryConfig.from_dict(
        {"artifacts": ["a", "b"], "name": "fam", "layer": 2}).layer == 2


def test_mixed_layers_refuse(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    a = write_optvec_artifact(tmp_path / "v", "seed-a", layer=2)
    b = write_optvec_artifact(tmp_path / "v", "seed-b", layer=1)
    with pytest.raises(OptVecGeometryError) as exc:
        optvec_geometry.geometry(OptVecGeometryConfig(artifacts=[a, b]))
    message = str(exc.value)
    assert "different optvec layers" in message
    assert "layer 1" in message and "layer 2" in message

    # A declared layer that disagrees with the artifacts' own is refused too:
    # it selects a shared layer for plain comparison vectors, it never
    # overrides a solution's.
    c = write_optvec_artifact(tmp_path / "v", "seed-c", layer=2)
    with pytest.raises(OptVecGeometryError) as exc:
        optvec_geometry.geometry(
            OptVecGeometryConfig(artifacts=[a, c], layer=3))
    assert "disagrees" in str(exc.value)

    # And a set with no optvec block at all must say which layer it is read at.
    per_layer = [[0.0] * HIDDEN for _ in range(LAYERS)]
    per_layer[OPTVEC_LAYER] = _basis(0)
    plain_a = write_artifact(tmp_path / "lib", "plain-a", per_layer=per_layer,
                             extraction_method="meanDifference")
    plain_b = write_artifact(tmp_path / "lib", "plain-b", per_layer=per_layer,
                             extraction_method="meanDifference")
    with pytest.raises(OptVecGeometryError) as exc:
        optvec_geometry.geometry(
            OptVecGeometryConfig(artifacts=[plain_a, plain_b]))
    assert "which layer" in str(exc.value)
    report = optvec_geometry.geometry(
        OptVecGeometryConfig(artifacts=[plain_a, plain_b],
                             layer=OPTVEC_LAYER))
    assert report["layer"] == OPTVEC_LAYER


# ------------------------------------------------------------------ end to end


def test_geometry_run_directory_and_report(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    # Three solutions: two identical directions at different α, one
    # orthogonal — PR on the unit-normalized stack is therefore 2.
    a = write_optvec_artifact(tmp_path / "v", "seed-a", direction=_basis(0, 6.0))
    b = write_optvec_artifact(tmp_path / "v", "seed-b", direction=_basis(0, 3.0),
                              alpha=3.0)
    c = write_optvec_artifact(tmp_path / "v", "seed-c", direction=_basis(1, 6.0))
    # A non-OptVec comparison vector at the same layer is welcome.
    per_layer = [[0.0] * HIDDEN for _ in range(LAYERS)]
    per_layer[OPTVEC_LAYER] = [0.6 * 6.0, 0.8 * 6.0] + [0.0] * (HIDDEN - 2)
    caa = write_artifact(tmp_path / "lib", "caa-companion", per_layer=per_layer,
                         extraction_method="meanDifference")

    report = optvec_geometry.geometry(
        OptVecGeometryConfig(artifacts=[a, b, c, caa], name="family"))
    run_dir = report["runDirectory"]
    assert os.path.basename(run_dir).endswith("optvec-geometry-family")

    on_disk = json.load(
        open(os.path.join(run_dir, optvec_geometry.GEOMETRY_JSON)))
    assert on_disk["layer"] == OPTVEC_LAYER
    assert on_disk["count"] == 4 and on_disk["hiddenSize"] == HIDDEN
    assert on_disk["formula"] == optvec_geometry.PR_FORMULA

    matrix = on_disk["cosineMatrix"]
    assert matrix[0][1] == pytest.approx(1.0, abs=1e-6)       # same direction
    assert matrix[0][2] == pytest.approx(0.0, abs=1e-6)       # orthogonal
    assert matrix[0][3] == pytest.approx(0.6, abs=1e-6)       # the CAA vector
    for i in range(4):
        assert matrix[i][i] == pytest.approx(1.0, abs=1e-12)
        for j in range(4):
            assert matrix[i][j] == matrix[j][i]

    # Three unit directions spanning a 2-D subspace: PR is strictly between
    # 1 and 2 in the raw stack too, and the norms are reported per vector.
    assert 1.0 < on_disk["participationRatioUnitNormalized"] < 2.0
    assert on_disk["perVectorNorms"] == pytest.approx(
        [6.0, 3.0, 6.0, 6.0], rel=1e-5)
    assert len(on_disk["singularValues"]) == 2 or \
        sum(1 for s in on_disk["singularValues"] if s > 1e-9) == 2

    entries = on_disk["entries"]
    assert [e["isOptVec"] for e in entries] == [True, True, True, False]
    assert [e["extractionMethod"] for e in entries] == [
        "optvec", "optvec", "optvec", "meanDifference"]
    assert [e["optvecLayer"] for e in entries] == [OPTVEC_LAYER] * 3 + [None]
    assert entries[1]["alphaAbsolute"] == pytest.approx(3.0)
    assert all(e["tensorSHA256"] and e["sidecarSHA256"] for e in entries)

    from tests.test_run_config import CONTRACT_KEYS
    config_json = json.load(open(os.path.join(run_dir, "config.json")))
    assert sorted(config_json.keys()) == CONTRACT_KEYS
    assert config_json["runType"] == "optvec-geometry"
    assert config_json["modelID"] == "test/tiny"
    notes = config_json["notes"]
    assert notes["layer"] == OPTVEC_LAYER and notes["count"] == 4
    assert notes["formula"] == optvec_geometry.PR_FORMULA
    assert notes["participationRatioUnitNormalized"] == pytest.approx(
        on_disk["participationRatioUnitNormalized"])
    assert notes["artifacts"] == [a, b, c, caa]


def test_geometry_refuses_a_zero_row_and_a_width_mismatch(tmp_path,
                                                          monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    a = write_optvec_artifact(tmp_path / "v", "seed-a", direction=_basis(0, 6.0))
    empty = write_artifact(
        tmp_path / "lib", "empty",
        per_layer=[[0.0] * HIDDEN for _ in range(LAYERS)],
        extraction_method="meanDifference")
    with pytest.raises(OptVecGeometryError) as exc:
        optvec_geometry.geometry(OptVecGeometryConfig(artifacts=[a, empty]))
    assert "no nonzero row at layer 2" in str(exc.value)

    narrow_layers = [[0.0] * 8 for _ in range(LAYERS)]
    narrow_layers[OPTVEC_LAYER] = [1.0] * 8
    narrow = write_artifact(tmp_path / "lib", "narrow",
                            per_layer=narrow_layers,
                            extraction_method="meanDifference")
    with pytest.raises(OptVecGeometryError) as exc:
        optvec_geometry.geometry(OptVecGeometryConfig(artifacts=[a, narrow]))
    assert "dimensional" in str(exc.value)


def test_identical_family_has_participation_ratio_one(tmp_path, monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    direction = [math.sqrt(HIDDEN) ** -1 * 6.0] * HIDDEN
    a = write_optvec_artifact(tmp_path / "v", "twin-a", direction=direction)
    b = write_optvec_artifact(tmp_path / "v", "twin-b", direction=direction)
    report = optvec_geometry.geometry(OptVecGeometryConfig(artifacts=[a, b]))
    assert report["participationRatio"] == pytest.approx(1.0, rel=1e-9)
    assert report["participationRatioUnitNormalized"] == pytest.approx(
        1.0, rel=1e-9)
    flattened = [value for row in report["cosineMatrix"] for value in row]
    assert flattened == pytest.approx([1.0, 1.0, 1.0, 1.0], abs=1e-12)


# ------------------------------------------------------- fracture (WP-S4b)
#
# Per-item multiplicity: how many BASINS an item's restarts land in, and how
# that count moves with dose. Hand-built vector sets again — every cluster
# count below is arithmetic on cosines the test states.


def _unit(*coordinates: float) -> list[float]:
    import numpy as np

    row = np.zeros(HIDDEN, dtype=np.float64)
    for index, value in enumerate(coordinates):
        row[index] = value
    return [float(x) for x in row / np.linalg.norm(row)]


def _solution(directory, name, *, item=None, dose=6.0, seed=0, direction=None,
              layer=OPTVEC_LAYER, extra=None):
    """One per-item solution artifact: full depth, one nonzero layer, an
    ``optvec`` block carrying the markers fracture groups by."""
    per_layer = [[0.0] * HIDDEN for _ in range(LAYERS)]
    per_layer[layer] = list(direction if direction is not None else _unit(1.0))
    block = {"layer": layer, "alphaAbsolute": dose, "seed": seed,
             "claim": "sufficiency"}
    if item is not None:
        block["item"] = item
    block.update(extra or {})
    return write_artifact(directory, name, per_layer=per_layer, optvec=block)


def _write_gradients(path, rows, *, stacked=False):
    import json as _json

    import numpy as np
    from safetensors.numpy import save_file

    if stacked:
        items = list(rows)
        matrix = np.asarray([rows[item] for item in items], dtype=np.float32)
        save_file({"gradients": matrix}, str(path),
                  metadata={"items": _json.dumps(items)})
    else:
        save_file({item: np.asarray(row, dtype=np.float32)
                   for item, row in rows.items()}, str(path))
    return str(path)


# ------------------------------------------------------------- pure clustering


def test_clustering_finds_one_basin_and_then_three():
    near = [_unit(1.0), _unit(1.0, 0.2), _unit(1.0, -0.15)]
    assert optvec_geometry.cluster_by_cosine(near, 0.9) == [[0, 1, 2]]

    # Three basins with known frequencies: 3, 2, 1 restarts.
    rows = ([_unit(1.0), _unit(1.0, 0.1), _unit(1.0, -0.1)]
            + [_unit(0.0, 1.0), _unit(0.05, 1.0)]
            + [_unit(0.0, 0.0, 1.0)])
    clusters = optvec_geometry.cluster_by_cosine(rows, 0.9)
    assert clusters == [[0, 1, 2], [3, 4], [5]]
    assert [len(c) / len(rows) for c in clusters] == pytest.approx(
        [0.5, 1 / 3, 1 / 6])


def test_clustering_is_threshold_sensitive_and_signed():
    pair = [_unit(1.0), _unit(1.0, 0.2)]            # cosine ≈ 0.9806
    assert optvec_geometry.cluster_by_cosine(pair, 0.9) == [[0, 1]]
    assert optvec_geometry.cluster_by_cosine(pair, 0.99) == [[0], [1]]

    # Single linkage: a chain of near-duplicates is one basin even though its
    # ends are far apart (0.9806² ≈ 0.96 < 0.98 for the ends).
    chain = [_unit(1.0), _unit(1.0, 0.2), _unit(1.0, 0.42)]
    assert optvec_geometry.cluster_by_cosine(chain, 0.97) == [[0, 1, 2]]

    # The cosine is SIGNED: an anti-parallel solution is its own basin.
    opposed = [_unit(1.0), [-x for x in _unit(1.0)]]
    assert optvec_geometry.cluster_by_cosine(opposed, 0.9) == [[0], [1]]


def test_item_of_reads_a_marker_and_never_guesses():
    assert optvec_geometry.item_of({"item": "t-1"}) == "t-1"
    assert optvec_geometry.item_of({"itemFilter": ["t-2"]}) == "t-2"
    assert optvec_geometry.item_of(
        {"datasets": {"itemFilter": ["t-3"]}}) == "t-3"
    # A filter naming several items is not a per-item vector.
    assert optvec_geometry.item_of({"itemFilter": ["t-1", "t-2"]}) is None
    assert optvec_geometry.item_of({"itemFilter": []}) is None
    assert optvec_geometry.item_of({"seed": 3}) is None
    assert optvec_geometry.item_of(None) is None


# ---------------------------------------------------------------- config rules


def test_fracture_config_rules(tmp_path):
    with pytest.raises(OptVecGeometryError):
        optvec_geometry.OptVecFractureConfig(artifacts=["only/one"])
    with pytest.raises(OptVecGeometryError) as exc:
        optvec_geometry.OptVecFractureConfig(artifacts=["a", "a"])
    assert "repeat" in str(exc.value)
    with pytest.raises(OptVecGeometryError) as exc:
        optvec_geometry.OptVecFractureConfig(artifacts=["a", "b"],
                                             threshold=1.4)
    assert "[-1, 1]" in str(exc.value)
    with pytest.raises(OptVecGeometryError) as exc:
        optvec_geometry.OptVecFractureConfig(artifacts=["a", "b"],
                                             items={"c": "t-1"})
    assert "not one of the artifacts" in str(exc.value)
    with pytest.raises(OptVecGeometryError) as exc:
        optvec_geometry.OptVecFractureConfig.from_dict(
            {"artifacts": ["a", "b"], "doses": [1]})
    assert "doses" in str(exc.value)
    config = optvec_geometry.OptVecFractureConfig.from_dict(
        {"artifacts": ["a", "b"], "items": {"a": "t-1"}, "threshold": 0.8,
         "name": "frac", "layer": 2, "gradients": "g.safetensors"})
    assert config.threshold == 0.8 and config.layer == 2
    assert optvec_geometry.OptVecFractureConfig(
        artifacts=["a", "b"]).threshold == \
        optvec_geometry.DEFAULT_CLUSTER_THRESHOLD


def test_a_solution_without_an_item_refuses_unless_mapped(tmp_path,
                                                          monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    unmarked_a = _solution(tmp_path / "v", "no-item-a", direction=_unit(1.0))
    unmarked_b = _solution(tmp_path / "v", "no-item-b", direction=_unit(0, 1.0))

    with pytest.raises(OptVecGeometryError) as exc:
        optvec_geometry.fracture(optvec_geometry.OptVecFractureConfig(
            artifacts=[unmarked_a, unmarked_b]))
    assert "names no item" in str(exc.value)

    report = optvec_geometry.fracture(optvec_geometry.OptVecFractureConfig(
        artifacts=[unmarked_a, unmarked_b],
        items={unmarked_a: "t-1", unmarked_b: "t-1"}))
    assert report["groups"][0]["item"] == "t-1"
    assert [e["itemSource"] for e in report["entries"]] == ["config", "config"]

    # A mapping that contradicts the sidecar refuses rather than picking one.
    marked = _solution(tmp_path / "v", "marked", item="t-1",
                       direction=_unit(1.0))
    with pytest.raises(OptVecGeometryError) as exc:
        optvec_geometry.fracture(optvec_geometry.OptVecFractureConfig(
            artifacts=[marked, unmarked_b],
            items={marked: "t-9", unmarked_b: "t-1"}))
    assert "wrong" in str(exc.value)

    # The dose is read, never assumed.
    doseless = write_artifact(
        tmp_path / "v", "doseless",
        per_layer=[[0.0] * HIDDEN if i != OPTVEC_LAYER else _unit(1.0)
                   for i in range(LAYERS)],
        optvec={"layer": OPTVEC_LAYER, "item": "t-1", "seed": 0})
    with pytest.raises(OptVecGeometryError) as exc:
        optvec_geometry.fracture(optvec_geometry.OptVecFractureConfig(
            artifacts=[marked, doseless]))
    assert "alphaAbsolute" in str(exc.value)


# ------------------------------------------------------------------ end to end


def test_fracture_groups_per_item_and_dose_and_writes_its_run(tmp_path,
                                                              monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    directory = tmp_path / "v"
    # Item t-1 at dose 6: four restarts, three in one basin, one apart.
    artifacts = [
        _solution(directory, "a0", item="t-1", dose=6.0, seed=0,
                  direction=_unit(1.0)),
        _solution(directory, "a1", item="t-1", dose=6.0, seed=1,
                  direction=_unit(1.0, 0.1)),
        _solution(directory, "a2", item="t-1", dose=6.0, seed=2,
                  direction=_unit(1.0, -0.1)),
        _solution(directory, "a3", item="t-1", dose=6.0, seed=3,
                  direction=_unit(0.0, 1.0)),
        # Item t-2 at the same dose: one basin.
        _solution(directory, "b0", item="t-2", dose=6.0, seed=0,
                  direction=_unit(0.0, 0.0, 1.0)),
        _solution(directory, "b1", item="t-2", dose=6.0, seed=1,
                  direction=_unit(0.05, 0.0, 1.0)),
    ]
    report = optvec_geometry.fracture(optvec_geometry.OptVecFractureConfig(
        artifacts=artifacts, name="s4"))

    run_dir = report["runDirectory"]
    assert os.path.basename(run_dir).endswith("optvec-fracture-s4")
    on_disk = json.load(
        open(os.path.join(run_dir, optvec_geometry.FRACTURE_JSON)))
    assert on_disk["runType"] == "optvec-fracture"
    assert on_disk["layer"] == OPTVEC_LAYER and on_disk["count"] == 6
    assert on_disk["threshold"] == optvec_geometry.DEFAULT_CLUSTER_THRESHOLD
    assert on_disk["clusterRule"] == optvec_geometry.CLUSTER_RULE
    assert on_disk["itemCount"] == 2
    assert on_disk["gradientReference"] is None

    groups = {(g["item"], g["dose"]): g for g in on_disk["groups"]}
    first = groups[("t-1", 6.0)]
    assert first["restarts"] == 4 and first["clusterCount"] == 2
    assert [c["size"] for c in first["clusters"]] == [3, 1]
    assert [c["frequency"] for c in first["clusters"]] == pytest.approx(
        [0.75, 0.25])
    assert [m["seed"] for m in first["clusters"][0]["members"]] == [0, 1, 2]
    assert first["clusters"][1]["meanWithinClusterCosine"] == 1.0
    assert all("gradientCosine" not in m
               for c in first["clusters"] for m in c["members"])
    assert first["gradientReferencePresent"] is False
    # Two basins among four unit directions in a plane: PR sits between them.
    assert 1.0 < first["participationRatioUnitNormalized"] < 2.0
    assert groups[("t-2", 6.0)]["clusterCount"] == 1

    from tests.test_run_config import CONTRACT_KEYS
    config_json = json.load(open(os.path.join(run_dir, "config.json")))
    assert sorted(config_json.keys()) == CONTRACT_KEYS
    assert config_json["runType"] == "optvec-fracture"
    assert config_json["modelID"] == "test/tiny"
    notes = config_json["notes"]
    assert notes["items"] == ["t-1", "t-2"] and notes["doses"] == [6.0]
    assert notes["clusterCounts"] == {"t-1@6.0": 2, "t-2@6.0": 1}
    assert notes["threshold"] == optvec_geometry.DEFAULT_CLUSTER_THRESHOLD
    assert notes["gradientReference"] is None


def test_the_fracture_table_reads_cluster_count_against_dose(tmp_path,
                                                             monkeypatch):
    """The fracture-α readout: one row per item, cluster count as a function
    of dose. No verdict — the count is the output."""
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    directory = tmp_path / "v"
    artifacts = []
    # Low dose: every restart lands in one basin.
    for seed, direction in enumerate([_unit(1.0), _unit(1.0, 0.1),
                                      _unit(1.0, -0.05)]):
        artifacts.append(_solution(directory, f"lo{seed}", item="t-1", dose=2.0,
                                   seed=seed, direction=direction))
    # High dose: the same three restarts scatter.
    for seed, direction in enumerate([_unit(1.0), _unit(0.0, 1.0),
                                      _unit(0.0, 0.0, 1.0)]):
        artifacts.append(_solution(directory, f"hi{seed}", item="t-1", dose=8.0,
                                   seed=seed, direction=direction))
    # A second item, measured at one dose only.
    artifacts.append(_solution(directory, "other0", item="t-2", dose=2.0,
                               seed=0, direction=_unit(1.0, 1.0)))
    artifacts.append(_solution(directory, "other1", item="t-2", dose=2.0,
                               seed=1, direction=_unit(1.0, 1.05)))

    report = optvec_geometry.fracture(optvec_geometry.OptVecFractureConfig(
        artifacts=artifacts))
    table = {row["item"]: row["doses"] for row in report["fractureTable"]}
    assert table["t-1"] == [
        {"dose": 2.0, "restarts": 3, "clusterCount": 1},
        {"dose": 8.0, "restarts": 3, "clusterCount": 3}]
    assert table["t-2"] == [{"dose": 2.0, "restarts": 2, "clusterCount": 1}]
    assert [row["item"] for row in report["fractureTable"]] == ["t-1", "t-2"]

    # A stricter threshold is a different reading of the same solutions, and
    # the report says which threshold produced its counts.
    strict = optvec_geometry.fracture(optvec_geometry.OptVecFractureConfig(
        artifacts=artifacts, threshold=0.999))
    strict_table = {row["item"]: row["doses"] for row in strict["fractureTable"]}
    assert strict_table["t-1"][0]["clusterCount"] == 3
    assert strict["threshold"] == 0.999


def test_the_gradient_reference_is_optional_and_never_guessed(tmp_path,
                                                              monkeypatch):
    monkeypatch.setenv("STEERLAB_ROOT", str(tmp_path))
    directory = tmp_path / "v"
    artifacts = [
        _solution(directory, "g0", item="t-1", dose=6.0, seed=0,
                  direction=_unit(1.0)),
        _solution(directory, "g1", item="t-1", dose=6.0, seed=1,
                  direction=_unit(1.0, 0.1)),
        _solution(directory, "g2", item="t-1", dose=6.0, seed=2,
                  direction=_unit(0.0, 1.0)),
        _solution(directory, "h0", item="t-2", dose=6.0, seed=0,
                  direction=_unit(0.0, 0.0, 1.0)),
        _solution(directory, "h1", item="t-2", dose=6.0, seed=1,
                  direction=_unit(0.0, 0.0, 1.0)),
    ]
    gradients = _write_gradients(tmp_path / "gradients.safetensors",
                                 {"t-1": _unit(1.0)})

    report = optvec_geometry.fracture(optvec_geometry.OptVecFractureConfig(
        artifacts=artifacts, gradients=gradients))
    groups = {g["item"]: g for g in report["groups"]}
    first, second = groups["t-1"]["clusters"]
    # The basin that contains the gradient direction reports cosine ≈ 1; the
    # orthogonal basin reports ≈ 0.
    assert first["meanGradientCosine"] == pytest.approx(0.9975, abs=1e-3)
    assert first["members"][0]["gradientCosine"] == pytest.approx(1.0,
                                                                  abs=1e-9)
    assert second["meanGradientCosine"] == pytest.approx(0.0, abs=1e-9)
    assert report["gradientReference"]["form"] == "perItem"
    assert report["gradientReference"]["sha256"]

    # The item with no row in the survey reports NO cosine, and says so.
    assert all("gradientCosine" not in member
               for cluster in groups["t-2"]["clusters"]
               for member in cluster["members"])
    assert all("meanGradientCosine" not in cluster
               for cluster in groups["t-2"]["clusters"])
    assert groups["t-2"]["gradientReferencePresent"] is False
    assert any("no row for item 't-2'" in advisory
               for advisory in report["advisories"])

    # The stacked form (one [N, d] tensor plus an item index) reads the same.
    stacked = _write_gradients(tmp_path / "stacked.safetensors",
                               {"t-1": _unit(1.0), "t-2": _unit(0.0, 0.0, 1.0)},
                               stacked=True)
    both = optvec_geometry.fracture(optvec_geometry.OptVecFractureConfig(
        artifacts=artifacts, gradients=stacked))
    assert both["gradientReference"]["form"] == "stacked"
    assert both["advisories"] == []
    stacked_groups = {g["item"]: g for g in both["groups"]}
    assert stacked_groups["t-2"]["clusters"][0]["meanGradientCosine"] == \
        pytest.approx(1.0, abs=1e-6)

    # Absent → the field is absent everywhere.
    plain = optvec_geometry.fracture(optvec_geometry.OptVecFractureConfig(
        artifacts=artifacts))
    assert plain["gradientReference"] is None
    assert plain["advisories"] == []

    with pytest.raises(OptVecGeometryError) as exc:
        optvec_geometry.fracture(optvec_geometry.OptVecFractureConfig(
            artifacts=artifacts, gradients=str(tmp_path / "missing.safetensors")))
    assert "names no file" in str(exc.value)
