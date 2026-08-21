"""Pure-CPU tests for the NumPy vector-math port (no GPU, no model)."""

import math

import numpy as np
import pytest

from steerlab_server.steering import vector_math as vm


def test_mean_difference():
    pos = [[1.0, 1.0], [3.0, 3.0]]
    neg = [[0.0, 0.0], [2.0, 0.0]]
    assert vm.mean_difference(pos, neg) == pytest.approx([1.0, 2.0])


def test_l2_norm_and_dot():
    assert vm.l2_norm([3.0, 4.0]) == pytest.approx(5.0)
    assert vm.dot([1.0, 2.0], [3.0, 4.0]) == pytest.approx(11.0)


def test_cosine_similarity():
    assert vm.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-6)
    assert vm.cosine_similarity([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0, abs=1e-6)


def test_rescaled():
    out = vm.rescaled([3.0, 4.0], to_norm=10.0)
    assert vm.l2_norm(out) == pytest.approx(10.0, abs=1e-5)


def test_norm_unit_scale_folds_out_vector_norm():
    # alpha * residual / ||v||  → 2 * 4 / 8 = 1
    assert vm.norm_unit_scale(2.0, 4.0, 8.0) == pytest.approx(1.0)
    with pytest.raises(vm.SteeringVectorError):
        vm.norm_unit_scale(1.0, 0.0, 1.0)


def test_first_principal_component_axis_aligned():
    rows = [[2.0, 0.0], [-2.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]
    pc = vm.first_principal_component(rows)
    assert abs(pc[0]) == pytest.approx(1.0, abs=1e-4)
    assert pc[1] == pytest.approx(0.0, abs=1e-4)


def test_first_principal_component_diagonal():
    rows = [[2.0, 1.0], [-2.0, -1.0], [1.0, 0.5], [-1.0, -0.5]]
    pc = vm.first_principal_component(rows)
    assert abs(vm.cosine_similarity(pc, [2.0, 1.0])) == pytest.approx(1.0, abs=1e-3)


def test_principal_components_explained_variance_decreasing():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(20, 5)).astype(np.float32)
    base[:, 0] *= 10.0  # dominant direction
    result = vm.principal_components_with_variance(base.tolist(), 3)
    assert len(result.components) == 3
    ev = result.explained_variance
    assert ev[0] >= ev[1] >= ev[2]
    assert result.total_explained_variance <= 1.0 + 1e-4


def test_projecting_out_removes_component():
    v = [1.0, 1.0, 0.0]
    e0 = [1.0, 0.0, 0.0]
    out = vm.projecting_out(v, [e0])
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[1] == pytest.approx(1.0)


def test_lat_direction_aligns_with_mean_difference_when_clean():
    # Clean, well-separated pairs: LAT's PC1 should align with the mean diff,
    # and be rescaled to its norm.
    pos = [[3.0, 0.0], [3.2, 0.1], [2.8, -0.1]]
    neg = [[0.0, 0.0], [0.1, 0.1], [-0.1, -0.1]]
    mean_diff = vm.mean_difference(pos, neg)
    lat = vm.direction(pos, neg, vm.ExtractionMethod.LAT)
    assert vm.cosine_similarity(lat, mean_diff) == pytest.approx(1.0, abs=1e-2)
    assert vm.l2_norm(lat) == pytest.approx(vm.l2_norm(mean_diff), rel=1e-3)


def test_scalar_probe_orientation_and_sign():
    direction = [1.0, 0.0]
    pos = [[2.0, 0.0], [3.0, 0.0]]
    neg = [[-2.0, 0.0], [-3.0, 0.0]]
    probe = vm.scalar_probe(direction, pos, neg)
    assert probe.orientation == 1.0
    assert probe.score([5.0, 0.0]) > 0
    assert probe.score([-5.0, 0.0]) < 0
    # Round-trips through the Swift-compatible dict shape.
    again = vm.ScalarProbe.from_dict(probe.to_dict())
    assert again.score([5.0, 0.0]) == pytest.approx(probe.score([5.0, 0.0]))


def test_mean_difference_dimension_mismatch():
    with pytest.raises(vm.SteeringVectorError):
        vm.mean_difference([[1.0, 2.0]], [[1.0, 2.0, 3.0]])
