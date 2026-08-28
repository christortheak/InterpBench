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
    lat = vm.direction(pos, neg, vm.ExtractionMethod.PAIRED_DIFFERENCE_PCA)
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


def test_principal_components_never_exceed_the_data_rank():
    """2026-08-28 audit, F2. n centred rows span at most n−1 dimensions, so
    after n−1 deflations the residual is float32 round-off — whose own Gram
    trace is tiny but POSITIVE, which is exactly what the power iteration's
    relative degenerate-start floor accepts. The count branch used to hand
    that noise back as unit "components": 4 rows with count=6 returned 6, with
    explained variances [0.406, 0.310, 0.284, 7.9e-15, 4.2e-15, 2.1e-15], and
    `extract()` then projected the concept vector's component along the three
    arbitrary noise directions OUT of the science vector.

    Clamped rather than refused: `neutralPCCount` is a study-level knob applied
    to whatever neutral corpus each concept has (4 texts is legal), so
    over-asking is an honest declaration. Swift twin:
    `SteeringVectorMath.principalComponentsWithVariance`.
    """
    rows = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    with pytest.warns(UserWarning, match="span at most 3 dimension"):
        result = vm.principal_components_with_variance(rows, 6)
    assert len(result.components) == 3
    assert len(result.explained_variance) == 3
    # Every component returned carries real variance — no float-noise tail.
    assert all(share > 1e-6 for share in result.explained_variance)


def test_a_request_within_the_rank_is_untouched_and_silent():
    rows = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        assert len(vm.principal_components(rows, 3)) == 3
        assert len(vm.principal_components(rows, 2)) == 2


# --- F5: the power iteration stamps its own convergence health --------------

def _near_degenerate_cloud(ratio: float, n: int = 40, d: int = 64,
                           seed: int = 1) -> list[list[float]]:
    """A cloud whose top two sample eigenvalues are separated by a controlled
    gap — the audit's own repro geometry. ``ratio`` is λ2/λ1 as constructed;
    the SAMPLE ratio comes out a little higher."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((d, d)))
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    a -= a.mean()
    b -= b.mean()
    a /= np.sqrt((a ** 2).sum())
    b /= np.sqrt((b ** 2).sum())
    rows = np.outer(a, basis[:, 0]) + np.sqrt(ratio) * np.outer(b, basis[:, 1])
    return rows.astype(np.float32).tolist()


def test_a_healthy_cloud_stamps_a_converged_diagnostic_and_says_nothing():
    """The diagnostic is additive and quiet on ordinary data. Note the
    5%-eigengap case: it uses ALL 200 iterations without meeting the 1e-7 delta
    in float32 and is still accurate to six figures, which is exactly why the
    warning thresholds the RESIDUAL and not `converged`."""
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        component, diagnostic = vm.first_principal_component_with_diagnostic(
            [[3.0, 0.1], [2.9, -0.1], [-3.0, 0.05], [-2.8, -0.05]])
        assert diagnostic.converged
        assert diagnostic.relative_residual < vm.POWER_ITERATION_RESIDUAL_WARN_THRESHOLD
        assert not diagnostic.ill_conditioned
        assert 0 < diagnostic.iterations <= vm.POWER_ITERATION_MAX_ITERATIONS
        assert component == vm.first_principal_component(
            [[3.0, 0.1], [2.9, -0.1], [-3.0, 0.05], [-2.8, -0.05]])

        five_percent = _near_degenerate_cloud(0.95)
        _, healthy = vm.first_principal_component_with_diagnostic(five_percent)
    assert not healthy.converged            # the cap, on perfectly good data
    assert not healthy.ill_conditioned      # and no warning, correctly


def test_a_near_degenerate_spectrum_warns_and_stamps_instead_of_refusing():
    """2026-08-28 audit, F5. The audit reproduced a "PC1" at |cos| = 0.148
    against the TRUE second eigenvector, returned deterministically with no
    warning and no way to tell from the artifact. The direction is unchanged —
    it is deterministic and cross-engine mirrored, so this is a fact about the
    DATA, not malformed input — but it no longer arrives silent.

    Swift twin: `SteeringVectorMath.firstPrincipalComponentWithDiagnostic`.
    """
    rows = _near_degenerate_cloud(0.9945)
    quiet = vm.first_principal_component  # the component itself must not move
    with pytest.warns(UserWarning, match="nearly tied"):
        component, diagnostic = vm.first_principal_component_with_diagnostic(rows)
    assert diagnostic.ill_conditioned
    assert diagnostic.relative_residual > vm.POWER_ITERATION_RESIDUAL_WARN_THRESHOLD
    assert diagnostic.iterations == vm.POWER_ITERATION_MAX_ITERATIONS
    assert not diagnostic.converged
    with pytest.warns(UserWarning):
        assert quiet(rows) == component


def test_the_diagnostic_round_trips_and_the_warning_is_a_twin_literal():
    """The stamp shape and the message are cross-engine contracts (Swift
    `PowerIterationDiagnostic` / `powerIterationWarning`); the literal is
    written out here so a wording change is a deliberate two-engine edit."""
    diagnostic = vm.PowerIterationDiagnostic(
        relative_residual=0.006171, iterations=200, converged=False)
    assert diagnostic.to_dict() == {
        "converged": False,
        "illConditioned": True,
        "iterations": 200,
        "maxIterations": 200,
        "relativeResidual": 0.006171,
    }
    assert vm.PowerIterationDiagnostic.from_dict(diagnostic.to_dict()) == diagnostic
    assert vm.power_iteration_warning(diagnostic) == (
        "PC1 power iteration left a relative Rayleigh residual of 0.00617 after "
        "200 iterations (warn above 0.0001): the top two eigenvalues of this "
        "cloud are nearly tied, so PC1 is ill-determined and the returned "
        "direction may be an arbitrary vector in the near-degenerate plane. The "
        "result is deterministic and identical on both engines — it is the DATA "
        "that does not single out a first component. Read the stamped diagnostic "
        "before interpreting this direction.")


def test_deflation_reports_one_diagnostic_per_component():
    result = vm.principal_components_with_variance(
        [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-3.0, 0.0, 0.1], [0.0, -2.0, 0.0]], 2)
    assert len(result.diagnostics) == len(result.components) == 2
    assert all(not d.ill_conditioned for d in result.diagnostics)
