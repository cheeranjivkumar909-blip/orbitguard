import math

import numpy as np
import pytest

from orbitguard.frames import encounter_frame, project_to_encounter_plane, rtn_basis
from orbitguard.probability import chan_pc, foster_pc, isotropic_pc, max_pc_covariance_scaling


@pytest.mark.parametrize("sigma,hbr", [(0.1, 0.01), (0.05, 0.02), (0.01, 0.02), (1.0, 0.005)])
def test_isotropic_zero_miss_closed_form(sigma, hbr):
    exact = 1.0 - math.exp(-(hbr**2) / (2 * sigma**2))
    cov = np.eye(2) * sigma**2
    assert foster_pc([0, 0], cov, hbr) == pytest.approx(exact, rel=1e-6)
    assert chan_pc([0, 0], cov, hbr) == pytest.approx(exact, rel=1e-10)
    assert isotropic_pc(0.0, sigma, hbr) == pytest.approx(exact)


def test_chan_matches_foster_for_anisotropic_case():
    cov = np.array([[0.04, 0.012], [0.012, 0.25]])
    for miss in ([0.1, 0.05], [0.3, -0.4], [0.0, 0.8]):
        f = foster_pc(miss, cov, 0.015)
        c = chan_pc(miss, cov, 0.015)
        assert c == pytest.approx(f, rel=0.01)


def test_pc_decreases_with_miss_distance():
    cov = np.diag([0.1, 0.4]) ** 2
    values = [foster_pc([d, 0.0], cov, 0.02) for d in (0.0, 0.1, 0.3, 0.6, 1.2)]
    assert all(a > b for a, b in zip(values, values[1:]))
    assert foster_pc([5.0, 0.0], cov, 0.02) < 1e-30


def test_max_pc_bounds_nominal_and_matches_theory():
    # For isotropic covariance the maximum over sigma occurs at sigma = d / sqrt(2)
    # giving Pc_max ~ R^2 / (e d^2) when R << d.
    d, hbr = 1.0, 0.01
    cov = np.eye(2) * 0.05**2
    pc_max, k = max_pc_covariance_scaling([d, 0.0], cov, hbr)
    assert pc_max >= foster_pc([d, 0.0], cov, hbr)
    assert pc_max == pytest.approx(hbr**2 / (math.e * d**2), rel=0.01)
    assert 0.05 * k == pytest.approx(d / math.sqrt(2), rel=0.01)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        foster_pc([0, 0], np.eye(2), -1.0)
    with pytest.raises(ValueError):
        chan_pc([0, 0], np.array([[1.0, 2.0], [2.0, 1.0]]), 0.1)


def test_frames_are_orthonormal():
    r = np.array([7000.0, 100.0, -50.0])
    v = np.array([0.1, 7.5, 1.0])
    for m in (rtn_basis(r, v), encounter_frame([0.1, 0.2, 0.3], [5.0, -3.0, 9.0])):
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(m) == pytest.approx(1.0)


def test_encounter_projection_preserves_miss():
    dr = np.array([0.1, -0.2, 0.05])
    dv = np.cross(dr, [1.0, 2.0, 3.0])  # perpendicular, as at TCA
    miss, cov = project_to_encounter_plane(dr, dv, np.eye(3) * 0.01)
    assert np.linalg.norm(miss) == pytest.approx(np.linalg.norm(dr))
    assert np.allclose(cov, np.eye(2) * 0.01)
