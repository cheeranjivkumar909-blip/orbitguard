import math

import numpy as np
import pytest

from orbitguard.constants import MU_EARTH, R_EARTH
from orbitguard.elements import KeplerianElements, elements_to_state, state_to_elements
from orbitguard.propagation import ForceModel, NumericalPropagator, SecularJ2Propagator, j2_secular_rates

LEO = KeplerianElements(a=R_EARTH + 550.0, e=0.001, i=math.radians(97.6), raan=0.3, argp=1.0, nu=0.0)


def _energy(s):
    return 0.5 * np.sum(s[:, 3:] ** 2, axis=1) - MU_EARTH / np.linalg.norm(s[:, :3], axis=1)


def test_two_body_invariants():
    r, v = elements_to_state(LEO)
    prop = NumericalPropagator(r, v, 0.0, ForceModel(j2=False))
    s = prop.states(np.linspace(0, 86400, 500))
    e = _energy(s)
    h = np.linalg.norm(np.cross(s[:, :3], s[:, 3:]), axis=1)
    assert np.ptp(e) / abs(e[0]) < 1e-10
    assert np.ptp(h) / h[0] < 1e-10


def test_two_body_matches_kepler():
    r, v = elements_to_state(LEO)
    numerical = NumericalPropagator(r, v, 0.0, ForceModel(j2=False))
    analytical = SecularJ2Propagator(LEO, 0.0, include_j2=False)
    t = np.linspace(-43200, 86400, 97)
    assert np.max(np.linalg.norm(numerical.states(t)[:, :3] - analytical.states(t)[:, :3], axis=1)) < 1e-3


def test_j2_nodal_regression_matches_theory():
    r, v = elements_to_state(LEO)
    prop = NumericalPropagator(r, v, 0.0, ForceModel(j2=True))
    period = LEO.period()
    n_orbits = 15
    s = prop.state(n_orbits * period)
    raan_numeric = state_to_elements(s[:3], s[3:]).raan - LEO.raan
    raan_dot, _, _ = j2_secular_rates(LEO)
    expected = raan_dot * n_orbits * period
    # Sun-synchronous orbit: about +0.9856 deg/day.
    assert math.degrees(raan_dot) * 86400 == pytest.approx(0.9856, abs=0.02)
    assert raan_numeric == pytest.approx(expected, rel=0.03)


def test_drag_decays_orbit():
    low = KeplerianElements(a=R_EARTH + 300.0, e=0.0005, i=0.9, raan=0.0, argp=0.0, nu=0.0)
    r, v = elements_to_state(low)
    prop = NumericalPropagator(r, v, 0.0, ForceModel(j2=False, drag=True, ballistic_coefficient=0.02))
    s = prop.state(86400.0)
    assert state_to_elements(s[:3], s[3:]).a < low.a - 0.1


def test_backward_and_forward_consistency():
    r, v = elements_to_state(LEO)
    prop = NumericalPropagator(r, v, 1000.0, ForceModel(j2=True, drag=True))
    s_back = prop.state(1000.0 - 7200.0)
    reverse = NumericalPropagator(s_back[:3], s_back[3:], 1000.0 - 7200.0, ForceModel(j2=True, drag=True))
    assert np.linalg.norm(reverse.state(1000.0)[:3] - r) < 1e-5


def test_secular_propagator_vectorised_shape():
    prop = SecularJ2Propagator(LEO, 0.0)
    assert prop.states(np.arange(10.0)).shape == (10, 6)
    assert prop.state(5.0).shape == (6,)
