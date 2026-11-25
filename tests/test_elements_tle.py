import math
from datetime import datetime, timezone

import numpy as np
import pytest

from orbitguard.elements import (
    KeplerianElements,
    elements_to_state,
    mean_to_true,
    solve_kepler,
    state_to_elements,
    true_to_mean,
)
from orbitguard.tle import TLEError, format_tle, parse_tle, parse_tle_file, tle_checksum

# Classic, widely published ISS example element set (valid checksums).
ISS_L1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
ISS_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"


@pytest.mark.parametrize("e", [0.0, 0.001, 0.3, 0.7, 0.95])
def test_kepler_equation_residual(e):
    M = np.linspace(0, 2 * np.pi, 361)
    E = solve_kepler(M, e)
    assert np.allclose(E - e * np.sin(E), np.mod(M, 2 * np.pi), atol=1e-12)


def test_anomaly_roundtrip():
    for e in (0.0, 0.2, 0.8):
        nu = np.linspace(0.01, 6.27, 50)
        assert np.allclose(mean_to_true(true_to_mean(nu, e), e), nu, atol=1e-10)


def test_state_element_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        el = KeplerianElements(
            a=rng.uniform(6700, 42164),
            e=rng.uniform(0.001, 0.8),
            i=rng.uniform(0.05, 3.09),
            raan=rng.uniform(0, 2 * np.pi),
            argp=rng.uniform(0, 2 * np.pi),
            nu=rng.uniform(0, 2 * np.pi),
        )
        r, v = elements_to_state(el)
        back = state_to_elements(r, v)
        r2, v2 = elements_to_state(back)
        assert np.allclose(r, r2, atol=1e-6) and np.allclose(v, v2, atol=1e-9)
        assert back.a == pytest.approx(el.a, rel=1e-10)
        assert back.e == pytest.approx(el.e, abs=1e-10)


def test_circular_equatorial_state_roundtrip():
    el = KeplerianElements(a=7000.0, e=0.0, i=0.0, raan=0.0, argp=0.0, nu=1.234)
    r, v = elements_to_state(el)
    r2, v2 = elements_to_state(state_to_elements(r, v))
    assert np.allclose(r, r2, atol=1e-8) and np.allclose(v, v2, atol=1e-11)


def test_vis_viva():
    el = KeplerianElements(a=8000.0, e=0.2, i=0.5, raan=1.0, argp=2.0, nu=0.7)
    r, v = elements_to_state(el)
    mu = 398600.4418
    assert np.dot(v, v) == pytest.approx(mu * (2 / np.linalg.norm(r) - 1 / el.a), rel=1e-12)


def test_parse_iss_tle():
    tle = parse_tle(ISS_L1, ISS_L2, name="ISS (ZARYA)")
    assert tle.norad_id == 25544
    assert tle.inclination_deg == pytest.approx(51.6416)
    assert tle.eccentricity == pytest.approx(0.0006703)
    assert tle.mean_motion_rev_per_day == pytest.approx(15.72125391)
    assert tle.bstar == pytest.approx(-0.11606e-4)
    assert tle.epoch.year == 2008 and tle.epoch.timetuple().tm_yday == 264
    assert 6720 < tle.semi_major_axis_km < 6740


def test_checksum_validation():
    assert tle_checksum(ISS_L1) == 7
    with pytest.raises(TLEError):
        parse_tle(ISS_L1[:-1] + "0", ISS_L2)


def test_format_parse_roundtrip():
    epoch = datetime(2026, 12, 1, 6, 30, tzinfo=timezone.utc)
    l1, l2 = format_tle(90001, epoch, 97.6, 359.99995, 0.0011, 90.0, 12.3456, 15.12345678, bstar=-3.2e-5)
    assert len(l1) == 69 and len(l2) == 69
    tle = parse_tle(l1, l2)
    assert tle.raan_deg == pytest.approx(0.0, abs=1e-4)
    assert tle.mean_motion_rev_per_day == pytest.approx(15.12345678)
    assert tle.bstar == pytest.approx(-3.2e-5)
    assert abs((tle.epoch - epoch).total_seconds()) < 1e-3


def test_parse_three_line_file():
    text = f"ISS (ZARYA)\n{ISS_L1}\n{ISS_L2}\n{ISS_L1}\n{ISS_L2}\n"
    tles = parse_tle_file(text)
    assert [t.name for t in tles] == ["ISS (ZARYA)", "NORAD 25544"]
    assert math.isclose(tles[0].to_elements().i, math.radians(51.6416))
