from pathlib import Path

import numpy as np
import pytest

from orbitguard.cli import main
from orbitguard.maneuver import plan_collision_avoidance
from orbitguard.risk import assess_conjunction, classify_risk
from orbitguard.scenario import build_demo_scenario
from orbitguard.screening import apsis_filter, find_conjunctions, screen_catalog
from orbitguard.tle import parse_tle_file

CATALOG = Path(__file__).resolve().parents[1] / "examples" / "synthetic_catalog.tle"


@pytest.fixture(scope="module")
def scenario():
    return build_demo_scenario()


@pytest.fixture(scope="module")
def event(scenario):
    events = find_conjunctions(scenario.primary, scenario.secondary, scenario.start_s, scenario.end_s, 20.0, 5.0)
    assert len(events) >= 1
    return min(events, key=lambda c: c.miss_distance_km)


def test_screening_recovers_designed_geometry(scenario, event):
    assert event.tca_s == pytest.approx(scenario.designed_tca_s, abs=1e-3)
    assert event.miss_distance_km == pytest.approx(scenario.designed_miss_km, abs=1e-6)
    assert event.relative_speed_km_s > 10.0
    # At TCA the relative position is perpendicular to the relative velocity.
    cos = np.dot(event.relative_position_km, event.relative_velocity_km_s) / (
        np.linalg.norm(event.relative_position_km) * event.relative_speed_km_s
    )
    assert abs(cos) < 1e-6


def test_risk_assessment(scenario, event):
    risk = assess_conjunction(event, scenario.primary_cov_rtn, scenario.secondary_cov_rtn, scenario.hard_body_radius_km)
    assert risk.risk_level == "RED"
    assert risk.pc_chan == pytest.approx(risk.pc, rel=0.01)
    assert risk.pc_max >= risk.pc


def test_maneuver_meets_target(scenario, event):
    plan = plan_collision_avoidance(
        scenario.primary, scenario.secondary, event,
        scenario.primary_cov_rtn, scenario.secondary_cov_rtn, scenario.hard_body_radius_km, target_pc=1e-6,
    )
    assert plan is not None
    assert plan.new_pc <= 1e-6
    assert 0.0 < plan.delta_v_m_s < 0.5
    assert plan.new_miss_distance_km > event.miss_distance_km
    assert plan.burn_time_s < event.tca_s


def test_classification_thresholds():
    assert classify_risk(2e-4) == "RED"
    assert classify_risk(5e-6) == "YELLOW"
    assert classify_risk(1e-9) == "GREEN"
    assert apsis_filter(6900, 6950, 6940, 7000, 5.0)
    assert not apsis_filter(6900, 6950, 7100, 7200, 10.0)


def test_catalog_screening_finds_designed_objects():
    tles = parse_tle_file(CATALOG.read_text())
    primary = next(t for t in tles if t.norad_id == 90001)
    res = screen_catalog(primary, tles, primary.epoch_seconds, primary.epoch_seconds + 48 * 3600, threshold_km=5.0)
    assert res.objects_after_apsis_filter < res.objects_in_catalog
    closest = {}
    for c in res.conjunctions:
        closest[c.secondary_id] = min(closest.get(c.secondary_id, np.inf), c.miss_distance_km)
    assert closest["DEBRIS 01 (DESIGNED)"] == pytest.approx(0.403, abs=0.05)
    assert closest["DEBRIS 02 (DESIGNED)"] == pytest.approx(2.012, abs=0.05)
    assert closest["DEBRIS 03 (DESIGNED)"] == pytest.approx(0.602, abs=0.05)


def test_cli_pc(capsys):
    assert main(["pc", "--miss-x", "0", "--sigma-x", "0.1", "--sigma-y", "0.1", "--hbr", "0.01"]) == 0
    assert "Pc Foster" in capsys.readouterr().out
