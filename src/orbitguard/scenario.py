"""Reference scenario: an operational satellite vs. a crossing debris fragment.

The secondary is constructed *at* a designed time of closest approach so that
the encounter geometry (miss vector and crossing angle) is known exactly. This
makes the scenario useful both as a demo and as an end-to-end regression test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .constants import R_EARTH
from .elements import KeplerianElements, elements_to_state
from .epochs import datetime_to_seconds
from .frames import rtn_basis
from .propagation import ForceModel, NumericalPropagator


@dataclass
class Scenario:
    primary: NumericalPropagator
    secondary: NumericalPropagator
    primary_name: str
    secondary_name: str
    start_s: float
    end_s: float
    designed_tca_s: float
    designed_miss_km: float
    primary_cov_rtn: np.ndarray
    secondary_cov_rtn: np.ndarray
    hard_body_radius_km: float


def _rotate(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of ``vec`` about unit ``axis``."""
    return (
        vec * math.cos(angle)
        + np.cross(axis, vec) * math.sin(angle)
        + axis * np.dot(axis, vec) * (1.0 - math.cos(angle))
    )


def build_demo_scenario(
    radial_offset_km: float = 0.12,
    crosstrack_offset_km: float = 0.06,
    crossing_angle_deg: float = 120.0,
    hours: float = 48.0,
    tca_after_hours: float = 32.4,
) -> Scenario:
    epoch = datetime_to_seconds(datetime(2026, 12, 1, 0, 0, 0, tzinfo=timezone.utc))
    # Sun-synchronous 550 km Earth-observation satellite.
    primary_el = KeplerianElements(
        a=R_EARTH + 550.0, e=0.0012, i=math.radians(97.59), raan=math.radians(45.0), argp=math.radians(90.0), nu=0.0
    )
    r0, v0 = elements_to_state(primary_el)
    primary = NumericalPropagator(r0, v0, epoch, ForceModel(j2=True, drag=True, ballistic_coefficient=0.02))

    tca = epoch + tca_after_hours * 3600.0
    s1 = primary.state(tca)
    r1, v1 = s1[:3], s1[3:]
    r_hat = rtn_basis(r1, v1)[0]
    # Debris on a crossing orbit at the same altitude: rotate the velocity about the radial axis.
    v2 = _rotate(v1, r_hat, math.radians(crossing_angle_deg))
    dv_hat = (v2 - v1) / np.linalg.norm(v2 - v1)
    w_hat = np.cross(dv_hat, r_hat)
    w_hat /= np.linalg.norm(w_hat)
    offset = radial_offset_km * r_hat + crosstrack_offset_km * w_hat  # perpendicular to relative velocity
    r2 = r1 + offset
    secondary = NumericalPropagator(r2, v2, tca, ForceModel(j2=True, drag=True, ballistic_coefficient=0.08))

    return Scenario(
        primary=primary,
        secondary=secondary,
        primary_name="ORBITGUARD-DEMO-SAT (550 km SSO)",
        secondary_name="DEBRIS FRAGMENT 99731",
        start_s=epoch,
        end_s=epoch + hours * 3600.0,
        designed_tca_s=tca,
        designed_miss_km=float(np.linalg.norm(offset)),
        # 1-sigma position uncertainties (R, T, N): owner/operator GNSS ephemeris vs. radar-tracked debris.
        primary_cov_rtn=np.diag([0.04, 0.25, 0.04]) ** 2,
        secondary_cov_rtn=np.diag([0.10, 0.60, 0.10]) ** 2,
        hard_body_radius_km=0.015,
    )
