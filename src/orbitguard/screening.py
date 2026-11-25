"""Conjunction screening: apsis pre-filter, coarse sampling and TCA refinement."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .epochs import format_iso, seconds_to_datetime
from .frames import rtn_basis
from .propagation import Propagator, SecularJ2Propagator
from .tle import TLE


@dataclass(frozen=True, eq=False)
class Conjunction:
    """A close approach at the time of closest approach (TCA)."""

    tca_s: float
    miss_distance_km: float
    relative_speed_km_s: float
    primary_state: np.ndarray
    secondary_state: np.ndarray
    primary_id: str = "PRIMARY"
    secondary_id: str = "SECONDARY"

    @property
    def tca(self):
        return seconds_to_datetime(self.tca_s)

    @property
    def relative_position_km(self) -> np.ndarray:
        return self.secondary_state[:3] - self.primary_state[:3]

    @property
    def relative_velocity_km_s(self) -> np.ndarray:
        return self.secondary_state[3:] - self.primary_state[3:]

    @property
    def miss_rtn_km(self) -> np.ndarray:
        """Secondary position relative to the primary in the primary's RTN frame."""
        return rtn_basis(self.primary_state[:3], self.primary_state[3:]) @ self.relative_position_km

    def to_dict(self) -> dict:
        return {
            "primary": self.primary_id,
            "secondary": self.secondary_id,
            "tca_utc": format_iso(self.tca_s),
            "miss_distance_m": round(self.miss_distance_km * 1000.0, 3),
            "miss_rtn_m": [round(x * 1000.0, 3) for x in self.miss_rtn_km],
            "relative_speed_km_s": round(self.relative_speed_km_s, 4),
        }


def apsis_filter(perigee_1: float, apogee_1: float, perigee_2: float, apogee_2: float, pad_km: float) -> bool:
    """Perigee/apogee filter (Hoots et al.). Returns True if the radial shells can overlap."""
    return max(perigee_1, perigee_2) - min(apogee_1, apogee_2) <= pad_km


def _range_rate_function(primary: Propagator, secondary: Propagator):
    def g(t: float) -> float:
        a = primary.state(t)
        b = secondary.state(t)
        return float(np.dot(b[:3] - a[:3], b[3:] - a[3:]))

    return g


def find_conjunctions(
    primary: Propagator,
    secondary: Propagator,
    start_s: float,
    end_s: float,
    step_s: float = 30.0,
    threshold_km: float = 5.0,
    primary_id: str = "PRIMARY",
    secondary_id: str = "SECONDARY",
) -> list[Conjunction]:
    """Find all local minima of the relative range below ``threshold_km``.

    The function g(t) = dr . dv (half the derivative of range squared) is
    sampled on a uniform grid; every - to + sign change brackets a range
    minimum which is then refined with Brent's method to microsecond level.
    """
    if end_s <= start_s:
        raise ValueError("end_s must be after start_s")
    n = int(np.ceil((end_s - start_s) / step_s)) + 1
    t = np.linspace(start_s, end_s, n)
    s1 = primary.states(t)
    s2 = secondary.states(t)
    g = np.einsum("ij,ij->i", s2[:, :3] - s1[:, :3], s2[:, 3:] - s1[:, 3:])
    g_fun = _range_rate_function(primary, secondary)
    events: list[Conjunction] = []
    for k in np.nonzero((g[:-1] < 0.0) & (g[1:] >= 0.0))[0]:
        tca = brentq(g_fun, t[k], t[k + 1], xtol=1e-6, rtol=4 * np.finfo(float).eps)
        a = primary.state(tca)
        b = secondary.state(tca)
        miss = float(np.linalg.norm(b[:3] - a[:3]))
        if miss <= threshold_km:
            events.append(
                Conjunction(
                    tca_s=float(tca),
                    miss_distance_km=miss,
                    relative_speed_km_s=float(np.linalg.norm(b[3:] - a[3:])),
                    primary_state=a,
                    secondary_state=b,
                    primary_id=primary_id,
                    secondary_id=secondary_id,
                )
            )
    return events


@dataclass
class ScreeningResult:
    conjunctions: list[Conjunction] = field(default_factory=list)
    objects_in_catalog: int = 0
    objects_after_apsis_filter: int = 0


def screen_catalog(
    primary: TLE,
    catalog: list[TLE],
    start_s: float,
    end_s: float,
    threshold_km: float = 10.0,
    pad_km: float = 10.0,
    step_s: float = 30.0,
) -> ScreeningResult:
    """Screen a primary TLE against a catalogue (analytical J2 propagation)."""
    result = ScreeningResult()
    primary_prop = SecularJ2Propagator.from_tle(primary)
    for obj in catalog:
        if obj.norad_id == primary.norad_id:
            continue
        result.objects_in_catalog += 1
        if not apsis_filter(
            primary.perigee_radius_km, primary.apogee_radius_km, obj.perigee_radius_km, obj.apogee_radius_km, pad_km
        ):
            continue
        result.objects_after_apsis_filter += 1
        result.conjunctions.extend(
            find_conjunctions(
                primary_prop,
                SecularJ2Propagator.from_tle(obj),
                start_s,
                end_s,
                step_s=step_s,
                threshold_km=threshold_km,
                primary_id=primary.name,
                secondary_id=obj.name,
            )
        )
    result.conjunctions.sort(key=lambda c: c.miss_distance_km)
    return result
