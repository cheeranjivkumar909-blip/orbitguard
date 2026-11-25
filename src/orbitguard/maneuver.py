"""Collision-avoidance manoeuvre (CAM) planning.

Searches impulsive along-track burns at several lead times before TCA and
returns the smallest delta-v that brings Pc below a target. Each candidate is
evaluated by re-propagating the primary with the high-fidelity numerical
propagator and re-computing TCA, miss distance and Pc against the secondary.

Along-track burns are the operational workhorse: a burn of dv changes the
semi-major axis by ~2 dv / n, which produces a radial offset of ~4 dv / n
half an orbit later and a growing along-track drift (~3 dv t).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .epochs import format_iso
from .frames import rtn_basis
from .propagation import ForceModel, NumericalPropagator, Propagator
from .risk import RiskAssessment, assess_conjunction
from .screening import Conjunction, find_conjunctions


@dataclass(frozen=True, eq=False)
class ManeuverPlan:
    burn_time_s: float
    lead_orbits: float
    delta_v_rtn_m_s: np.ndarray
    new_conjunction: Conjunction | None
    new_assessment: RiskAssessment | None
    original_pc: float
    target_pc: float
    evaluations: int

    @property
    def delta_v_m_s(self) -> float:
        return float(np.linalg.norm(self.delta_v_rtn_m_s))

    @property
    def new_pc(self) -> float:
        return self.new_assessment.pc if self.new_assessment else 0.0

    @property
    def new_miss_distance_km(self) -> float:
        return self.new_conjunction.miss_distance_km if self.new_conjunction else math.inf

    def to_dict(self) -> dict:
        return {
            "burn_epoch_utc": format_iso(self.burn_time_s),
            "lead_orbits": self.lead_orbits,
            "delta_v_rtn_m_s": [round(float(x), 6) for x in self.delta_v_rtn_m_s],
            "delta_v_m_s": round(self.delta_v_m_s, 6),
            "post_maneuver_miss_m": round(self.new_miss_distance_km * 1000.0, 3),
            "post_maneuver_pc": self.new_pc,
            "original_pc": self.original_pc,
            "target_pc": self.target_pc,
            "candidate_evaluations": self.evaluations,
        }


def orbital_period(state, mu: float = 398600.4418) -> float:
    r = np.linalg.norm(state[:3])
    v2 = float(np.dot(state[3:], state[3:]))
    a = -mu / (2.0 * (0.5 * v2 - mu / r))
    return 2.0 * math.pi * math.sqrt(a**3 / mu)


def plan_collision_avoidance(
    primary: Propagator,
    secondary: Propagator,
    conjunction: Conjunction,
    primary_cov_rtn,
    secondary_cov_rtn,
    hard_body_radius_km: float,
    target_pc: float = 1e-6,
    lead_orbits: tuple[float, ...] = (0.5, 1.0, 1.5),
    dv_min_m_s: float = 5e-4,
    dv_max_m_s: float = 2.0,
    growth: float = 1.6,
    bisection_steps: int = 8,
    force_model: ForceModel | None = None,
    window_s: float = 300.0,
) -> ManeuverPlan | None:
    """Find the minimum along-track delta-v that reduces Pc below ``target_pc``.

    Returns ``None`` if no manoeuvre is required or none within ``dv_max_m_s``
    satisfies the target.
    """
    original = assess_conjunction(
        conjunction, primary_cov_rtn, secondary_cov_rtn, hard_body_radius_km, compute_max_pc=False
    )
    if original.pc <= target_pc:
        return None
    force_model = force_model or ForceModel(j2=True, drag=False)
    tca = conjunction.tca_s
    period = orbital_period(conjunction.primary_state)
    evaluations = 0

    def evaluate(t_burn: float, dv_t: float):
        nonlocal evaluations
        evaluations += 1
        s = primary.state(t_burn)
        t_hat = rtn_basis(s[:3], s[3:])[1]
        prop = NumericalPropagator(s[:3], s[3:] + t_hat * dv_t / 1000.0, t_burn, force_model)
        events = find_conjunctions(prop, secondary, tca - window_s, tca + window_s, step_s=5.0, threshold_km=np.inf)
        if not events:
            return 0.0, None, None
        event = min(events, key=lambda c: c.miss_distance_km)
        ra = assess_conjunction(event, primary_cov_rtn, secondary_cov_rtn, hard_body_radius_km, compute_max_pc=False)
        return ra.pc, event, ra

    best: ManeuverPlan | None = None
    for lead in lead_orbits:
        t_burn = tca - lead * period
        for sign in (1.0, -1.0):
            ceiling = min(dv_max_m_s, best.delta_v_m_s if best else dv_max_m_s)
            lo, dv, hit = 0.0, dv_min_m_s, None
            while dv <= ceiling:
                result = evaluate(t_burn, sign * dv)
                if result[0] <= target_pc:
                    hit = (dv, result)
                    break
                lo, dv = dv, dv * growth
            if hit is None:
                continue
            hi = hit[0]
            for _ in range(bisection_steps):
                mid = 0.5 * (lo + hi)
                result = evaluate(t_burn, sign * mid)
                if result[0] <= target_pc:
                    hi, hit = mid, (mid, result)
                else:
                    lo = mid
            dv_mag, (_, event, ra) = hit
            if best is None or dv_mag < best.delta_v_m_s:
                best = ManeuverPlan(
                    burn_time_s=t_burn,
                    lead_orbits=lead,
                    delta_v_rtn_m_s=np.array([0.0, sign * dv_mag, 0.0]),
                    new_conjunction=event,
                    new_assessment=ra,
                    original_pc=original.pc,
                    target_pc=target_pc,
                    evaluations=0,
                )
    if best is None:
        return None
    return ManeuverPlan(**{**best.__dict__, "evaluations": evaluations})
