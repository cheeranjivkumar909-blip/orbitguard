"""Conjunction risk assessment: covariance combination, Pc and risk classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .frames import project_to_encounter_plane, rtn_to_eci_covariance
from .probability import chan_pc, foster_pc, max_pc_covariance_scaling
from .screening import Conjunction

# Thresholds in the spirit of common operator practice (e.g. NASA CARA uses
# Pc >= 1e-4 as the typical manoeuvre threshold). Both are configurable.
RED_THRESHOLD = 1e-4
YELLOW_THRESHOLD = 1e-6


def classify_risk(pc: float, red: float = RED_THRESHOLD, yellow: float = YELLOW_THRESHOLD) -> str:
    if pc >= red:
        return "RED"
    if pc >= yellow:
        return "YELLOW"
    return "GREEN"


@dataclass(frozen=True, eq=False)
class RiskAssessment:
    pc: float
    pc_chan: float
    pc_max: float
    covariance_scale_at_max: float
    miss_distance_km: float
    miss_bplane_km: np.ndarray
    covariance_bplane_km2: np.ndarray
    hard_body_radius_km: float
    risk_level: str

    def to_dict(self) -> dict:
        return {
            "pc_foster": self.pc,
            "pc_chan": self.pc_chan,
            "pc_max_covariance_scaling": self.pc_max,
            "covariance_scale_at_max": round(self.covariance_scale_at_max, 4),
            "miss_distance_m": round(self.miss_distance_km * 1000.0, 3),
            "bplane_sigma_m": [
                round(float(np.sqrt(self.covariance_bplane_km2[0, 0])) * 1000.0, 3),
                round(float(np.sqrt(self.covariance_bplane_km2[1, 1])) * 1000.0, 3),
            ],
            "hard_body_radius_m": round(self.hard_body_radius_km * 1000.0, 3),
            "risk_level": self.risk_level,
        }


def assess_conjunction(
    conjunction: Conjunction,
    primary_cov_rtn,
    secondary_cov_rtn,
    hard_body_radius_km: float,
    red: float = RED_THRESHOLD,
    yellow: float = YELLOW_THRESHOLD,
    compute_max_pc: bool = True,
) -> RiskAssessment:
    """Compute Pc for a conjunction given each object's 3x3 RTN position covariance [km^2]."""
    p, s = conjunction.primary_state, conjunction.secondary_state
    cov_eci = rtn_to_eci_covariance(primary_cov_rtn, p[:3], p[3:]) + rtn_to_eci_covariance(
        secondary_cov_rtn, s[:3], s[3:]
    )
    miss2d, cov2d = project_to_encounter_plane(
        conjunction.relative_position_km, conjunction.relative_velocity_km_s, cov_eci
    )
    pc = foster_pc(miss2d, cov2d, hard_body_radius_km)
    pc_chan = chan_pc(miss2d, cov2d, hard_body_radius_km)
    if compute_max_pc:
        pc_max, k = max_pc_covariance_scaling(miss2d, cov2d, hard_body_radius_km)
    else:
        pc_max, k = pc, 1.0
    return RiskAssessment(
        pc=pc,
        pc_chan=pc_chan,
        pc_max=pc_max,
        covariance_scale_at_max=k,
        miss_distance_km=conjunction.miss_distance_km,
        miss_bplane_km=miss2d,
        covariance_bplane_km2=cov2d,
        hard_body_radius_km=hard_body_radius_km,
        risk_level=classify_risk(pc, red, yellow),
    )
