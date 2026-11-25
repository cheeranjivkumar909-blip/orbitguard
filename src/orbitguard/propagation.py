"""Orbit propagators.

* :class:`SecularJ2Propagator` - analytical Kepler + J2 secular drift. Very fast,
  vectorised, used for broad-phase catalogue screening.
* :class:`NumericalPropagator` - Cowell integration (DOP853) of two-body + J2 +
  atmospheric drag with dense output. Used for high-fidelity refinement and
  collision-avoidance manoeuvre design.

Both expose ``states(t)`` returning an ``(N, 6)`` array of ECI position [km]
and velocity [km/s] at absolute epochs ``t`` in seconds since J2000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.integrate import solve_ivp

from .atmosphere import exponential_density
from .constants import J2, MU_EARTH, OMEGA_EARTH, R_EARTH
from .elements import KeplerianElements, kepler_to_cartesian, mean_to_true, state_to_elements
from .tle import TLE


class Propagator(Protocol):
    def states(self, t) -> np.ndarray: ...

    def state(self, t: float) -> np.ndarray: ...


@dataclass(frozen=True)
class ForceModel:
    """Force model configuration for :class:`NumericalPropagator`."""

    j2: bool = True
    drag: bool = False
    ballistic_coefficient: float = 0.01  # C_d * A / m [m^2 / kg]
    mu: float = MU_EARTH


def j2_secular_rates(el: KeplerianElements, mu: float = MU_EARTH) -> tuple[float, float, float]:
    """Secular rates (dRAAN/dt, dargp/dt, dM/dt) [rad/s] caused by J2."""
    n = math.sqrt(mu / el.a**3)
    p = el.a * (1.0 - el.e**2)
    k = 1.5 * J2 * (R_EARTH / p) ** 2 * n
    ci = math.cos(el.i)
    raan_dot = -k * ci
    argp_dot = 0.5 * k * (5.0 * ci**2 - 1.0)
    mean_dot = n + 0.5 * k * math.sqrt(1.0 - el.e**2) * (3.0 * ci**2 - 1.0)
    return raan_dot, argp_dot, mean_dot


class SecularJ2Propagator:
    """Analytical propagator: Keplerian motion with J2 secular node/perigee/anomaly drift."""

    def __init__(self, elements: KeplerianElements, epoch_s: float, include_j2: bool = True, mu: float = MU_EARTH):
        self.elements = elements
        self.epoch_s = float(epoch_s)
        self.mu = mu
        if include_j2:
            self.raan_dot, self.argp_dot, self.mean_dot = j2_secular_rates(elements, mu)
        else:
            self.raan_dot, self.argp_dot, self.mean_dot = 0.0, 0.0, elements.mean_motion(mu)
        self._m0 = elements.mean_anomaly

    @classmethod
    def from_tle(cls, tle: TLE, include_j2: bool = True) -> "SecularJ2Propagator":
        return cls(tle.to_elements(), tle.epoch_seconds, include_j2=include_j2)

    @classmethod
    def from_state(cls, r, v, epoch_s: float, include_j2: bool = True) -> "SecularJ2Propagator":
        return cls(state_to_elements(r, v), epoch_s, include_j2=include_j2)

    def states(self, t) -> np.ndarray:
        dt = np.atleast_1d(np.asarray(t, dtype=float)) - self.epoch_s
        el = self.elements
        nu = mean_to_true(self._m0 + self.mean_dot * dt, el.e)
        r, v = kepler_to_cartesian(
            el.a, el.e, el.i, el.raan + self.raan_dot * dt, el.argp + self.argp_dot * dt, nu, self.mu
        )
        return np.hstack([r, v])

    def state(self, t: float) -> np.ndarray:
        return self.states([t])[0]


class NumericalPropagator:
    """Cowell propagator (two-body + J2 + exponential-atmosphere drag), DOP853 with dense output.

    The integration is performed lazily, forwards and/or backwards from the
    reference epoch, and extended automatically when later epochs are requested.
    """

    def __init__(
        self,
        r0,
        v0,
        epoch_s: float,
        force_model: ForceModel | None = None,
        rtol: float = 1e-11,
        atol: float = 1e-9,
    ):
        self._y0 = np.concatenate([np.asarray(r0, dtype=float), np.asarray(v0, dtype=float)])
        self.epoch_s = float(epoch_s)
        self.force_model = force_model or ForceModel()
        self.rtol = rtol
        self.atol = atol
        self._forward = None
        self._forward_end = 0.0
        self._backward = None
        self._backward_end = 0.0

    def acceleration(self, y) -> tuple[float, float, float]:
        fm = self.force_model
        x, yy, z, vx, vy, vz = y
        r2 = x * x + yy * yy + z * z
        r = math.sqrt(r2)
        r3 = r2 * r
        ax = -fm.mu * x / r3
        ay = -fm.mu * yy / r3
        az = -fm.mu * z / r3
        if fm.j2:
            f = -1.5 * J2 * fm.mu * R_EARTH**2 / (r2 * r3)
            z2 = z * z / r2
            ax += f * x * (1.0 - 5.0 * z2)
            ay += f * yy * (1.0 - 5.0 * z2)
            az += f * z * (3.0 - 5.0 * z2)
        if fm.drag:
            rho = exponential_density(r - R_EARTH)
            # Velocity relative to the co-rotating atmosphere: v - omega x r.
            vrx = vx + OMEGA_EARTH * yy
            vry = vy - OMEGA_EARTH * x
            vrz = vz
            vrel = math.sqrt(vrx * vrx + vry * vry + vrz * vrz)
            # rho [kg/m^3] * B [m^2/kg] * v^2 [km^2/s^2] -> factor 1000 to get km/s^2.
            k = -0.5 * fm.ballistic_coefficient * rho * vrel * 1000.0
            ax += k * vrx
            ay += k * vry
            az += k * vrz
        return ax, ay, az

    def _rhs(self, _tau, y):
        ax, ay, az = self.acceleration(y)
        return [y[3], y[4], y[5], ax, ay, az]

    def _solve(self, tau_end: float):
        sol = solve_ivp(
            self._rhs,
            (0.0, tau_end),
            self._y0,
            method="DOP853",
            rtol=self.rtol,
            atol=self.atol,
            dense_output=True,
        )
        if not sol.success:  # pragma: no cover - defensive
            raise RuntimeError(f"integration failed: {sol.message}")
        return sol.sol

    def states(self, t) -> np.ndarray:
        tau = np.atleast_1d(np.asarray(t, dtype=float)) - self.epoch_s
        out = np.empty((tau.size, 6))
        fwd = tau >= 0.0
        if fwd.any():
            end = max(float(tau[fwd].max()), 1e-3)
            if self._forward is None or end > self._forward_end:
                self._forward = self._solve(end)
                self._forward_end = end
            out[fwd] = self._forward(tau[fwd]).T
        bwd = ~fwd
        if bwd.any():
            end = float(tau[bwd].min())
            if self._backward is None or end < self._backward_end:
                self._backward = self._solve(end)
                self._backward_end = end
            out[bwd] = self._backward(tau[bwd]).T
        return out

    def state(self, t: float) -> np.ndarray:
        return self.states([t])[0]
