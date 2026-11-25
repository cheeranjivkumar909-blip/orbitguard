"""Classical orbital elements, Kepler's equation and state-vector conversions.

Conventions: distances in km, velocities in km/s, angles in radians,
inertial frame is Earth-centred inertial (ECI, treated as GCRF/TEME-like).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import MU_EARTH

TWO_PI = 2.0 * np.pi
_EPS = 1e-11


@dataclass(frozen=True)
class KeplerianElements:
    """Osculating classical orbital elements of an elliptic orbit."""

    a: float  # semi-major axis [km]
    e: float  # eccentricity [-]
    i: float  # inclination [rad]
    raan: float  # right ascension of the ascending node [rad]
    argp: float  # argument of perigee [rad]
    nu: float  # true anomaly [rad]

    def __post_init__(self) -> None:
        if self.a <= 0.0:
            raise ValueError("semi-major axis must be positive (elliptic orbits only)")
        if not 0.0 <= self.e < 1.0:
            raise ValueError("eccentricity must satisfy 0 <= e < 1")

    @property
    def semi_latus_rectum(self) -> float:
        return self.a * (1.0 - self.e**2)

    @property
    def perigee_radius(self) -> float:
        return self.a * (1.0 - self.e)

    @property
    def apogee_radius(self) -> float:
        return self.a * (1.0 + self.e)

    @property
    def mean_anomaly(self) -> float:
        return float(true_to_mean(self.nu, self.e))

    def mean_motion(self, mu: float = MU_EARTH) -> float:
        """Keplerian mean motion [rad/s]."""
        return float(np.sqrt(mu / self.a**3))

    def period(self, mu: float = MU_EARTH) -> float:
        """Keplerian orbital period [s]."""
        return TWO_PI / self.mean_motion(mu)


def solve_kepler(mean_anomaly, e: float, tol: float = 1e-13, max_iter: int = 50):
    """Solve Kepler's equation ``M = E - e sin E`` for the eccentric anomaly.

    Vectorised Newton-Raphson iteration. Accepts scalars or arrays.
    """
    M = np.mod(np.asarray(mean_anomaly, dtype=float), TWO_PI)
    E = np.where(e < 0.8, M, np.pi) + 0.0 * M
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < tol):
            break
    else:  # pragma: no cover - only reached for pathological input
        raise RuntimeError("Kepler solver did not converge")
    return E if E.ndim else float(E)


def mean_to_true(mean_anomaly, e: float):
    """Convert mean anomaly to true anomaly."""
    E = np.asarray(solve_kepler(mean_anomaly, e))
    nu = 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0), np.sqrt(1.0 - e) * np.cos(E / 2.0))
    nu = np.mod(nu, TWO_PI)
    return nu if nu.ndim else float(nu)


def true_to_mean(nu, e: float):
    """Convert true anomaly to mean anomaly."""
    nu = np.asarray(nu, dtype=float)
    E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(nu / 2.0), np.sqrt(1.0 + e) * np.cos(nu / 2.0))
    M = np.mod(E - e * np.sin(E), TWO_PI)
    return M if M.ndim else float(M)


def kepler_to_cartesian(a, e, i, raan, argp, nu, mu: float = MU_EARTH):
    """Vectorised conversion of classical elements to ECI position and velocity.

    Every argument may be a scalar or an array (broadcast together).
    Returns ``(r, v)`` with a trailing dimension of size 3.
    """
    a, e, i, raan, argp, nu = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (a, e, i, raan, argp, nu)))
    p = a * (1.0 - e**2)
    r_mag = p / (1.0 + e * np.cos(nu))
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)
    # Perifocal unit vectors expressed in the inertial frame.
    P = np.stack([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si], axis=-1)
    Q = np.stack([-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si], axis=-1)
    cn, sn = np.cos(nu)[..., None], np.sin(nu)[..., None]
    r = r_mag[..., None] * (cn * P + sn * Q)
    v = np.sqrt(mu / p)[..., None] * (-sn * P + (e[..., None] + cn) * Q)
    return r, v


def elements_to_state(el: KeplerianElements, mu: float = MU_EARTH) -> tuple[np.ndarray, np.ndarray]:
    """Position [km] and velocity [km/s] for a set of classical elements."""
    return kepler_to_cartesian(el.a, el.e, el.i, el.raan, el.argp, el.nu, mu)


def _signed_angle(u: np.ndarray, w: np.ndarray, axis: np.ndarray) -> float:
    """Angle from u to w measured positively about ``axis`` (all 3-vectors)."""
    return float(np.arctan2(np.dot(axis, np.cross(u, w)), np.dot(u, w)))


def state_to_elements(r, v, mu: float = MU_EARTH) -> KeplerianElements:
    """Convert an ECI state vector to osculating classical elements.

    Circular and equatorial orbits are handled with the usual conventions:
    argument of perigee is set to zero for circular orbits (true anomaly then
    holds the argument of latitude) and RAAN is set to zero for equatorial ones.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_mag = np.linalg.norm(r)
    v2 = float(v @ v)
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)
    h_hat = h / h_mag
    n = np.cross([0.0, 0.0, 1.0], h)
    n_mag = np.linalg.norm(n)
    e_vec = ((v2 - mu / r_mag) * r - float(r @ v) * v) / mu
    e = float(np.linalg.norm(e_vec))
    energy = 0.5 * v2 - mu / r_mag
    if energy >= 0.0:
        raise ValueError("state is not on an elliptic orbit")
    a = -mu / (2.0 * energy)
    inc = float(np.arccos(np.clip(h[2] / h_mag, -1.0, 1.0)))

    x_axis = np.array([1.0, 0.0, 0.0])
    node = n / n_mag if n_mag > _EPS else x_axis
    raan = float(np.arctan2(node[1], node[0])) if n_mag > _EPS else 0.0
    if e > _EPS:
        e_hat = e_vec / e
        argp = _signed_angle(node, e_hat, h_hat)
        nu = _signed_angle(e_hat, r / r_mag, h_hat)
    else:
        e = 0.0
        argp = 0.0
        nu = _signed_angle(node, r / r_mag, h_hat)
    return KeplerianElements(
        a=float(a),
        e=e,
        i=inc,
        raan=raan % TWO_PI,
        argp=argp % TWO_PI,
        nu=nu % TWO_PI,
    )
