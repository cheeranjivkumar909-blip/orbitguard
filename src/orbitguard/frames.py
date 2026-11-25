"""Reference frames: RTN (radial / transverse / normal) and the encounter (B-) plane."""

from __future__ import annotations

import numpy as np


def rtn_basis(r, v) -> np.ndarray:
    """Rotation matrix ECI -> RTN. Rows are the R, T and N unit vectors in ECI."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    r_hat = r / np.linalg.norm(r)
    h = np.cross(r, v)
    n_hat = h / np.linalg.norm(h)
    t_hat = np.cross(n_hat, r_hat)
    return np.vstack([r_hat, t_hat, n_hat])


def rtn_to_eci_covariance(cov_rtn, r, v) -> np.ndarray:
    """Rotate a 3x3 position covariance from the object's RTN frame into ECI."""
    q = rtn_basis(r, v)
    return q.T @ np.asarray(cov_rtn, dtype=float) @ q


def _any_perpendicular(u: np.ndarray) -> np.ndarray:
    trial = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    w = trial - (trial @ u) * u
    return w / np.linalg.norm(w)


def encounter_frame(rel_position, rel_velocity) -> np.ndarray:
    """Rotation ECI -> encounter frame.

    z is along the relative velocity, x along the miss vector (the component of
    the relative position perpendicular to z) and y completes the triad. At TCA
    the relative position lies entirely in the x-y (encounter) plane.
    """
    dr = np.asarray(rel_position, dtype=float)
    dv = np.asarray(rel_velocity, dtype=float)
    z = dv / np.linalg.norm(dv)
    x = dr - (dr @ z) * z
    x_norm = np.linalg.norm(x)
    x = x / x_norm if x_norm > 1e-12 else _any_perpendicular(z)
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def project_to_encounter_plane(rel_position, rel_velocity, cov_eci) -> tuple[np.ndarray, np.ndarray]:
    """Project the relative position and combined covariance onto the encounter plane.

    Returns ``(miss_2d, cov_2d)``.
    """
    b = encounter_frame(rel_position, rel_velocity)
    miss = b @ np.asarray(rel_position, dtype=float)
    cov = b @ np.asarray(cov_eci, dtype=float) @ b.T
    return miss[:2], cov[:2, :2]
