"""Probability of collision (Pc) for short-term (high relative velocity) encounters.

Under the short-term encounter assumptions (rectilinear relative motion,
static Gaussian position uncertainty during the encounter), the collision
probability reduces to the integral of a bivariate Gaussian over the
hard-body circle in the encounter plane:

    Pc = 1 / (2 pi sqrt|C|) * integral_{|x| <= R} exp(-1/2 (x - m)^T C^-1 (x - m)) dx

where m is the 2-D miss vector, C the combined 2x2 covariance and R the
combined hard-body radius.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammainc, gammaln


def _validate(miss, cov, hbr):
    m = np.asarray(miss, dtype=float).reshape(2)
    c = np.asarray(cov, dtype=float).reshape(2, 2)
    if hbr <= 0.0:
        raise ValueError("hard-body radius must be positive")
    c = 0.5 * (c + c.T)
    if np.linalg.eigvalsh(c).min() <= 0.0:
        raise ValueError("covariance must be positive definite")
    return m, c


def foster_pc(miss, cov, hbr: float, n_radial: int = 64, n_angular: int = 180) -> float:
    """Foster's 2-D Pc by direct numerical quadrature over the hard-body disc.

    Gauss-Legendre quadrature in radius and the (spectrally accurate)
    trapezoidal rule in angle. Resolution is increased automatically when the
    covariance is small compared to the hard-body radius.
    """
    m, c = _validate(miss, cov, hbr)
    sigma_min = math.sqrt(np.linalg.eigvalsh(c).min())
    ratio = hbr / sigma_min
    n_r = int(min(2000, max(n_radial, 8 * ratio)))
    n_t = int(min(4000, max(n_angular, 16 * math.pi * ratio)))
    c_inv = np.linalg.inv(c)
    det = np.linalg.det(c)
    nodes, weights = np.polynomial.legendre.leggauss(n_r)
    radius = 0.5 * hbr * (nodes + 1.0)
    w_r = 0.5 * hbr * weights
    theta = np.linspace(0.0, 2.0 * math.pi, n_t, endpoint=False)
    rr, tt = np.meshgrid(radius, theta, indexing="ij")
    px = rr * np.cos(tt) - m[0]
    py = rr * np.sin(tt) - m[1]
    q = c_inv[0, 0] * px * px + 2.0 * c_inv[0, 1] * px * py + c_inv[1, 1] * py * py
    density = np.exp(-0.5 * q) / (2.0 * math.pi * math.sqrt(det))
    pc = float(np.sum(density * rr * w_r[:, None]) * (2.0 * math.pi / n_t))
    return min(max(pc, 0.0), 1.0)


def chan_pc(miss, cov, hbr: float, max_terms: int | None = None) -> float:
    """Chan's analytical series approximation (equivalent-area formulation).

    Exact for isotropic covariance; very accurate whenever the hard-body radius
    is small compared with the covariance, the usual operational regime.

        Pc = sum_m  Poisson(m; v/2) * P(m + 1, u/2)

    with u = R^2 / (sx sy), v = (x/sx)^2 + (y/sy)^2 in principal axes and
    P the regularised lower incomplete gamma function.
    """
    m, c = _validate(miss, cov, hbr)
    eigval, eigvec = np.linalg.eigh(c)
    sx, sy = math.sqrt(eigval[0]), math.sqrt(eigval[1])
    mp = eigvec.T @ m
    u = hbr**2 / (sx * sy)
    v = (mp[0] / sx) ** 2 + (mp[1] / sy) ** 2
    half_v = 0.5 * v
    if max_terms is None:
        max_terms = int(half_v + 12.0 * math.sqrt(half_v + 1.0) + 30)
    orders = np.arange(max_terms)
    if half_v > 0.0:
        log_poisson = orders * math.log(half_v) - half_v - gammaln(orders + 1)
    else:
        log_poisson = np.where(orders == 0, 0.0, -np.inf)
    terms = np.exp(log_poisson) * gammainc(orders + 1, 0.5 * u)
    return float(min(max(terms.sum(), 0.0), 1.0))


def isotropic_pc(miss_distance: float, sigma: float, hbr: float) -> float:
    """Closed form for the zero-miss isotropic case (reference / testing)."""
    if miss_distance != 0.0:
        return chan_pc([miss_distance, 0.0], np.eye(2) * sigma**2, hbr)
    return 1.0 - math.exp(-(hbr**2) / (2.0 * sigma**2))


def max_pc_covariance_scaling(miss, cov, hbr: float) -> tuple[float, float]:
    """Maximum Pc over a uniform scaling k^2 * C of the covariance (Alfano-style).

    Useful as a worst-case bound when covariance realism is in doubt.
    Returns ``(pc_max, k)``.
    """
    m, c = _validate(miss, cov, hbr)

    def neg_pc(log_k: float) -> float:
        return -foster_pc(m, c * math.exp(2.0 * log_k), hbr)

    res = minimize_scalar(neg_pc, bounds=(-8.0, 8.0), method="bounded", options={"xatol": 1e-4})
    k = math.exp(res.x)
    pc_at_k = -res.fun
    pc_nominal = foster_pc(m, c, hbr)
    if pc_nominal > pc_at_k:
        return pc_nominal, 1.0
    return pc_at_k, k
