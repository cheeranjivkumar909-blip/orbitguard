"""Generate the synthetic TLE catalogue used in the README screening example.

The catalogue contains one primary (NORAD 90001), three objects engineered to
pass close to it at known epochs, and a background population spread across
LEO. Everything is synthetic: no real satellite is represented.

    python examples/build_catalog.py  ->  examples/synthetic_catalog.tle
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from orbitguard.constants import MU_EARTH, R_EARTH, SECONDS_PER_DAY
from orbitguard.elements import KeplerianElements, state_to_elements
from orbitguard.epochs import datetime_to_seconds
from orbitguard.frames import rtn_basis
from orbitguard.propagation import SecularJ2Propagator, j2_secular_rates
from orbitguard.tle import format_tle

EPOCH = datetime(2026, 12, 1, 0, 0, 0, tzinfo=timezone.utc)
OUT = Path(__file__).with_name("synthetic_catalog.tle")


def to_tle(norad: int, el: KeplerianElements, intl: str) -> tuple[str, str]:
    n_rev_day = math.sqrt(MU_EARTH / el.a**3) * SECONDS_PER_DAY / (2.0 * math.pi)
    return format_tle(
        norad,
        EPOCH,
        math.degrees(el.i),
        math.degrees(el.raan),
        el.e,
        math.degrees(el.argp),
        math.degrees(el.mean_anomaly),
        n_rev_day,
        bstar=1.2e-4,
        international_designator=intl,
    )


def back_propagate(el: KeplerianElements, dt: float) -> KeplerianElements:
    """Invert the secular J2 model: elements at t* -> elements at t* - dt."""
    raan_dot, argp_dot, mean_dot = j2_secular_rates(el)
    from orbitguard.elements import mean_to_true

    m0 = el.mean_anomaly - mean_dot * dt
    return KeplerianElements(
        a=el.a, e=el.e, i=el.i,
        raan=(el.raan - raan_dot * dt) % (2 * math.pi),
        argp=(el.argp - argp_dot * dt) % (2 * math.pi),
        nu=float(mean_to_true(m0, el.e)),
    )


def rotate(vec, axis, angle):
    return vec * math.cos(angle) + np.cross(axis, vec) * math.sin(angle) + axis * np.dot(axis, vec) * (1 - math.cos(angle))


def main() -> None:
    rng = np.random.default_rng(20261201)
    epoch_s = datetime_to_seconds(EPOCH)
    primary = KeplerianElements(R_EARTH + 550.0, 0.0011, math.radians(97.6), math.radians(45.0), math.radians(90.0), 0.0)
    lines = ["ORBITGUARD-DEMO-SAT", *to_tle(90001, primary, "26001A")]
    prop = SecularJ2Propagator(primary, epoch_s)

    # (hours after epoch, radial offset km, cross offset km, crossing angle deg)
    designed = [(9.25, 0.35, 0.20, 95.0), (21.6, 1.8, -0.9, 150.0), (40.1, 0.05, 0.6, 60.0)]
    for k, (hours, d_r, d_c, angle) in enumerate(designed):
        t_star = epoch_s + hours * 3600.0
        s = prop.state(t_star)
        r_hat = rtn_basis(s[:3], s[3:])[0]
        v2 = rotate(s[3:], r_hat, math.radians(angle))
        dv_hat = (v2 - s[3:]) / np.linalg.norm(v2 - s[3:])
        w_hat = np.cross(dv_hat, r_hat)
        el_star = state_to_elements(s[:3] + d_r * r_hat + d_c * w_hat / np.linalg.norm(w_hat), v2)
        el0 = back_propagate(el_star, hours * 3600.0)
        lines += [f"DEBRIS {k + 1:02d} (DESIGNED)", *to_tle(90010 + k, el0, f"99{k:03d}A")]

    for k in range(240):
        alt = rng.uniform(450.0, 1200.0)
        el = KeplerianElements(
            a=R_EARTH + alt,
            e=float(rng.uniform(0.0, 0.01)),
            i=math.radians(rng.choice([53.0, 70.0, 86.4, 97.5, 98.7, rng.uniform(20, 110)])),
            raan=rng.uniform(0, 2 * math.pi),
            argp=rng.uniform(0, 2 * math.pi),
            nu=rng.uniform(0, 2 * math.pi),
        )
        lines += [f"BACKGROUND OBJECT {k + 1:03d}", *to_tle(91000 + k, el, f"98{k:03d}B")]

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines) // 3} objects to {OUT}")


if __name__ == "__main__":
    main()
