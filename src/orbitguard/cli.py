"""Command-line interface: ``orbitguard demo | screen | pc``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .epochs import datetime_to_seconds, format_iso, parse_iso
from .maneuver import plan_collision_avoidance
from .probability import chan_pc, foster_pc, max_pc_covariance_scaling
from .risk import assess_conjunction
from .scenario import build_demo_scenario
from .screening import screen_catalog
from .tle import parse_tle_file

RULE = "-" * 64


def _fmt_pc(pc: float) -> str:
    return "< 1e-30" if pc < 1e-30 else f"{pc:.3e}"


def run_demo(args: argparse.Namespace) -> int:
    from .screening import find_conjunctions

    t0 = time.perf_counter()
    sc = build_demo_scenario()
    print("OrbitGuard - Conjunction Assessment & Collision Avoidance")
    print("=" * 64)
    print(f"Primary    : {sc.primary_name}  [Cowell: 2-body + J2 + drag]")
    print(f"Secondary  : {sc.secondary_name}")
    print(f"Window     : {format_iso(sc.start_s)} -> {format_iso(sc.end_s)}")

    events = find_conjunctions(
        sc.primary, sc.secondary, sc.start_s, sc.end_s, step_s=20.0, threshold_km=args.threshold_km,
        primary_id=sc.primary_name, secondary_id=sc.secondary_name,
    )
    print(f"Close approaches below {args.threshold_km:g} km: {len(events)}")
    if not events:
        return 0
    event = min(events, key=lambda c: c.miss_distance_km)
    risk = assess_conjunction(event, sc.primary_cov_rtn, sc.secondary_cov_rtn, sc.hard_body_radius_km)
    rtn = event.miss_rtn_km * 1000.0

    print(RULE)
    print("CONJUNCTION")
    print(f"  TCA (UTC)             : {format_iso(event.tca_s)}")
    print(f"  Miss distance         : {event.miss_distance_km * 1000:.1f} m")
    print(f"  Miss R / T / N        : {rtn[0]:+.1f} / {rtn[1]:+.1f} / {rtn[2]:+.1f} m")
    print(f"  Relative speed        : {event.relative_speed_km_s:.2f} km/s")
    print(f"  Hard-body radius      : {sc.hard_body_radius_km * 1000:.1f} m")
    print(f"  Pc (Foster 2-D)       : {_fmt_pc(risk.pc)}")
    print(f"  Pc (Chan series)      : {_fmt_pc(risk.pc_chan)}")
    print(f"  Max Pc (cov. scaling) : {_fmt_pc(risk.pc_max)}  at k = {risk.covariance_scale_at_max:.2f}")
    print(f"  Risk level            : {risk.risk_level}")

    report = {"tool": "orbitguard", "version": __version__, "conjunction": event.to_dict(), "risk": risk.to_dict()}
    if risk.risk_level != "GREEN":
        plan = plan_collision_avoidance(
            sc.primary, sc.secondary, event, sc.primary_cov_rtn, sc.secondary_cov_rtn, sc.hard_body_radius_km,
            target_pc=args.target_pc,
        )
        print(RULE)
        print("COLLISION-AVOIDANCE MANOEUVRE")
        if plan is None:
            print("  No along-track manoeuvre within limits meets the target Pc.")
        else:
            dv = plan.delta_v_rtn_m_s * 100.0
            print(f"  Burn epoch (UTC)      : {format_iso(plan.burn_time_s)}  ({plan.lead_orbits:g} orbits before TCA)")
            print(f"  Delta-v R / T / N     : {dv[0]:+.2f} / {dv[1]:+.2f} / {dv[2]:+.2f} cm/s")
            print(f"  |Delta-v|             : {plan.delta_v_m_s * 100:.2f} cm/s")
            print(f"  Post-burn miss        : {plan.new_miss_distance_km * 1000:.1f} m")
            print(f"  Post-burn Pc          : {_fmt_pc(plan.new_pc)}  (target {args.target_pc:.0e})")
            print(f"  Candidates evaluated  : {plan.evaluations}")
            report["maneuver"] = plan.to_dict()
    print(RULE)
    print(f"Completed in {time.perf_counter() - t0:.1f} s")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON report written to {args.json}")
    return 0


def run_screen(args: argparse.Namespace) -> int:
    catalog = parse_tle_file(Path(args.catalog).read_text(encoding="utf-8"))
    primary = next((t for t in catalog if t.norad_id == args.primary), None)
    if primary is None:
        print(f"error: NORAD {args.primary} not found in {args.catalog}", file=sys.stderr)
        return 2
    start = datetime_to_seconds(parse_iso(args.start)) if args.start else primary.epoch_seconds
    end = start + args.hours * 3600.0
    t0 = time.perf_counter()
    res = screen_catalog(primary, catalog, start, end, args.threshold_km, args.pad_km, args.step)
    print(f"Primary {primary.name} (NORAD {primary.norad_id}) | window {format_iso(start)} + {args.hours:g} h")
    print(
        f"Catalogue objects: {res.objects_in_catalog} | after apsis filter: {res.objects_after_apsis_filter} | "
        f"conjunctions < {args.threshold_km:g} km: {len(res.conjunctions)} | {time.perf_counter() - t0:.1f} s"
    )
    print(f"{'TCA (UTC)':<26}{'Secondary':<28}{'Miss [km]':>10}{'Vrel [km/s]':>13}")
    for c in res.conjunctions:
        print(f"{format_iso(c.tca_s):<26}{c.secondary_id[:27]:<28}{c.miss_distance_km:>10.3f}{c.relative_speed_km_s:>13.2f}")
    if args.json:
        Path(args.json).write_text(json.dumps([c.to_dict() for c in res.conjunctions], indent=2), encoding="utf-8")
    return 0


def run_pc(args: argparse.Namespace) -> int:
    sx, sy, rho = args.sigma_x, args.sigma_y, args.rho
    cov = np.array([[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]])
    miss = [args.miss_x, args.miss_y]
    pc_max, k = max_pc_covariance_scaling(miss, cov, args.hbr)
    print(f"Pc Foster : {_fmt_pc(foster_pc(miss, cov, args.hbr))}")
    print(f"Pc Chan   : {_fmt_pc(chan_pc(miss, cov, args.hbr))}")
    print(f"Max Pc    : {_fmt_pc(pc_max)} (covariance scale k = {k:.3f})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbitguard", description="Space debris conjunction assessment toolkit")
    parser.add_argument("--version", action="version", version=f"orbitguard {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the end-to-end reference scenario")
    demo.add_argument("--threshold-km", type=float, default=5.0)
    demo.add_argument("--target-pc", type=float, default=1e-6)
    demo.add_argument("--json", help="write a JSON report to this path")
    demo.set_defaults(func=run_demo)

    screen = sub.add_parser("screen", help="screen a primary against a TLE catalogue")
    screen.add_argument("catalog", help="TLE file (2-line or 3-line format)")
    screen.add_argument("--primary", type=int, required=True, help="NORAD id of the primary")
    screen.add_argument("--start", help="ISO-8601 start epoch (default: primary TLE epoch)")
    screen.add_argument("--hours", type=float, default=48.0)
    screen.add_argument("--threshold-km", type=float, default=10.0)
    screen.add_argument("--pad-km", type=float, default=10.0, help="apsis filter pad")
    screen.add_argument("--step", type=float, default=30.0, help="coarse sampling step [s]")
    screen.add_argument("--json", help="write conjunctions as JSON")
    screen.set_defaults(func=run_screen)

    pc = sub.add_parser("pc", help="compute Pc from encounter-plane quantities (km)")
    pc.add_argument("--miss-x", type=float, required=True)
    pc.add_argument("--miss-y", type=float, default=0.0)
    pc.add_argument("--sigma-x", type=float, required=True)
    pc.add_argument("--sigma-y", type=float, required=True)
    pc.add_argument("--rho", type=float, default=0.0, help="correlation coefficient")
    pc.add_argument("--hbr", type=float, required=True, help="hard-body radius")
    pc.set_defaults(func=run_pc)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
