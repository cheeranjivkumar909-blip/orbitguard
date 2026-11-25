"""Two-Line Element (TLE) set parsing, validation and generation.

Note: TLE elements are SGP4 *mean* elements. OrbitGuard converts them to
Keplerian elements directly (no Kozai/Brouwer un-averaging), which introduces
kilometre-level absolute errors but preserves the secular behaviour needed for
broad-phase catalogue screening. Operational screening should refine flagged
events with owner/operator ephemerides.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .constants import MU_EARTH, SECONDS_PER_DAY
from .elements import KeplerianElements, mean_to_true
from .epochs import datetime_to_seconds


class TLEError(ValueError):
    """Raised when a TLE is malformed or fails validation."""


def tle_checksum(line: str) -> int:
    """Modulo-10 checksum over the first 68 columns (digits count, '-' counts as 1)."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


@dataclass(frozen=True)
class TLE:
    name: str
    norad_id: int
    epoch: datetime
    inclination_deg: float
    raan_deg: float
    eccentricity: float
    argp_deg: float
    mean_anomaly_deg: float
    mean_motion_rev_per_day: float
    bstar: float = 0.0
    line1: str = ""
    line2: str = ""

    @property
    def epoch_seconds(self) -> float:
        return datetime_to_seconds(self.epoch)

    @property
    def semi_major_axis_km(self) -> float:
        n = self.mean_motion_rev_per_day * 2.0 * math.pi / SECONDS_PER_DAY
        return (MU_EARTH / n**2) ** (1.0 / 3.0)

    @property
    def perigee_radius_km(self) -> float:
        return self.semi_major_axis_km * (1.0 - self.eccentricity)

    @property
    def apogee_radius_km(self) -> float:
        return self.semi_major_axis_km * (1.0 + self.eccentricity)

    def to_elements(self) -> KeplerianElements:
        e = self.eccentricity
        nu = mean_to_true(math.radians(self.mean_anomaly_deg), e)
        return KeplerianElements(
            a=self.semi_major_axis_km,
            e=e,
            i=math.radians(self.inclination_deg),
            raan=math.radians(self.raan_deg),
            argp=math.radians(self.argp_deg),
            nu=float(nu),
        )


def _parse_exp_field(field: str) -> float:
    """Parse the TLE 'assumed decimal point' exponent format, e.g. '-11606-4'."""
    field = field.strip()
    if not field:
        return 0.0
    sign = -1.0 if field[0] == "-" else 1.0
    body = field.lstrip("+-")
    mantissa, exponent = body[:-2], body[-2:]
    return sign * float("0." + mantissa) * 10.0 ** int(exponent)


def parse_tle(line1: str, line2: str, name: str = "", verify_checksum: bool = True) -> TLE:
    """Parse a two-line element set."""
    line1, line2 = line1.rstrip(), line2.rstrip()
    if len(line1) < 69 or len(line2) < 69:
        raise TLEError("TLE lines must be 69 characters long")
    if line1[0] != "1" or line2[0] != "2":
        raise TLEError("TLE lines must start with '1' and '2'")
    if verify_checksum:
        for ln in (line1, line2):
            if tle_checksum(ln) != int(ln[68]):
                raise TLEError(f"checksum mismatch on line {ln[0]}")
    norad1, norad2 = int(line1[2:7]), int(line2[2:7])
    if norad1 != norad2:
        raise TLEError("catalogue numbers on line 1 and line 2 differ")
    try:
        yy = int(line1[18:20])
        year = 2000 + yy if yy < 57 else 1900 + yy
        day_of_year = float(line1[20:32])
        epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_of_year - 1.0)
        return TLE(
            name=name.strip() or f"NORAD {norad1}",
            norad_id=norad1,
            epoch=epoch,
            inclination_deg=float(line2[8:16]),
            raan_deg=float(line2[17:25]),
            eccentricity=float("0." + line2[26:33].strip()),
            argp_deg=float(line2[34:42]),
            mean_anomaly_deg=float(line2[43:51]),
            mean_motion_rev_per_day=float(line2[52:63]),
            bstar=_parse_exp_field(line1[53:61]),
            line1=line1,
            line2=line2,
        )
    except ValueError as exc:
        raise TLEError(f"malformed TLE field: {exc}") from exc


def parse_tle_file(text: str, verify_checksum: bool = True) -> list[TLE]:
    """Parse a catalogue in 2-line or 3-line (named) format."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    tles: list[TLE] = []
    idx = 0
    while idx < len(lines):
        if lines[idx].startswith("1 ") and idx + 1 < len(lines) and lines[idx + 1].startswith("2 "):
            tles.append(parse_tle(lines[idx], lines[idx + 1], verify_checksum=verify_checksum))
            idx += 2
        elif idx + 2 < len(lines) and lines[idx + 1].startswith("1 ") and lines[idx + 2].startswith("2 "):
            name = lines[idx][2:] if lines[idx].startswith("0 ") else lines[idx]
            tles.append(parse_tle(lines[idx + 1], lines[idx + 2], name=name, verify_checksum=verify_checksum))
            idx += 3
        else:
            raise TLEError(f"unexpected line {idx + 1}: {lines[idx]!r}")
    return tles


def _format_exp_field(value: float) -> str:
    if value == 0.0:
        return " 00000-0"
    exponent = math.floor(math.log10(abs(value))) + 1
    digits = round(abs(value) / 10.0**exponent * 1e5)
    if digits >= 100000:
        digits //= 10
        exponent += 1
    sign = "-" if value < 0 else " "
    exp_sign = "-" if exponent < 0 else "+"
    return f"{sign}{digits:05d}{exp_sign}{abs(exponent)}"


def _angle(deg: float) -> float:
    return round(deg % 360.0, 4) % 360.0


def format_tle(
    norad_id: int,
    epoch: datetime,
    inclination_deg: float,
    raan_deg: float,
    eccentricity: float,
    argp_deg: float,
    mean_anomaly_deg: float,
    mean_motion_rev_per_day: float,
    bstar: float = 0.0,
    international_designator: str = "26001A",
    element_set: int = 999,
    revolution: int = 0,
) -> tuple[str, str]:
    """Produce a valid two-line element set (with checksums) from mean elements."""
    epoch = epoch.astimezone(timezone.utc)
    start = datetime(epoch.year, 1, 1, tzinfo=timezone.utc)
    day_of_year = (epoch - start).total_seconds() / SECONDS_PER_DAY + 1.0
    ecc_digits = round(eccentricity * 1e7)
    if not 0 <= ecc_digits <= 9999999:
        raise TLEError("eccentricity out of TLE range")
    body1 = (
        f"1 {norad_id:05d}U {international_designator:<8.8s} {epoch.year % 100:02d}{day_of_year:012.8f} "
        f" .00000000  00000-0 {_format_exp_field(bstar)} 0 {element_set % 10000:4d}"
    )
    body2 = (
        f"2 {norad_id:05d} {_angle(inclination_deg):8.4f} {_angle(raan_deg):8.4f} {ecc_digits:07d} "
        f"{_angle(argp_deg):8.4f} {_angle(mean_anomaly_deg):8.4f} {mean_motion_rev_per_day:11.8f}{revolution % 100000:5d}"
    )
    if len(body1) != 68 or len(body2) != 68:
        raise TLEError("internal formatting error: TLE body must be 68 characters")
    return body1 + str(tle_checksum(body1)), body2 + str(tle_checksum(body2))
