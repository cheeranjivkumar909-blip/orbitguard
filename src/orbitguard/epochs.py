"""Time handling.

All absolute epochs inside OrbitGuard are expressed as floating-point seconds
since the J2000 reference epoch (2000-01-01T12:00:00 UTC). Leap seconds are
ignored, which is acceptable for conjunction screening over days-long windows.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def datetime_to_seconds(dt: datetime) -> float:
    """Convert a datetime (naive values are treated as UTC) to seconds since J2000."""
    return (_as_utc(dt) - J2000).total_seconds()


def seconds_to_datetime(seconds: float) -> datetime:
    """Convert seconds since J2000 to a timezone-aware UTC datetime."""
    return J2000 + timedelta(seconds=float(seconds))


def parse_iso(text: str) -> datetime:
    """Parse an ISO-8601 timestamp such as ``2026-12-01T00:00:00Z``."""
    return _as_utc(datetime.fromisoformat(text.strip().replace("Z", "+00:00")))


def format_iso(seconds: float) -> str:
    """Format seconds since J2000 as an ISO-8601 UTC string with millisecond precision."""
    return seconds_to_datetime(seconds).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
