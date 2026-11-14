from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.config import settings


def now_iso() -> str:
    """Current time in UTC, as stored in the database."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tz() -> ZoneInfo:
    return ZoneInfo(settings.display_timezone)


def format_ts(iso: str) -> str:
    """Stored UTC timestamp -> friendly local time, e.g. 'Sat 14 Nov 2026, 15:30 IST'."""
    return datetime.fromisoformat(iso).astimezone(_tz()).strftime("%a %d %b %Y, %H:%M %Z")


def now_local() -> str:
    return datetime.now(_tz()).strftime("%A, %d %B %Y, %H:%M %Z")