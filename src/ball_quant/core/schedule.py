from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo


DERIVATIVE_TITLE_SUFFIXES = (
    " - Halftime Result",
    " - Exact Score",
    " - More Markets",
    " - First Team to Score",
    " - Player Props",
    " - Second Half Result",
    " - Total Corners",
)


def parse_event_start(event: Dict[str, Any]) -> Optional[datetime]:
    for key in ("startTime", "endDate", "startDate"):
        value = event.get(key)
        if value:
            return parse_iso_utc(str(value))
    return None


def parse_iso_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def is_core_match_event(event: Dict[str, Any]) -> bool:
    title = str(event.get("title") or "")
    slug = str(event.get("slug") or "")
    if " vs. " not in title and " vs " not in title.lower():
        return False
    if any(title.endswith(suffix) for suffix in DERIVATIVE_TITLE_SUFFIXES):
        return False
    if slug and "-player-props" in slug:
        return False
    return True


def event_lifecycle_status(
    event: Dict[str, Any],
    start_utc: datetime,
    now_utc: Optional[datetime] = None,
    expire_after_hours: float = 3.0,
) -> str:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if event.get("closed") is True or event.get("ended") is True:
        return "expired"
    expires_at = start_utc + timedelta(hours=expire_after_hours)
    if now >= expires_at:
        return "expired"
    if now >= start_utc:
        return "live"
    return "upcoming"


def schedule_row(
    event: Dict[str, Any],
    timezone_name: str = "Asia/Shanghai",
    now_utc: Optional[datetime] = None,
    expire_after_hours: float = 3.0,
) -> Optional[Dict[str, Any]]:
    start_utc = parse_event_start(event)
    if start_utc is None:
        return None
    local_tz = ZoneInfo(timezone_name)
    local_start = start_utc.astimezone(local_tz)
    status = event_lifecycle_status(
        event,
        start_utc,
        now_utc=now_utc,
        expire_after_hours=expire_after_hours,
    )
    home, away = split_match_title(str(event.get("title") or ""))
    return {
        "event_id": str(event.get("id") or ""),
        "event_slug": event.get("slug"),
        "event_title": event.get("title"),
        "home": home,
        "away": away,
        "polymarket_date": event.get("eventDate"),
        "start_time_utc": iso_z(start_utc),
        "local_timezone": timezone_name,
        "local_date": local_start.date().isoformat(),
        "local_time": local_start.strftime("%H:%M"),
        "local_start": local_start.isoformat(),
        "status": status,
        "active": event.get("active"),
        "closed": event.get("closed"),
        "ended": event.get("ended"),
        "updated_at": event.get("updatedAt"),
    }


def split_match_title(title: str) -> tuple[str, str]:
    normalized = title
    for suffix in DERIVATIVE_TITLE_SUFFIXES:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    marker = " vs. "
    if marker not in normalized:
        marker = " vs "
    if marker not in normalized:
        return "", ""
    home, away = normalized.split(marker, 1)
    return home.strip(), away.strip()


def select_schedule_rows(
    events: Iterable[Dict[str, Any]],
    timezone_name: str = "Asia/Shanghai",
    date: Optional[str] = None,
    date_mode: str = "poly",
    include_expired: bool = False,
    lookahead_hours: Optional[float] = None,
    now_utc: Optional[datetime] = None,
    expire_after_hours: float = 3.0,
) -> List[Dict[str, Any]]:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    rows: List[Dict[str, Any]] = []
    for event in events:
        if not is_core_match_event(event):
            continue
        row = schedule_row(
            event,
            timezone_name=timezone_name,
            now_utc=now,
            expire_after_hours=expire_after_hours,
        )
        if row is None:
            continue
        if not include_expired and row["status"] == "expired":
            continue
        if date and date_mode == "poly" and row.get("polymarket_date") != date:
            continue
        if date and date_mode == "local" and row.get("local_date") != date:
            continue
        if lookahead_hours is not None:
            start = parse_iso_utc(str(row["start_time_utc"]))
            if start < now or start > now + timedelta(hours=lookahead_hours):
                continue
        rows.append(row)
    rows.sort(key=lambda item: item.get("start_time_utc") or "")
    return rows


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

