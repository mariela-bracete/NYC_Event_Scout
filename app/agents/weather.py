"""Agent 3 — NWS weekend weather forecast.

Fetches a short-range forecast for NYC from the National Weather Service API
(api.weather.gov) — free, no API key required, but NWS's usage policy asks
consumers to identify themselves with a real User-Agent header (see
NWS_USER_AGENT below).

Two-step NWS flow: ``/points/{lat},{lon}`` resolves the forecast office + grid
cell for a point, and its response includes a ``forecast`` URL you then fetch
for the actual period-by-period forecast (~12-hour periods, several days out).

Any failure (network, non-200, malformed JSON, missing fields) degrades to
``{"available": False, "periods": []}`` rather than raising — the curator/
ranker treats missing weather as "no rationale available," never a hard
failure, matching Agents 1/2's never-500 convention.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Central Park — a single, central point used as a citywide proxy forecast.
# NWS forecasts are hyperlocal (per grid cell); one point can't represent all
# five boroughs precisely, but it stands in for a "citywide" forecast the same
# way the rest of the app treats NYC as one search area. Override via env if a
# different anchor point is preferred.
DEFAULT_LAT = os.environ.get("NWS_LAT", "40.7829")
DEFAULT_LON = os.environ.get("NWS_LON", "-73.9654")

NWS_POINTS_URL = "https://api.weather.gov/points/{lat},{lon}"
# NWS asks API consumers to identify themselves; no key required, but requests
# without a real User-Agent are sometimes throttled or rejected. Set
# NWS_CONTACT_EMAIL in .env so this isn't anonymous in production.
NWS_USER_AGENT = os.environ.get(
    "NWS_USER_AGENT",
    f"(NYC-Event-Scout, student project, contact: {os.environ.get('NWS_CONTACT_EMAIL', 'not-set')})",
)
HTTP_TIMEOUT_SECONDS = 10.0

_DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def _fetch_json(url: str) -> dict:
    """Thin wrapper around an HTTP GET -> JSON, isolated so tests can patch it
    without needing network access or a real httpx round trip."""
    import httpx

    response = httpx.get(
        url,
        headers={"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _empty_forecast(reason: str) -> Dict[str, Any]:
    return {"available": False, "periods": [], "reason": reason}


def _period_date(start_time: Optional[str]) -> Optional[date]:
    if not start_time:
        return None
    match = _DATE_PREFIX_RE.match(start_time)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def _upcoming_weekend_dates(today: Optional[date] = None) -> List[date]:
    """The current or next Fri/Sat/Sun.

    If today is already Friday, Saturday, or Sunday, returns *this* weekend
    (walking back to its Friday) rather than jumping ahead to next week's —
    otherwise a Saturday request would skip the weekend that's happening
    right now.
    """
    today = today or date.today()
    weekday = today.weekday()  # Monday=0 ... Sunday=6
    if weekday >= 4:  # Fri(4), Sat(5), or Sun(6) -> already in this weekend
        friday = today - timedelta(days=weekday - 4)
    else:
        friday = today + timedelta(days=4 - weekday)
    return [friday, friday + timedelta(days=1), friday + timedelta(days=2)]


def _shape_period(p: dict, start: Optional[date]) -> dict:
    return {
        "name": p.get("name", ""),
        "date": start.isoformat() if start else None,
        "is_daytime": bool(p.get("isDaytime")),
        "temperature": p.get("temperature"),
        "temperature_unit": p.get("temperatureUnit", "F"),
        "short_forecast": p.get("shortForecast", ""),
        "detailed_forecast": p.get("detailedForecast", ""),
    }


def get_weekend_forecast(
    lat: str = DEFAULT_LAT, lon: str = DEFAULT_LON, today: Optional[date] = None
) -> Dict[str, Any]:
    """Real NWS forecast filtered to the upcoming Fri/Sat/Sun, degrading gracefully
    to an "unavailable" shape on any failure.

    Returns:
        {
          "available": bool,
          "periods": [{"name": "Saturday", "date": "2026-07-25", "is_daytime": True,
                        "temperature": 82, "temperature_unit": "F",
                        "short_forecast": "Sunny", "detailed_forecast": "..."}],
          "reason": str,  # only present when available is False
        }
    """
    try:
        points = _fetch_json(NWS_POINTS_URL.format(lat=lat, lon=lon))
        forecast_url = points.get("properties", {}).get("forecast")
        if not forecast_url:
            return _empty_forecast("NWS points response had no forecast URL")

        forecast = _fetch_json(forecast_url)
        raw_periods = forecast.get("properties", {}).get("periods", [])
        if not raw_periods:
            return _empty_forecast("NWS forecast response had no periods")

        weekend_dates = set(_upcoming_weekend_dates(today))
        periods = []
        for p in raw_periods:
            start = _period_date(p.get("startTime"))
            if start is not None and start not in weekend_dates:
                continue
            periods.append(_shape_period(p, start))

        if not periods:
            # The forecast window (NWS gives ~7 days) doesn't happen to cover the
            # weekend right now (rare) — fall back to the first few periods so
            # the ranker still has *something* to reason with, rather than nothing.
            periods = [_shape_period(p, _period_date(p.get("startTime"))) for p in raw_periods[:4]]

        return {"available": True, "periods": periods}

    except Exception:
        logger.exception("NWS forecast fetch failed")
        return _empty_forecast("NWS request failed")
