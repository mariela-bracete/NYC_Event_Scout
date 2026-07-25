"""Offline unit tests for Agent 3's NWS weather module (app/agents/weather.py).

The only network seam, `_fetch_json`, is patched throughout — these tests
never hit the real api.weather.gov.
"""

from datetime import date
from unittest.mock import patch

import app.agents.weather as weather


def _points_response(forecast_url="https://api.weather.gov/gridpoints/OKX/33,37/forecast"):
    return {"properties": {"forecast": forecast_url}}


def _period(name, start_time, temp=75, short="Sunny", is_daytime=True):
    return {
        "name": name,
        "startTime": start_time,
        "isDaytime": is_daytime,
        "temperature": temp,
        "temperatureUnit": "F",
        "shortForecast": short,
        "detailedForecast": f"{short} with a high near {temp}.",
    }


def test_returns_only_weekend_periods():
    # Saturday, 2026-07-25 in the fixed "today" below is itself a Saturday, so
    # the weekend window is Fri 7/24 - Sun 7/26.
    today = date(2026, 7, 25)
    forecast_response = {
        "properties": {
            "periods": [
                _period("Thursday", "2026-07-23T06:00:00-04:00"),
                _period("Friday", "2026-07-24T06:00:00-04:00"),
                _period("Saturday", "2026-07-25T06:00:00-04:00", temp=88, short="Clear"),
                _period("Sunday", "2026-07-26T06:00:00-04:00"),
                _period("Monday", "2026-07-27T06:00:00-04:00"),
            ]
        }
    }

    with patch.object(
        weather, "_fetch_json", side_effect=[_points_response(), forecast_response]
    ):
        result = weather.get_weekend_forecast(today=today)

    assert result["available"] is True
    names = [p["name"] for p in result["periods"]]
    assert names == ["Friday", "Saturday", "Sunday"]
    saturday = result["periods"][1]
    assert saturday["date"] == "2026-07-25"
    assert saturday["temperature"] == 88
    assert saturday["short_forecast"] == "Clear"


def test_falls_back_when_points_lookup_fails():
    with patch.object(weather, "_fetch_json", side_effect=RuntimeError("network down")):
        result = weather.get_weekend_forecast()

    assert result == {"available": False, "periods": [], "reason": "NWS request failed"}


def test_falls_back_when_forecast_url_missing():
    with patch.object(weather, "_fetch_json", return_value={"properties": {}}):
        result = weather.get_weekend_forecast()

    assert result["available"] is False
    assert result["periods"] == []


def test_falls_back_when_no_periods_in_forecast():
    with patch.object(
        weather,
        "_fetch_json",
        side_effect=[_points_response(), {"properties": {"periods": []}}],
    ):
        result = weather.get_weekend_forecast()

    assert result["available"] is False


def test_uses_first_periods_when_window_misses_weekend():
    """If none of the returned periods happen to fall in the computed weekend
    window, fall back to the first few periods rather than returning nothing."""
    today = date(2026, 7, 25)
    forecast_response = {
        "properties": {
            "periods": [
                _period("Monday", "2026-08-10T06:00:00-04:00"),
                _period("Tuesday", "2026-08-11T06:00:00-04:00"),
            ]
        }
    }
    with patch.object(
        weather, "_fetch_json", side_effect=[_points_response(), forecast_response]
    ):
        result = weather.get_weekend_forecast(today=today)

    assert result["available"] is True
    assert len(result["periods"]) == 2


def test_upcoming_weekend_dates_from_a_weekday():
    # Wednesday 2026-07-22 -> next Fri/Sat/Sun is 7/24, 7/25, 7/26.
    dates = weather._upcoming_weekend_dates(today=date(2026, 7, 22))
    assert dates == [date(2026, 7, 24), date(2026, 7, 25), date(2026, 7, 26)]


def test_upcoming_weekend_dates_when_today_is_saturday():
    # Already Saturday -> weekend starts "today's" Friday (yesterday) per the
    # (4 - weekday) % 7 formula, i.e. still returns the current Fri/Sat/Sun set.
    dates = weather._upcoming_weekend_dates(today=date(2026, 7, 25))
    assert dates == [date(2026, 7, 24), date(2026, 7, 25), date(2026, 7, 26)]
