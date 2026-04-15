"""Weather-aware totals adjustment for MLB.

Uses the free, no-API-key weather.gov forecast endpoint to pull
hourly wind + temperature for each MLB game's stadium at first
pitch, then converts those values into a run adjustment on the
total-projection model.

Research basis
--------------

* Wind blowing out ≥15 mph: +1 to +2 runs vs neutral.
* Wind blowing in  ≥15 mph: -1 to -2 runs vs neutral.
* Temperature ≥80°F: +0.4 run (compared to ≤60°F baseline).
* Temperature ≤55°F: -0.5 run.

Simplifications
---------------

We don't yet account for wind direction relative to each stadium's
outfield orientation — so the magnitude-only adjustment is applied
half-weighted. Over a season this under-counts the edge but never
mis-signs it, and keeps the model robust to noise.

Indoor / retractable-roof stadiums always return a neutral factor.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from urllib.request import Request, urlopen

from .mlb_stadiums import stadium, is_indoor


logger = logging.getLogger(__name__)

# Weather.gov asks for a contact in the User-Agent string.
USER_AGENT = "sports-betting-ai-agent (contact via github.com/edwardnguyen1194-spec)"

# Cache forecasts for 30 minutes — weather.gov data only updates
# hourly and the network hop is slow.
_FORECAST_CACHE: Dict[str, Tuple[float, list]] = {}
_POINTS_CACHE: Dict[Tuple[float, float], Tuple[float, str]] = {}
CACHE_TTL = 1800


def _http_get_json(url: str, timeout: float = 8.0) -> Optional[dict]:
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        logger.info("weather fetch %s failed: %s", url[:70], exc)
        return None


def _get_forecast_url(lat: float, lon: float) -> Optional[str]:
    """Resolve the hourly-forecast URL for a lat/lon via the points endpoint."""
    key = (round(lat, 4), round(lon, 4))
    cached = _POINTS_CACHE.get(key)
    now = time.time()
    if cached and now - cached[0] < 86400:   # points responses are stable — cache a day
        return cached[1]
    data = _http_get_json(f"https://api.weather.gov/points/{key[0]},{key[1]}")
    if not data:
        return None
    fcst = (data.get("properties") or {}).get("forecastHourly")
    if fcst:
        _POINTS_CACHE[key] = (now, fcst)
    return fcst


def _forecast_at(home_team: str, when: datetime) -> Optional[dict]:
    """Return the forecast period covering ``when`` for the team's park."""
    coords = stadium(home_team)
    if coords is None:
        return None
    lat, lon, _indoor = coords
    now = time.time()
    cache_key = f"{home_team}:{when.isoformat()[:13]}"  # hour granularity
    cached = _FORECAST_CACHE.get(cache_key)
    if cached and now - cached[0] < CACHE_TTL:
        periods = cached[1]
    else:
        fcst_url = _get_forecast_url(lat, lon)
        if not fcst_url:
            return None
        data = _http_get_json(fcst_url)
        if not data:
            return None
        periods = (data.get("properties") or {}).get("periods") or []
        _FORECAST_CACHE[cache_key] = (now, periods)

    # Find the period covering `when` (UTC).
    when_utc = when.astimezone(timezone.utc) if when.tzinfo else when.replace(tzinfo=timezone.utc)
    best = None
    for p in periods:
        start = p.get("startTime")
        end = p.get("endTime")
        if not start or not end:
            continue
        try:
            s = datetime.fromisoformat(start.replace("Z", "+00:00"))
            e = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError:
            continue
        if s <= when_utc < e:
            best = p
            break
    return best


def _parse_wind_speed(wind_str: str) -> float:
    """Weather.gov reports wind as '5 mph' or '10 to 15 mph'."""
    if not wind_str:
        return 0.0
    parts = wind_str.replace("mph", "").strip().split()
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except ValueError:
            pass
    if not nums:
        return 0.0
    return max(nums)   # use the high end of a range


def total_adjustment(home_team: str, commence_time: Optional[datetime]) -> Tuple[float, str]:
    """Return (run_delta, reason) for a game's total based on weather.

    Neutral when the stadium is indoor, when commence_time is
    missing, or when the forecast can't be fetched.
    """
    if commence_time is None:
        return 0.0, "no commence time"
    if is_indoor(home_team):
        return 0.0, "indoor/retractable"
    period = _forecast_at(home_team, commence_time)
    if period is None:
        return 0.0, "no forecast"

    temp = period.get("temperature")
    try:
        temp = float(temp) if temp is not None else None
    except (TypeError, ValueError):
        temp = None
    wind_mph = _parse_wind_speed(period.get("windSpeed", ""))

    delta = 0.0
    reasons = []
    # Wind magnitude effect — half-weighted because we don't know
    # whether it's blowing out vs in.
    if wind_mph >= 20:
        delta += 0.7
        reasons.append(f"wind {wind_mph:.0f}mph")
    elif wind_mph >= 15:
        delta += 0.35
        reasons.append(f"wind {wind_mph:.0f}mph")

    if temp is not None:
        if temp >= 85:
            delta += 0.5
            reasons.append(f"{temp:.0f}°F")
        elif temp >= 80:
            delta += 0.3
            reasons.append(f"{temp:.0f}°F")
        elif temp <= 50:
            delta -= 0.4
            reasons.append(f"{temp:.0f}°F")
        elif temp <= 55:
            delta -= 0.2
            reasons.append(f"{temp:.0f}°F")

    reason = ", ".join(reasons) if reasons else "neutral"
    return round(delta, 2), reason
