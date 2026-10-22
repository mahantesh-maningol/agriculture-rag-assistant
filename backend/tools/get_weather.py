import logging
from collections import Counter
from datetime import date as Date, datetime, timedelta
from typing import Literal, Optional

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# ---- Foliar spray rules: typical good-practice values. The product label always wins. ----
MAX_WIND_KMH = 15.0
MAX_GUST_KMH = 25.0
MAX_RAIN_PROBABILITY = 30      # percent, per hour
MIN_TEMP_C = 10.0
MAX_TEMP_C = 32.0
MIN_HUMIDITY_PCT = 40
MIN_WINDOW_HOURS = 2
HORIZON_HOURS = 48
MAX_WINDOWS_RETURNED = 5

# ---- Soil application / fertigation rules ----
SOIL_MAX_DAY_RAIN_MM = 5.0
SOIL_MAX_NEXT_DAY_RAIN_MM = 10.0
SOIL_MAX_RAIN_PROBABILITY = 50

LABEL_NOTE = (
    "Weather-based guidance only. The product label's directions (rain-free period, "
    "wind limits, re-entry and pre-harvest intervals) take priority."
)

BLOCKER_TEXT = {
    "wind": "wind too strong",
    "gusts": "gusts too strong",
    "too_hot": "too hot",
    "too_cold": "too cold",
    "low_humidity": "air too dry",
    "rain_now": "rain expected",
    "rain_soon": "rain expected soon after",
    "missing_data": "forecast data missing",
}


def resolve_location(city: str) -> dict:
    """Resolve a place name to coordinates.

    Accepts 'Bengaluru' or 'Hubli, Karnataka'. Open-Meteo searches by name only,
    so the part before the first comma is searched and the rest is used to pick
    the best match (state or country).
    """
    parts = [p.strip() for p in (city or "").split(",") if p.strip()]
    if not parts:
        raise ValueError("A city name is required")

    name, qualifiers = parts[0], [q.lower() for q in parts[1:]]

    response = requests.get(
        GEOCODING_URL,
        params={"name": name, "count": 10, "language": "en", "format": "json"},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json().get("results") or []

    if not results:
        raise ValueError(f"Location not found: {city}")

    def score(r: dict) -> int:
        text = " ".join(
            str(r.get(k, "")) for k in ("admin1", "admin2", "country", "country_code")
        ).lower()
        return sum(q in text for q in qualifiers)

    best = max(results, key=score)   # on a tie, the first (most populous) result wins

    return {
        "name": best["name"],
        "region": best.get("admin1"),
        "country": best.get("country"),
        "latitude": best["latitude"],
        "longitude": best["longitude"],
    }


def _parse_requested_date(value: str, local_today: Date) -> Date:
    """Accepts YYYY-MM-DD, 'today' or 'tomorrow' (the model often does not know today's date)."""
    v = value.strip().lower()
    if v in ("today", "now"):
        return local_today
    if v == "tomorrow":
        return local_today + timedelta(days=1)
    try:
        return Date.fromisoformat(v)
    except ValueError:
        raise ValueError("date must be YYYY-MM-DD, 'today' or 'tomorrow'") from None


def _spray_windows(weather: dict, local_now: datetime, requested: Optional[Date],
                   rainfast_hours: int) -> dict:
    """Foliar application: find calm, dry, daylight hours with no rain soon afterwards."""
    hourly = weather.get("hourly") or {}
    daily = weather.get("daily") or {}
    times = hourly.get("time") or []
    n = len(times)

    def col(name):
        values = hourly.get(name)
        return values if values and len(values) == n else [None] * n

    temp, hum, prob, precip, wind, gust = (
        col(k) for k in (
            "temperature_2m", "relative_humidity_2m", "precipitation_probability",
            "precipitation", "wind_speed_10m", "wind_gusts_10m",
        )
    )
    sun = {
        d: (sr, ss)
        for d, sr, ss in zip(daily.get("time", []), daily.get("sunrise", []), daily.get("sunset", []))
    }

    def rain_ok(i: int) -> bool:
        return precip[i] is not None and precip[i] <= 0.1 and (prob[i] or 0) <= MAX_RAIN_PROBABILITY

    def is_daylight(i: int) -> bool:
        sunrise, sunset = sun.get(times[i][:10], (None, None))
        if not sunrise or not sunset:
            return True
        return sunrise <= times[i] < sunset      # same "YYYY-MM-DDTHH:MM" format, so strings compare

    def blockers(i: int) -> list:
        if any(v is None for v in (temp[i], hum[i], precip[i], wind[i], gust[i])):
            return ["missing_data"]
        reasons = []
        if wind[i] > MAX_WIND_KMH:
            reasons.append("wind")
        if gust[i] > MAX_GUST_KMH:
            reasons.append("gusts")
        if temp[i] > MAX_TEMP_C:
            reasons.append("too_hot")
        if temp[i] < MIN_TEMP_C:
            reasons.append("too_cold")
        if hum[i] < MIN_HUMIDITY_PCT:
            reasons.append("low_humidity")
        if not rain_ok(i):
            reasons.append("rain_now")
        elif any(j >= n or not rain_ok(j) for j in range(i + 1, i + 1 + rainfast_hours)):
            reasons.append("rain_soon")
        return reasons

    now_hour = local_now.strftime("%Y-%m-%dT%H:00")
    if requested:
        target = requested.isoformat()
        idx = [i for i, t in enumerate(times) if t.startswith(target) and t >= now_hour]
    else:
        start = next((i for i, t in enumerate(times) if t >= now_hour), n)
        idx = list(range(start, min(start + HORIZON_HOURS, n)))

    if not idx:
        return {"error": "No hourly forecast is available for that period."}

    good, blocked = set(), Counter()
    for i in idx:
        if not is_daylight(i):
            continue
        reasons = blockers(i)
        if reasons:
            blocked.update(reasons)
        else:
            good.add(i)

    runs, run = [], []
    for i in idx:
        if i in good and (not run or i == run[-1] + 1):
            run.append(i)
        else:
            if len(run) >= MIN_WINDOW_HOURS:
                runs.append(run)
            run = [i] if i in good else []
    if len(run) >= MIN_WINDOW_HOURS:
        runs.append(run)

    def describe(r: list) -> dict:
        start_dt = datetime.fromisoformat(times[r[0]])
        end_dt = datetime.fromisoformat(times[r[-1]]) + timedelta(hours=1)
        return {
            "date": start_dt.date().isoformat(),
            "start": start_dt.strftime("%H:%M"),
            "end": end_dt.strftime("%H:%M"),
            "hours": len(r),
            "avg_wind_kmh": round(sum(wind[i] for i in r) / len(r), 1),
            "max_gust_kmh": round(max(gust[i] for i in r), 1),
            "max_rain_probability_pct": max((prob[i] or 0) for i in r),
            "avg_temp_c": round(sum(temp[i] for i in r) / len(r), 1),
        }

    windows = [describe(r) for r in runs][:MAX_WINDOWS_RETURNED]
    best = max(windows, key=lambda w: (w["hours"], -w["avg_wind_kmh"])) if windows else None

    if best:
        summary = (
            f"Best window: {best['date']} {best['start']}-{best['end']} ({best['hours']} h), "
            f"wind about {best['avg_wind_kmh']} km/h, rain chance up to "
            f"{best['max_rain_probability_pct']}%."
        )
        earliest = windows[0]
        if earliest is not best:
            summary += (
                f" Earliest window: {earliest['date']} {earliest['start']}-{earliest['end']}."
            )
    else:
        top = [BLOCKER_TEXT[k] for k, _ in blocked.most_common(3)]
        summary = "No suitable spray window in this period" + (
            f". Main reasons: {', '.join(top)}." if top else "."
        )

    return {
        "application_type": "foliar",
        "criteria": {
            "max_wind_kmh": MAX_WIND_KMH,
            "max_gust_kmh": MAX_GUST_KMH,
            "max_hourly_rain_probability_pct": MAX_RAIN_PROBABILITY,
            "temperature_range_c": [MIN_TEMP_C, MAX_TEMP_C],
            "min_humidity_pct": MIN_HUMIDITY_PCT,
            "rain_free_hours_after_spraying": rainfast_hours,
            "daylight_only": True,
            "min_window_hours": MIN_WINDOW_HOURS,
        },
        "best_window": best,
        "windows": windows,
        "blockers": {BLOCKER_TEXT[k]: v for k, v in blocked.most_common()},
        "summary": summary,
    }


def _soil_days(weather: dict, local_today: Date, requested: Optional[Date]) -> dict:
    """Soil application / fertigation: wind hardly matters, heavy rain does (runoff, leaching)."""
    daily = weather.get("daily") or {}
    dates = daily.get("time") or []
    rain = daily.get("precipitation_sum") or []
    prob = daily.get("precipitation_probability_max") or []

    if requested:
        if requested.isoformat() not in dates:
            return {"error": "The requested date is outside the available forecast."}
        wanted = [dates.index(requested.isoformat())]
    else:
        wanted = [i for i, d in enumerate(dates) if d >= local_today.isoformat()][:3]

    days = []
    for i in wanted:
        mm = rain[i] if i < len(rain) else None
        p = prob[i] if i < len(prob) else None
        next_mm = rain[i + 1] if i + 1 < len(rain) else None

        if mm is None:
            suitable, why = False, "forecast data missing"
        elif mm > SOIL_MAX_DAY_RAIN_MM or (p or 0) > SOIL_MAX_RAIN_PROBABILITY:
            suitable, why = False, "rain expected that day"
        elif next_mm is not None and next_mm > SOIL_MAX_NEXT_DAY_RAIN_MM:
            suitable, why = False, "heavy rain expected the next day"
        else:
            suitable, why = True, "little rain expected"
        days.append({
            "date": dates[i],
            "rain_mm": mm,
            "max_rain_probability_pct": p,
            "next_day_rain_mm": next_mm,
            "suitable": suitable,
            "reason": why,
        })

    first_ok = next((d for d in days if d["suitable"]), None)
    summary = (
        f"Suitable day: {first_ok['date']} ({first_ok['reason']})."
        if first_ok else "No suitable day in this period: heavy rain is expected."
    )
    return {
        "application_type": "soil",
        "criteria": {
            "max_rain_on_day_mm": SOIL_MAX_DAY_RAIN_MM,
            "max_rain_next_day_mm": SOIL_MAX_NEXT_DAY_RAIN_MM,
            "max_rain_probability_pct": SOIL_MAX_RAIN_PROBABILITY,
        },
        "days": days,
        "best_day": first_ok,
        "summary": summary,
        "note_on_rain": (
            "Heavy rain soon after surface application can wash nutrients away; "
            "light rain afterwards can help carry them into the soil."
        ),
    }


@tool
def get_weather(
    city: str,
    query_type: Literal["current", "forecast", "spray_window"] = "current",
    date: Optional[str] = None,
    days: int = 3,
    crop: Optional[str] = None,
    application_type: Literal["foliar", "soil"] = "foliar",
    rainfast_hours: int = 4,
) -> dict:
    """
    Live weather for agricultural questions, including when to apply a product.

    query_type:
      current       - weather right now.
      forecast      - daily forecast for `days` days (1-16), or one day if `date` is given.
      spray_window  - use this for ANY question about when or whether to apply a product.
                      Returns the best time windows computed from wind, gusts, rain,
                      temperature, humidity and daylight, with a ready-made `summary`.

    date: YYYY-MM-DD, 'today' or 'tomorrow'. Leave empty for the next 48 hours.
    application_type: 'foliar' for sprays on leaves (fungicides, insecticides, miticides,
        foliar fertilizers); 'soil' for products applied to the soil or through irrigation
        (granular fertilizers, fertigation, soil drenches).
    rainfast_hours: hours the product needs to stay dry after spraying. Use the product
        label's value if it is known, otherwise leave the default of 4. Never invent it.
    crop: context only; it does not change the result.

    The result is weather-based guidance. The product label's directions take priority.
    """

    if not 1 <= days <= 16:
        return {"error": "days must be between 1 and 16"}
    if not 0 <= rainfast_hours <= 12:
        return {"error": "rainfast_hours must be between 0 and 12"}

    try:
        location = resolve_location(city)

        params = {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "timezone": "auto",
            # With a specific date, fetch the full range so no offset maths is needed.
            "forecast_days": 16 if date else min(max(days, 3), 16),
            "current": (
                "temperature_2m,relative_humidity_2m,apparent_temperature,"
                "precipitation,weather_code,wind_speed_10m,wind_gusts_10m"
            ),
            "hourly": (
                "temperature_2m,relative_humidity_2m,precipitation_probability,"
                "precipitation,wind_speed_10m,wind_gusts_10m"
            ),
            "daily": (
                "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                "precipitation_probability_max,wind_speed_10m_max,sunrise,sunset"
            ),
        }
        response = requests.get(FORECAST_URL, params=params, timeout=15)
        response.raise_for_status()
        weather = response.json()

        current = weather.get("current") or {}
        # "Today" must be the LOCATION's today, not the server's.
        local_now = (
            datetime.fromisoformat(current["time"]) if current.get("time") else datetime.now()
        )
        local_today = local_now.date()

        requested = _parse_requested_date(date, local_today) if date else None
        if requested and requested < local_today:
            return {"error": "Past dates need the historical weather API."}

        result = {
            "location": location,
            "timezone": weather.get("timezone"),
            "local_time_now": local_now.strftime("%Y-%m-%d %H:%M"),
            "query_type": query_type,
            "crop": crop,
        }

        if query_type == "current":
            result["current"] = current
            result["current_units"] = weather.get("current_units", {})
            return result

        if query_type == "forecast":
            daily = weather.get("daily") or {}
            if requested:
                dates = daily.get("time") or []
                if requested.isoformat() not in dates:
                    return {**result, "error": "The requested date is outside the available forecast."}
                i = dates.index(requested.isoformat())
                result["daily"] = {
                    k: v[i] for k, v in daily.items() if isinstance(v, list) and i < len(v)
                }
            else:
                result["daily"] = {
                    k: v[:days] for k, v in daily.items() if isinstance(v, list)
                }
            result["daily_units"] = weather.get("daily_units", {})
            return result

        # spray_window
        if application_type == "soil":
            outcome = _soil_days(weather, local_today, requested)
        else:
            outcome = _spray_windows(weather, local_now, requested, rainfast_hours)

        if "error" in outcome:
            return {**result, **outcome}
        return {**result, **outcome, "note": LABEL_NOTE}

    except requests.RequestException:
        logger.exception("Weather request failed")
        return {"error": "Weather service is temporarily unavailable."}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception:
        # Never let a bug crash the agent; tell it the lookup failed instead.
        logger.exception("Unexpected error in get_weather")
        return {"error": "Could not retrieve weather data."}