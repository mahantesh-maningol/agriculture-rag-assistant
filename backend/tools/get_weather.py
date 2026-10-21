
from datetime import date, timedelta
from typing import Literal, Optional

import requests
from langchain_core.tools import tool


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def resolve_location(city: str) -> dict:
    """Resolve a city name to coordinates."""
    response = requests.get(
        GEOCODING_URL,
        params={
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json",
        },
        timeout=10,
    )
    response.raise_for_status()

    results = response.json().get("results", [])

    if not results:
        raise ValueError(f"Location not found: {city}")

    location = results[0]

    return {
        "name": location["name"],
        "country": location.get("country"),
        "latitude": location["latitude"],
        "longitude": location["longitude"],
    }


@tool
def get_weather(
    city: str,
    query_type: Literal[
        "current", "forecast", "spray_window"
    ] = "current",
    date: Optional[str] = None,
    days: int = 3,
    crop: Optional[str] = None,
) -> dict:
    """
    Retrieve live weather information for agricultural conversations.

    Use current for present weather.
    Use forecast for upcoming weather or rainfall questions.
    Use spray_window for hourly weather conditions relevant to
    evaluating a crop spraying opportunity.

    date must use YYYY-MM-DD when supplied.
    days is the forecast duration, between 1 and 16.
    crop provides context but does not determine spray safety.
    """

    if not 1 <= days <= 16:
        return {"error": "days must be between 1 and 16"}

    if date:
        try:
            requested_date = date_module(date)
        except ValueError:
            return {"error": "date must use YYYY-MM-DD"}

        if requested_date < date_module.today():
            return {
                "error": (
                    "Past dates require the historical weather API."
                )
            }
    else:
        requested_date = None

    try:
        location = resolve_location(city)

        forecast_days = days

        if requested_date:
            offset = (requested_date - date_module.today()).days
            forecast_days = max(forecast_days, offset + 1)

        forecast_days = min(forecast_days, 16)

        params = {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "timezone": "auto",
            "forecast_days": forecast_days,
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "apparent_temperature,precipitation,"
                "weather_code,wind_speed_10m,wind_gusts_10m"
            ),
            "hourly": (
                "temperature_2m,relative_humidity_2m,"
                "precipitation_probability,precipitation,"
                "wind_speed_10m,wind_gusts_10m"
            ),
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,precipitation_probability_max,"
                "wind_speed_10m_max,sunrise,sunset"
            ),
        }

        response = requests.get(
            FORECAST_URL, params=params, timeout=15
        )
        response.raise_for_status()
        weather = response.json()

        result = {
            "location": location,
            "timezone": weather.get("timezone"),
            "query_type": query_type,
            "crop": crop,
        }

        if query_type == "current":
            result["current"] = weather.get("current", {})
            result["current_units"] = weather.get(
                "current_units", {}
            )
            return result

        daily = weather.get("daily", {})
        hourly = weather.get("hourly", {})

        selected_date = (
            requested_date.isoformat()
            if requested_date
            else None
        )

        if selected_date:
            daily_dates = daily.get("time", [])
            matching_indices = [
                i for i, day in enumerate(daily_dates)
                if day == selected_date
            ]

            if not matching_indices:
                return {
                    **result,
                    "error": (
                        "The requested date is outside "
                        "the available forecast."
                    ),
                }

            i = matching_indices[0]
            result["daily"] = {
                key: values[i]
                for key, values in daily.items()
                if isinstance(values, list)
                and i < len(values)
            }

        elif query_type == "forecast":
            result["daily"] = daily
            result["daily_units"] = weather.get(
                "daily_units", {}
            )

        if query_type == "spray_window":
            hourly_times = hourly.get("time", [])

            # Include only hourly entries for the requested date.
            target = selected_date or date_module.today().isoformat()

            indices = [
                i for i, timestamp in enumerate(hourly_times)
                if timestamp.startswith(target)
            ]

            result["hourly"] = {
                key: [values[i] for i in indices]
                for key, values in hourly.items()
                if isinstance(values, list)
                and len(values) == len(hourly_times)
            }
            result["hourly_units"] = weather.get(
                "hourly_units", {}
            )

        result["forecast_days"] = forecast_days
        return result

    except requests.RequestException:
        return {
            "error": "Weather service is temporarily unavailable."
        }
    except (ValueError, KeyError, IndexError) as exc:
        return {"error": str(exc)}


# Alias datetime.date to avoid conflict with the tool's date argument.
from datetime import date as date_module
