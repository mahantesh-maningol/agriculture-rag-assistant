import requests
from langchain_core.tools import tool

def geocode(city: str):
    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if not data.get("results"):
        raise ValueError(f"Location not found: {city}")

    location = data["results"][0]

    return {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "name": location["name"],
        "country": location["country"]
    }

def get_weather_data(latitude: float, longitude: float) -> str:
    """Get the current weather and today's forecast for a location."""

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    current = data["current"]
    daily = data["daily"]

    return f"""
Current weather:
Temperature: {current["temperature_2m"]} °C
Humidity: {current["relative_humidity_2m"]} %
Wind speed: {current["wind_speed_10m"]} km/h

Today's forecast:
Minimum temperature: {daily["temperature_2m_min"][0]} °C
Maximum temperature: {daily["temperature_2m_max"][0]} °C
Precipitation probability: {daily["precipitation_probability_max"][0]} %
"""


@tool
def get_weather(city: str) -> str:
    """Get the current weather and forecast for a city."""

    # 1. Convert city → latitude/longitude
    location = geocode(city)

    # 2. Call weather API
    weather = get_weather_data(
        location["latitude"],
        location["longitude"]
    )

    # 3. Return useful weather information
    return weather