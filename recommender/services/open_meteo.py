"""Open-Meteo weather forecast (free, no API key)."""

import requests

from ..core.cache import ttl_cache



HTTP_TIMEOUT = 10


@ttl_cache(seconds=15 * 60)
def fetch_forecast(latitude, longitude):
    """7-day hourly forecast from Open-Meteo, cached for 15 minutes. One retry on a timeout."""
    try:
        return _fetch_forecast_once(latitude, longitude, HTTP_TIMEOUT)
    except requests.exceptions.Timeout:
        return _fetch_forecast_once(latitude, longitude, HTTP_TIMEOUT + 10)


def _fetch_forecast_once(latitude, longitude, timeout):
    hourly_fields = ["temperature_2m", "apparent_temperature", "precipitation_probability",
                     "precipitation", "weather_code", "wind_speed_10m"]
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": ["temperature_2m", "apparent_temperature", "precipitation", "weather_code", "wind_speed_10m"],
            "hourly": hourly_fields,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "auto",
            "forecast_days": 7,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()
