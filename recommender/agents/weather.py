"""
Step 3 - Weather agent (Open-Meteo, no LLM)
===========================================
Gets the forecast for the planned time and turns it into an "outdoor
suitability" score between 0 and 1. Scoring then pushes outdoor places
down when the weather is bad. Rule-based on purpose: the LLM never does
the math.
"""

import time
from datetime import datetime

from ..core.state import failed, make_agent_response, skipped
from ..services.open_meteo import fetch_forecast



WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


SEVERE_CODES = {65, 67, 75, 82, 86, 95, 96, 99}


def calculate_weather_suitability(temperature_f, precipitation_probability, wind_speed_mph, weather_code):
    """Outdoor suitability between 0 and 1, plus plain-language reasons."""
    score, reasons = 1.0, []

    if temperature_f is not None:
        if 55 <= temperature_f <= 85:
            pass
        elif 45 <= temperature_f < 55:
            score *= 0.80; reasons.append("Cool outdoor temperature")
        elif 85 < temperature_f <= 95:
            score *= 0.75; reasons.append("Warm outdoor temperature")
        elif 35 <= temperature_f < 45:
            score *= 0.55; reasons.append("Cold outdoor temperature")
        elif 95 < temperature_f <= 105:
            score *= 0.45; reasons.append("Very hot outdoor temperature")
        else:
            score *= 0.25; reasons.append("Extreme outdoor temperature")

    if precipitation_probability is not None:
        if precipitation_probability < 20:
            pass
        elif precipitation_probability < 40:
            score *= 0.85; reasons.append("Some chance of precipitation")
        elif precipitation_probability < 60:
            score *= 0.65; reasons.append("Moderate precipitation risk")
        elif precipitation_probability < 80:
            score *= 0.40; reasons.append("High precipitation risk")
        else:
            score *= 0.20; reasons.append("Very high precipitation risk")

    if wind_speed_mph is not None:
        if wind_speed_mph < 15:
            pass
        elif wind_speed_mph < 25:
            score *= 0.80; reasons.append("Moderately windy")
        elif wind_speed_mph < 35:
            score *= 0.55; reasons.append("Strong winds")
        else:
            score *= 0.30; reasons.append("Very strong winds")

    if weather_code in SEVERE_CODES:
        score *= 0.35
        reasons.append("Severe or disruptive weather conditions")

    return {"weather_suitability": round(max(0.0, min(1.0, score)), 3), "weather_reasons": reasons}


def weather_agent(target_datetime=None, location=None):
    """
    Current weather if target_datetime is None, otherwise the hourly
    forecast closest to target_datetime (a naive local datetime).
    location: {latitude, longitude, ...} from services/location.py
    """
    start = time.time()
    if not location:
        return skipped("weather", "Weather Agent requires a location.")

    try:
        weather_data = fetch_forecast(location["latitude"], location["longitude"])
        hourly = weather_data.get("hourly", {})
        hourly_times = hourly.get("time", [])

        if target_datetime is None:
            mode = "current"
            current = weather_data.get("current", {})
            selected_time = current.get("time")
            temperature_f = current.get("temperature_2m")
            apparent_f = current.get("apparent_temperature")
            precipitation = current.get("precipitation")
            weather_code = current.get("weather_code")
            wind_mph = current.get("wind_speed_10m")
            precip_prob = None
            if selected_time:
                for i, t in enumerate(hourly_times):
                    if t[:13] == selected_time[:13]:
                        precip_prob = hourly.get("precipitation_probability", [None] * len(hourly_times))[i]
                        break
        else:
            mode = "forecast"
            if not hourly_times:
                raise ValueError("No hourly forecast data returned.")
            hourly_dts = [datetime.fromisoformat(t) for t in hourly_times]
            target = target_datetime.replace(tzinfo=None)
            idx = min(range(len(hourly_dts)), key=lambda i: abs((hourly_dts[i] - target).total_seconds()))
            if abs((hourly_dts[idx] - target).total_seconds()) / 3600 > 2:
                raise ValueError("Requested time is outside the 7-day forecast range.")
            selected_time = hourly_times[idx]
            temperature_f = hourly["temperature_2m"][idx]
            apparent_f = hourly["apparent_temperature"][idx]
            precip_prob = hourly["precipitation_probability"][idx]
            precipitation = hourly["precipitation"][idx]
            weather_code = hourly["weather_code"][idx]
            wind_mph = hourly["wind_speed_10m"][idx]

        suitability = calculate_weather_suitability(temperature_f, precip_prob, wind_mph, weather_code)

        data = {
            "place": location.get("label") or location.get("name"),
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "timezone": weather_data.get("timezone") or location.get("timezone"),
            "weather_mode": mode,
            "weather_datetime": selected_time,
            "temperature_f": temperature_f,
            "apparent_temperature_f": apparent_f,
            "precipitation_in": precipitation,
            "precipitation_probability": precip_prob,
            "wind_speed_mph": wind_mph,
            "weather_code": weather_code,
            "condition": WEATHER_CODES.get(weather_code, "unknown conditions"),
            **suitability,
            "runtime_seconds": time.time() - start,
        }
        return make_agent_response("weather", data=data, confidence=1.0,
                                   sources=["Open-Meteo Forecast API"])
    except Exception as exc:
        return failed("weather", exc)
