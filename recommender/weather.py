"""
City geocoding and the Weather Agent (Open-Meteo, no key needed).

Weather suitability is a deterministic score; the LLM never does the math.
"""

import time
from datetime import datetime

import requests

from .base import failed, make_agent_response, skipped, ttl_cache
from .data import classify_activity_environment

HTTP_TIMEOUT = 10


def city_name_only(city):
    """'Austin, Texas' -> 'Austin' (APIs that want a bare city name)."""
    return str(city).split(",")[0].strip() if city else city


US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "dc": "district of columbia", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland", "ma": "massachusetts",
    "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
    "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
}


def geocode_city(city):
    """Cached city lookup. A failed lookup (None) is not cached, so a network blip can't stick."""
    try:
        return _geocode_city(city)
    except LookupError:
        return None


geocode_city.cache_clear = lambda: _geocode_city.cache_clear()


@ttl_cache(seconds=24 * 60 * 60)
def _geocode_city(city):
    """
    Resolve a city to coordinates and timezone with Open-Meteo.
    Handles 'Austin, Texas' by searching 'Austin' and preferring the
    result whose state/region matches the text after the comma.
    """
    if not city or not str(city).strip():
        raise LookupError("no city")

    name = city_name_only(city)
    region_hint = str(city).split(",", 1)[1].strip().lower() if "," in str(city) else None
    if region_hint:
        region_hint = US_STATES.get(region_hint.replace(".", ""), region_hint)

    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 10, "language": "en", "format": "json"},
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json().get("results", []) or []
    except Exception as exc:
        raise LookupError(str(exc))

    if not results:
        raise LookupError(f"city not found: {city}")

    chosen = results[0]
    if region_hint:
        for result in results:
            region = f"{result.get('admin1', '')} {result.get('country', '')} {result.get('country_code', '')}".lower()
            if region_hint in region:
                chosen = result
                break

    return {
        "name": chosen.get("name"),
        "latitude": chosen.get("latitude"),
        "longitude": chosen.get("longitude"),
        "country": chosen.get("country"),
        "admin1": chosen.get("admin1"),
        "timezone": chosen.get("timezone"),
    }


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


def weather_agent(city=None, target_datetime=None, location=None):
    """
    Current weather if target_datetime is None, otherwise the hourly
    forecast closest to target_datetime (a naive datetime in the
    city's local time). `location` ({latitude, longitude, name}) skips
    the city lookup, e.g. for the visitor's own position.
    """
    start = time.time()
    if not city and not location:
        return skipped("weather", "Weather Agent requires a city or a location.")

    try:
        location = location or geocode_city(city)
        if location is None:
            return make_agent_response("weather", status="failed", error=f"Could not geocode city: {city}", confidence=0.0)

        hourly_fields = ["temperature_2m", "apparent_temperature", "precipitation_probability",
                         "precipitation", "weather_code", "wind_speed_10m"]
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
            "requested_city": city,
            "resolved_city": location.get("name"),
            "region": location.get("admin1"),
            "country": location.get("country"),
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
                                   sources=["Open-Meteo Forecast API", "Open-Meteo Geocoding API"])
    except Exception as exc:
        return failed("weather", exc)


# ---------------------------------------------------------
# Weather-aware reranking of activity concepts
# ---------------------------------------------------------

WEATHER_SENSITIVITY = {"indoor": 0.10, "mixed": 0.35, "outdoor": 0.65}


def calculate_weather_adjusted_score(discovery_score, environment, weather_suitability):
    if weather_suitability is None:
        return discovery_score
    sensitivity = WEATHER_SENSITIVITY.get(environment, 0.35)
    factor = (1 - sensitivity) + sensitivity * weather_suitability
    return round(discovery_score * factor, 6)


def apply_weather_reranking(candidates_df, weather_suitability):
    df = candidates_df.copy()
    if df.empty:
        return df
    if "environment" not in df.columns:
        df["environment"] = df["activity"].apply(classify_activity_environment)
    df["weather_adjusted_score"] = df.apply(
        lambda r: calculate_weather_adjusted_score(r["discovery_score"], r["environment"], weather_suitability), axis=1
    )
    df = df.sort_values("weather_adjusted_score", ascending=False).reset_index(drop=True)
    df["activity_rank"] = range(1, len(df) + 1)
    return df
