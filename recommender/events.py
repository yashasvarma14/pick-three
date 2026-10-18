"""Events Agent (Ticketmaster Discovery API)."""

import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from . import config
from .base import failed, make_agent_response, skipped
from .weather import city_name_only, geocode_city


def build_event_window_utc(target_datetime=None, tz_name=None, hours_before=3, hours_after=6):
    """
    Search window in UTC.

    Fix from the notebook: local times were sent to Ticketmaster with a
    'Z' (UTC) suffix, shifting a 7pm Austin search ~5 hours earlier.
    """
    if target_datetime is None:
        start = datetime.now(timezone.utc)
        return start, start + timedelta(hours=24)

    local_tz = ZoneInfo(tz_name) if tz_name else timezone.utc
    target_local = target_datetime if target_datetime.tzinfo else target_datetime.replace(tzinfo=local_tz)
    target_utc = target_local.astimezone(timezone.utc)
    return target_utc - timedelta(hours=hours_before), target_utc + timedelta(hours=hours_after)


def _fmt(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_event_result(event):
    start = event.get("dates", {}).get("start", {})
    classification = (event.get("classifications") or [{}])[0]
    venue = (event.get("_embedded", {}).get("venues") or [{}])[0]
    location = venue.get("location", {})
    price = (event.get("priceRanges") or [{}])[0]
    return {
        "event_id": event.get("id"),
        "event_name": event.get("name"),
        "event_url": event.get("url"),
        "local_date": start.get("localDate"),
        "local_time": start.get("localTime"),
        "event_status": event.get("dates", {}).get("status", {}).get("code"),
        "segment": classification.get("segment", {}).get("name"),
        "genre": classification.get("genre", {}).get("name"),
        "venue_name": venue.get("name"),
        "venue_address": venue.get("address", {}).get("line1"),
        "venue_city": venue.get("city", {}).get("name"),
        "latitude": pd.to_numeric(location.get("latitude"), errors="coerce"),
        "longitude": pd.to_numeric(location.get("longitude"), errors="coerce"),
        "price_min": price.get("min"),
        "price_max": price.get("max"),
    }


def geohash(latitude, longitude, precision=7):
    """Standard geohash, which Ticketmaster uses for 'events near this point'."""
    alphabet = "0123456789bcdefghjkmnpqrstuvwxyz"
    lat_range, lng_range = [-90.0, 90.0], [-180.0, 180.0]
    bits, bit, ch, even, out = [16, 8, 4, 2, 1], 0, 0, True, ""
    while len(out) < precision:
        rng, value = (lng_range, longitude) if even else (lat_range, latitude)
        mid = (rng[0] + rng[1]) / 2
        if value > mid:
            ch |= bits[bit]
            rng[0] = mid
        else:
            rng[1] = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            out += alphabet[ch]
            bit, ch = 0, 0
    return out


def events_agent(city=None, target_datetime=None, max_results=20, keyword=None, location=None, radius_miles=None):
    """
    Live events in a time window. With `location` ({latitude, longitude,
    timezone}) it searches around that point instead of by city name.
    """
    start_time = time.time()
    key = config.ticketmaster_api_key()
    if not key:
        return skipped("events", "Ticketmaster API is unavailable.")
    if not city and not location:
        return skipped("events", "Events Agent requires a city or a location.")

    try:
        place = location or geocode_city(city) or {}
        window_start, window_end = build_event_window_utc(target_datetime, place.get("timezone"))

        params = {
            "apikey": key,
            "startDateTime": _fmt(window_start),
            "endDateTime": _fmt(window_end),
            "size": min(max_results, 50),
            "sort": "date,asc",
        }
        if location:
            params.update(geoPoint=geohash(location["latitude"], location["longitude"]),
                          radius=int(max(5, radius_miles or 15)), unit="miles")
        else:
            # Ticketmaster wants a bare city name ("Austin", not "Austin, Texas").
            params["city"] = city_name_only(city)
        if keyword:
            params["keyword"] = keyword

        response = requests.get("https://app.ticketmaster.com/discovery/v2/events.json", params=params, timeout=15)
        response.raise_for_status()
        raw_events = response.json().get("_embedded", {}).get("events", [])

        events_df = pd.DataFrame([normalize_event_result(e) for e in raw_events])
        if not events_df.empty:
            events_df = events_df.drop_duplicates("event_id").reset_index(drop=True)
            events_df.insert(0, "event_rank", range(1, len(events_df) + 1))

        return make_agent_response(
            "events",
            data={
                "city": city or (location or {}).get("name"),
                "search_start_utc": _fmt(window_start),
                "search_end_utc": _fmt(window_end),
                "events_found": len(events_df),
                "results": events_df,
                "runtime_seconds": time.time() - start_time,
            },
            sources=["Ticketmaster Discovery API"],
        )
    except Exception as exc:
        return failed("events", exc)
