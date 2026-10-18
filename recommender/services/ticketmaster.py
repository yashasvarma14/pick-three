"""Ticketmaster Discovery API: live events near a point in a time window."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

EVENTS_URL = "https://app.ticketmaster.com/discovery/v2/events.json"



def build_event_window_utc(target_datetime=None, tz_name=None, hours_before=3, hours_after=6):
    """
    Search window in UTC.

    Ticketmaster expects real UTC. Sending local times with a 'Z' (UTC)
    suffix would shift a 7 PM Austin search about 5 hours earlier.
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


def search_events(api_key, latitude, longitude, window_start, window_end, radius_miles=15, max_results=20):
    """Raw Ticketmaster search around a point. Returns the list of event dicts."""
    params = {
        "apikey": api_key,
        "startDateTime": _fmt(window_start),
        "endDateTime": _fmt(window_end),
        "size": min(max_results, 50),
        "sort": "date,asc",
        "geoPoint": geohash(latitude, longitude),
        "radius": int(max(5, radius_miles or 15)),
        "unit": "miles",
    }
    response = requests.get(EVENTS_URL, params=params, timeout=15)
    response.raise_for_status()
    return response.json().get("_embedded", {}).get("events", [])
