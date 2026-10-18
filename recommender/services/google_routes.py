"""Google Routes API: traffic-aware drive time from the visitor to one place."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

from ..core import config
from ..core.cache import ttl_cache



def parse_google_duration(duration):
    """'1238s' -> 1238.0"""
    if duration is None or not str(duration).strip().endswith("s"):
        return None
    try:
        return float(str(duration).strip()[:-1])
    except ValueError:
        return None


def departure_time_utc(target_datetime, tz_name):
    """
    Routes only accepts a future departure time. Using the planned time
    (instead of 'now') makes the traffic estimate match the plan.
    """
    if target_datetime is None:
        return None
    local_tz = ZoneInfo(tz_name) if tz_name else timezone.utc
    target = target_datetime if target_datetime.tzinfo else target_datetime.replace(tzinfo=local_tz)
    target_utc = target.astimezone(timezone.utc)
    if target_utc <= datetime.now(timezone.utc) + timedelta(minutes=2):
        return None
    return target_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _waypoint(origin):
    text = str(origin).strip()
    parts = text.split(",")
    if len(parts) == 2:
        try:
            lat, lng = float(parts[0]), float(parts[1])
            return {"location": {"latLng": {"latitude": lat, "longitude": lng}}}
        except ValueError:
            pass
    return {"address": text}


@ttl_cache(seconds=10 * 60)
def get_google_route(origin, dest_lat, dest_lng, departure_time=None):
    key = config.google_maps_api_key()
    if not key:
        raise ValueError("GOOGLE_MAPS_API_KEY is not configured.")

    payload = {
        # origin is an address ("UT Austin") or "lat,lng" from the visitor's device.
        "origin": _waypoint(origin),
        "destination": {"location": {"latLng": {"latitude": float(dest_lat), "longitude": float(dest_lng)}}},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
    }
    if departure_time:
        payload["departureTime"] = departure_time

    response = requests.post(
        "https://routes.googleapis.com/directions/v2:computeRoutes",
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": key,
            "X-Goog-FieldMask": "routes.distanceMeters,routes.duration,routes.staticDuration",
        },
        json=payload,
        timeout=15,
    )
    response.raise_for_status()
    routes = response.json().get("routes", [])
    if not routes:
        return None

    route = routes[0]
    traffic_s = parse_google_duration(route.get("duration"))
    static_s = parse_google_duration(route.get("staticDuration"))
    meters = route.get("distanceMeters")
    return {
        "distance_miles": meters / 1609.344 if meters is not None else None,
        "traffic_minutes": traffic_s / 60 if traffic_s is not None else None,
        "static_minutes": static_s / 60 if static_s is not None else None,
    }
