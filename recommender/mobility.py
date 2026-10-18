"""Mobility Agent (Google Routes API): traffic-aware drive time per venue."""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from . import config
from .base import failed, make_agent_response, skipped, ttl_cache


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


def calculate_mobility_score(travel_minutes, max_travel_minutes=20):
    if travel_minutes is None:
        return None
    if travel_minutes <= 10:
        return 1.00
    if travel_minutes <= max_travel_minutes:
        return 0.80
    if travel_minutes <= 30:
        return 0.60
    if travel_minutes <= 45:
        return 0.40
    return 0.20


def mobility_agent(places_df, origin, max_travel_minutes=20, max_places=5, target_datetime=None, tz_name=None):
    start = time.time()
    if not config.google_maps_api_key():
        return skipped("mobility", "Google Routes API is unavailable.")
    if not origin:
        return skipped("mobility", "Mobility Agent requires an origin.")
    if places_df is None or places_df.empty:
        return skipped("mobility", "No candidate places were available.")

    try:
        subset = places_df.head(max_places).copy()
        departure = departure_time_utc(target_datetime, tz_name)

        def run(row):
            if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
                return None
            try:
                route = get_google_route(origin, row["latitude"], row["longitude"], departure)
            except Exception:
                return None  # one failed route just leaves that venue without a drive time
            if route is None:
                return None
            traffic, static = route["traffic_minutes"], route["static_minutes"]
            result = row.to_dict()
            result.update({
                "origin": origin,
                "distance_miles": round(route["distance_miles"], 2) if route["distance_miles"] is not None else None,
                "traffic_minutes": round(traffic, 1) if traffic is not None else None,
                "static_minutes": round(static, 1) if static is not None else None,
                "traffic_delay_minutes": round(max(0, traffic - static), 1) if None not in (traffic, static) else None,
                "mobility_score": calculate_mobility_score(traffic, max_travel_minutes),
            })
            return result

        with ThreadPoolExecutor(max_workers=5) as pool:
            rows = [r for r in pool.map(run, [row for _, row in subset.iterrows()]) if r is not None]

        mobility_df = pd.DataFrame(rows)
        if not mobility_df.empty:
            mobility_df = mobility_df.sort_values(["mobility_score", "place_quality_score"], ascending=False).reset_index(drop=True)

        return make_agent_response(
            "mobility",
            data={
                "origin": origin,
                "places_evaluated": len(mobility_df),
                "api_calls": len(subset),
                "max_travel_minutes": max_travel_minutes,
                "departure_time_utc": departure,
                "results": mobility_df,
                "runtime_seconds": time.time() - start,
            },
            sources=["Google Routes API"],
        )
    except Exception as exc:
        return failed("mobility", exc)
