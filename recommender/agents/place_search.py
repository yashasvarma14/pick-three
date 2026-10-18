"""
Step 4 - Place search agent (Google Places, no LLM)
===================================================
Runs every search phrase inside a box around the visitor, sized from
their "longest drive" setting, and merges the results (a place found by
several searches keeps all of them). Places that are too far away or
closed for good are dropped here.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..core import config
from ..core.helpers import distance_km
from ..core.state import failed, make_agent_response, skipped
from ..services.google_places import search_area_phrase



# Average city driving speed used only to size the search area.
KM_PER_DRIVE_MINUTE = 0.7


MIN_RADIUS_KM, MAX_RADIUS_KM = 3.0, 40.0


OUTDOOR_TYPES = {
    "park", "hiking_area", "beach", "campground", "national_park", "state_park", "garden",
    "botanical_garden", "playground", "dog_park", "marina", "golf_course", "ski_resort",
    "picnic_ground", "barbecue_area", "observation_deck", "plaza", "skateboard_park",
    "cycling_park", "off_roading_area", "fishing_pond", "fishing_charter", "lake", "river",
}


MIXED_TYPES = {"zoo", "amusement_park", "water_park", "stadium", "farmers_market", "market",
               "tourist_attraction", "amphitheatre", "sports_complex", "athletic_field", "wildlife_park"}


def search_radius_km(max_travel_minutes):
    minutes = max_travel_minutes or 20
    return float(min(MAX_RADIUS_KM, max(MIN_RADIUS_KM, minutes * KM_PER_DRIVE_MINUTE)))


def environment_from_types(types):
    types = set(types or [])
    if types & OUTDOOR_TYPES:
        return "outdoor"
    if types & MIXED_TYPES:
        return "mixed"
    return "indoor"


def _row(place, phrase, origin_lat, origin_lng):
    loc = place.get("location") or {}
    lat, lng = loc.get("latitude"), loc.get("longitude")
    km = distance_km(origin_lat, origin_lng, lat, lng) if lat is not None and lng is not None else None
    hours = place.get("regularOpeningHours") or {}
    return {
        "place_id": place.get("id"),
        "place_name": (place.get("displayName") or {}).get("text"),
        "primary_type": (place.get("primaryTypeDisplayName") or {}).get("text") or (place.get("primaryType") or "").replace("_", " "),
        "address": place.get("formattedAddress"),
        "latitude": lat,
        "longitude": lng,
        "straight_line_miles": round(km * 0.621371, 2) if km is not None else None,
        "rating": place.get("rating"),
        "review_count": place.get("userRatingCount"),
        "price_level": place.get("priceLevel"),
        "place_types": place.get("types", []),
        "environment": environment_from_types(place.get("types")),
        "business_status": place.get("businessStatus"),
        "google_maps_uri": place.get("googleMapsUri"),
        "website_uri": place.get("websiteUri"),
        "weekday_descriptions": hours.get("weekdayDescriptions", []),
        "opening_periods": hours.get("periods", []),
        "source_activity": phrase,
        "matched_searches": [phrase],
    }


def area_search_agent(phrases, location, max_travel_minutes=20, pages=1):
    """Local Places agent, "near me" version: every phrase, around the visitor."""
    start = time.time()
    if not config.google_maps_api_key():
        return skipped("local_places", "Google Places API is unavailable.")
    if not phrases:
        return skipped("local_places", "No searches to run.")
    try:
        radius = search_radius_km(max_travel_minutes)
        lat, lng = location["latitude"], location["longitude"]
        def one(phrase):
            # One rejected search no longer sinks the whole run.
            try:
                return phrase, search_area_phrase(phrase, lat, lng, radius, pages), None
            except Exception as exc:
                return phrase, [], exc

        # 3 at a time (was 5): fewer bursts, fewer "too many requests".
        with ThreadPoolExecutor(max_workers=3) as pool:
            outcomes = list(pool.map(one, phrases))
        errors = [(p, e) for p, _, e in outcomes if e is not None]
        if len(errors) == len(phrases):
            raise errors[0][1]

        rows = {}
        for phrase, batch, _ in outcomes:
            for place in batch:
                row = _row(place, phrase, lat, lng)
                if not row["place_id"]:
                    continue
                if row["place_id"] in rows:
                    rows[row["place_id"]]["matched_searches"].append(phrase)
                else:
                    rows[row["place_id"]] = row

        df = pd.DataFrame(list(rows.values()))
        too_far = closed_for_good = 0
        if not df.empty:
            # The box's corners reach ~1.4x the radius; trim to a circle.
            miles_limit = radius * 1.15 * 0.621371
            far = df["straight_line_miles"].notna() & (df["straight_line_miles"] > miles_limit)
            gone = df["business_status"].isin(["CLOSED_PERMANENTLY", "CLOSED_TEMPORARILY"])
            too_far, closed_for_good = int(far.sum()), int(gone.sum())
            df = df[~far & ~gone].reset_index(drop=True)

        return make_agent_response(
            "local_places",
            data={
                "results": df,
                "searches": list(phrases),
                "radius_km": radius,
                "places_found": len(df),
                "places_too_far_removed": too_far,
                "places_closed_removed": closed_for_good,
                "api_calls": len(phrases) * max(1, pages),
                "failed_searches": [p for p, _ in errors],
                "runtime_seconds": time.time() - start,
            },
            sources=["Google Places API (Text Search)"],
            warnings=[f'Google rejected the search "{p}": {e}' for p, e in errors],
        )
    except Exception as exc:
        return failed("local_places", exc)
