"""
Local Place Discovery Agent (Google Places API, New).

Turns activity concepts like "Go to a karaoke bar with some friends"
into real venues. The LLM never invents businesses or addresses.
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from . import config
from .base import contains_keyword, failed, make_agent_response, skipped, ttl_cache

PLACE_QUERY_OVERRIDES = {
    "escape room": "escape room",
    "rock climbing": "indoor climbing gym",
    "mini golf": "mini golf",
    "music festival": "music venue",
    "movie": "movie theater",
    "movies": "movie theater",
    "karaoke": "karaoke bar",
    "bowling": "bowling alley",
    "museum": "museum",
    "arcade": "arcade",
    "climbing": "climbing gym",
    "restaurant": "restaurant",
    "coffee": "coffee shop",
    "cafe": "cafe",
    "library": "library",
    "park": "park",
    "zoo": "zoo",
    "aquarium": "aquarium",
    "spa": "spa",
    "mall": "shopping mall",
    "fishing": "fishing area",
    "golf": "golf course",
    "concert": "live music venue",
    "nightlife": "nightlife",
    "bar": "bar",
    "brewery": "brewery",
    "winery": "winery",
    "hiking": "hiking trail",
    "hike": "hiking trail",
    "picnic": "park",
    "bonfire": "outdoor recreation area",
    "swimming": "swimming pool",
    "swim": "swimming pool",
}

REMOVE_PHRASES = [
    "go to ", "go see ", "go on ", "have a ", "have ", "take a ",
    "with your close friends", "with some friends", "with your friends",
    "with a few friends", "with friends", "some friends", "a few friends",
]


def build_place_query(activity_name, city):
    """Map an activity concept to a local search query (whole-word matching)."""
    for keyword, search_term in PLACE_QUERY_OVERRIDES.items():
        if contains_keyword(activity_name, keyword):
            return f"{search_term} in {city}"

    cleaned = str(activity_name).lower()
    for phrase in REMOVE_PHRASES:
        cleaned = cleaned.replace(phrase, "")
    return f"{cleaned.strip().strip('.')} in {city}"


FIELD_MASK = ",".join([
    "places.id", "places.displayName", "places.formattedAddress", "places.location",
    "places.rating", "places.userRatingCount", "places.priceLevel", "places.types",
    "places.googleMapsUri", "places.websiteUri", "places.regularOpeningHours",
])


def _search_body(query, max_results, center_lat, center_lng, radius_m):
    body = {"textQuery": query, "maxResultCount": max_results}
    if center_lat is not None and center_lng is not None:
        # Without this, "take a nap in Los Angeles" returned a New York venue.
        body["locationBias"] = {"circle": {"center": {"latitude": center_lat, "longitude": center_lng},
                                           "radius": float(radius_m)}}
    return body


def distance_km(lat1, lng1, lat2, lng2):
    """Straight-line (haversine) distance in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


@ttl_cache(seconds=30 * 60)
def search_google_places(query, max_results=5, max_retries=3, base_wait_seconds=2,
                         center_lat=None, center_lng=None, radius_m=25000):
    """
    Places Text Search with exponential backoff on HTTP 429.
    Cached for 30 minutes: "karaoke bar in Austin" doesn't change that fast.
    """
    key = config.google_maps_api_key()
    if not key:
        raise ValueError("GOOGLE_MAPS_API_KEY is not configured.")

    for attempt in range(max_retries + 1):
        response = requests.post(
            "https://places.googleapis.com/v1/places:searchText",
            headers={"Content-Type": "application/json", "X-Goog-Api-Key": key, "X-Goog-FieldMask": FIELD_MASK},
            json=_search_body(query, max_results, center_lat, center_lng, radius_m),
            timeout=15,
        )
        if response.status_code == 200:
            return response.json().get("places", [])
        if response.status_code == 429 and attempt < max_retries:
            time.sleep(base_wait_seconds * 2 ** attempt)
            continue
        response.raise_for_status()

    raise RuntimeError("Google Places stayed rate-limited after all retries.")


def normalize_place_result(place, source_activity, activity_rank, query):
    location = place.get("location", {})
    hours = place.get("regularOpeningHours", {})
    return {
        "source_activity": source_activity,
        "activity_rank": activity_rank,
        "search_query": query,
        "place_id": place.get("id"),
        "place_name": place.get("displayName", {}).get("text"),
        "address": place.get("formattedAddress"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "rating": place.get("rating"),
        "review_count": place.get("userRatingCount"),
        "price_level": place.get("priceLevel"),
        "place_types": place.get("types", []),
        "google_maps_uri": place.get("googleMapsUri"),
        "website_uri": place.get("websiteUri"),
        "weekday_descriptions": hours.get("weekdayDescriptions", []),
    }


MAX_KM_FROM_CITY = 50


def place_query_for(row, city):
    """Use the hand-checked outing search phrase when there is one."""
    query = row.get("place_query")
    if isinstance(query, str) and query.strip():
        return f"{query.strip()} in {city}"
    return build_place_query(row["activity"], city)


def local_place_agent(candidates_df, city, activities_to_search=3, places_per_activity=2):
    """
    Search the top activity concepts and return scored, de-duplicated venues.
    Searches are centred on the city, and anything more than 50 km from
    the city centre is dropped.
    """
    start = time.time()
    if not config.google_maps_api_key():
        return skipped("local_places", "Google Places API is unavailable.")
    if not city:
        return skipped("local_places", "Local Place Agent requires a city.")
    if candidates_df is None or candidates_df.empty:
        return skipped("local_places", "No candidate activities were available.")

    try:
        # activity_rank always reflects the latest reranking (discovery,
        # memory, weather), which fixes the stale-rank bug in the notebook.
        to_search = candidates_df.sort_values("activity_rank").head(activities_to_search)
        jobs = [(row["activity"], int(row["activity_rank"]), place_query_for(row, city))
                for _, row in to_search.iterrows()]

        from .weather import geocode_city
        center = geocode_city(city) or {}
        lat, lng = center.get("latitude"), center.get("longitude")

        def run(job):
            activity, rank, query = job
            found = search_google_places(query, max_results=places_per_activity, center_lat=lat, center_lng=lng)
            return [normalize_place_result(p, activity, rank, query) for p in found]

        with ThreadPoolExecutor(max_workers=4) as pool:
            batches = list(pool.map(run, jobs))

        places_df = pd.DataFrame([p for batch in batches for p in batch])
        too_far = 0
        if not places_df.empty and lat is not None:
            km = places_df.apply(
                lambda r: distance_km(lat, lng, r["latitude"], r["longitude"])
                if pd.notna(r["latitude"]) and pd.notna(r["longitude"]) else None, axis=1)
            keep = km.isna() | (km <= MAX_KM_FROM_CITY)
            too_far = int((~keep).sum())
            places_df = places_df[keep].reset_index(drop=True)

        if not places_df.empty:
            places_df = places_df.sort_values("activity_rank").drop_duplicates("place_id").reset_index(drop=True)
            places_df["rating_normalized"] = places_df["rating"].fillna(0) / 5
            places_df["review_strength"] = places_df["review_count"].fillna(0).apply(lambda x: min(1.0, x / 500))
            places_df["place_quality_score"] = 0.80 * places_df["rating_normalized"] + 0.20 * places_df["review_strength"]
            places_df = places_df.sort_values(["activity_rank", "place_quality_score"],
                                              ascending=[True, False]).reset_index(drop=True)

        return make_agent_response(
            "local_places",
            data={
                "city": city,
                "activities_searched": len(jobs),
                "places_found": len(places_df),
                "places_too_far_removed": too_far,
                "api_calls": len(jobs),
                "results": places_df,
                "runtime_seconds": time.time() - start,
            },
            sources=["Google Places API"],
        )
    except Exception as exc:
        return failed("local_places", exc)
