"""
Google Places API (New): text search, place details (reviews + photo
references) and photo links.

Only raw API calls live here. Deciding what to search for, and what to do
with the results, happens in the agents.
"""

import math
import time

import requests

from ..core import config
from ..core.cache import ttl_cache



PLACES_URL = "https://places.googleapis.com/v1/places:searchText"


SEARCH_FIELDS = ",".join([
    "places.id", "places.displayName", "places.formattedAddress", "places.location",
    "places.rating", "places.userRatingCount", "places.priceLevel", "places.types",
    "places.primaryType", "places.primaryTypeDisplayName", "places.businessStatus",
    "places.googleMapsUri", "places.websiteUri", "places.regularOpeningHours", "places.photos", "nextPageToken",
])


# Place Details is billed at the tier of the priciest field asked for, and
# reviews is already the top tier, so adding photo references costs nothing
# extra; downloading a photo (photo_uri below) is billed separately.
DETAIL_FIELDS = "id,editorialSummary,reviews,photos"


PHOTO_MEDIA_URL = "https://places.googleapis.com/v1/{name}/media"


# ---------------------------------------------------------------------------
# Calling Google politely (retry on 429 'too many requests')
# ---------------------------------------------------------------------------


class GoogleRateLimited(RuntimeError):
    """Google kept answering 429 (too many requests / quota exhausted)."""


def google_error(response):
    """Google's own explanation, e.g. which quota was exceeded."""
    try:
        err = response.json().get("error", {})
        return f"{response.status_code} {err.get('status', '')}: {err.get('message', '')}".strip()
    except Exception:
        return f"{response.status_code}: {response.text[:300]}"


def post_with_backoff(url, body, headers, attempts=5):
    """
    POST that waits and retries when Google says 429. Waits 1, 2, 4, 8 s
    (plus a little randomness so parallel searches don't retry in step),
    or whatever Google's Retry-After header asks for.
    """
    import random

    for attempt in range(attempts):
        response = requests.post(url, json=body, headers=headers, timeout=15)
        if response.status_code != 429:
            break
        if attempt < attempts - 1:
            wait = response.headers.get("Retry-After")
            try:
                wait = float(wait)
            except (TypeError, ValueError):
                wait = 2 ** attempt + random.uniform(0, 0.5)
            time.sleep(min(wait, 10))
    if response.status_code == 429:
        raise GoogleRateLimited(google_error(response))
    if response.status_code >= 400:
        raise RuntimeError(f"Google Places {google_error(response)}")
    return response


def _key():
    key = config.google_maps_api_key()
    if not key:
        raise ValueError("GOOGLE_MAPS_API_KEY is not configured.")
    return key


# ---------------------------------------------------------------------------
# Text search inside a box around the visitor
# ---------------------------------------------------------------------------


def _box(latitude, longitude, radius_km):
    dlat = radius_km / 111.0
    dlng = radius_km / (111.0 * max(0.2, math.cos(math.radians(latitude))))
    return {"rectangle": {"low": {"latitude": latitude - dlat, "longitude": longitude - dlng},
                          "high": {"latitude": latitude + dlat, "longitude": longitude + dlng}}}


@ttl_cache(seconds=30 * 60)
def search_area_phrase(phrase, latitude, longitude, radius_km, pages=1):
    """
    One Google Maps search, restricted to a box around the visitor.
    Up to 20 places per page; Google stops at 3 pages (60 places).
    """
    body = {"textQuery": phrase, "pageSize": 20,
            "locationRestriction": _box(latitude, longitude, radius_km)}
    found, token = [], None
    headers = {"Content-Type": "application/json", "X-Goog-Api-Key": _key(), "X-Goog-FieldMask": SEARCH_FIELDS}
    for _ in range(max(1, min(3, pages))):
        if token:
            body["pageToken"] = token
        payload = post_with_backoff(PLACES_URL, body, headers).json()
        found.extend(payload.get("places") or [])
        token = payload.get("nextPageToken")
        if not token:
            break
    return found


@ttl_cache(seconds=24 * 60 * 60)
def geocode_with_google(text):
    """Any address, landmark, neighborhood or ZIP -> coordinates (Google Places)."""
    response = post_with_backoff(
        PLACES_URL, {"textQuery": text, "pageSize": 1},
        {"Content-Type": "application/json", "X-Goog-Api-Key": _key(),
         "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location"})
    places = response.json().get("places") or []
    if not places:
        raise LookupError(f"Couldn't find '{text}' on Google Maps.")
    p = places[0]
    return {"latitude": p["location"]["latitude"], "longitude": p["location"]["longitude"],
            "label": p.get("formattedAddress") or text, "name": (p.get("displayName") or {}).get("text") or text}


# ---------------------------------------------------------------------------
# Details (reviews, summary, photo references) and photo links
# ---------------------------------------------------------------------------


@ttl_cache(seconds=6 * 60 * 60)
def place_details(place_id):
    """Editorial summary and up to 5 Google reviews for one place."""
    response = requests.get(f"https://places.googleapis.com/v1/places/{place_id}", timeout=10,
                            headers={"X-Goog-Api-Key": _key(), "X-Goog-FieldMask": DETAIL_FIELDS})
    response.raise_for_status()
    data = response.json()
    reviews = []
    for review in data.get("reviews") or []:
        text = ((review.get("text") or review.get("originalText") or {}).get("text") or "").strip()
        if len(text) >= 40:
            reviews.append({"text": text, "rating": review.get("rating"),
                            "when": review.get("relativePublishTimeDescription")})
    photos = [{"name": p.get("name"), "width": p.get("widthPx"), "height": p.get("heightPx"),
               "attributions": [a.get("displayName") for a in p.get("authorAttributions") or [] if a.get("displayName")]}
              for p in data.get("photos") or [] if p.get("name")]
    return {"editorial_summary": (data.get("editorialSummary") or {}).get("text"), "reviews": reviews, "photos": photos}


def safe_place_details(place_id):
    try:
        return place_details(place_id)
    except Exception:
        return {"editorial_summary": None, "reviews": [], "photos": []}


@ttl_cache(seconds=6 * 60 * 60, maxsize=2048)
def photo_uri(photo_name, max_width=640):
    """
    Public image URL for one Place photo. skipHttpRedirect returns the
    googleusercontent link instead of the image, so the API key never
    leaves this server (the vision model only sees the public link).
    """
    response = requests.get(PHOTO_MEDIA_URL.format(name=photo_name), timeout=10,
                            params={"maxWidthPx": max_width, "skipHttpRedirect": "true"},
                            headers={"X-Goog-Api-Key": config.google_maps_api_key() or ""})
    response.raise_for_status()
    return response.json().get("photoUri")
