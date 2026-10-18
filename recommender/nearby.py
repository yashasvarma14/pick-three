"""
"Near me" search: start from where the visitor is, search Google Maps
around them, and score every place found.

Steps
  1. resolve_location    a typed address/neighborhood or the browser's
                         GPS coordinates -> lat/lng, label, timezone
  2. search_area         several Google Maps searches, limited to a box
                         sized from the visitor's drive limit
  3. score_places        every place gets a fit score (embedding
                         similarity between the request and the place's
                         name/type), a quality score, a proximity score,
                         and an "open at your time" check
  4. place_details       Google reviews + editorial summary for the top
                         few, used as unstructured evidence by the RAG agent
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from . import config
from .base import failed, make_agent_response, skipped, ttl_cache
from .places import distance_km
from .weather import fetch_forecast

PLACES_URL = "https://places.googleapis.com/v1/places:searchText"

SEARCH_FIELDS = ",".join([
    "places.id", "places.displayName", "places.formattedAddress", "places.location",
    "places.rating", "places.userRatingCount", "places.priceLevel", "places.types",
    "places.primaryType", "places.primaryTypeDisplayName", "places.businessStatus",
    "places.googleMapsUri", "places.websiteUri", "places.regularOpeningHours", "nextPageToken",
])

# Place Details is billed at the tier of the priciest field asked for, and
# reviews is already the top tier, so adding photo references costs nothing
# extra; downloading a photo (photos.py) is billed separately.
DETAIL_FIELDS = "id,editorialSummary,reviews,photos"

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
# 1. Where is the visitor?
# ---------------------------------------------------------------------------

@ttl_cache(seconds=24 * 60 * 60)
def _timezone_at(latitude, longitude):
    return fetch_forecast(round(latitude, 2), round(longitude, 2)).get("timezone")


@ttl_cache(seconds=24 * 60 * 60)
def _geocode_text(text):
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


@ttl_cache(seconds=24 * 60 * 60)
def _geocode_free(text):
    """
    Address -> coordinates with OpenStreetMap's free geocoder, so looking
    up "where you are" doesn't spend the Google Places daily quota.
    (Its policy: identify the app, stay around 1 request per second.)
    """
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": text, "format": "jsonv2", "limit": 1},
        headers={"User-Agent": "TonightSorted/1.0 (MSBA student project; activity recommender demo)"},
        timeout=8,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        raise LookupError(f"Couldn't find '{text}'.")
    r = results[0]
    return {"latitude": float(r["lat"]), "longitude": float(r["lon"]),
            "label": r.get("display_name") or text, "name": r.get("name") or text}


def _geocode_any(text):
    try:
        return _geocode_free(text)
    except Exception:
        return _geocode_text(text)  # Google as a fallback


def resolve_location(text=None, latitude=None, longitude=None):
    """
    Returns {latitude, longitude, label, name, timezone, origin}.
    `origin` is what the drive-time agent uses ("lat,lng").
    Raises LookupError when nothing usable was given or found.
    """
    if latitude is not None and longitude is not None:
        loc = {"latitude": float(latitude), "longitude": float(longitude),
               "label": "Your current location", "name": "your area"}
    elif text and str(text).strip():
        loc = _geocode_any(str(text).strip())
    else:
        raise LookupError("Tell me where you are: use your location or type an address.")
    try:
        loc["timezone"] = _timezone_at(loc["latitude"], loc["longitude"])
    except Exception:
        loc["timezone"] = None
    loc["origin"] = f"{loc['latitude']:.6f},{loc['longitude']:.6f}"
    return loc


def search_radius_km(max_travel_minutes):
    minutes = max_travel_minutes or 20
    return float(min(MAX_RADIUS_KM, max(MIN_RADIUS_KM, minutes * KM_PER_DRIVE_MINUTE)))


def _box(latitude, longitude, radius_km):
    dlat = radius_km / 111.0
    dlng = radius_km / (111.0 * max(0.2, math.cos(math.radians(latitude))))
    return {"rectangle": {"low": {"latitude": latitude - dlat, "longitude": longitude - dlng},
                          "high": {"latitude": latitude + dlat, "longitude": longitude + dlng}}}


# ---------------------------------------------------------------------------
# 2. Search around them
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. Score every place
# ---------------------------------------------------------------------------

def is_open_at(periods, when):
    """
    True/False if Google's weekly hours say the place is open at `when`
    (a naive local datetime), None if hours are unknown.
    """
    if not periods or when is None:
        return None
    week = 7 * 24 * 60
    # Google: day 0 = Sunday. Python: Monday = 0.
    now = ((when.weekday() + 1) % 7) * 1440 + when.hour * 60 + when.minute
    for period in periods:
        o = period.get("open")
        c = period.get("close")
        if not o:
            continue
        if not c:  # open 24/7
            return True
        start = o.get("day", 0) * 1440 + o.get("hour", 0) * 60 + o.get("minute", 0)
        end = c.get("day", 0) * 1440 + c.get("hour", 0) * 60 + c.get("minute", 0)
        if end <= start:
            end += week
        for t in (now, now + week):
            if start <= t < end:
                return True
    return False


def _place_text(row):
    types = ", ".join(t.replace("_", " ") for t in (row.get("place_types") or [])[:6])
    return f"{row.get('place_name')}. {row.get('primary_type') or ''}. {types}. Found by searching: {', '.join(row.get('matched_searches') or [])}"


PRIOR_REVIEWS = 40  # how many "average" reviews every place starts with


def adjusted_rating(rating, review_count, prior_reviews=PRIOR_REVIEWS):
    """
    Bayesian average: each place's stars are pulled toward the typical
    rating of the places found, more strongly when it has few reviews.
    5.0 from 3 reviews -> ~4.3; 4.6 from 2,000 reviews stays ~4.6.
    """
    r = pd.to_numeric(rating, errors="coerce")
    v = pd.to_numeric(review_count, errors="coerce").fillna(0)
    # Capped at 4.4 so a strong neighborhood can't lift thin ratings over proven ones.
    typical = min(float(r.median()), 4.4) if r.notna().any() else 4.2
    return ((v * r.fillna(typical) + prior_reviews * typical) / (v + prior_reviews)).round(3)


def score_places(embedder, places_df, user_request, max_travel_minutes=20, when=None, weather_suitability=None):
    """
    Score every place found:
        fit        0.45  embedding similarity between the request and the place
        quality    0.25  Bayesian-adjusted rating (few reviews = pulled toward typical)
        proximity  0.20  closer is better, within the drive range
        open       0.10  open at the planned time (unknown hours = neutral)
    Outdoor places lose some score in bad weather.
    """
    df = places_df.copy()
    if df.empty:
        return df
    request_vec = np.asarray(embedder.encode([user_request]))[0]
    place_vecs = np.asarray(embedder.encode([_place_text(r) for r in df.to_dict("records")]))
    sim = place_vecs @ request_vec
    # Raw similarities are compressed (~0.1-0.6); spread them over 0.3-1.0
    # so they're comparable with the other 0-1 scores.
    spread = (sim - sim.min()) / (sim.max() - sim.min()) if sim.max() > sim.min() else np.ones_like(sim)
    df["semantic_similarity"] = sim
    df["venue_fit_score"] = 0.30 + 0.70 * spread
    # A place found by several of the searches is a stronger match.
    df["venue_fit_score"] = (df["venue_fit_score"] + 0.05 * (df["matched_searches"].apply(len) - 1)).clip(0, 1)

    df["adjusted_rating"] = adjusted_rating(df["rating"], df["review_count"])
    # 3.0 stars -> 0, 5.0 stars -> 1: spreads out the 4.0-4.8 range where most places sit.
    df["place_quality_score"] = ((df["adjusted_rating"] - 3.0) / 2.0).clip(0, 1).round(4)

    radius_miles = search_radius_km(max_travel_minutes) * 0.621371
    miles = pd.to_numeric(df["straight_line_miles"], errors="coerce")
    df["proximity_score"] = (1 - (miles / radius_miles)).clip(0, 1).fillna(0.5)

    df["open_at_planned_time"] = df["opening_periods"].apply(lambda p: is_open_at(p, when or datetime.now()))
    open_score = df["open_at_planned_time"].map({True: 1.0, False: 0.0}).fillna(0.6).astype(float)

    df["pre_score"] = (0.45 * df["venue_fit_score"] + 0.25 * df["place_quality_score"]
                       + 0.20 * df["proximity_score"] + 0.10 * open_score)
    if weather_suitability is not None:
        penalty = df["environment"].map({"outdoor": 0.65, "mixed": 0.35}).fillna(0.10)
        df["pre_score"] = df["pre_score"] * (1 - penalty * (1 - float(weather_suitability)))
    df = df.sort_values("pre_score", ascending=False).reset_index(drop=True)
    df.insert(0, "list_rank", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------
# 4. Reviews for the shortlist
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



# ---------------------------------------------------------------------------
# 5. One business, one spot
# ---------------------------------------------------------------------------

import re
import unicodedata

_FILLER = {"the", "a", "an", "and", "&", "of", "at", "on", "in", "co", "llc", "inc"}


def name_tokens(name):
    """'The Eureka Room - Downtown' -> ['eureka', 'room']"""
    text = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode().lower()
    text = re.split(r"\s[-|@]\s|\(|,|:", text)[0]
    return [t for t in re.findall(r"[a-z0-9']+", text) if t not in _FILLER]


_SHARED_HOSTS = {"facebook.com", "instagram.com", "google.com", "sites.google.com", "yelp.com", "linktr.ee",
                 "square.site", "squareup.com", "toasttab.com", "wixsite.com", "business.site", "godaddysites.com",
                 "weebly.com", "squarespace.com", "order.online", "doordash.com", "ubereats.com", "tripadvisor.com",
                 "eventbrite.com", "austintexas.gov", "tpwd.texas.gov", "nps.gov"}


def site_domain(url):
    """'https://www.mainevent.com/austin' -> 'mainevent.com' (None for shared hosts like facebook.com)."""
    if not isinstance(url, str) or "." not in url:
        return None
    host = re.sub(r"^https?://", "", url.strip().lower()).split("/")[0].split(":")[0]
    host = host[4:] if host.startswith("www.") else host
    if host in _SHARED_HOSTS or any(host.endswith("." + h) for h in _SHARED_HOSTS):
        return None
    return host or None


def same_business(name_a, name_b, website_a=None, website_b=None):
    """
    True for branches or duplicate listings of one business:
      * same website domain        Main Event Austin / Main Event Pflugerville (mainevent.com)
      * one name contains the other  Mohawk / Mohawk Austin, Eureka Room / The Eureka Room - Downtown
    Different businesses that only share words (Austin Pizza Garden /
    Austin Pizza Kitchen) are kept apart.
    """
    domain_a, domain_b = site_domain(website_a), site_domain(website_b)
    if domain_a and domain_a == domain_b:
        return True
    a, b = name_tokens(name_a), name_tokens(name_b)
    if not a or not b:
        return False
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return set(short) <= set(long_)


def remove_duplicate_listings(df, max_miles_apart=0.1):
    """
    Google sometimes lists one venue twice (different IDs, near-identical
    names, same spot). Keep the listing with more reviews.
    """
    if df.empty:
        return df, 0
    df = df.assign(_reviews=pd.to_numeric(df["review_count"], errors="coerce").fillna(0))
    df = df.sort_values("_reviews", ascending=False).reset_index(drop=True)
    keep, kept_rows = [], []
    for i, row in df.iterrows():
        dup = False
        for j in kept_rows:
            other = df.loc[j]
            same_address = row.get("address") and row.get("address") == other.get("address")
            close = None not in (row["latitude"], other["latitude"]) and distance_km(
                row["latitude"], row["longitude"], other["latitude"], other["longitude"]) <= max_miles_apart * 1.609
            if (close or same_address) and same_business(row["place_name"], other["place_name"],
                                                         row.get("website_uri"), other.get("website_uri")):
                # Credit the kept listing with the searches that found the duplicate.
                df.at[j, "matched_searches"] = list(dict.fromkeys(other["matched_searches"] + row["matched_searches"]))
                dup = True
                break
        if not dup:
            kept_rows.append(i)
        keep.append(not dup)
    out = df[keep].drop(columns="_reviews").reset_index(drop=True)
    return out, int(len(df) - len(out))


# Places most people wouldn't want suggested in a general recommender.
# They stay in the "All places" list; they just never become a top pick.
NOT_FOR_TOP_PICKS = ["kratom", "kava", "cbd", "hemp", "vape", "smoke shop", "tobacco", "head shop", "dispensary",
                     "liquor store", "adult entertainment", "adult store", "sex shop", "strip club", "gentlemen's club", "gentlemens club", "pawn"]

BAR_WORDS = ["bar", "pub", "brewery", "brewpub", "lounge", "night club", "nightclub", "tavern", "saloon",
             "taproom", "winery", "speakeasy"]
DRINK_WORDS = ["bar", "bars", "drink", "drinks", "cocktail", "cocktails", "beer", "beers", "wine", "brewery",
               "pub", "nightlife", "happy hour", "speakeasy", "margarita", "margaritas", "whiskey"]


def avoid_reason(row):
    """Why a place shouldn't be a top pick, or None."""
    from .base import contains_keyword

    text = f"{row.get('place_name') or ''} {row.get('primary_type') or ''} " + " ".join(
        t.replace("_", " ") for t in (row.get("place_types") or []) if isinstance(t, str))
    for word in NOT_FOR_TOP_PICKS:
        if contains_keyword(text, word):
            return word
    return None


def is_bar(row):
    from .base import contains_keyword

    kind = str(row.get("primary_type") or "")
    return any(contains_keyword(kind, w) for w in BAR_WORDS)


def asks_for_drinks(request):
    from .base import contains_keyword

    return any(contains_keyword(request or "", w) for w in DRINK_WORDS)


def diverse_selection(df, n, specific_request=False, max_same_type=2, min_reviews=10, request_text=None, max_bars=None):
    """
    Walk down a ranked list and pick n places:
      * never two branches/listings of the same business
      * never a place on the NOT_FOR_TOP_PICKS list (kratom bars, smoke shops...)
      * places with fewer than `min_reviews` reviews only if nothing better
      * for open-ended requests, at most `max_same_type` of one kind of place
        and at most `max_bars` bars/pubs/lounges, unless the request is about
        drinks; specific requests ("burgers") are exempt
    """
    rows = df.to_dict("records")
    limit_bars = max_bars is not None and not specific_request and not asks_for_drinks(request_text)
    for strict in (True, False):
        chosen, types, bars = [], {}, 0
        for row in rows:
            if avoid_reason(row):
                continue
            if any(same_business(row.get("place_name"), c.get("place_name"), row.get("website_uri"), c.get("website_uri"))
                   for c in chosen):
                continue
            reviews = row.get("review_count")
            if strict and (reviews is None or reviews != reviews or reviews < min_reviews):
                continue
            kind = str(row.get("primary_type") or "").lower()
            if strict and not specific_request and kind and types.get(kind, 0) >= max_same_type:
                continue
            bar = is_bar(row)
            if strict and limit_bars and bar and bars >= max_bars:
                continue
            chosen.append(row)
            types[kind] = types.get(kind, 0) + 1
            bars += bar
            if len(chosen) == n:
                return chosen
        if len(chosen) == n or not strict:
            return chosen
    return chosen
