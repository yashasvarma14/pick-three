"""
"Austin's top 10" lists for the home page, shown before anyone searches.

Each list is one Google Maps search around downtown Austin, ranked by the
same review-adjusted rating the recommender uses (a 5.0 from 12 reviews
doesn't beat a 4.7 from 3,000).

To keep a portfolio site free to leave running, the lists come from a
snapshot file (web/data/featured.json) made by scripts/build_featured.py,
not from Google on every visit. Re-run that script when you want fresh
lists (Google's terms ask that place details not be kept for long, so
about once a month). Set FEATURED_LIVE=1 to fetch live instead.
"""

import json
from pathlib import Path

from datetime import datetime, timedelta, timezone

import pandas as pd

from recommender.agents.place_scoring import adjusted_rating
from recommender.agents.place_search import _row
from recommender.agents.shortlist import avoid_reason, same_business
from recommender.core import config
from recommender.core.cache import ttl_cache
from recommender.services.google_places import search_area_phrase
from recommender.services.ticketmaster import normalize_event_result, search_events

SNAPSHOT = Path(__file__).resolve().parent.parent / "web" / "data" / "featured.json"

AUSTIN = {"latitude": 30.2672, "longitude": -97.7431, "label": "Austin, TX", "timezone": "America/Chicago"}
RADIUS_KM = 12  # downtown plus the close-in neighborhoods

CATEGORIES = {
    "restaurants": {"label": "Restaurants", "search": "best restaurants"},
    "bars": {"label": "Bars & cocktails", "search": "best cocktail bars"},
    "coffee": {"label": "Coffee", "search": "best coffee shops"},
    "music": {"label": "Live music", "search": "live music venues"},
    "outdoors": {"label": "Outdoors", "search": "parks swimming holes hiking trails"},
    "things": {"label": "Things to do", "search": "top attractions things to do"},
    "events": {"label": "This week", "search": None},  # Ticketmaster, when a key is set
}

ITEM_FIELDS = ["place_id", "place_name", "primary_type", "address", "rating", "review_count", "price_level",
               "google_maps_uri", "website_uri", "photo_names", "latitude", "longitude"]


def top_places(raw_places, n=10, min_reviews=150):
    """Rank Google's results by review-adjusted rating, one listing per business."""
    rows = [_row(p, "featured", AUSTIN["latitude"], AUSTIN["longitude"]) for p in raw_places]
    df = pd.DataFrame([r for r in rows if r["place_id"] and r.get("business_status") in (None, "OPERATIONAL")])
    if df.empty:
        return []
    df["score"] = adjusted_rating(df["rating"], df["review_count"])
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0)
    enough = df[df["review_count"] >= min_reviews]
    df = (enough if len(enough) >= n else df).sort_values(["score", "review_count"], ascending=False)

    picked = []
    for row in df.to_dict("records"):
        if avoid_reason(row):
            continue
        if any(same_business(row["place_name"], p["place_name"], row.get("website_uri"), p.get("website_uri"))
               for p in picked):
            continue
        picked.append(row)
        if len(picked) == n:
            break
    return [{k: row.get(k) for k in ITEM_FIELDS} | {"rank": i} for i, row in enumerate(picked, start=1)]


def load_snapshot():
    try:
        return json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def available_categories():
    """Categories the home page can show right now (snapshot first)."""
    snapshot = load_snapshot()
    if snapshot and not config.get_setting("FEATURED_LIVE"):
        return [{"id": key, "label": value["label"]} for key, value in snapshot["lists"].items() if value["items"]]
    has_events = bool(config.ticketmaster_api_key())
    return [{"id": key, "label": spec["label"]} for key, spec in CATEGORIES.items() if spec["search"] or has_events]


def featured_list(category):
    """The top-10 list for one category: from the snapshot, or live if there isn't one."""
    snapshot = load_snapshot()
    if snapshot and not config.get_setting("FEATURED_LIVE"):
        return snapshot["lists"].get(category, {}).get("items", [])
    return live_list(category)


@ttl_cache(seconds=12 * 60 * 60, maxsize=32)
def live_list(category):
    """Fetch one list from Google now (cached 12 hours). Used by the snapshot script."""
    spec = CATEGORIES[category]
    if spec["search"] is None:
        return upcoming_events()
    raw = search_area_phrase(spec["search"], AUSTIN["latitude"], AUSTIN["longitude"], RADIUS_KM, pages=1)
    return top_places(raw)


def upcoming_events(n=10):
    """The next events in Austin over the coming week (needs a Ticketmaster key)."""
    key = config.ticketmaster_api_key()
    if not key:
        return []
    start = datetime.now(timezone.utc)
    raw = search_events(key, AUSTIN["latitude"], AUSTIN["longitude"], start, start + timedelta(days=7),
                        radius_miles=15, max_results=50)
    seen, items = set(), []
    for event in (normalize_event_result(e) for e in raw):
        if not event["event_name"] or event["event_name"] in seen:
            continue
        seen.add(event["event_name"])
        items.append({"rank": len(items) + 1, "place_name": event["event_name"],
                      "primary_type": event.get("genre") or event.get("segment"), "address": event.get("venue_name"),
                      "event_url": event.get("event_url"), "local_date": event.get("local_date"),
                      "local_time": event.get("local_time")})
        if len(items) == n:
            break
    return items
