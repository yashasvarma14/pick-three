"""
Where is the visitor? Turns GPS coordinates or a typed address into
{latitude, longitude, label, timezone, origin}.

Typed addresses use OpenStreetMap's free geocoder first so they don't use
up the Google Places quota; Google is the fallback.
"""

import requests

from ..core.cache import ttl_cache
from .google_places import geocode_with_google
from .open_meteo import fetch_forecast



@ttl_cache(seconds=24 * 60 * 60)
def _timezone_at(latitude, longitude):
    return fetch_forecast(round(latitude, 2), round(longitude, 2)).get("timezone")


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
        headers={"User-Agent": "PickThree/1.0 (student project; activity recommender demo)"},
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
        return geocode_with_google(text)  # Google as a fallback


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
