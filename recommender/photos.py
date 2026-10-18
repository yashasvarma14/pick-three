"""
Photo Vibe Agent: looks at each shortlisted venue's Google Maps photos and
judges whether the place *looks* like what the visitor asked for (cozy,
lively, romantic, kid-friendly, scenic...).

Text evidence says what a venue offers; photos show the atmosphere, which
reviews and websites rarely describe well.

  Place Details "photos"  -> photo references (already fetched with reviews)
  Place Photo media        -> short-lived public image URLs (billed per photo)
  vision LLM, one call per venue, all at the same time
                           -> photo_vibe_score, vibe tags, one-line summary

The LLM only describes what it can see; the score is used by Matching like
any other signal and is dropped (weights re-normalized) when there are no
photos.
"""

import time

import requests

from . import config
from .base import failed, make_agent_response, skipped, ttl_cache
from .llm import call_llm_json, merge_usage, run_concurrently

PHOTO_MEDIA_URL = "https://places.googleapis.com/v1/{name}/media"

SYSTEM_PROMPT = """
You are the Photo Vibe Agent for a local activity recommendation system.

You see a few Google Maps photos of ONE venue, plus what the visitor asked for.
Judge only what is visible: setting, lighting, crowd, decor, space, view,
cleanliness, who the place seems to be for. Do not guess the menu, prices or
service. Photos can be old or unrepresentative: when they are unclear, say so
and stay near 0.5.

Return ONLY valid JSON:
{
  "place_id": "",
  "photo_vibe_score": 0.0,
  "vibe_tags": [],
  "photo_summary": "",
  "visual_concerns": []
}

photo_vibe_score is between 0 and 1: how well the look and atmosphere fit the
request (0.9+ looks made for it, 0.5 neutral or unclear, below 0.3 looks wrong
for it, e.g. a bright fast-food room for "romantic date night").
vibe_tags: 2-4 short words such as "cozy", "dim lighting", "outdoor seating",
"lively", "family-friendly". photo_summary: one sentence. Copy place_id exactly.
"""


@ttl_cache(seconds=60 * 60, maxsize=512)
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


def _photo_urls(photos, per_place):
    urls, credits = [], []
    for photo in (photos or [])[:per_place]:
        try:
            url = photo_uri(photo["name"])
        except Exception:
            continue
        if url:
            urls.append(url)
            credits.extend(photo.get("attributions") or [])
    return urls, list(dict.fromkeys(credits))


def photo_vibe_agent(places_df, photos_by_place, user_query, max_places=5, photos_per_place=3):
    """
    places_df        the shortlist (place_id, place_name, primary_type)
    photos_by_place  place_id -> photo references from place_details()
    """
    start = time.time()
    if not config.google_maps_api_key():
        return skipped("photo_vibe", "Google Places API is unavailable.")
    if places_df is None or places_df.empty:
        return skipped("photo_vibe", "No venues to look at.")
    rows = [r for r in places_df.head(max_places).to_dict("records") if photos_by_place.get(r["place_id"])]
    if not rows:
        return skipped("photo_vibe", "None of the shortlisted venues has Google photos.")

    try:
        # Photo links for every venue at once, then one vision call per venue at once.
        links = run_concurrently(lambda r: _photo_urls(photos_by_place[r["place_id"]], photos_per_place), rows)
        jobs = [(row, urls, credits) for row, ((urls, credits), _) in zip(rows, links) if urls]
        if not jobs:
            return skipped("photo_vibe", "Google returned no usable photos for the shortlist.")

        def look(job):
            row, urls, _ = job
            message = (f"USER REQUEST:\n{user_query}\n\nVENUE:\nplace_id: {row['place_id']}\n"
                       f"name: {row.get('place_name')}\ntype: {row.get('primary_type')}\n"
                       f"{len(urls)} photo(s) attached.")
            return call_llm_json(SYSTEM_PROMPT, message, image_urls=urls)

        outcomes = run_concurrently(look, jobs)
        assessments, usages = [], []
        for (row, urls, credits), (result, _) in zip(jobs, outcomes):
            if result is None:
                continue
            parsed, usage = result
            usages.append(usage)
            try:
                score = max(0.0, min(1.0, float(parsed.get("photo_vibe_score"))))
            except (TypeError, ValueError):
                continue
            assessments.append({
                "place_id": row["place_id"],  # ours, not the model's copy
                "place_name": row.get("place_name"),
                "photo_vibe_score": score,
                "vibe_tags": [str(t) for t in parsed.get("vibe_tags") or []][:4],
                "photo_summary": parsed.get("photo_summary"),
                "visual_concerns": [str(c) for c in parsed.get("visual_concerns") or []],
                "photo_urls": urls,
                "photo_attributions": credits,
            })
        if not assessments:
            error = next((exc for _, exc in outcomes if exc is not None), None)
            raise error or RuntimeError("No photo assessment could be parsed.")

        return make_agent_response(
            "photo_vibe",
            data={
                "photo_assessments": assessments,
                "venues_with_photos": len(jobs),
                "photos_used": sum(len(urls) for _, urls, _ in jobs),
                "parallel_calls": len(jobs),
                "runtime_seconds": time.time() - start,
            },
            sources=["Google Maps photos"],
            warnings=[f"No photo assessment for {len(jobs) - len(assessments)} venue(s)."]
            if len(assessments) < len(jobs) else [],
            usage=merge_usage(usages),
        )
    except Exception as exc:
        return failed("photo_vibe", exc)
