"""
Step 5 - Score every place found (embeddings + simple math, no LLM)
===================================================================
Every place Google returned gets a pre-score:

    0.45 fit        embedding similarity between the request and the place
    0.25 quality    star rating, adjusted for how many reviews it has
    0.20 proximity  closer is better, within the drive range
    0.10 open       open at the planned time (unknown hours = neutral)

and outdoor places lose some score in bad weather. This ranks the full
list the app shows; only the top of it goes on to the slower LLM checks.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from .place_search import search_radius_km



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
