"""
Step 8 - Travel time agent (Google Routes, no LLM)
==================================================
Live, traffic-aware drive times from the visitor to each shortlisted
place, all requested at the same time. enforce_drive_limit() then treats
"longest drive you'd accept" as a real limit.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..core import config
from ..core.state import failed, make_agent_response, skipped
from ..services.google_routes import departure_time_utc, get_google_route



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


def enforce_drive_limit(mob_df, max_travel_minutes, state):
    """
    Treat "longest drive you'd accept" as a real limit (with a 5-minute
    grace for traffic estimates). If nothing is close enough, keep the
    three closest and say so, instead of returning nothing.
    """
    if not max_travel_minutes or "traffic_minutes" not in mob_df.columns or mob_df.empty:
        return mob_df
    limit = max_travel_minutes + 5
    minutes = pd.to_numeric(mob_df["traffic_minutes"], errors="coerce")
    within = mob_df[minutes.isna() | (minutes <= limit)]
    removed = len(mob_df) - len(within)
    state["confidence"]["over_drive_limit_removed"] = removed
    if within.empty:
        state["adaptation_log"].append(
            f"Nothing found was within a {max_travel_minutes}-minute drive, so the closest options are shown instead.")
        return mob_df.assign(_m=minutes).sort_values("_m").drop(columns="_m").head(3)
    if removed:
        state["adaptation_log"].append(f"Removed {removed} venue(s) more than {max_travel_minutes} minutes away.")
    return within
