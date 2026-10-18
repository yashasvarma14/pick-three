"""
Events agent (Ticketmaster, no LLM; runs in the background)
===========================================================
Concerts, games and shows near the visitor around the planned time. They
are shown in their own tab next to the recommendations.
"""

import time

import pandas as pd

from ..core import config
from ..core.state import failed, make_agent_response, skipped
from ..services.ticketmaster import _fmt, build_event_window_utc, normalize_event_result, search_events



def events_agent(target_datetime=None, location=None, radius_miles=None, max_results=20):
    """Live events within radius_miles of `location` ({latitude, longitude, timezone})."""
    start_time = time.time()
    key = config.ticketmaster_api_key()
    if not key:
        return skipped("events", "Ticketmaster API is unavailable.")
    if not location:
        return skipped("events", "Events Agent requires a location.")

    try:
        window_start, window_end = build_event_window_utc(target_datetime, location.get("timezone"))
        raw_events = search_events(key, location["latitude"], location["longitude"], window_start, window_end,
                                   radius_miles, max_results)

        events_df = pd.DataFrame([normalize_event_result(e) for e in raw_events])
        if not events_df.empty:
            events_df = events_df.drop_duplicates("event_id").reset_index(drop=True)
            events_df.insert(0, "event_rank", range(1, len(events_df) + 1))

        return make_agent_response(
            "events",
            data={
                "search_start_utc": _fmt(window_start),
                "search_end_utc": _fmt(window_end),
                "events_found": len(events_df),
                "results": events_df,
                "runtime_seconds": time.time() - start_time,
            },
            sources=["Ticketmaster Discovery API"],
        )
    except Exception as exc:
        return failed("events", exc)
