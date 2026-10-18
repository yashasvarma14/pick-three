"""
Turn the pipeline's run state into plain JSON for the website.

The state has DataFrames, datetimes and NaNs in it; the browser only needs
a small, clean subset, so this picks what the page shows and nothing else
(no API keys, no raw agent internals, no exact visitor location).
"""

import math
from datetime import date, datetime

# Fields of a place that the page uses.
PLACE_FIELDS = [
    "place_id", "place_name", "primary_type", "address", "latitude", "longitude", "rating", "review_count",
    "price_level", "google_maps_uri", "website_uri", "straight_line_miles", "traffic_minutes", "distance_miles",
    "open_at_planned_time", "weekday_descriptions", "photo_names", "source_activity", "matched_searches",
    "list_rank", "top_pick", "final_rank",
    # scores (shown in the "why this ranked here" panel)
    "activity_context_score", "place_quality_score", "mobility_score", "evidence_fit_score", "photo_vibe_score",
    "final_recommendation_score", "place_activity_relevance_score",
    # evidence
    "rag_evidence_summary", "rag_strengths", "rag_concerns", "vibe_tags", "photo_summary", "photo_attributions",
]

EVENT_FIELDS = ["event_name", "event_url", "local_date", "local_time", "venue_name", "genre", "segment",
                "price_min", "latitude", "longitude"]


def clean(value):
    """Make one value JSON-safe: NaN -> None, datetimes -> text, numpy -> Python."""
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):  # numpy scalars
        return clean(value.item())
    return value


def pick(row, fields):
    return {f: clean(row.get(f)) for f in fields if f in row}


def public_result(state):
    plan = state.get("orchestrator_plan") or {}
    weather = state.get("weather") or {}
    usage = state.get("usage") or {}
    location = state.get("location") or {}
    return clean({
        "run_id": state.get("run_id"),
        "ok": bool(state.get("final_recommendations")),
        "finish_reason": state.get("finish_reason"),
        "runtime_seconds": state.get("runtime_seconds"),
        "time_used": state.get("time_used"),
        "search_phrases": state.get("search_phrases") or [],
        "picks": [pick(r, PLACE_FIELDS) for r in state.get("final_recommendations") or []],
        "places": [pick(r, PLACE_FIELDS) for r in state.get("all_places") or []],
        "events": [pick(e, EVENT_FIELDS) for e in state.get("events") or []],
        "write_up": state.get("final_response"),
        "notes": {
            "clarification_question": state.get("clarification_question"),
            "weak_match": bool(state.get("weak_match")),
            "searched_elsewhere": state.get("searched_elsewhere"),
            "adaptations": state.get("adaptation_log") or [],
        },
        "trace": {
            "goal": plan.get("goal_summary"),
            "reasoning": plan.get("planning_reasoning_summary") or [],
            "ran": state.get("agents_called") or [],
            "skipped": state.get("agents_skipped") or [],
            "timings": state.get("timings") or {},
        },
        "weather": {k: weather.get(k) for k in ["condition", "temperature_f", "precipitation_probability",
                                                 "weather_suitability"]} if weather else None,
        "usage": {"llm_tokens": usage.get("total_tokens"), "llm_cost": usage.get("llm_cost"),
                  "api_calls": usage.get("api_calls")},
        # The center of the map. Rounded so an exact GPS position never travels back.
        "center": {"latitude": round(location["latitude"], 3), "longitude": round(location["longitude"], 3),
                   "label": location.get("label")} if location.get("latitude") is not None else None,
        "errors": list(dict.fromkeys(state.get("errors") or [])),
        "warnings": list(dict.fromkeys(state.get("warnings") or [])),
    })
