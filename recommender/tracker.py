"""
Live tracker: logs every search and every piece of feedback to a Google
Sheet, so you can see what people actually ask for and how the app
performs (a real version of the planning "Agentic_Recommender_Tracker").

How it works
  The Sheet has a small Apps Script (tracker_apps_script.gs) deployed as a
  web app. The app POSTs rows to that URL with a shared token; the script
  appends them to the right tab, creating tabs and columns as needed.

Settings (Streamlit secrets or environment)
  TRACKER_URL     the web app URL ending in /exec   (no URL = logging off)
  TRACKER_TOKEN   the shared token written in the script

Logging never slows down or breaks the app: it runs in a background
thread, and any error is ignored.

Privacy: no exact device location is logged (place name, or coordinates
rounded to ~10 km for "Use my location"). Prompts are logged as typed.
"""

import threading
import uuid
from datetime import datetime, timezone

import requests

from . import config

APP_VERSION = "v9-near-me"


def enabled():
    return bool(config.get_setting("TRACKER_URL"))


def new_session_id():
    return "S-" + uuid.uuid4().hex[:8]


def _http_post(url, payload):
    return requests.post(url, json=payload, timeout=10)


def _post(payload):
    url = config.get_setting("TRACKER_URL")
    if not url:
        return False
    try:
        payload = dict(payload, token=config.get_setting("TRACKER_TOKEN") or "")
        response = _http_post(url, payload)
        return response.status_code < 400
    except Exception:
        return False


def send(payload, background=True):
    """Fire-and-forget by default; background=False is for tests and notebooks."""
    if not enabled():
        return False
    if background:
        threading.Thread(target=_post, args=(payload,), daemon=True).start()
        return True
    return _post(payload)


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _num(value, digits=4):
    try:
        if value is None or value != value:
            return ""
        return round(float(value), digits)
    except (TypeError, ValueError):
        return ""


def _location_label(state):
    loc = state.get("location") or {}
    label = loc.get("label") or ""
    if label == "Your current location" and loc.get("latitude") is not None:
        # Device GPS: keep only ~10 km precision.
        return f"Device location (~{round(loc['latitude'], 1)}, {round(loc['longitude'], 1)})"
    return label


def run_rows(state, session_id):
    """All rows describing one search, keyed by sheet tab."""
    run_id = state.get("run_id")
    prefs = state.get("preferences") or {}
    usage = state.get("usage") or {}
    calls = usage.get("api_calls") or {}
    picks = state.get("final_recommendations") or []
    ts = _now()

    run_log = {
        "Run_ID": run_id,
        "Timestamp": ts,
        "Session_ID": session_id,
        "App_Version": APP_VERSION,
        "Location": _location_label(state),
        "Searched_Elsewhere": state.get("searched_elsewhere") or "",
        "Prompt": state.get("user_prompt") or "",
        "Searches": ", ".join(state.get("search_phrases") or []),
        "Time_Checked": state.get("time_used") or "",
        "Group_Size": prefs.get("group_size") or "",
        "Budget": prefs.get("budget_level") or "",
        "Places_Found": len(state.get("all_places") or []),
        "Picks_Returned": len(picks),
        "Pick_1": picks[0].get("place_name") if len(picks) > 0 else "",
        "Pick_2": picks[1].get("place_name") if len(picks) > 1 else "",
        "Pick_3": picks[2].get("place_name") if len(picks) > 2 else "",
        "Weak_Match": "Yes" if state.get("weak_match") else "No",
        "Planner_Question": state.get("clarification_question") or "",
        "Runtime_Sec": _num(state.get("runtime_seconds"), 2),
        "Input_Tokens": usage.get("input_tokens", 0),
        "Output_Tokens": usage.get("output_tokens", 0),
        "Total_Tokens": usage.get("total_tokens", 0),
        "LLM_Cost_USD": _num(usage.get("llm_cost"), 5),
        "Google_Places_Calls": calls.get("google_places", 0),
        "Google_Routes_Calls": calls.get("google_routes", 0),
        "Ticketmaster_Calls": calls.get("ticketmaster", 0),
        "Agents_Called": ", ".join(state.get("agents_called") or []),
        "Agents_Skipped": ", ".join(state.get("agents_skipped") or []),
        "Adaptations": " | ".join(state.get("adaptation_log") or []),
        "Errors": " | ".join(state.get("errors") or []),
        "Finish_Reason": state.get("finish_reason") or "",
    }

    timings = state.get("timings") or {}
    agent_rows = []
    for name, result in (state.get("agent_results") or {}).items():
        u = (result or {}).get("usage") or {}
        agent_rows.append({
            "Run_ID": run_id,
            "Timestamp": ts,
            "Agent": name,
            "Status": (result or {}).get("status", ""),
            "Seconds": _num(timings.get(name), 2),
            "Input_Tokens": u.get("input_tokens", 0) or 0,
            "Output_Tokens": u.get("output_tokens", 0) or 0,
            "LLM_Cost_USD": _num(u.get("cost"), 5),
            "Cached": "Yes" if u.get("cached") else "No",
            "Note": (result or {}).get("error") or "; ".join((result or {}).get("warnings") or [])[:300],
        })
    for name in ("scoring",):
        if name in timings:
            agent_rows.append({"Run_ID": run_id, "Timestamp": ts, "Agent": name, "Status": "success",
                               "Seconds": _num(timings[name], 2), "Input_Tokens": 0, "Output_Tokens": 0,
                               "LLM_Cost_USD": 0, "Cached": "No", "Note": "code, no LLM"})

    rec_rows = []
    for rank, rec in enumerate(picks, start=1):
        rec_rows.append({
            "Run_ID": run_id,
            "Timestamp": ts,
            "Rank": rank,
            "Place": rec.get("place_name") or "",
            "Type": rec.get("primary_type") or "",
            "Found_By": rec.get("source_activity") or "",
            "Final_Score": _num(rec.get("final_recommendation_score")),
            "Request_Fit": _num(rec.get("activity_context_score")),
            "Place_Quality": _num(rec.get("place_quality_score")),
            "Travel_Score": _num(rec.get("mobility_score")),
            "Review_Evidence": _num(rec.get("evidence_fit_score")),
            "Relevance_Check": _num(rec.get("place_activity_relevance_score")),
            "Rating": _num(rec.get("rating"), 1),
            "Reviews": rec.get("review_count") or "",
            "Drive_Min": _num(rec.get("traffic_minutes"), 1),
            "Open_At_Time": {True: "Yes", False: "No"}.get(rec.get("open_at_planned_time"), "Unknown"),
            "Maps_Link": rec.get("google_maps_uri") or "",
        })

    return {"Run_Log": [run_log], "Agent_Usage": agent_rows, "Recommendations": rec_rows}


def log_run(state, session_id, background=True):
    if not state or not state.get("run_id"):
        return False
    return send({"kind": "run", "tabs": run_rows(state, session_id)}, background=background)


def log_feedback(run_id, session_id, helpful=None, text=None, summary=None, background=True):
    row = {
        "Run_ID": run_id,
        "Timestamp": _now(),
        "Session_ID": session_id,
        "Helpful": {True: "Yes", False: "No"}.get(helpful, ""),
        "Feedback_Text": text or "",
        "Feedback_Summary": summary or "",
    }
    return send({"kind": "feedback", "tabs": {"User_Feedback": [row]}}, background=background)
