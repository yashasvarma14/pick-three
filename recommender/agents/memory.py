"""
Memory + Feedback agent
=======================
After a search, the visitor can type feedback ("no more movies, nothing
over 15 minutes away"). The Feedback agent (LLM) turns that into
structured memory, which the next search uses:
  * build_memory_context()   -> extra context for the planner/preference agents
  * filter_rejected_places() -> drops places they said no to

Memory lives in the visitor's browser session only, so strangers never
share or overwrite each other's preferences.
"""

import copy
import json
import time
from datetime import datetime

from ..core import llm
from ..core.state import failed, make_agent_response



# ---------------------------------------------------------------------------
# The memory dictionary
# ---------------------------------------------------------------------------


LIST_FIELDS = ["liked_activities", "disliked_activities", "liked_places", "rejected_places",
               "positive_preferences", "negative_preferences"]


def create_empty_user_memory(user_id="visitor"):
    now = datetime.now().isoformat()
    return {"user_id": user_id, "created_at": now, "updated_at": now,
            **{field: [] for field in LIST_FIELDS},
            "budget_preference": None, "max_travel_minutes": None, "feedback_history": []}


def add_unique_items(existing, new):
    cleaned = []
    for item in (existing or []) + (new or []):
        item = str(item).strip() if item is not None else ""
        if item and item.lower() not in {c.lower() for c in cleaned}:
            cleaned.append(item)
    return cleaned


def update_memory_from_feedback(memory, feedback):
    updated = copy.deepcopy(memory)
    for field in LIST_FIELDS:
        updated[field] = add_unique_items(updated.get(field, []), feedback.get(field, []))
    if feedback.get("budget_preference") in {"free", "low", "medium", "high"}:
        updated["budget_preference"] = feedback["budget_preference"]
    if feedback.get("max_travel_minutes") is not None:
        try:
            updated["max_travel_minutes"] = int(feedback["max_travel_minutes"])
        except (TypeError, ValueError):
            pass
    updated["feedback_history"] = updated.get("feedback_history", []) + [{
        "timestamp": datetime.now().isoformat(),
        "feedback": feedback.get("raw_feedback"),
        "feedback_summary": feedback.get("feedback_summary"),
    }]
    updated["updated_at"] = datetime.now().isoformat()
    return updated


# ---------------------------------------------------------------------------
# Feedback agent (LLM)
# ---------------------------------------------------------------------------


FEEDBACK_PROMPT = """
You are the Feedback Agent for an activity recommendation system.

Convert explicit user feedback into structured memory updates. Use the
previous recommendation list only to resolve references like "the first
one", "that movie place", or "none of these". Store only what the user
actually says; do not invent permanent preferences.

Return ONLY valid JSON with exactly these fields:
{
    "liked_activities": [],
    "disliked_activities": [],
    "liked_places": [],
    "rejected_places": [],
    "positive_preferences": [],
    "negative_preferences": [],
    "budget_preference": null,
    "max_travel_minutes": null,
    "feedback_summary": ""
}

Rules:
- liked/disliked_activities: activity concepts explicitly liked or rejected.
- liked/rejected_places: specific venues explicitly liked or rejected
  (copy names exactly from the previous recommendations).
- positive_preferences: characteristics wanted more of (e.g. "active", "quiet").
- negative_preferences: characteristics to avoid (e.g. "crowded", "movies").
- budget_preference: "free", "low", "medium", "high", or null.
- max_travel_minutes: integer only when the user gives a travel-time limit.
- If the user rejects everything without a reason, reject the listed places
  but do not invent new characteristics.
- feedback_summary: one sentence on what should change next time.
"""


def feedback_agent(user_feedback, previous_recommendations=None):
    start = time.time()
    context = [
        {"place_name": r.get("place_name"), "source_activity": r.get("source_activity"), "final_rank": r.get("final_rank")}
        for r in (previous_recommendations or [])
    ]
    try:
        parsed, usage = llm.call_llm_json(
            FEEDBACK_PROMPT,
            f"PREVIOUS RECOMMENDATIONS:\n{json.dumps(context, indent=2)}\n\nUSER FEEDBACK:\n{user_feedback}",
        )
        parsed["raw_feedback"] = user_feedback
        parsed["runtime_seconds"] = time.time() - start
        return make_agent_response("feedback", data=parsed, usage=usage)
    except Exception as exc:
        return failed("feedback", exc)


# ---------------------------------------------------------------------------
# Using memory in a search
# ---------------------------------------------------------------------------


def build_memory_context(memory):
    if not memory:
        return ""
    labels = {
        "liked_activities": "Previously liked activities",
        "disliked_activities": "Previously disliked activities",
        "liked_places": "Previously liked venues",
        "rejected_places": "Previously rejected venues",
        "positive_preferences": "Preference signals to favor",
        "negative_preferences": "Preference signals to avoid",
    }
    lines = [f"{label}: {', '.join(memory[field])}" for field, label in labels.items() if memory.get(field)]
    if memory.get("budget_preference"):
        lines.append(f"Remembered budget preference: {memory['budget_preference']}")
    if memory.get("max_travel_minutes") is not None:
        lines.append(f"Remembered maximum travel time: {memory['max_travel_minutes']} minutes")
    return "\n".join(lines)


def filter_rejected_places(places_df, memory):
    rejected = {str(p).strip().lower() for p in (memory or {}).get("rejected_places", [])}
    if places_df is None or places_df.empty or not rejected:
        return places_df, 0
    mask = places_df["place_name"].astype(str).str.strip().str.lower().isin(rejected)
    return places_df[~mask].reset_index(drop=True), int(mask.sum())
