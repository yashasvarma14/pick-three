"""
Feedback and memory.

In the notebook, memory was a JSON file in the shared Drive folder. In
the public app each visitor's memory lives in their own browser session,
so strangers never share or overwrite each other's preferences. The
memory dict has the same structure in both versions.
"""

import copy
import json
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd

from .base import failed, make_agent_response
from .llm import call_llm_json

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
        parsed, usage = call_llm_json(
            FEEDBACK_PROMPT,
            f"PREVIOUS RECOMMENDATIONS:\n{json.dumps(context, indent=2)}\n\nUSER FEEDBACK:\n{user_feedback}",
        )
        parsed["raw_feedback"] = user_feedback
        parsed["runtime_seconds"] = time.time() - start
        return make_agent_response("feedback", data=parsed, usage=usage)
    except Exception as exc:
        return failed("feedback", exc)


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


GENERIC_WORDS = {"watch", "watching", "with", "friends", "friend", "going", "doing", "something",
                 "activity", "activities", "place", "places", "option", "options", "theater",
                 "theaters", "high", "cost", "environment", "environments"}


def _words(text, min_len=4):
    return [w for w in re.findall(r"[a-zA-Z]+", str(text).lower()) if len(w) >= min_len]


def apply_memory_activity_filter(embedder, candidates_df, memory, current_user_prompt,
                                 positive_weight=0.30, negative_weight=0.10, min_candidates_to_keep=6):
    """
    1. Hard-filter only activities that directly name a disliked concept.
    2. Boost activities similar to liked items; softly penalize similar-to-disliked.
    3. Current request overrides memory when it mentions the same concept.
    4. Re-number activity_rank so downstream agents see the new order.
    """
    df = candidates_df.copy()
    for col, val in [("memory_positive_similarity", 0.0), ("memory_negative_similarity", 0.0),
                     ("memory_adjustment", 0.0), ("memory_keyword_conflict", False)]:
        df[col] = val
    if df.empty or not memory:
        return df

    prompt_words = set(_words(current_user_prompt))

    def overridden(item):
        # Whole-word comparison (the notebook's substring test let
        # "interactive" override a remembered "active").
        return bool(set(_words(item)) & prompt_words)

    positive = [str(i).strip() for i in memory.get("liked_activities", []) + memory.get("positive_preferences", []) if str(i).strip()]
    negative = [str(i).strip() for i in memory.get("disliked_activities", []) + memory.get("negative_preferences", []) if str(i).strip()]
    positive = [i for i in positive if not overridden(i)]
    negative = [i for i in negative if not overridden(i)]

    hard_terms = {w.rstrip("s") for item in memory.get("disliked_activities", []) if not overridden(item)
                  for w in _words(item) if w not in GENERIC_WORDS}
    hard_terms |= {w.rstrip("s") for item in memory.get("negative_preferences", []) if not overridden(item)
                   for w in _words(item) if w not in GENERIC_WORDS and len(_words(item)) == 1}

    df["memory_keyword_conflict"] = df["activity"].apply(lambda a: bool({w.rstrip("s") for w in _words(a)} & hard_terms))

    activity_emb = np.asarray(embedder.encode(df["activity"].astype(str).tolist()))
    if positive:
        df["memory_positive_similarity"] = (activity_emb @ np.asarray(embedder.encode(positive)).T).max(axis=1)
    if negative:
        df["memory_negative_similarity"] = (activity_emb @ np.asarray(embedder.encode(negative)).T).max(axis=1)

    df["memory_adjustment"] = positive_weight * df["memory_positive_similarity"] - negative_weight * df["memory_negative_similarity"]
    df["pre_memory_discovery_score"] = df["discovery_score"]
    df["discovery_score"] = (df["discovery_score"] + df["memory_adjustment"]).clip(0.0, 1.0)

    kept = df[~df["memory_keyword_conflict"]].copy()
    if len(kept) < min_candidates_to_keep:
        kept = df.sort_values(["memory_keyword_conflict", "discovery_score"], ascending=[True, False]).head(
            min(min_candidates_to_keep, len(df))).copy()

    kept = kept.sort_values("discovery_score", ascending=False).reset_index(drop=True)
    kept["activity_rank"] = range(1, len(kept) + 1)
    return kept


def filter_rejected_places(places_df, memory):
    rejected = {str(p).strip().lower() for p in (memory or {}).get("rejected_places", [])}
    if places_df is None or places_df.empty or not rejected:
        return places_df, 0
    mask = places_df["place_name"].astype(str).str.strip().str.lower().isin(rejected)
    return places_df[~mask].reset_index(drop=True), int(mask.sum())
