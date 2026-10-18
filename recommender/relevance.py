"""
Place-Activity Relevance Agent: drops venues that are only textually
related to the activity (e.g. "Think of a new business idea" ->
"Austin Economic Development").
"""

import json
import time

import numpy as np
import pandas as pd

from .base import failed, make_agent_response, skipped
from .llm import call_llm_json

SYSTEM_PROMPT = """
You are a venue relevance validator for an activity recommendation system.

Your job is NOT to recommend places. Only decide whether each returned place
is a plausible real-world place for performing the associated activity.

Examples:
- Activity "Go to an escape room", place "The Escape Game Austin": highly relevant.
- Activity "Think of a new business idea", place "Austin Economic Development":
  not an activity venue, even though the concepts are related.
- Activity "Go bowling", place "Highland Lanes": highly relevant.
- Activity "Have a picnic", place "Zilker Metropolitan Park": relevant.

Consider the source activity, the place name, the Google place types, and
whether a normal person could realistically do the activity there. Do not
reward a place just because its name shares words with the activity.

Return ONLY valid JSON:
{
    "assessments": [
        {"row_id": 0, "place_activity_relevance_score": 0.0, "relevance_reason": ""}
    ]
}

Scoring: 0.90-1.00 clearly designed for the activity; 0.75-0.89 very plausible;
0.60-0.74 reasonably plausible; 0.40-0.59 questionable; 0.00-0.39 not realistic.

Every input row must receive exactly one assessment. Keep reasons short.
"""


def place_activity_relevance_agent(places_df, relevance_threshold=0.60, user_request=None):
    start = time.time()
    if places_df is None or places_df.empty:
        return skipped("place_relevance", "No candidate places to validate.")

    try:
        df = places_df.drop(
            columns=[c for c in ["row_id", "place_activity_relevance_score", "relevance_reason"] if c in places_df.columns]
        ).reset_index(drop=True)
        df["row_id"] = df.index.astype(int)

        records = [
            {
                "row_id": int(row["row_id"]),
                "source_activity": row.get("source_activity"),
                "place_name": row.get("place_name"),
                "place_types": row.get("place_types") if isinstance(row.get("place_types"), list) else [],
                "address": row.get("address"),
            }
            for _, row in df.iterrows()
        ]

        header = "Evaluate these place/activity pairs:"
        if user_request:
            # Judge against what the person actually asked for, not only the
            # search phrase that found the place ("affordable group activities").
            request = " ".join(str(user_request).split())
            header = (f"USER REQUEST (score low if the place clearly doesn't suit it, e.g. wrong kind of "
                      f"place, a business office, a store when they want an activity): {request} | " + header)
        parsed, usage = call_llm_json(SYSTEM_PROMPT, header + "\n" + json.dumps(records, indent=2))
        assessments = pd.DataFrame(parsed.get("assessments", []))

        if not assessments.empty and "row_id" in assessments.columns:
            # The model sometimes returns row_id as a string; a type mismatch
            # made the notebook's merge silently match nothing.
            assessments["row_id"] = pd.to_numeric(assessments["row_id"], errors="coerce")
            assessments = assessments.dropna(subset=["row_id"]).astype({"row_id": int}).drop_duplicates("row_id")
            keep = [c for c in ["row_id", "place_activity_relevance_score", "relevance_reason"] if c in assessments.columns]
            df = df.merge(assessments[keep], on="row_id", how="left")
        else:
            df["place_activity_relevance_score"] = np.nan
            df["relevance_reason"] = "No relevance assessment returned."

        df["place_activity_relevance_score"] = (
            pd.to_numeric(df.get("place_activity_relevance_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        )
        df["place_activity_relevant"] = df["place_activity_relevance_score"] >= relevance_threshold

        filtered = df[df["place_activity_relevant"]].drop(columns=["row_id"]).reset_index(drop=True)

        return make_agent_response(
            "place_relevance",
            data={
                "results": filtered,
                "all_assessments": df.drop(columns=["row_id"]),
                "places_before_validation": len(df),
                "places_after_validation": len(filtered),
                "places_removed": len(df) - len(filtered),
                "runtime_seconds": time.time() - start,
            },
            sources=["Google Places structured metadata"],
            usage=usage,
        )
    except Exception as exc:
        return failed("place_relevance", exc)
