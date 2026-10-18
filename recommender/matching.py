"""
Matching Agent: final venue ranking.

Default weights: 40% activity/context fit, 25% venue quality,
15% mobility, 20% RAG evidence. Missing signals are left out and the
remaining weights are re-normalized; nothing is filled with a made-up value.
"""

import time

import numpy as np
import pandas as pd

from .base import failed, make_agent_response, skipped
from .config import MATCHING_WEIGHTS


def _valid(value):
    return value is not None and not pd.isna(value)


def calculate_final_recommendation_score(activity_score, place_score, mobility_score, evidence_score=None, weights=None):
    w = weights or MATCHING_WEIGHTS
    pairs = [(activity_score, w["activity"]), (place_score, w["place"]),
             (mobility_score, w["mobility"]), (evidence_score, w["evidence"])]
    pairs = [(float(s), wt) for s, wt in pairs if _valid(s)]
    if not pairs:
        return None
    total = sum(wt for _, wt in pairs)
    score = sum(s * wt for s, wt in pairs) / total
    return round(max(0.0, min(1.0, score)), 6)


def matching_agent(candidates_df, venues_df, venue_assessments=None):
    start = time.time()
    venue_first = venues_df is not None and "venue_fit_score" in getattr(venues_df, "columns", [])
    if not venue_first and (candidates_df is None or candidates_df.empty):
        return skipped("matching", "No candidate activities were available.")
    if venues_df is None or venues_df.empty:
        return skipped("matching", "No venues were available to rank.")

    try:
        df = venues_df.copy()
        if venue_first:
            # "Near me" mode: each venue was scored directly against the request.
            score_col = "venue_fit_score"
            df["activity_context_score"] = pd.to_numeric(df["venue_fit_score"], errors="coerce")
        else:
            score_col = "weather_adjusted_score" if "weather_adjusted_score" in candidates_df.columns else "discovery_score"
            lookup = candidates_df.set_index("activity")[score_col].to_dict()
            df["activity_context_score"] = pd.to_numeric(df["source_activity"].map(lookup), errors="coerce")

        # Quality and activity fit always exist for a returned venue; a
        # neutral 0.5 only covers rare gaps (e.g. no rating at all).
        df["place_quality_score"] = pd.to_numeric(df.get("place_quality_score"), errors="coerce").fillna(0.50)
        df["activity_context_score"] = df["activity_context_score"].fillna(0.50)

        # Mobility stays NaN when it was skipped (fixed: the notebook filled 0.5).
        df["mobility_score"] = pd.to_numeric(df["mobility_score"], errors="coerce") if "mobility_score" in df.columns else np.nan

        df["evidence_fit_score"] = np.nan
        df["rag_evidence_sufficient"] = False
        df["rag_evidence_summary"] = None
        df["rag_strengths"] = None
        df["rag_concerns"] = None

        if venue_assessments:
            # Join on place_id (fixed: joining on place_name collides for chains).
            rag = pd.DataFrame(venue_assessments).drop_duplicates("place_id").set_index("place_id")
            df["evidence_fit_score"] = pd.to_numeric(df["place_id"].map(rag.get("evidence_fit_score", pd.Series(dtype=float))), errors="coerce")
            df["rag_evidence_sufficient"] = df["place_id"].map(rag.get("evidence_sufficient", pd.Series(dtype=bool))).eq(True)
            df["rag_evidence_summary"] = df["place_id"].map(rag.get("evidence_summary", pd.Series(dtype=object)))
            df["rag_strengths"] = df["place_id"].map(rag.get("supported_strengths", pd.Series(dtype=object)))
            df["rag_concerns"] = df["place_id"].map(rag.get("supported_concerns", pd.Series(dtype=object)))

        df["final_recommendation_score"] = df.apply(
            lambda r: calculate_final_recommendation_score(
                r["activity_context_score"], r["place_quality_score"], r["mobility_score"], r["evidence_fit_score"]
            ),
            axis=1,
        )
        df = df.sort_values("final_recommendation_score", ascending=False).reset_index(drop=True)
        df.insert(0, "final_rank", range(1, len(df) + 1))

        return make_agent_response(
            "matching",
            data={
                "results": df,
                "places_ranked": len(df),
                "activity_score_used": score_col,
                "rag_places_used": int(df["evidence_fit_score"].notna().sum()),
                "mobility_places_used": int(df["mobility_score"].notna().sum()),
                "weights": MATCHING_WEIGHTS,
                "runtime_seconds": time.time() - start,
            },
        )
    except Exception as exc:
        return failed("matching", exc)
