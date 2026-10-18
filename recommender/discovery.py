"""
Discovery Agent: rank the 196 activity concepts for a request.

Score = 0.70 semantic similarity + 0.15 group fit + 0.10 budget fit
        + 0.05 category fit
"""

import time

import numpy as np

from .base import failed, make_agent_response
from .config import DISCOVERY_WEIGHTS

BUDGET_LEVELS = {"free": 0.00, "low": 0.20, "medium": 0.50, "high": 1.00}


def participant_fit(activity_participants, requested_group_size):
    """Soft fit for the dataset's suggested participant count (not a hard limit)."""
    if requested_group_size is None:
        return 1.0
    difference = abs(activity_participants - requested_group_size)
    if difference == 0:
        return 1.0
    if difference == 1:
        return 0.85
    if difference <= 3:
        return 0.65
    return 0.40


def budget_fit(activity_price, budget_level):
    """Compare the dataset's relative price (0-1) with a broad budget level."""
    if budget_level is None or str(budget_level).lower() not in BUDGET_LEVELS:
        return 1.0
    max_price = BUDGET_LEVELS[str(budget_level).lower()]
    if activity_price <= max_price:
        return 1.0
    return max(0.20, 1.0 - (activity_price - max_price))


def category_fit(activity_type, preferred_types):
    if not preferred_types:
        return 1.0
    if isinstance(preferred_types, str):
        preferred_types = [preferred_types]
    return 1.0 if str(activity_type).lower() in {str(t).lower() for t in preferred_types} else 0.75


def discovery_agent(resources, user_query, group_size=None, budget_level=None, preferred_types=None, top_n=10):
    start = time.time()
    try:
        query_embedding = resources.embedder.encode([user_query])
        semantic = np.asarray(query_embedding) @ np.asarray(resources.activity_embeddings).T
        semantic = semantic[0]

        results = resources.activities.copy()
        results["semantic_similarity"] = semantic
        results["participant_fit"] = results["participants"].apply(lambda x: participant_fit(x, group_size))
        results["budget_fit"] = results["price"].apply(lambda x: budget_fit(x, budget_level))
        results["category_fit"] = results["type"].apply(lambda x: category_fit(x, preferred_types))

        w = DISCOVERY_WEIGHTS
        results["discovery_score"] = (
            w["semantic"] * results["semantic_similarity"]
            + w["group"] * results["participant_fit"]
            + w["budget"] * results["budget_fit"]
            + w["category"] * results["category_fit"]
        )

        # The app finds venues, so at-home activities ("Take a nap",
        # "Organize your pantry") are never candidates.
        if "place_query" in results.columns and results["place_query"].notna().any():
            results = results[results["place_query"].notna()]

        results = results.sort_values("discovery_score", ascending=False).reset_index(drop=True)
        results.insert(0, "discovery_rank", range(1, len(results) + 1))
        top = results.head(top_n).copy()

        return make_agent_response(
            "discovery",
            data={"results": top, "runtime_seconds": time.time() - start},
            # Diagnostic proxy, not a calibrated probability.
            confidence=float(top["semantic_similarity"].head(3).mean()),
            sources=["Activity catalog"],
        )
    except Exception as exc:
        return failed("discovery", exc)
