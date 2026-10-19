"""
Settings, API keys and every tunable number in one place.

Keys are read from Streamlit secrets (.streamlit/secrets.toml locally, or
the Secrets panel on Streamlit Community Cloud), then from environment
variables. Nothing secret is ever written in the code.
"""

import os


# ---------------------------------------------------------------------------
# Reading settings and keys
# ---------------------------------------------------------------------------

def get_setting(name, default=None):
    """Read a setting from st.secrets, then the environment, then a default."""
    try:
        import streamlit as st

        if name in st.secrets:
            value = st.secrets[name]
            # The template's placeholders ("...", "sk-ant-...") count as not set.
            if not (isinstance(value, str) and value.endswith("...")):
                return value
    except Exception:
        # No secrets file, or running outside Streamlit (tests, scripts).
        pass

    value = os.environ.get(name)
    if value in (None, "") or value.endswith("..."):
        return default
    return value


def anthropic_api_key():
    return get_setting("ANTHROPIC_API_KEY")


def google_maps_api_key():
    return get_setting("GOOGLE_MAPS_API_KEY")


def ticketmaster_api_key():
    return get_setting("TICKETMASTER_API_KEY")


def llm_model():
    return get_setting("LLM_MODEL", "claude-haiku-5-5")


def llm_effort():
    """low | medium | high | xhigh | max. Low suits these short JSON tasks."""
    return get_setting("LLM_EFFORT", "low")


def model_pricing():
    """USD per 1M tokens, only used for the cost estimate shown in the app."""
    return {
        # Claude Haiku 5.5 list prices (prompts up to 100K tokens).
        "input_per_1m": float(get_setting("MODEL_INPUT_PRICE_PER_1M", 0.10)),
        "output_per_1m": float(get_setting("MODEL_OUTPUT_PRICE_PER_1M", 0.50)),
    }


def api_status():
    """Which external services are configured right now."""
    return {
        "llm": bool(anthropic_api_key()),
        "google_maps": bool(google_maps_api_key()),
        "ticketmaster": bool(ticketmaster_api_key()),
        "weather": True,  # Open-Meteo needs no key
    }


# ---------------------------------------------------------------------------
# Pipeline settings (any of these can be overridden per run:
# run_pipeline(..., max_searches=2))
# ---------------------------------------------------------------------------

DEFAULTS = {
    # Planner
    "max_agent_calls": 10,
    # Searching
    "top_n_activities": 10,        # catalog ideas Discovery ranks
    "catalog_ideas": 1,            # catalog ideas added as searches for open-ended requests
    "max_searches": 3,             # Google searches per run (the priciest call: keep it low)
    "search_pages": 1,             # 20 places per page; "Search deeper" uses 3
    # Shortlist + venue check
    "shortlist_for_relevance": 12,
    "place_relevance_threshold": 0.60,
    "min_usable_places": 6,        # fewer than this after the check -> check the next batch
    "relevance_retries": 1,
    # Finalists
    "shortlist_for_travel": 8,     # drive times for this many
    "reviews_for_top": 5,          # Google reviews + photo references for this many (= what the RAG reads)
    "rag_max_places": 5,
    "rag_top_k_chunks": 10,
    "photo_vibe": "auto",          # "auto" follows the planner, or "always" / "off"
    "photo_max_places": 5,
    "photos_per_place": 2,         # each photo the vision model looks at is a billed download
    # Results
    "final_recommendations": 3,
    "strong_match_score": 0.75,    # best pick's venue-check score below this -> "no strong match" note
    "event_max_results": 20,
}

# Scoring weights, kept here so experiments only change one file.
DISCOVERY_WEIGHTS = {"semantic": 0.70, "group": 0.15, "budget": 0.10, "category": 0.05}
# Relative weights: whatever signals a venue has are re-normalized to sum
# to 1, so without photos the first four keep their 40/25/15/20 split.
MATCHING_WEIGHTS = {"activity": 0.40, "place": 0.25, "mobility": 0.15, "evidence": 0.20, "vibe": 0.15}
