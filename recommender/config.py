"""
Configuration and secrets.

Secrets are read from Streamlit secrets (`.streamlit/secrets.toml` locally,
or the Secrets panel on Streamlit Community Cloud), falling back to
environment variables. Nothing secret is ever hard-coded in the repo.
"""

import os


def get_setting(name, default=None):
    """Read a setting from st.secrets, then the environment, then a default."""
    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        # No secrets file, or running outside Streamlit (tests, scripts).
        pass

    value = os.environ.get(name)
    return value if value not in (None, "") else default


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
    """USD per 1M tokens. Check these against your provider's current pricing."""
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


# Recommendation defaults (mirrors PROJECT_CONFIG in the notebook)
DEFAULTS = {
    "top_n_activities": 10,
    "activities_to_search": 3,
    "places_per_activity": 2,
    "max_places_for_mobility": 5,
    "event_max_results": 20,
    "rag_max_places": 5,
    "rag_top_k_chunks": 10,
    "place_relevance_threshold": 0.60,
    "max_agent_calls": 10,
    "max_adaptation_retries": 1,
    "final_recommendations": 3,
    # Photo Vibe Agent: "auto" follows the planner, "always" or "off".
    "photo_vibe": "auto",
    "photo_max_places": 5,
    "photos_per_place": 3,
}

# Scoring weights, kept in one place so experiments can change them.
DISCOVERY_WEIGHTS = {"semantic": 0.70, "group": 0.15, "budget": 0.10, "category": 0.05}
# Relative weights: whatever signals a venue has are re-normalized to sum
# to 1, so without photos the first four keep their 40/25/15/20 split.
MATCHING_WEIGHTS = {"activity": 0.40, "place": 0.25, "mobility": 0.15, "evidence": 0.20, "vibe": 0.15}
