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


def openai_api_key():
    return get_setting("OPENAI_API_KEY")


def google_maps_api_key():
    return get_setting("GOOGLE_MAPS_API_KEY")


def ticketmaster_api_key():
    return get_setting("TICKETMASTER_API_KEY")


def llm_model():
    return get_setting("OPENAI_MODEL", "gpt-6-luna")


def model_pricing():
    """USD per 1M tokens. Check these against your provider's current pricing."""
    return {
        "input_per_1m": float(get_setting("MODEL_INPUT_PRICE_PER_1M", 0.10)),
        "output_per_1m": float(get_setting("MODEL_OUTPUT_PRICE_PER_1M", 0.50)),
    }


def api_status():
    """Which external services are configured right now."""
    return {
        "openai": bool(openai_api_key()),
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
    "max_agent_calls": 9,
    "max_adaptation_retries": 1,
    "final_recommendations": 3,
}

# Scoring weights, kept in one place so experiments can change them.
DISCOVERY_WEIGHTS = {"semantic": 0.70, "group": 0.15, "budget": 0.10, "category": 0.05}
MATCHING_WEIGHTS = {"activity": 0.40, "place": 0.25, "mobility": 0.15, "evidence": 0.20}
