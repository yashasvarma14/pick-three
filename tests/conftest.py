"""
Shared test setup. Every test gets:
  * settings from environment variables only (never your real keys)
  * empty caches (so one test's fake answers never leak into the next)
  * the `resources` fixture: the small catalog fixture, a fake embedder,
    fake keys, and every outside service replaced by the fakes in fakes.py
"""

import os

import pytest

from fakes import CALLS, FIXTURE, FakeEmbedder, nearby_get, nearby_llm, nearby_post
from recommender import build_resources
from recommender.core.cache import clear_all_caches


@pytest.fixture(autouse=True)
def settings_from_env_only(monkeypatch):
    """Ignore the real .streamlit/secrets.toml: tests only ever see fake keys."""
    monkeypatch.setattr("recommender.core.config.get_setting",
                        lambda name, default=None: os.environ.get(name) or default)


@pytest.fixture(autouse=True)
def fresh_caches():
    clear_all_caches()
    yield
    clear_all_caches()


@pytest.fixture
def resources(monkeypatch):
    CALLS["search"].clear()
    CALLS["details"] = 0
    CALLS["osm"] = 0
    monkeypatch.setattr("requests.get", nearby_get)
    monkeypatch.setattr("requests.post", nearby_post)
    monkeypatch.setattr("recommender.core.llm.call_llm_json", nearby_llm)
    for name in ["GOOGLE_MAPS_API_KEY", "TICKETMASTER_API_KEY", "ANTHROPIC_API_KEY"]:
        monkeypatch.setenv(name, "test")
    return build_resources(FIXTURE, embedder=FakeEmbedder())

