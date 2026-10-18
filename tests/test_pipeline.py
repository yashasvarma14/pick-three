"""
Offline end-to-end test: every external service is faked, so this runs
without API keys or network. Run with:  python -m pytest -q
"""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import recommender.llm as llm
from recommender import build_resources, create_empty_user_memory, run_adaptive_workflow
from recommender.base import contains_keyword
from recommender.events import build_event_window_utc
from recommender.matching import calculate_final_recommendation_score
from recommender.places import build_place_query

FIXTURE = Path(__file__).parent / "fixtures" / "activity_sample.json"


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (no model download)."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            v = np.zeros(64)
            for word in str(text).lower().split():
                v[int(hashlib.md5(word.strip(".,").encode()).hexdigest(), 16) % 64] += 1
            n = np.linalg.norm(v)
            vectors.append(v / n if n else v)
        return np.array(vectors)


class FakeResponse:
    def __init__(self, payload, status=200, content_type="application/json", text=None):
        self._payload, self.status_code = payload, status
        self.headers = {"Content-Type": content_type}
        self.text = text or json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def fake_get(url, params=None, headers=None, timeout=None):
    if "geocoding-api" in url:
        return FakeResponse({"results": [{"name": "Austin", "latitude": 30.27, "longitude": -97.74,
                                          "country": "United States", "admin1": "Texas", "timezone": "America/Chicago"}]})
    if "api.open-meteo.com" in url:
        start = datetime.now().replace(minute=0, second=0, microsecond=0)
        times = [(start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(48)]
        return FakeResponse({
            "timezone": "America/Chicago",
            "current": {"time": times[0], "temperature_2m": 72, "apparent_temperature": 72, "precipitation": 0,
                        "weather_code": 1, "wind_speed_10m": 6},
            "hourly": {"time": times, "temperature_2m": [72] * 48, "apparent_temperature": [72] * 48,
                       "precipitation_probability": [70] * 48, "precipitation": [0.1] * 48,
                       "weather_code": [63] * 48, "wind_speed_10m": [8] * 48},
        })
    if "ticketmaster" in url:
        return FakeResponse({"_embedded": {"events": [{
            "id": "E1", "name": "Indie Night", "url": "https://tm.example/e1",
            "dates": {"start": {"localDate": "2026-10-12", "localTime": "20:00:00"}, "status": {"code": "onsale"}},
            "classifications": [{"segment": {"name": "Music"}, "genre": {"name": "Rock"}}],
            "_embedded": {"venues": [{"name": "Mohawk", "city": {"name": "Austin"}, "address": {"line1": "912 Red River"},
                                      "location": {"latitude": "30.27", "longitude": "-97.73"}}]},
        }]}})
    # Venue websites
    return FakeResponse({}, content_type="text/html",
                        text="<html><body>" + "Great group games, karaoke rooms and bowling lanes for friends. " * 40 + "</body></html>")


PLACE_COUNTER = {"n": 0}


def fake_post(url, headers=None, json=None, timeout=None):
    if "places:searchText" in url:
        q = json["textQuery"]
        places = []
        for i in range(json["maxResultCount"]):
            PLACE_COUNTER["n"] += 1
            n = PLACE_COUNTER["n"]
            places.append({"id": f"P{n}", "displayName": {"text": f"{q.split(' in ')[0].title()} #{i + 1}"},
                           "formattedAddress": f"{n} Main St, Austin, TX", "location": {"latitude": 30.2 + n / 100, "longitude": -97.7},
                           "rating": 4.0 + (n % 10) / 10, "userRatingCount": 100 * n, "types": ["point_of_interest"],
                           "googleMapsUri": f"https://maps.example/{n}", "websiteUri": f"https://venue{n}.example"})
        return FakeResponse({"places": places})
    if "computeRoutes" in url:
        lat = json["destination"]["location"]["latLng"]["latitude"]
        minutes = int((lat - 30.2) * 300) + 5
        return FakeResponse({"routes": [{"distanceMeters": minutes * 900, "duration": f"{minutes * 60}s",
                                         "staticDuration": f"{minutes * 50}s"}]})
    raise AssertionError(f"Unexpected POST {url}")


def fake_llm(system_prompt, user_message, model=None):
    usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "cost": 0.00004}
    if "Planner / Orchestrator" in system_prompt:
        return {"goal_summary": "Fun group activity tonight",
                "agents_to_run": ["preference", ["discovery", "weather"], "local_places", "events", "mobility", "venue_rag", "matching", "bogus"],
                "human_checkpoint_needed": False, "planning_reasoning_summary": ["test"], "finish_conditions": []}, usage
    if "Preference Agent" in system_prompt:
        terms = ["cheap burger restaurant"] if "burger" in user_message.lower() else []
        return {"venue_search_terms": terms, "city": "Austin", "group_size": 4, "budget_level": "low", "preferred_types": ["social", "notatype"],
                "positive_preferences": ["interactive"], "negative_preferences": [], "hard_constraints": [],
                "soft_preferences": [], "needs_clarification": False, "clarification_question": None}, usage
    if "venue relevance validator" in system_prompt:
        rows = json_loads_tail(user_message)
        # Return row_id as strings to exercise the type fix; reject the first row.
        return {"assessments": [{"row_id": str(r["row_id"]), "place_activity_relevance_score": 0.2 if r["row_id"] == 0 else 0.9,
                                 "relevance_reason": "ok"} for r in rows]}, usage
    if "Venue Evidence RAG Agent" in system_prompt:
        ids = [line.split("place_id: ")[1].split(" |")[0] for line in user_message.splitlines() if line.startswith("- place_id: ")]
        return {"venue_assessments": [{"place_id": pid, "place_name": "x", "evidence_fit_score": 0.8,
                                       "supported_strengths": ["groups"], "supported_concerns": [],
                                       "evidence_summary": "Site mentions group games.", "evidence_sufficient": True}
                                      for pid in ids]}, usage
    if "final Synthesizer" in system_prompt:
        ctx = json.loads(user_message.split("WORKFLOW CONTEXT:\n", 1)[1])
        return {"overall_summary": "Three good picks.",
                "recommendations": [{"final_rank": r["final_rank"], "place_name": r["place_name"], "source_activity": r["source_activity"],
                                     "why_it_fits": "Fits.", "supporting_evidence": ["Well rated"], "important_caveat": None}
                                    for r in ctx["recommendations"]],
                "personalization_summary": "", "evidence_limitations": []}, usage
    if "Feedback Agent" in system_prompt:
        return {"liked_activities": [], "disliked_activities": ["movies"], "liked_places": [], "rejected_places": [],
                "positive_preferences": ["active"], "negative_preferences": [], "budget_preference": None,
                "max_travel_minutes": 15, "feedback_summary": "No movies."}, usage
    raise AssertionError("Unexpected LLM prompt")


def json_loads_tail(message):
    return json.loads(message.split("\n", 1)[1])


@pytest.fixture
def resources(monkeypatch):
    for module in ["recommender.weather", "recommender.events", "recommender.rag"]:
        monkeypatch.setattr(f"{module}.requests.get", fake_get)
    for module in ["recommender.places", "recommender.mobility"]:
        monkeypatch.setattr(f"{module}.requests.post", fake_post)
    for module in ["recommender.preference", "recommender.relevance", "recommender.rag",
                   "recommender.orchestrator", "recommender.synthesizer", "recommender.memory"]:
        monkeypatch.setattr(f"{module}.call_llm_json", fake_llm)
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test")
    monkeypatch.setenv("TICKETMASTER_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    from recommender.weather import geocode_city
    geocode_city.cache_clear()
    return build_resources(FIXTURE, embedder=FakeEmbedder())


def test_catalog_deduplicates(resources):
    assert resources.activities["activity"].is_unique
    assert len(resources.activities) < len(pd.read_json(FIXTURE, lines=True))


def test_full_adaptive_run(resources):
    steps = []
    state = run_adaptive_workflow(
        resources, "Four of us want something fun and interactive in Austin tonight",
        memory=None, city="Austin, TX", origin="UT Austin",
        target_datetime=datetime.now() + timedelta(hours=3),
        progress=lambda key, msg: steps.append(key),
    )
    assert not state["errors"], state["errors"]
    assert state["final_recommendations"], state["finish_reason"]
    assert state["final_response"]["overall_summary"]
    assert "bogus" not in state["orchestrator_plan"]["execution_order"]
    assert state["orchestrator_plan"]["execution_order"][0] == "preference"
    assert state["confidence"]["place_relevance_removed"] >= 1  # string row_ids were matched
    assert state["events"] and state["events"][0]["event_name"] == "Indie Night"
    assert all(r["evidence_fit_score"] == 0.8 for r in state["final_recommendations"])
    assert state["usage"]["api_calls"]["google_places"] > 0
    assert "synthesizer" in steps


def test_no_origin_leaves_mobility_missing(resources):
    state = run_adaptive_workflow(resources, "Indoor idea for four people in Austin", city="Austin", origin=None)
    assert state["final_recommendations"]
    assert all(pd.isna(r.get("mobility_score")) for r in state["ranked_results"])
    assert "mobility" in state["agents_skipped"]


def test_memory_feedback_round_trip(resources):
    from recommender import feedback_agent, update_memory_from_feedback

    memory = create_empty_user_memory()
    fb = feedback_agent("No more movies, something more active, nothing over 15 minutes away", [])
    memory = update_memory_from_feedback(memory, fb["data"])
    assert memory["disliked_activities"] == ["movies"] and memory["max_travel_minutes"] == 15

    state = run_adaptive_workflow(resources, "Something fun for two in Austin", memory=memory, city="Austin", origin="UT Austin")
    assert not any(contains_keyword(a["activity"], "movie") for a in state["candidate_activities"])
    ranks = [a["activity_rank"] for a in state["candidate_activities"]]
    assert ranks == list(range(1, len(ranks) + 1))


def test_helpers():
    assert build_place_query("Have a barbecue with some friends", "Austin").startswith("barbecue")
    from recommender.outings import outing_query
    assert outing_query("Take a nap") is None
    assert outing_query("Go to an Escape Room") == "escape room"
    assert build_place_query("Go to a karaoke bar with some friends", "Austin") == "karaoke bar in Austin"
    start, end = build_event_window_utc(datetime(2026, 10, 12, 19, 0), "America/Chicago")
    assert start.hour == 21 and end.hour == 6  # 7pm CDT = 00:00 UTC, window -3h/+6h
    assert calculate_final_recommendation_score(0.8, 0.6, None, None) == pytest.approx((0.8 * 0.4 + 0.6 * 0.25) / 0.65, rel=1e-5)


def test_only_outings_are_discovered(resources):
    state = run_adaptive_workflow(resources, "Something relaxing and quiet for two", city="Austin", origin=None)
    activities = {a["activity"] for a in state["candidate_activities"]}
    assert "Read a book" not in activities and "Learn to cook a new recipe" not in activities
    assert all(a.get("place_query") for a in state["candidate_activities"])


def test_specific_request_is_searched_directly(resources, monkeypatch):
    queries = []
    original = fake_post

    def spy(url, headers=None, json=None, timeout=None):
        if "places:searchText" in url:
            queries.append(json["textQuery"])
            assert "locationBias" in json  # searches are centred on the city
        return original(url, headers, json, timeout)

    monkeypatch.setattr("recommender.places.requests.post", spy)
    state = run_adaptive_workflow(resources, "I want burgers. Please find cheap burger options",
                                  city="Austin, TX", origin=None)
    assert queries and all(q.startswith("cheap burger restaurant") for q in queries)
    assert all(r["source_activity"] == "Cheap burger restaurant" for r in state["ranked_results"])
    assert any("searched directly" in line for line in state["adaptation_log"])


def test_far_away_venues_are_dropped(resources, monkeypatch):
    def new_york(url, headers=None, json=None, timeout=None):
        if "places:searchText" in url:
            resp = fake_post(url, headers, json, timeout)
            resp._payload["places"][0]["location"] = {"latitude": 40.77, "longitude": -73.97}  # Manhattan
            return resp
        return fake_post(url, headers, json, timeout)

    monkeypatch.setattr("recommender.places.requests.post", new_york)
    state = run_adaptive_workflow(resources, "Something fun for four", city="Austin", origin=None)
    assert all(abs(p["longitude"] + 97.7) < 1 for p in state["candidate_places"])
    assert any("outside" in line for line in state["adaptation_log"])


def test_drive_limit_is_enforced(resources):
    state = run_adaptive_workflow(resources, "Four of us want something fun", city="Austin",
                                  origin="UT Austin", max_travel_minutes=10)
    minutes = [r["traffic_minutes"] for r in state["ranked_results"]]
    assert minutes and (all(m <= 15 for m in minutes) or len(minutes) <= 3)


def test_ttl_cache_shares_and_skips_errors():
    import threading, time
    from recommender.base import ttl_cache

    calls = {"n": 0}

    @ttl_cache(seconds=60)
    def slow(x):
        calls["n"] += 1
        time.sleep(0.2)
        if x == "bad":
            raise ValueError("boom")
        return {"x": x}

    threads = [threading.Thread(target=slow, args=("a",)) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert calls["n"] == 1  # five simultaneous callers, one real call
    slow("a")["x"] = "changed"
    assert slow("a") == {"x": "a"}  # callers get copies
    for _ in range(2):
        with pytest.raises(ValueError):
            slow("bad")
    assert calls["n"] == 3  # errors are retried, not cached


def test_state_abbreviations_pick_the_right_city(monkeypatch):
    from recommender.weather import geocode_city

    def fake(url, params=None, headers=None, timeout=None):
        return FakeResponse({"results": [
            {"name": "Paris", "latitude": 48.85, "longitude": 2.35, "country": "France", "admin1": "Ile-de-France", "timezone": "Europe/Paris"},
            {"name": "Paris", "latitude": 33.66, "longitude": -95.55, "country": "United States", "admin1": "Texas", "timezone": "America/Chicago"},
        ]})

    monkeypatch.setattr("recommender.weather.requests.get", fake)
    assert geocode_city("Paris, TX")["admin1"] == "Texas"


def test_split_synthesis(resources):
    from recommender.workflow import synthesize_results

    state = run_adaptive_workflow(resources, "Indoor idea for four people", city="Austin", synthesize=False)
    assert state["final_recommendations"] and state["final_response"] is None
    synthesize_results(state)
    assert state["final_response"]["overall_summary"] and "synthesizer" in state["timings"]
