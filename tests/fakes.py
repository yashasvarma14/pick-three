"""
Fake versions of every outside service (Google, Open-Meteo, Ticketmaster,
venue websites, the LLM) so the tests run offline, for free, and always
give the same answers. conftest.py plugs them in for every test.
"""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).parent / "fixtures" / "activity_sample.json"


# ---------------------------------------------------------------------------
# Embeddings and HTTP responses
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Weather, events and venue websites (GET)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Google Maps around downtown Los Angeles
# ---------------------------------------------------------------------------


HOME = (34.0522, -118.2437)  # downtown Los Angeles


EVERY_DAY_5PM_TO_2AM = [{"open": {"day": d, "hour": 17, "minute": 0},
                         "close": {"day": (d + 1) % 7, "hour": 2, "minute": 0}} for d in range(7)]


MORNINGS_ONLY = [{"open": {"day": d, "hour": 7, "minute": 0}, "close": {"day": d, "hour": 11, "minute": 0}} for d in range(7)]


CALLS = {"search": [], "details": 0}


def nearby_post(url, headers=None, json=None, timeout=None):
    if "computeRoutes" in url:  # ~2 minutes per km from HOME, plus 4 minutes
        from recommender.core.helpers import distance_km
        o = json["origin"]["location"]["latLng"]
        d = json["destination"]["location"]["latLng"]
        minutes = 4 + 2 * distance_km(o["latitude"], o["longitude"], d["latitude"], d["longitude"])
        return FakeResponse({"routes": [{"distanceMeters": int(minutes * 500), "duration": f"{int(minutes * 60)}s",
                                         "staticDuration": f"{int(minutes * 50)}s"}]})
    if "places:searchText" not in url:
        raise AssertionError(f"Unexpected POST {url}")
    if json.get("pageSize") == 1:  # typed address lookup
        return FakeResponse({"places": [{"displayName": {"text": "Downtown LA"}, "formattedAddress": "Los Angeles, CA",
                                         "location": {"latitude": HOME[0], "longitude": HOME[1]}}]})
    CALLS["search"].append(json)
    q = json["textQuery"]
    slug = q.replace(" ", "-")
    places = []
    for i in range(6):
        places.append({
            "id": f"{slug}-{i}", "displayName": {"text": f"{q.title()} {i}"}, "formattedAddress": f"{i} Main St, Los Angeles",
            "location": {"latitude": HOME[0] + 0.01 * i, "longitude": HOME[1]},
            "rating": 4.0 + i / 20, "userRatingCount": 50 * (i + 1), "types": ["restaurant"],
            "primaryTypeDisplayName": {"text": "Restaurant"}, "businessStatus": "OPERATIONAL",
            "googleMapsUri": f"https://maps.example/{slug}{i}", "websiteUri": f"https://{slug}{i}.example",
            "regularOpeningHours": {"periods": MORNINGS_ONLY if i == 0 else EVERY_DAY_5PM_TO_2AM},
        })
    places.append({  # Nap York problem: a match on the other side of the country
        "id": f"{slug}-far", "displayName": {"text": "Far Away Place"}, "location": {"latitude": 40.77, "longitude": -73.97},
        "types": ["restaurant"], "businessStatus": "OPERATIONAL", "rating": 5.0, "userRatingCount": 9999})
    return FakeResponse({"places": places})


def nearby_get(url, params=None, headers=None, timeout=None):
    if "nominatim.openstreetmap.org" in url:
        CALLS.setdefault("osm", 0)
        CALLS["osm"] = CALLS.get("osm", 0) + 1
        return FakeResponse([{"lat": str(HOME[0]), "lon": str(HOME[1]), "display_name": "Los Angeles, CA", "name": "Los Angeles"}])
    if "places.googleapis.com/v1/places/" in url:
        CALLS["details"] += 1
        return FakeResponse({"editorialSummary": {"text": "Casual spot known for smash burgers."},
                             "reviews": [{"rating": 5, "relativePublishTimeDescription": "a month ago",
                                          "text": {"text": "Great cheap burgers and quick service, perfect for a weeknight."}}]})
    return fake_get(url, params, headers, timeout)


# ---------------------------------------------------------------------------
# The LLM: a canned answer for each agent's prompt
# ---------------------------------------------------------------------------


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


def nearby_llm(system_prompt, user_message, model=None):
    parsed, usage = fake_llm(system_prompt, user_message, model)
    if "Preference Agent" in system_prompt:
        parsed["search_phrases"] = ["smash burger", "burger joint"] if "burger" in user_message.lower() else ["quiet wine bar", "day spa", "botanical garden"]
    return parsed, usage


def use_llm(monkeypatch, fake):
    """Swap in a different fake LLM for one test."""
    monkeypatch.setattr("recommender.core.llm.call_llm_json", fake)
