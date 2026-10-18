"""
End-to-end tests: whole searches through run_pipeline(), with every
outside service faked (see fakes.py). No API keys or network needed.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from fakes import CALLS, HOME, FakeResponse, nearby_get, nearby_llm, nearby_post, use_llm
from recommender import (create_empty_user_memory, feedback_agent, resolve_location, run_pipeline,
                         synthesize_results, update_memory_from_feedback)


def test_resolve_location_from_gps_and_text(resources):
    gps = resolve_location(latitude=HOME[0], longitude=HOME[1])
    assert gps["origin"].startswith("34.05") and gps["timezone"]
    typed = resolve_location("downtown LA")
    assert abs(typed["latitude"] - HOME[0]) < 1e-6
    assert CALLS.get("osm") == 1  # typed address used the free geocoder...
    with pytest.raises(LookupError):
        resolve_location()


def test_near_me_finds_and_keeps_every_place(resources):
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_pipeline(resources, "I want cheap burgers", loc,
                                target_datetime=datetime(2026, 10, 16, 20, 0), max_travel_minutes=20)
    assert not state["errors"], state["errors"]
    # Every search is boxed around the visitor.
    assert CALLS["search"] and all("locationRestriction" in body for body in CALLS["search"])
    assert all("burger" in body["textQuery"] for body in CALLS["search"])
    places = pd.DataFrame(state["all_places"])
    assert len(places) >= 10  # every place found is kept for the map/list
    assert "Far Away Place" not in set(places["place_name"])  # 2,400 miles away: dropped
    assert places["top_pick"].notna().sum() == len(state["final_recommendations"]) == 3
    # The morning-only places are closed at 8 PM, so none is a top pick.
    assert not any(r["place_id"].endswith("-0") for r in state["final_recommendations"])
    # Reviews were read for the shortlist only.
    assert 0 < CALLS["details"] <= 12
    evidence = pd.DataFrame(state["review_evidence"]["retrieved_evidence"])
    assert evidence["website_uri"].str.startswith("Google review").any()
    assert state["final_response"]["overall_summary"]
    # Drive times came from the visitor's coordinates and respect the limit.
    assert all(r["traffic_minutes"] <= 25 for r in state["final_recommendations"])


def test_ai_searches_come_first_and_catalog_only_fills_gaps(resources, monkeypatch):
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_pipeline(resources, "Something relaxing and quiet for two", loc, synthesize=False)
    # The AI gave 3 searches: all used, no generic catalog idea squeezed in, no misleading note.
    assert state["search_phrases"] == ["quiet wine bar", "day spa", "botanical garden"]
    assert not any("not into" in line for line in state["adaptation_log"])

    def one_idea(system_prompt, user_message, model=None):
        parsed, usage = nearby_llm(system_prompt, user_message, model)
        if "Preference Agent" in system_prompt:
            parsed = dict(parsed, search_phrases=["quiet wine bar"])
        return parsed, usage

    use_llm(monkeypatch, one_idea)
    state = run_pipeline(resources, "Something relaxing and quiet for two", loc, synthesize=False)
    assert state["search_phrases"][0] == "quiet wine bar" and len(state["search_phrases"]) == 3  # catalog filled the gap


def test_memory_skips_disliked_searches(resources):
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    memory = {"user_id": "v", "disliked_activities": ["spa"], "liked_activities": [], "liked_places": [],
              "rejected_places": [], "positive_preferences": [], "negative_preferences": []}
    state = run_pipeline(resources, "Something relaxing and quiet for two", loc, memory=memory, synthesize=False)
    assert "day spa" not in state["search_phrases"]


def test_tight_drive_limit_removes_far_places(resources):
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_pipeline(resources, "I want cheap burgers", loc, max_travel_minutes=5,
                                target_datetime=datetime(2026, 10, 16, 20, 0), synthesize=False)
    assert state["final_recommendations"]
    assert all(r["traffic_minutes"] <= 10 for r in state["final_recommendations"])


def test_no_missing_numbers_reach_the_app(resources, monkeypatch):
    """Regression: a top pick without review/website evidence crashed the card (NaN summary)."""
    def no_evidence(system_prompt, user_message, model=None):
        parsed, usage = nearby_llm(system_prompt, user_message, model)
        if "Venue Evidence RAG Agent" in system_prompt:
            parsed = {"venue_assessments": []}
        return parsed, usage

    use_llm(monkeypatch, no_evidence)
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_pipeline(resources, "I want cheap burgers", loc, synthesize=False)
    assert state["final_recommendations"]
    for rows in [state["final_recommendations"], state["ranked_results"], state["all_places"]]:
        for row in rows:
            assert not any(isinstance(v, float) and v != v for v in row.values())


def _rate_limited(fail_query=None, fail_times=99):
    """Fake Google that answers 429 for one search (or all searches)."""
    seen = {}

    def post(url, headers=None, json=None, timeout=None):
        q = (json or {}).get("textQuery", "")
        if "places:searchText" in url and json.get("pageSize") != 1 and (fail_query is None or q == fail_query):
            seen[q] = seen.get(q, 0) + 1
            if seen[q] <= fail_times:
                resp = FakeResponse({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                               "message": "Quota exceeded for quota metric 'SearchTextRequest' per minute."}}, status=429)
                return resp
        return nearby_post(url, headers, json, timeout)
    return post, seen


def test_one_rejected_search_does_not_sink_the_run(resources, monkeypatch):
    monkeypatch.setattr("recommender.services.google_places.time.sleep", lambda s: None)
    post, _ = _rate_limited(fail_query="smash burger")
    monkeypatch.setattr("requests.post", post)
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_pipeline(resources, "I want cheap burgers", loc, synthesize=False)
    assert state["final_recommendations"], state["errors"]
    assert any("turned down 1 of" in line for line in state["adaptation_log"])


def test_brief_rate_limit_is_retried(resources, monkeypatch):
    waits = []
    monkeypatch.setattr("recommender.services.google_places.time.sleep", waits.append)
    post, seen = _rate_limited(fail_times=2)
    monkeypatch.setattr("requests.post", post)
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_pipeline(resources, "I want cheap burgers", loc, synthesize=False)
    assert state["final_recommendations"]
    assert not any("turned down" in line for line in state["adaptation_log"])
    assert waits and all(v == 3 for v in seen.values())  # two 429s, then success


def test_all_searches_rejected_explains_why(resources, monkeypatch):
    monkeypatch.setattr("recommender.services.google_places.time.sleep", lambda s: None)
    post, _ = _rate_limited()
    monkeypatch.setattr("requests.post", post)
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_pipeline(resources, "I want cheap burgers", loc, synthesize=False)
    assert not state["final_recommendations"]
    assert "RESOURCE_EXHAUSTED" in " ".join(state["errors"])


def test_relevance_check_sees_the_request(resources, monkeypatch):
    seen = []

    def spy(system_prompt, user_message, model=None):
        if "venue relevance validator" in system_prompt:
            seen.append(user_message)
        return nearby_llm(system_prompt, user_message, model)

    use_llm(monkeypatch, spy)
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    run_pipeline(resources, "I want cheap burgers", loc, synthesize=False)
    assert seen and "I want cheap burgers" in seen[0].split("\n")[0]


def test_no_duplicate_businesses_in_a_full_run(resources, monkeypatch):
    def chains(url, headers=None, json=None, timeout=None):
        resp = nearby_post(url, headers, json, timeout)
        if "places:searchText" in url and json.get("pageSize") != 1:
            for i, p in enumerate(resp._payload["places"]):
                p["websiteUri"] = "https://www.burgerchain.com/" + str(i)  # every result is one chain
                p["displayName"] = {"text": f"Burger Chain {json['textQuery']} {i}"}
        return resp

    monkeypatch.setattr("requests.post", chains)
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_pipeline(resources, "I want cheap burgers", loc, synthesize=False)
    assert len(state["final_recommendations"]) == 1  # one chain = one pick, never three branches


def _llm_with(planner_question=None, afternoon=False):
    from datetime import datetime as _dt, timedelta as _td

    def llm(system_prompt, user_message, model=None):
        parsed, usage = nearby_llm(system_prompt, user_message, model)
        if planner_question and "Planner / Orchestrator" in system_prompt:
            parsed = dict(parsed, human_checkpoint_needed=True, clarification_question=planner_question)
        if afternoon and "Preference Agent" in system_prompt:
            stamp = user_message.split("Current local date and time: ")[1][:26].split(".")[0]
            now = _dt.strptime(" ".join(stamp.split()[1:3]), "%Y-%m-%d %H:%M")
            parsed = dict(parsed, requested_datetime=(now + _td(hours=3)).strftime("%Y-%m-%d %H:%M"))
        return parsed, usage
    return llm


def test_planner_question_does_not_stop_the_search(resources, monkeypatch):
    use_llm(monkeypatch, _llm_with(planner_question="Did you mean this afternoon or tonight?"))
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_pipeline(resources, "Indoor activity for a rainy afternoon with my roommates.", loc,
                                target_datetime=datetime(2026, 10, 16, 20, 0), synthesize=False)
    assert state["final_recommendations"], state["finish_reason"]
    assert state["clarification_question"] == "Did you mean this afternoon or tonight?"


def test_time_in_the_request_overrides_the_app_setting(resources, monkeypatch):
    use_llm(monkeypatch, _llm_with(afternoon=True))
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    far_off = datetime.now() + timedelta(days=5)
    state = run_pipeline(resources, "Something fun this afternoon", loc, target_datetime=far_off, synthesize=False)
    assert state["time_used"] != far_off.isoformat(sep=" ", timespec="minutes")
    assert any(line.startswith("Your request mentions a time") for line in state["adaptation_log"])


def _relevance_scores(score_for):
    """Fake relevance check: score_for(place_name) -> score."""
    import json as _json

    def llm(system_prompt, user_message, model=None):
        parsed, usage = nearby_llm(system_prompt, user_message, model)
        if "venue relevance validator" in system_prompt:
            rows = _json.loads(user_message.split("\n", 1)[1])
            parsed = {"assessments": [{"row_id": r["row_id"], "place_activity_relevance_score": score_for(r["place_name"]),
                                       "relevance_reason": "test"} for r in rows]}
        return parsed, usage
    return llm


def test_relevance_check_retries_with_the_next_batch(resources, monkeypatch):
    # Only one search's places fit; the first batch of 12 doesn't hold enough of them.
    use_llm(monkeypatch, _relevance_scores(lambda name: 0.9 if name.startswith("Burger Joint") else 0.1))
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_pipeline(resources, "I want cheap burgers", loc, synthesize=False,
                                target_datetime=datetime(2026, 10, 16, 20, 0))
    assert "place_relevance_retry1" in state["agent_results"]
    assert any("more were checked" in line for line in state["adaptation_log"])
    assert all(p["place_name"].startswith("Burger Joint") for p in state["final_recommendations"])


def test_weak_matches_are_flagged(resources, monkeypatch):
    use_llm(monkeypatch, _relevance_scores(lambda name: 0.65))
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_pipeline(resources, "I want to go skiing", loc, synthesize=False)
    assert state.get("weak_match") is True
    use_llm(monkeypatch, _relevance_scores(lambda name: 0.9))
    strong = run_pipeline(resources, "I want cheap burgers", loc, synthesize=False)
    assert not strong.get("weak_match")


def test_a_city_named_in_the_request_moves_the_search(resources, monkeypatch):
    PARIS = (48.8566, 2.3522)

    def llm(system_prompt, user_message, model=None):
        parsed, usage = nearby_llm(system_prompt, user_message, model)
        if "Preference Agent" in system_prompt:
            parsed = dict(parsed, city="Paris, France")
        return parsed, usage

    def osm(url, params=None, headers=None, timeout=None):
        if "nominatim" in url and "Paris" in params["q"]:
            return FakeResponse([{"lat": str(PARIS[0]), "lon": str(PARIS[1]), "display_name": "Paris, France", "name": "Paris"}])
        return nearby_get(url, params, headers, timeout)

    use_llm(monkeypatch, llm)
    monkeypatch.setattr("requests.get", osm)
    routes = []
    original_post = nearby_post

    def post(url, headers=None, json=None, timeout=None):
        if "computeRoutes" in url:
            routes.append(1)
        if "places:searchText" in url and json.get("pageSize") != 1:
            box = json["locationRestriction"]["rectangle"]
            assert box["low"]["latitude"] < PARIS[0] < box["high"]["latitude"]  # searched around Paris
            resp = original_post(url, headers, json, timeout)
            for p in resp._payload["places"]:
                p["location"] = {"latitude": PARIS[0] + (p["location"]["latitude"] - HOME[0]), "longitude": PARIS[1]}
            return resp
        return original_post(url, headers, json, timeout)

    monkeypatch.setattr("requests.post", post)
    monkeypatch.setattr("requests.post", post)
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_pipeline(resources, "Best tacos in Paris, France", loc, synthesize=False)
    assert state["searched_elsewhere"] == "Paris, France"
    assert state["final_recommendations"] and not routes  # no meaningless 5,000-mile drive times


def test_write_up_runs_as_its_own_step(resources):
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_pipeline(resources, "I want cheap burgers", loc, synthesize=False)
    assert state["final_recommendations"] and state["final_response"] is None
    synthesize_results(state)
    assert state["final_response"]["overall_summary"] and "synthesizer" in state["timings"]


def test_feedback_shapes_the_next_search(resources):
    memory = create_empty_user_memory()
    fb = feedback_agent("No more movies, something more active, nothing over 15 minutes away", [])
    memory = update_memory_from_feedback(memory, fb["data"])
    assert memory["disliked_activities"] == ["movies"] and memory["max_travel_minutes"] == 15

    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    first = run_pipeline(resources, "I want cheap burgers", loc, synthesize=False)
    rejected = first["final_recommendations"][0]["place_name"]
    memory["rejected_places"] = [rejected]
    second = run_pipeline(resources, "I want cheap burgers", loc, memory=memory, synthesize=False)
    assert rejected not in {p["place_name"] for p in second["all_places"]}
    assert "memory" in second["agents_called"]


def test_agents_that_ran_are_never_listed_as_skipped(resources):
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_pipeline(resources, "I want cheap burgers", loc, synthesize=False)
    assert not set(state["agents_called"]) & set(state["agents_skipped"])
