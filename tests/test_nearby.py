"""Offline tests for "near me" mode (every external service faked)."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from recommender import build_resources
from recommender.nearby import is_open_at, resolve_location, score_places
from recommender.nearby_workflow import run_nearby_workflow
from test_pipeline import FIXTURE, FakeEmbedder, FakeResponse, fake_get, fake_llm, fake_post

HOME = (34.0522, -118.2437)  # downtown Los Angeles
EVERY_DAY_5PM_TO_2AM = [{"open": {"day": d, "hour": 17, "minute": 0},
                         "close": {"day": (d + 1) % 7, "hour": 2, "minute": 0}} for d in range(7)]
MORNINGS_ONLY = [{"open": {"day": d, "hour": 7, "minute": 0}, "close": {"day": d, "hour": 11, "minute": 0}} for d in range(7)]
CALLS = {"search": [], "details": 0}


def nearby_post(url, headers=None, json=None, timeout=None):
    if "computeRoutes" in url:  # ~2 minutes per km from HOME, plus 4 minutes
        from recommender.places import distance_km
        o = json["origin"]["location"]["latLng"]
        d = json["destination"]["location"]["latLng"]
        minutes = 4 + 2 * distance_km(o["latitude"], o["longitude"], d["latitude"], d["longitude"])
        return FakeResponse({"routes": [{"distanceMeters": int(minutes * 500), "duration": f"{int(minutes * 60)}s",
                                         "staticDuration": f"{int(minutes * 50)}s"}]})
    if "places:searchText" not in url:
        return fake_post(url, headers, json, timeout)
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


def nearby_llm(system_prompt, user_message, model=None):
    parsed, usage = fake_llm(system_prompt, user_message, model)
    if "Preference Agent" in system_prompt:
        parsed["search_phrases"] = ["smash burger", "burger joint"] if "burger" in user_message.lower() else ["quiet wine bar", "day spa", "botanical garden"]
    return parsed, usage


@pytest.fixture
def resources(monkeypatch):
    CALLS["search"].clear()
    CALLS["details"] = 0
    for module in ["recommender.weather", "recommender.events", "recommender.rag", "recommender.nearby"]:
        monkeypatch.setattr(f"{module}.requests.get", nearby_get)
    for module in ["recommender.places", "recommender.mobility", "recommender.nearby"]:
        monkeypatch.setattr(f"{module}.requests.post", nearby_post)
    for module in ["recommender.preference", "recommender.relevance", "recommender.rag",
                   "recommender.orchestrator", "recommender.synthesizer", "recommender.memory"]:
        monkeypatch.setattr(f"{module}.call_llm_json", nearby_llm)
    for name in ["GOOGLE_MAPS_API_KEY", "TICKETMASTER_API_KEY", "ANTHROPIC_API_KEY"]:
        monkeypatch.setenv(name, "test")
    return build_resources(FIXTURE, embedder=FakeEmbedder())


def test_open_hours():
    friday_8pm = datetime(2026, 10, 16, 20, 0)
    saturday_1am = datetime(2026, 10, 17, 1, 0)
    assert is_open_at(EVERY_DAY_5PM_TO_2AM, friday_8pm) is True
    assert is_open_at(EVERY_DAY_5PM_TO_2AM, saturday_1am) is True  # past midnight
    assert is_open_at(MORNINGS_ONLY, friday_8pm) is False
    assert is_open_at([], friday_8pm) is None
    assert is_open_at([{"open": {"day": 0, "hour": 0, "minute": 0}}], friday_8pm) is True  # 24/7


def test_resolve_location_from_gps_and_text(resources):
    gps = resolve_location(latitude=HOME[0], longitude=HOME[1])
    assert gps["origin"].startswith("34.05") and gps["timezone"]
    google_before = len(CALLS["search"])
    typed = resolve_location("downtown LA")
    assert abs(typed["latitude"] - HOME[0]) < 1e-6
    assert CALLS.get("osm") == 1  # typed address used the free geocoder...
    with pytest.raises(LookupError):
        resolve_location()


def test_near_me_finds_and_keeps_every_place(resources):
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_nearby_workflow(resources, "I want cheap burgers", loc,
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
    state = run_nearby_workflow(resources, "Something relaxing and quiet for two", loc, synthesize=False)
    # The AI gave 3 searches: all used, no generic catalog idea squeezed in, no misleading note.
    assert state["search_phrases"] == ["quiet wine bar", "day spa", "botanical garden"]
    assert not any("not into" in line for line in state["adaptation_log"])

    def one_idea(system_prompt, user_message, model=None):
        parsed, usage = nearby_llm(system_prompt, user_message, model)
        if "Preference Agent" in system_prompt:
            parsed = dict(parsed, search_phrases=["quiet wine bar"])
        return parsed, usage

    monkeypatch.setattr("recommender.preference.call_llm_json", one_idea)
    state = run_nearby_workflow(resources, "Something relaxing and quiet for two", loc, synthesize=False)
    assert state["search_phrases"][0] == "quiet wine bar" and len(state["search_phrases"]) == 3  # catalog filled the gap


def test_memory_skips_disliked_searches(resources):
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    memory = {"user_id": "v", "disliked_activities": ["spa"], "liked_activities": [], "liked_places": [],
              "rejected_places": [], "positive_preferences": [], "negative_preferences": []}
    state = run_nearby_workflow(resources, "Something relaxing and quiet for two", loc, memory=memory, synthesize=False)
    assert "day spa" not in state["search_phrases"]


def test_bad_weather_pushes_outdoor_places_down():
    df = pd.DataFrame([
        {"place_id": "a", "place_name": "Garden", "primary_type": "Garden", "place_types": ["park"], "environment": "outdoor",
         "matched_searches": ["x"], "rating": 4.5, "review_count": 500, "straight_line_miles": 2, "opening_periods": []},
        {"place_id": "b", "place_name": "Garden", "primary_type": "Garden", "place_types": ["museum"], "environment": "indoor",
         "matched_searches": ["x"], "rating": 4.5, "review_count": 500, "straight_line_miles": 2, "opening_periods": []},
    ])
    sunny = score_places(FakeEmbedder(), df, "garden", 20, None, 1.0)
    stormy = score_places(FakeEmbedder(), df, "garden", 20, None, 0.2)
    assert stormy.iloc[0]["place_id"] == "b"
    assert sunny["pre_score"].max() > stormy["pre_score"].min()


def test_tight_drive_limit_removes_far_places(resources):
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_nearby_workflow(resources, "I want cheap burgers", loc, max_travel_minutes=5,
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

    monkeypatch.setattr("recommender.rag.call_llm_json", no_evidence)
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_nearby_workflow(resources, "I want cheap burgers", loc, synthesize=False)
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
    monkeypatch.setattr("recommender.nearby.time.sleep", lambda s: None)
    post, _ = _rate_limited(fail_query="smash burger")
    monkeypatch.setattr("recommender.nearby.requests.post", post)
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_nearby_workflow(resources, "I want cheap burgers", loc, synthesize=False)
    assert state["final_recommendations"], state["errors"]
    assert any("turned down 1 of" in line for line in state["adaptation_log"])


def test_brief_rate_limit_is_retried(resources, monkeypatch):
    waits = []
    monkeypatch.setattr("recommender.nearby.time.sleep", waits.append)
    post, seen = _rate_limited(fail_times=2)
    monkeypatch.setattr("recommender.nearby.requests.post", post)
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_nearby_workflow(resources, "I want cheap burgers", loc, synthesize=False)
    assert state["final_recommendations"]
    assert not any("turned down" in line for line in state["adaptation_log"])
    assert waits and all(v == 3 for v in seen.values())  # two 429s, then success


def test_all_searches_rejected_explains_why(resources, monkeypatch):
    monkeypatch.setattr("recommender.nearby.time.sleep", lambda s: None)
    post, _ = _rate_limited()
    monkeypatch.setattr("recommender.nearby.requests.post", post)
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_nearby_workflow(resources, "I want cheap burgers", loc, synthesize=False)
    assert not state["final_recommendations"]
    assert "RESOURCE_EXHAUSTED" in " ".join(state["errors"])


# ---------------- quality rules ----------------

def _place(pid, name, kind="Arcade", rating=4.5, reviews=500, lat=HOME[0], lng=HOME[1], site=None, address=None):
    return {"place_id": pid, "place_name": name, "primary_type": kind, "rating": rating, "review_count": reviews,
            "latitude": lat, "longitude": lng, "website_uri": site, "address": address, "matched_searches": ["x"]}


def test_few_reviews_cannot_beat_many():
    from recommender.nearby import adjusted_rating
    adj = adjusted_rating(pd.Series([5.0, 4.6, 4.3]), pd.Series([3, 2000, 300]))
    assert adj[1] > adj[0]  # 4.6 from 2,000 reviews beats 5.0 from 3


def test_top_picks_never_repeat_a_business():
    from recommender.nearby import diverse_selection
    ranked = pd.DataFrame([
        _place("1", "Main Event Austin", site="https://www.mainevent.com/austin"),
        _place("2", "Main Event Pflugerville", site="https://mainevent.com/pflugerville", lat=HOME[0] + 0.2),
        _place("3", "Mohawk", kind="Bar"),
        _place("4", "Mohawk Austin", kind="Bar", lat=HOME[0] + 0.01),
        _place("5", "Austin Pizza Garden", kind="Restaurant"),
        _place("6", "Austin Pizza Kitchen", kind="Restaurant"),
    ])
    picks = [p["place_name"] for p in diverse_selection(ranked, 3)]
    assert picks == ["Main Event Austin", "Mohawk", "Austin Pizza Garden"]


def test_open_ended_picks_mix_kinds_but_specific_ones_dont_have_to():
    from recommender.nearby import diverse_selection
    ranked = pd.DataFrame([_place(str(i), f"Arcade Number {i}", kind="Arcade") for i in range(4)]
                          + [_place("b", "Some Bowling Alley", kind="Bowling alley")])
    assert [p["primary_type"] for p in diverse_selection(ranked, 3)].count("Arcade") == 2
    assert [p["primary_type"] for p in diverse_selection(ranked, 3, specific_request=True)].count("Arcade") == 3


def test_thin_review_counts_only_as_a_last_resort():
    from recommender.nearby import diverse_selection
    ranked = pd.DataFrame([_place("new", "Brand New Spot", reviews=2), _place("a", "Pinball Palace"),
                           _place("b", "Lazarus Brewing", kind="Bar"), _place("c", "Epoch Coffee", kind="Cafe")])
    assert "Brand New Spot" not in [p["place_name"] for p in diverse_selection(ranked, 3)]
    assert len(diverse_selection(ranked.head(2), 2)) == 2  # still fills the slots when it must


def test_duplicate_listings_are_merged():
    from recommender.nearby import remove_duplicate_listings
    df = pd.DataFrame([
        _place("a", "Hopdoddy Burger Bar", reviews=3000, address="1400 S Congress Ave"),
        _place("b", "Hopdoddy Burger Bar (Congress)", reviews=12, lat=HOME[0] + 0.0003),  # same spot, second listing
        _place("c", "Hopdoddy Burger Bar", reviews=900, lat=HOME[0] + 0.1),  # another branch, 7 miles away: kept
    ])
    out, removed = remove_duplicate_listings(df)
    assert removed == 1 and set(out["place_id"]) == {"a", "c"}


def test_relevance_check_sees_the_request(resources, monkeypatch):
    seen = []

    def spy(system_prompt, user_message, model=None):
        if "venue relevance validator" in system_prompt:
            seen.append(user_message)
        return nearby_llm(system_prompt, user_message, model)

    monkeypatch.setattr("recommender.relevance.call_llm_json", spy)
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    run_nearby_workflow(resources, "I want cheap burgers", loc, synthesize=False)
    assert seen and "I want cheap burgers" in seen[0].split("\n")[0]


def test_no_duplicate_businesses_in_a_full_run(resources, monkeypatch):
    def chains(url, headers=None, json=None, timeout=None):
        resp = nearby_post(url, headers, json, timeout)
        if "places:searchText" in url and json.get("pageSize") != 1:
            for i, p in enumerate(resp._payload["places"]):
                p["websiteUri"] = "https://www.burgerchain.com/" + str(i)  # every result is one chain
                p["displayName"] = {"text": f"Burger Chain {json['textQuery']} {i}"}
        return resp

    monkeypatch.setattr("recommender.nearby.requests.post", chains)
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_nearby_workflow(resources, "I want cheap burgers", loc, synthesize=False)
    assert len(state["final_recommendations"]) == 1  # one chain = one pick, never three branches


# ---------------- review fixes (Oct 13) ----------------

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


def _use_llm(monkeypatch, llm):
    for module in ["recommender.preference", "recommender.relevance", "recommender.rag",
                   "recommender.orchestrator", "recommender.synthesizer"]:
        monkeypatch.setattr(f"{module}.call_llm_json", llm)


def test_planner_question_does_not_stop_the_search(resources, monkeypatch):
    _use_llm(monkeypatch, _llm_with(planner_question="Did you mean this afternoon or tonight?"))
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_nearby_workflow(resources, "Indoor activity for a rainy afternoon with my roommates.", loc,
                                target_datetime=datetime(2026, 10, 16, 20, 0), synthesize=False)
    assert state["final_recommendations"], state["finish_reason"]
    assert state["clarification_question"] == "Did you mean this afternoon or tonight?"


def test_time_in_the_request_overrides_the_app_setting(resources, monkeypatch):
    _use_llm(monkeypatch, _llm_with(afternoon=True))
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    far_off = datetime.now() + timedelta(days=5)
    state = run_nearby_workflow(resources, "Something fun this afternoon", loc, target_datetime=far_off, synthesize=False)
    assert state["time_used"] != far_off.isoformat(sep=" ", timespec="minutes")
    assert any(line.startswith("Your request mentions a time") for line in state["adaptation_log"])


def test_bad_requested_times_are_ignored():
    from recommender.preference import _parse_requested_time
    now = datetime(2026, 10, 13, 9, 30)
    assert _parse_requested_time("2026-10-13 14:00", now) == datetime(2026, 10, 13, 14, 0)
    assert _parse_requested_time("2026-09-01 14:00", now) is None  # in the past
    assert _parse_requested_time("2027-01-01 14:00", now) is None  # beyond the forecast
    assert _parse_requested_time("this afternoon", now) is None


def test_kratom_bars_and_smoke_shops_never_become_top_picks():
    from recommender.nearby import diverse_selection
    ranked = pd.DataFrame([_place("k", "Tulum Botanical Kava & Kratom", kind="Bar"),
                           _place("s", "Cloud 9 Smoke Shop", kind="Store"),
                           _place("t", "Zhi Tea", kind="Tea house"), _place("m", "Blanton Museum", kind="Museum"),
                           _place("b", "BookPeople", kind="Book store")])
    assert [p["place_name"] for p in diverse_selection(ranked, 3)] == ["Zhi Tea", "Blanton Museum", "BookPeople"]


def test_open_ended_requests_get_one_bar_unless_they_ask_for_drinks():
    from recommender.nearby import diverse_selection
    ranked = pd.DataFrame([_place("1", "The Tigress Pub", kind="Bar"), _place("2", "Rosie's Wine Bar", kind="Wine bar"),
                           _place("3", "Here Nor There", kind="Cocktail Bar"), _place("4", "Zhi Tea", kind="Tea house"),
                           _place("5", "Blanton Museum", kind="Museum")])
    quiet = diverse_selection(ranked, 3, request_text="Something relaxing and quiet for two", max_bars=1)
    assert [p["place_name"] for p in quiet] == ["The Tigress Pub", "Zhi Tea", "Blanton Museum"]
    drinks = diverse_selection(ranked, 3, request_text="A date night with good cocktails", max_bars=1)
    assert [p["place_name"] for p in drinks] == ["The Tigress Pub", "Rosie's Wine Bar", "Here Nor There"]


# ---------------- stress-test fixes (Oct 13, round 2) ----------------

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
    monkeypatch.setattr("recommender.relevance.call_llm_json",
                        _relevance_scores(lambda name: 0.9 if name.startswith("Burger Joint") else 0.1))
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_nearby_workflow(resources, "I want cheap burgers", loc, synthesize=False,
                                target_datetime=datetime(2026, 10, 16, 20, 0))
    assert "place_relevance_retry1" in state["agent_results"]
    assert any("more were checked" in line for line in state["adaptation_log"])
    assert all(p["place_name"].startswith("Burger Joint") for p in state["final_recommendations"])


def test_weak_matches_are_flagged(resources, monkeypatch):
    monkeypatch.setattr("recommender.relevance.call_llm_json", _relevance_scores(lambda name: 0.65))
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_nearby_workflow(resources, "I want to go skiing", loc, synthesize=False)
    assert state.get("weak_match") is True
    monkeypatch.setattr("recommender.relevance.call_llm_json", _relevance_scores(lambda name: 0.9))
    strong = run_nearby_workflow(resources, "I want cheap burgers", loc, synthesize=False)
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

    monkeypatch.setattr("recommender.preference.call_llm_json", llm)
    monkeypatch.setattr("recommender.nearby.requests.get", osm)
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

    monkeypatch.setattr("recommender.nearby.requests.post", post)
    monkeypatch.setattr("recommender.mobility.requests.post", post)
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_nearby_workflow(resources, "Best tacos in Paris, France", loc, synthesize=False)
    assert state["searched_elsewhere"] == "Paris, France"
    assert state["final_recommendations"] and not routes  # no meaningless 5,000-mile drive times


def test_weather_lookup_retries_once_after_a_timeout(monkeypatch):
    import requests as _requests
    from recommender.weather import fetch_forecast
    calls = []

    def flaky(url, params=None, timeout=None, **kw):
        calls.append(timeout)
        if len(calls) == 1:
            raise _requests.exceptions.ReadTimeout("slow")
        return FakeResponse({"timezone": "America/Chicago", "hourly": {"time": []}})

    monkeypatch.setattr("recommender.weather.requests.get", flaky)
    assert fetch_forecast(10.0, 20.0)["timezone"] == "America/Chicago"
    assert len(calls) == 2 and calls[1] > calls[0]
