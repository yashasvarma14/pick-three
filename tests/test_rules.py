"""
Unit tests for the plain-Python rules: scoring, shortlist rules, opening
hours, time parsing, caching and small helpers. No pipeline runs here.
"""

import threading
import time
from datetime import datetime

import pandas as pd
import pytest

from fakes import EVERY_DAY_5PM_TO_2AM, FIXTURE, HOME, MORNINGS_ONLY, FakeEmbedder, FakeResponse
from recommender.agents.matching import calculate_final_recommendation_score
from recommender.agents.place_scoring import is_open_at, score_places
from recommender.core.catalog import outing_query
from recommender.services.ticketmaster import build_event_window_utc


def test_open_hours():
    friday_8pm = datetime(2026, 10, 16, 20, 0)
    saturday_1am = datetime(2026, 10, 17, 1, 0)
    assert is_open_at(EVERY_DAY_5PM_TO_2AM, friday_8pm) is True
    assert is_open_at(EVERY_DAY_5PM_TO_2AM, saturday_1am) is True  # past midnight
    assert is_open_at(MORNINGS_ONLY, friday_8pm) is False
    assert is_open_at([], friday_8pm) is None
    assert is_open_at([{"open": {"day": 0, "hour": 0, "minute": 0}}], friday_8pm) is True  # 24/7


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


def _place(pid, name, kind="Arcade", rating=4.5, reviews=500, lat=HOME[0], lng=HOME[1], site=None, address=None):
    return {"place_id": pid, "place_name": name, "primary_type": kind, "rating": rating, "review_count": reviews,
            "latitude": lat, "longitude": lng, "website_uri": site, "address": address, "matched_searches": ["x"]}


def test_few_reviews_cannot_beat_many():
    from recommender.agents.place_scoring import adjusted_rating
    adj = adjusted_rating(pd.Series([5.0, 4.6, 4.3]), pd.Series([3, 2000, 300]))
    assert adj[1] > adj[0]  # 4.6 from 2,000 reviews beats 5.0 from 3


def test_top_picks_never_repeat_a_business():
    from recommender.agents.shortlist import diverse_selection
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
    from recommender.agents.shortlist import diverse_selection
    ranked = pd.DataFrame([_place(str(i), f"Arcade Number {i}", kind="Arcade") for i in range(4)]
                          + [_place("b", "Some Bowling Alley", kind="Bowling alley")])
    assert [p["primary_type"] for p in diverse_selection(ranked, 3)].count("Arcade") == 2
    assert [p["primary_type"] for p in diverse_selection(ranked, 3, specific_request=True)].count("Arcade") == 3


def test_thin_review_counts_only_as_a_last_resort():
    from recommender.agents.shortlist import diverse_selection
    ranked = pd.DataFrame([_place("new", "Brand New Spot", reviews=2), _place("a", "Pinball Palace"),
                           _place("b", "Lazarus Brewing", kind="Bar"), _place("c", "Epoch Coffee", kind="Cafe")])
    assert "Brand New Spot" not in [p["place_name"] for p in diverse_selection(ranked, 3)]
    assert len(diverse_selection(ranked.head(2), 2)) == 2  # still fills the slots when it must


def test_duplicate_listings_are_merged():
    from recommender.agents.shortlist import remove_duplicate_listings
    df = pd.DataFrame([
        _place("a", "Hopdoddy Burger Bar", reviews=3000, address="1400 S Congress Ave"),
        _place("b", "Hopdoddy Burger Bar (Congress)", reviews=12, lat=HOME[0] + 0.0003),  # same spot, second listing
        _place("c", "Hopdoddy Burger Bar", reviews=900, lat=HOME[0] + 0.1),  # another branch, 7 miles away: kept
    ])
    out, removed = remove_duplicate_listings(df)
    assert removed == 1 and set(out["place_id"]) == {"a", "c"}


def test_bad_requested_times_are_ignored():
    from recommender.agents.preference import _parse_requested_time
    now = datetime(2026, 10, 13, 9, 30)
    assert _parse_requested_time("2026-10-13 14:00", now) == datetime(2026, 10, 13, 14, 0)
    assert _parse_requested_time("2026-09-01 14:00", now) is None  # in the past
    assert _parse_requested_time("2027-01-01 14:00", now) is None  # beyond the forecast
    assert _parse_requested_time("this afternoon", now) is None


def test_kratom_bars_and_smoke_shops_never_become_top_picks():
    from recommender.agents.shortlist import diverse_selection
    ranked = pd.DataFrame([_place("k", "Tulum Botanical Kava & Kratom", kind="Bar"),
                           _place("s", "Cloud 9 Smoke Shop", kind="Store"),
                           _place("t", "Zhi Tea", kind="Tea house"), _place("m", "Blanton Museum", kind="Museum"),
                           _place("b", "BookPeople", kind="Book store")])
    assert [p["place_name"] for p in diverse_selection(ranked, 3)] == ["Zhi Tea", "Blanton Museum", "BookPeople"]


def test_open_ended_requests_get_one_bar_unless_they_ask_for_drinks():
    from recommender.agents.shortlist import diverse_selection
    ranked = pd.DataFrame([_place("1", "The Tigress Pub", kind="Bar"), _place("2", "Rosie's Wine Bar", kind="Wine bar"),
                           _place("3", "Here Nor There", kind="Cocktail Bar"), _place("4", "Zhi Tea", kind="Tea house"),
                           _place("5", "Blanton Museum", kind="Museum")])
    quiet = diverse_selection(ranked, 3, request_text="Something relaxing and quiet for two", max_bars=1)
    assert [p["place_name"] for p in quiet] == ["The Tigress Pub", "Zhi Tea", "Blanton Museum"]
    drinks = diverse_selection(ranked, 3, request_text="A date night with good cocktails", max_bars=1)
    assert [p["place_name"] for p in drinks] == ["The Tigress Pub", "Rosie's Wine Bar", "Here Nor There"]


def test_weather_lookup_retries_once_after_a_timeout(monkeypatch):
    import requests as _requests
    from recommender.services.open_meteo import fetch_forecast
    calls = []

    def flaky(url, params=None, timeout=None, **kw):
        calls.append(timeout)
        if len(calls) == 1:
            raise _requests.exceptions.ReadTimeout("slow")
        return FakeResponse({"timezone": "America/Chicago", "hourly": {"time": []}})

    monkeypatch.setattr("requests.get", flaky)
    assert fetch_forecast(10.0, 20.0)["timezone"] == "America/Chicago"
    assert len(calls) == 2 and calls[1] > calls[0]


def test_ttl_cache_shares_and_skips_errors():
    from recommender.core.cache import ttl_cache

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


def test_catalog_keeps_one_row_per_activity(resources):
    assert resources.activities["activity"].is_unique
    assert len(resources.activities) < len(pd.read_json(FIXTURE, lines=True))


def test_helpers():
    assert outing_query("Take a nap") is None
    assert outing_query("Go to an Escape Room") == "escape room"
    start, end = build_event_window_utc(datetime(2026, 10, 12, 19, 0), "America/Chicago")
    assert start.hour == 21 and end.hour == 6  # 7pm CDT = 00:00 UTC, window -3h/+6h
    assert calculate_final_recommendation_score(0.8, 0.6, None, None) == pytest.approx(
        (0.8 * 0.4 + 0.6 * 0.25) / 0.65, rel=1e-5)
