"""Offline tests for the Photo Vibe Agent and the parallel LLM fan-out."""

import threading
import time
from datetime import datetime

import pandas as pd
import pytest

from fakes import HOME, FakeResponse, nearby_get, nearby_llm, use_llm
from recommender import resolve_location, run_pipeline
from recommender.agents.matching import calculate_final_recommendation_score
from recommender.agents.photo_vibe import photo_vibe_agent
from recommender.agents.relevance import place_activity_relevance_agent
from recommender.core.llm import merge_usage, run_concurrently

USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "cost": 0.0001, "cached": False}
SEEN = {"media": [], "vision": []}


def photo_get(url, params=None, headers=None, timeout=None):
    """Place Details with photo references, and the photo media endpoint."""
    if url.endswith("/media"):
        SEEN["media"].append((url, params, headers))
        name = url.split("/v1/", 1)[1].rsplit("/media", 1)[0]
        return FakeResponse({"photoUri": f"https://lh3.googleusercontent.example/{name.replace('/', '_')}"})
    response = nearby_get(url, params, headers, timeout)
    if "places.googleapis.com/v1/places/" in url:
        pid = url.rsplit("/", 1)[1]
        payload = dict(response.json(), photos=[
            {"name": f"places/{pid}/photos/p{i}", "widthPx": 800, "heightPx": 600,
             "authorAttributions": [{"displayName": f"Photographer {i}"}]} for i in range(4)])
        return FakeResponse(payload)
    return response


def vision_llm(system_prompt, user_message, model=None, image_urls=None):
    if "Photo Vibe Agent" in system_prompt:
        SEEN["vision"].append(list(image_urls or []))
        pid = user_message.split("place_id: ", 1)[1].split("\n", 1)[0]
        score = 0.95 if pid.endswith("-3") else 0.2
        return {"place_id": "the model got this wrong", "photo_vibe_score": score, "vibe_tags": ["cozy", "dim lighting"],
                "photo_summary": "Warm, dim room with booths.", "visual_concerns": []}, dict(USAGE)
    return nearby_llm(system_prompt, user_message, model)


@pytest.fixture
def photo_resources(resources, monkeypatch):
    SEEN["media"].clear()
    SEEN["vision"].clear()
    monkeypatch.setattr("requests.get", photo_get)
    use_llm(monkeypatch, vision_llm)
    return resources


def test_photo_vibe_ranks_by_look_and_keeps_key_private(photo_resources):
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_pipeline(photo_resources, "Cozy burger spot for a date night", loc, photo_vibe="always",
                                target_datetime=datetime(2026, 10, 16, 20, 0), synthesize=False)
    assert not state["errors"], state["errors"]
    assert "photo_vibe" in state["agents_called"]
    # 3 photos per venue, at most 5 venues, every one through the media endpoint.
    assert SEEN["vision"] and all(len(urls) == 3 for urls in SEEN["vision"]) and len(SEEN["vision"]) <= 5
    assert state["usage"]["api_calls"]["google_place_photos"] == 3 * len(SEEN["vision"])
    # The model only ever sees public links, never the Google key.
    assert all("key=" not in u and "test" not in u for urls in SEEN["vision"] for u in urls)
    assert all(params["skipHttpRedirect"] == "true" for _, params, _ in SEEN["media"])
    ranked = pd.DataFrame(state["ranked_results"])
    looked = ranked[ranked["photo_vibe_score"].notna()]
    assert not looked.empty
    # place_id comes from our shortlist, not from the model's (wrong) copy.
    assert set(looked["place_id"]) <= set(ranked["place_id"])
    top = state["final_recommendations"][0]
    assert top["vibe_tags"] == ["cozy", "dim lighting"] and top["photo_urls"]
    assert top["photo_attributions"][0] == "Photographer 0"


def test_photo_vibe_follows_the_planner(photo_resources):
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    # The fake planner doesn't ask for photo_vibe, so "auto" skips it: no photo costs.
    state = run_pipeline(photo_resources, "I want cheap burgers", loc, synthesize=False)
    assert "photo_vibe" in state["agents_skipped"] and not SEEN["media"] and not SEEN["vision"]
    state = run_pipeline(photo_resources, "I want cheap burgers", loc, synthesize=False, photo_vibe="off")
    assert not SEEN["vision"]


def test_photo_agent_skips_cleanly_without_photos(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test")
    df = pd.DataFrame([{"place_id": "a", "place_name": "A"}])
    assert photo_vibe_agent(df, {"a": []}, "anything")["status"] == "skipped"


def test_vibe_weight_is_dropped_when_missing():
    old_four = calculate_final_recommendation_score(0.8, 0.6, 0.5, 0.7)
    assert old_four == pytest.approx((0.8 * 0.40 + 0.6 * 0.25 + 0.5 * 0.15 + 0.7 * 0.20) / 1.0, rel=1e-5)
    with_vibe = calculate_final_recommendation_score(0.8, 0.6, 0.5, 0.7, vibe_score=1.0)
    assert with_vibe == pytest.approx((0.8 * 0.40 + 0.6 * 0.25 + 0.5 * 0.15 + 0.7 * 0.20 + 1.0 * 0.15) / 1.15, rel=1e-5)


def test_llm_calls_really_run_at_the_same_time():
    active, peak, lock = [0], [0], threading.Lock()

    def slow(i):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        if i == 2:
            raise ValueError("one call failed")
        return i

    started = time.time()
    outcomes = run_concurrently(slow, range(5))
    assert peak[0] > 1 and time.time() - started < 0.2
    assert [r for r, _ in outcomes] == [0, 1, None, 3, 4]
    assert isinstance(outcomes[2][1], ValueError)
    merged = merge_usage([dict(USAGE), dict(USAGE, cached=True)])
    assert merged["total_tokens"] == 240 and merged["calls"] == 2 and merged["cached"] is False


def test_relevance_batches_in_parallel_and_survives_a_failed_batch(monkeypatch):
    calls = []

    def llm(system_prompt, user_message, model=None):
        import json
        rows = json.loads(user_message.split("\n", 1)[1])
        calls.append([r["row_id"] for r in rows])
        if rows[0]["row_id"] == 4:
            raise RuntimeError("timeout")
        return {"assessments": [{"row_id": r["row_id"], "place_activity_relevance_score": 0.1 if r["row_id"] == 0 else 0.9,
                                 "relevance_reason": "ok"} for r in rows]}, dict(USAGE)

    use_llm(monkeypatch, llm)
    df = pd.DataFrame([{"place_id": f"p{i}", "place_name": f"P{i}", "source_activity": "bowling"} for i in range(10)])
    out = place_activity_relevance_agent(df, 0.6, "bowling tonight", batch_size=4)
    assert out["status"] == "success"
    assert sorted(calls) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    kept = set(out["data"]["results"]["place_id"])
    assert "p0" not in kept  # judged and rejected
    assert {"p4", "p5", "p6", "p7"} <= kept  # never judged: kept, as when the whole check fails
    assert out["data"]["parallel_calls"] == 3 and out["data"]["failed_calls"] == 1
    assert out["usage"]["total_tokens"] == 240
