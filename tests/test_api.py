"""
Tests for the website's API (api/main.py): the endpoints the browser calls, the daily
search limits, and input checks. Everything outside is faked, so no keys or network.
"""

import json
import os

os.environ["PICK_THREE_NO_WARMUP"] = "1"  # don't load the real embedding model on import

import pytest
from fastapi.testclient import TestClient

import api.main as server
from fakes import HOME

SEARCH = {"prompt": "I want cheap burgers", "where": {"latitude": HOME[0], "longitude": HOME[1]},
          "when": "2026-10-16T20:00:00", "max_travel_minutes": 20}


@pytest.fixture
def client(resources, monkeypatch):
    monkeypatch.setattr(server, "_resources", resources)  # the fake catalog + embedder
    server._usage.update(total=0, by_visitor={})
    server.RECENT_RUNS.clear()
    return TestClient(server.app)


def run_search(client, body=SEARCH, headers=None):
    response = client.post("/api/search", json=body, headers=headers or {})
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()] if response.is_success else []
    return response, events


def test_config_lists_top10_categories(client):
    cfg = client.get("/api/config").json()
    assert cfg["ready"] is True
    assert {"restaurants", "coffee"} <= {c["id"] for c in cfg["categories"]}


def test_top10_comes_from_the_snapshot(client, monkeypatch, tmp_path):
    # The real snapshot isn't committed (Google's terms), so the test makes its own.
    snapshot = tmp_path / "featured.json"
    items = [{"rank": i, "place_name": f"Place {i}"} for i in range(1, 11)]
    snapshot.write_text(json.dumps({"generated": "2026-10-19", "lists": {"restaurants": {"label": "Restaurants",
                                                                                         "items": items}}}))
    monkeypatch.setattr("api.featured.SNAPSHOT", snapshot)
    data = client.get("/api/featured/restaurants").json()
    assert len(data["items"]) == 10 and data["items"][0]["rank"] == 1
    assert client.get("/api/featured/nope").status_code == 404


def test_search_streams_progress_then_a_clean_result(client):
    response, events = run_search(client)
    assert response.status_code == 200
    assert events[0]["type"] == "progress" and events[-1]["type"] == "result"
    result = events[-1]["result"]
    assert result["ok"] and len(result["picks"]) == 3
    # Exact visitor location never goes back to the browser, only a rounded map center.
    assert result["center"]["latitude"] == round(HOME[0], 3)
    # The write-up endpoint finishes the same search.
    write_up = client.post("/api/writeup", json={"run_id": result["run_id"]}).json()["write_up"]
    assert write_up["overall_summary"]


def test_daily_limit_applies_online_but_not_locally(client, monkeypatch):
    monkeypatch.setattr(server, "RUNNING_LOCALLY", True)
    for _ in range(server.RUNS_PER_VISITOR + 2):
        assert run_search(client)[0].status_code == 200  # your own machine: no limit

    monkeypatch.setattr(server, "RUNNING_LOCALLY", False)
    headers = {"X-Forwarded-For": "203.0.113.7"}
    for _ in range(server.RUNS_PER_VISITOR):
        assert run_search(client, headers=headers)[0].status_code == 200
    assert run_search(client, headers=headers)[0].status_code == 429


def test_faked_forwarded_address_does_not_reset_the_limit(client, monkeypatch):
    monkeypatch.setattr(server, "RUNNING_LOCALLY", False)
    # A browser can put anything first; the proxy-added address at the end is the one that counts.
    for fake in range(server.RUNS_PER_VISITOR):
        assert run_search(client, headers={"X-Forwarded-For": f"10.0.0.{fake}, 203.0.113.9"})[0].status_code == 200
    blocked, _ = run_search(client, headers={"X-Forwarded-For": "127.0.0.1, 203.0.113.9"})
    assert blocked.status_code == 429


def test_bad_input_is_rejected(client):
    assert client.post("/api/search", json={**SEARCH, "prompt": "x"}).status_code == 422
    assert client.post("/api/search", json={**SEARCH, "max_travel_minutes": 500}).status_code == 422
    assert client.get("/api/photo", params={"name": "https://evil.example/x"}).status_code == 400
    assert client.post("/api/writeup", json={"run_id": "nope"}).status_code == 404


def test_feedback_returns_updated_memory(client):
    res = client.post("/api/feedback", json={"text": "No more movies, nothing over 15 minutes away",
                                             "picks": [], "memory": {"liked_activities": ["bowling"]}}).json()
    assert res["memory"]["disliked_activities"] == ["movies"]
    assert res["memory"]["liked_activities"] == ["bowling"] and res["memory"]["max_travel_minutes"] == 15
