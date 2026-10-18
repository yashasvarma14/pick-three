"""The live tracker: what gets logged, and that logging can't break the app."""

import math

import pytest

from fakes import HOME
from recommender import resolve_location, run_pipeline
from recommender.services import tracker


class Captured:
    def __init__(self):
        self.posts = []

    def post(self, url, payload):
        self.posts.append((url, payload))

        class R:
            status_code = 200
        return R()


@pytest.fixture
def captured(monkeypatch):
    cap = Captured()
    monkeypatch.setattr("recommender.services.tracker._http_post", cap.post)
    monkeypatch.setenv("TRACKER_URL", "https://script.google.com/macros/s/TEST/exec")
    monkeypatch.setenv("TRACKER_TOKEN", "secret")
    return cap


def test_logging_is_off_without_a_url(monkeypatch):
    monkeypatch.delenv("TRACKER_URL", raising=False)
    assert tracker.enabled() is False
    assert tracker.send({"kind": "run", "tabs": {}}) is False


def test_run_rows_cover_every_tab_and_hide_exact_gps(resources, captured):
    loc = resolve_location(latitude=HOME[0], longitude=HOME[1])
    state = run_pipeline(resources, "I want cheap burgers", loc, synthesize=False)
    assert tracker.log_run(state, "S-1", background=False)
    url, payload = captured.posts[0]
    assert payload["token"] == "secret" and payload["kind"] == "run"
    tabs = payload["tabs"]
    run = tabs["Run_Log"][0]
    assert run["Prompt"] == "I want cheap burgers" and run["Picks_Returned"] == len(state["final_recommendations"])
    assert run["Location"] == "Device location (~34.1, -118.2)"  # rounded, never the exact coordinates
    assert str(HOME[0]) not in str(payload)
    assert len(tabs["Recommendations"]) == len(state["final_recommendations"])
    assert {r["Agent"] for r in tabs["Agent_Usage"]} >= {"orchestrator", "preference", "local_places", "matching"}
    for rows in tabs.values():  # nothing a spreadsheet can't store
        for row in rows:
            assert not any(isinstance(v, float) and math.isnan(v) for v in row.values())


def test_feedback_rows(captured):
    tracker.log_feedback("r1", "S-1", helpful=False, text="Too loud", summary="Wants quieter", background=False)
    row = captured.posts[0][1]["tabs"]["User_Feedback"][0]
    assert row["Helpful"] == "No" and row["Feedback_Text"] == "Too loud" and row["Run_ID"] == "r1"


def test_a_broken_tracker_never_breaks_the_app(monkeypatch):
    monkeypatch.setenv("TRACKER_URL", "https://script.google.com/macros/s/TEST/exec")

    def boom(*a, **k):
        raise ConnectionError("sheet is down")

    monkeypatch.setattr("recommender.services.tracker._http_post", boom)
    assert tracker.send({"kind": "run", "tabs": {}}, background=False) is False  # no exception
