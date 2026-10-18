"""
The web API that the website talks to.

Run locally:   uvicorn api.main:app --reload --port 8000
then open      http://localhost:8000   (the site in web/ is served from here too)

Endpoints
  GET  /api/config               which features are available (keys set, categories)
  GET  /api/featured/{category}  Austin's top 10 for the home page
  GET  /api/photo?name=...       a Google Maps photo (redirect; keeps the API key private)
  POST /api/locate               typed address or GPS -> a place label
  POST /api/search               runs the pipeline, streaming progress as it goes
  POST /api/writeup              step 11 (the explanation) for a finished search
  POST /api/feedback             feedback text -> updated memory (memory lives in the browser)
"""

import json
import queue
import re
import threading
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recommender import (build_resources, create_empty_user_memory, feedback_agent, resolve_location, run_pipeline,
                         synthesize_results, update_memory_from_feedback)
from recommender.agents.memory import LIST_FIELDS
from recommender.core import config
from recommender.services.google_places import photo_uri

from .featured import CATEGORIES, featured_list
from .serialize import clean, public_result

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
PHOTO_NAME = re.compile(r"^places/[A-Za-z0-9_-]+/photos/[A-Za-z0-9_-]+$")

app = FastAPI(title="Tonight, Sorted API")
app.add_middleware(
    CORSMiddleware,
    # The deployed front end's address, e.g. https://tonight-sorted.vercel.app
    allow_origins=[o.strip() for o in str(config.get_setting("ALLOWED_ORIGINS", "")).split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Shared things: the embedding model (loaded once), recent runs, limits
# ---------------------------------------------------------------------------

_resources, _resources_lock = None, threading.Lock()


def get_resources():
    global _resources
    with _resources_lock:
        if _resources is None:
            _resources = build_resources()
    return _resources


RECENT_RUNS = OrderedDict()  # run_id -> state, so /api/writeup can finish a search
MAX_RECENT_RUNS = 100

RUNS_PER_VISITOR = int(config.get_setting("RUNS_PER_VISITOR_PER_DAY", 15))
RUNS_PER_DAY = int(config.get_setting("RUNS_PER_DAY", 150))
_usage = {"date": date.today(), "total": 0, "by_visitor": {}}
_usage_lock = threading.Lock()


def visitor_id(request):
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")


def take_a_search(visitor):
    """Count one search against the daily limits, or raise 429."""
    with _usage_lock:
        if _usage["date"] != date.today():
            _usage.update(date=date.today(), total=0, by_visitor={})
        mine = _usage["by_visitor"].get(visitor, 0)
        if mine >= RUNS_PER_VISITOR:
            raise HTTPException(429, f"You've used today's {RUNS_PER_VISITOR} searches. Come back tomorrow!")
        if _usage["total"] >= RUNS_PER_DAY:
            raise HTTPException(429, "The site has hit today's search limit. It resets at midnight.")
        _usage["by_visitor"][visitor] = mine + 1
        _usage["total"] += 1
        return RUNS_PER_VISITOR - mine - 1


def clean_memory(memory):
    """Memory comes from the browser: keep only known fields, short strings, short lists."""
    base = create_empty_user_memory("visitor")
    if not isinstance(memory, dict):
        return base
    for field in LIST_FIELDS:
        items = memory.get(field) or []
        base[field] = [str(i)[:80] for i in items if isinstance(i, (str, int, float))][:30]
    if memory.get("budget_preference") in {"free", "low", "medium", "high"}:
        base["budget_preference"] = memory["budget_preference"]
    try:
        base["max_travel_minutes"] = int(memory["max_travel_minutes"]) if memory.get("max_travel_minutes") else None
    except (TypeError, ValueError):
        pass
    return base


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class Where(BaseModel):
    text: str | None = Field(None, max_length=200)
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)


class SearchBody(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=500)
    where: Where
    when: datetime | None = None          # naive local time; None = right now
    max_travel_minutes: int = Field(20, ge=5, le=60)
    deep: bool = False
    use_memory: bool = True
    memory: dict | None = None


class WriteupBody(BaseModel):
    run_id: str = Field(..., max_length=16)


class FeedbackBody(BaseModel):
    text: str = Field(..., min_length=2, max_length=500)
    picks: list[dict] = []
    memory: dict | None = None


def locate(where):
    try:
        if where.latitude is not None and where.longitude is not None:
            return resolve_location(latitude=where.latitude, longitude=where.longitude)
        return resolve_location(where.text)
    except LookupError as exc:
        raise HTTPException(400, str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/config")
def get_config():
    status = config.api_status()
    categories = [{"id": key, "label": spec["label"]} for key, spec in CATEGORIES.items()
                  if spec["search"] or status["ticketmaster"]]
    return {"ready": status["llm"] and status["google_maps"], "categories": categories,
            "searches_per_day": RUNS_PER_VISITOR}


@app.get("/api/featured/{category}")
def get_featured(category: str):
    if category not in CATEGORIES:
        raise HTTPException(404, "Unknown category.")
    if not config.google_maps_api_key():
        raise HTTPException(503, "Google Maps isn't configured.")
    try:
        items = featured_list(category)
    except Exception as exc:
        raise HTTPException(502, f"Couldn't load this list right now ({type(exc).__name__}).")
    return {"category": category, "label": CATEGORIES[category]["label"], "items": clean(items)}


@app.get("/api/photo")
def get_photo(name: str, w: int = 640):
    if not PHOTO_NAME.match(name):
        raise HTTPException(400, "Not a Google Maps photo reference.")
    try:
        url = photo_uri(name, max_width=max(120, min(int(w), 1200)))
    except Exception:
        raise HTTPException(404, "Photo unavailable.")
    if not url:
        raise HTTPException(404, "Photo unavailable.")
    return RedirectResponse(url, status_code=302, headers={"Cache-Control": "public, max-age=3000"})


@app.post("/api/locate")
def post_locate(where: Where):
    loc = locate(where)
    return {"label": loc.get("label"), "timezone": loc.get("timezone")}


@app.post("/api/search")
def post_search(body: SearchBody, request: Request):
    """
    Streams newline-delimited JSON, one line per event:
        {"type": "progress", "step": "local_places", "message": "...", "seconds": 4.2}
        {"type": "result", "result": {...}, "searches_left": 12}
        {"type": "error", "message": "..."}
    """
    location = locate(body.where)
    searches_left = take_a_search(visitor_id(request))
    events = queue.Queue()
    started = datetime.now()

    def progress(step, message):
        events.put({"type": "progress", "step": step, "message": message,
                    "seconds": round((datetime.now() - started).total_seconds(), 1)})

    def work():
        try:
            state = run_pipeline(get_resources(), body.prompt.strip(), location,
                                 memory=clean_memory(body.memory),
                                 target_datetime=body.when.replace(tzinfo=None) if body.when else None,
                                 max_travel_minutes=body.max_travel_minutes, use_memory=body.use_memory,
                                 progress=progress, synthesize=False, search_pages=3 if body.deep else 1)
            RECENT_RUNS[state["run_id"]] = state
            while len(RECENT_RUNS) > MAX_RECENT_RUNS:
                RECENT_RUNS.popitem(last=False)
            events.put({"type": "result", "result": public_result(state), "searches_left": searches_left})
        except Exception as exc:
            events.put({"type": "error", "message": f"Something went wrong ({type(exc).__name__}). Try again?"})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (event := events.get()) is not None:
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.post("/api/writeup")
def post_writeup(body: WriteupBody):
    state = RECENT_RUNS.get(body.run_id)
    if state is None:
        raise HTTPException(404, "That search has expired. Run it again to get the write-up.")
    synthesize_results(state)
    return {"write_up": clean(state.get("final_response")), "runtime_seconds": state.get("runtime_seconds")}


@app.post("/api/feedback")
def post_feedback(body: FeedbackBody):
    memory = clean_memory(body.memory)
    picks = [{k: p.get(k) for k in ["place_name", "source_activity", "final_rank"]} for p in body.picks[:10]]
    response = feedback_agent(body.text.strip(), picks)
    if response["status"] != "success":
        raise HTTPException(502, "Couldn't read that feedback. Try rephrasing it?")
    updated = update_memory_from_feedback(memory, response["data"])
    updated.pop("feedback_history", None)  # the browser doesn't need the log
    return {"memory": clean(updated), "summary": response["data"].get("feedback_summary")}


# The website itself (for local runs and single-server hosting).
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
