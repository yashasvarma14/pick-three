"""
Shared building blocks: the standard agent response, shared state,
and small text helpers used by several agents.
"""

import re
import uuid
from datetime import datetime


def empty_usage():
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost": 0.0}


def make_agent_response(
    agent_name,
    status="success",
    data=None,
    confidence=None,
    sources=None,
    warnings=None,
    error=None,
    usage=None,
):
    """Standard response returned by every specialist agent."""
    return {
        "agent": agent_name,
        "status": status,
        "data": data if data is not None else {},
        "confidence": confidence,
        "sources": sources if sources is not None else [],
        "warnings": warnings if warnings is not None else [],
        "error": error,
        "usage": usage if usage is not None else empty_usage(),
    }


def failed(agent_name, exc):
    return make_agent_response(
        agent_name,
        status="failed",
        error=f"{type(exc).__name__}: {exc}",
        confidence=0.0,
    )


def skipped(agent_name, reason):
    return make_agent_response(agent_name, status="skipped", warnings=[reason], confidence=0.0)


def create_initial_state(user_prompt=None, user_id=None, city=None):
    """Shared state for one recommendation run."""
    return {
        "run_id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "user_prompt": user_prompt,
        "city": city,
        "user_profile": {},
        "memory_context": "",
        "preferences": {},
        "candidate_activities": [],
        "candidate_places": [],
        "weather": {},
        "events": [],
        "mobility": [],
        "review_evidence": {},
        "ranked_results": [],
        "final_recommendations": [],
        "agents_requested": [],
        "agents_called": [],
        "agents_skipped": [],
        "agent_results": {},
        "human_checkpoint_used": False,
        "clarification_question": None,
        "orchestrator_plan": {},
        "adaptation_log": [],
        "finish_reason": None,
        "final_response": None,
        "confidence": {},
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "llm_cost": 0.0,
            "api_calls": {"google_places": 0, "google_routes": 0, "ticketmaster": 0, "open_meteo": 0},
        },
        "runtime_seconds": None,
        "warnings": [],
        "errors": [],
    }


def contains_keyword(text, keyword):
    """
    Whole-word match. Plain substring checks made "bar" match "barbecue"
    and "run" match "brunch" in the notebook version.
    """
    return re.search(rf"\b{re.escape(keyword)}\b", str(text).lower()) is not None


# ---------------------------------------------------------------------------
# Short-lived cache shared by every visitor of one app server.
# ---------------------------------------------------------------------------

import copy
import functools
import threading
import time
from concurrent.futures import Future


_ALL_CACHES = []


def clear_all_caches():
    for store in _ALL_CACHES:
        store.clear()


def ttl_cache(seconds, maxsize=512):
    """
    Remember a function's result for `seconds`. Repeat searches (same venue
    website, same Places query, same forecast) come back instantly.

    Two threads asking for the same thing at once share one request
    instead of both making it. Errors are never cached. Callers get a
    copy, so changing a result can't corrupt the cache.
    """
    def decorator(func):
        store = {}
        lock = threading.Lock()
        _ALL_CACHES.append(store)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            with lock:
                hit = store.get(key)
                if hit and hit[0] > now:
                    future, owner = hit[1], False
                else:
                    future, owner = Future(), True
                    store[key] = (now + seconds, future)
                    if len(store) > maxsize:
                        for old in sorted(store, key=lambda k: store[k][0])[: len(store) - maxsize]:
                            if store[old][1].done():
                                store.pop(old, None)
            if owner:
                try:
                    future.set_result(func(*args, **kwargs))
                except BaseException as exc:
                    with lock:
                        if store.get(key, (None, None))[1] is future:
                            store.pop(key, None)
                    future.set_exception(exc)
            return copy.deepcopy(future.result())

        wrapper.cache_clear = store.clear
        return wrapper

    return decorator
