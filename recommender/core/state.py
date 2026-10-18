"""
The shared "state" dictionary that one run of the pipeline fills in, and
the standard response every agent returns.

Every agent returns the same shape:

    {"agent": "weather", "status": "success" | "skipped" | "failed",
     "data": {...}, "warnings": [...], "error": None, "usage": {...}}

so the pipeline can treat all of them the same way (see record()).
"""

import time
import uuid
from datetime import datetime



# ---------------------------------------------------------------------------
# Agent responses
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------


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
            "api_calls": {"google_places": 0, "google_place_photos": 0, "google_routes": 0, "ticketmaster": 0,
                          "open_meteo": 0},
        },
        "runtime_seconds": None,
        "warnings": [],
        "errors": [],
    }


def add_once(lst, item):
    """Append item unless it's already in the list."""
    if item and item not in lst:
        lst.append(item)


def add_usage(state, response):
    """Add an agent's LLM tokens and cost to the run's totals."""
    usage = (response or {}).get("usage") or {}
    for key in ["input_tokens", "output_tokens", "total_tokens"]:
        state["usage"][key] += usage.get(key, 0) or 0
    state["usage"]["llm_cost"] += usage.get("cost", 0.0) or 0.0


def record(state, name, response):
    """
    Store an agent's response in the run state and update the ran/skipped
    lists and usage totals. Returns True if the agent succeeded.
    """
    state["agent_results"][name] = response
    if response["status"] == "success":
        add_once(state["agents_called"], name)
        add_usage(state, response)
        # Weather and Events always run in the background, even when the
        # planner thought they weren't needed, so they can't be "skipped" too.
        if name in state["agents_skipped"]:
            state["agents_skipped"].remove(name)
    elif response["status"] == "skipped":
        add_once(state["agents_skipped"], name)
        state["warnings"].extend(response.get("warnings", []))
    else:
        if response.get("error"):
            state["warnings"].append(f"{name}: {response['error']}")
    return response["status"] == "success"


def timed(timings, name, func, *args, **kwargs):
    """Run func(*args, **kwargs) and save how many seconds it took under timings[name]."""
    t0 = time.time()
    try:
        return func(*args, **kwargs)
    finally:
        timings[name] = round(time.time() - t0, 2)


def records(df):
    """DataFrame -> list of dicts with missing values as None (never NaN)."""
    return [{k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
            for row in df.to_dict("records")]
