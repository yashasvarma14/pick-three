"""
Planner / Orchestrator: decides which specialist agents a request needs.
The LLM proposes a plan; code enforces dependencies, order and limits.
"""

import json
import time

from .base import failed, make_agent_response
from .llm import call_llm_json

ORCHESTRATOR_AGENT_REGISTRY = {
    "memory": {"purpose": "Retrieve persistent user preferences and prior feedback.", "requires": [], "optional": True},
    "preference": {"purpose": "Interpret the request into structured intent.", "requires": [], "optional": False},
    "discovery": {"purpose": "Retrieve candidate activity concepts from the activity catalog.", "requires": ["preference"], "optional": False},
    "weather": {"purpose": "Evaluate whether weather changes the suitability of candidate activities.", "requires": ["discovery"], "optional": True},
    "local_places": {"purpose": "Find real venues associated with candidate activities.", "requires": ["discovery"], "optional": False},
    "events": {"purpose": "Retrieve scheduled concerts, sports, performances, festivals, and other live events.", "requires": ["preference"], "optional": True},
    "mobility": {"purpose": "Evaluate distance, travel time, and travel feasibility.", "requires": ["local_places"], "optional": True},
    "venue_rag": {"purpose": "Retrieve unstructured evidence from official venue websites.", "requires": ["local_places"], "optional": True},
    "matching": {"purpose": "Integrate activity, venue, mobility, and evidence signals into final rankings.", "requires": ["discovery", "local_places"], "optional": False},
}

CANONICAL_ORDER = ["memory", "preference", "discovery", "weather", "local_places",
                   "events", "mobility", "venue_rag", "matching"]

SYSTEM_PROMPT = """
You are the Planner / Orchestrator for an agentic activity recommendation system.

You do NOT recommend activities yourself. Decide which specialist agents are
needed, their order, and when the system has enough information to finish.

GENERAL RULES:
1. Preference and Discovery are normally required.
2. Memory should be used when personalization is enabled and memory exists.
3. Weather should run when outdoor suitability matters, when the user refers
   to tonight / today / tomorrow / the weekend, or when discovered activities
   could reasonably be affected by weather.
4. Local Places should run when the user wants real places or things to do in a city.
5. Events should run when the user asks about concerts, sports, performances,
   festivals, shows, live events, or broadly what is happening at a time.
6. Mobility should run when an origin is available and travel time matters
   or venues should be checked for feasibility.
7. Venue RAG should run when real venues exist and website evidence would
   improve the qualitative fit judgment.
8. Matching should run once there are candidate activities and venues to rank.
9. Do not request an agent only because it exists.
10. Respect dependencies.
11. Request a human clarification checkpoint only if proceeding would produce
    unreliable recommendations. Never ask about timing (the app supplies a
    date and time, and a time named in the request is handled automatically)
    or about location (the app supplies it).
12. Prefer useful evidence over unnecessary calls, within the call limit.

Every item in agents_to_run, agents_to_skip and execution_order MUST be a
single plain string. Never return nested lists.

Return ONLY valid JSON:
{
    "goal_summary": "",
    "agents_to_run": [],
    "agents_to_skip": [],
    "execution_order": [],
    "human_checkpoint_needed": false,
    "clarification_question": null,
    "planning_reasoning_summary": [],
    "finish_conditions": [],
    "max_agent_calls": 0
}

planning_reasoning_summary: short, auditable decision statements only.
"""


def _flatten(values):
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    out = []
    for v in values if isinstance(values, list) else []:
        out.extend(_flatten(v) if isinstance(v, list) else [v] if isinstance(v, str) else [])
    return out


def validate_plan(plan, use_memory=True, max_agent_calls=9):
    """Deterministic clean-up of whatever the LLM proposed."""
    valid = set(ORCHESTRATOR_AGENT_REGISTRY)
    agents = [a for a in dict.fromkeys(_flatten(plan.get("agents_to_run"))) if a in valid]

    for required in ["preference", "discovery"]:
        if required not in agents:
            agents.append(required)

    if use_memory and "memory" not in agents:
        agents.insert(0, "memory")
    if not use_memory:
        agents = [a for a in agents if a != "memory"]

    # Add dependencies until nothing changes (handles chains of dependencies).
    changed = True
    while changed:
        changed = False
        for agent in list(agents):
            for dep in ORCHESTRATOR_AGENT_REGISTRY[agent]["requires"]:
                if dep not in agents:
                    agents.append(dep)
                    changed = True

    order = [a for a in CANONICAL_ORDER if a in agents][:max_agent_calls]
    plan.update({
        "agents_to_run": order,
        "execution_order": order,
        "agents_to_skip": [a for a in CANONICAL_ORDER if a not in order],
        "max_agent_calls": max_agent_calls,
    })
    return plan


def orchestrator_planner(user_prompt, memory_context="", city=None, origin=None,
                         target_datetime=None, use_memory=True, max_agent_calls=9):
    start = time.time()
    message = f"""
CURRENT USER REQUEST:
{user_prompt}

CITY PROVIDED: {city}
ORIGIN PROVIDED: {origin}
TARGET DATETIME PROVIDED: {target_datetime}
PERSISTENT MEMORY ENABLED: {use_memory}

AVAILABLE MEMORY CONTEXT:
{memory_context or "No stored memory context available."}

MAXIMUM AGENT CALLS: {max_agent_calls}

AVAILABLE SPECIALIST AGENTS:
{json.dumps(ORCHESTRATOR_AGENT_REGISTRY, indent=2)}
"""
    try:
        plan, usage = call_llm_json(SYSTEM_PROMPT, message)
        plan = validate_plan(plan, use_memory=use_memory, max_agent_calls=max_agent_calls)
        return make_agent_response("orchestrator", data={"plan": plan, "runtime_seconds": time.time() - start}, usage=usage)
    except Exception as exc:
        return failed("orchestrator", exc)
