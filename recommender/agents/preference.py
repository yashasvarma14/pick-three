"""
Step 1 - Preference agent (LLM, runs at the same time as the planner)
=====================================================================
Turns the free-text request into structured fields: group size, budget,
likes / dislikes, a time if one is mentioned ("this afternoon"), and the
Google Maps searches to run ("quiet wine bar", "day spa", ...).
"""

import time
from datetime import datetime, timedelta

from ..core import llm
from ..core.state import failed, make_agent_response



BUDGET_LEVELS = ["free", "low", "medium", "high"]


def _system_prompt(valid_types):
    return f"""
You are the Preference Agent for an activity recommendation system.

Interpret the user's natural-language request and convert it into
structured preferences.

Available activity categories:
{valid_types}

Return ONLY a valid JSON object with exactly these fields:

{{
    "city": null,
    "group_size": null,
    "budget_level": null,
    "preferred_types": [],
    "positive_preferences": [],
    "negative_preferences": [],
    "hard_constraints": [],
    "soft_preferences": [],
    "venue_search_terms": [],
    "search_phrases": [],
    "requested_datetime": null,
    "needs_clarification": false,
    "clarification_question": null
}}

Rules:
1. city: city name only if explicitly provided, otherwise null.
2. group_size: integer if clearly stated, otherwise null.
3. budget_level: one of "free", "low", "medium", "high", or null.
4. preferred_types: ONLY values from {valid_types}.
5. positive_preferences: characteristics the user wants.
6. negative_preferences: characteristics the user wants to avoid, even when
   phrased indirectly ("rather than just sitting around" -> "passive activities";
   "not too expensive" -> "high cost"; "don't want anything crowded" ->
   "crowded environments"). Only put them in hard_constraints when the wording
   clearly makes them mandatory.
7. hard_constraints: requirements stated as mandatory ("must be free",
   "within 20 minutes", "nothing outdoors").
8. soft_preferences: desired but not mandatory ("prefer something cheap").
9. Do not invent information.
10. Set needs_clarification=true only when missing information would
    prevent a meaningful recommendation. A city supplied separately by the
    application counts as provided.
11. If clarification is required, ask one concise question.
12. venue_search_terms: when the user names a SPECIFIC kind of place, food,
    or activity ("burgers", "sushi", "a rooftop bar", "bowling", "pottery
    class"), give 1-2 short Google Maps search phrases for it, keeping
    their qualifiers ("cheap burger restaurant", "rooftop bar"). Do not
    include the city. Leave it empty for open-ended requests ("something
    fun", "relaxing and quiet").
13. search_phrases: ALWAYS give 3-5 short, different Google Maps searches
    that would find good places for this request, as a local would type
    them, without the city ("quiet wine bar", "botanical garden",
    "day spa", "bookstore cafe"). If the request is specific, every phrase
    stays on that thing ("burger joint", "smash burger", "cheap burgers").
14. requested_datetime: ONLY when the request itself names a time or day,
    give it as "YYYY-MM-DD HH:MM" in local time, using the current date
    supplied below. Examples: "this afternoon" -> today 14:00 (tomorrow if
    already past); "tonight" -> today 19:00; "tomorrow morning" -> tomorrow
    10:00; "this weekend" -> the coming Saturday 14:00; "Friday night" ->
    the coming Friday 20:00. A part of the day WITHOUT a day also counts and
    means the next one: "morning" -> 10:00, "afternoon" -> 14:00,
    "evening" -> 18:00, "tonight"/"night" -> 19:00 ("a rainy morning" ->
    tomorrow 10:00 if this morning has passed). Explicit clock times count
    too ("at 4am" -> the next 04:00). Otherwise null. Never ask a
    clarification question about timing.
"""


def _parse_requested_time(value, now):
    """Keep the model's requested time only if it parses and is in the next 7 days."""
    if not value or now is None:
        return None
    try:
        when = datetime.strptime(str(value).strip()[:16], "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    if now - timedelta(hours=1) <= when <= now + timedelta(days=7):
        return when
    return None


def preference_agent(user_query, valid_types, app_city=None, now=None, app_time=None):
    """
    now       current local datetime where the user is (lets the model
              resolve "this afternoon" / "this weekend")
    app_time  the date/time chosen in the app, or None for right now
    """
    start = time.time()
    try:
        message = user_query
        if app_city:
            message += f"\n\n(The user selected this city in the app: {app_city})"
        if now is not None:
            chosen = app_time.strftime("%A %Y-%m-%d %H:%M") if app_time else "right now"
            message += (f"\n(Current local date and time: {now:%A %Y-%m-%d %H:%M}. "
                        f"Time set in the app: {chosen}.)")

        parsed, usage = llm.call_llm_json(_system_prompt(valid_types), message)

        # Light validation so downstream code can trust the types.
        if parsed.get("budget_level") not in BUDGET_LEVELS:
            parsed["budget_level"] = None
        parsed["preferred_types"] = [t for t in (parsed.get("preferred_types") or []) if t in valid_types]
        terms = parsed.get("venue_search_terms") or []
        if isinstance(terms, str):
            terms = [terms]
        parsed["venue_search_terms"] = [str(t).strip() for t in terms if str(t).strip()][:2]
        phrases = parsed.get("search_phrases") or []
        if isinstance(phrases, str):
            phrases = [phrases]
        parsed["search_phrases"] = list(dict.fromkeys(str(t).strip().lower() for t in phrases if str(t).strip()))[:5]
        try:
            parsed["group_size"] = int(parsed["group_size"]) if parsed.get("group_size") else None
        except (TypeError, ValueError):
            parsed["group_size"] = None

        parsed["requested_datetime"] = _parse_requested_time(parsed.get("requested_datetime"), now)
        parsed["runtime_seconds"] = time.time() - start
        return make_agent_response("preference", data=parsed, usage=usage)
    except Exception as exc:
        return failed("preference", exc)
