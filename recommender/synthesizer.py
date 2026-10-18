"""
Grounded Recommendation Synthesizer: explains the already-ranked results.
It cannot reorder or add venues; code checks that it didn't.
"""

import json
import time

from .base import failed, make_agent_response, skipped
from .llm import call_llm_json

EVIDENCE_FIELDS = [
    "final_rank", "place_name", "source_activity", "rating", "review_count",
    "activity_context_score", "place_quality_score", "place_activity_relevance_score",
    "distance_miles", "traffic_minutes", "mobility_score", "evidence_fit_score",
    "rag_evidence_summary", "rag_strengths", "rag_concerns", "final_recommendation_score",
    "primary_type", "open_at_planned_time", "straight_line_miles",
]

SYSTEM_PROMPT = """
You are the final Synthesizer for an agentic activity recommendation system.

The ranking has ALREADY been decided. You must NOT change the ranking,
introduce new venues, or invent facts, prices, hours, travel times, weather,
or events. Only explain the supplied recommendations using the supplied
workflow context.

When a signal is unavailable, do not mention it (no distance talk if
distance_miles is null, no website claims without RAG evidence, no weather
talk if weather was skipped). Only mention an event if it is clearly related.

Explain in plain, friendly language. Do not list internal model scores.

Return ONLY valid JSON:
{
    "overall_summary": "",
    "recommendations": [
        {
            "final_rank": 1,
            "place_name": "",
            "source_activity": "",
            "why_it_fits": "",
            "supporting_evidence": [],
            "important_caveat": null
        }
    ],
    "personalization_summary": "",
    "evidence_limitations": []
}

Rules: preserve the exact ranking and place names; exactly one explanation
per recommendation; supporting_evidence is short factual points;
important_caveat is null when there is no real caveat.
"""


def recommendation_synthesizer(state, max_recommendations=3):
    start = time.time()
    recs = (state.get("final_recommendations") or [])[:max_recommendations]
    if not recs:
        return skipped("synthesizer", "No final recommendations to explain.")

    weather = state.get("weather") or {}
    context = {
        "user_request": state.get("user_prompt"),
        "structured_preferences": {k: v for k, v in (state.get("preferences") or {}).items() if k != "runtime_seconds"},
        "memory_context": state.get("memory_context", ""),
        "weather": {k: weather.get(k) for k in ["condition", "temperature_f", "precipitation_probability",
                                                 "wind_speed_mph", "weather_suitability", "weather_reasons"]} if weather else {},
        "events": [{k: e.get(k) for k in ["event_name", "local_date", "local_time", "venue_name", "segment", "genre"]}
                   for e in (state.get("events") or [])[:10]],
        "adaptation_log": state.get("adaptation_log", []),
        "recommendations": [{k: r.get(k) for k in EVIDENCE_FIELDS} for r in recs],
    }

    try:
        parsed, usage = call_llm_json(SYSTEM_PROMPT, "WORKFLOW CONTEXT:\n" + json.dumps(context, indent=2, default=str))

        expected = [str(r.get("place_name")) for r in recs]
        returned = [str(r.get("place_name")) for r in parsed.get("recommendations", [])]
        if returned != expected:
            return make_agent_response("synthesizer", status="failed", confidence=0.0,
                                       error="Synthesizer changed the ranked venue order or introduced a venue.", usage=usage)

        return make_agent_response("synthesizer", data={"synthesis": parsed, "runtime_seconds": time.time() - start},
                                   confidence=1.0, usage=usage)
    except Exception as exc:
        return failed("synthesizer", exc)
